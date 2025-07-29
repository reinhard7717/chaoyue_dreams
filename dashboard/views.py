# dashboard/views.py
import asyncio
import json
from amqp import NotFound
from asgiref.sync import async_to_sync
from django.db.models import Max, F, Q, OuterRef, Subquery
from datetime import date, datetime
from collections import OrderedDict # 导入 OrderedDict
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.forms import ValidationError
from dao_manager.tushare_daos.strategies_dao import StrategiesDAO
from dao_manager.tushare_daos.user_dao import UserDAO
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from rest_framework import generics, viewsets
from rest_framework.permissions import IsAuthenticated

from django.core.serializers.json import DjangoJSONEncoder
from stock_models.stock_analytics import FavoriteStockTracker, TrendFollowStrategySignalLog, TrendFollowStrategyState
from stock_models.stock_basic import StockInfo
from utils.cache_manager import CacheManager
from utils.cash_key import IntradayEngineCashKey
from users.models import FavoriteStock
from utils.websockets import send_update_to_user_sync
from .serializers import StockInfoSerializer, FavoriteStockSerializer
from itertools import chain
import logging # 导入 logging

logger = logging.getLogger('dashboard') # 获取 logger 实例
target_queue = 'dashboard'

# 内部辅助函数，用于处理所有策略列表页的通用逻辑
def _render_strategy_list_page(request, base_queryset, page_title, template_name):
    """
    一个通用的辅助函数，用于渲染策略列表页面。
    """
    print(f"--- [View] 开始渲染页面: {page_title} ---")
    
    ordered_queryset = base_queryset
    print("--- [View] DAO已提供完整数据，跳过视图层注解 ---")

    paginator = Paginator(ordered_queryset, 25)  # 每页显示25条
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
        'total_count': paginator.count,
        'page_title': page_title,
    }
    return render(request, template_name, context)

# --- 页面视图 ---
@login_required
def dashboard_view(request):
    user_dao = UserDAO()
    """渲染主控台页面"""
    get_favorites_async = user_dao.get_user_favorites
    user_favorites = async_to_sync(get_favorites_async)(request.user.id)
    initial_favorites_data = [
        {
            'id': fav.id,
            'code': fav.stock.stock_code,
            'name': fav.stock.stock_name,
            "current_price": None,
            "high_price": None,
            "low_price": None,
            "open_price": None,
            "prev_close_price": None,
            "trade_time": None,
            'volume': None,
            'change_percent': None,
            'signal': None,
        } 
        for fav in user_favorites if fav.stock
    ]
    initial_favorites_json_string = json.dumps(initial_favorites_data, cls=DjangoJSONEncoder)
    context = {
        'initial_favorites_json': initial_favorites_json_string,
        'user_favorites': user_favorites, # 新增此行
    }
    return render(request, 'dashboard/home.html', context)

def get_playbook_priority(playbook_name):
    """
    【V2.0 修正版】根据剧本名称中的关键字为其分配优先级。
    此函数的判断逻辑现在与模板中的样式逻辑完全一致，确保排序正确。
    返回值越小，优先级越高。
    """
    # 安全检查，防止 playbook_name 不是字符串类型导致错误
    if not isinstance(playbook_name, str):
        return 99 # 返回一个很大的值，使其排在最后

    playbook_name_upper = playbook_name.upper()
    # 优先级 0: 火箭信号 (与模板逻辑一致)
    if 'ROCKET' in playbook_name_upper or '火箭' in playbook_name:
        return 0
    # 优先级 1: 王牌信号 (修正为与模板逻辑一致，检查'王牌')
    if 'BREAKOUT_TRIGGER_SCORE' in playbook_name_upper or '王牌' in playbook_name:
        return 1
    # 优先级 2: 专家信号 (修正为与模板逻辑一致，检查'专家')
    if '专家' in playbook_name:
        return 2
    # 优先级 4: 加分项 (逻辑保持不变)
    if '【加分】' in playbook_name or '资金流入' in playbook_name:
        return 4
    # 优先级 5: 基础形态 (逻辑保持不变)
    if 'MA20' in playbook_name_upper or '均线' in playbook_name:
        return 5
    # 默认优先级
    return 3


@login_required
def trend_following_list(request):
    """
    【V119.1 格式修正版】
    - 核心重构: 采用非相关子查询(GROUP BY + IN)替代原有的相关子查询(OuterRef/Subquery)，彻底解决数据库性能瓶颈。
    - 核心重构: 将数据过滤、合并、排序等逻辑移至Python内存中处理，极大减轻数据库压力。
    - 核心重构: 优化了“剧本筛选器”的数据获取方式，不再对数据库进行全量查询，而是复用内存中已有的数据。
    - 核心修正: 增加了对 triggered_playbooks 字段的解析逻辑，将数据库中的逗号分隔字符串正确转换为Python列表，解决剧本显示错误的问题。
    - 收益: 页面加载速度从分钟级提升至秒级，实现了质的飞跃，同时保证业务逻辑完全不变。
    """
    # print("--- [View] 开始渲染策略状态监控中心 (trend_following_list) V119.1 格式修正版 ---")
    # 步骤1: 使用更高效的 GROUP BY + IN 查询，一次性获取所有最新的日线级别“买入”信号记录
    # print("--- [View] 步骤1: 开始查询最新买入信号ID...")
    latest_buy_log_ids = TrendFollowStrategySignalLog.objects.filter(
        entry_signal=True,
        timeframe='D'
    ).values('stock_id').annotate(
        latest_id=Max('id')
    ).values_list('latest_id', flat=True)
    latest_buy_logs_qs = TrendFollowStrategySignalLog.objects.filter(
        id__in=list(latest_buy_log_ids)
    ).select_related('stock')
    # print(f"--- [View] 步骤1完成: 获取到 {latest_buy_logs_qs.count()} 条最新买入信号")
    # 步骤2: 一次性获取所有相关股票的最新“卖出”时间
    stock_ids = latest_buy_logs_qs.values_list('stock_id', flat=True)
    # print("--- [View] 步骤2: 开始查询最新卖出时间...")
    latest_sell_times_qs = TrendFollowStrategySignalLog.objects.filter(
        stock_id__in=stock_ids,
        exit_signal_code__gt=0,
        timeframe='D'
    ).values('stock_id').annotate(
        latest_sell_time=Max('trade_time')
    )
    sell_time_map = {item['stock_id']: item['latest_sell_time'] for item in latest_sell_times_qs}
    # print(f"--- [View] 步骤2完成: 获取到 {len(sell_time_map)} 个股票的最新卖出时间")
    # 步骤3: 在Python内存中进行数据合并和持仓过滤
    # print("--- [View] 步骤3: 开始在内存中合并数据并筛选持仓股...")
    held_logs = []
    for buy_log in latest_buy_logs_qs.iterator():
        latest_sell_time = sell_time_map.get(buy_log.stock_id)
        # 判断持仓条件：没有卖出记录，或者最后卖出时间早于最后买入时间
        if latest_sell_time is None or latest_sell_time < buy_log.trade_time:
            # [新增] 动态地将 latest_sell_time 附加到对象上，方便模板使用
            buy_log.latest_sell_time = latest_sell_time
            # [修改] 核心修正：将数据库中的剧本字符串转换为Python列表
            # 数据库中的 triggered_playbooks 是一个逗号分隔的字符串，必须在此处解析
            # 否则后续操作会错误地迭代字符串中的每个字符
            if buy_log.triggered_playbooks and isinstance(buy_log.triggered_playbooks, str):
                # 按逗号分割，并去除每个剧本名称前后的空白字符，过滤掉空字符串
                buy_log.triggered_playbooks = [p.strip() for p in buy_log.triggered_playbooks.split(',') if p.strip()]
            else:
                # 如果字段为空或不是字符串，则确保它是一个空列表以保持类型一致
                buy_log.triggered_playbooks = []
            held_logs.append(buy_log)
    # print(f"--- [View] 步骤3完成: 筛选出 {len(held_logs)} 只持仓股")
    # 步骤4: 高效地从内存数据中提取所有可用剧本，用于筛选器
    # 此处代码无需修改，因为它现在接收到的是正确的列表嵌套列表
    # print("--- [View] 步骤4: 开始从内存中聚合剧本列表...")
    all_playbook_lists = [log.triggered_playbooks for log in held_logs if log.triggered_playbooks]
    unique_playbooks = sorted(
        list(set(chain.from_iterable(all_playbook_lists))),
        key=get_playbook_priority
    )
    # print(f"--- [View] 步骤4完成: 找到 {len(unique_playbooks)} 个唯一剧本")
    # 步骤5: 在内存中根据请求参数进行剧本筛选
    # 此处代码无需修改，因为它现在可以正确地在列表中检查成员
    # print("--- [View] 步骤5: 开始根据用户选择筛选剧本...")
    selected_playbooks = request.GET.getlist('playbooks')
    final_logs = held_logs # 默认是全部持仓记录
    if selected_playbooks:
        final_logs = [
            log for log in held_logs
            if all(p in log.triggered_playbooks for p in selected_playbooks)
        ]
    # print(f"--- [View] 步骤5完成: 筛选后剩余 {len(final_logs)} 条记录")
    # 步骤6: 在内存中对最终结果进行排序
    # print("--- [View] 步骤6: 开始排序...")
    final_logs.sort(key=lambda log: (log.trade_time, log.entry_score or 0), reverse=True)
    # print("--- [View] 步骤6完成: 排序完成")
    # 步骤7: 使用Paginator对Python列表进行分页
    # print("--- [View] 步骤7: 开始分页...")
    paginator = Paginator(final_logs, 25)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    # print("--- [View] 步骤7完成: 分页完成")
    # 步骤8: 格式化当前页的数据用于模板渲染
    # 此处代码无需修改，因为它现在接收到的是正确的列表
    # print("--- [View] 步骤8: 开始格式化最终数据...")
    final_list_for_template = []
    for log in page_obj.object_list:
        final_list_for_template.append({
            'log_id': log.id,
            'stock': log.stock,
            'latest_trade_time': log.trade_time,
            'latest_score': log.entry_score,
            'last_buy_time': log.trade_time,
            'last_sell_time': log.latest_sell_time, # 使用我们之前附加的属性
            'active_playbooks': sorted(log.triggered_playbooks, key=get_playbook_priority),
            'strategy_names': [log.strategy_name],
            'stable_platform_price': log.stable_platform_price,
        })
    # print("--- [View] 步骤8完成: 格式化完成，准备渲染模板")
    # 步骤9: 准备上下文
    context = {
        'page_title': '策略状态监控中心',
        'items_for_display': final_list_for_template,
        'page_obj': page_obj,
        'total_count': paginator.count,
        'all_playbooks': unique_playbooks,
        'selected_playbooks': selected_playbooks,
    }
    return render(request, 'dashboard/trend_following_list.html', context)

@login_required
def fav_trend_following_list(request):
    """
    【V3.0 状态驱动版】
    - 核心升级: 直接从 FavoriteStockTracker 模型中读取预计算好的持仓状态，
                查询逻辑极度简化，性能大幅提升。
    - 新增功能: 增加了状态筛选功能，可以查看“持仓中”或“已平仓”的记录。
    """
    # --- 步骤1: 获取用户的所有追踪器 ---
    base_queryset = FavoriteStockTracker.objects.filter(
        user=request.user
    ).select_related(
        'stock', 
        'entry_log', 
        'latest_log',
        'exit_log'
    )

    # --- 步骤2: 根据前端请求进行状态筛选 ---
    status_filter = request.GET.get('status', 'holding') # 默认只显示持仓中的
    if status_filter == 'holding':
        queryset = base_queryset.filter(status='HOLDING')
        page_title = '自选股持仓监控'
    elif status_filter == 'sold':
        queryset = base_queryset.filter(status='SOLD')
        page_title = '自选股历史平仓'
    else:
        queryset = base_queryset
        page_title = '全部自选追踪'

    # --- 步骤3: 排序 ---
    # 持仓中的按最新更新时间倒序，已平仓的按平仓时间倒序
    if status_filter == 'sold':
        ordered_queryset = queryset.order_by('-exit_date')
    else:
        ordered_queryset = queryset.order_by('-latest_date')

    # --- 步骤4: 分页 ---
    paginator = Paginator(ordered_queryset, 25) # 每页显示25条
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # --- 步骤5: 准备上下文并渲染 ---
    context = {
        'page_title': page_title,
        'page_obj': page_obj,
        'total_count': paginator.count,
        'status_filter': status_filter, # 将当前筛选状态传给模板，用于高亮显示按钮
    }
    return render(request, 'dashboard/fav_trend_following_list.html', context)

@login_required
def realtime_engine_view(request):
    """
    【V1.0】渲染盘中引擎实时监控页面。
    - 核心职责:
      1. 从Redis中获取当前用户今日已产生的所有盘中信号作为初始数据。
      2. 将初始数据传递给模板进行渲染。
      3. 页面后续的更新将通过WebSocket实时推送。
    """
    user_id = request.user.id
    today_str = date.today().strftime('%Y-%m-%d')
    
    cache_manager = CacheManager()
    cache_key_builder = IntradayEngineCashKey()
    
    # 1. 生成当前用户今日的信号缓存键
    signals_key = cache_key_builder.user_signals_key(user_id, today_str)
    
    # 2. 从Redis的List中获取所有信号
    # 使用 async_to_sync 来在同步视图中调用异步缓存方法
    async def get_initial_signals():
        await cache_manager.initialize()
        # lrange(key, 0, -1) 获取列表中的所有元素
        raw_signals = await cache_manager.redis_client.lrange(signals_key, 0, -1)
        # Redis返回的是bytes，需要解码并用json加载
        return [json.loads(s.decode()) for s in raw_signals]

    initial_signals = async_to_sync(get_initial_signals)()
    
    # 3. 准备传递给模板的上下文
    context = {
        'page_title': '盘中引擎实时监控',
        'initial_signals': initial_signals,
        'total_count': len(initial_signals),
    }
    
    return render(request, 'dashboard/realtime_engine.html', context)

# --- DRF API 视图 ---

class StockSearchView(generics.ListAPIView):
    serializer_class = StockInfoSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        query = self.request.query_params.get('q', None)
        if query:
            return StockInfo.objects.filter(
                Q(stock_code__icontains=query) | Q(stock_name__icontains=query)
            )[:10]
        return StockInfo.objects.none()

class FavoriteStockViewSet(viewsets.ModelViewSet):
    serializer_class = FavoriteStockSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return FavoriteStock.objects.filter(user=self.request.user).select_related('stock')

    def perform_create(self, serializer):
        """
        【V2.1 完整实现版】
        - 核心逻辑: “加入自选”即“模拟建仓”。
        - 流程:
          1. 验证前端是否传递了有效的建仓信号ID (signal_log_id)。
          2. 以此信号为依据，创建 FavoriteStock 记录。
          3. 以此信号为依据，创建或更新一个状态为“持仓中”的 FavoriteStockTracker 记录。
          4. 发送 WebSocket 通知，实时更新前端UI。
          5. 触发 Celery 后台任务，为新持仓股预热数据。
        """
        user = self.request.user
        
        # --- 步骤1: 验证建仓信号 ---
        signal_log_id = self.request.data.get('signal_log_id')
        if not signal_log_id:
            logger.warning(f"用户 {user.username} 尝试添加自选但未提供 signal_log_id。")
            raise ValidationError({'detail': '必须提供一个有效的建仓信号ID (signal_log_id)。'})

        try:
            # 确保信号存在，且确实是一个买入信号
            entry_log = TrendFollowStrategySignalLog.objects.select_related('stock').get(
                id=signal_log_id, 
                entry_signal=True
            )
        except TrendFollowStrategySignalLog.DoesNotExist:
            logger.error(f"用户 {user.username} 尝试使用无效的 signal_log_id: {signal_log_id} 添加自选。")
            raise NotFound('指定的建仓信号不存在或无效。')

        stock = entry_log.stock

        # --- 步骤2: 创建 FavoriteStock (如果尚不存在) ---
        # 使用 get_or_create 确保原子性，避免并发问题
        favorite, favorite_created = FavoriteStock.objects.get_or_create(
            user=user, 
            stock=stock
        )
        if favorite_created:
            logger.info(f"为用户 {user.username} 创建了新的 FavoriteStock 记录 for {stock.stock_code}。")
        
        # --- 步骤3: 创建或更新 FavoriteStockTracker ---
        # 使用 update_or_create 来处理重复建仓的场景（例如，用户移除了又用同一个信号加回来）
        tracker, tracker_created = FavoriteStockTracker.objects.update_or_create(
            user=user,
            stock=stock,
            entry_log=entry_log, # 使用 entry_log 作为唯一性约束的一部分
            defaults={
                'status': 'HOLDING',
                'entry_price': entry_log.close_price,
                'entry_date': entry_log.trade_time,
                'entry_score': entry_log.entry_score or 0.0,
                
                # 初始时，最新状态就是建仓时的状态
                'latest_log': entry_log,
                'latest_price': entry_log.close_price,
                'latest_date': entry_log.trade_time,
                'holding_health_score': entry_log.holding_health_score or 0.0,
                'score_change_vs_entry': 0.0,
                'profit_loss_pct': 0.0,

                # 清空可能的历史平仓信息，确保这是一次全新的持仓记录
                'exit_log': None,
                'exit_price': None,
                'exit_date': None,
            }
        )
        
        log_action = "创建了新的" if tracker_created else "更新了现有的"
        logger.info(f"用户 {user.username} {log_action} 持仓追踪器 for {stock.stock_code}，关联建仓信号ID: {signal_log_id}。")

        # --- 步骤4: 发送 WebSocket 通知，实时更新前端UI ---
        # 准备一个与 dashboard home 页面兼容的 payload
        websocket_payload = {
            'id': favorite.id,
            'code': stock.stock_code,
            'name': stock.stock_name,
            "current_price": None, # 这些字段将由前端的实时行情WebSocket填充
            "high_price": None,
            "low_price": None,
            "open_price": None,
            "prev_close_price": None,
            "trade_time": None,
            'volume': None,
            'change_percent': None,
            'signal': None,
        }
        
        # 使用 async_to_sync 包装器在同步代码中调用异步函数
        send_update_to_user_sync(
            user_id=user.id,
            sub_type='favorite_added_with_data',
            payload=websocket_payload
        )
        logger.info(f"已通过 WebSocket向用户 {user.username} 推送新自选股 {stock.stock_code} 的更新。")

        # --- 步骤5: 触发 Celery 后台任务，为新持仓股预热数据 ---
        # 这是一个耗时操作，适合异步执行，避免阻塞API响应
        try:
            # 动态导入任务，避免循环依赖
            from tasks.tushare.stock_tasks import fetch_data_for_new_favorite
            
            task_args = (
                user.id,
                stock.stock_code,
                favorite.id
            )
            
            fetch_data_for_new_favorite.apply_async(
                args=task_args,
                queue=target_queue, # 发送到指定的队列
            )
            logger.info(f"已将后台任务发送到队列 '{target_queue}' 为用户 {user.id} 的新自选股 {stock.stock_code} 获取数据。")
        except ImportError:
            logger.error("无法导入后台任务 'fetch_data_for_new_favorite'，跳过任务触发。请检查 Celery 配置和任务路径。")
        except Exception as task_error:
            logger.error(f"触发后台任务 fetch_data_for_new_favorite 时出错: {task_error}", exc_info=True)

    def perform_destroy(self, instance):
        user_id = instance.user.id
        instance.delete()
        updated_list = self._get_formatted_favorites(user_id)
        send_update_to_user_sync(
            user_id=user_id,
            sub_type='favorites_update',
            payload=updated_list
        )
    
    def _get_formatted_favorites(self, user_or_id):
        if isinstance(user_or_id, int):
            favorites = FavoriteStock.objects.filter(user_id=user_or_id).select_related('stock').order_by('added_at')
        else:
             favorites = FavoriteStock.objects.filter(user=user_or_id).select_related('stock').order_by('added_at')
        return [
            {
                'id': fav.id,
                'code': fav.stock.stock_code,
                'name': fav.stock.stock_name,
                "current_price": None,
                "high_price": None,
                "low_price": None,
                "open_price": None,
                "prev_close_price": None,
                "trade_time": None,
                'volume': None,
                'change_percent': None,
                'signal': None,
            } 
            for fav in favorites
        ]
