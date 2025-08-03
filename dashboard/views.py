# dashboard/views.py
import asyncio
import json
from django.utils import timezone
from datetime import timedelta
from asgiref.sync import async_to_sync
from django.db.models import Q, Prefetch
from rest_framework import viewsets, status
from rest_framework.response import Response
from django.core.paginator import Paginator
from dao_manager.tushare_daos.user_dao import UserDAO
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from rest_framework import generics, viewsets
from rest_framework.permissions import IsAuthenticated
from utils.cash_key import IntradayEngineCashKey
from django.core.serializers.json import DjangoJSONEncoder
from strategies.trend_following_strategy import TrendFollowStrategy
from utils.config_loader import load_strategy_config
from stock_models.stock_analytics import TradingSignal, SignalPlaybookDetail, FavoriteStockTracker, Playbook
from stock_models.stock_basic import StockInfo
from utils.cache_manager import CacheManager
from utils.cash_key import IntradayEngineCashKey
from users.models import FavoriteStock
from utils.websockets import send_update_to_user_sync
from .serializers import StockInfoSerializer, FavoriteStockSerializer
from utils.task_helpers import with_cache_manager_for_views
import logging # 导入 logging

logger = logging.getLogger('dashboard') # 获取 logger 实例
target_queue = 'dashboard'

# --- 页面视图 ---
@login_required
@with_cache_manager_for_views
def dashboard_view(request, cache_manager=None):
    """
    【已重构】渲染主控台页面。
    使用 @with_cache_manager_for_views 装饰器自动管理Redis连接。
    """
    # 【代码修改】直接使用由装饰器注入的 cache_manager 实例
    user_dao = UserDAO(cache_manager_instance=cache_manager)
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
    【V405.0 新模型适配版】
    - 核心重构: 完全适配 TradingSignal 和 SignalPlaybookDetail 新模型。
    - 业务逻辑简化: 只查询并显示最近3个交易日内有“买入”信号的股票记录。
    - 性能优化: 使用 prefetch_related 高效获取关联的剧本详情。
    - 数据结构调整: 适配新模型，从 SignalPlaybookDetail 获取剧本和分数。
    """
    print("--- [View] 开始渲染策略状态监控中心 (trend_following_list) V405.0 新模型版 ---")
    
    # --- 代码修改开始 ---
    # [修改原因] 适配新模型和新业务需求

    # 步骤1: 计算最近3天的起始日期
    three_days_ago = timezone.now().date() - timedelta(days=3)
    
    # 步骤2: 查询最近3天内所有日线级别的“买入”信号
    # 使用 prefetch_related 一次性加载所有相关的剧本详情，避免N+1查询
    latest_buy_signals = TradingSignal.objects.filter(
        signal_type=TradingSignal.SignalType.BUY,
        timeframe='D',
        trade_time__date__gte=three_days_ago
    ).select_related('stock').prefetch_related(
        Prefetch('playbook_details', queryset=SignalPlaybookDetail.objects.select_related('playbook'))
    ).order_by('-trade_time', '-entry_score')

    # 步骤3: 在内存中处理数据，并聚合剧本信息
    final_logs = []
    all_playbook_objects = set() # 用于收集所有出现过的Playbook对象

    for signal in latest_buy_signals:
        active_playbooks = []
        if hasattr(signal, 'playbook_details'):
            for detail in signal.playbook_details.all():
                if detail.playbook:
                    # 将Playbook对象本身加入列表，方便后续使用
                    active_playbooks.append(detail.playbook)
                    all_playbook_objects.add(detail.playbook)
        
        # 按优先级排序剧本
        active_playbooks.sort(key=lambda p: get_playbook_priority(p.cn_name or p.name))

        final_logs.append({
            'log_id': signal.id,
            'stock': signal.stock,
            'latest_trade_time': signal.trade_time,
            'latest_score': signal.entry_score,
            'active_playbooks': active_playbooks, # 现在是Playbook对象列表
            'strategy_name': signal.strategy_name,
        })

    # 步骤4: 从收集到的Playbook对象中提取唯一剧本用于筛选器
    unique_playbooks = sorted(list(all_playbook_objects), key=lambda p: get_playbook_priority(p.cn_name or p.name))

    # 步骤5: 根据请求参数进行剧本筛选
    selected_playbook_ids = request.GET.getlist('playbooks') # 前端现在传递playbook的ID
    filtered_logs = final_logs
    if selected_playbook_ids:
        selected_ids_int = {int(pid) for pid in selected_playbook_ids}
        filtered_logs = [
            log for log in final_logs
            if selected_ids_int.issubset({p.id for p in log['active_playbooks']})
        ]

    # 步骤6: 分页
    paginator = Paginator(filtered_logs, 25)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # 步骤7: 准备上下文
    context = {
        'page_title': '策略状态监控中心 (近3日买入)',
        'items_for_display': page_obj.object_list, # 直接使用分页后的列表
        'page_obj': page_obj,
        'total_count': paginator.count,
        'all_playbooks': unique_playbooks, # Playbook对象列表
        'selected_playbooks': [int(pid) for pid in selected_playbook_ids], # 传递ID列表
    }
    # --- 代码修改结束 ---
    return render(request, 'dashboard/trend_following_list.html', context)


@login_required
def fav_trend_following_list(request):
    """
    【V405.0 新模型适配版】
    - 核心修改: 适配 FavoriteStockTracker 中关联的新 TradingSignal 模型。
    - 模板适配: 调整传递给模板的数据，以匹配新模型的字段。
    """
    # --- 代码修改开始 ---
    # [修改原因] 适配 FavoriteStockTracker 的外键变更
    
    # 步骤1: 获取用户的所有追踪器，预加载新的关联模型
    base_queryset = FavoriteStockTracker.objects.filter(
        user=request.user
    ).select_related(
        'stock', 
        'entry_signal', # 旧的 entry_log -> 新的 entry_signal
        'latest_signal',# 旧的 latest_log -> 新的 latest_signal
        'exit_signal'  # 旧的 exit_log -> 新的 exit_signal
    ).prefetch_related(
        # 预加载最新信号的剧本详情，以在模板中显示
        Prefetch(
            'latest_signal__playbook_details',
            queryset=SignalPlaybookDetail.objects.select_related('playbook'),
            to_attr='prefetched_playbook_details' # 将结果存入一个新属性
        )
    )
    
    # 获取元数据的方式不变
    unified_config = load_strategy_config('config/trend_follow_strategy.json')
    tactical_engine = TrendFollowStrategy(config=unified_config)
    # 注意：reporting_layer 不再有 signal_metadata，元数据现在直接在 playbook 对象上
    # 我们需要从数据库加载所有 playbook 以构建一个元数据映射
    playbook_metadata = {p.name: p.cn_name for p in Playbook.objects.all()}

    # --- 步骤2: 状态筛选 (逻辑不变) ---
    status_filter = request.GET.get('status', 'holding')
    if status_filter == 'holding':
        queryset = base_queryset.filter(status=FavoriteStockTracker.Status.HOLDING)
        page_title = '自选股持仓监控'
    elif status_filter == 'sold':
        queryset = base_queryset.filter(status=FavoriteStockTracker.Status.SOLD)
        page_title = '自选股历史平仓'
    else:
        page_title = '全部自选追踪' 
        queryset = base_queryset

    # --- 步骤3: 排序 (逻辑不变) ---
    if status_filter == 'sold':
        ordered_queryset = queryset.order_by('-exit_date')
    else:
        ordered_queryset = queryset.order_by('-latest_date')

    # --- 步骤4: 分页 (逻辑不变) ---
    paginator = Paginator(ordered_queryset, 25)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # --- 步骤5: 准备最终上下文并渲染 ---
    context = {
        'page_title': page_title,
        'page_obj': page_obj,
        'total_count': paginator.count,
        'status_filter': status_filter,
        'playbook_metadata': playbook_metadata, # 传递新的元数据字典
    }
    # --- 代码修改结束 ---
    return render(request, 'dashboard/fav_trend_following_list.html', context)

@login_required
@with_cache_manager_for_views
def realtime_engine_view(request, cache_manager=None):
    """
    【V2.1 - 策略信号版】渲染盘中引擎实时监控页面。
    - 核心修改: 只从最终的策略信号键 (ZSET) 中获取数据。
    """
    user = request.user
    today_str = date.today().strftime('%Y-%m-%d')
    cache_key_builder = IntradayEngineCashKey()
    user_dao = UserDAO(cache_manager_instance=cache_manager)

    async def get_initial_signals_for_favorites():
        # 1. 获取用户的自选股列表
        favorite_stocks = await user_dao.get_user_favorites(user.id)
        if not favorite_stocks:
            return []
        
        fav_stock_codes = [fav.stock.stock_code for fav in favorite_stocks if fav.stock]
        
        # 2. 构建所有自选股的信号键
        redis_keys = [cache_key_builder.stock_signals_key(code, today_str) for code in fav_stock_codes]
        
        # 3. 并发地从每个ZSET中获取所有信号
        fetch_tasks = [cache_manager.zrange(key, 0, -1, desc=True) for key in redis_keys]
        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        
        # 4. 合并和处理结果
        all_signals = []
        for stock_signals_json in results:
            if isinstance(stock_signals_json, list) and stock_signals_json:
                # 反序列化从Redis取出的JSON字符串
                for signal_json in stock_signals_json:
                    try:
                        all_signals.append(json.loads(signal_json))
                    except json.JSONDecodeError:
                        continue # 忽略无法解析的脏数据
        
        # 5. 按时间倒序排序所有信号
        all_signals.sort(key=lambda s: s.get('entry_time', ''), reverse=True)
        
        return all_signals

    # 在同步视图中调用异步函数
    initial_signals = async_to_sync(get_initial_signals_for_favorites)()
    
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

    def create(self, request, *args, **kwargs):
        """
        【V405.0 新模型适配版】
        """
        user = request.user
        
        # --- 代码修改开始 ---
        # [修改原因] 适配新的 TradingSignal 模型
        signal_id = request.data.get('signal_id') # 前端应传递 signal_id
        if not signal_id:
            return Response({'detail': '必须提供一个有效的建仓信号ID。'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # 从 TradingSignal 查询
            entry_signal = TradingSignal.objects.select_related('stock').get(
                id=signal_id, 
                signal_type=TradingSignal.SignalType.BUY
            )
        except TradingSignal.DoesNotExist:
            return Response({'detail': '指定的建仓信号不存在或无效。'}, status=status.HTTP_404_NOT_FOUND)

        stock = entry_signal.stock
        # --- 代码修改结束 ---

        if FavoriteStock.objects.filter(user=user, stock=stock).exists():
            favorite = FavoriteStock.objects.get(user=user, stock=stock)
        else:
            serializer = self.get_serializer(data={'stock_code': stock.stock_code})
            serializer.is_valid(raise_exception=True)
            favorite = FavoriteStock.objects.create(user=user, stock=stock)

        # --- 代码修改开始 ---
        # [修改原因] 适配 FavoriteStockTracker 的新字段
        tracker, _ = FavoriteStockTracker.objects.update_or_create(
            user=user, stock=stock,
            defaults={
                'status': FavoriteStockTracker.Status.HOLDING,
                'entry_signal': entry_signal,
                'entry_price': entry_signal.close_price,
                'entry_date': entry_signal.trade_time,
                'latest_signal': entry_signal,
                'latest_price': entry_signal.close_price,
                'latest_date': entry_signal.trade_time,
                'health_change_summary': entry_signal.health_change_summary or {},
                'exit_signal': None,
                'exit_price': None,
                'exit_date': None,
            }
        )
        # --- 代码修改结束 ---

        websocket_payload = {
            'id': favorite.id,
            'code': stock.stock_code,
            'name': stock.stock_name,
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
        
        send_update_to_user_sync(
            user_id=user.id,
            sub_type='favorite_added_with_data',
            payload=websocket_payload
        )
        logger.info(f"已通过 WebSocket向用户 {user.username} 推送新自选股 {stock.stock_code} 的更新。")

        try:
            from tasks.tushare.stock_tasks import fetch_data_for_new_favorite
            task_args = (user.id, stock.stock_code, favorite.id)
            fetch_data_for_new_favorite.apply_async(args=task_args, queue=target_queue)
            logger.info(f"已将后台任务发送到队列 '{target_queue}' 为用户 {user.id} 的新自选股 {stock.stock_code} 获取数据。")
        except ImportError:
            logger.error("无法导入后台任务 'fetch_data_for_new_favorite'，跳过任务触发。")
        except Exception as task_error:
            logger.error(f"触发后台任务 fetch_data_for_new_favorite 时出错: {task_error}", exc_info=True)

        response_data = FavoriteStockSerializer(instance=favorite).data
        return Response(response_data, status=status.HTTP_201_CREATED)

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
