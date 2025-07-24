# dashboard/views.py
import asyncio
import json
from asgiref.sync import async_to_sync
from django.db.models import Max, F, Q, OuterRef, Subquery
from collections import OrderedDict # 导入 OrderedDict
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from dao_manager.tushare_daos.strategies_dao import StrategiesDAO
from dao_manager.tushare_daos.user_dao import UserDAO
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from rest_framework import generics, viewsets
from rest_framework.permissions import IsAuthenticated

from django.core.serializers.json import DjangoJSONEncoder
from stock_models.stock_analytics import TrendFollowStrategySignalLog, TrendFollowStrategyState
from stock_models.stock_basic import StockInfo
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
    【V119.0 终极性能版】
    - 核心重构: 采用非相关子查询(GROUP BY + IN)替代原有的相关子查询(OuterRef/Subquery)，彻底解决数据库性能瓶颈。
    - 核心重构: 将数据过滤、合并、排序等逻辑移至Python内存中处理，极大减轻数据库压力。
    - 核心重构: 优化了“剧本筛选器”的数据获取方式，不再对数据库进行全量查询，而是复用内存中已有的数据。
    - 收益: 页面加载速度从分钟级提升至秒级，实现了质的飞跃，同时保证业务逻辑完全不变。
    """
    print("--- [View] 开始渲染策略状态监控中心 (trend_following_list) V119.0 终极性能版 ---")
    # [修改] 步骤1: 使用更高效的 GROUP BY + IN 查询，一次性获取所有最新的日线级别“买入”信号记录
    # 这是第一个高效的数据库查询，避免了相关子查询的性能陷阱。
    print("--- [View] 步骤1: 开始查询最新买入信号ID...")
    latest_buy_log_ids = TrendFollowStrategySignalLog.objects.filter(
        entry_signal=True,
        timeframe='D'
    ).values('stock_id').annotate(
        latest_id=Max('id')
    ).values_list('latest_id', flat=True)
    # [修改] 根据获取到的最新ID列表，批量查询完整的信号对象
    # 使用 id__in 查询，这是数据库最高效的查询方式之一。
    latest_buy_logs_qs = TrendFollowStrategySignalLog.objects.filter(
        id__in=list(latest_buy_log_ids)
    ).select_related('stock')
    print(f"--- [View] 步骤1完成: 获取到 {latest_buy_logs_qs.count()} 条最新买入信号")
    # [修改] 步骤2: 一次性获取所有相关股票的最新“卖出”时间
    # 这是第二个高效的数据库查询，用于替代原有的N+1相关子查询。
    # [优化] 直接从上一步的QuerySet中获取stock_ids，Django会将其优化为子查询，比传递巨大的ID列表更高效。
    stock_ids = latest_buy_logs_qs.values_list('stock_id', flat=True)
    print("--- [View] 步骤2: 开始查询最新卖出时间...")
    # [优化] 使用 .values() 而不是获取完整对象，减少内存消耗。
    latest_sell_times_qs = TrendFollowStrategySignalLog.objects.filter(
        stock_id__in=stock_ids,
        exit_signal_code__gt=0,
        timeframe='D'
    ).values('stock_id').annotate(
        latest_sell_time=Max('trade_time')
    )
    # [新增] 将查询结果构建成一个高效的查找字典 {stock_id: sell_time}
    sell_time_map = {item['stock_id']: item['latest_sell_time'] for item in latest_sell_times_qs}
    print(f"--- [View] 步骤2完成: 获取到 {len(sell_time_map)} 个股票的最新卖出时间")
    # [修改] 步骤3: 在Python内存中进行数据合并和持仓过滤
    # 这样做避免了复杂的数据库操作，速度极快。
    print("--- [View] 步骤3: 开始在内存中合并数据并筛选持仓股...")
    held_logs = []
    # [优化] 使用 .iterator() 遍历queryset，可以显著降低内存峰值，特别是当 latest_buy_logs_qs 很大时。
    for buy_log in latest_buy_logs_qs.iterator():
        latest_sell_time = sell_time_map.get(buy_log.stock_id)
        # 判断持仓条件：没有卖出记录，或者最后卖出时间早于最后买入时间
        if latest_sell_time is None or latest_sell_time < buy_log.trade_time:
            # [新增] 动态地将 latest_sell_time 附加到对象上，方便模板使用
            buy_log.latest_sell_time = latest_sell_time
            held_logs.append(buy_log)
    print(f"--- [View] 步骤3完成: 筛选出 {len(held_logs)} 只持仓股")
    # [修改] 步骤4: 高效地从内存数据中提取所有可用剧本，用于筛选器
    # 这个操作不再查询数据库，而是直接利用 `held_logs` 列表，性能极高。
    print("--- [View] 步骤4: 开始从内存中聚合剧本列表...")
    all_playbook_lists = [log.triggered_playbooks for log in held_logs if log.triggered_playbooks]
    unique_playbooks = sorted(
        list(set(chain.from_iterable(all_playbook_lists))),
        key=get_playbook_priority
    )
    print(f"--- [View] 步骤4完成: 找到 {len(unique_playbooks)} 个唯一剧本")
    # [修改] 步骤5: 在内存中根据请求参数进行剧本筛选
    print("--- [View] 步骤5: 开始根据用户选择筛选剧本...")
    selected_playbooks = request.GET.getlist('playbooks')
    final_logs = held_logs # 默认是全部持仓记录
    if selected_playbooks:
        # [修改] 直接在Python列表中进行过滤，并增加对 triggered_playbooks 可能为None的健壮性检查。
        final_logs = [
            log for log in held_logs
            if log.triggered_playbooks and all(p in log.triggered_playbooks for p in selected_playbooks)
        ]
    print(f"--- [View] 步骤5完成: 筛选后剩余 {len(final_logs)} 条记录")
    # [修改] 步骤6: 在内存中对最终结果进行排序
    print("--- [View] 步骤6: 开始排序...")
    # [优化] 增加对 trade_time 或 entry_score 可能为 None 的情况的健壮性处理。
    final_logs.sort(key=lambda log: (log.trade_time, log.entry_score or 0), reverse=True)
    print("--- [View] 步骤6完成: 排序完成")
    # [修改] 步骤7: 使用Paginator对Python列表进行分页
    # Paginator可以很好地处理列表，功能和处理QuerySet完全一样。
    print("--- [View] 步骤7: 开始分页...")
    paginator = Paginator(final_logs, 25)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    print("--- [View] 步骤7完成: 分页完成")
    # [修改] 步骤8: 格式化当前页的数据用于模板渲染
    # 这里的逻辑和原来完全一样，只是数据源从 page_obj.object_list (QuerySet) 变成了 page_obj.object_list (list)。
    print("--- [View] 步骤8: 开始格式化最终数据...")
    final_list_for_template = []
    for log in page_obj.object_list:
        final_list_for_template.append({
            'stock': log.stock,
            'latest_trade_time': log.trade_time,
            'latest_score': log.entry_score,
            'last_buy_time': log.trade_time,
            'last_sell_time': log.latest_sell_time, # 使用我们之前附加的属性
            'active_playbooks': sorted(log.triggered_playbooks or [], key=get_playbook_priority), # [优化] 增加对None的健壮性处理
            'strategy_names': [log.strategy_name],
            'stable_platform_price': log.stable_platform_price,
        })
    print("--- [View] 步骤8完成: 格式化完成，准备渲染模板")
    # [修改] 步骤9: 准备上下文
    context = {
        'page_title': '策略状态监控中心',
        'items_for_display': final_list_for_template,
        'page_obj': page_obj,
        'total_count': paginator.count, # paginator.count 现在作用于列表长度
        'all_playbooks': unique_playbooks,
        'selected_playbooks': selected_playbooks,
    }
    return render(request, 'dashboard/trend_following_list.html', context)

@login_required
def fav_trend_following_list(request):
    """
    【V137.0 标准化战报接收版】
    - 核心重构: 再次重构视图逻辑，使其能完美处理新的、统一的“标准化战报”。
    - 收益: 彻底解决所有状态显示不一致、信息丢失的问题。
    """
    print("--- [View] 开始渲染自选股持仓监控 (fav_trend_following_list) V137.0 ---")
    user_dao = UserDAO()
    
    user_favorites = async_to_sync(user_dao.get_user_favorites)(request.user.id)
    fav_codes = [fav.stock.stock_code for fav in user_favorites if fav.stock]
    fav_id_map = {fav.stock.stock_code: fav.id for fav in user_favorites if fav.stock}

    if not fav_codes:
        return render(request, 'dashboard/fav_trend_following_list.html', {'page_title': '自选股持仓监控', 'page_obj': None, 'total_count': 0})

    # 步骤1.1: 获取每个自选股的最新“买入”信号ID
    latest_buy_signal_ids = TrendFollowStrategySignalLog.objects.filter(
        stock__stock_code__in=fav_codes,
        entry_signal=True
    ).values('stock_id').annotate(
        latest_id=Max('id')
    ).values_list('latest_id', flat=True)
    
    # 步骤1.2: 根据ID列表获取完整的买入信号对象
    buy_logs = TrendFollowStrategySignalLog.objects.filter(id__in=list(latest_buy_signal_ids)).select_related('stock')
    buy_logs_map = {log.stock_id: log for log in buy_logs}

    # 步骤2.1: 获取每个自选股的最新“退出类”信号ID
    latest_sell_signal_ids = TrendFollowStrategySignalLog.objects.filter(
        stock__stock_code__in=fav_codes,
        entry_signal=False
    ).values('stock_id').annotate(
        latest_id=Max('id')
    ).values_list('latest_id', flat=True)

    # 步骤2.2: 根据ID列表获取完整的退出信号对象
    sell_logs = TrendFollowStrategySignalLog.objects.filter(id__in=list(latest_sell_signal_ids)).select_related('stock')
    sell_logs_map = {log.stock_id: log for log in sell_logs}

    # 步骤3: 遍历所有自选股，构建最终的展示列表
    processed_list = []
    for fav in user_favorites:
        stock_obj = fav.stock
        if not stock_obj: continue

        buy_log = buy_logs_map.get(stock_obj.stock_code)
        sell_log = sell_logs_map.get(stock_obj.stock_code)

        # 默认状态
        item = {'stock': stock_obj, 'favorite_id': fav.id, 'buy_info': None, 'sell_info': None, 'swing_status': '等待建仓', 'status_class': 'status-wait', 'sort_priority': 4}

        if buy_log:
            item['buy_info'] = {
                'time': buy_log.trade_time,
                'strategy_name': buy_log.strategy_name,
                'time_level': buy_log.timeframe,
                'stable_platform_price': buy_log.stable_platform_price,
                'score': buy_log.entry_score,
                'playbooks': sorted(buy_log.triggered_playbooks, key=get_playbook_priority)
            }
            
            is_holding = sell_log is None or sell_log.trade_time < buy_log.trade_time
            
            if is_holding:
                item['swing_status'] = '持仓观察'
                item['status_class'] = 'status-holding'
                item['sort_priority'] = 1
            else: # 已平仓 (最新的退出信号在买入之后)
                item['sell_info'] = {
                    'time': sell_log.trade_time,
                    'reason': sell_log.exit_signal_reason,
                    'time_level': sell_log.timeframe,
                    'severity_level': sell_log.exit_severity_level
                }
                if sell_log.exit_signal_code > 0:
                    item['swing_status'] = f'{sell_log.exit_severity_level}级警报'
                    item['status_class'] = f'status-alert-level-{sell_log.exit_severity_level}'
                    item['sort_priority'] = 2
                else:
                    item['swing_status'] = '风险已解除'
                    item['status_class'] = 'status-wait'
                    item['sort_priority'] = 3
        elif sell_log: # 无买入信号，但有历史卖出信号
            item['sell_info'] = {
                'time': sell_log.trade_time,
                'reason': sell_log.exit_signal_reason,
                'time_level': sell_log.timeframe,
                'severity_level': sell_log.exit_severity_level
            }
            item['swing_status'] = '空仓预警'
            item['status_class'] = f'status-alert-level-{sell_log.exit_severity_level or 2}'
            item['sort_priority'] = 4

        processed_list.append(item)

    # 步骤4: 排序和分页
    processed_list.sort(key=lambda x: (x['sort_priority'], -(x['buy_info']['time'].timestamp() if x.get('buy_info') and x['buy_info'].get('time') else 0)))
    paginator = Paginator(processed_list, 25)
    page_obj = paginator.get_page(request.GET.get('page'))

    context = {'page_title': '自选股持仓监控', 'page_obj': page_obj, 'total_count': len(processed_list)}
    return render(request, 'dashboard/fav_trend_following_list.html', context)




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
        favorite = serializer.save(user=self.request.user)
        payload = {
            'id': favorite.id,
            'code': favorite.stock.stock_code,
            'name': favorite.stock.stock_name,
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
            user_id=self.request.user.id,
            sub_type='favorite_added_with_data',
            payload=payload
        )
        try:
            from tasks.tushare.stock_tasks import fetch_data_for_new_favorite
            task_args = (
                self.request.user.id,
                favorite.stock.stock_code,
                favorite.id
            )
            
            fetch_data_for_new_favorite.apply_async(
                args=task_args,
                queue=target_queue,
            )
            logger.info(f"已将后台任务发送到队列 '{target_queue}' 为用户 {self.request.user.id} 的新自选股 {favorite.stock.stock_code} 获取数据")
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
