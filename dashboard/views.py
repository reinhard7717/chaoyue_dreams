# dashboard/views.py
import asyncio
import json
from asgiref.sync import async_to_sync
from django.db.models import Max, F, Q
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
    
    # ▼▼▼【代码修改】: 统一判断逻辑 ▼▼▼
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
    # ▲▲▲【代码修改结束】▲▲▲

@login_required
def trend_following_list(request):
    """
    【V3.1 最终排序修正】策略状态监控中心视图
    - 核心修正: get_playbook_priority 函数逻辑已与模板完全同步，确保排序正确。
    """
    print("--- [View] 开始渲染策略状态监控中心 (trend_following_list) V3.1 ---")
    held_status_query = Q(last_buy_time__isnull=False) & (
        Q(last_sell_time__isnull=True) | Q(last_buy_time__gt=F('last_sell_time'))
    )

    selected_playbooks = request.GET.getlist('playbooks')

    base_queryset = TrendFollowStrategyState.objects.filter(held_status_query)

    if selected_playbooks:
        for playbook in selected_playbooks:
            base_queryset = base_queryset.filter(active_playbooks__contains=playbook)
    
    all_playbook_lists = TrendFollowStrategyState.objects.filter(
        held_status_query
    ).values_list('active_playbooks', flat=True)
    
    # 此处调用已修正的 get_playbook_priority 函数，现在排序将完全正确
    unique_playbooks = sorted(
        list(set(chain.from_iterable(p for p in all_playbook_lists if p))), 
        key=get_playbook_priority
    )
    print(f"--- [View] 筛选区剧本已按最终优先级排序: {unique_playbooks}") # 调试信息

    all_held_states = base_queryset.select_related('stock').order_by('stock__stock_code')

    aggregated_results = OrderedDict()
    for state in all_held_states:
        stock_code = state.stock.stock_code
        if stock_code not in aggregated_results:
            aggregated_results[stock_code] = {
                'stock': state.stock,
                'latest_trade_time': state.latest_trade_time,
                'latest_score': state.latest_score,
                'last_buy_time': state.last_buy_time,
                'last_sell_time': state.last_sell_time,
                'active_playbooks': [],
                'strategy_names': set(),
            }
        
        if state.active_playbooks:
            aggregated_results[stock_code]['active_playbooks'].extend(state.active_playbooks)
        aggregated_results[stock_code]['strategy_names'].add(state.strategy_name)

        if state.latest_trade_time > aggregated_results[stock_code]['latest_trade_time']:
            aggregated_results[stock_code]['latest_trade_time'] = state.latest_trade_time

        if state.latest_score > aggregated_results[stock_code]['latest_score']:
            aggregated_results[stock_code]['latest_score'] = state.latest_score

    final_list = list(aggregated_results.values())
    for item in final_list:
        # 表格内的排序同样会使用修正后的函数，保持一致
        item['active_playbooks'] = sorted(list(set(item['active_playbooks'])), key=get_playbook_priority)
        item['strategy_names'] = sorted(list(item['strategy_names']))
    
    final_list.sort(key=lambda x: (x['latest_trade_time'], x['latest_score']), reverse=True)
    
    paginator = Paginator(final_list, 25)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_title': '策略状态监控中心',
        'page_obj': page_obj,
        'total_count': len(final_list),
        'all_playbooks': unique_playbooks,
        'selected_playbooks': selected_playbooks,
    }
    return render(request, 'dashboard/trend_following_list.html', context)

@login_required
def fav_trend_following_list(request):
    """
    【V117.9 链路修复版】
    - 核心修正: 调整了状态判断逻辑，确保始终基于最新的买入信号来判断后续的风险状态，
                使得持仓状态的展示更加精确和符合交易直觉。
    """
    user_dao = UserDAO()
    
    user_favorites = async_to_sync(user_dao.get_user_favorites)(request.user.id)
    fav_codes = [fav.stock.stock_code for fav in user_favorites if fav.stock]
    fav_id_map = {fav.stock.stock_code: fav.id for fav in user_favorites if fav.stock}

    if not fav_codes:
        return render(request, 'dashboard/fav_trend_following_list.html', {
            'page_title': '自选股持仓监控',
            'page_obj': None,
            'total_count': 0,
        })

    state_list_qs = TrendFollowStrategySignalLog.objects.filter(
        stock__stock_code__in=fav_codes
    ).select_related('stock').order_by('stock__stock_code', '-trade_time')

    stock_summary = {}
    for state in state_list_qs:
        stock_code = state.stock.stock_code
        if stock_code not in stock_summary:
            stock_summary[stock_code] = {
                'stock': state.stock,
                'buy_state': None,
                'sell_state': None,
                'latest_state': state, # 直接将第一条(最新的)记录设为latest_state
                'playbooks': set()
            }
        
        summary = stock_summary[stock_code]
        
        # ▼▼▼【代码修改 V117.9】: 优化状态判断逻辑 ▼▼▼
        # 寻找最新的买入信号
        if state.entry_signal and not summary['buy_state']:
            summary['buy_state'] = state
        
        # 如果已经找到了最新的买入信号，再寻找这个买入信号之后的最新的卖出信号
        if summary['buy_state'] and state.exit_signal_code > 0 and state.trade_time > summary['buy_state'].trade_time:
            if not summary['sell_state'] or state.trade_time > summary['sell_state'].trade_time:
                 summary['sell_state'] = state
        # ▲▲▲【代码修改 V117.9】▲▲▲

        if state.triggered_playbooks:
            summary['playbooks'].update(state.triggered_playbooks)

    processed_list = []
    for stock_code, summary in stock_summary.items():
        buy_state = summary['buy_state']
        sell_state = summary['sell_state']
        latest_state = summary['latest_state']

        if not latest_state:
            continue

        item = {
            'stock': summary['stock'],
            'favorite_id': fav_id_map.get(stock_code),
            'buy_info': None,
            'sell_info': None,
            'latest_trade_time': latest_state.trade_time,
            'latest_score': latest_state.entry_score,
            'active_playbooks': sorted(list(summary['playbooks']), key=get_playbook_priority),
            'swing_status': '等待建仓',
            'status_class': 'status-wait',
            'sort_priority': 3,
        }

        if buy_state:
            item['buy_info'] = {
                'time': buy_state.trade_time,
                'strategy_name': buy_state.strategy_name,
                'time_level': buy_state.timeframe
            }
            item['swing_status'] = '持仓观察'
            item['status_class'] = 'status-holding'
            item['sort_priority'] = 2

            # sell_state 现在是基于最新的 buy_state 找到的，逻辑更可靠
            if sell_state:
                item['sell_info'] = {
                    'time': sell_state.trade_time,
                    'strategy_name': sell_state.strategy_name,
                    'time_level': sell_state.timeframe,
                    'severity_level': sell_state.exit_severity_level,
                    'reason': sell_state.exit_signal_reason or "风险预警",
                    'code': sell_state.exit_signal_code,
                }
                
                level = sell_state.exit_severity_level
                if level == 1:
                    item['swing_status'] = '一级预警'
                    item['status_class'] = 'status-alert-level-1'
                elif level == 3:
                    item['swing_status'] = '三级警报'
                    item['status_class'] = 'status-alert-level-3'
                else:
                    item['swing_status'] = '二级警报'
                    item['status_class'] = 'status-alert-level-2'
                
                item['sort_priority'] = 1
        
        processed_list.append(item)

    processed_list.sort(key=lambda x: (
        x['sort_priority'],
        -(x['latest_trade_time'].timestamp() if x['latest_trade_time'] else 0)
    ))

    paginator = Paginator(processed_list, 25)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_title': '自选股持仓监控',
        'page_obj': page_obj,
        'total_count': len(processed_list),
    }
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
