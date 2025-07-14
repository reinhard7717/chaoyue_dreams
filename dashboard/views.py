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
    【V117.17 终极重构版】
    - 核心革命: 彻底废弃对 TrendFollowStrategyState 摘要表的依赖，回归最原始、最可靠的
                TrendFollowStrategySignalLog 作为数据源。
    - 工作流程:
      1. 获取所有股票的最新“买入”信号。
      2. 对于每一个买入信号，查找其后是否跟随了“卖出”信号。
      3. 只筛选出那些“未卖出”或“卖出时间早于最新买入时间”的股票，即真正“持仓中”的股票。
      4. 基于这些最新的、权威的买入信号记录，构建前端所需的所有信息。
    - 收益: 从根本上解决了因状态摘要表聚合逻辑错误导致的数据不一致和分数不更新问题。
            确保页面展示的每一条数据，都源自一个真实的、完整的、权威的信号事件。
    """
    print("--- [View] 开始渲染策略状态监控中心 (trend_following_list) V117.17 ---")

    # 步骤1: 获取所有股票的最新买入信号
    # 使用 .values() 和 .annotate() 在数据库层面高效完成
    latest_buy_signals = TrendFollowStrategySignalLog.objects.filter(
        entry_signal=True
    ).values('stock_id').annotate(
        latest_buy_id=Max('id')
    )
    latest_buy_ids = [item['latest_buy_id'] for item in latest_buy_signals]

    # 步骤2: 获取这些最新买入信号的完整对象
    # .select_related('stock') 优化数据库查询
    buy_logs = TrendFollowStrategySignalLog.objects.filter(
        id__in=latest_buy_ids
    ).select_related('stock')

    # 步骤3: 获取所有股票的最新卖出信号
    latest_sell_signals = TrendFollowStrategySignalLog.objects.filter(
        exit_signal_code__gt=0
    ).values('stock_id').annotate(
        latest_sell_time=Max('trade_time')
    )
    sell_time_map = {item['stock_id']: item['latest_sell_time'] for item in latest_sell_signals}

    # 步骤4: 筛选出真正“持仓中”的股票
    held_items = []
    for buy_log in buy_logs:
        last_sell_time = sell_time_map.get(buy_log.stock_id)
        # 持仓条件：没有卖出记录，或者最新的卖出时间早于最新的买入时间
        if last_sell_time is None or last_sell_time < buy_log.trade_time:
            # 这是一个真正持仓中的股票，基于这条权威的买入日志构建前端所需数据
            item = {
                'stock': buy_log.stock,
                'latest_trade_time': buy_log.trade_time, # 最新信号时间就是这次买入的时间
                'latest_score': buy_log.entry_score,     # 分数就是这次买入的分数
                'last_buy_time': buy_log.trade_time,     # 最新买入时间就是这次买入的时间
                'last_sell_time': last_sell_time,        # 记录下（可能存在的）旧的卖出时间
                'active_playbooks': sorted(buy_log.triggered_playbooks, key=get_playbook_priority),
                'strategy_names': [buy_log.strategy_name], # 策略名就是这次买入的策略名
            }
            held_items.append(item)

    # 步骤5: (可选) 剧本筛选逻辑
    selected_playbooks = request.GET.getlist('playbooks')
    if selected_playbooks:
        filtered_items = []
        for item in held_items:
            # 检查该持仓项的剧本是否包含所有被选中的剧本
            if all(playbook in item['active_playbooks'] for playbook in selected_playbooks):
                filtered_items.append(item)
        final_list = filtered_items
    else:
        final_list = held_items
        
    # 步骤6: 准备筛选器中的剧本列表
    all_playbook_lists = [item['active_playbooks'] for item in held_items]
    unique_playbooks = sorted(
        list(set(chain.from_iterable(p for p in all_playbook_lists if p))), 
        key=get_playbook_priority
    )

    # 步骤7: 排序和分页
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
    【V117.18 终极重构版】
    - 核心革命: 与 trend_following_list 保持同步，彻底废弃旧的、基于循环和字典聚合的
                复杂逻辑，回归最原始、最可靠的 TrendFollowStrategySignalLog 作为数据源。
    - 工作流程:
      1. 获取用户自选股列表。
      2. 在自选股范围内，获取所有股票的最新“买入”信号。
      3. 对于每一个买入信号，查找其后是否跟随了“卖出”信号。
      4. 基于这些权威的买入和卖出信号，构建前端所需的所有持仓状态信息。
    - 收益: 确保了自选股监控页面的数据与主监控中心完全一致，解决了所有潜在的数据
            不一致和状态判断错误问题。
    """
    print("--- [View] 开始渲染自选股持仓监控 (fav_trend_following_list) V117.18 ---")
    user_dao = UserDAO()
    
    # 步骤1: 获取用户自选股列表
    user_favorites = async_to_sync(user_dao.get_user_favorites)(request.user.id)
    fav_codes = [fav.stock.stock_code for fav in user_favorites if fav.stock]
    fav_id_map = {fav.stock.stock_code: fav.id for fav in user_favorites if fav.stock}

    if not fav_codes:
        return render(request, 'dashboard/fav_trend_following_list.html', {
            'page_title': '自选股持仓监控',
            'page_obj': None,
            'total_count': 0,
        })

    # 步骤2: 在自选股范围内，获取所有股票的最新买入信号
    latest_buy_signals = TrendFollowStrategySignalLog.objects.filter(
        stock__stock_code__in=fav_codes,
        entry_signal=True
    ).values('stock_id').annotate(
        latest_buy_id=Max('id')
    )
    latest_buy_ids = [item['latest_buy_id'] for item in latest_buy_signals]
    buy_logs = TrendFollowStrategySignalLog.objects.filter(id__in=latest_buy_ids).select_related('stock')
    buy_logs_map = {log.stock_id: log for log in buy_logs}

    # 步骤3: 在自选股范围内，获取所有股票的最新卖出信号
    latest_sell_signals = TrendFollowStrategySignalLog.objects.filter(
        stock__stock_code__in=fav_codes,
        exit_signal_code__gt=0
    ).values('stock_id').annotate(
        latest_sell_id=Max('id')
    )
    latest_sell_ids = [item['latest_sell_id'] for item in latest_sell_signals]
    sell_logs = TrendFollowStrategySignalLog.objects.filter(id__in=latest_sell_ids).select_related('stock')
    sell_logs_map = {log.stock_id: log for log in sell_logs}

    # 步骤4: 遍历所有自选股，构建最终的展示列表
    processed_list = []
    for fav_stock_code in fav_codes:
        # 从预先查好的 stock 对象中获取，避免N+1查询
        stock_obj = next((fav.stock for fav in user_favorites if fav.stock.stock_code == fav_stock_code), None)
        if not stock_obj: continue

        buy_log = buy_logs_map.get(stock_obj.stock_code)
        sell_log = sell_logs_map.get(stock_obj.stock_code)

        # 初始化默认状态
        item = {
            'stock': stock_obj,
            'favorite_id': fav_id_map.get(fav_stock_code),
            'buy_info': None,
            'sell_info': None,
            'latest_trade_time': None, # 将在后面填充
            'latest_score': 0,
            'active_playbooks': [],
            'swing_status': '等待建仓',
            'status_class': 'status-wait',
            'sort_priority': 3,
        }

        # 如果存在买入信号
        if buy_log:
            # 检查是否持仓 (卖出信号不存在，或卖出时间早于买入时间)
            is_holding = sell_log is None or sell_log.trade_time < buy_log.trade_time
            
            item['latest_trade_time'] = buy_log.trade_time
            item['latest_score'] = buy_log.entry_score
            item['active_playbooks'] = sorted(buy_log.triggered_playbooks, key=get_playbook_priority)
            item['buy_info'] = {
                'time': buy_log.trade_time,
                'strategy_name': buy_log.strategy_name,
                'time_level': buy_log.timeframe
            }

            if is_holding:
                item['swing_status'] = '持仓观察'
                item['status_class'] = 'status-holding'
                item['sort_priority'] = 2
            else: # 已卖出
                item['sell_info'] = {
                    'time': sell_log.trade_time,
                    'strategy_name': sell_log.strategy_name,
                    'time_level': sell_log.timeframe,
                    'severity_level': sell_log.exit_severity_level,
                    'reason': sell_log.exit_signal_reason or "风险预警",
                    'code': sell_log.exit_signal_code,
                }
                level = sell_log.exit_severity_level
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
        
        # 如果没有买入信号，但有卖出信号，也展示最新的卖出状态
        elif sell_log:
            item['latest_trade_time'] = sell_log.trade_time
            item['sell_info'] = {
                'time': sell_log.trade_time,
                'strategy_name': sell_log.strategy_name,
                'time_level': sell_log.timeframe,
                'severity_level': sell_log.exit_severity_level,
                'reason': sell_log.exit_signal_reason or "风险预警",
                'code': sell_log.exit_signal_code,
            }
            item['swing_status'] = '空仓预警'
            item['status_class'] = 'status-alert-level-2' # 给一个默认的警报样式
            item['sort_priority'] = 4 # 排在最后

        processed_list.append(item)

    # 步骤5: 排序和分页
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
