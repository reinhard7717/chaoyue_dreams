# dashboard/views.py
import asyncio
import json
from datetime import datetime
from django.utils import timezone
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
# 这个函数封装了排序、分页、数据处理和渲染的重复代码，使视图函数更简洁。
def _render_strategy_list_page(request, base_queryset, page_title, template_name):
    """
    一个通用的辅助函数，用于渲染策略列表页面。
    【最终修正版 V2】: 移除了此处的 .annotate() 调用。
                     所有计算字段现在都由DAO层提供。
    """
    print(f"--- [View] 开始渲染页面: {page_title} ---")
    
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    # 【代码修改】删除整个 .annotate() 块。base_queryset 已包含所有字段。
    #    现在 ordered_queryset 就是 base_queryset。
    ordered_queryset = base_queryset
    print("--- [View] DAO已提供完整数据，跳过视图层注解 ---")
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

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
    # 1. 获取当前用户的初始自选股列表
    # --- 使用 async_to_sync 调用异步 DAO 方法 ---
    get_favorites_async = user_dao.get_user_favorites # 获取异步方法本身
    user_favorites = async_to_sync(get_favorites_async)(request.user.id) # 调用包装后的同步版本
    # 2. 准备传递给模板的数据结构
    initial_favorites_data = [
        {
            'id': fav.id,
            'code': fav.stock.stock_code,
            'name': fav.stock.stock_name,
            # 初始加载时，实时数据为空，等待 WebSocket 推送
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
        for fav in user_favorites if fav.stock  # 只保留有 stock 的
    ]
    initial_favorites_json_string = json.dumps(initial_favorites_data, cls=DjangoJSONEncoder)
    # 3. 定义上下文
    context = {
        # 将初始数据转换为 JSON 字符串，以便在模板中安全地嵌入到 JavaScript 变量中
        'initial_favorites_json': initial_favorites_json_string,
    }
    # 4. 渲染模板
    return render(request, 'dashboard/home.html', context)

login_required
def monthly_trend_strategy_list(request):
    """
    【重构】月线趋势跟踪策略列表页。
    所有复杂逻辑已移至 _render_strategy_list_page 辅助函数。
    """
    str_dao = StrategiesDAO()
    # 假设 get_latest_monthly_trend_reports 返回的是 QuerySet
    all_reports_queryset = async_to_sync(str_dao.get_latest_monthly_trend_reports)()
    
    return _render_strategy_list_page(
        request=request,
        base_queryset=all_reports_queryset,
        page_title="月线趋势跟踪评分",
        template_name='dashboard/monthly_trend_strategy.html'
    )

@login_required
def fav_monthly_trend_list(request):
    """
    【重构】展示用户自选股的月线趋势策略信号。
    所有复杂逻辑已移至 _render_strategy_list_page 辅助函数。
    """
    user_dao = UserDAO()
    str_dao = StrategiesDAO()
    
    user_favorites = async_to_sync(user_dao.get_user_favorites)(request.user.id)
    fav_codes = [fav.stock.stock_code for fav in user_favorites if fav.stock]
    
    # 假设 get_latest_monthly_trend_reports_by_stock_codes 返回的是 QuerySet
    all_reports_queryset = async_to_sync(str_dao.get_latest_monthly_trend_reports_by_stock_codes)(fav_codes)

    return _render_strategy_list_page(
        request=request,
        base_queryset=all_reports_queryset,
        page_title="自选股-月线趋势跟踪",
        template_name='dashboard/fav_monthly_trend_list.html'
    )

def get_playbook_priority(playbook_name):
    """
    根据剧本名称中的关键字为其分配优先级。
    返回值越小，优先级越高。
    """
    playbook_name_upper = playbook_name.upper()
    if 'ROCKET' in playbook_name_upper or '火箭' in playbook_name:
        return 0  # 最高优先级：火箭信号
    if 'BREAKOUT_TRIGGER_SCORE' in playbook_name_upper or '王牌突破' in playbook_name:
        return 1  # 次高优先级：王牌信号
    if '【专家】' in playbook_name:
        return 2  # 专家信号
    if '【加分】' in playbook_name or '资金流入' in playbook_name:
        return 4  # 较低优先级：加分项和资金项
    if 'MA20' in playbook_name_upper or '均线' in playbook_name:
        return 5  # 最低优先级：基础形态确认
    return 3 # 默认优先级
# ▲▲▲【代码修改】: 结束 ▲▲▲


@login_required
def trend_following_list(request):
    """
    【V2.8 修正最新时间】策略状态监控中心视图
    - 核心修正: 聚合逻辑中增加对 latest_trade_time 的显式比较和更新，确保其与 latest_score 一样，总是反映最真实的情况。
    - 功能保持: 剧本筛选功能不变。
    """
    print("--- [View] 开始渲染策略状态监控中心 (trend_following_list) ---") # 调试信息
    # 1. 定义持仓状态的查询条件
    held_status_query = Q(last_buy_time__isnull=False) & (
        Q(last_sell_time__isnull=True) | Q(last_buy_time__gt=F('last_sell_time'))
    )
    
    # 2. 获取URL中的筛选参数
    selected_playbooks = request.GET.getlist('playbooks')

    # 3. 构建基础查询集
    base_queryset = TrendFollowStrategyState.objects.filter(held_status_query)

    # 4. 动态应用筛选条件
    if selected_playbooks:
        for playbook in selected_playbooks:
            base_queryset = base_queryset.filter(active_playbooks__contains=playbook)
    
    # 5. 获取所有可用的剧本标签
    all_playbook_lists = TrendFollowStrategyState.objects.filter(
        held_status_query
    ).values_list('active_playbooks', flat=True)
    unique_playbooks = sorted(list(set(chain.from_iterable(p for p in all_playbook_lists if p))))

    # 6. 从数据库获取满足条件的原始数据
    # 这里的排序仅为初步排序，真正的最值将在聚合步骤中确定
    all_held_states = base_queryset.select_related('stock').order_by('stock__stock_code')

    # 7. 聚合处理
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
        
        # ▼▼▼【代码修改】: 修正寻找最新时间和最高分的逻辑 ▼▼▼
        # 原逻辑只更新分数，不更新时间，导致时间戳可能不正确。
        # 新逻辑确保同时更新时间和分数，保证数据一致性。
        
        # 更新激活剧本和策略名 (逻辑不变)
        if state.active_playbooks:
            aggregated_results[stock_code]['active_playbooks'].extend(state.active_playbooks)
        aggregated_results[stock_code]['strategy_names'].add(state.strategy_name)

        # 新增：显式比较并更新为真正的最新交易时间
        if state.latest_trade_time > aggregated_results[stock_code]['latest_trade_time']:
            # print(f"调试: 股票 {stock_code} 的最新时间从 {aggregated_results[stock_code]['latest_trade_time']} 更新为 {state.latest_trade_time}") # 调试信息
            aggregated_results[stock_code]['latest_trade_time'] = state.latest_trade_time

        # 更新为最高分数 (逻辑不变，但现在与时间更新逻辑并列，更清晰)
        if state.latest_score > aggregated_results[stock_code]['latest_score']:
            aggregated_results[stock_code]['latest_score'] = state.latest_score
        # ▲▲▲【代码修改结束】▲▲▲

    # 8. 后处理和排序
    final_list = list(aggregated_results.values())
    for item in final_list:
        # 对剧本进行去重和排序
        item['active_playbooks'] = sorted(list(set(item['active_playbooks'])), key=get_playbook_priority)
        item['strategy_names'] = sorted(list(item['strategy_names']))
    
    # 使用正确的最新时间进行最终排序
    final_list.sort(key=lambda x: (x['latest_trade_time'], x['latest_score']), reverse=True)
    
    # 9. 分页
    paginator = Paginator(final_list, 25)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # 10. 准备上下文并渲染模板
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
    【V3.5 修正排序逻辑】
    - 核心修正: 重构列表排序逻辑，确保在每个优先级内部，都按最新时间倒序排列。
    - 排序规则: 1. 按优先级 (警报 > 持仓 > 等待)； 2. 在同一优先级内，按最新动态时间从新到旧。
    """
    print("--- [View] 开始渲染自选股持仓监控页面 (fav_trend_following_list) ---")
    user_dao = UserDAO()
    
    user_favorites = async_to_sync(user_dao.get_user_favorites)(request.user.id)
    fav_codes = [fav.stock.stock_code for fav in user_favorites if fav.stock]
    
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
                'latest_state': None,
                'playbooks': set()
            }
        
        summary = stock_summary[stock_code]
        
        if state.entry_signal and (not summary['buy_state'] or state.trade_time > summary['buy_state'].trade_time):
            summary['buy_state'] = state
        
        if state.exit_signal_code > 0 and (not summary['sell_state'] or state.trade_time > summary['sell_state'].trade_time):
            summary['sell_state'] = state
            
        if not summary['latest_state'] or state.trade_time > summary['latest_state'].trade_time:
            summary['latest_state'] = state

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
            'buy_info': None,
            'sell_info': None,
            'latest_trade_time': latest_state.trade_time,
            'latest_score': latest_state.entry_score,
            'active_playbooks': sorted(list(summary['playbooks']), key=get_playbook_priority),
            'swing_status': '等待建仓',
            'status_class': 'status-wait',
            'sort_priority': 3, # 值越小，优先级越高
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

            if sell_state and sell_state.trade_time > buy_state.trade_time:
                item['sell_info'] = {
                    'time': sell_state.trade_time,
                    'strategy_name': sell_state.strategy_name,
                    'time_level': sell_state.timeframe,
                    'severity_level': sell_state.exit_severity_level,
                    'reason': sell_state.exit_signal_reason
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

    # ▼▼▼【代码修改】: 修正排序逻辑 ▼▼▼
    # 定义一个极早的、带时区的时间，用于处理 None 值，确保它们排在最后
    min_aware_datetime = timezone.make_aware(datetime.min, timezone.utc)
    
    # 核心排序逻辑：
    # 1. 按 'sort_priority' 升序排 (1, 2, 3)，使得警报在最前。
    # 2. 在同一优先级内，按 'latest_trade_time' 降序排 (从新到旧)。
    processed_list.sort(
        key=lambda x: (x['sort_priority'], x['latest_trade_time'] or min_aware_datetime),
        reverse=False # 优先级升序
    )
    # 由于Python的sort是稳定的，我们可以先按次要标准（时间）排序，再按主要标准（优先级）排序
    # 这里采用一个更Pythonic的方式，直接在一个lambda中完成
    processed_list.sort(key=lambda x: (
        x['sort_priority'], # 主要排序键：优先级升序
        -(x['latest_trade_time'].timestamp() if x['latest_trade_time'] else 0) # 次要排序键：时间降序（通过取负的时间戳实现）
    ))
    print(f"调试: 排序后第一个元素的优先级: {processed_list[0]['sort_priority'] if processed_list else 'N/A'}, 时间: {processed_list[0]['latest_trade_time'] if processed_list else 'N/A'}")
    # ▲▲▲【代码修改结束】▲▲▲

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
    """
    股票搜索 API (GET /api/dashboard/search/?q=...)
    """
    serializer_class = StockInfoSerializer # <--- 使用对应的 Serializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        """根据 stock_code 或 stock_name 搜索 StockInfo"""
        query = self.request.query_params.get('q', None)
        if query:
            # --- 使用正确的字段名进行过滤 ---
            return StockInfo.objects.filter(
                Q(stock_code__icontains=query) | Q(stock_name__icontains=query)
            )[:10] # 限制返回结果数量
            # --- 修改结束 ---
        return StockInfo.objects.none() # 如果没有查询参数，返回空

class FavoriteStockViewSet(viewsets.ModelViewSet):
    """
    自选股 API (GET, POST, DELETE /api/dashboard/favorites/)
    """
    serializer_class = FavoriteStockSerializer
    permission_classes = [IsAuthenticated] # 必须登录

    def get_queryset(self):
        """只返回当前用户的自选股"""
        return FavoriteStock.objects.filter(user=self.request.user).select_related('stock')

    def perform_create(self, serializer):
        """
        处理添加自选股请求：保存到数据库，并触发后台任务获取详细数据后推送。
        """
        # 1. 保存 FavoriteStock 实例，并获取创建的对象
        favorite = serializer.save(user=self.request.user)
        # 立即推送基础数据
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
        # 2. 触发 Celery 任务，异步获取行情并推送
        try:
            from tasks.tushare.stock_tasks import fetch_data_for_new_favorite
            task_args = (
                self.request.user.id,
                favorite.stock.stock_code,  # 注意这里是 stock_code
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
        user_id = instance.user.id # 在删除前获取 user_id
        instance.delete()
        # --- 推送更新 ---
        updated_list = self._get_formatted_favorites(user_id) # 需要根据 user_id 获取
        send_update_to_user_sync(
            user_id=user_id,
            sub_type='favorites_update',
            payload=updated_list
        )
    
    def _get_formatted_favorites(self, user_or_id):
        """辅助方法获取格式化的自选股列表"""
        if isinstance(user_or_id, int):
            favorites = FavoriteStock.objects.filter(user_id=user_or_id).select_related('stock').order_by('added_at')
        else: # 假设是 User 对象
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
    # ModelViewSet 自动处理 list (GET), create (POST), retrieve (GET /id/),
    # update (PUT/PATCH /id/), destroy (DELETE /id/)
    # 我们主要用 list, create, destroy













