# dashboard/views.py
import asyncio
import json
import re
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
from stock_models.stock_analytics import TrendFollowStrategyState
from stock_models.stock_basic import StockInfo
from users.models import FavoriteStock
from utils.websockets import send_update_to_user_sync
from .serializers import StockInfoSerializer, FavoriteStockSerializer
from tasks.tushare.stock_tasks import fetch_data_for_new_favorite # 导入新任务
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

@login_required
def trend_following_list(request):
    """
    【V2.6 最终聚合版】策略状态监控中心视图

    功能设计:
    - 数据源为 TrendFollowStrategyState 摘要模型，只查询处于有效持仓状态的记录。
    - 核心逻辑是聚合展示，确保每只股票在列表中只出现一次，但会综合其所有活跃策略的信息。
    
    聚合规则:
    1. 股票聚合: 将同一股票代码下的所有活跃持仓记录（例如，日线和周线策略同时触发）合并为一条。
    2. 策略名称: 合并所有触发策略的名称，并去重展示。
    3. 激活剧本: 合并所有触发的剧本列表。
    4. 最高分数: 展示该股票所有活跃策略中的最高分。
    5. 最新时间: 以所有活跃策略中最新的信号时间（latest_trade_time）为准，并以此为主要排序依据。
    """
    # 1. 定义持仓状态的查询条件
    # 条件：最近一次买入时间存在，并且(最近一次卖出时间不存在 或 最近买入晚于最近卖出)
    held_status_query = Q(last_buy_time__isnull=False) & (
        Q(last_sell_time__isnull=True) | Q(last_buy_time__gt=F('last_sell_time'))
    )

    # 2. 从数据库获取所有满足条件的原始数据
    # - select_related('stock'): 预加载关联的股票信息，避免N+1查询，提升性能。
    # - order_by(...): 预排序是高效聚合的关键。它能保证：
    #   a) 同一只股票的记录会连续出现。
    #   b) 对于同一股票，最新的记录会排在最前面。
    all_held_states = TrendFollowStrategyState.objects.filter(
        held_status_query
    ).select_related('stock').order_by('stock__stock_code', '-latest_trade_time')

    # 3. 在Python内存中进行聚合处理
    # 使用有序字典来存储聚合结果，键为股票代码，值为聚合后的信息字典。
    aggregated_results = OrderedDict()

    for state in all_held_states:
        stock_code = state.stock.stock_code
        
        # 如果是第一次遇到这只股票，则初始化其聚合字典
        if stock_code not in aggregated_results:
            # 因为数据已按最新交易时间降序排列，所以我们遇到的第一条记录就是最新的。
            # 我们以此为基础，初始化该股票的聚合信息。
            aggregated_results[stock_code] = {
                'stock': state.stock,
                'latest_trade_time': state.latest_trade_time,
                'latest_score': state.latest_score,
                'last_buy_time': state.last_buy_time,
                'last_sell_time': state.last_sell_time,
                'active_playbooks': [],  # 初始化一个空列表，用于合并所有剧本
                'strategy_names': set(), # 使用集合来自动去重策略名称
            }
        
        # --- 开始合并信息 ---
        # a) 合并当前记录的剧本列表到聚合列表中
        if state.active_playbooks:
            aggregated_results[stock_code]['active_playbooks'].extend(state.active_playbooks)
        
        # b) 将当前记录的策略名称添加到集合中
        aggregated_results[stock_code]['strategy_names'].add(state.strategy_name)
        
        # c) 确保我们总是取到所有记录中最高的那个分数
        if state.latest_score > aggregated_results[stock_code]['latest_score']:
            aggregated_results[stock_code]['latest_score'] = state.latest_score

    # 4. 对聚合结果进行后处理和最终排序
    # a) 将聚合字典的值转换为列表
    final_list = list(aggregated_results.values())
    
    # b) 将策略名称的集合(set)转换为排序后的列表(list)，方便前端模板统一渲染
    for item in final_list:
        item['strategy_names'] = sorted(list(item['strategy_names']))

    # c) 按最终的排序规则（最新交易时间 -> 最高分）对聚合结果进行排序
    final_list.sort(key=lambda x: (x['latest_trade_time'], x['latest_score']), reverse=True)
    
    # 调试信息：打印第一个聚合后的结果，用于检查数据结构是否正确
    if final_list:
        print(f"【调试信息】聚合后的第一条记录: {final_list[0]}")

    # 5. 设置分页
    paginator = Paginator(final_list, 25)  # Paginator可以直接处理列表对象
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # 6. 准备上下文并渲染模板
    context = {
        'page_title': '策略状态监控中心',
        'page_obj': page_obj,
        'total_count': len(final_list), # 总数是聚合后列表的长度
    }
    return render(request, 'dashboard/trend_following_list.html', context)

@login_required
def fav_trend_following_list(request):
    """
    【V3.2 页面适配增强版】
    - 核心升级: 彻底重构数据聚合逻辑，以传递给模板一个包含完整信息的、结构化的字典列表。
    - 数据结构: 每个列表项现在是一个字典，清晰地包含 stock对象、buy_info、sell_info、active_playbooks等，解决了信息丢失问题。
    - 前后端同步: 此版本与 fav_trend_following_list.html (V3.2) 完全匹配，确保所有字段都能正确显示。
    """
    user_dao = UserDAO()
    
    # 1. 获取用户自选股代码列表
    user_favorites = async_to_sync(user_dao.get_user_favorites)(request.user.id)
    fav_codes = [fav.stock.stock_code for fav in user_favorites if fav.stock]
    
    if not fav_codes:
        return render(request, 'dashboard/fav_trend_following_list.html', {
            'page_title': '自选股持仓监控',
            'page_obj': None,
            'total_count': 0,
        })

    # 2. 获取所有自选股相关的【所有策略状态】
    state_list_qs = TrendFollowStrategyState.objects.filter(
        stock__stock_code__in=fav_codes
    ).select_related('stock').order_by('stock__stock_code', '-latest_trade_time')

    # ▼▼▼【代码修改】: 核心修改部分，聚合时保留详细信息 ▼▼▼
    # 3. 使用字典按股票聚合状态，保留完整的状态对象
    stock_summary = {}
    for state in state_list_qs:
        stock_code = state.stock.stock_code
        if stock_code not in stock_summary:
            stock_summary[stock_code] = {
                'stock': state.stock,
                'buy_state': None,
                'sell_state': None,
                'latest_state': state, # 默认最新的就是第一个
                'playbooks': set()
            }
        
        summary = stock_summary[stock_code]
        
        # 更新最新的买入状态
        if state.last_buy_time and (not summary['buy_state'] or state.last_buy_time > summary['buy_state'].last_buy_time):
            summary['buy_state'] = state
        
        # 更新最新的卖出状态
        if state.last_sell_time and (not summary['sell_state'] or state.last_sell_time > summary['sell_state'].last_sell_time):
            summary['sell_state'] = state
            
        # 更新最新的状态（用于获取分数）
        if state.latest_trade_time > summary['latest_state'].latest_trade_time:
            summary['latest_state'] = state

        # 合并所有剧本
        if state.active_playbooks:
            summary['playbooks'].update(state.active_playbooks)

    # 4. 基于聚合后的摘要信息，构建最终的、结构化的列表
    processed_list = []
    for summary in stock_summary.values():
        buy_state = summary['buy_state']
        sell_state = summary['sell_state']
        latest_state = summary['latest_state']

        item = {
            'stock': summary['stock'],
            'buy_info': None,
            'sell_info': None,
            'latest_trade_time': latest_state.latest_trade_time,
            'latest_score': latest_state.latest_score,
            'active_playbooks': sorted(list(summary['playbooks'])),
            'swing_status': '等待建仓',
            'status_class': 'status-wait',
            'sort_priority': 3
        }

        if buy_state:
            item['buy_info'] = {
                'time': buy_state.last_buy_time,
                'strategy_name': buy_state.strategy_name,
                'time_level': buy_state.time_level
            }
            # 默认是持仓中
            item['swing_status'] = '持仓观察'
            item['status_class'] = 'status-holding'
            item['sort_priority'] = 2

            if sell_state and sell_state.last_sell_time > buy_state.last_buy_time:
                item['sell_info'] = {
                    'time': sell_state.last_sell_time,
                    'strategy_name': sell_state.strategy_name,
                    'time_level': sell_state.time_level
                }
                # 出现更新的卖出信号，变为预警
                item['swing_status'] = '止盈预警'
                item['status_class'] = 'status-alert'
                item['sort_priority'] = 1
        
        processed_list.append(item)
    # ▲▲▲【代码修改】: 结束 ▲▲▲

    # 5. 根据我们计算的优先级和时间进行排序
    processed_list.sort(key=lambda x: (x['sort_priority'], x['latest_trade_time'] is None, x['latest_trade_time']), reverse=False)

    # 6. 分页处理
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













