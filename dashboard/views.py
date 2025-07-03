# dashboard/views.py
import asyncio
import json
import re
from asgiref.sync import async_to_sync
from django.db.models import Max, F, Q
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
    【V2.3 逻辑重构版】日线趋势跟踪列表视图
    - 数据源为 TrendFollowStrategyState 摘要模型。
    - (V2.3 新增) 重构筛选逻辑，通过比较 last_buy_time 和 last_sell_time 来判断持仓状态，不再依赖不存在的 'state' 字段。
    """
    strategy_name = 'multi_timeframe_collaboration' 

    # 定义持仓状态的查询条件：
    # 1. last_buy_time 必须不为空 (有过买入记录)
    # 2. 并且满足以下任一条件：
    #    a. last_sell_time 为空 (从未卖出)
    #    b. last_buy_time > last_sell_time (最近一次操作是买入)
    held_status_query = Q(last_buy_time__isnull=False) & (
        Q(last_sell_time__isnull=True) | Q(last_buy_time__gt=F('last_sell_time'))
    )

    # 应用新的查询逻辑，并修正字段名 time_level
    state_list = TrendFollowStrategyState.objects.filter(
        held_status_query,
        # time_level='D'
    ).select_related('stock').order_by('-latest_trade_time')

    paginator = Paginator(state_list, 25)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_title': '策略状态监控中心 (当前持仓)',
        'page_obj': page_obj,
        'total_count': paginator.count,
    }
    return render(request, 'dashboard/trend_following_list.html', context)

@login_required
def fav_trend_following_list(request):
    """
    【V3.1 聚合修复版】
    - 核心修复: 解决同一股票因多策略状态而重复显示的问题。
    - 数据聚合: 在后端按股票代码对所有策略状态进行分组，并计算出每只股票唯一的“最终买入时间”和“最终卖出时间”。
    - 唯一展示: 确保每只自选股在列表中只出现一次，展示其最核心的、聚合后的持仓状态。
    - 排序优化: 排序逻辑现在基于聚合后的最终状态，更加准确可靠。
    """
    user_dao = UserDAO()
    
    # 1. 获取用户自选股代码列表
    user_favorites = async_to_sync(user_dao.get_user_favorites)(request.user.id)
    fav_codes = [fav.stock.stock_code for fav in user_favorites if fav.stock]
    
    if not fav_codes:
        # 如果没有自选股，直接渲染空页面
        return render(request, 'dashboard/fav_trend_following_list.html', {
            'page_title': '自选股持仓监控',
            'page_obj': None,
            'total_count': 0,
        })

    # 2. 获取所有自选股相关的【所有策略状态】
    state_list_qs = TrendFollowStrategyState.objects.filter(
        stock__stock_code__in=fav_codes
    ).select_related('stock').order_by('stock__stock_code', '-latest_trade_time')

    # ▼▼▼ 核心修改部分，聚合处理状态 ▼▼▼
    # 3. 使用字典按股票聚合状态，确保每只股票只处理一次
    stock_summary = {}
    for state in state_list_qs:
        stock_code = state.stock.stock_code
        if stock_code not in stock_summary:
            # 首次遇到该股票，初始化其摘要信息
            stock_summary[stock_code] = {
                'stock_obj': state.stock, # 保存完整的stock对象
                'latest_buy_time': state.last_buy_time,
                'latest_sell_time': state.last_sell_time,
                'latest_trade_time': state.latest_trade_time,
            }
        else:
            # 如果已存在，则更新为最新的时间
            summary = stock_summary[stock_code]
            if state.last_buy_time and (not summary['latest_buy_time'] or state.last_buy_time > summary['latest_buy_time']):
                summary['latest_buy_time'] = state.last_buy_time
            
            if state.last_sell_time and (not summary['latest_sell_time'] or state.last_sell_time > summary['latest_sell_time']):
                summary['latest_sell_time'] = state.last_sell_time

            if state.latest_trade_time and (not summary['latest_trade_time'] or state.latest_trade_time > summary['latest_trade_time']):
                summary['latest_trade_time'] = state.latest_trade_time

    # 4. 基于聚合后的摘要信息，计算最终的“持仓状态”
    processed_list = []
    for summary in stock_summary.values():
        # 将聚合信息附加到stock对象上，方便模板和排序使用
        final_state_obj = summary['stock_obj']
        final_state_obj.last_buy_time = summary['latest_buy_time']
        final_state_obj.last_sell_time = summary['latest_sell_time']
        final_state_obj.latest_trade_time = summary['latest_trade_time']

        # 默认状态
        final_state_obj.swing_status = '等待建仓'
        final_state_obj.status_class = 'status-wait'
        final_state_obj.sort_priority = 3 # 排序优先级，数字越小越靠前

        if final_state_obj.last_buy_time:
            if final_state_obj.last_sell_time and final_state_obj.last_sell_time > final_state_obj.last_buy_time:
                # 情况1: 出现止盈信号
                final_state_obj.swing_status = '止盈预警'
                final_state_obj.status_class = 'status-alert'
                final_state_obj.sort_priority = 1 # 最高优先级
            else:
                # 情况2: 正常持仓中
                final_state_obj.swing_status = '持仓观察'
                final_state_obj.status_class = 'status-holding'
                final_state_obj.sort_priority = 2
        
        processed_list.append(final_state_obj)

    # 5. 根据我们计算的优先级和时间进行排序
    # 规则: 止盈预警 > 持仓观察 > 等待建仓。在同等优先级内，按最新交易时间排序。
    processed_list.sort(key=lambda x: (x.sort_priority, x.latest_trade_time is None, x.latest_trade_time), reverse=False)

    # 6. 分页处理 (分页器处理的是聚合后的唯一列表)
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













