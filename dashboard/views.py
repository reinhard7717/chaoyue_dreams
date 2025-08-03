# dashboard/views.py
import asyncio
import json
from django.utils import timezone
from datetime import datetime, time, timedelta
import functools
import operator
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
from stock_models.index import TradeCalendar # 导入模型
from utils.config_loader import load_strategy_config
from stock_models.stock_analytics import PositionTracker, TradingSignal, DailyPositionSnapshot, Playbook, SignalPlaybookDetail
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
    【V405.8 生产最终版】
    - 核心修复: 修正了筛选逻辑，确保其在分页前正确应用。
    - 清理: 移除了所有调试打印信息。
    """

    # 步骤1: 获取最近3个交易日
    latest_trade_day_obj = TradeCalendar.objects.filter(
        is_open=True,
        cal_date__lte=timezone.now().date()
    ).order_by('-cal_date').first()

    if not latest_trade_day_obj:
        latest_buy_signals = TradingSignal.objects.none()
    else:
        reference_date = latest_trade_day_obj.cal_date
        latest_3_trade_dates = TradeCalendar.get_latest_n_trade_dates(n=3, reference_date=reference_date)
        
        if not latest_3_trade_dates:
            latest_buy_signals = TradingSignal.objects.none()
        else:
            # 构建时区感知的查询范围
            query_conditions = []
            tz = timezone.get_current_timezone()
            for trade_date in latest_3_trade_dates:
                start_of_day = timezone.make_aware(datetime.combine(trade_date, time.min), tz)
                end_of_day = timezone.make_aware(datetime.combine(trade_date + timedelta(days=1), time.min), tz)
                query_conditions.append(Q(trade_time__gte=start_of_day, trade_time__lt=end_of_day))
            combined_query = functools.reduce(operator.or_, query_conditions)

            # 步骤2: 从数据库获取所有符合日期条件的信号
            latest_buy_signals = TradingSignal.objects.filter(
                combined_query,
                signal_type='BUY',
                timeframe='D'
            ).select_related('stock').prefetch_related(
                Prefetch('signalplaybookdetail_set', queryset=SignalPlaybookDetail.objects.select_related('playbook'))
            ).order_by('-trade_time', '-entry_score')

    # 步骤3: 在内存中处理数据，并聚合剧本信息
    all_logs_in_memory = []
    all_playbook_objects = set()
    for signal in latest_buy_signals:
        active_playbooks = []
        if hasattr(signal, 'signalplaybookdetail_set'):
            for detail in signal.signalplaybookdetail_set.all():
                if detail.playbook:
                    active_playbooks.append(detail.playbook)
                    all_playbook_objects.add(detail.playbook)
        
        active_playbooks.sort(key=lambda p: get_playbook_priority(p.cn_name or p.name))
        all_logs_in_memory.append({
            'log_id': signal.id,
            'stock': signal.stock,
            'latest_trade_time': signal.trade_time,
            'latest_score': signal.entry_score,
            'active_playbooks': active_playbooks,
            'strategy_name': signal.strategy_name,
        })

    # 步骤4: 从内存数据中提取唯一剧本，用于筛选器
    unique_playbooks = sorted(list(all_playbook_objects), key=lambda p: get_playbook_priority(p.cn_name or p.name))

    # --- 代码修改开始 ---
    # [修改原因] 修复筛选逻辑
    # 步骤5: 在内存中根据请求参数进行剧本筛选
    selected_playbook_ids_str = request.GET.getlist('playbooks')
    final_filtered_logs = all_logs_in_memory # 默认是全部记录

    if selected_playbook_ids_str:
        # 将URL参数中的字符串ID转换为整数集合，用于高效查找
        selected_ids_int = {int(pid) for pid in selected_playbook_ids_str}
        
        # 应用筛选逻辑
        final_filtered_logs = [
            log for log in all_logs_in_memory
            # 检查当前日志的剧本ID集合是否是所选ID集合的超集
            if selected_ids_int.issubset({p.id for p in log['active_playbooks']})
        ]
    # --- 代码修改结束 ---

    # 步骤6: 对筛选后的结果进行分页
    paginator = Paginator(final_filtered_logs, 25)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # 步骤7: 准备上下文
    context = {
        'page_title': '策略状态监控中心 (近3个交易日买入)',
        'items_for_display': page_obj.object_list, # 传递分页后的列表
        'page_obj': page_obj,
        'total_count': paginator.count,
        'all_playbooks': unique_playbooks,
        # 将选中的ID（整数形式）传回模板，用于高亮显示
        'selected_playbooks': [int(pid) for pid in selected_playbook_ids_str],
    }
    return render(request, 'dashboard/trend_following_list.html', context)

@login_required
def fav_trend_following_list(request):
    """
    【V406.0 终极模型重构版】
    - 核心重构: 彻底废弃 FavoriteStockTracker，完全基于 PositionTracker 和 DailyPositionSnapshot 新模型。
    - 性能优化: 使用 Prefetch 高效获取每个持仓的最新快照。
    - 逻辑适配: 视图和模板的数据流与新模型完全对齐。
    """
    print("--- [View] 开始渲染自选股监控 (fav_trend_following_list) V406.0 终极模型版 ---")
    
    # --- 代码修改开始 ---
    # [修改原因] 彻底切换到 PositionTracker 新模型

    # 步骤1: 准备一个 Prefetch 对象，用于高效获取每个持仓的“最新”快照
    # 我们只取每个 position 的 snapshots 关系中，按日期倒序的第一个
    prefetch_latest_snapshot = Prefetch(
        'snapshots',
        queryset=DailyPositionSnapshot.objects.order_by('-snapshot_date'),
        to_attr='latest_snapshot_list' # 将结果存入一个自定义属性
    )

    # 步骤2: 查询用户的所有持仓 PositionTracker
    base_queryset = PositionTracker.objects.filter(
        user=request.user
    ).select_related(
        'stock', 
        'entry_signal' # 预加载建仓信号
    ).prefetch_related(
        prefetch_latest_snapshot # 应用上面定义的 prefetch
    )

    # 步骤3: 状态筛选 (现在使用 PositionTracker.Status 枚举)
    status_filter = request.GET.get('status', 'holding')
    if status_filter == 'holding':
        queryset = base_queryset.filter(status=PositionTracker.Status.HOLDING)
        page_title = '自选股持仓监控'
    elif status_filter == 'sold':
        queryset = base_queryset.filter(status=PositionTracker.Status.SOLD)
        page_title = '自选股历史平仓'
    else:
        page_title = '全部自选追踪' 
        queryset = base_queryset

    # 步骤4: 排序
    if status_filter == 'sold':
        ordered_queryset = queryset.order_by('-exit_date')
    else:
        # 对于持仓股，可以按建仓日期排序
        ordered_queryset = queryset.order_by('-entry_date')

    # 步骤5: 分页
    paginator = Paginator(ordered_queryset, 25)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # 步骤6: 准备最终上下文并渲染
    context = {
        'page_title': page_title,
        'page_obj': page_obj, # 直接传递分页后的 PositionTracker 对象列表
        'total_count': paginator.count,
        'status_filter': status_filter,
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
        【V406.0 终极模型重构版】
        - 核心重构: 创建自选时，不再创建 FavoriteStockTracker，而是创建 PositionTracker。
        """
        user = request.user
        
        # --- 代码修改开始 ---
        # [修改原因] 适配新的 TradingSignal 和 PositionTracker 模型
        signal_id = request.data.get('signal_id')
        if not signal_id:
            return Response({'detail': '必须提供一个有效的建仓信号ID。'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            entry_signal = TradingSignal.objects.select_related('stock').get(
                id=signal_id, 
                signal_type=TradingSignal.SignalType.BUY
            )
        except TradingSignal.DoesNotExist:
            return Response({'detail': '指定的建仓信号不存在或无效。'}, status=status.HTTP_404_NOT_FOUND)

        stock = entry_signal.stock

        # 步骤1: 创建或获取 FavoriteStock (这个模型仍然用于简单的收藏夹列表)
        favorite, _ = FavoriteStock.objects.get_or_create(user=user, stock=stock)

        # 步骤2: 创建核心的 PositionTracker 记录
        # 使用 get_or_create 避免重复创建
        tracker, created = PositionTracker.objects.get_or_create(
            user=user, stock=stock, entry_signal=entry_signal,
            defaults={
                'status': PositionTracker.Status.HOLDING, # 使用正确的枚举
                'entry_price': entry_signal.close_price,
                'entry_date': entry_signal.trade_time,
            }
        )
        
        if not created:
            # 如果已存在，可以考虑更新其状态为 HOLDING (如果之前是 SOLD)
            tracker.status = PositionTracker.Status.HOLDING
            tracker.exit_signal = None
            tracker.exit_date = None
            tracker.exit_price = None
            tracker.save()

        # --- 代码修改结束 ---

        # WebSocket 和 Celery 任务逻辑保持不变
        websocket_payload = {
            'id': favorite.id,
            'code': stock.stock_code,
            'name': stock.stock_name,
            # ... (其他字段) ...
        }
        send_update_to_user_sync(
            user_id=user.id,
            sub_type='favorite_added_with_data',
            payload=websocket_payload
        )
        # ... (Celery 任务触发逻辑) ...

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
