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
from django.db.models import OuterRef, Subquery
from django.db.models import Prefetch
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
    【V405.13 筛选逻辑最终版】
    - 核心修复: 恢复并确认了正确的筛选逻辑 `selected_pks_set.issubset(...)`，并确保比较时双方都使用字符串类型的主键。
    """

    # 1. 动态加载配置文件并获取主策略名称
    try:
        unified_config = load_strategy_config('config/trend_follow_strategy.json')
        strategy_info = unified_config.get('strategy_params', {}).get('trend_follow', {}).get('strategy_info', {})
        main_strategy_name = strategy_info.get('name', {}).get('value')
        if not main_strategy_name:
            logger.warning("未能从配置文件中找到主策略名称，可能导致信号列表不准确。")
    except Exception as e:
        main_strategy_name = None
        logger.error(f"加载策略配置失败: {e}", exc_info=True)

    # 2. 获取最新的交易日期范围 (逻辑不变)
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
            query_conditions = []
            tz = timezone.get_current_timezone()
            for trade_date in latest_3_trade_dates:
                start_of_day = timezone.make_aware(datetime.combine(trade_date, time.min), tz)
                end_of_day = timezone.make_aware(datetime.combine(trade_date + timedelta(days=1), time.min), tz)
                query_conditions.append(Q(trade_time__gte=start_of_day, trade_time__lt=end_of_day))
            combined_query = functools.reduce(operator.or_, query_conditions)

            # 3. 构建基础查询，并应用主策略名称过滤器
            base_query = TradingSignal.objects.filter(
                combined_query,
                signal_type='BUY',
                timeframe='D'
            )
            
            # 【核心修复】只查询主策略的信号
            if main_strategy_name:
                base_query = base_query.filter(strategy_name=main_strategy_name)
            
            latest_buy_signals = base_query.select_related('stock').prefetch_related(
                Prefetch('signalplaybookdetail_set', queryset=SignalPlaybookDetail.objects.select_related('playbook'))
            ).order_by('-trade_time', '-entry_score')

    all_logs_in_memory = []
    all_playbook_objects = set()
    for signal in latest_buy_signals:
        active_playbooks = []
        if hasattr(signal, 'signalplaybookdetail_set'):
            for detail in signal.signalplaybookdetail_set.all():
                if detail.playbook and detail.playbook.pk is not None:
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

    unique_playbooks = sorted(list(all_playbook_objects), key=lambda p: get_playbook_priority(p.cn_name or p.name))

    # --- 代码修改开始 ---
    # [修改原因] 恢复并确认正确的筛选逻辑，并确保类型一致。
    selected_playbooks_pks = request.GET.getlist('playbooks')
    final_filtered_logs = all_logs_in_memory

    if selected_playbooks_pks:
        # 1. 将URL传入的筛选条件（字符串列表）转换为集合，用于高效查找
        selected_pks_set = set(selected_playbooks_pks)
        
        # 2. 应用筛选
        final_filtered_logs = [
            log for log in all_logs_in_memory
            # 3. 对每条记录，获取其激活剧本的主键集合（确保转换为字符串）
            # 4. 判断筛选条件集合是否是当前记录剧本集合的“子集”
            if selected_pks_set.issubset({str(p.pk) for p in log['active_playbooks']})
        ]
    # --- 代码修改结束 ---

    paginator = Paginator(final_filtered_logs, 25)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_title': '策略状态监控中心 (近3个交易日买入)',
        'items_for_display': page_obj.object_list,
        'page_obj': page_obj,
        'total_count': paginator.count,
        'all_playbooks': unique_playbooks,
        'selected_playbooks': selected_playbooks_pks,
    }
    return render(request, 'dashboard/trend_following_list.html', context)

@login_required
def fav_trend_following_list(request):
    """
    【V407.2 MySQL兼容性修复版】
    - 核心修复: 解决了因 MySQL 版本不支持 "LIMIT & IN subquery" 导致的 NotSupportedError。
    - 策略变更:
        1. 移除了在 Prefetch 中使用 Subquery 的复杂查询。
        2. 改为预加载每个 Tracker 的 *所有* 快照 (按日期降序排列)。
        3. 在 Python 循环中，通过获取列表的第一个元素 (`.latest_snapshot_list[0]`) 来安全地找到最新快照。
        4. 这种方法兼容性更强，虽然可能多获取少量数据，但对于持仓管理页面完全可以接受。
    """
    # --- 代码修改开始 ---
    # [修改原因] 绕过 MySQL 的 "LIMIT & IN" 子查询限制

    # 步骤 1: 主查询，使用更简单的 Prefetch
    base_queryset = PositionTracker.objects.filter(
        user=request.user
    ).select_related(
        'stock', 
        'entry_signal', 
        'exit_signal'
    ).prefetch_related(
        # 预加载建仓信号的战法 (逻辑不变)
        Prefetch('entry_signal__playbooks', queryset=Playbook.objects.all()),
        # 【核心修改】预加载 *所有* 快照，并按日期降序排列
        # 这样，最新的快照将永远是列表的第一个元素
        Prefetch(
            'snapshots',
            queryset=DailyPositionSnapshot.objects.select_related('daily_score').order_by('-snapshot_date'),
            to_attr='latest_snapshot_list' # 将结果存入自定义属性
        )
    )
    # --- 代码修改结束 ---

    # 步骤 2: 状态筛选 (逻辑不变)
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

    # 步骤 3: 排序 (逻辑不变)
    if status_filter == 'sold':
        ordered_queryset = queryset.order_by('-exit_date')
    else:
        ordered_queryset = queryset.order_by('-entry_date')

    # 步骤 4: 在Python中处理数据
    trackers_for_display = []
    for tracker in ordered_queryset:
        # --- 代码修改开始 ---
        # [修改原因] 从预加载的列表中安全地获取最新快照
        # 由于我们在 Prefetch 中已经按日期降序排列，第一个元素就是最新的
        latest_snapshot = tracker.latest_snapshot_list[0] if hasattr(tracker, 'latest_snapshot_list') and tracker.latest_snapshot_list else None
        # --- 代码修改结束 ---
        
        # 计算盈亏 (P/L)
        if tracker.status == PositionTracker.Status.HOLDING and latest_snapshot:
            profit_loss = latest_snapshot.profit_loss
            profit_loss_pct = latest_snapshot.profit_loss_pct
        elif tracker.status == PositionTracker.Status.SOLD and tracker.exit_price is not None and tracker.entry_price is not None:
            profit_loss = (tracker.exit_price - tracker.entry_price)
            profit_loss_pct = ((tracker.exit_price / tracker.entry_price) - 1) * 100 if tracker.entry_price > 0 else 0
        else:
            profit_loss = None
            profit_loss_pct = None
        
        # 准备一个字典，包含所有模板需要的数据
        tracker_data = {
            'tracker': tracker,
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss_pct,
            'latest_snapshot': latest_snapshot,
            'latest_daily_score': latest_snapshot.daily_score if latest_snapshot else None
        }
        trackers_for_display.append(tracker_data)

    # 步骤 5: 分页 (逻辑不变)
    paginator = Paginator(trackers_for_display, 25)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # 步骤 6: 准备最终上下文并渲染 (逻辑不变)
    context = {
        'page_title': page_title,
        'page_obj': page_obj,
        'total_count': paginator.count,
        'status_filter': status_filter,
    }
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
        【V406.1 兼容版】
        - 核心修复: 兼容两种添加自选的场景，修复主页添加自选功能。
        """
        user = request.user
        signal_id = request.data.get('signal_id')
        stock_code = request.data.get('stock_code')

        # 场景1: 从策略信号列表添加 (需要 signal_id)
        if signal_id:
            try:
                entry_signal = TradingSignal.objects.select_related('stock').get(
                    id=signal_id, 
                    signal_type=TradingSignal.SignalType.BUY
                )
            except TradingSignal.DoesNotExist:
                return Response({'detail': '指定的建仓信号不存在或无效。'}, status=status.HTTP_404_NOT_FOUND)

            stock = entry_signal.stock
            # 创建或获取 FavoriteStock
            favorite, _ = FavoriteStock.objects.get_or_create(user=user, stock=stock)
            # 创建核心的 PositionTracker 记录
            tracker, created = PositionTracker.objects.get_or_create(
                user=user, stock=stock, entry_signal=entry_signal,
                defaults={
                    'status': PositionTracker.Status.HOLDING,
                    'entry_price': entry_signal.close_price,
                    'entry_date': entry_signal.trade_time,
                }
            )
            if not created:
                tracker.status = PositionTracker.Status.HOLDING
                tracker.exit_signal = None
                tracker.exit_date = None
                tracker.exit_price = None
                tracker.save()
        
        # 场景2: 从主页搜索添加 (只需要 stock_code)
        elif stock_code:
            try:
                stock = StockInfo.objects.get(stock_code=stock_code)
            except StockInfo.DoesNotExist:
                return Response({'detail': f'股票代码 {stock_code} 不存在。'}, status=status.HTTP_404_NOT_FOUND)
            
            # 只创建或获取 FavoriteStock，不创建 PositionTracker
            favorite, created = FavoriteStock.objects.get_or_create(user=user, stock=stock)
            if not created:
                return Response({'detail': '该股票已在您的自选列表中。'}, status=status.HTTP_400_BAD_REQUEST)
        
        # 场景3: 无效请求
        else:
            return Response({'detail': '必须提供 signal_id 或 stock_code。'}, status=status.HTTP_400_BAD_REQUEST)

        # --- 通用的成功后操作 ---
        # WebSocket 推送
        websocket_payload = {
            'id': favorite.id,
            'code': stock.stock_code,
            'name': stock.stock_name,
            # 为了与JS addStockRow 函数兼容，提供基础字段
            "current_price": None,
            "change_percent": None,
            "volume": None,
            "signal": None,
        }
        send_update_to_user_sync(
            user_id=user.id,
            sub_type='favorite_added_with_data',
            payload=websocket_payload
        )
        
        # (可选) 触发Celery任务，例如立即获取一次行情
        # from your_tasks import fetch_realtime_quote_task
        # fetch_realtime_quote_task.delay(stock.stock_code)

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
