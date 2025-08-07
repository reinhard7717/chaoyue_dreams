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
from stock_models.stock_analytics import PositionTracker, TradingSignal, DailyPositionSnapshot, Playbook, SignalPlaybookDetail, Transaction
from stock_models.stock_basic import StockInfo
from utils.cache_manager import CacheManager
from utils.cash_key import IntradayEngineCashKey
from users.models import FavoriteStock
from utils.websockets import send_update_to_user_sync
from .serializers import StockInfoSerializer, FavoriteStockSerializer, TransactionSerializer
from services.transaction_service import TransactionService
from utils.task_helpers import with_cache_manager_for_views
from django.db import transaction as db_transaction
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
    【V5.3 - 深度探针调试版】
    - 核心修改: 在数据处理循环中加入了详细的 print() 探针，
                以诊断“最新动态追踪”无数据的问题。
    """
    # “两步查询法”获取 fav_id (不变)
    base_queryset = PositionTracker.objects.filter(
        user=request.user
    ).select_related(
        'stock'
    ).prefetch_related(
        Prefetch(
            'snapshots',
            queryset=DailyPositionSnapshot.objects.select_related('daily_score').order_by('-snapshot_date'),
            to_attr='latest_snapshot_list'
        )
    )
    status_filter = request.GET.get('status', 'holding')
    if status_filter == 'holding':
        queryset = base_queryset.filter(status=PositionTracker.Status.HOLDING)
        page_title = '自选股持仓监控'
    elif status_filter == 'sold':
        queryset = base_queryset.filter(status=PositionTracker.Status.WATCHING, current_quantity=0)
        page_title = '自选股历史平仓'
    else:
        page_title = '全部自选追踪'
        queryset = base_queryset
    ordered_queryset = queryset.order_by('-updated_at')
    stock_ids = [tracker.stock_id for tracker in ordered_queryset]
    fav_id_map = {}
    if stock_ids:
        favorite_stocks = FavoriteStock.objects.filter(user=request.user, stock_id__in=stock_ids).values('stock_id', 'id')
        fav_id_map = {item['stock_id']: item['id'] for item in favorite_stocks}

    # --- 核心探针区域 ---
    print("\n" + "="*30 + " 开始处理视图数据 " + "="*30)
    trackers_for_display = []
    for tracker in ordered_queryset:
        print(f"\n--- [视图探针] 正在处理 Tracker ID: {tracker.id} ({tracker.stock.stock_code}) ---")

        # 探针 1: 检查 prefetch 的原始结果
        snapshot_list = getattr(tracker, 'latest_snapshot_list', [])
        print(f"[视图探针 1] Prefetch 到的快照列表 (latest_snapshot_list) 包含 {len(snapshot_list)} 条记录。")
        if snapshot_list:
            # 只打印最新的几条，防止刷屏
            for i, snap in enumerate(snapshot_list[:3]):
                print(f"  - 快照 {i+1}: 日期={snap.snapshot_date}, 收盘价={snap.close_price}, 关联分数ID={getattr(snap.daily_score, 'id', '无')}")

        # 探针 2: 检查我们提取最新快照的逻辑
        latest_snapshot = snapshot_list[0] if snapshot_list else None
        print(f"[视图探针 2] 提取出的 'latest_snapshot' 是否存在: {'是' if latest_snapshot else '否'}")

        # 探针 3: 检查从最新快照中提取每日分数的逻辑
        latest_daily_score = latest_snapshot.daily_score if latest_snapshot else None
        print(f"[视图探针 3] 提取出的 'latest_daily_score' 是否存在: {'是' if latest_daily_score else '否'}")
        if latest_daily_score:
            print(f"  - 分数详情: 进攻分={latest_daily_score.offensive_score}, 风险分={latest_daily_score.risk_score}")

        profit_loss, profit_loss_pct = None, None
        if tracker.status == PositionTracker.Status.HOLDING and latest_snapshot:
            profit_loss = latest_snapshot.profit_loss
            profit_loss_pct = latest_snapshot.profit_loss_pct
        
        # 探针 4: 检查最终准备发送到模板的数据包
        tracker_data = {
            'tracker': tracker,
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss_pct,
            'latest_snapshot': latest_snapshot,
            'latest_daily_score': latest_daily_score,
            'fav_id': fav_id_map.get(tracker.stock_id)
        }
        print(f"[视图探针 4] 最终为模板准备的数据包中, 'latest_daily_score' 是否有值: {'是' if tracker_data['latest_daily_score'] else '否'}")
        trackers_for_display.append(tracker_data)
    
    print("\n" + "="*30 + " 视图数据处理完毕 " + "="*30 + "\n")
    # --- 探针区域结束 ---

    paginator = Paginator(trackers_for_display, 25)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

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
        【V5.0 交易流水版】
        - 核心修改: 创建 PositionTracker 和第一笔 Transaction，然后异步触发快照重建任务。
        """
        user = request.user
        stock_code = request.data.get('stock_code')
        entry_price_str = request.data.get('entry_price')
        entry_date_str = request.data.get('entry_date')
        quantity_str = request.data.get('quantity')

        # 参数校验
        if not all([stock_code, entry_price_str, entry_date_str, quantity_str]):
            return Response({'detail': '必须提供 stock_code, entry_price, entry_date 和 quantity。'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            stock = StockInfo.objects.get(stock_code=stock_code)
            entry_price = Decimal(entry_price_str)
            # 注意：前端传来的可能是 'YYYY-MM-DD'，需要转为带时区的datetime
            entry_date = timezone.make_aware(datetime.strptime(entry_date_str, '%Y-%m-%d'))
            quantity = int(quantity_str)
            if quantity <= 0:
                raise ValueError("数量必须为正数")
        except (StockInfo.DoesNotExist, ValueError, TypeError) as e:
            return Response({'detail': f'参数无效: {e}'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # 使用数据库事务确保数据一致性
            with db_transaction.atomic():
                # 1. 创建或获取持仓追踪器 (现在是每个用户/股票唯一的)
                tracker, _ = PositionTracker.objects.update_or_create(
                    user=user, stock=stock,
                    defaults={'status': PositionTracker.Status.HOLDING}
                )

                # 2. 创建第一笔买入交易流水
                Transaction.objects.create(
                    tracker=tracker,
                    transaction_type=Transaction.TransactionType.BUY,
                    quantity=quantity,
                    price=entry_price,
                    transaction_date=entry_date
                )

                # 3. 更新 Tracker 的平均成本和数量 (对于第一笔交易很简单)
                tracker.average_cost = entry_price
                tracker.current_quantity = quantity
                tracker.save()

            # 4. 【核心】异步触发快照重建任务
            rebuild_snapshots_for_tracker_task.delay(tracker.id)

            # (可选) 也可以在这里创建 FavoriteStock 记录
            FavoriteStock.objects.get_or_create(user=user, stock=stock)

            return Response({'detail': '持仓已创建，历史快照正在后台生成中...'}, status=status.HTTP_201_CREATED)

        except Exception as e:
            logger.error(f"创建持仓时发生错误: {e}", exc_info=True)
            return Response({'detail': '创建持仓时发生内部错误。'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

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

class TransactionViewSet(viewsets.ModelViewSet):
    """
    API endpoint for managing Transactions.
    """
    queryset = Transaction.objects.all()
    serializer_class = TransactionSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        """
        过滤，确保用户只能看到自己的交易流水，并支持按tracker_id查询。
        """
        queryset = Transaction.objects.filter(tracker__user=self.request.user)
        tracker_id = self.request.query_params.get('tracker_id')
        if tracker_id:
            queryset = queryset.filter(tracker_id=tracker_id)
        return queryset.order_by('transaction_date')

    def perform_create(self, serializer):
        """
        创建交易后，调用服务更新持仓状态并重建快照。
        """
        transaction = serializer.save()
        # 调用核心服务
        TransactionService.recalculate_tracker_state_and_rebuild_snapshots(transaction.tracker.id)

    def perform_update(self, serializer):
        """
        更新交易后，调用服务更新持仓状态并重建快照。
        """
        transaction = serializer.save()
        # 调用核心服务
        TransactionService.recalculate_tracker_state_and_rebuild_snapshots(transaction.tracker.id)

    def perform_destroy(self, instance):
        """
        删除交易后，调用服务更新持仓状态并重建快照。
        """
        tracker_id = instance.tracker.id
        instance.delete()
        # 调用核心服务
        TransactionService.recalculate_tracker_state_and_rebuild_snapshots(tracker_id)













