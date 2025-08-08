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
from stock_models.stock_analytics import PositionTracker, TradingSignal, DailyPositionSnapshot, Playbook, SignalPlaybookDetail, Transaction, StrategyDailyScore
from stock_models.stock_basic import StockInfo
from stock_models.time_trade import AdvancedChipMetrics
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
    【V5.0 - 日期选择版】
    - 核心修改: 废除“近3天”的固定逻辑，改为默认显示最新交易日的数据。
    - 功能增强: 增加日期选择功能，允许用户查询任意历史交易日的买入信号。
    """
    # 1. 确定要查询的目标日期
    selected_date_str = request.GET.get('date')
    target_date = None

    if selected_date_str:
        try:
            # 尝试从用户输入解析日期
            target_date = datetime.strptime(selected_date_str, '%Y-%m-%d').date()
        except (ValueError, TypeError):
            # 如果格式错误，则忽略，后续会使用最新交易日
            target_date = None
    
    # 如果没有有效的日期输入，则自动获取最新交易日
    if not target_date:
        latest_trade_day_obj = TradeCalendar.objects.filter(
            is_open=True,
            cal_date__lte=timezone.now().date()
        ).order_by('-cal_date').first()
        if latest_trade_day_obj:
            target_date = latest_trade_day_obj.cal_date

    # 2. 如果无法确定目标日期，则不进行查询
    if not target_date:
        latest_buy_signals = TradingSignal.objects.none()
        page_title = '策略状态监控中心 (无可用数据)'
    else:
        # 动态生成页面标题
        page_title = f'策略状态监控中心 ({target_date.strftime("%Y-%m-%d")} 买入信号)'
        
        # 3. 动态加载配置文件并获取主策略名称 (逻辑不变)
        try:
            unified_config = load_strategy_config('config/trend_follow_strategy.json')
            strategy_info = unified_config.get('strategy_params', {}).get('trend_follow', {}).get('strategy_info', {})
            main_strategy_name = strategy_info.get('name', {}).get('value')
        except Exception as e:
            main_strategy_name = None
            logger.error(f"加载策略配置失败: {e}", exc_info=True)

        # 4. 根据目标日期构建查询
        tz = timezone.get_current_timezone()
        start_of_day = timezone.make_aware(datetime.combine(target_date, time.min), tz)
        end_of_day = timezone.make_aware(datetime.combine(target_date, time.max), tz)

        base_query = TradingSignal.objects.filter(
            trade_time__range=(start_of_day, end_of_day),
            signal_type='BUY',
            timeframe='D'
        )
        
        # 如果找到主策略名称，则应用过滤
        if main_strategy_name:
            base_query = base_query.filter(strategy_name=main_strategy_name)
        
        latest_buy_signals = base_query.select_related('stock').prefetch_related(
            Prefetch('signalplaybookdetail_set', queryset=SignalPlaybookDetail.objects.select_related('playbook'))
        ).order_by('-entry_score') # 按分数倒序排列

    # 5. 数据处理与筛选 (逻辑与之前类似，但数据源已变为单日)
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
        
        # 【重要】从信号对象中直接获取收盘价
        all_logs_in_memory.append({
            'log_id': signal.id,
            'stock': signal.stock,
            'latest_trade_time': signal.trade_time,
            'latest_score': signal.entry_score,
            'active_playbooks': active_playbooks,
            'strategy_name': signal.strategy_name,
            'close_price': signal.close_price, # 确保传递收盘价
        })

    unique_playbooks = sorted(list(all_playbook_objects), key=lambda p: get_playbook_priority(p.cn_name or p.name))

    selected_playbooks_pks = request.GET.getlist('playbooks')
    final_filtered_logs = all_logs_in_memory

    if selected_playbooks_pks:
        selected_pks_set = set(selected_playbooks_pks)
        final_filtered_logs = [
            log for log in all_logs_in_memory
            if selected_pks_set.issubset({str(p.pk) for p in log['active_playbooks']})
        ]

    # 6. 分页与上下文准备
    paginator = Paginator(final_filtered_logs, 25)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_title': page_title,
        'items_for_display': page_obj.object_list,
        'page_obj': page_obj,
        'total_count': paginator.count,
        'all_playbooks': unique_playbooks,
        'selected_playbooks': selected_playbooks_pks,
        'selected_date': target_date.strftime('%Y-%m-%d') if target_date else '', # 传递当前选择的日期
    }
    return render(request, 'dashboard/trend_following_list.html', context)

@login_required
def fav_trend_following_list(request):
    """
    【V7.1 - 风险剧本透视版】
    - 核心升级: 集成高级筹码指标(AdvancedChipMetrics)，追踪持仓期间的关键筹码变化。
    - 功能增强: 查询并展示每日分数中的具体风险/离场剧本，为用户提供明确的风险预警。
    """

    def _calculate_score_deltas(current_score, baseline_score):
        if not current_score or not baseline_score: return None
        return {
            'offensive': current_score.offensive_score - baseline_score.offensive_score,
            'positional': current_score.positional_score - baseline_score.positional_score,
            'dynamic': current_score.dynamic_score - baseline_score.dynamic_score,
            'composite': current_score.composite_score - baseline_score.composite_score,
        }

    # --- 步骤1: 预抓取所有需要的数据 ---
    base_queryset = PositionTracker.objects.filter(user=request.user).select_related('stock').prefetch_related(
        Prefetch('snapshots', queryset=DailyPositionSnapshot.objects.select_related('daily_score').order_by('-snapshot_date'), to_attr='latest_snapshot_list'),
        Prefetch('transactions', queryset=Transaction.objects.order_by('transaction_date'), to_attr='sorted_transactions')
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

    # --- 步骤2: 收集所有需要查询的“关键日期” (扩展到筹码指标) ---
    score_lookups = set()
    chip_metrics_lookups = set() # 新增！用于收集筹码指标的查询需求
    trackers_with_key_dates = []

    for tracker in ordered_queryset:
        transactions = getattr(tracker, 'sorted_transactions', [])
        snapshot_list = getattr(tracker, 'latest_snapshot_list', [])
        key_dates = {'initial': None, 'last_buy': None, 'latest': None}
        
        if snapshot_list:
            key_dates['latest'] = snapshot_list[0].snapshot_date
            score_lookups.add((tracker.stock_id, key_dates['latest']))
            chip_metrics_lookups.add((tracker.stock_id, key_dates['latest']))

        if transactions:
            key_dates['initial'] = transactions[0].transaction_date.date()
            score_lookups.add((tracker.stock_id, key_dates['initial']))
            chip_metrics_lookups.add((tracker.stock_id, key_dates['initial']))
            
            last_buy_tx = next((tx for tx in reversed(transactions) if tx.transaction_type == Transaction.TransactionType.BUY), None)
            if last_buy_tx:
                key_dates['last_buy'] = last_buy_tx.transaction_date.date()
                score_lookups.add((tracker.stock_id, key_dates['last_buy']))
                chip_metrics_lookups.add((tracker.stock_id, key_dates['last_buy']))
        
        trackers_with_key_dates.append({'tracker': tracker, 'key_dates': key_dates})

    # --- 步骤3: 一次性查询所有关键数据 (分数 + 筹码) ---
    score_map = {}
    if score_lookups:
        queries = [Q(stock_id=sid, trade_date=td) for sid, td in score_lookups if td]
        if queries:
            all_key_scores = StrategyDailyScore.objects.filter(
                functools.reduce(operator.or_, queries)
            ).prefetch_related('components')
            score_map = {(s.stock_id, s.trade_date): s for s in all_key_scores}

    chip_metrics_map = {} # 新增！用于存储筹码指标的Map
    if chip_metrics_lookups:
        queries = [Q(stock_id=sid, trade_time=td) for sid, td in chip_metrics_lookups if td]
        if queries:
            all_key_chip_metrics = AdvancedChipMetrics.objects.filter(functools.reduce(operator.or_, queries))
            chip_metrics_map = {(cm.stock_id, cm.trade_time): cm for cm in all_key_chip_metrics}

    # --- 步骤4: 组装最终数据，进行精细化计算 ---
    trackers_for_display = []
    for item in trackers_with_key_dates:
        tracker = item['tracker']
        key_dates = item['key_dates']
        
        # 获取分数
        latest_daily_score = score_map.get((tracker.stock_id, key_dates['latest']))
        initial_score = score_map.get((tracker.stock_id, key_dates['initial']))
        last_buy_score = score_map.get((tracker.stock_id, key_dates['last_buy']))
        
        # 获取筹码指标
        latest_chip_metrics = chip_metrics_map.get((tracker.stock_id, key_dates['latest']))
        initial_chip_metrics = chip_metrics_map.get((tracker.stock_id, key_dates['initial']))
        last_buy_chip_metrics = chip_metrics_map.get((tracker.stock_id, key_dates['last_buy']))

        delta_from_initial = _calculate_score_deltas(latest_daily_score, initial_score)
        delta_from_last_buy = _calculate_score_deltas(latest_daily_score, last_buy_score)

        profit_loss, profit_loss_pct = None, None
        if tracker.status == PositionTracker.Status.HOLDING and getattr(tracker, 'latest_snapshot_list', []):
            profit_loss = tracker.latest_snapshot_list[0].profit_loss
            profit_loss_pct = tracker.latest_snapshot_list[0].profit_loss_pct

        risk_playbooks = []
        if latest_daily_score and hasattr(latest_daily_score, 'components'):
            # components 已经被 prefetch，所以这里不会产生新的数据库查询
            all_components = latest_daily_score.components.all()
            risk_playbooks = sorted(
                [comp for comp in all_components if comp.score_type == 'risk'],
                key=lambda c: c.score_value,
                reverse=True
            )
            if risk_playbooks:
                print(f"调试信息: 股票 {tracker.stock.stock_code} 在 {key_dates['latest']} 发现 {len(risk_playbooks)} 个风险剧本。")

        trackers_for_display.append({
            'tracker': tracker,
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss_pct,
            'latest_daily_score': latest_daily_score,
            'delta_from_initial': delta_from_initial,
            'delta_from_last_buy': delta_from_last_buy,
            'initial_score': initial_score, # 传递用于模板显示
            'last_buy_score': last_buy_score, # 传递用于模板显示
            'latest_chip_metrics': latest_chip_metrics,
            'initial_chip_metrics': initial_chip_metrics,
            'last_buy_chip_metrics': last_buy_chip_metrics,
            'risk_playbooks': risk_playbooks,
        })

    stock_ids = [d['tracker'].stock_id for d in trackers_for_display]
    fav_id_map = {}
    if stock_ids:
        favorite_stocks = FavoriteStock.objects.filter(user=request.user, stock_id__in=stock_ids).values('stock_id', 'id')
        fav_id_map = {item['stock_id']: item['id'] for item in favorite_stocks}
    
    for item in trackers_for_display:
        item['fav_id'] = fav_id_map.get(item['tracker'].stock_id)

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
        【V6.0 - 简化版】
        - 核心修改: 移除所有关于价格、日期、数量的参数依赖和交易创建逻辑。
        - 核心职责: 只负责创建 FavoriteStock 和一个初始状态的 PositionTracker。
                     真正的交易录入完全交由用户在前端完成。
        """
        user = request.user
        stock_code = request.data.get('stock_code')

        # 1. 参数校验：现在只需要 stock_code
        if not stock_code:
            return Response({'detail': '必须提供 stock_code。'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            stock = StockInfo.objects.get(stock_code=stock_code)
        except StockInfo.DoesNotExist:
            return Response({'detail': f'股票代码 {stock_code} 不存在。'}, status=status.HTTP_404_NOT_FOUND)

        try:
            # 2. 使用数据库事务确保原子性
            with db_transaction.atomic():
                # 2.1 创建或获取自选股记录 (幂等操作)
                favorite, fav_created = FavoriteStock.objects.get_or_create(user=user, stock=stock)

                # 2.2 创建或获取持仓追踪器 (幂等操作)
                # 这将创建一个 status='WATCHING', quantity=0, average_cost=0 的空追踪器
                tracker, tracker_created = PositionTracker.objects.get_or_create(user=user, stock=stock)

            # 3. 根据操作结果返回友好的提示信息
            if fav_created or tracker_created:
                message = f"股票 {stock.stock_code} 已成功添加至自选，请在“自选股监控”页面管理您的持仓。"
                return Response({'detail': message}, status=status.HTTP_201_CREATED)
            else:
                message = f"股票 {stock.stock_code} 已在您的自选中。"
                return Response({'detail': message}, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"添加自选股 {stock_code} 时发生错误: {e}", exc_info=True)
            return Response({'detail': '添加自选时发生内部错误。'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

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
        tracker = serializer.validated_data['tracker']
        
        # 1. 检查权限：确保操作的 tracker 属于当前用户
        if tracker.user != self.request.user:
            raise PermissionDenied("你没有权限操作此持仓记录。")

        # 2. 保存交易记录
        transaction = serializer.save()
        
        # 3. 调用服务更新持仓成本和数量 (这部分逻辑应在TransactionService中)
        transaction_service = TransactionService()
        transaction_service.update_tracker_after_transaction(transaction)

        # 4. 重建该 tracker 的所有历史快照
        snapshot_service = PositionSnapshotService()
        async_to_sync(snapshot_service.rebuild_snapshots_for_tracker)(tracker.id)
        logger.info(f"为 Tracker ID {tracker.id} 成功重建快照。")

        # 5. 发送WebSocket通知给前端
        send_update_to_user_sync(
            self.request.user.id, 
            'snapshot_rebuilt', 
            {'status': 'success', 'tracker_id': tracker.id}
        )
        logger.info(f"已向用户 {self.request.user.username} 发送快照重建通知。")

    def perform_destroy(self, instance):
        tracker = instance.tracker
        user = self.request.user

        # 1. 检查权限
        if tracker.user != user:
            raise PermissionDenied("你没有权限操作此持仓记录。")

        # 2. 调用服务处理删除逻辑 (包括更新持仓状态)
        transaction_service = TransactionService()
        transaction_service.handle_transaction_deletion(instance)
        
        # 3. 重建快照
        snapshot_service = PositionSnapshotService()
        async_to_sync(snapshot_service.rebuild_snapshots_for_tracker)(tracker.id)
        logger.info(f"因交易删除，为 Tracker ID {tracker.id} 成功重建快照。")

        # 4. 发送WebSocket通知
        send_update_to_user_sync(
            user.id, 
            'snapshot_rebuilt', 
            {'status': 'success', 'tracker_id': tracker.id}
        )
        logger.info(f"已向用户 {user.username} 发送快照重建通知。")

    def perform_update(self, serializer):
        """
        更新交易后，调用服务更新持仓状态并重建快照。
        """
        transaction = serializer.save()
        # 调用核心服务
        TransactionService.recalculate_tracker_state_and_rebuild_snapshots(transaction.tracker.id)












