# dashboard/views.py
import asyncio
import json
from django.shortcuts import get_object_or_404
from django.utils import timezone
from datetime import datetime, time, timedelta
import functools
import operator
from rest_framework.exceptions import PermissionDenied
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
from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO
from stock_models.stock_analytics import PositionTracker, TradingSignal, DailyPositionSnapshot, Playbook, SignalPlaybookDetail, Transaction, StrategyDailyScore
from stock_models.stock_basic import StockInfo
from utils.model_helpers import get_advanced_chip_metrics_model_by_code
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
    # 直接使用由装饰器注入的 cache_manager 实例
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
        'user_favorites': user_favorites, #此行
    }
    return render(request, 'dashboard/home.html', context)

@login_required
def trend_following_list(request):
    """
    【V5.4 · 智能日期定位版】
    - 核心升级: 优化了默认日期的选择逻辑。现在页面在首次加载时，会自动定位到【有信号的最新交易日】，而不是简单地显示最新的交易日。
    - 解决问题: 避免了进入页面后，因默认显示当天（通常无信号）而看到空列表的问题。
    """
    # 1. 确定要查询的目标日期
    selected_date_str = request.GET.get('date')
    target_date = None
    if selected_date_str:
        try:
            target_date = datetime.strptime(selected_date_str, '%Y-%m-%d').date()
        except (ValueError, TypeError):
            target_date = None
    # 优化默认日期的确定逻辑
    if not target_date:
        # 优先从 TradingSignal 表中查找最新的BUY信号的日期
        latest_buy_signal = TradingSignal.objects.filter(
            signal_type='BUY',
            timeframe='D',
            strategy_name='TrendFollow'
        ).order_by('-trade_time').first()
        if latest_buy_signal:
            # 如果找到了信号，则使用该信号的日期作为目标日期
            # 使用 timezone.localtime 确保将UTC时间正确转换为服务器本地时间再取日期
            print(f"调试信息: 找到最新信号，时间为 {latest_buy_signal.trade_time}，将使用其本地日期。")
            target_date = timezone.localtime(latest_buy_signal.trade_time).date()
        else:
            # 如果数据库中没有任何BUY信号（例如系统刚启动），则回退到原来的逻辑：使用最新的交易日
            print("调试信息: 未找到任何BUY信号，回退到使用最新交易日历。")
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
        page_title = f'策略状态监控中心 ({target_date.strftime("%Y-%m-%d")} 买入信号)'
        # 不再需要从配置中读取策略名，直接硬编码
        main_strategy_name = 'TrendFollow'
        tz = timezone.get_current_timezone()
        start_of_day = timezone.make_aware(datetime.combine(target_date, time.min), tz)
        end_of_day = timezone.make_aware(datetime.combine(target_date, time.max), tz)
        # 恢复数据库排序，并增加 strategy_name 过滤器
        latest_buy_signals = TradingSignal.objects.filter(
            trade_time__range=(start_of_day, end_of_day),
            signal_type='BUY',
            timeframe='D',
            strategy_name=main_strategy_name # 核心过滤条件
        ).select_related('stock').prefetch_related(
            Prefetch('signalplaybookdetail_set', queryset=SignalPlaybookDetail.objects.select_related('playbook'))
        ).order_by('-final_score') # 恢复数据库排序
    # 5. 数据处理与筛选 (恢复为简化逻辑)
    all_logs_in_memory = []
    all_playbook_objects = set()
    for signal in latest_buy_signals:
        active_playbooks = []
        if hasattr(signal, 'signalplaybookdetail_set'):
            for detail in signal.signalplaybookdetail_set.all():
                if detail.playbook and detail.playbook.pk is not None:
                    active_playbooks.append(detail.playbook)
                    all_playbook_objects.add(detail.playbook)
        active_playbooks.sort(key=lambda p: p.cn_name or p.name)
        all_logs_in_memory.append({
            'log_id': signal.id,
            'stock': signal.stock,
            'latest_trade_time': signal.trade_time,
            'latest_score': signal.final_score,
            'active_playbooks': active_playbooks,
            'strategy_name': signal.strategy_name,
            'close_price': signal.close_price,
        })
    unique_playbooks = sorted(list(all_playbook_objects), key=lambda p: p.cn_name or p.name)
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
        'selected_date': target_date.strftime('%Y-%m-%d') if target_date else '',
    }
    return render(request, 'dashboard/trend_following_list.html', context)

# 为“先知”建立独立的圣殿
@login_required
def prophet_signal_list(request):
    """
    【V1.1 · 智能日期定位版】渲染“先知信号监控中心”页面。
    - 核心升级: 优化了默认日期的选择逻辑。现在页面在首次加载时，会自动定位到【有“先知”信号的最新交易日】。
    - 解决问题: 避免了进入页面后看到空列表的问题。
    """
    # 1. 确定要查询的目标日期
    selected_date_str = request.GET.get('date')
    target_date = None
    if selected_date_str:
        try:
            target_date = datetime.strptime(selected_date_str, '%Y-%m-%d').date()
        except (ValueError, TypeError):
            target_date = None
    # 优化默认日期的确定逻辑
    if not target_date:
        # 优先从 TradingSignal 表中查找最新的“先知”BUY信号的日期
        prophet_strategy_name = 'ProphetSignal'
        latest_prophet_signal = TradingSignal.objects.filter(
            signal_type='BUY',
            timeframe='D',
            strategy_name=prophet_strategy_name # 核心过滤条件
        ).order_by('-trade_time').first()
        if latest_prophet_signal:
            # 如果找到了信号，则使用该信号的日期作为目标日期
            print(f"调试信息: 找到最新“先知”信号，时间为 {latest_prophet_signal.trade_time}，将使用其本地日期。")
            target_date = timezone.localtime(latest_prophet_signal.trade_time).date()
        else:
            # 如果数据库中没有任何“先知”信号，则回退到原来的逻辑：使用最新的交易日
            print("调试信息: 未找到任何“先知”信号，回退到使用最新交易日历。")
            latest_trade_day_obj = TradeCalendar.objects.filter(
                is_open=True,
                cal_date__lte=timezone.now().date()
            ).order_by('-cal_date').first()
            if latest_trade_day_obj:
                target_date = latest_trade_day_obj.cal_date
    # 2. 查询“先知”信号
    if not target_date:
        prophet_signals = TradingSignal.objects.none()
        page_title = '先知信号监控中心 (无可用数据)'
    else:
        page_title = f'先知信号监控中心 ({target_date.strftime("%Y-%m-%d")} 神谕)'
        prophet_strategy_name = 'ProphetSignal'
        tz = timezone.get_current_timezone()
        start_of_day = timezone.make_aware(datetime.combine(target_date, time.min), tz)
        end_of_day = timezone.make_aware(datetime.combine(target_date, time.max), tz)
        prophet_signals = TradingSignal.objects.filter(
            trade_time__range=(start_of_day, end_of_day),
            signal_type='BUY',
            timeframe='D',
            strategy_name=prophet_strategy_name # 核心过滤条件
        ).select_related('stock').order_by('-final_score')
    # 3. 分页与上下文准备
    paginator = Paginator(prophet_signals, 25)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    context = {
        'page_title': page_title,
        'page_obj': page_obj,
        'total_count': paginator.count,
        'selected_date': target_date.strftime('%Y-%m-%d') if target_date else '',
    }
    # 渲染一个新的、专用的模板
    return render(request, 'dashboard/prophet_signal_list.html', context)

@login_required
def fav_trend_following_list(request):
    """
    【V7.2 - 分表适配版】
    - 核心升级: 集成高级筹码指标(AdvancedChipMetrics)，追踪持仓期间的关键筹码变化。
    - 功能增强: 查询并展示每日分数中的具体风险/离场剧本，为用户提供明确的风险预警。
    - 技术改造: 适配 AdvancedChipMetrics 模型分表，实现跨表动态查询。
    """
    def _calculate_score_deltas(current_score, baseline_score):
        if not current_score or not baseline_score: return None
        return {
            'offensive': current_score.offensive_score - baseline_score.offensive_score,
            'positional': current_score.positional_score - baseline_score.positional_score,
            'dynamic': current_score.dynamic_score - baseline_score.dynamic_score,
            'composite': current_score.composite_score - baseline_score.composite_score,
        }
    # --- 步骤1: 预抓取所有需要的数据  ---
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
    # --- 步骤2: 收集所有需要查询的“关键日期” ---
    score_lookups = set()
    chip_metrics_lookups = set() # 用于收集筹码指标的查询需求
    trackers_with_key_dates = []
    for tracker in ordered_queryset:
        transactions = getattr(tracker, 'sorted_transactions', [])
        snapshot_list = getattr(tracker, 'latest_snapshot_list', [])
        key_dates = {'initial': None, 'last_buy': None, 'latest': None}
        # 获取股票代码，用于后续动态选择模型
        stock_code = tracker.stock.stock_code
        if snapshot_list:
            key_dates['latest'] = snapshot_list[0].snapshot_date
            score_lookups.add((tracker.stock_id, key_dates['latest']))
            # 收集筹码指标查询需求时，带上 stock_code
            chip_metrics_lookups.add((tracker.stock_id, stock_code, key_dates['latest']))
        if transactions:
            key_dates['initial'] = transactions[0].transaction_date.date()
            score_lookups.add((tracker.stock_id, key_dates['initial']))
            # 收集筹码指标查询需求时，带上 stock_code
            chip_metrics_lookups.add((tracker.stock_id, stock_code, key_dates['initial']))
            last_buy_tx = next((tx for tx in reversed(transactions) if tx.transaction_type == Transaction.TransactionType.BUY), None)
            if last_buy_tx:
                key_dates['last_buy'] = last_buy_tx.transaction_date.date()
                score_lookups.add((tracker.stock_id, key_dates['last_buy']))
                # 收集筹码指标查询需求时，带上 stock_code
                chip_metrics_lookups.add((tracker.stock_id, stock_code, key_dates['last_buy']))
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
    # 适配分表，动态查询 AdvancedChipMetrics
    chip_metrics_map = {}
    if chip_metrics_lookups:
        # 3.1 按分表模型对查询条件进行分组
        model_lookups = {}
        for stock_id, stock_code, trade_date in chip_metrics_lookups:
            if not trade_date:
                continue
            # 根据股票代码获取正确的模型类
            MetricsModel = get_advanced_chip_metrics_model_by_code(stock_code)
            if MetricsModel not in model_lookups:
                model_lookups[MetricsModel] = []
            # 将 Q 对象添加到对应模型的列表中
            model_lookups[MetricsModel].append(Q(stock_id=stock_id, trade_time=trade_date))
        # 3.2 对每个分表模型执行一次批量查询
        all_key_chip_metrics = []
        for model, queries in model_lookups.items():
            if queries:
                # 从对应的分表中查询数据
                results = model.objects.filter(functools.reduce(operator.or_, queries))
                all_key_chip_metrics.extend(list(results))
        # 3.3 将所有查询结果合并到最终的 map 中
        chip_metrics_map = {(cm.stock_id, cm.trade_time): cm for cm in all_key_chip_metrics}
    # --- 步骤4: 组装最终数据，进行精细化计算  ---
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
            # 根据您的要求，移除了此处的 print 调试语句
        trackers_for_display.append({
            'tracker': tracker,
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss_pct,
            'latest_daily_score': latest_daily_score,
            'delta_from_initial': delta_from_initial,
            'delta_from_last_buy': delta_from_last_buy,
            'initial_score': initial_score,
            'last_buy_score': last_buy_score,
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
    - 只从最终的策略信号键 (ZSET) 中获取数据。
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

@login_required
@with_cache_manager_for_views
def stock_detail_view(request, stock_code, cache_manager=None):
    """
    【V1.1 · 叠加先知信号版】渲染股票详情分析页面。
    - 职责: 获取K线、策略分数、先知信号，并处理成图表所需格式，传递给前端。
    """
    stock = get_object_or_404(StockInfo, stock_code=stock_code)
    # 1. 定义时间范围 (最近30天)
    end_date = timezone.now().date()
    start_date = end_date - timedelta(days=30)
    # 2. 获取策略得分数据
    daily_scores = StrategyDailyScore.objects.filter(
        stock=stock,
        trade_date__range=(start_date, end_date)
    ).order_by('-trade_date')
    # 3. 获取K线数据
    time_trade_dao = StockTimeTradeDAO(cache_manager_instance=cache_manager)
    get_kl_data_async = time_trade_dao.get_kl_data_for_chart
    kline_data = async_to_sync(get_kl_data_async)(stock_code, start_date, end_date)
    # 4. 获取“先知信号”数据
    prophet_signals = TradingSignal.objects.filter(
        stock=stock,
        strategy_name='ProphetSignal',
        signal_type='BUY',
        timeframe='D',
        trade_time__date__range=(start_date, end_date)
    ).order_by('trade_time')
    print(f"调试信息: 在 {start_date} 到 {end_date} 期间找到 {prophet_signals.count()} 个先知信号。")
    # 5. 为图表准备数据
    # K线图数据
    kline_chart_data = {
        'dates': [d['trade_time'].strftime('%Y-%m-%d') for d in kline_data],
        'values': [[d['open_qfq'], d['close_qfq'], d['low_qfq'], d['high_qfq']] for d in kline_data],
        'volumes': [d['vol'] for d in kline_data]
    }
    # 得分曲线图数据
    score_map = {score.trade_date: score.final_score for score in daily_scores}
    score_chart_data = {
        'dates': kline_chart_data['dates'],
        'scores': [score_map.get(datetime.strptime(d, '%Y-%m-%d').date(), None) for d in kline_chart_data['dates']]
    }
    # 新增开始: 为“先知信号”准备标记点数据
    prophet_signal_markers = []
    for signal in prophet_signals:
        signal_date = timezone.localtime(signal.trade_time).date()
        # 找到信号当天对应的策略得分，作为标记点的Y轴坐标
        score_on_signal_day = score_map.get(signal_date)
        if score_on_signal_day is not None:
            prophet_signal_markers.append({
                'xAxis': signal_date.strftime('%Y-%m-%d'),
                'yAxis': score_on_signal_day,
                'value': f'先知信号\n得分: {signal.final_score:.0f}', # 标记点上显示的内容
                'itemStyle': {'color': '#ffc107'} # 标记点颜色
            })
    score_chart_data['prophet_signals'] = prophet_signal_markers # 将标记点数据加入字典
    # 新增结束
    context = {
        'page_title': f'{stock.stock_name} ({stock.stock_code}) - 深度分析',
        'stock': stock,
        'daily_scores': daily_scores,
        'kline_chart_data_json': json.dumps(kline_chart_data, cls=DjangoJSONEncoder),
        'score_chart_data_json': json.dumps(score_chart_data, cls=DjangoJSONEncoder),
    }
    return render(request, 'dashboard/stock_detail.html', context)


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
        - 移除所有关于价格、日期、数量的参数依赖和交易创建逻辑。
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
        """
        【V1.1 修正版】创建交易后，调用服务更新持仓状态并重建快照。
        - 修正了对 TransactionService 的调用方式，使用正确的静态方法。
        - 简化了逻辑，将状态更新和快照重建完全委托给 TransactionService。
        """
        tracker = serializer.validated_data['tracker']
        # 1. 检查权限：确保操作的 tracker 属于当前用户
        if tracker.user != self.request.user:
            raise PermissionDenied("你没有权限操作此持仓记录。")
        # 2. 保存交易记录
        transaction = serializer.save()
        print(f"交易记录创建成功: ID={transaction.id}, Tracker ID={tracker.id}") # 调试信息
        # 3. 调用核心服务，一步完成持仓状态更新和快照重建触发
        TransactionService.recalculate_tracker_state_and_rebuild_snapshots(tracker.id)
        print(f"已为 Tracker ID {tracker.id} 调用核心服务进行状态更新。") # 调试信息
        # 4. 发送WebSocket通知给前端
        send_update_to_user_sync(
            self.request.user.id,
            'snapshot_rebuilt',
            {'status': 'success', 'tracker_id': tracker.id}
        )
        logger.info(f"已向用户 {self.request.user.username} 发送快照重建通知。")
    def perform_destroy(self, instance):
        """
        【V1.1 修正版】删除交易后，调用服务更新持仓状态并重建快照。
        - 修正了删除逻辑，确保先删除对象，再调用服务重新计算。
        - 统一使用 TransactionService.recalculate_tracker_state_and_rebuild_snapshots。
        """
        tracker = instance.tracker
        user = self.request.user
        tx_id_for_log = instance.id # 在实例被删除前保存ID
        print(f"准备删除交易记录: ID={tx_id_for_log}, Tracker ID={tracker.id}") # 调试信息
        # 1. 检查权限
        if tracker.user != user:
            raise PermissionDenied("你没有权限操作此持仓记录。")
        # 2. 先执行数据库删除操作
        instance.delete()
        print(f"交易记录 {tx_id_for_log} 已从数据库删除。") # 调试信息
        # 3. 调用核心服务，重新计算所有状态并触发快照重建
        TransactionService.recalculate_tracker_state_and_rebuild_snapshots(tracker.id)
        print(f"已为 Tracker ID {tracker.id} 调用核心服务进行状态更新。") # 调试信息
        # 4. 发送WebSocket通知
        send_update_to_user_sync(
            user.id,
            'snapshot_rebuilt',
            {'status': 'success', 'tracker_id': tracker.id}
        )
        logger.info(f"因交易删除，为 Tracker ID {tracker.id} 成功重建快照，并发送通知。")
    def perform_update(self, serializer):
        """
        更新交易后，调用服务更新持仓状态并重建快照。
        """
        transaction = serializer.save()
        # 调用核心服务
        TransactionService.recalculate_tracker_state_and_rebuild_snapshots(transaction.tracker.id)












