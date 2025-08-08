# 文件: tasks/stock_analysis_tasks.py
# 版本: V2.1 - 装饰器重构版

import asyncio
from datetime import date, datetime, timedelta
from django.utils import timezone
import logging
from decimal import Decimal
from asgiref.sync import sync_to_async
from asgiref.sync import async_to_sync
from celery import Celery
from celery import group, chain
from utils.task_helpers import with_cache_manager
import traceback
import numpy as np
import pandas as pd
from django.db import transaction
from chaoyue_dreams.celery import app as celery_app
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO
from dao_manager.tushare_daos.strategies_dao import StrategiesDAO
from stock_models.stock_analytics import DailyPositionSnapshot, PositionTracker, StrategyDailyScore
from stock_models.index import TradeCalendar
from services.chip_feature_calculator import ChipFeatureCalculator
from stock_models.stock_basic import StockInfo
from stock_models.time_trade import AdvancedChipMetrics, StockDailyBasic
from strategies.multi_timeframe_trend_strategy import MultiTimeframeTrendStrategy
from utils.cache_manager import CacheManager

logger = logging.getLogger('tasks')

async def _get_all_relevant_stock_codes_for_processing(stock_basic_dao: StockBasicInfoDao):
    """
    【V2.0 依赖注入版】
    异步获取所有需要处理的股票代码列表。
    - 核心修改: 不再自己创建DAO，而是接收一个外部传入的DAO实例。
    """
    favorite_stock_codes = set()
    all_stock_codes = set()
    
    try:
        # 直接使用传入的DAO实例
        favorite_stocks = await stock_basic_dao.get_all_favorite_stocks()
        if favorite_stocks:
            for fav in favorite_stocks:
                if fav and fav.get("stock_code"):
                    favorite_stock_codes.add(fav.get("stock_code"))
    except Exception as e:
        logger.error(f"获取自选股列表时出错: {e}", exc_info=True)
        
    try:
        # 直接使用传入的DAO实例
        all_stocks = await stock_basic_dao.get_stock_list()
        if all_stocks:
            for stock in all_stocks:
                if stock and not stock.stock_code.endswith('.BJ'):
                    all_stock_codes.add(stock.stock_code)
    except Exception as e:
        logger.error(f"获取全市场股票列表时出错: {e}", exc_info=True)
        
    non_favorite_stock_codes = list(all_stock_codes - favorite_stock_codes)
    favorite_stock_codes_list = list(favorite_stock_codes)
    
    # 返回排序后的列表，保证每次结果一致
    return sorted(favorite_stock_codes_list), sorted(non_favorite_stock_codes)

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.debug_stock_over_period', queue='debug_tasks')
@with_cache_manager
def debug_stock_over_period(self, stock_code: str, start_date: str, end_date: str, *, cache_manager: CacheManager):
    """
    【V-Debug 专用历史回溯任务 - 装饰器重构版】
    对单个股票在指定的历史时间段内，逐日运行策略分析并打印详细日志。
    """
    logger.info("="*80)
    logger.info(f"--- [历史回溯调试任务启动] ---")
    logger.info(f"  - 股票代码: {stock_code}")
    logger.info(f"  - 分析时段: {start_date} to {end_date}")
    logger.info("="*80)

    async def main():
        strategy_orchestrator = MultiTimeframeTrendStrategy(cache_manager)
        # 执行业务逻辑
        await strategy_orchestrator.debug_run_for_period(
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date
        )
        logger.info(f"--- [历史回溯调试任务完成] ---")
        logger.info(f"股票 [{stock_code}] 的回溯分析执行完毕。")
        logger.info("="*80)
        return {"status": "success", "stock_code": stock_code, "period": f"{start_date}-{end_date}"}

    try:
        return async_to_sync(main)()
    except Exception as e:
        logger.error(f"在执行历史回溯任务 for {stock_code} 时发生严重错误: {e}", exc_info=True)
        logger.info("="*80)
        return {"status": "error", "stock_code": stock_code, "reason": str(e)}


# =================================================================
# =================== 1. 策略任务 ==================
# =================================================================

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.run_multi_timeframe_strategy', queue='calculate_strategy')
@with_cache_manager
def run_multi_timeframe_strategy(self, stock_code: str, trade_date: str, latest_only: bool = False, *, cache_manager: CacheManager):
    """
    【V4.1 - 装饰器重构版】
    - 核心修改: 使用 @with_cache_manager 装饰器自动管理 CacheManager 生命周期。
    """
    async def main():
        # MODIFIED: 不再需要手动创建 CacheManager，直接使用装饰器注入的实例
        strategy_orchestrator = MultiTimeframeTrendStrategy(cache_manager)
        strategies_dao = StrategiesDAO(cache_manager)
        mode_str = "闪电突袭 (仅最新)" if latest_only else "全面战役 (全历史)"
        if trade_date:
            analysis_end_time = f"{trade_date} 16:00:00"
            logger.info(f"[{stock_code}] 开始执行核心策略逻辑 ({mode_str}) for date {trade_date}")
        else:
            analysis_end_time = None
            logger.info(f"[{stock_code}] 开始执行核心策略逻辑 ({mode_str}) for [全历史数据]")
        
        if latest_only:
            # run_for_latest_signal 现在返回四元组
            records_tuple = await strategy_orchestrator.run_for_latest_signal(
                stock_code=stock_code,
                trade_time=analysis_end_time
            )
        else:
            # run_for_stock 现在返回四元组
            records_tuple = await strategy_orchestrator.run_for_stock(
                stock_code=stock_code,
                trade_time=analysis_end_time
            )
        
        # 检查是否有任何需要保存的记录 (检查第一个和第三个列表)
        if not records_tuple or (not records_tuple[0] and not records_tuple[2]):
            logger.info(f"[{stock_code}] 策略运行完成，但未触发任何需要记录的信号或分数。")
            return {"status": "success", "saved_count": 0, "reason": "No DB records to save"}
        
        # 将完整的四元组传递给 DAO
        save_count = await strategies_dao.save_strategy_signals(records_tuple)
        logger.info(f"[{stock_code}] 成功保存 {save_count} 条记录 (包括信号和每日分数)。")
        
        # 这部分逻辑可以保持，因为它只关心 TradingSignal
        if save_count > 0 and records_tuple[0]:
            unique_signal_types = set()
            for signal_obj in records_tuple[0]:
                strategy_name = signal_obj.strategy_name
                timeframe = signal_obj.timeframe
                if strategy_name and timeframe:
                    unique_signal_types.add((strategy_name, timeframe))
        return {"status": "success", "saved_count": save_count}
    try:
        return async_to_sync(main)()
    except Exception as e:
        logger.error(f"执行核心策略逻辑 on {stock_code} 时出错: {e}", exc_info=True)
        return {"status": "error", "reason": str(e)}

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.analyze_all_stocks_full_history', queue='celery')
@with_cache_manager
def analyze_all_stocks_full_history(self, *, cache_manager: CacheManager):
    """
    【V7.0 终极解耦版】
    - 核心架构: 回归本源，此任务的【唯一职责】是建设和更新 StrategyDailyScore 公共数据库。
    - 工作流: 彻底移除所有下游任务链，只对所有股票并行执行 run_multi_timeframe_strategy。
    """
    try:
        logger.info("====== [公共数据库建设-全历史 V7.0] 启动 ======")
        favorite_codes, non_favorite_codes = async_to_sync(_get_all_relevant_stock_codes_for_processing)(StockBasicInfoDao(cache_manager))
        all_codes = favorite_codes + non_favorite_codes
        
        if not all_codes:
            logger.warning("[公共数据库] 未找到任何股票数据，任务终止")
            return {"status": "failed", "reason": "no stocks found"}
            
        stock_count = len(all_codes)
        logger.info(f"[公共数据库] 准备为 {stock_count} 只股票建设全历史策略分数。")
        
        # --- 代码修改开始：回归最简单的并行任务组 ---
        # [修改原因] 彻底解耦，此任务只负责计算公共分数，不关心任何个性化数据。
        analysis_tasks = [
            run_multi_timeframe_strategy.s(code, None, latest_only=False).set(queue='calculate_strategy') for code in all_codes
        ]
        
        workflow = group(analysis_tasks)
        workflow.apply_async()
        # --- 代码修改结束 ---
        
        logger.info(f"[公共数据库] 已成功为 {stock_count} 只股票启动【全历史】分数计算任务。")
        return {"status": "workflow_started", "stock_count": stock_count}
    except Exception as e:
        logger.error(f"[公共数据库-全历史] 任务启动时发生严重错误: {e}", exc_info=True)
        return {"status": "failed", "reason": str(e)}

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.analyze_all_stocks', queue='celery')
@with_cache_manager
def analyze_all_stocks(self, *, cache_manager: CacheManager):
    """
    【V7.0 终极解耦版】
    - 核心架构: 与全历史任务完全一致，此任务的【唯一职责】是更新当日的 StrategyDailyScore。
    """
    try:
        logger.info("====== [公共数据库建设-每日增量 V7.0] 启动 ======")

        # 1. 获取权威交易日 (逻辑不变)
        reference_date = timezone.now().date()
        latest_trade_dates = TradeCalendar.get_latest_n_trade_dates(n=1, reference_date=reference_date)
        if not latest_trade_dates:
            logger.error("【严重错误】无法从交易日历中获取最新的交易日，任务终止！")
            return {"status": "failed", "reason": "Cannot get latest trade date from calendar."}
        
        latest_trade_date = latest_trade_dates[0]
        trade_time_str = latest_trade_date.strftime('%Y-%m-%d')
        logger.info(f"[每日增量] 将使用权威的最新交易日进行分析: {trade_time_str}")
        
        all_codes = []
        favorite_codes, non_favorite_codes = async_to_sync(_get_all_relevant_stock_codes_for_processing)(StockBasicInfoDao(cache_manager))
        all_codes.extend(favorite_codes)
        all_codes.extend(non_favorite_codes)
        
        if not all_codes:
            logger.warning("[每日增量] 未找到任何股票数据，任务终止")
            return {"status": "failed", "reason": "no stocks found"}
            
        stock_count = len(all_codes)
        logger.info(f"[每日增量] 准备为 {stock_count} 只股票更新当日策略分数。")
        
        # --- 代码修改开始：回归最简单的并行任务组 ---
        analysis_tasks = [
            run_multi_timeframe_strategy.s(code, trade_time_str, latest_only=True).set(queue='calculate_strategy') for code in all_codes
        ]
        
        workflow = group(analysis_tasks)
        workflow.apply_async()
        # --- 代码修改结束 ---
        
        logger.info(f"[每日增量] 已成功为 {stock_count} 只股票启动【当日】分数计算任务。")
        return {"status": "workflow_started", "stock_count": stock_count}
        
    except Exception as e:
        logger.error(f"[每日增量] 任务调度时出错: {e}", exc_info=True)
        return {"status": "failed", "reason": str(e)}

# 辅助函数 _chunker 保持不变
def _chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

@celery_app.task(bind=True, name='dashboard.tasks.update_favorite_stock_trackers')
def update_favorite_stock_trackers(self):
    """
    【V3.1 - 健壮版】
    每日运行，为所有活跃的 PositionTracker 创建当日的状态快照。
    - 核心修改: 采用“先检查、再创建”的模式，彻底避免因任务重跑导致的 IntegrityError。
    """
    try:
        # 步骤 1: 确定需要生成快照的日期（最新的一个交易日）
        latest_trade_day = TradeCalendar.get_latest_trade_date()
        if not latest_trade_day:
            logger.warning("无法获取最新的交易日，任务终止。")
            return "无法获取最新的交易日，任务终止。"
        
        logger.info(f"开始为 {latest_trade_day} 生成每日持仓快照...")

        # 步骤 2: 筛选出所有需要创建快照的活跃追踪器
        trackers_to_snapshot = list(PositionTracker.objects.filter(
            Q(status=PositionTracker.Status.HOLDING) | Q(status=PositionTracker.Status.WATCHING)
        ).select_related('user', 'stock')) # 增加 select_related('user')

        if not trackers_to_snapshot:
            logger.info("没有找到需要创建快照的活跃追踪器。")
            return "没有需要更新的活跃追踪器。"
        
        # 步骤 3: 一次性查询当天已经存在的快照，存入一个集合以便快速查找
        existing_snapshots = set(
            DailyPositionSnapshot.objects.filter(
                tracker_id__in=[t.id for t in trackers_to_snapshot],
                snapshot_date=latest_trade_day
            ).values_list('tracker_id', 'snapshot_date')
        )
        logger.info(f"在 {latest_trade_day}，发现 {len(existing_snapshots)} 条已存在的快照。")

        # 步骤 4: 筛选出当天还没有快照的追踪器
        trackers_needing_snapshot = [
            tracker for tracker in trackers_to_snapshot 
            if (tracker.id, latest_trade_day) not in existing_snapshots
        ]

        if not trackers_needing_snapshot:
            logger.info(f"所有活跃追踪器在 {latest_trade_day} 均已有快照，无需创建。")
            return "所有活跃追踪器均已有快照。"
        
        logger.info(f"准备为 {len(trackers_needing_snapshot)} 个追踪器创建新快照...")
        
        # 步骤 5: 使用服务批量创建快照 (这里我们假设有一个服务来处理这个逻辑)
        # 注意：这里我们直接调用服务，服务内部应该处理快照的计算和创建
        snapshot_service = PositionSnapshotService()
        
        # 收集需要通知的用户
        users_to_notify = set()
        
        # 在循环中调用重建服务，并收集用户
        for tracker in trackers_needing_snapshot:
            try:
                # 假设服务方法是 create_snapshot_for_date
                # 这个服务会计算并创建单个快照
                snapshot_service.create_snapshot_for_date(tracker, latest_trade_day)
                users_to_notify.add(tracker.user.id)
            except Exception as e:
                logger.error(f"为 Tracker ID {tracker.id} 创建 {latest_trade_day} 快照时失败: {e}", exc_info=True)
        
        # 步骤 6: 任务完成后，向所有受影响的用户发送通知
        for user_id in users_to_notify:
            send_update_to_user_sync(user_id, 'snapshot_rebuilt', {'status': 'success', 'source': 'daily_task'})
            logger.info(f"已向用户 {user_id} 发送快照更新通知。")

        logger.info(f"每日快照任务完成。成功处理 {len(trackers_needing_snapshot)} 个追踪器。")
        return f"成功处理 {len(trackers_needing_snapshot)} 个追踪器的每日快照。"
            
    except Exception as e:
        logger.error(f"更新持仓追踪器（创建快照）时发生严重错误: {e}", exc_info=True)
        self.retry(exc=e, countdown=60)
        return f"任务失败: {e}"

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.rebuild_all_snapshots_for_all_trackers', queue='celery')
def rebuild_all_snapshots_for_all_trackers(self):
    """
    【V1.0】批量重建所有持仓中Tracker的快照 (总管任务)
    - 职责: 找到所有持仓中的Tracker，并为它们分别派发重建快照的原子任务。
    """
    logger.info("启动批量重建所有持仓快照的总管任务...")
    
    # 使用 .values_list('id', flat=True) 更高效地只获取ID
    holding_tracker_ids = list(PositionTracker.objects.filter(
        status=PositionTracker.Status.HOLDING,
        current_quantity__gt=0
    ).values_list('id', flat=True))

    if not holding_tracker_ids:
        logger.info("没有发现任何需要重建快照的持仓，任务结束。")
        return {"status": "skipped", "reason": "no active trackers found"}

    logger.info(f"发现 {len(holding_tracker_ids)} 个持仓，准备为它们派发重建任务...")

    # 为每个ID派发一个独立的重建任务
    for tracker_id in holding_tracker_ids:
        # 这里我们调用的是任务签名，而不是直接调用函数或服务，这是正确的解耦方式。
        rebuild_snapshots_for_tracker_task.delay(tracker_id)

    logger.info("所有重建任务已成功派发。")
    return {"status": "dispatched", "tracker_count": len(holding_tracker_ids)}

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.rebuild_snapshots_for_tracker_task', queue='calculate_strategy')
@with_cache_manager
def rebuild_snapshots_for_tracker_task(self, tracker_id: int, *, cache_manager: CacheManager):
    """
    【V1.1 修正版】单个持仓追踪器快照重建任务 (服务包装器)
    - 职责: 调用 PositionSnapshotService 为指定的 tracker_id 重建快照。
    """
    from services.position_snapshot_service import PositionSnapshotService

    logger.info(f"接收到重建快照任务，Tracker ID: {tracker_id}")
    service = PositionSnapshotService(cache_manager)
    # 因为 service 的方法是 async，而 Celery 任务是 sync，所以需要 async_to_sync
    result_count = async_to_sync(service.rebuild_snapshots_for_tracker)(tracker_id)
    logger.info(f"Tracker ID {tracker_id} 的快照重建任务完成，处理了 {result_count} 条记录。")
    return {"tracker_id": tracker_id, "snapshots_processed": result_count}

# =================================================================
# =================== 2. 高级筹码特征任务 ==================
# =================================================================

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.schedule_precompute_advanced_chips', queue='celery')
@with_cache_manager
def schedule_precompute_advanced_chips(self, *, cache_manager: CacheManager):
    """
    【V2.1 装饰器重构版】
    """
    try:
        logger.info("开始调度 [高级筹码指标预计算] 任务...")
        all_codes = []
        async def main():
            nonlocal all_codes
            stock_basic_dao = StockBasicInfoDao(cache_manager)
            favorite_codes, non_favorite_codes = await _get_all_relevant_stock_codes_for_processing(stock_basic_dao)
            all_codes.extend(favorite_codes)
            all_codes.extend(non_favorite_codes)
        async_to_sync(main)()
        if not all_codes:
            logger.warning("未找到任何股票数据，预计算任务终止。")
            return {"status": "failed", "reason": "no stocks found"}
        stock_count = len(all_codes)
        logger.info(f"找到 {stock_count} 只股票待进行高级筹码预计算。")
        for stock_code in all_codes:
            precompute_advanced_chips_for_stock.s(stock_code).set(queue='SaveHistoryData_TimeTrade').apply_async()
        logger.info(f"已为 {stock_count} 只股票调度 '高级筹码指标预计算' 任务。")
        return {"status": "started", "stock_count": stock_count}
    except Exception as e:
        logger.error(f"调度高级筹码预计算任务时出错: {e}", exc_info=True)
        return {"status": "failed", "reason": str(e)}

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.precompute_advanced_chips_for_stock', queue='SaveHistoryData_TimeTrade')
@with_cache_manager
def precompute_advanced_chips_for_stock(self, stock_code: str, is_incremental: bool = True, *, cache_manager: CacheManager):
    """
    【执行器 V10.4 - 装饰器重构版】
    - 核心修改: 使用 @with_cache_manager 装饰器自动管理 CacheManager 生命周期。
    """
    time_trade_dao = StockTimeTradeDAO(cache_manager)
    async def main(time_dao, incremental_flag: bool):
        mode = "增量更新" if incremental_flag else "全量刷新"
        logger.info(f"[{stock_code}] 开始执行高级筹码指标预计算 (V10.4, 模式: {mode})...")
        get_stock_info_async = sync_to_async(StockInfo.objects.get, thread_sensitive=True)
        @sync_to_async(thread_sensitive=True)
        def get_latest_metric_async(stock_info_obj):
            try:
                return AdvancedChipMetrics.objects.filter(stock=stock_info_obj).latest('trade_time')
            except AdvancedChipMetrics.DoesNotExist:
                return None
        @sync_to_async(thread_sensitive=True)
        def get_data_async(model, stock_info_obj, fields: tuple = None, date_field='trade_time', start_date=None):
            qs = model.objects.filter(stock=stock_info_obj)
            if start_date:
                filter_kwargs = {f'{date_field}__gte': start_date}
                qs = qs.filter(**filter_kwargs)
            if fields:
                return pd.DataFrame.from_records(qs.values(*fields))
            else:
                return pd.DataFrame.from_records(qs.values())
        @sync_to_async(thread_sensitive=True)
        def save_metrics_async(stock_info_obj, records_to_create_list, do_delete_first: bool):
            with transaction.atomic():
                if do_delete_first:
                    logger.info(f"[{stock_code}] 全量模式：删除所有旧数据...")
                    AdvancedChipMetrics.objects.filter(stock=stock_info_obj).delete()
                AdvancedChipMetrics.objects.bulk_create(records_to_create_list, batch_size=5000)
        try:
            stock_info = await get_stock_info_async(stock_code=stock_code)
            max_lookback_days = 160
            last_metric_date = None
            if incremental_flag:
                last_metric = await get_latest_metric_async(stock_info)
                if last_metric:
                    last_metric_date = last_metric.trade_time
                else:
                    logger.info(f"[{stock_code}] 未找到任何历史指标，自动切换到全量刷新模式。")
                    incremental_flag = False
            fetch_start_date = None
            if incremental_flag and last_metric_date:
                fetch_start_date = last_metric_date - timedelta(days=max_lookback_days + 20)
            chip_model = time_dao.get_cyq_chips_model_by_code(stock_code)
            daily_data_model = time_dao.get_daily_data_model_by_code(stock_code)
            data_tasks = {
                "cyq_chips": get_data_async(chip_model, stock_info, fields=('trade_time', 'price', 'percent'), start_date=fetch_start_date),
                "daily_data": get_data_async(daily_data_model, stock_info, fields=('trade_time', 'close_qfq', 'vol', 'high_qfq', 'low_qfq'), start_date=fetch_start_date),
                "daily_basic": get_data_async(StockDailyBasic, stock_info, fields=('trade_time', 'float_share'), start_date=fetch_start_date),
            }
            results = await asyncio.gather(*data_tasks.values())
            data_dfs = dict(zip(data_tasks.keys(), results))
            # logger.info(f"[{stock_code}] 正在执行法务级数据审计...")
            cyq_chips_df = data_dfs.get("cyq_chips")
            if cyq_chips_df is None or cyq_chips_df.empty:
                logger.error(f"[{stock_code}] [审计失败] 黄金标准数据源 'cyq_chips' 为空！任务终止。")
                return {"status": "failed", "reason": "Master data source 'cyq_chips' is empty"}
            cyq_chips_df['trade_time'] = pd.to_datetime(cyq_chips_df['trade_time'])
            master_dates = set(cyq_chips_df['trade_time'].dt.date.unique())
            is_data_healthy = True
            audit_warnings = []
            other_essential_dfs = {
                "daily_data": data_dfs.get("daily_data"),
                "daily_basic": data_dfs.get("daily_basic")
            }
            for name, df in other_essential_dfs.items():
                if df is None or df.empty:
                    logger.error(f"[{stock_code}] [审计失败] 关键数据源 '{name}' 为空！任务终止。")
                    return {"status": "failed", "reason": f"Essential data source '{name}' is empty"}
                df['trade_time'] = pd.to_datetime(df['trade_time'])
                source_dates = set(df['trade_time'].dt.date.unique())
                missing_in_source = sorted(list(master_dates - source_dates))
                if missing_in_source:
                    is_data_healthy = False
                    warning_msg = (f"[{stock_code}] [审计警告] 数据源 '{name}' "
                                   f"缺失了 {len(missing_in_source)} 个交易日的数据。 "
                                   f"缺失日期示例: {missing_in_source[:5]}...")
                    audit_warnings.append(warning_msg)
            # logger.info(f"[{stock_code}] 正在执行二级法务审计 (值有效性检查)...")
            daily_data_df = other_essential_dfs['daily_data']
            required_cols_in_daily = ['close_qfq', 'vol', 'high_qfq', 'low_qfq']
            if daily_data_df[required_cols_in_daily].isnull().values.any():
                is_data_healthy = False
                problematic_rows = daily_data_df[daily_data_df[required_cols_in_daily].isnull().any(axis=1)]
                for trade_date, row in problematic_rows.iterrows():
                    missing_fields = [col for col in required_cols_in_daily if pd.isna(row[col])]
                    warning_msg = (f"[{stock_code}] [审计警告] 在日期 {row['trade_time'].date()} 的行情数据中，"
                                   f"发现 NULL 值。缺失字段: {missing_fields}")
                    audit_warnings.append(warning_msg)
            if not is_data_healthy:
                logger.error(f"[{stock_code}] [审计失败] 数据一致性检查未通过，任务已熔断。详情如下：")
                for warning in audit_warnings:
                    logger.error(warning)
                return {"status": "failed", "reason": "Data consistency audit failed."}
            cyq_chips_data = data_dfs['cyq_chips']
            if cyq_chips_data.empty:
                logger.warning(f"[{stock_code}] 在指定范围内找不到原始筹码数据，任务终止。")
                return {"status": "skipped", "reason": "no raw chip data in range"}
            cyq_chips_data['trade_time'] = pd.to_datetime(cyq_chips_data['trade_time']).dt.date
            daily_sums = cyq_chips_data.groupby('trade_time')['percent'].transform('sum')
            mask_sum_to_one = np.isclose(daily_sums, 1.0, atol=0.1)
            if mask_sum_to_one.any():
                cyq_chips_data.loc[mask_sum_to_one, 'percent'] *= 100
            daily_data = data_dfs['daily_data']
            daily_data['trade_time'] = pd.to_datetime(daily_data['trade_time']).dt.date
            daily_basic_data = data_dfs['daily_basic']
            daily_basic_data['trade_time'] = pd.to_datetime(daily_basic_data['trade_time']).dt.date
            daily_data['daily_turnover_volume'] = daily_data['vol'] * 100
            daily_data = daily_data.rename(columns={'close_qfq': 'close_price', 'high_qfq': 'high_price', 'low_qfq': 'low_price'})
            daily_basic_data['total_chip_volume'] = daily_basic_data['float_share'] * 10000
            daily_basic_data = daily_basic_data.drop(columns=['float_share'])
            merged_df = pd.merge(cyq_chips_data, daily_data, on='trade_time', how='inner')
            merged_df = pd.merge(merged_df, daily_basic_data, on='trade_time', how='inner')
            if merged_df.empty:
                logger.warning(f"[{stock_code}] 数据源内连接(inner join)后结果为空，请检查诊断日志中天数最短的数据源。任务终止。")
                return {"status": "skipped", "reason": "data sources could not be merged"}
            merged_df = merged_df.sort_values('trade_time').reset_index(drop=True)
            daily_close_prices = merged_df[['trade_time', 'close_price']].drop_duplicates().set_index('trade_time')
            daily_close_prices['prev_20d_close'] = daily_close_prices['close_price'].shift(20)
            merged_df = pd.merge(merged_df, daily_close_prices[['prev_20d_close']], on='trade_time', how='left')
            grouped_data = merged_df.groupby('trade_time')
            all_metrics_list = []
            for trade_date, daily_full_df in grouped_data:
                if incremental_flag and last_metric_date and trade_date <= last_metric_date:
                    continue
                context_data = daily_full_df.iloc[0].to_dict()
                chip_data_for_calc = daily_full_df[['price', 'percent']]
                if not chip_data_for_calc.empty:
                    percent_sum = chip_data_for_calc['percent'].sum()
                    if not np.isclose(percent_sum, 100.0) and percent_sum > 0:
                        normalized_percent = chip_data_for_calc['percent'] / percent_sum
                    else:
                        normalized_percent = chip_data_for_calc['percent'] / 100.0
                    weight_avg_cost = np.average(chip_data_for_calc['price'], weights=normalized_percent)
                    context_data['weight_avg_cost'] = weight_avg_cost
                else:
                    continue
                calculator = ChipFeatureCalculator(chip_data_for_calc.sort_values(by='price'), context_data)
                daily_metrics = calculator.calculate_all_metrics()
                if daily_metrics:
                    daily_metrics['trade_time'] = trade_date
                    daily_metrics['prev_20d_close'] = context_data.get('prev_20d_close')
                    all_metrics_list.append(daily_metrics)
            if not all_metrics_list:
                logger.info(f"[{stock_code}] 没有需要计算的新指标。任务正常结束。")
                return {"status": "success", "processed_days": 0, "reason": "already up-to-date"}
            new_metrics_df = pd.DataFrame(all_metrics_list).set_index('trade_time')
            final_metrics_df = new_metrics_df
            if incremental_flag and last_metric_date:
                past_metrics_df = await get_data_async(
                    AdvancedChipMetrics, stock_info, 
                    start_date=fetch_start_date
                )
                if not past_metrics_df.empty:
                    past_metrics_df = past_metrics_df.set_index('trade_time')
                    final_metrics_df = pd.concat([past_metrics_df, new_metrics_df]).sort_index()
            slope_periods = [5, 8, 13, 21, 34, 55, 89, 144]
            accel_periods = [5, 21]
            if 'peak_cost' in final_metrics_df.columns:
                for period in slope_periods:
                    slope = final_metrics_df['peak_cost'].rolling(window=period, min_periods=2).apply(
                        lambda x: np.polyfit(range(len(x)), x.dropna(), 1)[0] if len(x.dropna()) > 1 else np.nan, raw=False
                    )
                    final_metrics_df[f'peak_cost_slope_{period}d'] = slope
                for period in accel_periods:
                    final_metrics_df[f'peak_cost_accel_{period}d'] = final_metrics_df[f'peak_cost_slope_{period}d'].diff()
            if 'concentration_90pct' in final_metrics_df.columns:
                final_metrics_df['concentration_90pct_slope_5d'] = final_metrics_df['concentration_90pct'].rolling(5).mean().diff()
            records_to_save_df = final_metrics_df.loc[new_metrics_df.index]
            records_to_create = []
            model_fields = {f.name for f in AdvancedChipMetrics._meta.get_fields()}
            for trade_date, row in records_to_save_df.iterrows():
                record_data = {}
                for field_name in model_fields:
                    if field_name in row.index:
                        value = row[field_name]
                        if isinstance(value, float) and not np.isfinite(value):
                            record_data[field_name] = None
                        elif pd.isna(value):
                            record_data[field_name] = None
                        elif isinstance(value, float):
                            if abs(value) < 1e-10:
                                record_data[field_name] = Decimal('0.0')
                            else:
                                record_data[field_name] = Decimal(str(round(value, 8)))
                        else:
                            record_data[field_name] = value
                record_data.pop('id', None)
                record_data.pop('stock', None)
                if record_data:
                    records_to_create.append(AdvancedChipMetrics(stock=stock_info, trade_time=trade_date, **record_data))
            await save_metrics_async(stock_info, records_to_create, not incremental_flag)
            logger.info(f"[{stock_code}] 成功！模式[{mode}]下，为 {len(records_to_create)} 个交易日计算并存储了高级筹码指标。")
            return {"status": "success", "processed_days": len(records_to_create)}

        except StockInfo.DoesNotExist:
            logger.error(f"[{stock_code}] 在StockInfo中找不到该股票，任务终止。")
            return {"status": "failed", "reason": "stock_code not found in StockInfo"}
        except Exception as e:
            logger.error(f"[{stock_code}] 高级筹码指标预计算失败: {e}", exc_info=True)
            return {"status": "failed", "reason": str(e)}

    try:
        result = async_to_sync(main)(time_trade_dao, is_incremental)
        return result
    except Exception as e:
        logger.error(f"--- CATCHING EXCEPTION in precompute_advanced_chips_for_stock for {stock_code}: {e}", exc_info=True)
        raise
