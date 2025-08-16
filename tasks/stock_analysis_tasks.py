# 文件: tasks/stock_analysis_tasks.py
# 版本: V2.1 - 装饰器重构版

import asyncio
import time
from datetime import date, datetime, timedelta
from django.utils import timezone
import logging
from decimal import Decimal
from collections import defaultdict
from asgiref.sync import sync_to_async
from asgiref.sync import async_to_sync
from celery import Celery
from celery import group, chain, chord
from django.db.models import Min, Max
from utils.task_helpers import with_cache_manager
from utils.model_helpers import get_daily_data_model_by_code, get_cyq_chips_model_by_code, get_advanced_chip_metrics_model_by_code
from tqdm import tqdm
from services.performance_analysis_service import PerformanceAnalysisService
import numpy as np
import pandas as pd
from django.db import transaction, connection
from chaoyue_dreams.celery import app as celery_app
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO
from dao_manager.tushare_daos.strategies_dao import StrategiesDAO
from stock_models.stock_analytics import DailyPositionSnapshot, PositionTracker, StrategyDailyScore, TradingSignal, AtomicSignalPerformance, StrategyDailyState
from stock_models.index import TradeCalendar
from services.chip_feature_calculator import ChipFeatureCalculator
from stock_models.stock_basic import StockInfo
from stock_models.time_trade import AdvancedChipMetrics, StockDailyBasic, AdvancedChipMetrics_SZ, AdvancedChipMetrics_SH, AdvancedChipMetrics_CY, AdvancedChipMetrics_KC, AdvancedChipMetrics_BJ
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
def run_multi_timeframe_strategy(self, stock_code: str, trade_date: str = None, latest_only: bool = False, start_date_str: str = None, *, cache_manager: CacheManager):
    """
    【V4.2 - 支持起始日期的策略计算任务】
    - 核心修改: 使用 @with_cache_manager 装饰器自动管理 CacheManager 生命周期。
    - 新增功能: 增加了可选参数 `start_date_str`。当 `latest_only=False` 时，
                可以指定一个起始日期，任务将只保存该日期之后（含当天）的策略记录。
                这保证了指标计算的准确性（使用全历史数据），同时提供了灵活的数据保存范围。

    Args:
        stock_code (str): 股票代码。
        trade_date (str, optional): 目标交易日 'YYYY-MM-DD'，主要用于 latest_only=True 模式。
        latest_only (bool, optional): 是否只处理最新数据。
        start_date_str (str, optional): 起始日期 'YYYY-MM-DD'，用于全历史模式下指定保存的起点。
        cache_manager (CacheManager): 由装饰器注入的缓存管理器实例。
    """
    async def main():
        strategy_orchestrator = MultiTimeframeTrendStrategy(cache_manager)
        strategies_dao = StrategiesDAO(cache_manager)
        
        # 【代码修改】增强日志，以反映新的 start_date_str 参数
        mode_str = "闪电突袭 (仅最新)" if latest_only else "全面战役 (全历史)"
        if latest_only:
            analysis_end_time = f"{trade_date} 16:00:00" if trade_date else None
            logger.info(f"[{stock_code}] 开始执行核心策略逻辑 ({mode_str}) for date {trade_date}")
        else:
            analysis_end_time = None
            # 【代码修改】在全历史模式下，检查并记录 start_date_str
            if start_date_str:
                logger.info(f"[{stock_code}] 开始执行核心策略逻辑 ({mode_str})，将从 [{start_date_str}] 开始保存记录。")
                print(f"调试信息 [{stock_code}]: 全历史模式，指定起始日期 {start_date_str}") # 调试输出
            else:
                logger.info(f"[{stock_code}] 开始执行核心策略逻辑 ({mode_str}) for [全部历史数据]")
                print(f"调试信息 [{stock_code}]: 全历史模式，处理所有数据") # 调试输出

        records_tuple = None # 初始化为 None
        if latest_only:
            # run_for_latest_signal 返回四元组 (逻辑不变)
            records_tuple = await strategy_orchestrator.run_for_latest_signal(
                stock_code=stock_code,
                trade_time=analysis_end_time
            )
        else:
            # run_for_stock 返回四元组 (逻辑不变)
            records_tuple = await strategy_orchestrator.run_for_stock(
                stock_code=stock_code,
                trade_time=analysis_end_time,
                start_date_str=start_date_str
            )

        # 检查是否有任何需要保存的记录 (检查第一个和第三个列表)
        # 此处逻辑不变，但 records_tuple 可能已经是过滤后的结果
        if not records_tuple or (not records_tuple[0] and not records_tuple[2]):
            logger.info(f"[{stock_code}] 策略运行完成，但未触发任何需要记录的信号或分数 (或已被日期过滤)。")
            return {"status": "success", "saved_count": 0, "reason": "No DB records to save"}
        
        # 将完整的（或过滤后的）四元组传递给 DAO
        save_count = await strategies_dao.save_strategy_signals(records_tuple)
        logger.info(f"[{stock_code}] 成功保存 {save_count} 条记录 (包括信号和每日分数)。")
        
        # 这部分逻辑可以保持，因为它只关心 TradingSignal (逻辑不变)
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
        # 【代码修改】在Celery任务中，最好更新任务状态以方便监控
        self.update_state(state='FAILURE', meta={'exc': str(e)})
        return {"status": "error", "reason": str(e)}

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.analyze_all_stocks_full_history', queue='celery')
@with_cache_manager
def analyze_all_stocks_full_history(self, *, start_date_str: str = None, cache_manager: CacheManager):
    """
    【V7.1 - 支持指定起始日期的全历史回测任务】
    - 核心架构: 回归本源，此任务的【唯一职责】是建设和更新 StrategyDailyScore 公共数据库。
    - 工作流: 彻底移除所有下游任务链，只对所有股票并行执行 run_multi_timeframe_strategy。
    - 新增功能: 支持可选的 `start_date_str` 参数，用于从指定日期开始回测，方便增量修复或分段回测。

    Args:
        start_date_str (str, optional): 起始日期字符串，格式为 'YYYY-MM-DD'。
                                        如果为 None，则处理所有历史。Defaults to None.
        cache_manager (CacheManager): 由装饰器注入的缓存管理器实例。
    """
    try:
        # 【代码修改】根据 start_date_str 参数更新日志
        if start_date_str:
            logger.info(f"====== [公共数据库建设-全历史 V7.1] 启动 (指定起始日期: {start_date_str}) ======")
            print(f"调试信息：任务将从 {start_date_str} 开始计算和保存策略数据。") # 调试输出
        else:
            logger.info("====== [公共数据库建设-全历史 V7.1] 启动 (处理全部历史) ======")
            print("调试信息：未指定起始日期，将处理全部历史数据。") # 调试输出

        # 原始逻辑不变
        favorite_codes, non_favorite_codes = async_to_sync(_get_all_relevant_stock_codes_for_processing)(StockBasicInfoDao(cache_manager))
        all_codes = favorite_codes + non_favorite_codes
        
        if not all_codes:
            logger.warning("[公共数据库] 未找到任何股票数据，任务终止")
            return {"status": "failed", "reason": "no stocks found"}
            
        stock_count = len(all_codes)
        logger.info(f"[公共数据库] 准备为 {stock_count} 只股票建设全历史策略分数。")
        
        # --- 代码修改开始：将 start_date_str 参数透传给子任务 ---
        # [修改原因] 将调度任务接收到的日期参数，分发给每一个具体的计算任务。
        analysis_tasks = [
            run_multi_timeframe_strategy.s(
                stock_code=code, 
                trade_date=None, 
                latest_only=False,
                start_date_str=start_date_str  # 【代码修改】将参数传递给子任务
            ).set(queue='calculate_strategy') for code in all_codes
        ]
        
        workflow = group(analysis_tasks)
        workflow.apply_async()
        # --- 代码修改结束 ---
        
        logger.info(f"[公共数据库] 已成功为 {stock_count} 只股票启动【全历史】分数计算任务。")
        
        # 【代码修改】在返回结果中也包含 start_date_str，方便追踪
        return {"status": "workflow_started", "stock_count": stock_count, "start_date": start_date_str}
    except Exception as e:
        logger.error(f"[公共数据库-全历史] 任务启动时发生严重错误: {e}", exc_info=True)
        return {"status": "failed", "reason": str(e)}

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.analyze_all_stocks', queue='celery')
@with_cache_manager
def analyze_all_stocks(self, *, cache_manager: CacheManager):
    """
    【V7.7 - 健壮性与日志增强版】
    - 核心重构: 适配 AdvancedChipMetrics 分表结构，使用原生 SQL 的 UNION ALL 进行高效的跨表聚合查询，以确定权威交易日。
    - 健壮性: 彻底解决了因模型分表导致无法确定权威日期的问题，同时保持了数据驱动的健壮性。
    - 日志增强: 当数据未就绪时，不再是简单退出，而是会明确记录当前最新日期的数据量与目标阈值的差距，极大提升了可调试性。
    """
    try:
        logger.info("====== [公共数据库建设-每日增量 V7.7 健壮性与日志增强版] 启动 ======")
        
        # --- 步骤1: 数据驱动的权威日期发现机制 ---
        logger.info("步骤1: 正在通过原生SQL查询所有 AdvancedChipMetrics 分表，以确定数据就绪的权威交易日...")
        
        # 1.1 获取市场上的股票总数，作为基准 
        stock_basic_dao = StockBasicInfoDao(cache_manager)
        all_stocks = async_to_sync(stock_basic_dao.get_stock_list)()
        total_stock_count = len(all_stocks)
        if total_stock_count == 0:
            logger.error("【严重错误】无法从 StockInfo 获取任何股票，任务终止！")
            return {"status": "failed", "reason": "Could not retrieve stock list."}

        # 1.2 定义数据就绪的阈值 
        readiness_threshold = int(total_stock_count * 0.9)
        logger.info(f"市场总股票数: {total_stock_count}, 数据就绪阈值: {readiness_threshold} 支股票。")

        # 1.3.1 动态获取所有分表的表名 
        metrics_models = [
            AdvancedChipMetrics_SZ, AdvancedChipMetrics_SH, AdvancedChipMetrics_CY,
            AdvancedChipMetrics_KC, AdvancedChipMetrics_BJ
        ]
        table_names = [model._meta.db_table for model in metrics_models]
        
        # 1.3.2 构建 UNION ALL 子查询部分 
        union_all_query = " UNION ALL ".join([f"SELECT trade_time, stock_id FROM {table}" for table in table_names])

        # 【代码修改】修改SQL查询逻辑：不再使用HAVING过滤，而是直接获取最新日期及其数据量
        # 这样可以获取到调试信息，即使数据未满足要求
        raw_sql = f"""
            SELECT
                trade_time,
                COUNT(stock_id) AS stock_count
            FROM (
                {union_all_query}
            ) AS combined_metrics
            GROUP BY
                trade_time
            ORDER BY
                trade_time DESC
            LIMIT 1
        """

        # 1.3.4 执行查询
        with connection.cursor() as cursor:
            # 【代码修改】移除了HAVING子句，因此不再需要传递参数
            cursor.execute(raw_sql)
            result = cursor.fetchone()

        # 在Python中进行数据就绪检查，并提供更详细的日志
        if not result:
            logger.warning("【任务暂停】AdvancedChipMetrics 所有分表中没有任何数据。请检查上游筹码计算任务是否已运行。任务安全退出。")
            return {"status": "skipped", "reason": "AdvancedChipMetrics tables are completely empty."}

        latest_trade_date, actual_stock_count = result[0], result[1]
        
        # 调试输出，无论成功与否都打印当前状态
        print(f"调试信息：数据库中最新的交易日是 {latest_trade_date.strftime('%Y-%m-%d')}，拥有 {actual_stock_count} 条数据。")

        if actual_stock_count < readiness_threshold:
            logger.warning(
                f"【任务暂停】数据尚未就绪。最新日期 {latest_trade_date.strftime('%Y-%m-%d')} "
                f"仅有 {actual_stock_count} 条数据，未达到 {readiness_threshold} 的阈值。可能是上游筹码计算任务尚未完成。任务安全退出。"
            )
            return {"status": "skipped", "reason": "No trade date met the data readiness threshold."}

        # 1.4 确定权威日期 
        trade_time_str = latest_trade_date.strftime('%Y-%m-%d')
        logger.info(f"【权威日期确定】: {trade_time_str}。该日已有 {actual_stock_count} 支股票筹码数据就绪。")


        # 步骤2: 使用权威日期进行精确的数据清理 
        logger.info(f"步骤2: 清理 {trade_time_str} 的旧策略数据，确保幂等性...")
        try:
            # 【代码修改】这里的 latest_trade_date 已经是 date 对象，可以直接使用
            start_of_day_aware = timezone.make_aware(datetime.combine(latest_trade_date, datetime.min.time()))
            end_of_day_aware = start_of_day_aware + timedelta(days=1)
            
            with transaction.atomic():
                deleted_scores_count, _ = StrategyDailyScore.objects.filter(trade_date=latest_trade_date).delete()
                deleted_signals_count, _ = TradingSignal.objects.filter(
                    trade_time__gte=start_of_day_aware,
                    trade_time__lt=end_of_day_aware
                ).delete()
                logger.info(f"清理完成。删除了 {deleted_scores_count} 条每日分数记录，{deleted_signals_count} 条交易信号记录。")
        except Exception as e:
            logger.error(f"清理 {trade_time_str} 的旧数据时发生严重错误，任务终止: {e}", exc_info=True)
            return {"status": "failed", "reason": "Data cleanup failed."}

        # 步骤3: 获取股票列表 (逻辑不变, 使用之前已获取的 all_stocks)
        all_codes = [stock.stock_code for stock in all_stocks]
        stock_count = len(all_codes)
        logger.info(f"[每日增量] 准备为 {stock_count} 只股票在权威日期 {trade_time_str} 上更新策略分数。")
        
        # 步骤4: 派发并行任务 
        # 假设 run_multi_timeframe_strategy 任务已正确导入
        # from .some_other_task_file import run_multi_timeframe_strategy 
        analysis_tasks = [
            run_multi_timeframe_strategy.s(code, trade_time_str, latest_only=True).set(queue='calculate_strategy') for code in all_codes
        ]
        workflow = group(analysis_tasks)
        workflow.apply_async()
        
        logger.info(f"[每日增量] 已成功为 {stock_count} 只股票启动【当日】分数计算任务。")
        return {"status": "workflow_started", "stock_count": stock_count, "authoritative_date": trade_time_str}
    except Exception as e:
        logger.error(f"[每日增量] 任务调度时出错: {e}", exc_info=True)
        return {"status": "failed", "reason": str(e)}

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

# 单个持仓追踪器快照重建任务 (服务包装器)
@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.rebuild_snapshots_for_tracker_task', queue='dashboard')
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

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.rebuild_snapshots_for_all_active_trackers_task', queue='dashboard')
def rebuild_snapshots_for_all_active_trackers_task(self):
    """
    【V1.0】每晚定期的快照重建任务（兜底）。
    - 职责: 遍历所有当前状态为 "HOLDING" 的 PositionTracker，
            并为它们调用核心的重建服务，以确保数据最终一致性。
            这可以修复因当天交易数据录入早于行情数据同步而导致的快照缺失问题。
    """
    logger.info("====== 开始执行【每晚活跃持仓快照重建】兜底任务 ======")
    
    # 筛选出所有当前持仓中的追踪器
    active_trackers = PositionTracker.objects.filter(status=PositionTracker.Status.HOLDING)
    
    if not active_trackers.exists():
        logger.info("没有找到任何活跃的持仓记录，任务结束。")
        return "没有找到任何活跃的持仓记录。"

    tracker_count = active_trackers.count()
    logger.info(f"发现 {tracker_count} 个活跃的持仓记录，准备开始重建...")
    
    success_count = 0
    failure_count = 0
    from services.transaction_service import TransactionService
    for tracker in active_trackers:
        try:
            logger.info(f"  -> 正在为 Tracker ID: {tracker.id} (股票: {tracker.stock.stock_code}) 触发重建...")
            # 调用核心服务，该服务会先重新计算持仓状态，然后触发异步的快照重建
            # 这是最可靠的方式，能确保 average_cost 等状态也是最新的
            result = TransactionService.recalculate_tracker_state_and_rebuild_snapshots(tracker.id)
            if result:
                success_count += 1
            else:
                failure_count += 1
                logger.warning(f"  Tracker ID: {tracker.id} 的状态更新和快照重建触发失败。")
        except Exception as e:
            failure_count += 1
            logger.error(f"  处理 Tracker ID: {tracker.id} 时发生意外错误: {e}", exc_info=True)

    summary = f"====== 【每晚活跃持仓快照重建】任务完成。总数: {tracker_count}, 成功: {success_count}, 失败: {failure_count} ======"
    logger.info(summary)
    return summary

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
    【执行器 V10.5 - 分表适配版】
    - 核心修改: 适配 AdvancedChipMetrics 模型分表，动态读写对应的数据库表。
    - 技术改造: 使用 @with_cache_manager 装饰器自动管理 CacheManager 生命周期。
    """
    time_trade_dao = StockTimeTradeDAO(cache_manager)
    async def main(time_dao, incremental_flag: bool):
        mode = "增量更新" if incremental_flag else "全量刷新"
        # logger.info(f"[{stock_code}] 开始执行高级筹码指标预计算 (V10.5, 模式: {mode})...")
        get_stock_info_async = sync_to_async(StockInfo.objects.get, thread_sensitive=True)
        
        # 【代码修改】使 get_latest_metric_async 接受动态模型作为参数
        @sync_to_async(thread_sensitive=True)
        def get_latest_metric_async(model, stock_info_obj):
            try:
                # 【代码修改】使用传入的 model 进行查询
                return model.objects.filter(stock=stock_info_obj).latest('trade_time')
            except model.DoesNotExist: # 【代码修改】捕获特定模型的异常
                return None
        
        # get_data_async 已是通用函数，无需修改
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
        def save_metrics_async(model, stock_info_obj, records_to_create_list, do_delete_first: bool):
            with transaction.atomic():
                if do_delete_first:
                    model.objects.filter(stock=stock_info_obj).delete()
                model.objects.bulk_create(records_to_create_list, batch_size=5000)
        try:
            stock_info = await get_stock_info_async(stock_code=stock_code)
            MetricsModel = get_advanced_chip_metrics_model_by_code(stock_code)
            max_lookback_days = 160
            last_metric_date = None
            if incremental_flag:
                last_metric = await get_latest_metric_async(MetricsModel, stock_info)
                if last_metric:
                    last_metric_date = last_metric.trade_time
                else:
                    logger.info(f"[{stock_code}] 未找到任何历史指标，自动切换到全量刷新模式。")
                    incremental_flag = False
            fetch_start_date = None
            if incremental_flag and last_metric_date:
                fetch_start_date = last_metric_date - timedelta(days=max_lookback_days + 20)
            chip_model = get_cyq_chips_model_by_code(stock_code)
            daily_data_model = get_daily_data_model_by_code(stock_code)
            data_tasks = {
                "cyq_chips": get_data_async(chip_model, stock_info, fields=('trade_time', 'price', 'percent'), start_date=fetch_start_date),
                "daily_data": get_data_async(daily_data_model, stock_info, fields=('trade_time', 'close_qfq', 'vol', 'high_qfq', 'low_qfq'), start_date=fetch_start_date),
                "daily_basic": get_data_async(StockDailyBasic, stock_info, fields=('trade_time', 'float_share'), start_date=fetch_start_date),
            }
            results = await asyncio.gather(*data_tasks.values())
            data_dfs = dict(zip(data_tasks.keys(), results))
            # --- 数据审计部分，逻辑不变 ---
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
            # --- 数据预处理和合并部分，逻辑不变 ---
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
            # --- 核心计算循环，逻辑不变 ---
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
                # logger.info(f"[{stock_code}] 没有需要计算的新指标。任务正常结束。")
                return {"status": "success", "processed_days": 0, "reason": "already up-to-date"}
            
            new_metrics_df = pd.DataFrame(all_metrics_list).set_index('trade_time')
            final_metrics_df = new_metrics_df
            if incremental_flag and last_metric_date:
                # 【代码修改】使用动态模型获取历史指标数据
                past_metrics_df = await get_data_async(
                    MetricsModel, stock_info, 
                    start_date=fetch_start_date
                )
                if not past_metrics_df.empty:
                    past_metrics_df = past_metrics_df.set_index('trade_time')
                    final_metrics_df = pd.concat([past_metrics_df, new_metrics_df]).sort_index()
            
            # --- 指标衍生计算，逻辑不变 ---
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
                concentration_slope_5d = final_metrics_df['concentration_90pct'].rolling(window=5, min_periods=2).apply(
                    lambda x: np.polyfit(range(len(x)), x.dropna(), 1)[0] if len(x.dropna()) > 1 else np.nan, raw=False
                )
                final_metrics_df['concentration_90pct_slope_5d'] = concentration_slope_5d
            
            records_to_save_df = final_metrics_df.loc[new_metrics_df.index]
            records_to_create = []
            # 【代码修改】从动态模型中获取字段信息
            model_fields = {f.name for f in MetricsModel._meta.get_fields()}
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
                    # 【代码修改】使用动态模型创建实例
                    records_to_create.append(MetricsModel(stock=stock_info, trade_time=trade_date, **record_data))
            
            # 【代码修改】调用重构后的函数，传入动态模型
            await save_metrics_async(MetricsModel, stock_info, records_to_create, not incremental_flag)
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

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.migrate_advanced_chip_metrics_chunk', queue='SaveHistoryData_TimeTrade')
def migrate_advanced_chip_metrics_chunk(self, start_id: int, end_id: int, dry_run: bool = False):
    """
    【数据迁移执行器】
    迁移指定ID范围 (chunk) 内的数据。此任务由调度器触发。
    """
    task_id_str = f"Chunk [{start_id}-{end_id-1}]"
    logger.info(f"====== {task_id_str} 执行器任务启动 ======")
    
    try:
        # 1. 查询指定ID范围的数据
        queryset = AdvancedChipMetrics.objects.filter(
            id__gte=start_id,
            id__lt=end_id
        ).select_related('stock')

        if not queryset.exists():
            logger.warning(f"{task_id_str} 在此ID范围内未找到数据，任务正常结束。")
            return {"status": "skipped", "reason": "No data in ID range."}

        # 2. 按目标模型对要创建的实例进行分组
        records_by_model = defaultdict(list)
        for record in queryset:
            stock_code = record.stock.stock_code
            TargetModel = get_advanced_chip_metrics_model_by_code(stock_code)
            
            field_data = {
                field.name: getattr(record, field.name)
                for field in AdvancedChipMetrics._meta.get_fields()
                if not field.is_relation and field.name != 'id'
            }
            
            new_instance = TargetModel(stock=record.stock, **field_data)
            records_by_model[TargetModel].append(new_instance)

        # 3. 对每个目标模型执行批量写入
        if not dry_run:
            for model, instances in records_by_model.items():
                if instances:
                    model.objects.bulk_create(instances, batch_size=2000, ignore_conflicts=True)
                    logger.info(f"{task_id_str} -> 成功向 {model._meta.db_table} 写入 {len(instances)} 条记录。")
        else:
            for model, instances in records_by_model.items():
                if instances:
                    logger.info(f"{task_id_str} -> [演练] 将向 {model._meta.db_table} 写入 {len(instances)} 条记录。")

        logger.info(f"====== {task_id_str} 执行器任务成功完成 ======")
        return {"status": "success", "chunk": f"[{start_id}-{end_id-1}]"}

    except Exception as e:
        logger.error(f"{task_id_str} 迁移过程中发生严重错误: {e}", exc_info=True)
        # 任务失败时重试，最多3次，每次间隔5分钟
        raise self.retry(exc=e, countdown=300, max_retries=3)

#  调度器任务 (Dispatcher Task)
@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.dispatch_advanced_chip_metrics_migration', queue='celery')
def dispatch_advanced_chip_metrics_migration(self, chunk_size: int = 10000, dry_run: bool = False):
    """
    【一次性数据迁移调度器】
    将 stock_advanced_chip_metrics 的迁移工作分解成多个小任务并派发。

    - 这是唯一需要手动调用的任务。
    - 它会根据主键ID将源表数据切分成多个区块。
    - 为每个区块启动一个独立的 'migrate_advanced_chip_metrics_chunk' 任务。

    参数:
    - chunk_size (int): 每个子任务处理的记录数。建议值为 5000 到 20000。
    - dry_run (bool): 如果为 True，则只派发任务并打印日志，子任务也不会实际写入数据库。

    如何执行:
    在 Django Shell 中:
    >>> from celery_tasks.migration_tasks import dispatch_advanced_chip_metrics_migration
    >>> dispatch_advanced_chip_metrics_migration.delay(chunk_size=10000, dry_run=True) # 先进行一次演练
    >>> dispatch_advanced_chip_metrics_migration.delay(chunk_size=10000, dry_run=False) # 确认无误后，正式执行
    """
    start_time = time.time()
    logger.info(f"====== [数据迁移调度器启动] AdvancedChipMetrics -> 分表 ======")
    logger.info(f"参数: chunk_size={chunk_size}, dry_run={dry_run}")

    if dry_run:
        logger.warning("!!! 当前为【演练模式 (dry_run=True)】，将不会对数据库进行任何写入操作。!!!")

    try:
        # 1. 获取源表中ID的最小和最大值，以确定工作范围
        id_range = AdvancedChipMetrics.objects.aggregate(min_id=Min('id'), max_id=Max('id'))
        min_id = id_range.get('min_id')
        max_id = id_range.get('max_id')

        if min_id is None or max_id is None:
            logger.info("源表 stock_advanced_chip_metrics 中没有数据，任务结束。")
            return {"status": "skipped", "reason": "Source table is empty."}

        logger.info(f"检测到需要迁移的数据ID范围为: [{min_id} - {max_id}]。")
        
        # 2. 循环派发子任务
        dispatched_jobs = 0
        for start_id in range(min_id, max_id + 1, chunk_size):
            end_id = start_id + chunk_size
            # 派发执行器任务
            migrate_advanced_chip_metrics_chunk.delay(start_id, end_id, dry_run)
            dispatched_jobs += 1
        
        end_time = time.time()
        duration = end_time - start_time
        logger.info("====== [数据迁移调度器成功完成] ======")
        logger.info(f"总共派发了 {dispatched_jobs} 个迁移子任务。")
        logger.info(f"调度任务耗时: {duration:.2f} 秒。")
        
        return {"status": "success", "dispatched_jobs": dispatched_jobs, "duration_seconds": duration}

    except Exception as e:
        logger.error(f"数据迁移调度过程中发生严重错误: {e}", exc_info=True)
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        raise

# =================================================================
# =================== 3. 回测任务 ==================
# =================================================================

# NEW: 新增的性能分析Celery任务
@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.analyze_performance_for_stock', queue='calculate_strategy')
@with_cache_manager
def analyze_performance_for_stock(self, stock_code: str, start_date: str, end_date: str, *, cache_manager: CacheManager):
    """
    【V1.1 报告生成版】
    对单个股票在指定的历史时间段内，运行策略并对所有买入信号的后续表现进行统计分析。
    - 核心修改: 此任务现在负责接收底层的原始分析数据，并将其格式化为一份完整的、人类可读的报告。
    """
    logger.info("="*80)
    logger.info(f"--- [单股票信号性能分析任务启动] ---")
    logger.info(f"  - 股票代码: {stock_code}")
    logger.info(f"  - 分析时段: {start_date} to {end_date}")
    logger.info("="*80)
    async def main():
        # 1. 初始化总指挥
        strategy_orchestrator = MultiTimeframeTrendStrategy(cache_manager)
        
        # MODIFIED: 调用新的分析方法并接收返回的原始数据
        raw_results = await strategy_orchestrator.analyze_signal_performance_for_period(
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date
        )
        
        # NEW: 新增报告生成逻辑
        if not raw_results:
            logger.info(f"[{stock_code}] 未发现任何可供分析的信号数据。")
        else:
            # 1. 将原始数据转换为DataFrame
            df = pd.DataFrame(raw_results)
            
            # 2. 计算成功率
            df['success_rate'] = (df['successes'] / df['triggers']).where(df['triggers'] > 0, 0)
            
            # 3. 格式化输出列
            df = df.rename(columns={
                'cn_name': '信号名称',
                'type': '类型',
                'triggers': '触发次数',
                'successes': '成功次数'
            })
            df['成功率(%)'] = df['success_rate'].apply(lambda x: f"{x:.1%}")
            
            # 4. 排序并选择最终展示的列
            report_df = df.sort_values(
                by=['类型', 'success_rate', '触发次数'], 
                ascending=[True, False, False]
            )[['信号名称', '类型', '触发次数', '成功次数', '成功率(%)']]
            
            # 5. 打印最终报告
            print("\n\n" + "="*30 + f" [{stock_code} 信号性能分析报告] " + "="*30)
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 120):
                print(report_df.to_string(index=False))
            print("=" * 88 + "\n")

        logger.info(f"--- [单股票信号性能分析任务完成] ---")
        logger.info(f"股票 [{stock_code}] 的信号性能分析执行完毕。")
        logger.info("="*80)
        return {"status": "success", "stock_code": stock_code, "period": f"{start_date}-{end_date}"}
    try:
        return async_to_sync(main)()
    except Exception as e:
        logger.error(f"在执行信号性能分析任务 for {stock_code} 时发生严重错误: {e}", exc_info=True)
        logger.info("="*80)
        return {"status": "error", "stock_code": stock_code, "reason": str(e)}

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.run_global_performance_analysis_v2_map_reduce', queue='celery')
@with_cache_manager
def run_global_performance_analysis_v2_map_reduce(self, start_date: str = None, end_date: str = None, *, cache_manager: CacheManager):
    """
    【新增 V2.0 - MapReduce并行版】
    对全市场所有股票，在指定时间段内，进行真正的并行回测分析。
    - 工作流:
      1. (总管任务) 获取所有股票代码。
      2. (Map) 为每只股票派发一个独立的 `analyze_performance_from_db` 子任务到任务队列。
      3. (Reduce) 使用 Celery Chord，在所有子任务完成后，自动调用 `aggregate_performance_results` 任务来汇总报告。
    """
    logger.info("="*80)
    logger.info(f"--- [全局信号性能扫描 V2.0 - MapReduce版 任务启动] ---")
    logger.info(f"  - 分析时段: {start_date} to {end_date}")
    logger.info("="*80)

    try:
        # 1. 获取全市场股票列表 (这部分仍然是异步的，但由总管任务一次性完成)
        stock_dao = StockBasicInfoDao(cache_manager)
        
        # 在同步的Celery任务中调用异步DAO方法
        all_stocks = async_to_sync(stock_dao.get_stock_list)()

        if not all_stocks:
            logger.error("无法获取股票列表，任务终止。")
            return {"status": "error", "reason": "Failed to get stock list."}
        
        total_stocks = len(all_stocks)
        logger.info(f"获取到 {total_stocks} 只股票，准备派发并行分析子任务...")

        # 2. (Map) 创建所有股票的独立分析子任务签名
        # 我们复用 `analyze_performance_from_db`，因为它已经是为单只股票设计的。
        # 注意：这里我们只创建任务的“签名”(.s())，而不是立即执行它们。
        map_tasks = [
            analyze_performance_from_db.s(
                stock_code=stock.stock_code,
                start_date=start_date,
                end_date=end_date
            ).set(queue='calculate_strategy') for stock in all_stocks
        ]

        # 3. (Reduce) 创建聚合任务的签名
        # 这个任务将在所有 map_tasks 完成后接收它们的结果列表。
        # 【注意】这里我们复用已有的 `aggregate_performance_results` 任务
        reduce_task = aggregate_performance_results.s().set(queue='celery')

        # 4. 使用 `chord` 编排工作流
        # chord(header, body) -> header是一组并行任务，body是它们完成后执行的回调任务
        # 这正是我们需要的 MapReduce 模式！
        workflow = chord(header=group(map_tasks), body=reduce_task)
        
        # 5. 异步执行整个工作流
        workflow.apply_async()

        logger.info(f"成功派发 {total_stocks} 个股票分析子任务。聚合报告将在所有子任务完成后自动生成。")
        logger.info(f"--- [全局信号性能扫描 V2.0 - 任务派发完成] ---")
        return {"status": "workflow_dispatched", "total_stocks": total_stocks}

    except Exception as e:
        logger.error(f"在派发全局性能分析任务时发生严重错误: {e}", exc_info=True)
        return {"status": "error", "reason": str(e)}

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.analyze_performance_for_one_stock', queue='calculate_strategy', acks_late=True)
@with_cache_manager
def analyze_performance_for_one_stock(self, stock_code: str, *, cache_manager: CacheManager):
    """
    【并行计算任务 V1.0 - Map】
    对单只股票进行全历史回测，并返回其所有信号的触发和成功次数。
    """
    logger.info(f"  [Map] 开始处理: {stock_code}")
    try:
        # 动态导入，避免循环依赖和不必要的加载
        from strategies.trend_following.performance_analyzer import PerformanceAnalyzer
        from strategies.trend_following.utils import get_params_block, get_param_value

        # 1. 初始化总指挥并运行策略
        strategy_orchestrator = MultiTimeframeTrendStrategy(cache_manager)
        # 使用async_to_sync在同步的Celery任务中调用异步方法
        async_to_sync(strategy_orchestrator.run_for_stock)(stock_code, latest_only=False)

        # 2. 获取策略运行结果
        df_indicators = strategy_orchestrator.daily_analysis_df
        score_details_df = getattr(strategy_orchestrator.tactical_engine, '_last_score_details_df', pd.DataFrame())
        
        if df_indicators is None or df_indicators.empty or score_details_df.empty:
            logger.warning(f"  [Map] {stock_code} 策略运行后未生成有效数据，跳过。")
            return []

        # 3. 运行分析器并返回结果
        analyzer_params = get_params_block(strategy_orchestrator.tactical_engine, 'performance_analysis_params')
        scoring_params = get_params_block(strategy_orchestrator.tactical_engine, 'four_layer_scoring_params')
        
        analyzer = PerformanceAnalyzer(
            df_indicators=df_indicators,
            score_details_df=score_details_df,
            analysis_params=analyzer_params,
            scoring_params=scoring_params
        )
        # run_analysis 现在返回一个列表
        result = analyzer.run_analysis()
        logger.info(f"  [Map] 完成处理: {stock_code}, 发现 {len(result)} 个有效信号统计。")
        return result

    except Exception as e:
        logger.error(f"  [Map] 处理 {stock_code} 时发生严重错误: {e}", exc_info=True)
        return [] # 返回空列表以保证整个工作流能继续进行

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.aggregate_performance_results', queue='celery')
def aggregate_performance_results(self, results: list):
    """
    【结果聚合任务 V1.0 - Reduce】
    收集所有并行计算任务的结果，进行全局聚合，并打印最终报告。
    """
    logger.info("====== [全局信号性能分析 V1.0] 聚合任务启动 ======")
    logger.info(f"已收到来自 {len(results)} 个并行任务的结果，开始聚合...")

    # 1. 扁平化结果列表
    # results 是一个列表的列表, e.g., [[...], [...], []]
    all_stats = [item for sublist in results if sublist for item in sublist]

    if not all_stats:
        logger.warning("[全局分析] 未能从任何股票中收集到有效的信号统计数据，无法生成报告。")
        return {"status": "finished", "reason": "no data to aggregate"}

    # 2. 聚合数据
    df = pd.DataFrame(all_stats)
    
    # 按信号的原始名称和类型分组，对触发和成功次数求和
    agg_df = df.groupby(['signal_name', 'cn_name', 'type']).agg(
        triggers=('triggers', 'sum'),
        successes=('successes', 'sum')
    ).reset_index()

    # 3. 计算全局成功率
    agg_df['success_rate'] = (agg_df['successes'] / agg_df['triggers']).where(agg_df['triggers'] > 0, 0)

    # 4. 格式化输出
    agg_df = agg_df.rename(columns={
        'cn_name': '信号名称',
        'type': '类型',
        'triggers': '总触发',
        'successes': '总成功',
        'success_rate': '成功率(%)'
    })
    agg_df['成功率(%)'] = agg_df['成功率(%)'].apply(lambda x: f"{x:.1%}")
    
    # 按类型和成功率排序
    final_report_df = agg_df.sort_values(
        by=['类型', '成功率(%)', '总触发'], 
        ascending=[True, False, False]
    )[['信号名称', '类型', '总触发', '总成功', '成功率(%)']]

    # 5. 打印最终报告
    print("\n\n" + "="*35 + " [全市场信号性能终极报告] " + "="*35)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 120):
        print(final_report_df.to_string(index=False))
    print("=" * 95 + "\n")
    
    logger.info("====== [全局信号性能分析 V1.0] 聚合任务完成 ======")
    return {"status": "success", "aggregated_signals": len(final_report_df)}

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.analyze_performance_from_db', queue='calculate_strategy')
@with_cache_manager
def analyze_performance_from_db(self, stock_code: str, start_date: str, end_date: str, *, cache_manager: CacheManager):
    """
    【V1.2 - 安静的Map任务】
    作为MapReduce中的Map阶段，此任务只负责计算并返回原始数据，不打印任何报告。
    - 核心修改: 移除了所有格式化和打印报告的逻辑，以避免在并行执行时产生大量日志噪音。
    """
    async def main():
        # 1. 初始化性能分析服务
        service = PerformanceAnalysisService(cache_manager)
        
        # 2. 调用服务执行分析，并获取原始结果
        raw_results = await service.run_analysis_for_stock(
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date
        )
        
        return raw_results

    try:
        result = async_to_sync(main)()
        if result:
            logger.info(f"[Map] {stock_code} 分析完成，发现 {len(result)} 条信号统计。")
        else:
            logger.info(f"[Map] {stock_code} 分析完成，无有效信号。")
        return result
    except Exception as e:
        logger.error(f"[Map] 在执行DB直读性能分析任务 for {stock_code} 时发生严重错误: {e}", exc_info=True)
        # 返回空列表，确保整个chord工作流不会因单个任务失败而中断
        return []

@celery_app.task(bind=True, name="tasks.stock_analysis_tasks.run_top_n_performance_analysis", queue='calculate_strategy')
def run_top_n_performance_analysis(
    self,
    top_n: int = 3,
    start_date_str: str = '2024-01-01',
    end_date_str: str = '2025-12-31',
    profit_threshold: float = 10.0,
    holding_days: int = 5,
    # 【代码修改】将 strategy_name 的默认值改为 None，使其成为可选参数
    strategy_name: str = None
):
    """
    【Celery任务 V3.2 - 策略可选版】
    分析每日得分最高的N个股票信号的后续表现和成功率。
    - 修正: 将 strategy_name 设为可选。如果不提供，则分析所有策略的Top-N信号。
    - 架构: 复用 utils.model_helpers 中的公共函数，避免代码重复。
    - 入场逻辑: 以信号触发后【下一个交易日】的开盘价作为入场成本。
    """
    # --- 1. 准备阶段 (无变化) ---
    if end_date_str:
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
    else:
        end_date = TradeCalendar.get_latest_trade_date()

    if start_date_str:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
    else:
        start_date = TradeCalendar.get_trade_date_offset(end_date, -30)

    logger.info("\n" + "="*60)
    logger.info(f"=======      Top-{top_n} 信号性能聚焦分析报告      =======")
    logger.info("="*60)
    # 【代码修改】根据 strategy_name 是否提供，动态生成日志信息
    if strategy_name:
        logger.info(f" 分析策略: {strategy_name}")
    else:
        logger.info(" 分析策略: [所有策略]")
    logger.info(f" 分析周期: {start_date} 至 {end_date}")
    logger.info(f" 入场价格: 信号日【次日开盘价】")
    logger.info(f" 成功定义: 入场后 {holding_days} 个交易日内，最高价涨幅达到 {profit_threshold}%")
    logger.info("-"*60)

    # --- 2. 筛选每日Top-N信号 ---
    trade_dates = list(TradeCalendar.get_trade_dates_between(start_date, end_date))
    if not trade_dates:
        logger.warning("在指定范围内未找到任何交易日，任务终止。")
        return

    logger.info("步骤1: 正在从数据库中筛选【每日】Top-N买入信号...")
    top_signals = []
    for trade_date in tqdm(trade_dates, desc="筛选每日信号"):
        # 【代码修改】构建动态查询条件
        filter_kwargs = {
            'trade_date': trade_date,
            'signal_type': '买入信号'
        }
        if strategy_name:
            filter_kwargs['strategy_name'] = strategy_name
        
        # 使用解包的关键字参数进行查询
        daily_top_scores = StrategyDailyScore.objects.filter(**filter_kwargs)\
            .select_related('stock').order_by('-final_score')[:top_n]
        
        top_signals.extend(list(daily_top_scores))

    if not top_signals:
        logger.warning("在指定时间段内未发现任何符合条件的Top-N买入信号，任务终止。")
        return
        
    total_signals = len(top_signals)
    logger.info(f"步骤1完成: 共发现 {total_signals} 个Top-{top_n}信号实例。")

    # --- 3. 高效评估信号表现 (无变化) ---
    logger.info("步骤2: 正在预加载所有相关价格数据以提升效率...")
    
    all_stock_codes = list(set(s.stock.stock_code for s in top_signals))
    # 增加一个健壮性检查
    if not all_stock_codes:
        logger.warning("信号列表为空，无法继续进行价格预加载。")
        return
        
    min_date_needed = start_date
    # 增加对 max_date_needed 的None检查
    max_date_needed_candidate = TradeCalendar.get_trade_date_offset(end_date, holding_days + 2)
    if not max_date_needed_candidate:
        logger.warning(f"无法计算出从 {end_date} 偏移 {holding_days + 2} 天后的交易日，将使用 {end_date} 作为价格数据上限。")
        max_date_needed = end_date
    else:
        max_date_needed = max_date_needed_candidate

    model_to_codes_map = defaultdict(list)
    for code in all_stock_codes:
        model_class = get_daily_data_model_by_code(code)
        model_to_codes_map[model_class].append(code)

    price_map = {}
    for model_class, codes in model_to_codes_map.items():
        daily_data_qs = model_class.objects.filter(
            stock__stock_code__in=codes,
            trade_time__gte=min_date_needed,
            trade_time__lte=max_date_needed
        ).values('stock__stock_code', 'trade_time', 'open_qfq', 'high_qfq', 'close_qfq')

        for item in daily_data_qs:
            key = (item['stock__stock_code'], item['trade_time'])
            price_map[key] = {'open': item['open_qfq'], 'high': item['high_qfq'], 'close': item['close_qfq']}
    
    logger.info(f"价格数据预加载完成，共加载 {len(price_map)} 条价格记录。")
    logger.info("步骤3: 正在评估每个信号的后续表现...")
    
    # --- 评估循环和报告部分 (无变化) ---
    success_count = 0
    evaluated_signals_count = 0
    for signal in tqdm(top_signals, desc="评估信号表现"):
        entry_date = TradeCalendar.get_trade_date_offset(signal.trade_date, 1)
        if not entry_date:
            continue

        entry_day_price_info = price_map.get((signal.stock.stock_code, entry_date))
        if not entry_day_price_info or not entry_day_price_info.get('open'):
            continue

        entry_price = entry_day_price_info['open']
        if not entry_price or entry_price <= 0:
            continue
        
        evaluated_signals_count += 1
        target_price = entry_price * (1 + profit_threshold / 100.0)
        check_dates = TradeCalendar.get_trade_date_offset_list(entry_date, 0, holding_days)

        for check_date in check_dates:
            future_price_info = price_map.get((signal.stock.stock_code, check_date))
            if future_price_info and future_price_info.get('high'):
                if future_price_info['high'] >= target_price:
                    success_count += 1
                    break

    logger.info("步骤3完成: 所有信号评估完毕。")
    logger.info("-"*60)

    success_rate = (success_count / evaluated_signals_count * 100) if evaluated_signals_count > 0 else 0

    logger.info("\n【最终分析结果】")
    logger.info(f"  - 总信号样本数: {total_signals}")
    logger.info(f"  - 有效评估样本数: {evaluated_signals_count}")
    logger.info(f"  - 成功信号数:   {success_count}")
    logger.info(f"  - 聚焦成功率:   {success_rate:.2f}% (基于有效评估样本)")
    logger.info("="*60 + "\n")

    return f"Top-{top_n} 信号性能聚焦分析完成。成功率: {success_rate:.2f}%"


# =================================================================
# =================== 3. 性能复盘任务 (V2.0 MapReduce 架构) ==================
# =================================================================

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.aggregate_performance_results', queue='celery')
@with_cache_manager
def aggregate_performance_results(self, results: list, *, cache_manager: CacheManager):
    """
    【V2.1 序列化修复版 - Reduce 任务】
    - 核心修复: 修正了数据持久化到Redis的逻辑。不再传递预先序列化的JSON字符串，
                而是传递原始的Python对象(List[Dict])，由CacheManager统一负责序列化。
    - 收益: 解决了因“双重序列化”导致数据无法存入Redis的根本性问题。
    """
    logger.info("====== [全局性能分析 V2.1 - Reduce] 聚合任务启动 ======")
    # ... (步骤 1 到 6 的聚合与计算逻辑完全不变) ...
    logger.info(f"已收到来自 {len(results)} 个 Map 任务的结果，开始聚合...")
    all_stats = [item for sublist in results if sublist for item in sublist]
    if not all_stats:
        logger.warning("[全局分析] 未能从任何股票中收集到有效的信号统计数据，无法生成报告。")
        return {"status": "finished", "reason": "no data to aggregate"}
    df = pd.DataFrame(all_stats)
    df['weighted_max_profit'] = df['avg_max_profit_pct'] * df['triggers']
    df['weighted_max_drawdown'] = df['avg_max_drawdown_pct'] * df['triggers']
    df['weighted_exit_days'] = df['avg_exit_days'] * df['triggers']
    agg_df = df.groupby(['signal_name', 'cn_name', 'type']).agg(
        triggers=('triggers', 'sum'),
        successes=('successes', 'sum'),
        total_weighted_profit=('weighted_max_profit', 'sum'),
        total_weighted_drawdown=('weighted_max_drawdown', 'sum'),
        total_weighted_days=('weighted_exit_days', 'sum')
    ).reset_index()
    agg_df['win_rate_pct'] = (agg_df['successes'] / agg_df['triggers']).where(agg_df['triggers'] > 0, 0)
    agg_df['avg_max_profit_pct'] = (agg_df['total_weighted_profit'] / agg_df['triggers']).where(agg_df['triggers'] > 0, 0)
    agg_df['avg_max_drawdown_pct'] = (agg_df['total_weighted_drawdown'] / agg_df['triggers']).where(agg_df['triggers'] > 0, 0)
    agg_df['avg_exit_days'] = (agg_df['total_weighted_days'] / agg_df['triggers']).where(agg_df['triggers'] > 0, 0)
    report_df_for_log = agg_df.sort_values(
        by=['win_rate_pct', 'triggers'], 
        ascending=[False, False]
    )
    final_columns = [
        'cn_name', 'type', 'triggers', 'successes', 
        'win_rate_pct', 'avg_max_profit_pct', 'avg_max_drawdown_pct', 'avg_exit_days'
    ]
    report_df_for_log = report_df_for_log[final_columns]
    
    # --- 步骤 7: 打印到日志 (逻辑不变) ---
    logger.info("\n\n" + "="*35 + " [全市场信号性能终极报告] " + "="*35)
    report_df_for_print = report_df_for_log.copy()
    report_df_for_print.columns = ['信号名称', '类型', '总触发', '总成功', '胜率(%)', '平均最大涨幅(%)', '平均最大回撤(%)', '平均退出天数']
    report_df_for_print['胜率(%)'] = report_df_for_print['胜率(%)'].apply(lambda x: f"{x:.2%}")
    report_df_for_print['平均最大涨幅(%)'] = report_df_for_print['平均最大涨幅(%)'].apply(lambda x: f"{x:.2f}")
    report_df_for_print['平均最大回撤(%)'] = report_df_for_print['平均最大回撤(%)'].apply(lambda x: f"{x:.2f}")
    report_df_for_print['平均退出天数'] = report_df_for_print['平均退出天数'].apply(lambda x: f"{x:.1f}")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 150):
        print(report_df_for_print.to_string(index=False))
    logger.info("=" * 95 + "\n")

    # 8. 【核心修复】将最终报告转换为Python原生对象(List[Dict])，再进行持久化
    #    我们使用未被格式化用于打印的 report_df_for_log，以保留原始的数值类型。
    report_data = report_df_for_log.to_dict(orient='records')
    
    cache_key = "strategy:performance_report:global_v2"
    # 将 report_data (一个Python列表) 传递给 cache_manager，让它处理序列化
    async_to_sync(cache_manager.set)(cache_key, report_data, timeout=60 * 60 * 24 * 7) # 缓存7天

    logger.info(f"终极报告已成功持久化到Redis缓存。Key: '{cache_key}'")

    logger.info("====== [全局性能分析 V2.1 - Reduce] 聚合任务完成 ======")
    return {"status": "success", "aggregated_signals": len(report_df_for_log), "cache_key": cache_key}

# [修改原因] 新增一个独立的、功能强大的Celery任务，用于对所有原子信号进行性能分析。
@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.run_atomic_signal_performance_analysis', queue='celery')
@with_cache_manager
def run_atomic_signal_performance_analysis(self, *, cache_manager: CacheManager):
    """
    【V1.0 全景沙盘推演任务】
    - 核心职责: 1. 从 StrategyDailyState 读取所有原子信号的触发记录。
                2. 对每个信号进行独立的性能回测。
                3. 将分析结果存入 AtomicSignalPerformance 表。
    """
    logger.info("====== [全景沙盘推演 V1.0] 任务启动 ======")
    
    # 实例化性能分析服务
    # 注意：这里我们假设您已将 PerformanceAnalyzer 的逻辑封装到一个服务中
    # 如果没有，可以直接在这里实现查询和分析逻辑
    performance_service = PerformanceAnalysisService(cache_manager)

    async def main():
        try:
            # 1. 调用服务执行核心分析逻辑
            # 服务内部会处理：查询、分组、模拟、聚合
            analysis_results = await performance_service.analyze_all_atomic_signals()

            if not analysis_results:
                logger.warning("[沙盘推演] 未生成任何分析结果。")
                return {"status": "success", "reason": "No results generated."}

            # 2. 准备批量更新/创建的对象
            records_to_update = []
            records_to_create = []
            existing_records = {
                p.signal_name: p for p in await sync_to_async(list)(AtomicSignalPerformance.objects.all())
            }

            for result in analysis_results:
                signal_name = result['signal_name']
                if signal_name in existing_records:
                    # 更新现有记录
                    record = existing_records[signal_name]
                    record.signal_cn_name = result['cn_name']
                    record.signal_type = result['type']
                    record.total_triggers = result['triggers']
                    record.successes = result['successes']
                    record.win_rate_pct = result['win_rate_pct']
                    record.avg_max_profit_pct = result['avg_max_profit_pct']
                    record.avg_max_drawdown_pct = result['avg_max_drawdown_pct']
                    record.avg_exit_days = result['avg_exit_days']
                    records_to_update.append(record)
                else:
                    # 创建新记录
                    records_to_create.append(AtomicSignalPerformance(**{
                        'signal_name': result['signal_name'],
                        'signal_cn_name': result['cn_name'],
                        'signal_type': result['type'],
                        'total_triggers': result['triggers'],
                        'successes': result['successes'],
                        'win_rate_pct': result['win_rate_pct'],
                        'avg_max_profit_pct': result['avg_max_profit_pct'],
                        'avg_max_drawdown_pct': result['avg_max_drawdown_pct'],
                        'avg_exit_days': result['avg_exit_days'],
                    }))
            
            # 3. 批量执行数据库操作
            if records_to_create:
                await sync_to_async(AtomicSignalPerformance.objects.bulk_create)(records_to_create)
                logger.info(f"[沙盘推演] 成功创建 {len(records_to_create)} 条新的信号性能记录。")
            
            if records_to_update:
                await sync_to_async(AtomicSignalPerformance.objects.bulk_update)(
                    records_to_update, 
                    ['signal_cn_name', 'signal_type', 'total_triggers', 'successes', 'win_rate_pct', 
                     'avg_max_profit_pct', 'avg_max_drawdown_pct', 'avg_exit_days', 'last_analyzed']
                )
                logger.info(f"[沙盘推演] 成功更新 {len(records_to_update)} 条已有的信号性能记录。")

            return {"status": "success", "created": len(records_to_create), "updated": len(records_to_update)}

        except Exception as e:
            logger.error(f"[沙盘推演] 任务执行时发生严重错误: {e}", exc_info=True)
            return {"status": "error", "reason": str(e)}

    return async_to_sync(main)()


@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.run_global_performance_analysis', queue='celery')
@with_cache_manager
def run_global_performance_analysis(self, stock_list: list = None, start_date: str = None, end_date: str = None, *, cache_manager: CacheManager):
    """
    【V3.1 联合作战版】
    对全市场（或指定列表）所有股票，在指定时间段内，同时启动【最终信号】和【原子信号】的性能分析。
    - 工作流:
      1. (调度器) 获取所有股票代码。
      2. (联合作战) 同时派发两个独立的分析工作流：
         - 工作流A (MapReduce): 并行分析【最终信号】，并将结果聚合到Redis。
         - 工作流B (独立任务): 分析【原子信号】，并将结果存入数据库功勋墙。
    """
    logger.info("="*80)
    logger.info(f"--- [全局性能分析 V3.1 - 联合作战总指挥启动] ---")
    logger.info(f"  - 分析时段: {start_date or '默认'} to {end_date or '默认'}")
    logger.info("="*80)

    try:
        codes_to_run = stock_list
        if not codes_to_run:
            logger.info("未提供股票列表，将自动获取全市场股票进行复盘...")
            stock_basic_dao = StockBasicInfoDao(cache_manager)
            fav_codes, non_fav_codes = async_to_sync(_get_all_relevant_stock_codes_for_processing)(stock_basic_dao)
            codes_to_run = fav_codes + non_fav_codes

        if not codes_to_run:
            logger.error("无法获取股票列表，任务终止。")
            return {"status": "error", "reason": "Failed to get stock list."}
        
        total_stocks = len(codes_to_run)
        logger.info(f"侦测到 {total_stocks} 个作战目标，准备下达联合作战指令...")

        # --- 作战指令一: 启动【最终信号】分析兵团 (MapReduce) ---
        logger.info("\n--- [指令 1/2] 正在向【最终信号分析兵团】派发 MapReduce 任务...")
        map_tasks = [
            analyze_performance_from_db.s(
                stock_code=code,
                start_date=start_date,
                end_date=end_date
            ).set(queue='calculate_strategy') for code in codes_to_run
        ]
        reduce_task = aggregate_performance_results.s().set(queue='celery')
        workflow = chord(header=group(map_tasks), body=reduce_task)
        workflow.apply_async()
        logger.info(f"-> 指令已下达！{total_stocks} 个【最终信号】分析子任务已派发。")

        # --- 作战指令二: 启动【原子信号】分析特遣队 (独立任务) ---
        logger.info("\n--- [指令 2/2] 正在向【原子信号分析特遣队】派发全景沙盘推演任务...")
        run_atomic_signal_performance_analysis.s().set(queue='celery').apply_async()
        logger.info("-> 指令已下达！【原子信号】全景沙盘推演任务已派发，将独立并行运行。")
        
        logger.info("\n" + "="*80)
        logger.info(f"--- [全局性能分析 V3.1 - 所有作战指令已下达] ---")
        return {"status": "all_workflows_dispatched", "total_stocks": total_stocks}

    except Exception as e:
        logger.error(f"在派发全局性能分析任务时发生严重错误: {e}", exc_info=True)
        return {"status": "error", "reason": str(e)}







