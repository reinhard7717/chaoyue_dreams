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
from utils.model_helpers import get_daily_data_model_by_code, get_cyq_chips_model_by_code, get_advanced_chip_metrics_model_by_code, get_advanced_fund_flow_metrics_model_by_code, get_fund_flow_model_by_code, get_fund_flow_ths_model_by_code, get_fund_flow_dc_model_by_code
from tqdm import tqdm
from services.performance_analysis_service import PerformanceAnalysisService
import numpy as np
import pandas as pd
import pandas_ta as ta
from django.db import transaction, connection
from chaoyue_dreams.celery import app as celery_app
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO
from dao_manager.tushare_daos.industry_dao import IndustryDao
from dao_manager.tushare_daos.strategies_dao import StrategiesDAO
from stock_models.stock_analytics import DailyPositionSnapshot, PositionTracker, StrategyDailyScore, TradingSignal, AtomicSignalPerformance, StrategyDailyState
from stock_models.index import TradeCalendar
from services.contextual_analysis_service import ContextualAnalysisService
from services.chip_feature_calculator import ChipFeatureCalculator
from services.chip_score_calculator import calculate_chip_health_score
from stock_models.stock_basic import StockInfo
from stock_models.time_trade import AdvancedChipMetrics, StockDailyBasic, AdvancedChipMetrics_SZ, AdvancedChipMetrics_SH, AdvancedChipMetrics_CY, AdvancedChipMetrics_KC, AdvancedChipMetrics_BJ, StockCyqPerf
from strategies.multi_timeframe_trend_strategy import MultiTimeframeTrendStrategy
from utils.cache_manager import CacheManager

logger = logging.getLogger('tasks')

async def _get_all_relevant_stock_codes_for_processing(stock_basic_dao: StockBasicInfoDao):
    """
    【V2.0 依赖注入版】
    异步获取所有需要处理的股票代码列表。
    - 不再自己创建DAO，而是接收一个外部传入的DAO实例。
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
    - 使用 @with_cache_manager 装饰器自动管理 CacheManager 生命周期。
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
        
        # 增强日志，以反映新的 start_date_str 参数
        mode_str = "闪电突袭 (仅最新)" if latest_only else "全面战役 (全历史)"
        if latest_only:
            analysis_end_time = f"{trade_date} 16:00:00" if trade_date else None
            # logger.info(f"[{stock_code}] 开始执行核心策略逻辑 ({mode_str}) for date {trade_date}")
        else:
            analysis_end_time = None
            # 在全历史模式下，检查并记录 start_date_str
            if start_date_str:
                # logger.info(f"[{stock_code}] 开始执行核心策略逻辑 ({mode_str})，将从 [{start_date_str}] 开始保存记录。")
                print(f"调试信息 [{stock_code}]: 全历史模式，指定起始日期 {start_date_str}") # 调试输出
            else:
                # logger.info(f"[{stock_code}] 开始执行核心策略逻辑 ({mode_str}) for [全部历史数据]")
                print(f"调试信息 [{stock_code}]: 全历史模式，处理所有数据") # 调试输出

        records_tuple = None # 初始化为 None
        if latest_only:
            # run_for_latest_signal 返回四元组
            records_tuple = await strategy_orchestrator.run_for_latest_signal(
                stock_code=stock_code,
                trade_time=analysis_end_time
            )
        else:
            # run_for_stock 返回四元组
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
        raise

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
        # 根据 start_date_str 参数更新日志
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
        
        # --- 代码将 start_date_str 参数透传给子任务 ---
        # 将调度任务接收到的日期参数，分发给每一个具体的计算任务。
        analysis_tasks = [
            run_multi_timeframe_strategy.s(
                stock_code=code, 
                trade_date=None, 
                latest_only=False,
                start_date_str=start_date_str  # 将参数传递给子任务
            ).set(queue='calculate_strategy') for code in all_codes
        ]
        
        workflow = group(analysis_tasks)
        workflow.apply_async()
        
        
        logger.info(f"[公共数据库] 已成功为 {stock_count} 只股票启动【全历史】分数计算任务。")
        
        # 在返回结果中也包含 start_date_str，方便追踪
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

        # 修改SQL查询逻辑：不再使用HAVING过滤，而是直接获取最新日期及其数据量
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
            # 移除了HAVING子句，因此不再需要传递参数
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
            # 这里的 latest_trade_date 已经是 date 对象，可以直接使用
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
    - 采用“先检查、再创建”的模式，彻底避免因任务重跑导致的 IntegrityError。
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

def _polyfit_slope(series: pd.Series) -> float:
    """
    【新增 V2.0 - 健壮版】一个稳健的辅助函数，用于计算Series的斜率，能正确处理NaN值。
    - 核心修复: 确保传递给 np.polyfit 的 x 和 y 数组长度一致。
    """
    # 1. 移除NaN值，得到干净的数据序列
    cleaned_series = series.dropna()
    # 2. 检查是否有足够的数据点进行线性回归（至少需要2个点）
    if len(cleaned_series) < 2:
        return np.nan
    # 3. 为有效数据点创建对应的x轴坐标（从0开始的整数序列）
    #    这是修复错误的关键：x轴的长度现在与清理后的y轴数据长度完全匹配。
    x_coords = np.arange(len(cleaned_series))
    # 4. 执行线性回归并返回斜率（polyfit返回的第一个系数）
    #    使用 .values 将Pandas Series转换为NumPy数组以获得最佳性能
    slope = np.polyfit(x_coords, cleaned_series.values, 1)[0]
    return slope

def _calculate_slope(series: pd.Series, window: int) -> pd.Series:
    """
    【修改 V3.1 - 增加健壮性】
    - 核心修改: 增加了对 pandas_ta.linreg 返回 None 的防御性检查。
                在极少数情况下（例如输入数据全是NaN或存在其他问题），linreg 可能会返回 None，
                导致 'NoneType' object has no attribute 'iloc' 错误。
                此修改通过捕获 None 返回值并返回一个全为 NaN 的 Series 来防止程序崩溃。
    - 历史修改 (V3.0): 放弃自定义的 rolling().apply() + polyfit 实现，
                直接调用 pandas_ta.linreg() 函数计算斜率。
                这与 indicator_services.py 的实现保持一致，更加健壮和高效。
    - 收益:
        1. 健壮性: 能处理 linreg 返回 None 的偶发性边缘情况，防止任务失败。
        2. 性能: pandas_ta 通常比自定义的 apply 函数性能更好。
        3. 一致性: 统一了项目中斜率的计算标准。
    """
    # 检查series是否为空或窗口是否有效，pandas_ta本身很健壮，但显式检查是好习惯
    if series.empty or window < 2:
        # if window < 2:
            # print(f"DEBUG: 调用 _calculate_slope 时接收到无效的窗口大小 window={window}，该值小于2，无法计算斜率，将返回NaN。")
        return pd.Series(np.nan, index=series.index)
    # 定义计算斜率所需的最小周期数
    min_p = max(2, window // 2)
    # 使用 pandas_ta 的 linreg 函数计算线性回归斜率
    # close=series: 指定要计算的序列
    # length=window: 指定滚动窗口大小
    # min_periods=min_p: 指定最小观测期
    # slope=True: 明确要求返回斜率序列
    # intercept=False, r=False: 不需要截距和R²值，提升性能
    linreg_result = ta.linreg(close=series, length=window, min_periods=min_p, slope=True, intercept=False, r=False)
    # 增加对 linreg_result 为 None 的防御性检查
    if linreg_result is None:
        # 在极少数情况下，如果 linreg 计算失败返回 None，则打印调试信息并返回一个充满 NaN 的序列，以防止程序崩溃
        print(f"DEBUG: pandas_ta.linreg 返回了 None。输入序列长度: {len(series)}, 窗口: {window}。将返回全为 NaN 的序列。")
        # 返回与输入序列索引一致的 NaN 序列
        return pd.Series(np.nan, index=series.index)
    # ta.linreg 返回的是一个DataFrame，我们只需要斜率那一列（通常是第一列或唯一一列）
    # 如果结果是Series，直接使用；如果是DataFrame，取第一列
    slope_series = linreg_result if isinstance(linreg_result, pd.Series) else linreg_result.iloc[:, 0]
    # 返回计算出的斜率序列
    return slope_series

def _load_strategy_config(): # 辅助函数，用于加载策略配置
    """加载并缓存策略配置文件"""
    # 实际应用中，您可能希望使用更健壮的缓存机制
    if not hasattr(_load_strategy_config, "config_cache"):
        from django.conf import settings
        import json
        config_path = settings.BASE_DIR / 'config' / 'trend_follow_strategy.json'
        with open(config_path, 'r', encoding='utf-8') as f:
            _load_strategy_config.config_cache = json.load(f)
    return _load_strategy_config.config_cache

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.schedule_precompute_advanced_chips', queue='celery')
@with_cache_manager
def schedule_precompute_advanced_chips(self, *, cache_manager: CacheManager):
    """
    【V2.1 装饰器重构版】
    """
    try:
        # logger.info("开始调度 [高级筹码指标预计算] 任务...")
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
        # logger.info(f"找到 {stock_count} 只股票待进行高级筹码预计算。")
        for stock_code in all_codes:
            precompute_advanced_chips_for_stock.s(stock_code).set(queue='SaveHistoryData_TimeTrade').apply_async()
            precompute_advanced_fund_flow_for_stock.s(stock_code).set(queue='SaveHistoryData_TimeTrade').apply_async()
        # logger.info(f"已为 {stock_count} 只股票调度 '高级筹码指标预计算' 任务。")
        return {"status": "started", "stock_count": stock_count}
    except Exception as e:
        logger.error(f"调度高级筹码预计算任务时出错: {e}", exc_info=True)
        return {"status": "failed", "reason": str(e)}

async def _initialize_task_context(stock_code: str, is_incremental: bool, max_lookback_days: int):
    """【辅助函数 V1.0】初始化任务上下文，获取模型、确定数据加载范围。"""
    # print(f"[{stock_code}] [初始化] 正在准备任务上下文...")
    get_stock_info_async = sync_to_async(StockInfo.objects.get, thread_sensitive=True)
    stock_info = await get_stock_info_async(stock_code=stock_code)
    MetricsModel = get_advanced_chip_metrics_model_by_code(stock_code)
    last_metric_date = None
    if is_incremental:
        @sync_to_async(thread_sensitive=True)
        def get_latest_metric_async(model, stock_info_obj):
            try:
                return model.objects.filter(stock=stock_info_obj).latest('trade_time')
            except model.DoesNotExist:
                return None
        last_metric = await get_latest_metric_async(MetricsModel, stock_info)
        if last_metric:
            last_metric_date = last_metric.trade_time
        else:
            is_incremental = False # 如果没有历史数据，则转为全量模式
    fetch_start_date = None
    if is_incremental and last_metric_date:
        fetch_start_date = last_metric_date - timedelta(days=max_lookback_days + 20)
    # print(f"[{stock_code}] [初始化] 上下文准备完毕。模式: {'增量' if is_incremental else '全量'}, 数据追溯起点: {fetch_start_date}")
    return stock_info, MetricsModel, is_incremental, last_metric_date, fetch_start_date

async def _load_and_audit_data_sources(stock_info, fetch_start_date):
    """【辅助函数 V1.1 - 审计逻辑优化】异步加载所有原始数据源，并进行严格的数据审计。"""
    # print(f"[{stock_info.stock_code}] [数据加载与审计] 开始加载所有数据源...") # 调试信息
    @sync_to_async(thread_sensitive=True)
    def get_data_async(model, stock_info_obj, fields: tuple = None, date_field='trade_time', start_date=None):
        qs = model.objects.filter(stock=stock_info_obj)
        if start_date:
            filter_kwargs = {f'{date_field}__gte': start_date}
            qs = qs.filter(**filter_kwargs)
        # 增加对数据是否存在的整体检查
        if not qs.exists():
            print(f"DEBUG: 数据源模型 {model.__name__} 中未找到股票 {stock_info_obj.stock_code} 的数据。")
            return pd.DataFrame() # 返回空的DataFrame
        return pd.DataFrame.from_records(qs.values(*fields) if fields else qs.values())
    chip_model = get_cyq_chips_model_by_code(stock_info.stock_code)
    daily_data_model = get_daily_data_model_by_code(stock_info.stock_code)
    data_tasks = {
        "cyq_chips": get_data_async(chip_model, stock_info, fields=('trade_time', 'price', 'percent'), start_date=fetch_start_date),
        "daily_data": get_data_async(daily_data_model, stock_info, fields=('trade_time', 'close_qfq', 'vol', 'high_qfq', 'low_qfq'), start_date=fetch_start_date),
        "daily_basic": get_data_async(StockDailyBasic, stock_info, fields=('trade_time', 'float_share'), start_date=fetch_start_date),
        "cyq_perf": get_data_async(StockCyqPerf, stock_info, start_date=fetch_start_date),
    }
    results = await asyncio.gather(*data_tasks.values())
    data_dfs = dict(zip(data_tasks.keys(), results))
    # --- 数据审计 ---
    # 【代码修改】调整审计逻辑，使其更加健壮
    # 核心思想：只在关键数据源完全缺失时才中断，对于日期不匹配问题只记录警告，让后续的inner join处理。
    cyq_chips_df = data_dfs.get("cyq_chips")
    if cyq_chips_df is None or cyq_chips_df.empty:
        raise ValueError("[审计失败] 核心数据源 'cyq_chips' 为空或加载失败！任务无法继续。")
    if data_dfs.get("cyq_perf") is None or data_dfs.get("cyq_perf").empty:
        raise ValueError("[审计失败] 关键数据源 'cyq_perf' 为空或加载失败！任务无法继续。")
    # 检查其他关键数据源是否完全为空
    if data_dfs.get("daily_data") is None or data_dfs.get("daily_data").empty:
        raise ValueError("[审计失败] 关键数据源 'daily_data' 为空或加载失败！任务无法继续。")
    if data_dfs.get("daily_basic") is None or data_dfs.get("daily_basic").empty:
        raise ValueError("[审计失败] 关键数据源 'daily_basic' 为空或加载失败！任务无法继续。")
    cyq_chips_df['trade_time'] = pd.to_datetime(cyq_chips_df['trade_time'])
    master_dates = set(cyq_chips_df['trade_time'].dt.date.unique())
    audit_warnings = []
    for name, df in data_dfs.items():
        if name == "cyq_chips": continue
        # 此处已在上方检查过df是否为空，这里再次确认以防万一
        if df is None or df.empty:
            # 此处理论上不会触发，因为已在前面检查过
            raise ValueError(f"[审计失败] 关键数据源 '{name}' 为空！")
        df['trade_time'] = pd.to_datetime(df['trade_time'])
        source_dates = set(df['trade_time'].dt.date.unique())
        missing_in_source = sorted(list(master_dates - source_dates))
        if missing_in_source:
            warning_msg = (f"数据源 '{name}' 相对核心数据源 'cyq_chips' 缺失 {len(missing_in_source)} 个交易日的数据。示例: {missing_in_source[:5]}...")
            audit_warnings.append(warning_msg)
    daily_data_df = data_dfs['daily_data']
    required_cols_in_daily = ['close_qfq', 'vol', 'high_qfq', 'low_qfq']
    if daily_data_df[required_cols_in_daily].isnull().values.any():
        problematic_rows = daily_data_df[daily_data_df[required_cols_in_daily].isnull().any(axis=1)]
        for _, row in problematic_rows.iterrows():
            missing_fields = [col for col in required_cols_in_daily if pd.isna(row[col])]
            warning_msg = f"在日期 {row['trade_time'].date()} 的行情数据中发现NULL值。缺失字段: {missing_fields}"
            audit_warnings.append(warning_msg)
    # 【代码修改】仅记录警告，不因此中断任务
    if audit_warnings:
        full_warning_message = f"[{stock_info.stock_code}] [审计警告] 数据一致性检查发现非致命问题，任务将继续执行。详情如下：\n" + "\n".join(audit_warnings)
        logger.warning(full_warning_message) # 使用 warning 级别日志
        print(full_warning_message) # 调试信息
    # print(f"[{stock_info.stock_code}] [数据加载与审计] 所有数据源加载并审计通过。") # 【代码修改】调整了消息文本
    return data_dfs

def _preprocess_and_merge_data(stock_code: str, data_dfs: dict) -> pd.DataFrame:
    """
    【辅助函数 V1.1 - 性能优化版】对加载的数据进行预处理和合并。
    - 核心优化: 优化了 'prev_20d_close' 的计算逻辑，使用 .map() 代替 .merge()，以提高处理效率和降低内存消耗。
    - 代码整洁: 移除了重复的空DataFrame检查。
    """
    # print(f"[{stock_code}] [数据预处理] 开始预处理与合并数据...")
    # 提取并预处理各数据源
    cyq_chips_data = data_dfs['cyq_chips']
    cyq_chips_data['trade_time'] = pd.to_datetime(cyq_chips_data['trade_time']).dt.date
    # 检查并标准化 'percent' 列的单位
    daily_sums = cyq_chips_data.groupby('trade_time')['percent'].transform('sum')
    mask_sum_to_one = np.isclose(daily_sums, 1.0, atol=0.1)
    if mask_sum_to_one.any():
        # 将以小数形式存储的百分比（如0.01）统一转换为百分比数值（如1.0）
        cyq_chips_data.loc[mask_sum_to_one, 'percent'] *= 100
    # 预处理每日行情数据
    daily_data = data_dfs['daily_data']
    daily_data['trade_time'] = pd.to_datetime(daily_data['trade_time']).dt.date
    daily_data['daily_turnover_volume'] = daily_data['vol'] * 100 # 'vol'单位是手，乘以100得到股
    daily_data = daily_data.rename(columns={'close_qfq': 'close_price', 'high_qfq': 'high_price', 'low_qfq': 'low_price'})
    # 预处理每日基本面数据
    daily_basic_data = data_dfs['daily_basic']
    daily_basic_data['trade_time'] = pd.to_datetime(daily_basic_data['trade_time']).dt.date
    daily_basic_data['total_chip_volume'] = daily_basic_data['float_share'] * 10000 # 'float_share'单位是万股
    daily_basic_data = daily_basic_data.drop(columns=['float_share'])
    # 预处理CYQ性能数据
    cyq_perf_data = data_dfs['cyq_perf']
    cyq_perf_data['trade_time'] = pd.to_datetime(cyq_perf_data['trade_time']).dt.date
    # 核心合并逻辑：使用内连接(inner join)确保所有数据源在同一天都存在数据
    merged_df = pd.merge(cyq_chips_data, daily_data, on='trade_time', how='inner')
    merged_df = pd.merge(merged_df, daily_basic_data, on='trade_time', how='inner')
    merged_df = pd.merge(merged_df, cyq_perf_data, on='trade_time', how='inner')
    # 检查合并后结果是否为空
    if merged_df.empty:
        raise ValueError("数据源在进行内连接(inner join)后结果为空，请检查各数据源的日期是否存在有效交集。")
    # 按交易时间排序并重置索引
    merged_df = merged_df.sort_values('trade_time').reset_index(drop=True)
    # 【代码修改】优化 prev_20d_close 的计算，避免使用 merge，提高效率
    # 1. 创建一个从 trade_time 到 close_price 的映射，并去除每日重复数据
    daily_close_map = merged_df.drop_duplicates('trade_time').set_index('trade_time')['close_price']
    # 2. 将收盘价序列向前移动20个周期，得到每个交易日对应的20天前的收盘价
    prev_20d_close_series = daily_close_map.shift(20)
    # 3. 使用 map 方法将计算出的20日前收盘价高效地映射回主 DataFrame
    #    对于这种单列查找的场景，map 通常比 merge 更快，内存占用更低。
    merged_df['prev_20d_close'] = merged_df['trade_time'].map(prev_20d_close_series)
    # print(f"[{stock_code}] [数据预处理] 数据合并完成，生成 {len(merged_df)} 行记录。")
    return merged_df

def _calculate_base_chip_metrics(merged_df: pd.DataFrame, is_incremental: bool, last_metric_date) -> pd.DataFrame:
    """【辅助函数 V1.0】逐日计算基础筹码指标。"""
    stock_code = merged_df['stock_code'].iloc[0] if 'stock_code' in merged_df.columns else 'UNKNOWN'
    # print(f"[{stock_code}] [基础指标计算] 开始逐日计算基础筹码指标...") # 调试信息
    all_metrics_list = []
    grouped_data = merged_df.groupby('trade_time')
    for trade_date, daily_full_df in grouped_data:
        if is_incremental and last_metric_date and trade_date <= last_metric_date:
            continue
        context_data = daily_full_df.iloc[0].to_dict()
        chip_data_for_calc = daily_full_df[['price', 'percent']]
        if chip_data_for_calc.empty:
            continue
        # context_data['weight_avg_cost'] = 0
        calculator = ChipFeatureCalculator(chip_data_for_calc.sort_values(by='price'), context_data)
        daily_metrics = calculator.calculate_all_metrics()
        if daily_metrics:
            daily_metrics['trade_time'] = trade_date
            daily_metrics['prev_20d_close'] = context_data.get('prev_20d_close')
            all_metrics_list.append(daily_metrics)
    if not all_metrics_list:
        # 【代码修改】调整了消息文本并增加了print
        message = f"[{stock_code}] [基础指标计算] 无新的交易日数据需要计算，任务提前结束。"
        logger.info(message)
        print(message)
        return pd.DataFrame()
    new_metrics_df = pd.DataFrame(all_metrics_list).set_index('trade_time')
    # print(f"[{stock_code}] [基础指标计算] 完成，共计算了 {len(new_metrics_df)} 个新交易日的基础指标。") # 【代码修改】调整了消息文本
    return new_metrics_df

async def _calculate_derivative_metrics(stock_info, final_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """【辅助函数 V1.3 - 健壮性与调试增强】自动化计算所有斜率、加速度和健康分等衍生指标。"""
    stock_code = stock_info.stock_code
    # 在函数入口处增加整体存在性检查，如果DataFrame为空则直接跳过所有计算
    if final_metrics_df.empty:
        print(f"[{stock_code}] [衍生指标计算] 传入的DataFrame为空，跳过衍生指标计算。")
        return final_metrics_df
    # print(f"[{stock_code}] [衍生指标计算] 开始自动化三阶段衍生计算...")
    MetricsModel = get_advanced_chip_metrics_model_by_code(stock_code)
    # 解决 'float' 和 'decimal.Decimal' TypeError 的关键步骤
    # 从数据库加载的数据(Decimal)与新计算的数据(float)合并后，列会变成object类型。
    # pandas_ta等数值计算库无法处理Decimal类型，因此在计算前必须将所有数值列统一转换为float64。
    # print(f"[{stock_code}] DEBUG: 衍生计算前，开始将DataFrame中的object/Decimal类型列转换为float64...") # 【代码修改】调整了消息文本
    for col in final_metrics_df.columns:
        if final_metrics_df[col].dtype == 'object':
            final_metrics_df[col] = pd.to_numeric(final_metrics_df[col], errors='coerce')
            # print(f"[{stock_code}] DEBUG: 列 '{col}' 已从 object 转换为 numeric。")
    # 阶段一：计算所有非健康分的基础及衍生指标
    # print(f"[{stock_code}] [阶段一] 计算基础指标和非健康分的衍生指标...") # 【代码修改】调整了消息文本
    if 'avg_cost_short_term' in final_metrics_df.columns and 'avg_cost_long_term' in final_metrics_df.columns:
        final_metrics_df['cost_divergence'] = final_metrics_df['avg_cost_short_term'] - final_metrics_df['avg_cost_long_term']
    model_fields = {f.name for f in MetricsModel._meta.get_fields()}
    # 自动化计算斜率
    for field_name in model_fields:
        if '_slope_' in field_name and 'chip_health_score' not in field_name:
            base_col, period_str = field_name.split('_slope_')
            period = int(period_str.replace('d', ''))
            calc_window = 2 if period == 1 else period
            if base_col in final_metrics_df.columns and field_name not in final_metrics_df.columns:
                # print(f"[{stock_code}]   -> 正在计算斜率: {field_name} (基于 {base_col}, 窗口 {calc_window})")
                final_metrics_df[field_name] = _calculate_slope(final_metrics_df[base_col], calc_window)
    # 自动化计算加速度
    for field_name in model_fields:
        if '_accel_' in field_name and 'chip_health_score' not in field_name:
            base_col_with_slope, period_str = field_name.split('_accel_')
            period = int(period_str.replace('d', ''))
            source_slope_col = f"{base_col_with_slope}_slope_{period}d"
            calc_window = 2 if period == 1 else period
            if source_slope_col in final_metrics_df.columns and field_name not in final_metrics_df.columns:
                # print(f"[{stock_code}]   -> 正在计算加速度: {field_name} (基于 {source_slope_col}, 窗口 {calc_window})")
                final_metrics_df[field_name] = _calculate_slope(final_metrics_df[source_slope_col], calc_window)
    # 阶段二：计算最终版的筹码健康分
    # print(f"[{stock_code}] [阶段二] 计算筹码健康分...") # 【代码修改】调整了消息文本
    health_score_dependencies = [
        'concentration_90pct', 'concentration_90pct_slope_5d',
        'winner_profit_margin', 'price_to_peak_ratio'
    ]
    # print(f"  - [健康分依赖检查] 正在检查并填充依赖列: {health_score_dependencies}")
    for dep_col in health_score_dependencies:
        if dep_col not in final_metrics_df.columns:
            print(f"    - [警告] 健康分依赖列 '{dep_col}' 不存在，将创建并填充为0。")
            final_metrics_df[dep_col] = 0
        else:
            final_metrics_df[dep_col] = final_metrics_df[dep_col].fillna(0)
    final_metrics_df['chip_health_score'] = final_metrics_df.apply(calculate_chip_health_score, axis=1)
    # 阶段三：基于最终版的健康分，计算其衍生指标
    # print(f"[{stock_code}] [阶段三] 计算筹码健康分的衍生指标...") # 【代码修改】调整了消息文本
    if 'chip_health_score' in final_metrics_df.columns:
        for field_name in model_fields:
            if 'chip_health_score_' in field_name:
                if '_slope_' in field_name:
                    period = int(field_name.split('_slope_')[1].replace('d', ''))
                    calc_window = 2 if period == 1 else period
                    final_metrics_df[field_name] = _calculate_slope(final_metrics_df['chip_health_score'], calc_window)
                elif '_accel_' in field_name:
                    period = int(field_name.split('_accel_')[1].replace('d', ''))
                    source_slope_col = f"chip_health_score_slope_{period}d"
                    calc_window = 2 if period == 1 else period
                    if source_slope_col in final_metrics_df.columns:
                        final_metrics_df[field_name] = _calculate_slope(final_metrics_df[source_slope_col], calc_window)
    # print(f"[{stock_code}] [衍生指标计算] 所有衍生指标计算完成。")
    return final_metrics_df

async def _prepare_and_save_data(stock_info, MetricsModel, final_df: pd.DataFrame, new_df_index, is_full_refresh: bool):
    """
    【辅助函数 V1.1 - 性能优化版】准备并保存最终计算结果到数据库。
    - 核心优化: 重构了数据行到模型实例的转换逻辑。放弃使用性能较低的 .iterrows()，
                改为效率更高的 .to_dict('index') 方法。此方法一次性将DataFrame转换为字典，
                后续迭代在原生Python字典上进行，显著提升了对象创建的速度。
    """
    stock_code = stock_info.stock_code
    # print(f"[{stock_code}] [数据保存] 开始准备并保存数据...")
    # 筛选出需要保存的新计算或更新的记录
    records_to_save_df = final_df.loc[new_df_index]
    records_to_create = []
    # 获取模型所有字段名，用于后续数据匹配
    model_fields = {f.name for f in MetricsModel._meta.get_fields()}
    # 【代码修改】优化记录创建过程，避免使用iterrows()，以大幅提高效率
    # .to_dict('index') 将DataFrame转换为 {index -> {column -> value}} 的字典格式。
    # 迭代这个字典的.items()，可以直接获取索引(trade_date)和行数据(row_data字典)，
    # 这比 .iterrows() 每次迭代都生成一个Series对象要快得多。
    for trade_date, row_data in records_to_save_df.to_dict('index').items():
        record_data = {}
        # 遍历模型字段，从行数据字典中提取对应的值
        for field_name in model_fields:
            if field_name in row_data:
                value = row_data[field_name]
                # 对不同类型的值进行处理，确保符合数据库要求
                if isinstance(value, float) and not np.isfinite(value):
                    # 将无穷大或NaN浮点数转换成None
                    record_data[field_name] = None
                elif pd.isna(value):
                    # 将Pandas的NA值转换成None
                    record_data[field_name] = None
                elif isinstance(value, float):
                    # 为保证精度，将浮点数转换为Decimal类型。
                    # 对极小值直接置零，避免不必要的精度问题。
                    record_data[field_name] = Decimal(str(round(value, 8))) if abs(value) > 1e-10 else Decimal('0.0')
                else:
                    # 其他类型的值直接使用
                    record_data[field_name] = value
        # 移除字典中不属于模型构造函数的键
        record_data.pop('id', None)
        record_data.pop('stock', None)
        # 如果记录数据非空，则创建模型实例
        if record_data:
            records_to_create.append(MetricsModel(stock=stock_info, trade_time=trade_date, **record_data))
    # 异步保存数据到数据库
    @sync_to_async(thread_sensitive=True)
    def save_metrics_async(model, stock_info_obj, records_to_create_list, do_delete_first: bool):
        # 使用数据库事务保证数据一致性
        with transaction.atomic():
            if do_delete_first:
                # 如果是全量刷新，先删除旧数据
                model.objects.filter(stock=stock_info_obj).delete()
            # 批量创建新记录，提高数据库写入效率
            model.objects.bulk_create(records_to_create_list, batch_size=5000)
    # 执行异步保存
    await save_metrics_async(MetricsModel, stock_info, records_to_create, is_full_refresh)
    print(f"[{stock_code}] [数据保存] 成功为 {len(records_to_create)} 个交易日存储了高级筹码指标。")
    return len(records_to_create)

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.precompute_advanced_chips_for_stock', queue='SaveHistoryData_TimeTrade')
@with_cache_manager
def precompute_advanced_chips_for_stock(self, stock_code: str, is_incremental: bool = True, *, cache_manager: CacheManager):
    """
    【执行器 V15.0 - 职责拆分重构版】
    - 核心重构 (V15.0): 将原先臃肿的函数拆分为多个职责单一的辅助函数，主任务函数负责流程编排。
    - 流程:
        1. _initialize_task_context: 初始化上下文。
        2. _load_and_audit_data_sources: 加载并审计数据。
        3. _preprocess_and_merge_data: 预处理并合并数据。
        4. _calculate_base_chip_metrics: 计算基础筹码指标。
        5. _calculate_derivative_metrics: 自动化计算所有衍生指标。
        6. _prepare_and_save_data: 准备并保存数据。
    """
    async def main(incremental_flag: bool):
        try:
            # 1. 初始化
            max_lookback_days = 160
            stock_info, MetricsModel, is_incremental_final, last_metric_date, fetch_start_date = await _initialize_task_context(
                stock_code, incremental_flag, max_lookback_days
            )
            # 2. 加载和审计数据
            data_dfs = await _load_and_audit_data_sources(stock_info, fetch_start_date)
            # 3. 预处理和合并
            merged_df = _preprocess_and_merge_data(stock_code, data_dfs)
            # 4. 计算基础指标
            new_metrics_df = _calculate_base_chip_metrics(merged_df, is_incremental_final, last_metric_date)
            if new_metrics_df.empty:
                return {"status": "success", "processed_days": 0, "reason": "already up-to-date or no new data"}
            # 在拼接(concat)和排序(sort_index)前，必须确保两者类型一致。
            if not isinstance(new_metrics_df.index, pd.DatetimeIndex):
                # print(f"[{stock_code}] DEBUG: 正在将 new_metrics_df 的索引从 {type(new_metrics_df.index)} 转换为 pd.DatetimeIndex...")
                new_metrics_df.index = pd.to_datetime(new_metrics_df.index)
            # 5. 准备用于衍生计算的完整DataFrame
            final_metrics_df = new_metrics_df
            if is_incremental_final and last_metric_date:
                @sync_to_async(thread_sensitive=True)
                def get_past_data_async(model, s_info, start_date):
                    qs = model.objects.filter(stock=s_info, trade_time__gte=start_date)
                    # 增加数据存在性检查
                    if not qs.exists():
                        return pd.DataFrame()
                    return pd.DataFrame.from_records(qs.values())
                past_metrics_df = await get_past_data_async(MetricsModel, stock_info, fetch_start_date)
                if not past_metrics_df.empty:
                    past_metrics_df = past_metrics_df.set_index('trade_time')
                    if not isinstance(past_metrics_df.index, pd.DatetimeIndex):
                        past_metrics_df.index = pd.to_datetime(past_metrics_df.index)
                    final_metrics_df = pd.concat([past_metrics_df, new_metrics_df]).sort_index()
                    final_metrics_df = final_metrics_df[~final_metrics_df.index.duplicated(keep='last')]
            # 6. 计算衍生指标
            final_metrics_df = await _calculate_derivative_metrics(stock_info, final_metrics_df)
            # 7. 准备并保存数据
            processed_days = await _prepare_and_save_data(
                stock_info, MetricsModel, final_metrics_df, new_metrics_df.index, not is_incremental_final
            )
            mode = "增量更新" if is_incremental_final else "全量刷新"
            logger.info(f"[{stock_code}] 成功！模式[{mode}]下，为 {processed_days} 个交易日计算并存储了高级筹码指标。")
            return {"status": "success", "processed_days": processed_days}
        except StockInfo.DoesNotExist:
            logger.error(f"[{stock_code}] 在StockInfo中找不到该股票，任务终止。")
            return {"status": "failed", "reason": "stock_code not found in StockInfo"}
        except ValueError as ve:
            logger.error(f"[{stock_code}] 高级筹码指标预计算失败 (数据问题): {ve}", exc_info=False) # 数据问题不打印完整堆栈
            return {"status": "failed", "reason": str(ve)}
        except Exception as e:
            logger.error(f"[{stock_code}] 高级筹码指标预计算失败 (未知异常): {e}", exc_info=True)
            return {"status": "failed", "reason": str(e)}
    try:
        result = async_to_sync(main)(is_incremental)
        return result
    except Exception as e:
        logger.error(f"--- CATCHING EXCEPTION in precompute_advanced_chips_for_stock for {stock_code}: {e}", exc_info=True)
        raise


# =================================================================
# =================== 3. 高级资金特征任务 ==================
# =================================================================

async def _initialize_ff_task_context(stock_code: str, is_incremental: bool):
    """【资金流辅助函数 V1.0】初始化任务上下文。"""
    # print(f"[{stock_code}] [资金流-初始化] 正在准备任务上下文...")
    stock_info = await sync_to_async(StockInfo.objects.get)(stock_code=stock_code)
    MetricsModel = get_advanced_fund_flow_metrics_model_by_code(stock_code)
    max_lookback_days = 200  # 需要比最长周期144天更长
    last_metric_date = None
    if is_incremental:
        @sync_to_async(thread_sensitive=True)
        def get_latest_metric_async(model, stock_info_obj):
            try:
                return model.objects.filter(stock=stock_info_obj).latest('trade_time')
            except model.DoesNotExist:
                return None
        latest_metric = await get_latest_metric_async(MetricsModel, stock_info)
        if latest_metric:
            last_metric_date = latest_metric.trade_time
        else:
            is_incremental = False
    fetch_start_date = None
    if is_incremental and last_metric_date:
        fetch_start_date = last_metric_date - timedelta(days=max_lookback_days)
    # print(f"[{stock_code}] [资金流-初始化] 上下文准备完毕。模式: {'增量' if is_incremental else '全量'}, 数据追溯起点: {fetch_start_date}")
    return stock_info, MetricsModel, is_incremental, last_metric_date, fetch_start_date

async def _load_and_merge_fund_flow_sources(stock_info, fetch_start_date):
    """【资金流辅助函数 V1.3 · 字段来源修正版】加载、标准化并合并多源资金流数据。"""
    # print(f"[{stock_info.stock_code}] [资金流-数据加载] 开始加载多源数据...")
    @sync_to_async(thread_sensitive=True)
    def get_data_async(model, stock_info_obj, fields: tuple = None, date_field='trade_time', start_date=None):
        if not model: return pd.DataFrame()
        qs = model.objects.filter(stock=stock_info_obj)
        if start_date:
            qs = qs.filter(**{f'{date_field}__gte': start_date})
        return pd.DataFrame.from_records(qs.values(*fields) if fields else qs.values())
    data_tasks = {
        "tushare": get_data_async(get_fund_flow_model_by_code(stock_info.stock_code), stock_info, start_date=fetch_start_date),
        "ths": get_data_async(get_fund_flow_ths_model_by_code(stock_info.stock_code), stock_info, start_date=fetch_start_date),
        "dc": get_data_async(get_fund_flow_dc_model_by_code(stock_info.stock_code), stock_info, start_date=fetch_start_date),
        # 修正 daily 源的查询字段，移除不存在的 'trade_count'
        "daily": get_data_async(get_daily_data_model_by_code(stock_info.stock_code), stock_info, fields=('trade_time', 'amount'), start_date=fetch_start_date),
    }
    results = await asyncio.gather(*data_tasks.values())
    data_dfs = dict(zip(data_tasks.keys(), results))
    def standardize_and_prepare(df: pd.DataFrame, source: str) -> pd.DataFrame:
        if df.empty: return df
        for col in df.columns:
            if 'amount' in col or 'net' in col or 'trade_count' in col or 'vol' in col:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if source == 'tushare':
            df['main_force_net_flow_tushare'] = df['buy_lg_amount'] + df['buy_elg_amount'] - df['sell_lg_amount'] - df['sell_elg_amount']
            df['retail_net_flow_tushare'] = df['buy_sm_amount'] + df['buy_md_amount'] - df['sell_sm_amount'] - df['sell_md_amount']
            df['net_xl_amount_tushare'] = df['buy_elg_amount'] - df['sell_elg_amount']
            df['net_lg_amount_tushare'] = df['buy_lg_amount'] - df['sell_lg_amount']
            df['net_md_amount_tushare'] = df['buy_md_amount'] - df['sell_md_amount']
            df['net_sh_amount_tushare'] = df['buy_sm_amount'] - df['sell_sm_amount']
            df['main_force_active_buy_tushare'] = df['buy_lg_amount'] + df['buy_elg_amount']
            df['main_force_active_sell_tushare'] = df['sell_lg_amount'] + df['sell_elg_amount']
            # 在返回的列中加入 'trade_count'，确保其从 tushare 源传递下去
            required_cols = [
                'trade_time', 'net_mf_amount', 'main_force_net_flow_tushare', 
                'retail_net_flow_tushare', 'net_xl_amount_tushare', 'net_lg_amount_tushare', 
                'net_md_amount_tushare', 'net_sh_amount_tushare', 'main_force_active_buy_tushare', 
                'main_force_active_sell_tushare', 'buy_sm_amount', 'sell_sm_amount', 
                'buy_md_amount', 'sell_md_amount', 'buy_lg_amount', 'sell_lg_amount',
                'buy_elg_amount', 'sell_elg_amount', 'trade_count'
            ]
            return df[required_cols].rename(columns={'net_mf_amount': 'net_flow_tushare'})
        elif source == 'ths':
            df['retail_net_flow_ths'] = df['buy_sm_amount'] + df['buy_md_amount']
            return df[['trade_time', 'net_amount', 'buy_lg_amount', 'retail_net_flow_ths']].rename(columns={'net_amount': 'net_flow_ths', 'buy_lg_amount': 'main_force_net_flow_ths'})
        elif source == 'dc':
            df['main_force_net_flow_dc'] = df['buy_elg_amount'] + df['buy_lg_amount']
            df['retail_net_flow_dc'] = df['buy_sm_amount'] + df['buy_md_amount']
            return df[['trade_time', 'net_amount', 'main_force_net_flow_dc', 'retail_net_flow_dc', 'buy_elg_amount']].rename(columns={'net_amount': 'net_flow_dc', 'buy_elg_amount': 'net_xl_amount_dc'})
        return df
    df_tushare = standardize_and_prepare(data_dfs['tushare'], 'tushare')
    df_ths = standardize_and_prepare(data_dfs['ths'], 'ths')
    df_dc = standardize_and_prepare(data_dfs['dc'], 'dc')
    dfs_to_merge = [df for df in [df_tushare, df_ths, df_dc] if not df.empty]
    if not dfs_to_merge:
        raise ValueError("所有资金流数据源均为空")
    merged_df = dfs_to_merge[0]
    for df_to_merge in dfs_to_merge[1:]:
        merged_df = pd.merge(merged_df, df_to_merge, on='trade_time', how='outer')
    merged_df['trade_time'] = pd.to_datetime(merged_df['trade_time'])
    merged_df = merged_df.sort_values('trade_time').set_index('trade_time')
    df_daily = data_dfs['daily']
    if not df_daily.empty:
        df_daily['trade_time'] = pd.to_datetime(df_daily['trade_time'])
        df_daily['amount'] = pd.to_numeric(df_daily['amount'], errors='coerce')
        # 修正 join 逻辑，daily 数据源只提供 amount
        merged_df = merged_df.join(df_daily.set_index('trade_time')['amount'])
    return merged_df

def _calculate_consensus_and_base_metrics(stock_code: str, merged_df: pd.DataFrame) -> pd.DataFrame:
    """【资金流辅助函数 V1.5 · 缺失源告警增强版】计算共识指标和基础比率。"""
    # print(f"[{stock_code}] [资金流-共识计算] 开始计算共识指标...")
    df = merged_df.copy()
    # --- 1. 计算共识资金流 (修正版，增加缺失数据源告警) ---
    # 修改开始: 增加对缺失列的显式打印告警
    print(f"[{stock_code}] [资金流-共识计算] 开始计算共识指标... 可用列: {df.columns.tolist()}")
    # 定义各共识指标及其可能的源列
    consensus_map = {
        'net_flow_consensus': ['net_flow_tushare', 'net_flow_ths', 'net_flow_dc'],
        'main_force_net_flow_consensus': ['main_force_net_flow_tushare', 'main_force_net_flow_ths', 'main_force_net_flow_dc'],
        'retail_net_flow_consensus': ['retail_net_flow_tushare', 'retail_net_flow_ths', 'retail_net_flow_dc'],
        'net_xl_amount_consensus': ['net_xl_amount_tushare', 'net_xl_amount_dc'],
        'net_lg_amount_consensus': ['net_lg_amount_tushare'],
        'net_md_amount_consensus': ['net_md_amount_tushare'],
        'net_sh_amount_consensus': ['net_sh_amount_tushare'],
    }
    # 动态地对存在的列求均值，并对缺失的列进行告警
    for target_col, source_cols in consensus_map.items():
        # 找出当前DataFrame中实际存在的源列
        existing_cols = [col for col in source_cols if col in df.columns]
        # 检查是否有源列缺失
        if len(existing_cols) < len(source_cols):
            missing_cols = list(set(source_cols) - set(existing_cols))
            if not existing_cols:
                # 严重告警：所有源列都缺失，该指标无法计算
                print(f"[{stock_code}] [共识计算错误] 目标 '{target_col}': 所有源列 {source_cols} 均缺失，指标将为NaN。")
            else:
                # 普通告警：部分源列缺失，使用可用列进行计算
                print(f"[{stock_code}] [共识计算警告] 目标 '{target_col}': 缺少数据源列: {missing_cols}。将使用可用列 {existing_cols} 计算。")
        # 核心计算逻辑
        if existing_cols:
            # 如果至少存在一个源列，则计算均值
            df[target_col] = df[existing_cols].mean(axis=1)
        else:
            # 如果所有源列都不存在，则将目标列填充为NaN
            df[target_col] = np.nan
    # 修改结束
    # --- 2. 计算基础衍生指标 ---
    df['flow_divergence_mf_vs_retail'] = df['main_force_net_flow_consensus'] - df['retail_net_flow_consensus']
    df['main_force_vs_xl_divergence'] = df['main_force_net_flow_consensus'] - df['net_xl_amount_consensus']
    # --- 3. 计算高级比率指标 (修正逻辑) ---
    safe_denom = lambda v: v.replace(0, np.nan)
    fill_na_safe = lambda series, fill_val: series.fillna(fill_val)
    main_force_buy = df.get('main_force_active_buy_tushare', pd.Series(dtype=float))
    retail_sell = df.get('sell_sm_amount', pd.Series(dtype=float)).fillna(0) + df.get('sell_md_amount', pd.Series(dtype=float)).fillna(0)
    df['active_buy_pressure'] = fill_na_safe((main_force_buy / safe_denom(retail_sell)), 0.5)
    retail_sell_amount = df.get('sell_sm_amount', pd.Series(dtype=float)).fillna(0) + df.get('sell_md_amount', pd.Series(dtype=float)).fillna(0)
    retail_buy_amount = df.get('buy_sm_amount', pd.Series(dtype=float)).fillna(0) + df.get('buy_md_amount', pd.Series(dtype=float)).fillna(0)
    df['retail_panic_index'] = fill_na_safe((retail_sell_amount / safe_denom(retail_buy_amount)), 1.0)
    buy_elg = df.get('buy_elg_amount', pd.Series(dtype=float))
    sell_elg = df.get('sell_elg_amount', pd.Series(dtype=float))
    df['main_force_conviction_buy_ratio'] = fill_na_safe((buy_elg / safe_denom(buy_elg + sell_elg)), 0.5)
    total_elg_trade = df.get('buy_elg_amount', pd.Series(dtype=float)).fillna(0) + df.get('sell_elg_amount', pd.Series(dtype=float)).fillna(0)
    total_trade_amount = df.get('amount', pd.Series(dtype=float)) # amount单位是千元
    df['trade_concentration_index'] = fill_na_safe((total_elg_trade * 10 / safe_denom(total_trade_amount)), 0.0) # 乘以10将万元转换为千元
    buy_lg = df.get('buy_lg_amount', pd.Series(dtype=float))
    df['main_force_conviction_ratio'] = fill_na_safe((buy_elg / safe_denom(buy_lg)), 0)
    total_active_flow = df.get('main_force_active_buy_tushare', pd.Series(dtype=float)).fillna(0) + df.get('main_force_active_sell_tushare', pd.Series(dtype=float)).fillna(0)
    df['main_force_flow_intensity_ratio'] = fill_na_safe((df.get('main_force_active_buy_tushare', pd.Series(dtype=float)) / safe_denom(total_active_flow)), 0.5)
    if 'amount' in df.columns and not df['amount'].isnull().all():
        valid_amount = df['amount'].astype(float).replace(0, np.nan)
        df['main_force_buy_rate_consensus'] = (df['main_force_net_flow_consensus'] * 10 / valid_amount) * 100 # 乘以10将万元转换为千元
    # --- 4. 修正 avg_order_value 计算 ---
    # 修正单位换算：'amount' 单位是千元，应乘以 1000 得到元。
    total_turnover_yuan = df.get('amount', pd.Series(dtype=float)).fillna(0) * 1000
    trade_count = df.get('trade_count', pd.Series(dtype=float))
    # 创建一个分母 Series，将0和NaN替换为NaN以进行安全除法
    denominator = trade_count.replace(0, np.nan)
    # 使用 .loc 避免 SettingWithCopyWarning，并填充最终的NaN为0
    df.loc[:, 'avg_order_value'] = (total_turnover_yuan / denominator).fillna(0)
    return df

def _calculate_standardized_derivatives(stock_code: str, consensus_df: pd.DataFrame) -> pd.DataFrame:
    """【资金流辅助函数 V1.3 · pandas_ta 语法修正版】使用标准化周期计算所有衍生指标。"""
    # print(f"[{stock_code}] [资金流-衍生计算] 开始标准化衍生计算...")
    final_df = consensus_df.copy()
    
    CORE_METRICS_TO_DERIVE = [
        'net_flow_consensus',
        'main_force_net_flow_consensus',
        'retail_net_flow_consensus',
        'net_xl_amount_consensus',
        'net_lg_amount_consensus',
        'net_md_amount_consensus',
        'net_sh_amount_consensus',
        'flow_divergence_mf_vs_retail',
        'main_force_flow_intensity_ratio',
        'main_force_vs_xl_divergence',
        'active_buy_pressure',
        'retail_panic_index',
        'main_force_conviction_buy_ratio',
        'trade_concentration_index',
        'avg_order_value',
        'main_force_conviction_ratio',
    ]
    
    UNIFIED_PERIODS = [1, 5, 13, 21, 55]
    
    for p in UNIFIED_PERIODS:
        calc_window = 2 if p == 1 else p
        
        if p > 1:
            sum_cols = [
                'net_flow_consensus', 'main_force_net_flow_consensus', 'retail_net_flow_consensus',
                'net_xl_amount_consensus', 'net_lg_amount_consensus', 'net_md_amount_consensus',
                'net_sh_amount_consensus', 'active_buy_pressure', 'retail_panic_index', 'avg_order_value'
            ]
            for col in sum_cols:
                if col in final_df.columns:
                    final_df[f'{col}_sum_{p}d'] = final_df[col].rolling(window=p, min_periods=max(2, p // 2)).sum()

        for col in CORE_METRICS_TO_DERIVE:
            if col in final_df.columns:
                # 修正 pandas_ta 调用语法，在 DataFrame 上调用 .ta，并通过 close 参数传递 Series
                final_df[f'{col}_slope_{p}d'] = final_df.ta.slope(close=final_df[col], length=calc_window)
        
        if p > 1:
            sum_slope_cols = [
                'net_flow_consensus', 'main_force_net_flow_consensus', 'retail_net_flow_consensus',
                'net_xl_amount_consensus', 'net_lg_amount_consensus', 'net_md_amount_consensus',
                'net_sh_amount_consensus', 'active_buy_pressure', 'retail_panic_index', 'avg_order_value'
            ]
            for col in sum_slope_cols:
                sum_col_name = f'{col}_sum_{p}d'
                if sum_col_name in final_df.columns:
                    # 修正 pandas_ta 调用语法
                    final_df[f'{sum_col_name}_slope_{p}d'] = final_df.ta.slope(close=final_df[sum_col_name], length=p)

        for col in CORE_METRICS_TO_DERIVE:
            source_slope_col = f'{col}_slope_{p}d'
            if source_slope_col in final_df.columns:
                # 修正 pandas_ta 调用语法
                final_df[f'{col}_accel_{p}d'] = final_df.ta.slope(close=final_df[source_slope_col], length=calc_window)
                
    return final_df

async def _prepare_and_save_ff_data(stock_info, MetricsModel, final_df: pd.DataFrame, is_incremental: bool, last_metric_date):
    """【资金流辅助函数 V1.0】准备并保存最终计算结果到数据库。"""
    stock_code = stock_info.stock_code
    # print(f"[{stock_code}] [资金流-数据保存] 开始准备并保存数据...")
    if is_incremental and last_metric_date:
        records_to_save_df = final_df[final_df.index.date > last_metric_date]
    else:
        records_to_save_df = final_df
    if records_to_save_df.empty:
        print(f"[{stock_code}] 数据已是最新，无需更新。")
        return 0
    records_to_create = []
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
                else:
                    record_data[field_name] = value
        record_data.pop('id', None)
        record_data.pop('stock', None)
        if record_data:
            records_to_create.append(MetricsModel(stock=stock_info, trade_time=trade_date.date(), **record_data))
    @sync_to_async(thread_sensitive=True)
    def save_metrics_async(model, stock_info_obj, records_to_create_list, do_delete_first: bool):
        with transaction.atomic():
            if do_delete_first:
                model.objects.filter(stock=stock_info_obj).delete()
            model.objects.bulk_create(records_to_create_list, batch_size=2000)
    await save_metrics_async(MetricsModel, stock_info, records_to_create, not is_incremental)
    # print(f"[{stock_code}] [资金流-数据保存] 成功为 {len(records_to_create)} 个交易日存储了高级资金流指标。")
    return len(records_to_create)

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.precompute_advanced_fund_flow_for_stock', queue='SaveHistoryData_TimeTrade')
def precompute_advanced_fund_flow_for_stock(self, stock_code: str, is_incremental: bool = True):
    """
    【执行器 V16.0 - 职责拆分重构版】
    - 核心重构 (V16.0): 将原先臃肿的函数拆分为多个职责单一的辅助函数，主任务函数负责流程编排。
    - 流程:
        1. _initialize_ff_task_context: 初始化上下文。
        2. _load_and_merge_fund_flow_sources: 加载并合并多源数据。
        3. _calculate_consensus_and_base_metrics: 计算共识指标。
        4. _calculate_standardized_derivatives: 标准化计算所有衍生指标。
        5. _prepare_and_save_ff_data: 准备并保存数据。
    """
    async def main(incremental_flag: bool):
        try:
            # 1. 初始化
            stock_info, MetricsModel, is_incremental_final, last_metric_date, fetch_start_date = await _initialize_ff_task_context(
                stock_code, incremental_flag
            )
            # 2. 加载和合并数据
            merged_df = await _load_and_merge_fund_flow_sources(stock_info, fetch_start_date)
            # 3. 计算共识指标
            consensus_df = _calculate_consensus_and_base_metrics(stock_code, merged_df)
            # 4. 计算衍生指标
            final_metrics_df = _calculate_standardized_derivatives(stock_code, consensus_df)
            # 5. 准备并保存数据
            processed_days = await _prepare_and_save_ff_data(
                stock_info, MetricsModel, final_metrics_df, is_incremental_final, last_metric_date
            )
            mode = "增量更新" if is_incremental_final else "全量刷新"
            logger.info(f"[{stock_code}] 成功！模式[{mode}]下，为 {processed_days} 个交易日计算并存储了高级资金流指标。")
            return {"status": "success", "processed_days": processed_days}
        except StockInfo.DoesNotExist:
            logger.error(f"[{stock_code}] 在StockInfo中找不到该股票，任务终止。")
            return {"status": "failed", "reason": "stock_code not found in StockInfo"}
        except ValueError as ve:
            logger.error(f"[{stock_code}] 高级资金流指标预计算失败 (数据问题): {ve}", exc_info=False)
            return {"status": "failed", "reason": str(ve)}
        except Exception as e:
            logger.error(f"[{stock_code}] 高级资金流指标预计算失败 (未知异常): {e}", exc_info=True)
            return {"status": "failed", "reason": str(e)}
    try:
        result = async_to_sync(main)(is_incremental)
        return result
    except Exception as e:
        logger.error(f"--- CATCHING EXCEPTION in precompute_advanced_fund_flow_for_stock for {stock_code}: {e}", exc_info=True)
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
# =================== 行业轮动预计算 ==================
# =================================================================
@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.precompute_industry_lifecycle', queue='celery')
@with_cache_manager
def precompute_industry_lifecycle(self, trade_date_str: str = None, *, cache_manager: CacheManager):
    """
    【V3.1 修复版】每日行业生命周期预计算任务 (调度器)
    - 核心升级: 为每个数据源 ('sw', 'ths', 'dc') 创建一个独立的并行计算工作流。
    - 修复: 修正了获取历史回溯日期范围的逻辑错误。
    """
    logger.info("====== [调度器 V3.1] 行业生命周期多源并行计算任务启动 ======")
    # 1. 确定分析的目标日期
    if trade_date_str:
        target_date = datetime.strptime(trade_date_str, '%Y-%m-%d').date()
    else:
        target_date = TradeCalendar.get_latest_trade_date()
        if not target_date:
            logger.error("无法获取最新交易日，任务终止。")
            return {"status": "failed", "reason": "Cannot get latest trade date."}
    target_date_str_formatted = target_date.strftime('%Y-%m-%d')
    logger.info(f"分析目标日期: {target_date_str_formatted}")
    # 2. 确定需要计算的历史日期范围
    config = _load_strategy_config()
    lookback_days = config.get('feature_engineering_params', {}).get('industry_context_params', {}).get('lookback_days', 21)
    print(f"DEBUG: 调用 TradeCalendar.get_latest_n_trade_dates, 参数: n={lookback_days}, reference_date={target_date}")
    trade_dates_needed = TradeCalendar.get_latest_n_trade_dates(lookback_days, target_date)
    if not trade_dates_needed or len(trade_dates_needed) < lookback_days:
        # 增加对获取到的日期数量的检查，确保数据完整性
        logger.error(f"无法为目标日期 {target_date_str_formatted} 获取足够的回溯交易日 (需要 {lookback_days} 天，实际获取 {len(trade_dates_needed) if trade_dates_needed else 0} 天)，任务终止。")
        return {"status": "failed", "reason": "Could not get enough historical trade dates."}
    # --- 为每个数据源启动一个工作流 ---
    sources_to_process = ['sw', 'ths', 'dc']
    dispatched_workflows = []
    for source in sources_to_process:
        logger.info(f"\n--- [调度器] 正在为来源 '{source.upper()}' 派发工作流 ---")
        # 3. (Map) 创建该来源的并行计算任务签名
        map_tasks = [
            calculate_strength_rank_for_date.s(
                trade_date_str=day.strftime('%Y-%m-%d'),
                source=source  # 传递 source 参数
            ).set(queue='SaveHistoryData_TimeTrade') for day in trade_dates_needed
        ]
        # 4. (Reduce) 创建该来源的聚合回调任务签名
        reduce_task = aggregate_and_save_lifecycle_data.s(
            target_date_str=target_date_str_formatted,
            source=source  # 传递 source 参数
        ).set(queue='celery')
        # 5. 使用 chord 编排并执行工作流
        workflow = chord(header=group(map_tasks), body=reduce_task)
        workflow.apply_async()
        dispatched_workflows.append(source)
        logger.info(f"--- [调度器] 来源 '{source.upper()}' 的工作流已成功派发。 ---")
    logger.info(f"====== [调度器] 所有工作流派发完成，涉及来源: {dispatched_workflows} ======")
    return {"status": "workflows_dispatched", "target_date": target_date_str_formatted, "sources": dispatched_workflows}

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.calculate_strength_rank_for_date', queue='SaveHistoryData_TimeTrade')
@with_cache_manager
def calculate_strength_rank_for_date(self, trade_date_str: str, source: str, *, cache_manager: CacheManager):
    """
    【V3.0-Map任务】计算单个指定日期、指定来源的行业强度排名。
    - 返回值: 返回一个包含日期和序列化DataFrame的字典，便于Reduce任务处理。
    """
    async def main():
        trade_date = datetime.strptime(trade_date_str, '%Y-%m-%d').date()
        logger.info(f"  [Map] 开始计算 {trade_date} (来源: {source.upper()}) 的行业强度排名...")
        
        context_service = ContextualAnalysisService(cache_manager)
        # 调用正确的、带 source 参数的方法
        rank_df = await context_service.calculate_industry_strength_rank(trade_date, source=source)
        
        if rank_df.empty:
            logger.warning(f"  [Map] {trade_date} (来源: {source.upper()}) 的排名计算结果为空。")
            return None
            
        # 返回一个结构化的字典，而不是纯JSON字符串
        return {
            "trade_date": trade_date_str,
            "source": source,
            "rank_data_json": rank_df.to_json(orient='split')
        }

    try:
        return async_to_sync(main)()
    except Exception as e:
        logger.error(f"  [Map] 计算 {trade_date_str} (来源: {source}) 排名时失败: {e}", exc_info=True)
        raise

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.aggregate_and_save_lifecycle_data', queue='celery')
@with_cache_manager
def aggregate_and_save_lifecycle_data(self, results: list, target_date_str: str, source: str, *, cache_manager: CacheManager):
    """
    【V3.2-Reduce任务-深度分析版】聚合每日排名结果，计算并保存最终的生命周期数据。
    - 核心升级: 1. 聚合了新增的 breadth_score 和 leader_score。
                 2. 引入了使用新维度的生命周期判定逻辑。
                 3. 将所有新维度数据保存到数据库。
    """
    async def main():
        logger.info(f"====== [Reduce V3.2] 聚合任务启动，目标日期: {target_date_str}, 来源: {source.upper()} ======")
        # 1. 反序列化并合并所有每日排名数据
        all_ranks_df = []
        for i, result_dict in enumerate(results):
            if result_dict and 'rank_data_json' in result_dict:
                try:
                    df = pd.read_json(result_dict['rank_data_json'], orient='split')
                    df['trade_date'] = pd.to_datetime(result_dict['trade_date'])
                    all_ranks_df.append(df)
                except Exception as e:
                    logger.error(f"[Reduce] 反序列化第 {i} 个结果时失败: {e}")
                    continue
        if not all_ranks_df:
            logger.error(f"[Reduce] 来源 '{source.upper()}' 的所有Map任务均未返回有效数据，聚合任务终止。")
            return {"status": "failed", "reason": "No data from map tasks."}
        rotation_df = pd.concat(all_ranks_df, ignore_index=True)
        print(f"DEBUG: [Reduce] 合并后的 rotation_df 行数: {len(rotation_df)}, 列: {rotation_df.columns.tolist()}")

        # 2. 执行生命周期计算
        def calculate_lifecycle_metrics(group):
            group = group.sort_values('trade_date')
            # 即使数据不足，也要返回最新一天的广度和龙头分
            if len(group) < 5:
                latest_row = group.iloc[-1]
                return pd.Series({
                    'latest_rank': latest_row['strength_rank'],
                    'rank_slope': 0.0,
                    'rank_accel': 0.0,
                    'latest_breadth': latest_row.get('breadth_score', 0.0), # 新增
                    'latest_leader': latest_row.get('leader_score', 0.0)   # 新增
                })
            ranks = group['strength_rank'].values
            slope = np.polyfit(np.arange(min(5, len(ranks))), ranks[-5:], 1)[0] if len(ranks) >= 2 else 0.0
            accel = 0.0
            if len(group) >= 10:
                slope_1 = np.polyfit(np.arange(5), ranks[-10:-5], 1)[0]
                slope_2 = np.polyfit(np.arange(5), ranks[-5:], 1)[0]
                accel = slope_2 - slope_1
            latest_row = group.iloc[-1]
            return pd.Series({
                'latest_rank': ranks[-1],
                'rank_slope': slope,
                'rank_accel': accel,
                'latest_breadth': latest_row.get('breadth_score', 0.0), # 新增
                'latest_leader': latest_row.get('leader_score', 0.0)   # 新增
            })
        lifecycle_metrics = rotation_df.groupby('concept_code').apply(calculate_lifecycle_metrics)
        def assign_lifecycle_stage(row):
            # 定义“初升期” (PREHEAT): 排名低位 + 排名趋势和加速度向上 + 出现龙头效应
            is_preheat = (
                row['latest_rank'] < 0.4 and
                row['rank_slope'] > 0.008 and
                row['rank_accel'] > 0.001 and
                row['latest_leader'] > 0.2 # 关键条件：必须出现首板或更强的龙头
            )
            if is_preheat:
                return 'PREHEAT'
            # 定义“主升段” (MARKUP): 排名靠前 + 趋势强劲 + 广度健康
            is_markup = (
                row['latest_rank'] > 0.6 and
                row['rank_slope'] > 0.01 and
                row['latest_breadth'] > 0.5 # 关键条件：板块内至少一半个股上涨
            )
            if is_markup:
                return 'MARKUP'
            # 沿用旧的滞涨和下跌定义
            if row['latest_rank'] > 0.8 and row['rank_slope'] < 0: return 'STAGNATION'
            if row['latest_rank'] < 0.4 and row['rank_slope'] < -0.005: return 'DOWNTREND'
            return 'TRANSITION'
        lifecycle_metrics['lifecycle_stage'] = lifecycle_metrics.apply(assign_lifecycle_stage, axis=1)

        # 3. 准备并保存目标日期的数据
        target_date = datetime.strptime(target_date_str, '%Y-%m-%d').date()
        latest_day_data = lifecycle_metrics.copy()
        latest_day_data['trade_date'] = target_date

        # 确保 IndustryLifecycle 模型已添加 breadth_score 和 leader_score 字段
        latest_day_data.rename(columns={
            'latest_rank': 'strength_rank',
            'latest_breadth': 'breadth_score',
            'latest_leader': 'leader_score'
        }, inplace=True)

        records_to_save = latest_day_data.reset_index().to_dict('records')

        # 4. 调用DAO保存
        industry_dao = IndustryDao(cache_manager)
        save_result = await industry_dao.save_industry_lifecycle(records_to_save)
        logger.info(f"====== [Reduce] 聚合任务完成，已为 {target_date_str} (来源: {source.upper()}) 保存 {len(records_to_save)} 条行业生命周期数据。 ======")
        return save_result
    try:
        return async_to_sync(main)()
    except Exception as e:
        logger.error(f"[Reduce] 聚合任务 (来源: {source}) 失败: {e}", exc_info=True)
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
    - 此任务现在负责接收底层的原始分析数据，并将其格式化为一份完整的、人类可读的报告。
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
            atomic_states=strategy_orchestrator.tactical_engine.atomic_states,
            trigger_events=strategy_orchestrator.tactical_engine.trigger_events,
            playbook_states=strategy_orchestrator.tactical_engine.playbook_states,
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
    【V2.0 指标感知版 - Reduce】
    - 核心升级: 能够识别 'metric_name' 字段，并根据其值为 'risk' 信号
                显示“风险规避率”，为其他信号显示“成功率”，解决了指标混淆问题。
    """
    logger.info("====== [全局信号性能分析 V2.0 - Reduce] 聚合任务启动 ======")
    logger.info(f"已收到来自 {len(results)} 个并行任务的结果，开始聚合...")

    all_stats = [item for sublist in results if sublist for item in sublist]
    if not all_stats:
        logger.warning("[全局分析] 未能从任何股票中收集到有效的信号统计数据，无法生成报告。")
        return {"status": "finished", "reason": "no data to aggregate"}

    df = pd.DataFrame(all_stats)

    agg_df = df.groupby(['signal_name', 'signal_cn_name', 'signal_type', 'metric_name']).agg(
        triggers=('total_triggers', 'sum'),
        successes=('successes', 'sum')
    ).reset_index()

    # 效能指标的计算保持不变，因为它现在是标准化的
    agg_df['effectiveness_pct'] = (agg_df['successes'] / agg_df['triggers']).where(agg_df['triggers'] > 0, 0)
    # 特别处理风险信号的效能指标
    is_risk = agg_df['signal_type'] == 'risk'
    agg_df.loc[is_risk, 'effectiveness_pct'] = 1 - agg_df.loc[is_risk, 'effectiveness_pct']

    # 格式化输出
    agg_df = agg_df.rename(columns={
        'signal_cn_name': '信号名称',
        'signal_type': '类型',
        'triggers': '总触发',
        'successes': '总成功'
    })
    def format_effectiveness(row):
        value = row['effectiveness_pct']
        return f"{value:.1%}"

    agg_df['效能指标(%)'] = agg_df.apply(format_effectiveness, axis=1)
    agg_df['指标类型'] = np.where(agg_df['类型'] == 'risk', '风险规避率', '成功率')
    
    # 按效能指标排序
    final_report_df = agg_df.sort_values(
        by=['类型', 'effectiveness_pct', '总触发'], 
        ascending=[True, False, False]
    )[['信号名称', '类型', '总触发', '总成功', '指标类型', '效能指标(%)']]

    # 打印最终报告
    print("\n\n" + "="*35 + " [全市场信号性能终极报告 V2.0] " + "="*35)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 120):
        print(final_report_df.to_string(index=False))
    print("=" * 95 + "\n")
    
    logger.info("====== [全局信号性能分析 V2.0 - Reduce] 聚合任务完成 ======")
    return {"status": "success", "aggregated_signals": len(final_report_df)}

@celery_app.task(bind=True, name="tasks.stock_analysis_tasks.run_top_n_performance_analysis", queue='calculate_strategy')
def run_top_n_performance_analysis(
    self,
    top_n: int = 3,
    start_date_str: str = '2024-01-01',
    end_date_str: str = '2025-12-31',
    profit_threshold: float = 10.0,
    holding_days: int = 5,
    # 将 strategy_name 的默认值改为 None，使其成为可选参数
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
    # 根据 strategy_name 是否提供，动态生成日志信息
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
        # 构建动态查询条件
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
    
    # --- 步骤 7: 打印到日志 ---
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

    # 8. 将最终报告转换为Python原生对象(List[Dict])，再进行持久化
    #    我们使用未被格式化用于打印的 report_df_for_log，以保留原始的数值类型。
    report_data = report_df_for_log.to_dict(orient='records')
    
    cache_key = "strategy:performance_report:global_v2"
    # 将 report_data (一个Python列表) 传递给 cache_manager，让它处理序列化
    async_to_sync(cache_manager.set)(cache_key, report_data, timeout=60 * 60 * 24 * 7) # 缓存7天

    logger.info(f"终极报告已成功持久化到Redis缓存。Key: '{cache_key}'")

    logger.info("====== [全局性能分析 V2.1 - Reduce] 聚合任务完成 ======")
    return {"status": "success", "aggregated_signals": len(report_df_for_log), "cache_key": cache_key}

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.analyze_atomic_signals_for_stock', queue='calculate_strategy')
@with_cache_manager
def analyze_atomic_signals_for_stock(self, stock_code: str, *, cache_manager: CacheManager):
    """
    【V1.0 新增 - Map 任务】分析单只股票的所有原子信号性能。
    这个任务是计算密集型的，会被大量并行执行。
    """
    logger.info(f"  -> [原子信号 Map] 开始分析股票: {stock_code}")
    try:
        # 假设 PerformanceAnalysisService 有一个方法可以处理单只股票
        performance_service = PerformanceAnalysisService(cache_manager)
        # 这个新方法需要您在 PerformanceAnalysisService 中实现，其逻辑是
        # 从 analyze_all_atomic_signals 中提取处理单只股票的部分。
        stock_results = async_to_sync(performance_service.analyze_atomic_signals_for_single_stock)(stock_code)
        logger.info(f"  -- [原子信号 Map] 完成分析: {stock_code}，发现 {len(stock_results)} 个有效信号。")
        return stock_results
    except Exception as e:
        logger.error(f"  !! [原子信号 Map] 分析股票 {stock_code} 时出错: {e}", exc_info=True)
        return [] # 出错时返回空列表，不影响主流程

# [核心新增] 2. 原子信号分析的 REDUCE 任务
@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.aggregate_atomic_signal_results', queue='celery')
@with_cache_manager
def aggregate_atomic_signal_results(self, results: list, *, cache_manager: CacheManager):
    """
    【V1.2 同步写入修复版 - Reduce 任务】
    - 核心修复: 移除了对同步ORM方法不必要的、且错误的 sync_to_async 包装，
                确保数据库写入操作被真正执行。
    """
    logger.info("====== [原子信号分析 V1.2 - Reduce] 聚合任务启动 ======")
    
    try:
        # 1. 聚合计算逻辑 (保持不变)
        all_stats = [item for sublist in results if sublist for item in sublist]
        if not all_stats:
            logger.warning("[原子信号 Reduce] 未能从任何股票中收集到有效的原子信号统计数据。")
            return {"status": "finished", "reason": "no atomic data to aggregate"}

        df = pd.DataFrame(all_stats)
        agg_df = df.groupby(['signal_name', 'cn_name', 'type']).agg(
            total_triggers=('triggers', 'sum'),
            total_successes=('successes', 'sum'),
            weighted_profit=('weighted_max_profit', 'sum'),
            weighted_drawdown=('weighted_max_drawdown', 'sum'),
            weighted_days=('weighted_exit_days', 'sum')
        ).reset_index()

        agg_df['win_rate_pct'] = (agg_df['total_successes'] / agg_df['total_triggers']).where(agg_df['total_triggers'] > 0, 0)
        agg_df['avg_max_profit_pct'] = (agg_df['weighted_profit'] / agg_df['total_triggers']).where(agg_df['total_triggers'] > 0, 0)
        agg_df['avg_max_drawdown_pct'] = (agg_df['weighted_drawdown'] / agg_df['total_triggers']).where(agg_df['total_triggers'] > 0, 0)
        agg_df['avg_exit_days'] = (agg_df['weighted_days'] / agg_df['total_triggers']).where(agg_df['total_triggers'] > 0, 0)

        logger.info(f"[原子信号 Reduce] 成功聚合了 {len(agg_df)} 个独特的原子信号。")

        # 2. 数据库持久化逻辑
        records_to_update = []
        records_to_create = []
        
        existing_records = {
            p.signal_name: p for p in AtomicSignalPerformance.objects.all()
        }

        for _, row in agg_df.iterrows():
            signal_name = row['signal_name']
            if signal_name in existing_records:
                record = existing_records[signal_name]
                record.signal_cn_name = row['cn_name']
                record.signal_type = row['type']
                record.total_triggers = row['total_triggers']
                record.successes = row['total_successes']
                record.win_rate_pct = row['win_rate_pct']
                record.avg_max_profit_pct = row['avg_max_profit_pct']
                record.avg_max_drawdown_pct = row['avg_max_drawdown_pct']
                record.avg_exit_days = row['avg_exit_days']
                records_to_update.append(record)
            else:
                records_to_create.append(AtomicSignalPerformance(
                    signal_name=row['signal_name'],
                    signal_cn_name=row['cn_name'],
                    signal_type=row['type'],
                    total_triggers=row['total_triggers'],
                    successes=row['total_successes'],
                    win_rate_pct=row['win_rate_pct'],
                    avg_max_profit_pct=row['avg_max_profit_pct'],
                    avg_max_drawdown_pct=row['avg_max_drawdown_pct'],
                    avg_exit_days=row['avg_exit_days'],
                ))
        
        
        # 在同步任务中，直接调用同步的ORM方法。
        #           之前的 sync_to_async(...) 调用只创建了协程但未执行。
        if records_to_create:
            print(f"调试信息: [原子信号 Reduce] 准备创建 {len(records_to_create)} 条记录...")
            AtomicSignalPerformance.objects.bulk_create(records_to_create)
            logger.info(f"[原子信号 Reduce] 成功创建 {len(records_to_create)} 条新的信号性能记录。")
        
        if records_to_update:
            print(f"调试信息: [原子信号 Reduce] 准备更新 {len(records_to_update)} 条记录...")
            AtomicSignalPerformance.objects.bulk_update(
                records_to_update, 
                ['signal_cn_name', 'signal_type', 'total_triggers', 'successes', 'win_rate_pct', 
                 'avg_max_profit_pct', 'avg_max_drawdown_pct', 'avg_exit_days', 'last_analyzed']
            )
            logger.info(f"[原子信号 Reduce] 成功更新 {len(records_to_update)} 条已有的信号性能记录。")
        

        logger.info("====== [原子信号分析 V1.2 - Reduce] 聚合任务完成 ======")
        return {"status": "success", "created": len(records_to_create), "updated": len(records_to_update)}

    except Exception as e:
        logger.error(f"[原子信号 Reduce] 聚合任务执行时发生严重错误: {e}", exc_info=True)
        return {"status": "error", "reason": str(e)}

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.analyze_performance_from_db', queue='calculate_strategy')
@with_cache_manager
def analyze_performance_from_db(self, stock_code: str, start_date: str, end_date: str, *, cache_manager: CacheManager):
    """
    【V1.2 - 安静的Map任务】
    作为MapReduce中的Map阶段，此任务只负责计算并返回原始数据，不打印任何报告。
    - 移除了所有格式化和打印报告的逻辑，以避免在并行执行时产生大量日志噪音。
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

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.run_global_performance_analysis', queue='celery')
@with_cache_manager
def run_global_performance_analysis(self, stock_list: list = None, start_date: str = None, end_date: str = None, *, cache_manager: CacheManager):
    """
    【V3.2 全并行架构版】
    """
    logger.info("="*80)
    logger.info(f"--- [全局性能分析 V3.2 - 全并行架构总指挥启动] ---")
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
        map_tasks_final = [
            analyze_performance_from_db.s(
                stock_code=code,
                start_date=start_date,
                end_date=end_date
            ).set(queue='calculate_strategy') for code in codes_to_run
        ]
        reduce_task_final = aggregate_performance_results.s().set(queue='celery')
        workflow_final = chord(header=group(map_tasks_final), body=reduce_task_final)
        workflow_final.apply_async()
        logger.info(f"-> 指令已下达！{total_stocks} 个【最终信号】分析子任务已派发。")

        # 使用新的 MapReduce 架构来派发原子信号分析任务
        # --- 作战指令二: 启动【原子信号】分析特遣队 (MapReduce) ---
        logger.info("\n--- [指令 2/2] 正在向【原子信号分析特遣队】派发 MapReduce 任务...")
        map_tasks_atomic = [
            analyze_atomic_signals_for_stock.s(
                stock_code=code
            ).set(queue='calculate_strategy') for code in codes_to_run
        ]
        reduce_task_atomic = aggregate_atomic_signal_results.s().set(queue='celery')
        workflow_atomic = chord(header=group(map_tasks_atomic), body=reduce_task_atomic)
        workflow_atomic.apply_async()
        logger.info(f"-> 指令已下达！{total_stocks} 个【原子信号】分析子任务已派发，将并行运行。")
        
        logger.info("\n" + "="*80)
        logger.info(f"--- [全局性能分析 V3.2 - 所有作战指令已下达] ---")
        return {"status": "all_workflows_dispatched", "total_stocks": total_stocks}

    except Exception as e:
        logger.error(f"在派发全局性能分析任务时发生严重错误: {e}", exc_info=True)
        return {"status": "error", "reason": str(e)}







