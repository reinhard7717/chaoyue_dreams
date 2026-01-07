# 文件: tasks/stock_analysis_tasks.py
# 版本: V2.1 - 装饰器重构版

import asyncio
from datetime import date, datetime, timedelta
from django.utils import timezone
import logging
from collections import defaultdict
from asgiref.sync import sync_to_async
from asgiref.sync import async_to_sync
from celery import group, chain, chord
from utils.task_helpers import with_cache_manager
from strategies.trend_following.utils import normalize_score
from utils.model_helpers import get_daily_data_model_by_code, get_cyq_chips_model_by_code, get_advanced_chip_metrics_model_by_code, get_advanced_fund_flow_metrics_model_by_code
from tqdm import tqdm
import numpy as np
import pandas as pd
import pandas_ta as ta
from django.db import transaction, connection
from chaoyue_dreams.celery import app as celery_app
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from stock_models.stock_analytics import StrategyDailyScore, TradingSignal, PositionTracker, DailyPositionSnapshot
from services.position_snapshot_service import PositionSnapshotService
from dao_manager.tushare_daos.industry_dao import IndustryDao
from dao_manager.tushare_daos.strategies_dao import StrategiesDAO
from stock_models.advanced_metrics import (
    AdvancedChipMetrics_SH, AdvancedChipMetrics_SZ, AdvancedChipMetrics_CY, AdvancedChipMetrics_KC, AdvancedChipMetrics_BJ,
)
from stock_models.index import TradeCalendar
from services.contextual_analysis_service import ContextualAnalysisService
from stock_models.stock_basic import StockInfo
from stock_models.time_trade import StockDailyBasic
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
# =================== 2. 高级资金、筹码特征任务 ==================
# =================================================================
async def _initialize_task_context_unified(stock_code: str, is_incremental: bool, start_date_str: str = None):
    """【V2.1 · 三级窗口终极版】引入回溯、处理、存储三级窗口，根除所有日期逻辑缺陷。"""
    from services.fund_flow_service import AdvancedFundFlowMetricsService
    from services.advanced_chip_metrics_service import AdvancedChipMetricsService
    from datetime import datetime, timedelta
    stock_info = await sync_to_async(StockInfo.objects.get)(stock_code=stock_code)
    ChipMetricsModel = get_advanced_chip_metrics_model_by_code(stock_code)
    FundFlowMetricsModel = get_advanced_fund_flow_metrics_model_by_code(stock_code)
    chip_service = AdvancedChipMetricsService()
    fund_flow_service = AdvancedFundFlowMetricsService()
    max_lookback_days = max(chip_service.max_lookback_days, fund_flow_service.max_lookback_days)
    # 核心修正：定义三级时间窗口变量
    lookback_start_date = None # 数据加载的最远边界
    process_start_date = None  # 计算循环的起点 (T-1)
    save_start_date = None     # 指标存储的起点 (T)
    if start_date_str:
        try:
            save_start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
            is_incremental = True
            process_start_date = await sync_to_async(TradeCalendar.get_trade_date_offset)(reference_date=save_start_date, offset=-1)
            if not process_start_date:
                process_start_date = save_start_date
            lookback_start_date = process_start_date - timedelta(days=max_lookback_days)
            chip_del_count, _ = await sync_to_async(ChipMetricsModel.objects.filter(stock=stock_info, trade_time__gte=save_start_date).delete)()
            ff_del_count, _ = await sync_to_async(FundFlowMetricsModel.objects.filter(stock=stock_info, trade_time__gte=save_start_date).delete)()
            logger.info(f"[{stock_code}] [统一回滚] 筹码指标删除 {chip_del_count} 条，资金流指标删除 {ff_del_count} 条。")
        except (ValueError, TypeError):
            logger.error(f"[{stock_code}] 提供的起始日期 '{start_date_str}' 格式错误，将忽略。")
            is_incremental = True
    if is_incremental and not start_date_str:
        @sync_to_async
        def get_latest_date(model, stock):
            try:
                return model.objects.filter(stock=stock).latest('trade_time').trade_time
            except model.DoesNotExist:
                return None
        latest_chip_date = await get_latest_date(ChipMetricsModel, stock_info)
        latest_ff_date = await get_latest_date(FundFlowMetricsModel, stock_info)
        last_metric_date = None
        if latest_chip_date and latest_ff_date:
            last_metric_date = min(latest_chip_date, latest_ff_date)
        elif latest_chip_date:
            last_metric_date = latest_chip_date
        elif latest_ff_date:
            last_metric_date = latest_ff_date
        if last_metric_date:
            process_start_date = last_metric_date
            save_start_date = await sync_to_async(TradeCalendar.get_next_trade_date)(last_metric_date)
            lookback_start_date = process_start_date - timedelta(days=max_lookback_days)
        else:
            is_incremental = False
    if not is_incremental:
        genesis_date_str = '2025-05-01'
        save_start_date = datetime.strptime(genesis_date_str, '%Y-%m-%d').date()
        process_start_date = await sync_to_async(TradeCalendar.get_trade_date_offset)(reference_date=save_start_date, offset=-1)
        if not process_start_date:
            process_start_date = save_start_date
        lookback_start_date = process_start_date - timedelta(days=max_lookback_days)
        # logger.info(f"[{stock_code}] [统一初始化] 创世模式启动。存储自: {save_start_date}, 处理自: {process_start_date}, 回溯自: {lookback_start_date}")
    # 核心修正：返回新的三级窗口日期
    return stock_info, ChipMetricsModel, FundFlowMetricsModel, is_incremental, lookback_start_date, process_start_date, save_start_date

async def _load_all_sources_unified(stock_info: StockInfo, daily_data_model, dates_in_chunk: pd.DatetimeIndex, cache_manager: CacheManager):
    """
    【V53.0 · 源头备粮】
    - 核心修正: 在数据加载总枢纽中，对 Level-5 盘口数据进行预处理，新增了计算订单流失衡(OFI)
                 的逻辑，并将其作为 `ofi` 列附加到 `stock_level5_data` DataFrame 中。
                 此举确保了下游指标（如 `midday_consolidation_level`）能够获取到必要的输入数据，
                 从根本上解决了因缺少预处理而导致的指标计算失效问题。
    """
    import pytz
    from utils.model_helpers import (
        get_fund_flow_model_by_code, get_fund_flow_ths_model_by_code, get_fund_flow_dc_model_by_code,
        get_cyq_chips_model_by_code
    )
    from stock_models.time_trade import StockDailyBasic, StockCyqPerf
    from django.utils import timezone
    from datetime import time, datetime, timedelta
    from dao_manager.tushare_daos.realtime_data_dao import StockRealtimeDAO
    from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO
    @sync_to_async(thread_sensitive=True)
    def get_data_async(model, stock_info_obj, fields: tuple = None, date_field='trade_time', dates_list: list = None):
        if not model or not dates_list: return pd.DataFrame()
        qs = model.objects.filter(stock=stock_info_obj, **{f'{date_field}__in': dates_list})
        return pd.DataFrame.from_records(qs.values(*fields) if fields else qs.values())
    realtime_dao = StockRealtimeDAO(cache_manager_instance=cache_manager)
    time_trade_dao = StockTimeTradeDAO(cache_manager_instance=cache_manager)
    chip_model = get_cyq_chips_model_by_code(stock_info.stock_code)
    all_daily_fields = (
        'trade_time', 'close', 'amount', 'vol', 'close_qfq', 'high_qfq', 'low_qfq', 'open_qfq', 'pre_close_qfq', 'pct_change'
    )
    # 修改：添加 'turnover_rate_f' 到 all_daily_basic_fields
    all_daily_basic_fields = (
        'trade_time', 'circ_mv', 'turnover_rate', 'float_share', 'turnover_rate_f'
    )
    fund_flow_tushare_fields = (
        'trade_time', 'buy_sm_vol', 'buy_sm_amount', 'sell_sm_vol', 'sell_sm_amount',
        'buy_md_vol', 'buy_md_amount', 'sell_md_vol', 'sell_md_amount',
        'buy_lg_vol', 'buy_lg_amount', 'sell_lg_vol', 'sell_lg_amount',
        'buy_elg_vol', 'buy_elg_amount', 'sell_elg_vol', 'sell_elg_amount',
        'net_mf_vol', 'net_mf_amount', 'trade_count'
    )
    chunk_dates_list = [d.date() for d in dates_in_chunk]
    chunk_start_date = dates_in_chunk.min().date()
    chunk_end_date = dates_in_chunk.max().date()
    model_name_map = {
        "cyq_chips": chip_model, "daily_data": daily_data_model, "daily_basic": StockDailyBasic,
        "cyq_perf": StockCyqPerf, "fund_flow_tushare": get_fund_flow_model_by_code(stock_info.stock_code),
        "fund_flow_ths": get_fund_flow_ths_model_by_code(stock_info.stock_code),
        "fund_flow_dc": get_fund_flow_dc_model_by_code(stock_info.stock_code),
    }
    data_tasks = {
        "cyq_chips": get_data_async(chip_model, stock_info, fields=('trade_time', 'price', 'percent'), dates_list=chunk_dates_list),
        "daily_data": get_data_async(daily_data_model, stock_info, fields=all_daily_fields, dates_list=chunk_dates_list),
        "daily_basic": get_data_async(StockDailyBasic, stock_info, fields=all_daily_basic_fields, dates_list=chunk_dates_list),
        "cyq_perf": get_data_async(StockCyqPerf, stock_info, dates_list=chunk_dates_list),
        "fund_flow_tushare": get_data_async(get_fund_flow_model_by_code(stock_info.stock_code), stock_info, fields=fund_flow_tushare_fields, dates_list=chunk_dates_list),
        "fund_flow_ths": get_data_async(get_fund_flow_ths_model_by_code(stock_info.stock_code), stock_info, dates_list=chunk_dates_list),
        "fund_flow_dc": get_data_async(get_fund_flow_dc_model_by_code(stock_info.stock_code), stock_info, dates_list=chunk_dates_list),
    }
    results = await asyncio.gather(*data_tasks.values())
    data_dfs = dict(zip(data_tasks.keys(), results))
    tick_data_df_list = []
    level5_data_df_list = []
    minute_data_df_list = []
    realtime_data_df_list = []
    for single_date in dates_in_chunk.normalize().unique().date:
        df_tick = await realtime_dao.get_daily_real_ticks(stock_info.stock_code, single_date.strftime('%Y-%m-%d'))
        if df_tick is not None and not df_tick.empty:
            tick_data_df_list.append(df_tick.reset_index())
        df_realtime, df_level5 = await realtime_dao._get_single_stock_quotes_and_level5_from_db(stock_info.stock_code, single_date)
        if df_realtime is not None and not df_realtime.empty:
            realtime_data_df_list.append(df_realtime.reset_index())
        if df_level5 is not None and not df_level5.empty:
            level5_data_df_list.append(df_level5.reset_index())
        df_minute = await time_trade_dao.get_intraday_kline_by_date(stock_info.stock_code, single_date, '1')
        if df_minute is not None and not df_minute.empty:
            minute_data_df_list.append(df_minute.reset_index())
    data_dfs["stock_tick_data"] = pd.concat(tick_data_df_list) if tick_data_df_list else pd.DataFrame()
    data_dfs["stock_level5_data"] = pd.concat(level5_data_df_list) if level5_data_df_list else pd.DataFrame()
    data_dfs["stock_minute_data"] = pd.concat(minute_data_df_list) if minute_data_df_list else pd.DataFrame()
    data_dfs["stock_realtime_data"] = pd.concat(realtime_data_df_list) if realtime_data_df_list else pd.DataFrame()
    if not data_dfs["stock_level5_data"].empty and len(data_dfs["stock_level5_data"]) > 1:
        level5_df = data_dfs["stock_level5_data"].sort_values('trade_time').reset_index(drop=True)
        df_prev = level5_df.shift(1)
        delta_buy_price = level5_df['buy_price1'] - df_prev['buy_price1']
        delta_sell_price = level5_df['sell_price1'] - df_prev['sell_price1']
        ofi_static = np.where((delta_buy_price == 0) & (delta_sell_price == 0), level5_df['buy_volume1'] - df_prev['buy_volume1'], 0)
        ofi_dynamic = np.where(delta_buy_price > 0, df_prev['buy_volume1'], 0)
        ofi_dynamic = np.where(delta_buy_price < 0, -level5_df['buy_volume1'], ofi_dynamic)
        ofi_dynamic = np.where(delta_sell_price > 0, ofi_dynamic + level5_df['sell_volume1'], ofi_dynamic)
        ofi_dynamic = np.where(delta_sell_price < 0, ofi_dynamic - df_prev['sell_volume1'], ofi_dynamic)
        level5_df['ofi'] = ofi_static + ofi_dynamic
        data_dfs["stock_level5_data"] = level5_df
    non_numeric_whitelist = ['stock_id', 'stock_code', 'trade_time', 'trade_date', 'type']
    for name, df in data_dfs.items():
        if isinstance(df, pd.DataFrame) and not df.empty and "_map" not in name:
            for col in df.columns:
                if col not in non_numeric_whitelist and not pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
    def _process_intraday_df_to_map(df: pd.DataFrame, stock_code_for_log: str, data_source_name: str) -> dict:
        if df.empty: return {}
        df['trade_time'] = pd.to_datetime(df['trade_time'])
        target_tz = pytz.timezone('Asia/Shanghai')
        if df['trade_time'].dt.tz is None:
            df['trade_time'] = df['trade_time'].dt.tz_localize(target_tz, ambiguous='infer')
        else:
            df['trade_time'] = df['trade_time'].dt.tz_convert(target_tz)
        df = df.set_index('trade_time')
        grouped_data = {}
        for date, group_df in df.groupby(df.index.date):
            grouped_data[date] = group_df
        return grouped_data
    data_dfs["stock_tick_data_map"] = _process_intraday_df_to_map(data_dfs["stock_tick_data"], stock_info.stock_code, "Tick Data")
    data_dfs["stock_level5_data_map"] = _process_intraday_df_to_map(data_dfs["stock_level5_data"], stock_info.stock_code, "Level5 Data")
    data_dfs["stock_minute_data_map"] = _process_intraday_df_to_map(data_dfs["stock_minute_data"], stock_info.stock_code, "Minute K-line Data")
    data_dfs["stock_realtime_data_map"] = _process_intraday_df_to_map(data_dfs["stock_realtime_data"], stock_info.stock_code, "Realtime Snapshot Data")
    for name, df in data_dfs.items():
        if name in ["stock_tick_data_map", "stock_level5_data_map", "stock_minute_data_map", "stock_realtime_data_map"]:
            continue
        if df is None or df.empty:
            if 'fund_flow_ths' in name or 'fund_flow_dc' in name:
                data_dfs[name] = pd.DataFrame()
                continue
            if name == "cyq_chips":
                logger.error(f"[{stock_info.stock_code}] [审计失败] 核心数据源 '{name}' 在日期列表查询中为空！查询日期列表: {chunk_dates_list}")
            else:
                logger.error(f"[{stock_info.stock_code}] [审计失败] 核心数据源 '{name}' 在日期列表查询中为空！")
            data_dfs[name] = pd.DataFrame()
    return data_dfs

# 核心定义“司令部”汇总任务
@celery_app.task(name='tasks.stock_analysis_tasks.summarize_computation_failures', queue='celery')
def summarize_computation_failures(results):
    """
    【汇总报告任务】收集所有子任务的计算失败记录并统一输出。
    """
    all_failures = []
    for result in results:
        # 健壮性检查，以防某个任务彻底失败未返回标准字典
        if isinstance(result, dict) and 'failures' in result:
            failures = result.get('failures', [])
            if failures:
                all_failures.extend(failures)
    if not all_failures:
        logger.info("【汇总报告】所有股票计算任务均未报告数据缺失问题。系统健康。")
        return {"status": "success", "total_failures": 0}
    logger.warning("【汇总报告】在计算过程中发现以下数据缺失问题：")
    # 按股票代码对失败记录进行分组，以便更清晰地展示
    failures_by_stock = {}
    for failure in all_failures:
        stock_code = failure.get('stock_code', 'UNKNOWN')
        if stock_code not in failures_by_stock:
            failures_by_stock[stock_code] = []
        failures_by_stock[stock_code].append(failure)
    for stock_code, failures in sorted(failures_by_stock.items()):
        logger.warning(f"--- 股票代码: {stock_code} ---")
        # 按日期排序
        sorted_failures = sorted(failures, key=lambda x: x['trade_date'])
        for f in sorted_failures:
            logger.warning(f"  - 日期: {f['trade_date']}, 原因: {f['reason']}")
    return {"status": "success", "total_failures": len(all_failures), "details": failures_by_stock}

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.precompute_advanced_structural_metrics_for_stock', queue='SaveHistoryData_TimeTrade')
@with_cache_manager
def precompute_advanced_structural_metrics_for_stock(self, stock_code: str, is_incremental: bool = True, start_date_str: str = None, *, cache_manager: CacheManager):
    """
    【V22.1 · 诊断驾驶舱】为单只股票预计算高级结构与行为指标的Celery任务。
    - 核心升级: 加载策略配置文件，并将 debug_params 传递给服务层，以实现探针的精确触发。
    """
    async def main(incremental_flag: bool, start_date_override: str):
        from services.advanced_structural_metrics_service import AdvancedStructuralMetricsService
        from stock_models.index import TradeCalendar
        from utils.model_helpers import get_daily_data_model_by_code
        from datetime import timedelta
        import pandas_ta as ta
        # 加载配置文件以获取调试参数
        import json
        import os
        from django.conf import settings
        config_path = os.path.join(settings.BASE_DIR, 'config', 'trend_follow_strategy.json')
        strategy_config = {}
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                strategy_config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass # 在结构指标任务中，即使配置失败也不中断，仅影响调试探针
        # 修改：从配置文件中正确读取 debug_params
        debug_params_from_config = strategy_config.get('strategy_params', {}).get('trend_follow', {}).get('debug_params', {})
        # 在服务初始化时传入调试参数
        structural_service = AdvancedStructuralMetricsService(debug_params=debug_params_from_config)
        stock_info, MetricsModel, _, last_metric_date, fetch_start_date = await structural_service._initialize_context(
            stock_code, incremental_flag, start_date_override
        )
        DailyModel = get_daily_data_model_by_code(stock_code)
        date_filter = {'stock': stock_info}
        if start_date_override:
            date_filter['trade_time__gte'] = start_date_override
        elif last_metric_date:
            date_filter['trade_time__gt'] = last_metric_date
        all_dates_qs = DailyModel.objects.filter(**date_filter).values_list('trade_time', flat=True).order_by('trade_time')
        dates_to_process = pd.to_datetime(await sync_to_async(list)(all_dates_qs))
        if dates_to_process.empty:
            logger.info(f"[{stock_code}] [结构指标任务] 无需计算的日期，任务终止。")
            return 0
        history_start_date = dates_to_process.min().date() - timedelta(days=100)
        # 修改：添加 'turnover_rate_f' 和 'vol' 到 daily_data_qs 的 values 列表
        # 注意：'turnover_rate_f' 实际上在 StockDailyBasic 中，而不是 DailyModel。
        # 需要从 StockDailyBasic 中单独加载或在 _load_all_sources_unified 中合并。
        # 为了简化，这里假设 DailyModel 包含了这些字段，或者在 _load_all_sources_unified 中会合并。
        # 实际情况是 _load_all_sources_unified 会加载 daily_basic，所以这里不需要重复加载 turnover_rate_f
        # 但 'vol' 字段在 DailyModel 中，需要确保加载。
        daily_data_qs = DailyModel.objects.filter(
            stock=stock_info,
            trade_time__gte=history_start_date,
            trade_time__lte=dates_to_process.max().date()
        ).values('trade_time', 'high_qfq', 'low_qfq', 'close_qfq', 'open_qfq', 'pre_close_qfq', 'amount', 'vol').order_by('trade_time')
        daily_df_with_atr = pd.DataFrame.from_records(await sync_to_async(list)(daily_data_qs))
        if daily_df_with_atr.empty:
            logger.error(f"[{stock_code}] [结构指标任务] 无法加载必要的日线数据，任务终止。")
            return 0
        daily_df_with_atr['trade_time'] = pd.to_datetime(daily_df_with_atr['trade_time'])
        daily_df_with_atr = daily_df_with_atr.set_index('trade_time')
        daily_df_with_atr.ta.atr(length=5, append=True, col_names=('ATR_5',))
        daily_df_with_atr.ta.atr(length=14, append=True, col_names=('ATR_14',))
        daily_df_with_atr.ta.atr(length=50, append=True, col_names=('ATR_50',))
        # _load_all_sources_unified 会加载 daily_basic，其中包含 turnover_rate_f
        # 所以这里不需要额外处理 turnover_rate_f，它会在 data_dfs['daily_basic'] 中
        data_dfs = await _load_all_sources_unified(stock_info, DailyModel, dates_to_process, cache_manager)
        # 合并 daily_basic 到 daily_df_with_atr，以获取 turnover_rate_f
        if 'daily_basic' in data_dfs and not data_dfs['daily_basic'].empty:
            daily_basic_df = data_dfs['daily_basic'].set_index('trade_time')
            # 确保索引类型一致
            daily_basic_df.index = pd.to_datetime(daily_basic_df.index)
            # 合并 turnover_rate_f
            daily_df_with_atr = daily_df_with_atr.join(daily_basic_df[['turnover_rate_f']], how='left')
        tick_data_map = data_dfs.get("stock_tick_data_map", {})
        level5_data_map = data_dfs.get("stock_level5_data_map", {})
        minute_data_map = data_dfs.get("stock_minute_data_map", {})
        realtime_data_map = data_dfs.get("stock_realtime_data_map", {})
        nested_intraday_data_map = {}
        all_dates_from_maps = set(tick_data_map.keys()) | set(level5_data_map.keys()) | set(minute_data_map.keys())
        for date_obj in all_dates_from_maps:
            if pd.to_datetime(date_obj) in dates_to_process:
                nested_intraday_data_map[date_obj] = {
                    'minute': minute_data_map.get(date_obj),
                    'tick': tick_data_map.get(date_obj),
                    'level5': level5_data_map.get(date_obj),
                    'realtime': realtime_data_map.get(date_obj),
                }
        processed_count = await structural_service.run_precomputation(
            stock_info=stock_info,
            dates_to_process=dates_to_process,
            daily_df_with_atr=daily_df_with_atr,
            intraday_data_map=nested_intraday_data_map
        )
        logger.info(f"[{stock_code}] [结构指标任务] 成功完成，处理了 {processed_count} 条记录。")
        return {"status": "success", "stock_code": stock_code, "processed_days": processed_count}
    try:
        result = async_to_sync(main)(is_incremental, start_date_str)
        return result
    except Exception as e:
        logger.error(f"--- CATCHING EXCEPTION in precompute_advanced_structural_metrics_for_stock for {stock_code}: {e}", exc_info=True)
        raise

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.precompute_advanced_chips_for_stock', queue='SaveHistoryData_TimeTrade')
@with_cache_manager
def precompute_advanced_chips_for_stock(self, stock_code: str, is_incremental: bool = True, start_date_str: str = None, *, cache_manager: CacheManager):
    """
    【V37.1 · 诊断驾驶舱升级版】
    - 核心升级: 引入 M-系列 (Main Task) 探针，监控任务入口、原始数据加载和对服务的调用。
    """
    async def main(incremental_flag: bool, start_date_override: str):
        from services.fund_flow_service import AdvancedFundFlowMetricsService
        from services.advanced_chip_metrics_service import AdvancedChipMetricsService
        import pandas_ta as ta
        from stock_models.index import TradeCalendar
        import json
        import os
        from django.conf import settings
        import copy
        config_path = os.path.join(settings.BASE_DIR, 'config', 'trend_follow_strategy.json')
        strategy_config = {}
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                strategy_config = json.load(f)
        except FileNotFoundError:
            logger.error(f"策略配置文件未找到: {config_path}")
        except json.JSONDecodeError:
            logger.error(f"解码策略配置文件 JSON 失败: {config_path}")
        debug_params = strategy_config.get('strategy_params', {}).get('trend_follow', {}).get('debug_params', {})
        # 修改：使用新的should_probe和probe_dates配置
        should_probe = debug_params.get('should_probe', False)
        probe_dates = debug_params.get('probe_dates', [])
        fund_flow_service = AdvancedFundFlowMetricsService(debug_params=debug_params)
        chip_service = AdvancedChipMetricsService()
        stock_info, ChipMetricsModel, FundFlowMetricsModel, is_incremental_final, lookback_start_date, process_start_date, save_start_date = await _initialize_task_context_unified(
            stock_code, incremental_flag, start_date_override
        )
        DailyModel = get_daily_data_model_by_code(stock_code)
        DateSourceModel = StockDailyBasic
        process_date_filter = {'stock': stock_info}
        if process_start_date:
            process_date_filter['trade_time__gte'] = process_start_date
        all_dates_qs = DateSourceModel.objects.filter(**process_date_filter).values_list('trade_time', flat=True).order_by('trade_time')
        raw_dates_to_process = pd.to_datetime(await sync_to_async(list)(all_dates_qs))
        if raw_dates_to_process.empty:
            logger.info(f"[{stock_code}] 无需计算的日期，合并任务终止。")
            return {"status": "skipped", "reason": "No new dates to process.", "failures": []}
        trade_dates_only = await sync_to_async(TradeCalendar.get_trade_dates_between)(
            start_date=raw_dates_to_process.min().date(),
            end_date=raw_dates_to_process.max().date()
        )
        dates_to_process = pd.to_datetime([d for d in trade_dates_only if pd.to_datetime(d) in raw_dates_to_process])
        if dates_to_process.empty:
            logger.info(f"[{stock_code}] 过滤非交易日后，没有需要计算的交易日，合并任务终止。")
            return {"status": "skipped", "reason": "No trade dates to process after filtering.", "failures": []}
        context_end_date = dates_to_process.min()
        ff_hist_df = await fund_flow_service._load_historical_metrics(FundFlowMetricsModel, stock_info, context_end_date)
        chip_hist_df = await chip_service._load_historical_metrics(ChipMetricsModel, stock_info, context_end_date)
        context_df = ff_hist_df.join(chip_hist_df, how='outer') if not ff_hist_df.empty or not chip_hist_df.empty else pd.DataFrame()
        max_lookback_days = max(chip_service.max_lookback_days, fund_flow_service.max_lookback_days, 260)
        global_lookback_start_date = dates_to_process.min() - timedelta(days=max_lookback_days)
        all_daily_data_for_lookback_qs = DailyModel.objects.filter(
            stock=stock_info,
            trade_time__gte=global_lookback_start_date,
            trade_time__lte=dates_to_process.max()
        ).values('trade_time', 'high_qfq', 'low_qfq', 'close_qfq', 'open_qfq', 'pre_close_qfq', 'vol').order_by('trade_time')
        all_daily_data_for_lookback_df = pd.DataFrame(await sync_to_async(list)(all_daily_data_for_lookback_qs))
        all_daily_data_for_lookback_df['trade_time'] = pd.to_datetime(all_daily_data_for_lookback_df['trade_time'])
        all_daily_data_for_lookback_df.set_index('trade_time', inplace=True)
        all_daily_data_for_lookback_df.ta.atr(length=14, append=True)
        atr_col_name = 'ATRr_14'
        if atr_col_name not in all_daily_data_for_lookback_df.columns:
            raise ValueError(f"[{stock_code}] ATR计算失败，未能找到列: {atr_col_name}")
        all_daily_data_for_lookback_df['high_20d'] = all_daily_data_for_lookback_df['high_qfq'].rolling(window=20, min_periods=1).max()
        all_daily_data_for_lookback_df['low_20d'] = all_daily_data_for_lookback_df['low_qfq'].rolling(window=20, min_periods=1).min()
        all_daily_data_for_lookback_df['high_5d'] = all_daily_data_for_lookback_df['high_qfq'].rolling(window=5, min_periods=1).max()
        all_daily_data_for_lookback_df['low_5d'] = all_daily_data_for_lookback_df['low_qfq'].rolling(window=5, min_periods=1).min()
        all_daily_data_for_lookback_df['turnover_vol_5d'] = all_daily_data_for_lookback_df['vol'].rolling(window=5, min_periods=1).sum() * 100
        high_5d_map_global = all_daily_data_for_lookback_df['high_5d'].to_dict()
        low_5d_map_global = all_daily_data_for_lookback_df['low_5d'].to_dict()
        turnover_vol_5d_map_global = all_daily_data_for_lookback_df['turnover_vol_5d'].to_dict()
        high_20d_map_global = all_daily_data_for_lookback_df['high_20d'].to_dict()
        low_20d_map_global = all_daily_data_for_lookback_df['low_20d'].to_dict()
        atr_map_global = all_daily_data_for_lookback_df[atr_col_name].to_dict()
        close_map_global = all_daily_data_for_lookback_df['close_qfq'].to_dict()
        open_map_global = all_daily_data_for_lookback_df['open_qfq'].to_dict()
        pre_close_map_global = all_daily_data_for_lookback_df['pre_close_qfq'].to_dict()
        trade_dates_series_global = all_daily_data_for_lookback_df.index.sort_values().to_series().reset_index(drop=True)
        date_index_map = {date: i for i, date in enumerate(trade_dates_series_global)}
        date_20d_ago_map_global = {
            date: trade_dates_series_global.iloc[idx - 20] if idx >= 20 else pd.NaT
            for date, idx in date_index_map.items()
        }
        health_score_hist_lookback_date = dates_to_process.min().date() - timedelta(days=365)
        health_score_components = [
            'concentration_70pct', 'cost_divergence_normalized', 'dominant_peak_profit_margin',
            'main_force_cost_advantage', 'suppressive_accumulation_intensity', 'upward_impulse_purity',
            'active_winner_profit_margin', 'winner_conviction_index'
        ]
        chip_hist_fields = [f for f in health_score_components if hasattr(ChipMetricsModel, f)]
        ff_hist_fields = [f for f in health_score_components if hasattr(FundFlowMetricsModel, f)]
        @sync_to_async
        def load_historical_components(model, fields, stock_obj, start_dt):
            if not fields: return pd.DataFrame()
            qs = model.objects.filter(stock=stock_obj, trade_time__gte=start_dt).values('trade_time', *fields)
            df = pd.DataFrame.from_records(qs)
            if not df.empty:
                df['trade_time'] = pd.to_datetime(df['trade_time'])
                df = df.set_index('trade_time')
            return df
        chip_hist_comp_df = await load_historical_components(ChipMetricsModel, chip_hist_fields, stock_info, health_score_hist_lookback_date)
        ff_hist_comp_df = await load_historical_components(FundFlowMetricsModel, ff_hist_fields, stock_info, health_score_hist_lookback_date)
        historical_components_df = chip_hist_comp_df.join(ff_hist_comp_df, how='outer')
        if not historical_components_df.empty and all(c in historical_components_df.columns for c in ['cost_15pct', 'cost_85pct', 'weight_avg_cost']):
            valid_mask = historical_components_df['weight_avg_cost'] > 0
            historical_components_df.loc[valid_mask, 'concentration_70pct'] = \
                (historical_components_df.loc[valid_mask, 'cost_85pct'] - historical_components_df.loc[valid_mask, 'cost_15pct']) / historical_components_df.loc[valid_mask, 'weight_avg_cost']
        CHUNK_SIZE = 50
        all_final_metrics_to_save = pd.DataFrame()
        cross_chunk_memory_bus = {
            'chip_memory': {},
            'fund_flow_memory': {}
        }
        all_failures = []
        first_processing_day = dates_to_process.min().date()
        seed_date = await sync_to_async(TradeCalendar.get_trade_date_offset)(reference_date=first_processing_day, offset=-1)
        if seed_date:
            seed_chunk_dates = pd.DatetimeIndex([pd.to_datetime(seed_date)])
            seed_data_dfs = await _load_all_sources_unified(stock_info, DailyModel, seed_chunk_dates, cache_manager)
            if not seed_data_dfs["daily_data"].empty and not seed_data_dfs["daily_basic"].empty:
                seed_daily_df = seed_data_dfs["daily_data"].set_index(pd.to_datetime(seed_data_dfs["daily_data"]['trade_time'])).drop(columns='trade_time')
                seed_daily_basic_df = seed_data_dfs["daily_basic"].set_index(pd.to_datetime(seed_data_dfs["daily_basic"]['trade_time'])).drop(columns='trade_time')
                overlap_cols = seed_daily_df.columns.intersection(seed_daily_basic_df.columns)
                seed_base_daily_df = seed_daily_df.join(seed_daily_basic_df.drop(columns=overlap_cols), how='left')
                seed_base_daily_df['atr_14d'] = seed_base_daily_df.index.map(atr_map_global)
                ff_data_dfs = {"tushare": seed_data_dfs["fund_flow_tushare"], "ths": seed_data_dfs["fund_flow_ths"], "dc": seed_data_dfs["fund_flow_dc"]}
                fund_flow_service.debug_params = debug_params
                seed_ff_raw_df = await fund_flow_service._load_and_merge_sources(stock_info, data_dfs=ff_data_dfs, base_daily_df=seed_base_daily_df)
                fund_flow_service._minute_df_daily_grouped = await fund_flow_service._get_daily_grouped_minute_data(
                    stock_info, seed_ff_raw_df.index,
                    tick_data_map=seed_data_dfs.get("stock_tick_data_map"),
                    level5_data_map=seed_data_dfs.get("stock_level5_data_map"),
                    minute_data_map=seed_data_dfs.get("stock_minute_data_map")
                )
                _, seed_ff_minute_map, _, cross_chunk_memory_bus['fund_flow_memory'] = fund_flow_service._synthesize_and_forge_metrics(
                    stock_code, seed_ff_raw_df,
                    tick_data_map=seed_data_dfs.get("stock_tick_data_map"),
                    level5_data_map=seed_data_dfs.get("stock_level5_data_map"),
                    minute_data_map=seed_data_dfs.get("stock_minute_data_map"),
                    realtime_data_map=seed_data_dfs.get("stock_realtime_data_map"),
                    memory=cross_chunk_memory_bus['fund_flow_memory']
                )
                seed_ff_minute_map_for_chip_service = copy.deepcopy(seed_ff_minute_map)
                seed_chip_data_dfs = {"cyq_chips": seed_data_dfs["cyq_chips"], "cyq_perf": seed_data_dfs["cyq_perf"]}
                seed_chip_raw_df = chip_service._preprocess_and_merge_data(
                    stock_code, seed_chip_data_dfs, seed_base_daily_df, close_map_global, date_20d_ago_map_global, atr_map_global,
                    high_20d_map_global, low_20d_map_global, high_5d_map_global, low_5d_map_global, turnover_vol_5d_map_global
                )
                seed_minute_map = await chip_service._load_minute_data_for_range(stock_info, seed_chunk_dates.min(), seed_chunk_dates.max(), tick_data_map=seed_data_dfs["stock_tick_data_map"], minute_data_map=seed_data_dfs["stock_minute_data_map"])
                _, cross_chunk_memory_bus['chip_memory'], seed_failures = chip_service._synthesize_and_forge_metrics(
                    stock_info, seed_chip_raw_df, seed_minute_map, seed_ff_minute_map_for_chip_service,
                    memory=cross_chunk_memory_bus['chip_memory'], historical_components=historical_components_df, debug_params=debug_params,
                    tick_data_map=seed_data_dfs.get("stock_tick_data_map"),
                    realtime_data_map=seed_data_dfs.get("stock_realtime_data_map"),
                    level5_data_map=seed_data_dfs.get("stock_level5_data_map")
                )
            else:
                logger.warning(f"[{stock_code}] [上下文播种] 播种日 {seed_date} 核心数据缺失，无法生成初始记忆。")
        for i in range(0, len(dates_to_process), CHUNK_SIZE):
            chunk_dates = dates_to_process[i:i + CHUNK_SIZE]
            if chunk_dates.empty: continue
            print(f"[探针 T.1 - {stock_code}] 开始处理第 {i // CHUNK_SIZE + 1} 数据块, 日期范围: {chunk_dates.min().date()} to {chunk_dates.max().date()}, 共 {len(chunk_dates)} 天。")
            data_dfs = await _load_all_sources_unified(stock_info, DailyModel, chunk_dates, cache_manager)
            # 修改：更新【M.2 - 原始数据审计探针】逻辑
            if should_probe and probe_dates: # 检查是否启用探针且有指定日期
                # 遍历当前数据块中的每个日期
                for current_chunk_date in chunk_dates:
                    current_chunk_date_str = current_chunk_date.strftime('%Y-%m-%d')
                    if current_chunk_date_str in probe_dates: # 检查当前日期是否在探针日期列表中
                        print(f"[探针 M.2 - {stock_code} - {current_chunk_date_str}] 原始数据审计探针启动。")
                        for name, data_map in [
                            ("Tick", data_dfs.get("stock_tick_data_map")),
                            ("Level5", data_dfs.get("stock_level5_data_map")),
                            ("Minute", data_dfs.get("stock_minute_data_map")),
                            ("Realtime", data_dfs.get("stock_realtime_data_map"))
                        ]:
                            # 检查数据映射是否存在且包含当前日期的实际数据
                            if data_map and current_chunk_date.date() in data_map:
                                print(f"  - [检查通过] {name} 数据已加载, Shape: {data_map[current_chunk_date.date()].shape}")
                            else:
                                print(f"  - [!!!] 关键警告: {name} 数据在目标日期 {current_chunk_date_str} 缺失！")
            tick_data_map = data_dfs.pop("stock_tick_data_map")
            level5_data_map = data_dfs.pop("stock_level5_data_map")
            minute_data_map = data_dfs.pop("stock_minute_data_map")
            realtime_data_map = data_dfs.pop("stock_realtime_data_map")
            critical_sources = ["cyq_chips", "daily_data", "daily_basic", "cyq_perf", "fund_flow_tushare"]
            if any(data_dfs.get(src) is None or data_dfs.get(src).empty for src in critical_sources):
                for src in critical_sources:
                    if data_dfs.get(src) is None or data_dfs.get(src).empty:
                        print(f"[探针 T.2 - {stock_code}] 审计熔断！原因: 关键数据源 '{src}' 在日期范围 {chunk_dates.min().date()} to {chunk_dates.max().date()} 内完全为空。")
                logger.warning(f"[{stock_code}] [审计熔断] 区块 {chunk_dates.min().date()} to {chunk_dates.max().date()} 因一个或多个关键数据源在整个批次内完全为空而被跳过。")
                for src in critical_sources:
                    if data_dfs.get(src) is None or data_dfs.get(src).empty:
                        all_failures.append({
                            'stock_code': stock_code,
                            'trade_date': f"{chunk_dates.min().date()} to {chunk_dates.max().date()}",
                            'reason': f"上游核心数据源 '{src}' 在整个批次内完全为空",
                            'missing_source': src
                        })
                continue
            daily_df = data_dfs["daily_data"].set_index(pd.to_datetime(data_dfs["daily_data"]['trade_time'])).drop(columns='trade_time')
            daily_basic_df = data_dfs["daily_basic"].set_index(pd.to_datetime(data_dfs["daily_basic"]['trade_time'])).drop(columns='trade_time')
            overlap_cols = daily_df.columns.intersection(daily_basic_df.columns)
            base_daily_df = daily_df.join(daily_basic_df.drop(columns=overlap_cols), how='left')
            base_daily_df['atr_14d'] = base_daily_df.index.map(atr_map_global)
            ff_data_dfs = {"tushare": data_dfs["fund_flow_tushare"], "ths": data_dfs["fund_flow_ths"], "dc": data_dfs["fund_flow_dc"]}
            fund_flow_service.debug_params = debug_params
            fund_flow_raw_df = await fund_flow_service._load_and_merge_sources(stock_info, data_dfs=ff_data_dfs, base_daily_df=base_daily_df)
            if fund_flow_raw_df.empty:
                logger.warning(f"[{stock_code}] 区块 {chunk_dates.min().date()} to {chunk_dates.max().date()} 资金流原始数据为空，跳过。")
                continue
            fund_flow_service._minute_df_daily_grouped = await fund_flow_service._get_daily_grouped_minute_data(
                stock_info, fund_flow_raw_df.index, tick_data_map=tick_data_map, level5_data_map=level5_data_map, minute_data_map=minute_data_map
            )
            fund_flow_metrics_df, fund_flow_attributed_minute_map, ff_failures, cross_chunk_memory_bus['fund_flow_memory'] = fund_flow_service._synthesize_and_forge_metrics(
                stock_code, fund_flow_raw_df, tick_data_map=tick_data_map, level5_data_map=level5_data_map, minute_data_map=minute_data_map, realtime_data_map=realtime_data_map, memory=cross_chunk_memory_bus['fund_flow_memory']
            )
            all_failures.extend(ff_failures)
            fund_flow_attributed_minute_map_for_chip_service = copy.deepcopy(fund_flow_attributed_minute_map)
            chip_data_dfs = {"cyq_chips": data_dfs["cyq_chips"], "cyq_perf": data_dfs["cyq_perf"]}
            chip_raw_df = chip_service._preprocess_and_merge_data(
                stock_code, chip_data_dfs, base_daily_df, close_map_global, date_20d_ago_map_global, atr_map_global,
                high_20d_map_global, low_20d_map_global, high_5d_map_global, low_5d_map_global, turnover_vol_5d_map_global
            )
            if 'trade_time' in chip_raw_df.columns:
                chip_raw_df.set_index('trade_time', inplace=True)
            if not fund_flow_metrics_df.empty:
                chip_raw_df = chip_raw_df.join(fund_flow_metrics_df, how='left')
            minute_data_map_for_chip = await chip_service._load_minute_data_for_range(
                stock_info, chunk_dates.min(), chunk_dates.max(), tick_data_map=tick_data_map, minute_data_map=minute_data_map
            )
            chip_metrics_df, cross_chunk_memory_bus['chip_memory'], chunk_failures = chip_service._synthesize_and_forge_metrics(
                stock_info, chip_raw_df, minute_data_map_for_chip, fund_flow_attributed_minute_map_for_chip_service,
                memory=cross_chunk_memory_bus['chip_memory'], historical_components=historical_components_df, debug_params=debug_params,
                tick_data_map=tick_data_map, realtime_data_map=realtime_data_map, level5_data_map=level5_data_map
            )
            all_failures.extend(chunk_failures)
            chunk_core_metrics_df = fund_flow_metrics_df.join(chip_metrics_df, how='outer')
            if not chunk_core_metrics_df.empty:
                new_hist_data = chunk_core_metrics_df[chunk_core_metrics_df.columns.intersection(health_score_components)]
                if not new_hist_data.empty:
                    historical_components_df = pd.concat([historical_components_df, new_hist_data])
                    historical_components_df = historical_components_df[~historical_components_df.index.duplicated(keep='last')].sort_index()
                    if all(c in historical_components_df.columns for c in ['cost_15pct', 'cost_85pct', 'weight_avg_cost']):
                        valid_mask = historical_components_df['weight_avg_cost'] > 0
                        historical_components_df.loc[valid_mask, 'concentration_70pct'] = \
                            (historical_components_df.loc[valid_mask, 'cost_85pct'] - historical_components_df.loc[valid_mask, 'cost_15pct']) / historical_components_df.loc[valid_mask, 'weight_avg_cost']
            full_sequence_for_derivatives = pd.concat([context_df, chunk_core_metrics_df]).sort_index()
            ff_derivatives = fund_flow_service._calculate_derivatives(stock_code, full_sequence_for_derivatives)
            chip_derivatives = chip_service._calculate_derivatives(full_sequence_for_derivatives)
            chunk_final_df = full_sequence_for_derivatives.join([ff_derivatives, chip_derivatives])
            all_final_metrics_to_save = pd.concat([all_final_metrics_to_save, chunk_final_df[chunk_final_df.index.date >= save_start_date]])
            context_df = full_sequence_for_derivatives
        if not all_final_metrics_to_save.empty:
            chunk_to_save = all_final_metrics_to_save[all_final_metrics_to_save.index.date >= save_start_date] if save_start_date else all_final_metrics_to_save
            if not chunk_to_save.empty:
                ff_model_fields = {f.name for f in FundFlowMetricsModel._meta.get_fields() if not f.is_relation}
                chip_model_fields = {f.name for f in ChipMetricsModel._meta.get_fields() if not f.is_relation}
                ff_cols_to_keep = [col for col in chunk_to_save.columns if col in ff_model_fields]
                chip_cols_to_keep = [col for col in chunk_to_save.columns if col in chip_model_fields]
                ff_df_to_save = chunk_to_save[ff_cols_to_keep]
                chip_df_to_save = chunk_to_save[chip_cols_to_keep]
                ff_save_count = await fund_flow_service._prepare_and_save_data(stock_info, FundFlowMetricsModel, ff_df_to_save)
                chip_save_count = await chip_service._prepare_and_save_data(stock_info, ChipMetricsModel, chip_df_to_save)
                logger.info(f"[{stock_code}] 成功！深度融合计算完成。资金流指标保存 {ff_save_count} 条，筹码指标保存 {chip_save_count} 条。")
                return {"status": "success", "fund_flow_days": ff_save_count, "chip_days": chip_save_count, "failures": all_failures}
            else:
                logger.info(f"[{stock_code}] 计算完成，但没有需要保存的新数据。")
                return {"status": "no_new_data_to_save", "failures": all_failures}
        else:
            logger.info(f"[{stock_code}] 深度融合计算未产生任何新指标。")
            return {"status": "no_new_data", "failures": all_failures}
    try:
        result = async_to_sync(main)(is_incremental, start_date_str)
        return result
    except Exception as e:
        logger.error(f"--- CATCHING EXCEPTION in precompute_advanced_chips_for_stock (merged task) for {stock_code}: {e}", exc_info=True)
        raise

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.precompute_geometric_patterns_for_stock', queue='SaveHistoryData_TimeTrade')
@with_cache_manager
def precompute_geometric_patterns_for_stock(self, stock_code: str, start_date_str: str = None, *, cache_manager: CacheManager):
    """
    【V2.17 · 回滚重算版】为单只股票预计算几何形态，并分析其动态演化事件。
    - V2.17 升级: 将 start_date_str 传递给服务层，以触发可能的数据回滚。
    """
    async def main():
        from services.geometric_pattern_service import GeometricPatternService
        from utils.model_helpers import get_daily_data_model_by_code
        from stock_models.models import StockInfo
        try:
            service = await GeometricPatternService.create(stock_code=stock_code)
            daily_model = service.daily_model
            stock_info = service.stock_instance
            all_dates_qs = daily_model.objects.filter(stock=stock_info)
            # 日期过滤逻辑保持不变，它决定了加载哪些原始数据
            if start_date_str:
                all_dates_qs = all_dates_qs.filter(trade_time__gte=start_date_str)
            all_dates_qs = all_dates_qs.values_list('trade_time', flat=True).order_by('trade_time')
            dates_to_process = pd.to_datetime(await sync_to_async(list)(all_dates_qs))
            if dates_to_process.empty or len(dates_to_process) < 60:
                logger.info(f"[{stock_code}] [几何形态任务] 数据不足 (<60天)，跳过计算。")
                return {"status": "skipped", "reason": "Insufficient data."}
            data_dfs = await _load_all_sources_unified(stock_info, daily_model, dates_to_process, cache_manager)
            # 将 start_date_str 传递给服务层
            await sync_to_async(service.calculate_and_save_all_patterns)(data_dfs, start_date_str=start_date_str)
            return {"status": "success", "stock_code": stock_code}
        except StockInfo.DoesNotExist:
            logger.error(f"[{stock_code}] [几何形态任务] 股票信息不存在，任务终止。")
            return {"status": "error", "reason": "StockInfo not found."}
        except Exception as e:
            logger.error(f"[{stock_code}] [几何形态任务] 发生未知错误: {e}", exc_info=True)
            raise
    try:
        result = async_to_sync(main)()
        return result
    except Exception as e:
        logger.error(f"--- CATCHING EXCEPTION in precompute_geometric_patterns_for_stock for {stock_code}: {e}", exc_info=True)
        raise

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.precompute_all_stocks_advanced_metrics', queue='celery')
def precompute_all_stocks_advanced_metrics(self, start_date_str: str = None, is_incremental: bool = True):
    """
    【总调度器 V3.7 · 日期逻辑探针版】
    - 核心增强: 在 `get_group_start_date` 辅助函数中植入探针，用于打印计算起始日期的详细决策过程，
                 包括各模型表的最新日期、最终同步日期以及与 `StockDailyBasic` 最新日期的比较结果，
                 旨在排查因日期逻辑错误导致任务无法正确触发的问题。
    """
    try:
        from celery import chord, group, chain
        from datetime import timedelta, datetime
        from stock_models.stock_basic import StockInfo
        from stock_models.time_trade import StockDailyBasic
        from stock_models.advanced_metrics import (
            AdvancedChipMetrics_SH, AdvancedChipMetrics_SZ, AdvancedChipMetrics_CY, AdvancedChipMetrics_KC, AdvancedChipMetrics_BJ,
            AdvancedFundFlowMetrics_SH, AdvancedFundFlowMetrics_SZ, AdvancedFundFlowMetrics_CY, AdvancedFundFlowMetrics_KC, AdvancedFundFlowMetrics_BJ,
            AdvancedStructuralMetrics_SH, AdvancedStructuralMetrics_SZ, AdvancedStructuralMetrics_CY, AdvancedStructuralMetrics_KC, AdvancedStructuralMetrics_BJ
        )
        chip_ff_models = [
            AdvancedChipMetrics_SH, AdvancedChipMetrics_SZ, AdvancedChipMetrics_CY, AdvancedChipMetrics_KC, AdvancedChipMetrics_BJ,
            AdvancedFundFlowMetrics_SH, AdvancedFundFlowMetrics_SZ, AdvancedFundFlowMetrics_CY, AdvancedFundFlowMetrics_KC, AdvancedFundFlowMetrics_BJ,
        ]
        structural_models = [
            AdvancedStructuralMetrics_SH, AdvancedStructuralMetrics_SZ, AdvancedStructuralMetrics_CY, AdvancedStructuralMetrics_KC, AdvancedStructuralMetrics_BJ
        ]
        # 植入日期逻辑探针
        def get_group_start_date(models, group_name: str):
            print(f"\n[探针 G.1 - {group_name}组] 开始确定计算起始日期...")
            all_latest_dates = []
            for model in models:
                latest_metric = model.objects.order_by('-trade_time').first()
                if latest_metric:
                    all_latest_dates.append(latest_metric.trade_time)
                    print(f"  - 模型 {model.__name__} 最新日期: {latest_metric.trade_time}")
                else:
                    print(f"  - 模型 {model.__name__} 无数据。")
            if not all_latest_dates:
                genesis_date = '2025-05-01'
                print(f"  -> [决策] {group_name}组未发现任何历史数据，将从创世日期 {genesis_date} 开始全量计算。")
                return genesis_date, False
            sync_date = min(all_latest_dates)
            print(f"  - 所有模型中最旧的同步日期 (sync_date): {sync_date}")
            latest_basic_data = StockDailyBasic.objects.order_by('-trade_time').first()
            if not latest_basic_data:
                raise ValueError("StockDailyBasic 为空，无法确定数据范围。")
            print(f"  - StockDailyBasic 最新日期: {latest_basic_data.trade_time}")
            start_date_obj = sync_date + timedelta(days=1)
            if start_date_obj > latest_basic_data.trade_time:
                print(f"  -> [决策] 所有指标已同步至 {sync_date}，该日期已是或晚于最新基础数据日期，无需更新。")
                return None, True
            final_start_date_str = start_date_obj.strftime('%Y-%m-%d')
            print(f"  -> [决策] 确定增量计算起始日期为: {final_start_date_str}")
            return final_start_date_str, True
        start_date_chip_ff, is_incremental_chip_ff = None, is_incremental
        start_date_structural, is_incremental_structural = None, is_incremental
        if start_date_str is None and is_incremental:
            start_date_chip_ff, is_incremental_chip_ff = get_group_start_date(chip_ff_models, "筹码-资金流")
            start_date_structural, is_incremental_structural = get_group_start_date(structural_models, "结构")
        else:
            start_date_chip_ff = start_date_str
            start_date_structural = start_date_str
            is_incremental_chip_ff = is_incremental
            is_incremental_structural = is_incremental
            print(f"[探针 G.0] 使用外部传入的起始日期: {start_date_str}, 增量模式: {is_incremental}")
        stock_codes = list(StockInfo.objects.filter(list_status='L').exclude(stock_code__endswith='.BJ').values_list('stock_code', flat=True))
        if not stock_codes:
            logger.warning("【总调度】在StockInfo中未找到任何符合条件的上市状态股票，任务终止。")
            return {"status": "skipped", "reason": "No listed stocks found."}
        all_stock_workflows = []
        run_precursor_tasks = (start_date_chip_ff is not None) and (start_date_structural is not None)
        if not run_precursor_tasks:
            logger.info("【总调度】前置高级指标任务无需更新，跳过本次调度。")
            return {"status": "skipped", "reason": "No precursor tasks to run."}
        for code in stock_codes:
            chip_ff_task = precompute_advanced_chips_for_stock.s(
                stock_code=code, is_incremental=is_incremental_chip_ff, start_date_str=start_date_chip_ff
            )
            structural_task = precompute_advanced_structural_metrics_for_stock.s(
                stock_code=code, is_incremental=is_incremental_structural, start_date_str=start_date_structural
            )
            trigger_task = _trigger_geometric_patterns_computation.s(stock_code=code, start_date_str=start_date_chip_ff)
            stock_chain = chain(
                group(chip_ff_task, structural_task),
                trigger_task
            )
            all_stock_workflows.append(stock_chain)
        if not all_stock_workflows:
            logger.warning("【总调度】未能为任何股票创建工作流，任务终止。")
            return {"status": "skipped", "reason": "No workflows created."}
        computation_workflow = group(all_stock_workflows)
        computation_workflow.apply_async()
        total_logical_tasks = len(stock_codes) * 3
        logger.info(f"【总调度】成功！已为 {len(stock_codes)} 支股票分发了包含适配器的责任链工作流，共计 {total_logical_tasks} 个核心计算任务。")
        return {
            "status": "success",
            "dispatched_stocks": len(stock_codes),
            "total_tasks_dispatched": total_logical_tasks
        }
    except Exception as e:
        logger.error(f"【总调度】任务分发过程中发生严重错误: {e}", exc_info=True)
        raise self.retry(exc=e, countdown=300, max_retries=3)

@celery_app.task(name='tasks.stock_analysis_tasks._trigger_geometric_patterns_computation', queue='SaveHistoryData_TimeTrade')
def _trigger_geometric_patterns_computation(previous_results, stock_code: str, start_date_str: str = None):
    """
    【V1.1 · 日期传递升级版】这是一个适配器任务，用于在Celery责任链中承接上游任务组的结果，
    然后以干净的参数启动真正的几何形态计算任务，从而避免参数传递冲突。
    - V1.1 升级: 新增 start_date_str 参数，确保计算的时间范围能够被正确传递。
    """
    logger.info(f"[{stock_code}] [责任链适配器] 接收到上游任务完成信号，准备触发几何形态计算 (起始日期: {start_date_str})...")
    # 丢弃 previous_results，使用明确传入的参数调用目标任务
    precompute_geometric_patterns_for_stock.delay(stock_code=stock_code, start_date_str=start_date_str)
    return {"status": "triggered", "stock_code": stock_code, "start_date": start_date_str}


@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.test_precompute_and_verify_structural_metrics', queue='SaveHistoryData_TimeTrade')
def test_precompute_and_verify_structural_metrics(self):
    # [代码新增开始]
    """
    【V1.0】用于测试结构指标计算并验证其完整性的独立任务。
    - 核心职责:
      1. 针对特定股票（'000001.SZ'）和特定起始日期（'2025-05-01'）触发一次性的高级结构指标计算。
      2. 计算完成后，查询数据库中该股票的最新一条结构指标记录。
      3. 检查并打印出最新记录中所有值为空（None/NaN）的字段，以供调试和验证。
    """
    from asgiref.sync import async_to_sync
    from utils.model_helpers import get_advanced_structural_metrics_model_by_code
    from stock_models.stock_basic import StockInfo
    import pandas as pd
    def run_test():
        stock_code = '000001.SZ'
        start_date_str = '2025-05-01'
        print(f"--- [结构指标测试任务] 开始执行计算，股票: {stock_code}, 日期: {start_date_str} ---")
        # 直接调用目标任务函数以同步执行，并等待其完成
        precompute_advanced_structural_metrics_for_stock(stock_code=stock_code, is_incremental=True, start_date_str=start_date_str)
        print("--- [结构指标测试任务] 计算执行完毕，开始验证数据... ---")
        # --- 验证阶段 ---
        StructuralMetricsModel = get_advanced_structural_metrics_model_by_code(stock_code)
        # 使用同步方式查询数据库
        try:
            stock_info = StockInfo.objects.get(stock_code=stock_code)
            latest_metric = StructuralMetricsModel.objects.filter(stock=stock_info).order_by('-trade_time').first()
        except StockInfo.DoesNotExist:
            print(f"--- [结构指标测试任务] 验证失败：未找到股票 {stock_code}。 ---")
            return
        except Exception as e:
            print(f"--- [结构指标测试任务] 数据库查询失败: {e} ---")
            return
        if not latest_metric:
            print("--- [结构指标测试任务] 验证失败：未在数据库中找到最新的结构指标记录。 ---")
            return
        latest_date = latest_metric.trade_time
        print(f"--- [结构指标测试任务] 开始验证最新日期 {latest_date} 的数据完整性 ---")
        data = latest_metric.__dict__
        empty_fields = []
        for field, value in data.items():
            # 忽略Django内部字段和外键ID
            if field.startswith('_') or field.endswith('_id'):
                continue
            # 检查 None 或 NaN
            if value is None or (isinstance(value, float) and pd.isna(value)):
                empty_fields.append(field)
        if not empty_fields:
            print(f"--- [结构指标测试任务] 验证通过！最新记录 ({latest_date}) 中没有发现空字段。 ---")
        else:
            print(f"--- [结构指标测试任务] 警告！在最新记录 ({latest_date}) 中发现以下空字段：---")
            for field in sorted(empty_fields):
                print(f"  - {field}")
    # 直接运行同步的测试逻辑
    run_test()
    # [代码新增结束]

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
        # 调用新的分析方法并接收返回的原始数据
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
# =================== 4. 数据维护任务 ==================
# =================================================================
@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.archive_historical_trade_data', queue='celery')
def archive_historical_trade_data(self, days_to_keep: int = 650):
    """
    归档含有 'trade_time' 字段的表的历史数据。
    - 保留最近 `days_to_keep` 个交易日的数据。
    - 将此之前的数据导出为独立的 SQL 文件。
    - 使用 xz -9 进行最高级别压缩。
    - 成功归档后从数据库中删除对应数据。
    """
    import os
    import subprocess
    from datetime import timedelta
    from django.conf import settings
    from django.apps import apps
    from django.db.models import Min
    logger.info(f"====== [历史数据归档任务] 启动，保留最近 {days_to_keep} 个交易日的数据 ======")
    # 1. 计算截止日期
    try:
        # 获取最近 days_to_keep 个交易日
        latest_trade_dates = TradeCalendar.get_latest_n_trade_dates(n=days_to_keep)
        if not latest_trade_dates or len(latest_trade_dates) < days_to_keep:
            logger.error(f"无法获取足够的交易日历数据 ({len(latest_trade_dates)}/{days_to_keep})，任务终止。")
            return {"status": "failed", "reason": "Insufficient trade calendar data."}
        # 保留数据的起始日期是第450个交易日，所以归档日期是这个日期之前的所有日期
        cutoff_date = latest_trade_dates[-1]
        logger.info(f"数据保留的起始日期为: {cutoff_date}。此日期之前的数据将被归档。")
        print(f"调试信息: 数据归档截止日期 (不含当天): {cutoff_date}")
    except Exception as e:
        logger.error(f"计算归档截止日期时出错: {e}", exc_info=True)
        return {"status": "failed", "reason": "Error calculating cutoff date."}
    # 2. 准备归档目录和数据库配置
    archive_dir = os.path.join(settings.BASE_DIR, 'db_archives')
    os.makedirs(archive_dir, exist_ok=True)
    logger.info(f"归档文件将保存至: {archive_dir}")
    db_config = settings.DATABASES['default']
    db_user = db_config['USER']
    db_password = db_config['PASSWORD']
    db_host = db_config['HOST']
    db_port = str(db_config.get('PORT', 3306))
    db_name = db_config['NAME']
    # 3. 动态查找所有带 'trade_time' 字段的模型
    target_models = []
    all_models = apps.get_models()
    for model in all_models:
        # 检查模型是否有 'trade_time' 字段，并且不是抽象模型
        if hasattr(model, '_meta') and not model._meta.abstract and 'trade_time' in [f.name for f in model._meta.get_fields()]:
            # 排除没有数据的模型
            if model.objects.exists():
                target_models.append(model)
    if not target_models:
        logger.info("未找到任何含有 'trade_time' 字段且有数据的表，任务结束。")
        return {"status": "skipped", "reason": "No target tables found."}
    logger.info(f"发现 {len(target_models)} 个目标模型需要处理。")
    print(f"调试信息: 待处理的模型列表: {[m._meta.db_table for m in target_models]}")
    # 4. 遍历模型，执行归档和删除
    success_count = 0
    failure_count = 0
    for model in target_models:
        table_name = model._meta.db_table
        logger.info(f"--- 开始处理表: {table_name} ---")
        try:
            # 检查是否有需要归档的数据
            min_date_obj = model.objects.filter(trade_time__lt=cutoff_date).aggregate(min_date=Min('trade_time'))
            min_date = min_date_obj.get('min_date')
            if not min_date:
                logger.info(f"表 {table_name} 中没有早于 {cutoff_date} 的数据，跳过。")
                continue
            # 格式化日期
            min_date_str = min_date.strftime('%Y%m%d')
            cutoff_date_str_for_file = (cutoff_date - timedelta(days=1)).strftime('%Y%m%d')
            # 构建文件名和命令
            archive_filename = f"{table_name}_{min_date_str}_to_{cutoff_date_str_for_file}.sql.xz"
            archive_filepath = os.path.join(archive_dir, archive_filename)
            where_clause = f"trade_time < '{cutoff_date.strftime('%Y-%m-%d')}'"
            # mysqldump 命令
            dump_cmd = [
                'mysqldump',
                f'--user={db_user}',
                f'--password={db_password}',
                f'--host={db_host}',
                f'--port={db_port}',
                '--single-transaction',
                '--skip-lock-tables',
                '--quick',
                db_name,
                table_name,
                f'--where={where_clause}'
            ]
            # xz 压缩命令
            compress_cmd = ['xz', '-9', '-c']
            logger.info(f"正在导出并压缩表 {table_name} 的数据...")
            print(f"调试信息: 导出命令 (密码已隐藏): mysqldump --user={db_user} ... --where=\"{where_clause}\"")
            with open(archive_filepath, 'wb') as f_out:
                # 使用管道连接 mysqldump 和 xz
                dump_process = subprocess.Popen(dump_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                compress_process = subprocess.Popen(compress_cmd, stdin=dump_process.stdout, stdout=f_out, stderr=subprocess.PIPE)
                # 等待 mysqldump 完成，并关闭其 stdout
                dump_process.stdout.close()
                # 获取 stderr 输出
                dump_stderr = dump_process.stderr.read().decode('utf-8')
                compress_stderr = compress_process.stderr.read().decode('utf-8')
                # 等待压缩进程完成
                compress_process.wait()
                dump_process.wait()
                if dump_process.returncode != 0:
                    logger.error(f"导出表 {table_name} 时 mysqldump 失败。返回码: {dump_process.returncode}。错误: {dump_stderr}")
                    os.remove(archive_filepath) # 删除不完整的文件
                    failure_count += 1
                    continue
                if compress_process.returncode != 0:
                    logger.error(f"压缩表 {table_name} 的导出文件时 xz 失败。返回码: {compress_process.returncode}。错误: {compress_stderr}")
                    os.remove(archive_filepath) # 删除不完整的文件
                    failure_count += 1
                    continue
            # 验证文件
            if not os.path.exists(archive_filepath) or os.path.getsize(archive_filepath) == 0:
                logger.error(f"归档文件 {archive_filepath} 创建失败或为空。")
                failure_count += 1
                continue
            logger.info(f"成功归档数据到: {archive_filepath}")
            # 删除数据
            logger.info(f"准备从表 {table_name} 中删除早于 {cutoff_date} 的数据...")
            with transaction.atomic():
                deleted_count, _ = model.objects.filter(trade_time__lt=cutoff_date).delete()
                logger.info(f"成功从表 {table_name} 中删除了 {deleted_count} 条历史记录。")
            success_count += 1
        except Exception as e:
            logger.error(f"处理表 {table_name} 时发生意外错误: {e}", exc_info=True)
            failure_count += 1
            continue
    logger.info("====== [历史数据归档任务] 完成 ======")
    logger.info(f"成功处理: {success_count} 个表, 失败: {failure_count} 个表。")
    return {"status": "completed", "success": success_count, "failed": failure_count}


