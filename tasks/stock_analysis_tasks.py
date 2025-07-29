# 文件: tasks/stock_analysis_tasks.py
# 版本: V2.0 - 引擎切换版

import asyncio
from datetime import datetime, timedelta
import logging
from celery import Celery
from celery import group, chain
from asgiref.sync import async_to_sync
import numpy as np
import pandas as pd
from django.db import transaction
from chaoyue_dreams.celery import app as celery_app
from dao_manager.tushare_daos.fund_flow_dao import FundFlowDao
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO
from dao_manager.tushare_daos.strategies_dao import StrategiesDAO

from django_celery_beat.models import PeriodicTask, CrontabSchedule
from intraday_engine.orchestrator import IntradayEngineOrchestrator


# ▼▼▼ 导入新的总指挥策略，并移除旧的策略导入 ▼▼▼
from services.chip_feature_calculator import ChipFeatureCalculator
from stock_models.stock_basic import StockInfo
from stock_models.time_trade import AdvancedChipMetrics, StockCyqPerf, StockDailyBasic
from strategies.multi_timeframe_trend_strategy import MultiTimeframeTrendStrategy
from utils.cache_manager import CacheManager
# from strategies.trend_following_strategy import TrendFollowStrategy # 不再直接调用

logger = logging.getLogger('tasks')

async def _get_all_relevant_stock_codes_for_processing():
    # ... 此函数保持不变 ...
    stock_basic_dao = StockBasicInfoDao()
    favorite_stock_codes = set()
    all_stock_codes = set()
    try:
        favorite_stocks = await stock_basic_dao.get_all_favorite_stocks()
        for fav in favorite_stocks:
            favorite_stock_codes.add(fav.get("stock_code"))
    except Exception as e:
        logger.error(f"获取自选股列表时出错: {e}", exc_info=True)
    try:
        all_stocks = await stock_basic_dao.get_stock_list()
        for stock in all_stocks:
            if not stock.stock_code.endswith('.BJ'):
                all_stock_codes.add(stock.stock_code)
    except Exception as e:
        logger.error(f"获取全市场股票列表时出错: {e}", exc_info=True)
    non_favorite_stock_codes = list(all_stock_codes - favorite_stock_codes)
    favorite_stock_codes_list = list(favorite_stock_codes)
    favorite_stock_codes_list = sorted(favorite_stock_codes_list)
    non_favorite_stock_codes = sorted(non_favorite_stock_codes)
    if not favorite_stock_codes_list and not non_favorite_stock_codes:
        logger.warning("未能获取到任何需要处理的股票代码")
    return favorite_stock_codes_list, non_favorite_stock_codes

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.debug_stock_over_period', queue='debug_tasks')
def debug_stock_over_period(self, stock_code: str, start_date: str, end_date: str):
    """
    【V-Debug 专用历史回溯任务】
    对单个股票在指定的历史时间段内，逐日运行策略分析并打印详细日志。
    """
    logger.info("="*80)
    logger.info(f"--- [历史回溯调试任务启动] ---")
    logger.info(f"  - 股票代码: {stock_code}")
    logger.info(f"  - 分析时段: {start_date} to {end_date}")
    logger.info("="*80)

    async def main():
        # 创建CacheManager实例
        cache_manager = CacheManager()
        # 创建策略实例并注入cache_manager（如策略支持注入）
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

# ▼▼▼ 将核心业务逻辑剥离到一个独立的、可复用的函数中 ▼▼▼
def _execute_strategy_logic(stock_code: str, trade_date: str, latest_only: bool = False):
    """
    【V3.0 - 双模式引擎版】
    - 核心升级: 增加 `latest_only` 开关，用于在“全面战役”和“闪电突袭”模式间切换。
    """
    # 根据开关决定日志标题
    mode_str = "闪电突袭 (仅最新)" if latest_only else "全面战役 (全历史)"
    logger.info(f"[{stock_code}] 开始执行核心策略逻辑 ({mode_str}) for date {trade_date}")
    try:
        # 1. 实例化总指挥策略和DAO
        strategy_orchestrator = MultiTimeframeTrendStrategy()
        strategies_dao = StrategiesDAO()

        analysis_end_time = f"{trade_date} 16:00:00"

        # 2. 【核心改造】根据开关，调用不同的作战模式
        if latest_only:
            # 执行“闪电突袭”
            db_records = async_to_sync(strategy_orchestrator.run_for_latest_signal)(
                stock_code=stock_code,
                trade_time=analysis_end_time
            )
        else:
            # 执行“全面战役”
            db_records = async_to_sync(strategy_orchestrator.run_for_stock)(
                stock_code=stock_code,
                trade_time=analysis_end_time
            )

        if not db_records:
            logger.info(f"[{stock_code}] 策略运行完成，但未触发任何需要记录的信号。")
            return {"status": "success", "saved_count": 0, "reason": "No DB records to save"}

        # 3. 保存到数据库 (逻辑不变)
        save_count = async_to_sync(strategies_dao.save_strategy_signals)(db_records)
        logger.info(f"[{stock_code}] 成功保存 {save_count} 条 'multi_timeframe_trend_strategy' 信号。")
        
        # 4. 更新策略状态摘要 (逻辑不变)
        if save_count > 0:
            unique_signal_types = set()
            for record in db_records:
                strategy_name = record.get('strategy_name')
                timeframe = record.get('timeframe')
                if strategy_name and timeframe:
                    unique_signal_types.add((strategy_name, timeframe))
            
            for strategy_name, timeframe in unique_signal_types:
                async_to_sync(strategies_dao.update_strategy_state)(
                    stock_code=stock_code,
                    strategy_name=strategy_name,
                    timeframe=timeframe
                )

        return {"status": "success", "saved_count": save_count}

    except Exception as e:
        logger.error(f"执行核心策略逻辑 on {stock_code} 时出错: {e}", exc_info=True)
        return {"status": "error", "reason": str(e)}

# ▼▼▼ 创建一个全新的、调用多时间框架策略的Celery任务 ▼▼▼
@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.run_multi_timeframe_strategy', queue='calculate_strategy')
def run_multi_timeframe_strategy(self, stock_code: str, trade_date: str, latest_only: bool = False):
    """
    【V3.0 - 双模式引擎版】
    - 核心升级: 增加 `latest_only` 参数，并将它传递给核心逻辑函数。
    """
    async def main():
        # 创建CacheManager实例
        cache_manager = CacheManager()
        # 创建策略和DAO实例并注入cache_manager
        strategy_orchestrator = MultiTimeframeTrendStrategy(cache_manager)
        strategies_dao = StrategiesDAO(cache_manager)
        mode_str = "闪电突袭 (仅最新)" if latest_only else "全面战役 (全历史)"
        logger.info(f"[{stock_code}] 开始执行核心策略逻辑 ({mode_str}) for date {trade_date}")
        analysis_end_time = f"{trade_date} 16:00:00"
        if latest_only:
            db_records = await strategy_orchestrator.run_for_latest_signal(
                stock_code=stock_code,
                trade_time=analysis_end_time
            )
        else:
            db_records = await strategy_orchestrator.run_for_stock(
                stock_code=stock_code,
                trade_time=analysis_end_time
            )
        if not db_records:
            logger.info(f"[{stock_code}] 策略运行完成，但未触发任何需要记录的信号。")
            return {"status": "success", "saved_count": 0, "reason": "No DB records to save"}
        save_count = await strategies_dao.save_strategy_signals(db_records)
        logger.info(f"[{stock_code}] 成功保存 {save_count} 条 'multi_timeframe_trend_strategy' 信号。")
        if save_count > 0:
            unique_signal_types = set()
            for record in db_records:
                strategy_name = record.get('strategy_name')
                timeframe = record.get('timeframe')
                if strategy_name and timeframe:
                    unique_signal_types.add((strategy_name, timeframe))
            for strategy_name, timeframe in unique_signal_types:
                await strategies_dao.update_strategy_state(
                    stock_code=stock_code,
                    strategy_name=strategy_name,
                    timeframe=timeframe
                )
        return {"status": "success", "saved_count": save_count}
    try:
        return async_to_sync(main)()
    except Exception as e:
        logger.error(f"执行核心策略逻辑 on {stock_code} 时出错: {e}", exc_info=True)
        return {"status": "error", "reason": str(e)}

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.analyze_all_stocks', queue='celery')
def analyze_all_stocks(self):
    """
    【V4.0 追踪器联动版】
    - 核心升级: 使用 Celery chain 工作流，确保在所有股票分析任务完成后，
                自动触发 `update_favorite_stock_trackers` 任务。
    """
    try:
        logger.info("开始调度所有股票的分析任务 (V4.0 追踪器联动版)")
        
        # 动态导入，避免在Celery worker启动时就加载Django模型
        from dashboard.tasks import update_favorite_stock_trackers
        
        # 假设这个函数能正确获取股票代码列表
        favorite_codes, non_favorite_codes = async_to_sync(_get_all_relevant_stock_codes_for_processing)()
        
        if not non_favorite_codes and not favorite_codes:
            logger.warning("未找到任何股票数据，任务终止")
            return {"status": "failed", "reason": "no stocks found"}
            
        stock_count = len(favorite_codes) + len(non_favorite_codes)
        logger.info(f"找到 {stock_count} 只股票待分析.")
        
        trade_time_str = datetime.now().strftime('%Y-%m-%d')
        logger.info(f"所有任务将使用统一的分析截止日期: {trade_time_str}")
        
        # --- 步骤1: 创建所有股票分析任务的签名列表 ---
        # 使用 group 来并行执行所有分析任务
        analysis_tasks = []
        
        # 为自选股创建任务签名，并指定到高优先级队列
        for stock_code in favorite_codes:
            analysis_tasks.append(
                run_multi_timeframe_strategy.s(stock_code, trade_time_str, latest_only=True).set(queue='favorite_calculate_strategy')
            )
        
        # 为非自选股创建任务签名
        for stock_code in non_favorite_codes:
            analysis_tasks.append(
                run_multi_timeframe_strategy.s(stock_code, trade_time_str, latest_only=True).set(queue='calculate_strategy')
            )
            
        # 将所有分析任务打包成一个 group，让它们可以并行执行
        parallel_analysis_group = group(analysis_tasks)
        
        # --- 步骤2: 创建追踪器更新任务的签名 ---
        # 这个任务将在所有分析任务完成后执行
        update_tracker_task = update_favorite_stock_trackers.s().set(queue='celery') # 假设更新任务在默认队列

        # --- 步骤3: 使用 chain 将两个步骤链接起来 ---
        # (并行分析组 | 更新追踪器任务)
        # 这意味着，只有当 group 中的所有任务都成功完成后，update_tracker_task 才会开始执行。
        workflow = chain(parallel_analysis_group, update_tracker_task)
        
        # --- 步骤4: 启动工作流 ---
        workflow.apply_async()
        
        logger.info(f"已成功创建并启动工作流：")
        logger.info(f"  - 步骤1: 并行分析 {stock_count} 只股票 (自选: {len(favorite_codes)}, 其他: {len(non_favorite_codes)})")
        logger.info(f"  - 步骤2: 更新所有自选股持仓追踪器")
        
        return {"status": "workflow_started", "stock_count": stock_count}
        
    except Exception as e:
        logger.error(f"调度所有股票分析任务时出错: {e}", exc_info=True)
        return {"status": "failed", "reason": str(e)}

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.analyze_all_stocks_full_history', queue='celery')
def analyze_all_stocks_full_history(self):
    """
    【V4.0 战略预备队 - 全面战役模式】
    - 核心职责: 对【所有股票】的【全部历史】进行一次完整的、深度策略分析。
    - 作战模式: 强制调用核心引擎的“全面战役”模式 (latest_only=False)。
    - 资源警告: 这是一个资源密集型任务，仅应在必要时手动触发，切勿用于每日定时任务！
    - 专属队列: 为了防止堵塞日常任务，此任务被指派到独立的 'full_history_queue' 队列。
    """
    try:
        logger.info("====== [战略预备队] 接到总动员令！开始执行全面历史回溯任务 (V4.0) ======")
        favorite_codes, non_favorite_codes = async_to_sync(_get_all_relevant_stock_codes_for_processing)()
        if not non_favorite_codes and not favorite_codes:
            logger.warning("[战略预备队] 未找到任何股票数据，总动员任务终止")
            return {"status": "failed", "reason": "no stocks found"}
        stock_count = len(favorite_codes) + len(non_favorite_codes)
        logger.info(f"[战略预备队] 发现 {stock_count} 只股票需要进行全面历史分析。")
        
        # 对于历史回溯，使用当前日期作为名义上的截止日期
        trade_time_str = datetime.now().strftime('%Y-%m-%d')
        
        # --- 强制执行“全面战役”模式 (latest_only=False) ---
        # --- 并将任务派发到专属的“战略任务队列” (full_history_queue) ---

        # --- 为自选股调度“全面战役”任务 ---
        for stock_code in favorite_codes:
            # 调用 run_multi_timeframe_strategy，但将 latest_only 明确设置为 False
            run_multi_timeframe_strategy.s(stock_code, trade_time_str, latest_only=False).set(queue='calculate_strategy').apply_async()
        
        # --- 为非自选股调度“全面战役”任务 ---
        for stock_code in non_favorite_codes:
            run_multi_timeframe_strategy.s(stock_code, trade_time_str, latest_only=False).set(queue='calculate_strategy').apply_async()
        
        logger.info(f"[战略预备队] 已为 {stock_count} 只股票下达了“全面战役”指令。")
        return {"status": "started",  "stock_count": stock_count}
    except Exception as e:
        logger.error(f"[战略预备队] 执行总动员任务时发生严重错误: {e}", exc_info=True)
        return {"status": "failed", "reason": str(e)}



# --- 任务一：盘前准备任务 ---
@celery_app.task(name='tasks.stock_analysis_tasks.prepare_pools', queue='intraday_queue')
def prepare_pools():
    """
    盘前准备任务：为所有相关股票池生成当日的分钟K线和衍生特征。
    【V2.0 - 异步上下文修复版】
    """
    logger.info("开始执行盘前准备任务...")
    
    # 【核心修复】定义一个异步的 main 函数
    async def main():
        # 1. 在异步上下文中创建顶层的 CacheManager
        cache_manager_instance = CacheManager()
        params = {} # 从配置加载
        # 2. 创建 Orchestrator 实例，并注入 cache_manager
        orchestrator = IntradayEngineOrchestrator(params, cache_manager_instance)
        
        # 3. 执行业务逻辑
        success = await orchestrator.initialize_pools()
        if success:
            logger.info("盘前准备任务成功完成。")
        else:
            logger.error("盘前准备任务执行失败。")

    try:
        # 使用 async_to_sync 运行这个总的 main 函数
        async_to_sync(main)()
    except Exception as e:
        logger.error(f"盘前准备任务失败: {e}", exc_info=True)

# --- 任务二：核心盘中循环任务 ---
@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.run_cycle', queue='intraday_queue')
def run_cycle(self):
    """
    【最佳实践】核心盘中循环任务，只负责执行一轮分析。
    它从Redis读取状态，执行计算，并将结果写回Redis。
    """
    try:
        params = {} # 从配置加载
        orchestrator = IntradayEngineOrchestrator(params)
        
        # 直接执行循环，不再需要初始化
        signals = async_to_sync(orchestrator.run_single_cycle)()
        
        if signals:
            print(f"本轮循环产生 {len(signals)} 条交易信号。")
        
        return {"status": "success", "signals_found": len(signals)}
    except Exception as e:
        print(f"盘中引擎循环任务失败: {e}", exc_info=True)
        return {"status": "error", "reason": str(e)}

# --- 任务三：引擎调度器 (启动/停止) ---
# 这部分可以简化为一个管理命令或在Django Admin中手动操作，
# 但用Celery任务来自动化是更佳实践。
@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.realtime_engine_scheduler', queue='celery')
def realtime_engine_scheduler(self, action: str):
    """
    【最佳实践】统一的引擎调度器，负责启动和停止盘中循环任务。
    """
    task_name = 'intraday-engine-main-loop'
    
    if action == 'start':
        logger.info("调度器：正在启动盘中引擎...")
        schedule, _ = CrontabSchedule.objects.get_or_create(
            minute='*', hour='9-11, 13-14', day_of_week='1-5',
            month_of_year='*', timezone='Asia/Shanghai'
        )
        PeriodicTask.objects.update_or_create(
            name=task_name,
            defaults={
                'task': 'tasks.intraday_engine.run_cycle',
                'crontab': schedule,
                'enabled': True,
                'queue': 'intraday_queue'
            }
        )
        logger.info("调度器：盘中引擎已设置为每分钟运行。")
        return {"status": "started"}
        
    elif action == 'stop':
        logger.info("调度器：正在停止盘中引擎...")
        try:
            task = PeriodicTask.objects.get(name=task_name)
            task.enabled = False
            task.save()
            logger.info("调度器：盘中引擎的定时任务已禁用。")
            return {"status": "stopped"}
        except PeriodicTask.DoesNotExist:
            logger.warning(f"未找到名为 '{task_name}' 的定时任务，无需停止。")
            return {"status": "not_found"}
    else:
        return {"status": "error", "reason": "Invalid action"}


# ▼▼▼ “阿尔法猎手”的Celery后台任务 ▼▼▼
@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.run_alpha_hunter_for_stock', queue='debug_tasks')
def run_alpha_hunter_for_stock(self, stock_code: str):
    """
    【V118.2 阿尔法猎手后台任务】
    对单个股票运行全历史回测，自动发现策略未能捕捉的“黄金上涨波段”，并生成详细的情报档案。
    """
    logger.info("="*80)
    logger.info(f"--- [阿尔法猎手后台任务启动] ---")
    logger.info(f"  - 股票代码: {stock_code}")
    logger.info("="*80)

    async def main():
        # 创建CacheManager实例
        cache_manager = CacheManager()
        # 创建策略实例并注入cache_manager
        strategy_orchestrator = MultiTimeframeTrendStrategy(cache_manager)
        # 执行业务逻辑
        await strategy_orchestrator.run_alpha_hunter(stock_code=stock_code)
        logger.info(f"--- [阿尔法猎手后台任务完成] ---")
        logger.info(f"股票 [{stock_code}] 的策略盲点扫描执行完毕。")
        logger.info("="*80)
        return {"status": "success", "stock_code": stock_code}

    try:
        return async_to_sync(main)()
    except Exception as e:
        logger.error(f"在执行阿尔法猎手任务 for {stock_code} 时发生严重错误: {e}", exc_info=True)
        logger.info("="*80)
        return {"status": "error", "stock_code": stock_code, "reason": str(e)}


# ▼▼▼ “全市场阿尔法扫描”的Celery调度任务 ▼▼▼
@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.run_alpha_hunter_for_all_stocks', queue='celery')
def run_alpha_hunter_for_all_stocks(self):
    """
    【V118.3 全市场阿尔法扫描调度器】
    调度“阿尔法猎手”任务对所有相关股票进行全历史回测和策略盲点扫描。
    这是一个顶层调度任务，它本身不进行计算，只负责将单个股票的扫描任务分发到工作队列中。
    """
    try:
        logger.info("="*80)
        logger.info("--- [全市场阿尔法扫描调度器启动] ---")
        
        # 1. 获取所有需要进行扫描的股票代码
        #    我们复用现有的逻辑来获取自选股和非自选股列表
        #    注意：_get_all_relevant_stock_codes_for_processing 需要在异步上下文中运行
        favorite_codes, non_favorite_codes = async_to_sync(_get_all_relevant_stock_codes_for_processing)()
        
        if not non_favorite_codes and not favorite_codes:
            logger.warning("未找到任何股票数据，全市场扫描任务终止。")
            return {"status": "failed", "reason": "no stocks found"}
            
        stock_count = len(favorite_codes) + len(non_favorite_codes)
        logger.info(f"发现 {stock_count} 只股票待进行阿尔法扫描。")
        
        # 2. 将单个股票的扫描任务分发到专用的长时任务队列
        #    我们使用之前创建的 run_alpha_hunter_for_stock 任务
        
        # --- 为自选股调度扫描任务 (可以优先分配到性能更好的队列) ---
        for stock_code in favorite_codes:
            run_alpha_hunter_for_stock.s(stock_code).set(queue='debug_tasks').apply_async()
        
        # --- 为非自选股调度扫描任务 ---
        for stock_code in non_favorite_codes:
            run_alpha_hunter_for_stock.s(stock_code).set(queue='debug_tasks').apply_async()
        
        logger.info(f"已为 {len(favorite_codes)} 只自选股调度 'run_alpha_hunter_for_stock' 任务到 'debug_tasks' 队列。")
        logger.info(f"已为 {len(non_favorite_codes)} 只非自选股调度 'run_alpha_hunter_for_stock' 任务到 'debug_tasks' 队列。")
        logger.info("--- [全市场阿尔法扫描调度器完成] ---")
        logger.info("="*80)
        
        return {"status": "dispatched", "stock_count": stock_count}
        
    except Exception as e:
        logger.error(f"调度全市场阿尔法扫描任务时出错: {e}", exc_info=True)
        return {"status": "failed", "reason": str(e)}


# ==============================================================================
# 调度任务 (Dispatcher Task) - 此部分无需修改，保持原样
# ==============================================================================
@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.schedule_precompute_advanced_chips', queue='celery')
def schedule_precompute_advanced_chips(self):
    """
    【调度器】
    调度所有股票的高级筹码指标预计算任务。
    """
    try:
        logger.info("开始调度 [高级筹码指标预计算] 任务...")
        favorite_codes, non_favorite_codes = async_to_sync(_get_all_relevant_stock_codes_for_processing)()
        all_codes = favorite_codes + non_favorite_codes
        
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

# ==============================================================================
# 执行任务 (Executor Task) - 【V4.4.1 ORM调用修正版】
# ==============================================================================
@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.precompute_advanced_chips_for_stock', queue='SaveHistoryData_TimeTrade')
def precompute_advanced_chips_for_stock(self, stock_code: str, is_incremental: bool = True):
    """
    【执行器 V10.1 - 数据源诊断版】
    - 核心逻辑与V10.0完全一致，确保数据源和计算的正确性。
    """
    async def main():
        # 创建CacheManager实例
        cache_manager = CacheManager()
        # 创建DAO实例并注入cache_manager
        fund_flow_dao = FundFlowDao(cache_manager)
        time_trade_dao = StockTimeTradeDAO(cache_manager)
        # 其余业务逻辑保持不变，原地粘贴原有代码
        mode = "增量更新" if is_incremental else "全量刷新"
        logger.info(f"[{stock_code}] 开始执行高级筹码指标预计算 (V10.1 数据源诊断版, 模式: {mode})...")
        try:
            stock_info = StockInfo.objects.get(stock_code=stock_code)
            max_lookback_days = 160
            last_metric_date = None
            if is_incremental:
                try:
                    last_metric = AdvancedChipMetrics.objects.filter(stock=stock_info).latest('trade_time')
                    last_metric_date = last_metric.trade_time
                except AdvancedChipMetrics.DoesNotExist:
                    logger.info(f"[{stock_code}] 未找到任何历史指标，自动切换到全量刷新模式。")
                    is_incremental = False
            fetch_start_date = None
            if is_incremental and last_metric_date:
                fetch_start_date = last_metric_date - timedelta(days=max_lookback_days + 20)
            def get_data(model, fields: tuple, date_field='trade_time'):
                qs = model.objects.filter(stock=stock_info)
                if fetch_start_date:
                    filter_kwargs = {f'{date_field}__gte': fetch_start_date}
                    qs = qs.filter(**filter_kwargs)
                return pd.DataFrame.from_records(qs.values(*fields))
            chip_model = time_trade_dao.get_cyq_chips_model_by_code(stock_code)
            cyq_chips_data = get_data(chip_model, fields=('trade_time', 'price', 'percent'))
            if cyq_chips_data.empty:
                logger.warning(f"[{stock_code}] 在指定范围内找不到原始筹码数据，任务终止。")
                return {"status": "skipped", "reason": "no raw chip data in range"}
            cyq_chips_data['trade_time'] = pd.to_datetime(cyq_chips_data['trade_time']).dt.date
            daily_sums = cyq_chips_data.groupby('trade_time')['percent'].transform('sum')
            mask_sum_to_one = np.isclose(daily_sums, 1.0, atol=0.1)
            if mask_sum_to_one.any():
                cyq_chips_data.loc[mask_sum_to_one, 'percent'] *= 100
            daily_data_model = time_trade_dao.get_daily_data_model_by_code(stock_code)
            daily_data = get_data(daily_data_model, fields=('trade_time', 'close_qfq', 'vol', 'high_qfq', 'low_qfq'))
            daily_data['trade_time'] = pd.to_datetime(daily_data['trade_time']).dt.date
            daily_basic_data = get_data(StockDailyBasic, fields=('trade_time', 'float_share'))
            daily_basic_data['trade_time'] = pd.to_datetime(daily_basic_data['trade_time']).dt.date
            perf_data = get_data(StockCyqPerf, fields=('trade_time', 'weight_avg'))
            perf_data['trade_time'] = pd.to_datetime(perf_data['trade_time']).dt.date
            fund_flow_cy_model = fund_flow_dao.get_fund_flow_model_by_code(stock_code)
            cy_fields = (
                'trade_time', 'buy_sm_vol', 'buy_sm_amount', 'sell_sm_vol', 'sell_sm_amount',
                'buy_md_vol', 'buy_md_amount', 'sell_md_vol', 'sell_md_amount',
                'buy_lg_vol', 'buy_lg_amount', 'sell_lg_vol', 'sell_lg_amount',
                'buy_elg_vol', 'buy_elg_amount', 'sell_elg_vol', 'sell_elg_amount'
            )
            fund_flow_cy_data = get_data(fund_flow_cy_model, fields=cy_fields)
            if not fund_flow_cy_data.empty:
                fund_flow_cy_data['trade_time'] = pd.to_datetime(fund_flow_cy_data['trade_time']).dt.date
            fund_flow_ths_model = fund_flow_dao.get_fund_flow_ths_model_by_code(stock_code)
            ths_fields = ('trade_time', 'buy_lg_amount')
            fund_flow_ths_data = get_data(fund_flow_ths_model, fields=ths_fields)
            if not fund_flow_ths_data.empty:
                fund_flow_ths_data['trade_time'] = pd.to_datetime(fund_flow_ths_data['trade_time']).dt.date
                fund_flow_ths_data = fund_flow_ths_data.rename(columns={'buy_lg_amount': 'ths_buy_lg_amount'})
            fund_flow_dc_model = fund_flow_dao.get_fund_flow_dc_model_by_code(stock_code)
            dc_fields = ('trade_time', 'net_amount')
            fund_flow_dc_data = get_data(fund_flow_dc_model, fields=dc_fields)
            if not fund_flow_dc_data.empty:
                fund_flow_dc_data['trade_time'] = pd.to_datetime(fund_flow_dc_data['trade_time']).dt.date
                fund_flow_dc_data = fund_flow_dc_data.rename(columns={'net_amount': 'dc_net_amount'})
            cyq_days = cyq_chips_data['trade_time'].nunique()
            daily_days = len(daily_data)
            fund_cy_days = len(fund_flow_cy_data) if not fund_flow_cy_data.empty else 0
            fund_ths_days = len(fund_flow_ths_data) if not fund_flow_ths_data.empty else 0
            fund_dc_days = len(fund_flow_dc_data) if not fund_flow_dc_data.empty else 0
            logger.info(f"[{stock_code}] 数据源诊断: 筹码({cyq_days}天), 行情({daily_days}天), 资金流[CY]({fund_cy_days}天), [THS]({fund_ths_days}天), [DC]({fund_dc_days}天)")
            daily_data['daily_turnover_volume'] = daily_data['vol'] * 100
            daily_data = daily_data.rename(columns={'close_qfq': 'close_price', 'high_qfq': 'high_price', 'low_qfq': 'low_price'})
            daily_basic_data['total_chip_volume'] = daily_basic_data['float_share'] * 10000
            daily_basic_data = daily_basic_data.drop(columns=['float_share'])
            perf_data = perf_data.rename(columns={'weight_avg': 'weight_avg_cost'})
            merged_df = pd.merge(cyq_chips_data, daily_data, on='trade_time', how='inner')
            merged_df = pd.merge(merged_df, daily_basic_data, on='trade_time', how='inner')
            merged_df = pd.merge(merged_df, perf_data, on='trade_time', how='inner')
            if not fund_flow_cy_data.empty:
                merged_df = pd.merge(merged_df, fund_flow_cy_data, on='trade_time', how='left')
            if not fund_flow_ths_data.empty:
                merged_df = pd.merge(merged_df, fund_flow_ths_data, on='trade_time', how='left')
            if not fund_flow_dc_data.empty:
                merged_df = pd.merge(merged_df, fund_flow_dc_data, on='trade_time', how='left')
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
                if is_incremental and last_metric_date and trade_date <= last_metric_date:
                    continue
                context_data = daily_full_df.iloc[0].to_dict()
                chip_data_for_calc = daily_full_df[['price', 'percent']]
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
            if is_incremental and last_metric_date:
                past_metrics_df = pd.DataFrame.from_records(
                    AdvancedChipMetrics.objects.filter(
                        stock=stock_info, 
                        trade_time__gte=fetch_start_date, 
                        trade_time__lte=last_metric_date
                    ).values()
                ).set_index('trade_time')
                if not past_metrics_df.empty:
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
            for trade_date, row in records_to_save_df.iterrows():
                record_data = row.dropna().to_dict()
                if 'id' in record_data: del record_data['id']
                if 'stock_id' in record_data: del record_data['stock_id']
                records_to_create.append(AdvancedChipMetrics(stock=stock_info, trade_time=trade_date, **record_data))
            with transaction.atomic():
                if not is_incremental:
                    logger.info(f"[{stock_code}] 全量模式：删除所有旧数据...")
                    AdvancedChipMetrics.objects.filter(stock=stock_info).delete()
                AdvancedChipMetrics.objects.bulk_create(records_to_create, batch_size=5000)
            logger.info(f"[{stock_code}] 成功！模式[{mode}]下，为 {len(records_to_create)} 个交易日计算并存储了高级筹码指标。")
            return {"status": "success", "processed_days": len(records_to_create)}
        except StockInfo.DoesNotExist:
            logger.error(f"[{stock_code}] 在StockInfo中找不到该股票，任务终止。")
            return {"status": "failed", "reason": "stock_code not found in StockInfo"}
        except Exception as e:
            logger.error(f"[{stock_code}] 高级筹码指标预计算失败: {e}", exc_info=True)
            return {"status": "failed", "reason": str(e)}
    return async_to_sync(main)()


# 保留旧的任务入口以实现兼容性，但调度器不再调用它
@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.run_trend_follow_strategy', queue='calculate_strategy')
def run_trend_follow_strategy(self, stock_code: str, trade_date: str):
    logger.warning(f"[{stock_code}] 正在调用已废弃的 'run_trend_follow_strategy' 任务入口。请尽快迁移到 'run_multi_timeframe_strategy'。")
    # 为了避免意外，这里可以直接转发到新任务
    return run_multi_timeframe_strategy.s(stock_code, trade_date).apply(task_id=self.request.id).get()
