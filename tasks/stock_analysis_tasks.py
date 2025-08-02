# 文件: tasks/stock_analysis_tasks.py
# 版本: V2.0 - 引擎切换版

import asyncio
from datetime import datetime, timedelta
import json
import logging
import traceback
import pytz
from asgiref.sync import sync_to_async
from celery import Celery
from celery import group, chain
from asgiref.sync import async_to_sync
import numpy as np
import pandas as pd
from django.db import transaction
from chaoyue_dreams.celery import app as celery_app
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO
from dao_manager.tushare_daos.strategies_dao import StrategiesDAO
from services.realtime_services import RealtimeServices
from strategies.realtime_strategy import RealtimeStrategy
from utils.cash_key import IntradayEngineCashKey
from decimal import Decimal, InvalidOperation


# ▼▼▼ 导入新的总指挥策略，并移除旧的策略导入 ▼▼▼
from services.chip_feature_calculator import ChipFeatureCalculator
from stock_models.stock_basic import StockInfo
from stock_models.time_trade import AdvancedChipMetrics, StockDailyBasic
from strategies.multi_timeframe_trend_strategy import MultiTimeframeTrendStrategy
from utils.cache_manager import CacheManager
# from strategies.trend_following_strategy import TrendFollowStrategy # 不再直接调用

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
        cache_manager = CacheManager()
        strategy_orchestrator = MultiTimeframeTrendStrategy(cache_manager)
        strategies_dao = StrategiesDAO(cache_manager)

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
    【V4.2 依赖注入修复版】
    - 核心修改: 保留了可复用的异步辅助函数，并通过依赖注入为其提供DAO实例，
                确保了代码的复用性和异步调用的正确性。
    - 工作流: 使用 Celery chain，确保在所有股票分析任务完成后，
              自动触发 `update_favorite_stock_trackers` 任务。
    """
    try:
        logger.info("开始调度所有股票的分析任务 (V4.2 依赖注入修复版)")
        
        # 动态导入，避免在Celery worker启动时就加载Django模型
        from dashboard.tasks import update_favorite_stock_trackers
        
        # 初始化用于接收结果的列表
        favorite_codes = []
        non_favorite_codes = []

        # 1. 定义一个异步 main 函数，用于安全地执行所有需要异步环境的操作
        async def main():
            # nonlocal 关键字允许内部函数修改外部函数的变量
            nonlocal favorite_codes, non_favorite_codes
            
            # 在异步上下文中创建 CacheManager 和 DAO
            cache_manager_instance = CacheManager()
            stock_basic_dao = StockBasicInfoDao(cache_manager_instance)
            
            # 调用改造后的辅助函数，并将DAO实例作为参数传递进去
            fav_codes, non_fav_codes = await _get_all_relevant_stock_codes_for_processing(stock_basic_dao)
            
            # 将获取到的结果赋值给外部变量
            favorite_codes.extend(fav_codes)
            non_favorite_codes.extend(non_fav_codes)

        # 2. 在同步代码中，安全地执行异步的 main 函数来准备数据
        async_to_sync(main)()
        
        # 3. 回到同步代码中，执行分派任务的逻辑
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
        update_tracker_task = update_favorite_stock_trackers.s().set(queue='celery')

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
    【V4.1 依赖注入修复版】
    """
    try:
        logger.info("====== [战略预备队] 接到总动员令！开始执行全面历史回溯任务 (V4.1) ======")
        
        favorite_codes = []
        non_favorite_codes = []

        async def main():
            nonlocal favorite_codes, non_favorite_codes
            cache_manager_instance = CacheManager()
            stock_basic_dao = StockBasicInfoDao(cache_manager_instance)
            fav_codes, non_fav_codes = await _get_all_relevant_stock_codes_for_processing(stock_basic_dao)
            favorite_codes.extend(fav_codes)
            non_favorite_codes.extend(non_fav_codes)

        async_to_sync(main)()

        if not non_favorite_codes and not favorite_codes:
            logger.warning("[战略预备队] 未找到任何股票数据，总动员任务终止")
            return {"status": "failed", "reason": "no stocks found"}
            
        stock_count = len(favorite_codes) + len(non_favorite_codes)
        logger.info(f"[战略预备队] 发现 {stock_count} 只股票需要进行全面历史分析。")
        
        trade_time_str = datetime.now().strftime('%Y-%m-%d')
        
        for stock_code in favorite_codes:
            run_multi_timeframe_strategy.s(stock_code, trade_time_str, latest_only=False).set(queue='calculate_strategy').apply_async()
        
        for stock_code in non_favorite_codes:
            run_multi_timeframe_strategy.s(stock_code, trade_time_str, latest_only=False).set(queue='calculate_strategy').apply_async()
        
        logger.info(f"[战略预备队] 已为 {stock_count} 只股票下达了“全面战役”指令。")
        return {"status": "started",  "stock_count": stock_count}
        
    except Exception as e:
        logger.error(f"[战略预备队] 执行总动员任务时发生严重错误: {e}", exc_info=True)
        return {"status": "failed", "reason": str(e)}


# ==============================================================================
#                盘中引擎任务
# ==============================================================================
# --- 任务一：盘前准备任务 ---
@celery_app.task(name='tasks.stock_analysis_tasks.prepare_pools', queue='intraday_queue')
def prepare_pools():
    """
    【V2.1 - 逻辑重构】盘前准备任务，负责生成当日的监控股票池并存入Redis。
    """
    print("【V2.1】开始执行盘前准备任务: 生成监控股票池...")
    
    async def main():
        cache_manager = CacheManager()
        # 初始化 RealtimeServices，它现在包含了股票池管理逻辑
        realtime_services = RealtimeServices(cache_manager)
        
        # 调用服务来更新股票池
        await realtime_services.update_and_cache_monitoring_pool()
        
        await cache_manager.close()

    try:
        async_to_sync(main)()
        print("【V2.1】盘前准备任务成功完成。")
    except Exception as e:
        logger.error(f"盘前准备任务(prepare_pools)失败: {e}", exc_info=True)
        raise

# --- 任务二：核心盘中循环任务 ---
@celery_app.task(name='tasks.stock_analysis_tasks.run_cycle', queue='intraday_queue')
def run_cycle():
    """
    【V3.3 - 逻辑重构】核心盘中批量循环任务。
    - 从Redis加载由prepare_pools任务生成的监控池，只处理池内股票。
    """
    print("【V3.3】开始执行核心盘中批量循环任务...")
    async def main():
        cache_manager = CacheManager()
        realtime_services = RealtimeServices(cache_manager)
        # 1. 从Redis加载当天要监控的股票池
        stock_codes_to_monitor = await realtime_services.get_monitoring_pool_from_cache()
        if not stock_codes_to_monitor:
            logger.error("监控股票池为空！盘前任务(prepare_pools)可能未成功执行。本次盘中循环任务终止。")
            await cache_manager.close()
            return
        # 2. 只对监控池中的股票进行处理
        await realtime_services.process_all_stocks_intraday_data(
            stock_codes=stock_codes_to_monitor,
            time_level='1T',
            trade_date=datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d')
        )
        await cache_manager.close()
    try:
        async_to_sync(main)()
        print(f"【V3.3】核心盘中批量循环任务成功完成。")
    except Exception as e:
        logger.error(f"【V3.3】核心盘中批量循环任务失败: {e}", exc_info=True)
        raise

@celery_app.task(name='tasks.run_realtime_strategy_for_stock', queue='cpu_intensive_queue')
def run_realtime_strategy_for_stock(calculated_data: list):
    """
    【V2.1 - 调试增强版】
    - 核心修改: 在 async def main() 内部增加了详细的调试打印和
                一个完整的 try...except 块，以捕获在数据转换或
                策略调用期间可能发生的任何静默错误。
    """
    if not calculated_data or not isinstance(calculated_data, list):
        print("    -> [策略执行器 V2.1] 上游任务失败或数据为空，直接退出。")
        return

    # 提前获取股票代码，以防数据列表为空导致后续出错
    stock_code = calculated_data[0].get('stock_code', 'Unknown')
    print(f"    -> [策略执行器 V2.1] 开始为 {stock_code} 执行策略分析...")
    
    cache_manager = CacheManager()
    cache_key_builder = IntradayEngineCashKey()

    async def main():
        # ▼▼▼ 核心修改：添加完整的异常捕获和调试打印 ▼▼▼
        try:
            print(f"      - 调试: 进入 async main 函数。准备将 {len(calculated_data)} 条记录转换为DataFrame。")
            
            # 1. 将 list of dicts 转换为 DataFrame
            df_minute = pd.DataFrame(calculated_data)
            print("      - 调试: DataFrame 创建成功。")
            
            if 'trade_time' not in df_minute.columns:
                print("      - 调试错误: DataFrame中缺少 'trade_time' 列！任务终止。")
                return

            df_minute['trade_time'] = pd.to_datetime(df_minute['trade_time'])
            print("      - 调试: 'trade_time' 列已转换为datetime对象。")

            df_minute.set_index('trade_time', inplace=True)
            print(f"      - 调试: 'trade_time' 已设置为索引。DataFrame现在有 {len(df_minute)} 行。")
            
            today_str = datetime.now().strftime('%Y-%m-%d')

            # 2. 初始化并运行策略
            strategy_params = {
                'min_data_points': 21,
                'breakout_vol_zscore': 2.5,
                'breakout_agg_zscore': 2.0,
                'max_price_cv': 0.005,
                'min_correlation': 0.6,
                'reversal_energy_ratio': 1.5,
            }
            strategy = RealtimeStrategy(strategy_params)
            print("      - 调试: RealtimeStrategy 实例已创建。准备调用 run_strategy...")
            
            daily_info = {'stock_code': stock_code}
            final_signal = strategy.run_strategy(df_minute, daily_info)
            print("      - 调试: strategy.run_strategy 调用完成。")

            # 3. 如果有信号，存入Redis
            if final_signal:
                print(f"    -> [策略执行器 V2.1] 股票 {stock_code} 触发信号: {final_signal.get('playbook')}")
                signals_redis_key = cache_key_builder.stock_signals_key(stock_code, today_str)
                signal_json = json.dumps(final_signal, default=str)
                timestamp = final_signal['entry_time'].timestamp()
                await cache_manager.zadd(signals_redis_key, {signal_json: timestamp})
                await cache_manager.expire(signals_redis_key, 86400)
                print(f"    -> [策略执行器 V2.1] 已将信号存入Redis键: {signals_redis_key}")
            else:
                # 这个打印现在应该能看到了
                print(f"    -> [策略执行器 V2.1] 股票 {stock_code} 未触发任何盘中信号。")

        except Exception as e:
            print(f"      - 调试错误: 在 async main 函数中发生严重异常!")
            print(f"        - 股票代码: {stock_code}")
            print(f"        - 异常类型: {type(e)}")
            print(f"        - 异常信息: {e}")
            print(f"        - 堆栈跟踪:")
            traceback.print_exc()
        # ▲▲▲ 修改结束 ▲▲▲

    async_to_sync(main)()

@celery_app.task(name='tasks.aggregate_intraday_results', queue='cpu_intensive_queue')
def aggregate_intraday_results(job_id: str, total_tasks: int):
    """
    【V7.0 - 两阶段解耦版】
    - 接收批次ID (job_id)，从Redis中收集所有该批次的结果。
    """
    print("\n" + "="*30)
    print(f"【聚合任务 V7.0】开始聚合批次 {job_id} 的结果...")
    
    # 实例化 CacheManager 和 Key 生成器
    cache_manager = CacheManager()
    cache_key_builder = IntradayEngineCashKey()

    # --- 从Redis中收集所有结果 ---
    async def gather_results():
        # 1. 获取该批次所有股票代码的键
        #    注意：这里需要一种方法来知道哪些股票代码属于这个job_id。
        #    为简单起见，我们假设可以通过某种模式匹配获取。
        #    一个更健壮的方法是在分派任务时将所有股票代码存入一个Redis Set。
        #    这里我们使用一个简化的方法：直接从Redis中scan键。
        #    注意：SCAN在生产环境中可能较慢，但对于调试和验证此逻辑是可行的。
        
        # 修正：我们不需要知道所有股票代码，因为计算任务已经把结果存好了。
        # 我们只需要一个方法来知道哪些键属于这个job_id。
        # 我们在计算任务中存入的键是 stock_calculated_data_key(stock_code, job_id)
        # 它的模式是 calc:stock:{stock_code}:intraday_full_metrics:{job_id}
        
        # 使用 SCAN 来查找所有匹配这个 job_id 的键
        pattern = f"calc:stock:*:intraday_full_metrics:{job_id}"
        print(f"  -> 正在使用 SCAN 模式 '{pattern}' 查找结果键...")
        result_keys = await cache_manager.scan_keys(pattern)
        
        print(f"  -> 找到了 {len(result_keys)} 个结果键。")
        if not result_keys:
            return []

        # 批量获取所有结果
        fetch_tasks = [cache_manager.get(key) for key in result_keys]
        all_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        
        # 过滤掉空结果和异常
        valid_results = [res for res in all_results if res and not isinstance(res, Exception) and isinstance(res, list)]
        return valid_results

    all_results_list = asyncio.run(gather_results())
    
    if not all_results_list:
        print("【聚合任务 V7.0】没有任何有效的计算结果，任务结束。")
        print("="*30 + "\n")
        return "Aggregation complete: No valid results."

    # --- 后续逻辑保持不变 ---
    # 注意：这里的 save_full_data_to_redis 实际上是多余的了，因为数据已经存好了。
    # 我们直接从 all_results_list 中提取 redis_keys 用于下一步。
    # 但为了最小化改动，我们暂时保留它，它会覆盖已有的键，没有副作用。
    redis_keys_for_signals = asyncio.run(save_full_data_to_redis(all_results_list))
    
    total_stocks = len(all_results_list)
    print(f"【聚合任务 V7.0】聚合完成。成功处理了 {total_stocks} 支股票的完整数据。")
    
    if redis_keys_for_signals:
        print(f"【聚合任务 V7.0】正在为 {len(redis_keys_for_signals)} 个数据键触发信号生成任务...")
        signal_generation_workflow = group(
            generate_signals_from_data.s(redis_key).set(queue='cpu_intensive_queue', routing_key='cpu_intensive_queue')
            for redis_key in redis_keys_for_signals
        )
        signal_generation_workflow.apply_async()
        print("【聚合任务 V7.0】信号生成任务已全部派发。")

    print("="*30 + "\n")
    return f"Aggregation and dispatch complete: {total_stocks} stocks processed."

async def save_full_data_to_redis(results: list) -> list:
    """
    (异步) 将每个股票的完整计算结果存入Redis。
    """
    cache_manager = CacheManager()
    # ▼▼▼ 核心修改：实例化我们自己的Key生成器 ▼▼▼
    cache_key_builder = IntradayEngineCashKey()
    # ▲▲▲ 修改结束 ▲▲▲
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    save_tasks = []
    redis_keys = []
    for stock_data_list in results:
        if not stock_data_list:
            continue
        
        stock_code = stock_data_list[0].get('stock_code')
        if not stock_code:
            continue
            
        # 使用新的方法生成键
        redis_key = cache_key_builder.stock_calculated_data_key(stock_code, today_str)
        redis_keys.append(redis_key)
        
        save_tasks.append(cache_manager.set(redis_key, stock_data_list, timeout=3600))
        
    await asyncio.gather(*save_tasks, return_exceptions=True)
    return redis_keys

# [修改后]
@celery_app.task(name='tasks.generate_signals_from_data', queue='cpu_intensive_queue')
def generate_signals_from_data(calculated_data_key: str):
    """
    【V1.1 - 真实Key版】
    1. 从Redis加载数据。
    2. 运行信号检测逻辑。
    3. 使用 IntradayEngineCashKey 将信号存入ZSET。
    """
    print(f"    -> [信号生成器] 开始处理数据键: {calculated_data_key}")
    asyncio.run(process_and_save_signals(calculated_data_key))

async def process_and_save_signals(calculated_data_key: str):
    cache_manager = CacheManager()
    # ▼▼▼ 核心修改：实例化我们自己的Key生成器 ▼▼▼
    cache_key_builder = IntradayEngineCashKey()
    # ▲▲▲ 修改结束 ▲▲▲
    
    data_list = await cache_manager.get(calculated_data_key)
    if not data_list:
        print(f"    -> [信号生成器] 数据键 {calculated_data_key} 中没有数据，跳过。")
        return

    df_minute = pd.DataFrame(data_list)
    df_minute['trade_time'] = pd.to_datetime(df_minute['trade_time'])
    stock_code = df_minute['stock_code'].iloc[0]
    today_str = datetime.now().strftime('%Y-%m-%d')

    # 信号检测逻辑 (保持不变)
    signals = []
    for _, row in df_minute.iterrows():
        if 'net_agg_vol_slope' in row and 'volume_zscore' in row and pd.notna(row['net_agg_vol_slope']) and pd.notna(row['volume_zscore']):
            if row['net_agg_vol_slope'] > 1.5 and row['volume_zscore'] > 2.0:
                signals.append({
                    'stock_code': stock_code, 'trade_time': row['trade_time'].isoformat(),
                    'signal_type': '强力资金流入', 'signal_level': '高',
                    'description': f"净主动买入斜率({row['net_agg_vol_slope']:.2f})且成交量Z-Score({row['volume_zscore']:.2f})均超阈值。",
                    'close_price': row['close'],
                })
        if 'energy_ratio_mean' in row and pd.notna(row['energy_ratio_mean']):
            if row['energy_ratio_mean'] > 2.5:
                signals.append({
                    'stock_code': stock_code, 'trade_time': row['trade_time'].isoformat(),
                    'signal_type': '委买能量突破', 'signal_level': '中',
                    'description': f"平均委买/委卖量比值达到 {row['energy_ratio_mean']:.2f}。",
                    'close_price': row['close'],
                })

    if not signals:
        return

    # 使用新的方法生成信号键
    signals_redis_key = cache_key_builder.stock_signals_key(stock_code, today_str)
    zadd_mapping = {
        signal: datetime.fromisoformat(signal['trade_time']).timestamp()
        for signal in signals
    }
    await cache_manager.zadd_and_trim(signals_redis_key, zadd_mapping, limit=100)
    print(f"    -> [信号生成器] 为 {stock_code} 成功存储 {len(signals)} 条信号。")

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
    【V118.4 依赖注入修复版】
    """
    try:
        logger.info("="*80)
        logger.info("--- [全市场阿尔法扫描调度器启动] (V118.4) ---")
        
        favorite_codes = []
        non_favorite_codes = []

        async def main():
            nonlocal favorite_codes, non_favorite_codes
            cache_manager_instance = CacheManager()
            stock_basic_dao = StockBasicInfoDao(cache_manager_instance)
            fav_codes, non_fav_codes = await _get_all_relevant_stock_codes_for_processing(stock_basic_dao)
            favorite_codes.extend(fav_codes)
            non_favorite_codes.extend(non_fav_codes)

        async_to_sync(main)()
        
        if not non_favorite_codes and not favorite_codes:
            logger.warning("未找到任何股票数据，全市场扫描任务终止。")
            return {"status": "failed", "reason": "no stocks found"}
            
        stock_count = len(favorite_codes) + len(non_favorite_codes)
        logger.info(f"发现 {stock_count} 只股票待进行阿尔法扫描。")
        
        for stock_code in favorite_codes:
            run_alpha_hunter_for_stock.s(stock_code).set(queue='debug_tasks').apply_async()
        
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
    【V2.0 依赖注入修复版】
    """
    try:
        logger.info("开始调度 [高级筹码指标预计算] 任务...")
        
        all_codes = []
        
        async def main():
            nonlocal all_codes
            cache_manager_instance = CacheManager()
            stock_basic_dao = StockBasicInfoDao(cache_manager_instance)
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

# ==============================================================================
# 执行任务 (Executor Task) - 【V4.4.1 ORM调用修正版】
# ==============================================================================
@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.precompute_advanced_chips_for_stock', queue='SaveHistoryData_TimeTrade')
def precompute_advanced_chips_for_stock(self, stock_code: str, is_incremental: bool = True):
    """
    【执行器 V10.3 - 依赖外置与全面异步化修复版】
    - 核心修复: 将DAO的创建移至同步上下文中，仅将DAO实例作为参数传入异步main函数，
                彻底解决了因在异步函数内创建复杂依赖导致的进程挂起问题。
    - 性能优化: 将所有同步的Django ORM调用都使用 sync_to_async 包装，并使用
                asyncio.gather 并发获取数据。
    """
    # 1. 【核心修改】在同步上下文中创建所有依赖对象
    try:
        cache_manager = CacheManager()
        # fund_flow_dao = FundFlowDao(cache_manager)
        time_trade_dao = StockTimeTradeDAO(cache_manager)
    except Exception as e:
        logger.error(f"[{stock_code}] 在创建DAO实例时失败，任务终止: {e}", exc_info=True)
        return {"status": "failed", "reason": "DAO initialization failed"}

    # 2. 定义异步 main 函数，它现在只接收依赖，不创建依赖
    async def main(time_dao, incremental_flag: bool):
        
        # 【核心修改 2】使用传入的参数 incremental_flag
        mode = "增量更新" if incremental_flag else "全量刷新"
        logger.info(f"[{stock_code}] 开始执行高级筹码指标预计算 (V10.4, 模式: {mode})...")
        
        # --- 将所有同步ORM操作包装成可复用的异步函数 ---
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
            
            # 如果 fields 不为 None，则解包；否则，不传参数调用 .values()
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

            # --- 并发获取所有数据源 ---
            chip_model = time_dao.get_cyq_chips_model_by_code(stock_code)
            daily_data_model = time_dao.get_daily_data_model_by_code(stock_code)
            data_tasks = {
                "cyq_chips": get_data_async(chip_model, stock_info, fields=('trade_time', 'price', 'percent'), start_date=fetch_start_date),
                "daily_data": get_data_async(daily_data_model, stock_info, fields=('trade_time', 'close_qfq', 'vol', 'high_qfq', 'low_qfq'), start_date=fetch_start_date),
                "daily_basic": get_data_async(StockDailyBasic, stock_info, fields=('trade_time', 'float_share'), start_date=fetch_start_date),
            }
            
            results = await asyncio.gather(*data_tasks.values())
            data_dfs = dict(zip(data_tasks.keys(), results))

            # --- 从这里开始，是您的原始Pandas数据处理逻辑 ---
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
                    # 确保百分比总和为100
                    percent_sum = chip_data_for_calc['percent'].sum()
                    if not np.isclose(percent_sum, 100.0) and percent_sum > 0:
                        normalized_percent = chip_data_for_calc['percent'] / percent_sum
                    else:
                        normalized_percent = chip_data_for_calc['percent'] / 100.0
                    
                    # 计算加权平均成本
                    weight_avg_cost = np.average(chip_data_for_calc['price'], weights=normalized_percent)
                    # 将计算结果注入到上下文中
                    context_data['weight_avg_cost'] = weight_avg_cost
                else:
                    # 如果当天没有筹码数据，跳过
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
                # 当需要获取所有字段时，不传递 fields 参数（它会使用默认值 None）
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
                # 遍历模型的所有字段，而不是依赖 .to_dict()
                for field_name in model_fields:
                    if field_name in row.index:
                        value = row[field_name]
                        
                        # 检查值是否是无效的浮点数 (NaN, inf, -inf)
                        if isinstance(value, float) and not np.isfinite(value):
                            record_data[field_name] = None
                        # 如果值是 None，则直接使用 None
                        elif pd.isna(value):
                            record_data[field_name] = None
                        # 如果是有效的浮点数，则进行处理
                        elif isinstance(value, float):
                            if abs(value) < 1e-10:
                                record_data[field_name] = Decimal('0.0')
                            else:
                                # 保留8位小数
                                record_data[field_name] = Decimal(str(round(value, 8)))
                        # 其他类型的值（如 int, bool, str）直接使用
                        else:
                            record_data[field_name] = value
                
                # 移除Django自动管理的字段
                record_data.pop('id', None)
                record_data.pop('stock', None) # stock 对象将单独传入
                
                if record_data: # 确保有数据可存
                    records_to_create.append(AdvancedChipMetrics(stock=stock_info, trade_time=trade_date, **record_data))
            
            # --- 异步保存到数据库 ---
            await save_metrics_async(stock_info, records_to_create, not incremental_flag)
            
            logger.info(f"[{stock_code}] 成功！模式[{mode}]下，为 {len(records_to_create)} 个交易日计算并存储了高级筹码指标。")
            return {"status": "success", "processed_days": len(records_to_create)}

        except StockInfo.DoesNotExist:
            logger.error(f"[{stock_code}] 在StockInfo中找不到该股票，任务终止。")
            return {"status": "failed", "reason": "stock_code not found in StockInfo"}
        except Exception as e:
            logger.error(f"[{stock_code}] 高级筹码指标预计算失败: {e}", exc_info=True)
            return {"status": "failed", "reason": str(e)}

    # 3. 调用 async_to_sync，并将依赖作为参数传入
    try:
        # 将创建好的DAO实例作为参数传递给main函数
        # result = async_to_sync(main)(fund_flow_dao, time_trade_dao, is_incremental)
        result = async_to_sync(main)(time_trade_dao, is_incremental)
        return result
    except Exception as e:
        logger.error(f"--- CATCHING EXCEPTION in precompute_advanced_chips_for_stock for {stock_code}: {e}", exc_info=True)
        raise

# 保留旧的任务入口以实现兼容性，但调度器不再调用它
@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.run_trend_follow_strategy', queue='calculate_strategy')
def run_trend_follow_strategy(self, stock_code: str, trade_date: str):
    logger.warning(f"[{stock_code}] 正在调用已废弃的 'run_trend_follow_strategy' 任务入口。请尽快迁移到 'run_multi_timeframe_strategy'。")
    # 为了避免意外，这里可以直接转发到新任务
    return run_multi_timeframe_strategy.s(stock_code, trade_date).apply(task_id=self.request.id).get()
