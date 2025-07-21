# 文件: tasks/stock_analysis_tasks.py
# 版本: V2.0 - 引擎切换版

import asyncio
from datetime import datetime, timedelta
import logging
from celery import Celery
from asgiref.sync import async_to_sync
import numpy as np
import pandas as pd
from django.db import transaction
from chaoyue_dreams.celery import app as celery_app
from dao_manager.tushare_daos.fund_flow_dao import FundFlowDao
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO
from dao_manager.tushare_daos.strategies_dao import StrategiesDAO

# ▼▼▼ 导入新的总指挥策略，并移除旧的策略导入 ▼▼▼
from services.chip_feature_calculator import ChipFeatureCalculator
from stock_models.stock_basic import StockInfo
from stock_models.time_trade import AdvancedChipMetrics, StockCyqPerf, StockDailyBasic
from strategies.multi_timeframe_trend_strategy import MultiTimeframeTrendStrategy
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



# ▼▼▼ 为调试任务增加更详细的注释，解释同步调用的原因 ▼▼▼
@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.debug_single_stock_analysis', queue='debug_tasks')
def debug_single_stock_analysis(self, stock_code: str):
    """
    【V1.2 - 专用调试任务，修复死锁问题】
    对单个股票执行最详细的策略分析，用于问题排查。
    现在直接调用核心逻辑函数，避免Celery死锁。
    """
    logger.info("="*80)
    logger.info(f"--- [调试任务启动] ---")
    logger.info(f"股票代码: {stock_code}")
    
    trade_date_str = datetime.now().strftime('%Y-%m-%d')
    logger.info(f"分析日期: {trade_date_str}")

    stock_time_trade_dao = StockTimeTradeDAO()
    try:
        latest_daily_quote = async_to_sync(stock_time_trade_dao.get_latest_daily_quote)(stock_code)
        if latest_daily_quote:
            pct_chg = latest_daily_quote.get('pct_chg', 0)
            close_price = latest_daily_quote.get('close', 'N/A')
            trade_date_db = latest_daily_quote.get('trade_date', 'N/A')
            logger.info(f"数据库最新行情: 日期={trade_date_db}, 收盘价={close_price}, 涨跌幅={pct_chg}%")
            if pct_chg > 0:
                logger.info(f"诊断结论: [{stock_code}] 今日为上涨状态，是理想的调试对象。")
            else:
                logger.info(f"诊断结论: [{stock_code}] 今日为下跌或平盘状态。")
        else:
            logger.warning(f"未能获取到 [{stock_code}] 的最新日线行情。")
    except Exception as e:
        logger.error(f"获取 [{stock_code}] 最新行情时出错: {e}", exc_info=True)

    logger.info("无论行情如何，都将强制执行详细的策略分析...")
    logger.info("="*80)

    try:
        # 【关键修改】: 直接调用普通的Python函数，而不是Celery任务。
        # 这样代码会在这里同步执行，日志连续，且完全避免了Celery的死锁问题。
        result = _execute_strategy_logic(stock_code, trade_date_str)
        
        logger.info(f"--- [调试任务完成] ---")
        logger.info(f"股票 [{stock_code}] 的策略分析执行完毕。")
        logger.info(f"返回结果: {result}")
        logger.info("="*80)
        return {"status": "success", "stock_code": stock_code, "details": result}
    except Exception as e:
        # 这里的异常捕获现在是双重保险，因为_execute_strategy_logic内部已经有try-except
        logger.error(f"在调试任务中调用 '_execute_strategy_logic' 时发生严重错误: {e}", exc_info=True)
        logger.info("="*80)
        return {"status": "error", "stock_code": stock_code, "reason": str(e)}


@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.debug_stock_over_period', queue='debug_tasks')
def debug_stock_over_period(self, stock_code: str, start_date: str, end_date: str):
    """
    【V-Debug 专用历史回溯任务】
    对单个股票在指定的历史时间段内，逐日运行策略分析并打印详细日志。
    - 调用 MultiTimeframeTrendStrategy.debug_run_for_period 方法。
    - 不会向数据库写入任何数据，纯日志分析。
    """
    logger.info("="*80)
    logger.info(f"--- [历史回溯调试任务启动] ---")
    logger.info(f"  - 股票代码: {stock_code}")
    logger.info(f"  - 分析时段: {start_date} to {end_date}")
    logger.info("="*80)

    try:
        # 1. 实例化总指挥策略
        strategy_orchestrator = MultiTimeframeTrendStrategy()

        # 2. 同步执行我们新创建的异步调试方法
        async_to_sync(strategy_orchestrator.debug_run_for_period)(
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date
        )

        logger.info(f"--- [历史回溯调试任务完成] ---")
        logger.info(f"股票 [{stock_code}] 的回溯分析执行完毕。")
        logger.info("="*80)
        return {"status": "success", "stock_code": stock_code, "period": f"{start_date}-{end_date}"}

    except Exception as e:
        logger.error(f"在执行历史回溯任务 for {stock_code} 时发生严重错误: {e}", exc_info=True)
        logger.info("="*80)
        return {"status": "error", "stock_code": stock_code, "reason": str(e)}

# ▼▼▼ 将核心业务逻辑剥离到一个独立的、可复用的函数中 ▼▼▼
def _execute_strategy_logic(stock_code: str, trade_date: str):
    """
    【V1.0 - 核心策略执行逻辑】
    这是一个普通的同步函数，包含了策略分析和保存的完整流程。
    它可以被任何Celery任务或代码直接调用。
    """
    logger.info(f"[{stock_code}] 开始执行核心策略逻辑 for date {trade_date}")
    try:
        # 1. 实例化总指挥策略和DAO
        strategy_orchestrator = MultiTimeframeTrendStrategy()
        strategies_dao = StrategiesDAO()

        analysis_end_time = f"{trade_date} 16:00:00"

        # 2. 调用总指挥的 run_for_stock 方法
        db_records = async_to_sync(strategy_orchestrator.run_for_stock)(
            stock_code=stock_code,
            trade_time=analysis_end_time
        )

        if not db_records:
            logger.info(f"[{stock_code}] 策略运行完成，但未触发任何需要记录的信号。")
            return {"status": "success", "saved_count": 0, "reason": "No DB records to save"}

        # 3. 保存到数据库
        save_count = async_to_sync(strategies_dao.save_strategy_signals)(db_records)
        logger.info(f"[{stock_code}] 成功保存 {save_count} 条 'multi_timeframe_trend_strategy' 信号。")
        
        # 4. 更新策略状态摘要
        if save_count > 0:
            unique_signal_types = set()
            for record in db_records:
                strategy_name = record.get('strategy_name')
                timeframe = record.get('timeframe')
                if strategy_name and timeframe:
                    unique_signal_types.add((strategy_name, timeframe))
            
            # logger.info(f"[{stock_code}] 检测到 {len(unique_signal_types)} 种唯一的信号类型需要更新状态: {unique_signal_types}")

            for strategy_name, timeframe in unique_signal_types:
                # logger.info(f"[{stock_code}] 准备更新策略状态摘要 for strategy '{strategy_name}' on timeframe '{timeframe}'...")
                async_to_sync(strategies_dao.update_strategy_state)(
                    stock_code=stock_code,
                    strategy_name=strategy_name,
                    timeframe=timeframe
                )
                # logger.info(f"[{stock_code}] 策略 '{strategy_name}' ({timeframe}) 状态摘要更新完成。")

        return {"status": "success", "saved_count": save_count}

    except Exception as e:
        logger.error(f"执行核心策略逻辑 on {stock_code} 时出错: {e}", exc_info=True)
        # 在函数内部处理异常并返回错误信息，而不是向上抛出
        return {"status": "error", "reason": str(e)}

# ▼▼▼ 创建一个全新的、调用多时间框架策略的Celery任务 ▼▼▼
@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.run_multi_timeframe_strategy', queue='calculate_strategy')
def run_multi_timeframe_strategy(self, stock_code: str, trade_date: str):
    """
    【V2.0 - 逻辑分离版】
    Celery任务封装器，调用核心策略执行逻辑。
    """
    # 直接调用核心逻辑函数
    return _execute_strategy_logic(stock_code, trade_date)

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.analyze_all_stocks', queue='celery')
def analyze_all_stocks(self):
    """
    【V2.0 引擎切换版】
    调度所有股票分析任务，现在调用新的多时间框架策略入口。
    """
    try:
        logger.info("开始调度所有股票的分析任务 (V2.0 引擎切换版)")
        favorite_codes, non_favorite_codes = asyncio.run(_get_all_relevant_stock_codes_for_processing())
        if not non_favorite_codes and not favorite_codes:
            logger.warning("未找到任何股票数据，任务终止")
            return {"status": "failed", "reason": "no stocks found"}
        stock_count = len(favorite_codes) + len(non_favorite_codes)
        logger.info(f"找到 {stock_count} 只股票待分析.")
        
        trade_time_str = datetime.now().strftime('%Y-%m-%d')
        logger.info(f"所有任务将使用统一的分析截止日期: {trade_time_str}")
        
        # ▼▼▼ 将调度的任务从旧的 run_trend_follow_strategy 更换为新的 run_multi_timeframe_strategy ▼▼▼
        # --- 为自选股调度新任务 ---
        for stock_code in favorite_codes:
            run_multi_timeframe_strategy.s(stock_code, trade_time_str).set(queue='favorite_calculate_strategy').apply_async()
        
        # --- 为非自选股调度新任务 ---
        for stock_code in non_favorite_codes:
            run_multi_timeframe_strategy.s(stock_code, trade_time_str).set(queue='calculate_strategy').apply_async()
        
        
        logger.info(f"已为 {len(favorite_codes)} 只自选股调度 'run_multi_timeframe_strategy' 任务")
        logger.info(f"已为 {len(non_favorite_codes)} 只非自选股调度 'run_multi_timeframe_strategy' 任务")
        return {"status": "started",  "stock_count": stock_count}
    except Exception as e:
        logger.error(f"调度所有股票分析任务时出错: {e}", exc_info=True)
        return {"status": "failed", "reason": str(e)}

# ▼▼▼ “阿尔法猎手”的Celery后台任务 ▼▼▼
@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.run_alpha_hunter_for_stock', queue='debug_tasks')
def run_alpha_hunter_for_stock(self, stock_code: str):
    """
    【V118.2 阿尔法猎手后台任务】
    对单个股票运行全历史回测，自动发现策略未能捕捉的“黄金上涨波段”，并生成详细的情报档案。
    这是一个计算密集型任务，应在专用的长时任务队列中运行。
    """
    logger.info("="*80)
    logger.info(f"--- [阿尔法猎手后台任务启动] ---")
    logger.info(f"  - 股票代码: {stock_code}")
    logger.info("="*80)

    try:
        # 1. 实例化总指挥策略
        strategy_orchestrator = MultiTimeframeTrendStrategy()

        # 2. 同步执行我们新创建的异步入口方法
        async_to_sync(strategy_orchestrator.run_alpha_hunter)(stock_code=stock_code)

        logger.info(f"--- [阿尔法猎手后台任务完成] ---")
        logger.info(f"股票 [{stock_code}] 的策略盲点扫描执行完毕。")
        logger.info("="*80)
        return {"status": "success", "stock_code": stock_code}

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
        favorite_codes, non_favorite_codes = asyncio.run(_get_all_relevant_stock_codes_for_processing())
        
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
        favorite_codes, non_favorite_codes = asyncio.run(_get_all_relevant_stock_codes_for_processing())
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
    - 新增: 在合并数据前，打印各源数据表的可用天数，便于诊断数据完整性问题。
    """
    mode = "增量更新" if is_incremental else "全量刷新"
    logger.info(f"[{stock_code}] 开始执行高级筹码指标预计算 (V10.1 数据源诊断版, 模式: {mode})...")
    
    try:
        stock_info = StockInfo.objects.get(stock_code=stock_code)
        time_trade_dao = StockTimeTradeDAO()
        fund_flow_dao = FundFlowDao()
        
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

        # 1. 获取筹码分布数据
        chip_model = time_trade_dao.get_cyq_chips_model_by_code(stock_code)
        cyq_chips_data = get_data(chip_model, fields=('trade_time', 'price', 'percent'))
        if cyq_chips_data.empty:
            logger.warning(f"[{stock_code}] 在指定范围内找不到原始筹码数据，任务终止。")
            return {"status": "skipped", "reason": "no raw chip data in range"}
        cyq_chips_data['trade_time'] = pd.to_datetime(cyq_chips_data['trade_time']).dt.date

        # 数据归一化
        daily_sums = cyq_chips_data.groupby('trade_time')['percent'].transform('sum')
        mask_sum_to_one = np.isclose(daily_sums, 1.0, atol=0.1)
        if mask_sum_to_one.any():
            cyq_chips_data.loc[mask_sum_to_one, 'percent'] *= 100
        
        # 2. 获取每日行情数据
        daily_data_model = time_trade_dao.get_daily_data_model_by_code(stock_code)
        daily_data = get_data(daily_data_model, fields=('trade_time', 'close_qfq', 'vol', 'high_qfq', 'low_qfq'))
        daily_data['trade_time'] = pd.to_datetime(daily_data['trade_time']).dt.date
        
        # 3. 获取每日基础指标数据
        daily_basic_data = get_data(StockDailyBasic, fields=('trade_time', 'float_share'))
        daily_basic_data['trade_time'] = pd.to_datetime(daily_basic_data['trade_time']).dt.date

        # 4. 获取筹码性能数据
        perf_data = get_data(StockCyqPerf, fields=('trade_time', 'weight_avg'))
        perf_data['trade_time'] = pd.to_datetime(perf_data['trade_time']).dt.date

        # 5. 获取日线资金流数据
        fund_flow_model = fund_flow_dao.get_fund_flow_model_by_code(stock_code)
        # 定义需要获取的资金流字段
        fund_flow_fields = (
            'trade_time', 
            'buy_sm_vol', 'buy_sm_amount', 'sell_sm_vol', 'sell_sm_amount',
            'buy_md_vol', 'buy_md_amount', 'sell_md_vol', 'sell_md_amount',
            'buy_lg_vol', 'buy_lg_amount', 'sell_lg_vol', 'sell_lg_amount',
            'buy_elg_vol', 'buy_elg_amount', 'sell_elg_vol', 'sell_elg_amount',
            'net_mf_vol' # net_mf_amount 也可以加上，但我们会自己计算
        )
        fund_flow_data = get_data(fund_flow_model, fields=fund_flow_fields)
        if not fund_flow_data.empty:
            fund_flow_data['trade_time'] = pd.to_datetime(fund_flow_data['trade_time']).dt.date

        # 计算每个数据源的独立天数
        cyq_days = cyq_chips_data['trade_time'].nunique()
        daily_days = len(daily_data)
        basic_days = len(daily_basic_data)
        perf_days = len(perf_data)
        fund_flow_days = len(fund_flow_data) if not fund_flow_data.empty else 0
        logger.info(f"[{stock_code}] 数据源诊断: 筹码分布({cyq_days}天), 行情({daily_days}天), 基础({basic_days}天), 性能({perf_days}天), 资金流({fund_flow_days}天)")

        # 打印诊断日志
        logger.info(f"[{stock_code}] 数据源诊断: 筹码分布({cyq_days}天), 行情({daily_days}天), 基础({basic_days}天), 性能({perf_days}天)")

        # --- 开始数据处理和合并 ---
        daily_data['daily_turnover_volume'] = daily_data['vol'] * 100
        daily_data = daily_data.rename(columns={'close': 'close_price', 'high': 'high_price', 'low': 'low_price'}).drop(columns=['vol'])
        
        daily_basic_data['total_chip_volume'] = daily_basic_data['float_share'] * 10000
        daily_basic_data = daily_basic_data.drop(columns=['float_share'])

        perf_data = perf_data.rename(columns={'weight_avg': 'weight_avg_cost'})

        # 按正确的顺序合并数据
        merged_df = pd.merge(cyq_chips_data, daily_data, on='trade_time', how='inner')
        merged_df = pd.merge(merged_df, daily_basic_data, on='trade_time', how='inner')
        merged_df = pd.merge(merged_df, perf_data, on='trade_time', how='inner')
        if not fund_flow_data.empty:
            merged_df = pd.merge(merged_df, fund_flow_data, on='trade_time', how='left')
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



# 保留旧的任务入口以实现兼容性，但调度器不再调用它
@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.run_trend_follow_strategy', queue='calculate_strategy')
def run_trend_follow_strategy(self, stock_code: str, trade_date: str):
    logger.warning(f"[{stock_code}] 正在调用已废弃的 'run_trend_follow_strategy' 任务入口。请尽快迁移到 'run_multi_timeframe_strategy'。")
    # 为了避免意外，这里可以直接转发到新任务
    return run_multi_timeframe_strategy.s(stock_code, trade_date).apply(task_id=self.request.id).get()
