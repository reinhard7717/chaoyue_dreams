# 文件: tasks/stock_analysis_tasks.py
# 版本: V2.0 - 引擎切换版

import asyncio
from datetime import datetime
import logging
from celery import Celery
from asgiref.sync import async_to_sync
import numpy as np
import pandas as pd
from django.db import transaction
from chaoyue_dreams.celery import app as celery_app
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO
from dao_manager.tushare_daos.strategies_dao import StrategiesDAO

# ▼▼▼ 导入新的总指挥策略，并移除旧的策略导入 ▼▼▼
from services.chip_feature_calculator import ChipFeatureCalculator
from stock_models.stock_basic import StockInfo
from stock_models.time_trade import AdvancedChipMetrics, StockCyqPerf
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
            
            logger.info(f"[{stock_code}] 检测到 {len(unique_signal_types)} 种唯一的信号类型需要更新状态: {unique_signal_types}")

            for strategy_name, timeframe in unique_signal_types:
                logger.info(f"[{stock_code}] 准备更新策略状态摘要 for strategy '{strategy_name}' on timeframe '{timeframe}'...")
                async_to_sync(strategies_dao.update_strategy_state)(
                    stock_code=stock_code,
                    strategy_name=strategy_name,
                    timeframe=timeframe
                )
                logger.info(f"[{stock_code}] 策略 '{strategy_name}' ({timeframe}) 状态摘要更新完成。")

        return {"status": "success", "saved_count": save_count}

    except Exception as e:
        logger.error(f"执行核心策略逻辑 on {stock_code} 时出错: {e}", exc_info=True)
        # 在函数内部处理异常并返回错误信息，而不是向上抛出
        return {"status": "error", "reason": str(e)}

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


# ==============================================================================
# 调度任务 (Dispatcher Task) - 此部分无需修改，保持原样
# ==============================================================================
@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.schedule_precompute_advanced_chips', queue='SaveHistoryData_TimeTrade')
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
            precompute_advanced_chips_for_stock.s(stock_code).set(queue='precompute_chips').apply_async()
        
        logger.info(f"已为 {stock_count} 只股票调度 '高级筹码指标预计算' 任务。")
        return {"status": "started", "stock_count": stock_count}
    except Exception as e:
        logger.error(f"调度高级筹码预计算任务时出错: {e}", exc_info=True)
        return {"status": "failed", "reason": str(e)}

# ==============================================================================
# 执行任务 (Executor Task) - 【V3.0 最终版】
# ==============================================================================
@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.precompute_advanced_chips_for_stock', queue='celery')
def precompute_advanced_chips_for_stock(self, stock_code: str):
    """
    【执行器 V3.0 - 双重分表最终版】
    为单个股票计算并存储高级筹码指标。
    此版本同时处理 StockCyqChips 和 StockDailyData 的分表逻辑。
    """
    logger.info(f"[{stock_code}] 开始执行高级筹码指标预计算 (V3.0 双重分表版)...")
    try:
        stock_info = StockInfo.objects.get(stock_code=stock_code)
        dao = StockTimeTradeDAO()
        
        # --- 步骤 1: 高效获取所有需要的数据 (严格遵循分表逻辑) ---
        
        # 1.1 获取原始筹码分布数据 (动态分表)
        print(f"[{stock_code}] 使用DAO获取 [筹码分布] 分表模型...") # 使用print调试
        chip_model = dao.get_cyq_chips_model_by_code(stock_code)
        logger.info(f"[{stock_code}] 筹码数据将从表: {chip_model.__name__} 获取。")
        print(f"[{stock_code}] 筹码数据将从表: {chip_model.__name__} 获取。") # 使用print调试
        cyq_chips_data = pd.DataFrame.from_records(
            chip_model.objects.filter(stock=stock_info).values('trade_time', 'price', 'percent')
        )
        if cyq_chips_data.empty:
            logger.warning(f"[{stock_code}] 在表 {chip_model.__name__} 中找不到原始筹码数据，任务终止。")
            return {"status": "skipped", "reason": f"no raw chip data in {chip_model.__name__}"}
        cyq_chips_data['trade_time'] = pd.to_datetime(cyq_chips_data['trade_time']).dt.date

        # 1.2 获取日线行情数据 (动态分表)
        print(f"[{stock_code}] 使用DAO获取 [日线行情] 分表模型...") # 使用print调试
        daily_data_model = dao.get_daily_data_model_by_code(stock_code)
        logger.info(f"[{stock_code}] 日线数据将从表: {daily_data_model.__name__} 获取。")
        print(f"[{stock_code}] 日线数据将从表: {daily_data_model.__name__} 获取。") # 使用print调试
        daily_data = pd.DataFrame.from_records(
            daily_data_model.objects.filter(stock=stock_info).values('trade_time', 'volume', 'high', 'low')
        ).rename(columns={'volume': 'daily_turnover_volume', 'high': 'high_price', 'low': 'low_price'})
        if daily_data.empty:
            logger.warning(f"[{stock_code}] 在表 {daily_data_model.__name__} 中找不到日线数据，任务终止。")
            return {"status": "skipped", "reason": f"no daily data in {daily_data_model.__name__}"}
        daily_data['trade_time'] = pd.to_datetime(daily_data['trade_time']).dt.date

        # 1.3 获取基础筹码指标 (非分表)
        perf_data = pd.DataFrame.from_records(
            StockCyqPerf.objects.filter(stock=stock_info).values('trade_time', 'weight_avg', 'his_high')
        ).rename(columns={'his_high': 'close_price'})
        perf_data['trade_time'] = pd.to_datetime(perf_data['trade_time']).dt.date

        # 1.4 获取总流通股本 (非分表)
        total_chip_volume = stock_info.circulating_share_capital 
        if not total_chip_volume:
            logger.error(f"[{stock_code}] 缺少流通股本数据，无法计算绝对值指标！")
            return {"status": "failed", "reason": "missing circulating_share_capital"}

        # --- 步骤 2: 整合数据 ---
        # 以日线数据为基础，合并其他数据
        merged_df = daily_data.set_index('trade_time')
        merged_df = merged_df.join(perf_data.set_index('trade_time'), how='left')
        merged_df['prev_20d_close'] = merged_df['close_price'].shift(20)
        
        # --- 步骤 3: 循环计算每日指标 ---
        grouped_chips = cyq_chips_data.groupby('trade_time')
        all_metrics_list = []

        for trade_date, daily_chips_df in grouped_chips:
            if trade_date not in merged_df.index:
                continue
            
            context_data = merged_df.loc[trade_date].to_dict()
            context_data['total_chip_volume'] = total_chip_volume
            
            # 检查是否有NaN值，避免计算错误
            if pd.isna(context_data.get('close_price')) or pd.isna(context_data.get('weight_avg_cost')):
                continue

            calculator = ChipFeatureCalculator(daily_chips_df.sort_values(by='price'), context_data)
            daily_metrics = calculator.calculate_all_metrics()
            
            if daily_metrics:
                daily_metrics['trade_time'] = trade_date
                # TODO: 在这里集成龙虎榜、股东、题材等V4.0数据的获取和填充
                all_metrics_list.append(daily_metrics)

        if not all_metrics_list:
            logger.warning(f"[{stock_code}] 未能计算出任何高级指标。")
            return {"status": "skipped", "reason": "calculation resulted in no metrics"}

        # --- 步骤 4: 计算跨时间序列的指标 (如斜率) ---
        metrics_df = pd.DataFrame(all_metrics_list).set_index('trade_time').sort_index()
        
        if 'peak_cost' in metrics_df.columns:
            for period in [5, 20]:
                slope = metrics_df['peak_cost'].rolling(window=period, min_periods=2).apply(
                    lambda x: np.polyfit(range(len(x)), x.dropna(), 1)[0] if len(x.dropna()) > 1 else np.nan, raw=False
                )
                metrics_df[f'peak_cost_slope_{period}d'] = slope
            metrics_df['peak_cost_accel_5d'] = metrics_df['peak_cost_slope_5d'].diff()

        if 'concentration_90pct' in metrics_df.columns:
            metrics_df['concentration_90pct_slope_5d'] = metrics_df['concentration_90pct'].rolling(5).mean().diff()

        # --- 步骤 5: 存储到数据库 ---
        records_to_create = []
        for trade_date, row in metrics_df.iterrows():
            record_data = row.dropna().to_dict()
            records_to_create.append(AdvancedChipMetrics(stock=stock_info, trade_time=trade_date, **record_data))

        with transaction.atomic():
            AdvancedChipMetrics.objects.filter(stock=stock_info).delete()
            AdvancedChipMetrics.objects.bulk_create(records_to_create, batch_size=5000)
        
        logger.info(f"[{stock_code}] 成功！已为 {len(records_to_create)} 个交易日计算并存储了高级筹码指标。")
        print(f"[{stock_code}] 成功！已为 {len(records_to_create)} 个交易日计算并存储了高级筹码指标。") # 使用print调试
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
