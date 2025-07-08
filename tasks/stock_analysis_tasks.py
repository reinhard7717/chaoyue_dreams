# 文件: tasks/stock_analysis_tasks.py
# 版本: V2.0 - 引擎切换版

import asyncio
from datetime import datetime
import json
import logging
from celery import Celery
from asgiref.sync import async_to_sync
from chaoyue_dreams.celery import app as celery_app
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO
from dao_manager.tushare_daos.strategies_dao import StrategiesDAO

# ▼▼▼ 导入新的总指挥策略，并移除旧的策略导入 ▼▼▼
from strategies.multi_timeframe_trend_strategy import MultiTimeframeTrendStrategy
from strategies.trend_following_strategy import TrendFollowStrategy
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


# ▼▼▼【代码新增】: 将核心业务逻辑剥离到一个独立的、可复用的函数中 ▼▼▼
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


def load_strategy_params(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
# ▼▼▼【代码更新】: 为调试任务增加更详细的注释，解释同步调用的原因 ▼▼▼
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
        # 1. 加载策略配置
        params = load_strategy_params('config/trend_follow_strategy.json')

        # 2. 初始化策略实例
        strategy = TrendFollowStrategy(params)

        # 3. <<<<<<<  执行新的调试方法  >>>>>>>
        # 定义要调试的股票、时间段和数据文件路径
        stock_to_debug = '000158.SZ'
        start_date = '2024-08-01'
        end_date = '2024-11-07'
        data_file_path = 'pasted_text_0.txt' # 确保这个文件和您的执行脚本在同一目录，或使用绝对路径

        # 调用调试方法
        strategy.debug_strategy_on_period(
            stock_code=stock_to_debug,
            start_date_str=start_date,
            end_date_str=end_date,
            data_path=data_file_path
        )
        
        # logger.info(f"--- [调试任务完成] ---")
        # logger.info(f"股票 [{stock_code}] 的策略分析执行完毕。")
        # logger.info(f"返回结果: {result}")
        # logger.info("="*80)
        # return {"status": "success", "stock_code": stock_code, "details": result}
    except Exception as e:
        # 这里的异常捕获现在是双重保险，因为_execute_strategy_logic内部已经有try-except
        logger.error(f"在调试任务中调用 '_execute_strategy_logic' 时发生严重错误: {e}", exc_info=True)
        logger.info("="*80)
        return {"status": "error", "stock_code": stock_code, "reason": str(e)}





# 保留旧的任务入口以实现兼容性，但调度器不再调用它
@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.run_trend_follow_strategy', queue='calculate_strategy')
def run_trend_follow_strategy(self, stock_code: str, trade_date: str):
    logger.warning(f"[{stock_code}] 正在调用已废弃的 'run_trend_follow_strategy' 任务入口。请尽快迁移到 'run_multi_timeframe_strategy'。")
    # 为了避免意外，这里可以直接转发到新任务
    return run_multi_timeframe_strategy.s(stock_code, trade_date).apply(task_id=self.request.id).get()
