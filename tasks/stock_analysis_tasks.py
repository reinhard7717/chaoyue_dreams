# 文件: tasks/stock_analysis_tasks.py
# 版本: V2.0 - 引擎切换版

import asyncio
from datetime import datetime
import logging
from celery import Celery
from asgiref.sync import async_to_sync
from chaoyue_dreams.celery import app as celery_app
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from dao_manager.tushare_daos.strategies_dao import StrategiesDAO

# ▼▼▼ 导入新的总指挥策略，并移除旧的策略导入 ▼▼▼
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
            favorite_stock_codes.add(fav.stock_id)
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

# ▼▼▼ 创建一个全新的、调用多时间框架策略的Celery任务 ▼▼▼
@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.run_multi_timeframe_strategy', queue='calculate_strategy')
def run_multi_timeframe_strategy(self, stock_code: str, trade_date: str):
    """
    【V1.0 - 新版策略入口】
    调用 MultiTimeframeTrendStrategy，执行包含周线、日线、分钟线的完整协同分析。
    """
    logger.info(f"[{stock_code}] 开始执行 'run_multi_timeframe_strategy' on {stock_code} for date {trade_date}")
    try:
        # 1. 实例化总指挥策略和DAO
        #    MultiTimeframeTrendStrategy 在其内部处理所有配置加载和子策略实例化
        strategy_orchestrator = MultiTimeframeTrendStrategy()
        strategies_dao = StrategiesDAO()

        # 2. 调用总指挥的 run_for_stock 方法
        #    这是一个异步方法，所以需要用 async_to_sync 包装
        db_records = async_to_sync(strategy_orchestrator.run_for_stock)(
            stock_code=stock_code,
            trade_time=trade_date
        )

        if not db_records:
            logger.info(f"[{stock_code}] 策略运行完成，但未触发任何需要记录的信号。")
            return {"status": "success", "saved_count": 0, "reason": "No DB records to save"}

        # 3. 保存到数据库
        save_count = async_to_sync(strategies_dao.save_strategy_signals)(db_records)
        logger.info(f"[{stock_code}] 成功保存 {save_count} 条 'multi_timeframe_trend_strategy' 信号。")
        
        # ▼▼▼ 新增策略状态更新逻辑 ▼▼▼
        # 解释: 在成功保存信号后，立即调用DAO方法更新策略状态摘要表。
        # 这样可以确保摘要信息始终反映最新的信号情况。
        if save_count > 0:
            # 步骤1: 找出所有需要更新状态的唯一信号类型
            unique_signal_types = set()
            for record in db_records:
                strategy_name = record.get('strategy_name')
                timeframe = record.get('timeframe')
                if strategy_name and timeframe:
                    unique_signal_types.add((strategy_name, timeframe))
            
            logger.info(f"[{stock_code}] 检测到 {len(unique_signal_types)} 种唯一的信号类型需要更新状态: {unique_signal_types}")

            # 步骤2: 遍历每一种信号类型，并调用状态更新
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
        logger.error(f"执行 'run_multi_timeframe_strategy' on {stock_code} 时出错: {e}", exc_info=True)
        return {"status": "error", "reason": str(e)}

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

# 保留旧的任务入口以实现兼容性，但调度器不再调用它
@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.run_trend_follow_strategy', queue='calculate_strategy')
def run_trend_follow_strategy(self, stock_code: str, trade_date: str):
    logger.warning(f"[{stock_code}] 正在调用已废弃的 'run_trend_follow_strategy' 任务入口。请尽快迁移到 'run_multi_timeframe_strategy'。")
    # 为了避免意外，这里可以直接转发到新任务
    return run_multi_timeframe_strategy.s(stock_code, trade_date).apply(task_id=self.request.id).get()
