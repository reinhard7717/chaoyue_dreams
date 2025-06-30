# 文件: tasks/stock_analysis_tasks.py

import asyncio
from datetime import datetime, timedelta
import logging
import time
from celery import group
import pandas as pd
from asgiref.sync import async_to_sync # 导入 Django/Celery 中调用异步代码的正确工具
from typing import Dict, Any
from chaoyue_dreams.celery import app as celery_app
from django.core.management.base import CommandError
from dao_manager.tushare_daos.index_basic_dao import IndexBasicDAO
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO
from dao_manager.tushare_daos.strategies_dao import StrategiesDAO
from services.indicator_services import IndicatorService
from strategies.multi_timeframe_trend_strategy import MultiTimeframeTrendStrategy
from strategies.trend_following_strategy import TrendFollowStrategy
from utils.cache_get import StrategyCacheGet

# 导入新策略和其对应的DAO
from strategies.monthly_trend_follow_strategy import MonthlyTrendFollowStrategy
from utils.config_loader import load_strategy_config

logger = logging.getLogger('tasks')

# _get_all_relevant_stock_codes_for_processing 函数保持不变...
async def _get_all_relevant_stock_codes_for_processing():
    """异步获取所有需要处理的股票代码列表，区分为自选股和非自-选股"""
    stock_basic_dao = StockBasicInfoDao()
    favorite_stock_codes = set()
    all_stock_codes = set()
    try:
        favorite_stocks = await stock_basic_dao.get_all_favorite_stocks()
        for fav in favorite_stocks:
            favorite_stock_codes.add(fav.stock_id)
        # logger.info(f"获取到 {len(favorite_stock_codes)} 个自选股代码")
    except Exception as e:
        logger.error(f"获取自选股列表时出错: {e}", exc_info=True)
    try:
        all_stocks = await stock_basic_dao.get_stock_list()
        for stock in all_stocks:
            if not stock.stock_code.endswith('.BJ'):
                all_stock_codes.add(stock.stock_code)
        # logger.info(f"获取到 {len(all_stock_codes)} 个全市场股票代码")
    except Exception as e:
        logger.error(f"获取全市场股票列表时出错: {e}", exc_info=True)
    non_favorite_stock_codes = list(all_stock_codes - favorite_stock_codes)
    favorite_stock_codes_list = list(favorite_stock_codes)
    favorite_stock_codes_list = sorted(favorite_stock_codes_list)
    non_favorite_stock_codes = sorted(non_favorite_stock_codes)
    total_unique_stocks = len(favorite_stock_codes_list) + len(non_favorite_stock_codes)
    # logger.info(f"总计需要处理的股票: {total_unique_stocks} (自选: {len(favorite_stock_codes_list)}, 非自选: {len(non_favorite_stock_codes)})")
    if not favorite_stock_codes_list and not non_favorite_stock_codes:
        logger.warning("未能获取到任何需要处理的股票代码")
    return favorite_stock_codes_list, non_favorite_stock_codes

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.run_trend_follow_strategy', queue='calculate_strategy')
def run_trend_follow_strategy(self, stock_code: str, trade_date: str):
    """
    【V3.0 - 架构对齐修复版】
    这是一个修复后的“遗留”策略任务，它直接调用 TrendFollowStrategy。
    - 核心修复: 不再传递文件路径，而是先加载配置字典，再将字典传递给服务和策略。
    - 这解决了 'got an unexpected keyword argument' 的系列错误。
    """
    logger.info(f"[{stock_code}] 开始执行 'trend_follow_strategy' (遗留任务修复版) on {stock_code} for date {trade_date}")
    try:
        # 1. 实例化服务
        indicator_service = IndicatorService()
        strategies_dao = StrategiesDAO()
        
        # 2. 【核心修复】加载配置文件为字典
        #    这是解决问题的关键一步，将文件IO操作提前。
        config_path = 'config/trend_follow_strategy.json'
        print(f"    [遗留任务调试] 正在加载策略配置文件: {config_path}")
        config_dict = load_strategy_config(config_path)
        if not config_dict:
            logger.error(f"[{stock_code}] 无法加载配置文件 {config_path}，任务终止。")
            return {"status": "error", "reason": "Config file not loaded"}
        print("    [遗留任务调试] 配置文件加载完成。")

        # 3. 【核心修复】使用正确的参数名 `config` 调用服务层
        print("    [遗留任务调试] 正在调用 IndicatorService.prepare_data...")
        all_dfs = async_to_sync(indicator_service.prepare_data)(
            stock_code=stock_code,
            config=config_dict,  # 使用加载好的配置字典
            trade_time=trade_date
        )
        print(f"    [遗留任务调试] IndicatorService 数据准备完成，获取的周期: {list(all_dfs.keys())}")

        if not all_dfs or 'D' not in all_dfs:
            logger.warning(f"[{stock_code}] 数据准备失败，未返回有效的日线数据。")
            return {"status": "success", "saved_count": 0, "reason": "No daily data"}

        # 4. 【核心修复】使用配置字典初始化策略类
        #    这解决了 TrendFollowStrategy 初始化时可能出现的参数不匹配问题。
        trend_follow_strategy = TrendFollowStrategy(config=config_dict)
        
        # 5. 调用策略分析，传递数据字典和配置字典
        final_df, atomic_signals = trend_follow_strategy.apply_strategy(all_dfs, config_dict)

        if final_df is None or final_df.empty:
            logger.info(f"[{stock_code}] 策略分析未生成有效结果DataFrame。")
            return {"status": "success", "saved_count": 0, "reason": "No signals from strategy"}

        # 6. 准备数据库记录，传递配置字典
        db_records = trend_follow_strategy.prepare_db_records(stock_code, final_df, atomic_signals, config_dict)
        
        if not db_records:
            logger.info(f"[{stock_code}] 策略运行完成，但未触发任何需要记录的信号。")
            return {"status": "success", "saved_count": 0, "reason": "No DB records to save"}

        # 7. 保存到数据库
        save_count = async_to_sync(strategies_dao.save_strategy_signals)(db_records)
        logger.info(f"[{stock_code}] 成功保存 {save_count} 条 'trend_follow_strategy' 信号。")

        return {"status": "success", "saved_count": save_count}

    except Exception as e:
        logger.error(f"执行 'trend_follow_strategy' on {stock_code} 时出错: {e}", exc_info=True)
        # 使用 self.retry 可以实现任务重试
        # raise self.retry(exc=e, countdown=60)
        return {"status": "error", "reason": str(e)}

# --- 调度任务 (修改以传递 is_favorite 标志) ---
@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.analyze_all_stocks', queue='celery')
def analyze_all_stocks(self):
    try:
        logger.info("开始调度所有股票的分析任务")
        favorite_codes, non_favorite_codes = asyncio.run(_get_all_relevant_stock_codes_for_processing())
        if not non_favorite_codes and not favorite_codes:
            logger.warning("未找到任何股票数据，任务终止")
            return {"status": "failed", "reason": "no stocks found"}
        stock_count = len(favorite_codes) + len(non_favorite_codes)
        logger.info(f"找到 {stock_count} 只股票待分析.")
        
        # ▼▼▼ 修改/新增 ▼▼▼
        # 核心修复：获取当前日期字符串，并将其传递给子任务
        trade_time_str = datetime.now().strftime('%Y-%m-%d')
        logger.info(f"所有任务将使用统一的分析截止日期: {trade_time_str}")
        # ▲▲▲ 修改/新增 ▲▲▲
        
        # --- 为任务传递 is_favorite=True ---
        for stock_code in favorite_codes:
            # ▼▼▼ 修改/新增 ▼▼▼
            # 核心修复：现在传递两个参数 stock_code 和 trade_time_str
            run_trend_follow_strategy.s(stock_code, trade_time_str).set(queue='favorite_calculate_strategy').apply_async()
            # ▲▲▲ 修改/新增 ▲▲▲
        
        # --- 为任务传递 is_favorite=False ---
        for stock_code in non_favorite_codes:
            # ▼▼▼ 修改/新增 ▼▼▼
            # 核心修复：现在传递两个参数 stock_code 和 trade_time_str
            run_trend_follow_strategy.s(stock_code, trade_time_str).set(queue='calculate_strategy').apply_async()
            # ▲▲▲ 修改/新增 ▲▲▲
        
        logger.info(f"已调度 {len(favorite_codes)} 只股票的favorite分析任务")
        logger.info(f"已调度 {len(non_favorite_codes)} 只股票的non_favorite分析任务")
        return {"status": "started",  "stock_count": stock_count}
    except Exception as e:
        logger.error(f"调度所有股票分析任务时出错: {e}", exc_info=True)
        return {"status": "failed", "reason": str(e)}
