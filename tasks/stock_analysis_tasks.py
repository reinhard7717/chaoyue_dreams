# tasks/stock_analysis_tasks.py
import asyncio
from datetime import datetime
import logging
import pandas as pd
from typing import Dict, Any, Optional
from celery import group
from django.conf import settings
from asgiref.sync import sync_to_async
from chaoyue_dreams.celery import app as celery_app
from django.core.management.base import CommandError
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from services.indicator_services import IndicatorService
from stock_models.stock_analytics import StockAnalysisResultTrendFollowing
from strategies.t_plus_0_strategy import TPlus0Strategy
from strategies.trend_following_strategy import TrendFollowingStrategy
from strategies.trend_reversal_strategy import TrendReversalStrategy
from utils.cache_get import StrategyCacheGet

logger = logging.getLogger(__name__)

# --- 辅助函数：获取需要处理的股票代码 (保持不变) ---
async def _get_all_relevant_stock_codes_for_processing():
    """异步获取所有需要处理的股票代码列表，区分为自选股和非自选股"""
    stock_basic_dao = StockBasicInfoDao()
    favorite_stock_codes = set()
    all_stock_codes = set()
    # 获取自选股
    try:
        favorite_stocks = await stock_basic_dao.get_all_favorite_stocks()
        for fav in favorite_stocks:
            favorite_stock_codes.add(fav.stock_id)
        logger.info(f"获取到 {len(favorite_stock_codes)} 个自选股代码")
    except Exception as e:
        logger.error(f"获取自选股列表时出错: {e}", exc_info=True)
    # 获取所有A股
    try:
        all_stocks = await stock_basic_dao.get_stock_list()
        for stock in all_stocks:
            if not stock.stock_code.endswith('.BJ'):  # 过滤掉以.BJ结尾的股票代码
                all_stock_codes.add(stock.stock_code)
        logger.info(f"获取到 {len(all_stock_codes)} 个全市场股票代码")
    except Exception as e:
        logger.error(f"获取全市场股票列表时出错: {e}", exc_info=True)

    # 计算非自选股代码
    non_favorite_stock_codes = list(all_stock_codes - favorite_stock_codes)
    favorite_stock_codes_list = list(favorite_stock_codes)

    favorite_stock_codes_list = sorted(favorite_stock_codes_list)  # 按stock_code排序
    non_favorite_stock_codes = sorted(non_favorite_stock_codes)    # 按stock_code排序

    total_unique_stocks = len(favorite_stock_codes_list) + len(non_favorite_stock_codes)
    logger.info(f"总计需要处理的股票: {total_unique_stocks} (自选: {len(favorite_stock_codes_list)}, 非自选: {len(non_favorite_stock_codes)})")

    if not favorite_stock_codes_list and not non_favorite_stock_codes:
        logger.warning("未能获取到任何需要处理的股票代码")

    return favorite_stock_codes_list, non_favorite_stock_codes

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.analyze_single_stock')
def analyze_single_stock(self, stock_code: str, params_file: str, day_count: int = 5):
    """
    对单只股票执行所有策略分析并保存结果
    """
    # 1. 实例化 IndicatorService
    indicator_service = IndicatorService()
    cache_get = StrategyCacheGet()
    stock_basic_dao = StockBasicInfoDao()
    stock_obj = asyncio.run(stock_basic_dao.get_stock_by_code(stock_code))
    # 2. 获取时间字符串列表
    trade_times_list = asyncio.run(indicator_service.get_5_min_kline_time_by_day_count(stock_code=stock_code, day_count=day_count))
    # 3. 转换成时间戳，升序排序
    trade_times_list = sorted(trade_times_list)
    # 5. 遍历检测
    first_missing_index = None
    for idx, t_str in enumerate(trade_times_list):
        # exists = asyncio.run(cache_get.analyze_signals_trend_following_datas_by_timestamp(stock_code, ts))
        exists = StockAnalysisResultTrendFollowing.objects.filter(stock=stock_obj, timestamp=t_str).exists()
        if not exists:
            first_missing_index = idx
            break
    # 6. 从第一个不存在的时间开始，执行策略分析
    start_index = first_missing_index if first_missing_index is not None else 0

    for idx in range(start_index, len(trade_times_list)):
        trade_time_str = trade_times_list[idx]
        logger.info(f"开始分析股票 {stock_code} - {trade_time_str}")
        execute_strategy_for_trade_time(stock_code=stock_code, params_file= params_file, trade_time_str=trade_time_str)

def execute_strategy_for_trade_time(stock_code: str, params_file: str, trade_time_str: str):
    indicator_service = IndicatorService()
    # 2. 准备数据 (基于 params_file 准备所有策略所需数据)
    try:
        result = asyncio.run(indicator_service.prepare_strategy_dataframe(stock_code=stock_code, params_file=params_file, base_needed_bars=1000, trade_time=trade_time_str))
        if result is None or not isinstance(result, tuple) or len(result) != 2:
            logger.warning(f"股票 {stock_code} 数据准备失败，跳过分析")
            return {"stock_code": stock_code, "status": "skipped", "reason": "no data"}
        data_df, indicator_configs = result
        if data_df is None or data_df.empty:
            logger.warning(f"股票 {stock_code} 数据为空，跳过分析")
            return {"stock_code": stock_code, "status": "skipped", "reason": "no data"}

        # 获取最新时间戳（假设从数据中取最新时间）
        timestamp = data_df.index[-1] if not data_df.empty else pd.Timestamp.now()
        print(f"execute_strategy_for_trade_time.timestamp: {timestamp}")
        # 3. 实例化需要运行的策略
        strategies_to_run: Dict[str, Any] = {}
        try:
            # 按需实例化策略
            strategies_to_run['trend_following'] = TrendFollowingStrategy(params_file=params_file)
            # strategies_to_run['trend_reversal'] = TrendReversalStrategy(params_file=params_file)
            # strategies_to_run['t_plus_0'] = TPlus0Strategy(params_file=params_file)
            logger.info(f"将要运行的策略: {', '.join(s.strategy_name for s in strategies_to_run.values())}")
        except (FileNotFoundError, ValueError, ImportError, KeyError) as e:
            logger.error(f"初始化策略时出错: {e}", exc_info=True)
            raise CommandError(f"初始化策略时出错: {e}")

        results = {}
        for strategy_name, strategy in strategies_to_run.items():
            try:
                # 执行策略生成信号
                signals = strategy.generate_signals(data=data_df, stock_code=stock_code, indicator_configs=indicator_configs)
                if signals is not None and not signals.empty:
                    # 先分析信号
                    analysis_result = strategy.analyze_signals(stock_code)
                    if analysis_result is not None:
                        # 保存分析结果
                        strategy.save_analysis_results(stock_code, timestamp, data_df)
                        results[strategy_name] = {"status": "success", "signal": signals.iloc[-1] if not signals.empty else None}
                    else:
                        results[strategy_name] = {"status": "failed", "reason": "no analysis result"}
                        logger.warning(f"策略 {strategy_name} 未能为 {stock_code} 生成分析结果")
                else:
                    results[strategy_name] = {"status": "failed", "reason": "no signal generated"}
                    logger.warning(f"策略 {strategy_name} 未能为 {stock_code} 生成信号")
            except Exception as e:
                logger.error(f"策略 {strategy_name} 分析 {stock_code} 时出错: {e}", exc_info=True)
                results[strategy_name] = {"status": "error", "reason": str(e)}

        logger.info(f"完成股票 {stock_code} 的分析")
        return {"stock_code": stock_code, "status": "completed", "results": results}
    except Exception as e:
        logger.error(f"分析股票 {stock_code} 时发生错误: {e}", exc_info=True)
        # raise

# --- 调度任务：获取所有股票并分配分析任务 ---
@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.analyze_all_stocks')
def analyze_all_stocks(self, params_file: str = "config/indicator_parameters.json"):
    """
    调度任务：获取所有股票并分配分析任务
    """
    try:
        logger.info("开始调度所有股票的分析任务")
        # 在同步任务中运行异步代码获取列表
        favorite_codes, non_favorite_codes = asyncio.run(_get_all_relevant_stock_codes_for_processing())
        if not non_favorite_codes and not favorite_codes:
            logger.warning("未找到任何股票数据，任务终止")
            return {"status": "failed", "reason": "no stocks found"}

        stock_count = len(favorite_codes) + len(non_favorite_codes)
        logger.info(f"找到 {stock_count} 只股票待分析")

        for stock_code in favorite_codes:
            analyze_single_stock.s(stock_code, params_file).set(queue='favorite_calculate_strategy').apply_async()
        for stock_code in non_favorite_codes:
            analyze_single_stock.s(stock_code, params_file).set(queue='calculate_strategy').apply_async()

        # 记录任务ID（如果有多个任务组，取最后一个或合并记录）
        logger.info(f"已调度 {favorite_codes.count} 只股票的favorite分析任务")
        logger.info(f"已调度 {non_favorite_codes.count} 只股票的non_favorite分析任务")
        return {"status": "started",  "stock_count": stock_count}
    except Exception as e:
        logger.error(f"调度所有股票分析任务时出错: {e}", exc_info=True)
        return {"status": "failed", "reason": str(e)}
