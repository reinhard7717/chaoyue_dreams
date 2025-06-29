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

# --- 为任务增加 is_favorite 参数 ---
@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.analyze_single_stock', queue='calculate_strategy')
def analyze_single_stock(self, stock_code: str, params_file: str, day_count: int = 0, is_favorite: bool = False):
    """
    分析单个股票的任务，增加了 is_favorite 标志。
    Args:
        stock_code (str): 股票代码。
        params_file (str): 参数文件路径。
        day_count (int): 回溯的天数。
        is_favorite (bool): 是否为自选股。
    """
    
    start_time = time.time()
    stt_dao = StockTimeTradeDAO()
    if day_count >= 0:
        index_basic_dao = IndexBasicDAO()
        open_dates = asyncio.run(index_basic_dao.get_last_n_trade_cal_open(n=day_count))
        print(f"获取到最近{day_count}个开盘日: {open_dates}")
        strategies_dao = StrategiesDAO()
        for open_date in open_dates:
            latest_klines = asyncio.run(stt_dao.get_5_min_kline_time_by_day(stock_code=stock_code, date=open_date))
            print(f"{open_date} 获取到{len(latest_klines)}个5分钟K线时间点")
            for kline_time_str in latest_klines:
                try:
                    kline_time = datetime.strptime(kline_time_str, '%Y-%m-%d %H:%M:%S')
                    strategy_result = asyncio.run(strategies_dao.get_strategy_result_by_timestamp(stock_code, kline_time))
                    if not strategy_result:
                        beijing_time = kline_time + timedelta(hours=8)
                        beijing_time_str = beijing_time.strftime('%Y-%m-%d %H:%M')
                        # logger.info(f"开始分析股票 {stock_code} - {beijing_time_str} (自选: {is_favorite})") # 修改：日志中增加自选股标识
                        # --- 传递 is_favorite 参数 ---
                        execute_strategy_for_trade_time(stock_code, params_file, kline_time, is_favorite=is_favorite)
                        
                    else:
                        beijing_time = kline_time + timedelta(hours=8)
                        beijing_time_str = beijing_time.strftime('%Y-%m-%d %H:%M')
                        print(f"{stock_code} {beijing_time_str} 已有分析结果，跳过")
                except Exception as e:
                    print(f"处理K线时间{kline_time_str}时出错: {e}")
    else:
        latest_kline = asyncio.run(stt_dao.get_latest_5_min_kline(stock_code=stock_code))
        kline_time = None
        if latest_kline and latest_kline.trade_time:
            if isinstance(latest_kline.trade_time, str):
                try:
                    kline_time = datetime.fromisoformat(latest_kline.trade_time.replace('Z', '+00:00'))
                except ValueError:
                    kline_time_str = latest_kline.trade_time.split('+')[0].split('.')[0].replace('T', ' ')
                    kline_time = datetime.strptime(kline_time_str, '%Y-%m-%d %H:%M:%S')
            else:
                kline_time = latest_kline.trade_time
        if kline_time:
            # logger.info(f"开始分析股票 {stock_code} - {time_plus_1min} (自选: {is_favorite})") # 修改：日志中增加自选股标识
            # --- 传递 is_favorite 参数 ---
            execute_strategy_for_trade_time(stock_code, params_file, kline_time, is_favorite=is_favorite)
            
        else:
            print(f"无法解析 {stock_code} 的最新K线时间，跳过分析")
    total_time = time.time() - start_time
    print(f"任务analyze_single_stock {stock_code} 执行总耗时: {total_time:.2f} 秒")

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.run_multi_timeframe_strategy_task', queue='celery')
def run_multi_timeframe_strategy_task(self, stock_code: str, trade_time_str: str) -> Dict[str, Any]: # 确保 self 在这里
    """
    【重构后的Celery任务】
    调用新的、集成的 MultiTimeframeTrendStrategy 来执行完整的多时间框架分析。
    此任务封装了所有逻辑，包括数据准备、多层策略应用和结果存储。
    """
    logger.info(f"[{stock_code}] 开始执行 '多时间框架策略' Celery 任务，交易日: {trade_time_str}...")
    try:
        # --- 步骤 1: 实例化策略引擎和DAO ---
        multi_timeframe_strategy = MultiTimeframeTrendStrategy()
        strategies_dao = StrategiesDAO()

        # --- 步骤 2: 执行完整的策略分析 ---
        logger.info(f"[{stock_code}] 正在调用多时间框架策略引擎...")
        db_records = async_to_sync(multi_timeframe_strategy.run_for_stock)(
            stock_code=stock_code,
            trade_time=trade_time_str
        )

        # --- 步骤 3: 保存结果到数据库 ---
        if not db_records:
            logger.info(f"[{stock_code}] 多时间框架策略执行完毕，无任何新的买卖信号需要记录。")
            return {"status": "success", "report": None, "saved_count": 0}

        logger.info(f"[{stock_code}] 发现 {len(db_records)} 条信号，正在保存到数据库...")
        # 假设您的保存方法是 save_strategy_signals
        save_count = async_to_sync(strategies_dao.save_strategy_signals)(db_records)
        
        if save_count > 0:
            logger.info(f"[{stock_code}] 成功保存 {save_count} 条多时间框架策略信号。")

            try:
                logger.info(f"[{stock_code}] 正在更新策略状态摘要...")
                # 假设 db_records 不为空，我们可以从中获取必要信息
                # 注意：这里假设一个批次的任务只处理一个策略和一个时间框架
                strategy_name = db_records[0]['strategy_name']
                timeframe = db_records[0]['timeframe']
                
                # 调用DAO中的更新方法 (推荐将更新逻辑也封装在DAO中)
                async_to_sync(strategies_dao.update_strategy_state)(
                    stock_code=stock_code,
                    strategy_name=strategy_name,
                    timeframe=timeframe
                )
                logger.info(f"[{stock_code}] 成功更新策略状态摘要。")
            except Exception as state_e:
                logger.error(f"[{stock_code}] 更新策略状态摘要时发生错误: {state_e}", exc_info=True)
        else:
            logger.warning(f"[{stock_code}] 尝试保存 {len(db_records)} 条信号，但数据库操作未返回成功计数。")

        return {"status": "success", "report": "部分记录已省略", "saved_count": save_count}

    except Exception as e:
        logger.error(f"执行 'run_multi_timeframe_strategy_task' on {stock_code} 时出错: {e}", exc_info=True)
        return {"status": "error", "reason": str(e)}

# --- 调度任务 (修改以传递 is_favorite 标志) ---
@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.analyze_all_stocks', queue='celery')
def analyze_all_stocks(self, day_count: int = -1, params_file: str = "config/monthly_trend_follow_strategy.json"):
    try:
        logger.info("开始调度所有股票的分析任务")
        favorite_codes, non_favorite_codes = asyncio.run(_get_all_relevant_stock_codes_for_processing())
        if not non_favorite_codes and not favorite_codes:
            logger.warning("未找到任何股票数据，任务终止")
            return {"status": "failed", "reason": "no stocks found"}
        stock_count = len(favorite_codes) + len(non_favorite_codes)
        logger.info(f"找到 {stock_count} 只股票待分析，使用统一参数文件: {params_file}")
        
        # ▼▼▼ 修改/新增 ▼▼▼
        # 核心修复：获取当前日期字符串，并将其传递给子任务
        trade_time_str = datetime.now().strftime('%Y-%m-%d')
        logger.info(f"所有任务将使用统一的分析截止日期: {trade_time_str}")
        # ▲▲▲ 修改/新增 ▲▲▲
        
        # --- 为任务传递 is_favorite=True ---
        for stock_code in favorite_codes:
            # ▼▼▼ 修改/新增 ▼▼▼
            # 核心修复：现在传递两个参数 stock_code 和 trade_time_str
            run_multi_timeframe_strategy_task.s(stock_code, trade_time_str).set(queue='favorite_calculate_strategy').apply_async()
            # ▲▲▲ 修改/新增 ▲▲▲
        
        # --- 为任务传递 is_favorite=False ---
        for stock_code in non_favorite_codes:
            # ▼▼▼ 修改/新增 ▼▼▼
            # 核心修复：现在传递两个参数 stock_code 和 trade_time_str
            run_multi_timeframe_strategy_task.s(stock_code, trade_time_str).set(queue='calculate_strategy').apply_async()
            # ▲▲▲ 修改/新增 ▲▲▲
        
        logger.info(f"已调度 {len(favorite_codes)} 只股票的favorite分析任务")
        logger.info(f"已调度 {len(non_favorite_codes)} 只股票的non_favorite分析任务")
        return {"status": "started",  "stock_count": stock_count}
    except Exception as e:
        logger.error(f"调度所有股票分析任务时出错: {e}", exc_info=True)
        return {"status": "failed", "reason": str(e)}

@celery_app.task(bind=True, name='tasks.stock_analysis_tasks.analyze_batch_stocks', queue='celery')
def analyze_batch_stocks(self, stock_codes: list, params_file: str = "config/monthly_trend_follow_strategy.json", day_count: int = -1):
    # 注意：此处的 params_file 必须包含所有要运行策略的指标配置
    print(f"批量分析任务启动，传入股票数: {len(stock_codes)}，使用统一参数文件: {params_file}")
    # 获取关注和非关注股票代码集合
    favorite_stock_codes, non_favorite_stock_codes = asyncio.run(_get_all_relevant_stock_codes_for_processing())
    favorite_stock_codes_set = set(favorite_stock_codes)
    non_favorite_stock_codes_set = set(non_favorite_stock_codes)

    # 优化：只遍历一次stock_codes，归类到两个列表
    favorite_list = []
    non_favorite_list = []
    for code in stock_codes:
        if code in favorite_stock_codes_set:
            favorite_list.append(code)
        elif code in non_favorite_stock_codes_set:
            non_favorite_list.append(code)
        # print(f"自选股{len(favorite_list)}个，非自选股{len(non_favorite_list)}个")
    
    # --- 为任务传递 is_favorite=True ---
    if favorite_list:
        favorite_group = group(analyze_single_stock.s(code, params_file, day_count, is_favorite=True).set(queue='favorite_calculate_strategy') for code in favorite_list)
        favorite_result = favorite_group.apply_async()
        print(f"已分发{len(favorite_list)}个自选股分析子任务")
        
    # --- 为任务传递 is_favorite=False ---
    if non_favorite_list:
        non_favorite_group = group(analyze_single_stock.s(code, params_file, day_count, is_favorite=False).set(queue='calculate_strategy') for code in non_favorite_list)
        non_favorite_result = non_favorite_group.apply_async()
        print(f"已分发{len(non_favorite_list)}个非自选股分析子任务")    
    
    return {"status": "dispatched", "favorite_count": len(favorite_list), "non_favorite_count": len(non_favorite_list)}




# 新增辅助函数 1: 负责执行【新版】月线趋势跟踪策略 (自包含数据准备)
def _run_monthly_strategy(stock_code: str, trade_time_str: str) -> Dict:
    """
    辅助函数：完整执行月线趋势跟踪策略，包括其专属的数据准备。
    它使用新的 `prepare_daily_centric_dataframe` 方法。
    """
    logger.info(f"[{stock_code}] 开始执行 '月线趋势跟踪策略'...")
    try:
        # 步骤 1: 使用新方法准备数据
        indicator_service = IndicatorService()
        params_file = "config/monthly_trend_follow_strategy.json"
        data_df, _ = asyncio.run(indicator_service.prepare_daily_centric_dataframe(
            stock_code=stock_code,
            params_file=params_file,
            trade_time=trade_time_str
        ))

        if data_df is None or data_df.empty:
            logger.warning(f"[{stock_code}] 月线策略数据准备失败，跳过执行。")
            return {"status": "skipped", "reason": "data preparation failed"}

        # 步骤 2: 执行策略分析
        monthly_strategy = MonthlyTrendFollowStrategy()
        _, analysis_result = monthly_strategy.run_analysis(
            stock_code=stock_code,
            params_file=params_file,
            trade_time=trade_time_str,
            data_df=data_df
        )
        
        # 步骤 3: 处理并保存结果
        if analysis_result:
            strategies_dao = StrategiesDAO()
            save_count = asyncio.run(strategies_dao.save_monthly_trend_strategy_reports(reports_data=[analysis_result]))
            if save_count > 0:
                logger.info(f"[{stock_code}] 月线策略发现信号并成功保存。")
            else:
                logger.warning(f"[{stock_code}] 月线策略发现信号但保存失败。")
            return {"status": "success", "report": analysis_result}
        else:
            logger.info(f"[{stock_code}] 月线策略执行完毕，无任何信号。")
            return {"status": "success", "report": None}
            
    except Exception as e:
        logger.error(f"执行 '月线趋势跟踪策略' on {stock_code} 时出错: {e}", exc_info=True)
        return {"status": "error", "reason": str(e)}

def _run_trend_follow_strategy(stock_code: str, trade_time_str: str) -> Dict[str, Any]:
    """
    辅助函数：完整执行趋势跟踪策略，并保存结果到数据库。
    此版本遵循“数据准备 -> 策略分析 -> 结果打包 -> 数据库存储”的清晰流程。
    并使用 asgiref.sync.async_to_sync 安全地在同步环境中调用异步代码。
    """
    logger.info(f"[{stock_code}] 开始执行 '趋势跟踪策略'...")
    try:
        # --- 步骤 1: 实例化所有需要的服务和DAO ---
        # 实例化策略类，其 __init__ 方法会自动加载所需配置
        trend_follow_strategy = TrendFollowStrategy(min_config_path='config/trend_follow_strategy.json')
        # 从策略实例中获取已经初始化好的 indicator_service
        indicator_service = trend_follow_strategy.indicator_service
        # 实例化DAO，用于后续数据存储
        strategies_dao = StrategiesDAO()

        # --- 步骤 2: 准备策略所需的数据 ---
        # 使用 async_to_sync 包装器来调用异步的数据准备方法
        # 假设 indicator_service 中有 prepare_daily_centric_dataframe 方法
        logger.info(f"[{stock_code}] 正在为【日线】趋势跟踪策略准备数据...")
        data_df, indicator_configs = async_to_sync(indicator_service.prepare_daily_centric_dataframe)(
            stock_code=stock_code,
            trade_time=trade_time_str,
            params_file=trend_follow_strategy.daily_config_path # 使用正确的日线配置路径
        )

        if data_df is None or data_df.empty:
            logger.warning(f"[{stock_code}] 数据准备失败或返回空DataFrame，无法执行策略。")
            return {"status": "error", "reason": "Data preparation failed."}

        # --- 步骤 3: 执行策略分析 ---
        # apply_strategy 是一个异步方法，同样需要用 async_to_sync 包装
        # 它现在返回包含所有历史信号的DataFrame和原子信号字典
        logger.info(f"[{stock_code}] 数据准备完毕，开始应用【日线】策略逻辑...")
        result_df, atomic_signals = async_to_sync(trend_follow_strategy.apply_strategy)(
            df=data_df,
            params=trend_follow_strategy.daily_params # 使用正确的日线参数
        )

        # --- 步骤 4: 将分析结果打包成数据库记录 ---
        # prepare_db_records 是一个同步方法，直接调用即可
        # 它会筛选出有信号的日期，并转换为字典列表
        logger.info(f"[{stock_code}] 策略分析完成，正在准备数据库记录...")
        db_records = trend_follow_strategy.prepare_db_records(
            stock_code, 
            result_df, 
            atomic_signals, 
            params=trend_follow_strategy.daily_params # 将日线参数传递下去
        )

        # --- 步骤 5: 保存结果到数据库 ---
        if not db_records:
            logger.info(f"[{stock_code}] 趋势跟踪策略执行完毕，无任何新的买卖信号需要记录。")
            return {"status": "success", "report": None, "saved_count": 0}

        logger.info(f"[{stock_code}] 发现 {len(db_records)} 条信号，正在保存到数据库...")
        # save_trend_follow_strategy_reports 是异步的，需要包装
        save_count = async_to_sync(strategies_dao.save_trend_follow_strategy_reports)(reports_data=db_records)
        
        if save_count > 0:
            logger.info(f"[{stock_code}] 成功保存 {save_count} 条趋势跟踪策略信号。")
        else:
            logger.warning(f"[{stock_code}] 尝试保存 {len(db_records)} 条信号，但数据库操作未返回成功计数。")

        return {"status": "success", "report": db_records, "saved_count": save_count}

    except Exception as e:
        logger.error(f"执行 'trend_follow_strategy' on {stock_code} 时出错: {e}", exc_info=True)
        return {"status": "error", "reason": str(e)}

# 【最终重构版】主协调函数
def execute_strategy_for_trade_time(stock_code: str, params_file: str, trade_time: datetime, is_favorite: bool = False):
    """
    【最终重构版】为给定股票和时间点，协调执行所有已配置的策略。
    此函数作为协调器，调用独立的、自包含的辅助函数来执行每个策略。
    """
    logger.info(f"--- 开始为股票 {stock_code} 在 {trade_time} 执行所有策略分析 ---")
    timestamp = pd.Timestamp(trade_time)
    trade_time_str = trade_time.strftime('%Y-%m-%d %H:%M:%S')
    
    results = {}
    
    # 任务 1: 执行月线趋势跟踪策略
    results['monthly_trend_follow'] = _run_monthly_strategy(stock_code, trade_time_str)
    # 任务 2: 执行日线趋势跟踪策略
    results['trend_follow'] = _run_trend_follow_strategy(stock_code, trade_time_str)
    
    # 任务 2: 执行传统趋势跟踪策略
    # results['trend_following'] = _run_legacy_strategy(stock_code, params_file, trade_time_str, timestamp, is_favorite)
    
    # 如果未来有策略3，在这里加一行即可:
    # results['new_strategy'] = _run_new_strategy(...)
    
    logger.info(f"--- 完成股票 {stock_code} 在 {trade_time} 的所有策略分析 ---")
    return {"stock_code": stock_code, "status": "completed", "results": results}
