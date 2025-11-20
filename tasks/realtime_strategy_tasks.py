# 文件: tasks/realtime_strategy_tasks.py

import asyncio
import logging
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
from asgiref.sync import async_to_sync, sync_to_async
from celery import group
from django.utils import timezone

# 导入项目相关模块
from chaoyue_dreams.celery import app as celery_app
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO
from dao_manager.tushare_daos.strategies_dao import StrategiesDAO
from stock_models.index import TradeCalendar
from stock_models.stock_analytics import StrategyDailyScore, TradingSignal
from stock_models.stock_basic import StockInfo # 用于获取股票基本信息
from stock_models.time_trade import StockDailyBasic # 用于获取前一日OHLC数据
from strategies.realtime_strategy import RealtimeStrategy
from utils.cache_manager import CacheManager
from utils.config_loader import load_strategy_config
from utils.task_helpers import with_cache_manager

logger = logging.getLogger('tasks')

# 在模块级别加载策略配置，避免重复加载
UNIFIED_CONFIG = load_strategy_config('config/trend_follow_strategy.json')
REALTIME_STRATEGY_CONFIG = UNIFIED_CONFIG['strategy_params']['trend_follow'].get('realtime_strategy_params', {})

# 辅助函数：获取所有相关股票代码（从 stock_analysis_tasks.py 复制并简化）
async def _get_all_relevant_stock_codes_for_processing(stock_basic_dao: StockBasicInfoDao) -> Tuple[List[str], List[str]]:
    """
    异步获取所有需要处理的股票代码列表。
    Args:
        stock_basic_dao (StockBasicInfoDao): 股票基本信息DAO实例。
    Returns:
        Tuple[List[str], List[str]]: (自选股代码列表, 非自选股代码列表)。
    """
    favorite_stock_codes = set()
    all_stock_codes = set()
    try:
        favorite_stocks = await stock_basic_dao.get_all_favorite_stocks()
        if favorite_stocks:
            for fav in favorite_stocks:
                if fav and fav.get("stock_code"):
                    favorite_stock_codes.add(fav.get("stock_code"))
    except Exception as e:
        logger.error(f"获取自选股列表时出错: {e}", exc_info=True)
    try:
        all_stocks = await stock_basic_dao.get_stock_list()
        if all_stocks:
            for stock in all_stocks:
                # 排除北交所股票，如果需要可以调整
                if stock and not stock.stock_code.endswith('.BJ'):
                    all_stock_codes.add(stock.stock_code)
    except Exception as e:
        logger.error(f"获取全市场股票列表时出错: {e}", exc_info=True)
    non_favorite_stock_codes = list(all_stock_codes - favorite_stock_codes)
    favorite_stock_codes_list = list(favorite_stock_codes)
    # 返回排序后的列表，保证每次结果一致
    return sorted(favorite_stock_codes_list), sorted(non_favorite_stock_codes)

@celery_app.task(bind=True, name='tasks.realtime_strategy_tasks.run_intraday_strategy_for_all_stocks', queue='realtime_strategy')
@with_cache_manager
def run_intraday_strategy_for_all_stocks(self, *, cache_manager: CacheManager):
    """
    【盘中策略总调度任务】
    在交易时段内，调度所有股票的盘中策略分析。
    此任务应由 Celery Beat 定时触发，例如每分钟或每5分钟。
    """
    logger.info("====== [盘中策略总调度任务] 启动 ======")
    current_dt = timezone.now() # 获取当前带时区的时间
    current_date = current_dt.date()
    current_time = current_dt.time()
    # 1. 检查是否在交易日和交易时段内
    is_trading_day = async_to_sync(TradeCalendar.is_trading_day_async)(current_date)
    if not is_trading_day:
        logger.info(f"今日 {current_date} 非交易日，跳过盘中策略调度。")
        return {"status": "skipped", "reason": "Not a trading day."}
    # 从配置中获取盘中策略的交易开始和结束时间
    trade_start_time = datetime.strptime(REALTIME_STRATEGY_CONFIG.get('trade_start_time', '09:45'), '%H:%M').time()
    trade_end_time = datetime.strptime(REALTIME_STRATEGY_CONFIG.get('trade_end_time', '14:50'), '%H:%M').time()
    if not (trade_start_time <= current_time <= trade_end_time):
        logger.info(f"当前时间 {current_time} 不在盘中策略执行时段 ({trade_start_time}-{trade_end_time}) 内，跳过调度。")
        return {"status": "skipped", "reason": "Not within trading hours."}
    logger.info(f"当前交易日: {current_date}, 当前时间: {current_time}。")
    # 2. 获取所有股票代码
    stock_basic_dao = StockBasicInfoDao(cache_manager)
    favorite_codes, non_favorite_codes = async_to_sync(_get_all_relevant_stock_codes_for_processing)(stock_basic_dao)
    all_codes = favorite_codes + non_favorite_codes
    if not all_codes:
        logger.warning("[盘中策略] 未找到任何股票数据，任务终止。")
        return {"status": "failed", "reason": "No stocks found."}
        
    total_stocks = len(all_codes)
    logger.info(f"[盘中策略] 准备为 {total_stocks} 只股票执行盘中分析。")
    # 3. 派发并行子任务
    analysis_tasks = [
        run_intraday_strategy_for_single_stock.s(
            stock_code=code,
            trade_date_str=current_date.strftime('%Y-%m-%d') # 传递当前交易日字符串
        ).set(queue='realtime_strategy') for code in all_codes
    ]
    workflow = group(analysis_tasks) # 使用 Celery group 实现并行
    workflow.apply_async() # 异步执行任务组
    logger.info(f"[盘中策略] 已成功为 {total_stocks} 只股票启动盘中分析任务。")
    return {"status": "workflow_started", "stock_count": total_stocks, "trade_date": current_date.strftime('%Y-%m-%d')}


@celery_app.task(bind=True, name='tasks.realtime_strategy_tasks.run_intraday_strategy_for_single_stock', queue='realtime_strategy', acks_late=True)
@with_cache_manager
def run_intraday_strategy_for_single_stock(self, stock_code: str, trade_date_str: str, *, cache_manager: CacheManager):
    """
    【盘中策略单股票执行任务】
    对单个股票获取当日所有1分钟K线，运行盘中策略，并保存信号。
    Args:
        stock_code (str): 股票代码。
        trade_date_str (str): 当前交易日字符串 'YYYY-MM-DD'。
        cache_manager (CacheManager): 缓存管理器实例。
    """
    async def main():
        logger.debug(f"  -> [盘中策略] 开始分析 {stock_code} for {trade_date_str}")
        trade_date = datetime.strptime(trade_date_str, '%Y-%m-%d').date()
        # 1. 获取前一日收盘价等数据 (用于 RealtimeStrategy 计算枢轴点)
        prev_day_data = await _get_prev_day_data(stock_code, trade_date)
        # 2. 获取当日所有1分钟K线数据
        time_trade_dao = StockTimeTradeDAO(cache_manager)
        # 获取从开盘到当前时间的所有1分钟K线数据
        all_1min_klines_df = await time_trade_dao.get_1_min_kline_time_by_day(stock_code, trade_date)
        # 检查数据量是否足够进行5分钟K线聚合和指标计算
        min_required_1min_data = REALTIME_STRATEGY_CONFIG.get('min_data_points_5min', 21) * 5 # 至少需要21根5分钟K线，即105根1分钟K线
        if all_1min_klines_df.empty or len(all_1min_klines_df) < min_required_1min_data:
            logger.debug(f"    - [盘中策略] {stock_code} 1分钟K线数据不足 ({len(all_1min_klines_df)}条，至少需要{min_required_1min_data}条)，跳过分析。")
            return {"status": "skipped", "stock_code": stock_code, "reason": "Insufficient 1min data."}
        # 3. 实例化 RealtimeStrategy
        # RealtimeStrategy 内部的 IntradayDataAggregator 会在初始化时创建空的data_buffer。
        # 每次任务运行时，我们都将当日所有1分钟K线数据喂给它，确保其内部状态完整。
        realtime_strategy = RealtimeStrategy(params={}, config=UNIFIED_CONFIG, prev_day_data=prev_day_data)
        # 4. 喂入所有1分钟K线数据
        # RealtimeStrategy.update_data 接收 pd.Series，所以需要逐行迭代
        for _, kline_series in all_1min_klines_df.iterrows():
            realtime_strategy.update_data(kline_series)
        # 5. 获取当日日线策略分数 (作为盘中策略的参考)
        daily_signal_info = await _get_daily_strategy_score(stock_code, trade_date)
        # 6. 运行盘中策略
        signal_result = realtime_strategy.run_strategy(stock_code, daily_signal_info)
        # 7. 保存信号 (如果存在)
        if signal_result:
            strategies_dao = StrategiesDAO(cache_manager)
            # 确保 signal_result['entry_time'] 是 aware datetime
            if signal_result['entry_time'].tzinfo is None:
                signal_result['entry_time'] = timezone.make_aware(signal_result['entry_time'])
            # 创建 TradingSignal 实例
            trading_signal = TradingSignal(
                stock_id=signal_result['stock_code'],
                trade_time=signal_result['entry_time'],
                timeframe='5min', # 盘中策略通常基于5分钟K线
                strategy_name=signal_result['playbook'],
                signal_type=signal_result['signal_type'], # 例如 "BUY" 或 "SELL"
                entry_score=signal_result.get('intraday_score', 0.0) if signal_result['signal_type'] == "BUY" else 0.0,
                risk_score=signal_result.get('intraday_score', 0.0) if signal_result['signal_type'] == "SELL" else 0.0,
                close_price=signal_result['entry_price'],
                signal_reason=signal_result['reason'],
                # 可以考虑将 intraday_rating 存入一个额外的 JSON 字段，如果模型支持的话
            )
            # 使用 StrategiesDAO 保存信号。
            # save_strategy_signals 方法期望一个包含五类记录的元组，
            # 对于盘中信号，我们只有 TradingSignal，其他列表为空。
            await strategies_dao.save_strategy_signals(([trading_signal], [], [], [], []))
            logger.info(f"    - [盘中策略] {stock_code} 成功保存盘中信号: {signal_result['signal_type']} (分数: {signal_result.get('intraday_score', 0)})")
            return {"status": "signal_saved", "stock_code": stock_code, "signal_type": signal_result['signal_type'], "score": signal_result.get('intraday_score', 0)}
        else:
            logger.debug(f"    - [盘中策略] {stock_code} 未触发盘中信号。")
            return {"status": "no_signal", "stock_code": stock_code}
    try:
        return async_to_sync(main)()
    except Exception as e:
        logger.error(f"  !! [盘中策略] 分析 {stock_code} 时发生严重错误: {e}", exc_info=True)
        # 任务失败时重试，最多3次，每次间隔1分钟
        raise self.retry(exc=e, countdown=60, max_retries=3)

@sync_to_async
def _get_prev_day_data(stock_code: str, current_date: date) -> Dict:
    """
    异步获取前一个交易日的OHLC数据，用于 RealtimeStrategy 计算枢轴点。
    Args:
        stock_code (str): 股票代码。
        current_date (date): 当前交易日。
    Returns:
        Dict: 包含前一日 'open', 'high', 'low', 'close' 的字典，如果获取失败则返回空字典。
    """
    prev_trade_date = TradeCalendar.get_trade_date_offset(current_date, -1)
    if not prev_trade_date:
        logger.warning(f"无法获取 {stock_code} 前一个交易日数据，跳过。")
        return {}
    try:
        # 使用 StockDailyBasic 模型获取前一日的OHLC数据
        prev_day_kline = StockDailyBasic.objects.filter(
            stock__stock_code=stock_code,
            trade_date=prev_trade_date
        ).values('open', 'high', 'low', 'close').first()
        if prev_day_kline:
            return {
                'open': prev_day_kline['open'],
                'high': prev_day_kline['high'],
                'low': prev_day_kline['low'],
                'close': prev_day_kline['close']
            }
    except Exception as e:
        logger.warning(f"获取 {stock_code} 前一日 ({prev_trade_date}) 数据失败: {e}", exc_info=True)
    return {}

@sync_to_async
def _get_daily_strategy_score(stock_code: str, trade_date: date) -> Dict:
    """
    异步获取当日的日线策略分数，作为盘中策略的参考输入。
    Args:
        stock_code (str): 股票代码。
        trade_date (date): 当前交易日。
    Returns:
        Dict: 包含 'entry_score' 和 'risk_score' 的字典，如果获取失败则返回默认值。
    """
    try:
        daily_score = StrategyDailyScore.objects.filter(
            stock__stock_code=stock_code,
            trade_date=trade_date
        ).values('entry_score', 'risk_score').first()
        if daily_score:
            return {
                'entry_score': daily_score.get('entry_score', 0),
                'risk_score': daily_score.get('risk_score', 0)
            }
    except Exception as e:
        logger.warning(f"获取 {stock_code} 日线策略分数 ({trade_date}) 失败: {e}", exc_info=True)
    return {'entry_score': 0, 'risk_score': 0}

