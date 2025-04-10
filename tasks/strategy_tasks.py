"""
策略相关任务
提供策略计算的定时任务和手动触发任务
"""
# import asyncio
import asyncio
import logging
from celery import shared_task
from chaoyue_dreams.celery import app as celery_app
import pandas as pd # 导入 pandas 用于处理时间戳
  # 从 celery.py 导入 app 实例并重命名为 celery_app
# from service.strategy_service import StrategyService
# from service.calculation_service import CalculationService
# from users.models import FavoriteStock

logger = logging.getLogger('strategy')

@celery_app.task(bind=True, name='tasks.strategy.run_macd_rsi_kdj_boll_strategy_for_stock')
async def run_macd_rsi_kdj_boll_strategy_for_stock(self):
    """
    定时任务：计算策略信号
    每1分钟执行一次
    包括各种交易策略的信号计算
    """
    logger.info("开始执行策略信号计算任务")
    from dao_manager.daos.stock_basic_dao import StockBasicDAO
    stock_basic_dao = StockBasicDAO()
    favorite_stocks = await stock_basic_dao.get_all_favorite_stocks()
    for favorite_stock in favorite_stocks:
        await strategy_macd_rsi_kdj_boll_strategy_for_stock(favorite_stock.stock_code)
    stocks = await stock_basic_dao.get_all_stocks()
    for stock in stocks:
        await strategy_macd_rsi_kdj_boll_strategy_for_stock(stock.stock_code)
    logger.info("所有股票的macd_rsi_kdj_boll策略信号计算任务完成")

async def strategy_macd_rsi_kdj_boll_strategy_for_stock(stock_code: str):
    from services.indicator_services import IndicatorService
    from strategies.macd_rsi_kdj_boll_strategy import MacdRsiKdjBollStrategy
    from utils.cache_set import StrategyCacheSet
    logger.info(f"开始为股票 {stock_code} 运行 MACD+RSI+KDJ+BOLL 策略")
    service = IndicatorService()
    strategy = MacdRsiKdjBollStrategy() # 使用默认参数或加载配置
    cache_setter = StrategyCacheSet()
    # 定义主操作时间周期 (需要与 prepare_strategy_dataframe 和缓存键生成保持一致)
    main_timeframe = '15m' # 或者从策略参数获取 strategy.params.get('main_timeframe', '15m')

    try:
        # 1. 准备数据
        merged_data = asyncio.run(service.prepare_strategy_dataframe(
            stock_code=stock_code,
            timeframes=['5', '15', '30', '60'], # 与策略定义一致
            strategy_params=strategy.params,
            limit_per_tf=1500 # 根据需要调整 limit
        ))

        if merged_data is None or merged_data.empty:
            logger.warning(f"[{stock_code}] 未能准备策略所需数据，策略无法运行。")
            return f"[{stock_code}] 数据准备失败"

        # 2. 运行策略
        signal_series = strategy.run(merged_data)
        logger.info(f"[{stock_code}] 策略运行完成，信号序列: {signal_series}")

        if signal_series.empty or signal_series.isna().all(): # 检查是否为空或全是 NaN
             logger.info(f"[{stock_code}] 策略运行完成，但未生成有效信号 (可能数据不足或全为 NaN)。")
             # --- 新增：缓存空信号 ---
             try:
                 # 获取 merged_data 的最后一个时间戳作为参考
                 latest_timestamp_ref = merged_data.index[-1] if not merged_data.empty else pd.Timestamp.now(tz='UTC') # Fallback
                 # 准备空信号数据
                 signal_data_to_cache = {
                     'stock_code': stock_code,
                     'strategy_name': strategy.strategy_name,
                     'time_level': main_timeframe, # 使用主时间周期
                     'timestamp': latest_timestamp_ref.isoformat(), # 使用参考时间戳
                     'signal': None, # 或者使用 strategy.SIGNAL_NONE (np.nan) - 但需确保序列化处理
                     'signal_display': 'No Signal',
                     'score': None, # 如果策略计算了 score，可以缓存
                     'generated_at': pd.Timestamp.now(tz='UTC').isoformat()
                 }
                 # 调用缓存设置方法 (注意：macd_rsi_kdj_boll_data 可能需要调整以适应这种结构)
                 # 这里假设 macd_rsi_kdj_boll_data 接受一个字典并缓存它
                 cache_success = asyncio.run(cache_setter.macd_rsi_kdj_boll_data(
                     stock_code=stock_code,
                     time_level=main_timeframe, # 使用主时间周期
                     data_to_cache=signal_data_to_cache
                 ))
                 if cache_success:
                     logger.info(f"[{stock_code}] 策略无有效信号，已缓存空信号状态。")
                 else:
                     logger.warning(f"[{stock_code}] 缓存空信号状态失败。")
             except Exception as cache_err:
                 logger.error(f"[{stock_code}] 缓存空信号状态时发生错误: {cache_err}", exc_info=False)
             # --- 结束新增 ---
             return f"[{stock_code}] 策略运行无有效信号"

        # 3. 获取最新信号 (忽略 NaN)
        # 使用 dropna() 移除 NaN 信号，然后取最后一个有效信号
        valid_signals = signal_series.dropna()
        if valid_signals.empty:
             logger.info(f"[{stock_code}] 策略运行完成，但所有信号均为 NaN。")
             # 此处逻辑同上，缓存空信号
             try:
                 latest_timestamp_ref = merged_data.index[-1] if not merged_data.empty else pd.Timestamp.now(tz='UTC')
                 signal_data_to_cache = {
                     'stock_code': stock_code, 'strategy_name': strategy.strategy_name,
                     'time_level': main_timeframe, 'timestamp': latest_timestamp_ref.isoformat(),
                     'signal': None, 'signal_display': 'No Signal (NaN)', 'score': None,
                     'generated_at': pd.Timestamp.now(tz='UTC').isoformat()
                 }
                 cache_success = asyncio.run(cache_setter.macd_rsi_kdj_boll_data(stock_code, main_timeframe, signal_data_to_cache))
                 if cache_success: logger.info(f"[{stock_code}] 策略信号全为 NaN，已缓存空信号状态。")
                 else: logger.warning(f"[{stock_code}] 缓存 NaN 信号状态失败。")
             except Exception as cache_err: logger.error(f"[{stock_code}] 缓存 NaN 信号状态时发生错误: {cache_err}", exc_info=False)
             return f"[{stock_code}] 策略运行信号全为 NaN"

        latest_signal = valid_signals.iloc[-1]
        latest_timestamp = valid_signals.index[-1]

        # 转换信号值为可读字符串 (可选，但方便缓存查看)
        signal_map = {
            strategy.SIGNAL_STRONG_BUY: "Strong Buy",
            strategy.SIGNAL_BUY: "Buy",
            strategy.SIGNAL_HOLD: "Hold",
            strategy.SIGNAL_SELL: "Sell",
            strategy.SIGNAL_STRONG_SELL: "Strong Sell",
        }
        signal_display = signal_map.get(latest_signal, "Unknown")

        # 4. TODO: 处理信号 (例如存入数据库、发送通知、执行交易等)
        logger.info(f"[{stock_code}] 策略信号 @ {latest_timestamp}: {latest_signal} ({signal_display})")
        # 例如: save_strategy_signal(stock_code, strategy.strategy_name, latest_timestamp, latest_signal)

        # --- 5. 新增：缓存策略结果 ---
        try:
            # 准备要缓存的数据字典
            signal_data_to_cache = {
                'stock_code': stock_code,
                'strategy_name': strategy.strategy_name,
                'time_level': main_timeframe, # 使用主时间周期
                'timestamp': latest_timestamp.isoformat(), # 将 Timestamp 转换为 ISO 格式字符串
                'signal': int(latest_signal) if not pd.isna(latest_signal) else None, # 确保存储整数或 None
                'signal_display': signal_display,
                # 如果策略计算了 score，也可以缓存
                # 'score': merged_data.loc[latest_timestamp, 'total_score'] if 'total_score' in merged_data.columns else None,
                'generated_at': pd.Timestamp.now(tz='UTC').isoformat() # 记录缓存生成时间
            }

            # 调用缓存设置方法
            # 注意：这里使用了 StrategyCacheSet 的 macd_rsi_kdj_boll_data 方法
            # 这个方法内部会调用 _stock_strategy_data，最终调用 cache_manager.set
            cache_success = asyncio.run(cache_setter.macd_rsi_kdj_boll_data(
                stock_code=stock_code,
                time_level=main_timeframe, # 使用主时间周期与 key 生成对应
                data_to_cache=signal_data_to_cache
            ))

            if cache_success:
                logger.info(f"[{stock_code}] 策略结果成功缓存到 Redis。")
            else:
                logger.warning(f"[{stock_code}] 缓存策略结果到 Redis 失败。")

        except Exception as cache_err:
            logger.error(f"[{stock_code}] 缓存策略结果时发生错误: {cache_err}", exc_info=True) # 记录完整堆栈
        # --- 结束新增 ---

        return f"[{stock_code}] 策略运行成功，最新信号: {latest_signal} ({signal_display})"

    except Exception as e:
        logger.error(f"为股票 {stock_code} 运行策略时发生严重错误: {e}", exc_info=True)
        # 可以根据需要重新引发异常或返回错误状态
        return f"[{stock_code}] 策略运行出错: {e}"

# # 服务实例
# strategy_service = StrategyService()
# calculation_service = CalculationService()

# @shared_task
# def calculate_intraday_strategy():
#     """
#     计算日内高抛低吸策略
#     交易时间段每5分钟执行
#     """
#     logger.info("开始计算日内高抛低吸策略")
    
#     # 获取所有自选股的代码
#     stock_codes = list(FavoriteStock.objects.values_list('stock_code', flat=True).distinct())
    
#     if not stock_codes:
#         logger.info("没有自选股，无需计算策略")
#         return "没有自选股，无需计算策略"
    
#     asyncio.run(strategy_service.calculate_intraday_strategy(stock_codes))
#     logger.info("计算日内高抛低吸策略完成")
#     return "计算日内高抛低吸策略完成"

# @shared_task
# def calculate_wave_tracking_strategy():
#     """
#     计算波段跟踪及高抛低吸策略
#     交易时间结束后执行
#     """
#     logger.info("开始计算波段跟踪及高抛低吸策略")
    
#     # 获取所有自选股的代码
#     stock_codes = list(FavoriteStock.objects.values_list('stock_code', flat=True).distinct())
    
#     if not stock_codes:
#         logger.info("没有自选股，无需计算策略")
#         return "没有自选股，无需计算策略"
    
#     asyncio.run(strategy_service.calculate_wave_tracking_strategy(stock_codes))
#     logger.info("计算波段跟踪及高抛低吸策略完成")
#     return "计算波段跟踪及高抛低吸策略完成"

# @shared_task
# def check_stock_reversal():
#     """
#     检查股票反转状态
#     每天开盘前和收盘后执行
#     """
#     logger.info("开始检查股票反转状态")
    
#     # 获取所有自选股的代码
#     stock_codes = list(FavoriteStock.objects.values_list('stock_code', flat=True).distinct())
    
#     if not stock_codes:
#         logger.info("没有自选股，无需检查反转状态")
#         return "没有自选股，无需检查反转状态"
    
#     asyncio.run(strategy_service.check_stock_reversal(stock_codes))
#     logger.info("检查股票反转状态完成")
#     return "检查股票反转状态完成"

# @shared_task
# def calculate_intraday_signals():
#     """
#     计算日内信号
#     交易时间段每10分钟执行
#     """
#     logger.info("开始计算日内信号")
    
#     # 获取所有自选股的代码
#     stock_codes = list(FavoriteStock.objects.values_list('stock_code', flat=True).distinct())
    
#     if not stock_codes:
#         logger.info("没有自选股，无需计算日内信号")
#         return "没有自选股，无需计算日内信号"
    
#     asyncio.run(calculation_service.calculate_intraday_signals(stock_codes))
#     logger.info("计算日内信号完成")
#     return "计算日内信号完成"

# @shared_task
# def calculate_daily_signals():
#     """
#     计算日线信号
#     每天收盘后执行
#     """
#     logger.info("开始计算日线信号")
    
#     # 获取所有自选股的代码
#     stock_codes = list(FavoriteStock.objects.values_list('stock_code', flat=True).distinct())
    
#     if not stock_codes:
#         logger.info("没有自选股，无需计算日线信号")
#         return "没有自选股，无需计算日线信号"
    
#     asyncio.run(calculation_service.calculate_daily_signals(stock_codes))
#     logger.info("计算日线信号完成")
#     return "计算日线信号完成"

# @shared_task
# def calculate_market_signals():
#     """
#     计算市场整体信号
#     每天收盘后执行
#     """
#     logger.info("开始计算市场整体信号")
#     asyncio.run(calculation_service.calculate_market_signals())
#     logger.info("计算市场整体信号完成")
#     return "计算市场整体信号完成"

# @shared_task
# def manual_calculate_stock_strategy(stock_code):
#     """
#     手动触发计算单个股票的策略
    
#     Args:
#         stock_code: 股票代码
#     """
#     logger.info(f"手动开始计算股票{stock_code}的策略")
    
#     # 计算日内高抛低吸策略
#     asyncio.run(strategy_service.calculate_intraday_strategy([stock_code]))
    
#     # 计算波段跟踪及高抛低吸策略
#     asyncio.run(strategy_service.calculate_wave_tracking_strategy([stock_code]))
    
#     # 检查股票反转状态
#     asyncio.run(strategy_service.check_stock_reversal([stock_code]))
    
#     # 计算日内信号
#     asyncio.run(calculation_service.calculate_intraday_signals([stock_code]))
    
#     # 计算日线信号
#     asyncio.run(calculation_service.calculate_daily_signals([stock_code]))
    
#     logger.info(f"手动计算股票{stock_code}的策略完成")
#     return f"手动计算股票{stock_code}的策略完成"

# @shared_task
# def manual_calculate_all_favorites_strategy():
#     """
#     手动触发计算所有自选股的策略
#     """
#     # 获取所有自选股的代码
#     stock_codes = list(FavoriteStock.objects.values_list('stock_code', flat=True).distinct())
    
#     if not stock_codes:
#         logger.info("没有自选股，无需计算策略")
#         return "没有自选股，无需计算策略"
    
#     logger.info(f"手动开始计算{len(stock_codes)}只自选股的策略")
    
#     # 计算日内高抛低吸策略
#     asyncio.run(strategy_service.calculate_intraday_strategy(stock_codes))
    
#     # 计算波段跟踪及高抛低吸策略
#     asyncio.run(strategy_service.calculate_wave_tracking_strategy(stock_codes))
    
#     # 检查股票反转状态
#     asyncio.run(strategy_service.check_stock_reversal(stock_codes))
    
#     # 计算日内信号
#     asyncio.run(calculation_service.calculate_intraday_signals(stock_codes))
    
#     # 计算日线信号
#     asyncio.run(calculation_service.calculate_daily_signals(stock_codes))
    
#     # 计算市场整体信号
#     asyncio.run(calculation_service.calculate_market_signals())
    
#     logger.info(f"手动计算{len(stock_codes)}只自选股的策略完成")
#     return f"手动计算{len(stock_codes)}只自选股的策略完成"

# 添加在settings.CELERY_BEAT_SCHEDULE中定义的任务
@shared_task
def calculate_strategy():
    """
    定时任务：计算策略信号
    每5分钟执行一次
    包括各种交易策略的信号计算
    """
    logger.info("开始执行策略信号计算任务")
    try:
        # TODO: 实现具体的策略计算逻辑
        # 例如：计算各种交易策略的信号
        logger.info("策略信号计算成功")
        return "策略信号计算成功"
    except Exception as e:
        logger.error(f"策略信号计算失败: {str(e)}")
        return f"策略信号计算失败: {str(e)}" 