"""
策略相关任务
提供策略计算的定时任务和手动触发任务
"""
# import asyncio
import asyncio
import logging
from celery import shared_task
from chaoyue_dreams.celery import app as celery_app  # 从 celery.py 导入 app 实例并重命名为 celery_app
# from service.strategy_service import StrategyService
# from service.calculation_service import CalculationService
# from users.models import FavoriteStock

logger = logging.getLogger(__name__)

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
    logger.info(f"开始为股票 {stock_code} 运行 MACD+RSI+KDJ+BOLL 策略")
    service = IndicatorService()
    strategy = MacdRsiKdjBollStrategy() # 使用默认参数或加载配置
    try:
        # 准备数据
        # 注意：需要传递 strategy.params 给 prepare_strategy_dataframe
        merged_data = asyncio.run(service.prepare_strategy_dataframe(
            stock_code=stock_code,
            timeframes=['5m', '15m', '30m', '60m'], # 与策略定义一致
            strategy_params=strategy.params, # 传递策略参数
            limit_per_tf=1500 # 根据需要调整 limit
        ))
        if merged_data is None or merged_data.empty:
            logger.warning(f"[{stock_code}] 未能准备策略所需数据，策略无法运行。")
            return f"[{stock_code}] 数据准备失败"
        # 运行策略
        signal_series = strategy.run(merged_data)
        if signal_series.empty:
             logger.info(f"[{stock_code}] 策略运行完成，但未生成信号 (可能数据不足)。")
             return f"[{stock_code}] 策略运行无信号"
        # 获取最新信号
        latest_signal = signal_series.iloc[-1]
        latest_timestamp = signal_series.index[-1]
        # TODO: 处理信号 (例如存入数据库、发送通知、执行交易等)
        logger.info(f"[{stock_code}] 策略信号 @ {latest_timestamp}: {latest_signal}")
        # 例如: save_strategy_signal(stock_code, strategy.strategy_name, latest_timestamp, latest_signal)
        return f"[{stock_code}] 策略运行成功，最新信号: {latest_signal}"
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