"""
策略相关任务
提供策略计算的定时任务和手动触发任务
"""
# import asyncio
# import logging
# from celery import shared_task
# from service.strategy_service import StrategyService
# from service.calculation_service import CalculationService
# from users.models import FavoriteStock

# logger = logging.getLogger(__name__)

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