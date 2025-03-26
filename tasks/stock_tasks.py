"""
股票相关任务
提供股票数据的定时更新和手动触发任务
"""
import asyncio
import logging
from celery import shared_task
from api_manager.apis.stock_basic_api import StockBasicAPI
from api_manager.apis.stock_realtime_api import StockRealtimeAPI
from api_manager.apis.stock_indicators_api import StockIndicatorsAPI
from dao_manager.daos.stock_basic_dao import StockBasicDAO
from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO
from dao_manager.daos.stock_realtime_dao import StockRealtimeDAO
from users.models import FavoriteStock

logger = logging.getLogger(__name__)

# API和DAO实例
stock_basic_api = StockBasicAPI()
stock_realtime_api = StockRealtimeAPI()
stock_indicators_api = StockIndicatorsAPI()
stock_basic_dao = StockBasicDAO()
stock_realtime_dao = StockRealtimeDAO()
stock_indicators_dao = StockIndicatorsDAO()

@shared_task
def refresh_stock_basic_info():
    """
    刷新股票基础信息
    每天早上7点执行
    """
    logger.info("开始刷新股票基础信息")
    asyncio.run(stock_basic_dao.refresh_all_stocks())
    logger.info("刷新股票基础信息完成")
    return "刷新股票基础信息完成"

@shared_task
def refresh_stock_industry_info():
    """
    刷新股票行业信息
    每周一早上7:30执行
    """
    logger.info("开始刷新股票行业信息")
    asyncio.run(stock_basic_dao.refresh_stock_industry())
    logger.info("刷新股票行业信息完成")
    return "刷新股票行业信息完成"

@shared_task
def refresh_stock_concept_info():
    """
    刷新股票概念信息
    每周一早上8:00执行
    """
    logger.info("开始刷新股票概念信息")
    asyncio.run(stock_basic_dao.refresh_stock_concept())
    logger.info("刷新股票概念信息完成")
    return "刷新股票概念信息完成"

@shared_task
def refresh_favorites_realtime_data():
    """
    刷新自选股的实时数据
    交易时间段每分钟执行
    """
    logger.info("开始刷新自选股的实时数据")
    # 获取所有自选股的代码
    favorite_stock_codes = list(FavoriteStock.objects.values_list('stock_code', flat=True).distinct())
    
    if favorite_stock_codes:
        asyncio.run(stock_realtime_dao.refresh_stocks_realtime(favorite_stock_codes))
        logger.info(f"刷新{len(favorite_stock_codes)}只自选股的实时数据完成")
    else:
        logger.info("没有自选股，无需刷新")
    
    return "刷新自选股的实时数据完成"

@shared_task
def refresh_active_stocks_realtime_data():
    """
    刷新活跃股票的实时数据
    交易时间段每2分钟执行
    """
    logger.info("开始刷新活跃股票的实时数据")
    asyncio.run(stock_realtime_dao.refresh_active_stocks_realtime())
    logger.info("刷新活跃股票的实时数据完成")
    return "刷新活跃股票的实时数据完成"

@shared_task
def refresh_stock_time_series(period, stock_codes=None):
    """
    刷新股票K线数据
    根据不同周期和不同股票执行
    
    Args:
        period: K线周期 (1, 5, 15, 30, 60, Day, Week, Month)
        stock_codes: 要刷新的股票代码列表，为None时刷新自选股
    """
    if stock_codes is None:
        # 获取所有自选股的代码
        stock_codes = list(FavoriteStock.objects.values_list('stock_code', flat=True).distinct())
    
    if not stock_codes:
        logger.info("没有需要刷新的股票，无需刷新")
        return "没有需要刷新的股票，无需刷新"
    
    logger.info(f"开始刷新{len(stock_codes)}只股票的{period}周期K线数据")
    asyncio.run(stock_indicators_dao.refresh_stocks_time_series(stock_codes, period))
    logger.info(f"刷新{len(stock_codes)}只股票的{period}周期K线数据完成")
    return f"刷新{len(stock_codes)}只股票的{period}周期K线数据完成"

@shared_task
def refresh_stock_technical_indicators(period, stock_codes=None):
    """
    刷新股票技术指标
    日线数据每个交易日收盘后执行
    
    Args:
        period: K线周期 (Day)
        stock_codes: 要刷新的股票代码列表，为None时刷新自选股
    """
    if stock_codes is None:
        # 获取所有自选股的代码
        stock_codes = list(FavoriteStock.objects.values_list('stock_code', flat=True).distinct())
    
    if not stock_codes:
        logger.info("没有需要刷新的股票，无需刷新")
        return "没有需要刷新的股票，无需刷新"
    
    logger.info(f"开始刷新{len(stock_codes)}只股票的{period}技术指标")
    asyncio.run(stock_indicators_dao.refresh_stocks_technical_indicators(stock_codes, period))
    logger.info(f"刷新{len(stock_codes)}只股票的{period}技术指标完成")
    return f"刷新{len(stock_codes)}只股票的{period}技术指标完成"

@shared_task
def refresh_stock_level5_data(stock_codes=None):
    """
    刷新股票买卖五档数据
    交易时间段每5分钟执行
    
    Args:
        stock_codes: 要刷新的股票代码列表，为None时刷新自选股
    """
    if stock_codes is None:
        # 获取所有自选股的代码
        stock_codes = list(FavoriteStock.objects.values_list('stock_code', flat=True).distinct())
    
    if not stock_codes:
        logger.info("没有需要刷新的股票，无需刷新")
        return "没有需要刷新的股票，无需刷新"
    
    logger.info(f"开始刷新{len(stock_codes)}只股票的买卖五档数据")
    asyncio.run(stock_realtime_dao.refresh_stocks_level5(stock_codes))
    logger.info(f"刷新{len(stock_codes)}只股票的买卖五档数据完成")
    return f"刷新{len(stock_codes)}只股票的买卖五档数据完成"

@shared_task
def manual_refresh_stock_data(stock_code):
    """
    手动触发刷新单个股票的全部数据
    
    Args:
        stock_code: 股票代码
    """
    logger.info(f"手动开始刷新股票{stock_code}的全部数据")
    
    # 刷新基础信息
    asyncio.run(stock_basic_dao.refresh_stock_info(stock_code))
    
    # 刷新实时数据
    asyncio.run(stock_realtime_dao.refresh_stocks_realtime([stock_code]))
    
    # 刷新买卖五档
    asyncio.run(stock_realtime_dao.refresh_stocks_level5([stock_code]))
    
    # 刷新不同周期的K线数据
    periods = ['1', '5', '15', '30', '60', 'Day', 'Week', 'Month']
    for period in periods:
        asyncio.run(stock_indicators_dao.refresh_stocks_time_series([stock_code], period))
    
    # 刷新技术指标
    asyncio.run(stock_indicators_dao.refresh_stocks_technical_indicators([stock_code], 'Day'))
    
    logger.info(f"手动刷新股票{stock_code}的全部数据完成")
    return f"手动刷新股票{stock_code}的全部数据完成"

@shared_task
def manual_refresh_all_favorites_data():
    """
    手动触发刷新所有自选股的全部数据
    """
    # 获取所有自选股的代码
    stock_codes = list(FavoriteStock.objects.values_list('stock_code', flat=True).distinct())
    
    if not stock_codes:
        logger.info("没有自选股，无需刷新")
        return "没有自选股，无需刷新"
    
    logger.info(f"手动开始刷新{len(stock_codes)}只自选股的全部数据")
    
    # 刷新基础信息
    for stock_code in stock_codes:
        asyncio.run(stock_basic_dao.refresh_stock_info(stock_code))
    
    # 刷新实时数据
    asyncio.run(stock_realtime_dao.refresh_stocks_realtime(stock_codes))
    
    # 刷新买卖五档
    asyncio.run(stock_realtime_dao.refresh_stocks_level5(stock_codes))
    
    # 刷新不同周期的K线数据
    periods = ['1', '5', '15', '30', '60', 'Day', 'Week', 'Month']
    for period in periods:
        asyncio.run(stock_indicators_dao.refresh_stocks_time_series(stock_codes, period))
    
    # 刷新技术指标
    asyncio.run(stock_indicators_dao.refresh_stocks_technical_indicators(stock_codes, 'Day'))
    
    logger.info(f"手动刷新{len(stock_codes)}只自选股的全部数据完成")
    return f"手动刷新{len(stock_codes)}只自选股的全部数据完成" 