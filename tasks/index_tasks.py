"""
指数相关任务
提供指数数据的定时更新和手动触发任务
"""
import asyncio
import logging
from celery import shared_task

from api_manager.apis.index_api import StockIndexAPI
from dao_manager.daos.index_dao import StockIndexDAO

logger = logging.getLogger(__name__)

# API和DAO实例
index_api = StockIndexAPI()
index_dao = StockIndexDAO()

@shared_task
def refresh_indexes():
    """
    刷新所有指数基础信息
    每天早上8点执行
    """
    logger.info("开始刷新所有指数基础信息")
    asyncio.run(index_dao.refresh_all_indexes())
    logger.info("刷新所有指数基础信息完成")
    return "刷新所有指数基础信息完成"

@shared_task
def refresh_main_indexes_realtime_data():
    """
    刷新主要指数的实时数据
    交易时间段每分钟执行
    """
    logger.info("开始刷新主要指数的实时数据")
    asyncio.run(index_dao.refresh_main_indexes_realtime())
    logger.info("刷新主要指数的实时数据完成")
    return "刷新主要指数的实时数据完成"

@shared_task
def refresh_market_overview():
    """
    刷新市场概览信息
    交易时间段每2分钟执行
    """
    logger.info("开始刷新市场概览信息")
    asyncio.run(index_dao.refresh_market_overview())
    logger.info("刷新市场概览信息完成")
    return "刷新市场概览信息完成"

@shared_task
def refresh_main_indexes_time_series(period):
    """
    刷新主要指数的K线数据
    根据不同周期定时执行
    
    Args:
        period: K线周期 (5, 15, 30, 60, Day)
    """
    logger.info(f"开始刷新主要指数的{period}分钟K线数据")
    asyncio.run(index_dao.refresh_main_indexes_time_series(period))
    logger.info(f"刷新主要指数的{period}分钟K线数据完成")
    return f"刷新主要指数的{period}分钟K线数据完成"

@shared_task
def refresh_main_indexes_technical_indicators(period):
    """
    刷新主要指数的技术指标
    日线数据每个交易日收盘后执行
    
    Args:
        period: K线周期 (Day)
    """
    logger.info(f"开始刷新主要指数的{period}技术指标")
    asyncio.run(index_dao.refresh_main_indexes_technical_indicators(period))
    logger.info(f"刷新主要指数的{period}技术指标完成")
    return f"刷新主要指数的{period}技术指标完成"

@shared_task
def manual_refresh_all_index_data():
    """
    手动触发刷新所有指数数据的任务
    包括基础信息、实时数据、K线数据和技术指标
    """
    logger.info("手动开始刷新所有指数数据")
    
    # 刷新基础信息
    asyncio.run(index_dao.refresh_all_indexes())
    
    # 刷新实时数据
    asyncio.run(index_dao.refresh_main_indexes_realtime())
    
    # 刷新市场概览
    asyncio.run(index_dao.refresh_market_overview())
    
    # 刷新不同周期的K线数据
    periods = ['5', '15', '30', '60', 'Day']
    for period in periods:
        asyncio.run(index_dao.refresh_main_indexes_time_series(period))
    
    # 刷新技术指标
    asyncio.run(index_dao.refresh_main_indexes_technical_indicators('Day'))
    
    logger.info("手动刷新所有指数数据完成")
    return "手动刷新所有指数数据完成" 