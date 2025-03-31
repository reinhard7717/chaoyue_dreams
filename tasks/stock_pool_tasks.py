"""
股票池相关任务
提供股票池数据的定时更新和手动触发任务
"""
import asyncio
import logging
from celery import shared_task



logger = logging.getLogger(__name__)

# API和DAO实例


@shared_task
def refresh_daily_limit_pools():
    """
    刷新涨跌停股票池
    交易时间段每10分钟执行
    """
    from dao_manager.daos.fund_flow_dao import StockPoolDAO
    stock_pool_dao = StockPoolDAO()
    logger.info("开始刷新涨跌停股票池")
    asyncio.run(stock_pool_dao.refresh_daily_limit_pools())
    logger.info("刷新涨跌停股票池完成")
    return "刷新涨跌停股票池完成"

@shared_task
def refresh_daily_strong_stocks():
    """
    刷新强势股票池
    交易时间段每10分钟执行
    """
    from dao_manager.daos.fund_flow_dao import StockPoolDAO
    stock_pool_dao = StockPoolDAO()
    logger.info("开始刷新强势股票池")
    asyncio.run(stock_pool_dao.refresh_daily_strong_stocks())
    logger.info("刷新强势股票池完成")
    return "刷新强势股票池完成"

@shared_task
def refresh_break_limit_pools():
    """
    刷新炸板股票池
    交易时间段每5分钟执行
    """
    from dao_manager.daos.fund_flow_dao import StockPoolDAO
    stock_pool_dao = StockPoolDAO()
    logger.info("开始刷新炸板股票池")
    asyncio.run(stock_pool_dao.refresh_break_limit_pools())
    logger.info("刷新炸板股票池完成")
    return "刷新炸板股票池完成"

@shared_task
def refresh_new_stock_pools():
    """
    刷新次新股票池
    每周一早上9:00执行
    """
    from dao_manager.daos.fund_flow_dao import StockPoolDAO
    stock_pool_dao = StockPoolDAO()
    logger.info("开始刷新次新股票池")
    asyncio.run(stock_pool_dao.refresh_new_stock_pools())
    logger.info("刷新次新股票池完成")
    return "刷新次新股票池完成"

@shared_task
def refresh_concept_top_stocks():
    """
    刷新概念排行榜前十股票池
    每天收盘后执行
    """
    from dao_manager.daos.fund_flow_dao import StockPoolDAO
    stock_pool_dao = StockPoolDAO()
    logger.info("开始刷新概念排行榜前十股票池")
    asyncio.run(stock_pool_dao.refresh_concept_top_stocks())
    logger.info("刷新概念排行榜前十股票池完成")
    return "刷新概念排行榜前十股票池完成"

@shared_task
def refresh_industry_top_stocks():
    """
    刷新行业排行榜前十股票池
    每天收盘后执行
    """
    from dao_manager.daos.fund_flow_dao import StockPoolDAO
    stock_pool_dao = StockPoolDAO()
    logger.info("开始刷新行业排行榜前十股票池")
    asyncio.run(stock_pool_dao.refresh_industry_top_stocks())
    logger.info("刷新行业排行榜前十股票池完成")
    return "刷新行业排行榜前十股票池完成"

@shared_task
def manual_refresh_all_stock_pools():
    """
    手动触发刷新所有股票池的任务
    """
    from dao_manager.daos.fund_flow_dao import StockPoolDAO
    stock_pool_dao = StockPoolDAO()
    logger.info("手动开始刷新所有股票池")
    
    # 刷新涨跌停股票池
    asyncio.run(stock_pool_dao.refresh_daily_limit_pools())
    
    # 刷新强势股票池
    asyncio.run(stock_pool_dao.refresh_daily_strong_stocks())
    
    # 刷新炸板股票池
    asyncio.run(stock_pool_dao.refresh_break_limit_pools())
    
    # 刷新次新股票池
    asyncio.run(stock_pool_dao.refresh_new_stock_pools())
    
    # 刷新概念排行榜前十股票池
    asyncio.run(stock_pool_dao.refresh_concept_top_stocks())
    
    # 刷新行业排行榜前十股票池
    asyncio.run(stock_pool_dao.refresh_industry_top_stocks())
    
    logger.info("手动刷新所有股票池完成")
    return "手动刷新所有股票池完成" 