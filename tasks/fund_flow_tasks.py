"""
资金流向相关任务
提供资金流向数据的定时更新和手动触发任务
"""
import asyncio
import logging
from celery import shared_task
from api_manager.apis.fund_flow_api import FundFlowAPI
from dao_manager.daos.fund_flow_dao import FundFlowDAO

logger = logging.getLogger(__name__)

# API和DAO实例
fund_flow_api = FundFlowAPI()
fund_flow_dao = FundFlowDAO()

@shared_task
def refresh_popular_stocks_fund_flow():
    """
    刷新热门股票的资金流向数据
    每天20:30执行
    """
    logger.info("开始刷新热门股票的资金流向数据")
    asyncio.run(fund_flow_dao.refresh_popular_stocks_fund_flow())
    logger.info("刷新热门股票的资金流向数据完成")
    return "刷新热门股票的资金流向数据完成"

@shared_task
def refresh_active_stocks_fund_flow_minute():
    """
    刷新活跃股票的分钟资金流向数据
    每天20:45执行
    """
    logger.info("开始刷新活跃股票的分钟资金流向数据")
    asyncio.run(fund_flow_dao.refresh_active_stocks_fund_flow_minute())
    logger.info("刷新活跃股票的分钟资金流向数据完成")
    return "刷新活跃股票的分钟资金流向数据完成"

@shared_task
def refresh_sector_fund_flow():
    """
    刷新行业和概念板块的资金流向数据
    每天收盘后执行
    """
    logger.info("开始刷新行业和概念板块的资金流向数据")
    asyncio.run(fund_flow_dao.refresh_sector_fund_flow())
    logger.info("刷新行业和概念板块的资金流向数据完成")
    return "刷新行业和概念板块的资金流向数据完成"

@shared_task
def refresh_market_main_force_phase():
    """
    刷新市场主力资金动向阶段数据
    每天收盘后执行
    """
    logger.info("开始刷新市场主力资金动向阶段数据")
    asyncio.run(fund_flow_dao.refresh_market_main_force_phase())
    logger.info("刷新市场主力资金动向阶段数据完成")
    return "刷新市场主力资金动向阶段数据完成"

@shared_task
def refresh_stock_transaction_distribution():
    """
    刷新股票历史成交分布数据
    每天收盘后执行
    """
    logger.info("开始刷新股票历史成交分布数据")
    asyncio.run(fund_flow_dao.refresh_stock_transaction_distribution())
    logger.info("刷新股票历史成交分布数据完成")
    return "刷新股票历史成交分布数据完成"

@shared_task
def refresh_north_south_fund_flow():
    """
    刷新北向南向资金流向数据
    交易时间每小时执行
    """
    logger.info("开始刷新北向南向资金流向数据")
    asyncio.run(fund_flow_dao.refresh_north_south_fund_flow())
    logger.info("刷新北向南向资金流向数据完成")
    return "刷新北向南向资金流向数据完成"

@shared_task
def manual_refresh_all_fund_flow_data():
    """
    手动触发刷新所有资金流向数据的任务
    """
    logger.info("手动开始刷新所有资金流向数据")
    
    # 刷新热门股票资金流向
    asyncio.run(fund_flow_dao.refresh_popular_stocks_fund_flow())
    
    # 刷新活跃股票分钟资金流向
    asyncio.run(fund_flow_dao.refresh_active_stocks_fund_flow_minute())
    
    # 刷新行业板块资金流向
    asyncio.run(fund_flow_dao.refresh_sector_fund_flow())
    
    # 刷新主力资金动向阶段数据
    asyncio.run(fund_flow_dao.refresh_market_main_force_phase())
    
    # 刷新历史成交分布
    asyncio.run(fund_flow_dao.refresh_stock_transaction_distribution())
    
    # 刷新北向南向资金流向
    asyncio.run(fund_flow_dao.refresh_north_south_fund_flow())
    
    logger.info("手动刷新所有资金流向数据完成")
    return "手动刷新所有资金流向数据完成" 