"""
数据中心相关任务
提供数据中心数据的定时更新和手动触发任务
"""
import asyncio
import logging
from celery import shared_task
from api_manager.apis.datacenter_api import DataCenterAPI
from dao_manager.daos.datacenter_dao import DataCenterDAO

logger = logging.getLogger(__name__)

# API和DAO实例
datacenter_api = DataCenterAPI()
datacenter_dao = DataCenterDAO()

@shared_task
def refresh_financial_data():
    """
    刷新财务数据
    每天21:00执行
    """
    logger.info("开始刷新财务数据")
    asyncio.run(datacenter_dao.refresh_financial_data())
    logger.info("刷新财务数据完成")
    return "刷新财务数据完成"

@shared_task
def refresh_capital_flow_data():
    """
    刷新资金流向数据
    每天21:30执行
    """
    logger.info("开始刷新资金流向数据")
    asyncio.run(datacenter_dao.refresh_capital_flow_data())
    logger.info("刷新资金流向数据完成")
    return "刷新资金流向数据完成"

@shared_task
def refresh_lhb_data():
    """
    刷新龙虎榜数据
    每天21:45执行
    """
    logger.info("开始刷新龙虎榜数据")
    asyncio.run(datacenter_dao.refresh_lhb_data())
    logger.info("刷新龙虎榜数据完成")
    return "刷新龙虎榜数据完成"

@shared_task
def refresh_institution_data():
    """
    刷新机构持股数据
    每周一22:00执行
    """
    logger.info("开始刷新机构持股数据")
    asyncio.run(datacenter_dao.refresh_institution_data())
    logger.info("刷新机构持股数据完成")
    return "刷新机构持股数据完成"

@shared_task
def refresh_north_south_data():
    """
    刷新北向南向资金数据
    每天22:15执行
    """
    logger.info("开始刷新北向南向资金数据")
    asyncio.run(datacenter_dao.refresh_north_south_data())
    logger.info("刷新北向南向资金数据完成")
    return "刷新北向南向资金数据完成"

@shared_task
def refresh_statistics_data():
    """
    刷新统计数据
    每天22:30执行
    """
    logger.info("开始刷新统计数据")
    asyncio.run(datacenter_dao.refresh_statistics_data())
    logger.info("刷新统计数据完成")
    return "刷新统计数据完成"

@shared_task
def refresh_market_data():
    """
    刷新市场数据
    每天22:45执行
    """
    logger.info("开始刷新市场数据")
    asyncio.run(datacenter_dao.refresh_market_data())
    logger.info("刷新市场数据完成")
    return "刷新市场数据完成"

@shared_task
def manual_refresh_all_datacenter_data():
    """
    手动触发刷新所有数据中心数据的任务
    """
    logger.info("手动开始刷新所有数据中心数据")
    
    # 刷新财务数据
    asyncio.run(datacenter_dao.refresh_financial_data())
    
    # 刷新资金流向数据
    asyncio.run(datacenter_dao.refresh_capital_flow_data())
    
    # 刷新龙虎榜数据
    asyncio.run(datacenter_dao.refresh_lhb_data())
    
    # 刷新机构持股数据
    asyncio.run(datacenter_dao.refresh_institution_data())
    
    # 刷新北向南向资金数据
    asyncio.run(datacenter_dao.refresh_north_south_data())
    
    # 刷新统计数据
    asyncio.run(datacenter_dao.refresh_statistics_data())
    
    # 刷新市场数据
    asyncio.run(datacenter_dao.refresh_market_data())
    
    logger.info("手动刷新所有数据中心数据完成")
    return "手动刷新所有数据中心数据完成" 