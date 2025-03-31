"""
数据中心相关任务
提供数据中心数据的定时更新和手动触发任务
"""
import sys
import asyncio
import logging
from celery import shared_task
from datetime import datetime

# 解决Python 3.12上asyncio.coroutines没有_DEBUG属性的问题
if sys.version_info >= (3, 12):
    # 在Python 3.12中，_DEBUG被替换为_is_debug_mode函数
    if not hasattr(asyncio.coroutines, '_DEBUG'):
        # 为了兼容性，添加一个_DEBUG属性，其值由_is_debug_mode()函数确定
        asyncio.coroutines._DEBUG = asyncio.coroutines._is_debug_mode()

logger = logging.getLogger(__name__)


@shared_task
def refresh_financial_data():
    """
    刷新财务数据
    每天21:00执行
    """
    from dao_manager.daos.data_center.financial_dao import FinancialDao
    financial_dao = FinancialDao()
    logger.info("开始刷新财务数据")
    asyncio.run(financial_dao.save_weekly_rank_change())
    asyncio.run(financial_dao.save_monthly_rank_change())
    asyncio.run(financial_dao.save_weekly_strong_stocks())
    asyncio.run(financial_dao.save_monthly_strong_stocks())
    asyncio.run(financial_dao.save_circ_market_value_rank())
    asyncio.run(financial_dao.save_pe_ratio_rank())
    logger.info("刷新财务数据完成")
    return "刷新财务数据完成"

@shared_task
def refresh_capital_flow_data():
    """
    刷新资金流向数据
    每天21:30执行
    """
    from dao_manager.daos.data_center.capital_flow_dao import CapitalFlowDao
    capital_flow_dao = CapitalFlowDao()
    logger.info("开始刷新资金流向数据")
    asyncio.run(capital_flow_dao.save_industry_capital_flow())
    asyncio.run(capital_flow_dao.save_concept_capital_flow())
    asyncio.run(capital_flow_dao.save_stock_capital_flow())
    logger.info("刷新资金流向数据完成")
    return "刷新资金流向数据完成"

@shared_task
def refresh_lhb_data():
    """
    刷新龙虎榜数据
    每天21:45执行
    """
    from dao_manager.daos.data_center.lhb_dao import LhbDAO
    lhb_dao = LhbDAO()
    periods = ['5', '10', '30', '60']
    logger.info("开始刷新龙虎榜数据")
    asyncio.run(lhb_dao.save_daily_lhb())
    for period in periods:
        # 获取保存近n日上榜个股
        lhb_dao.save_stock_on_list(period)
        logger.info(f'龙虎榜数据 - 保存近{period}日上榜个股')

        # 获取保存近n日上榜营业部
        lhb_dao.save_broker_on_list(period)
        logger.info('龙虎榜数据 - 保存近n日上榜营业部')

        # 获取保存近n日机构交易跟踪
        lhb_dao.save_institution_trade_track(period)
        logger.info('龙虎榜数据 - 保存近n日机构交易跟踪')

        # 获取保存近n日机构交易明细
        lhb_dao.save_institution_trade_detail(period)
        logger.info('龙虎榜数据 - 保存近n日机构交易明细')
    logger.info("刷新龙虎榜数据完成")
    return "刷新龙虎榜数据完成"

@shared_task
def refresh_institution_data():
    """
    刷新机构持股数据
    每周一22:00执行
    """
    from dao_manager.daos.data_center.institutional_shareholding_dao import InstitutionalShareholdingDao
    institutional_shareholding_dao = InstitutionalShareholdingDao()
    logger.info("开始刷新机构持股数据")
    
    # 获取当前年份和季度
    current_date = datetime.now()
    year = current_date.year
    # 根据当前月份确定季度
    month = current_date.month
    if month <= 3:
        quarter = 1
    elif month <= 6:
        quarter = 2
    elif month <= 9:
        quarter = 3
    else:
        quarter = 4
    
    logger.info(f"开始刷新{year}年第{quarter}季度机构持股数据")
    
    # 刷新机构持股汇总数据
    asyncio.run(institutional_shareholding_dao.save_institution_holding_summary(year, quarter))
    
    # 刷新基金重仓数据
    asyncio.run(institutional_shareholding_dao.save_fund_heavy_positions(year, quarter))
    
    # 刷新社保重仓数据
    asyncio.run(institutional_shareholding_dao.save_social_security_heavy_positions(year, quarter))
    
    # 刷新QFII重仓数据
    asyncio.run(institutional_shareholding_dao.save_qfii_heavy_positions(year, quarter))
    
    logger.info(f"刷新{year}年第{quarter}季度机构持股数据完成")
    return f"刷新{year}年第{quarter}季度机构持股数据完成"

@shared_task
def refresh_north_south_data():
    """
    刷新北向南向资金数据
    每天22:15执行
    """
    from dao_manager.daos.data_center.north_south_dao import NorthSouthDao
    north_south_dao = NorthSouthDao()
    logger.info("开始刷新北向南向资金数据")
    asyncio.run(north_south_dao.save_north_south_fund_overview())
    asyncio.run(north_south_dao.save_north_fund_trend())
    asyncio.run(north_south_dao.save_south_fund_trend())
    asyncio.run(north_south_dao.save_north_stock_holding())
    logger.info("刷新北向南向资金数据完成")
    return "刷新北向南向资金数据完成"

@shared_task
def refresh_statistics_data():
    """
    刷新统计数据
    每天22:30执行
    """
    from dao_manager.daos.data_center.stock_statistics_dao import StockStatisticsDao
    logger.info("开始刷新统计数据")
    stock_statistics_dao = StockStatisticsDao()
    asyncio.run(stock_statistics_dao.save_stage_high_low())
    asyncio.run(stock_statistics_dao.save_new_high_stocks())
    asyncio.run(stock_statistics_dao.save_new_low_stocks())
    logger.info("刷新统计数据完成")
    return "刷新统计数据完成"

@shared_task
def manual_refresh_all_datacenter_data():
    """
    手动触发刷新所有数据中心数据的任务
    """
    logger.info("手动开始刷新所有数据中心数据")

    
    logger.info("手动刷新所有数据中心数据完成")
    return "手动刷新所有数据中心数据完成" 