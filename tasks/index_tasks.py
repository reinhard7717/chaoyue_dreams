"""
指数相关任务
提供指数数据的定时更新和手动触发任务
"""
import asyncio
import logging
from celery import shared_task


logger = logging.getLogger(__name__)

# API和DAO实例


@shared_task
def save_index_all_latest_realtime_data():
    """
    保存指数最新实时数据
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始保存指数最新实时数据")
    asyncio.run(index_dao.fetch_and_save_all_realtime_data())
    logger.info("保存指数最新实时数据完成")
    return "保存指数最新实时数据完成"

@shared_task
def save_index_all_latest_market_overview():
    """
    保存指数最新市场概览数据
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始保存指数最新市场概览数据")
    asyncio.run(index_dao.fetch_and_save_market_overview())
    logger.info("保存指数最新市场概览数据完成")
    return "保存指数最新市场概览数据完成"

@shared_task
def save_index_all_latest_time_series():
    """
    保存指数时间序列数据
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始保存指数时间序列数据")
    asyncio.run(index_dao.fetch_and_save_all_latest_time_series())
    logger.info("保存指数时间序列数据完成")
    return "保存指数时间序列数据完成"

@shared_task
def save_index_all_history_time_series():
    """
    保存指数历史时间序列数据
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始保存指数历史时间序列数据")
    asyncio.run(index_dao.fetch_and_save_all_history_time_series())
    logger.info("保存指数历史时间序列数据完成")
    return "保存指数历史时间序列数据完成"

@shared_task
def save_index_all_latest_kdj():
    """
    保存指数最新KDJ数据
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始保存指数最新KDJ数据")
    asyncio.run(index_dao.fetch_and_save_all_latest_kdj())
    logger.info("保存指数最新KDJ数据完成")
    return "保存指数最新KDJ数据完成"

@shared_task
def save_index_all_history_kdj():
    """
    保存指数历史KDJ数据
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始保存指数历史KDJ数据")
    asyncio.run(index_dao.fetch_and_save_all_history_kdj())
    logger.info("保存指数历史KDJ数据完成")
    return "保存指数历史KDJ数据完成"

@shared_task
def save_index_all_latest_macd():
    """
    保存指数最新MACD数据
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始保存指数最新MACD数据")
    asyncio.run(index_dao.fetch_and_save_all_latest_macd())
    logger.info("保存指数最新MACD数据完成")
    return "保存指数最新MACD数据完成"

@shared_task
def save_index_all_history_macd():
    """
    保存指数历史MACD数据
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始保存指数历史MACD数据")
    asyncio.run(index_dao.fetch_and_save_all_history_macd())
    logger.info("保存指数历史MACD数据完成")
    return "保存指数历史MACD数据完成"

@shared_task
def save_index_all_latest_boll():
    """
    保存指数最新BOLL数据
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始保存指数最新BOLL数据")
    asyncio.run(index_dao.fetch_and_save_all_latest_boll())
    logger.info("保存指数最新BOLL数据完成")
    return "保存指数最新BOLL数据完成"

@shared_task
def save_index_all_history_boll():
    """
    保存指数历史BOLL数据
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始保存指数历史BOLL数据")
    asyncio.run(index_dao.fetch_and_save_all_history_boll())
    logger.info("保存指数历史BOLL数据完成")
    return "保存指数历史BOLL数据完成"

@shared_task
def save_index_all_latest_ma():
    """
    保存指数最新MA数据
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始保存指数最新MA数据")
    asyncio.run(index_dao.fetch_and_save_all_latest_ma())
    logger.info("保存指数最新MA数据完成")
    return "保存指数最新MA数据完成"

@shared_task
def save_index_all_history_ma():
    """
    保存指数历史MA数据
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始保存指数历史MA数据")
    asyncio.run(index_dao.fetch_and_save_all_history_ma())
    logger.info("保存指数历史MA数据完成")
    return "保存指数历史MA数据完成"
