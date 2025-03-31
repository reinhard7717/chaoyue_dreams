"""
指数相关任务
提供指数数据的定时更新和手动触发任务
"""
import asyncio
import logging
from celery import shared_task
from chaoyue_dreams.celery import app as celery_app  # 从 celery.py 导入 app 实例并重命名为 celery_app


logger = logging.getLogger(__name__)

# API和DAO实例


@celery_app.task(bind=True, name='tasks.index_tasks.save_index_all_latest_realtime_data')  # 使用 @celery_app.task 装饰器，并指定任务名称
def save_index_all_latest_realtime_data(self):
    """
    保存指数最新实时数据
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始保存指数最新实时数据")
    asyncio.run(index_dao.fetch_and_save_all_realtime_data())
    logger.info("保存指数最新实时数据完成")
    return "保存指数最新实时数据完成"

@celery_app.task(bind=True, name='tasks.index_tasks.save_index_all_latest_market_overview')  # 使用 @celery_app.task 装饰器，并指定任务名称
def save_index_all_latest_market_overview(self):
    """
    保存指数最新市场概览数据
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始保存指数最新市场概览数据")
    asyncio.run(index_dao.fetch_and_save_market_overview())
    logger.info("保存指数最新市场概览数据完成")
    return "保存指数最新市场概览数据完成"

@celery_app.task(bind=True, name='tasks.index_tasks.save_index_all_latest_time_series')  # 使用 @celery_app.task 装饰器，并指定任务名称
def save_index_all_latest_time_series(self):
    """
    保存指数时间序列数据
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始保存指数时间序列数据")
    asyncio.run(index_dao.fetch_and_save_all_latest_time_series())
    logger.info("保存指数时间序列数据完成")
    return "保存指数时间序列数据完成"

@celery_app.task(bind=True, name='tasks.index_tasks.save_index_all_history_time_series')  # 使用 @celery_app.task 装饰器，并指定任务名称
def save_index_all_history_time_series(self):
    """
    保存指数历史时间序列数据
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始保存指数历史时间序列数据")
    asyncio.run(index_dao.fetch_and_save_all_history_time_series())
    logger.info("保存指数历史时间序列数据完成")
    return "保存指数历史时间序列数据完成"

@celery_app.task(bind=True, name='tasks.index_tasks.save_index_all_latest_kdj')  # 使用 @celery_app.task 装饰器，并指定任务名称
def save_index_all_latest_kdj(self):
    """
    保存指数最新KDJ数据
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始保存指数最新KDJ数据")
    asyncio.run(index_dao.fetch_and_save_all_latest_kdj())
    logger.info("保存指数最新KDJ数据完成")
    return "保存指数最新KDJ数据完成"

@celery_app.task(bind=True, name='tasks.index_tasks.save_index_all_history_kdj')  # 使用 @celery_app.task 装饰器，并指定任务名称
def save_index_all_history_kdj(self):
    """
    保存指数历史KDJ数据
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始保存指数历史KDJ数据")
    asyncio.run(index_dao.fetch_and_save_all_history_kdj())
    logger.info("保存指数历史KDJ数据完成")
    return "保存指数历史KDJ数据完成"

@celery_app.task(bind=True, name='tasks.index_tasks.save_index_all_latest_macd')  # 使用 @celery_app.task 装饰器，并指定任务名称
def save_index_all_latest_macd(self):
    """
    保存指数最新MACD数据
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始保存指数最新MACD数据")
    asyncio.run(index_dao.fetch_and_save_all_latest_macd())
    logger.info("保存指数最新MACD数据完成")
    return "保存指数最新MACD数据完成"

@celery_app.task(bind=True, name='tasks.index_tasks.save_index_all_history_macd')  # 使用 @celery_app.task 装饰器，并指定任务名称
def save_index_all_history_macd(self):
    """
    保存指数历史MACD数据
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始保存指数历史MACD数据")
    asyncio.run(index_dao.fetch_and_save_all_history_macd())
    logger.info("保存指数历史MACD数据完成")
    return "保存指数历史MACD数据完成"

@celery_app.task(bind=True, name='tasks.index_tasks.save_index_all_latest_boll')  # 使用 @celery_app.task 装饰器，并指定任务名称
def save_index_all_latest_boll(self):
    """
    保存指数最新BOLL数据
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始保存指数最新BOLL数据")
    asyncio.run(index_dao.fetch_and_save_all_latest_boll())
    logger.info("保存指数最新BOLL数据完成")
    return "保存指数最新BOLL数据完成"

@celery_app.task(bind=True, name='tasks.index_tasks.save_index_all_history_boll')  # 使用 @celery_app.task 装饰器，并指定任务名称
def save_index_all_history_boll(self):
    """
    保存指数历史BOLL数据
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始保存指数历史BOLL数据")
    asyncio.run(index_dao.fetch_and_save_all_history_boll())
    logger.info("保存指数历史BOLL数据完成")
    return "保存指数历史BOLL数据完成"

@celery_app.task(bind=True, name='tasks.index_tasks.save_index_all_latest_ma')  # 使用 @celery_app.task 装饰器，并指定任务名称
def save_index_all_latest_ma(self):
    """
    保存指数最新MA数据
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始保存指数最新MA数据")
    asyncio.run(index_dao.fetch_and_save_all_latest_ma())
    logger.info("保存指数最新MA数据完成")
    return "保存指数最新MA数据完成"

@celery_app.task(bind=True, name='tasks.index_tasks.save_index_all_history_ma')  # 使用 @celery_app.task 装饰器，并指定任务名称
def save_index_all_history_ma(self):
    """
    保存指数历史MA数据
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始保存指数历史MA数据")
    asyncio.run(index_dao.fetch_and_save_all_history_ma())
    logger.info("保存指数历史MA数据完成")
    return "保存指数历史MA数据完成"
