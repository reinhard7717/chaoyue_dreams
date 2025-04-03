"""
指数相关任务
提供指数数据的定时更新和手动触发任务
"""
import asyncio
import logging
from celery import shared_task
from chaoyue_dreams.celery import app as celery_app  # 从 celery.py 导入 app 实例并重命名为 celery_app

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, name='tasks.stock_indicators.save_stock_all_latest_realtime_data')
def save_stock_all_latest_realtime_data(self):
    """
    保存股票最新实时数据
    """
    from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO
    stock_indicators_dao = StockIndicatorsDAO()
    logger.info("开始保存股票最新实时数据")
    asyncio.run(stock_indicators_dao.fetch_and_save_all_latest_time_trade())
    logger.info("保存股票最新实时数据完成")
    return "保存股票最新实时数据完成"

@celery_app.task(bind=True, name='tasks.stock_indicators.save_stock_all_latest_kdj')
def save_stock_all_latest_kdj(self):
    """
    保存股票最新KDJ数据
    """
    from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO
    stock_indicators_dao = StockIndicatorsDAO()
    logger.info("开始保存股票最新KDJ数据")
    asyncio.run(stock_indicators_dao.fetch_and_save_all_latest_kdj())
    logger.info("保存股票最新KDJ数据完成")
    return "保存股票最新KDJ数据完成"

@celery_app.task(bind=True, name='tasks.stock_indicators.save_stock_all_latest_macd')
def save_stock_all_latest_macd(self):
    """
    保存股票最新MACD数据
    """
    from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO
    stock_indicators_dao = StockIndicatorsDAO()
    logger.info("开始保存股票最新MACD数据")
    asyncio.run(stock_indicators_dao.fetch_and_save_all_latest_macd())
    logger.info("保存股票最新MACD数据完成")
    return "保存股票最新MACD数据完成"

@celery_app.task(bind=True, name='tasks.stock_indicators.save_stock_all_latest_ma')
def save_stock_all_latest_ma(self):
    """
    保存股票最新MA数据
    """
    from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO
    stock_indicators_dao = StockIndicatorsDAO()
    logger.info("开始保存股票最新MA数据")
    asyncio.run(stock_indicators_dao.fetch_and_save_all_latest_ma())
    logger.info("保存股票最新MA数据完成")
    return "保存股票最新MA数据完成"

@celery_app.task(bind=True, name='tasks.stock_indicators.save_stock_all_latest_boll')
def save_stock_all_latest_boll(self):
    """
    保存股票最新BOLL数据
    """
    from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO
    stock_indicators_dao = StockIndicatorsDAO()
    logger.info("开始保存股票最新BOLL数据")
    asyncio.run(stock_indicators_dao.fetch_and_save_all_latest_boll())
    logger.info("保存股票最新BOLL数据完成")
    return "保存股票最新BOLL数据完成"

@celery_app.task(bind=True, name='tasks.stock_indicators.save_all_history_time_trade')
def save_all_history_time_trade(stock_code: str):
    """
    保存股票历史实时数据
    """
    from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO
    stock_indicators_dao = StockIndicatorsDAO()
    logger.info("开始保存股票历史实时数据")
    asyncio.run(stock_indicators_dao.fetch_and_save_all_history_time_trade())
    logger.info("保存股票历史实时数据完成")
    return "保存股票历史实时数据完成"

@celery_app.task(bind=True, name='tasks.stock_indicators.save_all_history_kdj')
def save_all_history_kdj(stock_code: str):
    """
    保存股票历史KDJ数据
    """
    from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO
    stock_indicators_dao = StockIndicatorsDAO()
    logger.info("开始保存股票历史KDJ数据")
    asyncio.run(stock_indicators_dao.fetch_and_save_all_history_kdj())
    logger.info("保存股票历史KDJ数据完成")
    return "保存股票历史KDJ数据完成"

@celery_app.task(bind=True, name='tasks.stock_indicators.save_all_history_macd')
def save_all_history_macd(stock_code: str):
    """
    保存股票历史MACD数据
    """
    from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO
    stock_indicators_dao = StockIndicatorsDAO()
    logger.info("开始保存股票历史MACD数据")
    asyncio.run(stock_indicators_dao.fetch_and_save_all_history_macd())
    logger.info("保存股票历史MACD数据完成")
    return "保存股票历史MACD数据完成"

@celery_app.task(bind=True, name='tasks.stock_indicators.save_all_history_ma')
def save_all_history_ma(stock_code: str):
    """
    保存股票历史MA数据
    """
    from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO
    stock_indicators_dao = StockIndicatorsDAO()
    logger.info("开始保存股票历史MA数据")
    asyncio.run(stock_indicators_dao.fetch_and_save_all_history_ma())
    logger.info("保存股票历史MA数据完成")
    return "保存股票历史MA数据完成"

@celery_app.task(bind=True, name='tasks.stock_indicators.save_all_history_boll')
def save_all_history_boll(stock_code: str):
    """
    保存股票历史BOLL数据
    """
    from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO
    stock_indicators_dao = StockIndicatorsDAO()
    logger.info("开始保存股票历史BOLL数据")
    asyncio.run(stock_indicators_dao.fetch_and_save_all_history_boll())
    logger.info("保存股票历史BOLL数据完成")
    return "保存股票历史BOLL数据完成"
























