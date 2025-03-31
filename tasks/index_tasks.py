"""
指数相关任务
提供指数数据的定时更新和手动触发任务
"""
import asyncio
import logging
from celery import shared_task, group
from chaoyue_dreams.celery import app as celery_app
from utils.common import chunk_list  # 从 celery.py 导入 app 实例并重命名为 celery_app


logger = logging.getLogger(__name__)

# API和DAO实例


@celery_app.task(bind=True, name='tasks.index_tasks.save_index_all_latest_realtime_data')
async def save_index_all_latest_realtime_data(self):  # 添加 async def
    """
    保存指数最新实时数据
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始保存指数最新实时数据")
    await index_dao.fetch_and_save_all_realtime_data()  # 使用 await，移除 asyncio.run
    logger.info("保存指数最新实时数据完成")
    return "保存指数最新实时数据完成"

@celery_app.task(bind=True, name='tasks.index_tasks.save_index_all_latest_market_overview')  # 使用 @celery_app.task 装饰器，并指定任务名称
async def save_index_all_latest_market_overview(self): # 添加 async def
    """
    保存指数最新市场概览数据
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始保存指数最新市场概览数据")
    await index_dao.fetch_and_save_market_overview() # 使用 await，移除 asyncio.run
    logger.info("保存指数最新市场概览数据完成")
    return "保存指数最新市场概览数据完成"

@celery_app.task(bind=True, name='tasks.index_tasks.save_index_all_latest_time_series')  # 使用 @celery_app.task 装饰器，并指定任务名称
async def save_index_all_latest_time_series(self): # 添加 async def
    """
    保存指数时间序列数据
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始保存指数时间序列数据")
    await index_dao.fetch_and_save_all_latest_time_series() # 使用 await，移除 asyncio.run
    logger.info("保存指数时间序列数据完成")
    return "保存指数时间序列数据完成"

# @celery_app.task(bind=True, name='tasks.index_tasks.save_index_history_time_series_chunk')
# async def save_index_history_time_series_chunk(self, index_chunk):
#     """
#     保存指数历史时间序列数据块
#     """
#     from dao_manager.daos.index_dao import StockIndexDAO
#     index_dao = StockIndexDAO()
#     logger.info(f"开始保存指数历史时间序列数据块: {[index.code for index in index_chunk]}") # 打印指数代码，方便查看日志
#     await index_dao.fetch_and_save_history_time_series_for_indexes(index_chunk) # 修改方法名并确保 await
#     logger.info(f"保存指数历史时间序列数据块完成: {[index.code for index in index_chunk]}")
#     return f"保存指数历史时间序列数据块完成: {[index.code for index in index_chunk]}"

@celery_app.task(bind=True, name='tasks.index_tasks.save_index_all_history_time_series')
async def save_index_all_history_time_series(self):
    """
    保存指数历史时间序列数据 (简化版 - 无并行)
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始保存指数历史时间序列数据 (简化版)")
    await index_dao.fetch_and_save_all_history_time_series() # 直接调用原始方法
    logger.info("保存指数历史时间序列数据完成 (简化版)")
    return "保存指数历史时间序列数据完成 (简化版)"

@celery_app.task(bind=True, name='tasks.index_tasks.save_index_all_latest_kdj')  # 使用 @celery_app.task 装饰器，并指定任务名称
async def save_index_all_latest_kdj(self): # 添加 async def
    """
    保存指数最新KDJ数据
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始保存指数最新KDJ数据")
    await index_dao.fetch_and_save_all_latest_kdj() # 使用 await，移除 asyncio.run
    logger.info("保存指数最新KDJ数据完成")
    return "保存指数最新KDJ数据完成"

@celery_app.task(bind=True, name='tasks.index_tasks.save_index_all_history_kdj')  # 使用 @celery_app.task 装饰器，并指定任务名称
async def save_index_all_history_kdj(self): # 添加 async def
    """
    保存指数历史KDJ数据
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始保存指数历史KDJ数据")
    await index_dao.fetch_and_save_all_history_kdj() # 使用 await，移除 asyncio.run
    logger.info("保存指数历史KDJ数据完成")
    return "保存指数历史KDJ数据完成"

@celery_app.task(bind=True, name='tasks.index_tasks.save_index_all_latest_macd')  # 使用 @celery_app.task 装饰器，并指定任务名称
async def save_index_all_latest_macd(self): # 添加 async def
    """
    保存指数最新MACD数据
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始保存指数最新MACD数据")
    await index_dao.fetch_and_save_all_latest_macd() # 使用 await，移除 asyncio.run
    logger.info("保存指数最新MACD数据完成")
    return "保存指数最新MACD数据完成"

@celery_app.task(bind=True, name='tasks.index_tasks.save_index_all_history_macd')  # 使用 @celery_app.task 装饰器，并指定任务名称
async def save_index_all_history_macd(self): # 添加 async def
    """
    保存指数历史MACD数据
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始保存指数历史MACD数据")
    await index_dao.fetch_and_save_all_history_macd() # 使用 await，移除 asyncio.run
    logger.info("保存指数历史MACD数据完成")
    return "保存指数历史MACD数据完成"

@celery_app.task(bind=True, name='tasks.index_tasks.save_index_all_latest_boll')  # 使用 @celery_app.task 装饰器，并指定任务名称
async def save_index_all_latest_boll(self): # 添加 async def
    """
    保存指数最新BOLL数据
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始保存指数最新BOLL数据")
    await index_dao.fetch_and_save_all_latest_boll() # 使用 await，移除 asyncio.run
    logger.info("保存指数最新BOLL数据完成")
    return "保存指数最新BOLL数据完成"

@celery_app.task(bind=True, name='tasks.index_tasks.save_index_all_history_boll')  # 使用 @celery_app.task 装饰器，并指定任务名称
async def save_index_all_history_boll(self): # 添加 async def
    """
    保存指数历史BOLL数据
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始保存指数历史BOLL数据")
    await index_dao.fetch_and_save_all_history_boll() # 使用 await，移除 asyncio.run
    logger.info("保存指数历史BOLL数据完成")
    return "保存指数历史BOLL数据完成"

@celery_app.task(bind=True, name='tasks.index_tasks.save_index_all_latest_ma')  # 使用 @celery_app.task 装饰器，并指定任务名称
async def save_index_all_latest_ma(self): # 添加 async def
    """
    保存指数最新MA数据
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始保存指数最新MA数据")
    await index_dao.fetch_and_save_all_latest_ma() # 使用 await，移除 asyncio.run
    logger.info("保存指数最新MA数据完成")
    return "保存指数最新MA数据完成"

@celery_app.task(bind=True, name='tasks.index_tasks.save_index_all_history_ma')  # 使用 @celery_app.task 装饰器，并指定任务名称
async def save_index_all_history_ma(self): # 添加 async def
    """
    保存指数历史MA数据
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始保存指数历史MA数据")
    await index_dao.fetch_and_save_all_history_ma() # 使用 await，移除 asyncio.run
    logger.info("保存指数历史MA数据完成")
    return "保存指数历史MA数据完成"
