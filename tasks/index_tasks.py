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

@celery_app.task(bind=True, name='tasks.index_tasks.save_index_all_history_time_series')
def save_index_history_time_series_chunk(self, index_chunk):
    """
    保存指数历史时间序列数据块
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info(f"开始保存指数历史时间序列数据块: {[index.code for index in index_chunk]}") # 打印指数代码，方便查看日志
    asyncio.run(index_dao.fetch_and_save_history_time_series_by_indexs(index_chunk)) # 使用新的方法处理指数对象块
    logger.info(f"保存指数历史时间序列数据块完成: {[index.code for index in index_chunk]}")
    return f"保存指数历史时间序列数据块完成: {[index.code for index in index_chunk]}"

@celery_app.task(bind=True, name='tasks.index_tasks.save_index_all_history_time_series_chunk')
def save_index_all_history_time_series(self):
    """
    保存指数历史时间序列数据(并行版本)
    """
    from dao_manager.daos.index_dao import StockIndexDAO
    index_dao = StockIndexDAO()
    logger.info("开始并行保存指数历史时间序列数据")

    indexes = index_dao.get_all_indexes() # 获取所有指数对象列表
    if not indexes:
        logger.info("没有需要更新历史时间序列数据的指数")
        return "没有需要更新历史时间序列数据的指数"

    chunk_size = len(indexes) // 8 + 1  # 分割成 8 块，可以根据实际情况调整
    index_chunks = chunk_list(indexes, chunk_size) # 使用 chunk_list 分割指数对象列表

    tasks = [save_index_history_time_series_chunk.s(chunk) for chunk in index_chunks] # 创建子任务签名
    job = group(tasks) # 创建任务组
    async_result = job.apply_async() # 异步执行任务组

    # async_result.get() # 如果需要等待所有子任务完成，可以调用 get()，否则可以省略

    logger.info("并行保存指数历史时间序列数据任务调度完成，子任务正在后台执行")
    return "并行保存指数历史时间序列数据任务调度完成，子任务正在后台执行"

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
