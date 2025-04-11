# stock_data_app/tasks.py
import asyncio
import logging
from celery import shared_task
from django.db import models # 导入 models 以便 sync_to_async 可以找到它
from asgiref.sync import sync_to_async
import math

from core.constants import TIME_TEADE_TIME_LEVELS_LITE

# 获取 logger 实例
logger = logging.getLogger('celery') # 或者使用你项目配置的 logger


# --- 核心处理逻辑 (异步) ---
async def _process_stock_chunk_async(stock_pks):
    """
    异步处理单个股票片区的核心逻辑。
    注意：所有项目相关的导入都放在这个函数内部。
    """
    # --- 在任务执行时导入 ---
    from django.conf import settings
    from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO
    from dao_manager.daos.stock_basic_dao import StockBasicDAO # 替换为你的 DAO 实际路径
    from stock_models.stock_basic import StockBasic, StockTimeTrade # 替换为你的模型实际路径
    from utils.cash_key import StockCashKey # 替换为你的工具类实际路径
    # --- 结束导入 ---
    stock_indicators_dao = StockIndicatorsDAO()
    stock_basic_dao = StockBasicDAO()
    cache_limit = 233 * 2
    # TIME_LEVELS = ['Day'] # 测试时可以减少
    # 使用 sync_to_async 获取实际的 StockBasic 对象
    get_stocks_sync = sync_to_async(
        lambda pks: list(StockBasic.objects.filter(pk__in=pks)),
        thread_sensitive=True
    )
    stocks_in_chunk = await get_stocks_sync(stock_pks)
    if not stocks_in_chunk:
        logger.warning(f"任务接收到空的股票列表 (pks: {stock_pks})，跳过处理。")
        return
    logger.info(f"开始处理包含 {len(stocks_in_chunk)} 只股票的片区: {[s.stock_code for s in stocks_in_chunk]}")
    for stock in stocks_in_chunk:
        for time_level in TIME_TEADE_TIME_LEVELS_LITE:
            try:
                # 使用 sync_to_async 从数据库获取数据
                get_data_sync = sync_to_async(
                    lambda: list( # 将 QuerySet 转换为列表
                        StockTimeTrade.objects.filter(stock=stock, time_level=time_level)
                                            .order_by('-trade_time')[:cache_limit] # 使用切片
                    ),
                    thread_sensitive=True
                )
                datas = await get_data_sync()
                if not datas:
                    logger.debug(f"股票 {stock.stock_code} 在 {time_level} 级别没有数据，跳过缓存。")
                    continue
                cache_key = StockCashKey()
                cache_key_str = cache_key.history_time_trade(stock.stock_code, time_level)
                logger.info(f"重新缓存 {stock.stock_code} 股票 {time_level} 级别数据, 数量: {len(datas)}, key: {cache_key_str}, 最新时间: {datas[0].trade_time}")
                # 先清除旧的缓存数据
                await stock_indicators_dao.cache_manager.delete_cache(cache_key_str)
                logger.debug(f"已清除旧缓存 {cache_key_str}")
                # 批量处理缓存数据
                cache_data_batch = []
                for item in datas:
                    cache_data = stock_indicators_dao.data_format_process.set_time_trade_data(stock, time_level, item)
                    cache_data_batch.append(cache_data)
                # 批量设置缓存
                if cache_data_batch:
                    await stock_indicators_dao.cache_set.history_time_trade_batch(stock.stock_code, time_level, cache_data_batch)
                    logger.debug(f"已批量设置 {len(cache_data_batch)} 条缓存数据到 {cache_key_str}")
                # 修剪缓存至指定大小
                await stock_indicators_dao.cache_manager.trim_cache_zset(cache_key_str, cache_limit)
                logger.debug(f"已修剪缓存 {cache_key_str} 至 {cache_limit} 条记录")
            except Exception as e:
                logger.error(f"处理股票 {stock.stock_code} ({time_level} 级别) 时出错: {e}", exc_info=True) # 记录详细错误信息
    logger.info(f"完成处理股票片区: {[s.stock_code for s in stocks_in_chunk]}")

# --- Celery 任务 (同步包装器) ---
@shared_task(bind=True, name='stock_data_app.process_stock_chunk') # 使用明确的任务名称
def process_stock_chunk(self, stock_pks):
    """
    Celery 同步任务，负责调用异步处理函数。
    接收股票主键列表 (stock_pks)。
    """
    logger.info(f"Celery worker 接收到任务 {self.request.id}，处理股票 PKS: {stock_pks}")
    try:
        # 在同步的 Celery 任务中运行异步代码
        asyncio.run(_process_stock_chunk_async(stock_pks))
        logger.info(f"任务 {self.request.id} 成功完成。")
    except Exception as e:
        logger.error(f"任务 {self.request.id} (股票 PKS: {stock_pks}) 执行失败: {e}", exc_info=True)
        # 可以根据需要进行重试或其他错误处理
        # raise self.retry(exc=e, countdown=60) # 例如：60秒后重试
