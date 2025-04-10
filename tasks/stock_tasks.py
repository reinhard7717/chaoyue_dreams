from celery import shared_task
from chaoyue_dreams.celery import app as celery_app


@celery_app.task(bind=True, name='tasks.stock.fetch_stock_trade_data_task')
def fetch_stock_trade_data_task(batch_index=0, batch_size=50, total_batches=1):
    """股票交易数据获取任务 - 处理指定批次的股票"""
    # 导入包放在任务内部
    import logging
    import asyncio
    from asgiref.sync import sync_to_async
    from utils.cash_key import StockCashKey
    from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO
    from dao_manager.daos.stock_basic_dao import StockBasicDAO
    from stock_models.stock_basic import StockTimeTrade
    
    logger = logging.getLogger(__name__)
    
    # 将异步处理包装在一个函数中，以便在任务中调用
    async def process_batch():
        stock_indicators_dao = StockIndicatorsDAO()
        stock_basic_dao = StockBasicDAO()
        cache_limit = 233 * 3
        TIME_LEVELS = ['5','15','30','60','Day','Week','Month','Year']
        
        # 获取所有股票
        all_stocks = await stock_basic_dao.get_stock_list()
        
        # 计算当前批次应处理的股票
        start_idx = batch_index * batch_size
        end_idx = min(start_idx + batch_size, len(all_stocks))
        stocks = all_stocks[start_idx:end_idx]
        
        logger.info(f"批次 {batch_index+1}/{total_batches}: 开始处理 {len(stocks)} 只股票")
        
        for stock in stocks:
            for time_level in TIME_LEVELS:
                get_data_sync = sync_to_async(
                    lambda: list(
                        StockTimeTrade.objects.filter(stock=stock, time_level=time_level
                        ).order_by('-trade_time')[:cache_limit]
                    ),
                    thread_sensitive=True
                )
                datas = await get_data_sync()
                cache_key = StockCashKey()
                cache_key_str = cache_key.history_time_trade(stock.stock_code, time_level)
                
                logger.info(f"重新缓存{stock.stock_code}股票{time_level}级别历史分时成交数据, length: {len(datas)}, cache_key_str: {cache_key_str}")
                
                if datas:
                    for item in datas:
                        cache_data = stock_indicators_dao.data_format_process.set_time_trade_data(stock, time_level, item)
                        await stock_indicators_dao.cache_set.history_time_trade(stock.stock_code, time_level, cache_data)
                    
                    # 修剪缓存
                    await stock_indicators_dao.cache_manager.trim_cache_zset(cache_key_str, cache_limit)
    
    # 运行异步任务
    loop = asyncio.get_event_loop()
    loop.run_until_complete(process_batch())
    
    return f"批次 {batch_index+1}/{total_batches} 处理完成"

# 主命令方法修改
async def fetch_stock_trade_data_from_db(self):
    """从数据库获取股票交易数据 - 分片任务分发"""
    self.stdout.write('从数据库获取股票交易数据并分片...')
    
    # 导入放在方法内部
    from dao_manager.daos.stock_basic_dao import StockBasicDAO
    import math
    import logging
    
    logger = logging.getLogger(__name__)
    
    # 分片参数
    batch_size = 50  # 每批处理的股票数量
    
    # 获取所有股票
    stock_basic_dao = StockBasicDAO()
    stocks = await stock_basic_dao.get_stock_list()
    total_stocks = len(stocks)
    
    # 计算需要的批次数
    total_batches = math.ceil(total_stocks / batch_size)
    
    self.stdout.write(f"将 {total_stocks} 只股票分为 {total_batches} 个批次进行处理")
    
    # 创建并分发任务
    for batch_idx in range(total_batches):
        # 分发任务给 Celery
        fetch_stock_trade_data_task.delay(
            batch_index=batch_idx,
            batch_size=batch_size,
            total_batches=total_batches
        )
        
        self.stdout.write(f"已分发第 {batch_idx + 1}/{total_batches} 批股票数据处理任务")
    
    self.stdout.write(f"所有任务已分发完成，共 {total_batches} 个批次")
