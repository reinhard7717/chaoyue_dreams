import asyncio
import logging
from typing import List, Optional
import tushare as ts
from dao_manager.base_dao import BaseDAO
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from stock_models.stock_realtime import StockLevel5Data, StockRealtimeData
from utils.cache_get import StockInfoCacheGet, StockRealtimeCacheGet
from utils.cache_set import StockRealtimeCacheSet
from utils.data_format_process import StockRealtimeDataFormatProcess

logger = logging.getLogger("dao")

class StockRealtimeDAO(BaseDAO):
    """
    股票实时数据DAO，整合所有相关的实时数据访问功能
    """
    def __init__(self):
        """初始化StockRealtimeDAO"""
        super().__init__(None, None, 3600)  # 基类使用None作为model_class，因为本DAO管理多个模型
        self.stock_basic_dao = StockBasicInfoDao()
        self.data_format_process = StockRealtimeDataFormatProcess()
        self.cache_set = StockRealtimeCacheSet()  # 先实例化
        self.cache_get = StockRealtimeCacheGet()  # 先实例化
        self.stock_cache_get = StockInfoCacheGet()

    # ================= 实时盘口TICK快照(爬虫版) =================
    # 获取所有股票的实时盘口TICK快照数据并保存到数据库
    async def save_all_tick_data(self) -> Optional[StockRealtimeData]:
        """
        通过tushare获取实时盘口TICK快照数据并保存到数据库
        """
        ts.set_token('0793156bc63040ee46008f217c6e76c8b7c415e2748ac0a7bb509d2c')
        stocks = await self.stock_cache_get.all_stocks()
        # 先处理带后缀的stock_code
        stock_codes_list = [stock.stock_code for stock in stocks]
        # 每50个拼接成一个字符串
        grouped_stock_codes = [
            ','.join(stock_codes_list[i:i+50])
            for i in range(0, len(stock_codes_list), 50)
        ]
        for stock_codes in grouped_stock_codes:
            real_data_dicts = []
            level5_data_dicts = []
            # sina数据
            df = ts.realtime_quote(ts_code=stock_codes)
            for row in df.itertuples():
                stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                if stock:
                    real_dict = self.data_format_process.set_realtime_tick_data(stock, row)
                    level5_dict = self.data_format_process.set_level5_data(stock, row)
                    await self.cache_set.latest_realtime_data(stock.stock_code, real_dict)
                    real_data_dicts.append(real_dict)
                    level5_data_dicts.append(level5_dict)
            # 保存数据
            result = await self._save_all_to_db_native_upsert(
                model_class=StockRealtimeData,
                data_list=real_data_dicts,
                unique_fields=['stock', 'trade_time']
            )
            result = await self._save_all_to_db_native_upsert(
                model_class=StockLevel5Data,
                data_list=level5_data_dicts,
                unique_fields=['stock', 'trade_time']
            )
        return result

    # 根据传入的股票代码列表，获取实时盘口TICK快照数据并保存到数据库
    async def save_tick_data_by_stock_codes(self, stock_codes: List[str]) -> List:
        """
        【V2 - 高效版】通过tushare获取实时盘口TICK快照数据，并批量、并发地保存到数据库和缓存。
        """
        if not stock_codes:
            return []
        try:
            # 1. 一次性从Tushare获取所有股票的行情数据
            # 注意：建议将token初始化放在全局或应用启动时，而不是每次调用方法时设置
            # ts.set_token('YOUR_TOKEN_HERE') 
            stock_codes_str = ','.join(stock_codes)
            ts.set_token('0793156bc63040ee46008f217c6e76c8b7c415e2748ac0a7bb509d2c')
            df = ts.realtime_quote(ts_code=stock_codes_str)
            if df.empty:
                logger.warning(f"Tushare未返回股票 {stock_codes_str} 的实时行情数据。")
                return []
            # 2. 准备数据：一次性获取所有相关的StockInfo对象
            stocks_dict = await self.stock_basic_dao.get_stocks_by_codes(stock_codes)
            # 用于批量操作的容器
            realtime_data_list = []
            level5_data_list = []
            realtime_cache_payload = {}
            level5_cache_payload = {}
            # 3. 循环处理数据（仅在内存中，无IO操作）
            # print(f"调试信息: 开始在内存中处理 {len(df)} 条从Tushare返回的数据...")
            for row in df.itertuples():
                stock = stocks_dict.get(row.TS_CODE)
                if stock:
                    # 准备数据库数据
                    real_dict = self.data_format_process.set_realtime_tick_data(stock, row)
                    level5_dict = self.data_format_process.set_level5_data(stock, row)
                    realtime_data_list.append(real_dict)
                    level5_data_list.append(level5_dict)
                    # 【核心优化】准备缓存数据，而不是立即写入
                    realtime_cache_payload[row.TS_CODE] = real_dict
                    level5_cache_payload[row.TS_CODE] = level5_dict
                else:
                    logger.warning(f"在数据库中未找到股票代码 {row.TS_CODE} 的基本信息，已跳过。")

            if not realtime_data_list:
                logger.info("没有可处理的数据。")
                return []
            # 4. 并发执行所有IO密集型任务（数据库和缓存的批量写入）
            # print(f"调试信息: 准备并发执行 {len(realtime_data_list)} 条数据的数据库和缓存写入...")
            
            # 创建所有需要并发执行的异步任务
            tasks = [
                # 任务1: 批量写入实时行情到数据库
                self._save_all_to_db_native_upsert(
                    model_class=StockRealtimeData,
                    data_list=realtime_data_list,
                    unique_fields=['stock', 'trade_time']
                ),
                # 任务2: 批量写入Level5数据到数据库
                self._save_all_to_db_native_upsert(
                    model_class=StockLevel5Data,
                    data_list=level5_data_list,
                    unique_fields=['stock', 'trade_time']
                ),
                # 任务3: 批量写入实时行情到缓存
                self.cache_set.batch_set_latest_realtime_data(realtime_cache_payload),
                # 任务4: 批量写入Level5数据到缓存
                self.cache_set.batch_set_latest_level5_data(level5_cache_payload)
            ]
            
            # 使用 asyncio.gather 并发运行所有任务
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 检查执行结果
            db_realtime_result, db_level5_result, cache_realtime_result, cache_level5_result = results
            
            # 检查并记录任何在gather中发生的异常
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"并发任务 {i} 执行失败: {result}", exc_info=result)

            # print("调试信息: 所有数据库和缓存写入任务已完成。")
            
            # 返回数据库保存的结果，可以根据业务需求调整返回值
            # 这里我们返回第一个数据库操作的结果作为示例
            return db_realtime_result if not isinstance(db_realtime_result, Exception) else []

        except Exception as e:
            # 捕获Tushare API调用、数据库连接等其他异常
            print(f"save_tick_data_by_stock_codes 发生严重异常: {e}")
            logger.error(f"save_tick_data_by_stock_codes 发生严重异常: {e}", exc_info=True)
            return []

    async def get_latest_tick_data(self, stock_code: str) -> dict:
        """
        获取最新价格
        """
        # 从Redis缓存中获取数据
        data_dict = await self.cache_get.latest_tick_data(stock_code)
        if data_dict:
            change_percent = (data_dict.get('current_price') - data_dict.get('prev_close_price')) / data_dict.get('prev_close_price') * 100
            change_percent = round(change_percent, 2)  # 保留两位小数
            data_dict['change_percent'] = change_percent  # 计算涨跌幅（change_percent），并加入data_dict
            volume = data_dict.get('volume')
            volume = round(volume / 100, 2)
            data_dict['volume'] = volume
            return data_dict
        else:
            return None

    # ================= 实时成交快照(爬虫版) =================
    async def save_all_time_trade_data(self) -> None:
        """
        通过tushare获取所有股票的实时成交快照数据并保存到数据库
        """
        # 获取所有股票代码
        stocks = await self.stock_basic_dao.get_stock_list()
        pass

























