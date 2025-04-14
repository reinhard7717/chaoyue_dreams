# dao_manager\daos\strategies_dao.py

from typing import List
from dao_manager.base_dao import BaseDAO

class StrategiesDAO(BaseDAO):
    def __init__(self):
        super().__init__()
        from utils.data_format_process import StrategiesDataFormatProcess
        self.data_format_process = StrategiesDataFormatProcess()

    # 新增 close 方法
    async def close(self):
        """关闭内部持有的 API Client Session"""
        if hasattr(self, 'api') and self.api:
            await self.api.close() # 调用 StockRealtimeAPI 的 close 方法
        else:
            pass


    async def get_latest_strategies(self, stock_code: str) -> List[dict]:
        """
        根据股票代码获取最新策略
        Args:
            stock_code: 股票代码
        Returns:
            List[dict]: 策略列表
        """
        realtime_data = None
        # 从缓存获取
        cache_data = await self.cache_get.get_strategies(stock_code)
        if cache_data:
            realtime_data = self.data_format_process.set_strategies_data(cache_data)
        # 从数据库获取
        sql_data = await self.get_items_by_field('stock_code', stock_code)
        if sql_data:
            return sql_data
    
    async def get_strategies_by_user_id(self, user_id: int) -> List[dict]:
        return await self.get_items_by_field('user_id', user_id)
