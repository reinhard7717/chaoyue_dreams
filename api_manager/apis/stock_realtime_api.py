import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from api_manager.base_api import BaseAPI


logger = logging.getLogger(__name__)

class StockRealtimeAPI(BaseAPI):
    """
    股票实时数据API，实现与股票实时交易数据相关的接口调用
    """
    
    async def get_realtime_data(self, stock_code: str) -> Dict[str, Any]:
        """
        获取股票实时交易数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Dict[str, Any]: 实时交易数据
        """
        endpoint = f"/data/time/real/{stock_code}"
        # logger.info(f"获取实时交易数据: {stock_code}")
        return await self.get(endpoint, expected_type='list')
    
    async def get_level5_data(self, stock_code: str) -> Dict[str, Any]:
        """
        获取买卖五档盘口数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Dict[str, Any]: 买卖五档盘口数据
        """
        endpoint = f"/data/time/real/trace/level5/{stock_code}"
        # logger.info(f"获取买卖五档盘口数据: {stock_code}")
        return await self.get(endpoint, expected_type='list')
    
    async def get_onebyone_trades(self, stock_code: str) -> List[Dict[str, Any]]:
        """
        获取当天逐笔交易数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[Dict[str, Any]]: 当天逐笔交易数据列表
        """
        endpoint = f"/data/time/real/trace/onebyone/{stock_code}"
        # logger.info(f"获取当天逐笔交易数据: {stock_code}")
        return await self.get(endpoint, expected_type='list')
    
    async def get_time_deal(self, stock_code: str) -> List[Dict[str, Any]]:
        """
        获取当天分时成交数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[Dict[str, Any]]: 当天分时成交数据列表
        """
        endpoint = f"/data/time/real/trace/timedeal/{stock_code}"
        # logger.info(f"获取当天分时成交数据: {stock_code}")
        return await self.get(endpoint, expected_type='list')
    
    async def get_real_percent(self, stock_code: str) -> List[Dict[str, Any]]:
        """
        获取当天分价成交占比数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[Dict[str, Any]]: 当天分价成交占比数据列表
        """
        endpoint = f"/data/time/real/trace/realpercent/{stock_code}"
        # logger.info(f"获取当天分价成交占比数据: {stock_code}")
        return await self.get(endpoint, expected_type='list')
    
    async def get_big_deal(self, stock_code: str) -> List[Dict[str, Any]]:
        """
        获取当天逐笔大单交易数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[Dict[str, Any]]: 当天逐笔大单交易数据列表
        """
        endpoint = f"/data/time/real/trace/bigdeal/{stock_code}"
        # logger.info(f"获取当天逐笔大单交易数据: {stock_code}")
        return await self.get(endpoint, expected_type='list')
    
    async def get_abnormal_movements(self) -> List[Dict[str, Any]]:
        """
        获取盘中异动信息
        
        Returns:
            List[Dict[str, Any]]: 盘中异动信息列表
        """
        endpoint = "/data/all/pzyd"
        # logger.info("获取盘中异动信息")
        return await self.get(endpoint)
