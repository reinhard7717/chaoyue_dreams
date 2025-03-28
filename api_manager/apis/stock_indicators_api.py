import logging
from typing import Dict, List, Any, Optional, Union
from enum import Enum

from api_manager.base_api import BaseAPI


logger = logging.getLogger(__name__)

class TimeLevel(str, Enum):
    """股票分时级别枚举"""
    MIN_5 = "5"         # 5分钟
    MIN_15 = "15"       # 15分钟
    MIN_30 = "30"       # 30分钟
    MIN_60 = "60"       # 60分钟
    DAY = "Day"         # 日线
    DAY_QFQ = "Day_qfq" # 日线前复权
    DAY_HFQ = "Day_hfq" # 日线后复权
    WEEK = "Week"       # 周线
    WEEK_QFQ = "Week_qfq" # 周线前复权
    WEEK_HFQ = "Week_hfq" # 周线后复权
    MONTH = "Month"     # 月线
    MONTH_QFQ = "Month_qfq" # 月线前复权
    MONTH_HFQ = "Month_hfq" # 月线后复权
    YEAR = "Year"       # 年线
    YEAR_QFQ = "Year_qfq" # 年线前复权
    YEAR_HFQ = "Year_hfq" # 年线后复权


class StockIndicatorsAPI(BaseAPI):
    """
    股票分时和技术指标API
    提供股票分时交易数据和各类技术指标的API接口调用
    """
    
    async def get_time_trade(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Dict[str, Any]:
        """
        获取最新分时交易数据
        
        Args:
            stock_code: 股票代码
            time_level: 分时级别，可以是TimeLevel枚举或对应的字符串
            
        Returns:
            Dict[str, Any]: 最新分时交易数据
        """
        endpoint = f"/data/time/real/time/{stock_code}/{time_level}"
        logger.info(f"获取最新分时交易数据: {stock_code}, 级别: {time_level}")
        return await self.get(endpoint, expected_type='list')
    
    async def get_kdj(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Dict[str, Any]:
        """
        获取最新分时KDJ(9,3,3)指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 分时级别，可以是TimeLevel枚举或对应的字符串
            
        Returns:
            Dict[str, Any]: 最新KDJ指标数据
        """
        endpoint = f"/data/time/real/kdj/{stock_code}/{time_level}"
        logger.info(f"获取最新KDJ指标数据: {stock_code}, 级别: {time_level}")
        return await self.get(endpoint, expected_type='list')
    
    async def get_macd(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Dict[str, Any]:
        """
        获取最新分时MACD指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 分时级别，可以是TimeLevel枚举或对应的字符串
            
        Returns:
            Dict[str, Any]: 最新MACD指标数据
        """
        endpoint = f"/data/time/real/macd/{stock_code}/{time_level}"
        logger.info(f"获取最新MACD指标数据: {stock_code}, 级别: {time_level}")
        return await self.get(endpoint, expected_type='list')
    
    async def get_ma(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Dict[str, Any]:
        """
        获取最新分时MA指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 分时级别，可以是TimeLevel枚举或对应的字符串
            
        Returns:
            Dict[str, Any]: 最新MA指标数据
        """
        endpoint = f"/data/time/real/ma/{stock_code}/{time_level}"
        logger.info(f"获取最新MA指标数据: {stock_code}, 级别: {time_level}")
        return await self.get(endpoint, expected_type='list')
    
    async def get_boll(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Dict[str, Any]:
        """
        获取最新分时BOLL(20,2)指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 分时级别，可以是TimeLevel枚举或对应的字符串
            
        Returns:
            Dict[str, Any]: 最新BOLL指标数据
        """
        endpoint = f"/data/time/real/boll/{stock_code}/{time_level}"
        logger.info(f"获取最新BOLL指标数据: {stock_code}, 级别: {time_level}")
        return await self.get(endpoint, expected_type='list')
    
    async def get_history_trade(self, stock_code: str, time_level: Union[TimeLevel, str]) -> List[Dict[str, Any]]:
        """
        获取历史分时交易数据
        
        Args:
            stock_code: 股票代码
            time_level: 分时级别，可以是TimeLevel枚举或对应的字符串
            
        Returns:
            List[Dict[str, Any]]: 历史分时交易数据列表
        """
        endpoint = f"/data/time/history/trade/{stock_code}/{time_level}"
        logger.info(f"获取历史分时交易数据: {stock_code}, 级别: {time_level}")
        return await self.get(endpoint, expected_type='list')
    
    async def get_history_kdj(self, stock_code: str, time_level: Union[TimeLevel, str]) -> List[Dict[str, Any]]:
        """
        获取历史分时KDJ(9,3,3)指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 分时级别，可以是TimeLevel枚举或对应的字符串
            
        Returns:
            List[Dict[str, Any]]: 历史KDJ指标数据列表
        """
        endpoint = f"/data/time/history/kdj/{stock_code}/{time_level}"
        logger.info(f"获取历史KDJ指标数据: {stock_code}, 级别: {time_level}")
        return await self.get(endpoint, expected_type='list')
    
    async def get_history_macd(self, stock_code: str, time_level: Union[TimeLevel, str]) -> List[Dict[str, Any]]:
        """
        获取历史分时MACD指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 分时级别，可以是TimeLevel枚举或对应的字符串
            
        Returns:
            List[Dict[str, Any]]: 历史MACD指标数据列表
        """
        endpoint = f"/data/time/history/macd/{stock_code}/{time_level}"
        logger.info(f"获取历史MACD指标数据: {stock_code}, 级别: {time_level}")
        return await self.get(endpoint, expected_type='list')
    
    async def get_history_ma(self, stock_code: str, time_level: Union[TimeLevel, str]) -> List[Dict[str, Any]]:
        """
        获取历史分时MA指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 分时级别，可以是TimeLevel枚举或对应的字符串
            
        Returns:
            List[Dict[str, Any]]: 历史MA指标数据列表
        """
        endpoint = f"/data/time/history/ma/{stock_code}/{time_level}"
        logger.info(f"获取历史MA指标数据: {stock_code}, 级别: {time_level}")
        return await self.get(endpoint, expected_type='list')
    
    async def get_history_boll(self, stock_code: str, time_level: Union[TimeLevel, str]) -> List[Dict[str, Any]]:
        """
        获取历史分时BOLL(20,2)指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 分时级别，可以是TimeLevel枚举或对应的字符串
            
        Returns:
            List[Dict[str, Any]]: 历史BOLL指标数据列表
        """
        endpoint = f"/data/time/history/boll/{stock_code}/{time_level}"
        logger.info(f"获取历史BOLL指标数据: {stock_code}, 级别: {time_level}")
        return await self.get(endpoint, expected_type='list')
