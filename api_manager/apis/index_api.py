# api_manager/stock_index_api.py

import logging
import datetime
from typing import Dict, Any, List, Optional, Union
from django.conf import settings

from api_manager.base_api import BaseAPI

logger = logging.getLogger(__name__)

class StockIndexAPI(BaseAPI):
    """
    股票指数API调用类
    
    处理与股票指数相关的API请求
    """
    
    async def get_main_indexes(self) -> List[Dict[str, Any]]:
        """
        获取沪深主要指数列表
        
        Returns:
            List[Dict[str, Any]]: 包含主要指数信息的列表
        """
        endpoint = "data/base/shsz"
        logger.info("正在获取沪深主要指数列表")
        try:
            result = await self.get(endpoint, expected_type='list')
            logger.info(f"成功获取沪深主要指数列表，共{len(result)}条数据")
            return result
        except Exception as e:
            logger.error(f"获取沪深主要指数列表失败: {str(e)}")
            raise
    
    async def get_sh_indexes(self) -> List[Dict[str, Any]]:
        """
        获取沪市指数列表
        
        Returns:
            List[Dict[str, Any]]: 包含沪市指数信息的列表
        """
        endpoint = "data/base/sh"
        logger.info("正在获取沪市指数列表")
        try:
            result = await self.get(endpoint, expected_type='list')
            logger.info(f"成功获取沪市指数列表，共{len(result)}条数据")
            return result
        except Exception as e:
            logger.error(f"获取沪市指数列表失败: {str(e)}")
            raise
    
    async def get_sz_indexes(self) -> List[Dict[str, Any]]:
        """
        获取深市指数列表
        
        Returns:
            List[Dict[str, Any]]: 包含深市指数信息的列表
        """
        endpoint = "data/base/sz"
        logger.info("正在获取深市指数列表")
        try:
            result = await self.get(endpoint, expected_type='list')
            logger.info(f"成功获取深市指数列表，共{len(result)}条数据")
            return result
        except Exception as e:
            logger.error(f"获取深市指数列表失败: {str(e)}")
            raise
    
    async def get_index_realtime_data(self, index_code: str) -> Dict[str, Any]:
        """
        获取指数实时数据
        
        Args:
            index_code: 指数代码，需包含sh/sz前缀，如：sh000001
            
        Returns:
            Dict[str, Any]: 指数实时数据
        """
        endpoint = f"data/time/real/{index_code}"
        logger.info(f"正在获取指数[{index_code}]实时数据")
        try:
            result = await self.get(endpoint, expected_type='dict')
            logger.info(f"成功获取指数[{index_code}]实时数据")
            return result
        except Exception as e:
            logger.error(f"获取指数[{index_code}]实时数据失败: {str(e)}")
            raise
    
    async def get_market_overview(self) -> Dict[str, Any]:
        """
        获取沪深两市上涨下跌数概览
        
        Returns:
            Dict[str, Any]: 市场概览数据
        """
        endpoint = "data/time/real/shszzdbl"
        logger.info("正在获取沪深两市上涨下跌数概览")
        try:
            result = await self.get(endpoint, expected_type='dict')
            logger.info("成功获取沪深两市上涨下跌数概览")
            return result
        except Exception as e:
            logger.error(f"获取沪深两市上涨下跌数概览失败: {str(e)}")
            raise
    
    async def get_latest_time_series(self, index_code: str, time_level: str) -> Dict[str, Any]:
        """
        获取最新分时交易数据
        
        Args:
            index_code: 指数代码，需包含sh/sz前缀，如：sh000001
            time_level: 时间级别，可选值：5、15、30、60、Day、Week、Month、Year
            
        Returns:
            Dict[str, Any]: 最新分时交易数据
        """
        endpoint = f"data/time/real/time/{index_code}/{time_level}"
        logger.info(f"正在获取指数[{index_code}]的{time_level}级别最新分时交易数据")
        try:
            result = await self.get(endpoint, expected_type='list')
            logger.info(f"成功获取指数[{index_code}]的{time_level}级别最新分时交易数据")
            return result
        except Exception as e:
            logger.error(f"获取指数[{index_code}]的{time_level}级别最新分时交易数据失败: {str(e)}")
            raise
    
    async def get_history_time_series(self, index_code: str, time_level: str) -> List[Dict[str, Any]]:
        """
        获取历史分时交易数据
        
        Args:
            index_code: 指数代码，需包含sh/sz前缀，如：sh000001
            time_level: 时间级别，可选值：5、15、30、60、Day、Week、Month、Year
            
        Returns:
            List[Dict[str, Any]]: 历史分时交易数据列表
        """
        endpoint = f"data/time/history/trade/{index_code}/{time_level}"
        logger.info(f"正在获取指数[{index_code}]的{time_level}级别历史分时交易数据")
        try:
            result = await self.get(endpoint, expected_type='list')
            logger.info(f"成功获取指数[{index_code}]的{time_level}级别历史分时交易数据，共{len(result)}条")
            return result
        except Exception as e:
            logger.error(f"获取指数[{index_code}]的{time_level}级别历史分时交易数据失败: {str(e)}")
            raise
    
    # ================ KDJ指标相关API ================
    
    async def get_latest_kdj(self, index_code: str, time_level: str) -> Dict[str, Any]:
        """
        获取最新KDJ指标数据
        
        Args:
            index_code: 指数代码，需包含sh/sz前缀，如：sh000001
            time_level: 时间级别，可选值：5、15、30、60、Day、Week、Month、Year
            
        Returns:
            Dict[str, Any]: 最新KDJ指标数据
        """
        endpoint = f"data/time/real/kdj/{index_code}/{time_level}"
        logger.info(f"正在获取指数[{index_code}]的{time_level}级别最新KDJ数据")
        try:
            result = await self.get(endpoint, expected_type='list')
            logger.info(f"成功获取指数[{index_code}]的{time_level}级别最新KDJ数据")
            return result
        except Exception as e:
            logger.error(f"获取指数[{index_code}]的{time_level}级别最新KDJ数据失败: {str(e)}")
            raise
    
    async def get_history_kdj(self, index_code: str, time_level: str) -> List[Dict[str, Any]]:
        """
        获取历史KDJ指标数据
        
        Args:
            index_code: 指数代码，需包含sh/sz前缀，如：sh000001
            time_level: 时间级别，可选值：5、15、30、60、Day、Week、Month、Year
            
        Returns:
            List[Dict[str, Any]]: 历史KDJ指标数据列表
        """
        endpoint = f"data/time/history/kdj/{index_code}/{time_level}"
        logger.info(f"正在获取指数[{index_code}]的{time_level}级别历史KDJ数据")
        try:
            result = await self.get(endpoint, expected_type='list')
            logger.info(f"成功获取指数[{index_code}]的{time_level}级别历史KDJ数据，共{len(result)}条")
            return result
        except Exception as e:
            logger.error(f"获取指数[{index_code}]的{time_level}级别历史KDJ数据失败: {str(e)}")
            raise
    
    # ================ MACD指标相关API ================
    
    async def get_latest_macd(self, index_code: str, time_level: str) -> Dict[str, Any]:
        """
        获取最新MACD指标数据
        
        Args:
            index_code: 指数代码，需包含sh/sz前缀，如：sh000001
            time_level: 时间级别，可选值：5、15、30、60、Day、Week、Month、Year
            
        Returns:
            Dict[str, Any]: 最新MACD指标数据
        """
        endpoint = f"data/time/real/macd/{index_code}/{time_level}"
        logger.info(f"正在获取指数[{index_code}]的{time_level}级别最新MACD数据")
        try:
            result = await self.get(endpoint, expected_type='list')
            logger.info(f"成功获取指数[{index_code}]的{time_level}级别最新MACD数据")
            return result
        except Exception as e:
            logger.error(f"获取指数[{index_code}]的{time_level}级别最新MACD数据失败: {str(e)}")
            raise
    
    async def get_history_macd(self, index_code: str, time_level: str) -> List[Dict[str, Any]]:
        """
        获取历史MACD指标数据
        
        Args:
            index_code: 指数代码，需包含sh/sz前缀，如：sh000001
            time_level: 时间级别，可选值：5、15、30、60、Day、Week、Month、Year
            
        Returns:
            List[Dict[str, Any]]: 历史MACD指标数据列表
        """
        endpoint = f"data/time/history/macd/{index_code}/{time_level}"
        logger.info(f"正在获取指数[{index_code}]的{time_level}级别历史MACD数据")
        try:
            result = await self.get(endpoint, expected_type='list')
            logger.info(f"成功获取指数[{index_code}]的{time_level}级别历史MACD数据，共{len(result)}条")
            return result
        except Exception as e:
            logger.error(f"获取指数[{index_code}]的{time_level}级别历史MACD数据失败: {str(e)}")
            raise
    
    # ================ MA指标相关API ================
    
    async def get_latest_ma(self, index_code: str, time_level: str) -> Dict[str, Any]:
        """
        获取最新MA指标数据
        
        Args:
            index_code: 指数代码，需包含sh/sz前缀，如：sh000001
            time_level: 时间级别，可选值：5、15、30、60、Day、Week、Month、Year
            
        Returns:
            Dict[str, Any]: 最新MA指标数据
        """
        endpoint = f"data/time/real/ma/{index_code}/{time_level}"
        logger.info(f"正在获取指数[{index_code}]的{time_level}级别最新MA数据")
        try:
            result = await self.get(endpoint, expected_type='list')
            logger.info(f"成功获取指数[{index_code}]的{time_level}级别最新MA数据")
            return result
        except Exception as e:
            logger.error(f"获取指数[{index_code}]的{time_level}级别最新MA数据失败: {str(e)}")
            raise
    
    async def get_history_ma(self, index_code: str, time_level: str) -> List[Dict[str, Any]]:
        """
        获取历史MA指标数据
        
        Args:
            index_code: 指数代码，需包含sh/sz前缀，如：sh000001
            time_level: 时间级别，可选值：5、15、30、60、Day、Week、Month、Year
            
        Returns:
            List[Dict[str, Any]]: 历史MA指标数据列表
        """
        endpoint = f"data/time/history/ma/{index_code}/{time_level}"
        logger.info(f"正在获取指数[{index_code}]的{time_level}级别历史MA数据")
        try:
            result = await self.get(endpoint, expected_type='list')
            logger.info(f"成功获取指数[{index_code}]的{time_level}级别历史MA数据，共{len(result)}条")
            return result
        except Exception as e:
            logger.error(f"获取指数[{index_code}]的{time_level}级别历史MA数据失败: {str(e)}")
            raise
    
    # ================ BOLL指标相关API ================
    
    async def get_latest_boll(self, index_code: str, time_level: str) -> Dict[str, Any]:
        """
        获取最新BOLL指标数据
        
        Args:
            index_code: 指数代码，需包含sh/sz前缀，如：sh000001
            time_level: 时间级别，可选值：5、15、30、60、Day、Week、Month、Year
            
        Returns:
            Dict[str, Any]: 最新BOLL指标数据
        """
        endpoint = f"data/time/real/boll/{index_code}/{time_level}"
        logger.info(f"正在获取指数[{index_code}]的{time_level}级别最新BOLL数据")
        try:
            result = await self.get(endpoint, expected_type='list')
            logger.info(f"成功获取指数[{index_code}]的{time_level}级别最新BOLL数据")
            return result
        except Exception as e:
            logger.error(f"获取指数[{index_code}]的{time_level}级别最新BOLL数据失败: {str(e)}")
            raise
    
    async def get_history_boll(self, index_code: str, time_level: str) -> List[Dict[str, Any]]:
        """
        获取历史BOLL指标数据
        
        Args:
            index_code: 指数代码，需包含sh/sz前缀，如：sh000001
            time_level: 时间级别，可选值：5、15、30、60、Day、Week、Month、Year
            
        Returns:
            List[Dict[str, Any]]: 历史BOLL指标数据列表
        """
        endpoint = f"data/time/history/boll/{index_code}/{time_level}"
        logger.info(f"正在获取指数[{index_code}]的{time_level}级别历史BOLL数据")
        try:
            result = await self.get(endpoint, expected_type='list')
            logger.info(f"成功获取指数[{index_code}]的{time_level}级别历史BOLL数据，共{len(result)}条")
            return result
        except Exception as e:
            logger.error(f"获取指数[{index_code}]的{time_level}级别历史BOLL数据失败: {str(e)}")
            raise
