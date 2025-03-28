# api_manager/fund_flow_api.py

import logging
from typing import Dict, Any, List, Optional
from django.conf import settings

from api_manager.base_api import BaseAPI


logger = logging.getLogger(__name__)

class FundFlowAPI(BaseAPI):
    """
    资金流向API调用类
    
    处理与资金流向相关的API请求
    """
    
    async def get_fund_flow_trend(self, stock_code: str) -> List[Dict[str, Any]]:
        """
        获取资金走势对照数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[Dict[str, Any]]: 资金走势对照数据列表
        """
        endpoint = f"data/time/zijin/zlzjzs/{stock_code}"
        logger.info(f"正在获取股票[{stock_code}]的资金走势对照数据")
        try:
            result = await self.get(endpoint, expected_type='list')
            logger.info(f"成功获取股票[{stock_code}]的资金走势对照数据，共{len(result)}条")
            return result
        except Exception as e:
            logger.error(f"获取股票[{stock_code}]的资金走势对照数据失败: {str(e)}")
            raise
    
    async def get_capital_flow_trend(self, stock_code: str) -> List[Dict[str, Any]]:
        """
        获取资金流入趋势数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[Dict[str, Any]]: 资金流入趋势数据列表
        """
        endpoint = f"data/time/zijin/zjlrqs/{stock_code}"
        logger.info(f"正在获取股票[{stock_code}]的资金流入趋势数据")
        try:
            result = await self.get(endpoint, expected_type='list')
            logger.info(f"成功获取股票[{stock_code}]的资金流入趋势数据，共{len(result)}条")
            return result
        except Exception as e:
            logger.error(f"获取股票[{stock_code}]的资金流入趋势数据失败: {str(e)}")
            raise
    
    async def get_last10_capital_flow(self, stock_code: str) -> List[Dict[str, Any]]:
        """
        获取最近10天资金流入趋势数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[Dict[str, Any]]: 最近10天资金流入趋势数据列表
        """
        endpoint = f"data/time/zijin/zjlrqs/last10/{stock_code}"
        logger.info(f"正在获取股票[{stock_code}]的最近10天资金流入趋势数据")
        try:
            result = await self.get(endpoint, expected_type='list')
            logger.info(f"成功获取股票[{stock_code}]的最近10天资金流入趋势数据，共{len(result)}条")
            return result
        except Exception as e:
            logger.error(f"获取股票[{stock_code}]的最近10天资金流入趋势数据失败: {str(e)}")
            raise
    
    async def get_main_force_direction(self, stock_code: str) -> List[Dict[str, Any]]:
        """
        获取阶段主力动向数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[Dict[str, Any]]: 阶段主力动向数据列表
        """
        endpoint = f"data/time/zijin/jdzldx/{stock_code}"
        logger.info(f"正在获取股票[{stock_code}]的阶段主力动向数据")
        try:
            result = await self.get(endpoint, expected_type='list')
            logger.info(f"成功获取股票[{stock_code}]的阶段主力动向数据，共{len(result)}条")
            return result
        except Exception as e:
            logger.error(f"获取股票[{stock_code}]的阶段主力动向数据失败: {str(e)}")
            raise
    
    async def get_last10_main_force_direction(self, stock_code: str) -> List[Dict[str, Any]]:
        """
        获取最近10天阶段主力动向数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[Dict[str, Any]]: 最近10天阶段主力动向数据列表
        """
        endpoint = f"data/time/zijin/jdzldx/last10/{stock_code}"
        logger.info(f"正在获取股票[{stock_code}]的最近10天阶段主力动向数据")
        try:
            result = await self.get(endpoint, expected_type='list')
            logger.info(f"成功获取股票[{stock_code}]的最近10天阶段主力动向数据，共{len(result)}条")
            return result
        except Exception as e:
            logger.error(f"获取股票[{stock_code}]的最近10天阶段主力动向数据失败: {str(e)}")
            raise
    
    async def get_trading_distribution(self, stock_code: str) -> List[Dict[str, Any]]:
        """
        获取历史成交分布数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[Dict[str, Any]]: 历史成交分布数据列表
        """
        endpoint = f"data/time/zijin/lscjfb/{stock_code}"
        logger.info(f"正在获取股票[{stock_code}]的历史成交分布数据")
        try:
            result = await self.get(endpoint, expected_type='list')
            logger.info(f"成功获取股票[{stock_code}]的历史成交分布数据，共{len(result)}条")
            return result
        except Exception as e:
            logger.error(f"获取股票[{stock_code}]的历史成交分布数据失败: {str(e)}")
            raise
    
    async def get_last10_trading_distribution(self, stock_code: str) -> List[Dict[str, Any]]:
        """
        获取最近10天成交分布数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[Dict[str, Any]]: 最近10天成交分布数据列表
        """
        endpoint = f"data/time/zijin/lscjfb/last10/{stock_code}"
        logger.info(f"正在获取股票[{stock_code}]的最近10天成交分布数据")
        try:
            result = await self.get(endpoint, expected_type='list')
            logger.info(f"成功获取股票[{stock_code}]的最近10天成交分布数据，共{len(result)}条")
            return result
        except Exception as e:
            logger.error(f"获取股票[{stock_code}]的最近10天成交分布数据失败: {str(e)}")
            raise


class StockPoolAPI(BaseAPI):
    """
    股票池API调用类
    
    处理与各类股票池相关的API请求
    """
    
    async def get_limit_up_pool(self, date: str) -> List[Dict[str, Any]]:
        """
        获取涨停股池数据
        
        Args:
            date: 日期，格式yyyy-MM-dd
            
        Returns:
            List[Dict[str, Any]]: 涨停股池数据列表
        """
        endpoint = f"data/time/zdtgc/ztgc/{date}"
        logger.info(f"正在获取日期[{date}]的涨停股池数据")
        try:
            result = await self.get(endpoint, expected_type='list')
            logger.info(f"成功获取日期[{date}]的涨停股池数据，共{len(result)}条")
            return result
        except Exception as e:
            logger.error(f"获取日期[{date}]的涨停股池数据失败: {str(e)}")
            raise
    
    async def get_limit_down_pool(self, date: str) -> List[Dict[str, Any]]:
        """
        获取跌停股池数据
        
        Args:
            date: 日期，格式yyyy-MM-dd
            
        Returns:
            List[Dict[str, Any]]: 跌停股池数据列表
        """
        endpoint = f"data/time/zdtgc/dtgc/{date}"
        logger.info(f"正在获取日期[{date}]的跌停股池数据")
        try:
            result = await self.get(endpoint, expected_type='list')
            logger.info(f"成功获取日期[{date}]的跌停股池数据，共{len(result)}条")
            return result
        except Exception as e:
            logger.error(f"获取日期[{date}]的跌停股池数据失败: {str(e)}")
            raise
    
    async def get_strong_stock_pool(self, date: str) -> List[Dict[str, Any]]:
        """
        获取强势股池数据
        
        Args:
            date: 日期，格式yyyy-MM-dd
            
        Returns:
            List[Dict[str, Any]]: 强势股池数据列表
        """
        endpoint = f"data/time/zdtgc/qsgc/{date}"
        logger.info(f"正在获取日期[{date}]的强势股池数据")
        try:
            result = await self.get(endpoint, expected_type='list')
            logger.info(f"成功获取日期[{date}]的强势股池数据，共{len(result)}条")
            return result
        except Exception as e:
            logger.error(f"获取日期[{date}]的强势股池数据失败: {str(e)}")
            raise
    
    async def get_new_stock_pool(self, date: str) -> List[Dict[str, Any]]:
        """
        获取次新股池数据
        
        Args:
            date: 日期，格式yyyy-MM-dd
            
        Returns:
            List[Dict[str, Any]]: 次新股池数据列表
        """
        endpoint = f"data/time/zdtgc/cxgc/{date}"
        logger.info(f"正在获取日期[{date}]的次新股池数据")
        try:
            result = await self.get(endpoint, expected_type='list')
            logger.info(f"成功获取日期[{date}]的次新股池数据，共{len(result)}条")
            return result
        except Exception as e:
            logger.error(f"获取日期[{date}]的次新股池数据失败: {str(e)}")
            raise
    
    async def get_break_limit_pool(self, date: str) -> List[Dict[str, Any]]:
        """
        获取炸板股池数据
        
        Args:
            date: 日期，格式yyyy-MM-dd
            
        Returns:
            List[Dict[str, Any]]: 炸板股池数据列表
        """
        endpoint = f"data/time/zdtgc/zbgc/{date}"
        logger.info(f"正在获取日期[{date}]的炸板股池数据")
        try:
            result = await self.get(endpoint, expected_type='list')
            logger.info(f"成功获取日期[{date}]的炸板股池数据，共{len(result)}条")
            return result
        except Exception as e:
            logger.error(f"获取日期[{date}]的炸板股池数据失败: {str(e)}")
            raise
