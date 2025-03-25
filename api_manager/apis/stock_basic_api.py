# api_manager/apis/stock_basic.py

import logging
from typing import Dict, List, Any, Optional

from api_manager.base_api import BaseAPI


logger = logging.getLogger(__name__)

class StockBasicAPI(BaseAPI):
    """
    股票基础信息API，实现与股票基础数据相关的接口调用
    """
    
    async def get_stock_list(self) -> List[Dict[str, str]]:
        """
        获取股票列表
        
        Returns:
            List[Dict[str, str]]: 股票列表，包含代码、名称和交易所信息
        """
        endpoint = "/data/base/gplist"
        logger.info("获取股票列表")
        return await self.get(endpoint)
    
    async def get_new_stock_calendar(self) -> List[Dict[str, Any]]:
        """
        获取新股日历
        
        Returns:
            List[Dict[str, Any]]: 新股日历，按申购日期倒序
        """
        endpoint = "/data/all/xgrl"
        logger.info("获取新股日历")
        return await self.get(endpoint)
    
    async def get_st_stock_list(self) -> List[Dict[str, str]]:
        """
        获取风险警示（ST）股票列表
        
        Returns:
            List[Dict[str, str]]: ST股票列表，包含代码、名称和交易所信息
        """
        endpoint = "/data/all/stgplist"
        logger.info("获取风险警示股票列表")
        return await self.get(endpoint)
    
    async def get_company_info(self, stock_code: str) -> Dict[str, str]:
        """
        获取公司简介
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Dict[str, str]: 公司基本信息
        """
        endpoint = f"/data/time/f10/info/{stock_code}"
        logger.info(f"获取公司简介: {stock_code}")
        return await self.get(endpoint)
    
    async def get_company_index(self, stock_code: str) -> List[Dict[str, str]]:
        """
        获取所属指数
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[Dict[str, str]]: 公司所属指数列表
        """
        endpoint = f"/data/time/f10/index/{stock_code}"
        logger.info(f"获取所属指数: {stock_code}")
        return await self.get(endpoint)
    
    async def get_quarterly_profit(self, stock_code: str) -> List[Dict[str, str]]:
        """
        获取近一年各季度利润
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[Dict[str, str]]: 近一年各季度利润数据
        """
        endpoint = f"/data/time/f10/pf/{stock_code}"
        logger.info(f"获取季度利润: {stock_code}")
        return await self.get(endpoint)
    
    async def get_cash_flow(self, stock_code: str) -> List[Dict[str, str]]:
        """
        获取近一年各季度现金流
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[Dict[str, str]]: 近一年各季度现金流数据
        """
        endpoint = f"/data/time/f10/cf/{stock_code}"
        logger.info(f"获取季度现金流: {stock_code}")
        return await self.get(endpoint)
    
    async def get_earnings_forecast(self, stock_code: str) -> List[Dict[str, str]]:
        """
        获取近年业绩预告
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[Dict[str, str]]: 近年业绩预告数据
        """
        endpoint = f"/data/time/f10/ep/{stock_code}"
        logger.info(f"获取业绩预告: {stock_code}")
        return await self.get(endpoint)
    
    async def get_financial_indicators(self, stock_code: str) -> List[Dict[str, str]]:
        """
        获取财务指标
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[Dict[str, str]]: 财务指标数据
        """
        endpoint = f"/data/time/f10/fi/{stock_code}"
        logger.info(f"获取财务指标: {stock_code}")
        return await self.get(endpoint)
    
    async def get_major_shareholders(self, stock_code: str) -> List[Dict[str, Any]]:
        """
        获取十大股东
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[Dict[str, Any]]: 十大股东数据
        """
        endpoint = f"/data/time/f10/zygd/{stock_code}"
        logger.info(f"获取十大股东: {stock_code}")
        return await self.get(endpoint)
    
    async def get_major_floating_shareholders(self, stock_code: str) -> List[Dict[str, Any]]:
        """
        获取十大流通股东
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[Dict[str, Any]]: 十大流通股东数据
        """
        endpoint = f"/data/time/f10/zygdlt/{stock_code}"
        logger.info(f"获取十大流通股东: {stock_code}")
        return await self.get(endpoint)
    
    async def get_shareholder_changes(self, stock_code: str) -> List[Dict[str, str]]:
        """
        获取股东变化趋势
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[Dict[str, str]]: 股东变化趋势数据
        """
        endpoint = f"/data/time/f10/gdbh/{stock_code}"
        logger.info(f"获取股东变化趋势: {stock_code}")
        return await self.get(endpoint)
    
    async def get_fund_holdings(self, stock_code: str) -> List[Dict[str, Any]]:
        """
        获取基金持股
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[Dict[str, Any]]: 基金持股数据
        """
        endpoint = f"/data/time/f10/jjcg/{stock_code}"
        logger.info(f"获取基金持股: {stock_code}")
        return await self.get(endpoint)
    
    async def get_industry_category(self, stock_code: str) -> List[Dict[str, str]]:
        """
        获取所属板块
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[Dict[str, str]]: 所属板块数据
        """
        endpoint = f"/data/time/f10/ssbk/{stock_code}"
        logger.info(f"获取所属板块: {stock_code}")
        return await self.get(endpoint)
    
    async def get_business_scope(self, stock_code: str) -> List[Dict[str, str]]:
        """
        获取经营范围
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[Dict[str, str]]: 经营范围数据
        """
        endpoint = f"/data/time/f10/jyfw/{stock_code}"
        logger.info(f"获取经营范围: {stock_code}")
        return await self.get(endpoint)
    
    async def get_main_business(self, stock_code: str) -> List[Dict[str, str]]:
        """
        获取主营业务
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[Dict[str, str]]: 主营业务数据
        """
        endpoint = f"/data/time/f10/zyyw/{stock_code}"
        logger.info(f"获取主营业务: {stock_code}")
        return await self.get(endpoint)
    
    async def get_index_tree(self) -> List[Dict[str, Any]]:
        """
        获取指数、行业、概念树
        
        Returns:
            List[Dict[str, Any]]: 指数、行业、概念树数据
        """
        endpoint = "/data/base/it"
        logger.info("获取指数、行业、概念树")
        return await self.get(endpoint)
    
    async def get_stocks_by_index(self, index_code: str) -> List[Dict[str, str]]:
        """
        根据指数、行业、概念找相关股票
        
        Args:
            index_code: 指数、行业、概念代码
            
        Returns:
            List[Dict[str, str]]: 相关股票列表
        """
        endpoint = f"/data/time/indextree/{index_code}"
        logger.info(f"获取相关股票: {index_code}")
        return await self.get(endpoint)
    
    async def get_indexes_by_stock(self, stock_code: str) -> List[Dict[str, str]]:
        """
        根据股票找相关指数、行业、概念
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[Dict[str, str]]: 相关指数、行业、概念列表
        """
        endpoint = f"/data/time/iii/{stock_code}"
        logger.info(f"获取相关指数行业概念: {stock_code}")
        return await self.get(endpoint)
    
    async def get_margin_trading_stocks(self) -> List[Dict[str, str]]:
        """
        获取融资融券标的股
        
        Returns:
            List[Dict[str, str]]: 融资融券标的股列表
        """
        endpoint = "/data/base/rzrqGpList"
        logger.info("获取融资融券标的股")
        return await self.get(endpoint)
