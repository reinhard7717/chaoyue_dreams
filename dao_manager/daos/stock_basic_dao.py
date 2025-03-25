import logging
import asyncio
from datetime import date
from typing import Dict, List, Any, Optional, Tuple, Union

from django.db import transaction
from django.core.cache import cache

from api_manager.apis.stock_basic_api import StockBasicAPI
from api_manager.mappings.stock_basic import COMPANY_INFO_MAPPING, NEW_STOCK_CALENDAR_MAPPING, ST_STOCK_LIST_MAPPING, STOCK_BASIC_MAPPING
from dao_manager.base_dao import BaseDAO
from models.stock_basic import CompanyInfo, NewStockCalendar, STStockList, StockBasic

logger = logging.getLogger(__name__)

class StockBasicDAO(BaseDAO[StockBasic]):
    """
    股票基础信息DAO，负责股票基本信息的数据访问
    实现三层数据访问结构：Redis缓存 -> MySQL数据库 -> 外部API
    """
    
    def __init__(self):
        """初始化StockBasicDAO"""
        super().__init__(StockBasic, "stock_basic")
        self.api = StockBasicAPI()
        logger.info("初始化StockBasicDAO")
    
    async def get_by_code(self, stock_code: str) -> Optional[StockBasic]:
        """
        根据股票代码获取股票信息
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Optional[StockBasic]: 股票信息，如不存在则返回None
        """
        # 1. 首先尝试从缓存获取
        cache_key = f"stock:{stock_code}"
        cached_data = await self.get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 2. 缓存未命中，从数据库查询
        try:
            stock = await StockBasic.objects.filter(dm=stock_code).afirst()
            if stock:
                # 找到数据，存入缓存并返回
                await self.set_to_cache(cache_key, stock)
                return stock
        except Exception as e:
            logger.error(f"从数据库获取股票信息出错: {e}")
        
        # 3. 数据库未找到，从API获取
        try:
            stocks_data = await self.api.get_stock_list()
            
            # 处理API返回的数据，找到目标股票
            target_stock = None
            for stock_data in stocks_data:
                if stock_data.get('dm') == stock_code:
                    target_stock = stock_data
                    break
            
            if target_stock:
                # 映射字段并保存到数据库
                stock_data = self._map_api_to_model(target_stock, STOCK_BASIC_MAPPING)
                stock = await self._save_to_db(stock_data)
                
                # 保存到缓存
                await self.set_to_cache(cache_key, stock)
                return stock
        except Exception as e:
            logger.error(f"从API获取股票信息出错: {e}")
        
        return None
    
    async def get_all_stocks(self) -> List[StockBasic]:
        """
        获取所有股票信息
        
        Returns:
            List[StockBasic]: 所有股票信息列表
        """
        # 1. 首先尝试从缓存获取
        cache_key = "all_stocks"
        cached_data = await self.get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 2. 缓存未命中，从数据库查询
        try:
            stocks = await StockBasic.objects.all()
            if stocks:
                # 找到数据，存入缓存并返回
                await self.set_to_cache(cache_key, list(stocks))
                return stocks
        except Exception as e:
            logger.error(f"从数据库获取所有股票信息出错: {e}")
        
        # 3. 数据库未找到或为空，从API获取
        try:
            stocks_data = await self.api.get_stock_list()
            
            # 批量保存所有股票信息
            stocks = []
            for stock_data in stocks_data:
                mapped_data = self._map_api_to_model(stock_data, STOCK_BASIC_MAPPING)
                stock = await self._save_to_db(mapped_data)
                stocks.append(stock)
            
            # 保存到缓存
            await self.set_to_cache(cache_key, stocks)
            return stocks
        except Exception as e:
            logger.error(f"从API获取所有股票信息出错: {e}")
        
        return []
    
    async def refresh_stock_data(self, stock_code: Optional[str] = None) -> bool:
        """
        强制刷新股票数据（从API重新获取）
        
        Args:
            stock_code: 股票代码，如不指定则刷新所有股票
            
        Returns:
            bool: 是否成功刷新
        """
        try:
            if stock_code:
                # 刷新单个股票
                cache_key = f"stock:{stock_code}"
                await self.delete_from_cache(cache_key)
                
                # 从API获取最新数据
                stocks_data = await self.api.get_stock_list()
                
                for stock_data in stocks_data:
                    if stock_data.get('dm') == stock_code:
                        mapped_data = self._map_api_to_model(stock_data, STOCK_BASIC_MAPPING)
                        await self._update_db(stock_code, mapped_data)
                        logger.info(f"成功刷新股票数据: {stock_code}")
                        return True
                
                logger.warning(f"API中未找到股票: {stock_code}")
                return False
            else:
                # 刷新所有股票
                await self.delete_from_cache("all_stocks")
                await self.clear_cache_pattern("stock:*")
                
                # 从API获取最新数据
                stocks_data = await self.api.get_stock_list()
                
                # 批量更新所有股票
                for stock_data in stocks_data:
                    stock_code = stock_data.get('dm')
                    mapped_data = self._map_api_to_model(stock_data, STOCK_BASIC_MAPPING)
                    await self._update_db(stock_code, mapped_data)
                
                logger.info(f"成功刷新所有股票数据，共{len(stocks_data)}条")
                return True
        except Exception as e:
            logger.error(f"刷新股票数据出错: {e}")
            return False
    
    def _map_api_to_model(self, api_data: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        将API返回的数据映射到模型字段
        
        Args:
            api_data: API返回的数据
            mapping: 字段映射关系
            
        Returns:
            Dict[str, Any]: 映射后的数据
        """
        return {
            model_field: api_data.get(api_field)
            for api_field, model_field in mapping.items()
            if api_field in api_data and api_data.get(api_field) is not None
        }
    
    @transaction.atomic
    async def _save_to_db(self, data: Dict[str, Any]) -> StockBasic:
        """
        保存数据到数据库
        
        Args:
            data: 模型数据
            
        Returns:
            StockBasic: 保存后的模型实例
        """
        # 使用Django的ORM创建或更新记录
        try:
            stock, created = await StockBasic.objects.aupdate_or_create(
                dm=data['dm'],
                defaults=data
            )
            if created:
                logger.debug(f"创建新股票记录: {data['dm']}")
            else:
                logger.debug(f"更新股票记录: {data['dm']}")
            return stock
        except Exception as e:
            logger.error(f"保存股票数据出错: {e}")
            raise
    
    @transaction.atomic
    async def _update_db(self, stock_code: str, data: Dict[str, Any]) -> Optional[StockBasic]:
        """
        更新数据库中的股票信息
        
        Args:
            stock_code: 股票代码
            data: 更新的数据
            
        Returns:
            Optional[StockBasic]: 更新后的股票信息，如不存在则返回None
        """
        try:
            stock = await StockBasic.objects.filter(dm=stock_code).afirst()
            if stock:
                # 更新股票信息
                for field, value in data.items():
                    setattr(stock, field, value)
                await stock.asave()
                logger.debug(f"更新股票记录: {stock_code}")
                return stock
            else:
                # 股票不存在，创建新记录
                stock = StockBasic(**data)
                await stock.asave()
                logger.debug(f"创建新股票记录: {stock_code}")
                return stock
        except Exception as e:
            logger.error(f"更新股票数据出错: {e}")
            return None


class NewStockCalendarDAO(BaseDAO[NewStockCalendar]):
    """
    新股日历DAO，负责新股日历数据的访问
    """
    
    def __init__(self):
        """初始化NewStockCalendarDAO"""
        super().__init__(NewStockCalendar, "new_stock_calendar")
        self.api = StockBasicAPI()
        logger.info("初始化NewStockCalendarDAO")
    
    async def get_by_code(self, stock_code: str) -> Optional[NewStockCalendar]:
        """
        根据股票代码获取新股日历信息
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Optional[NewStockCalendar]: 新股日历信息，如不存在则返回None
        """
        # 1. 首先尝试从缓存获取
        cache_key = f"new_stock:{stock_code}"
        cached_data = await self.get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 2. 缓存未命中，从数据库查询
        try:
            new_stock = await NewStockCalendar.objects.filter(zqdm=stock_code).afirst()
            if new_stock:
                # 找到数据，存入缓存并返回
                await self.set_to_cache(cache_key, new_stock)
                return new_stock
        except Exception as e:
            logger.error(f"从数据库获取新股日历出错: {e}")
        
        # 3. 数据库未找到，从API获取
        try:
            new_stocks_data = await self.api.get_new_stock_calendar()
            
            # 处理API返回的数据，找到目标新股
            target_new_stock = None
            for new_stock_data in new_stocks_data:
                if new_stock_data.get('zqdm') == stock_code:
                    target_new_stock = new_stock_data
                    break
            
            if target_new_stock:
                # 处理日期字段
                self._process_date_fields(target_new_stock)
                
                # 映射字段并保存到数据库
                mapped_data = self._map_api_to_model(target_new_stock, NEW_STOCK_CALENDAR_MAPPING)
                new_stock = await self._save_to_db(mapped_data)
                
                # 保存到缓存
                await self.set_to_cache(cache_key, new_stock)
                return new_stock
        except Exception as e:
            logger.error(f"从API获取新股日历出错: {e}")
        
        return None
    
    async def get_all_new_stocks(self) -> List[NewStockCalendar]:
        """
        获取所有新股日历信息
        
        Returns:
            List[NewStockCalendar]: 所有新股日历信息列表
        """
        # 1. 首先尝试从缓存获取
        cache_key = "all_new_stocks"
        cached_data = await self.get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 2. 缓存未命中，从数据库查询
        try:
            new_stocks = await NewStockCalendar.objects.all()
            if new_stocks:
                # 找到数据，存入缓存并返回
                await self.set_to_cache(cache_key, list(new_stocks))
                return new_stocks
        except Exception as e:
            logger.error(f"从数据库获取所有新股日历出错: {e}")
        
        # 3. 数据库未找到或为空，从API获取
        try:
            new_stocks_data = await self.api.get_new_stock_calendar()
            
            # 批量保存所有新股日历信息
            new_stocks = []
            for new_stock_data in new_stocks_data:
                # 处理日期字段
                self._process_date_fields(new_stock_data)
                
                mapped_data = self._map_api_to_model(new_stock_data, NEW_STOCK_CALENDAR_MAPPING)
                new_stock = await self._save_to_db(mapped_data)
                new_stocks.append(new_stock)
            
            # 保存到缓存
            await self.set_to_cache(cache_key, new_stocks)
            return new_stocks
        except Exception as e:
            logger.error(f"从API获取所有新股日历出错: {e}")
        
        return []
    
    async def refresh_new_stock_data(self, stock_code: Optional[str] = None) -> bool:
        """
        强制刷新新股日历数据（从API重新获取）
        
        Args:
            stock_code: 股票代码，如不指定则刷新所有新股
            
        Returns:
            bool: 是否成功刷新
        """
        try:
            if stock_code:
                # 刷新单个新股
                cache_key = f"new_stock:{stock_code}"
                await self.delete_from_cache(cache_key)
                
                # 从API获取最新数据
                new_stocks_data = await self.api.get_new_stock_calendar()
                
                for new_stock_data in new_stocks_data:
                    if new_stock_data.get('zqdm') == stock_code:
                        # 处理日期字段
                        self._process_date_fields(new_stock_data)
                        
                        mapped_data = self._map_api_to_model(new_stock_data, NEW_STOCK_CALENDAR_MAPPING)
                        await self._update_db(stock_code, mapped_data)
                        logger.info(f"成功刷新新股日历数据: {stock_code}")
                        return True
                
                logger.warning(f"API中未找到新股: {stock_code}")
                return False
            else:
                # 刷新所有新股
                await self.delete_from_cache("all_new_stocks")
                await self.clear_cache_pattern("new_stock:*")
                
                # 从API获取最新数据
                new_stocks_data = await self.api.get_new_stock_calendar()
                
                # 批量更新所有新股
                for new_stock_data in new_stocks_data:
                    stock_code = new_stock_data.get('zqdm')
                    # 处理日期字段
                    self._process_date_fields(new_stock_data)
                    
                    mapped_data = self._map_api_to_model(new_stock_data, NEW_STOCK_CALENDAR_MAPPING)
                    await self._update_db(stock_code, mapped_data)
                
                logger.info(f"成功刷新所有新股日历数据，共{len(new_stocks_data)}条")
                return True
        except Exception as e:
            logger.error(f"刷新新股日历数据出错: {e}")
            return False
    
    def _process_date_fields(self, data: Dict[str, Any]) -> None:
        """
        处理日期字段
        
        Args:
            data: 包含日期字段的数据字典
        """
        date_fields = ['sgrq', 'zqgbrq', 'zqjkrq', 'ssrq']
        for field in date_fields:
            if field in data and data[field] and data[field] != 'null':
                try:
                    data[field] = date.fromisoformat(data[field])
                except (ValueError, TypeError):
                    logger.warning(f"日期字段处理失败: {field}={data[field]}")
                    data[field] = None
    
    def _map_api_to_model(self, api_data: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        将API返回的数据映射到模型字段
        
        Args:
            api_data: API返回的数据
            mapping: 字段映射关系
            
        Returns:
            Dict[str, Any]: 映射后的数据
        """
        return {
            model_field: api_data.get(api_field)
            for api_field, model_field in mapping.items()
            if api_field in api_data and api_data.get(api_field) is not None
        }
    
    @transaction.atomic
    async def _save_to_db(self, data: Dict[str, Any]) -> NewStockCalendar:
        """
        保存数据到数据库
        
        Args:
            data: 模型数据
            
        Returns:
            NewStockCalendar: 保存后的模型实例
        """
        try:
            new_stock, created = await NewStockCalendar.objects.aupdate_or_create(
                zqdm=data['zqdm'],
                defaults=data
            )
            if created:
                logger.debug(f"创建新股日历记录: {data['zqdm']}")
            else:
                logger.debug(f"更新新股日历记录: {data['zqdm']}")
            return new_stock
        except Exception as e:
            logger.error(f"保存新股日历数据出错: {e}")
            raise
    
    @transaction.atomic
    async def _update_db(self, stock_code: str, data: Dict[str, Any]) -> Optional[NewStockCalendar]:
        """
        更新数据库中的新股日历信息
        
        Args:
            stock_code: 股票代码
            data: 更新的数据
            
        Returns:
            Optional[NewStockCalendar]: 更新后的新股日历信息，如不存在则返回None
        """
        try:
            new_stock = await NewStockCalendar.objects.filter(zqdm=stock_code).afirst()
            if new_stock:
                # 更新新股日历信息
                for field, value in data.items():
                    setattr(new_stock, field, value)
                await new_stock.asave()
                logger.debug(f"更新新股日历记录: {stock_code}")
                return new_stock
            else:
                # 新股不存在，创建新记录
                new_stock = NewStockCalendar(**data)
                await new_stock.asave()
                logger.debug(f"创建新股日历记录: {stock_code}")
                return new_stock
        except Exception as e:
            logger.error(f"更新新股日历数据出错: {e}")
            return None


class STStockListDAO(BaseDAO[STStockList]):
    """
    风险警示股票DAO，负责ST股票数据的访问
    """
    
    def __init__(self):
        """初始化STStockListDAO"""
        super().__init__(STStockList, "st_stock")
        self.api = StockBasicAPI()
        logger.info("初始化STStockListDAO")
    
    async def get_by_code(self, stock_code: str) -> Optional[STStockList]:
        """
        根据股票代码获取ST股票信息
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Optional[STStockList]: ST股票信息，如不存在则返回None
        """
        # 1. 首先尝试从缓存获取
        cache_key = f"st_stock:{stock_code}"
        cached_data = await self.get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 2. 缓存未命中，从数据库查询
        try:
            st_stock = await STStockList.objects.filter(dm=stock_code).afirst()
            if st_stock:
                # 找到数据，存入缓存并返回
                await self.set_to_cache(cache_key, st_stock)
                return st_stock
        except Exception as e:
            logger.error(f"从数据库获取ST股票出错: {e}")
        
        # 3. 数据库未找到，从API获取
        try:
            st_stocks_data = await self.api.get_st_stock_list()
            
            # 处理API返回的数据，找到目标ST股票
            target_st_stock = None
            for st_stock_data in st_stocks_data:
                if st_stock_data.get('dm') == stock_code:
                    target_st_stock = st_stock_data
                    break
            
            if target_st_stock:
                # 映射字段并保存到数据库
                mapped_data = self._map_api_to_model(target_st_stock, ST_STOCK_LIST_MAPPING)
                st_stock = await self._save_to_db(mapped_data)
                
                # 保存到缓存
                await self.set_to_cache(cache_key, st_stock)
                return st_stock
        except Exception as e:
            logger.error(f"从API获取ST股票出错: {e}")
        
        return None
    
    async def get_all_st_stocks(self) -> List[STStockList]:
        """
        获取所有ST股票信息
        
        Returns:
            List[STStockList]: 所有ST股票信息列表
        """
        # 1. 首先尝试从缓存获取
        cache_key = "all_st_stocks"
        cached_data = await self.get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 2. 缓存未命中，从数据库查询
        try:
            st_stocks = await STStockList.objects.all()
            if st_stocks:
                # 找到数据，存入缓存并返回
                await self.set_to_cache(cache_key, list(st_stocks))
                return st_stocks
        except Exception as e:
            logger.error(f"从数据库获取所有ST股票出错: {e}")
        
        # 3. 数据库未找到或为空，从API获取
        try:
            st_stocks_data = await self.api.get_st_stock_list()
            
            # 批量保存所有ST股票信息
            st_stocks = []
            for st_stock_data in st_stocks_data:
                mapped_data = self._map_api_to_model(st_stock_data, ST_STOCK_LIST_MAPPING)
                st_stock = await self._save_to_db(mapped_data)
                st_stocks.append(st_stock)
            
            # 保存到缓存
            await self.set_to_cache(cache_key, st_stocks)
            return st_stocks
        except Exception as e:
            logger.error(f"从API获取所有ST股票出错: {e}")
        
        return []
    
    def _map_api_to_model(self, api_data: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        将API返回的数据映射到模型字段
        
        Args:
            api_data: API返回的数据
            mapping: 字段映射关系
            
        Returns:
            Dict[str, Any]: 映射后的数据
        """
        return {
            model_field: api_data.get(api_field)
            for api_field, model_field in mapping.items()
            if api_field in api_data and api_data.get(api_field) is not None
        }
    
    @transaction.atomic
    async def _save_to_db(self, data: Dict[str, Any]) -> STStockList:
        """
        保存数据到数据库
        
        Args:
            data: 模型数据
            
        Returns:
            STStockList: 保存后的模型实例
        """
        try:
            st_stock, created = await STStockList.objects.aupdate_or_create(
                dm=data['dm'],
                defaults=data
            )
            if created:
                logger.debug(f"创建ST股票记录: {data['dm']}")
            else:
                logger.debug(f"更新ST股票记录: {data['dm']}")
            return st_stock
        except Exception as e:
            logger.error(f"保存ST股票数据出错: {e}")
            raise


class CompanyInfoDAO(BaseDAO[CompanyInfo]):
    """
    公司简介DAO，负责公司基本信息的访问
    """
    
    def __init__(self):
        """初始化CompanyInfoDAO"""
        super().__init__(CompanyInfo, "company_info")
        self.api = StockBasicAPI()
        logger.info("初始化CompanyInfoDAO")
    
    async def get_by_code(self, stock_code: str) -> Optional[CompanyInfo]:
        """
        根据股票代码获取公司简介
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Optional[CompanyInfo]: 公司简介信息，如不存在则返回None
        """
        # 1. 首先尝试从缓存获取
        cache_key = f"company_info:{stock_code}"
        cached_data = await self.get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 2. 缓存未命中，从数据库查询
        try:
            company_info = await CompanyInfo.objects.filter(stock_code=stock_code).afirst()
            if company_info:
                # 找到数据，存入缓存并返回
                await self.set_to_cache(cache_key, company_info)
                return company_info
        except Exception as e:
            logger.error(f"从数据库获取公司简介出错: {e}")
        
        # 3. 数据库未找到，从API获取
        try:
            company_data = await self.api.get_company_info(stock_code)
            
            if company_data:
                # 处理日期字段
                if 'ldate' in company_data and company_data['ldate'] and company_data['ldate'] != 'null':
                    try:
                        company_data['ldate'] = date.fromisoformat(company_data['ldate'])
                    except (ValueError, TypeError):
                        logger.warning(f"日期字段处理失败: ldate={company_data['ldate']}")
                        company_data['ldate'] = None
                
                # 映射字段并保存到数据库
                mapped_data = self._map_api_to_model(company_data, COMPANY_INFO_MAPPING)
                mapped_data['stock_code'] = stock_code  # 添加股票代码
                
                company_info = await self._save_to_db(mapped_data)
                
                # 保存到缓存
                await self.set_to_cache(cache_key, company_info)
                return company_info
        except Exception as e:
            logger.error(f"从API获取公司简介出错: {e}")
        
        return None
    
    def _map_api_to_model(self, api_data: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        将API返回的数据映射到模型字段
        
        Args:
            api_data: API返回的数据
            mapping: 字段映射关系
            
        Returns:
            Dict[str, Any]: 映射后的数据
        """
        return {
            model_field: api_data.get(api_field)
            for api_field, model_field in mapping.items()
            if api_field in api_data and api_data.get(api_field) is not None
        }
    
    @transaction.atomic
    async def _save_to_db(self, data: Dict[str, Any]) -> CompanyInfo:
        """
        保存数据到数据库
        
        Args:
            data: 模型数据
            
        Returns:
            CompanyInfo: 保存后的模型实例
        """
        try:
            company_info, created = await CompanyInfo.objects.aupdate_or_create(
                stock_code=data['stock_code'],
                defaults=data
            )
            if created:
                logger.debug(f"创建公司简介记录: {data['stock_code']}")
            else:
                logger.debug(f"更新公司简介记录: {data['stock_code']}")
            return company_info
        except Exception as e:
            logger.error(f"保存公司简介数据出错: {e}")
            raise
