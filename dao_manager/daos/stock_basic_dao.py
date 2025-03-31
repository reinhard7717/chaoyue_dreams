import json
import logging
import asyncio
import sys
import functools
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Any, Optional
from django.contrib.auth.models import User
from users.models import FavoriteStock
from utils.models import ModelJSONEncoder
from django.db import transaction
from django.core.cache import cache
from asgiref.sync import sync_to_async
from utils.cache_manager import CacheManager

# 解决Python 3.12上asyncio.coroutines没有_DEBUG属性的问题
if sys.version_info >= (3, 12):
    # 在Python 3.12中，_DEBUG被替换为_is_debug_mode函数
    if not hasattr(asyncio.coroutines, '_DEBUG'):
        # 为了兼容性，添加一个_DEBUG属性，其值由_is_debug_mode()函数确定
        asyncio.coroutines._DEBUG = asyncio.coroutines._is_debug_mode()

from api_manager.apis.stock_basic_api import StockBasicAPI
from api_manager.mappings.stock_basic import COMPANY_INFO_MAPPING, NEW_STOCK_CALENDAR_MAPPING, ST_STOCK_LIST_MAPPING, STOCK_BASIC_MAPPING
from dao_manager.base_dao import BaseDAO
from stock_models.stock_basic import CompanyInfo, NewStockCalendar, STStockList, StockInfo, StockInfo

logger = logging.getLogger("dao")

class StockBasicDAO(BaseDAO):
    """
    股票基础信息DAO，整合所有相关的基础信息访问功能
    
    实现三层数据访问结构：Redis缓存 -> MySQL数据库 -> 外部API
    """
    
    def __init__(self):
        """初始化StockBasicDAO"""
        super().__init__(None, None, 3600)  # 基类使用None作为model_class，因为本DAO管理多个模型
        self.api = StockBasicAPI()
        self.cache_manager = CacheManager()  # 初始化缓存管理器
        logger.info("初始化StockBasicDAO")
    
    def _serialize_model(self, model_instance) -> dict:
        """
        将Django模型实例序列化为可JSON化的字典
        
        Args:
            model_instance: Django模型实例
            
        Returns:
            dict: 序列化后的字典
        """
        if model_instance is None:
            return None
            
        # 如果已经是字典，直接返回
        if isinstance(model_instance, dict):
            return model_instance
        
        # 确保是模型实例    
        if not hasattr(model_instance, '_meta'):
            raise TypeError(f"Expected Django model instance, got {type(model_instance)}")
            
        result = {}
        
        # 遍历所有字段
        for field in model_instance._meta.fields:
            field_name = field.name
            value = getattr(model_instance, field_name)
            
            # 处理None值
            if value is None:
                result[field_name] = None
                continue
                
            # 处理各种类型
            if isinstance(value, (str, int, float, bool)):
                # 基本类型可直接使用
                result[field_name] = value
                
            elif isinstance(value, datetime):
                # 日期时间转ISO格式字符串
                result[field_name] = value.isoformat()
                
            elif isinstance(value, Decimal):
                # Decimal转浮点数
                result[field_name] = float(value)
                
            elif hasattr(value, 'pk') and hasattr(value, '_meta'):
                # 处理外键关系-只保存主键
                result[field_name] = value.pk
                
            elif isinstance(value, (list, tuple)):
                # 列表或元组-尝试序列化内部元素
                try:
                    result[field_name] = [
                        self._serialize_model(item) if hasattr(item, '_meta') else item 
                        for item in value
                    ]
                except:
                    # 无法序列化的列表元素
                    result[field_name] = str(value)
                    
            elif hasattr(value, '__dict__'):
                # 尝试使用__dict__
                try:
                    result[field_name] = value.__dict__
                except:
                    result[field_name] = str(value)
                    
            else:
                # 其他类型转为字符串
                result[field_name] = str(value)
        
        # 移除Django内部字段
        result.pop('_state', None)
        
        return result
    
    # ================= 股票基本信息相关方法 =================
    
    async def get_stock_list(self) -> List[StockInfo]:
        """
        获取所有股票的基本信息
        
        Returns:
            List[StockInfo]: 股票基本信息列表
        """
        # 使用CacheManager生成标准化缓存键
        cache_key = self.cache_manager.generate_key('st', 'stock', 'all')
        
        # 添加更详细的日志
        logger.info("开始获取股票列表数据")
        
        # 1. 首先尝试从缓存获取
        cached_data = self.cache_manager.get(cache_key)
        if cached_data:
            logger.info(f"从缓存获取股票列表成功，共{len(cached_data)}条数据")
            # 将缓存的字典数据转换回模型对象
            return [StockInfo(**item) for item in cached_data]
        
        # 2. 缓存未命中，从数据库查询
        try:
            @sync_to_async
            def get_db_items():
                return list(StockInfo.objects.all())
            
            items = await get_db_items()
            
            if items:
                logger.info(f"从数据库获取股票列表成功，共{len(items)}条数据")
                # 序列化模型实例
                items_dict = [self._serialize_model(item) for item in items]
                # 保存到缓存
                self.cache_manager.set(cache_key, items_dict, 
                                      timeout=self.cache_manager.get_timeout('st'))
                return items
        except Exception as e:
            logger.error(f"从数据库获取股票列表失败: {e}")
        
        # 3. 数据库未找到或为空，从API获取
        try:
            # 直接调用API获取数据
            api_datas = await self.api.get_stock_list()
            
            if not api_datas:
                logger.warning(f"API未返回股票列表数据")
                return []
            
            data_dicts = []
            for api_data in api_datas:
                data_dict = {
                    'stock_code': api_data.get('dm'),
                    'stock_name': api_data.get('mc'),
                    'exchange': api_data.get('jys'),
                }
                data_dicts.append(data_dict)

            # 保存数据
            logger.info(f"开始保存股票列表数据")
            result = await self._save_all_to_db(
                model_class=StockInfo,
                data_list=data_dicts,
                unique_fields=['stock_code']
            )
            
            logger.info(f"股票列表数据保存完成，结果: {result}")
            return result
            
        except Exception as e:
            logger.error(f"从API获取股票列表失败: {e}")
            logger.exception("获取股票列表异常详情")
            return []

    async def get_stock_by_code(self, stock_code: str) -> Optional[StockInfo]:
        """
        根据股票代码获取股票信息
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Optional[StockInfo]: 股票信息
        """
        # 使用CacheManager生成标准化缓存键
        cache_key = self.cache_manager.generate_key('st', 'stock', stock_code)
        
        # 尝试从缓存获取
        cached_data = self.cache_manager.get_model(cache_key, StockInfo)
        if cached_data:
            return cached_data
            
        # 从数据库获取
        try:
            @sync_to_async
            def get_db_item():
                return StockInfo.objects.filter(stock_code=stock_code).first()
            
            item = await get_db_item()
            
            if item:
                # 序列化并缓存
                self.cache_manager.set(
                    cache_key, 
                    self._serialize_model(item),
                    timeout=self.cache_manager.get_timeout('st')
                )
                return item
            
            # 如果数据库中没有数据，从API获取并保存
            logger.info(f"股票代码[{stock_code}]不存在，从API获取")
            stock = await self._fetch_and_save_stock(stock_code)
            if stock:
                # 序列化并缓存
                self.cache_manager.set(
                    cache_key, 
                    self._serialize_model(stock),
                    timeout=self.cache_manager.get_timeout('st')
                )
                return stock
            
            return None
        except Exception as e:
            logger.error(f"获取股票[{stock_code}]失败: {e}")
            return None
    
    async def get_favorite_stocks_by_user(self, user: User) -> List[FavoriteStock]:  
        """
        获取用户自选股
        
        Args:
            user: 用户
        """
        # 使用CacheManager生成标准化缓存键
        cache_key = self.cache_manager.generate_key('st', 'favorite_stock', user.id)
        
        # 尝试从缓存获取
        cached_data = self.cache_manager.get(cache_key)
        if cached_data:
            return [FavoriteStock(**item) for item in cached_data]

        # 从数据库获取
        try:
            @sync_to_async
            def get_db_items():
                return list(FavoriteStock.objects.filter(user=user))
            
            items = await get_db_items()
            
            if items:
                # 序列化并缓存
                items_dict = [self._serialize_model(item) for item in items]
                self.cache_manager.set(
                    cache_key, 
                    items_dict,
                    timeout=self.cache_manager.get_timeout('st')
                )
                return items
            
            return []
        except Exception as e:
            logger.error(f"获取用户自选股失败: {e}")
            return []

    async def get_all_favorite_stocks(self) -> Optional[FavoriteStock]:
        """
        获取所有自选股
        """
        # 使用CacheManager生成标准化缓存键
        cache_key = self.cache_manager.generate_key('st', 'favorite_stock', 'all')
        
        # 尝试从缓存获取
        cached_data = self.cache_manager.get(cache_key)
        if cached_data:
            return [FavoriteStock(**item) for item in cached_data]
            
        # 从数据库获取  
        try:
            @sync_to_async
            def get_db_items():
                return list(FavoriteStock.objects.all())
            
            items = await get_db_items()
            
            if items:
                # 序列化并缓存
                items_dict = [self._serialize_model(item) for item in items]
                self.cache_manager.set(
                    cache_key, 
                    items_dict,
                    timeout=self.cache_manager.get_timeout('st')
                )
                return items
            
            return []
        except Exception as e:
            logger.error(f"获取所有自选股失败: {e}")
            return []

        
    # ================= 新股日历相关方法 =================
    
    async def get_new_stock_by_code(self, stock_code: str) -> Optional[NewStockCalendar]:
        """
        根据股票代码获取新股信息
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Optional[NewStockCalendar]: 新股信息
        """
        # 使用CacheManager生成标准化缓存键
        cache_key = self.cache_manager.generate_key('st', 'new_stock', stock_code)
        
        # 尝试从缓存获取
        cached_data = self.cache_manager.get_model(cache_key, NewStockCalendar)
        if cached_data:
            return cached_data
            
        # 从数据库获取
        try:
            @sync_to_async
            def get_db_item():
                return NewStockCalendar.objects.filter(stock_code=stock_code).first()
            
            item = await get_db_item()
            
            if item:
                # 序列化并缓存
                self.cache_manager.set(
                    cache_key, 
                    self._serialize_model(item),
                    timeout=self.cache_manager.get_timeout('st')
                )
                return item
                
            # 从API获取
            api_data = await self.api.get_new_stock_calendar()
            if not api_data:
                return None
                
            # 查找匹配的数据
            found_data = None
            for data in api_data:
                if data.get('dm') == stock_code:
                    found_data = data
                    break
                    
            if not found_data:
                return None
                
            # 映射数据
            mapped_data = self._map_api_to_model(found_data, NEW_STOCK_CALENDAR_MAPPING)
            self._process_date_fields(mapped_data)
            
            # 保存到数据库
            item = await self._save_to_db(NewStockCalendar, mapped_data)
            if item:
                # 序列化并缓存
                self.cache_manager.set(
                    cache_key, 
                    self._serialize_model(item),
                    timeout=self.cache_manager.get_timeout('st')
                )
                
            return item
        except Exception as e:
            logger.error(f"获取新股数据失败: {e}")
            return None
    
    async def get_all_new_stocks(self) -> List[NewStockCalendar]:
        """
        获取所有新股信息
        
        Returns:
            List[NewStockCalendar]: 新股信息列表
        """
        # 使用CacheManager生成标准化缓存键
        cache_key = self.cache_manager.generate_key('st', 'new_stock', 'all')
        
        # 尝试从缓存获取
        cached_data = self.cache_manager.get(cache_key)
        if cached_data:
            # 将缓存的字典数据转换回模型对象
            return [NewStockCalendar(**item) for item in cached_data]
            
        # 从数据库获取
        try:
            @sync_to_async
            def get_all_items():
                return list(NewStockCalendar.objects.all())
            
            items = await get_all_items()
            
            if items:
                # 序列化并缓存
                items_dict = [self._serialize_model(item) for item in items]
                self.cache_manager.set(
                    cache_key, 
                    items_dict,
                    timeout=self.cache_manager.get_timeout('st')
                )
                return items
                
            # 从API获取
            api_data = await self.api.get_new_stock_calendar()
            if not api_data:
                return []
                
            # 批量保存
            saved_items = []
            for data in api_data:
                # 映射数据
                mapped_data = self._map_api_to_model(data, NEW_STOCK_CALENDAR_MAPPING)
                self._process_date_fields(mapped_data)
                
                # 保存到数据库
                item = await self._save_to_db(NewStockCalendar, mapped_data)
                if item:
                    saved_items.append(item)
            
            if saved_items:
                # 序列化并缓存
                items_dict = [self._serialize_model(item) for item in saved_items]
                self.cache_manager.set(
                    cache_key, 
                    items_dict,
                    timeout=self.cache_manager.get_timeout('st')
                )
                
            return saved_items
        except Exception as e:
            logger.error(f"获取所有新股数据失败: {e}")
            return []
    
    async def refresh_new_stock_data(self) -> List[Dict]:
        """
        刷新新股数据
        
        Returns:
            List[Dict]: 刷新后的新股数据列表
        """
        try:
            # 从API获取数据
            data_list = await self.api.get_new_stock_calendar()
            
            if not data_list:
                logger.warning('获取新股数据失败')
                return []
            
            # 批量保存到数据库
            saved_items = []
            for data in data_list:
                # 映射数据
                mapped_data = self._map_api_to_model(data, NEW_STOCK_CALENDAR_MAPPING)
                
                # 处理日期字段
                self._process_date_fields(mapped_data)
                
                # 保存到数据库
                item = await self._save_to_db(NewStockCalendar, mapped_data)
                if item:
                    saved_items.append(item)
                    
                    # 更新单个缓存
                    stock_code = mapped_data.get('stock_code')
                    if stock_code:
                        single_cache_key = self.cache_manager.generate_key('st', 'new_stock', stock_code)
                        self.cache_manager.set(
                            single_cache_key, 
                            self._serialize_model(item),
                            timeout=self.cache_manager.get_timeout('st')
                        )
            
            # 更新全局缓存
            if saved_items:
                all_cache_key = self.cache_manager.generate_key('st', 'new_stock', 'all')
                items_dict = [self._serialize_model(item) for item in saved_items]
                self.cache_manager.set(
                    all_cache_key, 
                    items_dict,
                    timeout=self.cache_manager.get_timeout('st')
                )
            
            return saved_items
            
        except Exception as e:
            logger.error(f'刷新新股数据出错: {str(e)}')
            return []
    
    # ================= ST股票相关方法 =================
    
    async def get_st_stock_by_code(self, stock_code: str) -> Optional[STStockList]:
        """
        根据股票代码获取ST股票信息
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Optional[STStockList]: ST股票信息
        """
        # 使用CacheManager生成标准化缓存键
        cache_key = self.cache_manager.generate_key('st', 'st_stock', stock_code)
        
        # 尝试从缓存获取
        st_stock = self.cache_manager.get_model(cache_key, STStockList)
        if st_stock:
            return st_stock
            
        # 从数据库获取
        try:
            @sync_to_async
            def get_db_item():
                return STStockList.objects.filter(stock_code=stock_code).first()
            
            item = await get_db_item()
            
            if item:
                # 序列化并缓存
                self.cache_manager.set(
                    cache_key, 
                    self._serialize_model(item),
                    timeout=self.cache_manager.get_timeout('st')
                )
                return item
                
            # 从API获取
            api_data_list = await self.api.get_st_stock_list()
            if not api_data_list:
                return None
                
            # 查找匹配的数据
            found_data = None
            for data in api_data_list:
                if data.get('dm') == stock_code:
                    found_data = data
                    break
                    
            if not found_data:
                return None
                
            # 映射数据
            mapped_data = self._map_api_to_model(found_data, ST_STOCK_LIST_MAPPING)
            
            # 保存到数据库
            item = await self._save_to_db(STStockList, mapped_data)
            if item:
                # 序列化并缓存
                self.cache_manager.set(
                    cache_key, 
                    self._serialize_model(item),
                    timeout=self.cache_manager.get_timeout('st')
                )
                
            return item
        except Exception as e:
            logger.error(f"获取ST股票数据失败: {e}")
            return None
    
    async def get_all_st_stocks(self) -> List[STStockList]:
        """
        获取所有ST股票信息
        
        Returns:
            List[STStockList]: ST股票信息列表
        """
        # 使用CacheManager生成标准化缓存键
        cache_key = self.cache_manager.generate_key('st', 'st_stock', 'all')
        
        # 尝试从缓存获取
        cached_data = self.cache_manager.get(cache_key)
        if cached_data:
            # 将缓存的字典数据转换回模型对象
            return [STStockList(**item) for item in cached_data]
            
        # 从数据库获取
        try:
            @sync_to_async
            def get_all_items():
                return list(STStockList.objects.all())
            
            items = await get_all_items()
            
            if items:
                # 序列化并缓存
                items_dict = [self._serialize_model(item) for item in items]
                self.cache_manager.set(
                    cache_key, 
                    items_dict,
                    timeout=self.cache_manager.get_timeout('st')
                )
                return items
                
            # 从API获取
            api_data = await self.api.get_st_stock_list()
            if not api_data:
                return []
                
            # 批量保存
            saved_items = []
            for data in api_data:
                # 映射数据
                mapped_data = self._map_api_to_model(data, ST_STOCK_LIST_MAPPING)
                
                # 保存到数据库
                item = await self._save_to_db(STStockList, mapped_data)
                if item:
                    saved_items.append(item)
            
            if saved_items:
                # 序列化并缓存
                items_dict = [self._serialize_model(item) for item in saved_items]
                self.cache_manager.set(
                    cache_key, 
                    items_dict,
                    timeout=self.cache_manager.get_timeout('st')
                )
                
            return saved_items
        except Exception as e:
            logger.error(f"获取所有ST股票数据失败: {e}")
            return []
    
    # ================= 公司信息相关方法 =================
    
    async def get_company_info_by_code(self, stock_code: str) -> Optional[CompanyInfo]:
        """
        根据股票代码获取公司信息
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Optional[CompanyInfo]: 公司信息
        """
        # 使用CacheManager生成标准化缓存键
        cache_key = self.cache_manager.generate_key('st', 'company', stock_code)
        
        # 尝试从缓存获取
        company = self.cache_manager.get_model(cache_key, CompanyInfo)
        if company:
            return company
            
        # 从数据库获取
        try:
            @sync_to_async
            def get_db_item():
                return CompanyInfo.objects.filter(stock_code=stock_code).first()
            
            item = await get_db_item()
            
            if item:
                # 序列化并缓存
                self.cache_manager.set(
                    cache_key, 
                    self._serialize_model(item),
                    timeout=self.cache_manager.get_timeout('st')
                )
                return item
                
            # 从API获取
            api_data = await self.api.get_company_info(stock_code)
            if not api_data:
                return None
                
            # 映射数据
            mapped_data = self._map_api_to_model(api_data, COMPANY_INFO_MAPPING)
            mapped_data['stock_code'] = stock_code
            
            # 保存到数据库
            item = await self._save_to_db(CompanyInfo, mapped_data)
            if item:
                # 序列化并缓存
                self.cache_manager.set(
                    cache_key, 
                    self._serialize_model(item),
                    timeout=self.cache_manager.get_timeout('st')
                )
                
            return item
        except Exception as e:
            logger.error(f"获取公司信息失败: {e}")
            return None
    
    async def refresh_company_info(self, stock_code: str) -> bool:
        """
        刷新指定股票的公司信息
        
        Args:
            stock_code: 股票代码
            
        Returns:
            bool: 操作是否成功
        """
        try:
            logger.info(f"开始刷新公司信息: {stock_code}")
            
            # 从API获取公司信息
            company_data = await self.api.get_company_info(stock_code)
            
            if not company_data:
                logger.warning(f"从API获取的公司信息为空: {stock_code}")
                return False
            
            logger.debug(f"API返回的公司信息: {company_data}")
            
            # 映射数据
            mapped_data = self._map_api_to_model(company_data, COMPANY_INFO_MAPPING)
            
            # 确保股票代码字段存在
            mapped_data['stock_code'] = stock_code
            
            # 如果映射后的数据为空或只包含股票代码，返回失败
            if not mapped_data or len(mapped_data) <= 1:
                logger.warning(f"映射后的公司信息为空或不完整，原始数据: {company_data}")
                return False
                
            logger.debug(f"映射后的公司信息: {mapped_data}")
            
            # 保存到数据库
            try:
                # 将数据转换为字典格式，并处理日期和数字字段
                data_dict = {
                    'stock_code': mapped_data['stock_code'],
                    'company_name': mapped_data.get('company_name', ''),
                    'company_english_name': mapped_data.get('company_english_name', ''),
                    'market': mapped_data.get('market', ''),
                    'concepts': mapped_data.get('concepts', ''),
                    'listing_date': self._parse_datetime(mapped_data.get('listing_date')),
                    'issue_price': self._parse_number(mapped_data.get('issue_price')),
                    'lead_underwriter': mapped_data.get('lead_underwriter', ''),
                    'establishment_date': mapped_data.get('establishment_date', ''),
                    'registered_capital': self._parse_number(mapped_data.get('registered_capital')),
                    'institution_type': mapped_data.get('institution_type', ''),
                    'organization_form': mapped_data.get('organization_form', '')
                }
                
                # 使用自定义的_save_to_db方法保存数据
                saved_item = await self._save_to_db(CompanyInfo, data_dict)
                
                if not saved_item:
                    logger.error(f"保存公司信息失败: {stock_code}")
                    return False
                
                # 更新缓存
                cache_key = self.cache_manager.generate_key('st', 'company', stock_code)
                # 序列化并缓存
                self.cache_manager.set(
                    cache_key, 
                    self._serialize_model(saved_item),
                    timeout=self.cache_manager.get_timeout('st')
                )
                
                logger.info(f"成功刷新公司信息: {stock_code}")
                return True
            except Exception as e:
                logger.error(f"保存公司信息时出错: {e}")
                logger.exception(f"保存公司信息时出错，数据: {mapped_data}")
                return False
        except Exception as e:
            logger.error(f"刷新公司信息出错: {e}")
            logger.exception(f"刷新公司信息异常详情: {stock_code}")
            return False
