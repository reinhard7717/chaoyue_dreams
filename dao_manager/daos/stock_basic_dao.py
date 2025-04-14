import json
import logging
import asyncio
import sys
# import functools
from asgiref.sync import sync_to_async
# from datetime import datetime
# from decimal import Decimal
from typing import Dict, List, Any, Optional
# from utils.models import ModelJSONEncoder
# from django.db import transaction
# from django.core.cache import cache

from utils import cache_constants as cc
from utils.cache_get import UserCacheGet
from utils.cache_set import UserCacheSet
from utils.data_format_process import UserDataFormatProcess # 导入常量

# 解决Python 3.12上asyncio.coroutines没有_DEBUG属性的问题
if sys.version_info >= (3, 12):
    # 在Python 3.12中，_DEBUG被替换为_is_debug_mode函数
    if not hasattr(asyncio.coroutines, '_DEBUG'):
        # 为了兼容性，添加一个_DEBUG属性，其值由_is_debug_mode()函数确定
        asyncio.coroutines._DEBUG = asyncio.coroutines._is_debug_mode()


from api_manager.mappings.stock_basic import COMPANY_INFO_MAPPING, NEW_STOCK_CALENDAR_MAPPING, ST_STOCK_LIST_MAPPING, STOCK_BASIC_MAPPING
from dao_manager.base_dao import BaseDAO

logger = logging.getLogger("dao")

class StockBasicDAO(BaseDAO):
    """
    股票基础信息DAO，整合所有相关的基础信息访问功能
    
    实现三层数据访问结构：Redis缓存 -> MySQL数据库 -> 外部API
    """
    
    def __init__(self):
        """初始化StockBasicDAO"""
        super().__init__(None, None, 3600)  # 基类使用None作为model_class，因为本DAO管理多个模型
        from utils.cache_get import StockInfoCacheGet
        from utils.cash_key import StockCashKey
        from utils.data_format_process import StockInfoFormatProcess
        from utils.cache_manager import CacheManager
        from api_manager.apis.stock_basic_api import StockBasicAPI
        self.api = StockBasicAPI()
        self.cache_manager = CacheManager()  # 初始化缓存管理器
        self.data_format_process = StockInfoFormatProcess()
        self.cache_key = StockCashKey()
        self.cache_get = StockInfoCacheGet()
        self.user_cache_set = UserCacheSet()
        self.user_cache_get = UserCacheGet()
        self.user_data_format_process = UserDataFormatProcess()

    # 新增 close 方法
    async def close(self):
        """关闭内部持有的 API Client Session"""
        if hasattr(self, 'api') and self.api:
            # logger.debug("Closing StockBasicDAO's internal API client...") # 可选日志
            await self.api.close() # 调用 StockBasicAPI 的 close 方法
            # logger.debug("StockBasicDAO's internal API client closed.") # 可选日志
        else:
            # logger.debug("StockBasicDAO has no API client to close or it's already None.") # 可选日志
            pass
    
    # ================= 股票基本信息相关方法 =================
    
    async def get_stock_list(self) -> List['StockInfo']:
        """
        获取所有股票的基本信息
        
        Returns:
            List[StockInfo]: 股票基本信息列表
        """
        from stock_models.stock_basic import StockInfo
        try:
            # 尝试从缓存获取
            cached_data = await self.cache_get.all_stocks()
            if cached_data:
                # logger.debug("从缓存获取股票列表")
                # 将缓存数据转换为模型实例列表
                return_data = sorted([StockInfo(**stock_dict) for stock_dict in cached_data], key=lambda x: x.stock_code)
                # logger.info(f"从缓存获取股票列表成功，共{len(return_data)}只股票")
                return return_data
        except Exception as e:
            logger.error(f"从缓存获取股票列表失败: {e}")
        stocks = []
        try:
            # 从数据库读取
            get_stocks_sync = sync_to_async(
                lambda: list(StockInfo.objects.order_by('stock_code')),
                thread_sensitive=True # 对于 ORM 操作，通常建议设置为 True
            )
            stocks = await get_stocks_sync()
            if stocks:
                await self.set_cache_stocks(stocks)
                return stocks
        except Exception as e:
            logger.error(f"从数据库读取股票列表失败: {e}")
        
        # 如果数据库中没有数据，从API获取并保存
        logger.info("数据库中没有股票数据，从API获取")
        await self.fetch_and_save_stocks()
        # 从数据库读取
        get_stocks_sync = sync_to_async(
            lambda: list(StockInfo.objects.order_by('stock_code')),
            thread_sensitive=True # 对于 ORM 操作，通常建议设置为 True
        )
        stocks = await get_stocks_sync()
        return stocks

    async def get_stock_by_code(self, stock_code: str) -> Optional['StockInfo']:
        """
        根据股票代码获取股票信息
        Args:
            stock_code: 股票代码
        Returns:
            Optional[StockInfo]: 股票信息
        """
        from stock_models.stock_basic import StockInfo
        # 使用CacheManager生成标准化缓存键
        cache_key = self.cache_key.stock_data(stock_code)
        # 尝试从缓存获取，指定模型类进行自动转换
        stock = self.cache_manager.get_model(cache_key, StockInfo)
        if stock:
            return stock
            
        # 从数据库获取
        # logger.info(f"get_stock_by_code从数据库获取股票: {cache_key}, {stock_code}")
        stock = await sync_to_async(StockInfo.objects.get)(stock_code=stock_code)
        # 如果数据库中有数据，缓存并返回
        if stock:
            cache_data = self.data_format_process.set_stock_info_data(stock)
            # logger.info(f"get_stock_by_code,cache_data: {cache_data}, type: {type(cache_data)}")
            # *** 正确调用 CacheManager 缓存数据 ***
            success = self.cache_manager.set(
                key=cache_key,          # 第一个参数：缓存键 (字符串)
                data=cache_data,     # 第二个参数：要缓存的数据 (字典)
                timeout=self.cache_manager.get_timeout('st') # 超时时间
            )
            # logger.info(f"get_stock_by_code,success: {success}")
            return stock

    async def get_favorite_stocks_by_user(self, user: 'User') -> List['FavoriteStock']:  
        """
        获取用户自选股
        Args:
            user: 用户
        """
        from django.contrib.auth.models import User
        from users.models import FavoriteStock
        # 从缓存获取
        fav_datas = []
        # fav_datas = await self.user_cache_get.user_favorites(user.id)
        # if fav_datas:
        #     return fav_datas
        # 从数据库获取
        items = FavoriteStock.objects.filter(user=user)
        for item in items:
            fav_data = self.user_data_format_process.set_user_favorites(user.id, item)
            fav_datas.append(fav_data)
            await self.user_cache_set.user_favorites(user.id, item)
        return fav_datas

    async def get_all_favorite_stocks(self) -> Optional['FavoriteStock']:
        """
        获取所有自选股
        """
        from users.models import FavoriteStock
        # 从缓存获取
        fav_datas = []
        # fav_datas = await self.user_cache_get.user_favorites(user.id)
        # if fav_datas:
        #     return fav_datas
        # 从数据库获取
        items = FavoriteStock.objects.all()
        for item in items:
            fav_data = self.user_data_format_process.set_user_favorites(user.id, item)
            fav_datas.append(fav_data)
            await self.user_cache_set.all_favorites(item)
        return fav_datas
    
    async def get_cache_all_stocks(self) -> Optional[List[Dict]]:
        """从缓存中获取所有股票列表"""
        try:
            # 生成缓存键
            cache_key = self.cache_key.stocks_data()
            logger.info(f"尝试从缓存获取所有股票列表, key: {cache_key}")
            
            # 从缓存获取数据
            cache_data = self.cache_manager.get(cache_key)
            
            # 检查缓存数据类型并正确处理
            if cache_data is None:
                logger.warning(f"缓存中没有股票列表数据, key: {cache_key}")
                return None
                
            # 如果已经是列表类型，直接返回
            if isinstance(cache_data, list):
                logger.info(f"从缓存获取到股票列表，共{len(cache_data)}条")
                return cache_data
                
            # 如果是字符串，尝试解析为JSON
            if isinstance(cache_data, str):
                try:
                    data = json.loads(cache_data)
                    logger.info(f"从缓存获取的字符串成功解析为JSON，共{len(data)}条")
                    return data
                except json.JSONDecodeError as e:
                    logger.error(f"从缓存获取的字符串JSON解析失败: {str(e)}")
                    return None
            
            # 如果是字节对象，先解码为字符串
            if isinstance(cache_data, bytes):
                try:
                    cache_data = cache_data.decode('utf-8')
                    data = json.loads(cache_data)
                    logger.info(f"从缓存获取的字节数据成功解析为JSON，共{len(data)}条")
                    return data
                except (UnicodeDecodeError, json.JSONDecodeError) as e:
                    logger.error(f"从缓存获取的字节数据解析失败: {str(e)}")
                    return None
            
            # 其他类型，尝试转换为字符串后解析
            try:
                cache_data_str = str(cache_data)
                data = json.loads(cache_data_str)
                logger.info(f"从缓存获取的其他类型数据成功解析为JSON，共{len(data)}条")
                return data
            except json.JSONDecodeError as e:
                logger.error(f"从缓存获取的其他类型数据JSON解析失败: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"从缓存获取所有股票列表时发生错误: {str(e)}", exc_info=True)
            return None
        
    async def set_cache_stocks(self, stocks: List[Dict]) -> bool:
        """
        将提供的股票数据列表（简单字典格式）设置到缓存中。
        此方法用于手动将准备好的、符合缓存结构的数据放入缓存。
        数据格式应与 fetch_and_save_indexes 写入缓存的格式一致。
        Args:
            stocks: 包含股票信息的字典列表，格式应为
                   [{'stock_code': 'xxx', 'stock_name': 'yyy', 'exchange': 'zzz'}, ...]
                   注意：不应包含 id, created_at 等数据库特有字段。
        Returns:
            bool: 操作是否成功。
        """
        from stock_models.stock_basic import StockInfo
        # 1. 输入验证
        if not isinstance(stocks, list):
            logger.error("set_stocks_to_cache 失败: 输入数据不是列表")
            return False
        # 2. 生成缓存键
        cache_key = self.cache_key.stocks_data()
        try:
            # 3. 获取缓存超时时间
            cache_timeout = self.cache_manager.get_timeout(cc.TYPE_STATIC)
            # 4. 处理数据
            data_dicts = []
            for item in stocks:
                if isinstance(item, StockInfo):
                    # 如果是StockInfo对象，转换为字典
                    data_dict = self.data_format_process.set_stock_info_data(item)
                elif isinstance(item, dict):
                    # 如果已经是字典，确保包含必要的字段
                    if not all(key in item for key in ['stock_code', 'stock_name', 'exchange']):
                        logger.error(f"股票数据缺少必要字段: {item}")
                        continue
                    data_dict = item
                else:
                    logger.error(f"无效的股票数据类型: {type(item)}")
                    continue
                data_dicts.append(data_dict)
            
            if not data_dicts:
                logger.error("没有有效的股票数据可以缓存")
                return False
            logger.info(f"准备将 {len(data_dicts)} 条股票数据设置到缓存, key: {cache_key}, timeout: {cache_timeout}s")

            # *** 正确调用 CacheManager 缓存数据 ***
            set_cache_sync = sync_to_async(
                self.cache_manager.set,       # 要包装的同步方法
                thread_sensitive=False        # Redis 操作通常不需要线程敏感
            )
            # 调用包装后的异步函数
            success = await set_cache_sync(
                key=cache_key,
                data=data_dicts,
                timeout=self.cache_manager.get_timeout('st') # 假设 get_timeout 也是同步的，直接调用获取值
            )
            if success:
                logger.info(f"股票数据成功设置到缓存, key: {cache_key}")
                return True
            else:
                logger.warning(f"设置股票数据到缓存失败 (CacheManager.set 返回 False), key: {cache_key}")
                return False
        except Exception as e:
            logger.error(f"设置股票数据到缓存时发生异常: {str(e)}, key: {cache_key}", exc_info=True)
            return False

    async def fetch_and_save_stocks(self) -> Dict:
        """
        刷新所有股票数据
        
        从API获取股票列表并保存到数据库
        """
        from stock_models.stock_basic import StockInfo
        try:
            # 获取股票列表
            api_datas = await self.api.get_stock_list()
            if not api_datas:
                logger.warning("没有获取到股票数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            data_dicts = []
            for api_data in api_datas:
                data_dict = self.data_format_process.set_stock_info_data(api_data)
                data_dicts.append(data_dict)
            # 保存数据
            result = await self._save_all_to_db_native_upsert(
                model_class=StockInfo,
                data_list=data_dicts,
                unique_fields=['stock_code']
            )
            logger.info(f"股票数据保存完成，结果: {result}")
            # 4. 创建缓存键并保存缓存 (核心修改部分)
            await self.set_cache_stocks(data_dicts)
            return result
        except Exception as e:
            logger.error(f"fetch_and_save_stocks刷新股票数据失败: {str(e)}")
            raise

        
    # ================= 新股日历相关方法 =================
    
    async def get_new_stock_by_code(self, stock_code: str) -> Optional['NewStockCalendar']:
        """
        根据股票代码获取新股信息
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Optional[NewStockCalendar]: 新股信息
        """
        from stock_models.stock_basic import NewStockCalendar
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
    
    async def get_all_new_stocks(self) -> List['NewStockCalendar']:
        """
        获取所有新股信息
        
        Returns:
            List[NewStockCalendar]: 新股信息列表
        """
        from stock_models.stock_basic import NewStockCalendar
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
    
    
    # ================= ST股票相关方法 =================
    
    async def get_st_stock_by_code(self, stock_code: str) -> Optional['STStockList']:
        """
        根据股票代码获取ST股票信息
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Optional[STStockList]: ST股票信息
        """
        from stock_models.stock_basic import STStockList
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
    
    async def get_all_st_stocks(self) -> List['STStockList']:
        """
        获取所有ST股票信息
        
        Returns:
            List[STStockList]: ST股票信息列表
        """
        from stock_models.stock_basic import STStockList
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
    
    async def get_company_info_by_code(self, stock_code: str) -> Optional['CompanyInfo']:
        """
        根据股票代码获取公司信息
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Optional[CompanyInfo]: 公司信息
        """
        from stock_models.stock_basic import CompanyInfo
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
    
