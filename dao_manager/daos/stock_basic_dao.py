import logging
import asyncio
import sys
import functools
from datetime import date, datetime
from typing import Dict, List, Any, Optional, Tuple, Union, TypeVar, Type, Generic

from django.db import transaction, models
from django.core.cache import cache
from asgiref.sync import sync_to_async

# 解决Python 3.12上asyncio.coroutines没有_DEBUG属性的问题
if sys.version_info >= (3, 12):
    # 在Python 3.12中，_DEBUG被替换为_is_debug_mode函数
    if not hasattr(asyncio.coroutines, '_DEBUG'):
        # 为了兼容性，添加一个_DEBUG属性，其值由_is_debug_mode()函数确定
        asyncio.coroutines._DEBUG = asyncio.coroutines._is_debug_mode()

from api_manager.apis.stock_basic_api import StockBasicAPI
from api_manager.mappings.stock_basic import COMPANY_INFO_MAPPING, NEW_STOCK_CALENDAR_MAPPING, ST_STOCK_LIST_MAPPING, STOCK_BASIC_MAPPING
from dao_manager.base_dao import BaseDAO
from stock_models.stock_basic import CompanyInfo, NewStockCalendar, STStockList, StockBasic

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
        logger.info("初始化StockBasicDAO")
    
    # ================= 通用方法 =================
    
    def _map_api_to_model(self, api_data: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        将API数据映射为模型数据，并进行格式转换
        
        Args:
            api_data: API返回的数据
            mapping: 字段映射关系
            
        Returns:
            Dict[str, Any]: 映射后的模型数据
        """
        model_data = {}
        for api_field, model_field in mapping.items():
            if api_field in api_data and api_data.get(api_field) is not None:
                value = api_data[api_field]
                # 日期字段处理
                if model_field.endswith('_date') or model_field.endswith('_time') or model_field == 't':
                    model_data[model_field] = self._parse_datetime(value)
                # 数值字段处理，但排除 stock_code 字段
                elif (isinstance(value, (int, float)) or (
                    isinstance(value, str) and value.replace('.', '', 1).isdigit()
                )) and model_field != 'stock_code':
                    model_data[model_field] = self._parse_number(value)
                else:
                    model_data[model_field] = value
        return model_data
    
    async def get_from_cache(self, cache_key: str):
        """
        从缓存获取数据
        
        Args:
            cache_key: 缓存键
            
        Returns:
            缓存的数据，如不存在则返回None
        """
        return await sync_to_async(cache.get)(cache_key)
        
    async def set_to_cache(self, cache_key: str, data, timeout: int = 3600):
        """
        设置数据到缓存
        
        Args:
            cache_key: 缓存键
            data: 要缓存的数据
            timeout: 缓存超时时间(秒)，默认1小时
        """
        await sync_to_async(cache.set)(cache_key, data, timeout)
        
    async def delete_from_cache(self, cache_key: str):
        """
        从缓存删除数据
        
        Args:
            cache_key: 缓存键
        """
        await sync_to_async(cache.delete)(cache_key)

    async def clear_cache_pattern(self, pattern: str) -> None:
        """
        清除匹配模式的所有缓存
        
        Args:
            pattern: 缓存键模式
        """
        try:
            from django_redis import get_redis_connection
            
            # 使用sync_to_async包装所有Redis操作
            get_connection = sync_to_async(get_redis_connection)
            redis_client = await get_connection("default")
            
            full_pattern = f"{self.model_name}:{pattern}"
            
            # 使用sync_to_async包装keys操作
            get_keys = sync_to_async(redis_client.keys)
            keys = await get_keys(full_pattern)
            
            if keys:
                # 使用sync_to_async包装delete操作
                delete_keys = sync_to_async(redis_client.delete)
                await delete_keys(*keys)
                logger.debug(f"清除缓存模式: {full_pattern}, 共{len(keys)}个键")
            else:
                logger.debug(f"没有找到匹配模式的缓存: {full_pattern}")
                
        except Exception as e:
            logger.error(f"清除缓存模式错误: {str(e)}")

    def _get_code_field(self, model_class) -> str:
        """
        获取模型的代码字段名称，默认为 'dm'
        
        Args:
            model_class: 模型类
            
        Returns:
            str: 代码字段名称
        """
        return 'dm'
    
    def _process_date_fields(self, data: Dict[str, Any]) -> None:
        """
        处理字典中的日期字段
        
        Args:
            data: 要处理的数据字典
        """
        date_fields = ['sbdq', 'sgr', 'wsfxr', 'fxjg', 'szgsr', 'ssrq']
        for field in date_fields:
            if field in data and data[field] and isinstance(data[field], str):
                try:
                    data[field] = datetime.strptime(data[field], '%Y-%m-%d').date()
                except ValueError:
                    try:
                        data[field] = datetime.strptime(data[field], '%Y%m%d').date()
                    except ValueError:
                        logger.warning(f"无法解析日期字段 {field}: {data[field]}")
                        data[field] = None
    
    # ================= 通用数据访问方法 =================
    
    async def _save_to_db(self, model_class, data: Dict[str, Any]):
        """
        保存数据到数据库，并进行格式转换
        
        Args:
            model_class: 模型类
            data: 要保存的数据
            
        Returns:
            保存后的模型实例
        """
        try:
            # 获取模型对应的代码字段名
            model_code_field = self._get_code_field(model_class)
            
            # 查找映射后的字段名
            code_field = None
            for model_class_name in [StockBasic, CompanyInfo, NewStockCalendar, STStockList]:
                if isinstance(model_class(), model_class_name) or model_class == model_class_name:
                    # 根据模型类选择正确的映射
                    if model_class_name == StockBasic:
                        mapping = STOCK_BASIC_MAPPING
                    elif model_class_name == CompanyInfo:
                        mapping = COMPANY_INFO_MAPPING
                    elif model_class_name == NewStockCalendar:
                        mapping = NEW_STOCK_CALENDAR_MAPPING
                    elif model_class_name == STStockList:
                        mapping = ST_STOCK_LIST_MAPPING
                    
                    # 查找对应的映射字段
                    for api_field, mapped_field in mapping.items():
                        if mapped_field == 'stock_code' or api_field == 'dm' or api_field == 'zqdm':
                            code_field = mapped_field
                            break
                    break
            
            # 如果没有找到映射字段，使用默认字段
            if not code_field:
                code_field = 'stock_code'
            
            # 检查必要的字段是否存在
            if code_field not in data:
                logger.error(f"数据中缺少必要的字段: {code_field}，接收到的数据: {data}")
                return None
                
            stock_code = data[code_field]
            
            # 处理数据格式
            processed_data = {}
            for field, value in data.items():
                # 日期字段处理
                if field.endswith('_date') or field.endswith('_time') or field == 't':
                    processed_data[field] = self._parse_datetime(value)
                # 数值字段处理，但排除 stock_code 字段
                elif (isinstance(value, (int, float)) or (
                    isinstance(value, str) and value.replace('.', '', 1).isdigit()
                )) and field != 'stock_code':
                    processed_data[field] = self._parse_number(value)
                else:
                    processed_data[field] = value
            
            # 定义一个带有原子事务的同步函数
            @sync_to_async
            def update_or_create_item():
                with transaction.atomic():
                    filter_kwargs = {code_field: stock_code}
                    obj, created = model_class.objects.update_or_create(
                        **filter_kwargs,
                        defaults=processed_data
                    )
                    return obj, created
            
            # 调用同步函数
            item, created = await update_or_create_item()
            
            if created:
                logger.debug(f"创建新记录: {model_class.__name__} {stock_code}")
            else:
                logger.debug(f"更新记录: {model_class.__name__} {stock_code}")
            
            return item
        except Exception as e:
            logger.error(f"保存数据出错: {e}")
            logger.debug(f"保存数据出错，数据内容: {data}")
            return None
    
    async def _update_db(self, model_class, stock_code: str, data: Dict[str, Any]):
        """
        更新数据库中的记录，并进行格式转换
        
        Args:
            model_class: 模型类
            stock_code: 股票代码
            data: 要更新的数据
            
        Returns:
            更新后的记录，如不存在则创建新记录并返回
        """
        try:
            code_field = self._get_code_field(model_class)
            
            # 处理数据格式
            processed_data = {}
            for field, value in data.items():
                # 日期字段处理
                if field.endswith('_date') or field.endswith('_time') or field == 't':
                    processed_data[field] = self._parse_datetime(value)
                # 数值字段处理
                elif isinstance(value, (int, float)) or (
                    isinstance(value, str) and value.replace('.', '', 1).isdigit()
                ):
                    processed_data[field] = self._parse_number(value)
                else:
                    processed_data[field] = value
            
            # 定义获取对象的同步函数
            @sync_to_async
            def get_item():
                return model_class.objects.filter(**{code_field: stock_code}).first()
            
            # 查询对象
            item = await get_item()
            
            if item:
                # 定义更新对象的同步函数
                @sync_to_async
                def update_item():
                    for field, value in processed_data.items():
                        setattr(item, field, value)
                    item.save()
                    return item
                
                # 更新对象
                return await update_item()
            else:
                # 定义创建对象的同步函数
                @sync_to_async
                def create_item():
                    processed_data[code_field] = stock_code
                    return model_class.objects.create(**processed_data)
                
                # 创建新对象
                return await create_item()
                
        except Exception as e:
            logger.error(f"更新数据出错: {e}")
            logger.debug(f"更新数据出错，数据内容: {data}")
            return None
    
    async def _get_generic_by_code(self, 
                                  model_class,
                                  stock_code: str, 
                                  cache_key: str, 
                                  api_method: callable,
                                  mapping: Dict[str, str],
                                  code_field: str = 'dm',
                                  process_data: callable = None):
        """
        通用的按代码获取数据的方法，支持缓存和API获取
        
        Args:
            model_class: 模型类
            stock_code: 股票代码
            cache_key: 缓存键
            api_method: API获取方法
            mapping: 字段映射
            code_field: 代码字段名
            process_data: 数据处理函数
            
        Returns:
            模型实例或None
        """
        try:
            # 1. 尝试从缓存获取
            cached_data = await self.get_from_cache(cache_key)
            if cached_data:
                logger.debug(f"缓存命中: {cache_key}")
                return cached_data
            
            logger.debug(f"缓存未命中: {cache_key}")
            
            # 2. 从数据库获取
            @sync_to_async
            def get_db_item():
                return model_class.objects.filter(**{code_field: stock_code}).first()
            
            item = await get_db_item()
            
            if item:
                # 将对象转换为字典格式并更新缓存
                item_dict = {}
                for field in item._meta.fields:
                    value = getattr(item, field.name)
                    # 日期字段处理
                    if field.name.endswith('_date') or field.name.endswith('_time') or field.name == 't':
                        item_dict[field.name] = self._parse_datetime(value)
                    # 数值字段处理
                    elif isinstance(value, (int, float)) or (
                        isinstance(value, str) and value.replace('.', '', 1).isdigit()
                    ):
                        item_dict[field.name] = self._parse_number(value)
                    else:
                        item_dict[field.name] = value
                
                await self.set_to_cache(cache_key, item_dict)
                return item
            
            # 3. 从API获取
            api_data = await api_method(stock_code)
            if not api_data:
                logger.warning(f"API返回数据为空: {stock_code}")
                return None
            
            # 4. 处理API数据
            if process_data:
                api_data = process_data(api_data)
            
            # 5. 映射数据
            model_data = self._map_api_to_model(api_data, mapping)
            if not model_data:
                logger.warning(f"映射后的数据为空: {stock_code}")
                return None
            
            # 6. 保存到数据库
            item = await self._save_to_db(model_class, model_data)
            if not item:
                logger.warning(f"保存数据失败: {stock_code}")
                return None
            
            # 7. 更新缓存
            item_dict = {}
            for field in item._meta.fields:
                value = getattr(item, field.name)
                # 日期字段处理
                if field.name.endswith('_date') or field.name.endswith('_time') or field.name == 't':
                    item_dict[field.name] = self._parse_datetime(value)
                # 数值字段处理
                elif isinstance(value, (int, float)) or (
                    isinstance(value, str) and value.replace('.', '', 1).isdigit()
                ):
                    item_dict[field.name] = self._parse_number(value)
                else:
                    item_dict[field.name] = value
            
            await self.set_to_cache(cache_key, item_dict)
            return item
            
        except Exception as e:
            logger.error(f"获取数据失败: {e}")
            return None
    
    async def _get_generic_all(self, 
                              model_class,
                              cache_key: str, 
                              api_method: callable,
                              mapping: Dict[str, str],
                              process_data: callable = None):
        """
        通用的获取所有数据的方法，支持缓存和API获取
        
        Args:
            model_class: 模型类
            cache_key: 缓存键
            api_method: API获取方法
            mapping: 字段映射
            process_data: 数据处理函数
            
        Returns:
            模型实例列表
        """
        try:
            # 1. 尝试从缓存获取
            cached_data = await self.get_from_cache(cache_key)
            if cached_data:
                logger.debug(f"缓存命中: {cache_key}")
                return cached_data
            
            logger.debug(f"缓存未命中: {cache_key}")
            
            # 2. 从数据库获取
            @sync_to_async
            def get_all_items():
                return list(model_class.objects.all())
            
            items = await get_all_items()
            
            if items:
                # 将对象列表转换为字典格式并更新缓存
                cache_data = []
                for item in items:
                    item_dict = {}
                    for field in item._meta.fields:
                        value = getattr(item, field.name)
                        # 日期字段处理
                        if field.name.endswith('_date') or field.name.endswith('_time') or field.name == 't':
                            item_dict[field.name] = self._parse_datetime(value)
                        # 数值字段处理
                        elif isinstance(value, (int, float)) or (
                            isinstance(value, str) and value.replace('.', '', 1).isdigit()
                        ):
                            item_dict[field.name] = self._parse_number(value)
                        else:
                            item_dict[field.name] = value
                    cache_data.append(item_dict)
                
                await self.set_to_cache(cache_key, cache_data)
                return items
            
            # 3. 从API获取
            api_data_list = await api_method()
            if not api_data_list:
                logger.warning("API返回数据为空")
                return []
            
            # 4. 处理API数据
            if process_data:
                api_data_list = [process_data(item) for item in api_data_list]
            
            # 5. 映射数据
            model_data_list = []
            for api_data in api_data_list:
                model_data = self._map_api_to_model(api_data, mapping)
                if model_data:
                    model_data_list.append(model_data)
            
            if not model_data_list:
                logger.warning("映射后的数据为空")
                return []
            
            # 6. 保存到数据库
            saved_items = []
            for model_data in model_data_list:
                item = await self._save_to_db(model_class, model_data)
                if item:
                    saved_items.append(item)
            
            if not saved_items:
                logger.warning("保存数据失败")
                return []
            
            # 7. 更新缓存
            cache_data = []
            for item in saved_items:
                item_dict = {}
                for field in item._meta.fields:
                    value = getattr(item, field.name)
                    # 日期字段处理
                    if field.name.endswith('_date') or field.name.endswith('_time') or field.name == 't':
                        item_dict[field.name] = self._parse_datetime(value)
                    # 数值字段处理
                    elif isinstance(value, (int, float)) or (
                        isinstance(value, str) and value.replace('.', '', 1).isdigit()
                    ):
                        item_dict[field.name] = self._parse_number(value)
                    else:
                        item_dict[field.name] = value
                cache_data.append(item_dict)
            
            await self.set_to_cache(cache_key, cache_data)
            return saved_items
            
        except Exception as e:
            logger.error(f"获取数据失败: {e}")
            return []
    
    async def _refresh_generic_data(self, 
                                  model_class,
                                  stock_code: Optional[str], 
                                  api_method: callable,
                                  mapping: Dict[str, str],
                                  prefix: str,
                                  process_data: callable = None) -> bool:
        """
        通用的刷新数据方法
        
        Args:
            model_class: 模型类
            stock_code: 股票代码，如果为None则刷新所有数据
            api_method: 获取API数据的方法
            mapping: API数据到模型的映射
            prefix: 缓存键前缀
            process_data: 可选的数据处理方法
            
        Returns:
            bool: 刷新是否成功
        """
        try:
            # 从API获取最新数据
            api_data_list = await api_method()
            if not api_data_list:
                logger.warning("从API获取数据为空")
                return False
                
            # 检查API返回的数据是否是列表
            if not isinstance(api_data_list, list):
                logger.warning(f"API返回的数据类型无效，期望list类型，但得到: {type(api_data_list)}")
                return False
            
            # 如果指定了股票代码，只处理该股票的数据
            if stock_code:
                # 找到目标股票的数据
                target_item = None
                code_field = next((k for k, v in mapping.items() if v == self._get_code_field(model_class)), 'dm')
                
                for item_data in api_data_list:
                    if not isinstance(item_data, dict):
                        continue
                    if item_data.get(code_field) == stock_code:
                        target_item = item_data
                        break
                
                if target_item:
                    # 清除该股票的缓存
                    cache_key = f"{prefix}:{stock_code}"
                    await self.delete_from_cache(cache_key)
                    
                    # 映射并保存数据
                    mapped_data = self._map_api_to_model(target_item, mapping)
                    
                    # 如果映射后的数据为空，返回失败
                    if not mapped_data:
                        logger.warning(f"映射后的数据为空，原始数据: {target_item}")
                        return False
                    
                    # 如果有数据处理方法，则进行处理
                    if process_data:
                        process_data(mapped_data)
                        
                    # 保存到数据库
                    saved_item = await self._save_to_db(model_class, mapped_data)
                    if not saved_item:
                        return False
                        
                    logger.info(f"成功刷新{model_class.__name__}数据: {stock_code}")
                    return True
                else:
                    logger.warning(f"API中未找到该股票: {stock_code}")
                    return False
            else:
                # 刷新所有数据
                # 清除所有相关缓存
                cache_key = f"{prefix}:all"
                await self.delete_from_cache(cache_key)
                
                # 批量保存所有信息
                success_count = 0
                fail_count = 0
                code_field = next((k for k, v in mapping.items() if v == self._get_code_field(model_class)), 'dm')
                
                for item_data in api_data_list:
                    if not isinstance(item_data, dict):
                        fail_count += 1
                        continue
                        
                    # 映射数据
                    mapped_data = self._map_api_to_model(item_data, mapping)
                    
                    # 如果映射后的数据为空，跳过
                    if not mapped_data:
                        fail_count += 1
                        continue
                    
                    # 如果有数据处理方法，则进行处理
                    if process_data:
                        process_data(mapped_data)
                        
                    # 保存到数据库
                    stock_code = item_data.get(code_field)
                    if stock_code:
                        # 清除单个股票的缓存
                        item_cache_key = f"{prefix}:{stock_code}"
                        await self.delete_from_cache(item_cache_key)
                        
                        saved_item = await self._save_to_db(model_class, mapped_data)
                        if saved_item:
                            success_count += 1
                        else:
                            fail_count += 1
                    else:
                        fail_count += 1
                
                logger.info(f"成功刷新{model_class.__name__}数据: 成功 {success_count} 条，失败 {fail_count} 条")
                return success_count > 0
        except Exception as e:
            logger.error(f"刷新数据出错: {e}")
            logger.exception("刷新数据异常")
            return False
    
    # ================= 股票基本信息相关方法 =================
    
    async def get_stock_list(self) -> List[StockBasic]:
        """
        获取所有股票的基本信息
        
        Returns:
            List[StockBasic]: 股票基本信息列表
        """
        cache_key = "stock_basic:all"
        
        # 添加更详细的日志
        logger.info("开始获取股票列表数据")
        
        # 1. 首先尝试从缓存获取
        cached_data = await self.get_from_cache(cache_key)
        if cached_data:
            logger.info(f"从缓存获取股票列表成功，共{len(cached_data)}条数据")
            # 将缓存的字典数据转换回模型对象
            return [StockBasic(**item) for item in cached_data]
        
        # 2. 缓存未命中，从数据库查询
        try:
            # 定义获取所有记录的同步函数
            @sync_to_async
            def get_all_items():
                return list(StockBasic.objects.all())
            
            # 获取所有记录
            items = await get_all_items()
            
            if items:
                logger.info(f"从数据库获取股票列表成功，共{len(items)}条数据")
                # 将模型对象转换为字典并存入缓存
                items_dict = [item.__dict__ for item in items]
                # 移除不需要的字段
                for item_dict in items_dict:
                    item_dict.pop('_state', None)
                await self.set_to_cache(cache_key, items_dict)
                return items
        except Exception as e:
            logger.error(f"从数据库获取股票列表失败: {e}")
        
        # 3. 数据库未找到或为空，从API获取
        try:
            # 直接调用API获取数据
            api_data = await self.api.get_stock_list()
            
            if not api_data:
                logger.warning("从API获取的股票列表为空")
                return []
            
            logger.info(f"从API获取股票列表成功，共{len(api_data)}条数据")
            if api_data:
                logger.debug(f"API返回的第一条数据: {api_data[0]}")
            
            # 批量保存所有信息
            saved_items = []
            
            # 记录成功和失败的数量
            success_count = 0
            failed_count = 0
            
            for item_data in api_data:
                try:
                    # 检查API返回的数据是否有效
                    if not isinstance(item_data, dict):
                        logger.warning(f"API返回的数据类型无效: {type(item_data)}")
                        failed_count += 1
                        continue
                    
                    # 检查是否包含必要的API字段
                    if "dm" not in item_data:
                        logger.warning(f"API返回的数据缺少必要的dm字段: {item_data}")
                        failed_count += 1
                        continue
                    
                    # 映射数据
                    mapped_data = self._map_api_to_model(item_data, STOCK_BASIC_MAPPING)
                    
                    # 如果映射后的数据为空，跳过
                    if not mapped_data:
                        logger.warning(f"映射后的数据为空，原始数据: {item_data}")
                        failed_count += 1
                        continue
                    
                    
                    # 保存到数据库
                    item = await self._save_to_db(StockBasic, mapped_data)
                    if item:  # 只有成功保存的项目才添加到结果列表
                        saved_items.append(item)
                        success_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    logger.error(f"处理单条股票数据时出错: {e}, 数据: {item_data}")
                    failed_count += 1
            
            logger.info(f"股票列表处理完成: 成功 {success_count} 条，失败 {failed_count} 条")
            
            # 保存到缓存
            if saved_items:
                # 将模型对象转换为字典并存入缓存
                items_dict = [item.__dict__ for item in saved_items]
                # 移除不需要的字段
                for item_dict in items_dict:
                    item_dict.pop('_state', None)
                await self.set_to_cache(cache_key, items_dict)
            
            return saved_items
        except Exception as e:
            logger.error(f"从API获取股票列表失败: {e}")
            logger.exception("获取股票列表异常详情")
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
        cache_key = f"new_stock:{stock_code}"
        return await self._get_generic_by_code(
            model_class=NewStockCalendar,
            stock_code=stock_code,
            cache_key=cache_key,
            api_method=self.api.get_new_stock_calendar,
            mapping=NEW_STOCK_CALENDAR_MAPPING,
            process_data=self._process_date_fields
        )
    
    async def get_all_new_stocks(self) -> List[NewStockCalendar]:
        """
        获取所有新股信息
        
        Returns:
            List[NewStockCalendar]: 新股信息列表
        """
        cache_key = "new_stock:all"
        return await self._get_generic_all(
            model_class=NewStockCalendar,
            cache_key=cache_key,
            api_method=self.api.get_new_stock_calendar,
            mapping=NEW_STOCK_CALENDAR_MAPPING,
            process_data=self._process_date_fields
        )
    
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
                    
                    # 将模型对象转换为字典
                    item_dict = {}
                    for field in item._meta.fields:
                        value = getattr(item, field.name)
                        # 日期字段处理
                        if field.name.endswith('_date') or field.name.endswith('_time') or field.name == 't':
                            item_dict[field.name] = self._parse_datetime(value)
                        # 数值字段处理
                        elif isinstance(value, (int, float)) or (
                            isinstance(value, str) and value.replace('.', '', 1).isdigit()
                        ):
                            item_dict[field.name] = self._parse_number(value)
                        else:
                            item_dict[field.name] = value
                    
                    # 更新缓存
                    cache_key = f"new_stock:{data.get('dm')}"
                    await self.set_to_cache(cache_key, item_dict)
            
            # 更新全局缓存
            if saved_items:
                cache_data = []
                for item in saved_items:
                    item_dict = {}
                    for field in item._meta.fields:
                        value = getattr(item, field.name)
                        # 日期字段处理
                        if field.name.endswith('_date') or field.name.endswith('_time') or field.name == 't':
                            item_dict[field.name] = self._parse_datetime(value)
                        # 数值字段处理
                        elif isinstance(value, (int, float)) or (
                            isinstance(value, str) and value.replace('.', '', 1).isdigit()
                        ):
                            item_dict[field.name] = self._parse_number(value)
                        else:
                            item_dict[field.name] = value
                    cache_data.append(item_dict)
                
                cache_key = "new_stock:all"
                await self.set_to_cache(cache_key, cache_data)
            
            return saved_items
            
        except Exception as e:
            logger.error(f'刷新新股数据出错: {str(e)}')
            return []
    
    # ================= ST股票相关方法 =================
    
    async def get_st_stock_by_code(self, stock_code: str) -> Optional[Dict]:
        """
        根据股票代码获取ST股票信息
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Optional[Dict]: ST股票信息字典
        """
        cache_key = f"st_stock:{stock_code}"
        try:
            # 1. 尝试从缓存获取
            cached_data = await self.get_from_cache(cache_key)
            if cached_data:
                logger.debug(f"缓存命中: {cache_key}")
                return cached_data
            
            logger.debug(f"缓存未命中: {cache_key}")
            
            # 2. 从数据库获取
            @sync_to_async
            def get_db_item():
                return STStockList.objects.filter(stock_code=stock_code).first()
            
            item = await get_db_item()
            
            if item:
                # 将对象转换为字典格式并更新缓存
                item_dict = {}
                for field in item._meta.fields:
                    value = getattr(item, field.name)
                    # 日期字段处理
                    if field.name.endswith('_date') or field.name.endswith('_time') or field.name == 't':
                        item_dict[field.name] = self._parse_datetime(value)
                    # 数值字段处理
                    elif isinstance(value, (int, float)) or (
                        isinstance(value, str) and value.replace('.', '', 1).isdigit()
                    ):
                        item_dict[field.name] = self._parse_number(value)
                    else:
                        item_dict[field.name] = value
                
                await self.set_to_cache(cache_key, item_dict)
                return item_dict
            
            # 3. 从API获取
            api_data = await self.api.get_st_stock_list()
            if not api_data:
                logger.warning(f"API返回数据为空: {stock_code}")
                return None
            
            # 4. 映射数据
            model_data = self._map_api_to_model(api_data, ST_STOCK_LIST_MAPPING)
            if not model_data:
                logger.warning(f"映射后的数据为空: {stock_code}")
                return None
            
            # 5. 保存到数据库
            item = await self._save_to_db(STStockList, model_data)
            if not item:
                logger.warning(f"保存数据失败: {stock_code}")
                return None
            
            # 6. 更新缓存
            item_dict = {}
            for field in item._meta.fields:
                value = getattr(item, field.name)
                # 日期字段处理
                if field.name.endswith('_date') or field.name.endswith('_time') or field.name == 't':
                    item_dict[field.name] = self._parse_datetime(value)
                # 数值字段处理
                elif isinstance(value, (int, float)) or (
                    isinstance(value, str) and value.replace('.', '', 1).isdigit()
                ):
                    item_dict[field.name] = self._parse_number(value)
                else:
                    item_dict[field.name] = value
            
            await self.set_to_cache(cache_key, item_dict)
            return item_dict
            
        except Exception as e:
            logger.error(f"获取数据失败: {e}")
            return None
    
    async def get_all_st_stocks(self) -> List[STStockList]:
        """
        获取所有ST股票信息
        
        Returns:
            List[STStockList]: ST股票信息列表
        """
        cache_key = "st_stock:all"
        return await self._get_generic_all(
            model_class=STStockList,
            cache_key=cache_key,
            api_method=self.api.get_st_stock_list,
            mapping=ST_STOCK_LIST_MAPPING
        )
    
    async def refresh_st_stock_data(self) -> List[Dict]:
        """
        刷新ST股票数据
        
        Returns:
            List[Dict]: 刷新后的ST股票数据列表
        """
        try:
            # 从API获取数据
            data_list = await self.api.get_st_stock_list()
            
            if not data_list:
                logger.warning('获取ST股票数据失败')
                return []
            
            # 批量保存到数据库
            saved_items = []
            for data in data_list:
                # 映射数据
                mapped_data = self._map_api_to_model(data, ST_STOCK_LIST_MAPPING)
                
                # 保存到数据库
                item = await self._save_to_db(STStockList, mapped_data)
                saved_items.append(item)
                
                # 更新缓存
                cache_key = f"st_stock:{data.get('dm')}"
                await self.set_to_cache(cache_key, item)
            
            # 更新全局缓存
            cache_key = "st_stock:all"
            await self.set_to_cache(cache_key, saved_items)
            
            return saved_items
            
        except Exception as e:
            logger.error(f'刷新ST股票数据出错: {str(e)}')
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
        cache_key = f"company_info:{stock_code}"
        return await self._get_generic_by_code(
            model_class=CompanyInfo,
            stock_code=stock_code,
            cache_key=cache_key,
            api_method=self.api.get_company_info,
            mapping=COMPANY_INFO_MAPPING
        )
    
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
                    'establishment_date': mapped_data.get('establishment_date', ''),  # 保持为字符串格式
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
                cache_key = f"company_info:{stock_code}"
                # 将模型对象转换为字典并存入缓存
                cache_data = {
                    'stock_code': saved_item.stock_code,
                    'company_name': saved_item.company_name,
                    'company_english_name': saved_item.company_english_name,
                    'market': saved_item.market,
                    'concepts': saved_item.concepts,
                    'listing_date': saved_item.listing_date,
                    'issue_price': saved_item.issue_price,
                    'lead_underwriter': saved_item.lead_underwriter,
                    'establishment_date': saved_item.establishment_date,
                    'registered_capital': saved_item.registered_capital,
                    'institution_type': saved_item.institution_type,
                    'organization_form': saved_item.organization_form
                }
                await self.set_to_cache(cache_key, cache_data)
                
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
