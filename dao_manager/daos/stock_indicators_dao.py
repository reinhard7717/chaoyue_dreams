import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Set, Tuple, Type
from datetime import datetime, date
from decimal import Decimal

from django.db import transaction
from django.core.cache import cache
from django.db.models import Q
from django.db.models.base import Model

from api_manager.apis.stock_indicators_api import StockIndicatorsAPI, TimeLevel
from api_manager.mappings.stock_indicators import BOLL_INDICATOR_MAPPING, KDJ_INDICATOR_MAPPING, MA_INDICATOR_MAPPING, MACD_INDICATOR_MAPPING, TIME_TRADE_MAPPING
from dao_manager.base_dao import BaseDAO
from stock_models.stock_indicators import BOLLIndicator, KDJIndicator, MACDIndicator, MAIndicator, TimeTrade


logger = logging.getLogger(__name__)

class StockIndicatorsDAO(BaseDAO):
    """
    股票技术指标DAO，整合所有相关的技术指标访问功能
    """
    
    def __init__(self):
        """初始化StockIndicatorsDAO"""
        super().__init__(None, None, 3600)  # 基类使用None作为model_class，因为本DAO管理多个模型
        self.api = StockIndicatorsAPI()
        self.cache_timeout = 300  # 默认缓存5分钟
        logger.info("初始化StockIndicatorsDAO")
    
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
        mapped_data = {}
        for api_field, model_field in mapping.items():
            if api_field in api_data and api_data[api_field] is not None:
                value = api_data[api_field]
                # 日期字段处理
                if model_field.endswith('_date') or model_field.endswith('_time') or model_field == 't':
                    mapped_data[model_field] = self._parse_datetime(value)
                # 数值字段处理
                elif (isinstance(value, (int, float)) or (
                    isinstance(value, str) and value.replace('.', '', 1).isdigit()
                )) and 'code' not in model_field.lower():
                    mapped_data[model_field] = self._parse_number(value)
                else:
                    mapped_data[model_field] = value
        return mapped_data
    
    def _parse_datetime(self, datetime_str: str) -> datetime:
        """
        解析日期时间字符串
        
        Args:
            datetime_str: 日期时间字符串，格式可能是"yyyy-MM-dd HH:mm:ss"或"yyyy-MM-dd"
            
        Returns:
            datetime: 解析后的日期时间对象
        """
        try:
            if len(datetime_str) > 10:  # 包含时间部分
                return datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
            else:  # 仅包含日期部分
                return datetime.strptime(datetime_str, '%Y-%m-%d')
        except (ValueError, TypeError) as e:
            logger.warning(f"日期时间格式解析失败: {datetime_str}, 错误: {e}")
            return datetime.now()  # 返回当前时间作为默认值
    
    async def get_from_cache(self, cache_key: str):
        """
        从缓存获取数据
        
        Args:
            cache_key: 缓存键
            
        Returns:
            缓存的数据，如不存在则返回None
        """
        cached_data = cache.get(cache_key)
        if cached_data:
            return cached_data
        return None
        
    async def set_to_cache(self, cache_key: str, data, timeout: int = None):
        """
        设置数据到缓存
        
        Args:
            cache_key: 缓存键
            data: 要缓存的数据
            timeout: 缓存超时时间(秒)，默认使用类的cache_timeout
        """
        if timeout is None:
            timeout = self.cache_timeout
        cache.set(cache_key, data, timeout)
        
    async def delete_from_cache(self, cache_key: str):
        """
        从缓存删除数据
        
        Args:
            cache_key: 缓存键
        """
        cache.delete(cache_key)
    
    async def _check_if_exists(self, model_class: Type[Model], stock_code: str, time_level: str, trade_time: datetime) -> Optional[Model]:
        """
        检查指定条件的记录是否存在
        
        Args:
            model_class: 模型类
            stock_code: 股票代码
            time_level: 分时级别
            trade_time: 交易时间
            
        Returns:
            Optional[Model]: 如果存在返回记录对象，否则返回None
        """
        try:
            return await model_class.objects.filter(
                stock_code=stock_code,
                time_level=time_level,
                trade_time=trade_time
            ).afirst()
        except Exception as e:
            logger.error(f"检查记录是否存在出错: {e}")
            return None
    
    async def _check_if_data_identical(self, existing: Model, new_data: Dict[str, Any]) -> bool:
        """
        检查现有记录与新数据是否相同
        
        Args:
            existing: 现有记录对象
            new_data: 新数据字典
            
        Returns:
            bool: 如果数据相同返回True，否则返回False
        """
        for field, value in new_data.items():
            if field not in ('stock_code', 'time_level', 'trade_time', 'created_at', 'updated_at'):
                if hasattr(existing, field):
                    existing_value = getattr(existing, field)
                    
                    # 处理Decimal类型的比较
                    if isinstance(existing_value, Decimal) and value is not None:
                        if existing_value != Decimal(str(value)):
                            return False
                    # 常规比较
                    elif existing_value != value:
                        return False
        
        return True
    
    @transaction.atomic
    async def _save_or_update(self, model_class: Type[Model], data: Dict[str, Any]) -> Model:
        """
        保存或更新单条记录
        
        Args:
            model_class: 模型类
            data: 模型数据
            
        Returns:
            Model: 保存或更新后的记录对象
        """
        stock_code = data['stock_code']
        time_level = data['time_level']
        trade_time = data['trade_time']
        
        # 查找是否已存在记录
        existing = await self._check_if_exists(model_class, stock_code, time_level, trade_time)
        
        if existing:
            # 检查数据是否有变化
            if not await self._check_if_data_identical(existing, data):
                # 数据不同，更新记录
                for field, value in data.items():
                    if field not in ('created_at',):
                        setattr(existing, field, value)
                
                await existing.asave()
                logger.debug(f"更新记录: {stock_code}-{time_level}-{trade_time}")
            else:
                logger.debug(f"记录数据相同，跳过更新: {stock_code}-{time_level}-{trade_time}")
            
            return existing
        else:
            # 创建新记录
            instance = model_class(**data)
            await instance.asave()
            logger.debug(f"创建新记录: {stock_code}-{time_level}-{trade_time}")
            return instance
    
    async def _batch_save_or_update(self, model_class: Type[Model], data_list: List[Dict[str, Any]]) -> List[Model]:
        """
        批量保存或更新记录
        
        Args:
            model_class: 模型类
            data_list: 模型数据列表
            
        Returns:
            List[Model]: 保存或更新后的记录对象列表
        """
        if not data_list:
            return []
        
        # 提取所有记录的关键属性
        stock_codes = set()
        time_levels = set()
        trade_times = set()
        
        for data in data_list:
            stock_codes.add(data['stock_code'])
            time_levels.add(data['time_level'])
            trade_times.add(data['trade_time'])
        
        # 查询已存在的记录
        existing_records = {}
        try:
            queryset = model_class.objects.filter(
                stock_code__in=list(stock_codes),
                time_level__in=list(time_levels),
                trade_time__in=list(trade_times)
            )
            
            async for record in queryset:
                key = (record.stock_code, record.time_level, record.trade_time)
                existing_records[key] = record
        except Exception as e:
            logger.error(f"批量查询已存在记录出错: {e}")
        
        # 分类处理数据
        to_create = []
        to_update = []
        unchanged = []
        
        for data in data_list:
            key = (data['stock_code'], data['time_level'], data['trade_time'])
            
            if key in existing_records:
                existing = existing_records[key]
                if not await self._check_if_data_identical(existing, data):
                    # 需要更新的记录
                    for field, value in data.items():
                        if field not in ('created_at',):
                            setattr(existing, field, value)
                    to_update.append(existing)
                else:
                    # 无需更新的记录
                    unchanged.append(existing)
            else:
                # 需要创建的记录
                to_create.append(model_class(**data))
        
        # 批量创建新记录
        created_records = []
        if to_create:
            try:
                created_records = await model_class.objects.abulk_create(to_create)
                logger.info(f"批量创建记录: {len(created_records)}条")
            except Exception as e:
                logger.error(f"批量创建记录出错: {e}")
                # 回退到单条创建
                for item in to_create:
                    try:
                        await item.asave()
                        created_records.append(item)
                    except Exception as sub_e:
                        logger.error(f"单条创建记录出错: {sub_e}")
        
        # 批量更新记录
        updated_records = []
        if to_update:
            try:
                for record in to_update:
                    await record.asave()
                    updated_records.append(record)
                logger.info(f"批量更新记录: {len(updated_records)}条")
            except Exception as e:
                logger.error(f"批量更新记录出错: {e}")
        
        # 合并结果
        result = created_records + updated_records + unchanged
        return result
    
    async def _get_latest_indicator(self, 
                                   model_class: Type[Model], 
                                   stock_code: str, 
                                   time_level: Union[TimeLevel, str],
                                   api_method: callable, 
                                   mapping: Dict[str, str],
                                   cache_prefix: str) -> Optional[Model]:
        """
        获取最新指标数据
        
        Args:
            model_class: 模型类
            stock_code: 股票代码
            time_level: 时间级别
            api_method: API方法
            mapping: 字段映射
            cache_prefix: 缓存前缀
            
        Returns:
            Optional[Model]: 最新指标记录
        """
        # 确保time_level是字符串类型
        if isinstance(time_level, TimeLevel):
            time_level = time_level.value
        
        # 1. 从缓存获取
        cache_key = f"{cache_prefix}:{stock_code}:{time_level}:latest"
        cached_data = await self.get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 2. 从数据库获取
        try:
            latest = await model_class.objects.filter(
                stock_code=stock_code,
                time_level=time_level
            ).order_by('-trade_time').afirst()
            
            if latest:
                # 将对象转换为字典格式并存入缓存
                cache_dict = {}
                for field in latest._meta.fields:
                    value = getattr(latest, field.name)
                    # 日期字段处理
                    if field.name.endswith('_date') or field.name.endswith('_time') or field.name == 't':
                        cache_dict[field.name] = self._parse_datetime(value)
                    # 数值字段处理
                    elif (isinstance(value, (int, float)) or (
                        isinstance(value, str) and value.replace('.', '', 1).isdigit()
                    )) and 'code' not in field.lower():
                        cache_dict[field.name] = self._parse_number(value)
                    else:
                        cache_dict[field.name] = value
                await self.set_to_cache(cache_key, cache_dict)
                return latest
        except Exception as e:
            logger.error(f"从数据库获取最新指标出错: {e}")
        
        # 3. 从API获取
        try:
            api_data = await api_method(stock_code, time_level)
            
            if not api_data:
                logger.warning(f"API未返回数据: {stock_code} {time_level}")
                return None
            
            # 处理数据并保存
            processed_items = []
            for item in api_data:
                mapped_data = self._map_api_to_model(item, mapping)
                mapped_data['stock_code'] = stock_code
                mapped_data['time_level'] = time_level
                processed_items.append(mapped_data)
            
            # 批量保存
            if processed_items:
                saved_items = await self._batch_save_or_update(model_class, processed_items)
                if saved_items:
                    # 查找并返回最新记录
                    latest = max(saved_items, key=lambda x: x.trade_time)
                    # 将对象转换为字典格式并存入缓存
                    cache_dict = {}
                    for field in latest._meta.fields:
                        value = getattr(latest, field.name)
                        # 日期字段处理
                        if field.name.endswith('_date') or field.name.endswith('_time') or field.name == 't':
                            cache_dict[field.name] = self._parse_datetime(value)
                        # 数值字段处理
                        elif (isinstance(value, (int, float)) or (
                            isinstance(value, str) and value.replace('.', '', 1).isdigit()
                        )) and 'code' not in field.lower():
                            cache_dict[field.name] = self._parse_number(value)
                        else:
                            cache_dict[field.name] = value
                    await self.set_to_cache(cache_key, cache_dict)
                    return latest
        except Exception as e:
            logger.error(f"从API获取最新指标出错: {e}")
        
        return None
    
    async def _get_history_indicators(self, 
                                    model_class: Type[Model], 
                                    stock_code: str, 
                                    time_level: Union[TimeLevel, str],
                                    api_method: callable, 
                                    mapping: Dict[str, str],
                                    cache_prefix: str,
                                    limit: int = 1000) -> List[Model]:
        """
        获取历史指标数据
        
        Args:
            model_class: 模型类
            stock_code: 股票代码
            time_level: 时间级别
            api_method: API方法
            mapping: 字段映射
            cache_prefix: 缓存前缀
            limit: 限制返回记录数量
            
        Returns:
            List[Model]: 历史指标记录列表
        """
        # 确保time_level是字符串类型
        if isinstance(time_level, TimeLevel):
            time_level = time_level.value
        
        # 1. 从缓存获取
        cache_key = f"{cache_prefix}:{stock_code}:{time_level}:history:{limit}"
        cached_data = await self.get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 2. 从数据库获取
        try:
            history_data = await model_class.objects.filter(
                stock_code=stock_code,
                time_level=time_level
            ).order_by('-trade_time')[:limit].all()
            
            if history_data and len(history_data) > 0:
                # 将对象列表转换为字典列表并存入缓存
                cache_data = []
                for item in history_data:
                    item_dict = {}
                    for field in item._meta.fields:
                        value = getattr(item, field.name)
                        # 日期字段处理
                        if field.name.endswith('_date') or field.name.endswith('_time') or field.name == 't':
                            item_dict[field.name] = self._parse_datetime(value)
                        # 数值字段处理
                        elif (isinstance(value, (int, float)) or (
                            isinstance(value, str) and value.replace('.', '', 1).isdigit()
                        )) and 'code' not in field.lower():
                            item_dict[field.name] = self._parse_number(value)
                        else:
                            item_dict[field.name] = value
                    cache_data.append(item_dict)
                await self.set_to_cache(cache_key, cache_data)
                return history_data
        except Exception as e:
            logger.error(f"从数据库获取历史指标出错: {e}")
            history_data = []
        
        # 3. 如果数据库没有足够数据，从API获取
        if not history_data or len(history_data) < limit/2:  # 如果数据不到一半，从API获取
            try:
                api_data = await api_method(stock_code, time_level)
                
                if api_data:
                    # 处理数据并保存
                    processed_items = []
                    for item in api_data:
                        mapped_data = self._map_api_to_model(item, mapping)
                        mapped_data['stock_code'] = stock_code
                        mapped_data['time_level'] = time_level
                        processed_items.append(mapped_data)
                    
                    # 批量保存
                    if processed_items:
                        await self._batch_save_or_update(model_class, processed_items)
                        
                        # 重新查询数据库
                        history_data = await model_class.objects.filter(
                            stock_code=stock_code,
                            time_level=time_level
                        ).order_by('-trade_time')[:limit].all()
                        
                        if history_data:
                            # 将对象列表转换为字典列表并存入缓存
                            cache_data = []
                            for item in history_data:
                                item_dict = {}
                                for field in item._meta.fields:
                                    value = getattr(item, field.name)
                                    # 日期字段处理
                                    if field.name.endswith('_date') or field.name.endswith('_time') or field.name == 't':
                                        item_dict[field.name] = self._parse_datetime(value)
                                    # 数值字段处理
                                    elif (isinstance(value, (int, float)) or (
                                        isinstance(value, str) and value.replace('.', '', 1).isdigit()
                                    )) and 'code' not in field.lower():
                                        item_dict[field.name] = self._parse_number(value)
                                    else:
                                        item_dict[field.name] = value
                                cache_data.append(item_dict)
                            await self.set_to_cache(cache_key, cache_data)
                            return history_data
            except Exception as e:
                logger.error(f"从API获取历史指标出错: {e}")
        
        return history_data
    
    async def _refresh_indicator(self, 
                               model_class: Type[Model], 
                               stock_code: str, 
                               time_level: Union[TimeLevel, str],
                               api_method: callable, 
                               mapping: Dict[str, str],
                               cache_prefix: str) -> Optional[Model]:
        """
        刷新指标数据
        
        Args:
            model_class: 模型类
            stock_code: 股票代码
            time_level: 时间级别
            api_method: API方法
            mapping: 字段映射
            cache_prefix: 缓存前缀
            
        Returns:
            Optional[Model]: 刷新后的最新指标记录
        """
        # 确保time_level是字符串类型
        if isinstance(time_level, TimeLevel):
            time_level = time_level.value
        
        # 清除相关缓存
        latest_cache_key = f"{cache_prefix}:{stock_code}:{time_level}:latest"
        history_cache_key = f"{cache_prefix}:{stock_code}:{time_level}:history:"
        await self.delete_from_cache(latest_cache_key)
        # 所有history缓存可能有不同的limit，使用通配符删除
        for key in cache.keys(f"{history_cache_key}*"):
            await self.delete_from_cache(key)
        
        # 从API获取最新数据
        try:
            api_data = await api_method(stock_code, time_level)
            
            if not api_data:
                logger.warning(f"API未返回数据: {stock_code} {time_level}")
                return None
            
            # 处理数据并保存
            processed_items = []
            for item in api_data:
                mapped_data = self._map_api_to_model(item, mapping)
                mapped_data['stock_code'] = stock_code
                mapped_data['time_level'] = time_level
                processed_items.append(mapped_data)
            
            # 批量保存
            if processed_items:
                saved_items = await self._batch_save_or_update(model_class, processed_items)
                if saved_items:
                    # 查找并返回最新记录
                    latest = max(saved_items, key=lambda x: x.trade_time)
                    await self.set_to_cache(latest_cache_key, latest)
                    return latest
        except Exception as e:
            logger.error(f"刷新指标数据出错: {e}")
        
        return None
    
    # ================= 分时成交数据相关方法 =================
    
    async def get_latest_time_trade(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Optional[TimeTrade]:
        """
        获取最新的分时成交数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            
        Returns:
            Optional[TimeTrade]: 最新的分时成交数据
        """
        return await self._get_latest_indicator(
            model_class=TimeTrade,
            stock_code=stock_code,
            time_level=time_level,
            api_method=self.api.get_time_trade_data,
            mapping=TIME_TRADE_MAPPING,
            cache_prefix="time_trade"
        )
    
    async def get_history_time_trades(self, stock_code: str, time_level: Union[TimeLevel, str], 
                                    limit: int = 1000) -> List[TimeTrade]:
        """
        获取历史分时成交数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            limit: 返回记录数量限制
            
        Returns:
            List[TimeTrade]: 历史分时成交数据列表
        """
        return await self._get_history_indicators(
            model_class=TimeTrade,
            stock_code=stock_code,
            time_level=time_level,
            api_method=self.api.get_time_trade_data,
            mapping=TIME_TRADE_MAPPING,
            cache_prefix="time_trade",
            limit=limit
        )
    
    async def refresh_time_trade(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Optional[TimeTrade]:
        """
        刷新分时成交数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            
        Returns:
            Optional[TimeTrade]: 最新的分时成交数据
        """
        return await self._refresh_indicator(
            model_class=TimeTrade,
            stock_code=stock_code,
            time_level=time_level,
            api_method=self.api.get_time_trade_data,
            mapping=TIME_TRADE_MAPPING,
            cache_prefix="time_trade"
        )
    
    # ================= KDJ指标相关方法 =================
    
    async def get_latest_kdj(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Optional[KDJIndicator]:
        """
        获取最新的KDJ指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            
        Returns:
            Optional[KDJIndicator]: 最新的KDJ指标数据
        """
        return await self._get_latest_indicator(
            model_class=KDJIndicator,
            stock_code=stock_code,
            time_level=time_level,
            api_method=self.api.get_kdj_data,
            mapping=KDJ_INDICATOR_MAPPING,
            cache_prefix="kdj"
        )
    
    async def get_history_kdj(self, stock_code: str, time_level: Union[TimeLevel, str], 
                             limit: int = 1000) -> List[KDJIndicator]:
        """
        获取历史KDJ指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            limit: 返回记录数量限制
            
        Returns:
            List[KDJIndicator]: 历史KDJ指标数据列表
        """
        return await self._get_history_indicators(
            model_class=KDJIndicator,
            stock_code=stock_code,
            time_level=time_level,
            api_method=self.api.get_kdj_data,
            mapping=KDJ_INDICATOR_MAPPING,
            cache_prefix="kdj",
            limit=limit
        )
    
    async def refresh_kdj(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Optional[KDJIndicator]:
        """
        刷新KDJ指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            
        Returns:
            Optional[KDJIndicator]: 最新的KDJ指标数据
        """
        return await self._refresh_indicator(
            model_class=KDJIndicator,
            stock_code=stock_code,
            time_level=time_level,
            api_method=self.api.get_kdj_data,
            mapping=KDJ_INDICATOR_MAPPING,
            cache_prefix="kdj"
        )
    
    # ================= MACD指标相关方法 =================
    
    async def get_latest_macd(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Optional[MACDIndicator]:
        """
        获取最新的MACD指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            
        Returns:
            Optional[MACDIndicator]: 最新的MACD指标数据
        """
        return await self._get_latest_indicator(
            model_class=MACDIndicator,
            stock_code=stock_code,
            time_level=time_level,
            api_method=self.api.get_macd_data,
            mapping=MACD_INDICATOR_MAPPING,
            cache_prefix="macd"
        )
    
    async def get_history_macd(self, stock_code: str, time_level: Union[TimeLevel, str], 
                             limit: int = 1000) -> List[MACDIndicator]:
        """
        获取历史MACD指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            limit: 返回记录数量限制
            
        Returns:
            List[MACDIndicator]: 历史MACD指标数据列表
        """
        return await self._get_history_indicators(
            model_class=MACDIndicator,
            stock_code=stock_code,
            time_level=time_level,
            api_method=self.api.get_macd_data,
            mapping=MACD_INDICATOR_MAPPING,
            cache_prefix="macd",
            limit=limit
        )
    
    async def refresh_macd(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Optional[MACDIndicator]:
        """
        刷新MACD指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            
        Returns:
            Optional[MACDIndicator]: 最新的MACD指标数据
        """
        return await self._refresh_indicator(
            model_class=MACDIndicator,
            stock_code=stock_code,
            time_level=time_level,
            api_method=self.api.get_macd_data,
            mapping=MACD_INDICATOR_MAPPING,
            cache_prefix="macd"
        )
    
    # ================= MA指标相关方法 =================
    
    async def get_latest_ma(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Optional[MAIndicator]:
        """
        获取最新的MA指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            
        Returns:
            Optional[MAIndicator]: 最新的MA指标数据
        """
        return await self._get_latest_indicator(
            model_class=MAIndicator,
            stock_code=stock_code,
            time_level=time_level,
            api_method=self.api.get_ma_data,
            mapping=MA_INDICATOR_MAPPING,
            cache_prefix="ma"
        )
    
    async def get_history_ma(self, stock_code: str, time_level: Union[TimeLevel, str], 
                           limit: int = 1000) -> List[MAIndicator]:
        """
        获取历史MA指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            limit: 返回记录数量限制
            
        Returns:
            List[MAIndicator]: 历史MA指标数据列表
        """
        return await self._get_history_indicators(
            model_class=MAIndicator,
            stock_code=stock_code,
            time_level=time_level,
            api_method=self.api.get_ma_data,
            mapping=MA_INDICATOR_MAPPING,
            cache_prefix="ma",
            limit=limit
        )
    
    async def refresh_ma(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Optional[MAIndicator]:
        """
        刷新MA指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            
        Returns:
            Optional[MAIndicator]: 最新的MA指标数据
        """
        return await self._refresh_indicator(
            model_class=MAIndicator,
            stock_code=stock_code,
            time_level=time_level,
            api_method=self.api.get_ma_data,
            mapping=MA_INDICATOR_MAPPING,
            cache_prefix="ma"
        )
    
    # ================= BOLL指标相关方法 =================
    
    async def get_latest_boll(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Optional[BOLLIndicator]:
        """
        获取最新的BOLL指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            
        Returns:
            Optional[BOLLIndicator]: 最新的BOLL指标数据
        """
        return await self._get_latest_indicator(
            model_class=BOLLIndicator,
            stock_code=stock_code,
            time_level=time_level,
            api_method=self.api.get_boll_data,
            mapping=BOLL_INDICATOR_MAPPING,
            cache_prefix="boll"
        )
    
    async def get_history_boll(self, stock_code: str, time_level: Union[TimeLevel, str], 
                             limit: int = 1000) -> List[BOLLIndicator]:
        """
        获取历史BOLL指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            limit: 返回记录数量限制
            
        Returns:
            List[BOLLIndicator]: 历史BOLL指标数据列表
        """
        return await self._get_history_indicators(
            model_class=BOLLIndicator,
            stock_code=stock_code,
            time_level=time_level,
            api_method=self.api.get_boll_data,
            mapping=BOLL_INDICATOR_MAPPING,
            cache_prefix="boll",
            limit=limit
        )
    
    async def refresh_boll(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Optional[BOLLIndicator]:
        """
        刷新BOLL指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            
        Returns:
            Optional[BOLLIndicator]: 最新的BOLL指标数据
        """
        return await self._refresh_indicator(
            model_class=BOLLIndicator,
            stock_code=stock_code,
            time_level=time_level,
            api_method=self.api.get_boll_data,
            mapping=BOLL_INDICATOR_MAPPING,
            cache_prefix="boll"
        )
