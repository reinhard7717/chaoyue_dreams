import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Union, Any, Tuple
from django.db import transaction
from django.db.models import Q
from django.core.cache import cache
from django.db.models.fields import Field
from asgiref.sync import sync_to_async

from api_manager.apis.datacenter_api import DataCenterAPI
from api_manager.mappings.datacenter_mappings import *
from dao_manager.base_dao import BaseDAO
from stock_models.datacenter.statistics import NewHighStock, NewLowStock, StageHighLow

logger = logging.getLogger('dao')

def get_model_fields(model_class):
    """
    获取模型的字段列表
    
    Args:
        model_class: Django模型类
        
    Returns:
        list: 字段名称列表
    """
    return [field.name for field in model_class._meta.fields]

class StockStatisticsDao(BaseDAO):
    
    # 缓存时间配置（秒）
    CACHE_TIMEOUT = {
        'short': 60,  # 1分钟
        'medium': 300,  # 5分钟
        'long': 3600,  # 1小时
        'daily': 86400,  # 1天
    }

    async def _batch_process(self, model_class, data_list, mapping, unique_fields, **extra_fields) -> Dict:
        """
        简化的通用异步数据存储方法，实现批量处理（检查-创建-更新-略过）
        
        Args:
            model_class: Django模型类
            data_list: 要处理的数据列表，每项都是模型对应的字段字典
            unique_fields: 用于确定唯一记录的字段列表
            
        Returns:
            dict: 包含创建、更新和略过的记录数
        """
        if not data_list:
            logger.warning(f"未提供任何数据用于处理 - {model_class.__name__}")
            return {'创建': 0, '更新': 0, '跳过': 0}
        
        # 如果传入的不是列表，转换为列表
        if not isinstance(data_list, list):
            data_list = [data_list]
        
        # 统计计数
        created_count = 0
        updated_count = 0
        skipped_count = 0
        
        # 批量处理，分组进行以减小事务范围
        batch_size = 1000
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i+batch_size]
            
            @transaction.atomic
            def process_batch():
                nonlocal created_count, updated_count, skipped_count
                
                for item in batch:
                    # 构建查询条件
                    filter_kwargs = {field: item.get(field) for field in unique_fields if field in item}
                    
                    # 查找现有记录
                    existing = model_class.objects.filter(**filter_kwargs).first()
                    
                    if existing:
                        # 检查数据是否有变化
                        has_changes = False
                        for field, value in item.items():
                            if hasattr(existing, field) and getattr(existing, field) != value:
                                has_changes = True
                                break
                        
                        if has_changes:
                            # 更新记录
                            for field, value in item.items():
                                if hasattr(existing, field):
                                    setattr(existing, field, value)
                            existing.save()
                            updated_count += 1
                        else:
                            # 数据相同，略过
                            skipped_count += 1
                    else:
                        # 创建新记录
                        model_class.objects.create(**item)
                        created_count += 1
            
            # 执行批处理
            await sync_to_async(process_batch)()
        
        result = {
            '创建': created_count,
            '更新': updated_count,
            '跳过': skipped_count
        }
    
        logger.info(f"完成{model_class.__name__}数据处理: {result}")
        return result
    
    async def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """
        从缓存获取数据
        
        Args:
            cache_key: 缓存键
            
        Returns:
            Optional[Any]: 缓存的数据，如果不存在则返回None
        """
        return await asyncio.to_thread(lambda: cache.get(cache_key))
    
    async def _set_to_cache(self, cache_key: str, data: Any, timeout: int) -> None:
        """
        将数据存入缓存，并进行格式转换
        
        Args:
            cache_key: 缓存键
            data: 要缓存的数据
            timeout: 过期时间（秒）
        """
        try:
            # 如果数据是Django模型对象，转换为字典并处理格式
            if hasattr(data, '_meta'):
                cache_data = {}
                for field in get_model_fields(data.__class__):
                    value = getattr(data, field)
                    # 日期字段处理
                    if field.endswith('_time') or field.endswith('_date') or field == 't':
                        cache_data[field] = self._parse_datetime(value)
                    # 数值字段处理
                    elif isinstance(value, (int, float)) or (
                        isinstance(value, str) and value.replace('.', '', 1).isdigit()
                    ):
                        cache_data[field] = self._parse_number(value)
                    else:
                        cache_data[field] = value
                data = cache_data
            # 如果数据是Django模型对象列表，转换为字典列表并处理格式
            elif isinstance(data, list) and data and hasattr(data[0], '_meta'):
                cache_data = []
                for item in data:
                    item_dict = {}
                    for field in get_model_fields(item.__class__):
                        value = getattr(item, field)
                        # 日期字段处理
                        if field.endswith('_time') or field.endswith('_date') or field == 't':
                            item_dict[field] = self._parse_datetime(value)
                        # 数值字段处理
                        elif isinstance(value, (int, float)) or (
                            isinstance(value, str) and value.replace('.', '', 1).isdigit()
                        ):
                            item_dict[field] = self._parse_number(value)
                        else:
                            item_dict[field] = value
                    cache_data.append(item_dict)
                data = cache_data
            # 如果数据是字典，处理其中的日期和数值字段
            elif isinstance(data, dict):
                cache_data = {}
                for field, value in data.items():
                    # 日期字段处理
                    if field.endswith('_time') or field.endswith('_date') or field == 't':
                        cache_data[field] = self._parse_datetime(value)
                    # 数值字段处理
                    elif isinstance(value, (int, float)) or (
                        isinstance(value, str) and value.replace('.', '', 1).isdigit()
                    ):
                        cache_data[field] = self._parse_number(value)
                    else:
                        cache_data[field] = value
                data = cache_data
            # 如果数据是字典列表，处理每个字典中的日期和数值字段
            elif isinstance(data, list) and data and isinstance(data[0], dict):
                cache_data = []
                for item in data:
                    item_dict = {}
                    for field, value in item.items():
                        # 日期字段处理
                        if field.endswith('_time') or field.endswith('_date') or field == 't':
                            item_dict[field] = self._parse_datetime(value)
                        # 数值字段处理
                        elif isinstance(value, (int, float)) or (
                            isinstance(value, str) and value.replace('.', '', 1).isdigit()
                        ):
                            item_dict[field] = self._parse_number(value)
                        else:
                            item_dict[field] = value
                    cache_data.append(item_dict)
                data = cache_data
            
            # 存入缓存
            await asyncio.to_thread(lambda: cache.set(cache_key, data, timeout))
            
        except Exception as e:
            logger.error(f"缓存数据时出错: {str(e)}")
            logger.debug(f"缓存数据出错，数据内容: {data}")
            raise

    
    async def save_stage_high_low(self) -> Dict:
        """保存阶段高低点数据"""
        try:
            api = DataCenterAPI()
            response = await api.get_stage_high_low()
            
            # 处理text/plain或text/html格式的响应
            if isinstance(response, str):
                try:
                    # 尝试解析为JSON
                    import json
                    data_list = json.loads(response)
                except json.JSONDecodeError:
                    logger.error("解析阶段高低点数据失败，返回的不是有效的JSON格式")
                    return {'创建': 0, '更新': 0, '跳过': 0}
            else:
                data_list = response
            
            if not data_list:
                logger.warning("未获取到阶段高低点数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 处理数据，使用BaseDAO中的_parse_datetime和_parse_number方法
            processed_data_list = []
            for data in data_list:
                # 将API数据映射为标准字典，用于保存到数据库
                mapped_data = {}
                for api_field, model_field in STAGE_HIGH_LOW_MAPPING.items():
                    if api_field in data:
                        value = data[api_field]
                        # 处理日期字段
                        if model_field.endswith('_time') or model_field.endswith('_date') or model_field == 'trade_date':
                            mapped_data[model_field] = self._parse_datetime(value)
                        # 处理数值字段
                        elif isinstance(value, (int, float)) or (isinstance(value, str) and value.replace('.', '', 1).replace('-', '', 1).isdigit()):
                            mapped_data[model_field] = self._parse_number(value)
                        else:
                            mapped_data[model_field] = value
                
                processed_data_list.append(mapped_data)
            
            # 保存到数据库
            result = await self._batch_process(
                model_class=StageHighLow,
                data_list=processed_data_list,
                mapping=STAGE_HIGH_LOW_MAPPING,
                unique_fields=['stock_code', 'trade_date']
            )
            
            # 更新缓存
            if processed_data_list:
                # 获取日期用于缓存键
                date_str = processed_data_list[0]['trade_date'].strftime('%Y-%m-%d') if isinstance(processed_data_list[0]['trade_date'], datetime) else processed_data_list[0]['trade_date']
                cache_key = f'stage_high_low_{date_str}'
                
                # 转换为字典格式后存入缓存
                cache_data = []
                for item in processed_data_list:
                    cache_data.append(item)
                
                await self._set_to_cache(cache_key, cache_data, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"保存阶段高低点数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}
    
    async def save_new_high_stocks(self) -> Dict:
        """
        保存盘中创新高个股数据
        
        Returns:
            dict: 操作结果统计
        """
        try:
            # 从API获取数据
            api = DataCenterAPI()
            response = await api.get_new_high_stocks()
            
            # 处理text/plain或text/html格式的响应
            if isinstance(response, str):
                try:
                    # 尝试解析为JSON
                    import json
                    data_list = json.loads(response)
                except json.JSONDecodeError:
                    logger.error("解析盘中创新高个股数据失败，返回的不是有效的JSON格式")
                    return {'创建': 0, '更新': 0, '跳过': 0}
            else:
                data_list = response
            
            if not data_list:
                logger.warning("未获取到盘中创新高个股数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 处理数据，使用BaseDAO中的_parse_datetime和_parse_number方法
            processed_data_list = []
            for data in data_list:
                # 将API数据映射为标准字典，用于保存到数据库
                mapped_data = {}
                for api_field, model_field in NEW_HIGH_STOCK_MAPPING.items():
                    if api_field in data:
                        value = data[api_field]
                        # 处理日期字段
                        if model_field.endswith('_time') or model_field.endswith('_date') or model_field == 'trade_date':
                            mapped_data[model_field] = self._parse_datetime(value)
                        # 处理数值字段
                        elif isinstance(value, (int, float)) or (isinstance(value, str) and value.replace('.', '', 1).replace('-', '', 1).isdigit()):
                            mapped_data[model_field] = self._parse_number(value)
                        else:
                            mapped_data[model_field] = value
                
                processed_data_list.append(mapped_data)
            
            # 保存数据
            result = await self._batch_process(
                model_class=NewHighStock,
                data_list=processed_data_list,
                mapping=NEW_HIGH_STOCK_MAPPING,
                unique_fields=['stock_code', 'trade_date']
            )
            
            # 更新缓存
            cache_key = 'new_high_stocks'
            await self._set_to_cache(cache_key, processed_data_list, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"保存盘中创新高个股数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}
    
    
    async def get_new_high_stocks(self) -> List[Dict]:
        """
        获取盘中创新高个股数据
        
        Returns:
            List[Dict]: 盘中创新高个股数据列表
        """
        # 先查缓存
        cache_key = 'new_high_stocks'
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 缓存未命中，查数据库
        try:
            # 获取最新日期的数据
            latest_date = await sync_to_async(lambda: NewHighStock.objects.order_by('-trade_date').values('trade_date').first())()
            
            if latest_date:
                records = await sync_to_async(lambda: list(NewHighStock.objects.filter(trade_date=latest_date['trade_date']).order_by('-change_rate')))()
                
                if records:
                    result = [{field: getattr(record, field) for field in get_model_fields(NewHighStock)} 
                             for record in records]
                    await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['short'])
                    return result
            
            return []
        except Exception as e:
            logger.error(f"获取盘中创新高个股数据出错: {str(e)}")
            return []
       
    async def save_new_low_stocks(self) -> Dict:
        """
        保存盘中创新低个股数据
        
        Returns:
            dict: 操作结果统计
        """
        try:
            # 从API获取数据
            api = DataCenterAPI()
            response = await api.get_new_low_stocks()
            
            # 处理text/plain或text/html格式的响应
            if isinstance(response, str):
                try:
                    # 尝试解析为JSON
                    import json
                    data_list = json.loads(response)
                except json.JSONDecodeError:
                    logger.error("解析盘中创新低个股数据失败，返回的不是有效的JSON格式")
                    return {'创建': 0, '更新': 0, '跳过': 0}
            else:
                data_list = response
            
            if not data_list:
                logger.warning("未获取到盘中创新低个股数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 处理数据，使用BaseDAO中的_parse_datetime和_parse_number方法
            processed_data_list = []
            for data in data_list:
                # 将API数据映射为标准字典，用于保存到数据库
                mapped_data = {}
                for api_field, model_field in NEW_LOW_STOCK_MAPPING.items():
                    if api_field in data:
                        value = data[api_field]
                        # 处理日期字段
                        if model_field.endswith('_time') or model_field.endswith('_date') or model_field == 'trade_date':
                            mapped_data[model_field] = self._parse_datetime(value)
                        # 处理数值字段
                        elif isinstance(value, (int, float)) or (isinstance(value, str) and value.replace('.', '', 1).replace('-', '', 1).isdigit()):
                            mapped_data[model_field] = self._parse_number(value)
                        else:
                            mapped_data[model_field] = value
                
                processed_data_list.append(mapped_data)
            
            # 保存数据
            result = await self._batch_process(
                model_class=NewLowStock,
                data_list=processed_data_list,
                mapping=NEW_LOW_STOCK_MAPPING,
                unique_fields=['stock_code', 'trade_date']
            )
            
            # 更新缓存
            cache_key = 'new_low_stocks'
            await self._set_to_cache(cache_key, processed_data_list, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"保存盘中创新低个股数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}
    
    async def get_new_low_stocks(self) -> List[Dict]:
        """
        获取盘中创新低个股数据
        
        Returns:
            List[Dict]: 盘中创新低个股数据列表
        """
        # 先查缓存
        cache_key = 'new_low_stocks'
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 缓存未命中，查数据库
        try:
            # 获取最新日期的数据
            latest_date = await sync_to_async(lambda: NewLowStock.objects.order_by('-trade_date').values('trade_date').first())()
            
            if latest_date:
                records = await sync_to_async(lambda: list(NewLowStock.objects.filter(trade_date=latest_date['trade_date']).order_by('change_rate')))()
                
                if records:
                    result = [{field: getattr(record, field) for field in get_model_fields(NewLowStock)} 
                             for record in records]
                    await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['short'])
                    return result
            
            return []
        except Exception as e:
            logger.error(f"获取盘中创新低个股数据出错: {str(e)}")
            return []

