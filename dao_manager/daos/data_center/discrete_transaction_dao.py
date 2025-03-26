import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Any
from django.db import transaction
from django.db.models import Q
from django.core.cache import cache
from django.db.models.fields import Field
from asgiref.sync import sync_to_async

from api_manager.apis.datacenter_api import DataCenterAPI
from api_manager.mappings.datacenter_mappings import *
from dao_manager.base_dao import BaseDAO
from stock_models.datacenter.market_data import ContinuousFall, ContinuousRise, ContinuousVolumeDecrease, ContinuousVolumeIncrease, VolumeDecrease, VolumeIncrease

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

class DiscreteTransactionDao(BaseDAO):

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
            return {'created': 0, 'updated': 0, 'skipped': 0}
        
        # 如果传入的不是列表，转换为列表
        if not isinstance(data_list, list):
            data_list = [data_list]
        
        # 统计计数
        created_count = 0
        updated_count = 0
        skipped_count = 0
        
        # 批量处理，分组进行以减小事务范围
        batch_size = 100
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
            'created': created_count,
            'updated': updated_count,
            'skipped': skipped_count
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

    async def save_volume_increase(self) -> Dict:
        """
        保存成交骤增个股数据
        
        Returns:
            dict: 操作结果统计
        """
        try:
            # 从API获取数据
            async with DataCenterAPI() as api:
                data_list = await api.get_volume_increase()
            
            if not data_list:
                logger.warning("未获取到成交骤增个股数据")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 检查是否是字符串类型（text/plain或text/html格式）
            if isinstance(data_list, str):
                try:
                    # 尝试解析JSON字符串
                    import json
                    data_list = json.loads(data_list)
                except json.JSONDecodeError:
                    logger.error("转换成交骤增个股数据失败，无法解析为JSON")
                    return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 确保data_list是列表类型
            if not isinstance(data_list, list):
                data_list = [data_list]
            
            # 设置当前日期作为交易日期以确保不为空
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            # 处理数据
            processed_data = []
            for data in data_list:
                if not isinstance(data, dict):
                    logger.warning(f"跳过无效数据格式: {data}")
                    continue
                
                # 使用BaseDAO的_parse_datetime和_parse_number方法处理时间和数字
                # 按照映射表将API字段转换为模型字段，并使用标准字典格式
                processed_item = {
                    'trade_date': self._parse_datetime(data.get('t', current_date)),
                    'stock_code': data.get('dm', ''),
                    'stock_name': data.get('mc', ''),
                    'close_price': self._parse_number(data.get('c', 0)),
                    'change_rate': self._parse_number(data.get('zdf', 0)),
                    'is_ex_dividend': self._parse_number(data.get('iscq', 0)),
                    'volume': self._parse_number(data.get('v', 0)),
                    'previous_volume': self._parse_number(data.get('pv', 0)),
                    'volume_change': self._parse_number(data.get('zjl', 0)),
                    'volume_change_rate': self._parse_number(data.get('zjf', 0)),
                    'update_time': self._parse_datetime(current_date)
                }
                
                # 验证必要字段
                if not processed_item['stock_code'] or not processed_item['stock_name']:
                    logger.warning(f"跳过缺少必要字段的数据: {data}")
                    continue
                
                processed_data.append(processed_item)
            
            if not processed_data:
                logger.warning("没有有效的成交骤增个股数据需要保存")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 保存数据
            result = await self._batch_process(
                model_class=VolumeIncrease,
                data_list=processed_data,
                mapping=VOLUME_INCREASE_MAPPING,
                unique_fields=['stock_code', 'trade_date']
            )
            
            # 更新缓存，将对象转换为标准的字典格式
            cache_key = 'volume_increase'
            cache_data = []
            for item in processed_data:
                cache_item = {
                    'trade_date': item['trade_date'].strftime('%Y-%m-%d') if isinstance(item['trade_date'], datetime) else item['trade_date'],
                    'stock_code': item['stock_code'],
                    'stock_name': item['stock_name'],
                    'close_price': float(item['close_price']),
                    'change_rate': float(item['change_rate']),
                    'is_ex_dividend': int(item['is_ex_dividend']),
                    'volume': int(item['volume']),
                    'previous_volume': int(item['previous_volume']),
                    'volume_change': int(item['volume_change']),
                    'volume_change_rate': float(item['volume_change_rate']),
                    'update_time': item['update_time'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(item['update_time'], datetime) else item['update_time']
                }
                cache_data.append(cache_item)
            
            # 使用异步方式更新缓存
            await self._set_to_cache(cache_key, cache_data, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"保存成交骤增个股数据出错: {str(e)}")
            return {'created': 0, 'updated': 0, 'skipped': 0}
       
    async def get_volume_increase(self) -> List[Dict]:
        """
        获取成交骤增个股数据
        
        Returns:
            List[Dict]: 成交骤增个股数据列表
        """
        # 先查缓存
        cache_key = 'volume_increase'
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 缓存未命中，查数据库
        try:
            # 获取最新日期的数据
            latest_date = await sync_to_async(lambda: VolumeIncrease.objects.order_by('-trade_date').values('trade_date').first())()
            
            if latest_date:
                records = await sync_to_async(lambda: list(VolumeIncrease.objects.filter(trade_date=latest_date['trade_date']).order_by('-volume_ratio')))()
                
                if records:
                    result = [{field: getattr(record, field) for field in get_model_fields(VolumeIncrease)} 
                             for record in records]
                    await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['short'])
                    return result
            
            return []
        except Exception as e:
            logger.error(f"获取成交骤增个股数据出错: {str(e)}")
            return []
    
    async def save_volume_decrease(self) -> Dict:
        """
        保存成交骤减个股数据
        
        Returns:
            dict: 操作结果统计
        """
        try:
            # 从API获取数据
            async with DataCenterAPI() as api:
                data_list = await api.get_volume_decrease()
            
            if not data_list:
                logger.warning("未获取到成交骤减个股数据")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 检查是否是字符串类型（text/plain或text/html格式）
            if isinstance(data_list, str):
                try:
                    # 尝试解析JSON字符串
                    import json
                    data_list = json.loads(data_list)
                except json.JSONDecodeError:
                    logger.error("转换成交骤减个股数据失败，无法解析为JSON")
                    return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 确保data_list是列表类型
            if not isinstance(data_list, list):
                data_list = [data_list]
            
            # 设置当前日期作为交易日期以确保不为空
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            # 处理数据
            processed_data = []
            for data in data_list:
                if not isinstance(data, dict):
                    logger.warning(f"跳过无效数据格式: {data}")
                    continue
                
                # 使用BaseDAO的_parse_datetime和_parse_number方法处理时间和数字
                # 按照映射表将API字段转换为模型字段，并使用标准字典格式
                processed_item = {
                    'trade_date': self._parse_datetime(data.get('t', current_date)),
                    'stock_code': data.get('dm', ''),
                    'stock_name': data.get('mc', ''),
                    'close_price': self._parse_number(data.get('c', 0)),
                    'change_rate': self._parse_number(data.get('zdf', 0)),
                    'is_ex_dividend': self._parse_number(data.get('iscq', 0)),
                    'volume': self._parse_number(data.get('v', 0)),
                    'previous_volume': self._parse_number(data.get('pv', 0)),
                    'volume_change': self._parse_number(data.get('zjl', 0)),
                    'volume_change_rate': self._parse_number(data.get('zjf', 0)),
                    'update_time': self._parse_datetime(current_date)
                }
                
                # 验证必要字段
                if not processed_item['stock_code'] or not processed_item['stock_name']:
                    logger.warning(f"跳过缺少必要字段的数据: {data}")
                    continue
                
                processed_data.append(processed_item)
            
            if not processed_data:
                logger.warning("没有有效的成交骤减个股数据需要保存")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 保存数据
            result = await self._batch_process(
                model_class=VolumeDecrease,
                data_list=processed_data,
                mapping=VOLUME_DECREASE_MAPPING,
                unique_fields=['stock_code', 'trade_date']
            )
            
            # 更新缓存，将对象转换为标准的字典格式
            cache_key = 'volume_decrease'
            cache_data = []
            for item in processed_data:
                cache_item = {
                    'trade_date': item['trade_date'].strftime('%Y-%m-%d') if isinstance(item['trade_date'], datetime) else item['trade_date'],
                    'stock_code': item['stock_code'],
                    'stock_name': item['stock_name'],
                    'close_price': float(item['close_price']),
                    'change_rate': float(item['change_rate']),
                    'is_ex_dividend': int(item['is_ex_dividend']),
                    'volume': int(item['volume']),
                    'previous_volume': int(item['previous_volume']),
                    'volume_change': int(item['volume_change']),
                    'volume_change_rate': float(item['volume_change_rate']),
                    'update_time': item['update_time'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(item['update_time'], datetime) else item['update_time']
                }
                cache_data.append(cache_item)
            
            # 使用异步方式更新缓存
            await self._set_to_cache(cache_key, cache_data, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"保存成交骤减个股数据出错: {str(e)}")
            return {'created': 0, 'updated': 0, 'skipped': 0}
    
    async def get_volume_decrease(self) -> List[Dict]:
        """
        获取成交骤减个股数据
        
        Returns:
            List[Dict]: 成交骤减个股数据列表
        """
        # 先查缓存
        cache_key = 'volume_decrease'
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 缓存未命中，查数据库
        try:
            # 获取最新日期的数据
            latest_date = await sync_to_async(lambda: VolumeDecrease.objects.order_by('-trade_date').values('trade_date').first())()
            
            if latest_date:
                records = await sync_to_async(lambda: list(VolumeDecrease.objects.filter(trade_date=latest_date['trade_date']).order_by('-volume_ratio')))()
                
                if records:
                    result = [{field: getattr(record, field) for field in get_model_fields(VolumeDecrease)} 
                             for record in records]
                    await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['short'])
                    return result
            
            return []
        except Exception as e:
            logger.error(f"获取成交骤减个股数据出错: {str(e)}")
            return []
    
    async def save_continuous_volume_increase(self) -> Dict:
        """
        保存连续放量个股数据
        
        Returns:
            dict: 操作结果统计
        """
        try:
            # 从API获取数据
            async with DataCenterAPI() as api:
                data_list = await api.get_continuous_volume_increase()
            
            if not data_list:
                logger.warning("未获取到连续放量个股数据")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 检查是否是字符串类型（text/plain或text/html格式）
            if isinstance(data_list, str):
                try:
                    # 尝试解析JSON字符串
                    import json
                    data_list = json.loads(data_list)
                except json.JSONDecodeError:
                    logger.error("转换连续放量个股数据失败，无法解析为JSON")
                    return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 确保data_list是列表类型
            if not isinstance(data_list, list):
                data_list = [data_list]
            
            # 设置当前日期作为交易日期以确保不为空
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            # 处理数据
            processed_data = []
            for data in data_list:
                if not isinstance(data, dict):
                    logger.warning(f"跳过无效数据格式: {data}")
                    continue
                
                # 使用BaseDAO的_parse_datetime和_parse_number方法处理时间和数字
                # 按照映射表将API字段转换为模型字段，并使用标准字典格式
                processed_item = {
                    'trade_date': self._parse_datetime(data.get('t', current_date)),
                    'stock_code': data.get('dm', ''),
                    'stock_name': data.get('mc', ''),
                    'close_price': self._parse_number(data.get('c', 0)),
                    'change_rate': self._parse_number(data.get('zdf', 0)),
                    'is_ex_dividend': self._parse_number(data.get('iscq', 0)),
                    'volume': self._parse_number(data.get('v', 0)),
                    'previous_volume': self._parse_number(data.get('pv', 0)),
                    'volume_increase_days': self._parse_number(data.get('flday', 0)),
                    'period_change_rate': self._parse_number(data.get('pzdf', 0)),
                    'period_has_ex_dividend': self._parse_number(data.get('ispcq', 0)),
                    'period_turnover_rate': self._parse_number(data.get('phs', 0)),
                    'update_time': self._parse_datetime(current_date)
                }
                
                # 验证必要字段
                if not processed_item['stock_code'] or not processed_item['stock_name']:
                    logger.warning(f"跳过缺少必要字段的数据: {data}")
                    continue
                
                processed_data.append(processed_item)
            
            if not processed_data:
                logger.warning("没有有效的连续放量个股数据需要保存")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 保存数据
            result = await self._batch_process(
                model_class=ContinuousVolumeIncrease,
                data_list=processed_data,
                mapping=CONTINUOUS_VOLUME_INCREASE_MAPPING,
                unique_fields=['stock_code', 'trade_date']
            )
            
            # 更新缓存，将对象转换为标准的字典格式
            cache_key = 'continuous_volume_increase'
            cache_data = []
            for item in processed_data:
                cache_item = {
                    'trade_date': item['trade_date'].strftime('%Y-%m-%d') if isinstance(item['trade_date'], datetime) else item['trade_date'],
                    'stock_code': item['stock_code'],
                    'stock_name': item['stock_name'],
                    'close_price': float(item['close_price']),
                    'change_rate': float(item['change_rate']),
                    'is_ex_dividend': int(item['is_ex_dividend']),
                    'volume': int(item['volume']),
                    'previous_volume': int(item['previous_volume']),
                    'volume_increase_days': int(item['volume_increase_days']),
                    'period_change_rate': float(item['period_change_rate']),
                    'period_has_ex_dividend': int(item['period_has_ex_dividend']),
                    'period_turnover_rate': float(item['period_turnover_rate']),
                    'update_time': item['update_time'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(item['update_time'], datetime) else item['update_time']
                }
                cache_data.append(cache_item)
            
            # 使用异步方式更新缓存
            await self._set_to_cache(cache_key, cache_data, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"保存连续放量个股数据出错: {str(e)}")
            return {'created': 0, 'updated': 0, 'skipped': 0}
    
    async def get_continuous_volume_increase(self) -> List[Dict]:
        """
        获取连续放量上涨个股数据
        
        Returns:
            List[Dict]: 连续放量上涨个股数据列表
        """
        # 先查缓存
        cache_key = 'continuous_volume_increase'
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 缓存未命中，查数据库
        try:
            # 获取最新日期的数据
            latest_date = await sync_to_async(lambda: ContinuousVolumeIncrease.objects.order_by('-trade_date').values('trade_date').first())()
            
            if latest_date:
                records = await sync_to_async(lambda: list(ContinuousVolumeIncrease.objects.filter(trade_date=latest_date['trade_date']).order_by('-volume_ratio')))()
                
                if records:
                    result = [{field: getattr(record, field) for field in get_model_fields(ContinuousVolumeIncrease)} 
                             for record in records]
                    await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['short'])
                    return result
            
            return []
        except Exception as e:
            logger.error(f"获取连续放量上涨个股数据出错: {str(e)}")
            return []
            
    async def save_continuous_volume_decrease(self) -> Dict:
        """
        保存连续缩量个股数据
        
        Returns:
            dict: 操作结果统计
        """
        try:
            # 从API获取数据
            async with DataCenterAPI() as api:
                data_list = await api.get_continuous_volume_decrease()
            
            if not data_list:
                logger.warning("未获取到连续缩量个股数据")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 检查是否是字符串类型（text/plain或text/html格式）
            if isinstance(data_list, str):
                try:
                    # 尝试解析JSON字符串
                    import json
                    data_list = json.loads(data_list)
                except json.JSONDecodeError:
                    logger.error("转换连续缩量个股数据失败，无法解析为JSON")
                    return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 确保data_list是列表类型
            if not isinstance(data_list, list):
                data_list = [data_list]
            
            # 设置当前日期作为交易日期以确保不为空
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            # 处理数据
            processed_data = []
            for data in data_list:
                if not isinstance(data, dict):
                    logger.warning(f"跳过无效数据格式: {data}")
                    continue
                
                # 使用BaseDAO的_parse_datetime和_parse_number方法处理时间和数字
                # 按照映射表将API字段转换为模型字段，并使用标准字典格式
                processed_item = {
                    'trade_date': self._parse_datetime(data.get('t', current_date)),
                    'stock_code': data.get('dm', ''),
                    'stock_name': data.get('mc', ''),
                    'close_price': self._parse_number(data.get('c', 0)),
                    'change_rate': self._parse_number(data.get('zdf', 0)),
                    'is_ex_dividend': self._parse_number(data.get('iscq', 0)),
                    'volume': self._parse_number(data.get('v', 0)),
                    'previous_volume': self._parse_number(data.get('pv', 0)),
                    'volume_decrease_days': self._parse_number(data.get('flday', 0)),
                    'period_change_rate': self._parse_number(data.get('pzdf', 0)),
                    'period_has_ex_dividend': self._parse_number(data.get('ispcq', 0)),
                    'period_turnover_rate': self._parse_number(data.get('phs', 0)),
                    'update_time': self._parse_datetime(current_date)
                }
                
                # 验证必要字段
                if not processed_item['stock_code'] or not processed_item['stock_name']:
                    logger.warning(f"跳过缺少必要字段的数据: {data}")
                    continue
                
                processed_data.append(processed_item)
            
            if not processed_data:
                logger.warning("没有有效的连续缩量个股数据需要保存")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 保存数据
            result = await self._batch_process(
                model_class=ContinuousVolumeDecrease,
                data_list=processed_data,
                mapping=CONTINUOUS_VOLUME_DECREASE_MAPPING,
                unique_fields=['stock_code', 'trade_date']
            )
            
            # 更新缓存，将对象转换为标准的字典格式
            cache_key = 'continuous_volume_decrease'
            cache_data = []
            for item in processed_data:
                cache_item = {
                    'trade_date': item['trade_date'].strftime('%Y-%m-%d') if isinstance(item['trade_date'], datetime) else item['trade_date'],
                    'stock_code': item['stock_code'],
                    'stock_name': item['stock_name'],
                    'close_price': float(item['close_price']),
                    'change_rate': float(item['change_rate']),
                    'is_ex_dividend': int(item['is_ex_dividend']),
                    'volume': int(item['volume']),
                    'previous_volume': int(item['previous_volume']),
                    'volume_decrease_days': int(item['volume_decrease_days']),
                    'period_change_rate': float(item['period_change_rate']),
                    'period_has_ex_dividend': int(item['period_has_ex_dividend']),
                    'period_turnover_rate': float(item['period_turnover_rate']),
                    'update_time': item['update_time'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(item['update_time'], datetime) else item['update_time']
                }
                cache_data.append(cache_item)
            
            # 使用异步方式更新缓存
            await self._set_to_cache(cache_key, cache_data, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"保存连续缩量个股数据出错: {str(e)}")
            return {'created': 0, 'updated': 0, 'skipped': 0}
        
    async def get_continuous_volume_decrease(self) -> List[Dict]:
        """
        获取连续缩量下跌个股数据
        
        Returns:
            List[Dict]: 连续缩量下跌个股数据列表
        """
        # 先查缓存
        cache_key = 'continuous_volume_decrease'
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 缓存未命中，查数据库
        try:
            # 获取最新日期的数据
            latest_date = await sync_to_async(lambda: ContinuousVolumeDecrease.objects.order_by('-trade_date').values('trade_date').first())()
            
            if latest_date:
                records = await sync_to_async(lambda: list(ContinuousVolumeDecrease.objects.filter(trade_date=latest_date['trade_date']).order_by('-volume_ratio')))()
                
                if records:
                    result = [{field: getattr(record, field) for field in get_model_fields(ContinuousVolumeDecrease)} 
                             for record in records]
                    await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['short'])
                    return result
            
            return []
        except Exception as e:
            logger.error(f"获取连续缩量下跌个股数据出错: {str(e)}")
            return []
    
    async def save_continuous_rise(self) -> Dict:
        """
        保存连续上涨个股数据
        
        Returns:
            dict: 操作结果统计
        """
        try:
            # 从API获取数据
            async with DataCenterAPI() as api:
                data_list = await api.get_continuous_rise()
            
            if not data_list:
                logger.warning("未获取到连续上涨个股数据")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 检查是否是字符串类型（text/plain或text/html格式）
            if isinstance(data_list, str):
                try:
                    # 尝试解析JSON字符串
                    import json
                    data_list = json.loads(data_list)
                except json.JSONDecodeError:
                    logger.error("转换连续上涨个股数据失败，无法解析为JSON")
                    return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 确保data_list是列表类型
            if not isinstance(data_list, list):
                data_list = [data_list]
            
            # 设置当前日期作为交易日期以确保不为空
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            # 处理数据
            processed_data = []
            for data in data_list:
                if not isinstance(data, dict):
                    logger.warning(f"跳过无效数据格式: {data}")
                    continue
                
                # 使用BaseDAO的_parse_datetime和_parse_number方法处理时间和数字
                # 按照映射表将API字段转换为模型字段，并使用标准字典格式
                processed_item = {
                    'trade_date': self._parse_datetime(data.get('t', current_date)),
                    'stock_code': data.get('dm', ''),
                    'stock_name': data.get('mc', ''),
                    'close_price': self._parse_number(data.get('c', 0)),
                    'change_rate': self._parse_number(data.get('zdf', 0)),
                    'is_ex_dividend': self._parse_number(data.get('iscq', 0)),
                    'volume': self._parse_number(data.get('v', 0)),
                    'turnover_rate': self._parse_number(data.get('hs', 0)),
                    'rising_days': self._parse_number(data.get('szday', 0)),
                    'period_change_rate': self._parse_number(data.get('pzdf', 0)),
                    'period_has_ex_dividend': self._parse_number(data.get('ispcq', 0)),
                    'update_time': self._parse_datetime(current_date)
                }
                
                # 验证必要字段
                if not processed_item['stock_code'] or not processed_item['stock_name']:
                    logger.warning(f"跳过缺少必要字段的数据: {data}")
                    continue
                
                processed_data.append(processed_item)
            
            if not processed_data:
                logger.warning("没有有效的连续上涨个股数据需要保存")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 保存数据
            result = await self._batch_process(
                model_class=ContinuousRise,
                data_list=processed_data,
                mapping=CONTINUOUS_RISE_MAPPING,
                unique_fields=['stock_code', 'trade_date']
            )
            
            # 更新缓存，将对象转换为标准的字典格式
            cache_key = 'continuous_rise'
            cache_data = []
            for item in processed_data:
                cache_item = {
                    'trade_date': item['trade_date'].strftime('%Y-%m-%d') if isinstance(item['trade_date'], datetime) else item['trade_date'],
                    'stock_code': item['stock_code'],
                    'stock_name': item['stock_name'],
                    'close_price': float(item['close_price']),
                    'change_rate': float(item['change_rate']),
                    'is_ex_dividend': int(item['is_ex_dividend']),
                    'volume': int(item['volume']),
                    'turnover_rate': float(item['turnover_rate']),
                    'rising_days': int(item['rising_days']),
                    'period_change_rate': float(item['period_change_rate']),
                    'period_has_ex_dividend': int(item['period_has_ex_dividend']),
                    'update_time': item['update_time'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(item['update_time'], datetime) else item['update_time']
                }
                cache_data.append(cache_item)
            
            # 使用异步方式更新缓存
            await self._set_to_cache(cache_key, cache_data, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"保存连续上涨个股数据出错: {str(e)}")
            return {'created': 0, 'updated': 0, 'skipped': 0}
    
    async def get_continuous_rise(self) -> List[Dict]:
        """
        获取连续上涨个股数据
        
        Returns:
            List[Dict]: 连续上涨个股数据列表
        """
        # 先查缓存
        cache_key = 'continuous_rise'
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 缓存未命中，查数据库
        try:
            # 获取最新日期的数据
            latest_date = await sync_to_async(lambda: ContinuousRise.objects.order_by('-trade_date').values('trade_date').first())()
            
            if latest_date:
                records = await sync_to_async(lambda: list(ContinuousRise.objects.filter(trade_date=latest_date['trade_date']).order_by('-rising_days')))()
                
                if records:
                    result = [{field: getattr(record, field) for field in get_model_fields(ContinuousRise)} 
                             for record in records]
                    await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['short'])
                    return result
            
            return []
        except Exception as e:
            logger.error(f"获取连续上涨个股数据出错: {str(e)}")
            return []
    
    async def save_continuous_fall(self) -> Dict:
        """
        保存连续下跌个股数据
        
        Returns:
            dict: 操作结果统计
        """
        try:
            # 从API获取数据
            async with DataCenterAPI() as api:
                data_list = await api.get_continuous_fall()
            
            if not data_list:
                logger.warning("未获取到连续下跌个股数据")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 检查是否是字符串类型（text/plain或text/html格式）
            if isinstance(data_list, str):
                try:
                    # 尝试解析JSON字符串
                    import json
                    data_list = json.loads(data_list)
                except json.JSONDecodeError:
                    logger.error("转换连续下跌个股数据失败，无法解析为JSON")
                    return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 确保data_list是列表类型
            if not isinstance(data_list, list):
                data_list = [data_list]
            
            # 设置当前日期作为交易日期以确保不为空
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            # 处理数据
            processed_data = []
            for data in data_list:
                if not isinstance(data, dict):
                    logger.warning(f"跳过无效数据格式: {data}")
                    continue
                
                # 使用BaseDAO的_parse_datetime和_parse_number方法处理时间和数字
                # 按照映射表将API字段转换为模型字段，并使用标准字典格式
                processed_item = {
                    'trade_date': self._parse_datetime(data.get('t', current_date)),
                    'stock_code': data.get('dm', ''),
                    'stock_name': data.get('mc', ''),
                    'close_price': self._parse_number(data.get('c', 0)),
                    'change_rate': self._parse_number(data.get('zdf', 0)),
                    'is_ex_dividend': self._parse_number(data.get('iscq', 0)),
                    'volume': self._parse_number(data.get('v', 0)),
                    'turnover_rate': self._parse_number(data.get('hs', 0)),
                    'falling_days': self._parse_number(data.get('szday', 0)),
                    'period_change_rate': self._parse_number(data.get('pzdf', 0)),
                    'period_has_ex_dividend': self._parse_number(data.get('ispcq', 0)),
                    'update_time': self._parse_datetime(current_date)
                }
                
                # 验证必要字段
                if not processed_item['stock_code'] or not processed_item['stock_name']:
                    logger.warning(f"跳过缺少必要字段的数据: {data}")
                    continue
                
                processed_data.append(processed_item)
            
            if not processed_data:
                logger.warning("没有有效的连续下跌个股数据需要保存")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 保存数据
            result = await self._batch_process(
                model_class=ContinuousFall,
                data_list=processed_data,
                mapping=CONTINUOUS_FALL_MAPPING,
                unique_fields=['stock_code', 'trade_date']
            )
            
            # 更新缓存，将对象转换为标准的字典格式
            cache_key = 'continuous_fall'
            cache_data = []
            for item in processed_data:
                cache_item = {
                    'trade_date': item['trade_date'].strftime('%Y-%m-%d') if isinstance(item['trade_date'], datetime) else item['trade_date'],
                    'stock_code': item['stock_code'],
                    'stock_name': item['stock_name'],
                    'close_price': float(item['close_price']),
                    'change_rate': float(item['change_rate']),
                    'is_ex_dividend': int(item['is_ex_dividend']),
                    'volume': int(item['volume']),
                    'turnover_rate': float(item['turnover_rate']),
                    'falling_days': int(item['falling_days']),
                    'period_change_rate': float(item['period_change_rate']),
                    'period_has_ex_dividend': int(item['period_has_ex_dividend']),
                    'update_time': item['update_time'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(item['update_time'], datetime) else item['update_time']
                }
                cache_data.append(cache_item)
            
            # 使用异步方式更新缓存
            await self._set_to_cache(cache_key, cache_data, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"保存连续下跌个股数据出错: {str(e)}")
            return {'created': 0, 'updated': 0, 'skipped': 0}
    
    async def get_continuous_fall(self) -> List[Dict]:
        """
        获取连续下跌个股数据
        
        Returns:
            List[Dict]: 连续下跌个股数据列表
        """
        # 先查缓存
        cache_key = 'continuous_fall'
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 缓存未命中，查数据库
        try:
            # 获取最新日期的数据
            latest_date = await sync_to_async(lambda: ContinuousFall.objects.order_by('-trade_date').values('trade_date').first())()
            
            if latest_date:
                records = await sync_to_async(lambda: list(ContinuousFall.objects.filter(trade_date=latest_date['trade_date']).order_by('-falling_days')))()
                
                if records:
                    result = [{field: getattr(record, field) for field in get_model_fields(ContinuousFall)} 
                             for record in records]
                    await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['short'])
                    return result
            
            return []
        except Exception as e:
            logger.error(f"获取连续下跌个股数据出错: {str(e)}")
            return []
            