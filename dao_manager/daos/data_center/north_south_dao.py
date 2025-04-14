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
from stock_models.datacenter.north_south import NorthFundTrend, NorthSouthFundOverview, NorthStockHolding, SouthFundTrend

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

class NorthSouthDao(BaseDAO):

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

    async def save_north_south_fund_overview(self) -> Dict:
        """
        保存南北向资金流向概览数据
        
        Returns:
            dict: 操作结果统计
        """
        try:
            # 从API获取数据
            api = DataCenterAPI()
            data_list = await api.get_north_south_fund_overview()
            
            if not data_list:
                logger.warning("未获取到南北向资金流向概览数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 处理数据中的日期和数值
            for data in data_list:
                if 't' in data:
                    data['t'] = self._parse_datetime(data['t'])
                for key in ['hk2sh', 'hk2sz', 'bxzjlr', 'sh2hk', 'sz2hk', 'nxzjlr']:
                    if key in data:
                        data[key] = self._parse_number(data[key])
            
            # 保存数据到数据库
            result = await self._batch_process(
                model_class=NorthSouthFundOverview,
                data_list=data_list,
                mapping=NORTH_SOUTH_FUND_OVERVIEW_MAPPING,
                unique_fields=['trade_date']
            )
            
            # 更新缓存
            if data_list and 't' in data_list[0]:
                date_str = data_list[0]['t'].strftime('%Y-%m-%d')
                cache_key = f'north_south_fund_overview_{date_str}'
                await self._set_to_cache(cache_key, data_list, self.CACHE_TIMEOUT['short'])
            
            return result
        except Exception as e:
            logger.error(f"保存南北向资金流向概览数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def get_north_south_fund_overview(self) -> List[Dict]:
        """
        获取南北向资金流向概览数据
        
        Returns:
            List[Dict]: 南北向资金流向概览数据列表
        """
        # 先查缓存
        cache_key = 'north_south_fund_overview'
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 缓存未命中，查数据库
        try:
            # 获取最近30天的数据
            records = await asyncio.to_thread(
                lambda: NorthSouthFundOverview.objects.order_by('-t')[:30]
            )
            
            if records:
                result = [{field: getattr(record, field) for field in get_model_fields(NorthSouthFundOverview)} 
                         for record in records]
                await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['short'])
                return result
            
            return []
        except Exception as e:
            logger.error(f"获取南北向资金流向概览数据出错: {str(e)}")
            return []

    async def save_north_fund_trend(self) -> Dict:
        """
        保存北向资金历史走势数据
        
        Returns:
            dict: 操作结果统计
        """
        try:
            # 从API获取数据
            api = DataCenterAPI()
            data_list = await api.get_north_fund_history_trend()
            
            if not data_list:
                logger.warning("未获取到北向资金历史走势数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 处理数据中的日期和数值
            for data in data_list:
                if 't' in data:
                    data['t'] = self._parse_datetime(data['t'])
                for key in ['hk2sh', 'hk2sz', 'bxzjlr']:
                    if key in data:
                        data[key] = self._parse_number(data[key])
                if 'hsIndex' in data:
                    data['hsIndex'] = self._parse_number(data['hsIndex'])
            
            # 保存数据到数据库
            result = await self._batch_process(
                model_class=NorthFundTrend,
                data_list=data_list,
                mapping=NORTH_FUND_TREND_MAPPING,
                unique_fields=['trade_date']
            )
            
            # 更新缓存
            if data_list and 't' in data_list[0]:
                date_str = data_list[0]['t'].strftime('%Y-%m-%d')
                cache_key = f'north_fund_trend_{date_str}'
                await self._set_to_cache(cache_key, data_list, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"保存北向资金历史走势数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def get_north_fund_trend(self) -> List[Dict]:
        """
        获取北向资金历史走势数据
        
        Returns:
            List[Dict]: 北向资金历史走势数据列表
        """
        # 先查缓存
        cache_key = 'north_fund_trend'
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 缓存未命中，查数据库
        try:
            # 获取最近180天的数据
            records = await asyncio.to_thread(
                lambda: NorthFundTrend.objects.order_by('-t')[:180]
            )
            
            if records:
                result = [{field: getattr(record, field) for field in get_model_fields(NorthFundTrend)} 
                         for record in records]
                await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['medium'])
                return result
            
            return []
        except Exception as e:
            logger.error(f"获取北向资金历史走势数据出错: {str(e)}")
            return []

    async def save_south_fund_trend(self) -> Dict:
        """
        保存南向资金历史走势数据
        
        Returns:
            dict: 操作结果统计
        """
        try:
            # 从API获取数据
            api = DataCenterAPI()
            data_list = await api.get_south_fund_history_trend()
            
            if not data_list:
                logger.warning("未获取到南向资金历史走势数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 处理数据中的日期和数值
            for data in data_list:
                if 't' in data:
                    data['t'] = self._parse_datetime(data['t'])
                for key in ['sh2hk', 'sz2hk', 'nxzjlr']:
                    if key in data:
                        data[key] = self._parse_number(data[key])
                if 'hsi' in data:
                    data['hsi'] = self._parse_number(data['hsi'])
            
            # 保存数据到数据库
            result = await self._batch_process(
                model_class=SouthFundTrend,
                data_list=data_list,
                mapping=SOUTH_FUND_TREND_MAPPING,
                unique_fields=['trade_date']
            )
            
            # 更新缓存
            if data_list and 't' in data_list[0]:
                date_str = data_list[0]['t'].strftime('%Y-%m-%d')
                cache_key = f'south_fund_trend_{date_str}'
                await self._set_to_cache(cache_key, data_list, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"保存南向资金历史走势数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def get_south_fund_trend(self) -> List[Dict]:
        """
        获取南向资金历史走势数据
        
        Returns:
            List[Dict]: 南向资金历史走势数据列表
        """
        # 先查缓存
        cache_key = 'south_fund_trend'
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 缓存未命中，查数据库
        try:
            # 获取最近180天的数据
            records = await asyncio.to_thread(
                lambda: SouthFundTrend.objects.order_by('-t')[:180]
            )
            
            if records:
                result = [{field: getattr(record, field) for field in get_model_fields(SouthFundTrend)} 
                         for record in records]
                await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['medium'])
                return result
            
            return []
        except Exception as e:
            logger.error(f"获取南向资金历史走势数据出错: {str(e)}")
            return []

    async def save_north_stock_holding(self) -> Dict:
        """
        保存北向持股明细数据
        
        Returns:
            dict: 操作结果统计
        """
        try:
            # 从API获取数据
            api = DataCenterAPI()
            data_list = await api.get_north_stock_period_rank()
            
            if not data_list:
                logger.warning("未获取到北向持股明细数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 处理数据中的日期和数值
            for data in data_list:
                if 't' in data:
                    data['t'] = self._parse_datetime(data['t'])
                for key in ['cgs', 'cgsz', 'djcgs', 'djcgsz']:
                    if key in data:
                        data[key] = self._parse_number(data[key])
                for key in ['zltgbl', 'zdf', 'zdbl']:
                    if key in data:
                        data[key] = self._parse_number(data[key])
            
            # 保存数据到数据库
            result = await self._batch_process(
                model_class=NorthStockHolding,
                data_list=data_list,
                mapping=NORTH_STOCK_HOLDING_MAPPING,
                unique_fields=['stock_code', 'trade_date', 'period']
            )
            
            # 更新缓存
            if data_list and 't' in data_list[0]:
                date_str = data_list[0]['t'].strftime('%Y-%m-%d')
                cache_key = f'north_stock_holding_{date_str}'
                await self._set_to_cache(cache_key, data_list, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"保存北向持股明细数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def get_north_stock_holding(self, period: str) -> List[Dict]:
        """
        获取北向持股明细数据
        
        Args:
            period: 统计周期，可选 LD、3D、5D、10D、LM、LQ、LY
            
        Returns:
            List[Dict]: 北向持股明细数据列表
        """
        # 先查缓存
        cache_key = f'north_stock_holding_{period}'
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 缓存未命中，查数据库
        try:
            # 获取最新日期
            latest_date = await asyncio.to_thread(
                lambda: NorthStockHolding.objects.filter(period=period).order_by('-t').values('t').first()
            )
            
            if not latest_date:
                return []
            
            records = await asyncio.to_thread(
                lambda: NorthStockHolding.objects.filter(
                    period=period, 
                    t=latest_date['t']
                ).order_by('-held_amount')
            )
            
            if records:
                result = [{field: getattr(record, field) for field in get_model_fields(NorthStockHolding)} 
                         for record in records]
                await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['medium'])
                return result
            
            return []
        except Exception as e:
            logger.error(f"获取北向持股明细数据(周期:{period})出错: {str(e)}")
            return []
 