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
from stock_models.datacenter.financial import CircMarketValueRank, MonthlyRankChange, MonthlyStrongStock, PBRatioRank, PERatioRank, ROERank, WeeklyRankChange, WeeklyStrongStock

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

class FinancialDao(BaseDAO):

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


    async def save_weekly_rank_change(self) -> Dict:
        """
        保存周度排名变化数据
        
        Returns:
            dict: 操作结果统计
        """
        try:
            # 从API获取数据
            async with DataCenterAPI() as api:
                data_list = await api.get_weekly_rank_change()
            
            if not data_list:
                logger.warning("未获取到周度排名变化数据")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 判断返回的是否是字符串，如果是需要转换为列表
            if isinstance(data_list, str):
                logger.warning(f"周度排名变化数据是字符串类型，尝试转换: {data_list[:100]}...")
                try:
                    # 检查是否是HTTP错误响应
                    if "404" in data_list or "500" in data_list:
                        logger.error(f"API返回错误: {data_list}")
                        return {'created': 0, 'updated': 0, 'skipped': 0}
                    
                    # 尝试解析为JSON
                    import json
                    data_list = json.loads(data_list)
                except json.JSONDecodeError:
                    # 如果不是JSON格式，可能是纯文本格式，尝试解析
                    logger.error("转换周度排名变化数据失败，尝试解析为纯文本格式")
                    # 按行分割
                    lines = data_list.split('\n')
                    parsed_data = []
                    
                    # 假设第一行是列头
                    if len(lines) > 1:
                        headers = lines[0].split(',')
                        for line in lines[1:]:
                            if not line.strip():
                                continue
                            values = line.split(',')
                            if len(values) == len(headers):
                                item = dict(zip(headers, values))
                                parsed_data.append(item)
                    
                    if parsed_data:
                        data_list = parsed_data
                    else:
                        logger.error("无法解析纯文本数据格式")
                        return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 确保data_list是列表类型
            if not isinstance(data_list, list):
                data_list = [data_list]
            
            # 处理数据
            processed_data = []
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            for data in data_list:
                if not isinstance(data, dict):
                    logger.warning(f"跳过无效数据格式: {data}")
                    continue
                
                # 构建标准格式的字典
                processed_item = {}
                
                # 按照映射关系逐个处理字段
                for api_field, model_field in WEEKLY_RANK_CHANGE_MAPPING.items():
                    value = data.get(api_field)
                    
                    # 日期字段处理
                    if model_field.endswith('_time') or model_field.endswith('_date') or model_field == 'trade_date':
                        processed_item[model_field] = self._parse_datetime(value if value else current_date)
                    # 数值字段处理
                    elif model_field.endswith('_rate') or model_field.endswith('_price') or model_field.endswith('_volume') or model_field.endswith('_amount') or model_field.endswith('_amplitude'):
                        processed_item[model_field] = self._parse_number(value if value is not None else 0)
                    else:
                        processed_item[model_field] = value
                
                # 添加额外字段
                processed_item['update_time'] = self._parse_datetime(current_date)
                
                # 验证必要字段
                if not processed_item.get('stock_code'):
                    logger.warning(f"跳过缺少股票代码的数据: {data}")
                    continue
                
                processed_data.append(processed_item)
            
            if not processed_data:
                logger.warning("保存周度排名变化数据 - 没有有效的数据需要保存")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 保存数据
            result = await self._batch_process(
                model_class=WeeklyRankChange,
                data_list=processed_data,
                mapping=WEEKLY_RANK_CHANGE_MAPPING,
                unique_fields=['stock_code', 'trade_date']
            )
            
            # 更新缓存
            cache_key = 'weekly_rank_change'
            # 转换为字典格式再存入缓存
            await self._set_to_cache(cache_key, processed_data, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"保存周度排名变化数据出错: {str(e)}")
            return {'created': 0, 'updated': 0, 'skipped': 0}

    async def get_weekly_rank_change(self, order_by: str = 'zdf', ascending: bool = False) -> List[Dict]:
        """
        获取周涨跌排名数据
        
        Args:
            order_by: 排序字段，默认为'zdf'
            ascending: 是否升序，默认为False
            
        Returns:
            List[Dict]: 周涨跌排名数据列表
        """
        # 先查缓存
        cache_key = f'weekly_rank_change_{order_by}_{ascending}'
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 缓存未命中，查数据库
        try:
            latest_date = await asyncio.to_thread(
                lambda: WeeklyRankChange.objects.order_by('-t').values('t').first()
            )
            
            if latest_date:
                order_field = order_by if not ascending else f'-{order_by}'
                records = await asyncio.to_thread(
                    lambda: WeeklyRankChange.objects.filter(t=latest_date['t']).order_by(order_field)
                )
                
                if records:
                    result = [{field: getattr(record, field) for field in get_model_fields(WeeklyRankChange)} 
                             for record in records]
                    await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['medium'])
                    return result
            
            return []
        except Exception as e:
            logger.error(f"获取周涨跌排名数据出错: {str(e)}")
            return []

    async def save_monthly_rank_change(self) -> Dict:
        """
        保存月涨跌排名数据
        
        Returns:
            dict: 操作结果统计
        """
        try:
            # 从API获取数据
            async with DataCenterAPI() as api:
                data_list = await api.get_monthly_rank_change()

            if not data_list:
                logger.warning("未获取到月涨跌排名数据")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 判断返回的是否是字符串，如果是需要转换为列表
            if isinstance(data_list, str):
                logger.warning(f"月涨跌排名数据是字符串类型，尝试转换: {data_list[:100]}...")
                try:
                    # 检查是否是HTTP错误响应
                    if "404" in data_list or "500" in data_list:
                        logger.error(f"API返回错误: {data_list}")
                        return {'created': 0, 'updated': 0, 'skipped': 0}
                    
                    # 尝试解析为JSON
                    import json
                    data_list = json.loads(data_list)
                except json.JSONDecodeError:
                    # 如果不是JSON格式，可能是纯文本格式，尝试解析
                    logger.error("转换月涨跌排名数据失败，尝试解析为纯文本格式")
                    # 按行分割
                    lines = data_list.split('\n')
                    parsed_data = []
                    
                    # 假设第一行是列头
                    if len(lines) > 1:
                        headers = lines[0].split(',')
                        for line in lines[1:]:
                            if not line.strip():
                                continue
                            values = line.split(',')
                            if len(values) == len(headers):
                                item = dict(zip(headers, values))
                                parsed_data.append(item)
                    
                    if parsed_data:
                        data_list = parsed_data
                    else:
                        logger.error("无法解析纯文本数据格式")
                        return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 确保data_list是列表类型
            if not isinstance(data_list, list):
                data_list = [data_list]
            
            # 处理数据
            processed_data = []
            # 设置当前日期作为日期字段，确保不为空
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            for data in data_list:
                if not isinstance(data, dict):
                    logger.warning(f"跳过无效数据格式: {data}")
                    continue
                
                # 构建标准格式的字典
                processed_item = {}
                
                # 按照映射关系逐个处理字段
                for api_field, model_field in MONTHLY_RANK_CHANGE_MAPPING.items():
                    value = data.get(api_field)
                    
                    # 日期字段处理
                    if model_field.endswith('_time') or model_field.endswith('_date') or model_field == 'trade_date':
                        processed_item[model_field] = self._parse_datetime(value if value else current_date)
                    # 数值字段处理
                    elif model_field.endswith('_rate') or model_field.endswith('_price') or model_field.endswith('_volume') or model_field.endswith('_amount') or model_field.endswith('_amplitude'):
                        processed_item[model_field] = self._parse_number(value if value is not None else 0)
                    else:
                        processed_item[model_field] = value
                
                # 添加额外字段
                processed_item['update_time'] = self._parse_datetime(current_date)
                
                # 验证必要字段
                if not processed_item.get('stock_code'):
                    logger.warning(f"跳过缺少股票代码的数据: {data}")
                    continue
                
                processed_data.append(processed_item)
            
            if not processed_data:
                logger.warning("保存月涨跌排名数据 - 没有有效的数据需要保存")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 保存数据
            result = await self._batch_process(
                model_class=MonthlyRankChange,
                data_list=processed_data,
                mapping=MONTHLY_RANK_CHANGE_MAPPING,
                unique_fields=['stock_code', 'trade_date']
            )
            
            # 更新缓存
            cache_key = 'monthly_rank_change'
            # 转换为字典格式再存入缓存
            await self._set_to_cache(cache_key, processed_data, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"保存月涨跌排名数据出错: {str(e)}")
            return {'created': 0, 'updated': 0, 'skipped': 0}

    async def get_monthly_rank_change(self, order_by: str = 'zdf', ascending: bool = False) -> List[Dict]:
        """
        获取月涨跌排名数据
        
        Args:
            order_by: 排序字段，默认为'zdf'
            ascending: 是否升序，默认为False
            
        Returns:
            List[Dict]: 月涨跌排名数据列表
        """
        # 先查缓存
        cache_key = f'monthly_rank_change_{order_by}_{ascending}'
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 缓存未命中，查数据库
        try:
            latest_date = await asyncio.to_thread(
                lambda: MonthlyRankChange.objects.order_by('-t').values('t').first()
            )
            
            if latest_date:
                order_field = order_by if not ascending else f'-{order_by}'
                records = await asyncio.to_thread(
                    lambda: MonthlyRankChange.objects.filter(t=latest_date['t']).order_by(order_field)
                )
                
                if records:
                    result = [{field: getattr(record, field) for field in get_model_fields(MonthlyRankChange)} 
                             for record in records]
                    await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['medium'])
                    return result
            
            return []
        except Exception as e:
            logger.error(f"获取月涨跌排名数据出错: {str(e)}")
            return []

    async def save_weekly_strong_stocks(self) -> Dict:
        """
        保存本周强势股数据
        
        Returns:
            dict: 操作结果统计
        """
        try:
            # 从API获取数据
            async with DataCenterAPI() as api:
                data_list = await api.get_weekly_strong_stocks()
            
            if not data_list:
                logger.warning("未获取到本周强势股数据")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 判断返回的是否是字符串，如果是需要转换为列表
            if isinstance(data_list, str):
                logger.warning(f"本周强势股数据是字符串类型，尝试转换: {data_list[:100]}...")
                try:
                    # 检查是否是HTTP错误响应
                    if "404" in data_list or "500" in data_list:
                        logger.error(f"API返回错误: {data_list}")
                        return {'created': 0, 'updated': 0, 'skipped': 0}
                    
                    # 尝试解析为JSON
                    import json
                    data_list = json.loads(data_list)
                except json.JSONDecodeError:
                    # 如果不是JSON格式，可能是纯文本格式，尝试解析
                    logger.error("转换本周强势股数据失败，尝试解析为纯文本格式")
                    # 按行分割
                    lines = data_list.split('\n')
                    parsed_data = []
                    
                    # 假设第一行是列头
                    if len(lines) > 1:
                        headers = lines[0].split(',')
                        for line in lines[1:]:
                            if not line.strip():
                                continue
                            values = line.split(',')
                            if len(values) == len(headers):
                                item = dict(zip(headers, values))
                                parsed_data.append(item)
                    
                    if parsed_data:
                        data_list = parsed_data
                    else:
                        logger.error("无法解析纯文本数据格式")
                        return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 确保data_list是列表类型
            if not isinstance(data_list, list):
                data_list = [data_list]
            
            # 设置当前日期作为日期字段，确保不为空
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            # 处理数据
            processed_data = []
            for data in data_list:
                if not isinstance(data, dict):
                    logger.warning(f"跳过无效数据格式: {data}")
                    continue
                
                # 构建标准格式的字典
                processed_item = {}
                
                # 按照映射关系逐个处理字段
                for api_field, model_field in WEEKLY_STRONG_STOCK_MAPPING.items():
                    value = data.get(api_field)
                    
                    # 日期字段处理
                    if model_field.endswith('_time') or model_field.endswith('_date') or model_field == 'trade_date':
                        processed_item[model_field] = self._parse_datetime(value if value else current_date)
                    # 数值字段处理
                    elif (model_field.endswith('_rate') or model_field.endswith('_price') or 
                          model_field.endswith('_volume') or model_field.endswith('_amount') or 
                          model_field.endswith('_amplitude') or 'hs300' in model_field):
                        processed_item[model_field] = self._parse_number(value if value is not None else 0)
                    else:
                        processed_item[model_field] = value
                
                # 添加额外字段
                processed_item['update_time'] = self._parse_datetime(current_date)
                
                # 验证必要字段
                if not processed_item.get('stock_code'):
                    logger.warning(f"跳过缺少股票代码的数据: {data}")
                    continue
                
                processed_data.append(processed_item)
            
            if not processed_data:
                logger.warning("没有有效的本周强势股数据需要保存")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 保存数据
            result = await self._batch_process(
                model_class=WeeklyStrongStock,
                data_list=processed_data,
                mapping=WEEKLY_STRONG_STOCK_MAPPING,
                unique_fields=['stock_code', 'trade_date']
            )
            
            # 更新缓存
            cache_key = 'weekly_strong_stocks'
            # 转换为字典格式再存入缓存
            await self._set_to_cache(cache_key, processed_data, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"保存本周强势股数据出错: {str(e)}")
            return {'created': 0, 'updated': 0, 'skipped': 0}

    async def get_weekly_strong_stocks(self) -> List[Dict]:
        """
        获取本周强势股数据
        
        Returns:
            List[Dict]: 本周强势股数据列表
        """
        # 先查缓存
        cache_key = 'weekly_strong_stocks'
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 缓存未命中，查数据库
        try:
            latest_date = await asyncio.to_thread(
                lambda: WeeklyStrongStock.objects.order_by('-t').values('t').first()
            )
            
            if latest_date:
                records = await asyncio.to_thread(
                    lambda: WeeklyStrongStock.objects.filter(t=latest_date['t']).order_by('-zdf')
                )
                
                if records:
                    result = [{field: getattr(record, field) for field in get_model_fields(WeeklyStrongStock)} 
                             for record in records]
                    await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['medium'])
                    return result
            
            return []
        except Exception as e:
            logger.error(f"获取本周强势股数据出错: {str(e)}")
            return []

    async def save_monthly_strong_stocks(self) -> Dict:
        """
        保存本月强势股数据
        
        Returns:
            dict: 操作结果统计
        """
        try:
            # 从API获取数据
            async with DataCenterAPI() as api:
                data_list = await api.get_monthly_strong_stocks()
            
            if not data_list:
                logger.warning("未获取到本月强势股数据")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 判断返回的是否是字符串，如果是需要转换为列表
            if isinstance(data_list, str):
                logger.warning(f"本月强势股数据是字符串类型，尝试转换: {data_list[:100]}...")
                try:
                    # 检查是否是HTTP错误响应
                    if "404" in data_list or "500" in data_list:
                        logger.error(f"API返回错误: {data_list}")
                        return {'created': 0, 'updated': 0, 'skipped': 0}
                    
                    # 尝试解析为JSON
                    import json
                    data_list = json.loads(data_list)
                except json.JSONDecodeError:
                    # 如果不是JSON格式，可能是纯文本格式，尝试解析
                    logger.error("转换本月强势股数据失败，尝试解析为纯文本格式")
                    # 按行分割
                    lines = data_list.split('\n')
                    parsed_data = []
                    
                    # 假设第一行是列头
                    if len(lines) > 1:
                        headers = lines[0].split(',')
                        for line in lines[1:]:
                            if not line.strip():
                                continue
                            values = line.split(',')
                            if len(values) == len(headers):
                                item = dict(zip(headers, values))
                                parsed_data.append(item)
                    
                    if parsed_data:
                        data_list = parsed_data
                    else:
                        logger.error("无法解析纯文本数据格式")
                        return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 确保data_list是列表类型
            if not isinstance(data_list, list):
                data_list = [data_list]
            
            # 设置当前日期作为日期字段，确保不为空
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            # 处理数据
            processed_data = []
            for data in data_list:
                if not isinstance(data, dict):
                    logger.warning(f"跳过无效数据格式: {data}")
                    continue
                
                # 构建标准格式的字典
                processed_item = {}
                
                # 按照映射关系逐个处理字段
                for api_field, model_field in MONTHLY_STRONG_STOCK_MAPPING.items():
                    value = data.get(api_field)
                    
                    # 日期字段处理
                    if model_field.endswith('_time') or model_field.endswith('_date') or model_field == 'trade_date':
                        processed_item[model_field] = self._parse_datetime(value if value else current_date)
                    # 数值字段处理
                    elif (model_field.endswith('_rate') or model_field.endswith('_price') or 
                          model_field.endswith('_volume') or model_field.endswith('_amount') or 
                          model_field.endswith('_amplitude') or 'hs300' in model_field):
                        processed_item[model_field] = self._parse_number(value if value is not None else 0)
                    else:
                        processed_item[model_field] = value
                
                # 添加额外字段
                processed_item['update_time'] = self._parse_datetime(current_date)
                
                # 验证必要字段
                if not processed_item.get('stock_code'):
                    logger.warning(f"跳过缺少股票代码的数据: {data}")
                    continue
                
                processed_data.append(processed_item)
            
            if not processed_data:
                logger.warning("没有有效的本月强势股数据需要保存")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 保存数据
            result = await self._batch_process(
                model_class=MonthlyStrongStock,
                data_list=processed_data,
                mapping=MONTHLY_STRONG_STOCK_MAPPING,
                unique_fields=['stock_code', 'trade_date']
            )
            
            # 更新缓存
            cache_key = 'monthly_strong_stocks'
            # 转换为字典格式再存入缓存
            await self._set_to_cache(cache_key, processed_data, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"保存本月强势股数据出错: {str(e)}")
            return {'created': 0, 'updated': 0, 'skipped': 0}

    async def get_monthly_strong_stocks(self) -> List[Dict]:
        """
        获取本月强势股数据
        
        Returns:
            List[Dict]: 本月强势股数据列表
        """
        # 先查缓存
        cache_key = 'monthly_strong_stocks'
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 缓存未命中，查数据库
        try:
            latest_date = await asyncio.to_thread(
                lambda: MonthlyStrongStock.objects.order_by('-t').values('t').first()
            )
            
            if latest_date:
                records = await asyncio.to_thread(
                    lambda: MonthlyStrongStock.objects.filter(t=latest_date['t']).order_by('-zdf')
                )
                
                if records:
                    result = [{field: getattr(record, field) for field in get_model_fields(MonthlyStrongStock)} 
                             for record in records]
                    await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['medium'])
                    return result
            
            return []
        except Exception as e:
            logger.error(f"获取本月强势股数据出错: {str(e)}")
            return []

    async def save_circ_market_value_rank(self) -> Dict:
        """
        保存流通市值排行数据
        
        Returns:
            dict: 操作结果统计
        """
        try:
            # 从API获取数据
            async with DataCenterAPI() as api:
                data_list = await api.get_circ_market_value_rank()
            
            if not data_list:
                logger.warning("未获取到流通市值排行数据")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 判断返回的是否是字符串，如果是需要转换为列表
            if isinstance(data_list, str):
                logger.warning(f"流通市值排行数据是字符串类型，尝试转换: {data_list[:100]}...")
                try:
                    # 检查是否是HTTP错误响应
                    if "404" in data_list or "500" in data_list:
                        logger.error(f"API返回错误: {data_list}")
                        return {'created': 0, 'updated': 0, 'skipped': 0}
                    
                    # 尝试解析为JSON
                    import json
                    data_list = json.loads(data_list)
                except json.JSONDecodeError:
                    # 如果不是JSON格式，可能是纯文本格式，尝试解析
                    logger.error("转换流通市值排行数据失败，尝试解析为纯文本格式")
                    # 按行分割
                    lines = data_list.split('\n')
                    parsed_data = []
                    
                    # 假设第一行是列头
                    if len(lines) > 1:
                        headers = lines[0].split(',')
                        for line in lines[1:]:
                            if not line.strip():
                                continue
                            values = line.split(',')
                            if len(values) == len(headers):
                                item = dict(zip(headers, values))
                                parsed_data.append(item)
                    
                    if parsed_data:
                        data_list = parsed_data
                    else:
                        logger.error("无法解析纯文本数据格式")
                        return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 确保data_list是列表类型
            if not isinstance(data_list, list):
                data_list = [data_list]
            
            # 设置当前日期作为日期字段，确保不为空
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            # 处理数据
            processed_data = []
            for data in data_list:
                if not isinstance(data, dict):
                    logger.warning(f"跳过无效数据格式: {data}")
                    continue
                
                # 构建标准格式的字典
                processed_item = {}
                
                # 按照映射关系逐个处理字段
                for api_field, model_field in CIRC_MARKET_VALUE_RANK_MAPPING.items():
                    value = data.get(api_field)
                    
                    # 日期字段处理
                    if model_field.endswith('_time') or model_field.endswith('_date') or model_field == 'trade_date':
                        processed_item[model_field] = self._parse_datetime(value if value else current_date)
                    # 数值字段处理
                    elif (model_field.endswith('_rate') or model_field.endswith('_price') or 
                          model_field.endswith('_value') or model_field.endswith('_volume') or 
                          model_field == 'turnover_rate'):
                        processed_item[model_field] = self._parse_number(value if value is not None else 0)
                    else:
                        processed_item[model_field] = value
                
                # 添加额外字段
                processed_item['update_time'] = self._parse_datetime(current_date)
                
                # 验证必要字段
                if not processed_item.get('stock_code'):
                    logger.warning(f"跳过缺少股票代码的数据: {data}")
                    continue
                
                processed_data.append(processed_item)
            
            if not processed_data:
                logger.warning("没有有效的流通市值排行数据需要保存")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 保存数据
            result = await self._batch_process(
                model_class=CircMarketValueRank,
                data_list=processed_data,
                mapping=CIRC_MARKET_VALUE_RANK_MAPPING,
                unique_fields=['stock_code', 'trade_date']
            )
            
            # 更新缓存
            cache_key = 'circ_market_value_rank'
            # 转换为字典格式再存入缓存
            await self._set_to_cache(cache_key, processed_data, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"保存流通市值排行数据出错: {str(e)}")
            return {'created': 0, 'updated': 0, 'skipped': 0}

    async def get_circ_market_value_rank(self) -> List[Dict]:
        """
        获取流通市值排行数据
        
        Returns:
            List[Dict]: 流通市值排行数据列表
        """
        # 先查缓存
        cache_key = 'circ_market_value_rank'
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 缓存未命中，查数据库
        try:
            latest_date = await asyncio.to_thread(
                lambda: CircMarketValueRank.objects.order_by('-t').values('t').first()
            )
            
            if latest_date:
                records = await asyncio.to_thread(
                    lambda: CircMarketValueRank.objects.filter(t=latest_date['t']).order_by('-ltsz')
                )
                
                if records:
                    result = [{field: getattr(record, field) for field in get_model_fields(CircMarketValueRank)} 
                             for record in records]
                    await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['medium'])
                    return result
            
            return []
        except Exception as e:
            logger.error(f"获取流通市值排行数据出错: {str(e)}")
            return []

    async def save_pe_ratio_rank(self) -> Dict:
        """
        保存市盈率排行数据
        
        Returns:
            dict: 操作结果统计
        """
        try:
            # 从API获取数据
            async with DataCenterAPI() as api:
                data_list = await api.get_pe_ratio_rank()
            
            if not data_list:
                logger.warning("未获取到市盈率排行数据")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 判断返回的是否是字符串，如果是需要转换为列表
            if isinstance(data_list, str):
                logger.warning(f"市盈率排行数据是字符串类型，尝试转换: {data_list[:100]}...")
                try:
                    # 检查是否是HTTP错误响应
                    if "404" in data_list or "500" in data_list:
                        logger.error(f"API返回错误: {data_list}")
                        return {'created': 0, 'updated': 0, 'skipped': 0}
                    
                    # 尝试解析为JSON
                    import json
                    data_list = json.loads(data_list)
                except json.JSONDecodeError:
                    # 如果不是JSON格式，可能是纯文本格式，尝试解析
                    logger.error("转换市盈率排行数据失败，尝试解析为纯文本格式")
                    # 按行分割
                    lines = data_list.split('\n')
                    parsed_data = []
                    
                    # 假设第一行是列头
                    if len(lines) > 1:
                        headers = lines[0].split(',')
                        for line in lines[1:]:
                            if not line.strip():
                                continue
                            values = line.split(',')
                            if len(values) == len(headers):
                                item = dict(zip(headers, values))
                                parsed_data.append(item)
                    
                    if parsed_data:
                        data_list = parsed_data
                    else:
                        logger.error("无法解析纯文本数据格式")
                        return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 确保data_list是列表类型
            if not isinstance(data_list, list):
                data_list = [data_list]
            
            # 设置当前日期作为日期字段，确保不为空
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            # 处理数据
            processed_data = []
            for data in data_list:
                if not isinstance(data, dict):
                    logger.warning(f"跳过无效数据格式: {data}")
                    continue
                
                # 构建标准格式的字典
                processed_item = {}
                
                # 按照映射关系逐个处理字段
                for api_field, model_field in PE_RATIO_RANK_MAPPING.items():
                    value = data.get(api_field)
                    
                    # 日期字段处理
                    if model_field.endswith('_time') or model_field.endswith('_date') or model_field == 'trade_date':
                        processed_item[model_field] = self._parse_datetime(value if value else current_date)
                    # 数值字段处理
                    elif (model_field.endswith('_ratio') or model_field.endswith('_price') or 
                          model_field.endswith('_rate') or model_field.endswith('_volume')):
                        processed_item[model_field] = self._parse_number(value if value is not None else 0)
                    else:
                        processed_item[model_field] = value
                
                # 添加额外字段
                processed_item['update_time'] = self._parse_datetime(current_date)
                
                # 验证必要字段
                if not processed_item.get('stock_code'):
                    logger.warning(f"跳过缺少股票代码的数据: {data}")
                    continue
                
                processed_data.append(processed_item)
            
            if not processed_data:
                logger.warning("没有有效的市盈率排行数据需要保存")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 保存数据
            result = await self._batch_process(
                model_class=PERatioRank,
                data_list=processed_data,
                mapping=PE_RATIO_RANK_MAPPING,
                unique_fields=['stock_code', 'trade_date']
            )
            
            # 更新缓存
            cache_key = 'pe_ratio_rank'
            # 转换为字典格式再存入缓存
            await self._set_to_cache(cache_key, processed_data, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"保存市盈率排行数据出错: {str(e)}")
            return {'created': 0, 'updated': 0, 'skipped': 0}

    async def get_pe_ratio_rank(self) -> List[Dict]:
        """
        获取市盈率排行数据
        
        Returns:
            List[Dict]: 市盈率排行数据列表
        """
        # 先查缓存
        cache_key = 'pe_ratio_rank'
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 缓存未命中，查数据库
        try:
            latest_date = await asyncio.to_thread(
                lambda: PERatioRank.objects.order_by('-t').values('t').first()
            )
            
            if latest_date:
                records = await asyncio.to_thread(
                    lambda: PERatioRank.objects.filter(t=latest_date['t']).order_by('pe')
                )
                
                if records:
                    result = [{field: getattr(record, field) for field in get_model_fields(PERatioRank)} 
                             for record in records]
                    await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['medium'])
                    return result
            
            return []
        except Exception as e:
            logger.error(f"获取市盈率排行数据出错: {str(e)}")
            return []

    async def save_pb_ratio_rank(self) -> Dict:
        """
        保存市净率排行数据
        
        Returns:
            dict: 操作结果统计
        """
        try:
            # 从API获取数据
            async with DataCenterAPI() as api:
                data_list = await api.get_pb_ratio_rank()
            
            if not data_list:
                logger.warning("未获取到市净率排行数据")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 判断返回的是否是字符串，如果是需要转换为列表
            if isinstance(data_list, str):
                logger.warning(f"市净率排行数据是字符串类型，尝试转换: {data_list[:100]}...")
                try:
                    # 检查是否是HTTP错误响应
                    if "404" in data_list or "500" in data_list:
                        logger.error(f"API返回错误: {data_list}")
                        return {'created': 0, 'updated': 0, 'skipped': 0}
                    
                    # 尝试解析为JSON
                    import json
                    data_list = json.loads(data_list)
                except json.JSONDecodeError:
                    # 如果不是JSON格式，可能是纯文本格式，尝试解析
                    logger.error("转换市净率排行数据失败，尝试解析为纯文本格式")
                    # 按行分割
                    lines = data_list.split('\n')
                    parsed_data = []
                    
                    # 假设第一行是列头
                    if len(lines) > 1:
                        headers = lines[0].split(',')
                        for line in lines[1:]:
                            if not line.strip():
                                continue
                            values = line.split(',')
                            if len(values) == len(headers):
                                item = dict(zip(headers, values))
                                parsed_data.append(item)
                    
                    if parsed_data:
                        data_list = parsed_data
                    else:
                        logger.error("无法解析纯文本数据格式")
                        return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 确保data_list是列表类型
            if not isinstance(data_list, list):
                data_list = [data_list]
            
            # 设置当前日期作为日期字段，确保不为空
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            # 处理数据
            processed_data = []
            for data in data_list:
                if not isinstance(data, dict):
                    logger.warning(f"跳过无效数据格式: {data}")
                    continue
                
                # 构建标准格式的字典
                processed_item = {}
                
                # 按照映射关系逐个处理字段
                for api_field, model_field in PB_RATIO_RANK_MAPPING.items():
                    value = data.get(api_field)
                    
                    # 日期字段处理
                    if model_field.endswith('_time') or model_field.endswith('_date') or model_field == 'trade_date':
                        processed_item[model_field] = self._parse_datetime(value if value else current_date)
                    # 数值字段处理
                    elif (model_field.endswith('_ratio') or model_field.endswith('_price') or 
                          model_field.endswith('_rate') or model_field.endswith('_volume') or
                          model_field.endswith('_share') or model_field.endswith('_asset')):
                        processed_item[model_field] = self._parse_number(value if value is not None else 0)
                    else:
                        processed_item[model_field] = value
                
                # 添加额外字段
                processed_item['update_time'] = self._parse_datetime(current_date)
                
                # 验证必要字段
                if not processed_item.get('stock_code'):
                    logger.warning(f"跳过缺少股票代码的数据: {data}")
                    continue
                
                processed_data.append(processed_item)
            
            if not processed_data:
                logger.warning("没有有效的市净率排行数据需要保存")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 保存数据
            result = await self._batch_process(
                model_class=PBRatioRank,
                data_list=processed_data,
                mapping=PB_RATIO_RANK_MAPPING,
                unique_fields=['stock_code', 'trade_date']
            )
            
            # 更新缓存
            cache_key = 'pb_ratio_rank'
            # 转换为字典格式再存入缓存
            await self._set_to_cache(cache_key, processed_data, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"保存市净率排行数据出错: {str(e)}")
            return {'created': 0, 'updated': 0, 'skipped': 0}

    async def get_pb_ratio_rank(self) -> List[Dict]:
        """
        获取市净率排行数据
        
        Returns:
            List[Dict]: 市净率排行数据列表
        """
        # 先查缓存
        cache_key = 'pb_ratio_rank'
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 缓存未命中，查数据库
        try:
            latest_date = await asyncio.to_thread(
                lambda: PBRatioRank.objects.order_by('-t').values('t').first()
            )
            
            if latest_date:
                records = await asyncio.to_thread(
                    lambda: PBRatioRank.objects.filter(t=latest_date['t']).order_by('pb')
                )
                
                if records:
                    result = [{field: getattr(record, field) for field in get_model_fields(PBRatioRank)} 
                             for record in records]
                    await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['medium'])
                    return result
            
            return []
        except Exception as e:
            logger.error(f"获取市净率排行数据出错: {str(e)}")
            return []

    async def save_roe_rank(self) -> Dict:
        """
        保存ROE排行数据
        
        Returns:
            dict: 操作结果统计
        """
        try:
            # 从API获取数据
            async with DataCenterAPI() as api:
                data_list = await api.get_roe_rank()
            
            if not data_list:
                logger.warning("未获取到ROE排行数据")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 判断返回的是否是字符串，如果是需要转换为列表
            if isinstance(data_list, str):
                logger.warning(f"ROE排行数据是字符串类型，尝试转换: {data_list[:100]}...")
                try:
                    # 检查是否是HTTP错误响应
                    if "404" in data_list or "500" in data_list:
                        logger.error(f"API返回错误: {data_list}")
                        return {'created': 0, 'updated': 0, 'skipped': 0}
                    
                    # 尝试解析为JSON
                    import json
                    data_list = json.loads(data_list)
                except json.JSONDecodeError:
                    # 如果不是JSON格式，可能是纯文本格式，尝试解析
                    logger.error("转换ROE排行数据失败，尝试解析为纯文本格式")
                    # 按行分割
                    lines = data_list.split('\n')
                    parsed_data = []
                    
                    # 假设第一行是列头
                    if len(lines) > 1:
                        headers = lines[0].split(',')
                        for line in lines[1:]:
                            if not line.strip():
                                continue
                            values = line.split(',')
                            if len(values) == len(headers):
                                item = dict(zip(headers, values))
                                parsed_data.append(item)
                    
                    if parsed_data:
                        data_list = parsed_data
                    else:
                        logger.error("无法解析纯文本数据格式")
                        return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 确保data_list是列表类型
            if not isinstance(data_list, list):
                data_list = [data_list]
            
            # 处理数据
            processed_data = []
            update_time = datetime.now()
            
            for data in data_list:
                if not isinstance(data, dict):
                    logger.warning(f"跳过无效数据格式: {data}")
                    continue
                
                # 构建标准格式的字典
                processed_item = {}
                
                # 按照映射关系逐个处理字段
                for api_field, model_field in ROE_RANK_MAPPING.items():
                    value = data.get(api_field)
                    
                    # 数值字段处理
                    if (model_field.endswith('_ratio') or model_field.endswith('_roe') or 
                          model_field.endswith('_rate') or model_field.endswith('_profit') or
                          model_field.endswith('_value') or model_field.endswith('_margin') or
                          model_field.endswith('_assets') or model_field.endswith('_pe') or
                          model_field.endswith('_pb')):
                        processed_item[model_field] = self._parse_number(value if value is not None else 0)
                    # 整数字段处理
                    elif (model_field.endswith('_rank') or model_field.endswith('_count')):
                        try:
                            processed_item[model_field] = int(value) if value is not None else 0
                        except (ValueError, TypeError):
                            processed_item[model_field] = 0
                    else:
                        processed_item[model_field] = value
                
                # 添加额外字段
                processed_item['update_time'] = self._parse_datetime(update_time)
                
                # 验证必要字段
                if not processed_item.get('stock_code'):
                    logger.warning(f"跳过缺少股票代码的数据: {data}")
                    continue
                
                processed_data.append(processed_item)
            
            if not processed_data:
                logger.warning("没有有效的ROE排行数据需要保存")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 保存数据
            result = await self._batch_process(
                model_class=ROERank,
                data_list=processed_data,
                mapping=ROE_RANK_MAPPING,
                unique_fields=['stock_code', 'industry_name']
            )
            
            # 更新缓存
            cache_key = 'roe_rank'
            # 转换为字典格式再存入缓存
            await self._set_to_cache(cache_key, processed_data, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"保存ROE排行数据出错: {str(e)}")
            return {'created': 0, 'updated': 0, 'skipped': 0}

    async def get_roe_rank(self, report_period: str) -> List[Dict]:
        """
        获取ROE排行数据
        
        Args:
            report_period: 报告期，如'2021-03-31'
            
        Returns:
            List[Dict]: ROE排行数据列表
        """
        # 先查缓存
        cache_key = f'roe_rank_{report_period}'
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 缓存未命中，查数据库
        try:
            records = await asyncio.to_thread(
                lambda: ROERank.objects.filter(hym=report_period).order_by('-roe')
            )
            
            if records:
                result = [{field: getattr(record, field) for field in get_model_fields(ROERank)} 
                         for record in records]
                await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['long'])
                return result
            
            return []
        except Exception as e:
            logger.error(f"获取ROE排行数据出错: {str(e)}")
            return []

