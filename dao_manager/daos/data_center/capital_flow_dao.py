import logging
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
from django.db import transaction
from django.db.models import Q
from django.core.cache import cache
from django.db.models.fields import Field
from asgiref.sync import sync_to_async

from api_manager.apis.datacenter_api import DataCenterAPI
from api_manager.mappings.datacenter_mappings import (
    INDUSTRY_CAPITAL_FLOW_MAPPING, CONCEPT_CAPITAL_FLOW_MAPPING, NET_INFLOW_RANKING_MAPPING,
    NET_INFLOW_RATE_RANKING_MAPPING, MAIN_FORCE_NET_INFLOW_RANKING_MAPPING,
    MAIN_FORCE_NET_INFLOW_RATE_RANKING_MAPPING, RETAIL_NET_INFLOW_RANKING_MAPPING,
    RETAIL_NET_INFLOW_RATE_RANKING_MAPPING, INDUSTRY_CAPITAL_FLOW_ROUTE_MAPPING,
    CONCEPT_CAPITAL_FLOW_ROUTE_MAPPING, STOCK_PERIOD_STATISTICS_OVERVIEW_MAPPING,
    STOCK_PERIOD_STATISTICS_MAPPING, MAIN_FORCE_CONTINUOUS_FLOW_MAPPING,
    NEW_CAPITAL_FLOW_OVERVIEW_MAPPING
)
from dao_manager.base_dao import BaseDAO
from stock_models.datacenter.capital_flow import ConceptCapitalFlow, IndustryCapitalFlow, NetInflowRanking, NetInflowRateRanking, MainForceNetInflowRanking, MainForceNetInflowRateRanking, RetailNetInflowRanking, RetailNetInflowRateRanking, ConceptCapitalFlowRoute, StockPeriodStatisticsOverview, StockPeriodStatistics, MainForceContinuousFlow, NewCapitalFlowOverview

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

class CapitalFlowDao(BaseDAO):
    """
    资金流向数据访问对象
    
    负责处理与资金流向相关的数据访问操作，包括：
    1. 行业资金流向数据的获取和保存
    2. 概念板块资金流向数据的获取和保存
    3. 个股资金流向数据的获取和保存
    
    实现了三层数据获取机制：
    1. 先从Redis缓存获取
    2. 如缓存未命中，从MySQL数据库获取
    3. 如数据库无数据，从API获取并保存到数据库和缓存
    
    所有方法都实现了异步处理，提高并发性能
    """
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
                    elif (isinstance(value, (int, float)) or (
                        isinstance(value, str) and value.replace('.', '', 1).isdigit()
                    )) and 'code' not in field.name.lower():
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
                        elif (isinstance(value, (int, float)) or (
                            isinstance(value, str) and value.replace('.', '', 1).isdigit()
                        )) and 'code' not in field.name.lower():
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
                    elif (isinstance(value, (int, float)) or (
                        isinstance(value, str) and value.replace('.', '', 1).isdigit()
                    )) and 'code' not in field.name.lower():
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
                        elif (isinstance(value, (int, float)) or (
                            isinstance(value, str) and value.replace('.', '', 1).isdigit()
                        )) and 'code' not in field.name.lower():
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


    async def save_industry_capital_flow(self) -> Dict:
        """
        保存行业资金流向数据
        
        Returns:
            dict: 操作结果统计
        """
        try:
            # 从API获取数据
            async with DataCenterAPI() as api:
                data_list = await api.get_industry_capital_flow()
            
            if not data_list:
                logger.warning("未获取到行业资金流向数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 检查是否是字符串类型（text/plain或text/html格式）
            if isinstance(data_list, str):
                try:
                    # 尝试解析JSON字符串
                    import json
                    if data_list.strip() in ['{}', '[]', '[{}]']:
                        logger.warning("API返回了空的JSON数据")
                        return {'创建': 0, '更新': 0, '跳过': 0}
                    data_list = json.loads(data_list)
                except json.JSONDecodeError:
                    logger.error("转换行业资金流向数据失败，无法解析为JSON")
                    return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 确保data_list是列表类型
            if not isinstance(data_list, list):
                data_list = [data_list] if data_list else []
            
            # 设置当前日期作为交易日期以确保不为空
            current_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            
            # 处理数据
            processed_data = []
            for data in data_list:
                if not isinstance(data, dict):
                    logger.warning(f"跳过无效数据格式: {data}")
                    continue
                
                # 使用标准字典格式处理数据并使用映射关系
                processed_item = {}
                for api_field, model_field in INDUSTRY_CAPITAL_FLOW_MAPPING.items():
                    if api_field in data:
                        # 日期字段处理
                        if model_field.endswith('_time') or model_field.endswith('_date') or model_field == 'trade_time':
                            processed_item[model_field] = self._parse_datetime(data.get(api_field, current_date))
                        # 数值字段处理
                        elif any(field in model_field for field in ['net_inflow', 'rate', 'change']):
                            processed_item[model_field] = self._parse_number(data.get(api_field, 0))
                        else:
                            processed_item[model_field] = data.get(api_field, '')
                
                # 验证必要字段
                if not processed_item.get('industry_name'):
                    logger.warning(f"跳过缺少必要字段的数据: {data}")
                    continue
                
                processed_data.append(processed_item)
            
            if not processed_data:
                logger.warning("没有有效的行业资金流向数据需要保存")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 保存数据
            result = await self._batch_process(
                model_class=IndustryCapitalFlow,
                data_list=processed_data,
                mapping=INDUSTRY_CAPITAL_FLOW_MAPPING,
                unique_fields=['industry_code', 'trade_time']
            )
            
            # 更新缓存，确保数据是字典格式
            cache_key = 'industry_capital_flow'
            
            # 直接使用处理过的数据，不需要重新转换
            await self._set_to_cache(cache_key, processed_data, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"保存行业资金流向数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}
       
    async def get_industry_capital_flow(self, date_str: Optional[str] = None) -> List[Dict]:
        """
        获取行业资金流向数据
        
        Args:
            date_str: 日期字符串，格式为'YYYY-MM-DD'，如果为None则获取最新日期数据
            
        Returns:
            List[Dict]: 行业资金流向数据列表
        """
        try:
            # 构建缓存键
            cache_key = f'industry_capital_flow_{date_str or "latest"}'
            
            # 1. 先从缓存获取
            cached_data = await self._get_from_cache(cache_key)
            if cached_data:
                logger.debug(f"缓存命中: {cache_key}")
                return cached_data
            
            logger.debug(f"缓存未命中: {cache_key}")
            
            # 2. 从数据库获取
            if date_str:
                # 尝试解析日期
                target_date = self._parse_datetime(date_str)
                
                # 查询数据库
                records = await asyncio.to_thread(
                    lambda: list(IndustryCapitalFlow.objects.filter(trade_time=target_date).order_by('-net_inflow_rate'))
                )
            else:
                # 获取最新日期
                latest_date_record = await asyncio.to_thread(
                    lambda: IndustryCapitalFlow.objects.order_by('-trade_time').values('trade_time').first()
                )
                
                if not latest_date_record:
                    return []
                
                latest_date = latest_date_record['trade_time']
                
                # 查询数据库
                records = await asyncio.to_thread(
                    lambda: list(IndustryCapitalFlow.objects.filter(trade_time=latest_date).order_by('-net_inflow_rate'))
                )
            
            if not records:
                return []
            
            # 将查询结果转换为字典列表
            result = []
            for record in records:
                item_dict = {}
                for field in get_model_fields(IndustryCapitalFlow):
                    value = getattr(record, field)
                    # 处理日期和时间
                    if field.endswith('_time') or field.endswith('_date') or field == 'trade_time':
                        item_dict[field] = self._parse_datetime(value)
                        if isinstance(item_dict[field], datetime):
                            item_dict[field] = item_dict[field].strftime('%Y-%m-%d') if field == 'trade_time' else item_dict[field].strftime('%Y-%m-%d %H:%M:%S')
                    # 处理数值
                    elif (isinstance(value, (int, float))) and 'code' not in field.name.lower():
                        item_dict[field] = self._parse_number(value)
                    else:
                        item_dict[field] = value
                result.append(item_dict)
            
            # 更新缓存
            await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"获取行业资金流向数据出错: {str(e)}")
            return []

    async def save_concept_capital_flow(self) -> Dict:
        """
        保存概念资金流向数据
        
        Returns:
            dict: 操作结果统计
        """
        try:
            # 从API获取数据
            async with DataCenterAPI() as api:
                data_list = await api.get_concept_capital_flow()
            
            if not data_list:
                logger.warning("未获取到概念资金流向数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 检查是否是字符串类型（text/plain或text/html格式）
            if isinstance(data_list, str):
                try:
                    # 尝试解析JSON字符串
                    import json
                    if data_list.strip() in ['{}', '[]', '[{}]']:
                        logger.warning("API返回了空的JSON数据")
                        return {'创建': 0, '更新': 0, '跳过': 0}
                    data_list = json.loads(data_list)
                except json.JSONDecodeError:
                    logger.error("转换概念资金流向数据失败，无法解析为JSON")
                    return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 确保data_list是列表类型
            if not isinstance(data_list, list):
                data_list = [data_list] if data_list else []
            
            # 设置当前日期作为交易日期以确保不为空
            current_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            
            # 处理数据
            processed_data = []
            for data in data_list:
                if not isinstance(data, dict):
                    logger.warning(f"跳过无效数据格式: {data}")
                    continue
                
                # 使用标准字典格式处理数据并使用映射关系
                processed_item = {}
                for api_field, model_field in CONCEPT_CAPITAL_FLOW_MAPPING.items():
                    if api_field in data:
                        # 日期字段处理
                        if model_field.endswith('_time') or model_field.endswith('_date') or model_field == 'trade_time':
                            processed_item[model_field] = self._parse_datetime(data.get(api_field, current_date))
                        # 数值字段处理
                        elif any(field in model_field for field in ['net_inflow', 'rate', 'change']):
                            processed_item[model_field] = self._parse_number(data.get(api_field, 0))
                        else:
                            processed_item[model_field] = data.get(api_field, '')
                
                # 添加更新时间字段
                processed_item['trade_time'] = self._parse_datetime(current_date)
                
                # 验证必要字段
                if not processed_item.get('concept_name'):
                    logger.warning(f"跳过缺少必要字段的数据: {data}")
                    continue
                
                processed_data.append(processed_item)
            
            if not processed_data:
                logger.warning("没有有效的概念资金流向数据需要保存")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 保存数据
            result = await self._batch_process(
                model_class=ConceptCapitalFlow,
                data_list=processed_data,
                mapping=CONCEPT_CAPITAL_FLOW_MAPPING,
                unique_fields=['concept_code', 'trade_time']
            )
            
            # 更新缓存，确保数据是字典格式
            cache_key = 'concept_capital_flow'
            
            # 直接使用处理过的数据，不需要重新转换
            await self._set_to_cache(cache_key, processed_data, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"保存概念资金流向数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def get_concept_capital_flow(self, date_str: Optional[str] = None) -> List[Dict]:
        """
        获取概念板块资金流向数据
        
        Args:
            date_str: 日期字符串，格式为'YYYY-MM-DD'，如果为None则获取最新日期数据
            
        Returns:
            List[Dict]: 概念板块资金流向数据列表
        """
        try:
            # 构建缓存键
            cache_key = f'concept_capital_flow_{date_str or "latest"}'
            
            # 1. 先从缓存获取
            cached_data = await self._get_from_cache(cache_key)
            if cached_data:
                logger.debug(f"缓存命中: {cache_key}")
                return cached_data
            
            logger.debug(f"缓存未命中: {cache_key}")
            
            # 2. 从数据库获取
            if date_str:
                # 尝试解析日期
                target_date = self._parse_datetime(date_str)
                
                # 查询数据库
                records = await asyncio.to_thread(
                    lambda: list(ConceptCapitalFlow.objects.filter(trade_time=target_date).order_by('-net_inflow_rate'))
                )
            else:
                # 获取最新日期
                latest_date_record = await asyncio.to_thread(
                    lambda: ConceptCapitalFlow.objects.order_by('-trade_time').values('trade_time').first()
                )
                
                if not latest_date_record:
                    return []
                
                latest_date = latest_date_record['trade_time']
                
                # 查询数据库
                records = await asyncio.to_thread(
                    lambda: list(ConceptCapitalFlow.objects.filter(trade_time=latest_date).order_by('-net_inflow_rate'))
                )
            
            if not records:
                return []
            
            # 将查询结果转换为字典列表
            result = []
            for record in records:
                item_dict = {}
                for field in get_model_fields(ConceptCapitalFlow):
                    value = getattr(record, field)
                    # 处理日期和时间
                    if field.endswith('_time') or field.endswith('_date') or field == 'trade_time':
                        item_dict[field] = self._parse_datetime(value)
                        if isinstance(item_dict[field], datetime):
                            item_dict[field] = item_dict[field].strftime('%Y-%m-%d') if field == 'trade_time' else item_dict[field].strftime('%Y-%m-%d %H:%M:%S')
                    # 处理数值
                    elif (isinstance(value, (int, float))) and 'code' not in field.name.lower():
                        item_dict[field] = self._parse_number(value)
                    else:
                        item_dict[field] = value
                result.append(item_dict)
            
            # 更新缓存
            await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"获取概念板块资金流向数据出错: {str(e)}")
            return []

    async def save_stock_period_statistics_overview(self) -> Dict:
        """
        保存个股阶段统计总览数据
        
        Returns:
            dict: 操作结果统计
        """
        try:
            # 从API获取数据
            async with DataCenterAPI() as api:
                data_list = await api.get_stock_period_statistics_overview()
            
            if not data_list:
                logger.warning("未获取到个股阶段统计总览数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 检查是否是字符串类型（text/plain或text/html格式）
            if isinstance(data_list, str):
                try:
                    # 尝试解析JSON字符串
                    import json
                    if data_list.strip() in ['{}', '[]', '[{}]']:
                        logger.warning("个股阶段统计总览数据 - API返回了空的JSON数据")
                        return {'创建': 0, '更新': 0, '跳过': 0}
                    data_list = json.loads(data_list)
                except json.JSONDecodeError:
                    logger.error("转换个股阶段统计总览数据失败，无法解析为JSON")
                    return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 确保data_list是列表类型
            if not isinstance(data_list, list):
                data_list = [data_list] if data_list else []
            
            # 设置当前日期作为交易日期以确保不为空
            current_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            
            # 处理数据
            processed_data = []
            for data in data_list:
                if not isinstance(data, dict):
                    logger.warning(f"个股阶段统计总览数据 - 跳过无效数据格式: {data}")
                    continue
                
                # 使用标准字典格式处理数据并使用映射关系
                processed_item = {}
                for api_field, model_field in STOCK_PERIOD_STATISTICS_OVERVIEW_MAPPING.items():
                    if api_field in data:
                        # 日期字段处理
                        if model_field.endswith('_time') or model_field.endswith('_date') or model_field == 'trade_time':
                            processed_item[model_field] = self._parse_datetime(data.get(api_field, current_date))
                        # 数值字段处理
                        elif any(field in model_field for field in ['net_inflow', 'rate', 'change']):
                            processed_item[model_field] = self._parse_number(data.get(api_field, 0))
                        else:
                            processed_item[model_field] = data.get(api_field, '')
                
                # 验证必要字段
                if not processed_item.get('stock_code'):
                    logger.warning(f"个股阶段统计总览数据 - 跳过缺少必要字段的数据: {data}")
                    continue
                
                processed_data.append(processed_item)
            
            if not processed_data:
                logger.warning("没有有效的个股阶段统计总览数据需要保存")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 保存数据
            result = await self._batch_process(
                model_class=StockPeriodStatisticsOverview,
                data_list=processed_data,
                mapping=STOCK_PERIOD_STATISTICS_OVERVIEW_MAPPING,
                unique_fields=['stock_code', 'trade_time']
            )
            
            # 更新缓存，确保数据是字典格式
            cache_key = 'stock_period_statistics_overview'
            
            # 直接使用处理过的数据，不需要重新转换
            await self._set_to_cache(cache_key, processed_data, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"保存个股阶段统计总览数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}   

    async def get_stock_period_statistics_overview(self, date_str: Optional[str] = None) -> List[Dict]:
        """
        获取个股阶段统计总览数据
        
        Args:
            date_str: 日期字符串，格式为'YYYY-MM-DD'，如果为None则获取最新日期数据
            
        Returns:
            List[Dict]: 个股阶段统计总览数据列表
        """
        try:
            # 构建缓存键
            cache_key = f'stock_period_statistics_overview_{date_str or "latest"}'
            
            # 1. 先从缓存获取
            cached_data = await self._get_from_cache(cache_key)
            if cached_data:
                logger.debug(f"缓存命中: {cache_key}")
                return cached_data
            
            logger.debug(f"缓存未命中: {cache_key}")
            
            # 2. 从数据库获取
            if date_str:
                # 尝试解析日期
                target_date = self._parse_datetime(date_str)
                
                # 查询数据库
                records = await asyncio.to_thread(
                    lambda: list(StockPeriodStatisticsOverview.objects.filter(trade_time=target_date).order_by('-main_force_net_inflow'))
                )
            else:
                # 获取最新日期
                latest_date_record = await asyncio.to_thread(
                    lambda: StockPeriodStatisticsOverview.objects.order_by('-trade_time').values('trade_time').first()
                )
                
                if not latest_date_record:
                    return []
                
                latest_date = latest_date_record['trade_time']
                
                # 查询数据库
                records = await asyncio.to_thread(
                    lambda: list(StockPeriodStatisticsOverview.objects.filter(trade_time=latest_date).order_by('-main_force_net_inflow'))
                )
            
            if not records:
                return []
            
            # 将查询结果转换为字典列表
            result = []
            for record in records:
                item_dict = {}
                for field in get_model_fields(StockPeriodStatisticsOverview):
                    value = getattr(record, field)
                    # 处理日期和时间
                    if field.endswith('_time') or field.endswith('_date') or field == 'trade_time':
                        item_dict[field] = self._parse_datetime(value)
                        if isinstance(item_dict[field], datetime):
                            item_dict[field] = item_dict[field].strftime('%Y-%m-%d') if field == 'trade_time' else item_dict[field].strftime('%Y-%m-%d %H:%M:%S')
                    # 处理数值
                    elif (isinstance(value, (int, float))) and 'code' not in field.name.lower():
                        item_dict[field] = self._parse_number(value)
                    else:
                        item_dict[field] = value
                result.append(item_dict)
            
            # 更新缓存
            await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"获取个股阶段统计总览数据出错: {str(e)}")
            return []
    
    async def save_net_inflow_amount_rank(self) -> Dict:
        """
        保存净流入额排名数据
        
        Returns:
            dict: 操作结果统计
        """
        try:
            # 从API获取数据
            async with DataCenterAPI() as api:
                data_list = await api.get_net_inflow_amount_rank()
            
            if not data_list:
                logger.warning("未获取到净流入额排名数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 检查是否是字符串类型（text/plain或text/html格式）
            if isinstance(data_list, str):
                try:
                    # 尝试解析JSON字符串
                    import json
                    if data_list.strip() in ['{}', '[]', '[{}]']:
                        logger.warning("净流入额排名数据 - API返回了空的JSON数据")
                        return {'创建': 0, '更新': 0, '跳过': 0}
                    data_list = json.loads(data_list)
                except json.JSONDecodeError:
                    logger.error("转换净流入额排名数据失败，无法解析为JSON")
                    return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 确保data_list是列表类型
            if not isinstance(data_list, list):
                data_list = [data_list] if data_list else []
            
            # 设置当前日期作为交易日期以确保不为空
            current_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            
            # 处理数据
            processed_data = []
            for data in data_list:
                if not isinstance(data, dict):
                    logger.warning(f"跳过无效数据格式: {data}")
                    continue
                
                # 使用标准字典格式处理数据并使用映射关系
                processed_item = {}
                for api_field, model_field in NET_INFLOW_RANKING_MAPPING.items():
                    if api_field in data:
                        # 日期字段处理
                        if model_field.endswith('_time') or model_field.endswith('_date') or model_field == 'trade_time':
                            processed_item[model_field] = self._parse_datetime(data.get(api_field, current_date))
                        # 数值字段处理
                        elif any(field in model_field for field in ['net_inflow', 'rate', 'change']):
                            processed_item[model_field] = self._parse_number(data.get(api_field, 0))
                        else:
                            processed_item[model_field] = data.get(api_field, '')
                
                # 验证必要字段
                if not processed_item.get('stock_code'):
                    logger.warning(f"净流入额排名数据 - 跳过缺少必要字段的数据: {data}")
                    continue
                
                processed_data.append(processed_item)
            
            if not processed_data:
                logger.warning("没有有效的净流入额排名数据需要保存")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 保存数据
            result = await self._batch_process(
                model_class=NetInflowRanking,
                data_list=processed_data,
                mapping=NET_INFLOW_RANKING_MAPPING,
                unique_fields=['stock_code', 'trade_time']
            )
            
            # 更新缓存，确保数据是字典格式
            cache_key = 'net_inflow_amount_rank'
            
            # 直接使用处理过的数据，不需要重新转换
            await self._set_to_cache(cache_key, processed_data, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"保存净流入额排名数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}   

    async def get_net_inflow_amount_rank(self, date_str: Optional[str] = None) -> List[Dict]:
        """
        获取净流入额排名数据
        
        Args:
            date_str: 日期字符串，格式为'YYYY-MM-DD'，如果为None则获取最新日期数据
            
        Returns:
            List[Dict]: 净流入额排名数据列表
        """
        try:
            # 构建缓存键
            cache_key = f'net_inflow_amount_rank_{date_str or "latest"}'
            
            # 1. 先从缓存获取
            cached_data = await self._get_from_cache(cache_key)
            if cached_data:
                logger.debug(f"净流入额排名数据 - 缓存命中: {cache_key}")
                return cached_data
            
            logger.debug(f"净流入额排名数据 - 缓存未命中: {cache_key}")
            
            # 2. 从数据库获取
            if date_str:
                # 尝试解析日期
                target_date = self._parse_datetime(date_str)
                
                # 查询数据库
                records = await asyncio.to_thread(
                    lambda: list(NetInflowRanking.objects.filter(trade_time=target_date).order_by('-net_inflow_rate'))
                )
            else:
                # 获取最新日期
                latest_date_record = await asyncio.to_thread(
                    lambda: NetInflowRanking.objects.order_by('-trade_time').values('trade_time').first()
                )
                
                if not latest_date_record:
                    return []
                
                latest_date = latest_date_record['trade_time']
                
                # 查询数据库
                records = await asyncio.to_thread(
                    lambda: list(NetInflowRanking.objects.filter(trade_time=latest_date).order_by('-net_inflow_rate'))
                )
            
            if not records:
                return []
            
            # 将查询结果转换为字典列表
            result = []
            for record in records:
                item_dict = {}
                for field in get_model_fields(NetInflowRanking):
                    value = getattr(record, field)
                    # 处理日期和时间
                    if field.endswith('_time') or field.endswith('_date') or field == 'trade_time':
                        item_dict[field] = self._parse_datetime(value)
                        if isinstance(item_dict[field], datetime):
                            item_dict[field] = item_dict[field].strftime('%Y-%m-%d') if field == 'trade_time' else item_dict[field].strftime('%Y-%m-%d %H:%M:%S')
                    # 处理数值
                    elif (isinstance(value, (int, float))) and 'code' not in field.name.lower():
                        item_dict[field] = self._parse_number(value)
                    else:
                        item_dict[field] = value
                result.append(item_dict)
            
            # 更新缓存
            await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"获取净流入额排名数据出错: {str(e)}")
            return []
    
    async def save_net_inflow_rate_rank(self) -> Dict:
        """
        保存净流入率排名数据
        
        Returns:
            dict: 操作结果统计
        """
        try:
            # 从API获取数据
            async with DataCenterAPI() as api:
                data_list = await api.get_net_inflow_rate_rank()
            
            if not data_list:
                logger.warning("未获取到 净流入率排名数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 检查是否是字符串类型（text/plain或text/html格式）
            if isinstance(data_list, str):
                try:
                    # 尝试解析JSON字符串
                    import json
                    if data_list.strip() in ['{}', '[]', '[{}]']:
                        logger.warning("净流入率排名数据 - API返回了空的JSON数据")
                        return {'创建': 0, '更新': 0, '跳过': 0}
                    data_list = json.loads(data_list)
                except json.JSONDecodeError:
                    logger.error("转换 净流入率排名数据 失败，无法解析为JSON")
                    return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 确保data_list是列表类型
            if not isinstance(data_list, list):
                data_list = [data_list] if data_list else []
            
            # 设置当前日期作为交易日期以确保不为空
            current_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            
            # 处理数据
            processed_data = []
            for data in data_list:
                if not isinstance(data, dict):
                    logger.warning(f"净流入率排名数据 - 跳过无效数据格式: {data}")
                    continue
                
                # 使用标准字典格式处理数据并使用映射关系
                processed_item = {}
                for api_field, model_field in NET_INFLOW_RATE_RANKING_MAPPING.items():
                    if api_field in data:
                        # 日期字段处理
                        if model_field.endswith('_time') or model_field.endswith('_date') or model_field == 'trade_time':
                            processed_item[model_field] = self._parse_datetime(data.get(api_field, current_date))
                        # 数值字段处理
                        elif any(field in model_field for field in ['net_inflow', 'rate', 'change']):
                            processed_item[model_field] = self._parse_number(data.get(api_field, 0))
                        else:
                            processed_item[model_field] = data.get(api_field, '')
                
                # 验证必要字段
                if not processed_item.get('stock_code'):
                    logger.warning(f"净流入率排名数据 - 跳过缺少必要字段的数据: {data}")
                    continue
                
                processed_data.append(processed_item)
            
            if not processed_data:
                logger.warning("没有有效的 净流入率排名数据 需要保存")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 保存数据
            result = await self._batch_process(
                model_class=NetInflowRateRanking,
                data_list=processed_data,
                mapping=NET_INFLOW_RATE_RANKING_MAPPING,
                unique_fields=['stock_code', 'trade_time']
            )
            
            # 更新缓存，确保数据是字典格式
            cache_key = 'net_inflow_rate_rank'
            
            # 直接使用处理过的数据，不需要重新转换
            await self._set_to_cache(cache_key, processed_data, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"保存净流入额排名数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}
    
    async def get_net_inflow_rate_rank(self, date_str: Optional[str] = None) -> List[Dict]:
        """
        获取净流入率排名数据
        
        Args:
            date_str: 日期字符串，格式为'YYYY-MM-DD'，如果为None则获取最新日期数据
            
        Returns:
            List[Dict]: 净流入率排名数据列表
        """
        try:
            # 构建缓存键
            cache_key = f'net_inflow_rate_rank_{date_str or "latest"}'
            
            # 1. 先从缓存获取
            cached_data = await self._get_from_cache(cache_key)
            if cached_data:
                logger.debug(f"净流入率排名数据 - 缓存命中: {cache_key}")
                return cached_data
            
            logger.debug(f"净流入率排名数据 - 缓存未命中: {cache_key}")
            
            # 2. 从数据库获取
            if date_str:
                # 尝试解析日期
                target_date = self._parse_datetime(date_str)
                
                # 查询数据库
                records = await asyncio.to_thread(
                    lambda: list(NetInflowRateRanking.objects.filter(trade_time=target_date).order_by('-net_inflow_rate'))
                )
            else:
                # 获取最新日期
                latest_date_record = await asyncio.to_thread(
                    lambda: NetInflowRateRanking.objects.order_by('-trade_time').values('trade_time').first()
                )
                
                if not latest_date_record:
                    return []
                
                latest_date = latest_date_record['trade_time']
                
                # 查询数据库
                records = await asyncio.to_thread(
                    lambda: list(NetInflowRateRanking.objects.filter(trade_time=latest_date).order_by('-net_inflow_rate'))
                )
            
            if not records:
                return []
            
            # 将查询结果转换为字典列表
            result = []
            for record in records:
                item_dict = {}
                for field in get_model_fields(NetInflowRateRanking):
                    value = getattr(record, field)
                    # 处理日期和时间
                    if field.endswith('_time') or field.endswith('_date') or field == 'trade_time':
                        item_dict[field] = self._parse_datetime(value)
                        if isinstance(item_dict[field], datetime):
                            item_dict[field] = item_dict[field].strftime('%Y-%m-%d') if field == 'trade_time' else item_dict[field].strftime('%Y-%m-%d %H:%M:%S')
                    # 处理数值
                    elif (isinstance(value, (int, float))) and 'code' not in field.name.lower():
                        item_dict[field] = self._parse_number(value)
                    else:
                        item_dict[field] = value
                result.append(item_dict)
            
            # 更新缓存
            await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"获取 净流入率排名数据 出错: {str(e)}")
            return []
    
    async def save_main_net_inflow_amount_rank(self) -> Dict:
        """
        保存主力净流入额排名数据
        
        Returns:
            dict: 操作结果统计
        """
        try:
            # 从API获取数据
            async with DataCenterAPI() as api:
                data_list = await api.get_main_net_inflow_amount_rank()
            
            if not data_list:
                logger.warning("未获取到主力净流入额排名数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 检查是否是字符串类型（text/plain或text/html格式）
            if isinstance(data_list, str):
                try:
                    # 尝试解析JSON字符串
                    import json
                    if data_list.strip() in ['{}', '[]', '[{}]']:
                        logger.warning("主力净流入额排名数据 - API返回了空的JSON数据")
                        return {'创建': 0, '更新': 0, '跳过': 0}
                    data_list = json.loads(data_list)
                except json.JSONDecodeError:
                    logger.error("转换主力净流入额排名数据失败，无法解析为JSON")
                    return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 确保data_list是列表类型
            if not isinstance(data_list, list):
                data_list = [data_list] if data_list else []
            
            # 设置当前日期作为交易日期以确保不为空
            current_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            
            # 处理数据
            processed_data = []
            for data in data_list:
                if not isinstance(data, dict):
                    logger.warning(f"主力净流入额排名数据 - 跳过无效数据格式: {data}")
                    continue
                
                # 使用标准字典格式处理数据并使用映射关系
                processed_item = {}
                for api_field, model_field in MAIN_FORCE_NET_INFLOW_RANKING_MAPPING.items():
                    if api_field in data:
                        # 日期字段处理
                        if model_field.endswith('_time') or model_field.endswith('_date') or model_field == 'trade_time':
                            processed_item[model_field] = self._parse_datetime(data.get(api_field, current_date))
                        # 数值字段处理
                        elif any(field in model_field for field in ['net_inflow', 'rate', 'change']):
                            processed_item[model_field] = self._parse_number(data.get(api_field, 0))
                        else:
                            processed_item[model_field] = data.get(api_field, '')
                
                # 验证必要字段
                if not processed_item.get('stock_code'):
                    logger.warning(f"主力净流入额排名数据 - 跳过缺少必要字段的数据: {data}")
                    continue
                
                processed_data.append(processed_item)
            
            if not processed_data:
                logger.warning("没有有效的 主力净流入额排名数据 需要保存")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 保存数据
            result = await self._batch_process(
                model_class=MainForceNetInflowRanking,
                data_list=processed_data,
                mapping=MAIN_FORCE_NET_INFLOW_RANKING_MAPPING,
                unique_fields=['stock_code', 'trade_time']
            )
            
            # 更新缓存，确保数据是字典格式
            cache_key = 'main_net_inflow_amount_rank'
            
            # 直接使用处理过的数据，不需要重新转换
            await self._set_to_cache(cache_key, processed_data, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"保存净流入额排名数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}
    
    async def get_main_net_inflow_amount_rank(self, date_str: Optional[str] = None) -> List[Dict]:
        """
        获取主力净流入额排名数据
        
        Args:
            date_str: 日期字符串，格式为'YYYY-MM-DD'，如果为None则获取最新日期数据
            
        Returns:
            List[Dict]: 主力净流入额排名数据列表
        """
        try:
            # 构建缓存键
            cache_key = f'main_net_inflow_amount_rank_{date_str or "latest"}'
            
            # 1. 先从缓存获取
            cached_data = await self._get_from_cache(cache_key)
            if cached_data:
                logger.debug(f"主力净流入额排名数据 - 缓存命中: {cache_key}")
                return cached_data
            
            logger.debug(f"主力净流入额排名数据 - 缓存未命中: {cache_key}")
            
            # 2. 从数据库获取
            if date_str:
                # 尝试解析日期
                target_date = self._parse_datetime(date_str)
                
                # 查询数据库
                records = await asyncio.to_thread(
                    lambda: list(MainForceNetInflowRanking.objects.filter(trade_time=target_date).order_by('-main_force_net_inflow_rate'))
                )
            else:
                # 获取最新日期
                latest_date_record = await asyncio.to_thread(
                    lambda: MainForceNetInflowRanking.objects.order_by('-trade_time').values('trade_time').first()
                )
                
                if not latest_date_record:
                    return []
                
                latest_date = latest_date_record['trade_time']
                
                # 查询数据库
                records = await asyncio.to_thread(
                    lambda: list(MainForceNetInflowRanking.objects.filter(trade_time=latest_date).order_by('-main_force_net_inflow_rate'))
                )
            
            if not records:
                return []
            
            # 将查询结果转换为字典列表
            result = []
            for record in records:
                item_dict = {}
                for field in get_model_fields(MainForceNetInflowRanking):
                    value = getattr(record, field)
                    # 处理日期和时间
                    if field.endswith('_time') or field.endswith('_date') or field == 'trade_time':
                        item_dict[field] = self._parse_datetime(value)
                        if isinstance(item_dict[field], datetime):
                            item_dict[field] = item_dict[field].strftime('%Y-%m-%d') if field == 'trade_time' else item_dict[field].strftime('%Y-%m-%d %H:%M:%S')
                    # 处理数值
                    elif (isinstance(value, (int, float))) and 'code' not in field.name.lower():
                        item_dict[field] = self._parse_number(value)
                    else:
                        item_dict[field] = value
                result.append(item_dict)
            
            # 更新缓存
            await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"获取 主力净流入额排名数据 出错: {str(e)}")
            return []
    
    async def save_main_net_inflow_rate_rank(self) -> Dict:
        """
        保存主力净流入率排名数据
        
        Returns:
            dict: 操作结果统计
        """
        try:
            # 从API获取数据
            async with DataCenterAPI() as api:
                data_list = await api.get_main_net_inflow_rate_rank()
            
            if not data_list:
                logger.warning("未获取到 主力净流入率排名数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 检查是否是字符串类型（text/plain或text/html格式）
            if isinstance(data_list, str):
                try:
                    # 尝试解析JSON字符串
                    import json
                    if data_list.strip() in ['{}', '[]', '[{}]']:
                        logger.warning("主力净流入率排名数据 - API返回了空的JSON数据")
                        return {'创建': 0, '更新': 0, '跳过': 0}
                    data_list = json.loads(data_list)
                except json.JSONDecodeError:
                    logger.error("转换 主力净流入率排名数据 失败，无法解析为JSON")
                    return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 确保data_list是列表类型
            if not isinstance(data_list, list):
                data_list = [data_list] if data_list else []
            
            # 设置当前日期作为交易日期以确保不为空
            current_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            
            # 处理数据
            processed_data = []
            for data in data_list:
                if not isinstance(data, dict):
                    logger.warning(f"主力净流入率排名数据 - 跳过无效数据格式: {data}")
                    continue
                
                # 使用标准字典格式处理数据并使用映射关系
                processed_item = {}
                for api_field, model_field in MAIN_FORCE_NET_INFLOW_RATE_RANKING_MAPPING.items():
                    if api_field in data:
                        # 日期字段处理
                        if model_field.endswith('_time') or model_field.endswith('_date') or model_field == 'trade_time':
                            processed_item[model_field] = self._parse_datetime(data.get(api_field, current_date))
                        # 数值字段处理
                        elif any(field in model_field for field in ['net_inflow', 'rate', 'change']):
                            processed_item[model_field] = self._parse_number(data.get(api_field, 0))
                        else:
                            processed_item[model_field] = data.get(api_field, '')
                
                # 验证必要字段
                if not processed_item.get('stock_code'):
                    logger.warning(f"主力净流入率排名数据 - 跳过缺少必要字段的数据: {data}")
                    continue
                
                processed_data.append(processed_item)
            
            if not processed_data:
                logger.warning("没有有效的 主力净流入率排名数据 需要保存")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 保存数据
            result = await self._batch_process(
                model_class=MainForceNetInflowRateRanking,
                data_list=processed_data,
                mapping=MAIN_FORCE_NET_INFLOW_RATE_RANKING_MAPPING,
                unique_fields=['stock_code', 'trade_time']
            )
            
            # 更新缓存，确保数据是字典格式
            cache_key = 'main_net_inflow_rate_rank'
            
            # 直接使用处理过的数据，不需要重新转换
            await self._set_to_cache(cache_key, processed_data, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"保存净流入额排名数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def get_main_net_inflow_rate_rank(self, date_str: Optional[str] = None) -> List[Dict]:
        """
        获取主力净流入率排名数据
        
        Args:
            date_str: 日期字符串，格式为'YYYY-MM-DD'，如果为None则获取最新日期数据
            
        Returns:
            List[Dict]: 主力净流入率排名数据列表
        """
        try:
            # 构建缓存键
            cache_key = f'main_net_inflow_rate_rank_{date_str or "latest"}'
            
            # 1. 先从缓存获取
            cached_data = await self._get_from_cache(cache_key)
            if cached_data:
                logger.debug(f"主力净流入率排名数据 - 缓存命中: {cache_key}")
                return cached_data
            
            logger.debug(f"主力净流入率排名数据 - 缓存未命中: {cache_key}")
            
            # 2. 从数据库获取
            if date_str:
                # 尝试解析日期
                target_date = self._parse_datetime(date_str)
                
                # 查询数据库
                records = await asyncio.to_thread(
                    lambda: list(MainForceNetInflowRateRanking.objects.filter(trade_time=target_date).order_by('-main_force_net_inflow_rate'))
                )
            else:
                # 获取最新日期
                latest_date_record = await asyncio.to_thread(
                    lambda: MainForceNetInflowRateRanking.objects.order_by('-trade_time').values('trade_time').first()
                )
                
                if not latest_date_record:
                    return []
                
                latest_date = latest_date_record['trade_time']
                
                # 查询数据库
                records = await asyncio.to_thread(
                    lambda: list(MainForceNetInflowRateRanking.objects.filter(trade_time=latest_date).order_by('-main_force_net_inflow_rate'))
                )
            
            if not records:
                return []
            
            # 将查询结果转换为字典列表
            result = []
            for record in records:
                item_dict = {}
                for field in get_model_fields(MainForceNetInflowRateRanking):
                    value = getattr(record, field)
                    # 处理日期和时间
                    if field.endswith('_time') or field.endswith('_date') or field == 'trade_time':
                        item_dict[field] = self._parse_datetime(value)
                        if isinstance(item_dict[field], datetime):
                            item_dict[field] = item_dict[field].strftime('%Y-%m-%d') if field == 'trade_time' else item_dict[field].strftime('%Y-%m-%d %H:%M:%S')
                    # 处理数值
                    elif (isinstance(value, (int, float))) and 'code' not in field.name.lower():
                        item_dict[field] = self._parse_number(value)
                    else:
                        item_dict[field] = value
                result.append(item_dict)
            
            # 更新缓存
            await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"获取 主力净流入率排名数据 出错: {str(e)}")
            return []
    
    async def save_retail_net_inflow_amount_rank(self) -> Dict:
        """
        保存散户净流入额排名数据
        
        Returns:
            dict: 操作结果统计
        """
        try:
            # 从API获取数据
            async with DataCenterAPI() as api:
                data_list = await api.get_retail_net_inflow_amount_rank()
            
            if not data_list:
                logger.warning("未获取到 散户净流入额排名数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 检查是否是字符串类型（text/plain或text/html格式）
            if isinstance(data_list, str):
                try:
                    # 尝试解析JSON字符串
                    import json
                    if data_list.strip() in ['{}', '[]', '[{}]']:
                        logger.warning("散户净流入额排名数据 - API返回了空的JSON数据")
                        return {'创建': 0, '更新': 0, '跳过': 0}
                    data_list = json.loads(data_list)
                except json.JSONDecodeError:
                    logger.error("转换 散户净流入额排名数据 失败，无法解析为JSON")
                    return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 确保data_list是列表类型
            if not isinstance(data_list, list):
                data_list = [data_list] if data_list else []
            
            # 设置当前日期作为交易日期以确保不为空
            current_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            
            # 处理数据
            processed_data = []
            for data in data_list:
                if not isinstance(data, dict):
                    logger.warning(f"散户净流入额排名数据 - 跳过无效数据格式: {data}")
                    continue
                
                # 使用标准字典格式处理数据并使用映射关系
                processed_item = {}
                for api_field, model_field in RETAIL_NET_INFLOW_RANKING_MAPPING.items():
                    if api_field in data:
                        # 日期字段处理
                        if model_field.endswith('_time') or model_field.endswith('_date') or model_field == 'trade_time':
                            processed_item[model_field] = self._parse_datetime(data.get(api_field, current_date))
                        # 数值字段处理
                        elif any(field in model_field for field in ['net_inflow', 'rate', 'change']):
                            processed_item[model_field] = self._parse_number(data.get(api_field, 0))
                        else:
                            processed_item[model_field] = data.get(api_field, '')
                
                # 验证必要字段
                if not processed_item.get('stock_code'):
                    logger.warning(f"散户净流入额排名数据 - 跳过缺少必要字段的数据: {data}")
                    continue
                
                processed_data.append(processed_item)
            
            if not processed_data:
                logger.warning("没有有效的 散户净流入额排名数据 需要保存")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 保存数据
            result = await self._batch_process(
                model_class=RetailNetInflowRanking,
                data_list=processed_data,
                mapping=RETAIL_NET_INFLOW_RANKING_MAPPING,
                unique_fields=['stock_code', 'trade_time']
            )
            
            # 更新缓存，确保数据是字典格式
            cache_key = 'retail_net_inflow_amount_rank'
            
            # 直接使用处理过的数据，不需要重新转换
            await self._set_to_cache(cache_key, processed_data, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"保存净流入额排名数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}
    
    async def get_retail_net_inflow_amount_rank(self, date_str: Optional[str] = None) -> List[Dict]:
        """
        获取散户净流入额排名数据
        
        Args:
            date_str: 日期字符串，格式为'YYYY-MM-DD'，如果为None则获取最新日期数据
            
        Returns:
            List[Dict]: 散户净流入额排名数据列表
        """
        try:
            # 构建缓存键
            cache_key = f'retail_net_inflow_amount_rank_{date_str or "latest"}'
            
            # 1. 先从缓存获取
            cached_data = await self._get_from_cache(cache_key)
            if cached_data:
                logger.debug(f"散户净流入额排名数据 - 缓存命中: {cache_key}")
                return cached_data
            
            logger.debug(f"散户净流入额排名数据 - 缓存未命中: {cache_key}")
            
            # 2. 从数据库获取
            if date_str:
                # 尝试解析日期
                target_date = self._parse_datetime(date_str)
                
                # 查询数据库
                records = await asyncio.to_thread(
                    lambda: list(RetailNetInflowRanking.objects.filter(trade_time=target_date).order_by('-retail_net_inflow_rate'))
                )
            else:
                # 获取最新日期
                latest_date_record = await asyncio.to_thread(
                    lambda: RetailNetInflowRanking.objects.order_by('-trade_time').values('trade_time').first()
                )
                
                if not latest_date_record:
                    return []
                
                latest_date = latest_date_record['trade_time']
                
                # 查询数据库
                records = await asyncio.to_thread(
                    lambda: list(RetailNetInflowRanking.objects.filter(trade_time=latest_date).order_by('-retail_net_inflow_rate'))
                )
            
            if not records:
                return []
            
            # 将查询结果转换为字典列表
            result = []
            for record in records:
                item_dict = {}
                for field in get_model_fields(RetailNetInflowRanking):
                    value = getattr(record, field)
                    # 处理日期和时间
                    if field.endswith('_time') or field.endswith('_date') or field == 'trade_time':
                        item_dict[field] = self._parse_datetime(value)
                        if isinstance(item_dict[field], datetime):
                            item_dict[field] = item_dict[field].strftime('%Y-%m-%d') if field == 'trade_time' else item_dict[field].strftime('%Y-%m-%d %H:%M:%S')
                    # 处理数值
                    elif (isinstance(value, (int, float))) and 'code' not in field.name.lower():
                        item_dict[field] = self._parse_number(value)
                    else:
                        item_dict[field] = value
                result.append(item_dict)
            
            # 更新缓存
            await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"获取 散户净流入额排名数据 出错: {str(e)}")
            return []
    
    async def save_retail_net_inflow_rate_rank(self) -> Dict:
        """
        保存散户净流入率排名数据
        
        Returns:
            dict: 操作结果统计
        """
        try:
            # 从API获取数据
            async with DataCenterAPI() as api:
                data_list = await api.get_retail_net_inflow_rate_rank()
            
            if not data_list:
                logger.warning("未获取到 散户净流入率排名数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 检查是否是字符串类型（text/plain或text/html格式）
            if isinstance(data_list, str):
                try:
                    # 尝试解析JSON字符串
                    import json
                    if data_list.strip() in ['{}', '[]', '[{}]']:
                        logger.warning("散户净流入率排名数据 - API返回了空的JSON数据")
                        return {'创建': 0, '更新': 0, '跳过': 0}
                    data_list = json.loads(data_list)
                except json.JSONDecodeError:
                    logger.error("转换 散户净流入率排名数据 失败，无法解析为JSON")
                    return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 确保data_list是列表类型
            if not isinstance(data_list, list):
                data_list = [data_list] if data_list else []
            
            # 设置当前日期作为交易日期以确保不为空
            current_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            
            # 处理数据
            processed_data = []
            for data in data_list:
                if not isinstance(data, dict):
                    logger.warning(f"散户净流入率排名数据 - 跳过无效数据格式: {data}")
                    continue
                
                # 使用标准字典格式处理数据并使用映射关系
                processed_item = {}
                for api_field, model_field in RETAIL_NET_INFLOW_RATE_RANKING_MAPPING.items():
                    if api_field in data:
                        # 日期字段处理
                        if model_field.endswith('_time') or model_field.endswith('_date') or model_field == 'trade_time':
                            processed_item[model_field] = self._parse_datetime(data.get(api_field, current_date))
                        # 数值字段处理
                        elif any(field in model_field for field in ['net_inflow', 'rate', 'change']):
                            processed_item[model_field] = self._parse_number(data.get(api_field, 0))
                        else:
                            processed_item[model_field] = data.get(api_field, '')
                
                # 验证必要字段
                if not processed_item.get('stock_code'):
                    logger.warning(f"散户净流入率排名数据 - 跳过缺少必要字段的数据: {data}")
                    continue
                
                processed_data.append(processed_item)
            
            if not processed_data:
                logger.warning("没有有效的 散户净流入率排名数据 需要保存")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 保存数据
            result = await self._batch_process(
                model_class=RetailNetInflowRateRanking,
                data_list=processed_data,
                mapping=RETAIL_NET_INFLOW_RATE_RANKING_MAPPING,
                unique_fields=['stock_code', 'trade_time']
            )
            
            # 更新缓存，确保数据是字典格式
            cache_key = 'retail_net_inflow_rate_rank'
            
            # 直接使用处理过的数据，不需要重新转换
            await self._set_to_cache(cache_key, processed_data, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"保存净流入额排名数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def get_retail_net_inflow_rate_rank(self, date_str: Optional[str] = None) -> List[Dict]:
        """
        获取散户净流入率排名数据
        
        Args:
            date_str: 日期字符串，格式为'YYYY-MM-DD'，如果为None则获取最新日期数据
            
        Returns:
            List[Dict]: 散户净流入率排名数据列表
        """
        try:
            # 构建缓存键
            cache_key = f'retail_net_inflow_rate_rank_{date_str or "latest"}'
            
            # 1. 先从缓存获取
            cached_data = await self._get_from_cache(cache_key)
            if cached_data:
                logger.debug(f"散户净流入率排名数据 - 缓存命中: {cache_key}")
                return cached_data
            
            logger.debug(f"散户净流入率排名数据 - 缓存未命中: {cache_key}")
            
            # 2. 从数据库获取
            if date_str:
                # 尝试解析日期
                target_date = self._parse_datetime(date_str)
                
                # 查询数据库
                records = await asyncio.to_thread(
                    lambda: list(RetailNetInflowRateRanking.objects.filter(trade_time=target_date).order_by('-retail_net_inflow_rate'))
                )
            else:
                # 获取最新日期
                latest_date_record = await asyncio.to_thread(
                    lambda: RetailNetInflowRateRanking.objects.order_by('-trade_time').values('trade_time').first()
                )
                
                if not latest_date_record:
                    return []
                
                latest_date = latest_date_record['trade_time']
                
                # 查询数据库
                records = await asyncio.to_thread(
                    lambda: list(RetailNetInflowRateRanking.objects.filter(trade_time=latest_date).order_by('-retail_net_inflow_rate'))
                )
            
            if not records:
                return []
            
            # 将查询结果转换为字典列表
            result = []
            for record in records:
                item_dict = {}
                for field in get_model_fields(RetailNetInflowRateRanking):
                    value = getattr(record, field)
                    # 处理日期和时间
                    if field.endswith('_time') or field.endswith('_date') or field == 'trade_time':
                        item_dict[field] = self._parse_datetime(value)
                        if isinstance(item_dict[field], datetime):
                            item_dict[field] = item_dict[field].strftime('%Y-%m-%d') if field == 'trade_time' else item_dict[field].strftime('%Y-%m-%d %H:%M:%S')
                    # 处理数值
                    elif (isinstance(value, (int, float))) and 'code' not in field.name.lower():
                        item_dict[field] = self._parse_number(value)
                    else:
                        item_dict[field] = value
                result.append(item_dict)
            
            # 更新缓存
            await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"获取 散户净流入率排名数据 出错: {str(e)}")
            return []
    

