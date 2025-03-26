import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Any
from django.db import transaction
from django.db.models import Q
from django.core.cache import cache
from django.db.models.fields import Field
from asgiref.sync import sync_to_async
import json

from api_manager.apis.datacenter_api import DataCenterAPI
from api_manager.mappings.datacenter_mappings import *
from dao_manager.base_dao import BaseDAO
from stock_models.datacenter.institution import FundHeavyPosition, InstitutionHoldingSummary, QFIIHeavyPosition, SocialSecurityHeavyPosition

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

class InstitutionalShareholdingDao(BaseDAO):
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


    async def save_institution_holding_summary(self) -> Dict:
        """
        保存机构持股汇总数据
        
        Returns:
            dict: 操作结果统计
        """
        try:
            # 从API获取数据
            api = DataCenterAPI()
            response_data = await api.get_institution_holding_summary(year=datetime.now().year, quarter=1)
            
            # 处理API返回的响应
            data_list = []
            if not response_data:
                logger.warning("未获取到机构持股汇总数据")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 处理不同格式的响应
            if isinstance(response_data, str):
                try:
                    # 尝试解析JSON字符串
                    response_data = json.loads(response_data)
                except json.JSONDecodeError:
                    # 如果不是JSON格式，尝试解析为列表
                    try:
                        # 简单解析text/plain或text/html格式
                        lines = response_data.strip().split('\n')
                        header = lines[0].split(',')
                        response_data = []
                        for line in lines[1:]:
                            values = line.split(',')
                            if len(values) == len(header):
                                response_data.append(dict(zip(header, values)))
                    except Exception as e:
                        logger.error(f"解析机构持股汇总数据失败: {str(e)}")
                        return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 确保response_data是列表类型
            if isinstance(response_data, dict):
                response_data = [response_data]
            
            # 处理数据列表
            for item in response_data:
                # 使用BaseDAO的_parse_datetime和_parse_number方法处理数据
                processed_data = {}
                # 映射字段并处理数据格式
                for api_field, model_field in INSTITUTION_HOLDING_SUMMARY_MAPPING.items():
                    if api_field in item:
                        value = item[api_field]
                        if model_field.endswith('_time') or model_field.endswith('_date') or model_field == 't':
                            processed_data[api_field] = self._parse_datetime(value)
                        elif api_field in ['jgcgs', 'jjcgs', 'sbcgs', 'qfiicgs', 'baoxcgs', 
                                          'jgcgzb', 'jjcgzb', 'sbcgzb', 'qfiicgzb', 'baoxcgzb',
                                          'jgcgsz', 'jjcgsz', 'sbcgsz', 'qfiicgsz', 'baoxcgsz',
                                          'year', 'quarter']:
                            processed_data[api_field] = self._parse_number(value)
                        else:
                            processed_data[api_field] = value
                
                data_list.append(processed_data)
            
            # 保存数据到数据库
            result = await self._batch_process(
                model_class=InstitutionHoldingSummary,
                data_list=data_list,
                mapping=INSTITUTION_HOLDING_SUMMARY_MAPPING,
                unique_fields=['stock_code', 'trade_date']
            )
            
            # 转换数据为标准字典格式，用于缓存
            if data_list:
                year = data_list[0].get("year", datetime.now().year)
                quarter = data_list[0].get("quarter", 1)
                
                # 准备缓存数据
                cache_data = []
                for item in data_list:
                    cache_item = {}
                    for api_field, model_field in INSTITUTION_HOLDING_SUMMARY_MAPPING.items():
                        if api_field in item:
                            cache_item[model_field] = item[api_field]
                    cache_data.append(cache_item)
                
                # 更新缓存，使用异步方式
                cache_key = f'institution_holding_summary_{year}_{quarter}'
                await self._set_to_cache(cache_key, cache_data, self.CACHE_TIMEOUT['long'])
            
            return result
        except Exception as e:
            logger.error(f"保存机构持股汇总数据出错: {str(e)}")
            return {'created': 0, 'updated': 0, 'skipped': 0}

    async def get_institution_holding_summary(self, year: int, quarter: int) -> List[Dict]:
        """
        获取机构持股汇总数据
        
        Args:
            year: 报告年份
            quarter: 报告季度，1:一季报，2：中报，3：三季报，4：年报
            
        Returns:
            List[Dict]: 机构持股汇总数据列表
        """
        # 先查缓存
        cache_key = f'institution_holding_summary_{year}_{quarter}'
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 缓存未命中，查数据库
        try:
            records = await asyncio.to_thread(
                lambda: InstitutionHoldingSummary.objects.filter(
                    year=year, quarter=quarter
                ).order_by('-jglczb')
            )
            
            if records:
                result = [{field: getattr(record, field) for field in get_model_fields(InstitutionHoldingSummary)} 
                         for record in records]
                await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['long'])
                return result
            
            return []
        except Exception as e:
            logger.error(f"获取{year}年第{quarter}季度机构持股汇总数据出错: {str(e)}")
            return []

    async def save_fund_heavy_positions(self) -> Dict:
        """
        保存基金重仓数据
        
        Returns:
            dict: 操作结果统计
        """
        try:
            # 从API获取数据
            api = DataCenterAPI()
            response_data = await api.get_fund_heavy_positions(year=datetime.now().year, quarter=1)
            
            # 处理API返回的响应
            data_list = []
            if not response_data:
                logger.warning("未获取到基金重仓数据")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 处理不同格式的响应
            if isinstance(response_data, str):
                try:
                    # 尝试解析JSON字符串
                    response_data = json.loads(response_data)
                except json.JSONDecodeError:
                    # 如果不是JSON格式，尝试解析为列表
                    try:
                        # 简单解析text/plain或text/html格式
                        lines = response_data.strip().split('\n')
                        header = lines[0].split(',')
                        response_data = []
                        for line in lines[1:]:
                            values = line.split(',')
                            if len(values) == len(header):
                                response_data.append(dict(zip(header, values)))
                    except Exception as e:
                        logger.error(f"解析基金重仓数据失败: {str(e)}")
                        return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 确保response_data是列表类型
            if isinstance(response_data, dict):
                response_data = [response_data]
            
            # 处理数据列表
            for item in response_data:
                # 使用BaseDAO的_parse_datetime和_parse_number方法处理数据
                processed_data = {}
                # 映射字段并处理数据格式
                for api_field, model_field in FUND_HEAVY_POSITION_MAPPING.items():
                    if api_field in item:
                        value = item[api_field]
                        if model_field.endswith('_time') or model_field.endswith('_date') or model_field == 't':
                            processed_data[api_field] = self._parse_datetime(value)
                        elif api_field in ['jjsl', 'cgs', 'cgsz', 'jzc', 'jlr', 'cgbl', 'cgszbl', 'zltgbl', 'zdf', 'c', 'year', 'quarter']:
                            processed_data[api_field] = self._parse_number(value)
                        else:
                            processed_data[api_field] = value
                
                data_list.append(processed_data)
            
            # 保存数据到数据库
            result = await self._batch_process(
                model_class=FundHeavyPosition,
                data_list=data_list,
                mapping=FUND_HEAVY_POSITION_MAPPING,
                unique_fields=['stock_code', 'trade_date', 'year', 'quarter']
            )
            
            # 转换数据为标准字典格式，用于缓存
            if data_list:
                year = data_list[0].get("year", datetime.now().year)
                quarter = data_list[0].get("quarter", 1)
                
                # 准备缓存数据
                cache_data = []
                for item in data_list:
                    cache_item = {}
                    for api_field, model_field in FUND_HEAVY_POSITION_MAPPING.items():
                        if api_field in item:
                            cache_item[model_field] = item[api_field]
                    cache_data.append(cache_item)
                
                # 更新缓存，使用异步方式
                cache_key = f'fund_heavy_positions_{year}_{quarter}'
                await self._set_to_cache(cache_key, cache_data, self.CACHE_TIMEOUT['long'])
            
            return result
        except Exception as e:
            logger.error(f"保存基金重仓数据出错: {str(e)}")
            return {'created': 0, 'updated': 0, 'skipped': 0}

    async def get_fund_heavy_positions(self, year: int, quarter: int) -> List[Dict]:
        """
        获取基金重仓数据
        
        Args:
            year: 报告年份
            quarter: 报告季度，1:一季报，2：中报，3：三季报，4：年报
            
        Returns:
            List[Dict]: 基金重仓数据列表
        """
        # 先查缓存
        cache_key = f'fund_heavy_positions_{year}_{quarter}'
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 缓存未命中，查数据库
        try:
            records = await asyncio.to_thread(
                lambda: FundHeavyPosition.objects.filter(
                    year=year, quarter=quarter
                ).order_by('-jjsl')
            )
            
            if records:
                result = [{field: getattr(record, field) for field in get_model_fields(FundHeavyPosition)} 
                         for record in records]
                await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['long'])
                return result
            
            return []
        except Exception as e:
            logger.error(f"获取{year}年第{quarter}季度基金重仓数据出错: {str(e)}")
            return []

    async def save_social_security_heavy_positions(self) -> Dict:
        """
        保存社保重仓数据
        
        Returns:
            dict: 操作结果统计
        """
        try:
            # 从API获取数据
            api = DataCenterAPI()
            response_data = await api.get_social_security_heavy_positions(year=datetime.now().year, quarter=1)
            
            # 处理API返回的响应
            data_list = []
            if not response_data:
                logger.warning("未获取到社保重仓数据")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 处理不同格式的响应
            if isinstance(response_data, str):
                try:
                    # 尝试解析JSON字符串
                    response_data = json.loads(response_data)
                except json.JSONDecodeError:
                    # 如果不是JSON格式，尝试解析为列表
                    try:
                        # 简单解析text/plain或text/html格式
                        lines = response_data.strip().split('\n')
                        header = lines[0].split(',')
                        response_data = []
                        for line in lines[1:]:
                            values = line.split(',')
                            if len(values) == len(header):
                                response_data.append(dict(zip(header, values)))
                    except Exception as e:
                        logger.error(f"解析社保重仓数据失败: {str(e)}")
                        return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 确保response_data是列表类型
            if isinstance(response_data, dict):
                response_data = [response_data]
            
            # 处理数据列表
            for item in response_data:
                # 使用BaseDAO的_parse_datetime和_parse_number方法处理数据
                processed_data = {}
                # 映射字段并处理数据格式
                for api_field, model_field in SOCIAL_SECURITY_HEAVY_POSITION_MAPPING.items():
                    if api_field in item:
                        value = item[api_field]
                        if model_field.endswith('_time') or model_field.endswith('_date') or model_field == 't':
                            processed_data[api_field] = self._parse_datetime(value)
                        elif api_field in ['sbsl', 'cgs', 'cgsz', 'jzc', 'jlr', 'cgbl', 'cgszbl', 'zltgbl', 'zdf', 'c', 'year', 'quarter']:
                            processed_data[api_field] = self._parse_number(value)
                        else:
                            processed_data[api_field] = value
                
                data_list.append(processed_data)
            
            # 保存数据到数据库
            result = await self._batch_process(
                model_class=SocialSecurityHeavyPosition,
                data_list=data_list,
                mapping=SOCIAL_SECURITY_HEAVY_POSITION_MAPPING,
                unique_fields=['stock_code', 'trade_date', 'year', 'quarter']
            )
            
            # 转换数据为标准字典格式，用于缓存
            if data_list:
                year = data_list[0].get("year", datetime.now().year)
                quarter = data_list[0].get("quarter", 1)
                
                # 准备缓存数据
                cache_data = []
                for item in data_list:
                    cache_item = {}
                    for api_field, model_field in SOCIAL_SECURITY_HEAVY_POSITION_MAPPING.items():
                        if api_field in item:
                            cache_item[model_field] = item[api_field]
                    cache_data.append(cache_item)
                
                # 更新缓存，使用异步方式
                cache_key = f'social_security_heavy_positions_{year}_{quarter}'
                await self._set_to_cache(cache_key, cache_data, self.CACHE_TIMEOUT['long'])
            
            return result
        except Exception as e:
            logger.error(f"保存社保重仓数据出错: {str(e)}")
            return {'created': 0, 'updated': 0, 'skipped': 0}

    async def get_social_security_heavy_positions(self, year: int, quarter: int) -> List[Dict]:
        """
        获取社保重仓数据
        
        Args:
            year: 报告年份
            quarter: 报告季度，1:一季报，2：中报，3：三季报，4：年报
            
        Returns:
            List[Dict]: 社保重仓数据列表
        """
        # 先查缓存
        cache_key = f'social_security_heavy_positions_{year}_{quarter}'
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 缓存未命中，查数据库
        try:
            records = await asyncio.to_thread(
                lambda: SocialSecurityHeavyPosition.objects.filter(
                    year=year, quarter=quarter
                ).order_by('-sbsl')
            )
            
            if records:
                result = [{field: getattr(record, field) for field in get_model_fields(SocialSecurityHeavyPosition)} 
                         for record in records]
                await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['long'])
                return result
            
            return []
        except Exception as e:
            logger.error(f"获取{year}年第{quarter}季度社保重仓数据出错: {str(e)}")
            return []

    async def save_qfii_heavy_positions(self) -> Dict:
        """
        保存QFII重仓股数据
        
        Returns:
            dict: 操作结果统计
        """
        try:
            # 从API获取数据
            api = DataCenterAPI()
            response_data = await api.get_qfii_heavy_positions(year=datetime.now().year, quarter=1)
            
            # 处理API返回的响应
            data_list = []
            if not response_data:
                logger.warning("未获取到QFII重仓股数据")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 处理不同格式的响应
            if isinstance(response_data, str):
                try:
                    # 尝试解析JSON字符串
                    response_data = json.loads(response_data)
                except json.JSONDecodeError:
                    # 如果不是JSON格式，尝试解析为列表
                    try:
                        # 简单解析text/plain或text/html格式
                        lines = response_data.strip().split('\n')
                        header = lines[0].split(',')
                        response_data = []
                        for line in lines[1:]:
                            values = line.split(',')
                            if len(values) == len(header):
                                response_data.append(dict(zip(header, values)))
                    except Exception as e:
                        logger.error(f"解析QFII重仓股数据失败: {str(e)}")
                        return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 确保response_data是列表类型
            if isinstance(response_data, dict):
                response_data = [response_data]
            
            # 处理数据列表
            for item in response_data:
                # 使用BaseDAO的_parse_datetime和_parse_number方法处理数据
                processed_data = {}
                # 映射字段并处理数据格式
                for api_field, model_field in QFII_HEAVY_POSITION_MAPPING.items():
                    if api_field in item:
                        value = item[api_field]
                        if model_field.endswith('_time') or model_field.endswith('_date') or model_field == 't':
                            processed_data[api_field] = self._parse_datetime(value)
                        elif api_field in ['qfiis', 'cgs', 'cgsz', 'jzc', 'jlr', 'cgbl', 'cgszbl', 'zltgbl', 'zdf', 'c', 'year', 'quarter']:
                            processed_data[api_field] = self._parse_number(value)
                        else:
                            processed_data[api_field] = value
                
                data_list.append(processed_data)
            
            # 保存数据到数据库
            result = await self._batch_process(
                model_class=QFIIHeavyPosition,
                data_list=data_list,
                mapping=QFII_HEAVY_POSITION_MAPPING,
                unique_fields=['stock_code', 'trade_date', 'year', 'quarter']
            )
            
            # 转换数据为标准字典格式，用于缓存
            if data_list:
                year = data_list[0].get("year", datetime.now().year)
                quarter = data_list[0].get("quarter", 1)
                
                # 准备缓存数据
                cache_data = []
                for item in data_list:
                    cache_item = {}
                    for api_field, model_field in QFII_HEAVY_POSITION_MAPPING.items():
                        if api_field in item:
                            cache_item[model_field] = item[api_field]
                    cache_data.append(cache_item)
                
                # 更新缓存，使用异步方式
                cache_key = f'qfii_heavy_positions_{year}_{quarter}'
                await self._set_to_cache(cache_key, cache_data, self.CACHE_TIMEOUT['long'])
            
            return result
        except Exception as e:
            logger.error(f"保存QFII重仓股数据出错: {str(e)}")
            return {'created': 0, 'updated': 0, 'skipped': 0}

    async def get_qfii_heavy_positions(self, year: int, quarter: int) -> List[Dict]:
        """
        获取QFII重仓股数据
        
        Args:
            year: 报告年份
            quarter: 报告季度，1:一季报，2：中报，3：三季报，4：年报
            
        Returns:
            List[Dict]: QFII重仓股数据列表
        """
        # 先查缓存
        cache_key = f'qfii_heavy_positions_{year}_{quarter}'
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 缓存未命中，查数据库
        try:
            records = await asyncio.to_thread(
                lambda: QFIIHeavyPosition.objects.filter(
                    year=year, quarter=quarter
                ).order_by('-qfii_count')
            )
            
            if records:
                result = [{field: getattr(record, field) for field in get_model_fields(QFIIHeavyPosition)} 
                         for record in records]
                await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['long'])
                return result
            
            return []
        except Exception as e:
            logger.error(f"获取{year}年第{quarter}季度QFII重仓股数据出错: {str(e)}")
            return []
