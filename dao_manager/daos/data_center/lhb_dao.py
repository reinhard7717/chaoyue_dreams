# dao/datacenter_dao.py
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Union, Any, Tuple
from django.db import transaction
from django.db.models import Q
from django.core.cache import cache
from django.db.models.fields import Field
from asgiref.sync import sync_to_async
import json

from api_manager.apis.datacenter_api import DataCenterAPI
from api_manager.mappings.datacenter_mappings import *
from dao_manager.base_dao import BaseDAO
from stock_models.datacenter.lhb import BrokerOnList, InstitutionTradeDetail, InstitutionTradeTrack, LhbDaily, StockOnList

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

class LhbDAO(BaseDAO):

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
 

    # 龙虎榜相关DAO方法
    
    async def save_daily_lhb(self) -> Dict:
        """
        保存今日龙虎榜概览
        
        Returns:
            dict: 操作结果统计
        """
        try:
            # 从API获取数据
            async with DataCenterAPI() as api:
                logger.info("开始获取龙虎榜数据")
                data = await api.get_daily_lhb()
            
            if not data:
                logger.warning("未获取到龙虎榜数据")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 验证并处理数据格式
            if isinstance(data, str):
                # 处理text/plain或text/html格式的响应
                try:
                    # 尝试解析为JSON
                    data = json.loads(data)
                except json.JSONDecodeError:
                    # 如果不是JSON格式，尝试解析为字典
                    try:
                        lines = data.split('\n')
                        parsed_data = {}
                        for line in lines:
                            if ':' in line:
                                key, value = line.split(':', 1)
                                parsed_data[key.strip()] = value.strip()
                        data = parsed_data
                    except Exception as e:
                        logger.error(f"解析龙虎榜文本数据失败: {str(e)}")
                        return {'created': 0, 'updated': 0, 'skipped': 0}
            
            if not isinstance(data, dict):
                logger.error(f"龙虎榜数据格式错误，期望dict类型，实际类型: {type(data)}")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 处理日期格式
            trade_date = None
            if 't' in data:
                trade_date = self._parse_datetime(data['t'])
                logger.debug(f"处理后的日期: {trade_date}")
            else:
                trade_date = self._parse_datetime(datetime.now())
                logger.warning("龙虎榜数据中没有日期字段，使用当前日期")
            
            # 验证必要字段
            if not trade_date:
                logger.error("无法获取有效的交易日期")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 转换为标准字典格式并应用字段映射
            mapped_data = {}
            for api_field, model_field in LHB_DAILY_MAPPING.items():
                if api_field in data:
                    value = data[api_field]
                    # 处理日期字段
                    if model_field.endswith('_time') or model_field.endswith('_date') or model_field == 'trade_date':
                        mapped_data[model_field] = self._parse_datetime(value)
                    # 处理数值字段
                    elif isinstance(value, (int, float)) or (
                        isinstance(value, str) and value.replace('.', '', 1).isdigit()
                    ):
                        mapped_data[model_field] = self._parse_number(value)
                    # 处理列表类型的字段（如decline_deviation_7pct等）
                    elif isinstance(value, list):
                        mapped_data[model_field] = json.dumps(value, ensure_ascii=False)
                    else:
                        mapped_data[model_field] = value
            
            # 确保trade_date字段存在
            mapped_data['trade_date'] = trade_date
            
            # 保存数据
            logger.info("开始保存龙虎榜数据")
            result = await self._batch_process(
                model_class=LhbDaily,
                data_list=[mapped_data],
                mapping=LHB_DAILY_MAPPING,
                unique_fields=['trade_date']
            )
            
            # 清除相关缓存
            logger.info("清除龙虎榜数据缓存")
            await self._set_to_cache('daily_lhb', mapped_data, self.CACHE_TIMEOUT['daily'])
            
            logger.info(f"龙虎榜数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存龙虎榜数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data if 'data' in locals() else '未获取到数据'}")
            return {'created': 0, 'updated': 0, 'skipped': 0}
    
    async def get_daily_lhb(self) -> Optional[Dict]:
        """
        获取今日龙虎榜概览
        
        Returns:
            Optional[Dict]: 龙虎榜数据
        """
        # 先查缓存
        cache_key = 'daily_lhb'
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 缓存未命中，查数据库
        try:
            # 获取最新一条记录
            latest = await asyncio.to_thread(
                lambda: LhbDaily.objects.order_by('-t').first()
            )
            
            if latest:
                # 将数据库查询结果转换为字典
                result = {field: getattr(latest, field) for field in get_model_fields(LhbDaily)}
                # 存入缓存
                await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['daily'])
                return result
            
            return None
        except Exception as e:
            logger.error(f"获取龙虎榜数据出错: {str(e)}")
            return None
    
    async def save_stock_on_list(self, days: int) -> Dict:
        """
        保存近n日上榜个股
        
        Args:
            days: 统计天数，可选 5、10、30、60
            
        Returns:
            dict: 操作结果统计
        """
        try:           
            # 从API获取数据
            async with DataCenterAPI() as api:
                logger.info(f"开始获取近{days}日上榜个股数据")
                data_list = await api.get_stock_on_list(days)
            
            if not data_list:
                logger.warning(f"未获取到近{days}日上榜个股数据")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 处理text/plain或text/html格式的响应
            if isinstance(data_list, str):
                try:
                    # 尝试解析为JSON
                    data_list = json.loads(data_list)
                except json.JSONDecodeError:
                    # 如果不是JSON格式，尝试按行解析
                    try:
                        lines = data_list.split('\n')
                        parsed_data = []
                        for line in lines:
                            if line.strip():  # 非空行
                                # 按特定格式解析，这里需要根据实际API响应格式调整
                                fields = line.split(',')
                                if len(fields) >= 3:  # 至少包含代码、名称、上榜次数
                                    item = {
                                        'dm': fields[0].strip(),
                                        'mc': fields[1].strip(),
                                        'lbcs': fields[2].strip(),
                                        # 添加其他字段...
                                    }
                                    parsed_data.append(item)
                        data_list = parsed_data
                    except Exception as e:
                        logger.error(f"解析上榜个股文本数据失败: {str(e)}")
                        return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 确保数据是列表格式
            if not isinstance(data_list, list):
                logger.error(f"上榜个股数据格式错误，期望list类型，实际类型: {type(data_list)}")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 处理数据
            processed_data = []
            for data in data_list:
                if not isinstance(data, dict):
                    logger.warning(f"跳过无效数据格式: {data}")
                    continue
                
                # 转换为标准字典格式并应用字段映射
                processed_item = {}
                for api_field, model_field in STOCK_ON_LIST_MAPPING.items():
                    if api_field in data:
                        # 处理日期字段
                        if model_field.endswith('_time') or model_field.endswith('_date') or model_field == 'trade_date':
                            processed_item[model_field] = self._parse_datetime(data[api_field])
                        # 处理数值字段
                        elif isinstance(data[api_field], (int, float)) or (
                            isinstance(data[api_field], str) and data[api_field].replace('.', '', 1).isdigit()
                        ):
                            processed_item[model_field] = self._parse_number(data[api_field])
                        else:
                            processed_item[model_field] = data[api_field]
                
                # 添加附加字段
                processed_item['stats_days'] = days
                processed_item['update_time'] = self._parse_datetime(datetime.now())
                
                # 验证必要字段
                if 'stock_code' not in processed_item or not processed_item['stock_code']:
                    logger.warning(f"跳过缺少股票代码的数据: {data}")
                    continue
                    
                processed_data.append(processed_item)
            
            if not processed_data:
                logger.warning(f"保存近{days}日上榜个股 - 没有有效的数据需要保存")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 保存数据
            result = await self._batch_process(
                model_class=StockOnList,
                data_list=processed_data,
                mapping=STOCK_ON_LIST_MAPPING,
                unique_fields=['stock_code', 'stats_days']
            )
            
            # 清除相关缓存并更新
            cache_key = f'stock_on_list_{days}'
            await self._set_to_cache(cache_key, processed_data, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"保存近{days}日上榜个股数据出错: {str(e)}")
            return {'created': 0, 'updated': 0, 'skipped': 0}
    
    async def get_stock_on_list(self, days: int) -> List[Dict]:
        """
        获取近n日上榜个股
        
        Args:
            days: 统计天数，可选 5、10、30、60
            
        Returns:
            List[Dict]: 个股上榜统计数据列表
        """
        # 先查缓存
        cache_key = f'stock_on_list_{days}'
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 缓存未命中，查数据库
        try:
            # 查询指定天数的数据
            records = await asyncio.to_thread(
                lambda: StockOnList.objects.filter(days=days).order_by('-lbcs')
            )
            
            if records:
                # 将结果转换为列表并缓存
                result = [{field: getattr(record, field) for field in get_model_fields(StockOnList)} 
                         for record in records]
                await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['medium'])
                return result
            
            return []
        except Exception as e:
            logger.error(f"获取近{days}日上榜个股数据出错: {str(e)}")
            return []
    
    async def save_broker_on_list(self, days: int) -> Dict:
        """
        保存近n日营业部上榜统计
        
        Args:
            days: 统计天数，可选 5、10、30、60
            
        Returns:
            dict: 操作结果统计
        """
        try:                
            # 从API获取数据
            async with DataCenterAPI() as api:
                logger.info(f"开始获取近{days}日营业部上榜统计数据")
                data_list = await api.get_broker_on_list(days)
            
            if not data_list:
                logger.warning(f"未获取到近{days}日营业部上榜统计数据")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 处理text/plain或text/html格式的响应
            if isinstance(data_list, str):
                try:
                    # 尝试解析为JSON
                    data_list = json.loads(data_list)
                except json.JSONDecodeError:
                    # 如果不是JSON格式，尝试按行解析
                    try:
                        lines = data_list.split('\n')
                        parsed_data = []
                        for line in lines:
                            if line.strip():  # 非空行
                                # 按特定格式解析，这里需要根据实际API响应格式调整
                                fields = line.split(',')
                                if len(fields) >= 3:  # 至少包含营业部名称、上榜次数、成交金额
                                    item = {
                                        'yybmc': fields[0].strip(),
                                        'lbcs': fields[1].strip(),
                                        'cjje': fields[2].strip(),
                                        # 添加其他字段...
                                    }
                                    parsed_data.append(item)
                        data_list = parsed_data
                    except Exception as e:
                        logger.error(f"解析营业部上榜文本数据失败: {str(e)}")
                        return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 确保数据是列表格式
            if not isinstance(data_list, list):
                logger.error(f"营业部上榜数据格式错误，期望list类型，实际类型: {type(data_list)}")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 处理数据
            processed_data = []
            for data in data_list:
                if not isinstance(data, dict):
                    logger.warning(f"跳过无效数据格式: {data}")
                    continue
                    
                # 转换为标准字典格式并应用字段映射
                processed_item = {}
                for api_field, model_field in BROKER_ON_LIST_MAPPING.items():
                    if api_field in data:
                        # 处理日期字段
                        if model_field.endswith('_time') or model_field.endswith('_date') or model_field == 'trade_date':
                            processed_item[model_field] = self._parse_datetime(data[api_field])
                        # 处理数值字段
                        elif isinstance(data[api_field], (int, float)) or (
                            isinstance(data[api_field], str) and data[api_field].replace('.', '', 1).isdigit()
                        ):
                            processed_item[model_field] = self._parse_number(data[api_field])
                        else:
                            processed_item[model_field] = data[api_field]
                
                # 添加附加字段
                processed_item['stats_days'] = days
                processed_item['update_time'] = self._parse_datetime(datetime.now())
                
                # 验证必要字段
                if 'broker_name' not in processed_item or not processed_item['broker_name']:
                    logger.warning(f"跳过缺少营业部名称的数据: {data}")
                    continue
                    
                processed_data.append(processed_item)
            
            if not processed_data:
                logger.warning(f"保存近{days}日营业部上榜统计 - 没有有效的数据需要保存")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 保存数据
            result = await self._batch_process(
                model_class=BrokerOnList,
                data_list=processed_data,
                mapping=BROKER_ON_LIST_MAPPING,
                unique_fields=['broker_name', 'stats_days']
            )
            
            # 清除相关缓存并更新
            cache_key = f'broker_on_list_{days}'
            await self._set_to_cache(cache_key, processed_data, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"保存近{days}日营业部上榜统计数据出错: {str(e)}")
            return {'created': 0, 'updated': 0, 'skipped': 0}
    

    async def get_broker_on_list(self, days: int) -> List[Dict]:
        """
        获取近n日营业部上榜统计
        
        Args:
            days: 统计天数，可选 5、10、30、60
            
        Returns:
            List[Dict]: 营业部上榜统计数据列表
        """
        # 先查缓存
        cache_key = f'broker_on_list_{days}'
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 缓存未命中，查数据库
        try:
            records = await asyncio.to_thread(
                lambda: BrokerOnList.objects.filter(days=days).order_by('-lbcs')
            )
            
            if records:
                result = [{field: getattr(record, field) for field in get_model_fields(BrokerOnList)} 
                         for record in records]
                await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['medium'])
                return result
            
            return []
        except Exception as e:
            logger.error(f"获取近{days}日营业部上榜统计数据出错: {str(e)}")
            return []
    
    async def save_institution_trade_track(self, days: int) -> Dict:
        """
        保存近n日个股机构交易追踪
        
        Args:
            days: 统计天数，可选 5、10、30、60
            
        Returns:
            dict: 操作结果统计
        """
        try:                
            # 从API获取数据
            async with DataCenterAPI() as api:
                logger.info(f"开始获取近{days}日个股机构交易追踪数据")
                data_list = await api.get_institution_trade_track(days)
            
            if not data_list:
                logger.warning(f"未获取到近{days}日个股机构交易追踪数据")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 处理text/plain或text/html格式的响应
            if isinstance(data_list, str):
                try:
                    # 尝试解析为JSON
                    data_list = json.loads(data_list)
                except json.JSONDecodeError:
                    # 如果不是JSON格式，尝试按行解析
                    try:
                        lines = data_list.split('\n')
                        parsed_data = []
                        for line in lines:
                            if line.strip():  # 非空行
                                # 按特定格式解析，这里需要根据实际API响应格式调整
                                fields = line.split(',')
                                if len(fields) >= 4:  # 至少包含代码、名称、净额、涨跌幅
                                    item = {
                                        'dm': fields[0].strip(),
                                        'mc': fields[1].strip(),
                                        'jglczje': fields[2].strip(),
                                        'zdf': fields[3].strip(),
                                        # 添加其他字段...
                                    }
                                    parsed_data.append(item)
                        data_list = parsed_data
                    except Exception as e:
                        logger.error(f"解析机构交易追踪文本数据失败: {str(e)}")
                        return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 确保数据是列表格式
            if not isinstance(data_list, list):
                logger.error(f"机构交易追踪数据格式错误，期望list类型，实际类型: {type(data_list)}")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 处理数据
            processed_data = []
            for data in data_list:
                if not isinstance(data, dict):
                    logger.warning(f"跳过无效数据格式: {data}")
                    continue
                    
                # 转换为标准字典格式并应用字段映射
                processed_item = {}
                for api_field, model_field in INSTITUTION_TRADE_TRACK_MAPPING.items():
                    if api_field in data:
                        # 处理日期字段
                        if model_field.endswith('_time') or model_field.endswith('_date') or model_field == 'trade_date':
                            processed_item[model_field] = self._parse_datetime(data[api_field])
                        # 处理数值字段
                        elif isinstance(data[api_field], (int, float)) or (
                            isinstance(data[api_field], str) and data[api_field].replace('.', '', 1).isdigit()
                        ):
                            processed_item[model_field] = self._parse_number(data[api_field])
                        else:
                            processed_item[model_field] = data[api_field]
                
                # 添加附加字段
                processed_item['stats_days'] = days
                processed_item['update_time'] = self._parse_datetime(datetime.now())
                
                # 验证必要字段
                if 'stock_code' not in processed_item or not processed_item['stock_code']:
                    logger.warning(f"跳过缺少股票代码的数据: {data}")
                    continue
                    
                processed_data.append(processed_item)
            
            if not processed_data:
                logger.warning(f"保存近{days}日个股机构交易追踪 - 没有有效的数据需要保存")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 保存数据
            result = await self._batch_process(
                model_class=InstitutionTradeTrack,
                data_list=processed_data,
                mapping=INSTITUTION_TRADE_TRACK_MAPPING,
                unique_fields=['stock_code', 'stats_days']
            )
            
            # 清除相关缓存并更新
            cache_key = f'institution_trade_track_{days}'
            await self._set_to_cache(cache_key, processed_data, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"保存近{days}日个股机构交易追踪数据出错: {str(e)}")
            return {'created': 0, 'updated': 0, 'skipped': 0}
    

    async def get_institution_trade_track(self, days: int) -> List[Dict]:
        """
        获取近n日个股机构交易追踪
        
        Args:
            days: 统计天数，可选 5、10、30、60
            
        Returns:
            List[Dict]: 机构席位追踪数据列表
        """
        # 先查缓存
        cache_key = f'institution_trade_track_{days}'
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 缓存未命中，查数据库
        try:
            records = await asyncio.to_thread(
                lambda: InstitutionTradeTrack.objects.filter(days=days).order_by('-jglczje')
            )
            
            if records:
                result = [{field: getattr(record, field) for field in get_model_fields(InstitutionTradeTrack)} 
                         for record in records]
                await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['medium'])
                return result
            
            return []
        except Exception as e:
            logger.error(f"获取近{days}日个股机构交易追踪数据出错: {str(e)}")
            return []
    
    async def save_institution_trade_detail(self, days: int) -> Dict:
        """
        保存机构席位成交明细
        
        Args:
            days: 统计天数，可选 5、10、30、60
                    
        Returns:
            dict: 操作结果统计
        """
        try:                
            # 从API获取数据
            async with DataCenterAPI() as api:
                logger.info(f"开始获取近{days}日机构席位成交明细数据")
                data_list = await api.get_institution_trade_detail(days)
            
            if not data_list:
                logger.warning(f"未获取到近{days}日个机构席位成交明细数据")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 处理text/plain或text/html格式的响应
            if isinstance(data_list, str):
                try:
                    # 尝试解析为JSON
                    data_list = json.loads(data_list)
                except json.JSONDecodeError:
                    # 如果不是JSON格式，尝试按行解析
                    try:
                        lines = data_list.split('\n')
                        parsed_data = []
                        for line in lines:
                            if line.strip():  # 非空行
                                # 按特定格式解析，这里需要根据实际API响应格式调整
                                fields = line.split(',')
                                if len(fields) >= 5:  # 至少包含代码、名称、日期、类型、成交额
                                    item = {
                                        'dm': fields[0].strip(),
                                        'mc': fields[1].strip(),
                                        't': fields[2].strip(),
                                        'type': fields[3].strip(),
                                        'cjje': fields[4].strip(),
                                        # 添加其他字段...
                                    }
                                    parsed_data.append(item)
                        data_list = parsed_data
                    except Exception as e:
                        logger.error(f"解析机构席位成交明细文本数据失败: {str(e)}")
                        return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 确保数据是列表格式
            if not isinstance(data_list, list):
                logger.error(f"机构席位成交明细数据格式错误，期望list类型，实际类型: {type(data_list)}")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 处理数据
            processed_data = []
            for data in data_list:
                if not isinstance(data, dict):
                    logger.warning(f"跳过无效数据格式: {data}")
                    continue
                    
                # 转换为标准字典格式并应用字段映射
                processed_item = {}
                for api_field, model_field in INSTITUTION_TRADE_DETAIL_MAPPING.items():
                    if api_field in data:
                        # 处理日期字段
                        if model_field.endswith('_time') or model_field.endswith('_date') or model_field == 'trade_date':
                            processed_item[model_field] = self._parse_datetime(data[api_field])
                        # 处理数值字段
                        elif isinstance(data[api_field], (int, float)) or (
                            isinstance(data[api_field], str) and data[api_field].replace('.', '', 1).isdigit()
                        ):
                            processed_item[model_field] = self._parse_number(data[api_field])
                        else:
                            processed_item[model_field] = data[api_field]
                
                # 添加额外信息
                processed_item['update_time'] = self._parse_datetime(datetime.now())
                
                # 验证必要字段
                if 'stock_code' not in processed_item or not processed_item['stock_code'] or 'trade_type' not in processed_item or not processed_item['trade_type']:
                    logger.warning(f"跳过缺少必要字段的数据: {data}")
                    continue
                    
                processed_data.append(processed_item)
            
            if not processed_data:
                logger.warning("保存机构席位成交明细 - 没有有效的数据需要保存")
                return {'created': 0, 'updated': 0, 'skipped': 0}
            
            # 保存数据
            result = await self._batch_process(
                model_class=InstitutionTradeDetail,
                data_list=processed_data,
                mapping=INSTITUTION_TRADE_DETAIL_MAPPING,
                unique_fields=['stock_code', 'trade_date', 'trade_type']
            )
            
            # 如果有数据，缓存最近的交易记录
            if processed_data:
                # 按股票代码分组缓存
                stock_codes = set(item.get('stock_code') for item in processed_data if 'stock_code' in item)
                for stock_code in stock_codes:
                    stock_records = [item for item in processed_data if item.get('stock_code') == stock_code]
                    if stock_records:
                        cache_key = f'institution_trade_detail_{stock_code}'
                        await self._set_to_cache(cache_key, stock_records, self.CACHE_TIMEOUT['medium'])
            
            return result
        except Exception as e:
            logger.error(f"保存机构席位成交明细数据出错: {str(e)}")
            return {'created': 0, 'updated': 0, 'skipped': 0}
    

    async def get_institution_trade_detail(self, stock_code: str) -> List[Dict]:
        """
        获取指定股票的机构席位成交明细
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[Dict]: 机构席位成交明细数据列表
        """
        # 先查缓存
        cache_key = f'institution_trade_detail_{stock_code}'
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 缓存未命中，查数据库
        try:
            records = await asyncio.to_thread(
                lambda: InstitutionTradeDetail.objects.filter(dm=stock_code).order_by('-t')
            )
            
            if records:
                result = [{field: getattr(record, field) for field in get_model_fields(InstitutionTradeDetail)} 
                         for record in records]
                await self._set_to_cache(cache_key, result, self.CACHE_TIMEOUT['medium'])
                return result
            
            return []
        except Exception as e:
            logger.error(f"获取股票{stock_code}的机构席位成交明细数据出错: {str(e)}")
            return []
    