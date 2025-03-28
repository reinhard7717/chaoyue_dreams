# dao/stock_index_dao.py

import json
import logging
import datetime
import asyncio
from typing import Dict, Any, List, Optional, Union, Type
from django.db import transaction
from django.core.cache import cache
from django.utils import timezone
from asgiref.sync import sync_to_async
from django.conf import settings
from utils.cache_manager import CacheManager
from utils.models import ModelJSONEncoder

from dao_manager.base_dao import BaseDAO
from api_manager.apis.index_api import StockIndexAPI
from api_manager.mappings.index_mapping import BOLL_MAPPING, INDEX_REALTIME_DATA_MAPPING, KDJ_MAPPING, MA_MAPPING, MACD_MAPPING, MARKET_OVERVIEW_MAPPING, TIME_SERIES_MAPPING
from stock_models.index import *


logger = logging.getLogger("dao")

class StockIndexDAO(BaseDAO):
    """
    股票指数数据访问对象
    
    负责股票指数相关数据的读写操作，实现三层数据访问结构：API、Redis缓存、MySQL持久化
    """
    
    def __init__(self):
        """初始化DAO对象，创建API实例"""
        self.api = StockIndexAPI()
        self.cache_manager = CacheManager()
    
    # ================ 读取方法 ================
    
    async def get_all_indexes(self) -> List[IndexInfo]:
        """
        获取所有股票指数列表
        
        先尝试从缓存获取，如缓存未命中则从数据库读取，
        如数据库无数据则从API获取并保存
        
        Returns:
            List[StockIndex]: 指数对象列表
        """
        # 使用CacheManager生成标准化缓存键
        cache_key = self.cache_manager.generate_key('st', 'index', 'all')
        
        # 尝试从缓存获取
        cached_data = cache.get(cache_key)
        if cached_data:
            logger.debug("从缓存获取股票指数列表")
            # 将缓存数据转换为模型实例列表
            return [IndexInfo(**index_dict) for index_dict in cached_data]
        
        # 从数据库读取
        indexes = await sync_to_async(list)(IndexInfo.objects.all())
        
        # 如果数据库中有数据，缓存并返回
        if indexes:
            # 序列化对象列表
            serialized_indexes = [self._serialize_model(index) for index in indexes]
            logger.debug(f"从数据库获取股票指数列表，共{len(indexes)}条")
            
            # 使用CacheManager缓存数据
            self.cache_manager.set(
                cache_key, 
                serialized_indexes, 
                timeout=self.cache_manager.get_timeout('st')
            )
            return indexes
        
        # 如果数据库中没有数据，从API获取并保存
        logger.info("数据库中没有指数数据，从API获取")
        await self._fetch_and_save_indexes()
        indexes = await sync_to_async(list)(IndexInfo.objects.all())
        # 序列化并缓存
        if indexes:
            serialized_indexes = [self._serialize_model(index) for index in indexes]
            self.cache_manager.set(
                cache_key, 
                serialized_indexes,
                timeout=self.cache_manager.get_timeout('st')
            )
        return indexes
    
    async def get_index_by_code(self, code: str) -> Optional[IndexInfo]:
        """
        根据指数代码获取指数
        
        Args:
            code: 指数代码
            
        Returns:
            Optional[StockIndex]: 指数对象，如不存在返回None
        """
        # 使用CacheManager生成标准化缓存键
        cache_key = self.cache_manager.generate_key('st', 'index', code)
        
        # 尝试从缓存获取，指定模型类进行自动转换
        index = self.cache_manager.get_model(cache_key, IndexInfo)
        if index:
            return index
            
        # 从数据库获取
        try:
            index = await sync_to_async(IndexInfo.objects.get)(code=code)
            
            # 序列化并缓存
            self.cache_manager.set(
                cache_key, 
                self._serialize_model(index),
                timeout=self.cache_manager.get_timeout('st')
            )
            return index
        except IndexInfo.DoesNotExist:
            logger.warning(f"指数代码[{code}]不存在")
            return None
            
    async def get_latest_realtime_data(self, index_code: str) -> Optional[IndexRealTimeData]:
        """
        获取指数最新实时数据
        
        Args:
            index_code: 指数代码
            
        Returns:
            Optional[StockIndexRealTimeData]: 实时数据对象，如不存在返回None
        """
        # 使用CacheManager生成标准化缓存键
        cache_key = self.cache_manager.generate_key('rt', 'index', index_code, 'realtime')
        
        # 尝试从缓存获取
        realtime_data = self.cache_manager.get_model(cache_key, IndexRealTimeData)
        if realtime_data:
            return realtime_data
        
        # 从数据库获取最新数据
        index = await self.get_index_by_code(index_code)
        if not index:
            return None
        
        try:
            data = await sync_to_async(
                lambda: IndexRealTimeData.objects.filter(index=index).order_by('-update_time').first()
            )()
            
            # 检查数据是否过期（超过2分钟）
            if data and (timezone.now() - data.update_time).total_seconds() < 120:
                # 序列化并缓存
                self.cache_manager.set(
                    cache_key, 
                    self._serialize_model(data),
                    timeout=self.cache_manager.get_timeout('rt')
                )
                return data
            
            # 数据不存在或已过期，从API获取新数据
            logger.info(f"指数[{index_code}]实时数据不存在或已过期，从API获取")
            data = await self._fetch_and_save_realtime_data(index_code)
            if data:
                # 序列化并缓存
                self.cache_manager.set(
                    cache_key, 
                    self._serialize_model(data),
                    timeout=self.cache_manager.get_timeout('rt')
                )
            return data
        except Exception as e:
            logger.error(f"获取指数[{index_code}]实时数据失败: {str(e)}")
            return None
            
    async def get_latest_market_overview(self) -> Optional[MarketOverview]:
        """
        获取最新市场概览数据
        
        Returns:
            Optional[MarketOverview]: 市场概览数据对象，如不存在返回None
        """
        # 使用CacheManager生成标准化缓存键
        cache_key = self.cache_manager.generate_key('rt', 'market', 'overview')
        
        # 尝试从缓存获取
        market_overview = self.cache_manager.get_model(cache_key, MarketOverview)
        if market_overview:
            return market_overview
        
        # 从数据库获取最新数据
        try:
            data = await sync_to_async(
                lambda: MarketOverview.objects.order_by('-update_time').first()
            )()
            
            # 检查数据是否过期（超过2分钟）
            if data and (timezone.now() - data.update_time).total_seconds() < 120:
                # 序列化并缓存
                self.cache_manager.set(
                    cache_key, 
                    self._serialize_model(data),
                    timeout=self.cache_manager.get_timeout('rt')
                )
                return data
            
            # 数据不存在或已过期，从API获取新数据
            logger.info("市场概览数据不存在或已过期，从API获取")
            data = await self._fetch_and_save_market_overview()
            if data:
                # 序列化并缓存
                self.cache_manager.set(
                    cache_key, 
                    self._serialize_model(data),
                    timeout=self.cache_manager.get_timeout('rt')
                )
            return data
        except Exception as e:
            logger.error(f"获取市场概览数据失败: {str(e)}")
            return None
            
    async def get_time_series_data(self, index_code: str, time_level: str, 
                                  start_time: Optional[datetime.datetime] = None,
                                  end_time: Optional[datetime.datetime] = None,
                                  limit: int = 100) -> List[IndexTimeSeriesData]:
        """
        获取指数的时间序列数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别（5、15、30、60、Day、Week、Month、Year）
            start_time: 开始时间，默认为None
            end_time: 结束时间，默认为None
            limit: 返回记录数量限制，默认100条
            
        Returns:
            List[TimeSeriesData]: 时间序列数据列表
        """
        # 标准化日期参数用于缓存键
        start_time_str = start_time.strftime('%Y%m%d') if start_time else 'none'
        end_time_str = end_time.strftime('%Y%m%d') if end_time else 'none'
        
        # 使用CacheManager生成标准化缓存键
        cache_key = self.cache_manager.generate_key(
            'ts', 
            'index', 
            index_code, 
            f"{time_level}_{limit}_{start_time_str}_{end_time_str}"
        )
        
        # 尝试从缓存获取
        cached_data = self.cache_manager.get(cache_key)
        if cached_data and isinstance(cached_data, list):
            # 将缓存数据转换为模型实例列表
            return [IndexTimeSeriesData(**item) for item in cached_data]
        
        # 从数据库获取
        index = await self.get_index_by_code(index_code)
        if not index:
            return []
        
        try:
            query = {'index': index, 'time_level': time_level}
            if start_time:
                query['trade_time__gte'] = start_time
            if end_time:
                query['trade_time__lte'] = end_time
            
            data = await sync_to_async(list)(
                IndexTimeSeriesData.objects.filter(**query).order_by('-trade_time')[:limit]
            )
            
            # 如果没有数据或数据很少，从API获取
            if len(data) < 10:
                logger.info(f"指数[{index_code}]的{time_level}级别时间序列数据不足，从API获取")
                await self._fetch_and_save_time_series(index_code, time_level)
                data = await sync_to_async(list)(
                    IndexTimeSeriesData.objects.filter(**query).order_by('-trade_time')[:limit]
                )
            
            # 序列化并缓存
            if data:
                serialized_data = [self._serialize_model(item) for item in data]
                self.cache_manager.set(
                    cache_key, 
                    serialized_data,
                    timeout=self.cache_manager.get_timeout('ts')
                )
            
            return data
        except Exception as e:
            logger.error(f"获取指数[{index_code}]的{time_level}级别时间序列数据失败: {str(e)}")
            return []    
    # ================ 技术指标通用方法 ================
    
    async def _get_technical_indicator_data(self, 
                                          index_code: str, 
                                          time_level: str,
                                          indicator_name: str,
                                          model_class: Type[models.Model],
                                          fetch_and_save_method: callable,
                                          start_time: Optional[datetime.datetime] = None,
                                          end_time: Optional[datetime.datetime] = None,
                                          limit: int = 100) -> List[Any]:
        """
        获取技术指标数据的通用方法
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            indicator_name: 指标名称 (kdj, macd, ma, boll)
            model_class: 模型类
            fetch_and_save_method: 获取并保存数据的方法
            start_time: 开始时间
            end_time: 结束时间
            limit: 返回记录数量限制
            
        Returns:
            List[Any]: 技术指标数据列表
        """
        # 标准化日期参数用于缓存键
        start_time_str = start_time.strftime('%Y%m%d') if start_time else 'none'
        end_time_str = end_time.strftime('%Y%m%d') if end_time else 'none'
        
        # 使用CacheManager生成标准化缓存键
        cache_key = self.cache_manager.generate_key(
            'calc', 'index', index_code, indicator_name,
            params={
                'level': time_level,
                'start': start_time_str,
                'end': end_time_str,
                'limit': limit
            }
        )

        
        # 尝试从缓存获取
        cached_data = self.cache_manager.get(cache_key)
        if cached_data and isinstance(cached_data, list):
            # 将缓存数据转换为模型实例列表
            return [model_class(**item) for item in cached_data]
        
        # 从数据库获取
        index = await self.get_index_by_code(index_code)
        if not index:
            return []
        
        try:
            query = {'index': index, 'time_level': time_level}
            if start_time:
                query['trade_time__gte'] = start_time
            if end_time:
                query['trade_time__lte'] = end_time
            
            data = await sync_to_async(list)(
                model_class.objects.filter(**query).order_by('-trade_time')[:limit]
            )
            
            # 如果没有数据或数据很少，从API获取
            if len(data) < 10:
                logger.info(f"指数[{index_code}]的{time_level}级别{indicator_name}数据不足，从API获取")
                await fetch_and_save_method(index_code, time_level)
                data = await sync_to_async(list)(
                    model_class.objects.filter(**query).order_by('-trade_time')[:limit]
                )
            
            # 序列化并缓存
            if data:
                serialized_data = [self._serialize_model(item) for item in data]
                self.cache_manager.set(
                    cache_key, 
                    serialized_data,
                    timeout=self.cache_manager.get_timeout('calc')
                )
            
            return data
        except Exception as e:
            logger.error(f"获取指数[{index_code}]的{time_level}级别{indicator_name}数据失败: {str(e)}")
            return []
            
    async def _save_indicator_datas(self, index_code: str, time_level: str, indicator_type: str,
        model_class, data_dicts: List[dict], update_fields: List[str]) -> List:
        """
        通用方法：批量保存指数指标数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            indicator_type: 指标类型名称（用于日志）
            model_class: 数据模型类
            data_dicts: 已处理好的数据字典列表，每个字典包含模型所需的所有字段
            update_fields: 需要更新的字段名称列表
            
        Returns:
            List: 保存的指标数据对象列表
        """
        try:
            # 获取指数信息
            index = await self.get_index_by_code(index_code)
            if not index:
                logger.warning(f"指数代码[{index_code}]不存在，无法保存{indicator_type}指标数据")
                return []
            
            if not data_dicts:
                logger.warning(f"没有可保存的指数[{index_code}]的{time_level}级别{indicator_type}指标数据")
                return []
            
            # 批量保存数据
            saved_records = []
            try:
                # 批量处理，每500条一批
                batch_size = 500
                
                # 获取所有交易时间
                trade_times = [data['trade_time'] for data in data_dicts]
                
                # 查询已存在的记录
                existing_records = await sync_to_async(list)(
                    model_class.objects.filter(
                        index=index,
                        time_level=time_level,
                        trade_time__in=trade_times
                    )
                )
                
                # 构建已存在记录的查找字典
                existing_dict = {
                    (record.trade_time): record for record in existing_records
                }
                
                # 分离需要创建和需要更新的记录
                records_to_create = []
                records_to_update = []
                
                for data in data_dicts:
                    trade_time = data['trade_time']
                    existing_record = existing_dict.get(trade_time)
                    
                    # 添加index字段到数据字典中
                    data['index'] = index
                    data['time_level'] = time_level
                    
                    if existing_record:
                        # 检查数据是否有变化
                        has_changes = False
                        for field_name in update_fields:
                            if getattr(existing_record, field_name) != data[field_name]:
                                has_changes = True
                                break
                        
                        if has_changes:
                            # 更新现有记录
                            for field_name in update_fields:
                                setattr(existing_record, field_name, data[field_name])
                            records_to_update.append(existing_record)
                        saved_records.append(existing_record)
                    else:
                        # 创建新记录
                        new_record = model_class(**data)
                        records_to_create.append(new_record)
                        saved_records.append(new_record)
                
                # 分批执行批量创建
                for i in range(0, len(records_to_create), batch_size):
                    batch = records_to_create[i:i+batch_size]
                    await sync_to_async(model_class.objects.bulk_create)(
                        batch, 
                        ignore_conflicts=False
                    )
                    logger.info(f"批量创建了 {len(batch)} 条指数[{index_code}]的{time_level}级别{indicator_type}指标数据")
                
                # 分批执行批量更新
                for i in range(0, len(records_to_update), batch_size):
                    batch = records_to_update[i:i+batch_size]
                    await sync_to_async(model_class.objects.bulk_update)(
                        batch, 
                        update_fields
                    )
                    logger.info(f"批量更新了 {len(batch)} 条指数[{index_code}]的{time_level}级别{indicator_type}指标数据")
                
                logger.info(f"共处理 {len(data_dicts)} 条{indicator_type}记录，创建 {len(records_to_create)} 条，更新 {len(records_to_update)} 条")
                return saved_records
                
            except Exception as e:
                logger.error(f"批量保存指数{indicator_type}指标数据失败: {str(e)}")
                return []
            
        except Exception as e:
            logger.error(f"保存指数{indicator_type}指标数据失败: {str(e)}")
            return []
    # ================ 写入方法 ================
    
    async def _fetch_and_save_indexes(self) -> None:
        """
        刷新所有指数数据
        
        从API获取指数列表并保存到数据库
        """
        try:
            # 获取沪深主要指数、沪市指数和深市指数
            main_index = await self.api.get_main_indexes()
            logging.warning(f"沪深主要指数type: {type(main_index)}")
            sh_index = await self.api.get_sh_indexes()
            logging.warning(f"沪市指数type: {type(sh_index)}")
            sz_index = await self.api.get_sz_indexes()
            logging.warning(f"深市指数type: {type(sz_index)}")
            all_indexes = []
            all_indexes.extend(main_index)
            all_indexes.extend(sh_index)
            all_indexes.extend(sz_index)
            
            # 在数据库事务中保存指数数据
            @transaction.atomic
            def save_indexes():
                for index_data in all_indexes:
                    try:
                        # 检查数据格式
                        if not isinstance(index_data, dict):
                            logger.warning(f"指数数据格式不正确: {type(index_data)}")
                            continue
                        
                        # 将API返回的数据转换为标准字典格式
                        data_dict = {
                            'code': index_data.get('dm', ''),  # 指数代码
                            'name': index_data.get('mc', ''),  # 指数名称
                            'exchange': index_data.get('jys', ''),  # 交易所代码
                        }
                        
                        # 检查必要字段
                        if not data_dict['code'] or not data_dict['name']:
                            logger.warning(f"指数数据缺少必要字段: {index_data}")
                            continue
                        
                        # 创建或更新指数数据
                        index, created = IndexInfo.objects.update_or_create(
                            code=data_dict['code'],
                            defaults=data_dict
                        )
                        logger.debug(f"{'创建' if created else '更新'}指数: {index.code} {index.name}")
                    except Exception as e:
                        logger.error(f"保存指数数据失败 {index_data.get('dm', 'unknown')}: {str(e)}")
            
            await sync_to_async(save_indexes)()
            logger.info(f"成功保存{len(all_indexes)}个指数数据")
        except Exception as e:
            logger.error(f"刷新指数数据失败: {str(e)}")
            raise
    
    async def _fetch_and_save_realtime_data(self, index_code: str) -> Optional[IndexRealTimeData]:
        """
        获取并保存指数实时数据
        
        Args:
            index_code: 指数代码
            
        Returns:
            Optional[IndexRealTimeData]: 保存的实时数据对象
        """
        try:
            # 获取指数信息
            try:
                index = await self.get_index_by_code(index_code)
            except Exception as e:
                logger.error(f"self.get_index_by_code(index_code) 获取指数信息失败: {str(e)}")
                return None
            if not index:
                logger.warning(f"指数代码[{index_code}]不存在，无法获取实时数据")
                return None
            # 调用API获取实时数据
            api_data = await self.api.get_index_realtime_data(index_code)
            if not api_data:
                logger.warning(f"API未返回指数[{index_code}]的实时数据")
                return None
                      
            # 在数据库事务中保存数据
            @transaction.atomic
            def save_data():                
                # 将映射后的数据转换为标准字典格式，并处理日期和数字字段
                data_dict = {
                    'index': index,
                    'open_price': self._parse_number(api_data.get('o')),  # 开盘价
                    'high_price': self._parse_number(api_data.get('h')),  # 最高价
                    'low_price': self._parse_number(api_data.get('l')),  # 最低价
                    'current_price': self._parse_number(api_data.get('p')),  # 当前价格
                    'prev_close_price': self._parse_number(api_data.get('yc')),  # 昨日收盘价
                    'price_change': self._parse_number(api_data.get('ud')),  # 涨跌额
                    'price_change_percent': self._parse_number(api_data.get('pc')),  # 涨跌幅
                    'five_minute_change_percent': self._parse_number(api_data.get('fm')),  # 五分钟涨跌幅
                    'amplitude': self._parse_number(api_data.get('zf')),  # 振幅
                    'change_speed': self._parse_number(api_data.get('zs')),  # 涨速
                    'sixty_day_change_percent': self._parse_number(api_data.get('zdf60')),  # 60日涨跌幅
                    'ytd_change_percent': self._parse_number(api_data.get('zdfnc')),  # 年初至今涨跌幅
                    'volume': self._parse_number(api_data.get('v')),  # 成交量
                    'turnover': self._parse_number(api_data.get('cje')),  # 成交额
                    'turnover_rate': self._parse_number(api_data.get('hs')),  # 换手率
                    'volume_ratio': self._parse_number(api_data.get('lb')),  # 量比
                    'pe_ratio': self._parse_number(api_data.get('pe')),  # 市盈率
                    'pb_ratio': self._parse_number(api_data.get('sjl')),  # 市净率
                    'circulating_market_value': self._parse_number(api_data.get('lt')),  # 流通市值
                    'total_market_value': self._parse_number(api_data.get('sz')),  # 总市值
                    'update_time': self._parse_datetime(api_data.get('t')),  # 更新时间
                }
                
                # 创建实时数据记录
                realtime_data = IndexRealTimeData.objects.create(**data_dict)
                return realtime_data
            
            # 执行保存操作并返回结果
            result = await sync_to_async(save_data)()
            logger.info(f"成功保存指数[{index_code}]的实时数据")
            # 如果成功保存数据，也更新缓存
            if result:
                cache_key = self.cache_manager.generate_key('rt', 'index', index_code, 'realtime')
                self.cache_manager.set(
                    cache_key, 
                    self._serialize_model(result),
                    timeout=self.cache_manager.get_timeout('rt')
                )
            return result
        except Exception as e:
            logger.error(f"获取并保存指数[{index_code}]实时数据失败: {str(e)}")
            return None
    
    async def _fetch_and_save_market_overview(self) -> Optional[MarketOverview]:
        """
        获取并保存市场概览数据
        
        Returns:
            Optional[MarketOverview]: 保存的市场概览数据对象
        """
        try:
            # 调用API获取市场概览数据
            api_data = await self.api.get_market_overview()
            if not api_data:
                logger.warning("API未返回市场概览数据")
                return None
            
            # 在数据库事务中保存数据
            @transaction.atomic
            def save_data():
                # 将映射后的数据转换为标准字典格式，并处理日期和数字字段
                data_dict = {
                    'total_up': int(api_data.get('totalUp')), # 上涨总数
                    'total_down': int(api_data.get('totalDown')), # 下跌总数
                    'limit_up': int(api_data.get('zt')), # 涨停总数
                    'limit_down': int(api_data.get('dt')), # 跌停总数
                    'up_8_to_limit': int(api_data.get('up8ToZt')), # 上涨8%~涨停数量
                    'up_6_to_8': int(api_data.get('up6To8')), # 上涨6%~8%数量
                    'up_4_to_6': int(api_data.get('up4To6')), # 上涨4%~6%数量
                    'up_2_to_4': int(api_data.get('up2To4')), # 上涨2%~4%数量
                    'up_0_to_2': int(api_data.get('up0To2')), # 上涨0%~2%数量
                    'down_0_to_2': int(api_data.get('down0To2')), # 下跌0%~2%数量
                    'down_2_to_4': int(api_data.get('down2To4')), # 下跌2%~4%数量
                    'down_4_to_6': int(api_data.get('down4To6')), # 下跌4%~6%数量
                    'down_6_to_8': int(api_data.get('down6To8')), # 下跌6%~8%数量
                    'down_8_to_limit': int(api_data.get('down8ToDt')), # 下跌8%~跌停数量
                    'update_time': datetime.datetime.now()
                }
                
                # 创建实时数据记录
                realtime_data = MarketOverview.objects.create(**data_dict)
                return realtime_data
            
            # 执行保存操作并返回结果
            result = await sync_to_async(save_data)()
            logger.info("成功保存市场概览数据")
            return result
        except Exception as e:
            logger.error(f"获取并保存市场概览数据失败: {str(e)}")
            return None
    
    async def _fetch_and_save_time_series(self, index_code: str, time_level: str) -> List[IndexTimeSeriesData]:
        """
        获取并保存指数时间序列数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别（5、15、30、60、Day、Week、Month、Year）
            
        Returns:
            List[IndexTimeSeriesData]: 保存的时间序列数据对象列表
        """
        try:
            # 获取指数信息
            index = await self.get_index_by_code(index_code)
            if not index:
                logger.warning(f"指数代码[{index_code}]不存在，无法获取时间序列数据")
                return []
            
            if time_level in ['5', '15', '30', '60', 'Day', 'Week', 'Month']:
                api_data = await self.api.get_latest_time_series(index_code, time_level)
            else:
                logger.warning(f"不支持的时间级别: {time_level}")
                return []
            
            if not api_data:
                logger.warning(f"API未返回指数[{index_code}]的{time_level}级别时间序列数据")
                return []
            
            # 在数据库事务中保存数据
            @transaction.atomic
            def save_data():
                saved_items = []
                
                # 处理API返回的数据
                if isinstance(api_data, str):
                    try:
                        # 尝试解析JSON字符串
                        import json
                        data_list = json.loads(api_data)
                        if not isinstance(data_list, list):
                            data_list = [data_list]
                    except Exception as e:
                        logger.error(f"解析时间序列数据失败: {str(e)}")
                        return []
                else:
                    data_list = api_data if isinstance(api_data, list) else [api_data]
                
                for item in data_list:
                    # 将API返回的数据转换为标准字典格式，并处理日期和数字字段
                    data_dict = {
                        'index': index,
                        'time_level': time_level,
                        'trade_time': self._parse_datetime(item.get('d')),  # 交易时间
                        'open_price': self._parse_number(item.get('o')),  # 开盘价
                        'high_price': self._parse_number(item.get('h')),  # 最高价
                        'low_price': self._parse_number(item.get('l')),  # 最低价
                        'close_price': self._parse_number(item.get('c')),  # 收盘价
                        'volume': self._parse_number(item.get('v')),  # 成交量
                        'turnover': self._parse_number(item.get('e')),  # 成交额
                        'amplitude': self._parse_number(item.get('zf')),  # 振幅
                        'turnover_rate': self._parse_number(item.get('hs')),  # 换手率
                        'change_percent': self._parse_number(item.get('zd')),  # 涨跌幅
                        'change_amount': self._parse_number(item.get('zde')),  # 涨跌额
                    }
                    
                    # 尝试查找现有记录
                    try:
                        # 使用 index + time_level + trade_time 作为唯一约束
                        time_series, created = IndexTimeSeriesData.objects.update_or_create(
                            index=index,
                            time_level=time_level,
                            trade_time=data_dict['trade_time'],
                            defaults=data_dict
                        )
                        saved_items.append(time_series)
                    except Exception as e:
                        logger.error(f"保存时间序列数据失败: {str(e)}")
                
                return saved_items
            
            # 执行保存操作并返回结果
            result = await sync_to_async(save_data)()
            logger.info(f"成功保存指数[{index_code}]的{time_level}级别时间序列数据，共{len(result)}条")
            return result
        except Exception as e:
            logger.error(f"获取并保存指数[{index_code}]的{time_level}级别时间序列数据失败: {str(e)}")
            return []
    
    async def _fetch_and_save_history_time_series(self, index_code: str, time_level: str) -> List[IndexTimeSeriesData]:
        """
        获取并保存指数历史时间序列数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别

        Returns:
            List[IndexTimeSeriesData]: 保存的时间序列数据对象列表
        """
        try:
            # 获取指数信息
            index = await self.get_index_by_code(index_code)
            if not index:
                logger.warning(f"指数代码[{index_code}]不存在，无法获取时间序列数据")
                return []
            
            if time_level in ['5', '15', '30', '60', 'Day', 'Week', 'Month']:
                api_data = await self.api.get_history_time_series(index_code, time_level)
            else:
                logger.warning(f"不支持的时间级别: {time_level}")
                return []
            
            if not api_data:
                logger.warning(f"API未返回指数[{index_code}]的{time_level}级别时间序列数据")
                return []
            
            # 在数据库事务中保存数据
            @transaction.atomic
            def save_data():
                saved_items = []
                
                # 处理API返回的数据
                if isinstance(api_data, str):
                    try:
                        # 尝试解析JSON字符串
                        import json
                        data_list = json.loads(api_data)
                        if not isinstance(data_list, list):
                            data_list = [data_list]
                    except Exception as e:
                        logger.error(f"解析时间序列数据失败: {str(e)}")
                        return []
                else:
                    data_list = api_data if isinstance(api_data, list) else [api_data]
                
                for item in data_list:
                    # 将API返回的数据转换为标准字典格式，并处理日期和数字字段
                    data_dict = {
                        'index': index,
                        'time_level': time_level,
                        'trade_time': self._parse_datetime(item.get('d')),  # 交易时间
                        'open_price': self._parse_number(item.get('o')),  # 开盘价
                        'high_price': self._parse_number(item.get('h')),  # 最高价
                        'low_price': self._parse_number(item.get('l')),  # 最低价
                        'close_price': self._parse_number(item.get('c')),  # 收盘价
                        'volume': self._parse_number(item.get('v')),  # 成交量
                        'turnover': self._parse_number(item.get('e')),  # 成交额
                        'amplitude': self._parse_number(item.get('zf')),  # 振幅
                        'turnover_rate': self._parse_number(item.get('hs')),  # 换手率
                        'change_percent': self._parse_number(item.get('zd')),  # 涨跌幅
                        'change_amount': self._parse_number(item.get('zde')),  # 涨跌额
                    }
                    
                    # 尝试查找现有记录
                    try:
                        # 使用 index + time_level + trade_time 作为唯一约束
                        time_series, created = IndexTimeSeriesData.objects.update_or_create(
                            index=index,
                            time_level=time_level,
                            trade_time=data_dict['trade_time'],
                            defaults=data_dict
                        )
                        saved_items.append(time_series)
                    except Exception as e:
                        logger.error(f"保存时间序列数据失败: {str(e)}")
                
                return saved_items
            
            # 执行保存操作并返回结果
            result = await sync_to_async(save_data)()
            logger.info(f"成功保存指数[{index_code}]的{time_level}级别时间序列数据，共{len(result)}条")
            return result
        except Exception as e:
            logger.error(f"获取并保存指数[{index_code}]的{time_level}级别时间序列数据失败: {str(e)}")
            return []
    
    # ================ 公共方法 ================
    
    async def refresh_all_indexes(self) -> bool:
        """
        刷新所有指数信息
        
        Returns:
            bool: 是否成功刷新
        """
        try:
            await self._fetch_and_save_indexes()
            # 清除相关缓存
            cache.delete('stock_indexes_all')
            
            return True
        except Exception as e:
            logger.error(f"刷新所有指数信息失败: {str(e)}")
            return False
    
    async def refresh_index_realtime_data(self, index_code: str) -> Optional[IndexRealTimeData]:
        """
        刷新指数实时数据
        
        Args:
            index_code: 指数代码
            
        Returns:
            Optional[IndexRealTimeData]: 更新后的实时数据对象
        """
        try:
            data = await self._fetch_and_save_realtime_data(index_code)
            if data:
                # 清除相关缓存
                cache.delete(f'index_realtime_{index_code}')
            return data
        except Exception as e:
            logger.error(f"刷新指数[{index_code}]实时数据失败: {str(e)}")
            return None
    
    async def refresh_market_overview(self) -> Optional[MarketOverview]:
        """
        刷新市场概览数据
        
        Returns:
            Optional[MarketOverview]: 更新后的市场概览数据对象
        """
        try:
            data = await self._fetch_and_save_market_overview()
            if data:
                # 清除相关缓存
                cache.delete('market_overview_latest')
            return data
        except Exception as e:
            logger.error(f"刷新市场概览数据失败: {str(e)}")
            return None
    
    async def refresh_time_series_data(self, index_code: str, time_level: str) -> List[IndexTimeSeriesData]:
        """
        刷新指数时间序列数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            
        Returns:
            List[IndexTimeSeriesData]: 更新后的时间序列数据对象列表
        """
        try:
            data = await self._fetch_and_save_time_series(index_code, time_level)
            if data:
                # 清除所有相关的缓存键
                cache_keys = [k for k in cache._cache.keys() if k.startswith(f'time_series_{index_code}_{time_level}')]
                for key in cache_keys:
                    cache.delete(key)
            return data
        except Exception as e:
            logger.error(f"刷新指数[{index_code}]的{time_level}级别时间序列数据失败: {str(e)}")
            return []
    
    # ================ 技术指标方法 ================
    async def get_kdj_data(self, index_code: str, time_level: str, 
                        start_time: Optional[datetime.datetime] = None,
                        end_time: Optional[datetime.datetime] = None,
                        limit: int = 100) -> List[IndexKDJData]:
        """
        获取指数KDJ指标数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            start_time: 开始时间
            end_time: 结束时间
            limit: 返回记录数量限制
            
        Returns:
            List[IndexKDJData]: KDJ指标数据列表
        """
        return await self._get_technical_indicator_data(
            index_code=index_code,
            time_level=time_level,
            indicator_name='kdj',
            model_class=IndexKDJData,
            fetch_and_save_method=self._fetch_and_save_kdj,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
    
    async def get_macd_data(self, index_code: str, time_level: str, 
                            start_time: Optional[datetime.datetime] = None,
                            end_time: Optional[datetime.datetime] = None,
                            limit: int = 100) -> List[IndexMACDData]:
        """
        获取指数MACD指标数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            start_time: 开始时间
            end_time: 结束时间
            limit: 返回记录数量限制
            
        Returns:
            List[IndexMACDData]: MACD指标数据列表
        """
        return await self._get_technical_indicator_data(
            index_code=index_code,
            time_level=time_level,
            indicator_name='macd',
            model_class=IndexMACDData,
            fetch_and_save_method=self._fetch_and_save_macd,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
    
    async def get_ma_data(self, index_code: str, time_level: str, 
                        start_time: Optional[datetime.datetime] = None,
                        end_time: Optional[datetime.datetime] = None,
                        limit: int = 100) -> List[IndexMAData]:
        """
        获取指数MA指标数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            start_time: 开始时间
            end_time: 结束时间
            limit: 返回记录数量限制
            
        Returns:
            List[IndexMAData]: MA指标数据列表
        """
        return await self._get_technical_indicator_data(
            index_code=index_code,
            time_level=time_level,
            indicator_name='ma',
            model_class=IndexMAData,
            fetch_and_save_method=self._fetch_and_save_ma,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
    
    async def get_boll_data(self, index_code: str, time_level: str, 
                        start_time: Optional[datetime.datetime] = None,
                        end_time: Optional[datetime.datetime] = None,
                        limit: int = 100) -> List[IndexBOLLData]:
        """
        获取指数BOLL指标数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            start_time: 开始时间
            end_time: 结束时间
            limit: 返回记录数量限制
            
        Returns:
            List[IndexBOLLData]: BOLL指标数据列表
        """
        return await self._get_technical_indicator_data(
            index_code=index_code,
            time_level=time_level,
            indicator_name='boll',
            model_class=IndexBOLLData,
            fetch_and_save_method=self._fetch_and_save_boll,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
    
    # ================ 指数KDJ指标数据 ================
    async def _fetch_and_save_kdj(self, index_code: str, time_level: str) -> List[IndexKDJData]:
        """
        获取并保存指数KDJ指标数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            
        Returns:
            List[IndexKDJData]: 保存的KDJ指标数据对象列表
        """
        try:
            # 调用API获取KDJ指标数据
            api_data_list = await self.api.get_latest_kdj(index_code, time_level)
            if not api_data_list:
                logger.warning(f"API未返回指数[{index_code}]的{time_level}级别KDJ指标数据")
                return []
            
            # 确保api_data_list是列表类型，如果是单个记录，转换为列表
            if not isinstance(api_data_list, list):
                api_data_list = [api_data_list]
            
            # 处理API返回的数据
            data_dicts = []
            try:
                for api_data in api_data_list:
                    if isinstance(api_data, str):
                        # 尝试解析JSON字符串
                        import json
                        mapped_data = json.loads(api_data)
                    else:
                        # 处理JSON格式的数据
                        mapped_data = api_data
                    
                    # 将映射后的数据转换为标准字典格式
                    data_dict = {
                        'trade_time': self._parse_datetime(mapped_data.get('t')),  # 交易时间
                        'k_value': float(mapped_data.get('k')),  # K值
                        'd_value': float(mapped_data.get('d')),  # D值
                        'j_value': float(mapped_data.get('j')),  # J值
                    }
                    data_dicts.append(data_dict)
            except Exception as e:
                logger.error(f"解析指数KDJ指标数据失败: {str(e)}")
                return []
            
            # 使用通用方法批量保存数据
            update_fields = ['k_value', 'd_value', 'j_value']
            return await self._save_indicator_datas(
                index_code=index_code,
                time_level=time_level,
                indicator_type='KDJ',
                model_class=IndexKDJData,
                data_dicts=data_dicts,
                update_fields=update_fields
            )         
        except Exception as e:
            logger.error(f"获取并保存指数KDJ指标数据失败: {str(e)}")
            return []
    
    async def _fetch_and_save_history_kdj(self, index_code: str, time_level: str) -> List[IndexKDJData]:
        """
        获取并保存指数历史KDJ指标数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            
        Returns:
            List[IndexKDJData]: 保存的KDJ指标数据对象列表
        """
        try:
            # 调用API获取KDJ指标数据
            api_data_list = await self.api.get_history_kdj(index_code, time_level)
            if not api_data_list:
                logger.warning(f"API未返回指数[{index_code}]的{time_level}级别历史KDJ指标数据")
                return []
            
            # 确保api_data_list是列表类型，如果是单个记录，转换为列表
            if not isinstance(api_data_list, list):
                api_data_list = [api_data_list]
            
            # 处理API返回的数据
            data_dicts = []
            try:
                for api_data in api_data_list:
                    if isinstance(api_data, str):
                        # 尝试解析JSON字符串
                        import json
                        mapped_data = json.loads(api_data)
                    else:
                        # 处理JSON格式的数据
                        mapped_data = api_data
                    
                    # 将映射后的数据转换为标准字典格式
                    data_dict = {
                        'trade_time': self._parse_datetime(mapped_data.get('t')),  # 交易时间
                        'k_value': float(mapped_data.get('k')),  # K值
                        'd_value': float(mapped_data.get('d')),  # D值
                        'j_value': float(mapped_data.get('j')),  # J值
                    }
                    data_dicts.append(data_dict)
            except Exception as e:
                logger.error(f"解析指数历史KDJ指标数据失败: {str(e)}")
            return []
            
            # 使用通用方法批量保存数据
            update_fields = ['k_value', 'd_value', 'j_value']
            return await self._save_indicator_datas(
                index_code=index_code,
                time_level=time_level,
                indicator_type='KDJ',
                model_class=IndexKDJData,
                data_dicts=data_dicts,
                update_fields=update_fields
            )         
        except Exception as e:
            logger.error(f"获取并保存指数历史KDJ指标数据失败: {str(e)}")
            return []
    
    # ================ 指数MACD指标数据 ================
    async def _fetch_and_save_macd(self, index_code: str, time_level: str) -> List[IndexMACDData]:
        """
        获取并保存指数MACD指标数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            
        Returns:
            List[IndexMACDData]: 保存的MACD指标数据对象列表
        """
        try:
            # 调用API获取MACD指标数据
            api_data_list = await self.api.get_latest_macd(index_code, time_level)
            if not api_data_list:
                logger.warning(f"API未返回指数[{index_code}]的{time_level}级别MACD指标数据")
                return []
            
            # 确保api_data_list是列表类型，如果是单个记录，转换为列表
            if not isinstance(api_data_list, list):
                api_data_list = [api_data_list]
            
            # 处理API返回的数据
            data_dicts = []
            try:
                from decimal import Decimal
                
                for api_data in api_data_list:
                    if isinstance(api_data, str):
                        # 尝试解析JSON字符串
                        import json
                        mapped_data = json.loads(api_data)
                    else:
                        # 处理JSON格式的数据
                        mapped_data = api_data
                    
                    # 将映射后的数据转换为标准字典格式，使用Decimal而不是float
                    data_dict = {
                        'trade_time': self._parse_datetime(mapped_data.get('t')),  # 交易时间
                        'diff_value': Decimal(str(mapped_data.get('diff'))),  # DIFF值
                        'dea_value': Decimal(str(mapped_data.get('dea'))),    # DEA值
                        'macd_value': Decimal(str(mapped_data.get('macd'))),  # MACD值
                    }
                    data_dicts.append(data_dict)
            except Exception as e:
                logger.error(f"解析指数MACD指标数据失败: {str(e)}")
                return []
            
            # 使用通用方法批量保存数据
            update_fields = ['diff_value', 'dea_value', 'macd_value']
            return await self._save_indicator_datas(
                index_code=index_code,
                time_level=time_level,
                indicator_type='MACD',
                model_class=IndexMACDData,
                data_dicts=data_dicts,
                update_fields=update_fields
            )
        except Exception as e:
            logger.error(f"获取并保存指数MACD指标数据失败: {str(e)}")
            return []

    async def _fetch_and_save_history_macd(self, index_code: str, time_level: str) -> List[IndexMACDData]:
        """
        获取并保存指数历史MACD指标数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            
        Returns:
            List[IndexMACDData]: 保存的MACD指标数据对象列表
        """
        try:
            # 调用API获取MACD指标数据
            api_data_list = await self.api.get_history_macd(index_code, time_level)
            if not api_data_list:
                logger.warning(f"API未返回指数[{index_code}]的{time_level}级别历史MACD指标数据")
                return []
            
            # 确保api_data_list是列表类型，如果是单个记录，转换为列表
            if not isinstance(api_data_list, list):
                api_data_list = [api_data_list]
            
            # 处理API返回的数据
            data_dicts = []
            try:
                from decimal import Decimal
                
                for api_data in api_data_list:
                    if isinstance(api_data, str):
                        # 尝试解析JSON字符串
                        import json
                        mapped_data = json.loads(api_data)
                    else:
                        # 处理JSON格式的数据
                        mapped_data = api_data
                    
                    # 将映射后的数据转换为标准字典格式，使用Decimal而不是float
                    data_dict = {
                        'trade_time': self._parse_datetime(mapped_data.get('t')),  # 交易时间
                        'diff_value': Decimal(str(mapped_data.get('diff'))),  # DIFF值
                        'dea_value': Decimal(str(mapped_data.get('dea'))),    # DEA值
                        'macd_value': Decimal(str(mapped_data.get('macd'))),  # MACD值
                    }
                    data_dicts.append(data_dict)
            except Exception as e:
                logger.error(f"解析指数历史MACD指标数据失败: {str(e)}")
            return []
            
            # 使用通用方法批量保存数据
            update_fields = ['diff_value', 'dea_value', 'macd_value']
            return await self._save_indicator_datas(
                index_code=index_code,
                time_level=time_level,
                indicator_type='MACD',
                model_class=IndexMACDData,
                data_dicts=data_dicts,
                update_fields=update_fields
            )
        except Exception as e:
            logger.error(f"获取并保存指数历史MACD指标数据失败: {str(e)}")
            return []

    # ================ 指数MA指标数据 ================
    async def _fetch_and_save_ma(self, index_code: str, time_level: str) -> List[IndexMAData]:
        """
        获取并保存指数MA指标数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            
        Returns:
            List[IndexMAData]: 保存的MA指标数据对象列表
        """
        try:
            # 调用API获取MACD指标数据
            api_data_list = await self.api.get_latest_ma(index_code, time_level)
            if not api_data_list:
                logger.warning(f"API未返回指数[{index_code}]的{time_level}级别指数MA指标数据")
                return []
            
            # 确保api_data_list是列表类型，如果是单个记录，转换为列表
            if not isinstance(api_data_list, list):
                api_data_list = [api_data_list]
            
            # 处理API返回的数据
            data_dicts = []
            try:
                from decimal import Decimal
                
                for api_data in api_data_list:
                    if isinstance(api_data, str):
                        # 尝试解析JSON字符串
                        import json
                        mapped_data = json.loads(api_data)
                    else:
                        # 处理JSON格式的数据
                        mapped_data = api_data
                    
                    # 将映射后的数据转换为标准字典格式，使用Decimal而不是float
                    data_dict = {
                        'trade_time': self._parse_datetime(mapped_data.get('trade_time')),
                        'ma3': self._parse_number(mapped_data.get('ma3')),
                        'ma5': self._parse_number(mapped_data.get('ma5')),
                        'ma10': self._parse_number(mapped_data.get('ma10')),
                        'ma15': self._parse_number(mapped_data.get('ma15')),
                        'ma20': self._parse_number(mapped_data.get('ma20')),
                        'ma30': self._parse_number(mapped_data.get('ma30')),
                        'ma60': self._parse_number(mapped_data.get('ma60')),
                        'ma120': self._parse_number(mapped_data.get('ma120')),
                        'ma200': self._parse_number(mapped_data.get('ma200')),
                        'ma250': self._parse_number(mapped_data.get('ma250')),
                    }
                    data_dicts.append(data_dict)
            except Exception as e:
                logger.error(f"解析指数MA指标数据失败: {str(e)}")
                return []
            
            # 使用通用方法批量保存数据
            update_fields = ['ma3', 'ma5', 'ma10', 'ma15', 'ma20', 'ma30', 'ma60', 'ma120', 'ma200', 'ma250']
            return await self._save_indicator_datas(
                index_code=index_code,
                time_level=time_level,
                indicator_type='MA',
                model_class=IndexMAData,
                data_dicts=data_dicts,
                update_fields=update_fields
            )
        except Exception as e:
            logger.error(f"获取并保存指数MA指标数据失败: {str(e)}")
            return []
    
    async def _fetch_and_save_history_ma(self, index_code: str, time_level: str) -> List[IndexMAData]:
        """
        获取并保存历史指数MA指标数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            
        Returns:
            List[IndexMAData]: 保存的MA指标数据对象列表
        """
        try:
            # 调用API获取MACD指标数据
            api_data_list = await self.api.get_history_ma(index_code, time_level)
            if not api_data_list:
                logger.warning(f"API未返回指数[{index_code}]的{time_level}级别历史指数MA指标数据")
                return []
            
            # 确保api_data_list是列表类型，如果是单个记录，转换为列表
            if not isinstance(api_data_list, list):
                api_data_list = [api_data_list]
            
            # 处理API返回的数据
            data_dicts = []
            try:
                from decimal import Decimal
                
                for api_data in api_data_list:
                    if isinstance(api_data, str):
                        # 尝试解析JSON字符串
                        import json
                        mapped_data = json.loads(api_data)
                    else:
                        # 处理JSON格式的数据
                        mapped_data = api_data
                    
                    # 将映射后的数据转换为标准字典格式，使用Decimal而不是float
                    data_dict = {
                        'trade_time': self._parse_datetime(mapped_data.get('trade_time')),
                        'ma3': self._parse_number(mapped_data.get('ma3')),
                        'ma5': self._parse_number(mapped_data.get('ma5')),
                        'ma10': self._parse_number(mapped_data.get('ma10')),
                        'ma15': self._parse_number(mapped_data.get('ma15')),
                        'ma20': self._parse_number(mapped_data.get('ma20')),
                        'ma30': self._parse_number(mapped_data.get('ma30')),
                        'ma60': self._parse_number(mapped_data.get('ma60')),
                        'ma120': self._parse_number(mapped_data.get('ma120')),
                        'ma200': self._parse_number(mapped_data.get('ma200')),
                        'ma250': self._parse_number(mapped_data.get('ma250')),
                    }
                    data_dicts.append(data_dict)
            except Exception as e:
                logger.error(f"解析历史指数MA指标数据失败: {str(e)}")
            return []
            
            # 使用通用方法批量保存数据
            update_fields = ['ma3', 'ma5', 'ma10', 'ma15', 'ma20', 'ma30', 'ma60', 'ma120', 'ma200', 'ma250']
            return await self._save_indicator_datas(
                index_code=index_code,
                time_level=time_level,
                indicator_type='MA',
                model_class=IndexMAData,
                data_dicts=data_dicts,
                update_fields=update_fields
            )
        except Exception as e:
            logger.error(f"获取并保存历史指数MA指标数据失败: {str(e)}")
            return []
    
    # ================ 指数BOLL指标数据 ================
    async def _fetch_and_save_boll(self, index_code: str, time_level: str) -> List[IndexBOLLData]:
        """
        获取并保存指数BOLL指标数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            
        Returns:
            List[IndexBOLLData]: 保存的BOLL指标数据对象列表
        """
        try:
            # 调用API获取MACD指标数据
            api_data_list = await self.api.get_latest_ma(index_code, time_level)
            if not api_data_list:
                logger.warning(f"API未返回指数[{index_code}]的{time_level}级别指数BOLL指标数据")
                return []
            
            # 确保api_data_list是列表类型，如果是单个记录，转换为列表
            if not isinstance(api_data_list, list):
                api_data_list = [api_data_list]
            
            # 处理API返回的数据
            data_dicts = []
            try:
                from decimal import Decimal
                
                for api_data in api_data_list:
                    if isinstance(api_data, str):
                        # 尝试解析JSON字符串
                        import json
                        mapped_data = json.loads(api_data)
                    else:
                        # 处理JSON格式的数据
                        mapped_data = api_data
                    
                    # 将映射后的数据转换为标准字典格式，使用Decimal而不是float
                    data_dict = {
                        'trade_time': self._parse_datetime(mapped_data.get('t')),
                        'upper': self._parse_number(mapped_data.get('u')),
                        'middle': self._parse_number(mapped_data.get('m')),
                        'lower': self._parse_number(mapped_data.get('d')),
                    }
                    data_dicts.append(data_dict)
            except Exception as e:
                logger.error(f"解析指数BOLL指标数据失败: {str(e)}")
                return []
            
            # 使用通用方法批量保存数据
            update_fields = ['upper', 'middle', 'lower']
            return await self._save_indicator_datas(
                index_code=index_code,
                time_level=time_level,
                indicator_type='BOLL',
                model_class=IndexBOLLData,
                data_dicts=data_dicts,
                update_fields=update_fields
            )
        except Exception as e:
            logger.error(f"获取并保存指数BOLL指标数据失败: {str(e)}")
            return []
            
    async def _fetch_and_save_history_boll(self, index_code: str, time_level: str) -> List[IndexBOLLData]:
        """
        获取并保存历史指数BOLL指标数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            
        Returns:
            List[IndexBOLLData]: 保存的BOLL指标数据对象列表
        """
        try:
            # 调用API获取MACD指标数据
            api_data_list = await self.api.get_latest_ma(index_code, time_level)
            if not api_data_list:
                logger.warning(f"API未返回指数[{index_code}]的{time_level}级别历史指数BOLL指标数据")
                return []
            
            # 确保api_data_list是列表类型，如果是单个记录，转换为列表
            if not isinstance(api_data_list, list):
                api_data_list = [api_data_list]
            
            # 处理API返回的数据
            data_dicts = []
            try:
                from decimal import Decimal
                
                for api_data in api_data_list:
                    if isinstance(api_data, str):
                        # 尝试解析JSON字符串
                        import json
                        mapped_data = json.loads(api_data)
                    else:
                        # 处理JSON格式的数据
                        mapped_data = api_data
                    
                    # 将映射后的数据转换为标准字典格式，使用Decimal而不是float
                    data_dict = {
                        'trade_time': self._parse_datetime(mapped_data.get('t')),
                        'upper': self._parse_number(mapped_data.get('u')),
                        'middle': self._parse_number(mapped_data.get('m')),
                        'lower': self._parse_number(mapped_data.get('d')),
                    }
                    data_dicts.append(data_dict)
            except Exception as e:
                logger.error(f"解析历史指数BOLL指标数据失败: {str(e)}")
            return []
            
            # 使用通用方法批量保存数据
            update_fields = ['upper', 'middle', 'lower']
            return await self._save_indicator_datas(
                index_code=index_code,
                time_level=time_level,
                indicator_type='BOLL',
                model_class=IndexBOLLData,
                data_dicts=data_dicts,
                update_fields=update_fields
            )
        except Exception as e:
            logger.error(f"获取并保存历史指数BOLL指标数据失败: {str(e)}")
            return []
    
    
    async def refresh_kdj_data(self, index_code: str, time_level: str) -> List[IndexKDJData]:
        """
        刷新指数KDJ指标数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            
        Returns:
            List[IndexKDJData]: 更新后的KDJ指标数据对象列表
        """
        try:
            data = await self._fetch_and_save_kdj(index_code, time_level)
            if data:
                # 清除所有相关的缓存键
                cache_keys = [k for k in cache._cache.keys() if k.startswith(f'kdj_{index_code}_{time_level}')]
                for key in cache_keys:
                    cache.delete(key)
            return data
        except Exception as e:
            logger.error(f"刷新指数[{index_code}]的{time_level}级别KDJ指标数据失败: {str(e)}")
            return []
    
    async def refresh_macd_data(self, index_code: str, time_level: str) -> List[IndexMACDData]:
        """
        刷新指数MACD指标数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            
        Returns:
            List[IndexMACDData]: 更新后的MACD指标数据对象列表
        """
        try:
            data = await self._fetch_and_save_macd(index_code, time_level)
            if data:
                # 清除所有相关的缓存键
                cache_keys = [k for k in cache._cache.keys() if k.startswith(f'macd_{index_code}_{time_level}')]
                for key in cache_keys:
                    cache.delete(key)
            return data
        except Exception as e:
            logger.error(f"刷新指数[{index_code}]的{time_level}级别MACD指标数据失败: {str(e)}")
            return []
    
    async def refresh_ma_data(self, index_code: str, time_level: str) -> List[IndexMAData]:
        """
        刷新指数MA指标数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            
        Returns:
            List[IndexMAData]: 更新后的MA指标数据对象列表
        """
        try:
            data = await self._fetch_and_save_ma(index_code, time_level)
            if data:
                # 清除所有相关的缓存键
                cache_keys = [k for k in cache._cache.keys() if k.startswith(f'ma_{index_code}_{time_level}')]
                for key in cache_keys:
                    cache.delete(key)
            return data
        except Exception as e:
            logger.error(f"刷新指数[{index_code}]的{time_level}级别MA指标数据失败: {str(e)}")
            return []
    
    async def refresh_boll_data(self, index_code: str, time_level: str) -> List[IndexBOLLData]:
        """
        刷新指数BOLL指标数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            
        Returns:
            List[IndexBOLLData]: 更新后的BOLL指标数据对象列表
        """
        try:
            data = await self._fetch_and_save_boll(index_code, time_level)
            if data:
                # 清除所有相关的缓存键
                cache_keys = [k for k in cache._cache.keys() if k.startswith(f'boll_{index_code}_{time_level}')]
                for key in cache_keys:
                    cache.delete(key)
            return data
        except Exception as e:
            logger.error(f"刷新指数[{index_code}]的{time_level}级别BOLL指标数据失败: {str(e)}")
            return []
        
    async def refresh_main_indexes_realtime(self) -> List[IndexRealTimeData]:
        """
        刷新主要指数的实时数据
        
        Returns:
            List[IndexRealTimeData]: 刷新后的实时数据列表
        """
        logger.info("刷新主要指数的实时数据")
        main_indexes = ['000001', '399001', '399006', '000016', '000300', '000905', '000852']
        results = []
        
        for index_code in main_indexes:
            try:
                data = await self.refresh_index_realtime_data(index_code)
                if data:
                    results.append(data)
            except Exception as e:
                logger.error(f"刷新指数[{index_code}]实时数据出错: {str(e)}")
        
        logger.info(f"刷新主要指数实时数据完成，共{len(results)}条")
        return results

    async def refresh_main_indexes_time_series(self, period: str) -> Dict[str, List[IndexTimeSeriesData]]:
        """
        刷新主要指数的时间序列数据
        
        Args:
            period: 时间周期，如"5"、"15"、"30"、"60"、"Day"、"Week"、"Month"
        
        Returns:
            Dict[str, List[IndexTimeSeriesData]]: 指数代码到时间序列数据的映射
        """
        logger.info(f"刷新主要指数的{period}周期时间序列数据")
        main_indexes = ['000001', '399001', '399006', '000016', '000300', '000905', '000852']
        results = {}
        
        for index_code in main_indexes:
            try:
                data = await self.refresh_time_series_data(index_code, period)
                if data:
                    results[index_code] = data
            except Exception as e:
                logger.error(f"刷新指数[{index_code}]的{period}周期时间序列数据出错: {str(e)}")
        
        logger.info(f"刷新主要指数{period}周期时间序列数据完成，共{len(results)}个指数")
        return results

    async def refresh_main_indexes_technical_indicators(self, period: str) -> Dict[str, Dict[str, List]]:
        """
        刷新主要指数的技术指标数据
        
        Args:
            period: 时间周期，如"Day"、"Week"、"Month"
        
        Returns:
            Dict[str, Dict[str, List]]: 指数代码到技术指标数据的映射，格式为:
                                    {index_code: {'kdj': [...], 'macd': [...], 'ma': [...], 'boll': [...]}}
        """
        logger.info(f"刷新主要指数的{period}周期技术指标数据")
        main_indexes = ['000001', '399001', '399006', '000016', '000300', '000905', '000852']
        results = {}
        
        indicators = {
            'kdj': self.refresh_kdj_data,
            'macd': self.refresh_macd_data,
            'ma': self.refresh_ma_data,
            'boll': self.refresh_boll_data
        }
        
        for index_code in main_indexes:
            results[index_code] = {}
            for indicator_name, refresh_method in indicators.items():
                try:
                    data = await refresh_method(index_code, period)
                    if data:
                        results[index_code][indicator_name] = data
                except Exception as e:
                    logger.error(f"刷新指数[{index_code}]的{period}周期{indicator_name}指标出错: {str(e)}")
        
        logger.info(f"刷新主要指数{period}周期技术指标数据完成，共{len(results)}个指数")
        return results
