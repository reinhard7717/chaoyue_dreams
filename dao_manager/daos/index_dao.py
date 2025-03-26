# dao/stock_index_dao.py

import logging
import datetime
import asyncio
from typing import Dict, Any, List, Optional, Union, Type
from django.db import transaction
from django.core.cache import cache
from django.utils import timezone
from asgiref.sync import sync_to_async

from dao_manager.base_dao import BaseDAO
from api_manager.apis.index_api import StockIndexAPI
from api_manager.mappings.index_mapping import BOLL_MAPPING, INDEX_REALTIME_DATA_MAPPING, KDJ_MAPPING, MA_MAPPING, MACD_MAPPING, MARKET_OVERVIEW_MAPPING, TIME_SERIES_MAPPING
from stock_models.index import *


logger = logging.getLogger(__name__)

class StockIndexDAO(BaseDAO):
    """
    股票指数数据访问对象
    
    负责股票指数相关数据的读写操作，实现三层数据访问结构：API、Redis缓存、MySQL持久化
    """
    
    def __init__(self):
        """初始化DAO对象，创建API实例"""
        self.api = StockIndexAPI()
        # 设置缓存过期时间（秒）
        self.cache_timeout = {
            'index_list': 86400,  # 指数列表缓存1天
            'realtime_data': 60,  # 实时数据缓存1分钟
            'market_overview': 120,  # 市场概览缓存2分钟
            'time_series': 300,  # 分时数据缓存5分钟
            'technical_indicators': 300,  # 技术指标缓存5分钟
        }
        super().__init__(IndexInfo, self.api, self.cache_timeout['index_list'])
    
    # ================ 读取方法 ================
    
    async def get_all_indexes(self) -> List[IndexInfo]:
        """
        获取所有股票指数列表
        
        先尝试从缓存获取，如缓存未命中则从数据库读取，
        如数据库无数据则从API获取并保存
        
        Returns:
            List[StockIndex]: 指数对象列表
        """
        cache_key = 'stock_indexes_all'
        
        # 尝试从缓存获取
        cached_data = cache.get(cache_key)
        if cached_data:
            logger.debug("从缓存获取股票指数列表")
            return cached_data
        
        # 从数据库读取
        indexes = await sync_to_async(list)(IndexInfo.objects.all())
        
        # 如果数据库中有数据，缓存并返回
        if indexes:
            logger.debug(f"从数据库获取股票指数列表，共{len(indexes)}条")
            cache.set(cache_key, indexes, self.cache_timeout['index_list'])
            return indexes
        
        # 如果数据库中没有数据，从API获取并保存
        logger.info("数据库中没有指数数据，从API获取")
        await self._refresh_indexes()
        indexes = await sync_to_async(list)(IndexInfo.objects.all())
        cache.set(cache_key, indexes, self.cache_timeout['index_list'])
        return indexes
    
    async def get_index_by_code(self, code: str) -> Optional[IndexInfo]:
        """
        根据指数代码获取指数
        
        Args:
            code: 指数代码
            
        Returns:
            Optional[StockIndex]: 指数对象，如不存在返回None
        """
        cache_key = f'stock_index_{code}'
        
        # 尝试从缓存获取
        cached_data = cache.get(cache_key)
        if cached_data:
            return cached_data
        
        # 从数据库获取
        try:
            index = await sync_to_async(IndexInfo.objects.get)(code=code)
            cache.set(cache_key, index, self.cache_timeout['index_list'])
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
        cache_key = f'index_realtime_{index_code}'
        
        # 尝试从缓存获取
        cached_data = cache.get(cache_key)
        if cached_data:
            return cached_data
        
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
                cache.set(cache_key, data, self.cache_timeout['realtime_data'])
                return data
            
            # 数据不存在或已过期，从API获取新数据
            logger.info(f"指数[{index_code}]实时数据不存在或已过期，从API获取")
            data = await self._fetch_and_save_realtime_data(index_code)
            if data:
                cache.set(cache_key, data, self.cache_timeout['realtime_data'])
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
        cache_key = 'market_overview_latest'
        
        # 尝试从缓存获取
        cached_data = cache.get(cache_key)
        if cached_data:
            return cached_data
        
        # 从数据库获取最新数据
        try:
            data = await sync_to_async(
                lambda: MarketOverview.objects.order_by('-update_time').first()
            )()
            
            # 检查数据是否过期（超过2分钟）
            if data and (timezone.now() - data.update_time).total_seconds() < 120:
                cache.set(cache_key, data, self.cache_timeout['market_overview'])
                return data
            
            # 数据不存在或已过期，从API获取新数据
            logger.info("市场概览数据不存在或已过期，从API获取")
            data = await self._fetch_and_save_market_overview()
            if data:
                cache.set(cache_key, data, self.cache_timeout['market_overview'])
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
        cache_key = f'time_series_{index_code}_{time_level}_{start_time}_{end_time}_{limit}'
        
        # 尝试从缓存获取
        cached_data = cache.get(cache_key)
        if cached_data:
            return cached_data
        
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
            
            cache.set(cache_key, data, self.cache_timeout['time_series'])
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
        cache_key = f'{indicator_name}_{index_code}_{time_level}_{start_time}_{end_time}_{limit}'
        
        # 尝试从缓存获取
        cached_data = cache.get(cache_key)
        if cached_data:
            return cached_data
        
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
            
            cache.set(cache_key, data, self.cache_timeout['technical_indicators'])
            return data
        except Exception as e:
            logger.error(f"获取指数[{index_code}]的{time_level}级别{indicator_name}数据失败: {str(e)}")
            return []
    
    # ================ 写入方法 ================
    
    async def _refresh_indexes(self) -> None:
        """
        刷新所有指数数据
        
        从API获取指数列表并保存到数据库
        """
        try:
            # 获取沪深主要指数、沪市指数和深市指数
            tasks = [
                self.api.get_main_indexes(),
                self.api.get_sh_indexes(),
                self.api.get_sz_indexes()
            ]
            results = await asyncio.gather(*tasks)
            
            all_indexes = []
            for index_list in results:
                all_indexes.extend(index_list)
            
            # 在数据库事务中保存指数数据
            @transaction.atomic
            def save_indexes():
                for index_data in all_indexes:
                    try:
                        # 将API返回的数据转换为标准字典格式，并处理日期和数字字段
                        data_dict = {
                            'code': index_data.get('code'),
                            'name': index_data.get('name'),
                            'market': index_data.get('market', ''),
                            'publisher': index_data.get('publisher', ''),
                            'category': index_data.get('category', ''),
                            'base_date': self._parse_datetime(index_data.get('base_date')),
                            'base_point': self._parse_number(index_data.get('base_point')),
                        }
                        
                        index, created = IndexInfo.objects.update_or_create(
                            code=data_dict['code'],
                            defaults=data_dict
                        )
                        logger.debug(f"{'创建' if created else '更新'}指数: {index.code} {index.name}")
                    except Exception as e:
                        logger.error(f"保存指数数据失败 {index_data.get('code')}: {str(e)}")
            
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
            index = await self.get_index_by_code(index_code)
            if not index:
                logger.warning(f"指数代码[{index_code}]不存在，无法获取实时数据")
                return None
            
            # 调用API获取实时数据
            api_data = await self.api.get_realtime_data(index_code)
            if not api_data:
                logger.warning(f"API未返回指数[{index_code}]的实时数据")
                return None
            
            # 在数据库事务中保存数据
            @transaction.atomic
            def save_data():
                # 解析API返回的数据
                mapped_data = INDEX_REALTIME_DATA_MAPPING(api_data)
                
                # 将映射后的数据转换为标准字典格式，并处理日期和数字字段
                data_dict = {
                    'index': index,
                    'trade_time': self._parse_datetime(mapped_data.get('trade_time')),
                    'open': self._parse_number(mapped_data.get('open')),
                    'high': self._parse_number(mapped_data.get('high')),
                    'low': self._parse_number(mapped_data.get('low')),
                    'close': self._parse_number(mapped_data.get('close')),
                    'volume': self._parse_number(mapped_data.get('volume')),
                    'amount': self._parse_number(mapped_data.get('amount')),
                    'change': self._parse_number(mapped_data.get('change')),
                    'change_percent': self._parse_number(mapped_data.get('change_percent')),
                    'update_time': self._parse_datetime(mapped_data.get('update_time')),
                }
                
                # 创建实时数据记录
                realtime_data = IndexRealTimeData.objects.create(**data_dict)
                return realtime_data
            
            # 执行保存操作并返回结果
            result = await sync_to_async(save_data)()
            logger.info(f"成功保存指数[{index_code}]的实时数据")
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
                # 解析API返回的数据
                data = MARKET_OVERVIEW_MAPPING(api_data)
                
                # 创建市场概览数据记录
                overview = MarketOverview(**data)
                overview.save()
                return overview
            
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
            
            # 根据时间级别调用不同的API方法
            if time_level in ['5', '15', '30', '60']:
                # 分钟级别数据
                api_data = await self.api.get_minutes_data(index_code, time_level)
            elif time_level == 'Day':
                # 日线数据
                api_data = await self.api.get_daily_data(index_code)
            elif time_level == 'Week':
                # 周线数据
                api_data = await self.api.get_weekly_data(index_code)
            elif time_level == 'Month':
                # 月线数据
                api_data = await self.api.get_monthly_data(index_code)
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
                
                for item in api_data:
                    # 解析API返回的数据
                    mapped_data = TIME_SERIES_MAPPING(item)
                    
                    # 将映射后的数据转换为标准字典格式，并处理日期和数字字段
                    data_dict = {
                        'index': index,
                        'time_level': time_level,
                        'trade_time': self._parse_datetime(mapped_data.get('trade_time')),
                        'open': self._parse_number(mapped_data.get('open')),
                        'high': self._parse_number(mapped_data.get('high')),
                        'low': self._parse_number(mapped_data.get('low')),
                        'close': self._parse_number(mapped_data.get('close')),
                        'volume': self._parse_number(mapped_data.get('volume')),
                        'amount': self._parse_number(mapped_data.get('amount')),
                        'change': self._parse_number(mapped_data.get('change')),
                        'change_percent': self._parse_number(mapped_data.get('change_percent')),
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
            await self._refresh_indexes()
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
            # 获取指数信息
            index = await self.get_index_by_code(index_code)
            if not index:
                logger.warning(f"指数代码[{index_code}]不存在，无法获取KDJ指标数据")
                return []
            
            # 调用API获取KDJ指标数据
            api_data = await self.api.get_kdj_data(index_code, time_level)
            if not api_data:
                logger.warning(f"API未返回指数[{index_code}]的{time_level}级别KDJ指标数据")
                return []
            
            # 在数据库事务中保存数据
            @transaction.atomic
            def save_data():
                saved_items = []
                
                for item in api_data:
                    # 解析API返回的数据
                    data = KDJ_MAPPING(item)
                    
                    # 尝试查找现有记录
                    try:
                        # 使用 index + time_level + trade_time 作为唯一约束
                        kdj_data, created = IndexKDJData.objects.update_or_create(
                            index=index,
                            time_level=time_level,
                            trade_time=data['trade_time'],
                            defaults={
                                'k_value': data['k_value'],
                                'd_value': data['d_value'],
                                'j_value': data['j_value']
                            }
                        )
                        saved_items.append(kdj_data)
                    except Exception as e:
                        logger.error(f"保存KDJ指标数据失败: {str(e)}")
                
                return saved_items
            
            # 执行保存操作并返回结果
            result = await sync_to_async(save_data)()
            logger.info(f"成功保存指数[{index_code}]的{time_level}级别KDJ指标数据，共{len(result)}条")
            return result
        except Exception as e:
            logger.error(f"获取并保存指数[{index_code}]的{time_level}级别KDJ指标数据失败: {str(e)}")
            return []
    
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
            # 获取指数信息
            index = await self.get_index_by_code(index_code)
            if not index:
                logger.warning(f"指数代码[{index_code}]不存在，无法获取MACD指标数据")
                return []
            
            # 调用API获取MACD指标数据
            api_data = await self.api.get_macd_data(index_code, time_level)
            if not api_data:
                logger.warning(f"API未返回指数[{index_code}]的{time_level}级别MACD指标数据")
                return []
            
            # 在数据库事务中保存数据
            @transaction.atomic
            def save_data():
                saved_items = []
                
                for item in api_data:
                    # 解析API返回的数据
                    mapped_data = MACD_MAPPING(item)
                    
                    # 将映射后的数据转换为标准字典格式，并处理日期和数字字段
                    data_dict = {
                        'index': index,
                        'time_level': time_level,
                        'trade_time': self._parse_datetime(mapped_data.get('trade_time')),
                        'dif': self._parse_number(mapped_data.get('dif')),
                        'dea': self._parse_number(mapped_data.get('dea')),
                        'macd': self._parse_number(mapped_data.get('macd')),
                    }
                    
                    # 尝试查找现有记录
                    try:
                        # 使用 index + time_level + trade_time 作为唯一约束
                        macd_data, created = IndexMACDData.objects.update_or_create(
                            index=index,
                            time_level=time_level,
                            trade_time=data_dict['trade_time'],
                            defaults=data_dict
                        )
                        saved_items.append(macd_data)
                    except Exception as e:
                        logger.error(f"保存MACD指标数据失败: {str(e)}")
                
                return saved_items
            
            # 执行保存操作并返回结果
            result = await sync_to_async(save_data)()
            logger.info(f"成功保存指数[{index_code}]的{time_level}级别MACD指标数据，共{len(result)}条")
            return result
        except Exception as e:
            logger.error(f"获取并保存指数[{index_code}]的{time_level}级别MACD指标数据失败: {str(e)}")
            return []
    
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
            # 获取指数信息
            index = await self.get_index_by_code(index_code)
            if not index:
                logger.warning(f"指数代码[{index_code}]不存在，无法获取MA指标数据")
                return []
            
            # 调用API获取MA指标数据
            api_data = await self.api.get_ma_data(index_code, time_level)
            if not api_data:
                logger.warning(f"API未返回指数[{index_code}]的{time_level}级别MA指标数据")
                return []
            
            # 在数据库事务中保存数据
            @transaction.atomic
            def save_data():
                saved_items = []
                
                for item in api_data:
                    # 解析API返回的数据
                    data = MA_MAPPING(item)
                    
                    # 尝试查找现有记录
                    try:
                        # 使用 index + time_level + trade_time 作为唯一约束
                        ma_data, created = IndexMAData.objects.update_or_create(
                            index=index,
                            time_level=time_level,
                            trade_time=data['trade_time'],
                            defaults={
                                'ma5': data['ma5'],
                                'ma10': data['ma10'],
                                'ma20': data['ma20'],
                                'ma30': data['ma30'],
                                'ma60': data['ma60']
                            }
                        )
                        saved_items.append(ma_data)
                    except Exception as e:
                        logger.error(f"保存MA指标数据失败: {str(e)}")
                
                return saved_items
            
            # 执行保存操作并返回结果
            result = await sync_to_async(save_data)()
            logger.info(f"成功保存指数[{index_code}]的{time_level}级别MA指标数据，共{len(result)}条")
            return result
        except Exception as e:
            logger.error(f"获取并保存指数[{index_code}]的{time_level}级别MA指标数据失败: {str(e)}")
            return []
    
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
            # 获取指数信息
            index = await self.get_index_by_code(index_code)
            if not index:
                logger.warning(f"指数代码[{index_code}]不存在，无法获取BOLL指标数据")
                return []
            
            # 调用API获取BOLL指标数据
            api_data = await self.api.get_boll_data(index_code, time_level)
            if not api_data:
                logger.warning(f"API未返回指数[{index_code}]的{time_level}级别BOLL指标数据")
                return []
            
            # 在数据库事务中保存数据
            @transaction.atomic
            def save_data():
                saved_items = []
                
                for item in api_data:
                    # 解析API返回的数据
                    mapped_data = BOLL_MAPPING(item)
                    
                    # 将映射后的数据转换为标准字典格式，并处理日期和数字字段
                    data_dict = {
                        'index': index,
                        'time_level': time_level,
                        'trade_time': self._parse_datetime(mapped_data.get('trade_time')),
                        'upper': self._parse_number(mapped_data.get('upper')),
                        'mid': self._parse_number(mapped_data.get('mid')),
                        'lower': self._parse_number(mapped_data.get('lower')),
                    }
                    
                    # 尝试查找现有记录
                    try:
                        # 使用 index + time_level + trade_time 作为唯一约束
                        boll_data, created = IndexBOLLData.objects.update_or_create(
                            index=index,
                            time_level=time_level,
                            trade_time=data_dict['trade_time'],
                            defaults=data_dict
                        )
                        saved_items.append(boll_data)
                    except Exception as e:
                        logger.error(f"保存BOLL指标数据失败: {str(e)}")
                
                return saved_items
            
            # 执行保存操作并返回结果
            result = await sync_to_async(save_data)()
            logger.info(f"成功保存指数[{index_code}]的{time_level}级别BOLL指标数据，共{len(result)}条")
            return result
        except Exception as e:
            logger.error(f"获取并保存指数[{index_code}]的{time_level}级别BOLL指标数据失败: {str(e)}")
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
