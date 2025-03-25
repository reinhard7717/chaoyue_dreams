# dao/stock_index_dao.py

import logging
import datetime
import asyncio
from typing import Dict, Any, List, Optional, Union
from django.db import transaction
from django.core.cache import cache
from django.utils import timezone
from asgiref.sync import sync_to_async

from api_manager.apis.index_api import StockIndexAPI
from api_manager.mappings.index_mapping import BOLL_MAPPING, INDEX_REALTIME_DATA_MAPPING, KDJ_MAPPING, MA_MAPPING, MACD_MAPPING, MARKET_OVERVIEW_MAPPING, TIME_SERIES_MAPPING
from models.index import IndexBOLLData, IndexInfo, IndexKDJData, IndexMACDData, IndexMAData, IndexRealTimeData, IndexTimeSeriesData, MarketOverview


logger = logging.getLogger(__name__)

class StockIndexDAO:
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
        }
    
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
            
            # 去重
            unique_indexes = {}
            for index in all_indexes:
                unique_indexes[index['dm']] = index
            
            # 批量保存到数据库
            @transaction.atomic
            def save_indexes():
                for index_data in unique_indexes.values():
                    index, created = IndexInfo.objects.update_or_create(
                        code=index_data['dm'],
                        defaults={
                            'name': index_data['mc'],
                            'exchange': index_data['jys']
                        }
                    )
            
            await sync_to_async(save_indexes)()
            logger.info(f"成功保存{len(unique_indexes)}条指数数据")
            
            # 更新缓存
            cache.delete('stock_indexes_all')
        except Exception as e:
            logger.error(f"刷新指数数据失败: {str(e)}")
            raise
    
    async def _fetch_and_save_realtime_data(self, index_code: str) -> Optional[IndexRealTimeData]:
        """
        获取并保存指数实时数据
        
        Args:
            index_code: 指数代码
            
        Returns:
            Optional[StockIndexRealTimeData]: 保存的实时数据对象
        """
        try:
            # 获取指数对象
            index = await self.get_index_by_code(index_code)
            if not index:
                logger.warning(f"指数[{index_code}]不存在，无法获取实时数据")
                return None
            
            # 从API获取实时数据
            api_data = await self.api.get_index_realtime_data(index_code)
            
            # 将API数据转换为模型数据
            model_data = {}
            for api_field, model_field in INDEX_REALTIME_DATA_MAPPING.items():
                if api_field in api_data:
                    model_data[model_field] = api_data[api_field]
            
            # 保存到数据库
            @transaction.atomic
            def save_data():
                data = IndexRealTimeData(index=index, **model_data)
                data.save()
                return data
            
            data = await sync_to_async(save_data)()
            logger.info(f"成功保存指数[{index_code}]实时数据")
            
            # 更新缓存
            cache_key = f'index_realtime_{index_code}'
            cache.delete(cache_key)
            
            return data
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
            # 从API获取市场概览数据
            api_data = await self.api.get_market_overview()
            
            # 将API数据转换为模型数据
            model_data = {}
            for api_field, model_field in MARKET_OVERVIEW_MAPPING.items():
                if api_field in api_data:
                    model_data[model_field] = api_data[api_field]
            
            # 添加更新时间
            model_data['update_time'] = timezone.now()
            
            # 保存到数据库
            @transaction.atomic
            def save_data():
                data = MarketOverview(**model_data)
                data.save()
                return data
            
            data = await sync_to_async(save_data)()
            logger.info("成功保存市场概览数据")
            
            # 更新缓存
            cache.delete('market_overview_latest')
            
            return data
        except Exception as e:
            logger.error(f"获取并保存市场概览数据失败: {str(e)}")
            return None
    
    async def _fetch_and_save_time_series(self, index_code: str, time_level: str) -> List[IndexTimeSeriesData]:
        """
        获取并保存指数时间序列数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            
        Returns:
            List[TimeSeriesData]: 保存的时间序列数据对象列表
        """
        try:
            # 获取指数对象
            index = await self.get_index_by_code(index_code)
            if not index:
                logger.warning(f"指数[{index_code}]不存在，无法获取时间序列数据")
                return []
            
            # 获取最新数据
            latest_data = await self.api.get_latest_time_series(index_code, time_level)
            if not latest_data:
                logger.warning(f"获取指数[{index_code}]的{time_level}级别最新分时数据失败")
                return []
            
            # 获取历史数据
            history_data = await self.api.get_history_time_series(index_code, time_level)
            if not history_data:
                logger.warning(f"获取指数[{index_code}]的{time_level}级别历史分时数据失败")
                return []
            
            # 合并最新和历史数据，确保不重复
            all_data = [latest_data]
            
            # 以交易时间为键进行去重
            time_key = 'd'  # API返回的交易时间字段
            existing_times = {latest_data.get(time_key)}
            
            for item in history_data:
                if item.get(time_key) not in existing_times:
                    all_data.append(item)
                    existing_times.add(item.get(time_key))
            
            # 将API数据转换为模型数据并保存
            saved_data = []
            
            @transaction.atomic
            def save_data():
                saved_items = []
                for api_item in all_data:
                    model_data = {'index': index, 'time_level': time_level}
                    
                    for api_field, model_field in TIME_SERIES_MAPPING.items():
                        if api_field in api_item:
                            model_data[model_field] = api_item[api_field]
                    
                    # 使用update_or_create避免重复数据
                    data, created = IndexTimeSeriesData.objects.update_or_create(
                        index=index,
                        time_level=time_level,
                        trade_time=model_data['trade_time'],
                        defaults=model_data
                    )
                    saved_items.append(data)
                
                return saved_items
            
            saved_data = await sync_to_async(save_data)()
            logger.info(f"成功保存指数[{index_code}]的{time_level}级别时间序列数据，共{len(saved_data)}条")
            
            # 更新缓存
            cache_keys = [key for key in cache._cache.keys() if f'time_series_{index_code}_{time_level}' in key]
            for key in cache_keys:
                cache.delete(key)
            
            return saved_data
        except Exception as e:
            logger.error(f"获取并保存指数[{index_code}]的{time_level}级别时间序列数据失败: {str(e)}")
            return []
    
    # ================ 公共方法 ================
    
    async def refresh_all_indexes(self) -> bool:
        """
        刷新所有指数列表
        
        Returns:
            bool: 是否成功
        """
        try:
            await self._refresh_indexes()
            return True
        except Exception as e:
            logger.error(f"刷新指数列表失败: {str(e)}")
            return False
    
    async def refresh_index_realtime_data(self, index_code: str) -> Optional[IndexRealTimeData]:
        """
        刷新指数实时数据
        
        Args:
            index_code: 指数代码
            
        Returns:
            Optional[StockIndexRealTimeData]: 更新后的实时数据
        """
        try:
            return await self._fetch_and_save_realtime_data(index_code)
        except Exception as e:
            logger.error(f"刷新指数[{index_code}]实时数据失败: {str(e)}")
            return None
    
    async def refresh_market_overview(self) -> Optional[MarketOverview]:
        """
        刷新市场概览数据
        
        Returns:
            Optional[MarketOverview]: 更新后的市场概览数据
        """
        try:
            return await self._fetch_and_save_market_overview()
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
            List[TimeSeriesData]: 更新后的时间序列数据
        """
        try:
            return await self._fetch_and_save_time_series(index_code, time_level)
        except Exception as e:
            logger.error(f"刷新指数[{index_code}]的{time_level}级别时间序列数据失败: {str(e)}")
            return []
       
    # ================ 技术指标读取方法 ================
        
    async def get_kdj_data(self, index_code: str, time_level: str, 
                        start_time: Optional[datetime.datetime] = None,
                        end_time: Optional[datetime.datetime] = None,
                        limit: int = 100) -> List[IndexKDJData]:
        """
        获取指数的KDJ指标数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            start_time: 开始时间
            end_time: 结束时间
            limit: 返回数据条数限制
            
        Returns:
            List[KDJData]: KDJ指标数据列表
        """
        cache_key = f'kdj_data_{index_code}_{time_level}_{start_time}_{end_time}_{limit}'
        
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
                IndexKDJData.objects.filter(**query).order_by('-trade_time')[:limit]
            )
            
            # 如果没有数据或数据很少，从API获取
            if len(data) < 10:
                logger.info(f"指数[{index_code}]的{time_level}级别KDJ数据不足，从API获取")
                await self._fetch_and_save_kdj(index_code, time_level)
                data = await sync_to_async(list)(
                    IndexKDJData.objects.filter(**query).order_by('-trade_time')[:limit]
                )
            
            cache.set(cache_key, data, self.cache_timeout['time_series'])
            return data
        except Exception as e:
            logger.error(f"获取指数[{index_code}]的{time_level}级别KDJ数据失败: {str(e)}")
            return []

    async def get_macd_data(self, index_code: str, time_level: str, 
                            start_time: Optional[datetime.datetime] = None,
                            end_time: Optional[datetime.datetime] = None,
                            limit: int = 100) -> List[IndexMACDData]:
        """
        获取指数的MACD指标数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            start_time: 开始时间
            end_time: 结束时间
            limit: 返回数据条数限制
            
        Returns:
            List[MACDData]: MACD指标数据列表
        """
        cache_key = f'macd_data_{index_code}_{time_level}_{start_time}_{end_time}_{limit}'
        
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
                IndexMACDData.objects.filter(**query).order_by('-trade_time')[:limit]
            )
            
            # 如果没有数据或数据很少，从API获取
            if len(data) < 10:
                logger.info(f"指数[{index_code}]的{time_level}级别MACD数据不足，从API获取")
                await self._fetch_and_save_macd(index_code, time_level)
                data = await sync_to_async(list)(
                    IndexMACDData.objects.filter(**query).order_by('-trade_time')[:limit]
                )
            
            cache.set(cache_key, data, self.cache_timeout['time_series'])
            return data
        except Exception as e:
            logger.error(f"获取指数[{index_code}]的{time_level}级别MACD数据失败: {str(e)}")
            return []

    async def get_ma_data(self, index_code: str, time_level: str, 
                        start_time: Optional[datetime.datetime] = None,
                        end_time: Optional[datetime.datetime] = None,
                        limit: int = 100) -> List[IndexMAData]:
        """
        获取指数的MA指标数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            start_time: 开始时间
            end_time: 结束时间
            limit: 返回数据条数限制
            
        Returns:
            List[MAData]: MA指标数据列表
        """
        cache_key = f'ma_data_{index_code}_{time_level}_{start_time}_{end_time}_{limit}'
        
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
                IndexMAData.objects.filter(**query).order_by('-trade_time')[:limit]
            )
            
            # 如果没有数据或数据很少，从API获取
            if len(data) < 10:
                logger.info(f"指数[{index_code}]的{time_level}级别MA数据不足，从API获取")
                await self._fetch_and_save_ma(index_code, time_level)
                data = await sync_to_async(list)(
                    IndexMAData.objects.filter(**query).order_by('-trade_time')[:limit]
                )
            
            cache.set(cache_key, data, self.cache_timeout['time_series'])
            return data
        except Exception as e:
            logger.error(f"获取指数[{index_code}]的{time_level}级别MA数据失败: {str(e)}")
            return []

    async def get_boll_data(self, index_code: str, time_level: str, 
                        start_time: Optional[datetime.datetime] = None,
                        end_time: Optional[datetime.datetime] = None,
                        limit: int = 100) -> List[IndexBOLLData]:
        """
        获取指数的BOLL指标数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            start_time: 开始时间
            end_time: 结束时间
            limit: 返回数据条数限制
            
        Returns:
            List[BOLLData]: BOLL指标数据列表
        """
        cache_key = f'boll_data_{index_code}_{time_level}_{start_time}_{end_time}_{limit}'
        
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
                IndexBOLLData.objects.filter(**query).order_by('-trade_time')[:limit]
            )
            
            # 如果没有数据或数据很少，从API获取
            if len(data) < 10:
                logger.info(f"指数[{index_code}]的{time_level}级别BOLL数据不足，从API获取")
                await self._fetch_and_save_boll(index_code, time_level)
                data = await sync_to_async(list)(
                    IndexBOLLData.objects.filter(**query).order_by('-trade_time')[:limit]
                )
            
            cache.set(cache_key, data, self.cache_timeout['time_series'])
            return data
        except Exception as e:
            logger.error(f"获取指数[{index_code}]的{time_level}级别BOLL数据失败: {str(e)}")
            return []

    # ================ 技术指标写入方法 ================

    async def _fetch_and_save_kdj(self, index_code: str, time_level: str) -> List[IndexKDJData]:
        """
        获取并保存指数KDJ指标数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            
        Returns:
            List[KDJData]: 保存的KDJ指标数据列表
        """
        try:
            # 获取指数对象
            index = await self.get_index_by_code(index_code)
            if not index:
                logger.warning(f"指数[{index_code}]不存在，无法获取KDJ数据")
                return []
            
            # 获取最新数据
            latest_data = await self.api.get_latest_kdj(index_code, time_level)
            if not latest_data:
                logger.warning(f"获取指数[{index_code}]的{time_level}级别最新KDJ数据失败")
                return []
            
            # 获取历史数据
            history_data = await self.api.get_history_kdj(index_code, time_level)
            if not history_data:
                logger.warning(f"获取指数[{index_code}]的{time_level}级别历史KDJ数据失败")
                return []
            
            # 合并最新和历史数据，确保不重复
            all_data = [latest_data]
            
            # 以交易时间为键进行去重
            time_key = 't'  # API返回的交易时间字段
            existing_times = {latest_data.get(time_key)}
            
            for item in history_data:
                if item.get(time_key) not in existing_times:
                    all_data.append(item)
                    existing_times.add(item.get(time_key))
            
            # 将API数据转换为模型数据并保存
            saved_data = []
            
            @transaction.atomic
            def save_data():
                saved_items = []
                for api_item in all_data:
                    model_data = {'index': index, 'time_level': time_level}
                    
                    for api_field, model_field in KDJ_MAPPING.items():
                        if api_field in api_item:
                            model_data[model_field] = api_item[api_field]
                    
                    # 使用update_or_create避免重复数据
                    data, created = IndexKDJData.objects.update_or_create(
                        index=index,
                        time_level=time_level,
                        trade_time=model_data['trade_time'],
                        defaults=model_data
                    )
                    saved_items.append(data)
                
                return saved_items
            
            saved_data = await sync_to_async(save_data)()
            logger.info(f"成功保存指数[{index_code}]的{time_level}级别KDJ数据，共{len(saved_data)}条")
            
            # 更新缓存
            cache_keys = [key for key in cache._cache.keys() if f'kdj_data_{index_code}_{time_level}' in key]
            for key in cache_keys:
                cache.delete(key)
            
            return saved_data
        except Exception as e:
            logger.error(f"获取并保存指数[{index_code}]的{time_level}级别KDJ数据失败: {str(e)}")
            return []

    async def _fetch_and_save_macd(self, index_code: str, time_level: str) -> List[IndexMACDData]:
        """
        获取并保存MACD指标数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            
        Returns:
            List[MACDIndicator]: 保存的MACD指标数据列表
        """
        try:
            # 获取指数对象
            index = await self.get_index_by_code(index_code)
            if not index:
                logger.warning(f"指数[{index_code}]不存在，无法获取MACD数据")
                return []
            
            # 获取最新数据
            latest_data = await self.api.get_latest_macd(index_code, time_level)
            if not latest_data:
                logger.warning(f"获取指数[{index_code}]的{time_level}级别最新MACD数据失败")
                return []
            
            # 获取历史数据
            history_data = await self.api.get_history_macd(index_code, time_level)
            if not history_data:
                logger.warning(f"获取指数[{index_code}]的{time_level}级别历史MACD数据失败")
                return []
            
            # 合并最新和历史数据，确保不重复
            all_data = [latest_data]
            
            # 以交易时间为键进行去重
            time_key = 't'  # API返回的交易时间字段
            existing_times = {latest_data.get(time_key)}
            
            for item in history_data:
                if item.get(time_key) not in existing_times:
                    all_data.append(item)
                    existing_times.add(item.get(time_key))
            
            # 将API数据转换为模型数据并保存
            saved_data = []
            
            @transaction.atomic
            def save_data():
                saved_items = []
                for api_item in all_data:
                    model_data = {'index': index, 'time_level': time_level}
                    
                    for api_field, model_field in MACD_MAPPING.items():
                        if api_field in api_item:
                            model_data[model_field] = api_item[api_field]
                    
                    # 使用update_or_create避免重复数据
                    data, created = IndexMACDData.objects.update_or_create(
                        index=index,
                        time_level=time_level,
                        trade_time=model_data['trade_time'],
                        defaults=model_data
                    )
                    saved_items.append(data)
                
                return saved_items
            
            saved_data = await sync_to_async(save_data)()
            logger.info(f"成功保存指数[{index_code}]的{time_level}级别MACD数据，共{len(saved_data)}条")
            
            # 更新缓存
            cache_keys = [key for key in cache._cache.keys() if f'macd_{index_code}_{time_level}' in key]
            for key in cache_keys:
                cache.delete(key)
            
            return saved_data
        except Exception as e:
            logger.error(f"获取并保存指数[{index_code}]的{time_level}级别MACD数据失败: {str(e)}")
            return []

    async def _fetch_and_save_ma(self, index_code: str, time_level: str) -> List[IndexMAData]:
        """
        获取并保存MA指标数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            
        Returns:
            List[MAIndicator]: 保存的MA指标数据列表
        """
        try:
            # 获取指数对象
            index = await self.get_index_by_code(index_code)
            if not index:
                logger.warning(f"指数[{index_code}]不存在，无法获取MA数据")
                return []
            
            # 获取最新数据
            latest_data = await self.api.get_latest_ma(index_code, time_level)
            if not latest_data:
                logger.warning(f"获取指数[{index_code}]的{time_level}级别最新MA数据失败")
                return []
            
            # 获取历史数据
            history_data = await self.api.get_history_ma(index_code, time_level)
            if not history_data:
                logger.warning(f"获取指数[{index_code}]的{time_level}级别历史MA数据失败")
                return []
            
            # 合并最新和历史数据，确保不重复
            all_data = [latest_data]
            
            # 以交易时间为键进行去重
            time_key = 't'  # API返回的交易时间字段
            existing_times = {latest_data.get(time_key)}
            
            for item in history_data:
                if item.get(time_key) not in existing_times:
                    all_data.append(item)
                    existing_times.add(item.get(time_key))
            
            # 将API数据转换为模型数据并保存
            saved_data = []
            
            @transaction.atomic
            def save_data():
                saved_items = []
                for api_item in all_data:
                    model_data = {'index': index, 'time_level': time_level}
                    
                    for api_field, model_field in MA_MAPPING.items():
                        if api_field in api_item:
                            model_data[model_field] = api_item[api_field]
                    
                    # 使用update_or_create避免重复数据
                    data, created = IndexMAData.objects.update_or_create(
                        index=index,
                        time_level=time_level,
                        trade_time=model_data['trade_time'],
                        defaults=model_data
                    )
                    saved_items.append(data)
                
                return saved_items
            
            saved_data = await sync_to_async(save_data)()
            logger.info(f"成功保存指数[{index_code}]的{time_level}级别MA数据，共{len(saved_data)}条")
            
            # 更新缓存
            cache_keys = [key for key in cache._cache.keys() if f'ma_{index_code}_{time_level}' in key]
            for key in cache_keys:
                cache.delete(key)
            
            return saved_data
        except Exception as e:
            logger.error(f"获取并保存指数[{index_code}]的{time_level}级别MA数据失败: {str(e)}")
            return []

    async def _fetch_and_save_boll(self, index_code: str, time_level: str) -> List[IndexBOLLData]:
        """
        获取并保存BOLL指标数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            
        Returns:
            List[BOLLIndicator]: 保存的BOLL指标数据列表
        """
        try:
            # 获取指数对象
            index = await self.get_index_by_code(index_code)
            if not index:
                logger.warning(f"指数[{index_code}]不存在，无法获取BOLL数据")
                return []
            
            # 获取最新数据
            latest_data = await self.api.get_latest_boll(index_code, time_level)
            if not latest_data:
                logger.warning(f"获取指数[{index_code}]的{time_level}级别最新BOLL数据失败")
                return []
            
            # 获取历史数据
            history_data = await self.api.get_history_boll(index_code, time_level)
            if not history_data:
                logger.warning(f"获取指数[{index_code}]的{time_level}级别历史BOLL数据失败")
                return []
            
            # 合并最新和历史数据，确保不重复
            all_data = [latest_data]
            
            # 以交易时间为键进行去重
            time_key = 't'  # API返回的交易时间字段
            existing_times = {latest_data.get(time_key)}
            
            for item in history_data:
                if item.get(time_key) not in existing_times:
                    all_data.append(item)
                    existing_times.add(item.get(time_key))
            
            # 将API数据转换为模型数据并保存
            saved_data = []
            
            @transaction.atomic
            def save_data():
                saved_items = []
                for api_item in all_data:
                    model_data = {'index': index, 'time_level': time_level}
                    
                    for api_field, model_field in BOLL_MAPPING.items():
                        if api_field in api_item:
                            model_data[model_field] = api_item[api_field]
                    
                    # 使用update_or_create避免重复数据
                    data, created = IndexBOLLData.objects.update_or_create(
                        index=index,
                        time_level=time_level,
                        trade_time=model_data['trade_time'],
                        defaults=model_data
                    )
                    saved_items.append(data)
                
                return saved_items
            
            saved_data = await sync_to_async(save_data)()
            logger.info(f"成功保存指数[{index_code}]的{time_level}级别BOLL数据，共{len(saved_data)}条")
            
            # 更新缓存
            cache_keys = [key for key in cache._cache.keys() if f'boll_{index_code}_{time_level}' in key]
            for key in cache_keys:
                cache.delete(key)
            
            return saved_data
        except Exception as e:
            logger.error(f"获取并保存指数[{index_code}]的{time_level}级别BOLL数据失败: {str(e)}")
            return []

    # ================ 刷新技术指标数据的公共方法 ================

    async def refresh_kdj_data(self, index_code: str, time_level: str) -> List[IndexKDJData]:
        """
        刷新KDJ指标数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            
        Returns:
            List[KDJIndicator]: 更新后的KDJ指标数据
        """
        try:
            return await self._fetch_and_save_kdj(index_code, time_level)
        except Exception as e:
            logger.error(f"刷新指数[{index_code}]的{time_level}级别KDJ数据失败: {str(e)}")
            return []

    async def refresh_macd_data(self, index_code: str, time_level: str) -> List[IndexMACDData]:
        """
        刷新MACD指标数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            
        Returns:
            List[MACDIndicator]: 更新后的MACD指标数据
        """
        try:
            return await self._fetch_and_save_macd(index_code, time_level)
        except Exception as e:
            logger.error(f"刷新指数[{index_code}]的{time_level}级别MACD数据失败: {str(e)}")
            return []

    async def refresh_ma_data(self, index_code: str, time_level: str) -> List[IndexMAData]:
        """
        刷新MA指标数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            
        Returns:
            List[MAIndicator]: 更新后的MA指标数据
        """
        try:
            return await self._fetch_and_save_ma(index_code, time_level)
        except Exception as e:
            logger.error(f"刷新指数[{index_code}]的{time_level}级别MA数据失败: {str(e)}")
            return []

    async def refresh_boll_data(self, index_code: str, time_level: str) -> List[IndexBOLLData]:
        """
        刷新BOLL指标数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            
        Returns:
            List[BOLLIndicator]: 更新后的BOLL指标数据
        """
        try:
            return await self._fetch_and_save_boll(index_code, time_level)
        except Exception as e:
            logger.error(f"刷新指数[{index_code}]的{time_level}级别BOLL数据失败: {str(e)}")
            return []
