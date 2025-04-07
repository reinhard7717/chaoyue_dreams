# dao/stock_index_dao.py

from decimal import Decimal
import json
import logging
import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Set, Union, Type
from django.db import transaction
from django.core.cache import cache
from django.utils import timezone
from asgiref.sync import sync_to_async
from django.conf import settings
from stock_models.indicator.boll import IndexBOLLData
from stock_models.indicator.kdj import IndexKDJData
from stock_models.indicator.ma import IndexMAData
from stock_models.indicator.macd import IndexMACDData
from utils.cache_get import IndexCacheGet
from utils.cache_manager import CacheManager
from utils.cache_set import IndexCacheSet
from utils.cash_key import IndexCashKey
from utils.data_format_process import IndexDataFormatProcess
from utils.models import ModelJSONEncoder
from utils import cache_constants as cc # 导入常量

from dao_manager.base_dao import BaseDAO
from api_manager.apis.index_api import StockIndexAPI
from stock_models.index import *

logger = logging.getLogger("dao")

TIME_LEVELS = ['5', '15', '30', '60', 'Day', 'Week', 'Month', 'Year']

class StockIndexDAO(BaseDAO):
    """
    股票指数数据访问对象
    
    负责股票指数相关数据的读写操作，实现三层数据访问结构：API、Redis缓存、MySQL持久化
    """
    
    def __init__(self):
        """初始化DAO对象，创建API实例"""
        self.api = StockIndexAPI()
        self.cache_manager = CacheManager()
        self.cache_limit = 200 # 定义缓存数量上限
        self.cache_key = IndexCashKey()
        self.data_format_process = IndexDataFormatProcess()
        self.cache_set = IndexCacheSet()
        self.cache_get = IndexCacheGet()

    # 新增 close 方法
    async def close(self):
        """关闭内部持有的 API Client Session"""
        if hasattr(self, 'api') and self.api:
            # logger.debug("Closing StockIndexDAO's internal API client...") # 可选日志
            await self.api.close() # 调用 StockIndexAPI 的 close 方法
            # logger.debug("StockIndexDAO's internal API client closed.") # 可选日志
        else:
            # logger.debug("StockIndexDAO has no API client to close or it's already None.") # 可选日志
            pass

    # ================ 读取方法 ================
    
    async def get_all_indexes(self) -> List[IndexInfo]:
        """
        获取所有股票指数列表，按照指数代码排序
        
        先尝试从缓存获取，如缓存未命中则从数据库读取，
        如数据库无数据则从API获取并保存
        
        Returns:
            List[StockIndex]: 按指数代码排序的指数对象列表
        """
        # 尝试从缓存获取
        # cached_data = await self.cache_get.all_indexes()
        # if cached_data:
        #     logger.debug("从缓存获取股票指数列表")
        #     # logger.info(f"cached_data: {cached_data}")
        #     # 将缓存数据转换为模型实例列表并排序
        #     return sorted([IndexInfo(**index_dict) for index_dict in cached_data], key=lambda x: x.code)
            
        # 从数据库读取并排序
        indexes = await sync_to_async(list)(IndexInfo.objects.all().order_by('code'))
        if indexes:
            await self.cache_set.indexes(indexes)
            return indexes
        
        # 如果数据库中没有数据，从API获取并保存
        logger.info("数据库中没有指数数据，从API获取")
        await self.fetch_and_save_indexes()
        indexes = await sync_to_async(list)(IndexInfo.objects.all().order_by('code'))
        return indexes
    
    async def get_index_by_code(self, code: str) -> Optional[IndexInfo]:
        """
        根据指数代码获取指数
        
        Args:
            code: 指数代码
            
        Returns:
            Optional[StockIndex]: 指数对象，如不存在返回None
        """
        # # 使用CacheManager生成标准化缓存键
        cache_key = self.cache_key.index_data(code)
        # # 尝试从缓存获取，指定模型类进行自动转换
        # index = self.cache_manager.get_model(cache_key, IndexInfo)
        # if index:
        #     # logger.warning(f"get_index_by_code获取指数: {cache_key}, {index}, type: {type(index)}")
        #     return index
            
        # 从数据库获取
        index = await sync_to_async(IndexInfo.objects.get)(code=code)
        # 如果数据库中有数据，缓存并返回
        if index:
            # 序列化对象列表
            cache_data = await self.data_format_process.set_index_data(index)
            # logger.debug(f"从数据库获取股票指数列表，共{len(index)}条")
            
            # *** 正确调用 CacheManager 缓存数据 ***
            success = self.cache_manager.set(
                key=cache_key,          # 第一个参数：缓存键 (字符串)
                data=cache_data,     # 第二个参数：要缓存的数据 (字典)
                timeout=self.cache_manager.get_timeout('st') # 超时时间
            )
            return index
        
        return index
            
    async def get_latest_realtime_data(self, index_code: str) -> Optional[IndexRealTimeData]:
        """
        获取指数最新实时数据
        Args:
            index_code: 指数代码
        Returns:
            Optional[StockIndexRealTimeData]: 实时数据对象，如不存在返回None
        """
        cache_data = await self.cache_get.realtime_data(index_code)
        if cache_data:
            return cache_data
        
        # 从数据库获取最新数据
        index = await self.get_index_by_code(index_code)
        if not index:
            return None
        try:
            data = await sync_to_async(lambda: IndexRealTimeData.objects.filter(index=index).order_by('-trade_time').first())()
            # 检查数据是否过期（超过2分钟）
            if data and (timezone.now() - data.trade_time).total_seconds() < 120:
                await self.cache_set.realtime_data(index_code, data)
                return data
        except Exception as e:
            logger.error(f"从数据库获取最新指数[{index_code}]实时数据失败: {str(e)}")
            return None
        
        # 数据不存在或已过期，从API获取新数据
        logger.info(f"指数[{index_code}]实时数据不存在或已过期，从API获取")
        await self.fetch_and_save_realtime_data(index_code)
        data = await sync_to_async(lambda: IndexRealTimeData.objects.filter(index=index).order_by('-trade_time').first())()
        return data
            
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
                lambda: MarketOverview.objects.order_by('-trade_time').first()
            )()
            
            # 检查数据是否过期（超过2分钟）
            if data and (timezone.now() - data.trade_time).total_seconds() < 120:
                # *** 正确调用 CacheManager 缓存数据 ***
                success = self.cache_manager.set(
                    key=cache_key,          # 第一个参数：缓存键 (字符串)
                    data=data,     # 第二个参数：要缓存的数据 (字典)
                    timeout=self.cache_manager.get_timeout('st') # 超时时间
                )
                return data
            
            # 数据不存在或已过期，从API获取新数据
            logger.info("市场概览数据不存在或已过期，从API获取")
            data = await self.fetch_and_save_market_overview()
            if data:
                # *** 正确调用 CacheManager 缓存数据 ***
                success = self.cache_manager.set(
                    key=cache_key,          # 第一个参数：缓存键 (字符串)
                    data=data,     # 第二个参数：要缓存的数据 (字典)
                    timeout=self.cache_manager.get_timeout('st') # 超时时间
                )
            return data
        except Exception as e:
            logger.error(f"获取市场概览数据失败: {str(e)}")
            return None

    async def get_latest_time_series_data(self, index_code: str, time_level: str) -> Optional[IndexTimeSeriesData]:
        """
        获取最新时间序列数据
        Args:
            index_code: 指数代码
            time_level: 时间级别（5、15、30、60、Day、Week、Month、Year）
        Returns:
            Optional[IndexTimeSeriesData]: 最新时间序列数据对象，如不存在返回None
        """
        # 从缓存获取
        data = await self.cache_get.latest_time_series(index_code, time_level)
        if data:
            return data
        
        # 从数据库获取
        index = await self.get_index_by_code(index_code)
        if not index:
            return None
        
        try:
            # 从数据库获取最新数据
            data = await sync_to_async(lambda: IndexTimeSeriesData.objects.filter(index=index, time_level=time_level).order_by('-trade_time').first())()
            if data and (timezone.now() - data.trade_time).total_seconds() < 120:
                await self.cache_set.latest_time_series(index_code, time_level, data)
                return data
        except Exception as e:
            logger.error(f"从数据库获取最新指数[{index_code}]的{time_level}级别时间序列数据失败: {str(e)}")
            return None
        
        # 数据不存在或已过期，从API获取新数据
        logger.info(f"指数[{index_code}]的{time_level}级别时间序列数据不存在或已过期，从API获取")
        await self.fetch_and_save_latest_time_series(index_code, time_level)
        data = await sync_to_async(lambda: IndexTimeSeriesData.objects.filter(index=index, time_level=time_level).order_by('-trade_time').first())()
        return data

    async def get_lastest_kdj_data(self, index_code: str, time_level: str) -> Optional[IndexKDJData]:
        """
        获取最新KDJ指标数据
        Args:
            index_code: 指数代码
            time_level: 时间级别
        Returns:
            Optional[IndexKDJData]: 最新KDJ指标数据对象，如不存在返回None
        """
        # 从缓存获取
        data = await self.cache_get.latest_kdj(index_code, time_level)
        if data:
            return data
         # 从数据库获取
        index = await self.get_index_by_code(index_code)
        if not index:
            return None
        try:
            # 从数据库获取最新数据
            data = await sync_to_async(lambda: IndexKDJData.objects.filter(index=index, time_level=time_level).order_by('-trade_time').first())()
            if data and (timezone.now() - data.trade_time).total_seconds() < 120:
                await self.cache_set.latest_kdj(index_code, time_level, data)
                return data
        except Exception as e:
            logger.error(f"从数据库获取最新指数[{index_code}]的{time_level}级别KDJ指标数据失败: {str(e)}")
            return None
        
        # 数据不存在或已过期，从API获取新数据
        logger.info(f"指数[{index_code}]的{time_level}级别KDJ指标数据不存在或已过期，从API获取")
        await self.fetch_and_save_latest_kdj(index_code, time_level)
        data = await sync_to_async(lambda: IndexKDJData.objects.filter(index=index, time_level=time_level).order_by('-trade_time').first())()
        return data

    async def get_lastest_macd_data(self, index_code: str, time_level: str) -> Optional[IndexMAData]:
        """
        获取最新MA指标数据
        Args:
            index_code: 指数代码
            time_level: 时间级别
        Returns:
            Optional[IndexMAData]: 最新MA指标数据对象，如不存在返回None
        """
        # 从缓存获取
        data = await self.cache_get.latest_macd(index_code, time_level)
        if data:
            return data
         # 从数据库获取
        index = await self.get_index_by_code(index_code)
        if not index:
            return None
        try:
            # 从数据库获取最新数据
            data = await sync_to_async(lambda: IndexMACDData.objects.filter(index=index, time_level=time_level).order_by('-trade_time').first())()
            if data and (timezone.now() - data.trade_time).total_seconds() < 120:
                await self.cache_set.latest_macd(index_code, time_level, data)
                return data
        except Exception as e:
            logger.error(f"从数据库获取最新指数[{index_code}]的{time_level}级别MACD指标数据失败: {str(e)}")
            return None
        
        # 数据不存在或已过期，从API获取新数据
        logger.info(f"指数[{index_code}]的{time_level}级别MACD指标数据不存在或已过期，从API获取")
        await self.fetch_and_save_latest_macd(index_code, time_level)
        data = await sync_to_async(lambda: IndexMACDData.objects.filter(index=index, time_level=time_level).order_by('-trade_time').first())()
        return data

    async def get_lastest_ma_data(self, index_code: str, time_level: str) -> Optional[IndexMAData]:
        """
        获取最新MA指标数据
        Args:
            index_code: 指数代码
            time_level: 时间级别
        Returns:
            Optional[IndexMAData]: 最新MA指标数据对象，如不存在返回None
        """
        # 从缓存获取
        data = await self.cache_get.latest_ma(index_code, time_level)
        if data:
            return data
         # 从数据库获取
        index = await self.get_index_by_code(index_code)
        if not index:
            return None
        try:
            # 从数据库获取最新数据
            data = await sync_to_async(lambda: IndexMAData.objects.filter(index=index, time_level=time_level).order_by('-trade_time').first())()
            if data and (timezone.now() - data.trade_time).total_seconds() < 120:
                await self.cache_set.latest_ma(index_code, time_level, data)
                return data
        except Exception as e:
            logger.error(f"从数据库获取最新指数[{index_code}]的{time_level}级别MA指标数据失败: {str(e)}")
            return None
        
        # 数据不存在或已过期，从API获取新数据
        logger.info(f"指数[{index_code}]的{time_level}级别MA指标数据不存在或已过期，从API获取")
        await self.fetch_and_save_latest_ma(index_code, time_level)
        data = await sync_to_async(lambda: IndexMAData.objects.filter(index=index, time_level=time_level).order_by('-trade_time').first())()
        return data

    async def get_lastest_boll_data(self, index_code: str, time_level: str) -> Optional[IndexBOLLData]:
        """
        获取最新BOLL指标数据
        Args:
            index_code: 指数代码
            time_level: 时间级别
        Returns:
            Optional[IndexBOLLData]: 最新BOLL指标数据对象，如不存在返回None
        """
        # 从缓存获取
        data = await self.cache_get.latest_boll(index_code, time_level)
        if data:
            return data
         # 从数据库获取
        index = await self.get_index_by_code(index_code)
        if not index:
            return None
        try:
            # 从数据库获取最新数据
            data = await sync_to_async(lambda: IndexBOLLData.objects.filter(index=index, time_level=time_level).order_by('-trade_time').first())()
            if data and (timezone.now() - data.trade_time).total_seconds() < 120:
                await self.cache_set.latest_boll(index_code, time_level, data)
                return data
        except Exception as e:
            logger.error(f"从数据库获取最新指数[{index_code}]的{time_level}级别BOLL指标数据失败: {str(e)}")
            return None
        
        # 数据不存在或已过期，从API获取新数据
        logger.info(f"指数[{index_code}]的{time_level}级别BOLL指标数据不存在或已过期，从API获取")
        await self.fetch_and_save_latest_boll(index_code, time_level)
        data = await sync_to_async(lambda: IndexBOLLData.objects.filter(index=index, time_level=time_level).order_by('-trade_time').first())()
        return data

    async def get_history_time_series_data(self, index_code: str, time_level: str, 
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
        # 从缓存获取
        data = await self.cache_get.history_time_series(index_code, time_level, start_time, end_time)
        if data:
            return data
        
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
                await self.fetch_and_save_latest_time_series(index_code, time_level)
                data = await sync_to_async(list)(
                    IndexTimeSeriesData.objects.filter(**query).order_by('-trade_time')[:limit]
                )
            return data
        except Exception as e:
            logger.error(f"获取指数[{index_code}]的{time_level}级别时间序列数据失败: {str(e)}")
            return []

    async def get_history_kdj_data(self, index_code: str, time_level: str, 
                                  start_time: Optional[datetime.datetime] = None,
                                  end_time: Optional[datetime.datetime] = None,
                                  limit: int = 100) -> List[IndexTimeSeriesData]:
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
        # 从缓存获取
        data = await self.cache_get.history_kdj(index_code, time_level, start_time, end_time)
        if data:
            return data
        
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
                logger.info(f"指数[{index_code}]的{time_level}级别KDJ指标数据不足，从API获取")
                await self.fetch_and_save_latest_kdj(index_code, time_level)
                data = await sync_to_async(list)(
                    IndexKDJData.objects.filter(**query).order_by('-trade_time')[:limit]
                )
            return data
        except Exception as e:
            logger.error(f"获取指数[{index_code}]的{time_level}级别KDJ指标数据失败: {str(e)}")
            return []
    
    async def get_history_macd_data(self, index_code: str, time_level: str, 
                                  start_time: Optional[datetime.datetime] = None,
                                  end_time: Optional[datetime.datetime] = None,
                                  limit: int = 100) -> List[IndexTimeSeriesData]:
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
        # 从缓存获取
        data = await self.cache_get.history_macd(index_code, time_level, start_time, end_time)
        if data:
            return data
        
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
                logger.info(f"指数[{index_code}]的{time_level}级别MACD指标数据不足，从API获取")
                await self.fetch_and_save_latest_macd(index_code, time_level)
                data = await sync_to_async(list)(
                    IndexMACDData.objects.filter(**query).order_by('-trade_time')[:limit]
                )
            return data
        except Exception as e:
            logger.error(f"获取指数[{index_code}]的{time_level}级别MACD指标数据失败: {str(e)}")
            return []
    
    async def get_history_ma_data(self, index_code: str, time_level: str, 
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
        # 从缓存获取
        data = await self.cache_get.history_ma(index_code, time_level, start_time, end_time)
        if data:
            return data
        
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
                logger.info(f"指数[{index_code}]的{time_level}级别MA指标数据不足，从API获取")
                await self.fetch_and_save_latest_ma(index_code, time_level)
                data = await sync_to_async(list)(
                    IndexMAData.objects.filter(**query).order_by('-trade_time')[:limit]
                )
            return data
        except Exception as e:
            logger.error(f"获取指数[{index_code}]的{time_level}级别MACD指标数据失败: {str(e)}")
            return []
    
    async def get_history_boll_data(self, index_code: str, time_level: str, 
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
        # 从缓存获取
        data = await self.cache_get.history_boll(index_code, time_level, start_time, end_time)
        if data:
            return data
        
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
                logger.info(f"指数[{index_code}]的{time_level}级别MACD指标数据不足，从API获取")
                await self.fetch_and_save_latest_macd(index_code, time_level)
                data = await sync_to_async(list)(
                    IndexMACDData.objects.filter(**query).order_by('-trade_time')[:limit]
                )
            return data
        except Exception as e:
            logger.error(f"获取指数[{index_code}]的{time_level}级别MACD指标数据失败: {str(e)}")
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
                batch_size = 10000
                
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
    async def fetch_and_save_indexes(self) -> Dict:
        """
        刷新所有指数数据
        
        从API获取指数列表并保存到数据库
        """
        try:
            # 获取沪深主要指数、沪市指数和深市指数
            main_index = await self.api.get_main_indexes()
            sh_index = await self.api.get_sh_indexes()
            sz_index = await self.api.get_sz_indexes()
            all_indexes = []
            all_indexes.extend(main_index)
            all_indexes.extend(sh_index)
            all_indexes.extend(sz_index)
            if not all_indexes:
                logger.warning("没有获取到指数数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            data_dicts = []
            for index_data in all_indexes:
                data_dict = await self.data_format_process.set_index_data(index_data)
                data_dicts.append(data_dict)
            # 保存数据
            logger.info(f"开始保存指数数据")
            result = await self._save_all_to_db(
                model_class=IndexInfo,
                data_list=data_dicts,
                unique_fields=['code']
            )
            logger.info(f"指数数据保存完成，结果: {result}")
            # 4. 创建缓存键并保存缓存 (核心修改部分)
            await self.cache_set.indexes(data_dicts)
            return result
        except Exception as e:
            logger.error(f"刷新指数数据失败: {str(e)}")
            raise
    
    async def fetch_and_save_realtime_data_by_index_code(self, index_code: str) -> Dict:
        """
        获取并保存指数实时数据
        Args:
            index_code: 指数代码
        Returns:
            Optional[IndexRealTimeData]: 保存的实时数据对象
        """
        # 获取指数信息
        index = await self.get_index_by_code(index_code)
        logger.info(f"index: {index}")
        # breakpoint()
        if not index:
            logger.warning(f"指数代码[{index_code}]不存在，无法获取实时数据")
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:

            # 调用API获取实时数据
            logger.info(f"开始获取{index_code}指数实时数据")
            data_dicts = []
            api_data = await self.api.get_index_realtime_data(index_code)
            if not api_data:
                logger.warning(f"API未返回指数[{index_code}]的实时数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            data_dict = await self.data_format_process.set_realtime_data(index, api_data)
            data_dicts.append(data_dict)
            cache_dict = data_dict.copy()
            await self.cache_set.realtime_data(index_code, cache_dict)
            # logger.info(f"data_dict: {data_dict}")
            # 保存数据
            logger.info(f"开始保存{index_code}指数实时数据")
            result = await self._save_all_to_db(
                model_class=IndexRealTimeData,
                data_list=data_dicts,
                unique_fields=['index', 'trade_time']
            )
            logger.info(f"{index_code}指数实时数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"获取并保存指数[{index_code}]实时数据失败: {str(e)}")
            return None
    
    async def fetch_and_save_all_realtime_data(self) -> Dict:
        """
        获取并保存所有指数的实时数据，使用批量操作提高效率
        """
        try:
            # 获取指数信息
            indexs = await self.get_all_indexes()
            if not indexs:
                logger.warning(f"指数不存在，无法获取实时数据")
                return None
            # 调用API获取实时数据
            for index in indexs:
                await self.fetch_and_save_realtime_data_by_index_code(index.code)
        except Exception as e:
            logger.error(f"获取并保存指数实时数据失败: {str(e)}")
            return None
        
    async def fetch_and_save_market_overview(self) -> Dict:
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
                    'trade_time': datetime.datetime.now()
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

    async def fetch_and_save_latest_time_series(self, index_code: str, time_level: str) -> Dict:
        """
        获取并保存指数时间序列数据
        Args:
            index_code: 指数代码
            time_level: 时间级别
        Returns:
            Dict: 保存结果统计
        """
        try:
            # 获取指数信息
            index = await self.get_index_by_code(index_code)
            if not index:
                logger.warning(f"指数代码[{index_code}]不存在，无法获取时间序列数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 获取最新时间序列数据
            api_data = await self.api.get_latest_time_series(index_code, time_level)
            if not api_data:
                logger.warning(f"API未返回指数[{index_code}]的{time_level}级别最新时间序列数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            data_dicts = []
            data_dict = await self.data_format_process.set_time_series(index, time_level, api_data)
            data_dicts.append(data_dict)
            # 保存数据
            logger.info(f"开始保存{index_code}指数实时数据")
            result = await self._save_all_to_db(
                model_class=IndexTimeSeriesData,
                data_list=data_dicts,
                unique_fields=['index', 'trade_time', 'time_level']
            )
            await self.cache_set.latest_time_series(index_code, time_level, data_dict)
            logger.info(f"{index_code}指数实时数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"获取并保存指数[{index_code}]的最新时间序列数据失败: {str(e)}")
            return []
    
    async def fetch_and_save_latest_time_series_by_index_code(self, index_code: str) -> Dict:
        """
        获取并保存指数最新时间序列数据
        Args:
            index_code: 指数代码            
        Returns:
            Dict: 保存的时间序列数据对象列表
        """
        try:
            # 获取指数信息
            index = await self.get_index_by_code(index_code)
            logger.info(f"index: {index.id}")
            if not index:
                logger.warning(f"指数代码[{index_code}]不存在，无法获取时间序列数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            data_dicts = []
            # 获取最新时间序列数据
            for time_level in TIME_LEVELS:
                api_data = await self.api.get_latest_time_series(index_code, time_level)
                data_dict = await self.data_format_process.set_time_series(index, time_level, api_data)
                data_dicts.append(data_dict)
                cache_dict = data_dict.copy()
                await self.cache_set.latest_time_series(index_code, time_level, cache_dict)
            # 保存数据
            logger.info(f"开始保存{index_code}指数最新时间序列数据")
            result = await self._save_all_to_db(
                model_class=IndexTimeSeriesData,
                data_list=data_dicts,
                unique_fields=['index', 'trade_time', 'time_level']
            )
            logger.info(f"{index_code}指数最新时间序列数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"获取并保存指数[{index_code}]的最新时间序列数据失败: {str(e)}")
            return []

    async def fetch_and_save_all_latest_time_series(self) -> Dict:
        """
        获取并保存指数最新时间序列数据
        """
        try:
            # 获取所有指数
            indexes = await self.get_all_indexes()
            for index in indexes:
                await self.fetch_and_save_latest_time_series_by_index_code(index.code)
        except Exception as e:
            logger.error(f"获取并保存所有指数最新时间序列数据失败: {str(e)}")
            return []

    async def fetch_and_save_history_time_series(self, index_code: str, time_level: str) -> Dict:
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
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 获取最新时间序列数据
            api_datas = await self.api.get_history_time_series(index_code, time_level)
            if not api_datas:
                logger.warning(f"API未返回指数[{index_code}]的{time_level}级别历史时间序列数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            data_dicts = []
            processed_indices_in_batch = set() # 跟踪当前批次涉及的指数代码
            trim_results_log = {} # 收集修剪日志
            for api_data in api_datas:
                data_dict = await self.data_format_process.set_time_series(index, time_level, api_data)
                data_dicts.append(data_dict)
                processed_indices_in_batch.add(index_code) # 记录本批次处理了哪个指数
                await self.cache_set.history_time_series(index_code, time_level, data_dict)
            # 保存数据
            logger.info(f"开始保存{index_code}指数历史时间序列数据")
            result = await self._save_all_to_db(
                model_class=IndexTimeSeriesData,
                data_list=data_dicts,
                unique_fields=['index', 'trade_time', 'time_level']
            )
            # --- 函数末尾执行最终修剪 ---
            # --- 生成缓存键 ---
            cache_key =  self.cache_key.history_time_series(index_code, time_level)
            # --- 单行调用修剪方法 ---
            removed_count = await self.cache_manager.trim_cache_zset(cache_key, self.cache_limit)
            # --- 修剪调用结束 ---
            logger.info(f"{index_code}指数历史时间序列数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"获取并保存指数[{index_code}]的历史时间序列数据失败: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}
    
    async def fetch_and_save_history_time_series_by_index_code(self, index_code: str) -> Dict:
        """
        获取并保存指数历史时间序列数据
        
        Args:
            index_code: 指数代码
        Returns:
            Dict: 保存的时间序列数据对象列表
        """
        try:
            # 获取指数信息
            index = await self.get_index_by_code(index_code)
            if not index:
                logger.warning(f"指数代码[{index_code}]不存在，无法获取时间序列数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            logger.info(f"开始获取{index_code}指数历史时间序列数据")
            data_dicts = []
            total_result = {'创建': 0, '更新': 0, '跳过': 0}
            # 获取最新时间序列数据
            for time_level in TIME_LEVELS:
                api_datas = await self.api.get_history_time_series(index_code, time_level)
                for data_index, api_data in enumerate(api_datas):
                    data_dict = await self.data_format_process.set_time_series(index, time_level, api_data)
                    data_dicts.append(data_dict)
                    if data_index < self.cache_limit:
                        await self.cache_set.history_time_series(index_code, time_level, data_dict)
                # 当数据量超过5万时，保存一次
                if len(data_dicts) >= 10000:
                    logger.info(f"数据量达到{len(data_dicts)}，开始保存批次数据")
                    batch_result = await self._save_all_to_db(
                        model_class=IndexTimeSeriesData,
                        data_list=data_dicts,
                unique_fields=['index', 'time_level', 'trade_time']
            )
                    logger.info(f"批次数据保存完成，结果: {batch_result}")
                    # 累加结果
                    for key in total_result:
                        total_result[key] += batch_result.get(key, 0)
                    # 清空数据列表，准备下一批
                    data_dicts = []
                
            
            # 保存剩余数据
            if data_dicts:
                logger.info(f"开始保存剩余{len(data_dicts)}条数据")
                final_result = await self._save_all_to_db(
                    model_class=IndexTimeSeriesData,
                    data_list=data_dicts,
                    unique_fields=['index', 'time_level', 'trade_time']
                )
                logger.info(f"剩余数据保存完成，结果: {final_result}")
                # 累加最终结果
                for key in total_result:
                    total_result[key] += final_result.get(key, 0)
            # --- 函数末尾执行最终修剪 ---
            for time_level in TIME_LEVELS:
                # --- 生成缓存键 ---
                cache_key =  self.cache_key.history_time_series(index_code, time_level)
                # --- 单行调用修剪方法 ---
                removed_count = await self.cache_manager.trim_cache_zset(cache_key, self.cache_limit)
                # --- 修剪调用结束 ---
            # --- 最终修剪结束 ---
            logger.info(f"所有指数各级别历史时间序列数据保存完成，总结果: {total_result}")
            return total_result
        except Exception as e:
            logger.error(f"获取并保存指数[{index_code}]的历史时间序列数据失败: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_all_history_time_series(self) -> Dict:
        """
        获取并保存所有指数历史时间序列数据
        使用线程池并行处理多个指数的数据获取和保存
        
        Returns:
            Dict: 包含处理结果的字典
        """
        try:
            # 获取所有指数
            indexes = await self.get_all_indexes()
            results = {"success": [], "failed": []}
            
            # 创建线程池，最大工作线程数为10
            with ThreadPoolExecutor(max_workers=10) as executor:
                # 创建Future对象列表
                future_to_index = {
                    executor.submit(
                        asyncio.run,
                        self.fetch_and_save_history_time_series_by_index_code(index.code)
                    ): index.code
                    for index in indexes
                }
                
                # 处理完成的任务
                for future in as_completed(future_to_index):
                    index_code = future_to_index[future]
                    try:
                        future.result()
                        results["success"].append(index_code)
                        logger.info(f"成功获取并保存指数[{index_code}]的历史时间序列数据")
                    except Exception as e:
                        results["failed"].append({
                            "code": index_code,
                            "error": str(e)
                        })
                        logger.error(f"获取并保存指数[{index_code}]的历史时间序列数据失败: {str(e)}")
            
            return results
            
        except Exception as e:
            logger.error(f"获取并保存所有指数历史时间序列数据失败: {str(e)}")
            return {"success": [], "failed": [], "error": str(e)}

    # ================ 指数KDJ指标数据 ================
    async def fetch_and_save_latest_kdj(self, index_code: str, time_level: str) -> Dict:
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
            index = await self.get_index_by_code(index_code)
            if not index:
                return {'创建': 0, '更新': 0, '跳过': 0}
            api_data = await self.api.get_latest_kdj(index_code, time_level)
            if not api_data:
                logger.warning(f"API未返回指数[{index_code}]的{time_level}级别KDJ指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 处理API返回的数据
            data_dicts = []
            try:
                data_dict = await self.data_format_process.set_kdj_data(index, time_level, api_data)
                data_dicts.append(data_dict)
                await self.cache_set.latest_kdj(index_code, time_level, data_dict)
            except Exception as e:
                logger.error(f"解析指数KDJ指标数据失败: {str(e)}")
                return []
            
            # 保存数据
            logger.info(f"开始保存{index_code}指数{time_level}级别KDJ指标数据")
            result = await self._save_all_to_db(
                model_class=IndexKDJData,
                data_list=data_dicts,
                unique_fields=['index', 'time_level', 'trade_time']
            )
            
            logger.info(f"{index_code}指数{time_level}级别KDJ指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"获取并保存指数KDJ指标数据失败: {str(e)}")
            return []
    
    async def fetch_and_save_latest_kdj_by_time_level(self, time_level: str) -> Dict:
        """
        获取并保存所有指数的KDJ指标数据
        """
        # 获取指数信息
        indexs = await self.get_all_indexes()
        if not indexs:
            logger.warning("没有指数数据，无法保存KDJ指标数据")
            return {'创建': 0, '更新': 0, '跳过': 0}
        data_dicts = []
        # 调用API获取实时数据
        for index in indexs:
            api_data = await self.api.get_latest_kdj(index.code, time_level)
            data_dict = await self.data_format_process.set_kdj_data(index, time_level, api_data)
            await self.cache_set.latest_kdj(index.code, time_level, data_dict)
            data_dicts.append(data_dict)
        # 保存数据
        logger.info(f"开始保存所有指数{time_level}级别KDJ指标数据")
        result = await self._save_all_to_db(
            model_class=IndexKDJData,
            data_list=data_dicts,
            unique_fields=['index', 'time_level', 'trade_time']
        )
        return result

    async def fetch_and_save_latest_kdj_by_code(self, index_code: str) -> Dict:
        """
        获取并保存指数KDJ指标数据
        Args:
            index_code: 指数代码
        Returns:
            List[IndexKDJData]: 保存的KDJ指标数据对象列表
        """
        try:
            index = await self.get_index_by_code(index_code)
            if not index:
                logger.warning(f"指数代码[{index_code}]不存在，无法获取KDJ指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            # 调用API获取实时数据
            data_dicts = []
            for period in TIME_LEVELS:
                api_data = await self.api.get_latest_kdj(index_code, period)
                data_dict = await self.data_format_process.set_kdj_data(index, period, api_data)
                await self.cache_set.latest_kdj(index_code, period, data_dict)
                data_dicts.append(data_dict)
            # 保存数据
            logger.info(f"开始保存{index_code}指数各级别KDJ指标数据")
            result = await self._save_all_to_db(
                model_class=IndexKDJData,
                data_list=data_dicts,
                unique_fields=['index', 'time_level', 'trade_time']
            )
            return result
           
        except Exception as e:
            logger.error(f"获取并保存指数KDJ指标数据失败: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_all_latest_kdj(self) -> Dict:
        """
        获取并保存所有指数的KDJ指标数据
        """
        indexs = await self.get_all_indexes()
        if not indexs:
            logger.warning("没有指数数据，无法保存KDJ指标数据")
            return {'创建': 0, '更新': 0, '跳过': 0}
        data_dicts = []
        for index in indexs:
            for time_level in TIME_LEVELS:
                api_data = await self.api.get_latest_kdj(index.code, time_level)
                data_dict = await self.data_format_process.set_kdj_data(index, time_level, api_data)
                await self.cache_set.latest_kdj(index.code, time_level, data_dict)
                data_dicts.append(data_dict)
        # 保存数据
        logger.info(f"开始保存所有指数各级别KDJ指标数据")
        result = await self._save_all_to_db(
            model_class=IndexKDJData,
            data_list=data_dicts,
            unique_fields=['index', 'time_level', 'trade_time']
        )
        logger.info(f"所有指数各级别KDJ指标数据保存完成，结果: {result}")
        return result
        
    async def fetch_and_save_history_kdj(self, index_code: str, time_level: str) -> Dict:
        """
        获取并保存指数历史KDJ指标数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            
        Returns:
            List[IndexKDJData]: 保存的KDJ指标数据对象列表
        """
        index = await self.get_index_by_code(index_code)
        if not index:
            logger.warning(f"指数代码[{index_code}]不存在，无法获取历史KDJ指标数据")
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            # 调用API获取KDJ指标数据
            api_datas = await self.api.get_history_kdj(index_code, time_level)
            # logger.info(f"成功获取指数[{index_code}]的{time_level}级别历史KDJ指标数据: {api_data_list}")
            if not api_datas:
                logger.warning(f"API未返回指数[{index_code}]的{time_level}级别历史KDJ指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            data_dicts = []
            for api_data in api_datas:
                data_dict = await self.data_format_process.set_kdj_data(index, time_level, api_data)
                data_dicts.append(data_dict)
                # 检查是否在缓存限制内 (只对前 cache_limit 条执行)
                await self.cache_set.history_kdj(index.code, time_level, data_dict)
            # 保存数据
            logger.info(f"开始保存{index_code}指数{time_level}级别历史KDJ指标数据")
            result = await self._save_all_to_db(
                model_class=IndexKDJData,
                data_list=data_dicts,
                unique_fields=['index', 'time_level', 'trade_time']
            )
            # --- 函数末尾执行最终修剪 ---
            # --- 生成缓存键 ---
            cache_key =  self.cache_key.history_kdj(index_code, time_level)
            # --- 单行调用修剪方法 ---
            removed_count = await self.cache_manager.trim_cache_zset(cache_key, self.cache_limit)
            # --- 修剪调用结束 ---
            logger.info(f"{index_code}指数{time_level}级别历史KDJ指标数据保存完成，结果: {result}")
            return result

        except Exception as e:
            logger.error(f"获取并保存指数历史KDJ指标数据失败: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_history_kdj_by_index_code(self, index_code: str) -> Dict:
        """
        获取并保存指数的历史KDJ指标数据
        Args:
            index_code: 指数代码
        Returns:
            List[IndexKDJData]: 保存的KDJ指标数据对象列表
        """
        index = await self.get_index_by_code(index_code)
        total_result = {'创建': 0, '更新': 0, '跳过': 0}
        if not index:
            logger.warning(f"指数代码[{index_code}]不存在，无法获取历史KDJ指标数据")
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            # 调用API获取KDJ指标数据
            data_dicts = []
            for time_level in TIME_LEVELS:
                api_datas = await self.api.get_history_kdj(index_code, time_level)
                # logger.info(f"成功获取指数[{index_code}]的{time_level}级别历史KDJ指标数据: {api_data_list}")
                if not api_datas:
                    logger.warning(f"API未返回指数[{index_code}]的{time_level}级别历史KDJ指标数据")
                    return {'创建': 0, '更新': 0, '跳过': 0}
                for data_index, api_data in enumerate(api_datas):
                    data_dict = await self.data_format_process.set_kdj_data(index, time_level, api_data)
                    data_dicts.append(data_dict)
                    # 检查是否在缓存限制内 (只对前 cache_limit 条执行)
                    if data_index < self.cache_limit:
                        cache_dict = data_dict.copy()
                        await self.cache_set.history_kdj(index.code, time_level, cache_dict)
            
                # 当数据量超过10万时，保存一次
                if len(data_dicts) >= 10000:
                    logger.info(f"数据量达到{len(data_dicts)}，开始保存批次数据")
                    batch_result = await self._save_all_to_db(
                        model_class=IndexKDJData,
                        data_list=data_dicts,
                        unique_fields=['index', 'time_level', 'trade_time']
                    )
                    logger.info(f"批次数据保存完成，结果: {batch_result}")
                    # 累加结果
                    for key in total_result:
                        total_result[key] += batch_result.get(key, 0)
                    # 清空数据列表，准备下一批
                    data_dicts = []
            # 保存剩余数据
            if data_dicts:
                logger.info(f"开始保存剩余{len(data_dicts)}条数据")
                final_result = await self._save_all_to_db(
                    model_class=IndexKDJData,
                    data_list=data_dicts,
                    unique_fields=['index', 'time_level', 'trade_time']
                )
                logger.info(f"剩余数据保存完成，结果: {final_result}")
                # 累加最终结果
                for key in total_result:
                    total_result[key] += final_result.get(key, 0)
            # --- 函数末尾执行最终修剪 ---
            for time_level in TIME_LEVELS:
                # --- 生成缓存键 ---
                cache_key =  self.cache_key.history_kdj(index_code, time_level)
                # --- 单行调用修剪方法 ---
                removed_count = await self.cache_manager.trim_cache_zset(cache_key, self.cache_limit)
                # --- 修剪调用结束 ---
            # --- 最终修剪结束 ---
            logger.info(f"所有指数各级别历史KDJ指标数据保存完成，总结果: {total_result}")
            return total_result

        except Exception as e:
            logger.error(f"获取并保存指数历史KDJ指标数据失败: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_all_history_kdj(self) -> Dict:
        """
        获取并保存所有指数的历史KDJ指标数据
        """
        indexs = await self.get_all_indexes()
        try:
            # 获取所有指数
            indexes = await self.get_all_indexes()
            for index in indexes:
                await self.fetch_and_save_history_kdj_by_index_code(index.code)
        except Exception as e:
            logger.error(f"获取并保存所有指数历史KDJ指标数据失败: {str(e)}")
            return []
    
    # ================ 指数MACD指标数据 ================
    async def fetch_and_save_latest_macd(self, index_code: str, time_level: str) -> Dict:
        """
        获取并保存指数MACD指标数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            
        Returns:
            List[IndexMACDData]: 保存的MACD指标数据对象列表
        """
        try:
            # 调用API获取KDJ指标数据
            index = await self.get_index_by_code(index_code)
            if not index:
                return {'创建': 0, '更新': 0, '跳过': 0}
            api_data = await self.api.get_latest_macd(index_code, time_level)
            if not api_data:
                logger.warning(f"API未返回指数[{index_code}]的{time_level}级别MACD指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            data_dicts = []
            try:
                data_dict = await self.data_format_process.set_macd_data(index, time_level, api_data)
                await self.cache_set.latest_macd(index_code, time_level, data_dict)
                data_dicts.append(data_dict)
            except Exception as e:
                logger.error(f"解析指数MACD指标数据失败: {str(e)}")
                return []
            # 保存数据
            logger.info(f"开始保存{index_code}指数{time_level}级别MACD指标数据")
            result = await self._save_all_to_db(
                model_class=IndexMACDData,
                data_list=data_dicts,
                unique_fields=['index', 'time_level', 'trade_time']
            )
            cache_dict = data_dict.copy()
            await self.cache_set.latest_macd(index_code, time_level, cache_dict)
            logger.info(f"{index_code}指数{time_level}级别MACD指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"获取并保存指数MACD指标数据失败: {str(e)}")
            return []

    async def fetch_and_save_latest_macd_by_time_level(self, time_level: str) -> Dict:
        """
        获取并保存所有指数的MACD指标数据
        Args:
            time_level: 时间级别
        Returns:
            List[IndexMACDData]: 保存的MACD指标数据对象列表
        """
        # 获取指数信息
        indexs = await self.get_all_indexes()
        if not indexs:
            logger.warning("没有指数数据，无法保存MACD指标数据")
            return {'创建': 0, '更新': 0, '跳过': 0}
        data_dicts = []
        # 调用API获取实时数据
        for index in indexs:
            api_data = await self.api.get_latest_macd(index.code, time_level)
            data_dict = await self.data_format_process.set_macd_data(index, time_level, api_data)
            cache_dict = data_dict.copy()
            await self.cache_set.latest_macd(index.code, time_level, cache_dict)
            data_dicts.append(data_dict)
        # 保存数据
        logger.info(f"开始保存所有指数{time_level}级别MACD指标数据")
        result = await self._save_all_to_db(
            model_class=IndexMACDData,
            data_list=data_dicts,
            unique_fields=['index', 'time_level', 'trade_time']
        )
        return result

    async def fetch_and_save_latest_macd_by_code(self, index_code: str) -> Dict:
        """
        获取并保存指数MACD指标数据
        Args:
            index_code: 指数代码
        Returns:
            List[IndexMACDData]: 保存的MACD指标数据对象列表
        """
        try:
            index = await self.get_index_by_code(index_code)
            if not index:
                logger.warning(f"指数代码[{index_code}]不存在，无法获取MACD指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            # 调用API获取实时数据
            data_dicts = []
            for time_level in TIME_LEVELS:
                api_data = await self.api.get_latest_macd(index_code, time_level)
                data_dict = await self.data_format_process.set_macd_data(index, time_level, api_data)
                cache_dict = data_dict.copy()
                await self.cache_set.latest_macd(index_code, time_level, cache_dict)
                data_dicts.append(data_dict)
            # 保存数据
            logger.info(f"开始保存{index_code}指数各级别MACD指标数据")
            result = await self._save_all_to_db(
                model_class=IndexMACDData,
                data_list=data_dicts,
                unique_fields=['index', 'time_level', 'trade_time']
            )
            return result
           
        except Exception as e:
            logger.error(f"获取并保存指数MACD指标数据失败: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_all_latest_macd(self) -> Dict:
        """
        获取并保存所有指数的最新MACD指标数据
        """
        indexs = await self.get_all_indexes()
        if not indexs:
            logger.warning("没有指数数据，无法保存MACD指标数据")
            return {'创建': 0, '更新': 0, '跳过': 0}
        data_dicts = []
        for index in indexs:
            for time_level in TIME_LEVELS:
                api_data = await self.api.get_latest_macd(index.code, time_level)
                data_dict = await self.data_format_process.set_macd_data(index, time_level, api_data)
                cache_dict = data_dict.copy()
                await self.cache_set.latest_macd(index.code, time_level, cache_dict)
                data_dicts.append(data_dict)
        # 保存数据
        logger.info(f"开始保存所有指数各级别MACD指标数据")
        result = await self._save_all_to_db(
            model_class=IndexMACDData,
            data_list=data_dicts,
            unique_fields=['index', 'time_level', 'trade_time']
        )
        logger.info(f"所有指数各级别MACD指标数据保存完成，结果: {result}")
        return result
    
    async def fetch_and_save_history_macd(self, index_code: str, time_level: str) -> Dict:
        """
        获取并保存指数历史MACD指标数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            
        Returns:
            List[IndexMACDData]: 保存的MACD指标数据对象列表
        """
        index = await self.get_index_by_code(index_code)
        if not index:
            logger.warning(f"指数代码[{index_code}]不存在，无法获取历史MACD指标数据")
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            # 调用API获取KDJ指标数据
            api_datas = await self.api.get_history_macd(index_code, time_level)
            # logger.info(f"成功获取指数[{index_code}]的{time_level}级别历史KDJ指标数据: {api_data_list}")
            if not api_datas:
                logger.warning(f"API未返回指数[{index_code}]的{time_level}级别历史MACD指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            data_dicts = []
            processed_indices_in_batch = set() # 跟踪当前批次涉及的指数代码
            trim_results_log = {} # 收集修剪日志
            for index, api_data in enumerate(api_datas):
                data_dict = await self.data_format_process.set_macd_data(index, time_level, api_data)
                data_dicts.append(data_dict)
                processed_indices_in_batch.add(index.code)
                cache_dict = data_dict.copy()
                await self.cache_set.history_macd(index_code, time_level, cache_dict)
                    
            # 保存数据
            logger.info(f"开始保存{index_code}指数{time_level}级别历史MACD指标数据")
            result = await self._save_all_to_db(
                model_class=IndexMACDData,
                data_list=data_dicts,
                unique_fields=['index', 'time_level', 'trade_time']
            )
            # --- 函数末尾执行最终修剪 ---
             # --- 生成缓存键 ---
            cache_key =  self.cache_key.history_macd(index_code, time_level)
            # --- 单行调用修剪方法 ---
            removed_count = await self.cache_manager.trim_cache_zset(cache_key, self.cache_limit)
            # --- 修剪调用结束 ---
            trim_results_log[f"{index_code}_{time_level}_final"] = f"移除 {removed_count}" if removed_count is not None else "失败"

            # --- 最终修剪结束 ---
            logger.info(f"{index_code}指数{time_level}级别历史MACD指标数据保存完成，结果: {result}")
            return result

        except Exception as e:
            logger.error(f"获取并保存指数历史MACD指标数据失败: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_history_macd_by_index_code(self, index_code: str) -> Dict:
        """
        获取并保存指数历史MACD指标数据
        Args:
            index_code: 指数代码
        Returns:
            List[IndexMACDData]: 保存的MACD指标数据对象列表
        """
        index = await self.get_index_by_code(index_code)
        total_result = {'创建': 0, '更新': 0, '跳过': 0}
        if not index:
            logger.warning(f"指数代码[{index_code}]不存在，无法获取历史MACD指标数据")
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            # 调用API获取KDJ指标数据
            data_dicts = []
            processed_indices_in_batch = set() # 跟踪当前批次涉及的指数代码
            trim_results_log = {} # 收集修剪日志
            for time_level in TIME_LEVELS:
                api_datas = await self.api.get_history_macd(index_code, time_level)
                if not api_datas:
                    return {'创建': 0, '更新': 0, '跳过': 0}
                processed_indices_in_batch.add(time_level)
                for data_index, api_data in enumerate(api_datas):
                    data_dict = await self.data_format_process.set_macd_data(index, time_level, api_data)
                    data_dicts.append(data_dict)
                    # 检查是否在缓存限制内 (只对前 cache_limit 条执行)
                    if data_index < self.cache_limit:
                        cache_dict = data_dict.copy()
                        await self.cache_set.history_macd(index_code, time_level, cache_dict)
            
                # 当数据量超过10万时，保存一次
                if len(data_dicts) >= 10000:
                    logger.info(f"数据量达到{len(data_dicts)}，开始保存批次数据")
                    batch_result = await self._save_all_to_db(
                        model_class=IndexMACDData,
                        data_list=data_dicts,
                        unique_fields=['index', 'time_level', 'trade_time']
                    )
                    # 累加结果
                    for key in total_result:
                        total_result[key] += batch_result.get(key, 0)
                    # 清空数据列表，准备下一批
                    data_dicts = []
                
            
            # 保存剩余数据
            if data_dicts:
                logger.info(f"开始保存剩余{len(data_dicts)}条数据")
                final_result = await self._save_all_to_db(
                    model_class=IndexMACDData,
                    data_list=data_dicts,
                    unique_fields=['index', 'time_level', 'trade_time']
                )
                logger.info(f"剩余数据保存完成，结果: {final_result}")
                # 累加最终结果
                for key in total_result:
                    total_result[key] += final_result.get(key, 0)
            # --- 函数末尾执行最终修剪 ---
            for time_level_to_trim in processed_indices_in_batch:
                # --- 生成 KDJ 缓存键 ---
                cache_key =  self.cache_key.history_macd(index_code, time_level_to_trim)
                # --- 单行调用修剪方法 ---
                removed_count = await self.cache_manager.trim_cache_zset(cache_key, self.cache_limit)
                # --- 修剪调用结束 ---
                trim_results_log[f"{index_code}_{time_level_to_trim}_final"] = f"移除 {removed_count}" if removed_count is not None else "失败"
            # --- 最终修剪结束 ---
            logger.info(f"所有指数各级别历史MACD指标数据保存完成，总结果: {total_result}")
            return total_result

        except Exception as e:
            logger.error(f"获取并保存指数历史MACD指标数据失败: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}
    
    async def fetch_and_save_all_history_macd(self) -> Dict:
        """
        获取并保存所有指数的历史MACD指标数据
        """
        indexs = await self.get_all_indexes()
        try:
            # 获取所有指数
            indexes = await self.get_all_indexes()
            for index in indexes:
                await self.fetch_and_save_history_macd_by_index_code(index.code)
        except Exception as e:
            logger.error(f"获取并保存所有指数历史MACD指标数据失败: {str(e)}")
            return []
    
    # ================ 指数MA指标数据 ================
    async def fetch_and_save_latest_ma(self, index_code: str, time_level: str) -> Dict:
        """
        获取并保存指数MA指标数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            
        Returns:
            List[IndexMAData]: 保存的MA指标数据对象列表
        """
        try:
            # 调用API获取KDJ指标数据
            index = await self.get_index_by_code(index_code)
            if not index:
                return {'创建': 0, '更新': 0, '跳过': 0}
            api_data = await self.api.get_latest_ma(index_code, time_level)
            if not api_data:
                logger.warning(f"API未返回指数[{index_code}]的{time_level}级别MA指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 处理API返回的数据
            data_dicts = []
            try:
                data_dict = await self.data_format_process.set_ma_data(index, time_level, api_data)
                cache_dict = data_dict.copy()
                await self.cache_set.latest_ma(index_code, time_level, cache_dict)
                data_dicts.append(data_dict)
            except Exception as e:
                logger.error(f"解析指数MA指标数据失败: {str(e)}")
                return []
            
            # 保存数据
            logger.info(f"开始保存{index_code}指数{time_level}级别MA指标数据")
            result = await self._save_all_to_db(
                model_class=IndexMAData,
                data_list=data_dicts,
                unique_fields=['index', 'time_level', 'trade_time']
            )
            
            logger.info(f"{index_code}指数{time_level}级别MA指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"获取并保存指数MA指标数据失败: {str(e)}")
            return []
    
    async def fetch_and_save_latest_ma_by_time_level(self, time_level: str) -> Dict:
        """
        获取并保存所有指数的MA指标数据
        Args:
            time_level: 时间级别
        Returns:
            List[IndexMAData]: 保存的MA指标数据对象列表
        """
         # 获取指数信息
        indexs = await self.get_all_indexes()
        if not indexs:
            logger.warning("没有指数数据，无法保存MA指标数据")
            return {'创建': 0, '更新': 0, '跳过': 0}
        data_dicts = []
        # 调用API获取实时数据
        for index in indexs:
            api_data = await self.api.get_latest_ma(index.code, time_level)
            data_dict = await self.data_format_process.set_ma_data(index, time_level, api_data)
            cache_dict = data_dict.copy()
            await self.cache_set.latest_ma(index.code, time_level, cache_dict)
            data_dicts.append(data_dict)
        # 保存数据
        logger.info(f"开始保存所有指数{time_level}级别MA指标数据")
        result = await self._save_all_to_db(
            model_class=IndexMAData,
            data_list=data_dicts,
            unique_fields=['index', 'time_level', 'trade_time']
        )
        logger.info(f"所有指数{time_level}级别MA指标数据保存完成，结果: {result}")
        return result

    async def fetch_and_save_latest_ma_by_index_code(self, index_code: str) -> Dict:
        """
        获取并保存指数MA指标数据
        
        Args:
            index_code: 指数代码
        Returns:
            List[IndexMAData]: 保存的MA指标数据对象列表
        """
        try:
            index = await self.get_index_by_code(index_code)
            if not index:
                logger.warning(f"指数代码[{index_code}]不存在，无法获取MA指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            # 调用API获取实时数据
            data_dicts = []
            for time_level in TIME_LEVELS:
                api_data = await self.api.get_latest_ma(index_code, time_level)
                data_dict = await self.data_format_process.set_ma_data(index, time_level, api_data)
                cache_dict = data_dict.copy()
                await self.cache_set.latest_ma(index_code, time_level, cache_dict)
                data_dicts.append(data_dict)
            # 保存数据
            logger.info(f"开始保存{index_code}指数各级别MA指标数据")
            result = await self._save_all_to_db(
                model_class=IndexMAData,
                data_list=data_dicts,
                unique_fields=['index', 'time_level', 'trade_time']
            )
            return result
           
        except Exception as e:
            logger.error(f"获取并保存指数MA指标数据失败: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_all_latest_ma(self) -> Dict:
        """
        获取并保存所有指数的最新MA指标数据
        """
        indexs = await self.get_all_indexes()
        if not indexs:
            logger.warning("没有指数数据，无法保存MA指标数据")
            return {'创建': 0, '更新': 0, '跳过': 0}
        data_dicts = []
        for index in indexs:
            for time_level in TIME_LEVELS:
                api_data = await self.api.get_latest_ma(index.code, time_level)
                data_dict = await self.data_format_process.set_ma_data(index, time_level, api_data)
                cache_dict = data_dict.copy()
                await self.cache_set.latest_ma(index.code, time_level, cache_dict)
                data_dicts.append(data_dict)
        # 保存数据
        logger.info(f"开始保存所有指数各级别MA指标数据")
        result = await self._save_all_to_db(
            model_class=IndexMAData,
            data_list=data_dicts,
            unique_fields=['index', 'time_level', 'trade_time']
        )
        logger.info(f"所有指数各级别MA指标数据保存完成，结果: {result}")
        return result
    
    async def fetch_and_save_history_ma(self, index_code: str, time_level: str) -> Dict:
        """
        获取并保存历史指数MA指标数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            
        Returns:
            List[IndexMAData]: 保存的MA指标数据对象列表
        """
        index = await self.get_index_by_code(index_code)
        if not index:
            logger.warning(f"指数代码[{index_code}]不存在，无法获取历史MA指标数据")
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            # 调用API获取KDJ指标数据
            api_datas = await self.api.get_history_ma(index_code, time_level)
            # logger.info(f"成功获取指数[{index_code}]的{time_level}级别历史KDJ指标数据: {api_data_list}")
            if not api_datas:
                logger.warning(f"API未返回指数[{index_code}]的{time_level}级别历史MA指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            data_dicts = []
            for index, api_data in enumerate(api_datas):
                data_dict = await self.data_format_process.set_ma_data(index, time_level, api_data)
                data_dicts.append(data_dict)
                cache_dict = data_dict.copy()
                await self.cache_set.history_ma(index_code, time_level, cache_dict)
            # 保存数据
            logger.info(f"开始保存{index_code}指数{time_level}级别历史MA指标数据")
            result = await self._save_all_to_db(
                model_class=IndexMAData,
                data_list=data_dicts,
                unique_fields=['index', 'time_level', 'trade_time']
            )
            # --- 函数末尾执行最终修剪 ---
            # --- 生成缓存键 ---
            cache_key =  self.cache_key.history_ma(index_code, time_level)
            # --- 单行调用修剪方法 ---
            removed_count = await self.cache_manager.trim_cache_zset(cache_key, self.cache_limit)
            # --- 修剪调用结束 ---
            logger.info(f"{index_code}指数{time_level}级别历史MA指标数据保存完成，结果: {result}")
            return result

        except Exception as e:
            logger.error(f"获取并保存指数历史MA指标数据失败: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_history_ma_by_index_code(self, index_code: str) -> Dict:
        """
        获取并保存指数历史MA指标数据
        Args:
            index_code: 指数代码
        Returns:
            List[IndexMAData]: 保存的MA指标数据对象列表
        """
        index = await self.get_index_by_code(index_code)
        total_result = {'创建': 0, '更新': 0, '跳过': 0}
        if not index:
            logger.warning(f"指数代码[{index_code}]不存在，无法获取历史MA指标数据")
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            # 调用API获取KDJ指标数据
            data_dicts = []
            processed_indices_in_batch = set() # 跟踪当前批次涉及的指数代码
            trim_results_log = {} # 收集修剪日志
            for time_level in TIME_LEVELS:
                api_datas = await self.api.get_history_ma(index_code, time_level)
                # logger.info(f"成功获取指数[{index_code}]的{time_level}级别历史KDJ指标数据: {api_data_list}")
                if not api_datas:
                    logger.warning(f"API未返回指数[{index_code}]的{time_level}级别历史MA指标数据")
                    return {'创建': 0, '更新': 0, '跳过': 0}
                for data_index, api_data in enumerate(api_datas):
                    data_dict = await self.data_format_process.set_ma_data(index, time_level, api_data)
                    data_dicts.append(data_dict)
                    processed_indices_in_batch.add(index.code)
                    # 检查是否在缓存限制内 (只对前 cache_limit 条执行)
                    if data_index < self.cache_limit:
                        cache_dict = data_dict.copy()
                        await self.cache_set.history_ma(index_code, time_level, cache_dict)
            
                # 当数据量超过10万时，保存一次
                if len(data_dicts) >= 10000:
                    logger.info(f"数据量达到{len(data_dicts)}，开始保存批次数据")
                    batch_result = await self._save_all_to_db(
                        model_class=IndexMAData,
                        data_list=data_dicts,
                        unique_fields=['index', 'time_level', 'trade_time']
                    )
                    logger.info(f"批次数据保存完成，结果: {batch_result}")
                    # 累加结果
                    for key in total_result:
                        total_result[key] += batch_result.get(key, 0)
                    # 清空数据列表，准备下一批
                    data_dicts = []
                
            
            # 保存剩余数据
            if data_dicts:
                logger.info(f"开始保存剩余{len(data_dicts)}条数据")
                final_result = await self._save_all_to_db(
                    model_class=IndexMAData,
                    data_list=data_dicts,
                    unique_fields=['index', 'time_level', 'trade_time']
                )
                logger.info(f"剩余数据保存完成，结果: {final_result}")
                # 累加最终结果
                for key in total_result:
                    total_result[key] += final_result.get(key, 0)
            # --- 函数末尾执行最终修剪 ---
            for time_level_to_trim in processed_indices_in_batch:
                # --- 生成缓存键 ---
                cache_key =  self.cache_key.history_ma(index_code, time_level_to_trim)
                # --- 单行调用修剪方法 ---
                removed_count = await self.cache_manager.trim_cache_zset(cache_key, self.cache_limit)
                # --- 修剪调用结束 ---
                trim_results_log[f"{index_code}_{time_level_to_trim}_final"] = f"移除 {removed_count}" if removed_count is not None else "失败"
            # --- 最终修剪结束 ---
            logger.info(f"所有指数各级别历史MA指标数据保存完成，总结果: {total_result}")
            return total_result

        except Exception as e:
            logger.error(f"获取并保存指数历史MA指标数据失败: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}
    
    async def fetch_and_save_all_history_ma(self) -> Dict:
        """
        获取并保存所有指数的MA指标数据
        """
        indexs = await self.get_all_indexes()
        try:
            # 获取所有指数
            indexes = await self.get_all_indexes()
            for index in indexes:
                await self.fetch_and_save_history_ma_by_index_code(index.code)
        except Exception as e:
            logger.error(f"获取并保存所有指数历史MA指标数据失败: {str(e)}")
            return []
    
    # ================ 指数BOLL指标数据 ================
    async def fetch_and_save_latest_boll(self, index_code: str, time_level: str) -> Dict:
        """
        获取并保存指数BOLL指标数据
        
        Args:
            index_code: 指数代码
            time_level: 时间级别
            
        Returns:
            List[IndexBOLLData]: 保存的BOLL指标数据对象列表
        """
        try:
            # 调用API获取KDJ指标数据
            index = await self.get_index_by_code(index_code)
            if not index:
                return {'创建': 0, '更新': 0, '跳过': 0}
            api_data = await self.api.get_latest_boll(index_code, time_level)
            if not api_data:
                logger.warning(f"API未返回指数[{index_code}]的{time_level}级别BOLL指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 处理API返回的数据
            data_dicts = []
            try:
                data_dict = await self.data_format_process.set_boll_data(index, time_level, api_data)
                cache_dict = data_dict.copy()
                await self.cache_set.latest_boll(index_code, time_level, cache_dict)
                data_dicts.append(data_dict)
            except Exception as e:
                logger.error(f"解析指数BOLL指标数据失败: {str(e)}")
                return []
            
            # 保存数据
            logger.info(f"开始保存{index_code}指数{time_level}级别BOLL指标数据")
            result = await self._save_all_to_db(
                model_class=IndexBOLLData,
                data_list=data_dicts,
                unique_fields=['index', 'time_level', 'trade_time']
            )
            
            logger.info(f"{index_code}指数{time_level}级别BOLL指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"获取并保存指数BOLL指标数据失败: {str(e)}")
            return []

    async def fetch_and_save_latest_boll_by_time_level(self, time_level: str) -> Dict:
        """
        获取并保存所有指数的BOLL指标数据
        Args:
            time_level: 时间级别
        Returns:
            List[IndexBOLLData]: 保存的BOLL指标数据对象列表
        """
        # 获取指数信息
        indexs = await self.get_all_indexes()
        if not indexs:
            logger.warning("没有指数数据，无法保存BOLL指标数据")
            return {'创建': 0, '更新': 0, '跳过': 0}
        data_dicts = []
        # 调用API获取实时数据
        for index in indexs:
            api_data = await self.api.get_latest_boll(index.code, time_level)
            data_dict = await self.data_format_process.set_boll_data(index, time_level, api_data)
            cache_dict = data_dict.copy()
            await self.cache_set.latest_boll(index.code, time_level, cache_dict)
            data_dicts.append(data_dict)
        # 保存数据
        logger.info(f"开始保存所有指数{time_level}级别BOLL指标数据")
        result = await self._save_all_to_db(
            model_class=IndexBOLLData,
            data_list=data_dicts,
            unique_fields=['index', 'time_level', 'trade_time']
        )
        return result

    async def fetch_and_save_latest_boll_by_index_code(self, index_code: str) -> Dict:
        """
        获取并保存指数BOLL指标数据
        Args:
            index_code: 指数代码
        Returns:
            List[IndexBOLLData]: 保存的BOLL指标数据对象列表
        """
        try:
            index = await self.get_index_by_code(index_code)
            if not index:
                logger.warning(f"指数代码[{index_code}]不存在，无法获取BOLL指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            # 调用API获取实时数据
            data_dicts = []
            for period in TIME_LEVELS:
                api_data = await self.api.get_latest_boll(index_code, period)
                data_dict = await self.data_format_process.set_boll_data(index, period, api_data)
                cache_dict = data_dict.copy()
                await self.cache_set.latest_boll(index_code, period, cache_dict)
                data_dicts.append(data_dict)
            # 保存数据
            logger.info(f"开始保存{index_code}指数各级别BOLL指标数据")
            result = await self._save_all_to_db(
                model_class=IndexBOLLData,
                data_list=data_dicts,
                unique_fields=['index', 'time_level', 'trade_time']
            )
            return result
           
        except Exception as e:
            logger.error(f"获取并保存指数BOLL指标数据失败: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}
    
    async def fetch_and_save_all_latest_boll(self) -> Dict:
        """
        获取并保存所有指数的BOLL指标数据
        """
        indexs = await self.get_all_indexes()
        if not indexs:
            logger.warning("没有指数数据，无法保存BOLL指标数据")
            return {'创建': 0, '更新': 0, '跳过': 0}
        data_dicts = []
        for index in indexs:
            for time_level in TIME_LEVELS:
                api_data = await self.api.get_latest_boll(index.code, time_level)
                data_dict = await self.data_format_process.set_boll_data(index, time_level, api_data)
                cache_dict = data_dict.copy()
                await self.cache_set.latest_boll(index.code, time_level, cache_dict)
                data_dicts.append(data_dict)
        # 保存数据
        logger.info(f"开始保存所有指数各级别BOLL指标数据")
        result = await self._save_all_to_db(
            model_class=IndexBOLLData,
            data_list=data_dicts,
            unique_fields=['index', 'time_level', 'trade_time']
        )
        logger.info(f"所有指数各级别BOLL指标数据保存完成，结果: {result}")
        return result
    
    async def fetch_and_save_history_boll(self, index_code: str, time_level: str) -> Dict:
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
            index = await self.get_index_by_code(index_code)
            api_datas = await self.api.get_history_boll(index_code, time_level)
            if not api_datas:
                logger.warning(f"API未返回指数[{index_code}]的{time_level}级别历史指数BOLL指标数据")
                return []
            # 处理API返回的数据
            data_dicts = []
            try:
                for api_data in api_datas:
                    data_dict = await self.data_format_process.set_boll_data(index, time_level, api_data)
                    cache_dict = data_dict.copy()
                    await self.cache_set.history_boll(index_code, time_level, cache_dict)
                    data_dicts.append(data_dict)
            except Exception as e:
                logger.error(f"解析历史指数BOLL指标数据失败: {str(e)}")
                return []
            
            # 保存数据
            logger.info(f"开始保存{index_code}指数{time_level}级别历史MA指标数据")
            result = await self._save_all_to_db(
                model_class=IndexBOLLData,
                data_list=data_dicts,
                unique_fields=['index', 'time_level', 'trade_time']
            )
            # --- 函数末尾执行最终修剪 ---
            # --- 生成缓存键 ---
            cache_key =  self.cache_key.history_boll(index_code, time_level)
            # --- 单行调用修剪方法 ---
            removed_count = await self.cache_manager.trim_cache_zset(cache_key, self.cache_limit)
            # --- 修剪调用结束 ---
            logger.info(f"{index_code}指数{time_level}级别历史MA指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"获取并保存历史指数BOLL指标数据失败: {str(e)}")
            return []

    async def fetch_and_save_history_boll_by_index_code(self, index_code: str) -> Dict:
        """
        获取并保存指数BOLL指标数据
        Args:
            index_code: 指数代码
        Returns:
            List[IndexBOLLData]: 保存的BOLL指标数据对象列表
        """
        index = await self.get_index_by_code(index_code)
        total_result = {'创建': 0, '更新': 0, '跳过': 0}
        if not index:
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            # logger.warning(f"获取指数: {index}, {index.code}, type: {type(index)}")
            # 调用API获取KDJ指标数据
            data_dicts = []
            for time_level in TIME_LEVELS:
                api_datas = await self.api.get_history_boll(index_code, time_level)
                if not api_datas:
                    logger.warning(f"API未返回指数[{index_code}]的{time_level}级别历史BOLL指标数据")
                    return {'创建': 0, '更新': 0, '跳过': 0}
                for data_index, api_data in enumerate(api_datas):
                    data_dict = await self.data_format_process.set_boll_data(index, time_level, api_data)
                    data_dicts.append(data_dict)
                    # 检查是否在缓存限制内 (只对前 cache_limit 条执行)
                    if data_index < self.cache_limit:
                        cache_dict = data_dict.copy()
                        await self.cache_set.history_boll(index_code, time_level, cache_dict)
                # 当数据量超过10万时，保存一次
                if len(data_dicts) >= 10000:
                    logger.warning(f"数据量达到{len(data_dicts)}，开始保存批次数据")
                    batch_result = await self._save_all_to_db(
                        model_class=IndexBOLLData,
                        data_list=data_dicts,
                        unique_fields=['index', 'time_level', 'trade_time']
                    )
                    data_dicts = []
                    logger.warning(f"批次数据保存完成，结果: {batch_result}, type: {type(batch_result)}")
                    # 累加结果
                    for key in total_result:
                        total_result[key] += batch_result.get(key, 0)
                    # 清空数据列表，准备下一批
            # 保存剩余数据
            if data_dicts:
                logger.warning(f"开始保存剩余{len(data_dicts)}条数据")
                final_result = await self._save_all_to_db(
                    model_class=IndexBOLLData,
                    data_list=data_dicts,
                    unique_fields=['index', 'time_level', 'trade_time']
                )
                logger.warning(f"剩余数据保存完成，结果: {final_result}")
                # 累加最终结果
                for key in total_result:
                    total_result[key] += final_result.get(key, 0)
            # --- 函数末尾执行最终修剪 ---
            for time_level in TIME_LEVELS:
                # --- 生成缓存键 ---
                cache_key =  self.cache_key.history_ma(index_code, time_level)
                # --- 单行调用修剪方法 ---
                removed_count = await self.cache_manager.trim_cache_zset(cache_key, self.cache_limit)
                # --- 修剪调用结束 ---
            # --- 最终修剪结束 ---
            logger.info(f"所有指数各级别历史BOLL指标数据保存完成，总结果: {total_result}")
            return total_result

        except Exception as e:
            logger.error(f"获取并保存指数历史BOLL指标数据失败: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_all_history_boll(self) -> Dict:
        """
        获取并保存所有指数的BOLL指标数据
        """
        indexs = await self.get_all_indexes()
        try:
            # 获取所有指数
            indexes = await self.get_all_indexes()
            for index in indexes:
                # logger.info(f"获取所有指数: {index}, {index.code}, type: {type(index)}")
                await self.fetch_and_save_history_boll_by_index_code(index.code)
        except Exception as e:
            logger.error(f"获取并保存所有指数历史BOLL指标数据失败: {str(e)}")
            return []
