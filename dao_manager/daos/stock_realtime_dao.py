import asyncio
from decimal import Decimal
import logging
import time as time_lib  # 用于测量时间
from asyncio import Semaphore
from typing import Dict, List, Any, Optional, Tuple, Set, TypeVar, Generic, Type
from datetime import datetime, date, time
from channels.db import database_sync_to_async # 用于同步 ORM 查询
import json
from django.utils import timezone
from asgiref.sync import sync_to_async
from channels.layers import get_channel_layer
from django.conf import settings
from dao_manager.daos.stock_basic_dao import StockBasicDAO

from utils.cache_get import StockRealtimeCacheGet
from utils.cache_manager import CacheManager
from utils.cache_set import StockRealtimeCacheSet
from utils.cash_key import StockCashKey
from utils.data_format_process import StockRealtimeDataFormatProcess
from utils.models import ModelJSONEncoder

from django.db import transaction, models
from django.core.cache import cache
from django.db.models import Q, F

from api_manager.apis.stock_realtime_api import StockRealtimeAPI
from dao_manager.base_dao import BaseDAO
from stock_models.stock_basic import StockInfo
from stock_models.stock_realtime import StockAbnormalMovement, StockBigDeal, StockLevel5Data, StockPricePercent, StockRealtimeData, StockTimeDeal, StockTradeDetail

logger = logging.getLogger("dao")

class StockRealtimeDAO(BaseDAO):
    """
    股票实时数据DAO，整合所有相关的实时数据访问功能
    """
    
    def __init__(self):
        """初始化StockRealtimeDAO"""
        super().__init__(None, None, 3600)  # 基类使用None作为model_class，因为本DAO管理多个模型
        self.api = StockRealtimeAPI()
        self.stock_basic_dao = StockBasicDAO()
        self.data_format_process = StockRealtimeDataFormatProcess()
        self.cache_manager = None  # 初始化缓存管理器
        self.cache_get = None
        self.cache_set = None
        self.cache_key = StockCashKey()
        self.data_format_process = StockRealtimeDataFormatProcess()

    async def initialize_cache_objects(self):
        self.cache_manager = CacheManager()  # 先实例化
        await self.cache_manager.initialize()  # 然后 await 其异步初始化方法，如果存在

        self.cache_set = StockRealtimeCacheSet()  # 先实例化
        await self.cache_set.initialize()  # 添加异步初始化方法，如果需要

        self.cache_get = StockRealtimeCacheGet()  # 先实例化
        await self.cache_get.initialize()  # 添加异步初始化方法，如果需要

    @database_sync_to_async # 将同步的 ORM 查询包装成异步
    def _get_favorited_user_ids(self, stock_id: int) -> list[int]:
        """根据股票 ID 获取关注该股票的所有用户 ID 列表"""
        from users.models import FavoriteStock
        return list(FavoriteStock.objects.filter(stock_id=stock_id).values_list('user_id', flat=True))

    # 新增 close 方法
    async def close(self):
        """关闭内部持有的 API Client Session"""
        if hasattr(self, 'api') and self.api:
            await self.api.close()
        else:
            pass

    # ================= RealtimeData相关方法 =================
    async def get_latest_realtime_data(self, stock_code: str) -> Optional[StockRealtimeData]:
        """
        获取股票最新的实时交易数据
        Args:
            stock_code: 股票代码
        Returns:
            Optional[StockRealtimeData]: 最新的实时交易数据，如不存在则返回None
        """
        realtime_data = None
        try:
            if self.cache_get is None:
                await self.initialize_cache_objects()
            # 从缓存获取最新数据
            cache_data = await self.cache_get.latest_realtime_data(stock_code)
            if cache_data:
                # logger.info(f"从缓存获取最新股票[{stock_code}]实时数据,cache_data: {cache_data}")
                realtime_data_dict = json.loads(cache_data)
                # logger.info(f"从缓存获取最新股票[{stock_code}]实时数据,realtime_data_dict: {realtime_data_dict}")
                realtime_data = StockRealtimeData(**realtime_data_dict)
                # logger.info(f"从缓存获取最新股票[{stock_code}]实时数据,realtime_data: {realtime_data}")
                return realtime_data
        except Exception as e:
            logger.error(f"从缓存获取最新股票[{stock_code}]实时数据时发生异常: {str(e)}")
            return None
        # 从数据库获取最新数据
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if stock:
            try:
                realtime_data = await sync_to_async(
                    lambda: StockRealtimeData.objects.filter(stock=stock).select_related('stock').order_by('-trade_time').first()
                )()
                if realtime_data:
                    stock_code_str = realtime_data.stock.stock_code if realtime_data.stock else stock_code  # 安全获取
                    # logger.info(f"从数据库获取最新股票[{stock_code_str}]实时数据,sql_data: {realtime_data.id} - {realtime_data.trade_time}")  # 使用具体字段代替全对象
                    return realtime_data
            except Exception as e:
                logger.error(f"从数据库获取最新股票[{stock}]实时数据失败: {str(e)}", exc_info=True)
        if not realtime_data:
            # 数据不存在或已过期，从API获取新数据
            logger.info(f"股票[{stock_code}]实时数据不存在或已过期，从API获取")
            await self.fetch_and_save_realtime_data(stock_code)
            # 再次从数据库获取数据
            realtime_data = await sync_to_async(lambda: StockRealtimeData.objects.filter(stock=stock).order_by('-trade_time').first())()
        return realtime_data

    async def fetch_and_save_realtime_data(self, stock_code: str) -> Dict:
        """
        从API获取实时交易数据并保存到数据库
        Args:
            stock_code: 股票代码
        Returns:
            Optional[StockRealtimeData]: 保存后的实时交易数据，保存失败则返回None
        """
        try:
            if self.cache_set is None:
                await self.initialize_cache_objects()
            # 获取股票信息
            stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
            if not stock:
                logger.warning(f"股票代码[{stock_code}]不存在，无法获取实时数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            # 调用API获取实时数据
            api_data = await self.api.get_realtime_data(stock_code)
            if not api_data:
                logger.warning(f"API未返回股票[{stock}]的实时数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            data_dicts = []
            data_dict = self.data_format_process.set_realtime_data(stock, api_data)
            if data_dict.get('trade_time') is not None:
                # 1. 准备用于数据库保存的字典 (包含 StockInfo 实例)
                # 注意：我们直接将原始 data_dict 用于数据库操作，因为 ORM 可以处理外键实例
                data_dicts.append(data_dict)

                # 2. 准备用于缓存的字典
                # 创建 data_dict 的副本以进行修改，避免影响原始字典
                cache_data_dict = data_dict.copy()
                # --- 手动处理 stock 字段 ---
                if 'stock' in cache_data_dict and isinstance(cache_data_dict['stock'], StockInfo):
                    # 将 StockInfo 实例替换为 stock_code 
                    cache_data_dict['stock_code'] = cache_data_dict['stock'].stock_code
                    # 删除原始的 stock 实例键，避免混淆
                    del cache_data_dict['stock']
                # --- 结束手动处理 ---

                # 3. 调用 _prepare_data_for_cache 处理其他类型
                # 现在 cache_data_dict 中不再包含模型实例，只有基本类型或 stock_code
                prepared_cache_data = await self._prepare_data_for_cache(cache_data_dict, related_field_map=None)

                if prepared_cache_data: # 确保准备成功
                    # 使用 stock_code 作为缓存键，缓存处理后的数据
                    await self.cache_set.latest_realtime_data(stock_code, prepared_cache_data)
                else:
                    logger.warning(f"为股票 {stock_code} 准备缓存数据失败，跳过缓存写入。原始数据: {data_dict}")

                # 保存数据
                result = await self._save_all_to_db_native_upsert(
                    model_class=StockRealtimeData,
                    data_list=data_dicts,
                    unique_fields=['stock', 'trade_time']
                )
                if result.get('创建/更新成功') > 0:
                    logger.info(f"股票[{stock}]实时数据保存完成，结果: {result}")
                return result
            else:
                return {'创建': 0, '更新': 0, '跳过': 0}
        except Exception as e:
            logger.error(f"保存股票[{stock}]实时数据失败: {str(e)}", exc_info=True)
            return None

    async def fetch_and_save_favorite_stocks_realtime_data(self) -> Dict:
        """
        获取并保存所有自选股票的最新实时数据
        Returns:
            Dict[str, StockRealtimeData]: 股票代码到实时数据的映射
        """
        try:
            if self.cache_set is None:
                await self.initialize_cache_objects()
            # 获取指数信息
            favorite_stocks = await self.stock_basic_dao.get_all_favorite_stocks()
            for stock in favorite_stocks:
                await self.fetch_and_save_realtime_data(stock.stock_code)
        except Exception as e:
            logger.error(f"保存股票[{stock}]实时数据失败: {str(e)}")
            return None

    async def fetch_and_save_all_realtime_data(self) -> Dict[str, Dict]:
        """
        获取并保存所有股票的最新实时数据，使用批量并发触发，但增加并发数量限制
        Returns:
            Dict[str, Dict]: 股票代码到处理结果的映射，例如 {'stock_code': {'创建': 0, '更新': 0, '跳过': 0, '数据': StockRealtimeData}}
        """
        try:
            if self.cache_set is None:
                await self.initialize_cache_objects()
            stocks = await self.stock_basic_dao.get_stock_list()
            if not stocks:
                logger.warning("股票列表不存在，无法获取实时数据")
                return {}  # 返回空字典
            data_dicts_to_save = [] # 用于数据库批量保存
            cache_tasks = [] # 用于异步缓存写入
            process_start_time = time_lib.time()
            stocks_count = len(stocks)
            finished_count = 0
            for i, stock in enumerate(stocks):
                loop_start_time = time_lib.time()
                api_start_time = time_lib.time()
                api_data = await self.api.get_realtime_data(stock.stock_code)
                api_end_time = time_lib.time()
                api_call_duration = api_end_time - api_start_time
                if not api_data:
                    logger.warning(f"API未返回股票[{stock.stock_code}]的实时数据")
                    total_loop_duration = time_lib.time() - loop_start_time
                    sleep_time = max(0, 0.02 - total_loop_duration)
                    await asyncio.sleep(sleep_time)
                    continue
                if process_start_time is None:
                    process_start_time = time_lib.time()
                # data_dict 包含 StockInfo 实例
                data_dict = self.data_format_process.set_realtime_data(stock, api_data)
                if data_dict.get('trade_time') is not None:
                    # 1. 添加到数据库保存列表 (包含 StockInfo 实例)
                    data_dicts_to_save.append(data_dict)
                    # 2. 准备缓存数据
                    cache_data_dict = data_dict.copy()
                    if 'stock' in cache_data_dict and isinstance(cache_data_dict['stock'], StockInfo):
                        # 替换为 stock_code
                        cache_data_dict['stock_code'] = cache_data_dict['stock'].stock_code
                        del cache_data_dict['stock'] # 删除实例键
                    prepared_data = await self._prepare_data_for_cache(cache_data_dict, related_field_map=None)
                    if prepared_data:
                        await self.cache_set.latest_realtime_data(stock.stock_code, prepared_data)
                    else:
                        logger.warning(f"为股票 {stock.stock_code} 准备缓存数据失败，跳过缓存写入。原始数据: {data_dict}")
                total_loop_duration = time_lib.time() - loop_start_time
                if i % 200 == 0 and i > 10:
                    # --- 批量保存到数据库 ---
                    if data_dicts_to_save:
                        # 使用包含 StockInfo 实例的列表
                        result = await self._save_all_to_db_native_upsert(
                            model_class=StockRealtimeData,
                            data_list=data_dicts_to_save,
                            unique_fields=['stock', 'trade_time'] # ORM 能处理 stock 实例
                        )
                        process_end_time = time_lib.time()
                        process_duration = process_end_time - process_start_time
                        finished_count += len(data_dicts_to_save)
                        logger.info(f"{finished_count} / {stocks_count} 个股票实时数据保存完成, 耗时: {process_duration} 秒，平均每秒处理 {finished_count / process_duration} 个股票")
                        data_dicts_to_save = []
                        process_start_time = None
                    else:
                        logger.info("没有需要保存到数据库的股票实时数据。")
                        return {'尝试处理': 0, '失败': 0, '创建/更新成功': 0}
                sleep_time = max(0, 0.02 - total_loop_duration)
                await asyncio.sleep(sleep_time)

            return result
            

        except Exception as e:
            logger.error(f"获取并保存所有股票实时数据失败: {str(e)}", exc_info=True)
            return {} # 返回空字典表示整体失败
    
    # ================= Level5Data相关方法 =================
    async def get_level5_data_by_code(self, stock_code: str) -> Optional[StockLevel5Data]:
        """
        获取股票最新的Level5数据
        Args:
            stock_code: 股票代码
        Returns:
            Optional[StockLevel5Data]: 最新的Level5数据，如不存在则返回None
        """
        try:
            if self.cache_get is None:
                await self.initialize_cache_objects()
            cache_data = await self.cache_get.latest_level5_data(stock_code)
            if cache_data:
                level5_data_dict = json.loads(cache_data)
                level5_data = StockLevel5Data(**level5_data_dict)
                return level5_data
        except Exception as e:
            logger.error(f"从缓存获取最新股票[{stock_code}]Level5数据时发生异常: {str(e)}")
            return None
        
        # 从数据库获取最新数据
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return None
        try:
            if self.cache_set is None:
                await self.initialize_cache_objects()
            data = await sync_to_async(lambda: StockLevel5Data.objects.filter(stock=stock).order_by('-trade_time').first())()
            # 检查数据是否过期（超过2分钟）
            if data and (timezone.now() - data.trade_time).total_seconds() < 120:
                self.cache_set.latest_level5_data(stock_code, data)
                return data
        except Exception as e:
            logger.error(f"从数据库获取最新股票[{stock}]Level5数据失败: {str(e)}")
            return None
        
        # 数据不存在或已过期，从API获取新数据
        logger.info(f"股票[{stock}]Level5数据不存在或已过期，从API获取")
        await self.fetch_and_save_level5_data(stock_code)
        data = await sync_to_async(lambda: StockLevel5Data.objects.filter(stock=stock).order_by('-trade_time').first())()
        return data
    
    async def fetch_and_save_level5_data(self, stock_code: str) -> Dict:
        """
        从API获取Level5数据并保存到数据库
        Args:
            stock_code: 股票代码
        Returns:
            Optional[StockLevel5Data]: 保存后的Level5数据
        """
        try:
            if self.cache_set is None:
                await self.initialize_cache_objects()
            stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
            if not stock:
                return {'创建': 0, '更新': 0, '跳过': 0}
            data_dicts = []
            api_data = await self.api.get_level5_data(stock.stock_code)
            data_dict = self.data_format_process.set_level5_data(stock, api_data)
            data_dicts.append(data_dict)
            cache_dict = data_dict.copy()
            await self.cache_set.latest_level5_data(stock_code, cache_dict)
            if not api_data:
                logger.warning(f"API未返回{stock}的Level5数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
           # 保存数据
            result = await self._save_all_to_db_native_upsert(
                model_class=StockLevel5Data,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time']
            )
            logger.info(f"{stock}股票Level5数据保存完成，结果: {result}")
            return result            
        except Exception as e:
            logger.error(f"从API获取Level5数据出错: {e}")
            return None

    async def fetch_and_save_favorite_stocks_level5_data(self) -> Dict:
        """
        获取并保存所有自选股票的最新Level5数据
        """
        try:
            favorite_stocks = await self.stock_basic_dao.get_all_favorite_stocks()
            for stock in favorite_stocks:
                await self.fetch_and_save_level5_data(stock.stock_code)
        except Exception as e:
            logger.error(f"保存股票[{stock}]Level5数据失败: {str(e)}")
            return None
    
    async def fetch_and_save_all_level5_data(self) -> Dict:
        """
        获取并保存所有股票的最新Level5数据
        """
        try:
            stocks = await self.stock_basic_dao.get_stock_list()
            for stock in stocks:
                await self.fetch_and_save_level5_data(stock.stock_code)
        except Exception as e:
            logger.error(f"保存股票[{stock}]Level5数据失败: {str(e)}")
            return None

    # ================= TradeDetail相关方法 =================
    async def get_trade_details_by_code_and_date(self, stock_code: str, trade_date: Optional[date] = None) -> List[StockTradeDetail]:
        """
        获取股票指定日期的交易明细
        Args:
            stock_code: 股票代码
            trade_date: 交易日期，默认为当天
        Returns:
            List[StockTradeDetail]: 交易明细列表
        """
        if not trade_date:
            trade_date = datetime.now().date()
        try:
            if self.cache_get is None:
                await self.initialize_cache_objects()
            cache_data = await self.cache_get.history_onebyone_trade(stock_code, trade_date)
            if cache_data:
                trade_detail_dict = json.loads(cache_data)
                trade_detail = StockTradeDetail(**trade_detail_dict)
                return trade_detail
        except Exception as e:
            logger.error(f"从缓存获取股票[{stock_code}]交易明细数据时发生异常: {str(e)}")
            return None
        
        # 从数据库获取最新数据
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return None
        try:
            if self.cache_set is None:
                await self.initialize_cache_objects()
            data = await sync_to_async(lambda: StockTradeDetail.objects.filter(stock=stock, trade_date=trade_date).order_by('-trade_time').first())()
            # 检查数据是否过期（超过2分钟）
            if data and (timezone.now() - data.trade_time).total_seconds() < 120:
                self.cache_set.onebyone_trade(stock_code, trade_date, data)
                return data
        except Exception as e:
            logger.error(f"从数据库获取最新股票[{stock}]交易明细数据失败: {str(e)}")
            return None
        
        # 数据不存在或已过期，从API获取新数据
        logger.info(f"股票[{stock}]交易明细数据不存在或已过期，从API获取")
        await self.fetch_and_save_trade_detail(stock_code)
        data = await sync_to_async(lambda: StockTradeDetail.objects.filter(stock=stock, trade_date=trade_date).order_by('-trade_time').first())()
        return data
    
    async def fetch_and_save_trade_detail(self, stock_code: str) -> Dict:
        """
        处理并保存交易明细数据
        Args:
            stock_code: 股票代码
        Returns:
           Dict: 保存后的交易明细列表
        """
        result = []
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        data_dicts = []
        try:
            if self.cache_set is None:
                await self.initialize_cache_objects()
            api_datas = await self.api.get_onebyone_trades(stock.stock_code)
            if not api_datas:
                logger.warning(f"API未返回{stock}的逐笔交易数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            data_dicts = []
            for api_data in api_datas:
                data_dict = self.data_format_process.set_onebyone_trade_data(stock, api_data)
                data_dicts.append(data_dict)
                # cache_dict = data_dict.copy()
                # await self.cache_set.onebyone_trade(stock.stock_code, cache_dict)
            # 保存数据
            result = await self._save_all_to_db_native_upsert(
                model_class=StockTradeDetail,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_date', 'trade_time']
            )
            logger.info(f"{stock}股票逐笔交易数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存{stock}股票逐笔交易数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}
    
    async def fetch_and_save_favorite_stocks_trade_detail(self) -> Dict:
        """
        获取并保存所有自选股票的最新逐笔交易数据
        """
        try:
            favorite_stocks = await self.stock_basic_dao.get_all_favorite_stocks()
            for stock in favorite_stocks:
                await self.fetch_and_save_trade_detail(stock.stock_code)
        except Exception as e:
            logger.error(f"保存股票[{stock}]逐笔交易数据失败: {str(e)}")
            return None
    
    async def fetch_and_save_all_trade_detail(self) -> Dict:
        """
        获取并保存所有股票的最新逐笔交易数据
        """
        try:
            stocks = await self.stock_basic_dao.get_stock_list()
            for stock in stocks:
                await self.fetch_and_save_trade_detail(stock.stock_code)
        except Exception as e:
            logger.error(f"保存股票[{stock}]逐笔交易数据失败: {str(e)}")
            return None
    
    # ================= TimeDeal相关方法 =================

    async def get_daily_time_deals(self, stock_code: str, trade_date: Optional[date] = None) -> List[StockTimeDeal]:
        """
        获取股票指定日期的分时成交数据
        Args:
            stock_code: 股票代码
            trade_date: 交易日期，默认为当天
        Returns:
            List[StockTimeDeal]: 分时成交数据列表
        """
        if not trade_date:
            trade_date = datetime.now().date()
            
        try:
            if self.cache_get is None:
                await self.initialize_cache_objects()
            cache_data = await self.cache_get.history_time_deal(stock_code)
            if cache_data:
                time_deal_dict = json.loads(cache_data)
                time_deal = StockTimeDeal(**time_deal_dict)
                return time_deal
        except Exception as e:
            logger.error(f"从缓存获取股票[{stock_code}]分时成交数据时发生异常: {str(e)}")
            return None
        # 从数据库获取最新数据
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return None
        try:
            if self.cache_set is None:
                await self.initialize_cache_objects()
            data = await sync_to_async(lambda: StockTimeDeal.objects.filter(stock=stock, trade_date=trade_date).order_by('-trade_time').first())()
            # 检查数据是否过期（超过2分钟）
            if data and (timezone.now() - data.trade_time).total_seconds() < 120:
                self.cache_set.time_deal(stock_code, trade_date, data)
                return data
        except Exception as e:
            logger.error(f"从数据库获取最新股票[{stock}]分时成交数据失败: {str(e)}")
            return None
        
        # 数据不存在或已过期，从API获取新数据
        logger.info(f"股票[{stock}]分时成交数据不存在或已过期，从API获取")
        await self.fetch_and_save_time_deals(stock_code)
        data = await sync_to_async(lambda: StockTimeDeal.objects.filter(stock=stock, trade_date=trade_date).order_by('-trade_time').first())()
        return data

    async def get_latest_time_deal(self, stock_code: str) -> Optional[StockTimeDeal]:
        """
        获取股票最新的分时成交数据
        Args:
            stock_code: 股票代码
        Returns:
            Optional[StockTimeDeal]: 最新的分时成交数据，如不存在则返回None
        """
        time_deal = None
        try:
            if self.cache_get is None:
                await self.initialize_cache_objects()
            # 从缓存获取最新数据
            cache_data = await self.cache_get.latest_time_deal(stock_code)
            if cache_data:
                time_deal_dict = json.loads(cache_data)
                time_deal = StockTimeDeal(**time_deal_dict)
                return time_deal
        except Exception as e:
            logger.error(f"从缓存获取最新股票[{stock_code}]分时成交数据时发生异常: {str(e)}")
            return None
        # 从数据库获取最新数据
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return None
        try:
            if self.cache_set is None:
                await self.initialize_cache_objects()
            data = await sync_to_async(lambda: StockTimeDeal.objects.filter(stock=stock).order_by('-trade_time').first())()
            time_deal = self.data_format_process.set_time_deal_data(stock, data)
        except Exception as e:
            logger.error(f"从数据库获取最新股票[{stock}]分时成交数据失败: {str(e)}")
            return None
        # 数据不存在或已过期，从API获取新数据
        logger.info(f"股票[{stock}]分时成交数据不存在或已过期，从API获取")
        await self.fetch_and_save_time_deals(stock_code)
        data = await sync_to_async(lambda: StockTimeDeal.objects.filter(stock=stock).order_by('-trade_time').first())()
        return data
    
    async def fetch_and_save_time_deals(self, stock_code: str) -> Dict:
        """
        批量保存分时成交明细数据 (修正版：使用 async with 管理 API 实例)
        Args:
            stock_code: 股票代码
        Returns:
            Dict: 保存结果统计
        """
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            logger.warning(f"股票代码[{stock_code}]不存在，无法获取分时成交明细")
            return {'创建': 0, '更新': 0, '跳过': 0}
        data_dicts = []
        result = {'创建': 0, '更新': 0, '跳过': 0} # 初始化 result
        # --- 使用 async with 创建和管理 API 实例 ---
        try:
            if self.cache_set is None:
                await self.initialize_cache_objects()
            async with self.api as api_client: # 在这里创建临时的 API 客户端实例
                api_datas = await api_client.get_time_deal(stock.stock_code) # 假设返回列表
                if not isinstance(api_datas, list):
                    logger.warning(f"API未返回 {stock.stock_code} 的分时成交明细列表，收到类型: {type(api_datas)}")
                    # 检查是否是错误字典
                    if isinstance(api_datas, dict) and 'error' in api_datas:
                         logger.error(f"API返回错误: {api_datas['error']}")
                    return {'创建': 0, '更新': 0, '跳过': 0} # 如果不是列表，则无法处理

                for api_data in api_datas:
                    try:
                        data_dict = self.data_format_process.set_time_deal_data(stock, api_data)
                        data_dicts.append(data_dict)
                        # cache_dict = data_dict.copy()
                        # await self.cache_set.time_deal(stock_code, cache_dict)
                    except Exception as inner_e:
                         # 捕获单条数据处理错误
                         logger.error(f"处理 {stock.stock_code} 的单条成交明细时出错: {str(inner_e)} - 数据: {api_data}", exc_info=True)
                         continue # 继续处理下一条
            # --- async with 块结束，api_client 会自动关闭 ---
            if not data_dicts:
                logger.warning(f"未能成功处理 {stock.stock_code} 的任何分时成交明细数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            # 批量保存数据
            result = await self._save_all_to_db_native_upsert(
                model_class=StockTimeDeal, # 确保 StockTimeDeal 已导入
                data_list=data_dicts,
                unique_fields=['stock', 'trade_date', 'trade_time']
            )
            # --- 1. 获取关注此股票的用户 ID ---
            favorited_user_ids = await self._get_favorited_user_ids(stock.stock_code)
            # --- 2. 准备并推送 WebSocket 更新 ---
            if data_dict and favorited_user_ids: # 确保有最新数据和需要通知的用户
                channel_layer = get_channel_layer()
                payload_data = {
                    'code': stock.stock_code,
                    # 'name': stock.stock_name, # 前端可能已经有名称，或者需要传递
                    'latest_price': data_dict.get('price'),
                    # 格式化时间为 HH:MM:SS
                    'trade_time': self._parse_datetime(data_dict.get('trade_time')) if data_dict.get('trade_time') else None,
                    'volume': data_dict.get('volume'),
                    # 注意：这里的 'change_percent' 和 'signal' 无法从此数据直接获得
                    # 需要由其他任务（如计算指标、执行策略的任务）来推送
                    # 'change_percent': None,
                    # 'signal': None,
                }
                message_data = {
                    'type': 'user.message', # 对应 Consumer 中的 user_message 方法
                    'data': {
                        'sub_type': 'stock_update', # 让 JS 知道这是股票数据更新
                        'payload': payload_data,
                    }
                }

                for user_id in favorited_user_ids:
                    try:
                        await channel_layer.group_send(
                            f'user_{user_id}', # 发送到特定用户的组
                            message_data
                        )
                        logger.debug(f"成功推送 {stock_code} 更新给用户 {user_id}")
                    except Exception as push_error:
                        # 记录推送错误，但不应中断整个流程
                        logger.error(f"推送 {stock_code} 更新给用户 {user_id} 时失败: {push_error}", exc_info=False)

            logger.info(f"{stock.stock_code} 股票分时成交明细数据保存完成，结果: {result}")
        except Exception as e:
            # 捕获 async with 外部或数据库保存过程中的错误
            logger.error(f"保存 {stock.stock_code} 股票分时成交明细数据过程中发生意外错误: {str(e)}", exc_info=True)
            result = {'创建': 0, '更新': 0, '跳过': 0} # 确保返回字典
        return result
    
    async def fetch_and_save_favorite_stocks_time_deals(self) -> Dict:
        """
        获取并保存所有自选股票的最新分时成交数据
        """
        try:
            favorite_stocks = await self.stock_basic_dao.get_all_favorite_stocks()
            for stock in favorite_stocks:
                await self.fetch_and_save_time_deals(stock.stock_code)
        except Exception as e:
            logger.error(f"保存股票[{stock}]分时成交数据失败: {str(e)}")
            return None

    async def fetch_and_save_all_time_deals(self) -> Dict:
        """
        获取并保存所有股票的最新分时成交数据
        """
        try:
            stocks = await self.stock_basic_dao.get_stock_list()
            for stock in stocks:
                await self.fetch_and_save_time_deals(stock.stock_code)
        except Exception as e:
            logger.error(f"保存股票[{stock}]分时成交数据失败: {str(e)}")
            return None

    # ================= RealPercent相关方法 =================

    async def fetch_and_save_real_percent(self, stock_code: str) -> Dict:
        """
        批量保存分价成交占比数据
        Args:
            stock_code: 股票代码
        Returns:
            List[StockPricePercent]: 保存后的分价成交占比数据列表
        """
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        data_dicts = []
        try:
            if self.cache_set is None:
                await self.initialize_cache_objects()
            api_datas = await self.api.get_real_percent(stock.stock_code)
            for api_data in api_datas:
                data_dict = self.data_format_process.set_real_percent_data(stock, api_data)
                data_dicts.append(data_dict)
                # cache_dict = data_dict.copy()
                # await self.cache_set.real_percent(stock_code, cache_dict)
            result = await self._save_all_to_db_native_upsert(
                model_class=StockPricePercent,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_date', 'price']
            )
            logger.info(f"{stock}股票分价成交占比数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存分价成交占比数据出错: {e}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_favorite_stocks_real_percent(self) -> Dict:
        """
        获取并保存所有自选股票的最新分价成交占比数据
        """
        try:
            favorite_stocks = await self.stock_basic_dao.get_all_favorite_stocks()
            for stock in favorite_stocks:
                await self.fetch_and_save_real_percent(stock.stock_code)
        except Exception as e:
            logger.error(f"保存股票[{stock}]分价成交占比数据失败: {str(e)}")
            return None
        
    async def fetch_and_save_all_real_percent(self) -> Dict:
        """
        获取并保存所有股票的最新分价成交占比数据
        """
        try:
            stocks = await self.stock_basic_dao.get_stock_list()
            for stock in stocks:
                await self.fetch_and_save_real_percent(stock.stock_code)
        except Exception as e:
            logger.error(f"保存股票[{stock}]分价成交占比数据失败: {str(e)}")
            return None


    # ================= BigDeal相关方法 =================
    
    async def fetch_and_save_big_deal(self, stock_code: str) -> Dict:
        """
        批量保存逐笔大单交易数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[StockBigDeal]: 保存后的逐笔大单交易数据列表
        """
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        data_dicts = []
        try:
            if self.cache_set is None:
                await self.initialize_cache_objects()
            api_datas = await self.api.get_big_deal(stock.stock_code)
            for api_data in api_datas:
                data_dict = self.data_format_process.set_big_deal_data(stock, api_data)
                data_dicts.append(data_dict)
                cache_dict = data_dict.copy()
                await self.cache_set.big_deal(stock_code, cache_dict)
            result = await self._save_all_to_db_native_upsert(
                model_class=StockBigDeal,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_date', 'trade_time']
            )
            logger.info(f"{stock}股票逐笔大单交易数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存逐笔大单交易数据出错: {e}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_favorite_stocks_big_deal(self) -> Dict:
        """
        获取并保存所有自选股票的最新逐笔大单交易数据
        """
        try:
            favorite_stocks = await self.stock_basic_dao.get_all_favorite_stocks()
            for stock in favorite_stocks:
                await self.fetch_and_save_big_deal(stock.stock_code)
        except Exception as e:
            logger.error(f"保存股票[{stock}]逐笔大单交易数据失败: {str(e)}")
            return None
        
    async def fetch_and_save_all_big_deal(self) -> Dict:
        """
        获取并保存所有股票的最新逐笔大单交易数据
        """
        try:
            stocks = await self.stock_basic_dao.get_stock_list()
            for stock in stocks:
                await self.fetch_and_save_big_deal(stock.stock_code)
        except Exception as e:
            logger.error(f"保存股票[{stock}]逐笔大单交易数据失败: {str(e)}")
            return None

    # ================= AbnormalMovement相关方法 =================  
    
    async def fetch_and_save_abnormal_movements(self) -> Dict:
        """
        批量保存盘中异动数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[StockAbnormalMovement]: 保存后的盘中异动数据列表
        """
        data_dicts = []
        try:
            if self.cache_set is None:
                await self.initialize_cache_objects()
            api_datas = await self.api.get_abnormal_movements()
            for api_data in api_datas:
                data_dict = self.data_format_process.set_abnormal_movement_data(api_data)
                data_dicts.append(data_dict)
                cache_dict = data_dict.copy()
                await self.cache_set.abnormal_movement(cache_dict)
            result = await self._save_all_to_db_native_upsert(
                model_class=StockAbnormalMovement,
                data_list=data_dicts,
                unique_fields=['stock', 'movement_time', 'movement_type']
            )
            logger.info(f"盘中异动数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存盘中异动数据出错: {e}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    # ================= 其他相关方法 =================
    # 由于代码过长，这些方法的实现与上面的方法类似，按照相同的模式实现即可
