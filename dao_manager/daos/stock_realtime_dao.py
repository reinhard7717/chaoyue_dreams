from decimal import Decimal
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set, TypeVar, Generic, Type
from datetime import datetime, date, time
import json
from django.utils import timezone
from asgiref.sync import sync_to_async
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
        self.cache_manager = CacheManager()  # 初始化缓存管理器
        self.data_format_process = StockRealtimeDataFormatProcess()
        self.cache_get = StockRealtimeCacheGet()
        self.cache_set = StockRealtimeCacheSet()
        self.cache_key = StockCashKey()

    # 新增 close 方法
    async def close(self):
        """关闭内部持有的 API Client Session"""
        if hasattr(self, 'api') and self.api:
            # logger.debug("Closing StockRealtimeDAO's internal API client...") # 可选日志
            await self.api.close() # 调用 StockRealtimeAPI 的 close 方法
            # logger.debug("StockRealtimeDAO's internal API client closed.") # 可选日志
        else:
            # logger.debug("StockRealtimeDAO has no API client to close or it's already None.") # 可选日志
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
        cache_data = await self.cache_get.latest_realtime_data(stock_code)
        if cache_data:
            realtime_data_dict = json.loads(cache_data)
            realtime_data = StockRealtimeData(**realtime_data_dict)
            return realtime_data
        
        # 从数据库获取最新数据
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return None
        try:
            data = await sync_to_async(lambda: StockRealtimeData.objects.filter(stock=stock).order_by('-trade_time').first())()
            # 检查数据是否过期（超过2分钟）
            if data and (timezone.now() - data.trade_time).total_seconds() < 120:
                self.cache_set.latest_realtime_data(stock_code, data)
                return data
        except Exception as e:
            logger.error(f"从数据库获取最新股票[{stock}]实时数据失败: {str(e)}")
            return None
        
        # 数据不存在或已过期，从API获取新数据
        logger.info(f"股票[{stock}]实时数据不存在或已过期，从API获取")
        await self.fetch_and_save_realtime_data(stock_code)
        data = await sync_to_async(lambda: StockRealtimeData.objects.filter(stock=stock).order_by('-trade_time').first())()
        return data
    
    async def fetch_and_save_realtime_data(self, stock_code: str) -> Dict:
        """
        从API获取实时交易数据并保存到数据库
        Args:
            stock_code: 股票代码
        Returns:
            Optional[StockRealtimeData]: 保存后的实时交易数据，保存失败则返回None
        """
        try:
            # 获取指数信息
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
            data_dict = await self.data_format_process.set_realtime_data(stock, api_data)
            data_dicts.append(data_dict)
            self.cache_set.latest_realtime_data(stock_code, data_dict)
            # 保存数据
            result = await self._save_all_to_db_native_upsert(
                model_class=StockRealtimeData,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time']
            )
            return result
        except Exception as e:
            logger.error(f"保存股票[{stock}]实时数据失败: {str(e)}")
            return None

    async def fetch_and_save_favorite_stocks_realtime_data(self) -> Dict:
        """
        获取并保存所有自选股票的最新实时数据
        
        Returns:
            Dict[str, StockRealtimeData]: 股票代码到实时数据的映射
        """
        try:
            # 获取指数信息
            favorite_stocks = await self.stock_basic_dao.get_all_favorite_stocks()
            for stock in favorite_stocks:
                await self.fetch_and_save_realtime_data(stock.stock_code)
        except Exception as e:
            logger.error(f"保存股票[{stock}]实时数据失败: {str(e)}")
            return None

    async def fetch_and_save_all_realtime_data(self) -> Dict:
        """
        获取并保存所有股票的最新实时数据
        Returns:
            Dict[str, StockRealtimeData]: 股票代码到实时数据的映射
        """
        try:
            # 获取指数信息
            stocks = await self.stock_basic_dao.get_stock_list()
            for stock in stocks:
                await self.fetch_and_save_realtime_data(stock.stock_code)
        except Exception as e:
            logger.error(f"保存股票[{stock}]实时数据失败: {str(e)}")
            return None
    
    # ================= Level5Data相关方法 =================
    async def get_level5_data_by_code(self, stock_code: str) -> Optional[StockLevel5Data]:
        """
        获取股票最新的Level5数据
        Args:
            stock_code: 股票代码
        Returns:
            Optional[StockLevel5Data]: 最新的Level5数据，如不存在则返回None
        """
        cache_data = await self.cache_get.latest_level5_data(stock_code)
        if cache_data:
            level5_data_dict = json.loads(cache_data)
            level5_data = StockLevel5Data(**level5_data_dict)
            return level5_data
        
        # 从数据库获取最新数据
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return None
        try:
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
            stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
            if not stock:
                return {'创建': 0, '更新': 0, '跳过': 0}
            data_dicts = []
            api_data = await self.api.get_level5_data(stock.stock_code)
            data_dict = await self.data_format_process.set_level5_data(stock, api_data)
            data_dicts.append(data_dict)
            self.cache_set.latest_level5_data(stock_code, data_dict)
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
            
        cache_data = await self.cache_get.history_onebyone_trade(stock_code, trade_date)
        if cache_data:
            trade_detail_dict = json.loads(cache_data)
            trade_detail = StockTradeDetail(**trade_detail_dict)
            return trade_detail
        
        # 从数据库获取最新数据
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return None
        try:
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
            api_datas = await self.api.get_onebyone_trades(stock.stock_code)
            if not api_datas:
                logger.warning(f"API未返回{stock}的逐笔交易数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            data_dicts = []
            for api_data in api_datas:
                data_dict = await self.data_format_process.set_onebyone_trade_data(stock, api_data)
                data_dicts.append(data_dict)
                self.cache_set.onebyone_trade(stock.stock_code, data_dict)
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
            
        cache_data = await self.cache_get.history_time_deal(stock_code)
        if cache_data:
            time_deal_dict = json.loads(cache_data)
            time_deal = StockTimeDeal(**time_deal_dict)
            return time_deal
        
        # 从数据库获取最新数据
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return None
        try:
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

    async def fetch_and_save_time_deals(self, stock_code: str) -> Dict:
        """
        批量保存分时成交数据
        Args:
            stock_code: 股票代码
        Returns:
            List[StockTimeDeal]: 保存后的分时成交数据列表
        """
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        data_dicts = []
        try:
            api_datas = await self.api.get_time_deal(stock.stock_code)
            for api_data in api_datas:
                data_dict = await self.data_format_process.set_time_deal_data(stock, api_data)
                data_dicts.append(data_dict)
                self.cache_set.time_deal(stock_code, data_dict)
            result = await self._save_all_to_db_native_upsert(
                model_class=StockTimeDeal,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_date', 'trade_time']
            )
            logger.info(f"{stock} 股票分时成交数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存分时成交数据出错: {e}")
            return {'创建': 0, '更新': 0, '跳过': 0}

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
            api_datas = await self.api.get_real_percent(stock.stock_code)
            for api_data in api_datas:
                data_dict = await self.data_format_process.set_real_percent_data(stock, api_data)
                data_dicts.append(data_dict)
                self.cache_set.real_percent(stock_code, data_dict)
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
            api_datas = await self.api.get_big_deal(stock.stock_code)
            for api_data in api_datas:
                data_dict = await self.data_format_process.set_big_deal_data(stock, api_data)
                data_dicts.append(data_dict)
                self.cache_set.big_deal(stock_code, data_dict)
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
            api_datas = await self.api.get_abnormal_movements()
            for api_data in api_datas:
                data_dict = await self.data_format_process.set_abnormal_movement_data(api_data)
                data_dicts.append(data_dict)
                self.cache_set.abnormal_movement(data_dict)
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
