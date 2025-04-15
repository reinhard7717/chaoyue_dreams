import json
import logging
import asyncio
from asyncio import Semaphore
from typing import Dict, List, Any, Optional, Union, Set, Tuple, Type
from datetime import datetime, date
from asgiref.sync import sync_to_async
from core.constants import TIME_TEADE_TIME_LEVELS, TIME_TEADE_TIME_LEVELS_LITE, TIME_TEADE_TIME_LEVELS_PER_TRADING
from stock_models.indicator.boll import StockBOLLIndicator
from stock_models.indicator.kdj import StockKDJIndicator
from stock_models.indicator.ma import StockMAIndicator
from stock_models.indicator.macd import StockMACDIndicator
from utils import cache_constants as cc # 导入常量
from django.utils import timezone

from api_manager.apis.stock_indicators_api import StockIndicatorsAPI, TimeLevel
from api_manager.mappings.stock_indicators_mapping import BOLL_INDICATOR_MAPPING, KDJ_INDICATOR_MAPPING, MA_INDICATOR_MAPPING, MACD_INDICATOR_MAPPING, TIME_TRADE_MAPPING
from dao_manager.base_dao import BaseDAO
from dao_manager.daos.stock_basic_dao import StockBasicDAO
from dao_manager.daos.user_dao import UserDAO
from stock_models.stock_basic import StockInfo, StockTimeTrade

from utils.cache_get import StockIndicatorsCacheGet
from utils.cache_manager import CacheManager
from utils.cache_set import StockIndicatorsCacheSet
from utils.cash_key import StockCashKey
from utils.data_format_process import StockIndicatorsDataFormatProcess

logger = logging.getLogger("dao")


class StockIndicatorsDAO(BaseDAO):
    """
    股票技术指标DAO，整合所有相关的技术指标访问功能
    """
    def __init__(self):
        """初始化StockIndicatorsDAO"""
        super().__init__(None, None, 3600)  # 基类使用None作为model_class，因为本DAO管理多个模型
        self.api = StockIndicatorsAPI()
        self.cache_manager = CacheManager()
        self.stock_basic_dao = StockBasicDAO()
        self.cache_timeout = 300  # 默认缓存5分钟
        self.cache_limit = 333 # 定义缓存数量上限
        self.user_dao = UserDAO()
        self.cache_key = StockCashKey()
        self.data_format_process = StockIndicatorsDataFormatProcess()
        self.cache_get = StockIndicatorsCacheGet()
        self.cache_set = StockIndicatorsCacheSet()

    # 新增 close 方法
    async def close(self):
        """关闭内部持有的 API Client Session"""
        if hasattr(self, 'api') and self.api:
            # logger.debug("Closing StockIndicatorsDAO's internal API client...") # 可选日志
            await self.api.close() # 调用 StockIndicatorsAPI 的 close 方法
            # logger.debug("StockIndicatorsDAO's internal API client closed.") # 可选日志
        else:
            # logger.debug("StockIndicatorsDAO has no API client to close or it's already None.") # 可选日志
            pass
    # ================= 分时成交数据相关方法 =================
    async def get_latest_time_trade(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Optional[StockTimeTrade]:
        """
        获取最新的分时成交数据
        Args:
            stock_code: 股票代码
            time_level: 时间级别
        Returns:
            Optional[StockTimeTrade]: 最新的分时成交数据
        """
        cache_data = await self.cache_get.latest_time_trade(stock_code, time_level)
        if cache_data:
            time_trade_dict = json.loads(cache_data)
            time_trade = StockTimeTrade(**time_trade_dict)
            return time_trade
        
        # 从数据库获取最新数据
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return None
        try:
            data = await sync_to_async(lambda: StockTimeTrade.objects.filter(stock=stock, time_level=time_level).order_by('-trade_time').first())()
            # 检查数据是否过期（超过2分钟）
            if data and (timezone.now() - data.trade_time).total_seconds() < 120:
                await self.cache_set.latest_time_trade(stock_code, time_level, data)
                return data
        except Exception as e:
            logger.error(f"从数据库获取最新股票[{stock}]{time_level}级别分时成交数据失败: {str(e)}")
            return None
        
        # 数据不存在或已过期，从API获取新数据
        logger.info(f"股票[{stock}]{time_level}级别分时成交数据不存在或已过期，从API获取")
        await self.fetch_and_save_latest_time_trade(stock_code, time_level)
        data = await sync_to_async(lambda: StockTimeTrade.objects.filter(stock=stock, time_level=time_level).order_by('-trade_time').first())()
        return data
    
    async def get_all_latest_time_trade_by_stock_code(self, stock_code: str) -> Optional[StockTimeTrade]:
        """
        获取指定股票的最新分时成交数据
        Args:
            stock_code: 股票代码
        Returns:
            Optional[StockTimeTrade]: 指定股票的最新分时成交数据
        """
        datas = []
        for time_level in TIME_TEADE_TIME_LEVELS:
            data = await self.get_latest_time_trade(stock_code, time_level)
            if data:
                datas.append(data)
        return datas

    async def get_favorite_stocks_latest_time_trade(self) -> Optional[StockTimeTrade]:
        """
        获取自选股最新分时成交数据
        
        Returns:
            Optional[StockTimeTrade]: 自选股最新分时成交数据
        """
        # 获取自选股
        datas = []
        favorite_stocks = await self.user_dao.get_all_favorite_stocks()
        # 获取自选股最新分时成交数据
        for stock in favorite_stocks:
            for time_level in TIME_TEADE_TIME_LEVELS:
                data = await self.get_latest_time_trade(stock.stock_code, time_level)
                if data:
                    datas.append(data)
        return datas

    async def get_history_time_trades(self, stock_code: str, time_level: Union[TimeLevel, str], 
                                    start_time: datetime, end_time: datetime, limit: int = 1000) -> List[StockTimeTrade]:
        """
        获取历史分时成交数据
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            limit: 返回记录数量限制
            
        Returns:
            List[StockTimeTrade]: 历史分时成交数据列表
        """
        cache_data = await self.cache_get.history_data(stock_code, time_level, start_time, end_time)
        if cache_data:
            return cache_data
        # 从数据库获取最新数据
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return None
        try:
            data = await sync_to_async(lambda: StockTimeTrade.objects.filter(stock=stock, time_level=time_level, trade_time__range=(start_time, end_time)).order_by('-trade_time').first())()
            # 检查数据是否过期（超过2分钟）
            if data and (timezone.now() - data.trade_time).total_seconds() < 120:
                await self.cache_set.history_time_trade(stock_code, time_level, data)
                return data
        except Exception as e:
            logger.error(f"从数据库获取最新股票[{stock}]{time_level}级别分时成交数据失败: {str(e)}")
            return None
        # 数据不存在或已过期，从API获取新数据
        logger.info(f"股票[{stock}]{time_level}级别分时成交数据不存在或已过期，从API获取")
        await self.fetch_and_save_history_time_trade(stock_code, time_level, start_time, end_time)
        data = await sync_to_async(lambda: StockTimeTrade.objects.filter(stock=stock, time_level=time_level, trade_time__range=(start_time, end_time)).order_by('-trade_time').first())()
        return data
    
    async def get_history_time_trades_by_limit(self, stock_code: str, time_level: Union[TimeLevel, str], limit: int = 1000) -> List[StockTimeTrade]:
        """
        获取指定股票和时间级别的最新分时成交数据
        """
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return None
        # 从Redis获取数据
        cache_data = await self.cache_get.history_time_trade_by_limit(stock_code, time_level, limit)
        if cache_data:
            return cache_data
        # 从数据库获取数据
        try:
            data = await sync_to_async(lambda: StockTimeTrade.objects.filter(stock=stock, time_level=time_level).order_by('-trade_time').limit(limit))()
            return data
        except Exception as e:
            logger.error(f"从数据库获取最新股票[{stock}]{time_level}级别分时成交数据失败: {str(e)}")
            return None
        
    async def fetch_and_save_latest_time_trade(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Dict:
        """
        从API获取并保存最新分时成交数据
        Args:
            stock_code: 股票代码
            time_level: 时间级别
        Returns:
            Optional[StockTimeTrade]: 保存的数据
        """
        # 获取股票信息
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            data_dicts = []
            api_data = await self.api.get_time_trade(stock.stock_code, time_level)
            if api_data:
                data_dict = self.data_format_process.set_time_trade_data(stock, time_level, api_data)
                if data_dict.get('trade_time') is None:
                    # logger.warning(f"API未返回{stock} {time_level}级别时间序列数据")
                    pass
                else:
                    data_dicts.append(data_dict)
                    cache_dict = data_dict.copy()
                    # await self.cache_set.latest_time_trade(stock_code, time_level, cache_dict)
                    await self.cache_set.history_time_trade(stock.stock_code, time_level, cache_dict) 
                    # 保存数据
                    result = await self._save_all_to_db_native_upsert(
                        model_class=StockTimeTrade,
                        data_list=data_dicts,
                        unique_fields=['stock', 'time_level', 'trade_time']
                    )
                    # --- 函数末尾执行最终修剪 ---
                    cache_key =  self.cache_key.history_time_trade(stock_code, time_level)
                    removed_count = await self.cache_manager.trim_cache_zset(cache_key, self.cache_limit)
                    # --- 修剪调用结束 ---
                    logger.info(f"股票[{stock}] {time_level}级别分时成交数据保存完成，结果: {result}")
                    return result
        except Exception as e:
            logger.error(f"保存{stock}股票{time_level}级别  分时成交数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_latest_time_trade_by_time_level(self, time_level: Union[TimeLevel, str]) -> Dict:
        """
        从API获取并保存最新股票分时成交数据，使用批量并发触发，但增加并发数量限制
        Args:
            time_level: 时间级别
        Returns:
            Dict: 保存结果，如 {'创建': 0, '更新': 0, '跳过': 0}
        """
        try:
            stocks = await self.stock_basic_dao.get_stock_list()
            if not stocks:
                logger.warning("股票列表不存在，无法获取时间序列数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            # 定义并发限制，设置为50（可根据需要调整）
            semaphore = Semaphore(30)  # 限制同时执行的任务数量
            async def process_stock(stock):
                """异步处理单个股票的任务，添加Semaphore限制"""
                async with semaphore:  # 使用Semaphore控制并发
                    try:
                        api_data = await self.api.get_time_trade(stock.stock_code, time_level)
                        if isinstance(api_data, dict):
                            data_dict = self.data_format_process.set_time_trade_data(stock, time_level, api_data)
                            if data_dict.get('trade_time') is not None:
                                cache_dict = data_dict.copy()
                                await self.cache_set.latest_time_trade(stock.stock_code, time_level, cache_dict)
                                await self.cache_set.history_time_trade(stock.stock_code, time_level, cache_dict)
                                return data_dict  # 返回成功的 data_dict
                            else:
                                logger.warning(f"API未返回{stock.stock_code} {time_level}级别时间序列数据")
                                return None  # 跳过
                    except Exception as e:
                        logger.error(f"处理股票[{stock.stock_code}]的分时成交数据出错: {str(e)}")
                        return None  # 返回 None 表示失败
                    return None
            # 创建任务列表，并发执行，但受Semaphore限制
            tasks = [process_stock(stock) for stock in stocks]
            results = await asyncio.gather(*tasks, return_exceptions=True)  # 并发执行，捕获异常
            # 过滤成功的 data_dict
            data_dicts: List[Dict] = [result for result in results if result is not None]
            if not data_dicts:
                logger.warning(f"API未返回任何{time_level}级别时间序列数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            # 批量保存到数据库
            result = await self._save_all_to_db_native_upsert(
                model_class=StockTimeTrade,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            # 执行缓存修剪
            for stock in stocks:
                cache_key = self.cache_key.history_time_trade(stock.stock_code, time_level)  # 假设cache_key方法已定义
                await self.cache_manager.trim_cache_zset(cache_key, self.cache_limit)
            logger.info(f"所有股票分时成交数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存股票分时成交数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}
    
    async def fetch_and_save_latest_time_trade_by_stock_code(self, stock_code: str) -> Dict:
        """
        从API获取并保存最新股票分时成交数据
        Args:
            stock_code: 股票代码
        Returns:
            Optional[StockTimeTrade]: 保存的数据
        """
        # 获取股票信息
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            logger.warning(f"股票代码[{stock_code}]不存在，无法获取时间序列数据")
            return {'创建': 0, '更新': 0, '跳过': 0}
        data_dicts = []
        try:
            for time_level in TIME_TEADE_TIME_LEVELS_LITE:
                api_data = await self.api.get_time_trade(stock.stock_code, time_level)
                if isinstance(api_data, dict):
                    data_dict = self.data_format_process.set_time_trade_data(stock, time_level, api_data)
                    if data_dict.get('trade_time') is None:
                        logger.warning(f"API未返回{stock} {time_level}级别时间序列数据")
                        pass
                    cache_dict = data_dict.copy()
                    data_dicts.append(data_dict)
                    await self.cache_set.latest_time_trade(stock.stock_code, time_level, cache_dict)
                    await self.cache_set.history_time_trade(stock.stock_code, time_level, cache_dict) 
                    # --- 生成缓存键 ---
                    cache_key =  self.cache_key.history_time_trade(stock_code, time_level)
                    # --- 单行调用修剪方法 ---
                    await self.cache_manager.trim_cache_zset(cache_key, self.cache_limit)
                    # --- 修剪调用结束 ---
            if not data_dicts:
                logger.warning(f"API未返回{stock} {time_level}级别时间序列数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            # 保存数据
            result = await self._save_all_to_db_native_upsert(
                model_class=StockTimeTrade,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            logger.info(f"{stock} 股票分时成交数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存{stock}股票分时成交数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_latest_time_trade_trading_hours_by_stock_code(self, stock_code: str) -> Dict:
        """
        从API获取并保存最新股票分时成交数据 (修正版：使用 async with 管理 API 实例)
        Args:
            stock_code: 股票代码
        Returns:
            Dict: 保存结果统计
        """
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            logger.warning(f"股票代码[{stock_code}]不存在，无法获取时间序列数据")
            return {'创建': 0, '更新': 0, '跳过': 0}
        data_dicts = []
        result = {'创建': 0, '更新': 0, '跳过': 0} # 初始化 result
        # --- 使用 async with 创建和管理 API 实例 ---
        try:
            async with self.api as api_client: # 在这里创建临时的 API 客户端实例
                for time_level in TIME_TEADE_TIME_LEVELS_PER_TRADING:
                    try:
                        # --- 使用临时的 api_client ---
                        api_data = await api_client.get_time_trade(stock.stock_code, time_level)
                        data_dict = self.data_format_process.set_time_trade_data(stock, time_level, api_data)
                        if data_dict.get('trade_time') is None:
                            logger.warning(f"API未返回{stock.stock_code} {time_level}级别时间序列数据, data_dict: {data_dict}")
                            # 根据策略，可以选择跳过这个 time_level 或直接返回
                            continue # 跳过这个 time_level，继续下一个
                        data_dicts.append(data_dict)
                        cache_dict = data_dict.copy()
                        await self.cache_set.latest_time_trade(stock.stock_code, time_level, cache_dict)
                        await self.cache_set.history_time_trade(stock.stock_code, time_level, cache_dict)
                    except Exception as inner_e:
                        # 捕获单个 time_level 处理中的错误，记录并继续处理下一个 time_level
                        logger.error(f"处理 {stock.stock_code} 的 {time_level} 级别数据时出错: {str(inner_e)}", exc_info=True)
                        continue # 继续下一个 time_level
            # --- async with 块结束，api_client 会自动关闭 ---

            if not data_dicts:
                logger.warning(f"未能成功获取并处理 {stock.stock_code} 的任何时间级别数据")
                return {'创建': 0, '更新': 0, '跳过': 0}

            # 批量保存数据
            result = await self._save_all_to_db_native_upsert(
                model_class=StockTimeTrade, # 确保 StockTimeTrade 已导入
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            cache_key = self.cache_key.history_time_trade(stock_code, time_level)
            await self.cache_manager.trim_cache_zset(cache_key, self.cache_limit)
            logger.info(f"{stock.stock_code} 股票分时成交数据保存完成，结果: {result}")

        except Exception as e:
            # 捕获 async with 外部或数据库保存过程中的错误
            logger.error(f"保存 {stock.stock_code} 股票分时成交数据过程中发生意外错误: {str(e)}", exc_info=True)
            result = {'创建': 0, '更新': 0, '跳过': 0} # 确保返回字典

        return result


    async def fetch_and_save_favorite_stocks_latest_time_trade(self) -> Dict:
        """
        从API获取并保存最新自选股分时成交数据
            
        Returns:
            Optional[StockTimeTrade]: 保存的数据
        """
        favorite_stocks = await self.user_dao.get_all_favorite_stocks()
        try:
            for stock in favorite_stocks:
                await self.fetch_and_save_latest_time_trade_by_stock_code(stock.stock_code)
        except Exception as e:
            logger.error(f"保存自选股分时成交数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_all_latest_time_trade(self) -> Dict:
        """
        从API获取并保存所有最新股票分时成交数据，使用异步并发处理
        """
        import asyncio
        
        # 获取所有股票列表
        stocks = await self.stock_basic_dao.get_stock_list()
        total_result = {'创建': 0, '更新': 0, '未更改': 0, '失败': 0, '跳过': 0}
        
        # 将股票列表分成每组100个
        batch_size = 100
        stock_batches = [stocks[i:i+batch_size] for i in range(0, len(stocks), batch_size)]
        
        # 定义处理单个股票的任务
        async def process_stock(stock):
            return await self.fetch_and_save_latest_time_trade_by_stock_code(stock.stock_code)
        
        # 并发处理每个批次
        for batch in stock_batches:
            # 为每个批次创建任务列表
            tasks = [process_stock(stock) for stock in batch]
            # 并发执行该批次的所有任务
            batch_results = await asyncio.gather(*tasks)
            
            # 聚合批次结果
            for result in batch_results:
                total_result['创建'] += result.get('创建', 0)
                total_result['更新'] += result.get('更新', 0)
                total_result['未更改'] += result.get('未更改', 0)
                total_result['失败'] += result.get('失败', 0)
                total_result['跳过'] += result.get('跳过', 0)
        
        return total_result

    async def fetch_and_save_history_time_trade(self, stock_code: str, time_level: Union[TimeLevel, str], limit: int = 1000) -> Dict:
        """
        从API获取并保存历史股票分时成交数据
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            limit: 返回记录数量限制
        Returns:
            Optional[StockTimeTrade]: 保存的数据
        """
        # 获取股票信息
        logger.info(f"==> 进入 fetch_and_save: {stock_code} {time_level}") # 入口日志
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        logger.info(f"    完成 await get_stock_by_code: {stock_code}, 结果: {'找到' if stock else '未找到'}")
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            api_datas = await self.api.get_history_trade(stock.stock_code, time_level)
            if not api_datas:
                logger.warning(f"API未返回 {stock} 的 {time_level} 级别历史时间序列数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            data_dicts = []
            for api_data in api_datas:
                data_dict = self.data_format_process.set_time_trade_data(stock, time_level, api_data)
                data_dicts.append(data_dict)
                await self.cache_set.history_time_trade(stock_code, time_level, data_dict)
            # 保存数据
            result = await self._save_all_to_db_native_upsert(
                model_class=StockTimeTrade,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            # --- 函数末尾执行最终修剪 ---
            # --- 生成缓存键 ---
            cache_key =  self.cache_key.history_time_trade(stock_code, time_level)
            # --- 单行调用修剪方法 ---
            removed_count = await self.cache_manager.trim_cache_zset(cache_key, self.cache_limit)
            # --- 修剪调用结束 ---
            logger.info(f"{stock}股票{time_level}级别历史分时成交数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存{stock}股票{time_level}级别历史分时成交数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_history_time_trade_by_stock_code(self, stock_code: str) -> Dict:
        """
        从API获取并保存历史股票分时成交数据
        Args:
            stock_code: 股票代码
        """
        # 获取股票信息
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            data_dicts = []
            total_result = {'创建': 0, '更新': 0, '跳过': 0}
            for time_level in TIME_TEADE_TIME_LEVELS_LITE:
                api_datas = await self.api.get_history_trade(stock_code, time_level)
                if not api_datas:
                    logger.warning(f"API未返回{stock.stock_code}股票的{time_level}级别历史分时成交数据")
                else:
                    # logger.info(f"获取{stock.stock_code}股票{time_level}级别历史分时成交数据, length: {len(api_datas)}")
                    for api_data in api_datas:
                        if isinstance(api_data, dict):
                            data_dict = self.data_format_process.set_time_trade_data(stock, time_level, api_data)
                            data_dicts.append(data_dict)
                            await self.cache_set.history_time_trade(stock.stock_code, time_level, data_dict)
                # 当数据量超过10万时，保存一次
                if len(data_dicts) >= 50000:
                    logger.info(f"数据量达到{len(data_dicts)}，开始保存批次数据")
                    batch_result = await self._save_all_to_db_native_upsert(
                        model_class=StockTimeTrade,
                        data_list=data_dicts,
                        unique_fields=['stock', 'time_level', 'trade_time']
                    )
                    logger.info(f"批次数据保存完成，结果: {batch_result}")
                    # 累加结果
                    for key in total_result:
                        total_result[key] += batch_result.get(key, 0)
                    # 清空数据列表，准备下一批
                    data_dicts = []
                # logger.warning(f"当前data_dicts总量: {len(data_dicts)}")
            # 保存剩余数据
            if data_dicts:
                final_result = await self._save_all_to_db_native_upsert(
                    model_class=StockTimeTrade,
                    data_list=data_dicts,
                    unique_fields=['stock', 'time_level', 'trade_time']
                )
                logger.info(f"剩余数据保存完成，结果: {final_result}")
                # 累加最终结果
                for key in total_result:
                    total_result[key] += final_result.get(key, 0)
            # --- 函数末尾执行最终修剪 ---
            for time_level in TIME_TEADE_TIME_LEVELS:
                # --- 生成缓存键 ---
                cache_key =  self.cache_key.history_time_trade(stock_code, time_level)
                # --- 单行调用修剪方法 ---
                removed_count = await self.cache_manager.trim_cache_zset(cache_key, self.cache_limit)
                # --- 修剪调用结束 ---
            return total_result
        except Exception as e:
            logger.error(f"保存出错 - {stock}股票分时成交数据: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_all_history_time_trade(self) -> Dict:
        """
        从API获取并保存所有历史股票分时成交数据
        """
        stocks = await self.stock_basic_dao.get_stock_list()
        try:
            for stock in stocks:
                await self.fetch_and_save_history_time_trade_by_stock_code(stock.stock_code)
        except Exception as e:
            logger.error(f"保存{stock}股票历史分时成交数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    # ================= KDJ指标相关方法 =================
    async def get_latest_kdj(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Optional[StockKDJIndicator]:
        """
        获取最新的KDJ指标数据
        Args:
            stock_code: 股票代码
            time_level: 时间级别
        Returns:
            Optional[StockKDJIndicator]: 最新的KDJ指标数据
        """
        cache_data = await self.cache_get.latest_kdj(stock_code, time_level)
        if cache_data:
            return cache_data
        
        # 从数据库获取最新数据
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return None
        try:
            data = await sync_to_async(lambda: StockKDJIndicator.objects.filter(stock=stock, time_level=time_level).order_by('-trade_time').first())()
            # 检查数据是否过期（超过2分钟）
            if data and (timezone.now() - data.trade_time).total_seconds() < 120:
                await self.cache_set.latest_kdj(stock_code, time_level, data)
                return data
        except Exception as e:
            logger.error(f"从数据库获取最新股票[{stock}]{time_level}级别KDJ指标数据失败: {str(e)}")
            return None
        
        # 数据不存在或已过期，从API获取新数据
        logger.info(f"股票[{stock}]{time_level}级别KDJ指标数据不存在或已过期，从API获取")
        await self.fetch_and_save_latest_kdj(stock_code, time_level)
        data = await sync_to_async(lambda: StockKDJIndicator.objects.filter(stock=stock, time_level=time_level).order_by('-trade_time').first())()
        return data
    
    async def fetch_and_save_latest_kdj(self, stock_code: str, time_level: str) -> Dict:
        """
        从API获取并保存最新KDJ指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
        """
        # 获取股票信息
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            api_data = await self.api.get_kdj(stock.stock_code, time_level)
                
            if not api_data:
                logger.warning(f"API未返回{stock}的{time_level}级别KDJ指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            data_dicts = []
            data_dict = self.data_format_process.set_kdj_data(stock, time_level, api_data)
            data_dicts.append(data_dict)
            cache_dict = data_dict.copy()
            await self.cache_set.latest_kdj(stock_code, time_level, cache_dict)

            # 保存数据
            result = await self._save_all_to_db_native_upsert(
                model_class=StockKDJIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            
            logger.info(f"{stock}股票{time_level}级别KDJ指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存{stock}股票{time_level}级别KDJ指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_latest_kdj_by_stock_code(self, stock_code: str) -> Dict:
        """
        从API获取并保存最新KDJ指标数据
        
        Args:
            stock_code: 股票代码
        """
        # 获取股票信息
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            logger.warning(f"股票代码[{stock_code}]不存在，无法获取KDJ指标数据")
            return {'创建': 0, '更新': 0, '跳过': 0}
        data_dicts = []
        try:
            for time_level in TIME_TEADE_TIME_LEVELS:
                api_data = await self.api.get_kdj(stock.stock_code, time_level)
                data_dict = self.data_format_process.set_kdj_data(stock, time_level, api_data)
                # logger.info(f"data_dict: {data_dict}")
                if data_dict.get('trade_time') is None:
                    logger.debug(f"未获取到{stock}股票{time_level}级别KDJ指标数据")
                else:
                    data_dicts.append(data_dict)
                    cache_dict = data_dict.copy()
                    await self.cache_set.latest_kdj(stock_code, time_level, cache_dict)
            if not data_dicts:
                logger.warning(f"API未返回{stock}股票的{time_level}级别KDJ指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
        except Exception as e:
            logger.error(f"fetch_and_save_latest_kdj_by_stock_code.获取和缓存{stock}股票KDJ指标数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            # 保存数据
            result = await self._save_all_to_db_native_upsert(
                model_class=StockKDJIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            logger.info(f"{stock}股票KDJ指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"fetch_and_save_latest_kdj_by_stock_code.Mysql保存{stock}股票KDJ指标数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_latest_kdj_by_time_level(self, time_level: str) -> Dict:
        """
        从API获取并保存最新KDJ指标数据
        
        Args:
            time_level: 时间级别
        """
        stocks = await self.stock_basic_dao.get_stock_list()
        try:
            data_dicts = []
            for stock in stocks:
                for time_level in TIME_TEADE_TIME_LEVELS:
                    api_data = await self.api.get_kdj(stock.stock_code, time_level)
                    data_dict = self.data_format_process.set_kdj_data(stock, time_level, api_data)
                    data_dicts.append(data_dict)
            if not data_dicts:
                logger.warning(f"API未返回{time_level}级别的所有股票KDJ指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            # 保存数据
            result = await self._save_all_to_db_native_upsert(
                model_class=StockKDJIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            
            logger.info(f"{time_level}级别KDJ指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存{time_level}级别KDJ指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_favorite_stocks_latest_kdj(self) -> Dict:
        """
        从API获取并保存自选股最新KDJ指标数据
        """
        favorite_stocks = await self.user_dao.get_all_favorite_stocks()
        try:
            for stock in favorite_stocks:
                await self.fetch_and_save_latest_kdj_by_stock_code(stock.stock_code)
        except Exception as e:
            logger.error(f"保存自选股KDJ指标数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_all_latest_kdj(self) -> Dict:
        """
        从API获取并保存所有股票最新KDJ指标数据
        """
        stocks = await self.stock_basic_dao.get_stock_list()
        try:
            for stock in stocks:
                await self.fetch_and_save_latest_kdj_by_stock_code(stock.stock_code)
        except Exception as e:
            logger.error(f"保存所有股票KDJ指标数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    # ================= 历史KDJ指标相关方法 =================
    async def get_history_kdj(self, stock_code: str, time_level: str, 
                             limit: int = 1000) -> List[StockKDJIndicator]:
        """
        获取历史KDJ指标数据
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            limit: 返回记录数量限制
        Returns:
            List[StockKDJIndicator]: 历史KDJ指标数据列表
        """
        cache_datas = await self.cache_get.history_kdj_by_limit(stock_code, time_level, limit)
        logger.info(f"get_history_kdj.cache_datas: {cache_datas}")
        if cache_datas:
            return cache_datas
        
        # 从数据库获取最新数据
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return None
        try:
            data = await sync_to_async(lambda: StockKDJIndicator.objects.filter(stock=stock, time_level=time_level).order_by('-trade_time').first())()
            # 检查数据是否过期（超过2分钟）
            if data and (timezone.now() - data.trade_time).total_seconds() < 120:
                await self.cache_set.latest_kdj(stock_code, time_level, data)
                return data
        except Exception as e:
            logger.error(f"从数据库获取最新股票[{stock}]{time_level}级别KDJ指标数据失败: {str(e)}")
            return None
        
        # 数据不存在或已过期，从API获取新数据
        logger.info(f"股票[{stock}]{time_level}级别KDJ指标数据不存在或已过期，从API获取")
        await self.fetch_and_save_latest_kdj(stock_code, time_level)
        data = await sync_to_async(lambda: StockKDJIndicator.objects.filter(stock=stock, time_level=time_level).order_by('-trade_time').first())()
        return data
    
    async def fetch_and_save_history_kdj(self, stock_code: str, time_level: str, limit: int = 1000) -> Dict:
        """
        从API获取并保存历史KDJ指标数据
        """
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            api_datas = await self.api.get_history_kdj(stock.stock_code, time_level)
            if not api_datas:
                logger.warning(f"API未返回{stock}的{time_level}级别KDJ指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            data_dicts = []
            for api_data in api_datas:
                data_dict = self.data_format_process.set_kdj_data(stock, time_level, api_data)
                data_dicts.append(data_dict)
                cache_dict = data_dict.copy()
                await self.cache_set.history_kdj(stock_code, time_level, cache_dict)
            # 保存数据
            result = await self._save_all_to_db_native_upsert(
                model_class=StockKDJIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            # --- 函数末尾执行最终修剪 ---
            # --- 生成缓存键 ---
            cache_key =  self.cache_key.history_kdj(stock_code, time_level)
            # --- 单行调用修剪方法 ---
            await self.cache_manager.trim_cache_zset(cache_key, self.cache_limit)
            # --- 修剪调用结束 ---
            logger.info(f"{stock}股票{time_level}级别KDJ指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存{stock}股票{time_level}级别KDJ指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_history_kdj_by_stock_code(self, stock_code: str) -> Dict:
        """
        从API获取并保存股票历史KDJ指标数据
        Args:
            stock_code: 股票代码
        Returns:
            Dict: 保存结果
        """
        # 获取股票信息
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            logger.info(f"开始获取{stock.stock_code}股票历史KDJ指标数据")
            data_dicts = []
            total_result = {'创建': 0, '更新': 0, '跳过': 0}
            for time_level in TIME_TEADE_TIME_LEVELS:
                api_datas = await self.api.get_history_kdj(stock.stock_code, time_level)
                for index, api_data in enumerate(api_datas):
                    data_dict = self.data_format_process.set_kdj_data(stock, time_level, api_data)
                    data_dicts.append(data_dict)
                    cache_dict = data_dict.copy()
                    if index < self.cache_limit:
                        await self.cache_set.history_kdj(stock_code, time_level, cache_dict)
                # 当数据量超过10万时，保存一次
                if len(data_dicts) >= 20000:
                    logger.info(f"数据量达到{len(data_dicts)}，开始保存批次数据")
                    batch_result = await self._save_all_to_db_native_upsert(
                        model_class=StockKDJIndicator,
                        data_list=data_dicts,
                        unique_fields=['stock', 'time_level', 'trade_time']
                    )
                    logger.info(f"批次数据保存完成，结果: {batch_result}")
                    # 累加结果
                    for key in total_result:
                        total_result[key] += batch_result.get(key, 0)
                    # 清空数据列表，准备下一批
                    data_dicts = []
                # logger.warning(f"当前data_dicts总量: {len(data_dicts)}")
            if not api_datas:
                logger.warning(f"API未返回{stock.stock_code}股票的{time_level}级别历史KDJ指标数据")
                return {'创建': 0, '更新': 0, '未更改': 0, '失败': 0, '跳过': 0}
            # 保存剩余数据
            if data_dicts:
                final_result = await self._save_all_to_db_native_upsert(
                    model_class=StockKDJIndicator,
                    data_list=data_dicts,
                    unique_fields=['stock', 'time_level', 'trade_time']
                )
                logger.info(f"剩余数据保存完成，结果: {final_result}")
                # 累加最终结果
                for key in total_result:
                    total_result[key] += final_result.get(key, 0)
            # --- 函数末尾执行最终修剪 ---
            for time_level in TIME_TEADE_TIME_LEVELS:
                # --- 生成缓存键 ---
                cache_key =  self.cache_key.history_kdj(stock_code, time_level)
                # --- 单行调用修剪方法 ---
                await self.cache_manager.trim_cache_zset(cache_key, self.cache_limit)
                # --- 修剪调用结束 ---
            
            # --- 最终修剪结束 ---
            logger.info(f"所有股票历史KDJ指标数据保存完成，总结果: {total_result}")
            return total_result
        except Exception as e:
            logger.error(f"保存{stock}股票历史KDJ指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_favorite_stocks_history_kdj(self) -> Dict:
        """
        从API获取并保存自选股历史KDJ指标数据
        """
        favorite_stocks = await self.user_dao.get_all_favorite_stocks()
        try:
            for stock in favorite_stocks:
                await self.fetch_and_save_history_kdj_by_stock_code(stock.stock_code)
        except Exception as e:
            logger.error(f"保存自选股历史KDJ指标数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_all_history_kdj(self) -> Dict:
        """
        从API获取并保存所有股票历史KDJ指标数据
        """
        stocks = await self.stock_basic_dao.get_stock_list()
        try:
            for stock in stocks:
                await self.fetch_and_save_history_kdj_by_stock_code(stock.stock_code)
        except Exception as e:
            logger.error(f"保存所有股票历史KDJ指标数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    # ================= MACD指标相关方法 =================
    async def get_latest_macd(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Optional[StockMACDIndicator]:
        """
        获取最新的MACD指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            
        Returns:
            Optional[StockMACDIndicator]: 最新的MACD指标数据
        """
        cache_data = await self.cache_get.latest_macd(stock_code, time_level)
        if cache_data:
            return cache_data
        
        return await self._get_latest_indicator(
            model_class=StockMACDIndicator,
            stock_code=stock_code,
            time_level=time_level,
            api_method=self.api.get_macd_data,
            mapping=MACD_INDICATOR_MAPPING,
            cache_prefix="macd"
        )

    async def fetch_and_save_latest_macd(self, stock_code: str, time_level: str) -> Dict:
        """
        从API获取并保存最新MACD指标数据
        """
        # 获取股票信息
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            api_data = await self.api.get_macd(stock.stock_code, time_level)
                
            if not api_data:
                logger.warning(f"API未返回{stock}的{time_level}级别最新MACD指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            data_dicts = []
            data_dict = self.data_format_process.set_macd_data(stock, time_level, api_data)
            data_dicts.append(data_dict)

            # 保存数据
            result = await self._save_all_to_db_native_upsert(
                model_class=StockMACDIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            cache_dict = data_dict.copy()
            await self.cache_set.latest_macd(stock_code, time_level, cache_dict)
            logger.info(f"{stock}股票{time_level}级别最新MACD指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存{stock}股票{time_level}级别最新MACD指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_latest_macd_by_stock_code(self, stock_code: str) -> Dict:
        """
        从API获取并保存所有股票最新MACD指标数据
        """
        # 获取股票信息
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            logger.info(f"开始获取{stock.stock_code}股票最新MACD指标数据")
            data_dicts = []
            for time_level in TIME_TEADE_TIME_LEVELS:
                api_data = await self.api.get_macd(stock.stock_code, time_level)
                data_dict = self.data_format_process.set_macd_data(stock, time_level, api_data)
                if data_dict.get('trade_time') is None:
                    logger.debug(f"未获取到{stock}股票{time_level}级别MACD指标数据")
                else:
                    data_dicts.append(data_dict)
                    cache_dict = data_dict.copy()
                    await self.cache_set.latest_macd(stock_code, time_level, cache_dict)
            if not data_dicts:
                logger.warning(f"API未返回{stock}股票的{time_level}级别最新MACD指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}            
            # 保存数据
            result = await self._save_all_to_db_native_upsert(
                model_class=StockMACDIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            
            logger.info(f"{stock}股票最新MACD指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存{stock}股票最新MACD指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_favorite_stocks_latest_macd(self) -> Dict:
        """
        从API获取并保存自选股最新MACD指标数据
        """
        favorite_stocks = await self.user_dao.get_all_favorite_stocks()
        try:
            for stock in favorite_stocks:
                await self.fetch_and_save_latest_macd_by_stock_code(stock.stock_code)
        except Exception as e:
            logger.error(f"保存自选股最新MACD指标数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_all_latest_macd(self) -> Dict:
        """
        从API获取并保存所有股票最新MACD指标数据
        """
        stocks = await self.stock_basic_dao.get_stock_list()
        try:
            for stock in stocks:
                await self.fetch_and_save_latest_macd_by_stock_code(stock.stock_code)
        except Exception as e:
            logger.error(f"保存所有股票最新MACD指标数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    # ================= 历史MACD指标相关方法 =================
    async def get_history_macd(self, stock_code: str, time_level: Union[TimeLevel, str], 
                             limit: int = 1000) -> List[StockMACDIndicator]:
        """
        获取历史MACD指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            limit: 返回记录数量限制
            
        Returns:
            List[StockMACDIndicator]: 历史MACD指标数据列表
        """
        return await self._get_history_indicators(
            model_class=StockMACDIndicator,
            stock_code=stock_code,
            time_level=time_level,
            api_method=self.api.get_macd_data,
            mapping=MACD_INDICATOR_MAPPING,
            cache_prefix="macd",
            limit=limit
        )
    
    async def fetch_and_save_history_macd(self, stock_code: str, time_level: str) -> Dict:
        """
        从API获取并保存历史MACD指标数据
        Args:
            stock_code: 股票代码
            time_level: 时间级别
        Returns:
            Optional[StockMACDIndicator]: 保存结果
        """
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            api_datas = await self.api.get_history_macd(stock.stock_code, time_level)
                
            if not api_datas:
                logger.warning(f"API未返回{stock}的{time_level}级别历史MACD指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            data_dicts = []
            for api_data in api_datas:
                data_dict = self.data_format_process.set_macd_data(stock, time_level, api_data)
                data_dicts.append(data_dict)
                cache_dict = data_dict.copy()
                await self.cache_set.history_macd(stock_code, time_level, cache_dict)

            # 保存数据
            result = await self._save_all_to_db_native_upsert(
                model_class=StockMACDIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            # --- 函数末尾执行最终修剪 ---
            # --- 生成缓存键 ---
            cache_key =  self.cache_key.history_macd(stock_code, time_level)
            # --- 单行调用修剪方法 ---
            await self.cache_manager.trim_cache_zset(cache_key, self.cache_limit)
            # --- 修剪调用结束 ---
            logger.info(f"{stock}股票{time_level}级别历史MACD指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存{stock}股票{time_level}级别历史MACD指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_history_macd_by_stock_code(self, stock_code: str) -> Dict:
        """
        从API获取并保存所有股票历史MACD指标数据
        Args:
            stock_code: 股票代码
        Returns:
            Optional[StockMACDIndicator]: 保存结果
        """
        # 获取股票信息
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            logger.info(f"开始获取{stock.stock_code}股票历史MACD指标数据")
            data_dicts = []
            total_result = {'创建': 0, '更新': 0, '跳过': 0}
            for time_level in TIME_TEADE_TIME_LEVELS:
                api_datas = await self.api.get_history_macd(stock_code, time_level)
                for index, api_data in enumerate(api_datas):
                    data_dict = self.data_format_process.set_macd_data(stock, time_level, api_data)
                    data_dicts.append(data_dict)
                    if index <= self.cache_limit:
                        cache_dict = data_dict.copy()
                        await self.cache_set.history_macd(stock_code, time_level, cache_dict)
                # 当数据量超过10万时，保存一次
                if len(data_dicts) >= 20000:
                    logger.info(f"数据量达到{len(data_dicts)}，开始保存批次数据")
                    batch_result = await self._save_all_to_db_native_upsert(
                        model_class=StockMACDIndicator,
                        data_list=data_dicts,
                        unique_fields=['stock', 'time_level', 'trade_time']
                    )
                    logger.info(f"批次数据保存完成，结果: {batch_result}")
                    # 累加结果
                    for key in total_result:
                        total_result[key] += batch_result.get(key, 0)
                    # 清空数据列表，准备下一批
                    data_dicts = []
                # logger.warning(f"当前data_dicts总量: {len(data_dicts)}")
            if not api_datas:
                logger.warning(f"API未返回{stock.stock_code}股票的{time_level}级别历史MACD指标数据")
                return {'创建': 0, '更新': 0, '未更改': 0, '失败': 0, '跳过': 0}
            
            # 保存剩余数据
            if data_dicts:
                final_result = await self._save_all_to_db_native_upsert(
                    model_class=StockMACDIndicator,
                    data_list=data_dicts,
                    unique_fields=['stock', 'time_level', 'trade_time']
                )
                logger.info(f"剩余数据保存完成，结果: {final_result}")
                
                # 累加最终结果
                for key in total_result:
                    total_result[key] += final_result.get(key, 0)
            # --- 函数末尾执行最终修剪 ---
            for time_level in TIME_TEADE_TIME_LEVELS:
                # --- 生成缓存键 ---
                cache_key =  self.cache_key.history_macd(stock_code, time_level)
                # --- 单行调用修剪方法 ---
                await self.cache_manager.trim_cache_zset(cache_key, self.cache_limit)
                # --- 修剪调用结束 ---
            
            # --- 最终修剪结束 ---
            logger.info(f"所有股票历史MACD指标数据保存完成，总结果: {total_result}")
            return total_result
        except Exception as e:
            logger.error(f"保存{stock}股票历史MACD指标数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_favorite_stocks_history_macd(self) -> Dict:
        """
        从API获取并保存自选股历史MACD指标数据
        """
        favorite_stocks = await self.user_dao.get_all_favorite_stocks()
        try:
            for stock in favorite_stocks:
                await self.fetch_and_save_history_macd_by_stock_code(stock.stock_code)
        except Exception as e:
            logger.error(f"保存自选股历史MACD指标数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_all_history_macd(self) -> Dict:
        """
        从API获取并保存所有股票历史MACD指标数据
        """
        stocks = await self.stock_basic_dao.get_stock_list()
        try:
            for stock in stocks:
                await self.fetch_and_save_history_macd_by_stock_code(stock.stock_code)
        except Exception as e:
            logger.error(f"保存所有股票历史MACD指标数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}
        
    # ================= MA指标相关方法 =================
    
    async def get_latest_ma(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Optional[StockMAIndicator]:
        """
        获取最新的MA指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            
        Returns:
            Optional[StockMAIndicator]: 最新的MA指标数据
        """
        return await self._get_latest_indicator(
            model_class=StockMAIndicator,
            stock_code=stock_code,
            time_level=time_level,
            api_method=self.api.get_ma_data,
            mapping=MA_INDICATOR_MAPPING,
            cache_prefix="ma"
        )
    
    async def fetch_and_save_latest_ma(self, stock_code: str, time_level: str) -> Dict:
        """
        从API获取并保存股票历史MA指标数据
        Args:
            stock_code: 股票代码
            time_level: 时间级别
        Returns:
            Optional[StockMAIndicator]: 保存结果
        """
        # 获取股票信息
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            api_data = await self.api.get_ma(stock.stock_code, time_level)
                
            if not api_data:
                logger.warning(f"API未返回{stock}的{time_level}级别最新MA指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            data_dicts = []
            data_dict = self.data_format_process.set_ma_data(stock, time_level, api_data)
            data_dicts.append(data_dict)
            cache_dict = data_dict.copy()
            await self.cache_set.latest_ma(stock_code, time_level, cache_dict)

            # 保存数据
            result = await self._save_all_to_db_native_upsert(
                model_class=StockMAIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            
            logger.info(f"{stock}股票{time_level}级别最新MA指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存{stock}股票{time_level}级别最新MA指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_latest_ma_by_stock_code(self, stock_code: str) -> Dict:
        """
        从API获取并保存股票历史MA指标数据
        Args:
            stock_code: 股票代码
        Returns:
            Optional[StockMAIndicator]: 保存结果
        """
        # 获取股票信息
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            logger.info(f"开始获取{stock.stock_code}股票最新MA指标数据")
            data_dicts = []
            for time_level in TIME_TEADE_TIME_LEVELS:
                api_data = await self.api.get_ma(stock.stock_code, time_level)
                data_dict = self.data_format_process.set_ma_data(stock, time_level, api_data)
                if data_dict.get('trade_time') is None:
                    logger.debug(f"未获取到{stock}股票{time_level}级别MA指标数据")
                else:
                    data_dicts.append(data_dict)
                    cache_dict = data_dict.copy()
                    await self.cache_set.latest_ma(stock_code, time_level, cache_dict)
            if not data_dicts:
                logger.warning(f"API未返回{stock}股票的{time_level}级别最新MA指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            # 保存数据
            result = await self._save_all_to_db_native_upsert(
                model_class=StockMAIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            
            logger.info(f"{stock}股票最新MA指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存{stock}股票最新MA指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_favorite_stocks_latest_ma(self) -> Dict:
        """
        从API获取并保存自选股最新MA指标数据
        """
        favorite_stocks = await self.user_dao.get_all_favorite_stocks()
        try:
            for stock in favorite_stocks:
                await self.fetch_and_save_latest_ma_by_stock_code(stock.stock_code)
        except Exception as e:
            logger.error(f"保存自选股最新MA指标数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_all_latest_ma(self) -> Dict:
        """
        从API获取并保存所有股票最新MA指标数据
        """
        stocks = await self.stock_basic_dao.get_stock_list()
        try:
            for stock in stocks:
                await self.fetch_and_save_latest_ma_by_stock_code(stock.stock_code)
        except Exception as e:
            logger.error(f"保存所有股票最新MA指标数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}
            
    # ================= 历史MA指标相关方法 =================
    async def get_history_ma(self, stock_code: str, time_level: Union[TimeLevel, str], 
                           limit: int = 1000) -> List[StockMAIndicator]:
        """
        获取历史MA指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            limit: 返回记录数量限制
            
        Returns:
            List[StockMAIndicator]: 历史MA指标数据列表
        """
        return await self._get_history_indicators(
            model_class=StockMAIndicator,
            stock_code=stock_code,
            time_level=time_level,
            api_method=self.api.get_ma_data,
            mapping=MA_INDICATOR_MAPPING,
            cache_prefix="ma",
            limit=limit
        )
    
    async def fetch_and_save_history_ma(self, stock_code: str, time_level: str) -> Dict:
        """
        从API获取并保存股票历史MA指标数据
        Args:
            stock_code: 股票代码
            time_level: 时间级别
        Returns:
            Optional[StockMAIndicator]: 保存结果
        """
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            api_datas = await self.api.get_history_ma(stock.stock_code, time_level)
            if not api_datas:
                logger.warning(f"API未返回{stock}的{time_level}级别历史MA指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            data_dicts = []
            for api_data in api_datas:
                data_dict = self.data_format_process.set_ma_data(stock, time_level, api_data)
                data_dicts.append(data_dict)
                cache_dict = data_dict.copy()
                await self.cache_set.history_ma(stock_code, time_level, cache_dict)
            # 保存数据
            result = await self._save_all_to_db_native_upsert(
                model_class=StockMAIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            # --- 函数末尾执行最终修剪 ---
            # --- 生成缓存键 ---
            cache_key =  self.cache_key.history_ma(stock_code, time_level)
            # --- 单行调用修剪方法 ---
            await self.cache_manager.trim_cache_zset(cache_key, self.cache_limit)
            # --- 修剪调用结束 ---
            logger.info(f"{stock}股票{time_level}级别历史MA指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存{stock}股票{time_level}级别历史MA指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_history_ma_by_stock_code(self, stock_code: str) -> Dict:
        """
        从API获取并保存股票历史MA指标数据
        Args:
            stock_code: 股票代码
        Returns:
            Optional[StockMAIndicator]: 保存结果
        """
        # 获取股票信息
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            logger.info(f"开始获取{stock.stock_code}股票历史MA指标数据")
            data_dicts = []
            total_result = {'创建': 0, '更新': 0, '跳过': 0}
            for time_level in TIME_TEADE_TIME_LEVELS:
                api_datas = await self.api.get_history_ma(stock_code, time_level)
                for index, api_item in enumerate(api_datas):
                    data_dict = self.data_format_process.set_ma_data(stock, time_level, api_item)
                    data_dicts.append(data_dict)
                    if index <= self.cache_limit:
                        cache_dict = data_dict.copy()
                        await self.cache_set.history_ma(stock_code, time_level, cache_dict)
                # 当数据量超过10万时，保存一次
                if len(data_dicts) >= 20000:
                    logger.info(f"数据量达到{len(data_dicts)}，开始保存批次数据")
                    batch_result = await self._save_all_to_db_native_upsert(
                        model_class=StockMAIndicator,
                        data_list=data_dicts,
                        unique_fields=['stock', 'time_level', 'trade_time']
                    )
                    logger.info(f"批次数据保存完成，结果: {batch_result}")
                    # 累加结果
                    for key in total_result:
                        total_result[key] += batch_result.get(key, 0)
                    # 清空数据列表，准备下一批
                    data_dicts = []
                # logger.warning(f"当前data_dicts总量: {len(data_dicts)}")
                if not api_datas:
                    logger.warning(f"API未返回{stock.stock_code}股票的{time_level}级别历史BOLL指标数据")
                    return {'创建': 0, '更新': 0, '跳过': 0}
            # 保存剩余数据
            if data_dicts:
                final_result = await self._save_all_to_db_native_upsert(
                    model_class=StockMAIndicator,
                    data_list=data_dicts,
                    unique_fields=['stock', 'time_level', 'trade_time']
                )
                logger.info(f"剩余数据保存完成，结果: {final_result}")
                
                # 累加最终结果
                for key in total_result:
                    total_result[key] += final_result.get(key, 0)
            # --- 函数末尾执行最终修剪 ---
            for time_level in TIME_TEADE_TIME_LEVELS:
                # --- 生成缓存键 ---
                cache_key =  self.cache_key.history_ma(stock_code, time_level)
                # --- 单行调用修剪方法 ---
                await self.cache_manager.trim_cache_zset(cache_key, self.cache_limit)
                # --- 修剪调用结束 ---
            
            # --- 最终修剪结束 ---
            logger.info(f"所有股票历史MA指标数据保存完成，总结果: {total_result}")
            return total_result
        except Exception as e:
            logger.error(f"保存{stock}股票历史MA指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_favorite_stocks_history_ma(self) -> Dict:
        """
        从API获取并保存自选股历史MA指标数据
        """
        favorite_stocks = await self.user_dao.get_all_favorite_stocks()
        try:
            for stock in favorite_stocks:
                await self.fetch_and_save_history_ma_by_stock_code(stock.stock_code)
        except Exception as e:
            logger.error(f"保存自选股历史MA指标数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_all_history_ma(self) -> Dict:
        """
        从API获取并保存所有股票历史MA指标数据
        """
        stocks = await self.stock_basic_dao.get_stock_list()
        try:
            for stock in stocks:
                await self.fetch_and_save_history_ma_by_stock_code(stock.stock_code)
        except Exception as e:
            logger.error(f"保存所有股票历史MA指标数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}
    
    # ================= BOLL指标相关方法 =================
    async def get_latest_boll(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Optional[StockBOLLIndicator]:
        """
        获取最新的BOLL指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            
        Returns:
            Optional[StockBOLLIndicator]: 最新的BOLL指标数据
        """
        return await self._get_latest_indicator(
            model_class=StockBOLLIndicator,
            stock_code=stock_code,
            time_level=time_level,
            api_method=self.api.get_boll_data,
            mapping=BOLL_INDICATOR_MAPPING,
            cache_prefix="boll"
        )
    
    async def fetch_and_save_latest_boll(self, stock_code: str, time_level: str) -> Dict:
        """
        从API获取并保存股票最新BOLL指标数据
        Args:
            stock_code: 股票代码
            time_level: 时间级别
        Returns:
            Optional[StockBOLLIndicator]: 保存结果
        """
        # 获取股票信息
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            api_data = await self.api.get_boll(stock.stock_code, time_level)
                
            if not api_data:
                logger.warning(f"API未返回{stock}的{time_level}级别最新BOLL指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            data_dicts = []
            data_dict = self.data_format_process.set_boll_data(stock, time_level, api_data)
            data_dicts.append(data_dict)
            cache_dict = data_dict.copy()
            await self.cache_set.latest_boll(stock.stock_code, time_level, cache_dict)

            # 保存数据
            result = await self._save_all_to_db_native_upsert(
                model_class=StockBOLLIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            
            logger.info(f"{stock}股票{time_level}级别最新BOLL指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存{stock}股票{time_level}级别最新BOLL指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_latest_boll_by_stock_code(self, stock_code: str) -> Dict:
        """
        从API获取并保存股票最新BOLL指标数据
        Args:
            stock_code: 股票代码
        Returns:
            Optional[StockBOLLIndicator]: 保存结果
        """
        # 获取股票信息
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        # logger.warning(f"stock: {stock}, type: {type(stock)}")
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            logger.info(f"开始获取{stock.stock_code}股票最新BOLL指标数据")
            data_dicts = []
            for time_level in TIME_TEADE_TIME_LEVELS:
                api_data = await self.api.get_boll(stock.stock_code, time_level)
                if not api_data:
                    continue
                data_dict = self.data_format_process.set_boll_data(stock, time_level, api_data)
                data_dicts.append(data_dict)
                cache_dict = data_dict.copy()
                await self.cache_set.latest_boll(stock.stock_code, time_level, cache_dict)
            if not data_dicts:
                logger.warning(f"API未返回{stock}股票的{time_level}级别最新BOLL指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            # 保存数据
            result = await self._save_all_to_db_native_upsert(
                model_class=StockBOLLIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            
            logger.info(f"{stock}股票最新BOLL指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存{stock}股票最新BOLL指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}
        
    async def fetch_and_save_favorite_stocks_latest_boll(self) -> Dict:
        """
        从API获取并保存自选股最新BOLL指标数据
        """
        favorite_stocks = await self.user_dao.get_all_favorite_stocks()
        try:
            for stock in favorite_stocks:
                await self.fetch_and_save_latest_boll_by_stock_code(stock.stock_code)
        except Exception as e:
            logger.error(f"保存自选股最新BOLL指标数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}
        
    async def fetch_and_save_all_latest_boll(self) -> Dict:
        """
        从API获取并保存所有股票最新BOLL指标数据
        """
        stocks = await self.stock_basic_dao.get_stock_list()
        # logger.warning(f"stocks: {stocks}, type: {type(stocks)}")
        try:
            for stock in stocks:
                await self.fetch_and_save_latest_boll_by_stock_code(stock.stock_code)
        except Exception as e:
            logger.error(f"保存所有股票最新BOLL指标数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    # ================= 历史BOLL指标相关方法 =================
    async def get_history_boll(self, stock_code: str, time_level: Union[TimeLevel, str], 
                             limit: int = 1000) -> List[StockBOLLIndicator]:
        """
        获取历史BOLL指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            limit: 返回记录数量限制
            
        Returns:
            List[StockBOLLIndicator]: 历史BOLL指标数据列表
        """
        return await self._get_history_indicators(
            model_class=StockBOLLIndicator,
            stock_code=stock_code,
            time_level=time_level,
            api_method=self.api.get_boll_data,
            mapping=BOLL_INDICATOR_MAPPING,
            cache_prefix="boll",
            limit=limit
        )
    
    async def fetch_and_save_history_boll(self, stock_code: str, time_level: str) -> Dict: 
        """
        从API获取并保存股票历史BOLL指标数据
        Args:
            stock_code: 股票代码
            time_level: 时间级别
        Returns:
            Optional[StockBOLLIndicator]: 保存结果
        """
        # 获取股票信息
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            api_datas = await self.api.get_history_boll(stock.stock_code, time_level)
                
            if not api_datas:
                logger.warning(f"API未返回{stock}的{time_level}级别历史BOLL指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            data_dicts = []
            for api_data in api_datas:
                data_dict = self.data_format_process.set_boll_data(stock, time_level, api_data)
                data_dicts.append(data_dict)
                cache_dict = data_dict.copy()
                await self.cache_set.history_boll(stock.stock_code, time_level, cache_dict)

            # 保存数据
            result = await self._save_all_to_db_native_upsert(
                model_class=StockBOLLIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            # --- 函数末尾执行最终修剪 ---
            # --- 生成缓存键 ---
            cache_key =  self.cache_key.history_boll(stock_code, time_level)
            # --- 单行调用修剪方法 ---
            await self.cache_manager.trim_cache_zset(cache_key, self.cache_limit)
            # --- 修剪调用结束 ---
            logger.info(f"{stock}股票{time_level}级别历史BOLL指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存{stock}股票{time_level}级别历史BOLL指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}
        
    async def fetch_and_save_history_boll_by_stock_code(self, stock_code: str) -> Dict:
        """
        从API获取并保存股票历史BOLL指标数据
        """
        # 获取股票信息
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            logger.info(f"开始获取{stock.stock_code}股票历史BOLL指标数据")
            data_dicts = []
            total_result = {'创建': 0, '更新': 0, '跳过': 0}
            for time_level in TIME_TEADE_TIME_LEVELS:
                api_datas = await self.api.get_history_boll(stock_code, time_level)
                for data_index, api_data in enumerate(api_datas):
                    data_dict = self.data_format_process.set_boll_data(stock, time_level, api_data)
                    data_dicts.append(data_dict)
                    if data_index < self.cache_limit:
                        cache_dict = data_dict.copy()
                        await self.cache_set.history_boll(stock.stock_code, time_level, cache_dict)
                    
                    # 当数据量超过10万时，保存一次
                    if len(data_dicts) >= 20000:
                        logger.info(f"数据量达到{len(data_dicts)}，开始保存批次数据")
                        batch_result = await self._save_all_to_db_native_upsert(
                            model_class=StockBOLLIndicator,
                            data_list=data_dicts,
                            unique_fields=['stock', 'time_level', 'trade_time']
                        )
                        logger.info(f"批次数据保存完成，结果: {batch_result}")
                        
                        # 累加结果
                        for key in total_result:
                            total_result[key] += batch_result.get(key, 0)
                        
                        # 清空数据列表，准备下一批
                        data_dicts = []
                    
                    # logger.warning(f"当前data_dicts总量: {len(data_dicts)}")
                if not api_datas:
                    logger.warning(f"API未返回{stock.stock_code}股票的{time_level}级别历史BOLL指标数据")
                    return {'创建': 0, '更新': 0, '跳过': 0}
            # 保存剩余数据
            if data_dicts:
                final_result = await self._save_all_to_db_native_upsert(
                    model_class=StockBOLLIndicator,
                    data_list=data_dicts,
                    unique_fields=['stock', 'time_level', 'trade_time']
                )
                logger.info(f"剩余数据保存完成，结果: {final_result}")
                
                # 累加最终结果
                for key in total_result:
                    total_result[key] += final_result.get(key, 0)
            # --- 函数末尾执行最终修剪 ---
            for time_level in TIME_TEADE_TIME_LEVELS:
                # --- 生成缓存键 ---
                cache_key =  self.cache_key.history_boll(stock_code, time_level)
                # --- 单行调用修剪方法 ---
                removed_count = await self.cache_manager.trim_cache_zset(cache_key, self.cache_limit)
                # --- 修剪调用结束 ---
            
            # --- 最终修剪结束 ---
            logger.info(f"所有股票历史BOLL指标数据保存完成，总结果: {total_result}")
            return total_result
        except Exception as e:
            logger.error(f"保存{stock}股票历史BOLL指标数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_favorite_stocks_history_boll(self) -> Dict:
        """
        从API获取并保存自选股历史BOLL指标数据
        """
        favorite_stocks = await self.user_dao.get_all_favorite_stocks()
        try:
            for stock in favorite_stocks:
                await self.fetch_and_save_history_boll_by_stock_code(stock.stock_code)
        except Exception as e:
            logger.error(f"保存自选股历史BOLL指标数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}
            
    async def fetch_and_save_all_history_boll(self) -> Dict:
        """
        从API获取并保存所有股票历史BOLL指标数据
        """
        try:
            stocks = await self.stock_basic_dao.get_stock_list()
            for stock in stocks:
                await self.fetch_and_save_history_boll_by_stock_code(stock.stock_code)
        except Exception as e:
            logger.error(f"保存所有股票历史BOLL指标数据出错: {str(e)}")
            return {'创建': 0, '更新': 0, '跳过': 0}


