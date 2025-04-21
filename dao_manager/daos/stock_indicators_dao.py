import json
import logging
import time as time_lib  # 用于测量时间
from typing import Dict, List, Optional, Union
from datetime import datetime
from asgiref.sync import sync_to_async
from core.constants import TIME_TEADE_TIME_LEVELS, TIME_TEADE_TIME_LEVELS_LITE, TIME_TEADE_TIME_LEVELS_PER_TRADING
from stock_models.indicator.ma import StockMAIndicator
from utils import cache_constants as cc # 导入常量
from django.utils import timezone

from api_manager.apis.stock_indicators_api import StockIndicatorsAPI, TimeLevel
from api_manager.mappings.stock_indicators_mapping import MA_INDICATOR_MAPPING
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
        self.stock_basic_dao = StockBasicDAO()
        self.cache_timeout = 300  # 默认缓存5分钟
        self.cache_limit = 333 # 定义缓存数量上限
        self.user_dao = UserDAO()
        self.cache_key = StockCashKey()
        self.data_format_process = StockIndicatorsDataFormatProcess()
        self.cache_manager = None
        self.cache_get = None
        self.cache_set = None

    async def initialize_cache_objects(self):
        self.cache_manager = CacheManager()  # 先实例化
        await self.cache_manager.initialize()  # 然后 await 其异步初始化方法，如果存在

        self.cache_set = StockIndicatorsCacheSet()  # 先实例化
        await self.cache_set.initialize()  # 添加异步初始化方法，如果需要

        self.cache_get = StockIndicatorsCacheGet()  # 先实例化
        await self.cache_get.initialize()  # 添加异步初始化方法，如果需要

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
        if self.cache_get is None:
            await self.initialize_cache_objects()
        try:
            cache_data = await self.cache_get.latest_time_trade(stock_code, time_level)
            if cache_data:
                time_trade_dict = json.loads(cache_data)
                time_trade = StockTimeTrade(**time_trade_dict)
                return time_trade
        except Exception as e:
            logger.error(f"从缓存获取最新股票[{stock_code}]{time_level}级别分时成交数据时发生异常: {str(e)}")
            return None
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
            start_time: 开始时间
            end_time: 结束时间
        Returns:
            List[StockTimeTrade]: 历史分时成交数据列表
        """
        try:
            if self.cache_get is None:
                await self.initialize_cache_objects()
            cache_data = await self.cache_get.history_data(stock_code, time_level, start_time, end_time)
            if cache_data:
                return cache_data
        except Exception as e:
            logger.error(f"从缓存获取股票[{stock_code}]{time_level}级别历史分时成交数据时发生异常: {str(e)}")
            return None
        # 从数据库获取最新数据
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return None
        try:
            if self.cache_set is None:
                await self.initialize_cache_objects()
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
        try:
            if self.cache_get is None:
                await self.initialize_cache_objects()
            stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
            if not stock:
                return None
            # 从Redis获取数据
            cache_data = await self.cache_get.history_time_trade_by_limit(stock_code, time_level, limit)
            if cache_data:
                return cache_data
        except Exception as e:
            logger.error(f"从缓存获取股票[{stock_code}]{time_level}级别历史分时成交数据时发生异常: {str(e)}")
            return None
        
        # 从数据库获取数据
        try:
            if self.cache_set is None:
                await self.initialize_cache_objects()
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
            if self.cache_set is None:
                await self.initialize_cache_objects()
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
                    await self.cache_set.history_time_trade(stock.stock_code, time_level, cache_dict) 
                    # 保存数据
                    result = await self._save_all_to_db_native_upsert(
                        model_class=StockTimeTrade,
                        data_list=data_dicts,
                        unique_fields=['stock', 'time_level', 'trade_time']
                    )
                    # --- 函数末尾执行最终修剪 ---
                    cache_key =  self.cache_key.history_time_trade(stock_code, time_level)
                    removed_count = await self.cache_manager.ztrim_by_rank(cache_key, self.cache_limit)
                    # --- 修剪调用结束 ---
                    logger.info(f"股票[{stock}] {time_level}级别分时成交数据保存完成，结果: {result}")
                    return result
        except Exception as e:
            logger.error(f"保存{stock}股票{time_level}级别  分时成交数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_latest_time_trade_by_time_level(self, time_level: Union[TimeLevel, str]) -> Dict:
        """
        从API获取并保存最新分时成交数据
        Args:
            stock_code: 股票代码
            time_level: 时间级别
        Returns:
            Optional[StockTimeTrade]: 保存的数据
        """
        # 获取股票信息
        stocks = await self.stock_basic_dao.get_stock_list()
        if not stocks:
            return {'创建': 0, '更新': 0, '跳过': 0}
        if self.cache_set is None:
                await self.initialize_cache_objects()
        data_dicts = []
        try:
            for i, stock in enumerate(stocks):
                api_data = await self.api.get_time_trade(stock.stock_code, time_level)
                if api_data:
                    data_dict = self.data_format_process.set_time_trade_data(stock, time_level, api_data)
                    if data_dict.get('trade_time') is None:
                        # logger.warning(f"API未返回{stock} {time_level}级别时间序列数据")
                        pass
                    else:
                        data_dicts.append(data_dict)
                        cache_dict = data_dict.copy()
                        await self.cache_set.history_time_trade(stock.stock_code, time_level, cache_dict)
                        if i % 100 == 0:
                            logger.info(f"{time_level}级别分时成交数据获取完成 {i}/{len(stocks)}")
                # --- 函数末尾执行最终修剪 ---
                cache_key =  self.cache_key.history_time_trade(stock.stock_code, time_level)
                await self.cache_manager.ztrim_by_rank(cache_key, self.cache_limit)
                # --- 修剪调用结束 ---
            # 保存数据
            result = await self._save_all_to_db_native_upsert(
                model_class=StockTimeTrade,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            
            logger.info(f"股票[{stock}] {time_level}级别分时成交数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存{stock}股票{time_level}级别  分时成交数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_latest_time_trade_by_time_level_and_stock_codes(self, stock_codes: List[str], time_level: Union[TimeLevel, str]) -> Dict:
        """
        从API获取并保存最新分时成交数据
        Args:
            stock_code: 股票代码
            time_level: 时间级别
        Returns:
            Optional[StockTimeTrade]: 保存的数据
        """
        # 获取股票信息
        stocks = await self.stock_basic_dao.get_stock_list()
        if not stocks:
            return {'创建': 0, '更新': 0, '跳过': 0}
        if self.cache_set is None:
                await self.initialize_cache_objects()
        data_dicts = []
        process_start_time = time_lib.time()
        stocks_count = len(stock_codes)
        finished_count = 0
        try:
            for i, stock_code in enumerate(stock_codes):
                loop_start_time = time_lib.time()
                if process_start_time is None:
                    process_start_time = loop_start_time
                api_data = await self.api.get_time_trade(stock_code, time_level)
                if api_data:
                    stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
                    data_dict = self.data_format_process.set_time_trade_data(stock, time_level, api_data)
                    if data_dict.get('trade_time') is None:
                        # logger.warning(f"API未返回{stock} {time_level}级别时间序列数据")
                        pass
                    else:
                        data_dicts.append(data_dict)
                        cache_dict = data_dict.copy()
                        await self.cache_set.history_time_trade(stock.stock_code, time_level, cache_dict)
                        if i % 100 == 0:
                            logger.info(f"{time_level}级别分时成交数据获取完成 {i}/{len(stocks)}")
                # --- 函数末尾执行最终修剪 ---
                cache_key =  self.cache_key.history_time_trade(stock.stock_code, time_level)
                await self.cache_manager.ztrim_by_rank(cache_key, self.cache_limit)
                # --- 修剪调用结束 ---
            # 保存数据
            result = await self._save_all_to_db_native_upsert(
                model_class=StockTimeTrade,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            process_end_time = time_lib.time()
            process_duration = process_end_time - process_start_time
            finished_count += len(data_dicts)
            logger.info(f"{finished_count} / {stocks_count} 个股票实时数据保存完成, 耗时: {process_duration} 秒，平均每秒处理 {len(data_dicts) / process_duration} 个股票")
            process_start_time = None
            return result
        except Exception as e:
            logger.warning(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}", exc_info=True)
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
        if self.cache_set is None:
            await self.initialize_cache_objects()
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
                    await self.cache_manager.ztrim_by_rank(cache_key, self.cache_limit)
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
        if self.cache_set is None:
            await self.initialize_cache_objects()
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
            await self.cache_manager.ztrim_by_rank(cache_key, self.cache_limit)
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
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if self.cache_set is None:
            await self.initialize_cache_objects()
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
            await self.cache_manager.ztrim_by_rank(cache_key, self.cache_limit)
            # --- 修剪调用结束 ---
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
            if self.cache_set is None:
                await self.initialize_cache_objects()
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
                await self.cache_manager.ztrim_by_rank(cache_key, self.cache_limit)
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
        if self.cache_set is None:
            await self.initialize_cache_objects()
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
            if self.cache_set is None:
                await self.initialize_cache_objects()
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
            if self.cache_set is None:
                await self.initialize_cache_objects()
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
        if self.cache_set is None:
            await self.initialize_cache_objects()
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
            if self.cache_set is None:
                await self.initialize_cache_objects()
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
            await self.cache_manager.ztrim_by_rank(cache_key, self.cache_limit)
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
                await self.cache_manager.ztrim_by_rank(cache_key, self.cache_limit)
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
    

