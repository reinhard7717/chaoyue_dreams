# dao_manager\tushare_daos\strategies_dao.py
import logging
import time
from asgiref.sync import sync_to_async
from typing import List
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from api_manager.apis.stock_indicators_api import StockIndicatorsAPI
from dao_manager.base_dao import BaseDAO
from dao_manager.tushare_daos import stock_basic_info_dao
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from stock_models.stock_analytics import StockAnalysisResultTrendFollowing
from stock_models.stock_basic import StockInfo
from stock_models.time_trade import StockCyqChips, StockCyqPerf, StockDailyBasic, StockDailyData, StockMinuteData, StockWeeklyData, StockMonthlyData
from utils.cache_get import StockInfoCacheGet, StockTimeTradeCacheGet
from utils.cache_manager import CacheManager
from utils.cache_set import StockInfoCacheSet, StockTimeTradeCacheSet
from utils.cash_key import StockCashKey
from utils.data_format_process import StockInfoFormatProcess, StockTimeTradeFormatProcess

logger = logging.getLogger("dao")

class StrategiesDAO(BaseDAO):
    def __init__(self):
        pass
    
    async def get_latest_strategy_result(self, stock_code: str):
        """
        获取指定股票的最新策略信号。
        :param stock_code: 股票代码
        :return: 最新的策略信号对象或None
        """
        stock_basic_info_dao = StockBasicInfoDao()
        # 异步获取股票对象
        stock_obj = await stock_basic_info_dao.get_stock_by_code(stock_code)
        if not stock_obj:
            print(f"未找到股票代码为{stock_code}的股票信息")
            return None
        # 异步查询最新的策略信号，按时间倒序取第一个
        latest_strategy = await sync_to_async(
            lambda: StockAnalysisResultTrendFollowing.objects.filter(stock=stock_obj).order_by('-timestamp').first()
        )()
        if not latest_strategy:
            print(f"未找到股票{stock_code}的最新策略信号")
        return latest_strategy  # 返回最新策略信号对象
    
    async def get_strategy_result_by_timestamp(self, stock_code: str, timestamp: datetime):
        """
        获取指定股票在指定时间戳之前的所有策略信号。
        :param stock_code: 股票代码
        :param timestamp: 时间戳
        :return: 策略信号对象列表
        """
        stock_basic_info_dao = StockBasicInfoDao()
        # 异步获取股票对象
        stock_obj = await stock_basic_info_dao.get_stock_by_code(stock_code)
        if not stock_obj:
            print(f"未找到股票代码为{stock_code}的股票信息")
            return []
        # 异步查询指定时间戳之前的所有策略信号
        strategy_result = await sync_to_async(
            lambda: StockAnalysisResultTrendFollowing.objects.filter(stock=stock_obj, timestamp__lt=timestamp).first()
        )()
        if not strategy_result:
            print(f"未找到股票{stock_code}在{timestamp}的策略信号")
        return strategy_result  # 返回策略信号对象列表

    async def save_strategy_results(self, stock_code: str, timestamp: datetime, data: pd.DataFrame):
        """
        保存策略分析结果到数据库。
        :param stock_code: 股票代码
        :param timestamp: 时间戳
        :param data: 策略分析结果数据
        """
        stock_basic_info_dao = StockBasicInfoDao()
        # 异步获取股票对象
        stock_obj = await stock_basic_info_dao.get_stock_by_code(stock_code)
        if not stock_obj:
            print(f"未找到股票代码为{stock_code}的股票信息")
            return None
        # 异步保存策略分析结果到数据库
        await sync_to_async(
            lambda: StockAnalysisResultTrendFollowing.objects.update_or_create(stock=stock_obj, timestamp=timestamp, data=data)
        )()





















