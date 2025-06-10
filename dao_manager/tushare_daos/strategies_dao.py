# dao_manager\tushare_daos\strategies_dao.py
import logging
from asgiref.sync import sync_to_async
from typing import List
from datetime import datetime
from dao_manager.base_dao import BaseDAO
from dao_manager.tushare_daos import stock_basic_info_dao
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from stock_models.stock_analytics import StockAnalysisResultTrendFollowing
from utils.cache_get import StrategyCacheGet
from utils.cache_set import StrategyCacheSet

logger = logging.getLogger("dao")

class StrategiesDAO(BaseDAO):
    def __init__(self):
        self.cache_set = StrategyCacheSet()
        self.cache_get = StrategyCacheGet()
    
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

    async def save_strategy_results(self, stock_code: str, timestamp: datetime, defaults_kwargs: dict):
        """
        保存策略分析结果到数据库，并根据时间戳判断是否写入Redis缓存。
        :param stock_code: 股票代码
        :param timestamp: 时间戳
        :param defaults_kwargs: 策略分析结果数据
        """
        stock_basic_info_dao = StockBasicInfoDao()
        # 异步获取股票对象
        stock_obj = await stock_basic_info_dao.get_stock_by_code(stock_code)
        if not stock_obj:
            print(f"未找到股票代码为{stock_code}的股票信息")
            return None

        # 用 sync_to_async 包装 update_or_create 方法，保证异步环境下数据库操作安全
        @sync_to_async
        def update_or_create_analysis():
            return StockAnalysisResultTrendFollowing.objects.update_or_create(
                stock=stock_obj,  # 外键直接传入 StockInfo 对象
                timestamp=timestamp,
                defaults=defaults_kwargs  # 传入所有其他字段作为 defaults
            )

        # 异步调用数据库保存方法
        analysis_record, created = await update_or_create_analysis()

        if created:
            print(f"[{stock_code}] 在时间点 {timestamp.strftime('%Y-%m-%d %H:%M')} 策略分析结果已成功创建。")
        else:
            print(f"[{stock_code}] 在时间点 {timestamp.strftime('%Y-%m-%d %H:%M')} 策略分析结果已成功更新。")

        # 1. 获取Redis缓存中的最新数据
        cache_data = await self.cache_get.lastest_analyze_signals_trend_following_data(stock_code)
        cache_ts = None
        if cache_data and 'timestamp' in cache_data:
            # 兼容字符串和datetime类型
            try:
                if isinstance(cache_data['timestamp'], str):
                    cache_ts = datetime.fromisoformat(cache_data['timestamp'])
                else:
                    cache_ts = cache_data['timestamp']
            except Exception as e:
                print(f"缓存时间戳解析失败: {e}")
                cache_ts = None

        # 2. 获取数据库最新的时间戳
        db_ts = analysis_record.timestamp

        # 3. 比较时间戳，数据库的更晚或Redis无数据时才写入Redis
        if (cache_ts is None) or (db_ts > cache_ts):
            # 组装要缓存的数据，假设defaults_kwargs里有所有需要的字段
            data_to_cache = dict(defaults_kwargs)
            data_to_cache['timestamp'] = db_ts.isoformat()  # 建议用ISO格式字符串
            # 写入Redis
            cache_result = await self.cache_set.lastest_analyze_signals_trend_following_data(stock_code, data_to_cache)
            print(f"写入Redis缓存结果: {cache_result}, 数据时间: {db_ts}")
        else:
            print(f"Redis缓存已是最新，无需更新。缓存时间: {cache_ts}, 数据库时间: {db_ts}")




















