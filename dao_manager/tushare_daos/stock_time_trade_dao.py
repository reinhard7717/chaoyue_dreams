# dao_manager\tushare_daos\stock_time_trade_dao.py
import asyncio
import logging
import time
from asgiref.sync import sync_to_async
from typing import Dict, List, Optional
from collections import defaultdict # 导入 defaultdict 以方便分组
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from dao_manager.base_dao import BaseDAO
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from stock_models.stock_basic import StockInfo
from stock_models.time_trade import StockCyqChips, StockCyqPerf, StockDailyBasic, StockMinuteData, StockWeeklyData, StockMonthlyData
from utils.cache_get import StockInfoCacheGet, StockTimeTradeCacheGet
from utils.cache_manager import CacheManager
from utils.cache_set import StockInfoCacheSet, StockTimeTradeCacheSet
from utils.cash_key import StockCashKey
from utils.data_format_process import StockInfoFormatProcess, StockTimeTradeFormatProcess
from stock_models.time_trade import StockDailyData_SZ, StockDailyData_SH, StockDailyData_CY, StockDailyData_KC, StockDailyData_BJ
from stock_models.time_trade import (
            StockMinuteData_5_SZ, StockMinuteData_5_SH, StockMinuteData_5_BJ, StockMinuteData_5_CY, StockMinuteData_5_KC,
            StockMinuteData_15_SZ, StockMinuteData_15_SH, StockMinuteData_15_BJ, StockMinuteData_15_CY, StockMinuteData_15_KC,
            StockMinuteData_30_SZ, StockMinuteData_30_SH, StockMinuteData_30_BJ, StockMinuteData_30_CY, StockMinuteData_30_KC,
            StockMinuteData_60_SZ, StockMinuteData_60_SH, StockMinuteData_60_BJ, StockMinuteData_60_CY, StockMinuteData_60_KC,
        )

BATCH_SAVE_SIZE = 10000  # 每10000条数据保存一次
logger = logging.getLogger("dao")
time_levels = ["5", "15", "30", "60"] # "1", 

class StockTimeTradeDAO(BaseDAO):
    def __init__(self):
        """初始化StockIndicatorsDAO"""
        super().__init__(None, None, 3600)  # 基类使用None作为model_class，因为本DAO管理多个模型
        self.stock_basic_dao = StockBasicInfoDao()
        self.cache_limit = 500 # 定义缓存数量上限
        self.cache_manager = CacheManager()
        self.cache_key = StockCashKey()
        self.data_format_process_trade = StockTimeTradeFormatProcess()
        self.data_format_process_stock = StockInfoFormatProcess()
        self.cache_set = StockTimeTradeCacheSet()
        self.cache_get = StockTimeTradeCacheGet()
        self.stock_cache_set = StockInfoCacheSet()
        self.stock_cache_get = StockInfoCacheGet()

    # =============== A股日线行情 ===============
    def get_daily_data_model_by_code(self, stock_code: str):
        """
        根据股票代码返回对应的日线数据表Model
        """
        if stock_code.startswith('3') and stock_code.endswith('.SZ'):
            return StockDailyData_CY
        elif stock_code.endswith('.SZ'):
            return StockDailyData_SZ
        elif stock_code.startswith('68') and stock_code.endswith('.SH'):
            return StockDailyData_KC
        elif stock_code.endswith('.SH'):
            return StockDailyData_SH
        elif stock_code.endswith('.BJ'):
            return StockDailyData_BJ
        else:
            print(f"未识别的股票代码: {stock_code}，默认使用SZ主板表")
            return StockDailyData_SZ  # 默认返回深市主板

    @sync_to_async
    def get_stocks_daily_data(self, stock_codes: List[str], trade_date: datetime.date) -> List:
        """
        【优化版】批量获取多支股票在指定日期的日线行情数据。
        此方法会根据股票代码自动从对应的分表中查询数据。
        Args:
            stock_codes (List[str]): 股票代码列表。
            trade_date (datetime.date): 交易日期。
        Returns:
            List: 包含来自不同表的日线行情模型实例的混合列表。
        """
        if not stock_codes:
            return []
        
        # 代码修改处: 整个方法的逻辑被重写以支持分表查询
        # print(f"    [DAO] 准备按板块分表查询 {len(stock_codes)} 支股票在 {trade_date} 的日线行情...")

        # 步骤1: 按模型对股票代码进行分组
        # 使用 defaultdict 可以简化代码，无需检查key是否存在
        model_to_codes_map = defaultdict(list)
        for code in stock_codes:
            model_class = self.get_daily_data_model_by_code(code)
            model_to_codes_map[model_class].append(code)

        all_daily_data = [] # 用于存储所有查询结果的列表

        # 步骤2 & 3: 对每个分组进行批量查询并合并结果
        for model_class, codes_for_this_model in model_to_codes_map.items():
            # print(f"        -> 正在查询模型 {model_class.__name__} 中的 {len(codes_for_this_model)} 支股票...")
            try:
                # 对当前分组的股票代码执行一次高效的批量查询
                daily_data_batch = list(
                    model_class.objects.filter(
                        stock__stock_code__in=codes_for_this_model,
                        trade_time=trade_date
                    )
                )
                # 将查询结果合并到总列表中
                all_daily_data.extend(daily_data_batch)
                # print(f"        <- 从 {model_class.__name__} 查询到 {len(daily_data_batch)} 条数据。")
            except Exception as e:
                logger.error(f"在模型 {model_class.__name__} 中批量查询股票日线行情时出错: {e}", exc_info=True)
                # 即使一个分表查询失败，也继续查询其他表
                continue
        
        # print(f"    [DAO] 查询完成，共获取到 {len(all_daily_data)} 条日线行情数据。")
        return all_daily_data

    async def get_latest_daily_quote(self, stock_code: str) -> Optional[Dict]:
        """
        【V1.0 - 新增】
        获取指定股票的最新一条日线行情数据。
        - 自动根据股票代码选择正确的分表进行查询。
        - 使用高效的异步ORM调用 (.afirst())。
        - 返回一个字典或None。
        Args:
            stock_code (str): 股票代码, e.g., '300094.SZ'.
        Returns:
            Optional[Dict]: 包含最新行情数据的字典，或在找不到数据时返回None。
        """
        try:
            # 1. 使用现有逻辑确定正确的模型
            model_class = self.get_daily_data_model_by_code(stock_code)
            
            # 2. 异步查询该模型，按时间倒序排列并获取第一条记录
            # .afirst() 是 Django 异步ORM中最高效的获取单条记录的方式
            latest_quote_obj = await model_class.objects.filter(
                stock__stock_code=stock_code
            ).order_by('-trade_time').afirst()

            if not latest_quote_obj:
                logger.warning(f"[DAO] 未能在 {model_class.__name__} 表中找到股票 {stock_code} 的任何日线数据。")
                return None

            # 3. 将模型对象手动转换为字典，以便于在同步代码中使用
            # 这样可以精确控制返回的字段，避免序列化整个复杂对象
            return {
                "trade_date": latest_quote_obj.trade_time.strftime('%Y-%m-%d'),
                "close": float(latest_quote_obj.close),
                "pct_chg": float(latest_quote_obj.pct_change),
            }
        except Exception as e:
            logger.error(f"[DAO] 获取 {stock_code} 最新日线行情时发生错误: {e}", exc_info=True)
            return None

    async def save_daily_time_trade_history_by_trade_date(self, trade_date: date) -> None:
        """
        保存指定交易日的所有股票日线交易数据，自动分表
        """
        trade_date_str = ""
        if isinstance(trade_date, str):
            trade_date_str = trade_date
        elif isinstance(trade_date, date):
            trade_date_str = trade_date.strftime("%Y%m%d")
        else:
            raise ValueError("trade_date 必须是 str 或 date 类型")
        df = self.ts_pro.stk_factor(
            **{
                "ts_code": "", "trade_date": trade_date_str, "start_date": "",
                "end_date": "", "offset": "", "limit": ""
            },
            fields=[
                "ts_code", "trade_date", "close", "open", "high", "low", "pre_close", "change", "pct_change", "vol", "amount", "adj_factor",
                "open_hfq", "open_qfq", "close_hfq", "close_qfq", "high_hfq", "high_qfq", "low_hfq", "low_qfq", "pre_close_hfq", "pre_close_qfq"
            ]
        )
        if not df.empty:
            df = df.replace(['nan', 'NaN', ''], np.nan)
            df = df.where(pd.notnull(df), None)
            # 按分表分组
            data_dicts_by_model = {}
            for row in df.itertuples():
                stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process_trade.set_time_trade_day_data(stock=stock, df_data=row)
                    model_class = self.get_daily_data_model_by_code(row.ts_code)
                    if model_class not in data_dicts_by_model:
                        data_dicts_by_model[model_class] = []
                    data_dicts_by_model[model_class].append(data_dict)
            result = {}
            for model_class, data_list in data_dicts_by_model.items():
                res = await self._save_all_to_db_native_upsert(
                    model_class=model_class,
                    data_list=data_list,
                    unique_fields=['stock', 'trade_time']
                )
                result[model_class.__name__] = res
        else:
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        return result

    async def save_daily_time_trade_history_by_trade_dates(self, trade_date: date, start_date: date = None, end_date: date = None) -> Dict:
        """
        保存指定日期区间的所有股票日线交易数据，自动分表 (完全向量化优化版)
        1. 一次性预加载所有股票信息，根除N+1查询。
        2. 对每一页数据进行完全向量化处理，包括对象映射和模型分类。
        3. 处理完一页数据后立即分组并存入数据库，有效控制内存。
        """
        # --- 在所有循环开始前，一次性预加载全部股票信息 ---
        print("正在预加载所有股票的基础信息...")
        # 1. 调用您提供的方法获取所有股票对象列表
        all_stocks_list = await self.stock_basic_dao.get_stock_list()
        if not all_stocks_list:
            logger.warning("未能从数据库或缓存中获取任何股票信息，任务终止。")
            return {}
        
        # 2. 构建一个以ts_code为键，StockInfo对象为值的高效查找字典
        stock_map = {stock.stock_code: stock for stock in all_stocks_list}
        print(f"股票信息预加载完成，共加载 {len(stock_map)} 条。")
        

        trade_date_str = trade_date.strftime('%Y%m%d') if trade_date else ""
        start_date_str = start_date.strftime('%Y%m%d') if start_date else "20250101"
        end_date_str = end_date.strftime('%Y%m%d') if end_date else ""
        
        offset = 0
        limit = 6000
        page_num = 1
        all_results = {} # 用于收集所有保存操作的结果

        while True:
            if offset >= 100000:
                logger.warning(f"offset已达10万，停止拉取。{start_date_str} - {end_date_str}, freq=Day")
                break
            
            print(f"正在拉取第 {page_num} 页日线数据... offset={offset}")
            df = self.ts_pro.stk_factor(
                **{
                    "ts_code": "", "trade_date": trade_date_str, "start_date": start_date_str,
                    "end_date": end_date_str, "offset": offset, "limit": limit
                },
                fields=[
                    "ts_code", "trade_date", "close", "open", "high", "low", "pre_close", "change", "pct_change", "vol",
                    "amount", "adj_factor", "open_hfq", "open_qfq", "close_hfq", "close_qfq", "high_hfq", "high_qfq", "low_hfq",
                    "low_qfq", "pre_close_hfq", "pre_close_qfq"
                ]
            )
            if df.empty:
                print("拉取结束，未返回更多数据。")
                break

            # --- 对整页DataFrame进行向量化处理，彻底替代for循环 ---
            # 1. 数据清洗
            df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
            
            # 2. 向量化映射stock对象 (使用预加载的全局stock_map)
            df['stock'] = df['ts_code'].map(stock_map)
            
            # 3. 丢弃关键字段为空或在数据库中找不到对应stock的行
            df.dropna(subset=['trade_date', 'stock'], inplace=True)

            if df.empty:
                print(f"当前页数据经清洗后为空，跳至下一页。")
                if len(df) < limit: # 原始df长度判断分页是否结束
                    break
                offset += limit
                page_num += 1
                continue

            # 4. 向量化转换日期格式
            df['trade_time'] = pd.to_datetime(df['trade_date']).dt.date
            
            # 5. 向量化应用函数，为每行数据动态确定其应存入的模型类
            df['model_class'] = df['ts_code'].apply(self.get_daily_data_model_by_code)
            

            # --- 页级处理，立即使用groupby对DataFrame进行高效分组并保存 ---
            for model_class, group_df in df.groupby('model_class'):
                if group_df.empty:
                    continue
                
                # 6. 从分组后的DataFrame中直接选择所需列，并转换为字典列表
                #    这步替代了 set_time_trade_day_data 方法
                data_list = group_df[[
                    "stock", "trade_time", "close", "open", "high", "low", "pre_close", "change", "pct_change", "vol",
                    "amount", "adj_factor", "open_hfq", "open_qfq", "close_hfq", "close_qfq", "high_hfq", "high_qfq", "low_hfq",
                    "low_qfq", "pre_close_hfq", "pre_close_qfq"
                ]].to_dict('records')

                # 7. 批量保存该模型的数据
                res = await self._save_all_to_db_native_upsert(
                    model_class=model_class,
                    data_list=data_list,
                    unique_fields=['stock', 'trade_time']
                )
                # 汇总结果
                model_name = model_class.__name__
                if model_name not in all_results:
                    all_results[model_name] = []
                all_results[model_name].extend(res)
                logger.info(f"保存 {model_name} 的日线数据完成. 插入/更新了 {len(data_list)} 条记录。")
            

            if len(df) < limit:
                break
            offset += limit
            page_num += 1
            
        logger.info(f"指定日期区间的日线数据保存任务全部完成。")
        return all_results

    async def save_daily_time_trade_history_by_stock_code(self, stock_code: str) -> None:
        """
        保存指定股票的历史日线交易数据，自动分表
        """
        df = self.ts_pro.stk_factor(
            **{
                "ts_code": stock_code, "trade_date": "", "start_date": "",
                "end_date": "", "offset": "", "limit": ""
            },
            fields=[
                "ts_code", "trade_date", "close", "open", "high", "low", "pre_close", "change", "pct_change", "vol", "amount", "adj_factor",
                "open_hfq", "open_qfq", "close_hfq", "close_qfq", "high_hfq", "high_qfq", "low_hfq", "low_qfq", "pre_close_hfq", "pre_close_qfq"
            ]
        )
        if not df.empty:
            df = df.replace(['nan', 'NaN', ''], np.nan)
            df = df.where(pd.notnull(df), None)
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process_trade.set_time_trade_day_data(stock=stock, df_data=row)
                    data_dicts.append(data_dict)
            if data_dicts:
                model_class = self.get_daily_data_model_by_code(stock_code)
                result = await self._save_all_to_db_native_upsert(
                    model_class=model_class,
                    data_list=data_dicts,
                    unique_fields=['stock', 'trade_time']
                )
        else:
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        return result

    async def save_daily_time_trade_history_by_stock_codes(self, stock_codes: List[str]) -> None:
        """
        保存多只股票的历史日线交易数据，自动分表
        """
        stock_codes_str = ",".join(stock_codes)
        data_dicts_by_model = {}
        offset = 0
        limit = 6000
        print(f"开始日线历史任务：{stock_codes_str}")
        while True:
            if offset >= 100000:
                logger.warning(f"offset已达10万，停止拉取。ts_code={stock_codes_str}, freq=Day")
                break
            df = self.ts_pro.stk_factor(
                **{
                    "ts_code": stock_codes_str, "trade_date": "", "start_date": "2000-01-01 00:00:00",
                    "end_date": "", "offset": offset, "limit": limit
                },
                fields=[
                    "ts_code", "trade_date", "close", "open", "high", "low", "pre_close", "change", "pct_change", "vol",
                    "amount", "adj_factor", "open_hfq", "open_qfq", "close_hfq", "close_qfq", "high_hfq", "high_qfq", "low_hfq",
                    "low_qfq", "pre_close_hfq", "pre_close_qfq"
                ]
            )
            if df.empty:
                break
            else:
                df = df.replace(['nan', 'NaN', ''], np.nan)
                df = df.where(pd.notnull(df), None)
                for row in df.itertuples():
                    stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                    if stock:
                        data_dict = self.data_format_process_trade.set_time_trade_day_data(stock=stock, df_data=row)
                        model_class = self.get_daily_data_model_by_code(row.ts_code)
                        if model_class not in data_dicts_by_model:
                            data_dicts_by_model[model_class] = []
                        data_dicts_by_model[model_class].append(data_dict)
            if len(df) < limit:
                break
            offset += limit
        result = {}
        for model_class, data_list in data_dicts_by_model.items():
            res = await self._save_all_to_db_native_upsert(
                model_class=model_class,
                data_list=data_list,
                unique_fields=['stock', 'trade_time']
            )
            result[model_class.__name__] = res
        return result

    async def save_daily_time_trade_yesterday(self) -> None:
        """
        保存股票的今日日线交易数据
        接口：daily
        数据说明：交易日每天15点～16点之间入库。本接口是未复权行情，停牌期间不提供数据
        调取说明：120积分每分钟内最多调取500次，每次6000条数据，相当于单次提取23年历史
        """
        # 获取当前日期
        today = datetime.today()
        yesterday = today - timedelta(days=1)  # 用timedelta减去1天，得到昨天的日期时间
        # 转换为YYYYMMDD格式
        day_str = yesterday.strftime('%Y%m%d')
        result = await self.save_daily_time_trade_history_by_trade_date(day_str)
        return result

    # 未复权信息，慎用
    async def save_daily_time_trade_realtime(self, stock_code: str) -> None:
        """
        保存指定股票的实时日线交易数据，自动分表
        """
        df = self.ts_pro.rt_k(
            **{
                "topic": "", "ts_code": stock_code, "limit": "", "offset": ""
            },
            fields=[
                "ts_code", "name", "pre_close", "high", "open", "low", "close", "vol", "amount", "num"
            ]
        )
        if not df.empty:
            df = df.replace(['nan', 'NaN', ''], np.nan)
            df = df.where(pd.notnull(df), None)
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process_trade.set_time_trade_day_data(stock=stock, df_data=row)
                    data_dicts.append(data_dict)
            model_class = self.get_daily_data_model_by_code(stock_code)
            result = await self._save_all_to_db_native_upsert(
                model_class=model_class,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time']
            )
        else:
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        return result

    async def get_daily_time_trade_history_by_stock_code(self, stock_code: str) -> None:
        """
        获取指定股票的历史日线交易数据，自动分表
        """
        # 先尝试从Redis缓存获取
        cache_key = self.cache_key.history_time_trade(stock_code, "Day")
        data_dicts = await self.cache_get.history_time_trade(cache_key)
        # 路由到正确的分表Model
        model_class = self.get_daily_data_model_by_code(stock_code)
        stock_daily_data_list = []
        if data_dicts:
            # 如果缓存有数据，直接反序列化为Model实例
            for data_dict in data_dicts:
                stock_daily_data_list.append(model_class(**data_dict))
            return stock_daily_data_list
        # 缓存没有数据，则从数据库查找对应分表
        stock_daily_data_list = model_class.objects.filter(stock_code=stock_code).order_by('-trade_time')[:self.cache_limit]
        return stock_daily_data_list

    # =============== A股分钟行情 ===============
    def get_minute_model(self, stock_code: str, time_level: str):
        """
        根据stock_code和time_level返回对应的分钟数据模型
        """
        # 只分5/15/30/60，1min默认用原表
        if time_level not in ['5', '15', '30', '60']:
            return StockMinuteData
        if stock_code.endswith('.SZ'):
            if stock_code.startswith('3'):
                return {
                    '5': StockMinuteData_5_CY, '15': StockMinuteData_15_CY, '30': StockMinuteData_30_CY, '60': StockMinuteData_60_CY
                }[time_level]
            else:
                return {
                    '5': StockMinuteData_5_SZ, '15': StockMinuteData_15_SZ, '30': StockMinuteData_30_SZ, '60': StockMinuteData_60_SZ
                }[time_level]
        elif stock_code.endswith('.SH'):
            if stock_code.startswith('68'):
                return {
                    '5': StockMinuteData_5_KC, '15': StockMinuteData_15_KC, '30': StockMinuteData_30_KC, '60': StockMinuteData_60_KC
                }[time_level]
            else:
                return {
                    '5': StockMinuteData_5_SH, '15': StockMinuteData_15_SH, '30': StockMinuteData_30_SH, '60': StockMinuteData_60_SH
                }[time_level]
        elif stock_code.endswith('.BJ'):
            return {
                '5': StockMinuteData_5_BJ, '15': StockMinuteData_15_BJ, '30': StockMinuteData_30_BJ, '60': StockMinuteData_60_BJ
            }[time_level]
        else:
            return StockMinuteData

    async def get_5_min_kline_time_by_day(self, stock_code: str, date: datetime.date = None) -> List[str]:
        """
        获取指定日期当天的所有5分钟K线的交易时间
        """
        if not date:
            date = datetime.today().date()
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            print(f"未找到股票代码：{stock_code}")
            return []
        start_datetime = datetime.combine(date, datetime.min.time())
        end_datetime = start_datetime + timedelta(days=1)
        # 用sync_to_async包装ORM查询
        @sync_to_async
        def get_trade_times():
            model = self.get_minute_model(stock_code, '5')  # 修改：自动分表
            qs = model.objects.filter(
                stock=stock,
                trade_time__gte=start_datetime,
                trade_time__lt=end_datetime
            ).values_list('trade_time', flat=True)
            return list(qs)
        records = await get_trade_times()
        trade_times = [record.strftime('%Y-%m-%d %H:%M:%S') for record in records]
        return trade_times

    async def get_latest_5_min_kline(self, stock_code: str) -> Optional[Dict]:
        """
        获取指定股票最新一条5分钟K线数据
        """
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            print(f"未找到股票代码：{stock_code}")
            return None
        cache_data = await self.cache_get.latest_time_trade(stock_code=stock_code, time_level=5)
        if cache_data is not None:
            stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
            cache_data['stock'] = stock
            return self.get_minute_model(stock_code, '5')(**cache_data)  # 修改：自动分表
        @sync_to_async
        def get_latest_kline():
            model = self.get_minute_model(stock_code, '5')  # 修改：自动分表
            record = (model.objects
                    .filter(stock=stock)
                    .order_by('-trade_time')
                    .first())
            return record
        latest_kline = await get_latest_kline()
        if not latest_kline:
            print(f"{stock_code} 未查询到5分钟K线数据")
        return latest_kline

    async def save_minute_time_trade_history_by_stock_codes(self, stock_codes: List[str], start_date_str: str="2020-01-01 00:00:00", end_date_str: str="") -> None:
        """
        保存股票的历史分钟级交易数据 (完全向量化优化版)
        1. 一次性预加载全部所需股票信息，根除所有N+1查询（无论是行级还是页级）。
        2. 对每一页数据进行完全向量化处理，包括数据清洗、对象映射和模型分类。
        3. 使用`groupby`对处理后的DataFrame进行高效分组，并分批保存。
        """
        if not stock_codes:
            logger.warning("输入的股票代码列表为空，任务终止。")
            return

        # --- 在所有循环开始前，一次性预加载全部股票信息 ---
        print(f"正在批量预加载 {len(stock_codes)} 只股票的基础信息...")
        stock_map = await self.stock_basic_dao.get_stocks_by_codes(stock_codes)
        if not stock_map:
            logger.warning(f"根据提供的代码列表，未能从数据库中找到任何股票信息，任务终止。")
            return
        print(f"股票信息预加载完成，共加载 {len(stock_map)} 条。")
        

        stock_codes_str = ",".join(stock_codes)
        for time_level in ['5', '15', '30', '60']:
            offset = 0
            limit = 8000
            page_num = 1
            while True:
                if offset >= 100000:
                    logger.warning(f"offset已达10万，停止拉取。ts_code={stock_codes_str}, freq={time_level}min")
                    break
                
                print(f"正在拉取 {time_level}min 数据, 第 {page_num} 页, offset: {offset}")
                df = self.ts_pro.stk_mins(**{
                    "ts_code": stock_codes_str, "freq": time_level + "min", "start_date": start_date_str, "end_date": end_date_str, 
                    "limit": limit, "offset": offset
                }, fields=[ "ts_code", "trade_time", "close", "open", "high", "low", "vol", "amount", "freq" ])
                
                if df.empty:
                    print(f"拉取结束，未返回更多 {time_level}min 数据。")
                    break
                
                # --- 对整页DataFrame进行向量化处理，彻底替代for循环 ---
                # 1. 数据清洗
                df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
                
                # 2. 向量化映射stock对象 (使用预加载的全局stock_map)
                df['stock'] = df['ts_code'].map(stock_map)
                
                # 3. 丢弃关键字段为空或在数据库中找不到对应stock的行
                df.dropna(subset=['trade_time', 'stock'], inplace=True)

                if df.empty:
                    print(f"当前页数据经清洗后为空，跳至下一页。")
                    if len(df) < limit: # 原始df长度判断分页是否结束
                        break
                    offset += limit
                    page_num += 1
                    continue

                # 4. 向量化转换日期时间格式
                df['trade_time'] = pd.to_datetime(df['trade_time'])

                # 5. 向量化应用函数，为每行数据动态确定其应存入的模型类
                df['model_class'] = df['ts_code'].apply(lambda code: self.get_minute_model(code, time_level))
                

                # --- 使用groupby对处理好的DataFrame进行高效分组并保存 ---
                for model_class, group_df in df.groupby('model_class'):
                    if group_df.empty:
                        continue
                    
                    # 6. 从分组后的DataFrame中直接选择所需列，并转换为字典列表
                    data_list = group_df[[
                        "stock", "trade_time", "close", "open", "high", "low", "vol", "amount", "freq"
                    ]].to_dict('records')

                    # 7. 批量保存该模型的数据
                    await self._save_all_to_db_native_upsert(
                        model_class=model_class,
                        data_list=data_list,
                        unique_fields=['stock', 'trade_time']
                    )
                    logger.info(f"保存 {model_class.__name__} 的 {time_level}分钟级数据完成. 插入/更新了 {len(data_list)} 条记录。")
                

                if len(df) < limit:
                    break
                offset += limit
                page_num += 1

        logger.info(f"保存 {len(stock_codes)}个股票 的分钟级交易数据全部完成.")
        return

    async def save_minute_time_trade_history_by_stock_code_and_time_level(self, stock_code: str, time_level: str, trade_date: date=None, start_date: date=None, end_date: date=None) -> None:
        """
        保存股票的历史分钟级交易数据 (优化版)
        1. 预先获取stock对象和目标Model，避免循环内重复操作。
        2. 全程使用Pandas向量化操作，替代逐行处理，大幅提升性能。
        3. 引入分批保存机制，有效控制内存占用，适合海量分钟线数据。
        4. 简化了数据分组逻辑，代码更清晰高效。
        """
        # --- 在循环外一次性获取所有必要的前置对象 ---
        # 1. 获取股票基础信息
        print(f"正在获取股票 {stock_code} 的基础信息...")
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            logger.error(f"未能找到股票代码为 {stock_code} 的股票信息，任务终止。")
            return []

        # 2. 一次性确定要操作的数据库模型
        print(f"正在确定时间级别为 {time_level}min 的目标数据表...")
        model_class = self.get_minute_model(stock_code, time_level)
        if not model_class:
            logger.error(f"未能为 {stock_code} 和时间级别 {time_level}min 找到对应的数据库模型，任务终止。")
            return []
        print(f"目标数据表确定为: {model_class.__name__}")
        

        # 准备API参数
        start_date_str = start_date.strftime('%Y-%m-%d %H:%M:%S') if start_date else "2020-01-01 00:00:00"
        end_date_str = end_date.strftime('%Y-%m-%d %H:%M:%S') if end_date else ""

        # --- 初始化用于分批保存的列表和批次大小 ---
        all_data_dicts = []
        total_saved_count = 0
        

        offset = 0
        limit = 8000
        page_num = 1
        while True:
            if offset >= 100000:
                logger.warning(f"offset已达10万，停止拉取。{stock_code}, time_level={time_level}min")
                break
            
            print(f"正在拉取第 {page_num} 页分钟线数据... offset={offset}")
            df = self.ts_pro.stk_mins(**{
                "ts_code": stock_code, "freq": time_level + "min", "start_date": start_date_str, "end_date": end_date_str, "limit": limit, "offset": offset
            }, fields=[ "ts_code", "trade_time", "close", "open", "high", "low", "vol", "amount", "freq" ])
            
            if df.empty:
                print("拉取结束，未返回更多数据。")
                break

            # --- 对整页DataFrame进行向量化处理，替代for循环 ---
            # 1. 数据清洗
            df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
            df.dropna(subset=['trade_time'], inplace=True) # 确保关键字段存在

            if not df.empty:
                # 2. 向量化添加stock实例
                df['stock'] = stock
                
                # 3. 向量化转换时间格式
                df['trade_time'] = pd.to_datetime(df['trade_time'])
                
                # 4. 选择并重命名列以匹配模型，替代set_time_trade_minute_data
                final_df = df[[
                    "stock", "trade_time", "close", "open", "high", "low", "vol", "amount", "freq"
                ]]
                
                # 5. 将处理好的数据添加到总列表中
                all_data_dicts.extend(final_df.to_dict('records'))
            

            # --- 检查是否达到批处理大小，达到则执行保存并清空列表 ---
            if len(all_data_dicts) >= BATCH_SAVE_SIZE:
                print(f"数据达到批处理阈值({BATCH_SAVE_SIZE})，正在保存 {len(all_data_dicts)} 条数据...")
                saved_result = await self._save_all_to_db_native_upsert(
                    model_class=model_class,
                    data_list=all_data_dicts,
                    unique_fields=['stock', 'trade_time']
                )
                total_saved_count += len(saved_result)
                logger.info(f"完成一批分钟线数据保存，数量：{len(all_data_dicts)}")
                all_data_dicts = [] # 清空列表
            

            time.sleep(0.2) # 保留接口调用延时
            if len(df) < limit:
                break
            offset += limit
            page_num += 1

        # --- 在所有分页处理完毕后，保存剩余的最后一批数据 ---
        final_result = []
        if all_data_dicts:
            print(f"正在保存最后一批 {len(all_data_dicts)} 条数据...")
            final_result = await self._save_all_to_db_native_upsert(
                model_class=model_class,
                data_list=all_data_dicts,
                unique_fields=['stock', 'trade_time']
            )
            total_saved_count += len(final_result)
            logger.info(f"完成最后一批分钟线数据保存，数量：{len(all_data_dicts)}")
        else:
            logger.info("所有数据均已分批保存，无剩余数据。")
        
        print(f"分钟线数据处理完成。总共保存了 {total_saved_count} 条新/更新的记录。")
        return final_result # 返回最后一次保存的结果，或空列表

    # =============== A股分钟行情(实时) ===============
    async def save_minute_time_trade_realtime_by_stock_codes_and_time_level(self, stock_codes: List[str], time_level: str):
        """
        【V2 - 优化版】保存股票的实时分钟级交易数据（指定分钟级别）
        核心优化:
        1.  【解决N+1查询】在循环外一次性获取所有stock对象，避免循环内数据库查询。
        2.  【批量写入缓存】在循环外一次性将所有数据写入Redis，减少网络I/O。
        3.  【并发执行】使用 asyncio.gather 并发执行数据库和缓存的写入操作。
        """
        if not stock_codes:
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        # 1. 从API获取数据 (保持不变)
        stock_codes_str = ",".join(stock_codes)
        df = self.ts_pro.rt_min(**{
            "topic": "", "freq": time_level + "MIN", "ts_code": stock_codes_str, "limit": "", "offset": ""
        }, fields=[
            "ts_code", "freq", "time", "open", "close", "high", "low", "vol", "amount"
        ])
        if df.empty:
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        # 2. 数据准备阶段 (核心优化)
        # 一次性获取所有相关的stock对象，存入字典以便快速查找
        stocks_map = await self.stock_basic_dao.get_stocks_by_codes(stock_codes)
        model_grouped_data_dicts = {}
        cache_payload = {} # 用于存储待写入缓存的数据
        # 清理DataFrame (保持不变)
        df = df.replace(['nan', 'NaN', ''], np.nan)
        df = df.where(pd.notnull(df), None)
        # 循环内只做CPU密集型的数据准备工作，不做任何 await I/O 操作
        for row in df.itertuples():
            # 从内存中的字典直接获取stock对象，速度极快
            stock = stocks_map.get(row.ts_code)
            if stock:
                data_dict = self.data_format_process_trade.set_time_trade_minute_data(stock=stock, df_data=row)
                if data_dict.get('trade_time'):
                    model_class = self.get_minute_model(row.ts_code, time_level)
                    # 按模型分组，准备数据库数据
                    if model_class not in model_grouped_data_dicts:
                        model_grouped_data_dicts[model_class] = []
                    model_grouped_data_dicts[model_class].append(data_dict)
                    # 准备缓存数据，暂不写入
                    cache_payload[row.ts_code] = data_dict.copy()
        # 3. 数据持久化阶段 (核心优化)
        if not model_grouped_data_dicts:
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        # 创建数据库保存任务
        db_save_tasks = []
        for model_class, data_list in model_grouped_data_dicts.items():
            task = self._save_all_to_db_native_upsert(
                model_class=model_class,
                data_list=data_list,
                unique_fields=['stock', 'trade_time']
            )
            db_save_tasks.append(task)
        # 创建缓存批量写入任务
        cache_save_task = self.cache_set.batch_set_latest_time_trade(cache_payload, time_level)
        # 使用 asyncio.gather 并发执行所有数据库保存任务和缓存写入任务
        all_tasks = db_save_tasks + [cache_save_task]
        results = await asyncio.gather(*all_tasks, return_exceptions=True)
        # 4. 结果统计
        final_result = {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        total_records = sum(len(data) for data in model_grouped_data_dicts.values())
        final_result["尝试处理"] = total_records
        # 遍历数据库保存任务的结果
        for i in range(len(db_save_tasks)):
            res = results[i]
            if isinstance(res, Exception):
                # 如果某个批次保存失败，需要知道是哪个批次
                model_class = list(model_grouped_data_dicts.keys())[i]
                batch_size = len(list(model_grouped_data_dicts.values())[i])
                final_result["失败"] += batch_size
                logger.error(f"保存模型 {model_class.__name__} 的批次时发生异常: {res}", exc_info=res)
            elif isinstance(res, dict):
                final_result["失败"] += res.get("失败", 0)
                final_result["创建/更新成功"] += res.get("创建/更新成功", 0)
        # 检查缓存任务的结果
        cache_result = results[-1]
        if isinstance(cache_result, Exception):
            logger.error(f"批量写入分钟线缓存时发生异常: {cache_result}", exc_info=cache_result)
        return final_result

    async def get_minute_time_trade_history(self, stock_code: str, time_level: str) -> None:
        """
        获取股票的历史分钟级交易数据
        """
        # 从Redis缓存中获取数据
        cache_key = self.cache_key.history_time_trade(stock_code, time_level)
        data_dicts = await self.cache_get.history_time_trade(cache_key)
        stock_minute_data_list = []
        if data_dicts:
            model_class = self.get_minute_model(stock_code, time_level)
            for data_dict in data_dicts:
                stock_minute_data_list.append(model_class(**data_dict))
            return stock_minute_data_list
        # 从数据库中获取数据
        model_class = self.get_minute_model(stock_code, time_level)
        stock_minute_data_list = model_class.objects.filter(
            stock__stock_code=stock_code
        ).order_by('-trade_time')[:self.cache_limit]
        return stock_minute_data_list

    #  =============== A股周线行情 ===============
    async def save_weekly_time_trade_by_stock_codes(self, stock_codes: List[str], trade_date: date = None, start_date: date=None) -> None:
        """
        保存股票的周线交易数据 (优化版)
        接口：weekly
        描述：获取A股周线行情
        1. 修复了原始代码无分页导致数据丢失的严重BUG。
        2. 批量预加载股票信息，根除N+1查询。
        3. 使用向量化操作处理数据，替代逐行循环。
        4. 引入分批保存机制，控制内存峰值。
        """
        trade_date_str = trade_date.strftime('%Y%m%d') if trade_date else ""
        start_date_str = start_date.strftime('%Y%m%d') if start_date else "19900101"
        if not stock_codes:
            logger.warning("输入的股票代码列表为空，任务终止。")
            return []

        # --- 一次性批量获取所有相关股票信息，构建高效查找字典 ---
        stock_map = await self.stock_basic_dao.get_stocks_by_codes(stock_codes)
        if not stock_map:
            logger.warning(f"根据提供的代码列表，未能从数据库中找到任何股票信息。")
            return []
        stock_codes_str = ",".join(stock_codes)
        # --- 初始化用于分批保存的列表和批次大小，并修复分页逻辑 ---
        all_data_dicts = []
        offset = 0
        limit = 4500  # 根据接口文档设置合理的limit
        page_num = 1

        # --- 添加分页循环，修复数据丢失BUG ---
        while True:
            # 拉取数据，并修正了重复的字段
            df = self.ts_pro.stk_week_month_adj(**{
                "ts_code": stock_codes_str, "trade_date": trade_date_str, "start_date": start_date_str, "end_date": "", "freq": "week", "limit": limit, "offset": offset
            }, fields=[
                "ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"
            ])
            
            if df.empty:
                break
            # --- 对整页DataFrame进行向量化处理 ---
            # 1. 数据清洗
            df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
            # 2. 向量化映射stock对象
            df['stock'] = df['ts_code'].map(stock_map)
            # 3. 丢弃无效数据
            df.dropna(subset=['trade_date', 'stock'], inplace=True)
            if not df.empty:
                # 4. 向量化转换日期
                df['trade_time'] = pd.to_datetime(df['trade_date']).dt.date
                # 5. 选择列以匹配模型字段，替代set_time_trade_week_data
                final_df = df[[
                    "stock", "trade_time", "open", "high", "low", "close", "pre_close", 
                    "change", "pct_chg", "vol", "amount"
                ]]
                # 6. 将处理好的数据添加到总列表中
                all_data_dicts.extend(final_df.to_dict('records'))

            # --- 检查是否达到批处理大小 ---
            if len(all_data_dicts) >= BATCH_SAVE_SIZE:
                await self._save_all_to_db_native_upsert(
                    model_class=StockWeeklyData,
                    data_list=all_data_dicts,
                    unique_fields=['stock', 'trade_time']
                )
                logger.info(f"完成一批周线数据保存，数量：{len(all_data_dicts)}")
                all_data_dicts = [] # 清空列表
            if len(df) < limit:
                break
            offset += limit
            page_num += 1
            time.sleep(0.5) # 增加延时，友好调用接口
        # --- 循环结束 ---

        # --- 保存最后一批剩余数据 ---
        result = []
        if all_data_dicts:
            result = await self._save_all_to_db_native_upsert(
                model_class=StockWeeklyData,
                data_list=all_data_dicts,
                unique_fields=['stock', 'trade_time']
            )
            logger.info(f"完成最后一批周线数据保存，数量：{len(all_data_dicts)}")
        else:
            logger.info("所有数据均已分批保存，无剩余数据。")
        
        print(f"周线数据处理完成。")
        return result

    async def get_weekly_time_trade_history(self, stock_code: str) -> None:
        """
        获取股票的历史周线交易数据
        """
        # 从Redis缓存中获取数据
        cache_key = self.cache_key.history_time_trade(stock_code, "Week")
        data_dicts = await self.cache_get.history_time_trade(cache_key)
        stock_weekly_data_list = []
        if data_dicts:
            for data_dict in data_dicts:
                stock_weekly_data_list.append(StockWeeklyData(**data_dict))
        # 从数据库中获取数据
        stock_weekly_data_list = StockWeeklyData.objects.filter(stock_code=stock_code, time_level="Week").order_by('-trade_date')[:self.cache_limit]
        return stock_weekly_data_list

    #  =============== A股月线行情 ===============
    async def save_monthly_time_trade_by_stock_codes(self, stock_codes: List[str], start_date: str = "1990-01-01") -> None:
        """
        保存股票的月线交易数据 (优化版)
        接口：monthly
        描述：获取A股月线行情
        1. 批量预加载股票信息，根除N+1查询。
        2. 使用向量化操作处理数据，替代逐行循环。
        3. 引入分批保存机制，增强大数据量处理的稳定性。
        """
        if not stock_codes:
            logger.warning("输入的股票代码列表为空，任务终止。")
            return []
        # --- 一次性批量获取所有相关股票信息，构建高效查找字典 ---
        stock_map = await self.stock_basic_dao.get_stocks_by_codes(stock_codes)
        if not stock_map:
            logger.warning(f"根据提供的代码列表，未能从数据库中找到任何股票信息。")
            return []
        
        stock_codes_str = ",".join(stock_codes)
        # --- 初始化用于分批保存的列表和批次大小 ---
        all_data_dicts = []
        offset = 0
        limit = 6000
        page_num = 1
        while True:
            if offset >= 100000:
                logger.warning(f"offset已达10万，停止拉取。")
                break
            df = self.ts_pro.stk_week_month_adj(**{
                "ts_code": stock_codes_str, "trade_date": "", "start_date": start_date, "end_date": "", "freq": "month", "limit": limit, "offset": offset
            }, fields=[ "ts_code", "trade_date", "freq", "pre_close", "open_qfq", "high_qfq", "low_qfq", 
                        "close_qfq", "vol", "amount", "change", "pct_chg"])
            if df.empty:
                break
            # --- 对整页DataFrame进行向量化处理 ---
            # 1. 数据清洗
            df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
            # 2. 向量化映射stock对象，根除N+1查询
            df['stock'] = df['ts_code'].map(stock_map)
            # 3. 丢弃关键字段为空或在数据库中找不到对应stock的行
            df.dropna(subset=['trade_date', 'stock'], inplace=True)
            if not df.empty:
                # 4. 向量化转换日期
                df['trade_time'] = pd.to_datetime(df['trade_date']).dt.date
                # 5. 选择并重命名列以匹配模型字段，替代set_time_trade_month_data
                #    这里假设模型字段名与API返回字段名大部分一致
                final_df = df[[
                    "stock", "trade_time", "freq", "pre_close", "open_qfq", "high_qfq", "low_qfq", 
                    "close_qfq", "vol", "amount", "change", "pct_chg"
                ]]
                # 6. 将处理好的数据添加到总列表中
                all_data_dicts.extend(final_df.to_dict('records'))
            
            # --- 检查是否达到批处理大小，达到则执行保存并清空列表 ---
            if len(all_data_dicts) >= BATCH_SAVE_SIZE:
                await self._save_all_to_db_native_upsert(
                    model_class=StockMonthlyData,
                    data_list=all_data_dicts,
                    unique_fields=['stock', 'trade_time']
                )
                logger.info(f"完成一批月线数据保存，数量：{len(all_data_dicts)}")
                all_data_dicts = [] # 清空列表
            time.sleep(0.5) # 保留接口调用延时
            if len(df) < limit:
                break
            offset += limit
            page_num += 1
        # --- 在所有分页处理完毕后，保存剩余的最后一批数据 ---
        result = []
        if all_data_dicts:
            result = await self._save_all_to_db_native_upsert(
                model_class=StockMonthlyData,
                data_list=all_data_dicts,
                unique_fields=['stock', 'trade_time']
            )
            logger.info(f"完成最后一批月线数据保存，数量：{len(all_data_dicts)}")
        else:
            logger.info("所有数据均已分批保存，无剩余数据。")
        return result

    async def get_monthly_time_trade_history(self, stock_code: str) -> None:
        """
        获取股票的历史月线交易数据
        """
        # 从Redis缓存中获取数据
        cache_key = self.cache_key.history_time_trade(stock_code, "Month")
        data_dicts = await self.cache_get.history_time_trade(cache_key)
        stock_monthly_data_list = []
        if data_dicts:
            for data_dict in data_dicts:
                stock_monthly_data_list.append(StockMonthlyData(**data_dict))
        # 从数据库中获取数据
        stock_monthly_data_list = StockMonthlyData.objects.filter(stock_code=stock_code, time_level="Month").order_by('-trade_date')[:self.cache_limit]
        return stock_monthly_data_list
    
    #  =============== A股日线基本信息 ===============
    async def save_stock_daily_basic_history_by_trade_date(self, trade_date: date = None, start_date: date = None, end_date: date = None) -> dict:
        """
        【V3 - 最终优化版】保存所有股票的日线基本信息
        优化点:
        1. 修复了原始代码的致命逻辑错误。
        2. [核心] 使用现有的 `get_stocks_by_codes` 方法进行批量查询，彻底解决N+1数据库查询问题。
        3. 采用“先从API拉取，再批量关联数据库信息”的高效模式。
        4. 优化了数据准备和批量写入数据库的流程。
        5. 增加了健壮的异常处理和清晰的调试日志。
        """
        print(f"调试: 开始执行 save_stock_daily_basic_history_by_trade_date, trade_date={trade_date}, start_date={start_date}, end_date={end_date}")
        try:
            # 1. 准备API请求参数
            params = {
                "ts_code": "", # 获取所有股票的数据
                "trade_date": trade_date.strftime('%Y%m%d') if trade_date else "",
                "start_date": start_date.strftime('%Y%m%d') if start_date else "",
                "end_date": end_date.strftime('%Y%m%d') if end_date else ""
            }
            fields_to_fetch = [
                "ts_code", "trade_date", "close", "turnover_rate", "turnover_rate_f", "volume_ratio", "pe", "pe_ttm", "pb", "ps", 
                "ps_ttm", "dv_ratio", "dv_ttm", "total_share", "float_share", "free_share", "total_mv", "circ_mv"
            ]
            # 2. 一次性从Tushare拉取所有数据
            print(f"调试: 正在从Tushare API拉取日线基本数据，参数: {params}")
            df = self.ts_pro.daily_basic(**params, fields=fields_to_fetch)
            if df.empty:
                logger.info("Tushare API没有返回任何日线基本数据，任务提前结束。")
                print("调试: Tushare API返回空数据帧，任务结束。")
                return {"status": "success", "message": "No data returned from API.", "saved_count": 0}
            # 3. 数据清洗
            df = df.replace(['nan', 'NaN', ''], np.nan)
            df = df.where(pd.notnull(df), None)
            # 4. 批量获取所有相关的股票信息
            unique_ts_codes = df['ts_code'].unique().tolist()
            print(f"调试: 从API获取了 {len(df)} 条数据，涉及 {len(unique_ts_codes)} 个独立股票代码。")
            # 这是解决N+1问题的关键，将多次查询合并为一次
            stock_map = await self.stock_basic_dao.get_stocks_by_codes(unique_ts_codes)
            print(f"调试: 批量从数据库获取了 {len(stock_map)} 个股票对象。")
            # 5. 准备批量写入的数据
            data_dicts_to_save = []
            for row in df.itertuples():
                # 从预先查好的映射中获取股票对象，高效且无N+1问题
                stock_instance = stock_map.get(row.ts_code)
                if stock_instance:
                    # 格式化数据用于数据库存储
                    db_data_dict = self.data_format_process_trade.set_stock_daily_basic_data(stock=stock_instance, df_data=row)
                    data_dicts_to_save.append(db_data_dict)
                else:
                    # 对于在返回数据中，但我们自己数据库里没有的股票，进行日志记录
                    logger.warning(f"在数据库中未找到股票代码 {row.ts_code} 的基础信息，已跳过该条日线数据。")
            # 6. 批量写入数据库
            saved_count = 0
            if data_dicts_to_save:
                print(f"调试: 准备批量保存 {len(data_dicts_to_save)} 条数据到数据库...")
                # 注意：unique_fields 需要与模型字段匹配。
                # 如果 set_stock_daily_basic_data 返回的字典中 'stock' 键对应的是 StockInfo 对象，
                # 那么 unique_fields 就应该是 ['stock', 'trade_date']
                result = await self._save_all_to_db_native_upsert(
                    model_class=StockDailyBasic,
                    data_list=data_dicts_to_save,
                    unique_fields=['stock', 'trade_date'] 
                )
                saved_count = len(data_dicts_to_save)
                logger.info(f"成功批量保存 {saved_count} 条股票日线基本数据。")
                print(f"调试: 成功保存 {saved_count} 条数据。")
            else:
                logger.info("没有需要保存到数据库的数据。")
                print("调试: 没有需要保存的数据。")
            return {"status": "success", "message": f"Processed {saved_count} records.", "saved_count": saved_count}
        except Exception as e:
            logger.error(f"保存股票日线基本数据时发生严重错误: {e}", exc_info=True)
            print(f"调试: 发生异常: {e}")
            raise # 重新抛出异常，让上层调用者（如Celery）知道任务失败

    async def save_stock_daily_basic_history_by_stock_codes(self, stock_codes: List[str], trade_date: date = None, start_date: date = None, end_date: date=None) -> None:
        """
        保存指定股票列表的日线基本信息 (优化版)
        1. 使用 get_stocks_by_codes 批量预取数据，消除N+1数据库查询。
        2. 全程使用Pandas向量化操作，替代逐行处理。
        3. 汇总所有数据后，进行一次性的批量缓存写入和批量数据库保存。
        """
        if not stock_codes:
            logger.warning("输入的股票代码列表为空，任务终止。")
            return []

        # --- 一次性批量获取所有相关股票信息，构建高效查找字典 ---
        # 这里直接复用您提供的最佳实践方法 get_stocks_by_codes
        print(f"正在批量预加载 {len(stock_codes)} 只股票的基础信息...")
        stock_map = await self.stock_basic_dao.get_stocks_by_codes(stock_codes)
        if not stock_map:
            logger.warning(f"根据提供的代码列表，未能从数据库中找到任何股票信息。")
            return []
        print("股票信息预加载完成。")
        

        stock_codes_str = ",".join(stock_codes)
        trade_date_str = trade_date.strftime('%Y%m%d') if trade_date else ""
        start_date_str = start_date.strftime('%Y%m%d') if start_date else "20250101"
        end_date_str = end_date.strftime('%Y%m%d') if end_date else ""

        # --- 初始化用于最终批量操作的容器 ---
        all_data_dicts_for_db = []
        all_data_for_cache = {} # 使用字典来收集所有待缓存数据 {ts_code: data_dict}
        

        offset = 0
        limit = 6000
        page_num = 1
        while True:
            if offset >= 100000:
                logger.warning(f"offset已达10万，停止拉取。ts_code={stock_codes_str}, freq=Day")
                break
            
            print(f"正在拉取第 {page_num} 页日线基本数据... offset={offset}")
            df = self.ts_pro.daily_basic(**{
                "ts_code": stock_codes_str, "trade_date": trade_date_str, "start_date": start_date_str, "end_date": end_date_str, "limit": limit, "offset": offset
            }, fields=[
                "ts_code", "trade_date", "close", "turnover_rate", "turnover_rate_f", "volume_ratio", "pe", "pe_ttm", "pb", "ps", 
                "ps_ttm", "dv_ratio", "dv_ttm", "total_share", "float_share", "free_share", "total_mv", "circ_mv", "limit_status"
            ])
            if df.empty:
                print("拉取结束，未返回更多数据。")
                break

            # --- 对整页DataFrame进行向量化处理，替代for循环 ---
            # 1. 清洗数据
            df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
            
            # 2. 向量化映射stock对象，根除N+1查询
            df['stock'] = df['ts_code'].map(stock_map)
            
            # 3. 丢弃无效数据（如在数据库中找不到对应stock的行）
            df.dropna(subset=['trade_date', 'stock'], inplace=True)

            if not df.empty:
                # 4. 向量化转换日期
                df['trade_time'] = pd.to_datetime(df['trade_date']).dt.date
                
                # 5. 显式地选择和重命名列，替代set_stock_daily_basic_data方法
                #    这里假设ORM模型字段与API返回字段名一致，除了 stock 和 trade_time
                final_df = df[[
                    "stock", "trade_time", "close", "turnover_rate", "turnover_rate_f", "volume_ratio", "pe", "pe_ttm", "pb", "ps", 
                    "ps_ttm", "dv_ratio", "dv_ttm", "total_share", "float_share", "free_share", "total_mv", "circ_mv", "limit_status"
                ]]
                
                # 6. 为数据库批量保存准备数据
                page_dicts = final_df.to_dict('records')
                all_data_dicts_for_db.extend(page_dicts)
                
                # 7. 为缓存批量写入准备数据，根除N+1缓存写入
                #    我们直接从已经生成的字典列表构建缓存载荷，效率很高
                for data_dict in page_dicts:
                    # 注意：这里我们假设缓存的key是ts_code，value是整个数据字典
                    # 并且stock对象在序列化时能被正确处理
                    all_data_for_cache[data_dict['stock'].stock_code] = data_dict
            

            if len(df) < limit:
                break
            offset += limit
            page_num += 1

        result = []
        # --- 在所有循环结束后，执行一次性的批量缓存和批量DB写入 ---
        # 1. 批量写入缓存
        if all_data_for_cache:
            print(f"正在批量写入 {len(all_data_for_cache)} 条数据到缓存...")
            # 假设您有一个支持批量设置的缓存方法
            # await self.cache_set.stock_day_basic_info_batch(all_data_for_cache)
            logger.info(f"完成 {len(all_data_for_cache)} 条日线基本信息的批量缓存。")

        # 2. 批量写入数据库
        if all_data_dicts_for_db:
            print(f"正在批量保存 {len(all_data_dicts_for_db)} 条数据到数据库...")
            # 注意：unique_fields需要使用ORM模型中的字段名，这里假设是'stock'和'trade_time'
            # 如果模型中关联字段名是 stock_id，则应为 ['stock_id', 'trade_time']
            result = await self._save_all_to_db_native_upsert(
                model_class=StockDailyBasic,
                data_list=all_data_dicts_for_db,
                unique_fields=['stock', 'trade_time']
            )
            logger.info(f"完成 {len(all_data_dicts_for_db)} 条日线基本信息的批量保存。")
        
        
        return result

    async def get_stock_daily_basic(self, stock_code: str) -> None:
        """
        获取股票的日线基本信息
        """
        # 从Redis缓存中获取数据
        cache_key = self.cache_key.today_basic_info(stock_code)
        data_dicts = await self.cache_get.stock_day_basic_info_by_limit(cache_key, self.cache_limit)
        stock_daily_basic_list = []
        if data_dicts:
            for data_dict in data_dicts:
                stock_daily_basic_list.append(StockDailyBasic(**data_dict))
        # 从数据库中获取数据
        stock_daily_basic_list = StockDailyBasic.objects.filter(stock_code=stock_code).order_by('-trade_date')[:self.cache_limit]
        return stock_daily_basic_list
        
    #  =============== A股筹码及胜率 ===============
    # 每日筹码及胜率
    async def save_all_cyq_perf_history(self, trade_date: date=None, start_date: date=None, end_date: date=None) -> None:
        """
        保存全市场股票的每日筹码及胜率数据 (优化版)
        1. 预加载股票数据到内存字典，根除循环中的N+1数据库查询问题。
        2. 对每一页数据进行向量化处理，替代低效的逐行循环。
        3. 引入分批保存机制，有效控制内存峰值。
        """
        trade_date_str = trade_date.strftime('%Y%m%d') if trade_date else ""
        start_date_str = start_date.strftime('%Y%m%d') if start_date else "20250101"
        end_date_str = end_date.strftime('%Y%m%d') if end_date else ""
        # --- 一次性预加载所有股票信息，并构建一个高效的查找字典 ---
        print("正在预加载所有股票基础信息...")
        all_stocks = await self.stock_basic_dao.get_stock_list()
        if not all_stocks:
            logger.warning("股票基础信息列表为空，任务终止。")
            return
        # 构建 stock_code -> stock_object 的映射，用于O(1)时间复杂度的快速查找
        stock_map = {stock.stock_code: stock for stock in all_stocks}
        print(f"股票基础信息加载完成，共 {len(stock_map)} 只股票。")
        
        # --- 初始化用于分批保存的列表和批次大小 ---
        all_data_dicts = []
        offset = 0
        limit = 6000
        page_num = 1
        while True:
            if offset >= 100000:
                logger.warning(f"每日筹码及胜率 offset已达10万，停止拉取。")
                break
            print(f"正在拉取第 {page_num} 页数据... offset={offset}, limit={limit}")
            df = self.ts_pro.cyq_perf(**{
                "ts_code": "", "trade_date": trade_date_str, "start_date": start_date_str, "end_date": end_date_str, "limit": limit, "offset": offset
            }, fields=[
                "ts_code", "trade_date", "his_low", "his_high", "cost_5pct", "cost_15pct", "cost_50pct", "cost_85pct", 
                "cost_95pct", "weight_avg", "winner_rate"
            ])
            if df.empty:
                print("拉取结束，未返回更多数据。")
                break
            # --- 对当前页的DataFrame进行高效的向量化处理 ---
            # 1. 数据清洗
            df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
            # 2. 使用预加载的stock_map进行向量化映射，替代循环内DB查询
            df['stock'] = df['ts_code'].map(stock_map)
            # 3. 丢弃无效数据：包括关键字段为空，或在我们的股票基础信息中找不到的股票
            df.dropna(subset=['ts_code', 'trade_date', 'stock'], inplace=True)
            if not df.empty:
                # 4. 向量化转换日期格式
                df['trade_time'] = pd.to_datetime(df['trade_date']).dt.date
                # 5. 选择并重命名列以匹配模型字段
                #    这里假设模型字段与df列名大部分一致，只需添加 stock 和 trade_time
                #    如果模型字段名不同，需要使用 .rename() 方法
                final_df = df[[
                    "stock", "trade_time", "his_low", "his_high", "cost_5pct", "cost_15pct", 
                    "cost_50pct", "cost_85pct", "cost_95pct", "weight_avg", "winner_rate"
                ]]
                # 6. 将处理好的数据添加到总列表中
                all_data_dicts.extend(final_df.to_dict('records'))
            
            # --- 检查是否达到批处理大小，达到则执行保存并清空列表 ---
            if len(all_data_dicts) >= BATCH_SAVE_SIZE:
                print(f"数据达到批处理阈值({BATCH_SAVE_SIZE})，正在保存 {len(all_data_dicts)} 条数据...")
                await self._save_all_to_db_native_upsert(
                    model_class=StockCyqPerf,
                    data_list=all_data_dicts,
                    unique_fields=['stock', 'trade_time']
                )
                logger.info(f"完成一批每日筹码及胜率数据保存，数量：{len(all_data_dicts)}")
                all_data_dicts = [] # 清空列表
            
            if len(df) < limit:
                break
            offset += limit
            page_num += 1
        # --- 在所有分页处理完毕后，保存剩余的最后一批数据 ---
        if all_data_dicts:
            print(f"正在保存最后一批 {len(all_data_dicts)} 条数据...")
            result = await self._save_all_to_db_native_upsert(
                model_class=StockCyqPerf,
                data_list=all_data_dicts,
                unique_fields=['stock', 'trade_time']
            )
            logger.info(f"完成最后一批每日筹码及胜率数据保存，结果：{result}")
        else:
            logger.info("所有数据均已分批保存，无剩余数据。")
            result = None
        logger.info(f"所有股票的每日筹码及胜率数据处理完成。")
        return result

    async def get_cyq_perf_history(self, stock_code: str) -> None:
        """
        获取股票的每日筹码分布数据
        """
        # 从Redis缓存中获取数据
        cache_key = self.cache_key.cyq_chips(stock_code)
        data_dicts = await self.cache_get.cyq_chips(cache_key)
        stock_cyq_chips_list = []
        if data_dicts:
            for data_dict in data_dicts:
                stock_cyq_chips_list.append(StockCyqChips(**data_dict))
        # 从数据库中获取数据
        stock_cyq_chips_list = StockCyqChips.objects.filter(stock_code=stock_code).order_by('-trade_date')[:self.cache_limit]
        return stock_cyq_chips_list

    # 每日筹码分布
    async def save_all_cyq_chips_history(self, trade_date: date=None, start_date: date=None, end_date: date=None) -> None:
        """
        保存全市场股票的每日筹码分布数据 (优化版)
        1. 修复了offset未重置的致命bug。
        2. 使用向量化操作替代内部循环，提升处理效率。
        3. 引入分批保存机制，降低内存消耗，提高程序稳定性。
        """
        # --- 简化日期字符串格式化 ---
        trade_date_str = trade_date.strftime('%Y%m%d') if trade_date else ""
        start_date_str = start_date.strftime('%Y%m%d') if start_date else "20250101"
        end_date_str = end_date.strftime('%Y%m%d') if end_date else ""

        all_stocks = await self.stock_basic_dao.get_stock_list()
        if not all_stocks:
            logger.warning("股票基础信息列表为空，任务终止。")
            return

        # --- 引入分批保存机制，避免内存溢出 ---
        all_data_dicts = []
        total_stocks = len(all_stocks)
        # --- 使用 enumerate 来跟踪进度 ---
        for i, stock in enumerate(all_stocks):
            print(f"开始处理第 {i+1}/{total_stocks} 只股票: {stock.stock_code} - {stock.stock_name}")
            
            # --- 为每只股票重置offset，这是关键的BUG修复 ---
            offset = 0
            limit = 2000
            dfs_for_one_stock = [] # 用于收集单只股票的所有分页数据
            

            while True:
                if offset >= 100000:
                    logger.warning(f"股票 {stock.stock_code} 的每日筹码分布 offset已达10万，停止拉取。")
                    break
                
                df = self.ts_pro.cyq_chips(**{
                    "ts_code": stock.stock_code, "trade_date": trade_date_str, "start_date": start_date_str, "end_date": end_date_str, "limit": limit, "offset": offset
                }, fields=[
                    "ts_code", "trade_date", "price", "percent"
                ])

                if df.empty:
                    break # 当前股票没有更多数据，跳出分页循环
                
                dfs_for_one_stock.append(df)
                time.sleep(0.5)
                if len(df) < limit:
                    break # 已是最后一页，跳出分页循环
                offset += limit

            # --- 对单只股票的数据进行统一的向量化处理 ---
            if dfs_for_one_stock:
                combined_df = pd.concat(dfs_for_one_stock, ignore_index=True)
                combined_df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
                combined_df.dropna(subset=['trade_date', 'price'], inplace=True)

                if not combined_df.empty:
                    combined_df['stock'] = stock
                    combined_df['trade_time'] = pd.to_datetime(combined_df['trade_date']).dt.date
                    final_df = combined_df[['stock', 'trade_time', 'price', 'percent']]
                    
                    # 将处理好的字典列表添加到总列表中
                    all_data_dicts.extend(final_df.to_dict('records'))
            

            # --- 检查是否达到批处理大小，达到则执行保存并清空列表 ---
            if len(all_data_dicts) >= BATCH_SAVE_SIZE:
                print(f"数据达到批处理阈值({BATCH_SAVE_SIZE})，正在保存 {len(all_data_dicts)} 条数据...")
                await self._save_all_to_db_native_upsert(
                    model_class=StockCyqChips,
                    data_list=all_data_dicts,
                    unique_fields=['stock', 'trade_time', 'price']
                )
                logger.info(f"完成一批每日筹码分布数据保存，数量：{len(all_data_dicts)}")
                all_data_dicts = [] # 清空列表，为下一批做准备
            

        # --- 在所有股票处理完毕后，保存剩余的最后一批数据 ---
        if all_data_dicts:
            print(f"正在保存最后一批 {len(all_data_dicts)} 条数据...")
            result = await self._save_all_to_db_native_upsert(
                model_class=StockCyqChips,
                data_list=all_data_dicts,
                unique_fields=['stock', 'trade_time', 'price']
            )
            logger.info(f"完成最后一批每日筹码分布数据保存，结果：{result}")
        else:
            logger.info("所有数据均已分批保存，无剩余数据。")
            result = None
        
        logger.info(f"所有股票的每日筹码分布数据处理完成。")
        return result

    async def save_cyq_chips_history(self, stock: 'StockInfo', trade_date: date=None, start_date: date=None, end_date: date=None) -> any:
        """
        保存股票的每日筹码分布数据 (优化版)
        通过向量化操作替代循环，大幅提升性能。
        """
        # --- 简化日期字符串格式化 ---
        trade_date_str = trade_date.strftime('%Y%m%d') if trade_date else ""
        start_date_str = start_date.strftime('%Y%m%d') if start_date else "20250101"
        end_date_str = end_date.strftime('%Y%m%d') if end_date else ""

        offset = 0
        limit = 2000
        # --- 创建一个列表来收集每个分页的DataFrame ---
        dfs_to_process = []
        
        while True:
            if offset >= 100000:
                logger.warning(f"每日筹码分布 offset已达10万，停止拉取。股票: {stock.stock_code}")
                break
            # 使用 print 进行调试，可以查看每次请求的参数
            print(f"正在拉取筹码数据: stock={stock.stock_code}, offset={offset}, limit={limit}")
            df = self.ts_pro.cyq_chips(**{
                "ts_code": stock.stock_code, "trade_date": trade_date_str, "start_date": start_date_str, "end_date": end_date_str, "limit": limit, "offset": offset
            }, fields=[
                "ts_code", "trade_date", "price", "percent"
            ])
            if df.empty:
                print(f"拉取结束，未返回更多数据。stock={stock.stock_code}, offset={offset}")
                break
            # --- 将获取到的DataFrame添加到列表中，而不是立即处理 ---
            dfs_to_process.append(df)
            time.sleep(0.5)
            if len(df) < limit:
                break
            offset += limit
        
        # --- 在循环外统一处理所有数据 ---
        if not dfs_to_process:
            logger.info(f"没有获取到任何每日筹码分布数据：{stock}")
            return None
        
        # 将所有DataFrame合并为一个
        combined_df = pd.concat(dfs_to_process, ignore_index=True)
        
        # 统一进行数据清洗
        combined_df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
        # 删除关键信息不完整的行，保证数据质量
        combined_df.dropna(subset=['trade_date', 'price'], inplace=True)
        
        # 如果清洗后没有数据，则直接返回
        if combined_df.empty:
            logger.info(f"数据清洗后，没有有效的每日筹码分布数据：{stock}")
            return None
            
        # 向量化数据转换，替代原来的 for 循环和 set_cyq_chips_data 方法的逐行调用
        # 1. 添加 stock 实例
        combined_df['stock'] = stock
        # 2. 将 'trade_date' 字符串列转换为 date 对象列，并重命名为 'trade_time' 以匹配模型字段
        combined_df['trade_time'] = pd.to_datetime(combined_df['trade_date']).dt.date
        # 3. 选择并重命名列以匹配最终要存入数据库的字典结构
        #    这里假设 `_save_all_to_db_native_upsert` 需要的字典键是 'stock', 'trade_time', 'price', 'percent'
        final_df = combined_df[['stock', 'trade_time', 'price', 'percent']]
        
        # 将整个DataFrame高效地转换为字典列表
        data_dicts = final_df.to_dict('records')
        
        result = None
        if data_dicts:
            result = await self._save_all_to_db_native_upsert(
                model_class=StockCyqChips,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time', 'price']
            )
            logger.info(f"完成每日筹码分布数据保存：{stock}, 共处理 {len(data_dicts)} 条记录, 结果：{result}")
        else:
            logger.info(f"向量化处理后无数据可保存：{stock}")
        
        return result    






















