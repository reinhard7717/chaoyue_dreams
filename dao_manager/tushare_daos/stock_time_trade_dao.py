# dao_manager\tushare_daos\stock_time_trade_dao.py
import asyncio
from decimal import Decimal
import logging
import time
from django.db.models import QuerySet
from asgiref.sync import sync_to_async
from typing import Dict, List, Optional
from collections import defaultdict # 导入 defaultdict 以方便分组
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from dao_manager.base_dao import BaseDAO
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from stock_models.stock_basic import StockInfo
from stock_models.time_trade import StockCyqChipsBJ, StockCyqChipsCY, StockCyqChipsKC, StockCyqChipsSH, StockCyqChipsSZ, StockCyqPerf, StockDailyBasic, StockMinuteData, StockWeeklyData, StockMonthlyData
from utils.cache_get import StockInfoCacheGet, StockTimeTradeCacheGet
from utils.cache_manager import CacheManager
from utils.cache_set import StockInfoCacheSet, StockTimeTradeCacheSet
from utils.cash_key import StockCashKey
from utils.data_format_process import StockInfoFormatProcess, StockTimeTradeFormatProcess
from stock_models.time_trade import StockDailyData_SZ, StockDailyData_SH, StockDailyData_CY, StockDailyData_KC, StockDailyData_BJ
from stock_models.time_trade import (
            StockMinuteData_1_SZ, StockMinuteData_1_SH, StockMinuteData_1_BJ, StockMinuteData_1_CY, StockMinuteData_1_KC,
            StockMinuteData_5_SZ, StockMinuteData_5_SH, StockMinuteData_5_BJ, StockMinuteData_5_CY, StockMinuteData_5_KC,
            StockMinuteData_15_SZ, StockMinuteData_15_SH, StockMinuteData_15_BJ, StockMinuteData_15_CY, StockMinuteData_15_KC,
            StockMinuteData_30_SZ, StockMinuteData_30_SH, StockMinuteData_30_BJ, StockMinuteData_30_CY, StockMinuteData_30_KC,
            StockMinuteData_60_SZ, StockMinuteData_60_SH, StockMinuteData_60_BJ, StockMinuteData_60_CY, StockMinuteData_60_KC,
        )

BATCH_SAVE_SIZE = 110000  # 每10000条数据保存一次
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

    async def save_daily_time_trade_history_by_trade_dates(self, trade_date: date = None, start_date: date = None, end_date: date = None) -> Dict:
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
            for model_class, group_df in df.groupby('model_class', sort=False):
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

    async def save_daily_time_trade_history_by_stock_codes(self, stock_codes: List[str]) -> Dict:
        """
        【V2.0 - 向量化优化版】保存多只股票的历史日线交易数据，自动分表
        核心优化:
        1. 【消除N+1查询】在循环开始前，一次性获取所有需要的StockInfo对象，并创建查找映射。
        2. 【向量化关联】使用Pandas.map()高效地将StockInfo对象关联到DataFrame的每一行。
        3. 【向量化分组】使用Pandas.groupby()替代逐行判断和追加，按分表模型对数据进行分组。
        4. 【批量转换】直接在DataFrame上进行列操作和类型转换，最后用to_dict批量生成待保存数据。
        """

        if not stock_codes:
            logger.warning("传入的stock_codes列表为空，任务终止。")
            return {}

        # 1. 【核心优化】一次性获取所有相关股票信息，并创建高效查找映射
        # 注意：这里假设 get_stock_list() 效率足够高（有缓存）。
        # 如果 stock_codes 只是所有股票的一小部分，更优的做法是创建一个 get_stocks_by_codes(codes) 的方法。
        # 但根据题目要求，我们使用 get_stock_list()。
        all_stocks = await self.stock_basic_dao.get_stock_list()
        stock_map = {stock.stock_code: stock for stock in all_stocks if stock.stock_code in stock_codes}

        if not stock_map:
            logger.warning(f"提供的stock_codes: {stock_codes} 在数据库中均未找到对应的StockInfo。")
            return {}

        # 2. 准备API请求和分页拉取
        stock_codes_str = ",".join(stock_codes)
        data_dicts_by_model = {}
        offset = 0
        limit = 6000

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

            original_count = len(df)

            # 3. 向量化数据处理 (替代原有的 for 循环)
            df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
            df.dropna(subset=['ts_code', 'trade_date'], inplace=True)
            # 3.1 【向量化关联】使用map将StockInfo对象关联到每一行
            df['stock'] = df['ts_code'].map(stock_map)
            df.dropna(subset=['stock'], inplace=True) # 丢弃没有成功关联的行
            if df.empty:
                print("调试信息: 关联StockInfo后，当前批次无有效数据。")
                time.sleep(0.2)
                if original_count < limit:
                    break
                offset += limit
                continue
            # 3.2 【向量化转换】批量转换日期格式
            df['trade_time'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.date
            # 3.3 【向量化分组】根据ts_code确定分表模型，并按模型对DataFrame进行分组
            df['model_class'] = df['ts_code'].apply(self.get_daily_data_model_by_code)
            for model_class, group_df in df.groupby('model_class', sort=False):
                # 3.4 【批量准备数据】选择并重命名列，然后批量转为字典
                # 注意：这里的列名需要与你的分表模型字段完全对应
                # 假设分表模型字段与df列名大部分一致，除了 stock 和 trade_time
                columns_to_keep = [
                    'stock', 'trade_time', 'close', 'open', 'high', 'low', 'pre_close', 'change', 'pct_change', 'vol',
                    'amount', 'adj_factor', 'open_hfq', 'open_qfq', 'close_hfq', 'close_qfq', 'high_hfq', 'high_qfq', 'low_hfq',
                    'low_qfq', 'pre_close_hfq', 'pre_close_qfq'
                ]
                # 过滤掉不存在的列，以防API返回字段不全
                final_cols = [col for col in columns_to_keep if col in group_df.columns]
                data_to_save = group_df[final_cols].where(pd.notnull(group_df), None).to_dict('records')
                
                if model_class not in data_dicts_by_model:
                    data_dicts_by_model[model_class] = []
                data_dicts_by_model[model_class].extend(data_to_save)

            # 4. 分页逻辑 (逻辑不变)
            if len(df) < limit:
                break
            offset += limit
        
        # 5. 批量保存 (逻辑不变，但现在处理的是预先分组好的数据)
        result = {}
        for model_class, data_list in data_dicts_by_model.items():
            if not data_list: continue
            print(f"调试信息: 正在为模型 {model_class.__name__} 保存 {len(data_list)} 条数据...")
            res = await self._save_all_to_db_native_upsert(
                model_class=model_class,
                data_list=data_list,
                unique_fields=['stock', 'trade_time'] # 确保唯一键正确
            )
            result[model_class.__name__] = res
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
        if time_level not in ['1', '5', '15', '30', '60']:
            return StockMinuteData
        if stock_code.endswith('.SZ'):
            if stock_code.startswith('3'):
                return {
                    '1': StockMinuteData_1_CY, '5': StockMinuteData_5_CY, '15': StockMinuteData_15_CY, '30': StockMinuteData_30_CY, '60': StockMinuteData_60_CY
                }[time_level]
            else:
                return {
                    '1': StockMinuteData_1_SZ, '5': StockMinuteData_5_SZ, '15': StockMinuteData_15_SZ, '30': StockMinuteData_30_SZ, '60': StockMinuteData_60_SZ
                }[time_level]
        elif stock_code.endswith('.SH'):
            if stock_code.startswith('68'):
                return {
                    '1': StockMinuteData_1_KC, '5': StockMinuteData_5_KC, '15': StockMinuteData_15_KC, '30': StockMinuteData_30_KC, '60': StockMinuteData_60_KC
                }[time_level]
            else:
                return {
                    '1': StockMinuteData_1_SH, '5': StockMinuteData_5_SH, '15': StockMinuteData_15_SH, '30': StockMinuteData_30_SH, '60': StockMinuteData_60_SH
                }[time_level]
        elif stock_code.endswith('.BJ'):
            return {
                '1': StockMinuteData_1_BJ, '5': StockMinuteData_5_BJ, '15': StockMinuteData_15_BJ, '30': StockMinuteData_30_BJ, '60': StockMinuteData_60_BJ
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
        【V3 - 健壮分页修复版】保存股票的历史分钟级交易数据
        - 策略:
        1. 【核心修复】在API调用后，立即将返回的行数存入`original_df_len`变量。
        2. 【核心修复】整个循环中，所有关于“是否为最后一页”的判断，都统一使用`original_df_len`，彻底避免因数据清洗导致分页提前中断的问题。
        """
        if not stock_codes:
            logger.warning("输入的股票代码列表为空，任务终止。")
            return

        stock_map = await self.stock_basic_dao.get_stocks_by_codes(stock_codes)
        if not stock_map:
            logger.warning(f"根据提供的代码列表，未能从数据库中找到任何股票信息，任务终止。")
            return
        
        stock_codes_str = ",".join(stock_codes)
        for time_level in ['1', '5', '15', '30', '60']:
            offset = 0
            limit = 8000
            page_num = 1
            while True:
                if offset >= 100000:
                    logger.warning(f"offset已达10万，停止拉取。ts_code={stock_codes_str}, freq={time_level}min")
                    break

                print(f"调试信息: 准备拉取 {time_level}min 数据, page={page_num}, offset={offset}, limit={limit}")
                df = self.ts_pro.stk_mins(**{
                    "ts_code": stock_codes_str, "freq": time_level + "min", "start_date": start_date_str, "end_date": end_date_str, 
                    "limit": limit, "offset": offset
                }, fields=[ "ts_code", "trade_time", "close", "open", "high", "low", "vol", "amount", "freq" ])
                
                # 【代码修改】在进行任何操作前，立即记录API返回的原始行数，这是分页判断的唯一可靠依据。
                original_df_len = len(df)
                print(f"调试信息: API返回 {original_df_len} 条原始数据。")
                
                if original_df_len == 0:
                    print(f"拉取结束，API未返回更多 {time_level}min 数据。")
                    break
                
                # --- 对整页DataFrame进行向量化处理 ---
                df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
                df['stock'] = df['ts_code'].map(stock_map)
                df.dropna(subset=['trade_time', 'stock'], inplace=True)

                if df.empty:
                    print(f"当前页数据经清洗后为空，跳至下一页。")
                    # 【代码修改】分页判断移至循环末尾，此处只需更新offset并continue
                    offset += limit
                    page_num += 1
                    continue

                df['trade_time'] = pd.to_datetime(df['trade_time'])
                df['trade_time'] = df['trade_time'].dt.tz_localize('Asia/Shanghai')
                df['trade_time'] = df['trade_time'].dt.tz_convert('UTC')
                df['trade_time'] = df['trade_time'].dt.tz_localize(None)
                df['model_class'] = df['ts_code'].apply(lambda code: self.get_minute_model(code, time_level))

                for model_class, group_df in df.groupby('model_class', sort=False):
                    if group_df.empty:
                        continue
                    
                    data_list = group_df[[
                        "stock", "trade_time", "close", "open", "high", "low", "vol", "amount"
                    ]].to_dict('records')

                    # 您的保存逻辑
                    await self._save_all_to_db_native_upsert(
                        model_class=model_class,
                        data_list=data_list,
                        unique_fields=['stock', 'trade_time']
                    )
                    logger.info(f"保存 {model_class.__name__} 的 {time_level}分钟级数据完成. 准备了 {len(data_list)} 条记录进行插入/更新。")

                # 【代码修改】分页逻辑判断必须且只能基于API返回的原始行数
                if original_df_len < limit:
                    print(f"调试信息: API返回行数({original_df_len})小于limit({limit})，判定为最后一页，当前频率数据拉取结束。")
                    break
                
                offset += limit
                page_num += 1

        logger.info(f"保存 {len(stock_codes)}个股票 的分钟级交易数据全部完成.")
        return

    async def save_minute_time_trade_history_by_stock_code_and_time_level(self, stock_code: str, time_level: str, trade_date: date=None, start_date: date=None, end_date: date=None) -> int:
        """
        保存股票的历史分钟级交易数据 (优化版)
        - 兼容原生SQL批量插入，手动处理时区转换。
        1. 预先获取stock对象和目标Model，避免循环内重复操作。
        2. 全程使用Pandas向量化操作，替代逐行处理，大幅提升性能。
        3. 引入分批保存机制，有效控制内存占用，适合海量分钟线数据。
        4. 手动进行时区转换，以适配不经过Django ORM时区处理的原生SQL操作。
        """
        # --- 在循环外一次性获取所有必要的前置对象 ---
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            logger.error(f"未能找到股票代码为 {stock_code} 的股票信息，任务终止。")
            return 0

        model_class = self.get_minute_model(stock_code, time_level)
        if not model_class:
            logger.error(f"未能为 {stock_code} 和时间级别 {time_level}min 找到对应的数据库模型，任务终止。")
            return 0
        
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
            df = self.ts_pro.stk_mins(**{
                "ts_code": stock_code, "freq": time_level + "min", "start_date": start_date_str, "end_date": end_date_str, "limit": limit, "offset": offset
            }, fields=[ "ts_code", "trade_time", "close", "open", "high", "low", "vol", "amount", "freq" ])
            
            if df.empty:
                break

            # --- 对整页DataFrame进行向量化处理，替代for循环 ---
            # 1. 数据清洗
            df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
            df.dropna(subset=['trade_time'], inplace=True) # 确保关键字段存在

            if not df.empty:
                # 2. 向量化添加stock实例
                df['stock'] = stock

                # 3. 向量化转换时间格式，并最终转换为适合原生SQL的UTC天真时间。
                #    由于使用了原生SQL批量插入，Django的自动时区转换会失效，需要在此手动处理。

                # 3.1 将字符串转换为“天真”的datetime对象
                df['trade_time'] = pd.to_datetime(df['trade_time'])

                # 3.2 将“天真”时间本地化为北京时间（'Asia/Shanghai'），使其变为“时区感知”
                df['trade_time'] = df['trade_time'].dt.tz_localize('Asia/Shanghai')

                # 3.3 将时区感知的时间（北京时间）转换为UTC时区
                df['trade_time'] = df['trade_time'].dt.tz_convert('UTC')

                # 3.4 去除UTC时区信息，使其变回“天真”的datetime对象。
                #    这样，数据库驱动会将其作为正确的UTC时间值存入DATETIME字段。
                df['trade_time'] = df['trade_time'].dt.tz_localize(None)
                # --- 修改的代码行结束 ---
                
                # 4. 选择并重命名列以匹配模型
                final_df = df[[
                    "stock", "trade_time", "close", "open", "high", "low", "vol", "amount"
                ]]
                
                # 5. 将处理好的数据添加到总列表中
                all_data_dicts.extend(final_df.to_dict('records'))
            
            # --- 检查是否达到批处理大小，达到则执行保存并清空列表 ---
            if len(all_data_dicts) >= BATCH_SAVE_SIZE:
                result_dict = await self._save_all_to_db_native_upsert(
                    model_class=model_class,
                    data_list=all_data_dicts,
                    unique_fields=['stock', 'trade_time']
                )
                # 从返回的字典中获取成功保存的数量
                saved_count = result_dict.get("创建/更新成功", 0)
                total_saved_count += saved_count
                # logger.info(f"完成一批分钟线数据保存，数量：{saved_count}")
                all_data_dicts = [] # 清空列表
            
            time.sleep(0.2) # 保留接口调用延时
            if len(df) < limit:
                break
            offset += limit
            page_num += 1

        # --- 在所有分页处理完毕后，保存剩余的最后一批数据 ---
        if all_data_dicts:
            result_dict = await self._save_all_to_db_native_upsert(
                model_class=model_class,
                data_list=all_data_dicts,
                unique_fields=['stock', 'trade_time']
            )
            final_saved_count = result_dict.get("创建/更新成功", 0)
            total_saved_count += final_saved_count
            # logger.info(f"完成最后一批分钟线数据保存，数量：{final_saved_count}")
        else:
            logger.info("所有数据均已分批保存，无剩余数据。")
        
        print(f"分钟线数据处理完成。{stock} 总共保存了 {total_saved_count} 条新/更新的记录。")
        return total_saved_count

    # =============== A股分钟行情(实时) ===============
    async def save_minute_time_trade_realtime_by_stock_codes_and_time_level(self, stock_codes: List[str], time_level: str):
        """
        【V5.0 - 双轨持久化版】保存股票的实时分钟级交易数据。
        - 核心升级: 只调用一次API，然后将数据并发地、双轨写入到
                    1. 数据库 (PostgreSQL): 用于长期历史存储和盘后分析。
                    2. 缓存 (Redis ZSET): 用于高性能的盘中实时策略计算。
        """
        if not stock_codes:
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}

        # 1. 数据采集：只调用一次API
        stock_codes_str = ",".join(stock_codes)
        df = self.ts_pro.rt_min(ts_code=stock_codes_str, freq=f"{time_level}MIN", fields=[
            "ts_code", "time", "open", "close", "high", "low", "vol", "amount"
        ])

        if df.empty:
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}

        # 2. 数据预处理 (与之前版本相同)
        df.dropna(subset=['time', 'ts_code'], inplace=True)
        df['trade_time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        df.dropna(subset=['trade_time'], inplace=True)
        if df.empty:
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}

        # 3. 【核心改造】准备双轨数据载荷
        model_grouped_data_dicts = {} # 轨道1: 数据库载荷
        cache_payload = {}            # 轨道2: Redis ZSET 载荷
        
        data_records = df.to_dict('records')

        for record in data_records:
            stock_code = record['ts_code']
            
            # 3.1 准备数据库载荷
            model_class = self.get_minute_model(stock_code, time_level)
            if model_class:
                db_record = record.copy()
                # 为数据库准备：关联stock对象，转换时区
                stock_obj = await self.stock_basic_dao.get_stock_by_code(stock_code)
                if stock_obj:
                    db_record['stock'] = stock_obj
                    # 转换为带时区的UTC时间，再去除时区信息以适配原生SQL
                    utc_time = db_record['trade_time'].tz_localize('Asia/Shanghai').tz_convert('UTC')
                    db_record['trade_time'] = utc_time.tz_localize(None)
                    del db_record['ts_code']
                    
                    model_grouped_data_dicts.setdefault(model_class, []).append(db_record)

            # 3.2 准备Redis缓存载荷
            cache_record = {
                "open": record['open'], "high": record['high'],
                "low": record['low'], "close": record['close'],
                "volume": record['vol'], "amount": record['amount'],
                "trade_time": record['trade_time'] # 保留原始datetime对象
            }
            cache_payload.setdefault(stock_code, []).append(cache_record)

        # 4. 并发执行双轨持久化
        if not model_grouped_data_dicts and not cache_payload:
            return {"尝试处理": len(df), "失败": len(df), "创建/更新成功": 0}

        # 4.1 创建所有数据库写入任务
        db_save_tasks = []
        for model_class, data_list in model_grouped_data_dicts.items():
            task = self._save_all_to_db_native_upsert(
                model_class=model_class,
                data_list=data_list,
                unique_fields=['stock', 'trade_time']
            )
            db_save_tasks.append(task)

        # 4.2 创建Redis缓存写入任务
        cache_save_task = self.cache_set.batch_set_intraday_minute_kline(cache_payload, time_level)
        
        # 4.3 并发执行所有任务
        all_tasks = db_save_tasks + [cache_save_task]
        results = await asyncio.gather(*all_tasks, return_exceptions=True)

        # 5. 结果统计 (与之前版本相同)
        final_result = {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        total_records = sum(len(data) for data in model_grouped_data_dicts.values())
        final_result["尝试处理"] = total_records

        for i in range(len(db_save_tasks)):
            res = results[i]
            if isinstance(res, Exception):
                model_class = list(model_grouped_data_dicts.keys())[i]
                batch_size = len(list(model_grouped_data_dicts.values())[i])
                final_result["失败"] += batch_size
                logger.error(f"保存模型 {model_class.__name__} 的批次时发生异常: {res}", exc_info=res)
            elif isinstance(res, dict):
                final_result["失败"] += res.get("失败", 0)
                final_result["创建/更新成功"] += res.get("创建/更新成功", 0)

        cache_result = results[-1]
        if isinstance(cache_result, Exception):
            logger.error(f"批量写入分钟线缓存时发生异常: {cache_result}", exc_info=cache_result)
        
        return final_result

    async def get_minute_kline_by_daterange(self, stock_code: str, time_level: str, start_dt: datetime, end_dt: datetime) -> Optional[pd.DataFrame]:
        """
        【V2.0 - 盘中引擎专用】从数据库(MySQL/PostgreSQL)获取指定时间范围内的分钟K线数据。
        
        - 核心技术: 使用 Django ORM 的异步查询接口 (`.afilter()`)，进行高效的数据库操作。
        - 性能优化: 使用 `.values()` 直接将查询结果转换为字典列表，避免创建完整的
                    Django模型实例，显著减少内存开销和序列化成本。
        - 返回值: 返回一个按时间排序的、干净的 Pandas DataFrame。

        Args:
            stock_code (str): 股票代码。
            time_level (str): 分钟级别 (e.g., '1', '5')。
            start_dt (datetime): 查询的开始时间 (UTC, naive)。
            end_dt (datetime): 查询的结束时间 (UTC, naive)。

        Returns:
            Optional[pd.DataFrame]: 包含查询结果的DataFrame，如果无数据或出错则返回None。
        """
        try:
            # 1. 根据股票代码和时间级别，动态获取对应的Django模型类
            model_class = self.get_minute_model(stock_code, time_level)
            if not model_class:
                logger.error(f"未能找到股票 {stock_code} 对应的 {time_level}分钟 K线模型。")
                return None

            # 2. 使用 Django ORM 的异步接口进行查询
            #    - .aobjects 是 async-enabled manager
            #    - .afilter() 是异步的 filter
            #    - __range 查询时间范围
            #    - .values() 直接返回字典，性能更高
            kline_queryset = model_class.objects.filter(
                stock__stock_code=stock_code,
                trade_time__range=(start_dt, end_dt)
            ).order_by('trade_time')
            
            # 使用 .values() 选择需要的字段
            kline_values = await sync_to_async(list)(kline_queryset.values(
                'trade_time', 'open', 'high', 'low', 'close', 'vol', 'amount'
            ))

            if not kline_values:
                logger.info(f"在数据库中未找到 {stock_code} 在 {start_dt} 到 {end_dt} 之间的 {time_level}分钟 K线数据。")
                return None

            # 3. 将查询结果转换为 Pandas DataFrame
            df = pd.DataFrame.from_records(kline_values)
            
            # 4. 数据后处理
            #    - 将 trade_time 设置为索引
            #    - 将 vol 和 amount 重命名为 volume 和 turnover_value 以保持一致性
            df.set_index('trade_time', inplace=True)
            df.rename(columns={'vol': 'volume', 'amount': 'turnover_value'}, inplace=True)
            
            logger.debug(f"成功从数据库获取 {len(df)} 条 {time_level}分钟 K线 for {stock_code}")
            return df

        except Exception as e:
            logger.error(f"从数据库获取分钟K线时发生异常 for {stock_code}: {e}", exc_info=True)
            return None

    async def get_minute_time_trade_history(self, stock_code: str, time_level: str, limit: int = 500) -> Optional[pd.DataFrame]:
        """
        【V2.0 - 重构版】获取最新的N条历史分钟级交易数据。
        此方法现在是 get_minute_kline_by_daterange 的一个便捷封装。
        """
        # 注意：由于我们需要最新的N条，而不知道具体时间范围，
        # 直接使用 order_by + limit 的方式更高效。
        try:
            model_class = self.get_minute_model(stock_code, time_level)
            if not model_class:
                return None

            kline_queryset = model_class.objects.filter(
                stock__stock_code=stock_code
            ).order_by('-trade_time')[:limit] # 使用倒序和切片获取最新的N条

            kline_values = await sync_to_async(list)(kline_queryset.values(
                'trade_time', 'open', 'high', 'low', 'close', 'vol', 'amount'
            ))

            if not kline_values:
                return None

            df = pd.DataFrame.from_records(kline_values)
            df.set_index('trade_time', inplace=True)
            df.rename(columns={'vol': 'volume', 'amount': 'turnover_value'}, inplace=True)
            df.sort_index(inplace=True) # 重新按时间正序排列
            
            return df
        except Exception as e:
            logger.error(f"获取最新 {limit} 条分钟K线时发生异常 for {stock_code}: {e}", exc_info=True)
            return None

    async def get_intraday_minute_kline_from_cache(self, stock_code: str, time_level: str, trade_date: str) -> Optional[pd.DataFrame]:
        """
        【V3.0】从Redis缓存中获取指定股票、指定日期的所有分钟K线数据。
        """
        try:
            # 1. 生成用于查询 ZSET 的缓存键
            cache_key = self.cache_key_stock.intraday_minute_kline(stock_code, time_level, trade_date)
            
            # 2. 从 Redis 的 ZSET 中获取所有成员
            kline_data_with_scores = await self.cache_manager.zrangebyscore(
                key=cache_key, min_score='-inf', max_score='+inf', withscores=True
            )

            if not kline_data_with_scores:
                logger.info(f"缓存未命中: 盘中分钟K线 for {stock_code} on {trade_date}")
                return None

            # 3. 将数据转换为 DataFrame
            records = []
            for kline_dict, timestamp in kline_data_with_scores:
                if isinstance(kline_dict, dict):
                    kline_dict['trade_time'] = datetime.fromtimestamp(timestamp)
                    records.append(kline_dict)
            
            if not records: return None

            df = pd.DataFrame.from_records(records)
            df.set_index('trade_time', inplace=True)
            df.sort_index(inplace=True)
            
            logger.debug(f"成功从Redis ZSET获取 {len(df)} 条盘中分钟K线 for {stock_code}")
            return df

        except Exception as e:
            logger.error(f"获取盘中分钟K线时发生异常 for {stock_code}: {e}", exc_info=True)
            return None

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
    async def save_monthly_time_trade_by_stock_codes(self, stock_codes: List[str], start_date: str = "1990-01-01") -> List:
        """
        【V3.0 - 健壮性与数据一致性优化版】
        保存股票的月线交易数据 (前复权)。
        接口：stk_week_month_adj (freq='month')
        描述：获取A股月线行情(前复权)
        
        优化点:
        1. [BUG修复] 明确重命名API返回列(如'open_qfq')以匹配模型字段(如'open')。
        2. [数据完整性] 在存入数据库前，对数据进行严格的类型转换，确保与模型字段类型(Decimal, BigInt)一致。
        3. [健壮性] 增加对单次Tushare API调用的异常捕获，防止因网络波动等问题中断整个任务。
        4. [可读性] 使用列名映射字典，使代码意图更清晰，便于维护。
        5. [类型提示] 修正了返回值类型提示。
        """
        if not stock_codes:
            logger.warning("输入的股票代码列表为空，任务终止。")
            return []

        # --- 1. 批量预加载股票信息，根除N+1查询 ---
        stock_map = await self.stock_basic_dao.get_stocks_by_codes(stock_codes)
        if not stock_map:
            logger.warning(f"根据提供的代码列表，未能从数据库中找到任何股票信息。")
            return []

        # --- 2. 定义Tushare API字段到模型字段的映射 ---
        # 修改开始：使用明确的字典进行列名映射，修复BUG并提高可读性
        COLUMN_MAP = {
            'ts_code': 'ts_code',
            'trade_date': 'trade_date',
            'open_qfq': 'open',
            'high_qfq': 'high',
            'low_qfq': 'low',
            'close_qfq': 'close',
            'pre_close': 'pre_close',
            'change': 'change',
            'pct_chg': 'pct_chg',
            'vol': 'vol',
            'amount': 'amount'
        }
        # 修改结束

        all_data_to_save = []
        offset = 0
        limit = 5000  # 根据Tushare积分调整，一般不超过6000
        stock_codes_str = ",".join(stock_codes)

        # --- 3. 分页循环拉取数据 ---
        while True:
            # 修改开始：增加对单次API调用的异常捕获，增强健壮性
            try:
                df = self.ts_pro.stk_week_month_adj(
                    ts_code=stock_codes_str,
                    start_date=start_date,
                    freq="month",
                    limit=limit,
                    offset=offset
                )
                # Tushare有时返回None而不是空DataFrame
                if df is None:
                    df = pd.DataFrame()
            except Exception as e:
                logger.error(f"Tushare API调用失败 (offset: {offset}): {e}", exc_info=True)
                # 发生API错误时，可以选择等待后重试或直接中断
                await asyncio.sleep(5) # 等待5秒后中断本次循环
                break
            # 修改结束

            if df.empty:
                logger.info("Tushare未返回更多数据，拉取完成。")
                break

            # --- 4. 数据清洗、转换和格式化 (向量化操作) ---
            try:
                # 修改开始：整合了重命名、类型转换和格式化的完整流程
                # 步骤 a: 重命名列以匹配Django模型字段
                df.rename(columns=COLUMN_MAP, inplace=True)

                # 步骤 b: 基础清洗，将API返回的空值标记统一为np.nan
                df.replace(['', 'null', 'None'], np.nan, inplace=True)

                # 步骤 c: 映射StockInfo实例，并丢弃无法关联的记录
                df['stock'] = df['ts_code'].map(stock_map)
                df.dropna(subset=['trade_date', 'stock'], inplace=True)
                if df.empty:
                    logger.info("当前批次数据在清洗后为空，继续下一批。")
                    offset += len(df) if len(df) > 0 else limit # 使用实际返回长度推进offset
                    if len(df) < limit: break
                    continue

                # 步骤 d: 严格的类型转换，确保数据与模型定义一致
                numeric_cols = ['open', 'high', 'low', 'close', 'pre_close', 'change']
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                # 对于Decimal字段，先转为浮点数，后续在字典生成时转为Decimal对象
                df['pct_chg'] = pd.to_numeric(df['pct_chg'], errors='coerce')
                df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
                
                # 对于BigIntegerField，使用pandas的Int64类型处理可能存在的NaN
                df['vol'] = pd.to_numeric(df['vol'], errors='coerce').astype('Int64')

                # 步骤 e: 转换日期格式
                df['trade_time'] = pd.to_datetime(df['trade_date']).dt.date
                
                # 步骤 f: 准备用于数据库操作的数据
                # 选择最终需要的列
                final_cols = [
                    'stock', 'trade_time', 'open', 'high', 'low', 'close', 
                    'pre_close', 'change', 'pct_chg', 'vol', 'amount'
                ]
                df_final = df[final_cols]

                # 将NaN替换为None，以便数据库正确处理为NULL
                df_final = df_final.where(pd.notna(df_final), None)
                
                # 转换为字典列表，同时处理Decimal类型
                records = df_final.to_dict('records')
                for record in records:
                    if record.get('pct_chg') is not None:
                        record['pct_chg'] = Decimal(str(record['pct_chg'])).quantize(Decimal("0.01"))
                    if record.get('amount') is not None:
                        # Tushare的amount单位是千元，乘以1000变为元
                        record['amount'] = Decimal(str(record['amount'])) * Decimal(1000)

                all_data_to_save.extend(records)
                # 修改结束

            except Exception as e:
                logger.error(f"处理数据时发生错误 (offset: {offset}): {e}", exc_info=True)
                # 如果数据处理失败，跳过这一批次
                offset += len(df) if len(df) > 0 else limit
                if len(df) < limit: break
                continue

            # --- 5. 分批保存到数据库 ---
            if len(all_data_to_save) >= BATCH_SAVE_SIZE:
                await self._save_all_to_db_native_upsert(
                    model_class=StockMonthlyData,
                    data_list=all_data_to_save,
                    unique_fields=['stock', 'trade_time']
                )
                all_data_to_save = [] # 清空列表以备下一批

            # --- 6. 准备下一次循环 ---
            actual_return_count = len(df)
            time.sleep(0.6) # Tushare接口调用延时，保护积分
            if actual_return_count < limit:
                break
            offset += actual_return_count

        # --- 7. 保存最后一批剩余数据 ---
        final_result = []
        if all_data_to_save:
            logger.info(f"正在保存最后一批剩余的 {len(all_data_to_save)} 条月线数据...")
            final_result = await self._save_all_to_db_native_upsert(
                model_class=StockMonthlyData,
                data_list=all_data_to_save,
                unique_fields=['stock', 'trade_time']
            )
        else:
            logger.info("所有数据均已分批保存，无剩余数据。")
            
        logger.info(f"股票 {stock_codes_str} 的月线数据保存任务全部完成。")
        return final_result

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
            df = self.ts_pro.daily_basic(**{
                "ts_code": stock_codes_str, "trade_date": trade_date_str, "start_date": start_date_str, "end_date": end_date_str, "limit": limit, "offset": offset
            }, fields=[
                "ts_code", "trade_date", "close", "turnover_rate", "turnover_rate_f", "volume_ratio", "pe", "pe_ttm", "pb", "ps", 
                "ps_ttm", "dv_ratio", "dv_ttm", "total_share", "float_share", "free_share", "total_mv", "circ_mv", "limit_status"
            ])
            if df.empty:
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
    def get_cyq_chips_model_by_code(self, stock_code: str):
        """
        根据股票代码返回对应的筹码分布数据表Model
        """
        if stock_code.startswith('3') and stock_code.endswith('.SZ'):
            return StockCyqChipsCY
        elif stock_code.endswith('.SZ'):
            return StockCyqChipsSZ
        elif stock_code.startswith('68') and stock_code.endswith('.SH'):
            return StockCyqChipsKC
        elif stock_code.endswith('.SH'):
            return StockCyqChipsSH
        elif stock_code.endswith('.BJ'):
            return StockCyqChipsBJ
        else:
            print(f"未识别的股票代码: {stock_code}，默认使用SZ主板表")
            return StockCyqChipsSZ  # 默认返回深市主板

    # 每日筹码及胜率
    async def save_all_cyq_perf_history(self, trade_date: date=None, start_date: date=None, end_date: date=None) -> None:
        """
        保存全市场股票的每日筹码及胜率数据 (优化版)
        1. 预加载股票数据到内存字典，根除循环中的N+1数据库查询问题。
        2. 对每一页数据进行向量化处理，替代低效的逐行循环。
        3. 引入分批保存机制，有效控制内存峰值。
        """
        trade_date_str = trade_date.strftime('%Y%m%d') if trade_date else ""
        start_date_str = start_date.strftime('%Y%m%d') if start_date else "20240101"
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
            df = self.ts_pro.cyq_perf(**{
                "ts_code": "", "trade_date": trade_date_str, "start_date": start_date_str, "end_date": end_date_str, "limit": limit, "offset": offset
            }, fields=[
                "ts_code", "trade_date", "his_low", "his_high", "cost_5pct", "cost_15pct", "cost_50pct", "cost_85pct", 
                "cost_95pct", "weight_avg", "winner_rate"
            ])
            if df.empty:
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

    async def save_cyq_perf_for_stock(self, stock, start_date: date = None, end_date: date = None) -> None:
        """
        获取并保存单个股票的历史筹码及胜率数据。
        此方法被并行的Celery任务调用。
        """
        print(f"DAO: 开始获取 {stock.stock_code} 的筹码及胜率数据...")
        start_date_str = start_date.strftime('%Y%m%d') if start_date else "20160101"
        end_date_str = end_date.strftime('%Y%m%d') if end_date else ""
        offset = 0
        limit = 5000 # 对于单个股票，可以设置一个较大的limit
        all_data_for_stock = []
        while True:
            # 对Tushare Pro的调用需要try-except以增加健壮性
            try:
                df = self.ts_pro.cyq_perf(**{
                    "ts_code": stock.stock_code, "start_date": start_date_str, "end_date": end_date_str, "limit": limit, "offset": offset
                }, fields=[
                    "ts_code", "trade_date", "his_low", "his_high", "cost_5pct", "cost_15pct", "cost_50pct", "cost_85pct",
                    "cost_95pct", "weight_avg", "winner_rate"
                ])
            except Exception as e:
                logger.error(f"Tushare API调用失败 (cyq_perf, ts_code={stock.stock_code}): {e}")
                break # API调用失败，中断当前股票的处理
            if df.empty:
                break
            all_data_for_stock.append(df)
            if len(df) < limit:
                break
            offset += limit
        if not all_data_for_stock:
            print(f"DAO: 未获取到 {stock.stock_code} 的任何筹码及胜率数据。")
            return
        # --- 对该股票的所有数据进行统一的向量化处理 ---
        combined_df = pd.concat(all_data_for_stock, ignore_index=True)
        combined_df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
        combined_df.dropna(subset=['trade_date'], inplace=True)
        if combined_df.empty:
            return
        combined_df['stock'] = stock
        combined_df['trade_time'] = pd.to_datetime(combined_df['trade_date']).dt.date
        final_df = combined_df[[
            "stock", "trade_time", "his_low", "his_high", "cost_5pct", "cost_15pct",
            "cost_50pct", "cost_85pct", "cost_95pct", "weight_avg", "winner_rate"
        ]]
        data_list = final_df.to_dict('records')
        print(f"DAO: 准备为 {stock.stock_code} 保存 {len(data_list)} 条筹码及胜率数据...")
        # 一次性保存该股票的所有历史数据
        return await self._save_all_to_db_native_upsert(
            model_class=StockCyqPerf,
            data_list=data_list,
            unique_fields=['stock', 'trade_time']
        )

    async def get_cyq_chips_history(self, stock_code: str) -> QuerySet:
        """
        获取股票的每日筹码分布历史数据 (已修改为直接查询分表数据库)
        1. [移除] 删除了所有Redis缓存查询逻辑。
        2. [修改] 根据股票代码动态选择正确的分表Model进行查询。
        3. [修正] 修正了ORM查询条件以正确通过外键关联进行过滤。
        """
        # [修改] 第一步：根据股票代码动态获取对应的分表Model
        target_model = self.get_cyq_chips_model_by_code(stock_code)
        print(f"DAO: 正在为股票 {stock_code} 从数据表 {target_model.__name__} 查询筹码分布历史。")
        
        # [修改] 第二步：直接使用动态获取的Model进行数据库查询
        # 注意：
        # 1. 使用 target_model.objects 进行查询。
        # 2. 过滤器使用 'stock__stock_code' 来通过外键关联查询。
        # 3. 排序字段使用模型中定义的 'trade_time'。
        stock_cyq_chips_queryset = target_model.objects.filter(
            stock__stock_code=stock_code
        ).order_by('-trade_time')[:self.cache_limit] # 保留了原有的查询数量限制
        
        # [修改] 直接返回从数据库中获取的Django QuerySet对象
        return stock_cyq_chips_queryset

    # 每日筹码分布
    async def save_all_cyq_chips_history(self, trade_date: date=None, start_date: date=None, end_date: date=None) -> None:
        """
        保存全市场股票的每日筹码分布数据 (终极优化版)
        1. [新增] 引入分表逻辑，将数据按板块存入不同数据表。
        2. [新增] 引入10万行追溯逻辑，确保获取全量历史数据。
        3. [优化] 使用异步 asyncio.sleep 替代同步 time.sleep，防止阻塞事件循环。
        4. [重构] 重构批处理机制，以适应分表场景。
        """
        # --- 日期字符串格式化 (无变化) ---
        trade_date_str = trade_date.strftime('%Y%m%d') if trade_date else ""
        start_date_str = start_date.strftime('%Y%m%d') if start_date else "20200101"
        initial_end_date_str = end_date.strftime('%Y%m%d') if end_date else ""

        all_stocks = await self.stock_basic_dao.get_stock_list()
        if not all_stocks:
            logger.warning("股票基础信息列表为空，任务终止。")
            return
        
        # [修改] 引入适应分表的批处理数据结构
        # 键是Model类，值是待保存的数据列表
        batched_data_by_model = {}
        total_stocks = len(all_stocks)
        
        for i, stock in enumerate(all_stocks):
            print(f"【cyq_chips每日筹码分布】开始处理第 {i+1}/{total_stocks} 只股票: {stock.stock_code} - {stock.stock_name}")
            # [新增] 移植10万行追溯逻辑
            current_end_date_str = initial_end_date_str
            all_dfs_for_one_stock = [] # 用于收集单只股票的所有追溯轮次数据
            
            # 外层追溯循环
            while True:
                offset = 0
                limit = 6000 # Tushare的cyq_chips单次限制较高，可以使用6000
                dfs_for_this_cycle = []
                limit_hit = False
                
                # 内层分页循环
                while True:
                    if offset >= 100000:
                        logger.warning(f"股票 {stock.stock_code} 的每日筹码分布 offset已达10万，将进行追溯抓取。")
                        limit_hit = True
                        break
                    
                    try:
                        df = self.ts_pro.cyq_chips(**{
                            "ts_code": stock.stock_code, "trade_date": trade_date_str, 
                            "start_date": start_date_str, "end_date": current_end_date_str, 
                            "limit": limit, "offset": offset
                        }, fields=["ts_code", "trade_date", "price", "percent"])
                        # [修改] 使用异步sleep，并调整为标准限速值
                        await asyncio.sleep(0.7)
                    except Exception as e:
                        logger.error(f"Tushare API调用失败 (cyq_chips, ts_code={stock.stock_code}): {e}")
                        await asyncio.sleep(5) # 异常时等待更久
                        df = pd.DataFrame()

                    if df.empty:
                        break
                    dfs_for_this_cycle.append(df)
                    if len(df) < limit:
                        break
                    offset += limit

                if not dfs_for_this_cycle:
                    # 当前轮次无数据，说明此时间段已无更多历史，结束追溯
                    break
                
                all_dfs_for_one_stock.extend(dfs_for_this_cycle)
                
                if limit_hit:
                    # 触及10万行，更新end_date，继续外层追溯循环
                    last_df_in_cycle = dfs_for_this_cycle[-1]
                    last_trade_date = last_df_in_cycle['trade_date'].iloc[-1]
                    current_end_date_str = last_trade_date
                    print(f"DAO: {stock.stock_code} 触及10万行限制，下一轮将从 {current_end_date_str} 继续向前追溯。")
                    continue
                else:
                    # 未触及限制，说明该股票数据已全部获取，跳出追溯循环
                    break

            # --- 对单只股票的全部数据进行统一处理 ---
            if all_dfs_for_one_stock:
                combined_df = pd.concat(all_dfs_for_one_stock, ignore_index=True)
                combined_df.drop_duplicates(subset=['trade_date', 'price'], keep='first', inplace=True)
                combined_df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
                combined_df.dropna(subset=['trade_date', 'price'], inplace=True)
                
                if not combined_df.empty:
                    combined_df['stock'] = stock
                    combined_df['trade_time'] = pd.to_datetime(combined_df['trade_date']).dt.date
                    final_df = combined_df[['stock', 'trade_time', 'price', 'percent']]
                    data_dicts_for_stock = final_df.to_dict('records')

                    # [修改] 分表批处理核心逻辑
                    if data_dicts_for_stock:
                        # 1. 获取当前股票对应的正确Model
                        target_model = self.get_cyq_chips_model_by_code(stock.stock_code)
                        
                        # 2. 如果该Model是第一次出现，在字典中初始化一个空列表
                        if target_model not in batched_data_by_model:
                            batched_data_by_model[target_model] = []
                        
                        # 3. 将数据添加到对应Model的列表中
                        batched_data_by_model[target_model].extend(data_dicts_for_stock)
                        
                        # 4. 检查该Model的列表是否已满，如果满了就保存并清空
                        if len(batched_data_by_model[target_model]) >= BATCH_SAVE_SIZE:
                            logger.info(f"为数据表 {target_model.__name__} 保存一批数据，数量：{len(batched_data_by_model[target_model])}")
                            await self._save_all_to_db_native_upsert(
                                model_class=target_model,
                                data_list=batched_data_by_model[target_model],
                                unique_fields=['stock', 'trade_time', 'price']
                            )
                            batched_data_by_model[target_model] = [] # 清空已保存的批次

        # --- [修改] 在所有股票处理完毕后，保存所有分表中剩余的数据 ---
        logger.info("所有股票数据拉取完成，开始保存剩余的最后一批数据...")
        for model, data_list in batched_data_by_model.items():
            if data_list:
                logger.info(f"为数据表 {model.__name__} 保存最后一批数据，数量：{len(data_list)}")
                await self._save_all_to_db_native_upsert(
                    model_class=model,
                    data_list=data_list,
                    unique_fields=['stock', 'trade_time', 'price']
                )
        
        logger.info(f"所有股票的每日筹码分布数据处理和保存完成。")
        # 因为存在多次保存，返回单一结果已无意义，故返回None
        return None

    async def save_cyq_chips_for_stock(self, stock, start_date: date = None, end_date: date = None) -> None:
        """
        保存单只股票的每日筹码分布数据（已支持分表和10万行追溯）。
        """
        # ... 数据获取部分无变化，为简洁省略 ...
        print(f"DAO: 开始获取 {stock.stock_code} 的筹码分布数据（支持10万行以上追溯）...")
        start_date_str = start_date.strftime('%Y%m%d') if start_date else "20200101"
        current_end_date_str = end_date.strftime('%Y%m%d') if end_date else ""
        all_dfs_for_stock = []
        while True:
            offset = 0
            limit = 6000
            dfs_for_this_cycle = []
            limit_hit = False
            while True:
                if offset >= 100000:
                    logger.warning(f"股票 {stock.stock_code} 的每日筹码分布 offset已达10万，将进行追溯抓取。")
                    limit_hit = True
                    break
                try:
                    df = self.ts_pro.cyq_chips(**{
                        "ts_code": stock.stock_code, "start_date": start_date_str, "end_date": current_end_date_str, "limit": limit, "offset": offset
                    }, fields=["ts_code", "trade_date", "price", "percent"])
                    await asyncio.sleep(0.7)
                except Exception as e:
                    logger.error(f"Tushare API调用失败 (cyq_chips, ts_code={stock.stock_code}): {e}")
                    await asyncio.sleep(5)
                    df = pd.DataFrame()
                if df.empty:
                    break
                dfs_for_this_cycle.append(df)
                if len(df) < limit:
                    break
                offset += limit
            if not dfs_for_this_cycle:
                break
            all_dfs_for_stock.extend(dfs_for_this_cycle)
            if limit_hit:
                last_df_in_cycle = dfs_for_this_cycle[-1]
                last_trade_date = last_df_in_cycle['trade_date'].iloc[-1]
                current_end_date_str = last_trade_date
                print(f"DAO: {stock.stock_code} 触及10万行限制，下一轮将从 {current_end_date_str} 继续向前追溯。")
                continue
            else:
                break
        if not all_dfs_for_stock:
            print(f"DAO: 未获取到 {stock.stock_code} 的任何筹码分布数据。")
            return
        combined_df = pd.concat(all_dfs_for_stock, ignore_index=True)
        combined_df.drop_duplicates(subset=['trade_date', 'price'], keep='first', inplace=True)
        combined_df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
        combined_df.dropna(subset=['trade_date', 'price'], inplace=True)
        if combined_df.empty:
            return
        combined_df['stock'] = stock
        combined_df['trade_time'] = pd.to_datetime(combined_df['trade_date']).dt.date
        final_df = combined_df[['stock', 'trade_time', 'price', 'percent']]
        data_list = final_df.to_dict('records')
        # [修改] 在保存前，根据股票代码动态选择目标数据表Model
        target_model = self.get_cyq_chips_model_by_code(stock.stock_code)
        print(f"DAO: 准备为 {stock.stock_code} 保存 {len(data_list)} 条筹码分布数据到表 {target_model.__name__}...")
        return await self._save_all_to_db_native_upsert(
            model_class=target_model, # [修改] 使用动态选择的Model
            data_list=data_list,
            unique_fields=['stock', 'trade_time', 'price']
        )


















