# dao_manager\tushare_daos\stock_time_trade_dao.py
import os
import asyncio
from decimal import Decimal
import pytz
import logging
import time
from utils.rate_limiter import with_rate_limit
from django.db.models import QuerySet
from asgiref.sync import sync_to_async
from typing import Dict, List, Optional
from collections import defaultdict # 导入 defaultdict 以方便分组
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from dao_manager.base_dao import BaseDAO
from stock_models.index import TradeCalendar
from utils.model_helpers import get_daily_data_model_by_code, get_cyq_chips_model_by_code, get_minute_data_model_by_code_and_timelevel, get_stk_limit_model_by_code
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from stock_models.stock_basic import StockInfo
from stock_models.time_trade import StockDailyBasic, StockWeeklyData, StockMonthlyData
from stock_models.chip import StockCyqPerf
from utils.cache_get import StockInfoCacheGet, StockTimeTradeCacheGet
from utils.cache_manager import CacheManager
from utils.cache_set import StockInfoCacheSet, StockTimeTradeCacheSet
from utils.cash_key import StockCashKey
from utils.data_format_process import StockInfoFormatProcess, StockTimeTradeFormatProcess


BATCH_SAVE_SIZE = 110000  # 每10000条数据保存一次
logger = logging.getLogger("dao")
time_levels = ["5", "15", "30", "60"] # "1", 

class StockTimeTradeDAO(BaseDAO):
    def __init__(self, cache_manager_instance: CacheManager):
        # 调用 super() 时，将 cache_manager_instance 传递进去
        super().__init__(cache_manager_instance=cache_manager_instance, model_class=None)
        self.stock_basic_dao = StockBasicInfoDao(cache_manager_instance)
        self.cache_limit = 500 # 定义缓存数量上限
        self.cache_key = StockCashKey()
        self.data_format_process_trade = StockTimeTradeFormatProcess(cache_manager_instance)
        self.data_format_process_stock = StockInfoFormatProcess(cache_manager_instance)
        self.cache_set = StockTimeTradeCacheSet(self.cache_manager)
        self.cache_get = StockTimeTradeCacheGet(self.cache_manager)
        self.stock_cache_set = StockInfoCacheSet(self.cache_manager)
        self.stock_cache_get = StockInfoCacheGet(self.cache_manager)
    # =============== A股日线行情 ===============
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
            model_class = get_daily_data_model_by_code(code)
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

    async def get_daily_data_for_stocks(self, stock_codes: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        【V2.1 - 字段补全版】
        """
        print(f"[DAO] StockTimeTradeDAO.get_daily_data_for_stocks: 正在为 {len(stock_codes)} 支股票获取 {start_date} 到 {end_date} 的数据...")
        if not stock_codes:
            return pd.DataFrame()
        try:
            model_to_codes_map = defaultdict(list)
            for code in stock_codes:
                model_class = get_daily_data_model_by_code(code)
                model_to_codes_map[model_class].append(code)
            start_dt = datetime.strptime(start_date, '%Y%m%d').date()
            end_dt = datetime.strptime(end_date, '%Y%m%d').date()
            query_tasks = []
            for model_class, codes in model_to_codes_map.items():
                # [修正] 在 .values() 中增加 'open_qfq' 字段
                queryset = model_class.objects.filter(
                    stock__stock_code__in=codes,
                    trade_time__gte=start_dt,
                    trade_time__lte=end_dt
                ).values('stock__stock_code', 'trade_time', 'open_qfq', 'close_qfq', 'high_qfq', 'low_qfq')
                query_tasks.append(sync_to_async(list)(queryset))
            results_list_of_lists = await asyncio.gather(*query_tasks)
            all_data_list = [item for sublist in results_list_of_lists for item in sublist]
            if not all_data_list:
                logger.warning(f"[DAO] 未能在任何分表中找到 {len(stock_codes)} 支股票在 {start_date}-{end_date} 期间的日线数据。")
                return pd.DataFrame()
            df = pd.DataFrame(all_data_list)
            # [修正] 增加对 'open_qfq' 的重命名
            df.rename(columns={
                'stock__stock_code': 'stock_code',
                'open_qfq': 'open',
                'close_qfq': 'close',
                'high_qfq': 'high',
                'low_qfq': 'low'
            }, inplace=True)
            df['trade_time'] = pd.to_datetime(df['trade_time']).dt.date
            print(f"[DAO] 成功获取并合并了 {len(df)} 条日线数据。")
            return df
        except Exception as e:
            logger.error(f"[DAO] 批量获取日线数据时发生错误: {e}", exc_info=True)
            return pd.DataFrame()

    async def get_daily_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        【V1.3 - 字段补全版】
        修改思路：
        1. 增加 'amount' (成交额) 字段，用于计算 net_amount_ratio (净流入占比)。
        2. 保留之前的 'pct_change' 等字段。
        """
        # print(f"[DAO] StockTimeTradeDAO.get_daily_data: 正在为 {stock_code} 获取 {start_date} 到 {end_date} 的数据...")
        try:
            model_class = get_daily_data_model_by_code(stock_code)
            start_dt = datetime.strptime(start_date, '%Y%m%d').date()
            end_dt = datetime.strptime(end_date, '%Y%m%d').date()
            queryset = model_class.objects.filter(
                stock__stock_code=stock_code,
                trade_time__gte=start_dt,
                trade_time__lte=end_dt
            ).order_by('trade_time')
            
            # [修正] 增加 'amount' 字段
            data_list = [item async for item in queryset.values('trade_time', 'open_qfq', 'close_qfq', 'high_qfq', 'low_qfq', 'pct_change', 'amount')]
            
            if not data_list:
                logger.warning(f"[DAO] 未能在 {model_class.__name__} 表中找到 {stock_code} 在 {start_date}-{end_date} 期间的日线数据。")
                return pd.DataFrame()
            df = pd.DataFrame(data_list)
            # [修正] 重命名
            df.rename(columns={'open_qfq': 'open', 'close_qfq': 'close', 'high_qfq': 'high', 'low_qfq': 'low'}, inplace=True)
            df['trade_time'] = pd.to_datetime(df['trade_time'])
            df.set_index('trade_time', inplace=True)
            return df
        except Exception as e:
            logger.error(f"[DAO] 获取 {stock_code} 日线数据范围查询时发生错误: {e}", exc_info=True)
            return pd.DataFrame()

    async def get_daily_data_by_date(self, stock_code: str, trade_date: str) -> pd.DataFrame:
        """
        【V1.1 - 字段补全版】
        """
        # print(f"[DAO] StockTimeTradeDAO.get_daily_data: 正在为 {stock_code} 获取 {start_date} 到 {end_date} 的数据...")
        try:
            model_class = get_daily_data_model_by_code(stock_code)
            queryset = model_class.objects.filter(stock__stock_code=stock_code, trade_time=trade_date).first()
            # [修正] 在 .values() 中增加 'open_qfq' 字段
            data_list = [item async for item in queryset.values('trade_time', 'open_qfq', 'close_qfq', 'high_qfq', 'low_qfq')]
            if not data_list:
                logger.warning(f"[DAO] 未能在 {model_class.__name__} 表中找到 {stock_code} 在 {start_date}-{end_date} 期间的日线数据。")
                return pd.DataFrame()
            df = pd.DataFrame(data_list)
            # [修正] 增加对 'open_qfq' 的重命名
            df.rename(columns={'open_qfq': 'open', 'close_qfq': 'close', 'high_qfq': 'high', 'low_qfq': 'low'}, inplace=True)
            df['trade_time'] = pd.to_datetime(df['trade_time'])
            df.set_index('trade_time', inplace=True)
            return df
        except Exception as e:
            logger.error(f"[DAO] 获取 {stock_code} 日线数据范围查询时发生错误: {e}", exc_info=True)
            return pd.DataFrame()

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
            model_class = get_daily_data_model_by_code(stock_code)
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

    async def get_kl_data_for_chart(self, stock_code: str, start_date: date, end_date: date) -> List[Dict]:
        """
        【V1.0】为前端图表获取K线数据。
        - 核心职责: 根据股票代码和日期范围，从正确的分表中查询OHLC和成交量数据。
        - 数据格式: 返回一个适合直接JSON序列化的字典列表。
        """
        try:
            model_class = get_daily_data_model_by_code(stock_code)
            if not model_class:
                logger.error(f"[DAO] get_kl_data_for_chart: 未能为股票 {stock_code} 找到日线数据模型。")
                return []
            # 使用异步ORM进行查询
            queryset = model_class.objects.filter(
                stock__stock_code=stock_code,
                trade_time__gte=start_date,
                trade_time__lte=end_date
            ).order_by('trade_time')
            # 使用.values()直接获取字典，并指定字段，性能更优
            # 注意：我们使用前复权数据 'close_qfq' 等
            kline_data = await sync_to_async(list)(queryset.values(
                'trade_time', 'open_qfq', 'close_qfq', 'low_qfq', 'high_qfq', 'vol'
            ))
            return kline_data
        except Exception as e:
            logger.error(f"[DAO] get_kl_data_for_chart: 查询 {stock_code} 的K线数据时发生错误: {e}", exc_info=True)
            return []

    @with_rate_limit(name='api_daily')
    async def save_daily_time_trade_history_by_trade_dates(self, trade_date: date = None, start_date: date = None,
                                                           end_date: date = None, *, limiter) -> dict:
        """
        【V2.0 智能参数修正版】根据指定的日期或日期范围，获取并保存所有股票的日线交易数据。
        - 核心修正: 动态构建API参数。
          - 当 trade_date 被提供时，API调用使用 'trade_date' 参数。
          - 当 start_date/end_date 被提供时，API调用使用 'start_date'/'end_date' 参数。
        这解决了当 start_date 和 end_date 相同时，API返回空数据的问题。
        - 【修复】移除对不存在的 `format_daily_dataframe` 方法的调用，将数据处理逻辑内联。
        - 【优化】采用向量化处理和分表批量保存，提高效率。
        :param trade_date: 单个交易日期。
        :param start_date: 开始日期。
        :param end_date: 结束日期。
        :param limiter: 由 @with_rate_limit 装饰器注入的 DistributedRateLimiter 实例。
        :return: 包含处理结果的字典。
        """
        all_stocks = await self.stock_basic_dao.get_stock_list()
        if not all_stocks:
            logger.warning("数据库中没有股票基础数据，无法执行日线数据保存任务。")
            return {"status": "warning", "message": "No stock basic data found."}
        # 格式化所有可能的日期参数，None值保持为None
        trade_date_str = trade_date.strftime('%Y%m%d') if trade_date else None
        start_date_str = start_date.strftime('%Y%m%d') if start_date else None
        end_date_str = end_date.strftime('%Y%m%d') if end_date else None
        all_dfs = []
        for stock in all_stocks:
            try:
                while not await limiter.acquire():
                    print(f"PID[{os.getpid()}] API[api_daily] 速率超限，等待10秒后重试... (股票: {stock.stock_code})")
                    await asyncio.sleep(10)
                # 动态构建API参数字典，这是解决问题的关键
                api_params = {
                    "ts_code": stock.stock_code,
                }
                if trade_date_str:
                    api_params['trade_date'] = trade_date_str
                else:
                    # 只有在没有指定单日查询时，才使用日期范围
                    if start_date_str:
                        api_params['start_date'] = start_date_str
                    if end_date_str:
                        api_params['end_date'] = end_date_str
                print(f"调试: DAO准备调用Tushare API [daily]，动态参数: {api_params}")
                df = self.ts_pro.daily(**api_params)
                if not df.empty:
                    # 将 stock 对象直接添加到 DataFrame 中，方便后续处理
                    df['stock'] = stock
                    all_dfs.append(df)
            except Exception as e:
                logger.error(f"获取股票 {stock.stock_code} 的日线数据时出错: {e}", exc_info=True)
                await asyncio.sleep(5)
        if not all_dfs:
            print("DAO: 未获取到任何股票的日线数据。")
            return {"status": "success", "message": "No daily data fetched, task completed."}
        combined_df = pd.concat(all_dfs, ignore_index=True)
        # --- 开始向量化处理和分表逻辑 ---
        # 1. 数据清洗
        combined_df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
        # 确保关键字段和stock对象存在，ts_code用于获取模型，trade_date用于日期，stock用于外键
        combined_df.dropna(subset=['ts_code', 'trade_date', 'stock'], inplace=True)
        if combined_df.empty:
            logger.warning("合并后的日线数据为空或清洗后为空，任务终止。")
            return {"status": "success", "message": "Combined daily data is empty after cleaning.", "创建/更新成功": 0}
        # 2. 向量化转换日期
        combined_df['trade_time'] = pd.to_datetime(combined_df['trade_date'], format='%Y%m%d').dt.date
        # 3. 确定模型类并分组
        # 使用 apply 方法动态确定每行数据应属的模型
        combined_df['model_class'] = combined_df['ts_code'].apply(get_daily_data_model_by_code)
        data_dicts_by_model = defaultdict(list)
        # 定义需要保存的列，并处理名称不一致的情况 (pct_chg -> pct_change)
        # Tushare daily API返回的字段：ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount
        columns_to_keep = [
            'stock', 'trade_time', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount'
        ]
        for model_class, group_df in combined_df.groupby('model_class', sort=False):
            if group_df.empty:
                continue
            # 筛选出当前模型需要的列
            final_cols = [col for col in columns_to_keep if col in group_df.columns]
            # 重命名 'pct_chg' 为 'pct_change' 以匹配模型字段
            group_df_renamed = group_df[final_cols].rename(columns={'pct_chg': 'pct_change'})
            # 【新增】处理 amount 字段单位，Tushare daily API的amount是万元，转换为元
            # 先确保amount列是数值类型，再进行乘法
            if 'amount' in group_df_renamed.columns:
                group_df_renamed['amount'] = pd.to_numeric(group_df_renamed['amount'], errors='coerce') * 10000
            # 将 NaN 转换为 None 以适配数据库
            data_to_save = group_df_renamed.where(pd.notnull(group_df_renamed), None).to_dict('records')
            data_dicts_by_model[model_class].extend(data_to_save)
        # 4. 批量保存到数据库
        total_saved_count = 0
        for model_class, data_list in data_dicts_by_model.items():
            if not data_list:
                continue
            print(f"调试信息: 正在为模型 {model_class.__name__} 保存 {len(data_list)} 条数据...")
            res = await self._save_all_to_db_native_upsert(
                model_class=model_class,
                data_list=data_list,
                unique_fields=['stock', 'trade_time']
            )
            total_saved_count += res.get("创建/更新成功", 0)
        print(f"DAO: 保存完成，共 {total_saved_count} 条日线数据。")
        return {"status": "success", "message": f"Saved {total_saved_count} daily trade records.", "创建/更新成功": total_saved_count}

    async def save_daily_time_trade_history_by_stock_codes(
        self, 
        stock_codes: List[str], 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict:
        """
        【V2.2 - 支持日期范围查询，保留stk_factor】
        保存多只股票在指定日期范围内的日线交易数据。
        - 如果 start_date 和 end_date 相同 (或只提供了 start_date)，则执行单日查询。
        - 如果 start_date 和 end_date 不同，则执行范围查询（带分页）。
        - 如果均未提供，则获取全部历史数据（带分页）。
        """
        if not stock_codes:
            logger.warning("传入的stock_codes列表为空，任务终止。")
            return {}
        # 1. 一次性获取股票信息
        all_stocks = await self.stock_basic_dao.get_stock_list()
        stock_map = {stock.stock_code: stock for stock in all_stocks if stock.stock_code in stock_codes}
        if not stock_map:
            logger.warning(f"提供的stock_codes: {stock_codes} 在数据库中均未找到对应的StockInfo。")
            return {}
        stock_codes_str = ",".join(stock_codes)
        data_dicts_by_model = {}
        # 内部函数，用于处理DataFrame，避免代码重复
        def process_dataframe(df: pd.DataFrame):
            # (This helper function remains the same as the previous version)
            if df.empty: return
            df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
            df.dropna(subset=['ts_code', 'trade_date'], inplace=True)
            df['stock'] = df['ts_code'].map(stock_map)
            df.dropna(subset=['stock'], inplace=True)
            if df.empty: return
            df['trade_time'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.date
            df['model_class'] = df['ts_code'].apply(get_daily_data_model_by_code)
            for model_class, group_df in df.groupby('model_class', sort=False):
                columns_to_keep = [
                    'stock', 'trade_time', 'close', 'open', 'high', 'low', 'pre_close', 'change', 'pct_change', 'vol',
                    'amount', 'adj_factor', 'open_hfq', 'open_qfq', 'close_hfq', 'close_qfq', 'high_hfq', 'high_qfq', 'low_hfq',
                    'low_qfq', 'pre_close_hfq', 'pre_close_qfq'
                ]
                final_cols = [col for col in columns_to_keep if col in group_df.columns]
                data_to_save = group_df[final_cols].where(pd.notnull(group_df), None).to_dict('records')
                if model_class not in data_dicts_by_model:
                    data_dicts_by_model[model_class] = []
                data_dicts_by_model[model_class].extend(data_to_save)
        # 2. 准备API参数
        api_params = {
            "ts_code": stock_codes_str,
            "fields": [
                "ts_code", "trade_date", "close", "open", "high", "low", "pre_close", "change", "pct_change", "vol",
                "amount", "adj_factor", "open_hfq", "open_qfq", "close_hfq", "close_qfq", "high_hfq", "high_qfq", "low_hfq",
                "low_qfq", "pre_close_hfq", "pre_close_qfq"
            ]
        }
        # 3. 根据日期参数决定API调用策略
        # Case 1: Single Day Query
        if start_date and (end_date is None or start_date == end_date):
            logger.info(f"执行单日查询: date={start_date}, stocks={len(stock_codes)}个")
            api_params["trade_date"] = start_date
            df = self.ts_pro.stk_factor(**api_params)
            process_dataframe(df)
        # Case 2: Date Range or Full History Query (both require pagination)
        else:
            if start_date and end_date:
                logger.info(f"执行范围查询: {start_date} to {end_date}, stocks={len(stock_codes)}个")
                api_params["start_date"] = start_date
                api_params["end_date"] = end_date
            else:
                logger.info(f"执行历史查询 (无日期范围): stocks={len(stock_codes)}个")
                api_params["start_date"] = "20000101"
            offset = 0
            limit = 6000
            while True:
                if offset >= 100000:
                    logger.warning(f"offset已达10万，停止拉取。params={api_params}")
                    break
                api_params["offset"] = offset
                api_params["limit"] = limit
                df = self.ts_pro.stk_factor(**api_params)
                if df.empty:
                    break
                process_dataframe(df)
                if len(df) < limit:
                    break
                offset += limit
        # 4. 批量保存
        result = {}
        for model_class, data_list in data_dicts_by_model.items():
            if not data_list: continue
            print(f"调试信息: 正在为模型 {model_class.__name__} 保存 {len(data_list)} 条数据...")
            res = await self._save_all_to_db_native_upsert(
                model_class=model_class,
                data_list=data_list,
                unique_fields=['stock', 'trade_time']
            )
            result[model_class.__name__] = res
        return result

    # =============== A股分钟行情 ===============
    async def get_intraday_kline_by_date(self, stock_code: str, trade_date: datetime.date, time_level: str = '1') -> Optional[pd.DataFrame]:
        """
        【V2.2 · 统一输出UTC aware datetime版】获取指定股票在指定交易日的所有分钟K线数据。
        - 核心升级: 统一时区处理逻辑，确保返回的DataFrame索引是标准的UTC aware datetime。
        - 【修正】使用明确的 UTC aware datetime 范围进行过滤，并添加调试探针。
        - 【修复】修正 NameError: 'kline_values' 未定义的问题。
        """
        from django.utils import timezone
        from datetime import datetime, time, timedelta
        try:
            model_class = get_minute_data_model_by_code_and_timelevel(stock_code, time_level)
            if not model_class:
                logger.error(f"未能找到股票 {stock_code} 对应的 {time_level}分钟 K线模型。")
                return None
            start_dt_aware = timezone.make_aware(datetime.combine(trade_date, time.min), timezone=pytz.timezone('Asia/Shanghai')).astimezone(pytz.utc)
            end_dt_aware = timezone.make_aware(datetime.combine(trade_date + timedelta(days=1), time.min), timezone=pytz.timezone('Asia/Shanghai')).astimezone(pytz.utc)
            kline_queryset = model_class.objects.filter(
                stock__stock_code=stock_code,
                trade_time__gte=start_dt_aware,
                trade_time__lt=end_dt_aware
            ).order_by('trade_time').values( # 添加 .values() 来指定获取的字段
                'trade_time', 'open', 'high', 'low', 'close', 'vol', 'amount'
            )
            kline_values = await sync_to_async(list)(kline_queryset) # 执行 QuerySet 并赋值给 kline_values
            if not kline_values:
                # logger.warning(f"在数据库 {model_class._meta.db_table} 中未找到 {stock_code} 在 {trade_date} 的 {time_level}分钟 K线数据。")
                return None
            df = pd.DataFrame.from_records(kline_values)
            df['trade_time'] = pd.to_datetime(df['trade_time'], utc=True)
            df.set_index('trade_time', inplace=True)
            df.rename(columns={'vol': 'volume'}, inplace=True)
            logger.debug(f"成功从数据库获取 {len(df)} 条 {trade_date} 的 {time_level}分钟 K线 for {stock_code}")
            return df
        except Exception as e:
            logger.error(f"获取当日 {time_level}分钟 K线时发生异常 for {stock_code} on {trade_date}: {e}", exc_info=True)
            return None

    async def get_1_min_kline_time_by_day(self, stock_code: str, trade_date: datetime.date) -> Optional[pd.DataFrame]:
        """
        【V1.2 · 统一输出UTC aware datetime版】获取指定股票在指定日期的所有1分钟K线数据。
        - 核心升级: 统一时区处理逻辑，确保返回的DataFrame索引是标准的UTC aware datetime。
        - 【修正】添加调试探针。
        - 【修复】修正 NameError: 'kline_values' 未定义的问题。
        """
        from django.utils import timezone
        from datetime import datetime, time, timedelta
        try:
            model_class = get_minute_data_model_by_code_and_timelevel(stock_code, '1')
            if not model_class:
                logger.error(f"未能找到股票 {stock_code} 对应的1分钟K线模型。")
                return None
            start_datetime = timezone.make_aware(datetime.combine(trade_date, time.min), timezone=pytz.timezone('Asia/Shanghai')).astimezone(pytz.utc)
            end_datetime = timezone.make_aware(datetime.combine(trade_date + timedelta(days=1), time.min), timezone=pytz.timezone('Asia/Shanghai')).astimezone(pytz.utc)
            kline_queryset = model_class.objects.filter(
                stock__stock_code=stock_code,
                trade_time__gte=start_datetime,
                trade_time__lt=end_datetime
            ).order_by('trade_time').values( # 添加 .values() 来指定获取的字段
                'trade_time', 'open', 'high', 'low', 'close', 'vol', 'amount'
            )
            kline_values = await sync_to_async(list)(kline_queryset) # 执行 QuerySet 并赋值给 kline_values
            if not kline_values:
                logger.warning(f"在数据库 {model_class._meta.db_table} 中未找到 {stock_code} 在 {trade_date} 的1分钟K线数据。")
                return None
            df = pd.DataFrame.from_records(kline_values)
            df['trade_time'] = pd.to_datetime(df['trade_time'], utc=True)
            df.set_index('trade_time', inplace=True)
            df.rename(columns={'vol': 'volume'}, inplace=True)
            logger.debug(f"成功从数据库获取 {len(df)} 条 {trade_date} 的1分钟K线 for {stock_code}")
            return df
        except Exception as e:
            logger.error(f"获取当日1分钟K线时发生异常 for {stock_code} on {trade_date}: {e}", exc_info=True)
            return None

    @with_rate_limit(name='api_stk_mins') # 添加速率限制装饰器
    async def save_minute_time_trade_history_by_stock_codes(self, stock_codes: List[str], start_date_str: str="2020-01-01 00:00:00", end_date_str: str="", *, limiter) -> None: # 增加limiter参数
        """
        【V5.1 速率限制版】保存股票的历史分钟级交易数据
        - 核心优化:
          1. 【向量化处理】使用Pandas的向量化操作替代了原有的 `groupby()` 和循环，大幅提升了数据处理效率。
          2. 【内存优化】在处理完每一页数据后，及时进行分表和保存，避免将所有数据加载到内存中。
          3. 【速率限制】集成了 'api_stk_mins' 分布式速率限制器。
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
                # --- 新增的代码行开始 ---
                # 在API调用前获取速率许可
                while not await limiter.acquire():
                    print(f"PID[{os.getpid()}] API[api_stk_mins] 速率超限，等待10秒后重试... (股票: {stock_codes_str}, Freq: {time_level}min)")
                    await asyncio.sleep(10)
                # --- 新增的代码行结束 ---
                try:
                    df = self.ts_pro.stk_mins(**{
                        "ts_code": stock_codes_str, "freq": time_level + "min", "start_date": start_date_str, "end_date": end_date_str,
                        "limit": limit, "offset": offset
                    }, fields=["ts_code", "trade_time", "close", "open", "high", "low", "vol", "amount", "freq"])
                except Exception as e:
                    logger.error(f"Tushare API调用失败 (stk_mins): {e}", exc_info=True)
                    df = pd.DataFrame()
                    await asyncio.sleep(5)
                if df.empty:
                    print(f"拉取结束，API未返回更多 {time_level}min 数据。")
                    break
                # --- 开始向量化处理 ---
                # 1. 数据清洗与预处理
                df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
                df.dropna(subset=['ts_code', 'trade_time'], inplace=True)
                if df.empty:
                    if len(df) < limit: break
                    offset += limit
                    page_num += 1
                    continue
                # 2. 向量化关联与转换
                df['stock'] = df['ts_code'].map(stock_map)
                df.dropna(subset=['stock'], inplace=True)
                if df.empty:
                    if len(df) < limit: break
                    offset += limit
                    page_num += 1
                    continue
                # 向量化处理时区，适配原生SQL批量插入
                df['trade_time'] = pd.to_datetime(df['trade_time']).dt.tz_localize('Asia/Shanghai').dt.tz_convert('UTC').dt.tz_localize(None)
                # 3. 动态分表并保存
                # 使用 apply 方法动态确定每行数据应属的模型
                df['model_class'] = df['ts_code'].apply(lambda code: get_minute_data_model_by_code_and_timelevel(code, time_level))
                # 使用 groupby 按模型进行分组
                for model_class, group_df in df.groupby('model_class', sort=False):
                    if group_df.empty:
                        continue
                    # 为当前分组准备数据并批量保存
                    data_list = group_df[["stock", "trade_time", "close", "open", "high", "low", "vol", "amount"]].to_dict('records')
                    await self._save_all_to_db_native_upsert(
                        model_class=model_class,
                        data_list=data_list,
                        unique_fields=['stock', 'trade_time']
                    )
                    logger.info(f"保存 {model_class.__name__} 的 {time_level}分钟级数据完成. 插入/更新了 {len(data_list)} 条记录。")
                # --- 向量化处理结束 ---
                if len(df) < limit:
                    print(f"调试信息: API返回行数({len(df)})小于limit({limit})，判定为最后一页。")
                    break
                offset += limit
                page_num += 1
        logger.info(f"保存 {len(stock_codes)}个股票 的分钟级交易数据全部完成.")

    @with_rate_limit(name='api_stk_mins') # 速率限制装饰器
    async def save_1min_time_trade_history_by_stock_code(self, stock_code: str, *, limiter) -> int:
        """
        【V1.1 · 增加回溯停止日期】按30个交易日分块，从后向前获取并保存单只股票的全部历史1分钟K线数据，直到2019-03-01。
        - 核心逻辑:
          1. 从最新的交易日开始，使用交易日历分块（每块30天）向前追溯。
          2. 对每个时间块调用 Tushare 的 stk_mins 接口。
          3. 使用 @with_rate_limit 装饰器进行分布式速率限制。
          4. 当API返回空数据或日期早于2019-03-01时，任务自动终止。
        Args:
            stock_code (str): 股票代码。
            limiter: 由 @with_rate_limit 装饰器注入的 DistributedRateLimiter 实例。
        Returns:
            int: 总共保存的记录条数。
        """
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            logger.error(f"未能找到股票代码为 {stock_code} 的股票信息，任务终止。")
            return 0
        model_class = get_minute_data_model_by_code_and_timelevel(stock_code, '1')
        if not model_class:
            logger.error(f"未能为 {stock_code} 和时间级别 1min 找到对应的数据库模型，任务终止。")
            return 0
        total_saved_count = 0
        reference_date = datetime.now().date() # 从今天开始向前追溯
        # --- 新增的代码行开始 ---
        stop_date = date(2019, 3, 1) # 定义回溯停止日期
        # --- 新增的代码行结束 ---
        while True:
            # 1. 异步安全地获取一个30个交易日的批次
            get_trade_dates_async = sync_to_async(TradeCalendar.get_latest_n_trade_dates, thread_sensitive=True)
            trade_dates = await get_trade_dates_async(n=33, reference_date=reference_date)
            if not trade_dates:
                print(f"[{stock_code}] 交易日历中在 {reference_date} 之前已无更多交易日，任务结束。")
                break
            # 2. 确定该批次的开始和结束日期
            end_date_obj = trade_dates[0]      # 批次中最近的日期
            start_date_obj = trade_dates[-1]   # 批次中最远的日期
            # 检查批次的开始日期是否已经早于我们设定的停止日期
            if start_date_obj < stop_date:
                print(f"[{stock_code}] 数据追溯已到达或早于设定的停止日期 {stop_date}，任务结束。")
                break
            start_date_str = f"{start_date_obj.strftime('%Y-%m-%d')} 00:00:00"
            end_date_str = f"{end_date_obj.strftime('%Y-%m-%d')} 23:59:59"
            print(f"[{stock_code}] 准备获取时间段 {start_date_str} 到 {end_date_str} 的1分钟数据...")
            # 3. 在API调用前获取速率许可
            while not await limiter.acquire():
                print(f"PID[{os.getpid()}] API[api_stk_mins] 速率超限，等待10秒后重试... (股票: {stock_code})")
                await asyncio.sleep(10)
            try:
                # 4. 调用Tushare API
                df = self.ts_pro.stk_mins(
                    ts_code=stock_code,
                    freq='1min',
                    start_date=start_date_str,
                    end_date=end_date_str
                )
            except Exception as e:
                logger.error(f"为 {stock_code} 获取1分钟数据时Tushare API调用失败: {e}", exc_info=True)
                await asyncio.sleep(60) # API异常时等待更长时间
                continue # 继续下一次循环尝试
            if df.empty:
                print(f"[{stock_code}] 在时间段 {start_date_str} 到 {end_date_str} 未获取到数据，可能已达历史数据尽头，任务结束。")
                break
            # 5. 数据处理与保存
            df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
            df.dropna(subset=['trade_time'], inplace=True)
            if df.empty:
                print(f"[{stock_code}] 数据清洗后为空，跳过此批次。")
                reference_date = start_date_obj - timedelta(days=1)
                continue
            df['stock'] = stock
            df['trade_time'] = pd.to_datetime(df['trade_time']).dt.tz_localize('Asia/Shanghai').dt.tz_convert('UTC').dt.tz_localize(None)
            data_list = df[["stock", "trade_time", "close", "open", "high", "low", "vol", "amount"]].to_dict('records')
            result_dict = await self._save_all_to_db_native_upsert(
                model_class=model_class,
                data_list=data_list,
                unique_fields=['stock', 'trade_time']
            )
            saved_count = result_dict.get("创建/更新成功", 0)
            total_saved_count += saved_count
            # 6. 更新下一次追溯的参考日期
            reference_date = start_date_obj - timedelta(days=1)
            # 7. 礼貌性等待
            await asyncio.sleep(0.2)
        return total_saved_count

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
        model_class = get_minute_data_model_by_code_and_timelevel(stock_code, time_level)
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
            model_class = get_minute_data_model_by_code_and_timelevel(stock_code, time_level)
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
        【V2.2 - 统一输出UTC aware datetime版】
        - 核心修复: 统一将从数据库获取的分钟线数据转换为 UTC aware datetime。
        """
        try:
            time_level_digit = "".join(filter(str.isdigit, time_level))
            if not time_level_digit:
                logger.error(f"无法从 time_level='{time_level}' 中提取出有效的数字级别。")
                return None
            model_class = get_minute_data_model_by_code_and_timelevel(stock_code, time_level_digit)
            if not model_class:
                logger.error(f"未能找到股票 {stock_code} 对应的 {time_level} K线模型。")
                return None
            kline_queryset = model_class.objects.filter(
                stock__stock_code=stock_code,
                trade_time__range=(start_dt, end_dt)
            ).order_by('trade_time')
            kline_values = await sync_to_async(list)(kline_queryset.values(
                'trade_time', 'open', 'high', 'low', 'close', 'vol', 'amount'
            ))
            if not kline_values:
                # logger.warning(f"在数据库 {model_class._meta.db_table} 中未找到 {stock_code} 在 {start_dt} 到 {end_dt} 之间的 {time_level} K线数据。")
                return None
            df = pd.DataFrame.from_records(kline_values)
            # 从数据库取出的时间已经是UTC，直接设置为索引，并确保是 aware UTC
            df['trade_time'] = pd.to_datetime(df['trade_time'], utc=True)
            df.set_index('trade_time', inplace=True)
            df.rename(columns={'vol': 'volume', 'amount': 'turnover_value'}, inplace=True)
            logger.debug(f"成功从数据库获取 {len(df)} 条 {time_level} K线 for {stock_code}")
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
            model_class = get_minute_data_model_by_code_and_timelevel(stock_code, time_level)
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
    async def save_weekly_time_trade(self, trade_date: date = None, start_date: date=None) -> None:
        """
        【V2.0 向量化与N+1优化版】保存股票的周线交易数据
        - 核心优化:
          1. 【消除N+1查询】不再预加载全市场股票，而是在获取API数据后，根据返回的股票代码一次性批量查询所需股票信息。
          2. 【向量化处理】使用Pandas的向量化操作替代了原有的 `itertuples()` 循环，大幅提升了数据处理效率。
          3. 保留了原有的正确分页逻辑和分批入库机制。
        """
        trade_date_str = trade_date.strftime('%Y%m%d') if trade_date else ""
        start_date_str = start_date.strftime('%Y%m%d') if start_date else "19900101"
        all_data_dicts = []
        offset = 0
        limit = 6000
        while True:
            # 拉取数据
            df = self.ts_pro.stk_week_month_adj(**{
                "ts_code": "", "trade_date": trade_date_str, "start_date": start_date_str, "end_date": "", "freq": "week", "limit": limit, "offset": offset
            }, fields=[
                "ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"
            ])
            if df is None or df.empty:
                break
            # --- 开始向量化处理 ---
            # 1. 数据清洗
            df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
            df.dropna(subset=['ts_code', 'trade_date'], inplace=True)
            if df.empty:
                if len(df) < limit: break
                offset += limit
                continue
            # 2. 批量获取关联对象 (消除N+1查询)
            unique_ts_codes = df['ts_code'].unique().tolist()
            stock_map = await self.stock_basic_dao.get_stocks_by_codes(unique_ts_codes)
            # 3. 向量化映射、转换和选择
            df['stock'] = df['ts_code'].map(stock_map)
            df.dropna(subset=['stock'], inplace=True)
            if df.empty:
                if len(df) < limit: break
                offset += limit
                continue
            df['trade_time'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.date
            df.rename(columns={'pct_chg': 'pct_change'}, inplace=True)
            # 对所有数值列进行一次性转换
            numeric_cols = ['open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_change', 'vol', 'amount']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            model_cols = ['stock', 'trade_time', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_change', 'vol', 'amount']
            final_df = df[model_cols]
            # 4. 转换为字典列表，并将NaN转为None
            all_data_dicts.extend(final_df.where(pd.notnull(final_df), None).to_dict('records'))
            # --- 向量化处理结束 ---
            # 检查是否达到批处理大小
            if len(all_data_dicts) >= BATCH_SAVE_SIZE:
                await self._save_all_to_db_native_upsert(
                    model_class=StockWeeklyData,
                    data_list=all_data_dicts,
                    unique_fields=['stock', 'trade_time']
                )
                logger.info(f"完成一批周线数据保存，数量：{len(all_data_dicts)}")
                all_data_dicts = []
            if len(df) < limit:
                break
            offset += limit
            await asyncio.sleep(0.5) # 使用异步sleep
        # 保存最后一批剩余数据
        if all_data_dicts:
            await self._save_all_to_db_native_upsert(
                model_class=StockWeeklyData,
                data_list=all_data_dicts,
                unique_fields=['stock', 'trade_time']
            )
            logger.info(f"完成最后一批周线数据保存，数量：{len(all_data_dicts)}")
        print(f"周线数据处理完成。")

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
    async def save_monthly_time_trade(self, start_date: str = "1990-01-01") -> List:
        """
        【V3.1 向量化重构版】保存股票的月线交易数据 (前复权)。
        - 核心优化:
          1. 【消除N+1查询】不再预加载全市场股票，而是在获取API数据后，根据返回的股票代码一次性批量查询所需股票信息。
          2. 【向量化处理】使用Pandas的向量化操作替代了原有的数据处理循环，大幅提升了数据清洗、转换和格式化的效率。
          3. 【类型安全】在向量化层面统一处理数值和日期类型转换，代码更简洁、健壮。
        """
        all_data_to_save = []
        offset = 0
        limit = 6000
        while True:
            try:
                df = self.ts_pro.stk_week_month_adj(
                    start_date=start_date, freq="month", limit=limit, offset=offset
                )
                if df is None: df = pd.DataFrame()
            except Exception as e:
                logger.error(f"Tushare API调用失败 (offset: {offset}): {e}", exc_info=True)
                await asyncio.sleep(5)
                break
            if df.empty:
                logger.info("Tushare未返回更多数据，拉取完成。")
                break
            # --- 开始向量化处理 ---
            # 1. 数据清洗
            df.replace(['', 'null', 'None', 'nan', 'NaN'], np.nan, inplace=True)
            df.dropna(subset=['ts_code', 'trade_date'], inplace=True)
            if df.empty:
                if len(df) < limit: break
                offset += limit
                continue
            # 2. 批量获取关联对象 (消除N+1查询)
            unique_ts_codes = df['ts_code'].unique().tolist()
            stock_map = await self.stock_basic_dao.get_stocks_by_codes(unique_ts_codes)
            # 3. 向量化映射、转换和选择
            df['stock'] = df['ts_code'].map(stock_map)
            df.dropna(subset=['stock'], inplace=True)
            if df.empty:
                if len(df) < limit: break
                offset += limit
                continue
            df['trade_time'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.date
            df.rename(columns={'pct_chg': 'pct_change', 'open_qfq': 'open', 'high_qfq': 'high', 'low_qfq': 'low', 'close_qfq': 'close'}, inplace=True)
            # 4. 向量化数值类型转换
            numeric_cols = ['open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_change', 'vol', 'amount']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            # Tushare的amount单位是千元，乘以1000变为元
            if 'amount' in df.columns:
                df['amount'] = df['amount'] * 1000
            # 5. 选择最终列并转换为字典列表
            model_cols = ['stock', 'trade_time', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_change', 'vol', 'amount']
            final_df = df[[col for col in model_cols if col in df.columns]]
            records = final_df.where(pd.notnull(final_df), None).to_dict('records')
            all_data_to_save.extend(records)
            # --- 向量化处理结束 ---
            # 分批保存到数据库
            if len(all_data_to_save) >= BATCH_SAVE_SIZE:
                await self._save_all_to_db_native_upsert(
                    model_class=StockMonthlyData,
                    data_list=all_data_to_save,
                    unique_fields=['stock', 'trade_time']
                )
                all_data_to_save = []
            # 准备下一次循环
            await asyncio.sleep(0.6)
            if len(df) < limit:
                break
            offset += len(df)
        # 保存最后一批剩余数据
        if all_data_to_save:
            logger.info(f"正在保存最后一批剩余的 {len(all_data_to_save)} 条月线数据...")
            await self._save_all_to_db_native_upsert(
                model_class=StockMonthlyData,
                data_list=all_data_to_save,
                unique_fields=['stock', 'trade_time']
            )
        logger.info(f"股票的月线数据保存任务全部完成。")
        return all_data_to_save # 返回最后一次保存或空列表

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
        【V3.1 - 向量化重构与字段补全版】保存所有股票的日线基本信息
        - 核心重构: 废弃了原有的循环 + 外部格式化函数(set_stock_daily_basic_data)的低效模式，
                    改为采用Pandas向量化操作，直接在方法内部完成数据清洗、关联和准备，
                    与 save_stock_daily_basic_history_by_stock_codes 方法的实现保持一致，
                    显著提升了性能和代码可读性。
        - 核心修复: 1. 在API请求的字段列表中补全了 'limit_status' 字段。
                    2. 新的向量化处理逻辑确保了 'dv_ratio', 'dv_ttm', 'limit_status' 
                       能够被正确处理并包含在最终存入数据库的数据中。
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
            # 【代码修改】在请求字段列表中增加 'limit_status'
            fields_to_fetch = [
                "ts_code", "trade_date", "close", "turnover_rate", "turnover_rate_f", "volume_ratio", "pe", "pe_ttm", "pb", "ps",
                "ps_ttm", "dv_ratio", "dv_ttm", "total_share", "float_share", "free_share", "total_mv", "circ_mv", "limit_status"
            ]
            # 2. 一次性从Tushare拉取所有数据
            print(f"调试: 正在从Tushare API拉取日线基本数据，参数: {params}")
            df = self.ts_pro.daily_basic(**params, fields=fields_to_fetch)
            if df.empty:
                logger.info("Tushare API没有返回任何日线基本数据，任务提前结束。")
                print("调试: Tushare API返回空数据帧，任务结束。")
                return {"status": "success", "message": "No data returned from API.", "创建/更新成功": 0}
            # 3. 数据清洗：将API返回的空值统一替换为np.nan，以便后续处理
            df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
            # 4. 批量获取所有相关的股票信息
            unique_ts_codes = df['ts_code'].unique().tolist()
            print(f"调试: 从API获取了 {len(df)} 条数据，涉及 {len(unique_ts_codes)} 个独立股票代码。")
            # 这是解决N+1问题的关键，将多次查询合并为一次
            stock_map = await self.stock_basic_dao.get_stocks_by_codes(unique_ts_codes)
            print(f"调试: 批量从数据库获取了 {len(stock_map)} 个股票对象。")
            # 5. 【新增-代码行】使用向量化操作准备批量写入的数据
            # 5.1 向量化映射stock对象，根除N+1查询
            df['stock'] = df['ts_code'].map(stock_map)
            # 5.2 丢弃无效数据（如在数据库中找不到对应stock的行）
            df.dropna(subset=['trade_date', 'stock'], inplace=True)
            # 5.3 向量化转换日期
            df['trade_time'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.date
            # 5.4 显式地选择所有模型需要的列，确保新增字段被包含
            model_columns = [
                "stock", "trade_time", "close", "turnover_rate", "turnover_rate_f", "volume_ratio", "pe", "pe_ttm", "pb", "ps",
                "ps_ttm", "dv_ratio", "dv_ttm", "total_share", "float_share", "free_share", "total_mv", "circ_mv", "limit_status"
            ]
            final_df = df[model_columns]
            # 5.5 将整理好的DataFrame转换为字典列表，并将NaN转为None以适配数据库
            data_dicts_to_save = final_df.where(pd.notnull(final_df), None).to_dict('records')
            # 6. 批量写入数据库
            saved_count = 0
            if data_dicts_to_save:
                print(f"调试: 准备批量保存 {len(data_dicts_to_save)} 条数据到数据库...")
                result = await self._save_all_to_db_native_upsert(
                    model_class=StockDailyBasic,
                    data_list=data_dicts_to_save,
                    unique_fields=['stock', 'trade_time']
                )
                saved_count = result.get("创建/更新成功", 0)
                logger.info(f"成功批量保存 {saved_count} 条股票日线基本数据。")
                print(f"调试: 成功保存 {saved_count} 条数据。")
            else:
                logger.info("没有需要保存到数据库的数据。")
                print("调试: 没有需要保存的数据。")
            return {"status": "success", "message": f"Processed {saved_count} records.", "创建/更新成功": saved_count}
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
        # print(f"正在批量预加载 {len(stock_codes)} 只股票的基础信息...")
        stock_map = await self.stock_basic_dao.get_stocks_by_codes(stock_codes)
        if not stock_map:
            logger.warning(f"根据提供的代码列表，未能从数据库中找到任何股票信息。")
            return []
        # print("股票信息预加载完成。")
        stock_codes_str = ",".join(stock_codes)
        trade_date_str = trade_date.strftime('%Y%m%d') if trade_date else ""
        start_date_str = start_date.strftime('%Y%m%d') if start_date else "20200101"
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
        # if all_data_for_cache:
            # print(f"正在批量写入 {len(all_data_for_cache)} 条数据到缓存...")
            # 假设您有一个支持批量设置的缓存方法
            # await self.cache_set.stock_day_basic_info_batch(all_data_for_cache)
            # logger.info(f"完成 {len(all_data_for_cache)} 条日线基本信息的批量缓存。")
        # 2. 批量写入数据库
        if all_data_dicts_for_db:
            # print(f"正在批量保存 {len(all_data_dicts_for_db)} 条数据到数据库...")
            # 注意：unique_fields需要使用ORM模型中的字段名，这里假设是'stock'和'trade_time'
            # 如果模型中关联字段名是 stock_id，则应为 ['stock_id', 'trade_time']
            result = await self._save_all_to_db_native_upsert(
                model_class=StockDailyBasic,
                data_list=all_data_dicts_for_db,
                unique_fields=['stock', 'trade_time']
            )
            # logger.info(f"完成 {len(all_data_dicts_for_db)} 条日线基本信息的批量保存。")
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

    async def get_stock_daily_basic_by_date(self, stock_code: str, trade_date: date) -> Optional[StockDailyBasic]:
        """
        获取股票的日线基本信息
        修改思路：
        1. 移除导致 AttributeError 的 self.cache_key.today_basic_info 调用。
        2. 修正数据库查询字段：stock_code -> stock__stock_code, trade_date -> trade_time。
        3. 使用 Django 5 的异步 ORM 方法 afirst() 替代同步的 filter().first()，避免阻塞事件循环。
        """
        try:
            # 直接从数据库查询，确保数据的准确性
            # 使用 afirst() 进行高效异步查询
            stock_daily_basic = await StockDailyBasic.objects.filter(
                stock__stock_code=stock_code, 
                trade_time=trade_date
            ).afirst()
            return stock_daily_basic
        except Exception as e:
            logger.error(f"获取股票 {stock_code} 在 {trade_date} 的日线基本信息失败: {e}")
            return None

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

    @with_rate_limit(name='api_cyq_perf') #: 添加速率限制装饰器
    async def save_cyq_perf_for_stock(self, stock, start_date: date = None, end_date: date = None, *, limiter) -> None: # 修改方法签名，接收limiter
        """
        获取并保存单个股票的历史筹码及胜率数据。
        此方法被并行的Celery任务调用。
        已适配分布式速率限制。
        """
        # print(f"DAO: 开始获取 {stock.stock_code} 的筹码及胜率数据...")
        start_date_str = start_date.strftime('%Y%m%d') if start_date else "20160101"
        end_date_str = end_date.strftime('%Y%m%d') if end_date else ""
        offset = 0
        limit = 5000 # 对于单个股票，可以设置一个较大的limit
        all_data_for_stock = []
        while True:
            try:
                #: 在调用API前获取速率许可
                while not await limiter.acquire():
                    print(f"PID[{os.getpid()}] API[api_cyq_perf] 速率超限，等待10秒后重试... (股票: {stock.stock_code})")
                    await asyncio.sleep(10)
                # print(f"PID[{os.getpid()}] API[api_cyq_perf] 成功获取许可，正在为 {stock.stock_code} (offset={offset}) 调用API...")
                df = self.ts_pro.cyq_perf(**{
                    "ts_code": stock.stock_code, "start_date": start_date_str, "end_date": end_date_str, "limit": limit, "offset": offset
                }, fields=[
                    "ts_code", "trade_date", "his_low", "his_high", "cost_5pct", "cost_15pct", "cost_50pct", "cost_85pct",
                    "cost_95pct", "weight_avg", "winner_rate"
                ])
            except Exception as e:
                logger.error(f"Tushare API调用失败 (cyq_perf, ts_code={stock.stock_code}): {e}")
                break
            if df.empty:
                break
            all_data_for_stock.append(df)
            if len(df) < limit:
                break
            offset += limit
        if not all_data_for_stock:
            print(f"DAO: 未获取到 {stock.stock_code} 的任何筹码及胜率数据。")
            return
        # --- 数据处理和保存逻辑保持不变 ---
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
        # print(f"DAO: 准备为 {stock.stock_code} 保存 {len(data_list)} 条筹码及胜率数据...")
        return await self._save_all_to_db_native_upsert(
            model_class=StockCyqPerf,
            data_list=data_list,
            unique_fields=['stock', 'trade_time']
        )

    async def get_cyq_chips_history(self, stock_code: str) -> QuerySet:
        """
        获取股票的每日筹码分布历史数据 (已修改为直接查询分表数据库)
        1. [移除] 删除了所有Redis缓存查询逻辑。
        2. 根据股票代码动态选择正确的分表Model进行查询。
        3. [修正] 修正了ORM查询条件以正确通过外键关联进行过滤。
        """
        # 第一步：根据股票代码动态获取对应的分表Model
        target_model = get_cyq_chips_model_by_code(stock_code)
        print(f"DAO: 正在为股票 {stock_code} 从数据表 {target_model.__name__} 查询筹码分布历史。")
        # 第二步：直接使用动态获取的Model进行数据库查询
        # 注意：
        # 1. 使用 target_model.objects 进行查询。
        # 2. 过滤器使用 'stock__stock_code' 来通过外键关联查询。
        # 3. 排序字段使用模型中定义的 'trade_time'。
        stock_cyq_chips_queryset = target_model.objects.filter(
            stock__stock_code=stock_code
        ).order_by('-trade_time')[:self.cache_limit] # 保留了原有的查询数量限制
        return stock_cyq_chips_queryset

    # 每日筹码分布
    async def save_all_cyq_chips_history(self, trade_date: date=None, start_date: date=None, end_date: date=None) -> None:
        """
        保存全市场股票的每日筹码分布数据 (终极优化版)
        1. 引入分表逻辑，将数据按板块存入不同数据表。
        2. 引入10万行追溯逻辑，确保获取全量历史数据。
        3. [优化] 使用异步 asyncio.sleep 替代同步 time.sleep，防止阻塞事件循环。
        4. [重构] 重构批处理机制，以适应分表场景。
        5. 引入 aiolimiter 进行专业、高效的API限流，替换固定的 asyncio.sleep。
        """
        # --- 日期字符串格式化 (无变化) ---
        trade_date_str = trade_date.strftime('%Y%m%d') if trade_date else ""
        start_date_str = start_date.strftime('%Y%m%d') if start_date else "20200101"
        initial_end_date_str = end_date.strftime('%Y%m%d') if end_date else ""
        all_stocks = await self.stock_basic_dao.get_stock_list()
        if not all_stocks:
            logger.warning("股票基础信息列表为空，任务终止。")
            return
        # 键是Model类，值是待保存的数据列表
        batched_data_by_model = {}
        total_stocks = len(all_stocks)
        for i, stock in enumerate(all_stocks):
            print(f"【cyq_chips每日筹码分布】开始处理第 {i+1}/{total_stocks} 只股票: {stock.stock_code} - {stock.stock_name}")
            # 移植10万行追溯逻辑
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
                        # 使用 aiolimiter 作为异步上下文管理器来包裹API调用
                        # 这是更专业、高效和健壮的限流方式
                        async with self.limiter:
                            print(f"正在请求 {stock.stock_code} 的数据, offset={offset}...")
                            df = self.ts_pro.cyq_chips(**{
                                "ts_code": stock.stock_code, "trade_date": trade_date_str,
                                "start_date": start_date_str, "end_date": current_end_date_str,
                                "limit": limit, "offset": offset
                            }, fields=["ts_code", "trade_date", "price", "percent"])
                        # [删除] 不再需要手动的、不精确的 sleep
                        # await asyncio.sleep(0.4)
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
                    # 分表批处理核心逻辑
                    if data_dicts_for_stock:
                        # 1. 获取当前股票对应的正确Model
                        target_model = get_cyq_chips_model_by_code(stock.stock_code)
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
        # --- 在所有股票处理完毕后，保存所有分表中剩余的数据 ---
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

    @with_rate_limit(name='api_cyq_chips')
    async def save_cyq_chips_for_stock(self, stock: StockInfo, start_date: date = None, end_date: date = None, *, limiter) -> None:
        """
        【V3.0 统一逻辑修正版】保存单只股票的每日筹码分布数据。
        - 核心修正: 采用统一的分页和追溯逻辑，并通过动态构建参数字典来精确调用API。
                    这解决了因错误处理可选日期参数（None vs "" vs 默认值）而导致的API调用失败问题。
                    此版本对所有场景（范围查询、历史追溯）都同样健壮。
        :param stock: StockInfo 模型实例。
        :param start_date: 开始日期，可选。
        :param end_date: 结束日期，可选。
        :param limiter: 由 @with_rate_limit 装饰器注入的 DistributedRateLimiter 实例。
        """
        # 正确处理可选日期参数，将None转换为None，而不是默认值或空字符串
        start_date_str = start_date.strftime('%Y%m%d') if start_date else None
        current_end_date_str = end_date.strftime('%Y%m%d') if end_date else None
        all_dfs_for_stock = []
        # 回归统一的、健壮的内外双循环分页逻辑
        # 外层循环：处理10万行限制，通过调整 end_date 实现追溯
        while True:
            offset = 0
            limit = 6000 # Tushare推荐的单次最大limit
            dfs_for_this_cycle = []
            limit_hit = False
            # 内层循环：处理标准分页
            while True:
                if offset >= 100000:
                    logger.warning(f"股票 {stock.stock_code} 的每日筹码分布 offset已达10万，将进行追溯抓取。")
                    limit_hit = True
                    break
                try:
                    while not await limiter.acquire():
                        print(f"PID[{os.getpid()}] API[api_cyq_chips] 速率超限，等待10秒后重试... (股票: {stock.stock_code})")
                        await asyncio.sleep(10)
                    # 动态构建API参数，只包含有值的参数，这是最关键的修正
                    api_params = {
                        "ts_code": stock.stock_code,
                        "limit": limit,
                        "offset": offset
                    }
                    if start_date_str:
                        api_params['start_date'] = start_date_str
                    if current_end_date_str:
                        api_params['end_date'] = current_end_date_str
                    # print(f"调试: DAO准备调用Tushare API [cyq_chips]，动态参数: {api_params}")
                    df = self.ts_pro.cyq_chips(**api_params, fields=["ts_code", "trade_date", "price", "percent"])
                except Exception as e:
                    logger.error(f"Tushare API调用失败 (cyq_chips, ts_code={stock.stock_code}): {e}", exc_info=True)
                    await asyncio.sleep(5)
                    df = pd.DataFrame()
                if df.empty:
                    break
                dfs_for_this_cycle.append(df)
                if len(df) < limit:
                    break # 如果返回的数据量小于请求量，说明是最后一页了
                offset += limit
            if not dfs_for_this_cycle:
                break # 如果当前追溯周期没有获取到任何数据，则结束整个过程
            all_dfs_for_stock.extend(dfs_for_this_cycle)
            if limit_hit:
                last_df_in_cycle = dfs_for_this_cycle[-1]
                last_trade_date = last_df_in_cycle['trade_date'].iloc[-1]
                current_end_date_str = last_trade_date
                print(f"DAO: {stock.stock_code} 触及10万行限制，下一轮将从 {current_end_date_str} 继续向前追溯。")
                continue # 继续外层循环，开始下一轮追溯
            else:
                break # 如果没有触及10万行限制，说明所有数据已获取完毕
        # --- 数据处理和保存部分保持不变 ---
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
        target_model = get_cyq_chips_model_by_code(stock.stock_code)
        # print(f"DAO: 准备为 {stock.stock_code} 保存 {len(data_list)} 条筹码分布数据到表 {target_model.__name__}...")
        await self._save_all_to_db_native_upsert(
            model_class=target_model,
            data_list=data_list,
            unique_fields=['stock', 'trade_time', 'price']
        )

    # 新增保存每日涨跌停价格的方法
    @with_rate_limit(name='api_stk_limit')
    async def save_stk_limit_history(self, trade_date: date = None, start_date: date = None, end_date: date = None, *, limiter) -> dict:
        """
        【V1.1 · 速率限制版】根据指定的日期或日期范围，获取并保存全市场股票的每日涨跌停价格。
        - 采用分表策略进行存储。
        - 使用向量化操作和批量写入，性能高效。
        - 集成了 'api_stk_limit' 速率限制，确保调用安全。
        """
        api_params = {}
        if trade_date:
            api_params['trade_date'] = trade_date.strftime('%Y%m%d')
        if start_date:
            api_params['start_date'] = start_date.strftime('%Y%m%d')
        if end_date:
            api_params['end_date'] = end_date.strftime('%Y%m%d')
        if not api_params:
            logger.warning("save_stk_limit_history: 必须提供 trade_date 或 start_date/end_date。")
            return {"status": "error", "message": "Date parameter is required."}
        try:
            # 在API调用前，使用注入的limiter进行速率检查和等待
            while not await limiter.acquire():
                print(f"PID[{os.getpid()}] API[api_stk_limit] 速率超限，等待10秒后重试...")
                await asyncio.sleep(10)
            print(f"调试: DAO准备调用Tushare API [stk_limit]，参数: {api_params}")
            df = self.ts_pro.stk_limit(**api_params)
            if df.empty:
                logger.info("Tushare API [stk_limit] 没有返回任何数据，任务结束。")
                return {"status": "success", "message": "No data returned from API.", "saved_count": 0}
            # --- 数据处理与分表 ---
            df.rename(columns={'ts_code': 'stock_code'}, inplace=True)
            df['trade_time'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.date
            # 批量获取股票对象
            unique_codes = df['stock_code'].unique().tolist()
            stock_map = await self.stock_basic_dao.get_stocks_by_codes(unique_codes)
            df['stock'] = df['stock_code'].map(stock_map)
            df.dropna(subset=['stock'], inplace=True)
            # 按模型分发数据
            data_by_model = defaultdict(list)
            for record in df.to_dict('records'):
                model_class = get_stk_limit_model_by_code(record['stock_code'])
                if model_class:
                    data_by_model[model_class].append({
                        'stock': record['stock'],
                        'trade_time': record['trade_time'],
                        'pre_close': record.get('pre_close'),
                        'up_limit': record['up_limit'],
                        'down_limit': record['down_limit'],
                    })
            # 批量保存
            save_tasks = []
            for model_class, data_list in data_by_model.items():
                task = self._save_all_to_db_native_upsert(
                    model_class=model_class,
                    data_list=data_list,
                    unique_fields=['stock', 'trade_time']
                )
                save_tasks.append(task)
            results = await asyncio.gather(*save_tasks)
            total_saved = sum(res.get("创建/更新成功", 0) for res in results if isinstance(res, dict))
            return {"status": "success", "message": f"Saved {total_saved} stock limit price records.", "saved_count": total_saved}
        except Exception as e:
            logger.error(f"保存每日涨跌停价格时发生错误: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    # 新增根据股票代码和交易日期获取涨跌停价的方法
    async def get_price_limit_by_date(self, stock_code: str, trade_date: date) -> Optional[Dict]:
        """
        【V1.0 · 精确制导数据接口】
        根据股票代码和交易日期，从正确的分表中获取单日的涨跌停价格数据。
        Args:
            stock_code (str): 股票代码, 例如 '000001.SZ'。
            trade_date (date): 交易日期 (datetime.date 对象)。
        Returns:
            Optional[Dict]: 包含涨跌停价信息的字典，如果未找到则返回 None。
                            例如: {'up_limit': 15.06, 'down_limit': 12.32, 'pre_close': 13.69}
        """
        try:
            # 步骤1: 使用辅助函数动态确定需要查询的模型
            model_class = get_stk_limit_model_by_code(stock_code)
            if not model_class:
                logger.warning(f"[DAO] get_price_limit_by_date: 未能为股票 {stock_code} 找到对应的涨跌停价格模型。")
                return None
            # 步骤2: 使用 Django 5 的原生异步 ORM 方法 .afirst() 高效查询
            # .afirst() 在找到第一条匹配记录后立即返回，或在无匹配时返回 None，非常适合此场景。
            limit_data_obj = await model_class.objects.filter(
                stock__stock_code=stock_code,
                trade_time=trade_date
            ).afirst()
            # 步骤3: 处理查询结果
            if not limit_data_obj:
                # 如果数据库中没有当天的记录，则明确返回 None
                return None
            # 步骤4: 将查询到的模型对象转换为干净的字典格式返回，与策略层解耦
            return {
                'up_limit': limit_data_obj.up_limit,
                'down_limit': limit_data_obj.down_limit,
                'pre_close': limit_data_obj.pre_close,
            }
        except Exception as e:
            logger.error(f"[DAO] get_price_limit_by_date: 查询 {stock_code} 在 {trade_date} 的涨跌停价时发生错误: {e}", exc_info=True)
            return None

    async def get_price_limit_data(self, stock_code: str, end_date: Optional[datetime.date], limit: int) -> pd.DataFrame:
        """
        【V1.0 · 服务层专用接口】
        获取指定股票在截止日期前的N条涨跌停价格历史数据。
        Args:
            stock_code (str): 股票代码。
            end_date (Optional[datetime.date]): 查询的截止日期。如果为None，则从最新日期开始。
            limit (int): 需要获取的记录数量。
        Returns:
            pd.DataFrame: 包含涨跌停价历史数据的DataFrame，索引为'trade_time'。
        """
        try:
            # 步骤1: 动态确定查询模型
            model_class = get_stk_limit_model_by_code(stock_code)
            if not model_class:
                logger.warning(f"[DAO] get_price_limit_data: 未能为股票 {stock_code} 找到模型。")
                return pd.DataFrame()
            # 步骤2: 构建查询
            queryset = model_class.objects.filter(stock__stock_code=stock_code)
            if end_date:
                queryset = queryset.filter(trade_time__lte=end_date)
            # 步骤3: 执行查询并转换为DataFrame
            # 使用 .values() 直接获取字典列表，比获取模型实例更高效
            data_list = await sync_to_async(list)(
                queryset.order_by('-trade_time')[:limit].values(
                    'trade_time', 'up_limit', 'down_limit', 'pre_close'
                )
            )
            if not data_list:
                return pd.DataFrame()
            # 步骤4: 格式化为标准DataFrame
            df = pd.DataFrame(data_list)
            df['trade_time'] = pd.to_datetime(df['trade_time'], utc=True)
            df.set_index('trade_time', inplace=True)
            df.sort_index(inplace=True) # 确保返回的DataFrame是按时间升序排列的
            return df
        except Exception as e:
            logger.error(f"[DAO] get_price_limit_data: 查询 {stock_code} 涨跌停价时出错: {e}", exc_info=True)
            return pd.DataFrame()











