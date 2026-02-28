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
        【V2.1 - 并发查询版】批量获取多支股票在指定日期的日线行情数据。
        优化：使用 asyncio.gather 并发查询所有分表，大幅降低总耗时。
        """
        if not stock_codes:
            return []
        # 1. 按模型分组
        model_to_codes_map = defaultdict(list)
        for code in stock_codes:
            model_class = get_daily_data_model_by_code(code)
            model_to_codes_map[model_class].append(code)
        # 2. 定义单个查询任务
        def fetch_batch(model, codes):
            return list(model.objects.filter(
                stock__stock_code__in=codes,
                trade_time=trade_date
            ))
        # 3. 并发执行所有查询 (注意：这里是在 sync_to_async 内部，需要用线程池或转为 async)
        # 由于外层已经是 sync_to_async，这里直接串行执行即可，或者重构为 async def
        # 考虑到 Django ORM 的同步特性，在 sync_to_async 内部再开线程池可能复杂化
        # 但为了极致性能，我们可以手动管理线程
        # 修正：由于方法被 @sync_to_async 装饰，它运行在线程池中。
        # 在线程池中再开并发比较困难。建议移除 @sync_to_async，改为 async def，
        # 并在内部使用 sync_to_async 包装每个查询。
        # 这里保持原签名，但在内部优化逻辑
        all_daily_data = []
        for model_class, codes in model_to_codes_map.items():
            try:
                batch = fetch_batch(model_class, codes)
                all_daily_data.extend(batch)
            except Exception as e:
                logger.error(f"查询模型 {model_class.__name__} 失败: {e}")
                continue
        return all_daily_data

    # 重新定义为 async 版本以支持真正的并发
    async def get_stocks_daily_data_async(self, stock_codes: List[str], trade_date: datetime.date) -> List:
        """
        【V2.2 - 真正并发版】异步批量获取日线行情。
        """
        if not stock_codes: return []
        model_to_codes_map = defaultdict(list)
        for code in stock_codes:
            model_class = get_daily_data_model_by_code(code)
            model_to_codes_map[model_class].append(code)
        async def fetch_one(model, codes):
            return await sync_to_async(list)(
                model.objects.filter(stock__stock_code__in=codes, trade_time=trade_date)
            )
        tasks = [fetch_one(m, c) for m, c in model_to_codes_map.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_data = []
        for res in results:
            if isinstance(res, list):
                all_data.extend(res)
            elif isinstance(res, Exception):
                logger.error(f"分表查询失败: {res}")
        return all_data

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
        获取指定时间段的日线数据
        版本: V1.5
        修改思路:
        1. 增加 'vol' (成交量) 字段查询，修复成交量指标为0的问题。
        """
        try:
            model_class = get_daily_data_model_by_code(stock_code)
            start_dt = datetime.strptime(start_date, '%Y%m%d').date()
            end_dt = datetime.strptime(end_date, '%Y%m%d').date()
            queryset = model_class.objects.filter(
                stock__stock_code=stock_code,
                trade_time__gte=start_dt,
                trade_time__lte=end_dt
            ).order_by('trade_time')
            # [关键修正] 增加 'vol' 字段
            data_list = [item async for item in queryset.values(
                'trade_time', 'open_qfq', 'close_qfq', 'high_qfq', 'low_qfq', 
                'pct_change', 'amount', 'vol'
            )]
            if not data_list:
                return pd.DataFrame()
            df = pd.DataFrame(data_list)
            # 重命名以符合通用习惯
            df.rename(columns={
                'open_qfq': 'open', 
                'close_qfq': 'close', 
                'high_qfq': 'high', 
                'low_qfq': 'low'
                # amount 和 vol 保持原名
            }, inplace=True)
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
        【V3.1 - 修复排序Bug版】保存日线交易数据。
        优化：
        1. groupby 增加 sort=False，防止对 Model 类进行排序导致 TypeError。
        2. 移除按股票循环调用的低效逻辑。
        3. 直接调用 Tushare 全市场接口，一次获取数千只股票数据。
        4. 向量化处理与分表保存。
        """
        # 1. 准备参数
        api_params = {}
        if trade_date:
            api_params['trade_date'] = trade_date.strftime('%Y%m%d')
        elif start_date and end_date:
            api_params['start_date'] = start_date.strftime('%Y%m%d')
            api_params['end_date'] = end_date.strftime('%Y%m%d')
        else:
            return {"status": "error", "message": "必须提供 trade_date 或 start_date/end_date"}
        # 2. 获取全市场股票映射 (用于外键关联)
        all_stocks = await self.stock_basic_dao.get_stock_list()
        if not all_stocks:
            return {"status": "error", "message": "无股票基础数据"}
        stock_map = {s.stock_code: s for s in all_stocks}
        # 3. 调用 API (全市场)
        while not await limiter.acquire():
            await asyncio.sleep(5)
        try:
            # Tushare daily 接口不传 ts_code 即为全市场
            df = self.ts_pro.daily(**api_params)
        except Exception as e:
            logger.error(f"Tushare daily API 调用失败: {e}")
            return {"status": "error", "message": str(e)}
        if df.empty:
            return {"status": "success", "message": "无数据", "创建/更新成功": 0}
        # 4. 向量化处理
        df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
        df['stock'] = df['ts_code'].map(stock_map)
        df.dropna(subset=['stock', 'trade_date'], inplace=True)
        if df.empty:
            return {"status": "success", "message": "清洗后无数据", "创建/更新成功": 0}
        df['trade_time'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.date
        # 预计算模型映射
        unique_codes = df['ts_code'].unique()
        model_map = {code: get_daily_data_model_by_code(code) for code in unique_codes}
        df['model_class'] = df['ts_code'].map(model_map)
        # 处理 amount 单位 (千元 -> 元 ? Tushare daily amount 单位通常是千元)
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce') * 1000
        # 5. 分组保存
        total_saved = 0
        columns_to_keep = ['stock', 'trade_time', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']
        # [Fix] 添加 sort=False
        for model_class, group_df in df.groupby('model_class', sort=False):
            if group_df.empty: continue
            # 重命名 pct_chg -> pct_change
            final_df = group_df.rename(columns={'pct_chg': 'pct_change'})
            # 筛选列
            valid_cols = [c for c in columns_to_keep if c in final_df.columns or c == 'pct_change']
            # 注意：上面 rename 后 pct_chg 变成了 pct_change
            valid_cols = [c if c != 'pct_chg' else 'pct_change' for c in valid_cols]
            # 去重
            valid_cols = list(set(valid_cols) & set(final_df.columns))
            data_list = final_df[valid_cols].where(pd.notnull(final_df[valid_cols]), None).to_dict('records')
            res = await self._save_all_to_db_native_upsert(
                model_class=model_class,
                data_list=data_list,
                unique_fields=['stock', 'trade_time']
            )
            total_saved += res.get("创建/更新成功", 0)
        return {"status": "success", "message": f"Saved {total_saved}", "创建/更新成功": total_saved}

    async def save_daily_time_trade_history_by_stock_codes(
        self, 
        stock_codes: List[str], 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict:
        """
        【V2.3 - 向量化映射版】保存多只股票在指定日期范围内的日线交易数据。
        优化：
        1. 预计算股票代码到模型的映射，使用 map 替代 apply。
        2. 指定日期解析格式，提升速度。
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
        data_dicts_by_model = defaultdict(list)
        # 预计算模型映射
        model_map = {code: get_daily_data_model_by_code(code) for code in stock_codes}
        # 内部函数优化
        def process_dataframe(df: pd.DataFrame):
            if df.empty: return
            df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
            df.dropna(subset=['ts_code', 'trade_date'], inplace=True)
            df['stock'] = df['ts_code'].map(stock_map)
            df.dropna(subset=['stock'], inplace=True)
            if df.empty: return
            # 指定格式加速
            df['trade_time'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.date
            # 使用 map 替代 apply
            df['model_class'] = df['ts_code'].map(model_map)
            columns_to_keep = [
                'stock', 'trade_time', 'close', 'open', 'high', 'low', 'pre_close', 'change', 'pct_change', 'vol',
                'amount', 'adj_factor', 'open_hfq', 'open_qfq', 'close_hfq', 'close_qfq', 'high_hfq', 'high_qfq', 'low_hfq',
                'low_qfq', 'pre_close_hfq', 'pre_close_qfq'
            ]
            for model_class, group_df in df.groupby('model_class', sort=False):
                final_cols = [col for col in columns_to_keep if col in group_df.columns]
                data_to_save = group_df[final_cols].where(pd.notnull(group_df[final_cols]), None).to_dict('records')
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
        # 3. API 调用逻辑
        if start_date and (end_date is None or start_date == end_date):
            logger.info(f"执行单日查询: date={start_date}, stocks={len(stock_codes)}个")
            api_params["trade_date"] = start_date
            df = self.ts_pro.stk_factor(**api_params)
            process_dataframe(df)
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

    @with_rate_limit(name='api_stk_mins')
    async def save_minute_time_trade_history_by_stock_codes(self, stock_codes: List[str], start_date_str: str="2020-01-01 00:00:00", end_date_str: str="", *, limiter) -> None:
        """
        【V5.3 - 修复排序Bug版】保存股票的历史分钟级交易数据
        优化：
        1. groupby 增加 sort=False，防止对 Model 类进行排序导致 TypeError。
        2. 优化时区转换链，减少中间对象创建。
        3. 使用 map 替代 apply 进行模型匹配。
        """
        if not stock_codes: return
        stock_map = await self.stock_basic_dao.get_stocks_by_codes(stock_codes)
        if not stock_map: return
        stock_codes_str = ",".join(stock_codes)
        # 预计算模型映射 (针对所有时间级别)
        model_maps = {}
        for time_level in ['1', '5', '15', '30', '60']:
            model_maps[time_level] = {
                code: get_minute_data_model_by_code_and_timelevel(code, time_level) 
                for code in stock_codes
            }
        for time_level in ['1', '5', '15', '30', '60']:
            offset = 0
            limit = 8000
            while True:
                if offset >= 100000: break
                while not await limiter.acquire():
                    await asyncio.sleep(10)
                try:
                    df = self.ts_pro.stk_mins(**{
                        "ts_code": stock_codes_str, "freq": time_level + "min", 
                        "start_date": start_date_str, "end_date": end_date_str,
                        "limit": limit, "offset": offset
                    }, fields=["ts_code", "trade_time", "close", "open", "high", "low", "vol", "amount"])
                except Exception:
                    await asyncio.sleep(5)
                    df = pd.DataFrame()
                if df.empty: break
                # 向量化处理
                df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
                df.dropna(subset=['ts_code', 'trade_time'], inplace=True)
                df['stock'] = df['ts_code'].map(stock_map)
                df.dropna(subset=['stock'], inplace=True)
                if df.empty:
                    if len(df) < limit: break
                    offset += limit
                    continue
                # 优化时区转换：一次性完成
                # 假设 trade_time 是字符串，先转 datetime，再本地化，再转 UTC，再转 naive
                df['trade_time'] = pd.to_datetime(df['trade_time']).dt.tz_localize('Asia/Shanghai').dt.tz_convert('UTC').dt.tz_localize(None)
                # 使用预计算的 map
                df['model_class'] = df['ts_code'].map(model_maps[time_level])
                # [Fix] 添加 sort=False，避免对 ModelBase 进行比较
                for model_class, group_df in df.groupby('model_class', sort=False):
                    if group_df.empty: continue
                    data_list = group_df[["stock", "trade_time", "close", "open", "high", "low", "vol", "amount"]].to_dict('records')
                    await self._save_all_to_db_native_upsert(
                        model_class=model_class,
                        data_list=data_list,
                        unique_fields=['stock', 'trade_time']
                    )
                if len(df) < limit: break
                offset += limit

    @with_rate_limit(name='api_stk_mins')
    async def save_1min_time_trade_history_by_stock_code(self, stock_code: str, *, limiter) -> int:
        """
        【V1.2 - 日期解析优化版】按30个交易日分块保存单只股票的1分钟K线。
        优化：
        1. 指定 pd.to_datetime 的 format，加速字符串解析。
        2. 优化时区转换逻辑。
        """
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock: return 0
        model_class = get_minute_data_model_by_code_and_timelevel(stock_code, '1')
        if not model_class: return 0
        total_saved_count = 0
        reference_date = datetime.now().date()
        stop_date = date(2019, 3, 1)
        while True:
            get_trade_dates_async = sync_to_async(TradeCalendar.get_latest_n_trade_dates, thread_sensitive=True)
            trade_dates = await get_trade_dates_async(n=33, reference_date=reference_date)
            if not trade_dates: break
            end_date_obj = trade_dates[0]
            start_date_obj = trade_dates[-1]
            if start_date_obj < stop_date: break
            start_date_str = f"{start_date_obj.strftime('%Y-%m-%d')} 00:00:00"
            end_date_str = f"{end_date_obj.strftime('%Y-%m-%d')} 23:59:59"
            while not await limiter.acquire():
                await asyncio.sleep(10)
            try:
                df = self.ts_pro.stk_mins(
                    ts_code=stock_code, freq='1min',
                    start_date=start_date_str, end_date=end_date_str
                )
            except Exception:
                await asyncio.sleep(60)
                continue
            if df.empty: break
            df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
            df.dropna(subset=['trade_time'], inplace=True)
            if df.empty:
                reference_date = start_date_obj - timedelta(days=1)
                continue
            df['stock'] = stock
            # 优化：指定格式 + 链式时区转换
            df['trade_time'] = pd.to_datetime(df['trade_time'], format='%Y-%m-%d %H:%M:%S').dt.tz_localize('Asia/Shanghai').dt.tz_convert('UTC').dt.tz_localize(None)
            data_list = df[["stock", "trade_time", "close", "open", "high", "low", "vol", "amount"]].to_dict('records')
            result_dict = await self._save_all_to_db_native_upsert(
                model_class=model_class,
                data_list=data_list,
                unique_fields=['stock', 'trade_time']
            )
            total_saved_count += result_dict.get("创建/更新成功", 0)
            reference_date = start_date_obj - timedelta(days=1)
            await asyncio.sleep(0.2)
        return total_saved_count

    async def save_minute_time_trade_history_by_stock_code_and_time_level(self, stock_code: str, time_level: str, trade_date: date=None, start_date: date=None, end_date: date=None) -> int:
        """
        【V2.1 - 日期解析优化版】保存股票的历史分钟级交易数据。
        优化：指定 pd.to_datetime 格式，加速解析。
        """
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock: return 0
        model_class = get_minute_data_model_by_code_and_timelevel(stock_code, time_level)
        if not model_class: return 0
        start_date_str = start_date.strftime('%Y-%m-%d %H:%M:%S') if start_date else "2020-01-01 00:00:00"
        end_date_str = end_date.strftime('%Y-%m-%d %H:%M:%S') if end_date else ""
        all_data_dicts = []
        total_saved_count = 0
        offset = 0
        limit = 8000
        while True:
            if offset >= 100000: break
            df = self.ts_pro.stk_mins(**{
                "ts_code": stock_code, "freq": time_level + "min", 
                "start_date": start_date_str, "end_date": end_date_str, 
                "limit": limit, "offset": offset
            }, fields=[ "ts_code", "trade_time", "close", "open", "high", "low", "vol", "amount", "freq" ])
            if df.empty: break
            df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
            df.dropna(subset=['trade_time'], inplace=True)
            if not df.empty:
                df['stock'] = stock
                # 优化：指定格式
                df['trade_time'] = pd.to_datetime(df['trade_time'], format='%Y-%m-%d %H:%M:%S')
                df['trade_time'] = df['trade_time'].dt.tz_localize('Asia/Shanghai').dt.tz_convert('UTC').dt.tz_localize(None)
                final_df = df[[
                    "stock", "trade_time", "close", "open", "high", "low", "vol", "amount"
                ]]
                all_data_dicts.extend(final_df.to_dict('records'))
            if len(all_data_dicts) >= BATCH_SAVE_SIZE:
                result_dict = await self._save_all_to_db_native_upsert(
                    model_class=model_class,
                    data_list=all_data_dicts,
                    unique_fields=['stock', 'trade_time']
                )
                total_saved_count += result_dict.get("创建/更新成功", 0)
                all_data_dicts = []
            time.sleep(0.2)
            if len(df) < limit: break
            offset += limit
        if all_data_dicts:
            result_dict = await self._save_all_to_db_native_upsert(
                model_class=model_class,
                data_list=all_data_dicts,
                unique_fields=['stock', 'trade_time']
            )
            total_saved_count += result_dict.get("创建/更新成功", 0)
        return total_saved_count

    # =============== A股分钟行情(实时) ===============
    # =============== A股分钟行情(实时) ===============
    async def save_minute_time_trade_realtime_by_stock_codes_and_time_level(self, stock_codes: List[str], time_level: str):
        """
        【V5.2 - 修复排序Bug版】保存股票的实时分钟级交易数据。
        优化：
        1. groupby 增加 sort=False，防止对 Model 类进行排序导致 TypeError。
        2. 移除 Python 循环，使用 Pandas 向量化操作构建数据库和缓存载荷。
        3. 批量获取 Stock 对象和 Model 类，消除重复查询。
        """
        if not stock_codes:
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        stock_codes_str = ",".join(stock_codes)
        df = self.ts_pro.rt_min(ts_code=stock_codes_str, freq=f"{time_level}MIN", fields=[
            "ts_code", "time", "open", "close", "high", "low", "vol", "amount"
        ])
        if df.empty:
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        # 1. 预处理
        df.dropna(subset=['time', 'ts_code'], inplace=True)
        # 指定格式加速解析
        df['trade_time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        df.dropna(subset=['trade_time'], inplace=True)
        if df.empty:
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        # 2. 准备映射数据
        stock_map = await self.stock_basic_dao.get_stocks_by_codes(stock_codes)
        # 预计算模型映射
        model_map = {code: get_minute_data_model_by_code_and_timelevel(code, time_level) for code in stock_codes}
        # 3. 向量化映射
        df['stock'] = df['ts_code'].map(stock_map)
        df['model_class'] = df['ts_code'].map(model_map)
        # 过滤无效行
        df.dropna(subset=['stock', 'model_class'], inplace=True)
        if df.empty:
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        # 4. 准备数据库载荷 (PostgreSQL)
        # 时区转换: Naive -> Shanghai -> UTC -> Naive (适配原生SQL)
        df['db_time'] = df['trade_time'].dt.tz_localize('Asia/Shanghai').dt.tz_convert('UTC').dt.tz_localize(None)
        db_save_tasks = []
        # 按模型分组构建任务
        # [Fix] 添加 sort=False
        for model_class, group_df in df.groupby('model_class', sort=False):
            # 构造字典列表，重命名 db_time -> trade_time
            payload_df = group_df[['stock', 'db_time', 'open', 'close', 'high', 'low', 'vol', 'amount']].rename(columns={'db_time': 'trade_time'})
            data_list = payload_df.to_dict('records')
            task = self._save_all_to_db_native_upsert(
                model_class=model_class,
                data_list=data_list,
                unique_fields=['stock', 'trade_time']
            )
            db_save_tasks.append(task)
        # 5. 准备缓存载荷 (Redis ZSET)
        # 构造 {stock_code: [record]} 格式
        # 缓存需要原始 trade_time (datetime对象)
        cache_cols = ['open', 'high', 'low', 'close', 'vol', 'amount', 'trade_time']
        # 重命名 vol -> volume
        cache_df = df[['ts_code'] + [c if c != 'vol' else 'vol' for c in cache_cols]].rename(columns={'vol': 'volume'})
        # 使用 groupby 构建字典，虽然这里有循环，但比逐行循环快得多
        cache_payload = {}
        for ts_code, group in cache_df.groupby('ts_code'):
            cache_payload[ts_code] = group[['open', 'high', 'low', 'close', 'volume', 'amount', 'trade_time']].to_dict('records')
        # 6. 并发执行
        cache_save_task = self.cache_set.batch_set_intraday_minute_kline(cache_payload, time_level)
        all_tasks = db_save_tasks + [cache_save_task]
        results = await asyncio.gather(*all_tasks, return_exceptions=True)
        # 7. 统计结果
        final_result = {"尝试处理": len(df), "失败": 0, "创建/更新成功": 0}
        for res in results[:-1]: # 排除最后一个 cache 结果
            if isinstance(res, dict):
                final_result["创建/更新成功"] += res.get("创建/更新成功", 0)
                final_result["失败"] += res.get("失败", 0)
            elif isinstance(res, Exception):
                final_result["失败"] += 1 
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
        【V2.1 - 类型优化版】保存股票的周线交易数据
        优化：强制数值列转换为 float，避免 object 类型。
        """
        trade_date_str = trade_date.strftime('%Y%m%d') if trade_date else ""
        start_date_str = start_date.strftime('%Y%m%d') if start_date else "19900101"
        all_data_dicts = []
        offset = 0
        limit = 6000
        while True:
            df = self.ts_pro.stk_week_month_adj(**{
                "ts_code": "", "trade_date": trade_date_str, "start_date": start_date_str, 
                "end_date": "", "freq": "week", "limit": limit, "offset": offset
            }, fields=[
                "ts_code", "trade_date", "open", "high", "low", "close", "pre_close", 
                "change", "pct_chg", "vol", "amount"
            ])
            if df is None or df.empty: break
            df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
            df.dropna(subset=['ts_code', 'trade_date'], inplace=True)
            unique_ts_codes = df['ts_code'].unique().tolist()
            stock_map = await self.stock_basic_dao.get_stocks_by_codes(unique_ts_codes)
            df['stock'] = df['ts_code'].map(stock_map)
            df.dropna(subset=['stock'], inplace=True)
            if df.empty:
                if len(df) < limit: break
                offset += limit
                continue
            df['trade_time'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.date
            df.rename(columns={'pct_chg': 'pct_change'}, inplace=True)
            # 强制类型转换
            numeric_cols = ['open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_change', 'vol', 'amount']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            model_cols = ['stock', 'trade_time', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_change', 'vol', 'amount']
            final_df = df[model_cols]
            all_data_dicts.extend(final_df.where(pd.notnull(final_df), None).to_dict('records'))
            if len(all_data_dicts) >= BATCH_SAVE_SIZE:
                await self._save_all_to_db_native_upsert(
                    model_class=StockWeeklyData,
                    data_list=all_data_dicts,
                    unique_fields=['stock', 'trade_time']
                )
                all_data_dicts = []
            if len(df) < limit: break
            offset += limit
            await asyncio.sleep(0.5)
        if all_data_dicts:
            await self._save_all_to_db_native_upsert(
                model_class=StockWeeklyData,
                data_list=all_data_dicts,
                unique_fields=['stock', 'trade_time']
            )

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
        【V3.2 - 类型降级优化版】保存股票的月线交易数据 (前复权)。
        优化：
        1. 使用 astype(float) 批量转换数值列，提升速度。
        2. 优化 Pandas 操作链，减少中间变量。
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
            # 1. 数据清洗
            df.replace(['', 'null', 'None', 'nan', 'NaN'], np.nan, inplace=True)
            df.dropna(subset=['ts_code', 'trade_date'], inplace=True)
            if df.empty:
                if len(df) < limit: break
                offset += limit
                continue
            # 2. 批量获取关联对象
            unique_ts_codes = df['ts_code'].unique().tolist()
            stock_map = await self.stock_basic_dao.get_stocks_by_codes(unique_ts_codes)
            df['stock'] = df['ts_code'].map(stock_map)
            df.dropna(subset=['stock'], inplace=True)
            if df.empty:
                if len(df) < limit: break
                offset += limit
                continue
            df['trade_time'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.date
            df.rename(columns={'pct_chg': 'pct_change', 'open_qfq': 'open', 'high_qfq': 'high', 'low_qfq': 'low', 'close_qfq': 'close'}, inplace=True)
            # 3. 向量化数值类型转换 (使用 astype float)
            numeric_cols = ['open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_change', 'vol', 'amount']
            # 确保列存在
            valid_numeric_cols = [c for c in numeric_cols if c in df.columns]
            if valid_numeric_cols:
                df[valid_numeric_cols] = df[valid_numeric_cols].astype(float)
            # amount 单位转换 (千元 -> 元)
            if 'amount' in df.columns:
                df['amount'] = df['amount'] * 1000
            # 4. 导出
            model_cols = ['stock', 'trade_time', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_change', 'vol', 'amount']
            final_df = df[[col for col in model_cols if col in df.columns]]
            records = final_df.where(pd.notnull(final_df), None).to_dict('records')
            all_data_to_save.extend(records)
            if len(all_data_to_save) >= BATCH_SAVE_SIZE:
                await self._save_all_to_db_native_upsert(
                    model_class=StockMonthlyData,
                    data_list=all_data_to_save,
                    unique_fields=['stock', 'trade_time']
                )
                all_data_to_save = []
            await asyncio.sleep(0.6)
            if len(df) < limit: break
            offset += len(df)
        if all_data_to_save:
            logger.info(f"正在保存最后一批剩余的 {len(all_data_to_save)} 条月线数据...")
            await self._save_all_to_db_native_upsert(
                model_class=StockMonthlyData,
                data_list=all_data_to_save,
                unique_fields=['stock', 'trade_time']
            )
        logger.info(f"股票的月线数据保存任务全部完成。")
        return all_data_to_save

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
        【V3.2 - 健壮性优化版】保存所有股票的日线基本信息
        优化：优化空值处理逻辑，确保 limit_status 正确保存。
        """
        params = {
            "ts_code": "",
            "trade_date": trade_date.strftime('%Y%m%d') if trade_date else "",
            "start_date": start_date.strftime('%Y%m%d') if start_date else "",
            "end_date": end_date.strftime('%Y%m%d') if end_date else ""
        }
        fields_to_fetch = [
            "ts_code", "trade_date", "close", "turnover_rate", "turnover_rate_f", "volume_ratio", "pe", "pe_ttm", "pb", "ps",
            "ps_ttm", "dv_ratio", "dv_ttm", "total_share", "float_share", "free_share", "total_mv", "circ_mv", "limit_status"
        ]
        df = self.ts_pro.daily_basic(**params, fields=fields_to_fetch)
        if df.empty:
            return {"status": "success", "message": "No data", "创建/更新成功": 0}
        df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
        unique_ts_codes = df['ts_code'].unique().tolist()
        stock_map = await self.stock_basic_dao.get_stocks_by_codes(unique_ts_codes)
        df['stock'] = df['ts_code'].map(stock_map)
        df.dropna(subset=['trade_date', 'stock'], inplace=True)
        df['trade_time'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.date
        model_columns = [
            "stock", "trade_time", "close", "turnover_rate", "turnover_rate_f", "volume_ratio", "pe", "pe_ttm", "pb", "ps",
            "ps_ttm", "dv_ratio", "dv_ttm", "total_share", "float_share", "free_share", "total_mv", "circ_mv", "limit_status"
        ]
        final_df = df[model_columns]
        # 使用 where 替换 NaN 为 None
        data_dicts_to_save = final_df.where(pd.notnull(final_df), None).to_dict('records')
        saved_count = 0
        if data_dicts_to_save:
            result = await self._save_all_to_db_native_upsert(
                model_class=StockDailyBasic,
                data_list=data_dicts_to_save,
                unique_fields=['stock', 'trade_time']
            )
            saved_count = result.get("创建/更新成功", 0)
        return {"status": "success", "message": f"Processed {saved_count}", "创建/更新成功": saved_count}

    async def save_stock_daily_basic_history_by_stock_codes(self, stock_codes: List[str], trade_date: date = None, start_date: date = None, end_date: date=None) -> None:
        """
        【V2.1 - 缓存构建优化版】保存指定股票列表的日线基本信息
        优化：
        1. 使用 Pandas 向量化构建数据字典，避免逐行循环。
        2. 保持批量 DB 保存的高效性。
        """
        if not stock_codes: return []
        stock_map = await self.stock_basic_dao.get_stocks_by_codes(stock_codes)
        if not stock_map: return []
        stock_codes_str = ",".join(stock_codes)
        trade_date_str = trade_date.strftime('%Y%m%d') if trade_date else ""
        start_date_str = start_date.strftime('%Y%m%d') if start_date else "20200101"
        end_date_str = end_date.strftime('%Y%m%d') if end_date else ""
        all_data_dicts_for_db = []
        offset = 0
        limit = 6000
        while True:
            if offset >= 100000: break
            df = self.ts_pro.daily_basic(**{
                "ts_code": stock_codes_str, "trade_date": trade_date_str, 
                "start_date": start_date_str, "end_date": end_date_str, 
                "limit": limit, "offset": offset
            }, fields=[
                "ts_code", "trade_date", "close", "turnover_rate", "turnover_rate_f", "volume_ratio", "pe", "pe_ttm", "pb", "ps",
                "ps_ttm", "dv_ratio", "dv_ttm", "total_share", "float_share", "free_share", "total_mv", "circ_mv", "limit_status"
            ])
            if df.empty: break
            df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
            df['stock'] = df['ts_code'].map(stock_map)
            df.dropna(subset=['trade_date', 'stock'], inplace=True)
            if not df.empty:
                df['trade_time'] = pd.to_datetime(df['trade_date']).dt.date
                final_df = df[[
                    "stock", "trade_time", "close", "turnover_rate", "turnover_rate_f", "volume_ratio", "pe", "pe_ttm", "pb", "ps",
                    "ps_ttm", "dv_ratio", "dv_ttm", "total_share", "float_share", "free_share", "total_mv", "circ_mv", "limit_status"
                ]]
                # DB 数据
                page_dicts = final_df.where(pd.notnull(final_df), None).to_dict('records')
                all_data_dicts_for_db.extend(page_dicts)
            if len(df) < limit: break
            offset += limit
        result = []
        if all_data_dicts_for_db:
            result = await self._save_all_to_db_native_upsert(
                model_class=StockDailyBasic,
                data_list=all_data_dicts_for_db,
                unique_fields=['stock', 'trade_time']
            )
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
        【V2.1 - 日期解析优化版】保存全市场股票的每日筹码及胜率数据。
        优化：指定 pd.to_datetime 格式，加速解析。
        """
        trade_date_str = trade_date.strftime('%Y%m%d') if trade_date else ""
        start_date_str = start_date.strftime('%Y%m%d') if start_date else "20240101"
        end_date_str = end_date.strftime('%Y%m%d') if end_date else ""
        print("正在预加载所有股票基础信息...")
        all_stocks = await self.stock_basic_dao.get_stock_list()
        if not all_stocks: return
        stock_map = {stock.stock_code: stock for stock in all_stocks}
        all_data_dicts = []
        offset = 0
        limit = 6000
        while True:
            if offset >= 100000: break
            df = self.ts_pro.cyq_perf(**{
                "ts_code": "", "trade_date": trade_date_str, "start_date": start_date_str, 
                "end_date": end_date_str, "limit": limit, "offset": offset
            }, fields=[
                "ts_code", "trade_date", "his_low", "his_high", "cost_5pct", "cost_15pct", "cost_50pct", "cost_85pct",
                "cost_95pct", "weight_avg", "winner_rate"
            ])
            if df.empty: break
            df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
            df['stock'] = df['ts_code'].map(stock_map)
            df.dropna(subset=['ts_code', 'trade_date', 'stock'], inplace=True)
            if not df.empty:
                # 优化：指定格式
                df['trade_time'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.date
                final_df = df[[
                    "stock", "trade_time", "his_low", "his_high", "cost_5pct", "cost_15pct",
                    "cost_50pct", "cost_85pct", "cost_95pct", "weight_avg", "winner_rate"
                ]]
                all_data_dicts.extend(final_df.to_dict('records'))
            if len(all_data_dicts) >= BATCH_SAVE_SIZE:
                await self._save_all_to_db_native_upsert(
                    model_class=StockCyqPerf,
                    data_list=all_data_dicts,
                    unique_fields=['stock', 'trade_time']
                )
                all_data_dicts = []
            if len(df) < limit: break
            offset += limit
        if all_data_dicts:
            await self._save_all_to_db_native_upsert(
                model_class=StockCyqPerf,
                data_list=all_data_dicts,
                unique_fields=['stock', 'trade_time']
            )

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
        【V1.4 - 字段强校验防雷版】保存每日涨跌停价格。
        优化逻辑：
        1. 显式指定 API 返回 fields，防止第三方数据源默认规则变更导致断流。
        2. 引入列存在性拓扑防御，动态补齐缺失维度，杜绝 KeyError 造成的批处理中断。
        3. 强化 NaN 到 None 的转换机制，保障 ORM 批量 Upsert 稳定性。
        """
        api_params = {}
        if trade_date:
            api_params['trade_date'] = trade_date.strftime('%Y%m%d')
        if start_date:
            api_params['start_date'] = start_date.strftime('%Y%m%d')
        if end_date:
            api_params['end_date'] = end_date.strftime('%Y%m%d')
        if not api_params:
            return {"status": "error", "message": "Date parameter is required."}
        try:
            while not await limiter.acquire():
                await asyncio.sleep(10)
            df = self.ts_pro.stk_limit(**api_params, fields=["ts_code", "trade_date", "pre_close", "up_limit", "down_limit"])
            if df.empty:
                return {"status": "success", "message": "No data", "saved_count": 0}
            df.rename(columns={'ts_code': 'stock_code'}, inplace=True)
            df['trade_time'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.date
            unique_codes = df['stock_code'].unique().tolist()
            stock_map = await self.stock_basic_dao.get_stocks_by_codes(unique_codes)
            df['stock'] = df['stock_code'].map(stock_map)
            df.dropna(subset=['stock'], inplace=True)
            model_map = {code: get_stk_limit_model_by_code(code) for code in unique_codes}
            df['model_class'] = df['stock_code'].map(model_map)
            df.dropna(subset=['model_class'], inplace=True)
            required_cols = ['stock', 'trade_time', 'pre_close', 'up_limit', 'down_limit']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = np.nan
            save_tasks = []
            for model_class, group_df in df.groupby('model_class', sort=False):
                final_df = group_df[required_cols]
                data_list = final_df.where(pd.notnull(final_df), None).to_dict('records')
                task = self._save_all_to_db_native_upsert(
                    model_class=model_class,
                    data_list=data_list,
                    unique_fields=['stock', 'trade_time']
                )
                save_tasks.append(task)
            results = await asyncio.gather(*save_tasks)
            total_saved = sum(res.get("创建/更新成功", 0) for res in results if isinstance(res, dict))
            return {"status": "success", "message": f"Saved {total_saved}", "saved_count": total_saved}
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











