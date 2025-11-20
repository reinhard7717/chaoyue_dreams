# 文件: dao_manager/tushare_daos/realtime_data_dao.py

import asyncio
from asgiref.sync import sync_to_async
import logging
import pandas as pd
from typing import List, Optional, Dict, Tuple, Type
import tushare as ts
import pytz
from datetime import datetime
from decimal import Decimal, InvalidOperation
from collections import defaultdict
from dao_manager.base_dao import BaseDAO
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao

from utils.model_helpers import get_stock_tick_data_model_by_code, get_stock_realtime_data_model_by_code, get_stock_level5_data_model_by_code

from utils.cache_get import StockInfoCacheGet, StockRealtimeCacheGet
from utils.cache_manager import CacheManager
from utils.cache_set import StockRealtimeCacheSet
from utils.cash_key import StockCashKey
from utils.data_format_process import StockRealtimeDataFormatProcess


logger = logging.getLogger("dao")

class StockRealtimeDAO(BaseDAO):
    """
    【V4.0 - 核心换代版】
    - 引入 ts.realtime_tick 接口，获取真实的逐笔成交数据。
    - 核心方法: 新增 get_realtime_tick_in_bulk 用于并发获取逐笔数据。
    - 调整: 区分处理 realtime_quote (快照) 和 realtime_tick (逐笔) 的数据。
    """
    def __init__(self, cache_manager_instance: CacheManager):
        super().__init__(cache_manager_instance=cache_manager_instance, model_class=None)
        self.stock_basic_dao = StockBasicInfoDao(cache_manager_instance)
        self.data_format_process = StockRealtimeDataFormatProcess(cache_manager_instance)
        self.cache_set = StockRealtimeCacheSet(self.cache_manager)
        self.cache_get = StockRealtimeCacheGet(self.cache_manager)
        self.stock_cache_get = StockInfoCacheGet(self.cache_manager)
        self.cache_key_stock = StockCashKey()
        try:
            self.ts = ts
            token = '0793156bc63040ee46008f217c6e76c8b7c415e2748ac0a7bb509d2c'
            self.ts.set_token(token)
            self.pro = self.ts.pro_api(token) # 初始化 pro 接口
            if self.pro is None: # 检查 pro_api 初始化是否成功
                raise ConnectionError("Tushare Pro API 初始化失败，返回 None。")
        except Exception as e:
            logger.critical(f"Tushare 库初始化失败，可能是 token 文件问题或网络问题: {e}", exc_info=True)
            self.ts = None # 初始化失败时，将 ts 设置为 None
            self.pro = None # 初始化失败时，将 pro 设置为 None

    # =================== 真实逐笔数据 (Tick Data) 核心接口 =============

    # --- 写操作 (Write Operation) ---
    async def save_realtime_tick_in_bulk(self, stock_codes: List[str], trade_date: str) -> Tuple[bool, str]:
        """
        【V2.2 逐笔数据分表-无缓存版】并发获取、清洗、并持久化多支股票的当日实时逐笔数据。
        - 核心修改: 移除所有缓存操作，只进行数据库持久化。
        """
        if not stock_codes:
            return True, "股票列表为空，无需处理。"
        tick_data_map = await self._fetch_raw_ticks_in_bulk(stock_codes, trade_date)
        if not tick_data_map:
            logger.warning("未能获取到任何股票的实时逐笔数据。")
            return False, "未能获取到任何股票的实时逐笔数据。"
        df_list_with_codes = [(code, df.reset_index()) for code, df in tick_data_map.items()]
        if not df_list_with_codes:
            return False, "获取到的逐笔数据为空或格式不正确。"
        all_ticks_df = pd.concat(
            [df.assign(stock_code=code) for code, df in df_list_with_codes],
            ignore_index=True
        )
        all_stock_codes_in_df = all_ticks_df['stock_code'].unique().tolist()
        stocks_map = await self.stock_basic_dao.get_stocks_by_codes(all_stock_codes_in_df)
        all_ticks_df['stock'] = all_ticks_df['stock_code'].map(stocks_map)
        all_ticks_df.dropna(subset=['stock'], inplace=True)
        if all_ticks_df.empty:
            logger.warning("所有逐笔数据都无法关联到股票对象，无数据可保存。")
            return False, "所有逐笔数据都无法关联到股票对象。"
        all_ticks_df['model_class'] = all_ticks_df['stock_code'].apply(get_stock_tick_data_model_by_code)
        all_ticks_df.dropna(subset=['model_class'], inplace=True)
        if all_ticks_df.empty:
            logger.warning("所有逐笔数据都无法映射到有效的分表模型，无数据可保存。")
            return False, "所有逐笔数据都无法映射到有效的分表模型。"
        final_cols = ['stock', 'trade_time', 'price', 'volume', 'amount', 'type']
        db_payload_df = all_ticks_df[final_cols + ['model_class']]
        try:
            db_tasks = []
            for model_class, group_df in db_payload_df.groupby('model_class'):
                payload_for_model = group_df[final_cols].to_dict('records')
                if payload_for_model:
                    db_tasks.append(self._save_all_to_db_native_upsert(model_class, payload_for_model, ['stock', 'trade_time', 'price', 'volume']))
            # 移除缓存保存任务
            tasks = db_tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            error_messages = []
            for i, db_result in enumerate(results): # 遍历所有结果，因为现在只有DB任务
                if isinstance(db_result, Exception):
                    msg = f"数据库分表保存失败 (任务 {i+1}): {db_result}"
                    logger.error(msg, exc_info=db_result)
                    error_messages.append(msg)
            # 移除缓存结果检查
            if error_messages:
                return False, f"为 {stock_codes} 保存数据时发生错误: " + "; ".join(error_messages)
            return True, f"成功为 {stock_codes} 处理了逐笔数据。"
        except Exception as e:
            logger.error(f"save_realtime_tick_in_bulk 发生严重异常: {e}", exc_info=True)
            return False, f"save_realtime_tick_in_bulk 发生严重异常: {e}"
    async def _fetch_raw_ticks_in_bulk(self, stock_codes: List[str], trade_date: str) -> Dict[str, pd.DataFrame]:
        """
        【辅助】使用 asyncio.gather 并发调用 tushare 接口获取原始逐笔数据。
        - 核心修改: 为单个股票的 Tushare API 调用添加重试机制。
        """
        type_mapping = {'买盘': 'B', '卖盘': 'S', '中性盘': 'M'}
        async def fetch_one_stock(code: str):
            max_retries = 3
            initial_delay = 5
            for attempt in range(max_retries + 1):
                try:
                    df = await sync_to_async(self.ts.realtime_tick)(ts_code=code, src='tx')
                    if df is None or df.empty:
                        print(f"    -> [探针] Tushare接口为 {code} 返回了空数据。")
                        return code, None
                    df['trade_time'] = pd.to_datetime(f"{trade_date} " + df['TIME'], errors='coerce')
                    df['PRICE'] = pd.to_numeric(df['PRICE'], errors='coerce')
                    df['VOLUME'] = pd.to_numeric(df['VOLUME'], errors='coerce')
                    df['AMOUNT'] = pd.to_numeric(df['AMOUNT'], errors='coerce')
                    initial_rows = len(df)
                    df.dropna(subset=['trade_time', 'PRICE', 'VOLUME', 'AMOUNT'], inplace=True)
                    if len(df) < initial_rows:
                        print(f"      -> [探针] {code}: 清理了 {initial_rows - len(df)} 条包含NaN的记录。")
                    if df.empty:
                        print(f"      -> [探针] {code}: 清理NaN后数据为空。")
                        return code, None
                    df['VOLUME'] = (df['VOLUME'] * 100).astype(int)
                    # 应用类型映射
                    df['TYPE'] = df['TYPE'].map(type_mapping).fillna('M') # 默认中性盘
                    df.rename(columns={'PRICE': 'price', 'VOLUME': 'volume', 'AMOUNT': 'amount', 'TYPE': 'type'}, inplace=True)
                    df.set_index('trade_time', inplace=True)
                    print(f"      -> [探针] {code}: 数据处理完成，最终有效数据 {len(df)} 条。")
                    return code, df[['price', 'volume', 'amount', 'type']]
                except Exception as e:
                    if attempt < max_retries:
                        delay = initial_delay * (2 ** attempt)
                        logger.warning(f"获取 {code} 的realtime_tick数据失败: {e}。第 {attempt + 1}/{max_retries} 次重试，等待 {delay} 秒...")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"获取 {code} 的realtime_tick数据失败: {e}。已达最大重试次数，放弃。", exc_info=True)
                        return code, None
        tasks = [fetch_one_stock(code) for code in stock_codes]
        results = await asyncio.gather(*tasks)
        final_map = {code: df for code, df in results if df is not None and not df.empty}
        return final_map
    
    # --- 读操作 (Read Operation) ---
    async def get_daily_real_ticks(self, stock_code: str, trade_date: str) -> Optional[pd.DataFrame]:
        """
        【核心读取-无缓存版】获取单只股票指定日期的真实逐笔数据。
        - 核心修改: 移除缓存优先策略，直接从数据库获取。
        """
        try:
            # 移除缓存读取和回填逻辑，直接从数据库获取
            df_ticks_from_db = await self._get_daily_real_ticks_from_db(stock_code, trade_date)
            if df_ticks_from_db is not None and not df_ticks_from_db.empty:
                return df_ticks_from_db
            return None
        except Exception as e:
            logger.error(f"get_daily_real_ticks 发生严重异常 for {stock_code}: {e}", exc_info=True)
            return None
    async def _get_daily_real_ticks_from_db(self, stock_code: str, trade_date_str: str) -> Optional[pd.DataFrame]:
        """
        【辅助】从数据库获取指定股票和日期的真实逐笔数据。
        - 核心修改: 根据股票代码动态选择对应的 StockTickData 分表。
        - 【修正】统一将索引转换为 UTC aware datetime。
        - 【修正】使用明确的 UTC aware datetime 范围进行过滤，并添加调试探针。
        - 【修复】修正 NameError: 'ticks_list' 未定义的问题。
        - 【修正】根据最新澄清，数据库存储的逐笔数据是北京时间的 naive datetime，修正时区本地化逻辑。
        """
        from django.utils import timezone
        from datetime import datetime, time, timedelta
        try:
            trade_date_obj = datetime.strptime(trade_date_str, '%Y-%m-%d').date()
            tick_data_model = get_stock_tick_data_model_by_code(stock_code)
            if tick_data_model is None:
                logger.warning(f"无法为股票 {stock_code} 找到对应的 StockTickData 模型，无法从数据库获取数据。")
                return None
            start_of_day_beijing = datetime.combine(trade_date_obj, time.min)
            end_of_day_beijing = datetime.combine(trade_date_obj + timedelta(days=1), time.min)
            start_dt_aware = timezone.make_aware(start_of_day_beijing, timezone=pytz.timezone('Asia/Shanghai')).astimezone(pytz.utc)
            end_dt_aware = timezone.make_aware(end_of_day_beijing, timezone=pytz.timezone('Asia/Shanghai')).astimezone(pytz.utc)
            query = tick_data_model.objects.filter(
                stock__stock_code=stock_code,
                trade_time__gte=start_dt_aware,
                trade_time__lt=end_dt_aware
            ).order_by('trade_time').values(
                'trade_time', 'price', 'volume', 'amount', 'type'
            )
            ticks_list = await sync_to_async(list)(query)
            if not ticks_list:
                return None
            df = pd.DataFrame(ticks_list)
            df.set_index('trade_time', inplace=True)
            if df.index.tz is None:
                # 如果是 naive datetime，根据用户澄清，它实际上是北京时间
                df.index = df.index.tz_localize('Asia/Shanghai', ambiguous='infer')
            else:
                # 如果已经是 aware datetime (例如被错误地标记为UTC)，先转换为 naive，再本地化为北京时间
                df.index = df.index.tz_convert(None).tz_localize('Asia/Shanghai', ambiguous='infer') # 先转naive再localize为Asia/Shanghai
            # DAO统一输出UTC aware datetime，所以最后转换为UTC
            df.index = df.index.tz_convert('UTC') # 确保DAO输出UTC aware
            return df
        except Exception as e:
            logger.error(f"从数据库获取 {stock_code} 逐笔数据失败: {e}", exc_info=True)
            return None

    # =================== 市场整体快照 (Market Snapshot) 接口 ==========
    async def get_realtime_market_snapshot(self, src: str = 'dc') -> Optional[pd.DataFrame]:
        """
        获取实时涨跌幅排名，作为市场雷达。
        数据源默认为东方财富(dc)，信息更全。
        此数据为瞬时快照，通常不进行持久化。
        """
        try:
            # 使用 sync_to_async 包装同步的 tushare 调用
            df = await sync_to_async(self.ts.realtime_list)(src=src)
            if df is None or df.empty:
                logger.warning(f"未能从 Tushare 获取实时市场快照 (src={src})。")
                return None
            return df
        except Exception as e:
            logger.error(f"获取实时市场快照 (realtime_list) 失败: {e}", exc_info=True)
            return None

    # =================== 行情快照 (Quote) 历史接口 ===================
    # ▼▼▼ 此方法现在专用于获取行情快照 ▼▼▼
    async def save_quote_data_by_stock_codes(self, stock_codes: List[str]) -> List:
        """
        【改造-分表版-健壮性增强-无缓存】获取实时行情快照(realtime_quote)并持久化到对应的分表。
        - 核心修改: 移除所有缓存操作，只进行数据库持久化。
        """
        if not self.ts:
            logger.error("Tushare 实例未成功初始化，跳过 save_quote_data_by_stock_codes 任务。")
            return []
        if not stock_codes:
            return []
        try:
            stock_codes_str = ','.join(stock_codes)
            try:
                df = self.ts.realtime_quote(ts_code=stock_codes_str, src='sina')
            except pd.errors.EmptyDataError as e:
                # logger.error(f"Tushare 读取 token 文件失败 (EmptyDataError)，请检查 token 文件是否为空或损坏: {e}", exc_info=True)
                return []
            except Exception as e:
                logger.error(f"调用 Tushare realtime_quote 接口失败: {e}", exc_info=True)
                return []
            if df.empty:
                logger.warning(f"Tushare未返回股票 {stock_codes_str} 的实时行情快照。")
                return []
            stocks_dict = await self.stock_basic_dao.get_stocks_by_codes(stock_codes)
            db_realtime_payloads = defaultdict(list)
            db_level5_payloads = defaultdict(list)
            # 移除缓存相关变量的定义
            # cache_latest_realtime, cache_latest_level5 = {}, {}
            # cache_append_realtime, cache_append_level5 = {}, {}
            for row in df.itertuples():
                stock = stocks_dict.get(row.TS_CODE)
                if stock:
                    realtime_model = get_stock_realtime_data_model_by_code(row.TS_CODE)
                    level5_model = get_stock_level5_data_model_by_code(row.TS_CODE)
                    if not realtime_model or not level5_model:
                        logger.warning(f"无法为 {row.TS_CODE} 找到对应的实时数据分表模型，跳过此股票。")
                        continue
                    real_dict_db = self.data_format_process.set_realtime_tick_data(stock, row)
                    level5_dict_db = self.data_format_process.set_level5_data(stock, row)
                    db_realtime_payloads[realtime_model].append(real_dict_db)
                    db_level5_payloads[level5_model].append(level5_dict_db)
                    # 移除缓存数据填充逻辑
                    # real_dict_cache = self.data_format_process.set_realtime_tick_data(None, row)
                    # level5_dict_cache = self.data_format_process.set_level5_data(None, row)
                    # cache_latest_realtime[row.TS_CODE] = real_dict_cache
                    # cache_latest_level5[row.TS_CODE] = level5_dict_cache
                    # cache_append_realtime[row.TS_CODE] = real_dict_cache
                    # cache_append_level5[row.TS_CODE] = level5_dict_cache
            if not db_realtime_payloads: return []
            db_tasks = []
            for model, payload in db_realtime_payloads.items():
                db_tasks.append(self._save_all_to_db_native_upsert(model, payload, ['stock', 'trade_time']))
            for model, payload in db_level5_payloads.items():
                db_tasks.append(self._save_all_to_db_native_upsert(model, payload, ['stock', 'trade_time']))
            # 移除缓存保存任务
            tasks = db_tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results): # 遍历所有结果，因为现在只有DB任务
                if isinstance(result, Exception):
                    logger.error(f"批量保存行情快照到数据库分表失败 (任务 {i+1}): {result}", exc_info=result)
            return results[0] if not isinstance(results[0], Exception) else []
        except Exception as e:
            logger.error(f"save_quote_data_by_stock_codes 发生严重异常: {e}", exc_info=True)
            return []
    # ▼▼▼ 此方法现在用于获取快照数据，并明确其数据源 ▼▼▼
    async def get_daily_quotes_and_level5_in_bulk(self, stock_codes: List[str], trade_date: str) -> Dict[str, tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]]:
        """
        【改造-无缓存版】从数据库批量获取行情快照(Quote)和Level5数据。
        - 核心修改: 移除缓存读取，直接从数据库分表查询。
        """
        if not stock_codes: return {}
        bulk_data_map = {}
        trade_date_obj = datetime.strptime(trade_date, '%Y-%m-%d').date()
        tasks = []
        for stock_code in stock_codes:
            tasks.append(self._get_single_stock_quotes_and_level5_from_db(stock_code, trade_date_obj))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, stock_code in enumerate(stock_codes):
            result = results[i]
            if isinstance(result, Exception):
                logger.error(f"从数据库获取 {stock_code} 的行情快照和Level5数据失败: {result}", exc_info=result)
                bulk_data_map[stock_code] = (None, None)
            else:
                bulk_data_map[stock_code] = result
        return bulk_data_map
    async def _get_single_stock_quotes_and_level5_from_db(self, stock_code: str, trade_date_obj: datetime.date) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        【辅助】从数据库获取单只股票指定日期的行情快照和Level5数据。
        - 【修正】统一将索引转换为 UTC aware datetime。
        - 【修正】使用明确的 UTC aware datetime 范围进行过滤，并添加调试探针。
        - 【修复】修正 NameError: 'quotes_list' 和 'level5_list' 未定义的问题。
        - 【修正】根据最新澄清，数据库存储的Level5数据是北京时间的 naive datetime，修正时区本地化逻辑。
        """
        from django.utils import timezone
        from datetime import datetime, time, timedelta
        realtime_model = get_stock_realtime_data_model_by_code(stock_code)
        level5_model = get_stock_level5_data_model_by_code(stock_code)
        df_quotes = None
        df_level5 = None
        start_of_day_beijing = datetime.combine(trade_date_obj, time.min)
        end_of_day_beijing = datetime.combine(trade_date_obj + timedelta(days=1), time.min)
        start_dt_aware = timezone.make_aware(start_of_day_beijing, timezone=pytz.timezone('Asia/Shanghai')).astimezone(pytz.utc)
        end_dt_aware = timezone.make_aware(end_of_day_beijing, timezone=pytz.timezone('Asia/Shanghai')).astimezone(pytz.utc)
        if realtime_model:
            query_quotes = realtime_model.objects.filter(
                stock__stock_code=stock_code,
                trade_time__gte=start_dt_aware,
                trade_time__lt=end_dt_aware
            ).order_by('trade_time').values(
                'trade_time', 'open_price', 'prev_close_price', 'current_price',
                'high_price', 'low_price', 'volume', 'turnover_value'
            )
            quotes_list = await sync_to_async(list)(query_quotes)
            if quotes_list:
                df_quotes = pd.DataFrame(quotes_list)
                df_quotes.set_index('trade_time', inplace=True)
                if df_quotes.index.tz is None:
                    df_quotes.index = df_quotes.index.tz_localize('Asia/Shanghai', ambiguous='infer')
                else:
                    df_quotes.index = df_quotes.index.tz_convert(None).tz_localize('Asia/Shanghai', ambiguous='infer') # 先转naive再localize为Asia/Shanghai
                df_quotes.index = df_quotes.index.tz_convert('UTC') # 确保DAO输出UTC aware
        if level5_model:
            query_level5 = level5_model.objects.filter(
                stock__stock_code=stock_code,
                trade_time__gte=start_dt_aware,
                trade_time__lt=end_dt_aware
            ).order_by('trade_time').values(
                'trade_time', 'buy_price1', 'buy_volume1', 'buy_price2', 'buy_volume2',
                'buy_price3', 'buy_volume3', 'buy_price4', 'buy_volume4', 'buy_price5', 'buy_volume5',
                'sell_price1', 'sell_volume1', 'sell_price2', 'sell_volume2',
                'sell_price3', 'sell_volume3', 'sell_price4', 'sell_volume4', 'sell_price5', 'sell_volume5'
            )
            level5_list = await sync_to_async(list)(query_level5)
            if level5_list:
                df_level5 = pd.DataFrame(level5_list)
                df_level5.set_index('trade_time', inplace=True)
                if df_level5.index.tz is None:
                    df_level5.index = df_level5.index.tz_localize('Asia/Shanghai', ambiguous='infer')
                else:
                    df_level5.index = df_level5.index.tz_convert(None).tz_localize('Asia/Shanghai', ambiguous='infer') # 先转naive再localize为Asia/Shanghai
                df_level5.index = df_level5.index.tz_convert('UTC') # 确保DAO输出UTC aware
        return df_quotes, df_level5
    async def get_latest_tick_data(self, stock_code: str) -> dict:
        """
        【无缓存版】从数据库获取最新一条行情快照数据。
        - 核心修改: 移除缓存读取，直接从数据库查询最新记录。
        """
        realtime_model = get_stock_realtime_data_model_by_code(stock_code)
        if not realtime_model:
            logger.warning(f"无法为 {stock_code} 找到对应的实时数据分表模型，无法获取最新数据。")
            return {}
        try:
            latest_data = await sync_to_async(realtime_model.objects.filter(stock__stock_code=stock_code).order_by('-trade_time').first)()
            if latest_data:
                return {
                    'code': stock_code,
                    'current_price': str(latest_data.current_price),
                    'high_price': str(latest_data.high_price),
                    'low_price': str(latest_data.low_price),
                    'open_price': str(latest_data.open_price),
                    'prev_close_price': str(latest_data.prev_close_price),
                    'trade_time': latest_data.trade_time.isoformat(),
                    'turnover_value': str(latest_data.turnover_value),
                    'volume': latest_data.volume,
                    # change_percent 需要计算，这里只返回原始数据
                    'change_percent': None # 无法直接从模型获取，需要额外计算
                }
            return {}
        except Exception as e:
            logger.error(f"从数据库获取 {stock_code} 最新行情快照失败: {e}", exc_info=True)
            return {}
