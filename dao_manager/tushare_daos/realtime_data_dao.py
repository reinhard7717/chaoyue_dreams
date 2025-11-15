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

from dao_manager.base_dao import BaseDAO
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao

from stock_models.stock_realtime import StockLevel5Data, StockRealtimeData
from utils.model_helpers import get_stock_tick_data_model_by_code

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
        self.ts = ts
        try:
            self.ts.set_token('0793156bc63040ee46008f217c6e76c8b7c415e2748ac0a7bb509d2c')
        except Exception as e:
            logger.error(f"Tushare Token设置失败: {e}", exc_info=True)

    # =================================================================
    # =================== 真实逐笔数据 (Tick Data) 核心接口 =============
    # =================================================================

    # --- 写操作 (Write Operation) ---
    async def save_realtime_tick_in_bulk(self, stock_codes: List[str], trade_date: str) -> bool:
        """
        【V2.1 逐笔数据分表版】并发获取、清洗、并持久化（DB+Cache）多支股票的当日实时逐笔数据。
        - 核心优化: 彻底移除了原有的双重 for 循环，采用 Pandas 向量化操作 (explode, map, merge)
                      来准备数据库载荷 (db_payload)。这极大地提升了处理海量逐笔数据时的性能。
        - 核心修改: 支持 StockTickData 分表存储，根据股票代码将数据保存到对应的分表。
        Returns:
            bool: 操作是否整体成功。
        """
        if not stock_codes:
            return True
        tick_data_map = await self._fetch_raw_ticks_in_bulk(stock_codes, trade_date)
        if not tick_data_map:
            logger.warning("未能获取到任何股票的实时逐笔数据。")
            return False
        df_list_with_codes = [(code, df.reset_index()) for code, df in tick_data_map.items()]
        if not df_list_with_codes:
            return False
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
            return False
        all_ticks_df['model_class'] = all_ticks_df['stock_code'].apply(get_stock_tick_data_model_by_code) # 新增代码行: 映射到对应的分表模型
        all_ticks_df.dropna(subset=['model_class'], inplace=True) # 新增代码行: 过滤掉无法映射到模型的行
        if all_ticks_df.empty: # 新增代码行
            logger.warning("所有逐笔数据都无法映射到有效的分表模型，无数据可保存。") # 新增代码行
            return False # 新增代码行
        final_cols = ['stock', 'trade_time', 'price', 'volume', 'amount', 'type']
        db_payload_df = all_ticks_df[final_cols + ['model_class']] # 修改代码行: 包含 model_class 列
        try:
            db_tasks = [] # 修改代码行: 初始化数据库保存任务列表
            for model_class, group_df in db_payload_df.groupby('model_class'): # 新增代码行: 按分表模型分组数据
                payload_for_model = group_df[final_cols].to_dict('records') # 新增代码行: 获取当前分表的数据载荷
                if payload_for_model: # 新增代码行
                    db_tasks.append(self._save_all_to_db_native_upsert(model_class, payload_for_model, ['stock', 'trade_time', 'price', 'volume'])) # 新增代码行: 为每个分表创建保存任务
            tasks = [ # 修改代码行: 整合所有任务
                *db_tasks, # 修改代码行: 展开数据库保存任务
                self.cache_set.batch_append_real_ticks(tick_data_map)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success = True
            for i, db_result in enumerate(results[:len(db_tasks)]): # 新增代码行: 检查数据库保存任务结果
                if isinstance(db_result, Exception): # 新增代码行
                    logger.error(f"批量保存逐笔数据到数据库分表失败 (任务 {i+1}/{len(db_tasks)}): {db_result}", exc_info=db_result) # 新增代码行
                    success = False # 新增代码行
            cache_result_index = len(db_tasks) # 新增代码行: 缓存任务结果的索引
            if not results[cache_result_index] or isinstance(results[cache_result_index], Exception): # 修改代码行: 检查缓存保存任务结果
                logger.error(f"批量保存逐笔数据到缓存失败: {results[cache_result_index]}", exc_info=isinstance(results[cache_result_index], Exception) and results[cache_result_index] or None) # 修改代码行
                success = False
            return success
        except Exception as e:
            logger.error(f"save_realtime_tick_in_bulk 发生严重异常: {e}", exc_info=True)
            return False

    async def _fetch_raw_ticks_in_bulk(self, stock_codes: List[str], trade_date: str) -> Dict[str, pd.DataFrame]:
        """
        【辅助】使用 asyncio.gather 并发调用 tushare 接口获取原始逐笔数据。
        - 核心修改: 为单个股票的 Tushare API 调用添加重试机制。
        """
        print(f"  -> [探针] 进入 _fetch_raw_ticks_in_bulk，准备获取 {len(stock_codes)} 支股票的逐笔数据...")
        async def fetch_one_stock(code: str):
            max_retries = 3 # 修改代码行: 定义最大重试次数
            initial_delay = 5 # 修改代码行: 定义初始重试延迟（秒）
            for attempt in range(max_retries + 1): # 修改代码行: 循环进行重试
                try:
                    # 使用 sync_to_async 包装同步的 tushare 调用，这是最佳实践
                    df = await sync_to_async(self.ts.realtime_tick)(ts_code=code, src='tx')
                    # 探针 2: 检查 Tushare 原始返回
                    if df is None or df.empty:
                        print(f"    -> [探针] Tushare接口为 {code} 返回了空数据。")
                        return code, None
                    print(f"    -> [探针] 成功获取 {code} 的原始逐笔数据 {len(df)} 条。开始处理...")
                    # 数据清洗与格式化
                    df['trade_time'] = pd.to_datetime(f"{trade_date} " + df['TIME'], errors='coerce')
                    df['PRICE'] = pd.to_numeric(df['PRICE'], errors='coerce')
                    df['VOLUME'] = pd.to_numeric(df['VOLUME'], errors='coerce')
                    df['AMOUNT'] = pd.to_numeric(df['AMOUNT'], errors='coerce')
                    # 探针 3: 检查关键列是否存在NaN，并删除
                    initial_rows = len(df)
                    df.dropna(subset=['trade_time', 'PRICE', 'VOLUME', 'AMOUNT'], inplace=True)
                    if len(df) < initial_rows:
                        print(f"      -> [探针] {code}: 清理了 {initial_rows - len(df)} 条包含NaN的记录。")
                    if df.empty:
                        print(f"      -> [探针] {code}: 清理NaN后数据为空。")
                        return code, None
                    # 单位转换：手 -> 股
                    df['VOLUME'] = (df['VOLUME'] * 100).astype(int)
                    # 重命名并设置索引
                    df.rename(columns={'PRICE': 'price', 'VOLUME': 'volume', 'AMOUNT': 'amount', 'TYPE': 'type'}, inplace=True)
                    df.set_index('trade_time', inplace=True)
                    # 探针 4: 确认最终处理完成
                    print(f"      -> [探针] {code}: 数据处理完成，最终有效数据 {len(df)} 条。")
                    return code, df[['price', 'volume', 'amount', 'type']]
                except Exception as e:
                    if attempt < max_retries: # 修改代码行: 如果未达到最大重试次数，则进行重试
                        delay = initial_delay * (2 ** attempt) # 修改代码行: 指数退避策略
                        print(f"    -> [探针-重试] 获取 {code} 的realtime_tick数据时发生异常: {e}。第 {attempt + 1}/{max_retries} 次重试，等待 {delay} 秒...") # 修改代码行
                        logger.warning(f"获取 {code} 的realtime_tick数据失败: {e}。第 {attempt + 1}/{max_retries} 次重试，等待 {delay} 秒...") # 修改代码行
                        await asyncio.sleep(delay) # 修改代码行: 等待
                    else: # 修改代码行: 达到最大重试次数，放弃
                        # 探针 5: 捕获单次请求的异常
                        print(f"    -> [探针-错误] 获取 {code} 的realtime_tick数据时发生异常: {e}。已达最大重试次数，放弃。") # 修改代码行
                        logger.error(f"获取 {code} 的realtime_tick数据失败: {e}。已达最大重试次数，放弃。", exc_info=True) # 修改代码行
                        return code, None
        tasks = [fetch_one_stock(code) for code in stock_codes]
        results = await asyncio.gather(*tasks)
        final_map = {code: df for code, df in results if df is not None and not df.empty}
        # 探针 6: 打印最终结果
        print(f"  -> [探针] _fetch_raw_ticks_in_bulk 完成。成功获取了 {len(final_map)}/{len(stock_codes)} 支股票的有效逐笔数据。")
        return final_map

    # --- 读操作 (Read Operation) ---
    async def get_daily_real_ticks(self, stock_code: str, trade_date: str) -> Optional[pd.DataFrame]:
        """
        【核心读取】获取单只股票指定日期的真实逐笔数据。
        采用“缓存优先，数据库兜底”策略。
        """
        try:
            # 1. 优先从缓存获取
            df_ticks = await self.cache_get.get_daily_real_ticks(stock_code, trade_date)
            if df_ticks is not None and not df_ticks.empty:
                return df_ticks
            # 2. 缓存未命中，从数据库获取
            logger.info(f"缓存未命中，尝试从数据库获取 {stock_code} on {trade_date} 的真实逐笔数据。")
            df_ticks_from_db = await self._get_daily_real_ticks_from_db(stock_code, trade_date)
            if df_ticks_from_db is not None and not df_ticks_from_db.empty:
                # 3. DB获取成功后，回填缓存，以便下次快速访问
                logger.info(f"数据库命中，正在将 {stock_code} 的逐笔数据回填到缓存...")
                await self.cache_set.batch_append_real_ticks({stock_code: df_ticks_from_db})
                return df_ticks_from_db
            logger.warning(f"缓存和数据库中均未找到 {stock_code} on {trade_date} 的真实逐笔数据。")
            return None
        except Exception as e:
            logger.error(f"get_daily_real_ticks 发生严重异常 for {stock_code}: {e}", exc_info=True)
            return None

    async def _get_daily_real_ticks_from_db(self, stock_code: str, trade_date_str: str) -> Optional[pd.DataFrame]:
        """
        【辅助】从数据库获取指定股票和日期的真实逐笔数据。
        - 核心修改: 根据股票代码动态选择对应的 StockTickData 分表。
        """
        try:
            trade_date_obj = datetime.strptime(trade_date_str, '%Y-%m-%d').date()
            tick_data_model = get_stock_tick_data_model_by_code(stock_code) # 新增代码行: 根据股票代码获取对应的 StockTickData 模型
            if tick_data_model is None: # 新增代码行
                logger.warning(f"无法为股票 {stock_code} 找到对应的 StockTickData 模型，无法从数据库获取数据。") # 新增代码行
                return None # 新增代码行
            query = tick_data_model.objects.filter( # 修改代码行: 使用动态获取的分表模型进行查询
                stock__stock_code=stock_code,
                trade_time__date=trade_date_obj
            ).order_by('trade_time').values(
                'trade_time', 'price', 'volume', 'amount', 'type'
            )
            ticks_list = await sync_to_async(list)(query)
            if not ticks_list:
                return None
            df = pd.DataFrame(ticks_list)
            df.set_index('trade_time', inplace=True)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
            return df
        except Exception as e:
            logger.error(f"从数据库获取 {stock_code} 逐笔数据失败: {e}", exc_info=True)
            return None


    # =================================================================
    # =================== 市场整体快照 (Market Snapshot) 接口 ==========
    # =================================================================
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

    # =================================================================
    # =================== 行情快照 (Quote) 历史接口 ===================
    # =================================================================
    # ▼▼▼ 此方法现在专用于获取行情快照 ▼▼▼
    async def save_quote_data_by_stock_codes(self, stock_codes: List[str]) -> List:
        """
        【改造】获取实时行情快照(realtime_quote)并持久化。
        """
        if not stock_codes:
            return []
        try:
            stock_codes_str = ','.join(stock_codes)
            # 使用 sina 接口，其 volume 单位是“股”
            df = self.ts.realtime_quote(ts_code=stock_codes_str, src='sina')
            if df.empty:
                logger.warning(f"Tushare未返回股票 {stock_codes_str} 的实时行情快照。")
                return []
            stocks_dict = await self.stock_basic_dao.get_stocks_by_codes(stock_codes)
            db_realtime_list, db_level5_list = [], []
            cache_latest_realtime, cache_latest_level5 = {}, {}
            cache_append_realtime, cache_append_level5 = {}, {}
            for row in df.itertuples():
                stock = stocks_dict.get(row.TS_CODE)
                if stock:
                    # FormatProcess 现在应该能正确处理 sina 接口的数据
                    real_dict_db = self.data_format_process.set_realtime_tick_data(stock, row)
                    level5_dict_db = self.data_format_process.set_level5_data(stock, row)
                    db_realtime_list.append(real_dict_db)
                    db_level5_list.append(level5_dict_db)
                    # 准备缓存数据
                    real_dict_cache = self.data_format_process.set_realtime_tick_data(None, row)
                    level5_dict_cache = self.data_format_process.set_level5_data(None, row)
                    cache_latest_realtime[row.TS_CODE] = real_dict_cache
                    cache_latest_level5[row.TS_CODE] = level5_dict_cache
                    # 注意：这里存入 intraday_ticks 的是快照，而非真实逐笔
                    cache_append_realtime[row.TS_CODE] = real_dict_cache
                    cache_append_level5[row.TS_CODE] = level5_dict_cache
            if not db_realtime_list: return []
            tasks = [
                self._save_all_to_db_native_upsert(StockRealtimeData, db_realtime_list, ['stock', 'trade_time']),
                self._save_all_to_db_native_upsert(StockLevel5Data, db_level5_list, ['stock', 'trade_time']),
                self.cache_set.batch_set_latest_realtime_data(cache_latest_realtime),
                self.cache_set.batch_set_latest_level5_data(cache_latest_level5),
                self.cache_set.batch_append_intraday_ticks(cache_append_realtime, cache_append_level5)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # ... (异常处理逻辑不变) ...
            return results[0] if not isinstance(results[0], Exception) else []
        except Exception as e:
            logger.error(f"save_quote_data_by_stock_codes 发生严重异常: {e}", exc_info=True)
            return []

    # ▼▼▼ 此方法现在用于获取快照数据，并明确其数据源 ▼▼▼
    async def get_daily_quotes_and_level5_in_bulk(self, stock_codes: List[str], trade_date: str) -> Dict[str, tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]]:
        """
        【改造】从缓存批量获取行情快照(Quote)和Level5数据。
        """
        if not stock_codes: return {}
        try:
            # ... (时间戳和pipeline准备逻辑不变) ...
            shanghai_tz = pytz.timezone('Asia/Shanghai')
            start_dt_aware = shanghai_tz.localize(datetime.strptime(f"{trade_date} 09:15:00", "%Y-%m-%d %H:%M:%S"))
            end_dt_aware = shanghai_tz.localize(datetime.strptime(f"{trade_date} 15:05:00", "%Y-%m-%d %H:%M:%S"))
            min_score, max_score = int(start_dt_aware.timestamp()), int(end_dt_aware.timestamp())
            date_str_no_hyphen = trade_date.replace('-', '')
            pipe = await self.cache_manager.pipeline()
            for code in stock_codes:
                # 键名不变，但我们现在清楚里面存的是快照
                ticks_key = self.cache_key_stock.intraday_ticks_realtime(code, date_str_no_hyphen)
                level5_key = self.cache_key_stock.intraday_ticks_level5(code, date_str_no_hyphen)
                pipe.zrangebyscore(ticks_key, min_score, max_score, withscores=True)
                pipe.zrangebyscore(level5_key, min_score, max_score, withscores=True)
            results = await pipe.execute()
            bulk_data_map = {}
            for i, stock_code in enumerate(stock_codes):
                quotes_with_scores = results[i * 2]
                level5_with_scores = results[i * 2 + 1]
                # 使用修正后的 _process_serialized_data
                df_quotes = self._process_serialized_data(quotes_with_scores, source='quote')
                df_level5 = self._process_serialized_data(level5_with_scores, source='level5')
                if df_quotes is not None or df_level5 is not None:
                    bulk_data_map[stock_code] = (df_quotes, df_level5)
            return bulk_data_map
        except Exception as e:
            logger.error(f"批量获取行情快照和Level5数据时发生严重错误: {e}", exc_info=True)
            return {}

    # ▼▼▼ 区分数据源，处理不同单位 ▼▼▼
    def _process_serialized_data(self, data_with_scores: list, source: str) -> Optional[pd.DataFrame]:
        """
        【V4.0 - 数据源区分版】
        - 新增 source 参数 ('quote' 或 'level5') 来区分数据源。
        - 核心修正: 根据 source 判断 volume 的单位。
          - 'quote' (来自realtime_quote的sina接口): volume单位是“股”，无需转换。
          - 'level5' (来自realtime_quote的sina接口): 盘口量单位是“手”，需要 * 100。
        """
        if not data_with_scores: return None
        processed_data = [self.cache_manager._deserialize(item) for item, score in data_with_scores]
        processed_data = [d for d in processed_data if d is not None and isinstance(d, dict)]
        if not processed_data: return None
        df = pd.DataFrame(processed_data)
        # 用更可靠的 score (Unix时间戳) 创建或覆盖 trade_time
        df['trade_time'] = [pd.to_datetime(score, unit='s') for item, score in data_with_scores]
        price_cols = [
            'open_price', 'prev_close_price', 'current_price', 'high_price', 'low_price',
            'turnover_value',
            'buy_price1', 'buy_price2', 'buy_price3', 'buy_price4', 'buy_price5',
            'sell_price1', 'sell_price2', 'sell_price3', 'sell_price4', 'sell_price5'
        ]
        for col in price_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
        # --- 关键的单位转换逻辑 ---
        if source == 'quote':
            # realtime_quote (sina) 的 volume 单位是“股”，无需转换
            if 'volume' in df.columns:
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)
        elif source == 'level5':
            # realtime_quote (sina) 的盘口量单位是“手”，需要 * 100
            volume_cols = [
                'buy_volume1', 'buy_volume2', 'buy_volume3', 'buy_volume4', 'buy_volume5',
                'sell_volume1', 'sell_volume2', 'sell_volume3', 'sell_volume4', 'sell_volume5'
            ]
            for col in volume_cols:
                if col in df.columns:
                    df[col] = (pd.to_numeric(df[col], errors='coerce').fillna(0) * 100).astype(int)
        if df['trade_time'].dt.tz is None:
            df['trade_time'] = df['trade_time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
        df.set_index('trade_time', inplace=True)
        return df

    # (保留 get_daily_ticks_from_cache 和 get_latest_tick_data 等辅助方法，但需注意其数据源是快照)
    async def get_daily_ticks_from_cache(self, stock_code: str, trade_date: str) -> Optional[pd.DataFrame]:
        # 注意：此方法获取的是行情快照，而非真实逐笔
        return await self.cache_get.get_intraday_ticks(stock_code, trade_date)

    async def get_latest_tick_data(self, stock_code: str) -> dict:
        # 注意：此方法获取的是最新行情快照
        data_dict = await self.cache_get.latest_tick_data(stock_code)
        return data_dict
