# dao_manager\tushare_daos\index_basic_dao.py
import asyncio
import logging
import time
from django.utils import timezone
from asgiref.sync import sync_to_async
import datetime
import calendar
import pandas as pd
import numpy as np
from typing import TYPE_CHECKING, Dict, List, Any, Optional
from dao_manager.base_dao import BaseDAO
from stock_models.index import IndexDailyBasic, IndexInfo, IndexWeight, TradeCalendar
from stock_models.time_trade import IndexDaily
from utils.cache_get import IndexCacheGet
from utils.cache_manager import CacheManager
from utils.cache_set import IndexCacheSet
from utils.data_format_process import IndexDataFormatProcess


logger = logging.getLogger("dao")

class IndexBasicDAO(BaseDAO):
    def __init__(self, cache_manager_instance: CacheManager):
        # 调用 super() 时，将 cache_manager_instance 传递进去
        super().__init__(cache_manager_instance=cache_manager_instance, model_class=None)
        self.data_format_process = IndexDataFormatProcess(cache_manager_instance)
        self.index_cache_set = IndexCacheSet(self.cache_manager)
        self.index_cache_get = IndexCacheGet(self.cache_manager)
    def get_month_first_last_day(self, date=None):
        """
        获取指定日期所在月份的第一天和最后一天
        :param date: datetime.date 或 datetime.datetime 对象，默认为今天
        :return: (first_day, last_day) 元组，都是 datetime.date 类型
        """
        if date is None:
            date = datetime.date.today()
        else:
            # 如果传入的是 datetime.datetime 类型，转为 date
            if isinstance(date, datetime.datetime):
                date = date.date()
        # 本月第一天
        first_day = date.replace(day=1)
        # 本月最后一天
        last_day = date.replace(day=calendar.monthrange(date.year, date.month)[1])
        return first_day, last_day
    # ============== 交易日历 ==============
    async def get_trade_cal(self, start_date: str, end_date: str) -> List['TradeCalendar']:
        """
        获取指定日期范围内的交易日历
        Args:
            start_date: 开始日期，格式为 YYYYMMDD
            end_date: 结束日期，格式为 YYYYMMDD
        Returns:
            List[IndexDailyBasic]: 交易日历列表
        """
        # 从数据库获取
        trade_cals = await sync_to_async(lambda: TradeCalendar.objects.filter(cal_date__range=[start_date, end_date]).all())()
        return trade_cals
    async def is_today_trade_day(self, exchange='SSE'):
        """
        异步判断今天是否为指定交易所的交易日
        :param exchange: 交易所代码，默认'SSE'
        :return: True表示是交易日，False表示休市或无数据
        """
        today = timezone.localdate()  # 获取当前本地日期
        try:
            # 异步查询当天该交易所的交易日状态
            calendar = await TradeCalendar.objects.aget(exchange=exchange, cal_date=today)
            print(f"查询到交易日信息: {calendar}")  # 调试输出
            return calendar.is_open
        except TradeCalendar.DoesNotExist:
            print(f"未查询到{exchange}交易所{today}的交易日信息")  # 调试输出
            return False
    async def get_trade_cal_open(self, start_date: str, end_date: str) -> list:
        """
        获取指定日期范围内的交易日历
        Args:
            start_date: 开始日期，格式为 YYYYMMDD
            end_date: 结束日期，格式为 YYYYMMDD
        Returns:
            List[str]: 交易日字符串列表
        """
        # 一定要加list()，否则返回的是QuerySet，不能在async里遍历
        trade_days = await sync_to_async(
            lambda: list(
                TradeCalendar.objects.filter(
                    cal_date__range=[start_date, end_date],
                    is_open=1
                ).order_by('-cal_date').values_list('cal_date', flat=True)
            )
        )()
        return trade_days
    async def get_last_n_trade_cal_open(self, n: int = 333, trade_date: datetime.date = None) -> list[datetime.date]:
        """
        从数据库中，从trade_date（默认今天）往前读取n个开盘日期
        :param n: 需要获取的开盘日数量
        :param trade_date: 查询的基准日期（datetime.date类型），为空则默认今天
        :return: 开盘日期列表（datetime.date类型）
        """
        if not trade_date:
            trade_date = datetime.date.today()
        # print(f"基准日期为: {trade_date}") 
        trade_date_str = trade_date.strftime('%Y%m%d')
        trade_days_raw = await sync_to_async(
            lambda: list(
                TradeCalendar.objects.filter(
                    cal_date__lte=trade_date_str,
                    is_open=1
                ).order_by('-cal_date').values_list('cal_date', flat=True)[:n]
            )
        )()
        # print(f"查询到的开盘日数量: {len(trade_days_raw)}") 
        # 兼容字符串和datetime.date类型
        trade_days = []
        for day in trade_days_raw:
            if isinstance(day, datetime.date):
                trade_days.append(day)
            elif isinstance(day, str):
                trade_days.append(datetime.datetime.strptime(day, '%Y%m%d').date())
            else:
                print(f"未知类型: {type(day)}, 值: {day}")
        return trade_days
    async def get_trade_cal_list(self) -> List['TradeCalendar']:
        """
        获取全部日期范围的交易日历
        Args:
            start_date: 开始日期，格式为 YYYYMMDD
            end_date: 结束日期，格式为 YYYYMMDD
        Returns:
            List[IndexDailyBasic]: 交易日历列表
        """
        # 从数据库获取
        trade_cal = await sync_to_async(lambda: TradeCalendar.objects.all())()
        return trade_cal
    async def get_trade_cal_by_exchange(self, exchange: str) -> Optional['TradeCalendar']:
        """
        获得指数信息
        Args:
            exchange: 交易所代码
        """
        # 从数据库获取
        trade_cal = await sync_to_async(lambda: TradeCalendar.objects.filter(exchange=exchange).all())()
        if trade_cal:
            return trade_cal
        else:
            return None
    async def save_trade_cal(self) -> Dict:
        """
        【V2.0 向量化优化版】保存交易日历到数据库
        - 核心优化: 使用Pandas向量化操作替代 `itertuples()` 循环，大幅提升数据处理效率。
        """
        # 拉取数据
        df = self.ts_pro.trade_cal(**{
            "exchange": "", "cal_date": "", "start_date": "", "end_date": "", "is_open": "", "limit": "", "offset": ""
        }, fields=[
            "exchange", "cal_date", "is_open", "pretrade_date"
        ])
        if df is None or df.empty:
            return {}
        # --- 向量化数据处理 ---
        # 1. 统一处理空值
        df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
        # 2. 向量化转换日期列，无效日期会转换为NaT(Not a Time)
        df['cal_date'] = pd.to_datetime(df['cal_date'], format='%Y%m%d').dt.date
        df['pretrade_date'] = pd.to_datetime(df['pretrade_date'], format='%Y%m%d', errors='coerce').dt.date
        # 3. 将Pandas的NaT/NaN转换成数据库能接受的None，并直接生成字典列表
        trade_cal_dicts = df.where(pd.notnull(df), None).to_dict('records')
        # --- 原有的 itertuples() 循环已被移除 ---
        if not trade_cal_dicts:
            return {}
        # 保存到数据库
        result = await self._save_all_to_db_native_upsert(
            model_class=TradeCalendar,
            data_list=trade_cal_dicts,
            unique_fields=['exchange', 'cal_date']
        )
        return result
    # ============== 指数基本信息 ==============
    async def get_index_list(self) -> List['IndexInfo']:
        """
        获取所有指数的基本信息
        Returns:
            List[StockInfo]: 指数基本信息列表
        """
        return_data = await sync_to_async(lambda: list(IndexInfo.objects.all()))()
        # 如果数据库中有数据，缓存并返回
        if return_data:
            data_to_cache = []
            for index in return_data:
                index_dict = self.data_format_process.set_index_info_data(index)
                data_to_cache.append(index_dict)
                await self.index_cache_set.index_info(index.index_code, index_dict)
            await self.index_cache_set.all_indexes(data_to_cache)
        return return_data
    async def get_index_by_code(self, index_code) -> Optional['IndexInfo']:
        """
        获得指数信息
        Args
        """
        # 从数据库获取
        index_info = await sync_to_async(lambda: IndexInfo.objects.filter(index_code=index_code).first())()
        if index_info:
            index_data_dict = self.data_format_process.set_index_info_data(index_info)
            await self.index_cache_set.index_info(index_code, index_data_dict)
            return index_info
        return None
    async def get_indices_by_codes(self, index_codes: List[str]) -> Dict[str, 'IndexInfo']:
        """
        【V1.0 新增】根据指数代码列表，一次性从数据库获取所有IndexInfo对象，并返回一个 code -> object 的映射字典。
        """
        if not index_codes:
            return {}
        # print(f"    - [IndexBasicDAO] 正在批量获取 {len(index_codes)} 个指数的信息...")
        # 使用 Django ORM 的异步接口 afilter 和异步推导式
        indices_qs = IndexInfo.objects.filter(index_code__in=index_codes)
        index_map = {index.index_code: index async for index in indices_qs}
        # print(f"    - [IndexBasicDAO] 成功获取并映射了 {len(index_map)} 个指数对象。")
        return index_map
    async def get_or_create_index(self, ts_code: str, defaults: Dict = None) -> IndexInfo:
        """
        获取或创建一条指数信息记录。
        这是一个非常实用的辅助方法，用于确保外键关联的数据存在。
        Args:
            ts_code (str): 指数代码，用于查询或创建。
            defaults (Dict, optional): 创建新记录时使用的默认值字典。
        Returns:
            IndexInfo: 获取到的或新创建的 IndexInfo 对象。
        """
        # 尝试从数据库获取
        index_info = await self.get_index_by_code(ts_code)
        if index_info:
            return index_info
        # 如果不存在，则创建
        logger.info(f"数据库中未找到指数 {ts_code}，将根据提供的数据进行创建...")
        # 准备创建数据
        create_data = defaults or {}
        create_data['index_code'] = ts_code # 确保 index_code 存在
        # 使用 Django ORM 的 get_or_create 异步版本
        # aget_or_create 返回一个元组 (object, created_boolean)
        index_info, created = await IndexInfo.objects.aget_or_create(
            index_code=ts_code,
            defaults=create_data
        )
        if created:
            logger.info(f"成功创建了新的指数记录: {index_info}")
        else:
            # 理论上，在我们的逻辑中，如果 get_index_by_code 没找到，这里应该总是 created=True
            # 但为了健壮性，处理并发场景下可能出现的 race condition
            logger.info(f"在尝试创建时，发现指数 {ts_code} 已存在 (可能由并发操作创建)。")
        return index_info
    async def get_indexs_by_publisher(self, publisher: str="中证指数有限公司") -> Optional[list]:
        """
        获得指数信息
        Args
        """
        # 从数据库获取
        # 用list强制执行ORM查询，避免惰性查询在async上下文触发
        index_infos = await sync_to_async(lambda: list(IndexInfo.objects.filter(publisher=publisher, exp_date=None)))()
        if index_infos:
            return index_infos
        else:
            return None
    @sync_to_async
    def get_all_index_codes(self) -> list[str]:
        """
        【新增优化方法】高效地从数据库获取所有指数的代码列表。
        使用 values_list('index_code', flat=True) 避免加载整个对象，极大提升性能和降低内存消耗。
        """
        print("    [DAO] 正在从数据库高效获取所有指数代码...")
        # 使用 values_list 和 flat=True 直接返回一个字符串列表 ['000001.SH', '399001.SZ', ...]
        codes = list(IndexInfo.objects.values_list('index_code', flat=True))
        print(f"    [DAO] 成功获取 {len(codes)} 个指数代码。")
        return codes
    async def save_indexs(self) -> Dict:
        """
        【V2.0 向量化优化版】保存指数信息到数据库
        - 核心优化: 使用Pandas向量化操作替代 `itertuples()` 循环，提升数据处理效率。
        """
        result = {}
        # 拉取数据
        all_dfs = []
        offset = 0
        limit = 8000
        while True:
            df = self.ts_pro.index_basic(**{
                "ts_code": "", "market": "", "publisher": "", "category": "", "name": "", "limit": limit, "offset": offset
            }, fields=[
                "ts_code", "name", "market", "publisher", "category", "base_date", "base_point", "list_date",
                "fullname", "index_type", "weight_rule", "desc", "exp_date"
            ])
            all_dfs.append(df)
            if len(df) < limit:
                break
            offset += limit
        result_df = pd.concat(all_dfs, ignore_index=True)
        if result_df is None or result_df.empty:
            return {}
        # --- 向量化数据处理 ---
        # 1. 统一处理空值
        result_df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
        # 2. 向量化转换所有日期列
        date_cols = ['base_date', 'list_date', 'exp_date']
        for col in date_cols:
            if col in result_df.columns:
                result_df[col] = pd.to_datetime(result_df[col], format='%Y%m%d', errors='coerce').dt.date
        # 3. 向量化重命名列以匹配模型字段
        result_df.rename(columns={'ts_code': 'index_code'}, inplace=True)
        # 4. 将Pandas的NaT/NaN转换成None，并生成字典列表
        index_dicts = result_df.where(pd.notnull(result_df), None).to_dict('records')
        # --- 原有的 itertuples() 循环已被移除 ---
        # 缓存操作
        for index_dict in index_dicts:
            if index_dict.get('index_code'):
                await self.index_cache_set.index_info(index_dict['index_code'], index_dict)
        await self.index_cache_set.all_indexes(index_dicts)
        if index_dicts:
            # 保存到数据库
            result = await self._save_all_to_db_native_upsert(
                model_class=IndexInfo,
                data_list=index_dicts,
                unique_fields=['index_code']
            )
        return result
    # ============== 指数成分和权重 ==============
    async def get_index_weight(self, index_code):
        """
        获得指数成分
        Args
        """
        # 从数据库获取
        index_weight = await sync_to_async(lambda: IndexWeight.objects.filter(index__index_code=index_code).all())()
        return index_weight
    async def save_index_weight_monthly(self) -> Dict:
        """
        【V2.0 向量化与N+1优化版】保存指数成分到数据库
        - 核心优化:
          1. 【消除N+1查询】一次性批量获取所有指数信息，避免在循环中查询数据库。
          2. 【向量化处理】使用Pandas的向量化操作替代 `itertuples()` 循环，大幅提升效率。
          3. 【逻辑修正】使用正确的外键对象和字段作为唯一性约束，提升数据库操作性能。
        """
        first_day, last_day = self.get_month_first_last_day()
        # 拉取数据
        df = self.ts_pro.index_weight(**{
            "index_code": "", "trade_date": "", "start_date": first_day.strftime('%Y%m%d'), "end_date": last_day.strftime('%Y%m%d'), "limit": "", "offset": ""
        }, fields=["index_code", "con_code", "trade_date", "weight"])
        if df is None or df.empty:
            return {}
        # --- 数据清洗与预处理 ---
        df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
        df.dropna(subset=['index_code', 'con_code', 'trade_date'], inplace=True)
        if df.empty:
            return {}
        # --- N+1查询优化 ---
        unique_index_codes = df['index_code'].unique().tolist()
        index_info_map = await self.get_indices_by_codes(unique_index_codes)
        # --- 向量化处理 ---
        # 1. 将指数对象映射到DataFrame的新列'index'中
        df['index'] = df['index_code'].map(index_info_map)
        # 2. 过滤掉无法关联到指数对象的行
        df.dropna(subset=['index'], inplace=True)
        if df.empty:
            logger.warning("所有指数成分数据都无法关联到已知的指数信息，任务终止。")
            return {}
        # 3. 向量化转换日期和重命名列
        df['trade_time'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.date
        df.rename(columns={'con_code': 'stock_code'}, inplace=True)
        # 4. 选择模型需要的最终列
        final_cols = ['index', 'stock_code', 'trade_time', 'weight']
        df_final = df[final_cols]
        # 5. 将NaN转为None，并生成字典列表
        index_weight_dicts = df_final.where(pd.notnull(df_final), None).to_dict('records')
        # --- 原有的 itertuples() 循环和N+1查询已被移除 ---
        if not index_weight_dicts:
            return {}
        # 保存到数据库
        # 使用 'index', 'stock_code', 'trade_time' 作为联合唯一键，更高效且符合ORM规范
        result = await self._save_all_to_db_native_upsert(
            model_class=IndexWeight,
            data_list=index_weight_dicts,
            unique_fields=['index', 'stock_code', 'trade_time']
        )
        return result
    async def save_index_weight_history_by_index_code(self, index_code) -> Dict:
        """
        保存指数成分到数据库
        接口：index_weight
        描述：获取各类指数成分和权重，月度数据 ，建议输入参数里开始日期和结束日分别输入当月第一天和最后一天的日期。
        来源：指数公司网站公开数据
        积分：用户需要至少2000积分才可以调取，具体请参阅积分获取办法
        """
        result = {}
        # 拉取数据
        df = self.ts_pro.index_weight(**{
            "index_code": index_code, "trade_date": "", "start_date": "", "end_date": "", "ts_code": "", "limit": "", "offset": ""
        }, fields=[ "index_code", "con_code", "trade_date", "weight" ])
        index_weight_dicts = []
        if df is not None:
            df = df.replace(['nan', 'NaN', ''], None)  # 先把字符串nan等变成None
            for row in df.itertuples():
                index_weight_dict = self.data_format_process.set_index_weight_data(row)
                index_weight_dicts.append(index_weight_dict)
    # ============== 指数日线行情 ==============
    async def get_index_daily(self, index_code: str, start_date: str, end_date: str) -> List['IndexDaily']:
        """
        获得指数每日指标
        Args:
            index_code: 指数代码
        """
        # 从数据库获取
        index_daily_basic = await sync_to_async(
                    lambda: list(IndexDaily.objects.filter(index__index_code=index_code, trade_time__range=[start_date, end_date]).all())
                )()
        if index_daily_basic:
            return index_daily_basic
        else:
            return None
    async def get_index_daily_by_limit(self, index_code: str, limit: int) -> List['IndexDaily']:
        """
        获得指数每日指标
        Args:
            index_code: 指数代码
        """
        # 从数据库获取
        index_daily_basic = await sync_to_async(
            lambda: list(IndexDaily.objects.filter(index__index_code=index_code).order_by('-trade_time')[:limit])
        )()
        return index_daily_basic
    async def save_index_daily_history(self, start_date: datetime.date = None, end_date: datetime.date = None, index_codes: list = None) -> Dict:
        """
        【V2.0 向量化内循环优化版】保存指数每日指标到数据库
        - 核心优化:
          1. 【向量化内循环】将分页获取数据后的 `itertuples()` 循环替换为Pandas向量化操作。
          2. 【逻辑修正】使用正确的外键对象作为唯一性约束字段。
          3. 保留了原有的批量获取指数信息和分批入库的优点。
        """
        BATCH_SAVE_SIZE = 100000
        API_REQUEST_LIMIT = 8000
        API_CALL_DELAY_SECONDS = 0.5
        today = datetime.date.today()
        if not index_codes:
            logger.info("未提供指数代码列表，将获取所有指数代码。")
            index_codes = await self.get_all_index_codes()
        if not index_codes:
            logger.warning("数据库中无任何指数信息，任务提前结束。")
            return {"status": "warning", "message": "No index codes found."}
        logger.info(f"准备处理 {len(index_codes)} 个指数。正在一次性从数据库获取其详细信息...")
        index_info_map = await self.get_indices_by_codes(index_codes)
        logger.info(f"成功获取并映射了 {len(index_info_map)} 个指数的详细信息。")
        index_daily_dicts = []
        final_result = None
        for i, index_code in enumerate(index_codes):
            index_info = index_info_map.get(index_code)
            if not index_info:
                logger.warning(f"跳过处理: 在数据库中未找到代码为 {index_code} 的指数信息。")
                continue
            start_date_str = start_date.strftime('%Y%m%d') if start_date else index_info.list_date
            end_date_str = end_date.strftime('%Y%m%d') if end_date else today.strftime('%Ym%d')
            offset = 0
            while True:
                if offset >= 100000:
                    logger.warning(f"Tushare API offset达到10万上限，已停止为指数 {index_code} 继续拉取更早的数据。")
                    break
                df = self.ts_pro.index_daily(
                    ts_code=index_code, start_date=start_date_str, end_date=end_date_str,
                    limit=API_REQUEST_LIMIT, offset=offset
                )
                if df is not None and not df.empty:
                    logger.info(f"进度: {i+1}/{len(index_codes)} | 指数: {index_code} | 日期: {start_date_str}-{end_date_str} | 本次获取: {len(df)}条 | 累计待存: {len(index_daily_dicts) + len(df)}条")
                    # --- 向量化内循环处理 ---
                    df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
                    df['index'] = index_info
                    df['trade_time'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.date
                    # 移除不正确的重命名操作
                    model_cols = ['index', 'trade_time', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount'] # 使用正确的模型字段名 'pct_chg'
                    df_final = df[[col for col in model_cols if col in df.columns]]
                    new_dicts = df_final.where(pd.notnull(df_final), None).to_dict('records')
                    index_daily_dicts.extend(new_dicts)
                    # --- 原有的 itertuples() 循环已被移除 ---
                    if len(index_daily_dicts) >= BATCH_SAVE_SIZE:
                        logger.info(f"数据缓存池达到 {len(index_daily_dicts)} 条，开始批量写入数据库...")
                        final_result = await self._save_all_to_db_native_upsert(
                            model_class=IndexDaily,
                            data_list=index_daily_dicts,
                            unique_fields=['index', 'trade_time'] # 逻辑修正
                        )
                        logger.info(f"批量写入完成。结果: {final_result}")
                        index_daily_dicts.clear()
                await asyncio.sleep(API_CALL_DELAY_SECONDS)
                if df is None or len(df) < API_REQUEST_LIMIT:
                    break
                offset += API_REQUEST_LIMIT
        if index_daily_dicts:
            logger.info(f"所有指数处理完毕，正在保存最后剩余的 {len(index_daily_dicts)} 条数据...")
            final_result = await self._save_all_to_db_native_upsert(
                model_class=IndexDaily,
                data_list=index_daily_dicts,
                unique_fields=['index', 'trade_time'] # 逻辑修正
            )
            logger.info(f"最后的批量写入完成。结果: {final_result}")
        logger.info("指数每日指标历史数据保存任务全部完成。")
        return final_result
    # ============== 大盘指数每日指标 ==============
    async def get_index_daily_basic_by_limit(self, index_code: str, limit: int) -> List['IndexDailyBasic']:
        """
        获得指数每日指标
        Args:
            index_code: 指数代码
        """
        # 从数据库获取
        index_daily_basic = await sync_to_async(lambda: IndexDailyBasic.objects.filter(index__index_code=index_code).order_by('-trade_time')[:limit])()
        return index_daily_basic
    async def save_index_daily_basic_today(self) -> Dict:
        """
        【V2.0 优化版】保存当天的大盘指数每日指标。
        - 核心优化: 调用可复用的 `_save_index_daily_basic_by_date` 辅助方法，实现高效的向量化处理和N+1查询消除。
        """
        # 直接调用重构后的辅助方法
        today = datetime.date.today()
        return await self._save_index_daily_basic_by_date(today)
    async def save_index_daily_basic_yesterday(self) -> Dict:
        """
        【V2.0 优化版】保存昨天的大盘指数每日指标。
        - 核心优化: 调用可复用的 `_save_index_daily_basic_by_date` 辅助方法，实现高效的向量化处理和N+1查询消除。
        """
        # 直接调用重构后的辅助方法
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        return await self._save_index_daily_basic_by_date(yesterday)
    async def save_index_daily_basic_history(self, start_date: datetime.date = None, end_date: datetime.date = None) -> Dict:
        """
        【V2.0 向量化内循环优化版】保存历史大盘指数每日指标
        - 核心优化:
          1. 【向量化内循环】将分页获取数据后的 `itertuples()` 循环替换为Pandas向量化操作。
          2. 【逻辑修正】使用正确的外键对象作为唯一性约束字段。
        """
        today = datetime.date.today()
        # 注意：此接口只提供部分大盘指数，直接获取所有指数可能导致不必要的API调用。
        # Tushare文档建议直接调用，它会返回所有可用的指数数据。
        # 因此，我们不再循环获取指数列表，而是直接进行一次范围查询。
        start_date_str = start_date.strftime('%Y%m%d') if start_date else '20040101' # 数据最早从2004年开始
        end_date_str = end_date.strftime('%Y%m%d') if end_date else today.strftime('%Y%m%d')
        all_dfs = []
        offset = 0
        limit = 8000
        while True:
            if offset >= 100000:
                logger.warning(f"大盘指数每日指标 offset已达10万，停止拉取。")
                break
            df = self.ts_pro.index_dailybasic(**{
                "ts_code": "", "start_date": start_date_str, "end_date": end_date_str, "limit": limit, "offset": offset
            }, fields=[
                "ts_code", "trade_date", "total_mv", "float_mv", "total_share", "float_share", "free_share",
                "turnover_rate", "turnover_rate_f", "pe", "pe_ttm", "pb"
            ])
            if df is None or df.empty:
                break
            all_dfs.append(df)
            await asyncio.sleep(0.3) # 使用异步sleep
            if len(df) < limit:
                break
            offset += limit
        if not all_dfs:
            return {}
        # --- 向量化处理 ---
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.drop_duplicates(subset=['ts_code', 'trade_date'], keep='first', inplace=True)
        combined_df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
        combined_df.dropna(subset=['ts_code'], inplace=True)
        if combined_df.empty:
            return {}
        # --- N+1查询优化 ---
        unique_ts_codes = combined_df['ts_code'].unique().tolist()
        index_info_map = await self.get_indices_by_codes(unique_ts_codes)
        # --- 向量化处理 ---
        combined_df['index'] = combined_df['ts_code'].map(index_info_map)
        combined_df.dropna(subset=['index'], inplace=True)
        if combined_df.empty:
            return {}
        combined_df['trade_time'] = pd.to_datetime(combined_df['trade_date'], format='%Y%m%d').dt.date
        model_cols = ['index', 'trade_time', 'total_mv', 'float_mv', 'total_share', 'float_share', 'free_share', 'turnover_rate', 'turnover_rate_f', 'pe', 'pe_ttm', 'pb']
        final_df = combined_df[[col for col in model_cols if col in combined_df.columns]]
        data_list = final_df.where(pd.notnull(final_df), None).to_dict('records')
        if not data_list:
            return {}
        # --- 批量保存 ---
        result = await self._save_all_to_db_native_upsert(
            model_class=IndexDailyBasic,
            data_list=data_list,
            unique_fields=['index', 'trade_time'] # 逻辑修正
        )
        print(f"保存大盘指数每日指标历史数据完成。")
        return result
    async def _save_index_daily_basic_by_date(self, target_date: datetime.date) -> Optional[Dict]:
        """
        【V1.0 新增辅助方法】根据指定日期，获取并保存大盘指数每日指标。
        - 核心优化:
          1. 【消除N+1查询】一次性批量获取所有指数信息，避免在循环中查询数据库。
          2. 【向量化处理】使用Pandas的向量化操作替代 itertuples() 循环，大幅提升效率。
          3. 【逻辑修正】使用正确的外键对象作为唯一性约束字段。
        Args:
            target_date (datetime.date): 目标日期
        Returns:
            Optional[Dict]: 保存结果或None
        """
        # 将日期转换为Tushare API要求的YYYYMMDD格式字符串
        date_str = target_date.strftime('%Y%m%d')
        # 调用API获取数据
        df = self.ts_pro.index_dailybasic(**{
            "trade_date": date_str, "ts_code": "", "start_date": "", "end_date": "", "limit": "", "offset": ""
        }, fields=[
            "ts_code", "trade_date", "total_mv", "float_mv", "total_share", "float_share", "free_share",
            "turnover_rate", "turnover_rate_f", "pe", "pe_ttm", "pb"
        ])
        if df is None or df.empty:
            logger.info(f"日期 {date_str} 没有大盘指数每日指标数据。")
            return None
        # --- 数据清洗与预处理 ---
        df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
        df.dropna(subset=['ts_code'], inplace=True)
        if df.empty:
            return None
        # --- N+1查询优化 ---
        # 1. 一次性获取所有需要的指数代码
        unique_ts_codes = df['ts_code'].unique().tolist()
        # 2. 一次性从数据库查询所有指数对象
        index_info_map = await self.get_indices_by_codes(unique_ts_codes)
        # --- 向量化处理 ---
        # 3. 将指数对象映射到DataFrame的新列'index'中
        df['index'] = df['ts_code'].map(index_info_map)
        # 4. 过滤掉无法关联到指数对象的行
        df.dropna(subset=['index'], inplace=True)
        if df.empty:
            return None
        # 5. 向量化转换日期格式
        df['trade_time'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.date
        # 6. 选择模型需要的列
        model_cols = ['index', 'trade_time', 'total_mv', 'float_mv', 'total_share', 'float_share', 'free_share', 'turnover_rate', 'turnover_rate_f', 'pe', 'pe_ttm', 'pb']
        final_df = df[[col for col in model_cols if col in df.columns]]
        # 7. 将NaN转为None，并生成字典列表
        data_list = final_df.where(pd.notnull(final_df), None).to_dict('records')
        if not data_list:
            return None
        # --- 批量保存 ---
        # 使用 'index' 对象作为唯一键，更高效且符合ORM规范
        result = await self._save_all_to_db_native_upsert(
            model_class=IndexDailyBasic,
            data_list=data_list,
            unique_fields=['index', 'trade_time']
        )
        return result















