# dao_manager\tushare_daos\index_basic_dao.py
import logging
import time
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
from utils.cache_set import IndexCacheSet
from utils.data_format_process import IndexDataFormatProcess


logger = logging.getLogger("dao")

class IndexBasicDAO(BaseDAO):
    def __init__(self):
        super().__init__(None, None, 3600)
        self.data_format_process = IndexDataFormatProcess()
        self.index_cache_set = IndexCacheSet()
        self.index_cache_get = IndexCacheGet()

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
        print(f"基准日期为: {trade_date}")  # 调试信息
        trade_date_str = trade_date.strftime('%Y%m%d')
        trade_days_raw = await sync_to_async(
            lambda: list(
                TradeCalendar.objects.filter(
                    cal_date__lte=trade_date_str,
                    is_open=1
                ).order_by('-cal_date').values_list('cal_date', flat=True)[:n]
            )
        )()
        print(f"查询到的开盘日数量: {len(trade_days_raw)}")  # 调试信息
        # 修改：兼容字符串和datetime.date类型
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
        保存交易日历到数据库
        接口：trade_cal
        描述：获取交易日历数据，默认提取的是上交所交易日历。
        权限：用户需要1000积分可以调取，具体请参阅积分获取办法
        Returns:
            Dict: 保存结果
        """
        result = {}
        # 拉取数据
        df = self.ts_pro.trade_cal(**{
            "exchange": "", "cal_date": "", "start_date": "", "end_date": "", "is_open": "", "limit": "", "offset": ""
        }, fields=[
            "exchange", "cal_date", "is_open", "pretrade_date"
        ])
        trade_cal_dicts = []
        if df is not None:
            df = df.replace(['nan', 'NaN', ''], None)  # 先把字符串nan等变成None
            for row in df.itertuples():
                trade_cal_dict = self.data_format_process.set_trade_calendar_data(row)
                trade_cal_dicts.append(trade_cal_dict)
        if trade_cal_dicts:
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
        return_data = []
        # 先从缓存中获取
        index_list = await self.index_cache_get.all_indexes()
        if index_list:
            for index_dict in index_list:
                return_data.append(index_dict)
            return return_data
        # 从数据库获取
        # logger.info(f"get_stock_by_code从数据库获取股票: {cache_key}, {stock_code}")
        index_list = await sync_to_async(lambda: list(IndexInfo.objects.all()))()
        # 如果数据库中有数据，缓存并返回
        if index_list:
            data_to_cache = []
            for index in index_list:
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
        # 先从缓存中获取
        index_info = await self.index_cache_get.index_data_by_code(index_code)
        if index_info:
            return index_info
        else:
            # 从数据库获取
            index_info = await sync_to_async(lambda: IndexInfo.objects.filter(index_code=index_code).first())()
            if index_info:
                index_data_dict = self.data_format_process.set_index_info_data(index_info)
                await self.index_cache_set.index_info(index_code, index_info)
                return index_data_dict
        return None

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

    async def save_indexs(self) -> Dict:
        """
        保存指数信息到数据库
        接口：index_basic，可以通过数据工具调试和查看数据。
        描述：获取指数基础信息。
        """
        result = {}
        # 拉取数据
        all_dfs = []
        offset = 0
        limit = 8000  # tushare pro接口最大limit一般为8000
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
        # 合并所有df
        result_df = pd.concat(all_dfs, ignore_index=True)
        
        index_dicts = []
        if result_df is not None:
            result_df = result_df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
            result_df = result_df.where(pd.notnull(result_df), None)          # 再把所有np.nan变成None
            for row in result_df.itertuples():
                index_dict = self.data_format_process.set_index_info_data(row)
                index_dicts.append(index_dict)
                await self.index_cache_set.index_info(row.ts_code, index_dict)
            await self.index_cache_set.all_indexes(index_dicts)
        if index_dicts:
            # 保存到数据库
            result =  await self._save_all_to_db_native_upsert(
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
        保存指数成分到数据库
        接口：index_weight
        描述：获取各类指数成分和权重，月度数据 ，建议输入参数里开始日期和结束日分别输入当月第一天和最后一天的日期。
        来源：指数公司网站公开数据
        积分：用户需要至少2000积分才可以调取，具体请参阅积分获取办法
        """
        result = {}
        first_day, last_day = self.get_month_first_last_day()
        # 拉取数据
        df = self.ts_pro.index_weight(**{
            "index_code": "", "trade_date": "", "start_date": first_day.strftime('%Y%m%d'), "end_date": last_day.strftime('%Y%m%d'), "ts_code": "", "limit": "", "offset": ""
        }, fields=[ "index_code", "con_code", "trade_date", "weight" ])
        index_weight_dicts = []
        if df is not None:
            df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
            df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
            for row in df.itertuples():
                index_info = await self.get_index_by_code(row.index_code)
                index_weight_dict = self.data_format_process.set_index_weight_data(index_info=index_info,api_data=row)
                if index_weight_dict.get('index_code') is None:
                    print(f"row: {row}, index_info: {index_info}, index_weight_dict: {index_weight_dict}")
                index_weight_dicts.append(index_weight_dict)
        if index_weight_dicts:
            # 保存到数据库
            result =  await self._save_all_to_db_native_upsert(
                model_class=IndexWeight,
                data_list=index_weight_dicts,
                unique_fields=['index_code', 'trade_date']
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
        index_daily_basic = await sync_to_async(lambda: IndexDaily.objects.filter(index__index_code=index_code, trade_time__range=[start_date, end_date]).all())()
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
        index_daily_basic = await sync_to_async(lambda: IndexDaily.objects.filter(index__index_code=index_code).order_by('-trade_time')[:limit])()
        return index_daily_basic
    
    async def save_index_daily_today(self) -> Dict:
        """
        保存指数每日指标到数据库
        接口：index_daily，可以通过数据工具调试和查看数据。
        描述：目前只提供上证综指，深证成指，上证50，中证500，中小板指，创业板指的每日行情数据
        数据来源：Tushare社区统计计算
        """
        # 获取当前日期
        today = datetime.datetime.today()
        # 转换为YYYYMMDD格式
        today_str = today.strftime('%Y%m%d')
        indexs = await self.get_indexs_by_publisher(publisher="中证指数有限公司")
        all_index_codes = [index.index_code for index in indexs]
        index_dailybasic_dicts = []
        # 切片每50个一组，合成逗号分隔字符串
        batch_size = 50
        for i in range(0, len(all_index_codes), batch_size):
            index_batch = all_index_codes[i:i+batch_size]
            index_codes_str = ",".join(index_batch)
            offset = 0
            limit = 8000
            while True:
                if offset >= 100000:
                    logger.warning(f"offset已达10万，停止拉取。{index_codes_str} 指数日线行情, freq=Day")
                    break
                df = self.ts_pro.index_dailybasic(**{
                    "trade_date": today_str, "ts_code": index_codes_str, "start_date": "", "end_date": "", "limit": limit, "offset": offset
                }, fields=[
                    "ts_code", "trade_date", "total_mv", "float_mv", "total_share", "float_share", "free_share",
                    "turnover_rate", "turnover_rate_f", "pe", "pe_ttm", "pb"
                ])
                if not df.empty:
                    df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
                    df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
                    for row in df.itertuples():
                        index_info = await self.get_index_by_code(row.ts_code)
                        index_dailybasic_dict = self.data_format_process.set_index_daily_data(index_info=index_info, api_data=row)
                        index_dailybasic_dicts.append(index_dailybasic_dict)
                if len(df) < limit:
                    break
                offset += limit
        if index_dailybasic_dicts:
            # 保存到数据库
            result =  await self._save_all_to_db_native_upsert(
                model_class=IndexDaily,
                data_list=index_dailybasic_dicts,
                unique_fields=['index_code', 'trade_time']
            )
        return result        

    async def save_index_daily_history(self, start_date: datetime.date = None, end_date: datetime.date = None) -> Dict:
        """
        保存指数每日指标到数据库
        接口：index_daily，可以通过数据工具调试和查看数据。
        描述：目前只提供上证综指，深证成指，上证50，中证500，中小板指，创业板指的每日行情数据
        数据来源：Tushare社区统计计算
        """
        # 获取当前日期
        today = datetime.datetime.today()
        # 转换为YYYYMMDD格式
        today_str = today.strftime('%Y%m%d')
        start_date_str = "20220101"
        end_date_str = today_str
        if start_date is not None:
            start_date_str = start_date.strftime('%Y%m%d')
        if end_date is not None:
            end_date_str = end_date.strftime('%Y%m%d')
        indexs = await self.get_indexs_by_publisher(publisher="中证指数有限公司")
        for index_info in indexs:
            index_dailybasic_dicts = []
            offset = 0
            limit = 8000
            while True:
                if offset >= 100000:
                    logger.warning(f"offset已达10万，停止拉取。{index_info} 指数日线行情, freq=Day")
                    break
                df = self.ts_pro.index_dailybasic(**{
                    "trade_date": today_str, "ts_code": index_info.index_code, "start_date": start_date_str, "end_date": end_date_str, "limit": limit, "offset": offset
                }, fields=[
                    "ts_code", "trade_date", "total_mv", "float_mv", "total_share", "float_share", "free_share",
                    "turnover_rate", "turnover_rate_f", "pe", "pe_ttm", "pb"
                ])
                if not df.empty:
                    print(f"获取指数日线行情: {index_info}, start_date: {start_date_str}, end_date: {end_date_str}，数据长度: {len(df)}")
                    df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
                    df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
                    for row in df.itertuples():
                        index_dailybasic_dict = self.data_format_process.set_index_daily_data(index_info=index_info, api_data=row)
                        index_dailybasic_dicts.append(index_dailybasic_dict)
                time.sleep(0.3)
                if len(df) < limit:
                    break
                offset += limit
            if index_dailybasic_dicts:
                # 保存到数据库
                result =  await self._save_all_to_db_native_upsert(
                    model_class=IndexDaily,
                    data_list=index_dailybasic_dicts,
                    unique_fields=['index_code', 'trade_time']
                )
                print(f"保存指数日线行情到数据库，{index_info}, start_date: {start_date_str}, end_date: {end_date_str}, result: {result}")
        return result        


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
        接口：index_dailybasic，可以通过数据工具调试和查看数据。
        描述：目前只提供上证综指，深证成指，上证50，中证500，中小板指，创业板指的每日指标数据
        数据来源：Tushare社区统计计算
        数据历史：从2004年1月开始提供
        数据权限：用户需要至少400积分才可以调取，具体请参阅积分获取办法
        """
        # 获取当前日期
        today = datetime.datetime.today()
        # 转换为YYYYMMDD格式
        today_str = today.strftime('%Y%m%d')
        indexs = await self.get_indexs_by_publisher(publisher="中证指数有限公司")
        all_index_codes = [index.index_code for index in indexs]
        index_dailybasic_dicts = []
        # 切片每50个一组，合成逗号分隔字符串
        batch_size = 50
        for i in range(0, len(all_index_codes), batch_size):
            index_batch = all_index_codes[i:i+batch_size]
            index_codes_str = ",".join(index_batch)
            offset = 0
            limit = 8000
            while True:
                if offset >= 100000:
                    logger.warning(f"offset已达10万，停止拉取。{index_codes_str} 指数日线行情, freq=Day")
                    break
                df = self.ts_pro.index_dailybasic(**{
                    "trade_date": today_str, "ts_code": index_codes_str, "start_date": "", "end_date": "", "limit": limit, "offset": offset
                }, fields=[
                    "ts_code", "trade_date", "total_mv", "float_mv", "total_share", "float_share", "free_share",
                    "turnover_rate", "turnover_rate_f", "pe", "pe_ttm", "pb"
                ])
                if not df.empty:
                    df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
                    df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
                    for row in df.itertuples():
                        index_info = await self.get_index_by_code(row.ts_code)
                        # print(f"index_info: {index_info}, type {type(index_info)}")  # 添加日志输出以检查 index_info 的内容和类型
                        index_dailybasic_dict = self.data_format_process.set_index_daily_basic_data(index_info=index_info, api_data=row)
                        index_dailybasic_dicts.append(index_dailybasic_dict)
                if len(df) < limit:
                    break
                offset += limit
        if index_dailybasic_dicts:
            # 保存到数据库
            result =  await self._save_all_to_db_native_upsert(
                model_class=IndexDailyBasic,
                data_list=index_dailybasic_dicts,
                unique_fields=['index_code', 'trade_time']
            )
        return result

    async def save_index_daily_basic_history(self) -> Dict:
        """
        接口：index_dailybasic，可以通过数据工具调试和查看数据。
        描述：目前只提供上证综指，深证成指，上证50，中证500，中小板指，创业板指的每日指标数据
        数据来源：Tushare社区统计计算
        数据历史：从2004年1月开始提供
        数据权限：用户需要至少400积分才可以调取，具体请参阅积分获取办法
        """
        # 获取当前日期
        today = datetime.datetime.today()
        # 转换为YYYYMMDD格式
        today_str = today.strftime('%Y%m%d')
        indexs = await self.get_indexs_by_publisher(publisher="中证指数有限公司")
        for index in indexs:
            index_dailybasic_dicts = []
            # 拉取数据
            offset = 0
            limit = 6000  # tushare pro接口最大limit一般为8000
            while True:
                if offset >= 100000:
                    logger.warning(f"每日筹码及胜率 offset已达10万，停止拉取。")
                    break
                # 拉取数据
                df = self.ts_pro.index_daily(**{
                    "trade_date": index.index_code, "ts_code": "", "start_date": "20220101", "end_date": today_str, "limit": limit, "offset": offset
                }, fields=[
                    "ts_code", "trade_date", "total_mv", "float_mv", "total_share", "float_share", "free_share",
                    "turnover_rate", "turnover_rate_f", "pe", "pe_ttm", "pb"
                ])
                if not df.empty:
                    df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
                    df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
                    for row in df.itertuples():
                        index_info = await self.get_index_by_code(row.ts_code)
                        index_dailybasic_dict = self.data_format_process.set_index_daily_basic_data(index_info=index_info, api_data=row)
                        index_dailybasic_dicts.append(index_dailybasic_dict)
                print(f"index: {index.index_code}, len: {len(index_dailybasic_dicts)}")
                time.sleep(0.3)
                if len(df) < limit:
                    break
                offset += limit
            if index_dailybasic_dicts:
                # 保存到数据库
                result =  await self._save_all_to_db_native_upsert(
                    model_class=IndexDailyBasic,
                    data_list=index_dailybasic_dicts,
                    unique_fields=['index_code', 'trade_time']
                )
                print(f"保存 {index.index_code} 大盘指数每日指标, freq=Day")
        return result
















