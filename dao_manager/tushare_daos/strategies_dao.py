# dao_manager\tushare_daos\strategies_dao.py
import logging
from asgiref.sync import sync_to_async
from typing import Any, Dict, List, Optional
from datetime import date, datetime, timedelta
from django.db.models import Max, Q, F, Case, When, Value, BooleanField, Window, CharField
from django.db.models.functions import Concat
from django.db.models.functions import RowNumber
import numpy as np
import pandas as pd
from dao_manager.base_dao import BaseDAO
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from stock_models.fund_flow import FundFlowDailyTHS
from stock_models.stock_analytics import MonthlyTrendStrategyReport, TrendFollowStrategyReport, StockAnalysisResultTrendFollowing, TrendFollowStrategySignalLog, TrendFollowStrategyState
from stock_models.stock_basic import StockInfo
from stock_models.time_trade import StockCyqPerf
from utils.cache_get import StrategyCacheGet
from utils.cache_set import StrategyCacheSet
from functools import reduce
import operator

logger = logging.getLogger("dao")

class StrategiesDAO(BaseDAO):
    def __init__(self):
        self.cache_set = StrategyCacheSet()
        self.cache_get = StrategyCacheGet()
        self.stock_basic_dao = StockBasicInfoDao()
    
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
            beijing_time = timestamp + timedelta(hours=8)  # 直接加8小时得到北京时间
            beijing_time_str = beijing_time.strftime('%Y-%m-%d %H:%M')
            print(f"未找到股票{stock_code}在{beijing_time_str}的策略信号")
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

        # if created:
        #     print(f"[{stock_code}] 在时间点 {timestamp.strftime('%Y-%m-%d %H:%M')} 策略分析结果已成功创建。")
        # else:
        #     print(f"[{stock_code}] 在时间点 {timestamp.strftime('%Y-%m-%d %H:%M')} 策略分析结果已成功更新。")

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

        beijing_time = db_ts + timedelta(hours=8)  # 直接加8小时得到北京时间
        beijing_time_str = beijing_time.strftime('%Y-%m-%d %H:%M')

        # 3. 比较时间戳，数据库的更晚或Redis无数据时才写入Redis
        if (cache_ts is None) or (db_ts > cache_ts):
            # 组装要缓存的数据，假设defaults_kwargs里有所有需要的字段
            data_to_cache = dict(defaults_kwargs)
            data_to_cache['timestamp'] = db_ts.isoformat()  # 建议用ISO格式字符串
            # 写入Redis
            cache_result = await self.cache_set.lastest_analyze_signals_trend_following_data(stock_code, data_to_cache)
            # print(f"写入Redis缓存结果: {cache_result}, 数据时间: {db_ts}")
        # else:
            # print(f"Redis缓存已是最新，无需更新。缓存时间: {cache_ts}, 数据库时间: {db_ts}")

    # --- 一个更通用的、基于字典列表的保存方法 ---
    async def save_monthly_trend_strategy_reports(self, reports_data: List[Dict[str, Any]]) -> int:
        """
        【最终版DAO方法】根据标准化的字典列表，批量创建或更新月线趋势策略报告。
        此方法取代了旧的、基于DataFrame的 save_..._by_trade_date 方法。

        Args:
            reports_data (List[Dict[str, Any]]): 
                一个字典列表，每个字典都由策略的 _prepare_db_record 方法生成，
                包含了所有需要存入数据库的字段。

        Returns:
            int: 成功创建或更新的记录数量。
        """
        if not reports_data:
            print("调试信息: [DAO] 传入的报告数据列表为空，不执行任何操作。")
            return 0

        stock_basic_dao = StockBasicInfoDao()
        data_list_for_db = []

        for report_dict in reports_data:
            # 从传入的字典中获取股票代码
            stock_code = report_dict.get("stock_code")
            if not stock_code:
                print(f"调试信息: [DAO] 报告字典缺少 'stock_code'，跳过此条记录: {report_dict}")
                continue
            
            # 异步获取StockInfo对象
            stock_instance = await stock_basic_dao.get_stock_by_code(stock_code)
            if not stock_instance:
                print(f"调试信息: [DAO] 在数据库中未找到股票: {stock_code}，跳过此条记录。")
                continue
            
            # 构造用于数据库操作的最终字典
            # 核心逻辑：用获取到的 stock_instance 替换掉原来的 stock_code
            db_ready_dict = report_dict.copy()
            db_ready_dict['stock'] = stock_instance
            del db_ready_dict['stock_code'] # 删除临时的stock_code键

            # Django ORM 在处理 None 值时比 np.nan 更健壮，这里做一个转换确保万无一失
            for key, value in db_ready_dict.items():
                if pd.isna(value):
                    db_ready_dict[key] = None
            
            data_list_for_db.append(db_ready_dict)

        if not data_list_for_db:
            print("调试信息: [DAO] 所有记录都因数据问题被跳过，未执行数据库操作。")
            return 0

        # 调用底层的批量更新/插入方法
        # 注意：这里的 unique_fields 必须是模型中 unique_together 定义的字段名
        result_stats = await self._save_all_to_db_native_upsert(
            model_class=MonthlyTrendStrategyReport,
            data_list=data_list_for_db,
            unique_fields=['stock', 'trade_time']
        )
        
        # 返回成功处理的记录数
        success_count = result_stats.get("创建/更新成功", 0)
        print(f"调试信息: [DAO] 批量保存完成。尝试: {len(reports_data)}, 成功: {success_count}")
        return success_count

    async def save_trend_follow_strategy_reports(self, reports_data: List[Dict[str, Any]]) -> int:
        """
        【新增DAO方法】根据标准化的字典列表，批量创建或更新趋势跟踪策略报告。
        此方法由 TrendFollowStrategy 的 prepare_db_records 方法生成的数据驱动。

        Args:
            reports_data (List[Dict[str, Any]]): 
                一个字典列表，每个字典都包含了所有需要存入数据库的字段。

        Returns:
            int: 成功创建或更新的记录数量。
        """
        if not reports_data:
            # print("调试信息: [DAO-TrendFollow] 传入的报告数据列表为空，不执行任何操作。")
            return 0

        stock_basic_dao = StockBasicInfoDao()
        data_list_for_db = []

        for report_dict in reports_data:
            stock_code = report_dict.get("stock_code")
            if not stock_code:
                print(f"调试信息: [DAO-TrendFollow] 报告字典缺少 'stock_code'，跳过: {report_dict}")
                continue
            
            stock_instance = await stock_basic_dao.get_stock_by_code(stock_code)
            if not stock_instance:
                print(f"调试信息: [DAO-TrendFollow] 在数据库中未找到股票: {stock_code}，跳过。")
                continue
            
            db_ready_dict = report_dict.copy()
            db_ready_dict['stock'] = stock_instance
            del db_ready_dict['stock_code']

            for key, value in db_ready_dict.items():
                if pd.isna(value):
                    db_ready_dict[key] = None
            
            data_list_for_db.append(db_ready_dict)

        if not data_list_for_db:
            print("调试信息: [DAO-TrendFollow] 所有记录都因数据问题被跳过，未执行数据库操作。")
            return 0

        # 调用底层的批量更新/插入方法
        # 注意：这里的 unique_fields 必须是模型中 unique_together 定义的字段名
        result_stats = await self._save_all_to_db_native_upsert(
            model_class=TrendFollowStrategyReport, # 使用新模型
            data_list=data_list_for_db,
            unique_fields=['stock', 'trade_time'] # 使用新模型的唯一约束
        )
        
        success_count = result_stats.get("创建/更新成功", 0)
        # print(f"调试信息: [DAO-TrendFollow] 批量保存完成。尝试: {len(reports_data)}, 成功: {success_count}")
        return success_count

    # --- 【终极性能优化】使用窗口函数替换所有逻辑 ---
    @sync_to_async
    def get_latest_monthly_trend_reports(self):
        """
        【终极优化版】使用窗口函数高效获取每只股票最新的月线趋势策略报告。
        此方法是解决此类问题的行业标准，性能最高。

        :return: 一个Django QuerySet，包含最新的报告对象，按买入评分降序排列。
        """
        # print("开始执行【窗口函数版】get_latest_monthly_trend_reports 查询...")

        # 1. 定义窗口函数
        #    - PARTITION BY stock_id: 将数据按股票ID分组
        #    - ORDER BY trade_time DESC: 在每个分组内，按交易时间倒序排列
        #    - RowNumber(): 为排序后的每一行分配一个行号（最新的为1）
        window = Window(
            expression=RowNumber(),
            partition_by=[F('stock_id')],
            order_by=F('trade_time').desc()
        )

        # 2. 使用 annotate 创建一个包含行号的子查询
        #    Django 会将此转换为一个子查询或 CTE (Common Table Expression)
        #    SQL 等价于: SELECT *, ROW_NUMBER() OVER(...) as rn FROM ...
        ranked_reports = MonthlyTrendStrategyReport.objects.annotate(
            row_number=window
        )

        # 3. 从子查询中筛选出我们想要的行 (rn=1)
        #    注意: Django ORM 要求对窗口函数的结果进行筛选时，必须通过 .filter() 作用于 annotate() 之后
        #    为了让数据库能直接处理，我们把它包装成一个子查询
        latest_ids = ranked_reports.filter(row_number=1).values('id')

        # 4. 获取最终的完整报告对象
        #    使用 __in 查询，这比之前的复杂 OR 条件要快得多
        latest_reports_queryset = MonthlyTrendStrategyReport.objects.filter(
            id__in=latest_ids
        ).select_related('stock').order_by('-buy_score', '-trade_time')

        # 调试信息：使用 .explain() 可以看到数据库的执行计划，是性能调试的利器
        # print("数据库的执行计划:")
        # print(latest_reports_queryset.explain(analyze=True))
        
        # print(f"窗口函数查询完成，准备返回结果。")

        # 返回 .values() 以便在视图中直接使用
        return latest_reports_queryset.values(
            'stock__stock_code',
            'stock__stock_name',
            'trade_time',
            'close_D',
            'signal_type',
            'buy_score',
            'analysis_text',
            'signal_breakout_trigger',
            'signal_pullback_entry',
            'signal_continuation_entry',
            'signal_take_profit'
        )

    @sync_to_async
    def get_latest_monthly_trend_reports_by_stock_codes(self, stock_codes):
        """
        【终极优化版】使用窗口函数高效获取【指定股票列表】中每只股票最新的月线趋势策略报告。
        此方法是解决此类问题的行业标准，性能最高。

        :param stock_codes: 一个包含股票代码的列表。
        :return: 一个Django QuerySet，包含最新的报告对象，按买入评分降序排列。
        """
        # 调试信息：打印传入的股票代码列表
        # print(f"DAO层接收到待查询的股票代码: {stock_codes}")
        # --- 代码修改开始 ---
        # 增加对空列表的判断，如果列表为空，则直接返回一个空的QuerySet，避免数据库空查
        if not stock_codes:
            print("股票代码列表为空，直接返回空QuerySet。")
            # 返回一个结构一致的空QuerySet
            return MonthlyTrendStrategyReport.objects.none().values(
                'stock__stock_code',
                'stock__stock_name',
                'trade_time',
                'close_D',
                'signal_type',
                'buy_score',
                'analysis_text',
                'signal_breakout_trigger',
                'signal_pullback_entry',
                'signal_continuation_entry',
                'signal_take_profit'
            )
        # --- 代码修改结束 ---
        # 1. 定义窗口函数 (逻辑不变)
        #    - PARTITION BY stock_id: 将数据按股票ID分组
        #    - ORDER BY trade_time DESC: 在每个分组内，按交易时间倒序排列
        #    - RowNumber(): 为排序后的每一行分配一个行号（最新的为1）
        window = Window(
            expression=RowNumber(),
            partition_by=[F('stock_id')],
            order_by=F('trade_time').desc()
        )
        # --- 代码修改开始 ---
        # 2. 使用 annotate 创建一个包含行号的子查询
        #    【关键修改】在应用窗口函数前，先用传入的 stock_codes 列表进行过滤
        #    这会极大地减少窗口函数需要处理的数据量，是本次性能优化的核心。
        #    通过 stock__stock_code__in 实现跨表查询。
        base_queryset = MonthlyTrendStrategyReport.objects.filter(stock__stock_code__in=stock_codes)
        ranked_reports = base_queryset.annotate(
            row_number=window
        )
        # --- 代码修改结束 ---
        # 3. 从子查询中筛选出我们想要的行 (rn=1) (逻辑不变)
        #    注意: Django ORM 要求对窗口函数的结果进行筛选时，必须通过 .filter() 作用于 annotate() 之后
        #    为了让数据库能直接处理，我们把它包装成一个子查询
        latest_ids = ranked_reports.filter(row_number=1).values('id')
        # 4. 获取最终的完整报告对象 (逻辑不变)
        #    使用 __in 查询，这比之前的复杂 OR 条件要快得多
        latest_reports_queryset = MonthlyTrendStrategyReport.objects.filter(
            id__in=latest_ids
        ).select_related('stock').order_by('-buy_score', '-trade_time')
        # 调试信息：打印查询结果数量
        print(f"查询完成，共找到 {latest_reports_queryset.count()} 条符合条件的最新报告。")
        # 返回 .values() 以便在视图中直接使用 (逻辑不变)
        return latest_reports_queryset.values(
            'stock__stock_code',
            'stock__stock_name',
            'trade_time',
            'close_D',
            'signal_type',
            'buy_score',
            'analysis_text',
            'signal_breakout_trigger',
            'signal_pullback_entry',
            'signal_continuation_entry',
            'signal_take_profit'
        )

    # 内部辅助方法，用于获取日线策略报告的核心查询逻辑
    def _get_latest_trend_follow_reports_queryset(self, base_queryset=None):
        """
        一个内部辅助函数，用于构建获取每只股票最新日线策略报告的查询。
        【最终修正版 V7】: 
        1. 将 signal_details 和 analysis_details 的计算逻辑移入DAO。
        2. 使用 Concat 和 Case/When 在数据库层面直接生成这些字符串。
        3. 这保证了所有注解在一次调用中完成，解决了跨方法注解的引用问题。
        """
        print("--- [DAO] 正在执行 _get_latest_trend_follow_reports_queryset (V7 统一注解版) ---")
        if base_queryset is None:
            base_queryset = TrendFollowStrategyReport.objects.all()

        latest_reports_info = base_queryset.values('stock').annotate(latest_trade_time=Max('trade_time'))

        if not latest_reports_info:
            return TrendFollowStrategyReport.objects.none()

        q_objects = [
            Q(stock_id=item['stock'], trade_time=item['latest_trade_time'])
            for item in latest_reports_info
        ]
        filter_condition = reduce(operator.or_, q_objects)

        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # 【代码修改】将所有计算字段（包括展示用的字符串）在同一个 .annotate() 中完成
        final_queryset = TrendFollowStrategyReport.objects.filter(filter_condition).select_related('stock').annotate(
            # --- 基础字段和别名 ---
            stock_code=F('stock__stock_code'),
            stock_name=F('stock__stock_name'),
            buy_score=F('entry_score'),
            close_D=F('close_price'),
            signal_type=Value('日线趋势'),
            
            # --- 布尔型信号字段 ---
            signal_take_profit=Case(
                When(exit_signal_code__gt=0, then=Value(True)),
                default=Value(False), output_field=BooleanField()
            ),
            signal_pullback_entry=Case(
                When(triggered_playbooks__icontains='pullback', then=Value(True)),
                default=Value(False), output_field=BooleanField()
            ),
            signal_continuation_entry=Case(
                When(triggered_playbooks__icontains='continuation', then=Value(True)),
                default=Value(False), output_field=BooleanField()
            ),
            signal_breakout_trigger=F('is_mid_term_bullish'),

            # --- 拼接展示用的字符串字段 (在数据库层面完成) ---
            signal_details=Concat(
                Case(When(Q(triggered_playbooks__icontains='pullback'), then=Value('回撤买入 ')), default=Value('')),
                Case(When(Q(triggered_playbooks__icontains='continuation'), then=Value('持续买入 ')), default=Value('')),
                Case(When(Q(exit_signal_code__gt=0), then=Value('止盈预警 ')), default=Value('')),
                output_field=CharField()
            ),
            analysis_details=Concat(
                Case(When(is_mid_term_bullish=True, then=Value('中期看涨 ')), default=Value('')),
                Case(When(is_long_term_bullish=True, then=Value('长期看涨 ')), default=Value('')),
                output_field=CharField()
            )
        ).order_by('-trade_time', '-buy_score')
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        
        print("--- [DAO] _get_latest_trend_follow_reports_queryset 执行完毕 (已包含所有字段) ---")
        return final_queryset
    
    # 获取所有股票最新日线策略报告的方法
    @sync_to_async
    def get_latest_trend_follow_reports(self):
        """
        异步获取所有股票的最新日线趋势跟踪报告。
        这个方法直接调用内部辅助函数来完成工作。
        """
        print("--- [DAO] 正在调用 get_latest_trend_follow_reports ---")
        return self._get_latest_trend_follow_reports_queryset()

    # 根据股票代码列表获取最新日线策略报告的方法
    @sync_to_async
    def get_latest_trend_follow_reports_by_stock_codes(self, stock_codes: List[str]):
        """
        异步获取指定股票列表的最新日线趋势跟踪报告。
        这个方法首先过滤出指定股票的报告，然后将结果传递给内部辅助函数。
        """
        print(f"--- [DAO] 正在调用 get_latest_trend_follow_reports_by_stock_codes，代码: {stock_codes} ---")
        if not stock_codes:
            return TrendFollowStrategyReport.objects.none()
        
        # 先按股票代码过滤
        initial_queryset = TrendFollowStrategyReport.objects.filter(stock__stock_code__in=stock_codes)
        
        # 将预过滤的queryset传给辅助函数
        return self._get_latest_trend_follow_reports_queryset(base_queryset=initial_queryset)

    # 获取指定股票的日线资金流和筹码性能数据。
    @sync_to_async
    def get_fund_flow_and_chips_data(self, stock_code: str, trade_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        【优化版】获取指定股票的日线资金流和筹码性能数据。
        - 整个方法被异步化，避免了内部嵌套。
        """
        # print(f"调试信息: [DAO] 正在获取 {stock_code} 的资金流与筹码数据...")
        
        # 获取资金流数据
        fund_flow_qs = FundFlowDailyTHS.objects.filter(stock__stock_code=stock_code).order_by('trade_time')
        if trade_time:
            # 直接使用datetime对象进行过滤
            fund_flow_qs = fund_flow_qs.filter(trade_time__lte=trade_time.date())
        
        # 获取筹码性能数据
        cyq_perf_qs = StockCyqPerf.objects.filter(stock__stock_code=stock_code).order_by('trade_time')
        if trade_time:
            cyq_perf_qs = cyq_perf_qs.filter(trade_time__lte=trade_time.date())

        # 转换为DataFrame
        df_flow = pd.DataFrame.from_records(fund_flow_qs.values('trade_time', 'buy_lg_amount', 'net_d5_amount'))
        df_cyq = pd.DataFrame.from_records(cyq_perf_qs.values('trade_time', 'winner_rate', 'weight_avg', 'cost_95pct'))

        if df_flow.empty or df_cyq.empty:
            print(f"调试信息: [DAO] {stock_code} 的资金流或筹码数据为空。")
            return pd.DataFrame()

        # 合并两个DataFrame
        df_flow['trade_time'] = pd.to_datetime(df_flow['trade_time'])
        df_cyq['trade_time'] = pd.to_datetime(df_cyq['trade_time'])
        
        # 使用外连接保留所有数据，然后处理
        df_merged = pd.merge(df_flow, df_cyq, on='trade_time', how='outer')
        df_merged.sort_values('trade_time', inplace=True)
        
        # 向前填充，因为这些数据是每日更新的，缺失值可以用前一天的数据
        df_merged.ffill(inplace=True)
        df_merged.set_index('trade_time', inplace=True)
        
        # print(f"调试信息: [DAO] 成功获取并合并了 {stock_code} 的 {len(df_merged)} 条资金筹码数据。")
        return df_merged

    async def save_strategy_signals(self, signals_data: List[Dict[str, Any]]) -> int:
        """
        【V2.3 ORM对齐最终修复版】根据标准化的字典列表，批量创建或更新策略信号日志。
        此版本通过提供完整的StockInfo对象实例来解决底层DAO无法识别stock_id的问题。

        Args:
            signals_data (List[Dict[str, Any]]): 
                一个字典列表，每个字典都包含了所有需要存入 TrendFollowStrategySignalLog 模型的字段。
                关键字段必须包括: stock_code, trade_time, timeframe, strategy_name 等。

        Returns:
            int: 成功创建或更新的记录数量。
        """
        if not signals_data:
            print("调试信息: [DAO-SignalLog] 传入的信号数据列表为空，不执行任何操作。")
            return 0
        
        print(f"调试信息: [DAO-SignalLog] 收到 {len(signals_data)} 条原始信号数据，准备进行批量保存。")

        # ▼▼▼【代码修改】: 核心逻辑重构，提供对象实例而非ID ▼▼▼
        # 解释: 底层 BaseDAO 在构建原生SQL时，很可能是通过模型字段名('stock')来查找数据，
        #      而不是数据库列名('stock_id')。因此，我们必须提供一个以'stock'为键，
        #      以StockInfo对象实例为值的数据项。

        # 步骤 1: 提取所有唯一的 stock_code
        stock_codes = {item['stock_code'] for item in signals_data if 'stock_code' in item}
        if not stock_codes:
            print("错误: [DAO-SignalLog] 信号数据中缺少 'stock_code' 字段，无法保存。")
            return 0

        # 步骤 2: 【核心修改】使用 in_bulk 一次性获取所有 StockInfo 对象实例。
        # in_bulk 返回一个以主键(这里是stock_code)为键，以对象实例为值的字典。
        print(f"调试信息: [DAO-SignalLog] 批量获取 {len(stock_codes)} 个 StockInfo 对象实例...")
        stocks_map = await sync_to_async(
            lambda: StockInfo.objects.in_bulk(stock_codes)
        )()
        
        # 步骤 3: 预处理数据列表，用 StockInfo 对象替换 stock_code
        processed_data = []
        for item in signals_data:
            code = item.get('stock_code')
            # 检查我们是否成功获取了该股票的对象实例
            if code in stocks_map:
                processed_item = item.copy()
                # 【核心修改】将键名设置为模型字段名'stock'，值为获取到的对象实例
                processed_item['stock'] = stocks_map[code]
                # 移除不再需要的业务字段'stock_code'
                del processed_item['stock_code']
                processed_data.append(processed_item)
            else:
                print(f"警告: [DAO-SignalLog] 股票代码 {code} 在 StockInfo 表中不存在，该条信号将被跳过。")

        if not processed_data:
            print("调试信息: [DAO-SignalLog] 经过预处理后，没有可供保存的有效信号数据。")
            return 0
        
        print(f"调试信息: [DAO-SignalLog] 预处理完成，{len(processed_data)} 条数据将进行批量更新/插入。")
        # ▲▲▲【代码修改】: 修改结束 ▲▲▲

        # 调用底层的批量更新/插入方法。
        # 底层方法现在会从 processed_item['stock'] 中正确地提取出主键值用于SQL语句。
        result_stats = await self._save_all_to_db_native_upsert(
            model_class=TrendFollowStrategySignalLog,
            data_list=processed_data,
            # unique_fields 仍然使用数据库列名，这是正确的，因为它用于构建原生SQL的
            # ON DUPLICATE KEY UPDATE 部分，这部分直接与数据库列交互。
            unique_fields=['stock_id', 'trade_time', 'strategy_name', 'timeframe']
        )
        
        success_count = result_stats.get("创建/更新成功", 0)
        total_attempted = result_stats.get("尝试处理", 0)
        failed_count = result_stats.get("失败", 0)

        print(f"调试信息: [DAO-SignalLog] 批量保存完成。尝试: {total_attempted}, 成功: {success_count}, 失败: {failed_count}")
        
        return success_count

    async def update_strategy_state(self, stock_code: str, strategy_name: str, timeframe: str):
        """
        【V1.1】在信号生成后，更新策略状态摘要表 (支持多时间框架)。
        """
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        print(f"    [状态摘要] 正在为 {stock.stock_code} ({timeframe}周期) 更新策略状态...")
        try:
            latest_signal = await TrendFollowStrategySignalLog.objects.filter(
                stock=stock, strategy_name=strategy_name, timeframe=timeframe
            ).alatest('trade_time')

            last_buy_signal = await TrendFollowStrategySignalLog.objects.filter(
                stock=stock, strategy_name=strategy_name, timeframe=timeframe, entry_signal=True
            ).order_by('-trade_time').afirst()

            last_sell_signal = await TrendFollowStrategySignalLog.objects.filter(
                stock=stock, strategy_name=strategy_name, timeframe=timeframe, exit_signal_code__gt=0
            ).order_by('-trade_time').afirst()

            state_data = {
                'latest_score': latest_signal.entry_score,
                'latest_trade_time': latest_signal.trade_time,
                'active_playbooks': latest_signal.triggered_playbooks,
                'last_buy_time': last_buy_signal.trade_time if last_buy_signal else None,
                'last_sell_time': last_sell_signal.trade_time if last_sell_signal else None,
            }

            obj, created = await TrendFollowStrategyState.objects.aupdate_or_create(
                stock=stock, strategy_name=strategy_name, time_level=timeframe,
                defaults=state_data
            )
            
            action = "创建" if created else "更新"
            print(f"    [状态摘要] 成功 {action} {stock.stock_code} ({timeframe}周期) 的策略状态。")

        except TrendFollowStrategySignalLog.DoesNotExist:
            print(f"    [状态摘要] 警告: 未找到 {stock.stock_code} ({timeframe}周期) 的信号日志，跳过更新。")
        except Exception as e:
            print(f"    [状态摘要] 错误: 更新 {stock.stock_code} ({timeframe}周期) 状态时发生异常: {e}")

    # 内部辅助方法，用于获取策略信号日志的核心查询逻辑
    def _get_latest_signals_queryset(self, base_queryset=None):
        """
        一个内部辅助函数，用于构建获取每只股票【最新策略信号】的查询。
        该方法是通用的，可以通过 pre-filtering base_queryset 来获取特定时间周期或策略的最新信号。
        例如，要获取最新的60分钟信号，可以传入一个已经 filter(timeframe='60') 的 queryset。
        
        核心逻辑:
        1. 基于传入的 base_queryset，按 stock 分组，找到每个 stock 最新的 signal_time。
        2. 使用 Q 对象构建过滤条件，精确匹配 (stock, latest_signal_time) 的记录。
        3. 在数据库层面使用 annotate 创建丰富的展示字段，适配前端需求。
        """
        print("--- [DAO] 正在执行 _get_latest_signals_queryset (适配SignalLog模型) ---")
        if base_queryset is None:
            # 如果未提供基础查询，则默认查询所有信号日志
            base_queryset = TrendFollowStrategySignalLog.objects.all()

        # 步骤1: 按股票分组，找到每个股票的最新信号时间
        latest_signals_info = base_queryset.values('stock').annotate(latest_signal_time=Max('signal_time'))

        if not latest_signals_info:
            # 如果查询结果为空，直接返回一个空的QuerySet
            return TrendFollowStrategySignalLog.objects.none()

        # 步骤2: 构建 Q 对象，用于过滤出这些最新的记录
        q_objects = [
            Q(stock_id=item['stock'], signal_time=item['latest_signal_time'])
            for item in latest_signals_info
        ]
        filter_condition = reduce(operator.or_, q_objects)

        # 步骤3: 查询并注解，生成最终结果
        final_queryset = TrendFollowStrategySignalLog.objects.filter(filter_condition).select_related('stock').annotate(
            # --- 基础字段和别名，适配新模型 ---
            stock_code=F('stock__stock_code'),
            stock_name=F('stock__stock_name'),
            buy_score=F('entry_score'),
            signal_price=F('close_price'), # 使用更通用的名字 signal_price
            
            # --- 动态生成信号类型描述 ---
            signal_type=Case(
                When(timeframe='D', then=Value('日线信号')),
                When(timeframe='60', then=Value('60分钟信号')),
                When(timeframe='30', then=Value('30分钟信号')),
                When(timeframe='15', then=Value('15分钟信号')),
                default=Concat(F('timeframe'), Value('周期信号')),
                output_field=CharField()
            ),
            
            # --- 布尔型信号字段，逻辑与旧方法类似 ---
            signal_take_profit=Case(
                When(exit_signal_code__gt=0, then=Value(True)),
                default=Value(False), output_field=BooleanField()
            ),
            # 假设 playbook 名称中包含 'pullback', 'continuation', 'breakout' 等关键字
            signal_pullback_entry=Case(
                When(triggered_playbooks__icontains='pullback', then=Value(True)),
                default=Value(False), output_field=BooleanField()
            ),
            signal_continuation_entry=Case(
                When(triggered_playbooks__icontains='continuation', then=Value(True)),
                default=Value(False), output_field=BooleanField()
            ),
            signal_breakout_trigger=Case(
                When(triggered_playbooks__icontains='breakout', then=Value(True)),
                default=Value(False), output_field=BooleanField()
            ),

            # --- 拼接展示用的字符串字段 (在数据库层面完成) ---
            signal_details=Concat(
                Case(When(Q(triggered_playbooks__icontains='pullback'), then=Value('回撤买入 ')), default=Value('')),
                Case(When(Q(triggered_playbooks__icontains='continuation'), then=Value('持续买入 ')), default=Value('')),
                Case(When(Q(triggered_playbooks__icontains='breakout'), then=Value('突破买入 ')), default=Value('')),
                Case(When(Q(exit_signal_code__gt=0), then=Value('止盈预警 ')), default=Value('')),
                output_field=CharField()
            ),
            # 从 context_snapshot (JSONB) 中提取分析细节
            analysis_details=Concat(
                Case(When(Q(context_snapshot__context_mid_term_bullish=True), then=Value('中期看涨 ')), default=Value('')),
                Case(When(Q(context_snapshot__context_long_term_bullish=True), then=Value('长期看涨 ')), default=Value('')),
                output_field=CharField()
            )
        ).order_by('-signal_time', '-buy_score')
        
        print("--- [DAO] _get_latest_signals_queryset 执行完毕 (已包含所有注解字段) ---")
        return final_queryset

    # 获取指定时间周期的最新信号报告
    @sync_to_async
    def get_latest_signals_by_timeframe(self, timeframe: str, stock_codes: Optional[List[str]] = None):
        """
        异步获取指定时间周期下，一批或全部股票的最新信号。
        这是查询60分钟等非日线级别信号的主要入口。

        Args:
            timeframe (str): 时间周期, 例如 '60', '30', 'D'.
            stock_codes (Optional[List[str]]): 可选的股票代码列表。如果为 None, 则查询所有股票。
        """
        print(f"--- [DAO] 正在调用 get_latest_signals_by_timeframe, 周期: {timeframe}, 代码: {stock_codes or '全部'} ---")
        
        # 步骤1: 构建基础查询集，预先过滤时间周期和股票代码
        initial_queryset = TrendFollowStrategySignalLog.objects.filter(timeframe=timeframe)
        if stock_codes:
            initial_queryset = initial_queryset.filter(stock__stock_code__in=stock_codes)
        
        # 步骤2: 将预过滤的 queryset 传给通用的辅助函数
        return self._get_latest_signals_queryset(base_queryset=initial_queryset)

    # 获取所有股票、所有时间周期的最新信号
    @sync_to_async
    def get_latest_signals_for_all(self):
        """
        异步获取所有股票的最新信号（不区分时间周期）。
        对于每只股票，返回其时间上最近的一条信号记录，无论它是日线还是60分钟线。
        """
        print("--- [DAO] 正在调用 get_latest_signals_for_all ---")
        return self._get_latest_signals_queryset()







