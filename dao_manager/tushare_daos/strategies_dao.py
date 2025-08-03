# dao_manager\tushare_daos\strategies_dao.py
import logging
from asgiref.sync import sync_to_async
from typing import Any, Dict, List, Optional, Tuple
from datetime import date, datetime, timedelta
from django.db.models import Max, Q, F, Case, When, Value, BooleanField, Window, CharField
from django.db.models.functions import Concat
from django.db.models.functions import RowNumber
import numpy as np
from django.db import transaction
import pandas as pd
from dao_manager.base_dao import BaseDAO
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from dao_manager.tushare_daos.fund_flow_dao import FundFlowDao
from stock_models.stock_analytics import TradingSignal, SignalPlaybookDetail
from stock_models.stock_analytics import TrendFollowStrategySignalLog
from stock_models.time_trade import AdvancedChipMetrics, StockCyqChipsBJ, StockCyqChipsCY, StockCyqChipsKC, StockCyqChipsSH, StockCyqChipsSZ, StockCyqPerf, StockDailyBasic
from utils.cache_get import StrategyCacheGet
from utils.cache_manager import CacheManager
from utils.cache_set import StrategyCacheSet
from functools import reduce
import operator

logger = logging.getLogger("dao")

class StrategiesDAO(BaseDAO):
    def __init__(self, cache_manager_instance: CacheManager):
        # 【核心修改】调用 super() 时，将 cache_manager_instance 传递进去
        super().__init__(cache_manager_instance=cache_manager_instance, model_class=None)
        self.cache_set = StrategyCacheSet(self.cache_manager)
        self.cache_get = StrategyCacheGet(self.cache_manager)
        self.stock_basic_dao = StockBasicInfoDao(cache_manager_instance)
        self.fund_flow_dao = FundFlowDao(cache_manager_instance)
        self.cache_manager = cache_manager_instance

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

    async def get_latest_strategy_result(self, stock_code: str):
        """
        获取指定股票的最新策略信号。
        :param stock_code: 股票代码
        :return: 最新的策略信号对象或None
        """
        # MODIFIED: 替换为使用 self.stock_basic_dao 而不是创建新实例
        # 异步获取股票对象
        stock_obj = await self.stock_basic_dao.get_stock_by_code(stock_code)
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
        # MODIFIED: 替换为使用 self.stock_basic_dao 而不是创建新实例
        # 异步获取股票对象
        stock_obj = await self.stock_basic_dao.get_stock_by_code(stock_code)
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
        # MODIFIED: 替换为使用 self.stock_basic_dao 而不是创建新实例
        # 异步获取股票对象
        stock_obj = await self.stock_basic_dao.get_stock_by_code(stock_code)
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

        # MODIFIED: 替换为使用 self.stock_basic_dao 而不是创建新实例
        data_list_for_db = []

        for report_dict in reports_data:
            # 从传入的字典中获取股票代码
            stock_code = report_dict.get("stock_code")
            if not stock_code:
                print(f"调试信息: [DAO] 报告字典缺少 'stock_code'，跳过此条记录: {report_dict}")
                continue
            
            # 异步获取StockInfo对象
            stock_instance = await self.stock_basic_dao.get_stock_by_code(stock_code)
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

        # MODIFIED: 替换为使用 self.stock_basic_dao 而不是创建新实例
        data_list_for_db = []

        for report_dict in reports_data:
            stock_code = report_dict.get("stock_code")
            if not stock_code:
                print(f"调试信息: [DAO-TrendFollow] 报告字典缺少 'stock_code'，跳过: {report_dict}")
                continue
            
            stock_instance = await self.stock_basic_dao.get_stock_by_code(stock_code)
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
    def get_fund_flow_and_chips_data(self, stock_code: str, trade_time: Optional[datetime] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """
        【V2.0 架构优化版】获取指定股票的日线资金流和筹码性能数据。
        - 动态模型: 使用 get_fund_flow_model_by_code 自动选择正确的资金流模型。
        - 性能优化: 增加 limit 参数，从数据库层面限制返回记录数，避免全量查询。
        - 数据丰富: 提取了更多有价值的资金流和筹码字段。
        - 健壮合并: 使用 'outer' 合并，并统一使用 ffill() 填充，确保数据连续性。
        """
        # print(f"    - [DAO] 正在为 {stock_code} 获取补充数据 (资金流、筹码)...")
        
        FundFlowModel = self.fund_flow_dao.get_fund_flow_model_by_code(stock_code)
        if not FundFlowModel:
            logger.error(f"无法为 {stock_code} 找到对应的资金流模型，跳过补充数据获取。")
            return pd.DataFrame()

        # 1. 获取资金流数据
        fund_flow_qs = FundFlowModel.objects.filter(stock__stock_code=stock_code).order_by('-trade_time')
        if trade_time:
            fund_flow_qs = fund_flow_qs.filter(trade_time__lte=trade_time.date())
        if limit:
            fund_flow_qs = fund_flow_qs[:limit]
        
        # 2. 获取筹码性能数据
        cyq_perf_qs = StockCyqPerf.objects.filter(stock__stock_code=stock_code).order_by('-trade_time')
        if trade_time:
            cyq_perf_qs = cyq_perf_qs.filter(trade_time__lte=trade_time.date())
        if limit:
            cyq_perf_qs = cyq_perf_qs[:limit]

        # 3. 转换为DataFrame
        # ▼▼▼ 提取更丰富的字段 ▼▼▼
        flow_fields = ['trade_time', 'buy_sm_amount', 'sell_sm_amount', 'buy_md_amount', 'sell_md_amount', 'buy_lg_amount', 'sell_lg_amount', 'buy_elg_amount', 'sell_elg_amount', 'net_mf_amount']
        cyq_fields = ['trade_time', 'his_low', 'his_high', 'cost_5pct', 'cost_15pct', 'cost_50pct', 'cost_85pct', 'cost_95pct', 'weight_avg', 'winner_rate']
        
        df_flow = pd.DataFrame.from_records(fund_flow_qs.values(*flow_fields))
        df_cyq = pd.DataFrame.from_records(cyq_perf_qs.values(*cyq_fields))
        if df_flow.empty and df_cyq.empty:
            print(f"    - [DAO] {stock_code} 的资金流和筹码数据均为空。")
            return pd.DataFrame()

        # 4. 合并两个DataFrame
        # ▼▼▼ 统一和健壮的合并与填充逻辑 ▼▼▼
        if not df_flow.empty:
            df_flow['trade_time'] = pd.to_datetime(df_flow['trade_time'], utc=True)
        if not df_cyq.empty:
            df_cyq['trade_time'] = pd.to_datetime(df_cyq['trade_time'], utc=True)

        if df_flow.empty:
            df_merged = df_cyq
        elif df_cyq.empty:
            df_merged = df_flow
        else:
            # 使用外连接(outer)保留所有日期的数据，然后填充
            df_merged = pd.merge(df_flow, df_cyq, on='trade_time', how='outer')
        
        # 排序是填充前的重要步骤
        df_merged.sort_values('trade_time', inplace=True)
        
        # 使用ffill向前填充所有补充数据，这是正确的处理方式
        df_merged.ffill(inplace=True)
        df_merged.set_index('trade_time', inplace=True)
        
        # print(f"    - [DAO] 成功获取并合并了 {stock_code} 的 {len(df_merged)} 条补充数据。")
        return df_merged

    @sync_to_async(thread_sensitive=True)
    def get_daily_basic_data(self, stock_code: str, trade_time: Optional[datetime] = None, limit: int = 1200) -> Optional[pd.DataFrame]:
        """
        【新增】获取指定股票的历史每日基本面数据 (StockDailyBasic)。
        """
        try:
            qs = StockDailyBasic.objects.filter(stock__stock_code=stock_code)
            
            if trade_time:
                qs = qs.filter(trade_time__lte=trade_time.date())
            
            qs = qs.order_by('-trade_time')[:limit]

            fields_to_get = [
                'trade_time',
                'turnover_rate',
                'turnover_rate_f',
                'volume_ratio',
                'pe_ttm',
                'pb',
                'total_mv',
                'circ_mv'
            ]
            data = list(qs.values(*fields_to_get))
            if not data:
                return None
            df = pd.DataFrame.from_records(data)
            df['trade_time'] = pd.to_datetime(df['trade_time'], utc=True)
            df.set_index('trade_time', inplace=True)
            # 删除不需要的列，只保留核心数据
            df.drop(columns=['id', 'stock_id'], inplace=True, errors='ignore')
            return df
        except Exception as e:
            logger.error(f"获取 {stock_code} 的每日基本面数据时出错: {e}", exc_info=True)
            return None

    async def save_strategy_signals(self, signals_tuple: Tuple[List[TradingSignal], List[SignalPlaybookDetail]]) -> int:
        """
        【V401.0 异步事务修复版】
        - 核心修复: 解决了 'Atomic' object does not support the asynchronous context manager protocol 的错误。
        - 新工作流程:
          1. 在主异步函数中准备好所有数据。
          2. 定义一个内部的、完全同步的函数 `_save_all_sync`，其中包含 `with transaction.atomic()` 块。
          3. 在这个同步函数内部，使用所有操作的同步版本（如 .delete(), .bulk_create()）。
          4. 使用 `sync_to_async` 来异步地、安全地执行整个同步事务块。
        - 收益:
          - **事务正确性**: 保证了所有数据库操作在异步环境中依然享有事务的原子性。
          - **代码清晰**: 将同步的事务逻辑与异步的流程控制清晰地分离开。
        """
        if not signals_tuple or not signals_tuple[0]:
            print("调试信息: [DAO-SignalLog V401.0] 传入的信号元组为空，不执行任何操作。")
            return 0

        signals_to_process, details_to_create = signals_tuple
        
        # --- 代码修改开始 ---
        # [修改原因] 将所有数据库操作封装到一个同步函数中，以正确使用 transaction.atomic
        
        def _save_all_sync():
            """这个函数是完全同步的，它将在一个单独的线程中被异步执行。"""
            try:
                with transaction.atomic():
                    # --- 步骤1: 批量删除旧详情 (使用同步方法) ---
                    signal_lookup_keys = [
                        Q(stock_id=s.stock_id, trade_time=s.trade_time, timeframe=s.timeframe, strategy_name=s.strategy_name)
                        for s in signals_to_process
                    ]
                    
                    if not signal_lookup_keys:
                        return 0

                    existing_signals_pks = list(
                        TradingSignal.objects.filter(reduce(operator.or_, signal_lookup_keys)).values_list('pk', flat=True)
                    )
                    
                    if existing_signals_pks:
                        SignalPlaybookDetail.objects.filter(signal_id__in=existing_signals_pks).delete()

                    # --- 步骤2: 批量更新/创建主信号 (使用同步方法) ---
                    conflict_target = ['stock_id', 'trade_time', 'timeframe', 'strategy_name']
                    update_fields = ['signal_type', 'entry_score', 'risk_score', 'veto_votes', 'close_price', 'health_change_summary']
                    
                    TradingSignal.objects.bulk_create(
                        signals_to_process,
                        update_conflicts=True,
                        unique_fields=conflict_target,
                        update_fields=update_fields
                    )

                    # --- 步骤3: 批量创建新详情 (使用同步方法) ---
                    if details_to_create:
                        # 重新查询以获取主键
                        refreshed_signals_map = {
                            (s.stock_id, s.trade_time, s.timeframe, s.strategy_name): s.pk
                            for s in TradingSignal.objects.filter(reduce(operator.or_, signal_lookup_keys)).only('pk', 'stock_id', 'trade_time', 'timeframe', 'strategy_name')
                        }

                        valid_details = []
                        for detail in details_to_create:
                            key = (detail.signal.stock_id, detail.signal.trade_time, detail.signal.timeframe, detail.signal.strategy_name)
                            signal_pk = refreshed_signals_map.get(key)
                            if signal_pk:
                                detail.signal_id = signal_pk
                                valid_details.append(detail)
                        
                        if valid_details:
                            SignalPlaybookDetail.objects.bulk_create(valid_details, ignore_conflicts=True)
                
                return len(signals_to_process)

            except Exception as e:
                # 打印异常，但让上层处理返回
                print(f"错误: [DAO-SignalLog V401.0 - SYNC_BLOCK] 在同步事务块中发生异常: {e}")
                import traceback
                traceback.print_exc()
                # 抛出异常，以便上层可以捕获它
                raise

        try:
            # 使用 sync_to_async 异步执行整个同步事务块
            # thread_sensitive=True 对于数据库操作至关重要
            saved_count = await sync_to_async(_save_all_sync, thread_sensitive=True)()
            # print(f"调试信息: [DAO-SignalLog V401.0] 流程完成。成功处理 {saved_count} 条主信号。")
            return saved_count
        except Exception as e:
            # 捕获从同步块中抛出的异常
            print(f"错误: [DAO-SignalLog V401.0] 异步执行事务时捕获到异常: {e}")
            return 0

    # 筹码高级信息AdvancedChipMetrics
    async def get_advanced_chip_metrics_data(
        self,
        stock_code: str,
        trade_time_dt: Optional[pd.Timestamp],
        limit: int
    ) -> pd.DataFrame:
        """
        【V199.0 正常勤务版】
        - 核心升级: 移除所有用于调试的“超级探针”代码，恢复到最高效、最健壮的
                    生产状态。保留了直接获取所有字段的逻辑，以确保数据链路的
                    长期稳定性和免维护性。
        """
        # 1. 构建基础查询集
        queryset = AdvancedChipMetrics.objects.filter(stock__stock_code=stock_code)

        # 2. 应用日期过滤器
        if trade_time_dt and pd.notna(trade_time_dt):
            end_date = trade_time_dt.date()
            queryset = queryset.filter(trade_time__lte=end_date)

        # 3. 排序并限制数量
        queryset = queryset.order_by('-trade_time')[:limit]
        
        # 4. 直接获取所有字段，不再使用脆弱的 field_names 列表
        data_records = [item async for item in queryset.values()]

        # 5. 如果无数据，返回空DataFrame
        if not data_records:
            return pd.DataFrame()

        # 6. 转换为DataFrame并进行标准化处理
        df = pd.DataFrame.from_records(data_records)
        df['trade_time'] = pd.to_datetime(df['trade_time'], utc=True)
        df = df.set_index('trade_time')
        df = df.sort_index(ascending=True)

        return df

    async def get_daily_buy_signals(self, trade_date: date) -> List[TrendFollowStrategySignalLog]:
        """
        【V1.0 - 盘中引擎专用】
        获取指定交易日的所有日线级别最终买入信号。

        - 核心逻辑: 筛选 entry_signal=True, timeframe='D' 的记录。
        - 性能优化: 使用 .select_related('stock') 预先加载关联的股票信息，
                    避免 N+1 查询问题，大幅提升性能。
        - 异步执行: 使用 Django 的异步ORM接口 (`async for`) 执行查询。

        Args:
            trade_date (date): 需要查询的交易日期 (注意: 类型是 date)。

        Returns:
            List[TrendFollowStrategySignalLog]: 包含所有符合条件的信号日志模型实例列表。
        """
        logger.info(f"正在从数据库查询 {trade_date} 的日线级别买入信号...")
        try:
            # 1. 构建基础查询集
            queryset = TrendFollowStrategySignalLog.objects.filter(
                # 条件1: 必须是最终的买入信号
                entry_signal=True,
                # 条件2: 必须是日线('D')级别
                timeframe='D',
                # 条件3: 信号的生成日期必须是指定的交易日
                # __date 是Django ORM提供的强大查询，可直接匹配DateTimeField的日期部分
                trade_time__date=trade_date
            ).select_related('stock') # 【关键性能优化】

            # 2. 使用异步迭代器执行查询
            signals = [signal async for signal in queryset]

            logger.info(f"查询完成，共找到 {len(signals)} 条日线买入信号。")
            return signals

        except Exception as e:
            logger.error(f"查询日线买入信号时发生严重错误: {e}", exc_info=True)
            # 在生产环境中，发生错误时返回空列表是安全的做法
            return []




