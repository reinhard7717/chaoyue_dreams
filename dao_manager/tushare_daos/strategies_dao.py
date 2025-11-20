# dao_manager\tushare_daos\strategies_dao.py
import logging
from asgiref.sync import sync_to_async
from typing import Any, Dict, List, Optional, Tuple
from decimal import Decimal
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
from utils.model_helpers import get_advanced_chip_metrics_model_by_code
from stock_models.time_trade import StockCyqPerf, StockDailyBasic
from utils.cache_get import StrategyCacheGet
from utils.cache_manager import CacheManager
from stock_models.stock_analytics import TradingSignal, SignalPlaybookDetail, StrategyDailyScore, StrategyScoreComponent, StrategyDailyState
from functools import reduce
import operator

logger = logging.getLogger("dao")

class StrategiesDAO(BaseDAO):
    def __init__(self, cache_manager_instance: CacheManager):
        # 调用 super() 时，将 cache_manager_instance 传递进去
        super().__init__(cache_manager_instance=cache_manager_instance, model_class=None)
        self.stock_basic_dao = StockBasicInfoDao(cache_manager_instance)
        self.fund_flow_dao = FundFlowDao(cache_manager_instance)
        self.cache_manager = cache_manager_instance
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
    # --- 一个更通用的、基于字典列表的保存方法 ---
    async def save_monthly_trend_strategy_reports(self, reports_data: List[Dict[str, Any]]) -> int:
        """
        【V2.0 N+1查询优化版】根据标准化的字典列表，批量创建或更新月线趋势策略报告。
        - 核心优化: 彻底解决了N+1查询问题。通过一次性批量获取所有需要的StockInfo对象，
                      将原先在循环中的N次数据库查询优化为1次，大幅提升了性能。
        Args:
            reports_data (List[Dict[str, Any]]): 
                一个字典列表，每个字典都由策略的 _prepare_db_record 方法生成，
                包含了所有需要存入数据库的字段。
        Returns:
            int: 成功创建或更新的记录数量。
        """
        if not reports_data:
            return 0
        # --- 新增-修改-优化: 开始批量预取数据以解决N+1查询问题 ---
        # 1. 从所有报告中提取出所有不重复的股票代码
        all_stock_codes = list(set(report.get("stock_code") for report in reports_data if report.get("stock_code")))
        if not all_stock_codes:
            print("调试信息: [DAO] 所有报告数据均缺少 'stock_code'，无法继续。")
            return 0
        # 2. 一次性从数据库批量获取所有需要的StockInfo对象，并构建一个高效的查找字典
        stock_map = await self.stock_basic_dao.get_stocks_by_codes(all_stock_codes)
        print(f"调试信息: [DAO] 批量预获取了 {len(stock_map)} 个股票对象。")
        # --- 批量预取结束 ---
        data_list_for_db = []
        # 3. 遍历报告数据，此时从内存中的字典获取股票对象，而非查询数据库
        for report_dict in reports_data:
            stock_code = report_dict.get("stock_code")
            if not stock_code:
                continue
            # 新增-修改-优化: 从预先获取的 stock_map 中高效查找，避免了数据库查询
            stock_instance = stock_map.get(stock_code)
            if not stock_instance:
                print(f"调试信息: [DAO] 在预获取的股票映射中未找到: {stock_code}，跳过此条记录。")
                continue
            # 构造用于数据库操作的最终字典 (后续逻辑保持不变)
            db_ready_dict = report_dict.copy()
            db_ready_dict['stock'] = stock_instance
            del db_ready_dict['stock_code']
            for key, value in db_ready_dict.items():
                if pd.isna(value):
                    db_ready_dict[key] = None
            data_list_for_db.append(db_ready_dict)
        if not data_list_for_db:
            print("调试信息: [DAO] 所有记录都因数据问题被跳过，未执行数据库操作。")
            return 0
        # 调用底层的批量更新/插入方法
        result_stats = await self._save_all_to_db_native_upsert(
            model_class=MonthlyTrendStrategyReport,
            data_list=data_list_for_db,
            unique_fields=['stock', 'trade_time']
        )
        success_count = result_stats.get("创建/更新成功", 0)
        print(f"调试信息: [DAO] 批量保存完成。尝试: {len(reports_data)}, 成功: {success_count}")
        return success_count
    async def save_trend_follow_strategy_reports(self, reports_data: List[Dict[str, Any]]) -> int:
        """
        【V2.0 N+1查询优化版】根据标准化的字典列表，批量创建或更新趋势跟踪策略报告。
        - 核心优化: 彻底解决了N+1查询问题。通过一次性批量获取所有需要的StockInfo对象，
                      将原先在循环中的N次数据库查询优化为1次，大幅提升了性能。
        Args:
            reports_data (List[Dict[str, Any]]): 
                一个字典列表，每个字典都包含了所有需要存入数据库的字段。
        Returns:
            int: 成功创建或更新的记录数量。
        """
        if not reports_data:
            return 0
        # --- 新增-修改-优化: 开始批量预取数据以解决N+1查询问题 ---
        # 1. 从所有报告中提取出所有不重复的股票代码
        all_stock_codes = list(set(report.get("stock_code") for report in reports_data if report.get("stock_code")))
        if not all_stock_codes:
            print("调试信息: [DAO-TrendFollow] 所有报告数据均缺少 'stock_code'，无法继续。")
            return 0
        # 2. 一次性从数据库批量获取所有需要的StockInfo对象，并构建一个高效的查找字典
        stock_map = await self.stock_basic_dao.get_stocks_by_codes(all_stock_codes)
        print(f"调试信息: [DAO-TrendFollow] 批量预获取了 {len(stock_map)} 个股票对象。")
        # --- 批量预取结束 ---
        data_list_for_db = []
        # 3. 遍历报告数据，此时从内存中的字典获取股票对象
        for report_dict in reports_data:
            stock_code = report_dict.get("stock_code")
            if not stock_code:
                continue
            # 新增-修改-优化: 从预先获取的 stock_map 中高效查找，避免了数据库查询
            stock_instance = stock_map.get(stock_code)
            if not stock_instance:
                print(f"调试信息: [DAO-TrendFollow] 在预获取的股票映射中未找到: {stock_code}，跳过。")
                continue
            # 构造用于数据库操作的最终字典 (后续逻辑保持不变)
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
        result_stats = await self._save_all_to_db_native_upsert(
            model_class=TrendFollowStrategyReport,
            data_list=data_list_for_db,
            unique_fields=['stock', 'trade_time']
        )
        success_count = result_stats.get("创建/更新成功", 0)
        return success_count
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
        #    【关键修改】在应用窗口函数前，先用传入的 stock_codes 列表进行过滤
        #    这会极大地减少窗口函数需要处理的数据量，是本次性能优化的核心。
        #    通过 stock__stock_code__in 实现跨表查询。
        base_queryset = MonthlyTrendStrategyReport.objects.filter(stock__stock_code__in=stock_codes)
        ranked_reports = base_queryset.annotate(
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
        # 调试信息：打印查询结果数量
        print(f"查询完成，共找到 {latest_reports_queryset.count()} 条符合条件的最新报告。")
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
        # 将所有计算字段（包括展示用的字符串）在同一个 .annotate() 中完成
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
        获取指定股票的历史每日基本面数据 (StockDailyBasic)。
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
    async def save_strategy_signals(self, records_tuple: Tuple[List, List, List, List, List]) -> int:
        """
        【V507.0 全景沙盤版】
        - 核心升级: 接收并处理一个包含五类对象的元组，增加了对 StrategyDailyState 的保存逻辑。
        """
        if not records_tuple:
            return 0
        # 解包五元组，增加 daily_states 列表。
        signals, signal_details, daily_scores, score_components, daily_states = records_tuple
        if not signals and not daily_scores:
            print("调试信息: [DAO-V507.0] 传入的信号和每日分数均为空，不执行任何操作。")
            return 0
        # --- Part A & B: 数据清洗 ---
        cleaned_signals = []
        if signals:
            numeric_fields = ['entry_score', 'risk_score', 'close_price']
            for signal_obj in signals:
                for field_name in numeric_fields:
                    value = getattr(signal_obj, field_name)
                    is_nan = (isinstance(value, float) and np.isnan(value)) or \
                             (isinstance(value, Decimal) and value.is_nan())
                    if is_nan:
                        setattr(signal_obj, field_name, None)
                cleaned_signals.append(signal_obj)
        cleaned_daily_scores = []
        if daily_scores:
            numeric_fields = ['offensive_score', 'risk_score', 'final_score']
            for score_obj in daily_scores:
                for field_name in numeric_fields:
                    value = getattr(score_obj, field_name)
                    is_nan = (isinstance(value, float) and np.isnan(value)) or \
                             (isinstance(value, Decimal) and value.is_nan())
                    if is_nan:
                        setattr(score_obj, field_name, None)
                cleaned_daily_scores.append(score_obj)
        def _save_all_sync():
            """这个同步函数将所有数据库操作包裹在一个事务中。"""
            try:
                with transaction.atomic():
                    # --- Section 1 & 2: 处理 TradingSignal 和 StrategyDailyScore ---
                    if cleaned_signals:
                        signal_lookup_keys = [
                            Q(stock_id=s.stock_id, trade_time=s.trade_time, timeframe=s.timeframe, strategy_name=s.strategy_name)
                            for s in cleaned_signals
                        ]
                        if not signal_lookup_keys: return 0 # 增加健壮性检查
                        existing_signals_pks = list(
                            TradingSignal.objects.filter(reduce(operator.or_, signal_lookup_keys)).values_list('pk', flat=True)
                        )
                        if existing_signals_pks:
                            SignalPlaybookDetail.objects.filter(signal_id__in=existing_signals_pks).delete()
                        existing_signals_map = {
                            (s.stock_id, s.trade_time, s.timeframe, s.strategy_name): s
                            for s in TradingSignal.objects.filter(reduce(operator.or_, signal_lookup_keys))
                        }
                        signals_to_update, signals_to_create = [], []
                        update_fields = ['signal_type', 'entry_score', 'risk_score', 'final_score', 'veto_votes', 'close_price', 'health_change_summary']
                        for signal_obj in cleaned_signals:
                            key = (signal_obj.stock_id, signal_obj.trade_time, signal_obj.timeframe, signal_obj.strategy_name)
                            if key in existing_signals_map:
                                existing_signal = existing_signals_map[key]
                                for field in update_fields:
                                    setattr(existing_signal, field, getattr(signal_obj, field))
                                signals_to_update.append(existing_signal)
                            else:
                                signals_to_create.append(signal_obj)
                        if signals_to_update:
                            TradingSignal.objects.bulk_update(signals_to_update, update_fields)
                        if signals_to_create:
                            TradingSignal.objects.bulk_create(signals_to_create)
                        if signal_details:
                            refreshed_signals_map = {
                                (s.stock_id, s.trade_time, s.timeframe, s.strategy_name): s.pk
                                for s in TradingSignal.objects.filter(reduce(operator.or_, signal_lookup_keys))
                            }
                            valid_details = []
                            for detail in signal_details:
                                key = (detail.signal.stock_id, detail.signal.trade_time, detail.signal.timeframe, detail.signal.strategy_name)
                                if key in refreshed_signals_map:
                                    detail.signal_id = refreshed_signals_map[key]
                                    valid_details.append(detail)
                            if valid_details:
                                SignalPlaybookDetail.objects.bulk_create(valid_details, ignore_conflicts=True)
                    refreshed_scores_map = {} # 在Section 2和3之间共享
                    if cleaned_daily_scores:
                        score_lookup_keys = [
                            Q(stock_id=s.stock_id, trade_date=s.trade_date, strategy_name=s.strategy_name)
                            for s in cleaned_daily_scores
                        ]
                        if not score_lookup_keys: return 0 # 增加健壮性检查
                        existing_scores_pks = list(
                            StrategyDailyScore.objects.filter(reduce(operator.or_, score_lookup_keys)).values_list('pk', flat=True)
                        )
                        if existing_scores_pks:
                            StrategyScoreComponent.objects.filter(daily_score_id__in=existing_scores_pks).delete()
                            # 在删除旧分数成分的同时，也删除旧的每日状态记录，确保数据一致性。
                            StrategyDailyState.objects.filter(daily_score_id__in=existing_scores_pks).delete()
                        existing_scores_map = {
                            (s.stock_id, s.trade_date, s.strategy_name): s
                            for s in StrategyDailyScore.objects.filter(reduce(operator.or_, score_lookup_keys))
                        }
                        scores_to_update, scores_to_create = [], []
                        update_fields = ['offensive_score', 'risk_score', 'final_score', 'signal_type', 'score_details_json', 'positional_score', 'dynamic_score', 'composite_score']
                        for score_obj in cleaned_daily_scores:
                            key = (score_obj.stock_id, score_obj.trade_date, score_obj.strategy_name)
                            if key in existing_scores_map:
                                existing_score = existing_scores_map[key]
                                for field in update_fields:
                                    setattr(existing_score, field, getattr(score_obj, field))
                                scores_to_update.append(existing_score)
                            else:
                                scores_to_create.append(score_obj)
                        if scores_to_update:
                            StrategyDailyScore.objects.bulk_update(scores_to_update, update_fields)
                        if scores_to_create:
                            StrategyDailyScore.objects.bulk_create(scores_to_create)
                        # 刷新分数ID映射，供后续的成分和状态使用
                        refreshed_scores_map = {
                            (s.stock_id, s.trade_date, s.strategy_name): s.pk
                            for s in StrategyDailyScore.objects.filter(reduce(operator.or_, score_lookup_keys))
                        }
                        if score_components:
                            valid_components = []
                            for comp in score_components:
                                key = (comp.daily_score.stock_id, comp.daily_score.trade_date, comp.daily_score.strategy_name)
                                if key in refreshed_scores_map:
                                    comp.daily_score_id = refreshed_scores_map[key]
                                    valid_components.append(comp)
                            if valid_components:
                                StrategyScoreComponent.objects.bulk_create(valid_components, ignore_conflicts=True)
                    # --- Section 3: 处理 StrategyDailyState ---
                    if daily_states:
                        valid_states = []
                        for state in daily_states:
                            # 使用上面已经刷新过的 refreshed_scores_map
                            key = (state.daily_score.stock_id, state.daily_score.trade_date, state.daily_score.strategy_name)
                            if key in refreshed_scores_map:
                                state.daily_score_id = refreshed_scores_map[key]
                                valid_states.append(state)
                        if valid_states:
                            StrategyDailyState.objects.bulk_create(valid_states, ignore_conflicts=True)
                return len(cleaned_signals) + len(cleaned_daily_scores)
            except Exception as e:
                print(f"错误: [DAO-V507.0 - SYNC_BLOCK] 在同步事务块中发生异常: {e}")
                import traceback
                traceback.print_exc()
                raise
        try:
            saved_count = await sync_to_async(_save_all_sync, thread_sensitive=True)()
            return saved_count
        except Exception as e:
            print(f"错误: [DAO-V507.0] 异步执行事务时捕获到异常: {e}")
            return 0
    # 筹码高级信息AdvancedChipMetrics
    async def get_advanced_chip_metrics_data(
        self,
        stock_code: str,
        trade_time_dt: Optional[pd.Timestamp],
        limit: int
    ) -> pd.DataFrame:
        """
        【V201.0 字段精确选择版】
        - 核心修复: 解决了因 `queryset.values()` 获取全部字段（包括'id', 'stock_id'）而导致的
                      下游 `DataFrame.join` 操作列名冲突的严重错误。
        - 解决方案: 不再使用无参数的 `.values()`，而是动态获取模型的所有非关系字段名，并从中
                      排除 'id' 和 'stock_id'，从而精确地只查询业务需要的字段。
        - 收益: 彻底根除 `Indexes have overlapping values` 异常，同时减少了不必要的数据传输，
                  提升了查询效率和代码的健壮性。
        """
        # 1. 根据股票代码动态获取对应的分表模型
        MetricsModel = get_advanced_chip_metrics_model_by_code(stock_code)
        # 2. 使用动态获取的模型构建基础查询集
        queryset = MetricsModel.objects.filter(stock__stock_code=stock_code)
        # 3. 应用日期过滤器
        if trade_time_dt and pd.notna(trade_time_dt):
            end_date = trade_time_dt.date()
            queryset = queryset.filter(trade_time__lte=end_date)
        # 4. 排序并限制数量
        queryset = queryset.order_by('-trade_time')[:limit]
        # --- 新增-修改-优化: 开始精确选择字段 ---
        # 5.1 动态获取模型的所有非关系字段名
        all_model_fields = [f.name for f in MetricsModel._meta.get_fields() if not f.is_relation]
        # 5.2 从字段列表中排除 'id' 和 'stock_id'，避免冲突
        fields_to_fetch = [field for field in all_model_fields if field not in ['id', 'stock_id']]
        # 5.3 使用明确的字段列表进行查询
        data_records = [item async for item in queryset.values(*fields_to_fetch)]
        # --- 精确选择字段结束 ---
        # 6. 如果无数据，返回空DataFrame
        if not data_records:
            return pd.DataFrame()
        # 7. 转换为DataFrame并进行标准化处理
        df = pd.DataFrame.from_records(data_records)
        df['trade_time'] = pd.to_datetime(df['trade_time'], utc=True)
        df = df.set_index('trade_time')
        df = df.sort_index(ascending=True)
        return df
    async def get_daily_buy_signals(self, trade_date: date) -> List['TrendFollowStrategySignalLog']:
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
    async def get_latest_daily_data_for_stocks(self, stock_codes: List[str], end_date: str) -> Dict[str, pd.DataFrame]:
        """
        【V1.0 新增】使用窗口函数高效获取一批股票在指定日期或之前的最新日线行情数据。
        这是解决“获取每个分组最新N条记录”问题的最佳实践，性能极高。
        Args:
            stock_codes (List[str]): 股票代码列表。
            end_date (str): 截止日期，格式为 'YYYYMMDD' 或 'YYYY-MM-DD'。
        Returns:
            Dict[str, pd.DataFrame]: 一个字典，键是股票代码，值是包含该股票最新一条
                                     日线数据的DataFrame。如果某股票无数据，则字典中
                                     不会包含该股票的键。
        """
        print(f"DEBUG: [DAO] 正在为 {len(stock_codes)} 只股票获取截至 {end_date} 的最新日线数据...")
        if not stock_codes:
            return {}
        # 1. 定义窗口函数
        #    - PARTITION BY stock_id: 按股票ID分组
        #    - ORDER BY trade_time DESC: 在每个组内按交易日倒序排列
        window = Window(
            expression=RowNumber(),
            partition_by=[F('stock_id')],
            order_by=F('trade_time').desc()
        )
        # 2. 构建查询
        #    - 首先过滤出所有相关股票在截止日期之前的所有数据
        #    - 然后使用窗口函数为每个组内的记录进行排名
        ranked_data_qs = StockDailyBasic.objects.filter(
            stock__stock_code__in=stock_codes,
            trade_time__lte=end_date
        ).annotate(row_number=window)
        # 3. 从排名后的结果中只筛选出最新的记录 (row_number=1)
        #    使用 .values() 获取所需字段，这比获取完整的模型实例更高效
        latest_data_values = await sync_to_async(list)(
            ranked_data_qs.filter(row_number=1).values(
                'stock__stock_code', 'trade_time', 'close', 'open', 'high', 'low', 'vol'
            )
        )
        if not latest_data_values:
            logger.warning(f"未能为股票列表在 {end_date} 或之前找到任何日线数据。")
            return {}
        # 4. 将查询结果组织成目标格式：{stock_code: DataFrame}
        results_map = {}
        for item in latest_data_values:
            stock_code = item.pop('stock__stock_code') # 弹出stock_code作为键
            # 将单个记录的字典转换为DataFrame
            df = pd.DataFrame([item])
            df['trade_time'] = pd.to_datetime(df['trade_time'])
            df.set_index('trade_time', inplace=True)
            results_map[stock_code] = df
        print(f"DEBUG: [DAO] 成功获取了 {len(results_map)} 只股票的最新日线数据。")
        return results_map


