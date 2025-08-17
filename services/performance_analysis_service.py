# 文件: services/performance_analysis_service.py
# 版本: V2.0 - 全景沙盘推演版
import logging
import pandas as pd
from asgiref.sync import sync_to_async
from typing import Tuple, Optional, Dict, List
from datetime import date, timedelta
from collections import defaultdict
import numpy as np

from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO
# --- 代码修改开始 ---
# [修改原因] 导入新的数据源模型 StrategyDailyState，这是我们分析的基础。
from stock_models.stock_analytics import StrategyDailyScore, StrategyScoreComponent, StrategyDailyState
# --- 代码修改结束 ---
from strategies.trend_following.utils import get_param_value, get_params_block
from utils.cache_manager import CacheManager
from utils.config_loader import load_strategy_config

logger = logging.getLogger(__name__)

class PerformanceAnalysisService:
    """
    【V2.0 全景沙盘推演版】
    - 核心升级: 新增 analyze_all_atomic_signals 方法，用于对全市场所有原子信号进行性能分析。
    - 逻辑内聚: 将原 PerformanceAnalyzer 的核心模拟逻辑内聚到本服务中，实现端到端的分析。
    - 数据驱动: 分析的数据源从计分信号扩展为更底层的 StrategyDailyState 记录。
    """
    def __init__(self, df_indicators: pd.DataFrame, score_details_df: pd.DataFrame, 
                 atomic_states: Dict, trigger_events: Dict, 
                 analysis_params: dict, scoring_params: dict):
        """
        初始化分析器
        :param df_indicators: 包含最终信号和K线数据的主DataFrame。
        :param score_details_df: 包含每日各信号得分详情的DataFrame。
        :param atomic_states: 包含所有原子状态的字典。
        :param trigger_events: 包含所有触发事件的字典。
        :param analysis_params: 性能分析模块的专属配置。
        :param scoring_params: 四层计分模型的配置，用于获取信号元数据。
        """
        self.df = df_indicators
        self.score_details_df = score_details_df
        self.atomic_states = atomic_states if atomic_states is not None else {}
        self.trigger_events = trigger_events if trigger_events is not None else {}
        self.analysis_params = analysis_params
        self.scoring_params = scoring_params
        if self.df is None or self.df.empty:
            raise ValueError("PerformanceAnalyzer 接收到的 df_indicators 为空。")
        
        # 从配置中获取分析参数
        self.look_forward_days = get_param_value(self.analysis_params.get('look_forward_days'), 20)
        self.profit_target_pct = get_param_value(self.analysis_params.get('profit_target_pct'), 0.15)
        self.stop_loss_pct = get_param_value(self.analysis_params.get('stop_loss_pct'), 0.07)
        
    # 一个全新的公共方法，作为新Celery任务的入口。
    async def analyze_all_atomic_signals(self) -> List[Dict]:
        """
        【V2.0 核心】执行全市场原子信号的性能分析。
        """
        print("-> [Service V2.0] 启动全景沙盘推演...")
        
        # 1. 从数据库获取所有需要分析的数据
        all_states_df, all_prices_df = await self._fetch_atomic_analysis_data_from_db()

        if all_states_df is None or all_prices_df is None:
            logger.warning("[Service V2.0] 无法获取原子状态或价格数据，分析终止。")
            return []

        # 2. 识别所有首次触发事件
        # 使用 groupby().apply() 来为每只股票独立计算首次触发日
        print("    -> [Service V2.0] 正在识别所有信号的首次触发事件...")
        first_trigger_events = all_states_df.groupby(['stock_code', 'signal_name'])['trade_date'].min().reset_index()
        
        # 3. 模拟每一次交易
        print(f"    -> [Service V2.0] 准备对 {len(first_trigger_events)} 个首次触发事件进行交易模拟...")
        trade_outcomes = []
        
        # 为了效率，将价格数据转换为字典 {stock_code: price_df}
        prices_by_stock = dict(iter(all_prices_df.groupby('stock_code')))

        for _, event in first_trigger_events.iterrows():
            stock_code = event['stock_code']
            signal_name = event['signal_name']
            entry_date = event['trade_date']
            
            price_df = prices_by_stock.get(stock_code)
            if price_df is None:
                continue

            outcome = self._analyze_single_trade_performance(entry_date, price_df)
            if outcome:
                trade_outcomes.append({
                    'signal_name': signal_name,
                    **outcome
                })
        
        # 4. 聚合结果并生成最终报告
        print(f"    -> [Service V2.0] 模拟完成，正在聚合 {len(trade_outcomes)} 条交易结果...")
        final_report = self._aggregate_atomic_results(trade_outcomes)
        print(f"-> [Service V2.0] 全景沙盘推演完成，生成 {len(final_report)} 条信号的性能报告。")
        return final_report

    async def _fetch_atomic_analysis_data_from_db(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        【V2.0 新增】为全景分析从数据库批量获取数据。
        """
        # 定义分析的时间范围，例如过去一年
        end_date = date.today()
        start_date = end_date - timedelta(days=365)
        
        print(f"    -> [DB Service V2.0] 正在加载全市场 {start_date} 到 {end_date} 的原子状态和价格数据...")

        # 1. 异步获取所有股票在时间范围内的原子状态
        states_qs = StrategyDailyState.objects.filter(
            daily_score__trade_date__range=(start_date, end_date)
        ).select_related('daily_score__stock')
        
        states_list = await sync_to_async(list)(
            states_qs.values('daily_score__stock__stock_code', 'daily_score__trade_date', 'signal_name')
        )
        if not states_list:
            logger.warning("在指定日期内未找到任何原子状态数据。")
            return None, None
        
        all_states_df = pd.DataFrame(states_list)
        all_states_df.rename(columns={
            'daily_score__stock__stock_code': 'stock_code',
            'daily_score__trade_date': 'trade_date'
        }, inplace=True)

        # 2. 异步获取所有相关股票的日线行情
        unique_stock_codes = all_states_df['stock_code'].unique().tolist()
        all_prices_df = await self.time_trade_dao.get_daily_data_for_stocks(
            unique_stock_codes, 
            start_date.strftime('%Y%m%d'), 
            (end_date + timedelta(days=self.look_forward_days)).strftime('%Y%m%d') # 多获取一些数据用于前瞻
        )

        if all_prices_df.empty:
            logger.warning("未能获取到任何相关股票的日线行情数据。")
            return None, None
            
        all_prices_df.rename(columns={'close': 'close_D', 'high': 'high_D', 'low': 'low_D'}, inplace=True)
        all_prices_df.index = all_prices_df.index.date

        return all_states_df, all_prices_df.reset_index().rename(columns={'index': 'trade_date'})

    def _analyze_single_trade_performance(self, entry_date: date, price_df: pd.DataFrame) -> Optional[Dict]:
        """
        【V2.0 迁移&重构】深度分析单次交易的性能表现 (从PerformanceAnalyzer迁移而来)。
        - 这是一个纯计算函数，不依赖self.df。
        """
        try:
            price_df_indexed = price_df.set_index('trade_date')
            entry_price = price_df_indexed.loc[entry_date, 'close_D']
            entry_idx = price_df_indexed.index.get_loc(entry_date)
            look_forward_df = price_df_indexed.iloc[entry_idx + 1 : entry_idx + 1 + self.look_forward_days]
        except (KeyError, IndexError):
            return None

        if look_forward_df.empty:
            return None

        target_price = entry_price * (1 + self.profit_target_pct)
        stop_price = entry_price * (1 - self.stop_loss_pct)

        max_profit_pct, days_to_max_profit = 0.0, 0
        max_drawdown_pct, days_to_max_drawdown = 0.0, 0
        exit_reason, exit_days, final_outcome = 'timeout', self.look_forward_days, 'timeout'

        for i, (date, row) in enumerate(look_forward_df.iterrows()):
            day_num = i + 1
            daily_max_profit = (row['high_D'] / entry_price) - 1
            if daily_max_profit > max_profit_pct:
                max_profit_pct = daily_max_profit
                days_to_max_profit = day_num

            daily_max_drawdown = (row['low_D'] / entry_price) - 1
            if daily_max_drawdown < max_drawdown_pct:
                max_drawdown_pct = daily_max_drawdown
                days_to_max_drawdown = day_num

            hit_target = row['high_D'] >= target_price
            hit_stop = row['low_D'] <= stop_price

            if hit_target:
                exit_reason, final_outcome, exit_days = 'profit_target', 'success', day_num
                break
            if hit_stop:
                exit_reason, final_outcome, exit_days = 'stop_loss', 'failure', day_num
                break
        
        return {
            'outcome': final_outcome, 'exit_days': exit_days,
            'max_profit_pct': max_profit_pct, 'max_drawdown_pct': max_drawdown_pct,
        }

    def _aggregate_atomic_results(self, trade_outcomes: List[Dict]) -> List[Dict]:
        """
        【V2.0 新增】聚合所有原子信号的交易结果。
        """
        if not trade_outcomes:
            return []
            
        outcomes_df = pd.DataFrame(trade_outcomes)
        score_map = self.scoring_params.get('score_type_map', {})
        
        signal_groups = outcomes_df.groupby('signal_name')

        analysis_results = []
        for signal_name, group_df in signal_groups:
            signal_meta = score_map.get(signal_name, {})
            signal_type = signal_meta.get('type', 'State' if signal_name.isupper() else 'Trigger').capitalize()

            total_triggers = len(group_df)
            success_count = (group_df['outcome'] == 'success').sum()
            
            win_rate = (success_count / total_triggers) * 100 if total_triggers > 0 else 0
            avg_max_profit = group_df['max_profit_pct'].mean() * 100
            avg_max_drawdown = group_df['max_drawdown_pct'].mean() * 100
            avg_exit_days = group_df['exit_days'].mean()
            
            analysis_results.append({
                'signal_name': signal_name,
                'cn_name': signal_meta.get('cn_name', signal_name),
                'type': signal_type,
                'triggers': int(total_triggers),
                'successes': int(success_count),
                'win_rate_pct': round(win_rate, 2),
                'avg_max_profit_pct': round(avg_max_profit, 2),
                'avg_max_drawdown_pct': round(avg_max_drawdown, 2),
                'avg_exit_days': round(avg_exit_days, 1),
            })
            
        analysis_results.sort(key=lambda x: x['win_rate_pct'], reverse=True)
        return analysis_results
    # --- 代码修改结束 ---

    async def run_analysis_for_stock(self, stock_code: str, start_date: Optional[str], end_date: Optional[str]) -> list:
        """
        【V2.1 兼容适配版】
        为单个股票执行基于数据库的回测分析 (分析最终买入信号)。
        - 核心修改: 在调用新版 PerformanceAnalyzer 时，传入空的原子状态和触发器，
                    使其只分析 score_details_df 中的计分信号，从而兼容旧的业务逻辑。
        """
        if not start_date:
            start_date = '1990-01-01'
        if not end_date:
            end_date = date.today().strftime('%Y-%m-%d')

        print(f"-> [Service V2.1] 开始分析股票 {stock_code} 的最终信号，日期范围: {start_date} 到 {end_date}")

        df_indicators, score_details_df = await self._fetch_analysis_data_from_db(stock_code, start_date, end_date)

        if df_indicators is None or df_indicators.empty or score_details_df is None:
            logger.warning(f"[{stock_code}] 获取并合并数据后，无有效数据可供分析。")
            return []

        if not get_param_value(self.analyzer_params.get('enabled'), False):
            logger.info("性能分析模块在配置文件中被禁用。")
            return []
            
        try:
            # 适配新版 PerformanceAnalyzer 的构造函数。
            # 我们只关心最终的计分信号，因此传入空的 atomic_states 和 trigger_events。
            # 新的 analyzer 会在 _identify_all_events 中只处理 score_details_df，完美实现我们的目标。
            from strategies.trend_following.performance_analyzer import PerformanceAnalyzer
            analyzer = PerformanceAnalyzer(
                df_indicators=df_indicators,
                score_details_df=score_details_df,
                atomic_states={},  # 对于只分析最终信号的旧任务，传入空字典
                trigger_events={}, # 对于只分析最终信号的旧任务，传入空字典
                analysis_params=self.analyzer_params,
                scoring_params=self.scoring_params
            )

            return analyzer.run_analysis()
        except Exception as e:
            logger.error(f"[{stock_code}] (兼容模式)性能分析器在执行过程中发生异常: {e}", exc_info=True)
            return []

    async def _fetch_analysis_data_from_db(self, stock_code: str, start_date: str, end_date: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        (旧方法) 从数据库中异步获取并构建分析所需的核心DataFrame。
        """
        print(f"    -> [DB Service V1.3] 正在为 {stock_code} 从数据库加载 {start_date} 到 {end_date} 的数据...")
        
        daily_price_df = await self.time_trade_dao.get_daily_data(stock_code, start_date.replace('-', ''), end_date.replace('-', ''))
        
        if daily_price_df.empty:
            logger.warning(f"[{stock_code}] 在指定日期内未找到日线行情数据。")
            return None, None
        daily_price_df.rename(columns={'close': 'close_D', 'high': 'high_D', 'low': 'low_D'}, inplace=True)
        daily_price_df.index = daily_price_df.index.date

        daily_scores_qs = StrategyDailyScore.objects.filter(
            stock__stock_code=stock_code,
            trade_date__range=(start_date, end_date)
        ).order_by('trade_date')
        daily_scores_list = await sync_to_async(list)(daily_scores_qs.values('trade_date', 'signal_type'))
        if not daily_scores_list:
            logger.warning(f"[{stock_code}] 在指定日期内未找到策略分数数据。")
            return None, None
        daily_scores_df = pd.DataFrame(daily_scores_list)
        daily_scores_df.set_index('trade_date', inplace=True)

        df_indicators = daily_price_df.join(daily_scores_df, how='inner')
        if df_indicators.empty:
            logger.warning(f"[{stock_code}] 日线行情与策略分数数据无法合并（日期不匹配）。")
            return None, None
        
        score_components_qs = StrategyScoreComponent.objects.filter(
            daily_score__stock__stock_code=stock_code,
            daily_score__trade_date__range=(start_date, end_date)
        ).select_related('daily_score')
        
        components_list = await sync_to_async(list)(
            score_components_qs.values('daily_score__trade_date', 'signal_name', 'score_value')
        )
        if not components_list:
            logger.warning(f"[{stock_code}] 在指定日期内未找到分数构成详情数据。")
            return df_indicators, pd.DataFrame()

        components_df = pd.DataFrame(components_list)
        components_df.rename(columns={'daily_score__trade_date': 'trade_date'}, inplace=True)
        
        score_details_df = components_df.pivot_table(
            index='trade_date',
            columns='signal_name',
            values='score_value',
            fill_value=0
        )
        
        print(f"    -> [DB Service V1.3] 数据加载与转换完成[{stock_code}]。行情: {len(df_indicators)}天, 信号详情: {len(score_details_df)}天。")
        return df_indicators, score_details_df


