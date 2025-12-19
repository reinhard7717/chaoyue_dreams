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

# 导入新的数据源模型 StrategyDailyState，这是我们分析的基础。
from stock_models.stock_analytics import StrategyDailyScore, StrategyScoreComponent, StrategyDailyState

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
    def __init__(self, cache_manager: CacheManager):
        """
        【V2.2 构造函数修复版】
        """
        self.time_trade_dao = StockTimeTradeDAO(cache_manager)
        unified_config_path = 'config/trend_follow_strategy.json'
        self.unified_config = load_strategy_config(unified_config_path)
        self.analyzer_params = get_params_block({'unified_config': self.unified_config}, 'performance_analysis_params')
        self.scoring_params = get_params_block({'unified_config': self.unified_config}, 'four_layer_scoring_params')
        self.look_forward_days = get_param_value(self.analyzer_params.get('look_forward_days'), 20)
        self.profit_target_pct = get_param_value(self.analyzer_params.get('profit_target_pct'), 0.15)
        self.stop_loss_pct = get_param_value(self.analyzer_params.get('stop_loss_pct'), 0.07)
    @staticmethod
    def _simulate_trade_outcome(
        entry_date: date, 
        price_df: pd.DataFrame, 
        look_forward_days: int, 
        profit_target_pct: float, 
        stop_loss_pct: float,
        is_offensive: bool = True
    ) -> Optional[Dict]:
        """
        【权威交易模拟引擎 V1.0】
        根据给定的入场点和参数，模拟未来N天的交易表现。
        这是一个纯函数，不依赖任何实例状态。
        Args:
            entry_date (date): 信号触发日期 (T日)。
            price_df (pd.DataFrame): 包含 'open_D', 'high_D', 'low_D' 列的价格数据，索引为datetime。
            look_forward_days (int): 向前看的天数。
            profit_target_pct (float): 止盈目标百分比 (例如 0.15)。
            stop_loss_pct (float): 止损目标百分比 (例如 0.07)。
            is_offensive (bool): 信号类型。True为进攻型(看涨)，False为防御型(看跌)。
        Returns:
            Optional[Dict]: 包含交易结果详情的字典，如果无法评估则返回None。
        """
        try:
            # 确保索引是 datetime 类型
            if not isinstance(price_df.index, pd.DatetimeIndex):
                price_df.index = pd.to_datetime(price_df.index)
            # 找到T+1日
            entry_datetime = pd.to_datetime(entry_date)
            future_dates = price_df.index[price_df.index > entry_datetime]
            if future_dates.empty:
                return None
            trade_day = future_dates[0]
            # 获取入场价和回测窗口
            entry_price = price_df.loc[trade_day, 'open_D']
            if pd.isna(entry_price) or entry_price <= 0:
                return None
            look_forward_df = price_df.loc[trade_day:].head(look_forward_days)
        except (KeyError, IndexError):
            return None
        if look_forward_df.empty:
            return None
        # 根据信号类型定义止盈止损价
        if is_offensive:
            target_price = entry_price * (1 + profit_target_pct)
            stop_price = entry_price * (1 - stop_loss_pct)
        else: # 防御型信号
            target_price = entry_price * (1 - stop_loss_pct) # 成功是下跌
            stop_price = entry_price * (1 + profit_target_pct) # 失败是上涨
        outcome, exit_days = 'timeout', len(look_forward_df)
        max_profit_pct = 0.0
        max_drawdown_pct = 0.0
        for i, row in enumerate(look_forward_df.itertuples()):
            day_num = i + 1
            high_price = row.high_D
            low_price = row.low_D
            if is_offensive:
                max_profit_pct = max(max_profit_pct, (high_price / entry_price) - 1)
                max_drawdown_pct = min(max_drawdown_pct, (low_price / entry_price) - 1)
                if high_price >= target_price:
                    outcome, exit_days = 'success', day_num
                    break
                if low_price <= stop_price:
                    outcome, exit_days = 'failure', day_num
                    break
            else: # 防御型信号
                max_profit_pct = max(max_profit_pct, 1 - (low_price / entry_price)) # 利润是向下跌幅
                max_drawdown_pct = min(max_drawdown_pct, (high_price / entry_price) - 1) # 回撤是向上涨幅
                if low_price <= target_price:
                    outcome, exit_days = 'success', day_num
                    break
                if high_price >= stop_price:
                    outcome, exit_days = 'failure', day_num
                    break
        return {
            'outcome': outcome, 'exit_days': exit_days,
            'max_profit_pct': max_profit_pct, 'max_drawdown_pct': max_drawdown_pct,
        }
    # 实现 Celery Map 任务所需的核心业务逻辑。
    async def analyze_atomic_signals_for_single_stock(self, stock_code: str) -> List[Dict]:
        """
        【V1.0 新增】为单只股票执行原子信号性能分析 (Map任务核心)。
        """
        # print(f"    -> [Service-Map] 开始分析 {stock_code}...")
        # 1. 获取该股票所需的数据
        stock_states_df, price_df = await self._fetch_data_for_single_stock(stock_code)
        if stock_states_df is None or price_df is None:
            # print(f"    -> [Service-Map] {stock_code} 数据不足，跳过。")
            return []
        # 2. 评估逻辑 (从 analyze_all_atomic_signals 中剥离)
        trade_outcomes = []
        try:
            pivoted_df = stock_states_df.pivot_table(
                index='trade_date', columns='signal_name', aggfunc='size', fill_value=0
            ).astype(bool)
        except Exception:
            return []
        for signal_name in pivoted_df.columns:
            signal_series = pivoted_df[signal_name].sort_index()
            signal_meta = self.scoring_params.get('score_type_map', {}).get(signal_name, {})
            signal_type = signal_meta.get('type', 'positional').lower()
            event_dates = []
            if signal_type in ['trigger', 'composite', 'playbook']:
                event_dates = signal_series[signal_series].index.tolist()
            else:
                is_first_day = signal_series & ~signal_series.shift(1).fillna(False)
                event_dates = is_first_day[is_first_day].index.tolist()
            for entry_date in event_dates:
                outcome = None
                if signal_type == 'risk':
                    outcome = self._evaluate_defensive_signal(entry_date, price_df)
                else:
                    outcome = self._evaluate_offensive_signal(entry_date, price_df)
                if outcome:
                    trade_outcomes.append({
                        'signal_name': signal_name,
                        'signal_type_role': signal_type,
                        **outcome
                    })
        # 3. 聚合单只股票的结果，并为 Reduce 任务准备好加权数据
        if not trade_outcomes:
            return []
        outcomes_df = pd.DataFrame(trade_outcomes)
        signal_groups = outcomes_df.groupby('signal_name')
        stock_results = []
        for signal_name, group_df in signal_groups:
            signal_meta = self.scoring_params.get('score_type_map', {}).get(signal_name, {})
            total_triggers = len(group_df)
            success_count = (group_df['outcome'] == 'success').sum()
            # 计算平均值
            avg_max_profit = group_df['max_profit_pct'].mean()
            avg_max_drawdown = group_df['max_drawdown_pct'].mean()
            avg_exit_days = group_df['exit_days'].mean()
            stock_results.append({
                'signal_name': signal_name,
                'cn_name': signal_meta.get('cn_name', signal_name),
                # 直接使用从 score_map 中获取的真实类型
                'type': signal_meta.get('type', 'unknown'),
                'triggers': total_triggers,
                'successes': success_count,
                # [核心] 为Reduce任务传递加权值
                'weighted_max_profit': avg_max_profit * total_triggers,
                'weighted_max_drawdown': avg_max_drawdown * total_triggers,
                'weighted_exit_days': avg_exit_days * total_triggers,
            })
        # print(f"    -> [Service-Map] 完成 {stock_code} 分析，返回 {len(stock_results)} 条信号统计。")
        return stock_results
    async def _fetch_data_for_single_stock(self, stock_code: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        【V1.0 新增】为单只股票获取原子状态和价格数据。
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=365)
        states_qs = StrategyDailyState.objects.filter(
            daily_score__stock__stock_code=stock_code,
            daily_score__trade_date__range=(start_date, end_date)
        )
        states_list = await sync_to_async(list)(
            states_qs.values('daily_score__trade_date', 'signal_name')
        )
        if not states_list:
            return None, None
        stock_states_df = pd.DataFrame(states_list)
        stock_states_df.rename(columns={'daily_score__trade_date': 'trade_date'}, inplace=True)
        stock_states_df['trade_date'] = pd.to_datetime(stock_states_df['trade_date'])
        price_df_raw = await self.time_trade_dao.get_daily_data_for_stocks(
            [stock_code],
            start_date.strftime('%Y%m%d'),
            (end_date + timedelta(days=self.look_forward_days)).strftime('%Y%m%d')
        )
        if price_df_raw.empty:
            return None, None
        price_df_raw.rename(columns={'open': 'open_D', 'close': 'close_D', 'high': 'high_D', 'low': 'low_D'}, inplace=True)
        price_df_raw.rename(columns={'trade_time': 'trade_date'}, inplace=True)
        price_df_raw['trade_date'] = pd.to_datetime(price_df_raw['trade_date'])
        # 设置索引以便快速查找
        price_df = price_df_raw.set_index('trade_date').sort_index()
        return stock_states_df, price_df
    # 一个全新的公共方法，作为新Celery任务的入口。
    async def analyze_all_atomic_signals(self) -> List[Dict]:
        """
        【V4.2 状态/事件区分版】执行全市场原子信号的性能分析。
        """
        print("-> [Service V4.2] 启动全景沙盘推演 (状态/事件区分版)...")
        all_states_df, all_prices_df = await self._fetch_atomic_analysis_data_from_db()
        if all_states_df is None or all_prices_df is None or all_states_df.empty or all_prices_df.empty:
            logger.warning("[Service V4.2] 无法获取原子状态或价格数据，分析终止。")
            return []
        print("    -> [Service V4.2] 数据加载完成，开始按角色和性质识别并评估所有信号...")
        trade_outcomes = []
        prices_by_stock = dict(iter(all_prices_df.groupby('stock_code')))
        for stock_code, stock_states_df in all_states_df.groupby('stock_code'):
            price_df_unindexed = prices_by_stock.get(stock_code)
            if price_df_unindexed is None or price_df_unindexed.empty:
                continue
            price_df = price_df_unindexed.set_index('trade_date').sort_index()
            try:
                pivoted_df = stock_states_df.pivot_table(
                    index='trade_date', columns='signal_name', aggfunc='size', fill_value=0
                ).astype(bool)
            except Exception:
                continue
            for signal_name in pivoted_df.columns:
                signal_series = pivoted_df[signal_name].sort_index()
                signal_meta = self.scoring_params.get('score_type_map', {}).get(signal_name, {})
                signal_type = signal_meta.get('type', 'positional').lower()
                event_dates = []
                # 根据信号的性质（状态 vs 事件）应用不同的评估协议
                if signal_type in ['trigger', 'composite', 'playbook']:
                    # 性质：事件。评估每一次发生。
                    event_dates = signal_series[signal_series].index.tolist()
                else:
                    # 性质：状态。只评估首次进入。
                    is_first_day = signal_series & ~signal_series.shift(1).fillna(False)
                    event_dates = is_first_day[is_first_day].index.tolist()
                for entry_date in event_dates:
                    outcome = None
                    # 根据角色调用评估函数 (此逻辑保持不变)
                    if signal_type == 'risk':
                        outcome = self._evaluate_defensive_signal(entry_date, price_df)
                    else:
                        outcome = self._evaluate_offensive_signal(entry_date, price_df)
                    if outcome:
                        trade_outcomes.append({
                            'signal_name': signal_name,
                            'signal_type_role': signal_type,
                            **outcome
                        })
        # 这条日志现在应该能被正确打印，并且数量大于0
        print(f"    -> [Service V4.2] 模拟完成，正在聚合 {len(trade_outcomes)} 条评估结果...")
        final_report = self._aggregate_atomic_results(trade_outcomes)
        print(f"-> [Service V4.2] 全景沙盘推演完成，生成 {len(final_report)} 条信号的性能报告。")
        return final_report
    def _evaluate_offensive_signal(self, entry_date: date, price_df: pd.DataFrame) -> Optional[Dict]:
        """
        【V4.3 逻辑统一版】评估“进攻型”信号的表现 (看涨)。
        现在是 _simulate_trade_outcome 的一个简单包装器。
        """
        # 调用统一的静态模拟函数，消除冗余代码。
        return self._simulate_trade_outcome(
            entry_date=entry_date,
            price_df=price_df,
            look_forward_days=self.look_forward_days,
            profit_target_pct=self.profit_target_pct,
            stop_loss_pct=self.stop_loss_pct,
            is_offensive=True
        )
    def _evaluate_defensive_signal(self, entry_date: date, price_df: pd.DataFrame) -> Optional[Dict]:
        """
        【V4.3 逻辑统一版】评估“防御/风险型”信号的表现 (看跌)。
        现在是 _simulate_trade_outcome 的一个简单包装器。
        """
        # 调用统一的静态模拟函数，消除冗余代码。
        return self._simulate_trade_outcome(
            entry_date=entry_date,
            price_df=price_df,
            look_forward_days=self.look_forward_days,
            profit_target_pct=self.profit_target_pct,
            stop_loss_pct=self.stop_loss_pct,
            is_offensive=False
        )
    async def _fetch_atomic_analysis_data_from_db(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        【V4.2 原生日期标准化版】为全景分析从数据库批量获取数据。
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=365)
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
        # 强制转换为Pandas原生datetime64[ns]类型，不再使用.dt.date
        all_states_df['trade_date'] = pd.to_datetime(all_states_df['trade_date'])
        print(f"调试信息: all_states_df['trade_date'] 的类型是 {all_states_df['trade_date'].dtype}")
        unique_stock_codes = all_states_df['stock_code'].unique().tolist()
        if len(unique_stock_codes) == 0:
            logger.warning("原子状态数据中未能提取出有效的股票代码。")
            return None, None
        all_prices_df = await self.time_trade_dao.get_daily_data_for_stocks(
            unique_stock_codes,
            start_date.strftime('%Y%m%d'),
            (end_date + timedelta(days=self.look_forward_days)).strftime('%Y%m%d')
        )
        if all_prices_df.empty:
            logger.warning("未能获取到任何相关股票的日线行情数据。")
            return None, None
        all_prices_df.rename(columns={'open': 'open_D', 'close': 'close_D', 'high': 'high_D', 'low': 'low_D'}, inplace=True)
        all_prices_df.rename(columns={'trade_time': 'trade_date'}, inplace=True)
        # 同样，将价格数据的日期也强制转换为datetime64[ns]类型
        all_prices_df['trade_date'] = pd.to_datetime(all_prices_df['trade_date'])
        print(f"调试信息: all_prices_df['trade_date'] 的类型是 {all_prices_df['trade_date'].dtype}")
        return all_states_df, all_prices_df
    def _analyze_single_trade_performance(self, entry_date: date, price_df: pd.DataFrame) -> Optional[Dict]:
        """
        【V2.1 T+1交易修正版】深度分析单次交易的性能表现。
        """
        try:
            # 确保 price_df 的索引是日期类型，以便进行定位
            price_df_indexed = price_df.set_index('trade_date').sort_index()
            # 找到T日的索引位置
            entry_idx = price_df_indexed.index.get_loc(entry_date)
            # 检查是否存在T+1日的数据
            if entry_idx + 1 >= len(price_df_indexed):
                return None
            # 获取T+1日的数据行
            trade_day_row = price_df_indexed.iloc[entry_idx + 1]
            # 使用T+1日的开盘价作为买入价
            # 注意：DAO层返回的原始列名是 open, high, low, close，服务层重命名为了带 _D 后缀的
            # 这里需要确保 price_df 中有 open_D 列，或者使用原始的 open 列
            # 假设 _fetch_atomic_analysis_data_from_db 已经提供了 open_D
            if 'open_D' not in trade_day_row:
                 # 如果没有 open_D，可能是因为 _fetch_atomic_analysis_data_from_db 没有重命名 open 列
                 # 我们在这里做一个兼容处理
                 if 'open' in trade_day_row:
                     trade_day_row['open_D'] = trade_day_row['open']
                 else:
                     # 如果连 open 都没有，则无法继续
                     return None
            entry_price = trade_day_row['open_D']
            # 回测观察窗口从T+1日开始
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
        【V4.0 角色扮演版 - 聚合器】
        聚合所有原子信号的交易结果，并体现角色差异。
        - 核心逻辑:
          1. 按信号名称分组。
          2. 对每个组，根据其角色（进攻/风险）计算相应的统计指标。
          3. 将结果格式化为统一的字典结构，字段名与 AtomicSignalPerformance 模型匹配。
        """
        if not trade_outcomes:
            return []
        outcomes_df = pd.DataFrame(trade_outcomes)
        score_map = self.scoring_params.get('score_type_map', {})
        # 按信号名称分组
        signal_groups = outcomes_df.groupby('signal_name')
        analysis_results = []
        for signal_name, group_df in signal_groups:
            # 从分组的第一行获取角色信息，因为同一信号的角色是固定的
            role = group_df['signal_type_role'].iloc[0]
            signal_meta = score_map.get(signal_name, {})
            total_triggers = len(group_df)
            success_count = (group_df['outcome'] == 'success').sum()
            win_rate = (success_count / total_triggers) * 100 if total_triggers > 0 else 0
            # 这里的 profit 和 drawdown 的含义是根据角色动态决定的
            # _evaluate_... 方法已经确保了这一点
            avg_max_profit = group_df['max_profit_pct'].mean() * 100
            avg_max_drawdown = group_df['max_drawdown_pct'].mean() * 100
            avg_exit_days = group_df['exit_days'].mean()
            # 最终返回的字典结构是统一的，与数据库模型字段对应
            analysis_results.append({
                'signal_name': signal_name,
                'cn_name': signal_meta.get('cn_name', signal_name),
                'type': role.capitalize(), # 存储信号的角色 (Risk, Positional, etc.)
                'triggers': int(total_triggers),
                'successes': int(success_count),
                'win_rate_pct': round(win_rate, 2),
                'avg_max_profit_pct': round(avg_max_profit, 2),
                'avg_max_drawdown_pct': round(avg_max_drawdown, 2),
                'avg_exit_days': round(avg_exit_days, 1),
            })
        analysis_results.sort(key=lambda x: x['win_rate_pct'], reverse=True)
        return analysis_results
    async def run_analysis_for_stock(self, stock_code: str, start_date: Optional[str], end_date: Optional[str]) -> list:
        """
        【V2.3 精确归因版】
        为单个股票执行基于数据库的回测分析 (分析最终买入信号的构成组件)。
        - 核心重构:
          1. 获取完整的 score_details_df。
          2. 筛选出 signal_type 为 '买入信号' 的日期。
          3. 使用这些日期来过滤 score_details_df，得到一份只包含“功臣”信号的详情表。
          4. 将这份精确的详情表交给 PerformanceAnalyzer 进行分析。
        """
        if not start_date:
            start_date = '1990-01-01'
        if not end_date:
            end_date = date.today().strftime('%Y-%m-%d')
        # 步骤1: 获取包含价格、最终信号类型和【完整信号组件详情】的数据
        df_indicators, score_details_df = await self._fetch_analysis_data_from_db(stock_code, start_date, end_date)
        if df_indicators is None or df_indicators.empty or score_details_df is None or score_details_df.empty:
            return []
        if not get_param_value(self.analyzer_params.get('enabled'), False):
            logger.info("性能分析模块在配置文件中被禁用。")
            return []
        try:
            # 步骤2: 筛选出“功臣”信号的详情
            buy_signal_dates = df_indicators[df_indicators['signal_type'] == '买入信号'].index
            if buy_signal_dates.empty:
                return []
            # 过滤出仅在买入日活跃的信号及其得分
            filtered_score_details_df = score_details_df.loc[score_details_df.index.isin(buy_signal_dates)]
            # 移除在这些天从未触发过的信号列，进行优化
            filtered_score_details_df = filtered_score_details_df.loc[:, (filtered_score_details_df != 0).any(axis=0)]
            if filtered_score_details_df.empty:
                return []
            # 步骤3: 遍历所有“功臣”信号，评估其在触发日的后续表现
            all_trade_outcomes = []
            # 遍历每一列（即每一个信号）
            for signal_name in filtered_score_details_df.columns:
                # 找到该信号具体被触发的日期
                event_series = filtered_score_details_df[signal_name] > 0
                event_dates = event_series[event_series].index
                for entry_date in event_dates:
                    # 调用服务自身的评估方法
                    # 注意：df_indicators 包含了完整的价格数据，是评估所需的基础
                    outcome = self._evaluate_offensive_signal(entry_date, df_indicators)
                    if outcome:
                        all_trade_outcomes.append({
                            'signal_name': signal_name,
                            'signal_type_role': self.scoring_params.get('score_type_map', {}).get(signal_name, {}).get('type', 'unknown'),
                            **outcome
                        })
            # 步骤4: 使用服务自身的聚合方法，生成最终报告
            return self._aggregate_atomic_results(all_trade_outcomes)
        except Exception as e:
            logger.error(f"[{stock_code}] (精确归因模式)性能分析器在执行过程中发生异常: {e}", exc_info=True)
            return []
    async def _fetch_analysis_data_from_db(self, stock_code: str, start_date: str, end_date: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        【V1.5 日期标准化版】(旧方法) 从数据库中异步获取并构建分析所需的核心DataFrame。
        """
        daily_price_df = await self.time_trade_dao.get_daily_data(stock_code, start_date.replace('-', ''), end_date.replace('-', ''))
        if daily_price_df.empty:
            logger.warning(f"[{stock_code}] 在指定日期内未找到日线行情数据。")
            return None, None
        daily_price_df.rename(columns={'open': 'open_D', 'close': 'close_D', 'high': 'high_D', 'low': 'low_D'}, inplace=True)
        # [修正] 确保索引是 date 对象
        daily_price_df.index = pd.to_datetime(daily_price_df.index).date
        daily_scores_qs = StrategyDailyScore.objects.filter(
            stock__stock_code=stock_code,
            trade_date__range=(start_date, end_date)
        ).order_by('trade_date')
        daily_scores_list = await sync_to_async(list)(daily_scores_qs.values('trade_date', 'signal_type'))
        if not daily_scores_list:
            logger.warning(f"[{stock_code}] 在指定日期内未找到策略分数数据。")
            return None, None
        daily_scores_df = pd.DataFrame(daily_scores_list)
        # [修正] 确保用于 join 的索引也是 date 对象
        daily_scores_df['trade_date'] = pd.to_datetime(daily_scores_df['trade_date']).dt.date
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
            # logger.warning(f"[{stock_code}] 在指定日期内未找到分数构成详情数据。")
            return df_indicators, pd.DataFrame()
        components_df = pd.DataFrame(components_list)
        components_df.rename(columns={'daily_score__trade_date': 'trade_date'}, inplace=True)
        # [修正] 确保 pivot_table 的索引也是 date 对象
        components_df['trade_date'] = pd.to_datetime(components_df['trade_date']).dt.date
        score_details_df = components_df.pivot_table(
            index='trade_date',
            columns='signal_name',
            values='score_value',
            fill_value=0
        )
        return df_indicators, score_details_df

