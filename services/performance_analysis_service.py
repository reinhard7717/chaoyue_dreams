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
    def __init__(self, cache_manager: CacheManager):
        """
        【V2.2 构造函数修复版】
        """
        self.time_trade_dao = StockTimeTradeDAO(cache_manager)
        # 加载策略配置以获取分析参数
        unified_config_path = 'config/trend_follow_strategy.json'
        self.unified_config = load_strategy_config(unified_config_path)
        self.analyzer_params = get_params_block({'unified_config': self.unified_config}, 'performance_analysis_params')
        self.scoring_params = get_params_block({'unified_config': self.unified_config}, 'four_layer_scoring_params')
        
        # 从配置中预加载分析参数
        self.look_forward_days = get_param_value(self.analyzer_params.get('look_forward_days'), 20)
        self.profit_target_pct = get_param_value(self.analyzer_params.get('profit_target_pct'), 0.15)
        self.stop_loss_pct = get_param_value(self.analyzer_params.get('stop_loss_pct'), 0.07)

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
                # [核心修正] 根据信号的性质（状态 vs 事件）应用不同的评估协议
                if signal_type in ['trigger', 'composite', 'playbook']:
                    # 性质：事件。评估每一次发生。
                    print(f"  -> [探针] 信号 {signal_name} (类型: {signal_type}) 被视为瞬时事件。")
                    event_dates = signal_series[signal_series].index.tolist()
                else:
                    # 性质：状态。只评估首次进入。
                    print(f"  -> [探针] 信号 {signal_name} (类型: {signal_type}) 被视为持续状态。")
                    is_first_day = signal_series & ~signal_series.shift(1).fillna(False)
                    event_dates = is_first_day[is_first_day].index.tolist()

                if event_dates:
                    print(f"    -> [探针] 为 {signal_name} 生成了 {len(event_dates)} 个评估日期。")

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
        【V4.2 T+1基准修正版】评估“进攻型”信号的表现 (看涨)。
        """
        try:
            # 步骤1: 定位到 T+1 日
            entry_idx = price_df.index.get_loc(entry_date)
            trade_day_idx = entry_idx + 1
            if trade_day_idx >= len(price_df):
                return None

            # 步骤2: 获取 T+1 日的开盘价作为基准
            trade_day_row = price_df.iloc[trade_day_idx]
            entry_price = trade_day_row['open_D']
            
            # 如果T+1开盘价无效，则无法进行评估
            if pd.isna(entry_price) or entry_price <= 0:
                return None

            # 步骤3: 回测窗口从 T+1 日开始
            look_forward_df = price_df.iloc[trade_day_idx : trade_day_idx + self.look_forward_days]
        except (KeyError, IndexError):
            return None
        
        if look_forward_df.empty:
            return None

        target_price = entry_price * (1 + self.profit_target_pct)
        stop_price = entry_price * (1 - self.stop_loss_pct)
        
        outcome, exit_days = 'timeout', self.look_forward_days
        max_profit_pct = 0.0
        max_drawdown_pct = 0.0

        for i, row in enumerate(look_forward_df.itertuples()):
            day_num = i + 1
            # 回测从T+1日当天开始，所以要考虑当天的最高/最低价
            high_price = row.high_D
            low_price = row.low_D
            
            max_profit_pct = max(max_profit_pct, (high_price / entry_price) - 1)
            max_drawdown_pct = min(max_drawdown_pct, (low_price / entry_price) - 1)

            if high_price >= target_price:
                outcome, exit_days = 'success', day_num
                break
            if low_price <= stop_price:
                outcome, exit_days = 'failure', day_num
                break
        
        return {
            'outcome': outcome, 'exit_days': exit_days,
            'max_profit_pct': max_profit_pct, 'max_drawdown_pct': max_drawdown_pct,
        }

    def _evaluate_defensive_signal(self, entry_date: date, price_df: pd.DataFrame) -> Optional[Dict]:
        """
        【V4.2 T+1基准修正版】评估“防御/风险型”信号的表现 (看跌)。
        """
        try:
            # 步骤1: 定位到 T+1 日
            entry_idx = price_df.index.get_loc(entry_date)
            trade_day_idx = entry_idx + 1
            if trade_day_idx >= len(price_df):
                return None

            # 步骤2: 获取 T+1 日的开盘价作为基准
            trade_day_row = price_df.iloc[trade_day_idx]
            entry_price = trade_day_row['open_D']

            if pd.isna(entry_price) or entry_price <= 0:
                return None

            # 步骤3: 回测窗口从 T+1 日开始
            look_forward_df = price_df.iloc[trade_day_idx : trade_day_idx + self.look_forward_days]
        except (KeyError, IndexError):
            return None
        
        if look_forward_df.empty:
            return None

        target_price_fall = entry_price * (1 - self.stop_loss_pct)
        stop_price_rise = entry_price * (1 + self.profit_target_pct)
        
        outcome, exit_days = 'timeout', self.look_forward_days
        max_fall_pct = 0.0
        max_rise_pct = 0.0

        for i, row in enumerate(look_forward_df.itertuples()):
            day_num = i + 1
            high_price = row.high_D
            low_price = row.low_D
            
            max_fall_pct = max(max_fall_pct, 1 - (low_price / entry_price))
            max_rise_pct = max(max_rise_pct, (high_price / entry_price) - 1)

            if low_price <= target_price_fall:
                outcome, exit_days = 'success', day_num
                break
            if high_price >= stop_price_rise:
                outcome, exit_days = 'failure', day_num
                break
        
        return {
            'outcome': outcome, 'exit_days': exit_days,
            'max_profit_pct': max_fall_pct,
            'max_drawdown_pct': max_rise_pct,
        }

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

        # [核心修正] 强制转换为Pandas原生datetime64[ns]类型，不再使用.dt.date
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
        
        # [核心修正] 同样，将价格数据的日期也强制转换为datetime64[ns]类型
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
            # 步骤2: 【核心修正】筛选出“功臣”信号，而不是创建虚拟信号
            # 找到所有被标记为“买入信号”的日期
            buy_signal_dates = df_indicators[df_indicators['signal_type'] == '买入信号'].index
            
            if buy_signal_dates.empty:
                return []

            # 使用这些“买入日”来过滤完整的信号详情表
            # .reindex() 可以保证即使某些日期在 score_details_df 中不存在也能安全处理
            # .loc[] 在这里更直接，因为我们知道日期肯定存在
            filtered_score_details_df = score_details_df.loc[score_details_df.index.isin(buy_signal_dates)]

            # 优化：移除在这些买入日中从未出现过的信号（即整列都为0的信号）
            filtered_score_details_df = filtered_score_details_df.loc[:, (filtered_score_details_df != 0).any(axis=0)]

            if filtered_score_details_df.empty:
                return []

            # 步骤3: 使用精确的“功臣名单”调用分析器
            from strategies.trend_following.performance_analyzer import PerformanceAnalyzer
            analyzer = PerformanceAnalyzer(
                df_indicators=df_indicators,
                score_details_df=filtered_score_details_df, # 传递被精确过滤后的、包含真实信号的详情
                atomic_states={},
                trigger_events={},
                analysis_params=self.analyzer_params,
                scoring_params=self.scoring_params
            )

            return analyzer.run_analysis()
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

