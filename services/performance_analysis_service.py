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
        【V3.0 统计学修正版】执行全市场原子信号的性能分析。
        - 核心重构: 废除了原有的 `groupby().min()` 逻辑。
        - 新逻辑:
          1. 遍历所有股票，对每只股票独立进行全周期事件识别。
          2. 区分“状态”和“触发器”：
             - 状态(State): 只在状态首次进入时触发一次回测事件。
             - 触发器(Trigger): 每次出现都触发回测事件。
          3. 确保对回测周期内的所有有效事件进行分析，提供统计学上可靠的结果。
        """
        print("-> [Service V3.0] 启动全景沙盘推演 (统计学修正版)...")

        # 1. 从数据库获取所有需要分析的数据 (此部分逻辑不变)
        all_states_df, all_prices_df = await self._fetch_atomic_analysis_data_from_db()
        if all_states_df is None or all_prices_df is None or all_states_df.empty or all_prices_df.empty:
            logger.warning("[Service V3.0] 无法获取原子状态或价格数据，分析终止。")
            return []

        print("    -> [Service V3.0] 数据加载完成，开始按股票识别所有有效触发事件...")
        
        # 2. 【核心重构】识别所有有效触发事件
        trade_outcomes = []
        prices_by_stock = dict(iter(all_prices_df.groupby('stock_code')))
        
        # 按股票代码分组，对每只股票进行独立分析
        for stock_code, stock_states_df in all_states_df.groupby('stock_code'):
            price_df = prices_by_stock.get(stock_code)
            if price_df is None or price_df.empty:
                continue

            # 将该股票的状态数据透视为以日期为索引，信号为列的DataFrame
            try:
                pivoted_df = stock_states_df.pivot_table(
                    index='trade_date', columns='signal_name', aggfunc='size', fill_value=0
                ).astype(bool)
            except Exception:
                continue # 如果透视失败，跳过该股票

            # 遍历每个信号
            for signal_name in pivoted_df.columns:
                signal_series = pivoted_df[signal_name].sort_index()
                signal_meta = self.scoring_params.get('score_type_map', {}).get(signal_name, {})
                signal_type = signal_meta.get('type', 'State').capitalize() # 默认为State

                event_dates = []
                if signal_type == 'State':
                    # 对于状态信号，只在状态首次进入时触发
                    is_first_day = signal_series & ~signal_series.shift(1).fillna(False)
                    event_dates = is_first_day[is_first_day].index.tolist()
                else: # Trigger, Positional, Risk, etc. 都被视为瞬时事件
                    # 对于触发器信号，每次出现都触发
                    event_dates = signal_series[signal_series].index.tolist()

                # 对识别出的所有事件日期进行交易模拟
                for entry_date in event_dates:
                    outcome = self._analyze_single_trade_performance(entry_date, price_df)
                    if outcome:
                        trade_outcomes.append({
                            'signal_name': signal_name,
                            **outcome
                        })

        # 3. 聚合结果并生成最终报告 (此部分逻辑不变)
        print(f"    -> [Service V3.0] 模拟完成，正在聚合 {len(trade_outcomes)} 条交易结果...")
        final_report = self._aggregate_atomic_results(trade_outcomes)
        print(f"-> [Service V3.0] 全景沙盘推演完成，生成 {len(final_report)} 条信号的性能报告。")
        return final_report

    async def _fetch_atomic_analysis_data_from_db(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        【V2.2 数据接口修正版】为全景分析从数据库批量获取数据。
        """
        # ... (方法前半部分逻辑不变) ...
        end_date = date.today()
        start_date = end_date - timedelta(days=365)
        print(f"    -> [DB Service V2.0] 正在加载全市场 {start_date} 到 {end_date} 的原子状态和价格数据...")
        states_qs = StrategyDailyState.objects.filter(daily_score__trade_date__range=(start_date, end_date)).select_related('daily_score__stock')
        states_list = await sync_to_async(list)(states_qs.values('daily_score__stock__stock_code', 'daily_score__trade_date', 'signal_name'))
        if not states_list:
            logger.warning("在指定日期内未找到任何原子状态数据。")
            return None, None
        all_states_df = pd.DataFrame(states_list)
        all_states_df.rename(columns={'daily_score__stock__stock_code': 'stock_code', 'daily_score__trade_date': 'trade_date'}, inplace=True)
        unique_stock_codes = all_states_df['stock_code'].unique().tolist()
        all_prices_df = await self.time_trade_dao.get_daily_data_for_stocks(
            unique_stock_codes, 
            start_date.strftime('%Y%m%d'), 
            (end_date + timedelta(days=self.look_forward_days)).strftime('%Y%m%d')
        )
        if all_prices_df.empty:
            logger.warning("未能获取到任何相关股票的日线行情数据。")
            return None, None
            
        # [修正] 补充对 'open' 列的重命名
        all_prices_df.rename(columns={'open': 'open_D', 'close': 'close_D', 'high': 'high_D', 'low': 'low_D'}, inplace=True)
        all_prices_df.rename(columns={'trade_time': 'trade_date'}, inplace=True)
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

    async def run_analysis_for_stock(self, stock_code: str, start_date: Optional[str], end_date: Optional[str]) -> list:
        """
        【V2.2 逻辑重塑版】
        为单个股票执行基于数据库的回测分析 (分析最终买入信号)。
        - 核心重构: 不再传递包含所有信号组件的 score_details_df。
                    取而代之，我们根据 StrategyDailyScore 中的 '买入信号' 记录，
                    动态合成一个只包含最终买入事件的“纯净版”score_details_df，
                    确保 PerformanceAnalyzer 只分析其应该分析的目标。
        """
        if not start_date:
            start_date = '1990-01-01'
        if not end_date:
            end_date = date.today().strftime('%Y-%m-%d')

        # print(f"-> [Service V2.2] 开始分析股票 {stock_code} 的最终信号，日期范围: {start_date} 到 {end_date}")

        # 步骤1: 获取包含价格和每日信号类型的基础数据 (df_indicators)
        # 注意：我们不再需要从 _fetch_analysis_data_from_db 获取 score_details_df
        df_indicators, _ = await self._fetch_analysis_data_from_db(stock_code, start_date, end_date)

        if df_indicators is None or df_indicators.empty:
            # logger.warning(f"[{stock_code}] 获取并合并数据后，无有效数据可供分析。")
            return []

        if not get_param_value(self.analyzer_params.get('enabled'), False):
            logger.info("性能分析模块在配置文件中被禁用。")
            return []
            
        try:
            # 步骤2: 【核心修正】合成只包含最终买入信号的“纯净”信号详情
            # 找到所有被标记为“买入信号”的日期
            buy_signal_dates = df_indicators[df_indicators['signal_type'] == '买入信号'].index
            
            if buy_signal_dates.empty:
                # logger.info(f"[{stock_code}] 在指定时间段内未发现任何'买入信号'。")
                return []

            # 创建一个与价格数据对齐的、只包含一个“虚拟”信号的DataFrame
            synthetic_score_details_df = pd.DataFrame(0, index=df_indicators.index, columns=['FINAL_BUY_SIGNAL'])
            # 在买入信号当天，将这个虚拟信号的值设为1
            synthetic_score_details_df.loc[buy_signal_dates, 'FINAL_BUY_SIGNAL'] = 1

            # 步骤3: 使用纯净的情报调用分析器
            from strategies.trend_following.performance_analyzer import PerformanceAnalyzer
            analyzer = PerformanceAnalyzer(
                df_indicators=df_indicators,
                score_details_df=synthetic_score_details_df, # 传递合成的、纯净的信号详情
                atomic_states={},
                trigger_events={},
                analysis_params=self.analyzer_params,
                scoring_params=self.scoring_params
            )

            return analyzer.run_analysis()
        except Exception as e:
            logger.error(f"[{stock_code}] (兼容模式)性能分析器在执行过程中发生异常: {e}", exc_info=True)
            return []

    async def _fetch_analysis_data_from_db(self, stock_code: str, start_date: str, end_date: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        【V1.4 数据接口修正版】(旧方法) 从数据库中异步获取并构建分析所需的核心DataFrame。
        """
        print(f"    -> [DB Service V1.4] 正在为 {stock_code} 从数据库加载 {start_date} 到 {end_date} 的数据...")
        
        daily_price_df = await self.time_trade_dao.get_daily_data(stock_code, start_date.replace('-', ''), end_date.replace('-', ''))
        
        if daily_price_df.empty:
            logger.warning(f"[{stock_code}] 在指定日期内未找到日线行情数据。")
            return None, None
        
        # [修正] 补充对 'open' 列的重命名
        daily_price_df.rename(columns={'open': 'open_D', 'close': 'close_D', 'high': 'high_D', 'low': 'low_D'}, inplace=True)
        daily_price_df.index = daily_price_df.index.date

        # ... (方法余下部分逻辑不变) ...
        daily_scores_qs = StrategyDailyScore.objects.filter(stock__stock_code=stock_code, trade_date__range=(start_date, end_date)).order_by('trade_date')
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
        score_components_qs = StrategyScoreComponent.objects.filter(daily_score__stock__stock_code=stock_code, daily_score__trade_date__range=(start_date, end_date)).select_related('daily_score')
        components_list = await sync_to_async(list)(score_components_qs.values('daily_score__trade_date', 'signal_name', 'score_value'))
        if not components_list:
            logger.warning(f"[{stock_code}] 在指定日期内未找到分数构成详情数据。")
            return df_indicators, pd.DataFrame()
        components_df = pd.DataFrame(components_list)
        components_df.rename(columns={'daily_score__trade_date': 'trade_date'}, inplace=True)
        score_details_df = components_df.pivot_table(index='trade_date', columns='signal_name', values='score_value', fill_value=0)
        print(f"    -> [DB Service V1.4] 数据加载与转换完成[{stock_code}]。行情: {len(df_indicators)}天, 信号详情: {len(score_details_df)}天。")
        return df_indicators, score_details_df


