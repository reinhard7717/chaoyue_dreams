# 文件: strategies/trend_following_strategy.py
# 版本: V62.0 - 增强调试日志版
from copy import deepcopy
import logging
from decimal import Decimal
from utils.data_sanitizer import sanitize_for_json
from typing import Any, Dict, List, Optional, Tuple
from scipy.signal import find_peaks
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences
from strategies.kline_pattern_recognizer import KlinePatternRecognizer
from utils.config_loader import load_strategy_config

logger = logging.getLogger(__name__)

class TrendFollowStrategy:
    """
    趋势跟踪策略 (V62.0 - 增强调试日志版)
    - 核心升级: 新增 _format_debug_dates 辅助函数，并全局应用于所有诊断日志，使其能输出具体发生日期，极大提升调试效率。
    """
    def __init__(self, config: dict):
        """
        【V21.1 构造函数优化版】
        """
        self.daily_params = config
        kline_params = self._get_params_block(self.daily_params, 'kline_pattern_params')
        self.pattern_recognizer = KlinePatternRecognizer(params=kline_params)
        self.signals = {}
        self.scores = {}
        self._last_score_details_df = None
        self.debug_params = self._get_params_block(self.daily_params, 'debug_params')
        self.verbose_logging = self.debug_params.get('enabled', False) and self.debug_params.get('verbose_logging', False)
        # ▼▼▼ 在初始化时缓存剧本蓝图 ▼▼▼
        self.playbook_blueprints = self._get_playbook_blueprints()
        print(f"--- [战术策略 TrendFollowStrategy (V83.0 蓝图与水合架构)] 初始化完成，已缓存 {len(self.playbook_blueprints)} 个剧本蓝图。---")

    #  日志格式化辅助函数 ▼▼▼
    def _format_debug_dates(self, signal_series: pd.Series, display_limit: int = 10) -> str:
        """
        【V62.0 新增】一个用于增强调试日志的辅助函数。
        - 功能: 从一个布尔型的Series中提取为True的日期，并格式化为字符串。
        - 特性: 当日期过多时，会自动截断并显示总数，避免日志刷屏。
        """
        if not isinstance(signal_series, pd.Series) or signal_series.dtype != bool:
            return ""
        
        start_date_filter = pd.to_datetime('2024-05-01', utc=True) # 修正后的代码
        active_dates = signal_series[signal_series & (signal_series.index >= start_date_filter)].index
        count = len(active_dates)
        
        if count == 0:
            return ""
            
        # 将日期格式化为 'YYYY-MM-DD'
        date_strings = [d.strftime('%Y-%m-%d') for d in active_dates]
        
        if count > display_limit:
            # 如果日期太多，只显示前N个并附上总数
            # return f" -> 日期: {date_strings[:display_limit]}... (共 {count} 天)"
            return f" -> 日期: {date_strings[-display_limit:]}... (共 {count} 天)"
        else:
            # 否则全部显示
            return f" -> 日期: {date_strings}"

    def _get_param_value(self, param: Any, default: Any = None) -> Any:
        """
        【V40.3 新增】健壮的参数值解析器。
        """
        if isinstance(param, dict) and 'value' in param:
            return param['value']
        if param is not None:
            return param
        return default

    def _get_params_block(self, params: dict, block_name: str, default_return: Any = None) -> dict:
        if default_return is None:
            default_return = {}
        return params.get('strategy_params', {}).get('trend_follow', {}).get(block_name, default_return)

    def _get_periods_for_timeframe(self, indicator_params: dict, timeframe: str) -> Optional[list]:
        """
        【V5.0 】根据时间周期从指标配置中智能获取正确的'periods'。
        """
        if not indicator_params:
            return None
        if 'configs' in indicator_params and isinstance(indicator_params['configs'], list):
            for config_item in indicator_params['configs']:
                if timeframe in config_item.get('apply_on', []):
                    return config_item.get('periods')
            return None
        elif 'periods' in indicator_params:
            apply_on = indicator_params.get('apply_on', [])
            if not apply_on or timeframe in apply_on:
                return indicator_params.get('periods')
            else:
                return None
        return None

    def _ensure_numeric_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V40.6 智能净化版】数据类型标准化引擎
        """
        converted_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                first_valid_item = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if isinstance(first_valid_item, Decimal):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    converted_cols.append(col)
        if not converted_cols:
            print("      -> 所有数值列类型正常，无需转换。")
        return df

    # 波段跟踪模拟器
    def simulate_wave_tracking(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        【V85.0 新增】波段跟踪模拟器
        - 核心功能: 将无状态的每日信号，转换为有状态的持仓管理和交易生命周期模拟。
        - 实现了分层的减仓与清仓逻辑，是策略从“信号生成”到“实战模拟”的关键一步。
        """
        print("\n" + "="*60)
        print("====== 开始执行【波段跟踪模拟器 V85.0】 ======")
        
        # --- 1. 从JSON配置中加载波段跟踪参数 ---
        tracking_params = self._get_params_block(params, 'wave_tracking_params', {})
        if not self._get_param_value(tracking_params.get('enabled'), False):
            print("    - [信息] 波段跟踪模拟器被禁用，跳过。")
            return df

        profit_target_partial = self._get_param_value(tracking_params.get('profit_target_partial'), 0.30)
        trailing_stop_pct = self._get_param_value(tracking_params.get('trailing_stop_pct'), 0.15)
        exit_code_partial = self._get_param_value(tracking_params.get('exit_code_partial'), 77)
        exit_code_full = self._get_param_value(tracking_params.get('exit_code_full'), 99)
        life_line_ma_period = self._get_param_value(tracking_params.get('life_line_ma'), 21)
        life_line_ma_col = f'EMA_{life_line_ma_period}_D'

        print(f"    - 减仓规则: 利润 > {profit_target_partial*100}% 或 风险码 == {exit_code_partial}")
        print(f"    - 清仓规则: 风险码 == {exit_code_full} 或 从最高点回撤 > {trailing_stop_pct*100}% 或 失守{life_line_ma_col}")

        # --- 2. 初始化状态列和变量 ---
        df['position_status'] = 0.0  # 0: 无仓位, 0.5: 半仓, 1.0: 满仓
        df['trade_action'] = ''      # 'ENTRY', 'PARTIAL_EXIT', 'FULL_EXIT'
        
        in_position = False
        position_size = 0.0
        entry_price = 0.0
        highest_price_since_entry = 0.0
        partial_exit_done = False

        # --- 3. 逐日迭代，模拟交易状态机 ---
        # 使用 .itertuples() 以获得更好的性能
        for row in df.itertuples():
            current_date = row.Index
            
            # 状态机核心逻辑
            if not in_position:
                # 检查入场信号
                if row.signal_entry:
                    in_position = True
                    position_size = 1.0
                    entry_price = row.close_D
                    highest_price_since_entry = row.close_D
                    partial_exit_done = False
                    
                    df.loc[current_date, 'position_status'] = position_size
                    df.loc[current_date, 'trade_action'] = 'ENTRY'
                    print(f"      -> {current_date.date()}: [入场] 价格: {entry_price:.2f}, 仓位: {position_size*100}%")
            
            else: # 如果在持仓中
                # 3.1 更新波段最高价
                highest_price_since_entry = max(highest_price_since_entry, row.high_D)
                
                # 3.2 定义清仓/减仓的理由
                exit_reason = ""
                
                # 3.3 检查清仓条件 (优先级最高)
                should_full_exit = False
                if row.exit_signal_code == exit_code_full:
                    should_full_exit = True
                    exit_reason = f"高危风险码({exit_code_full})"
                elif row.close_D < highest_price_since_entry * (1 - trailing_stop_pct):
                    should_full_exit = True
                    exit_reason = f"移动止损(最高价{highest_price_since_entry:.2f}, 回撤>{trailing_stop_pct*100}%)"
                elif partial_exit_done and row.close_D < getattr(row, life_line_ma_col, float('inf')):
                    # 只有在减仓后，生命线失守才作为清仓条件
                    should_full_exit = True
                    exit_reason = f"失守生命线({life_line_ma_col})"

                if should_full_exit:
                    in_position = False
                    position_size = 0.0
                    
                    df.loc[current_date, 'position_status'] = position_size
                    df.loc[current_date, 'trade_action'] = 'FULL_EXIT'
                    print(f"      -> {current_date.date()}: [清仓] 价格: {row.close_D:.2f}, 原因: {exit_reason}")
                    continue # 当天清仓后，不再执行减仓逻辑

                # 3.4 检查减仓条件 (如果未清仓)
                should_partial_exit = False
                if not partial_exit_done: # 确保只减仓一次
                    if row.close_D > entry_price * (1 + profit_target_partial):
                        should_partial_exit = True
                        exit_reason = f"目标利润(>{profit_target_partial*100}%)达成"
                    elif row.exit_signal_code == exit_code_partial:
                        should_partial_exit = True
                        exit_reason = f"中度风险码({exit_code_partial})"

                if should_partial_exit:
                    position_size = 0.5
                    partial_exit_done = True
                    
                    df.loc[current_date, 'position_status'] = position_size
                    df.loc[current_date, 'trade_action'] = 'PARTIAL_EXIT'
                    print(f"      -> {current_date.date()}: [减仓] 价格: {row.close_D:.2f}, 仓位降至50%, 原因: {exit_reason}")

                # 3.5 如果当天无操作，则继承前一天的仓位状态
                if not should_full_exit and not should_partial_exit:
                    df.loc[current_date, 'position_status'] = position_size

        # --- 4. 填充未持仓期间的仓位状态 ---
        df['position_status'] = df['position_status'].ffill().fillna(0)
        
        print("====== 【波段跟踪模拟器 V85.0】执行完毕 ======")
        print("="*60 + "\n")
        return df

    def apply_strategy(self, df: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        【V80.0 总指挥版】
        - 核心升级: 引入了最终的、分层的评分哲学。
        - 1. 调用 V79.0 版本的 _calculate_entry_score 计算出“原始总分”。
        - 2. 新增 _apply_final_score_adjustments 函数，作为“指挥棒模型”，对原始总分进行质量乘数调整。
        - 3. 最终的交易决策基于调整后的分数 (adjusted_score)。
        """
        print("\n" + "="*60)
        print(f"====== 日期: {df.index[-1].date()} | 开始执行【战术引擎 V80.0】 ======")
        if df is None or df.empty:
            print("    - [错误] 传入的DataFrame为空，战术引擎终止。")
            return pd.DataFrame(), {}
        df = self._ensure_numeric_types(df)
        timeframe_suffixes = ['_D', '_W', '_M', '_5', '_15', '_30', '_60']
        rename_map = {
            col: f"{col}_D" for col in df.columns 
            if not any(col.endswith(suffix) for suffix in timeframe_suffixes) 
            and not col.startswith(('VWAP_', 'BASE_', 'playbook_', 'signal_', 'kline_', 'context_', 'cond_'))
        }
        if rename_map:
            df = df.rename(columns=rename_map)
        if 'close_D' in df.columns:
            df['pct_change_D'] = df['close_D'].pct_change()
        
        print("--- [总指挥] 步骤1: 核心数据引擎启动 ---")
        # 假设 _prepare_derived_features 存在，如果不存在，可以注释掉
        # df = self._prepare_derived_features(df, params) 
        df = self._calculate_trend_slopes(df, params)
        df = self.pattern_recognizer.identify_all(df)
        
        print("--- [总指挥] 步骤1.5: 原子状态诊断中心启动 ---")
        atomic_states = {
            **self._diagnose_chip_states(df, params),
            **self._diagnose_ma_states(df, params),
            **self._diagnose_oscillator_states(df, params),
            **self._diagnose_capital_states(df, params),
            **self._diagnose_volatility_states(df, params),
            **self._diagnose_box_states(df, params),
            **self._diagnose_kline_patterns(df, params),
            **self._diagnose_board_patterns(df, params)
        }
        # 将风险/机会因子也注入到原子状态中
        risk_and_opp_factors = self._diagnose_risk_factors(df, params)
        atomic_states.update(risk_and_opp_factors) # 合并，确保所有状态可用

        print("--- [总指挥] 原子状态诊断中心完成，所有原子状态已生成。 ---")
        
        print("--- [总指挥] 步骤2: 准备状态评审引擎启动 ---")
        setup_scores = self._calculate_setup_conditions(df, params, atomic_states)
        
        print("--- [总指挥] 步骤3: 触发事件定义引擎启动 ---")
        # 假设 _define_trigger_events 存在，如果不存在，需要实现或用空字典代替
        trigger_events = self._define_trigger_events(df, params, atomic_states) 
        
        print("--- [总指挥] 步骤4: 剧本家族计分引擎启动 ---")
        # ▼▼▼  _calculate_entry_score 现在返回原始分数 ▼▼▼
        # 确保您已经将 _calculate_entry_score 更新为 V79.0 "家族继承引擎" 版本
        df, score_details_df = self._calculate_entry_score(df, params, trigger_events, setup_scores, atomic_states)
        raw_total_score = df['entry_score'].copy() # 这是V79.0计算出的原始总分
        
        print("--- [总指挥] 步骤4.5: 指挥棒模型启动，进行最终得分调整 ---")
        # ▼▼▼ 调用指挥棒模型 ▼▼▼
        adjusted_score, adjustment_details = self._apply_final_score_adjustments(df, raw_total_score, params)
        df['entry_score_raw'] = raw_total_score # 保留原始分，便于分析
        df['entry_score'] = adjusted_score # 使用调整后的分数作为最终得分
        # 将调整细节合并到总细节中，便于分析
        self._last_score_details_df = pd.concat([score_details_df, adjustment_details], axis=1).fillna(0)
        
        print("--- [总指挥] 步骤5: 智能风险评审与出场决策引擎启动 ---")
        risk_score = self._calculate_risk_score(df, params, risk_and_opp_factors)
        df['exit_signal_code'] = self._calculate_exit_signals(df, params, risk_score)
        
        print("--- [总指挥] 步骤6: 最终信号合成与日志输出 ---")
        entry_scoring_params = self._get_params_block(params, 'entry_scoring_params', {})
        score_threshold = self._get_param_value(entry_scoring_params.get('score_threshold'), 100)
        # ▼▼▼ 使用调整后的分数进行决策 ▼▼▼
        df['signal_entry'] = df['entry_score'] >= score_threshold
        
        print(f"====== 【战术引擎 V80.0】执行完毕 ======")
        print("="*60 + "\n")
        return df, {}

    def prepare_db_records(self, stock_code: str, result_df: pd.DataFrame, atomic_signals: Dict[str, pd.Series], params: dict, result_timeframe: str = 'D') -> List[Dict[str, Any]]:
        """
        【V85.0 波段跟踪版】
        - 核心升级: 不再基于零散的信号列，而是基于波段跟踪模拟器生成的 'trade_action' 列来创建记录。
                    这使得每一条记录都代表一个明确的、有状态的交易动作（入场、减仓、清仓）。
        """
        # ▼▼▼ 筛选条件变更为 'trade_action' 列 ▼▼▼
        if 'trade_action' not in result_df.columns or result_df['trade_action'].eq('').all():
            # 如果没有波段跟踪列，或者该列为空，则说明没有发生任何交易动作
            return []
        df_with_actions = result_df[result_df['trade_action'] != ''].copy()

        if df_with_actions.empty:
            return []
        
        records = []
        strategy_info_block = self._get_params_block(params, 'strategy_info', {})
        name_param = strategy_info_block.get('name')
        strategy_name = self._get_param_value(name_param, 'unknown_strategy')
        timeframe = result_timeframe
        
        playbook_cn_name_map = {p['name']: p.get('cn_name', p['name']) for p in self.playbook_blueprints}
        
        for timestamp, row in df_with_actions.iterrows():
            trade_action = row.get('trade_action')
            
            # 根据交易动作，智能判断是入场还是出场信号
            is_entry = trade_action == 'ENTRY'
            is_exit = trade_action in ['PARTIAL_EXIT', 'FULL_EXIT']
            
            triggered_playbooks_list = []
            triggered_playbooks_cn_list = []
            if self._last_score_details_df is not None and timestamp in self._last_score_details_df.index:
                playbooks_with_scores = self._last_score_details_df.loc[timestamp]
                active_items = playbooks_with_scores[playbooks_with_scores > 0].index
                excluded_prefixes = ('BASE_', 'BONUS_', 'PENALTY_', 'INDUSTRY_', 'WATCHING_SETUP', 'CHIP_PURITY_MULTIPLIER', 'VOLATILITY_SILENCE_MULTIPLIER')
                triggered_playbooks_list = [ item for item in active_items if not item.startswith(excluded_prefixes) ]
                triggered_playbooks_cn_list = [ playbook_cn_name_map.get(item, item) for item in triggered_playbooks_list ]
            
            active_setups = [s.replace('SETUP_', '') for s in row.index if s.startswith('SETUP_') and row[s] is True]
            context_dict = {k: v for k, v in row.items() if pd.notna(v)}
            sanitized_context = sanitize_for_json(context_dict)
            pct_change = row.get('pct_change_D', row.get('pct_change', 0.0))
            
            record = {
                "stock_code": stock_code,
                "trade_time": sanitize_for_json(timestamp),
                "timeframe": timeframe,
                "strategy_name": strategy_name,
                "close_price": sanitize_for_json(row.get('close_D')),
                "pct_change": sanitize_for_json(pct_change),
                "entry_score": sanitize_for_json(row.get('entry_score', 0.0)),
                # ▼▼▼ 基于 trade_action 设置信号和记录 ▼▼▼
                "entry_signal": is_entry,
                "exit_signal_code": sanitize_for_json(row.get('exit_signal_code', 0)) if is_exit else 0,
                "trade_action": trade_action, # 新增字段
                "position_status": row.get('position_status', 0.0), # 新增字段
                "triggered_playbooks": triggered_playbooks_list,
                "triggered_playbooks_cn": triggered_playbooks_cn_list,
                "active_setups": active_setups,
                "context_snapshot": sanitized_context,
            }
            records.append(record)
        return records

    def _apply_final_score_adjustments(self, df: pd.DataFrame, raw_scores: pd.Series, params: dict) -> Tuple[pd.Series, pd.DataFrame]:
        """
        【V80.0 指挥棒模型】最终得分调整层
        - 对已经计算出的原始总分，根据一系列“纯粹度”指标，进行乘数调整。
        - 这如同总指挥对乐团的最终演绎进行微调，赋予信号艺术性。
        """
        print("    - [指挥棒模型 V80.0] 启动，开始对原始总分进行最终调整...")
        
        adjustment_params = self._get_params_block(params, 'final_score_adjustments', {})
        if not self._get_param_value(adjustment_params.get('enabled'), False):
            print("      -> 最终得分调整被禁用，返回原始分数。")
            return raw_scores, pd.DataFrame(index=df.index)

        total_multiplier = pd.Series(0.0, index=df.index)
        adjustment_details_df = pd.DataFrame(index=df.index)
        multiplier_rules = adjustment_params.get('multipliers', [])

        for rule in multiplier_rules:
            name = rule.get('name')
            cn_name = rule.get('cn_name', name)
            source_col = rule.get('source_column')
            condition = rule.get('condition')
            multiplier_value = rule.get('multiplier_value', 0)

            if source_col not in df.columns:
                print(f"      -> [警告] 指挥棒规则'{cn_name}'所需的列'{source_col}'不存在，跳过。")
                continue

            mask = pd.Series(False, index=df.index)
            if condition == 'is_lowest_percentile':
                window = rule.get('window', 250)
                percentile = rule.get('percentile', 0.1)
                threshold = df[source_col].rolling(window=window, min_periods=window//2).quantile(percentile)
                mask = df[source_col] < threshold
            # 未来可以扩展更多条件，如 is_highest_percentile 等

            # 应用乘数
            total_multiplier.loc[mask] += multiplier_value
            adjustment_details_df.loc[mask, name] = multiplier_value
            
            if mask.any():
                print(f"      -> 指挥棒规则 '{cn_name}' 已激活，在 {mask.sum()} 天提供了 {multiplier_value*100:.0f}% 的质量乘数。")

        adjusted_score = raw_scores * (1 + total_multiplier)
        print("    - [指挥棒模型 V80.0] 调整完成。")
        return adjusted_score, adjustment_details_df

    def _calculate_trend_slopes(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        【V58.0 归一化与动态任务版】
        """
        print("    - [斜率中心 V58.0 归一化与动态任务版] 启动...")
        slope_params = self._get_params_block(params, 'slope_params', {})
        if not self._get_param_value(slope_params.get('enabled'), False):
            print("      -> 斜率计算被禁用，跳过。")
            return df
        series_to_slope = self._get_param_value(slope_params.get('series_to_slope'), {})
        auto_detect_patterns = self._get_param_value(slope_params.get('auto_detect_patterns'), [])
        auto_detect_lookbacks = self._get_param_value(slope_params.get('auto_detect_default_lookbacks'), [10, 20])
        for pattern in auto_detect_patterns:
            for col in df.columns:
                if pattern in col and col not in series_to_slope:
                    series_to_slope[col] = auto_detect_lookbacks
                    print(f"      -> [动态注入] 发现新特征 '{col}'，已加入斜率计算任务。")
        for col_name, lookbacks in series_to_slope.items():
            if col_name not in df.columns:
                print(f"      -> [警告] 列 '{col_name}' 不存在，跳过斜率计算。")
                continue
            source_series = df[col_name].astype(float)
            for lookback in lookbacks:
                min_p = max(2, lookback // 2)
                slope_col_name = f'SLOPE_{lookback}_{col_name}'
                linreg_result = df.ta.linreg(close=source_series, length=lookback, min_periods=min_p, slope=True, intercept=False, r=False)
                if isinstance(linreg_result, pd.DataFrame):
                    slope_series = linreg_result.iloc[:, 0]
                elif isinstance(linreg_result, pd.Series):
                    slope_series = linreg_result
                else:
                    slope_series = pd.Series(np.nan, index=df.index)
                df[slope_col_name] = slope_series.fillna(0)
                accel_col_name = f'ACCEL_{lookback}_{col_name}'
                if not df[slope_col_name].dropna().empty:
                    accel_linreg_result = df.ta.linreg(close=df[slope_col_name], length=lookback, min_periods=min_p, slope=True, intercept=False, r=False)
                    if isinstance(accel_linreg_result, pd.DataFrame):
                        accel_series = accel_linreg_result.iloc[:, 0]
                    elif isinstance(accel_linreg_result, pd.Series):
                        accel_series = accel_linreg_result
                    else:
                        accel_series = pd.Series(np.nan, index=df.index)
                    df[accel_col_name] = accel_series.fillna(0)
                else:
                    df[accel_col_name] = np.nan
                norm_slope_col_name = f'SLOPE_NORM_{lookback}_{col_name}'
                slope_std = df[slope_col_name].rolling(window=lookback * 2).std()
                df[norm_slope_col_name] = np.divide(df[slope_col_name], slope_std, out=np.zeros_like(df[slope_col_name], dtype=float), where=slope_std!=0)
            # print(f"        -> 完成对 '{col_name}' 的所有斜率计算 (周期: {lookbacks})。")
        print("    - [斜率中心 V58.0] 所有斜率计算完成。")
        return df

    # ▼▼▼ 剧本的静态“蓝图”知识库 ▼▼▼
    def _get_playbook_blueprints(self) -> List[Dict]:
        """
        【V83.0 新增】剧本蓝图知识库
        - 职责: 定义所有剧本的静态属性（名称、家族、评分规则等）。
        - 特性: 这是一个纯粹的数据结构，不依赖任何动态数据，可以在初始化时被安全地缓存。
        """
        return [
            # --- 反转/逆势家族 (REVERSAL_CONTRARIAN) ---
            {
                'name': 'ABYSS_GAZE_S', 'cn_name': '【S级】深渊凝视', 'family': 'REVERSAL_CONTRARIAN',
                'type': 'setup', 'score': 320, 'side': 'left', 'comment': 'S级: 市场极度恐慌后的第一个功能性反转。'
            },
            {
                'name': 'CAPITULATION_PIT_REVERSAL', 'cn_name': '【动态】投降坑反转', 'family': 'REVERSAL_CONTRARIAN',
                'type': 'setup_score', 'side': 'left', 'comment': '根据坑的深度和反转K强度动态给分。',
                'scoring_rules': { 'min_setup_score_to_trigger': 51, 'base_score': 160, 'score_multiplier': 1.0, 'trigger_bonus': {'TRIGGER_REVERSAL_CONFIRMATION_CANDLE': 50}}
            },
            {
                'name': 'CAPITAL_DIVERGENCE_REVERSAL', 'cn_name': '【A-级】资本逆行者', 'family': 'REVERSAL_CONTRARIAN',
                'type': 'setup', 'score': 230, 'side': 'left', 'comment': 'A-级: 在“底背离”机会窗口中出现的反转确认。'
            },
            {
                'name': 'BEAR_TRAP_RALLY', 'cn_name': '【C+级】熊市反弹', 'family': 'REVERSAL_CONTRARIAN',
                'type': 'setup', 'score': 180, 'side': 'left', 'comment': 'C+级: 熊市背景下，价格首次钝化，并由长期趋势企稳信号触发。'
            },
            {
                'name': 'WASHOUT_REVERSAL_A', 'cn_name': '【A级】巨阴洗盘反转', 'family': 'REVERSAL_CONTRARIAN',
                'type': 'setup', 'score': 260, 'side': 'left', 'comment': 'A级: 极端洗盘后的拉升前兆。'
            },
            {
                'name': 'BOTTOM_STABILIZATION_B', 'cn_name': '【B级】底部企稳', 'family': 'REVERSAL_CONTRARIAN',
                'type': 'setup', 'score': 190, 'side': 'left', 'comment': 'B级: 股价严重超卖偏离均线后，出现企稳阳线。'
            },
            # --- 趋势/动能家族 (TREND_MOMENTUM) ---
            {
                'name': 'DEEP_ACCUMULATION_BREAKOUT', 'cn_name': '【动态】潜龙出海', 'family': 'TREND_MOMENTUM',
                'type': 'setup_score', 'side': 'right', 'comment': '根据深度吸筹分数动态给分。若伴随筹码点火，则确定性更高。',
                'scoring_rules': { 'min_setup_score_to_trigger': 51, 'base_score': 200, 'score_multiplier': 1.5, 'trigger_bonus': {'CHIP_EVENT_IGNITION': 60} }
            },
            {
                'name': 'ENERGY_COMPRESSION_BREAKOUT', 'cn_name': '【动态】能量压缩突破', 'family': 'TREND_MOMENTUM',
                'type': 'precondition_score', 'side': 'right', 'comment': '根据当天前提和历史准备共同给分。若伴随筹码点火，则确定性更高。',
                'scoring_rules': { 'base_score': 150, 'min_score_to_trigger': 180, 'conditions': {'VOL_STATE_SQUEEZE_WINDOW': 50, 'CHIP_STATE_CONCENTRATION_SQUEEZE': 30, 'MA_STATE_CONVERGING': 20}, 'setup_bonus': {'ENERGY_COMPRESSION': 0.2, 'HEALTHY_MARKUP': 0.1}, 'trigger_bonus': {'CHIP_EVENT_IGNITION': 60} }
            },
            {
                'name': 'PLATFORM_SUPPORT_PULLBACK', 'cn_name': '【动态】平台支撑回踩', 'family': 'TREND_MOMENTUM',
                'type': 'precondition_score', 'side': 'right', 'comment': '根据平台质量、支撑级别和主升浪背景动态给分。',
                'scoring_rules': { 'base_score': 160, 'min_score_to_trigger': 180, 'conditions': {'MA_STATE_CONVERGING': 30, 'CHIP_STATE_CONCENTRATION_SQUEEZE': 20}, 'event_conditions': {'MA_STATE_TOUCHING_SUPPORT_21': 10, 'MA_STATE_TOUCHING_SUPPORT_34': 20, 'MA_STATE_TOUCHING_SUPPORT_55': 35, 'MA_STATE_TOUCHING_SUPPORT_89': 50, 'MA_STATE_TOUCHING_SUPPORT_144': 65, 'MA_STATE_TOUCHING_SUPPORT_233': 80}, 'setup_bonus': {'HEALTHY_MARKUP': 0.2}}
            },
            {
                'name': 'HEALTHY_BOX_BREAKOUT', 'cn_name': '【A-级】健康箱体突破', 'family': 'TREND_MOMENTUM',
                'type': 'setup', 'score': 220, 'side': 'right', 'comment': 'A-级: 在一个位于趋势均线上方的健康箱体内盘整后，发生的向上突破。'
            },
            {
                'name': 'HEALTHY_MARKUP_A', 'cn_name': '【A级】健康主升浪', 'family': 'TREND_MOMENTUM',
                'type': 'setup_score', 'side': 'right', 'comment': 'A级: 在主升浪中回踩或延续，根据主升浪健康分动态加成。',
                'scoring_rules': { 
                    'min_setup_score_to_trigger': 60, # 要求主升浪健康分至少达到60
                    'base_score': 240, 
                    'score_multiplier': 1.2, # 允许根据健康分进行加成
                    'conditions': { 'MA_STATE_DIVERGING': 20, 'OSC_STATE_MACD_BULLISH': 15 } 
                }
            },
            {
                'name': 'N_SHAPE_CONTINUATION_A', 'cn_name': '【A级】N字板接力', 'family': 'TREND_MOMENTUM',
                'type': 'precondition_score', 'side': 'right', 'comment': 'A级: 强势股的经典趋势中继信号。若伴随筹码点火，则确定性更高。',
                'scoring_rules': { 'base_score': 250, 'min_score_to_trigger': 250, 'conditions': {'SETUP_SCORE_N_SHAPE_CONTINUATION_ABOVE_80': 0}, 'trigger_bonus': {'CHIP_EVENT_IGNITION': 60} }
            },
            {
                'name': 'GAP_SUPPORT_PULLBACK_B_PLUS', 'cn_name': '【B+级】缺口支撑回踩', 'family': 'TREND_MOMENTUM',
                'type': 'setup', 'score': 190, 'side': 'right', 'comment': 'B+级: 回踩前期跳空缺口获得支撑并反弹。'
            },
            # --- 特殊事件家族 (SPECIAL_EVENT) ---
            {
                'name': 'EARTH_HEAVEN_BOARD', 'cn_name': '【S+】地天板', 'family': 'SPECIAL_EVENT',
                'type': 'event_driven', 'score': 380, 'side': 'left', 'comment': '市场情绪的极致反转。'
            },
        ]

    # ▼▼▼ 此方法现在是“水合”引擎 ▼▼▼
    def _get_playbook_definitions(self, df: pd.DataFrame, trigger_events: Dict[str, pd.Series], setup_scores: Dict[str, pd.Series], atomic_states: Dict[str, pd.Series]) -> List[Dict]:
        """
        【V85.1 右侧增强与结构优化版】
        - 职责: 接收静态的剧本“蓝图”，并用动态计算出的数据进行“水合”。
        - 核心升级:
          1. 为“健康主升浪”(HEALTHY_MARKUP_A)剧本增加了新的“趋势延续确认K线”(TRIGGER_TREND_CONTINUATION_CANDLE)触发器，
             使其能捕捉更多样的右侧买点，解决了“温和上涨无信号”的问题。
          2. 优化了代码结构，将原本分散和重复的触发器分配逻辑进行了整合，移除了冗余的 `if playbook['type'] == 'precondition_score'` 代码块，
             使每个剧本的触发器分配都在唯一的 `if/elif` 分支中完成，逻辑更清晰。
        """
        print("    - [剧本水合引擎 V85.2 蓝图修正版] 启动...")
        
        # 步骤1: 深度复制蓝图，以防修改缓存的原始版本
        hydrated_playbooks = deepcopy(self.playbook_blueprints)
        
        # 步骤2: 准备所有需要用到的动态数据
        default_series = pd.Series(False, index=df.index)
        
        # 准备分
        score_cap_pit = setup_scores.get('SETUP_SCORE_CAPITULATION_PIT', pd.Series(0, index=df.index))
        score_deep_accum = setup_scores.get('SETUP_SCORE_DEEP_ACCUMULATION', pd.Series(0, index=df.index))
        score_nshape_cont = setup_scores.get('SETUP_SCORE_N_SHAPE_CONTINUATION', pd.Series(0, index=df.index))
        score_gap_support = setup_scores.get('SETUP_SCORE_GAP_SUPPORT_PULLBACK', pd.Series(0, index=df.index))
        score_bottoming_process = setup_scores.get('SETUP_SCORE_BOTTOMING_PROCESS', pd.Series(0, index=df.index))
        score_healthy_markup = setup_scores.get('SETUP_SCORE_HEALTHY_MARKUP', pd.Series(0, index=df.index))

        # 原子状态
        capital_divergence_window = atomic_states.get('CAPITAL_STATE_DIVERGENCE_WINDOW', default_series)
        setup_bottom_passivation = atomic_states.get('MA_STATE_BOTTOM_PASSIVATION', default_series)
        setup_washout_reversal = atomic_states.get('KLINE_STATE_WASHOUT_WINDOW', default_series)
        setup_healthy_box = atomic_states.get('BOX_STATE_HEALTHY_CONSOLIDATION', default_series)
        
        # 动态布尔条件
        atomic_states['SETUP_SCORE_N_SHAPE_CONTINUATION_ABOVE_80'] = score_nshape_cont > 80
        atomic_states['SETUP_SCORE_HEALTHY_MARKUP_ABOVE_60'] = score_healthy_markup > 60

        # 步骤3: 遍历蓝图，注入动态数据
        for playbook in hydrated_playbooks:
            name = playbook['name']
            
            # 为每个剧本注入其专属的动态数据
            if name == 'ABYSS_GAZE_S':
                playbook['trigger'] = trigger_events.get('TRIGGER_DOMINANT_REVERSAL', default_series)
                playbook['setup'] = score_cap_pit > 80
            elif name == 'CAPITULATION_PIT_REVERSAL':
                playbook['setup_score_series'] = score_cap_pit
                playbook['trigger'] = (trigger_events.get('TRIGGER_REVERSAL_CONFIRMATION_CANDLE', default_series) | trigger_events.get('TRIGGER_BREAKOUT_CANDLE', default_series))
            elif name == 'CAPITAL_DIVERGENCE_REVERSAL':
                playbook['setup'] = capital_divergence_window
                playbook['trigger'] = trigger_events.get('TRIGGER_REVERSAL_CONFIRMATION_CANDLE', default_series)
            elif name == 'BEAR_TRAP_RALLY':
                playbook['setup'] = setup_bottom_passivation
                playbook['trigger'] = trigger_events.get('TRIGGER_TREND_STABILIZING', default_series)
            elif name == 'WASHOUT_REVERSAL_A':
                playbook['setup'] = setup_washout_reversal
                playbook['trigger'] = trigger_events.get('TRIGGER_BREAKOUT_CANDLE', default_series)
            elif name == 'BOTTOM_STABILIZATION_B':
                playbook['setup'] = score_bottoming_process > 50
                playbook['trigger'] = trigger_events.get('TRIGGER_BREAKOUT_CANDLE', default_series)
            elif name == 'DEEP_ACCUMULATION_BREAKOUT':
                playbook['setup_score_series'] = score_deep_accum
                playbook['trigger'] = trigger_events.get('TRIGGER_BREAKOUT_CANDLE', default_series)
            elif name == 'ENERGY_COMPRESSION_BREAKOUT':
                # 修正：能量压缩突破使用更合适的突破型触发器
                playbook['trigger'] = trigger_events.get('TRIGGER_BREAKOUT_CANDLE', default_series)
            elif name == 'PLATFORM_SUPPORT_PULLBACK':
                # 平台支撑回踩使用标准的回踩反弹触发器
                playbook['trigger'] = trigger_events.get('TRIGGER_PULLBACK_REBOUND', default_series)
            elif name == 'HEALTHY_MARKUP_A':
                playbook['setup_score_series'] = score_healthy_markup # 分配准备分序列
                trigger_rebound = trigger_events.get('TRIGGER_PULLBACK_REBOUND', default_series)
                trigger_continuation = trigger_events.get('TRIGGER_TREND_CONTINUATION_CANDLE', default_series)
                playbook['trigger'] = trigger_rebound | trigger_continuation # 分配双重触发器
            elif name == 'HEALTHY_BOX_BREAKOUT':
                playbook['setup'] = setup_healthy_box
                playbook['trigger'] = trigger_events.get('BOX_EVENT_BREAKOUT', default_series)
            elif name == 'N_SHAPE_CONTINUATION_A':
                playbook['trigger'] = trigger_events.get('TRIGGER_N_SHAPE_BREAKOUT', default_series)
            elif name == 'GAP_SUPPORT_PULLBACK_B_PLUS':
                playbook['setup'] = score_gap_support > 60
                playbook['trigger'] = trigger_events.get('TRIGGER_PULLBACK_REBOUND', default_series)
            elif name == 'EARTH_HEAVEN_BOARD':
                playbook['trigger'] = trigger_events.get('TRIGGER_EARTH_HEAVEN_BOARD', default_series)

        print(f"    - [剧本水合引擎 V85.2] 完成，所有剧本已注入动态数据。")
        return hydrated_playbooks

    def _calculate_entry_score(
        self, 
        df: pd.DataFrame, 
        params: dict, 
        trigger_events: Dict[str, pd.Series], 
        setup_scores: Dict[str, pd.Series],
        atomic_states: Dict[str, pd.Series]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        【V79.0 家族继承引擎版】
        - 核心革命: 实现“胜者为王，败者为辅”的家族继承范式，这是逻辑的最终形态。
        - 1. 基础分竞价: 家族内剧本仅凭 base_score 竞争“主剧本”地位。
        - 2. 家族加分池: 所有被触发的剧本，无论胜败，其所有加分项都被汇入“家族加分池”。
        - 3. 最终得分 = 主剧本的base_score + 家族加分池总和。
        - 这既避免了基础分的重复计算，又聚合了所有有价值的信号，讲述了最完整的故事。
        """
        print("    - [计分引擎 V79.0 家族继承引擎版] 启动...")
        
        playbook_definitions = self._get_playbook_definitions(df, trigger_events, setup_scores, atomic_states)
        default_series = pd.Series(False, index=df.index)
        context_window = self._get_param_value(
            self._get_params_block(params, 'entry_scoring_params', {}).get('context_window'), 10
        )
        
        # ==================== 步骤1: 向量化预计算所有“基础分”和“加分项” ====================
        print("      -> 步骤1: 向量化预计算所有基础分和加分项...")
        base_scores_df = pd.DataFrame(index=df.index)
        bonus_scores_df = pd.DataFrame(index=df.index) # 存储所有加分项的总和

        for playbook in playbook_definitions:
            name = playbook['name']
            rules = playbook.get('scoring_rules', {})
            
            # 通用过滤器
            trigger_mask = playbook.get('trigger', default_series)
            side_mask = (df['close_D'] > df.get('EMA_55_D', -1)) if playbook.get('side') == 'right' else pd.Series(True, index=df.index)
            valid_mask = trigger_mask & side_mask

            # ▼▼▼ 为 'setup' 类型剧本增加当天有效性检查 ▼▼▼
            setup_mask = pd.Series(True, index=df.index) # 默认为True
            if playbook.get('type') == 'setup':
                setup_mask = playbook.get('setup', default_series)
            elif playbook.get('type') == 'setup_score':
                # 对于动态评分剧本，检查当天的准备分是否满足最低触发门槛
                min_score_req = rules.get('min_setup_score_to_trigger', 0)
                if min_score_req > 0:
                    setup_score_series = playbook.get('setup_score_series', default_series)
                    setup_mask = setup_score_series >= min_score_req
            valid_mask = trigger_mask & side_mask & setup_mask

            # 计算基础分 (只在有效时才有分)
            base_score = rules.get('base_score', playbook.get('score', 0))
            base_scores_df[name] = valid_mask.astype(int) * base_score

            # 计算该剧本的所有加分项总和
            playbook_bonus = pd.Series(0.0, index=df.index)
            if rules:
                condition_bonus = sum(atomic_states.get(s, default_series).astype(int) * v for s, v in rules.get('conditions', {}).items())
                event_bonus = sum(atomic_states.get(s, default_series).astype(int) * v for s, v in rules.get('event_conditions', {}).items())
                setup_bonus = sum(setup_scores.get(f'SETUP_SCORE_{s}', default_series).rolling(window=context_window, min_periods=1).max().fillna(0) * v for s, v in rules.get('setup_bonus', {}).items())
                trigger_bonus = sum(trigger_events.get(s, default_series).astype(int) * v for s, v in rules.get('trigger_bonus', {}).items())
                
                # 针对 setup_score 的特殊乘数加成
                setup_multiplier_bonus = pd.Series(0.0, index=df.index)
                if playbook.get('type') == 'setup_score':
                    setup_score_series = playbook.get('setup_score_series', default_series)
                    max_setup_in_context = setup_score_series.rolling(window=context_window, min_periods=1).max()
                    multiplier = rules.get('score_multiplier', 1.0)
                    # 基础分已在base_score中计算，这里只计算乘数带来的额外加成
                    setup_multiplier_bonus = max_setup_in_context * (multiplier - 1) if multiplier > 1 else pd.Series(0.0, index=df.index)

                playbook_bonus = condition_bonus + event_bonus + setup_bonus + trigger_bonus + setup_multiplier_bonus
            
            bonus_scores_df[name] = playbook_bonus * valid_mask # 只有有效时，加分项才被考虑

        # ==================== 步骤2: 向量化执行“家族继承”逻辑 ====================
        print("      -> 步骤2: 向量化执行“家族继承”...")
        families = {p['name']: p.get('family', 'UNCATEGORIZED') for p in playbook_definitions}
        
        # 2.1 基础分竞价，确立优胜者
        base_scores_grouped = base_scores_df.groupby(by=families, axis=1)
        family_winner_names = base_scores_grouped.idxmax()
        
        # 2.2 汇集家族加分池 (对每个家族，将所有成员的加分项相加)
        bonus_scores_grouped = bonus_scores_df.groupby(by=families, axis=1)
        family_bonus_pool = bonus_scores_grouped.sum()

        # 2.3 计算优胜者的基础分
        winner_base_scores = base_scores_grouped.max()

        # 2.4 最终得分加冕 = 优胜者基础分 + 家族加分池
        final_family_scores = winner_base_scores + family_bonus_pool
        
        # 最终总分是所有家族得分的总和
        final_score = final_family_scores.sum(axis=1)

        # ==================== 步骤3: 填充细节并进行日志输出 ====================
        print("      -> 步骤3: 生成最终得分详情...")
        score_details_df = pd.DataFrame(0.0, index=df.index, columns=base_scores_df.columns)
        for family_name in final_family_scores.columns:
            winner_names_series = family_winner_names[family_name]
            family_total_score_series = final_family_scores[family_name]
            
            for playbook_name in winner_names_series.unique():
                if pd.isna(playbook_name): continue
                mask = (winner_names_series == playbook_name) & (family_total_score_series > 0)
                # 记录的是整个家族的得分，归功于主剧本
                score_details_df.loc[mask, playbook_name] = family_total_score_series[mask]

        # ... (日志和收尾部分与V78.0类似，但现在能讲述更完整的故事) ...
        print("\n--- 剧本触发详情 (家族继承模式) ---")
        probe_start_date = self._get_param_value(params.get('probe_start_date'), '2025-06-01')
        key_dates = df.index[final_score > 0]
        key_dates = key_dates[key_dates >= probe_start_date]

        for date in key_dates:
            print(f"{date.strftime('%Y-%m-%d')} | 最终总分: {final_score.loc[date]:.0f}")
            winning_playbooks = score_details_df.loc[date][score_details_df.loc[date] > 0]
            for name, score in winning_playbooks.items():
                playbook_info = next((p for p in playbook_definitions if p['name'] == name), None)
                if playbook_info:
                    cn_name = playbook_info.get('cn_name', name)
                    family = playbook_info.get('family', 'N/A')
                    base_score = base_scores_df.loc[date, name]
                    bonus_score = score - base_score
                    print(f"             -> 家族[{family}]主剧本: '{cn_name}', 基础分:{base_score:.0f}, 家族加成:{bonus_score:.0f}, 家族总分:{score:.0f}")
        print("========================= 全局剧本探针分析结束 =========================")

        df['entry_score'] = final_score.round(0)
        score_details_df.fillna(0, inplace=True)
        
        print(f"\n--- [计分引擎 V79.0] 计算完成。最终有 { (final_score > 0).sum() } 个交易日产生得分。 ---")
        
        return df, score_details_df

    def _calculate_risk_score(self, df: pd.DataFrame, params: dict, risk_factors: Dict[str, pd.Series]) -> pd.Series:
        """
        【V57.0 风险评分引擎】
        """
        print("    - [风险评分引擎 V57.0] 启动，开始量化每日风险...")
        risk_score = pd.Series(0.0, index=df.index)
        risk_matrix = self._get_params_block(params, 'exit_risk_scoring_matrix', {})
        if not risk_matrix:
            print("      -> [警告] 未找到风险评分矩阵，风险分数为0。")
            return risk_score
        for factor_name, points in risk_matrix.items():
            factor_signal = risk_factors.get(factor_name, pd.Series(False, index=df.index))
            if factor_signal.any():
                score_to_add = self._get_param_value(points, 0)
                risk_score.loc[factor_signal] += score_to_add
                print(f"      -> 风险因子 '{factor_name}' 触发，风险分 +{score_to_add}")
        df['risk_score'] = risk_score
        print(f"    - [风险评分引擎 V57.0] 风险评分完成，最高风险分: {risk_score.max():.0f}")
        return risk_score

    def _calculate_exit_signals(self, df: pd.DataFrame, params: dict, risk_score: pd.Series) -> pd.Series:
        """
        【V57.0 出场决策引擎】
        """
        print("    - [出场决策引擎 V57.0] 启动，开始根据风险分做出决策...")
        threshold_params = self._get_params_block(params, 'exit_threshold_params', {})
        if not threshold_params:
            return pd.Series(0, index=df.index)
        levels = sorted(threshold_params.items(), key=lambda item: self._get_param_value(item[1].get('level')), reverse=True)
        conditions = []
        choices = []
        for level_name, config in levels:
            threshold = self._get_param_value(config.get('level'))
            exit_code = self._get_param_value(config.get('code'))
            conditions.append(risk_score >= threshold)
            choices.append(exit_code)
            print(f"      -> 定义决策规则: 风险分 >= {threshold}，则出场，代码: {exit_code}")
        exit_signal = np.select(conditions, choices, default=0)
        final_exit_signal = pd.Series(exit_signal, index=df.index)
        print(f"    - [出场决策引擎 V57.0] 决策完成，共产生 { (final_exit_signal > 0).sum() } 个出场信号。")
        return final_exit_signal

    def _calculate_setup_conditions(self, df: pd.DataFrame, params: dict, atomic_states: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V65.0 投降坑识别版】
        - 核心升级: 为'CAPITULATION_PIT'引入专属评分逻辑，使其能识别并重奖“筹码发散”的投降式崩盘。
        """
        print("    - [准备状态中心 V65.0 投降坑识别版] 启动...")
        setup_scores = {}
        default_series = pd.Series(False, index=df.index)
        scoring_matrix = self._get_params_block(params, 'setup_scoring_matrix', {})
        for setup_name, rules in scoring_matrix.items():
            if not self._get_param_value(rules.get('enabled'), True):
                continue
            print(f"          -> 正在评审 '{setup_name}'...")
            
            # ▼▼▼ 为“投降坑”设置专属评分逻辑 ▼▼▼
            if setup_name == 'CAPITULATION_PIT':
                p_cap_pit = self._get_params_block(params, 'setup_scoring_matrix', {}).get('CAPITULATION_PIT', {})
                must_have_score = self._get_param_value(p_cap_pit.get('must_have_score'), 40)
                bonus_score = self._get_param_value(p_cap_pit.get('bonus_score'), 25)
                
                # 核心条件：价格必须处于负向乖离的超卖区
                must_have_conditions = atomic_states.get('OPP_STATE_NEGATIVE_DEVIATION', default_series)
                
                # 加分项：获利盘极低 + 筹码极度发散 (这才是投降的真正信号!)
                bonus_conditions_1 = atomic_states.get('CHIP_STATE_LOW_PROFIT', default_series)
                bonus_conditions_2 = atomic_states.get('CHIP_STATE_SCATTERED', default_series)
                
                base_score = must_have_conditions.astype(int) * must_have_score
                bonus_score_total = (bonus_conditions_1.astype(int) * bonus_score) + (bonus_conditions_2.astype(int) * bonus_score)
                
                final_score = (base_score + bonus_score_total).where(must_have_conditions, 0)
                setup_scores[f'SETUP_SCORE_{setup_name}'] = final_score
            else:
                # 其他所有剧本使用通用评分逻辑
                current_score = pd.Series(0.0, index=df.index)
                must_have_rules = rules.get('must_have', {})
                must_have_passed = pd.Series(True, index=df.index)
                for state, score in must_have_rules.items():
                    state_series = atomic_states.get(state, default_series)
                    current_score += state_series * score
                    must_have_passed &= state_series
                any_of_rules = rules.get('any_of_must_have', {})
                any_of_passed = pd.Series(False, index=df.index)
                if any_of_rules:
                    any_of_score_component = pd.Series(0.0, index=df.index)
                    for state, score in any_of_rules.items():
                        state_series = atomic_states.get(state, default_series)
                        any_of_score_component.loc[state_series] = score
                        any_of_passed |= state_series
                    current_score += any_of_score_component
                else:
                    any_of_passed = pd.Series(True, index=df.index)
                bonus_rules = rules.get('bonus', {})
                for state, score in bonus_rules.items():
                    state_series = atomic_states.get(state, default_series)
                    current_score += state_series * score
                final_validity = must_have_passed & any_of_passed
                setup_scores[f'SETUP_SCORE_{setup_name}'] = current_score.where(final_validity, 0)

            max_score = setup_scores.get(f'SETUP_SCORE_{setup_name}', pd.Series(0)).max()
            print(f"            -> '{setup_name}' 评审完成，最高置信度得分: {max_score:.0f}")
        print("    - [准备状态中心 V65.0] 所有状态置信度评审完成。")
        return setup_scores

    def _define_trigger_events(self, df: pd.DataFrame, params: dict, atomic_states: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V66.1 结构为王最终版】
        - 新增 TRIGGER_PANIC_REVERSAL，专门捕捉恐慌坑后的“功能性”强反转K线。
        """
        print("    - [触发事件中心 V66.1 结构为王最终版] 启动，开始定义所有原子化触发事件...")
        triggers = {}
        default_series = pd.Series(False, index=df.index) # 新增一个默认的空Series，用于后续代码健壮性
        trigger_params = self._get_params_block(params, 'trigger_event_params', {})
        if not self._get_param_value(trigger_params.get('enabled'), True):
            print("      -> 触发事件引擎被禁用，跳过。")
            return triggers
        vol_ma_col = 'VOL_MA_21_D'
        if 'CHIP_EVENT' in df.columns:
            triggers['TRIGGER_CHIP_IGNITION'] = (df['CHIP_EVENT'] == 'IGNITION')
            print(f"      -> '筹码点火' 事件定义完成，发现 {triggers['TRIGGER_CHIP_IGNITION'].sum()} 天。")

        p_candle = trigger_params.get('positive_candle', {})
        if self._get_param_value(p_candle.get('enabled'), True):
            is_green = df['close_D'] > df['open_D']
            body_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
            body_ratio = (df['close_D'] - df['open_D']) / body_range
            is_strong_body = body_ratio > self._get_param_value(p_candle.get('min_body_ratio'), 0.6)
            triggers['TRIGGER_STRONG_POSITIVE_CANDLE'] = is_green & is_strong_body
            
            signal = triggers.get('TRIGGER_STRONG_POSITIVE_CANDLE', default_series)
            dates_str = self._format_debug_dates(signal)
            print(f"      -> '强势阳线' 事件定义完成，发现 {signal.sum()} 天。{dates_str}")

        # --- 步骤1: 定义标准的“反转确认阳线” (通用级) ---
        p_reversal = trigger_params.get('reversal_confirmation_candle', {})
        if self._get_param_value(p_reversal.get('enabled'), True):
            is_green = df['close_D'] > df['open_D']
            is_strong_rally = df['pct_change_D'] > self._get_param_value(p_reversal.get('min_pct_change'), 0.03)
            is_closing_strong = df['close_D'] > (df['high_D'] + df['low_D']) / 2
            triggers['TRIGGER_REVERSAL_CONFIRMATION_CANDLE'] = is_green & is_strong_rally & is_closing_strong
            
            signal = triggers.get('TRIGGER_REVERSAL_CONFIRMATION_CANDLE', default_series)
            dates_str = self._format_debug_dates(signal)
            print(f"      -> '反转确认阳线' 事件定义完成，发现 {signal.sum()} 天。{dates_str}")

        # --- 步骤2: 定义新增的“显性反转阳线” (精英级) ---
        p_dominant = trigger_params.get('dominant_reversal_candle', {}) # 允许未来在JSON中配置
        if self._get_param_value(p_dominant.get('enabled'), True):
            # 条件A: 必须首先是一个高质量的通用反转信号
            base_reversal_signal = triggers.get('TRIGGER_REVERSAL_CONFIRMATION_CANDLE', default_series)
            
            # 条件B: 今天的阳线实体，必须在力量上压制昨天的阴线实体
            today_body_size = df['close_D'] - df['open_D']
            yesterday_body_size = abs(df['close_D'].shift(1) - df['open_D'].shift(1))
            
            # 只有当昨天是阴线时，才进行此项比较
            was_yesterday_red = df['close_D'].shift(1) < df['open_D'].shift(1)
            
            # 恢复比例，默认为0.5，即要求收复昨天实体的一半
            recovery_ratio = self._get_param_value(p_dominant.get('recovery_ratio'), 0.5)
            
            is_power_recovered = today_body_size >= (yesterday_body_size * recovery_ratio)
            
            # 最终的精英信号 = 高质量反转 AND (昨天不是阴线 OR 力量已压制)
            triggers['TRIGGER_DOMINANT_REVERSAL'] = base_reversal_signal & (~was_yesterday_red | is_power_recovered)
            
            signal = triggers.get('TRIGGER_DOMINANT_REVERSAL', default_series)
            print(f"      -> '显性反转阳线(精英级)' 事件定义完成，发现 {signal.sum()} 天。")
            
        p_breakout = trigger_params.get('volume_spike_breakout', {})
        if self._get_param_value(p_breakout.get('enabled'), True) and vol_ma_col in df.columns:
            volume_ratio = self._get_param_value(p_breakout.get('volume_ratio'), 2.0)
            lookback = self._get_param_value(p_breakout.get('lookback_period'), 20)
            is_volume_spike = df['volume_D'] > df[vol_ma_col] * volume_ratio
            is_price_breakout = df['close_D'] > df['high_D'].shift(1).rolling(lookback).max()
            triggers['TRIGGER_VOLUME_SPIKE_BREAKOUT'] = is_volume_spike & is_price_breakout
            
            signal = triggers.get('TRIGGER_VOLUME_SPIKE_BREAKOUT', default_series)
            dates_str = self._format_debug_dates(signal)
            print(f"      -> '放量突破近期高点' 事件定义完成，发现 {signal.sum()} 天。{dates_str}")
            
        p_rebound = self._get_params_block(params, 'trigger_event_params', {}).get('pullback_rebound_trigger_params', {})
        if self._get_param_value(p_rebound.get('enabled'), True):
            support_ma_period = self._get_param_value(p_rebound.get('support_ma'), 21)
            support_ma_col = f'EMA_{support_ma_period}_D'
            if support_ma_col in df.columns:
                was_touching_support = df['low_D'].shift(1) <= df[support_ma_col].shift(1)
                is_rebounded_above = df['close_D'] > df[support_ma_col]
                is_positive_day = df['close_D'] > df['open_D']
                triggers['TRIGGER_PULLBACK_REBOUND'] = was_touching_support & is_rebounded_above & is_positive_day
                
                signal = triggers.get('TRIGGER_PULLBACK_REBOUND', default_series)
                dates_str = self._format_debug_dates(signal)
                print(f"      -> '回踩反弹' 触发器定义完成，发现 {signal.sum()} 天。{dates_str}")

        # ▼▼▼ 趋势延续触发器 ▼▼▼
        p_cont = trigger_params.get('trend_continuation_candle', {})
        if self._get_param_value(p_cont.get('enabled'), True):
            lookback_period = self._get_param_value(p_cont.get('lookback_period'), 8)
            is_positive_day = df['close_D'] > df['open_D']
            is_new_high = df['close_D'] >= df['high_D'].shift(1).rolling(window=lookback_period).max()
            triggers['TRIGGER_TREND_CONTINUATION_CANDLE'] = is_positive_day & is_new_high
            
            signal = triggers.get('TRIGGER_TREND_CONTINUATION_CANDLE', default_series)
            print(f"      -> '趋势延续确认K线' 触发器定义完成 (周期:{lookback_period})，发现 {signal.sum()} 天。{self._format_debug_dates(signal)}")

        p_nshape = self._get_params_block(params, 'kline_pattern_params', {}).get('n_shape_params', {})
        if self._get_param_value(p_nshape.get('enabled'), True):
            is_positive_day = df['close_D'] > df['open_D']
            n_shape_consolidation_state = atomic_states.get('KLINE_STATE_N_SHAPE_CONSOLIDATION', pd.Series(False, index=df.index))
            consolidation_high = df['high_D'].where(n_shape_consolidation_state, np.nan).ffill()
            is_breaking_consolidation = df['close_D'] > consolidation_high.shift(1)
            is_volume_ok = df['volume_D'] > df.get(vol_ma_col, 0)
            triggers['TRIGGER_N_SHAPE_BREAKOUT'] = is_positive_day & is_breaking_consolidation & is_volume_ok
            
            signal = triggers.get('TRIGGER_N_SHAPE_BREAKOUT', default_series)
            dates_str = self._format_debug_dates(signal)
            print(f"      -> 'N字形态突破' 专属事件定义完成，发现 {signal.sum()} 天。{dates_str}")

        p_cross = trigger_params.get('indicator_cross_params', {})
        if self._get_param_value(p_cross.get('enabled'), True):
            if self._get_param_value(p_cross.get('dmi_cross', {}).get('enabled'), True):
                pdi_col, mdi_col = 'PDI_14_D', 'NDI_14_D'
                if all(c in df.columns for c in [pdi_col, mdi_col]):
                    triggers['TRIGGER_DMI_CROSS'] = (df[pdi_col] > df[mdi_col]) & (df[pdi_col].shift(1) <= df[mdi_col].shift(1))
                    
                    signal = triggers.get('TRIGGER_DMI_CROSS', default_series)
                    dates_str = self._format_debug_dates(signal)
                    print(f"      -> 'DMI金叉' 事件定义完成，发现 {signal.sum()} 天。{dates_str}")
                    
            macd_p = p_cross.get('macd_cross', {})
            if self._get_param_value(macd_p.get('enabled'), True):
                macd_col, signal_col = 'MACD_13_34_8_D', 'MACDs_13_34_8_D'
                if all(c in df.columns for c in [macd_col, signal_col]):
                    is_golden_cross = (df[macd_col] > df[signal_col]) & (df[macd_col].shift(1) <= df[signal_col].shift(1))
                    low_level = self._get_param_value(macd_p.get('low_level'), -0.5)
                    triggers['TRIGGER_MACD_LOW_CROSS'] = is_golden_cross & (df[macd_col] < low_level)
                    
                    signal = triggers.get('TRIGGER_MACD_LOW_CROSS', default_series)
                    dates_str = self._format_debug_dates(signal)
                    print(f"      -> 'MACD低位金叉' 事件定义完成，发现 {signal.sum()} 天。{dates_str}")
                    
        box_states = self._diagnose_box_states(df, params)
        triggers['TRIGGER_BOX_BREAKOUT'] = box_states.get('BOX_EVENT_BREAKOUT', pd.Series(False, index=df.index))
        
        signal = triggers.get('TRIGGER_BOX_BREAKOUT', default_series)
        dates_str = self._format_debug_dates(signal)
        print(f"      -> '箱体突破' 事件定义完成，发现 {signal.sum()} 天。{dates_str}")
        
        board_events = self._diagnose_board_patterns(df, params)
        triggers['TRIGGER_EARTH_HEAVEN_BOARD'] = board_events.get('BOARD_EVENT_EARTH_HEAVEN', pd.Series(False, index=df.index))
        
        signal = triggers.get('TRIGGER_EARTH_HEAVEN_BOARD', default_series)
        dates_str = self._format_debug_dates(signal)
        print(f"      -> '地天板' 触发事件定义完成，发现 {signal.sum()} 天。{dates_str}")
        
        p_breakout_stabilize = trigger_params.get('breakout_candle', {}) # 修正变量名以示区分
        if self._get_param_value(p_breakout_stabilize.get('enabled'), True):
            boll_mid_col = 'BBM_21_2.0_D'
            required_cols = ['open_D', 'high_D', 'low_D', 'close_D', boll_mid_col]
            if all(col in df.columns for col in required_cols):
                min_body_ratio = self._get_param_value(p_breakout_stabilize.get('min_body_ratio'), 0.4)
                is_strong_positive_candle = (
                    (df['close_D'] > df['open_D']) &
                    (((df['close_D'] - df['open_D']) / (df['high_D'] - df['low_D']).replace(0, np.nan)).fillna(1.0) >= min_body_ratio)
                )
                is_breaking_boll_mid = df['close_D'] > df[boll_mid_col]
                triggers['TRIGGER_BREAKOUT_CANDLE'] = is_strong_positive_candle & is_breaking_boll_mid
                signal = triggers.get('TRIGGER_BREAKOUT_CANDLE', default_series)
                print(f"      -> '突破阳线(企稳型)' 触发器定义完成，发现 {signal.sum()} 天。{self._format_debug_dates(signal)}")
            else:
                print(f"      -> [警告] 缺少定义'突破阳线(企稳型)'所需的列 (如: {boll_mid_col})，跳过该触发器。")

        p_energy = trigger_params.get('energy_release', {})
        if self._get_param_value(p_energy.get('enabled'), True):
            required_cols = ['open_D', 'high_D', 'low_D', 'close_D', 'volume_D', vol_ma_col]
            if all(col in df.columns for col in required_cols):
                is_positive_day = df['close_D'] > df['open_D']
                body_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
                body_ratio = (df['close_D'] - df['open_D']) / body_range
                is_strong_body = body_ratio.fillna(1.0) > self._get_param_value(p_energy.get('min_body_ratio'), 0.5)
                volume_ratio = self._get_param_value(p_energy.get('volume_ratio'), 1.5)
                is_volume_spike = df['volume_D'] > df[vol_ma_col] * volume_ratio
                triggers['TRIGGER_ENERGY_RELEASE'] = is_positive_day & is_strong_body & is_volume_spike
                signal = triggers.get('TRIGGER_ENERGY_RELEASE', default_series)
                print(f"      -> '能量释放(突破型)' 专属事件定义完成，发现 {signal.sum()} 天。{self._format_debug_dates(signal)}")
            else:
                print(f"      -> [警告] 缺少定义'能量释放(突破型)'所需的列 (如: {vol_ma_col})，跳过该触发器。")

        triggers['TRIGGER_TREND_STABILIZING'] = atomic_states.get('MA_STATE_D_STABILIZING', default_series)

        for key in triggers:
            if triggers[key] is None:
                triggers[key] = pd.Series(False, index=df.index)
            else:
                triggers[key] = triggers[key].fillna(False)
        print("    - [触发事件中心 V66.1] 所有触发事件定义完成。")
        return triggers

    def _diagnose_chip_states(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V65.0 投降坑识别版】筹码分布与流动状态诊断
        """
        print("        -> [诊断模块] 正在执行筹码状态诊断...")
        states = {}
        default_series = pd.Series(False, index=df.index) # 新增，确保在任何分支下都有默认值
        p = self._get_params_block(params, 'chip_feature_params', {})
        if not self._get_param_value(p.get('enabled'), False):
            print("          -> 筹码诊断模块被禁用，跳过。")
            return states
        required_cols = {
            'dynamic_concentration_slope': 'CHIP_concentration_90pct_slope_5d_D',
            'dynamic_winner_rate_short': 'CHIP_winner_rate_short_term_D',
            'dynamic_winner_rate_long': 'CHIP_winner_rate_long_term_D',
            'dynamic_slope_8d': 'CHIP_peak_cost_slope_8d_D',
            'dynamic_slope_21d': 'CHIP_peak_cost_slope_21d_D',
            'dynamic_accel_21d': 'CHIP_peak_cost_accel_21d_D',
            'base_close': 'close_D'
        }
        if not all(col in df.columns for col in required_cols.values()):
            missing = [k for k, v in required_cols.items() if v not in df.columns]
            print(f"          -> [严重警告] 缺少筹码诊断所需的列: {missing}。引擎将返回空结果。")
            return states
        is_markup_base = (df[required_cols['dynamic_slope_21d']] > 0) & \
                         (df.get('CHIP_peak_cost_slope_55d', 0) > 0)
        p_dist = p.get('distribution_params', {})
        is_distributing = df[required_cols['dynamic_concentration_slope']] > self._get_param_value(p_dist.get('divergence_threshold'), 0.01)
        is_at_high = df[required_cols['base_close']] > df[required_cols['base_close']].rolling(window=55).quantile(0.8)
        is_distribution_base = is_distributing & is_at_high
        p_accum = p.get('accumulation_params', {})
        lookback_accum = self._get_param_value(p_accum.get('lookback_days'), 21)
        concentrating_days = (df[required_cols['dynamic_concentration_slope']] < 0).rolling(window=lookback_accum).sum()
        is_concentrating = concentrating_days >= (lookback_accum * self._get_param_value(p_accum.get('required_days_ratio'), 0.6))
        is_not_rising = df[required_cols['dynamic_slope_21d']] <= 0
        is_accumulation_base = is_concentrating & is_not_rising
        conditions = [is_markup_base, is_distribution_base, is_accumulation_base]
        choices = ['MARKUP', 'DISTRIBUTION', 'ACCUMULATION']
        primary_state = pd.Series(np.select(conditions, choices, default='TRANSITION'), index=df.index)
        p_struct = p.get('structure_params', {})
        if self._get_param_value(p_struct.get('enabled'), True):
            conc_col = 'CHIP_concentration_90pct_D'
            if conc_col not in df.columns:
                print(f"            -> [警告] 缺少列 '{conc_col}'，筹码结构诊断跳过。")
            else:
                conc_thresh_abs = self._get_param_value(p_struct.get('high_concentration_threshold'), 0.15)
                states['CHIP_STATE_HIGHLY_CONCENTRATED'] = df[conc_col] < conc_thresh_abs
                signal = states['CHIP_STATE_HIGHLY_CONCENTRATED']
                dates_str = self._format_debug_dates(signal)
                print(f"            -> '筹码高度集中 (绝对)' 状态诊断完成 (阈值<{conc_thresh_abs*100}%)，共激活 {signal.sum()} 天。{dates_str}")
                if self._get_param_value(p_struct.get('enable_relative_squeeze'), True):
                    squeeze_window = self._get_param_value(p_struct.get('squeeze_window'), 120)
                    squeeze_percentile = self._get_param_value(p_struct.get('squeeze_percentile'), 0.2)
                    squeeze_threshold_series = df[conc_col].rolling(window=squeeze_window).quantile(squeeze_percentile)
                    states['CHIP_STATE_CONCENTRATION_SQUEEZE'] = df[conc_col] < squeeze_threshold_series
                    signal = states['CHIP_STATE_CONCENTRATION_SQUEEZE']
                    dates_str = self._format_debug_dates(signal)
                    print(f"            -> '筹码集中度压缩 (相对)' 状态诊断完成 (周期:{squeeze_window}, 分位:{squeeze_percentile})，共激活 {signal.sum()} 天。{dates_str}")
                # ▼▼▼【代码新增 V65.0】: 新增“筹码发散”状态，用于识别投降坑 ▼▼▼
                p_scattered = p.get('scattered_params', {})
                if self._get_param_value(p_scattered.get('enabled'), True):
                    scattered_threshold_pct = self._get_param_value(p_scattered.get('threshold'), 30.0)
                    scattered_threshold_ratio = scattered_threshold_pct / 100.0 # 将百分比转换为比率
                    states['CHIP_STATE_SCATTERED'] = df[conc_col] > scattered_threshold_ratio

                    # ▼▼▼【代码新增 V65.2 深度调试探针】▼▼▼
                    # debug_date = pd.to_datetime('2025-04-09', utc=True)
                    # if debug_date in df.index:
                    #     value_on_date = df.loc[debug_date, conc_col]
                    #     is_triggered = value_on_date > scattered_threshold_ratio
                    #     print("="*80)
                    #     print(f"      -> [深度调试探针] 日期: {debug_date.date()}")
                    #     print(f"      -> 筹码集中度(90%)列名: {conc_col}")
                    #     print(f"      -> 当日实际计算值 (比率): {value_on_date:.4f}")
                    #     print(f"      -> 设定的发散阈值 (百分比): > {scattered_threshold_pct}%")
                    #     print(f"      -> 换算后的阈值 (比率): > {scattered_threshold_ratio}")
                    #     print(f"      -> 是否触发'筹码发散'状态: {is_triggered}")
                    #     print("="*80)
                    # ▲▲▲【代码新增 V65.2】▲▲▲


                    signal = states.get('CHIP_STATE_SCATTERED', default_series)
                    dates_str = self._format_debug_dates(signal)
                    print(f"            -> '筹码高度发散 (投降信号)' 状态诊断完成 (基于{conc_col} > {scattered_threshold_pct}%)，共激活 {signal.sum()} 天。{dates_str}")
                # ▲▲▲【代码新增 V65.0】▲▲▲
        states['CHIP_STATE_ACCUMULATION'] = (primary_state == 'ACCUMULATION')
        states['CHIP_STATE_MARKUP'] = (primary_state == 'MARKUP')
        states['CHIP_STATE_DISTRIBUTION'] = (primary_state == 'DISTRIBUTION')
        print("          -> 主状态诊断完成 (吸筹/拉升/派发/过渡)。")
        is_deep = concentrating_days >= (lookback_accum * self._get_param_value(p_accum.get('deep_ratio'), 0.85))
        states['CHIP_STATE_ACCUMULATION_DEEP'] = states['CHIP_STATE_ACCUMULATION'] & is_deep
        signal = states['CHIP_STATE_ACCUMULATION_DEEP']
        dates_str = self._format_debug_dates(signal)
        print(f"          -> “深度吸筹”子状态诊断完成，发现 {signal.sum()} 天。{dates_str}")
        p_capit = p.get('capitulation_params', {})
        winner_rate_col = 'winner_rate_D' # 使用基础获利盘数据
        if winner_rate_col in df.columns:
            is_washed_out = df[winner_rate_col] < self._get_param_value(p_capit.get('winner_rate_threshold'), 8.0)
            states['CHIP_STATE_LOW_PROFIT'] = is_washed_out # 将其定义为独立状态
            states['CHIP_STATE_PIT_OPPORTUNITY'] = is_washed_out & states['CHIP_STATE_ACCUMULATION']
        else:
            print(f"          -> [警告] 缺少列 '{winner_rate_col}'，无法诊断获利盘相关状态。")
            states['CHIP_STATE_LOW_PROFIT'] = default_series
            states['CHIP_STATE_PIT_OPPORTUNITY'] = default_series
        signal = states['CHIP_STATE_PIT_OPPORTUNITY']
        dates_str = self._format_debug_dates(signal)
        print(f"          -> “投降坑机会”标签诊断完成，发现 {signal.sum()} 天。{dates_str}")
        is_still_rising = df[required_cols['dynamic_slope_21d']] > 0
        is_decelerating = df[required_cols['dynamic_accel_21d']] < 0
        states['CHIP_RISK_EXHAUSTION'] = is_still_rising & is_decelerating
        signal = states['CHIP_RISK_EXHAUSTION']
        dates_str = self._format_debug_dates(signal)
        print(f"          -> “趋势衰竭”风险标签诊断完成，发现 {signal.sum()} 天。{dates_str}")
        is_short_slope_down = df[required_cols['dynamic_slope_8d']] < 0
        is_mid_slope_up = df[required_cols['dynamic_slope_21d']] > 0
        states['CHIP_RISK_DIVERGENCE'] = is_short_slope_down & is_mid_slope_up & is_at_high
        signal = states['CHIP_RISK_DIVERGENCE']
        dates_str = self._format_debug_dates(signal)
        print(f"          -> “斜率背离”风险标签诊断完成，发现 {signal.sum()} 天。{dates_str}")
        p_ignite = p.get('ignition_params', {})
        is_accelerating = df[required_cols['dynamic_accel_21d']] > self._get_param_value(p_ignite.get('accel_threshold'), 0.01)
        winner_rate_col_dyn = required_cols['dynamic_winner_rate_short']
        is_winner_rate_increasing = df[winner_rate_col_dyn] > df[winner_rate_col_dyn].shift(1)
        was_in_setup_state = primary_state.shift(1).isin(['ACCUMULATION', 'TRANSITION'])
        states['CHIP_EVENT_IGNITION'] = is_accelerating & is_winner_rate_increasing & was_in_setup_state
        signal = states['CHIP_EVENT_IGNITION']
        dates_str = self._format_debug_dates(signal)
        print(f"          -> “点火事件”诊断完成，发现 {signal.sum()} 天。{dates_str}")
        for key in states:
            if states[key] is None:
                states[key] = pd.Series(False, index=df.index)
            else:
                states[key] = states[key].fillna(False)
        print("        -> [诊断模块] 筹码状态诊断执行完毕。")
        return states

    def _diagnose_ma_states(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V74.0 三维评分版】均线结构与动能状态诊断
        - 核心升级: 能够动态地为多条关键均线（如21, 34, 55, 89, 144, 233）诊断“触碰支撑”状态。
        - 这为“平台支撑回踩”剧本的三维动态评分（背景分+事件分）提供了核心数据。
        """
        print("        -> [诊断模块] 正在执行均线状态诊断...")
        states = {}
        default_series = pd.Series(False, index=df.index)
        p = self._get_params_block(params, 'ma_state_params', {})
        if not self._get_param_value(p.get('enabled'), False):
            print("          -> 均线诊断模块被禁用，跳过。")
            return states

        # --- 1. 核心趋势状态诊断 ---
        short_p = self._get_param_value(p.get('short_ma'), 13)
        mid_p = self._get_param_value(p.get('mid_ma'), 34)
        long_p = self._get_param_value(p.get('long_ma'), 89)
        short_ma, mid_ma, long_ma = f'EMA_{short_p}_D', f'EMA_{mid_p}_D', f'EMA_{long_p}_D'

        if not all(c in df.columns for c in [short_ma, mid_ma, long_ma]):
            print(f"          -> [警告] 缺少核心均线列({short_ma}, {mid_ma}, {long_ma})，部分均线状态诊断跳过。")
            return states

        states['MA_STATE_PRICE_ABOVE_LONG_MA'] = df['close_D'] > df[long_ma]
        states['MA_STATE_STABLE_BULLISH'] = (df[short_ma] > df[mid_ma]) & (df[mid_ma] > df[long_ma])
        states['MA_STATE_STABLE_BEARISH'] = (df[short_ma] < df[mid_ma]) & (df[mid_ma] < df[long_ma])
        states['MA_STATE_BOTTOM_PASSIVATION'] = states['MA_STATE_STABLE_BEARISH'] & (df['close_D'] > df[short_ma])

        # --- 2. 均线收敛/发散状态诊断 ---
        ma_spread = (df[short_ma] - df[long_ma]) / df[long_ma].replace(0, np.nan)
        ma_spread_zscore = (ma_spread - ma_spread.rolling(60).mean()) / ma_spread.rolling(60).std().replace(0, np.nan)
        states['MA_STATE_CONVERGING'] = ma_spread_zscore < self._get_param_value(p.get('converging_zscore'), -1.0)
        states['MA_STATE_DIVERGING'] = ma_spread_zscore > self._get_param_value(p.get('diverging_zscore'), 1.0)

        # --- 3. 均线加速度状态诊断 (用于识别趋势拐点) ---
        lookback_period = 10
        accel_d_col = f'ACCEL_{lookback_period}_{long_ma}'
        if accel_d_col in df.columns:
            states['MA_STATE_D_STABILIZING'] = (df[accel_d_col].shift(1).fillna(0) < 0) & (df[accel_d_col] >= 0)
        else:
            states['MA_STATE_D_STABILIZING'] = default_series
            print(f"          -> [警告] 缺少日线加速度列 '{accel_d_col}'，'MA_STATE_D_STABILIZING' 无法计算。")

        simulated_w_ma_period = 105
        simulated_w_lookback = 25
        slope_w_simulated_col = f'SLOPE_{simulated_w_lookback}_EMA_{simulated_w_ma_period}_D'
        if slope_w_simulated_col in df.columns:
            states['MA_STATE_W_STABILIZING'] = (df[slope_w_simulated_col].shift(1).fillna(0) < 0) & (df[slope_w_simulated_col] >= 0)
        else:
            states['MA_STATE_W_STABILIZING'] = default_series
            print(f"          -> [警告] 缺少周线斜率列 '{slope_w_simulated_col}'，'MA_STATE_W_STABILIZING' 无法计算。")

        # --- 4. 关键支撑均线触碰状态诊断 (为三维评分提供事件分) ---
        # ▼▼▼【代码重构 V74.0】: 动态诊断对多条关键均线的触碰 ▼▼▼
        key_support_mas = [21, 34, 55, 89, 144, 233] # 定义所有我们关心的均线级别
        
        for ma_period in key_support_mas:
            ma_col = f'EMA_{ma_period}_D'
            if ma_col in df.columns:
                is_touching = df['low_D'] <= df[ma_col]
                is_closing_above = df['close_D'] >= df[ma_col]
                state_name = f'MA_STATE_TOUCHING_SUPPORT_{ma_period}'
                states[state_name] = is_touching & is_closing_above
                
                signal = states[state_name]
                if signal.any(): # 只在有信号时打印，避免日志刷屏
                    dates_str = self._format_debug_dates(signal)
                    print(f"          -> '触碰{ma_period}日线支撑' 状态诊断完成，共激活 {signal.sum()} 天。{dates_str}")
            else:
                # 即使缺少该均线，也创建一个全False的Series，保证后续逻辑的健壮性
                states[f'MA_STATE_TOUCHING_SUPPORT_{ma_period}'] = default_series
                print(f"          -> [警告] 缺少均线列 '{ma_col}'，无法诊断其支撑状态。")
        # ▲▲▲【代码重构 V74.0】▲▲▲

        # --- 5. 老鸭头形态诊断 (一个独立的、持久化的状态) ---
        p_duck = self._get_params_block(params, 'duck_neck_params', {})
        if self._get_param_value(p_duck.get('enabled'), True):
            duck_short_p = self._get_param_value(p_duck.get('short_ma'), 5)
            duck_mid_p = self._get_param_value(p_duck.get('mid_ma'), 10)
            duck_long_p = self._get_param_value(p_duck.get('long_ma'), 60)
            duck_short_ma, duck_mid_ma, duck_long_ma = f'EMA_{duck_short_p}_D', f'EMA_{duck_mid_p}_D', f'EMA_{duck_long_p}_D'
            
            if all(c in df.columns for c in [duck_short_ma, duck_mid_ma, duck_long_ma]):
                golden_cross_event = (df[duck_short_ma] > df[duck_mid_ma]) & (df[duck_short_ma].shift(1) <= df[duck_mid_ma].shift(1))
                break_condition = (df[duck_short_ma] < df[duck_mid_ma])
                persistence_days = self._get_param_value(p_duck.get('persistence_days'), 20)
                states['MA_STATE_DUCK_NECK_FORMING'] = self._create_persistent_state(
                    df, entry_event=golden_cross_event, persistence_days=persistence_days, break_condition=break_condition
                )
                
                signal = states.get('MA_STATE_DUCK_NECK_FORMING', default_series)
                if signal.any():
                    dates_str = self._format_debug_dates(signal)
                    print(f"          -> '老鸭颈形成中' 持久化状态诊断完成，共激活 {signal.sum()} 天。{dates_str}")
            else:
                states['MA_STATE_DUCK_NECK_FORMING'] = default_series
        
        # --- 6. 最终数据清洗 ---
        for key in states:
            if states[key] is None:
                states[key] = default_series
            else:
                states[key] = states[key].fillna(False)

        return states

    def _diagnose_oscillator_states(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """【V64.5 最终融合版】震荡指标状态诊断中心"""
        print("        -> [诊断模块] 正在执行震荡指标状态诊断...")
        states = {}
        p = self._get_params_block(params, 'oscillator_state_params', {})
        if not self._get_param_value(p.get('enabled'), False): return states
        
        # --- RSI 相关状态 ---
        rsi_col = 'RSI_13_D'
        if rsi_col in df.columns:
            states['OSC_STATE_RSI_OVERBOUGHT'] = df[rsi_col] > self._get_param_value(p.get('rsi_overbought'), 80)
            states['OSC_STATE_RSI_OVERSOLD'] = df[rsi_col] < self._get_param_value(p.get('rsi_oversold'), 25)
        
        # --- MACD 相关状态 ---
        macd_h_col = 'MACDh_13_34_8_D'
        macd_z_col = 'MACD_HIST_ZSCORE_D'
        if macd_h_col in df.columns:
            states['OSC_STATE_MACD_BULLISH'] = df[macd_h_col] > 0
        if macd_z_col in df.columns:
            is_price_higher = df['close_D'] > df['close_D'].rolling(10).max().shift(1)
            is_macd_z_lower = df[macd_z_col] < df[macd_z_col].rolling(10).max().shift(1)
            states['OSC_STATE_MACD_DIVERGENCE'] = is_price_higher & is_macd_z_lower

        # ▼▼▼ BIAS机会状态的诊断 ▼▼▼
        # ▼▼▼【代码修改 V64.7】: 使用动态分位数阈值 ▼▼▼
        p_bias = self._get_params_block(params, 'playbook_specific_params', {}).get('bias_reversal_params', {})
        if self._get_param_value(p_bias.get('enabled'), True):
            bias_col = 'BIAS_55_D'
            
            if bias_col in df.columns:
                # 获取动态阈值参数
                dynamic_threshold_params = p_bias.get('dynamic_threshold', {})
                window = self._get_param_value(dynamic_threshold_params.get('window'), 120)
                quantile = self._get_param_value(dynamic_threshold_params.get('quantile'), 0.1)

                # 计算滚动的分位数阈值
                dynamic_oversold_threshold = df[bias_col].rolling(window=window).quantile(quantile)

                # 判断当前BIAS是否低于动态阈值
                states['OPP_STATE_NEGATIVE_DEVIATION'] = df[bias_col] < dynamic_oversold_threshold
                
                signal = states.get('OPP_STATE_NEGATIVE_DEVIATION', pd.Series(False, index=df.index))
                dates_str = self._format_debug_dates(signal)
                print(f"          -> '价格负向乖离' 机会状态诊断完成 (基于{bias_col}动态分位数 窗口:{window}, 分位:{quantile})，共激活 {signal.sum()} 天。{dates_str}")
            else:
                print(f"          -> [警告] 缺少列 '{bias_col}'，价格负向乖离状态无法诊断。")
        
        print("        -> [诊断模块] 震荡指标状态诊断执行完毕。")
        return states

    def _diagnose_capital_states(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """【V58.0 升级版 - 集成持久化状态】资金流状态诊断"""
        states = {}
        p = self._get_params_block(params, 'capital_state_params', {})
        if not self._get_param_value(p.get('enabled'), False): return states
        mf_slope_short_col = f'SLOPE_10_net_mf_amount_D'
        mf_slope_long_col = f'SLOPE_20_net_mf_amount_D'
        if mf_slope_short_col in df.columns and mf_slope_long_col in df.columns:
            mf_slope_short = df[mf_slope_short_col]
            mf_slope_long = df[mf_slope_long_col]
            states['CAPITAL_STATE_INFLOW_CONFIRMED'] = mf_slope_short > 0
            states['CAPITAL_EVENT_SLOPE_CROSS'] = (mf_slope_short > mf_slope_long) & (mf_slope_short.shift(1) <= mf_slope_long.shift(1))
        cmf_col = 'CMF_21_D'
        if cmf_col in df.columns:
            states['CAPITAL_STATE_CMF_BULLISH'] = df[cmf_col] > self._get_param_value(p.get('cmf_bullish_threshold'), 0.05)
        price_down = df['pct_change_D'] < 0
        capital_up = df['net_mf_amount_D'] > 0
        bottom_divergence_event = price_down & capital_up
        p_context = p.get('divergence_context', {})
        persistence_days = self._get_param_value(p_context.get('persistence_days'), 15)
        trend_ma_period = self._get_param_value(p_context.get('trend_ma_period'), 55)
        trend_ma_col = f'EMA_{trend_ma_period}_D'
        if trend_ma_col in df.columns:
            break_condition = df['close_D'] > df[trend_ma_col]
            states['CAPITAL_STATE_DIVERGENCE_WINDOW'] = self._create_persistent_state(
                df, entry_event=bottom_divergence_event, persistence_days=persistence_days, break_condition=break_condition
            )
            
            signal = states['CAPITAL_STATE_DIVERGENCE_WINDOW']
            dates_str = self._format_debug_dates(signal)
            print(f"          -> '资本底背离机会窗口' 持久化状态诊断完成，共激活 {signal.sum()} 天。{dates_str}")
            
        return states

    def _diagnose_volatility_states(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """【V58.1 升级版 - 集成持久化状态】波动率与成交量状态诊断"""
        states = {}
        p = self._get_params_block(params, 'volatility_state_params', {})
        if not self._get_param_value(p.get('enabled'), False): return states
        bbw_col = 'BBW_21_2.0_D'
        if bbw_col in df.columns:
            squeeze_threshold = df[bbw_col].rolling(60).quantile(self._get_param_value(p.get('squeeze_percentile'), 0.1))
            squeeze_event = (df[bbw_col] < squeeze_threshold) & (df[bbw_col].shift(1) >= squeeze_threshold)
            states['VOL_EVENT_SQUEEZE'] = squeeze_event
            
            signal = states['VOL_EVENT_SQUEEZE']
            dates_str = self._format_debug_dates(signal)
            print(f"          -> '波动率极度压缩' 事件诊断完成，发现 {signal.sum()} 天。{dates_str}")
            
        vol_ma_col = 'VOL_MA_21_D'
        if 'volume_D' in df.columns and vol_ma_col in df.columns:
            states['VOL_STATE_SHRINKING'] = df['volume_D'] < df[vol_ma_col] * self._get_param_value(p.get('shrinking_ratio'), 0.8)
        p_context = p.get('squeeze_context', {})
        persistence_days = self._get_param_value(p_context.get('persistence_days'), 10)
        if vol_ma_col in df.columns:
            volume_break_ratio = self._get_param_value(p_context.get('volume_break_ratio'), 1.5)
            break_condition = df['volume_D'] > df[vol_ma_col] * volume_break_ratio
            states['VOL_STATE_SQUEEZE_WINDOW'] = self._create_persistent_state(
                df, entry_event=states.get('VOL_EVENT_SQUEEZE', pd.Series(False, index=df.index)),
                persistence_days=persistence_days, break_condition=break_condition
            )
            
            signal = states['VOL_STATE_SQUEEZE_WINDOW']
            dates_str = self._format_debug_dates(signal)
            print(f"          -> '能量待爆发窗口' 持久化状态诊断完成，共激活 {signal.sum()} 天。{dates_str}")
            
        return states

    def _diagnose_box_states(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V57.0 诊断模块 - 箱体状态诊断引擎】
        """
        print("        -> [诊断模块] 正在执行箱体状态诊断...")
        states = {}
        box_params = self._get_params_block(params, 'dynamic_box_params', {})
        if not self._get_param_value(box_params.get('enabled'), False) or df.empty:
            print("          -> 箱体诊断模块被禁用或数据为空，跳过。")
            return states
        peak_distance = self._get_param_value(box_params.get('peak_distance'), 10)
        peak_prominence = self._get_param_value(box_params.get('peak_prominence'), 0.02)
        price_range = df['high_D'].max() - df['low_D'].min()
        absolute_prominence = price_range * peak_prominence if price_range > 0 else 0.01
        peak_indices, _ = find_peaks(df['close_D'], distance=peak_distance, prominence=absolute_prominence)
        trough_indices, _ = find_peaks(-df['close_D'], distance=peak_distance, prominence=absolute_prominence)
        last_peak_price = pd.Series(np.nan, index=df.index)
        if len(peak_indices) > 0:
            last_peak_price.iloc[peak_indices] = df['close_D'].iloc[peak_indices]
        last_peak_price.ffill(inplace=True)
        last_trough_price = pd.Series(np.nan, index=df.index)
        if len(trough_indices) > 0:
            last_trough_price.iloc[trough_indices] = df['close_D'].iloc[trough_indices]
        last_trough_price.ffill(inplace=True)
        box_top = last_peak_price
        box_bottom = last_trough_price
        is_valid_box = (box_top.notna()) & (box_bottom.notna()) & (box_top > box_bottom)
        was_below_top = df['close_D'].shift(1) <= box_top.shift(1)
        is_above_top = df['close_D'] > box_top
        states['BOX_EVENT_BREAKOUT'] = is_valid_box & is_above_top & was_below_top
        
        signal = states['BOX_EVENT_BREAKOUT']
        dates_str = self._format_debug_dates(signal)
        print(f"          -> '箱体向上突破' 事件诊断完成，发现 {signal.sum()} 天。{dates_str}")
        
        was_above_bottom = df['close_D'].shift(1) >= box_bottom.shift(1)
        is_below_bottom = df['close_D'] < box_bottom
        states['BOX_EVENT_BREAKDOWN'] = is_valid_box & is_below_bottom & was_above_bottom
        
        signal = states['BOX_EVENT_BREAKDOWN']
        dates_str = self._format_debug_dates(signal)
        print(f"          -> '箱体向下突破' 事件诊断完成，发现 {signal.sum()} 天。{dates_str}")
        
        ma_params = self._get_params_block(params, 'ma_state_params', {})
        mid_ma_period = self._get_param_value(ma_params.get('mid_ma'), 55)
        mid_ma_col = f"EMA_{mid_ma_period}_D"
        is_in_box = (df['close_D'] < box_top) & (df['close_D'] > box_bottom)
        if mid_ma_col in df.columns:
            box_midpoint = (box_top + box_bottom) / 2
            is_box_above_ma = box_midpoint > df[mid_ma_col]
            states['BOX_STATE_HEALTHY_CONSOLIDATION'] = is_valid_box & is_in_box & is_box_above_ma
        else:
            states['BOX_STATE_HEALTHY_CONSOLIDATION'] = is_valid_box & is_in_box
        
        signal = states['BOX_STATE_HEALTHY_CONSOLIDATION']
        dates_str = self._format_debug_dates(signal)
        print(f"          -> '健康箱体盘整' 状态诊断完成，发现 {signal.sum()} 天。{dates_str}")
        
        for key in states:
            if states[key] is None:
                states[key] = pd.Series(False, index=df.index)
            else:
                states[key] = states[key].fillna(False)
        print("        -> [诊断模块] 箱体状态诊断执行完毕。")
        return states

    def _diagnose_risk_factors(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V59.1 诊断模块 - 风险因子全景诊断版】
        """
        print("    - [风险诊断引擎 V59.1] 启动，开始诊断所有原子风险因子...")
        risks = {}
        exit_params = self._get_params_block(params, 'exit_strategy_params', {})
        if not self._get_param_value(exit_params.get('enabled'), False):
            print("      -> 出场策略被禁用，风险诊断跳过。")
            return risks
        p = exit_params.get('upthrust_distribution_params', {})
        if self._get_param_value(p.get('enabled'), True):
            overextension_ma_col = f"EMA_{self._get_param_value(p.get('overextension_ma_period'), 55)}_D"
            if all(c in df.columns for c in [overextension_ma_col, 'net_mf_amount_D']):
                is_overextended = (df['close_D'] / df[overextension_ma_col] - 1) > self._get_param_value(p.get('overextension_threshold'), 0.3)
                total_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
                upper_shadow = (df['high_D'] - np.maximum(df['open_D'], df['close_D'])) / total_range
                has_long_upper_shadow = upper_shadow > self._get_param_value(p.get('upper_shadow_ratio'), 0.6)
                is_high_volume = df['volume_D'] > df['volume_D'].rolling(30).quantile(self._get_param_value(p.get('high_volume_quantile'), 0.85))
                is_main_force_selling = df['net_mf_amount_D'] < 0
                risks['RISK_EVENT_UPTHRUST_DISTRIBUTION'] = is_overextended & has_long_upper_shadow & is_high_volume & is_main_force_selling
                
                signal = risks.get('RISK_EVENT_UPTHRUST_DISTRIBUTION', pd.Series([]))
                dates_str = self._format_debug_dates(signal)
                print(f"      -> '主力叛逃' 风险事件诊断完成，发现 {signal.sum()} 天。{dates_str}")
                
        p = exit_params.get('volume_breakdown_params', {})
        if self._get_param_value(p.get('enabled'), True):
            support_ma_col = f"EMA_{self._get_param_value(p.get('support_ma_period'), 55)}_D"
            vol_ma_col = 'VOL_MA_21_D'
            if all(c in df.columns for c in [support_ma_col, vol_ma_col, 'net_mf_amount_D', 'amount_D']):
                is_ma_broken = (df['close_D'] < df[support_ma_col]) & (df['close_D'].shift(1) >= df[support_ma_col].shift(1))
                is_volume_surge = df['volume_D'] > df[vol_ma_col] * self._get_param_value(p.get('volume_surge_ratio'), 1.8)
                avg_amount_20d = df['amount_D'].rolling(20).mean()
                is_main_force_dumping = df['net_mf_amount_D'] < -(avg_amount_20d * self._get_param_value(p.get('main_force_dump_ratio'), 0.1))
                is_conviction_candle = (df['open_D'] - df['close_D']) > (df['high_D'] - df['low_D']) * 0.6
                risks['RISK_EVENT_STRUCTURE_BREAKDOWN'] = is_ma_broken & is_volume_surge & is_main_force_dumping & is_conviction_candle
                
                signal = risks.get('RISK_EVENT_STRUCTURE_BREAKDOWN', pd.Series([]))
                dates_str = self._format_debug_dates(signal)
                print(f"      -> '结构崩溃' 风险事件诊断完成，发现 {signal.sum()} 天。{dates_str}")
                
        p_div = exit_params.get('divergence_exit_params', {})
        if self._get_param_value(p_div.get('enabled'), True):
            lookback = self._get_param_value(p_div.get('lookback_period'), 20)
            price_is_new_high = df['close_D'] == df['close_D'].rolling(lookback).max()
            rsi_not_new_high = df['RSI_13_D'] < df['RSI_13_D'].rolling(lookback).max().shift(1)
            macd_z_not_new_high = df.get('MACD_HIST_ZSCORE_D', pd.Series(0, index=df.index)) < df.get('MACD_HIST_ZSCORE_D', pd.Series(0, index=df.index)).rolling(lookback).max().shift(1)
            top_divergence_event = price_is_new_high & (rsi_not_new_high | macd_z_not_new_high)
            risks['RISK_EVENT_TOP_DIVERGENCE'] = top_divergence_event
            
            signal = risks.get('RISK_EVENT_TOP_DIVERGENCE', pd.Series([]))
            dates_str = self._format_debug_dates(signal)
            print(f"      -> '顶背离' 风险事件诊断完成，发现 {signal.sum()} 天。{dates_str}")
            
            p_context = exit_params.get('divergence_context', {})
            persistence_days = self._get_param_value(p_context.get('persistence_days'), 5)
            break_ma_period = self._get_param_value(p_context.get('break_ma_period'), 10)
            break_ma_col = f'EMA_{break_ma_period}_D'
            if break_ma_col in df.columns:
                break_condition = df['close_D'] < df[break_ma_col]
                risks['RISK_STATE_DIVERGENCE_WINDOW'] = self._create_persistent_state(
                    df, entry_event=top_divergence_event, persistence_days=persistence_days, break_condition=break_condition
                )
                
                signal = risks.get('RISK_STATE_DIVERGENCE_WINDOW', pd.Series([]))
                dates_str = self._format_debug_dates(signal)
                print(f"      -> '顶背离高危窗口' 持久化风险状态诊断完成，共激活 {signal.sum()} 天。{dates_str}")
                
        p = exit_params.get('indicator_exit_params', {})
        if self._get_param_value(p.get('enabled'), True):
            risks['RISK_STATE_RSI_OVERBOUGHT'] = df.get('RSI_13_D', 50) > self._get_param_value(p.get('rsi_threshold'), 85)
            risks['RISK_STATE_BIAS_OVERBOUGHT'] = df.get('BIAS_20_D', 0) > self._get_param_value(p.get('bias_threshold'), 20.0)
            print(f"      -> '指标超买' (RSI/BIAS) 风险状态诊断完成。")
        board_events = self._diagnose_board_patterns(df, params)
        risks['RISK_EVENT_HEAVEN_EARTH_BOARD'] = board_events.get('BOARD_EVENT_HEAVEN_EARTH', pd.Series(False, index=df.index))

        p_bias = self._get_params_block(params, 'playbook_specific_params', {}).get('bias_reversal_params', {})
        if self._get_param_value(p_bias.get('enabled'), True):
            bias_period = self._get_param_value(p_bias.get('bias_period'), 20)
            bias_col = f'BIAS_{bias_period}_D'
            if bias_col in df.columns:
                overbought_threshold = self._get_param_value(p_bias.get('overbought_threshold'), 15.0)
                risks['RISK_STATE_BIAS_OVERBOUGHT'] = df[bias_col] > overbought_threshold
        
        signal = risks.get('RISK_EVENT_HEAVEN_EARTH_BOARD', pd.Series([]))
        dates_str = self._format_debug_dates(signal)
        print(f"      -> '天地板' 极端风险事件诊断完成，发现 {signal.sum()} 天。{dates_str}")
        
        for key in risks:
            risks[key] = risks[key].fillna(False)
        print("    - [风险诊断引擎 V59.1] 所有风险因子诊断完成。")
        return risks

    def _diagnose_board_patterns(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V58.0 诊断模块 - 板形态诊断引擎】
        """
        print("        -> [诊断模块] 正在执行板形态诊断...")
        states = {}
        p = self._get_params_block(params, 'board_pattern_params', {})
        if not self._get_param_value(p.get('enabled'), False):
            return states
        prev_close = df['close_D'].shift(1)
        limit_up_threshold = self._get_param_value(p.get('limit_up_threshold'), 0.098)
        limit_down_threshold = self._get_param_value(p.get('limit_down_threshold'), -0.098)
        price_buffer = self._get_param_value(p.get('price_buffer'), 0.005)
        limit_up_price = prev_close * (1 + limit_up_threshold)
        limit_down_price = prev_close * (1 + limit_down_threshold)
        is_limit_up_close = df['close_D'] >= limit_up_price * (1 - price_buffer)
        is_limit_up_high = df['high_D'] >= limit_up_price * (1 - price_buffer)
        is_limit_down_low = df['low_D'] <= limit_down_price * (1 + price_buffer)
        states['BOARD_EVENT_EARTH_HEAVEN'] = is_limit_down_low & is_limit_up_close
        
        signal = states['BOARD_EVENT_EARTH_HEAVEN']
        dates_str = self._format_debug_dates(signal)
        print(f"          -> '地天板' 事件诊断完成，发现 {signal.sum()} 天。{dates_str}")
        
        is_limit_down_close = df['close_D'] <= limit_down_price * (1 + price_buffer)
        states['BOARD_EVENT_HEAVEN_EARTH'] = is_limit_up_high & is_limit_down_close
        
        signal = states['BOARD_EVENT_HEAVEN_EARTH']
        dates_str = self._format_debug_dates(signal)
        print(f"          -> '天地板' 事件诊断完成，发现 {signal.sum()} 天。{dates_str}")
        
        return states

    def _diagnose_kline_patterns(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V60.3 状态持久化修复版】
        """
        print("        -> [诊断模块] 正在执行K线组合形态诊断...")
        states = {}
        p = self._get_params_block(params, 'kline_pattern_params', {})
        if not self._get_param_value(p.get('enabled'), False):
            return states
        p_washout = p.get('washout_params', {})
        if self._get_param_value(p_washout.get('enabled'), True):
            lookback = self._get_param_value(p_washout.get('lookback'), 60)
            pct_thresh = self._get_param_value(p_washout.get('pct_change_threshold'), -0.05)
            vol_multiplier = self._get_param_value(p_washout.get('volume_multiplier'), 1.5)
            window_days = self._get_param_value(p_washout.get('window_days'), 15)
            is_large_decline = df['pct_change_D'] <= pct_thresh
            is_high_volume = df['volume_D'] > df['VOL_MA_21_D'] * vol_multiplier
            is_near_low = df['close_D'] <= df['low_D'].rolling(window=lookback).min() * 1.05
            washout_event = is_large_decline & is_high_volume & is_near_low
            states['KLINE_EVENT_WASHOUT'] = washout_event
            
            signal = states['KLINE_EVENT_WASHOUT']
            dates_str = self._format_debug_dates(signal)
            print(f"          -> '巨阴洗盘' 事件诊断完成，发现 {signal.sum()} 天。{dates_str}")
            
            counter = pd.Series(0, index=df.index)
            counter[washout_event] = window_days
            counter = counter.replace(0, np.nan).ffill().fillna(0)
            days_in_window = counter.groupby(washout_event.cumsum()).cumcount()
            states['KLINE_STATE_WASHOUT_WINDOW'] = (days_in_window < window_days) & (counter > 0)
            
            signal = states['KLINE_STATE_WASHOUT_WINDOW']
            dates_str = self._format_debug_dates(signal)
            print(f"          -> '巨阴反转观察窗口' 持久化状态诊断完成，共激活 {signal.sum()} 天。{dates_str}")
            
        p_gap = p.get('gap_support_params', {})
        if self._get_param_value(p_gap.get('enabled'), True):
            gap_up_event = df['low_D'] > df['high_D'].shift(1)
            gap_support_level = df['high_D'].shift(1)
            break_condition = df['low_D'] <= gap_support_level
            persistence_days = self._get_param_value(p_gap.get('persistence_days'), 10)
            states['KLINE_STATE_GAP_SUPPORT_ACTIVE'] = self._create_persistent_state(
                df, entry_event=gap_up_event, persistence_days=persistence_days, break_condition=break_condition
            )
            
            signal = states['KLINE_STATE_GAP_SUPPORT_ACTIVE']
            dates_str = self._format_debug_dates(signal)
            print(f"          -> '缺口支撑有效' 状态诊断完成，共激活 {signal.sum()} 天。{dates_str}")
            
        p_nshape = p.get('n_shape_params', {})
        if self._get_param_value(p_nshape.get('enabled'), True):
            rally_threshold = self._get_param_value(p_nshape.get('rally_threshold'), 0.097)
            consolidation_days_max = self._get_param_value(p_nshape.get('consolidation_days_max'), 3)
            n_shape_start_event = df['pct_change_D'] >= rally_threshold
            rally_open = df['open_D'].where(n_shape_start_event, np.nan)
            rally_high = df['high_D'].where(n_shape_start_event, np.nan)
            rally_open_ffill = rally_open.ffill()
            rally_high_ffill = rally_high.ffill()
            is_after_rally = n_shape_start_event.cumsum() > 0
            is_holding_above_open = df['close_D'] > rally_open_ffill
            is_not_new_rally = ~n_shape_start_event
            potential_consolidation = is_after_rally & is_holding_above_open & is_not_new_rally
            days_since_rally = potential_consolidation.groupby(n_shape_start_event.cumsum()).cumcount() + 1
            states['KLINE_STATE_N_SHAPE_CONSOLIDATION'] = potential_consolidation & (days_since_rally <= consolidation_days_max)
            states['KLINE_N_SHAPE_RALLY_HIGH'] = rally_high_ffill.where(states['KLINE_STATE_N_SHAPE_CONSOLIDATION'])
            
            signal = states['KLINE_STATE_N_SHAPE_CONSOLIDATION']
            dates_str = self._format_debug_dates(signal)
            print(f"          -> 'N字形态整理期' 状态诊断完成，共激活 {signal.sum()} 天。{dates_str}")
            
        print("        -> [诊断模块] K线组合形态诊断执行完毕。")
        return states

    def _prepare_derived_features(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        【V58.0 资金博弈深度解析版】
        """
        print("    - [衍生特征中心 V58.0 资金博弈深度解析版] 启动...")
        p = self._get_params_block(params, 'derived_feature_params', {})
        if not self._get_param_value(p.get('enabled'), True):
            print("      -> 衍生特征引擎被禁用，跳过。")
            return df
        if 'net_mf_amount_D' not in df.columns or df['net_mf_amount_D'].isnull().all():
            if all(c in df.columns for c in ['buy_lg_amount_D', 'sell_lg_amount_D', 'buy_elg_amount_D', 'sell_elg_amount_D']):
                df['net_mf_amount_D'] = (df['buy_lg_amount_D'] + df['buy_elg_amount_D']) - (df['sell_lg_amount_D'] + df['sell_elg_amount_D'])
            else:
                df['net_mf_amount_D'] = 0
        if 'net_retail_amount_D' not in df.columns or df['net_retail_amount_D'].isnull().all():
            if all(c in df.columns for c in ['buy_sm_amount_D', 'sell_sm_amount_D']):
                df['net_retail_amount_D'] = df['buy_sm_amount_D'] - df['sell_sm_amount_D']
            else:
                df['net_retail_amount_D'] = 0
        df = self._ensure_numeric_types(df)
        print("      -> 核心资金流计算与类型净化完成。")
        if 'amount_D' in df.columns:
            df['net_mf_ratio_D'] = np.divide(df['net_mf_amount_D'], df['amount_D'], out=np.zeros_like(df['net_mf_amount_D'], dtype=float), where=df['amount_D']!=0)
            print(f"      -> 已计算 'net_mf_ratio_D' (主力净流入占比)。")
        if all(c in df.columns for c in ['buy_lg_amount_D', 'sell_lg_amount_D', 'buy_elg_amount_D', 'sell_elg_amount_D']):
            main_force_buy = df['buy_lg_amount_D'] + df['buy_elg_amount_D']
            main_force_sell = df['sell_lg_amount_D'] + df['sell_elg_amount_D']
            df['mf_buy_sell_ratio_D'] = np.divide(main_force_buy, main_force_sell, out=np.ones_like(main_force_buy, dtype=float), where=main_force_sell!=0)
            print(f"      -> 已计算 'mf_buy_sell_ratio_D' (主力多空比)。")
        capital_p = p.get('capital_flow_params', {})
        accum_period = self._get_param_value(capital_p.get('accumulation_period'), 20)
        df[f'mf_accumulation_{accum_period}_D'] = df['net_mf_amount_D'].rolling(window=accum_period).sum()
        print(f"      -> 已计算 'mf_accumulation_{accum_period}_D' ({accum_period}日主力资金累积)。")
        zscore_period = self._get_param_value(p.get('momentum_params', {}).get('zscore_period'), 20)
        rolling_mean = df['net_mf_amount_D'].rolling(window=zscore_period).mean()
        rolling_std = df['net_mf_amount_D'].rolling(window=zscore_period).std()
        df['net_mf_zscore_D'] = np.divide((df['net_mf_amount_D'] - rolling_mean), rolling_std, out=np.zeros_like(df['net_mf_amount_D'], dtype=float), where=rolling_std!=0)
        print(f"      -> 已计算 'net_mf_zscore_D' ({zscore_period}日主力资金Z-Score)。")
        df['mf_retail_divergence_D'] = np.divide(df['net_mf_amount_D'], df['net_retail_amount_D'].abs(), out=np.zeros_like(df['net_mf_amount_D'], dtype=float), where=df['net_retail_amount_D']!=0)
        print(f"      -> 已计算 'mf_retail_divergence_D' (机构散户背离度)。")
        divergence_p = p.get('divergence_params', {})
        price_up = df['pct_change_D'] > 0
        price_down = df['pct_change_D'] < 0
        capital_up = df['net_mf_amount_D'] > 0
        capital_down = df['net_mf_amount_D'] < 0
        conditions = [
            (price_up & capital_up),
            (price_down & capital_down),
            (price_up & capital_down),
            (price_down & capital_up)
        ]
        choices = [1, 1, -1, -1]
        df['capital_price_score_D'] = np.select(conditions, choices, default=0)
        trend_period = self._get_param_value(divergence_p.get('trend_period'), 10)
        df['capital_price_trend_D'] = df['capital_price_score_D'].rolling(window=trend_period).sum()
        print(f"      -> 已计算 'capital_price_trend_D' ({trend_period}日价资确认趋势)。")
        print("    - [衍生特征中心 V58.0] 所有衍生特征准备完成。")
        return df


    def _calculate_risk_states(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V41.12 新增】风险状态定义中心 (Risk State Center)
        """
        print("    - [风险状态中心 V41.12] 启动，开始计算所有风险状态...")
        risks = {}
        exit_params = self._get_params_block(params, 'exit_strategy_params', {})
        try:
            risks['RISK_UPTHRUST_DISTRIBUTION'] = self._find_upthrust_distribution_exit(df, params)
            print(f"      -> '主力叛逃'风险状态定义完成，发现 {risks.get('RISK_UPTHRUST_DISTRIBUTION', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'主力叛逃'风险状态时出错: {e}")
        try:
            risks['RISK_STRUCTURE_BREAKDOWN'] = self._find_volume_breakdown_exit(df, params)
            print(f"      -> '结构崩溃'风险状态定义完成，发现 {risks.get('RISK_STRUCTURE_BREAKDOWN', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'结构崩溃'风险状态时出错: {e}")
        try:
            p = exit_params.get('divergence_exit_params', {})
            if self._get_param_value(p.get('enabled'), False):
                risks['RISK_TOP_DIVERGENCE'] = self._find_top_divergence_exit(df, params)
                print(f"      -> '顶背离'风险状态定义完成，发现 {risks.get('RISK_TOP_DIVERGENCE', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'顶背离'风险状态时出错: {e}")
        try:
            p = exit_params.get('indicator_exit_params', {})
            if self._get_param_value(p.get('enabled'), False):
                rsi_col = 'RSI_14_D'
                bias_col = 'BIAS_20_D'
                risks['RISK_RSI_OVERBOUGHT'] = df.get(rsi_col, pd.Series(False, index=df.index)) > self._get_param_value(p.get('rsi_threshold'), 85)
                risks['RISK_BIAS_OVERBOUGHT'] = df.get(bias_col, pd.Series(False, index=df.index)) > self._get_param_value(p.get('bias_threshold'), 20.0)
                print(f"      -> 'RSI超买'风险状态定义完成，发现 {risks.get('RISK_RSI_OVERBOUGHT', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'指标超买'风险状态时出错: {e}")
        all_needed_risks = [
            'RISK_UPTHRUST_DISTRIBUTION', 'RISK_STRUCTURE_BREAKDOWN',
            'RISK_TOP_DIVERGENCE', 'RISK_RSI_OVERBOUGHT', 'RISK_BIAS_OVERBOUGHT'
        ]
        for risk_name in all_needed_risks:
            if risk_name not in risks:
                risks[risk_name] = pd.Series(False, index=df.index)
        print("    - [风险状态中心 V41.13] 所有风险状态计算完成。")
        return risks

    def _create_persistent_state(
        self, 
        df: pd.DataFrame, 
        entry_event: pd.Series, 
        persistence_days: int, 
        break_condition: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        【V58.0 新增 - 状态机生成器】
        """
        if not entry_event.any():
            return pd.Series(False, index=df.index)
        event_groups = entry_event.cumsum()
        days_since_event = df.groupby(event_groups).cumcount()
        is_within_persistence = days_since_event < persistence_days
        is_active_period = event_groups > 0
        persistent_state = is_active_period & is_within_persistence
        if break_condition is not None:
            break_points = break_condition & persistent_state
            break_groups = break_points.groupby(event_groups).transform('idxmax')
            persistent_state &= (df.index < break_groups) | (break_groups.isna())
        return persistent_state
