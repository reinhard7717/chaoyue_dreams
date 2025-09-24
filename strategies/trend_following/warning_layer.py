# 文件: strategies/trend_following/warning_layer.py
# 预警层
import pandas as pd
import numpy as np
from scipy.stats import linregress
from typing import Dict, Tuple
from .utils import get_params_block

class WarningLayer:
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance
        # [代码修改] risk_metadata 现在从 signal_dictionary.json 加载
        self.risk_metadata = get_params_block(self.strategy, 'score_type_map', {})

    def run_all_warnings(self) -> Tuple[pd.Series, pd.DataFrame, pd.Series, pd.Series]:
        """
        【V2.1 · 终极信号适配版】预警层总指挥
        - 核心重构: 全面消费由认知层和各情报层产出的终极风险信号。
        """
        print("        -> [预警层分析中心 V2.1 · 终极信号适配版] 启动...")
        atomic_states = self.strategy.atomic_states
        default_series = pd.Series(0.0, index=self.strategy.df_indicators.index)
        
        # --- 1. 获取认知层计算的融合风险总分 ---
        total_risk_score = atomic_states.get('COGNITIVE_FUSED_RISK_SCORE', default_series).copy()
        
        # --- 2. 获取所有S+级风险信号，用于动态诊断 ---
        risk_prefixes = ('SCORE_CHIP_BEARISH_RESONANCE_S_PLUS', 'SCORE_CHIP_TOP_REVERSAL_S_PLUS',
                         'SCORE_BEHAVIOR_BEARISH_RESONANCE_S_PLUS', 'SCORE_BEHAVIOR_TOP_REVERSAL_S_PLUS',
                         'SCORE_FF_BEARISH_RESONANCE_S_PLUS', 'SCORE_FF_TOP_REVERSAL_S_PLUS',
                         'SCORE_DYN_BEARISH_RESONANCE_S_PLUS', 'SCORE_DYN_TOP_REVERSAL_S_PLUS',
                         'SCORE_STRUCTURE_BEARISH_RESONANCE_S_PLUS', 'SCORE_STRUCTURE_TOP_REVERSAL_S_PLUS',
                         'SCORE_FOUNDATION_BEARISH_RESONANCE_S_PLUS', 'SCORE_FOUNDATION_TOP_REVERSAL_S_PLUS',
                         'COGNITIVE_SCORE_RISK_')
        risk_details_cols = {
            key: atomic_states[key]
            for key in atomic_states
            if key.startswith(risk_prefixes) and isinstance(atomic_states[key], pd.Series)
        }
        risk_details_df = pd.DataFrame(risk_details_cols)
        
        # --- 3. 调用二次分析引擎 ---
        risk_momentum_summary = self._diagnose_risk_momentum(total_risk_score)
        risk_dynamics_summary = self._diagnose_risk_dynamics(risk_details_df)
        
        print("        -> [预警层分析中心 V2.1] 所有风险分析完成。")
        return total_risk_score, risk_details_df, risk_momentum_summary, risk_dynamics_summary

    # _diagnose_risk_momentum 和 _diagnose_risk_dynamics 方法保持不变
    def _diagnose_risk_momentum(self, total_risk_score_series: pd.Series) -> pd.Series:
        window = 3
        accel_threshold = 20.0
        def calculate_slope(y: np.ndarray) -> float:
            if np.isnan(y).any() or len(y) < window: return np.nan
            return linregress(np.arange(len(y)), y).slope
        risk_slope = total_risk_score_series.rolling(window).apply(calculate_slope, raw=True).fillna(0)
        risk_accel = risk_slope.diff().fillna(0)
        conditions = [(risk_slope > 0) & (risk_accel > accel_threshold), (risk_slope > 0) & (risk_accel < -accel_threshold), risk_slope < 0]
        choices = ["ESCALATING", "DECELERATING", "IMPROVING"]
        states = np.select(conditions, choices, default="STABLE")
        reports = [{'momentum_state': state, 'risk_slope': round(slope, 2), 'risk_accel': round(accel, 2)} if state != "STABLE" else {} for state, slope, accel in zip(states, risk_slope, risk_accel)]
        return pd.Series(reports, index=total_risk_score_series.index)

    def _diagnose_risk_dynamics(self, combined_risk_details_df: pd.DataFrame) -> pd.Series:
        if combined_risk_details_df.empty:
            return pd.Series([{} for _ in range(len(combined_risk_details_df))], index=combined_risk_details_df.index)
        risk_today_long = combined_risk_details_df.reset_index(names='trade_time').melt(id_vars='trade_time', var_name='risk_name', value_name='score')
        risk_yesterday_long = combined_risk_details_df.shift(1).reset_index(names='trade_time').melt(id_vars='trade_time', var_name='risk_name', value_name='prev_score')
        merged_risks = pd.merge(risk_today_long, risk_yesterday_long, on=['trade_time', 'risk_name']).fillna(0)
        active_risks = merged_risks[(merged_risks['score'] > 0) | (merged_risks['prev_score'] > 0)].copy()
        if active_risks.empty:
            return pd.Series([{} for _ in range(len(combined_risk_details_df))], index=combined_risk_details_df.index)
        active_risks['change'] = active_risks['score'] - active_risks['prev_score']
        active_risks['change_pct'] = (active_risks['change'] / active_risks['prev_score'].replace(0, np.nan) * 100).fillna(99999.0)
        conditions = [(active_risks['score'] > 0) & (active_risks['prev_score'] == 0), (active_risks['score'] > 0) & (active_risks['prev_score'] > 0), (active_risks['score'] == 0) & (active_risks['prev_score'] > 0)]
        choices = ['new', 'persistent', 'resolved']
        active_risks['category'] = np.select(conditions, choices, default=None)
        active_risks['cn_name'] = active_risks['risk_name'].apply(lambda x: self.risk_metadata.get(x, {}).get('cn_name', x))
        active_risks['abs_change'] = active_risks['change'].abs()
        def format_group(group):
            return group[['risk_name', 'cn_name', 'score', 'prev_score', 'change', 'change_pct']].rename(columns={'risk_name': 'name'}).round(2).to_dict('records')
        grouped = active_risks.sort_values('abs_change', ascending=False).groupby(['trade_time', 'category']).apply(format_group)
        final_summary = grouped.unstack(level='category').apply(lambda row: row.dropna().to_dict(), axis=1)
        return final_summary.reindex(combined_risk_details_df.index, fill_value={})
