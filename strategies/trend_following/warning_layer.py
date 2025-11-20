# 文件: strategies/trend_following/warning_layer.py
# 预警层
import pandas as pd
import numpy as np
from scipy.stats import linregress
from typing import Dict, Tuple
from strategies.trend_following.utils import get_params_block

class WarningLayer:
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance
        # risk_metadata 现在从 signal_dictionary.json 加载
        self.risk_metadata = get_params_block(self.strategy, 'score_type_map', {})
    def run_all_warnings(self) -> pd.DataFrame:
        """
        【V3.0 · 配置驱动重构版】预警层总指挥
        - 核心重构: 不再返回多个零散的结果。唯一职责是根据 score_type_map 配置，
                      搜集所有 type 为 'risk' 的信号的原始分，并返回一个完整的 risk_details_df。
                      这使其成为一个纯粹的、由配置驱动的风险信号收集器。
        """
        # print("        -> [预警层分析中心 V3.0 · 配置驱动重构版] 启动...") # 更新版本号和说明
        atomic_states = self.strategy.atomic_states
        df_index = self.strategy.df_indicators.index
        risk_details_cols = {}
        # 遍历配置，而不是硬编码前缀
        for signal_name, meta in self.risk_metadata.items():
            # 只收集被明确定义为 'risk' 类型的信号
            if isinstance(meta, dict) and meta.get('type') == 'risk':
                if signal_name in atomic_states and isinstance(atomic_states[signal_name], pd.Series):
                    risk_details_cols[signal_name] = atomic_states[signal_name]
        if not risk_details_cols:
            print("        -> [预警层分析中心 V3.0] 未在配置中找到任何 'risk' 类型信号。")
            return pd.DataFrame(index=df_index)
        risk_details_df = pd.DataFrame(risk_details_cols)
        # print(f"        -> [预警层分析中心 V3.0] 已根据配置收集 {len(risk_details_df.columns)} 个风险信号。")
        # 只返回一个完整的、包含所有原始风险分的DataFrame
        return risk_details_df
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
