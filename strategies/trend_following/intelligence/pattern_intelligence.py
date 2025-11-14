import pandas as pd
import numpy as np
from typing import Dict, Any
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score, normalize_to_bipolar, bipolar_to_exclusive_unipolar

class PatternIntelligence:
    """
    【V8.0 · 清洁版】形态智能引擎
    - 核心修改: 移除了所有用于调试的print探针和过程性打印，恢复静默运行模式。
    """
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def _get_safe_series(self, df: pd.DataFrame, column_name: str, default_value: Any = 0.0, method_name: str = "未知方法") -> pd.Series:
        """
        安全地从DataFrame获取Series，如果不存在则打印警告并返回默认Series。
        """
        if column_name not in df.columns:
            print(f"    -> [形态情报警告] 方法 '{method_name}' 缺少数据 '{column_name}'，使用默认值 {default_value}。")
            return pd.Series(default_value, index=df.index)
        return df[column_name]

    def run_pattern_analysis_command(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V8.3 · 纯粹原子版】形态分析总指挥
        - 核心升级: 废弃原子层面的“共振”和“领域健康度”信号。
        - 核心职责: 只输出形态领域的原子公理信号和形态背离信号。
        - 移除信号: SCORE_PATTERN_BULLISH_RESONANCE, SCORE_PATTERN_BEARISH_RESONANCE, BIPOLAR_PATTERN_DOMAIN_HEALTH, SCORE_PATTERN_BOTTOM_REVERSAL, SCORE_PATTERN_TOP_REVERSAL。
        """
        all_states = {}
        p_conf = get_params_block(self.strategy, 'pattern_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            return {}
        norm_window = get_param_value(p_conf.get('norm_window'), 60)
        # --- 步骤一: 诊断三大公理 ---
        axiom_divergence = self._diagnose_axiom_divergence(df, norm_window)
        axiom_reversal = self._diagnose_axiom_reversal(df, norm_window)
        axiom_breakout = self._diagnose_axiom_breakout(df, norm_window)
        all_states['SCORE_PATTERN_AXIOM_DIVERGENCE'] = axiom_divergence
        all_states['SCORE_PATTERN_AXIOM_REVERSAL'] = axiom_reversal
        all_states['SCORE_PATTERN_AXIOM_BREAKOUT'] = axiom_breakout
        # 将形态公理一（背离）的双极性分数分裂为看涨/看跌背离信号
        bullish_divergence, bearish_divergence = bipolar_to_exclusive_unipolar(axiom_divergence)
        all_states['SCORE_PATTERN_BULLISH_DIVERGENCE'] = bullish_divergence.astype(np.float32)
        all_states['SCORE_PATTERN_BEARISH_DIVERGENCE'] = bearish_divergence.astype(np.float32)
        return all_states

    def _diagnose_axiom_divergence(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V3.0 · 清洁版】形态公理一：诊断“背离”
        - 核心修复: 增加对所有依赖数据的存在性检查。
        """
        price_slope = self._get_safe_series(df, 'SLOPE_13_close_D', 0, method_name="_diagnose_axiom_divergence")
        momentum_slope = self._get_safe_series(df, 'SLOPE_13_intraday_vwap_div_index_D', 0, method_name="_diagnose_axiom_divergence")
        bullish_divergence_strength = ((price_slope < 0) & (momentum_slope > 0)).astype(float)
        bearish_divergence_strength = ((price_slope > 0) & (momentum_slope < 0)).astype(float)
        raw_divergence_score = bullish_divergence_strength - bearish_divergence_strength
        divergence_score = normalize_to_bipolar(raw_divergence_score, df.index, window=norm_window)
        return divergence_score.astype(np.float32)

    def _diagnose_axiom_reversal(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V3.0 · 清洁版】形态公理二：诊断“反转”
        - 核心修复: 增加对所有依赖数据的存在性检查。
        """
        raw_reversal_score = self._get_safe_series(df, 'counterparty_exhaustion_index_D', 0, method_name="_diagnose_axiom_reversal")
        reversal_score = normalize_to_bipolar(raw_reversal_score, df.index, window=norm_window)
        return reversal_score.astype(np.float32)

    def _diagnose_axiom_breakout(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V3.0 · 清洁版】形态公理三：诊断“突破”
        - 核心修复: 增加对所有依赖数据的存在性检查。
        """
        is_breakout_up = (self._get_safe_series(df, 'close_D', method_name="_diagnose_axiom_breakout") > self._get_safe_series(df, 'dynamic_consolidation_high_D', np.inf, method_name="_diagnose_axiom_breakout")).astype(float)
        is_breakout_down = (self._get_safe_series(df, 'close_D', method_name="_diagnose_axiom_breakout") < self._get_safe_series(df, 'dynamic_consolidation_low_D', -np.inf, method_name="_diagnose_axiom_breakout")).astype(float)
        breakout_direction = is_breakout_up - is_breakout_down
        breakout_quality = self._get_safe_series(df, 'breakout_quality_score_D', pd.Series(0, index=df.index), method_name="_diagnose_axiom_breakout")
        raw_breakout_score = breakout_direction * breakout_quality
        breakout_score = normalize_to_bipolar(raw_breakout_score, df.index, window=norm_window)
        return breakout_score.astype(np.float32)

