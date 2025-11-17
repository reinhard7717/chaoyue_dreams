import pandas as pd
import numpy as np
from typing import Dict, Any
from strategies.trend_following.utils import get_params_block, get_param_value, get_adaptive_mtf_normalized_bipolar_score, bipolar_to_exclusive_unipolar

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
        【V8.4 · 回踩确认二次启动版】形态分析总指挥
        - 核心升级: 废弃原子层面的“共振”和“领域健康度”信号。
        - 核心职责: 只输出形态领域的原子公理信号和形态背离信号。
        - 移除信号: SCORE_PATTERN_BULLISH_RESONANCE, SCORE_PATTERN_BEARISH_RESONANCE, BIPOLAR_PATTERN_DOMAIN_HEALTH, SCORE_PATTERN_BOTTOM_REVERSAL, SCORE_PATTERN_TOP_REVERSAL。
        - 【新增】诊断“回踩确认二次启动”形态。
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
        # 修改行: 诊断“回踩确认二次启动”形态
        axiom_pullback_confirmation = self._diagnose_axiom_pullback_confirmation(df)
        all_states['SCORE_PATTERN_PULLBACK_CONFIRMATION'] = axiom_pullback_confirmation
        return all_states

    def _diagnose_axiom_divergence(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V3.1 · 清洁与多时间维度归一化版】形态公理一：诊断“背离”
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `divergence_score` 的归一化方式改为多时间维度自适应归一化。
        """
        price_slope = self._get_safe_series(df, 'SLOPE_13_close_D', 0, method_name="_diagnose_axiom_divergence")
        momentum_slope = self._get_safe_series(df, 'SLOPE_13_intraday_vwap_div_index_D', 0, method_name="_diagnose_axiom_divergence")
        bullish_divergence_strength = ((price_slope < 0) & (momentum_slope > 0)).astype(float)
        bearish_divergence_strength = ((price_slope > 0) & (momentum_slope < 0)).astype(float)
        raw_divergence_score = bullish_divergence_strength - bearish_divergence_strength
        p_conf_pattern = get_params_block(self.strategy, 'pattern_params', {})
        tf_weights_pattern = get_param_value(p_conf_pattern.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}) # 借用筹码的MTF权重配置
        divergence_score = get_adaptive_mtf_normalized_bipolar_score(raw_divergence_score, df.index, tf_weights_pattern)
        return divergence_score.astype(np.float32)

    def _diagnose_axiom_reversal(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V3.1 · 清洁与多时间维度归一化版】形态公理二：诊断“反转”
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `reversal_score` 的归一化方式改为多时间维度自适应归一化。
        """
        raw_reversal_score = self._get_safe_series(df, 'counterparty_exhaustion_index_D', 0, method_name="_diagnose_axiom_reversal")
        p_conf_pattern = get_params_block(self.strategy, 'pattern_params', {})
        tf_weights_pattern = get_param_value(p_conf_pattern.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        reversal_score = get_adaptive_mtf_normalized_bipolar_score(raw_reversal_score, df.index, tf_weights_pattern)
        return reversal_score.astype(np.float32)

    def _diagnose_axiom_breakout(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V3.1 · 清洁与多时间维度归一化版】形态公理三：诊断“突破”
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `breakout_score` 的归一化方式改为多时间维度自适应归一化。
        """
        is_breakout_up = (self._get_safe_series(df, 'close_D', method_name="_diagnose_axiom_breakout") > self._get_safe_series(df, 'dynamic_consolidation_high_D', np.inf, method_name="_diagnose_axiom_breakout")).astype(float)
        is_breakout_down = (self._get_safe_series(df, 'close_D', method_name="_diagnose_axiom_breakout") < self._get_safe_series(df, 'dynamic_consolidation_low_D', -np.inf, method_name="_diagnose_axiom_breakout")).astype(float)
        breakout_direction = is_breakout_up - is_breakout_down
        breakout_quality = self._get_safe_series(df, 'breakout_quality_score_D', pd.Series(0, index=df.index), method_name="_diagnose_axiom_breakout")
        raw_breakout_score = breakout_direction * breakout_quality
        p_conf_pattern = get_params_block(self.strategy, 'pattern_params', {})
        tf_weights_pattern = get_param_value(p_conf_pattern.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        breakout_score = get_adaptive_mtf_normalized_bipolar_score(raw_breakout_score, df.index, tf_weights_pattern)
        return breakout_score.astype(np.float32)

    def _diagnose_axiom_pullback_confirmation(self, df: pd.DataFrame) -> pd.Series:
        """
        【V1.0】形态公理四：诊断“回踩确认二次启动”形态
        - 核心逻辑: 识别股价在量能萎缩后放量突破，随后缩量回调，再放量突破的二次启动形态。
        - 信号输出: 在形态的“二次启动日”（B日）输出1.0；否则输出0.0。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        """
        df_index = df.index
        pullback_confirmation_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        # 获取形态参数
        p_conf_pattern = get_params_block(self.strategy, 'pattern_params', {})
        n_pre_A = get_param_value(p_conf_pattern.get('pullback_confirmation_n_pre_A'), 10) # A日之前量能萎缩的天数
        n_pullback_max = get_param_value(p_conf_pattern.get('pullback_confirmation_n_pullback_max'), 13) # A日到B日之间回调的最大天数
        # 获取所需数据
        close_D = self._get_safe_series(df, 'close_D', method_name="_diagnose_axiom_pullback_confirmation")
        pct_change_D = self._get_safe_series(df, 'pct_change_D', method_name="_diagnose_axiom_pullback_confirmation")
        volume_D = self._get_safe_series(df, 'volume_D', method_name="_diagnose_axiom_pullback_confirmation")
        vol_ma5_D = self._get_safe_series(df, 'VOL_MA_5_D', method_name="_diagnose_axiom_pullback_confirmation")
        vol_ma21_D = self._get_safe_series(df, 'VOL_MA_21_D', method_name="_diagnose_axiom_pullback_confirmation")
        # 检查所有必要列是否存在且非空
        if (close_D.isnull().all() or pct_change_D.isnull().all() or volume_D.isnull().all() or
                vol_ma5_D.isnull().all() or vol_ma21_D.isnull().all()):
            print("    -> [形态情报警告] 缺少核心量价数据，无法诊断回踩确认二次启动形态。")
            return pullback_confirmation_score
        # 计算每日的成交量均线最大值
        max_vol_ma = pd.concat([vol_ma5_D, vol_ma21_D], axis=1).max(axis=1)
        # 遍历DataFrame寻找形态
        # 从足够历史数据的位置开始，确保能检查到 pre_A_days
        for i in range(n_pre_A, len(df_index)):
            day_A_idx = i
            # Condition 1: Pre-A Day Volume Atrophy (A日之前的N天内量能萎缩)
            # 检查 day_A_idx 之前的 n_pre_A 天 (不包括 day_A_idx)
            pre_A_slice_start = day_A_idx - n_pre_A
            if pre_A_slice_start < 0:
                continue # 历史数据不足
            # 确保这N天内的成交量都低于各自的max_vol_ma
            pre_A_volume_atrophy = (volume_D.iloc[pre_A_slice_start:day_A_idx] < max_vol_ma.iloc[pre_A_slice_start:day_A_idx]).all()
            if not pre_A_volume_atrophy:
                continue
            # Condition 2: Day A Initial Breakout (A日股价上涨伴随量能突破)
            day_A_pct_change = pct_change_D.iloc[day_A_idx]
            day_A_volume = volume_D.iloc[day_A_idx]
            day_A_max_vol_ma = max_vol_ma.iloc[day_A_idx]
            if not (day_A_pct_change > 0 and day_A_volume > day_A_max_vol_ma):
                continue
            # Condition 3 & 4: A to B Pullback/Consolidation and Day B Second Launch
            # 遍历 Day A 之后最多 n_pullback_max + 1 天 (Day B 最多在 Day A 之后 n_pullback_max 天)
            for j in range(day_A_idx + 1, min(day_A_idx + n_pullback_max + 2, len(df_index))):
                day_B_idx = j
                # Condition 4: Day B Second Launch (B日股价再次上涨伴随量能突破)
                day_B_pct_change = pct_change_D.iloc[day_B_idx]
                day_B_volume = volume_D.iloc[day_B_idx]
                day_B_max_vol_ma = max_vol_ma.iloc[day_B_idx]
                if day_B_pct_change > 0 and day_B_volume > day_B_max_vol_ma:
                    # Potential Day B found, now validate the pullback period (between Day A and Day B)
                    pullback_slice_start = day_A_idx + 1
                    pullback_slice_end = day_B_idx # Exclusive of Day B
                    # If pullback_slice_start >= pullback_slice_end, it means 0 days pullback, which is valid.
                    if pullback_slice_start < pullback_slice_end:
                        pullback_pct_change = pct_change_D.iloc[pullback_slice_start:pullback_slice_end]
                        pullback_volume = volume_D.iloc[pullback_slice_start:pullback_slice_end]
                        pullback_max_vol_ma = max_vol_ma.iloc[pullback_slice_start:pullback_slice_end]
                        # Check for volume shrinkage during pullback
                        # 确保回调期间的成交量都低于各自的max_vol_ma
                        volume_shrunk_during_pullback = (pullback_volume < pullback_max_vol_ma).all()
                        if not volume_shrunk_during_pullback:
                            continue
                        # Check for NO bearish volume breakouts during pullback
                        # 排除股价下跌但量能突破MA_VOL_5或MA_VOL_21（以最大值为准）的情况
                        no_bearish_volume_breakout = ~((pullback_pct_change < 0) & (pullback_volume > pullback_max_vol_ma)).any()
                        if not no_bearish_volume_breakout:
                            continue
                    # All conditions met, mark Day B with a score of 1.0
                    pullback_confirmation_score.iloc[day_B_idx] = 1.0
                    # Found a pattern for this Day A, move to next potential Day A
                    break # Break from inner loop (j) to continue with the next Day A (i)
        # Debugging output for probe date
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df.index:
                print(f"    -> [回踩确认二次启动探针] @ {probe_date_for_loop.date()}:")
                print(f"       - SCORE_PATTERN_PULLBACK_CONFIRMATION: {pullback_confirmation_score.loc[probe_date_for_loop]:.4f}")
        return pullback_confirmation_score.astype(np.float32)
















