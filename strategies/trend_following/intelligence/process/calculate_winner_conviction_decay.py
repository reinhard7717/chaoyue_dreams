# strategies\trend_following\intelligence\process\calculate_winner_conviction_decay.py
import json
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional, Any, Tuple

from strategies.trend_following.utils import (
    get_params_block, get_param_value, get_adaptive_mtf_normalized_score,
    is_limit_up, get_adaptive_mtf_normalized_bipolar_score,
    normalize_score, _robust_geometric_mean
)
from strategies.trend_following.intelligence.process.helper import ProcessIntelligenceHelper

class CalculateWinnerConvictionDecay:
    """
    【V4.1 · 全息动态审判版】“赢家信念衰减”专属计算引擎
    PROCESS_META_WINNER_CONVICTION_DECAY
    """
    def __init__(self, strategy_instance, helper_instance: ProcessIntelligenceHelper):
        self.strategy = strategy_instance
        self.helper = helper_instance

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V7.3.0 · 异常暴露版】计算流程调度中心
        - 修改思路：移除防御性 fillna，增加启动日志。
        - 版本号：V7.3.0
        """
        print(f"\n{'#'*30} [CalculateWinnerConvictionDecay V7.3.0] 开始计算... {'#'*30}")
        method_name = "calculate_winner_conviction_decay"
        is_debug_enabled = get_param_value(self.helper.debug_params.get('enabled'), False)
        probe_ts = None
        if is_debug_enabled and self.helper.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.helper.probe_dates]
            for date in reversed(df.index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        df_index = df.index
        params_dict, all_required_signals = self._get_decay_params_and_signals(config, method_name)
        if not self.helper._validate_required_signals(df, all_required_signals, method_name): return pd.Series(dtype=np.float32)
        _temp_debug_values = {"conviction_dynamics": {}}
        raw_signals = self._get_raw_signals(df, df_index, params_dict, method_name)
        conv_s = self._calculate_conviction_strength(df, df_index, raw_signals, params_dict, method_name, _temp_debug_values)
        res_s = self._calculate_pressure_resilience(df, df_index, raw_signals, params_dict, method_name, _temp_debug_values)
        dec_f = self._calculate_deception_filter(df, df_index, raw_signals, params_dict, method_name, _temp_debug_values)
        ctx_m = self._calculate_contextual_modulator(df, df_index, raw_signals, params_dict, method_name, _temp_debug_values, is_debug_enabled, probe_ts)
        st_b = self._calculate_stealth_accumulation_bonus(df_index, raw_signals, _temp_debug_values)
        syn_f = self._calculate_synergy_factor(conv_s, res_s, _temp_debug_values)
        final_s = self._perform_final_fusion(df_index, conv_s, res_s, dec_f, st_b, params_dict, _temp_debug_values)
        ewd_f = self._calculate_ewd_factor(conv_s, res_s, ctx_m, _temp_debug_values)
        latched_s = self._apply_latch_logic(df_index, final_s, ewd_f, params_dict, _temp_debug_values)
        if is_debug_enabled and probe_ts: self._execute_intelligence_probe(method_name, probe_ts, _temp_debug_values, latched_s)
        return latched_s.astype(np.float32)

    def _execute_intelligence_probe(self, method_name: str, probe_ts: pd.Timestamp, _temp_debug_values: Dict, final_score: pd.Series):
        """
        【V7.6.0 · 物理信号保护版】全息审计探针更新
        - 修改思路：暴露指数幂运算前的底数 RawNet，防止 10^-39 级别的信号坍塌。
        - 版本号：V7.6.0
        """
        print(f"\n{'='*38} [V7.6 PHYSICAL AUDIT: {probe_ts.strftime('%Y-%m-%d')}] {'='*38}")
        conv = _temp_debug_values.get("conviction_dynamics", {})
        latch = _temp_debug_values.get("latch_state", {})
        fus = _temp_debug_values.get("final_fusion_debug", {})
        print(f"--- [FUSION BASELINE] ---")
        print(f"  > RawIntensity: {fus.get('intensity', 0.0):.4f} | NetAfterStealth: {fus.get('raw_net', 0.0):.4f}")
        print(f"  > PowerExponent: {fus.get('exponent', 0.0):.1f}")
        print(f"--- [FINAL DECISION] ---")
        print(f"  > LatchTrigger: {latch.get('latch_trigger').loc[probe_ts]} | Count: {latch.get('rolling_count').loc[probe_ts]}")
        print(f"  > FINAL_SCORE: {final_score.loc[probe_ts]:.4e}")
        print(f"{'='*105}\n")

    def _get_decay_params_and_signals(self, config: Dict, method_name: str) -> Tuple[Dict, List[str]]:
        """
        【V7.7.0 · 稀释保护版】重构非线性压缩指数与依赖清单
        - 修改思路：下调 exponent 至 3.5 防止信号坍塌；补全所有军械库监控列。
        - 版本号：V7.7.0
        """
        decay_params = get_param_value(config.get('winner_conviction_decay_params'), {})
        fibo_periods = ["5", "13", "21", "34"]
        belief_decay_weights = {
            "mid_long_sync_decay": 0.15, "chain_collapse_resonance": 0.15, "chaotic_collapse_resonance": 0.10,
            "institutional_vacuum_meltdown": 0.10, "institutional_stalling_jerk": 0.10, "kinetic_transition_impact": 0.10,
            "macro_sector_slippage": 0.05, "parabolic_sprint_risk": 0.05, "regime_switching_risk": 0.05,
            "structural_anchor_risk": 0.04, "vpa_exhaustion_risk": 0.04, "sector_domino_risk": 0.04,
            "handover_risk": 0.04, "resonance_collapse_risk": 0.03, "golden_trap_risk": 0.03, "inst_erosion_risk": 0.03
        }
        required_df_columns = [
            'mid_long_sync_D', 'volatility_adjusted_concentration_D', 'STATE_TRENDING_STAGE_D',
            'SMART_MONEY_INST_NET_BUY_D', 'tick_large_order_net_D', 'PRICE_ENTROPY_D',
            'MA_COHERENCE_RESONANCE_D', 'PRICE_FRACTAL_DIM_D', 'STATE_MARKET_LEADER_D',
            'SMART_MONEY_SYNERGY_BUY_D', 'industry_leader_score_D', 'THEME_HOTNESS_SCORE_D',
            'industry_rank_slope_D', 'industry_rank_accel_D', 'STATE_ROUNDING_BOTTOM_D',
            'breakout_potential_D', 'STATE_GOLDEN_PIT_D', 'breakout_quality_score_D',
            'VPA_ACCELERATION_5D', 'VPA_EFFICIENCY_D', 'MA_RUBBER_BAND_EXTENSION_D',
            'STATE_PARABOLIC_WARNING_D', 'SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D',
            'SMART_MONEY_HM_NET_BUY_D', 'STATE_EMOTIONAL_EXTREME_D', 'tick_abnormal_volume_ratio_D',
            'tick_chip_transfer_efficiency_D', 'anomaly_intensity_D', 'stealth_flow_ratio_D',
            'uptrend_strength_D', 'market_sentiment_score_D', 'pressure_profit_D', 'net_amount_ratio_D'
        ]
        kinetic_targets = ['mid_long_sync_D', 'SMART_MONEY_INST_NET_BUY_D', 'PRICE_FRACTAL_DIM_D', 'volatility_adjusted_concentration_D', 'SMART_MONEY_SYNERGY_BUY_D', 'market_sentiment_score_D']
        for target in kinetic_targets:
            for p in ["5"]:
                required_df_columns.extend([f'SLOPE_{p}_{target}', f'ACCEL_{p}_{target}', f'JERK_{p}_{target}'])
        params_dict = {
            'decay_params': decay_params, 'fibo_periods': fibo_periods, 'belief_decay_weights': belief_decay_weights,
            'hab_settings': {"short": 13, "medium": 21, "long": 34},
            'latch_params': {"window": 5, "hit_count": 3, "high_score_threshold": 0.618, "core_threshold": 0.382, "momentum_protection_factor": 0.95, "entropy_threshold": 0.75},
            'vacuum_threshold': 0.1, 'final_exponent': 3.5 # [修正：3.5 次幂保证信号具备穿透力]
        }
        return params_dict, list(set(required_df_columns))

    def _get_raw_signals(self, df: pd.DataFrame, df_index: pd.Index, params_dict: Dict, method_name: str) -> Dict[str, pd.Series]:
        """
        【V7.6.0 · 鲁棒量纲版】优化动力学请求链条，补全分母保底
        - 修改思路：从全量导数请求中剔除 VPA_ACCEL 以消除警告；STD/MAD 强制 replace(0, 1e-4)。
        - 版本号：V7.6.0
        """
        raw_signals = {}
        hab_cfg = params_dict['hab_settings']
        targets = [
            'mid_long_sync_D', 'volatility_adjusted_concentration_D', 'SMART_MONEY_INST_NET_BUY_D',
            'tick_large_order_net_D', 'VPA_ACCELERATION_5D', 'VPA_EFFICIENCY_D',
            'MA_COHERENCE_RESONANCE_D', 'PRICE_FRACTAL_DIM_D', 'industry_leader_score_D',
            'THEME_HOTNESS_SCORE_D', 'industry_rank_slope_D', 'breakout_potential_D', 'SMART_MONEY_SYNERGY_BUY_D',
            'MA_RUBBER_BAND_EXTENSION_D', 'industry_breadth_score_D', 'industry_stagnation_score_D',
            'tick_abnormal_volume_ratio_D', 'breakout_quality_score_D', 'market_sentiment_score_D',
            'uptrend_strength_D', 'pressure_profit_D', 'net_amount_ratio_D',
            'SMART_MONEY_HM_NET_BUY_D', 'SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D',
            'tick_chip_transfer_efficiency_D', 'anomaly_intensity_D', 'stealth_flow_ratio_D'
        ]
        for col in targets:
            series = self.helper._get_safe_series(df, col, 0.0)
            raw_signals[col] = series
            raw_signals[f'HAB_LONG_{col}'] = series.rolling(window=hab_cfg['long']).mean().fillna(0)
            raw_signals[f'HAB_STD_{col}'] = series.rolling(window=hab_cfg['long']).std().fillna(0).replace(0, 1e-4)
            if col in ['tick_abnormal_volume_ratio_D', 'anomaly_intensity_D', 'tick_large_order_net_D']:
                mad_val = (series - series.rolling(window=hab_cfg['long']).median()).abs().rolling(window=hab_cfg['long']).median()
                raw_signals[f'HAB_MAD_{col}'] = mad_val.fillna(0).replace(0, 1e-4)
        # 全量导数链条：已剔除 VPA_ACCELERATION_5D 防止冗余阶数警告
        kinetic_targets = ['mid_long_sync_D', 'SMART_MONEY_INST_NET_BUY_D', 'PRICE_FRACTAL_DIM_D', 'volatility_adjusted_concentration_D', 'SMART_MONEY_SYNERGY_BUY_D', 'market_sentiment_score_D']
        for target in kinetic_targets:
            for d_type in ['SLOPE', 'ACCEL', 'JERK']:
                col_name = f'{d_type}_5_{target}'
                val = self.helper._get_safe_series(df, col_name, 0.0)
                raw_signals[col_name] = val
                if d_type == 'JERK':
                    j_mad = (val - val.rolling(34).median()).abs().rolling(34).median()
                    raw_signals[f'HAB_MAD_{col_name}'] = j_mad.fillna(0).replace(0, 1e-4)
        # 补全仅斜率项
        for target in ['VPA_ACCELERATION_5D', 'industry_breadth_score_D', 'industry_stagnation_score_D', 'MA_COHERENCE_RESONANCE_D', 'breakout_potential_D']:
            col_name = f'SLOPE_5_{target}'
            raw_signals[col_name] = self.helper._get_safe_series(df, col_name, 0.0)
        # 状态指标强制实数化
        for s_col in ['STATE_PARABOLIC_WARNING_D', 'STATE_MARKET_LEADER_D', 'STATE_ROUNDING_BOTTOM_D', 'STATE_GOLDEN_PIT_D', 'STATE_TRENDING_STAGE_D', 'STATE_EMOTIONAL_EXTREME_D']:
            raw_signals[s_col] = self.helper._get_safe_series(df, s_col, 0.0).fillna(0.0)
        raw_signals['hab_net_inflow'] = raw_signals['net_amount_ratio_D'].rolling(window=hab_cfg['medium']).sum().fillna(0).replace(0, 1e-4)
        raw_signals['hab_pressure_max'] = raw_signals['pressure_profit_D'].rolling(window=hab_cfg['medium']).max().fillna(0).replace(0, 1e-4)
        raw_signals['industry_rank_accel_D'] = self.helper._get_safe_series(df, 'industry_rank_accel_D', 0.0)
        return raw_signals

    def _calculate_parabolic_sprint_risk(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.3.0 · 异常暴露版】判定抛物线末端冲刺
        - 修改思路：打印 raw_extension 与 HAB 均值对比，暴露 Z-Score 为 0 的原因。
        - 版本号：V7.3.0
        """
        raw_ext = raw_signals['MA_RUBBER_BAND_EXTENSION_D']
        hab_ext = raw_signals['HAB_LONG_MA_RUBBER_BAND_EXTENSION_D']
        std_ext = raw_signals['HAB_STD_MA_RUBBER_BAND_EXTENSION_D']
        extension_z = np.tanh((raw_ext - hab_ext) / std_ext)
        sprint_risk = (raw_signals['STATE_PARABOLIC_WARNING_D'] * 0.7 + extension_z.clip(0) * 0.3).clip(0, 1)
        print(f"[NODE_PROBE] Parabolic - RawExt: {raw_ext.iloc[-1]:.4f}, HabExt: {hab_ext.iloc[-1]:.4f}, WarningState: {raw_signals['STATE_PARABOLIC_WARNING_D'].iloc[-1]}")
        return sprint_risk

    def _calculate_vpa_exhaustion_risk(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.3.0 · 异常暴露版】判定 VPA 效能衰竭
        - 修改思路：打印异常放量比率与 VPA 效率，检测两者是否因量纲不一致导致背离失效。
        - 版本号：V7.3.0
        """
        abn_vol = raw_signals['tick_abnormal_volume_ratio_D']
        vpa_eff = raw_signals['VPA_EFFICIENCY_D']
        abn_vol_z = np.tanh((abn_vol - raw_signals['HAB_LONG_tick_abnormal_volume_ratio_D']) / raw_signals['HAB_STD_tick_abnormal_volume_ratio_D'])
        vpa_eff_inv = -np.tanh(vpa_eff)
        exhaustion_risk = (abn_vol_z.clip(0) * 0.6 + vpa_eff_inv.clip(0) * 0.4).clip(0, 1)
        print(f"[NODE_PROBE] VPA_Exhaustion - AbnVol: {abn_vol.iloc[-1]:.4f}, VpaEff: {vpa_eff.iloc[-1]:.4f}, Risk: {exhaustion_risk.iloc[-1]:.4f}")
        return exhaustion_risk

    def _calculate_ewd_factor(self, conviction: pd.Series, resilience: pd.Series, context: pd.Series, _temp_debug_values: Dict) -> pd.Series:
        """
        【V4.7 · 动态锁存共振版】计算熵权衰减系数 (EWD Factor)
        - 逻辑: 衡量系统多维分量的共振度。标准差越小，熵值越低，共振度越高，锁存效力越强。
        """
        # 将各分量对齐到 [0, 1] 区间进行方差分析
        s1 = (conviction.abs() + 1) / 2
        s2 = (resilience.abs() + 1) / 2
        s3 = context # 情境调制器本就在 [0.5, 1.5] 或类似区间，此处简化
        components_df = pd.concat([s1, s2, s3], axis=1)
        # 计算行标准差
        std_series = components_df.std(axis=1).fillna(1.0)
        # 映射至 [0, 1] 的 EWD 因子，使用指数函数强化共振敏感度
        ewd_factor = np.exp(-std_series * 2.5)
        _temp_debug_values["ewd_analysis"] = {"ewd_factor": ewd_factor, "component_std": std_series}
        return ewd_factor

    def _apply_latch_logic(self, df_index: pd.Index, fused_score: pd.Series, ewd_factor: pd.Series, params_dict: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V4.7 · 动态锁存共振版】执行时域积分锁存与动能保护
        - 逻辑: 统计窗口内高分频率，结合EWD因子通过Tanh加速锁定，并提供回撤保护。
        """
        lp = params_dict['latch_params']
        # 1. 时域积分：计算窗口内处于高分区间的频次
        is_high = fused_score.abs() > lp["high_score_threshold"]
        rolling_count = is_high.rolling(window=lp["window"]).sum()
        # 2. 锁存触发条件：频次达标且处于低熵共振状态
        latch_trigger = (rolling_count >= lp["hit_count"]) & (ewd_factor > lp["entropy_threshold"])
        # 3. 执行非线性锁存加速
        latched_score = np.tanh(fused_score * 1.5) # 初步加速
        # 4. 动能保护：利用 cummax 思想在锁存激活期间维持信号
        # 如果锁存触发，且分值未跌破核心阈值，则保持前期高点的一定比例
        protected_score = fused_score.copy()
        for i in range(1, len(fused_score)):
            if latch_trigger.iloc[i]:
                # 动能锁存：取当前分值与昨日分值衰减后的较大者（仅限未跌破核心阈值时）
                if abs(fused_score.iloc[i]) > lp["core_threshold"]:
                    prev_val = protected_score.iloc[i-1]
                    curr_val = fused_score.iloc[i]
                    if np.sign(prev_val) == np.sign(curr_val):
                        protected_score.iloc[i] = curr_val if abs(curr_val) > abs(prev_val) else prev_val * lp["momentum_protection_factor"]
        final_output = protected_score.clip(-1, 1)
        _temp_debug_values["latch_state"] = {
            "rolling_count": rolling_count,
            "latch_trigger": latch_trigger,
            "protected_score": protected_score
        }
        return final_output

    def _calculate_conviction_strength(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], params_dict: Dict, method_name: str, _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.7.0 · 稀释保护版】博弈中枢：实施活跃权重动态重整逻辑
        - 修改思路：计算触发信号的权重占比，对 Conviction 进行动态补偿，防止多维度导致的 0 值稀释。
        - 版本号：V7.7.0
        """
        w = params_dict['belief_decay_weights']
        sync_decay = (-np.tanh((raw_signals['mid_long_sync_D'] - raw_signals['HAB_LONG_mid_long_sync_D'] + raw_signals['SLOPE_5_mid_long_sync_D']) / raw_signals['HAB_STD_mid_long_sync_D'])).clip(0)
        # 依次获取各子项结果（内部已含 0 值探针）
        sub_risks = {
            "mid_long_sync_decay": sync_decay,
            "parabolic_sprint_risk": self._calculate_parabolic_sprint_risk(df_index, raw_signals, _temp_debug_values),
            "vpa_exhaustion_risk": self._calculate_vpa_exhaustion_risk(df_index, raw_signals, _temp_debug_values),
            "sector_domino_risk": self._calculate_sector_spillover_domino_risk(df_index, raw_signals, _temp_debug_values),
            "regime_switching_risk": self._calculate_market_regime_switching_risk(df_index, raw_signals, _temp_debug_values),
            "handover_risk": self._calculate_smart_money_handover_risk(df_index, raw_signals, _temp_debug_values),
            "resonance_collapse_risk": self._calculate_institutional_resonance_collapse(df_index, raw_signals, _temp_debug_values),
            "institutional_stalling_jerk": self._calculate_institutional_stalling_jerk_risk(df_index, raw_signals, _temp_debug_values),
            "market_leader_impact": self._calculate_market_leader_bellwether_impact(df_index, raw_signals, _temp_debug_values),
            "structural_anchor_risk": self._calculate_long_cycle_structural_anchor(df_index, raw_signals, _temp_debug_values),
            "golden_trap_risk": self._calculate_false_golden_pit_trap(df_index, raw_signals, _temp_debug_values),
            "chaotic_collapse_resonance": self._calculate_chaotic_collapse_resonance(df_index, raw_signals, _temp_debug_values),
            "macro_sector_slippage": self._calculate_macro_sector_synergy(df_index, raw_signals, _temp_debug_values),
            "inst_erosion_risk": self._calculate_institutional_erosion_index(df_index, raw_signals, _temp_debug_values),
            "institutional_vacuum_meltdown": self._calculate_institutional_vacuum_meltdown(df_index, raw_signals, params_dict, _temp_debug_values),
            "chain_collapse_resonance": self._calculate_chain_collapse_resonance(df_index, raw_signals, _temp_debug_values),
            "kinetic_transition_impact": self._calculate_kinetic_transition_point(df_index, raw_signals, _temp_debug_values)
        }
        # 活跃权重重整：防止 silent 维度摊薄最终得分
        weighted_sum = pd.Series(0.0, index=df_index)
        active_weight_total = 0.0
        print(f"\n[CONVICTION_ACTIVE_WEIGHT_AUDIT]")
        for key, risk_series in sub_risks.items():
            current_risk = risk_series.iloc[-1]
            if current_risk > 1e-4:
                active_weight_total += w.get(key, 0.02)
                print(f"  > ACTIVE: {key} | RawRisk: {current_risk:.4f} | Weight: {w.get(key):.3f}")
            weighted_sum += risk_series * w.get(key, 0.02)
        # 如果有活跃信号，则除以活跃权重总和（带保底 0.4 防止单信号过载）
        dilution_compensation = 1.0 / max(active_weight_total, 0.4)
        fused = (weighted_sum * dilution_compensation).clip(-1, 1).fillna(0)
        _temp_debug_values["conviction_dynamics"].update({"fused_conviction": fused, "active_weight": active_weight_total})
        print(f"  >> WeightedSum: {weighted_sum.iloc[-1]:.4f} | Compensation: {dilution_compensation:.2f} | FUSED_STRENGTH: {fused.iloc[-1]:.4f}")
        return fused

    def _calculate_sector_spillover_domino_risk(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.7.0 · 零值穿透版】判定行业多米诺风险
        - 修改思路：暴露原始行业广度数据。
        - 版本号：V7.7.0
        """
        raw_breadth = raw_signals['industry_breadth_score_D']
        breadth_slope = raw_signals['SLOPE_5_industry_breadth_score_D']
        stagnation_val = raw_signals['industry_stagnation_score_D']
        stagnation_num = stagnation_val - raw_signals['HAB_LONG_industry_stagnation_score_D']
        stagnation_z = np.tanh(stagnation_num / (raw_signals['HAB_STD_industry_stagnation_score_D'] + 1e-4))
        risk = ((-np.tanh(breadth_slope)).clip(0) * 0.7 + stagnation_z.clip(0) * 0.3).clip(0, 1)
        if risk.iloc[-1] <= 1e-4:
            print(f"[ZERO_PROBE] Domino - BreadthRaw: {raw_breadth.iloc[-1]:.2f}, Slope: {breadth_slope.iloc[-1]:.4f}, StagnationRaw: {stagnation_val.iloc[-1]:.2f}")
        return risk

    def _calculate_market_regime_switching_risk(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.7.0 · 零值穿透版】判定状态切换风险
        - 修改思路：暴露原始分形维度与共振数据。
        - 版本号：V7.7.0
        """
        raw_frac = raw_signals['PRICE_FRACTAL_DIM_D']
        raw_coher = raw_signals['MA_COHERENCE_RESONANCE_D']
        coherence_slope = raw_signals['SLOPE_5_MA_COHERENCE_RESONANCE_D']
        chaos_accel = raw_signals['SLOPE_5_PRICE_FRACTAL_DIM_D']
        risk = ((-np.tanh(coherence_slope)).clip(0) * 0.5 + np.tanh(chaos_accel).clip(0) * 0.5).clip(0, 1)
        if risk.iloc[-1] <= 1e-4:
            print(f"[ZERO_PROBE] Regime - FracRaw: {raw_frac.iloc[-1]:.4f}, CoherRaw: {raw_coher.iloc[-1]:.4f}, ChaosSlope: {chaos_accel.iloc[-1]:.4f}")
        return risk

    def _calculate_smart_money_handover_risk(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.3.0 · 异常暴露版】判定筹码置换风险
        - 修改思路：打印聪明钱背离原始值。
        - 版本号：V7.3.0
        """
        div_val = raw_signals['SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D']
        handover_risk = np.tanh(div_val).clip(0, 1)
        print(f"[NODE_PROBE] Chip_Handover - Divergence_Raw: {div_val.iloc[-1]:.4f}, Risk: {handover_risk.iloc[-1]:.4f}")
        return handover_risk

    def _calculate_institutional_resonance_collapse(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.3.0 · 异常暴露版】判定机构共振瓦解
        - 修改思路：监控协同买入加速度。
        - 版本号：V7.3.0
        """
        synergy_accel = raw_signals['ACCEL_5_SMART_MONEY_SYNERGY_BUY_D']
        collapse_risk = (-np.tanh(synergy_accel)).clip(0, 1)
        print(f"[NODE_PROBE] Inst_Resonance - SynergyAccel: {synergy_accel.iloc[-1]:.4f}, Risk: {collapse_risk.iloc[-1]:.4f}")
        return collapse_risk

    def _calculate_institutional_stalling_jerk_risk(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.7.0 · 零值穿透版】判定三阶失速冲击
        - 修改思路：暴露原始机构净买入数据及其 Jerk。
        - 版本号：V7.7.0
        """
        raw_inst = raw_signals['SMART_MONEY_INST_NET_BUY_D']
        jerk_val = raw_signals['JERK_5_SMART_MONEY_INST_NET_BUY_D']
        jerk_mad = raw_signals['HAB_MAD_JERK_5_SMART_MONEY_INST_NET_BUY_D']
        risk = (-np.tanh(jerk_val / (jerk_mad * 1.4826 + 1e-4))).clip(0, 1)
        if risk.iloc[-1] <= 1e-4:
            print(f"[ZERO_PROBE] StallJerk - InstNetRaw: {raw_inst.iloc[-1]:.1f}, JerkVal: {jerk_val.iloc[-1]:.4e}, MAD: {jerk_mad.iloc[-1]:.4e}")
        return risk

    def _calculate_chaotic_collapse_resonance(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.5.0 · 鲁棒量纲版】混沌共振：消除分母为零引发的 nan
        - 修改思路：在除法前通过 1e-7 进行物理保底，并对 nan 执行强校验归零。
        - 版本号：V7.5.0
        """
        i_j = raw_signals['JERK_5_SMART_MONEY_INST_NET_BUY_D']
        i_m = raw_signals['HAB_MAD_JERK_5_SMART_MONEY_INST_NET_BUY_D'].replace(0, 1e-7)
        f_j = raw_signals['JERK_5_PRICE_FRACTAL_DIM_D']
        f_m = raw_signals['HAB_MAD_JERK_5_PRICE_FRACTAL_DIM_D'].replace(0, 1e-7)
        inst_j_z = -np.tanh(i_j / (i_m * 3.7))
        frac_j_z = np.tanh(f_j / (f_m * 3.7))
        chaos_res = (inst_j_z.clip(0) * frac_j_z.clip(0)).clip(0, 1).fillna(0)
        print(f"[NODE_AUDIT] ChaosRes - InstJerkZ: {inst_j_z.iloc[-1]:.4f}, FracJerkZ: {frac_j_z.iloc[-1]:.4f}")
        return chaos_res

    def _calculate_institutional_vacuum_meltdown(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], params_dict: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.3.0 · 异常暴露版】判定买盘真空熔断
        - 修改思路：打印支撑比率及其对应熔断阈值。
        - 版本号：V7.3.0
        """
        inst_net = raw_signals['SMART_MONEY_INST_NET_BUY_D']
        inst_hab = raw_signals['HAB_LONG_SMART_MONEY_INST_NET_BUY_D'].abs().replace(0, 1e-6)
        support_ratio = (inst_net.clip(0) / inst_hab).clip(0, 1)
        vacuum_activation = 1 / (1 + np.exp((support_ratio - params_dict['vacuum_threshold']) * 30))
        print(f"[NODE_PROBE] Buying_Vacuum - SupportRatio: {support_ratio.iloc[-1]:.4f}, Threshold: {params_dict['vacuum_threshold']}")
        return vacuum_activation.clip(0, 1)

    def _calculate_chain_collapse_resonance(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.3.0 · 异常暴露版】判定筹码连锁崩塌
        - 修改思路：打印趋势阶段权重与集中度斜率。
        - 版本号：V7.3.0
        """
        stage = raw_signals['STATE_TRENDING_STAGE_D']
        vac_slope = raw_signals['SLOPE_5_volatility_adjusted_concentration_D']
        stage_w = np.where(stage >= 4, 1.0, 0.0)
        vac_diss = -np.tanh(vac_slope)
        chain_risk = (stage_w * vac_diss.clip(0)).clip(0, 1)
        print(f"[NODE_PROBE] Chain_Collapse - Stage: {stage.iloc[-1]}, VacSlope: {vac_slope.iloc[-1]:.4f}")
        return chain_risk

    def _calculate_market_leader_bellwether_impact(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.3.0 · 异常暴露版】判定领头羊风向标风险
        - 修改思路：打印领导力分数与状态标识位。
        - 版本号：V7.3.0
        """
        leader_val = raw_signals['industry_leader_score_D']
        leader_hab = raw_signals['HAB_LONG_industry_leader_score_D']
        leader_std = raw_signals['HAB_STD_industry_leader_score_D']
        leader_z = (leader_val - leader_hab) / leader_std
        bellwether_risk = (raw_signals['STATE_MARKET_LEADER_D'] * np.tanh(leader_z.clip(0))).clip(0, 1)
        print(f"[NODE_PROBE] Leader_Impact - LeaderScore: {leader_val.iloc[-1]:.4f}, LeaderState: {raw_signals['STATE_MARKET_LEADER_D'].iloc[-1]}")
        return bellwether_risk

    def _calculate_long_cycle_structural_anchor(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.3.0 · 异常暴露版】判定结构锚定失效
        - 修改思路：监控圆弧底状态下的潜力衰减。
        - 版本号：V7.3.0
        """
        rb_state = raw_signals['STATE_ROUNDING_BOTTOM_D']
        pot_slope = raw_signals['SLOPE_5_breakout_potential_D']
        anchor_risk = (rb_state * (-np.tanh(pot_slope)).clip(0)).clip(0, 1)
        print(f"[NODE_PROBE] Structural_Anchor - RoundingBottom: {rb_state.iloc[-1]}, PotSlope: {pot_slope.iloc[-1]:.4f}")
        return anchor_risk

    def _calculate_false_golden_pit_trap(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.3.0 · 异常暴露版】判定黄金坑陷阱
        - 修改思路：显式输出质量分数异常 Z 值。
        - 版本号：V7.3.0
        """
        had_pit = raw_signals['STATE_GOLDEN_PIT_D'].rolling(21).max()
        qual_val = raw_signals['breakout_quality_score_D']
        qual_hab = raw_signals['HAB_LONG_breakout_quality_score_D']
        qual_std = raw_signals['HAB_STD_breakout_quality_score_D']
        quality_z = (qual_val - qual_hab) / qual_std
        trap_risk = (had_pit * (-np.tanh(quality_z)).clip(0)).clip(0, 1)
        print(f"[NODE_PROBE] Golden_Trap - HadPit_21D: {had_pit.iloc[-1]}, Qual_Z: {quality_z.iloc[-1]:.4f}")
        return trap_risk

    def _calculate_pressure_resilience(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], params_dict: Dict, method_name: str, _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.3.0 · 异常暴露版】重构抛压韧性计算
        - 修改思路：打印流出比率 log2 分项，识别韧性分 0 值来源。
        - 版本号：V7.3.0
        """
        outflow = raw_signals['net_amount_ratio_D'].clip(upper=0).abs()
        hab_inflow = raw_signals['hab_net_inflow'].clip(lower=1e-6)
        log_ratio = np.log2(outflow / hab_inflow + 0.5)
        flow_imp = np.tanh(log_ratio)
        curr_p = raw_signals['pressure_profit_D']
        hab_p_max = raw_signals['hab_pressure_max'].replace(0, 1e-6)
        press_imp = 2 / (1 + np.exp(-4 * (curr_p / hab_p_max - 0.5))) - 1
        resilience = (flow_imp * 0.5 + press_imp * 0.5).clip(-1, 1)
        print(f"[NODE_PROBE] Pressure_Resilience - FlowLogRatio: {log_ratio.iloc[-1]:.4f}, PressureRatio: {(curr_p/hab_p_max).iloc[-1]:.4f}")
        return resilience

    def _calculate_synergy_factor(self, conviction_strength_score: pd.Series, pressure_resilience_score: pd.Series, _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.3.0 · 异常暴露版】共振因子计算
        - 修改思路：显式打印归一化后的信念与韧性组件。
        - 版本号：V7.3.0
        """
        n_conv = (conviction_strength_score + 1) / 2
        n_res = (pressure_resilience_score + 1) / 2
        syn = (n_conv * n_res + (1 - n_conv) * (1 - n_res)).clip(0, 1)
        print(f"[NODE_PROBE] Synergy - NormConv: {n_conv.iloc[-1]:.4f}, NormRes: {n_res.iloc[-1]:.4f}")
        return syn

    def _calculate_deception_filter(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], params_dict: Dict, method_name: str, _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.5.2 · 物理量纲平滑版】诡道过滤器：引入分母平滑底座
        - 修改思路：在异常量能计算中加入 0.1 的分母平滑，防止在低基数下产生脉冲 Key。
        - 版本号：V7.5.2
        """
        ab_v = raw_signals['tick_abnormal_volume_ratio_D'].fillna(0)
        ab_h = raw_signals['HAB_LONG_tick_abnormal_volume_ratio_D'].fillna(0)
        ab_m = raw_signals['HAB_MAD_tick_abnormal_volume_ratio_D'].replace(0, 1e-4)
        # Z-Score 平滑：分母额外增加 0.1 底座
        abn_z = np.tanh((ab_v - ab_h) / (ab_m * 1.4826 + 0.1))
        tr_v = raw_signals['tick_chip_transfer_efficiency_D'].fillna(0)
        tr_h = raw_signals['HAB_LONG_tick_chip_transfer_efficiency_D'].fillna(0)
        tr_s = raw_signals['HAB_STD_tick_chip_transfer_efficiency_D'].replace(0, 1e-4)
        trans_z = np.tanh((tr_v - tr_h) / (tr_s + 1e-5))
        eff_gap = (abn_z.clip(0) - trans_z).clip(0).fillna(0)
        an_v = raw_signals['anomaly_intensity_D'].fillna(0)
        an_h = raw_signals['HAB_LONG_anomaly_intensity_D'].fillna(0)
        an_s = raw_signals['HAB_STD_anomaly_intensity_D'].replace(0, 1e-4)
        anom_int = np.tanh((an_v - an_h) / (an_s + 0.5)).fillna(0)
        penalty = (eff_gap * 0.4 + anom_int.clip(0) * 0.3).clip(0, 1).fillna(0)
        print(f"[NODE_AUDIT] Deception - SmoothAbnZ: {abn_z.iloc[-1]:.4f}, TransZ: {trans_z.iloc[-1]:.4f}, FinalFilter: {1-penalty.iloc[-1]:.4f}")
        return 1 - penalty

    def _calculate_contextual_modulator(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], params_dict: Dict, method_name: str, _temp_debug_values: Dict, is_debug_enabled: bool, probe_ts: pd.Timestamp) -> pd.Series:
        """
        【V7.3.0 · 异常暴露版】情境调制器
        - 修改思路：监控情绪 Jerk 冲击值。
        - 版本号：V7.3.0
        """
        s_acc = raw_signals.get('ACCEL_5_market_sentiment_score_D', pd.Series(0, index=df_index))
        s_jerk = raw_signals.get('JERK_5_market_sentiment_score_D', pd.Series(0, index=df_index))
        ex_raw = (s_acc > 0) * (s_jerk.clip(upper=0).abs())
        ex_mad = ex_raw.rolling(21).std().replace(0, 1e-6)
        ex_score = np.tanh(ex_raw / (ex_mad * 3.0))
        u_str = raw_signals['uptrend_strength_D']
        u_hab = raw_signals['HAB_LONG_uptrend_strength_D']
        supp_ratio = u_str / u_hab.replace(0, 1e-6)
        modulator = 0.5 + ((1 - ex_score * 0.8) * (0.5 + 0.5 * np.tanh(supp_ratio - 1))).clip(0, 1)
        print(f"[NODE_PROBE] Context_Mod - SentJerk: {s_jerk.iloc[-1]:.4f}, ExScore: {ex_score.iloc[-1]:.4f}, SuppRatio: {supp_ratio.iloc[-1]:.4f}")
        return modulator

    def _calculate_stealth_accumulation_bonus(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.5.2 · 对冲权重修正版】隐秘吸筹增益：重构合成逻辑，压制过度对冲
        - 修改思路：利用几何平均平滑 Bonus 产出，将对冲系数降级，保护信念信号。
        - 版本号：V7.5.2
        """
        s_v = raw_signals['stealth_flow_ratio_D'].fillna(0)
        s_h = raw_signals['HAB_LONG_stealth_flow_ratio_D'].fillna(0)
        s_s = raw_signals['HAB_STD_stealth_flow_ratio_D'].replace(0, 1e-4)
        # 增加 0.1 平滑项防止极小波动产生 Z=1.0
        st_z = np.tanh((s_v - s_h) / (s_s + 0.1))
        # 转移效率归一化处理
        trans_eff = np.tanh(raw_signals['tick_chip_transfer_efficiency_D'].fillna(0) / 5e6)
        # 对冲权重压缩：从 0.5 下调至 0.25
        bonus = (st_z.clip(0) * 0.25 + trans_eff.clip(0) * 0.25).clip(0, 0.5).fillna(0)
        print(f"[NODE_AUDIT] StealthBonus - Z: {st_z.iloc[-1]:.4f}, TransEff: {trans_eff.iloc[-1]:.4f}, Bonus: {bonus.iloc[-1]:.4f}")
        return bonus

    def _calculate_macro_sector_synergy(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.3.0 · 异常暴露版】判定行业位阶滑坡
        - 修改思路：监控排名导数与主题。
        - 版本号：V7.3.0
        """
        rank_s = raw_signals['industry_rank_slope_D']
        rank_a = raw_signals['industry_rank_accel_D']
        theme_val = raw_signals['THEME_HOTNESS_SCORE_D']
        theme_hab = raw_signals['HAB_LONG_THEME_HOTNESS_SCORE_D']
        theme_std = raw_signals['HAB_STD_THEME_HOTNESS_SCORE_D']
        rank_decay = -np.tanh(rank_s + rank_a)
        theme_decay = -np.tanh((theme_val - theme_hab) / theme_std)
        sector_risk = (rank_decay.clip(0) * 0.6 + theme_decay.clip(0) * 0.4).clip(0, 1)
        print(f"[NODE_PROBE] Macro_Sector - RankS+A: {(rank_s + rank_a).iloc[-1]:.4f}, ThemeDecay: {theme_decay.iloc[-1]:.4f}")
        return sector_risk

    def _calculate_institutional_erosion_index(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.3.0 · 异常暴露版】判定机构资金侵蚀
        - 修改思路：打印净买入相对 HAB 偏离度。
        - 版本号：V7.3.0
        """
        sm_val = raw_signals['SMART_MONEY_HM_NET_BUY_D']
        sm_hab = raw_signals['HAB_LONG_SMART_MONEY_HM_NET_BUY_D']
        sm_std = raw_signals['HAB_STD_SMART_MONEY_HM_NET_BUY_D']
        erosion_z = (sm_val - sm_hab) / sm_std
        erosion_index = -np.tanh(erosion_z).clip(0, 1)
        print(f"[NODE_PROBE] Inst_Erosion - NetBuy: {sm_val.iloc[-1]:.4f}, ErosionZ: {erosion_z.iloc[-1]:.4f}")
        return erosion_index

    def _calculate_kinetic_transition_point(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.4.0 · 物理量纲全暴露版】判定价量加速度动能转换点
        - 修改思路：打印 Z-Score 映射前的原始加速度与 HAB 均值对比。
        - 版本号：V7.4.0
        """
        acc_v = raw_signals['VPA_ACCELERATION_5D']
        acc_h = raw_signals['HAB_LONG_VPA_ACCELERATION_5D']
        acc_s = raw_signals['HAB_STD_VPA_ACCELERATION_5D']
        acc_z = (acc_v - acc_h) / acc_s
        slp_v = raw_signals['SLOPE_5_VPA_ACCELERATION_5D']
        slp_inv = 1 / (1 + np.exp(slp_v * 10))
        trans_score = (np.tanh(acc_z.clip(0)) * slp_inv).clip(0, 1)
        print(f"[NODE_EXPOSURE] KineTrans - AccRaw: {acc_v.iloc[-1]:.4f}, AccHab: {acc_h.iloc[-1]:.4f}, AccStd: {acc_s.iloc[-1]:.4f}")
        print(f"  > SlpRaw: {slp_v.iloc[-1]:.4f}, SlpInvFactor: {slp_inv.iloc[-1]:.4f}, FinalTrans: {trans_score.iloc[-1]:.4f}")
        return trans_score

    def _perform_final_fusion(self, df_index: pd.Index, conviction_score: pd.Series, resilience_score: pd.Series, deception_filter: pd.Series, stealth_bonus: pd.Series, params_dict: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.7.0 · 稀释保护版】重构终核融合：采用乘法抑制并调整非线性压缩
        - 修改思路：使用 3.5 次幂保证信号可见度；将对冲逻辑改为 (1 - Bonus * 0.5)。
        - 版本号：V7.7.0
        """
        exp = params_dict['final_exponent']
        intensity = (conviction_score * 0.7 + resilience_score * 0.3).clip(0, 1).fillna(0)
        # 修改点：乘法抑制代替减法。即便吸筹分很高，也只能将信念分压低，不会抹除。
        raw_net = (intensity * (2 - deception_filter.fillna(1))) * (1 - stealth_bonus.fillna(0) * 0.5)
        net_decay = raw_net.clip(-1, 1).fillna(0)
        final = np.sign(net_decay) * (net_decay.abs() ** exp)
        _temp_debug_values["final_fusion_debug"] = {"intensity": intensity.iloc[-1], "raw_net": raw_net.iloc[-1], "exponent": exp}
        print(f"\n[FINAL_FUSION_COMPONENTS]")
        print(f"  > Intensity: {intensity.iloc[-1]:.4f} | MultiplicativeNet: {raw_net.iloc[-1]:.4f}")
        print(f"  > Final_With_Exp{exp}: {final.iloc[-1]:.4e}")
        return final.clip(-1, 1).fillna(0)












