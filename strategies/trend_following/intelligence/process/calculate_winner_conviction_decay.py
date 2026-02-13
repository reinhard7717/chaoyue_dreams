# strategies\trend_following\intelligence\process\calculate_winner_conviction_decay.py
# 【V8.0 · 全息相位坍塌版】“赢家信念衰减”高端计算引擎 已完成pro
import json
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from numba import jit
from typing import Dict, List, Optional, Any, Tuple
from strategies.trend_following.utils import (
    get_params_block, get_param_value, get_adaptive_mtf_normalized_score,
    is_limit_up, get_adaptive_mtf_normalized_bipolar_score,
    normalize_score, _robust_geometric_mean
)
from strategies.trend_following.intelligence.process.helper import ProcessIntelligenceHelper
class CalculateWinnerConvictionDecay:
    """
    【V8.0 · 全息相位坍塌版】“赢家信念衰减”高端计算引擎
    升级重点：非线性灾变模型、筹码熵增监控、真空一票否决、活跃维度重整。
    版本号：V8.0.0
    """
    def __init__(self, strategy_instance, helper_instance: ProcessIntelligenceHelper):
        self.strategy = strategy_instance
        self.helper = helper_instance

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V21.0 · 高性能计算内核】引入 Float32 降级与 Numba 加速
        - 修改思路：
            1. 全局数据类型降级：强制转换为 float32，减少显存/内存带宽占用，提升 CPU 缓存命中率。
            2. 保持原有全息逻辑，仅在数据底层进行精度换效率的优化（对量化信号无实质影响）。
        - 版本号：V21.0.0
        """
        print(f"\n{'#'*35} [WINNER_CONVICTION_DECAY V21.0] HIGH-PERF PHASE {'#'*35}")
        method_name = "calculate_winner_conviction_decay"
        # 1. 数据类型降级优化 (Data Type Downgrade)
        # 将 float64 降级为 float32，精度足够用于趋势打分，但能显著提升矢量计算速度
        df = df.astype(np.float32, errors='ignore')
        is_debug_enabled = get_param_value(self.helper.debug_params.get('enabled'), True)
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
        _temp_debug_values = {"conviction_dynamics": {}, "cross_module_signals": {}}
        raw_signals = self._get_raw_signals(df, df_index, params_dict, method_name)
        vacuum_risk = self._calculate_institutional_vacuum_meltdown(df_index, raw_signals, params_dict, _temp_debug_values)
        _temp_debug_values["cross_module_signals"]["vacuum_risk"] = vacuum_risk
        conv_s = self._calculate_conviction_strength(df, df_index, raw_signals, params_dict, method_name, _temp_debug_values, vacuum_risk)
        res_s = self._calculate_pressure_resilience(df, df_index, raw_signals, params_dict, method_name, _temp_debug_values, vacuum_risk)
        dec_f = self._calculate_deception_filter(df, df_index, raw_signals, params_dict, method_name, _temp_debug_values)
        ctx_m = self._calculate_contextual_modulator(df, df_index, raw_signals, params_dict, method_name, _temp_debug_values, is_debug_enabled, probe_ts)
        st_b = self._calculate_stealth_accumulation_bonus(df_index, raw_signals, _temp_debug_values)
        syn_f = self._calculate_synergy_factor(conv_s, res_s, raw_signals, _temp_debug_values)
        final_s = self._perform_final_fusion(df_index, conv_s, res_s, dec_f, st_b, params_dict, _temp_debug_values)
        ewd_f = self._calculate_ewd_factor(conv_s, res_s, ctx_m, raw_signals, _temp_debug_values)
        latched_s = self._apply_latch_logic(df_index, final_s, ewd_f, params_dict, _temp_debug_values)
        if is_debug_enabled and probe_ts: self._execute_intelligence_probe(method_name, probe_ts, _temp_debug_values, latched_s)
        return latched_s.astype(np.float32)

    def _get_decay_params_and_signals(self, config: Dict, method_name: str) -> Tuple[Dict, List[str]]:
        """
        【V19.2 · 全维数据视界加载】
        - 修改思路：
            1. 修复 KeyError: 'entropy_threshold'，在 latch_params 中补充该键值。
            2. 新增 'MA_COHERENCE_RESONANCE_D' 至依赖列表，支持 V19.2 结构锚定计算。
        - 版本号：V19.2.0
        """
        decay_params = get_param_value(config.get('winner_conviction_decay_params'), {})
        fibo_periods = ["5", "13", "21", "34"]
        belief_decay_weights = {
            "mid_long_sync_decay": 0.08, "handover_erosion": 0.12, "consistency_meltdown": 0.10,
            "institutional_vacuum_meltdown": 0.15, "institutional_stalling_jerk": 0.15,
            "institutional_kinetic_crash": 0.10, "inst_erosion_risk": 0.15,
            "chaotic_collapse_resonance": 0.10, "macro_sector_synergy": 0.20,
            "chain_collapse_resonance": 0.25,
            "stealth_accumulation_bonus": 0.05, "deception_penalty": 0.02
        }
        required_df_columns = [
            'OCH_D', 'OCH_ACCELERATION_D', 'days_since_last_peak_D', 'turnover_rate_D', 'down_limit_pct_D',
            'net_amount_ratio_D', 'profit_pressure_D', 'winner_rate_D', 'uptrend_strength_D', 'close_D',
            'SMART_MONEY_INST_NET_BUY_D', 'SMART_MONEY_HM_NET_BUY_D', 'SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D',
            'SMART_MONEY_HM_COORDINATED_ATTACK_D', 'SMART_MONEY_SYNERGY_BUY_D',
            'flow_consistency_D', 'flow_zscore_D', 'flow_resistance_level_D', 'flow_impact_ratio_D',
            'flow_cluster_intensity_D', 'closing_flow_intensity_D', 'inflow_persistence_D',
            'stealth_flow_ratio_D', 'buy_lg_amount_rate_D', 'buy_sm_amount_rate_D',
            'high_freq_flow_kurtosis_D', 'high_freq_flow_divergence_D',
            'pressure_trapped_D', 'chip_entropy_D', 'chip_stability_D', 'chip_kurtosis_D',
            'chip_stability_change_5d_D', 'concentration_entropy_D', 'energy_concentration_D',
            'intraday_high_lock_ratio_D', 'high_position_lock_ratio_90_D',
            'intraday_chip_consolidation_degree_D', 'intraday_chip_game_index_D',
            'intraday_cost_center_migration_D', 'intraday_cost_center_volatility_D',
            'INTRADAY_SUPPORT_INTENT_D', 'intraday_accumulation_confidence_D',
            'intraday_distribution_confidence_D', 'main_force_activity_index_D',
            'intraday_support_test_count_D', 'cost_5pct_D',
            'intraday_trough_filling_degree_D', 'intraday_low_lock_ratio_D',
            'market_sentiment_score_D', 'THEME_HOTNESS_SCORE_D',
            'industry_leader_score_D', 'industry_rank_accel_D', 'industry_rank_slope_D',
            'industry_breadth_score_D', 'industry_downtrend_score_D', 'industry_markup_score_D',
            'industry_stagnation_score_D', 'industry_preheat_score_D',
            'mid_long_sync_D', 'daily_monthly_sync_D', 'trend_confirmation_score_D',
            'VPA_EFFICIENCY_D', 'PRICE_ENTROPY_D', 'PRICE_FRACTAL_DIM_D',
            'GEOM_ARC_CURVATURE_D', 'MA_POTENTIAL_COMPRESSION_RATE_D', 'MA_COHERENCE_RESONANCE_D', # V19.2 新增
            'TURNOVER_STABILITY_INDEX_D', 'tick_large_order_net_D',
            'tick_abnormal_volume_ratio_D', 'tick_clustering_index_D', 'tick_chip_transfer_efficiency_D',
            'price_flow_divergence_D', 'reversal_warning_score_D',
            'behavior_distribution_D', 'reversal_prob_D'
        ]
        params_dict = {
            'decay_params': decay_params,
            'fibo_periods': fibo_periods,
            'belief_decay_weights': belief_decay_weights,
            'hab_settings': {"short": 13, "medium": 21, "long": 34},
            # V19.2 修复: 补充 entropy_threshold 和 momentum_protection_factor
            'latch_params': {
                "window": 5, "hit_count": 3,
                "high_score_threshold": 0.55, "core_threshold": 0.35,
                "entropy_threshold": 0.6, "momentum_protection_factor": 0.8
            },
            'final_exponent': get_param_value(config.get('final_exponent'), 3.5),
            'kinetic_targets': [
                'intraday_high_lock_ratio_D', 'pressure_trapped_D', 'high_freq_flow_kurtosis_D',
                'industry_leader_score_D', 'down_limit_pct_D', 'turnover_rate_D', 'behavior_distribution_D',
                'reversal_prob_D', 'SMART_MONEY_INST_NET_BUY_D', 'flow_consistency_D', 'buy_lg_amount_rate_D',
                'VPA_EFFICIENCY_D', 'mid_long_sync_D', 'SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D',
                'stealth_flow_ratio_D', 'chip_entropy_D', 'MA_POTENTIAL_COMPRESSION_RATE_D',
                'tick_abnormal_volume_ratio_D', 'price_flow_divergence_D', 'buy_sm_amount_rate_D',
                'high_position_lock_ratio_90_D', 'THEME_HOTNESS_SCORE_D', 'market_sentiment_score_D',
                'main_force_activity_index_D', 'flow_resistance_level_D', 'flow_impact_ratio_D',
                'INTRADAY_SUPPORT_INTENT_D', 'intraday_cost_center_migration_D', 'intraday_cost_center_volatility_D',
                'chip_kurtosis_D', 'PRICE_ENTROPY_D', 'concentration_entropy_D', 'high_freq_flow_divergence_D',
                'GEOM_ARC_CURVATURE_D', 'flow_cluster_intensity_D', 'industry_stagnation_score_D',
                'industry_markup_score_D', 'OCH_D'
            ],
            'stat_targets': {
                'long_std': [
                    'mid_long_sync_D', 'chip_entropy_D', 'tick_clustering_index_D',
                    'intraday_chip_game_index_D', 'SMART_MONEY_INST_NET_BUY_D',
                    'high_freq_flow_kurtosis_D', 'tick_abnormal_volume_ratio_D'
                ],
                'long_only': ['pressure_trapped_D'],
                'accum_21': [
                    'behavior_distribution_D', 'SMART_MONEY_INST_NET_BUY_D', 'stealth_flow_ratio_D',
                    'market_sentiment_score_D', 'intraday_chip_consolidation_degree_D', 'PRICE_ENTROPY_D',
                    'SMART_MONEY_HM_COORDINATED_ATTACK_D', 'pressure_trapped_D',
                    'intraday_support_test_count_D',
                    'intraday_trough_filling_degree_D'
                ]
            }
        }
        return params_dict, list(set(required_df_columns))

    def _get_raw_signals(self, df: pd.DataFrame, df_index: pd.Index, params_dict: Dict, method_name: str) -> Dict[str, pd.Series]:
        """
        【V21.0 · 动力学场生成器】Float32 强制约束
        - 修改思路：
            1. 确保所有 .slope(), .rolling() 等操作产生的结果强制转换为 float32。
            2. 避免 pandas 在计算过程中自动升级为 float64。
        - 版本号：V21.0.0
        """
        raw_signals = {}
        hab_cfg = params_dict['hab_settings']
        kinetic_targets = params_dict['kinetic_targets']
        stat_targets = params_dict['stat_targets']
        print(f"\n[V21.0_KINETIC_FIELD_GENERATION]")
        all_cols = set(kinetic_targets + stat_targets['long_std'] + stat_targets['long_only'] + stat_targets['accum_21'])
        manual_additions = [
            'days_since_last_peak_D', 'winner_rate_D', 'net_amount_ratio_D', 'TURNOVER_STABILITY_INDEX_D',
            'tick_chip_transfer_efficiency_D', 'intraday_distribution_confidence_D', 'chip_stability_D',
            'uptrend_strength_D', 'industry_rank_accel_D', 'industry_rank_slope_D', 'SMART_MONEY_SYNERGY_BUY_D',
            'daily_monthly_sync_D', 'industry_downtrend_score_D', 'OCH_ACCELERATION_D', 'energy_concentration_D',
            'reversal_warning_score_D', 'close_D', 'profit_pressure_D', 'intraday_accumulation_confidence_D',
            'cost_5pct_D', 'intraday_support_test_count_D', 'chip_stability_change_5d_D',
            'intraday_trough_filling_degree_D', 'intraday_low_lock_ratio_D',
            'closing_flow_intensity_D', 'inflow_persistence_D', 'MA_COHERENCE_RESONANCE_D'
        ]
        for col in manual_additions:
            all_cols.add(col)
        for col in all_cols:
            raw_signals[col] = self.helper._get_safe_series(df, col, 0.0).astype(np.float32)
        # 2. 动力学衍生 (Kinetic Derivatives)
        period = 5
        for target in kinetic_targets:
            base_series = raw_signals[target]
            slope = ta.slope(base_series, length=period).fillna(0).astype(np.float32)
            raw_signals[f'SLOPE_{period}_{target}'] = slope
            accel = ta.slope(slope, length=period).fillna(0).astype(np.float32)
            raw_signals[f'ACCEL_{period}_{target}'] = accel
            jerk = ta.slope(accel, length=period).fillna(0).astype(np.float32)
            raw_signals[f'JERK_{period}_{target}'] = jerk
            calc_list = [
                (base_series, target),
                (slope, f'SLOPE_{period}_{target}'),
                (accel, f'ACCEL_{period}_{target}'),
                (jerk, f'JERK_{period}_{target}')
            ]
            for series, name in calc_list:
                rolling_median = series.rolling(window=hab_cfg['long']).median()
                mad = (series - rolling_median).abs().rolling(window=hab_cfg['long']).median().fillna(0).replace(0, 1e-6).astype(np.float32)
                raw_signals[f'HAB_MAD_{name}'] = mad
        # 3. 统计学衍生 (Statistical Derivatives)
        for target in stat_targets['long_std']:
            s = raw_signals[target]
            raw_signals[f'HAB_LONG_{target}'] = s.rolling(window=hab_cfg['long']).mean().fillna(0).astype(np.float32)
            raw_signals[f'HAB_STD_{target}'] = s.rolling(window=hab_cfg['long']).std().fillna(0).replace(0, 1e-4).astype(np.float32)
        for target in stat_targets['long_only']:
            s = raw_signals[target]
            raw_signals[f'HAB_LONG_{target}'] = s.rolling(window=hab_cfg['long']).mean().fillna(0).astype(np.float32)
        for target in stat_targets['accum_21']:
            s = raw_signals[target]
            raw_signals[f'HAB_ACCUM_21_{target}'] = s.rolling(window=21).sum().fillna(0).astype(np.float32)
        if 'OCH_ACCELERATION_D' not in raw_signals:
             raw_signals['OCH_ACCELERATION_D'] = pd.Series(0.0, index=df_index, dtype=np.float32)
        return raw_signals

    def _calculate_conviction_strength(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], params_dict: Dict, method_name: str, _temp_debug_values: Dict, vacuum_risk: pd.Series) -> pd.Series:
        """
        【V21.0 · 重力坍塌融合】Float32 深度优化
        - 优化内容：权重系数与物理常数全部降级为 float32，避免计算权重矩阵时发生 Upcasting。
        - 版本号：V21.0.0
        """
        w = params_dict['belief_decay_weights']
        sync_num = raw_signals['mid_long_sync_D'] - raw_signals['HAB_LONG_mid_long_sync_D'] + raw_signals['SLOPE_5_mid_long_sync_D']
        sync_decay = (-np.tanh(sync_num / raw_signals['HAB_STD_mid_long_sync_D'])).clip(0)
        handover_risk = np.tanh(raw_signals['SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D'].clip(0) * np.float32(2.0))
        peak_days = raw_signals['days_since_last_peak_D']
        potential_energy = (np.float32(1.0) - np.exp(-peak_days / np.float32(13.0))).clip(0, 1)
        sub_risks = {
            "mid_long_sync_decay": sync_decay * (np.float32(1.0) + handover_risk),
            "handover_erosion_risk": handover_risk,
            "potential_energy_exhaustion": potential_energy,
            "institutional_vacuum_meltdown": vacuum_risk,
            "institutional_stalling_jerk": self._calculate_institutional_stalling_jerk_risk(df_index, raw_signals, _temp_debug_values).astype(np.float32),
            "vpa_efficiency_collapse": (-np.tanh(raw_signals['SLOPE_5_VPA_EFFICIENCY_D'])).clip(0),
            "chaotic_collapse_resonance": self._calculate_chaotic_collapse_resonance(df_index, raw_signals, _temp_debug_values).astype(np.float32),
            "inst_erosion_risk": self._calculate_institutional_erosion_index(df_index, raw_signals, _temp_debug_values).astype(np.float32),
            "chain_collapse_resonance": self._calculate_chain_collapse_resonance(df_index, raw_signals, _temp_debug_values).astype(np.float32)
        }
        active_sum, active_weight = pd.Series(0.0, index=df_index, dtype=np.float32), np.float32(0.0)
        print(f"\n[V8.5_CONVICTION_AUDIT]")
        for k, v in sub_risks.items():
            val = v.iloc[-1]
            if val > 1e-3:
                weight = np.float32(w.get(k, 0.05))
                active_weight += weight
                active_sum += v * weight
                print(f"  - [PHYSICS] {k}: {val:.4f} (W:{weight:.2f})")
        comp = np.float32(1.0) / max(active_weight, np.float32(0.4))
        fused = (active_sum * comp).clip(0, 1.0)
        _temp_debug_values["conviction_dynamics"].update({"fused": fused, "handover": handover_risk})
        print(f"  >> FUSED_CONVICTION: {fused.iloc[-1]:.4f} | Comp: {comp:.2f}")
        return fused.astype(np.float32)

    def _calculate_pressure_resilience(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], params_dict: Dict, method_name: str, _temp_debug_values: Dict, vacuum_risk: pd.Series) -> pd.Series:
        """
        【V21.0 · 超材料力学与主动修复核心】Float32 深度优化
        - 优化内容：力学公式中的模量、系数全部转为 float32，确保 SIMD 指令高效执行。
        - 版本号：V21.0.0
        """
        # 1. Base Modulus
        test_count = raw_signals['intraday_support_test_count_D']
        intent = raw_signals['INTRADAY_SUPPORT_INTENT_D']
        intent_factor = np.tanh((intent - np.float32(40.0)) / np.float32(20.0))
        stress_log = np.log1p(test_count)
        elastic_modulus = intent_factor * stress_log
        close_p = raw_signals['close_D']
        cost_floor = raw_signals['cost_5pct_D']
        floor_gap = ((close_p - cost_floor) / (close_p + np.float32(1e-4))).clip(lower=-0.1)
        rigidity_score = np.exp(-(floor_gap ** np.float32(2.0)) * np.float32(50.0))
        stab_change = raw_signals['chip_stability_change_5d_D']
        healing_score = np.tanh(stab_change * np.float32(8.0)).clip(-1, 1)
        # 2. Hyper-Material Mechanics
        filling_degree = raw_signals['intraday_trough_filling_degree_D']
        active_repair = np.tanh(filling_degree / np.float32(40.0)).clip(0, 1.2)
        low_lock = raw_signals['intraday_low_lock_ratio_D']
        lock_bonus = np.tanh((low_lock - np.float32(30.0)) / np.float32(20.0)).clip(0, 0.8)
        base_modulus = (np.tanh(elastic_modulus) * np.float32(0.3) + rigidity_score * np.float32(0.25) + healing_score * np.float32(0.2) + active_repair * np.float32(0.25) + lock_bonus * np.float32(0.2))
        # 3. Kinetic Dynamics
        price_velocity = raw_signals['SLOPE_5_OCH_D']
        impact_ratio = raw_signals['flow_impact_ratio_D']
        shock_energy = np.float32(0.0)
        if price_velocity.iloc[-1] < 0:
            v_factor = abs(price_velocity.iloc[-1]) * np.float32(20.0)
            i_factor = np.tanh(impact_ratio.iloc[-1] / np.float32(5.0))
            shock_energy = (v_factor * (np.float32(1.0) + i_factor)).clip(0, 1.5)
        intent_slope = raw_signals['SLOPE_5_INTRADAY_SUPPORT_INTENT_D']
        fatigue_penalty = np.float32(0.0)
        if intent_slope.iloc[-1] < 0:
            fatigue_penalty = np.tanh(abs(intent_slope.iloc[-1]) * np.float32(0.5)).clip(0, 1)
        kinetic_factor = np.float32(1.0) - (shock_energy * np.float32(0.5) + fatigue_penalty * np.float32(0.5))
        # 4. HAB Memory
        accum_tests = raw_signals['HAB_ACCUM_21_intraday_support_test_count_D']
        accum_fill = raw_signals['HAB_ACCUM_21_intraday_trough_filling_degree_D']
        hardening_raw = (accum_tests / np.float32(50.0)) + (accum_fill / np.float32(1000.0))
        hardening_bonus = np.tanh(hardening_raw).clip(0, 0.6)
        accum_trapped = raw_signals['HAB_ACCUM_21_pressure_trapped_D']
        corrosion_penalty = np.tanh(accum_trapped / np.float32(2000.0)).clip(0, 0.8)
        hab_factor = np.float32(1.0) + hardening_bonus - corrosion_penalty
        # 5. Final Resilience
        raw_resilience = base_modulus * kinetic_factor * hab_factor
        brittle_fracture = np.where(vacuum_risk > 0.6, vacuum_risk * np.float32(1.5), np.float32(0.0))
        final_resilience = (raw_resilience - brittle_fracture).clip(-1, 1)
        print(f"\n[V8.8.1_HYPER_MATERIAL_MECHANICS_PROBE]")
        print(f"  > [BASE] Modulus: {base_modulus.iloc[-1]:.4f} (Repair:{active_repair.iloc[-1]:.2f}, Lock:{lock_bonus.iloc[-1]:.2f})")
        print(f"  > [KINETIC] ShockEnergy: {shock_energy:.4f} | Fatigue: {fatigue_penalty:.4f}")
        print(f"  > [MEMORY] Hardening: +{hardening_bonus.iloc[-1]:.4f} (AccumFill:{accum_fill.iloc[-1]:.0f}) | Corrosion: -{corrosion_penalty.iloc[-1]:.4f}")
        print(f"  > FINAL_RESILIENCE: {final_resilience.iloc[-1]:.4f}")
        _temp_debug_values["resilience_analysis"] = {"base": base_modulus, "repair": active_repair, "hardening": hardening_bonus, "fracture": brittle_fracture}
        return final_resilience.astype(np.float32)

    def _calculate_synergy_factor(self, conviction: pd.Series, resilience: pd.Series, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V21.0 · 全息超导协同网络】Float32 深度优化
        - 优化内容：
            1. 强制 ta.slope 动态计算返回 float32。
            2. 所有协同系数计算使用 float32 标量。
        - 版本号：V21.0.0
        """
        # 1. Theoretical Field
        c_vec = np.tanh(conviction * np.float32(1.2))
        r_vec = np.tanh(resilience * np.float32(1.2))
        base_interference = c_vec * r_vec * np.float32(2.0)
        flow_cons = raw_signals.get('flow_consistency_D', pd.Series(0.5, index=conviction.index, dtype=np.float32))
        chip_stab = raw_signals.get('chip_stability_D', pd.Series(0.5, index=conviction.index, dtype=np.float32))
        transmission_coef = np.float32(0.5) + (np.tanh(flow_cons) * np.float32(0.4) + np.tanh(chip_stab) * np.float32(0.3))
        energy_conc = raw_signals.get('energy_concentration_D', pd.Series(50.0, index=conviction.index, dtype=np.float32))
        excitation_multiplier = np.float32(1.0) + np.tanh((energy_conc - np.float32(40.0)) / np.float32(30.0)).clip(0) * np.float32(0.5)
        vacuum_risk = _temp_debug_values.get("cross_module_signals", {}).get("vacuum_risk", pd.Series(0.0, index=conviction.index, dtype=np.float32))
        damping_factor = np.float32(1.0) - (vacuum_risk ** np.float32(2.0))
        raw_synergy = base_interference * transmission_coef * excitation_multiplier * damping_factor
        # Kinetic Correction (Force Float32)
        syn_slope = ta.slope(raw_synergy, length=5).fillna(0).astype(np.float32)
        syn_accel = ta.slope(syn_slope, length=5).fillna(0).astype(np.float32)
        syn_jerk = ta.slope(syn_accel, length=5).fillna(0).astype(np.float32)
        locking_bonus = np.where((raw_synergy > 0) & (syn_accel > 0), np.tanh(syn_accel * np.float32(5.0)) * np.float32(0.3), np.float32(0.0))
        decoherence_penalty = np.where(syn_jerk < -0.05, np.tanh(abs(syn_jerk) * np.float32(10.0)) * np.float32(0.4), np.float32(0.0))
        synergy_accum = raw_synergy.rolling(window=13).sum().fillna(0)
        density_coef = np.float32(0.8) + np.tanh(synergy_accum / np.float32(6.0)).clip(0) * np.float32(0.4)
        theoretical_synergy = (raw_synergy + locking_bonus - decoherence_penalty) * density_coef
        # 2. Holographic Calibration
        vpa_eff = raw_signals.get('VPA_EFFICIENCY_D', pd.Series(0.5, index=conviction.index, dtype=np.float32))
        efficiency_gate = np.tanh(vpa_eff * np.float32(2.0)).clip(0.3, 1.0)
        coord_attack = raw_signals.get('SMART_MONEY_HM_COORDINATED_ATTACK_D', pd.Series(0.0, index=conviction.index, dtype=np.float32))
        syn_buy = raw_signals.get('SMART_MONEY_SYNERGY_BUY_D', pd.Series(0.0, index=conviction.index, dtype=np.float32))
        attack_boost = np.float32(0.0)
        if theoretical_synergy.iloc[-1] > 0:
            is_attacking = (coord_attack > 0) | (syn_buy > 0.5)
            attack_boost = np.where(is_attacking, np.float32(0.25), np.float32(0.0))
        ma_coherence = raw_signals.get('MA_COHERENCE_RESONANCE_D', pd.Series(50.0, index=conviction.index, dtype=np.float32))
        structure_anchor = np.tanh(ma_coherence / np.float32(40.0)).clip(0.6, 1.1)
        # 3. Final Fusion
        final_synergy_raw = (theoretical_synergy + attack_boost) * efficiency_gate * structure_anchor
        final_synergy = np.tanh(final_synergy_raw).clip(-1, 1).fillna(0)
        print(f"\n[V19.2_HOLOGRAPHIC_SUPERCONDUCTING_PROBE]")
        print(f"  > [THEORY] Kinetic: {theoretical_synergy.iloc[-1]:.4f} (Accum13: {synergy_accum.iloc[-1]:.2f})")
        print(f"  > [GATE] VPA_Eff: {vpa_eff.iloc[-1]:.2f} -> Gate: {efficiency_gate.iloc[-1]:.2f} (Thermodynamic Loss Check)")
        print(f"  > [BOOST] CoordAttack: {coord_attack.iloc[-1]:.1f} | SynBuy: {syn_buy.iloc[-1]:.2f} -> Boost: +{pd.Series(attack_boost).iloc[-1]:.2f}")
        print(f"  > [ANCHOR] MA_Coherence: {ma_coherence.iloc[-1]:.1f} -> Anchor: {structure_anchor.iloc[-1]:.2f}")
        print(f"  > FINAL_SYNERGY_FACTOR: {final_synergy.iloc[-1]:.4f}")
        _temp_debug_values["cross_module_signals"]["synergy_factor"] = final_synergy
        return final_synergy.astype(np.float32)

    def _perform_final_fusion(self, df_index: pd.Index, conviction_score: pd.Series, resilience_score: pd.Series, deception_filter: pd.Series, stealth_bonus: pd.Series, params_dict: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V21.0 · 几何共振融合版】Float32 深度优化
        - 优化内容：指数运算幂次降级，抑制非线性函数计算中的自动提升。
        - 版本号：V21.0.0
        """
        exp = np.float32(params_dict['final_exponent'])
        res_collapse = resilience_score.clip(upper=0).abs()
        res_resist = resilience_score.clip(lower=0)
        raw_intensity = conviction_score * (np.float32(1.0) + (res_collapse ** np.float32(2.0)) * np.float32(1.8)) - res_resist * np.float32(0.3)
        intensity = raw_intensity.clip(0, 1.2)
        st_b_constrained = stealth_bonus.fillna(0).clip(0, 0.4)
        raw_net = (intensity * (np.float32(2.0) - deception_filter.fillna(1))) * (np.float32(1.0) - st_b_constrained * np.float32(0.4))
        net_decay = raw_net.clip(-1, 1).fillna(0)
        final = np.sign(net_decay) * (net_decay.abs() ** exp)
        _temp_debug_values["final_fusion_debug"] = {"intensity": intensity.iloc[-1], "raw_net": raw_net.iloc[-1], "exponent": exp}
        print(f"\n[V8.0_FINAL_FUSION_COMPONENTS]")
        print(f"  - Conviction: {conviction_score.iloc[-1]:.4f} | ResCollapseBoost: {(res_collapse**2*1.8).iloc[-1]:.4f}")
        print(f"  - Intensity: {intensity.iloc[-1]:.4f} | StealthHedging: {(st_b_constrained*0.4).iloc[-1]:.4f}")
        print(f"  - FinalScore: {final.iloc[-1]:.4e}")
        return final.clip(-1, 1).fillna(0).astype(np.float32)

    @staticmethod
    def _numba_latch_core(fused_values: np.ndarray, trigger_values: np.ndarray, emergency_values: np.ndarray, core_thresh: float, mom_factor: float) -> np.ndarray:
        """
        【V21.0 · Numba JIT 内核】高性能锁存迭代器
        - 功能：将 Python 的显式循环编译为机器码，解决状态依赖计算的性能瓶颈。
        - 装饰器：在实际环境中应添加 @numba.jit(nopython=True) 
        - 注意：本方法不包含 self 参数，为纯函数。
        """
        n = len(fused_values)
        protected_values = fused_values.copy()
        latched_values = np.tanh(fused_values * 1.618) # 预计算饱和值
        # 显式循环，Numba 将对其进行 Loop Unrolling 和 SIMD 优化
        for i in range(1, n):
            if trigger_values[i]:
                if emergency_values[i]:
                    protected_values[i] = latched_values[i]
                elif abs(fused_values[i]) > core_thresh:
                    curr = fused_values[i]
                    prev = protected_values[i-1]
                    # 动量保护：同向且减速时，启动惯性保护
                    if (np.sign(prev) == np.sign(curr)) and (abs(curr) < abs(prev)):
                        protected_values[i] = prev * mom_factor
                    else:
                        protected_values[i] = curr
            # else: 保持 copy 时的原始值
        return protected_values

    def _apply_latch_logic(self, df_index: pd.Index, fused_score: pd.Series, ewd_factor: pd.Series, params_dict: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V21.0 · 时空动量锁存系统】Numba 加速版
        - 修改思路：
            1. 尝试导入 numba 进行 JIT 编译加速。
            2. 将 Pandas Series 转换为 Numpy Array 传入静态内核。
            3. 回落机制：若无 Numba 环境，则使用纯 Numpy 实现（但仍保留静态方法结构）。
        - 版本号：V21.0.0
        """
        lp = params_dict['latch_params']
        is_high = fused_score.abs() > lp["high_score_threshold"]
        rolling_count = is_high.rolling(window=lp["window"]).sum().fillna(0)
        vacuum_risk = _temp_debug_values.get("cross_module_signals", {}).get("vacuum_risk", pd.Series(0.0, index=df_index))
        is_emergency = (vacuum_risk > 0.8) | (fused_score.abs() > 0.85)
        entropy_gate = ewd_factor > lp.get("entropy_threshold", 0.6)
        latch_trigger = ((rolling_count >= lp["hit_count"]) & entropy_gate) | is_emergency
        # 准备 Numpy 数组 (Float32)
        fused_values = fused_score.values.astype(np.float32)
        trigger_values = latch_trigger.values
        emergency_values = is_emergency.values
        core_thresh = float(lp["core_threshold"])
        mom_factor = float(lp["momentum_protection_factor"])
        # 尝试调用 Numba 编译
        try:
            # 定义 JIT 版本的内核 (即时编译)
            jit_func = jit(nopython=True)(self._numba_latch_core)
            protected_values = jit_func(fused_values, trigger_values, emergency_values, core_thresh, mom_factor)
        except ImportError:
            # 降级：直接调用静态方法 (纯 Python/Numpy)
            protected_values = self._numba_latch_core(fused_values, trigger_values, emergency_values, core_thresh, mom_factor)
        final_output = pd.Series(protected_values, index=df_index).clip(-1, 1)
        print(f"\n[V21.0_SPATIOTEMPORAL_LATCH_PROBE]")
        print(f"  > [INPUT] FusedScore: {fused_score.iloc[-1]:.4f} | EWD: {ewd_factor.iloc[-1]:.4f}")
        print(f"  > [GATE] RollingHits: {rolling_count.iloc[-1]:.0f}/{lp['window']} | EntropyGate: {entropy_gate.iloc[-1]}")
        print(f"  > [STATE] Trigger: {latch_trigger.iloc[-1]} | Emergency: {is_emergency.iloc[-1]}")
        print(f"  > FINAL_LATCHED_SCORE: {final_output.iloc[-1]:.4f}")
        _temp_debug_values["latch_state"] = {"count": rolling_count, "trigger": latch_trigger, "emergency": is_emergency}
        return final_output

    def _calculate_institutional_vacuum_meltdown(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], params_dict: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V21.0 · 盾构防御熔断核心】Float32 深度优化
        - 优化内容：将所有标量常数强制转换为 np.float32，防止运算过程中自动升级为 float64。
        - 版本号：V21.0.0
        """
        # A. 攻击侧
        inst_jerk = raw_signals['JERK_5_SMART_MONEY_INST_NET_BUY_D']
        inst_jerk_mad = raw_signals['HAB_MAD_JERK_5_SMART_MONEY_INST_NET_BUY_D']
        f_1_618 = np.float32(1.618)
        f_1e_6 = np.float32(1e-6)
        shock_norm = (-inst_jerk / (inst_jerk_mad * f_1_618 + f_1e_6)).clip(0)
        cons_slope = raw_signals['SLOPE_5_flow_consistency_D']
        cons_mad = raw_signals['HAB_MAD_SLOPE_5_flow_consistency_D']
        consistency_risk = (-np.tanh(cons_slope / (cons_mad + f_1e_6))).clip(0, 1)
        attack_force = (np.tanh(shock_norm) * np.float32(0.6) + consistency_risk * np.float32(0.4)).clip(0, 1)
        # B. 防御侧
        support_intent = raw_signals['INTRADAY_SUPPORT_INTENT_D']
        intent_factor = (support_intent / np.float32(80.0)).clip(0.1, 1.2)
        buy_rate_slope = raw_signals['SLOPE_5_buy_lg_amount_rate_D']
        withdrawal_factor = (np.float32(1.0) + np.tanh(buy_rate_slope * np.float32(5.0))).clip(0.2, 1.5)
        closing_flow = raw_signals['closing_flow_intensity_D']
        closing_penalty = np.float32(1.0)
        if closing_flow.iloc[-1] < 0:
            closing_penalty = np.float32(1.0) - np.tanh(abs(closing_flow.iloc[-1]) / np.float32(1e7))
        defense_power = (intent_factor * withdrawal_factor * closing_penalty).clip(0.1, 2.0)
        # C. 存量缓冲
        accum_21 = raw_signals['HAB_ACCUM_21_SMART_MONEY_INST_NET_BUY_D']
        std_accum_ref = raw_signals['HAB_STD_SMART_MONEY_INST_NET_BUY_D'] * np.sqrt(np.float32(21.0))
        buffer_strength = (np.float32(1.0) / (np.float32(1.0) + np.exp(-accum_21 / (std_accum_ref + f_1e_6) * np.float32(2.0)))).clip(0.1, 1.0)
        # D. 消耗率
        inst_net = raw_signals['SMART_MONEY_INST_NET_BUY_D']
        is_outflow = inst_net < 0
        depletion_impact = np.float32(0.0)
        if is_outflow.iloc[-1]:
             if accum_21.iloc[-1] > 0:
                 depletion_impact = (abs(inst_net.iloc[-1]) / accum_21.iloc[-1]).clip(0, 1)
             else:
                 depletion_impact = np.float32(1.0)
        # E. 综合熔断
        numerator = attack_force * (np.float32(1.0) + depletion_impact * np.float32(1.5))
        denominator = defense_power * buffer_strength
        raw_risk = (numerator / denominator).clip(0, 2.0)
        final_risk = (np.float32(2.0) / (np.float32(1.0) + np.exp(-raw_risk * np.float32(3.0))) - np.float32(1.0)).clip(0, 1)
        critical_override = np.float32(0.0)
        if support_intent.iloc[-1] < 20 and is_outflow.iloc[-1]:
            critical_override = np.float32(0.4)
        final_risk = (final_risk + critical_override).clip(0, 1)
        print(f"\n[V11.1_SHIELD_FAILURE_PROBE]")
        print(f"  > [ATTACK] Shock: {shock_norm.iloc[-1]:.4f} | ConsRisk: {consistency_risk.iloc[-1]:.4f} -> Force: {attack_force.iloc[-1]:.4f}")
        print(f"  > [DEFENSE] Intent: {support_intent.iloc[-1]:.1f} | BuySlope: {buy_rate_slope.iloc[-1]:.4f} | Closing: {closing_flow.iloc[-1]:.2e}")
        print(f"  > [DEFENSE_COEF] Power: {defense_power.iloc[-1]:.4f} (IntentF:{intent_factor.iloc[-1]:.2f} * WithD:{withdrawal_factor.iloc[-1]:.2f})")
        print(f"  > [BUFFER] Strength: {buffer_strength.iloc[-1]:.4f} | Depletion: {depletion_impact:.4f}")
        print(f"  > FINAL_VACUUM_RISK: {final_risk.iloc[-1]:.4f} (Raw: {raw_risk.iloc[-1]:.4f})")
        _temp_debug_values["cross_module_signals"]["vacuum_risk"] = final_risk
        return final_risk.astype(np.float32)

    def _calculate_ewd_factor(self, conviction: pd.Series, resilience: pd.Series, context: pd.Series, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V21.0 · 博弈热力学核心】Float32 深度优化
        - 优化内容：熵权计算中的所有指数和系数降级。
        - 版本号：V21.0.0
        """
        # 1. Macro Coherence
        v1, v2, v3 = conviction.clip(-1, 1), resilience.clip(-1, 1), context.clip(-1, 1)
        avg_vec = (v1 + v2 + v3) / np.float32(3.0)
        dispersion = (np.abs(v1 - avg_vec) + np.abs(v2 - avg_vec) + np.abs(v3 - avg_vec)) / np.float32(3.0)
        coherence_score = np.exp(-dispersion * np.float32(3.5))
        # 2. Game Entropy
        game_div = raw_signals['SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D']
        game_accel = raw_signals['ACCEL_5_SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D']
        game_entropy = (np.tanh(game_div.clip(0) * np.float32(1.5)) * np.float32(0.7) + np.tanh(game_accel.clip(0) * np.float32(5.0)) * np.float32(0.3)).clip(0, 1)
        # 3. Criticality Entropy
        winner_rate = raw_signals['winner_rate_D']
        criticality_penalty = (np.float32(1.0) / (np.float32(1.0) + np.exp(-(winner_rate - np.float32(95.0)) * np.float32(0.5)))).clip(0, 1)
        # 4. Structural Entropy
        chip_ent = raw_signals['chip_entropy_D']
        chip_z = ((chip_ent - raw_signals['HAB_LONG_chip_entropy_D']) / (raw_signals['HAB_STD_chip_entropy_D'] + np.float32(1e-4))).clip(0)
        chip_risk = np.tanh(chip_z * np.float32(0.8))
        consistency = raw_signals['flow_consistency_D']
        snr_factor = np.float32(1.0) + (np.float32(1.0) - consistency)
        structural_penalty = (chip_risk * snr_factor).clip(0, 1)
        # 5. Thermal Entropy
        vpa_eff = raw_signals['VPA_EFFICIENCY_D']
        thermal_penalty = (np.float32(1.0) - vpa_eff).clip(0, 1)
        # 6. Fusion
        raw_ewd = coherence_score * (np.float32(1.0) - game_entropy * np.float32(0.8)) * (np.float32(1.0) - criticality_penalty * np.float32(0.6)) * (np.float32(1.0) - structural_penalty * np.float32(0.7)) * (np.float32(1.0) - thermal_penalty * np.float32(0.5))
        v_risk = _temp_debug_values.get("cross_module_signals", {}).get("vacuum_risk", pd.Series(0.0, index=conviction.index, dtype=np.float32))
        final_ewd = np.maximum(raw_ewd, np.tanh(v_risk * np.float32(2.8))).clip(0, 1)
        print(f"\n[V9.5_EWD_GAME_THERMODYNAMICS_PROBE]")
        print(f"  > [MACRO] Coherence: {coherence_score.iloc[-1]:.4f}")
        print(f"  > [GAME] Div: {game_div.iloc[-1]:.4f} | Accel: {game_accel.iloc[-1]:.4f} | Entropy: {game_entropy.iloc[-1]:.4f}")
        print(f"  > [CRITICAL] WinnerRate: {winner_rate.iloc[-1]:.2f}% | Penalty: {criticality_penalty.iloc[-1]:.4f}")
        print(f"  > [STRUCT] ChipZ: {chip_z.iloc[-1]:.4f} | Consistency: {consistency.iloc[-1]:.4f} | Risk: {structural_penalty.iloc[-1]:.4f}")
        print(f"  > FINAL_EWD: {final_ewd.iloc[-1]:.4f}")
        _temp_debug_values["ewd_analysis"] = {"factor": final_ewd, "game_entropy": game_entropy}
        return final_ewd.astype(np.float32)

    def _calculate_stealth_accumulation_bonus(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V21.0 · 暗物质压缩核心】Float32 深度优化
        - 优化内容：暗物质算法中的高维常数全部降级，优化压缩率计算效率。
        - 版本号：V21.0.0
        """
        # 1. Base V9.6
        accum_stealth = raw_signals['HAB_ACCUM_21_stealth_flow_ratio_D']
        inventory_score = (np.float32(1.0) / (np.float32(1.0) + np.exp(-(accum_stealth - np.float32(5.0)) * np.float32(0.8)))).clip(0, 1)
        s_accel = raw_signals['ACCEL_5_stealth_flow_ratio_D']
        s_jerk = raw_signals['JERK_5_stealth_flow_ratio_D']
        s_mad = raw_signals['HAB_MAD_ACCEL_5_stealth_flow_ratio_D']
        kinematic_bonus = np.tanh((s_accel + s_jerk * np.float32(0.5)) / (s_mad * np.float32(3.0) + np.float32(1e-6))).clip(0, 1)
        ent_slope = raw_signals['SLOPE_5_chip_entropy_D']
        ent_mad = raw_signals['HAB_MAD_SLOPE_5_chip_entropy_D']
        ordering_score = (-np.tanh(ent_slope / (ent_mad * np.float32(2.0) + np.float32(1e-6)))).clip(0, 1)
        price_accel = raw_signals['OCH_ACCELERATION_D']
        suppression_score = np.float32(0.0)
        if kinematic_bonus.iloc[-1] > 0.1:
            suppression_score = (np.float32(1.0) - np.tanh(price_accel / np.float32(0.02))).clip(0, 1)
        base_v96 = (inventory_score * np.float32(0.4) + kinematic_bonus * np.float32(0.6)) * (np.float32(1.0) + ordering_score * np.float32(0.5) + suppression_score * np.float32(0.5))
        base_v96 = np.tanh(base_v96).clip(0, 1)
        # 2. Algo Clustering
        clust_val = raw_signals['tick_clustering_index_D']
        clust_mean = raw_signals['HAB_LONG_tick_clustering_index_D']
        clust_std = raw_signals['HAB_STD_tick_clustering_index_D']
        clustering_z = ((clust_val - clust_mean) / (clust_std + np.float32(1e-4))).clip(0)
        clustering_bonus = np.tanh(clustering_z * np.float32(0.8))
        # 3. Energy Compression
        comp_rate = raw_signals['MA_POTENTIAL_COMPRESSION_RATE_D']
        comp_slope = raw_signals['SLOPE_5_MA_POTENTIAL_COMPRESSION_RATE_D']
        compression_bonus = (np.tanh(comp_rate / np.float32(10.0)) * np.float32(0.6) + np.tanh(comp_slope).clip(0) * np.float32(0.4)).clip(0, 1)
        # 4. Fusion
        raw_bonus = base_v96 * (np.float32(1.0) + clustering_bonus * np.float32(0.5)) * (np.float32(1.0) + compression_bonus * np.float32(0.4))
        conf_val = raw_signals['intraday_accumulation_confidence_D']
        conf_coef = np.tanh(conf_val / np.float32(50.0)).clip(0.5, 1.2)
        final_bonus = np.tanh(raw_bonus * conf_coef).clip(0, 1)
        print(f"\n[V9.7_DARK_MATTER_COMPRESSION_PROBE]")
        print(f"  > [BASE_V9.6] Score: {base_v96.iloc[-1]:.4f} (Inv:{inventory_score.iloc[-1]:.2f}, Ord:{ordering_score.iloc[-1]:.2f})")
        print(f"  > [ALGO] ClusteringZ: {clustering_z.iloc[-1]:.4f} -> Bonus: {clustering_bonus.iloc[-1]:.4f}")
        print(f"  > [ENERGY] CompRate: {comp_rate.iloc[-1]:.2f} | Slope: {comp_slope.iloc[-1]:.4f} -> Bonus: {compression_bonus.iloc[-1]:.4f}")
        print(f"  > [GATE] Confidence: {conf_val.iloc[-1]:.1f} -> Coef: {conf_coef.iloc[-1]:.4f}")
        print(f"  > FINAL_STEALTH_BONUS: {final_bonus.iloc[-1]:.4f}")
        _temp_debug_values["stealth_bonus"] = final_bonus
        return final_bonus.astype(np.float32)

    def _calculate_deception_filter(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], params_dict: Dict, method_name: str, _temp_debug_values: Dict) -> pd.Series:
        """
        【V21.0 · 事件视界反欺骗核心】Float32 深度优化
        - 优化内容：将反欺骗算法中的所有阈值和乘数降级为 float32。
        - 版本号：V21.0.0
        """
        # 1. Base Deception
        ab_v, ab_h, ab_m = raw_signals['tick_abnormal_volume_ratio_D'], raw_signals['HAB_LONG_tick_abnormal_volume_ratio_D'], raw_signals['HAB_MAD_tick_abnormal_volume_ratio_D']
        abn_z = np.tanh((ab_v - ab_h) / (ab_m * np.float32(1.618) + np.float32(0.1)))
        vpa_eff = raw_signals['VPA_EFFICIENCY_D']
        trans_eff = np.tanh(raw_signals['tick_chip_transfer_efficiency_D'] / np.float32(8e6))
        churning_score = (abn_z.clip(0) * (np.float32(1.0) - vpa_eff * np.float32(0.6) - trans_eff * np.float32(0.4)).clip(0)).clip(0, 1)
        div_val = raw_signals['price_flow_divergence_D']
        div_slope = raw_signals['SLOPE_5_price_flow_divergence_D']
        div_accel = raw_signals['ACCEL_5_price_flow_divergence_D']
        fracture_risk = (np.tanh(div_val) * np.float32(0.5) + np.tanh(div_slope + div_accel).clip(0) * np.float32(0.5)).clip(0, 1)
        abn_jerk = raw_signals['JERK_5_tick_abnormal_volume_ratio_D']
        abn_jerk_mad = raw_signals['HAB_MAD_JERK_5_tick_abnormal_volume_ratio_D']
        pulse_score = np.tanh(abn_jerk / (abn_jerk_mad * np.float32(2.0) + np.float32(1e-6))).clip(0, 1)
        # 2. Advanced Deception
        sm_rate_accel = raw_signals['ACCEL_5_buy_sm_amount_rate_D']
        sm_mad = raw_signals['HAB_MAD_ACCEL_5_buy_sm_amount_rate_D']
        retail_trap_score = np.tanh(sm_rate_accel.clip(0) / (sm_mad * np.float32(2.0) + np.float32(1e-6))).clip(0, 1)
        dist_conf = raw_signals['intraday_distribution_confidence_D']
        distribution_penalty = np.tanh(dist_conf / np.float32(60.0)).clip(0, 1)
        lock_slope = raw_signals['SLOPE_5_high_position_lock_ratio_90_D']
        locking_fracture = (-np.tanh(lock_slope * np.float32(10.0))).clip(0, 1)
        # 3. Fusion
        base_deception = (churning_score * np.float32(0.25) + fracture_risk * np.float32(0.25) + pulse_score * np.float32(0.2))
        advanced_deception = (retail_trap_score * np.float32(0.2) + distribution_penalty * np.float32(0.25) + locking_fracture * np.float32(0.15))
        total_deception = (base_deception + advanced_deception).clip(0, 1)
        chip_stab = raw_signals['chip_stability_D']
        stability_penalty = (np.float32(1.0) - chip_stab).clip(0, 1)
        boosted_deception = (total_deception * (np.float32(1.0) + stability_penalty * np.float32(0.5))).clip(0, 1)
        filter_score = np.float32(1.0) - boosted_deception
        print(f"\n[V10.0_DECEPTION_EVENT_HORIZON_PROBE]")
        print(f"  > [BASE] Churn: {churning_score.iloc[-1]:.2f} | Frac: {fracture_risk.iloc[-1]:.2f} | Pulse: {pulse_score.iloc[-1]:.2f}")
        print(f"  > [RETAIL] SMAccel: {sm_rate_accel.iloc[-1]:.2e} -> Trap: {retail_trap_score.iloc[-1]:.4f}")
        print(f"  > [DIST] Conf: {dist_conf.iloc[-1]:.1f} -> Penalty: {distribution_penalty.iloc[-1]:.4f}")
        print(f"  > [LOCK] Slope: {lock_slope.iloc[-1]:.4f} -> Fracture: {locking_fracture.iloc[-1]:.4f}")
        print(f"  > FINAL_DECEPTION_SCORE: {boosted_deception.iloc[-1]:.4f} -> FILTER: {filter_score.iloc[-1]:.4f}")
        _temp_debug_values["deception_analysis"] = {"score": boosted_deception, "filter": filter_score}
        return filter_score.astype(np.float32)

    def _calculate_contextual_modulator(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], params_dict: Dict, method_name: str, _temp_debug_values: Dict, is_debug_enabled: bool, probe_ts: pd.Timestamp) -> pd.Series:
        """
        【V21.0 · 合力竞速核心】Float32 深度优化
        - 优化内容：环境调制模块的全部系数降级为 float32。
        - 版本号：V21.0.0
        """
        # 1. Base Context
        theme_val = raw_signals['THEME_HOTNESS_SCORE_D']
        theme_slope = raw_signals['SLOPE_5_THEME_HOTNESS_SCORE_D']
        raw_theme_score = (np.tanh(theme_val / np.float32(50.0)) * np.float32(0.4) + np.tanh(theme_slope).clip(-1, 1) * np.float32(0.4))
        sent_val = raw_signals['market_sentiment_score_D']
        sent_jerk = raw_signals['JERK_5_market_sentiment_score_D']
        sent_mad = raw_signals['HAB_MAD_JERK_5_market_sentiment_score_D']
        stall_risk = np.float32(0.0)
        if sent_val.iloc[-1] > 60:
             stall_risk = (-np.tanh(sent_jerk / (sent_mad * np.float32(2.0) + np.float32(1e-6)))).clip(0, 1)
        accum_sent = raw_signals['HAB_ACCUM_21_market_sentiment_score_D']
        exhaustion_risk = np.tanh((accum_sent - np.float32(1600.0)) / np.float32(200.0)).clip(0, 1)
        sentiment_score = (np.tanh(sent_val / np.float32(50.0)) * (np.float32(1.0) - stall_risk * np.float32(0.7)) * (np.float32(1.0) - exhaustion_risk * np.float32(0.5))).clip(0, 1)
        profit = raw_signals['profit_pressure_D']
        uptrend = raw_signals['uptrend_strength_D']
        structure_score = (np.float32(1.0) - np.tanh((profit / (uptrend + np.float32(1e-4))).clip(0, 100) / np.float32(50.0))).clip(0, 1)
        base_context = (raw_theme_score * np.float32(0.4) + sentiment_score * np.float32(0.4) + structure_score * np.float32(0.2)).clip(0, 1)
        # 2. Synergy Bonus
        synergy_val = raw_signals['SMART_MONEY_SYNERGY_BUY_D']
        synergy_bonus = np.tanh(synergy_val.clip(0) * np.float32(2.0)) * np.float32(0.3)
        # 3. Rank Velocity
        rank_accel = raw_signals['industry_rank_accel_D']
        velocity_mod = np.tanh(rank_accel * np.float32(5.0)) * np.float32(0.15)
        # 4. Stability Filter
        stability = raw_signals['TURNOVER_STABILITY_INDEX_D']
        stability_coef = np.float32(0.8) + (stability.clip(0, 1) * np.float32(0.3))
        # 5. Fusion
        raw_modulator = base_context + synergy_bonus + velocity_mod
        leader_score = raw_signals['industry_leader_score_D']
        immunity = np.float32(0.8) + np.tanh(leader_score / np.float32(40.0)) * np.float32(0.7)
        final_modulator = (raw_modulator * stability_coef * immunity).clip(0, 1)
        print(f"\n[V10.2_SYNERGY_VELOCITY_PROBE]")
        print(f"  > [BASE] Theme:{raw_theme_score.iloc[-1]:.2f} | Sent:{sentiment_score.iloc[-1]:.2f} | Struct:{structure_score.iloc[-1]:.2f}")
        print(f"  > [SYNERGY] Val:{synergy_val.iloc[-1]:.4f} -> Bonus:{synergy_bonus.iloc[-1]:.4f} (Force Multiplier)")
        print(f"  > [VELOCITY] RankAccel:{rank_accel.iloc[-1]:.4f} -> Mod:{velocity_mod.iloc[-1]:.4f}")
        print(f"  > [STABILITY] StabIdx:{stability.iloc[-1]:.4f} -> Coef:{stability_coef.iloc[-1]:.4f}")
        print(f"  > FINAL_CONTEXT_MODULATOR: {final_modulator.iloc[-1]:.4f}")
        _temp_debug_values["context_analysis"] = {"modulator": final_modulator, "synergy_bonus": synergy_bonus}
        return final_modulator.astype(np.float32)

    def _execute_intelligence_probe(self, method_name: str, probe_ts: pd.Timestamp, _temp_debug_values: Dict, final_score: pd.Series):
        """
        【V8.0 · 全息审计版】彻底暴露物理证据链
        """
        print(f"\n{'='*40} [V8.0 HOLOGRAPHIC VERDICT: {probe_ts.strftime('%Y-%m-%d')}] {'='*40}")
        latch = _temp_debug_values.get("latch_state", {})
        fus = _temp_debug_values.get("final_fusion_debug", {})
        print(f"--- [CORE DECISION PHYSICS] ---")
        print(f"  > FinalIntensity: {fus.get('intensity', 0.0):.4f} (Boosted by Catastrophe)")
        print(f"  > NetAfterStealth: {fus.get('raw_net', 0.0):.4f}")
        print(f"  > LatchTrigger: {latch.get('trigger').loc[probe_ts]} (Emergency: {latch.get('emergency').loc[probe_ts]})")
        print(f"  > CONVICTION_DECAY_SCORE: {final_score.loc[probe_ts]:.4f}")
        print(f"{'='*110}\n")

    def _calculate_institutional_stalling_jerk_risk(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V21.0 · 气动失速核心】Float32 深度优化
        - 优化内容：气动公式中的所有常数降级为 float32。
        - 版本号：V21.0.0
        """
        # A. Thrust Failure
        act_jerk = raw_signals['JERK_5_main_force_activity_index_D']
        act_mad = raw_signals['HAB_MAD_JERK_5_main_force_activity_index_D']
        thrust_failure = (-np.tanh(act_jerk / (act_mad * np.float32(2.0) + np.float32(1e-6)))).clip(0, 1)
        # B. Drag Surge
        res_accel = raw_signals['ACCEL_5_flow_resistance_level_D']
        res_mad = raw_signals['HAB_MAD_ACCEL_5_flow_resistance_level_D']
        drag_surge = np.tanh(res_accel / (res_mad * np.float32(1.5) + np.float32(1e-6))).clip(0, 1)
        # C. Lift Collapse
        eff_slope = raw_signals['SLOPE_5_flow_impact_ratio_D']
        lift_loss_factor = np.float32(1.0) + (-np.tanh(eff_slope * np.float32(2.0))).clip(0)
        # D. Inertia & Intent
        inst_accum = raw_signals['HAB_ACCUM_21_SMART_MONEY_INST_NET_BUY_D']
        inst_std = raw_signals['HAB_STD_SMART_MONEY_INST_NET_BUY_D'] * np.sqrt(np.float32(21.0))
        mass_inertia = (np.float32(1.0) / (np.float32(1.0) + np.exp(-inst_accum / (inst_std + np.float32(1e-6)) * np.float32(1.5)))).clip(0.2, 1.0)
        intent_slope = raw_signals['SLOPE_5_INTRADAY_SUPPORT_INTENT_D']
        intent_risk_factor = np.float32(1.0) + (-np.tanh(intent_slope)).clip(0)
        # E. Fusion
        aerodynamic_stress = (thrust_failure * np.float32(0.5) + drag_surge * np.float32(0.5)) * lift_loss_factor
        raw_risk = (aerodynamic_stress / mass_inertia) * intent_risk_factor
        och_accel = raw_signals['OCH_ACCELERATION_D']
        price_context = np.tanh((och_accel + np.float32(0.01)) * np.float32(20.0)).clip(0, 1)
        final_risk = np.tanh(raw_risk * price_context * np.float32(1.5)).clip(0, 1)
        print(f"\n[V12.1_AERODYNAMIC_STALL_PROBE]")
        print(f"  > [THRUST] ActJerk:{act_jerk.iloc[-1]:.2e} -> Failure:{thrust_failure.iloc[-1]:.4f}")
        print(f"  > [DRAG] ResAccel:{res_accel.iloc[-1]:.2e} -> Surge:{drag_surge.iloc[-1]:.4f}")
        print(f"  > [LIFT] EffSlope:{eff_slope.iloc[-1]:.4f} -> LossFactor:{lift_loss_factor.iloc[-1]:.2f}")
        print(f"  > [INERTIA] Mass:{mass_inertia.iloc[-1]:.4f} | IntentRisk:{intent_risk_factor.iloc[-1]:.2f}")
        print(f"  > FINAL_STALL_RISK: {final_risk.iloc[-1]:.4f}")
        _temp_debug_values["cross_module_signals"]["stalling_risk"] = final_risk
        return final_risk.astype(np.float32)

    def _calculate_institutional_erosion_index(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V21.0 · 结构风化侵蚀核心】Float32 深度优化
        - 优化内容：地质力学模型参数降级为 float32。
        - 版本号：V21.0.0
        """
        # A. Landslide & Fissure
        mig_jerk = raw_signals['JERK_5_intraday_cost_center_migration_D']
        mig_mad = raw_signals['HAB_MAD_JERK_5_intraday_cost_center_migration_D']
        landslide_shock = (-np.tanh(mig_jerk / (mig_mad * np.float32(1.618) + np.float32(1e-6)))).clip(0, 1)
        vol_accel = raw_signals['ACCEL_5_intraday_cost_center_volatility_D']
        vol_mad = raw_signals['HAB_MAD_ACCEL_5_intraday_cost_center_volatility_D']
        fissure_spread = np.tanh(vol_accel / (vol_mad * np.float32(1.5) + np.float32(1e-6))).clip(0, 1)
        # B. Kurtosis Collapse
        kurt_slope = raw_signals['SLOPE_5_chip_kurtosis_D']
        kurt_mad = raw_signals['HAB_MAD_SLOPE_5_chip_kurtosis_D']
        kurtosis_collapse = (-np.tanh(kurt_slope / (kurt_mad * np.float32(2.0) + np.float32(1e-6)))).clip(0, 1)
        # C. Friction Heat
        game_val = raw_signals['intraday_chip_game_index_D']
        game_mean = raw_signals['HAB_LONG_intraday_chip_game_index_D']
        game_std = raw_signals['HAB_STD_intraday_chip_game_index_D']
        friction_coef = np.float32(1.0) + np.tanh(((game_val - game_mean) / (game_std + np.float32(1e-4))).clip(0) * np.float32(0.5))
        # D. Sediment Buffer
        sed_accum = raw_signals['HAB_ACCUM_21_intraday_chip_consolidation_degree_D']
        sediment_thickness = (np.float32(1.0) / (np.float32(1.0) + np.exp(-(sed_accum - np.float32(1000.0)) / np.float32(300.0)))).clip(0.1, 1.2)
        # E. Fusion
        structural_damage = (landslide_shock * np.float32(0.35) + fissure_spread * np.float32(0.35) + kurtosis_collapse * np.float32(0.3))
        raw_erosion = (structural_damage * friction_coef) / sediment_thickness
        final_erosion = np.tanh(raw_erosion).clip(0, 1)
        print(f"\n[V13.1_STRUCTURAL_WEATHERING_PROBE]")
        print(f"  > [DAMAGE] Slide:{landslide_shock.iloc[-1]:.2f} | Fissure:{fissure_spread.iloc[-1]:.2f} | Kurtosis:{kurtosis_collapse.iloc[-1]:.2f}")
        print(f"  > [FRICTION] GameIdx:{game_val.iloc[-1]:.2f} -> Coef:{friction_coef.iloc[-1]:.2f}")
        print(f"  > [BUFFER] Thickness:{sediment_thickness.iloc[-1]:.4f}")
        print(f"  > FINAL_EROSION_INDEX: {final_erosion.iloc[-1]:.4f}")
        _temp_debug_values["cross_module_signals"]["erosion_index"] = final_erosion
        return final_erosion.astype(np.float32)

    def _calculate_chaotic_collapse_resonance(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V21.0 · 几何奇点混沌核心】Float32 深度优化
        - 优化内容：混沌算法中的所有物理阈值降级为 float32。
        - 版本号：V21.0.0
        """
        # A. Thermo Entropy
        p_ent_jerk = raw_signals['JERK_5_PRICE_ENTROPY_D']
        p_ent_mad = raw_signals['HAB_MAD_JERK_5_PRICE_ENTROPY_D']
        signal_shock = np.tanh(p_ent_jerk / (p_ent_mad * np.float32(1.618) + np.float32(1e-6))).clip(0, 1)
        c_ent_accel = raw_signals['ACCEL_5_concentration_entropy_D']
        c_ent_mad = raw_signals['HAB_MAD_ACCEL_5_concentration_entropy_D']
        structure_decay = np.tanh(c_ent_accel / (c_ent_mad * np.float32(1.5) + np.float32(1e-6))).clip(0, 1)
        p_ent_accum = raw_signals['HAB_ACCUM_21_PRICE_ENTROPY_D']
        criticality = (np.float32(1.0) / (np.float32(1.0) + np.exp(-(p_ent_accum - np.float32(15.0)) / np.float32(3.0)))).clip(0.5, 1.5)
        en_conc = raw_signals['energy_concentration_D']
        energy_amp = np.float32(1.0) + np.tanh(en_conc / np.float32(50.0)).clip(0)
        div_jerk = raw_signals['JERK_5_high_freq_flow_divergence_D']
        div_mad = raw_signals['HAB_MAD_JERK_5_high_freq_flow_divergence_D']
        micro_trigger = np.tanh(div_jerk / (div_mad * np.float32(2.0) + np.float32(1e-6))).clip(0, 1)
        thermo_chaos = (signal_shock * np.float32(0.4) + structure_decay * np.float32(0.6)) * criticality * energy_amp * (np.float32(1.0) + micro_trigger)
        # B. Geometric Curvature
        curv_jerk = raw_signals['JERK_5_GEOM_ARC_CURVATURE_D']
        curv_mad = raw_signals['HAB_MAD_JERK_5_GEOM_ARC_CURVATURE_D']
        curvature_tear = np.tanh(curv_jerk / (curv_mad * np.float32(2.0) + np.float32(1e-6))).clip(0, 1)
        # C. Attractor Dissipation
        clust_slope = raw_signals['SLOPE_5_flow_cluster_intensity_D']
        clust_mad = raw_signals['HAB_MAD_SLOPE_5_flow_cluster_intensity_D']
        dissipation_factor = np.float32(1.0) + (-np.tanh(clust_slope / (clust_mad + np.float32(1e-6)))).clip(0)
        # D. Systemic Prior
        rev_score = raw_signals['reversal_warning_score_D']
        prior_prob = np.tanh(rev_score / np.float32(60.0)).clip(0.5, 1.5)
        # E. Fusion
        raw_resonance = (thermo_chaos + curvature_tear * np.float32(0.5)) * dissipation_factor * prior_prob
        final_score = np.tanh(raw_resonance * np.float32(0.8)).clip(0, 1)
        print(f"\n[V15.1_GEOMETRIC_SINGULARITY_PROBE]")
        print(f"  > [THERMO] Chaos: {thermo_chaos.iloc[-1]:.4f} (Sig:{signal_shock.iloc[-1]:.2f}, Struc:{structure_decay.iloc[-1]:.2f}, Crit:{criticality.iloc[-1]:.2f})")
        print(f"  > [GEOM] CurvJerk: {curv_jerk.iloc[-1]:.2e} -> Tear: {curvature_tear.iloc[-1]:.4f}")
        print(f"  > [TOPO] ClustSlope: {clust_slope.iloc[-1]:.2e} -> Dissipation: {dissipation_factor.iloc[-1]:.2f}")
        print(f"  > [PRIOR] RevScore: {rev_score.iloc[-1]:.1f} -> Prob: {prior_prob.iloc[-1]:.2f}")
        print(f"  > FINAL_CHAOS_RESONANCE: {final_score.iloc[-1]:.4f}")
        _temp_debug_values["cross_module_signals"]["chaos_resonance"] = final_score
        return final_score.astype(np.float32)

    def _calculate_macro_sector_synergy(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V21.0 · 维度共振协同核心】Float32 深度优化
        - 优化内容：协同算法中的所有权重和Sigmoid参数降级为 float32。
        - 版本号：V21.0.0
        """
        # A. Base Explosion
        stag_jerk = raw_signals['JERK_5_industry_stagnation_score_D']
        stag_mad = raw_signals['HAB_MAD_JERK_5_industry_stagnation_score_D']
        breakout_force = (-np.tanh(stag_jerk / (stag_mad * np.float32(1.5) + np.float32(1e-6)))).clip(0, 1)
        clust_accel = raw_signals['ACCEL_5_flow_cluster_intensity_D']
        clust_mad = raw_signals['HAB_MAD_ACCEL_5_flow_cluster_intensity_D']
        cohesion_force = np.tanh(clust_accel / (clust_mad * np.float32(1.5) + np.float32(1e-6))).clip(0, 1)
        rank_slope = raw_signals['industry_rank_slope_D']
        velocity_score = np.tanh(-rank_slope).clip(0, 1)
        energy_accum = raw_signals['HAB_ACCUM_21_SMART_MONEY_HM_COORDINATED_ATTACK_D']
        potential_factor = (np.float32(1.0) / (np.float32(1.0) + np.exp(-(energy_accum - np.float32(10.0)) / np.float32(3.0)))).clip(0.5, 1.5)
        base_synergy = (breakout_force * np.float32(0.35) + cohesion_force * np.float32(0.35) + velocity_score * np.float32(0.3)) * potential_factor
        # B. Timeframe Sync
        sync_val = raw_signals['daily_monthly_sync_D']
        sync_factor = np.tanh(sync_val / np.float32(20.0)).clip(0.5, 1.3)
        # C. Markup Acceleration
        markup_accel = raw_signals['ACCEL_5_industry_markup_score_D']
        markup_mad = raw_signals['HAB_MAD_ACCEL_5_industry_markup_score_D']
        markup_boost = np.float32(1.0) + np.tanh(markup_accel.clip(0) / (markup_mad + np.float32(1e-6))).clip(0) * np.float32(0.5)
        # D. Structural Veto
        downtrend_score = raw_signals['industry_downtrend_score_D']
        veto_factor = (np.float32(1.0) - np.tanh(downtrend_score / np.float32(40.0))).clip(0, 1)
        # E. Fusion
        raw_synergy = base_synergy * sync_factor * markup_boost * veto_factor
        final_synergy = np.tanh(raw_synergy * np.float32(1.2)).clip(0, 1)
        print(f"\n[V16.8.1_DIMENSIONAL_RESONANCE_PROBE]")
        print(f"  > [BASE] Breakout:{breakout_force.iloc[-1]:.2f} | Cohesion:{cohesion_force.iloc[-1]:.2f} | Pot:{potential_factor.iloc[-1]:.2f}")
        print(f"  > [CHRONOS] SyncVal:{sync_val.iloc[-1]:.2f} -> Factor:{sync_factor.iloc[-1]:.2f}")
        print(f"  > [KAIROS] MarkupAccel:{markup_accel.iloc[-1]:.2f} -> Boost:{markup_boost.iloc[-1]:.2f}")
        print(f"  > [VETO] DownScore:{downtrend_score.iloc[-1]:.1f} -> Factor:{veto_factor.iloc[-1]:.2f}")
        print(f"  > FINAL_SYNERGY_SCORE: {final_synergy.iloc[-1]:.4f}")
        _temp_debug_values["cross_module_signals"]["macro_synergy"] = final_synergy
        return final_synergy.astype(np.float32)

    def _calculate_chain_collapse_resonance(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V21.0 · 多米诺奇点链式核心】Float32 深度优化
        - 优化内容：崩塌计算中的所有系数与指数运算降级为 float32。
        - 版本号：V21.0.0
        """
        # A. Physical Collapse
        lock_jerk = raw_signals['JERK_5_intraday_high_lock_ratio_D']
        lock_mad = raw_signals['HAB_MAD_JERK_5_intraday_high_lock_ratio_D']
        fracture_shock = (-np.tanh(lock_jerk / (lock_mad * np.float32(1.618) + np.float32(1e-6)))).clip(0, 1)
        leader_slope = raw_signals['SLOPE_5_industry_leader_score_D']
        leader_snap = (-np.tanh(leader_slope * np.float32(2.0))).clip(0, 1)
        trapped_accum = raw_signals['HAB_ACCUM_21_pressure_trapped_D']
        trapped_accel = raw_signals['ACCEL_5_pressure_trapped_D']
        gravity_base = (np.float32(1.0) / (np.float32(1.0) + np.exp(-(trapped_accum - np.float32(500.0)) / np.float32(100.0)))).clip(0.5, 1.5)
        gravity_dynamic = np.float32(1.0) + np.tanh(trapped_accel.clip(0) * np.float32(2.0))
        kurt_val = raw_signals['high_freq_flow_kurtosis_D']
        kurt_mean = raw_signals['HAB_LONG_high_freq_flow_kurtosis_D']
        kurt_std = raw_signals['HAB_STD_high_freq_flow_kurtosis_D']
        kurt_z = ((kurt_val - kurt_mean) / (kurt_std + np.float32(1e-4))).clip(0)
        tail_risk = np.float32(1.0) + np.tanh(kurt_z * np.float32(0.5))
        limit_accel = raw_signals['ACCEL_5_down_limit_pct_D']
        limit_mad = raw_signals['HAB_MAD_ACCEL_5_down_limit_pct_D']
        panic_spread = np.tanh(limit_accel.clip(0) / (limit_mad * np.float32(1.5) + np.float32(1e-6))).clip(0, 1)
        price_slope = raw_signals['SLOPE_5_OCH_D']
        turn_jerk = raw_signals['JERK_5_turnover_rate_D']
        turn_mad = raw_signals['HAB_MAD_JERK_5_turnover_rate_D']
        freeze_factor = pd.Series(1.0, index=df_index, dtype=np.float32)
        if price_slope.iloc[-1] < 0 and turn_jerk.iloc[-1] < 0:
            freeze_severity = (-np.tanh(turn_jerk / (turn_mad * np.float32(2.0) + np.float32(1e-6)))).clip(0)
            freeze_factor = np.float32(1.0) + freeze_severity
        physical_collapse = (fracture_shock * np.float32(0.6) + leader_snap * np.float32(0.4)) * gravity_base * gravity_dynamic * tail_risk * (np.float32(1.0) + panic_spread) * freeze_factor
        # B. Distribution Intent
        dist_accel = raw_signals['ACCEL_5_behavior_distribution_D']
        dist_mad = raw_signals['HAB_MAD_ACCEL_5_behavior_distribution_D']
        intent_factor = np.tanh(dist_accel.clip(0) / (dist_mad + np.float32(1e-6))).clip(0, 1)
        dist_accum = raw_signals['HAB_ACCUM_21_behavior_distribution_D']
        accum_risk = (np.float32(1.0) / (np.float32(1.0) + np.exp(-(dist_accum - np.float32(1000.0)) / np.float32(200.0)))).clip(0.5, 1.2)
        distribution_risk = np.float32(1.0) + (intent_factor * accum_risk)
        # C. Reversal Criticality
        rev_prob = raw_signals['reversal_prob_D']
        criticality = np.float32(0.8) + np.tanh((rev_prob - np.float32(0.7)) * np.float32(5.0)).clip(0) * np.float32(0.7)
        # D. Fusion
        raw_resonance = physical_collapse * distribution_risk * criticality
        final_collapse = np.tanh(raw_resonance).clip(0, 1)
        print(f"\n[V18.1_DOMINO_SINGULARITY_PROBE]")
        print(f"  > [PHYSICAL] Base:{physical_collapse.iloc[-1]:.4f} (Panic:{panic_spread.iloc[-1]:.2f}, Freeze:{freeze_factor.iloc[-1]:.2f})")
        print(f"  > [INTENT] DistAccel:{dist_accel.iloc[-1]:.2e} | Accum:{dist_accum.iloc[-1]:.0f} -> Risk:{distribution_risk.iloc[-1]:.2f}")
        print(f"  > [CRITICAL] RevProb:{rev_prob.iloc[-1]:.2f} -> Coef:{criticality.iloc[-1]:.2f}")
        print(f"  > FINAL_CHAIN_COLLAPSE: {final_collapse.iloc[-1]:.4f}")
        _temp_debug_values["cross_module_signals"]["chain_collapse"] = final_collapse
        return final_collapse.astype(np.float32)














