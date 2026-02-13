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
    【V8.0 · 全息相位坍塌版】“赢家信念衰减”高端计算引擎
    升级重点：非线性灾变模型、筹码熵增监控、真空一票否决、活跃维度重整。
    版本号：V8.0.0
    """
    def __init__(self, strategy_instance, helper_instance: ProcessIntelligenceHelper):
        self.strategy = strategy_instance
        self.helper = helper_instance

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V8.0 · 主执行流】引入全量数据探针，执行非线性物理融合
        - 修改思路：由线性权重改为活跃维度动态重整，并增加真空熔断的强制干预。
        """
        print(f"\n{'#'*35} [WINNER_CONVICTION_DECAY V8.0] PHASE INITIATED {'#'*35}")
        method_name = "calculate_winner_conviction_decay"
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
        # 1. 物理层判定：真空熔断 (关键独立因子)
        vacuum_risk = self._calculate_institutional_vacuum_meltdown(df_index, raw_signals, params_dict, _temp_debug_values)
        _temp_debug_values["cross_module_signals"]["vacuum_risk"] = vacuum_risk
        # 2. 结构层判定：信念强度与韧性 (核心对立)
        conv_s = self._calculate_conviction_strength(df, df_index, raw_signals, params_dict, method_name, _temp_debug_values, vacuum_risk)
        res_s = self._calculate_pressure_resilience(df, df_index, raw_signals, params_dict, method_name, _temp_debug_values, vacuum_risk)
        # 3. 修正层判定：诡道、情境与吸筹
        dec_f = self._calculate_deception_filter(df, df_index, raw_signals, params_dict, method_name, _temp_debug_values)
        ctx_m = self._calculate_contextual_modulator(df, df_index, raw_signals, params_dict, method_name, _temp_debug_values, is_debug_enabled, probe_ts)
        st_b = self._calculate_stealth_accumulation_bonus(df_index, raw_signals, _temp_debug_values)
        # 4. 融合层：基于相位坍塌逻辑的终极缝合
        syn_f = self._calculate_synergy_factor(conv_s, res_s, _temp_debug_values)
        final_s = self._perform_final_fusion(df_index, conv_s, res_s, dec_f, st_b, params_dict, _temp_debug_values)
        # 5. 决策层：EWD 因子与极速锁存
        ewd_f = self._calculate_ewd_factor(conv_s, res_s, ctx_m, _temp_debug_values)
        latched_s = self._apply_latch_logic(df_index, final_s, ewd_f, params_dict, _temp_debug_values)
        if is_debug_enabled and probe_ts: self._execute_intelligence_probe(method_name, probe_ts, _temp_debug_values, latched_s)
        return latched_s.astype(np.float32)

    def _get_decay_params_and_signals(self, config: Dict, method_name: str) -> Tuple[Dict, List[str]]:
        """
        【V18.4 · 全维数据视界加载】
        - 修改思路：
            1. 新增 'intraday_trough_filling_degree_D' (主动修复) 和 'intraday_low_lock_ratio_D' (低位锁仓)。
            2. 将 'intraday_trough_filling_degree_D' 加入 accum_21 统计，以监测长周期的"修复基因"。
        - 版本号：V18.4.0
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
        # 核心原子指标清单 (Source: 最终军械库清单.txt)
        required_df_columns = [
            # 基础量价与结构
            'OCH_D', 'OCH_ACCELERATION_D', 'days_since_last_peak_D', 'turnover_rate_D', 'down_limit_pct_D',
            'net_amount_ratio_D', 'profit_pressure_D', 'winner_rate_D', 'uptrend_strength_D', 'close_D',
            # 聪明钱与资金流
            'SMART_MONEY_INST_NET_BUY_D', 'SMART_MONEY_HM_NET_BUY_D', 'SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D',
            'SMART_MONEY_HM_COORDINATED_ATTACK_D', 'SMART_MONEY_SYNERGY_BUY_D',
            'flow_consistency_D', 'flow_zscore_D', 'flow_resistance_level_D', 'flow_impact_ratio_D',
            'flow_cluster_intensity_D', 'closing_flow_intensity_D', 'inflow_persistence_D',
            'stealth_flow_ratio_D', 'buy_lg_amount_rate_D', 'buy_sm_amount_rate_D',
            'high_freq_flow_kurtosis_D', 'high_freq_flow_divergence_D',
            # 筹码与博弈
            'pressure_trapped_D', 'chip_entropy_D', 'chip_stability_D', 'chip_kurtosis_D',
            'chip_stability_change_5d_D', 'concentration_entropy_D', 'energy_concentration_D',
            'intraday_high_lock_ratio_D', 'high_position_lock_ratio_90_D',
            'intraday_chip_consolidation_degree_D', 'intraday_chip_game_index_D',
            'intraday_cost_center_migration_D', 'intraday_cost_center_volatility_D',
            'INTRADAY_SUPPORT_INTENT_D', 'intraday_accumulation_confidence_D',
            'intraday_distribution_confidence_D', 'main_force_activity_index_D',
            'intraday_support_test_count_D', 'cost_5pct_D', # V8.6
            'intraday_trough_filling_degree_D', 'intraday_low_lock_ratio_D', # V8.8 新增
            # 宏观与行业
            'market_sentiment_score_D', 'THEME_HOTNESS_SCORE_D',
            'industry_leader_score_D', 'industry_rank_accel_D', 'industry_rank_slope_D',
            'industry_breadth_score_D', 'industry_downtrend_score_D', 'industry_markup_score_D',
            'industry_stagnation_score_D', 'industry_preheat_score_D',
            'mid_long_sync_D', 'daily_monthly_sync_D', 'trend_confirmation_score_D',
            # 技术形态与信号
            'VPA_EFFICIENCY_D', 'PRICE_ENTROPY_D', 'PRICE_FRACTAL_DIM_D',
            'GEOM_ARC_CURVATURE_D', 'MA_POTENTIAL_COMPRESSION_RATE_D',
            'TURNOVER_STABILITY_INDEX_D', 'tick_large_order_net_D',
            'tick_abnormal_volume_ratio_D', 'tick_clustering_index_D', 'tick_chip_transfer_efficiency_D',
            'price_flow_divergence_D', 'reversal_warning_score_D',
            # V18.0 奇点新增
            'behavior_distribution_D', 'reversal_prob_D'
        ]
        # 配置包
        params_dict = {
            'decay_params': decay_params, 
            'fibo_periods': fibo_periods, 
            'belief_decay_weights': belief_decay_weights,
            'hab_settings': {"short": 13, "medium": 21, "long": 34},
            'latch_params': {"window": 5, "hit_count": 3, "high_score_threshold": 0.55, "core_threshold": 0.35},
            'final_exponent': get_param_value(config.get('final_exponent'), 3.5),
            # 动力学计算清单 (Kinetic Targets for Slope/Accel/Jerk)
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
            # 统计学计算清单 (Statistical Bases for Long/Std/Accum)
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
                    'intraday_support_test_count_D', # 加工硬化
                    'intraday_trough_filling_degree_D' # V8.8: 修复基因
                ]
            }
        }
        return params_dict, list(set(required_df_columns))

    def _get_raw_signals(self, df: pd.DataFrame, df_index: pd.Index, params_dict: Dict, method_name: str) -> Dict[str, pd.Series]:
        """
        【V18.6 · 动力学场生成器】修复基础指标 MAD 缺失问题
        - 修改思路：
            1. 修复 KeyError: 'HAB_MAD_tick_abnormal_volume_ratio_D'。
            2. 在动力学循环中，显式计算 Base Series 的 MAD 值，而不仅仅是 Slope/Accel/Jerk 的 MAD。
            3. 保持所有手动加载和统计学计算逻辑不变。
        - 版本号：V18.6.0
        """
        raw_signals = {}
        hab_cfg = params_dict['hab_settings']
        kinetic_targets = params_dict['kinetic_targets']
        stat_targets = params_dict['stat_targets']
        
        print(f"\n[V18.6_KINETIC_FIELD_GENERATION]")
        
        # 1. 基础加载 (Base Loading)
        all_cols = set(kinetic_targets + stat_targets['long_std'] + stat_targets['long_only'] + stat_targets['accum_21'])
        
        # 额外需要但未在上述自动列表中的独立列 (Manual Additions)
        manual_additions = [
            'days_since_last_peak_D', 'winner_rate_D', 'net_amount_ratio_D', 'TURNOVER_STABILITY_INDEX_D',
            'tick_chip_transfer_efficiency_D', 'intraday_distribution_confidence_D', 'chip_stability_D',
            'uptrend_strength_D', 'industry_rank_accel_D', 'industry_rank_slope_D', 'SMART_MONEY_SYNERGY_BUY_D',
            'daily_monthly_sync_D', 'industry_downtrend_score_D', 'OCH_ACCELERATION_D', 'energy_concentration_D',
            'reversal_warning_score_D', 'close_D',
            'cost_5pct_D', 'intraday_support_test_count_D', 'chip_stability_change_5d_D',
            'intraday_trough_filling_degree_D', 'intraday_low_lock_ratio_D',
            'closing_flow_intensity_D', 'inflow_persistence_D'
        ]
        
        for col in manual_additions:
            all_cols.add(col)
            
        for col in all_cols:
            raw_signals[col] = self.helper._get_safe_series(df, col, 0.0)

        # 2. 动力学衍生 (Kinetic Derivatives)
        period = 5
        for target in kinetic_targets:
            base_series = raw_signals[target]
            slope = ta.slope(base_series, length=period).fillna(0)
            raw_signals[f'SLOPE_{period}_{target}'] = slope
            accel = ta.slope(slope, length=period).fillna(0)
            raw_signals[f'ACCEL_{period}_{target}'] = accel
            jerk = ta.slope(accel, length=period).fillna(0)
            raw_signals[f'JERK_{period}_{target}'] = jerk
            # 修复：将 (base_series, target) 加入循环，确保计算基础指标的 HAB_MAD
            calc_list = [
                (base_series, target), # 新增：计算基础指标的 MAD (如 HAB_MAD_tick_abnormal_volume_ratio_D)
                (slope, f'SLOPE_{period}_{target}'), 
                (accel, f'ACCEL_{period}_{target}'), 
                (jerk, f'JERK_{period}_{target}')
            ]
            for series, name in calc_list:
                rolling_median = series.rolling(window=hab_cfg['long']).median()
                mad = (series - rolling_median).abs().rolling(window=hab_cfg['long']).median().fillna(0).replace(0, 1e-6)
                raw_signals[f'HAB_MAD_{name}'] = mad

        # 3. 统计学衍生 (Statistical Derivatives)
        for target in stat_targets['long_std']:
            s = raw_signals[target]
            raw_signals[f'HAB_LONG_{target}'] = s.rolling(window=hab_cfg['long']).mean().fillna(0)
            raw_signals[f'HAB_STD_{target}'] = s.rolling(window=hab_cfg['long']).std().fillna(0).replace(0, 1e-4)
            
        for target in stat_targets['long_only']:
            s = raw_signals[target]
            raw_signals[f'HAB_LONG_{target}'] = s.rolling(window=hab_cfg['long']).mean().fillna(0)
            
        for target in stat_targets['accum_21']:
            s = raw_signals[target]
            raw_signals[f'HAB_ACCUM_21_{target}'] = s.rolling(window=21).sum().fillna(0)

        # 4. 特殊修复 (Specific Fixes)
        if 'OCH_ACCELERATION_D' not in raw_signals:
             raw_signals['OCH_ACCELERATION_D'] = pd.Series(0.0, index=df_index)

        return raw_signals

    def _calculate_conviction_strength(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], params_dict: Dict, method_name: str, _temp_debug_values: Dict, vacuum_risk: pd.Series) -> pd.Series:
        """
        【V8.5 · 重力坍塌融合】引入游资接力与时空势能耗尽
        - 版本号：V8.5.0
        """
        w = params_dict['belief_decay_weights']
        sync_num = raw_signals['mid_long_sync_D'] - raw_signals['HAB_LONG_mid_long_sync_D'] + raw_signals['SLOPE_5_mid_long_sync_D']
        sync_decay = (-np.tanh(sync_num / raw_signals['HAB_STD_mid_long_sync_D'])).clip(0)
        handover_risk = np.tanh(raw_signals['SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D'].clip(0) * 2.0)
        peak_days = raw_signals['days_since_last_peak_D']
        potential_energy = (1.0 - np.exp(-peak_days / 13.0)).clip(0, 1)
        sub_risks = {
            "mid_long_sync_decay": sync_decay * (1.0 + handover_risk),
            "handover_erosion_risk": handover_risk,
            "potential_energy_exhaustion": potential_energy,
            "institutional_vacuum_meltdown": vacuum_risk,
            "institutional_stalling_jerk": self._calculate_institutional_stalling_jerk_risk(df_index, raw_signals, _temp_debug_values),
            "vpa_efficiency_collapse": (-np.tanh(raw_signals['SLOPE_5_VPA_EFFICIENCY_D'])).clip(0),
            "chaotic_collapse_resonance": self._calculate_chaotic_collapse_resonance(df_index, raw_signals, _temp_debug_values),
            "inst_erosion_risk": self._calculate_institutional_erosion_index(df_index, raw_signals, _temp_debug_values),
            "chain_collapse_resonance": self._calculate_chain_collapse_resonance(df_index, raw_signals, _temp_debug_values)
        }
        active_sum, active_weight = pd.Series(0.0, index=df_index), 0.0
        print(f"\n[V8.5_CONVICTION_AUDIT]")
        for k, v in sub_risks.items():
            val = v.iloc[-1]
            if val > 1e-3:
                weight = w.get(k, 0.05)
                active_weight += weight
                active_sum += v * weight
                print(f"  - [PHYSICS] {k}: {val:.4f} (W:{weight:.2f})")
        comp = 1.0 / max(active_weight, 0.4)
        fused = (active_sum * comp).clip(0, 1.0)
        _temp_debug_values["conviction_dynamics"].update({"fused": fused, "handover": handover_risk})
        print(f"  >> FUSED_CONVICTION: {fused.iloc[-1]:.4f} | Comp: {comp:.2f}")
        return fused

    def _calculate_pressure_resilience(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], params_dict: Dict, method_name: str, _temp_debug_values: Dict, vacuum_risk: pd.Series) -> pd.Series:
        """
        【V8.8.1 · 超材料力学与主动修复核心】修复探针打印格式错误
        - 修改思路：
            1. 修复 hardening_bonus 和 corrosion_penalty 在打印时未转为标量的 TypeError。
            2. 确保 shock_energy 和 fatigue_penalty 保持标量计算逻辑不变。
        - 版本号：V8.8.1
        """
        # --- 1. Base Modulus V8.6 (弹性 + 刚性 + 自愈) ---
        test_count = raw_signals['intraday_support_test_count_D']
        intent = raw_signals['INTRADAY_SUPPORT_INTENT_D']
        intent_factor = np.tanh((intent - 40.0) / 20.0)
        stress_log = np.log1p(test_count)
        elastic_modulus = intent_factor * stress_log # 弹性
        
        close_p = raw_signals['close_D']
        cost_floor = raw_signals['cost_5pct_D']
        floor_gap = ((close_p - cost_floor) / (close_p + 1e-4)).clip(lower=-0.1)
        rigidity_score = np.exp(-(floor_gap ** 2) * 50.0) # 刚性
        
        stab_change = raw_signals['chip_stability_change_5d_D']
        healing_score = np.tanh(stab_change * 8.0).clip(-1, 1) # 自愈
        
        # --- 2. Hyper-Material Mechanics V8.8 (超材料特性) ---
        # A. 主动修复 (Active Repair)
        # 填坑度：假设 0-100。高填坑度代表多头在日内的反击能力。
        filling_degree = raw_signals['intraday_trough_filling_degree_D']
        active_repair = np.tanh(filling_degree / 40.0).clip(0, 1.2) # 奖励系数
        
        # B. 低位超导 (Low-Temp Superconductivity)
        # 低位锁仓率：假设 0-100。在低位不卖，说明底部极其坚固。
        low_lock = raw_signals['intraday_low_lock_ratio_D']
        lock_bonus = np.tanh((low_lock - 30.0) / 20.0).clip(0, 0.8)
        
        # 综合 Base Modulus
        # 权重调整：主动修复和低位锁仓是实战检验过的硬指标，权重较高
        base_modulus = (np.tanh(elastic_modulus) * 0.3 + rigidity_score * 0.25 + healing_score * 0.2 + active_repair * 0.25 + lock_bonus * 0.2)
        
        # --- 3. Kinetic Dynamics V8.7 (动力学修正) ---
        price_velocity = raw_signals['SLOPE_5_OCH_D']
        impact_ratio = raw_signals['flow_impact_ratio_D']
        # 注意：shock_energy 和 fatigue_penalty 在此处通过 iloc[-1] 计算，已经是标量
        shock_energy = 0.0
        if price_velocity.iloc[-1] < 0:
            v_factor = abs(price_velocity.iloc[-1]) * 20.0
            i_factor = np.tanh(impact_ratio.iloc[-1] / 5.0)
            shock_energy = (v_factor * (1.0 + i_factor)).clip(0, 1.5)
            
        intent_slope = raw_signals['SLOPE_5_INTRADAY_SUPPORT_INTENT_D']
        fatigue_penalty = 0.0
        if intent_slope.iloc[-1] < 0:
            fatigue_penalty = np.tanh(abs(intent_slope.iloc[-1]) * 0.5).clip(0, 1)
            
        kinetic_factor = 1.0 - (shock_energy * 0.5 + fatigue_penalty * 0.5)
        
        # --- 4. HAB Memory V8.7 & V8.8 (存量意识) ---
        # A. 结构硬化 (Structure Hardening)
        # 累积测试次数
        accum_tests = raw_signals['HAB_ACCUM_21_intraday_support_test_count_D']
        # 累积修复基因 (V8.8新增)
        accum_fill = raw_signals['HAB_ACCUM_21_intraday_trough_filling_degree_D']
        # 综合硬化分：测试多 且 修复好
        hardening_raw = (accum_tests / 50.0) + (accum_fill / 1000.0) # 归一化估算
        hardening_bonus = np.tanh(hardening_raw).clip(0, 0.6)
        
        # B. 锈蚀负担 (Corrosion)
        accum_trapped = raw_signals['HAB_ACCUM_21_pressure_trapped_D']
        corrosion_penalty = np.tanh(accum_trapped / 2000.0).clip(0, 0.8)
        
        hab_factor = 1.0 + hardening_bonus - corrosion_penalty
        
        # --- 5. 综合韧性 (Final Resilience) ---
        raw_resilience = base_modulus * kinetic_factor * hab_factor
        
        # --- 6. 真空脆性熔断 ---
        brittle_fracture = np.where(vacuum_risk > 0.6, vacuum_risk * 1.5, 0.0)
        
        final_resilience = (raw_resilience - brittle_fracture).clip(-1, 1)
        
        # 7. 全息探针 (修复：增加 .iloc[-1] 给 Series 变量)
        print(f"\n[V8.8.1_HYPER_MATERIAL_MECHANICS_PROBE]")
        print(f"  > [BASE] Modulus: {base_modulus.iloc[-1]:.4f} (Repair:{active_repair.iloc[-1]:.2f}, Lock:{lock_bonus.iloc[-1]:.2f})")
        # shock_energy 和 fatigue_penalty 已经是标量，直接打印
        print(f"  > [KINETIC] ShockEnergy: {shock_energy:.4f} | Fatigue: {fatigue_penalty:.4f}")
        # hardening_bonus 和 corrosion_penalty 是 Series，需要 iloc[-1]
        print(f"  > [MEMORY] Hardening: +{hardening_bonus.iloc[-1]:.4f} (AccumFill:{accum_fill.iloc[-1]:.0f}) | Corrosion: -{corrosion_penalty.iloc[-1]:.4f}")
        print(f"  > FINAL_RESILIENCE: {final_resilience.iloc[-1]:.4f}")
        
        _temp_debug_values["resilience_analysis"] = {
            "base": base_modulus, "repair": active_repair, "hardening": hardening_bonus, "fracture": brittle_fracture
        }
        return final_resilience

    def _perform_final_fusion(self, df_index: pd.Index, conviction_score: pd.Series, resilience_score: pd.Series, deception_filter: pd.Series, stealth_bonus: pd.Series, params_dict: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V8.0 · 几何共振融合版】执行非线性衰减合成
        - 修改思路：韧性崩塌与信念衰减由加法改为几何增强，并抑制过度对冲。
        - 版本号：V8.0.0
        """
        exp = params_dict['final_exponent']
        # 1. 韧性崩塌分量
        res_collapse = resilience_score.clip(upper=0).abs()
        res_resist = resilience_score.clip(lower=0)
        # 2. 核心增强公式：信念 * (1 + 崩塌^2 * 1.5)
        # 使用平方项强化“断崖式”崩塌的敏感度
        raw_intensity = conviction_score * (1.0 + (res_collapse ** 2) * 1.8) - res_resist * 0.3
        intensity = raw_intensity.clip(0, 1.2) # 允许轻微溢出用于非线性压缩
        # 3. 隐秘对冲抑制 (上限封锁)
        st_b_constrained = stealth_bonus.fillna(0).clip(0, 0.4)
        raw_net = (intensity * (2 - deception_filter.fillna(1))) * (1 - st_b_constrained * 0.4)
        # 4. 指数映射 (V8.0 使用 3.5 保证信号穿透)
        net_decay = raw_net.clip(-1, 1).fillna(0)
        final = np.sign(net_decay) * (net_decay.abs() ** exp)
        _temp_debug_values["final_fusion_debug"] = {"intensity": intensity.iloc[-1], "raw_net": raw_net.iloc[-1], "exponent": exp}
        print(f"\n[V8.0_FINAL_FUSION_COMPONENTS]")
        print(f"  - Conviction: {conviction_score.iloc[-1]:.4f} | ResCollapseBoost: {(res_collapse**2*1.8).iloc[-1]:.4f}")
        print(f"  - Intensity: {intensity.iloc[-1]:.4f} | StealthHedging: {(st_b_constrained*0.4).iloc[-1]:.4f}")
        print(f"  - FinalScore: {final.iloc[-1]:.4e}")
        return final.clip(-1, 1).fillna(0)

    def _apply_latch_logic(self, df_index: pd.Index, fused_score: pd.Series, ewd_factor: pd.Series, params_dict: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V8.0 · 物理极速锁存版】
        - 修改思路：集成真空与筹码熵的联合锁存。
        """
        lp = params_dict['latch_params']
        is_high = fused_score.abs() > lp["high_score_threshold"]
        rolling_count = is_high.rolling(window=lp["window"]).sum()
        vacuum_risk = _temp_debug_values.get("cross_module_signals", {}).get("vacuum_risk", pd.Series(0.0, index=df_index))
        # 锁存判定：频次、熵权或极速通道 (真空熔断或筹码崩坏)
        is_emergency = (vacuum_risk > 0.8) | (fused_score.abs() > 0.85)
        latch_trigger = ((rolling_count >= lp["hit_count"]) & (ewd_factor > lp["entropy_threshold"])) | is_emergency
        latched_score = np.tanh(fused_score * 1.618)
        protected_score = fused_score.copy()
        for i in range(1, len(fused_score)):
            if latch_trigger.iloc[i]:
                if is_emergency.iloc[i]: protected_score.iloc[i] = latched_score.iloc[i]
                elif abs(fused_score.iloc[i]) > lp["core_threshold"]:
                    prev, curr = protected_score.iloc[i-1], fused_score.iloc[i]
                    if np.sign(prev) == np.sign(curr):
                        protected_score.iloc[i] = curr if abs(curr) > abs(prev) else prev * lp["momentum_protection_factor"]
        final_output = protected_score.clip(-1, 1)
        _temp_debug_values["latch_state"] = {"count": rolling_count, "trigger": latch_trigger, "emergency": is_emergency}
        return final_output

    def _calculate_institutional_vacuum_meltdown(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], params_dict: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V11.1 · 盾构防御熔断核心】修复探针打印格式错误
        - 修改思路：
            1. 修复在 print f-string 中直接格式化 Series 导致的 TypeError。
            2. 确保所有探针输出均使用 .iloc[-1] 获取最新标量值。
        - 版本号：V11.1.1
        """
        # --- A. 攻击侧 (Attack Side) ---
        # 1. 动力学冲击 (Kinetic Shock - Jerk)
        inst_jerk = raw_signals['JERK_5_SMART_MONEY_INST_NET_BUY_D']
        inst_jerk_mad = raw_signals['HAB_MAD_JERK_5_SMART_MONEY_INST_NET_BUY_D']
        shock_norm = (-inst_jerk / (inst_jerk_mad * 1.618 + 1e-6)).clip(0)
        # 2. 一致性崩塌 (Consistency Collapse)
        cons_slope = raw_signals['SLOPE_5_flow_consistency_D']
        cons_mad = raw_signals['HAB_MAD_SLOPE_5_flow_consistency_D']
        consistency_risk = (-np.tanh(cons_slope / (cons_mad + 1e-6))).clip(0, 1)
        attack_force = (np.tanh(shock_norm) * 0.6 + consistency_risk * 0.4).clip(0, 1)
        # --- B. 防御侧 (Defense Side) - NEW ---
        # 1. 护盘意愿 (Support Intent)
        # 假设 intent 范围 0~100。归一化后，意愿越高(1.0)，分母越大，风险越小。
        support_intent = raw_signals['INTRADAY_SUPPORT_INTENT_D']
        intent_factor = (support_intent / 80.0).clip(0.1, 1.2) # 0.1保底防止除零
        # 2. 买盘撤退 (Bid Withdrawal)
        # 关注 buy_lg_amount_rate_D 的 Slope。如果 Slope < 0，说明买盘在撤退。
        buy_rate_slope = raw_signals['SLOPE_5_buy_lg_amount_rate_D']
        # 如果 Slope < 0，def_factor < 1，风险放大；Slope > 0，def_factor > 1，风险缩小
        withdrawal_factor = 1.0 + np.tanh(buy_rate_slope * 5.0) # 范围 0~2
        withdrawal_factor = withdrawal_factor.clip(0.2, 1.5)
        # 3. 尾盘放弃 (Closing Failure)
        closing_flow = raw_signals['closing_flow_intensity_D']
        # 如果尾盘流出( < 0)，防御力打折
        closing_penalty = 1.0
        if closing_flow.iloc[-1] < 0:
            closing_penalty = 1.0 - np.tanh(abs(closing_flow.iloc[-1]) / 1e7) # 流出越大，系数越小
        # 综合防御系数
        defense_power = (intent_factor * withdrawal_factor * closing_penalty).clip(0.1, 2.0)
        # --- C. 存量缓冲 (Inventory Buffer) ---
        accum_21 = raw_signals['HAB_ACCUM_21_SMART_MONEY_INST_NET_BUY_D']
        std_accum_ref = raw_signals['HAB_STD_SMART_MONEY_INST_NET_BUY_D'] * np.sqrt(21)
        buffer_strength = (1 / (1 + np.exp(-accum_21 / (std_accum_ref + 1e-6) * 2))).clip(0.1, 1.0) # 0.1保底
        # --- D. 消耗率 (Depletion) ---
        inst_net = raw_signals['SMART_MONEY_INST_NET_BUY_D']
        is_outflow = inst_net < 0
        depletion_impact = 0.0
        if is_outflow.iloc[-1]:
             if accum_21.iloc[-1] > 0:
                 depletion_impact = (abs(inst_net.iloc[-1]) / accum_21.iloc[-1]).clip(0, 1)
             else:
                 depletion_impact = 1.0 # 存量为负还在流出，消耗率拉满
        # --- E. 综合熔断计算 ---
        # 核心公式：Risk = (Attack * (1 + Depletion)) / (Defense * Buffer)
        # 解释：攻击力越强、消耗越快，分子越大；护盘越强、存量越厚，分母越大。
        numerator = attack_force * (1.0 + depletion_impact * 1.5)
        denominator = defense_power * buffer_strength
        raw_risk = (numerator / denominator).clip(0, 2.0) # 允许溢出
        # Sigmoid 映射到 0~1
        final_risk = (2 / (1 + np.exp(-raw_risk * 3)) - 1).clip(0, 1)
        # 极端修正：如果意愿极低(intent<20)且流出，强制熔断
        critical_override = 0.0
        if support_intent.iloc[-1] < 20 and is_outflow.iloc[-1]:
            critical_override = 0.4
        final_risk = (final_risk + critical_override).clip(0, 1)
        # F. 全息探针
        print(f"\n[V11.1_SHIELD_FAILURE_PROBE]")
        print(f"  > [ATTACK] Shock: {shock_norm.iloc[-1]:.4f} | ConsRisk: {consistency_risk.iloc[-1]:.4f} -> Force: {attack_force.iloc[-1]:.4f}")
        print(f"  > [DEFENSE] Intent: {support_intent.iloc[-1]:.1f} | BuySlope: {buy_rate_slope.iloc[-1]:.4f} | Closing: {closing_flow.iloc[-1]:.2e}")
        # 修复：intent_factor 和 withdrawal_factor 增加 .iloc[-1]
        print(f"  > [DEFENSE_COEF] Power: {defense_power.iloc[-1]:.4f} (IntentF:{intent_factor.iloc[-1]:.2f} * WithD:{withdrawal_factor.iloc[-1]:.2f})")
        print(f"  > [BUFFER] Strength: {buffer_strength.iloc[-1]:.4f} | Depletion: {depletion_impact:.4f}")
        print(f"  > FINAL_VACUUM_RISK: {final_risk.iloc[-1]:.4f} (Raw: {raw_risk.iloc[-1]:.4f})")
        _temp_debug_values["cross_module_signals"]["vacuum_risk"] = final_risk
        return final_risk

    def _calculate_ewd_factor(self, conviction: pd.Series, resilience: pd.Series, context: pd.Series, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V9.5 · 博弈热力学核心】
        逻辑公式：EWD = Coherence * (1 - GameEntropy) * (1 - Criticality) * (1 - Structure) * (1 - Thermal)
        - GameEntropy: 机构跑游资接 = 高熵。
        - Criticality: 获利盘 > 95% = 临界高熵。
        - Structure: 筹码熵/分形维数高 = 高熵。
        - Thermal: 效率低 = 高熵。
        - 版本号：V9.5.0
        """
        # 1. 宏观相干熵 (Coherence - 保持不变)
        v1, v2, v3 = conviction.clip(-1, 1), resilience.clip(-1, 1), context.clip(-1, 1)
        avg_vec = (v1 + v2 + v3) / 3.0
        dispersion = (np.abs(v1 - avg_vec) + np.abs(v2 - avg_vec) + np.abs(v3 - avg_vec)) / 3.0
        coherence_score = np.exp(-dispersion * 3.5)
        
        # 2. 博弈熵 (Game Entropy) [新增] - "对手盘质量"
        # Divergence > 0 表示机构卖、游资买 (背离)。值越大，筹码质量越差。
        game_div = raw_signals['SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D']
        # 动力学：背离是否在加速扩大 (Accel > 0)
        game_accel = raw_signals['ACCEL_5_SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D']
        # 博弈风险：静态背离 + 动态恶化
        game_entropy = (np.tanh(game_div.clip(0) * 1.5) * 0.7 + np.tanh(game_accel.clip(0) * 5.0) * 0.3).clip(0, 1)
        
        # 3. 临界熵 (Criticality Entropy) [新增] - "沙堆崩塌风险"
        # 获利盘 > 95% 时，处于临界态。此时任何风吹草动都是高风险。
        winner_rate = raw_signals['winner_rate_D']
        # 临界惩罚：90%以下风险低，95%以上风险指数级上升
        criticality_penalty = (1 / (1 + np.exp(-(winner_rate - 95.0) * 0.5))).clip(0, 1)
        
        # 4. 结构熵 (Structural Entropy) - "微观混乱"
        chip_ent = raw_signals['chip_entropy_D']
        chip_z = ((chip_ent - raw_signals['HAB_LONG_chip_entropy_D']) / (raw_signals['HAB_STD_chip_entropy_D'] + 1e-4)).clip(0)
        chip_risk = np.tanh(chip_z * 0.8)
        
        # 信噪比修正：如果一致性低，结构熵会被放大
        consistency = raw_signals['flow_consistency_D'] # 0~1
        snr_factor = 1.0 + (1.0 - consistency) # 一致性越低，因子越大(最大2.0)
        structural_penalty = (chip_risk * snr_factor).clip(0, 1)
        
        # 5. 热力熵 (Thermal Entropy) - "无效做功"
        vpa_eff = raw_signals['VPA_EFFICIENCY_D']
        thermal_penalty = (1.0 - vpa_eff).clip(0, 1)
        
        # 6. 融合计算
        # EWD = 相干性 * (1 - 各种熵增惩罚)
        # 注意：每一项都是 1.0 (完美) ~ 0.0 (糟糕)
        raw_ewd = coherence_score * (1.0 - game_entropy * 0.8) * (1.0 - criticality_penalty * 0.6) * (1.0 - structural_penalty * 0.7) * (1.0 - thermal_penalty * 0.5)
        
        # 7. 真空奇点保护
        v_risk = _temp_debug_values.get("cross_module_signals", {}).get("vacuum_risk", pd.Series(0.0, index=conviction.index))
        final_ewd = np.maximum(raw_ewd, np.tanh(v_risk * 2.8)).clip(0, 1)
        
        # 8. 全息探针
        print(f"\n[V9.5_EWD_GAME_THERMODYNAMICS_PROBE]")
        print(f"  > [MACRO] Coherence: {coherence_score.iloc[-1]:.4f}")
        print(f"  > [GAME] Div: {game_div.iloc[-1]:.4f} | Accel: {game_accel.iloc[-1]:.4f} | Entropy: {game_entropy.iloc[-1]:.4f}")
        print(f"  > [CRITICAL] WinnerRate: {winner_rate.iloc[-1]:.2f}% | Penalty: {criticality_penalty.iloc[-1]:.4f}")
        print(f"  > [STRUCT] ChipZ: {chip_z.iloc[-1]:.4f} | Consistency: {consistency.iloc[-1]:.4f} | Risk: {structural_penalty.iloc[-1]:.4f}")
        print(f"  > FINAL_EWD: {final_ewd.iloc[-1]:.4f}")
        
        _temp_debug_values["ewd_analysis"] = {"factor": final_ewd, "game_entropy": game_entropy}
        return final_ewd

    def _calculate_stealth_accumulation_bonus(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V9.7 · 暗物质压缩核心】
        逻辑公式：Bonus = V9.6基础 * (1 + 算法聚集) * (1 + 能量压缩) * 置信度校验
        - 算法聚集：Clustering Index 异常高，说明拆单严重。
        - 能量压缩：Compression Rate 高，说明波动率窒息，吸筹末端。
        - 置信度：Intraday Confidence 作为最终的“门控开关”。
        - 版本号：V9.7.0
        """
        # --- 1. V9.6 基础逻辑 (存量 + 动力学 + 熵减 + 压盘) ---
        accum_stealth = raw_signals['HAB_ACCUM_21_stealth_flow_ratio_D']
        inventory_score = (1 / (1 + np.exp(-(accum_stealth - 5.0) * 0.8))).clip(0, 1)
        s_accel = raw_signals['ACCEL_5_stealth_flow_ratio_D']
        s_jerk = raw_signals['JERK_5_stealth_flow_ratio_D']
        s_mad = raw_signals['HAB_MAD_ACCEL_5_stealth_flow_ratio_D']
        kinematic_bonus = np.tanh((s_accel + s_jerk * 0.5) / (s_mad * 3.0 + 1e-6)).clip(0, 1)
        ent_slope = raw_signals['SLOPE_5_chip_entropy_D']
        ent_mad = raw_signals['HAB_MAD_SLOPE_5_chip_entropy_D']
        ordering_score = (-np.tanh(ent_slope / (ent_mad * 2.0 + 1e-6))).clip(0, 1)
        price_accel = raw_signals['OCH_ACCELERATION_D']
        suppression_score = 0.0
        if kinematic_bonus.iloc[-1] > 0.1:
            suppression_score = (1.0 - np.tanh(price_accel / 0.02)).clip(0, 1)
        
        # 基础分
        base_v96 = (inventory_score * 0.4 + kinematic_bonus * 0.6) * (1.0 + ordering_score * 0.5 + suppression_score * 0.5)
        base_v96 = np.tanh(base_v96).clip(0, 1)
        
        # --- 2. 算法聚集 (Algo Clustering) [新增] ---
        # 逻辑：Tick聚集度越高，拆单越明显。计算 Z-Score。
        clust_val = raw_signals['tick_clustering_index_D']
        clust_mean = raw_signals['HAB_LONG_tick_clustering_index_D']
        clust_std = raw_signals['HAB_STD_tick_clustering_index_D']
        # 异常聚集：超过历史均值 1 个标准差以上
        clustering_z = ((clust_val - clust_mean) / (clust_std + 1e-4)).clip(0)
        clustering_bonus = np.tanh(clustering_z * 0.8) # 映射到 0~1
        
        # --- 3. 能量压缩 (Energy Compression) [新增] ---
        # 逻辑：压缩率越高，爆发越近。且要求压缩正在“收敛”(Slope > 0)
        comp_rate = raw_signals['MA_POTENTIAL_COMPRESSION_RATE_D']
        comp_slope = raw_signals['SLOPE_5_MA_POTENTIAL_COMPRESSION_RATE_D']
        # 压缩分：绝对值高 + 正在收敛
        compression_bonus = (np.tanh(comp_rate / 10.0) * 0.6 + np.tanh(comp_slope).clip(0) * 0.4).clip(0, 1)
        
        # --- 4. 综合计算 ---
        # 乘数效应：如果有算法拆单，吸筹概率大幅提升；如果有压缩，爆发概率提升
        raw_bonus = base_v96 * (1.0 + clustering_bonus * 0.5) * (1.0 + compression_bonus * 0.4)
        
        # --- 5. 置信度门控 (Confidence Gate) ---
        # 利用数据层的 intraday_accumulation_confidence_D (假设范围0-100)
        conf_val = raw_signals['intraday_accumulation_confidence_D']
        conf_coef = np.tanh(conf_val / 50.0).clip(0.5, 1.2) # 最低0.5，最高1.2倍放大
        
        final_bonus = np.tanh(raw_bonus * conf_coef).clip(0, 1)
        
        # 6. 全息探针
        print(f"\n[V9.7_DARK_MATTER_COMPRESSION_PROBE]")
        print(f"  > [BASE_V9.6] Score: {base_v96.iloc[-1]:.4f} (Inv:{inventory_score.iloc[-1]:.2f}, Ord:{ordering_score.iloc[-1]:.2f})")
        print(f"  > [ALGO] ClusteringZ: {clustering_z.iloc[-1]:.4f} -> Bonus: {clustering_bonus.iloc[-1]:.4f}")
        print(f"  > [ENERGY] CompRate: {comp_rate.iloc[-1]:.2f} | Slope: {comp_slope.iloc[-1]:.4f} -> Bonus: {compression_bonus.iloc[-1]:.4f}")
        print(f"  > [GATE] Confidence: {conf_val.iloc[-1]:.1f} -> Coef: {conf_coef.iloc[-1]:.4f}")
        print(f"  > FINAL_STEALTH_BONUS: {final_bonus.iloc[-1]:.4f}")
        
        _temp_debug_values["stealth_bonus"] = final_bonus
        return final_bonus

    def _calculate_deception_filter(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], params_dict: Dict, method_name: str, _temp_debug_values: Dict) -> pd.Series:
        """
        【V10.0 · 事件视界反欺骗核心】
        逻辑公式：Filter = 1 - (空转 + 断裂 + 脉冲 + 散户陷阱 + 派发确信 + 锁仓崩塌)
        - 散户陷阱：散户买入加速 (Accel > 0) -> 欺骗。
        - 派发确信：Confidence > 60 -> 欺骗。
        - 锁仓崩塌：Lock Ratio Slope < 0 -> 欺骗。
        - 版本号：V10.0.0
        """
        # --- 1. V9.9 基础欺骗逻辑 (空转 + 断裂 + 脉冲) ---
        # 量价空转
        ab_v, ab_h, ab_m = raw_signals['tick_abnormal_volume_ratio_D'], raw_signals['HAB_LONG_tick_abnormal_volume_ratio_D'], raw_signals['HAB_MAD_tick_abnormal_volume_ratio_D']
        abn_z = np.tanh((ab_v - ab_h) / (ab_m * 1.618 + 0.1))
        vpa_eff = raw_signals['VPA_EFFICIENCY_D']
        trans_eff = np.tanh(raw_signals['tick_chip_transfer_efficiency_D'] / 8e6)
        churning_score = (abn_z.clip(0) * (1.0 - vpa_eff * 0.6 - trans_eff * 0.4).clip(0)).clip(0, 1)
        # 价流断裂
        div_val = raw_signals['price_flow_divergence_D']
        div_slope = raw_signals['SLOPE_5_price_flow_divergence_D']
        div_accel = raw_signals['ACCEL_5_price_flow_divergence_D']
        fracture_risk = (np.tanh(div_val) * 0.5 + np.tanh(div_slope + div_accel).clip(0) * 0.5).clip(0, 1)
        # 脉冲诱多
        abn_jerk = raw_signals['JERK_5_tick_abnormal_volume_ratio_D']
        abn_jerk_mad = raw_signals['HAB_MAD_JERK_5_tick_abnormal_volume_ratio_D']
        pulse_score = np.tanh(abn_jerk / (abn_jerk_mad * 2.0 + 1e-6)).clip(0, 1)
        
        # --- 2. 散户陷阱 (Retail Trap) [新增] ---
        # 逻辑：散户买入占比(SM Rate) 如果在加速上升 (Accel > 0)，说明散户在疯狂接盘
        sm_rate_accel = raw_signals['ACCEL_5_buy_sm_amount_rate_D']
        sm_mad = raw_signals['HAB_MAD_ACCEL_5_buy_sm_amount_rate_D']
        # 仅当加速为正时计算风险
        retail_trap_score = np.tanh(sm_rate_accel.clip(0) / (sm_mad * 2.0 + 1e-6)).clip(0, 1)
        
        # --- 3. 派发确信 (Distribution Certainty) [新增] ---
        # 逻辑：直接使用微观派发置信度
        dist_conf = raw_signals['intraday_distribution_confidence_D']
        distribution_penalty = np.tanh(dist_conf / 60.0).clip(0, 1) # 60分以上风险迅速饱和
        
        # --- 4. 锁仓崩塌 (Locking Fracture) [新增] ---
        # 逻辑：高位锁仓率(Lock90) 如果在下降 (Slope < 0)，说明底仓松动
        lock_slope = raw_signals['SLOPE_5_high_position_lock_ratio_90_D']
        # 斜率越负，风险越大
        locking_fracture = (-np.tanh(lock_slope * 10.0)).clip(0, 1)
        
        # --- 5. 综合计算 ---
        # 基础欺骗分
        base_deception = (churning_score * 0.25 + fracture_risk * 0.25 + pulse_score * 0.2)
        # 新增维度欺骗分
        advanced_deception = (retail_trap_score * 0.2 + distribution_penalty * 0.25 + locking_fracture * 0.15)
        
        total_deception = (base_deception + advanced_deception).clip(0, 1)
        
        # 稳定性修正
        chip_stab = raw_signals['chip_stability_D']
        stability_penalty = (1.0 - chip_stab).clip(0, 1)
        boosted_deception = (total_deception * (1.0 + stability_penalty * 0.5)).clip(0, 1)
        
        # 输出过滤器系数 (1.0 = 无欺骗)
        filter_score = 1.0 - boosted_deception
        
        # 6. 全息探针
        print(f"\n[V10.0_DECEPTION_EVENT_HORIZON_PROBE]")
        print(f"  > [BASE] Churn: {churning_score.iloc[-1]:.2f} | Frac: {fracture_risk.iloc[-1]:.2f} | Pulse: {pulse_score.iloc[-1]:.2f}")
        print(f"  > [RETAIL] SMAccel: {sm_rate_accel.iloc[-1]:.2e} -> Trap: {retail_trap_score.iloc[-1]:.4f}")
        print(f"  > [DIST] Conf: {dist_conf.iloc[-1]:.1f} -> Penalty: {distribution_penalty.iloc[-1]:.4f}")
        print(f"  > [LOCK] Slope: {lock_slope.iloc[-1]:.4f} -> Fracture: {locking_fracture.iloc[-1]:.4f}")
        print(f"  > FINAL_DECEPTION_SCORE: {boosted_deception.iloc[-1]:.4f} -> FILTER: {filter_score.iloc[-1]:.4f}")
        
        _temp_debug_values["deception_analysis"] = {"score": boosted_deception, "filter": filter_score}
        return filter_score

    def _calculate_contextual_modulator(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], params_dict: Dict, method_name: str, _temp_debug_values: Dict, is_debug_enabled: bool, probe_ts: pd.Timestamp) -> pd.Series:
        """
        【V10.2 · 合力竞速核心】
        逻辑公式：Context = (基础环境 + 协同增益 + 竞速优势) * 稳态系数
        - 协同增益：Smart Money Synergy > 0 -> 1.2倍暴击。
        - 竞速优势：Rank Accel > 0 -> 加分；Rank Accel < 0 -> 减分。
        - 稳态系数：Stability 高 -> 1.0；Stability 低 -> 0.8。
        - 版本号：V10.2.0
        """
        # --- 1. 基础环境 (V10.0逻辑保留) ---
        # 题材
        theme_val = raw_signals['THEME_HOTNESS_SCORE_D']
        theme_slope = raw_signals['SLOPE_5_THEME_HOTNESS_SCORE_D']
        raw_theme_score = (np.tanh(theme_val / 50.0) * 0.4 + np.tanh(theme_slope).clip(-1, 1) * 0.4)
        # 情绪
        sent_val = raw_signals['market_sentiment_score_D']
        sent_jerk = raw_signals['JERK_5_market_sentiment_score_D']
        sent_mad = raw_signals['HAB_MAD_JERK_5_market_sentiment_score_D']
        stall_risk = 0.0
        if sent_val.iloc[-1] > 60:
             stall_risk = (-np.tanh(sent_jerk / (sent_mad * 2.0 + 1e-6))).clip(0, 1)
        accum_sent = raw_signals['HAB_ACCUM_21_market_sentiment_score_D']
        exhaustion_risk = np.tanh((accum_sent - 1600) / 200).clip(0, 1)
        sentiment_score = (np.tanh(sent_val / 50.0) * (1.0 - stall_risk * 0.7) * (1.0 - exhaustion_risk * 0.5)).clip(0, 1)
        # 结构张力
        profit = raw_signals['profit_pressure_D']
        uptrend = raw_signals['uptrend_strength_D']
        structure_score = (1.0 - np.tanh((profit / (uptrend + 1e-4)).clip(0, 100) / 50.0)).clip(0, 1)
        
        base_context = (raw_theme_score * 0.4 + sentiment_score * 0.4 + structure_score * 0.2).clip(0, 1)
        
        # --- 2. 协同增益 (Synergy Bonus) [新增] ---
        # 逻辑：如果协同指标为正，说明机构游资在合力做多，环境分给予非线性加成
        synergy_val = raw_signals['SMART_MONEY_SYNERGY_BUY_D']
        # 协同加成：synergy > 0 时，最高给予 0.3 的绝对加分
        synergy_bonus = np.tanh(synergy_val.clip(0) * 2.0) * 0.3
        
        # --- 3. 竞速优势 (Rank Velocity) [新增] ---
        # 逻辑：行业排名加速度，衡量板块的“超车”能力
        rank_accel = raw_signals['industry_rank_accel_D']
        # 竞速修正：加速向上(>0)奖励，加速向下(<0)惩罚
        # 映射到 -0.15 ~ +0.15
        velocity_mod = np.tanh(rank_accel * 5.0) * 0.15
        
        # --- 4. 稳态过滤 (Stability Filter) [新增] ---
        # 逻辑：换手稳定性越高，环境越可靠
        stability = raw_signals['TURNOVER_STABILITY_INDEX_D']
        # 稳态系数：0.8 ~ 1.1 (极度稳定给予 1.1 倍奖励)
        stability_coef = 0.8 + (stability.clip(0, 1) * 0.3)
        
        # --- 5. 综合计算 ---
        # Raw = 基础 + 协同 + 竞速
        raw_modulator = base_context + synergy_bonus + velocity_mod
        # Apply Stability & Leader Immunity
        leader_score = raw_signals['industry_leader_score_D']
        immunity = 0.8 + np.tanh(leader_score / 40.0) * 0.7
        
        final_modulator = (raw_modulator * stability_coef * immunity).clip(0, 1)
        
        # 6. 全息探针
        print(f"\n[V10.2_SYNERGY_VELOCITY_PROBE]")
        print(f"  > [BASE] Theme:{raw_theme_score.iloc[-1]:.2f} | Sent:{sentiment_score.iloc[-1]:.2f} | Struct:{structure_score.iloc[-1]:.2f}")
        print(f"  > [SYNERGY] Val:{synergy_val.iloc[-1]:.4f} -> Bonus:{synergy_bonus.iloc[-1]:.4f} (Force Multiplier)")
        print(f"  > [VELOCITY] RankAccel:{rank_accel.iloc[-1]:.4f} -> Mod:{velocity_mod.iloc[-1]:.4f}")
        print(f"  > [STABILITY] StabIdx:{stability.iloc[-1]:.4f} -> Coef:{stability_coef.iloc[-1]:.4f}")
        print(f"  > FINAL_CONTEXT_MODULATOR: {final_modulator.iloc[-1]:.4f}")
        
        _temp_debug_values["context_analysis"] = {"modulator": final_modulator, "synergy_bonus": synergy_bonus}
        return final_modulator

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
        【V12.1 · 气动失速核心】修复探针打印格式错误
        - 修改思路：
            1. 修复在 print f-string 中直接格式化 Series (thrust_failure 等) 导致的 TypeError。
            2. 确保所有中间变量在输出时均使用 .iloc[-1] 转为标量。
        - 版本号：V12.1.0
        """
        # --- A. 推力熄火 (Thrust Failure) ---
        # 主力活跃度的 Jerk。如果瞬间变为极大的负值，说明主力“拔插头”了。
        act_jerk = raw_signals['JERK_5_main_force_activity_index_D']
        act_mad = raw_signals['HAB_MAD_JERK_5_main_force_activity_index_D']
        # 归一化：只关注负向突变
        thrust_failure = (-np.tanh(act_jerk / (act_mad * 2.0 + 1e-6))).clip(0, 1)
        
        # --- B. 阻力激增 (Drag Surge) ---
        # 流向阻力的 Accel。如果阻力在加速变大，说明上方抛压正在非线性增长。
        res_accel = raw_signals['ACCEL_5_flow_resistance_level_D']
        res_mad = raw_signals['HAB_MAD_ACCEL_5_flow_resistance_level_D']
        drag_surge = np.tanh(res_accel / (res_mad * 1.5 + 1e-6)).clip(0, 1)
        
        # --- C. 升力崩塌 (Lift Collapse) ---
        # 冲击比率的 Slope。如果斜率为负，说明同样的资金带来的价格涨幅在变小（效率降低）。
        eff_slope = raw_signals['SLOPE_5_flow_impact_ratio_D']
        # 效率越低，Risk 放大系数越大 (1.0 ~ 2.0)
        lift_loss_factor = 1.0 + (-np.tanh(eff_slope * 2.0)).clip(0)
        
        # --- D. 惯性与意愿 (Mass & Intent - V11.5保留) ---
        # 质量惯性
        inst_accum = raw_signals['HAB_ACCUM_21_SMART_MONEY_INST_NET_BUY_D']
        inst_std = raw_signals['HAB_STD_SMART_MONEY_INST_NET_BUY_D'] * np.sqrt(21)
        mass_inertia = (1 / (1 + np.exp(-inst_accum / (inst_std + 1e-6) * 1.5))).clip(0.2, 1.0)
        # 意愿修正
        intent_slope = raw_signals['SLOPE_5_INTRADAY_SUPPORT_INTENT_D']
        intent_risk_factor = 1.0 + (-np.tanh(intent_slope)).clip(0)
        
        # --- E. 综合计算 ---
        # 核心气动方程
        aerodynamic_stress = (thrust_failure * 0.5 + drag_surge * 0.5) * lift_loss_factor
        
        # 结合质量与意愿
        raw_risk = (aerodynamic_stress / mass_inertia) * intent_risk_factor
        
        # 价格修正 (Price Context - 只有在高位或上涨中失速才危险)
        och_accel = raw_signals['OCH_ACCELERATION_D']
        price_context = np.tanh((och_accel + 0.01) * 20).clip(0, 1)
        
        final_risk = np.tanh(raw_risk * price_context * 1.5).clip(0, 1)
        
        # F. 全息探针 (修复：增加 .iloc[-1])
        print(f"\n[V12.1_AERODYNAMIC_STALL_PROBE]")
        print(f"  > [THRUST] ActJerk:{act_jerk.iloc[-1]:.2e} -> Failure:{thrust_failure.iloc[-1]:.4f}")
        print(f"  > [DRAG] ResAccel:{res_accel.iloc[-1]:.2e} -> Surge:{drag_surge.iloc[-1]:.4f}")
        print(f"  > [LIFT] EffSlope:{eff_slope.iloc[-1]:.4f} -> LossFactor:{lift_loss_factor.iloc[-1]:.2f}")
        print(f"  > [INERTIA] Mass:{mass_inertia.iloc[-1]:.4f} | IntentRisk:{intent_risk_factor.iloc[-1]:.2f}")
        print(f"  > FINAL_STALL_RISK: {final_risk.iloc[-1]:.4f}")
        
        _temp_debug_values["cross_module_signals"]["stalling_risk"] = final_risk
        return final_risk

    def _calculate_institutional_erosion_index(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V13.1 · 结构风化侵蚀核心】修复探针打印格式错误
        - 修改思路：
            1. 修复在 print f-string 中直接格式化 Series (landslide_shock 等) 导致的 TypeError。
            2. 确保所有中间变量在输出时均使用 .iloc[-1] 转为标量。
        - 版本号：V13.1.0
        """
        # --- A. 滑坡与裂隙 (保留 V12.8) ---
        mig_jerk = raw_signals['JERK_5_intraday_cost_center_migration_D']
        mig_mad = raw_signals['HAB_MAD_JERK_5_intraday_cost_center_migration_D']
        landslide_shock = (-np.tanh(mig_jerk / (mig_mad * 1.618 + 1e-6))).clip(0, 1)
        
        vol_accel = raw_signals['ACCEL_5_intraday_cost_center_volatility_D']
        vol_mad = raw_signals['HAB_MAD_ACCEL_5_intraday_cost_center_volatility_D']
        fissure_spread = np.tanh(vol_accel / (vol_mad * 1.5 + 1e-6)).clip(0, 1)
        
        # --- B. 峰度坍塌 (Morphological Weathering) [新增] ---
        # 筹码峰度的 Slope。如果 Slope < 0，说明峰度在降低，筹码在发散。
        kurt_slope = raw_signals['SLOPE_5_chip_kurtosis_D']
        kurt_mad = raw_signals['HAB_MAD_SLOPE_5_chip_kurtosis_D']
        # 归一化：负斜率越大，风险越大
        kurtosis_collapse = (-np.tanh(kurt_slope / (kurt_mad * 2.0 + 1e-6))).clip(0, 1)
        
        # --- C. 摩擦生热 (Friction Heat) [新增] ---
        # 日内博弈指数。计算 Z-Score，衡量当前的博弈强度是否异常。
        game_val = raw_signals['intraday_chip_game_index_D']
        game_mean = raw_signals['HAB_LONG_intraday_chip_game_index_D']
        game_std = raw_signals['HAB_STD_intraday_chip_game_index_D']
        # 摩擦系数：至少 1.0，如果博弈激烈则放大侵蚀效果
        friction_coef = 1.0 + np.tanh(((game_val - game_mean) / (game_std + 1e-4)).clip(0) * 0.5)
        
        # --- D. 沉积缓冲 (Sediment Buffer) ---
        sed_accum = raw_signals['HAB_ACCUM_21_intraday_chip_consolidation_degree_D']
        sediment_thickness = (1 / (1 + np.exp(-(sed_accum - 1000) / 300))).clip(0.1, 1.2)
        
        # --- E. 综合计算 ---
        # 物理意义：(位移破坏 + 结构破坏 + 形态破坏) * 能量强度 / 抵抗力
        
        structural_damage = (landslide_shock * 0.35 + fissure_spread * 0.35 + kurtosis_collapse * 0.3)
        
        raw_erosion = (structural_damage * friction_coef) / sediment_thickness
        
        final_erosion = np.tanh(raw_erosion).clip(0, 1)
        
        # F. 全息探针 (修复：增加 .iloc[-1])
        print(f"\n[V13.1_STRUCTURAL_WEATHERING_PROBE]")
        print(f"  > [DAMAGE] Slide:{landslide_shock.iloc[-1]:.2f} | Fissure:{fissure_spread.iloc[-1]:.2f} | Kurtosis:{kurtosis_collapse.iloc[-1]:.2f}")
        print(f"  > [FRICTION] GameIdx:{game_val.iloc[-1]:.2f} -> Coef:{friction_coef.iloc[-1]:.2f}")
        print(f"  > [BUFFER] Thickness:{sediment_thickness.iloc[-1]:.4f}")
        print(f"  > FINAL_EROSION_INDEX: {final_erosion.iloc[-1]:.4f}")
        
        _temp_debug_values["cross_module_signals"]["erosion_index"] = final_erosion
        return final_erosion

    def _calculate_chaotic_collapse_resonance(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V15.1 · 几何奇点混沌核心】修复探针打印格式错误
        - 修改思路：
            1. 修复在 print f-string 中直接格式化 Series (thermo_chaos 等) 导致的 TypeError。
            2. 确保所有中间变量在输出时均使用 .iloc[-1] 转为标量。
        - 版本号：V15.1.0
        """
        # --- A. 热力熵 (Thermodynamic Entropy - V14.5 保留) ---
        p_ent_jerk = raw_signals['JERK_5_PRICE_ENTROPY_D']
        p_ent_mad = raw_signals['HAB_MAD_JERK_5_PRICE_ENTROPY_D']
        signal_shock = np.tanh(p_ent_jerk / (p_ent_mad * 1.618 + 1e-6)).clip(0, 1)
        
        c_ent_accel = raw_signals['ACCEL_5_concentration_entropy_D']
        c_ent_mad = raw_signals['HAB_MAD_ACCEL_5_concentration_entropy_D']
        structure_decay = np.tanh(c_ent_accel / (c_ent_mad * 1.5 + 1e-6)).clip(0, 1)
        
        p_ent_accum = raw_signals['HAB_ACCUM_21_PRICE_ENTROPY_D']
        criticality = (1 / (1 + np.exp(-(p_ent_accum - 15.0) / 3.0))).clip(0.5, 1.5)
        
        en_conc = raw_signals['energy_concentration_D']
        energy_amp = 1.0 + np.tanh(en_conc / 50.0).clip(0)
        
        div_jerk = raw_signals['JERK_5_high_freq_flow_divergence_D']
        div_mad = raw_signals['HAB_MAD_JERK_5_high_freq_flow_divergence_D']
        micro_trigger = np.tanh(div_jerk / (div_mad * 2.0 + 1e-6)).clip(0, 1)
        
        thermo_chaos = (signal_shock * 0.4 + structure_decay * 0.6) * criticality * energy_amp * (1.0 + micro_trigger)
        
        # --- B. 几何曲率 (Geometric Curvature) [新增] ---
        # 监测"几何奇点"。曲率 Jerk 极大，说明走势从平滑突然折断。
        curv_jerk = raw_signals['JERK_5_GEOM_ARC_CURVATURE_D']
        curv_mad = raw_signals['HAB_MAD_JERK_5_GEOM_ARC_CURVATURE_D']
        # 归一化：关注正向曲率突变
        curvature_tear = np.tanh(curv_jerk / (curv_mad * 2.0 + 1e-6)).clip(0, 1)
        
        # --- C. 吸引子耗散 (Attractor Dissipation) [新增] ---
        # 资金聚类强度。如果强度在快速衰减 (Slope < 0)，说明系统失去了"主心骨"。
        clust_slope = raw_signals['SLOPE_5_flow_cluster_intensity_D']
        clust_mad = raw_signals['HAB_MAD_SLOPE_5_flow_cluster_intensity_D']
        # 耗散系数：1.0 ~ 2.0
        dissipation_factor = 1.0 + (-np.tanh(clust_slope / (clust_mad + 1e-6))).clip(0)
        
        # --- D. 先验校准 (Systemic Prior) [新增] ---
        # 反转预警分 (假设 0-100)
        rev_score = raw_signals['reversal_warning_score_D']
        prior_prob = np.tanh(rev_score / 60.0).clip(0.5, 1.5)
        
        # --- E. 综合共振 ---
        # 逻辑：(热力学混乱 + 几何断裂) * 拓扑结构瓦解 * 系统先验
        raw_resonance = (thermo_chaos + curvature_tear * 0.5) * dissipation_factor * prior_prob
        
        final_score = np.tanh(raw_resonance * 0.8).clip(0, 1)
        
        # F. 全息探针 (修复：增加 .iloc[-1])
        print(f"\n[V15.1_GEOMETRIC_SINGULARITY_PROBE]")
        print(f"  > [THERMO] Chaos: {thermo_chaos.iloc[-1]:.4f} (Sig:{signal_shock.iloc[-1]:.2f}, Struc:{structure_decay.iloc[-1]:.2f}, Crit:{criticality.iloc[-1]:.2f})")
        print(f"  > [GEOM] CurvJerk: {curv_jerk.iloc[-1]:.2e} -> Tear: {curvature_tear.iloc[-1]:.4f}")
        print(f"  > [TOPO] ClustSlope: {clust_slope.iloc[-1]:.2e} -> Dissipation: {dissipation_factor.iloc[-1]:.2f}")
        print(f"  > [PRIOR] RevScore: {rev_score.iloc[-1]:.1f} -> Prob: {prior_prob.iloc[-1]:.2f}")
        print(f"  > FINAL_CHAOS_RESONANCE: {final_score.iloc[-1]:.4f}")
        
        _temp_debug_values["cross_module_signals"]["chaos_resonance"] = final_score
        return final_score

    def _calculate_macro_sector_synergy(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V16.8.1 · 维度共振协同核心】修复探针打印格式错误
        - 修改思路：
            1. 修复 breakout_force, cohesion_force, sync_factor 等 Series 变量在打印时未转为标量的 TypeError。
            2. 确保所有中间变量在输出时均使用 .iloc[-1] 转为标量。
        - 版本号：V16.8.1
        """
        # --- A. 基础爆发 (Base Explosion - V16.5 保留) ---
        stag_jerk = raw_signals['JERK_5_industry_stagnation_score_D']
        stag_mad = raw_signals['HAB_MAD_JERK_5_industry_stagnation_score_D']
        breakout_force = (-np.tanh(stag_jerk / (stag_mad * 1.5 + 1e-6))).clip(0, 1)
        
        clust_accel = raw_signals['ACCEL_5_flow_cluster_intensity_D']
        clust_mad = raw_signals['HAB_MAD_ACCEL_5_flow_cluster_intensity_D']
        cohesion_force = np.tanh(clust_accel / (clust_mad * 1.5 + 1e-6)).clip(0, 1)
        
        rank_slope = raw_signals['industry_rank_slope_D']
        velocity_score = np.tanh(-rank_slope).clip(0, 1)
        
        energy_accum = raw_signals['HAB_ACCUM_21_SMART_MONEY_HM_COORDINATED_ATTACK_D']
        potential_factor = (1 / (1 + np.exp(-(energy_accum - 10.0) / 3.0))).clip(0.5, 1.5)
        
        base_synergy = (breakout_force * 0.35 + cohesion_force * 0.35 + velocity_score * 0.3) * potential_factor
        
        # --- B. 时空共振 (Timeframe Sync) [新增] ---
        # 日月同步性：正值代表共振，负值代表背离。
        sync_val = raw_signals['daily_monthly_sync_D']
        # 如果共振度极高，给予奖励；如果背离，给予惩罚
        sync_factor = np.tanh(sync_val / 20.0).clip(0.5, 1.3) # 0.5 ~ 1.3
        
        # --- C. 主升浪加速 (Markup Acceleration) [新增] ---
        # 行业拉升分。如果正在加速进入拉升期 (Accel > 0)，这是最肥美的鱼身。
        markup_accel = raw_signals['ACCEL_5_industry_markup_score_D']
        markup_mad = raw_signals['HAB_MAD_ACCEL_5_industry_markup_score_D']
        # 仅奖励正向加速
        markup_boost = 1.0 + np.tanh(markup_accel.clip(0) / (markup_mad + 1e-6)).clip(0) * 0.5 # 1.0 ~ 1.5
        
        # --- D. 结构性否决 (Structural Veto) [新增] ---
        # 行业下降分。如果分数过高 (>60)，说明板块处于结构性熊市，协同不可信。
        downtrend_score = raw_signals['industry_downtrend_score_D']
        # 否决系数：分数越高，系数越接近 0
        veto_factor = (1.0 - np.tanh(downtrend_score / 40.0)).clip(0, 1)
        
        # --- E. 综合计算 ---
        # 物理意义：(点火 + 燃料) * 大势 * 阶段 * 滤网
        
        raw_synergy = base_synergy * sync_factor * markup_boost * veto_factor
        
        final_synergy = np.tanh(raw_synergy * 1.2).clip(0, 1)
        
        # F. 全息探针 (修复：增加 .iloc[-1])
        print(f"\n[V16.8.1_DIMENSIONAL_RESONANCE_PROBE]")
        print(f"  > [BASE] Breakout:{breakout_force.iloc[-1]:.2f} | Cohesion:{cohesion_force.iloc[-1]:.2f} | Pot:{potential_factor.iloc[-1]:.2f}")
        print(f"  > [CHRONOS] SyncVal:{sync_val.iloc[-1]:.2f} -> Factor:{sync_factor.iloc[-1]:.2f}")
        print(f"  > [KAIROS] MarkupAccel:{markup_accel.iloc[-1]:.2f} -> Boost:{markup_boost.iloc[-1]:.2f}")
        print(f"  > [VETO] DownScore:{downtrend_score.iloc[-1]:.1f} -> Factor:{veto_factor.iloc[-1]:.2f}")
        print(f"  > FINAL_SYNERGY_SCORE: {final_synergy.iloc[-1]:.4f}")
        
        _temp_debug_values["cross_module_signals"]["macro_synergy"] = final_synergy
        return final_synergy

    def _calculate_chain_collapse_resonance(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V18.1 · 多米诺奇点链式核心】修复探针打印格式错误
        - 修改思路：
            1. 修复在 print f-string 中直接格式化 Series (physical_collapse, freeze_factor 等) 导致的 TypeError。
            2. 确保 freeze_factor 始终初始化为 Series，避免类型不一致导致的打印错误。
            3. 确保所有探针输出均使用 .iloc[-1] 获取最新标量值。
        - 版本号：V18.1.1
        """
        # --- A. 物理崩塌 (Physical Collapse - V17.5 保留) ---
        lock_jerk = raw_signals['JERK_5_intraday_high_lock_ratio_D']
        lock_mad = raw_signals['HAB_MAD_JERK_5_intraday_high_lock_ratio_D']
        fracture_shock = (-np.tanh(lock_jerk / (lock_mad * 1.618 + 1e-6))).clip(0, 1)
        
        leader_slope = raw_signals['SLOPE_5_industry_leader_score_D']
        leader_snap = (-np.tanh(leader_slope * 2.0)).clip(0, 1)
        
        trapped_accum = raw_signals['HAB_ACCUM_21_pressure_trapped_D']
        trapped_accel = raw_signals['ACCEL_5_pressure_trapped_D']
        gravity_base = (1 / (1 + np.exp(-(trapped_accum - 500.0) / 100.0))).clip(0.5, 1.5)
        gravity_dynamic = 1.0 + np.tanh(trapped_accel.clip(0) * 2.0)
        
        kurt_val = raw_signals['high_freq_flow_kurtosis_D']
        kurt_mean = raw_signals['HAB_LONG_high_freq_flow_kurtosis_D']
        kurt_std = raw_signals['HAB_STD_high_freq_flow_kurtosis_D']
        kurt_z = ((kurt_val - kurt_mean) / (kurt_std + 1e-4)).clip(0)
        tail_risk = 1.0 + np.tanh(kurt_z * 0.5)
        
        limit_accel = raw_signals['ACCEL_5_down_limit_pct_D']
        limit_mad = raw_signals['HAB_MAD_ACCEL_5_down_limit_pct_D']
        panic_spread = np.tanh(limit_accel.clip(0) / (limit_mad * 1.5 + 1e-6)).clip(0, 1)
        
        price_slope = raw_signals['SLOPE_5_OCH_D']
        turn_jerk = raw_signals['JERK_5_turnover_rate_D']
        turn_mad = raw_signals['HAB_MAD_JERK_5_turnover_rate_D']
        
        # 修复：初始化为 Series 确保类型统一
        freeze_factor = pd.Series(1.0, index=df_index)
        if price_slope.iloc[-1] < 0 and turn_jerk.iloc[-1] < 0:
            freeze_severity = (-np.tanh(turn_jerk / (turn_mad * 2.0 + 1e-6))).clip(0)
            freeze_factor = 1.0 + freeze_severity
            
        physical_collapse = (fracture_shock * 0.6 + leader_snap * 0.4) * gravity_base * gravity_dynamic * tail_risk * (1.0 + panic_spread) * freeze_factor
        
        # --- B. 派发意图 (Distribution Intent) [新增] ---
        # 派发行为分的 Accel。如果加速，说明主力去意已决。
        dist_accel = raw_signals['ACCEL_5_behavior_distribution_D']
        dist_mad = raw_signals['HAB_MAD_ACCEL_5_behavior_distribution_D']
        # 归一化：正向加速
        intent_factor = np.tanh(dist_accel.clip(0) / (dist_mad + 1e-6)).clip(0, 1)
        
        # HAB 校验：如果长期处于高派发状态(Accum high)，风险更高
        dist_accum = raw_signals['HAB_ACCUM_21_behavior_distribution_D']
        accum_risk = (1 / (1 + np.exp(-(dist_accum - 1000.0) / 200.0))).clip(0.5, 1.2)
        
        distribution_risk = 1.0 + (intent_factor * accum_risk)
        
        # --- C. 反转临界 (Reversal Criticality) [新增] ---
        # 反转概率 (0-1)。
        rev_prob = raw_signals['reversal_prob_D']
        # 临界系数：只有当概率 > 0.7 时开始生效，> 0.9 时极度危险
        # 映射到 0.8 ~ 1.5
        criticality = 0.8 + np.tanh((rev_prob - 0.7) * 5.0).clip(0) * 0.7
        
        # --- D. 综合计算 ---
        # 物理意义：物理崩塌 * 主观恶意 * 数学必然
        
        raw_resonance = physical_collapse * distribution_risk * criticality
        
        final_collapse = np.tanh(raw_resonance).clip(0, 1)
        
        # E. 全息探针 (修复：增加 .iloc[-1])
        print(f"\n[V18.1_DOMINO_SINGULARITY_PROBE]")
        print(f"  > [PHYSICAL] Base:{physical_collapse.iloc[-1]:.4f} (Panic:{panic_spread.iloc[-1]:.2f}, Freeze:{freeze_factor.iloc[-1]:.2f})")
        print(f"  > [INTENT] DistAccel:{dist_accel.iloc[-1]:.2e} | Accum:{dist_accum.iloc[-1]:.0f} -> Risk:{distribution_risk.iloc[-1]:.2f}")
        print(f"  > [CRITICAL] RevProb:{rev_prob.iloc[-1]:.2f} -> Coef:{criticality.iloc[-1]:.2f}")
        print(f"  > FINAL_CHAIN_COLLAPSE: {final_collapse.iloc[-1]:.4f}")
        
        _temp_debug_values["cross_module_signals"]["chain_collapse"] = final_collapse
        return final_collapse














