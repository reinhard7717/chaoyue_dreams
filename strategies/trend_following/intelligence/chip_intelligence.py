import pandas as pd
import numpy as np
import numba as nb
from typing import Dict, Tuple, Any, Union
from strategies.trend_following import utils
from strategies.trend_following.utils import (
    get_params_block, get_param_value, get_adaptive_mtf_normalized_score, load_external_json_config,
    get_adaptive_mtf_normalized_bipolar_score, _robust_geometric_mean, normalize_score
)

@nb.njit(cache=True)
def _numba_calculate_deception_modulator_core(
    norm_deception_index: np.ndarray,
    norm_deception_lure_long: np.ndarray,
    norm_deception_lure_short: np.ndarray,
    norm_wash_trade_intensity: np.ndarray,
    norm_main_force_conviction: np.ndarray,
    norm_chip_health: np.ndarray,
    deception_conviction_threshold: float,
    deception_health_threshold: float,
    deception_boost_factor: float,
    deception_penalty_factor: float,
    wash_trade_penalty_factor: float,
    deception_lure_long_penalty_factor: float,
    deception_lure_short_boost_factor: float
) -> np.ndarray:
    """
    Numba优化后的核心函数，用于计算deception_modulator。
    直接操作NumPy数组，避免Pandas Series的内部开销。
    """
    n = len(norm_deception_index)
    deception_modulator = np.ones(n, dtype=np.float32)
    for i in range(n):
        strong_conviction_healthy_chip_mask = (norm_main_force_conviction[i] > deception_conviction_threshold) and \
                                              (norm_chip_health[i] > deception_health_threshold)
        weak_conviction_unhealthy_chip_mask = (norm_main_force_conviction[i] < -deception_conviction_threshold) or \
                                               (norm_chip_health[i] < (1 - deception_health_threshold))
        # Bear trap boost (诱空反吸增强)
        if strong_conviction_healthy_chip_mask and ((norm_deception_index[i] < 0) or (norm_deception_lure_short[i] > 0)):
            deception_modulator[i] *= (1 + (np.abs(norm_deception_index[i]) * deception_boost_factor + \
                                            norm_deception_lure_short[i] * deception_lure_short_boost_factor))
        # Bull trap penalty (诱多惩罚)
        if (norm_deception_index[i] > 0) or (norm_deception_lure_long[i] > 0):
            deception_modulator[i] *= (1 - (np.maximum(0.0, norm_deception_index[i]) * deception_penalty_factor + \
                                            norm_deception_lure_long[i] * deception_lure_long_penalty_factor))
        # Wash trade penalty (对倒惩罚)
        wash_trade_penalty_mod = norm_wash_trade_intensity[i] * wash_trade_penalty_factor
        if strong_conviction_healthy_chip_mask:
            deception_modulator[i] *= (1 - wash_trade_penalty_mod * 0.5) # 主力信念强且筹码健康时，对倒惩罚减半
        elif weak_conviction_unhealthy_chip_mask:
            deception_modulator[i] *= (1 - wash_trade_penalty_mod * 1.5) # 主力信念弱或筹码不健康时，对倒惩罚加倍
        else:
            deception_modulator[i] *= (1 - wash_trade_penalty_mod) # 其他情况正常惩罚
    return np.clip(deception_modulator, 0.1, 2.0) # 限制调制范围

@nb.njit(cache=True)
def _numba_calculate_synergy_factor_core(
    formation_deployment_score: np.ndarray,
    commanders_resolve_score: np.ndarray,
    battlefield_control_score: np.ndarray,
    synergy_bonus_factor: float,
    conflict_penalty_factor: float
) -> np.ndarray:
    """
    Numba优化后的核心函数，用于计算synergy_factor。
    直接操作NumPy数组，避免Pandas Series的内部开销。
    """
    n = len(formation_deployment_score)
    synergy_factor = np.zeros(n, dtype=np.float32)
    for i in range(n):
        fd_sign = np.sign(formation_deployment_score[i])
        cr_sign = np.sign(commanders_resolve_score[i])
        bc_sign = np.sign(battlefield_control_score[i])
        # Positive synergy (正向协同)
        if fd_sign > 0 and cr_sign > 0 and bc_sign > 0:
            synergy_factor[i] = synergy_bonus_factor
        # Negative synergy (负向协同)
        elif fd_sign < 0 and cr_sign < 0 and bc_sign < 0:
            synergy_factor[i] = -synergy_bonus_factor
        # Conflict (冲突)
        elif ((fd_sign > 0 and cr_sign < 0) or (fd_sign < 0 and cr_sign > 0) or \
              (bc_sign > 0 and cr_sign < 0) or (bc_sign < 0 and cr_sign > 0) or \
              (fd_sign > 0 and bc_sign < 0) or (fd_sign < 0 and bc_sign > 0)):
            synergy_factor[i] = -conflict_penalty_factor
    return synergy_factor

@nb.njit(cache=True)
def _numba_calculate_deception_filter_factor_core(
    base_terrain_advantage_score: np.ndarray,
    norm_deception: np.ndarray,
    norm_chip_fault: np.ndarray,
    deception_penalty_sensitivity: float,
    chip_fault_penalty_sensitivity: float,
    deception_mitigation_sensitivity: float
) -> np.ndarray:
    """
    Numba优化后的核心函数，用于计算deception_filter_factor。
    直接操作NumPy数组，避免Pandas Series的内部开销。
    """
    n = len(base_terrain_advantage_score)
    deception_filter_factor = np.ones(n, dtype=np.float32)
    for i in range(n):
        # 牛市陷阱惩罚 (有利地形伴随诱多或虚假支撑时惩罚)
        if base_terrain_advantage_score[i] > 0 and \
           ((norm_deception[i] > 0) or (norm_chip_fault[i] > 0)):
            penalty_val = (np.maximum(0.0, norm_deception[i]) * deception_penalty_sensitivity + \
                           np.maximum(0.0, norm_chip_fault[i]) * chip_fault_penalty_sensitivity)
            # 修复：使用min/max组合进行标量裁剪
            deception_filter_factor[i] = 1 - max(0.0, min(penalty_val, 1.0))
        # 熊市陷阱缓解 (不利地形伴随诱空洗盘或虚假阻力时缓解)
        elif base_terrain_advantage_score[i] < 0 and \
             ((norm_deception[i] < 0) or (norm_chip_fault[i] < 0)):
            mitigation_val = (np.abs(np.minimum(0.0, norm_deception[i])) * deception_mitigation_sensitivity + \
                              np.abs(np.minimum(0.0, norm_chip_fault[i])) * deception_mitigation_sensitivity)
            # 修复：使用min/max组合进行标量裁剪
            deception_filter_factor[i] = 1 + max(0.0, min(mitigation_val, 0.5)) # 缓解上限0.5
    return np.clip(deception_filter_factor, 0.1, 2.0) # 限制范围

@nb.njit(cache=True)
def _numba_calculate_impurity_deception_modulator_core(
    conviction_base_unipolar: np.ndarray,
    norm_deception_index_bipolar: np.ndarray,
    norm_wash_trade_intensity: np.ndarray,
    norm_main_force_conviction_bipolar: np.ndarray,
    deception_modulator_weights_deception_index_boost: float,
    deception_modulator_weights_deception_index_penalty: float,
    deception_modulator_weights_wash_trade_penalty: float,
    conviction_threshold: float
) -> np.ndarray:
    """
    Numba优化后的核心函数，用于计算conviction_base_unipolar的欺骗调制。
    """
    n = len(conviction_base_unipolar)
    result_conviction_base_unipolar = conviction_base_unipolar.copy()
    for i in range(n):
        # 欺骗增强 (诱空反吸)
        if norm_deception_index_bipolar[i] < 0 and norm_main_force_conviction_bipolar[i] > conviction_threshold:
            result_conviction_base_unipolar[i] *= (1 + np.abs(norm_deception_index_bipolar[i]) * deception_modulator_weights_deception_index_boost)
        # 欺骗惩罚 (诱多派发)
        elif norm_deception_index_bipolar[i] > 0 and norm_main_force_conviction_bipolar[i] < -conviction_threshold:
            result_conviction_base_unipolar[i] *= (1 - norm_deception_index_bipolar[i] * deception_modulator_weights_deception_index_penalty)
        # 对倒惩罚
        result_conviction_base_unipolar[i] *= (1 - norm_wash_trade_intensity[i] * deception_modulator_weights_wash_trade_penalty)
    return np.clip(result_conviction_base_unipolar, 0.0, 1.0)

@nb.njit(cache=True)
def _numba_calculate_fuel_quality_modulators_core(
    base_fuel_quality: np.ndarray,
    norm_chip_fault: np.ndarray,
    norm_synergy_context: np.ndarray,
    chip_fault_raw: np.ndarray, # 原始chip_fault_raw用于判断positive_fault_mask
    fuel_purity_deception_penalty_factor: float,
    synergy_bonus_base: float,
    synergy_bonus_context_sensitivity: float,
    synergy_activation_threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba优化后的核心函数，用于计算fuel_quality_score中的deception_penalty和synergy_bonus。
    """
    n = len(base_fuel_quality)
    deception_penalty = np.zeros(n, dtype=np.float32)
    synergy_bonus = np.zeros(n, dtype=np.float32)
    fuel_quality_score_after_deception = np.zeros(n, dtype=np.float32)

    for i in range(n):
        positive_fault_mask = chip_fault_raw[i] > 0 # 筹码故障为正，视为负面影响 (诱多)
        # Deception Penalty
        if positive_fault_mask:
            deception_penalty[i] = norm_chip_fault[i] * fuel_purity_deception_penalty_factor * 4.0
        # 修复：使用min/max组合进行标量裁剪
        fuel_quality_score_after_deception[i] = base_fuel_quality[i] - max(0.0, min(deception_penalty[i], 1.0))
        # Synergy Bonus
        dynamic_synergy_bonus_factor = synergy_bonus_base * (1 + norm_synergy_context[i] * synergy_bonus_context_sensitivity)
        # 修复：使用min/max组合进行标量裁剪
        dynamic_synergy_bonus_factor = max(0.1, min(dynamic_synergy_bonus_factor, 0.5))
        # 当存在正向筹码故障（诱多）时，取消协同奖励
        if not positive_fault_mask: # 只有在没有诱多故障时才激活协同奖励
            synergy_potential = (base_fuel_quality[i] + 1) / 2 # 映射到 [0,1]
            synergy_activation = 1 / (1 + np.exp(-(synergy_potential - synergy_activation_threshold) * 10))
            synergy_bonus[i] = synergy_activation * dynamic_synergy_bonus_factor
            
    return fuel_quality_score_after_deception, synergy_bonus

@nb.njit(cache=True)
def _numba_calculate_divergence_deception_modulator_core(
    disagreement_vector_sign: np.ndarray,
    chip_fault_sign: np.ndarray,
    norm_chip_fault: np.ndarray,
    norm_deception_index: np.ndarray,
    norm_main_force_flow_directionality: np.ndarray,
    deception_modulator_impact_clip: float,
    deception_modulator_reinforce_factor: float,
    bearish_deception_penalty_factor: float
) -> np.ndarray:
    """
    Numba优化后的核心函数，用于计算_diagnose_axiom_divergence中的deception_modulator_factor。
    """
    n = len(disagreement_vector_sign)
    deception_modulator_factor = np.ones(n, dtype=np.float32)
    for i in range(n):
        bearish_deception_and_mf_out_mask = (norm_deception_index[i] < 0) and (norm_main_force_flow_directionality[i] < 0)
        if bearish_deception_and_mf_out_mask:
            deception_modulator_factor[i] *= (1 - np.abs(norm_deception_index[i]) * \
                                                  np.abs(norm_main_force_flow_directionality[i]) * \
                                                  bearish_deception_penalty_factor)
        else:
            # 筹码故障与分歧方向一致时惩罚
            if disagreement_vector_sign[i] == chip_fault_sign[i]:
                deception_modulator_factor[i] = 1 - norm_chip_fault[i] * deception_modulator_impact_clip
            # 筹码故障与分歧方向相反时增强
            elif disagreement_vector_sign[i] != chip_fault_sign[i]:
                deception_modulator_factor[i] = 1 + norm_chip_fault[i] * deception_modulator_reinforce_factor
    return np.clip(deception_modulator_factor, 0.01, 2.0)

@nb.njit(cache=True)
def _numba_calculate_absorption_echo_deception_modulator_core(
    net_conviction_flow_quality: np.ndarray,
    norm_deception_index_bipolar: np.ndarray,
    norm_wash_trade_intensity: np.ndarray,
    norm_chip_fault_magnitude_bipolar: np.ndarray,
    norm_supportive_distribution_intensity: np.ndarray,
    deception_boost_factor_negative: float,
    deception_index_penalty_weight: float,
    wash_trade_penalty_weight: float,
    chip_fault_penalty_factor: float,
    supportive_distribution_penalty_factor: float
) -> np.ndarray:
    """
    Numba优化后的核心函数，用于计算_diagnose_absorption_echo中的deception_modulator。
    """
    n = len(net_conviction_flow_quality)
    deception_modulator = np.ones(n, dtype=np.float32)
    for i in range(n):
        # 诱空反吸增强
        if net_conviction_flow_quality[i] > 0 and norm_deception_index_bipolar[i] < 0:
            deception_modulator[i] *= (1 + np.abs(norm_deception_index_bipolar[i]) * deception_boost_factor_negative)
        # 诱多惩罚
        elif net_conviction_flow_quality[i] > 0 and norm_deception_index_bipolar[i] > 0:
            deception_modulator[i] *= (1 - norm_deception_index_bipolar[i] * deception_index_penalty_weight)
        # 对倒惩罚
        deception_modulator[i] *= (1 - norm_wash_trade_intensity[i] * wash_trade_penalty_weight)
        # 筹码故障惩罚
        deception_modulator[i] *= (1 - np.maximum(0.0, norm_chip_fault_magnitude_bipolar[i]) * chip_fault_penalty_factor)
        # 支持性派发惩罚
        deception_modulator[i] *= (1 - norm_supportive_distribution_intensity[i] * supportive_distribution_penalty_factor)
    return np.clip(deception_modulator, 0.1, 2.0)

@nb.njit(cache=True)
def _numba_calculate_distribution_whisper_deception_modulator_core(
    norm_chip_fault_magnitude_bipolar: np.ndarray,
    norm_main_force_conviction_bipolar: np.ndarray,
    norm_deception_index_bipolar: np.ndarray,
    deception_modulator_params_boost_factor: float,
    deception_modulator_params_penalty_factor: float,
    deception_modulator_params_conviction_threshold: float,
    deception_modulator_params_deception_index_weight: float
) -> np.ndarray:
    """
    Numba优化后的核心函数，用于计算_diagnose_distribution_whisper中的deception_modulator。
    """
    n = len(norm_chip_fault_magnitude_bipolar)
    deception_modulator = np.ones(n, dtype=np.float32)

    for i in range(n):
        conviction_threshold = deception_modulator_params_conviction_threshold
        # 欺骗性看涨且主力信念弱 (诱多)
        deceptive_bullish_and_weak_conviction_mask = (norm_chip_fault_magnitude_bipolar[i] > 0) and \
                                                     (norm_main_force_conviction_bipolar[i] < -conviction_threshold)
        if deceptive_bullish_and_weak_conviction_mask:
            deception_modulator[i] *= (1 + norm_chip_fault_magnitude_bipolar[i] * deception_modulator_params_boost_factor)
        # 诱导恐慌且主力信念强 (诱空)
        induced_panic_and_conviction_mask = (norm_chip_fault_magnitude_bipolar[i] < 0) and \
                                            (norm_main_force_conviction_bipolar[i] > conviction_threshold)
        if induced_panic_and_conviction_mask:
            deception_modulator[i] *= (1 - np.abs(norm_chip_fault_magnitude_bipolar[i]) * deception_modulator_params_penalty_factor)
        # 欺骗指数增强 (正向欺骗且主力信念弱)
        deception_index_boost_mask = (norm_deception_index_bipolar[i] > 0) and \
                                     (norm_main_force_conviction_bipolar[i] < -conviction_threshold)
        if deception_index_boost_mask:
            deception_modulator[i] += norm_deception_index_bipolar[i] * deception_modulator_params_deception_index_weight
        # 欺骗指数惩罚 (负向欺骗且主力信念强)
        deception_index_penalty_mask = (norm_deception_index_bipolar[i] < 0) and \
                                       (norm_main_force_conviction_bipolar[i] > conviction_threshold)
        if deception_index_penalty_mask:
            deception_modulator[i] -= np.abs(norm_deception_index_bipolar[i]) * deception_modulator_params_deception_index_weight
    return np.clip(deception_modulator, 0.1, 2.0)

@nb.njit(cache=True)
def _numba_calculate_historical_potential_dgm_score_core(
    norm_deception_index: np.ndarray,
    norm_wash_trade_intensity: np.ndarray,
    norm_retail_panic_surrender: np.ndarray,
    norm_main_force_conviction: np.ndarray,
    chip_flow_directionality_proxy: np.ndarray,
    norm_pct_change: np.ndarray, # 新增参数：归一化后的日涨跌幅
    dgm_weights_deception_impact: float,
    dgm_weights_wash_trade_penalty: float,
    dgm_weights_flow_directionality_boost: float,
    dgm_weights_retail_panic_impact: float,
    dgm_weights_main_force_conviction_impact: float,
    bearish_deception_and_mf_out_penalty_factor: float,
    retail_panic_positive_pct_penalty_factor: float # 新增参数：散户恐慌上涨惩罚因子
) -> np.ndarray:
    """
    Numba优化后的核心函数，用于计算_diagnose_axiom_historical_potential中的dgm_score。
    """
    n = len(norm_deception_index)
    dgm_score_base = np.zeros(n, dtype=np.float32)
    dgm_score = np.zeros(n, dtype=np.float32)
    for i in range(n):
        bull_trap_mask = (norm_deception_index[i] > 0) and (chip_flow_directionality_proxy[i] < 0)
        bear_trap_absorption_mask = (norm_deception_index[i] < 0) and (chip_flow_directionality_proxy[i] > 0)
        bearish_deception_and_mf_out_mask = (norm_deception_index[i] < 0) and (chip_flow_directionality_proxy[i] < 0)

        # Bull trap penalty
        if bull_trap_mask:
            dgm_score_base[i] -= (norm_deception_index[i] * np.abs(chip_flow_directionality_proxy[i])) * dgm_weights_deception_impact * 1.5 # 惩罚因子加倍
        # Bear trap absorption
        elif bear_trap_absorption_mask:
            # 修复：当诱空反吸发生时，如果股价正在上涨，则减少奖励，因为这可能是一个诱多陷阱
            bear_trap_bonus = (np.abs(norm_deception_index[i]) * chip_flow_directionality_proxy[i]) * dgm_weights_deception_impact * 1.2
            if norm_pct_change[i] > 0.1: # 如果股价上涨超过一定阈值 (归一化后)
                # 减少奖励，因为上涨可能削弱了诱空反吸的有效性，或暗示诱多
                bear_trap_bonus *= (1.0 - norm_pct_change[i] * retail_panic_positive_pct_penalty_factor)
                bear_trap_bonus = max(0.0, bear_trap_bonus) # 确保奖励不为负
            dgm_score_base[i] += bear_trap_bonus

        dgm_score_base[i] -= norm_wash_trade_intensity[i] * dgm_weights_wash_trade_penalty

        if chip_flow_directionality_proxy[i] > 0 and not bull_trap_mask:
            dgm_score_base[i] += chip_flow_directionality_proxy[i] * dgm_weights_flow_directionality_boost

        # 修复：当散户恐慌指数高但股价上涨时，减少其正面影响
        retail_panic_impact_contribution = norm_retail_panic_surrender[i] * dgm_weights_retail_panic_impact
        if norm_retail_panic_surrender[i] > 0.5 and norm_pct_change[i] > 0.1: # 如果散户恐慌高且股价上涨
            # 减少恐慌带来的正面影响，因为这可能不是真正的底部恐慌，而是卖出到高点
            retail_panic_impact_contribution *= (1.0 - norm_pct_change[i] * retail_panic_positive_pct_penalty_factor)
            retail_panic_impact_contribution = max(0.0, retail_panic_impact_contribution) # 确保贡献不为负
        dgm_score_base[i] += retail_panic_impact_contribution

        dgm_score_base[i] += np.abs(norm_main_force_conviction[i]) * dgm_weights_main_force_conviction_impact

        # 最终的 dgm_score 赋值，处理极低负值情况
        if bearish_deception_and_mf_out_mask:
            dgm_score[i] = -0.9 # 强制设置为极低负值
        else:
            dgm_score[i] = dgm_score_base[i]
    return np.clip(dgm_score, -1.0, 1.0)

@nb.njit(cache=True)
def _numba_calculate_tactical_exchange_deception_core(
    chip_deception_direction: np.ndarray,
    norm_chip_fault: np.ndarray,
    norm_retail_panic_surrender: np.ndarray,
    norm_loser_pain: np.ndarray,
    norm_winner_profit_margin_avg: np.ndarray,
    norm_suppressive_accum: np.ndarray,
    norm_profit_realization_quality_inverse: np.ndarray, # (1 - norm_profit_realization)
    deception_outcome_effectiveness_threshold: float,
    deception_outcome_cost_threshold: float,
    deception_outcome_weights_effectiveness: float,
    deception_outcome_weights_cost: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba优化后的核心函数，用于计算_diagnose_tactical_exchange中的deception_modulator和chip_deception_score_refined。
    """
    n = len(chip_deception_direction)
    deception_effectiveness_score = np.zeros(n, dtype=np.float32)
    deception_cost_score = np.zeros(n, dtype=np.float32)
    deception_quality_modulator = np.zeros(n, dtype=np.float32)
    chip_deception_score_refined = np.zeros(n, dtype=np.float32)

    for i in range(n):
        # 诱空反吸增强
        if chip_deception_direction[i] > 0: # 诱空
            deception_effectiveness_score[i] = (norm_retail_panic_surrender[i] + norm_loser_pain[i]) / 2
            deception_cost_score[i] = norm_suppressive_accum[i]
        elif chip_deception_direction[i] < 0: # 诱多
            deception_effectiveness_score[i] = norm_winner_profit_margin_avg[i]
            deception_cost_score[i] = norm_profit_realization_quality_inverse[i]
        # Deception Quality Modulator
        deception_quality_modulator[i] = (
            deception_outcome_weights_effectiveness * max(0.0, min(deception_effectiveness_score[i], 1.0)) + # 修复
            deception_outcome_weights_cost * max(0.0, min(deception_cost_score[i], 1.0)) # 修复
        )
        high_quality_deception_mask = (deception_effectiveness_score[i] > deception_outcome_effectiveness_threshold) and \
                                      (deception_cost_score[i] > deception_outcome_cost_threshold)
        if not high_quality_deception_mask:
            deception_quality_modulator[i] *= 0.5 # 低质量欺骗减半调制效果
        # Refined Chip Deception Score
        chip_deception_score_refined[i] = norm_chip_fault[i] * chip_deception_direction[i] * (1 + max(0.0, min(deception_quality_modulator[i], 1.0))) # 修复
    return np.clip(chip_deception_score_refined, -1.0, 1.0), np.clip(deception_quality_modulator, 0.0, 1.0)

@nb.njit(cache=True)
def _numba_calculate_harmony_conflict_penalty_core(
    strategic_posture: np.ndarray,
    tactical_exchange: np.ndarray,
    norm_deception: np.ndarray,
    conflict_threshold: float,
    conflict_penalty_factor: float,
    deception_penalty_sensitivity: float
) -> np.ndarray:
    """
    Numba优化后的核心函数，用于计算_diagnose_strategic_tactical_harmony中的conflict_penalty_factor_adjusted。
    """
    n = len(strategic_posture)
    conflict_penalty_factor_adjusted = np.ones(n, dtype=np.float32)

    for i in range(n):
        strong_bullish_strategic_bearish_tactical = (strategic_posture[i] > conflict_threshold) and (tactical_exchange[i] < -conflict_threshold)
        strong_bearish_strategic_bullish_tactical = (strategic_posture[i] < -conflict_threshold) and (tactical_exchange[i] > conflict_threshold)
        conflict_mask = strong_bullish_strategic_bearish_tactical or strong_bearish_strategic_bullish_tactical
        if conflict_mask:
            deception_impact = 0.0
            if norm_deception[i] > 0: # 伴随欺骗的冲突
                deception_impact = norm_deception[i] * deception_penalty_sensitivity
            penalty_val = conflict_penalty_factor + deception_impact
            # 修复：使用min/max组合进行标量裁剪
            conflict_penalty_factor_adjusted[i] = 1 - max(0.0, min(penalty_val, 1.0))
            
    return np.clip(conflict_penalty_factor_adjusted, 0.0, 1.0)

@nb.njit(cache=True)
def _numba_calculate_inflection_strength_core(
    norm_velocity: np.ndarray,
    norm_acceleration: np.ndarray,
    positive_strength_tanh_factor: float,
    negative_strength_tanh_factor: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba优化后的核心函数，用于计算_diagnose_harmony_inflection中的positive/negative_inflection_strength。
    """
    n = len(norm_velocity)
    positive_inflection_strength = np.zeros(n, dtype=np.float32)
    negative_inflection_strength = np.zeros(n, dtype=np.float32)
    for i in range(n):
        # Positive Inflection Strength
        # 检查前一个速度是否存在，避免索引错误
        prev_norm_velocity = norm_velocity[i-1] if i > 0 else norm_velocity[i] # 简化处理，实际应更严谨
        positive_inflection_mask = ((prev_norm_velocity < 0) and (norm_velocity[i] >= 0)) or \
                                   ((norm_velocity[i] < 0) and (norm_acceleration[i] > 0)) or \
                                   ((norm_velocity[i] >= 0) and (norm_acceleration[i] > 0))
        if positive_inflection_mask:
            positive_inflection_strength[i] = np.tanh((np.maximum(0.0, norm_velocity[i]) + np.maximum(0.0, norm_acceleration[i])) * positive_strength_tanh_factor)
        # Negative Inflection Strength
        negative_inflection_mask = ((prev_norm_velocity > 0) and (norm_velocity[i] <= 0)) or \
                                   ((norm_velocity[i] > 0) and (norm_acceleration[i] < 0)) or \
                                   ((norm_velocity[i] <= 0) and (norm_acceleration[i] < 0))
        if negative_inflection_mask:
            negative_inflection_strength[i] = np.tanh((np.abs(np.minimum(0.0, norm_velocity[i])) + np.abs(np.minimum(0.0, norm_acceleration[i]))) * negative_strength_tanh_factor)
    return positive_inflection_strength, negative_inflection_strength

@nb.njit(cache=True)
def _numba_calculate_harmony_inflection_deception_modulator_core(
    inflection_strength: np.ndarray,
    norm_deception: np.ndarray,
    norm_wash_trade: np.ndarray,
    deception_boost_factor_negative: float,
    deception_penalty_sensitivity: float,
    wash_trade_mitigation_sensitivity: float
) -> np.ndarray:
    """
    Numba优化后的核心函数，用于计算_diagnose_harmony_inflection中的deception_modulator。
    """
    n = len(inflection_strength)
    deception_modulator = np.ones(n, dtype=np.float32)

    for i in range(n):
        # 正向拐点且诱空反吸增强
        if inflection_strength[i] > 0 and norm_deception[i] < 0:
            deception_modulator[i] *= (1 + np.abs(norm_deception[i]) * deception_boost_factor_negative * 1.5)
        # 正向拐点且诱多惩罚
        elif inflection_strength[i] > 0 and norm_deception[i] > 0:
            penalty_val = norm_deception[i] * deception_penalty_sensitivity
            # 修复：使用min/max组合进行标量裁剪
            deception_modulator[i] *= (1 - max(0.0, min(penalty_val, 1.0)))
        # 负向拐点且对倒缓解
        elif inflection_strength[i] < 0 and norm_wash_trade[i] > 0:
            mitigation_val = norm_wash_trade[i] * wash_trade_mitigation_sensitivity
            # 修复：使用min/max组合进行标量裁剪
            deception_modulator[i] *= (1 + max(0.0, min(mitigation_val, 0.5)))
    return np.clip(deception_modulator, 0.1, 2.0)

@nb.njit(cache=True)
def _numba_calculate_retail_vulnerability_modulator_core(
    norm_volatility_instability: np.ndarray,
    norm_market_sentiment_extreme: np.ndarray,
    volatility_weight: float,
    sentiment_weight: float
) -> np.ndarray:
    """
    Numba优化后的核心函数，用于计算_diagnose_chip_retail_vulnerability中的modulator。
    """
    n = len(norm_volatility_instability)
    modulator_values = np.zeros(n, dtype=np.float32)
    for i in range(n):
        # 几何平均融合
        numerator = (norm_volatility_instability[i] + 1e-9)**volatility_weight * \
                    (norm_market_sentiment_extreme[i] + 1e-9)**sentiment_weight
        denominator = volatility_weight + sentiment_weight
        if denominator > 0:
            modulator_values[i] = numerator**(1/denominator)
        else:
            modulator_values[i] = 0.0 # 避免除以零
    return np.clip(modulator_values, 0.0, 1.0)

@nb.njit(cache=True)
def _numba_calculate_main_force_cost_intent_deception_modulator_core(
    net_conviction_flow_quality: np.ndarray,
    norm_deception_index_bipolar: np.ndarray,
    norm_wash_trade_intensity: np.ndarray,
    deception_boost_factor: float,
    deception_penalty_factor: float,
    wash_trade_penalty_factor: float
) -> np.ndarray:
    """
    Numba优化后的核心函数，用于计算_diagnose_chip_main_force_cost_intent中的deception_modulator。
    """
    n = len(net_conviction_flow_quality)
    deception_modulator = np.ones(n, dtype=np.float32)
    for i in range(n):
        # 诱空反吸增强
        if net_conviction_flow_quality[i] > 0 and norm_deception_index_bipolar[i] < 0:
            deception_modulator[i] *= (1 + np.abs(norm_deception_index_bipolar[i]) * deception_boost_factor)
        # 诱多反吸增强
        elif net_conviction_flow_quality[i] < 0 and norm_deception_index_bipolar[i] > 0:
            deception_modulator[i] *= (1 + norm_deception_index_bipolar[i] * deception_boost_factor)
        # 诱多惩罚
        elif net_conviction_flow_quality[i] > 0 and norm_deception_index_bipolar[i] > 0:
            deception_modulator[i] *= (1 - norm_deception_index_bipolar[i] * deception_penalty_factor)
        # 对倒惩罚
        deception_modulator[i] *= (1 - norm_wash_trade_intensity[i] * wash_trade_penalty_factor)
    return np.clip(deception_modulator, 0.1, 2.0)

@nb.njit(cache=True)
def _numba_calculate_main_force_cost_intent_core(
    close_raw: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    net_conviction_flow_quality: np.ndarray,
    norm_main_force_conviction: np.ndarray,
    price_deviation_factor_buy: np.ndarray,
    price_deviation_factor_sell: np.ndarray,
    in_zone_intent_base_multiplier: float,
    in_zone_intent_modulator: np.ndarray,
    in_zone_health_slope_sensitivity: float,
    dynamic_weight_below_vpoc: np.ndarray,
    dynamic_weight_above_vpoc: np.ndarray,
    dynamic_weight_in_vpoc: np.ndarray
) -> np.ndarray:
    """
    Numba优化后的核心函数，用于计算_diagnose_chip_main_force_cost_intent中的main_force_cost_intent_raw。
    """
    n = len(close_raw)
    main_force_cost_intent_raw = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if close_raw[i] < lower_bound[i]: # 价格低于成本区
            intent_below_vpoc = np.maximum(0.0, net_conviction_flow_quality[i]) * \
                                norm_main_force_conviction[i] * \
                                (1 + price_deviation_factor_buy[i]) * \
                                dynamic_weight_below_vpoc[i]
            main_force_cost_intent_raw[i] = intent_below_vpoc
        elif close_raw[i] > upper_bound[i]: # 价格高于成本区
            intent_above_vpoc = - (np.abs(np.minimum(0.0, net_conviction_flow_quality[i])) * \
                                   norm_main_force_conviction[i] * \
                                   (1 + price_deviation_factor_sell[i]) * \
                                   dynamic_weight_above_vpoc[i])
            main_force_cost_intent_raw[i] = intent_above_vpoc
        elif close_raw[i] >= lower_bound[i] and close_raw[i] <= upper_bound[i]: # 价格在成本区内
            # 修复：使用min/max组合进行标量裁剪
            clipped_in_zone_modulator = max(-0.5, min(in_zone_intent_modulator[i], 0.5))
            intent_in_vpoc = net_conviction_flow_quality[i] * \
                             norm_main_force_conviction[i] * \
                             (in_zone_intent_base_multiplier + clipped_in_zone_modulator * in_zone_health_slope_sensitivity) * \
                             dynamic_weight_in_vpoc[i]
            main_force_cost_intent_raw[i] = intent_in_vpoc
    return main_force_cost_intent_raw

@nb.njit(cache=True)
def _numba_calculate_hollowing_out_risk_core(
    dispersion_weakness_score: np.ndarray,
    distribution_pressure_score: np.ndarray,
    main_force_deception_score: np.ndarray,
    market_vulnerability_score: np.ndarray,
    dynamic_fusion_weights_dispersion: np.ndarray,
    dynamic_fusion_weights_distribution: np.ndarray,
    dynamic_fusion_weights_deception: np.ndarray,
    dynamic_fusion_weights_vulnerability: np.ndarray,
    deception_amplification_factor: float,
    non_linear_exponent: float
) -> np.ndarray:
    """
    Numba优化后的核心函数，用于计算_diagnose_chip_hollowing_out_risk中的最终分数。
    """
    n = len(dispersion_weakness_score)
    final_score_values = np.zeros(n, dtype=np.float32)
    for i in range(n):
        hollowing_out_risk_score = (
            dispersion_weakness_score[i] * dynamic_fusion_weights_dispersion[i] +
            distribution_pressure_score[i] * dynamic_fusion_weights_distribution[i] +
            main_force_deception_score[i] * dynamic_fusion_weights_deception[i] +
            market_vulnerability_score[i] * dynamic_fusion_weights_vulnerability[i]
        )
        deception_amplifier = 1 + main_force_deception_score[i] * deception_amplification_factor
        hollowing_out_risk_score *= deception_amplifier
        # 修复：使用min/max组合进行标量裁剪
        final_score_values[i] = np.tanh(max(0.0, min(hollowing_out_risk_score, 1.0))**non_linear_exponent)
    return np.clip(final_score_values, 0.0, 1.0)

@nb.njit(cache=True)
def _numba_calculate_turnover_purity_cost_optimization_core(
    norm_wash_trade_intensity: np.ndarray,
    norm_net_conviction_flow: np.ndarray,
    norm_winner_profit_margin_avg: np.ndarray,
    norm_loser_pain_index: np.ndarray,
    norm_turnover_rate: np.ndarray
) -> np.ndarray:
    """
    Numba优化后的核心函数，用于计算_diagnose_chip_turnover_purity_cost_optimization中的最终分数。
    """
    n = len(norm_wash_trade_intensity)
    final_score_values = np.zeros(n, dtype=np.float32)
    for i in range(n):
        purity_factor = (1 - norm_wash_trade_intensity[i])
        cost_optimization_factor = (norm_winner_profit_margin_avg[i] + norm_loser_pain_index[i]) / 2
        turnover_quality_factor = purity_factor * cost_optimization_factor * norm_net_conviction_flow[i]
        turnover_purity_cost_optimization = turnover_quality_factor * (1 + norm_turnover_rate[i] * 0.5)
        final_score_values[i] = turnover_purity_cost_optimization
    return np.clip(final_score_values, -1.0, 1.0)

@nb.njit(cache=True)
def _numba_calculate_despair_temptation_zones_core(
    norm_loser_pain_index: np.ndarray,
    norm_total_loser_rate: np.ndarray,
    norm_panic_buy_absorption_contribution: np.ndarray,
    norm_retail_fomo_premium_index: np.ndarray,
    norm_winner_profit_margin_avg: np.ndarray,
    norm_total_winner_rate_temptation: np.ndarray
) -> np.ndarray:
    """
    Numba优化后的核心函数，用于计算_diagnose_chip_despair_temptation_zones中的despair_temptation_score。
    """
    n = len(norm_loser_pain_index)
    despair_temptation_score_values = np.zeros(n, dtype=np.float32)
    for i in range(n):
        despair_strength = (
            norm_loser_pain_index[i]**0.4 *
            norm_total_loser_rate[i]**0.3 *
            norm_panic_buy_absorption_contribution[i]**0.3
        )
        temptation_strength = (
            norm_retail_fomo_premium_index[i]**0.4 *
            norm_winner_profit_margin_avg[i]**0.3 *
            norm_total_winner_rate_temptation[i]**0.3
        )
        despair_temptation_score_values[i] = temptation_strength - despair_strength
    return np.tanh(despair_temptation_score_values * 2)

@nb.njit(cache=True)
def _numba_calculate_bull_trap_penalty_core(
    has_recent_sharp_drop: np.ndarray,
    composite_positive_deception_score: np.ndarray,
    has_positive_deception: np.ndarray,
    deception_penalty_multiplier: float,
    dynamic_penalty_sensitivity: np.ndarray
) -> np.ndarray:
    """
    Numba优化后的核心函数，用于计算_calculate_bull_trap_context_penalty中的penalty_factor。
    """
    n = len(has_recent_sharp_drop)
    penalty_factor = np.ones(n, dtype=np.float32)

    for i in range(n):
        bull_trap_condition = has_recent_sharp_drop[i] and has_positive_deception[i]
        if bull_trap_condition:
            penalty_strength = composite_positive_deception_score[i] * deception_penalty_multiplier * dynamic_penalty_sensitivity[i]
            # 修复：使用min/max组合进行标量裁剪
            penalty_factor[i] = 1 - max(0.0, min(penalty_strength, 1.0))
            
    return penalty_factor

class ChipIntelligence:
    def __init__(self, strategy_instance):
        """
        【V2.3 · 外部配置加载版】
        - 核心升级: `chip_ultimate_params` 现在从外部文件 `config/intelligence/chip.json` 加载，
                     解决了配置块转移的问题，并确保了模块化。
        - 核心修复: 注入 debug_params，并预处理 probe_dates，解决 AttributeError。
        """
        self.strategy = strategy_instance
        self.score_type_map = get_params_block(self.strategy, 'score_type_map', {})
        process_params = get_params_block(self.strategy, 'process_intelligence_params', {})
        self.bipolar_sensitivity = get_param_value(process_params.get('bipolar_sensitivity'), 1.0)
        # 从外部文件加载 chip_ultimate_params
        # loaded_chip_config 应该直接是 chip.json 的内容
        loaded_chip_config = load_external_json_config("config/intelligence/chip.json", {})
        # 直接从加载的配置中获取 chip_ultimate_params 块，而不是通过 get_params_block
        self.chip_ultimate_params = loaded_chip_config.get('chip_ultimate_params', {})
        self.debug_params = get_params_block(self.strategy, 'debug_params', {})
        self.should_probe = self.debug_params.get('should_probe', False)
        self.probe_dates_set = {pd.to_datetime(d).date() for d in self.debug_params.get('probe_dates', [])}

    def _get_safe_series(self, df: pd.DataFrame, data_source: Union[pd.DataFrame, Dict[str, pd.Series]], column_name: str, default_value: Any = 0.0, method_name: str = "未知方法") -> pd.Series:
        """
        【V2.0 · 上下文修复版】安全地从DataFrame或字典中获取Series，如果不存在则打印警告并返回默认Series。
        - 核心修复: 接收 df 参数，并使用其索引创建默认 Series，确保上下文一致。
        """
        df_index = df.index # 使用传入的 df.index
        if isinstance(data_source, pd.DataFrame):
            if column_name not in data_source.columns:
                print(f"    -> [筹码情报警告] 方法 '{method_name}' 缺少DataFrame数据 '{column_name}'，使用默认值 {default_value}。")
                return pd.Series(default_value, index=df_index)
            return data_source[column_name]
        elif isinstance(data_source, dict):
            if column_name not in data_source:
                print(f"    -> [筹码情报警告] 方法 '{method_name}' 缺少字典数据 '{column_name}'，使用默认值 {default_value}。")
                return pd.Series(default_value, index=df_index)
            series = data_source[column_name]
            if isinstance(series, pd.Series):
                return series.reindex(df_index, fill_value=default_value)
            else:
                return pd.Series(series, index=df_index)
        else:
            print(f"    -> [筹码情报警告] 方法 '{method_name}' 接收到未知数据源类型 {type(data_source)}，无法获取 '{column_name}'，使用默认值 {default_value}。")
            return pd.Series(default_value, index=df_index)

    def _get_all_required_signals(self, df: pd.DataFrame, required_signals: list, method_name: str) -> Dict[str, pd.Series]:
        """
        【V1.0 · 恢复版】高效地从DataFrame中获取所有必需的Series，如果不存在则打印警告并返回默认Series。
        """
        df_index = df.index
        signals_data = {}
        for signal_name in required_signals:
            if signal_name not in df.columns:
                print(f"    -> [筹码情报警告] 方法 '{method_name}' 缺少DataFrame数据 '{signal_name}'，使用默认值 0.0。")
                signals_data[signal_name] = pd.Series(0.0, index=df_index)
            else:
                signals_data[signal_name] = df[signal_name]
        return signals_data

    def _validate_required_signals(self, df: pd.DataFrame, required_signals: list, method_name: str) -> bool:
        """
        【V1.0 · 战前情报校验】内部辅助方法，用于在方法执行前验证所有必需的数据信号是否存在。
        """
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            print(f"    -> [筹码情报校验] 方法 '{method_name}' 启动失败：缺少核心信号 {missing_signals}。")
            return False
        return True

    def run_chip_intelligence_command(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V19.0 · 诡道反吸版】筹码情报总指挥
        - 核心升维: 升级“吸筹回声”信号到 V2.0，严格遵循纯筹码原则，深度融入诡道博弈特性。
        - 新增功能: 整合5个新的筹码信号诊断方法，并为所有核心信号添加校验打印。
        """
        print("启动【V19.0 · 诡道反吸版】筹码情报分析...")
        all_chip_states = {}
        periods = [5, 13, 21, 55]
        # 借用行为层的MTF权重配置
        p_behavior_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_behavior_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        # 调用并记录持仓信念韧性信号
        holder_sentiment_scores = self._diagnose_axiom_holder_sentiment(df, periods)
        # 确保 holder_sentiment_scores 是 Series
        if not isinstance(holder_sentiment_scores, pd.Series):
            print(f"    -> [筹码情报警告] _diagnose_axiom_holder_sentiment 返回无效类型，使用默认值。")
            holder_sentiment_scores = pd.Series(0.0, index=df.index)
        all_chip_states['SCORE_CHIP_AXIOM_HOLDER_SENTIMENT'] = holder_sentiment_scores
        # 调用并记录价筹张力信号
        divergence_scores = self._diagnose_axiom_divergence(df, periods)
        # 确保 divergence_scores 是 Series
        if not isinstance(divergence_scores, pd.Series):
            print(f"    -> [筹码情报警告] _diagnose_axiom_divergence 返回无效类型，使用默认值。")
            divergence_scores = pd.Series(0.0, index=df.index)
        all_chip_states['SCORE_CHIP_AXIOM_DIVERGENCE'] = divergence_scores
        # 调用并记录战略态势信号
        strategic_posture = self._diagnose_strategic_posture(df)
        # 确保 strategic_posture 是 Series
        if not isinstance(strategic_posture, pd.Series):
            print(f"    -> [筹码情报警告] _diagnose_strategic_posture 返回无效类型，使用默认值。")
            strategic_posture = pd.Series(0.0, index=df.index)
        all_chip_states['SCORE_CHIP_STRATEGIC_POSTURE'] = strategic_posture
        # 调用并记录战场地形信号
        battlefield_geography = self._diagnose_battlefield_geography(df)
        # 确保 battlefield_geography 是 Series
        if not isinstance(battlefield_geography, pd.Series):
            print(f"    -> [筹码情报警告] _diagnose_battlefield_geography 返回无效类型，使用默认值。")
            battlefield_geography = pd.Series(0.0, index=df.index)
        all_chip_states['SCORE_CHIP_BATTLEFIELD_GEOGRAPHY'] = battlefield_geography
        # 调用并记录筹码趋势动量信号
        chip_trend_momentum_scores = self._diagnose_axiom_trend_momentum(df, periods, strategic_posture, battlefield_geography, holder_sentiment_scores)
        # 确保 chip_trend_momentum_scores 是 Series
        if not isinstance(chip_trend_momentum_scores, pd.Series):
            print(f"    -> [筹码情报警告] _diagnose_axiom_trend_momentum 返回无效类型，使用默认值。")
            chip_trend_momentum_scores = pd.Series(0.0, index=df.index)
        all_chip_states['SCORE_CHIP_AXIOM_TREND_MOMENTUM'] = chip_trend_momentum_scores
        # 调用并记录筹码历史潜力信号
        historical_potential = self._diagnose_axiom_historical_potential(df)
        # 确保 historical_potential 是 Series
        if not isinstance(historical_potential, pd.Series):
            print(f"    -> [筹码情报警告] _diagnose_axiom_historical_potential 返回无效类型，使用默认值。")
            historical_potential = pd.Series(0.0, index=df.index)
        all_chip_states['SCORE_CHIP_AXIOM_HISTORICAL_POTENTIAL'] = historical_potential
        # 调用并记录吸筹回声信号
        absorption_echo = self._diagnose_absorption_echo(df, divergence_scores)
        # 确保 absorption_echo 是 Series
        if not isinstance(absorption_echo, pd.Series):
            print(f"    -> [筹码情报警告] _diagnose_absorption_echo 返回无效类型，使用默认值。")
            absorption_echo = pd.Series(0.0, index=df.index)
        all_chip_states['SCORE_CHIP_OPP_ABSORPTION_ECHO'] = absorption_echo
        # 调用并记录派发诡影信号
        distribution_whisper = self._diagnose_distribution_whisper(df, divergence_scores)
        # 确保 distribution_whisper 是 Series
        if not isinstance(distribution_whisper, pd.Series):
            print(f"    -> [筹码情报警告] _diagnose_distribution_whisper 返回无效类型，使用默认值。")
            distribution_whisper = pd.Series(0.0, index=df.index)
        all_chip_states['SCORE_CHIP_RISK_DISTRIBUTION_WHISPER'] = distribution_whisper
        # 调用并记录筹码一致驱动信号
        coherent_drive = self._diagnose_structural_consensus(df, battlefield_geography, holder_sentiment_scores)
        # 确保 coherent_drive 是 Series
        if not isinstance(coherent_drive, pd.Series):
            print(f"    -> [筹码情报警告] _diagnose_structural_consensus 返回无效类型，使用默认值。")
            coherent_drive = pd.Series(0.0, index=df.index)
        all_chip_states['SCORE_CHIP_COHERENT_DRIVE'] = coherent_drive
        # 调用并记录战术换手博弈信号
        tactical_exchange = self._diagnose_tactical_exchange(df, battlefield_geography)
        # 确保 tactical_exchange 是 Series
        if not isinstance(tactical_exchange, pd.Series):
            print(f"    -> [筹码情报警告] _diagnose_tactical_exchange 返回无效类型，使用默认值。")
            tactical_exchange = pd.Series(0.0, index=df.index)
        all_chip_states['SCORE_CHIP_TACTICAL_EXCHANGE'] = tactical_exchange
        # 调用并记录战略战术和谐度信号
        strategic_tactical_harmony = self._diagnose_strategic_tactical_harmony(df, strategic_posture, tactical_exchange, holder_sentiment_scores)
        # 确保 strategic_tactical_harmony 是 Series
        if not isinstance(strategic_tactical_harmony, pd.Series):
            print(f"    -> [筹码情报警告] _diagnose_strategic_tactical_harmony 返回无效类型，使用默认值。")
            strategic_tactical_harmony = pd.Series(0.0, index=df.index)
        all_chip_states['SCORE_CHIP_STRATEGIC_TACTICAL_HARMONY'] = strategic_tactical_harmony
        # 调用并记录和谐拐点信号
        harmony_inflection = self._diagnose_harmony_inflection(df, strategic_tactical_harmony)
        # 确保 harmony_inflection 是 Series
        if not isinstance(harmony_inflection, pd.Series):
            print(f"    -> [筹码情报警告] _diagnose_harmony_inflection 返回无效类型，使用默认值。")
            harmony_inflection = pd.Series(0.0, index=df.index)
        all_chip_states['SCORE_CHIP_HARMONY_INFLECTION'] = harmony_inflection
        # --- 调用新的筹码信号诊断方法 ---
        # 调用并记录散户筹码脆弱性指数信号
        retail_vulnerability = self._diagnose_chip_retail_vulnerability(df)
        # 确保 retail_vulnerability 是 Series
        if not isinstance(retail_vulnerability, pd.Series):
            print(f"    -> [筹码情报警告] _diagnose_chip_retail_vulnerability 返回无效类型，使用默认值。")
            retail_vulnerability = pd.Series(0.0, index=df.index)
        all_chip_states['SCORE_CHIP_RETAIL_VULNERABILITY'] = retail_vulnerability
        # 调用并记录主力成本区攻防意图信号
        main_force_cost_intent = self._diagnose_chip_main_force_cost_intent(df)
        # 确保 main_force_cost_intent 是 Series
        if not isinstance(main_force_cost_intent, pd.Series):
            print(f"    -> [筹码情报警告] _diagnose_chip_main_force_cost_intent 返回无效类型，使用默认值。")
            main_force_cost_intent = pd.Series(0.0, index=df.index)
        all_chip_states['SCORE_CHIP_MAIN_FORCE_COST_INTENT'] = main_force_cost_intent
        # 调用并记录筹码空心化风险信号
        hollowing_out_risk = self._diagnose_chip_hollowing_out_risk(df)
        # 确保 hollowing_out_risk 是 Series
        if not isinstance(hollowing_out_risk, pd.Series):
            print(f"    -> [筹码情报警告] _diagnose_chip_hollowing_out_risk 返回无效类型，使用默认值。")
            hollowing_out_risk = pd.Series(0.0, index=df.index)
        all_chip_states['SCORE_CHIP_HOLLOWING_OUT_RISK'] = hollowing_out_risk
        # 调用并记录换手纯度与成本优化信号
        turnover_purity_cost_optimization = self._diagnose_chip_turnover_purity_cost_optimization(df)
        # 确保 turnover_purity_cost_optimization 是 Series
        if not isinstance(turnover_purity_cost_optimization, pd.Series):
            print(f"    -> [筹码情报警告] _diagnose_chip_turnover_purity_cost_optimization 返回无效类型，使用默认值。")
            turnover_purity_cost_optimization = pd.Series(0.0, index=df.index)
        all_chip_states['SCORE_CHIP_TURNOVER_PURITY_COST_OPTIMIZATION'] = turnover_purity_cost_optimization
        # 调用并记录筹码绝望与诱惑区信号
        despair_temptation_zones = self._diagnose_chip_despair_temptation_zones(df)
        # 确保 despair_temptation_zones 是 Series
        if not isinstance(despair_temptation_zones, pd.Series):
            print(f"    -> [筹码情报警告] _diagnose_chip_despair_temptation_zones 返回无效类型，使用默认值。")
            despair_temptation_zones = pd.Series(0.0, index=df.index)
        all_chip_states['SCORE_CHIP_DESPAIR_TEMPTATION_ZONES'] = despair_temptation_zones
        # 更新最终生成的筹码原子信号数量
        print(f"【V19.0 · 诡道反吸版】分析完成，生成 {len(all_chip_states)} 个筹码原子信号。")
        return all_chip_states

    def _diagnose_strategic_posture(self, df: pd.DataFrame) -> pd.Series:
        """
        【V9.2 · Numba优化版】诊断主力的综合战略态势。
        - 核心优化: 将deception_modulator和synergy_factor的计算逻辑迁移至Numba加速的辅助函数。
        - 核心升级1: 诡道博弈深度融合与情境调制：引入主力信念和筹码健康度作为情境，动态调整欺骗指数和对倒强度的影响，实现非对称调制，更精准识别和应对主力诡道博弈。
        - 核心升级2: 动态权重自适应：根据筹码波动不稳定性、筹码健康度斜率等情境因子，动态调整基础态势、速度和加速度的融合权重，使信号自适应市场动态。
        - 核心升级3: 维度间非线性互动增强：引入“协同/冲突”因子，评估阵型部署、指挥官决心、战场控制各维度之间非线性互动，提高信号的敏感性和准确性。
        - 核心升级4: 全局情境调制器：引入筹码健康度、市场情绪作为全局调制器，对最终战略态态势分数进行校准，提高信号在不同市场情境下的可靠性。
        - 核心升级5: 新增筹码指标整合：
            - 诱多/诱空欺骗强度 (`deception_lure_long_intensity_D`, `deception_lure_short_intensity_D`) 进一步精细化诡道调制。
            - 主力成本区买卖意图 (`mf_cost_zone_buy_intent_D`, `mf_cost_zone_sell_intent_D`) 增强指挥官决心维度。
            - 隐蔽派发信号 (`covert_distribution_signal_D`) 作为负向调制器。
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        method_name = "_diagnose_strategic_posture"
        df_index = df.index
        p_conf = self.chip_ultimate_params
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        parsed_tf_data = utils._parse_tf_weights(tf_weights)
        sp_params = get_param_value(p_conf.get('strategic_posture_params'), {})
        deception_fusion_weights = get_param_value(sp_params.get('deception_fusion_weights'), {"bear_trap_positive": 0.6, "bull_trap_negative": 0.2, "wash_trade_negative": 0.2})
        deception_context_mod_enabled = get_param_value(sp_params.get('deception_context_mod_enabled'), True)
        deception_conviction_threshold = get_param_value(sp_params.get('deception_conviction_threshold'), 0.2)
        deception_health_threshold = get_param_value(sp_params.get('deception_health_threshold'), 0.5)
        deception_boost_factor = get_param_value(sp_params.get('deception_boost_factor'), 0.5)
        deception_penalty_factor = get_param_value(sp_params.get('deception_penalty_factor'), 0.7)
        wash_trade_penalty_factor = get_param_value(sp_params.get('wash_trade_penalty_factor'), 0.3)
        deception_lure_long_penalty_factor = get_param_value(sp_params.get('deception_lure_long_penalty_factor'), 0.3)
        deception_lure_short_boost_factor = get_param_value(sp_params.get('deception_lure_short_boost_factor'), 0.3)
        mf_cost_zone_buy_intent_weight = get_param_value(sp_params.get('mf_cost_zone_buy_intent_weight'), 0.1)
        mf_cost_zone_sell_intent_weight = get_param_value(sp_params.get('mf_cost_zone_sell_intent_weight'), 0.1)
        covert_distribution_penalty_factor = get_param_value(sp_params.get('covert_distribution_penalty_factor'), 0.2)
        dynamic_fusion_weights_base = get_param_value(sp_params.get('dynamic_fusion_weights_base'), {'base_score': 0.6, 'velocity': 0.2, 'acceleration': 0.2})
        dynamic_weight_modulator_signal_1_name = get_param_value(sp_params.get('dynamic_weight_modulator_signal_1'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        dynamic_weight_modulator_signal_2_name = get_param_value(sp_params.get('dynamic_weight_modulator_signal_2'), 'SLOPE_5_chip_health_score_D')
        dynamic_weight_sensitivity_volatility = get_param_value(sp_params.get('dynamic_weight_sensitivity_volatility'), 0.4)
        dynamic_weight_sensitivity_health_slope = get_param_value(sp_params.get('dynamic_weight_sensitivity_health_slope'), 0.3)
        inter_dimension_interaction_enabled = get_param_value(sp_params.get('inter_dimension_interaction_enabled'), True)
        synergy_bonus_factor = get_param_value(sp_params.get('synergy_bonus_factor'), 0.15)
        conflict_penalty_factor = get_param_value(sp_params.get('conflict_penalty_factor'), 0.2)
        global_context_modulator_enabled = get_param_value(sp_params.get('global_context_modulator_enabled'), True)
        global_context_signal_1_name = get_param_value(sp_params.get('global_context_signal_1'), 'chip_health_score_D')
        global_context_signal_2_name = get_param_value(sp_params.get('global_context_signal_2'), 'market_sentiment_score_D')
        global_context_sensitivity_health = get_param_value(sp_params.get('global_context_sensitivity_health'), 0.5)
        global_context_sensitivity_sentiment = get_param_value(sp_params.get('global_context_sensitivity_sentiment'), 0.3)
        smoothing_ema_span = get_param_value(sp_params.get('smoothing_ema_span'), 5)
        required_signals = [
            'cost_gini_coefficient_D', 'covert_accumulation_signal_D', 'peak_exchange_purity_D',
            'main_force_cost_advantage_D', 'control_solidity_index_D', 'SLOPE_5_main_force_conviction_index_D',
            'floating_chip_cleansing_efficiency_D', 'dominant_peak_solidity_D',
            'deception_index_D', 'wash_trade_intensity_D',
            'main_force_conviction_index_D', 'chip_health_score_D',
            dynamic_weight_modulator_signal_1_name, dynamic_weight_modulator_signal_2_name,
            global_context_signal_1_name, global_context_signal_2_name,
            'deception_lure_long_intensity_D', 'deception_lure_short_intensity_D',
            'mf_cost_zone_buy_intent_D', 'mf_cost_zone_sell_intent_D', 'covert_distribution_signal_D'
        ]
        # 获取调试信息
        is_debug_enabled = self.should_probe # 假设 self.should_probe 已经从外部设置
        probe_ts = None
        if is_debug_enabled and self.probe_dates_set:
            for date in reversed(df_index):
                if date.date() in self.probe_dates_set:
                    probe_ts = date
                    break
        # print(f"  -- [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在诊断主力战略态势...") if is_debug_enabled and probe_ts and probe_ts in df.index else None
        if not self._validate_required_signals(df, required_signals, method_name):
            print(f"  -- [筹码层调试] {probe_ts.strftime('%Y-%m-%d')}: 缺少必要信号，返回0。") if is_debug_enabled and probe_ts and probe_ts in df.index else None
            return pd.Series(0.0, index=df.index)
        signals_data = self._get_all_required_signals(df, required_signals, method_name)
        # 提取原始信号
        cost_gini_coefficient_raw = signals_data['cost_gini_coefficient_D']
        covert_accumulation_raw = signals_data['covert_accumulation_signal_D']
        peak_exchange_purity_raw = signals_data['peak_exchange_purity_D']
        main_force_cost_advantage_raw = signals_data['main_force_cost_advantage_D']
        control_solidity_index_raw = signals_data['control_solidity_index_D']
        conviction_slope_raw = signals_data['SLOPE_5_main_force_conviction_index_D']
        deception_index_raw = signals_data['deception_index_D']
        wash_trade_intensity_raw = signals_data['wash_trade_intensity_D']
        cleansing_efficiency_raw = signals_data['floating_chip_cleansing_efficiency_D']
        dominant_peak_solidity_raw = signals_data['dominant_peak_solidity_D']
        main_force_conviction_raw = signals_data['main_force_conviction_index_D']
        chip_health_raw = signals_data['chip_health_score_D']
        volatility_instability_raw = signals_data[dynamic_weight_modulator_signal_1_name]
        chip_health_slope_raw = signals_data[dynamic_weight_modulator_signal_2_name]
        market_sentiment_raw = signals_data[global_context_signal_2_name]
        deception_lure_long_intensity_raw = signals_data['deception_lure_long_intensity_D']
        deception_lure_short_intensity_raw = signals_data['deception_lure_short_intensity_D']
        mf_cost_zone_buy_intent_raw = signals_data['mf_cost_zone_buy_intent_D']
        mf_cost_zone_sell_intent_raw = signals_data['mf_cost_zone_sell_intent_D']
        covert_distribution_signal_raw = signals_data['covert_distribution_signal_D']
        # if is_debug_enabled and probe_ts and probe_ts in df.index:
        #     print(f"      [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---")
        #     for sig_name in required_signals:
        #         val = signals_data[sig_name].loc[probe_ts] if probe_ts in signals_data[sig_name].index else np.nan
        #         print(f"        '{sig_name}': {val:.4f}")
        # 1. 阵型部署 (Formation Deployment)
        concentration_level = 1 - cost_gini_coefficient_raw
        level_score = utils.get_adaptive_mtf_normalized_bipolar_score(concentration_level, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_covert_accumulation = utils.get_adaptive_mtf_normalized_bipolar_score(covert_accumulation_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_peak_exchange_purity = utils.get_adaptive_mtf_normalized_bipolar_score(peak_exchange_purity_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        efficiency_score = (
            (norm_covert_accumulation.add(1)/2) *
            (norm_peak_exchange_purity.add(1)/2)
        ).pow(0.5) * 2 - 1
        formation_deployment_score = ((level_score.add(1)/2) * (efficiency_score.add(1)/2)).pow(0.5) * 2 - 1
        # if is_debug_enabled and probe_ts and probe_ts in df.index:
        #     print(f"      [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 阵型部署计算 ---")
        #     print(f"        集中度水平 (level_score): {level_score.loc[probe_ts]:.4f}")
        #     print(f"        隐蔽吸筹归一化 (norm_covert_accumulation): {norm_covert_accumulation.loc[probe_ts]:.4f}")
        #     print(f"        峰值换手纯度归一化 (norm_peak_exchange_purity): {norm_peak_exchange_purity.loc[probe_ts]:.4f}")
        #     print(f"        效率得分 (efficiency_score): {efficiency_score.loc[probe_ts]:.4f}")
        #     print(f"        阵型部署得分 (formation_deployment_score): {formation_deployment_score.loc[probe_ts]:.4f}")
        # 2. 指挥官决心 (Commanders Resolve)
        advantage_score = utils.get_adaptive_mtf_normalized_bipolar_score(main_force_cost_advantage_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        solidity_score = utils.get_adaptive_mtf_normalized_bipolar_score(control_solidity_index_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        intent_score = utils.get_adaptive_mtf_normalized_bipolar_score(conviction_slope_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_mf_cost_zone_buy_intent = utils.get_adaptive_mtf_normalized_score(mf_cost_zone_buy_intent_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_mf_cost_zone_sell_intent = utils.get_adaptive_mtf_normalized_score(mf_cost_zone_sell_intent_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        commanders_resolve_score = (
            (advantage_score.add(1)/2) * (solidity_score.add(1)/2) *
            (intent_score.clip(lower=-1, upper=1).add(1)/2)
        ).pow(1/3) * 2 - 1
        commanders_resolve_score = commanders_resolve_score + \
                                   (norm_mf_cost_zone_buy_intent * mf_cost_zone_buy_intent_weight) - \
                                   (norm_mf_cost_zone_sell_intent * mf_cost_zone_sell_intent_weight)
        commanders_resolve_score = commanders_resolve_score.clip(-1, 1)
        # if is_debug_enabled and probe_ts and probe_ts in df.index:
        #     print(f"      [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 指挥官决心计算 ---")
        #     print(f"        成本优势得分 (advantage_score): {advantage_score.loc[probe_ts]:.4f}")
        #     print(f"        控制坚实度得分 (solidity_score): {solidity_score.loc[probe_ts]:.4f}")
        #     print(f"        信念斜率得分 (intent_score): {intent_score.loc[probe_ts]:.4f}")
        #     print(f"        主力成本区买入意图归一化 (norm_mf_cost_zone_buy_intent): {norm_mf_cost_zone_buy_intent.loc[probe_ts]:.4f}")
        #     print(f"        主力成本区卖出意图归一化 (norm_mf_cost_zone_sell_intent): {norm_mf_cost_zone_sell_intent.loc[probe_ts]:.4f}")
        #     print(f"        指挥官决心得分 (commanders_resolve_score): {commanders_resolve_score.loc[probe_ts]:.4f}")
        # 2.1 诡道博弈深度融合与情境调制
        norm_deception_index = utils.get_adaptive_mtf_normalized_bipolar_score(deception_index_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_wash_trade_intensity = utils.get_adaptive_mtf_normalized_score(wash_trade_intensity_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_main_force_conviction = utils.get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_chip_health = utils.get_adaptive_mtf_normalized_score(chip_health_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_deception_lure_long = utils.get_adaptive_mtf_normalized_score(deception_lure_long_intensity_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_deception_lure_short = utils.get_adaptive_mtf_normalized_score(deception_lure_short_intensity_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        # --- Numba优化区域：deception_modulator ---
        if deception_context_mod_enabled:
            deception_modulator_values = _numba_calculate_deception_modulator_core(
                norm_deception_index.values,
                norm_deception_lure_long.values,
                norm_deception_lure_short.values,
                norm_wash_trade_intensity.values,
                norm_main_force_conviction.values,
                norm_chip_health.values,
                deception_conviction_threshold,
                deception_health_threshold,
                deception_boost_factor,
                deception_penalty_factor,
                wash_trade_penalty_factor,
                deception_lure_long_penalty_factor,
                deception_lure_short_boost_factor
            )
            deception_modulator = pd.Series(deception_modulator_values, index=df_index, dtype=np.float32)
        else:
            deception_modulator = pd.Series(1.0, index=df_index, dtype=np.float32)
        # --- Numba优化区域结束 ---
        commanders_resolve_score = commanders_resolve_score * deception_modulator.pow(np.sign(commanders_resolve_score))
        # if is_debug_enabled and probe_ts and probe_ts in df.index:
        #     print(f"      [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 诡道调制计算 ---")
        #     print(f"        欺骗指数归一化 (norm_deception_index): {norm_deception_index.loc[probe_ts]:.4f}")
        #     print(f"        对倒强度归一化 (norm_wash_trade_intensity): {norm_wash_trade_intensity.loc[probe_ts]:.4f}")
        #     print(f"        主力信念归一化 (norm_main_force_conviction): {norm_main_force_conviction.loc[probe_ts]:.4f}")
        #     print(f"        筹码健康度归一化 (norm_chip_health): {norm_chip_health.loc[probe_ts]:.4f}")
        #     print(f"        诱多强度归一化 (norm_deception_lure_long): {norm_deception_lure_long.loc[probe_ts]:.4f}")
        #     print(f"        诱空强度归一化 (norm_deception_lure_short): {norm_deception_lure_short.loc[probe_ts]:.4f}")
        #     print(f"        欺骗调制器 (deception_modulator): {deception_modulator.loc[probe_ts]:.4f}")
        #     print(f"        调制后指挥官决心得分 (commanders_resolve_score): {commanders_resolve_score.loc[probe_ts]:.4f}")
        # 3. 战场控制 (Battlefield Control)
        cleansing_score = utils.get_adaptive_mtf_normalized_bipolar_score(cleansing_efficiency_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        peak_solidity_score = utils.get_adaptive_mtf_normalized_bipolar_score(dominant_peak_solidity_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        battlefield_control_score = ((cleansing_score.add(1)/2) * (peak_solidity_score.add(1)/2)).pow(0.5) * 2 - 1
        # if is_debug_enabled and probe_ts and probe_ts in df.index:
        #     print(f"      [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 战场控制计算 ---")
        #     print(f"        清洗效率得分 (cleansing_score): {cleansing_score.loc[probe_ts]:.4f}")
        #     print(f"        峰值坚实度得分 (peak_solidity_score): {peak_solidity_score.loc[probe_ts]:.4f}")
        #     print(f"        战场控制得分 (battlefield_control_score): {battlefield_control_score.loc[probe_ts]:.4f}")
        # 4. 基础战略态势 (Base Strategic Posture)
        norm_covert_distribution_signal = utils.get_adaptive_mtf_normalized_score(covert_distribution_signal_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        # 在进行几何平均之前，对 commanders_resolve_score 进行裁剪，确保其在 [-1, 1] 范围内
        commanders_resolve_score_clipped = commanders_resolve_score.clip(-1, 1)
        base_strategic_posture_score = (
            (commanders_resolve_score_clipped.add(1)/2).pow(0.5) *
            (formation_deployment_score.add(1)/2).pow(0.3) *
            (battlefield_control_score.add(1)/2).pow(0.2)
        ).pow(1/(0.5+0.3+0.2)) * 2 - 1
        base_strategic_posture_score = base_strategic_posture_score * (1 - norm_covert_distribution_signal * covert_distribution_penalty_factor)
        base_strategic_posture_score = base_strategic_posture_score.clip(-1, 1)
        # if is_debug_enabled and probe_ts and probe_ts in df.index:
        #     print(f"      [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 基础战略态势计算 ---")
        #     print(f"        裁剪后指挥官决心得分 (commanders_resolve_score_clipped): {commanders_resolve_score_clipped.loc[probe_ts]:.4f}")
        #     print(f"        隐蔽派发信号归一化 (norm_covert_distribution_signal): {norm_covert_distribution_signal.loc[probe_ts]:.4f}")
        #     print(f"        基础战略态势得分 (base_strategic_posture_score): {base_strategic_posture_score.loc[probe_ts]:.4f}")
        # 5. 维度间非线性互动增强
        if inter_dimension_interaction_enabled:
            # --- Numba优化区域：synergy_factor ---
            synergy_factor_values = _numba_calculate_synergy_factor_core(
                formation_deployment_score.values,
                commanders_resolve_score.values,
                battlefield_control_score.values,
                synergy_bonus_factor,
                conflict_penalty_factor
            )
            synergy_factor = pd.Series(synergy_factor_values, index=df_index, dtype=np.float32)
            # --- Numba优化区域结束 ---
            # 注意：这里对 base_strategic_posture_score 进行 tanh 激活，它应该在 [-1, 1] 范围内
            base_strategic_posture_score = np.tanh(base_strategic_posture_score + synergy_factor)
            # if is_debug_enabled and probe_ts and probe_ts in df.index:
            #     print(f"      [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 维度互动增强 ---")
            #     print(f"        协同因子 (synergy_factor): {synergy_factor.loc[probe_ts]:.4f}")
            #     print(f"        互动增强后基础战略态势得分 (base_strategic_posture_score): {base_strategic_posture_score.loc[probe_ts]:.4f}")
        # 6. 速度与加速度融合
        smoothed_base_score = base_strategic_posture_score.ewm(span=smoothing_ema_span, adjust=False).mean()
        velocity = smoothed_base_score.diff(1).fillna(0)
        acceleration = velocity.diff(1).fillna(0)
        norm_velocity = utils.get_adaptive_mtf_normalized_bipolar_score(velocity, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_acceleration = utils.get_adaptive_mtf_normalized_bipolar_score(acceleration, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        # if is_debug_enabled and probe_ts and probe_ts in df.index:
        #     print(f"      [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 速度与加速度融合 ---")
        #     print(f"        平滑基础得分 (smoothed_base_score): {smoothed_base_score.loc[probe_ts]:.4f}")
        #     print(f"        速度 (velocity): {velocity.loc[probe_ts]:.4f}")
        #     print(f"        加速度 (acceleration): {acceleration.loc[probe_ts]:.4f}")
        #     print(f"        归一化速度 (norm_velocity): {norm_velocity.loc[probe_ts]:.4f}")
        #     print(f"        归一化加速度 (norm_acceleration): {norm_acceleration.loc[probe_ts]:.4f}")
        # 7. 动态权重自适应
        dynamic_base_weight = pd.Series(dynamic_fusion_weights_base.get('base_score', 0.6), index=df_index)
        dynamic_velocity_weight = pd.Series(dynamic_fusion_weights_base.get('velocity', 0.2), index=df_index)
        dynamic_acceleration_weight = pd.Series(dynamic_fusion_weights_base.get('acceleration', 0.2), index=df_index)
        norm_volatility_instability = utils.get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_chip_health_slope = utils.get_adaptive_mtf_normalized_bipolar_score(chip_health_slope_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        mod_factor = (norm_volatility_instability * dynamic_weight_sensitivity_volatility) - \
                     (norm_chip_health_slope.clip(upper=0).abs() * dynamic_weight_sensitivity_health_slope)
        dynamic_base_weight = dynamic_base_weight * (1 - mod_factor)
        dynamic_velocity_weight = dynamic_velocity_weight * (1 + mod_factor * 0.5)
        dynamic_acceleration_weight = dynamic_acceleration_weight * (1 + mod_factor * 0.5)
        sum_dynamic_weights = dynamic_base_weight + dynamic_velocity_weight + dynamic_acceleration_weight
        # 避免除以零，如果总权重为0，则保持原始权重比例
        sum_dynamic_weights = sum_dynamic_weights.replace(0, 1.0)
        # 归一化所有动态权重
        dynamic_base_weight = dynamic_base_weight / sum_dynamic_weights
        dynamic_velocity_weight = dynamic_velocity_weight / sum_dynamic_weights
        dynamic_acceleration_weight = dynamic_acceleration_weight / sum_dynamic_weights
        # 8. 最终战略态势得分融合
        final_strategic_posture_score = (
            smoothed_base_score * dynamic_base_weight +
            norm_velocity * dynamic_velocity_weight +
            norm_acceleration * dynamic_acceleration_weight
        ).clip(-1, 1).fillna(0.0).astype(np.float32)
        # if is_debug_enabled and probe_ts and probe_ts in df.index:
        #     print(f"      [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 动态权重融合 ---")
        #     print(f"        动态基础权重 (dynamic_base_weight): {dynamic_base_weight.loc[probe_ts]:.4f}")
        #     print(f"        动态速度权重 (dynamic_velocity_weight): {dynamic_velocity_weight.loc[probe_ts]:.4f}")
        #     print(f"        动态加速度权重 (dynamic_acceleration_weight): {dynamic_acceleration_weight.loc[probe_ts]:.4f}")
        #     print(f"        最终战略态势得分 (final_strategic_posture_score): {final_strategic_posture_score.loc[probe_ts]:.4f}")
        return final_strategic_posture_score

    def _diagnose_battlefield_geography(self, df: pd.DataFrame) -> pd.Series:
        """
        【V9.2 · Numba优化版】诊断筹码的战场地形，旨在提供一个双极的、具备诡道过滤和情境自适应能力的信号。
        - 核心优化: 将deception_filter_factor的计算逻辑迁移至Numba加速的辅助函数。
        - 核心升级1: 核心地形优势量化：重新定义地形优势为“支撑强度 - 阻力强度”，直接输出双极分数 [-1, 1]，正值代表地形有利，负值代表地形不利。
        - 核心升级2: 最小阻力路径动态调制：路径效率（真空区大小与穿越效率）不再简单相乘，而是作为非线性调制因子，放大或削弱核心地形优势。
        - 核心升级3: 动态演化趋势强化：地形趋势变化（支撑与阻力斜率之差）作为乘数，对地形优势进行非线性强化，引入前瞻性。
        - 核心升级4: 诡道地形过滤与惩罚：引入欺骗指数和筹码故障幅度作为诡道因子，对地形优势进行过滤和惩罚，例如在有利地形伴随诱多或虚假支撑时进行惩罚，在不利地形伴随诱空洗盘或虚假阻力时进行缓解。
        - 核心升级5: 情境感知与自适应权重：引入筹码健康度、筹码波动不稳定性等情境因子，动态调整各维度的融合权重，使模型在不同市场环境下自适应地调整对地形特征的关注重点。
        - 核心升级6: 新增筹码指标整合：
            - 向上/向下脉冲强度 (`upward_impulse_strength_D`, `downward_impulse_strength_D`) 增强支撑/阻力强度。
            - 主力成本区买卖意图 (`mf_cost_zone_buy_intent_D`, `mf_cost_zone_sell_intent_D`) 进一步强化支撑/阻力。
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        method_name = "_diagnose_battlefield_geography"
        df_index = df.index
        p_conf = self.chip_ultimate_params
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        parsed_tf_data = utils._parse_tf_weights(tf_weights)
        bg_params = get_param_value(p_conf.get('battlefield_geography_params'), {})
        path_efficiency_mod_factor = get_param_value(bg_params.get('path_efficiency_mod_factor'), 0.5)
        path_efficiency_non_linear_exponent = get_param_value(bg_params.get('path_efficiency_non_linear_exponent'), 1.5)
        dynamic_evolution_mod_factor = get_param_value(bg_params.get('dynamic_evolution_mod_factor'), 0.3)
        dynamic_evolution_non_linear_exponent = get_param_value(bg_params.get('dynamic_evolution_non_linear_exponent'), 1.2)
        deception_signal_name = get_param_value(bg_params.get('deception_signal'), 'deception_index_D')
        chip_fault_signal_name = get_param_value(bg_params.get('chip_fault_signal'), 'chip_fault_magnitude_D')
        deception_penalty_sensitivity = get_param_value(bg_params.get('deception_penalty_sensitivity'), 0.6)
        chip_fault_penalty_sensitivity = get_param_value(bg_params.get('chip_fault_penalty_sensitivity'), 0.4)
        deception_mitigation_sensitivity = get_param_value(bg_params.get('deception_mitigation_sensitivity'), 0.3)
        context_modulator_signal_1_name = get_param_value(bg_params.get('context_modulator_signal_1'), 'chip_health_score_D')
        context_modulator_signal_2_name = get_param_value(bg_params.get('context_modulator_signal_2'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        context_modulator_sensitivity_health = get_param_value(bg_params.get('context_modulator_sensitivity_health'), 0.4)
        context_modulator_sensitivity_volatility = get_param_value(bg_params.get('context_modulator_sensitivity_volatility'), 0.3)
        upward_impulse_strength_weight = get_param_value(bg_params.get('upward_impulse_strength_weight'), 0.1)
        downward_impulse_strength_weight = get_param_value(bg_params.get('downward_impulse_strength_weight'), 0.1)
        mf_cost_zone_buy_intent_weight = get_param_value(bg_params.get('mf_cost_zone_buy_intent_weight'), 0.1)
        mf_cost_zone_sell_intent_weight = get_param_value(bg_params.get('mf_cost_zone_sell_intent_weight'), 0.1)
        required_signals = [
            'dominant_peak_solidity_D', 'support_validation_strength_D', 'chip_fault_blockage_ratio_D',
            'pressure_rejection_strength_D', 'vacuum_zone_magnitude_D', 'vacuum_traversal_efficiency_D',
            'SLOPE_5_support_validation_strength_D', 'SLOPE_5_pressure_rejection_strength_D',
            deception_signal_name, chip_fault_signal_name,
            context_modulator_signal_1_name, context_modulator_signal_2_name,
            'upward_impulse_strength_D', 'downward_impulse_strength_D',
            'mf_cost_zone_buy_intent_D', 'mf_cost_zone_sell_intent_D'
        ]
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index)
        signals_data = self._get_all_required_signals(df, required_signals, method_name)
        is_debug_enabled = self.should_probe
        probe_ts = None
        if is_debug_enabled and self.probe_dates_set:
            for date in reversed(df_index):
                if date.date() in self.probe_dates_set:
                    probe_ts = date
                    break
        debug_info_tuple = False
        peak_solidity = signals_data['dominant_peak_solidity_D']
        support_validation = signals_data['support_validation_strength_D']
        fault_blockage = signals_data['chip_fault_blockage_ratio_D']
        pressure_rejection = signals_data['pressure_rejection_strength_D']
        vacuum_magnitude = signals_data['vacuum_zone_magnitude_D']
        vacuum_efficiency = signals_data['vacuum_traversal_efficiency_D']
        support_trend_raw = signals_data['SLOPE_5_support_validation_strength_D']
        resistance_trend_raw = signals_data['SLOPE_5_pressure_rejection_strength_D']
        deception_raw = signals_data[deception_signal_name]
        chip_fault_raw = signals_data[chip_fault_signal_name]
        chip_health_raw = signals_data[context_modulator_signal_1_name]
        volatility_instability_raw = signals_data[context_modulator_signal_2_name]
        upward_impulse_strength_raw = signals_data['upward_impulse_strength_D']
        downward_impulse_strength_raw = signals_data['downward_impulse_strength_D']
        mf_cost_zone_buy_intent_raw = signals_data['mf_cost_zone_buy_intent_D']
        mf_cost_zone_sell_intent_raw = signals_data['mf_cost_zone_sell_intent_D']
        solidity_score = utils.get_adaptive_mtf_normalized_score(peak_solidity, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        validation_score = utils.get_adaptive_mtf_normalized_score(support_validation, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        support_strength_score = (solidity_score * validation_score).pow(0.5)
        norm_upward_impulse_strength = utils.get_adaptive_mtf_normalized_score(upward_impulse_strength_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_mf_cost_zone_buy_intent = utils.get_adaptive_mtf_normalized_score(mf_cost_zone_buy_intent_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        support_strength_score = support_strength_score * (1 + norm_upward_impulse_strength * upward_impulse_strength_weight + \
                                                              norm_mf_cost_zone_buy_intent * mf_cost_zone_buy_intent_weight)
        support_strength_score = support_strength_score.clip(0, 1)
        blockage_score = utils.get_adaptive_mtf_normalized_score(fault_blockage, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        rejection_score = utils.get_adaptive_mtf_normalized_score(pressure_rejection, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        resistance_strength_score = (blockage_score * rejection_score).pow(0.5)
        norm_downward_impulse_strength = utils.get_adaptive_mtf_normalized_score(downward_impulse_strength_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_mf_cost_zone_sell_intent = utils.get_adaptive_mtf_normalized_score(mf_cost_zone_sell_intent_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        resistance_strength_score = resistance_strength_score * (1 + norm_downward_impulse_strength * downward_impulse_strength_weight + \
                                                                    norm_mf_cost_zone_sell_intent * mf_cost_zone_sell_intent_weight)
        resistance_strength_score = resistance_strength_score.clip(0, 1)
        base_terrain_advantage_score = support_strength_score - resistance_strength_score
        norm_vacuum_magnitude = utils.get_adaptive_mtf_normalized_score(vacuum_magnitude, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_vacuum_efficiency = utils.get_adaptive_mtf_normalized_score(vacuum_efficiency, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        path_efficiency = (norm_vacuum_magnitude * norm_vacuum_efficiency).pow(0.5)
        path_modulation_factor = (1 + path_efficiency * path_efficiency_mod_factor).pow(path_efficiency_non_linear_exponent)
        norm_support_trend = utils.get_adaptive_mtf_normalized_bipolar_score(support_trend_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_resistance_trend = utils.get_adaptive_mtf_normalized_bipolar_score(resistance_trend_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        terrain_advantage_change = norm_support_trend - norm_resistance_trend
        dynamic_evolution_modulator = (1 + terrain_advantage_change * dynamic_evolution_mod_factor).pow(dynamic_evolution_non_linear_exponent)
        dynamic_evolution_modulator = dynamic_evolution_modulator.clip(0.5, 1.5)
        norm_deception = utils.get_adaptive_mtf_normalized_bipolar_score(deception_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_chip_fault = utils.get_adaptive_mtf_normalized_bipolar_score(chip_fault_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        # --- Numba优化区域：deception_filter_factor ---
        deception_filter_factor_values = _numba_calculate_deception_filter_factor_core(
            base_terrain_advantage_score.values,
            norm_deception.values,
            norm_chip_fault.values,
            deception_penalty_sensitivity,
            chip_fault_penalty_sensitivity,
            deception_mitigation_sensitivity
        )
        deception_filter_factor = pd.Series(deception_filter_factor_values, index=df_index, dtype=np.float32)
        # --- Numba优化区域结束 ---
        norm_chip_health = utils.get_adaptive_mtf_normalized_score(chip_health_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_volatility_instability = utils.get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        context_modulator = (
            (1 + norm_chip_health * context_modulator_sensitivity_health) *
            (1 + norm_volatility_instability * context_modulator_sensitivity_volatility)
        ).clip(0.5, 1.5)
        final_score = base_terrain_advantage_score * path_modulation_factor * dynamic_evolution_modulator * deception_filter_factor * context_modulator
        final_score = final_score.clip(-1, 1).fillna(0.0).astype(np.float32)
        return final_score

    def _diagnose_axiom_holder_sentiment(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V9.2 · Numba优化版】筹码公理三：诊断“持仓信念韧性”
        - 核心优化: 将conviction_base_unipolar的欺骗调制逻辑迁移至Numba加速的辅助函数。
        - 核心升级1: 纯筹码指标强化。严格遵循纯筹码原则，将全局市场情绪替换为筹码主力信念，并新增恐慌买入吸收贡献、低吸吸收强度等纯筹码指标。
        - 核心升级2: 诡道反噬机制深化。在“杂质削弱”维度中，引入诱多/诱空欺骗强度，根据主力信念动态放大或削弱杂质影响，实现“诡道反噬”。
        - 核心升级3: 韧性重构机制引入。在“杂质削弱”维度中，引入筹码健康度斜率和结构性紧张指数，动态评估筹码结构在压力下的自我修复或恶化加速，实现“韧性重构”。
        - 核心升级4: 压力测试精细化。在V8.2基础上，新增恐慌买入吸收贡献和低吸吸收强度，更全面评估主力在恐慌和下跌中的承接能力。
        - 核心升级5: 全局情境调制器优化。将全局市场情绪替换为筹码主力信念，使情境调制更聚焦于筹码层面。
        - 核心修复: 修正 `panic_source_score` 对恐慌动态的判断，当散户恐慌快速消退时，降低其贡献。
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        method_name = "_diagnose_axiom_holder_sentiment"
        df_index = df.index
        required_signals = [
            'winner_stability_index_D', 'loser_pain_index_D', 'active_buying_support_D',
            'support_validation_strength_D', 'winner_concentration_90pct_D',
            'winner_profit_margin_avg_D', 'capitulation_absorption_index_D',
            'SLOPE_55_winner_concentration_90pct_D',
            'chip_fatigue_index_D', 'chip_fault_magnitude_D', 'chip_health_score_D',
            'total_winner_rate_D', 'total_loser_rate_D', 'winner_loser_momentum_D',
            'SLOPE_5_winner_stability_index_D', 'ACCEL_5_loser_pain_index_D', 'SLOPE_5_winner_loser_momentum_D',
            'opening_gap_defense_strength_D', 'control_solidity_index_D', 'order_book_clearing_rate_D',
            'micro_price_impact_asymmetry_D', 'SLOPE_5_support_validation_strength_D',
            'ACCEL_5_capitulation_absorption_index_D', 'SLOPE_5_active_buying_support_D',
            'upper_shadow_selling_pressure_D', 'rally_distribution_pressure_D', 'retail_fomo_premium_index_D',
            'SLOPE_5_winner_profit_margin_avg_D', 'ACCEL_5_retail_fomo_premium_index_D',
            'deception_index_D', 'wash_trade_intensity_D',
            'VOLATILITY_INSTABILITY_INDEX_21d_D', 'flow_credibility_index_D',
            'main_force_conviction_index_D',
            'conviction_flow_buy_intensity_D', 'conviction_flow_sell_intensity_D',
            'deception_lure_long_intensity_D', 'deception_lure_short_intensity_D',
            'panic_buy_absorption_contribution_D', 'dip_buy_absorption_strength_D',
            'structural_tension_index_D', 'SLOPE_5_chip_health_score_D',
            'SLOPE_5_retail_panic_surrender_index_D'
        ]
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index)
        signals_data = self._get_all_required_signals(df, required_signals, method_name)
        p_conf = self.chip_ultimate_params
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        parsed_tf_data = utils._parse_tf_weights(tf_weights)
        holder_sentiment_params = get_param_value(p_conf.get('holder_sentiment_params'), {})
        sentiment_trend_modulator_signal_name = get_param_value(holder_sentiment_params.get('sentiment_trend_modulator_signal_name'), 'SLOPE_55_winner_concentration_90pct_D')
        sentiment_trend_mod_factor = get_param_value(holder_sentiment_params.get('sentiment_trend_mod_factor'), 0.5)
        panic_reward_modulator_signal_name = get_param_value(holder_sentiment_params.get('panic_reward_modulator_signal_name'), 'chip_fatigue_index_D')
        panic_reward_mod_tanh_factor = get_param_value(holder_sentiment_params.get('panic_reward_mod_tanh_factor'), 1.0)
        panic_reward_mod_factor = get_param_value(holder_sentiment_params.get('panic_reward_mod_factor'), 1.0)
        capitulation_base_reward_multiplier = get_param_value(holder_sentiment_params.get('capitulation_base_reward_multiplier'), 0.3)
        impurity_non_linear_enabled = get_param_value(holder_sentiment_params.get('impurity_non_linear_enabled'), True)
        fomo_tanh_factor = get_param_value(holder_sentiment_params.get('fomo_tanh_factor'), 1.0)
        fomo_sentiment_sensitivity = get_param_value(holder_sentiment_params.get('fomo_sentiment_sensitivity'), 0.5)
        profit_taking_tanh_factor = get_param_value(holder_sentiment_params.get('profit_taking_tanh_factor'), 1.0)
        profit_taking_sentiment_sensitivity = get_param_value(holder_sentiment_params.get('profit_taking_sentiment_sensitivity'), 0.5)
        deception_factor_enabled = get_param_value(holder_sentiment_params.get('deception_factor_enabled'), True)
        deception_signal_name = get_param_value(holder_sentiment_params.get('deception_signal_name'), 'chip_fault_magnitude_D')
        deception_impact_factor = get_param_value(holder_sentiment_params.get('deception_impact_factor'), 0.2)
        positive_deception_penalty_enabled = get_param_value(holder_sentiment_params.get('positive_deception_penalty_enabled'), True)
        positive_deception_impact_factor = get_param_value(holder_sentiment_params.get('positive_deception_impact_factor'), 0.15)
        impurity_context_modulation_enabled = get_param_value(holder_sentiment_params.get('impurity_context_modulation_enabled'), True)
        impurity_context_modulator_signal_name = get_param_value(holder_sentiment_params.get('impurity_context_modulator_signal_name'), 'chip_health_score_D')
        impurity_context_overbought_amp_factor = get_param_value(holder_sentiment_params.get('impurity_context_overbought_amp_factor'), 0.5)
        impurity_context_oversold_damp_factor = get_param_value(holder_sentiment_params.get('impurity_context_oversold_damp_factor'), 0.2)
        dynamic_fusion_enabled = get_param_value(holder_sentiment_params.get('dynamic_fusion_enabled'), True)
        min_pressure_weight = get_param_value(holder_sentiment_params.get('min_pressure_weight'), 0.3)
        max_pressure_weight = get_param_value(holder_sentiment_params.get('max_pressure_weight'), 0.7)
        impurity_fusion_exponent_base = get_param_value(holder_sentiment_params.get('impurity_fusion_exponent_base'), 0.7)
        impurity_fusion_exponent_sensitivity = get_param_value(holder_sentiment_params.get('impurity_fusion_exponent_sensitivity'), 0.5)
        fomo_concentration_optimal_target = get_param_value(holder_sentiment_params.get('fomo_concentration_optimal_target'), 0.5)
        profit_taking_threshold = get_param_value(holder_sentiment_params.get('profit_taking_threshold'), 5.0)
        belief_core_weights = {k: v for k, v in get_param_value(holder_sentiment_params.get('belief_core_weights'), {}).items() if isinstance(v, (int, float))}
        pressure_test_weights = {k: v for k, v in get_param_value(holder_sentiment_params.get('pressure_test_weights'), {}).items() if isinstance(v, (int, float))}
        impurity_weights = {k: v for k, v in get_param_value(holder_sentiment_params.get('impurity_weights'), {}).items() if isinstance(v, (int, float))}
        deception_modulator_params = get_param_value(holder_sentiment_params.get('deception_modulator_params'), {'boost_factor': 0.6, 'penalty_factor': 0.4, 'conviction_threshold': 0.2, 'deception_index_weight': 0.5})
        deception_modulator_weights = {k: v for k, v in get_param_value(holder_sentiment_params.get('deception_modulator_weights'), {}).items() if isinstance(v, (int, float))}
        context_modulator_weights = {k: v for k, v in get_param_value(holder_sentiment_params.get('context_modulator_weights'), {}).items() if isinstance(v, (int, float))}
        impurity_deception_mod_enabled = get_param_value(holder_sentiment_params.get('impurity_deception_mod_enabled'), True)
        deception_lure_long_impurity_amp_factor = get_param_value(holder_sentiment_params.get('deception_lure_long_impurity_amp_factor'), 0.3)
        deception_lure_short_impurity_damp_factor = get_param_value(holder_sentiment_params.get('deception_lure_short_impurity_damp_factor'), 0.2)
        impurity_resilience_mod_enabled = get_param_value(holder_sentiment_params.get('impurity_resilience_mod_enabled'), True)
        chip_health_slope_impurity_damp_factor = get_param_value(holder_sentiment_params.get('chip_health_slope_impurity_damp_factor'), 0.2)
        structural_tension_impurity_amp_factor = get_param_value(holder_sentiment_params.get('structural_tension_impurity_amp_factor'), 0.2)
        global_context_modulator_enabled = get_param_value(holder_sentiment_params.get('global_context_modulator_enabled'), True)
        global_context_sensitivity_health = get_param_value(holder_sentiment_params.get('global_context_sensitivity_health'), 0.5)
        global_context_sensitivity_conviction = get_param_value(holder_sentiment_params.get('global_context_sensitivity_conviction'), 0.3)
        panic_slope_dampening_enabled = get_param_value(holder_sentiment_params.get('panic_slope_dampening_enabled'), True)
        panic_slope_dampening_sensitivity = get_param_value(holder_sentiment_params.get('panic_slope_dampening_sensitivity'), 0.5)
        is_debug_enabled = self.should_probe
        probe_ts = None
        if is_debug_enabled and self.probe_dates_set:
            for date in reversed(df_index):
                if date.date() in self.probe_dates_set:
                    probe_ts = date
                    break
        debug_info_tuple = False
        chip_health_raw = signals_data['chip_health_score_D']
        winner_stability = signals_data['winner_stability_index_D']
        loser_pain = signals_data['loser_pain_index_D']
        total_winner_rate_raw = signals_data['total_winner_rate_D']
        total_loser_rate_raw = signals_data['total_loser_rate_D']
        winner_loser_momentum_raw = signals_data['winner_loser_momentum_D']
        slope_5_winner_stability_raw = signals_data['SLOPE_5_winner_stability_index_D']
        accel_5_loser_pain_raw = signals_data['ACCEL_5_loser_pain_index_D']
        slope_5_winner_loser_momentum_raw = signals_data['SLOPE_5_winner_loser_momentum_D']
        absorption_power = signals_data['active_buying_support_D']
        defense_intent = signals_data['support_validation_strength_D']
        capitulation_absorption = signals_data['capitulation_absorption_index_D']
        opening_gap_defense_strength_raw = signals_data['opening_gap_defense_strength_D']
        control_solidity_raw = signals_data['control_solidity_index_D']
        order_book_clearing_rate_raw = signals_data['order_book_clearing_rate_D']
        micro_price_impact_asymmetry_raw = signals_data['micro_price_impact_asymmetry_D']
        slope_5_support_validation_raw = signals_data['SLOPE_5_support_validation_strength_D']
        accel_5_capitulation_absorption_raw = signals_data['ACCEL_5_capitulation_absorption_index_D']
        slope_5_active_buying_support_raw = signals_data['SLOPE_5_active_buying_support_D']
        fomo_index_raw = signals_data['winner_concentration_90pct_D']
        profit_taking_quality_raw = signals_data['winner_profit_margin_avg_D']
        upper_shadow_selling_pressure_raw = signals_data['upper_shadow_selling_pressure_D']
        rally_distribution_pressure_raw = signals_data['rally_distribution_pressure_D']
        retail_fomo_premium_raw = signals_data['retail_fomo_premium_index_D']
        slope_5_winner_profit_margin_raw = signals_data['SLOPE_5_winner_profit_margin_avg_D']
        accel_5_retail_fomo_premium_raw = signals_data['ACCEL_5_retail_fomo_premium_index_D']
        chip_fatigue_raw = signals_data['chip_fatigue_index_D']
        deception_raw = signals_data[deception_signal_name]
        deception_index_raw = signals_data['deception_index_D']
        wash_trade_intensity_raw = signals_data['wash_trade_intensity_D']
        main_force_conviction_raw = signals_data['main_force_conviction_index_D']
        volatility_instability_raw = signals_data['VOLATILITY_INSTABILITY_INDEX_21d_D']
        flow_credibility_raw = signals_data['flow_credibility_index_D']
        conviction_flow_buy_intensity_raw = signals_data['conviction_flow_buy_intensity_D']
        conviction_flow_sell_intensity_raw = signals_data['conviction_flow_sell_intensity_D']
        deception_lure_long_intensity_raw = signals_data['deception_lure_long_intensity_D']
        deception_lure_short_intensity_raw = signals_data['deception_lure_short_intensity_D']
        panic_buy_absorption_contribution_raw = signals_data['panic_buy_absorption_contribution_D']
        dip_buy_absorption_strength_raw = signals_data['dip_buy_absorption_strength_D']
        structural_tension_raw = signals_data['structural_tension_index_D']
        slope_5_chip_health_raw = signals_data['SLOPE_5_chip_health_score_D']
        slope_5_retail_panic_surrender_raw = signals_data['SLOPE_5_retail_panic_surrender_index_D']
        norm_winner_stability = utils.get_adaptive_mtf_normalized_bipolar_score(winner_stability, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_loser_pain = utils.get_adaptive_mtf_normalized_bipolar_score(loser_pain, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_total_winner_rate = utils.get_adaptive_mtf_normalized_score(total_winner_rate_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_total_loser_rate = utils.get_adaptive_mtf_normalized_score(total_loser_rate_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_winner_loser_momentum = utils.get_adaptive_mtf_normalized_bipolar_score(winner_loser_momentum_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_slope_5_winner_stability = utils.get_adaptive_mtf_normalized_bipolar_score(slope_5_winner_stability_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_accel_5_loser_pain = utils.get_adaptive_mtf_normalized_bipolar_score(accel_5_loser_pain_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_slope_5_winner_loser_momentum = utils.get_adaptive_mtf_normalized_bipolar_score(slope_5_winner_loser_momentum_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_conviction_flow_buy_intensity = utils.get_adaptive_mtf_normalized_score(conviction_flow_buy_intensity_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_conviction_flow_sell_intensity = utils.get_adaptive_mtf_normalized_score(conviction_flow_sell_intensity_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        sentiment_trend_raw = signals_data[sentiment_trend_modulator_signal_name]
        norm_sentiment_trend = utils.get_adaptive_mtf_normalized_score(sentiment_trend_raw, df_index, tf_weights=tf_weights, ascending=True, debug_info=False, _parsed_tf_data=parsed_tf_data)
        x = (norm_sentiment_trend * sentiment_trend_mod_factor).clip(-0.4, 0.4)
        dynamic_stability_weight = 0.5 + x
        dynamic_pain_weight = 0.5 - x
        belief_core_numeric_weights = {k: v for k, v in belief_core_weights.items() if isinstance(v, (int, float))}
        total_belief_core_weight = sum(belief_core_numeric_weights.values())
        belief_core_components = {
            'winner_stability': (norm_winner_stability + 1) / 2,
            'loser_pain': (norm_loser_pain + 1) / 2,
            'total_winner_rate': norm_total_winner_rate,
            'total_loser_rate': (1 - norm_total_loser_rate),
            'winner_loser_momentum': (norm_winner_loser_momentum + 1) / 2,
            'winner_stability_slope': (norm_slope_5_winner_stability + 1) / 2,
            'loser_pain_accel': (norm_accel_5_loser_pain + 1) / 2,
            'winner_loser_momentum_slope': (norm_slope_5_winner_loser_momentum + 1) / 2,
            'conviction_flow_buy': norm_conviction_flow_buy_intensity,
            'conviction_flow_sell': (1 - norm_conviction_flow_sell_intensity)
        }
        belief_core_component_weights = {
            'winner_stability': belief_core_numeric_weights.get('winner_stability', 0.15) * dynamic_stability_weight,
            'loser_pain': belief_core_numeric_weights.get('loser_pain', 0.15) * dynamic_pain_weight,
            'total_winner_rate': belief_core_numeric_weights.get('total_winner_rate', 0.08),
            'total_loser_rate': belief_core_numeric_weights.get('total_loser_rate', 0.08),
            'winner_loser_momentum': belief_core_numeric_weights.get('winner_loser_momentum', 0.08),
            'winner_stability_slope': belief_core_numeric_weights.get('winner_stability_slope', 0.08),
            'loser_pain_accel': belief_core_numeric_weights.get('loser_pain_accel', 0.08),
            'winner_loser_momentum_slope': belief_core_numeric_weights.get('winner_loser_momentum_slope', 0.08),
            'conviction_flow_buy': belief_core_numeric_weights.get('conviction_flow_buy', 0.1),
            'conviction_flow_sell': belief_core_numeric_weights.get('conviction_flow_sell', 0.1)
        }
        belief_core_score_unipolar = utils._robust_geometric_mean(belief_core_components, belief_core_component_weights, df_index)
        belief_core_score = (belief_core_score_unipolar * 2 - 1).clip(-1, 1)
        norm_absorption_power = utils.get_adaptive_mtf_normalized_bipolar_score(absorption_power, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_defense_intent = utils.get_adaptive_mtf_normalized_bipolar_score(defense_intent, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_capitulation_absorption = utils.get_adaptive_mtf_normalized_score(capitulation_absorption, df_index, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_opening_gap_defense_strength = utils.get_adaptive_mtf_normalized_score(opening_gap_defense_strength_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_control_solidity = utils.get_adaptive_mtf_normalized_score(control_solidity_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_order_book_clearing_rate = utils.get_adaptive_mtf_normalized_score(order_book_clearing_rate_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_micro_price_impact_asymmetry = utils.get_adaptive_mtf_normalized_score(micro_price_impact_asymmetry_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_slope_5_support_validation = utils.get_adaptive_mtf_normalized_score(slope_5_support_validation_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_accel_5_capitulation_absorption = utils.get_adaptive_mtf_normalized_score(accel_5_capitulation_absorption_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_slope_5_active_buying_support = utils.get_adaptive_mtf_normalized_score(slope_5_active_buying_support_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_panic_buy_absorption_contribution = utils.get_adaptive_mtf_normalized_score(panic_buy_absorption_contribution_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_dip_buy_absorption_strength = utils.get_adaptive_mtf_normalized_score(dip_buy_absorption_strength_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        base_pressure_test_numeric_weights = {k: v for k, v in pressure_test_weights.items() if isinstance(v, (int, float))}
        total_base_pressure_test_weight = sum(base_pressure_test_numeric_weights.values())
        pressure_test_components = {
            'absorption_power': (norm_absorption_power + 1) / 2,
            'defense_intent': (norm_defense_intent + 1) / 2,
            'capitulation_absorption': norm_capitulation_absorption,
            'opening_gap_defense_strength': norm_opening_gap_defense_strength,
            'control_solidity': norm_control_solidity,
            'order_book_clearing_rate': norm_order_book_clearing_rate,
            'micro_price_impact_asymmetry': norm_micro_price_impact_asymmetry,
            'support_validation_slope': norm_slope_5_support_validation,
            'capitulation_absorption_accel': norm_accel_5_capitulation_absorption,
            'active_buying_support_slope': norm_slope_5_active_buying_support,
            'panic_buy_absorption_contribution': norm_panic_buy_absorption_contribution,
            'dip_buy_absorption_strength': norm_dip_buy_absorption_strength
        }
        pressure_test_score_unipolar = utils._robust_geometric_mean(pressure_test_components, base_pressure_test_numeric_weights, df_index)
        base_pressure_score = (pressure_test_score_unipolar * 2 - 1).clip(-1, 1)
        panic_modulator_raw = signals_data[panic_reward_modulator_signal_name]
        normalized_panic_modulator = utils.get_adaptive_mtf_normalized_score(panic_modulator_raw, df_index, tf_weights=tf_weights, ascending=True, debug_info=False, _parsed_tf_data=parsed_tf_data)
        panic_reward_adjustment_factor = np.tanh(normalized_panic_modulator * panic_reward_mod_tanh_factor) * panic_reward_mod_factor
        dynamic_capitulation_reward_multiplier = capitulation_base_reward_multiplier * (1 + panic_reward_adjustment_factor)
        dynamic_capitulation_reward_multiplier = dynamic_capitulation_reward_multiplier.clip(0.1, 0.8)
        capitulation_bonus = norm_capitulation_absorption * dynamic_capitulation_reward_multiplier
        deception_impact = pd.Series(0.0, index=df_index)
        deception_raw = signals_data[deception_signal_name]
        if deception_factor_enabled:
            negative_deception = deception_raw.clip(upper=0).abs()
            normalized_negative_deception = utils.get_adaptive_mtf_normalized_score(negative_deception, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
            deception_impact = normalized_negative_deception * deception_impact_factor
        panic_slope_dampening_factor = pd.Series(1.0, index=df_index)
        if panic_slope_dampening_enabled:
            norm_panic_slope = utils.get_adaptive_mtf_normalized_bipolar_score(slope_5_retail_panic_surrender_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
            panic_slope_dampening_factor = (1 - norm_panic_slope.clip(upper=0).abs() * panic_slope_dampening_sensitivity).clip(0.1, 1.0)
        pressure_test_score = base_pressure_score * (1 + capitulation_bonus + deception_impact) * panic_slope_dampening_factor
        pressure_test_score = pressure_test_score.clip(-1, 1)
        s_belief_core = belief_core_score.add(1)/2
        s_pressure_test = pressure_test_score.add(1)/2
        dynamic_belief_core_weight = pd.Series(0.5, index=df_index)
        dynamic_pressure_test_weight = pd.Series(0.5, index=df_index)
        if dynamic_fusion_enabled:
            dynamic_pressure_test_weight = min_pressure_weight + (max_pressure_weight - min_pressure_weight) * normalized_panic_modulator
            dynamic_belief_core_weight = 1.0 - dynamic_pressure_test_weight
        conviction_base_unipolar = (s_belief_core.pow(dynamic_belief_core_weight) * s_pressure_test.pow(dynamic_pressure_test_weight))
        positive_deception_penalty = pd.Series(0.0, index=df_index)
        if positive_deception_penalty_enabled:
            positive_deception_raw = deception_raw.clip(lower=0)
            normalized_positive_deception = utils.get_adaptive_mtf_normalized_score(positive_deception_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
            positive_deception_penalty = normalized_positive_deception * positive_deception_impact_factor
            conviction_base_unipolar = conviction_base_unipolar * (1 - positive_deception_penalty)
            conviction_base_unipolar = conviction_base_unipolar.clip(0, 1)
        norm_deception_index_bipolar = utils.get_adaptive_mtf_normalized_bipolar_score(deception_index_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_wash_trade_intensity = utils.get_adaptive_mtf_normalized_score(wash_trade_intensity_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_main_force_conviction_bipolar = utils.get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        # --- Numba优化区域：conviction_base_unipolar的欺骗调制 ---
        # 从 deception_modulator_params 中获取 conviction_threshold
        conviction_threshold = get_param_value(deception_modulator_params.get('conviction_threshold'), 0.2) # 修复：确保conviction_threshold已定义
        conviction_base_unipolar_values = _numba_calculate_impurity_deception_modulator_core(
            conviction_base_unipolar.values,
            norm_deception_index_bipolar.values,
            norm_wash_trade_intensity.values,
            norm_main_force_conviction_bipolar.values,
            deception_modulator_weights.get('deception_index_boost', 0.5),
            deception_modulator_weights.get('deception_ll_penalty_weight', 0.5), # 修复：使用正确的键名
            deception_modulator_weights.get('wash_trade_penalty', 0.3),
            conviction_threshold # 修复：传递已定义的conviction_threshold
        )
        conviction_base_unipolar = pd.Series(conviction_base_unipolar_values, index=df_index, dtype=np.float32)
        # --- Numba优化区域结束 ---
        # 确保 fomo_index_raw 在使用前被定义
        fomo_index_raw = signals_data['winner_concentration_90pct_D']
        norm_fomo_deviation = utils.get_adaptive_mtf_normalized_score((fomo_index_raw - fomo_concentration_optimal_target).abs(), df_index, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        profit_taking_quality_thresholded = (profit_taking_quality_raw - profit_taking_threshold).clip(lower=0)
        norm_profit_taking_quality = utils.get_adaptive_mtf_normalized_score(profit_taking_quality_thresholded, df_index, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_upper_shadow_selling_pressure = utils.get_adaptive_mtf_normalized_score(upper_shadow_selling_pressure_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_rally_distribution_pressure = utils.get_adaptive_mtf_normalized_score(rally_distribution_pressure_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_retail_fomo_premium = utils.get_adaptive_mtf_normalized_score(retail_fomo_premium_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_slope_5_winner_profit_margin = utils.get_adaptive_mtf_normalized_score(slope_5_winner_profit_margin_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_accel_5_retail_fomo_premium = utils.get_adaptive_mtf_normalized_score(accel_5_retail_fomo_premium_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_deception_lure_long_intensity = utils.get_adaptive_mtf_normalized_score(deception_lure_long_intensity_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_deception_lure_short_intensity = utils.get_adaptive_mtf_normalized_score(deception_lure_short_intensity_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_slope_5_chip_health = utils.get_adaptive_mtf_normalized_bipolar_score(slope_5_chip_health_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_structural_tension = utils.get_adaptive_mtf_normalized_score(structural_tension_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        fomo_effect = pd.Series(0.0, index=df_index)
        profit_taking_effect = pd.Series(0.0, index=df_index)
        other_impurity_effect = pd.Series(0.0, index=df_index)
        final_impurity_effect = pd.Series(0.0, index=df_index)
        if impurity_non_linear_enabled:
            current_sentiment_strength = (conviction_base_unipolar * 2 - 1).abs()
            normalized_sentiment_strength = utils.normalize_score(current_sentiment_strength, df_index, windows=21, ascending=True)
            context_adjustment_factor = pd.Series(1.0, index=df_index)
            norm_volatility_instability = utils.get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
            norm_flow_credibility = utils.get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
            norm_chip_health_for_context = utils.get_adaptive_mtf_normalized_score(chip_health_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
            context_modulator_numeric_weights = {k: v for k, v in context_modulator_weights.items() if isinstance(v, (int, float))}
            total_context_modulator_weight = sum(context_modulator_numeric_weights.values())
            if total_context_modulator_weight > 0:
                fused_context_modulator = (
                    norm_volatility_instability.pow(context_modulator_numeric_weights.get('volatility_instability', 0.4)) *
                    norm_flow_credibility.pow(context_modulator_numeric_weights.get('flow_credibility', 0.3)) *
                    norm_chip_health_for_context.pow(context_modulator_numeric_weights.get('chip_health', 0.3))
                ).pow(1 / total_context_modulator_weight)
                context_adjustment_factor = context_adjustment_factor * (1 + (fused_context_modulator - 0.5) * 0.5)
            if impurity_context_modulation_enabled:
                context_modulator_raw = signals_data[impurity_context_modulator_signal_name]
                normalized_context_modulator = utils.get_adaptive_mtf_normalized_score(context_modulator_raw, df_index, tf_weights=tf_weights, ascending=True, debug_info=False, _parsed_tf_data=parsed_tf_data)
                overbought_mask = normalized_context_modulator > 0.7
                oversold_mask = normalized_context_modulator < 0.3
                context_adjustment_factor.loc[overbought_mask] = context_adjustment_factor.loc[overbought_mask] * (1 + (normalized_context_modulator.loc[overbought_mask] - 0.7) * impurity_context_overbought_amp_factor / 0.3)
                context_adjustment_factor.loc[oversold_mask] = context_adjustment_factor.loc[oversold_mask] * (1 - (0.3 - normalized_context_modulator.loc[oversold_mask]) * impurity_context_oversold_damp_factor / 0.3)
            dynamic_fomo_tanh_factor = fomo_tanh_factor * (1 + normalized_sentiment_strength * fomo_sentiment_sensitivity)
            dynamic_fomo_tanh_factor = dynamic_fomo_tanh_factor * context_adjustment_factor
            dynamic_fomo_tanh_factor = dynamic_fomo_tanh_factor.clip(0.5, 3.0)
            fomo_effect = np.tanh(norm_fomo_deviation * dynamic_fomo_tanh_factor)
            dynamic_profit_taking_tanh_factor = profit_taking_tanh_factor * (1 + normalized_sentiment_strength * profit_taking_sentiment_sensitivity)
            dynamic_profit_taking_tanh_factor = dynamic_profit_taking_tanh_factor * context_adjustment_factor
            dynamic_profit_taking_tanh_factor = dynamic_profit_taking_tanh_factor.clip(0.5, 3.0)
            profit_taking_effect = np.tanh(norm_profit_taking_quality * dynamic_profit_taking_tanh_factor)
            other_impurity_numeric_weights = {k: v for k, v in impurity_weights.items() if isinstance(v, (int, float)) and k not in ['fomo_concentration', 'profit_taking_margin', 'deception_lure_long', 'deception_lure_short', 'chip_health_slope', 'structural_tension']}
            total_other_impurity_weight = sum(other_impurity_numeric_weights.values())
            if total_other_impurity_weight > 0:
                other_impurity_components = {
                    'upper_shadow_selling_pressure': norm_upper_shadow_selling_pressure,
                    'rally_distribution_pressure': norm_rally_distribution_pressure,
                    'retail_fomo_premium': norm_retail_fomo_premium,
                    'winner_profit_margin_slope': norm_slope_5_winner_profit_margin,
                    'retail_fomo_premium_accel': norm_accel_5_retail_fomo_premium
                }
                other_impurity_score = utils._robust_geometric_mean(other_impurity_components, other_impurity_numeric_weights, df_index)
                other_impurity_effect = np.tanh(other_impurity_score * context_adjustment_factor)
            impurity_deception_modulator = pd.Series(1.0, index=df_index)
            if impurity_deception_mod_enabled:
                impurity_deception_modulator = impurity_deception_modulator * (1 + norm_deception_lure_long_intensity * deception_lure_long_impurity_amp_factor)
                deception_lure_short_damp_mask = (norm_deception_lure_short_intensity > 0) & (norm_main_force_conviction_bipolar > conviction_threshold)
                impurity_deception_modulator.loc[deception_lure_short_damp_mask] = impurity_deception_modulator.loc[deception_lure_short_damp_mask] * (1 - norm_deception_lure_short_intensity.loc[deception_lure_short_damp_mask] * deception_lure_short_impurity_damp_factor)
            impurity_resilience_modulator = pd.Series(1.0, index=df_index)
            if impurity_resilience_mod_enabled:
                impurity_resilience_modulator = impurity_resilience_modulator * (1 - norm_slope_5_chip_health.clip(lower=0) * chip_health_slope_impurity_damp_factor)
                impurity_resilience_modulator = impurity_resilience_modulator * (1 + norm_structural_tension * structural_tension_impurity_amp_factor)
            dynamic_impurity_fusion_exponent = impurity_fusion_exponent_base * (1 - normalized_sentiment_strength * impurity_fusion_exponent_sensitivity)
            dynamic_impurity_fusion_exponent = dynamic_impurity_fusion_exponent.clip(0.1, 1.0)
            final_impurity_effect = 1 - ((1 - fomo_effect) * (1 - profit_taking_effect) * (1 - other_impurity_effect)).pow(dynamic_impurity_fusion_exponent)
            final_impurity_effect = final_impurity_effect * impurity_deception_modulator * impurity_resilience_modulator
            final_impurity_effect = final_impurity_effect.clip(0, 1)
        else:
            final_impurity_effect = pd.Series(0.0, index=df_index)
        global_modulator_effect = pd.Series(1.0, index=df_index)
        if global_context_modulator_enabled:
            norm_global_chip_health = utils.get_adaptive_mtf_normalized_score(chip_health_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
            norm_global_main_force_conviction = utils.get_adaptive_mtf_normalized_score(main_force_conviction_raw.abs(), df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
            global_modulator_effect = (
                (1 + norm_global_chip_health * global_context_sensitivity_health) *
                (1 + norm_global_main_force_conviction * global_context_sensitivity_conviction)
            ).clip(0.5, 1.5)
        final_score = (conviction_base_unipolar * (1 - final_impurity_effect)) * 2 - 1
        final_score = final_score * global_modulator_effect
        final_score = final_score.clip(-1, 1).fillna(0.0).astype(np.float32)
        return final_score

    def _diagnose_axiom_trend_momentum(self, df: pd.DataFrame, periods: list, strategic_posture: pd.Series, battlefield_geography: pd.Series, holder_sentiment: pd.Series) -> pd.Series:
        """
        【V7.8 · Numba优化版】筹码公理六：诊断“结构性推力”
        - 核心优化: 将fuel_quality_score中的deception_penalty和synergy_bonus计算逻辑迁移至Numba加速的辅助函数。
        - 核心升级1: 引擎功率动态权重。引入筹码健康度趋势作为调制器，动态调整静态基础分与动态变化率的融合权重。
        - 核心升级2: 燃料品质诡道调制。引入筹码故障幅度作为负向调制器，削弱被“诱多”等诡道污染的燃料品质，并使协同奖励情境感知。
        - 核心升级3: 喷管效率多维深化。融合真空区大小、真空区趋势和穿越效率，更全面评估最小阻力路径。
        - 核心升级4: 最终融合动态权重。引入战略态势作为情境调制器，动态调整引擎功率、燃料品质、喷管效率的融合权重。
        - 核心升级5: 新增筹码指标整合：
            - 向上脉冲强度 (`upward_impulse_strength_D`) 增强燃料品质维度。
        - 升级: 优化 synergy_bonus 计算，引入平滑激活函数，避免硬性截断。
        - 升级: 增强最终融合动态权重的情境感知，引入多情境调制器进行综合调整。
        - 核心修复1: 修正 `engine_power_score` 的动态权重调制逻辑，使其在筹码健康度斜率负向时，增加静态权重。
        - 核心修复2: 增强 `fuel_quality_score` 中 `deception_penalty` 对正向筹码故障的惩罚力度，并取消诱多情境下的协同奖励。
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        method_name = "_diagnose_axiom_trend_momentum"
        df_index = df.index
        required_signals = [
            'main_force_conviction_index_D', 'vacuum_zone_magnitude_D', 'upward_impulse_purity_D',
            'chip_health_score_D', 'chip_fault_magnitude_D', 'SLOPE_5_vacuum_zone_magnitude_D',
            'vacuum_traversal_efficiency_D',
            'upward_impulse_strength_D',
            'SLOPE_5_chip_health_score_D'
        ]
        p_conf = self.chip_ultimate_params
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # 优化：预解析 tf_weights 一次
        parsed_tf_data = utils._parse_tf_weights(tf_weights)
        trend_momentum_params = get_param_value(p_conf.get('trend_momentum_params'), {})
        health_weights = get_param_value(trend_momentum_params.get('health_weights'), {'posture': 0.4, 'geography': 0.4, 'sentiment': 0.2})
        engine_power_dynamic_weight_modulator_signal_name = get_param_value(trend_momentum_params.get('engine_power_dynamic_weight_modulator_signal_name'), 'SLOPE_5_chip_health_score_D')
        engine_power_dynamic_weight_sensitivity = get_param_value(trend_momentum_params.get('engine_power_dynamic_weight_sensitivity'), 0.5)
        static_engine_power_base_weight = get_param_value(trend_momentum_params.get('static_engine_power_base_weight'), 0.5)
        dynamic_engine_power_base_weight = get_param_value(trend_momentum_params.get('dynamic_engine_power_base_weight'), 0.5)
        fuel_purity_deception_penalty_factor = get_param_value(trend_momentum_params.get('fuel_purity_deception_penalty_factor'), 0.3)
        synergy_bonus_base = get_param_value(trend_momentum_params.get('synergy_bonus_base'), 0.25)
        synergy_bonus_context_modulator_signal_name = get_param_value(trend_momentum_params.get('synergy_bonus_context_modulator_signal_name'), 'chip_health_score_D')
        synergy_bonus_context_sensitivity = get_param_value(trend_momentum_params.get('synergy_bonus_context_sensitivity'), 0.5)
        synergy_activation_threshold = get_param_value(trend_momentum_params.get('synergy_activation_threshold'), 0.0)
        nozzle_efficiency_weights = get_param_value(trend_momentum_params.get('nozzle_efficiency_weights'), {'magnitude': 0.5, 'trend': 0.3, 'traversal': 0.2})
        final_fusion_dynamic_weights_enabled = get_param_value(trend_momentum_params.get('final_fusion_dynamic_weights_enabled'), True)
        final_fusion_weights_base = get_param_value(trend_momentum_params.get('final_fusion_weights_base'), {'engine': 0.33, 'fuel': 0.33, 'nozzle': 0.34})
        final_fusion_weights_sensitivity = get_param_value(trend_momentum_params.get('final_fusion_weights_sensitivity'), {'engine': 0.5, 'fuel': 0.5, 'nozzle': 0.5})
        final_fusion_context_modulators_config = get_param_value(trend_momentum_params.get('final_fusion_context_modulators'), {
            'strategic_posture': {'signal': "strategic_posture", 'weight': 0.5, 'sensitivity': 0.5},
            'battlefield_geography': {'signal': "battlefield_geography", 'weight': 0.3, 'sensitivity': 0.3},
            'holder_sentiment': {'signal': "holder_sentiment", 'weight': 0.2, 'sensitivity': 0.2}
        })
        upward_impulse_strength_weight = get_param_value(trend_momentum_params.get('upward_impulse_strength_weight'), 0.2)
        # Ensure modulator signals are in required_signals
        if engine_power_dynamic_weight_modulator_signal_name not in required_signals:
            required_signals.append(engine_power_dynamic_weight_modulator_signal_name)
        if synergy_bonus_context_modulator_signal_name not in required_signals:
            required_signals.append(synergy_bonus_context_modulator_signal_name)
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index)
        signals_data = self._get_all_required_signals(df, required_signals, method_name)
        # --- 调试信息构建 ---
        is_debug_enabled = self.should_probe
        probe_ts = None
        if is_debug_enabled and self.probe_dates_set:
            for date in reversed(df_index):
                if date.date() in self.probe_dates_set:
                    probe_ts = date
                    break
        debug_info_tuple = (is_debug_enabled, probe_ts, method_name)
        # --- 原始数据获取 ---
        signal_map = {
            "strategic_posture": strategic_posture,
            "battlefield_geography": battlefield_geography,
            "holder_sentiment": holder_sentiment
        }
        health_score_slope_raw = signals_data[engine_power_dynamic_weight_modulator_signal_name]
        conviction_raw = signals_data['main_force_conviction_index_D']
        impulse_purity_raw = signals_data['upward_impulse_purity_D']
        upward_impulse_strength_raw = signals_data['upward_impulse_strength_D']
        chip_fault_raw = signals_data['chip_fault_magnitude_D']
        synergy_context_raw = signals_data[synergy_bonus_context_modulator_signal_name]
        vacuum_magnitude_raw = signals_data['vacuum_zone_magnitude_D']
        vacuum_trend_raw = signals_data['SLOPE_5_vacuum_zone_magnitude_D']
        vacuum_traversal_raw = signals_data['vacuum_traversal_efficiency_D']
        # --- 1. 引擎功率 (Engine Power) ---
        static_engine_power = (
            strategic_posture * health_weights['posture'] +
            battlefield_geography * health_weights['geography'] +
            holder_sentiment * health_weights['sentiment']
        )
        norm_health_score_slope = utils.get_adaptive_mtf_normalized_bipolar_score(health_score_slope_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        # 当筹码健康度斜率为负时，增加静态权重，降低动态权重
        dynamic_weight_mod = (norm_health_score_slope * engine_power_dynamic_weight_sensitivity)
        current_static_weight = (static_engine_power_base_weight - dynamic_weight_mod).clip(0.1, 0.9)
        current_dynamic_weight = (dynamic_engine_power_base_weight + dynamic_weight_mod).clip(0.1, 0.9)
        # 重新归一化，确保总和为1
        sum_current_weights = current_static_weight + current_dynamic_weight
        current_static_weight = current_static_weight / sum_current_weights
        current_dynamic_weight = current_dynamic_weight / sum_current_weights
        slope = static_engine_power.diff(1).fillna(0)
        accel = slope.diff(1).fillna(0)
        norm_slope = utils.get_adaptive_mtf_normalized_bipolar_score(slope, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_accel = utils.get_adaptive_mtf_normalized_bipolar_score(accel, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        dynamic_engine_power = ((norm_slope.add(1)/2) * (norm_accel.clip(lower=-1, upper=1).add(1)/2)).pow(0.5) * 2 - 1
        engine_power_score = static_engine_power * current_static_weight + dynamic_engine_power * current_dynamic_weight
        # --- 2. 燃料品质 (Fuel Quality) ---
        conviction_score = utils.get_adaptive_mtf_normalized_bipolar_score(conviction_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        purity_score = utils.get_adaptive_mtf_normalized_bipolar_score(impulse_purity_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_upward_impulse_strength = utils.get_adaptive_mtf_normalized_score(upward_impulse_strength_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        base_fuel_quality = ((conviction_score.add(1)/2) * (purity_score.add(1)/2)).pow(0.5) * 2 - 1
        base_fuel_quality = base_fuel_quality * (1 + norm_upward_impulse_strength * upward_impulse_strength_weight)
        base_fuel_quality = base_fuel_quality.clip(-1, 1)
        norm_chip_fault = utils.get_adaptive_mtf_normalized_score(chip_fault_raw.abs(), df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        # --- Numba优化区域：deception_penalty和synergy_bonus ---
        norm_synergy_context = utils.get_adaptive_mtf_normalized_score(synergy_context_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        fuel_quality_score_after_deception_values, synergy_bonus_values = _numba_calculate_fuel_quality_modulators_core(
            base_fuel_quality.values,
            norm_chip_fault.values,
            norm_synergy_context.values,
            chip_fault_raw.values,
            fuel_purity_deception_penalty_factor,
            synergy_bonus_base,
            synergy_bonus_context_sensitivity,
            synergy_activation_threshold
        )
        fuel_quality_score_after_deception = pd.Series(fuel_quality_score_after_deception_values, index=df_index, dtype=np.float32)
        synergy_bonus = pd.Series(synergy_bonus_values, index=df_index, dtype=np.float32)
        # --- Numba优化区域结束 ---
        fuel_quality_score = fuel_quality_score_after_deception + synergy_bonus
        fuel_quality_score = fuel_quality_score.clip(-1, 1)
        # --- 3. 喷管效率 (Nozzle Efficiency) ---
        norm_vacuum_magnitude = utils.get_adaptive_mtf_normalized_bipolar_score(vacuum_magnitude_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_vacuum_trend = utils.get_adaptive_mtf_normalized_bipolar_score(vacuum_trend_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_traversal_efficiency = utils.get_adaptive_mtf_normalized_bipolar_score(vacuum_traversal_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        nozzle_efficiency_score = (
            norm_vacuum_magnitude * nozzle_efficiency_weights.get('magnitude', 0.5) +
            norm_vacuum_trend * nozzle_efficiency_weights.get('trend', 0.3) +
            norm_traversal_efficiency * nozzle_efficiency_weights.get('traversal', 0.2)
        ).clip(-1, 1)
        # --- 4. 最终融合动态权重 (Final Fusion Dynamic Weights) ---
        engine_score_normalized = (engine_power_score + 1) / 2
        fuel_score_normalized = (fuel_quality_score + 1) / 2
        nozzle_score_normalized = (nozzle_efficiency_score + 1) / 2
        final_engine_weight = pd.Series(final_fusion_weights_base.get('engine', 0.33), index=df_index)
        final_fuel_weight = pd.Series(final_fusion_weights_base.get('fuel', 0.33), index=df_index)
        final_nozzle_weight = pd.Series(final_fusion_weights_base.get('nozzle', 0.34), index=df_index)
        if final_fusion_dynamic_weights_enabled:
            context_modulator_components = []
            total_context_weight = 0.0
            for ctx_name, ctx_config in final_fusion_context_modulators_config.items():
                signal_key = ctx_config.get('signal')
                signal_series = signal_map.get(signal_key) # 从传入的 signal_map 获取
                weight = ctx_config.get('weight', 0.0)
                sensitivity = ctx_config.get('sensitivity', 0.0)
                if signal_series is not None and weight > 0:
                    # 优化：传递预解析的 tf_weights 数据
                    norm_signal = utils.get_adaptive_mtf_normalized_bipolar_score(signal_series, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
                    context_modulator_components.append(norm_signal * weight * sensitivity)
                    total_context_weight += weight * sensitivity
            if context_modulator_components and total_context_weight > 0:
                context_fusion_modulator = sum(context_modulator_components) / total_context_weight
                normalized_fusion_modulator = context_fusion_modulator # 已经归一化到 [-1, 1]
            else:
                normalized_fusion_modulator = pd.Series(0.0, index=df_index)
            # 根据情境调制器调整权重
            engine_mod = normalized_fusion_modulator * final_fusion_weights_sensitivity.get('engine', 0.5)
            fuel_mod = normalized_fusion_modulator * final_fusion_weights_sensitivity.get('fuel', 0.5)
            nozzle_mod = -normalized_fusion_modulator * final_fusion_weights_sensitivity.get('nozzle', 0.5) # 负向调制，当情境有利时，喷管权重降低
            final_engine_weight = (final_fusion_weights_base.get('engine', 0.33) + engine_mod).clip(0.1, 0.6)
            final_fuel_weight = (final_fusion_weights_base.get('fuel', 0.33) + fuel_mod).clip(0.1, 0.6)
            final_nozzle_weight = (final_fusion_weights_base.get('nozzle', 0.34) + nozzle_mod).clip(0.1, 0.6)
            # 重新归一化权重，确保总和为1
            sum_dynamic_fusion_weights = final_engine_weight + final_fuel_weight + final_nozzle_weight
            final_engine_weight = final_engine_weight / sum_dynamic_fusion_weights
            final_fuel_weight = final_fuel_weight / sum_dynamic_fusion_weights
            final_nozzle_weight = final_nozzle_weight / sum_dynamic_fusion_weights
        # --- 最终融合 ---
        final_score_unipolar = (
            engine_score_normalized.pow(final_engine_weight) *
            fuel_score_normalized.pow(final_fuel_weight) *
            nozzle_score_normalized.pow(final_nozzle_weight)
        ).pow(1 / (final_engine_weight + final_fuel_weight + final_nozzle_weight)) # 几何平均
        final_score = (final_score_unipolar * 2 - 1).clip(-1, 1)
        final_score = final_score.clip(-1, 1).fillna(0.0).astype(np.float32)
        return final_score

    def _diagnose_axiom_divergence(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V7.4 · Numba优化版】筹码公理五：诊断“价筹张力”
        - 核心优化: 将deception_modulator_factor的计算逻辑迁移至Numba加速的辅助函数。
        - 核心数学升级1: 将“主力共谋验证”从依赖资金流信号升级为更纯粹、更稳健的“主力筹码意图验证”模型。
                          该模型直接评估1)主力筹码信念是否与背离方向一致(同谋), 2)主力信念强度是否足够大(兵力)。
                          只有当两者都满足时，才确认为一次高置信度的“战术性背离”，并给予显著加成。
        - 核心数学升级2: “筹码趋势”的多元化解读。引入赢家集中度与赢家/输家动量共同构建复合筹码趋势，更全面捕捉筹码结构与价格的分歧。
        - 核心数学升级3: “持续性”的优化。将持续性量化为分歧方向的一致性累积，而非波动性，更准确反映张力积蓄。
        - 核心数学升级4: “能量注入”的筹码化。替换通用成交量为建设性换手率，更精准反映筹码层面的活跃度与质量。
        - 核心数学升级5: “诡道双向调制”。引入筹码故障幅度对分歧强度进行情境调制，根据故障与分歧方向的匹配关系，动态地放大或削弱价筹张力信号。
        - 核心数学升级6: “情境自适应放大器”。引入筹码健康度作为情境调制器，动态调整张力强度和主力意图验证的放大倍数。
        - 核心数学升级7: “非线性放大控制”。对放大项引入tanh变换，使其增长更平滑，并有饱和上限，防止过度放大。
        - 核心数学升级8: “动态复合筹码趋势权重”。引入筹码波动不稳定性指数作为调制器，自适应调整复合筹码趋势中动量和集中度的权重。
        - 核心修复: 增强 `deception_modulator_factor` 的惩罚机制，当出现“诱空”且主力资金流出时，大幅降低分数。
        """
        method_name = "_diagnose_axiom_divergence"
        df_index = df.index
        required_signals = [
            'winner_loser_momentum_D', 'winner_concentration_90pct_D', 'SLOPE_5_close_D',
            'constructive_turnover_ratio_D', 'main_force_conviction_index_D', 'chip_fault_magnitude_D',
            'chip_health_score_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D',
            'main_force_flow_directionality_D', 'deception_index_D'
        ]
        p_conf = self.chip_ultimate_params
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        parsed_tf_data = utils._parse_tf_weights(tf_weights)
        divergence_params = get_param_value(p_conf.get('divergence_params'), {})
        chip_trend_momentum_weight_base = get_param_value(divergence_params.get('chip_trend_momentum_weight'), 0.6)
        chip_trend_concentration_weight_base = get_param_value(divergence_params.get('chip_trend_concentration_weight'), 0.4)
        tension_magnitude_amplifier_base = get_param_value(divergence_params.get('tension_magnitude_amplifier'), 1.5)
        chip_intent_factor_amplifier_base = get_param_value(divergence_params.get('chip_intent_factor_amplifier'), 0.5)
        deception_modulator_impact_clip = get_param_value(divergence_params.get('deception_modulator_impact_clip'), 0.5)
        deception_modulator_reinforce_factor = get_param_value(divergence_params.get('deception_modulator_reinforce_factor'), 0.5)
        conflict_bonus = get_param_value(divergence_params.get('conflict_bonus'), 0.5)
        contextual_amplification_enabled = get_param_value(divergence_params.get('contextual_amplification_enabled'), True)
        context_modulator_signal_name = get_param_value(divergence_params.get('context_modulator_signal_name'), 'chip_health_score_D')
        context_sensitivity_tension = get_param_value(divergence_params.get('context_sensitivity_tension'), 0.5)
        context_sensitivity_intent = get_param_value(divergence_params.get('context_sensitivity_intent'), 0.5)
        non_linear_amplification_enabled = get_param_value(divergence_params.get('non_linear_amplification_enabled'), True)
        non_linear_amp_tanh_factor = get_param_value(divergence_params.get('non_linear_amp_tanh_factor'), 1.0)
        dynamic_chip_trend_weights_enabled = get_param_value(divergence_params.get('dynamic_chip_trend_weights_enabled'), True)
        chip_trend_weight_modulator_signal_name = get_param_value(divergence_params.get('chip_trend_weight_modulator_signal_name'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        chip_trend_weight_mod_sensitivity = get_param_value(divergence_params.get('chip_trend_weight_mod_sensitivity'), 0.5)
        bearish_deception_penalty_factor = get_param_value(divergence_params.get('bearish_deception_penalty_factor'), 0.8)
        if chip_trend_weight_modulator_signal_name not in required_signals:
            required_signals.append(chip_trend_weight_modulator_signal_name)
        if context_modulator_signal_name not in required_signals:
            required_signals.append(context_modulator_signal_name)
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index)
        signals_data = self._get_all_required_signals(df, required_signals, method_name)
        is_debug_enabled = self.should_probe
        probe_ts = None
        if is_debug_enabled and self.probe_dates_set:
            for date in reversed(df_index):
                if date.date() in self.probe_dates_set:
                    probe_ts = date
                    break
        debug_info_tuple = False
        chip_momentum_raw = signals_data['winner_loser_momentum_D']
        chip_concentration_raw = signals_data['winner_concentration_90pct_D']
        price_trend_raw = signals_data['SLOPE_5_close_D']
        constructive_turnover_raw = signals_data['constructive_turnover_ratio_D']
        mf_chip_conviction_raw = signals_data['main_force_conviction_index_D']
        chip_fault_raw = signals_data['chip_fault_magnitude_D']
        chip_health_raw = signals_data['chip_health_score_D']
        chip_trend_modulator_raw = signals_data[chip_trend_weight_modulator_signal_name]
        context_modulator_raw = signals_data[context_modulator_signal_name]
        main_force_flow_directionality_raw = signals_data['main_force_flow_directionality_D']
        deception_index_raw = signals_data['deception_index_D']
        dynamic_momentum_weight = pd.Series(chip_trend_momentum_weight_base, index=df_index)
        dynamic_concentration_weight = pd.Series(chip_trend_concentration_weight_base, index=df_index)
        if dynamic_chip_trend_weights_enabled:
            normalized_chip_trend_modulator = utils.get_adaptive_mtf_normalized_score(chip_trend_modulator_raw, df_index, tf_weights=tf_weights, ascending=True, debug_info=False, _parsed_tf_data=parsed_tf_data)
            dynamic_momentum_weight = chip_trend_momentum_weight_base * (1 + normalized_chip_trend_modulator * chip_trend_weight_mod_sensitivity)
            dynamic_concentration_weight = chip_trend_concentration_weight_base * (1 - normalized_chip_trend_modulator * chip_trend_weight_mod_sensitivity)
            sum_dynamic_weights = dynamic_momentum_weight + dynamic_concentration_weight
            dynamic_momentum_weight = (dynamic_momentum_weight / sum_dynamic_weights).clip(0.1, 0.9)
            dynamic_concentration_weight = (dynamic_concentration_weight / sum_dynamic_weights).clip(0.1, 0.9)
        norm_chip_momentum = utils.get_adaptive_mtf_normalized_bipolar_score(chip_momentum_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_chip_concentration = utils.get_adaptive_mtf_normalized_bipolar_score(chip_concentration_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        composite_chip_trend = (
            norm_chip_momentum * dynamic_momentum_weight +
            norm_chip_concentration * dynamic_concentration_weight
        )
        norm_price_trend = utils.get_adaptive_mtf_normalized_bipolar_score(price_trend_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        disagreement_vector = composite_chip_trend - norm_price_trend
        persistence_raw = np.sign(disagreement_vector).rolling(window=13, min_periods=5).sum().fillna(0)
        norm_persistence = utils.get_adaptive_mtf_normalized_score(persistence_raw.abs(), df_index, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_constructive_turnover = utils.get_adaptive_mtf_normalized_score(constructive_turnover_raw, df_index, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        energy_injection = norm_constructive_turnover * disagreement_vector.abs()
        tension_magnitude = (norm_persistence * energy_injection).pow(0.5)
        norm_mf_chip_conviction = utils.get_adaptive_mtf_normalized_bipolar_score(mf_chip_conviction_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        is_aligned = (np.sign(disagreement_vector) * np.sign(norm_mf_chip_conviction)) > 0
        intent_strength = norm_mf_chip_conviction.abs()
        chip_intent_verification_score = is_aligned * intent_strength
        dynamic_tension_amplifier = pd.Series(tension_magnitude_amplifier_base, index=df_index)
        dynamic_chip_intent_factor_amplifier = pd.Series(chip_intent_factor_amplifier_base, index=df_index)
        if contextual_amplification_enabled:
            normalized_context = utils.get_adaptive_mtf_normalized_score(context_modulator_raw, df_index, tf_weights=tf_weights, ascending=True, debug_info=False, _parsed_tf_data=parsed_tf_data)
            dynamic_tension_amplifier = tension_magnitude_amplifier_base * (1 + normalized_context * context_sensitivity_tension)
            dynamic_chip_intent_factor_amplifier = chip_intent_factor_amplifier_base * (1 + normalized_context * context_sensitivity_intent)
            dynamic_tension_amplifier = dynamic_tension_amplifier.clip(tension_magnitude_amplifier_base * 0.5, tension_magnitude_amplifier_base * 2.0)
            dynamic_chip_intent_factor_amplifier = dynamic_chip_intent_factor_amplifier.clip(chip_intent_factor_amplifier_base * 0.5, chip_intent_factor_amplifier_base * 2.0)
        tension_amplification_term = tension_magnitude * dynamic_tension_amplifier
        chip_intent_amplification_term = chip_intent_verification_score * dynamic_chip_intent_factor_amplifier
        if non_linear_amplification_enabled:
            tension_amplification_term = np.tanh(tension_amplification_term * non_linear_amp_tanh_factor)
            chip_intent_amplification_term = np.tanh(chip_intent_amplification_term * non_linear_amp_tanh_factor)
        chip_intent_factor = 1.0 + chip_intent_amplification_term
        norm_chip_fault = utils.get_adaptive_mtf_normalized_score(chip_fault_raw.abs(), df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        # --- Numba优化区域：deception_modulator_factor ---
        deception_modulator_factor_values = _numba_calculate_divergence_deception_modulator_core(
            np.sign(disagreement_vector).values,
            np.sign(chip_fault_raw).values,
            norm_chip_fault.values,
            utils.get_adaptive_mtf_normalized_bipolar_score(deception_index_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data).values,
            utils.get_adaptive_mtf_normalized_bipolar_score(main_force_flow_directionality_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data).values,
            deception_modulator_impact_clip,
            deception_modulator_reinforce_factor,
            bearish_deception_penalty_factor
        )
        deception_modulator_factor = pd.Series(deception_modulator_factor_values, index=df_index, dtype=np.float32)
        # --- Numba优化区域结束 ---
        base_final_score = disagreement_vector * (1 + tension_amplification_term) * chip_intent_factor * deception_modulator_factor
        conflict_mask = (np.sign(composite_chip_trend) * np.sign(norm_price_trend) < 0)
        conflict_amplifier = pd.Series(1.0, index=df_index)
        conflict_amplifier.loc[conflict_mask] = 1.0 + conflict_bonus
        safe_base_score = base_final_score.clip(-0.999, 0.999)
        final_score = np.tanh(np.arctanh(safe_base_score) * conflict_amplifier)
        final_score = final_score.clip(-1, 1).fillna(0.0).astype(np.float32)
        return final_score

    def _diagnose_structural_consensus(self, df: pd.DataFrame, cost_structure_scores: pd.Series, holder_sentiment_scores: pd.Series) -> pd.Series:
        """
        【V7.18 · 最终分数敏感度动态版 (生产就绪版)】诊断筹码同调驱动力
        一个基于“引擎-传动”思想的终极信号，旨在量化筹码结构对上涨意愿的真实转化效率。
        它将“持股心态”视为提供上涨意愿的引擎，将“成本结构”视为决定能量损耗的传动系统。
        核心升级:
        - 筹码健康度 `chip_health_score_D` 作为非线性调制参数（amplification_power, dampening_power）的动态调节器。
        - 筹码健康度对幂指数的敏感度根据另一个筹码层面的信号（例如 `VOLATILITY_INSTABILITY_INDEX_21d_D` 筹码波动性）进行动态调整。
        - 筹码结构分数 `cost_structure_scores` 对情绪驱动力的调制强度，根据持股心态 `holder_sentiment_scores` 的正负方向，进行非对称的非线性动态调整。
        - 情绪与筹码结构之间的耦合强度也实现了动态调整。
        - 筹码健康度调制敏感度引入了非对称性。
        - 动态中性阈值使得判断情绪和筹码结构是看涨/看跌或顺风/逆风的“中性”界限，将根据筹码健康度动态调整。
        - 情绪激活阈值使得持股心态的原始强度在参与驱动力计算之前，会根据其与动态中性阈值的相对关系进行“激活”或“去激活”处理。
        - 情绪强度对筹码结构调制效果的动态缩放，激活后的情绪强度将动态缩放筹码结构分数对驱动力的最终影响。
        - 结构强度对幂指数的自适应调整，amplification_power 和 dampening_power 将根据最终用于调制的筹码结构分数的绝对强度进行进一步的动态调整。
        - 结构强度对幂指数自适应调整的敏感度动态调制，使得模型在不同市场环境下对筹码结构信号的反应更加精细和智能。
        - 结构强度对幂指数自适应调整的非对称非线性映射，为正向和负向结构强度引入独立的 tanh 因子和可选的偏移量。
        - 最终分数敏感度的动态调整，final_score 的饱和速度将根据市场环境进行动态调整。
        高分代表市场不仅想涨，而且其内部筹码结构健康且具备高效转化这种意愿的能力。
        """
        df_index = df.index
        p_conf = self.chip_ultimate_params
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        coherent_drive_params = get_param_value(p_conf.get('coherent_drive_params'), {})
        base_amplification_power = get_param_value(coherent_drive_params.get('amplification_power'), 1.2)
        base_dampening_power = get_param_value(coherent_drive_params.get('dampening_power'), 1.5)
        chip_health_modulation_enabled = get_param_value(coherent_drive_params.get('chip_health_modulation_enabled'), True)
        default_chip_health_sensitivity_amp = get_param_value(coherent_drive_params.get('chip_health_sensitivity_amp'), 0.5)
        default_chip_health_sensitivity_damp = get_param_value(coherent_drive_params.get('chip_health_sensitivity_damp'), 0.5)
        chip_health_mtf_norm_params = get_param_value(coherent_drive_params.get('chip_health_mtf_norm_params'), {})
        chip_health_tanh_factor_amp = get_param_value(coherent_drive_params.get('chip_health_tanh_factor_amp'), 1.0)
        chip_health_tanh_factor_damp = get_param_value(coherent_drive_params.get('chip_health_tanh_factor_damp'), 1.0)
        chip_health_sensitivity_modulation_enabled = get_param_value(coherent_drive_params.get('chip_health_sensitivity_modulation_enabled'), False)
        chip_sensitivity_modulator_signal_name = get_param_value(coherent_drive_params.get('chip_sensitivity_modulator_signal_name'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        chip_sensitivity_mod_norm_window = get_param_value(coherent_drive_params.get('chip_sensitivity_mod_norm_window'), 21)
        chip_sensitivity_mod_factor_amp = get_param_value(coherent_drive_params.get('chip_sensitivity_mod_factor_amp'), 1.0)
        chip_sensitivity_mod_factor_damp = get_param_value(coherent_drive_params.get('chip_sensitivity_mod_factor_damp'), 1.0)
        chip_sensitivity_mod_tanh_factor_amp = get_param_value(coherent_drive_params.get('chip_health_sensitivity_mod_tanh_factor_amp'), 1.0)
        chip_sensitivity_mod_tanh_factor_damp = get_param_value(coherent_drive_params.get('chip_health_sensitivity_mod_tanh_factor_damp'), 1.0)
        cost_structure_asymmetric_impact_enabled = get_param_value(coherent_drive_params.get('cost_structure_asymmetric_impact_enabled'), False)
        cost_structure_impact_base_factor_bullish = get_param_value(coherent_drive_params.get('cost_structure_impact_base_factor_bullish'), 1.0)
        cost_structure_impact_base_factor_bearish = get_param_value(coherent_drive_params.get('cost_structure_impact_base_factor_bearish'), 1.0)
        cost_structure_impact_sentiment_sensitivity_bullish = get_param_value(coherent_drive_params.get('cost_structure_impact_sentiment_sensitivity_bullish'), 1.0)
        cost_structure_impact_sentiment_tanh_factor_bullish = get_param_value(coherent_drive_params.get('cost_structure_impact_sentiment_tanh_factor_bullish'), 1.0)
        cost_structure_impact_sentiment_sensitivity_bearish = get_param_value(coherent_drive_params.get('cost_structure_impact_sentiment_sensitivity_bearish'), 1.0)
        cost_structure_impact_sentiment_tanh_factor_bearish = get_param_value(coherent_drive_params.get('cost_structure_impact_sentiment_tanh_factor_bearish'), 1.0)
        sentiment_cost_structure_coupling_enabled = get_param_value(coherent_drive_params.get('sentiment_cost_structure_coupling_enabled'), False)
        sentiment_coupling_base_factor = get_param_value(coherent_drive_params.get('sentiment_coupling_base_factor'), 1.0)
        sentiment_coupling_tanh_factor = get_param_value(coherent_drive_params.get('sentiment_coupling_tanh_factor'), 1.0)
        sentiment_coupling_sensitivity = get_param_value(coherent_drive_params.get('sentiment_coupling_sensitivity'), 1.0)
        chip_health_asymmetric_sensitivity_enabled = get_param_value(coherent_drive_params.get('chip_health_asymmetric_sensitivity_enabled'), False)
        chip_health_sensitivity_amp_positive_health = get_param_value(coherent_drive_params.get('chip_health_sensitivity_amp_positive_health'), 0.5)
        chip_health_sensitivity_amp_negative_health = get_param_value(coherent_drive_params.get('chip_health_sensitivity_amp_negative_health'), 0.5)
        chip_health_sensitivity_damp_positive_health = get_param_value(coherent_drive_params.get('chip_health_sensitivity_damp_positive_health'), 0.5)
        chip_health_sensitivity_damp_negative_health = get_param_value(coherent_drive_params.get('chip_health_sensitivity_damp_negative_health'), 0.5)
        dynamic_neutrality_thresholds_enabled = get_param_value(coherent_drive_params.get('dynamic_neutrality_thresholds_enabled'), False)
        sentiment_neutrality_base_threshold = get_param_value(coherent_drive_params.get('sentiment_neutrality_base_threshold'), 0.0)
        sentiment_neutrality_chip_health_sensitivity = get_param_value(coherent_drive_params.get('sentiment_neutrality_chip_health_sensitivity'), 0.1)
        cost_structure_neutrality_base_threshold = get_param_value(coherent_drive_params.get('cost_structure_neutrality_base_threshold'), 0.0)
        cost_structure_neutrality_chip_health_sensitivity = get_param_value(coherent_drive_params.get('cost_structure_neutrality_chip_health_sensitivity'), 0.1)
        sentiment_activation_enabled = get_param_value(coherent_drive_params.get('sentiment_activation_enabled'), False)
        sentiment_activation_tanh_factor = get_param_value(coherent_drive_params.get('sentiment_activation_tanh_factor'), 1.0)
        sentiment_activation_strength = get_param_value(coherent_drive_params.get('sentiment_activation_strength'), 1.0)
        structure_modulation_strength_enabled = get_param_value(coherent_drive_params.get('structure_modulation_strength_enabled'), False)
        structure_modulation_base_strength = get_param_value(coherent_drive_params.get('structure_modulation_base_strength'), 1.0)
        structure_modulation_sentiment_tanh_factor = get_param_value(coherent_drive_params.get('structure_modulation_sentiment_tanh_factor'), 1.0)
        structure_modulation_sentiment_sensitivity = get_param_value(coherent_drive_params.get('structure_modulation_sentiment_sensitivity'), 1.0)
        structural_power_adjustment_enabled = get_param_value(coherent_drive_params.get('structural_power_adjustment_enabled'), False)
        default_structural_power_sensitivity_amp = get_param_value(coherent_drive_params.get('structural_power_sensitivity_amp'), 0.5)
        default_structural_power_sensitivity_damp = get_param_value(coherent_drive_params.get('structural_power_sensitivity_damp'), 0.5)
        default_structural_power_tanh_factor_amp = get_param_value(coherent_drive_params.get('structural_power_tanh_factor_amp'), 1.0)
        default_structural_power_tanh_factor_damp = get_param_value(coherent_drive_params.get('structural_power_tanh_factor_damp'), 1.0)
        structural_power_asymmetric_tanh_enabled = get_param_value(coherent_drive_params.get('structural_power_asymmetric_tanh_enabled'), False)
        structural_power_tanh_factor_positive_structure = get_param_value(coherent_drive_params.get('structural_power_tanh_factor_positive_structure'), 1.0)
        structural_power_tanh_factor_negative_structure = get_param_value(coherent_drive_params.get('structural_power_tanh_factor_negative_structure'), 1.0)
        structural_power_offset_positive_structure = get_param_value(coherent_drive_params.get('structural_power_offset_positive_structure'), 0.0)
        structural_power_offset_negative_structure = get_param_value(coherent_drive_params.get('structural_power_offset_negative_structure'), 0.0)
        final_score_sensitivity_modulation_enabled = get_param_value(coherent_drive_params.get('final_score_sensitivity_modulation_enabled'), False)
        final_score_modulator_signal_name = get_param_value(coherent_drive_params.get('final_score_modulator_signal_name'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        final_score_mod_norm_window = get_param_value(coherent_drive_params.get('final_score_mod_norm_window'), 21)
        final_score_mod_factor = get_param_value(coherent_drive_params.get('final_score_mod_factor'), 1.0)
        final_score_mod_tanh_factor = get_param_value(coherent_drive_params.get('final_score_mod_tanh_factor'), 1.0)
        final_score_base_sensitivity_multiplier = get_param_value(coherent_drive_params.get('final_score_base_sensitivity_multiplier'), 2.0)
        amplification_power = pd.Series(base_amplification_power, index=df.index)
        dampening_power = pd.Series(base_dampening_power, index=df.index)
        modulation_factor = pd.Series(1.0, index=df.index)
        current_chip_health_score_raw = pd.Series(0.0, index=df.index)
        normalized_chip_health = pd.Series(0.0, index=df.index)
        dynamic_chip_health_sensitivity_amp = pd.Series(default_chip_health_sensitivity_amp, index=df.index)
        dynamic_chip_health_sensitivity_damp = pd.Series(default_chip_health_sensitivity_damp, index=df.index)
        dynamic_cost_structure_impact_factor_bullish = pd.Series(cost_structure_impact_base_factor_bullish, index=df.index)
        dynamic_cost_structure_impact_factor_bearish = pd.Series(cost_structure_impact_base_factor_bearish, index=df.index)
        dynamic_coupling_factor = pd.Series(sentiment_coupling_base_factor, index=df.index)
        final_cost_structure_for_modulation = pd.Series(0.0, index=df.index)
        dynamic_sentiment_neutrality_threshold = pd.Series(sentiment_neutrality_base_threshold, index=df.index)
        dynamic_cost_structure_neutrality_threshold = pd.Series(cost_structure_neutrality_base_threshold, index=df.index)
        activated_holder_sentiment_scores = holder_sentiment_scores.copy()
        dynamic_structure_modulation_strength = pd.Series(structure_modulation_base_strength, index=df.index)
        final_cost_structure_for_modulation_scaled = pd.Series(0.0, index=df.index)
        dynamic_structural_power_sensitivity_amp = pd.Series(default_structural_power_sensitivity_amp, index=df.index)
        dynamic_structural_power_sensitivity_damp = pd.Series(default_structural_power_sensitivity_damp, index=df.index)
        dynamic_final_score_sensitivity_multiplier = pd.Series(final_score_base_sensitivity_multiplier, index=df.index)
        required_signals = [
            'chip_health_score_D',
        ]
        if chip_sensitivity_modulator_signal_name not in required_signals:
            required_signals.append(chip_sensitivity_modulator_signal_name)
        if final_score_modulator_signal_name not in required_signals:
            required_signals.append(final_score_modulator_signal_name)
        if not self._validate_required_signals(df, required_signals, "_diagnose_structural_consensus"):
            return pd.Series(0.0, index=df.index)
        signals_data = self._get_all_required_signals(df, required_signals, "_diagnose_structural_consensus")
        parsed_tf_data = utils._parse_tf_weights(tf_weights)
        # 优化：预解析 chip_health_tf_weights 一次
        chip_health_tf_weights = chip_health_mtf_norm_params.get('weights', {})
        parsed_chip_health_tf_data = utils._parse_tf_weights(chip_health_tf_weights)
        if chip_health_modulation_enabled:
            current_chip_health_score_raw = signals_data['chip_health_score_D']
            # 优化：传递预解析的 chip_health_tf_weights 数据
            normalized_chip_health = utils.get_adaptive_mtf_normalized_bipolar_score(
                current_chip_health_score_raw,
                df.index,
                tf_weights=chip_health_tf_weights,
                sensitivity=chip_health_mtf_norm_params.get('sensitivity', 2.0),
                _parsed_tf_data=parsed_chip_health_tf_data
            )
            base_amp_sensitivity_series = pd.Series(default_chip_health_sensitivity_amp, index=df.index)
            base_damp_sensitivity_series = pd.Series(default_chip_health_sensitivity_damp, index=df.index)
            if chip_health_asymmetric_sensitivity_enabled:
                positive_health_mask = normalized_chip_health > 0
                negative_health_mask = normalized_chip_health < 0
                base_amp_sensitivity_series.loc[positive_health_mask] = chip_health_sensitivity_amp_positive_health
                base_amp_sensitivity_series.loc[negative_health_mask] = chip_health_sensitivity_amp_negative_health
                base_damp_sensitivity_series.loc[positive_health_mask] = chip_health_sensitivity_damp_positive_health
                base_damp_sensitivity_series.loc[negative_health_mask] = chip_health_sensitivity_damp_negative_health
            if chip_health_sensitivity_modulation_enabled:
                modulator_signal_raw = signals_data[chip_sensitivity_modulator_signal_name]
                normalized_modulator_signal = utils.normalize_score(
                    modulator_signal_raw,
                    df_index,
                    windows=chip_sensitivity_mod_norm_window,
                    ascending=True
                )
                modulator_bipolar = (normalized_modulator_signal * 2) - 1
                non_linear_modulator_effect_amp = np.tanh(modulator_bipolar * chip_sensitivity_mod_tanh_factor_amp)
                non_linear_modulator_effect_damp = np.tanh(modulator_bipolar * chip_sensitivity_mod_tanh_factor_damp)
                dynamic_chip_health_sensitivity_amp = base_amp_sensitivity_series * (1 + non_linear_modulator_effect_amp * chip_sensitivity_mod_factor_amp)
                dynamic_chip_health_sensitivity_damp = base_damp_sensitivity_series * (1 + non_linear_modulator_effect_damp * chip_sensitivity_mod_factor_damp)
                dynamic_chip_health_sensitivity_amp = dynamic_chip_health_sensitivity_amp.clip(base_amp_sensitivity_series * 0.1, base_amp_sensitivity_series * 2.0)
                dynamic_chip_health_sensitivity_damp = dynamic_chip_health_sensitivity_damp.clip(base_damp_sensitivity_series * 0.1, base_damp_sensitivity_series * 2.0)
            else:
                dynamic_chip_health_sensitivity_amp = base_amp_sensitivity_series
                dynamic_chip_health_sensitivity_damp = base_damp_sensitivity_series
            modulated_chip_health_amp = np.tanh(normalized_chip_health * chip_health_tanh_factor_amp)
            modulated_chip_health_damp = np.tanh(normalized_chip_health * chip_health_tanh_factor_damp)
            amplification_power = base_amplification_power * (1 + modulated_chip_health_amp * dynamic_chip_health_sensitivity_amp)
            dampening_power = base_dampening_power * (1 - modulated_chip_health_damp * dynamic_chip_health_sensitivity_damp)
            amplification_power = amplification_power.clip(0.5, 2.0)
            dampening_power = dampening_power.clip(0.5, 2.0)
        if dynamic_neutrality_thresholds_enabled:
            dynamic_sentiment_neutrality_threshold = sentiment_neutrality_base_threshold + (normalized_chip_health * sentiment_neutrality_chip_health_sensitivity)
            dynamic_cost_structure_neutrality_threshold = cost_structure_neutrality_base_threshold + (normalized_chip_health * cost_structure_neutrality_chip_health_sensitivity)
            dynamic_sentiment_neutrality_threshold = dynamic_sentiment_neutrality_threshold.clip(-0.2, 0.2)
            dynamic_cost_structure_neutrality_threshold = dynamic_cost_structure_neutrality_threshold.clip(-0.2, 0.2)
        if sentiment_activation_enabled:
            positive_active_mask = holder_sentiment_scores > dynamic_sentiment_neutrality_threshold
            negative_active_mask = holder_sentiment_scores < -dynamic_sentiment_neutrality_threshold
            neutral_mask = ~(positive_active_mask | negative_active_mask)
            activated_holder_sentiment_scores.loc[positive_active_mask] = \
                holder_sentiment_scores.loc[positive_active_mask] - dynamic_sentiment_neutrality_threshold.loc[positive_active_mask]
            activated_holder_sentiment_scores.loc[negative_active_mask] = \
                holder_sentiment_scores.loc[negative_active_mask] + dynamic_sentiment_neutrality_threshold.loc[negative_active_mask]
            activated_holder_sentiment_scores.loc[neutral_mask] = 0.0
            activated_holder_sentiment_scores = np.tanh(activated_holder_sentiment_scores * sentiment_activation_tanh_factor) * sentiment_activation_strength
        if cost_structure_asymmetric_impact_enabled:
            positive_sentiment_mask = holder_sentiment_scores > 0
            if positive_sentiment_mask.any():
                positive_sentiment_strength = holder_sentiment_scores[positive_sentiment_mask]
                normalized_positive_sentiment_tanh = np.tanh(positive_sentiment_strength * cost_structure_impact_sentiment_tanh_factor_bullish)
                dynamic_cost_structure_impact_factor_bullish.loc[positive_sentiment_mask] = \
                    cost_structure_impact_base_factor_bullish * (1 + (normalized_positive_sentiment_tanh - 0.5) * cost_structure_impact_sentiment_sensitivity_bullish)
                dynamic_cost_structure_impact_factor_bullish = dynamic_cost_structure_impact_factor_bullish.clip(0.1, 2.0)
            negative_sentiment_mask = holder_sentiment_scores < 0
            if negative_sentiment_mask.any():
                negative_sentiment_strength = holder_sentiment_scores[negative_sentiment_mask].abs()
                normalized_negative_sentiment_tanh = np.tanh(negative_sentiment_strength * cost_structure_impact_sentiment_tanh_factor_bearish)
                dynamic_cost_structure_impact_factor_bearish.loc[negative_sentiment_mask] = \
                    cost_structure_impact_base_factor_bearish * (1 + (normalized_negative_sentiment_tanh - 0.5) * cost_structure_impact_sentiment_sensitivity_bearish)
                dynamic_cost_structure_impact_factor_bearish = dynamic_cost_structure_impact_factor_bearish.clip(0.1, 2.0)
        selected_dynamic_cost_structure_impact_factor = pd.Series(1.0, index=df.index)
        selected_dynamic_cost_structure_impact_factor.loc[holder_sentiment_scores > 0] = dynamic_cost_structure_impact_factor_bullish.loc[holder_sentiment_scores > 0]
        selected_dynamic_cost_structure_impact_factor.loc[holder_sentiment_scores < 0] = dynamic_cost_structure_impact_factor_bearish.loc[holder_sentiment_scores < 0]
        adjusted_cost_structure_scores = cost_structure_scores * selected_dynamic_cost_structure_impact_factor
        if sentiment_cost_structure_coupling_enabled:
            abs_holder_sentiment = holder_sentiment_scores.abs()
            sentiment_tanh_modulated = np.tanh(abs_holder_sentiment * sentiment_coupling_tanh_factor)
            dynamic_coupling_factor = sentiment_coupling_base_factor * (1 + sentiment_tanh_modulated * sentiment_coupling_sensitivity)
            dynamic_coupling_factor = dynamic_coupling_factor.clip(0.1, 2.0)
        final_cost_structure_for_modulation = adjusted_cost_structure_scores * dynamic_coupling_factor
        if structure_modulation_strength_enabled:
            abs_activated_sentiment = activated_holder_sentiment_scores.abs()
            sentiment_tanh_modulated_for_structure = np.tanh(abs_activated_sentiment * structure_modulation_sentiment_tanh_factor)
            dynamic_structure_modulation_strength = structure_modulation_base_strength * (1 + sentiment_tanh_modulated_for_structure * structure_modulation_sentiment_sensitivity)
            dynamic_structure_modulation_strength = dynamic_structure_modulation_strength.clip(0.1, 2.0)
        final_cost_structure_for_modulation_scaled = final_cost_structure_for_modulation * dynamic_structure_modulation_strength
        if structural_power_adjustment_enabled:
            positive_structure_mask = final_cost_structure_for_modulation_scaled > 0
            negative_structure_mask = final_cost_structure_for_modulation_scaled < 0
            if structural_power_asymmetric_tanh_enabled:
                if positive_structure_mask.any():
                    positive_structure_strength = final_cost_structure_for_modulation_scaled[positive_structure_mask]
                    boost_amp = np.tanh((positive_structure_strength + structural_power_offset_positive_structure) * structural_power_tanh_factor_positive_structure) * dynamic_structural_power_sensitivity_amp.loc[positive_structure_mask]
                    amplification_power.loc[positive_structure_mask] = amplification_power.loc[positive_structure_mask] * (1 + boost_amp)
                if negative_structure_mask.any():
                    negative_structure_strength = final_cost_structure_for_modulation_scaled[negative_structure_mask].abs()
                    boost_damp = np.tanh((negative_structure_strength + structural_power_offset_negative_structure) * structural_power_tanh_factor_negative_structure) * dynamic_structural_power_sensitivity_damp.loc[negative_structure_mask]
                    dampening_power.loc[negative_structure_mask] = dampening_power.loc[negative_structure_mask] * (1 + boost_damp)
            else:
                if positive_structure_mask.any():
                    positive_structure_strength = final_cost_structure_for_modulation_scaled[positive_structure_mask]
                    boost_amp = np.tanh(positive_structure_strength * default_structural_power_tanh_factor_amp) * dynamic_structural_power_sensitivity_amp.loc[positive_structure_mask]
                    amplification_power.loc[positive_structure_mask] = amplification_power.loc[positive_structure_mask] * (1 + boost_amp)
                if negative_structure_mask.any():
                    negative_structure_strength = final_cost_structure_for_modulation_scaled[negative_structure_mask].abs()
                    boost_damp = np.tanh(negative_structure_strength * default_structural_power_tanh_factor_damp) * dynamic_structural_power_sensitivity_damp.loc[negative_structure_mask]
                    dampening_power.loc[negative_structure_mask] = dampening_power.loc[negative_structure_mask] * (1 + boost_damp)
            amplification_power = amplification_power.clip(0.5, 3.0)
            dampening_power = dampening_power.clip(0.5, 3.0)
        bullish_mask = holder_sentiment_scores > dynamic_sentiment_neutrality_threshold
        bearish_mask = holder_sentiment_scores < -dynamic_sentiment_neutrality_threshold
        bullish_tailwind_mask = bullish_mask & (final_cost_structure_for_modulation_scaled > dynamic_cost_structure_neutrality_threshold)
        modulation_factor.loc[bullish_tailwind_mask] = (1 + final_cost_structure_for_modulation_scaled.loc[bullish_tailwind_mask]) ** amplification_power.loc[bullish_tailwind_mask]
        bullish_headwind_mask = bullish_mask & (final_cost_structure_for_modulation_scaled < -dynamic_cost_structure_neutrality_threshold)
        modulation_factor.loc[bullish_headwind_mask] = (1 - final_cost_structure_for_modulation_scaled.loc[bullish_headwind_mask].abs()) ** dampening_power.loc[bullish_headwind_mask]
        bearish_tailwind_mask = bearish_mask & (final_cost_structure_for_modulation_scaled < -dynamic_cost_structure_neutrality_threshold)
        modulation_factor.loc[bearish_tailwind_mask] = (1 + final_cost_structure_for_modulation_scaled.loc[bearish_tailwind_mask].abs()) ** amplification_power.loc[bearish_tailwind_mask]
        bearish_headwind_mask = bearish_mask & (final_cost_structure_for_modulation_scaled > dynamic_cost_structure_neutrality_threshold)
        modulation_factor.loc[bearish_headwind_mask] = (1 - final_cost_structure_for_modulation_scaled.loc[bearish_headwind_mask]) ** dampening_power.loc[bearish_headwind_mask]
        coherent_drive_raw = activated_holder_sentiment_scores * modulation_factor
        if final_score_sensitivity_modulation_enabled:
            final_score_modulator_signal_raw = signals_data[final_score_modulator_signal_name]
            final_score_normalized_modulator_signal = utils.normalize_score(
                final_score_modulator_signal_raw,
                df_index,
                windows=final_score_mod_norm_window,
                ascending=True
            )
            final_score_modulator_bipolar = (final_score_normalized_modulator_signal * 2) - 1
            final_score_non_linear_modulator_effect = np.tanh(final_score_modulator_bipolar * final_score_mod_tanh_factor)
            dynamic_final_score_sensitivity_multiplier = final_score_base_sensitivity_multiplier * (1 + final_score_non_linear_modulator_effect * final_score_mod_factor)
            dynamic_final_score_sensitivity_multiplier = dynamic_final_score_sensitivity_multiplier.clip(final_score_base_sensitivity_multiplier * 0.5, final_score_base_sensitivity_multiplier * 2.0)
        else:
            dynamic_final_score_sensitivity_multiplier = pd.Series(final_score_base_sensitivity_multiplier, index=df.index)
        final_score = np.tanh(coherent_drive_raw * (self.bipolar_sensitivity * dynamic_final_score_sensitivity_multiplier))
        return final_score.astype(np.float32)

    def _diagnose_absorption_echo(self, df: pd.DataFrame, divergence_scores: pd.Series) -> pd.Series:
        """
        【V5.4 · Numba优化版】吸筹回声探针
        - 核心优化: 将deception_modulator的计算逻辑迁移至Numba加速的辅助函数。
        - 核心升级1: 恐慌声源精细化。在V4.0基础上，引入总输家比例短期加速度、散户恐慌投降指数短期斜率、结构性紧张指数短期加速度，更精准捕捉恐慌蔓延。
        - 核心升级2: 逆流介质强化。在V4.0基础上，引入浮动筹码清洗效率短期斜率、订单簿清算率短期加速度、微观价格冲击不对称性短期斜率、VWAP控制强度短期斜率、VWAP穿越强度短期加速度，更全面评估承接能力。
        - 核心升级3: 主力回声深化。在V4.0基础上，引入隐蔽吸筹信号短期加速度、压制式吸筹强度短期斜率、主力成本优势短期加速度、主力资金流向方向性短期斜率、主力VPOC短期加速度、智能资金净买入短期斜率，更细致刻画主力吸筹意图。
        - 核心升级4: 诡道背景调制智能化。优化诡道调制逻辑，引入“诱空反吸”的增强机制，即当出现“诱空”式欺骗且主力信念坚定，则增强吸筹回声信号；同时细化对“诱多”式欺骗和对倒行为的惩罚。
        - 核心升级5: 情境调制器引入。引入资金流可信度指数、结构性紧张指数、筹码健康度作为最终分数的调制器，提供更丰富的宏观情境感知。
        - 核心升级6: 新增筹码指标整合：
            - 支持性派发强度 (`supportive_distribution_intensity_D`) 作为负向调制器。
        - 核心修复1: 修正 `panic_source_score` 对恐慌动态的判断，当散户恐慌快速消退时，降低其贡献。
        - 核心修复2: 调整 `deception_modulator` 逻辑，当 `norm_deception_index_bipolar` 为负时，应增强 `absorption_echo` 信号。
        - **新增业务逻辑：引入“牛市陷阱情境惩罚”，在近期大幅下跌后伴随正向欺骗时，大幅降低吸筹回声的得分。**
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        method_name = "_diagnose_absorption_echo"
        df_index = df.index
        required_signals = [
            'retail_panic_surrender_index_D', 'loser_pain_index_D', 'chip_fatigue_index_D',
            'structural_tension_index_D', 'panic_selling_cascade_D', 'total_loser_rate_D',
            'loser_loss_margin_avg_D', 'SLOPE_5_loser_pain_index_D', 'ACCEL_5_chip_fatigue_index_D',
            'VOLATILITY_INSTABILITY_INDEX_21d_D', 'capitulation_absorption_index_D',
            'floating_chip_cleansing_efficiency_D', 'support_validation_strength_D',
            'main_force_execution_alpha_D', 'active_buying_support_D', 'opening_gap_defense_strength_D',
            'control_solidity_index_D', 'SLOPE_5_support_validation_strength_D',
            'ACCEL_5_main_force_execution_alpha_D', 'order_book_clearing_rate_D',
            'covert_accumulation_signal_D', 'suppressive_accumulation_intensity_D',
            'main_force_cost_advantage_D', 'peak_control_transfer_D', 'main_force_conviction_index_D',
            'main_force_net_flow_calibrated_D', 'main_force_flow_directionality_D', 'main_force_vpoc_D',
            'main_force_activity_ratio_D', 'SLOPE_5_covert_accumulation_signal_D',
            'ACCEL_5_main_force_conviction_index_D', 'SMART_MONEY_HM_NET_BUY_D',
            'chip_fault_magnitude_D', 'deception_index_D', 'wash_trade_intensity_D',
            'chip_health_score_D', 'main_force_conviction_index_D',
            'ACCEL_5_total_loser_rate_D', 'SLOPE_5_retail_panic_surrender_index_D', 'ACCEL_5_structural_tension_index_D',
            'SLOPE_5_floating_chip_cleansing_efficiency_D', 'ACCEL_5_order_book_clearing_rate_D',
            'SLOPE_5_micro_price_impact_asymmetry_D', 'SLOPE_5_vwap_control_strength_D', 'ACCEL_5_vwap_crossing_intensity_D',
            'ACCEL_5_covert_accumulation_signal_D', 'SLOPE_5_suppressive_accumulation_intensity_D',
            'ACCEL_5_main_force_cost_advantage_D', 'SLOPE_5_main_force_flow_directionality_D',
            'ACCEL_5_main_force_vpoc_D', 'SLOPE_5_SMART_MONEY_HM_NET_BUY_D',
            'flow_credibility_index_D',
            'supportive_distribution_intensity_D',
            'vwap_control_strength_D', 'vwap_crossing_intensity_D', 'micro_price_impact_asymmetry_D',
            'pct_change_D'
        ]
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index)
        signals_data = self._get_all_required_signals(df, required_signals, method_name)
        p_conf = self.chip_ultimate_params
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        parsed_tf_data = utils._parse_tf_weights(tf_weights)
        absorption_echo_params = get_param_value(p_conf.get('absorption_echo_params'), {})
        panic_source_weights = get_param_value(absorption_echo_params.get('panic_source_weights'), {})
        panic_context_threshold = get_param_value(absorption_echo_params.get('panic_context_threshold'), 0.3)
        counter_flow_medium_weights = get_param_value(absorption_echo_params.get('counter_flow_medium_weights'), {})
        main_force_echo_weights = get_param_value(absorption_echo_params.get('main_force_echo_weights'), {})
        deception_modulator_params = get_param_value(absorption_echo_params.get('deception_modulator_params'), {})
        final_fusion_exponent = get_param_value(absorption_echo_params.get('final_fusion_exponent'), 0.25)
        context_modulator_weights = get_param_value(absorption_echo_params.get('context_modulator_weights'), {})
        supportive_distribution_penalty_factor = get_param_value(absorption_echo_params.get('supportive_distribution_penalty_factor'), 0.2)
        panic_slope_dampening_enabled = get_param_value(absorption_echo_params.get('panic_slope_dampening_enabled'), True)
        panic_slope_dampening_sensitivity = get_param_value(absorption_echo_params.get('panic_slope_dampening_sensitivity'), 0.5)
        deception_boost_factor_negative = get_param_value(absorption_echo_params.get('deception_boost_factor_negative'), 0.5)
        is_debug_enabled = self.should_probe
        probe_ts = None
        if is_debug_enabled and self.probe_dates_set:
            for date in reversed(df_index):
                if date.date() in self.probe_dates_set:
                    probe_ts = date
                    break
        debug_info_tuple = False
        retail_panic_surrender_raw = signals_data['retail_panic_surrender_index_D']
        loser_pain_raw = signals_data['loser_pain_index_D']
        chip_fatigue_raw = signals_data['chip_fatigue_index_D']
        structural_tension_raw = signals_data['structural_tension_index_D']
        panic_selling_cascade_raw = signals_data['panic_selling_cascade_D']
        total_loser_rate_raw = signals_data['total_loser_rate_D']
        loser_loss_margin_avg_raw = signals_data['loser_loss_margin_avg_D']
        slope_5_loser_pain_raw = signals_data['SLOPE_5_loser_pain_index_D']
        accel_5_chip_fatigue_raw = signals_data['ACCEL_5_chip_fatigue_index_D']
        volatility_instability_raw = signals_data['VOLATILITY_INSTABILITY_INDEX_21d_D']
        accel_5_total_loser_rate_raw = signals_data['ACCEL_5_total_loser_rate_D']
        slope_5_retail_panic_surrender_raw = signals_data['SLOPE_5_retail_panic_surrender_index_D']
        accel_5_structural_tension_raw = signals_data['ACCEL_5_structural_tension_index_D']
        divergence_bullish_raw = divergence_scores
        capitulation_absorption_raw = signals_data['capitulation_absorption_index_D']
        cleansing_efficiency_raw = signals_data['floating_chip_cleansing_efficiency_D']
        support_validation_raw = signals_data['support_validation_strength_D']
        main_force_execution_alpha_raw = signals_data['main_force_execution_alpha_D']
        active_buying_support_raw = signals_data['active_buying_support_D']
        opening_gap_defense_strength_raw = signals_data['opening_gap_defense_strength_D']
        control_solidity_raw = signals_data['control_solidity_index_D']
        slope_5_support_validation_raw = signals_data['SLOPE_5_support_validation_strength_D']
        accel_5_main_force_execution_alpha_raw = signals_data['ACCEL_5_main_force_execution_alpha_D']
        order_book_clearing_rate_raw = signals_data['order_book_clearing_rate_D']
        slope_5_floating_chip_cleansing_raw = signals_data['SLOPE_5_floating_chip_cleansing_efficiency_D']
        accel_5_order_book_clearing_raw = signals_data['ACCEL_5_order_book_clearing_rate_D']
        micro_price_impact_asymmetry_raw = signals_data['micro_price_impact_asymmetry_D']
        slope_5_micro_price_impact_asymmetry_raw = signals_data['SLOPE_5_micro_price_impact_asymmetry_D']
        vwap_control_strength_raw = signals_data['vwap_control_strength_D']
        slope_5_vwap_control_strength_raw = signals_data['SLOPE_5_vwap_control_strength_D']
        vwap_crossing_intensity_raw = signals_data['vwap_crossing_intensity_D']
        accel_5_vwap_crossing_intensity_raw = signals_data['ACCEL_5_vwap_crossing_intensity_D']
        covert_accumulation_raw = signals_data['covert_accumulation_signal_D']
        suppressive_accumulation_raw = signals_data['suppressive_accumulation_intensity_D']
        main_force_cost_advantage_raw = signals_data['main_force_cost_advantage_D']
        peak_control_transfer_raw = signals_data['peak_control_transfer_D']
        main_force_conviction_raw = signals_data['main_force_conviction_index_D']
        main_force_net_flow_calibrated_raw = signals_data['main_force_net_flow_calibrated_D']
        main_force_flow_directionality_raw = signals_data['main_force_flow_directionality_D']
        main_force_vpoc_raw = signals_data['main_force_vpoc_D']
        main_force_activity_ratio_raw = signals_data['main_force_activity_ratio_D']
        slope_5_covert_accumulation_raw = signals_data['SLOPE_5_covert_accumulation_signal_D']
        accel_5_main_force_conviction_raw = signals_data['ACCEL_5_main_force_conviction_index_D']
        smart_money_net_buy_raw = signals_data['SMART_MONEY_HM_NET_BUY_D']
        accel_5_covert_accumulation_raw = signals_data['ACCEL_5_covert_accumulation_signal_D']
        slope_5_suppressive_accumulation_raw = signals_data['SLOPE_5_suppressive_accumulation_intensity_D']
        accel_5_main_force_cost_advantage_raw = signals_data['ACCEL_5_main_force_cost_advantage_D']
        slope_5_main_force_flow_directionality_raw = signals_data['SLOPE_5_main_force_flow_directionality_D']
        accel_5_main_force_vpoc_raw = signals_data['ACCEL_5_main_force_vpoc_D']
        slope_5_smart_money_net_buy_raw = signals_data['SLOPE_5_SMART_MONEY_HM_NET_BUY_D']
        chip_fault_magnitude_raw = signals_data['chip_fault_magnitude_D']
        deception_index_raw = signals_data['deception_index_D']
        wash_trade_intensity_raw = signals_data['wash_trade_intensity_D']
        chip_health_score_raw = signals_data['chip_health_score_D']
        flow_credibility_raw = signals_data['flow_credibility_index_D']
        supportive_distribution_intensity_raw = signals_data['supportive_distribution_intensity_D']
        norm_retail_panic_surrender = utils.get_adaptive_mtf_normalized_score(retail_panic_surrender_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_loser_pain = utils.get_adaptive_mtf_normalized_score(loser_pain_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_chip_fatigue = utils.get_adaptive_mtf_normalized_score(chip_fatigue_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_structural_tension_negative = utils.get_adaptive_mtf_normalized_score(structural_tension_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_panic_selling_cascade = utils.get_adaptive_mtf_normalized_score(panic_selling_cascade_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_total_loser_rate = utils.get_adaptive_mtf_normalized_score(total_loser_rate_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_loser_loss_margin_avg = utils.get_adaptive_mtf_normalized_score(loser_loss_margin_avg_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_slope_5_loser_pain = utils.get_adaptive_mtf_normalized_score(slope_5_loser_pain_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_accel_5_chip_fatigue = utils.get_adaptive_mtf_normalized_score(accel_5_chip_fatigue_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_volatility_instability = utils.get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_accel_5_total_loser_rate = utils.get_adaptive_mtf_normalized_score(accel_5_total_loser_rate_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_slope_5_retail_panic_surrender = utils.get_adaptive_mtf_normalized_score(slope_5_retail_panic_surrender_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_accel_5_structural_tension = utils.get_adaptive_mtf_normalized_score(accel_5_structural_tension_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        panic_source_numeric_weights = {k: v for k, v in panic_source_weights.items() if isinstance(v, (int, float))}
        panic_source_score = utils._robust_geometric_mean(
            {
                'retail_panic_surrender': norm_retail_panic_surrender,
                'loser_pain': norm_loser_pain,
                'chip_fatigue': norm_chip_fatigue,
                'structural_tension_negative': norm_structural_tension_negative,
                'panic_selling_cascade': norm_panic_selling_cascade,
                'total_loser_rate': norm_total_loser_rate,
                'loser_loss_margin_avg': norm_loser_loss_margin_avg,
                'loser_pain_slope': norm_slope_5_loser_pain,
                'chip_fatigue_accel': norm_accel_5_chip_fatigue,
                'volatility_instability': norm_volatility_instability,
                'total_loser_rate_accel': norm_accel_5_total_loser_rate,
                'retail_panic_surrender_slope': norm_slope_5_retail_panic_surrender,
                'structural_tension_accel': norm_accel_5_structural_tension
            },
            panic_source_numeric_weights, df_index
        )
        panic_slope_dampening_factor = pd.Series(1.0, index=df_index)
        if panic_slope_dampening_enabled:
            norm_panic_slope = utils.get_adaptive_mtf_normalized_bipolar_score(slope_5_retail_panic_surrender_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
            panic_slope_dampening_factor = (1 - norm_panic_slope.clip(upper=0).abs() * panic_slope_dampening_sensitivity).clip(0.1, 1.0)
        panic_source_score_modulated = panic_source_score * panic_slope_dampening_factor
        is_panic_context = panic_source_score_modulated > panic_context_threshold
        norm_divergence_bullish = utils.get_adaptive_mtf_normalized_score(divergence_bullish_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_capitulation_absorption = utils.get_adaptive_mtf_normalized_score(capitulation_absorption_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_cleansing_efficiency = utils.get_adaptive_mtf_normalized_score(cleansing_efficiency_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_support_validation = utils.get_adaptive_mtf_normalized_score(support_validation_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_main_force_execution_alpha = utils.get_adaptive_mtf_normalized_score(main_force_execution_alpha_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_active_buying_support = utils.get_adaptive_mtf_normalized_score(active_buying_support_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_opening_gap_defense_strength = utils.get_adaptive_mtf_normalized_score(opening_gap_defense_strength_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_control_solidity = utils.get_adaptive_mtf_normalized_score(control_solidity_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_support_validation_slope = utils.get_adaptive_mtf_normalized_score(slope_5_support_validation_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_main_force_execution_alpha_accel = utils.get_adaptive_mtf_normalized_score(accel_5_main_force_execution_alpha_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_order_book_clearing_rate = utils.get_adaptive_mtf_normalized_score(order_book_clearing_rate_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_slope_5_floating_chip_cleansing = utils.get_adaptive_mtf_normalized_score(slope_5_floating_chip_cleansing_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_accel_5_order_book_clearing = utils.get_adaptive_mtf_normalized_score(accel_5_order_book_clearing_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_micro_price_impact_asymmetry = utils.get_adaptive_mtf_normalized_score(micro_price_impact_asymmetry_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_slope_5_micro_price_impact_asymmetry = utils.get_adaptive_mtf_normalized_score(slope_5_micro_price_impact_asymmetry_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_vwap_control_strength = utils.get_adaptive_mtf_normalized_score(vwap_control_strength_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_slope_5_vwap_control_strength = utils.get_adaptive_mtf_normalized_score(slope_5_vwap_control_strength_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_vwap_crossing_intensity = utils.get_adaptive_mtf_normalized_score(vwap_crossing_intensity_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_accel_5_vwap_crossing_intensity = utils.get_adaptive_mtf_normalized_score(accel_5_vwap_crossing_intensity_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        counter_flow_medium_numeric_weights = {k: v for k, v in counter_flow_medium_weights.items() if isinstance(v, (int, float))}
        counter_flow_medium_score = utils._robust_geometric_mean(
            {
                'divergence_bullish': norm_divergence_bullish,
                'capitulation_absorption': norm_capitulation_absorption,
                'cleansing_efficiency': norm_cleansing_efficiency,
                'support_validation': norm_support_validation,
                'main_force_execution_alpha': norm_main_force_execution_alpha,
                'active_buying_support': norm_active_buying_support,
                'opening_gap_defense_strength': norm_opening_gap_defense_strength,
                'control_solidity': norm_control_solidity,
                'support_validation_slope': norm_support_validation_slope,
                'main_force_execution_alpha_accel': norm_main_force_execution_alpha_accel,
                'order_book_clearing_rate': norm_order_book_clearing_rate,
                'cleansing_efficiency_slope': norm_slope_5_floating_chip_cleansing,
                'order_book_clearing_accel': norm_accel_5_order_book_clearing,
                'micro_impact_asymmetry': norm_micro_price_impact_asymmetry,
                'micro_impact_asymmetry_slope': norm_slope_5_micro_price_impact_asymmetry,
                'vwap_control_strength': norm_vwap_control_strength,
                'vwap_control_strength_slope': norm_slope_5_vwap_control_strength,
                'vwap_crossing_intensity': norm_vwap_crossing_intensity,
                'vwap_crossing_intensity_accel': norm_accel_5_vwap_crossing_intensity
            },
            counter_flow_medium_numeric_weights, df_index
        )
        norm_covert_accumulation = utils.get_adaptive_mtf_normalized_score(covert_accumulation_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_suppressive_accumulation = utils.get_adaptive_mtf_normalized_score(suppressive_accumulation_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_main_force_cost_advantage = utils.get_adaptive_mtf_normalized_score(main_force_cost_advantage_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_peak_control_transfer = utils.get_adaptive_mtf_normalized_score(peak_control_transfer_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_main_force_conviction_positive = utils.get_adaptive_mtf_normalized_score(main_force_conviction_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_main_force_net_flow_positive = utils.get_adaptive_mtf_normalized_score(main_force_net_flow_calibrated_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_main_force_flow_directionality_positive = utils.get_adaptive_mtf_normalized_score(main_force_flow_directionality_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_main_force_vpoc = utils.get_adaptive_mtf_normalized_score(main_force_vpoc_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_main_force_activity_ratio = utils.get_adaptive_mtf_normalized_score(main_force_activity_ratio_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_slope_5_covert_accumulation = utils.get_adaptive_mtf_normalized_score(slope_5_covert_accumulation_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_accel_5_main_force_conviction = utils.get_adaptive_mtf_normalized_score(accel_5_main_force_conviction_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_smart_money_net_buy_positive = utils.get_adaptive_mtf_normalized_score(smart_money_net_buy_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_accel_5_covert_accumulation = utils.get_adaptive_mtf_normalized_score(accel_5_covert_accumulation_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_slope_5_suppressive_accumulation = utils.get_adaptive_mtf_normalized_score(slope_5_suppressive_accumulation_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_accel_5_main_force_cost_advantage = utils.get_adaptive_mtf_normalized_score(accel_5_main_force_cost_advantage_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_slope_5_main_force_flow_directionality = utils.get_adaptive_mtf_normalized_score(slope_5_main_force_flow_directionality_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_accel_5_main_force_vpoc = utils.get_adaptive_mtf_normalized_score(accel_5_main_force_vpoc_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_slope_5_smart_money_net_buy = utils.get_adaptive_mtf_normalized_score(slope_5_smart_money_net_buy_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        main_force_echo_numeric_weights = {k: v for k, v in main_force_echo_weights.items() if isinstance(v, (int, float))}
        main_force_echo_score = utils._robust_geometric_mean(
            {
                'covert_accumulation': norm_covert_accumulation,
                'suppressive_accumulation': norm_suppressive_accumulation,
                'cost_advantage': norm_main_force_cost_advantage,
                'peak_control_transfer': norm_peak_control_transfer,
                'main_force_conviction_positive': norm_main_force_conviction_positive,
                'main_force_net_flow_positive': norm_main_force_net_flow_positive,
                'main_force_flow_directionality_positive': norm_main_force_flow_directionality_positive,
                'main_force_vpoc': norm_main_force_vpoc,
                'main_force_activity_ratio': norm_main_force_activity_ratio,
                'covert_accumulation_slope': norm_slope_5_covert_accumulation,
                'main_force_conviction_accel': norm_accel_5_main_force_conviction,
                'smart_money_net_buy_positive': norm_smart_money_net_buy_positive,
                'covert_accumulation_accel': norm_accel_5_covert_accumulation,
                'suppressive_accumulation_slope': norm_slope_5_suppressive_accumulation,
                'cost_advantage_accel': norm_accel_5_main_force_cost_advantage,
                'flow_directionality_slope': norm_slope_5_main_force_flow_directionality,
                'main_force_vpoc_accel': norm_accel_5_main_force_vpoc,
                'smart_money_net_buy_slope': norm_slope_5_smart_money_net_buy
            },
            main_force_echo_numeric_weights, df_index
        )
        # --- Numba优化区域：deception_modulator ---
        norm_chip_fault_magnitude_bipolar = utils.get_adaptive_mtf_normalized_bipolar_score(chip_fault_magnitude_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_deception_index_bipolar = utils.get_adaptive_mtf_normalized_bipolar_score(deception_index_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_wash_trade_intensity = utils.get_adaptive_mtf_normalized_score(wash_trade_intensity_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_main_force_conviction_bipolar = utils.get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_supportive_distribution_intensity = utils.get_adaptive_mtf_normalized_score(supportive_distribution_intensity_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        deception_modulator_values = _numba_calculate_absorption_echo_deception_modulator_core(
            counter_flow_medium_score.values, # 使用 counter_flow_medium_score 作为 net_conviction_flow_quality 的代理
            norm_deception_index_bipolar.values,
            norm_wash_trade_intensity.values,
            norm_chip_fault_magnitude_bipolar.values,
            norm_supportive_distribution_intensity.values,
            deception_boost_factor_negative,
            deception_modulator_params.get('deception_index_penalty_weight', 0.7), # 从params获取
            deception_modulator_params.get('wash_trade_penalty_weight', 0.3), # 从params获取
            deception_modulator_params.get('penalty_factor', 0.4), # 从params获取
            supportive_distribution_penalty_factor
        )
        deception_modulator = pd.Series(deception_modulator_values, index=df_index, dtype=np.float32)
        # --- Numba优化区域结束 ---
        norm_flow_credibility = utils.get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_structural_tension = utils.get_adaptive_mtf_normalized_score(structural_tension_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_chip_health_score = utils.get_adaptive_mtf_normalized_score(chip_health_score_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        context_modulator_numeric_weights = {k: v for k, v in context_modulator_weights.items() if isinstance(v, (int, float))}
        total_context_modulator_weight = sum(context_modulator_numeric_weights.values())
        context_modulator = pd.Series(1.0, index=df_index)
        if total_context_modulator_weight > 0:
            fused_context_modulator_raw = (
                norm_flow_credibility.pow(context_modulator_numeric_weights.get('flow_credibility', 0.4)) *
                (1 - norm_structural_tension).pow(context_modulator_numeric_weights.get('structural_tension_inverse', 0.3)) *
                norm_chip_health_score.pow(context_modulator_numeric_weights.get('chip_health', 0.3))
            ).pow(1 / total_context_modulator_weight)
            context_modulator = 1 + (fused_context_modulator_raw - 0.5) * 0.5
        context_modulator = context_modulator.clip(0.5, 1.5)
        base_score = pd.Series(0.0, index=df_index)
        valid_mask = is_panic_context
        if valid_mask.any():
            base_score.loc[valid_mask] = (
                counter_flow_medium_score.loc[valid_mask].pow(0.5) *
                main_force_echo_score.loc[valid_mask].pow(0.5)
            )
        final_score = base_score * deception_modulator * context_modulator
        final_score = final_score.pow(final_fusion_exponent)
        bull_trap_penalty = self._calculate_bull_trap_context_penalty(df)
        final_score = final_score * bull_trap_penalty
        final_score = final_score.clip(0.0, 1.0).fillna(0.0).astype(np.float32)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"  -- [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在诊断“吸筹回声”信号...")
            print(f"      [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---")
            for sig_name in required_signals:
                val = signals_data[sig_name].loc[probe_ts] if probe_ts in signals_data[sig_name].index else np.nan
                print(f"        '{sig_name}': {val:.4f}")
            print(f"      [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 归一化信号值 ---")
            print(f"        norm_retail_panic_surrender: {norm_retail_panic_surrender.loc[probe_ts]:.4f}")
            print(f"        norm_loser_pain: {norm_loser_pain.loc[probe_ts]:.4f}")
            print(f"        norm_chip_fatigue: {norm_chip_fatigue.loc[probe_ts]:.4f}")
            print(f"        norm_structural_tension_negative: {norm_structural_tension_negative.loc[probe_ts]:.4f}")
            print(f"        norm_panic_selling_cascade: {norm_panic_selling_cascade.loc[probe_ts]:.4f}")
            print(f"        norm_total_loser_rate: {norm_total_loser_rate.loc[probe_ts]:.4f}")
            print(f"        norm_loser_loss_margin_avg: {norm_loser_loss_margin_avg.loc[probe_ts]:.4f}")
            print(f"        norm_slope_5_loser_pain: {norm_slope_5_loser_pain.loc[probe_ts]:.4f}")
            print(f"        norm_accel_5_chip_fatigue: {norm_accel_5_chip_fatigue.loc[probe_ts]:.4f}")
            print(f"        norm_volatility_instability: {norm_volatility_instability.loc[probe_ts]:.4f}")
            print(f"        norm_accel_5_total_loser_rate: {norm_accel_5_total_loser_rate.loc[probe_ts]:.4f}")
            print(f"        norm_slope_5_retail_panic_surrender: {norm_slope_5_retail_panic_surrender.loc[probe_ts]:.4f}")
            print(f"        norm_accel_5_structural_tension: {norm_accel_5_structural_tension.loc[probe_ts]:.4f}")
            print(f"        norm_divergence_bullish: {norm_divergence_bullish.loc[probe_ts]:.4f}")
            print(f"        norm_capitulation_absorption: {norm_capitulation_absorption.loc[probe_ts]:.4f}")
            print(f"        norm_cleansing_efficiency: {norm_cleansing_efficiency.loc[probe_ts]:.4f}")
            print(f"        norm_support_validation: {norm_support_validation.loc[probe_ts]:.4f}")
            print(f"        norm_main_force_execution_alpha: {norm_main_force_execution_alpha.loc[probe_ts]:.4f}")
            print(f"        norm_active_buying_support: {norm_active_buying_support.loc[probe_ts]:.4f}")
            print(f"        norm_opening_gap_defense_strength: {norm_opening_gap_defense_strength.loc[probe_ts]:.4f}")
            print(f"        norm_control_solidity: {norm_control_solidity.loc[probe_ts]:.4f}")
            print(f"        norm_support_validation_slope: {norm_support_validation_slope.loc[probe_ts]:.4f}")
            print(f"        norm_main_force_execution_alpha_accel: {norm_main_force_execution_alpha_accel.loc[probe_ts]:.4f}")
            print(f"        norm_order_book_clearing_rate: {norm_order_book_clearing_rate.loc[probe_ts]:.4f}")
            print(f"        norm_slope_5_floating_chip_cleansing: {norm_slope_5_floating_chip_cleansing.loc[probe_ts]:.4f}")
            print(f"        norm_accel_5_order_book_clearing: {norm_accel_5_order_book_clearing.loc[probe_ts]:.4f}")
            print(f"        norm_micro_price_impact_asymmetry: {norm_micro_price_impact_asymmetry.loc[probe_ts]:.4f}")
            print(f"        norm_slope_5_micro_price_impact_asymmetry: {norm_slope_5_micro_price_impact_asymmetry.loc[probe_ts]:.4f}")
            print(f"        norm_vwap_control_strength: {norm_vwap_control_strength.loc[probe_ts]:.4f}")
            print(f"        norm_slope_5_vwap_control_strength: {norm_slope_5_vwap_control_strength.loc[probe_ts]:.4f}")
            print(f"        norm_vwap_crossing_intensity: {norm_vwap_crossing_intensity.loc[probe_ts]:.4f}")
            print(f"        norm_accel_5_vwap_crossing_intensity: {norm_accel_5_vwap_crossing_intensity.loc[probe_ts]:.4f}")
            print(f"        norm_covert_accumulation: {norm_covert_accumulation.loc[probe_ts]:.4f}")
            print(f"        norm_suppressive_accumulation: {norm_suppressive_accumulation.loc[probe_ts]:.4f}")
            print(f"        norm_main_force_cost_advantage: {norm_main_force_cost_advantage.loc[probe_ts]:.4f}")
            print(f"        norm_peak_control_transfer: {norm_peak_control_transfer.loc[probe_ts]:.4f}")
            print(f"        norm_main_force_conviction_positive: {norm_main_force_conviction_positive.loc[probe_ts]:.4f}")
            print(f"        norm_main_force_net_flow_positive: {norm_main_force_net_flow_positive.loc[probe_ts]:.4f}")
            print(f"        norm_main_force_flow_directionality_positive: {norm_main_force_flow_directionality_positive.loc[probe_ts]:.4f}")
            print(f"        norm_main_force_vpoc: {norm_main_force_vpoc.loc[probe_ts]:.4f}")
            print(f"        norm_main_force_activity_ratio: {norm_main_force_activity_ratio.loc[probe_ts]:.4f}")
            print(f"        norm_slope_5_covert_accumulation: {norm_slope_5_covert_accumulation.loc[probe_ts]:.4f}")
            print(f"        norm_accel_5_main_force_conviction: {norm_accel_5_main_force_conviction.loc[probe_ts]:.4f}")
            print(f"        norm_smart_money_net_buy_positive: {norm_smart_money_net_buy_positive.loc[probe_ts]:.4f}")
            print(f"        norm_accel_5_covert_accumulation: {norm_accel_5_covert_accumulation.loc[probe_ts]:.4f}")
            print(f"        norm_slope_5_suppressive_accumulation: {norm_slope_5_suppressive_accumulation.loc[probe_ts]:.4f}")
            print(f"        norm_accel_5_main_force_cost_advantage: {norm_accel_5_main_force_cost_advantage.loc[probe_ts]:.4f}")
            print(f"        norm_slope_5_main_force_flow_directionality: {norm_slope_5_main_force_flow_directionality.loc[probe_ts]:.4f}")
            print(f"        norm_accel_5_main_force_vpoc: {norm_accel_5_main_force_vpoc.loc[probe_ts]:.4f}")
            print(f"        norm_slope_5_smart_money_net_buy: {norm_slope_5_smart_money_net_buy.loc[probe_ts]:.4f}")
            print(f"        norm_chip_fault_magnitude_bipolar: {norm_chip_fault_magnitude_bipolar.loc[probe_ts]:.4f}")
            print(f"        norm_deception_index_bipolar: {norm_deception_index_bipolar.loc[probe_ts]:.4f}")
            print(f"        norm_wash_trade_intensity: {norm_wash_trade_intensity.loc[probe_ts]:.4f}")
            print(f"        norm_main_force_conviction_bipolar: {norm_main_force_conviction_bipolar.loc[probe_ts]:.4f}")
            print(f"        norm_supportive_distribution_intensity: {norm_supportive_distribution_intensity.loc[probe_ts]:.4f}")
            print(f"        norm_flow_credibility: {norm_flow_credibility.loc[probe_ts]:.4f}")
            print(f"        norm_structural_tension: {norm_structural_tension.loc[probe_ts]:.4f}")
            print(f"        norm_chip_health_score: {norm_chip_health_score.loc[probe_ts]:.4f}")
            print(f"      [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 关键计算节点值 ---")
            print(f"        panic_source_score: {panic_source_score.loc[probe_ts]:.4f}")
            print(f"        panic_slope_dampening_factor: {panic_slope_dampening_factor.loc[probe_ts]:.4f}")
            print(f"        panic_source_score_modulated: {panic_source_score_modulated.loc[probe_ts]:.4f}")
            print(f"        is_panic_context: {is_panic_context.loc[probe_ts]}")
            print(f"        counter_flow_medium_score: {counter_flow_medium_score.loc[probe_ts]:.4f}")
            print(f"        main_force_echo_score: {main_force_echo_score.loc[probe_ts]:.4f}")
            print(f"        deception_modulator: {deception_modulator.loc[probe_ts]:.4f}")
            print(f"        context_modulator: {context_modulator.loc[probe_ts]:.4f}")
            print(f"        base_score: {base_score.loc[probe_ts]:.4f}")
            print(f"        bull_trap_penalty: {bull_trap_penalty.loc[probe_ts]:.4f}")
            print(f"      [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 最终吸筹回声得分 (final_score): {final_score.loc[probe_ts]:.4f}")
            print(f"  -- [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: “吸筹回声”信号诊断完成。")
        else:
            print(f"  -- [筹码层] “吸筹回声”信号诊断完成，最新分值: {final_score.iloc[-1]:.4f}")
        return final_score

    def _diagnose_distribution_whisper(self, df: pd.DataFrame, divergence_score: pd.Series) -> pd.Series:
        """
        【V4.2 · Numba优化版 & 探针增强版】诊断“派发诡影”信号
        - 核心优化: 将deception_modulator的计算逻辑迁移至Numba加速的辅助函数。
        - 核心升级1: 狂热背景深度化。在V3.0基础上，引入总赢家比例、总输家比例、赢家输家动量及其短期斜率，更全面刻画市场狂热和筹码结构膨胀。
        - 核心升级2: 背离诡影精细化。在V3.0基础上，引入主峰利润率、主峰坚实度、上影线抛压、压力拒绝强度及其短期斜率，评估主力派发动力、筹码结构松动和承接力减弱。
        - 核心升级3: 主力抽离多维度验证。在V3.0基础上，引入主力净流量校准、主力滑点指数及其短期加速度、反弹派发压力、控制坚实度、对手盘枯竭和智能资金净买入负向，多角度验证主力隐蔽、坚决派发。
        - 核心升级4: 诡道背景调制强化。引入欺骗指数，结合筹码故障幅度与主力信念指数，更智能地判断诡道意图并进行调制。
        - 探针增强: 详细输出所有原始数据、归一化数据、各维度子分数、动态权重、最终分数，以便于检查和调试。
        """
        method_name = "_diagnose_distribution_whisper"
        df_index = df.index
        required_signals = [
            'retail_fomo_premium_index_D', 'winner_profit_margin_avg_D', 'THEME_HOTNESS_SCORE_D', 'market_sentiment_score_D', 'winner_concentration_90pct_D',
            'total_winner_rate_D', 'winner_loser_momentum_D', 'SLOPE_5_winner_loser_momentum_D',
            'dispersal_by_distribution_D', 'profit_taking_flow_ratio_D', 'chip_fault_magnitude_D',
            'cost_structure_skewness_D', 'winner_stability_index_D', 'chip_fault_blockage_ratio_D',
            'dominant_peak_profit_margin_D', 'dominant_peak_solidity_D', 'upper_shadow_selling_pressure_D', 'pressure_rejection_strength_D',
            'SLOPE_5_dominant_peak_solidity_D', 'SLOPE_5_pressure_rejection_strength_D',
            'covert_accumulation_signal_D', 'wash_trade_intensity_D', 'main_force_conviction_index_D', 'retail_flow_dominance_index_D',
            'main_force_net_flow_calibrated_D', 'main_force_slippage_index_D', 'rally_distribution_pressure_D', 'control_solidity_index_D',
            'counterparty_exhaustion_index_D', 'SMART_MONEY_HM_NET_BUY_D',
            'SLOPE_5_main_force_net_flow_calibrated_D', 'ACCEL_5_main_force_slippage_index_D',
            'deception_index_D'
        ]
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index)
        signals_data = self._get_all_required_signals(df, required_signals, method_name)
        p_conf = self.chip_ultimate_params
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        parsed_tf_data = utils._parse_tf_weights(tf_weights)
        distribution_whisper_params = get_param_value(p_conf.get('distribution_whisper_params'), {})
        fomo_backdrop_weights = get_param_value(distribution_whisper_params.get('fomo_backdrop_weights'), {'retail_fomo_premium': 0.2, 'winner_profit_margin': 0.2, 'theme_hotness': 0.15, 'market_sentiment_positive': 0.1, 'winner_concentration_negative': 0.1, 'total_winner_rate': 0.15, 'winner_loser_momentum': 0.05, 'winner_loser_momentum_slope': 0.05})
        fomo_context_threshold = get_param_value(distribution_whisper_params.get('fomo_context_threshold'), 0.3)
        divergence_shadow_weights = get_param_value(distribution_whisper_params.get('divergence_shadow_weights'), {'divergence_bearish': 0.15, 'distribution_intensity': 0.15, 'chip_fault_magnitude': 0.1, 'cost_structure_negative': 0.1, 'winner_stability_negative': 0.1, 'chip_fault_blockage': 0.1, 'dominant_peak_profit_margin': 0.1, 'dominant_peak_solidity_negative': 0.05, 'upper_shadow_selling_pressure': 0.05, 'pressure_rejection_strength_negative': 0.05, 'dominant_peak_solidity_slope_negative': 0.025, 'pressure_rejection_strength_slope_negative': 0.025})
        main_force_retreat_weights = get_param_value(distribution_whisper_params.get('main_force_retreat_weights'), {'profit_taking_flow': 0.15, 'dispersal_by_distribution': 0.15, 'covert_accumulation_negative': 0.1, 'wash_trade_intensity': 0.1, 'main_force_conviction_negative': 0.1, 'retail_flow_dominance': 0.1, 'main_force_net_flow_negative': 0.1, 'main_force_slippage': 0.05, 'rally_distribution_pressure': 0.05, 'control_solidity_negative': 0.05, 'counterparty_exhaustion': 0.025, 'smart_money_net_buy_negative': 0.025, 'main_force_net_flow_slope_negative': 0.025, 'main_force_slippage_accel': 0.025})
        deception_modulator_params = get_param_value(distribution_whisper_params.get('deception_modulator_params'), {'boost_factor': 0.6, 'penalty_factor': 0.4, 'conviction_threshold': 0.2, 'deception_index_weight': 0.5})
        final_fusion_exponent = get_param_value(distribution_whisper_params.get('final_fusion_exponent'), 0.25)
        is_debug_enabled = self.should_probe
        probe_ts = None
        if is_debug_enabled and self.probe_dates_set:
            for date in reversed(df_index):
                if date.date() in self.probe_dates_set:
                    probe_ts = date
                    break
        debug_info_tuple = (is_debug_enabled, probe_ts, method_name)
        if is_debug_enabled and probe_ts and probe_ts in df_index:
            print(f"  -- [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在诊断“派发诡影”信号...")
            print(f"      [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---")
            for sig_name in required_signals:
                val = signals_data[sig_name].loc[probe_ts] if probe_ts in signals_data[sig_name].index else np.nan
                print(f"        '{sig_name}': {val:.4f}")
        retail_fomo_premium_raw = signals_data['retail_fomo_premium_index_D']
        winner_profit_margin_raw = signals_data['winner_profit_margin_avg_D']
        theme_hotness_raw = signals_data['THEME_HOTNESS_SCORE_D']
        market_sentiment_raw = signals_data['market_sentiment_score_D']
        winner_concentration_raw = signals_data['winner_concentration_90pct_D']
        total_winner_rate_raw = signals_data['total_winner_rate_D']
        winner_loser_momentum_raw = signals_data['winner_loser_momentum_D']
        slope_5_winner_loser_momentum_raw = signals_data['SLOPE_5_winner_loser_momentum_D']
        dispersal_by_distribution_raw = signals_data['dispersal_by_distribution_D']
        chip_fault_magnitude_raw = signals_data['chip_fault_magnitude_D']
        cost_structure_skewness_raw = signals_data['cost_structure_skewness_D']
        winner_stability_raw = signals_data['winner_stability_index_D']
        chip_fault_blockage_raw = signals_data['chip_fault_blockage_ratio_D']
        dominant_peak_profit_margin_raw = signals_data['dominant_peak_profit_margin_D']
        dominant_peak_solidity_raw = signals_data['dominant_peak_solidity_D']
        upper_shadow_selling_pressure_raw = signals_data['upper_shadow_selling_pressure_D']
        pressure_rejection_strength_raw = signals_data['pressure_rejection_strength_D']
        slope_5_dominant_peak_solidity_raw = signals_data['SLOPE_5_dominant_peak_solidity_D']
        slope_5_pressure_rejection_strength_raw = signals_data['SLOPE_5_pressure_rejection_strength_D']
        profit_taking_flow_ratio_raw = signals_data['profit_taking_flow_ratio_D']
        covert_accumulation_raw = signals_data['covert_accumulation_signal_D']
        wash_trade_intensity_raw = signals_data['wash_trade_intensity_D']
        main_force_conviction_raw = signals_data['main_force_conviction_index_D']
        retail_flow_dominance_raw = signals_data['retail_flow_dominance_index_D']
        main_force_net_flow_calibrated_raw = signals_data['main_force_net_flow_calibrated_D']
        main_force_slippage_raw = signals_data['main_force_slippage_index_D']
        rally_distribution_pressure_raw = signals_data['rally_distribution_pressure_D']
        control_solidity_raw = signals_data['control_solidity_index_D']
        counterparty_exhaustion_raw = signals_data['counterparty_exhaustion_index_D']
        smart_money_net_buy_raw = signals_data['SMART_MONEY_HM_NET_BUY_D']
        slope_5_main_force_net_flow_calibrated_raw = signals_data['SLOPE_5_main_force_net_flow_calibrated_D']
        accel_5_main_force_slippage_raw = signals_data['ACCEL_5_main_force_slippage_index_D']
        deception_index_raw = signals_data['deception_index_D']
        # 1. 狂热背景深度化 (FOMO Backdrop)
        norm_retail_fomo_premium = get_adaptive_mtf_normalized_score(retail_fomo_premium_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_winner_profit_margin = get_adaptive_mtf_normalized_score(winner_profit_margin_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_theme_hotness = get_adaptive_mtf_normalized_score(theme_hotness_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_market_sentiment_positive = get_adaptive_mtf_normalized_bipolar_score(market_sentiment_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data).clip(0, 1)
        norm_winner_concentration_negative = get_adaptive_mtf_normalized_score(winner_concentration_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_total_winner_rate = get_adaptive_mtf_normalized_score(total_winner_rate_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_winner_loser_momentum = get_adaptive_mtf_normalized_score(winner_loser_momentum_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_slope_5_winner_loser_momentum = get_adaptive_mtf_normalized_score(slope_5_winner_loser_momentum_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        if is_debug_enabled and probe_ts and probe_ts in df_index:
            print(f"      [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 狂热背景归一化信号 ---")
            print(f"        norm_retail_fomo_premium: {norm_retail_fomo_premium.loc[probe_ts]:.4f}")
            print(f"        norm_winner_profit_margin: {norm_winner_profit_margin.loc[probe_ts]:.4f}")
            print(f"        norm_theme_hotness: {norm_theme_hotness.loc[probe_ts]:.4f}")
            print(f"        norm_market_sentiment_positive: {norm_market_sentiment_positive.loc[probe_ts]:.4f}")
            print(f"        norm_winner_concentration_negative: {norm_winner_concentration_negative.loc[probe_ts]:.4f}")
            print(f"        norm_total_winner_rate: {norm_total_winner_rate.loc[probe_ts]:.4f}")
            print(f"        norm_winner_loser_momentum: {norm_winner_loser_momentum.loc[probe_ts]:.4f}")
            print(f"        norm_slope_5_winner_loser_momentum: {norm_slope_5_winner_loser_momentum.loc[probe_ts]:.4f}")
        fomo_backdrop_numeric_weights = {k: v for k, v in fomo_backdrop_weights.items() if isinstance(v, (int, float))}
        fomo_backdrop_score = _robust_geometric_mean(
            {
                'retail_fomo_premium': norm_retail_fomo_premium,
                'winner_profit_margin': norm_winner_profit_margin,
                'theme_hotness': norm_theme_hotness,
                'market_sentiment_positive': norm_market_sentiment_positive,
                'winner_concentration_negative': norm_winner_concentration_negative,
                'total_winner_rate': norm_total_winner_rate,
                'winner_loser_momentum': norm_winner_loser_momentum,
                'winner_loser_momentum_slope': norm_slope_5_winner_loser_momentum
            },
            fomo_backdrop_numeric_weights, df_index
        )
        if is_debug_enabled and probe_ts and probe_ts in df_index:
            print(f"      [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 狂热背景得分 (fomo_backdrop_score): {fomo_backdrop_score.loc[probe_ts]:.4f}")
        # 初始化 is_fomo_context，确保它总是有值
        is_fomo_context = pd.Series(False, index=df_index, dtype=bool)
        is_fomo_context = fomo_backdrop_score > fomo_context_threshold
        if is_debug_enabled and probe_ts and probe_ts in df_index:
            print(f"        是否处于FOMO情境 (is_fomo_context): {is_fomo_context.loc[probe_ts]}")
        # 2. 背离诡影精细化 (Divergence Shadow)
        norm_divergence_bearish = divergence_score.clip(-1, 0).abs() # 仅取负向部分（看跌背离）的绝对值
        norm_dispersal_by_distribution = utils.get_adaptive_mtf_normalized_score(dispersal_by_distribution_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_chip_fault_magnitude_for_shadow = utils.get_adaptive_mtf_normalized_score(chip_fault_magnitude_raw.abs(), df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_cost_structure_negative = utils.get_adaptive_mtf_normalized_score(cost_structure_skewness_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_winner_stability_negative = utils.get_adaptive_mtf_normalized_score(winner_stability_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_chip_fault_blockage = utils.get_adaptive_mtf_normalized_score(chip_fault_blockage_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_dominant_peak_profit_margin = utils.get_adaptive_mtf_normalized_score(dominant_peak_profit_margin_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_dominant_peak_solidity_negative = utils.get_adaptive_mtf_normalized_score(dominant_peak_solidity_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_upper_shadow_selling_pressure = utils.get_adaptive_mtf_normalized_score(upper_shadow_selling_pressure_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_pressure_rejection_strength_negative = utils.get_adaptive_mtf_normalized_score(pressure_rejection_strength_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_slope_5_dominant_peak_solidity_negative = utils.get_adaptive_mtf_normalized_score(slope_5_dominant_peak_solidity_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_slope_5_pressure_rejection_strength_negative = utils.get_adaptive_mtf_normalized_score(slope_5_pressure_rejection_strength_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        if is_debug_enabled and probe_ts and probe_ts in df_index:
            print(f"      [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 背离诡影归一化信号 ---")
            print(f"        norm_divergence_bearish: {norm_divergence_bearish.loc[probe_ts]:.4f}")
            print(f"        norm_dispersal_by_distribution: {norm_dispersal_by_distribution.loc[probe_ts]:.4f}")
            print(f"        norm_chip_fault_magnitude_for_shadow: {norm_chip_fault_magnitude_for_shadow.loc[probe_ts]:.4f}")
            print(f"        norm_cost_structure_negative: {norm_cost_structure_negative.loc[probe_ts]:.4f}")
            print(f"        norm_winner_stability_negative: {norm_winner_stability_negative.loc[probe_ts]:.4f}")
            print(f"        norm_chip_fault_blockage: {norm_chip_fault_blockage.loc[probe_ts]:.4f}")
            print(f"        norm_dominant_peak_profit_margin: {norm_dominant_peak_profit_margin.loc[probe_ts]:.4f}")
            print(f"        norm_dominant_peak_solidity_negative: {norm_dominant_peak_solidity_negative.loc[probe_ts]:.4f}")
            print(f"        norm_upper_shadow_selling_pressure: {norm_upper_shadow_selling_pressure.loc[probe_ts]:.4f}")
            print(f"        norm_pressure_rejection_strength_negative: {norm_pressure_rejection_strength_negative.loc[probe_ts]:.4f}")
            print(f"        norm_slope_5_dominant_peak_solidity_negative: {norm_slope_5_dominant_peak_solidity_negative.loc[probe_ts]:.4f}")
            print(f"        norm_slope_5_pressure_rejection_strength_negative: {norm_slope_5_pressure_rejection_strength_negative.loc[probe_ts]:.4f}")
        divergence_shadow_numeric_weights = {k: v for k, v in divergence_shadow_weights.items() if isinstance(v, (int, float))}
        divergence_shadow_score = utils._robust_geometric_mean(
            {
                'divergence_bearish': norm_divergence_bearish,
                'distribution_intensity': norm_dispersal_by_distribution,
                'chip_fault_magnitude': norm_chip_fault_magnitude_for_shadow,
                'cost_structure_negative': norm_cost_structure_negative,
                'winner_stability_negative': norm_winner_stability_negative,
                'chip_fault_blockage': norm_chip_fault_blockage,
                'dominant_peak_profit_margin': norm_dominant_peak_profit_margin,
                'dominant_peak_solidity_negative': norm_dominant_peak_solidity_negative,
                'upper_shadow_selling_pressure': norm_upper_shadow_selling_pressure,
                'pressure_rejection_strength_negative': norm_pressure_rejection_strength_negative,
                'dominant_peak_solidity_slope_negative': norm_slope_5_dominant_peak_solidity_negative,
                'pressure_rejection_strength_slope_negative': norm_slope_5_pressure_rejection_strength_negative
            },
            divergence_shadow_numeric_weights, df_index)
        if is_debug_enabled and probe_ts and probe_ts in df_index:
            print(f"      [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 背离诡影得分 (divergence_shadow_score): {divergence_shadow_score.loc[probe_ts]:.4f}")
        # 3. 主力抽离多维度验证 (Main Force Retreat)
        norm_profit_taking_flow_ratio = utils.get_adaptive_mtf_normalized_score(profit_taking_flow_ratio_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_covert_accumulation_negative = utils.get_adaptive_mtf_normalized_score(covert_accumulation_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_wash_trade_intensity = utils.get_adaptive_mtf_normalized_score(wash_trade_intensity_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_main_force_conviction_negative = utils.get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data).clip(-1, 0).abs()
        norm_retail_flow_dominance = utils.get_adaptive_mtf_normalized_score(retail_flow_dominance_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_main_force_net_flow_negative = utils.get_adaptive_mtf_normalized_score(main_force_net_flow_calibrated_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_main_force_slippage = utils.get_adaptive_mtf_normalized_score(main_force_slippage_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_rally_distribution_pressure = utils.get_adaptive_mtf_normalized_score(rally_distribution_pressure_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_control_solidity_negative = utils.get_adaptive_mtf_normalized_score(control_solidity_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_counterparty_exhaustion = utils.get_adaptive_mtf_normalized_score(counterparty_exhaustion_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_smart_money_net_buy_negative = utils.get_adaptive_mtf_normalized_score(smart_money_net_buy_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_slope_5_main_force_net_flow_calibrated_negative = utils.get_adaptive_mtf_normalized_score(slope_5_main_force_net_flow_calibrated_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_accel_5_main_force_slippage = utils.get_adaptive_mtf_normalized_score(accel_5_main_force_slippage_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        if is_debug_enabled and probe_ts and probe_ts in df_index:
            print(f"      [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 主力抽离归一化信号 ---")
            print(f"        norm_profit_taking_flow_ratio: {norm_profit_taking_flow_ratio.loc[probe_ts]:.4f}")
            print(f"        norm_covert_accumulation_negative: {norm_covert_accumulation_negative.loc[probe_ts]:.4f}")
            print(f"        norm_wash_trade_intensity: {norm_wash_trade_intensity.loc[probe_ts]:.4f}")
            print(f"        norm_main_force_conviction_negative: {norm_main_force_conviction_negative.loc[probe_ts]:.4f}")
            print(f"        norm_retail_flow_dominance: {norm_retail_flow_dominance.loc[probe_ts]:.4f}")
            print(f"        norm_main_force_net_flow_negative: {norm_main_force_net_flow_negative.loc[probe_ts]:.4f}")
            print(f"        norm_main_force_slippage: {norm_main_force_slippage.loc[probe_ts]:.4f}")
            print(f"        norm_rally_distribution_pressure: {norm_rally_distribution_pressure.loc[probe_ts]:.4f}")
            print(f"        norm_control_solidity_negative: {norm_control_solidity_negative.loc[probe_ts]:.4f}")
            print(f"        norm_counterparty_exhaustion: {norm_counterparty_exhaustion.loc[probe_ts]:.4f}")
            print(f"        norm_smart_money_net_buy_negative: {norm_smart_money_net_buy_negative.loc[probe_ts]:.4f}")
            print(f"        norm_slope_5_main_force_net_flow_calibrated_negative: {norm_slope_5_main_force_net_flow_calibrated_negative.loc[probe_ts]:.4f}")
            print(f"        norm_accel_5_main_force_slippage: {norm_accel_5_main_force_slippage.loc[probe_ts]:.4f}")
        main_force_retreat_numeric_weights = {k: v for k, v in main_force_retreat_weights.items() if isinstance(v, (int, float))}
        main_force_retreat_score = utils._robust_geometric_mean(
            {
                'profit_taking_flow': norm_profit_taking_flow_ratio,
                'dispersal_by_distribution': norm_dispersal_by_distribution,
                'covert_accumulation_negative': norm_covert_accumulation_negative,
                'wash_trade_intensity': norm_wash_trade_intensity,
                'main_force_conviction_negative': norm_main_force_conviction_negative,
                'retail_flow_dominance': norm_retail_flow_dominance,
                'main_force_net_flow_negative': norm_main_force_net_flow_negative,
                'main_force_slippage': norm_main_force_slippage,
                'rally_distribution_pressure': norm_rally_distribution_pressure,
                'control_solidity_negative': norm_control_solidity_negative,
                'counterparty_exhaustion': norm_counterparty_exhaustion,
                'smart_money_net_buy_negative': norm_smart_money_net_buy_negative,
                'main_force_net_flow_slope_negative': norm_slope_5_main_force_net_flow_calibrated_negative,
                'main_force_slippage_accel': norm_accel_5_main_force_slippage
            },
            main_force_retreat_numeric_weights, df_index)
        if is_debug_enabled and probe_ts and probe_ts in df_index:
            print(f"      [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 主力抽离得分 (main_force_retreat_score): {main_force_retreat_score.loc[probe_ts]:.4f}")
        # 4. 诡道背景调制强化 (Deception Modulator)
        norm_chip_fault_magnitude_bipolar = utils.get_adaptive_mtf_normalized_bipolar_score(chip_fault_magnitude_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_main_force_conviction_bipolar = utils.get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_deception_index_bipolar = utils.get_adaptive_mtf_normalized_bipolar_score(deception_index_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        if is_debug_enabled and probe_ts and probe_ts in df_index:
            print(f"      [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 诡道背景调制归一化信号 ---")
            print(f"        norm_chip_fault_magnitude_bipolar: {norm_chip_fault_magnitude_bipolar.loc[probe_ts]:.4f}")
            print(f"        norm_main_force_conviction_bipolar: {norm_main_force_conviction_bipolar.loc[probe_ts]:.4f}")
            print(f"        norm_deception_index_bipolar: {norm_deception_index_bipolar.loc[probe_ts]:.4f}")
        deception_modulator_values = _numba_calculate_distribution_whisper_deception_modulator_core(
            norm_chip_fault_magnitude_bipolar.values,
            norm_main_force_conviction_bipolar.values,
            norm_deception_index_bipolar.values,
            deception_modulator_params.get('boost_factor', 0.6),
            deception_modulator_params.get('penalty_factor', 0.4),
            deception_modulator_params.get('conviction_threshold', 0.2),
            deception_modulator_params.get('deception_index_weight', 0.5)
        )
        deception_modulator = pd.Series(deception_modulator_values, index=df_index, dtype=np.float32)
        if is_debug_enabled and probe_ts and probe_ts in df_index:
            print(f"      [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 诡道调制器 (deception_modulator): {deception_modulator.loc[probe_ts]:.4f}")
        # 5. 最终融合
        base_score = (
            fomo_backdrop_score.pow(final_fusion_exponent) *
            divergence_shadow_score.pow(final_fusion_exponent) *
            main_force_retreat_score.pow(final_fusion_exponent)
        ).pow(1 / (3 * final_fusion_exponent))
        if is_debug_enabled and probe_ts and probe_ts in df_index:
            print(f"      [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 基础融合得分 (base_score): {base_score.loc[probe_ts]:.4f}")
        final_score = (base_score * deception_modulator) * is_fomo_context
        final_score = final_score.clip(0, 1).fillna(0.0).astype(np.float32)
        if is_debug_enabled and probe_ts and probe_ts in df_index:
            print(f"      [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 最终派发诡影得分 (final_score): {final_score.loc[probe_ts]:.4f}")
            print(f"  -- [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: “派发诡影”信号诊断完成。")
        else:
            print(f"  -- [筹码层] “派发诡影”信号诊断完成，最新分值: {final_score.iloc[-1]:.4f}")
        return final_score

    def _diagnose_axiom_historical_potential(self, df: pd.DataFrame) -> pd.Series:
        """
        【V5.9 · Numba优化版】筹码公理六：诊断“筹码势能”
        - 核心优化: 将dgm_score的计算逻辑迁移至Numba加速的辅助函数。
        - 核心升级1: 主力吸筹质量 (MF_AQ)。引入“吸筹效率的非对称性”，结合主力成本优势和筹码健康度动态调整吸筹模式权重，并考虑主力执行效率和非对称摩擦指数等高频聚合信号。
        - 核心升级2: 筹码结构张力 (CST)。引入“结构临界点识别”，结合赢家/输家集中度斜率预判结构转折，并考虑结构张力指数和结构熵变。
        - 核心升级3: 势能转化效率 (PCE)。引入“阻力位博弈强度”，评估关键阻力位和支撑位的博弈激烈程度，并考虑订单簿清算率和微观价格冲击不对称性等微观层面的阻力消化能力。
        - 核心升级4: 诡道博弈调制 (DGM)。引入“诡道博弈的非对称影响”，对诱多/诱空施加不同敏感度的调制，并考虑散户恐慌和主力信念对诡道博弈有效性的影响。
        - 核心升级5: 情境自适应权重 (ACW)。引入“市场情绪与流动性情境”，增加市场情绪分数和资金流可信度指数作为情境调制器。
        - 核心修复: 增强 `dgm_score` 的惩罚机制，当出现“诱空”且主力资金流出时，强制将分数设置为极低负值，并确保其优先级，修正主力流出判断逻辑。
        - 纯筹码化：移除对资金类信号（如主力资金流向、资金流可信度）的依赖，替换为纯筹码类指标。
        - **新增业务逻辑：引入“牛市陷阱情境惩罚”，在近期大幅下跌后伴随正向欺骗时，大幅降低筹码势能的得分。**
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        method_name = "_diagnose_axiom_historical_potential"
        df_index = df.index
        required_signals = [
            'covert_accumulation_signal_D', 'suppressive_accumulation_intensity_D',
            'main_force_cost_advantage_D', 'floating_chip_cleansing_efficiency_D',
            'chip_health_score_D', 'dominant_peak_solidity_D',
            'SLOPE_5_cost_structure_skewness_D', 'SLOPE_5_peak_separation_ratio_D',
            'vacuum_zone_magnitude_D', 'vacuum_traversal_efficiency_D',
            'VOLATILITY_INSTABILITY_INDEX_21d_D', 'chip_fatigue_index_D',
            'chip_fault_magnitude_D',
            'winner_stability_index_D', 'loser_pain_index_D',
            'active_selling_pressure_D', 'capitulation_absorption_index_D',
            'deception_index_D', 'wash_trade_intensity_D',
            'main_force_execution_alpha_D', 'asymmetric_friction_index_D',
            'SLOPE_5_winner_concentration_90pct_D', 'SLOPE_5_loser_concentration_90pct_D',
            'structural_tension_index_D', 'structural_entropy_change_D',
            'pressure_rejection_strength_D', 'support_validation_strength_D',
            'order_book_clearing_rate_D', 'micro_price_impact_asymmetry_D',
            'retail_panic_surrender_index_D', 'main_force_conviction_index_D',
            'market_sentiment_score_D',
            'conviction_flow_buy_intensity_D', 'conviction_flow_sell_intensity_D',
            'pct_change_D'
        ]
        p_conf = self.chip_ultimate_params
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        parsed_tf_data = utils._parse_tf_weights(tf_weights)
        historical_potential_params = get_param_value(p_conf.get('historical_potential_params'), {})
        mf_aq_weights = get_param_value(historical_potential_params.get('mf_aq_weights'), {
            'covert_accumulation': 0.25, 'suppressive_accumulation': 0.15,
            'cost_advantage': 0.25, 'cleansing_efficiency': 0.15, 'deception_purity_factor': 0.1,
            'execution_alpha': 0.05, 'friction_index': 0.05
        })
        mf_aq_asymmetry_params = get_param_value(historical_potential_params.get('mf_aq_asymmetry_params'), {
            'cost_advantage_threshold': 0.0, 'chip_health_threshold': 0.0,
            'covert_weight_boost': 0.2, 'suppressive_weight_boost': 0.1
        })
        cst_weights = get_param_value(historical_potential_params.get('cst_weights'), {
            'chip_health': 0.2, 'peak_solidity': 0.2,
            'cost_skewness_slope': 0.1, 'peak_separation_slope': 0.1, 'structural_elasticity': 0.15,
            'concentration_slope_divergence': 0.15, 'structural_tension': 0.05, 'structural_entropy': 0.05
        })
        pce_weights = get_param_value(historical_potential_params.get('pce_weights'), {
            'vacuum_magnitude': 0.3, 'vacuum_efficiency': 0.3, 'resistance_absorption': 0.2,
            'resistance_game_strength_weight': 0.2,
            'order_book_clearing_rate': 0.05, 'micro_price_impact_asymmetry': 0.05
        })
        dgm_weights = get_param_value(historical_potential_params.get('dgm_weights'), {
            'deception_impact': 0.4, 'wash_trade_penalty': 0.2, 'flow_directionality_boost': 0.1,
            'retail_panic_impact': 0.15, 'main_force_conviction_impact': 0.15
        })
        dgm_asymmetry_params = get_param_value(historical_potential_params.get('dgm_asymmetry_params'), {
            'bull_trap_penalty_factor': 1.5, 'bear_trap_bonus_factor': 1.2,
            'bull_trap_ll_penalty_factor': 0.5, 'bear_trap_ls_bonus_factor': 0.5
        })
        final_fusion_weights = get_param_value(historical_potential_params.get('final_fusion_weights'), {
            'mf_aq': 0.35, 'cst': 0.3, 'pce': 0.35
        })
        context_modulator_signals = get_param_value(historical_potential_params.get('context_modulator_signals'), {
            'volatility_instability': {'signal_name': 'VOLATILITY_INSTABILITY_INDEX_21d_D', 'weight': 0.3, 'ascending': False},
            'chip_fatigue': {'signal_name': 'chip_fatigue_index_D', 'weight': 0.2, 'ascending': False},
            'market_sentiment': {'signal_name': 'market_sentiment_score_D', 'weight': 0.3, 'ascending': True},
            'flow_credibility': {'signal_name': 'flow_credibility_index_D', 'weight': 0.2, 'ascending': True}
        })
        context_modulator_sensitivity = get_param_value(historical_potential_params.get('context_modulator_sensitivity'), 0.5)
        dgm_modulator_sensitivity = get_param_value(historical_potential_params.get('dgm_modulator_sensitivity'), 0.8)
        bearish_deception_and_mf_out_penalty_factor = get_param_value(historical_potential_params.get('bearish_deception_and_mf_out_penalty_factor'), 1.0)
        retail_panic_positive_pct_penalty_factor = get_param_value(historical_potential_params.get('retail_panic_positive_pct_penalty_factor'), 0.5) # 新增参数获取
        context_modulator_signals['flow_credibility']['signal_name'] = 'winner_stability_index_D'
        context_modulator_signals['flow_credibility']['ascending'] = True
        for ctx_key, ctx_config in context_modulator_signals.items():
            signal_name = ctx_config.get('signal_name')
            if signal_name and signal_name not in required_signals:
                required_signals.append(signal_name)
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index)
        signals_data = self._get_all_required_signals(df, required_signals, method_name)
        is_debug_enabled = self.should_probe
        probe_ts = None
        if is_debug_enabled and self.probe_dates_set:
            for date in reversed(df_index):
                if date.date() in self.probe_dates_set:
                    probe_ts = date
                    break
        debug_info_tuple = False
        chip_health_raw = signals_data['chip_health_score_D']
        norm_chip_health = utils.get_adaptive_mtf_normalized_bipolar_score(chip_health_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        covert_accumulation_raw = signals_data['covert_accumulation_signal_D']
        suppressive_accumulation_raw = signals_data['suppressive_accumulation_intensity_D']
        main_force_cost_advantage_raw = signals_data['main_force_cost_advantage_D']
        floating_chip_cleansing_efficiency_raw = signals_data['floating_chip_cleansing_efficiency_D']
        chip_fault_magnitude_raw = signals_data['chip_fault_magnitude_D']
        main_force_execution_alpha_raw = signals_data['main_force_execution_alpha_D']
        asymmetric_friction_index_raw = signals_data['asymmetric_friction_index_D']
        conviction_flow_buy_raw = signals_data['conviction_flow_buy_intensity_D']
        conviction_flow_sell_raw = signals_data['conviction_flow_sell_intensity_D']
        winner_stability_raw = signals_data['winner_stability_index_D']
        norm_covert_accumulation = utils.get_adaptive_mtf_normalized_score(covert_accumulation_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_suppressive_accumulation = utils.get_adaptive_mtf_normalized_score(suppressive_accumulation_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_main_force_cost_advantage = utils.get_adaptive_mtf_normalized_bipolar_score(main_force_cost_advantage_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_floating_chip_cleansing_efficiency = utils.get_adaptive_mtf_normalized_score(floating_chip_cleansing_efficiency_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_chip_fault_magnitude = utils.get_adaptive_mtf_normalized_bipolar_score(chip_fault_magnitude_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_main_force_execution_alpha = utils.get_adaptive_mtf_normalized_score(main_force_execution_alpha_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_asymmetric_friction_index = utils.get_adaptive_mtf_normalized_score(asymmetric_friction_index_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        deception_purity_adjustment = pd.Series(1.0, index=df_index)
        deception_purity_adjustment = 1 + (norm_chip_fault_magnitude * -1) * mf_aq_weights.get('deception_purity_factor', 0.1)
        deception_purity_adjustment = deception_purity_adjustment.clip(0.5, 1.5)
        dynamic_covert_weight = pd.Series(mf_aq_weights.get('covert_accumulation', 0.25), index=df_index)
        dynamic_suppressive_weight = pd.Series(mf_aq_weights.get('suppressive_accumulation', 0.15), index=df_index)
        low_health_low_cost_advantage_mask = (norm_chip_health < mf_aq_asymmetry_params.get('chip_health_threshold', 0.0)) & \
                                             (norm_main_force_cost_advantage < mf_aq_asymmetry_params.get('cost_advantage_threshold', 0.0))
        dynamic_covert_weight.loc[low_health_low_cost_advantage_mask] += mf_aq_asymmetry_params.get('covert_weight_boost', 0.2)
        dynamic_suppressive_weight.loc[low_health_low_cost_advantage_mask] -= mf_aq_asymmetry_params.get('suppressive_weight_boost', 0.1)
        base_mf_aq_total_weight = mf_aq_weights.get('covert_accumulation', 0.25) + mf_aq_weights.get('suppressive_accumulation', 0.15) + \
                                  mf_aq_weights.get('cost_advantage', 0.25) + mf_aq_weights.get('cleansing_efficiency', 0.15) + \
                                  mf_aq_weights.get('execution_alpha', 0.05) + mf_aq_weights.get('friction_index', 0.05)
        sum_dynamic_weights_mf_aq = dynamic_covert_weight + dynamic_suppressive_weight + \
                                    mf_aq_weights.get('cost_advantage', 0.25) + mf_aq_weights.get('cleansing_efficiency', 0.15) + \
                                    mf_aq_weights.get('execution_alpha', 0.05) + mf_aq_weights.get('friction_index', 0.05)
        mf_aq_score = (
            (norm_covert_accumulation * dynamic_covert_weight) +
            (norm_suppressive_accumulation * dynamic_suppressive_weight) +
            ((norm_main_force_cost_advantage.add(1)/2) * mf_aq_weights.get('cost_advantage', 0.25)) +
            (norm_floating_chip_cleansing_efficiency * mf_aq_weights.get('cleansing_efficiency', 0.15)) +
            (norm_main_force_execution_alpha * mf_aq_weights.get('execution_alpha', 0.05)) +
            (norm_asymmetric_friction_index * mf_aq_weights.get('friction_index', 0.05))
        ) / sum_dynamic_weights_mf_aq.replace(0, 1e-6) * base_mf_aq_total_weight
        mf_aq_score = mf_aq_score * deception_purity_adjustment
        mf_aq_score = mf_aq_score.clip(0, 1)
        dominant_peak_solidity_raw = signals_data['dominant_peak_solidity_D']
        cost_structure_skewness_slope_raw = signals_data['SLOPE_5_cost_structure_skewness_D']
        peak_separation_ratio_slope_raw = signals_data['SLOPE_5_peak_separation_ratio_D']
        winner_stability_raw = signals_data['winner_stability_index_D']
        loser_pain_raw = signals_data['loser_pain_index_D']
        winner_concentration_slope_raw = signals_data['SLOPE_5_winner_concentration_90pct_D']
        loser_concentration_slope_raw = signals_data['SLOPE_5_loser_concentration_90pct_D']
        structural_tension_raw = signals_data['structural_tension_index_D']
        structural_entropy_change_raw = signals_data['structural_entropy_change_D']
        norm_dominant_peak_solidity = utils.get_adaptive_mtf_normalized_score(dominant_peak_solidity_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_cost_structure_skewness_slope = utils.get_adaptive_mtf_normalized_bipolar_score(cost_structure_skewness_slope_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_peak_separation_ratio_slope = utils.get_adaptive_mtf_normalized_bipolar_score(peak_separation_ratio_slope_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_winner_stability = utils.get_adaptive_mtf_normalized_score(winner_stability_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_loser_pain = utils.get_adaptive_mtf_normalized_score(loser_pain_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_winner_concentration_slope = utils.get_adaptive_mtf_normalized_bipolar_score(winner_concentration_slope_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_loser_concentration_slope = utils.get_adaptive_mtf_normalized_bipolar_score(loser_concentration_slope_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_structural_tension = utils.get_adaptive_mtf_normalized_score(structural_tension_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_structural_entropy_change = utils.get_adaptive_mtf_normalized_score(structural_entropy_change_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        structural_elasticity_score = (norm_winner_stability * 0.5 + norm_loser_pain * 0.5).clip(0, 1)
        concentration_slope_divergence = (norm_winner_concentration_slope - norm_loser_concentration_slope).clip(-1, 1)
        cst_components = {
            'chip_health': (norm_chip_health + 1) / 2,
            'peak_solidity': norm_dominant_peak_solidity,
            'cost_skewness_slope': (1 - (norm_cost_structure_skewness_slope + 1) / 2),
            'peak_separation_slope': (1 - (norm_peak_separation_ratio_slope + 1) / 2),
            'structural_elasticity': structural_elasticity_score,
            'concentration_slope_divergence': (concentration_slope_divergence + 1) / 2,
            'structural_tension': norm_structural_tension,
            'structural_entropy': norm_structural_entropy_change
        }
        cst_score = utils._robust_geometric_mean(cst_components, cst_weights, df_index)
        cst_score = cst_score.clip(0, 1)
        vacuum_zone_magnitude_raw = signals_data['vacuum_zone_magnitude_D']
        vacuum_traversal_efficiency_raw = signals_data['vacuum_traversal_efficiency_D']
        active_selling_pressure_raw = signals_data['active_selling_pressure_D']
        capitulation_absorption_raw = signals_data['capitulation_absorption_index_D']
        pressure_rejection_strength_raw = signals_data['pressure_rejection_strength_D']
        support_validation_strength_raw = signals_data['support_validation_strength_D']
        order_book_clearing_rate_raw = signals_data['order_book_clearing_rate_D']
        micro_price_impact_asymmetry_raw = signals_data['micro_price_impact_asymmetry_D']
        norm_vacuum_zone_magnitude = utils.get_adaptive_mtf_normalized_score(vacuum_zone_magnitude_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_vacuum_traversal_efficiency = utils.get_adaptive_mtf_normalized_score(vacuum_traversal_efficiency_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_active_selling_pressure = utils.get_adaptive_mtf_normalized_score(active_selling_pressure_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_capitulation_absorption = utils.get_adaptive_mtf_normalized_score(capitulation_absorption_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_pressure_rejection_strength = utils.get_adaptive_mtf_normalized_score(pressure_rejection_strength_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_support_validation_strength = utils.get_adaptive_mtf_normalized_score(support_validation_strength_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_order_book_clearing_rate = utils.get_adaptive_mtf_normalized_score(order_book_clearing_rate_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_micro_price_impact_asymmetry = utils.get_adaptive_mtf_normalized_score(micro_price_impact_asymmetry_raw.abs(), df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        resistance_absorption_score = (norm_active_selling_pressure * 0.5 + norm_capitulation_absorption * 0.5).clip(0, 1)
        resistance_game_strength = (norm_pressure_rejection_strength * 0.5 + norm_support_validation_strength * 0.5).clip(0, 1)
        pce_components = {
            'vacuum_magnitude': norm_vacuum_zone_magnitude,
            'vacuum_efficiency': norm_vacuum_traversal_efficiency,
            'resistance_absorption': resistance_absorption_score,
            'resistance_game_strength': resistance_game_strength,
            'order_book_clearing_rate': norm_order_book_clearing_rate,
            'micro_price_impact_asymmetry': norm_micro_price_impact_asymmetry
        }
        pce_score = utils._robust_geometric_mean(pce_components, pce_weights, df_index)
        pce_score = pce_score.clip(0, 1)
        deception_index_raw = signals_data['deception_index_D']
        wash_trade_intensity_raw = signals_data['wash_trade_intensity_D']
        retail_panic_surrender_raw = signals_data['retail_panic_surrender_index_D']
        main_force_conviction_raw = signals_data['main_force_conviction_index_D']
        conviction_flow_buy_raw = signals_data['conviction_flow_buy_intensity_D']
        conviction_flow_sell_raw = signals_data['conviction_flow_sell_intensity_D']
        pct_change_raw = signals_data['pct_change_D'] # 获取原始 pct_change_D
        norm_deception_index = utils.get_adaptive_mtf_normalized_bipolar_score(deception_index_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_wash_trade_intensity = utils.get_adaptive_mtf_normalized_score(wash_trade_intensity_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_retail_panic_surrender = utils.get_adaptive_mtf_normalized_score(retail_panic_surrender_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_main_force_conviction = utils.get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_conviction_flow_buy = utils.get_adaptive_mtf_normalized_score(conviction_flow_buy_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_conviction_flow_sell = utils.get_adaptive_mtf_normalized_score(conviction_flow_sell_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        chip_flow_directionality_proxy = (norm_conviction_flow_buy - norm_conviction_flow_sell).clip(-1, 1)
        norm_pct_change = utils.get_adaptive_mtf_normalized_bipolar_score(pct_change_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data) # 归一化 pct_change_D
        # --- Numba优化区域：dgm_score ---
        dgm_score_values = _numba_calculate_historical_potential_dgm_score_core(
            norm_deception_index.values,
            norm_wash_trade_intensity.values,
            norm_retail_panic_surrender.values,
            norm_main_force_conviction.values,
            chip_flow_directionality_proxy.values,
            norm_pct_change.values, # 新增参数
            dgm_weights.get('deception_impact', 0.4),
            dgm_weights.get('wash_trade_penalty', 0.2),
            dgm_weights.get('flow_directionality_boost', 0.1),
            dgm_weights.get('retail_panic_impact', 0.15),
            dgm_weights.get('main_force_conviction_impact', 0.15),
            bearish_deception_and_mf_out_penalty_factor,
            retail_panic_positive_pct_penalty_factor # 新增参数
        )
        dgm_score = pd.Series(dgm_score_values, index=df_index, dtype=np.float32)
        # --- Numba优化区域结束 ---
        context_modulator_components = []
        total_context_weight = 0.0
        for ctx_key, ctx_config in context_modulator_signals.items():
            signal_name = ctx_config.get('signal_name')
            weight = ctx_config.get('weight', 0.0)
            ascending = ctx_config.get('ascending', True)
            if signal_name and weight > 0:
                raw_signal = signals_data[signal_name]
                norm_signal = utils.get_adaptive_mtf_normalized_score(raw_signal, df_index, ascending=ascending, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
                context_modulator_components.append(norm_signal * weight)
                total_context_weight += weight
        if context_modulator_components and total_context_weight > 0:
            combined_context_modulator = sum(context_modulator_components) / total_context_weight
        else:
            combined_context_modulator = pd.Series(0.5, index=df_index)
        dynamic_final_fusion_weights = {
            'mf_aq': final_fusion_weights.get('mf_aq', 0.35) * (1 + combined_context_modulator * context_modulator_sensitivity),
            'cst': final_fusion_weights.get('cst', 0.3) * (1 + combined_context_modulator * context_modulator_sensitivity),
            'pce': final_fusion_weights.get('pce', 0.35) * (1 + combined_context_modulator * context_modulator_sensitivity)
        }
        sum_dynamic_weights = sum(dynamic_final_fusion_weights.values())
        normalized_dynamic_weights = {k: v / sum_dynamic_weights for k, v in dynamic_final_fusion_weights.items()}
        base_potential_score = (
            mf_aq_score * normalized_dynamic_weights.get('mf_aq', 0.35) +
            cst_score * normalized_dynamic_weights.get('cst', 0.3) +
            pce_score * normalized_dynamic_weights.get('pce', 0.35)
        ).clip(0, 1)
        dgm_multiplier = 1 + dgm_score * dgm_modulator_sensitivity
        dgm_multiplier = dgm_multiplier.clip(0.01, 2.0)
        final_potential_score = (base_potential_score * dgm_multiplier).clip(0, 1)
        bull_trap_penalty = self._calculate_bull_trap_context_penalty(df)
        final_score = final_potential_score * bull_trap_penalty
        final_score = final_score.clip(0, 1).fillna(0.0).astype(np.float32)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"  -- [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在诊断“筹码势能”信号...")
            print(f"      [筹码层调试] {method_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---")
            for sig_name in required_signals:
                val = signals_data[sig_name].loc[probe_ts] if probe_ts in signals_data[sig_name].index else np.nan
                print(f"        '{sig_name}': {val:.4f}")
            print(f"      [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 归一化信号值 ---")
            print(f"        norm_chip_health: {norm_chip_health.loc[probe_ts]:.4f}")
            print(f"        norm_covert_accumulation: {norm_covert_accumulation.loc[probe_ts]:.4f}")
            print(f"        norm_suppressive_accumulation: {norm_suppressive_accumulation.loc[probe_ts]:.4f}")
            print(f"        norm_main_force_cost_advantage: {norm_main_force_cost_advantage.loc[probe_ts]:.4f}")
            print(f"        norm_floating_chip_cleansing_efficiency: {norm_floating_chip_cleansing_efficiency.loc[probe_ts]:.4f}")
            print(f"        norm_chip_fault_magnitude: {norm_chip_fault_magnitude.loc[probe_ts]:.4f}")
            print(f"        norm_main_force_execution_alpha: {norm_main_force_execution_alpha.loc[probe_ts]:.4f}")
            print(f"        norm_asymmetric_friction_index: {norm_asymmetric_friction_index.loc[probe_ts]:.4f}")
            print(f"        norm_dominant_peak_solidity: {norm_dominant_peak_solidity.loc[probe_ts]:.4f}")
            print(f"        norm_cost_structure_skewness_slope: {norm_cost_structure_skewness_slope.loc[probe_ts]:.4f}")
            print(f"        norm_peak_separation_ratio_slope: {norm_peak_separation_ratio_slope.loc[probe_ts]:.4f}")
            print(f"        norm_winner_stability: {norm_winner_stability.loc[probe_ts]:.4f}")
            print(f"        norm_loser_pain: {norm_loser_pain.loc[probe_ts]:.4f}")
            print(f"        norm_winner_concentration_slope: {norm_winner_concentration_slope.loc[probe_ts]:.4f}")
            print(f"        norm_loser_concentration_slope: {norm_loser_concentration_slope.loc[probe_ts]:.4f}")
            print(f"        norm_structural_tension: {norm_structural_tension.loc[probe_ts]:.4f}")
            print(f"        norm_structural_entropy_change: {norm_structural_entropy_change.loc[probe_ts]:.4f}")
            print(f"        norm_vacuum_zone_magnitude: {norm_vacuum_zone_magnitude.loc[probe_ts]:.4f}")
            print(f"        norm_vacuum_traversal_efficiency: {norm_vacuum_traversal_efficiency.loc[probe_ts]:.4f}")
            print(f"        norm_active_selling_pressure: {norm_active_selling_pressure.loc[probe_ts]:.4f}")
            print(f"        norm_capitulation_absorption: {norm_capitulation_absorption.loc[probe_ts]:.4f}")
            print(f"        norm_pressure_rejection_strength: {norm_pressure_rejection_strength.loc[probe_ts]:.4f}")
            print(f"        norm_support_validation_strength: {norm_support_validation_strength.loc[probe_ts]:.4f}")
            print(f"        norm_order_book_clearing_rate: {norm_order_book_clearing_rate.loc[probe_ts]:.4f}")
            print(f"        norm_micro_price_impact_asymmetry: {norm_micro_price_impact_asymmetry.loc[probe_ts]:.4f}")
            print(f"        norm_deception_index: {norm_deception_index.loc[probe_ts]:.4f}")
            print(f"        norm_wash_trade_intensity: {norm_wash_trade_intensity.loc[probe_ts]:.4f}")
            print(f"        norm_retail_panic_surrender: {norm_retail_panic_surrender.loc[probe_ts]:.4f}")
            print(f"        norm_main_force_conviction: {norm_main_force_conviction.loc[probe_ts]:.4f}")
            print(f"        norm_conviction_flow_buy: {norm_conviction_flow_buy.loc[probe_ts]:.4f}")
            print(f"        norm_conviction_flow_sell: {norm_conviction_flow_sell.loc[probe_ts]:.4f}")
            print(f"        norm_pct_change: {norm_pct_change.loc[probe_ts]:.4f}") # 新增打印
            print(f"      [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 关键计算节点值 ---")
            print(f"        deception_purity_adjustment: {deception_purity_adjustment.loc[probe_ts]:.4f}")
            print(f"        dynamic_covert_weight: {dynamic_covert_weight.loc[probe_ts]:.4f}")
            print(f"        dynamic_suppressive_weight: {dynamic_suppressive_weight.loc[probe_ts]:.4f}")
            print(f"        mf_aq_score: {mf_aq_score.loc[probe_ts]:.4f}")
            print(f"        structural_elasticity_score: {structural_elasticity_score.loc[probe_ts]:.4f}")
            print(f"        concentration_slope_divergence: {concentration_slope_divergence.loc[probe_ts]:.4f}")
            print(f"        cst_score: {cst_score.loc[probe_ts]:.4f}")
            print(f"        resistance_absorption_score: {resistance_absorption_score.loc[probe_ts]:.4f}")
            print(f"        resistance_game_strength: {resistance_game_strength.loc[probe_ts]:.4f}")
            print(f"        pce_score: {pce_score.loc[probe_ts]:.4f}")
            print(f"        chip_flow_directionality_proxy: {chip_flow_directionality_proxy.loc[probe_ts]:.4f}")
            print(f"        dgm_score: {dgm_score.loc[probe_ts]:.4f}")
            if context_modulator_components:
                print(f"        combined_context_modulator: {combined_context_modulator.loc[probe_ts]:.4f}")
            print(f"        dynamic_final_fusion_weights: {normalized_dynamic_weights}")
            print(f"        base_potential_score: {base_potential_score.loc[probe_ts]:.4f}")
            print(f"        dgm_multiplier: {dgm_multiplier.loc[probe_ts]:.4f}")
            print(f"        final_potential_score: {final_potential_score.loc[probe_ts]:.4f}")
            print(f"        bull_trap_penalty: {bull_trap_penalty.loc[probe_ts]:.4f}")
            print(f"      [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 最终筹码势能得分 (final_score): {final_score.loc[probe_ts]:.4f}")
            print(f"  -- [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: “筹码势能”信号诊断完成。")
        else:
            print(f"  -- [筹码层] “筹码势能”信号诊断完成，最新分值: {final_score.iloc[-1]:.4f}")
        return final_score

    def _diagnose_tactical_exchange(self, df: pd.DataFrame, battlefield_geography: pd.Series) -> pd.Series:
        """
        【V6.1 · Numba优化版】诊断战术换手博弈的质量与意图
        - 核心优化: 将deception_quality_modulator和chip_deception_score_refined的计算逻辑迁移至Numba加速的辅助函数。
        - 核心升级1: 筹码“微观结构”与“订单流执行效率”评估。引入意图执行质量，作为意图维度的一个重要组成部分。
        - 核心升级2: 筹码“多峰结构”与“共振/冲突”分析。引入筹码峰动态，作为质量维度的一个新组成部分。
        - 核心升级3: 筹码“情绪”与“行为模式”识别。引入筹码行为模式强度，作为意图或质量维度的调制器。
        - 核心升级4: 非线性融合的“自学习”与“情境权重矩阵”。升级元调制器，使其能够更精细地调整融合权重。
        """
        df_index = df.index
        required_signals = [
            'peak_control_transfer_D', 'floating_chip_cleansing_efficiency_D',
            'suppressive_accumulation_intensity_D', 'gathering_by_chasing_D', 'gathering_by_support_D',
            'chip_fault_magnitude_D', 'main_force_conviction_index_D',
            'retail_panic_surrender_index_D', 'loser_pain_index_D', 'winner_profit_margin_avg_D',
            'peak_exchange_purity_D',
            'SLOPE_5_winner_concentration_90pct_D', 'SLOPE_5_cost_structure_skewness_D', 'SLOPE_5_peak_separation_ratio_D',
            'winner_loser_momentum_D', 'chip_health_score_D',
            'capitulation_absorption_index_D', 'upward_impulse_purity_D', 'profit_realization_quality_D',
            'chip_fatigue_index_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D',
            'dominant_peak_solidity_D', 'SLOPE_5_dominant_peak_solidity_D',
            'total_loser_rate_D', 'total_winner_rate_D',
            'SLOPE_5_total_loser_rate_D', 'SLOPE_5_total_winner_rate_D',
            'volume_D',
            'winner_stability_index_D',
            'active_buying_support_D', 'active_selling_pressure_D', 'micro_price_impact_asymmetry_D',
            'order_book_clearing_rate_D', 'flow_credibility_index_D',
            'secondary_peak_cost_D', 'dominant_peak_volume_ratio_D',
            'main_force_activity_ratio_D', 'main_force_flow_directionality_D',
            'SLOPE_5_main_force_activity_ratio_D', 'SLOPE_5_main_force_flow_directionality_D'
        ]
        p_conf = self.chip_ultimate_params
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # 优化：预解析 tf_weights 一次
        parsed_tf_data = utils._parse_tf_weights(tf_weights)
        tactical_exchange_params = get_param_value(p_conf.get('tactical_exchange_params'), {})
        intent_weights = get_param_value(tactical_exchange_params.get('intent_weights'), {'control_transfer': 0.3, 'cleansing_efficiency': 0.2, 'accumulation_intent': 0.3, 'intent_execution_quality': 0.2})
        deception_arbitration_power = get_param_value(tactical_exchange_params.get('deception_arbitration_power'), 2.0)
        deception_impact_sensitivity = get_param_value(tactical_exchange_params.get('deception_impact_sensitivity'), 0.5)
        deception_context_modulator_signal_name = get_param_value(tactical_exchange_params.get('deception_context_modulator_signal_name'), 'chip_health_score_D')
        deception_context_sensitivity = get_param_value(tactical_exchange_params.get('deception_context_sensitivity'), 0.3)
        deception_outcome_weights = get_param_value(tactical_exchange_params.get('deception_outcome_weights'), {'effectiveness': 0.6, 'cost': 0.4})
        deception_outcome_effectiveness_threshold = get_param_value(tactical_exchange_params.get('deception_outcome_effectiveness_threshold'), 0.3)
        deception_outcome_cost_threshold = get_param_value(tactical_exchange_params.get('deception_outcome_cost_threshold'), 0.3)
        intent_execution_quality_params = get_param_value(tactical_exchange_params.get('intent_execution_quality_params'), {})
        quality_weights = get_param_value(tactical_exchange_params.get('quality_weights'), {'bullish_absorption': 0.15, 'bullish_purity': 0.15, 'bearish_distribution': 0.15, 'exchange_purity': 0.15, 'structural_optimization': 0.1, 'psychological_pressure_absorption': 0.1, 'exchange_efficiency': 0.05, 'chip_peak_dynamics': 0.15})
        quality_context_signal_name = get_param_value(tactical_exchange_params.get('quality_context_signal_name'), 'winner_loser_momentum_D')
        structural_optimization_slope_period = get_param_value(tactical_exchange_params.get('structural_optimization_slope_period'), 5)
        psychological_pressure_absorption_slope_period = get_param_value(tactical_exchange_params.get('psychological_pressure_absorption_slope_period'), 5)
        chip_peak_dynamics_params = get_param_value(tactical_exchange_params.get('chip_peak_dynamics_params'), {})
        chip_behavioral_pattern_intensity_params = get_param_value(tactical_exchange_params.get('chip_behavioral_pattern_intensity_params'), {})
        chip_behavioral_pattern_intensity_modulator_factor = get_param_value(chip_behavioral_pattern_intensity_params.get('modulator_factor'), 0.2)
        environment_weights = get_param_value(tactical_exchange_params.get('environment_weights'), {'geography': 0.3, 'chip_fatigue': 0.2, 'chip_stability': 0.2, 'dominant_peak_health': 0.15, 'chip_patience_and_stability': 0.15})
        chip_fatigue_impact_factor = get_param_value(tactical_exchange_params.get('chip_fatigue_impact_factor'), 0.5)
        chip_stability_modulator_signal_name = get_param_value(tactical_exchange_params.get('chip_stability_modulator_signal_name'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        chip_stability_sensitivity = get_param_value(tactical_exchange_params.get('chip_stability_sensitivity'), 0.5)
        dominant_peak_health_slope_period = get_param_value(tactical_exchange_params.get('dominant_peak_health_slope_period'), 5)
        rhythm_persistence_slope_period = get_param_value(tactical_exchange_params.get('rhythm_persistence_slope_period'), 5)
        rhythm_persistence_sensitivity = get_param_value(tactical_exchange_params.get('rhythm_persistence_sensitivity'), 0.5)
        final_fusion_weights = get_param_value(tactical_exchange_params.get('final_fusion_weights'), {'intent': 0.35, 'quality': 0.35, 'environment': 0.2, 'rhythm_persistence': 0.1})
        meta_modulator_weights = get_param_value(tactical_exchange_params.get('meta_modulator_weights'), {'chip_health': 0.25, 'volatility_instability': 0.25, 'main_force_conviction': 0.25, 'main_force_activity': 0.15, 'flow_credibility': 0.1})
        meta_modulator_sensitivity = get_param_value(tactical_exchange_params.get('meta_modulator_sensitivity'), 0.5)
        if deception_context_modulator_signal_name not in required_signals:
            required_signals.append(deception_context_modulator_signal_name)
        if quality_context_signal_name not in required_signals:
            required_signals.append(quality_context_signal_name)
        if chip_stability_modulator_signal_name not in required_signals:
            required_signals.append(chip_stability_modulator_signal_name)
        slope_wc_signal = f'SLOPE_{structural_optimization_slope_period}_winner_concentration_90pct_D'
        if slope_wc_signal not in required_signals: required_signals.append(slope_wc_signal)
        slope_css_signal = f'SLOPE_{structural_optimization_slope_period}_cost_structure_skewness_D'
        if slope_css_signal not in required_signals: required_signals.append(slope_css_signal)
        slope_psr_signal = f'SLOPE_{structural_optimization_slope_period}_peak_separation_ratio_D'
        if slope_psr_signal not in required_signals: required_signals.append(slope_psr_signal)
        slope_loser_rate_signal = f'SLOPE_{psychological_pressure_absorption_slope_period}_total_loser_rate_D'
        if slope_loser_rate_signal not in required_signals: required_signals.append(slope_loser_rate_signal)
        slope_winner_rate_signal = f'SLOPE_{psychological_pressure_absorption_slope_period}_total_winner_rate_D'
        if slope_winner_rate_signal not in required_signals: required_signals.append(slope_winner_rate_signal)
        slope_dps_signal = f'SLOPE_{dominant_peak_health_slope_period}_dominant_peak_solidity_D'
        if slope_dps_signal not in required_signals: required_signals.append(slope_dps_signal)
        if not self._validate_required_signals(df, required_signals, "_diagnose_tactical_exchange"):
            return pd.Series(0.0, index=df.index)
        signals_data = self._get_all_required_signals(df, required_signals, "_diagnose_tactical_exchange")
        is_debug_enabled = self.should_probe
        probe_ts = None
        if is_debug_enabled and self.probe_dates_set:
            for date in reversed(df_index):
                if date.date() in self.probe_dates_set:
                    probe_ts = date
                    break
        debug_info_tuple = False
        control_transfer_raw = signals_data['peak_control_transfer_D']
        cleansing_efficiency_raw = signals_data['floating_chip_cleansing_efficiency_D']
        suppressive_accum_raw = signals_data['suppressive_accumulation_intensity_D']
        gathering_chasing_raw = signals_data['gathering_by_chasing_D']
        gathering_support_raw = signals_data['gathering_by_support_D']
        chip_fault_raw = signals_data['chip_fault_magnitude_D']
        mf_conviction_raw = signals_data['main_force_conviction_index_D']
        retail_panic_surrender_raw = signals_data['retail_panic_surrender_index_D']
        loser_pain_raw = signals_data['loser_pain_index_D']
        winner_profit_margin_avg_raw = signals_data['winner_profit_margin_avg_D']
        peak_exchange_purity_raw = signals_data['peak_exchange_purity_D']
        slope_wc_raw = signals_data[f'SLOPE_{structural_optimization_slope_period}_winner_concentration_90pct_D']
        slope_css_raw = signals_data[f'SLOPE_{structural_optimization_slope_period}_cost_structure_skewness_D']
        slope_psr_raw = signals_data[f'SLOPE_{structural_optimization_slope_period}_peak_separation_ratio_D']
        winner_loser_momentum_raw = signals_data['winner_loser_momentum_D']
        chip_health_raw = signals_data['chip_health_score_D']
        capitulation_absorption_raw = signals_data['capitulation_absorption_index_D']
        upward_impulse_purity_raw = signals_data['upward_impulse_purity_D']
        profit_realization_quality_raw = signals_data['profit_realization_quality_D']
        chip_fatigue_raw = signals_data['chip_fatigue_index_D']
        volatility_instability_raw = signals_data['VOLATILITY_INSTABILITY_INDEX_21d_D']
        dominant_peak_solidity_raw = signals_data['dominant_peak_solidity_D']
        slope_dps_raw = signals_data[f'SLOPE_{dominant_peak_health_slope_period}_dominant_peak_solidity_D']
        total_loser_rate_raw = signals_data['total_loser_rate_D']
        total_winner_rate_raw = signals_data['total_winner_rate_D']
        slope_loser_rate_raw = signals_data[f'SLOPE_{psychological_pressure_absorption_slope_period}_total_loser_rate_D']
        slope_winner_rate_raw = signals_data[f'SLOPE_{psychological_pressure_absorption_slope_period}_total_winner_rate_D']
        volume_raw = signals_data['volume_D']
        winner_stability_index_raw = signals_data['winner_stability_index_D']
        active_buying_support_raw = signals_data['active_buying_support_D']
        active_selling_pressure_raw = signals_data['active_selling_pressure_D']
        micro_price_impact_asymmetry_raw = signals_data['micro_price_impact_asymmetry_D']
        order_book_clearing_rate_raw = signals_data['order_book_clearing_rate_D']
        flow_credibility_index_raw = signals_data['flow_credibility_index_D']
        secondary_peak_cost_raw = signals_data['secondary_peak_cost_D']
        dominant_peak_volume_ratio_raw = signals_data['dominant_peak_volume_ratio_D']
        main_force_activity_ratio_raw = signals_data['main_force_activity_ratio_D']
        main_force_flow_directionality_raw = signals_data['main_force_flow_directionality_D']
        slope_5_main_force_activity_ratio_raw = signals_data['SLOPE_5_main_force_activity_ratio_D']
        slope_5_main_force_flow_directionality_raw = signals_data['SLOPE_5_main_force_flow_directionality_D']
        deception_context_modulator_raw = signals_data[deception_context_modulator_signal_name]
        quality_context_raw = signals_data[quality_context_signal_name]
        chip_stability_modulator_raw = signals_data[chip_stability_modulator_signal_name]
        # 优化：传递预解析的 tf_weights 数据
        norm_control_transfer = utils.get_adaptive_mtf_normalized_bipolar_score(control_transfer_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_cleansing_efficiency = utils.get_adaptive_mtf_normalized_score(cleansing_efficiency_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_suppressive_accum = utils.get_adaptive_mtf_normalized_score(suppressive_accum_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_gathering_chasing = utils.get_adaptive_mtf_normalized_score(gathering_chasing_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_gathering_support = utils.get_adaptive_mtf_normalized_score(gathering_support_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        accumulation_intent_score = (norm_suppressive_accum * 0.4 + norm_gathering_chasing * 0.3 + norm_gathering_support * 0.3)
        # 优化：传递预解析的 tf_weights 数据
        norm_active_buying_support = utils.get_adaptive_mtf_normalized_score(active_buying_support_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_active_selling_pressure = utils.get_adaptive_mtf_normalized_score(active_selling_pressure_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_micro_price_impact_asymmetry = utils.get_adaptive_mtf_normalized_bipolar_score(micro_price_impact_asymmetry_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_order_book_clearing_rate = utils.get_adaptive_mtf_normalized_score(order_book_clearing_rate_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_flow_credibility_index = utils.get_adaptive_mtf_normalized_score(flow_credibility_index_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        intent_execution_quality_score = (
            norm_active_buying_support * intent_execution_quality_params.get('buying_support_weight', 0.3) +
            norm_active_selling_pressure * intent_execution_quality_params.get('selling_pressure_weight', 0.2) +
            (1 - norm_micro_price_impact_asymmetry.abs()) * intent_execution_quality_params.get('price_impact_weight', 0.2) +
            norm_order_book_clearing_rate * intent_execution_quality_params.get('clearing_rate_weight', 0.15) +
            norm_flow_credibility_index * intent_execution_quality_params.get('flow_credibility_weight', 0.15)
        ).clip(0, 1)
        base_intent_score = (
            norm_control_transfer * intent_weights.get('control_transfer', 0.3) +
            norm_cleansing_efficiency * intent_weights.get('cleansing_efficiency', 0.2) +
            accumulation_intent_score * intent_weights.get('accumulation_intent', 0.3) +
            intent_execution_quality_score * intent_weights.get('intent_execution_quality', 0.2)
        ).clip(-1, 1)
        chip_deception_direction = np.sign(utils.get_adaptive_mtf_normalized_bipolar_score(mf_conviction_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data))
        # 优化：传递预解析的 tf_weights 数据
        norm_chip_fault = utils.get_adaptive_mtf_normalized_score(chip_fault_raw.abs(), df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_retail_panic_surrender = utils.get_adaptive_mtf_normalized_score(retail_panic_surrender_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_loser_pain = utils.get_adaptive_mtf_normalized_score(loser_pain_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_winner_profit_margin_avg = utils.get_adaptive_mtf_normalized_score(winner_profit_margin_avg_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        # --- Numba优化区域：deception_quality_modulator和chip_deception_score_refined ---
        chip_deception_score_refined_values, deception_quality_modulator_values = _numba_calculate_tactical_exchange_deception_core(
            chip_deception_direction.values,
            norm_chip_fault.values,
            norm_retail_panic_surrender.values,
            norm_loser_pain.values,
            norm_winner_profit_margin_avg.values,
            norm_suppressive_accum.values,
            (1 - utils.get_adaptive_mtf_normalized_score(profit_realization_quality_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)).values,
            deception_outcome_effectiveness_threshold,
            deception_outcome_cost_threshold,
            deception_outcome_weights.get('effectiveness', 0.6),
            deception_outcome_weights.get('cost', 0.4)
        )
        chip_deception_score_refined = pd.Series(chip_deception_score_refined_values, index=df_index, dtype=np.float32)
        deception_quality_modulator = pd.Series(deception_quality_modulator_values, index=df_index, dtype=np.float32)
        # --- Numba优化区域结束 ---
        deception_context_modulator_raw = signals_data[deception_context_modulator_signal_name]
        # 优化：传递预解析的 tf_weights 数据
        norm_deception_context = utils.get_adaptive_mtf_normalized_score(deception_context_modulator_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        dynamic_deception_impact_sensitivity = deception_impact_sensitivity * (1 - norm_deception_context * deception_context_sensitivity)
        dynamic_deception_impact_sensitivity = dynamic_deception_impact_sensitivity.clip(0.1, 1.0)
        arbitration_weight = (norm_chip_fault * dynamic_deception_impact_sensitivity).pow(deception_arbitration_power).clip(0, 1)
        intent_score = base_intent_score * (1 - arbitration_weight) + chip_deception_score_refined * arbitration_weight
        intent_score = intent_score.clip(-1, 1)
        # 优化：传递预解析的 tf_weights 数据
        norm_main_force_activity = utils.get_adaptive_mtf_normalized_score(main_force_activity_ratio_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_main_force_flow_directionality = utils.get_adaptive_mtf_normalized_bipolar_score(main_force_flow_directionality_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        chip_behavioral_pattern_intensity_score = (norm_main_force_activity * 0.6 + norm_main_force_flow_directionality.abs() * 0.4).clip(0, 1)
        intent_score = intent_score * (1 + chip_behavioral_pattern_intensity_score * chip_behavioral_pattern_intensity_modulator_factor)
        intent_score = intent_score.clip(-1, 1)
        chip_momentum_raw = signals_data[quality_context_signal_name]
        # 优化：传递预解析的 tf_weights 数据
        norm_chip_momentum_context = utils.get_adaptive_mtf_normalized_bipolar_score(chip_momentum_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        # 优化：传递预解析的 tf_weights 数据
        norm_absorption = utils.get_adaptive_mtf_normalized_score(capitulation_absorption_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_impulse_purity = utils.get_adaptive_mtf_normalized_score(upward_impulse_purity_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        dynamic_bullish_quality_weight = (norm_chip_momentum_context.add(1)/2) * 0.5 + 0.5
        bullish_quality_score = (
            norm_absorption * quality_weights.get('bullish_absorption', 0.15) +
            norm_impulse_purity * quality_weights.get('bullish_purity', 0.15)
        ) * dynamic_bullish_quality_weight
        # 优化：传递预解析的 tf_weights 数据
        norm_profit_realization = utils.get_adaptive_mtf_normalized_score(profit_realization_quality_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        dynamic_bearish_quality_weight = (1 - norm_chip_momentum_context.add(1)/2) * 0.5 + 0.5
        bearish_quality_score = norm_profit_realization * quality_weights.get('bearish_distribution', 0.15) * dynamic_bearish_quality_weight
        # 优化：传递预解析的 tf_weights 数据
        exchange_purity_score = utils.get_adaptive_mtf_normalized_score(peak_exchange_purity_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        # 优化：传递预解析的 tf_weights 数据
        norm_slope_wc = utils.get_adaptive_mtf_normalized_score(slope_wc_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_slope_css = utils.get_adaptive_mtf_normalized_score(slope_css_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_slope_psr = utils.get_adaptive_mtf_normalized_score(slope_psr_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        structural_optimization_score = (norm_slope_wc + norm_slope_css + norm_slope_psr) / 3
        structural_optimization_score = structural_optimization_score.clip(0, 1)
        # 优化：传递预解析的 tf_weights 数据
        norm_loser_absorption_quality = utils.get_adaptive_mtf_normalized_bipolar_score(slope_loser_rate_raw, df_index, tf_weights, sensitivity=1.0, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_winner_resilience_quality = utils.get_adaptive_mtf_normalized_bipolar_score(slope_winner_rate_raw, df_index, tf_weights, sensitivity=1.0, debug_info=False, _parsed_tf_data=parsed_tf_data)
        psychological_pressure_absorption_score = (norm_loser_absorption_quality.clip(lower=0) + norm_winner_resilience_quality.clip(upper=0).abs()) / 2
        psychological_pressure_absorption_score = psychological_pressure_absorption_score.clip(0, 1)
        # 优化：传递预解析的 tf_weights 数据
        norm_volume = utils.get_adaptive_mtf_normalized_score(volume_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        exchange_efficiency_score = structural_optimization_score / (norm_volume.replace(0, 1e-6))
        exchange_efficiency_score = exchange_efficiency_score.clip(0, 1)
        # 优化：传递预解析的 tf_weights 数据
        norm_secondary_peak_cost = utils.get_adaptive_mtf_normalized_score(secondary_peak_cost_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_dominant_peak_volume_ratio = utils.get_adaptive_mtf_normalized_score(dominant_peak_volume_ratio_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        chip_peak_dynamics_score = (norm_secondary_peak_cost * chip_peak_dynamics_params.get('secondary_cost_weight', 0.5) +
                                    norm_dominant_peak_volume_ratio * chip_peak_dynamics_params.get('secondary_volume_weight', 0.5)).clip(0, 1)
        quality_score = (
            bullish_quality_score * (1 - dynamic_bearish_quality_weight) +
            bearish_quality_score * (1 - dynamic_bullish_quality_weight) +
            exchange_purity_score * quality_weights.get('exchange_purity', 0.15) +
            structural_optimization_score * quality_weights.get('structural_optimization', 0.1) +
            psychological_pressure_absorption_score * quality_weights.get('psychological_pressure_absorption', 0.1) +
            exchange_efficiency_score * quality_weights.get('exchange_efficiency', 0.05) +
            chip_peak_dynamics_score * quality_weights.get('chip_peak_dynamics', 0.15)
        ).clip(-1, 1)
        quality_score = quality_score * (1 + chip_behavioral_pattern_intensity_score * chip_behavioral_pattern_intensity_modulator_factor)
        quality_score = quality_score.clip(-1, 1)
        # 优化：传递预解析的 tf_weights 数据
        norm_chip_fatigue = utils.get_adaptive_mtf_normalized_score(chip_fatigue_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        chip_stability_modulator_raw = signals_data[chip_stability_modulator_signal_name]
        norm_chip_stability_modulator = utils.get_adaptive_mtf_normalized_score(chip_stability_modulator_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_chip_health = utils.get_adaptive_mtf_normalized_bipolar_score(chip_health_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        dynamic_chip_fatigue_impact = norm_chip_fatigue * chip_fatigue_impact_factor
        dynamic_chip_stability_bonus = norm_chip_stability_modulator * chip_stability_sensitivity
        dominant_peak_solidity_raw = signals_data['dominant_peak_solidity_D']
        slope_dps_raw = signals_data[f'SLOPE_{dominant_peak_health_slope_period}_dominant_peak_solidity_D']
        # 优化：传递预解析的 tf_weights 数据
        norm_dps = utils.get_adaptive_mtf_normalized_score(dominant_peak_solidity_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_slope_dps = utils.get_adaptive_mtf_normalized_bipolar_score(slope_dps_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        dominant_peak_health_score = (norm_dps * 0.7 + norm_slope_dps * 0.3).clip(0, 1)
        winner_stability_index_raw = signals_data['winner_stability_index_D']
        # 优化：传递预解析的 tf_weights 数据
        norm_winner_stability = utils.get_adaptive_mtf_normalized_score(winner_stability_index_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        chip_patience_ratio = norm_gathering_support / (norm_gathering_support + norm_gathering_chasing + 1e-6)
        # 优化：传递预解析的 tf_weights 数据
        chip_patience_score = utils.get_adaptive_mtf_normalized_score(chip_patience_ratio, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        chip_patience_and_stability_score = (norm_winner_stability * 0.5 + chip_patience_score * 0.5).clip(0, 1)
        context_score = (
            battlefield_geography * environment_weights.get('geography', 0.3) -
            dynamic_chip_fatigue_impact * environment_weights.get('chip_fatigue', 0.2) +
            dynamic_chip_stability_bonus * environment_weights.get('chip_stability', 0.2) +
            dominant_peak_health_score * environment_weights.get('dominant_peak_health', 0.15) +
            chip_patience_and_stability_score * environment_weights.get('chip_patience_and_stability', 0.15)
        ).clip(-1, 1)
        # --- 计算节奏和持续性 (Rhythm and Persistence) ---
        # 意图得分的斜率
        rhythm_intent_slope = intent_score.diff(rhythm_persistence_slope_period).fillna(0)
        # 质量得分的斜率
        rhythm_quality_slope = quality_score.diff(rhythm_persistence_slope_period).fillna(0)
        # 优化：传递预解析的 tf_weights 数据
        norm_rhythm_intent_slope = utils.get_adaptive_mtf_normalized_bipolar_score(rhythm_intent_slope, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_rhythm_quality_slope = utils.get_adaptive_mtf_normalized_bipolar_score(rhythm_quality_slope, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        rhythm_and_persistence_score = (norm_rhythm_intent_slope + norm_rhythm_quality_slope) / 2
        rhythm_and_persistence_score = (rhythm_and_persistence_score * rhythm_persistence_sensitivity).clip(-1, 1)
        volatility_instability_raw = signals_data['VOLATILITY_INSTABILITY_INDEX_21d_D']
        # 优化：传递预解析的 tf_weights 数据
        norm_volatility_instability = utils.get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        main_force_conviction_abs_raw = signals_data['main_force_conviction_index_D'].abs()
        norm_main_force_conviction = utils.get_adaptive_mtf_normalized_score(main_force_conviction_abs_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        main_force_activity_abs_raw = signals_data['main_force_activity_ratio_D'].abs()
        norm_main_force_activity_meta = utils.get_adaptive_mtf_normalized_score(main_force_activity_abs_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        flow_credibility_index_meta_raw = signals_data['flow_credibility_index_D']
        norm_flow_credibility_index_meta = utils.get_adaptive_mtf_normalized_score(flow_credibility_index_meta_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        market_context_meta_modulator = (
            norm_chip_health * meta_modulator_weights.get('chip_health', 0.25) +
            norm_volatility_instability * meta_modulator_weights.get('volatility_instability', 0.25) +
            norm_main_force_conviction * meta_modulator_weights.get('main_force_conviction', 0.25) +
            norm_main_force_activity_meta * meta_modulator_weights.get('main_force_activity', 0.15) +
            norm_flow_credibility_index_meta * meta_modulator_weights.get('flow_credibility', 0.1)
        ).clip(0, 1)
        dynamic_final_fusion_weights = {
            'intent': final_fusion_weights.get('intent', 0.35) * (1 + market_context_meta_modulator * meta_modulator_sensitivity),
            'quality': final_fusion_weights.get('quality', 0.35) * (1 + market_context_meta_modulator * meta_modulator_sensitivity),
            'environment': final_fusion_weights.get('environment', 0.2) * (1 + market_context_meta_modulator * meta_modulator_sensitivity),
            'rhythm_persistence': final_fusion_weights.get('rhythm_persistence', 0.1) * (1 + market_context_meta_modulator * meta_modulator_sensitivity)
        }
        sum_dynamic_weights = sum(dynamic_final_fusion_weights.values())
        normalized_dynamic_weights = {k: v / sum_dynamic_weights for k, v in dynamic_final_fusion_weights.items()}
        final_score = (
            intent_score * normalized_dynamic_weights.get('intent', 0.35) +
            quality_score * normalized_dynamic_weights.get('quality', 0.35) +
            context_score * normalized_dynamic_weights.get('environment', 0.2) +
            rhythm_and_persistence_score * normalized_dynamic_weights.get('rhythm_persistence', 0.1)
        ).clip(-1, 1)
        return final_score.clip(-1, 1).fillna(0.0).astype(np.float32)

    def _diagnose_strategic_tactical_harmony(self, df: pd.DataFrame, strategic_posture: pd.Series, tactical_exchange: pd.Series, holder_sentiment_scores: pd.Series) -> pd.Series:
        """
        【V3.1 · Numba优化版】诊断战略与战术的和谐度
        - 核心优化: 将conflict_penalty_factor_adjusted的计算逻辑迁移至Numba加速的辅助函数。
        - 核心升级1: 战术执行微观深化。引入高频微观筹码行为（如日内筹码流平衡、订单簿压力等）作为“当日战术执行”的更精细化输入，提升战术评估的颗粒度和准确性。
        - 核心升级2: 动态权重调制精细化。战略与战术的融合权重不再固定，而是根据筹码波动不稳定性、筹码疲劳指数等筹码层情境因子动态调整，以适应不同市场阶段的侧重点。
        - 核心升级3: 和谐因子情境纯筹码化。和谐度因子的情境调制器严格限定为筹码层信号（如持仓信念韧性、价筹张力），确保信号的纯粹性。
        - 核心升级4: 冲突情境诡道深化。明确识别战略与战术方向完全背离的“冲突区”，并引入诡道因子（如欺骗指数、对倒强度）进行调制，对伴随欺骗的冲突施加更严厉惩罚。
        - 核心升级5: 趋势一致性品质校准。当战略与战术在同一方向上高度协同并具备足够强度时，引入筹码品质因子（如筹码健康度、主力信念指数）进行校准，确保奖励的是高质量、可持续的趋势。
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        method_name = "_diagnose_strategic_tactical_harmony"
        df_index = df.index
        p_conf = self.chip_ultimate_params
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        parsed_tf_data = utils._parse_tf_weights(tf_weights)
        harmony_params = get_param_value(p_conf.get('strategic_tactical_harmony_params'), {})
        strategic_weight_base = get_param_value(harmony_params.get('strategic_weight_base'), 0.6)
        tactical_weight_base = get_param_value(harmony_params.get('tactical_weight_base'), 0.4)
        dynamic_weight_modulator_signal_name = get_param_value(harmony_params.get('dynamic_weight_modulator_signal'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        dynamic_weight_sensitivity = get_param_value(harmony_params.get('dynamic_weight_sensitivity'), 0.3)
        harmony_non_linear_exponent = get_param_value(harmony_params.get('harmony_non_linear_exponent'), 2.5)
        harmony_context_modulator_signal_name = get_param_value(harmony_params.get('harmony_context_modulator_signal'), 'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT')
        harmony_context_sensitivity = get_param_value(harmony_params.get('harmony_context_sensitivity'), 0.4)
        conflict_threshold = get_param_value(harmony_params.get('conflict_threshold'), 0.6)
        conflict_penalty_factor = get_param_value(harmony_params.get('conflict_penalty_factor'), 0.7)
        deception_modulator_signal_name = get_param_value(harmony_params.get('deception_modulator_signal'), 'deception_index_D')
        deception_penalty_sensitivity = get_param_value(harmony_params.get('deception_penalty_sensitivity'), 0.5)
        wash_trade_mitigation_sensitivity = get_param_value(harmony_params.get('wash_trade_mitigation_sensitivity'), 0.3)
        trend_alignment_threshold = get_param_value(harmony_params.get('trend_alignment_threshold'), 0.75)
        trend_bonus_factor = get_param_value(harmony_params.get('trend_bonus_factor'), 0.15)
        quality_calibrator_signal_name = get_param_value(harmony_params.get('quality_calibrator_signal'), 'chip_health_score_D')
        quality_calibration_sensitivity = get_param_value(harmony_params.get('quality_calibration_sensitivity'), 0.5)
        high_harmony_threshold = get_param_value(harmony_params.get('high_harmony_threshold'), 0.8)
        required_signals = [
            dynamic_weight_modulator_signal_name,
            deception_modulator_signal_name,
            quality_calibrator_signal_name
        ]
        if harmony_context_modulator_signal_name != 'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT':
            required_signals.append(harmony_context_modulator_signal_name)
        if not self._validate_required_signals(df, required_signals, "_diagnose_strategic_tactical_harmony"):
            return pd.Series(0.0, index=df.index)
        signals_data = self._get_all_required_signals(df, required_signals, "_diagnose_strategic_tactical_harmony")
        is_debug_enabled = self.should_probe
        probe_ts = None
        if is_debug_enabled and self.probe_dates_set:
            for date in reversed(df_index):
                if date.date() in self.probe_dates_set:
                    probe_ts = date
                    break
        debug_info_tuple = False
        dynamic_weight_modulator_raw = signals_data[dynamic_weight_modulator_signal_name]
        deception_raw = signals_data[deception_modulator_signal_name]
        quality_calibrator_raw = signals_data[quality_calibrator_signal_name]
        if harmony_context_modulator_signal_name == 'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT':
            harmony_context_modulator_raw = holder_sentiment_scores
        else:
            harmony_context_modulator_raw = signals_data[harmony_context_modulator_signal_name]
        norm_dynamic_weight_modulator = utils.get_adaptive_mtf_normalized_score(dynamic_weight_modulator_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        dynamic_strategic_weight = strategic_weight_base * (1 - norm_dynamic_weight_modulator * dynamic_weight_sensitivity)
        dynamic_tactical_weight = tactical_weight_base * (1 + norm_dynamic_weight_modulator * dynamic_weight_sensitivity)
        sum_dynamic_weights = dynamic_strategic_weight + dynamic_tactical_weight
        dynamic_strategic_weight = dynamic_strategic_weight / sum_dynamic_weights
        dynamic_tactical_weight = dynamic_tactical_weight / sum_dynamic_weights
        base_intent_score = strategic_posture * dynamic_strategic_weight + tactical_exchange * dynamic_tactical_weight
        norm_harmony_context = utils.get_adaptive_mtf_normalized_bipolar_score(harmony_context_modulator_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        raw_difference = (strategic_posture - tactical_exchange).abs() / 2
        non_linear_diff = raw_difference.pow(harmony_non_linear_exponent)
        harmony_factor = (1 - non_linear_diff).clip(lower=0)
        context_modulation_effect = (norm_harmony_context * harmony_context_sensitivity).clip(-0.5, 0.5)
        harmony_factor = harmony_factor * (1 + context_modulation_effect)
        harmony_factor = harmony_factor.clip(0, 1)
        # --- Numba优化区域：conflict_penalty_factor_adjusted ---
        norm_deception = utils.get_adaptive_mtf_normalized_bipolar_score(deception_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        conflict_penalty_factor_adjusted_values = _numba_calculate_harmony_conflict_penalty_core(
            strategic_posture.values,
            tactical_exchange.values,
            norm_deception.values,
            conflict_threshold,
            conflict_penalty_factor,
            deception_penalty_sensitivity
        )
        conflict_penalty_factor_adjusted = pd.Series(conflict_penalty_factor_adjusted_values, index=df_index, dtype=np.float32)
        # --- Numba优化区域结束 ---
        alignment_bonus = pd.Series(0.0, index=df_index)
        bullish_alignment_mask = (strategic_posture > trend_alignment_threshold) & \
                                 (tactical_exchange > trend_alignment_threshold) & \
                                 (harmony_factor > high_harmony_threshold)
        bearish_alignment_mask = (strategic_posture < -trend_alignment_threshold) & \
                                 (tactical_exchange < -trend_alignment_threshold) & \
                                 (harmony_factor > high_harmony_threshold)
        norm_quality_calibrator = utils.get_adaptive_mtf_normalized_score(quality_calibrator_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        calibrated_bonus_factor = trend_bonus_factor * (1 + norm_quality_calibrator * quality_calibration_sensitivity)
        calibrated_bonus_factor = calibrated_bonus_factor.clip(0, trend_bonus_factor * 2)
        alignment_bonus.loc[bullish_alignment_mask] = calibrated_bonus_factor.loc[bullish_alignment_mask]
        alignment_bonus.loc[bearish_alignment_mask] = -calibrated_bonus_factor.loc[bearish_alignment_mask]
        final_score = base_intent_score * harmony_factor * conflict_penalty_factor_adjusted + alignment_bonus
        final_score = final_score.clip(-1, 1).fillna(0.0).astype(np.float32)
        return final_score

    def _diagnose_harmony_inflection(self, df: pd.DataFrame, harmony_score: pd.Series) -> pd.Series:
        """
        【V3.5 · Numba优化版】诊断战略与战术和谐度的动态转折点，旨在构建一个诡道拐点判别与确认系统。
        - 核心优化: 将inflection_strength和deception_modulator的计算逻辑迁移至Numba加速的辅助函数。
        - 核心升级1: 动态阈值自适应：和谐度所处区间（低位、中位、高位）的判断阈值不再固定，而是根据市场波动性或筹码健康度动态调整，提高对拐点“位置”判断的适应性。
        - 核心升级2: 非对称拐点动能融合：采用更复杂的非线性函数融合速度和加速度，并允许正向和负向拐点使用不同的融合参数，以更精细地量化拐点背后的真实动能，并反映市场情绪的非对称性。
        - 核心升级3: 诡道博弈过滤与惩罚：引入欺骗指数、对倒强度等诡道因子作为调制器，识别并惩罚伴随诱多欺骗的正向拐点，或适度削弱伴随诱空洗盘的负向拐点，提高信号真实性。
        - 核心升级4: 拐点延续性确认奖励：引入短期延续性检查机制，如果拐点方向在后续几天得到确认，则给予额外奖励，增强信号可靠性和鲁棒性。
        - 核心升级5: 增强情境调制器：除了筹码健康度和波动性，再引入主力信念指数作为情境调制器，更全面评估拐点信号在不同市场参与者意图下的可靠性。
        - 核心修复: 确保 `deception_modulator` 在 `norm_deception` 为负时，能够正确增强正向拐点信号，并增加增强敏感度。
        - **新增业务逻辑：引入“牛市陷阱情境惩罚”，在近期大幅下跌后伴随正向欺骗时，大幅降低和谐拐点的得分。**
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        method_name = "_diagnose_harmony_inflection"
        df_index = df.index
        p_conf = self.chip_ultimate_params
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # 优化：预解析 tf_weights 一次
        parsed_tf_data = utils._parse_tf_weights(tf_weights)
        inflection_params = get_param_value(p_conf.get('harmony_inflection_params'), {})
        velocity_period = get_param_value(inflection_params.get('velocity_period'), 1)
        acceleration_period = get_param_value(inflection_params.get('acceleration_period'), 1)
        positive_strength_tanh_factor = get_param_value(inflection_params.get('positive_strength_tanh_factor'), 1.5)
        negative_strength_tanh_factor = get_param_value(inflection_params.get('negative_strength_tanh_factor'), 1.5)
        base_low_harmony_threshold = get_param_value(inflection_params.get('base_low_harmony_threshold'), 0.2)
        base_high_harmony_threshold = get_param_value(inflection_params.get('base_high_harmony_threshold'), 0.8)
        threshold_modulator_signal_name = get_param_value(inflection_params.get('threshold_modulator_signal'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        threshold_modulator_sensitivity = get_param_value(inflection_params.get('threshold_modulator_sensitivity'), 0.2)
        low_harmony_boost_factor = get_param_value(inflection_params.get('low_harmony_boost_factor'), 1.5)
        high_harmony_boost_factor = get_param_value(inflection_params.get('high_harmony_boost_factor'), 1.5)
        mid_harmony_neutral_factor = get_param_value(inflection_params.get('mid_harmony_neutral_factor'), 1.0)
        deception_signal_name = get_param_value(inflection_params.get('deception_signal'), 'deception_index_D')
        wash_trade_signal_name = get_param_value(inflection_params.get('wash_trade_signal'), 'wash_trade_intensity_D')
        deception_penalty_sensitivity = get_param_value(inflection_params.get('deception_penalty_sensitivity'), 0.7)
        wash_trade_mitigation_sensitivity = get_param_value(inflection_params.get('wash_trade_mitigation_sensitivity'), 0.3)
        persistence_period = get_param_value(inflection_params.get('persistence_period'), 2)
        persistence_bonus_factor = get_param_value(inflection_params.get('persistence_bonus_factor'), 0.1)
        context_modulator_signal_1_name = get_param_value(inflection_params.get('context_modulator_signal_1'), 'chip_health_score_D')
        context_modulator_signal_2_name = get_param_value(inflection_params.get('context_modulator_signal_2'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        context_modulator_signal_3_name = get_param_value(inflection_params.get('context_modulator_signal_3'), 'main_force_conviction_index_D')
        context_modulator_sensitivity_health = get_param_value(inflection_params.get('context_modulator_sensitivity_health'), 0.5)
        context_modulator_sensitivity_volatility = get_param_value(inflection_params.get('context_modulator_sensitivity_volatility'), 0.3)
        context_modulator_sensitivity_conviction = get_param_value(inflection_params.get('context_modulator_sensitivity_conviction'), 0.4)
        deception_boost_factor_negative = get_param_value(inflection_params.get('deception_boost_factor_negative'), 0.5)
        required_signals = [
            threshold_modulator_signal_name,
            deception_signal_name,
            wash_trade_signal_name,
            context_modulator_signal_1_name,
            context_modulator_signal_2_name,
            context_modulator_signal_3_name,
            'pct_change_D'
        ]
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index)
        signals_data = self._get_all_required_signals(df, required_signals, method_name)
        is_debug_enabled = self.should_probe
        probe_ts = None
        if is_debug_enabled and self.probe_dates_set:
            for date in reversed(df_index):
                if date.date() in self.probe_dates_set:
                    probe_ts = date
                    break
        debug_info_tuple = (is_debug_enabled, probe_ts, method_name)
        threshold_modulator_raw = signals_data[threshold_modulator_signal_name]
        deception_raw = signals_data[deception_signal_name]
        wash_trade_raw = signals_data[wash_trade_signal_name]
        chip_health_raw = signals_data[context_modulator_signal_1_name]
        volatility_instability_raw = signals_data[context_modulator_signal_2_name]
        main_force_conviction_raw = signals_data[context_modulator_signal_3_name]
        # --- 修正：先计算 harmony_velocity 和 harmony_acceleration ---
        harmony_velocity = harmony_score.diff(velocity_period).fillna(0)
        harmony_acceleration = harmony_velocity.diff(acceleration_period).fillna(0)
        # 优化：传递预解析的 tf_weights 数据
        norm_velocity = utils.get_adaptive_mtf_normalized_bipolar_score(harmony_velocity, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_acceleration = utils.get_adaptive_mtf_normalized_bipolar_score(harmony_acceleration, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        # --- Numba优化区域：inflection_strength ---
        positive_inflection_strength_values, negative_inflection_strength_values = _numba_calculate_inflection_strength_core(
            norm_velocity.values,
            norm_acceleration.values,
            positive_strength_tanh_factor,
            negative_strength_tanh_factor
        )
        positive_inflection_strength = pd.Series(positive_inflection_strength_values, index=df_index, dtype=np.float32)
        negative_inflection_strength = pd.Series(negative_inflection_strength_values, index=df_index, dtype=np.float32)
        inflection_strength = positive_inflection_strength - negative_inflection_strength
        # --- Numba优化区域结束 ---
        # 优化：传递预解析的 tf_weights 数据
        norm_threshold_modulator = utils.get_adaptive_mtf_normalized_score(threshold_modulator_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        dynamic_low_harmony_threshold = base_low_harmony_threshold * (1 - norm_threshold_modulator * threshold_modulator_sensitivity)
        dynamic_high_harmony_threshold = base_high_harmony_threshold * (1 + norm_threshold_modulator * threshold_modulator_sensitivity)
        dynamic_low_harmony_threshold = dynamic_low_harmony_threshold.clip(0.05, 0.3)
        dynamic_high_harmony_threshold = dynamic_high_harmony_threshold.clip(0.7, 0.95)
        position_sensitivity_factor = pd.Series(mid_harmony_neutral_factor, index=df_index)
        low_harmony_zone_mask = harmony_score < dynamic_low_harmony_threshold
        position_sensitivity_factor.loc[low_harmony_zone_mask & (inflection_strength > 0)] = low_harmony_boost_factor
        position_sensitivity_factor.loc[low_harmony_zone_mask & (inflection_strength < 0)] = 1 / low_harmony_boost_factor
        high_harmony_zone_mask = harmony_score > dynamic_high_harmony_threshold
        position_sensitivity_factor.loc[high_harmony_zone_mask & (inflection_strength < 0)] = high_harmony_boost_factor
        position_sensitivity_factor.loc[high_harmony_zone_mask & (inflection_strength > 0)] = 1 / high_harmony_boost_factor
        # 优化：传递预解析的 tf_weights 数据
        norm_deception = utils.get_adaptive_mtf_normalized_bipolar_score(deception_raw, df_index, tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_wash_trade = utils.get_adaptive_mtf_normalized_score(wash_trade_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        # --- Numba优化区域：deception_modulator ---
        deception_modulator_values = _numba_calculate_harmony_inflection_deception_modulator_core(
            inflection_strength.values,
            norm_deception.values,
            norm_wash_trade.values,
            deception_boost_factor_negative,
            deception_penalty_sensitivity,
            wash_trade_mitigation_sensitivity
        )
        deception_modulator = pd.Series(deception_modulator_values, index=df_index, dtype=np.float32)
        # --- Numba优化区域结束 ---
        inflection_strength_modulated = inflection_strength * deception_modulator
        persistence_bonus = pd.Series(0.0, index=df_index)
        positive_persistence_mask = (inflection_strength_modulated > 0) & \
                                    (inflection_strength_modulated.rolling(window=persistence_period, min_periods=1).mean() > 0)
        persistence_bonus.loc[positive_persistence_mask] = persistence_bonus_factor
        negative_persistence_mask = (inflection_strength_modulated < 0) & \
                                    (inflection_strength_modulated.rolling(window=persistence_period, min_periods=1).mean() < 0)
        persistence_bonus.loc[negative_persistence_mask] = -persistence_bonus_factor
        # 优化：传递预解析的 tf_weights 数据
        norm_chip_health = utils.get_adaptive_mtf_normalized_score(chip_health_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_volatility_instability = utils.get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_main_force_conviction = utils.get_adaptive_mtf_normalized_score(main_force_conviction_raw.abs(), df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        context_modulator = (
            (1 + norm_chip_health * context_modulator_sensitivity_health) *
            (1 + norm_volatility_instability * context_modulator_sensitivity_volatility) *
            (1 + norm_main_force_conviction * context_modulator_sensitivity_conviction)
        ).clip(0.5, 2.0)
        final_score = (inflection_strength_modulated * position_sensitivity_factor * context_modulator) + persistence_bonus
        bull_trap_penalty = self._calculate_bull_trap_context_penalty(df)
        final_score = final_score * bull_trap_penalty
        final_score = final_score.clip(-1, 1).fillna(0.0).astype(np.float32)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"  -- [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在诊断“和谐拐点”信号...")
            print(f"      [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---")
            print(f"        harmony_score: {harmony_score.loc[probe_ts]:.4f}")
            print(f"        threshold_modulator_raw: {threshold_modulator_raw.loc[probe_ts]:.4f}")
            print(f"        deception_raw: {deception_raw.loc[probe_ts]:.4f}")
            print(f"        wash_trade_raw: {wash_trade_raw.loc[probe_ts]:.4f}")
            print(f"        chip_health_raw: {chip_health_raw.loc[probe_ts]:.4f}")
            print(f"        volatility_instability_raw: {volatility_instability_raw.loc[probe_ts]:.4f}")
            print(f"        main_force_conviction_raw: {main_force_conviction_raw.loc[probe_ts]:.4f}")
            print(f"      [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 归一化信号值 ---")
            print(f"        norm_velocity: {norm_velocity.loc[probe_ts]:.4f}")
            print(f"        norm_acceleration: {norm_acceleration.loc[probe_ts]:.4f}")
            print(f"        norm_threshold_modulator: {norm_threshold_modulator.loc[probe_ts]:.4f}")
            print(f"        norm_deception: {norm_deception.loc[probe_ts]:.4f}")
            print(f"        norm_wash_trade: {norm_wash_trade.loc[probe_ts]:.4f}")
            print(f"        norm_chip_health: {norm_chip_health.loc[probe_ts]:.4f}")
            print(f"        norm_volatility_instability: {norm_volatility_instability.loc[probe_ts]:.4f}")
            print(f"        norm_main_force_conviction: {norm_main_force_conviction.loc[probe_ts]:.4f}")
            print(f"      [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 关键计算节点值 ---")
            print(f"        harmony_velocity: {harmony_velocity.loc[probe_ts]:.4f}")
            print(f"        harmony_acceleration: {harmony_acceleration.loc[probe_ts]:.4f}")
            print(f"        positive_inflection_strength: {positive_inflection_strength.loc[probe_ts]:.4f}")
            print(f"        negative_inflection_strength: {negative_inflection_strength.loc[probe_ts]:.4f}")
            print(f"        inflection_strength: {inflection_strength.loc[probe_ts]:.4f}")
            print(f"        dynamic_low_harmony_threshold: {dynamic_low_harmony_threshold.loc[probe_ts]:.4f}")
            print(f"        dynamic_high_harmony_threshold: {dynamic_high_harmony_threshold.loc[probe_ts]:.4f}")
            print(f"        position_sensitivity_factor: {position_sensitivity_factor.loc[probe_ts]:.4f}")
            print(f"        deception_modulator: {deception_modulator.loc[probe_ts]:.4f}")
            print(f"        inflection_strength_modulated: {inflection_strength_modulated.loc[probe_ts]:.4f}")
            print(f"        persistence_bonus: {persistence_bonus.loc[probe_ts]:.4f}")
            print(f"        context_modulator: {context_modulator.loc[probe_ts]:.4f}")
            print(f"        bull_trap_penalty: {bull_trap_penalty.loc[probe_ts]:.4f}")
            print(f"      [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 最终和谐拐点得分 (final_score): {final_score.loc[probe_ts]:.4f}")
            print(f"  -- [筹码层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: “和谐拐点”信号诊断完成。")
        else:
            print(f"  -- [筹码层] “和谐拐点”信号诊断完成，最新分值: {final_score.iloc[-1]:.4f}")
        return final_score

    def _diagnose_chip_retail_vulnerability(self, df: pd.DataFrame) -> pd.Series:
        """
        【V2.1 · Numba优化版】散户筹码脆弱性指数
        - 核心优化: 将modulator的计算逻辑迁移至Numba加速的辅助函数。
        量化散户持仓的集中度、平均成本与当前价格的偏离程度，以及其在市场波动下的潜在抛压。
        高分代表散户筹码结构高度不稳定，易受主力诱导而产生恐慌或盲目追涨行为。
        - 核心升级1: 引入“散户筹码结构脆弱性”维度，评估散户持仓的集中度、分散度及流量主导。
        - 核心升级2: 引入“散户行为极端化”维度，评估散户情绪和行为的非理性程度。
        - 核心升级3: 引入“主力诱导情境”维度，评估主力是否在制造有利于诱导散户的情境。
        - 核心升级4: 引入情境调制器，根据市场波动性和情绪动态调整最终分数。
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        method_name = "_diagnose_chip_retail_vulnerability"
        df_index = df.index
        df_dates_set = set(df_index.date)
        probe_dates_in_df = sorted([d for d in self.probe_dates_set if d in df_dates_set])
        should_probe_overall = self.should_probe and bool(probe_dates_in_df)
        p_conf = self.chip_ultimate_params
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        parsed_tf_data = utils._parse_tf_weights(tf_weights)
        rv_params = get_param_value(p_conf.get('chip_retail_vulnerability_params'), {})
        structure_fragility_weights = get_param_value(rv_params.get('structure_fragility_weights'), {})
        behavior_extremism_weights = get_param_value(rv_params.get('behavior_extremism_weights'), {})
        inducement_context_weights = get_param_value(rv_params.get('inducement_context_weights'), {})
        final_fusion_weights = get_param_value(rv_params.get('final_fusion_weights'), {})
        contextual_modulator_enabled = get_param_value(rv_params.get('contextual_modulator_enabled'), True)
        context_modulator_weights = get_param_value(rv_params.get('context_modulator_weights'), {})
        context_modulator_sensitivity = get_param_value(rv_params.get('context_modulator_sensitivity'), 0.5)
        final_exponent = get_param_value(rv_params.get('final_exponent'), 2.0)
        required_signals = [
            'total_winner_rate_D', 'total_loser_rate_D', 'retail_fomo_premium_index_D',
            'panic_buy_absorption_contribution_D', 'winner_concentration_90pct_D', 'loser_concentration_90pct_D',
            'cost_gini_coefficient_D', 'retail_flow_dominance_index_D', 'retail_net_flow_calibrated_D',
            'deception_index_D', 'wash_trade_intensity_D', 'main_force_conviction_index_D',
            'VOLATILITY_INSTABILITY_INDEX_21d_D', 'market_sentiment_score_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_chip_retail_vulnerability"):
            return pd.Series(0.0, index=df.index)
        signals_data = self._get_all_required_signals(df, required_signals, "_diagnose_chip_retail_vulnerability")
        is_debug_enabled = self.should_probe
        probe_ts = None
        if is_debug_enabled and self.probe_dates_set:
            for date in reversed(df_index):
                if date.date() in self.probe_dates_set:
                    probe_ts = date
                    break
        debug_info_tuple = False
        winner_concentration_raw = signals_data['winner_concentration_90pct_D']
        loser_concentration_raw = signals_data['loser_concentration_90pct_D']
        cost_gini_coefficient_raw = signals_data['cost_gini_coefficient_D']
        retail_flow_dominance_raw = signals_data['retail_flow_dominance_index_D']
        retail_fomo_premium_index_raw = signals_data['retail_fomo_premium_index_D']
        panic_buy_absorption_contribution_raw = signals_data['panic_buy_absorption_contribution_D']
        retail_net_flow_calibrated_raw = signals_data['retail_net_flow_calibrated_D']
        total_winner_rate_raw = signals_data['total_winner_rate_D']
        total_loser_rate_raw = signals_data['total_loser_rate_D']
        deception_index_raw = signals_data['deception_index_D']
        wash_trade_intensity_raw = signals_data['wash_trade_intensity_D']
        main_force_conviction_raw = signals_data['main_force_conviction_index_D']
        volatility_instability_raw = signals_data['VOLATILITY_INSTABILITY_INDEX_21d_D']
        market_sentiment_raw = signals_data['market_sentiment_score_D']
        norm_winner_concentration_inverse = utils.get_adaptive_mtf_normalized_score(winner_concentration_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_loser_concentration_inverse = utils.get_adaptive_mtf_normalized_score(loser_concentration_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_cost_gini_coefficient_inverse = utils.get_adaptive_mtf_normalized_score(cost_gini_coefficient_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_retail_flow_dominance = utils.get_adaptive_mtf_normalized_score(retail_flow_dominance_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        structure_fragility_score = utils._robust_geometric_mean(
            {
                'winner_concentration_inverse': norm_winner_concentration_inverse,
                'loser_concentration_inverse': norm_loser_concentration_inverse,
                'cost_gini_coefficient_inverse': norm_cost_gini_coefficient_inverse,
                'retail_flow_dominance': norm_retail_flow_dominance
            },
            structure_fragility_weights, df_index
        )
        norm_retail_fomo_premium = utils.get_adaptive_mtf_normalized_score(retail_fomo_premium_index_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_panic_buy_absorption_inverse = utils.get_adaptive_mtf_normalized_score(panic_buy_absorption_contribution_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_retail_net_flow_abs = utils.get_adaptive_mtf_normalized_bipolar_score(retail_net_flow_calibrated_raw, df_index, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data).abs()
        norm_total_winner_rate = utils.get_adaptive_mtf_normalized_score(total_winner_rate_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_total_loser_rate = utils.get_adaptive_mtf_normalized_score(total_loser_rate_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        behavior_extremism_score = utils._robust_geometric_mean(
            {
                'retail_fomo_premium': norm_retail_fomo_premium,
                'panic_buy_absorption_inverse': norm_panic_buy_absorption_inverse,
                'retail_net_flow_abs': norm_retail_net_flow_abs,
                'total_winner_rate': norm_total_winner_rate,
                'total_loser_rate': norm_total_loser_rate
            },
            behavior_extremism_weights, df_index
        )
        norm_deception_index_positive = utils.get_adaptive_mtf_normalized_bipolar_score(deception_index_raw, df_index, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data).clip(lower=0)
        norm_wash_trade_intensity = utils.get_adaptive_mtf_normalized_score(wash_trade_intensity_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_main_force_conviction_negative_abs = utils.get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data).clip(upper=0).abs()
        inducement_context_score = utils._robust_geometric_mean(
            {
                'deception_index_positive': norm_deception_index_positive,
                'wash_trade_intensity': norm_wash_trade_intensity,
                'main_force_conviction_negative_abs': norm_main_force_conviction_negative_abs
            },
            inducement_context_weights, df_index
        )
        initial_vulnerability_score = utils._robust_geometric_mean(
            {
                'structure_fragility': structure_fragility_score,
                'behavior_extremism': behavior_extremism_score,
                'inducement_context': inducement_context_score
            },
            final_fusion_weights, df_index
        )
        final_score = initial_vulnerability_score
        if contextual_modulator_enabled:
            norm_volatility_instability = utils.get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
            norm_market_sentiment_extreme = utils.get_adaptive_mtf_normalized_bipolar_score(market_sentiment_raw, df_index, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data).abs()
            # --- Numba优化区域：modulator ---
            modulator_values = _numba_calculate_retail_vulnerability_modulator_core(
                norm_volatility_instability.values,
                norm_market_sentiment_extreme.values,
                context_modulator_weights.get('volatility_instability', 0.5),
                context_modulator_weights.get('market_sentiment_extreme', 0.5)
            )
            modulator = pd.Series(modulator_values, index=df_index, dtype=np.float32)
            modulator = 1 + (modulator - 0.5) * context_modulator_sensitivity
            modulator = modulator.clip(0.5, 1.5)
            # --- Numba优化区域结束 ---
            final_score = final_score * modulator
        final_score = np.tanh(final_score * final_exponent)
        final_score = final_score.clip(0, 1).fillna(0.0).astype(np.float32)
        return final_score

    def _diagnose_chip_main_force_cost_intent(self, df: pd.DataFrame) -> pd.Series:
        """
        【V2.2 · Numba优化版】主力成本区攻防意图
        - 核心优化: 将deception_modulator和main_force_cost_intent_raw的计算逻辑迁移至Numba加速的辅助函数。
        诊断主力资金在其核心持仓成本区域（或关键筹码峰区域）进行主动买入或卖出的强度。
        正分代表主力在其成本区下方或附近积极承接，显示出强烈的防守或吸筹意图；
        负分代表主力在其成本区上方或附近主动派发，显示出减仓或打压意图。
        - 核心升级1: 动态成本区定义：引入 `dominant_peak_cost_D` 作为核心成本区，并根据 `BBW_21_2.0_D` 和 `chip_concentration_90pct_D` 动态调整成本区间的容忍度。
        - 核心升级2: 增强净流量质量：结合 `main_force_activity_ratio_D` 和 `main_force_flow_directionality_D` 调制 `net_conviction_flow`，评估资金流质量。
        - 核心升级3: 非线性价格偏离放大：采用 `tanh` 函数对价格偏离因子进行非线性放大，更敏感地捕捉主力意图。
        - 核心升级4: 情境化成本区内逻辑：在成本区内，结合 `SLOPE_5_chip_health_score_D` 和 `SLOPE_5_main_force_conviction_index_D` 动态调整意图强度。
        - 核心升级5: 诡道调制：引入 `deception_index_D` 和 `wash_trade_intensity_D` 作为调制器，对主力意图进行增强或削弱。
        - 核心升级6: 宏观情境调制：引入 `chip_health_score_D` 和 `VOLATILITY_INSTABILITY_INDEX_21d_D` 作为全局情境调制器，对最终分数进行校准。
        - 核心升级7: 动态权重融合：根据市场波动性和情绪，动态调整不同意图场景（低于、高于、在成本区内）的融合权重。
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        method_name = "_diagnose_chip_main_force_cost_intent"
        df_index = df.index
        df_dates_set = set(df_index.date)
        probe_dates_in_df = sorted([d for d in self.probe_dates_set if d in df_dates_set])
        should_probe_overall = self.should_probe and bool(probe_dates_in_df)
        p_conf = self.chip_ultimate_params
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        # 优化：预解析 tf_weights 一次
        parsed_tf_data = utils._parse_tf_weights(tf_weights)
        mfci_params = get_param_value(p_conf.get('main_force_cost_intent_params'), {})
        cost_zone_tolerance_base = get_param_value(mfci_params.get('cost_zone_tolerance_base'), 0.02)
        dynamic_tolerance_mod_enabled = get_param_value(mfci_params.get('dynamic_tolerance_mod_enabled'), True)
        dynamic_tolerance_mod_signal = get_param_value(mfci_params.get('dynamic_tolerance_mod_signal'), 'BBW_21_2.0_D')
        dynamic_tolerance_sensitivity = get_param_value(mfci_params.get('dynamic_tolerance_sensitivity'), 0.5)
        price_deviation_tanh_factor = get_param_value(mfci_params.get('price_deviation_tanh_factor'), 5.0)
        in_zone_intent_base_multiplier = get_param_value(mfci_params.get('in_zone_intent_base_multiplier'), 0.5)
        in_zone_health_slope_sensitivity = get_param_value(mfci_params.get('in_zone_health_slope_sensitivity'), 0.3)
        deception_mod_enabled = get_param_value(mfci_params.get('deception_mod_enabled'), True)
        deception_boost_factor = get_param_value(mfci_params.get('deception_boost_factor'), 0.5)
        deception_penalty_factor = get_param_value(mfci_params.get('deception_penalty_factor'), 0.7)
        wash_trade_penalty_factor = get_param_value(mfci_params.get('wash_trade_penalty_factor'), 0.3)
        global_context_mod_enabled = get_param_value(mfci_params.get('global_context_mod_enabled'), True)
        global_context_sensitivity_health = get_param_value(mfci_params.get('global_context_sensitivity_health'), 0.5)
        global_context_sensitivity_volatility = get_param_value(mfci_params.get('global_context_sensitivity_volatility'), 0.3)
        dynamic_fusion_weights_enabled = get_param_value(mfci_params.get('dynamic_fusion_weights_enabled'), True)
        dynamic_fusion_weights_base = get_param_value(mfci_params.get('dynamic_fusion_weights_base'), {'below_vpoc': 0.4, 'above_vpoc': 0.4, 'in_vpoc': 0.2})
        dynamic_weight_mod_signal = get_param_value(mfci_params.get('dynamic_weight_mod_signal'), 'market_sentiment_score_D')
        dynamic_weight_sensitivity = get_param_value(mfci_params.get('dynamic_weight_sensitivity'), 0.3)
        required_signals = [
            'close_D', 'vpoc_D', 'dominant_peak_cost_D', 'main_force_conviction_index_D',
            'conviction_flow_buy_intensity_D', 'conviction_flow_sell_intensity_D',
            'main_force_activity_ratio_D', 'main_force_flow_directionality_D',
            'deception_index_D', 'wash_trade_intensity_D', 'chip_health_score_D',
            'SLOPE_5_chip_health_score_D', 'SLOPE_5_main_force_conviction_index_D',
            'BBW_21_2.0_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D', 'market_sentiment_score_D'
        ]
        if dynamic_tolerance_mod_enabled and dynamic_tolerance_mod_signal not in required_signals:
            required_signals.append(dynamic_tolerance_mod_signal)
        if dynamic_fusion_weights_enabled and dynamic_weight_mod_signal not in required_signals:
            required_signals.append(dynamic_weight_mod_signal)
        if not self._validate_required_signals(df, required_signals, "_diagnose_chip_main_force_cost_intent"):
            return pd.Series(0.0, index=df.index)
        signals_data = self._get_all_required_signals(df, required_signals, "_diagnose_chip_main_force_cost_intent")
        is_debug_enabled = self.should_probe
        probe_ts = None
        if is_debug_enabled and self.probe_dates_set:
            for date in reversed(df_index):
                if date.date() in self.probe_dates_set:
                    probe_ts = date
                    break
        debug_info_tuple = (is_debug_enabled, probe_ts, "_diagnose_chip_main_force_cost_intent")
        close_raw = signals_data['close_D']
        vpoc_raw = signals_data['vpoc_D']
        dominant_peak_cost_raw = signals_data['dominant_peak_cost_D']
        main_force_conviction_raw = signals_data['main_force_conviction_index_D']
        conviction_flow_buy_raw = signals_data['conviction_flow_buy_intensity_D']
        conviction_flow_sell_raw = signals_data['conviction_flow_sell_intensity_D']
        main_force_activity_ratio_raw = signals_data['main_force_activity_ratio_D']
        main_force_flow_directionality_raw = signals_data['main_force_flow_directionality_D']
        deception_index_raw = signals_data['deception_index_D']
        wash_trade_intensity_raw = signals_data['wash_trade_intensity_D']
        chip_health_score_raw = signals_data['chip_health_score_D']
        slope_5_chip_health_raw = signals_data['SLOPE_5_chip_health_score_D']
        slope_5_main_force_conviction_raw = signals_data['SLOPE_5_main_force_conviction_index_D']
        volatility_instability_raw = signals_data['VOLATILITY_INSTABILITY_INDEX_21d_D']
        market_sentiment_raw = signals_data['market_sentiment_score_D']
        dynamic_tolerance_mod_raw = signals_data[dynamic_tolerance_mod_signal] if dynamic_tolerance_mod_enabled else None
        dynamic_weight_mod_raw = signals_data[dynamic_weight_mod_signal] if dynamic_fusion_weights_enabled else None
        cost_center = dominant_peak_cost_raw.fillna(vpoc_raw)
        dynamic_cost_zone_tolerance = pd.Series(cost_zone_tolerance_base, index=df_index)
        if dynamic_tolerance_mod_enabled and dynamic_tolerance_mod_raw is not None:
            # 优化：传递预解析的 tf_weights 数据
            norm_dynamic_tolerance_mod = utils.get_adaptive_mtf_normalized_score(dynamic_tolerance_mod_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
            dynamic_cost_zone_tolerance = cost_zone_tolerance_base * (1 + norm_dynamic_tolerance_mod * dynamic_tolerance_sensitivity)
            dynamic_cost_zone_tolerance = dynamic_cost_zone_tolerance.clip(cost_zone_tolerance_base * 0.5, cost_zone_tolerance_base * 2.0)
        upper_bound = cost_center * (1 + dynamic_cost_zone_tolerance)
        lower_bound = cost_center * (1 - dynamic_cost_zone_tolerance)
        # --- 修正：先定义 net_conviction_flow ---
        net_conviction_flow = conviction_flow_buy_raw - conviction_flow_sell_raw
        # 优化：传递预解析的 tf_weights 数据
        norm_positive_flow = utils.get_adaptive_mtf_normalized_score(net_conviction_flow.clip(lower=0), df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_negative_flow = utils.get_adaptive_mtf_normalized_score(net_conviction_flow.clip(upper=0).abs(), df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_net_conviction_flow_directional = norm_positive_flow - norm_negative_flow
        # 优化：传递预解析的 tf_weights 数据
        norm_main_force_activity = utils.get_adaptive_mtf_normalized_score(main_force_activity_ratio_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_main_force_flow_directionality = utils.get_adaptive_mtf_normalized_bipolar_score(main_force_flow_directionality_raw, df_index, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        flow_quality_modulator = (norm_main_force_activity * 0.5 + norm_main_force_flow_directionality.abs() * 0.5).clip(0, 1)
        net_conviction_flow_quality = norm_net_conviction_flow_directional * (1 + flow_quality_modulator * 0.5)
        net_conviction_flow_quality = net_conviction_flow_quality.clip(-1, 1)
        # 优化：传递预解析的 tf_weights 数据
        norm_main_force_conviction = utils.get_adaptive_mtf_normalized_score(main_force_conviction_raw.abs(), df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        price_deviation_factor_buy = (cost_center - close_raw) / cost_center.replace(0, np.nan)
        price_deviation_factor_buy = np.tanh(price_deviation_factor_buy.clip(0, 0.1) * price_deviation_tanh_factor)
        price_deviation_factor_sell = (close_raw - cost_center) / cost_center.replace(0, np.nan)
        price_deviation_factor_sell = np.tanh(price_deviation_factor_sell.clip(0, 0.1) * price_deviation_tanh_factor)
        # 优化：传递预解析的 tf_weights 数据
        norm_slope_5_chip_health = utils.get_adaptive_mtf_normalized_bipolar_score(slope_5_chip_health_raw, df_index, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_slope_5_main_force_conviction = utils.get_adaptive_mtf_normalized_bipolar_score(slope_5_main_force_conviction_raw, df_index, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        in_zone_intent_modulator = (norm_slope_5_chip_health * 0.5 + norm_slope_5_main_force_conviction * 0.5).clip(-1, 1)
        # --- Numba优化区域：deception_modulator ---
        deception_modulator = pd.Series(1.0, index=df_index)
        if deception_mod_enabled:
            norm_deception_index_bipolar = utils.get_adaptive_mtf_normalized_bipolar_score(deception_index_raw, df_index, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
            norm_wash_trade_intensity = utils.get_adaptive_mtf_normalized_score(wash_trade_intensity_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
            deception_modulator_values = _numba_calculate_main_force_cost_intent_deception_modulator_core(
                net_conviction_flow_quality.values,
                norm_deception_index_bipolar.values,
                norm_wash_trade_intensity.values,
                deception_boost_factor,
                deception_penalty_factor,
                wash_trade_penalty_factor
            )
            deception_modulator = pd.Series(deception_modulator_values, index=df_index, dtype=np.float32)
        # --- Numba优化区域结束 ---
        global_context_modulator = pd.Series(1.0, index=df_index)
        if global_context_mod_enabled:
            # 优化：传递预解析的 tf_weights 数据
            norm_chip_health = utils.get_adaptive_mtf_normalized_score(chip_health_score_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
            norm_volatility_instability = utils.get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
            global_context_modulator = (
                (1 + norm_chip_health * global_context_sensitivity_health) *
                (1 + norm_volatility_instability * global_context_sensitivity_volatility)
            ).clip(0.5, 1.5)
        dynamic_weight_below_vpoc = pd.Series(dynamic_fusion_weights_base.get('below_vpoc', 0.4), index=df_index)
        dynamic_weight_above_vpoc = pd.Series(dynamic_fusion_weights_base.get('above_vpoc', 0.4), index=df_index)
        dynamic_weight_in_vpoc = pd.Series(dynamic_fusion_weights_base.get('in_vpoc', 0.2), index=df_index)
        if dynamic_fusion_weights_enabled and dynamic_weight_mod_raw is not None:
            # 优化：传递预解析的 tf_weights 数据
            norm_dynamic_weight_mod = utils.get_adaptive_mtf_normalized_bipolar_score(dynamic_weight_mod_raw, df_index, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
            mod_factor = norm_dynamic_weight_mod * dynamic_weight_sensitivity
            dynamic_weight_below_vpoc = (dynamic_weight_below_vpoc - mod_factor).clip(0.1, 0.7)
            dynamic_weight_above_vpoc = (dynamic_weight_above_vpoc + mod_factor).clip(0.1, 0.7)
            dynamic_weight_in_vpoc = (dynamic_weight_in_vpoc - mod_factor.abs() * 0.5).clip(0.05, 0.4)
            sum_dynamic_weights = dynamic_weight_below_vpoc + dynamic_weight_above_vpoc + dynamic_weight_in_vpoc
            dynamic_weight_below_vpoc /= sum_dynamic_weights
            dynamic_weight_above_vpoc /= sum_dynamic_weights
            dynamic_weight_in_vpoc /= sum_dynamic_weights
        # --- Numba优化区域：main_force_cost_intent_raw ---
        main_force_cost_intent_raw_values = _numba_calculate_main_force_cost_intent_core(
            close_raw.values,
            lower_bound.values,
            upper_bound.values,
            net_conviction_flow_quality.values,
            norm_main_force_conviction.values,
            price_deviation_factor_buy.values,
            price_deviation_factor_sell.values,
            in_zone_intent_base_multiplier,
            in_zone_intent_modulator.values,
            in_zone_health_slope_sensitivity,
            dynamic_weight_below_vpoc.values,
            dynamic_weight_above_vpoc.values,
            dynamic_weight_in_vpoc.values
        )
        main_force_cost_intent_raw = pd.Series(main_force_cost_intent_raw_values, index=df_index, dtype=np.float32)
        # --- Numba优化区域结束 ---
        final_score = main_force_cost_intent_raw * deception_modulator * global_context_modulator
        final_score = final_score.clip(-1, 1).fillna(0.0).astype(np.float32)
        return final_score

    def _diagnose_chip_hollowing_out_risk(self, df: pd.DataFrame) -> pd.Series:
        """
        【V3.1 · Numba优化版】筹码空心化风险
        - 核心优化: 将hollowing_out_risk_score和deception_amplifier的计算逻辑迁移至Numba加速的辅助函数。
        评估筹码结构中，主力核心持仓的稳定性与数量，以及高位套牢盘或短期获利盘的比例。
        高分代表主力核心筹码正在流失，市场筹码结构出现“空心化”迹象，即大部分筹码由不稳定资金在高位持有，
        一旦下跌容易引发连锁抛售。
        - 核心升级1: 引入四大核心维度：核心筹码分散与弱化、派发压力与获利了结、主力意图与诡道、市场情境与脆弱性。
        - 核心升级2: 动态融合权重：根据市场波动性和情绪动态调整四大维度的融合权重，使信号自适应市场环境。
        - 核心升级3: 诡道放大机制：主力意图与诡道维度对最终分数进行乘性放大，增强对欺骗性风险的识别。
        - 核心升级4: 引入更多筹码相关原始数据，并结合其斜率，更全面地捕捉空心化风险的动态演变。
        - 探针增强: 详细输出所有原始数据、归一化数据、各维度子分数、动态权重、最终分数，以便于检查和调试。
        """
        method_name = "_diagnose_chip_hollowing_out_risk"
        df_index = df.index
        p_conf = self.chip_ultimate_params
        hollow_params = get_param_value(p_conf.get('chip_hollowing_out_risk_params'), {})
        probe_enabled = get_param_value(hollow_params.get('probe_enabled'), False)
        should_probe_overall = self.should_probe and probe_enabled
        df_dates_set = set(df_index.date)
        probe_dates_in_df = sorted([d for d in self.probe_dates_set if d in df_dates_set])
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        parsed_tf_data = utils._parse_tf_weights(tf_weights)
        dispersion_weakness_weights = get_param_value(hollow_params.get('dispersion_weakness_weights'), {})
        distribution_pressure_weights = get_param_value(hollow_params.get('distribution_pressure_weights'), {})
        main_force_deception_weights = get_param_value(hollow_params.get('main_force_deception_weights'), {})
        market_vulnerability_weights = get_param_value(hollow_params.get('market_vulnerability_weights'), {})
        final_fusion_weights_base = get_param_value(hollow_params.get('final_fusion_weights'), {})
        dynamic_fusion_modulator_params = get_param_value(hollow_params.get('dynamic_fusion_modulator_params'), {})
        deception_amplification_factor = get_param_value(hollow_params.get('deception_amplification_factor'), 1.5)
        non_linear_exponent = get_param_value(hollow_params.get('non_linear_exponent'), 2.0)
        required_signals = [
            'winner_concentration_90pct_D', 'loser_concentration_90pct_D', 'cost_gini_coefficient_D',
            'dominant_peak_solidity_D', 'peak_separation_ratio_D', 'SLOPE_5_winner_concentration_90pct_D',
            'SLOPE_5_chip_health_score_D', 'total_winner_rate_D', 'winner_profit_margin_avg_D',
            'rally_distribution_pressure_D', 'profit_taking_flow_ratio_D', 'upper_shadow_selling_pressure_D',
            'covert_distribution_signal_D', 'SLOPE_5_rally_distribution_pressure_D',
            'deception_index_D', 'wash_trade_intensity_D', 'main_force_conviction_index_D',
            'main_force_net_flow_calibrated_D', 'main_force_cost_advantage_D',
            'retail_fomo_premium_index_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D',
            'market_sentiment_score_D', 'flow_credibility_index_D', 'structural_tension_index_D'
        ]
        if get_param_value(dynamic_fusion_modulator_params.get('enabled'), False):
            mod_signal_1 = get_param_value(dynamic_fusion_modulator_params.get('modulator_signal_1'))
            mod_signal_2 = get_param_value(dynamic_fusion_modulator_params.get('modulator_signal_2'))
            if mod_signal_1 and mod_signal_1 not in required_signals: required_signals.append(mod_signal_1)
            if mod_signal_2 and mod_signal_2 not in required_signals: required_signals.append(mod_signal_2)
        if not self._validate_required_signals(df, required_signals, "_diagnose_chip_hollowing_out_risk"):
            return pd.Series(0.0, index=df.index)
        signals_data = self._get_all_required_signals(df, required_signals, "_diagnose_chip_hollowing_out_risk")
        is_debug_enabled = self.should_probe and probe_enabled
        probe_ts = None
        if is_debug_enabled and self.probe_dates_set:
            for date in reversed(df_index):
                if date.date() in self.probe_dates_set:
                    probe_ts = date
                    break
        debug_info_tuple = False
        winner_concentration_raw = signals_data['winner_concentration_90pct_D']
        loser_concentration_raw = signals_data['loser_concentration_90pct_D']
        cost_gini_coefficient_raw = signals_data['cost_gini_coefficient_D']
        dominant_peak_solidity_raw = signals_data['dominant_peak_solidity_D']
        peak_separation_ratio_raw = signals_data['peak_separation_ratio_D']
        slope_winner_concentration_raw = signals_data['SLOPE_5_winner_concentration_90pct_D']
        slope_chip_health_raw = signals_data['SLOPE_5_chip_health_score_D']
        total_winner_rate_raw = signals_data['total_winner_rate_D']
        winner_profit_margin_avg_raw = signals_data['winner_profit_margin_avg_D']
        rally_distribution_pressure_raw = signals_data['rally_distribution_pressure_D']
        profit_taking_flow_ratio_raw = signals_data['profit_taking_flow_ratio_D']
        upper_shadow_selling_pressure_raw = signals_data['upper_shadow_selling_pressure_D']
        covert_distribution_signal_raw = signals_data['covert_distribution_signal_D']
        slope_rally_distribution_pressure_raw = signals_data['SLOPE_5_rally_distribution_pressure_D']
        deception_index_raw = signals_data['deception_index_D']
        wash_trade_intensity_raw = signals_data['wash_trade_intensity_D']
        main_force_conviction_raw = signals_data['main_force_conviction_index_D']
        main_force_net_flow_calibrated_raw = signals_data['main_force_net_flow_calibrated_D']
        main_force_cost_advantage_raw = signals_data['main_force_cost_advantage_D']
        retail_fomo_premium_index_raw = signals_data['retail_fomo_premium_index_D']
        volatility_instability_raw = signals_data['VOLATILITY_INSTABILITY_INDEX_21d_D']
        market_sentiment_raw = signals_data['market_sentiment_score_D']
        flow_credibility_raw = signals_data['flow_credibility_index_D']
        structural_tension_raw = signals_data['structural_tension_index_D']
        norm_winner_concentration_inverse = utils.get_adaptive_mtf_normalized_score(winner_concentration_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_loser_concentration_high_price = utils.get_adaptive_mtf_normalized_score(loser_concentration_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_cost_gini_coefficient_inverse = utils.get_adaptive_mtf_normalized_score(cost_gini_coefficient_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_dominant_peak_solidity_inverse = utils.get_adaptive_mtf_normalized_score(dominant_peak_solidity_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_peak_separation_ratio = utils.get_adaptive_mtf_normalized_score(peak_separation_ratio_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_winner_concentration_slope_inverse = utils.get_adaptive_mtf_normalized_score(slope_winner_concentration_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_chip_health_score_inverse_slope = utils.get_adaptive_mtf_normalized_score(slope_chip_health_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        dispersion_weakness_score = utils._robust_geometric_mean(
            {
                'winner_concentration_inverse': norm_winner_concentration_inverse,
                'loser_concentration_high_price': norm_loser_concentration_high_price,
                'cost_gini_coefficient_inverse': norm_cost_gini_coefficient_inverse,
                'dominant_peak_solidity_inverse': norm_dominant_peak_solidity_inverse,
                'peak_separation_ratio': norm_peak_separation_ratio,
                'winner_concentration_slope_inverse': norm_winner_concentration_slope_inverse,
                'chip_health_score_inverse_slope': norm_chip_health_score_inverse_slope
            },
            dispersion_weakness_weights, df_index
        )
        norm_total_winner_rate = utils.get_adaptive_mtf_normalized_score(total_winner_rate_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_winner_profit_margin_avg = utils.get_adaptive_mtf_normalized_score(winner_profit_margin_avg_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_rally_distribution_pressure = utils.get_adaptive_mtf_normalized_score(rally_distribution_pressure_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_profit_taking_flow_ratio = utils.get_adaptive_mtf_normalized_score(profit_taking_flow_ratio_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_upper_shadow_selling_pressure = utils.get_adaptive_mtf_normalized_score(upper_shadow_selling_pressure_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_covert_distribution_signal = utils.get_adaptive_mtf_normalized_score(covert_distribution_signal_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_rally_distribution_pressure_slope = utils.get_adaptive_mtf_normalized_score(slope_rally_distribution_pressure_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        distribution_pressure_score = utils._robust_geometric_mean(
            {
                'total_winner_rate': norm_total_winner_rate,
                'winner_profit_margin_avg': norm_winner_profit_margin_avg,
                'rally_distribution_pressure': norm_rally_distribution_pressure,
                'profit_taking_flow_ratio': norm_profit_taking_flow_ratio,
                'upper_shadow_selling_pressure': norm_upper_shadow_selling_pressure,
                'covert_distribution_signal': norm_covert_distribution_signal,
                'rally_distribution_pressure_slope': norm_rally_distribution_pressure_slope
            },
            distribution_pressure_weights, df_index
        )
        norm_deception_index_positive = utils.get_adaptive_mtf_normalized_bipolar_score(deception_index_raw, df_index, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data).clip(lower=0)
        norm_wash_trade_intensity = utils.get_adaptive_mtf_normalized_score(wash_trade_intensity_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_main_force_conviction_negative = utils.get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data).clip(upper=0).abs()
        norm_main_force_net_flow_negative = utils.get_adaptive_mtf_normalized_score(main_force_net_flow_calibrated_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_main_force_cost_advantage_negative = utils.get_adaptive_mtf_normalized_bipolar_score(main_force_cost_advantage_raw, df_index, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data).clip(upper=0).abs()
        main_force_deception_score = utils._robust_geometric_mean(
            {
                'deception_index_positive': norm_deception_index_positive,
                'wash_trade_intensity': norm_wash_trade_intensity,
                'main_force_conviction_negative': norm_main_force_conviction_negative,
                'main_force_net_flow_negative': norm_main_force_net_flow_negative,
                'main_force_cost_advantage_negative': norm_main_force_cost_advantage_negative
            },
            main_force_deception_weights, df_index
        )
        norm_retail_fomo_premium_index = utils.get_adaptive_mtf_normalized_score(retail_fomo_premium_index_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_volatility_instability_index = utils.get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_market_sentiment_extreme = utils.get_adaptive_mtf_normalized_bipolar_score(market_sentiment_raw, df_index, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data).abs()
        norm_flow_credibility_inverse = utils.get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_structural_tension_index = utils.get_adaptive_mtf_normalized_score(structural_tension_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        market_vulnerability_score = utils._robust_geometric_mean(
            {
                'retail_fomo_premium_index': norm_retail_fomo_premium_index,
                'volatility_instability_index': norm_volatility_instability_index,
                'market_sentiment_extreme': norm_market_sentiment_extreme,
                'flow_credibility_inverse': norm_flow_credibility_inverse,
                'structural_tension_index': norm_structural_tension_index
            },
            market_vulnerability_weights, df_index
        )
        dynamic_fusion_weights = final_fusion_weights_base.copy()
        if get_param_value(dynamic_fusion_modulator_params.get('enabled'), False):
            mod_signal_1_name = get_param_value(dynamic_fusion_modulator_params.get('modulator_signal_1'))
            mod_signal_2_name = get_param_value(dynamic_fusion_modulator_params.get('modulator_signal_2'))
            sensitivity_volatility = get_param_value(dynamic_fusion_modulator_params.get('sensitivity_volatility'))
            sensitivity_sentiment = get_param_value(dynamic_fusion_modulator_params.get('sensitivity_sentiment'))
            norm_mod_signal_1 = utils.get_adaptive_mtf_normalized_score(signals_data[mod_signal_1_name], df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
            norm_mod_signal_2 = utils.get_adaptive_mtf_normalized_bipolar_score(signals_data[mod_signal_2_name], df_index, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
            current_dispersion_weight = pd.Series(final_fusion_weights_base.get('dispersion_weakness', 0.0), index=df_index)
            current_distribution_weight = pd.Series(final_fusion_weights_base.get('distribution_pressure', 0.0), index=df_index)
            current_deception_weight = pd.Series(final_fusion_weights_base.get('main_force_deception', 0.0), index=df_index)
            current_vulnerability_weight = pd.Series(final_fusion_weights_base.get('market_vulnerability', 0.0), index=df_index)
            volatility_impact_weights = get_param_value(dynamic_fusion_modulator_params.get('volatility_impact_weights'), {})
            current_dispersion_weight += norm_mod_signal_1 * sensitivity_volatility * volatility_impact_weights.get('dispersion_weakness', 0.0)
            current_distribution_weight += norm_mod_signal_1 * sensitivity_volatility * volatility_impact_weights.get('distribution_pressure', 0.0)
            current_deception_weight += norm_mod_signal_1 * sensitivity_volatility * volatility_impact_weights.get('main_force_deception', 0.0)
            current_vulnerability_weight += norm_mod_signal_1 * sensitivity_volatility * volatility_impact_weights.get('market_vulnerability', 0.0)
            sentiment_impact_weights = get_param_value(dynamic_fusion_modulator_params.get('sentiment_impact_weights'), {})
            current_dispersion_weight += norm_mod_signal_2 * sensitivity_sentiment * sentiment_impact_weights.get('dispersion_weakness', 0.0)
            current_distribution_weight += norm_mod_signal_2 * sensitivity_sentiment * sentiment_impact_weights.get('distribution_pressure', 0.0)
            current_deception_weight += norm_mod_signal_2 * sensitivity_sentiment * sentiment_impact_weights.get('main_force_deception', 0.0)
            current_vulnerability_weight += norm_mod_signal_2 * sensitivity_sentiment * sentiment_impact_weights.get('market_vulnerability', 0.0)
            current_dispersion_weight = current_dispersion_weight.clip(0.05, 0.5)
            current_distribution_weight = current_distribution_weight.clip(0.05, 0.5)
            current_deception_weight = current_deception_weight.clip(0.05, 0.5)
            current_vulnerability_weight = current_vulnerability_weight.clip(0.05, 0.5)
            sum_dynamic_weights = current_dispersion_weight + current_distribution_weight + current_deception_weight + current_vulnerability_weight
            dynamic_fusion_weights['dispersion_weakness'] = current_dispersion_weight / sum_dynamic_weights
            dynamic_fusion_weights['distribution_pressure'] = current_distribution_weight / sum_dynamic_weights
            dynamic_fusion_weights['main_force_deception'] = current_deception_weight / sum_dynamic_weights
            dynamic_fusion_weights['market_vulnerability'] = current_vulnerability_weight / sum_dynamic_weights
        else:
            dynamic_fusion_weights = {k: pd.Series(v, index=df_index) for k, v in final_fusion_weights_base.items()}
        # --- Numba优化区域：hollowing_out_risk_score ---
        final_score_values = _numba_calculate_hollowing_out_risk_core(
            dispersion_weakness_score.values,
            distribution_pressure_score.values,
            main_force_deception_score.values,
            market_vulnerability_score.values,
            dynamic_fusion_weights['dispersion_weakness'].values,
            dynamic_fusion_weights['distribution_pressure'].values,
            dynamic_fusion_weights['main_force_deception'].values,
            dynamic_fusion_weights['market_vulnerability'].values,
            deception_amplification_factor,
            non_linear_exponent
        )
        final_score = pd.Series(final_score_values, index=df_index, dtype=np.float32)
        # --- Numba优化区域结束 ---
        return final_score

    def _diagnose_chip_turnover_purity_cost_optimization(self, df: pd.DataFrame) -> pd.Series:
        """
        【V1.1 · Numba优化版】换手纯度与成本优化
        - 核心优化: 将turnover_purity_cost_optimization的计算逻辑迁移至Numba加速的辅助函数。
        评估换手过程中，筹码从高成本、不稳定持仓向低成本、稳定持仓转移的效率和纯度。
        高分代表换手是健康的，有助于优化筹码结构，降低整体持仓成本，为后续上涨奠定基础；
        低分或负分代表换手是恶性的，筹码从低成本向高成本转移，或伴随大量对倒和虚假交易。
        """
        df_index = df.index
        required_signals = [
            'wash_trade_intensity_D', 'conviction_flow_buy_intensity_D', 'conviction_flow_sell_intensity_D',
            'winner_profit_margin_avg_D', 'loser_pain_index_D', 'turnover_rate_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_chip_turnover_purity_cost_optimization"):
            return pd.Series(0.0, index=df.index)
        signals_data = self._get_all_required_signals(df, required_signals, "_diagnose_chip_turnover_purity_cost_optimization")
        p_conf = self.chip_ultimate_params
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        parsed_tf_data = utils._parse_tf_weights(tf_weights)
        is_debug_enabled = self.should_probe
        probe_ts = None
        if is_debug_enabled and self.probe_dates_set:
            for date in reversed(df_index):
                if date.date() in self.probe_dates_set:
                    probe_ts = date
                    break
        debug_info_tuple = False
        wash_trade_intensity_raw = signals_data['wash_trade_intensity_D']
        conviction_flow_buy_raw = signals_data['conviction_flow_buy_intensity_D']
        conviction_flow_sell_raw = signals_data['conviction_flow_sell_intensity_D']
        winner_profit_margin_avg_raw = signals_data['winner_profit_margin_avg_D']
        loser_pain_index_raw = signals_data['loser_pain_index_D']
        turnover_rate_raw = signals_data['turnover_rate_D']
        norm_wash_trade_intensity = utils.get_adaptive_mtf_normalized_score(wash_trade_intensity_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        net_conviction_flow = conviction_flow_buy_raw - conviction_flow_sell_raw
        norm_net_conviction_flow = utils.get_adaptive_mtf_normalized_bipolar_score(net_conviction_flow, df_index, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_winner_profit_margin_avg = utils.get_adaptive_mtf_normalized_score(winner_profit_margin_avg_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_loser_pain_index = utils.get_adaptive_mtf_normalized_score(loser_pain_index_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_turnover_rate = utils.get_adaptive_mtf_normalized_score(turnover_rate_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        # --- Numba优化区域：turnover_purity_cost_optimization ---
        final_score_values = _numba_calculate_turnover_purity_cost_optimization_core(
            norm_wash_trade_intensity.values,
            norm_net_conviction_flow.values,
            norm_winner_profit_margin_avg.values,
            norm_loser_pain_index.values,
            norm_turnover_rate.values
        )
        final_score = pd.Series(final_score_values, index=df_index, dtype=np.float32)
        # --- Numba优化区域结束 ---
        return final_score

    def _diagnose_chip_despair_temptation_zones(self, df: pd.DataFrame) -> pd.Series:
        """
        【V1.1 · Numba优化版】筹码绝望与诱惑区
        - 核心优化: 将despair_strength和temptation_strength的计算逻辑迁移至Numba加速的辅助函数。
        识别当前筹码分布中，散户或弱势资金处于极端亏损（绝望区）或极端浮盈（诱惑区）的价格区间。
        正分代表诱惑区风险（主力派发），负分代表绝望区机会（主力吸筹）。
        """
        df_index = df.index
        required_signals = [
            'loser_pain_index_D', 'total_loser_rate_D', 'panic_buy_absorption_contribution_D',
            'retail_fomo_premium_index_D', 'winner_profit_margin_avg_D', 'total_winner_rate_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_chip_despair_temptation_zones"):
            return pd.Series(0.0, index=df.index)
        signals_data = self._get_all_required_signals(df, required_signals, "_diagnose_chip_despair_temptation_zones")
        p_conf = self.chip_ultimate_params
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        parsed_tf_data = utils._parse_tf_weights(tf_weights)
        is_debug_enabled = self.should_probe
        probe_ts = None
        if is_debug_enabled and self.probe_dates_set:
            for date in reversed(df_index):
                if date.date() in self.probe_dates_set:
                    probe_ts = date
                    break
        debug_info_tuple = False
        loser_pain_index_raw = signals_data['loser_pain_index_D']
        total_loser_rate_raw = signals_data['total_loser_rate_D']
        panic_buy_absorption_contribution_raw = signals_data['panic_buy_absorption_contribution_D']
        retail_fomo_premium_index_raw = signals_data['retail_fomo_premium_index_D']
        winner_profit_margin_avg_raw = signals_data['winner_profit_margin_avg_D']
        total_winner_rate_raw = signals_data['total_winner_rate_D']
        norm_loser_pain_index = utils.get_adaptive_mtf_normalized_score(loser_pain_index_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_total_loser_rate = utils.get_adaptive_mtf_normalized_score(total_loser_rate_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_panic_buy_absorption_contribution = utils.get_adaptive_mtf_normalized_score(panic_buy_absorption_contribution_raw, df_index, ascending=False, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_retail_fomo_premium_index = utils.get_adaptive_mtf_normalized_score(retail_fomo_premium_index_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_winner_profit_margin_avg = utils.get_adaptive_mtf_normalized_score(winner_profit_margin_avg_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        norm_total_winner_rate_temptation = utils.get_adaptive_mtf_normalized_score(total_winner_rate_raw, df_index, ascending=True, tf_weights=tf_weights, debug_info=False, _parsed_tf_data=parsed_tf_data)
        # --- Numba优化区域：despair_temptation_score ---
        final_score_values = _numba_calculate_despair_temptation_zones_core(
            norm_loser_pain_index.values,
            norm_total_loser_rate.values,
            norm_panic_buy_absorption_contribution.values,
            norm_retail_fomo_premium_index.values,
            norm_winner_profit_margin_avg.values,
            norm_total_winner_rate_temptation.values
        )
        final_score = pd.Series(final_score_values, index=df_index, dtype=np.float32)
        # --- Numba优化区域结束 ---
        return final_score

    def _calculate_bull_trap_context_penalty(self, df: pd.DataFrame) -> pd.Series:
        """
        【V1.6 · Numba优化版】计算在近期大幅下跌后，伴随欺骗性反弹情境下的惩罚因子。
        - 核心优化: 将penalty_factor的计算逻辑迁移至Numba加速的辅助函数。
        - 核心逻辑: 检测近期是否存在大幅下跌，同时当前是否存在正向欺骗信号。
                    正向欺骗信号的判断现在基于 `deception_index_D` 及其多时间周期（斜率、加速度）的融合。
                    结合市场波动性作为情境调制器，动态调整惩罚强度。
        - 调试增强: 增加详细打印，追踪牛市陷阱检测的各个中间步骤，包括多维欺骗信号的融合。
        - 业务逻辑修正: 恢复使用 `deception_index_D`，并增强其“正向欺骗”的判断逻辑，引入多时间周期分析。
        - 返回值: 一个 Series，值为 0 到 1 之间。1 表示无惩罚，0 表示完全惩罚。
        """
        df_index = df.index
        p_conf = self.chip_ultimate_params
        bt_params = get_param_value(p_conf.get('bull_trap_detection_params'), {})
        if not get_param_value(bt_params.get('enabled'), False):
            return pd.Series(1.0, index=df_index, dtype=np.float32)
        recent_sharp_drop_window = get_param_value(bt_params.get('recent_sharp_drop_window'), 3)
        min_sharp_drop_pct = get_param_value(bt_params.get('min_sharp_drop_pct'), -0.05)
        deception_penalty_multiplier = get_param_value(bt_params.get('deception_penalty_multiplier'), 2.5)
        context_modulator_signal_name = get_param_value(bt_params.get('context_modulator_signal'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        context_modulator_sensitivity = get_param_value(bt_params.get('context_modulator_sensitivity'), 0.7)
        deception_signal_to_use = 'deception_index_D'
        required_signals = ['pct_change_D', deception_signal_to_use, context_modulator_signal_name]
        if not self._validate_required_signals(df, required_signals, "_calculate_bull_trap_context_penalty"):
            return pd.Series(1.0, index=df_index, dtype=np.float32)
        signals_data = self._get_all_required_signals(df, required_signals, "_calculate_bull_trap_context_penalty")
        pct_change_raw = signals_data['pct_change_D'] / 100
        deception_index_raw = signals_data[deception_signal_to_use]
        context_modulator_raw = signals_data[context_modulator_signal_name]
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        parsed_tf_data = utils._parse_tf_weights(tf_weights)
        is_debug_enabled = self.should_probe
        probe_dates_set = self.probe_dates_set
        probe_ts = None
        if is_debug_enabled and probe_dates_set:
            for date in reversed(df_index):
                if date.date() in probe_dates_set:
                    probe_ts = date
                    break
        min_pct_change_in_window = pct_change_raw.rolling(window=recent_sharp_drop_window, min_periods=1).min()
        has_recent_sharp_drop = (min_pct_change_in_window <= min_sharp_drop_pct)
        deception_slope_raw = deception_index_raw.diff(1).fillna(0)
        deception_accel_raw = deception_slope_raw.diff(1).fillna(0)
        norm_deception_index_bipolar = utils.get_adaptive_mtf_normalized_bipolar_score(deception_index_raw, df_index, tf_weights, _parsed_tf_data=parsed_tf_data)
        norm_deception_slope_bipolar = utils.get_adaptive_mtf_normalized_bipolar_score(deception_slope_raw, df_index, tf_weights, _parsed_tf_data=parsed_tf_data)
        norm_deception_accel_bipolar = utils.get_adaptive_mtf_normalized_bipolar_score(deception_accel_raw, df_index, tf_weights, _parsed_tf_data=parsed_tf_data)
        deception_index_weight = get_param_value(bt_params.get('deception_index_weight', 0.5), 0.5)
        deception_slope_weight = get_param_value(bt_params.get('deception_slope_weight', 0.3), 0.3)
        deception_accel_weight = get_param_value(bt_params.get('deception_accel_weight', 0.2), 0.2)
        positive_deception_threshold = get_param_value(bt_params.get('positive_deception_threshold', 0.3), 0.3)
        composite_positive_deception_score = (
            norm_deception_index_bipolar * deception_index_weight +
            norm_deception_slope_bipolar * deception_slope_weight +
            norm_deception_accel_bipolar * deception_accel_weight
        ) / (deception_index_weight + deception_slope_weight + deception_accel_weight)
        has_positive_deception = (composite_positive_deception_score > positive_deception_threshold)
        norm_context_modulator = utils.get_adaptive_mtf_normalized_score(context_modulator_raw, df_index, ascending=True, tf_weights=tf_weights, _parsed_tf_data=parsed_tf_data)
        dynamic_penalty_sensitivity = 1 + norm_context_modulator * context_modulator_sensitivity
        # --- Numba优化区域：penalty_factor ---
        penalty_factor_values = _numba_calculate_bull_trap_penalty_core(
            has_recent_sharp_drop.values,
            composite_positive_deception_score.values,
            has_positive_deception.values,
            deception_penalty_multiplier,
            dynamic_penalty_sensitivity.values
        )
        penalty_factor = pd.Series(penalty_factor_values, index=df_index, dtype=np.float32)
        # --- Numba优化区域结束 ---
        return penalty_factor


