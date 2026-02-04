import json
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional, Any, Tuple, Union

from strategies.trend_following.utils import (
    get_params_block, get_param_value, get_adaptive_mtf_normalized_score,
    is_limit_up, get_adaptive_mtf_normalized_bipolar_score,
    normalize_score, _robust_geometric_mean
)
from strategies.trend_following.intelligence.process.helper import ProcessIntelligenceHelper

def _weighted_sum_fusion(components: Dict[str, pd.Series], weights: Dict[str, Union[float, pd.Series]], index: pd.Index) -> pd.Series:
    """
    计算加权和，对0值进行鲁棒处理。
    如果某个组件的权重为0，则不计入总和。
    """
    if not components:
        return pd.Series(0.0, index=index, dtype=np.float32)
    weight_series_map = {}
    for k, w in weights.items():
        if isinstance(w, (int, float)):
            weight_series_map[k] = pd.Series(w, index=index, dtype=np.float32)
        elif isinstance(w, pd.Series): # 明确检查是否为 Series
            weight_series_map[k] = w.astype(np.float32)
        else:
            # 如果遇到非预期的类型（例如字符串），则跳过此权重，并打印警告
            print(f"  [警告] _weighted_sum_fusion: 权重 '{k}' 的类型为 {type(w)}，不是 float 或 pd.Series，将忽略此权重。")
            continue # 跳过此权重
    fused_score = pd.Series(0.0, index=index, dtype=np.float32)
    total_effective_weight = pd.Series(0.0, index=index, dtype=np.float32)
    # 只遍历那些成功添加到 weight_series_map 的组件
    for k, series in components.items():
        if k not in weight_series_map: # 如果权重被忽略，则跳过此组件
            continue
        series = series.astype(np.float32)
        current_weight = weight_series_map[k]
        non_zero_weight_mask = (current_weight.abs() > 1e-9)
        fused_score.loc[non_zero_weight_mask] += series.loc[non_zero_weight_mask] * current_weight.loc[non_zero_weight_mask]
        total_effective_weight.loc[non_zero_weight_mask] += current_weight.loc[non_zero_weight_mask]
    result = pd.Series(0.0, index=index, dtype=np.float32)
    non_zero_total_weight_mask = (total_effective_weight.abs() > 1e-9)
    result.loc[non_zero_total_weight_mask] = fused_score.loc[non_zero_total_weight_mask] / total_effective_weight.loc[non_zero_total_weight_mask]
    return result

class CalculatePriceMomentumDivergence:
    """
    计算价格-动量背离分数。 
    PROCESS_META_PRICE_VS_MOMENTUM_DIVERGENCE
    """
    def __init__(self, strategy_instance, helper_instance: ProcessIntelligenceHelper):
        self.strategy = strategy_instance
        self.helper = helper_instance

    def _print_debug_output_pmd(self, debug_output: Dict, probe_ts: pd.Timestamp, method_name: str, final_score: pd.Series):
        """V1.1 · 增加RDI调试信息输出"""
        if not debug_output:
            return
        if f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---" in debug_output:
            self.helper._print_debug_output({f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---": ""})
            self.helper._print_debug_output({f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算价势背离...": ""})
        if "原始信号值" in debug_output:
            self.helper._print_debug_output({f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---": ""})
            for key, value in debug_output["原始信号值"].items():
                if isinstance(value, dict):
                    self.helper._print_debug_output({f"        {key}:": ""})
                    for sub_key, sub_series in value.items():
                        val = sub_series.loc[probe_ts] if probe_ts in sub_series.index else np.nan
                        self.helper._print_debug_output({f"          {sub_key}: {val:.4f}": ""})
                else:
                    val = value.loc[probe_ts] if probe_ts in value.index else np.nan
                    self.helper._print_debug_output({f"        '{key}': {val:.4f}": ""})
        sections = [
            "融合价格方向", "融合动量方向", "基础背离分数", "量能确认分数",
            "主力/筹码确认分数", "背离质量分数", "情境调制器", "最终融合组件",
            "动态融合权重调整", "原始融合分数", "价格-动量RDI", "价格-主力RDI", "RDI调制器", "RDI调制后的分数", # 新增RDI相关部分
            "协同/冲突与最终分数"
        ]
        for section in sections:
            if section in debug_output:
                self.helper._print_debug_output({f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- {section} ---": ""})
                for key, series in debug_output[section].items():
                    if isinstance(series, pd.Series):
                        val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                        self.helper._print_debug_output({f"        {key}: {val:.4f}": ""})
                    elif isinstance(series, dict):
                        self.helper._print_debug_output({f"        {key}:": ""})
                        for sub_key, sub_value in series.items():
                            if isinstance(sub_value, pd.Series):
                                val = sub_value.loc[probe_ts] if probe_ts in sub_value.index else np.nan
                                self.helper._print_debug_output({f"          {sub_key}: {val:.4f}": ""})
                            else:
                                self.helper._print_debug_output({f"          {sub_key}: {sub_value}": ""})
                    else:
                        self.helper._print_debug_output({f"        {key}: {series}": ""})
        self.helper._print_debug_output({f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 价势背离诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}": ""})

    def _get_pmd_params(self, config: Dict) -> Dict:
        """V1.7 · 参数精简与替换版 (移除未提供信号，替换语义相近信号，新增RDI参数，修复RDI权重提取)"""
        params = get_param_value(config.get('price_momentum_divergence_params'), {})
        # 提取 rdi_params 配置块
        rdi_config = get_param_value(params.get('rdi_params'), {})
        # 默认的 rdi_period_weights
        default_rdi_period_weights = {"1": 0.4, "5": 0.3, "13": 0.2, "21": 0.1}
        # 从配置中获取 rdi_period_weights，并确保只包含数字键的权重
        configured_rdi_period_weights = get_param_value(rdi_config.get('rdi_period_weights'), default_rdi_period_weights)
        # 过滤掉非数字键（例如 "description"）
        filtered_rdi_period_weights = {k: v for k, v in configured_rdi_period_weights.items() if str(k).isdigit()}
        return {
            "price_components_weights": get_param_value(params.get('price_components_weights'), {"close_D": 0.6, "upward_efficiency": 0.2, "price_momentum_quality": 0.2}),
            "momentum_components_weights": get_param_value(params.get('momentum_components_weights'), {"MACDh_13_34_8_D": 0.5, "RSI_13_D": 0.3, "ROC_13_D": 0.2, "momentum_quality": 0.2}),
            "mtf_slope_weights": get_param_value(params.get('mtf_slope_weights'), {"5": 0.4, "13": 0.3, "21": 0.2, "34": 0.1}),
            "mtf_accel_weights": get_param_value(params.get('mtf_accel_weights'), {"5": 0.6, "13": 0.4}),
            "volume_confirmation_weights": get_param_value(params.get('volume_confirmation_weights'), {"volume_slope": 0.5, "volume_burst": 0.2, "volume_atrophy": 0.3, "constructive_turnover": 0.1, "volume_structure_skew_inverted": 0.1, "volume_profile_entropy_inverted": 0.05}),
            "dynamic_volume_confirmation_modulators": get_param_value(params.get('dynamic_volume_confirmation_modulators'), {"enabled": False}),
            "main_force_confirmation_weights": get_param_value(params.get('main_force_confirmation_weights'), {"mf_net_flow_slope": 0.4, "deception_index": 0.2, "distribution_intent": 0.2, "covert_accumulation": 0.1, "chip_divergence": 0.1, "main_force_conviction": 0.1, "chip_health": 0.1, "mf_buy_ofi_positive": 0.05, "order_book_imbalance_positive": 0.05, "micro_price_impact_asymmetry_positive": 0.05, "main_force_slippage_inverted": 0.05, "winner_concentration_positive": 0.05, "loser_pain_positive": 0.05, "smart_money_inst_net_buy": 0.05, "intraday_vwap_div_index_inverted": 0.05}),
            "dynamic_main_force_confirmation_modulators": get_param_value(params.get('dynamic_main_force_confirmation_modulators'), {"enabled": False}),
            "context_modulator_weights": get_param_value(params.get('context_modulator_weights'), {"volatility_inverse": 0.3, "trend_strength_inverse": 0.2, "sentiment_neutrality": 0.2, "liquidity_tide_calm": 0.15, "market_constitution_neutrality": 0.15, "mean_reversion_inverse": 0.05, "trend_alignment": 0.05, "theme_hotness": 0.05, "retail_panic_surrender_inverted": 0.05}),
            "divergence_quality_weights": get_param_value(params.get('divergence_quality_weights'), {"duration": 0.4, "depth": 0.3, "stability": 0.15, "chip_potential": 0.15, "divergence_purity": 0.1}),
            "final_fusion_exponent": get_param_value(params.get('final_fusion_exponent'), 1.5),
            "synergy_threshold": get_param_value(params.get('synergy_threshold'), 0.6),
            "synergy_bonus_factor": get_param_value(params.get('synergy_bonus_factor'), 0.1),
            "conflict_penalty_factor": get_param_value(params.get('conflict_penalty_factor'), 0.15),
            "dynamic_fusion_weights_params": get_param_value(params.get('dynamic_fusion_weights_params'), {"enabled": False}),
            "dynamic_exponent_params": get_param_value(params.get('dynamic_exponent_params'), {"enabled": False, "modulator_signal": "VOLATILITY_INSTABILITY_INDEX_21d_D", "sensitivity": 0.5, "base_exponent": 1.5, "min_exponent": 1.0, "max_exponent": 2.0}),
            "interaction_terms_weights": get_param_value(params.get('interaction_terms_weights'), {"price_momentum_main_force_synergy": 0.1}),
            "rdi_params": { # 新增RDI参数
                "enabled": get_param_value(rdi_config.get('enabled'), False),
                "rdi_periods": get_param_value(rdi_config.get('rdi_periods'), [1, 5, 13, 21]),
                "resonance_reward_factor": get_param_value(rdi_config.get('resonance_reward_factor'), 0.1),
                "divergence_penalty_factor": get_param_value(rdi_config.get('divergence_penalty_factor'), 0.15),
                "inflection_reward_factor": get_param_value(rdi_config.get('inflection_reward_factor'), 0.05),
                "rdi_period_weights": filtered_rdi_period_weights, # 使用过滤后的权重
                "rdi_modulator_weight": get_param_value(rdi_config.get('rdi_modulator_weight'), 0.1)
            }
        }

    def _validate_pmd_signals(self, df: pd.DataFrame, pmd_params: Dict, method_name: str) -> bool:
        """V1.6 · 信号精简与替换版 (移除未提供信号，替换语义相近信号，新增加速度信号和结构张力指数校验)"""
        mtf_slope_weights = pmd_params['mtf_slope_weights']
        mtf_accel_weights = pmd_params['mtf_accel_weights']
        valid_mtf_periods = [p_str for p_str in mtf_slope_weights.keys() if p_str.isdigit()]
        required_signals = [
            *[f'SLOPE_{p}_close_D' for p in valid_mtf_periods],
            *[f'SLOPE_{p}_MACDh_13_34_8_D' for p in valid_mtf_periods],
            *[f'SLOPE_{p}_RSI_13_D' for p in valid_mtf_periods],
            *[f'SLOPE_{p}_ROC_13_D' for p in valid_mtf_periods],
            *[f'SLOPE_{p}_volume_D' for p in valid_mtf_periods],
            'volume_burstiness_index_D',
            *[f'SLOPE_{p}_main_force_net_flow_calibrated_D' for p in valid_mtf_periods],
            'deception_index_D',
            'VOLATILITY_INSTABILITY_INDEX_21d_D', 'ADX_14_D', 'market_sentiment_score_D',
            'constructive_turnover_ratio_D',
            'volume_structure_skew_D',
            'main_force_conviction_index_D',
            'chip_health_score_D',
            'volume_profile_entropy_D',
            'upward_impulse_strength_D', 'downward_impulse_strength_D',
            'main_force_buy_ofi_D', 'main_force_sell_ofi_D',
            'order_book_imbalance_D', 'micro_price_impact_asymmetry_D',
            'main_force_slippage_index_D',
            'winner_concentration_90pct_D', 'loser_concentration_90pct_D',
            'winner_profit_margin_avg_D', 'loser_loss_margin_avg_D',
            'mean_reversion_frequency_D', 'trend_alignment_index_D',
            'SMART_MONEY_INST_NET_BUY_D',
            'THEME_HOTNESS_SCORE_D',
            'intraday_vwap_div_index_D',
            'retail_panic_surrender_index_D',
            'structural_tension_index_D' # 新增结构张力指数校验
        ]
        for p_str in mtf_accel_weights.keys():
            p = int(p_str)
            required_signals.append(f'ACCEL_{p}_close_D') # 新增加速度信号校验
            required_signals.append(f'ACCEL_{p}_MACDh_13_34_8_D')
            required_signals.append(f'ACCEL_{p}_RSI_13_D')
            required_signals.append(f'ACCEL_{p}_ROC_13_D')
            required_signals.append(f'ACCEL_{p}_volume_D')
            required_signals.append(f'ACCEL_{p}_main_force_net_flow_calibrated_D')
        return self.helper._validate_required_signals(df, required_signals, method_name)

    def _get_pmd_raw_data(self, df: pd.DataFrame, pmd_params: Dict, method_name: str) -> Dict[str, pd.Series]:
        """V1.7 · 原始数据精简与替换版 (修复原始信号依赖顺序，移除原子信号，替换为复合计算或安全获取原始数据)"""
        mtf_slope_weights = pmd_params['mtf_slope_weights']
        valid_mtf_periods = [p_str for p_str in mtf_slope_weights.keys() if p_str.isdigit()]
        raw_data = {}
        # --- 1. 首先获取所有直接的原始信号 (不依赖其他 raw_data 中的复合信号) ---
        raw_data['price_slopes_raw'] = {p: self.helper._get_safe_series(df, f'SLOPE_{p}_close_D', 0.0, method_name=method_name) for p in valid_mtf_periods}
        raw_data['macdh_slopes_raw'] = {p: self.helper._get_safe_series(df, f'SLOPE_{p}_MACDh_13_34_8_D', 0.0, method_name=method_name) for p in valid_mtf_periods}
        raw_data['rsi_slopes_raw'] = {p: self.helper._get_safe_series(df, f'SLOPE_{p}_RSI_13_D', 0.0, method_name=method_name) for p in valid_mtf_periods}
        raw_data['roc_slopes_raw'] = {p: self.helper._get_safe_series(df, f'SLOPE_{p}_ROC_13_D', 0.0, method_name=method_name) for p in valid_mtf_periods}
        raw_data['volume_slopes_raw'] = {p: self.helper._get_safe_series(df, f'SLOPE_{p}_volume_D', 0.0, method_name=method_name) for p in valid_mtf_periods}
        raw_data['mf_net_flow_slopes_raw'] = {p: self.helper._get_safe_series(df, f'SLOPE_{p}_main_force_net_flow_calibrated_D', 0.0, method_name=method_name) for p in valid_mtf_periods}
        raw_data['volume_burstiness_raw'] = self.helper._get_safe_series(df, 'volume_burstiness_index_D', 0.0, method_name=method_name)
        raw_data['deception_index_raw'] = self.helper._get_safe_series(df, 'deception_index_D', 0.0, method_name=method_name)
        raw_data['volatility_instability_raw'] = self.helper._get_safe_series(df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', 0.0, method_name=method_name)
        raw_data['adx_raw'] = self.helper._get_safe_series(df, 'ADX_14_D', 0.0, method_name=method_name)
        raw_data['market_sentiment_raw'] = self.helper._get_safe_series(df, 'market_sentiment_score_D', 0.0, method_name=method_name)
        raw_data['constructive_turnover_raw'] = self.helper._get_safe_series(df, 'constructive_turnover_ratio_D', 0.0, method_name=method_name)
        raw_data['volume_structure_skew_raw'] = self.helper._get_safe_series(df, 'volume_structure_skew_D', 0.0, method_name=method_name)
        raw_data['main_force_conviction_raw'] = self.helper._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name=method_name)
        raw_data['chip_health_raw'] = self.helper._get_safe_series(df, 'chip_health_score_D', 0.0, method_name=method_name)
        raw_data['volume_profile_entropy_raw'] = self.helper._get_safe_series(df, 'volume_profile_entropy_D', 0.0, method_name=method_name)
        raw_data['upward_impulse_strength_raw'] = self.helper._get_safe_series(df, 'upward_impulse_strength_D', 0.0, method_name=method_name)
        raw_data['downward_impulse_strength_raw'] = self.helper._get_safe_series(df, 'downward_impulse_strength_D', 0.0, method_name=method_name)
        raw_data['main_force_buy_ofi_raw'] = self.helper._get_safe_series(df, 'main_force_buy_ofi_D', 0.0, method_name=method_name)
        raw_data['main_force_sell_ofi_raw'] = self.helper._get_safe_series(df, 'main_force_sell_ofi_D', 0.0, method_name=method_name)
        raw_data['order_book_imbalance_raw'] = self.helper._get_safe_series(df, 'order_book_imbalance_D', 0.0, method_name=method_name)
        raw_data['micro_price_impact_asymmetry_raw'] = self.helper._get_safe_series(df, 'micro_price_impact_asymmetry_D', 0.0, method_name=method_name)
        raw_data['main_force_slippage_index_raw'] = self.helper._get_safe_series(df, 'main_force_slippage_index_D', 0.0, method_name=method_name)
        raw_data['winner_concentration_90pct_raw'] = self.helper._get_safe_series(df, 'winner_concentration_90pct_D', 0.0, method_name=method_name)
        raw_data['loser_concentration_90pct_raw'] = self.helper._get_safe_series(df, 'loser_concentration_90pct_D', 0.0, method_name=method_name)
        raw_data['winner_profit_margin_avg_raw'] = self.helper._get_safe_series(df, 'winner_profit_margin_avg_D', 0.0, method_name=method_name)
        raw_data['loser_loss_margin_avg_raw'] = self.helper._get_safe_series(df, 'loser_loss_margin_avg_D', 0.0, method_name=method_name)
        raw_data['mean_reversion_frequency_raw'] = self.helper._get_safe_series(df, 'mean_reversion_frequency_D', 0.0, method_name=method_name)
        raw_data['trend_alignment_index_raw'] = self.helper._get_safe_series(df, 'trend_alignment_index_D', 0.0, method_name=method_name)
        raw_data['smart_money_inst_net_buy_raw'] = self.helper._get_safe_series(df, 'SMART_MONEY_INST_NET_BUY_D', 0.0, method_name=method_name)
        raw_data['theme_hotness_raw'] = self.helper._get_safe_series(df, 'THEME_HOTNESS_SCORE_D', 0.0, method_name=method_name)
        raw_data['intraday_vwap_div_index_raw'] = self.helper._get_safe_series(df, 'intraday_vwap_div_index_D', 0.0, method_name=method_name)
        raw_data['retail_panic_surrender_index_raw'] = self.helper._get_safe_series(df, 'retail_panic_surrender_index_D', 0.0, method_name=method_name)
        raw_data['structural_tension_index_D'] = self.helper._get_safe_series(df, 'structural_tension_index_D', 0.0, method_name=method_name)
        # --- 2. 然后计算依赖于上述原始信号的复合分数 ---
        raw_data['volume_atrophy_score'] = self._calculate_composite_volume_atrophy_score(df, df.index, raw_data, pmd_params, method_name)
        raw_data['distribution_intent_score'] = self._calculate_composite_distribution_intent_score(df, df.index, raw_data, pmd_params, method_name)
        raw_data['covert_accumulation_score'] = self._calculate_composite_covert_accumulation_score(df, df.index, raw_data, pmd_params, method_name)
        raw_data['chip_divergence_score'] = self._calculate_composite_chip_divergence_score(df, df.index, raw_data, pmd_params, method_name)
        raw_data['upward_efficiency_score'] = self._calculate_composite_upward_efficiency_score(df, df.index, raw_data, pmd_params, method_name)
        raw_data['price_upward_momentum_score'] = self._calculate_composite_price_upward_momentum_score(df, df.index, raw_data, pmd_params, method_name)
        raw_data['price_downward_momentum_score'] = self._calculate_composite_price_downward_momentum_score(df, df.index, raw_data, pmd_params, method_name)
        raw_data['momentum_quality_score'] = self._calculate_composite_momentum_quality_score(df, df.index, raw_data, pmd_params, method_name)
        raw_data['stability_score'] = self._calculate_composite_stability_score(df, df.index, raw_data, pmd_params, method_name)
        raw_data['chip_historical_potential_score'] = self._calculate_composite_chip_historical_potential_score(df, df.index, raw_data, pmd_params, method_name)
        raw_data['liquidity_tide_score'] = self._calculate_composite_liquidity_tide_score(df, df.index, raw_data, pmd_params, method_name)
        raw_data['market_constitution_score'] = self._calculate_composite_market_constitution_score(df, df.index, raw_data, pmd_params, method_name)
        raw_data['market_tension_score'] = self._calculate_composite_market_tension_score(df, df.index, raw_data, pmd_params, method_name)
        return raw_data

    def _calculate_fused_price_direction(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> Tuple[pd.Series, Dict]:
        """V1.3 · 价格动量品质增强版 (价格动量品质分数归一化，使用复合分数)"""
        mtf_slope_weights = pmd_params['mtf_slope_weights']
        price_components_weights = pmd_params['price_components_weights']
        fused_price_direction_base = self.helper._get_mtf_slope_score(df, 'close_D', mtf_slope_weights, df_index, method_name, bipolar=True)
        # 优化价格动量品质：结合上涨纯度和上涨冲动强度
        # upward_efficiency_score, price_upward_momentum_score, price_downward_momentum_score 现在是复合分数
        bullish_price_momentum_quality_raw = (raw_data['upward_efficiency_score'] * raw_data['price_upward_momentum_score']).pow(0.5)
        bearish_price_momentum_quality_raw = raw_data['price_downward_momentum_score']
        # 对原始品质分数进行归一化
        bullish_price_momentum_quality_norm = self.helper._normalize_series(bullish_price_momentum_quality_raw, df_index, ascending=True)
        bearish_price_momentum_quality_norm = self.helper._normalize_series(bearish_price_momentum_quality_raw, df_index, ascending=True)
        # 根据价格方向决定使用哪个品质分数，并确保其在 [-1, 1] 范围内
        price_momentum_quality_score = bullish_price_momentum_quality_norm.where(fused_price_direction_base > 0, -bearish_price_momentum_quality_norm)
        # 最终确保在 [-1, 1] 范围内
        price_momentum_quality_score = price_momentum_quality_score.clip(-1, 1)
        fused_price_direction_components = {
            "close_D": fused_price_direction_base,
            "upward_efficiency": self.helper._normalize_series(raw_data['upward_efficiency_score'], df_index, ascending=True),
            "price_momentum_quality": price_momentum_quality_score
        }
        fused_price_direction = _robust_geometric_mean(fused_price_direction_components, price_components_weights, df_index)
        debug_values = {
            "fused_price_direction_base": fused_price_direction_base,
            "bullish_price_momentum_quality_raw": bullish_price_momentum_quality_raw,
            "bearish_price_momentum_quality_raw": bearish_price_momentum_quality_raw,
            "bullish_price_momentum_quality_norm": bullish_price_momentum_quality_norm,
            "bearish_price_momentum_quality_norm": bearish_price_momentum_quality_norm,
            "price_momentum_quality_score": price_momentum_quality_score,
            "fused_price_direction": fused_price_direction
        }
        return fused_price_direction, debug_values

    def _calculate_fused_momentum_direction(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> Tuple[pd.Series, Dict]:
        """V1.1 · 动量方向融合 (使用复合动量品质分数)"""
        mtf_slope_weights = pmd_params['mtf_slope_weights']
        momentum_components_weights = pmd_params['momentum_components_weights']
        fused_macdh_direction = self.helper._get_mtf_slope_score(df, 'MACDh_13_34_8_D', mtf_slope_weights, df_index, method_name, bipolar=True)
        fused_rsi_direction = self.helper._get_mtf_slope_score(df, 'RSI_13_D', mtf_slope_weights, df_index, method_name, bipolar=True)
        fused_roc_direction = self.helper._get_mtf_slope_score(df, 'ROC_13_D', mtf_slope_weights, df_index, method_name, bipolar=True)
        fused_momentum_direction_components = {
            "MACDh_13_34_8_D": fused_macdh_direction,
            "RSI_13_D": fused_rsi_direction,
            "ROC_13_D": fused_roc_direction,
            "momentum_quality": raw_data['momentum_quality_score'] # 使用复合分数
        }
        momentum_components_weights_extended = momentum_components_weights.copy()
        momentum_components_weights_extended["momentum_quality"] = get_param_value(pmd_params.get('momentum_components_weights', {}).get("momentum_quality"), 0.2)
        fused_momentum_direction = _robust_geometric_mean(fused_momentum_direction_components, momentum_components_weights_extended, df_index)
        debug_values = {
            "fused_macdh_direction": fused_macdh_direction,
            "fused_rsi_direction": fused_rsi_direction,
            "fused_roc_direction": fused_roc_direction,
            "fused_momentum_direction": fused_momentum_direction
        }
        return fused_momentum_direction, debug_values

    def _calculate_volume_confirmation_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, base_divergence_score: pd.Series, method_name: str) -> Tuple[pd.Series, Dict]:
        """V1.1 · 量能结构增强版"""
        mtf_slope_weights = pmd_params['mtf_slope_weights']
        volume_confirmation_weights = pmd_params['volume_confirmation_weights']
        dynamic_volume_confirmation_modulators = pmd_params['dynamic_volume_confirmation_modulators']
        fused_volume_slope = self.helper._get_mtf_slope_score(df, 'volume_D', mtf_slope_weights, df_index, method_name, bipolar=True)
        volume_burst_norm = self.helper._normalize_series(raw_data['volume_burstiness_raw'], df_index, ascending=True)
        volume_atrophy_norm = self.helper._normalize_series(raw_data['volume_atrophy_score'], df_index, ascending=True)
        constructive_turnover_norm = self.helper._normalize_series(raw_data['constructive_turnover_raw'], df_index, ascending=True)
        volume_structure_skew_inverted_norm = self.helper._normalize_series(raw_data['volume_structure_skew_raw'].abs(), df_index, ascending=False)
        # 新增：量能轮廓熵反向归一化，熵越低（结构越清晰）分数越高
        volume_profile_entropy_inverted_norm = self.helper._normalize_series(raw_data['volume_profile_entropy_raw'], df_index, ascending=False)
        current_volume_confirmation_weights = volume_confirmation_weights.copy()
        if get_param_value(dynamic_volume_confirmation_modulators.get('enabled'), False):
            modulator_signal_raw = self.helper._get_atomic_score(df, dynamic_volume_confirmation_modulators['modulator_signal'], 0.0)
            modulator_signal = self.helper._normalize_series(modulator_signal_raw, df_index, bipolar=True)
            sensitivity = dynamic_volume_confirmation_modulators['sensitivity']
            min_factor = dynamic_volume_confirmation_modulators['min_factor']
            max_factor = dynamic_volume_confirmation_modulators['max_factor']
            modulator_factor = (1 + modulator_signal * sensitivity).clip(min_factor, max_factor)
            for k in current_volume_confirmation_weights:
                current_volume_confirmation_weights[k] = current_volume_confirmation_weights[k] * modulator_factor
        top_vol_conf_components = {
            "volume_slope_negative": fused_volume_slope.clip(upper=0).abs(),
            "volume_burst": volume_burst_norm,
            "constructive_turnover": constructive_turnover_norm,
            "volume_structure_skew_inverted": volume_structure_skew_inverted_norm,
            "volume_profile_entropy_inverted": volume_profile_entropy_inverted_norm # 新增
        }
        bottom_vol_conf_components = {
            "volume_slope_positive": fused_volume_slope.clip(lower=0),
            "volume_atrophy": volume_atrophy_norm,
            "constructive_turnover": constructive_turnover_norm,
            "volume_structure_skew_inverted": volume_structure_skew_inverted_norm,
            "volume_profile_entropy_inverted": volume_profile_entropy_inverted_norm # 新增
        }
        top_vol_conf = _robust_geometric_mean(top_vol_conf_components, current_volume_confirmation_weights, df_index)
        bottom_vol_conf = _robust_geometric_mean(bottom_vol_conf_components, current_volume_confirmation_weights, df_index)
        volume_confirmation_score = pd.Series([
            top_vol_conf.loc[idx] if x > 0 else (-bottom_vol_conf.loc[idx] if x < 0 else 0)
            for idx, x in base_divergence_score.items()
        ], index=df_index, dtype=np.float32)
        debug_values = {
            "fused_volume_slope": fused_volume_slope,
            "volume_burst_norm": volume_burst_norm,
            "volume_atrophy_norm": volume_atrophy_norm,
            "constructive_turnover_norm": constructive_turnover_norm,
            "volume_structure_skew_inverted_norm": volume_structure_skew_inverted_norm,
            "volume_profile_entropy_inverted_norm": volume_profile_entropy_inverted_norm, # 新增
            "top_vol_conf": top_vol_conf,
            "bottom_vol_conf": bottom_vol_conf,
            "volume_confirmation_score": volume_confirmation_score
        }
        return volume_confirmation_score, debug_values

    def _calculate_main_force_confirmation_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, base_divergence_score: pd.Series, method_name: str) -> Tuple[pd.Series, pd.Series, Dict]:
        """V1.7 · 主力微观与筹码结构、聪明钱及微观执行力增强版 (移除累积资金流、大单不平衡，使用复合分数)"""
        mtf_slope_weights = pmd_params['mtf_slope_weights']
        main_force_confirmation_weights = pmd_params['main_force_confirmation_weights']
        dynamic_main_force_confirmation_modulators = pmd_params['dynamic_main_force_confirmation_modulators']
        fused_mf_net_flow_slope = self.helper._get_mtf_slope_score(df, 'main_force_net_flow_calibrated_D', mtf_slope_weights, df_index, method_name, bipolar=True)
        deception_index_norm = self.helper._normalize_series(raw_data.get('deception_index_raw', pd.Series(0.0, index=df_index)), df_index, bipolar=True)
        distribution_intent_norm = self.helper._normalize_series(raw_data.get('distribution_intent_score', pd.Series(0.0, index=df_index)), df_index, ascending=True)
        covert_accumulation_norm = self.helper._normalize_series(raw_data.get('covert_accumulation_score', pd.Series(0.0, index=df_index)), df_index, ascending=True)
        chip_divergence_norm = self.helper._normalize_series(raw_data.get('chip_divergence_score', pd.Series(0.0, index=df_index)), df_index, bipolar=True)
        main_force_conviction_norm = self.helper._normalize_series(raw_data.get('main_force_conviction_raw', pd.Series(0.0, index=df_index)), df_index, bipolar=True)
        chip_health_norm = self.helper._normalize_series(raw_data.get('chip_health_raw', pd.Series(0.0, index=df_index)), df_index, bipolar=False)
        mf_buy_ofi_positive = self.helper._normalize_series(raw_data.get('main_force_buy_ofi_raw', pd.Series(0.0, index=df_index)), df_index, ascending=True)
        mf_sell_ofi_negative = self.helper._normalize_series(raw_data.get('main_force_sell_ofi_raw', pd.Series(0.0, index=df_index)), df_index, ascending=False)
        order_book_imbalance_positive = self.helper._normalize_series(raw_data.get('order_book_imbalance_raw', pd.Series(0.0, index=df_index)).clip(lower=0), df_index, ascending=True)
        micro_price_impact_asymmetry_positive = self.helper._normalize_series(raw_data.get('micro_price_impact_asymmetry_raw', pd.Series(0.0, index=df_index)).clip(lower=0), df_index, ascending=True)
        main_force_slippage_inverted = self.helper._normalize_series(raw_data.get('main_force_slippage_index_raw', pd.Series(0.0, index=df_index)), df_index, ascending=False)
        winner_concentration_positive = self.helper._normalize_series(raw_data.get('winner_concentration_90pct_raw', pd.Series(0.0, index=df_index)), df_index, ascending=True)
        loser_pain_positive = self.helper._normalize_series(raw_data.get('loser_loss_margin_avg_raw', pd.Series(0.0, index=df_index)), df_index, ascending=True) # 使用 loser_loss_margin_avg_raw 作为 loser_pain_positive
        smart_money_inst_net_buy_norm = self.helper._normalize_series(raw_data.get('smart_money_inst_net_buy_raw', pd.Series(0.0, index=df_index)), df_index, ascending=True)
        intraday_vwap_div_index_inverted_norm = self.helper._normalize_series(raw_data.get('intraday_vwap_div_index_raw', pd.Series(0.0, index=df_index)).abs(), df_index, ascending=False)
        current_main_force_confirmation_weights = main_force_confirmation_weights.copy()
        if get_param_value(dynamic_main_force_confirmation_modulators.get('enabled'), False):
            modulator_signal_raw = raw_data.get('market_tension_score', pd.Series(0.0, index=df_index)) # 使用复合分数
            modulator_signal = self.helper._normalize_series(modulator_signal_raw, df_index, bipolar=True)
            sensitivity = dynamic_main_force_confirmation_modulators['sensitivity']
            min_factor = dynamic_main_force_confirmation_modulators['min_factor']
            max_factor = dynamic_main_force_confirmation_modulators['max_factor']
            modulator_factor = (1 + modulator_signal * sensitivity).clip(min_factor, max_factor)
            for k in current_main_force_confirmation_weights:
                current_main_force_confirmation_weights[k] = current_main_force_confirmation_weights[k] * modulator_factor
        top_mf_conf_components = {
            "mf_net_flow_slope_negative": fused_mf_net_flow_slope.clip(upper=0).abs(),
            "deception_index_positive": deception_index_norm.clip(lower=0),
            "distribution_intent": distribution_intent_norm,
            "chip_divergence_positive": chip_divergence_norm.clip(lower=0),
            "main_force_conviction": main_force_conviction_norm.clip(lower=0),
            "chip_health": chip_health_norm,
            "mf_buy_ofi_positive": mf_buy_ofi_positive,
            "order_book_imbalance_positive": order_book_imbalance_positive,
            "micro_price_impact_asymmetry_positive": micro_price_impact_asymmetry_positive,
            "main_force_slippage_inverted": main_force_slippage_inverted,
            "winner_concentration_positive": winner_concentration_positive,
            "loser_pain_positive": loser_pain_positive,
            "smart_money_inst_net_buy": smart_money_inst_net_buy_norm,
            "intraday_vwap_div_index_inverted": intraday_vwap_div_index_inverted_norm
        }
        bottom_mf_conf_components = {
            "mf_net_flow_slope_positive": fused_mf_net_flow_slope.clip(lower=0),
            "deception_index_negative": deception_index_norm.clip(upper=0).abs(),
            "covert_accumulation": covert_accumulation_norm,
            "chip_divergence_negative": chip_divergence_norm.clip(upper=0).abs(),
            "main_force_conviction": main_force_conviction_norm.clip(upper=0).abs(),
            "chip_health": chip_health_norm,
            "mf_sell_ofi_negative": mf_sell_ofi_negative,
            "order_book_imbalance_negative": self.helper._normalize_series(raw_data.get('order_book_imbalance_raw', pd.Series(0.0, index=df_index)).clip(upper=0).abs(), df_index, ascending=True),
            "micro_price_impact_asymmetry_negative": self.helper._normalize_series(raw_data.get('micro_price_impact_asymmetry_raw', pd.Series(0.0, index=df_index)).clip(upper=0).abs(), df_index, ascending=True),
            "main_force_slippage_positive": self.helper._normalize_series(raw_data.get('main_force_slippage_index_raw', pd.Series(0.0, index=df_index)), df_index, ascending=True),
            "loser_concentration_positive": self.helper._normalize_series(raw_data.get('loser_concentration_90pct_raw', pd.Series(0.0, index=df_index)), df_index, ascending=True),
            "winner_profit_margin_high": self.helper._normalize_series(raw_data.get('winner_profit_margin_avg_raw', pd.Series(0.0, index=df_index)), df_index, ascending=True),
            "smart_money_inst_net_buy": smart_money_inst_net_buy_norm,
            "intraday_vwap_div_index_inverted": intraday_vwap_div_index_inverted_norm
        }
        top_mf_conf = _robust_geometric_mean(top_mf_conf_components, current_main_force_confirmation_weights, df_index)
        bottom_mf_conf = _robust_geometric_mean(bottom_mf_conf_components, current_main_force_confirmation_weights, df_index)
        main_force_confirmation_score = pd.Series([
            top_mf_conf.loc[idx] if x > 0 else (-bottom_mf_conf.loc[idx] if x < 0 else 0)
            for idx, x in base_divergence_score.items()
        ], index=df_index, dtype=np.float32)
        debug_values = {
            "fused_mf_net_flow_slope": fused_mf_net_flow_slope,
            "deception_index_norm": deception_index_norm,
            "distribution_intent_norm": distribution_intent_norm,
            "covert_accumulation_norm": covert_accumulation_norm,
            "chip_divergence_norm": chip_divergence_norm,
            "main_force_conviction_norm": main_force_conviction_norm,
            "chip_health_norm": chip_health_norm,
            "mf_buy_ofi_positive": mf_buy_ofi_positive,
            "mf_sell_ofi_negative": mf_sell_ofi_negative,
            "order_book_imbalance_positive": order_book_imbalance_positive,
            "micro_price_impact_asymmetry_positive": micro_price_impact_asymmetry_positive,
            "main_force_slippage_inverted": main_force_slippage_inverted,
            "winner_concentration_positive": winner_concentration_positive,
            "loser_pain_positive": loser_pain_positive,
            "smart_money_inst_net_buy_norm": smart_money_inst_net_buy_norm,
            "intraday_vwap_div_index_inverted_norm": intraday_vwap_div_index_inverted_norm,
            "top_mf_conf": top_mf_conf,
            "bottom_mf_conf": bottom_mf_conf,
            "main_force_confirmation_score": main_force_confirmation_score
        }
        return main_force_confirmation_score, fused_mf_net_flow_slope, debug_values

    def _calculate_divergence_quality_score(self, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, base_divergence_score: pd.Series, fused_price_direction: pd.Series, fused_momentum_direction: pd.Series) -> Tuple[pd.Series, Dict]:
        """V1.3 · 背离纯度增强版 (移除筹码分布熵)"""
        divergence_quality_weights = pmd_params['divergence_quality_weights']
        is_top_divergence_bool = (base_divergence_score > 0.1)
        is_bottom_divergence_bool = (base_divergence_score < -0.1)
        top_divergence_duration = is_top_divergence_bool.astype(int).rolling(window=5, min_periods=1).sum()
        bottom_divergence_duration = is_bottom_divergence_bool.astype(int).rolling(window=5, min_periods=1).sum()
        top_divergence_duration_norm = (top_divergence_duration / 5).clip(0,1)
        bottom_divergence_duration_norm = (bottom_divergence_duration / 5).clip(0,1)
        divergence_depth_norm = base_divergence_score.abs()
        stability_norm = self.helper._normalize_series(raw_data['stability_score'], df_index, bipolar=False)
        chip_potential_norm = self.helper._normalize_series(raw_data['chip_historical_potential_score'], df_index, bipolar=False)
        total_movement_magnitude = (fused_price_direction.abs() + fused_momentum_direction.abs()).replace(0, 1e-9)
        divergence_alignment = (1 - (fused_price_direction + fused_momentum_direction).abs() / total_movement_magnitude).fillna(0)
        divergence_purity_score = (total_movement_magnitude.clip(0,1) * divergence_alignment).pow(0.5)
        # 移除 chip_distribution_entropy_inverted_norm
        # chip_distribution_entropy_inverted_norm = self.helper._normalize_series(raw_data.get('chip_distribution_entropy_raw', pd.Series(0.0, index=df_index)), df_index, ascending=False)
        divergence_quality_score = pd.Series([
            (_robust_geometric_mean(
                {"duration": pd.Series(top_divergence_duration_norm.loc[idx], index=[idx]),
                 "depth": pd.Series(divergence_depth_norm.loc[idx], index=[idx]),
                 "stability": pd.Series(stability_norm.loc[idx], index=[idx]),
                 "chip_potential": pd.Series(chip_potential_norm.loc[idx], index=[idx]),
                 "divergence_purity": pd.Series(divergence_purity_score.loc[idx], index=[idx])}, # 移除 chip_distribution_entropy_inverted
                divergence_quality_weights,
                pd.Index([idx])
            ).iloc[0] if x > 0 else
             (_robust_geometric_mean(
                 {"duration": pd.Series(bottom_divergence_duration_norm.loc[idx], index=[idx]),
                  "depth": pd.Series(divergence_depth_norm.loc[idx], index=[idx]),
                  "stability": pd.Series(stability_norm.loc[idx], index=[idx]),
                  "chip_potential": pd.Series(chip_potential_norm.loc[idx], index=[idx]),
                  "divergence_purity": pd.Series(divergence_purity_score.loc[idx], index=[idx])}, # 移除 chip_distribution_entropy_inverted
                 divergence_quality_weights,
                 pd.Index([idx])
             ).iloc[0] if x < 0 else 0))
            for idx, x in base_divergence_score.items()
        ], index=df_index, dtype=np.float32)
        debug_values = {
            "divergence_purity_score": divergence_purity_score,
            # 移除 chip_distribution_entropy_inverted_norm
            # "chip_distribution_entropy_inverted_norm": chip_distribution_entropy_inverted_norm,
            "divergence_quality_score": divergence_quality_score
        }
        return divergence_quality_score, debug_values

    def _calculate_context_modulator(self, df_index: pd.Index, raw_data: Dict, pmd_params: Dict) -> Tuple[pd.Series, Dict]:
        """V1.4 · 市场机制感知版 (替换散户恐慌信号，动态权重调整)"""
        context_modulator_weights = pmd_params['context_modulator_weights']
        volatility_instability_norm_inverted = self.helper._normalize_series(raw_data['volatility_instability_raw'], df_index, ascending=False)
        adx_norm_inverted = self.helper._normalize_series(raw_data['adx_raw'], df_index, ascending=False)
        market_sentiment_norm_bipolar = self.helper._normalize_series(raw_data['market_sentiment_raw'], df_index, bipolar=True)
        liquidity_tide_calm_norm = self.helper._normalize_series(raw_data['liquidity_tide_score'].abs(), df_index, ascending=False)
        market_constitution_neutrality_norm = 1 - self.helper._normalize_series(raw_data['market_constitution_score'].abs(), df_index, ascending=True)
        mean_reversion_inverse_norm = self.helper._normalize_series(raw_data['mean_reversion_frequency_raw'], df_index, ascending=False)
        trend_alignment_norm = self.helper._normalize_series(raw_data['trend_alignment_index_raw'], df_index, ascending=True)
        theme_hotness_norm = self.helper._normalize_series(raw_data.get('theme_hotness_raw', pd.Series(0.0, index=df_index)), df_index, ascending=True)
        retail_panic_surrender_inverted_norm = self.helper._normalize_series(raw_data.get('retail_panic_surrender_index_raw', pd.Series(0.0, index=df_index)), df_index, ascending=False) # 替换 retail_panic_inverted_norm
        context_modulator_components = {
            "volatility_inverse": volatility_instability_norm_inverted,
            "trend_strength_inverse": adx_norm_inverted,
            "sentiment_neutrality": 1 - market_sentiment_norm_bipolar.abs(),
            "liquidity_tide_calm": liquidity_tide_calm_norm,
            "market_constitution_neutrality": market_constitution_neutrality_norm,
            "mean_reversion_inverse": mean_reversion_inverse_norm,
            "trend_alignment": trend_alignment_norm,
            "theme_hotness": theme_hotness_norm,
            "retail_panic_surrender_inverted": retail_panic_surrender_inverted_norm # 替换 retail_panic_inverted
        }
        dynamic_context_modulator_weights = context_modulator_weights.copy()
        sentiment_neutrality_base_weight = dynamic_context_modulator_weights.get("sentiment_neutrality", 0.0)
        dynamic_sentiment_neutrality_weight = pd.Series(sentiment_neutrality_base_weight, index=df_index, dtype=np.float32)
        is_market_sentiment_raw_zero = (raw_data.get('market_sentiment_raw', pd.Series(0.0, index=df_index)).abs() < 1e-9)
        dynamic_sentiment_neutrality_weight.loc[is_market_sentiment_raw_zero] = 0.0
        dynamic_context_modulator_weights["sentiment_neutrality"] = dynamic_sentiment_neutrality_weight
        theme_hotness_base_weight = dynamic_context_modulator_weights.get("theme_hotness", 0.0)
        dynamic_theme_hotness_weight = pd.Series(theme_hotness_base_weight, index=df_index, dtype=np.float32)
        is_theme_hotness_raw_zero = (raw_data.get('theme_hotness_raw', pd.Series(0.0, index=df_index)).abs() < 1e-9)
        dynamic_theme_hotness_weight.loc[is_theme_hotness_raw_zero] = 0.0
        dynamic_context_modulator_weights["theme_hotness"] = dynamic_theme_hotness_weight
        retail_panic_surrender_base_weight = dynamic_context_modulator_weights.get("retail_panic_surrender_inverted", 0.0) # 替换 retail_panic_base_weight
        dynamic_retail_panic_surrender_weight = pd.Series(retail_panic_surrender_base_weight, index=df_index, dtype=np.float32)
        is_retail_panic_surrender_raw_zero = (raw_data.get('retail_panic_surrender_index_raw', pd.Series(0.0, index=df_index)).abs() < 1e-9) # 替换 is_retail_panic_raw_zero
        dynamic_retail_panic_surrender_weight.loc[is_retail_panic_surrender_raw_zero] = 0.0
        dynamic_context_modulator_weights["retail_panic_surrender_inverted"] = dynamic_retail_panic_surrender_weight # 替换 retail_panic_inverted
        context_modulator = _robust_geometric_mean(context_modulator_components, dynamic_context_modulator_weights, df_index)
        debug_values = {
            "volatility_instability_norm_inverted": volatility_instability_norm_inverted,
            "adx_norm_inverted": adx_norm_inverted,
            "market_sentiment_norm_bipolar": market_sentiment_norm_bipolar,
            "liquidity_tide_calm_norm": liquidity_tide_calm_norm,
            "market_constitution_neutrality_norm": market_constitution_neutrality_norm,
            "mean_reversion_inverse_norm": mean_reversion_inverse_norm,
            "trend_alignment_norm": trend_alignment_norm,
            "theme_hotness_norm": theme_hotness_norm,
            "retail_panic_surrender_inverted_norm": retail_panic_surrender_inverted_norm, # 替换 retail_panic_inverted_norm
            "dynamic_sentiment_neutrality_weight": dynamic_sentiment_neutrality_weight,
            "dynamic_theme_hotness_weight": dynamic_theme_hotness_weight,
            "dynamic_retail_panic_surrender_weight": dynamic_retail_panic_surrender_weight, # 替换 dynamic_retail_panic_weight
            "context_modulator": context_modulator
        }
        return context_modulator, debug_values

    def _perform_pmd_final_fusion(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, fused_price_direction: pd.Series, fused_momentum_direction: pd.Series, fused_mf_net_flow_slope: pd.Series, base_divergence_score: pd.Series, volume_confirmation_score: pd.Series, main_force_confirmation_score: pd.Series, divergence_quality_score: pd.Series, context_modulator: pd.Series, price_momentum_quality_score: pd.Series, _temp_debug_values: Dict) -> Tuple[pd.Series, Dict]:
        """V1.6 · 动态融合权重、动态指数、显式交互项及RDI增强版 (RDI功能已验证)"""
        final_fusion_exponent_base = pmd_params['final_fusion_exponent']
        synergy_threshold = pmd_params['synergy_threshold']
        synergy_bonus_factor = pmd_params['synergy_bonus_factor']
        conflict_penalty_factor = pmd_params['conflict_penalty_factor']
        dynamic_fusion_weights_params = pmd_params['dynamic_fusion_weights_params']
        dynamic_exponent_params = pmd_params['dynamic_exponent_params']
        interaction_terms_weights = pmd_params['interaction_terms_weights']
        rdi_params = pmd_params['rdi_params']
        final_components = {
            "base_divergence": base_divergence_score.abs(),
            "volume_confirmation": volume_confirmation_score.abs(),
            "main_force_confirmation": main_force_confirmation_score.abs(),
            "divergence_quality": divergence_quality_score,
            "context_modulator": context_modulator
        }
        final_fusion_weights_dict = get_param_value(dynamic_fusion_weights_params.get('base_weights'), {
            "base_divergence": 0.3,
            "volume_confirmation": 0.2,
            "main_force_confirmation": 0.25,
            "divergence_quality": 0.15,
            "context_modulator": 0.1
        })
        _temp_debug_values["最终融合组件"] = {
            "base_divergence_abs": base_divergence_score.abs(),
            "volume_confirmation_abs": volume_confirmation_score.abs(),
            "main_force_confirmation_abs": main_force_confirmation_score.abs(),
            "divergence_quality": divergence_quality_score,
            "context_modulator": context_modulator
        }
        # --- 动态融合权重调整 ---
        if get_param_value(dynamic_fusion_weights_params.get('enabled'), False):
            modulator_signal_1_raw = self.helper._get_atomic_score(df, dynamic_fusion_weights_params['modulator_signal_1'], 0.0)
            modulator_signal_2_raw = self.helper._get_atomic_score(df, dynamic_fusion_weights_params['modulator_signal_2'], 0.0)
            modulator_signal_3_raw = raw_data.get('mean_reversion_frequency_raw', pd.Series(0.0, index=df_index))
            modulator_signal_4_raw = raw_data.get('trend_alignment_index_raw', pd.Series(0.0, index=df_index))
            modulator_signal_5_raw = raw_data.get('retail_panic_surrender_index_raw', pd.Series(0.0, index=df_index))
            modulator_signal_1 = self.helper._normalize_series(modulator_signal_1_raw, df_index, bipolar=True)
            modulator_signal_2 = self.helper._normalize_series(modulator_signal_2_raw, df_index, bipolar=True)
            modulator_signal_3 = self.helper._normalize_series(modulator_signal_3_raw, df_index, bipolar=True, ascending=False)
            modulator_signal_4 = self.helper._normalize_series(modulator_signal_4_raw, df_index, bipolar=True)
            modulator_signal_5 = self.helper._normalize_series(modulator_signal_5_raw, df_index, bipolar=True, ascending=True)
            sensitivity_tension = dynamic_fusion_weights_params['sensitivity_tension']
            sensitivity_liquidity = dynamic_fusion_weights_params['sensitivity_liquidity']
            sensitivity_mean_reversion = dynamic_fusion_weights_params.get('sensitivity_mean_reversion', 0.1)
            sensitivity_trend_alignment = dynamic_fusion_weights_params.get('sensitivity_trend_alignment', 0.1)
            sensitivity_retail_panic_surrender = dynamic_fusion_weights_params.get('sensitivity_retail_panic_surrender', 0.2)
            tension_impact_weights = dynamic_fusion_weights_params['tension_impact_weights']
            liquidity_impact_weights = dynamic_fusion_weights_params['liquidity_impact_weights']
            mean_reversion_impact_weights = dynamic_fusion_weights_params.get('mean_reversion_impact_weights', {})
            trend_alignment_impact_weights = dynamic_fusion_weights_params.get('trend_alignment_impact_weights', {})
            retail_panic_surrender_impact_weights = dynamic_fusion_weights_params.get('retail_panic_surrender_impact_weights', {})
            adjusted_weights_series = pd.DataFrame(final_fusion_weights_dict, index=df_index)
            for k in final_fusion_weights_dict:
                adjusted_weights_series[k] = adjusted_weights_series[k] + (modulator_signal_1 * tension_impact_weights.get(k, 0.0) * sensitivity_tension)
                adjusted_weights_series[k] = adjusted_weights_series[k] + (modulator_signal_2 * liquidity_impact_weights.get(k, 0.0) * sensitivity_liquidity)
                adjusted_weights_series[k] = adjusted_weights_series[k] + (modulator_signal_3 * mean_reversion_impact_weights.get(k, 0.0) * sensitivity_mean_reversion)
                adjusted_weights_series[k] = adjusted_weights_series[k] + (modulator_signal_4 * trend_alignment_impact_weights.get(k, 0.0) * sensitivity_trend_alignment)
                adjusted_weights_series[k] = adjusted_weights_series[k] + (modulator_signal_5 * retail_panic_surrender_impact_weights.get(k, 0.0) * sensitivity_retail_panic_surrender)
            total_dynamic_weight = adjusted_weights_series.sum(axis=1)
            if (total_dynamic_weight > 0).all():
                final_fusion_weights_dict = (adjusted_weights_series.div(total_dynamic_weight, axis=0)).to_dict('series')
            else:
                final_fusion_weights_dict = get_param_value(dynamic_fusion_weights_params.get('base_weights'), {
                    "base_divergence": 0.3, "volume_confirmation": 0.2, "main_force_confirmation": 0.25, "divergence_quality": 0.15, "context_modulator": 0.1
                })
            _temp_debug_values["动态融合权重调整"] = {
                "modulator_signal_1": modulator_signal_1,
                "modulator_signal_2": modulator_signal_2,
                "modulator_signal_3": modulator_signal_3,
                "modulator_signal_4": modulator_signal_4,
                "modulator_signal_5": modulator_signal_5,
                "adjusted_weights_series": adjusted_weights_series,
                "final_fusion_weights_dict_dynamic": final_fusion_weights_dict
            }
        # --- 显式交互项 ---
        price_momentum_main_force_synergy = (fused_price_direction.abs() * fused_mf_net_flow_slope.abs()).pow(0.5)
        price_momentum_main_force_synergy = price_momentum_main_force_synergy.where(np.sign(fused_price_direction) == np.sign(fused_mf_net_flow_slope), 0.0)
        if get_param_value(interaction_terms_weights.get('price_momentum_main_force_synergy'), 0.0) > 0:
            final_components["price_momentum_main_force_synergy"] = price_momentum_main_force_synergy
            final_fusion_weights_dict["price_momentum_main_force_synergy"] = get_param_value(interaction_terms_weights.get('price_momentum_main_force_synergy'), 0.0)
        raw_fused_score = _robust_geometric_mean(final_components, final_fusion_weights_dict, df_index)
        _temp_debug_values["原始融合分数"] = {
            "raw_fused_score": raw_fused_score,
            "price_momentum_main_force_synergy": price_momentum_main_force_synergy
        }
        synergy_conflict_factor = pd.Series(1.0, index=df_index, dtype=np.float32)
        sign_base = np.sign(base_divergence_score.replace(0, 1e-9))
        sign_vol = np.sign(volume_confirmation_score.replace(0, 1e-9))
        sign_mf = np.sign(main_force_confirmation_score.replace(0, 1e-9))
        sign_price_momentum_quality = np.sign(price_momentum_quality_score.replace(0, 1e-9))
        aligned_count = (sign_base == sign_vol).astype(int) + \
                        (sign_base == sign_mf).astype(int) + \
                        (sign_base == sign_price_momentum_quality).astype(int)
        is_synergistic = (aligned_count >= 3) & (base_divergence_score.abs() > synergy_threshold)
        is_conflicting = (aligned_count < 1) & (base_divergence_score.abs() > synergy_threshold)
        synergy_conflict_factor.loc[is_synergistic] = 1 + synergy_bonus_factor
        synergy_conflict_factor.loc[is_conflicting] = 1 - conflict_penalty_factor
        raw_fused_score_modulated = raw_fused_score * synergy_conflict_factor
        # --- RDI (共振、背离、拐点) 分析与调制 ---
        rdi_modulator = pd.Series(1.0, index=df_index, dtype=np.float32)
        if get_param_value(rdi_params.get('enabled'), False):
            # 1. 价格方向 vs 动量方向 RDI
            price_momentum_rdi_score, debug_pm_rdi = self._calculate_rdi_for_pair(
                fused_price_direction, fused_momentum_direction, df_index, rdi_params, "PMD_RDI", "Price_Momentum"
            )
            _temp_debug_values["价格-动量RDI"] = debug_pm_rdi
            # 2. 价格方向 vs 主力资金流斜率 RDI
            price_mf_rdi_score, debug_pmf_rdi = self._calculate_rdi_for_pair(
                fused_price_direction, fused_mf_net_flow_slope, df_index, rdi_params, "PMD_RDI", "Price_MainForce"
            )
            _temp_debug_values["价格-主力RDI"] = debug_pmf_rdi
            # 融合两种RDI分数
            combined_rdi_score = (price_momentum_rdi_score + price_mf_rdi_score) / 2
            # 将RDI分数转换为调制因子，使其在 [1-weight, 1+weight] 之间
            rdi_modulator_weight = rdi_params.get('rdi_modulator_weight', 0.1)
            rdi_modulator = (1 + combined_rdi_score * rdi_modulator_weight).clip(1 - rdi_modulator_weight, 1 + rdi_modulator_weight)
            _temp_debug_values["RDI调制器"] = {
                "price_momentum_rdi_score": price_momentum_rdi_score,
                "price_mf_rdi_score": price_mf_rdi_score,
                "combined_rdi_score": combined_rdi_score,
                "rdi_modulator": rdi_modulator
            }
        raw_fused_score_modulated_by_rdi = raw_fused_score_modulated * rdi_modulator
        _temp_debug_values["RDI调制后的分数"] = {"raw_fused_score_modulated_by_rdi": raw_fused_score_modulated_by_rdi}
        # --- 动态融合指数 ---
        final_fusion_exponent = pd.Series(final_fusion_exponent_base, index=df_index, dtype=np.float32)
        if get_param_value(dynamic_exponent_params.get('enabled'), False):
            exponent_modulator_signal_raw = self.helper._get_safe_series(df, dynamic_exponent_params['modulator_signal'], 0.0)
            exponent_modulator_signal = self.helper._normalize_series(exponent_modulator_signal_raw, df_index, bipolar=False, ascending=False)
            sensitivity = dynamic_exponent_params['sensitivity']
            base_exponent = dynamic_exponent_params['base_exponent']
            min_exponent = dynamic_exponent_params['min_exponent']
            max_exponent = dynamic_exponent_params['max_exponent']
            dynamic_exponent_values = base_exponent + (exponent_modulator_signal * sensitivity)
            final_fusion_exponent = dynamic_exponent_values.clip(min_exponent, max_exponent)
        final_score = np.sign(raw_fused_score_modulated_by_rdi) * (raw_fused_score_modulated_by_rdi.abs().pow(final_fusion_exponent))
        final_score = final_score.clip(-1, 1).fillna(0.0)
        _temp_debug_values["协同/冲突与最终分数"] = {
            "sign_base": sign_base,
            "sign_vol": sign_vol,
            "sign_mf": sign_mf,
            "sign_price_momentum_quality": sign_price_momentum_quality,
            "aligned_count": aligned_count,
            "is_synergistic": is_synergistic,
            "is_conflicting": is_conflicting,
            "synergy_conflict_factor": synergy_conflict_factor,
            "raw_fused_score_modulated": raw_fused_score_modulated,
            "raw_fused_score_modulated_by_rdi": raw_fused_score_modulated_by_rdi,
            "dynamic_fusion_exponent": final_fusion_exponent,
            "final_score": final_score
        }
        return final_score, _temp_debug_values

    def _calculate_rdi_for_pair(self, series_A: pd.Series, series_B: pd.Series, df_index: pd.Index, rdi_params: Dict, method_name: str, pair_name: str) -> Tuple[pd.Series, Dict]:
        """V1.1 · 修复RDI周期键匹配问题"""
        rdi_periods = rdi_params['rdi_periods']
        resonance_reward_factor = rdi_params['resonance_reward_factor']
        divergence_penalty_factor = rdi_params['divergence_penalty_factor']
        inflection_reward_factor = rdi_params['inflection_reward_factor']
        rdi_period_weights = rdi_params['rdi_period_weights']
        all_rdi_scores_by_period = {}
        period_debug_values = {}
        for p in rdi_periods:
            # 计算信号在当前周期内的趋势倾向
            # 使用 rolling mean 来平滑方向，避免短期噪音
            tendency_A = series_A.rolling(window=p, min_periods=1).mean().fillna(0)
            tendency_B = series_B.rolling(window=p, min_periods=1).mean().fillna(0)
            # 共振：两个信号趋势方向一致且均有方向性
            resonance_term = ((np.sign(tendency_A) == np.sign(tendency_B)) & (tendency_A.abs() > 1e-9) & (tendency_B.abs() > 1e-9)).astype(np.float32) * resonance_reward_factor
            # 背离：两个信号趋势方向相反且均有方向性
            divergence_term = ((np.sign(tendency_A) != np.sign(tendency_B)) & (tendency_A.abs() > 1e-9) & (tendency_B.abs() > 1e-9)).astype(np.float32) * divergence_penalty_factor
            # 拐点：信号趋势方向发生改变 (即从正到负或从负到正)
            # 这里判断的是信号的趋势倾向本身是否发生零轴穿越
            inflection_A_term = (tendency_A.shift(1) * tendency_A < 0).astype(np.float32) * inflection_reward_factor
            inflection_B_term = (tendency_B.shift(1) * tendency_B < 0).astype(np.float32) * inflection_reward_factor
            inflection_term = ((inflection_A_term + inflection_B_term) / 2).fillna(0)
            # 结合RDI项，背离作为惩罚项
            period_rdi_score = resonance_term - divergence_term + inflection_term
            # 修复：将键改为 str(p) 以匹配 rdi_period_weights 的键
            all_rdi_scores_by_period[str(p)] = period_rdi_score
            period_debug_values[f"{pair_name}_tendency_A_p{p}"] = tendency_A
            period_debug_values[f"{pair_name}_tendency_B_p{p}"] = tendency_B
            period_debug_values[f"{pair_name}_resonance_term_p{p}"] = resonance_term
            period_debug_values[f"{pair_name}_divergence_term_p{p}"] = divergence_term
            period_debug_values[f"{pair_name}_inflection_term_p{p}"] = inflection_term
            period_debug_values[f"{pair_name}_period_rdi_score_p{p}"] = period_rdi_score
        # 融合不同周期的RDI分数
        fused_rdi_score = _weighted_sum_fusion(all_rdi_scores_by_period, rdi_period_weights, df_index)
        return fused_rdi_score, period_debug_values

    def _calculate_composite_volume_atrophy_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> pd.Series:
        """
        计算复合的量能萎缩分数。
        量能萎缩通常表现为成交量低迷、爆发度低以及建设性换手率低。
        融合负向的量能斜率、反向的量能爆发度以及反向的建设性换手率来构建此分数。
        """
        # 负向量能斜率：量能越萎缩，分数越高
        fused_volume_slope_norm = self.helper._get_mtf_slope_score(df, 'volume_D', pmd_params['mtf_slope_weights'], df_index, method_name, bipolar=True)
        volume_slope_negative_norm = self.helper._normalize_series(fused_volume_slope_norm.clip(upper=0).abs(), df_index, ascending=True)
        # 反向量能爆发度：爆发度越低，分数越高
        volume_burst_inverted_norm = self.helper._normalize_series(raw_data['volume_burstiness_raw'], df_index, ascending=False)
        # 反向建设性换手率：换手率越低，分数越高
        constructive_turnover_inverted_norm = self.helper._normalize_series(raw_data['constructive_turnover_raw'], df_index, ascending=False)
        components = {
            "volume_slope_negative": volume_slope_negative_norm,
            "volume_burst_inverted": volume_burst_inverted_norm,
            "constructive_turnover_inverted": constructive_turnover_inverted_norm
        }
        weights = {"volume_slope_negative": 0.4, "volume_burst_inverted": 0.3, "constructive_turnover_inverted": 0.3}
        return _robust_geometric_mean(components, weights, df_index)

    def _calculate_composite_distribution_intent_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> pd.Series:
        """
        计算复合的派发意图分数。
        派发意图通常伴随着主力卖出、价格下跌、量能放大以及市场欺骗行为。
        融合主力卖出OFI、正向欺骗指数、负向价格斜率和正向量能斜率来构建此分数。
        """
        # 主力卖出OFI：越高越好
        mf_sell_ofi_norm = self.helper._normalize_series(raw_data['main_force_sell_ofi_raw'], df_index, ascending=True)
        # 欺骗指数正向：越高越好
        deception_index_positive_norm = self.helper._normalize_series(raw_data['deception_index_raw'].clip(lower=0), df_index, ascending=True)
        # 负向价格斜率：价格下跌，分数越高
        fused_price_direction_norm = self.helper._get_mtf_slope_score(df, 'close_D', pmd_params['mtf_slope_weights'], df_index, method_name, bipolar=True)
        price_slope_negative_norm = self.helper._normalize_series(fused_price_direction_norm.clip(upper=0).abs(), df_index, ascending=True)
        # 正向量能斜率：量能放大，分数越高
        fused_volume_slope_norm = self.helper._get_mtf_slope_score(df, 'volume_D', pmd_params['mtf_slope_weights'], df_index, method_name, bipolar=True)
        volume_slope_positive_norm = self.helper._normalize_series(fused_volume_slope_norm.clip(lower=0), df_index, ascending=True)
        components = {
            "mf_sell_ofi": mf_sell_ofi_norm,
            "deception_index_positive": deception_index_positive_norm,
            "price_slope_negative": price_slope_negative_norm,
            "volume_slope_positive": volume_slope_positive_norm
        }
        weights = {"mf_sell_ofi": 0.3, "deception_index_positive": 0.3, "price_slope_negative": 0.2, "volume_slope_positive": 0.2}
        return _robust_geometric_mean(components, weights, df_index)

    def _calculate_composite_covert_accumulation_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> pd.Series:
        """
        计算复合的隐蔽吸筹分数。
        隐蔽吸筹通常发生在价格下跌或横盘时，伴随着主力买入、量能放大以及诱空欺骗行为。
        融合主力买入OFI、负向欺骗指数、负向价格斜率和正向量能斜率来构建此分数。
        """
        # 主力买入OFI：越高越好
        mf_buy_ofi_norm = self.helper._normalize_series(raw_data['main_force_buy_ofi_raw'], df_index, ascending=True)
        # 欺骗指数负向：越低越好 (即欺骗指数为负，代表诱空，有利于吸筹)
        deception_index_negative_norm = self.helper._normalize_series(raw_data['deception_index_raw'].clip(upper=0).abs(), df_index, ascending=True)
        # 负向价格斜率：价格下跌或横盘，分数越高
        fused_price_direction_norm = self.helper._get_mtf_slope_score(df, 'close_D', pmd_params['mtf_slope_weights'], df_index, method_name, bipolar=True)
        price_slope_negative_or_flat_norm = self.helper._normalize_series(fused_price_direction_norm.clip(upper=0).abs(), df_index, ascending=True) # 简化为负向
        # 正向量能斜率：量能放大，分数越高
        fused_volume_slope_norm = self.helper._get_mtf_slope_score(df, 'volume_D', pmd_params['mtf_slope_weights'], df_index, method_name, bipolar=True)
        volume_slope_positive_norm = self.helper._normalize_series(fused_volume_slope_norm.clip(lower=0), df_index, ascending=True)
        components = {
            "mf_buy_ofi": mf_buy_ofi_norm,
            "deception_index_negative": deception_index_negative_norm,
            "price_slope_negative_or_flat": price_slope_negative_or_flat_norm,
            "volume_slope_positive": volume_slope_positive_norm
        }
        weights = {"mf_buy_ofi": 0.3, "deception_index_negative": 0.3, "price_slope_negative_or_flat": 0.2, "volume_slope_positive": 0.2}
        return _robust_geometric_mean(components, weights, df_index)

    def _calculate_composite_chip_divergence_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> pd.Series:
        """
        计算复合的筹码背离分数。
        筹码背离通常指价格与筹码结构之间的不一致。这里我们简化为当筹码健康度高、赢家集中度低（分散）
        且输家集中度高时，筹码结构趋于优化。结合价格方向，形成双极性背离分数。
        """
        # 筹码健康度：越高越好
        chip_health_norm = self.helper._normalize_series(raw_data['chip_health_raw'], df_index, ascending=True)
        # 赢家集中度反向：赢家越分散越好 (有利于筹码换手)
        winner_concentration_inverted_norm = self.helper._normalize_series(raw_data['winner_concentration_90pct_raw'], df_index, ascending=False)
        # 输家集中度正向：输家越集中越好 (有利于洗盘结束)
        loser_concentration_norm = self.helper._normalize_series(raw_data['loser_concentration_90pct_raw'], df_index, ascending=True)
        # 计算一个单极性的“筹码结构优化”分数
        chip_structure_optimization = _robust_geometric_mean({
            "chip_health": chip_health_norm,
            "winner_concentration_inverted": winner_concentration_inverted_norm,
            "loser_concentration": loser_concentration_norm
        }, {"chip_health": 0.4, "winner_concentration_inverted": 0.3, "loser_concentration": 0.3}, df_index)
        # 结合价格方向，使其成为双极性
        fused_price_direction_norm = self.helper._get_mtf_slope_score(df, 'close_D', pmd_params['mtf_slope_weights'], df_index, method_name, bipolar=True)
        chip_divergence_score = chip_structure_optimization * fused_price_direction_norm
        return chip_divergence_score.clip(-1, 1)

    def _calculate_composite_upward_efficiency_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> pd.Series:
        """
        计算复合的上涨效率分数。
        上涨效率高通常表现为价格上涨，但量能相对较小或爆发度低，即“轻量化”上涨。
        融合正向价格斜率、反向量能斜率和反向量能爆发度来构建此分数。
        """
        # 正向价格斜率：越高越好
        fused_price_slope_norm = self.helper._get_mtf_slope_score(df, 'close_D', pmd_params['mtf_slope_weights'], df_index, method_name, bipolar=True)
        price_slope_positive_norm = self.helper._normalize_series(fused_price_slope_norm.clip(lower=0), df_index, ascending=True)
        # 反向量能斜率：量能越小，分数越高
        fused_volume_slope_norm = self.helper._get_mtf_slope_score(df, 'volume_D', pmd_params['mtf_slope_weights'], df_index, method_name, bipolar=True)
        volume_slope_inverted_norm = self.helper._normalize_series(fused_volume_slope_norm.abs(), df_index, ascending=False) # 绝对值越小越好
        # 反向量能爆发度：爆发度越低，分数越高
        volume_burst_inverted_norm = self.helper._normalize_series(raw_data['volume_burstiness_raw'], df_index, ascending=False)
        components = {
            "price_slope_positive": price_slope_positive_norm,
            "volume_slope_inverted": volume_slope_inverted_norm,
            "volume_burst_inverted": volume_burst_inverted_norm
        }
        weights = {"price_slope_positive": 0.4, "volume_slope_inverted": 0.3, "volume_burst_inverted": 0.3}
        return _robust_geometric_mean(components, weights, df_index)

    def _calculate_composite_price_upward_momentum_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> pd.Series:
        """
        计算复合的价格上涨动量分数。
        价格上涨动量强通常表现为价格持续上涨且加速。
        融合正向价格斜率和正向价格加速度来构建此分数。
        """
        # 正向价格斜率：越高越好
        fused_price_slope_norm = self.helper._get_mtf_slope_score(df, 'close_D', pmd_params['mtf_slope_weights'], df_index, method_name, bipolar=True)
        price_slope_positive_norm = self.helper._normalize_series(fused_price_slope_norm.clip(lower=0), df_index, ascending=True)
        # 正向价格加速度：越高越好
        fused_price_accel_norm = self.helper._get_mtf_slope_accel_score(df, 'close_D', pmd_params['mtf_accel_weights'], df_index, method_name, bipolar=True)
        price_accel_positive_norm = self.helper._normalize_series(fused_price_accel_norm.clip(lower=0), df_index, ascending=True)
        components = {
            "price_slope_positive": price_slope_positive_norm,
            "price_accel_positive": price_accel_positive_norm
        }
        weights = {"price_slope_positive": 0.5, "price_accel_positive": 0.5}
        return _robust_geometric_mean(components, weights, df_index)

    def _calculate_composite_price_downward_momentum_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> pd.Series:
        """
        计算复合的价格下跌动量分数。
        价格下跌动量强通常表现为价格持续下跌且加速。
        融合负向价格斜率和负向价格加速度来构建此分数。
        """
        # 负向价格斜率：越低越好 (绝对值越大越好)
        fused_price_slope_norm = self.helper._get_mtf_slope_score(df, 'close_D', pmd_params['mtf_slope_weights'], df_index, method_name, bipolar=True)
        price_slope_negative_norm = self.helper._normalize_series(fused_price_slope_norm.clip(upper=0).abs(), df_index, ascending=True)
        # 负向价格加速度：越低越好 (绝对值越大越好)
        fused_price_accel_norm = self.helper._get_mtf_slope_accel_score(df, 'close_D', pmd_params['mtf_accel_weights'], df_index, method_name, bipolar=True)
        price_accel_negative_norm = self.helper._normalize_series(fused_price_accel_norm.clip(upper=0).abs(), df_index, ascending=True)
        components = {
            "price_slope_negative": price_slope_negative_norm,
            "price_accel_negative": price_accel_negative_norm
        }
        weights = {"price_slope_negative": 0.5, "price_accel_negative": 0.5}
        return _robust_geometric_mean(components, weights, df_index)

    def _calculate_composite_momentum_quality_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> pd.Series:
        """
        计算复合的动量品质分数。
        动量品质高通常表现为多个动量指标（MACDh、RSI、ROC）在多时间框架上具有一致的方向和强度。
        这里直接复用 helper 中的 _get_mtf_cohesion_score 来评估这种协同性。
        """
        return self.helper._get_mtf_cohesion_score(df, ['MACDh_13_34_8_D', 'RSI_13_D', 'ROC_13_D'], pmd_params['mtf_slope_weights'], df_index, method_name)

    def _calculate_composite_stability_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> pd.Series:
        """
        计算复合的稳定性分数。
        市场稳定性高通常表现为波动率低、均值回归频率低以及趋势对齐度高。
        融合反向波动率不稳定性、反向均值回归频率和正向趋势对齐指数来构建此分数。
        """
        # 波动率不稳定性反向：越低越好
        volatility_instability_inverted_norm = self.helper._normalize_series(raw_data['volatility_instability_raw'], df_index, ascending=False)
        # 均值回归频率反向：越低越好
        mean_reversion_frequency_inverted_norm = self.helper._normalize_series(raw_data['mean_reversion_frequency_raw'], df_index, ascending=False)
        # 趋势对齐指数正向：越高越好
        trend_alignment_norm = self.helper._normalize_series(raw_data['trend_alignment_index_raw'], df_index, ascending=True)
        components = {
            "volatility_instability_inverted": volatility_instability_inverted_norm,
            "mean_reversion_frequency_inverted": mean_reversion_frequency_inverted_norm,
            "trend_alignment": trend_alignment_norm
        }
        weights = {"volatility_instability_inverted": 0.4, "mean_reversion_frequency_inverted": 0.3, "trend_alignment": 0.3}
        return _robust_geometric_mean(components, weights, df_index)

    def _calculate_composite_chip_historical_potential_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> pd.Series:
        """
        计算复合的筹码历史潜力分数。
        筹码历史潜力高通常表现为筹码健康度高、量能轮廓清晰（熵低）以及主力信念指数高。
        融合筹码健康度、反向量能轮廓熵和主力信念指数来构建此分数。
        """
        # 筹码健康度：越高越好
        chip_health_norm = self.helper._normalize_series(raw_data['chip_health_raw'], df_index, ascending=True)
        # 量能轮廓熵反向：熵越低（结构越清晰），分数越高
        volume_profile_entropy_inverted_norm = self.helper._normalize_series(raw_data['volume_profile_entropy_raw'], df_index, ascending=False)
        # 主力信念指数：越高越好
        main_force_conviction_norm = self.helper._normalize_series(raw_data['main_force_conviction_raw'], df_index, ascending=True)
        components = {
            "chip_health": chip_health_norm,
            "volume_profile_entropy_inverted": volume_profile_entropy_inverted_norm,
            "main_force_conviction": main_force_conviction_norm
        }
        weights = {"chip_health": 0.4, "volume_profile_entropy_inverted": 0.3, "main_force_conviction": 0.3}
        return _robust_geometric_mean(components, weights, df_index)

    def _calculate_composite_liquidity_tide_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> pd.Series:
        """
        计算复合的流动性潮汐分数。
        流动性潮汐平静通常表现为订单簿失衡度低（接近中性）、微观价格冲击不对称性低（接近中性）
        以及主力滑点指数低。
        融合反向订单簿失衡绝对值、反向微观价格冲击不对称性绝对值和反向主力滑点指数来构建此分数。
        """
        # 订单簿失衡绝对值反向：越接近中性，分数越高
        order_book_imbalance_abs_inverted_norm = self.helper._normalize_series(raw_data['order_book_imbalance_raw'].abs(), df_index, ascending=False)
        # 微观价格冲击不对称性绝对值反向：越接近中性，分数越高
        micro_price_impact_asymmetry_abs_inverted_norm = self.helper._normalize_series(raw_data['micro_price_impact_asymmetry_raw'].abs(), df_index, ascending=False)
        # 主力滑点指数反向：滑点越低，分数越高
        main_force_slippage_inverted_norm = self.helper._normalize_series(raw_data['main_force_slippage_index_raw'], df_index, ascending=False)
        components = {
            "order_book_imbalance_abs_inverted": order_book_imbalance_abs_inverted_norm,
            "micro_price_impact_asymmetry_abs_inverted": micro_price_impact_asymmetry_abs_inverted_norm,
            "main_force_slippage_inverted": main_force_slippage_inverted_norm
        }
        weights = {"order_book_imbalance_abs_inverted": 0.4, "micro_price_impact_asymmetry_abs_inverted": 0.3, "main_force_slippage_inverted": 0.3}
        return _robust_geometric_mean(components, weights, df_index)

    def _calculate_composite_market_constitution_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> pd.Series:
        """
        计算复合的市场体质分数。
        市场体质健康通常表现为趋势不明显（ADX低）、波动率稳定以及量能轮廓清晰（熵低）。
        融合反向ADX、反向波动率不稳定性以及反向量能轮廓熵来构建此分数。
        """
        # ADX反向：ADX越低（趋势越不明显），分数越高 (代表中性市场体质)
        adx_inverted_norm = self.helper._normalize_series(raw_data['adx_raw'], df_index, ascending=False)
        # 波动率不稳定性反向：波动率越稳定，分数越高
        volatility_instability_inverted_norm = self.helper._normalize_series(raw_data['volatility_instability_raw'], df_index, ascending=False)
        # 量能轮廓熵反向：熵越低（结构越清晰），分数越高
        volume_profile_entropy_inverted_norm = self.helper._normalize_series(raw_data['volume_profile_entropy_raw'], df_index, ascending=False)
        components = {
            "adx_inverted": adx_inverted_norm,
            "volatility_instability_inverted": volatility_instability_inverted_norm,
            "volume_profile_entropy_inverted": volume_profile_entropy_inverted_norm
        }
        weights = {"adx_inverted": 0.3, "volatility_instability_inverted": 0.4, "volume_profile_entropy_inverted": 0.3}
        return _robust_geometric_mean(components, weights, df_index)

    def _calculate_composite_market_tension_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> pd.Series:
        """
        计算复合的市场张力分数。
        市场张力高通常表现为波动率不稳定性高、趋势强度高（ADX高）以及结构张力指数高。
        融合波动率不稳定性、ADX和结构张力指数（如果可用）来构建此分数。
        """
        # 波动率不稳定性：越高越好
        volatility_instability_norm = self.helper._normalize_series(raw_data['volatility_instability_raw'], df_index, ascending=True)
        # ADX：越高越好
        adx_norm = self.helper._normalize_series(raw_data['adx_raw'], df_index, ascending=True)
        # 结构张力指数 (structural_tension_index_D) - 假设它作为原始数据可用
        structural_tension_raw = self.helper._get_safe_series(df, 'structural_tension_index_D', np.nan, method_name=method_name)
        structural_tension_norm = self.helper._normalize_series(structural_tension_raw, df_index, ascending=True)
        components = {
            "volatility_instability": volatility_instability_norm,
            "adx": adx_norm,
            "structural_tension": structural_tension_norm
        }
        weights = {"volatility_instability": 0.4, "adx": 0.3, "structural_tension": 0.3}
        return _robust_geometric_mean(components, weights, df_index)

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """V1.3 · 模块化与增强版 (传递 fused_momentum_direction 给最终融合方法)"""
        method_name = "_calculate_price_momentum_divergence"
        is_debug_enabled_for_method = get_param_value(self.helper.debug_params.get('enabled'), False) and get_param_value(self.helper.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug_enabled_for_method and self.helper.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.helper.probe_dates]
            for date in reversed(df.index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        if probe_ts is None:
            is_debug_enabled_for_method = False
        debug_output = {}
        _temp_debug_values = {}
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算价势背离..."] = ""
        df_index = df.index
        pmd_params = self._get_pmd_params(config)
        if not self._validate_pmd_signals(df, pmd_params, method_name):
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。"] = ""
                self._print_debug_output_pmd(debug_output, probe_ts, method_name, pd.Series(0.0, index=df.index, dtype=np.float32))
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        raw_data = self._get_pmd_raw_data(df, pmd_params, method_name)
        _temp_debug_values["原始信号值"] = raw_data
        fused_price_direction, debug_price_direction = self._calculate_fused_price_direction(df, df_index, raw_data, pmd_params, method_name)
        _temp_debug_values["融合价格方向"] = debug_price_direction
        fused_momentum_direction, debug_momentum_direction = self._calculate_fused_momentum_direction(df, df_index, raw_data, pmd_params, method_name)
        _temp_debug_values["融合动量方向"] = debug_momentum_direction
        base_divergence_score = (fused_price_direction - fused_momentum_direction).clip(-1, 1)
        _temp_debug_values["基础背离分数"] = {"base_divergence_score": base_divergence_score}
        volume_confirmation_score, debug_volume_confirmation = self._calculate_volume_confirmation_score(df, df_index, raw_data, pmd_params, base_divergence_score, method_name)
        _temp_debug_values["量能确认分数"] = debug_volume_confirmation
        main_force_confirmation_score, fused_mf_net_flow_slope, debug_mf_confirmation = self._calculate_main_force_confirmation_score(df, df_index, raw_data, pmd_params, base_divergence_score, method_name)
        _temp_debug_values["主力/筹码确认分数"] = debug_mf_confirmation
        divergence_quality_score, debug_divergence_quality = self._calculate_divergence_quality_score(df_index, raw_data, pmd_params, base_divergence_score, fused_price_direction, fused_momentum_direction)
        _temp_debug_values["背离质量分数"] = debug_divergence_quality
        context_modulator, debug_context_modulator = self._calculate_context_modulator(df_index, raw_data, pmd_params)
        _temp_debug_values["情境调制器"] = debug_context_modulator
        final_score, debug_final_fusion = self._perform_pmd_final_fusion(
            df, df_index, raw_data, pmd_params,
            fused_price_direction, fused_momentum_direction, fused_mf_net_flow_slope, # 新增传递 fused_momentum_direction
            base_divergence_score, volume_confirmation_score, main_force_confirmation_score,
            divergence_quality_score, context_modulator, debug_price_direction['price_momentum_quality_score'],
            _temp_debug_values
        )
        # if is_debug_enabled_for_method and probe_ts:
        #     self._print_debug_output_pmd(_temp_debug_values, probe_ts, method_name, final_score)
        return final_score.astype(np.float32)




