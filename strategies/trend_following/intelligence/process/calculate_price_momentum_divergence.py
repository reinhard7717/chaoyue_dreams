# strategies\trend_following\intelligence\process\calculate_price_momentum_divergence.py
# 【V1.0.0 · 价格动量背离计算器】 计算“价格动量背离”的专属关系分数。
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
        """V2.5.0 · 协同动力学参数体系：引入协同进攻指数与共振权重偏移"""
        params = get_param_value(config.get('price_momentum_divergence_params'), {})
        return {
            "fib_periods": [5, 13, 21, 34, 55],
            "momentum_kinematic_weights": {"slope": 0.2, "accel": 0.4, "jerk": 0.4},
            "momentum_components_weights": {"MACDh_13_34_8_D": 0.3, "RSI_13_D": 0.25, "CHIP_RSI_DIVERGENCE_D": 0.2, "HM_COORDINATED_ATTACK_D": 0.25},
            "synergy_boost_factor": 1.5, # 协同共振时的动量增强系数
            "momentum_asymmetry": 1.4, # A股杀跌动量敏感度因子
            "kinematic_deadzone": 1e-5,
            "final_fusion_exponent": get_param_value(params.get('final_fusion_exponent'), 1.8),
            "rdi_params": {
                "enabled": True,
                "rdi_periods": [5, 13, 21],
                "rdi_modulator_weight": 0.2,
                "rdi_period_weights": {"5": 0.4, "13": 0.3, "21": 0.3}
            }
        }

    def _validate_pmd_signals(self, df: pd.DataFrame, pmd_params: Dict, method_name: str) -> bool:
        """V2.5.0 · 协同动量数据链校验：确保博弈共振信号完整性"""
        fib_periods = pmd_params['fib_periods']
        base_momentum_cols = ["MACDh_13_34_8_D", "RSI_13_D"]
        required_signals = base_momentum_cols + ["CHIP_RSI_DIVERGENCE_D", "HM_COORDINATED_ATTACK_D", "SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D", "BIAS_55_D"]
        for col in base_momentum_cols:
            for p in fib_periods:
                required_signals.extend([f'SLOPE_{p}_{col}', f'ACCEL_{p}_{col}', f'JERK_{p}_{col}'])
        missing = [s for s in required_signals if s not in df.columns]
        if missing:
            raise ValueError(f"[{method_name}] 关键协同动量信号缺失: {missing}")
        return True

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
        """V2.2.0 · 非对称动力学路径版 (集成死区过滤与抛物线权重偏移)"""
        fib_periods = pmd_params['fib_periods']
        deadzone = pmd_params['kinematic_deadzone']
        asym_factor = pmd_params['asymmetric_accel_factor']
        is_parabolic = df['STATE_PARABOLIC_WARNING_D'] > 0.5
        kinematic_scores = {}
        debug_matrix = {}
        for base_col in pmd_params['price_components_weights'].keys():
            period_scores = []
            for p in fib_periods:
                # 1. 原始信号获取与死区过滤 (去噪)
                s_raw = df[f'SLOPE_{p}_{base_col}'].where(df[f'SLOPE_{p}_{base_col}'].abs() > deadzone, 0.0)
                a_raw = df[f'ACCEL_{p}_{base_col}'].where(df[f'ACCEL_{p}_{base_col}'].abs() > deadzone, 0.0)
                j_raw = df[f'JERK_{p}_{base_col}'].where(df[f'JERK_{p}_{base_col}'].abs() > deadzone, 0.0)
                # 2. 非对称灵敏度处理 (强化负向加速度)
                a_processed = np.where(a_raw < 0, a_raw * asym_factor, a_raw)
                j_processed = np.where(j_raw < 0, j_raw * asym_factor, j_raw)
                # 3. 鲁棒归一化 (tanh映射回避零基陷阱)
                s_norm = self.helper._normalize_series(s_raw, df_index, bipolar=True)
                a_norm = self.helper._normalize_series(pd.Series(a_processed, index=df_index), df_index, bipolar=True)
                j_norm = self.helper._normalize_series(pd.Series(j_processed, index=df_index), df_index, bipolar=True)
                # 4. 动态权重决策
                current_weights = np.where(is_parabolic, pmd_params['parabolic_kinematic_weights']['slope'], pmd_params['base_kinematic_weights']['slope'])
                w_s = pd.Series(current_weights, index=df_index)
                w_a = pd.Series(np.where(is_parabolic, pmd_params['parabolic_kinematic_weights']['accel'], pmd_params['base_kinematic_weights']['accel']), index=df_index)
                w_j = pd.Series(np.where(is_parabolic, pmd_params['parabolic_kinematic_weights']['jerk'], pmd_params['base_kinematic_weights']['jerk']), index=df_index)
                combined = (s_norm * w_s + a_norm * w_a + j_norm * w_j)
                period_scores.append(combined)
            kinematic_scores[base_col] = pd.concat(period_scores, axis=1).mean(axis=1)
            debug_matrix[f"{base_col}_kinematic"] = kinematic_scores[base_col]
        # 价格组件最终融合
        fused_price_direction = pd.Series(0.0, index=df_index)
        for col, weight in pmd_params['price_components_weights'].items():
            fused_price_direction += kinematic_scores[col] * weight
        debug_matrix["final_price_direction"] = fused_price_direction
        return fused_price_direction, debug_matrix

    def _calculate_volume_kinematics(self, df: pd.DataFrame, df_index: pd.Index, pmd_params: Dict) -> Tuple[pd.Series, Dict]:
        """V2.3.0 · 成交量动力学建模：去噪处理与三阶导数矩阵 [cite: 1, 3, 4]"""
        fib_periods = pmd_params['fib_periods']
        deadzone = pmd_params['kinematic_deadzone']
        k_weights = pmd_params['base_kinematic_weights']
        period_scores = []
        debug_v = {}
        for p in fib_periods:
            v_s = df[f'SLOPE_{p}_volume_D'].where(df[f'SLOPE_{p}_volume_D'].abs() > deadzone, 0.0) [cite: 4]
            v_a = df[f'ACCEL_{p}_volume_D'].where(df[f'ACCEL_{p}_volume_D'].abs() > deadzone, 0.0) [cite: 4]
            v_j = df[f'JERK_{p}_volume_D'].where(df[f'JERK_{p}_volume_D'].abs() > deadzone, 0.0) [cite: 4]
            s_n = self.helper._normalize_series(v_s, df_index, bipolar=True)
            a_n = self.helper._normalize_series(v_a, df_index, bipolar=True)
            j_n = self.helper._normalize_series(v_j, df_index, bipolar=True)
            combined = (s_n * k_weights['slope'] + a_n * k_weights['accel'] + j_n * k_weights['jerk'])
            period_scores.append(combined)
            debug_v[f"vol_kinematic_p{p}"] = combined
        fused_volume_kinematics = pd.concat(period_scores, axis=1).mean(axis=1)
        return fused_volume_kinematics, debug_v

    def _calculate_pv_sync_anomaly(self, price_k: pd.Series, vol_k: pd.Series, df: pd.DataFrame, df_index: pd.Index) -> Tuple[pd.Series, Dict]:
        """V2.3.0 · 量价同步异动监测：捕获“有量无果”与“虚空驱动” """
        # 计算价量动力学差值：正值代表价格驱动强于量能支撑（虚空），负值代表量能驱动强于价格表现（滞涨）
        pv_divergence = (price_k - vol_k)
        # 引入价量效率修正 
        vpa_eff = self.helper._normalize_series(df['VPA_EFFICIENCY_D'], df_index, bipolar=True) [cite: 1]
        vpa_mf = self.helper._normalize_series(df['VPA_MF_ADJUSTED_EFF_D'], df_index, bipolar=True) [cite: 1]
        # 异动得分：当价量背离且效率低下时，异动分值升高
        anomaly_score = (pv_divergence.abs() * (1 - vpa_eff.abs())).clip(0, 1)
        debug_sync = {"pv_divergence": pv_divergence, "vpa_eff": vpa_eff, "vpa_mf": vpa_mf, "pv_anomaly": anomaly_score}
        return anomaly_score, debug_sync

    def _calculate_fused_momentum_direction(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> Tuple[pd.Series, Dict]:
        """V2.5.0 · 协同动力学融合版 (集成协同进攻共振与三阶导数动态调制)"""
        fib_periods = pmd_params['fib_periods']
        k_w = pmd_params['momentum_kinematic_weights']
        deadzone = pmd_params['kinematic_deadzone']
        asym = pmd_params['momentum_asymmetry']
        m_comp_w = pmd_params['momentum_components_weights']
        # 1. 协同进攻指数归一化 (核心权重调节器)
        synergy_idx = self.helper._normalize_series(df['HM_COORDINATED_ATTACK_D'], df_index, bipolar=False)
        kinematic_scores = {}
        for m_col in ["MACDh_13_34_8_D", "RSI_13_D"]:
            period_scores = []
            for p in fib_periods:
                s_r = df[f'SLOPE_{p}_{m_col}'].where(df[f'SLOPE_{p}_{m_col}'].abs() > deadzone, 0.0)
                a_r = df[f'ACCEL_{p}_{m_col}'].where(df[f'ACCEL_{p}_{m_col}'].abs() > deadzone, 0.0)
                j_r = df[f'JERK_{p}_{m_col}'].where(df[f'JERK_{p}_{m_col}'].abs() > deadzone, 0.0)
                # 非对称性处理：强化空头加速度敏感度
                a_p = np.where(a_r < 0, a_r * asym, a_r)
                j_p = np.where(j_r < 0, j_r * asym, j_r)
                s_n = self.helper._normalize_series(s_r, df_index, bipolar=True)
                a_n = self.helper._normalize_series(pd.Series(a_p, index=df_index), df_index, bipolar=True)
                j_n = self.helper._normalize_series(pd.Series(j_p, index=df_index), df_index, bipolar=True)
                # 动态调制：协同共振越高，Jerk权重越高，捕捉爆发力
                w_j = k_w['jerk'] * (1 + synergy_idx * 0.5)
                w_s = k_w['slope'] * (1 - synergy_idx * 0.2)
                combined = (s_n * w_s + a_n * k_w['accel'] + j_n * w_j)
                period_scores.append(combined)
            kinematic_scores[m_col] = pd.concat(period_scores, axis=1).mean(axis=1)
        # 2. 筹码与协同组件
        chip_momentum = self.helper._normalize_series(df['CHIP_RSI_DIVERGENCE_D'], df_index, bipolar=True)
        synergy_momentum = self.helper._normalize_series(df['HM_COORDINATED_ATTACK_D'], df_index, bipolar=True)
        # 3. 基础动量方向融合
        fused_momentum_base = (kinematic_scores["MACDh_13_34_8_D"] * m_comp_w["MACDh_13_34_8_D"] +
                               kinematic_scores["RSI_13_D"] * m_comp_w["RSI_13_D"] +
                               chip_momentum * m_comp_w["CHIP_RSI_DIVERGENCE_D"] +
                               synergy_momentum * m_comp_w["HM_COORDINATED_ATTACK_D"])
        # 4. 聪明钱与回归拉力修正
        sm_div = self.helper._normalize_series(df['SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D'], df_index, bipolar=True)
        bias_drag = self.helper._normalize_series(df['BIAS_55_D'], df_index, bipolar=True, ascending=False)
        correction = (sm_div * 0.4 + bias_drag * 0.6)
        # 最终动量：当协同度极低时，对正向动量进行抑制（防止散户追高陷阱）
        fused_momentum_direction = (fused_momentum_base * 0.8 + correction * 0.2)
        fused_momentum_direction = np.where((fused_momentum_direction > 0) & (synergy_idx < 0.2), fused_momentum_direction * 0.5, fused_momentum_direction)
        debug_v = {
            "node_macd_kinematic": kinematic_scores["MACDh_13_34_8_D"],
            "node_rsi_kinematic": kinematic_scores["RSI_13_D"],
            "node_synergy_raw": synergy_idx,
            "node_chip_momentum": chip_momentum,
            "final_momentum_direction": pd.Series(fused_momentum_direction, index=df_index)
        }
        return pd.Series(fused_momentum_direction, index=df_index), debug_v

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
        """V2.0.0 · A股博弈环境调制：引入情绪极端与物理回归压力"""
        weights = pmd_params['context_weights']
        emotional_extreme = self.helper._normalize_series(raw_data['STATE_EMOTIONAL_EXTREME_D'], df_index, bipolar=True)
        rubber_band = self.helper._normalize_series(raw_data['MA_RUBBER_BAND_EXTENSION_D'].abs(), df_index, ascending=True)
        tension = self.helper._normalize_series(raw_data['MA_POTENTIAL_TENSION_INDEX_D'], df_index, ascending=True)
        components = {
            "emotional_extreme": emotional_extreme.abs(),
            "rubber_band_extension": rubber_band,
            "tension_index": tension
        }
        context_modulator = _robust_geometric_mean(components, weights, df_index)
        debug_info = {f"context_{k}": v for k, v in components.items()}
        debug_info["final_context_modulator"] = context_modulator
        return context_modulator, debug_info

    def _perform_pmd_final_fusion(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, fused_price_direction: pd.Series, fused_momentum_direction: pd.Series, base_divergence_score: pd.Series, confirmation_score: pd.Series, context_modulator: pd.Series, _temp_debug_values: Dict) -> Tuple[pd.Series, Dict]:
        """V2.0.0 · 弹性张力融合逻辑：全探针暴露"""
        exponent = pmd_params['final_fusion_exponent']
        rdi_params = pmd_params['rdi_params']
        # 基础背离强度：由价格动量差值与大单异常程度共同决定
        large_order_confirm = self.helper._normalize_series(raw_data['LARGE_ORDER_ANOMALY_D'], df_index, ascending=True)
        raw_fusion = (base_divergence_score.abs() * 0.5 + confirmation_score.abs() * 0.3 + large_order_confirm * 0.2)
        # RDI 调制
        rdi_modulator = pd.Series(1.0, index=df_index, dtype=np.float32)
        if rdi_params['enabled']:
            price_momentum_rdi, rdi_debug = self._calculate_rdi_for_pair(fused_price_direction, fused_momentum_direction, df_index, rdi_params, "PMD_RDI", "P_M")
            _temp_debug_values["RDI_Node"] = rdi_debug
            rdi_modulator = (1 + price_momentum_rdi * rdi_params['rdi_modulator_weight']).clip(0.5, 1.5)
        # 最终得分计算：符号(背离方向) * (原始分 * 环境调制 * RDI调制)^指数
        modulated_score = raw_fusion * context_modulator * rdi_modulator
        final_score = np.sign(base_divergence_score) * (modulated_score.pow(exponent))
        final_score = final_score.clip(-1, 1).fillna(0.0)
        _temp_debug_values["Fusion_Matrix"] = {
            "base_divergence": base_divergence_score,
            "confirmation_score": confirmation_score,
            "context_modulator": context_modulator,
            "rdi_modulator": rdi_modulator,
            "large_order_confirm": large_order_confirm,
            "modulated_score": modulated_score,
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
        """V2.5.0 · 顶层调度：协同动力学全周期背离分析引擎"""
        method_name = "PriceMomentumSynergyDivergence"
        pmd_params = self._get_pmd_params(config)
        df_index = df.index
        if not self._validate_pmd_signals(df, pmd_params, method_name):
            return pd.Series(0.0, index=df_index, dtype=np.float32)
        # 1. 计算融合价格动力学 (已集成抛物线权重偏移)
        fused_p, p_debug = self._calculate_fused_price_direction(df, df_index, {}, pmd_params, method_name)
        # 2. 计算融合量能动力学 (已集成三阶导数矩阵)
        fused_v, v_debug = self._calculate_volume_kinematics(df, df_index, pmd_params)
        # 3. 计算融合协同动量方向 (集成三阶导数与协同进攻指数)
        fused_m, m_debug = self._calculate_fused_momentum_direction(df, df_index, {}, pmd_params, method_name)
        # 4. 量价同步异动监测 (捕获动力错位)
        pv_anomaly, sync_debug = self._calculate_pv_sync_anomaly(fused_p, fused_v, df, df_index)
        # 5. 最终复合背离分值：(价格 - 动量) * 异动因子 * 路径品质
        base_div = (fused_p - fused_m) * (1 + pv_anomaly)
        # 6. RDI 调制逻辑集成
        if pmd_params['rdi_params']['enabled']:
            rdi_score, _ = self._calculate_rdi_for_pair(fused_p, fused_m, df_index, pmd_params['rdi_params'], "PMD_RDI", "P_M")
            base_div = base_div * (1 + rdi_score * pmd_params['rdi_params']['rdi_modulator_weight']).clip(0.5, 1.5)
        # 指数化增强与死区裁剪
        final_score = np.sign(base_div) * (base_div.abs().pow(pmd_params['final_fusion_exponent']))
        if get_param_value(self.helper.debug_params.get('enabled'), False):
            probe_ts = df.index[-1]
            diag = {**p_debug, **v_debug, **m_debug, **sync_debug, "final_score": final_score}
            self._print_debug_output_pmd(diag, probe_ts, method_name, final_score)
        return final_score.clip(-1, 1).fillna(0.0).astype(np.float32)



