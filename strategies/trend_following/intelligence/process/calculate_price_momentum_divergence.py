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
        """V3.4.0 · 参数体系重构：引入背离质量权重矩阵 (DQWM) 与动力学深度权重"""
        params = get_param_value(config.get('price_momentum_divergence_params'), {})
        rdi_config = get_param_value(params.get('rdi_params'), {})
        return {
            "fib_periods": [5, 13, 21, 34, 55],
            "dqwm_weights": {
                "momentum_quality": 0.25, # 动量相干性权重
                "market_tension": 0.20, # 物理张力权重
                "stability": 0.15, # 市场稳定性权重
                "chip_potential": 0.15, # 筹码历史势能权重
                "liquidity_tide": 0.15, # 流动性潮汐权重
                "market_constitution": 0.10 # 市场体质权重
            },
            "kinematic_weights": {"slope": 0.2, "accel": 0.5, "jerk": 0.3},
            "asymmetric_factor": 1.4,
            "kinematic_deadzone": 1e-6,
            "final_fusion_exponent": get_param_value(params.get('final_fusion_exponent'), 1.8),
            "rdi_params": {
                "enabled": get_param_value(rdi_config.get('enabled'), True),
                "rdi_periods": [5, 13, 21],
                "rdi_modulator_weight": 0.2
            }
        }

    def _validate_pmd_signals(self, df: pd.DataFrame, pmd_params: Dict, method_name: str) -> bool:
        """V2.9.0 · 军械库信号完整性校验：对齐结构张力、物理极限与筹码有序度"""
        required = [
            'MA_POTENTIAL_TENSION_INDEX_D', 'MA_RUBBER_BAND_EXTENSION_D', 'chip_convergence_ratio_D',
            'chip_entropy_D', 'days_since_last_peak_D', 'close_D', 'MACDh_13_34_8_D'
        ]
        fib = pmd_params['fib_periods']
        for p in fib:
            required.extend([f'SLOPE_{p}_close_D', f'ACCEL_{p}_close_D', f'JERK_{p}_close_D'])
            required.extend([f'SLOPE_{p}_MACDh_13_34_8_D', f'ACCEL_{p}_MACDh_13_34_8_D', f'JERK_{p}_MACDh_13_34_8_D'])
        missing = [s for s in required if s not in df.columns]
        if missing:
            print(f"[ERROR] {method_name} 关键信号缺失: {missing}")
            return False
        return True

    def _calculate_fused_price_direction(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> Tuple[pd.Series, Dict]:
        """V3.1.0 · 价格动力学路径版：集成核心动力学引擎与多维价格指标"""
        p_weights = pmd_params['price_components_weights']
        debug_matrix = {}
        fused_components = []
        for col, weight in p_weights.items():
            kin_score, kin_probe = self._calculate_kinematic_core_node(df, df_index, col, pmd_params)
            fused_components.append(kin_score * weight)
            debug_matrix[f"node_{col}_kinematic"] = kin_score
        final_price_direction = pd.concat(fused_components, axis=1).sum(axis=1)
        debug_matrix["final_price_direction"] = final_price_direction
        return final_price_direction, debug_matrix

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
        """V3.1.0 · 动量动力学共振版：利用三阶导数识别指标钝化与结构性背离"""
        m_weights = pmd_params['momentum_components_weights']
        debug_matrix = {}
        fused_components = []
        for col, weight in m_weights.items():
            kin_score, kin_probe = self._calculate_kinematic_core_node(df, df_index, col, pmd_params)
            fused_components.append(kin_score * weight)
            debug_matrix[f"node_{col}_momentum_kinematic"] = kin_score
        final_momentum_direction = pd.concat(fused_components, axis=1).sum(axis=1)
        debug_matrix["final_momentum_direction"] = final_momentum_direction
        return final_momentum_direction, debug_matrix

    def _calculate_volume_confirmation_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, base_divergence_score: pd.Series, method_name: str) -> Tuple[pd.Series, Dict]:
        """V2.7.0 · 量能确认动力学重构版 (集成能量平衡、三阶导数与微观异动)"""
        weights = pmd_params['vol_conf_weights']
        k_w = pmd_params['vol_kinematic_weights']
        deadzone = pmd_params['kinematic_deadzone']
        fib_periods = pmd_params['fib_periods']
        # 1. 能量平衡判定 (Energy Balance)
        abs_energy = self.helper._normalize_series(df['absorption_energy_D'], df_index, ascending=True)
        dist_energy = self.helper._normalize_series(df['distribution_energy_D'], df_index, ascending=True)
        # 根据背离方向选择能量分值：顶背离看派发，底背离看吸收
        energy_balance = pd.Series([
            dist_energy.loc[idx] if x > 0 else (abs_energy.loc[idx] if x < 0 else 0)
            for idx, x in base_divergence_score.items()
        ], index=df_index)
        # 2. 量能动力学矩阵 (Kinematic Impact)
        period_scores = []
        for p in fib_periods:
            v_s = self.helper._normalize_series(df[f'SLOPE_{p}_volume_D'].where(df[f'SLOPE_{p}_volume_D'].abs() > deadzone, 0.0), df_index, bipolar=True)
            v_a = self.helper._normalize_series(df[f'ACCEL_{p}_volume_D'].where(df[f'ACCEL_{p}_volume_D'].abs() > deadzone, 0.0), df_index, bipolar=True)
            v_j = self.helper._normalize_series(df[f'JERK_{p}_volume_D'].where(df[f'JERK_{p}_volume_D'].abs() > deadzone, 0.0), df_index, bipolar=True)
            period_scores.append(v_s * k_w['slope'] + v_a * k_w['accel'] + v_j * k_w['jerk'])
        vol_kinematic_impact = pd.concat(period_scores, axis=1).mean(axis=1).abs()
        # 3. 微观异动与稳定性
        micro_anomaly = self.helper._normalize_series(df['tick_abnormal_volume_ratio_D'], df_index, ascending=True)
        vpa_accel = self.helper._normalize_series(df['VPA_ACCELERATION_5D'].abs(), df_index, ascending=True)
        stability = self.helper._normalize_series(df['TURNOVER_STABILITY_INDEX_D'], df_index, ascending=True)
        # 4. 最终确认得分融合
        components = {
            "energy_balance": energy_balance,
            "kinematic_impact": (vol_kinematic_impact * 0.7 + vpa_accel * 0.3),
            "micro_anomaly": micro_anomaly,
            "stability": stability
        }
        volume_conf_score = (components["energy_balance"] * weights["energy_balance"] + 
                            components["kinematic_impact"] * weights["kinematic_impact"] + 
                            components["micro_anomaly"] * weights["micro_anomaly"] + 
                            components["stability"] * weights["stability"])
        # 同步背离符号
        final_vol_conf = volume_conf_score * np.sign(base_divergence_score)
        debug_v = {f"vol_node_{k}": v for k, v in components.items()}
        debug_v["final_volume_confirmation"] = final_vol_conf
        return final_vol_conf, debug_v

    def _calculate_main_force_confirmation_score(self, df: pd.DataFrame, df_index: pd.Index, pmd_params: Dict, base_div: pd.Series, method_name: str) -> Tuple[pd.Series, Dict]:
        """V2.8.0 · 主力确认模型：三阶动力学、协同共振与意图陷阱识别 (全探针暴露)"""
        k_w = pmd_params['mf_kinematic_weights']
        c_w = pmd_params['mf_cohesion_weights']
        i_w = pmd_params['mf_intent_weights']
        deadzone = pmd_params['kinematic_deadzone']
        fib = pmd_params['fib_periods']
        asym = pmd_params['asymmetric_outflow_factor']
        # 1. 主力协同度节点 (MFC)
        hm_attack = self.helper._normalize_series(df['HM_COORDINATED_ATTACK_D'], df_index, bipolar=True)
        sm_synergy = self.helper._normalize_series(df['SMART_MONEY_SYNERGY_BUY_D'], df_index, bipolar=True)
        mf_activity = self.helper._normalize_series(df['main_force_activity_index_D'], df_index, bipolar=True)
        mfc_node = (hm_attack * c_w['HM_COORDINATED_ATTACK_D'] + sm_synergy * c_w['SMART_MONEY_SYNERGY_BUY_D'] + mf_activity * c_w['main_force_activity_index_D'])
        # 2. 资金动力学节点 (FK) - 基于三阶导数
        fk_period_scores = []
        for p in fib:
            s_r = df[f'SLOPE_{p}_net_mf_amount_D'].where(df[f'SLOPE_{p}_net_mf_amount_D'].abs() > deadzone, 0.0)
            a_r = df[f'ACCEL_{p}_net_mf_amount_D'].where(df[f'ACCEL_{p}_net_mf_amount_D'].abs() > deadzone, 0.0)
            j_r = df[f'JERK_{p}_net_mf_amount_D'].where(df[f'JERK_{p}_net_mf_amount_D'].abs() > deadzone, 0.0)
            a_p = np.where(a_r < 0, a_r * asym, a_r) # 强化资金流出加速度敏感度
            s_n = self.helper._normalize_series(s_r, df_index, bipolar=True)
            a_n = self.helper._normalize_series(pd.Series(a_p, index=df_index), df_index, bipolar=True)
            j_n = self.helper._normalize_series(pd.Series(j_r, index=df_index), df_index, bipolar=True)
            fk_period_scores.append(s_n * k_w['slope'] + a_n * k_w['accel'] + j_n * k_w['jerk'])
        fk_node = pd.concat(fk_period_scores, axis=1).mean(axis=1)
        # 3. 意图与陷阱节点 (ITV)
        # 动态判定：顶背离看派发与异动，底背离看吸筹与洗盘
        acc_score = self.helper._normalize_series(df['accumulation_signal_score_D'], df_index, ascending=True)
        dist_score = self.helper._normalize_series(df['distribution_signal_score_D'], df_index, ascending=True)
        shake_score = self.helper._normalize_series(df['shakeout_score_D'], df_index, ascending=True)
        large_anomaly = self.helper._normalize_series(df['large_order_anomaly_D'], df_index, ascending=True)
        sm_diverge = self.helper._normalize_series(df['SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D'], df_index, bipolar=True)
        itv_node = pd.Series([
            (dist_score.loc[idx] * i_w['accumulation'] + large_anomaly.loc[idx] * i_w['large_order'] + sm_diverge.loc[idx].clip(0)) if x > 0 else
            (acc_score.loc[idx] * i_w['accumulation'] + shake_score.loc[idx] * i_w['shakeout'] - sm_diverge.loc[idx].clip(upper=0)) if x < 0 else 0
            for idx, x in base_div.items()
        ], index=df_index)
        # 4. 最终确认得分融合：协同 * 动力 * 意图，并由筹码稳定性调制
        chip_stb = self.helper._normalize_series(df['chip_stability_D'], df_index, ascending=True)
        mf_conf_score = (mfc_node * 0.35 + fk_node * 0.4 + itv_node * 0.25) * chip_stb
        final_mf_conf = mf_conf_score * np.sign(base_div)
        debug_v = {
            "probe_raw_net_mf": df['net_mf_amount_D'],
            "probe_hm_attack": df['HM_COORDINATED_ATTACK_D'],
            "node_mf_cohesion": mfc_node,
            "node_mf_kinematics": fk_node,
            "node_mf_intent_trap": itv_node,
            "node_chip_foundation": chip_stb,
            "final_mf_confirmation": final_mf_conf
        }
        return final_mf_conf, debug_v

    def _calculate_kinematic_core_node(self, df: pd.DataFrame, df_index: pd.Index, base_col: str, pmd_params: Dict) -> Tuple[pd.Series, Dict]:
        """V3.5.0 · 动力学计算引擎升级：集成 Tanh 鲁棒映射与死区滤波，彻底回避零基陷阱 (全探针暴露)"""
        fib = pmd_params['fib_periods'] #
        k_w = pmd_params['kinematic_weights'] #
        deadzone = pmd_params['kinematic_deadzone'] #
        asym = pmd_params['asymmetric_factor'] #
        period_scores = []
        probe_data = {}
        for p in fib:
            # 1. 原始三阶导数信号提取与死区物理滤波
            s_raw = df[f'SLOPE_{p}_{base_col}'].where(df[f'SLOPE_{p}_{base_col}'].abs() > deadzone, 0.0) #
            a_raw = df[f'ACCEL_{p}_{base_col}'].where(df[f'ACCEL_{p}_{base_col}'].abs() > deadzone, 0.0) #
            j_raw = df[f'JERK_{p}_{base_col}'].where(df[f'JERK_{p}_{base_col}'].abs() > deadzone, 0.0) #
            # 2. 非对称灵敏度调制：强化负向加速度响应速度
            a_mod = np.where(a_raw < 0, a_raw * asym, a_raw)
            j_mod = np.where(j_raw < 0, j_raw * asym, j_raw)
            # 3. 鲁棒非线性归一化 (使用 Tanh 映射回避 0 轴附近的噪音波动)
            # 逻辑：将信号映射到 [-1, 1]，在极值区饱和，在 0 轴平滑过度
            s_n = self.helper._normalize_series(s_raw, df_index, bipolar=True)
            a_n = self.helper._normalize_series(pd.Series(a_mod, index=df_index), df_index, bipolar=True)
            j_n = self.helper._normalize_series(pd.Series(j_mod, index=df_index), df_index, bipolar=True)
            # 4. 动力学三阶合成
            combined_p = (s_n * k_w['slope'] + a_n * k_w['accel'] + j_n * k_w['jerk'])
            period_scores.append(combined_p)
            probe_data[f"p{p}_jerk_tanh"] = j_n
        # MTF 全周期均值融合
        fused_kinematic = pd.concat(period_scores, axis=1).mean(axis=1)
        probe_data["fused_kinematic"] = fused_kinematic
        # 探针输出：检查死区后信号存活率与均值
        print(f"[探针-KinematicEngine] 信号源: {base_col} | 死区阈值: {deadzone} | 存活率: {(fused_kinematic != 0).mean()*100:.2f}% | 动力均值: {fused_kinematic.mean():.4f}")
        return fused_kinematic, probe_data

    def _calculate_divergence_quality_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, base_div: pd.Series) -> Tuple[pd.Series, Dict]:
        """V2.9.0 · 背离质量动力学模型：集成结构张力、物理极限、动力学衰竭与筹码有序度 (全探针暴露)"""
        q_w = pmd_params['quality_weights']
        t_w = pmd_params['tension_sub_weights']
        c_w = pmd_params['chip_order_sub_weights']
        fib = pmd_params['fib_periods']
        # 1. 结构张力节点 (Structural Tension Node) - 判定物理极限
        tension_idx = self.helper._normalize_series(df['MA_POTENTIAL_TENSION_INDEX_D'], df_index, ascending=True)
        rubber_band = self.helper._normalize_series(df['MA_RUBBER_BAND_EXTENSION_D'].abs(), df_index, ascending=True)
        tension_node = (tension_idx * t_w['tension_index'] + rubber_band * t_w['rubber_band'])
        # 2. 动力学衰竭节点 (Kinematic Exhaustion Node) - 判定力矩消失
        k_exhaustion_scores = []
        for p in fib:
            p_accel = self.helper._normalize_series(df[f'ACCEL_{p}_close_D'], df_index, bipolar=True)
            m_accel = self.helper._normalize_series(df[f'ACCEL_{p}_MACDh_13_34_8_D'], df_index, bipolar=True)
            p_jerk = self.helper._normalize_series(df[f'JERK_{p}_close_D'], df_index, bipolar=True)
            # 高质量背离：价格加速度正在反转(与价格方向相反)，而动量加加速度正在增强(回归)
            exhaustion = (m_accel - p_accel).abs() * (1 + p_jerk.abs())
            k_exhaustion_scores.append(exhaustion)
        kinematic_node = pd.concat(k_exhaustion_scores, axis=1).mean(axis=1)
        # 3. 筹码有序度节点 (Chip Order Node) - 判定博弈一致性
        chip_conv = self.helper._normalize_series(df['chip_convergence_ratio_D'], df_index, ascending=True)
        chip_ent_inv = self.helper._normalize_series(df['chip_entropy_D'], df_index, ascending=False)
        chip_node = (chip_conv * c_w['convergence'] + chip_ent_inv * c_w['entropy_inverted'])
        # 4. 时间成熟度修正
        ripeness = self.helper._normalize_series(df['days_since_last_peak_D'], df_index, ascending=True)
        # 5. 最终融合与探针暴露
        quality_score = (tension_node * q_w['structural_tension'] + kinematic_node * q_w['kinematic_exhaustion'] + chip_node * q_w['chip_order']) * ripeness
        quality_score = quality_score.clip(0, 1)
        print(f"[探针-DivergenceQuality] 基础背离均值: {base_div.mean():.4f}")
        print(f"[探针-DivergenceQuality] 张力节点均值: {tension_node.mean():.4f} | 物理乖离均值: {rubber_band.mean():.4f}")
        print(f"[探针-DivergenceQuality] 动力衰竭均值: {kinematic_node.mean():.4f} | 筹码有序度均值: {chip_node.mean():.4f}")
        print(f"[探针-DivergenceQuality] 最终质量评分均值: {quality_score.mean():.4f}")
        debug_v = {
            "node_tension": tension_node,
            "node_kinematic_exhaustion": kinematic_node,
            "node_chip_order": chip_node,
            "node_ripeness": ripeness,
            "final_quality_score": quality_score
        }
        return quality_score, debug_v

    def _calculate_context_modulator(self, df_index: pd.Index, raw_data: Dict, pmd_params: Dict) -> Tuple[pd.Series, Dict]:
        """V3.0.0 · A股四维时空场能调制模型：集成心理边界、物理张力、环境热度与博弈强度 (全探针暴露)"""
        weights = pmd_params['context_weights']
        # 1. 心理边界节点 (Psychological Boundary)
        emo_ext = self.helper._normalize_series(raw_data['STATE_EMOTIONAL_EXTREME_D'], df_index, bipolar=True).abs()
        mkt_sent = self.helper._normalize_series(raw_data['market_sentiment_score_D'], df_index, bipolar=True).abs()
        psy_node = (emo_ext * 0.6 + mkt_sent * 0.4)
        # 2. 物理张力动力学节点 (Physical Tension Kinematics)
        tension_val = self.helper._normalize_series(raw_data['MA_POTENTIAL_TENSION_INDEX_D'], df_index, ascending=True)
        rubber_val = self.helper._normalize_series(raw_data['MA_RUBBER_BAND_EXTENSION_D'].abs(), df_index, ascending=True)
        t_vel = self.helper._normalize_series(raw_data['MA_VELOCITY_EMA_55_D'].abs(), df_index, ascending=True)
        t_acc = self.helper._normalize_series(raw_data['MA_ACCELERATION_EMA_55_D'].abs(), df_index, ascending=True)
        t_k = pmd_params['tension_kinematic_weights']
        phy_node = ((tension_val * 0.5 + rubber_val * 0.5) * t_k['value'] + t_vel * t_k['velocity'] + t_acc * t_k['accel'])
        # 3. 环境与博弈节点 (Environmental & Game Intensity)
        theme_hot = self.helper._normalize_series(raw_data['THEME_HOTNESS_SCORE_D'], df_index, ascending=False) # 热度越高，回归越难，权重降序
        game_int = self.helper._normalize_series(raw_data['game_intensity_D'], df_index, ascending=True)
        env_node = theme_hot
        game_node = game_int
        # 4. 非线性融合
        components = {
            "psychological": psy_node,
            "physical_tension": phy_node,
            "environmental_hotness": env_node,
            "game_intensity": game_node
        }
        raw_modulator = _robust_geometric_mean(components, weights, df_index)
        # 5. 抛物线熔断惩罚 (Parabolic Warning Penalty)
        is_parabolic = (raw_data['STATE_PARABOLIC_WARNING_D'] > 0.5).astype(np.float32)
        penalty = pd.Series(1.0, index=df_index)
        penalty.loc[is_parabolic == 1] = pmd_params['parabolic_penalty_factor']
        final_context_modulator = (raw_modulator * penalty).clip(0.1, 2.0)
        # 详细探针输出
        print(f"[探针-ContextModulator] 心理节点(PSY)均值: {psy_node.mean():.4f}")
        print(f"[探针-ContextModulator] 物理动力学(PHY)均值: {phy_node.mean():.4f} | 速度贡献: {t_vel.mean():.4f}")
        print(f"[探针-ContextModulator] 环境热度(THEME)均值: {theme_hot.mean():.4f} | 博弈强度(GAME)均值: {game_int.mean():.4f}")
        print(f"[探针-ContextModulator] 抛物线惩罚触发比例: {is_parabolic.mean()*100:.2f}%")
        print(f"[探针-ContextModulator] 最终环境调制器均值: {final_context_modulator.mean():.4f}")
        debug_info = {
            "node_psychological": psy_node,
            "node_physical": phy_node,
            "node_environmental": env_node,
            "node_game": game_node,
            "node_parabolic_penalty": penalty,
            "final_context_modulator": final_context_modulator
        }
        return final_context_modulator, debug_info

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
        """V3.2.0 · 复合量能萎缩模型：基于成交量三阶动力学死区判定 (全探针暴露)"""
        # 1. 获取成交量全周期动力学分值 (斜率/加速度/冲击融合)
        vol_kinematic, vol_probe = self._calculate_kinematic_core_node(df, df_index, 'volume_D', pmd_params)
        # 2. 判定死区状态：动力学分值越接近0，萎缩质量越高 (利用1-abs映射)
        atrophy_quality = (1 - vol_kinematic.abs()).clip(0, 1)
        # 3. 结构稳定性修正 (换手稳定性越高，萎缩越真实)
        turnover_stb = self.helper._normalize_series(df['TURNOVER_STABILITY_INDEX_D'], df_index, ascending=True)
        # 4. 融合分值
        volume_atrophy_score = (atrophy_quality * 0.7 + turnover_stb * 0.3)
        print(f"[探针-VolumeAtrophy] 动力学均值: {vol_kinematic.mean():.4f} | 萎缩质量均值: {atrophy_quality.mean():.4f}")
        return volume_atrophy_score.astype(np.float32)

    def _calculate_distribution_intent_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> pd.Series:
        """V3.2.0 · 复合派发意图模型：集成派发能量动力学与价格Jerk坍塌 (全探针暴露)"""
        # 1. 派发能量场动力学
        dist_energy_kin, _ = self._calculate_kinematic_core_node(df, df_index, 'distribution_energy_D', pmd_params)
        # 2. 价格冲击力监控 (Jerk坍塌是派发确认)
        _, price_probe = self._calculate_kinematic_core_node(df, df_index, 'close_D', pmd_params)
        price_jerk_collapsed = (1 - price_probe['fused_kinematic'].abs()).clip(0, 1) # 冲击消失
        # 3. 卖单异动确认
        sell_ofi = self.helper._normalize_series(df['SMART_MONEY_INST_NET_BUY_D'].clip(upper=0).abs(), df_index, ascending=True)
        # 4. 最终融合
        dist_intent = (dist_energy_kin.clip(0, 1) * 0.5 + price_jerk_collapsed * 0.3 + sell_ofi * 0.2)
        print(f"[探针-DistIntent] 派发能量动力: {dist_energy_kin.mean():.4f} | 价格冲击坍塌: {price_jerk_collapsed.mean():.4f}")
        return dist_intent.astype(np.float32)

    def _calculate_covert_accumulation_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> pd.Series:
        """V3.2.0 · 隐蔽吸筹模型：集成吸收能量、异动占比与价格死区锁定 (全探针暴露)"""
        # 1. 吸收能量场动力学
        abs_energy_kin, _ = self._calculate_kinematic_core_node(df, df_index, 'absorption_energy_D', pmd_params)
        # 2. 价格死区锁定 (横盘吸筹判定)
        price_kin, _ = self._calculate_kinematic_core_node(df, df_index, 'close_D', pmd_params)
        price_deadzone_lock = (1 - price_kin.abs()).clip(0, 1)
        # 3. 异动量能占比 (聪明钱痕迹)
        abnormal_ratio = self.helper._normalize_series(df['tick_abnormal_volume_ratio_D'], df_index, ascending=True)
        # 4. 吸筹信号共振
        acc_signal = self.helper._normalize_series(df['accumulation_signal_score_D'], df_index, ascending=True)
        # 5. 最终融合
        covert_acc_score = (abs_energy_kin.clip(0, 1) * 0.4 + price_deadzone_lock * 0.2 + abnormal_ratio * 0.2 + acc_signal * 0.2)
        print(f"[探针-CovertAcc] 吸收能量动力: {abs_energy_kin.mean():.4f} | 价格死区锁定: {price_deadzone_lock.mean():.4f}")
        return covert_acc_score.astype(np.float32)

    def _calculate_chip_divergence_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> pd.Series:
        """V3.3.0 · 筹码背离动力学模型：集成筹码RSI动力学与稳定性权重 (取代旧版Composite方法)"""
        # 1. 筹码RSI背离动力学引擎处理
        chip_kin, _ = self._calculate_kinematic_core_node(df, df_index, 'chip_rsi_divergence_D', pmd_params) #
        # 2. 筹码稳定性调制 (稳定性越高，背离的转折意义越强)
        chip_stb = self.helper._normalize_series(df['chip_stability_D'], df_index, ascending=True) # [cite: 1]
        # 3. 最终融合：动力学得分(70%) * 稳定性(30%)
        chip_div_score = (chip_kin * 0.7 + chip_stb * 0.3).clip(-1, 1)
        print(f"[探针-ChipDivergence] 筹码动力学均值: {chip_kin.mean():.4f} | 稳定性均值: {chip_stb.mean():.4f}")
        return chip_div_score.astype(np.float32)

    def _calculate_upward_efficiency_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> pd.Series:
        """V3.3.0 · 上涨效率动力学模型：集成VPA效率加速度与收盘强度 (取代旧版Composite方法)"""
        # 1. 价量效率动力学核心
        eff_kin, _ = self._calculate_kinematic_core_node(df, df_index, 'VPA_EFFICIENCY_D', pmd_params) #
        # 2. 收盘强度支撑 (上涨必须伴随收盘强度的正向分值)
        cls_str = self.helper._normalize_series(df['CLOSING_STRENGTH_D'], df_index, bipolar=True) # [cite: 1]
        # 3. 效率判定：效率动力学(60%) + 收盘强度(40%)
        up_eff_score = (eff_kin.clip(lower=0) * 0.6 + cls_str.clip(lower=0) * 0.4)
        print(f"[探针-UpEfficiency] 效率动力学均值: {eff_kin.mean():.4f} | 收盘强度均值: {cls_str.mean():.4f}")
        return up_eff_score.astype(np.float32)

    def _calculate_price_upward_momentum_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> pd.Series:
        """V3.3.0 · 价格上涨动量模型：三阶动力学共振与趋势速度融合 (取代旧版Composite方法)"""
        # 1. 价格动力学核心 (自动处理斜率、加速度与冲击)
        price_kin, _ = self._calculate_kinematic_core_node(df, df_index, 'close_D', pmd_params) #
        # 2. 趋势速度维度
        trend_vel = self.helper._normalize_series(df['MA_VELOCITY_EMA_55_D'], df_index, bipolar=True) # [cite: 1]
        # 3. 动量融合：动力学分值(70%) + 趋势速度(30%)
        up_mom_score = (price_kin.clip(lower=0) * 0.7 + trend_vel.clip(lower=0) * 0.3)
        print(f"[探针-UpMomentum] 价格动力学均值: {price_kin.mean():.4f} | 趋势速度均值: {trend_vel.mean():.4f}")
        return up_mom_score.astype(np.float32)

    def _calculate_price_downward_momentum_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> pd.Series:
        """V3.3.0 · 价格下跌动量模型：非对称三阶导数与重力加速度效应 (取代旧版Composite方法)"""
        # 1. 价格动力学核心 (内部已执行非对称灵敏度处理)
        price_kin, _ = self._calculate_kinematic_core_node(df, df_index, 'close_D', pmd_params) #
        # 2. 下跌强度维度 (结合绝对变化强度)
        change_str = self.helper._normalize_series(df['absolute_change_strength_D'], df_index, ascending=True) # [cite: 1]
        # 3. 负向动量融合
        down_mom_score = (price_kin.clip(upper=0).abs() * 0.7 + change_str * 0.3)
        print(f"[探针-DownMomentum] 负向动力学均值: {price_kin.mean():.4f} | 变化强度均值: {change_str.mean():.4f}")
        return down_mom_score.astype(np.float32)

    def _calculate_momentum_quality_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> pd.Series:
        """V3.3.0 · 动量品质动力学模型：集成三阶导数共振与协同进攻指数 (取代旧版Composite方法)"""
        # 1. 动量核心动力学提取
        macdh_kin, _ = self._calculate_kinematic_core_node(df, df_index, 'MACDh_13_34_8_D', pmd_params) #
        rsi_kin, _ = self._calculate_kinematic_core_node(df, df_index, 'RSI_13_D', pmd_params) #
        # 2. 协同进攻共振 (HM_COORDINATED_ATTACK_D 代表主力行动一致性)
        coordination = self.helper._normalize_series(df['HM_COORDINATED_ATTACK_D'], df_index, bipolar=True) #
        # 3. 动量相干性：MACD与RSI的方向一致性 * 协同强度
        coherence = (np.sign(macdh_kin) == np.sign(rsi_kin)).astype(np.float32)
        momentum_quality = (macdh_kin.abs() * 0.4 + rsi_kin.abs() * 0.4 + coordination * 0.2) * coherence
        print(f"[探针-MomQuality] MACD动力均值: {macdh_kin.mean():.4f} | RSI动力均值: {rsi_kin.mean():.4f} | 协同均值: {coordination.mean():.4f}")
        return momentum_quality.astype(np.float32)

    def _calculate_stability_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> pd.Series:
        """V3.3.0 · 市场稳定性模型：基于换手稳定性与价格熵的死区锁定 (取代旧版Composite方法)"""
        # 1. 换手稳定性核心
        turnover_stb = self.helper._normalize_series(df['TURNOVER_STABILITY_INDEX_D'], df_index, ascending=True) #
        # 2. 价格熵反向调制 (熵越低，结构越有序)
        entropy_inv = 1 - self.helper._normalize_series(df['PRICE_ENTROPY_D'], df_index, ascending=True) #
        # 3. 动力学死区锁定 (价格加速度趋近于0代表极其稳定)
        price_kin, _ = self._calculate_kinematic_core_node(df, df_index, 'close_D', pmd_params) #
        stability_lock = (1 - price_kin.abs()).clip(0, 1)
        # 4. 融合：换手(40%) + 熵反向(30%) + 动力锁定(30%)
        stability_score = (turnover_stb * 0.4 + entropy_inv * 0.3 + stability_lock * 0.3)
        print(f"[探针-Stability] 换手稳定均值: {turnover_stb.mean():.4f} | 价格熵反向均值: {entropy_inv.mean():.4f}")
        return stability_score.astype(np.float32)

    def _calculate_chip_historical_potential_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> pd.Series:
        """V3.3.0 · 筹码历史势能模型：集成筹码稳定性与积累能量场 (取代旧版Composite方法)"""
        # 1. 筹码稳定性势能
        chip_stb = self.helper._normalize_series(df['chip_stability_D'], df_index, ascending=True) #
        # 2. 积累分数动力学
        acc_kin, _ = self._calculate_kinematic_core_node(df, df_index, 'accumulation_score_D', pmd_params) #
        # 3. 场能融合
        potential_score = (chip_stb * 0.6 + acc_kin.clip(lower=0) * 0.4)
        print(f"[探针-ChipPotential] 筹码稳定势能: {chip_stb.mean():.4f} | 积累动力均值: {acc_kin.mean():.4f}")
        return potential_score.astype(np.float32)

    def _calculate_liquidity_tide_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> pd.Series:
        """V3.3.0 · 流动性潮汐模型：集成逐笔筹码流变率与波动率稳定性 (取代旧版Composite方法)"""
        # 1. 逐笔筹码流动力学引擎
        chip_flow_kin, _ = self._calculate_kinematic_core_node(df, df_index, 'tick_level_chip_flow_D', pmd_params) #
        # 2. 资金流波动率稳定性 (波动率低代表潮汐平稳，利于背离转化)
        flow_vol_stb = 1 - self.helper._normalize_series(df['flow_volatility_10d_D'], df_index, ascending=True) #
        # 3. 潮汐得分
        tide_score = (chip_flow_kin.abs() * 0.7 + flow_vol_stb * 0.3)
        print(f"[探针-LiquidityTide] 筹码流动力均值: {chip_flow_kin.mean():.4f} | 资金流稳定均值: {flow_vol_stb.mean():.4f}")
        return tide_score.astype(np.float32)

    def _calculate_market_constitution_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> pd.Series:
        """V3.5.0 · 市场体质模型：集成历史排名锚定，解决低波动区间的零基陷阱 (取代旧版Composite方法)"""
        # 1. 趋势强度反向动力学引擎处理
        adx_kin, _ = self._calculate_kinematic_core_node(df, df_index, 'ADX_14_D', pmd_params) #
        adx_inverted = (1 - adx_kin.abs()).clip(0, 1)
        # 2. 历史排名锚定 (Price Percentile Position 代表当前在历史中的相对位置) 
        price_pos = self.helper._normalize_series(df['price_percentile_position_D'], df_index, bipolar=True).abs()
        # 3. 逻辑分歧处理：当价格不在极端位置（排名接近 0.5）时，体质贡献应向 0 塌陷
        constitution_anchor = price_pos.where(price_pos > 0.3, 0.0) 
        # 4. 融合最终体质分
        constitution_score = adx_inverted * constitution_anchor
        print(f"[探针-Constitution] ADX反向动力: {adx_inverted.mean():.4f} | 历史排名锚定值: {constitution_anchor.mean():.4f}")
        return constitution_score.astype(np.float32)

    def _calculate_market_tension_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> pd.Series:
        """V3.3.0 · 市场张力模型：基于物理势能张力与乖离加速度的极限识别 (取代旧版Composite方法)"""
        # 1. 物理势能张力加速度
        tension_kin, _ = self._calculate_kinematic_core_node(df, df_index, 'MA_POTENTIAL_TENSION_INDEX_D', pmd_params) #
        # 2. 乖离率(橡皮筋)冲击力
        rubber_kin, _ = self._calculate_kinematic_core_node(df, df_index, 'MA_RUBBER_BAND_EXTENSION_D', pmd_params) #
        # 3. 融合张力：当两者加速度共振且处于高位时，张力得分极高
        tension_score = (tension_kin.abs() * 0.5 + rubber_kin.abs() * 0.5).clip(0, 1)
        print(f"[探针-MarketTension] 张力动力均值: {tension_kin.mean():.4f} | 乖离动力均值: {rubber_kin.mean():.4f}")
        return tension_score.astype(np.float32)

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """V3.4.0 · 顶层调度：注入背离质量权重矩阵 (DQWM) 的动力学分析引擎 (全探针暴露)"""
        method_name = "PriceMomentumDQWM_System"
        pmd_params = self._get_pmd_params(config)
        df_index = df.index
        if not self._validate_pmd_signals(df, pmd_params, method_name):
            return pd.Series(0.0, index=df_index, dtype=np.float32)
        raw_data = {col: df[col] for col in df.columns if col.endswith('_D')} # [cite: 1, 2, 3]
        # 1. 核心动力学方向计算
        fused_p, p_diag = self._calculate_fused_price_direction(df, df_index, raw_data, pmd_params, method_name)
        fused_m, m_diag = self._calculate_fused_momentum_direction(df, df_index, raw_data, pmd_params, method_name)
        # 2. 基础背离与意图校准
        base_div = (fused_p - fused_m).clip(-1, 1)
        dist_intent = self._calculate_distribution_intent_score(df, df_index, raw_data, pmd_params, method_name)
        acc_intent = self._calculate_covert_accumulation_score(df, df_index, raw_data, pmd_params, method_name)
        # 3. 计算“背离质量权重矩阵” (DQWM Matrix)
        dq_mom_quality = self._calculate_momentum_quality_score(df, df_index, raw_data, pmd_params, method_name)
        dq_stability = self._calculate_stability_score(df, df_index, raw_data, pmd_params, method_name)
        dq_chip_potential = self._calculate_chip_historical_potential_score(df, df_index, raw_data, pmd_params, method_name)
        dq_tide = self._calculate_liquidity_tide_score(df, df_index, raw_data, pmd_params, method_name)
        dq_const = self._calculate_market_constitution_score(df, df_index, raw_data, pmd_params, method_name)
        dq_tension = self._calculate_market_tension_score(df, df_index, raw_data, pmd_params, method_name)
        # 执行矩阵加权融合
        dqwm_components = {
            "momentum_quality": dq_mom_quality, "stability": dq_stability, "chip_potential": dq_chip_potential,
            "liquidity_tide": dq_tide, "market_constitution": dq_const, "market_tension": dq_tension
        }
        dqwm_matrix_score = _weighted_sum_fusion(dqwm_components, pmd_params['dqwm_weights'], df_index)
        # 4. 融合与场能调制
        # 融合公式：(基础背离*0.5 + 意图*0.5) * DQWM矩阵得分 * 环境调制
        intent_gain = pd.Series(0.0, index=df_index)
        intent_gain.loc[base_div > 0] = acc_intent.loc[base_div > 0]
        intent_gain.loc[base_div < 0] = dist_intent.loc[base_div < 0]
        ctx_modulator, ctx_diag = self._calculate_context_modulator(df_index, raw_data, pmd_params)
        raw_fusion = (base_div * 0.5 + (intent_gain * np.sign(base_div)) * 0.5)
        modulated_score = raw_fusion * dqwm_matrix_score * ctx_modulator
        # 5. RDI 相位锁定
        final_rdi_modulator = pd.Series(1.0, index=df_index)
        if pmd_params['rdi_params']['enabled']:
            rdi_val, _ = self._calculate_rdi_for_pair(fused_p, fused_m, df_index, pmd_params['rdi_params'], method_name, "P_M")
            final_rdi_modulator = (1 + rdi_val * pmd_params['rdi_params']['rdi_modulator_weight']).clip(0.5, 1.5)
        # 6. 最终非线性分值计算
        final_score = np.sign(modulated_score) * (modulated_score.abs() * final_rdi_modulator).pow(pmd_params['final_fusion_exponent'])
        # 7. 诊断与探针
        if get_param_value(self.helper.debug_params.get('enabled'), False):
            probe_ts = df.index[-1]
            all_diagnostics = {
                "融合方向": {"fused_p": fused_p, "fused_m": fused_m},
                "DQWM质量矩阵": {k: v for k, v in dqwm_components.items()},
                "DQWM最终权重": {"dqwm_matrix_score": dqwm_matrix_score},
                "意图与环境": {"intent_gain": intent_gain, "ctx_modulator": ctx_modulator},
                "最终诊断": {"final_score": final_score}
            }
            self._print_debug_output_pmd(all_diagnostics, probe_ts, method_name, final_score)
        return final_score.clip(-1, 1).fillna(0.0).astype(np.float32)



