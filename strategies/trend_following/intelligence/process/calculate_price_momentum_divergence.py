# strategies\trend_following\intelligence\process\calculate_price_momentum_divergence.py
# 【V1.0.0 · 价格动量背离计算器】 计算“价格动量背离”的专属关系分数。 已完成
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
    """V3.15.0 · 鲁棒加权融合：引入基准存活逻辑，防止局部零值导致整体信号坍塌"""
    if not components:
        return pd.Series(0.0, index=index, dtype=np.float32)
    fused_score = pd.Series(0.0, index=index, dtype=np.float32)
    total_weight = pd.Series(0.0, index=index, dtype=np.float32)
    for k, series in components.items():
        if k not in weights: continue
        w = weights[k]
        w_series = w if isinstance(w, pd.Series) else pd.Series(w, index=index)
        # 核心逻辑：只有非零组件才贡献权重分母，实现自适应平滑
        active_mask = (series.abs() > 1e-9)
        fused_score[active_mask] += series[active_mask] * w_series[active_mask]
        total_weight[active_mask] += w_series[active_mask]
    # 对于全零行，total_weight 为 0，结果自然为 0
    return (fused_score / total_weight.clip(lower=1e-9)).fillna(0.0)

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
        """V3.19.0 · 黄金坑参数重构：新增 structural_compensation 补偿系数"""
        params = get_param_value(config.get('price_momentum_divergence_params'), {})
        rdi_config = get_param_value(params.get('rdi_params'), {})
        return {
            "fib_periods": [5, 13, 21, 34, 55],
            "hab_periods": [13, 21, 34],
            "intent_dampening_factor": 0.25,
            # 黄金坑结构解锁参数 (新增)
            "golden_pit_params": {
                "enable_compensation": True,  # 开启补偿
                "compensation_factor": 1.35,  # 多头得分放大系数
                "min_quality_threshold": 0.25 # 允许的最低 DQWM 门槛 (降低门槛以适应左侧)
            },
            "kinematic_weights": {"slope": 0.2, "accel": 0.5, "jerk": 0.3},
            "asymmetric_factor": 1.4,
            "kinematic_deadzone": 1e-6,
            "price_components_weights": {"close_D": 0.4, "CLOSING_STRENGTH_D": 0.3, "VPA_EFFICIENCY_D": 0.3},
            "momentum_components_weights": {"MACDh_13_34_8_D": 0.4, "RSI_13_D": 0.3, "chip_rsi_divergence_D": 0.3},
            "path_sensitivity": {"bullish": 1.0, "bearish": 1.5},
            "dqwm_weights": {"momentum_quality": 0.25, "market_tension": 0.20, "stability": 0.15, "chip_potential": 0.15, "liquidity_tide": 0.15, "market_constitution": 0.10},
            "vol_kinematic_weights": {"slope": 0.2, "accel": 0.5, "jerk": 0.3},
            "vol_conf_weights": {"energy_balance": 0.4, "kinematic_impact": 0.3, "micro_anomaly": 0.2, "stability": 0.1},
            "mf_kinematic_weights": {"slope": 0.2, "accel": 0.5, "jerk": 0.3},
            "mf_cohesion_weights": {"HM_COORDINATED_ATTACK_D": 0.4, "SMART_MONEY_SYNERGY_BUY_D": 0.3, "main_force_activity_index_D": 0.3},
            "mf_intent_weights": {"shakeout": 0.35, "accumulation": 0.35, "large_order": 0.3},
            "quality_weights": {"structural_tension": 0.2, "kinematic_exhaustion": 0.25, "chip_order": 0.15, "volume_ripeness": 0.2, "st_consistency": 0.2},
            "tension_sub_weights": {"tension_index": 0.6, "rubber_band": 0.4},
            "chip_order_sub_weights": {"convergence": 0.6, "entropy_inverted": 0.4},
            "consistency_weights": {"polarity_resonance": 0.4, "pv_coherence": 0.4, "entropy_order": 0.2},
            "context_weights": {"psychological": 0.3, "physical_tension": 0.4, "game_intensity": 0.3},
            "circuit_breaker_params": {"emotion_threshold": 0.75, "tension_confirmation": 0.8, "dampening_factor": 0.4, "amplification_factor": 1.4},
            "asymmetric_outflow_factor": 1.6,
            "conflict_penalty_exponent": 2.5,
            "parabolic_penalty_factor": 0.4,
            "tension_kinematic_weights": {"value": 0.6, "velocity": 0.3, "accel": 0.1},
            "final_fusion_exponent": get_param_value(params.get('final_fusion_exponent'), 1.8),
            "rdi_params": {
                "enabled": get_param_value(rdi_config.get('enabled'), True),
                "rdi_periods": [5, 13, 21],
                "rdi_modulator_weight": 0.2,
                "rdi_period_weights": {"5": 0.4, "13": 0.3, "21": 0.3},
                "resonance_reward_factor": 0.15,
                "divergence_penalty_factor": 0.25,
                "inflection_reward_factor": 0.10,
                "adaptive_sensitivity": 0.8
            }
        }

    def _validate_pmd_signals(self, df: pd.DataFrame, pmd_params: Dict, method_name: str) -> bool:
        """V3.6.0 · 军械库全信号链校验：杜绝计算中途因信号缺失导致的崩溃"""
        required = [
            'close_D', 'MACDh_13_34_8_D', 'RSI_13_D', 'volume_D', 'VPA_EFFICIENCY_D',
            'CLOSING_STRENGTH_D', 'chip_rsi_divergence_D', 'HM_COORDINATED_ATTACK_D',
            'SMART_MONEY_SYNERGY_BUY_D', 'main_force_activity_index_D', 'net_mf_amount_D',
            'absorption_energy_D', 'distribution_energy_D', 'tick_abnormal_volume_ratio_D',
            'MA_POTENTIAL_TENSION_INDEX_D', 'MA_RUBBER_BAND_EXTENSION_D', 'chip_convergence_ratio_D',
            'chip_entropy_D', 'STATE_EMOTIONAL_EXTREME_D', 'market_sentiment_score_D',
            'days_since_last_peak_D', 'STATE_PARABOLIC_WARNING_D', 'THEME_HOTNESS_SCORE_D',
            'game_intensity_D', 'chip_stability_D', 'price_percentile_position_D'
        ]
        # 批量添加斐波那契全周期的三阶导数信号校验
        fib = pmd_params['fib_periods']
        dynamic_cols = ['close_D', 'MACDh_13_34_8_D', 'volume_D', 'net_mf_amount_D', 'distribution_energy_D', 'absorption_energy_D']
        for col in dynamic_cols:
            for p in fib:
                required.extend([f'SLOPE_{p}_{col}', f'ACCEL_{p}_{col}', f'JERK_{p}_{col}'])
        missing = [s for s in required if s not in df.columns]
        if missing:
            print(f"  [严重警告] {method_name} 军械库信号断裂: {missing[:10]}... (共缺失{len(missing)}项)")
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
        """V3.6.0 · 成交量动力学建模修正：对齐 vol_kinematic_weights 配置"""
        fib_periods = pmd_params['fib_periods']
        deadzone = pmd_params['kinematic_deadzone']
        k_weights = pmd_params['vol_kinematic_weights'] # 修复配置引用错误
        period_scores = []
        debug_v = {}
        for p in fib_periods:
            v_s = df[f'SLOPE_{p}_volume_D'].where(df[f'SLOPE_{p}_volume_D'].abs() > deadzone, 0.0)
            v_a = df[f'ACCEL_{p}_volume_D'].where(df[f'ACCEL_{p}_volume_D'].abs() > deadzone, 0.0)
            v_j = df[f'JERK_{p}_volume_D'].where(df[f'JERK_{p}_volume_D'].abs() > deadzone, 0.0)
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
        """V3.12.0 · 量能确认 HAB 增强模型：集成 21D 能量平衡累积与周期稳定性对冲 (全探针暴露)"""
        weights = pmd_params['vol_conf_weights']
        # 1. 能量平衡判定 (Energy Balance) - 引入 HAB 累积记忆
        abs_energy_hab = df['absorption_energy_D'].rolling(window=21).sum().fillna(0)
        dist_energy_hab = df['distribution_energy_D'].rolling(window=21).sum().fillna(0)
        # 计算当日能量相对于 HAB 周期均值的偏离度，作为强度放大器
        abs_intensity = (df['absorption_energy_D'] / (abs_energy_hab / 21).clip(lower=1e-9))
        dist_intensity = (df['distribution_energy_D'] / (dist_energy_hab / 21).clip(lower=1e-9))
        # 根据背离方向选择能量分值：顶背离校验 HAB 派发稳定性，底背离校验 HAB 吸收韧性
        energy_balance = pd.Series([
            (dist_intensity.loc[idx] * 0.7 + (dist_energy_hab.loc[idx] / (abs_energy_hab.loc[idx] + 1e-9)) * 0.3) if x > 0 else
            (abs_intensity.loc[idx] * 0.7 + (abs_energy_hab.loc[idx] / (dist_energy_hab.loc[idx] + 1e-9)) * 0.3) if x < 0 else 0
            for idx, x in base_divergence_score.items()
        ], index=df_index)
        # 2. 量能动力学矩阵 (Kinematic Impact)
        v_kin, _ = self._calculate_kinematic_core_node(df, df_index, 'volume_D', pmd_params)
        vol_impact = v_kin.abs()
        # 3. 稳定性与异动
        micro_anomaly = self.helper._normalize_series(df['tick_abnormal_volume_ratio_D'], df_index, ascending=True)
        stability = self.helper._normalize_series(df['TURNOVER_STABILITY_INDEX_D'], df_index, ascending=True)
        # 4. 最终融合：HAB 能量平衡(40%) + 动力学冲击(30%) + 异动(20%) + 稳定性(10%)
        volume_conf_score = (energy_balance.clip(0, 1.5) * weights["energy_balance"] + vol_impact * weights["kinematic_impact"] + micro_anomaly * weights["micro_anomaly"] + stability * weights["stability"])
        final_vol_conf = volume_conf_score * np.sign(base_divergence_score)
        print(f"  [探针-VolConfHAB] HAB吸收累积: {abs_energy_hab.mean():.2f} | HAB派发累积: {dist_energy_hab.mean():.2f} | 能量平衡均值: {energy_balance.mean():.4f}")
        debug_v = {"node_energy_hab": energy_balance, "node_vol_impact": vol_impact, "final_volume_confirmation": final_vol_conf}
        return final_vol_conf.astype(np.float32), debug_v

    def _calculate_main_force_confirmation_score(self, df: pd.DataFrame, df_index: pd.Index, pmd_params: Dict, base_div: pd.Series, method_name: str) -> Tuple[pd.Series, Dict]:
        """V3.9.0 · 主力确认记忆模型：集成 HAB 历史持仓韧性与三阶动力学修正 (全探针暴露)"""
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
        # 2. 资金三阶动力学节点 (FK)
        fk_period_scores = []
        for p in fib:
            s_r = df[f'SLOPE_{p}_net_mf_amount_D'].where(df[f'SLOPE_{p}_net_mf_amount_D'].abs() > deadzone, 0.0)
            a_r = df[f'ACCEL_{p}_net_mf_amount_D'].where(df[f'ACCEL_{p}_net_mf_amount_D'].abs() > deadzone, 0.0)
            j_r = df[f'JERK_{p}_net_mf_amount_D'].where(df[f'JERK_{p}_net_mf_amount_D'].abs() > deadzone, 0.0)
            a_p = np.where(a_r < 0, a_r * asym, a_r)
            s_n = self.helper._normalize_series(s_r, df_index, bipolar=True)
            a_n = self.helper._normalize_series(pd.Series(a_p, index=df_index), df_index, bipolar=True)
            j_n = self.helper._normalize_series(pd.Series(j_r, index=df_index), df_index, bipolar=True)
            fk_period_scores.append(s_n * k_w['slope'] + a_n * k_w['accel'] + j_n * k_w['jerk'])
        fk_node = pd.concat(fk_period_scores, axis=1).mean(axis=1)
        # 3. HAB 历史累积记忆修正 (核心升级点)
        acc_mf_21d = df['net_mf_amount_D'].rolling(window=21).sum().fillna(0)
        today_mf_abs = df['net_mf_amount_D'].abs()
        # 计算流出占比：今日流出相对于21日累积买入的比例
        mf_resilience_ratio = (today_mf_abs / acc_mf_21d.clip(lower=1e-9)).replace([np.inf, -np.inf], 0).fillna(0)
        # 护盘因子：若21日累积买入且今日流出占比小于10%，则大幅衰减空头意图
        mf_dampener = np.where((acc_mf_21d > 0) & (df['net_mf_amount_D'] < 0) & (mf_resilience_ratio < 0.1), pmd_params['intent_dampening_factor'], 1.0)
        mf_dampener_series = pd.Series(mf_dampener, index=df_index)
        # 4. 意图与陷阱节点 (ITV) - 注入 HAB 修正
        acc_score = self.helper._normalize_series(df['accumulation_signal_score_D'], df_index, ascending=True)
        dist_score = self.helper._normalize_series(df['distribution_signal_score_D'], df_index, ascending=True)
        shake_score = self.helper._normalize_series(df['shakeout_score_D'], df_index, ascending=True)
        large_anomaly = self.helper._normalize_series(df['large_order_anomaly_D'], df_index, ascending=True)
        sm_diverge = self.helper._normalize_series(df['SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D'], df_index, bipolar=True)
        itv_raw = pd.Series([
            (dist_score.loc[idx] * i_w['accumulation'] + large_anomaly.loc[idx] * i_w['large_order'] + sm_diverge.loc[idx].clip(0)) if x > 0 else
            (acc_score.loc[idx] * i_w['accumulation'] + shake_score.loc[idx] * i_w['shakeout'] - sm_diverge.loc[idx].clip(upper=0)) if x < 0 else 0
            for idx, x in base_div.items()
        ], index=df_index)
        itv_node = itv_raw * mf_dampener_series # 历史累积记忆对陷阱判定的修正
        # 5. 最终融合：协同 * 动力 * 修正后的意图
        chip_stb = self.helper._normalize_series(df['chip_stability_D'], df_index, ascending=True)
        mf_conf_score = (mfc_node * 0.35 + fk_node * 0.4 + itv_node * 0.25) * chip_stb
        final_mf_conf = mf_conf_score * np.sign(base_div)
        print(f"[探针-MFConfirmation] 21D累积均值: {acc_mf_21d.mean():.2f} | 意图衰减器均值: {mf_dampener_series.mean():.4f} | 持仓韧性占比均值: {mf_resilience_ratio.mean():.4f}")
        debug_v = {
            "probe_acc_mf_21d": acc_mf_21d, "node_mf_cohesion": mfc_node, "node_mf_kinematics": fk_node,
            "node_mf_intent_hab_fixed": itv_node, "final_mf_confirmation": final_mf_conf
        }
        return final_mf_conf, debug_v

    def _calculate_kinematic_core_node(self, df: pd.DataFrame, df_index: pd.Index, base_col: str, pmd_params: Dict) -> Tuple[pd.Series, Dict]:
        """V3.13.0 · 动力学引擎修复版：强化参数键值探针，彻底杜绝零基陷阱"""
        # 探针：参数完整性硬校验
        required_keys = ['fib_periods', 'kinematic_weights', 'kinematic_deadzone', 'asymmetric_factor']
        missing = [k for k in required_keys if k not in pmd_params]
        if missing:
            print(f"  [严重异常] _calculate_kinematic_core_node: 缺少核心配置项 {missing}")
            raise KeyError(f"PMD动力学引擎参数不完整，缺失: {missing}")
        fib = pmd_params['fib_periods']
        k_w = pmd_params['kinematic_weights']
        deadzone = pmd_params['kinematic_deadzone']
        asym = pmd_params['asymmetric_factor']
        period_scores = []
        probe_data = {}
        for p in fib:
            # 1. 信号提取与物理死区滤波
            s_raw = df[f'SLOPE_{p}_{base_col}'].where(df[f'SLOPE_{p}_{base_col}'].abs() > deadzone, 0.0)
            a_raw = df[f'ACCEL_{p}_{base_col}'].where(df[f'ACCEL_{p}_{base_col}'].abs() > deadzone, 0.0)
            j_raw = df[f'JERK_{p}_{base_col}'].where(df[f'JERK_{p}_{base_col}'].abs() > deadzone, 0.0)
            # 2. 非对称灵敏度处理与 Tanh 鲁棒映射
            a_mod = np.where(a_raw < 0, a_raw * asym, a_raw)
            j_mod = np.where(j_raw < 0, j_raw * asym, j_raw)
            s_n = self.helper._normalize_series(s_raw, df_index, bipolar=True)
            a_n = self.helper._normalize_series(pd.Series(a_mod, index=df_index), df_index, bipolar=True)
            j_n = self.helper._normalize_series(pd.Series(j_mod, index=df_index), df_index, bipolar=True)
            # 3. 三阶动力学合成
            combined_p = (s_n * k_w['slope'] + a_n * k_w['accel'] + j_n * k_w['jerk'])
            period_scores.append(combined_p)
            probe_data[f"p{p}_jerk_tanh"] = j_n
        fused_kinematic = pd.concat(period_scores, axis=1).mean(axis=1)
        probe_data["fused_kinematic"] = fused_kinematic
        print(f"  [探针-KinematicEngine] 信号: {base_col} | 动力均值: {fused_kinematic.mean():.4f} | 参数校验已通过")
        return fused_kinematic, probe_data

    def _calculate_spatio_temporal_consistency_score(self, df: pd.DataFrame, df_index: pd.Index, pmd_params: Dict, method_name: str) -> Tuple[pd.Series, Dict]:
        """V3.11.0 · 时空一致性验证模型：执行全频段背离相干性与冲突惩罚 (全探针暴露)"""
        fib = pmd_params['fib_periods']
        weights = pmd_params['consistency_weights']
        penalty_exp = pmd_params['conflict_penalty_exponent']
        # 1. 跨周期极性一致性：计算价格与动量在所有斐波那契周期下的方向背离一致性
        period_polarities = []
        for p in fib:
            p_slope = self.helper._normalize_series(df[f'SLOPE_{p}_close_D'], df_index, bipolar=True)
            m_slope = self.helper._normalize_series(df[f'SLOPE_{p}_MACDh_13_34_8_D'], df_index, bipolar=True)
            period_polarities.append(np.sign(p_slope - m_slope))
        polarity_matrix = pd.concat(period_polarities, axis=1)
        resonance_score = polarity_matrix.mean(axis=1).abs()
        # 2. 冲突惩罚：若多周期方向发生严重冲突 (均值趋近于0)，则产生指数级分值坍缩
        conflict_penalty = resonance_score.pow(penalty_exp)
        # 3. 价量动力学相干性：判定价格减速是否伴随量能加速度的正向确认
        pv_coherence_scores = []
        for p in fib:
            p_accel = self.helper._normalize_series(df[f'ACCEL_{p}_close_D'], df_index, bipolar=True)
            v_accel = self.helper._normalize_series(df[f'ACCEL_{p}_volume_D'], df_index, bipolar=True)
            coherence = (v_accel - p_accel).clip(0, 1)
            pv_coherence_scores.append(coherence)
        pv_coherence_node = pd.concat(pv_coherence_scores, axis=1).mean(axis=1)
        # 4. 时空熵有序度：熵值越低一致性越强
        price_entropy = self.helper._normalize_series(df['PRICE_ENTROPY_D'], df_index, ascending=True)
        order_node = 1 - price_entropy
        # 5. 融合与探针
        st_consistency_score = (resonance_score * weights['polarity_resonance'] + pv_coherence_node * weights['pv_coherence'] + order_node * weights['entropy_order']) * conflict_penalty
        print(f"  [探针-STConsistency] 周期一致性均值: {resonance_score.mean():.4f} | 冲突惩罚比例均值: {conflict_penalty.mean():.4f} | 价量相干均值: {pv_coherence_node.mean():.4f}")
        debug_v = {"node_resonance": resonance_score, "node_conflict_penalty": conflict_penalty, "node_pv_coherence": pv_coherence_node, "node_order": order_node, "final_st_consistency": st_consistency_score}
        return st_consistency_score.astype(np.float32), debug_v

    def _calculate_divergence_quality_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, base_div: pd.Series) -> Tuple[pd.Series, Dict]:
        """V3.11.0 · 背离质量增强模型：集成 HAB 缩量成熟度与时空一致性验证 (全探针暴露)"""
        q_w = pmd_params['quality_weights']
        t_w = pmd_params['tension_sub_weights']
        c_w = pmd_params['chip_order_sub_weights']
        fib = pmd_params['fib_periods']
        # 1. 结构张力节点：判定空间极性极限
        tension_idx = self.helper._normalize_series(df['MA_POTENTIAL_TENSION_INDEX_D'], df_index, ascending=True)
        rubber_band = self.helper._normalize_series(df['MA_RUBBER_BAND_EXTENSION_D'].abs(), df_index, ascending=True)
        tension_node = (tension_idx * t_w['tension_index'] + rubber_band * t_w['rubber_band'])
        # 2. 动力学衰竭节点：捕捉价格惯性与驱动力的背离
        k_exhaustion_scores = []
        for p in fib:
            p_accel = self.helper._normalize_series(df[f'ACCEL_{p}_close_D'], df_index, bipolar=True)
            m_accel = self.helper._normalize_series(df[f'ACCEL_{p}_MACDh_13_34_8_D'], df_index, bipolar=True)
            p_jerk = self.helper._normalize_series(df[f'JERK_{p}_close_D'], df_index, bipolar=True)
            exhaustion = (m_accel - p_accel).abs() * (1 + p_jerk.abs())
            k_exhaustion_scores.append(exhaustion)
        kinematic_node = pd.concat(k_exhaustion_scores, axis=1).mean(axis=1)
        # 3. 筹码有序度节点：判定博弈共识状态
        chip_conv = self.helper._normalize_series(df['chip_convergence_ratio_D'], df_index, ascending=True)
        chip_ent_inv = 1 - self.helper._normalize_series(df['chip_entropy_D'], df_index, ascending=True)
        chip_node = (chip_conv * c_w['convergence'] + chip_ent_inv * c_w['entropy_inverted'])
        # 4. HAB 成交量成熟度节点：极致缩量累积识别
        vol_rolling_mean = df['volume_D'].rolling(window=21).mean()
        vol_atrophy_hab = 1 - self.helper._normalize_series(vol_rolling_mean, df_index, ascending=True)
        turnover_stb = self.helper._normalize_series(df['TURNOVER_STABILITY_INDEX_D'], df_index, ascending=True)
        ripeness_node = (vol_atrophy_hab * 0.7 + turnover_stb * 0.3)
        # 5. 时空一致性校验：引入跨周期相干系数
        st_consistency, st_diag = self._calculate_spatio_temporal_consistency_score(df, df_index, pmd_params, "QualityConsistency")
        # 6. 最终融合与探针暴露
        quality_score = (tension_node * q_w['structural_tension'] + kinematic_node * q_w['kinematic_exhaustion'] + chip_node * q_w['chip_order'] + ripeness_node * q_w['volume_ripeness'] + st_consistency * q_w['st_consistency']).clip(0, 1)
        print(f"  [探针-QualityHAB] 物理张力节点: {tension_node.mean():.4f} | 动力衰竭节点: {kinematic_node.mean():.4f}")
        print(f"  [探针-QualityHAB] 筹码有序节点: {chip_node.mean():.4f} | 缩量成熟度节点: {ripeness_node.mean():.4f} | 时空一致性节点: {st_consistency.mean():.4f}")
        print(f"  [探针-QualityHAB] 最终质量评分均值: {quality_score.mean():.4f}")
        debug_v = {"node_tension": tension_node, "node_kinematic_exhaustion": kinematic_node, "node_chip_order": chip_node, "node_volume_ripeness_hab": ripeness_node, "node_st_consistency": st_consistency, "final_quality_score": quality_score}
        return quality_score.astype(np.float32), debug_v

    def _calculate_context_modulator(self, df_index: pd.Index, raw_data: Dict, pmd_params: Dict) -> Tuple[pd.Series, Dict]:
        """V3.17.0 · 环境场能调制模型：新增情绪极端熔断机制 (Emotional Circuit Breaker)"""
        weights = pmd_params['context_weights']
        cb_params = pmd_params['circuit_breaker_params']
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
        theme_hot = self.helper._normalize_series(raw_data['THEME_HOTNESS_SCORE_D'], df_index, ascending=False)
        game_int = self.helper._normalize_series(raw_data['game_intensity_D'], df_index, ascending=True)
        # 4. 基础几何平均融合
        components = {
            "psychological": psy_node,
            "physical_tension": phy_node,
            "environmental_hotness": theme_hot,
            "game_intensity": game_int
        }
        raw_modulator = _robust_geometric_mean(components, weights, df_index)
        # 5. 情绪极端熔断机制 (Emotional Circuit Breaker)
        # 逻辑：当情绪过热(>0.75)时，检查物理张力。
        # 若张力未确认(Low)，视为强趋势中的良性亢奋，抑制背离信号 (Dampen)
        # 若张力已确认(High)，视为双重极致，放大背离信号 (Amplify)
        emotional_intensity = emo_ext
        tension_confirmation = rubber_val
        # 默认系数为 1.0
        circuit_breaker = pd.Series(1.0, index=df_index)
        # 场景 A: 情绪过热但物理张力未到极限 -> 强趋势，抑制假背离
        mask_dampen = (emotional_intensity > cb_params['emotion_threshold']) & (tension_confirmation < cb_params['tension_confirmation'])
        circuit_breaker[mask_dampen] = cb_params['dampening_factor']
        # 场景 B: 情绪过热且物理张力爆表 -> 顶/底部结构共振，放大信号
        mask_amplify = (emotional_intensity > cb_params['emotion_threshold']) & (tension_confirmation >= cb_params['tension_confirmation'])
        circuit_breaker[mask_amplify] = cb_params['amplification_factor']
        # 6. 抛物线熔断叠加
        is_parabolic = (raw_data['STATE_PARABOLIC_WARNING_D'] > 0.5).astype(np.float32)
        penalty = pd.Series(1.0, index=df_index)
        penalty.loc[is_parabolic == 1] = pmd_params['parabolic_penalty_factor']
        # 7. 最终融合
        final_context_modulator = (raw_modulator * penalty * circuit_breaker).clip(0.1, 2.5)
        # 探针输出
        print(f"  [探针-ContextModulator] 心理节点: {psy_node.mean():.4f} | 物理动力学: {phy_node.mean():.4f}")
        print(f"  [探针-ContextModulator] 情绪熔断均值: {circuit_breaker.mean():.4f} | 抑制触发: {mask_dampen.mean()*100:.1f}% | 共振放大: {mask_amplify.mean()*100:.1f}%")
        debug_info = {
            "node_psychological": psy_node, "node_physical": phy_node,
            "node_circuit_breaker": circuit_breaker, "final_context_modulator": final_context_modulator
        }
        return final_context_modulator, debug_info

    def _calculate_rdi_for_pair(self, series_A: pd.Series, series_B: pd.Series, df_index: pd.Index, rdi_params: Dict, method_name: str, pair_name: str, field_quality: pd.Series = None) -> Tuple[pd.Series, Dict]:
        """V3.18.0 · 场能自适应 RDI：根据 field_quality 动态调整相位锁定系数"""
        # 参数提取与默认值防御
        rdi_periods = rdi_params.get('rdi_periods', [5, 13, 21])
        base_reward = rdi_params.get('resonance_reward_factor', 0.15)
        base_penalty = rdi_params.get('divergence_penalty_factor', 0.25)
        i_reward = rdi_params.get('inflection_reward_factor', 0.10)
        rdi_period_weights = rdi_params.get('rdi_period_weights', {"5": 0.4, "13": 0.3, "21": 0.3})
        sensitivity = rdi_params.get('adaptive_sensitivity', 0.0)
        # 动态系数计算
        if field_quality is not None and sensitivity > 0:
            # 质量越高(>0)，modulator 越大
            modulator = (1 + field_quality.clip(0, 1) * sensitivity)
            # 奖励增强，惩罚减弱 (高质量背离允许微小相位错位)
            eff_reward = base_reward * modulator
            eff_penalty = base_penalty / modulator
            is_adaptive = True
        else:
            eff_reward = pd.Series(base_reward, index=df_index)
            eff_penalty = pd.Series(base_penalty, index=df_index)
            modulator = pd.Series(1.0, index=df_index)
            is_adaptive = False
        all_rdi_scores_by_period = {}
        period_debug_values = {}
        # 探针：输出动态调整后的系数均值
        mean_r = eff_reward.mean() if isinstance(eff_reward, pd.Series) else eff_reward
        mean_p = eff_penalty.mean() if isinstance(eff_penalty, pd.Series) else eff_penalty
        print(f"  [探针-AdaptiveRDI] 模式: {'自适应' if is_adaptive else '静态'} | 质量均值: {field_quality.mean() if field_quality is not None else 0:.4f}")
        print(f"  [探针-AdaptiveRDI] 动态奖励均值: {mean_r:.4f} (基准{base_reward}) | 动态惩罚均值: {mean_p:.4f} (基准{base_penalty})")
        for p in rdi_periods:
            # 1. 趋势倾向计算
            tendency_A = series_A.rolling(window=p, min_periods=1).mean().fillna(0)
            tendency_B = series_B.rolling(window=p, min_periods=1).mean().fillna(0)
            # 2. 核心三项式 (使用动态系数)
            resonance_term = ((np.sign(tendency_A) == np.sign(tendency_B)) & (tendency_A.abs() > 1e-9) & (tendency_B.abs() > 1e-9)).astype(np.float32) * eff_reward
            divergence_term = ((np.sign(tendency_A) != np.sign(tendency_B)) & (tendency_A.abs() > 1e-9) & (tendency_B.abs() > 1e-9)).astype(np.float32) * eff_penalty
            inflection_A = (tendency_A.shift(1) * tendency_A < 0).astype(np.float32) * i_reward
            inflection_B = (tendency_B.shift(1) * tendency_B < 0).astype(np.float32) * i_reward
            inflection_term = (inflection_A + inflection_B) / 2
            # 3. 周期分数
            period_rdi_score = resonance_term - divergence_term + inflection_term
            all_rdi_scores_by_period[str(p)] = period_rdi_score
            period_debug_values[f"p{p}_rdi"] = period_rdi_score
        fused_rdi_score = _weighted_sum_fusion(all_rdi_scores_by_period, rdi_period_weights, df_index)
        return fused_rdi_score.astype(np.float32), period_debug_values

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
        """V3.8.0 · 派发意图记忆模型：引入历史累积买入对当日流出的对冲逻辑 (全探针暴露)"""
        # 1. 基础派发能量动力学
        dist_energy_kin, _ = self._calculate_kinematic_core_node(df, df_index, 'distribution_energy_D', pmd_params)
        # 2. HAB 历史记忆系统处理
        # 计算 21 日主力累积净买入总额
        acc_mf_21d = df['net_mf_amount_D'].rolling(window=21).sum().fillna(0)
        today_mf_out = df['net_mf_amount_D'].clip(upper=0).abs()
        # 3. 意图衰减因子计算：若累积买入远超今日流出，则衰减派发意图
        # 逻辑：当日流出占累积量的比例，比例越小，衰减越狠
        flow_ratio = (today_mf_out / acc_mf_21d.clip(lower=1e-9)).replace([np.inf, -np.inf], 0).fillna(0)
        dampener = np.where((acc_mf_21d > 0) & (flow_ratio < 0.1), pmd_params['intent_dampening_factor'], 1.0)
        dampener_series = pd.Series(dampener, index=df_index)
        # 4. 融合派发意图
        _, price_probe = self._calculate_kinematic_core_node(df, df_index, 'close_D', pmd_params)
        price_jerk_collapsed = (1 - price_probe['fused_kinematic'].abs()).clip(0, 1)
        sell_ofi = self.helper._normalize_series(df['SMART_MONEY_INST_NET_BUY_D'].clip(upper=0).abs(), df_index, ascending=True)
        # 最终意图受到 HAB 衰减器的修正
        dist_intent = (dist_energy_kin.clip(0, 1) * 0.5 + price_jerk_collapsed * 0.3 + sell_ofi * 0.2) * dampener_series
        print(f"[探针-DistIntent] 21D累积主力值: {acc_mf_21d.mean():.2f} | 今日流出: {today_mf_out.mean():.2f} | 意图衰减器均值: {dampener_series.mean():.4f}")
        return dist_intent.astype(np.float32)

    def _calculate_covert_accumulation_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> pd.Series:
        """V3.8.0 · 隐蔽吸筹记忆模型：集成历史派发压力对当日吸收能量的对冲校验 (全探针暴露)"""
        # 1. 吸收能量核心
        abs_energy_kin, _ = self._calculate_kinematic_core_node(df, df_index, 'absorption_energy_D', pmd_params)
        # 2. HAB 历史记忆对冲
        # 若 21 日内累积净派发严重，则抑制吸筹分值
        dist_mf_21d = df['net_mf_amount_D'].rolling(window=21).sum().clip(upper=0).abs()
        today_abs_val = df['absorption_energy_D']
        # 逻辑：如果今日吸收能量不足以对冲 21D 累积派发量的 20%，则视为无效吸筹
        acc_dampener = np.where((dist_mf_21d > 1e6) & (today_abs_val < (dist_mf_21d * 0.2)), pmd_params['intent_dampening_factor'], 1.0)
        acc_dampener_series = pd.Series(acc_dampener, index=df_index)
        # 3. 基础组件
        price_kin, _ = self._calculate_kinematic_core_node(df, df_index, 'close_D', pmd_params)
        price_deadzone_lock = (1 - price_kin.abs()).clip(0, 1)
        abnormal_ratio = self.helper._normalize_series(df['tick_abnormal_volume_ratio_D'], df_index, ascending=True)
        acc_signal = self.helper._normalize_series(df['accumulation_signal_score_D'], df_index, ascending=True)
        # 最终融合
        covert_acc_score = (abs_energy_kin.clip(0, 1) * 0.4 + price_deadzone_lock * 0.2 + abnormal_ratio * 0.2 + acc_signal * 0.2) * acc_dampener_series
        print(f"[探针-CovertAcc] 21D累积派发值: {dist_mf_21d.mean():.2f} | 吸筹衰减器均值: {acc_dampener_series.mean():.4f}")
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
        """V3.12.0 · 流动性潮汐 HAB 累积模型：集成 21D 筹码流水位与波动率稳定性 (取代旧版Composite方法)"""
        # 1. 逐笔筹码流动力学与 HAB 水位
        chip_flow_kin, _ = self._calculate_kinematic_core_node(df, df_index, 'tick_level_chip_flow_D', pmd_params)
        flow_hab_sum = df['tick_level_chip_flow_D'].rolling(window=21).sum().fillna(0)
        flow_hab_zscore = (df['tick_level_chip_flow_D'] - (flow_hab_sum / 21)) / (df['tick_level_chip_flow_D'].rolling(window=21).std().clip(lower=1e-9))
        # 2. 资金流波动率 HAB 稳定性
        flow_vol_stb = 1 - self.helper._normalize_series(df['flow_volatility_13d_D'], df_index, ascending=True)
        # 3. 潮汐得分融合：动力学变率(40%) + HAB水位Z值(40%) + 波动率稳定(20%)
        tide_score = (chip_flow_kin.abs() * 0.4 + flow_hab_zscore.abs().clip(0, 1) * 0.4 + flow_vol_stb * 0.2)
        print(f"  [探针-LiquidityTideHAB] 21D筹码流累积: {flow_hab_sum.mean():.2f} | 潮汐Z值均值: {flow_hab_zscore.mean():.4f}")
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

    def _calculate_spatio_temporal_consistency_score(self, df: pd.DataFrame, df_index: pd.Index, pmd_params: Dict, method_name: str) -> Tuple[pd.Series, Dict]:
        """V3.11.0 · 时空一致性验证模型：执行全频段背离相干性与冲突惩罚 (全探针暴露)"""
        fib = pmd_params['fib_periods']
        weights = pmd_params['consistency_weights']
        penalty_exp = pmd_params['conflict_penalty_exponent']
        # 1. 跨周期极性一致性 (Polarity Resonance)
        period_polarities = []
        for p in fib:
            p_slope = self.helper._normalize_series(df[f'SLOPE_{p}_close_D'], df_index, bipolar=True)
            m_slope = self.helper._normalize_series(df[f'SLOPE_{p}_MACDh_13_34_8_D'], df_index, bipolar=True)
            period_polarities.append(np.sign(p_slope - m_slope))
        polarity_matrix = pd.concat(period_polarities, axis=1)
        # 一致性均值：绝对值越接近1代表所有周期方向越统一
        resonance_score = polarity_matrix.mean(axis=1).abs()
        # 冲突惩罚：如果均值接近0 (方向混乱)，则产生非线性坍缩
        conflict_penalty = resonance_score.pow(penalty_exp)
        # 2. 价量动力学相干性 (PV Coherence)
        pv_coherence_scores = []
        for p in fib:
            p_accel = self.helper._normalize_series(df[f'ACCEL_{p}_close_D'], df_index, bipolar=True)
            v_accel = self.helper._normalize_series(df[f'ACCEL_{p}_volume_D'], df_index, bipolar=True)
            # 在高质量反转中，价格减速(Accel<0)通常伴随成交量加速(Accel>0)的承接
            coherence = (v_accel - p_accel).clip(0, 1)
            pv_coherence_scores.append(coherence)
        pv_coherence_node = pd.concat(pv_coherence_scores, axis=1).mean(axis=1)
        # 3. 时空熵有序度 (Entropy Order)
        price_entropy = self.helper._normalize_series(df['PRICE_ENTROPY_D'], df_index, ascending=True)
        order_node = 1 - price_entropy # 熵越低，一致性越强
        # 4. 融合一致性得分
        st_consistency_score = (resonance_score * weights['polarity_resonance'] + 
                                pv_coherence_node * weights['pv_coherence'] + 
                                order_node * weights['entropy_order']) * conflict_penalty
        print(f"[探针-STConsistency] 极性一致性均值: {resonance_score.mean():.4f} | 冲突惩罚后均值: {st_consistency_score.mean():.4f}")
        print(f"[探针-STConsistency] 价量相干均值: {pv_coherence_node.mean():.4f} | 有序度均值: {order_node.mean():.4f}")
        debug_v = {
            "node_polarity_resonance": resonance_score, "node_conflict_penalty": conflict_penalty,
            "node_pv_coherence": pv_coherence_node, "node_order": order_node,
            "final_st_consistency": st_consistency_score
        }
        return st_consistency_score.astype(np.float32), debug_v

    def _calculate_dqwm_matrix(self, df: pd.DataFrame, df_index: pd.Index, pmd_params: Dict, method_name: str) -> pd.Series:
        """V3.16.0 · DQWM 矩阵聚合引擎：封装六维质量分量的加权融合逻辑 (修复 AttributeError)"""
        # 1. 计算六大动力学质量分量
        dq_mom_quality = self._calculate_momentum_quality_score(df, df_index, {}, pmd_params, method_name)
        dq_tension = self._calculate_market_tension_score(df, df_index, {}, pmd_params, method_name)
        dq_stability = self._calculate_stability_score(df, df_index, {}, pmd_params, method_name)
        dq_chip_potential = self._calculate_chip_historical_potential_score(df, df_index, {}, pmd_params, method_name)
        dq_tide = self._calculate_liquidity_tide_score(df, df_index, {}, pmd_params, method_name)
        dq_const = self._calculate_market_constitution_score(df, df_index, {}, pmd_params, method_name)
        # 2. 构建组件字典
        dqwm_components = {
            "momentum_quality": dq_mom_quality, 
            "market_tension": dq_tension, 
            "stability": dq_stability, 
            "chip_potential": dq_chip_potential,
            "liquidity_tide": dq_tide, 
            "market_constitution": dq_const
        }
        # 3. 执行加权融合 (使用 V3.15.0 的鲁棒融合逻辑)
        dqwm_score = _weighted_sum_fusion(dqwm_components, pmd_params['dqwm_weights'], df_index)
        # 4. 探针输出
        print(f"  [探针-DQWM_Matrix] 动量品质: {dq_mom_quality.mean():.4f} | 物理张力: {dq_tension.mean():.4f} | 稳定性: {dq_stability.mean():.4f}")
        print(f"  [探针-DQWM_Matrix] 筹码势能: {dq_chip_potential.mean():.4f} | 流动性潮汐: {dq_tide.mean():.4f} | 市场体质: {dq_const.mean():.4f}")
        print(f"  [探针-DQWM_Matrix] 矩阵最终加权得分: {dqwm_score.mean():.4f}")
        return dqwm_score.astype(np.float32)

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """V3.19.0 · 黄金坑解锁引擎：集成结构性补偿逻辑，修复左侧低动量漏单问题 (全探针暴露)"""
        method_name = "PriceMomentumStructuralUnlock_System"
        pmd_params = self._get_pmd_params(config)
        df_index = df.index
        if not self._validate_pmd_signals(df, pmd_params, method_name):
            return pd.Series(0.0, index=df_index, dtype=np.float32)
        # 1. 基础动力学路径
        f_p, p_diag = self._calculate_fused_price_direction(df, df_index, {}, pmd_params, method_name)
        f_m, m_diag = self._calculate_fused_momentum_direction(df, df_index, {}, pmd_params, method_name)
        base_div = (f_p - f_m).clip(-1, 1)
        # 2. 质量矩阵融合
        dqwm_matrix = self._calculate_dqwm_matrix(df, df_index, pmd_params, method_name)
        # 3. 环境场能调制 (含情绪熔断)
        ctx_mod, _ = self._calculate_context_modulator(df_index, {c: df[c] for c in df.columns}, pmd_params)
        # 4. 黄金坑结构补偿 (Golden Pit Compensation) - 核心新增逻辑
        gp_params = pmd_params['golden_pit_params']
        # 归一化黄金坑状态 (0或1)
        is_golden_pit = self.helper._normalize_series(df['STATE_GOLDEN_PIT_D'], df_index, bipolar=False).round()
        # 补偿系数计算：仅在 DQWM > 阈值 且 黄金坑激活时生效
        # 逻辑：左侧交易允许低动量，但必须有基础质量支撑 (min_quality_threshold)
        gp_booster = pd.Series(1.0, index=df_index)
        if gp_params['enable_compensation']:
            qualifying_pit = (is_golden_pit > 0.5) & (dqwm_matrix > gp_params['min_quality_threshold'])
            gp_booster[qualifying_pit] = gp_params['compensation_factor']
        # 5. 双轨并行 HAB 修正 (多空独立计算)
        # 多头轨道：应用黄金坑补偿 (boost)
        bull_intent = self._calculate_covert_accumulation_score(df, df_index, {}, pmd_params, method_name)
        bull_score = ((base_div.where(base_div > 0, 0.0) * 0.6 + bull_intent * 0.4) * gp_booster)
        # 空头轨道：维持原状
        bear_intent = self._calculate_distribution_intent_score(df, df_index, {}, pmd_params, method_name)
        bear_score = (base_div.where(base_div < 0, 0.0).abs() * 0.6 + bear_intent * 0.4)
        # 6. 灵敏度增强融合
        raw_diff = (bull_score - bear_score) * dqwm_matrix * ctx_mod
        boosted_diff = np.where(raw_diff.abs() < 0.1, np.sign(raw_diff) * np.sqrt(raw_diff.abs() * 0.1), raw_diff)
        final_modulated = pd.Series(boosted_diff, index=df_index)
        # 7. 自适应 RDI 相位锁定
        if pmd_params['rdi_params']['enabled']:
            # 黄金坑状态下，进一步放宽 RDI (通过人为提升传入的 Quality，骗过 RDI 让其降低惩罚)
            # 逻辑：如果是黄金坑，我们将 RDI 看到的质量视为 (原始质量 + 0.2)，从而获得更宽松的相位锁定
            effective_quality = dqwm_matrix + (is_golden_pit * 0.2)
            rdi_val, _ = self._calculate_rdi_for_pair(f_p, f_m, df_index, pmd_params['rdi_params'], method_name, "P_M", field_quality=effective_quality)
            final_modulated *= (1 + rdi_val * pmd_params['rdi_params']['rdi_modulator_weight']).clip(0.5, 1.5)
        final_score = np.sign(final_modulated) * (final_modulated.abs().pow(pmd_params['final_fusion_exponent']))
        # 探针输出
        if is_golden_pit.iloc[-1] > 0.5:
            print(f"  [探针-GoldenPit] 黄金坑结构激活! 补偿系数: {gp_booster.iloc[-1]:.2f} | 原始DQWM: {dqwm_matrix.iloc[-1]:.4f}")
        if final_score.loc[df_index[-1]] == 0 and final_modulated.loc[df_index[-1]] != 0:
            print(f"  [告警] {method_name}: 最终分值发生指数级坍塌，RawDiff: {raw_diff.loc[df_index[-1]]:.6f}")
        return final_score.clip(-1, 1).fillna(0.0).astype(np.float32)



