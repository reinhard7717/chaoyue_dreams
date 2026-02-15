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
        """【V6.0 · 三维相空间动力学参数】引入Jerk(三阶导)、软阈值去噪、MTF指数衰减权重及A股非对称重力系数。"""
        params = get_param_value(config.get('price_momentum_divergence_params'), {})
        return {
            "phase_space_params": params.get("phase_space_params", {"velocity_weight": 0.5, "acceleration_weight": 0.3, "jerk_weight": 0.2, "angular_sensitivity": 1.5, "min_vector_magnitude": 1e-6}),
            "arsenal_weights": params.get("arsenal_native_divergence_weights", {"percent_change_divergence_D": 0.3, "divergence_strength_D": 0.3, "net_migration_direction_D": 0.2, "pressure_release_index_D": 0.2}),
            "price_components_weights": params.get("price_components_weights", {"close_D": 0.4, "cost_50pct_D": 0.3, "VPA_EFFICIENCY_D": 0.3}),
            "momentum_components_weights": params.get("momentum_components_weights", {"MACDh_13_34_8_D": 0.4, "net_mf_amount_D": 0.3, "chip_rsi_divergence_D": 0.3}),
            "fib_periods": params.get("fib_periods", [5, 13, 21, 34]),
            "mtf_decay_weights": params.get("mtf_decay_weights", {"5": 0.4, "13": 0.3, "21": 0.2, "34": 0.1}),
            "kinematic_soft_deadzone": params.get("kinematic_soft_deadzone", 1e-5),
            "asymmetric_gravity_factor": params.get("asymmetric_gravity_factor", 1.5),
            "final_fusion_exponent": params.get("final_fusion_exponent", 1.5),
            "dqwm_weights": params.get("dqwm_weights", {"momentum_quality": 0.25, "market_tension": 0.25, "stability": 0.25, "chip_potential": 0.25})
        }

    def _validate_pmd_signals(self, df: pd.DataFrame, pmd_params: Dict, method_name: str) -> bool:
        """
        【Step 1: V7.0 动态前置装甲】
        全口径校验军械库数据，涵盖 DQWM、潮汐、体质、张力等所有子系统所需的 50+ 个核心特征列。
        """
        print(f"  [探针-Validate-V7.0] 启动 {method_name} 信号全链路完整性校验...")
        # 1. 基础动力学核心
        core_cols = {
            'close_D', 'volume_D', 'turnover_rate_D', 'MACDh_13_34_8_D', 
            'net_mf_amount_D', 'tick_level_chip_flow_D'
        }
        # 2. 军械库特征 (Arsenal) - 自动提取配置中的权重键值
        arsenal_cols = set(pmd_params.get('arsenal_weights', {}).keys())
        # 3. DQWM 矩阵组件所需特征
        dqwm_cols = {
            # 动量品质
            'RSI_13_D', 'HM_COORDINATED_ATTACK_D',
            # 市场张力
            'MA_POTENTIAL_TENSION_INDEX_D', 'MA_RUBBER_BAND_EXTENSION_D', 
            'BIAS_55_D', 'ATR_14_D',
            # 稳定性
            'TURNOVER_STABILITY_INDEX_D', 'PRICE_ENTROPY_D',
            # 筹码势能
            'chip_stability_D', 'accumulation_score_D',
            # 流动性潮汐
            'flow_percentile_D', 'flow_volatility_13d_D',
            # 市场体质
            'ADX_14_D', 'price_percentile_position_D', 'market_sentiment_score_D', 'cost_50pct_D'
        }
        # 4. 确认与一致性所需特征
        confirm_cols = {
            'absorption_energy_D', 'distribution_energy_D', 'tick_abnormal_volume_ratio_D',
            'SMART_MONEY_SYNERGY_BUY_D', 'main_force_activity_index_D', 'chip_flow_intensity_D',
            'profit_pressure_D', 'upper_shadow_selling_pressure_D', 'SMART_MONEY_INST_NET_BUY_D',
            'chip_entropy_D', 'PRICE_FRACTAL_DIM_D', 'state_trending_stage_D'
        }
        # 合并所有需求列
        required_cols = core_cols | arsenal_cols | dqwm_cols | confirm_cols
        # 执行校验
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"  [严重警告] {method_name} 信号链断裂！缺失特征数量: {len(missing)}")
            print(f"  [缺失明细] {missing[:10]} ...") # 仅打印前10个避免刷屏
            return False
        print(f"  [探针-Validate] 校验通过！共核验 {len(required_cols)} 项高维特征，管道完整。")
        return True

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【Step 2 & 3: V7.0 全息逻辑编排与总线探针】
        执行 7 步推演流程，融合 3D 相空间背离、DQWM 质量矩阵与 HAB 确认系统。
        """
        method_name = "PriceMomentum_Holographic_System_V7"
        print(f"========== 启动 {method_name} 计算引擎 ==========")
        # 0. 参数加载
        pmd_params = self._get_pmd_params(config)
        df_index = df.index
        # 1. 前置校验
        if not self._validate_pmd_signals(df, pmd_params, method_name):
            print(f"  [拦截] {method_name} 触发安全熔断，返回零值。")
            return pd.Series(0.0, index=df_index, dtype=np.float32)
        # 2. 核心动力源：3D 相空间背离 (Phase Space Divergence)
        #    计算价格与动量张量的夹角，这是背离信号的方向基础
        #    (注：复用 V6.0 逻辑，但在 calculate 内部重组调用)
        p_v_list, p_a_list, p_j_list = [], [], []
        # 聚合价格分量
        for col, weight in pmd_params['price_components_weights'].items():
            v, a, j, _ = self._calculate_kinematic_tensors(df, df_index, col, pmd_params)
            p_v_list.append(v * weight)
            p_a_list.append(a * weight)
            p_j_list.append(j * weight)
        price_V = pd.concat(p_v_list, axis=1).sum(axis=1)
        price_A = pd.concat(p_a_list, axis=1).sum(axis=1)
        price_J = pd.concat(p_j_list, axis=1).sum(axis=1)
        # 聚合动量分量
        m_v_list, m_a_list, m_j_list = [], [], []
        for col, weight in pmd_params['momentum_components_weights'].items():
            v, a, j, _ = self._calculate_kinematic_tensors(df, df_index, col, pmd_params)
            m_v_list.append(v * weight)
            m_a_list.append(a * weight)
            m_j_list.append(j * weight)
        mom_V = pd.concat(m_v_list, axis=1).sum(axis=1)
        mom_A = pd.concat(m_a_list, axis=1).sum(axis=1)
        mom_J = pd.concat(m_j_list, axis=1).sum(axis=1)
        # 计算核心背离分数
        phase_divergence = self._calculate_phase_space_divergence(
            df, df_index, price_V, price_A, price_J, mom_V, mom_A, mom_J, pmd_params
        )
        print(f"  [总线探针] 核心相空间背离均值: {phase_divergence.mean():.6f}")
        # 3. 质量滤波器：DQWM 六维矩阵 (Quality Matrix)
        #    衡量当前信号发生环境的“可信度”。若 DQWM 低，背离再大也是噪音。
        dqwm_score = self._calculate_dqwm_matrix(df, df_index, pmd_params, method_name)
        print(f"  [总线探针] DQWM质量矩阵均值: {dqwm_score.mean():.6f}")
        # 4. 强度确认器：量能与主力 (Confirmation)
        #    量价时空的最后验证
        vol_conf, _ = self._calculate_volume_confirmation_score(
            df, df_index, {}, pmd_params, phase_divergence, method_name
        )
        mf_conf, _ = self._calculate_main_force_confirmation_score(
            df, df_index, pmd_params, phase_divergence, method_name
        )
        print(f"  [总线探针] 量能确认均值: {vol_conf.mean():.6f} | 主力确认均值: {mf_conf.mean():.6f}")
        # 5. 环境调制器：情境感知 (Context Modulator)
        #    情绪过热或过冷时的非线性修正
        ctx_mod, _ = self._calculate_context_modulator(df, df_index, pmd_params)
        print(f"  [总线探针] 环境调制器均值: {ctx_mod.mean():.6f}")
        # 6. RDI 相对散度修正 (Relative Divergence Index)
        #    相位锁定调整
        rdi_val, _ = self._calculate_rdi_for_pair(
            price_V, mom_V, df_index, pmd_params.get('rdi_params', {}), 
            method_name, "Price_Momentum_Main", field_quality=dqwm_score
        )
        rdi_factor = 1.0 + rdi_val * pmd_params.get('rdi_params', {}).get('rdi_modulator_weight', 0.2)
        print(f"  [总线探针] RDI修正因子均值: {rdi_factor.mean():.6f}")
        # 7. 全息融合 (Holographic Fusion)
        #    公式: Final = Sign(Div) * |Div| * DQWM(质量) * Vol(确认) * MF(确认) * Context(环境) * RDI
        #    注意：Confirmations 已经包含了符号逻辑或绝对值逻辑，需仔细处理
        #    vol_conf 和 mf_conf 在 V6/V7 中已包含 sign(base_div)，因此这里取 abs() 作为强度乘数，
        #    或者直接利用其作为带符号的增强项。
        #    最佳实践：以 phase_divergence 定方向，其他项作为标量强度系数 (0~1 或 >1)
        # 归一化强度系数 (防止连乘数值爆炸)
        # DQWM [0,1], Vol [0, ~1.5], MF [0, ~1.5], Ctx [0, ~2.0]
        final_modulated = (
            phase_divergence * # 基础方向与强度
            dqwm_score * # 质量置信度 (0~1)
            vol_conf.abs() * # 量能支持度 (强度)
            mf_conf.abs() * # 主力支持度 (强度)
            ctx_mod * # 环境修正
            rdi_factor                           # 相位修正
        )
        # 应用最终非线性增益 (Gamma Correction)
        fusion_exponent = pmd_params.get('final_fusion_exponent', 1.5)
        final_score = np.sign(final_modulated) * (final_modulated.abs().pow(fusion_exponent))
        # 最终探针统计
        pos_count = (final_score > 0.1).sum()
        neg_count = (final_score < -0.1).sum()
        print(f"  [总线探针] 融合计算结束。")
        print(f"    -> 最终得分范围: [{final_score.min():.6f}, {final_score.max():.6f}]")
        print(f"    -> 有效信号统计: 多头 {pos_count} / 空头 {neg_count}")
        print(f"========== {method_name} 执行完毕 ==========")
        return final_score.astype(np.float32)

    def _calculate_kinematic_tensors(self, df: pd.DataFrame, df_index: pd.Index, base_col: str, pmd_params: Dict) -> Tuple[pd.Series, pd.Series, pd.Series, Dict]:
        """【v2.0.0_Refined · 7步推演HAB张量引擎】融合软阈值门控、HAB历史存量意识与非对称重力增益，执行高维动力学淬炼。"""
        fib = pmd_params.get('fib_periods', [5, 13, 21, 34])
        deadzone = pmd_params.get('kinematic_soft_deadzone', 1e-5)
        gravity = pmd_params.get('asymmetric_gravity_factor', 1.5)
        mtf_w = pmd_params.get('mtf_decay_weights', {"5": 0.4, "13": 0.3, "21": 0.2, "34": 0.1})
        hab_window = pmd_params.get('hab_window', 21)
        gamma = pmd_params.get('nonlinear_gamma', 1.3)
        epsilon = 1e-9
        all_v, all_a, all_j = {}, {}, {}
        probe_data = {}
        # 第三步：HAB 历史累积记忆系统 - 计算相对冲击强度
        hab_mean = df[base_col].rolling(window=hab_window, min_periods=1).mean().abs()
        hab_impact_multiplier = (df[base_col].abs() / (hab_mean + epsilon)).clip(upper=3.0)
        # 第四步：定义专属 Robust-Tanh 归一化逻辑
        def _custom_robust_norm(series: pd.Series) -> pd.Series:
            m = series.rolling(window=55, min_periods=1).median()
            s = (series - m).abs().rolling(window=55, min_periods=1).median() * 1.4826
            z = (series - m) / (s + epsilon)
            return np.tanh(z * 0.5)
        # 第五步：非线性增益函数
        def _nonlinear_gain(x: pd.Series, g: float = 1.3) -> pd.Series:
            return np.sign(x) * (x.abs() ** g)
        for p in fib:
            # 第二步：获取预置微积分列并应用软阈值门控（零基陷阱修复）
            v_raw = df[f'SLOPE_{p}_{base_col}'].fillna(0)
            a_raw = df[f'ACCEL_{p}_{base_col}'].fillna(0)
            j_raw = df[f'JERK_{p}_{base_col}'].fillna(0)
            v_gate = np.sign(v_raw) * np.maximum(v_raw.abs() - deadzone, 0.0)
            a_gate = np.sign(a_raw) * np.maximum(a_raw.abs() - deadzone, 0.0)
            j_gate = np.sign(j_raw) * np.maximum(j_raw.abs() - deadzone, 0.0)
            # 第六步：引入非对称重力因子（向下加速度强化）
            a_asym = np.where(a_gate < 0, a_gate * gravity, a_gate)
            j_asym = np.where(j_gate < 0, j_gate * gravity, j_gate)
            # 第七步：HAB 存量意识融合与归一化
            v_hab = pd.Series(v_gate, index=df_index) * hab_impact_multiplier
            a_hab = pd.Series(a_asym, index=df_index) * hab_impact_multiplier
            j_hab = pd.Series(j_asym, index=df_index) * hab_impact_multiplier
            all_v[str(p)] = _nonlinear_gain(_custom_robust_norm(v_hab), g=gamma)
            all_a[str(p)] = _nonlinear_gain(_custom_robust_norm(a_hab), g=gamma)
            all_j[str(p)] = _nonlinear_gain(_custom_robust_norm(j_hab), g=gamma)
        # 多时间框架加权聚合
        def _mtf_weighted_sum(components: Dict[str, pd.Series], weights: Dict[str, float]) -> pd.Series:
            fused = pd.Series(0.0, index=df_index, dtype=np.float32)
            tw = 0.0
            for k, s in components.items():
                w = weights.get(k, 0.0)
                fused += s * w
                tw += w
            return fused / max(tw, epsilon)
        fused_v = _mtf_weighted_sum(all_v, mtf_w)
        fused_a = _mtf_weighted_sum(all_a, mtf_w)
        fused_j = _mtf_weighted_sum(all_j, mtf_w)
        probe_data.update({'V': fused_v, 'A': fused_a, 'J': fused_j})
        return fused_v, fused_a, fused_j, probe_data

    def _calculate_phase_space_divergence(self, df: pd.DataFrame, df_index: pd.Index, v_p: pd.Series, a_p: pd.Series, j_p: pd.Series, v_m: pd.Series, a_m: pd.Series, j_m: pd.Series, pmd_params: Dict) -> pd.Series:
        """【v2.0.0_Refined · 7步推演HAB相空间背离引擎】引入3D张量场能权重、自适应模长门控与HAB交互势能，深度识别相位失步。"""
        vw = pmd_params['phase_space_params'].get('velocity_weight', 0.5)
        aw = pmd_params['phase_space_params'].get('acceleration_weight', 0.3)
        jw = pmd_params['phase_space_params'].get('jerk_weight', 0.2)
        sens = pmd_params['phase_space_params'].get('angular_sensitivity', 1.5)
        hab_window = pmd_params.get('phase_hab_window', 21)
        gamma = pmd_params.get('divergence_gamma', 1.5)
        epsilon = 1e-9
        # 第一步：引入新增维度 - 场能权重（价量效率与博弈烈度）
        vpa_eff = df['VPA_EFFICIENCY_D'].fillna(0.5).clip(0.1, 1.0)
        game_int = df['game_intensity_D'].fillna(50.0) / 100.0
        field_energy = (vpa_eff * 0.6 + game_int * 0.4).rolling(window=5, min_periods=1).mean()
        # 第二步：解决“零基陷阱” - 构建权重向量并应用自适应模长门控
        mass_p = df['chip_stability_D'].fillna(0.5).clip(0.1, 1.0)
        mass_m = df['turnover_rate_D'].fillna(0.05).clip(0.01, 0.5)
        vp_eff, ap_eff, jp_eff = v_p * vw * mass_p * field_energy, a_p * aw * mass_p * field_energy, j_p * jw * mass_p * field_energy
        vm_eff, am_eff, jm_eff = v_m * vw * mass_m * field_energy, a_m * aw * mass_m * field_energy, j_m * jw * mass_m * field_energy
        mag_p_sq = vp_eff**2 + ap_eff**2 + jp_eff**2
        mag_m_sq = vm_eff**2 + am_eff**2 + jm_eff**2
        # 应用 Tanh 模长压缩，滤除微小噪声波动
        soft_gate_p = np.tanh(np.sqrt(mag_p_sq) / (pmd_params.get('kinematic_soft_deadzone', 1e-5) * 10))
        soft_gate_m = np.tanh(np.sqrt(mag_m_sq) / (pmd_params.get('kinematic_soft_deadzone', 1e-5) * 10))
        # 第三步：HAB 系统 - 计算点积交互势能与历史存量
        dot_product = (vp_eff * vm_eff) + (ap_eff * am_eff) + (jp_eff * jm_eff)
        mag_product = np.sqrt(mag_p_sq * mag_m_sq) + epsilon
        hab_dot = dot_product.rolling(window=hab_window, min_periods=1).mean()
        hab_mag = mag_product.rolling(window=hab_window, min_periods=1).mean()
        # 第四步：专属归一化 - 计算 HAB 相位余弦相似度并映射至背离度
        cos_theta_hab = (dot_product / mag_product).clip(-1.0, 1.0)
        # 第五、六、七步：逻辑优化 - 引入非线性增益与方向锁存
        raw_divergence = (1.0 - cos_theta_hab) * soft_gate_p * soft_gate_m
        # 计算价格趋势方向，用于为背离赋符号（正分为顶背离风险，负分为底背离机会）
        price_direction = np.sign(v_p.rolling(window=3, min_periods=1).mean())
        # 非线性幂律增益应用
        fused_divergence = price_direction * (raw_divergence ** gamma) * sens
        # 第七步：最终平滑与鲁棒性处理
        def _final_robust_norm(series: pd.Series) -> pd.Series:
            roll_mean = series.rolling(window=34, min_periods=1).mean()
            roll_std = series.rolling(window=34, min_periods=1).std().replace(0, epsilon)
            return np.tanh((series - roll_mean) / (roll_std * 1.5))
        return _final_robust_norm(fused_divergence).astype(np.float32)

    def _calculate_phase_space_pv_anomaly(self, df: pd.DataFrame, df_index: pd.Index, p_v: pd.Series, p_a: pd.Series, p_j: pd.Series, v_v: pd.Series, v_a: pd.Series, v_j: pd.Series, pmd_params: Dict) -> Tuple[pd.Series, Dict]:
        """【v2.0.0_Refined · 7步推演HAB量价相空间异常模型】引入微观异动比例与HAB潮汐存量，通过3D张量对撞识别价量相位背离。"""
        vw = pmd_params.get('phase_space_params', {}).get('velocity_weight', 0.5)
        aw = pmd_params.get('phase_space_params', {}).get('acceleration_weight', 0.3)
        jw = pmd_params.get('phase_space_params', {}).get('jerk_weight', 0.2)
        hab_window = pmd_params.get('phase_hab_window', 21)
        gamma = pmd_params.get('anomaly_gamma', 1.4)
        epsilon = 1e-9
        # 第一步：引入新增维度 - 注入微观与质量数据
        vpa_eff = df['VPA_EFFICIENCY_D'].fillna(0.5)
        tick_anomaly = df['tick_abnormal_volume_ratio_D'].fillna(0)
        turnover_f = df['turnover_rate_f_D'].fillna(0)
        # 第二、三步：建立 HAB 系统与零基阈值处理
        vol_hab_mean = df['volume_D'].rolling(window=hab_window, min_periods=1).mean().abs()
        vol_impact_ratio = (df['volume_D'].abs() / (vol_hab_mean + epsilon)).clip(upper=5.0)
        # 构建专属归一化与门控函数
        def _custom_pv_norm(series: pd.Series, gate: float = 1e-5) -> pd.Series:
            gated = np.sign(series) * np.maximum(series.abs() - gate, 0.0)
            return np.tanh(gated / (series.std() + epsilon))
        # 第四、五步：3D张量对撞逻辑优化与非线性增益
        vp_eff, ap_eff, jp_eff = _custom_pv_norm(p_v) * vw, _custom_pv_norm(p_a) * aw, _custom_pv_norm(p_j) * jw
        vv_eff, av_eff, jv_eff = _custom_pv_norm(v_v) * vw, _custom_pv_norm(v_a) * aw, _custom_pv_norm(v_j) * jw
        # 计算价量矢量的点积（衡量同向性）与模长
        dot_pv = (vp_eff * vv_eff) + (ap_eff * av_eff) + (jp_eff * jv_eff)
        mag_p = np.sqrt(vp_eff**2 + ap_eff**2 + jp_eff**2) + epsilon
        mag_v = np.sqrt(vv_eff**2 + av_eff**2 + jv_eff**2) + epsilon
        # 计算相位夹角余弦，1 - cos 代表异动程度（相位越不一致，异动越大）
        cos_theta_pv = (dot_pv / (mag_p * mag_v)).clip(-1.0, 1.0)
        pv_phase_divergence = 1.0 - cos_theta_pv
        # 第六步：引入质量算子进行逻辑对冲（如：低效率下的高相位差代表真实的异常）
        inefficiency_weight = 1.0 - vpa_eff.clip(0, 1)
        raw_anomaly = pv_phase_divergence * inefficiency_weight * (1.0 + tick_anomaly) * vol_impact_ratio
        # 第七步：最终非线性增益与符号锁存
        # 异动通常取正值代表异常程度，符号由价格斜率决定，识别是“上行异常”还是“下行异常”
        price_sign = np.sign(p_v.rolling(window=3, min_periods=1).mean())
        final_anomaly = price_sign * (raw_anomaly ** gamma)
        # 构建探针数据
        debug_sync = {
            "pv_phase_divergence": pv_phase_divergence,
            "vol_impact_ratio": vol_impact_ratio,
            "vpa_inefficiency": inefficiency_weight,
            "final_pv_anomaly": final_anomaly
        }
        return final_anomaly.astype(np.float32), debug_sync

    def _calculate_volume_confirmation_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, base_divergence_score: pd.Series, method_name: str) -> Tuple[pd.Series, Dict]:
        """【v2.0.0_Refined · 7步推演HAB量能确认张量模型】集成蔡金流场、自由换手冲击与3D量能张量，深度验证背离真实性。"""
        weights = pmd_params.get('vol_conf_weights', {"energy_balance": 0.35, "kinematic_impact": 0.25, "money_flow": 0.2, "micro_anomaly": 0.1, "stability": 0.1})
        hab_window = pmd_params.get('vol_hab_window', 21)
        gamma = pmd_params.get('vol_gain_gamma', 1.5)
        epsilon = 1e-9
        def _custom_robust_norm(series: pd.Series) -> pd.Series:
            median = series.rolling(window=55, min_periods=1).median()
            std = series.rolling(window=55, min_periods=1).std().replace(0, epsilon)
            return np.tanh((series - median) / (std * 1.2))
        def _nonlinear_gain(x: pd.Series, g: float = 1.5) -> pd.Series:
            return np.sign(x) * (x.abs() ** g)
        # 第三步：HAB 系统 - 计算能量平衡冲击
        abs_raw, dist_raw = df['absorption_energy_D'].fillna(0), df['distribution_energy_D'].fillna(0)
        abs_hab = abs_raw.rolling(window=hab_window, min_periods=1).mean()
        dist_hab = dist_raw.rolling(window=hab_window, min_periods=1).mean()
        abs_impact = abs_raw / (abs_hab + epsilon)
        dist_impact = dist_raw / (dist_hab + epsilon)
        energy_balance = pd.Series(0.0, index=df_index, dtype=np.float32)
        mask_pos = base_divergence_score > 0 # 顶背离风险确认：寻找卖盘冲击
        mask_neg = base_divergence_score < 0 # 底背离机会确认：寻找买盘吸收
        energy_balance[mask_pos] = _custom_robust_norm(dist_impact[mask_pos] - abs_impact[mask_pos])
        energy_balance[mask_neg] = _custom_robust_norm(abs_impact[mask_neg] - dist_impact[mask_neg])
        # 第二步：MTF 3D 量能张量冲击
        v_vol, a_vol, j_vol, _ = self._calculate_kinematic_tensors(df, df_index, 'volume_D', pmd_params)
        vol_kinematic = _custom_robust_norm(v_vol * 0.5 + a_vol * 0.3 + j_vol * 0.2)
        # 第一步：引入新增维度 - 资金流场确认 (CMF + NetMF)
        cmf_score = _custom_robust_norm(df['CMF_21_D'].fillna(0))
        v_mf, a_mf, j_mf, _ = self._calculate_kinematic_tensors(df, df_index, 'net_mf_amount_D', pmd_params)
        mf_kinematic = _custom_robust_norm(v_mf * 0.5 + a_mf * 0.3 + j_mf * 0.2)
        # 第六步：逻辑优化 - 资金流场需根据背离方向对冲
        money_flow_conf = pd.Series(0.0, index=df_index)
        money_flow_conf[mask_pos] = _custom_robust_norm((-cmf_score) + (-mf_kinematic)) # 顶背离：需资金流出
        money_flow_conf[mask_neg] = _custom_robust_norm(cmf_score + mf_kinematic) # 底背离：需资金流入
        # 微观与稳定性因子
        micro_anomaly = _custom_robust_norm(df['tick_abnormal_volume_ratio_D'].fillna(0))
        stability = 1.0 - _custom_robust_norm(df['TURNOVER_STABILITY_INDEX_D'].fillna(0)).abs()
        # 第七步：多维场能融合
        raw_vol_conf = (
            energy_balance * weights["energy_balance"] +
            vol_kinematic.abs() * weights["kinematic_impact"] +
            money_flow_conf * weights["money_flow"] +
            micro_anomaly * weights["micro_anomaly"] +
            stability * weights["stability"]
        )
        final_vol_conf = _nonlinear_gain(raw_vol_conf, g=gamma) * np.sign(base_divergence_score)
        debug_v = {
            "node_energy_balance": energy_balance,
            "node_vol_kinematic": vol_kinematic,
            "node_money_flow_conf": money_flow_conf,
            "final_volume_confirmation": final_vol_conf
        }
        return final_vol_conf.astype(np.float32), debug_v

    def _calculate_main_force_confirmation_score(self, df: pd.DataFrame, df_index: pd.Index, pmd_params: Dict, base_div: pd.Series, method_name: str) -> Tuple[pd.Series, Dict]:
        """【v2.0.0_Refined · 7步推演HAB主力相空间确认引擎】集成机构净买入张量、日内吸筹置信度与HAB存量对冲，深度锁定主力真实意图。"""
        c_w = pmd_params.get('mf_cohesion_weights', {"HM_COORDINATED_ATTACK_D": 0.2, "SMART_MONEY_SYNERGY_BUY_D": 0.2, "SMART_MONEY_INST_NET_BUY_D": 0.2, "main_force_activity_index_D": 0.15, "intraday_accumulation_confidence_D": 0.15, "chip_flow_intensity_D": 0.1})
        hab_window = pmd_params.get('mf_hab_window', 34)
        gamma = pmd_params.get('mf_gain_gamma', 1.4)
        epsilon = 1e-9
        def _custom_mf_norm(series: pd.Series) -> pd.Series:
            roll_m = series.rolling(window=55, min_periods=1).median()
            roll_s = (series - roll_m).abs().rolling(window=55, min_periods=1).median() * 1.4826
            return np.tanh((series - roll_m) / (roll_s + epsilon))
        def _nonlinear_exp_gain(x: pd.Series, g: float = 1.4) -> pd.Series:
            return np.sign(x) * (np.exp(x.abs()) - 1.0) * g
        # 第三步：HAB 系统 - 资金流存量意识与冲击强度
        mf_net_raw = df['net_mf_amount_D'].fillna(0)
        mf_hab_stock = mf_net_raw.rolling(window=hab_window, min_periods=1).mean().abs()
        mf_impact_ratio = (mf_net_raw.abs() / (mf_hab_stock + epsilon)).clip(upper=4.0)
        # 第一步：引入新增维度 - 机构净买入与日内吸筹置信度
        inst_net = _custom_mf_norm(df['SMART_MONEY_INST_NET_BUY_D'].fillna(0))
        intra_acc = _custom_mf_norm(df['intraday_accumulation_confidence_D'].fillna(50.0))
        # 第二步：MTF 3D 主力资金张量冲击 (Slope/Accel/Jerk)
        v_mf, a_mf, j_mf, _ = self._calculate_kinematic_tensors(df, df_index, 'net_mf_amount_D', pmd_params)
        mf_kinematic = _custom_mf_norm(v_mf * 0.5 + a_mf * 0.3 + j_mf * 0.2)
        # 第六步：逻辑优化 - 构建主力协同核心矩阵
        hm_attack = _custom_mf_norm(df['HM_COORDINATED_ATTACK_D'].fillna(0))
        sm_synergy = _custom_mf_norm(df['SMART_MONEY_SYNERGY_BUY_D'].fillna(0))
        mf_activity = _custom_mf_norm(df['main_force_activity_index_D'].fillna(50.0))
        chip_flow = _custom_mf_norm(df['chip_flow_intensity_D'].fillna(0))
        cohesion_matrix = (
            hm_attack * c_w["HM_COORDINATED_ATTACK_D"] +
            sm_synergy * c_w["SMART_MONEY_SYNERGY_BUY_D"] +
            inst_net * c_w["SMART_MONEY_INST_NET_BUY_D"] +
            mf_activity * c_w["main_force_activity_index_D"] +
            intra_acc * c_w["intraday_accumulation_confidence_D"] +
            chip_flow * c_w["chip_flow_intensity_D"]
        )
        # 意图对冲逻辑：根据背离方向锁定吸筹或派发信号
        bull_intent = self._calculate_covert_accumulation_score(df, df_index, {}, pmd_params, method_name)
        bear_intent = self._calculate_distribution_intent_score(df, df_index, {}, pmd_params, method_name)
        intent_node = pd.Series(0.0, index=df_index)
        mask_pos = base_div > 0 # 顶背离：需验证主力派发强度
        mask_neg = base_div < 0 # 底背离：需验证主力吸筹强度
        intent_node[mask_pos] = bear_intent[mask_pos]
        intent_node[mask_neg] = bull_intent[mask_neg]
        # 第七步：多维场能融合与非线性增益
        # 融合公式：(协同矩阵 + 张量冲击 + 意图节点) * HAB冲击比率
        raw_mf_conf = (cohesion_matrix * 0.4 + mf_kinematic * 0.3 + intent_node * 0.3) * mf_impact_ratio
        final_mf_conf = _nonlinear_exp_gain(raw_mf_conf, g=gamma) * np.sign(base_div)
        debug_v = {
            "node_cohesion_matrix": cohesion_matrix,
            "node_mf_kinematic": mf_kinematic,
            "node_intent_hab": intent_node,
            "final_mf_confirmation": final_mf_conf
        }
        return final_mf_conf.astype(np.float32), debug_v

    def _calculate_spatio_temporal_consistency_score(self, df: pd.DataFrame, df_index: pd.Index, pmd_params: Dict, method_name: str) -> Tuple[pd.Series, Dict]:
        """【V7.1.0_Refined · 7步推演HAB时空一致性全息张量模型】融合日周长短周期同步与多维均线共振，精准挂载软阈值提取价量流多期3D微积分，结合HAB冲击对冲与定制双极映射，并施加指数级冲突惩罚重构高维时空有序度。"""
        print(f"  [探针-STConsistencyHAB-V7.1.0] 启动时空一致性全息校验...")
        fib = pmd_params.get('fib_periods', [5, 13, 21, 34])
        mtf_w = pmd_params.get('mtf_decay_weights', {"5": 0.4, "13": 0.3, "21": 0.2, "34": 0.1})
        weights = pmd_params.get('consistency_weights', {'phase_resonance': 0.35, 'flow_coherence': 0.35, 'entropy_order': 0.3})
        penalty_exp = pmd_params.get('conflict_penalty_exponent', 3.0)
        hab_window = pmd_params.get('consistency_hab_window', 13)
        deadzone = pmd_params.get('kinematic_soft_deadzone', 1e-5)
        epsilon = 1e-9
        def custom_robust_norm(series: pd.Series, mode: str = 'bipolar', sensitivity: float = 0.8) -> pd.Series:
            roll_mean = series.rolling(window=34, min_periods=1).mean()
            roll_std = series.rolling(window=34, min_periods=1).std().replace(0, epsilon)
            z_score = (series - roll_mean) / roll_std
            if mode == 'bipolar':
                return np.tanh(z_score * sensitivity)
            else:
                return 1.0 / (1.0 + np.exp(-z_score * sensitivity))
        def extract_kinematic_polarity(base_col: str) -> pd.Series:
            fused_polarity = pd.Series(0.0, index=df_index, dtype=np.float32)
            tw = 0.0
            for p in fib:
                w = mtf_w.get(str(p), 0.0)
                if w == 0: continue
                v = df[f'SLOPE_{p}_{base_col}'].fillna(0)
                a = df[f'ACCEL_{p}_{base_col}'].fillna(0)
                j = df[f'JERK_{p}_{base_col}'].fillna(0)
                v_soft = np.sign(v) * np.maximum(v.abs() - deadzone, 0.0)
                a_soft = np.sign(a) * np.maximum(a.abs() - deadzone, 0.0)
                j_soft = np.sign(j) * np.maximum(j.abs() - deadzone, 0.0)
                p_kin = np.sign(v_soft) * 0.5 + np.sign(a_soft) * 0.3 + np.sign(j_soft) * 0.2
                fused_polarity += p_kin * w
                tw += w
            if tw > 0:
                fused_polarity = fused_polarity / tw
            return fused_polarity.clip(-1.0, 1.0)
        ma_coherence = custom_robust_norm(df['MA_COHERENCE_RESONANCE_D'].fillna(50.0), 'unipolar')
        short_mid_sync = custom_robust_norm(df['short_mid_sync_D'].fillna(50.0), 'unipolar')
        daily_weekly_sync = custom_robust_norm(df['daily_weekly_sync_D'].fillna(50.0), 'unipolar')
        sync_composite = (ma_coherence * 0.4 + short_mid_sync * 0.3 + daily_weekly_sync * 0.3)
        hab_sync = sync_composite.rolling(window=hab_window, min_periods=1).mean()
        sync_impact = (sync_composite / (hab_sync + epsilon)).clip(upper=3.0)
        kin_p_polar = extract_kinematic_polarity('close_D')
        kin_v_polar = extract_kinematic_polarity('volume_D')
        pv_coherence = (kin_p_polar * kin_v_polar)
        phase_resonance = pv_coherence * sync_impact * sync_composite
        print(f"    -> [节点] 相位共振场(价量极性与多期同步)提取完成。均值: {phase_resonance.mean():.6f}")
        flow_cons = custom_robust_norm(df['flow_consistency_D'].fillna(50.0), 'unipolar')
        pf_div = custom_robust_norm(df['price_flow_divergence_D'].fillna(0.0), 'unipolar')
        kin_f_polar = extract_kinematic_polarity('net_mf_amount_D')
        pf_tensor_coherence = (kin_p_polar * kin_f_polar)
        hab_pf_coherence = pf_tensor_coherence.rolling(window=hab_window, min_periods=1).mean()
        flow_coherence = ((hab_pf_coherence * 0.6 + flow_cons * 0.4) * (1.0 - pf_div)).clip(-1.0, 1.0)
        print(f"    -> [节点] 资金相干场(价流极性与资金一致性)提取完成。均值: {flow_coherence.mean():.6f}")
        chip_ent = df['chip_entropy_D'].fillna(1.0)
        price_ent = df['PRICE_ENTROPY_D'].fillna(1.0)
        frac_dim = df['PRICE_FRACTAL_DIM_D'].fillna(1.5)
        composite_entropy = (custom_robust_norm(chip_ent, 'unipolar') * 0.4 + custom_robust_norm(price_ent, 'unipolar') * 0.4 + custom_robust_norm(frac_dim, 'unipolar') * 0.2)
        hab_entropy = composite_entropy.rolling(window=hab_window, min_periods=1).mean()
        entropy_order = 1.0 - hab_entropy
        print(f"    -> [节点] 系统熵场(价筹高维秩序度)提取完成。均值: {entropy_order.mean():.6f}")
        polar_matrix = pd.concat([np.sign(phase_resonance), np.sign(flow_coherence)], axis=1)
        conflict_intensity = polar_matrix.mean(axis=1).abs()
        conflict_penalty = conflict_intensity.pow(penalty_exp)
        raw_consistency = (phase_resonance * weights.get('phase_resonance', 0.35) + flow_coherence * weights.get('flow_coherence', 0.35) + entropy_order * weights.get('entropy_order', 0.3))
        penalized_consistency = raw_consistency * conflict_penalty
        def nonlinear_gain(x: pd.Series, gamma: float = 1.3) -> pd.Series:
            return np.sign(x) * (x.abs() ** gamma)
        final_st_consistency = nonlinear_gain(penalized_consistency).clip(-1.0, 1.0)
        print(f"  [探针-STConsistencyHAB-V7.1.0] 时空一致性全息校验完成。非线性得分均值: {final_st_consistency.mean():.6f}, 极值: [{final_st_consistency.min():.6f}, {final_st_consistency.max():.6f}], NaN统计: {final_st_consistency.isna().sum()}")
        debug_v = {"node_phase_resonance": phase_resonance, "node_flow_coherence": flow_coherence, "node_entropy_order": entropy_order, "node_conflict_penalty": conflict_penalty, "final_st_consistency": final_st_consistency}
        return final_st_consistency.astype(np.float32), debug_v

    def _calculate_divergence_quality_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, base_div: pd.Series) -> Tuple[pd.Series, Dict]:
        """【v7.2.0_Refined · 7步推演HAB背离质量张量模型】全息融合原生背离置信度、3D张量衰竭、HAB结构张力与量能萎缩干涸度，挂载软阈值门控提取多期微积分，采用MAD中位数鲁棒归一化与幂律增益，严厉暴晒劣质假背离。"""
        print(f"  [探针-Quality-V7.2.0_Refined] 启动背离质量高维评估...")
        q_w = pmd_params.get('quality_weights', {"structural_tension": 0.15, "kinematic_exhaustion": 0.20, "chip_order": 0.15, "volume_ripeness": 0.10, "signal_confidence": 0.25, "st_consistency": 0.15})
        hab_window = pmd_params.get('quality_hab_window', 21)
        deadzone = pmd_params.get('kinematic_soft_deadzone', 1e-5)
        fib_periods = pmd_params.get('fib_periods', [5, 13, 21, 34])
        mtf_w = pmd_params.get('mtf_decay_weights', {"5": 0.4, "13": 0.3, "21": 0.2, "34": 0.1})
        gamma = pmd_params.get('quality_gamma', 1.4)
        epsilon = 1e-9
        def custom_robust_norm(series: pd.Series, mode: str = 'unipolar', sensitivity: float = 0.8) -> pd.Series:
            roll_median = series.rolling(window=34, min_periods=1).median()
            roll_mad = (series - roll_median).abs().rolling(window=34, min_periods=1).median() * 1.4826 + epsilon
            z_score = (series - roll_median) / roll_mad
            if mode == 'bipolar':
                return np.tanh(z_score * sensitivity)
            else:
                return 1.0 / (1.0 + np.exp(-z_score * sensitivity))
        def extract_quality_kinematics(base_col: str) -> pd.Series:
            fused_kin = pd.Series(0.0, index=df_index, dtype=np.float32)
            tw = 0.0
            for p in fib_periods:
                w = mtf_w.get(str(p), 0.0)
                if w == 0: continue
                v = df.get(f'SLOPE_{p}_{base_col}', pd.Series(0.0, index=df_index)).fillna(0)
                a = df.get(f'ACCEL_{p}_{base_col}', pd.Series(0.0, index=df_index)).fillna(0)
                j = df.get(f'JERK_{p}_{base_col}', pd.Series(0.0, index=df_index)).fillna(0)
                v_soft = np.sign(v) * np.maximum(v.abs() - deadzone, 0.0)
                a_soft = np.sign(a) * np.maximum(a.abs() - deadzone, 0.0)
                j_soft = np.sign(j) * np.maximum(j.abs() - deadzone, 0.0)
                p_kin = v_soft * 0.5 + a_soft * 0.3 + j_soft * 0.2
                fused_kin += p_kin * w
                tw += w
            if tw > 0:
                fused_kin = fused_kin / tw
            return fused_kin
        tension_raw = df['MA_POTENTIAL_TENSION_INDEX_D'].fillna(0)
        rubber_raw = df['MA_RUBBER_BAND_EXTENSION_D'].fillna(0).abs()
        profit_pressure = df['profit_pressure_D'].fillna(0)
        hab_tension = tension_raw.abs().rolling(window=hab_window, min_periods=1).mean()
        tension_impact = (tension_raw.abs() / (hab_tension + epsilon)).clip(upper=3.0)
        tension_composite = (custom_robust_norm(tension_raw.abs(), 'unipolar') * 0.4 + custom_robust_norm(rubber_raw, 'unipolar') * 0.3 + custom_robust_norm(profit_pressure, 'unipolar') * 0.3) * tension_impact
        tension_node = custom_robust_norm(tension_composite, 'unipolar')
        print(f"    -> [节点] HAB结构张力提取完成。均值: {tension_node.mean():.6f}, NaN: {tension_node.isna().sum()}")
        v_p, a_p, j_p, _ = self._calculate_kinematic_tensors(df, df_index, 'close_D', pmd_params)
        v_m, a_m, j_m, _ = self._calculate_kinematic_tensors(df, df_index, 'MACDh_13_34_8_D', pmd_params)
        kinematic_diff = (a_m - a_p).abs() * 0.6 + (j_m - j_p).abs() * 0.4
        hab_kin_diff = kinematic_diff.rolling(window=hab_window, min_periods=1).mean()
        kin_impact = (kinematic_diff / (hab_kin_diff + epsilon)).clip(upper=3.0)
        kinematic_node = custom_robust_norm(kinematic_diff * kin_impact, 'unipolar')
        print(f"    -> [节点] HAB动力学衰竭张量提取完成。均值: {kinematic_node.mean():.6f}, NaN: {kinematic_node.isna().sum()}")
        chip_conv = df['chip_convergence_ratio_D'].fillna(0)
        chip_ent_inv = 1.0 - df['chip_entropy_D'].fillna(1.0)
        chip_rsi_div = df['chip_rsi_divergence_D'].fillna(0).abs()
        hab_chip_conv = chip_conv.rolling(window=hab_window, min_periods=1).mean()
        chip_impact = (chip_conv / (hab_chip_conv + epsilon)).clip(upper=3.0)
        chip_node = custom_robust_norm(chip_conv * chip_impact, 'unipolar') * 0.4 + custom_robust_norm(chip_ent_inv, 'unipolar') * 0.3 + custom_robust_norm(chip_rsi_div, 'unipolar') * 0.3
        print(f"    -> [节点] HAB筹码秩序度提取完成。均值: {chip_node.mean():.6f}, NaN: {chip_node.isna().sum()}")
        turnover_f = df['turnover_rate_f_D'].fillna(df['turnover_rate_D'].fillna(0.01))
        hab_turnover = turnover_f.rolling(window=hab_window, min_periods=1).mean()
        shrinkage_impact = np.maximum(hab_turnover - turnover_f, 0.0) / (hab_turnover + epsilon)
        ripeness_node = custom_robust_norm(shrinkage_impact, 'unipolar')
        print(f"    -> [节点] HAB量能成熟缓冲度(干涸度)提取完成。均值: {ripeness_node.mean():.6f}, NaN: {ripeness_node.isna().sum()}")
        div_str = df['divergence_strength_D'].fillna(0).abs()
        sig_qual = df['signal_quality_score_D'].fillna(50.0)
        pat_conf = df['pattern_confidence_D'].fillna(50.0)
        hab_div_str = div_str.rolling(window=hab_window, min_periods=1).mean()
        div_impact = (div_str / (hab_div_str + epsilon)).clip(upper=3.0)
        div_kin = extract_quality_kinematics('divergence_strength_D').abs()
        confidence_node = custom_robust_norm(div_str * div_impact, 'unipolar') * 0.4 + custom_robust_norm(sig_qual, 'unipolar') * 0.2 + custom_robust_norm(pat_conf, 'unipolar') * 0.2 + custom_robust_norm(div_kin, 'unipolar') * 0.2
        print(f"    -> [节点] 军械库原生置信度与3D质量提取完成。均值: {confidence_node.mean():.6f}, NaN: {confidence_node.isna().sum()}")
        st_consistency, _ = self._calculate_spatio_temporal_consistency_score(df, df_index, pmd_params, "QualityConsistency")
        st_consistency_node = custom_robust_norm(st_consistency.abs(), 'unipolar')
        print(f"    -> [节点] ST时空一致性映射完成。均值: {st_consistency_node.mean():.6f}, NaN: {st_consistency_node.isna().sum()}")
        raw_quality = (tension_node * q_w.get('structural_tension', 0.15) + kinematic_node * q_w.get('kinematic_exhaustion', 0.20) + chip_node * q_w.get('chip_order', 0.15) + ripeness_node * q_w.get('volume_ripeness', 0.10) + confidence_node * q_w.get('signal_confidence', 0.25) + st_consistency_node * q_w.get('st_consistency', 0.15))
        def power_law_gain(x: pd.Series, g: float = 1.4) -> pd.Series:
            return np.sign(x) * (x.abs() ** g)
        final_quality = power_law_gain(raw_quality, g=gamma).clip(0.0, 1.0)
        print(f"  [探针-Quality-V7.2.0_Refined] 质量背离矩阵计算完成。非线性最终得分均值: {final_quality.mean():.6f}, 极值: [{final_quality.min():.6f}, {final_quality.max():.6f}], NaN统计: {final_quality.isna().sum()}")
        debug_v = {"node_tension": tension_node, "node_kinematic_exhaustion": kinematic_node, "node_chip_order": chip_node, "node_volume_ripeness": ripeness_node, "node_confidence": confidence_node, "node_st_consistency": st_consistency_node, "final_quality_score": final_quality}
        return final_quality.astype(np.float32), debug_v

    def _calculate_context_modulator(self, df: pd.DataFrame, df_index: pd.Index, pmd_params: Dict) -> Tuple[pd.Series, Dict]:
        """【V6.8 · 7步推演HAB情境调制器】引入情绪与张力的3D张量、HAB水温记忆、专属Tanh归一化，通过自然指数(exp)实现无界情境共振映射，彻底剥离clip防错。"""
        print(f"  [探针-ContextMod-V6.8] 启动情境调制网络评估...")
        weights = pmd_params.get('context_weights', {"psychological": 0.3, "physical_tension": 0.3, "environmental": 0.4})
        cb_params = pmd_params.get('circuit_breaker_params', {"emotion_threshold": 0.6, "tension_confirmation": 0.6, "dampening_factor": 0.4, "amplification_factor": 1.5})
        hab_window = pmd_params.get('context_hab_window', 21)
        epsilon = 1e-9
        def custom_robust_norm(series: pd.Series, factor: float = 0.8) -> pd.Series:
            roll_mean = series.rolling(window=34, min_periods=1).mean()
            roll_std = series.rolling(window=34, min_periods=1).std().replace(0, epsilon)
            z_score = (series - roll_mean) / roll_std
            return np.tanh(z_score * factor)
        sent_raw = df['market_sentiment_score_D']
        hab_sent = sent_raw.rolling(window=hab_window, min_periods=1).mean()
        v_sent, a_sent, j_sent, _ = self._calculate_kinematic_tensors(df, df_index, 'market_sentiment_score_D', pmd_params)
        sent_kinematic = (v_sent * 0.5 + a_sent * 0.3 + j_sent * 0.2)
        psy_node = custom_robust_norm(hab_sent) * 0.5 + sent_kinematic * 0.5
        print(f"    -> [节点] HAB心理3D节点提取完成。均值: {psy_node.mean():.6f}, NaN: {psy_node.isna().sum()}")
        tension_raw = df['MA_POTENTIAL_TENSION_INDEX_D']
        hab_tension = tension_raw.rolling(window=hab_window, min_periods=1).mean()
        v_ten, a_ten, j_ten, _ = self._calculate_kinematic_tensors(df, df_index, 'MA_POTENTIAL_TENSION_INDEX_D', pmd_params)
        tension_kinematic = (v_ten * 0.5 + a_ten * 0.3 + j_ten * 0.2)
        phy_node = custom_robust_norm(hab_tension) * 0.5 + tension_kinematic * 0.5
        print(f"    -> [节点] HAB物理张力3D节点提取完成。均值: {phy_node.mean():.6f}, NaN: {phy_node.isna().sum()}")
        hotness_raw = df['THEME_HOTNESS_SCORE_D']
        game_int_raw = df['game_intensity_D']
        hab_hotness = hotness_raw.rolling(window=hab_window, min_periods=1).mean()
        hab_game = game_int_raw.rolling(window=hab_window, min_periods=1).mean()
        env_node = custom_robust_norm(hab_hotness) * 0.5 + custom_robust_norm(hab_game) * 0.5
        print(f"    -> [节点] HAB环境博弈节点提取完成。均值: {env_node.mean():.6f}, NaN: {env_node.isna().sum()}")
        raw_context = (psy_node * weights.get('psychological', 0.3) + phy_node * weights.get('physical_tension', 0.3) + env_node * weights.get('environmental', 0.4))
        def nonlinear_gain(x: pd.Series, gamma: float = 1.2) -> pd.Series:
            return np.sign(x) * (x.abs() ** gamma)
        gained_context = nonlinear_gain(raw_context)
        emo_ext = custom_robust_norm(df['STATE_EMOTIONAL_EXTREME_D']).abs()
        rubber_val = custom_robust_norm(df['MA_RUBBER_BAND_EXTENSION_D'].abs()).abs()
        circuit_breaker = pd.Series(1.0, index=df_index, dtype=np.float32)
        mask_dampen = (emo_ext > cb_params['emotion_threshold']) & (rubber_val < cb_params['tension_confirmation'])
        circuit_breaker[mask_dampen] = cb_params['dampening_factor']
        mask_amplify = (emo_ext > cb_params['emotion_threshold']) & (rubber_val >= cb_params['tension_confirmation'])
        circuit_breaker[mask_amplify] = cb_params['amplification_factor']
        is_parabolic = (df['STATE_PARABOLIC_WARNING_D'] > 0.5).astype(np.float32)
        penalty = pd.Series(1.0, index=df_index, dtype=np.float32)
        penalty.loc[is_parabolic == 1] = pmd_params.get('parabolic_penalty_factor', 0.4)
        final_context_modulator = np.exp(gained_context) * circuit_breaker * penalty
        print(f"  [探针-ContextMod-V6.8] 情境指数映射网络计算完成。最终无界调制器均值: {final_context_modulator.mean():.6f}, 极值: [{final_context_modulator.min():.6f}, {final_context_modulator.max():.6f}], NaN统计: {final_context_modulator.isna().sum()}")
        debug_info = {"node_psychological": psy_node, "node_physical": phy_node, "node_environmental": env_node, "node_circuit_breaker": circuit_breaker, "final_context_modulator": final_context_modulator}
        return final_context_modulator.astype(np.float32), debug_info

    def _calculate_rdi_for_pair(self, series_A: pd.Series, series_B: pd.Series, df_index: pd.Index, rdi_params: Dict, method_name: str, pair_name: str, field_quality: pd.Series = None) -> Tuple[pd.Series, Dict]:
        """【V6.9 · 自适应相空间RDI网络】接入张量级RDI演化、HAB共振记忆、定制Tanh映射与非线性增益，彻底移除截断。"""
        print(f"  [探针-RDI-V6.9] 启动 {pair_name} 3D相空间自适应相位锁定与RDI共振评估...")
        rdi_periods = rdi_params.get('rdi_periods', [5, 13, 21])
        base_reward = rdi_params.get('resonance_reward_factor', 0.15)
        base_penalty = rdi_params.get('divergence_penalty_factor', 0.25)
        i_reward = rdi_params.get('inflection_reward_factor', 0.10)
        rdi_weights = rdi_params.get('rdi_period_weights', {"5": 0.4, "13": 0.3, "21": 0.3})
        sensitivity = rdi_params.get('adaptive_sensitivity', 0.8)
        hab_window = rdi_params.get('rdi_hab_window', 5)
        epsilon = 1e-9
        modulator = (1.0 + field_quality * sensitivity) if field_quality is not None else pd.Series(1.0, index=df_index)
        eff_reward = base_reward * modulator
        eff_penalty = base_penalty / (modulator.abs() + epsilon)
        def custom_robust_norm(series: pd.Series, factor: float = 0.8) -> pd.Series:
            roll_mean = series.rolling(window=34, min_periods=1).mean()
            roll_std = series.rolling(window=34, min_periods=1).std().replace(0, epsilon)
            z_score = (series - roll_mean) / roll_std
            return np.tanh(z_score * factor)
        all_scores = {}
        for p in rdi_periods:
            t_A = series_A.rolling(window=p, min_periods=1).mean()
            t_B = series_B.rolling(window=p, min_periods=1).mean()
            v_A, a_A, j_A = t_A.diff(), t_A.diff(2), t_A.diff(3)
            v_B, a_B, j_B = t_B.diff(), t_B.diff(2), t_B.diff(3)
            kin_A = v_A * 0.5 + a_A * 0.3 + j_A * 0.2
            kin_B = v_B * 0.5 + a_B * 0.3 + j_B * 0.2
            res_term = (np.sign(kin_A) == np.sign(kin_B)).astype(np.float32) * eff_reward * (kin_A.abs() + kin_B.abs())
            div_term = (np.sign(kin_A) != np.sign(kin_B)).astype(np.float32) * eff_penalty * (kin_A.abs() + kin_B.abs())
            hab_res = res_term.rolling(window=hab_window, min_periods=1).mean()
            hab_div = div_term.rolling(window=hab_window, min_periods=1).mean()
            inf_A = (kin_A.shift(1) * kin_A < 0).astype(np.float32) * i_reward * kin_A.abs()
            inf_B = (kin_B.shift(1) * kin_B < 0).astype(np.float32) * i_reward * kin_B.abs()
            raw_period_rdi = hab_res - hab_div + (inf_A + inf_B) * 0.5
            all_scores[str(p)] = custom_robust_norm(raw_period_rdi)
        fused_rdi = pd.Series(0.0, index=df_index, dtype=np.float32)
        tot_w = 0.0
        for k, s in all_scores.items():
            w = rdi_weights.get(k, 0.0)
            fused_rdi += s * w
            tot_w += w
        fused_rdi = fused_rdi / (tot_w + epsilon)
        def nonlinear_gain(x: pd.Series, gamma: float = 1.2) -> pd.Series:
            return np.sign(x) * (x.abs() ** gamma)
        final_rdi = nonlinear_gain(fused_rdi)
        print(f"    -> [节点] 动态奖励极值: [{eff_reward.min():.6f}, {eff_reward.max():.6f}] | 动态惩罚极值: [{eff_penalty.min():.6f}, {eff_penalty.max():.6f}]")
        print(f"  [探针-RDI-V6.9] {pair_name} 最终自适应相位RDI均值: {final_rdi.mean():.6f}, 极值: [{final_rdi.min():.6f}, {final_rdi.max():.6f}], NaN统计: {final_rdi.isna().sum()}")
        return final_rdi.astype(np.float32), {}

    def _calculate_distribution_intent_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> pd.Series:
        """【V6.10 · 7步推演HAB主力派发张量引擎】引入获利压力与上影抛压，挂载3D派发张量与HAB存量对冲，定制Tanh归一化及非线性增益，彻底剥离防错掩码。"""
        print(f"  [探针-DistIntent-V6.10] 启动主力派发意图全息评估...")
        epsilon = 1e-9
        def custom_robust_norm(series: pd.Series, factor: float = 0.8) -> pd.Series:
            roll_mean = series.rolling(window=34, min_periods=1).mean()
            roll_std = series.rolling(window=34, min_periods=1).std().replace(0, epsilon)
            z_score = (series - roll_mean) / roll_std
            return np.tanh(z_score * factor)
        v_dist, a_dist, j_dist, _ = self._calculate_kinematic_tensors(df, df_index, 'distribution_energy_D', pmd_params)
        dist_energy_kin = (v_dist * 0.5 + a_dist * 0.3 + j_dist * 0.2)
        print(f"    -> [节点] 3D 派发能量张量提取完成。均值: {dist_energy_kin.mean():.6f}, NaN: {dist_energy_kin.isna().sum()}")
        acc_mf_21d = df['net_mf_amount_D'].rolling(window=21, min_periods=1).sum()
        today_mf_out = df['net_mf_amount_D'].where(df['net_mf_amount_D'] < 0, 0.0).abs()
        flow_ratio = today_mf_out / (acc_mf_21d.abs() + epsilon)
        dampener_series = pd.Series(np.where((acc_mf_21d > 0) & (df['net_mf_amount_D'] < 0) & (flow_ratio < 0.1), pmd_params.get('intent_dampening_factor', 0.25), 1.0), index=df_index)
        print(f"    -> [节点] HAB存量衰减器提取完成。极值: [{dampener_series.min():.6f}, {dampener_series.max():.6f}]")
        v_p, a_p, j_p, _ = self._calculate_kinematic_tensors(df, df_index, 'close_D', pmd_params)
        price_jerk_collapsed = 1.0 - (v_p * 0.5 + a_p * 0.3 + j_p * 0.2).abs()
        sell_ofi_raw = df['SMART_MONEY_INST_NET_BUY_D'].where(df['SMART_MONEY_INST_NET_BUY_D'] < 0, 0.0).abs()
        sell_ofi_norm = custom_robust_norm(sell_ofi_raw).abs()
        profit_pressure_norm = custom_robust_norm(df['profit_pressure_D'])
        upper_shadow_norm = custom_robust_norm(df['upper_shadow_selling_pressure_D'])
        print(f"    -> [节点] 获利压力均值: {profit_pressure_norm.mean():.6f} | 上影抛压均值: {upper_shadow_norm.mean():.6f}")
        raw_dist_intent = (dist_energy_kin * 0.3 + price_jerk_collapsed * 0.2 + sell_ofi_norm * 0.2 + profit_pressure_norm * 0.15 + upper_shadow_norm * 0.15) * dampener_series
        def nonlinear_gain(x: pd.Series, gamma: float = 1.2) -> pd.Series:
            return np.sign(x) * (x.abs() ** gamma)
        final_dist_intent = nonlinear_gain(raw_dist_intent)
        print(f"  [探针-DistIntent-V6.10] 派发意图计算完成。非线性最终得分均值: {final_dist_intent.mean():.6f}, 极值: [{final_dist_intent.min():.6f}, {final_dist_intent.max():.6f}], NaN统计: {final_dist_intent.isna().sum()}")
        return final_dist_intent.astype(np.float32)

    def _calculate_covert_accumulation_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> pd.Series:
        """【V7.0 · 7步推演HAB隐蔽吸筹张量引擎】全面引入军械库增广数据、3D动力学张量、HAB历史缓冲、专属Tanh归一化与非线性增益，彻底重构隐蔽吸筹逻辑。"""
        print(f"  [探针-CovertAcc-V7.0] 启动隐蔽吸筹高维评估...")
        epsilon = 1e-9
        deadzone = pmd_params.get('kinematic_soft_deadzone', 1e-5)
        fib_periods = pmd_params.get('fib_periods', [5, 13, 21, 34])
        mtf_w = pmd_params.get('mtf_decay_weights', {"5": 0.4, "13": 0.3, "21": 0.2, "34": 0.1})
        def custom_robust_norm(series: pd.Series, factor: float = 0.8, window: int = 34) -> pd.Series:
            roll_mean = series.rolling(window=window, min_periods=1).mean()
            roll_std = series.rolling(window=window, min_periods=1).std().replace(0, epsilon)
            z_score = (series - roll_mean) / roll_std
            return np.tanh(z_score * factor)
        def nonlinear_gain(x: pd.Series, gamma: float = 1.2) -> pd.Series:
            return np.sign(x) * (x.abs() ** gamma)
        def extract_kinematic_node(base_col: str, gravity: float = 1.0) -> pd.Series:
            fused_v = pd.Series(0.0, index=df_index, dtype=np.float32)
            fused_a = pd.Series(0.0, index=df_index, dtype=np.float32)
            fused_j = pd.Series(0.0, index=df_index, dtype=np.float32)
            tot_w = 0.0
            for p in fib_periods:
                w = mtf_w.get(str(p), 0.0)
                if w == 0: continue
                s_raw = df[f'SLOPE_{p}_{base_col}']
                a_raw = df[f'ACCEL_{p}_{base_col}']
                j_raw = df[f'JERK_{p}_{base_col}']
                s_soft = np.sign(s_raw) * np.maximum(s_raw.abs() - deadzone, 0.0)
                a_soft = np.sign(a_raw) * np.maximum(a_raw.abs() - deadzone, 0.0)
                j_soft = np.sign(j_raw) * np.maximum(j_raw.abs() - deadzone, 0.0)
                a_asym = np.where(a_soft < 0, a_soft * gravity, a_soft)
                j_asym = np.where(j_soft < 0, j_soft * gravity, j_soft)
                fused_v += custom_robust_norm(pd.Series(s_soft, index=df_index)) * w
                fused_a += custom_robust_norm(pd.Series(a_asym, index=df_index)) * w
                fused_j += custom_robust_norm(pd.Series(j_asym, index=df_index)) * w
                tot_w += w
            if tot_w > 0:
                fused_v, fused_a, fused_j = fused_v / tot_w, fused_a / tot_w, fused_j / tot_w
            return (fused_v * 0.5 + fused_a * 0.3 + fused_j * 0.2)
        abs_energy_kin = extract_kinematic_node('absorption_energy_D', gravity=0.5)
        stealth_flow_kin = extract_kinematic_node('stealth_flow_ratio_D', gravity=0.5)
        price_kin = extract_kinematic_node('close_D', gravity=1.5)
        acc_mf_21d = df['net_mf_amount_D'].rolling(window=21, min_periods=1).sum()
        today_mf = df['net_mf_amount_D']
        mf_resilience_ratio = today_mf.abs() / (acc_mf_21d.abs() + epsilon)
        acc_dampener_series = pd.Series(np.where((acc_mf_21d > 0) & (today_mf < 0) & (mf_resilience_ratio < 0.2), pmd_params.get('intent_dampening_factor', 0.25), 1.0), index=df_index)
        hab_abs = df['absorption_energy_D'].rolling(window=21, min_periods=1).mean()
        today_abs = df['absorption_energy_D']
        abs_surge = custom_robust_norm(today_abs - hab_abs)
        action_node = (abs_energy_kin * 0.5 + stealth_flow_kin * 0.3 + abs_surge * 0.2) * acc_dampener_series
        price_deadzone_lock = 1.0 - price_kin.abs()
        vpa_eff_norm = custom_robust_norm(df['VPA_EFFICIENCY_D'])
        env_node = price_deadzone_lock * 0.6 + vpa_eff_norm.abs() * 0.4
        chip_stb_norm = custom_robust_norm(df['chip_stability_D']).abs()
        abnormal_ratio_norm = custom_robust_norm(df['tick_abnormal_volume_ratio_D']).abs()
        acc_signal_norm = custom_robust_norm(df['accumulation_signal_score_D']).abs()
        consequence_node = chip_stb_norm * 0.4 + abnormal_ratio_norm * 0.3 + acc_signal_norm * 0.3
        raw_covert_acc = action_node * 0.4 + env_node * 0.3 + consequence_node * 0.3
        final_covert_acc = nonlinear_gain(raw_covert_acc, gamma=1.2)
        print(f"    -> [节点] HAB资金21D净流入均值: {acc_mf_21d.mean():.6f}, 包含NaN: {acc_mf_21d.isna().sum()}")
        print(f"  [探针-CovertAcc-V7.0] 隐蔽吸筹均值: {final_covert_acc.mean():.6f}, 极值: [{final_covert_acc.min():.6f}, {final_covert_acc.max():.6f}]")
        return final_covert_acc.astype(np.float32)

    def _calculate_momentum_quality_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> pd.Series:
        """【V3.4.0 · 7步推演HAB动量品质张量引擎】全面引入增广数据、3D动力学张量、HAB历史缓冲、专属Tanh归一化与非线性增益，重构动量品质逻辑。"""
        print(f"  [探针-MomQuality-V3.4.0] 启动动量品质高维评估...")
        epsilon = 1e-9
        deadzone = pmd_params.get('kinematic_soft_deadzone', 1e-5)
        fib_periods = pmd_params.get('fib_periods', [5, 13, 21, 34])
        mtf_w = pmd_params.get('mtf_decay_weights', {"5": 0.4, "13": 0.3, "21": 0.2, "34": 0.1})
        def custom_robust_norm(series: pd.Series, factor: float = 0.8, window: int = 34) -> pd.Series:
            roll_mean = series.rolling(window=window, min_periods=1).mean()
            roll_std = series.rolling(window=window, min_periods=1).std().replace(0, epsilon)
            z_score = (series - roll_mean) / roll_std
            return np.tanh(z_score * factor)
        def nonlinear_gain(x: pd.Series, gamma: float = 1.2) -> pd.Series:
            return np.sign(x) * (x.abs() ** gamma)
        def extract_kinematic_node(base_col: str, gravity: float = 1.0) -> pd.Series:
            fused_v = pd.Series(0.0, index=df_index, dtype=np.float32)
            fused_a = pd.Series(0.0, index=df_index, dtype=np.float32)
            fused_j = pd.Series(0.0, index=df_index, dtype=np.float32)
            tot_w = 0.0
            for p in fib_periods:
                w = mtf_w.get(str(p), 0.0)
                if w == 0: continue
                s_raw = df[f'SLOPE_{p}_{base_col}']
                a_raw = df[f'ACCEL_{p}_{base_col}']
                j_raw = df[f'JERK_{p}_{base_col}']
                s_soft = np.sign(s_raw) * np.maximum(s_raw.abs() - deadzone, 0.0)
                a_soft = np.sign(a_raw) * np.maximum(a_raw.abs() - deadzone, 0.0)
                j_soft = np.sign(j_raw) * np.maximum(j_raw.abs() - deadzone, 0.0)
                a_asym = np.where(a_soft < 0, a_soft * gravity, a_soft)
                j_asym = np.where(j_soft < 0, j_soft * gravity, j_soft)
                fused_v += custom_robust_norm(pd.Series(s_soft, index=df_index)) * w
                fused_a += custom_robust_norm(pd.Series(a_asym, index=df_index)) * w
                fused_j += custom_robust_norm(pd.Series(j_asym, index=df_index)) * w
                tot_w += w
            if tot_w > 0:
                fused_v, fused_a, fused_j = fused_v / tot_w, fused_a / tot_w, fused_j / tot_w
            return (fused_v * 0.5 + fused_a * 0.3 + fused_j * 0.2)
        macdh_kin = extract_kinematic_node('MACDh_13_34_8_D', gravity=1.0)
        rsi_kin = extract_kinematic_node('RSI_13_D', gravity=1.0)
        vpa_eff_kin = extract_kinematic_node('VPA_EFFICIENCY_D', gravity=1.0)
        trend_str_norm = custom_robust_norm(df['uptrend_strength_D'])
        acc_coord_13d = df['HM_COORDINATED_ATTACK_D'].rolling(window=13, min_periods=1).mean()
        acc_coord_21d = df['HM_COORDINATED_ATTACK_D'].rolling(window=21, min_periods=1).mean()
        today_coord = df['HM_COORDINATED_ATTACK_D']
        coord_surge = custom_robust_norm(today_coord - acc_coord_21d)
        hab_coord_node = custom_robust_norm(acc_coord_13d) * 0.6 + coord_surge * 0.4
        dot_product_coherence = (macdh_kin * rsi_kin) + (macdh_kin * vpa_eff_kin) + (rsi_kin * vpa_eff_kin)
        coherence_penalty = np.where(dot_product_coherence < 0, 0.5, 1.0)
        raw_mom_quality = (macdh_kin.abs() * 0.3 + rsi_kin.abs() * 0.3 + vpa_eff_kin.abs() * 0.2 + trend_str_norm.abs() * 0.1 + hab_coord_node.abs() * 0.1) * coherence_penalty
        final_mom_quality = nonlinear_gain(pd.Series(raw_mom_quality, index=df_index), gamma=1.5)
        print(f"    -> [节点] MACD动力均值: {macdh_kin.mean():.4f} | RSI动力均值: {rsi_kin.mean():.4f} | VPA效率均值: {vpa_eff_kin.mean():.4f}")
        print(f"    -> [节点] HAB协同记忆均值: {hab_coord_node.mean():.4f} | 张量相干惩罚均值: {pd.Series(coherence_penalty).mean():.4f}")
        print(f"  [探针-MomQuality-V3.4.0] 最终动量品质均值: {final_mom_quality.mean():.4f}, 极值: [{final_mom_quality.min():.4f}, {final_mom_quality.max():.4f}]")
        return final_mom_quality.astype(np.float32)

    def _calculate_stability_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> pd.Series:
        """【V3.4.0 · 7步推演HAB市场稳定性张量引擎】引入筹码/布林带增广数据、3D张量死区锁定、HAB缓冲池、专属归一化及非线性增益。"""
        print(f"  [探针-Stability-V3.4.0] 启动市场稳定性高维重构评估...")
        epsilon = 1e-9
        deadzone = pmd_params.get('kinematic_soft_deadzone', 1e-5)
        fib_periods = pmd_params.get('fib_periods', [5, 13, 21, 34])
        mtf_w = pmd_params.get('mtf_decay_weights', {"5": 0.4, "13": 0.3, "21": 0.2, "34": 0.1})
        def custom_robust_norm(series: pd.Series, factor: float = 0.8, window: int = 34) -> pd.Series:
            roll_mean = series.rolling(window=window, min_periods=1).mean()
            roll_std = series.rolling(window=window, min_periods=1).std().replace(0, epsilon)
            z_score = (series - roll_mean) / roll_std
            return np.tanh(z_score * factor)
        def nonlinear_gain(x: pd.Series, gamma: float = 1.2) -> pd.Series:
            return np.sign(x) * (x.abs() ** gamma)
        def extract_kinematic_node(base_col: str, gravity: float = 1.0) -> pd.Series:
            fused_v = pd.Series(0.0, index=df_index, dtype=np.float32)
            fused_a = pd.Series(0.0, index=df_index, dtype=np.float32)
            fused_j = pd.Series(0.0, index=df_index, dtype=np.float32)
            tot_w = 0.0
            for p in fib_periods:
                w = mtf_w.get(str(p), 0.0)
                if w == 0: continue
                s_raw = df[f'SLOPE_{p}_{base_col}']
                a_raw = df[f'ACCEL_{p}_{base_col}']
                j_raw = df[f'JERK_{p}_{base_col}']
                s_soft = np.sign(s_raw) * np.maximum(s_raw.abs() - deadzone, 0.0)
                a_soft = np.sign(a_raw) * np.maximum(a_raw.abs() - deadzone, 0.0)
                j_soft = np.sign(j_raw) * np.maximum(j_raw.abs() - deadzone, 0.0)
                a_asym = np.where(a_soft < 0, a_soft * gravity, a_soft)
                j_asym = np.where(j_soft < 0, j_soft * gravity, j_soft)
                fused_v += custom_robust_norm(pd.Series(s_soft, index=df_index)) * w
                fused_a += custom_robust_norm(pd.Series(a_asym, index=df_index)) * w
                fused_j += custom_robust_norm(pd.Series(j_asym, index=df_index)) * w
                tot_w += w
            if tot_w > 0:
                fused_v, fused_a, fused_j = fused_v / tot_w, fused_a / tot_w, fused_j / tot_w
            return (fused_v * 0.5 + fused_a * 0.3 + fused_j * 0.2)
        acc_turnover_21d = df['TURNOVER_STABILITY_INDEX_D'].rolling(window=21, min_periods=1).mean()
        turnover_surge = df['TURNOVER_STABILITY_INDEX_D'] - acc_turnover_21d
        hab_turnover_node = custom_robust_norm(acc_turnover_21d) * 0.7 + custom_robust_norm(turnover_surge) * 0.3
        acc_chip_21d = df['chip_stability_D'].rolling(window=21, min_periods=1).mean()
        chip_surge = df['chip_stability_D'] - acc_chip_21d
        hab_chip_node = custom_robust_norm(acc_chip_21d) * 0.7 + custom_robust_norm(chip_surge) * 0.3
        flow_chip_node = hab_turnover_node.abs() * 0.5 + hab_chip_node.abs() * 0.5
        entropy_inv = 1.0 - custom_robust_norm(df['PRICE_ENTROPY_D']).abs()
        bbw_inv = 1.0 - custom_robust_norm(df['BBW_21_2.0_D']).abs()
        structural_node = entropy_inv * 0.5 + bbw_inv * 0.5
        price_kin = extract_kinematic_node('close_D', gravity=1.0)
        price_deadzone_lock = (1.0 - price_kin.abs()).clip(lower=0.0)
        raw_stability = flow_chip_node * 0.4 + structural_node * 0.3 + price_deadzone_lock * 0.3
        final_stability = nonlinear_gain(raw_stability, gamma=1.2)
        print(f"    -> [节点] 换手HAB稳定度均值: {hab_turnover_node.mean():.4f} | 筹码HAB稳定度均值: {hab_chip_node.mean():.4f}")
        print(f"    -> [节点] 负熵结构秩序度均值: {entropy_inv.mean():.4f} | BBW布林压缩度均值: {bbw_inv.mean():.4f}")
        print(f"    -> [节点] 价格3D动力死区锁定均值: {price_deadzone_lock.mean():.4f}")
        print(f"  [探针-Stability-V3.4.0] 最终稳定性品质均值: {final_stability.mean():.4f}, 极值: [{final_stability.min():.4f}, {final_stability.max():.4f}]")
        return final_stability.astype(np.float32)

    def _calculate_chip_historical_potential_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> pd.Series:
        """【V3.4.0 · 7步推演HAB筹码历史势能张量引擎】引入集中度增广数据、3D累积张量、HAB历史缓冲、专属归一化及非线性势能增益。"""
        print(f"  [探针-ChipPotential-V3.4.0] 启动筹码历史势能高维重构评估...")
        epsilon = 1e-9
        deadzone = pmd_params.get('kinematic_soft_deadzone', 1e-5)
        fib_periods = pmd_params.get('fib_periods', [5, 13, 21, 34])
        mtf_w = pmd_params.get('mtf_decay_weights', {"5": 0.4, "13": 0.3, "21": 0.2, "34": 0.1})
        def custom_robust_norm(series: pd.Series, factor: float = 0.8, window: int = 55) -> pd.Series:
            roll_mean = series.rolling(window=window, min_periods=1).mean()
            roll_std = series.rolling(window=window, min_periods=1).std().replace(0, epsilon)
            z_score = (series - roll_mean) / roll_std
            return np.tanh(z_score * factor)
        def nonlinear_gain(x: pd.Series, gamma: float = 1.2) -> pd.Series:
            return np.sign(x) * (x.abs() ** gamma)
        def extract_kinematic_node(base_col: str, gravity: float = 1.0) -> pd.Series:
            fused_v = pd.Series(0.0, index=df_index, dtype=np.float32)
            fused_a = pd.Series(0.0, index=df_index, dtype=np.float32)
            fused_j = pd.Series(0.0, index=df_index, dtype=np.float32)
            tot_w = 0.0
            for p in fib_periods:
                w = mtf_w.get(str(p), 0.0)
                if w == 0: continue
                s_raw = df[f'SLOPE_{p}_{base_col}']
                a_raw = df[f'ACCEL_{p}_{base_col}']
                j_raw = df[f'JERK_{p}_{base_col}']
                s_soft = np.sign(s_raw) * np.maximum(s_raw.abs() - deadzone, 0.0)
                a_soft = np.sign(a_raw) * np.maximum(a_raw.abs() - deadzone, 0.0)
                j_soft = np.sign(j_raw) * np.maximum(j_raw.abs() - deadzone, 0.0)
                a_asym = np.where(a_soft < 0, a_soft * gravity, a_soft)
                j_asym = np.where(j_soft < 0, j_soft * gravity, j_soft)
                fused_v += custom_robust_norm(pd.Series(s_soft, index=df_index)) * w
                fused_a += custom_robust_norm(pd.Series(a_asym, index=df_index)) * w
                fused_j += custom_robust_norm(pd.Series(j_asym, index=df_index)) * w
                tot_w += w
            if tot_w > 0:
                fused_v, fused_a, fused_j = fused_v / tot_w, fused_a / tot_w, fused_j / tot_w
            return (fused_v * 0.5 + fused_a * 0.3 + fused_j * 0.2)
        acc_score_kin = extract_kinematic_node('accumulation_score_D', gravity=1.0)
        chip_conc_kin = extract_kinematic_node('chip_concentration_ratio_D', gravity=1.0)
        hab_acc_55d = df['accumulation_score_D'].rolling(window=55, min_periods=1).mean()
        today_acc = df['accumulation_score_D']
        acc_surge = custom_robust_norm(today_acc - hab_acc_55d)
        hab_acc_node = custom_robust_norm(hab_acc_55d) * 0.6 + acc_surge * 0.4
        hab_chip_conc_55d = df['chip_concentration_ratio_D'].rolling(window=55, min_periods=1).mean()
        today_conc = df['chip_concentration_ratio_D']
        conc_surge = custom_robust_norm(today_conc - hab_chip_conc_55d)
        hab_conc_node = custom_robust_norm(hab_chip_conc_55d) * 0.6 + conc_surge * 0.4
        chip_stb_norm = custom_robust_norm(df['chip_stability_D']).abs()
        structure_node = chip_stb_norm * 0.5 + hab_conc_node.abs() * 0.5
        kinematic_node = (acc_score_kin.clip(lower=0) * 0.6 + chip_conc_kin.clip(lower=0) * 0.4)
        raw_potential = structure_node * 0.4 + hab_acc_node.abs() * 0.3 + kinematic_node * 0.3
        final_potential = nonlinear_gain(raw_potential, gamma=1.2)
        print(f"    -> [节点] 筹码集中HAB节点均值: {hab_conc_node.mean():.4f} | 积累分数HAB节点均值: {hab_acc_node.mean():.4f}")
        print(f"    -> [节点] 结构稳固度均值: {structure_node.mean():.4f} | 3D动力张量均值: {kinematic_node.mean():.4f}")
        print(f"  [探针-ChipPotential-V3.4.0] 最终筹码势能均值: {final_potential.mean():.4f}, 极值: [{final_potential.min():.4f}, {final_potential.max():.4f}]")
        return final_potential.astype(np.float32)

    def _calculate_liquidity_tide_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> pd.Series:
        """
        【V7.0 · HAB流动性潮汐张量模型】
        基于七步推演法重构：融合宏观换手动力学、资金HAB存量记忆与微观逐笔结构，
        采用专属Tanh归一化与非线性增益，构建全息流动性健康度评分。
        """
        print(f"  [探针-TideHAB-V7.0] 启动流动性潮汐全息评估...")
        # --- 参数提取 ---
        weights = pmd_params.get('tide_weights', {"tidal_force": 0.3, "water_level": 0.4, "micro_structure": 0.3})
        hab_window = pmd_params.get('tide_hab_window', 21)
        epsilon = 1e-9
        # --- 第四步：定义专属 Tanh 鲁棒归一化 (Local Custom Norm) ---
        def custom_tide_norm(series: pd.Series, window: int = 34, sensitivity: float = 0.6) -> pd.Series:
            """针对流动性数据的尖峰肥尾特征设计的归一化"""
            roll_mean = series.rolling(window=window, min_periods=1).mean()
            roll_std = series.rolling(window=window, min_periods=1).std().replace(0, epsilon)
            z_score = (series - roll_mean) / roll_std
            # 使用 tanh 柔性压缩异常值
            return np.tanh(z_score * sensitivity)
        # --- 第五步：定义非线性增益函数 ---
        def nonlinear_gain(x: pd.Series, gamma: float = 1.5) -> pd.Series:
            """强化潮汐极值效应，压制中间态噪音"""
            return np.sign(x) * (x.abs() ** gamma)
        # --- 第二步：换手率 3D 动力学 (Tidal Force) ---
        # 引入换手率作为潮汐质量 
        # 使用 soft_deadzone 避免低换手区的零基陷阱
        v_turn, a_turn, j_turn, _ = self._calculate_kinematic_tensors(df, df_index, 'turnover_rate_D', pmd_params)
        raw_force = (v_turn * 0.5 + a_turn * 0.3 + j_turn * 0.2)
        # 只有当基础换手率超过一定阈值(如1%)时，动力学才有效，否则视为死水
        active_mask = df['turnover_rate_D'] > 1.0 
        tidal_force = nonlinear_gain(custom_tide_norm(raw_force)) * active_mask.astype(np.float32)
        print(f"    -> [节点] 潮汐动力学提取完成。均值: {tidal_force.mean():.6f}, 极值: [{tidal_force.min():.6f}, {tidal_force.max():.6f}]")
        # --- 第三步：HAB 存量记忆系统 (Water Level) ---
        # 引入主力净额 
        mf_amount = df['net_mf_amount_D'].fillna(0)
        # 计算 21日 累积水位 (Accumulated Water Level)
        hab_accumulated = mf_amount.rolling(window=hab_window, min_periods=1).sum()
        # 计算潮汐韧性：当日流出相对于历史累积的比例
        # 如果累积为正(水位深)且当日流出(负)，但流出量占比很小，则给予豁免
        flow_ratio = mf_amount.abs() / (hab_accumulated.abs() + epsilon)
        # 豁免逻辑：HAB > 0 (蓄水态) AND Current < 0 (流出) AND Ratio < 0.15 (微量) -> 视为良性分歧，不扣分
        benign_divergence = (hab_accumulated > 0) & (mf_amount < 0) & (flow_ratio < 0.15)
        # 基础水位得分
        water_level_raw = custom_tide_norm(hab_accumulated)
        # 应用豁免修正：良性分歧时，强制修正水位得分为正向支撑
        water_level_adjusted = pd.Series(np.where(benign_divergence, 0.2, water_level_raw), index=df_index)
        # 叠加 flow_percentile_D  确认历史相对位置
        pct_pos = df['flow_percentile_D'].fillna(0.5) * 2 - 1 # 映射到 [-1, 1]
        final_water_level = (water_level_adjusted * 0.7 + pct_pos * 0.3)
        print(f"    -> [节点] HAB水位存量分析完成。均值: {final_water_level.mean():.6f}, 良性分歧修正点数: {benign_divergence.sum()}")
        # --- 微观结构与波动率惩罚 ---
        # 引入逐笔筹码流 
        tick_flow = df['tick_level_chip_flow_D'].fillna(0)
        micro_structure = nonlinear_gain(custom_tide_norm(tick_flow, sensitivity=0.8))
        # 引入波动率惩罚 
        volatility = df['flow_volatility_13d_D'].fillna(0)
        vol_penalty = (1.0 - custom_tide_norm(volatility).abs()).clip(lower=0.1) # 波动越大，置信度越低，系数越小
        # --- 第六步 & 第七步：张量融合 ---
        # 融合逻辑：(动力 + 水位 + 微观) * 稳定性系数
        raw_tide_score = (tidal_force * weights.get("tidal_force", 0.3) + 
                          final_water_level * weights.get("water_level", 0.4) + 
                          micro_structure * weights.get("micro_structure", 0.3))
        final_tide_score = raw_tide_score * vol_penalty
        print(f"  [探针-TideHAB-V7.0] 流动性潮汐计算完成。最终得分均值: {final_tide_score.mean():.6f}, NaN统计: {final_tide_score.isna().sum()}")
        return final_tide_score.astype(np.float32)

    def _calculate_market_constitution_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> pd.Series:
        """
        【V7.0 · HAB市场体质张量模型】
        基于七步推演法重构：融合ADX趋势活性、筹码结构支撑与情绪HAB记忆，
        引入短板效应（木桶原理）与混合归一化，构建非对称的市场健康度体质评分。
        """
        print(f"  [探针-ConstHAB-V7.0] 启动市场体质全息评估...")
        # --- 参数提取 ---
        weights = pmd_params.get('constitution_weights', {"structural_support": 0.4, "trend_vitality": 0.3, "position_risk": 0.3})
        hab_window = pmd_params.get('constitution_hab_window', 21)
        epsilon = 1e-9
        # --- 第四步：定义专属混合归一化 (Hybrid Normalization) ---
        def custom_constitution_norm(series: pd.Series, mode: str = 'sigmoid') -> pd.Series:
            """
            针对体质指标设计的归一化：
            - sigmoid: 适用于单向指标（如ADX，筹码稳定），对低值敏感。
            - tanh: 适用于双向指标（如位置偏离），关注两端极值。
            """
            roll_mean = series.rolling(window=34, min_periods=1).mean()
            roll_std = series.rolling(window=34, min_periods=1).std().replace(0, epsilon)
            z_score = (series - roll_mean) / roll_std
            if mode == 'sigmoid':
                return 1 / (1 + np.exp(-z_score))  # 映射到 [0, 1]
            else: # tanh
                return np.tanh(z_score * 0.6)      # 映射到 [-1, 1]
        # --- 第五步：定义非线性短板惩罚函数 ---
        def bucket_penalty(scores: List[pd.Series]) -> pd.Series:
            """木桶效应：体质受限于最弱的一环"""
            min_score = pd.concat(scores, axis=1).min(axis=1)
            # 如果最弱项低于 0.3，则产生强烈的非线性惩罚
            penalty = np.where(min_score < 0.3, min_score * min_score, 1.0) 
            return pd.Series(penalty, index=min_score.index)
        # --- 第二步 & 第三步：结构支撑 (Structural Support) ---
        # 核心：筹码稳定性 (chip_stability_D)
        # 引入 HAB 存量记忆：体质由长期筹码结构决定
        chip_stb_raw = df['chip_stability_D'].fillna(0)
        hab_chip_stb = chip_stb_raw.rolling(window=hab_window, min_periods=1).mean()
        # 引入 3D 动力学：关注筹码松动的加速度
        v_chip, a_chip, _, _ = self._calculate_kinematic_tensors(df, df_index, 'chip_stability_D', pmd_params)
        # 如果筹码稳定性正在加速下降 (v < 0, a < 0)，则是体质恶化的预警
        deterioration_signal = ((v_chip < 0) & (a_chip < 0)).astype(np.float32)
        # 结构得分：HAB基础分 - 恶化扣分
        structural_support = custom_constitution_norm(hab_chip_stb, mode='sigmoid') * (1.0 - deterioration_signal * 0.3)
        print(f"    -> [节点] 结构支撑(筹码HAB)计算完成。均值: {structural_support.mean():.6f}")
        # --- 第二步 & 第三步：趋势活性 (Trend Vitality) ---
        # 核心：ADX (ADX_14_D) + 市场情绪 (market_sentiment_score_D)
        # ADX 代表“肌肉力量”，情绪代表“神经系统”
        adx_raw = df['ADX_14_D'].fillna(0)
        sentiment_raw = df['market_sentiment_score_D'].fillna(0.5)
        # ADX 3D 动力学
        v_adx, _, _, _ = self._calculate_kinematic_tensors(df, df_index, 'ADX_14_D', pmd_params)
        # 只有当 ADX > 20 且 V > 0 时，才视为有效活性；ADX 下降代表体质虚弱
        adx_vitality = custom_constitution_norm(adx_raw, mode='sigmoid') * (1.0 + np.sign(v_adx) * 0.2)
        # 情绪 HAB：平滑短期情绪波动
        hab_sentiment = sentiment_raw.rolling(window=13, min_periods=1).mean()
        sentiment_health = custom_constitution_norm(hab_sentiment, mode='tanh').abs() # 无论多空，情绪饱满即为活性，但过热过冷需后续修正
        trend_vitality = (adx_vitality * 0.6 + sentiment_health * 0.4)
        print(f"    -> [节点] 趋势活性(ADX+情绪)计算完成。均值: {trend_vitality.mean():.6f}")
        # --- 第一步：位置风险修正 (Position Risk) ---
        # 核心：价格百分位 (price_percentile_position_D) + 成本乖离
        # 即使体质再好，处于 99% 历史高位也是一种透支
        pos_raw = df['price_percentile_position_D'].fillna(0.5)
        # 成本乖离：当前价格 / 50%筹码成本
        cost_line = df['cost_50pct_D'].replace(0, np.nan).fillna(df['close_D'])
        bias_ratio = (df['close_D'] / cost_line) - 1.0
        # 归一化：位置越高/乖离越大，风险分越高（即体质分越低）
        # 使用 sigmoid 将高位风险快速放大
        pos_risk_score = 1.0 - custom_constitution_norm(pos_raw, mode='sigmoid')
        bias_safety_score = 1.0 - custom_constitution_norm(bias_ratio.abs(), mode='sigmoid')
        # 位置风险得分（越高越安全）
        position_safety = (pos_risk_score * 0.5 + bias_safety_score * 0.5)
        print(f"    -> [节点] 位置安全度计算完成。均值: {position_safety.mean():.6f}")
        # --- 第六步 & 第七步：木桶效应融合 ---
        # 计算短板惩罚系数
        bucket_factor = bucket_penalty([structural_support, trend_vitality, position_safety])
        # 加权融合
        raw_constitution = (structural_support * weights.get("structural_support", 0.4) + 
                            trend_vitality * weights.get("trend_vitality", 0.3) + 
                            position_safety * weights.get("position_risk", 0.3))
        # 应用短板惩罚
        final_constitution = raw_constitution * bucket_factor
        # 最终逻辑分歧处理：复用原有逻辑，当位置极度中庸时，体质特征可能不明显，但在HAB框架下，
        # 中庸位置且结构稳定(Stable)本身就是一种好体质，因此移除原有的“向0塌陷”逻辑，
        # 改为保留 HAB 记忆的真实体质分。
        print(f"  [探针-ConstHAB-V7.0] 市场体质计算完成。最终得分均值: {final_constitution.mean():.6f}, 短板惩罚均值: {bucket_factor.mean():.6f}")
        return final_constitution.astype(np.float32)

    def _calculate_market_tension_score(self, df: pd.DataFrame, df_index: pd.Index, raw_data: Dict, pmd_params: Dict, method_name: str) -> pd.Series:
        """
        【V7.0 · HAB市场张力物理模型】
        基于七步推演法重构：融合多频段乖离、ATR波动率约束与HAB弹性疲劳记忆，
        引入弹性势能非线性增益，构建符合物理定律的市场张力评分。
        """
        print(f"  [探针-TensionHAB-V7.0] 启动市场物理张力全息评估...")
        # --- 参数提取 ---
        weights = pmd_params.get('tension_weights', {"rubber_band": 0.3, "potential_index": 0.4, "bias_struct": 0.3})
        hab_window = pmd_params.get('tension_hab_window', 21)
        epsilon = 1e-9
        # --- 第四步：定义专属波动率自适应归一化 (ATR Adaptive Norm) ---
        # 准备 ATR 数据用于动态调整阈值
        atr_series = df['ATR_14_D'].fillna(0)
        close_series = df['close_D'].fillna(1)
        atr_pct = atr_series / (close_series + epsilon) # 相对波动率
        def custom_tension_norm(series: pd.Series, volatility_adj: pd.Series, sensitivity: float = 0.8) -> pd.Series:
            """
            引入波动率约束的归一化：
            波动率越高(volatility_adj大)，分母越大，相同的数值产生的Z-Score越小(容忍度越高)。
            """
            roll_mean = series.rolling(window=34, min_periods=1).mean()
            roll_std = series.rolling(window=34, min_periods=1).std().replace(0, epsilon)
            # 动态标准差：波动率每增加1%，标准差扩容 (1 + vol * 5) 倍
            dynamic_std = roll_std * (1.0 + volatility_adj * 5.0) 
            z_score = (series - roll_mean) / dynamic_std
            return np.tanh(z_score * sensitivity)
        # --- 第五步：定义弹性势能增益函数 (Hooke's Gain) ---
        def elastic_energy_gain(x: pd.Series) -> pd.Series:
            """遵循 E = 1/2*k*x^2，对极端张力进行平方级放大"""
            return np.sign(x) * (x.abs() ** 2.0)
        # --- 第二步 & 第三步：中频张力 (Potential Index) ---
        # 核心：MA_POTENTIAL_TENSION_INDEX_D
        tension_raw = df['MA_POTENTIAL_TENSION_INDEX_D'].fillna(0)
        # 3D 动力学：关注张力积聚的加速度
        v_ten, a_ten, j_ten, _ = self._calculate_kinematic_tensors(df, df_index, 'MA_POTENTIAL_TENSION_INDEX_D', pmd_params)
        # HAB 记忆：计算突发应力 (Acute Stress)
        hab_tension = tension_raw.rolling(window=hab_window, min_periods=1).mean()
        acute_stress = tension_raw - hab_tension
        # 中频得分：基础突发应力 + 动力学冲击 (速度同向加速代表张力急剧恶化)
        mid_freq_score = custom_tension_norm(acute_stress, atr_pct) + (v_ten * 0.3 + a_ten * 0.2)
        mid_freq_final = elastic_energy_gain(mid_freq_score.clip(-1, 1))
        print(f"    -> [节点] 中频张力(HAB突发)计算完成。均值: {mid_freq_final.mean():.6f}")
        # --- 第二步 & 第三步：高频张力 (Rubber Band) ---
        # 核心：MA_RUBBER_BAND_EXTENSION_D
        rubber_raw = df['MA_RUBBER_BAND_EXTENSION_D'].fillna(0)
        # 对于高频数据，直接计算其绝对值的张力
        v_rub, a_rub, _, _ = self._calculate_kinematic_tensors(df, df_index, 'MA_RUBBER_BAND_EXTENSION_D', pmd_params)
        # 橡皮筋不仅看拉伸长度，更看回弹速度(Velocity的反向)
        # 如果 rubber > 0 (超买) 且 v_rub < 0 (开始掉头)，则是极强的释放信号
        snap_back_signal = (np.sign(rubber_raw) != np.sign(v_rub)).astype(np.float32) * v_rub.abs()
        high_freq_score = custom_tension_norm(rubber_raw, atr_pct, sensitivity=1.0)
        # 融合：位置张力 + 动态回弹信号
        high_freq_final = elastic_energy_gain(high_freq_score) * 0.7 + snap_back_signal * 0.3
        print(f"    -> [节点] 高频张力(橡皮筋+回弹)计算完成。均值: {high_freq_final.mean():.6f}")
        # --- 第一步：低频张力 (Bias Structural) ---
        # 核心：BIAS_55_D
        bias_raw = df['BIAS_55_D'].fillna(0)
        # 低频张力代表均值回归的“地心引力”，不需要太复杂的动力学，主要看偏离幅度
        low_freq_score = custom_tension_norm(bias_raw, atr_pct, sensitivity=0.6)
        low_freq_final = elastic_energy_gain(low_freq_score)
        print(f"    -> [节点] 低频张力(Bias55)计算完成。均值: {low_freq_final.mean():.6f}")
        # --- 第六步 & 第七步：物理张力融合 ---
        # 融合逻辑：多频段共振
        raw_tension = (high_freq_final * weights.get("rubber_band", 0.3) + 
                       mid_freq_final * weights.get("potential_index", 0.4) + 
                       low_freq_final * weights.get("bias_struct", 0.3))
        # 最终输出限制在 [0, 1] 区间 (张力通常视为标量强度，但在背离模型中需要保留符号以区分顶部/底部张力)
        # 此处保留符号：正值代表顶部张力(超买)，负值代表底部张力(超卖)
        final_tension_score = raw_tension.clip(-1, 1)
        print(f"  [探针-TensionHAB-V7.0] 市场张力计算完成。最终得分均值: {final_tension_score.mean():.6f}, 极值: [{final_tension_score.min():.6f}, {final_tension_score.max():.6f}]")
        return final_tension_score.astype(np.float32)

    def _calculate_spatio_temporal_consistency_score(self, df: pd.DataFrame, df_index: pd.Index, pmd_params: Dict, method_name: str) -> Tuple[pd.Series, Dict]:
        """
        【V7.0 · HAB时空一致性场论模型】
        基于七步推演法重构：融合多周期相位共振、能量场动力学与系统熵场，
        引入HAB记忆与非线性冲突惩罚，构建全息时空有序度评分。
        """
        print(f"  [探针-STConsistencyHAB-V7.0] 启动时空一致性全息校验...")
        # --- 参数提取 ---
        fib = pmd_params.get('fib_periods', [5, 13, 21, 34])
        weights = pmd_params.get('consistency_weights', {'phase_field': 0.4, 'energy_field': 0.3, 'entropy_field': 0.3})
        penalty_exp = pmd_params.get('conflict_penalty_exponent', 3.0)
        hab_window = pmd_params.get('consistency_hab_window', 13)
        epsilon = 1e-9
        # --- 第四步：定义专属归一化 ---
        def custom_consistency_norm(series: pd.Series, mode: str = 'bipolar') -> pd.Series:
            """
            - bipolar: 映射到 [-1, 1] (方向一致性)
            - unipolar: 映射到 [0, 1] (质量/熵)
            """
            roll_mean = series.rolling(window=34, min_periods=1).mean()
            roll_std = series.rolling(window=34, min_periods=1).std().replace(0, epsilon)
            z_score = (series - roll_mean) / roll_std
            if mode == 'bipolar':
                return np.tanh(z_score * 0.8)
            else:
                return 1 / (1 + np.exp(-z_score))
        # --- 第五步：非线性冲突惩罚 ---
        def calculate_conflict_penalty(polarity_matrix: pd.DataFrame) -> pd.Series:
            """
            计算多周期方向的冲突程度。
            如果所有周期方向一致(全1或全-1)，均值绝对值为1，惩罚为0。
            如果方向杂乱(均值为0)，惩罚最大。
            Penalty_Factor = Mean_Abs ^ Exponent
            """
            # 计算行均值的绝对值，代表一致性强度 [0, 1]
            consistency_strength = polarity_matrix.mean(axis=1).abs()
            # 再次进行非线性映射，强者恒强
            return consistency_strength.pow(penalty_exp)
        # --- 第二步 & 第三步：相位场 (Phase Field) ---
        # 计算多周期 (Price vs MACD) 的极性一致性
        # 引入 3D 动力学：不仅看 Slope 方向，更看 Accel 加速度方向
        phase_polarities = []
        for p in fib:
            # 获取价格与动量的速度和加速度
            v_p = df[f'SLOPE_{p}_close_D'].fillna(0)
            a_p = df[f'ACCEL_{p}_close_D'].fillna(0)
            v_m = df[f'SLOPE_{p}_MACDh_13_34_8_D'].fillna(0)
            # 动力学极性：速度占 70%，加速度占 30% (预判拐点)
            kin_p = np.sign(v_p) * 0.7 + np.sign(a_p) * 0.3
            kin_m = np.sign(v_m) * 0.7  # 动量本身就是二阶导，再求加速度噪音大，主要看斜率
            # 单周期一致性：两者同号则为正，异号为负
            # 使用 sign 乘积判断同向性
            period_coherence = np.sign(kin_p * kin_m).astype(np.float32)
            phase_polarities.append(period_coherence)
        # 汇总相位矩阵
        phase_matrix = pd.concat(phase_polarities, axis=1)
        # 计算相位一致性基础分 [-1, 1]
        raw_phase_score = phase_matrix.mean(axis=1)
        # 引入冲突惩罚因子 [0, 1]
        phase_penalty_factor = calculate_conflict_penalty(phase_matrix)
        # 最终相位场得分：基础分 * 惩罚因子 (方向乱时得分趋近0)
        phase_field = raw_phase_score * phase_penalty_factor
        print(f"    -> [节点] 相位场(多周期共振)计算完成。均值: {phase_field.mean():.6f}, 冲突因子均值: {phase_penalty_factor.mean():.6f}")
        # --- 第二步 & 第三步：能量场 (Energy Field) ---
        # 核心：价量时空动力学 (Price vs Volume Kinematics)
        # 好的趋势需要量能配合：价格加速(A>0)应伴随量能温和放大或加速
        v_vol, a_vol, _, _ = self._calculate_kinematic_tensors(df, df_index, 'volume_D', pmd_params)
        v_price, a_price, _, _ = self._calculate_kinematic_tensors(df, df_index, 'close_D', pmd_params)
        # 能量匹配逻辑：
        # 1. 价升量增 (v_p > 0, v_v > 0) -> 正向能量
        # 2. 缩量回调 (v_p < 0, v_v < 0) -> 良性清洗
        # 3. 这里的 Energy Field 更关注“有序性”，即价量矢量的夹角
        # 归一化后计算点积一致性
        norm_v_p = custom_consistency_norm(v_price, 'bipolar')
        norm_v_v = custom_consistency_norm(v_vol, 'bipolar')
        # 基础能量共振
        energy_resonance = norm_v_p * norm_v_v
        # HAB 记忆：能量场需要持续性
        hab_energy = energy_resonance.rolling(window=hab_window, min_periods=1).mean()
        energy_field = hab_energy # 直接使用 HAB 平滑后的能量场
        print(f"    -> [节点] 能量场(价量HAB)计算完成。均值: {energy_field.mean():.6f}")
        # --- 第一步：熵场 (Entropy Field) ---
        # 核心：chip_entropy_D (筹码熵) + PRICE_FRACTAL_DIM_D (分形维数)
        # 熵越低，系统越有序，时空一致性越强
        chip_ent = df['chip_entropy_D'].fillna(1.0)
        frac_dim = df['PRICE_FRACTAL_DIM_D'].fillna(1.5)
        # 归一化：反向映射，值越小分越高
        # 筹码熵通常在 [0, Inf)，分形维数在 [1, 2]
        # 对 chip_entropy 进行 sigmoid 反向
        ent_score = 1.0 - custom_consistency_norm(chip_ent, 'unipolar')
        # 对 fractal_dim，越接近 1.0 (线性趋势) 越好，接近 1.5 (随机) 越差
        dim_score = (1.5 - frac_dim).clip(0, 0.5) * 2.0 # 映射到 [0, 1]
        entropy_field = (ent_score * 0.6 + dim_score * 0.4)
        print(f"    -> [节点] 熵场(有序度)计算完成。均值: {entropy_field.mean():.6f}")
        # --- 第六步 & 第七步：多维场论融合 ---
        # 熵场主要作为权重系数：系统越有序，相位和能量信号越可信
        # 熵场 [0, 1]，映射为放大系数 [0.5, 1.5]
        entropy_amplifier = 0.5 + entropy_field
        # 融合：(相位场 * 权重 + 能量场 * 权重) * 熵放大系数
        raw_consistency = (phase_field * weights.get('phase_field', 0.4) + 
                           energy_field * weights.get('energy_field', 0.3)) * entropy_amplifier
        # 再次应用整体非线性增益，奖励高一致性状态
        def nonlinear_gain(x: pd.Series) -> pd.Series:
            return np.sign(x) * (x.abs() ** 1.2)
        final_st_consistency = nonlinear_gain(raw_consistency).clip(-1, 1)
        print(f"  [探针-STConsistencyHAB-V7.0] 时空一致性计算完成。最终得分均值: {final_st_consistency.mean():.6f}")
        debug_v = {
            "node_phase_field": phase_field,
            "node_phase_penalty": phase_penalty_factor,
            "node_energy_field": energy_field,
            "node_entropy_field": entropy_field,
            "final_st_consistency": final_st_consistency
        }
        return final_st_consistency.astype(np.float32), debug_v

    def _calculate_dqwm_matrix(self, df: pd.DataFrame, df_index: pd.Index, pmd_params: Dict, method_name: str) -> pd.Series:
        """
        【V7.0 · HAB-DQWM动态矩阵引擎】
        基于七步推演法重构：融合六维动力学分量，引入HAB存量记忆与动态场景权重，
        实施木桶短板惩罚与共振增益，构建自适应的市场全息质量矩阵。
        """
        print(f"  [探针-DQWM_HAB-V7.0] 启动 DQWM 矩阵全息聚合...")
        # --- 参数提取 ---
        base_weights = pmd_params.get('dqwm_weights', {
            "momentum_quality": 0.2, "market_tension": 0.15, "stability": 0.15, 
            "chip_potential": 0.15, "liquidity_tide": 0.2, "market_constitution": 0.15
        })
        hab_window = pmd_params.get('dqwm_hab_window', 21)
        epsilon = 1e-9
        # --- 第四步：定义专属矩阵归一化 ---
        def custom_matrix_norm(series: pd.Series) -> pd.Series:
            """将任意分布映射到 [0, 1] 的质量得分区间"""
            roll_min = series.rolling(window=55, min_periods=1).min()
            roll_max = series.rolling(window=55, min_periods=1).max()
            # MinMax 缩放，带安全边际
            norm = (series - roll_min) / (roll_max - roll_min + epsilon)
            return norm.clip(0, 1)
        # --- 第五步：短板惩罚与共振增益 ---
        def calculate_synergy_factor(components: Dict[str, pd.Series]) -> pd.Series:
            """
            计算系统的协同效应：
            1. 短板惩罚：如果存在极低分项(<0.2)，系数 < 1
            2. 共振奖励：如果所有项均及格(>0.5)，系数 > 1
            """
            df_comp = pd.DataFrame(components)
            min_score = df_comp.min(axis=1)
            mean_score = df_comp.mean(axis=1)
            # 短板惩罚：最弱项越弱，惩罚越重
            # min=0.1 -> penalty = 0.5; min=0.4 -> penalty=1.0
            penalty = np.where(min_score < 0.4, 0.5 + min_score * 1.25, 1.0).clip(max=1.0)
            # 共振奖励：均值高且无短板
            reward = np.where((min_score > 0.4) & (mean_score > 0.6), 1.0 + (mean_score - 0.6) * 0.5, 1.0)
            return pd.Series(penalty * reward, index=df_comp.index)
        # --- 第一步 & 原始逻辑：计算六大分量 ---
        # 1. 动量品质 (Momentum Quality)
        dq_mom = self._calculate_momentum_quality_score(df, df_index, {}, pmd_params, method_name)
        # 2. 市场张力 (Market Tension) - 注意：原方法返回的是[-1, 1]，需映射回 [0, 1] 代表"张力健康度"
        # 这里的逻辑是：张力适中(0)最好，极端张力(-1或1)代表风险。因此健康度 = 1 - abs(tension)
        raw_tension = self._calculate_market_tension_score(df, df_index, {}, pmd_params, method_name)
        dq_ten = 1.0 - raw_tension.abs() 
        # 3. 稳定性 (Stability)
        dq_stb = self._calculate_stability_score(df, df_index, {}, pmd_params, method_name)
        # 4. 筹码势能 (Chip Potential)
        dq_chip = self._calculate_chip_historical_potential_score(df, df_index, {}, pmd_params, method_name)
        # 5. 流动性潮汐 (Liquidity Tide)
        dq_tide = self._calculate_liquidity_tide_score(df, df_index, {}, pmd_params, method_name)
        # 6. 市场体质 (Market Constitution)
        dq_const = self._calculate_market_constitution_score(df, df_index, {}, pmd_params, method_name)
        # 归一化所有分量到 [0, 1]
        components = {
            "momentum_quality": custom_matrix_norm(dq_mom),
            "market_tension": custom_matrix_norm(dq_ten),
            "stability": custom_matrix_norm(dq_stb),
            "chip_potential": custom_matrix_norm(dq_chip),
            "liquidity_tide": custom_matrix_norm(dq_tide),
            "market_constitution": custom_matrix_norm(dq_const)
        }
        # --- 第三步：HAB 存量记忆修正 ---
        # 对每个分量应用 HAB 滤波，平滑噪音，提取真实系统状态
        hab_components = {}
        for k, v in components.items():
            hab_val = v.rolling(window=hab_window, min_periods=1).mean()
            # 存量支撑逻辑：当日得分仅占 30%，历史信用占 70%
            hab_components[k] = v * 0.3 + hab_val * 0.7
        # --- 第六步 & 第七步：动态自适应权重 ---
        # 引入场景因子
        trend_stage = df['state_trending_stage_D'].fillna(0) # 假设 1=主升, 0=震荡/下跌
        sentiment = df['market_sentiment_score_D'].fillna(0.5)
        # 动态权重生成
        # 进攻场景 (Attack): 动量 + 潮汐 权重增加
        attack_mask = (trend_stage > 0.5) & (sentiment > 0.6)
        # 防御场景 (Defense): 稳定性 + 体质 + 张力 权重增加
        defense_mask = ~attack_mask
        final_weights = {}
        for k, base_w in base_weights.items():
            # 基础权重向量
            w_series = pd.Series(base_w, index=df_index)
            if k in ["momentum_quality", "liquidity_tide"]:
                w_series[attack_mask] *= 1.5
                w_series[defense_mask] *= 0.8
            elif k in ["stability", "market_constitution", "market_tension"]:
                w_series[attack_mask] *= 0.8
                w_series[defense_mask] *= 1.2
            # chip_potential 保持中性
            final_weights[k] = w_series
        # --- 执行加权融合 ---
        # 1. 计算动态权重的总和，用于归一化
        total_weight = pd.Series(0.0, index=df_index)
        weighted_sum = pd.Series(0.0, index=df_index)
        for k, v in hab_components.items():
            w = final_weights[k]
            weighted_sum += v * w
            total_weight += w
        raw_dqwm = weighted_sum / (total_weight + epsilon)
        # --- 应用协同效应 (短板惩罚 & 共振增益) ---
        synergy_factor = calculate_synergy_factor(hab_components)
        final_dqwm = raw_dqwm * synergy_factor
        # --- 第二步：矩阵动力学修正 ---
        # 如果矩阵总分正在加速变好 (V>0, A>0)，给予额外奖励
        v_mat = final_dqwm.diff()
        a_mat = v_mat.diff()
        # 只有在总分及格(>0.5)的情况下，加速才有效
        momentum_bonus = np.where((final_dqwm > 0.5) & (v_mat > 0) & (a_mat > 0), 1.1, 1.0)
        final_score = (final_dqwm * momentum_bonus).clip(0, 1)
        print(f"  [探针-DQWM_HAB-V7.0] 矩阵计算完成。最终得分均值: {final_score.mean():.6f}, 协同因子均值: {synergy_factor.mean():.6f}")
        return final_score.astype(np.float32)


