# strategies\trend_following\intelligence\process\calculate_cost_advantage_trend_relationship.py
# 【V11.0.0 · 全息成本资金五维共振计算器】
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from strategies.trend_following.utils import get_params_block, get_param_value
from strategies.trend_following.intelligence.process.helper import ProcessIntelligenceHelper

class CalculateCostAdvantageTrendRelationship:
    """
    【V11.0.0 · 全息成本资金五维共振计算器】
    PROCESS_META_COST_ADVANTAGE_TREND
    - 核心职责: 基于五维超平面模型诊断趋势的爆发力与持续性。
    - 核心模型: 筹码势垒 + 捕食者攻击 + 趋势效率 + 成本弹性(New) + 结构熵逆(New)。
    - 升级重点: 引入时间微分视角，审判'价-本剪刀差'与'系统有序度'。
    - 数据源: 最终军械库清单
    """
    def __init__(self, strategy_instance, helper_instance: ProcessIntelligenceHelper):
        self.strategy = strategy_instance
        self.helper = helper_instance
        self.params = self.helper.params
        self.debug_params = self.helper.debug_params
        self.probe_dates = self.helper.probe_dates

    def _initialize_debug_context(self, method_name: str, df: pd.DataFrame) -> Tuple[bool, Optional[pd.Timestamp], Dict, Dict]:
        is_debug = get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug and self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            for date in reversed(df.index):
                if pd.to_datetime(date).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        debug_output = {}
        temp_vals = {}
        if is_debug and probe_ts:
            print(f"【V11.0探针激活】目标日期: {probe_ts.strftime('%Y-%m-%d')} | 方法: {method_name}")
            debug_output[f"--- {method_name} Probe @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
        return is_debug and (probe_ts is not None), probe_ts, debug_output, temp_vals

    def _probe_val(self, key: str, val: Any, temp_vals: Dict, section: str = "General"):
        if section not in temp_vals:
            temp_vals[section] = {}
        if isinstance(val, (pd.Series, np.ndarray, float, int)):
            try:
                v_str = f"{val:.4f}"
            except:
                v_str = str(val)
            temp_vals[section][key] = v_str
        else:
            temp_vals[section][key] = str(val)

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V11.0.0 · 五维共振计算主入口】"""
        method_name = "CalculateCostAdvantage_V11"
        print(f"【V11.0.0】启动五维成本资金共振计算，数据形状: {df.shape}")
        is_debug, probe_ts, debug_out, temp_vals = self._initialize_debug_context(method_name, df)
        df_processed = self._check_and_repair_signals(df.copy(), method_name)
        df_index = df_processed.index
        # 1. 维度一：筹码势垒稳固度 (Chip Barrier Solidity)
        chip_solidity = self._calculate_chip_barrier_solidity(df_processed, df_index, is_debug, probe_ts, temp_vals)
        # 2. 维度二：捕食者攻击向量 (Predator Attack Vector)
        predator_attack = self._calculate_predator_attack_vector(df_processed, df_index, is_debug, probe_ts, temp_vals)
        # 3. 维度三：趋势做功效率 (Kinematic Efficiency)
        trend_efficiency = self._calculate_kinematic_efficiency(df_processed, df_index, is_debug, probe_ts, temp_vals)
        # 4. 维度四：成本迁移弹性 (Cost Migration Elasticity) [NEW]
        cost_elasticity = self._calculate_cost_migration_elasticity(df_processed, df_index, is_debug, probe_ts, temp_vals)
        # 5. 维度五：结构熵逆 (Structure Negentropy) [NEW]
        structure_negentropy = self._calculate_structure_negentropy(df_processed, df_index, is_debug, probe_ts, temp_vals)
        # 6. 五维张量共振融合
        final_score = self._calculate_pentagonal_resonance(
            chip_solidity, predator_attack, trend_efficiency, 
            cost_elasticity, structure_negentropy,
            df_processed, df_index, is_debug, probe_ts, temp_vals
        )
        if is_debug:
            self._log_debug_values(debug_out, temp_vals, probe_ts, method_name)
        return final_score.astype(np.float32).fillna(0).clip(-1, 1)

    def _calculate_chip_barrier_solidity(self, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        """【V14.0 · 非线性临界相变筹码势垒】
        PROCESS_META_COST_ADVANTAGE_TREND::Dimension_1
        版本特性：
        1. Nonlinear Activation (非线性激活): 对结构、稳定、行为三大子维度应用 Sigmoid-like 激活，压制平庸，奖赏卓越。
        2. Synergy Factor (协同因子): 计算子维度的标准差，惩罚"偏科"形态（如结构好但行为差）。
        3. Phase Transition (临界相变): 当综合分突破阈值时，触发指数级增益，模拟主升浪的能量爆发。
        4. Inheritance (继承): 保留V13的物理跨度、微观致密、HAB存量与运动学逻辑。
        """
        # --- 1. 基础数据获取 (继承V13) ---
        winner_rate = self.helper._get_safe_series(df, 'winner_rate_D', 50.0, "winner_rate")
        stability = self.helper._get_safe_series(df, 'chip_stability_D', 1.0, "stability")
        concentration = self.helper._get_safe_series(df, 'chip_concentration_ratio_D', 0.2, "concentration")
        kurtosis = self.helper._get_safe_series(df, 'chip_kurtosis_D', 3.0, "kurtosis")
        lock_ratio = self.helper._get_safe_series(df, 'high_position_lock_ratio_90_D', 0.0, "lock_ratio")
        cost_5 = self.helper._get_safe_series(df, 'cost_5pct_D', 1.0, "cost_5")
        cost_50 = self.helper._get_safe_series(df, 'cost_50pct_D', 1.0, "cost_50")
        cost_95 = self.helper._get_safe_series(df, 'cost_95pct_D', 1.0, "cost_95")
        intra_consolidation = self.helper._get_safe_series(df, 'intraday_chip_consolidation_degree_D', 0.5, "intra_consolidation")

        # --- 2. 子维度计算 (Sub-Dimensions Calculation) ---
        
        # [A] 结构维度 (Structure Dimension)
        # 物理跨度
        span_ratio = (cost_95 - cost_5) / (cost_50 + 1e-8)
        score_span = (0.4 - span_ratio) * 3.33
        # 静态集中度
        score_concent = (0.25 - concentration) * 5.0
        # 峰度倍增
        morph_multiplier = 0.5 + np.tanh((kurtosis - 2.5) / 2.0)
        # 结构分合成
        raw_structure = (score_concent * 0.5 + score_span * 0.5) * morph_multiplier
        # **非线性激活**: 使用 tanh 将分数推向两极，使得 >0.5 的分数趋向 1，<0 的分数趋向 -1
        # 目的: 不想要"模棱两可"的结构，要么好要么坏
        dim_structure = np.tanh(raw_structure * 1.5)

        # [B] 稳定维度 (Stability Dimension)
        # HAB 存量记忆
        hab_stability_34 = stability.rolling(window=34, min_periods=1).mean()
        raw_stability = 0.3 * np.tanh(np.log1p(stability)) + 0.7 * np.tanh(np.log1p(hab_stability_34))
        # **非线性激活**
        dim_stability = np.tanh(raw_stability * 2.0)

        # [C] 行为维度 (Behavior Dimension)
        # 锁仓 + 微观致密
        score_micro = (intra_consolidation - 0.3) * 2.0
        score_lock = (lock_ratio * 2.0)
        raw_behavior = score_lock * 0.6 + score_micro * 0.4
        # **非线性激活**
        dim_behavior = np.tanh(raw_behavior * 1.5)

        # [D] 运动维度 (Kinematics Dimension)
        slope_conc = self.helper._get_safe_series(df, 'SLOPE_13_chip_concentration_ratio_D', 0.0, "slope_conc")
        accel_conc = self.helper._get_safe_series(df, 'ACCEL_13_chip_concentration_ratio_D', 0.0, "accel_conc")
        jerk_conc = self.helper._get_safe_series(df, 'JERK_13_chip_concentration_ratio_D', 0.0, "jerk_conc")
        norm_slope = np.tanh(slope_conc * -10.0)
        norm_accel = np.tanh(accel_conc * -5.0)
        norm_jerk_penalty = np.exp(-(jerk_conc * 5.0) ** 2)
        dim_kinematics = (norm_slope * 0.6 + norm_accel * 0.4) * norm_jerk_penalty

        # --- 3. 协同因子 (Synergy Factor) [V14核心] ---
        # 逻辑: 计算三大核心维度(结构、稳定、行为)的一致性。
        # 如果三个分数差异很大(标准差大)，说明主力做盘不坚决或有分歧，给予惩罚。
        # 堆叠维度用于计算std
        dims_matrix = pd.concat([dim_structure, dim_stability, dim_behavior], axis=1)
        # 计算行标准差 (ddof=0)
        synergy_std = dims_matrix.std(axis=1, ddof=0).fillna(0.5)
        # 协同系数: Std越小越好。0 -> 1.0, 0.5 -> 0.5
        synergy_factor = (1.0 - synergy_std).clip(0.1, 1.0)

        # --- 4. 基础加权合成 (Base Fusion) ---
        # 结构(30%) + 稳定(30%) + 行为(20%) + 运动(20%)
        # 此时已经经过了Tanh非线性激活
        base_score = (dim_structure * 0.3 + 
                      dim_stability * 0.3 + 
                      dim_behavior * 0.2 + 
                      dim_kinematics * 0.2)
        
        # 应用协同惩罚
        synergy_score = base_score * synergy_factor

        # --- 5. 临界相变增益 (Critical Phase Transition) [V14核心] ---
        # 逻辑: 当协同后的分数突破临界点(如0.5)时，给予指数级奖励。
        # 模拟主升浪时的"能量爆发"。
        # 增益函数: G(x) = x * (1 + alpha * exp(beta * (x - threshold)))
        # 仅对正分进行增益
        
        threshold = 0.5
        phase_transition_gain = np.where(
            synergy_score > threshold,
            # 如果 > 0.5: 原始分 * (1 + 0.5 * exp(2 * (x - 0.5)))
            # 例: x=0.6 -> 0.6 * (1 + 0.5 * 1.22) = 0.96 (显著放大)
            # 例: x=0.8 -> 0.8 * (1 + 0.5 * 1.82) = 1.52 (溢出，需截断)
            synergy_score * (1 + 0.5 * np.exp(3.0 * (synergy_score - threshold))),
            synergy_score # 低于阈值保持线性
        )
        
        # --- 6. 最终边界处理 ---
        # 熔断逻辑保留
        collapse_risk = (norm_jerk_penalty < 0.2) & (score_span < 0)
        final_solidity = np.where(collapse_risk, -0.8, phase_transition_gain).clip(-1, 1.2) # 允许轻微溢出以体现极强

        # --- 7. 探针输出 ---
        if is_debug and probe_ts:
            self._probe_val("Dim_Structure_Tanh", dim_structure.loc[probe_ts], temp_vals, "ChipSolidity_V14")
            self._probe_val("Dim_Stability_Tanh", dim_stability.loc[probe_ts], temp_vals, "ChipSolidity_V14")
            self._probe_val("Dim_Behavior_Tanh", dim_behavior.loc[probe_ts], temp_vals, "ChipSolidity_V14")
            self._probe_val("Synergy_Std", synergy_std.loc[probe_ts], temp_vals, "ChipSolidity_V14")
            self._probe_val("Synergy_Factor", synergy_factor.loc[probe_ts], temp_vals, "ChipSolidity_V14")
            self._probe_val("Base_Synergy_Score", synergy_score.loc[probe_ts], temp_vals, "ChipSolidity_V14")
            self._probe_val("Final_Phase_Trans_Score", final_solidity[idx.get_loc(probe_ts)] if probe_ts in idx else 0, temp_vals, "ChipSolidity_V14")
            
        return pd.Series(final_solidity, index=idx).clip(-1, 1)

    def _calculate_predator_attack_vector(self, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        """【V18.0 · 超临界相变捕食者矢量】
        PROCESS_META_COST_ADVANTAGE_TREND::Dimension_2
        版本特性：
        1. Triadic Synergy (三元协同): 将攻击分解为 Power(力量/HAB), Guile(战术/隐蔽), Quality(效率/一致性) 三大维度，计算协同离散度。
        2. Nonlinear Activation (非线性激活): 引入指数增益函数，对突破临界阈值的高协同攻击进行"暴击"放大，模拟主升浪起爆。
        3. Noise Suppression (噪音抑制): 对低协同或低强度的信号进行非线性压缩，过滤无效的试盘行为。
        4. Inheritance (全继承): 完整保留 V16/V17 的 HAB存量、高阶导数去噪、效率加权与结构突变逻辑。
        """
        # --- 1. 基础数据提取 (继承 V17) ---
        sm_net_buy = self.helper._get_safe_series(df, 'SMART_MONEY_HM_NET_BUY_D', 0.0, "sm_net_buy")
        inst_buy = self.helper._get_safe_series(df, 'SMART_MONEY_INST_NET_BUY_D', 0.0, "inst_buy")
        coord_attack = self.helper._get_safe_series(df, 'SMART_MONEY_HM_COORDINATED_ATTACK_D', 0.0, "coord_attack")
        # 运动学导数 (零基防御准备)
        sm_accel = self.helper._get_safe_series(df, 'ACCEL_13_SMART_MONEY_HM_NET_BUY_D', 0.0, "sm_accel")
        sm_jerk = self.helper._get_safe_series(df, 'JERK_13_SMART_MONEY_HM_NET_BUY_D', 0.0, "sm_jerk")
        # 战术意图
        stealth_flow = self.helper._get_safe_series(df, 'stealth_flow_ratio_D', 0.0, "stealth_flow")
        morning_ratio = self.helper._get_safe_series(df, 'morning_flow_ratio_D', 0.0, "morning_ratio")
        closing_intensity = self.helper._get_safe_series(df, 'closing_flow_intensity_D', 0.0, "closing_intensity")
        # 质量与效率
        flow_impact = self.helper._get_safe_series(df, 'flow_impact_ratio_D', 1.0, "flow_impact")
        anomaly_val = self.helper._get_safe_series(df, 'large_order_anomaly_D', 0.0, "anomaly_val")
        cluster_intensity = self.helper._get_safe_series(df, 'flow_cluster_intensity_D', 0.0, "cluster_intensity")
        # 博弈环境
        consistency = self.helper._get_safe_series(df, 'flow_consistency_D', 0.5, "consistency")
        slope_consistency = self.helper._get_safe_series(df, 'SLOPE_13_flow_consistency_D', 0.0, "slope_consistency")
        activity = self.helper._get_safe_series(df, 'main_force_activity_index_D', 0.0, "activity")

        # --- 2. 维度一：力量矢量 (Vec_Power) [HAB + Impulse] ---
        # HAB 存量势能
        sm_hab_21 = sm_net_buy.rolling(window=21, min_periods=1).sum()
        hab_capacity = sm_hab_21.abs().rolling(55).max().replace(0, 1.0)
        norm_hab = (sm_hab_21 / hab_capacity).clip(-1, 1)
        # 脉冲动能 (含导数去噪)
        roll_max_sm = sm_net_buy.abs().rolling(21).max().replace(0, 1.0)
        norm_vel = (sm_net_buy / roll_max_sm).clip(-1, 1)
        accel_scale = sm_accel.rolling(21).std().replace(0, 1.0)
        jerk_scale = sm_jerk.rolling(21).std().replace(0, 1.0)
        norm_accel = np.tanh(sm_accel / (accel_scale + 1e-8))
        norm_jerk = np.tanh(sm_jerk / (jerk_scale + 1e-8))
        instant_impulse = norm_vel * 0.5 + norm_accel * 0.3 + norm_jerk * 0.2
        # Power 合成
        vec_power = (norm_hab * 0.6 + instant_impulse * 0.4).clip(-1, 1)

        # --- 3. 维度二：诡道矢量 (Vec_Guile) [Tactical + Stealth] ---
        norm_stealth = np.tanh(stealth_flow * 2.0)
        score_morning = ((morning_ratio - 0.2) * 2.5).clip(-0.5, 1)
        score_closing = np.tanh(closing_intensity)
        # Guile 合成
        vec_guile = (norm_stealth * 0.5 + score_morning * 0.3 + score_closing * 0.2).clip(-1, 1)

        # --- 4. 维度三：质量矢量 (Vec_Quality) [Efficiency + Consistency + Structure] ---
        # 效率修正
        eff_score = np.tanh(flow_impact - 1.0) # >1为正，<1为负
        # 结构突变
        struct_score = np.tanh(anomaly_val / 2.0) * 0.6 + np.tanh(cluster_intensity) * 0.4
        # 高阶一致性
        gate_consistency = ((consistency - 0.5) * 2.0).clip(0, 1)
        slope_scale = slope_consistency.rolling(21).std().replace(0, 0.1)
        norm_slope_consist = np.tanh(slope_consistency / (slope_scale + 1e-8))
        consist_score = (gate_consistency * 0.7 + norm_slope_consist * 0.3)
        # Quality 合成
        vec_quality = (eff_score * 0.3 + struct_score * 0.3 + consist_score * 0.4).clip(-1, 1)

        # --- 5. 三元协同 (Triadic Synergy) [V18核心] ---
        # 逻辑：Power, Guile, Quality 三者方向一致且数值接近时，协同性最高。
        # 堆叠三个矢量
        vectors = np.column_stack([vec_power, vec_guile, vec_quality]) # Shape: (N, 3)
        # 计算行标准差 (Synergy Penalty)
        # std 越小，协同越好。
        synergy_std = np.std(vectors, axis=1)
        # 协同系数: 0.1(差) -> 1.0(好)。Std=0.5 -> Factor=0.5; Std=0 -> Factor=1.0
        synergy_factor = (1.0 - synergy_std * 1.5).clip(0.1, 1.0)
        
        # 基础合成 (加权平均)
        base_score = (vec_power * 0.4 + vec_guile * 0.3 + vec_quality * 0.3)
        # 应用协同系数
        synergized_score = base_score * synergy_factor

        # --- 6. 非线性相变激活 (Nonlinear Phase Transition) [V18核心] ---
        # 逻辑: 当协同分 > 0.5 时，触发指数奖励。
        # 鲨鱼突袭函数: G(x) = x * (1 + alpha * exp(beta * (x - thresh)))
        threshold = 0.5
        # 仅对正向攻击进行非线性放大
        final_attack = np.where(
            synergized_score > threshold,
            # 指数增益区域: alpha=0.6, beta=2.5
            synergized_score * (1.0 + 0.6 * np.exp(2.5 * (synergized_score - threshold))),
            # 线性或抑制区域: 如果是负分，保持原样；如果是低正分，保留线性
            synergized_score
        )
        
        # 暴击修正 (Coordinated Attack) - 直接突破物理极限
        crit_multiplier = np.where(coord_attack > 0.5, 1.3, 1.0)
        final_attack = (final_attack * crit_multiplier).clip(-1, 2.0) # 允许高达2.0的超强信号

        # --- 7. 探针输出 ---
        if is_debug and probe_ts:
            self._probe_val("Vec_Power", vec_power.loc[probe_ts], temp_vals, "PredatorVector_V18")
            self._probe_val("Vec_Guile", vec_guile.loc[probe_ts], temp_vals, "PredatorVector_V18")
            self._probe_val("Vec_Quality", vec_quality.loc[probe_ts], temp_vals, "PredatorVector_V18")
            self._probe_val("Synergy_Std", synergy_std[idx.get_loc(probe_ts)] if probe_ts in idx else 0, temp_vals, "PredatorVector_V18")
            self._probe_val("Synergy_Factor", synergy_factor[idx.get_loc(probe_ts)] if probe_ts in idx else 0, temp_vals, "PredatorVector_V18")
            self._probe_val("Base_Score", base_score.loc[probe_ts], temp_vals, "PredatorVector_V18")
            self._probe_val("Synergized_Score", synergized_score.loc[probe_ts], temp_vals, "PredatorVector_V18")
            self._probe_val("Final_Attack", final_attack[idx.get_loc(probe_ts)] if probe_ts in idx else 0, temp_vals, "PredatorVector_V18")

        return pd.Series(final_attack, index=idx)

    def _calculate_kinematic_efficiency(self, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        """【V22.0 · 超导相变与非线性熵减】
        PROCESS_META_COST_ADVANTAGE_TREND::Dimension_3
        版本特性：
        1. Macro-Micro Synergy (宏微协同): 强制要求宏观资金冲击与微观Tick传输方向一致。差异过大视为"虚假做功"，实施惩罚。
        2. Superconducting Activation (超导激活): 引入非线性指数增益函数。当系统效率突破临界阈值时，模拟"零阻力"状态，放大信号权值。
        3. Entropy Suppression (熵减抑制): 在高熵(混乱)状态下，强制关闭非线性增益，防止在震荡市中信号漂移。
        4. Inheritance (全继承): 保留 V21 的主力修正VPA、HAB飞轮、高阶导数及摩擦修正逻辑。
        """
        # --- 1. 基础运动学数据 (Kinematics Base) ---
        velocity = self.helper._get_safe_series(df, 'MA_VELOCITY_EMA_55_D', 0.0, "velocity")
        accel = self.helper._get_safe_series(df, 'MA_ACCELERATION_EMA_55_D', 0.0, "accel")
        vel_norm = np.tanh(velocity / velocity.rolling(89).std().replace(0, 1.0))
        accel_norm = np.tanh(accel / accel.rolling(89).std().replace(0, 1.0))

        # --- 2. 宏观热力学效率 (Macro-Thermodynamics) ---
        # [A] 主力修正VPA [Source 1]
        mf_vpa = self.helper._get_safe_series(df, 'VPA_MF_ADJUSTED_EFF_D', 0.0, "mf_vpa")
        mf_vpa_norm = mf_vpa / (mf_vpa.abs().rolling(21).max().replace(0, 1.0))
        # 资金冲击率 [Source 2]
        flow_impact = self.helper._get_safe_series(df, 'flow_impact_ratio_D', 1.0, "flow_impact")
        impact_norm = np.tanh(flow_impact - 1.0)
        # 宏观合成
        macro_work = (mf_vpa_norm * 0.6 + impact_norm * 0.4).clip(-1, 1)

        # --- 3. 微观结构效率 (Micro-Structural Efficiency) ---
        # [B] Tick级传输效率 [Source 3]
        tick_eff = self.helper._get_safe_series(df, 'tick_chip_transfer_efficiency_D', 0.5, "tick_eff")
        # 0.5 -> 0, 0.8 -> 1.0
        micro_work = ((tick_eff - 0.5) * 3.33).clip(-1, 1)

        # [C] 价格结构熵 (Price Entropy) [Source 1]
        entropy = self.helper._get_safe_series(df, 'price_entropy_D', 0.5, "entropy")
        # 熵越低越好。0.2 -> 1.0, 0.8 -> -0.5
        negentropy = (0.6 - entropy) * 2.5
        negentropy = negentropy.clip(-1, 1)

        # --- 4. 宏微协同与熵减合成 (Synergy & Negentropy) [V22核心] ---
        
        # [D] 宏微协同惩罚 (Synergy Penalty)
        # 逻辑: 宏观强但微观弱(差值大)，说明是"空心萝卜"。
        # 计算欧氏距离或绝对差。这里用 1 - 绝对差/2
        synergy_gap = np.abs(macro_work - micro_work)
        synergy_factor = (1.0 - synergy_gap * 0.8).clip(0.2, 1.0) # 差异越大，系数越小
        
        # [E] 基础瞬时效率
        # (宏观 + 微观) * 协同 * 熵修正
        # 熵本身作为方向修正: 低熵(正)增强，高熵(负)衰减
        base_efficiency = ((macro_work + micro_work) / 2.0) * synergy_factor
        # 只有在低熵状态下，效率才被认为是有效的
        # 如果 negentropy < 0 (高熵), 效率将被严重抑制
        entropy_gate = (negentropy + 1.0) / 2.0 # 映射到 0~1
        instant_efficiency = base_efficiency * entropy_gate
        
        # [F] 摩擦修正 (Friction) [继承]
        resistance = self.helper._get_safe_series(df, 'resistance_strength_D', 0.0, "resistance")
        resist_norm = np.tanh(resistance / 5.0)
        friction_factor = np.where(
            (instant_efficiency > 0.3) & (resist_norm > 0.5), 1.4, # 强效率+高阻力=超导突围
            np.where((instant_efficiency < 0) & (resist_norm > 0.5), 0.6, 1.0)
        )
        instant_efficiency = (instant_efficiency * friction_factor).clip(-1, 1)

        # --- 5. 非线性超导激活 (Nonlinear Superconducting Activation) [V22核心] ---
        # 逻辑: 当效率突破阈值(0.5)且处于低熵有序状态时，触发指数增益
        threshold = 0.5
        super_conductive_gain = np.where(
            (instant_efficiency > threshold),
            # G(x) = x * (1 + 0.8 * exp(2.0 * (x - 0.5)))
            # Ex: x=0.8 -> 0.8 * (1 + 0.8 * 1.82) = 1.96 (翻倍)
            instant_efficiency * (1.0 + 0.8 * np.exp(2.0 * (instant_efficiency - threshold))),
            instant_efficiency # 线性区域
        )
        # 限制最大溢出
        super_conductive_gain = super_conductive_gain.clip(-1, 2.0)

        # --- 6. HAB 效率飞轮 (HAB Flywheel) [继承V20] ---
        # 使用超导激活后的分数进入HAB，让"高光时刻"被记忆
        hab_efficiency_34 = super_conductive_gain.rolling(window=34, min_periods=1).mean()
        score_hab = np.tanh(hab_efficiency_34) # 这里的hab已经是放大过的，tanh将其拉回 [-1, 1] 区间作为基准

        # --- 7. 高阶热力学导数 (High-Order Derivatives) [继承V20] ---
        eff_slope = super_conductive_gain.diff(13).fillna(0)
        eff_accel = eff_slope.diff(5).fillna(0)
        eff_jerk = eff_accel.diff(3).fillna(0)
        # 去噪
        slope_scale = eff_slope.rolling(21).std().replace(0, 0.01)
        accel_scale = eff_accel.rolling(21).std().replace(0, 0.01)
        jerk_scale = eff_jerk.rolling(21).std().replace(0, 0.01)
        norm_slope = np.tanh(eff_slope / (slope_scale + 1e-8))
        norm_accel = np.tanh(eff_accel / (accel_scale + 1e-8))
        norm_jerk = np.tanh(eff_jerk / (jerk_scale + 1e-8))
        jerk_penalty = np.exp(-(norm_jerk * 2.0) ** 2)
        derivative_score = (norm_slope * 0.6 + norm_accel * 0.4) * jerk_penalty

        # --- 8. 最终全息热力学合成 ---
        # 飞轮(40%) + 超导瞬时(30%) + 导数(30%)
        # 注意: instant_efficiency 使用的是 super_conductive_gain (可能>1)
        # 最终 score 会被 clip
        total_thermodynamics = (score_hab * 0.4 + 
                                super_conductive_gain * 0.3 + 
                                derivative_score * 0.3)

        # 动能-热能共振
        alignment = np.sign(vel_norm) * np.sign(total_thermodynamics)
        final_efficiency = np.where(
            alignment > 0,
            vel_norm * (0.6 + 0.4 * np.abs(total_thermodynamics)), # 保持一定的线性
            vel_norm * (1.0 - np.abs(total_thermodynamics)) # 背离惩罚
        ).clip(-1, 1.5) # 允许溢出以体现超导状态

        # --- 9. 探针输出 ---
        if is_debug and probe_ts:
            self._probe_val("Macro_Work", macro_work.loc[probe_ts], temp_vals, "KinematicEff_V22")
            self._probe_val("Micro_Work", micro_work.loc[probe_ts], temp_vals, "KinematicEff_V22")
            self._probe_val("Synergy_Gap", synergy_gap.loc[probe_ts], temp_vals, "KinematicEff_V22")
            self._probe_val("Negentropy", negentropy.loc[probe_ts], temp_vals, "KinematicEff_V22")
            self._probe_val("Base_Inst_Eff", instant_efficiency.loc[probe_ts], temp_vals, "KinematicEff_V22")
            self._probe_val("Super_Gain_Eff", super_conductive_gain.loc[probe_ts], temp_vals, "KinematicEff_V22")
            self._probe_val("HAB_Flywheel", score_hab.loc[probe_ts], temp_vals, "KinematicEff_V22")
            self._probe_val("Deriv_Score", derivative_score.loc[probe_ts], temp_vals, "KinematicEff_V22")
            self._probe_val("Total_Thermo", total_thermodynamics.loc[probe_ts], temp_vals, "KinematicEff_V22")
            self._probe_val("Final_Efficiency", final_efficiency[idx.get_loc(probe_ts)] if probe_ts in idx else 0, temp_vals, "KinematicEff_V22")

        return pd.Series(final_efficiency, index=idx)

    def _calculate_cost_migration_elasticity(self, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        """【V26.0 · 超弹性共振与反脆弱增益】
        PROCESS_META_COST_ADVANTAGE_TREND::Dimension_4
        版本特性：
        1. Dimensional Synergy (维度协同): 计算 HAB(历史)、Modulus(当下)、Fractal(微观) 的协同离散度。高度协同触发共振奖励。
        2. Antifragile Activation (反脆弱激活): 引入非线性指数增益。当弹性突破临界阈值时，模拟"越压越强"的爆发力。
        3. Stress-Strain Modulus (应力模量): 继承 V25，量化高压锁仓能力。
        4. Intraday Fractal (日内分形): 继承 V25，微观验证。
        5. Zero-Base Defense (零基防御): 全程 Tanh 归一化，确保非线性函数的输入稳定性。
        """
        # --- 1. 基础数据获取 ---
        close = self.helper._get_safe_series(df, 'close_D', 1.0, "close")
        cost_50 = self.helper._get_safe_series(df, 'cost_50pct_D', 1.0, "cost_50")
        cost_5 = self.helper._get_safe_series(df, 'cost_5pct_D', 1.0, "cost_5")
        cost_95 = self.helper._get_safe_series(df, 'cost_95pct_D', 1.0, "cost_95")
        turnover = self.helper._get_safe_series(df, 'turnover_rate_f_D', 1.0, "turnover")
        
        profit_pressure = self.helper._get_safe_series(df, 'profit_pressure_D', 0.0, "profit_pressure")
        trapped_pressure = self.helper._get_safe_series(df, 'pressure_trapped_D', 0.0, "trapped_pressure")
        intra_migration = self.helper._get_safe_series(df, 'intraday_cost_center_migration_D', 0.0, "intra_migration")
        intra_volatility = self.helper._get_safe_series(df, 'intraday_cost_center_volatility_D', 0.0, "intra_volatility")

        # --- 2. 高阶运动学导数 (去噪处理) ---
        slope_c50 = cost_50.diff(13).fillna(0)
        slope_c5 = cost_5.diff(13).fillna(0)
        slope_price = close.diff(13).fillna(0)
        accel_c5 = slope_c5.diff(5).fillna(0)
        jerk_c5 = accel_c5.diff(3).fillna(0)

        # 波动率缩放 (Volatility Scaling)
        scale_c50 = slope_c50.rolling(21).std().replace(0, 0.01)
        scale_c5 = slope_c5.rolling(21).std().replace(0, 0.01)
        scale_price = slope_price.rolling(21).std().replace(0, 1.0)
        scale_accel = accel_c5.rolling(21).std().replace(0, 0.005)
        scale_jerk = jerk_c5.rolling(21).std().replace(0, 0.001)

        norm_slope_c50 = np.tanh(slope_c50 / (scale_c50 + 1e-8))
        norm_slope_c5 = np.tanh(slope_c5 / (scale_c5 + 1e-8))
        norm_slope_price = np.tanh(slope_price / (scale_price + 1e-8))
        norm_accel_c5 = np.tanh(accel_c5 / (scale_accel + 1e-8))
        norm_jerk_c5 = np.tanh(jerk_c5 / (scale_jerk + 1e-8))

        # --- 3. 子维度计算 (Sub-Dimensions) ---

        # [A] HAB 粘弹性 (Viscoelasticity)
        # 换手阻尼 (Golden Zone: 3-8%)
        turnover_score = 0.5 + 0.7 * np.exp(-((turnover - 5.5)**2) / 50.0)
        # 剪刀差: Price Up, Cost Slow
        scissor_inst = norm_slope_price - (norm_slope_c50 * 0.8)
        viscous_inst = scissor_inst * turnover_score
        # HAB 记忆
        viscous_hab = viscous_inst.rolling(window=34, min_periods=1).mean()
        # 归一化到 [-1, 1]
        dim_hab = np.tanh(viscous_hab * 2.0)

        # [B] 应力-应变模量 (Modulus)
        # Stress: 获利抛压
        norm_stress = np.tanh(profit_pressure / 10.0)
        # Strain: 成本移动 (绝对值)
        norm_strain = np.abs(norm_slope_c50)
        # Modulus: High Stress + Low Strain = Super Elasticity
        # 基础模量
        raw_modulus = norm_stress * (1.0 - norm_strain) * 2.5
        # 只有在有压力(Stress>0.2)时才激活模量逻辑，否则给中性分
        dim_modulus = np.where(norm_stress > 0.2, np.tanh(raw_modulus), 0.3)

        # [C] 日内分形 (Fractal)
        # 迁移向上且波动小
        norm_intra_mig = np.tanh(intra_migration * 5.0)
        norm_intra_vol = np.tanh(intra_volatility * 5.0)
        # Fractal Score
        dim_fractal = (norm_intra_mig * 0.6 + (1.0 - norm_intra_vol) * 0.4).clip(-1, 1)

        # --- 4. 协同共振 (Synergy Resonance) [V26核心] ---
        # 逻辑: HAB、Modulus、Fractal 三者必须共振。
        # 堆叠向量
        dims = np.column_stack([dim_hab, dim_modulus, dim_fractal])
        # 计算离散度 (Std)
        synergy_std = np.std(dims, axis=1)
        # 协同系数: 离散度越小，协同越高。 0.2 -> 1.0, 0.6 -> 0.5
        synergy_factor = (1.0 - synergy_std * 1.2).clip(0.3, 1.0)
        
        # 基础合成 (线性)
        # HAB(40%) + Modulus(40%) + Fractal(20%)
        base_score = (dim_hab * 0.4 + dim_modulus * 0.4 + dim_fractal * 0.2)
        # 应用协同
        synergized_score = base_score * synergy_factor

        # --- 5. 反脆弱非线性激活 (Antifragile Activation) [V26核心] ---
        # 逻辑: 当弹性得分突破阈值(0.6)时，触发指数增益。
        # 模拟"压强越大，反弹越猛"的物理特性。
        threshold = 0.6
        final_elasticity = np.where(
            synergized_score > threshold,
            # G(x) = x * (1 + 0.8 * exp(2.5 * (x - 0.6)))
            # Ex: x=0.8 -> 0.8 * (1 + 0.8 * 1.64) = 1.85 (显著放大)
            synergized_score * (1.0 + 0.8 * np.exp(2.5 * (synergized_score - threshold))),
            # 低于阈值保持线性
            synergized_score
        )

        # --- 6. 物理约束修正 (Physical Constraints) ---
        # [A] 锚定破坏 (Anchorage Violation)
        # 底仓 Cost5 加速上移是极危信号
        violation = (np.maximum(0, norm_slope_c5) * 0.2 + 
                     np.maximum(0, norm_accel_c5) * 0.3 + 
                     np.maximum(0, norm_jerk_c5) * 0.5)
        anchorage_penalty = np.exp(-violation * 4.0) # 敏感度调高

        # [B] 套牢天花板 (Ceiling)
        ceiling_penalty = np.exp(-trapped_pressure * 2.5)
        
        # [C] 结构压缩 (Compression) - 额外加分
        span = (cost_95 - cost_5) / (cost_50 + 1e-8)
        slope_span = span.diff(13).fillna(0)
        scale_span = slope_span.rolling(21).std().replace(0, 0.01)
        norm_slope_span = np.tanh(slope_span / (scale_span + 1e-8))
        compression_bonus = np.maximum(0, -norm_slope_span) * 0.2 # 仅奖励收缩

        # 最终应用约束
        final_score = (final_elasticity * anchorage_penalty * ceiling_penalty + compression_bonus).clip(-1, 2.0)

        # --- 7. 探针输出 ---
        if is_debug and probe_ts:
            self._probe_val("Dim_HAB", dim_hab.loc[probe_ts], temp_vals, "CostElasticity_V26")
            self._probe_val("Dim_Modulus", dim_modulus.loc[probe_ts], temp_vals, "CostElasticity_V26")
            self._probe_val("Dim_Fractal", dim_fractal.loc[probe_ts], temp_vals, "CostElasticity_V26")
            self._probe_val("Synergy_Std", synergy_std[idx.get_loc(probe_ts)] if probe_ts in idx else 0, temp_vals, "CostElasticity_V26")
            self._probe_val("Base_Score", base_score.loc[probe_ts], temp_vals, "CostElasticity_V26")
            self._probe_val("Synergized", synergized_score.loc[probe_ts], temp_vals, "CostElasticity_V26")
            self._probe_val("Anchor_Penalty", anchorage_penalty.loc[probe_ts], temp_vals, "CostElasticity_V26")
            self._probe_val("Final_Elasticity", final_score[idx.get_loc(probe_ts)] if probe_ts in idx else 0, temp_vals, "CostElasticity_V26")

        return pd.Series(final_score, index=idx)

    def _calculate_structure_negentropy(self, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        """【V30.0 · 超序参量与乘法锁定熵】
        PROCESS_META_COST_ADVANTAGE_TREND::Dimension_5
        版本特性：
        1. Multiplicative Coupling (乘法耦合): 摒弃线性加权，使用几何平均构建"超序参量"(Hyper-Order Parameter)。强制微观与宏观必须同时优秀，彻底杜绝"表里不一"的诱多陷阱。
        2. Structural Lock-in (结构锁定): 引入非线性锁定增益。当微观与宏观共振达到临界值时，模拟结构的"结晶化"过程，给予爆发性奖励。
        3. Kinetics Modulation (动力学调制): 将熵减动力学作为方向修正系数，而非简单的加分项。
        4. Inheritance (全继承): 保留 V29 的 HAB、分形、拓扑曲率与死寂防御逻辑。
        """
        # --- 1. 基础数据获取 ---
        # [A] 内部微观熵 (Internal)
        chip_entropy = self.helper._get_safe_series(df, 'chip_entropy_D', 0.5, "chip_entropy")
        conc_entropy = self.helper._get_safe_series(df, 'concentration_entropy_D', 0.5, "conc_entropy")
        intra_entropy = self.helper._get_safe_series(df, 'intraday_chip_entropy_D', 0.5, "intra_entropy")
        
        # [B] 外部宏观几何 (External)
        price_entropy = self.helper._get_safe_series(df, 'price_entropy_D', 0.5, "price_entropy")
        reg_r2 = self.helper._get_safe_series(df, 'GEOM_REG_R2_D', 0.0, "reg_r2")
        reg_slope = self.helper._get_safe_series(df, 'GEOM_REG_SLOPE_D', 0.0, "reg_slope")
        
        # [C] 拓扑与分形
        curvature = self.helper._get_safe_series(df, 'GEOM_ARC_CURVATURE_D', 0.0, "curvature")
        fractal_dim = self.helper._get_safe_series(df, 'price_fractal_dim_D', 1.5, "fractal_dim")
        flow_stab = self.helper._get_safe_series(df, 'turnover_stability_index_D', 0.5, "flow_stab")

        # --- 2. 子系统状态计算 ---
        
        # [Internal System] 微观有序度
        total_micro_entropy = chip_entropy * 0.4 + conc_entropy * 0.3 + intra_entropy * 0.3
        hab_entropy_34 = total_micro_entropy.rolling(window=34, min_periods=1).mean()
        # 归一化: 0.2(有序) -> 1.0, 0.8(乱) -> 0.0
        # 使用 Sigmoid 增强两极分化
        score_internal = 1.0 / (1.0 + np.exp(12.0 * (hab_entropy_34 - 0.55))) # Center=0.55
        
        # [External System] 宏观几何序
        # 价格熵逆
        negent_price = 1.0 / (1.0 + np.exp(12.0 * (price_entropy - 0.55)))
        # R2 线性度 (R2 < 0.6 无意义, >0.9 极强)
        score_r2 = ((reg_r2 - 0.6) * 2.5).clip(0, 1)
        # 宏观序 = 价格熵逆 * R2 (乘法耦合: 必须既低熵又直线)
        score_external = np.sqrt(negent_price * score_r2)

        # [Topology System] 拓扑光滑度
        curv_volatility = curvature.rolling(13).std().replace(0, 0.01)
        score_smoothness = np.tanh((0.1 - curv_volatility) * 10.0)

        # --- 3. 超序参量 (Hyper-Order Parameter) [V30核心] ---
        # 逻辑: 只有 内(筹码) 外(价格) 兼修，才是真有序。
        # 乘法耦合: 任何一方有短板，整体迅速衰减。
        # Order = sqrt(Internal * External)
        order_parameter = np.sqrt(score_internal * score_external)

        # --- 4. 结构锁定增益 (Structural Lock-in) [V30核心] ---
        # 逻辑: 当 Order > 0.7 (临界点) 时，结构趋于"晶体化" (Lock-in)。
        # 此时给予非线性奖励。
        # Lock Gain = 1 + alpha * (Order)^4 (高阶幂次奖励高分)
        lock_gain = 1.0 + 1.0 * np.power(order_parameter, 4)
        
        # 初步晶体分
        crystal_score = order_parameter * lock_gain

        # --- 5. 熵减动力学调制 (Kinetics Modulation) ---
        # 逻辑: 动力学不再是加分项，而是"变异系数"。
        # 如果动力学显示正在熵增(变乱)，则对晶体分进行"消融" (Melting)。
        
        slope_ent = total_micro_entropy.diff(13).fillna(0)
        scale_slope = slope_ent.rolling(21).std().replace(0, 0.01)
        norm_slope_ent = np.tanh(-slope_ent / (scale_slope + 1e-8)) # >0 代表正在变有序
        
        # 调制系数: 
        # 正在变好(>0) -> 保持或微增 (1.0 ~ 1.2)
        # 正在变坏(<0) -> 严重惩罚 (0.5 ~ 1.0)
        kinetics_modulator = np.where(
            norm_slope_ent > 0,
            1.0 + norm_slope_ent * 0.2,
            1.0 + norm_slope_ent * 0.5 # 负数时惩罚更重
        )
        
        # --- 6. 辅助修正 (Auxiliary Correction) ---
        # 分形修正: 必须是非随机游走 (Fractal < 1.4)
        fractal_gate = np.where(fractal_dim < 1.4, 1.0, 0.5)
        # 层流修正: 换手必须稳定
        laminar_gate = np.where(flow_stab > 0.6, 1.0, 0.7)
        # 光滑度修正
        smooth_gate = (score_smoothness + 1.0) / 2.0 # 0~1
        
        # --- 7. 最终合成 ---
        # 晶体分 * 动力学 * 门控
        synergized_negentropy = crystal_score * kinetics_modulator * fractal_gate * laminar_gate * (0.8 + 0.2 * smooth_gate)
        
        # --- 8. 麦克斯韦趋势门控 ---
        # 必须是上涨趋势
        trend_gate = np.tanh(reg_slope * 10.0)
        
        final_negentropy = synergized_negentropy * trend_gate
        
        # 二次非线性激发 (模拟极值爆发)
        threshold = 0.6
        final_score = np.where(
            final_negentropy > threshold,
            final_negentropy * (1.0 + 0.5 * np.exp(2.0 * (final_negentropy - threshold))),
            final_negentropy
        ).clip(-1, 2.0)
        
        # 死寂防御
        turnover = self.helper._get_safe_series(df, 'turnover_rate_f_D', 1.0, "turnover")
        final_score = np.where(turnover < 0.5, 0.0, final_score)

        # --- 9. 探针输出 ---
        if is_debug and probe_ts:
            self._probe_val("Score_Internal", score_internal.loc[probe_ts], temp_vals, "Negentropy_V30")
            self._probe_val("Score_External", score_external.loc[probe_ts], temp_vals, "Negentropy_V30")
            self._probe_val("Order_Param", order_parameter.loc[probe_ts], temp_vals, "Negentropy_V30")
            self._probe_val("Lock_Gain", lock_gain.loc[probe_ts], temp_vals, "Negentropy_V30")
            self._probe_val("Crystal_Score", crystal_score.loc[probe_ts], temp_vals, "Negentropy_V30")
            self._probe_val("Kinetics_Mod", kinetics_modulator.loc[probe_ts], temp_vals, "Negentropy_V30")
            self._probe_val("Final_Negentropy", final_score[idx.get_loc(probe_ts)] if probe_ts in idx else 0, temp_vals, "Negentropy_V30")
            
        return pd.Series(final_score, index=idx)

    def _calculate_pentagonal_resonance(self, D1: pd.Series, D2: pd.Series, D3: pd.Series, D4: pd.Series, D5: pd.Series, 
                                      df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        """【V36.0 · 动态自适应相变增益 (APT-Gain)】
        PROCESS_META_COST_ADVANTAGE_TREND::Dimension_Fusion
        版本特性：
        1. APT-Gain (自适应相变): 替代静态指数激活。引入 Market Temperature (市场温度) 概念，动态调节激活阈值(Threshold)和爆发系数(Gamma)。
           - 牛市/主升: 阈值降低(易共振), Gamma升高(高溢价)。
           - 熊市/震荡: 阈值升高(严过滤), Gamma降低(防骗炮)。
        2. Full Inheritance: 完整保留 V35 的所有逻辑 (HAB惯性、运动学、张量交互、外部验证、反身性)。
        """
        # --- 1. 数据源提取 (Data Extraction) ---
        close = df['close_D']
        adx = df.get('ADX_14_D') # [Source 1]
        sentiment = df.get('market_sentiment_score_D') # [Source 3]
        sm_divergence = df.get('SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D')
        
        # [V35] 身份与能量数据
        is_leader = df.get('IS_MARKET_LEADER_D')
        leader_score = df.get('industry_leader_score_D')
        turnover = df.get('turnover_rate_f_D')
        vol_ratio = df.get('volume_ratio_D')
        breakout_quality = df.get('breakout_quality_score_D')

        # --- 2. 核心共振计算 (Core Resonance) [继承 V35] ---
        # 动态权重 & 线性分
        norm_adx = (adx / 50.0).clip(0, 2.0) if adx is not None else pd.Series(0.5, index=idx)
        trend_factor = np.tanh(norm_adx - 0.5)
        
        base_w = {'w1': 0.20, 'w2': 0.30, 'w3': 0.20, 'w4': 0.15, 'w5': 0.15}
        adjust_strength = 0.1
        W1 = base_w['w1'] - trend_factor * adjust_strength
        W5 = base_w['w5'] - trend_factor * adjust_strength
        W2 = base_w['w2'] + trend_factor * adjust_strength
        W3 = base_w['w3'] + trend_factor * adjust_strength
        W4 = base_w['w4']
        
        linear_score = (D1 * W1 + D2 * W2 + D3 * W3 + D4 * W4 + D5 * W5)
        
        # 张量交互
        def soft_prod(a, b): return (a * b) / (1.0 + (a * b).abs())
        i1, i2, i3 = soft_prod(D1, D2) * 1.5, soft_prod(D2, D3) * 1.2, soft_prod(D3, D4)
        i4, i5 = soft_prod(D4, D5), soft_prod(D5, D1)
        tensor_score = (i1 + i2 + i3 + i4 + i5) / 5.0
        min_interaction = pd.concat([i1, i2, i3, i4, i5], axis=1).min(axis=1)
        chain_break_penalty = np.where(min_interaction < -0.1, 1.0 + min_interaction, 1.0)
        
        base_resonance = linear_score * (1.0 + tensor_score) * chain_break_penalty

        # HAB 惯性 & 运动学
        hab_resonance_21 = base_resonance.rolling(window=21, min_periods=1).mean()
        inertia_diff = base_resonance - hab_resonance_21
        inertia_factor = np.where(inertia_diff > 0, 1.0 + inertia_diff * 0.5, 1.0 + inertia_diff * 1.5).clip(0.5, 1.5)
        
        res_slope = base_resonance.diff(8).fillna(0)
        res_accel = res_slope.diff(5).fillna(0)
        res_jerk = res_accel.diff(3).fillna(0)
        
        scale_slope = res_slope.rolling(21).std().replace(0, 0.01)
        scale_accel = res_accel.rolling(21).std().replace(0, 0.005)
        scale_jerk = res_jerk.rolling(21).std().replace(0, 0.001)
        
        norm_slope = np.tanh(res_slope / (scale_slope + 1e-8))
        norm_accel = np.tanh(res_accel / (scale_accel + 1e-8))
        norm_jerk = np.tanh(res_jerk / (scale_jerk + 1e-8))
        
        kinematic_score = (norm_slope * 0.6 + norm_accel * 0.4)
        jerk_penalty = np.exp(-(norm_jerk * 1.2)**2)
        kinematic_factor = (1.0 + kinematic_score * 0.4) * jerk_penalty

        # 微分协同
        slopes_matrix = np.column_stack([D1.diff(5).fillna(0), D2.diff(5).fillna(0), D3.diff(5).fillna(0), D4.diff(5).fillna(0), D5.diff(5).fillna(0)])
        deriv_synergy_factor = (1.0 - np.std(slopes_matrix, axis=1) * 2.0).clip(0.6, 1.0)
        
        # 核心分
        core_score = base_resonance * inertia_factor * deriv_synergy_factor * kinematic_factor

        # --- 3. 外部验证 (External Validation) [继承 V35] ---
        # 身份溢价
        status_multiplier = pd.Series(1.0, index=idx)
        if is_leader is not None: status_multiplier = np.where(is_leader > 0, 1.2, status_multiplier)
        if leader_score is not None: status_multiplier = np.where(leader_score > 80, np.maximum(status_multiplier, 1.1), status_multiplier)
        
        # 流动性门控
        liquidity_factor = pd.Series(1.0, index=idx)
        if turnover is not None and vol_ratio is not None:
            zombie_mask = (vol_ratio < 0.6) & (turnover < 1.5)
            churning_mask = (turnover > 25.0) & (status_multiplier < 1.2)
            liquidity_factor = np.where(zombie_mask | churning_mask, 0.8, 1.0)
            
        # 突破确认
        breakout_factor = pd.Series(1.0, index=idx)
        if breakout_quality is not None:
            norm_bq = breakout_quality / 100.0
            breakout_factor = np.where((core_score > 0.5) & (norm_bq < 0.3), 0.7, np.where(norm_bq > 0.8, 1.1, 1.0))
            
        final_score = core_score * status_multiplier * liquidity_factor * breakout_factor
        
        # 熔断与反身性
        if sm_divergence is not None:
             veto_multiplier = np.where(sm_divergence > 1.5, 0.0, np.where(sm_divergence > 0.8, 0.5, 1.0))
             final_score = final_score * veto_multiplier
             
        res_diff = pd.Series(final_score, index=idx).diff(8).fillna(0)
        price_diff = close.diff(8).fillna(0)
        raw_corr = res_diff.rolling(13).corr(price_diff).fillna(0)
        res_std = res_diff.rolling(13).std().fillna(0)
        active_mask = (res_std > 0.05)
        reflexivity_factor = np.where(active_mask & (raw_corr > 0.5), 1.0 + (raw_corr - 0.5), 
                                    np.where(active_mask & (raw_corr < -0.3), 1.0 + (raw_corr + 0.3), 1.0))
        final_score = final_score * reflexivity_factor

        # --- 4. 动态自适应相变增益 (APT-Gain) [V36核心] ---
        # 逻辑: 替代静态激活。根据 Market Temperature 动态调整 Threshold 和 Gamma。
        
        # 1. 计算市场温度 (0~1)
        # Sentiment 权重 0.6, Trend 权重 0.4
        temp_sentiment = sentiment if sentiment is not None else pd.Series(0.5, index=idx)
        temp_trend = (adx / 60.0).clip(0, 1.0) if adx is not None else pd.Series(0.5, index=idx) # ADX 60 视为极热
        
        market_temp = temp_sentiment * 0.6 + temp_trend * 0.4
        
        # 2. 动态阈值 (Dynamic Threshold)
        # 基准 0.6。
        # 温度高(0.8) -> 阈值降至 0.5 (容易触发)
        # 温度低(0.2) -> 阈值升至 0.7 (严格过滤)
        # 公式: Base - (Temp - 0.5) * Sensitivity
        dyn_threshold = 0.6 - (market_temp - 0.5) * 0.4 
        dyn_threshold = dyn_threshold.clip(0.4, 0.8) # 限制范围
        
        # 3. 动态伽马 (Dynamic Gamma)
        # 基准 2.0。
        # 温度高 -> Gamma 升至 3.0 (高溢价，高爆发)
        # 温度低 -> Gamma 降至 1.0 (线性，低溢价)
        # 公式: Base + (Temp - 0.5) * Sensitivity
        dyn_gamma = 2.0 + (market_temp - 0.5) * 2.0
        dyn_gamma = dyn_gamma.clip(1.0, 4.0)
        
        # 4. 应用 APT-Gain
        # 仅对超过阈值的正分进行增益
        # G(x) = x * (1 + alpha * exp(gamma * (x - threshold)))
        # alpha 设为 0.6 (基础增益幅度)
        
        apt_score = np.where(
            final_score > dyn_threshold,
            final_score * (1.0 + 0.6 * np.exp(dyn_gamma * (final_score - dyn_threshold))),
            final_score
        ).clip(-1, 4.0) # 允许更高的上限以体现高溢价

        # --- 5. 裸露探针输出 ---
        if is_debug and probe_ts:
            self._probe_val("Market_Temp", market_temp.loc[probe_ts], temp_vals, "Pentagonal_V36")
            self._probe_val("Dyn_Threshold", dyn_threshold.loc[probe_ts], temp_vals, "Pentagonal_V36")
            self._probe_val("Dyn_Gamma", dyn_gamma.loc[probe_ts], temp_vals, "Pentagonal_V36")
            self._probe_val("Pre_Gain_Score", final_score[idx.get_loc(probe_ts)] if probe_ts in idx else 0, temp_vals, "Pentagonal_V36")
            self._probe_val("Final_V36_APT_Score", apt_score[idx.get_loc(probe_ts)] if probe_ts in idx else 0, temp_vals, "Pentagonal_V36")

        return pd.Series(apt_score, index=idx)

    def _check_and_repair_signals(self, df: pd.DataFrame, method_name: str) -> pd.DataFrame:
        """【V11.0 信号检查】"""
        required_cols = [
            'winner_rate_D', 'chip_stability_D', 'chip_concentration_ratio_D',
            'SMART_MONEY_HM_NET_BUY_D', 'SMART_MONEY_HM_COORDINATED_ATTACK_D',
            'SMART_MONEY_INST_NET_BUY_D', 'MA_VELOCITY_EMA_55_D',
            'VPA_EFFICIENCY_D', 'market_sentiment_score_D',
            'cost_50pct_D', 'close_D', 'chip_entropy_D'
        ]
        # intraday_chip_entropy_D 是可选的，如果不存在，在计算方法内部会用 chip_entropy_D 替代
        
        for col in required_cols:
            if col not in df.columns:
                print(f"!!! CRITICAL WARNING: Missing Column {col} in {method_name} !!!")
                df[col] = 0.0
        return df

    def _log_debug_values(self, debug_out: Dict, temp_vals: Dict, probe_ts: pd.Timestamp, method_name: str):
        print(f"\n====== {method_name} @ {probe_ts.strftime('%Y-%m-%d')} ======")
        for section, data in temp_vals.items():
            print(f"[{section}]")
            for k, v in data.items():
                print(f"  {k:<25}: {v}")
        print("=========================================================\n")