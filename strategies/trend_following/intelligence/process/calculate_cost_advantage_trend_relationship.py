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
        # 1. 获取调试开关状态
        is_debug = get_param_value(self.debug_params.get('enabled'), False)
        probe_ts = None
        
        # 2. 只有开启调试且数据不为空时才进行匹配
        if is_debug and not df.empty:
            # 【核心修复】将目标日期转为字符串集合 (例如 {'2025-12-30'})，彻底解决时区/时间精度不一致问题
            target_dates_str = set()
            if self.probe_dates:
                # 兼容处理：确保输入转为字符串
                target_dates_str = set(pd.to_datetime(d).strftime('%Y-%m-%d') for d in self.probe_dates)
            
            # 遍历数据索引（倒序查找，优先匹配最近的日期）
            for date in reversed(df.index):
                # 将当前数据的索引日期也转为字符串
                current_date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
                
                # 字符串比对：只要日期文字一样，就算匹配成功
                if current_date_str in target_dates_str:
                    probe_ts = date
                    break
            
            # 【兜底机制】如果比对失败（例如数据日期是 2025-12-29 而探针是 30 号），强制使用最后一行数据
            if probe_ts is None and target_dates_str:
                last_date = df.index[-1]
                last_date_str = pd.to_datetime(last_date).strftime('%Y-%m-%d')
                print(f"!!! [探针修正] 指定日期 {self.probe_dates} 未匹配，强制使用数据最后一天: {last_date_str} !!!")
                probe_ts = last_date

        debug_output = {}
        temp_vals = {}
        
        # 3. 激活探针上下文
        if is_debug and probe_ts is not None:
            # 打印时也用字符串，确保显示整洁
            ts_display = pd.to_datetime(probe_ts).strftime('%Y-%m-%d')
            print(f"【V11.0探针激活】目标日期: {ts_display} | 方法: {method_name}")
            debug_output[f"--- {method_name} Probe @ {ts_display} ---"] = ""
            
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
        print(f" ====== Start CalculateCostAdvantageTrendRelationship【V11.0.0】启动五维成本资金共振计算，数据形状: {df.shape} ======")
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
        print(f" ====== End【V11.0.0】五维成本资金共振计算完成 ======")
        return final_score.astype(np.float32).fillna(0).clip(-1, 1)

    def _calculate_chip_barrier_solidity(self, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        """【V14.1.1 · 非线性临界相变筹码势垒 (全链路探针版)】"""
        winner_rate = self.helper._get_safe_series(df, 'winner_rate_D', 50.0, "winner_rate")
        stability = self.helper._get_safe_series(df, 'chip_stability_D', 1.0, "stability")
        concentration = self.helper._get_safe_series(df, 'chip_concentration_ratio_D', 0.2, "concentration")
        kurtosis = self.helper._get_safe_series(df, 'chip_kurtosis_D', 3.0, "kurtosis")
        lock_ratio = self.helper._get_safe_series(df, 'high_position_lock_ratio_90_D', 0.0, "lock_ratio")
        cost_5 = self.helper._get_safe_series(df, 'cost_5pct_D', 1.0, "cost_5")
        cost_50 = self.helper._get_safe_series(df, 'cost_50pct_D', 1.0, "cost_50")
        cost_95 = self.helper._get_safe_series(df, 'cost_95pct_D', 1.0, "cost_95")
        intra_consolidation = self.helper._get_safe_series(df, 'intraday_chip_consolidation_degree_D', 0.5, "intra_consolidation")
        span_ratio = (cost_95 - cost_5) / (cost_50 + 1e-8)
        score_span = (0.4 - span_ratio) * 3.33
        score_concent = (0.25 - concentration) * 5.0
        morph_multiplier = 0.5 + np.tanh((kurtosis - 2.5) / 2.0)
        raw_structure = (score_concent * 0.5 + score_span * 0.5) * morph_multiplier
        dim_structure = np.tanh(raw_structure * 1.5)
        hab_stability_34 = stability.rolling(window=34, min_periods=1).mean()
        raw_stability = 0.3 * np.tanh(np.log1p(stability)) + 0.7 * np.tanh(np.log1p(hab_stability_34))
        dim_stability = np.tanh(raw_stability * 2.0)
        score_micro = (intra_consolidation - 0.3) * 2.0
        score_lock = (lock_ratio * 2.0)
        raw_behavior = score_lock * 0.6 + score_micro * 0.4
        dim_behavior = np.tanh(raw_behavior * 1.5)
        slope_conc = self.helper._get_safe_series(df, 'SLOPE_13_chip_concentration_ratio_D', 0.0, "slope_conc")
        accel_conc = self.helper._get_safe_series(df, 'ACCEL_13_chip_concentration_ratio_D', 0.0, "accel_conc")
        jerk_conc = self.helper._get_safe_series(df, 'JERK_13_chip_concentration_ratio_D', 0.0, "jerk_conc")
        norm_slope = np.tanh(slope_conc * -10.0)
        norm_accel = np.tanh(accel_conc * -5.0)
        norm_jerk_penalty = np.exp(-(jerk_conc * 5.0) ** 2)
        dim_kinematics = (norm_slope * 0.6 + norm_accel * 0.4) * norm_jerk_penalty
        dims_matrix = pd.concat([dim_structure, dim_stability, dim_behavior], axis=1)
        synergy_std = dims_matrix.std(axis=1, ddof=0).fillna(0.5)
        synergy_factor = (1.0 - synergy_std).clip(0.1, 1.0)
        base_score = (dim_structure * 0.3 + dim_stability * 0.3 + dim_behavior * 0.2 + dim_kinematics * 0.2)
        synergy_score = base_score * synergy_factor
        threshold = 0.5
        raw_phase_gain = np.where(synergy_score > threshold, synergy_score * (1 + 0.5 * np.exp(3.0 * (synergy_score - threshold))), synergy_score)
        phase_transition_gain = pd.Series(raw_phase_gain, index=idx)
        collapse_risk = (norm_jerk_penalty < 0.2) & (score_span < 0)
        final_solidity = pd.Series(np.where(collapse_risk, -0.8, phase_transition_gain), index=idx).clip(-1, 1.2)
        if is_debug and probe_ts:
            p_val = lambda s: s.loc[probe_ts] if probe_ts in s.index else 0
            print(f"[Probe-D1] 筹码势垒详情 @ {probe_ts.strftime('%Y-%m-%d')}")
            print(f"  > 原始输入: SpanRatio={p_val(span_ratio):.3f}, Kurtosis={p_val(kurtosis):.3f}, Stability={p_val(stability):.3f}")
            print(f"  > 子维度分: Structure={p_val(dim_structure):.3f}, Stability={p_val(dim_stability):.3f}, Behavior={p_val(dim_behavior):.3f}")
            print(f"  > 运动学分: Kinematics={p_val(dim_kinematics):.3f} (JerkPenalty={p_val(norm_jerk_penalty):.3f})")
            print(f"  > 协同计算: BaseScore={p_val(base_score):.3f}, SynergyFactor={p_val(synergy_factor):.3f} -> SynergyScore={p_val(synergy_score):.3f}")
            print(f"  > 最终结果: PhaseTransition={p_val(phase_transition_gain):.3f} -> Final={p_val(final_solidity):.3f}")
            self._probe_val("Final_Solidity", final_solidity.loc[probe_ts], temp_vals, "ChipSolidity_V14.1")
        return final_solidity

    def _calculate_predator_attack_vector(self, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        """【V18.1.1 · 超临界相变捕食者矢量 (全链路探针版)】"""
        sm_net_buy = self.helper._get_safe_series(df, 'SMART_MONEY_HM_NET_BUY_D', 0.0, "sm_net_buy")
        inst_buy = self.helper._get_safe_series(df, 'SMART_MONEY_INST_NET_BUY_D', 0.0, "inst_buy")
        coord_attack = self.helper._get_safe_series(df, 'SMART_MONEY_HM_COORDINATED_ATTACK_D', 0.0, "coord_attack")
        sm_accel = self.helper._get_safe_series(df, 'ACCEL_13_SMART_MONEY_HM_NET_BUY_D', 0.0, "sm_accel")
        sm_jerk = self.helper._get_safe_series(df, 'JERK_13_SMART_MONEY_HM_NET_BUY_D', 0.0, "sm_jerk")
        stealth_flow = self.helper._get_safe_series(df, 'stealth_flow_ratio_D', 0.0, "stealth_flow")
        morning_ratio = self.helper._get_safe_series(df, 'morning_flow_ratio_D', 0.0, "morning_ratio")
        closing_intensity = self.helper._get_safe_series(df, 'closing_flow_intensity_D', 0.0, "closing_intensity")
        flow_impact = self.helper._get_safe_series(df, 'flow_impact_ratio_D', 1.0, "flow_impact")
        anomaly_val = self.helper._get_safe_series(df, 'large_order_anomaly_D', 0.0, "anomaly_val")
        cluster_intensity = self.helper._get_safe_series(df, 'flow_cluster_intensity_D', 0.0, "cluster_intensity")
        consistency = self.helper._get_safe_series(df, 'flow_consistency_D', 0.5, "consistency")
        slope_consistency = self.helper._get_safe_series(df, 'SLOPE_13_flow_consistency_D', 0.0, "slope_consistency")
        activity = self.helper._get_safe_series(df, 'main_force_activity_index_D', 0.0, "activity")
        sm_hab_21 = sm_net_buy.rolling(window=21, min_periods=1).sum()
        hab_capacity = sm_hab_21.abs().rolling(55).max().replace(0, 1.0)
        norm_hab = (sm_hab_21 / hab_capacity).clip(-1, 1)
        roll_max_sm = sm_net_buy.abs().rolling(21).max().replace(0, 1.0)
        norm_vel = (sm_net_buy / roll_max_sm).clip(-1, 1)
        accel_scale = sm_accel.rolling(21).std().replace(0, 1.0)
        jerk_scale = sm_jerk.rolling(21).std().replace(0, 1.0)
        norm_accel = np.tanh(sm_accel / (accel_scale + 1e-8))
        norm_jerk = np.tanh(sm_jerk / (jerk_scale + 1e-8))
        instant_impulse = norm_vel * 0.5 + norm_accel * 0.3 + norm_jerk * 0.2
        vec_power = (norm_hab * 0.6 + instant_impulse * 0.4).clip(-1, 1)
        norm_stealth = np.tanh(stealth_flow * 2.0)
        score_morning = ((morning_ratio - 0.2) * 2.5).clip(-0.5, 1)
        score_closing = np.tanh(closing_intensity)
        vec_guile = (norm_stealth * 0.5 + score_morning * 0.3 + score_closing * 0.2).clip(-1, 1)
        eff_score = np.tanh(flow_impact - 1.0)
        struct_score = np.tanh(anomaly_val / 2.0) * 0.6 + np.tanh(cluster_intensity) * 0.4
        gate_consistency = ((consistency - 0.5) * 2.0).clip(0, 1)
        slope_scale = slope_consistency.rolling(21).std().replace(0, 0.1)
        norm_slope_consist = np.tanh(slope_consistency / (slope_scale + 1e-8))
        consist_score = (gate_consistency * 0.7 + norm_slope_consist * 0.3)
        vec_quality = (eff_score * 0.3 + struct_score * 0.3 + consist_score * 0.4).clip(-1, 1)
        vectors = np.column_stack([vec_power, vec_guile, vec_quality])
        synergy_std = pd.Series(np.std(vectors, axis=1), index=idx)
        synergy_factor = (1.0 - synergy_std * 1.5).clip(0.1, 1.0)
        base_score = (vec_power * 0.4 + vec_guile * 0.3 + vec_quality * 0.3)
        synergized_score = base_score * synergy_factor
        threshold = 0.5
        raw_attack = np.where(synergized_score > threshold, synergized_score * (1.0 + 0.6 * np.exp(2.5 * (synergized_score - threshold))), synergized_score)
        final_attack = pd.Series(raw_attack, index=idx)
        crit_multiplier = pd.Series(np.where(coord_attack > 0.5, 1.3, 1.0), index=idx)
        final_attack = (final_attack * crit_multiplier).clip(-1, 2.0)
        if is_debug and probe_ts:
            p_val = lambda s: s.loc[probe_ts] if probe_ts in s.index else 0
            print(f"[Probe-D2] 捕食者矢量详情 @ {probe_ts.strftime('%Y-%m-%d')}")
            print(f"  > 原始输入: SM_NetBuy={p_val(sm_net_buy):.2f}, Stealth={p_val(stealth_flow):.2f}, Impact={p_val(flow_impact):.2f}")
            print(f"  > 向量分量: Power={p_val(vec_power):.3f}, Guile={p_val(vec_guile):.3f}, Quality={p_val(vec_quality):.3f}")
            print(f"  > 协同计算: SynergyStd={p_val(synergy_std):.3f} -> Factor={p_val(synergy_factor):.3f}")
            print(f"  > 综合得分: Base={p_val(base_score):.3f} -> Synergized={p_val(synergized_score):.3f}")
            print(f"  > 暴击修正: CoordCrit={p_val(crit_multiplier):.1f} -> Final={p_val(final_attack):.3f}")
            self._probe_val("Final_Attack", final_attack.loc[probe_ts], temp_vals, "PredatorVector_V18.1")
        return final_attack

    def _calculate_kinematic_efficiency(self, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        """【V22.1.1 · 超导相变与非线性熵减 (全链路探针版)】"""
        velocity = self.helper._get_safe_series(df, 'MA_VELOCITY_EMA_55_D', 0.0, "velocity")
        accel = self.helper._get_safe_series(df, 'MA_ACCELERATION_EMA_55_D', 0.0, "accel")
        vel_norm = np.tanh(velocity / velocity.rolling(89).std().replace(0, 1.0))
        accel_norm = np.tanh(accel / accel.rolling(89).std().replace(0, 1.0))
        mf_vpa = self.helper._get_safe_series(df, 'VPA_MF_ADJUSTED_EFF_D', 0.0, "mf_vpa")
        mf_vpa_norm = mf_vpa / (mf_vpa.abs().rolling(21).max().replace(0, 1.0))
        flow_impact = self.helper._get_safe_series(df, 'flow_impact_ratio_D', 1.0, "flow_impact")
        impact_norm = np.tanh(flow_impact - 1.0)
        macro_work = (mf_vpa_norm * 0.6 + impact_norm * 0.4).clip(-1, 1)
        tick_eff = self.helper._get_safe_series(df, 'tick_chip_transfer_efficiency_D', 0.5, "tick_eff")
        micro_work = ((tick_eff - 0.5) * 3.33).clip(-1, 1)
        entropy = self.helper._get_safe_series(df, 'PRICE_ENTROPY_D', 0.5, "entropy")
        negentropy = (0.6 - entropy) * 2.5
        negentropy = negentropy.clip(-1, 1)
        synergy_gap = np.abs(macro_work - micro_work)
        synergy_factor = (1.0 - synergy_gap * 0.8).clip(0.2, 1.0)
        base_efficiency = ((macro_work + micro_work) / 2.0) * synergy_factor
        entropy_gate = (negentropy + 1.0) / 2.0
        instant_efficiency = base_efficiency * entropy_gate
        resistance = self.helper._get_safe_series(df, 'resistance_strength_D', 0.0, "resistance")
        resist_norm = np.tanh(resistance / 5.0)
        friction_factor = np.where((instant_efficiency > 0.3) & (resist_norm > 0.5), 1.4, np.where((instant_efficiency < 0) & (resist_norm > 0.5), 0.6, 1.0))
        instant_efficiency = (instant_efficiency * friction_factor).clip(-1, 1)
        threshold = 0.5
        raw_super_conductive_gain = np.where((instant_efficiency > threshold), instant_efficiency * (1.0 + 0.8 * np.exp(2.0 * (instant_efficiency - threshold))), instant_efficiency)
        super_conductive_gain = pd.Series(raw_super_conductive_gain, index=idx).clip(-1, 2.0)
        hab_efficiency_34 = super_conductive_gain.rolling(window=34, min_periods=1).mean()
        score_hab = np.tanh(hab_efficiency_34)
        eff_slope = super_conductive_gain.diff(13).fillna(0)
        eff_accel = eff_slope.diff(5).fillna(0)
        eff_jerk = eff_accel.diff(3).fillna(0)
        slope_scale = eff_slope.rolling(21).std().replace(0, 0.01)
        accel_scale = eff_accel.rolling(21).std().replace(0, 0.01)
        jerk_scale = eff_jerk.rolling(21).std().replace(0, 0.01)
        norm_slope = np.tanh(eff_slope / (slope_scale + 1e-8))
        norm_accel = np.tanh(eff_accel / (accel_scale + 1e-8))
        norm_jerk = np.tanh(eff_jerk / (jerk_scale + 1e-8))
        jerk_penalty = np.exp(-(norm_jerk * 2.0) ** 2)
        derivative_score = (norm_slope * 0.6 + norm_accel * 0.4) * jerk_penalty
        total_thermodynamics = (score_hab * 0.4 + super_conductive_gain * 0.3 + derivative_score * 0.3)
        alignment = np.sign(vel_norm) * np.sign(total_thermodynamics)
        final_efficiency = np.where(alignment > 0, vel_norm * (0.6 + 0.4 * np.abs(total_thermodynamics)), vel_norm * (1.0 - np.abs(total_thermodynamics))).clip(-1, 1.5)
        final_efficiency_series = pd.Series(final_efficiency, index=idx)
        if is_debug and probe_ts:
            p_val = lambda s: s.loc[probe_ts] if probe_ts in s.index else 0
            print(f"[Probe-D3] 趋势做功效率详情 @ {probe_ts.strftime('%Y-%m-%d')}")
            print(f"  > 做功分量: MacroWork={p_val(macro_work):.3f}, MicroWork={p_val(micro_work):.3f} (TickEff={p_val(tick_eff):.3f})")
            print(f"  > 协同与熵: SynergyGap={p_val(synergy_gap):.3f}, EntropyGate={p_val(entropy_gate):.3f}")
            print(f"  > 阻力摩擦: Resistance={p_val(resistance):.2f}, FrictionFactor={float(friction_factor[idx.get_loc(probe_ts)]) if probe_ts in idx else 0:.2f}")
            print(f"  > 瞬时效率: Base={p_val(base_efficiency):.3f} -> Instant={p_val(instant_efficiency):.3f}")
            print(f"  > 超导增益: SuperGain={p_val(super_conductive_gain):.3f}, HAB={p_val(score_hab):.3f}")
            print(f"  > 最终热力: TotalThermo={p_val(total_thermodynamics):.3f}, Alignment={p_val(pd.Series(alignment, index=idx)):.1f} -> Final={p_val(final_efficiency_series):.3f}")
            self._probe_val("Final_Efficiency", final_efficiency_series.loc[probe_ts], temp_vals, "KinematicEff_V22.1")
        return final_efficiency_series

    def _calculate_cost_migration_elasticity(self, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        """【V26.1.1 · 超弹性共振与反脆弱增益 (全链路探针版)】"""
        close = self.helper._get_safe_series(df, 'close_D', 1.0, "close")
        cost_50 = self.helper._get_safe_series(df, 'cost_50pct_D', 1.0, "cost_50")
        cost_5 = self.helper._get_safe_series(df, 'cost_5pct_D', 1.0, "cost_5")
        cost_95 = self.helper._get_safe_series(df, 'cost_95pct_D', 1.0, "cost_95")
        turnover = self.helper._get_safe_series(df, 'turnover_rate_f_D', 1.0, "turnover")
        profit_pressure = self.helper._get_safe_series(df, 'profit_pressure_D', 0.0, "profit_pressure")
        trapped_pressure = self.helper._get_safe_series(df, 'pressure_trapped_D', 0.0, "trapped_pressure")
        intra_migration = self.helper._get_safe_series(df, 'intraday_cost_center_migration_D', 0.0, "intra_migration")
        intra_volatility = self.helper._get_safe_series(df, 'intraday_cost_center_volatility_D', 0.0, "intra_volatility")
        slope_c50 = cost_50.diff(13).fillna(0)
        slope_c5 = cost_5.diff(13).fillna(0)
        slope_price = close.diff(13).fillna(0)
        accel_c5 = slope_c5.diff(5).fillna(0)
        jerk_c5 = accel_c5.diff(3).fillna(0)
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
        turnover_score = 0.5 + 0.7 * np.exp(-((turnover - 5.5)**2) / 50.0)
        scissor_inst = norm_slope_price - (norm_slope_c50 * 0.8)
        viscous_inst = scissor_inst * turnover_score
        viscous_hab = viscous_inst.rolling(window=34, min_periods=1).mean()
        dim_hab = np.tanh(viscous_hab * 2.0)
        norm_stress = np.tanh(profit_pressure / 10.0)
        norm_strain = np.abs(norm_slope_c50)
        raw_modulus = norm_stress * (1.0 - norm_strain) * 2.5
        dim_modulus = pd.Series(np.where(norm_stress > 0.2, np.tanh(raw_modulus), 0.3), index=idx)
        norm_intra_mig = np.tanh(intra_migration * 5.0)
        norm_intra_vol = np.tanh(intra_volatility * 5.0)
        dim_fractal = (norm_intra_mig * 0.6 + (1.0 - norm_intra_vol) * 0.4).clip(-1, 1)
        dims = np.column_stack([dim_hab, dim_modulus, dim_fractal])
        synergy_std = pd.Series(np.std(dims, axis=1), index=idx)
        synergy_factor = (1.0 - synergy_std * 1.2).clip(0.3, 1.0)
        base_score = (dim_hab * 0.4 + dim_modulus * 0.4 + dim_fractal * 0.2)
        synergized_score = base_score * synergy_factor
        threshold = 0.6
        raw_elast = np.where(synergized_score > threshold, synergized_score * (1.0 + 0.8 * np.exp(2.5 * (synergized_score - threshold))), synergized_score)
        final_elasticity = pd.Series(raw_elast, index=idx)
        violation = (np.maximum(0, norm_slope_c5) * 0.2 + np.maximum(0, norm_accel_c5) * 0.3 + np.maximum(0, norm_jerk_c5) * 0.5)
        anchorage_penalty = pd.Series(np.exp(-violation * 4.0), index=idx)
        ceiling_penalty = pd.Series(np.exp(-trapped_pressure * 2.5), index=idx)
        span = (cost_95 - cost_5) / (cost_50 + 1e-8)
        slope_span = span.diff(13).fillna(0)
        scale_span = slope_span.rolling(21).std().replace(0, 0.01)
        norm_slope_span = np.tanh(slope_span / (scale_span + 1e-8))
        compression_bonus = np.maximum(0, -norm_slope_span) * 0.2
        final_score = (final_elasticity * anchorage_penalty * ceiling_penalty + compression_bonus).clip(-1, 2.0)
        if is_debug and probe_ts:
            p_val = lambda s: s.loc[probe_ts] if probe_ts in s.index else 0
            print(f"[Probe-D4] 成本迁移弹性详情 @ {probe_ts.strftime('%Y-%m-%d')}")
            print(f"  > 状态输入: Turnover={p_val(turnover):.2f}, ProfitPress={p_val(profit_pressure):.2f}, TrappedPress={p_val(trapped_pressure):.2f}")
            print(f"  > 剪刀差与粘性: ScissorInst={p_val(scissor_inst):.3f}, ViscousHAB={p_val(viscous_hab):.3f} -> DimHAB={p_val(dim_hab):.3f}")
            print(f"  > 模量与分形: Modulus={p_val(dim_modulus):.3f}, Fractal={p_val(dim_fractal):.3f}")
            print(f"  > 基础得分: Synergized={p_val(synergized_score):.3f} -> FinalElasticity={p_val(final_elasticity):.3f}")
            print(f"  > 关键惩罚: Violation(底松动)={p_val(pd.Series(violation, index=idx)):.3f} -> AnchorPenalty={p_val(anchorage_penalty):.4f}")
            print(f"  > 关键惩罚: Trapped(顶套牢)={p_val(trapped_pressure):.3f} -> CeilingPenalty={p_val(ceiling_penalty):.4f}")
            print(f"  > 最终结果: Final={p_val(final_score):.4f}")
            self._probe_val("Final_Elasticity", final_score.loc[probe_ts], temp_vals, "CostElasticity_V26.1")
        return final_score

    def _calculate_structure_negentropy(self, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        """【V30.1.1 · 超序参量与乘法锁定熵 (全链路探针版)】"""
        chip_entropy = self.helper._get_safe_series(df, 'chip_entropy_D', 0.5, "chip_entropy")
        conc_entropy = self.helper._get_safe_series(df, 'concentration_entropy_D', 0.5, "conc_entropy")
        intra_entropy = self.helper._get_safe_series(df, 'intraday_chip_entropy_D', 0.5, "intra_entropy")
        price_entropy = self.helper._get_safe_series(df, 'PRICE_ENTROPY_D', 0.5, "price_entropy")
        reg_r2 = self.helper._get_safe_series(df, 'GEOM_REG_R2_D', 0.0, "reg_r2")
        reg_slope = self.helper._get_safe_series(df, 'GEOM_REG_SLOPE_D', 0.0, "reg_slope")
        curvature = self.helper._get_safe_series(df, 'GEOM_ARC_CURVATURE_D', 0.0, "curvature")
        fractal_dim = self.helper._get_safe_series(df, 'PRICE_FRACTAL_DIM_D', 1.5, "fractal_dim")
        flow_stab = self.helper._get_safe_series(df, 'TURNOVER_STABILITY_INDEX_D', 0.5, "flow_stab")
        total_micro_entropy = chip_entropy * 0.4 + conc_entropy * 0.3 + intra_entropy * 0.3
        hab_entropy_34 = total_micro_entropy.rolling(window=34, min_periods=1).mean()
        score_internal = 1.0 / (1.0 + np.exp(12.0 * (hab_entropy_34 - 0.55)))
        negent_price = 1.0 / (1.0 + np.exp(12.0 * (price_entropy - 0.55)))
        score_r2 = ((reg_r2 - 0.6) * 2.5).clip(0, 1)
        score_external = np.sqrt(negent_price * score_r2)
        curv_volatility = curvature.rolling(13).std().replace(0, 0.01)
        score_smoothness = np.tanh((0.1 - curv_volatility) * 10.0)
        order_parameter = np.sqrt(score_internal * score_external)
        lock_gain = 1.0 + 1.0 * np.power(order_parameter, 4)
        crystal_score = order_parameter * lock_gain
        slope_ent = total_micro_entropy.diff(13).fillna(0)
        scale_slope = slope_ent.rolling(21).std().replace(0, 0.01)
        norm_slope_ent = np.tanh(-slope_ent / (scale_slope + 1e-8))
        raw_modulator = np.where(norm_slope_ent > 0, 1.0 + norm_slope_ent * 0.2, 1.0 + norm_slope_ent * 0.5)
        kinetics_modulator = pd.Series(raw_modulator, index=idx)
        fractal_gate = pd.Series(np.where(fractal_dim < 1.4, 1.0, 0.5), index=idx)
        laminar_gate = pd.Series(np.where(flow_stab > 0.6, 1.0, 0.7), index=idx)
        smooth_gate = (score_smoothness + 1.0) / 2.0
        synergized_negentropy = crystal_score * kinetics_modulator * fractal_gate * laminar_gate * (0.8 + 0.2 * smooth_gate)
        trend_gate = np.tanh(reg_slope * 10.0)
        final_negentropy = synergized_negentropy * trend_gate
        threshold = 0.6
        raw_final = np.where(final_negentropy > threshold, final_negentropy * (1.0 + 0.5 * np.exp(2.0 * (final_negentropy - threshold))), final_negentropy)
        final_score = pd.Series(raw_final, index=idx).clip(-1, 2.0)
        turnover = self.helper._get_safe_series(df, 'turnover_rate_f_D', 1.0, "turnover")
        final_score = pd.Series(np.where(turnover < 0.5, 0.0, final_score), index=idx)
        if is_debug and probe_ts:
            p_val = lambda s: s.loc[probe_ts] if probe_ts in s.index else 0
            print(f"[Probe-D5] 结构熵逆详情 @ {probe_ts.strftime('%Y-%m-%d')}")
            print(f"  > 熵值输入: MicroEntropy={p_val(total_micro_entropy):.3f}, PriceEntropy={p_val(price_entropy):.3f}, HAB={p_val(hab_entropy_34):.3f}")
            print(f"  > 核心评分: Internal={p_val(score_internal):.3f}, External={p_val(score_external):.3f} -> OrderParam={p_val(order_parameter):.3f}")
            print(f"  > 晶体化: CrystalScore={p_val(crystal_score):.3f} (LockGain={p_val(lock_gain):.2f})")
            print(f"  > 动力学门控: Kinetics={p_val(kinetics_modulator):.2f}, FractalDim={p_val(fractal_dim):.2f}(Gate={p_val(fractal_gate):.1f})")
            print(f"  > 最终门控: TrendGate={p_val(pd.Series(trend_gate, index=idx)):.2f}, Turnover={p_val(turnover):.2f}")
            print(f"  > 最终结果: Synergized={p_val(synergized_negentropy):.4f} -> Final={p_val(final_score):.4f} {'(熔断)' if p_val(turnover) < 0.5 else ''}")
            self._probe_val("Final_Negentropy", final_score.loc[probe_ts], temp_vals, "Negentropy_V30.1")
        return final_score

    def _calculate_pentagonal_resonance(self, D1: pd.Series, D2: pd.Series, D3: pd.Series, D4: pd.Series, D5: pd.Series, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        """【V36.1.1 · 动态自适应相变增益 APT-Gain (全链路探针版)】"""
        close = df['close_D']
        adx = df.get('ADX_14_D')
        sentiment = df.get('market_sentiment_score_D')
        sm_divergence = df.get('SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D')
        is_leader = df.get('IS_MARKET_LEADER_D')
        leader_score = df.get('industry_leader_score_D')
        turnover = df.get('turnover_rate_f_D')
        vol_ratio = df.get('volume_ratio_D')
        breakout_quality = df.get('breakout_quality_score_D')
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
        def soft_prod(a, b): return (a * b) / (1.0 + (a * b).abs())
        i1, i2, i3 = soft_prod(D1, D2) * 1.5, soft_prod(D2, D3) * 1.2, soft_prod(D3, D4)
        i4, i5 = soft_prod(D4, D5), soft_prod(D5, D1)
        tensor_score = (i1 + i2 + i3 + i4 + i5) / 5.0
        min_interaction = pd.concat([i1, i2, i3, i4, i5], axis=1).min(axis=1)
        chain_break_penalty = pd.Series(np.where(min_interaction < -0.1, 1.0 + min_interaction, 1.0), index=idx)
        base_resonance = linear_score * (1.0 + tensor_score) * chain_break_penalty
        hab_resonance_21 = base_resonance.rolling(window=21, min_periods=1).mean()
        inertia_diff = base_resonance - hab_resonance_21
        inertia_factor = pd.Series(np.where(inertia_diff > 0, 1.0 + inertia_diff * 0.5, 1.0 + inertia_diff * 1.5), index=idx).clip(0.5, 1.5)
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
        slopes_matrix = np.column_stack([D1.diff(5).fillna(0), D2.diff(5).fillna(0), D3.diff(5).fillna(0), D4.diff(5).fillna(0), D5.diff(5).fillna(0)])
        deriv_synergy_factor = pd.Series(1.0 - np.std(slopes_matrix, axis=1) * 2.0, index=idx).clip(0.6, 1.0)
        core_score = base_resonance * inertia_factor * deriv_synergy_factor * kinematic_factor
        status_multiplier = pd.Series(1.0, index=idx)
        if is_leader is not None: status_multiplier = pd.Series(np.where(is_leader > 0, 1.2, status_multiplier), index=idx)
        if leader_score is not None: status_multiplier = pd.Series(np.where(leader_score > 80, np.maximum(status_multiplier, 1.1), status_multiplier), index=idx)
        liquidity_factor = pd.Series(1.0, index=idx)
        if turnover is not None and vol_ratio is not None:
            zombie_mask = (vol_ratio < 0.6) & (turnover < 1.5)
            churning_mask = (turnover > 25.0) & (status_multiplier < 1.2)
            liquidity_factor = pd.Series(np.where(zombie_mask | churning_mask, 0.8, 1.0), index=idx)
        breakout_factor = pd.Series(1.0, index=idx)
        if breakout_quality is not None:
            norm_bq = breakout_quality / 100.0
            breakout_factor = pd.Series(np.where((core_score > 0.5) & (norm_bq < 0.3), 0.7, np.where(norm_bq > 0.8, 1.1, 1.0)), index=idx)
        final_score = core_score * status_multiplier * liquidity_factor * breakout_factor
        if sm_divergence is not None:
             veto_multiplier = pd.Series(np.where(sm_divergence > 1.5, 0.0, np.where(sm_divergence > 0.8, 0.5, 1.0)), index=idx)
             final_score = final_score * veto_multiplier
        res_diff = final_score.diff(8).fillna(0)
        price_diff = close.diff(8).fillna(0)
        raw_corr = res_diff.rolling(13).corr(price_diff).fillna(0)
        res_std = res_diff.rolling(13).std().fillna(0)
        active_mask = (res_std > 0.05)
        reflexivity_factor = pd.Series(np.where(active_mask & (raw_corr > 0.5), 1.0 + (raw_corr - 0.5), np.where(active_mask & (raw_corr < -0.3), 1.0 + (raw_corr + 0.3), 1.0)), index=idx)
        final_score = final_score * reflexivity_factor
        temp_sentiment = sentiment if sentiment is not None else pd.Series(0.5, index=idx)
        temp_trend = (adx / 60.0).clip(0, 1.0) if adx is not None else pd.Series(0.5, index=idx)
        market_temp = temp_sentiment * 0.6 + temp_trend * 0.4
        dyn_threshold = 0.6 - (market_temp - 0.5) * 0.4
        dyn_threshold = dyn_threshold.clip(0.4, 0.8)
        dyn_gamma = 2.0 + (market_temp - 0.5) * 2.0
        dyn_gamma = dyn_gamma.clip(1.0, 4.0)
        raw_apt = np.where(final_score > dyn_threshold, final_score * (1.0 + 0.6 * np.exp(dyn_gamma * (final_score - dyn_threshold))), final_score)
        apt_score = pd.Series(raw_apt, index=idx).clip(-1, 4.0)
        if is_debug and probe_ts:
            p_val = lambda s: s.loc[probe_ts] if probe_ts in s.index else 0
            print(f"[Probe-Fusion] 张量共振融合 @ {probe_ts.strftime('%Y-%m-%d')}")
            print(f"  > 线性部分: D1={p_val(D1):.2f}, D2={p_val(D2):.2f}, D3={p_val(D3):.2f}, D4={p_val(D4):.2f}, D5={p_val(D5):.2f} -> Linear={p_val(linear_score):.3f}")
            print(f"  > 张量交互: TensorScore={p_val(tensor_score):.3f} (MinInteract={p_val(min_interaction):.3f})")
            print(f"  > 动力学修正: Inertia={p_val(inertia_factor):.2f}, Kinematic={p_val(kinematic_factor):.2f}, DerivSynergy={p_val(deriv_synergy_factor):.2f}")
            print(f"  > 外部修正: Status={p_val(status_multiplier):.2f}, Liquidity={p_val(liquidity_factor):.2f}, Breakout={p_val(breakout_factor):.2f}")
            print(f"  > 核心分值: Base={p_val(base_resonance):.3f} -> Core={p_val(core_score):.3f} -> FinalPreAPT={p_val(final_score):.3f}")
            print(f"  > APT相变: MarketTemp={p_val(market_temp):.2f} -> Threshold={p_val(dyn_threshold):.2f} -> APT_Score={p_val(apt_score):.4f}")
            self._probe_val("Final_V36_APT_Score", apt_score.loc[probe_ts], temp_vals, "Pentagonal_V36.1")
        return apt_score

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