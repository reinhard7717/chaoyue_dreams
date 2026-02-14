# 文件: strategies/trend_following/intelligence/process/calculate_main_force_rally_intent.py
# 版本: V15.0 · 全息推力-阻力张量版 已升级pro
# 说明: 引入“推力-阻力”物理模型，深度集成VPA效率与控盘坚实度，废弃线性加权，采用张量乘积合成。全链路暴露中间变量。
import pandas as pd
import numpy as np
import numba
from typing import Dict, List, Any
from strategies.trend_following.utils import get_params_block, get_param_value

class CalculateMainForceRallyIntent:
    """
    PROCESS_META_MAIN_FORCE_RALLY_INTENT
    【V15.0 · 全息推力-阻力张量版】
    基于 A 股“资金-结构-效率”三维物理场。
    核心方程：RallyIntent = (Kinetics * Control) / (1 + Resistance)
    1. 引入 VPA_EFFICIENCY_D 衡量拉升效率（阻力系数）。
    2. 引入 control_solidity_index_D 衡量控盘状态。
    3. 引入 flow_consistency_D 剔除突击一日游资金。
    """
    def __init__(self, strategy_instance, process_intelligence_helper_instance):
        self.strategy = strategy_instance
        self.helper = process_intelligence_helper_instance
        self.debug_params = self.helper.debug_params
        self.probe_dates = self.helper.probe_dates
        self._probe_cache = []

    def _load_data(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V15.0】加载数据，NaN填充为0.0以保持张量计算的连续性
        """
        data = {}
        col_map = self._get_required_column_map()
        for key, col_name in col_map.items():
            series = self.helper._get_safe_series(df, col_name, 0.0)
            data[key] = series.astype(np.float32)
        return data

    def _get_required_column_map(self) -> Dict[str, str]:
        """
        【V48.0】数据映射重构：完整继承V47.0，并引入军械库中的均线势能压缩率、换手率稳定性及推力高阶动力学。
        """
        return {
            'close': 'close_D',
            'cost_avg': 'cost_50pct_D',
            'sm_net_buy': 'SMART_MONEY_HM_NET_BUY_D',
            'hab_inventory': 'total_net_amount_21d_D',
            'sm_slope_13': 'SLOPE_13_SMART_MONEY_HM_NET_BUY_D',
            'sm_accel_13': 'ACCEL_13_SMART_MONEY_HM_NET_BUY_D',
            'sm_jerk_13': 'JERK_13_SMART_MONEY_HM_NET_BUY_D',
            'sm_synergy': 'SMART_MONEY_SYNERGY_BUY_D',
            'pushing_score': 'pushing_score_D',
            'market_sentiment': 'market_sentiment_score_D',
            'tick_large_net': 'tick_large_order_net_D',
            'intra_accel': 'flow_acceleration_intraday_D',
            'breakout_flow': 'breakout_fundflow_score_D',
            'mf_activity': 'intraday_main_force_activity_D',
            'energy_conc': 'energy_concentration_D',
            'winner_rate': 'winner_rate_D',
            'control_solidity': 'control_solidity_index_D',
            'chip_entropy': 'chip_entropy_D',
            'chip_stability': 'chip_stability_D',
            'peak_conc': 'peak_concentration_D',
            'accumulation_score': 'accumulation_signal_score_D',
            'ma_coherence': 'MA_COHERENCE_RESONANCE_D',
            'hab_structure': 'long_term_chip_ratio_D',
            'conc_slope': 'SLOPE_5_peak_concentration_D',
            'winner_accel': 'ACCEL_5_winner_rate_D',
            'platform_quality': 'consolidation_quality_score_D',
            'foundation_strength': 'support_strength_D',
            'vpa_efficiency': 'VPA_EFFICIENCY_D',
            'profit_pressure': 'profit_pressure_D',
            'turnover': 'turnover_rate_D',
            'trapped_pressure': 'pressure_trapped_D',
            'dist_score': 'distribution_score_D',
            'intraday_dist': 'intraday_distribution_confidence_D',
            'instability': 'VOLATILITY_INSTABILITY_INDEX_21d_D',
            'pressure_release': 'pressure_release_index_D',
            'shakeout_score': 'shakeout_score_D',
            'chip_divergence': 'chip_divergence_ratio_D',
            'dist_slope': 'SLOPE_5_distribution_score_D',
            'dist_accel': 'ACCEL_5_distribution_score_D',
            'dist_jerk': 'JERK_5_distribution_score_D',
            'gap_momentum': 'GAP_MOMENTUM_STRENGTH_D',
            'emotional_extreme': 'STATE_EMOTIONAL_EXTREME_D',
            'reversal_prob': 'reversal_prob_D',
            'is_leader': 'STATE_MARKET_LEADER_D',
            'theme_hotness': 'THEME_HOTNESS_SCORE_D',
            'lock_ratio': 'high_position_lock_ratio_90_D',
            'coordinated_attack': 'SMART_MONEY_HM_COORDINATED_ATTACK_D',
            'flow_21d': 'total_net_amount_21d_D',
            'flow_55d': 'total_net_amount_55d_D',
            'buy_elg_rate': 'buy_elg_amount_rate_D',
            'flow_consistency': 'flow_consistency_D',
            'flow_persistence': 'flow_persistence_minutes_D',
            'closing_intensity': 'closing_flow_intensity_D',
            'industry_markup': 'industry_markup_score_D',
            'tick_abnormal_vol': 'tick_abnormal_volume_ratio_D',
            'intra_acc_conf': 'intraday_accumulation_confidence_D',
            'tick_net_slope_13': 'SLOPE_13_tick_large_order_net_D',
            'tick_net_accel_13': 'ACCEL_13_tick_large_order_net_D',
            'tick_net_jerk_13': 'JERK_13_tick_large_order_net_D',
            'pushing_slope_13': 'SLOPE_13_pushing_score_D',
            'pushing_accel_13': 'ACCEL_13_pushing_score_D',
            'pushing_jerk_13': 'JERK_13_pushing_score_D',
            'chip_convergence': 'chip_convergence_ratio_D',
            'intra_consolidation': 'intraday_chip_consolidation_degree_D',
            'ma_tension': 'MA_POTENTIAL_TENSION_INDEX_D',
            'consolidation_chip_conc': 'consolidation_chip_concentration_D',
            'rounding_bottom': 'STATE_ROUNDING_BOTTOM_D',
            'sr_ratio': 'support_resistance_ratio_D',
            'ctrl_slope_13': 'SLOPE_13_control_solidity_index_D',
            'ctrl_accel_13': 'ACCEL_13_control_solidity_index_D',
            'ctrl_jerk_13': 'JERK_13_control_solidity_index_D',
            'outflow_qual': 'outflow_quality_D',
            'intra_skew': 'intraday_price_distribution_skewness_D',
            'ind_downtrend': 'industry_downtrend_score_D',
            'downtrend_str': 'downtrend_strength_D',
            'dist_energy': 'distribution_energy_D',
            'hf_flow_div': 'high_freq_flow_divergence_D',
            'dist_slope_13': 'SLOPE_13_distribution_score_D',
            'dist_accel_13': 'ACCEL_13_distribution_score_D',
            'dist_jerk_13': 'JERK_13_distribution_score_D',
            'game_intensity': 'game_intensity_D',
            'golden_pit': 'STATE_GOLDEN_PIT_D',
            'breakout_conf': 'STATE_BREAKOUT_CONFIRMED_D',
            'hm_top_tier': 'HM_ACTIVE_TOP_TIER_D',
            't1_premium': 'T1_PREMIUM_EXPECTATION_D',
            'breakout_pot': 'breakout_potential_D',
            'ma_compression': 'MA_POTENTIAL_COMPRESSION_RATE_D',
            'turnover_stability': 'TURNOVER_STABILITY_INDEX_D'
        }

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V38.0】全息张量计算执行器：统一字典传参，并升级张量合成签名为全量透传模式。
        """
        self._probe_cache = []
        raw = self._load_data(df)
        idx = df.index
        count = len(idx)
        if count < 5:
            print(f"[PROBE-FATAL] 数据行数不足5行，当前行数: {count}，直接阻断。")
            return pd.Series(0.0, index=idx)
        print(f"[PROBE-INFO] 开始执行全息推力计算(含HAB与Kinematics)，处理条目数: {count}")
        self._probe_cache_raw = raw
        self._probe_cache_idx = idx
        thrust = self._calc_thrust_component(raw, idx)
        structure = self._calc_structure_component(raw, idx)
        drag = self._calc_drag_component(raw, idx)
        raw_intent = self._calc_tensor_synthesis(thrust, structure, drag, raw, idx)
        med = np.median(raw_intent)
        mad = np.median(np.abs(raw_intent - med)) + 1e-9
        print(f"[PROBE-STAT] Raw Intent | Median: {med:.4f} | MAD: {mad:.4f}")
        z_scores = (raw_intent - med) / (mad * 3.0)
        final_scores = 1.0 / (1.0 + np.exp(-z_scores))
        if self._is_probe_enabled():
            self._generate_probe_report(idx, raw, thrust, structure, drag, raw_intent, final_scores)
        return pd.Series(final_scores, index=idx, dtype=np.float32)

    def _calc_thrust_component(self, raw: Dict[str, np.ndarray], idx: pd.Index) -> np.ndarray:
        """
        【V41.0】A股交叉耦合场推力模型 - 相位对齐版 
        核心改进：
        1. 引入交叉耦合：动力学因子现在直接干预宏观底座的方向，而非简单的缩放。
        2. 射流能量损耗：根据资金流一致性计算能量耗散熵，对无效射流进行惩罚。
        3. 相位对齐：个股推力与板块 Beta 场进行相位乘法，增强龙头识别度。
        """
        sm_net_buy = raw['sm_net_buy'].values
        sm_synergy = raw['sm_synergy'].values
        flow_21d = raw['flow_21d'].values
        flow_55d = raw['flow_55d'].values
        tick_large_net = raw['tick_large_net'].values
        intra_accel = raw['intra_accel'].values
        breakout_flow = raw['breakout_flow'].values
        pushing_score = raw['pushing_score'].values
        sentiment = raw['market_sentiment'].values
        mf_activity = raw['mf_activity'].values
        k_slope = raw['sm_slope_13'].values
        k_accel = raw['sm_accel_13'].values
        k_jerk = raw['sm_jerk_13'].values
        buy_elg_rate = raw['buy_elg_rate'].values
        flow_consistency = raw['flow_consistency'].values
        flow_persistence = raw['flow_persistence'].values
        closing_intensity = raw['closing_intensity'].values
        industry_markup = raw['industry_markup'].values
        tick_abnormal_vol = raw['tick_abnormal_vol'].values
        intra_acc_conf = raw['intra_acc_conf'].values
        tick_net_slope = raw['tick_net_slope_13'].values
        tick_net_accel = raw['tick_net_accel_13'].values
        tick_net_jerk = raw['tick_net_jerk_13'].values
        push_slope = raw['pushing_slope_13'].values
        push_accel = raw['pushing_accel_13'].values
        push_jerk = raw['pushing_jerk_13'].values
        hab_total_pool = (flow_21d * 0.6) + (flow_55d * 0.4)
        hab_cushion = np.where((sm_net_buy < 0) & (hab_total_pool > 0), np.clip(hab_total_pool / (np.abs(sm_net_buy) + 1e-9), 0.0, 1.0) * np.abs(sm_net_buy) * 0.8, 0.0)
        effective_net_buy = sm_net_buy + hab_cushion
        macro_base = effective_net_buy + (sm_synergy * 1.5)
        norm_macro_base = np.sign(macro_base) * np.log1p(np.abs(macro_base) / 1000000.0)
        macro_damping = np.tanh(np.abs(effective_net_buy) / 10000000.0)
        tick_damping = np.tanh(np.abs(tick_large_net) / 5000000.0)
        push_damping = (pushing_score - 50.0) / 50.0
        macro_kinematics = (k_slope + k_accel + k_jerk) * macro_damping
        tick_kinematics = (tick_net_slope + tick_net_accel + tick_net_jerk) * tick_damping
        push_kinematics = (push_slope + push_accel + push_jerk) * np.maximum(0, push_damping)
        coupling_field = np.tanh(macro_kinematics + tick_kinematics + push_kinematics)
        kinematic_multiplier = 1.0 + np.maximum(0.0, coupling_field)
        purity_multiplier = 1.0 + (np.tanh(buy_elg_rate * 5.0))
        acc_conf_norm = (intra_acc_conf - 50.0) / 50.0
        acc_confidence_multiplier = 1.0 + np.maximum(0.0, acc_conf_norm)
        macro_momentum = norm_macro_base * purity_multiplier * acc_confidence_multiplier * (1.0 + coupling_field * 0.5)
        persistence_factor = np.tanh(flow_persistence / 120.0)
        tick_intensity = tick_large_net / (np.abs(effective_net_buy) + 1e-9)
        detonation_boost = 1.0 + np.tanh(np.maximum(0, tick_abnormal_vol - 1.0))
        energy_dissipation = 1.0 - np.clip(flow_consistency, 0.0, 0.9)
        micro_jet_raw = (intra_accel * tick_intensity * (pushing_score / 100.0) * mf_activity * persistence_factor * detonation_boost) / (1.0 + energy_dissipation)
        jet_exponent = np.tanh(micro_jet_raw) * (breakout_flow / 50.0) * np.maximum(0.1, flow_consistency)
        micro_multiplier = np.exp(np.clip(jet_exponent, -2.0, 2.0))
        closing_amplifier = 1.0 + np.maximum(0.0, np.tanh(closing_intensity / 50.0))
        sentiment_amplifier = 1.0 + np.maximum(0.0, (sentiment - 50.0) / 50.0)
        industry_resonance = 1.0 + np.maximum(0.0, (industry_markup - 50.0) / 50.0)
        phase_alignment = np.where((macro_momentum > 0) & (industry_markup > 50), 1.2, 0.8)
        base_final_thrust = macro_momentum * micro_multiplier * kinematic_multiplier * sentiment_amplifier * closing_amplifier * industry_resonance * phase_alignment
        excess_kine = np.maximum(0.0, kinematic_multiplier - 1.0)
        excess_jet = np.maximum(0.0, micro_multiplier - 1.0)
        excess_beta = np.maximum(0.0, industry_resonance - 1.0)
        critical_resonance_index = excess_kine * excess_jet * excess_beta * acc_confidence_multiplier
        nonlinear_gain = 1.0 + np.expm1(np.clip(critical_resonance_index * 2.5, 0.0, 4.0))
        ultimate_thrust = base_final_thrust * nonlinear_gain
        if self._is_probe_enabled():
            target_dates = pd.to_datetime(self.probe_dates).tz_localize(None).normalize()
            current_dates = idx.tz_localize(None).normalize()
            locs = np.where(current_dates.isin(target_dates))[0]
            for i in locs:
                ts = idx[i].strftime('%Y-%m-%d')
                probe_log = [
                    f"\n[PROBE-THRUST-V41.0] 交叉耦合与相位对齐探针 @ {ts}",
                    f"  |- 耦合场 (Coupling Field):",
                    f"     Macro Momentum: {macro_momentum[i]:.4f} | Coupling Field: {coupling_field[i]:.4f}",
                    f"     Phase Alignment: {phase_alignment[i]:.2f} | Acc Conf Mult: {acc_confidence_multiplier[i]:.4f}",
                    f"  |- 能量损耗 (Energy Dissipation):",
                    f"     Flow Consistency: {flow_consistency[i]:.4f} | Dissipation Factor: {energy_dissipation[i]:.4f}",
                    f"     Micro Jet (Post-Dissipation): {micro_jet_raw[i]:.4f}",
                    f"  |- 临界共振 (Resonance):",
                    f"     CRI: {critical_resonance_index[i]:.6f} | Exponential Gain: x{nonlinear_gain[i]:.4f}",
                    f"  |- 终极输出 (Ultimate Tensor):",
                    f"     -> ULTIMATE THRUST: {ultimate_thrust[i]:.4f}\n"
                ]
                self._probe_cache.extend(probe_log)
                for line in probe_log: print(line)
        return ultimate_thrust

    def _calc_structure_component(self, raw: Dict[str, np.ndarray], idx: pd.Index) -> np.ndarray:
        """
        【V31.0】维度B：晶格相变结构场 (Lattice Phase Transition Structure) - 物理场交叉耦合版
        核心逻辑：
        1. 继承V30.0的全套非线性映射与雪崩增益架构。
        2. 【内部逻辑重构 1】熵稳动态平衡：引入 entropy_penalty，高稳态下对洗盘导致的筹码熵增进行宽容，防误杀。
        3. 【内部逻辑重构 2】弹性压缩度交叉张量：耦合均线张力(Tension)与筹码收敛度(Convergence)。
        4. 【内部逻辑重构 3】动力学耦合放大：将弹性压缩度作为杠杆，直接放大斜率、加速度与控盘动量，实现“压簧爆发”效应。
        """
        cost_avg = raw['cost_avg'].values
        close = raw['close'].values
        chip_entropy = raw['chip_entropy'].values
        chip_stability = raw['chip_stability'].values
        intra_consolidation = raw['intra_consolidation'].values
        ma_coherence = raw['ma_coherence'].values
        chip_convergence = raw['chip_convergence'].values
        ma_tension = raw['ma_tension'].values
        peak_conc = raw['peak_conc'].values
        winner_rate = raw['winner_rate'].values
        control_solidity = raw['control_solidity'].values
        accumulation_score = raw['accumulation_score'].values
        hab_structure = raw['hab_structure'].values
        platform_quality = raw['platform_quality'].values
        foundation_strength = raw['foundation_strength'].values
        conc_slope = raw['conc_slope'].values
        winner_accel = raw['winner_accel'].values
        consolidation_chip_conc = raw['consolidation_chip_conc'].values
        rounding_bottom = raw['rounding_bottom'].values
        sr_ratio = raw['sr_ratio'].values
        flow_21d = raw['flow_21d'].values
        flow_55d = raw['flow_55d'].values
        ctrl_slope_13 = raw['ctrl_slope_13'].values
        ctrl_accel_13 = raw['ctrl_accel_13'].values
        ctrl_jerk_13 = raw['ctrl_jerk_13'].values
        cost_gap = (close - cost_avg) / (cost_avg + 1e-9)
        cost_rbf = np.exp(-10.0 * (cost_gap - 0.05)**2)
        entropy_raw = np.maximum(0.01, chip_entropy)
        norm_intra_consolidation = 1.0 / (1.0 + np.exp(-0.05 * (intra_consolidation - 50.0)))
        stability_raw = np.maximum(0.0, chip_stability) + norm_intra_consolidation * 0.5
        entropy_penalty = entropy_raw / (1.0 + stability_raw)
        norm_coherence = 1.0 / (1.0 + np.exp(-0.1 * (ma_coherence - 50.0)))
        lattice_orderliness = (stability_raw * np.maximum(0.1, norm_coherence)) / np.maximum(0.01, entropy_penalty)
        norm_convergence = np.tanh(np.maximum(0.0, chip_convergence) / 50.0)
        convergence_factor = 1.0 + norm_convergence
        norm_tension = 1.0 / (1.0 + np.exp(-0.05 * (ma_tension - 50.0)))
        elastic_compression = np.maximum(0.0, norm_tension * norm_convergence)
        norm_peak_conc = 1.0 / (1.0 + np.exp(-0.1 * (peak_conc - 50.0)))
        peak_efficiency = norm_peak_conc * winner_rate * convergence_factor
        norm_control = np.tanh(control_solidity / 50.0)
        control_factor = 1.0 + norm_control * 0.5
        norm_acc = 1.0 / (1.0 + np.exp(-0.05 * (accumulation_score - 50.0)))
        acc_factor = 1.0 + norm_acc
        static_lattice_energy = lattice_orderliness * peak_efficiency * cost_rbf * control_factor * acc_factor
        hab_pool = flow_21d * 0.618 + flow_55d * 0.382
        hab_immunity = 1.0 - (1.0 / (1.0 + np.exp(hab_pool / 50000000.0)))
        hab_immunity = np.maximum(0.0, np.minimum(hab_immunity, 0.85))
        ctrl_damping = np.abs(np.tanh(control_solidity / 20.0))
        raw_ctrl_kine = ctrl_slope_13 + ctrl_accel_13 + ctrl_jerk_13
        protected_ctrl_kine = np.where(raw_ctrl_kine < 0, raw_ctrl_kine * (1.0 - hab_immunity), raw_ctrl_kine)
        effective_ctrl_kine = protected_ctrl_kine * ctrl_damping
        k_conc_slope = np.tanh(conc_slope * 2.0)
        k_winner_accel = np.tanh(winner_accel * 1.5)
        kine_vector = (k_conc_slope * 0.2) + (k_winner_accel * 0.15) + (np.tanh(effective_ctrl_kine) * 0.35)
        evolution_kinematics = 1.0 + kine_vector * (1.0 + elastic_compression * 2.0)
        norm_consolidation_conc = 1.0 / (1.0 + np.exp(-0.1 * (consolidation_chip_conc - 50.0)))
        consolidation_boost = 1.0 + norm_consolidation_conc
        norm_platform = 1.0 / (1.0 + np.exp(-0.1 * (platform_quality - 50.0)))
        platform_factor = 1.0 + norm_platform * 0.6 * consolidation_boost
        sr_factor = np.exp(np.tanh(sr_ratio - 1.0))
        norm_foundation = 1.0 / (1.0 + np.exp(-0.1 * (foundation_strength - 50.0)))
        foundation_factor = 1.0 + norm_foundation * 0.4 * sr_factor
        pattern_bonus = 1.0 + (rounding_bottom * 0.3)
        base_structure = static_lattice_energy * inertia_bonus * evolution_kinematics * platform_factor * foundation_factor * pattern_bonus
        sri = (lattice_orderliness * norm_platform * hab_structure * np.maximum(0.1, norm_tension))
        excitation_gain = 1.0 + np.maximum(0.0, np.expm1(sri - 0.5)) * 1.5
        resonance_core = base_structure * excitation_gain
        avalanche_threshold = 1.5
        avalanche_gain = 1.0 + (np.maximum(0.0, resonance_core - avalanche_threshold) ** 2) * 2.0
        final_structure = resonance_core * avalanche_gain
        if self._is_probe_enabled():
            target_dates = pd.to_datetime(self.probe_dates).tz_localize(None).normalize()
            current_dates = idx.tz_localize(None).normalize()
            locs = np.where(current_dates.isin(target_dates))[0]
            for i in locs:
                ts = idx[i].strftime('%Y-%m-%d')
                probe_log = [
                    f"\n[PROBE-STRUCTURE-V31.0] 晶格相变全息审计(物理场交叉耦合版) @ {ts}",
                    f"  |- 熵稳平衡机制 (Entropy-Stability):",
                    f"     Raw Entropy: {entropy_raw[i]:.4f} | Stability: {stability_raw[i]:.4f}",
                    f"     Entropy Penalty: {entropy_penalty[i]:.4f} -> Lattice Orderliness: {lattice_orderliness[i]:.4f}",
                    f"  |- 弹性交叉张量 (Elastic Compression):",
                    f"     Tension (Norm): {norm_tension[i]:.4f} * Convergence (Norm): {norm_convergence[i]:.4f}",
                    f"     -> Elastic Compression Ratio: {elastic_compression[i]:.4f}",
                    f"  |- 动量耦合放大 (Kinematic Amplification):",
                    f"     Raw Kine Vector: {kine_vector[i]:.4f} | Amplifier (1 + Elastic*2): {(1.0 + elastic_compression[i]*2.0):.4f}",
                    f"     -> Evolution Kinematics: {evolution_kinematics[i]:.4f}",
                    f"  |- 历史存量防守 (HAB Immunity):",
                    f"     HAB Pool: {hab_pool[i]:.0f} -> Immunity Cushion: {hab_immunity[i]*100:.2f}%",
                    f"  |- 共振与雪崩增益 (Resonance & Avalanche):",
                    f"     SRI Index: {sri[i]:.6f} -> Excitation Gain: x{excitation_gain[i]:.4f}",
                    f"     Resonance Core: {resonance_core[i]:.4f} | Avalanche Multiplier: x{avalanche_gain[i]:.4f}",
                    f"  |- 终极结构张量 (Final Structure):",
                    f"     Base Structure: {base_structure[i]:.4f} -> FINAL: {final_structure[i]:.4f}\n"
                ]
                self._probe_cache.extend(probe_log)
                for line in probe_log: print(line)
        return final_structure

    def _calc_drag_component(self, raw: Dict[str, np.ndarray], idx: pd.Index) -> np.ndarray:
        """
        【V36.0】维度C：非线性临界阻力模型 (Stampede Blackhole Resistance) - 六步全息交叉版
        核心逻辑：
        1. 军械库引入：合并派发能量与高频资金背离特征，精准评估真实抛压质地。
        2. 动力学去噪：通过13日派发分值的Slope/Accel/Jerk构建动量传导，应用Tanh阻尼过滤零基陷阱噪音。
        3. HAB存量意识：组合21日与55日资金净额构建蓄水池，为偶然的微观流出提供最高90%的拖拽免疫；为长期流出施加额外惩罚。
        4. 原生映射：废弃clip，套牢盘使用Exp-Tanh进行无限拉伸，派发特征使用Sigmoid平滑过渡临界跃迁。
        5. 交叉耦合：主动倾泻核(质量×能量×动力学) 与 环境粘滞核(不稳定×效率衰减×Beta逆风) 发生场间乘法共振。
        6. 雪崩增益：核心阻力中枢一旦超越阈值，引发二次幂雪崩指数膨胀，模拟A股流动性衰竭与连环踩踏。
        """
        profit_pressure = raw['profit_pressure'].values
        trapped_pressure = raw['trapped_pressure'].values
        dist_score = raw['dist_score'].values
        intraday_dist = raw['intraday_dist'].values
        instability = raw['instability'].values
        vpa_efficiency = raw['vpa_efficiency'].values
        turnover_rate = raw['turnover'].values
        pressure_release = raw['pressure_release'].values
        shakeout_score = raw['shakeout_score'].values
        outflow_qual = raw['outflow_qual'].values
        intra_skew = raw['intra_skew'].values
        ind_downtrend = raw['ind_downtrend'].values
        downtrend_str = raw['downtrend_str'].values
        dist_energy = raw['dist_energy'].values
        hf_flow_div = raw['hf_flow_div'].values
        dist_slope_13 = raw['dist_slope_13'].values
        dist_accel_13 = raw['dist_accel_13'].values
        dist_jerk_13 = raw['dist_jerk_13'].values
        flow_21d = raw['flow_21d'].values
        flow_55d = raw['flow_55d'].values
        dist_damping = np.abs(np.tanh(dist_score / 20.0))
        raw_dist_kine = dist_slope_13 + dist_accel_13 + dist_jerk_13
        effective_dist_kine = raw_dist_kine * dist_damping
        kine_multiplier = 1.0 + np.maximum(0.0, np.tanh(effective_dist_kine) * 1.5)
        hab_pool = flow_21d * 0.618 + flow_55d * 0.382
        hab_immunity = 1.0 - (1.0 / (1.0 + np.exp(hab_pool / 50000000.0)))
        hab_immunity = np.clip(hab_immunity, 0.0, 0.9)
        hab_burden = np.maximum(0.0, -hab_pool) / 50000000.0
        hab_drag_penalty = 1.0 + np.tanh(hab_burden)
        norm_profit_pressure = np.expm1(np.maximum(0.0, profit_pressure) / 50.0)
        norm_trapped_pressure = np.expm1(np.maximum(0.0, trapped_pressure) / 50.0) * 1.5
        norm_dist = 1.0 / (1.0 + np.exp(-0.1 * (dist_score - 50.0)))
        norm_intra_dist = 1.0 / (1.0 + np.exp(-0.1 * (intraday_dist - 50.0)))
        norm_instability = 1.0 / (1.0 + np.exp(-0.05 * (instability - 50.0)))
        norm_downtrend = 1.0 / (1.0 + np.exp(-0.1 * (downtrend_str - 50.0)))
        dump_quality_factor = 1.0 + np.maximum(0.0, outflow_qual / 100.0) * 1.5
        energy_factor = 1.0 + np.maximum(0.0, dist_energy / 100.0)
        coupled_active_dump = (norm_dist + norm_intra_dist * 0.5) * dump_quality_factor * energy_factor * kine_multiplier
        beta_headwind = 1.0 + np.maximum(0.0, ind_downtrend / 100.0)
        friction_vpa = 1.0 + np.maximum(0.0, 1.0 - (vpa_efficiency / 100.0))
        skew_penalty = 1.0 + np.maximum(0.0, -intra_skew) * 0.5
        coupled_viscosity = (1.0 + norm_instability) * beta_headwind * friction_vpa * skew_penalty
        coupled_gravity = (norm_profit_pressure + norm_trapped_pressure) * (1.0 + norm_downtrend)
        norm_release = 1.0 / (1.0 + np.exp(-0.05 * (pressure_release - 50.0)))
        norm_shakeout = 1.0 / (1.0 + np.exp(-0.05 * (shakeout_score - 50.0)))
        relief_valve = 1.0 + norm_release * 1.5 + norm_shakeout * 1.0
        hf_hidden_div = np.maximum(0.0, hf_flow_div / 50.0)
        turnover_drag = np.expm1(np.maximum(0.0, turnover_rate - 0.05) * 10.0) * 0.5
        core_drag_raw = ((coupled_gravity + coupled_active_dump) * coupled_viscosity * hab_drag_penalty) / relief_valve
        core_drag_shielded = core_drag_raw * (1.0 - hab_immunity) + turnover_drag + hf_hidden_div
        avalanche_threshold = 1.5
        avalanche_gain = 1.0 + (np.maximum(0.0, core_drag_shielded - avalanche_threshold) ** 2) * 2.5
        final_drag = core_drag_shielded * avalanche_gain
        if self._is_probe_enabled():
            target_dates = pd.to_datetime(self.probe_dates).tz_localize(None).normalize()
            current_dates = idx.tz_localize(None).normalize()
            locs = np.where(current_dates.isin(target_dates))[0]
            for i in locs:
                ts = idx[i].strftime('%Y-%m-%d')
                probe_log = [
                    f"\n[PROBE-DRAG-V36.0] 踩踏黑洞共振全息审计(六步交叉版) @ {ts}",
                    f"  |- 动力学抗噪传导 (Kinematic Denoising):",
                    f"     Raw Kine (S+A+J): {raw_dist_kine[i]:.4f} | Damping: {dist_damping[i]:.4f}",
                    f"     Effective Kine: {effective_dist_kine[i]:.4f} -> Kine Multiplier: x{kine_multiplier[i]:.4f}",
                    f"  |- HAB存量护城河 (Historical Accumulation Buffer):",
                    f"     HAB Pool: {hab_pool[i]:.0f} -> Immunity Shield: {hab_immunity[i]*100:.2f}% | Burden Penalty: x{hab_drag_penalty[i]:.4f}",
                    f"  |- 主动倾泻张量 (Coupled Active Dump):",
                    f"     Dist Norm: {norm_dist[i]:.4f} | Quality Factor: x{dump_quality_factor[i]:.4f} | Energy Factor: x{energy_factor[i]:.4f}",
                    f"     -> Final Active Dump: {coupled_active_dump[i]:.4f}",
                    f"  |- 环境粘滞张量 (Coupled Viscosity):",
                    f"     Instability: {norm_instability[i]:.4f} | VPA Friction: {friction_vpa[i]:.4f}",
                    f"     Beta Headwind: {beta_headwind[i]:.4f} | Skew Penalty: {skew_penalty[i]:.4f}",
                    f"     -> Final Viscosity: {coupled_viscosity[i]:.4f}",
                    f"  |- 核心中枢与泄压 (Core Synthesis):",
                    f"     Coupled Gravity: {coupled_gravity[i]:.4f} | Relief Valve: /{relief_valve[i]:.4f}",
                    f"     Raw Core Drag: {core_drag_raw[i]:.4f} -> Shielded Core: {core_drag_shielded[i]:.4f}",
                    f"  |- 踩踏黑洞雪崩增益 (Avalanche Non-linear Gain):",
                    f"     Threshold: {avalanche_threshold} | Core Excess: {max(0, core_drag_shielded[i] - avalanche_threshold):.4f}",
                    f"     Avalanche Multiplier: x{avalanche_gain[i]:.4f}",
                    f"  |- 终极阻力张量 (Final Drag Tensor):",
                    f"     -> FINAL DRAG: {final_drag[i]:.4f}\n"
                ]
                self._probe_cache.extend(probe_log)
                for line in probe_log: print(line)
        return final_drag

    def _calc_tensor_synthesis(self, thrust: np.ndarray, structure: np.ndarray, drag: np.ndarray, raw: Dict[str, np.ndarray], idx: pd.Index) -> np.ndarray:
        """
        【V38.0】张量合成 - 全息奇点共振与A股生态博弈模型 (多维动力学交叉耦合版)
        核心逻辑：
        1. 高维弹药：融入均线压缩率(ma_compression)与换手率稳定性(turnover_stability)。
        2. 全息动力学与抗噪：组合资金动力学与推力评分动力学，双重阻尼(Damping)完美过滤微波动噪音，破解“零基陷阱”。
        3. HAB存量底座：21日与55日资金蓄水池不仅免疫拖拽，更化作奇点爆发的底仓燃料。
        4. 原生拓扑映射：Sigmoid处理换手稳定性，Tanh处理均线压缩，Exp-Tanh拉升T+1溢价。
        5. 黄金坑轧空涡流：高博弈烈度遭遇黄金坑时，有效阻力反转为轧空势能。
        6. 奇点共振增益(Singularity Gain)：当全息共振指数(HRI)跨越临界，且均线被极度压缩时，触发大爆炸非线性跃迁。
        """
        sm_net_buy = raw['sm_net_buy'].values
        pushing_score = raw['pushing_score'].values
        energy_conc = raw['energy_conc'].values
        sm_slope_13 = raw['sm_slope_13'].values
        sm_accel_13 = raw['sm_accel_13'].values
        sm_jerk_13 = raw['sm_jerk_13'].values
        push_slope_13 = raw['pushing_slope_13'].values
        push_accel_13 = raw['pushing_accel_13'].values
        push_jerk_13 = raw['pushing_jerk_13'].values
        flow_21d = raw['flow_21d'].values
        flow_55d = raw['flow_55d'].values
        theme_hotness = raw['theme_hotness'].values
        is_leader = raw['is_leader'].values
        gap_momentum = raw['gap_momentum'].values
        reversal_prob = raw['reversal_prob'].values
        lock_ratio = raw['lock_ratio'].values
        coordinated_attack = raw['coordinated_attack'].values
        emotional_extreme = raw['emotional_extreme'].values
        game_intensity = raw['game_intensity'].values
        golden_pit = raw['golden_pit'].values
        breakout_conf = raw['breakout_conf'].values
        hm_top_tier = raw['hm_top_tier'].values
        t1_premium = raw['t1_premium'].values
        breakout_pot = raw['breakout_pot'].values
        ma_compression = raw['ma_compression'].values
        turnover_stability = raw['turnover_stability'].values
        norm_push = 1.0 / (1.0 + np.exp(-0.1 * (pushing_score - 50.0)))
        kine_damping = np.tanh(np.abs(sm_net_buy) / 10000000.0) * norm_push
        k_sm = np.tanh(sm_slope_13) * 0.3 + np.tanh(sm_accel_13) * 0.3 + np.tanh(sm_jerk_13) * 0.4
        k_push = np.tanh(push_slope_13) * 0.3 + np.tanh(push_accel_13) * 0.3 + np.tanh(push_jerk_13) * 0.4
        kinematic_burst = 1.0 + np.maximum(0.0, (k_sm + k_push * 0.5) * kine_damping)
        combined_inventory = (flow_21d * 0.618) + (flow_55d * 0.382)
        hab_immunity = 1.0 - (1.0 / (1.0 + np.exp(combined_inventory / 50000000.0)))
        hab_fuel = np.maximum(0.0, np.tanh(combined_inventory / 100000000.0))
        norm_theme = 1.0 / (1.0 + np.exp(-0.05 * (theme_hotness - 50.0)))
        norm_breakout_pot = 1.0 / (1.0 + np.exp(-0.05 * (breakout_pot - 50.0)))
        norm_turnover_stab = 1.0 / (1.0 + np.exp(-0.05 * (turnover_stability - 50.0)))
        eco_premium = 1.0 + (is_leader * 0.8) + (hm_top_tier * 0.6) + (breakout_conf * 0.4) + (norm_theme * 0.3) + coordinated_attack * 0.5
        base_tensor = thrust * structure * (1.0 + gap_momentum) * eco_premium * kinematic_burst * (1.0 + norm_breakout_pot) * (1.0 + norm_turnover_stab * 0.5)
        norm_lock_ratio = 1.0 / (1.0 + np.exp(-0.1 * (lock_ratio - 50.0)))
        raw_effective_drag = drag * (1.0 - np.maximum(0.0, np.minimum(hab_immunity, 0.90)))
        squeeze_transition = 1.0 / (1.0 + np.exp(-2.0 * (base_tensor - 1.5 * raw_effective_drag)))
        norm_game_intensity = 1.0 / (1.0 + np.exp(-0.1 * (game_intensity - 50.0)))
        trap_reversal_factor = 1.0 + (golden_pit * 2.0)
        norm_energy = 1.0 / (1.0 + np.exp(-0.05 * (energy_conc - 50.0)))
        squeeze_bonus = squeeze_transition * raw_effective_drag * emotional_extreme * norm_game_intensity * kinematic_burst * trap_reversal_factor * norm_energy
        final_drag = (raw_effective_drag * raw_effective_drag) * (1.0 - squeeze_transition) * (1.0 - reversal_prob) * (1.0 - norm_lock_ratio * 0.5)
        raw_intent = (base_tensor / (1.0 + final_drag)) + squeeze_bonus
        t1_multiplier = np.exp(np.tanh((t1_premium - 50.0) / 20.0))
        norm_compression = np.tanh(np.maximum(0.0, ma_compression) / 50.0)
        hri = (base_tensor * (1.0 + squeeze_bonus)) / (1.0 + final_drag)
        hri_threshold = 3.0
        hri_excess = np.maximum(0.0, hri - hri_threshold)
        singularity_gain = 1.0 + np.expm1(hri_excess * t1_multiplier * (1.0 + norm_compression + hab_fuel))
        final_intent = raw_intent * singularity_gain
        if self._is_probe_enabled():
            target_dates = pd.to_datetime(self.probe_dates).tz_localize(None).normalize()
            current_dates = idx.tz_localize(None).normalize()
            locs = np.where(current_dates.isin(target_dates))[0]
            for i in locs:
                ts = idx[i].strftime('%Y-%m-%d')
                probe_log = [
                    f"\n[PROBE-SYNTHESIS-V38.0] 张量奇点共振全息探针(高维耦合爆破版) @ {ts}",
                    f"  |- 多维动力学与抗噪 (Multi-Kinematics):",
                    f"     SM Kine: {k_sm[i]:.4f} | Push Kine: {k_push[i]:.4f} | Damping: {kine_damping[i]:.4f}",
                    f"     Kinematic Burst Multiplier: x{kinematic_burst[i]:.4f}",
                    f"  |- 生态溢价与锁仓 (Ecosystem & Lock-in):",
                    f"     Leader: {is_leader[i]:.0f} | HM Top Tier: {hm_top_tier[i]:.0f} | Breakout Conf: {breakout_conf[i]:.0f}",
                    f"     Eco Premium: x{eco_premium[i]:.4f} | Turnover Stab Norm: {norm_turnover_stab[i]:.4f}",
                    f"  |- HAB 蓄水池防御与燃烧 (HAB Defense & Fuel):",
                    f"     HAB Pool: {combined_inventory[i]:.0f} -> Immunity Shield: {hab_immunity[i]*100:.2f}%",
                    f"     Drag Extinguishment: {drag[i]:.4f} -> Effective Drag: {raw_effective_drag[i]:.4f}",
                    f"  |- 黄金坑轧空涡流 (Golden Pit & Squeeze):",
                    f"     Squeeze Transition: {squeeze_transition[i]:.4f} | Game Intensity Norm: {norm_game_intensity[i]:.4f}",
                    f"     Golden Pit: {golden_pit[i]:.0f} -> Trap Reversal Factor: {trap_reversal_factor[i]:.4f}",
                    f"     Squeeze Bonus (Vortex Fuel): +{squeeze_bonus[i]:.4f} | Final Suppressed Drag: {final_drag[i]:.4f}",
                    f"  |- 奇点爆炸增益 (Singularity Resonance Gain):",
                    f"     HRI Index: {hri[i]:.6f} (Threshold: {hri_threshold}) | HRI Excess: {hri_excess[i]:.4f}",
                    f"     T+1 Premium Multiplier: x{t1_multiplier[i]:.4f} | MA Compression Tanh: {norm_compression[i]:.4f}",
                    f"     HAB Base Fuel: {hab_fuel[i]:.4f} -> Singularity Multiplier: x{singularity_gain[i]:.4f}",
                    f"  |- 终极意图张量 (Final Intent Tensor):",
                    f"     Raw Intent: {raw_intent[i]:.4f} -> FINAL SYNTHESIS: {final_intent[i]:.4f}\n"
                ]
                self._probe_cache.extend(probe_log)
                for line in probe_log: print(line)
        return final_intent

    def _is_probe_enabled(self) -> bool:
        return get_param_value(self.debug_params.get('enabled'), False) and \
               get_param_value(self.debug_params.get('should_probe'), False)

    def _generate_probe_report(self, idx, raw, thrust, structure, drag, raw_intent, final):
        """
        【V33.1】探针终极升级：修复动力学因子引用错误，全息共振、HAB存量与动力学突变透视
        不再只是输出结果，而是通过“反向推演”展示每一个关键物理量对最终意图的贡献。修复了13日加速度特征键名引用导致的KeyError。
        """
        if not self.probe_dates: return
        target_dates = pd.to_datetime(self.probe_dates).tz_localize(None).normalize()
        current_dates = idx.tz_localize(None).normalize()
        locs = np.where(current_dates.isin(target_dates))[0]
        if len(locs) == 0: locs = [-1]
        for i in locs:
            ts = idx[i]
            net_buy = raw['sm_net_buy'].values[i]
            energy_damping = np.tanh(np.abs(net_buy) / 10000000.0) * np.clip(raw['energy_conc'].values[i] / 100.0, 0.0, 1.0)
            k_burst = 1.0 + max(0.0, (np.tanh(raw['sm_slope_13'].values[i]) * 0.3 + np.tanh(raw['sm_accel_13'].values[i]) * 0.3 + np.tanh(raw['sm_jerk_13'].values[i]) * 0.4) * energy_damping)
            comb_inv = (raw['flow_21d'].values[i] * 0.6) + (raw['flow_55d'].values[i] * 0.4)
            hab_imm = np.clip(1.0 - (1.0 / (1.0 + np.exp(comb_inv / 50000000.0))), 0.0, 0.9)
            eff_drag = drag[i] * (1.0 - hab_imm)
            hri = (thrust[i] * structure[i] * (1.0 + raw['gap_momentum'].values[i]) * (1.0 + (raw['is_leader'].values[i]*0.5)) * k_burst) / (1.0 + eff_drag)
            res_gain = 1.0 + np.expm1(np.clip(hri - 3.0, 0.0, 2.5) * 1.5)
            report = [
                f"\n=== [PROBE V33.1] CalculateMainForceRallyIntent Holographic Resonance Audit @ {ts.strftime('%Y-%m-%d')} ===",
                f"【A. Kinematics (动力学)】 Burst: x{k_burst:.4f} | Damping: {energy_damping:.4f} | Jerk: {raw['sm_jerk_13'].values[i]:.2f}",
                f"【B. HAB (存量意识)】 21d/55d Inv: {raw['flow_21d'].values[i]:.0f}/{raw['flow_55d'].values[i]:.0f} | Immunity: {hab_imm*100:.1f}%",
                f"【C. Ecosystem (生态)】 Leader: {raw['is_leader'].values[i]} | LockRatio: {raw['lock_ratio'].values[i]:.2f}% | Attack: {raw['coordinated_attack'].values[i]}",
                f"【D. Resonance (共振)】 HRI: {hri:.4f} (Threshold: 3.0) -> Resonance Multiplier: x{res_gain:.4f}",
                f"【E. Synthesis (合成)】 Thrust: {thrust[i]:.4f} | Structure: {structure[i]:.4f} | EffectiveDrag: {eff_drag:.4f}",
                f"【F. Result (最终)】 Raw Intent: {raw_intent[i]:.4f} | Final Normalized Score: {final[i]:.4f}",
                f"===============================================================\n"
            ]
            self._probe_cache.extend(report)
            for line in report: print(line)
















