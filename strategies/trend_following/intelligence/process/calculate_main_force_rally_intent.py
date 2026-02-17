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

    def _get_neutral_defaults(self) -> Dict[str, float]:
        """
        【V18.0 · 绝对均衡态注入与量纲纠偏】
        针对不同因子的真实物理边界，精细化定义绝对中性填充值，杜绝量纲错位。
        """
        defaults = {k: 0.0 for k in self._get_required_column_map().keys()}
        score_keys = [
            'pushing_score', 'winner_rate', 'peak_conc', 'accumulation_score', 
            'platform_quality', 'dist_score', 'instability', 'pressure_release', 
            'shakeout_score', 'theme_hotness', 'lock_ratio', 'consolidation_chip_conc', 
            'downtrend_str', 't1_premium', 'industry_markup', 'breakout_flow', 
            'flow_consistency', 'closing_intensity'
        ]
        for k in score_keys:
            defaults[k] = 50.0
        defaults['mf_activity'] = 0.5
        defaults['intra_consolidation'] = 0.5
        defaults['ma_coherence'] = 0.0
        defaults['ma_tension'] = 0.0
        defaults['control_solidity'] = 0.0
        defaults['foundation_strength'] = 0.5
        defaults['intra_acc_conf'] = 0.5
        defaults['intraday_dist'] = 0.0
        defaults['ind_downtrend'] = 0.0
        defaults['game_intensity'] = 0.5
        defaults['energy_conc'] = 0.5
        defaults['ma_compression'] = 0.0
        defaults['trend_confirm'] = 0.0
        defaults['breakout_pot'] = 0.0
        defaults['turnover_stability'] = 0.5
        defaults['chip_convergence'] = 0.0
        defaults['chip_stability'] = 0.0
        defaults['market_sentiment'] = 50.0
        defaults['hab_structure'] = 0.6
        defaults['tick_abnormal_vol'] = 1.0
        defaults['sr_ratio'] = 1.0
        defaults['chip_entropy'] = 1.0
        defaults['vpa_efficiency'] = 100.0
        defaults['turnover'] = 5.0
        defaults['close'] = 1.0
        defaults['cost_avg'] = 1.0
        return defaults

    def _load_data(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V18.0 · 物理中立基底注入版】
        取代暴力的 fillna(0.0)，赋予缺失特征平滑的隐身态，消除 NaN 级联破坏。
        """
        data = {}
        col_map = self._get_required_column_map()
        defaults = self._get_neutral_defaults()
        for key, col_name in col_map.items():
            series = self.helper._get_safe_series(df, col_name, np.nan)
            if key in ['close', 'cost_avg']:
                pass
            else:
                series = series.fillna(defaults.get(key, 0.0))
            data[key] = series.astype(np.float32)
        if 'close' in data:
            data['close'] = data['close'].ffill().bfill().fillna(1.0)
        if 'cost_avg' in data and 'close' in data:
            data['cost_avg'] = data['cost_avg'].replace(0.0, np.nan).fillna(data['close'])
        return data

    def _get_required_column_map(self) -> Dict[str, str]:
        """
        【V50.0】数据映射重构：剔除聪明钱依赖版
        1. 废弃所有 SMART_MONEY_* 相关指标。
        2. 采用 net_mf_amount_D (主力净额) 替代聪明钱净买入。
        3. 采用 HM_COORDINATED_ATTACK_D (高阶协同攻击) 替代聪明钱协同。
        4. 采用 trend_confirmation_score_D (趋势确认评分) 替代聪明钱协同攻击。
        """
        return {
            'close': 'close_D',
            'cost_avg': 'cost_50pct_D',
            'mf_net_buy': 'net_mf_amount_D',
            'hab_inventory': 'total_net_amount_21d_D',
            'mf_slope_13': 'SLOPE_13_net_mf_amount_D',
            'mf_accel_13': 'ACCEL_13_net_mf_amount_D',
            'mf_jerk_13': 'JERK_13_net_mf_amount_D',
            'hm_synergy': 'HM_COORDINATED_ATTACK_D',
            'pushing_score': 'pushing_score_D',
            'market_sentiment': 'market_sentiment_score_D',
            'tick_large_net': 'tick_large_order_net_D',
            'intra_accel': 'flow_acceleration_intraday_D',
            'breakout_flow': 'breakout_fundflow_score_D',
            'mf_activity': 'intraday_main_force_activity_D',
            'energy_conc': 'energy_concentration_D',
            'winner_rate': 'winner_rate_D',
            'control_solidity': 'consolidation_chip_stability_D',
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
            'instability': 'flow_volatility_21d_D',
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
            'trend_confirm': 'trend_confirmation_score_D',
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
            'ctrl_slope_13': 'SLOPE_13_consolidation_chip_stability_D',
            'ctrl_accel_13': 'ACCEL_13_consolidation_chip_stability_D',
            'ctrl_jerk_13': 'JERK_13_consolidation_chip_stability_D',
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

    def _get_probe_locs(self, idx: pd.Index, target_tensor: np.ndarray = None) -> List[int]:
        """
        【V16.0 · 全息探针寻址器】
        自动捕获计算过程中产生 NaN/Inf 的病态节点，并混合首尾及指定日期的索引。
        """
        locs = set()
        if self.probe_dates:
            target_dates = pd.to_datetime(self.probe_dates).tz_localize(None).normalize()
            current_dates = idx.tz_localize(None).normalize()
            matched = np.where(current_dates.isin(target_dates))[0]
            locs.update(matched.tolist())
        if target_tensor is not None:
            abnormal = np.where(np.isnan(target_tensor) | np.isinf(target_tensor))[0]
            locs.update(abnormal.tolist()[:3])
            valid_mask = ~(np.isnan(target_tensor) | np.isinf(target_tensor))
            if np.any(valid_mask):
                valid_idx = np.where(valid_mask)[0]
                locs.add(valid_idx[np.argmax(target_tensor[valid_mask])])
                locs.add(valid_idx[np.argmin(target_tensor[valid_mask])])
        count = len(idx)
        if count > 0:
            locs.update([0, count - 1])
        return sorted(list(locs))

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V18.0 · 统计重整执行器】
        通过滤除填0基底造成的死水平原，寻找真实的波动中枢，修正极化畸变。
        """
        self._probe_cache = []
        raw = self._load_data(df)
        idx = df.index
        count = len(idx)
        if count < 5:
            print(f"[PROBE-FATAL] 数据行数不足5行，当前行数: {count}，直接阻断。")
            return pd.Series(0.0, index=idx)
        self._probe_cache_raw = raw
        self._probe_cache_idx = idx
        thrust = self._calc_thrust_component(raw, idx)
        structure = self._calc_structure_component(raw, idx)
        drag = self._calc_drag_component(raw, idx)
        raw_intent = self._calc_tensor_synthesis(thrust, structure, drag, raw, idx)
        raw_intent_clean = np.clip(np.nan_to_num(raw_intent, nan=0.0, posinf=1000.0, neginf=-1000.0), -1000.0, 1000.0)
        valid_mask = np.abs(raw_intent_clean) > 1e-4
        valid_intent = raw_intent_clean[valid_mask]
        if len(valid_intent) > 5:
            med = np.median(valid_intent)
            mad = np.median(np.abs(valid_intent - med))
        else:
            med = np.median(raw_intent_clean)
            mad = np.median(np.abs(raw_intent_clean - med))
        robust_mad = np.maximum(mad, 0.1)
        z_scores = (raw_intent_clean - med) / (robust_mad * 3.0)
        final_scores = 1.0 / (1.0 + np.exp(np.clip(-z_scores, -20.0, 20.0)))
        if self._is_probe_enabled():
            print(f"[PROBE-STAT-V18.0] Intent | Valid(>0): {len(valid_intent)} | Median: {med:.4f} | Raw MAD: {mad:.8f} | Robust MAD: {robust_mad:.4f}")
            self._generate_probe_report(idx, raw, thrust, structure, drag, raw_intent_clean, final_scores)
        return pd.Series(final_scores, index=idx, dtype=np.float32)

    def _calc_thrust_component(self, raw: Dict[str, np.ndarray], idx: pd.Index) -> np.ndarray:
        """
        【V18.0 · 推力微观对齐纠偏版】
        修复了微观方向“负负得正”导致的方向失真；同步所有域激活量纲。
        """
        mf_net_buy = raw['mf_net_buy'].values
        hm_synergy = raw['hm_synergy'].values
        flow_21d = raw['flow_21d'].values
        flow_55d = raw['flow_55d'].values
        tick_large_net = raw['tick_large_net'].values
        intra_accel = raw['intra_accel'].values
        breakout_flow = raw['breakout_flow'].values
        pushing_score = raw['pushing_score'].values
        sentiment = raw['market_sentiment'].values
        mf_activity = raw['mf_activity'].values
        k_slope = raw['mf_slope_13'].values
        k_accel = raw['mf_accel_13'].values
        k_jerk = raw['mf_jerk_13'].values
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
        hab_cushion = np.where((mf_net_buy < 0) & (hab_total_pool > 0), np.clip(hab_total_pool / (np.abs(mf_net_buy) + 1e-9), 0.0, 1.0) * np.abs(mf_net_buy) * 0.8, 0.0)
        effective_net_buy = mf_net_buy + hab_cushion
        synergy_multiplier = 1.0 + np.maximum(0.0, hm_synergy / 100.0)
        macro_base = effective_net_buy * synergy_multiplier
        norm_macro_base = np.sign(macro_base) * np.log1p(np.abs(macro_base) / 1000.0)
        macro_damping = np.tanh(np.abs(effective_net_buy) / 10000.0)
        tick_damping = np.tanh(np.abs(tick_large_net) / 5000.0)
        push_damping = np.clip((pushing_score - 50.0) / 50.0, -1.0, 1.0)
        macro_kinematics = (k_slope + k_accel + k_jerk) * macro_damping
        tick_kinematics = (tick_net_slope + tick_net_accel + tick_net_jerk) * tick_damping
        push_kinematics = (push_slope + push_accel + push_jerk) * np.maximum(0.0, push_damping)
        coupling_field = np.tanh(macro_kinematics + tick_kinematics + push_kinematics)
        kinematic_multiplier = 1.0 + np.maximum(0.0, coupling_field)
        purity_multiplier = 1.0 + np.maximum(0.0, np.tanh(buy_elg_rate / 20.0))
        acc_conf_norm = np.clip((intra_acc_conf - 0.5) * 2.0, -1.0, 1.0)
        acc_confidence_multiplier = 1.0 + np.maximum(0.0, acc_conf_norm)
        macro_momentum = norm_macro_base * purity_multiplier * acc_confidence_multiplier * (1.0 + coupling_field * 0.5)
        persistence_factor = np.tanh(flow_persistence / 120.0)
        tick_intensity = np.clip(tick_large_net / (np.abs(effective_net_buy) + 1e-9), -50.0, 50.0)
        detonation_boost = 1.0 + np.tanh(np.maximum(0.0, tick_abnormal_vol - 1.0))
        norm_flow_consistency = np.clip(flow_consistency / 100.0, 0.0, 1.0)
        energy_dissipation = np.maximum(0.01, 1.0 - norm_flow_consistency)
        micro_jet_raw = (np.clip(intra_accel / 10.0, -5.0, 5.0) + tick_intensity) * (pushing_score / 100.0) * mf_activity * persistence_factor * detonation_boost / energy_dissipation
        jet_exponent = np.tanh(micro_jet_raw) * (breakout_flow / 50.0) * np.maximum(0.1, norm_flow_consistency)
        micro_multiplier = np.exp(np.clip(jet_exponent, -2.0, 2.0))
        closing_amplifier = 1.0 + np.maximum(0.0, np.tanh((closing_intensity - 50.0) / 50.0))
        sentiment_amplifier = 1.0 + np.maximum(0.0, np.clip((sentiment - 50.0) / 50.0, -1.0, 1.0))
        industry_resonance = 1.0 + np.maximum(0.0, np.clip((industry_markup - 50.0) / 50.0, -1.0, 1.0))
        phase_alignment = np.where((macro_momentum > 0) & (industry_markup > 50), 1.2, 0.8)
        base_final_thrust = macro_momentum * micro_multiplier * kinematic_multiplier * sentiment_amplifier * closing_amplifier * industry_resonance * phase_alignment
        excess_kine = np.maximum(0.0, kinematic_multiplier - 1.0)
        excess_jet = np.maximum(0.0, micro_multiplier - 1.0)
        excess_beta = np.maximum(0.0, industry_resonance - 1.0)
        critical_resonance_index = excess_kine * excess_jet * excess_beta * acc_confidence_multiplier
        nonlinear_gain = 1.0 + np.expm1(np.clip(critical_resonance_index * 2.5, 0.0, 5.0))
        ultimate_thrust = np.clip(np.nan_to_num(base_final_thrust * nonlinear_gain, nan=0.0), -1000.0, 1000.0)
        if self._is_probe_enabled():
            locs = self._get_probe_locs(idx, ultimate_thrust)
            for i in locs:
                ts = idx[i].strftime('%Y-%m-%d')
                probe_log = [
                    f"\n[PROBE-THRUST-V18.0] 交叉耦合推力(微观对齐无损版) @ {ts}",
                    f"  |- 耦合场 (Coupling Field):",
                    f"     [Raw Kine] MFSlope:{k_slope[i]:.2f}, MFAccel:{k_accel[i]:.2f}, MFJerk:{k_jerk[i]:.2f} | PushSlope:{push_slope[i]:.2f}, PushAccel:{push_accel[i]:.2f}, PushJerk:{push_jerk[i]:.2f}",
                    f"     [Raw Eco]  HM_Synergy:{hm_synergy[i]:.2f}, BuyElgRate:{buy_elg_rate[i]:.2f}, IntraAccConf:{intra_acc_conf[i]:.2f}, IndMarkup:{industry_markup[i]:.2f}",
                    f"     Macro Momentum: {macro_momentum[i]:.4f} | Coupling Field: {coupling_field[i]:.4f}",
                    f"     Phase Alignment: {phase_alignment[i]:.2f} | Acc Conf Mult: {acc_confidence_multiplier[i]:.4f}",
                    f"  |- 能量微观结构 (Micro Structure):",
                    f"     [Raw Jet]  IntraAccel:{intra_accel[i]:.2f}, TickLargeNet:{tick_large_net[i]:.0f}, PushingScore:{pushing_score[i]:.2f}, MFActivity:{mf_activity[i]:.2f}",
                    f"     [Raw Aux]  FlowConsist:{norm_flow_consistency[i]:.2f}, FlowPersist:{flow_persistence[i]:.1f}, TickAbnVol:{tick_abnormal_vol[i]:.2f}, BreakoutFlow:{breakout_flow[i]:.2f}",
                    f"     Dissipation Factor: {energy_dissipation[i]:.4f} | Micro Jet (Post-Dissipation): {micro_jet_raw[i]:.4f}",
                    f"  |- 环境共振 (Environment & Resonance):",
                    f"     [Raw Env]  Sentiment:{sentiment[i]:.2f}, ClosingIntens:{closing_intensity[i]:.2f}",
                    f"     CRI: {critical_resonance_index[i]:.6f} | Exponential Gain: x{nonlinear_gain[i]:.4f}",
                    f"  |- 终极输出 (Ultimate Tensor):",
                    f"     -> ULTIMATE THRUST: {ultimate_thrust[i]:.4f}\n"
                ]
                self._probe_cache.extend(probe_log)
                for line in probe_log: print(line)
        return ultimate_thrust

    def _calc_structure_component(self, raw: Dict[str, np.ndarray], idx: pd.Index) -> np.ndarray:
        """
        【V18.0 · 晶格相变归一修正版】
        重设底层结构因子的连续激活域，平抑过度膨胀；恢复断层的控制动力学变量。
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
        cost_rbf = np.exp(np.clip(-10.0 * (cost_gap - 0.05)**2, -20.0, 20.0))
        entropy_raw = np.maximum(0.01, chip_entropy)
        norm_intra_consolidation = 1.0 / (1.0 + np.exp(np.clip(-5.0 * (intra_consolidation - 0.5), -10.0, 10.0)))
        stability_raw = np.clip(chip_stability, 0.0, 1.0) + norm_intra_consolidation * 0.5
        entropy_penalty = np.maximum(1e-4, entropy_raw / (1.0 + stability_raw))
        norm_coherence = 1.0 / (1.0 + np.exp(np.clip(-2.0 * ma_coherence, -10.0, 10.0)))
        lattice_orderliness = (stability_raw * np.maximum(0.1, norm_coherence)) / entropy_penalty
        norm_convergence = np.clip(chip_convergence, 0.0, 1.0)
        convergence_factor = 1.0 + norm_convergence
        norm_tension = np.tanh(np.maximum(0.0, ma_tension) / 2.0)
        elastic_compression = np.maximum(0.0, norm_tension * norm_convergence)
        norm_peak_conc = 1.0 / (1.0 + np.exp(np.clip(-0.1 * (peak_conc - 50.0), -20.0, 20.0)))
        peak_efficiency = norm_peak_conc * (winner_rate / 100.0) * convergence_factor
        norm_control = np.clip(control_solidity, 0.0, 1.0)
        control_factor = 1.0 + norm_control * 0.5
        norm_acc = 1.0 / (1.0 + np.exp(np.clip(-0.05 * (accumulation_score - 50.0), -20.0, 20.0)))
        acc_factor = 1.0 + norm_acc
        static_lattice_energy = lattice_orderliness * peak_efficiency * cost_rbf * control_factor * acc_factor
        inertia_bonus = 1.0 + np.maximum(0.0, (hab_structure - 0.6) * 1.5)
        hab_pool = flow_21d * 0.618 + flow_55d * 0.382
        hab_immunity = 1.0 - (1.0 / (1.0 + np.exp(np.clip(hab_pool / 50000.0, -20.0, 20.0))))
        hab_immunity = np.maximum(0.0, np.minimum(hab_immunity, 0.85))
        ctrl_damping = np.maximum(0.1, norm_control)
        raw_ctrl_kine = ctrl_slope_13 + ctrl_accel_13 + ctrl_jerk_13
        protected_ctrl_kine = np.where(raw_ctrl_kine < 0, raw_ctrl_kine * (1.0 - hab_immunity), raw_ctrl_kine)
        effective_ctrl_kine = protected_ctrl_kine * ctrl_damping
        k_conc_slope = np.tanh(conc_slope * 2.0)
        k_winner_accel = np.tanh(winner_accel * 1.5)
        kine_vector = (k_conc_slope * 0.2) + (k_winner_accel * 0.15) + (np.tanh(effective_ctrl_kine) * 0.35)
        evolution_kinematics = 1.0 + kine_vector * (1.0 + elastic_compression * 2.0)
        norm_consolidation_conc = 1.0 / (1.0 + np.exp(np.clip(-0.1 * (consolidation_chip_conc - 50.0), -20.0, 20.0)))
        consolidation_boost = 1.0 + norm_consolidation_conc
        norm_platform = 1.0 / (1.0 + np.exp(np.clip(-0.1 * (platform_quality - 50.0), -20.0, 20.0)))
        platform_factor = 1.0 + norm_platform * 0.6 * consolidation_boost
        sr_factor = np.exp(np.clip(np.tanh(sr_ratio - 1.0), -5.0, 5.0))
        norm_foundation = 1.0 / (1.0 + np.exp(np.clip(-5.0 * (foundation_strength - 0.5), -10.0, 10.0)))
        foundation_factor = 1.0 + norm_foundation * 0.4 * sr_factor
        pattern_bonus = 1.0 + (rounding_bottom * 0.3)
        base_structure = static_lattice_energy * inertia_bonus * evolution_kinematics * platform_factor * foundation_factor * pattern_bonus
        sri = (lattice_orderliness * norm_platform * hab_structure * np.maximum(0.1, norm_tension))
        excitation_gain = 1.0 + np.expm1(np.clip(np.maximum(0.0, sri - 0.5), 0.0, 5.0)) * 1.5
        resonance_core = base_structure * excitation_gain
        avalanche_threshold = 1.5
        excess_res = np.clip(np.maximum(0.0, resonance_core - avalanche_threshold), 0.0, 10.0)
        avalanche_gain = 1.0 + (excess_res ** 2) * 2.0
        final_structure = np.clip(np.nan_to_num(resonance_core * avalanche_gain, nan=1.0, posinf=1000.0, neginf=0.01), 0.01, 1000.0)
        if self._is_probe_enabled():
            locs = self._get_probe_locs(idx, final_structure)
            for i in locs:
                ts = idx[i].strftime('%Y-%m-%d')
                probe_log = [
                    f"\n[PROBE-STRUCTURE-V18.0] 晶格相变全息审计(量纲收拢版) @ {ts}",
                    f"  |- 熵稳平衡机制 (Entropy-Stability):",
                    f"     [Raw Lattice] ChipEntropy:{chip_entropy[i]:.4f}, ChipStability:{chip_stability[i]:.4f}, IntraConsol:{intra_consolidation[i]:.2f}, MACoherence:{ma_coherence[i]:.2f}",
                    f"     Stability Adjusted: {stability_raw[i]:.4f} | Entropy Penalty: {entropy_penalty[i]:.4f} -> Lattice Orderliness: {lattice_orderliness[i]:.4f}",
                    f"  |- 弹性交叉张量 (Elastic Compression):",
                    f"     [Raw Elastic] MATension:{ma_tension[i]:.2f}, ChipConverge:{chip_convergence[i]:.2f}",
                    f"     Tension (Norm): {norm_tension[i]:.4f} * Convergence (Norm): {norm_convergence[i]:.4f} -> Ratio: {elastic_compression[i]:.4f}",
                    f"  |- 动量耦合放大 (Kinematic Amplification):",
                    f"     [Raw Status]  PeakConc:{peak_conc[i]:.2f}, WinnerRate:{winner_rate[i]:.2f}, CtrlSolid:{control_solidity[i]:.2f}, AccumScore:{accumulation_score[i]:.2f}",
                    f"     [Raw Kine]    ConcSlope:{conc_slope[i]:.2f}, WinnerAccel:{winner_accel[i]:.2f}, CtrlSlope13:{ctrl_slope_13[i]:.2f}",
                    f"     Raw Kine Vector: {kine_vector[i]:.4f} | Amplifier (1 + Elastic*2): {(1.0 + elastic_compression[i]*2.0):.4f} -> Evolution: {evolution_kinematics[i]:.4f}",
                    f"  |- 共振与雪崩增益 (Resonance & Avalanche):",
                    f"     [Raw Base]    PlatformQual:{platform_quality[i]:.2f}, FoundStrength:{foundation_strength[i]:.2f}, SR_Ratio:{sr_ratio[i]:.2f}, RoundBtm:{rounding_bottom[i]:.0f}",
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
        【V18.0 · 阻力引力场平滑限幅版】
        针对 intraday_dist 与 ind_downtrend 的自然比率边界进行激活修复。
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
        hab_immunity = 1.0 - (1.0 / (1.0 + np.exp(np.clip(hab_pool / 50000.0, -20.0, 20.0))))
        hab_immunity = np.clip(hab_immunity, 0.0, 0.9)
        hab_burden = np.maximum(0.0, -hab_pool) / 50000.0
        hab_drag_penalty = 1.0 + np.tanh(hab_burden)
        norm_profit_pressure = np.expm1(np.clip(np.maximum(0.0, profit_pressure) / 50.0, 0.0, 10.0))
        norm_trapped_pressure = np.expm1(np.clip(np.maximum(0.0, trapped_pressure) / 50.0, 0.0, 10.0)) * 1.5
        norm_dist = 1.0 / (1.0 + np.exp(np.clip(-0.1 * (dist_score - 50.0), -20.0, 20.0)))
        norm_intra_dist = 1.0 / (1.0 + np.exp(np.clip(-5.0 * (intraday_dist - 0.5), -10.0, 10.0)))
        norm_instability = 1.0 / (1.0 + np.exp(np.clip(-0.05 * (instability - 50.0), -20.0, 20.0)))
        norm_downtrend = 1.0 / (1.0 + np.exp(np.clip(-0.1 * (downtrend_str - 50.0), -20.0, 20.0)))
        dump_quality_factor = 1.0 + np.maximum(0.0, outflow_qual / 100.0) * 1.5
        energy_factor = 1.0 + np.maximum(0.0, dist_energy / 100.0)
        coupled_active_dump = (norm_dist + norm_intra_dist * 0.5) * dump_quality_factor * energy_factor * kine_multiplier
        beta_headwind = 1.0 + np.maximum(0.0, ind_downtrend)
        friction_vpa = 1.0 + np.maximum(0.0, 1.0 - (vpa_efficiency / 100.0))
        skew_penalty = 1.0 + np.maximum(0.0, -intra_skew) * 0.5
        coupled_viscosity = (1.0 + norm_instability) * beta_headwind * friction_vpa * skew_penalty
        coupled_gravity = (norm_profit_pressure + norm_trapped_pressure) * (1.0 + norm_downtrend)
        norm_release = 1.0 / (1.0 + np.exp(np.clip(-0.05 * (pressure_release - 50.0), -20.0, 20.0)))
        norm_shakeout = 1.0 / (1.0 + np.exp(np.clip(-0.05 * (shakeout_score - 50.0), -20.0, 20.0)))
        relief_valve = np.maximum(0.1, 1.0 + norm_release * 1.5 + norm_shakeout * 1.0)
        hf_hidden_div = np.maximum(0.0, hf_flow_div / 50.0)
        turnover_drag = np.expm1(np.clip((turnover_rate / 100.0) - 0.05, 0.0, 5.0) * 10.0) * 0.5
        core_drag_raw = np.clip(((coupled_gravity + coupled_active_dump) * coupled_viscosity * hab_drag_penalty) / relief_valve, 0.0, 1000.0)
        core_drag_shielded = np.clip(core_drag_raw * (1.0 - hab_immunity) + turnover_drag + hf_hidden_div, 0.0, 1000.0)
        avalanche_threshold = 1.5
        excess_drag = np.clip(np.maximum(0.0, core_drag_shielded - avalanche_threshold), 0.0, 10.0)
        avalanche_gain = 1.0 + (excess_drag ** 2) * 2.5
        final_drag = np.clip(np.nan_to_num(core_drag_shielded * avalanche_gain, nan=0.0, posinf=10000.0, neginf=0.0), 0.0, 10000.0)
        if self._is_probe_enabled():
            locs = self._get_probe_locs(idx, final_drag)
            for i in locs:
                ts = idx[i].strftime('%Y-%m-%d')
                probe_log = [
                    f"\n[PROBE-DRAG-V18.0] 踩踏黑洞物理域受控探针 @ {ts}",
                    f"  |- 动力学抗噪传导 (Kinematic Denoising):",
                    f"     [Raw Dist] DistScore:{dist_score[i]:.2f}, DistSlope:{dist_slope_13[i]:.2f}, DistAccel:{dist_accel_13[i]:.2f}, DistJerk:{dist_jerk_13[i]:.2f}",
                    f"     Raw Kine (S+A+J): {raw_dist_kine[i]:.4f} | Damping: {dist_damping[i]:.4f} -> Effective Kine: {effective_dist_kine[i]:.4f}",
                    f"  |- 主动倾泻张量 (Coupled Active Dump):",
                    f"     [Raw Dump] IntraDist:{intraday_dist[i]:.2f}, OutflowQual:{outflow_qual[i]:.2f}, DistEnergy:{dist_energy[i]:.2f}",
                    f"     Dist Norm: {norm_dist[i]:.4f} | Quality Factor: x{dump_quality_factor[i]:.4f} | Energy Factor: x{energy_factor[i]:.4f} -> Active Dump: {coupled_active_dump[i]:.4f}",
                    f"  |- 环境粘滞张量 (Coupled Viscosity):",
                    f"     [Raw Visc] Instability:{instability[i]:.2f}, VPA_Eff:{vpa_efficiency[i]:.2f}, IndDowntrend:{ind_downtrend[i]:.2f}, IntraSkew:{intra_skew[i]:.2f}",
                    f"     Instability: {norm_instability[i]:.4f} | VPA Friction: {friction_vpa[i]:.4f} | Beta Headwind: {beta_headwind[i]:.4f} -> Viscosity: {coupled_viscosity[i]:.4f}",
                    f"  |- 核心中枢与泄压 (Core Synthesis):",
                    f"     [Raw Core] ProfitPres:{profit_pressure[i]:.2f}, TrapPres:{trapped_pressure[i]:.2f}, DowntrendStr:{downtrend_str[i]:.2f}, PresRel:{pressure_release[i]:.2f}, Shakeout:{shakeout_score[i]:.2f}",
                    f"     [Raw Aux]  Turnover:{turnover_rate[i]:.2f}, HfFlowDiv:{hf_flow_div[i]:.2f}",
                    f"     Coupled Gravity: {coupled_gravity[i]:.4f} | Relief Valve: /{relief_valve[i]:.4f} | Shielded Core: {core_drag_shielded[i]:.4f}",
                    f"  |- 终极阻力张量 (Final Drag Tensor):",
                    f"     Avalanche Multiplier: x{avalanche_gain[i]:.4f} -> FINAL DRAG: {final_drag[i]:.4f}\n"
                ]
                self._probe_cache.extend(probe_log)
                for line in probe_log: print(line)
        return final_drag

    def _calc_tensor_synthesis(self, thrust: np.ndarray, structure: np.ndarray, drag: np.ndarray, raw: Dict[str, np.ndarray], idx: pd.Index) -> np.ndarray:
        """
        【V18.0 · 奇点张量合成修复版】
        修复了由于情感极值 (emotional_extreme) 归零导致的整个涡流死锁现象。
        """
        mf_net_buy = raw['mf_net_buy'].values
        pushing_score = raw['pushing_score'].values
        energy_conc = raw['energy_conc'].values
        mf_slope_13 = raw['mf_slope_13'].values
        mf_accel_13 = raw['mf_accel_13'].values
        mf_jerk_13 = raw['mf_jerk_13'].values
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
        trend_confirm = raw['trend_confirm'].values
        emotional_extreme = raw['emotional_extreme'].values
        game_intensity = raw['game_intensity'].values
        golden_pit = raw['golden_pit'].values
        breakout_conf = raw['breakout_conf'].values
        hm_top_tier = raw['hm_top_tier'].values
        t1_premium = raw['t1_premium'].values
        breakout_pot = raw['breakout_pot'].values
        ma_compression = raw['ma_compression'].values
        turnover_stability = raw['turnover_stability'].values
        norm_push = 1.0 / (1.0 + np.exp(np.clip(-0.1 * (pushing_score - 50.0), -20.0, 20.0)))
        kine_damping = np.tanh(np.abs(mf_net_buy) / 10000.0) * norm_push
        k_mf = np.tanh(mf_slope_13) * 0.3 + np.tanh(mf_accel_13) * 0.3 + np.tanh(mf_jerk_13) * 0.4
        k_push = np.tanh(push_slope_13) * 0.3 + np.tanh(push_accel_13) * 0.3 + np.tanh(push_jerk_13) * 0.4
        kinematic_burst = 1.0 + np.maximum(0.0, (k_mf + k_push * 0.5) * kine_damping)
        combined_inventory = (flow_21d * 0.618) + (flow_55d * 0.382)
        hab_immunity = 1.0 - (1.0 / (1.0 + np.exp(np.clip(combined_inventory / 50000.0, -20.0, 20.0))))
        hab_fuel = np.maximum(0.0, np.tanh(combined_inventory / 100000.0))
        norm_theme = 1.0 / (1.0 + np.exp(np.clip(-0.05 * (theme_hotness - 50.0), -20.0, 20.0)))
        norm_breakout_pot = 1.0 / (1.0 + np.exp(np.clip(-5.0 * (breakout_pot - 0.5), -10.0, 10.0)))
        norm_turnover_stab = 1.0 / (1.0 + np.exp(np.clip(-5.0 * (turnover_stability - 0.5), -10.0, 10.0)))
        eco_premium = 1.0 + (is_leader * 0.8) + (hm_top_tier * 0.6) + (breakout_conf * 0.4) + (norm_theme * 0.3) + (trend_confirm * 0.5)
        eff_structure = np.where(thrust >= 0, structure, 1.0 / np.clip(structure, 0.01, 1000.0))
        eff_eco_premium = np.where(thrust >= 0, eco_premium, 1.0 / np.clip(eco_premium, 0.01, 100.0))
        eff_gap_mom = np.where(thrust >= 0, 1.0 + gap_momentum, 1.0 / np.clip(1.0 + gap_momentum, 0.01, 10.0))
        eff_breakout = np.where(thrust >= 0, 1.0 + norm_breakout_pot, 1.0 / np.clip(1.0 + norm_breakout_pot, 0.01, 10.0))
        eff_turnover = np.where(thrust >= 0, 1.0 + norm_turnover_stab * 0.5, 1.0 / np.clip(1.0 + norm_turnover_stab * 0.5, 0.01, 10.0))
        base_tensor = thrust * eff_structure * eff_gap_mom * eff_eco_premium * kinematic_burst * eff_breakout * eff_turnover
        norm_lock_ratio = 1.0 / (1.0 + np.exp(np.clip(-0.1 * (lock_ratio - 50.0), -20.0, 20.0)))
        raw_effective_drag = drag * (1.0 - np.maximum(0.0, np.minimum(hab_immunity, 0.90)))
        exp_arg = np.clip(-2.0 * (base_tensor - 1.5 * raw_effective_drag), -20.0, 20.0)
        squeeze_transition = 1.0 / (1.0 + np.exp(exp_arg))
        norm_game_intensity = 1.0 / (1.0 + np.exp(np.clip(-5.0 * (game_intensity - 0.5), -10.0, 10.0)))
        trap_reversal_factor = 1.0 + (golden_pit * 2.0)
        norm_energy = 1.0 / (1.0 + np.exp(np.clip(-5.0 * (energy_conc - 0.5), -10.0, 10.0)))
        squeeze_bonus = np.where(base_tensor >= 0, squeeze_transition * raw_effective_drag * (1.0 + emotional_extreme) * norm_game_intensity * kinematic_burst * trap_reversal_factor * norm_energy, 0.0)
        norm_reversal_prob = np.clip(reversal_prob / 100.0, 0.0, 1.0)
        final_drag = (raw_effective_drag * raw_effective_drag) * (1.0 - squeeze_transition) * (1.0 - norm_reversal_prob) * (1.0 - norm_lock_ratio * 0.5)
        raw_intent = np.where(
            base_tensor >= 0,
            (base_tensor / np.maximum(1.0 + final_drag, 1.0)) + squeeze_bonus,
            base_tensor * (1.0 + np.sqrt(np.clip(final_drag, 0.0, 10000.0)))
        )
        t1_multiplier = np.exp(np.clip(np.tanh((t1_premium - 50.0) / 20.0), -2.0, 2.0))
        norm_compression = np.clip(ma_compression, 0.0, 1.0)
        hri = np.where(
            base_tensor >= 0,
            (base_tensor * (1.0 + squeeze_bonus)) / np.maximum(1.0 + final_drag, 1.0),
            base_tensor * (1.0 + np.sqrt(np.clip(final_drag, 0.0, 10000.0)))
        )
        hri_threshold = 3.0
        hri_magnitude = np.abs(hri)
        hri_excess = np.clip(np.maximum(0.0, hri_magnitude - hri_threshold), 0.0, 15.0)
        exponent_gain = np.clip(hri_excess * t1_multiplier * (1.0 + norm_compression + hab_fuel), 0.0, 8.0)
        singularity_gain = 1.0 + np.expm1(exponent_gain)
        final_intent = np.clip(np.nan_to_num(raw_intent * singularity_gain, nan=0.0), -10000.0, 10000.0)
        if self._is_probe_enabled():
            locs = self._get_probe_locs(idx, final_intent)
            for i in locs:
                ts = idx[i].strftime('%Y-%m-%d')
                probe_log = [
                    f"\n[PROBE-SYNTHESIS-V18.0] 张量奇点黑洞逃逸纠偏探针 @ {ts}",
                    f"  |- 多维动力学与抗噪 (Multi-Kinematics):",
                    f"     [Raw Config] MFNetBuy:{mf_net_buy[i]:.2f}, PushingScore:{pushing_score[i]:.2f}",
                    f"     MF Kine: {k_mf[i]:.4f} | Push Kine: {k_push[i]:.4f} | Damping: {kine_damping[i]:.4f} -> Burst: x{kinematic_burst[i]:.4f}",
                    f"  |- 定向耦合结构与生态 (Directional Coupling):",
                    f"     [Raw Eco]    ThemeHot:{theme_hotness[i]:.2f}, Leader:{is_leader[i]:.0f}, BreakoutConf:{breakout_conf[i]:.0f}, HMTopTier:{hm_top_tier[i]:.0f}, TrendConf:{trend_confirm[i]:.2f}",
                    f"     Eff_Structure: {eff_structure[i]:.4f} | Eff_Eco_Premium: {eff_eco_premium[i]:.4f} | Eff_Breakout: {eff_breakout[i]:.4f}",
                    f"  |- HAB 蓄水池防御与燃烧 (HAB Defense & Fuel):",
                    f"     HAB Pool: {combined_inventory[i]:.0f} -> Immunity Shield: {hab_immunity[i]*100:.2f}%",
                    f"     Drag Extinguishment: {drag[i]:.4f} -> Effective Drag: {raw_effective_drag[i]:.4f}",
                    f"  |- 黄金坑轧空涡流 (Golden Pit & Squeeze):",
                    f"     [Raw Vortex] GoldenPit:{golden_pit[i]:.0f}, GameIntens:{game_intensity[i]:.2f}, EnergyConc:{energy_conc[i]:.2f}, EmoExtreme:{emotional_extreme[i]:.0f}, RevProb:{reversal_prob[i]:.2f}",
                    f"     Squeeze Transition: {squeeze_transition[i]:.4f} | Trap Reversal Factor: {trap_reversal_factor[i]:.4f}",
                    f"     Squeeze Bonus (Vortex Fuel): +{squeeze_bonus[i]:.4f} | Final Suppressed Drag: {final_drag[i]:.4f}",
                    f"  |- 奇点爆炸增益 (Singularity Resonance Gain):",
                    f"     [Raw Sing]   T1Premium:{t1_premium[i]:.2f}, MACompress:{ma_compression[i]:.2f}",
                    f"     HRI Index: {hri[i]:.6f} (Threshold: {hri_threshold}) | Abs HRI Excess: {hri_excess[i]:.4f}",
                    f"     T+1 Premium Multiplier: x{t1_multiplier[i]:.4f} | MA Compression Tanh: {norm_compression[i]:.4f}",
                    f"     HAB Base Fuel: {hab_fuel[i]:.4f} -> Singularity Multiplier: x{singularity_gain[i]:.4f}",
                    f"  |- 终极意图张量 (Final Intent Tensor):",
                    f"     Base Tensor: {base_tensor[i]:.4f} | Gravity Accel Enabled: {'YES' if base_tensor[i] < 0 else 'NO'}",
                    f"     Raw Intent: {raw_intent[i]:.4f} -> FINAL SYNTHESIS: {final_intent[i]:.4f}\n"
                ]
                self._probe_cache.extend(probe_log)
                for line in probe_log: print(line)
        return final_intent

    def _generate_probe_report(self, idx, raw, thrust, structure, drag, raw_intent, final):
        """
        【V16.0 · 探针终极全景抓取版】
        对接异常行与极值点，全方位铺开最终参数供调试分析。
        """
        locs = self._get_probe_locs(idx, raw_intent)
        for i in locs:
            ts = idx[i]
            net_buy = raw['mf_net_buy'].values[i]
            energy_conc = raw['energy_conc'].values[i]
            energy_damping = np.tanh(np.abs(net_buy) / 10000.0) * (energy_conc / 100.0)
            k_burst = 1.0 + ((np.tanh(raw['mf_slope_13'].values[i]) * 0.3 + np.tanh(raw['mf_accel_13'].values[i]) * 0.3 + np.tanh(raw['mf_jerk_13'].values[i]) * 0.4) * energy_damping)
            comb_inv = (raw['flow_21d'].values[i] * 0.6) + (raw['flow_55d'].values[i] * 0.4)
            hab_imm = 1.0 - (1.0 / (1.0 + np.exp(comb_inv / 50000.0)))
            eff_drag = drag[i] * (1.0 - hab_imm)
            report = [
                f"\n=== [PROBE V16.0] CalculateMainForceRallyIntent Full-Chain Audit (无掩模暴露) @ {ts.strftime('%Y-%m-%d')} ===",
                f"【0. Raw Data Overview (底层核心数据快照)】",
                f"   [Thrust] MF_NetBuy: {raw['mf_net_buy'].values[i]:.2f} | Tick_Large_Net: {raw['tick_large_net'].values[i]:.2f} | Flow_21d: {raw['flow_21d'].values[i]:.2f} | Flow_55d: {raw['flow_55d'].values[i]:.2f}",
                f"   [Struct] Close: {raw['close'].values[i]:.2f} | Cost_Avg: {raw['cost_avg'].values[i]:.2f} | Chip_Entropy: {raw['chip_entropy'].values[i]:.4f} | Control_Solidity: {raw['control_solidity'].values[i]:.4f}",
                f"   [Drag]   Profit_Pres: {raw['profit_pressure'].values[i]:.4f} | Trapped_Pres: {raw['trapped_pressure'].values[i]:.4f} | Dist_Score: {raw['dist_score'].values[i]:.4f} | Turnover: {raw['turnover'].values[i]:.4f}",
                f"   [Eco]    Market_Sentiment: {raw['market_sentiment'].values[i]:.4f} | Is_Leader: {raw['is_leader'].values[i]:.1f} | Reversal_Prob: {raw['reversal_prob'].values[i]:.4f}",
                f"---------------------------------------------------------------",
                f"【A. Kinematics (动力学)】 Burst: x{k_burst:.4f} | Damping: {energy_damping:.4f} | Jerk: {raw['mf_jerk_13'].values[i]:.2f}",
                f"【B. HAB (存量意识)】 21d/55d Inv: {raw['flow_21d'].values[i]:.0f}/{raw['flow_55d'].values[i]:.0f} | Immunity: {hab_imm*100:.1f}%",
                f"【C. Ecosystem (生态)】 Leader: {raw['is_leader'].values[i]} | LockRatio: {raw['lock_ratio'].values[i]:.2f}% | Trend Confirm: {raw['trend_confirm'].values[i]:.2f}",
                f"【E. Synthesis (合成)】 Thrust: {thrust[i]:.4f} | Structure: {structure[i]:.4f} | EffectiveDrag: {eff_drag:.4f}",
                f"【F. Result (最终)】 Raw Intent: {raw_intent[i]:.4f} | Final Normalized Score: {final[i]:.4f}",
                f"===============================================================\n"
            ]
            self._probe_cache.extend(report)
            for line in report: print(line)

    def _is_probe_enabled(self) -> bool:
        return get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)











