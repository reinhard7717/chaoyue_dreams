import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from strategies.trend_following.utils import get_params_block, get_param_value, _robust_geometric_mean
from strategies.trend_following.intelligence.process.helper import ProcessIntelligenceHelper

class CalculateSplitOrderAccumulation:
    """
    PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY
    拆单吸筹强度
    【V5.1.0 · 微观动力学势场与探针裸露版】
    - 理论升级：A股算法拆单呈现显著的高频碎化特征。基于此构建“微观拆单动能（Tick聚类、高频偏度/峰度、单笔均值萎缩）”与“吸筹势垒（筹码熵降、均线势能压缩）”的相空间动力学模型。
    - 数据净化：剔除原依赖中缺失的资金指标，严丝合缝对齐380项底层军械库原生因子。
    - 破除防御：全量移除 fillna(0) 与极小值平滑掩盖，允许极值、断层表现为 NaN，倒逼底层数据管线治理。
    - 全息探针：将原料张量、衍生特征、动能势垒乘积至最终校准Gamma全部压入探针字典树输出。
    """
    def __init__(self, strategy_instance, helper: ProcessIntelligenceHelper):
        """【V5.1.0】初始化计算器，加载相空间张量配置。"""
        self.strategy = strategy_instance
        self.helper = helper
        self.params = self.helper.params
        self.debug_params = self.helper.debug_params
        self.probe_dates = self.helper.probe_dates
        p_conf = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_weights'), {})
        self.actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})

    def _internal_normalize(self, series: pd.Series, mode: str = 'unipolar', window: int = 60) -> pd.Series:
        """【V5.1.0】绝对量纲归一化，剔除平滑常数，暴露0极差导致的NaN断层。"""
        if series.empty: return series
        if mode == 'bipolar':
            roll_mean = series.rolling(window=window, min_periods=1).mean()
            roll_std = series.rolling(window=window, min_periods=1).std()
            return np.tanh(((series - roll_mean) / roll_std) * 0.5)
        elif mode == 'rank':
            return series.rolling(window=window, min_periods=1).rank(pct=True)
        elif mode == 'reverse_absolute':
            return 1.0 - series
        elif mode == 'raw_clip':
            return series.clip(0, 1)
        else:
            roll_min = series.rolling(window=window, min_periods=1).min()
            roll_max = series.rolling(window=window, min_periods=1).max()
            return (series - roll_min) / (roll_max - roll_min)

    def _get_raw_signals(self, df: pd.DataFrame, method_name: str) -> Dict[str, pd.Series]:
        """【V5.1.0】直连军械库原生因子，不进行任何兜底填充。"""
        raw_columns = [
            'stealth_flow_ratio_D', 'tick_clustering_index_D', 'high_freq_flow_skewness_D',
            'high_freq_flow_kurtosis_D', 'amount_D', 'trade_count_D', 'chip_convergence_ratio_D',
            'intraday_chip_entropy_D', 'MA_POTENTIAL_COMPRESSION_RATE_D', 'behavior_accumulation_D',
            'VPA_MF_ADJUSTED_EFF_D', 'market_sentiment_score_D', 'TURNOVER_STABILITY_INDEX_D',
            'tick_data_quality_score_D', 'STATE_PARABOLIC_WARNING_D', 'IS_MARKET_LEADER_D'
        ]
        return {col: self.helper._get_safe_series(df, col, np.nan, method_name=method_name) for col in raw_columns}

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【A股量化策略五维共振引擎】主计算总管线 V10.0.0
        实现全息动能、引力势垒、极化门槛、相空间折叠与防爆衰竭校准的全数据流贯通。
        """
        method_name = "calculate_split_order_accumulation"
        is_debug = get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug and self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            for date in reversed(df.index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        df_index = df.index
        config_params = config.get('split_order_accumulation_params', {})
        raw_signals = self._get_raw_signals(df, method_name)
        ke_series, ke_debug = self._calculate_kinetic_energy(df, df_index, config_params)
        pb_series, pb_debug = self._calculate_potential_barrier(df, df_index, config_params)
        baseline, baseline_debug = self._calculate_dynamic_baseline(df, df_index, config_params)
        holo_score, holo_debug = self._calculate_holographic_validation(df, ke_series, pb_series, df_index, config_params)
        final_output, calib_debug = self._apply_calibration_and_warning(df, holo_score, baseline, df_index, config_params)
        if probe_ts is not None:
            debug_dict = {
                "1.原始物理量场(Raw_Signals)": {k: v.loc[probe_ts] if probe_ts in v.index else np.nan for k, v in raw_signals.items()},
                "2.拆单动能张量(Kinetic_Energy)": {k: v.loc[probe_ts] if probe_ts in v.index else np.nan for k, v in ke_debug.items()},
                "3.吸筹势垒张量(Potential_Barrier)": {k: v.loc[probe_ts] if probe_ts in v.index else np.nan for k, v in pb_debug.items()},
                "4.极化基准面(Dynamic_Baseline)": {k: v.loc[probe_ts] if probe_ts in v.index else np.nan for k, v in baseline_debug.items()},
                "5.相空间做功(Holographic_Work)": {k: v.loc[probe_ts] if probe_ts in v.index else np.nan for k, v in holo_debug.items()},
                "6.免疫防爆校准(Calibration_Warning)": {k: v.loc[probe_ts] if probe_ts in v.index else np.nan for k, v in calib_debug.items()}
            }
            debug_output = {f"--- {method_name} V10.0.0 满级五维共振物理引擎探针 @ {probe_ts.strftime('%Y-%m-%d')} ---": ""}
            self._print_debug_info(method_name, probe_ts, debug_output, debug_dict, final_output)
        return final_output.astype(np.float32)

    def _calculate_kinetic_energy(self, df: pd.DataFrame, df_index: pd.Index, config: Dict) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """
        【A股量化策略五维共振引擎】_calculate_kinetic_energy 全息重构版
        严格执行七步级联逻辑：全息哨兵交叉验证、多阶微分运动学、HAB存量记忆反身性衰减、
        非线性相变指数增益、动态张量融合与一票否决、Regime环境自适应、反身性免疫与代码硬化。
        """
        eps = 1e-6
        emo_extreme = self.helper._get_safe_series(df, 'STATE_EMOTIONAL_EXTREME_D', 0.0)
        chip_game = self.helper._get_safe_series(df, 'intraday_chip_game_index_D', 0.0)
        turnover_stab = self.helper._get_safe_series(df, 'TURNOVER_STABILITY_INDEX_D', 0.0)
        vpa_eff = self.helper._get_safe_series(df, 'VPA_EFFICIENCY_D', 0.0)
        coord_attack = self.helper._get_safe_series(df, 'HM_COORDINATED_ATTACK_D', 0.0)
        geom_r2 = self.helper._get_safe_series(df, 'GEOM_REG_R2_D', 0.0)
        validated_vpa = (vpa_eff * coord_attack * (1.0 + geom_r2)).fillna(0.0).astype(np.float32)
        flow_cluster = self.helper._get_safe_series(df, 'flow_cluster_intensity_D', 0.0)
        tick_cluster = self.helper._get_safe_series(df, 'tick_clustering_index_D', 0.0)
        def _compute_kinematics(series: pd.Series) -> pd.Series:
            k_score = pd.Series(0.0, index=df_index, dtype=np.float32)
            for w in [5, 13, 21, 34, 55]:
                roll_std_0 = series.rolling(window=w, min_periods=1).std().fillna(0.0) + eps
                slope = series.diff(w).fillna(0.0) / roll_std_0
                roll_std_1 = slope.rolling(window=w, min_periods=1).std().fillna(0.0) + eps
                accel = slope.diff(w).fillna(0.0) / roll_std_1
                roll_std_2 = accel.rolling(window=w, min_periods=1).std().fillna(0.0) + eps
                jerk = accel.diff(w).fillna(0.0) / roll_std_2
                k_score += (slope * 0.5 + accel * 0.3 + jerk * 0.2) * (1.0 / 5.0)
            return k_score
        kin_cluster = _compute_kinematics(tick_cluster)
        kin_flow = _compute_kinematics(flow_cluster)
        kinematic_score = (np.tanh(kin_cluster * 0.5 + kin_flow * 0.5) * 0.5 + 0.5).fillna(0.0).astype(np.float32)
        net_amount = self.helper._get_safe_series(df, 'net_amount_D', 0.0)
        days_peak = self.helper._get_safe_series(df, 'days_since_last_peak_D', 0.0).clip(lower=0.0)
        hab_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        for w in [13, 21, 34, 55]:
            r_mean = net_amount.rolling(window=w, min_periods=1).mean().fillna(0.0)
            r_std = net_amount.rolling(window=w, min_periods=1).std().fillna(0.0) + eps
            breakout = (net_amount > (r_mean + 1.5 * r_std)).astype(np.float32)
            hab_score += np.where(breakout > 0.0, 1.0, 0.5)
        hab_score = (hab_score / 4.0).astype(np.float32)
        time_decay = (1.0 / np.log1p(days_peak + 1.0)).astype(np.float32)
        hab_memory = (hab_score * time_decay).fillna(0.0).astype(np.float32)
        winner_rate = self.helper._get_safe_series(df, 'winner_rate_D', 0.5)
        turnover_rate = self.helper._get_safe_series(df, 'turnover_rate_f_D', 0.0)
        winner_nl = (1.0 / (1.0 + np.exp(-10.0 * (winner_rate - 0.5)))).astype(np.float32)
        turnover_ma = turnover_rate.rolling(window=21, min_periods=1).mean().fillna(0.0) + eps
        turnover_nl = np.tanh(turnover_rate / turnover_ma).astype(np.float32)
        phase_energy = (winner_nl * 0.5 + turnover_nl * 0.5).astype(np.float32)
        resonance_trigger = (kinematic_score > 0.85) & (phase_energy > 0.85) & (chip_game > 0.85)
        phase_multiplier = np.where(resonance_trigger, np.exp(1.5), 1.0).astype(np.float32)
        flow_cons = self.helper._get_safe_series(df, 'flow_consistency_D', 0.0)
        price_flow_div = self.helper._get_safe_series(df, 'price_flow_divergence_D', 0.0)
        large_anomaly = self.helper._get_safe_series(df, 'large_order_anomaly_D', 0.0)
        dyn_weight_kin = np.where(flow_cons < 0.3, 0.4, 1.0).astype(np.float32)
        veto_trigger = (large_anomaly > 0.8) | (price_flow_div > 0.8)
        veto_penalty = np.where(veto_trigger, 0.4, 1.0).astype(np.float32)
        trend_stage = self.helper._get_safe_series(df, 'STATE_TRENDING_STAGE_D', 0.0)
        vol_adj_conc = self.helper._get_safe_series(df, 'volatility_adjusted_concentration_D', 0.0)
        interact_mult = np.where((turnover_stab > 0.7) & (coord_attack > 0.7), 1.5, np.where((turnover_stab < 0.3) | (coord_attack < 0.3), 0.8, 1.0)).astype(np.float32)
        regime_factor = np.where(trend_stage > 0.6, 1.2, np.where(trend_stage < 0.3, vol_adj_conc * 1.5, 1.0)).astype(np.float32)
        conflict_smooth = (1.0 / (1.0 + price_flow_div.clip(lower=0.0))).astype(np.float32)
        rev_prob = self.helper._get_safe_series(df, 'reversal_prob_D', 0.0)
        parab_warn = self.helper._get_safe_series(df, 'STATE_PARABOLIC_WARNING_D', 0.0)
        val_score = self.helper._get_safe_series(df, 'validation_score_D', 100.0)
        reflexivity_imm = np.exp(-1.5 * (rev_prob * 0.6 + parab_warn * 0.4 + emo_extreme * 0.2)).astype(np.float32)
        emergency_stop = np.where(val_score < 40.0, 0.0, 1.0).astype(np.float32)
        raw_tensor = (kinematic_score * dyn_weight_kin * 0.4 + hab_memory * 0.3 + phase_energy * 0.3)
        raw_tensor = raw_tensor * validated_vpa.clip(0.5, 1.5)
        final_kinetic = (raw_tensor * phase_multiplier * interact_mult * regime_factor * veto_penalty * conflict_smooth * reflexivity_imm * emergency_stop)
        final_kinetic = final_kinetic.fillna(0.0).clip(-1.0, 1.0).astype(np.float32)
        debug_dict = {
            "validated_vpa": validated_vpa,
            "kinematic_score": kinematic_score,
            "hab_memory": hab_memory,
            "phase_energy": phase_energy,
            "phase_multiplier": pd.Series(phase_multiplier, index=df_index).astype(np.float32),
            "dyn_weight_kin": pd.Series(dyn_weight_kin, index=df_index).astype(np.float32),
            "veto_penalty": pd.Series(veto_penalty, index=df_index).astype(np.float32),
            "interact_mult": pd.Series(interact_mult, index=df_index).astype(np.float32),
            "regime_factor": pd.Series(regime_factor, index=df_index).astype(np.float32),
            "conflict_smooth": pd.Series(conflict_smooth, index=df_index).astype(np.float32),
            "reflexivity_imm": pd.Series(reflexivity_imm, index=df_index).astype(np.float32),
            "emergency_stop": pd.Series(emergency_stop, index=df_index).astype(np.float32),
            "final_kinetic": final_kinetic
        }
        return final_kinetic, debug_dict

    def _calculate_potential_barrier(self, df: pd.DataFrame, df_index: pd.Index, config: Dict) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """
        【A股量化策略五维共振引擎】_calculate_potential_barrier 全息重构版
        严格执行七步级联逻辑：全息哨兵交叉验证、多阶微分运动学、HAB存量记忆反身性衰减、
        非线性相变指数增益、动态张量融合与一票否决、Regime环境自适应、反身性免疫与代码硬化。
        """
        eps = 1e-6
        chip_conv = self.helper._get_safe_series(df, 'chip_convergence_ratio_D', 0.0)
        chip_stab = self.helper._get_safe_series(df, 'chip_stability_D', 0.0)
        intra_consol = self.helper._get_safe_series(df, 'intraday_chip_consolidation_degree_D', 0.0)
        ma_comp = self.helper._get_safe_series(df, 'MA_POTENTIAL_COMPRESSION_RATE_D', 0.0)
        support_str = self.helper._get_safe_series(df, 'support_strength_D', 0.0)
        golden_pit = self.helper._get_safe_series(df, 'STATE_GOLDEN_PIT_D', 0.0)
        chip_game = self.helper._get_safe_series(df, 'intraday_chip_game_index_D', 0.0)
        validated_conv = (chip_conv * chip_stab * (1.0 + intra_consol)).fillna(0.0).astype(np.float32)
        validated_comp = (ma_comp * support_str).fillna(0.0).astype(np.float32)
        def _compute_kinematics(series: pd.Series) -> pd.Series:
            k_score = pd.Series(0.0, index=df_index, dtype=np.float32)
            for w in [5, 13, 21, 34, 55]:
                roll_std_0 = series.rolling(window=w, min_periods=1).std().fillna(0.0) + eps
                slope = series.diff(w).fillna(0.0) / roll_std_0
                roll_std_1 = slope.rolling(window=w, min_periods=1).std().fillna(0.0) + eps
                accel = slope.diff(w).fillna(0.0) / roll_std_1
                roll_std_2 = accel.rolling(window=w, min_periods=1).std().fillna(0.0) + eps
                jerk = accel.diff(w).fillna(0.0) / roll_std_2
                k_score += (slope * 0.5 + accel * 0.3 + jerk * 0.2) * 0.2
            return k_score
        kin_conv = _compute_kinematics(validated_conv)
        kin_comp = _compute_kinematics(validated_comp)
        kinematic_barrier = (np.tanh(kin_conv * 0.5 + kin_comp * 0.5) * 0.5 + 0.5).fillna(0.0).astype(np.float32)
        net_amount = self.helper._get_safe_series(df, 'net_amount_D', 0.0)
        days_peak = self.helper._get_safe_series(df, 'days_since_last_peak_D', 0.0).clip(lower=0.0)
        hab_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        for w in [13, 21, 34, 55]:
            r_mean = net_amount.rolling(window=w, min_periods=1).mean().fillna(0.0)
            r_std = net_amount.rolling(window=w, min_periods=1).std().fillna(0.0) + eps
            breakout = (net_amount > (r_mean + 1.5 * r_std)).astype(np.float32)
            hab_score += pd.Series(np.where(breakout > 0.0, 1.0, 0.5), index=df_index).astype(np.float32)
        hab_score = (hab_score / 4.0).astype(np.float32)
        time_decay = (1.0 / np.log1p(days_peak + 1.0)).astype(np.float32)
        hab_memory = (hab_score * time_decay).fillna(0.0).astype(np.float32)
        winner_rate = self.helper._get_safe_series(df, 'winner_rate_D', 0.5)
        turnover_rate = self.helper._get_safe_series(df, 'turnover_rate_f_D', 0.0)
        winner_nl = (1.0 / (1.0 + np.exp(-10.0 * (winner_rate - 0.5)))).astype(np.float32)
        turnover_ma = turnover_rate.rolling(window=21, min_periods=1).mean().fillna(0.0) + eps
        turnover_nl = np.tanh(turnover_rate / turnover_ma).astype(np.float32)
        phase_energy = (winner_nl * 0.5 + turnover_nl * 0.5).astype(np.float32)
        resonance_trigger = (kinematic_barrier > 0.85) & (hab_memory > 0.85) & (golden_pit > 0.5)
        phase_multiplier = pd.Series(np.where(resonance_trigger, np.exp(1.5), 1.0), index=df_index).astype(np.float32)
        flow_cons = self.helper._get_safe_series(df, 'flow_consistency_D', 0.0)
        price_flow_div = self.helper._get_safe_series(df, 'price_flow_divergence_D', 0.0)
        large_anomaly = self.helper._get_safe_series(df, 'large_order_anomaly_D', 0.0)
        dyn_weight_base = pd.Series(np.where(flow_cons < 0.3, 0.5, 1.0), index=df_index).astype(np.float32)
        veto_trigger = (large_anomaly > 0.8) | (price_flow_div > 0.8)
        veto_penalty = pd.Series(np.where(veto_trigger, 0.4, 1.0), index=df_index).astype(np.float32)
        trend_stage = self.helper._get_safe_series(df, 'STATE_TRENDING_STAGE_D', 0.0)
        vol_adj_conc = self.helper._get_safe_series(df, 'volatility_adjusted_concentration_D', 0.0)
        interact_mult = pd.Series(np.where((chip_stab > 0.7) & (chip_game > 0.7), 1.5, np.where((chip_stab < 0.3) | (chip_game < 0.3), 0.8, 1.0)), index=df_index).astype(np.float32)
        regime_factor = pd.Series(np.where(trend_stage < 0.3, vol_adj_conc * 1.5, np.where(trend_stage > 0.6, 0.8, 1.0)), index=df_index).astype(np.float32)
        conflict_smooth = (1.0 / (1.0 + price_flow_div.clip(lower=0.0))).astype(np.float32)
        rev_prob = self.helper._get_safe_series(df, 'reversal_prob_D', 0.0)
        parab_warn = self.helper._get_safe_series(df, 'STATE_PARABOLIC_WARNING_D', 0.0)
        emo_extreme = self.helper._get_safe_series(df, 'STATE_EMOTIONAL_EXTREME_D', 0.0)
        val_score = self.helper._get_safe_series(df, 'validation_score_D', 100.0)
        reflexivity_imm = np.exp(-1.5 * (rev_prob * 0.5 + parab_warn * 0.3 + emo_extreme * 0.2)).astype(np.float32)
        emergency_stop = pd.Series(np.where(val_score < 40.0, 0.0, 1.0), index=df_index).astype(np.float32)
        raw_tensor = (kinematic_barrier * dyn_weight_base * 0.4 + hab_memory * 0.4 + phase_energy * 0.2)
        final_barrier = (raw_tensor * phase_multiplier * interact_mult * regime_factor * veto_penalty * conflict_smooth * reflexivity_imm * emergency_stop)
        final_barrier = final_barrier.fillna(0.0).clip(-1.0, 1.0).astype(np.float32)
        debug_dict = {
            "validated_conv": validated_conv,
            "validated_comp": validated_comp,
            "kinematic_barrier": kinematic_barrier,
            "hab_memory": hab_memory,
            "phase_energy": phase_energy,
            "phase_multiplier": phase_multiplier,
            "dyn_weight_base": dyn_weight_base,
            "veto_penalty": veto_penalty,
            "interact_mult": interact_mult,
            "regime_factor": regime_factor,
            "conflict_smooth": conflict_smooth,
            "reflexivity_imm": reflexivity_imm,
            "emergency_stop": emergency_stop,
            "final_barrier": final_barrier
        }
        return final_barrier, debug_dict

    def _calculate_dynamic_baseline(self, df: pd.DataFrame, df_index: pd.Index, config: Dict) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """
        【PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY - 环境极化基准面 V8.1.0】
        严格执行七步级联逻辑：全息哨兵交叉验证、多阶微分运动学、HAB存量记忆反身性衰减、
        非线性相变指数增益、动态张量融合与一票否决、Regime环境自适应、反身性免疫与代码硬化。
        基准面(Baseline)物理意义：信号放行门槛。越低代表环境越优越，越高代表环境恶劣执行物理隔绝。
        """
        eps = 1e-6
        base_baseline = get_param_value(config.get('dynamic_efficiency_baseline_params', {}).get('base_baseline'), 0.15)
        sentiment = self.helper._get_safe_series(df, 'market_sentiment_score_D', 0.5)
        entropy = self.helper._get_safe_series(df, 'intraday_chip_entropy_D', 0.5)
        emo_extreme = self.helper._get_safe_series(df, 'STATE_EMOTIONAL_EXTREME_D', 0.0)
        geom_r2 = self.helper._get_safe_series(df, 'GEOM_REG_R2_D', 0.0)
        chip_game = self.helper._get_safe_series(df, 'intraday_chip_game_index_D', 0.0)
        turnover_stab = self.helper._get_safe_series(df, 'TURNOVER_STABILITY_INDEX_D', 0.0)
        val_sentiment = (sentiment * (1.0 - emo_extreme * 0.5) * (0.5 + turnover_stab * 0.5)).fillna(0.0).astype(np.float32)
        val_chaos = (entropy * chip_game * (1.0 - geom_r2 * 0.5)).fillna(0.0).astype(np.float32)
        def _compute_kinematics(series: pd.Series) -> pd.Series:
            k_score = pd.Series(0.0, index=df_index, dtype=np.float32)
            for w in [5, 13, 21, 34, 55]:
                r_std_0 = series.rolling(window=w, min_periods=1).std().fillna(0.0) + eps
                slope = series.diff(w).fillna(0.0) / r_std_0
                r_std_1 = slope.rolling(window=w, min_periods=1).std().fillna(0.0) + eps
                accel = slope.diff(w).fillna(0.0) / r_std_1
                r_std_2 = accel.rolling(window=w, min_periods=1).std().fillna(0.0) + eps
                jerk = accel.diff(w).fillna(0.0) / r_std_2
                k_score += (slope * 0.5 + accel * 0.3 + jerk * 0.2) * 0.2
            return k_score
        kin_sentiment = _compute_kinematics(val_sentiment)
        kin_chaos = _compute_kinematics(val_chaos)
        net_amount = self.helper._get_safe_series(df, 'net_amount_D', 0.0)
        days_peak = self.helper._get_safe_series(df, 'days_since_last_peak_D', 0.0).clip(lower=0.0)
        hab_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        for w in [13, 21, 34, 55]:
            r_mean = net_amount.rolling(window=w, min_periods=1).mean().fillna(0.0)
            r_std = net_amount.rolling(window=w, min_periods=1).std().fillna(0.0) + eps
            breakout = (net_amount > (r_mean + 1.5 * r_std)).astype(np.float32)
            hab_score += pd.Series(np.where(breakout > 0.0, 1.0, 0.5), index=df_index).astype(np.float32)
        hab_score = (hab_score / 4.0).astype(np.float32)
        time_decay = (1.0 / np.log1p(days_peak + 1.0)).astype(np.float32)
        hab_memory = (hab_score * time_decay).fillna(0.0).astype(np.float32)
        winner_rate = self.helper._get_safe_series(df, 'winner_rate_D', 0.5)
        turnover_rate = self.helper._get_safe_series(df, 'turnover_rate_f_D', 0.0)
        winner_nl = (1.0 / (1.0 + np.exp(-10.0 * (winner_rate - 0.5)))).astype(np.float32)
        turnover_ma = turnover_rate.rolling(window=21, min_periods=1).mean().fillna(0.0) + eps
        turnover_nl = np.tanh(turnover_rate / turnover_ma).astype(np.float32)
        phase_energy = (winner_nl * 0.5 + turnover_nl * 0.5).astype(np.float32)
        resonance_trigger = (kin_sentiment > 0.85) & (hab_memory > 0.85) & (phase_energy > 0.85)
        phase_multiplier = pd.Series(np.where(resonance_trigger, np.exp(-1.5), 1.0), index=df_index).astype(np.float32)
        flow_cons = self.helper._get_safe_series(df, 'flow_consistency_D', 0.0)
        dyn_weight_sent = pd.Series(np.where(flow_cons < 0.3, 0.3, 0.7), index=df_index).astype(np.float32)
        dyn_weight_chaos = pd.Series(np.where(flow_cons < 0.3, 0.7, 0.3), index=df_index).astype(np.float32)
        large_anomaly = self.helper._get_safe_series(df, 'large_order_anomaly_D', 0.0)
        price_flow_div = self.helper._get_safe_series(df, 'price_flow_divergence_D', 0.0)
        sm_divergence = self.helper._get_safe_series(df, 'SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D', 0.0)
        veto_trigger = (large_anomaly > 0.8) | (price_flow_div > 0.8) | (sm_divergence > 0.8)
        veto_penalty = pd.Series(np.where(veto_trigger, 2.0, 1.0), index=df_index).astype(np.float32)
        trend_stage = self.helper._get_safe_series(df, 'STATE_TRENDING_STAGE_D', 0.0)
        vol_adj_conc = self.helper._get_safe_series(df, 'volatility_adjusted_concentration_D', 0.0)
        hm_attack = self.helper._get_safe_series(df, 'HM_COORDINATED_ATTACK_D', 0.0)
        regime_factor = pd.Series(np.where(trend_stage > 0.6, 0.8, np.where(trend_stage < 0.3, 1.0 + vol_adj_conc * 0.5, 1.0)), index=df_index).astype(np.float32)
        interact_mult = pd.Series(np.where((turnover_stab > 0.7) & (hm_attack > 0.7), 0.8, np.where((turnover_stab < 0.3) & (hm_attack > 0.7), 1.5, 1.0)), index=df_index).astype(np.float32)
        conflict_smooth = (1.0 + price_flow_div.clip(lower=0.0)).astype(np.float32)
        rev_prob = self.helper._get_safe_series(df, 'reversal_prob_D', 0.0)
        parab_warn = self.helper._get_safe_series(df, 'STATE_PARABOLIC_WARNING_D', 0.0)
        reflexivity_imm = np.exp(1.5 * (rev_prob * 0.6 + parab_warn * 0.4 + emo_extreme * 0.2)).astype(np.float32)
        val_score = self.helper._get_safe_series(df, 'validation_score_D', 100.0)
        emergency_stop = pd.Series(np.where(val_score < 40.0, 1.0, 0.0), index=df_index).astype(np.float32)
        sentiment_impact = pd.Series(np.where(kin_sentiment > 0.0, -kin_sentiment * 0.15, np.abs(kin_sentiment) * 0.1), index=df_index).astype(np.float32)
        chaos_impact = (kin_chaos * 0.2).astype(np.float32)
        hab_impact = (-hab_memory * 0.1).astype(np.float32)
        shift = (sentiment_impact * dyn_weight_sent + chaos_impact * dyn_weight_chaos + hab_impact).astype(np.float32)
        baseline_raw = (base_baseline * (1.0 + np.tanh(shift)) * phase_multiplier * regime_factor * interact_mult * veto_penalty * conflict_smooth * reflexivity_imm).astype(np.float32)
        dynamic_baseline = pd.Series(np.where(emergency_stop == 1.0, 1.0, baseline_raw), index=df_index)
        final_baseline = dynamic_baseline.fillna(1.0).clip(0.01, 1.0).astype(np.float32)
        debug_dict = {
            "val_sentiment": val_sentiment,
            "val_chaos": val_chaos,
            "kin_sentiment": kin_sentiment,
            "kin_chaos": kin_chaos,
            "hab_memory": hab_memory,
            "phase_energy": phase_energy,
            "phase_multiplier": phase_multiplier,
            "dyn_weight_sent": dyn_weight_sent,
            "dyn_weight_chaos": dyn_weight_chaos,
            "veto_penalty": veto_penalty,
            "regime_factor": regime_factor,
            "interact_mult": interact_mult,
            "reflexivity_imm": reflexivity_imm,
            "emergency_stop": emergency_stop,
            "final_baseline": final_baseline
        }
        return final_baseline, debug_dict

    def _calculate_holographic_validation(self, df: pd.DataFrame, ke_series: pd.Series, pb_series: pd.Series, df_index: pd.Index, config: Dict) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """
        【PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY - 全息验证做功 V9.0.0】
        严格执行七步级联逻辑：全息哨兵交叉验证、多阶微分运动学、HAB存量记忆反身性衰减、
        非线性相变指数增益、动态张量融合与一票否决、Regime环境自适应、反身性免疫与代码硬化。
        """
        eps = 1e-6
        emo_extreme = self.helper._get_safe_series(df, 'STATE_EMOTIONAL_EXTREME_D', 0.0)
        large_anomaly = self.helper._get_safe_series(df, 'large_order_anomaly_D', 0.0)
        chip_game = self.helper._get_safe_series(df, 'intraday_chip_game_index_D', 0.0)
        flow_cluster = self.helper._get_safe_series(df, 'flow_cluster_intensity_D', 0.0)
        turnover_stab = self.helper._get_safe_series(df, 'TURNOVER_STABILITY_INDEX_D', 0.0)
        geom_r2 = self.helper._get_safe_series(df, 'GEOM_REG_R2_D', 0.0)
        coord_attack = self.helper._get_safe_series(df, 'HM_COORDINATED_ATTACK_D', 0.0)
        vpa_eff = self.helper._get_safe_series(df, 'VPA_EFFICIENCY_D', 0.0)
        validated_vpa = (vpa_eff * (0.5 + geom_r2 * 0.5) * (0.5 + coord_attack * 0.5)).fillna(0.0).astype(np.float32)
        val_ke = (ke_series * (0.5 + flow_cluster * 0.5)).fillna(0.0).astype(np.float32)
        val_pb = (pb_series * (0.5 + turnover_stab * 0.5)).fillna(0.0).astype(np.float32)
        def _compute_kinematics(series: pd.Series) -> pd.Series:
            k_score = pd.Series(0.0, index=df_index, dtype=np.float32)
            for w in [5, 13, 21, 34, 55]:
                r_std_0 = series.rolling(window=w, min_periods=1).std().fillna(0.0) + eps
                slope = series.diff(w).fillna(0.0) / r_std_0
                r_std_1 = slope.rolling(window=w, min_periods=1).std().fillna(0.0) + eps
                accel = slope.diff(w).fillna(0.0) / r_std_1
                r_std_2 = accel.rolling(window=w, min_periods=1).std().fillna(0.0) + eps
                jerk = accel.diff(w).fillna(0.0) / r_std_2
                k_score += (slope * 0.5 + accel * 0.3 + jerk * 0.2) * 0.2
            return k_score
        kin_ke = _compute_kinematics(val_ke)
        kin_pb = _compute_kinematics(val_pb)
        kin_vpa = _compute_kinematics(validated_vpa)
        kinematic_holo = (np.tanh(kin_ke * 0.4 + kin_pb * 0.4 + kin_vpa * 0.2) * 0.5 + 0.5).fillna(0.0).astype(np.float32)
        net_amount = self.helper._get_safe_series(df, 'net_amount_D', 0.0)
        days_peak = self.helper._get_safe_series(df, 'days_since_last_peak_D', 0.0).clip(lower=0.0)
        hab_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        for w in [13, 21, 34, 55]:
            r_mean = net_amount.rolling(window=w, min_periods=1).mean().fillna(0.0)
            r_std = net_amount.rolling(window=w, min_periods=1).std().fillna(0.0) + eps
            breakout = (net_amount > (r_mean + 1.5 * r_std)).astype(np.float32)
            hab_score += pd.Series(np.where(breakout > 0.0, 1.0, 0.5), index=df_index).astype(np.float32)
        hab_score = (hab_score / 4.0).astype(np.float32)
        time_decay = (1.0 / np.log1p(days_peak + 1.0)).astype(np.float32)
        hab_memory = (hab_score * time_decay).fillna(0.0).astype(np.float32)
        winner_rate = self.helper._get_safe_series(df, 'winner_rate_D', 0.5)
        turnover_rate = self.helper._get_safe_series(df, 'turnover_rate_f_D', 0.0)
        winner_nl = (1.0 / (1.0 + np.exp(-10.0 * (winner_rate - 0.5)))).astype(np.float32)
        turnover_ma = turnover_rate.rolling(window=21, min_periods=1).mean().fillna(0.0) + eps
        turnover_nl = np.tanh(turnover_rate / turnover_ma).astype(np.float32)
        phase_energy = (winner_nl * 0.5 + turnover_nl * 0.5).astype(np.float32)
        resonance_trigger = (kinematic_holo > 0.85) & (phase_energy > 0.85) & (coord_attack > 0.85)
        phase_multiplier = pd.Series(np.where(resonance_trigger, np.exp(1.5), 1.0), index=df_index).astype(np.float32)
        flow_cons = self.helper._get_safe_series(df, 'flow_consistency_D', 0.0)
        sm_div = self.helper._get_safe_series(df, 'SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D', 0.0)
        price_flow_div = self.helper._get_safe_series(df, 'price_flow_divergence_D', 0.0)
        dyn_w_ke = pd.Series(np.where(flow_cons < 0.3, 0.2, 0.5), index=df_index).astype(np.float32)
        dyn_w_pb = pd.Series(np.where(flow_cons < 0.3, 0.8, 0.5), index=df_index).astype(np.float32)
        veto_trigger = (sm_div > 0.8) | (large_anomaly > 0.8) | (price_flow_div > 0.8)
        veto_penalty = pd.Series(np.where(veto_trigger, 0.4, 1.0), index=df_index).astype(np.float32)
        trend_stage = self.helper._get_safe_series(df, 'STATE_TRENDING_STAGE_D', 0.0)
        vol_adj_conc = self.helper._get_safe_series(df, 'volatility_adjusted_concentration_D', 0.0)
        interact_mult = pd.Series(np.where((val_pb > 0.7) & (val_ke > 0.7), 1.5, np.where(((val_pb > 0.7) & (val_ke < 0.3)) | ((val_pb < 0.3) & (val_ke > 0.7)), 0.8, 1.0)), index=df_index).astype(np.float32)
        regime_factor = pd.Series(np.where(trend_stage < 0.3, vol_adj_conc * 1.5, np.where(trend_stage > 0.6, 1.2, 1.0)), index=df_index).astype(np.float32)
        conflict_smooth = (1.0 / (1.0 + price_flow_div.clip(lower=0.0))).astype(np.float32)
        rev_prob = self.helper._get_safe_series(df, 'reversal_prob_D', 0.0)
        parab_warn = self.helper._get_safe_series(df, 'STATE_PARABOLIC_WARNING_D', 0.0)
        val_score = self.helper._get_safe_series(df, 'validation_score_D', 100.0)
        reflexivity_imm = np.exp(-1.5 * (rev_prob * 0.6 + parab_warn * 0.4 + emo_extreme * 0.2)).astype(np.float32)
        emergency_stop = pd.Series(np.where(val_score < 40.0, 0.0, 1.0), index=df_index).astype(np.float32)
        raw_holo = (kinematic_holo * dyn_w_ke * val_ke + kinematic_holo * dyn_w_pb * val_pb + hab_memory * 0.2 + phase_energy * 0.2).astype(np.float32)
        final_holo = (raw_holo * phase_multiplier * interact_mult * regime_factor * veto_penalty * conflict_smooth * reflexivity_imm * emergency_stop)
        final_holo = final_holo.fillna(0.0).clip(-1.0, 1.0).astype(np.float32)
        debug_dict = {
            "validated_vpa": validated_vpa,
            "kinematic_holo": kinematic_holo,
            "hab_memory": hab_memory,
            "phase_energy": phase_energy,
            "phase_multiplier": phase_multiplier,
            "dyn_w_ke": dyn_w_ke,
            "dyn_w_pb": dyn_w_pb,
            "veto_penalty": veto_penalty,
            "interact_mult": interact_mult,
            "regime_factor": regime_factor,
            "conflict_smooth": conflict_smooth,
            "reflexivity_imm": reflexivity_imm,
            "emergency_stop": emergency_stop,
            "final_holo": final_holo
        }
        return final_holo, debug_dict

    def _apply_calibration_and_warning(self, df: pd.DataFrame, holo_score: pd.Series, baseline: pd.Series, df_index: pd.Index, config: Dict) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """
        【PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY - 终极校准与反身性防御 V10.0.0】
        严格执行七步级联逻辑：全息哨兵交叉验证、多阶微分运动学、HAB存量记忆反身性衰减、
        非线性相变指数增益、动态张量融合与一票否决、Regime环境自适应、反身性免疫与代码硬化。
        """
        eps = 1e-6
        data_qual = self.helper._get_safe_series(df, 'tick_data_quality_score_D', 1.0)
        turn_stab = self.helper._get_safe_series(df, 'TURNOVER_STABILITY_INDEX_D', 0.5)
        emo_extreme = self.helper._get_safe_series(df, 'STATE_EMOTIONAL_EXTREME_D', 0.0)
        large_anomaly = self.helper._get_safe_series(df, 'large_order_anomaly_D', 0.0)
        pf_div = self.helper._get_safe_series(df, 'price_flow_divergence_D', 0.0)
        sm_div = self.helper._get_safe_series(df, 'SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D', 0.0)
        validated_sm_div = (sm_div * (0.5 + emo_extreme * 0.5 + (1.0 - turn_stab) * 0.5)).fillna(0.0).astype(np.float32)
        validated_pf_div = (pf_div * (0.5 + emo_extreme * 0.5 + (1.0 - data_qual) * 0.5)).fillna(0.0).astype(np.float32)
        def _compute_kinematics(series: pd.Series) -> pd.Series:
            k_score = pd.Series(0.0, index=df_index, dtype=np.float32)
            for w in [5, 13, 21, 34, 55]:
                r_std_0 = series.rolling(window=w, min_periods=1).std().fillna(0.0) + eps
                slope = series.diff(w).fillna(0.0) / r_std_0
                r_std_1 = slope.rolling(window=w, min_periods=1).std().fillna(0.0) + eps
                accel = slope.diff(w).fillna(0.0) / r_std_1
                r_std_2 = accel.rolling(window=w, min_periods=1).std().fillna(0.0) + eps
                jerk = accel.diff(w).fillna(0.0) / r_std_2
                k_score += (slope * 0.5 + accel * 0.3 + jerk * 0.2) * 0.2
            return k_score
        kin_sm_div = _compute_kinematics(validated_sm_div)
        kin_pf_div = _compute_kinematics(validated_pf_div)
        kin_risk = (np.tanh(kin_sm_div * 0.5 + kin_pf_div * 0.5)).clip(lower=0.0).fillna(0.0).astype(np.float32)
        sell_lg = self.helper._get_safe_series(df, 'sell_lg_amount_D', 0.0)
        days_peak = self.helper._get_safe_series(df, 'days_since_last_peak_D', 0.0).clip(lower=0.0)
        hab_risk_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        for w in [13, 21, 34, 55]:
            r_mean = sell_lg.rolling(window=w, min_periods=1).mean().fillna(0.0)
            r_std = sell_lg.rolling(window=w, min_periods=1).std().fillna(0.0) + eps
            breakout = (sell_lg > (r_mean + 1.5 * r_std)).astype(np.float32)
            hab_risk_score += pd.Series(np.where(breakout > 0.0, 1.0, 0.5), index=df_index).astype(np.float32)
        hab_risk_score = (hab_risk_score / 4.0).astype(np.float32)
        time_decay = (1.0 / np.log1p(days_peak + 1.0)).astype(np.float32)
        hab_risk_mem = (hab_risk_score * time_decay).fillna(0.0).astype(np.float32)
        winner_rate = self.helper._get_safe_series(df, 'winner_rate_D', 0.5)
        turnover_rate = self.helper._get_safe_series(df, 'turnover_rate_f_D', 0.0)
        winner_nl = (1.0 / (1.0 + np.exp(-10.0 * (winner_rate - 0.5)))).astype(np.float32)
        turnover_ma = turnover_rate.rolling(window=21, min_periods=1).mean().fillna(0.0) + eps
        turnover_nl = np.tanh(turnover_rate / turnover_ma).astype(np.float32)
        phase_overheat = (winner_nl * 0.5 + turnover_nl * 0.5).astype(np.float32)
        risk_resonance = (kin_risk > 0.85) & (hab_risk_mem > 0.85) & (phase_overheat > 0.85)
        risk_explosion = pd.Series(np.where(risk_resonance, np.exp(1.5), 1.0), index=df_index).astype(np.float32)
        flow_cons = self.helper._get_safe_series(df, 'flow_consistency_D', 0.0)
        dyn_w_risk = pd.Series(np.where(flow_cons < 0.3, 1.5, 0.8), index=df_index).astype(np.float32)
        veto_trigger = (large_anomaly > 0.85) | (sm_div > 0.85) | (kin_risk > 0.85)
        veto_penalty = pd.Series(np.where(veto_trigger, 0.4, 1.0), index=df_index).astype(np.float32)
        trend_stage = self.helper._get_safe_series(df, 'STATE_TRENDING_STAGE_D', 0.0)
        vol_adj_conc = self.helper._get_safe_series(df, 'volatility_adjusted_concentration_D', 0.0)
        regime_risk_factor = pd.Series(np.where(trend_stage > 0.6, 0.7, np.where(trend_stage < 0.3, 1.0 + vol_adj_conc * 0.5, 1.0)), index=df_index).astype(np.float32)
        conflict_smooth = (1.0 / (1.0 + pf_div.clip(lower=0.0))).astype(np.float32)
        rev_prob = self.helper._get_safe_series(df, 'reversal_prob_D', 0.0)
        parab_warn = self.helper._get_safe_series(df, 'STATE_PARABOLIC_WARNING_D', 0.0)
        reflexivity_imm = np.exp(-1.5 * (rev_prob * 0.6 + parab_warn * 0.4)).astype(np.float32)
        val_score = self.helper._get_safe_series(df, 'validation_score_D', 100.0)
        emergency_stop = pd.Series(np.where(val_score < 40.0, 0.0, 1.0), index=df_index).astype(np.float32)
        calibrated_base = (holo_score - baseline).fillna(0.0).astype(np.float32)
        base_penalty = ((1.0 - turn_stab) * 0.2 + (1.0 - data_qual) * 0.2).astype(np.float32)
        raw_exponent = (1.0 - calibrated_base + base_penalty + kin_risk * dyn_w_risk + hab_risk_mem * 0.5).astype(np.float32)
        adjusted_exponent = (raw_exponent * regime_risk_factor * risk_explosion).clip(0.1, 10.0).astype(np.float32)
        sign_base = np.sign(calibrated_base).astype(np.float32)
        abs_base = np.abs(calibrated_base).astype(np.float32)
        base_calibrated = (sign_base * (abs_base ** adjusted_exponent)).astype(np.float32)
        final_adjusted = (base_calibrated * veto_penalty * conflict_smooth * reflexivity_imm * emergency_stop).fillna(0.0).clip(-1.0, 1.0).astype(np.float32)
        debug_dict = {
            "validated_sm_div": validated_sm_div,
            "kin_risk": kin_risk,
            "hab_risk_mem": hab_risk_mem,
            "phase_overheat": phase_overheat,
            "risk_explosion": risk_explosion,
            "dyn_w_risk": dyn_w_risk,
            "veto_penalty": veto_penalty,
            "regime_risk_factor": regime_risk_factor,
            "conflict_smooth": conflict_smooth,
            "reflexivity_imm": reflexivity_imm,
            "adjusted_exponent": adjusted_exponent,
            "emergency_stop": emergency_stop,
            "final_adjusted": final_adjusted
        }
        return final_adjusted, debug_dict

    def _print_debug_info(self, method_name: str, probe_ts: pd.Timestamp, debug_output: Dict, debug_dict: Dict, final_score: pd.Series):
        """【V5.1.0】极客化无损打印，真实映射矩阵断点。"""
        for section, values in debug_dict.items():
            debug_output[f"  -> [{section}]"] = ""
            for k, v in values.items():
                debug_output[f"      {k}: {v}"] = ""
        debug_output[f"  => 拆单吸筹微观动能最终输出: {final_score.loc[probe_ts] if probe_ts in final_score.index else np.nan}"] = ""
        self.helper._print_debug_output(debug_output)