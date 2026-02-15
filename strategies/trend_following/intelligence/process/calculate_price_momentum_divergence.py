import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from strategies.trend_following.utils import get_params_block, get_param_value, _robust_geometric_mean
from strategies.trend_following.intelligence.process.helper import ProcessIntelligenceHelper
class CalculatePriceMomentumDivergence:
    """
    【PROCESS_META_PRICE_VS_MOMENTUM_DIVERGENCE - V5.0.0 相空间张量高密折叠版】
    重构说明：彻底废除防御性代码掩盖，内嵌自适应张量压缩防线。将DQWM与HAB多维体系折叠为极致向量化矩阵。
    融合三阶动力学、HAB双向蓄水池与反身性指数熔断，精准捕获顶底背离物理学特征。
    """
    def __init__(self, strategy_instance, helper_instance: ProcessIntelligenceHelper):
        self.strategy = strategy_instance
        self.helper = helper_instance
        self.debug_params = self.helper.debug_params
        self.probe_dates = getattr(self.helper, 'probe_dates', [])
    def _z_norm(self, series: pd.Series, window: int = 55, mode: str = 'unipolar') -> pd.Series:
        """【量纲压缩器】强行收束至安全张量空间，剥离fillna暴露底层极差归零断层"""
        rmin = series.rolling(window, min_periods=1).min()
        rmax = series.rolling(window, min_periods=1).max()
        norm = (series - rmin) / (rmax - rmin + 1e-6)
        return norm.clip(0.0, 1.0).astype(np.float32) if mode == 'unipolar' else (norm * 2.0 - 1.0).astype(np.float32)
    def _compute_kinematics(self, series: pd.Series, df_index: pd.Index) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """【物理层】三阶张量求导：计算V/A/J，利用动态标准差提取波函数，无掩盖暴露0方差"""
        v_fused = pd.Series(0.0, index=df_index, dtype=np.float32)
        a_fused = pd.Series(0.0, index=df_index, dtype=np.float32)
        j_fused = pd.Series(0.0, index=df_index, dtype=np.float32)
        for w in [5, 13, 21, 34]:
            r_std0 = series.rolling(w, min_periods=1).std() + 1e-6
            slope = series.diff(w) / r_std0
            r_std1 = slope.rolling(w, min_periods=1).std() + 1e-6
            accel = slope.diff(w) / r_std1
            r_std2 = accel.rolling(w, min_periods=1).std() + 1e-6
            jerk = accel.diff(w) / r_std2
            v_fused += slope * 0.4; a_fused += accel * 0.3; j_fused += jerk * 0.3
        return v_fused.astype(np.float32), a_fused.astype(np.float32), j_fused.astype(np.float32)
    def _calculate_phase_space_divergence(self, df: pd.DataFrame, df_index: pd.Index, params: Dict) -> pd.Series:
        """【能量层】3D相空间角散度：计算价格向量与动量向量的三维空间点积，并融合军械库原生背离簇"""
        p_params = params.get('phase_space_params', {})
        vw = p_params.get('velocity_weight', 0.6)
        aw = p_params.get('acceleration_weight', 0.4)
        jw = 0.2
        v_p, a_p, j_p = self._compute_kinematics(self.helper._get_safe_series(df, 'close_D', np.nan), df_index)
        macd = self.helper._get_safe_series(df, 'MACDh_13_34_8_D', np.nan)
        rsi = self.helper._get_safe_series(df, 'RSI_13_D', np.nan)
        v_m, a_m, j_m = self._compute_kinematics(macd * 0.6 + rsi * 0.4, df_index)
        dot_product = (v_p * v_m * vw) + (a_p * a_m * aw) + (j_p * j_m * jw)
        mag_p = np.sqrt(v_p**2 * vw + a_p**2 * aw + j_p**2 * jw) + 1e-6
        mag_m = np.sqrt(v_m**2 * vw + a_m**2 * aw + j_m**2 * jw) + 1e-6
        cos_theta = (dot_product / (mag_p * mag_m)).clip(-1.0, 1.0)
        raw_div = (1.0 - cos_theta) * 0.5
        price_dir = np.sign(v_p.rolling(5, min_periods=1).mean())
        base_div = price_dir * (raw_div ** p_params.get('angular_sensitivity', 1.5))
        arsenal_w = params.get("arsenal_native_divergence_weights", {})
        pct_div = self._z_norm(self.helper._get_safe_series(df, 'percent_change_divergence_D', np.nan), mode='bipolar')
        div_str = self._z_norm(self.helper._get_safe_series(df, 'divergence_strength_D', np.nan), mode='bipolar')
        net_mig = self._z_norm(self.helper._get_safe_series(df, 'net_migration_direction_D', np.nan), mode='bipolar')
        press_rel = self._z_norm(self.helper._get_safe_series(df, 'pressure_release_index_D', np.nan), mode='bipolar')
        inflow_pers = self._z_norm(self.helper._get_safe_series(df, 'inflow_persistence_D', np.nan), mode='bipolar')
        native_div = (pct_div * arsenal_w.get('percent_change_divergence_D', 0.3) + div_str * arsenal_w.get('divergence_strength_D', 0.3) + net_mig * arsenal_w.get('net_migration_direction_D', 0.2) + press_rel * arsenal_w.get('pressure_release_index_D', 0.1) + inflow_pers * arsenal_w.get('inflow_persistence_D', 0.1))
        return (base_div * 0.6 + native_div * 0.4).astype(np.float32)
    def _calculate_dense_dqwm_matrix(self, df: pd.DataFrame, df_index: pd.Index, params: Dict) -> pd.Series:
        """【张量融合】DQWM高密折叠矩阵：动量品质、张力、稳定性、筹码，应用木桶短板暴露缺陷"""
        mom_q = self._z_norm(self.helper._get_safe_series(df, 'RSI_13_D', np.nan)) * 0.5 + self._z_norm(self.helper._get_safe_series(df, 'MACDh_13_34_8_D', np.nan), mode='bipolar').abs() * 0.5
        tension = 1.0 - self._z_norm(self.helper._get_safe_series(df, 'BIAS_55_D', np.nan).abs())
        stability = self._z_norm(self.helper._get_safe_series(df, 'TURNOVER_STABILITY_INDEX_D', np.nan))
        chip_pot = self._z_norm(self.helper._get_safe_series(df, 'accumulation_score_D', np.nan))
        matrix = pd.DataFrame({'mq': mom_q, 'ten': tension, 'stb': stability, 'cp': chip_pot})
        min_score = matrix.min(axis=1)
        penalty = np.where(min_score < 0.3, min_score * 2.0, 1.0)
        w = params.get('dqwm_weights', {})
        weighted_sum = mom_q * w.get('momentum_quality', 0.25) + tension * w.get('market_tension', 0.25) + stability * w.get('stability', 0.25) + chip_pot * w.get('chip_potential', 0.25)
        return (weighted_sum * penalty).clip(0.0, 1.0).astype(np.float32)
    def _calculate_hab_confirmations(self, df: pd.DataFrame, df_index: pd.Index, base_div: pd.Series) -> pd.Series:
        """【HAB存量记忆】量价主力双向确认：基于真实大单流入流出的历史均值穿透，不掩盖零方差"""
        vpa_eff = self._z_norm(self.helper._get_safe_series(df, 'VPA_EFFICIENCY_D', np.nan))
        cmf = self._z_norm(self.helper._get_safe_series(df, 'CMF_21_D', np.nan), mode='bipolar')
        sell_lg = self.helper._get_safe_series(df, 'sell_lg_amount_D', np.nan)
        buy_lg = self.helper._get_safe_series(df, 'buy_lg_amount_D', np.nan)
        sell_hab = sell_lg / (sell_lg.rolling(21, min_periods=1).mean() + 1e-6)
        buy_hab = buy_lg / (buy_lg.rolling(21, min_periods=1).mean() + 1e-6)
        is_bull = base_div < 0
        hab_impact = pd.Series(np.where(is_bull, buy_hab, sell_hab), index=df_index).clip(0.0, 3.0) / 3.0
        cmf_impact = pd.Series(np.where(is_bull, cmf.clip(lower=0.0), cmf.clip(upper=0.0).abs()), index=df_index)
        return (vpa_eff * 0.4 + hab_impact * 0.4 + cmf_impact * 0.2).clip(0.0, 1.0).astype(np.float32)
    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【主中枢干线】执行全息张量对撞、动态环境Gamma极化与反身性免疫的无损级联，探针全量暴晒"""
        method_name = "calculate_price_momentum_divergence"
        is_debug = get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug and self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            for date in reversed(df.index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date; break
        df_index = df.index
        params = config.get('price_momentum_divergence_params', {})
        phase_div = self._calculate_phase_space_divergence(df, df_index, params)
        dqwm = self._calculate_dense_dqwm_matrix(df, df_index, params)
        confirm = self._calculate_hab_confirmations(df, df_index, phase_div)
        base_fusion = phase_div * dqwm * confirm
        emo_extreme = self._z_norm(self.helper._get_safe_series(df, 'STATE_EMOTIONAL_EXTREME_D', np.nan))
        large_anomaly = self._z_norm(self.helper._get_safe_series(df, 'large_order_anomaly_D', np.nan))
        pf_div = self._z_norm(self.helper._get_safe_series(df, 'price_flow_divergence_D', np.nan))
        sm_div = self._z_norm(self.helper._get_safe_series(df, 'SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D', np.nan))
        is_bull = base_fusion < 0
        veto_trigger = (large_anomaly > 0.8) | (sm_div > 0.8) | (pf_div > 0.8)
        veto_penalty = pd.Series(np.where(veto_trigger & is_bull, 0.4, np.where(veto_trigger & ~is_bull, 1.5, 1.0)), index=df_index).astype(np.float32)
        trend_stage = self._z_norm(self.helper._get_safe_series(df, 'STATE_TRENDING_STAGE_D', np.nan))
        vol_adj_conc = self._z_norm(self.helper._get_safe_series(df, 'volatility_adjusted_concentration_D', np.nan))
        regime_factor = pd.Series(np.where(trend_stage > 0.6, np.where(is_bull, 1.2, 0.7), np.where(trend_stage < 0.3, 1.0 + vol_adj_conc * 0.5, 1.0)), index=df_index).astype(np.float32)
        conflict_smooth = (1.0 / (1.0 + pf_div)).astype(np.float32)
        rev_prob = self._z_norm(self.helper._get_safe_series(df, 'reversal_prob_D', np.nan))
        parab_warn = self._z_norm(self.helper._get_safe_series(df, 'STATE_PARABOLIC_WARNING_D', np.nan))
        reflexivity_imm = pd.Series(np.where(is_bull, np.exp(-1.5 * (rev_prob * 0.6 + parab_warn * 0.4 + emo_extreme * 0.2)), np.exp(1.5 * (rev_prob * 0.6 + parab_warn * 0.4 + emo_extreme * 0.2))), index=df_index).astype(np.float32)
        val_score = self.helper._get_safe_series(df, 'validation_score_D', np.nan)
        emergency_stop = pd.Series(np.where((val_score < 40.0) & (val_score > 0.1), 0.0, 1.0), index=df_index).astype(np.float32)
        raw_exponent = (1.0 + pf_div * 0.5 + sm_div * 0.5).astype(np.float32)
        gamma = params.get('final_fusion_exponent', 1.5)
        adjusted_exp = pd.Series(np.where(is_bull, gamma * raw_exponent * regime_factor, gamma * raw_exponent * regime_factor), index=df_index).clip(0.1, 5.0).astype(np.float32)
        final_score = (np.sign(base_fusion) * (np.abs(base_fusion) ** adjusted_exp) * veto_penalty * conflict_smooth * reflexivity_imm * emergency_stop).fillna(0.0).clip(-1.0, 1.0).astype(np.float32)
        if probe_ts is not None:
            debug_out = {f"--- {method_name} V5.0.0 满级物理背离引擎探针 @ {probe_ts.strftime('%Y-%m-%d')} ---": ""}
            d_dict = {"phase_div": phase_div, "dqwm_matrix": dqwm, "hab_confirm": confirm, "base_fusion": base_fusion, "emo_extreme": emo_extreme, "large_anomaly": large_anomaly, "pf_div": pf_div, "sm_div": sm_div, "veto_penalty": veto_penalty, "regime_factor": regime_factor, "conflict_smooth": conflict_smooth, "reflexivity_imm": reflexivity_imm, "adjusted_exp": adjusted_exp, "emergency_stop": emergency_stop, "final_score": final_score}
            for k, v in d_dict.items():
                debug_out[f"  -> {k}: {v.loc[probe_ts] if probe_ts in v.index else np.nan}"] = ""
            self.helper._print_debug_output(debug_out)
        return final_score