# strategies\trend_following\intelligence\process\calculate_price_momentum_divergence.py
import pandas as pd
import numpy as np
from typing import Dict
from strategies.trend_following.utils import get_param_value
from strategies.trend_following.intelligence.process.helper import ProcessIntelligenceHelper
class CalculatePriceMomentumDivergence:
    """
    PROCESS_META_PRICE_VS_MOMENTUM_DIVERGENCE - V16.0.0 量子相空间双极张量背离引擎
    升级重点：
    1. 彻底纠正单极截断Bug：废除abs()和clip(0,1)，基于张量错位精准还原[-1, 1]双极性(Bipolar)信号。
    2. 拆除防御性数据掩码：废除fillna、兜底值及分母1e-8平滑，允许除零异常透传为NaN/Inf，倒逼上游数据对齐。
    3. 数学模型升维：由一维线性对比跃升为运动学张量、能量耗散张量与流形拓扑张量的三维相空间计算。
    4. 军械库火力扩容：引入MACDh、RSI、CMF、BBP、均线橡皮筋等高阶指标，实现多维共振检验。
    """
    def __init__(self, strategy_instance, helper_instance: ProcessIntelligenceHelper):
        self.strategy = strategy_instance
        self.helper = helper_instance
        self.debug_params = getattr(self.helper, 'debug_params', {})
        self.probe_dates = getattr(self.helper, 'probe_dates', [])

    def _extract_raw(self, df: pd.DataFrame, col: str) -> pd.Series:
        return df[col]

    def _robust_norm(self, series: pd.Series, window: int = 55, k: float = 1.0) -> pd.Series:
        rmed = series.rolling(window).median()
        rmad = (series - rmed).abs().rolling(window).mean()
        return np.tanh(k * (series - rmed) / rmad)

    def _noise_gate(self, series: pd.Series, window: int) -> pd.Series:
        """v17.0.0 零基陷阱防御：自适应软门限过滤无穷小量的震荡市微积分噪音"""
        sigma = series.rolling(window, min_periods=1).std()
        return series * np.tanh(series.abs() / sigma)

    def _hab_impact(self, series: pd.Series, window: int) -> pd.Series:
        """v17.0.0 HAB系统：计算当日增量相对于历史存量绝对积分的真实冲击强度"""
        increment = series.diff()
        stock = series.abs().rolling(window, min_periods=1).sum()
        return increment / stock

    def _quantum_norm(self, series: pd.Series, window: int, power: float = 1.0) -> pd.Series:
        """v17.0.0 量子归一化：Robust Z-Score + 幂律增益，废除兜底平滑允许异常断层透传"""
        med = series.rolling(window, min_periods=1).median()
        mad = (series - med).abs().rolling(window, min_periods=1).mean()
        z = (series - med) / mad
        gained = np.sign(z) * (z.abs() ** power)
        return np.tanh(gained)

    def _calc_kinematic_divergence(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """v17.0.0 第一维度：运动学张量 (融合ROC、微积分导数与HAB冲击缓冲对撞)"""
        p_vel = self._extract_raw(df, 'MA_VELOCITY_EMA_55_D')
        m_acc = self._extract_raw(df, 'MA_ACCELERATION_EMA_55_D')
        roc = self._extract_raw(df, 'ROC_13_D')
        vel_slope = self._extract_raw(df, 'SLOPE_13_MA_VELOCITY_EMA_55_D')
        roc_accel = self._extract_raw(df, 'ACCEL_21_ROC_13_D')
        m_acc_jerk = self._extract_raw(df, 'JERK_34_MA_ACCELERATION_EMA_55_D')
        gated_vel_slope = self._noise_gate(vel_slope, 13)
        gated_roc_accel = self._noise_gate(roc_accel, 21)
        gated_m_acc_jerk = self._noise_gate(m_acc_jerk, 34)
        hab_p_vel = self._hab_impact(p_vel, 34)
        hab_roc = self._hab_impact(roc, 21)
        n_pvel = self._quantum_norm(p_vel, 34, 1.2)
        n_macc = self._quantum_norm(m_acc, 21, 1.0)
        n_roc = self._quantum_norm(roc, 21, 1.5)
        n_vel_slope = self._quantum_norm(gated_vel_slope, 13, 1.2)
        n_roc_accel = self._quantum_norm(gated_roc_accel, 21, 1.2)
        n_m_acc_jerk = self._quantum_norm(gated_m_acc_jerk, 34, 1.0)
        n_hab_p_vel = self._quantum_norm(hab_p_vel, 34, 1.0)
        n_hab_roc = self._quantum_norm(hab_roc, 21, 1.0)
        cpv = n_pvel * 0.6 + n_vel_slope * 0.4
        cmv = n_macc * 0.3 + n_roc * 0.3 + n_roc_accel * 0.2 + n_m_acc_jerk * 0.2
        kin_div = (cpv - cmv) * cpv.abs() * (1.0 + n_hab_p_vel + n_hab_roc)
        return {"DIM": kin_div, "CPV": cpv, "CMV": cmv, "P_VEL": p_vel, "M_ACC": m_acc, "ROC": roc}

    def _calc_energy_hollowness(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """v17.0.0 第二维度：能量结构耗散 (主力净额资金池错配与高阶换手熵增崩塌)"""
        vpa_eff = self._extract_raw(df, 'VPA_EFFICIENCY_D')
        p_entropy = self._extract_raw(df, 'PRICE_ENTROPY_D')
        net_mf = self._extract_raw(df, 'net_mf_amount_D')
        turnover = self._extract_raw(df, 'turnover_rate_D')
        dist_conf = self._extract_raw(df, 'intraday_distribution_confidence_D')
        net_mf_slope = self._extract_raw(df, 'SLOPE_13_net_mf_amount_D')
        turnover_accel = self._extract_raw(df, 'ACCEL_21_turnover_rate_D')
        dist_jerk = self._extract_raw(df, 'JERK_34_intraday_distribution_confidence_D')
        gated_mf_slope = self._noise_gate(net_mf_slope, 13)
        gated_turnover_accel = self._noise_gate(turnover_accel, 21)
        gated_dist_jerk = self._noise_gate(dist_jerk, 34)
        hab_net_mf = self._hab_impact(net_mf, 34)
        hab_turnover = self._hab_impact(turnover, 21)
        n_vpa = self._quantum_norm(vpa_eff, 34, 1.0)
        n_ent = self._quantum_norm(p_entropy, 34, 2.0)
        n_net_mf = self._quantum_norm(net_mf, 34, 1.5)
        n_dist = self._quantum_norm(dist_conf, 21, 1.2)
        n_mf_slope = self._quantum_norm(gated_mf_slope, 13, 1.2)
        n_turn_accel = self._quantum_norm(gated_turnover_accel, 21, 1.2)
        n_dist_jerk = self._quantum_norm(gated_dist_jerk, 34, 1.0)
        n_hab_mf = self._quantum_norm(hab_net_mf, 34, 1.0)
        n_hab_turn = self._quantum_norm(hab_turnover, 21, 1.0)
        capital_vector = n_net_mf * 0.5 + n_mf_slope * 0.5
        struct_vector = n_ent * 0.4 + n_dist * 0.3 + n_turn_accel * 0.15 + n_dist_jerk * 0.15
        ene_mismatch = -capital_vector * (1.0 + n_hab_mf)
        struct_decay = struct_vector * (1.0 + n_hab_turn) - n_vpa
        hollowness = (ene_mismatch * 0.5 + struct_decay * 0.5) * capital_vector.abs()
        return {"DIM": hollowness, "ENE_MISMATCH": ene_mismatch, "STRUCT_DECAY": struct_decay, "VPA_EFF": vpa_eff, "P_ENTROPY": p_entropy, "NET_MF": net_mf, "TURNOVER": turnover}

    def _calc_geometric_tension(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """v17.0.0 第三维度：流形拓扑张量 (乖离率极值撕裂与几何曲率引力断裂)"""
        g_slope = self._extract_raw(df, 'GEOM_REG_SLOPE_D')
        g_r2 = self._extract_raw(df, 'GEOM_REG_R2_D')
        bias = self._extract_raw(df, 'BIAS_21_D')
        arc = self._extract_raw(df, 'GEOM_ARC_CURVATURE_D')
        bias_slope = self._extract_raw(df, 'SLOPE_13_BIAS_21_D')
        g_slope_accel = self._extract_raw(df, 'ACCEL_21_GEOM_REG_SLOPE_D')
        arc_jerk = self._extract_raw(df, 'JERK_34_GEOM_ARC_CURVATURE_D')
        gated_bias_slope = self._noise_gate(bias_slope, 13)
        gated_slope_accel = self._noise_gate(g_slope_accel, 21)
        gated_arc_jerk = self._noise_gate(arc_jerk, 34)
        hab_bias = self._hab_impact(bias, 55)
        n_slope = self._quantum_norm(g_slope, 55, 1.2)
        n_bias = self._quantum_norm(bias, 21, 1.5)
        n_arc = self._quantum_norm(arc, 34, 1.5)
        n_bias_slope = self._quantum_norm(gated_bias_slope, 13, 1.2)
        n_slope_accel = self._quantum_norm(gated_slope_accel, 21, 1.2)
        n_arc_jerk = self._quantum_norm(gated_arc_jerk, 34, 1.0)
        n_hab_bias = self._quantum_norm(hab_bias, 55, 1.0)
        tension = n_slope * 0.3 + n_bias * 0.3 + n_arc * 0.2 + n_bias_slope * 0.1 + n_slope_accel * 0.05 + n_arc_jerk * 0.05
        geom_div = tension * g_r2 * (1.0 + n_hab_bias)
        return {"DIM": geom_div, "TENSION": tension, "G_SLOPE": g_slope, "G_R2": g_r2, "BIAS": bias, "ARC": arc}

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """v17.0.0 主计算入口：相空间坍缩与Gamma极化 (适配最新量子微积分张量结构)"""
        p = config.get('price_momentum_divergence_params', {})
        res_kin = self._calc_kinematic_divergence(df)
        res_ene = self._calc_energy_hollowness(df)
        res_geo = self._calc_geometric_tension(df)
        snt = self._extract_raw(df, 'market_sentiment_score_D')
        n_snt = self._quantum_norm(snt, 55, p.get('sentiment_polarization_factor', 1.2))
        tensor_core = (res_kin['DIM'] * 0.4 + res_ene['DIM'] * 0.3 + res_geo['DIM'] * 0.3)
        final_score = (tensor_core * (1.0 + n_snt))
        final_score = np.sign(final_score) * (np.abs(final_score) ** p.get('tensor_folding_power', 1.8))
        final_score = final_score.clip(-1.0, 1.0).astype(np.float32)
        probe_ts = next((d for d in reversed(df.index) if pd.to_datetime(d).tz_localize(None).normalize() in [pd.to_datetime(pd_date).normalize() for pd_date in self.probe_dates]), None) if self.probe_dates else None
        self._execute_holographic_probe(probe_ts, {"KINEMATIC": res_kin, "ENERGY": res_ene, "GEOMETRIC": res_geo, "SENTIMENT": {"RAW": snt, "NORM": n_snt}, "RESULT": {"FINAL": final_score}})
        return final_score

    def _execute_holographic_probe(self, probe_ts, data_map: Dict):
        if probe_ts is None or not getattr(self.helper, '_print_debug_output', None): return
        output = {f"--- PROCESS_META_PRICE_VS_MOMENTUM_DIVERGENCE_V16_PROBE @ {probe_ts.strftime('%Y-%m-%d')} ---": ""}
        for dim_name, metrics in data_map.items():
            output[f"[{dim_name}]"] = ""
            for k, v in metrics.items():
                if isinstance(v, pd.Series):
                    val = v.loc[probe_ts] if probe_ts in v.index else 'NaN_OR_MISSING'
                else:
                    val = v
                output[f"  -> {k}: {val}"] = ""
        self.helper._print_debug_output(output)












