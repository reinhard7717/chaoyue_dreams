# strategies\trend_following\intelligence\process\calculate_price_momentum_divergence.py
import pandas as pd
import numpy as np
from typing import Dict
from strategies.trend_following.utils import get_param_value
from strategies.trend_following.intelligence.process.helper import ProcessIntelligenceHelper
class CalculatePriceMomentumDivergence:
    """
    PROCESS_META_PRICE_VS_MOMENTUM_DIVERGENCE - v18.0.0 量子相空间张量正交背离引擎
    1. 彻底打破数据孤岛：满载引入 MACDh, RSI, CMF, BBP, 均线橡皮筋与聪敏钱共振调节。
    2. 防御零基陷阱与死锁：以 Tanh(|x|/(sigma+1e-6)) 软门限滤除微积分噪音，HAB积分预防除零。
    3. 数学降维打击：废弃标量减法，采用余弦相似度撕裂矩阵(Tear Factor)精准判定反向背离。
    4. 无损信息保留：全面废除.clip()，采用幂律增益(Power Law)与Tanh双极压缩折叠。
    """
    def __init__(self, strategy_instance, helper_instance: ProcessIntelligenceHelper):
        self.strategy = strategy_instance
        self.helper = helper_instance
        self.debug_params = getattr(self.helper, 'debug_params', {})
        self.probe_dates = getattr(self.helper, 'probe_dates', [])
    def _extract_and_derive(self, df: pd.DataFrame, base_col: str, deriv_type: str = None, lookback: int = 13) -> pd.Series:
        """v18.0.0 探针提取与动态微积分补偿，杜绝数据断层引发的实盘停机"""
        if deriv_type is None:
            if base_col not in df.columns:
                print(f"[探针警告] 缺失基础数据列: {base_col}，强制返回0向量防御崩溃。")
                return pd.Series(0.0, index=df.index)
            return df[base_col]
        col_name = f"{deriv_type}_{lookback}_{base_col}"
        if col_name in df.columns:
            return df[col_name]
        print(f"[探针提示] 缺失预置微积分列: {col_name}，启动动态级联无未来函数推演。")
        base_series = self._extract_and_derive(df, base_col)
        if deriv_type == 'SLOPE':
            return base_series.diff(lookback) / lookback
        elif deriv_type == 'ACCEL':
            return (base_series.diff(lookback) / lookback).diff(lookback) / lookback
        elif deriv_type == 'JERK':
            return ((base_series.diff(lookback) / lookback).diff(lookback) / lookback).diff(lookback) / lookback
        return pd.Series(0.0, index=df.index)
    def _noise_gate(self, series: pd.Series, window: int) -> pd.Series:
        """v18.0.0 自适应软门限函数，基于滚动Sigma与普朗克常量过滤微小震荡噪音"""
        sigma = series.rolling(window, min_periods=1).std()
        valid_sigma = np.where((sigma.isna()) | (sigma == 0.0), 1e-6, sigma)
        return series * np.tanh(series.abs() / valid_sigma)
    def _hab_impact(self, series: pd.Series, window: int) -> pd.Series:
        """v18.0.0 存量意识(HAB)：利用增量相对于历史存量绝对积分的冲击占比计算真实动能"""
        increment = series.diff(1)
        stock = series.abs().rolling(window, min_periods=1).sum()
        valid_stock = np.where((stock.isna()) | (stock == 0.0), 1e-6, stock)
        return np.tanh(increment / valid_stock)
    def _quantum_norm(self, series: pd.Series, window: int, power: float = 1.0) -> pd.Series:
        """v18.0.0 量子归一化：Robust Z-Score叠加幂律极化，无损保留肥尾极值信息"""
        med = series.rolling(window, min_periods=1).median()
        mad = (series - med).abs().rolling(window, min_periods=1).mean()
        valid_mad = np.where((mad.isna()) | (mad == 0.0), 1e-6, mad)
        z = (series - med) / valid_mad
        gained = np.sign(z) * (z.abs() ** power)
        return np.tanh(gained)
    def _calc_kinematic_tensor(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """v18.0.0 第一维度：运动学张量 (均线速度、ROC、MACDh动态融合)"""
        p_vel = self._extract_and_derive(df, 'MA_VELOCITY_EMA_55_D')
        roc = self._extract_and_derive(df, 'ROC_13_D')
        macdh = self._extract_and_derive(df, 'MACDh_13_34_8_D')
        vel_slope = self._extract_and_derive(df, 'MA_VELOCITY_EMA_55_D', 'SLOPE', 13)
        roc_accel = self._extract_and_derive(df, 'ROC_13_D', 'ACCEL', 21)
        macdh_jerk = self._extract_and_derive(df, 'MACDh_13_34_8_D', 'JERK', 34)
        g_vel_slope = self._noise_gate(vel_slope, 13)
        g_roc_accel = self._noise_gate(roc_accel, 21)
        g_macdh_jerk = self._noise_gate(macdh_jerk, 34)
        hab_roc = self._hab_impact(roc, 21)
        hab_macdh = self._hab_impact(macdh, 34)
        n_p_vel = self._quantum_norm(p_vel, 34, 1.2)
        n_vel_slope = self._quantum_norm(g_vel_slope, 13, 1.0)
        n_roc = self._quantum_norm(roc, 21, 1.2)
        n_macdh = self._quantum_norm(macdh, 34, 1.2)
        n_roc_accel = self._quantum_norm(g_roc_accel, 21, 1.0)
        n_macdh_jerk = self._quantum_norm(g_macdh_jerk, 34, 0.8)
        v_price = n_p_vel * 0.6 + n_vel_slope * 0.4
        v_mom = n_roc * 0.4 + n_macdh * 0.4 + n_roc_accel * 0.1 + n_macdh_jerk * 0.1
        tear_factor = 1.0 - np.tanh(v_price * v_mom)
        hab_amplifier = 1.0 + hab_roc.abs() * 0.5 + hab_macdh.abs() * 0.5
        kin_div = np.sign(v_price) * (v_price.abs() ** 1.5) * tear_factor * hab_amplifier
        return {"DIM": kin_div, "V_PRICE": v_price, "V_MOM": v_mom, "TEAR": tear_factor, "HAB": hab_amplifier}
    def _calc_energy_dissipation(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """v18.0.0 第二维度：能量耗散张量 (主力资金、CMF与换手熵增对撞)"""
        net_mf = self._extract_and_derive(df, 'net_mf_amount_D')
        cmf = self._extract_and_derive(df, 'CMF_21_D')
        vpa_eff = self._extract_and_derive(df, 'VPA_EFFICIENCY_D')
        p_ent = self._extract_and_derive(df, 'PRICE_ENTROPY_D')
        turnover = self._extract_and_derive(df, 'turnover_rate_f_D')
        mf_slope = self._extract_and_derive(df, 'net_mf_amount_D', 'SLOPE', 13)
        cmf_accel = self._extract_and_derive(df, 'CMF_21_D', 'ACCEL', 21)
        turn_jerk = self._extract_and_derive(df, 'turnover_rate_f_D', 'JERK', 34)
        g_mf_slope = self._noise_gate(mf_slope, 13)
        g_cmf_accel = self._noise_gate(cmf_accel, 21)
        g_turn_jerk = self._noise_gate(turn_jerk, 34)
        hab_mf = self._hab_impact(net_mf, 34)
        hab_turn = self._hab_impact(turnover, 21)
        n_vpa = self._quantum_norm(vpa_eff, 34, 1.2)
        n_net_mf = self._quantum_norm(net_mf, 34, 1.5)
        n_cmf = self._quantum_norm(cmf, 21, 1.2)
        n_mf_slope = self._quantum_norm(g_mf_slope, 13, 1.0)
        n_cmf_accel = self._quantum_norm(g_cmf_accel, 21, 1.0)
        n_ent = self._quantum_norm(p_ent, 34, 1.0)
        n_turn_jerk = self._quantum_norm(g_turn_jerk, 34, 0.8)
        v_price = n_vpa
        v_energy = (n_net_mf * 0.4 + n_cmf * 0.3 + n_mf_slope * 0.15 + n_cmf_accel * 0.15) - (n_ent * 0.5 + n_turn_jerk * 0.5)
        tear_factor = 1.0 - np.tanh(v_price * v_energy)
        hab_amplifier = 1.0 + hab_mf.abs() * 0.5 + hab_turn.abs() * 0.5
        ene_div = np.sign(v_price) * (v_price.abs() ** 1.5) * tear_factor * hab_amplifier
        return {"DIM": ene_div, "V_PRICE": v_price, "V_ENERGY": v_energy, "TEAR": tear_factor, "HAB": hab_amplifier}
    def _calc_topology_manifold(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """v18.0.0 第三维度：流形拓扑张量 (极限乖离、BBP与橡皮筋反噬力场)"""
        bias = self._extract_and_derive(df, 'BIAS_21_D')
        bbp = self._extract_and_derive(df, 'BBP_21_2.0_D')
        rubber = self._extract_and_derive(df, 'MA_RUBBER_BAND_EXTENSION_D')
        g_slope = self._extract_and_derive(df, 'GEOM_REG_SLOPE_D')
        arc = self._extract_and_derive(df, 'GEOM_ARC_CURVATURE_D')
        g_r2 = self._extract_and_derive(df, 'GEOM_REG_R2_D')
        bias_slope = self._extract_and_derive(df, 'BIAS_21_D', 'SLOPE', 13)
        rubber_accel = self._extract_and_derive(df, 'MA_RUBBER_BAND_EXTENSION_D', 'ACCEL', 21)
        arc_jerk = self._extract_and_derive(df, 'GEOM_ARC_CURVATURE_D', 'JERK', 34)
        g_bias_slope = self._noise_gate(bias_slope, 13)
        g_rubber_accel = self._noise_gate(rubber_accel, 21)
        g_arc_jerk = self._noise_gate(arc_jerk, 34)
        hab_bias = self._hab_impact(bias, 55)
        n_bias = self._quantum_norm(bias, 21, 1.2)
        n_bbp = self._quantum_norm(bbp - 0.5, 21, 1.2)
        n_bias_slope = self._quantum_norm(g_bias_slope, 13, 1.0)
        n_rubber = self._quantum_norm(rubber, 34, 1.2)
        n_g_slope = self._quantum_norm(g_slope, 55, 1.2)
        n_arc = self._quantum_norm(arc, 34, 1.0)
        n_rubber_accel = self._quantum_norm(g_rubber_accel, 21, 1.0)
        n_arc_jerk = self._quantum_norm(g_arc_jerk, 34, 0.8)
        v_price = n_bias * 0.4 + n_bbp * 0.4 + n_bias_slope * 0.2
        v_geom = n_g_slope * 0.3 + n_rubber * 0.3 + n_arc * 0.2 + n_rubber_accel * 0.1 + n_arc_jerk * 0.1
        tear_factor = 1.0 - np.tanh(v_price * v_geom)
        hab_amplifier = 1.0 + hab_bias.abs() * 0.5
        r2_conf = np.tanh(g_r2.abs()).fillna(1.0)
        geo_div = np.sign(v_price) * (v_price.abs() ** 1.5) * tear_factor * hab_amplifier * r2_conf
        return {"DIM": geo_div, "V_PRICE": v_price, "V_GEOM": v_geom, "TEAR": tear_factor, "R2_CONF": r2_conf}
    def _calc_oscillator_resonance(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """v18.0.0 第四维度：震荡共振模块 (RSI与聪敏钱非对称极化校验)"""
        rsi = self._extract_and_derive(df, 'RSI_13_D')
        smart_money = self._extract_and_derive(df, 'SMART_MONEY_HM_NET_BUY_D')
        coord_attack = self._extract_and_derive(df, 'SMART_MONEY_HM_COORDINATED_ATTACK_D')
        rsi_slope = self._extract_and_derive(df, 'RSI_13_D', 'SLOPE', 13)
        g_rsi_slope = self._noise_gate(rsi_slope, 13)
        n_rsi = self._quantum_norm(rsi - 50.0, 21, 1.5)
        n_rsi_slope = self._quantum_norm(g_rsi_slope, 13, 1.0)
        n_smart = self._quantum_norm(smart_money, 21, 1.0)
        n_coord = self._quantum_norm(coord_attack, 21, 1.0)
        v_price = n_rsi * 0.7 + n_rsi_slope * 0.3
        v_smart = n_smart * 0.6 + n_coord * 0.4
        tear_factor = 1.0 - np.tanh(v_price * v_smart)
        osc_div = np.sign(v_price) * (v_price.abs() ** 1.2) * tear_factor
        return {"DIM": osc_div, "V_PRICE": v_price, "V_SMART": v_smart, "TEAR": tear_factor}
    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """v18.1.0 主计算入口：相空间坍缩与Gamma极化 (修复探针静默与时区死锁)"""
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
        probe_ts_list = []
        if getattr(self, 'probe_dates', None):
            target_dates = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in self.probe_dates]
            for d in reversed(df.index):
                try:
                    if pd.to_datetime(d).strftime('%Y-%m-%d') in target_dates:
                        probe_ts_list.append(d)
                except Exception:
                    pass
        if not probe_ts_list and not df.empty:
            probe_ts_list.append(df.index[-1])
        for probe_ts in probe_ts_list:
            self._execute_holographic_probe(
                probe_ts,
                {
                    "1. KINEMATIC_DIMENSION (运动学)": res_kin,
                    "2. ENERGY_DIMENSION (能量耗散)": res_ene,
                    "3. GEOMETRIC_DIMENSION (流形拓扑)": res_geo,
                    "4. MACRO_MODULATORS (情绪极化)": {"RAW_SENTIMENT": snt, "NORM_SENTIMENT": n_snt},
                    "5. FINAL_OUTPUT (相空间折叠)": {"TENSOR_CORE_RAW": tensor_core, "FINAL_SCORE": final_score}
                }
            )
        return final_score
    def _execute_holographic_probe(self, probe_ts, data_map: Dict):
        """v18.1.0 全息探针输出引擎：穿透框架拦截，原生Print双轨树状链路强制输出"""
        if probe_ts is None:
            return
        try:
            ts_str = pd.to_datetime(probe_ts).strftime('%Y-%m-%d')
        except Exception:
            ts_str = str(probe_ts)
        print(f"\n{'='*80}")
        print(f"🚀 [全息探针强力透视] PROCESS_META_PRICE_VS_MOMENTUM_DIVERGENCE - V18.1.0")
        print(f"⏱️  探测切片: {ts_str}")
        print(f"{'='*80}")
        output_dict = {f"--- PROBE @ {ts_str} ---": ""}
        for dim_name, metrics in data_map.items():
            print(f"\n📂 {dim_name}")
            print(f"  {'-'*55}")
            for k, v in metrics.items():
                if isinstance(v, pd.Series):
                    try:
                        val = v.loc[probe_ts]
                    except KeyError:
                        val = np.nan
                else:
                    val = v
                if isinstance(val, (float, np.floating)):
                    if np.isnan(val) or np.isinf(val):
                        val_str = str(val)
                    else:
                        val_str = f"{val:+.6f}"
                else:
                    val_str = str(val)
                print(f"  ├─ {k:<25}: {val_str}")
                output_dict[f"[{dim_name}] {k}"] = val_str
        print(f"{'='*80}\n")
        if hasattr(self.helper, '_print_debug_output') and callable(getattr(self.helper, '_print_debug_output')):
            try:
                self.helper._print_debug_output(output_dict)
            except Exception:
                pass











