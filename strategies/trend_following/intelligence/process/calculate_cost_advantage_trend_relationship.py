# strategies\trend_following\intelligence\process\calculate_cost_advantage_trend_relationship.py
# 【V11.0.0 · 全息成本资金五维共振计算器】 已升级pro
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from strategies.trend_following.utils import get_params_block, get_param_value
from strategies.trend_following.intelligence.process.helper import ProcessIntelligenceHelper
class CalculateCostAdvantageTrendRelationship:
    """
    【V12.0.0 · 全息成本资金五维张量共振计算器】
    PROCESS_META_COST_ADVANTAGE_TREND
    - 核心职责: 基于物理场超平面模型诊断趋势爆发力与持续性。
    - 升级重点: 彻底消灭0值死锁，引入HAB存量缓冲记忆，应用谐振子、洛伦兹力与卡诺机效率模型。
    """
    def __init__(self, strategy_instance, helper_instance: ProcessIntelligenceHelper):
        self.strategy = strategy_instance
        self.helper = helper_instance
        self.params = self.helper.params
        self.debug_params = self.helper.debug_params
        self.probe_dates = self.helper.probe_dates
        self.version = "V12.0.0"
    def _initialize_debug_context(self, method_name: str, df: pd.DataFrame) -> Tuple[bool, Optional[pd.Timestamp], Dict, Dict]:
        is_debug = get_param_value(self.debug_params.get('enabled'), False)
        probe_ts = None
        if is_debug and not df.empty:
            target_dates_str = set()
            if self.probe_dates:
                target_dates_str = set(pd.to_datetime(d).strftime('%Y-%m-%d') for d in self.probe_dates)
            for date in reversed(df.index):
                current_date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
                if current_date_str in target_dates_str:
                    probe_ts = date
                    break
            if probe_ts is None and target_dates_str:
                probe_ts = df.index[-1]
                print(f"!!! [探针修正] 指定日期未匹配，强制使用最后一天: {pd.to_datetime(probe_ts).strftime('%Y-%m-%d')} !!!")
        debug_output = {}
        temp_vals = {}
        if is_debug and probe_ts is not None:
            ts_display = pd.to_datetime(probe_ts).strftime('%Y-%m-%d')
            debug_output[f"--- {method_name} Probe @ {ts_display} ---"] = ""
        return is_debug and (probe_ts is not None), probe_ts, debug_output, temp_vals
    def _probe_val(self, key: str, val: Any, temp_vals: Dict, section: str = "General"):
        if section not in temp_vals:
            temp_vals[section] = {}
        if isinstance(val, (pd.Series, np.ndarray, float, int)):
            try:
                temp_vals[section][key] = f"{float(val):.4f}"
            except:
                temp_vals[section][key] = str(val)
        else:
            temp_vals[section][key] = str(val)
    def _soft_norm(self, series: pd.Series, window: int = 21) -> pd.Series:
        roll_median = series.rolling(window=window, min_periods=1).median()
        roll_mad = series.rolling(window=window, min_periods=1).std() * 0.6745
        roll_mad = roll_mad.replace(0.0, 1e-8).fillna(1e-8)
        z_score = (series - roll_median) / roll_mad
        return np.tanh(z_score / 3.0)
    def _calc_hab_impact(self, series: pd.Series, window: int) -> pd.Series:
        hab_stock = series.abs().rolling(window=window, min_periods=1).mean()
        impact = series / (hab_stock + 1e-8)
        return np.tanh(impact / 3.0)
    def _get_kinematics(self, df: pd.DataFrame, base_series: pd.Series, col_name: str, lookback: int, temp_vals: Dict, section: str) -> Tuple[pd.Series, pd.Series, pd.Series]:
        slope_col = f"SLOPE_{lookback}_{col_name}"
        accel_col = f"ACCEL_{lookback}_{col_name}"
        jerk_col = f"JERK_{lookback}_{col_name}"
        if slope_col in df.columns:
            slope = df[slope_col].fillna(0.0)
        else:
            slope = base_series.diff(lookback).fillna(0.0)
        if accel_col in df.columns:
            accel = df[accel_col].fillna(0.0)
        else:
            accel = slope.diff(max(1, lookback // 2)).fillna(0.0)
        if jerk_col in df.columns:
            jerk = df[jerk_col].fillna(0.0)
        else:
            jerk = accel.diff(max(1, lookback // 3)).fillna(0.0)
        gate_thresh = 1e-5
        slope = pd.Series(np.where(np.abs(slope) < gate_thresh, 0.0, slope), index=df.index)
        accel = pd.Series(np.where(np.abs(accel) < gate_thresh, 0.0, accel), index=df.index)
        jerk = pd.Series(np.where(np.abs(jerk) < gate_thresh, 0.0, jerk), index=df.index)
        if temp_vals is not None:
            self._probe_val(f"{col_name}_SLOPE", slope.iloc[-1] if not slope.empty else 0.0, temp_vals, section)
            self._probe_val(f"{col_name}_ACCEL", accel.iloc[-1] if not accel.empty else 0.0, temp_vals, section)
            self._probe_val(f"{col_name}_JERK", jerk.iloc[-1] if not jerk.empty else 0.0, temp_vals, section)
        return slope, accel, jerk
    def _check_and_repair_signals(self, df: pd.DataFrame, method_name: str) -> pd.DataFrame:
        required_cols = [
            'winner_rate_D', 'chip_stability_D', 'chip_concentration_ratio_D', 'SMART_MONEY_HM_NET_BUY_D',
            'SMART_MONEY_HM_COORDINATED_ATTACK_D', 'SMART_MONEY_SYNERGY_BUY_D', 'MA_VELOCITY_EMA_55_D',
            'VPA_EFFICIENCY_D', 'market_sentiment_score_D', 'cost_50pct_D', 'cost_5pct_D', 'cost_95pct_D',
            'close_D', 'chip_entropy_D', 'turnover_rate_f_D', 'profit_pressure_D', 'pressure_trapped_D',
            'net_energy_flow_D', 'tick_large_order_net_D', 'stealth_flow_ratio_D', 'uptrend_strength_D',
            'downtrend_strength_D', 'pressure_release_index_D', 'chip_flow_intensity_D', 'PRICE_ENTROPY_D',
            'GEOM_REG_R2_D', 'GEOM_REG_SLOPE_D', 'ADX_14_D', 'SMART_MONEY_INST_NET_BUY_D'
        ]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"!!! CRITICAL WARNING: Missing Base Columns {missing} in {method_name} !!!")
            for col in missing:
                df[col] = 0.0
        return df
    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        method_name = "CalculateCostAdvantage_V12"
        print(f" ====== Start CalculateCostAdvantageTrendRelationship【V12.0.0】启动五维成本资金共振计算，数据形状: {df.shape} ======")
        is_debug, probe_ts, debug_out, temp_vals = self._initialize_debug_context(method_name, df)
        df_processed = self._check_and_repair_signals(df.copy(), method_name)
        df_index = df_processed.index
        D1 = self._calculate_chip_barrier_solidity(df_processed, df_index, is_debug, probe_ts, temp_vals)
        D2 = self._calculate_predator_attack_vector(df_processed, df_index, is_debug, probe_ts, temp_vals)
        D3 = self._calculate_kinematic_efficiency(df_processed, df_index, is_debug, probe_ts, temp_vals)
        D4 = self._calculate_cost_migration_elasticity(df_processed, df_index, is_debug, probe_ts, temp_vals)
        D5 = self._calculate_structure_negentropy(df_processed, df_index, is_debug, probe_ts, temp_vals)
        final_score = self._calculate_pentagonal_resonance(D1, D2, D3, D4, D5, df_processed, df_index, is_debug, probe_ts, temp_vals)
        if is_debug:
            self._log_debug_values(debug_out, temp_vals, probe_ts, method_name)
        print(f" ====== End【V12.0.0】五维成本资金共振计算完成 ======")
        return final_score.astype(np.float32).fillna(0.0)
    def _calculate_chip_barrier_solidity(self, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        concentration = df['chip_concentration_ratio_D']
        stability = df['chip_stability_D']
        cost_50 = df['cost_50pct_D']
        close = df['close_D']
        pressure_release = df['pressure_release_index_D']
        k_spring = self._soft_norm(concentration, 34) + 1.0
        displacement = (close - cost_50) / (cost_50 + 1e-8)
        potential_energy = 0.5 * k_spring * (displacement ** 2)
        pe_norm = np.tanh(potential_energy * 10.0)
        hab_stability = self._calc_hab_impact(stability, 55)
        slope_conc, accel_conc, jerk_conc = self._get_kinematics(df, concentration, 'chip_concentration_ratio_D', 13, temp_vals if is_debug and probe_ts in idx else None, "D1_ChipSolidity")
        kinematic_mod = np.tanh(slope_conc * 0.5 + accel_conc * 0.3) * np.exp(-(jerk_conc * 5.0) ** 2)
        base_score = pe_norm * 0.4 + hab_stability * 0.4 + kinematic_mod * 0.2 + np.tanh(pressure_release - 0.5) * 0.1
        final_d1 = pd.Series(np.tanh(base_score), index=idx)
        if is_debug and probe_ts in idx:
            self._probe_val("K_Spring", k_spring.loc[probe_ts], temp_vals, "D1_ChipSolidity")
            self._probe_val("Potential_Energy", potential_energy.loc[probe_ts], temp_vals, "D1_ChipSolidity")
            self._probe_val("Final_D1", final_d1.loc[probe_ts], temp_vals, "D1_ChipSolidity")
        return final_d1
    def _calculate_predator_attack_vector(self, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        hm_buy = df['SMART_MONEY_HM_NET_BUY_D']
        synergy_buy = df['SMART_MONEY_SYNERGY_BUY_D']
        tick_net = df['tick_large_order_net_D']
        stealth = df['stealth_flow_ratio_D']
        coord_attack = df['SMART_MONEY_HM_COORDINATED_ATTACK_D']
        e_field_raw = hm_buy + synergy_buy + tick_net
        hab_e_field = self._calc_hab_impact(e_field_raw, 21)
        b_field = self._soft_norm(stealth, 21)
        v_price, _, _ = self._get_kinematics(df, df['close_D'], 'close_D', 5, temp_vals if is_debug and probe_ts in idx else None, "D2_PredatorAttack")
        v_norm = self._soft_norm(v_price, 21)
        q_charge = 1.0 + coord_attack
        lorentz_force = q_charge * (hab_e_field + v_norm * b_field)
        slope_sm, accel_sm, jerk_sm = self._get_kinematics(df, hm_buy, 'SMART_MONEY_HM_NET_BUY_D', 13, temp_vals if is_debug and probe_ts in idx else None, "D2_PredatorAttack")
        kinematic_mod = np.tanh(slope_sm * 0.5 + accel_sm * 0.3) * np.exp(-(jerk_sm * 3.0) ** 2)
        base_score = lorentz_force * 0.7 + kinematic_mod * 0.3
        final_d2 = pd.Series(np.tanh(base_score), index=idx)
        if is_debug and probe_ts in idx:
            self._probe_val("E_Field", hab_e_field.loc[probe_ts], temp_vals, "D2_PredatorAttack")
            self._probe_val("Lorentz_Force", pd.Series(lorentz_force, index=idx).loc[probe_ts], temp_vals, "D2_PredatorAttack")
            self._probe_val("Final_D2", final_d2.loc[probe_ts], temp_vals, "D2_PredatorAttack")
        return final_d2
    def _calculate_kinematic_efficiency(self, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        vpa_eff = df['VPA_EFFICIENCY_D']
        entropy = df['PRICE_ENTROPY_D']
        uptrend = df['uptrend_strength_D']
        downtrend = df['downtrend_strength_D']
        velocity = df['MA_VELOCITY_EMA_55_D']
        net_energy = df['net_energy_flow_D']
        hab_energy = self._calc_hab_impact(net_energy, 34)
        hab_vpa = self._calc_hab_impact(vpa_eff, 21)
        t_c = entropy
        t_h = np.maximum(uptrend, downtrend) + 1.0
        carnot_eta = pd.Series(np.maximum(0.0, 1.0 - (t_c / t_h)), index=idx)
        raw_work = hab_energy * hab_vpa
        useful_work = raw_work * carnot_eta
        slope_v, accel_v, jerk_v = self._get_kinematics(df, velocity, 'MA_VELOCITY_EMA_55_D', 13, temp_vals if is_debug and probe_ts in idx else None, "D3_KinematicEff")
        kinematic_mod = np.tanh(slope_v * 0.5 + accel_v * 0.3)
        base_score = useful_work * 0.7 + kinematic_mod * 0.3
        final_d3 = pd.Series(np.tanh(base_score * np.sign(velocity)), index=idx)
        if is_debug and probe_ts in idx:
            self._probe_val("Carnot_Eta", carnot_eta.loc[probe_ts], temp_vals, "D3_KinematicEff")
            self._probe_val("Useful_Work", pd.Series(useful_work, index=idx).loc[probe_ts], temp_vals, "D3_KinematicEff")
            self._probe_val("Final_D3", final_d3.loc[probe_ts], temp_vals, "D3_KinematicEff")
        return final_d3
    def _calculate_cost_migration_elasticity(self, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        turnover = df['turnover_rate_f_D']
        profit_pres = df['profit_pressure_D']
        cost_50 = df['cost_50pct_D']
        chip_flow = df['chip_flow_intensity_D']
        slope_c50, accel_c50, jerk_c50 = self._get_kinematics(df, cost_50, 'cost_50pct_D', 13, temp_vals if is_debug and probe_ts in idx else None, "D4_Elasticity")
        hab_turnover = self._calc_hab_impact(turnover, 21)
        stress = hab_turnover + np.tanh(profit_pres / 10.0)
        strain = slope_c50 / (cost_50 + 1e-8)
        raw_modulus = stress / (np.abs(strain) + 1e-8)
        modulus_norm = np.tanh(raw_modulus)
        flow_mod = np.tanh(chip_flow)
        base_score = modulus_norm * 0.6 + flow_mod * 0.4
        deadlock_risk = pd.Series(np.where((stress > 1.0) & (flow_mod < -0.5), -1.0, 1.0), index=idx)
        final_d4 = pd.Series(np.tanh(base_score * deadlock_risk), index=idx)
        if is_debug and probe_ts in idx:
            self._probe_val("Stress", pd.Series(stress, index=idx).loc[probe_ts], temp_vals, "D4_Elasticity")
            self._probe_val("Strain", pd.Series(strain, index=idx).loc[probe_ts], temp_vals, "D4_Elasticity")
            self._probe_val("Final_D4", final_d4.loc[probe_ts], temp_vals, "D4_Elasticity")
        return final_d4
    def _calculate_structure_negentropy(self, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        chip_entropy = df['chip_entropy_D']
        price_entropy = df['PRICE_ENTROPY_D']
        reg_r2 = df['GEOM_REG_R2_D']
        reg_slope = df['GEOM_REG_SLOPE_D']
        total_micro_entropy = chip_entropy * 0.5 + price_entropy * 0.5
        hab_entropy = total_micro_entropy.rolling(window=34, min_periods=1).mean()
        entropy_delta = hab_entropy - total_micro_entropy
        negentropy = np.tanh(entropy_delta * 5.0)
        order_param = np.sqrt(np.abs(negentropy * reg_r2)) * np.sign(negentropy)
        slope_ent, _, _ = self._get_kinematics(df, chip_entropy, 'chip_entropy_D', 13, temp_vals if is_debug and probe_ts in idx else None, "D5_Negentropy")
        kinetics_mod = 1.0 - np.tanh(slope_ent)
        trend_gate = np.tanh(reg_slope * 10.0)
        base_score = order_param * 0.6 + kinetics_mod * 0.4
        final_d5 = pd.Series(np.tanh(base_score * np.sign(trend_gate)), index=idx)
        if is_debug and probe_ts in idx:
            self._probe_val("Negentropy", pd.Series(negentropy, index=idx).loc[probe_ts], temp_vals, "D5_Negentropy")
            self._probe_val("Order_Param", pd.Series(order_param, index=idx).loc[probe_ts], temp_vals, "D5_Negentropy")
            self._probe_val("Final_D5", final_d5.loc[probe_ts], temp_vals, "D5_Negentropy")
        return final_d5
    def _calculate_pentagonal_resonance(self, D1: pd.Series, D2: pd.Series, D3: pd.Series, D4: pd.Series, D5: pd.Series, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        close = df['close_D']
        adx = df['ADX_14_D']
        def soft_prod(a, b): return (a * b) / (1.0 + np.abs(a * b))
        i12 = soft_prod(D1, D2)
        i23 = soft_prod(D2, D3)
        i34 = soft_prod(D3, D4)
        i45 = soft_prod(D4, D5)
        i51 = soft_prod(D5, D1)
        tensor_vol = (i12 + i23 + i34 + i45 + i51) / 5.0
        w1, w2, w3, w4, w5 = 0.2, 0.3, 0.2, 0.15, 0.15
        linear_score = D1*w1 + D2*w2 + D3*w3 + D4*w4 + D5*w5
        base_resonance = linear_score + tensor_vol * 1.5
        res_diff = base_resonance.diff(8).fillna(0.0)
        price_diff = close.diff(8).fillna(0.0)
        roll_cov = res_diff.rolling(13, min_periods=1).cov(price_diff).fillna(0.0)
        roll_var_res = res_diff.rolling(13, min_periods=1).var().fillna(0.0)
        roll_var_price = price_diff.rolling(13, min_periods=1).var().fillna(0.0)
        raw_corr = roll_cov / (np.sqrt(roll_var_res * roll_var_price) + 1e-8)
        price_slope, _, _ = self._get_kinematics(df, close, 'close_D', 13, temp_vals if is_debug and probe_ts in idx else None, "Fusion")
        reflexivity_factor = pd.Series(np.where(price_slope > 0.0, np.maximum(1.0, 1.0 + raw_corr * 0.5), np.minimum(1.0, 1.0 + raw_corr * 0.5)), index=idx)
        core_score = base_resonance * reflexivity_factor
        market_temp = np.tanh(adx / 50.0)
        dyn_threshold = 0.6 - (market_temp - 0.5) * 0.4
        dyn_gamma = 2.0 + (market_temp - 0.5) * 2.0
        gate_val = core_score - dyn_threshold
        apt_score = pd.Series(np.where(gate_val > 0.0, core_score * (1.0 + 0.6 * np.power(gate_val, 1.2) * dyn_gamma), core_score), index=idx)
        final_normalized_score = pd.Series(np.tanh(apt_score), index=idx)
        if is_debug and probe_ts in idx:
            self._probe_val("Tensor_Volume", tensor_vol.loc[probe_ts], temp_vals, "D6_Fusion")
            self._probe_val("Raw_Corr", raw_corr.loc[probe_ts], temp_vals, "D6_Fusion")
            self._probe_val("Reflexivity", reflexivity_factor.loc[probe_ts], temp_vals, "D6_Fusion")
            self._probe_val("APT_Score", apt_score.loc[probe_ts], temp_vals, "D6_Fusion")
            self._probe_val("Final_Score", final_normalized_score.loc[probe_ts], temp_vals, "D6_Fusion")
        return final_normalized_score
    def _log_debug_values(self, debug_out: Dict, temp_vals: Dict, probe_ts: pd.Timestamp, method_name: str):
        print(f"\n====== {method_name} 全链路探针输出 @ {pd.to_datetime(probe_ts).strftime('%Y-%m-%d')} ======")
        for section, data in temp_vals.items():
            print(f"[{section}]")
            for k, v in data.items():
                print(f"  {k:<20}: {v}")
        print("========================================================================\n")















