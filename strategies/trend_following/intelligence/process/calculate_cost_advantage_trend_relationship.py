# strategies\trend_following\intelligence\process\calculate_cost_advantage_trend_relationship.py
# 【V11.0.0 · 全息成本资金五维共振计算器】 已升级pro
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from strategies.trend_following.utils import get_params_block, get_param_value
from strategies.trend_following.intelligence.process.helper import ProcessIntelligenceHelper
class CalculateCostAdvantageTrendRelationship:
    """
    【V18.0.0 · 强类型物理装甲防崩版】
    PROCESS_META_COST_ADVANTAGE_TREND
    - 用途: 基于高维物理场模型（谐振子势垒/洛伦兹偏转/卡诺热机/杨氏模量/结构熵）对筹码与资金的爆发力进行全息诊断。
    - 本次修改要点:
      1. [类型塌陷死锁修复] 彻底解决因 np.where 等底层 C 级函数运算时剥离 Pandas 时序索引，导致 ndarray 无法使用 .loc 寻址从而引发战术引擎崩溃的致命 BUG。
      2. [时空骨架加固] 所有向量计算中间态（特别是张量代理、反身性门控、极性算子）的裸数组产物，全部使用 pd.Series(..., index=idx) 进行时空骨架的二次熔铸。
    """
    def __init__(self, strategy_instance, helper_instance: ProcessIntelligenceHelper):
        self.strategy = strategy_instance
        self.helper = helper_instance
        self.params = self.helper.params
        self.debug_params = self.helper.debug_params
        self.probe_dates = self.helper.probe_dates
        self.version = "V18.0.0"
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
    def _center_and_scale(self, series: pd.Series, window: int = 21) -> pd.Series:
        roll_median = series.rolling(window=window, min_periods=1).median()
        roll_mad = series.rolling(window=window, min_periods=1).std() * 0.6745
        roll_mad = roll_mad.replace(0.0, 1e-8).fillna(1e-8)
        z_score = (series - roll_median) / roll_mad
        return pd.Series(np.tanh(z_score / 3.0), index=series.index)
    def _scale_by_volatility(self, series: pd.Series, window: int = 21) -> pd.Series:
        roll_std = series.rolling(window=window, min_periods=1).std().replace(0.0, 1e-8).fillna(1e-8)
        return pd.Series(np.tanh(series / (roll_std * 2.0)), index=series.index)
    def _calc_hab_impact(self, series: pd.Series, window: int) -> pd.Series:
        hab_stock = series.abs().rolling(window=window, min_periods=1).mean().replace(0.0, 1e-8).fillna(1e-8)
        impact = series / hab_stock
        return pd.Series(np.tanh(impact / 3.0), index=series.index)
    def _get_kinematics(self, df: pd.DataFrame, base_series: pd.Series, col_name: str, lookback: int, temp_vals: Optional[Dict], section: str) -> Tuple[pd.Series, pd.Series, pd.Series]:
        slope_col = f"SLOPE_{lookback}_{col_name}"
        accel_col = f"ACCEL_{lookback}_{col_name}"
        jerk_col = f"JERK_{lookback}_{col_name}"
        if slope_col in df.columns:
            slope = df[slope_col].fillna(0.0)
        else:
            slope = base_series.diff(lookback).fillna(0.0)
        accel_lookback = max(1, int(lookback / 1.618))
        if accel_col in df.columns:
            accel = df[accel_col].fillna(0.0)
        else:
            accel = slope.diff(accel_lookback).fillna(0.0)
        jerk_lookback = max(1, int(accel_lookback / 1.618))
        if jerk_col in df.columns:
            jerk = df[jerk_col].fillna(0.0)
        else:
            jerk = accel.diff(jerk_lookback).fillna(0.0)
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
        fallback_close = df['close_D'] if 'close_D' in df.columns and not df.empty else pd.Series(10.0, index=df.index)
        default_map = {
            'winner_rate_D': 50.0, 'chip_stability_D': 1.0, 'chip_concentration_ratio_D': 0.2,
            'SMART_MONEY_HM_NET_BUY_D': 0.0, 'SMART_MONEY_HM_COORDINATED_ATTACK_D': 0.0,
            'SMART_MONEY_SYNERGY_BUY_D': 0.0, 'MA_VELOCITY_EMA_55_D': 0.0, 'VPA_EFFICIENCY_D': 0.0,
            'market_sentiment_score_D': 50.0, 'cost_50pct_D': fallback_close, 'cost_5pct_D': fallback_close * 0.8,
            'cost_95pct_D': fallback_close * 1.2, 'close_D': 10.0, 'chip_entropy_D': 0.5, 'turnover_rate_f_D': 1.0,
            'profit_pressure_D': 0.0, 'pressure_trapped_D': 0.0, 'net_energy_flow_D': 0.0, 'tick_large_order_net_D': 0.0,
            'stealth_flow_ratio_D': 0.0, 'uptrend_strength_D': 50.0, 'downtrend_strength_D': 50.0,
            'pressure_release_index_D': 50.0, 'chip_flow_intensity_D': 0.0, 'PRICE_ENTROPY_D': 0.5,
            'GEOM_REG_R2_D': 0.5, 'GEOM_REG_SLOPE_D': 0.0, 'ADX_14_D': 20.0
        }
        missing = [col for col in default_map.keys() if col not in df.columns]
        if missing:
            print(f"!!! CRITICAL WARNING: Missing Columns {missing} in {method_name}. Filling sensible defaults. !!!")
            for col in missing:
                df[col] = default_map[col]
        return df
    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        method_name = f"CalculateCostAdvantage_{self.version}"
        print(f" ====== Start {method_name} 启动强类型装甲张量计算，数据形状: {df.shape} ======")
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
        print(f" ====== End {method_name} 强类型装甲张量计算完成 ======")
        return final_score.astype(np.float32).fillna(0.0)
    def _calculate_chip_barrier_solidity(self, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        concentration = df['chip_concentration_ratio_D']
        stability = df['chip_stability_D']
        cost_50 = df['cost_50pct_D']
        close = df['close_D']
        pressure_release = df['pressure_release_index_D']
        k_spring = pd.Series(self._center_and_scale(concentration, 34) + 1.0, index=idx)
        displacement = pd.Series((close - cost_50) / (cost_50 + 1e-8), index=idx)
        norm_disp = pd.Series(displacement * 5.0, index=idx)
        disp_sign = pd.Series(np.where(displacement >= 0, 1.0, -1.0), index=idx)
        pe_raw = pd.Series(0.5 * k_spring * (norm_disp ** 2) * disp_sign, index=idx)
        pe_norm = pd.Series(np.tanh(pe_raw), index=idx)
        hab_stability = pd.Series(self._calc_hab_impact(stability, 55), index=idx)
        pr_norm = pd.Series(np.tanh((pressure_release - 50.0) / 20.0), index=idx)
        slope_conc, accel_conc, jerk_conc = self._get_kinematics(df, concentration, 'chip_concentration_ratio_D', 13, temp_vals if is_debug and probe_ts in idx else None, "D1_ChipSolidity")
        kinematic_mod = pd.Series(np.tanh(self._scale_by_volatility(slope_conc, 21) * 0.5 + self._scale_by_volatility(accel_conc, 21) * 0.3), index=idx)
        base_score = pd.Series(pe_norm * 0.4 + hab_stability * 0.3 + pr_norm * 0.2 + kinematic_mod * 0.1, index=idx)
        final_d1 = pd.Series(np.tanh(base_score), index=idx)
        if is_debug and probe_ts in idx:
            self._probe_val("K_Spring", k_spring.loc[probe_ts], temp_vals, "D1_ChipSolidity")
            self._probe_val("Potential_Energy", pe_raw.loc[probe_ts], temp_vals, "D1_ChipSolidity")
            self._probe_val("Final_D1", final_d1.loc[probe_ts], temp_vals, "D1_ChipSolidity")
        return final_d1
    def _calculate_predator_attack_vector(self, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        hm_buy = df['SMART_MONEY_HM_NET_BUY_D']
        synergy_buy = df['SMART_MONEY_SYNERGY_BUY_D']
        tick_net = df['tick_large_order_net_D']
        stealth = df['stealth_flow_ratio_D']
        coord_attack = df['SMART_MONEY_HM_COORDINATED_ATTACK_D']
        e_field_raw = pd.Series(hm_buy + synergy_buy + tick_net, index=idx)
        hab_e_field = pd.Series(self._calc_hab_impact(e_field_raw, 21), index=idx)
        b_field = pd.Series(self._center_and_scale(stealth, 21), index=idx)
        v_price, _, _ = self._get_kinematics(df, df['close_D'], 'close_D', 5, temp_vals if is_debug and probe_ts in idx else None, "D2_PredatorAttack")
        v_norm = pd.Series(self._scale_by_volatility(v_price, 21), index=idx)
        q_charge = pd.Series(1.0 + np.tanh(coord_attack), index=idx)
        lorentz_force = pd.Series(q_charge * (hab_e_field + v_norm * b_field), index=idx)
        primary = pd.Series(np.tanh(lorentz_force), index=idx)
        slope_sm, accel_sm, jerk_sm = self._get_kinematics(df, hm_buy, 'SMART_MONEY_HM_NET_BUY_D', 13, temp_vals if is_debug and probe_ts in idx else None, "D2_PredatorAttack")
        kinematic_mod = pd.Series(np.tanh(self._scale_by_volatility(slope_sm, 21) * 0.5 + self._scale_by_volatility(accel_sm, 21) * 0.3), index=idx)
        base_score = pd.Series(primary * (1.0 + np.sign(primary) * kinematic_mod * 0.5).clip(0.0, 2.0), index=idx)
        final_d2 = pd.Series(np.tanh(base_score), index=idx)
        if is_debug and probe_ts in idx:
            self._probe_val("E_Field", hab_e_field.loc[probe_ts], temp_vals, "D2_PredatorAttack")
            self._probe_val("Lorentz_Force", lorentz_force.loc[probe_ts], temp_vals, "D2_PredatorAttack")
            self._probe_val("Final_D2", final_d2.loc[probe_ts], temp_vals, "D2_PredatorAttack")
        return final_d2
    def _calculate_kinematic_efficiency(self, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        vpa_eff = df['VPA_EFFICIENCY_D']
        entropy = df['PRICE_ENTROPY_D']
        uptrend = df['uptrend_strength_D']
        downtrend = df['downtrend_strength_D']
        velocity = df['MA_VELOCITY_EMA_55_D']
        net_energy = df['net_energy_flow_D']
        hab_energy = pd.Series(self._calc_hab_impact(net_energy, 34), index=idx)
        hab_vpa = pd.Series(self._calc_hab_impact(vpa_eff, 21), index=idx)
        t_c = pd.Series(entropy.fillna(0.5).clip(lower=0.1, upper=1.0), index=idx)
        t_h = pd.Series(1.0 + np.maximum(uptrend.fillna(50.0), downtrend.fillna(50.0)) / 100.0, index=idx)
        carnot_eta = pd.Series(np.maximum(0.0, 1.0 - (t_c / (t_h + 1e-8))), index=idx)
        useful_work = pd.Series(hab_energy * hab_vpa.abs() * carnot_eta, index=idx)
        primary = pd.Series(np.tanh(useful_work), index=idx)
        slope_v, accel_v, jerk_v = self._get_kinematics(df, velocity, 'MA_VELOCITY_EMA_55_D', 13, temp_vals if is_debug and probe_ts in idx else None, "D3_KinematicEff")
        kinematic_mod = pd.Series(np.tanh(self._scale_by_volatility(slope_v, 21) * 0.5 + self._scale_by_volatility(accel_v, 21) * 0.3), index=idx)
        base_score = pd.Series(primary * (1.0 + np.sign(primary) * kinematic_mod * 0.5).clip(0.0, 2.0), index=idx)
        final_d3 = pd.Series(np.tanh(base_score), index=idx)
        if is_debug and probe_ts in idx:
            self._probe_val("T_Cold_Clipped", t_c.loc[probe_ts], temp_vals, "D3_KinematicEff")
            self._probe_val("T_Hot_Scaled", t_h.loc[probe_ts], temp_vals, "D3_KinematicEff")
            self._probe_val("Carnot_Eta", carnot_eta.loc[probe_ts], temp_vals, "D3_KinematicEff")
            self._probe_val("Useful_Work", useful_work.loc[probe_ts], temp_vals, "D3_KinematicEff")
            self._probe_val("Final_D3", final_d3.loc[probe_ts], temp_vals, "D3_KinematicEff")
        return final_d3
    def _calculate_cost_migration_elasticity(self, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        turnover = df['turnover_rate_f_D']
        profit_pres = df['profit_pressure_D']
        trapped_pres = df['pressure_trapped_D']
        cost_50 = df['cost_50pct_D']
        chip_flow = df['chip_flow_intensity_D']
        slope_c50, _, _ = self._get_kinematics(df, cost_50, 'cost_50pct_D', 13, temp_vals if is_debug and probe_ts in idx else None, "D4_Elasticity")
        hab_turnover = pd.Series(self._calc_hab_impact(turnover, 21).clip(lower=0.0), index=idx)
        profit_eff = pd.Series(np.maximum(0.0, np.tanh(profit_pres / 50.0)), index=idx)
        trapped_eff = pd.Series(np.maximum(0.0, np.tanh(trapped_pres / 50.0)), index=idx)
        stress_raw = pd.Series(hab_turnover + profit_eff + trapped_eff, index=idx)
        stress = pd.Series(np.maximum(0.1, stress_raw), index=idx)
        strain = pd.Series((slope_c50 / (cost_50 + 1e-8)) * 2.0, index=idx)
        compliance = pd.Series(strain / stress, index=idx)
        primary = pd.Series(np.tanh(compliance), index=idx)
        flow_mod = pd.Series(np.tanh(self._scale_by_volatility(chip_flow, 21)), index=idx)
        base_score = pd.Series(primary * (1.0 + np.sign(primary) * flow_mod * 0.5).clip(0.0, 2.0), index=idx)
        final_d4 = pd.Series(np.tanh(base_score), index=idx)
        if is_debug and probe_ts in idx:
            self._probe_val("Stress", stress.loc[probe_ts], temp_vals, "D4_Elasticity")
            self._probe_val("Strain_Ratio", strain.loc[probe_ts], temp_vals, "D4_Elasticity")
            self._probe_val("Final_D4", final_d4.loc[probe_ts], temp_vals, "D4_Elasticity")
        return final_d4
    def _calculate_structure_negentropy(self, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        chip_entropy = df['chip_entropy_D']
        price_entropy = df['PRICE_ENTROPY_D']
        reg_r2 = df['GEOM_REG_R2_D']
        reg_slope = df['GEOM_REG_SLOPE_D']
        total_micro_entropy = pd.Series(chip_entropy * 0.5 + price_entropy * 0.5, index=idx)
        hab_entropy = pd.Series(total_micro_entropy.rolling(window=34, min_periods=1).mean(), index=idx)
        negentropy = pd.Series(np.tanh((hab_entropy - total_micro_entropy) * 5.0), index=idx)
        negent_sign = pd.Series(np.where(negentropy >= 0, 1.0, -1.0), index=idx)
        primary = pd.Series(negent_sign * np.sqrt(np.abs(negentropy * reg_r2)), index=idx)
        slope_ent, _, _ = self._get_kinematics(df, chip_entropy, 'chip_entropy_D', 13, temp_vals if is_debug and probe_ts in idx else None, "D5_Negentropy")
        kinetics_mod = pd.Series(-np.tanh(self._scale_by_volatility(slope_ent, 21)), index=idx)
        primary_sign = pd.Series(np.where(primary >= 0, 1.0, -1.0), index=idx)
        core_negentropy = pd.Series(primary * (1.0 + primary_sign * kinetics_mod * 0.5).clip(0.0, 2.0), index=idx)
        trend_gate = pd.Series(np.tanh(reg_slope * 10.0), index=idx)
        divergence_penalty = pd.Series(np.where((core_negentropy < 0) & (trend_gate > 0), 1.0 + np.abs(trend_gate), 1.0), index=idx)
        alignment_multiplier = pd.Series(np.where(core_negentropy * trend_gate > 0, 1.0 + np.abs(trend_gate) * 0.5, 1.0), index=idx)
        base_score = pd.Series(core_negentropy * divergence_penalty * alignment_multiplier, index=idx)
        final_d5 = pd.Series(np.tanh(base_score), index=idx)
        if is_debug and probe_ts in idx:
            self._probe_val("Negentropy", negentropy.loc[probe_ts], temp_vals, "D5_Negentropy")
            self._probe_val("Order_Param", primary.loc[probe_ts], temp_vals, "D5_Negentropy")
            self._probe_val("Core_Negentropy", core_negentropy.loc[probe_ts], temp_vals, "D5_Negentropy")
            self._probe_val("Final_D5", final_d5.loc[probe_ts], temp_vals, "D5_Negentropy")
        return final_d5
    def _calculate_pentagonal_resonance(self, D1: pd.Series, D2: pd.Series, D3: pd.Series, D4: pd.Series, D5: pd.Series, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        close = df['close_D']
        adx = df['ADX_14_D']
        def tensor_proxy(a: pd.Series, b: pd.Series) -> pd.Series:
            a_sign = pd.Series(np.where(a > 0, 1.0, -1.0), index=idx)
            raw_val = np.where(a * b > 0, a_sign * np.sqrt(np.abs(a * b)), -np.sqrt(np.abs(a * b)))
            return pd.Series(raw_val, index=idx)
        i12 = tensor_proxy(D1, D2)
        i23 = tensor_proxy(D2, D3)
        i34 = tensor_proxy(D3, D4)
        i45 = tensor_proxy(D4, D5)
        i51 = tensor_proxy(D5, D1)
        tensor_vol = pd.Series((i12 + i23 + i34 + i45 + i51) / 5.0, index=idx)
        w1, w2, w3, w4, w5 = 0.2, 0.3, 0.2, 0.15, 0.15
        linear_score = pd.Series(D1*w1 + D2*w2 + D3*w3 + D4*w4 + D5*w5, index=idx)
        base_resonance = pd.Series(linear_score + tensor_vol * 1.5, index=idx)
        res_diff = pd.Series(base_resonance.diff(8).fillna(0.0), index=idx)
        price_diff = pd.Series(close.diff(8).fillna(0.0), index=idx)
        roll_cov = pd.Series(res_diff.rolling(13, min_periods=1).cov(price_diff).fillna(0.0), index=idx)
        roll_var_res = pd.Series(res_diff.rolling(13, min_periods=1).var().fillna(0.0), index=idx)
        roll_var_price = pd.Series(price_diff.rolling(13, min_periods=1).var().fillna(0.0), index=idx)
        raw_corr = pd.Series(roll_cov / (np.sqrt(np.maximum(0.0, roll_var_res * roll_var_price)) + 1e-8), index=idx)
        price_slope, _, _ = self._get_kinematics(df, close, 'close_D', 13, temp_vals if is_debug and probe_ts in idx else None, "Fusion")
        price_dir = pd.Series(np.where(np.abs(price_slope / (close + 1e-8)) > 0.005, np.sign(price_slope), 0.0), index=idx)
        alignment_raw = np.where(base_resonance * price_dir > 0, 1.0, np.where(base_resonance * price_dir < 0, -1.0, 0.0))
        alignment = pd.Series(alignment_raw, index=idx)
        reflexivity_raw = np.where(alignment > 0, 1.0 + np.abs(raw_corr) * 0.5, np.where(alignment < 0, 1.0 / (1.0 + np.abs(raw_corr) * 0.5), 1.0))
        reflexivity_factor = pd.Series(reflexivity_raw, index=idx)
        core_score = pd.Series(base_resonance * reflexivity_factor, index=idx)
        market_temp = pd.Series(np.tanh(adx / 50.0), index=idx)
        dyn_threshold = pd.Series(0.6 - (market_temp - 0.5) * 0.4, index=idx)
        dyn_gamma = pd.Series(2.0 + (market_temp - 0.5) * 2.0, index=idx)
        gate_val = pd.Series(np.abs(core_score) - dyn_threshold, index=idx)
        safe_gate = pd.Series(np.maximum(0.0, gate_val), index=idx)
        apt_raw = np.where(gate_val > 0.0, core_score * (1.0 + 0.6 * np.power(safe_gate, 1.2) * dyn_gamma), core_score)
        apt_score = pd.Series(apt_raw, index=idx)
        final_normalized_score = pd.Series(np.tanh(apt_score), index=idx)
        if is_debug and probe_ts in idx:
            self._probe_val("Tensor_Volume", tensor_vol.loc[probe_ts], temp_vals, "D6_Fusion")
            self._probe_val("Raw_Corr", raw_corr.loc[probe_ts], temp_vals, "D6_Fusion")
            self._probe_val("Alignment", alignment.loc[probe_ts], temp_vals, "D6_Fusion")
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













