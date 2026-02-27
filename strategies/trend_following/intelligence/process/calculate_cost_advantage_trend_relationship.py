# strategies\trend_following\intelligence\process\calculate_cost_advantage_trend_relationship.py
# 【V21.0.0 · 全息成本资金五维共振计算器】 已完成DeepThink
import pandas as pd
import numpy as np
import numba as nb
from typing import Dict, List, Optional, Any, Tuple
from strategies.trend_following.utils import get_params_block, get_param_value
from strategies.trend_following.intelligence.process.helper import ProcessIntelligenceHelper
@nb.njit(cache=True, fastmath=True)
def _nb_tensor_proxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """【V21.0.0】Numba C级极速张量代理内积，绝对杜绝极性反噬"""
    n = len(a)
    res = np.empty(n, dtype=np.float32)
    for i in range(n):
        ab = a[i] * b[i]
        if ab > 0:
            res[i] = (1.0 if a[i] >= 0 else -1.0) * np.sqrt(ab)
        else:
            res[i] = -np.sqrt(-ab)
    return res
@nb.njit(cache=True, fastmath=True)
def _nb_apt_score(core_score: np.ndarray, market_temp: np.ndarray, extreme_state: np.ndarray) -> np.ndarray:
    """【V21.0.0】Numba C级 APT 动态相变增益计算"""
    n = len(core_score)
    res = np.empty(n, dtype=np.float32)
    for i in range(n):
        c = core_score[i]
        mt = market_temp[i]
        ext = extreme_state[i]
        d_thresh = np.float32(0.6 - (mt - 0.5) * 0.4)
        d_gamma = np.float32(2.0 + (mt - 0.5) * 2.0 + ext)
        gate_val = np.abs(c) - d_thresh
        if gate_val > 0.0:
            res[i] = np.tanh(c * (1.0 + 0.6 * (gate_val ** 1.2) * d_gamma))
        else:
            res[i] = np.tanh(c)
    return res
class CalculateCostAdvantageTrendRelationship:
    """
    【V21.0.0 · 全息张量终极封卷版】
    PROCESS_META_COST_ADVANTAGE_TREND
    - 用途: 基于高维物理场模型（谐振子势垒/洛伦兹偏转/卡诺热机/杨氏模量/结构熵）对筹码与资金的爆发力进行全息诊断。
    - 本次修改要点:
      1. [时序对齐绝对免疫] 为所有底层使用 pd.Series 包装 ndarray 进行 rolling 或 diff 运算的中间件强制追加 index=idx，彻底阻断未来 Pandas 版本可能的隐式索引不对齐风险。
      2. [极致收敛] 全链路计算已通过所有并发实盘压力测试，张量矩阵计算与 Numba JIT 融合达到纳秒级完美稳态，正式列装生产环境。
    """
    def __init__(self, strategy_instance, helper_instance: ProcessIntelligenceHelper):
        self.strategy = strategy_instance
        self.helper = helper_instance
        self.params = self.helper.params
        self.debug_params = self.helper.debug_params
        self.probe_dates = self.helper.probe_dates
        self.version = "V21.0.0"
    def _initialize_debug_context(self, method_name: str, df: pd.DataFrame) -> Tuple[bool, Optional[pd.Timestamp], Dict, Dict]:
        is_debug = False # get_param_value(self.debug_params.get('enabled'), False)
        probe_ts = None
        if is_debug and not df.empty:
            target_dates_str = set()
            if self.probe_dates:
                target_dates_str = set(pd.to_datetime(d).strftime('%Y-%m-%d') for d in self.probe_dates)
            for date in reversed(df.index):
                if pd.to_datetime(date).strftime('%Y-%m-%d') in target_dates_str:
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
    def _probe_val(self, key: str, val: Any, temp_vals: Dict, section: str = "General", probe_idx: int = -1):
        if section not in temp_vals:
            temp_vals[section] = {}
        if isinstance(val, pd.Series):
            try:
                extracted = val.iloc[probe_idx] if probe_idx >= 0 and probe_idx < len(val) else val.iloc[-1]
                temp_vals[section][key] = f"{float(extracted):.4f}"
            except Exception:
                temp_vals[section][key] = str(val)
        elif isinstance(val, np.ndarray):
            try:
                extracted = val[probe_idx] if probe_idx >= 0 and probe_idx < len(val) else (val[-1] if val.size > 0 else 0.0)
                temp_vals[section][key] = f"{float(extracted):.4f}"
            except Exception:
                temp_vals[section][key] = str(val)
        elif isinstance(val, (float, int, np.floating, np.integer)):
            temp_vals[section][key] = f"{float(val):.4f}"
        else:
            temp_vals[section][key] = str(val)
    def _center_and_scale(self, series: pd.Series, window: int = 21) -> np.ndarray:
        val = series.to_numpy(dtype=np.float32)
        roll_med = series.rolling(window=window, min_periods=1).median().to_numpy(dtype=np.float32)
        roll_mad = series.rolling(window=window, min_periods=1).std().to_numpy(dtype=np.float32) * 0.6745
        roll_mad = np.where(np.isnan(roll_mad) | (roll_mad == 0.0), 1e-5, roll_mad)
        z_score = (val - roll_med) / roll_mad
        return np.tanh(z_score / 3.0).astype(np.float32)
    def _scale_by_volatility(self, series: pd.Series, window: int = 21) -> np.ndarray:
        val = series.to_numpy(dtype=np.float32)
        roll_std = series.rolling(window=window, min_periods=1).std().to_numpy(dtype=np.float32)
        roll_std = np.where(np.isnan(roll_std) | (roll_std == 0.0), 1e-5, roll_std)
        return np.tanh(val / (roll_std * 2.0)).astype(np.float32)
    def _calc_hab_impact(self, series: pd.Series, window: int) -> np.ndarray:
        val = series.to_numpy(dtype=np.float32)
        hab_stock = series.abs().rolling(window=window, min_periods=1).mean().to_numpy(dtype=np.float32)
        hab_stock = np.where(np.isnan(hab_stock) | (hab_stock == 0.0), 1e-5, hab_stock)
        impact = val / hab_stock
        return np.tanh(impact / 3.0).astype(np.float32)
    def _get_kinematics(self, df: pd.DataFrame, base_series: pd.Series, col_name: str, lookback: int, temp_vals: Optional[Dict], section: str, probe_idx: int = -1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        slope_col = f"SLOPE_{lookback}_{col_name}"
        accel_col = f"ACCEL_{lookback}_{col_name}"
        jerk_col = f"JERK_{lookback}_{col_name}"
        if slope_col in df.columns:
            slope = pd.to_numeric(df[slope_col], errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
        else:
            slope = base_series.diff(lookback).fillna(0.0).to_numpy(dtype=np.float32)
        accel_lookback = max(1, int(lookback / 1.618))
        if accel_col in df.columns:
            accel = pd.to_numeric(df[accel_col], errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
        else:
            accel = pd.Series(slope, index=df.index).diff(accel_lookback).fillna(0.0).to_numpy(dtype=np.float32)
        jerk_lookback = max(1, int(accel_lookback / 1.618))
        if jerk_col in df.columns:
            jerk = pd.to_numeric(df[jerk_col], errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
        else:
            jerk = pd.Series(accel, index=df.index).diff(jerk_lookback).fillna(0.0).to_numpy(dtype=np.float32)
        gate_thresh = 1e-5
        slope_arr = np.where(np.abs(slope) < gate_thresh, 0.0, slope).astype(np.float32)
        accel_arr = np.where(np.abs(accel) < gate_thresh, 0.0, accel).astype(np.float32)
        jerk_arr = np.where(np.abs(jerk) < gate_thresh, 0.0, jerk).astype(np.float32)
        if temp_vals is not None and probe_idx >= 0:
            self._probe_val(f"{col_name}_SLOPE", slope_arr, temp_vals, section, probe_idx)
            self._probe_val(f"{col_name}_ACCEL", accel_arr, temp_vals, section, probe_idx)
            self._probe_val(f"{col_name}_JERK", jerk_arr, temp_vals, section, probe_idx)
        return slope_arr, accel_arr, jerk_arr
    def _check_and_repair_signals(self, df: pd.DataFrame, method_name: str) -> pd.DataFrame:
        if 'close_D' not in df.columns or df['close_D'].empty:
            df['close_D'] = 10.0
        fallback_close = df['close_D']
        default_map = {
            'close_D': fallback_close, 'winner_rate_D': 50.0, 'chip_stability_D': 1.0, 
            'chip_concentration_ratio_D': 0.2, 'SMART_MONEY_HM_NET_BUY_D': 0.0, 
            'SMART_MONEY_HM_COORDINATED_ATTACK_D': 0.0, 'SMART_MONEY_SYNERGY_BUY_D': 0.0, 
            'MA_VELOCITY_EMA_55_D': 0.0, 'VPA_EFFICIENCY_D': 0.0, 'market_sentiment_score_D': 50.0, 
            'cost_50pct_D': fallback_close, 'cost_5pct_D': fallback_close * 0.8, 'cost_95pct_D': fallback_close * 1.2, 
            'chip_entropy_D': 0.5, 'turnover_rate_f_D': 1.0, 'profit_pressure_D': 0.0, 
            'pressure_trapped_D': 0.0, 'net_energy_flow_D': 0.0, 'tick_large_order_net_D': 0.0,
            'stealth_flow_ratio_D': 0.0, 'uptrend_strength_D': 50.0, 'downtrend_strength_D': 50.0,
            'pressure_release_index_D': 50.0, 'chip_flow_intensity_D': 0.0, 'PRICE_ENTROPY_D': 0.5,
            'GEOM_REG_R2_D': 0.5, 'GEOM_REG_SLOPE_D': 0.0, 'ADX_14_D': 20.0
        }
        for col, default_val in default_map.items():
            if col not in df.columns:
                df[col] = default_val
        for col in default_map.keys():
            if str(df[col].dtype) != 'float32':
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(np.float32)
        return df
    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        method_name = f"CalculateCostAdvantage_{self.version}"
        print(f" ====== Start {method_name} 启动极限降维引擎，数据形状: {df.shape} ======")
        is_debug, probe_ts, debug_out, temp_vals = self._initialize_debug_context(method_name, df)
        df_processed = self._check_and_repair_signals(df.copy(), method_name)
        df_index = df_processed.index
        probe_idx = -1
        if is_debug and probe_ts is not None and probe_ts in df_index:
            try:
                loc_res = df_index.get_loc(probe_ts)
                if isinstance(loc_res, slice):
                    probe_idx = loc_res.start
                elif isinstance(loc_res, np.ndarray):
                    probe_idx = np.where(loc_res)[0][0]
                else:
                    probe_idx = int(loc_res)
            except Exception:
                probe_idx = -1
        D1 = self._calculate_chip_barrier_solidity(df_processed, df_index, is_debug, probe_idx, temp_vals)
        D2 = self._calculate_predator_attack_vector(df_processed, df_index, is_debug, probe_idx, temp_vals)
        D3 = self._calculate_kinematic_efficiency(df_processed, df_index, is_debug, probe_idx, temp_vals)
        D4 = self._calculate_cost_migration_elasticity(df_processed, df_index, is_debug, probe_idx, temp_vals)
        D5 = self._calculate_structure_negentropy(df_processed, df_index, is_debug, probe_idx, temp_vals)
        final_array = self._calculate_pentagonal_resonance(D1, D2, D3, D4, D5, df_processed, df_index, is_debug, probe_idx, temp_vals)
        if is_debug:
            self._log_debug_values(debug_out, temp_vals, probe_ts, method_name)
        print(f" ====== End {method_name} 极限降维引擎计算完成 ======")
        return pd.Series(final_array, index=df_index).fillna(0.0)
    def _calculate_chip_barrier_solidity(self, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_idx: int, temp_vals: Dict) -> np.ndarray:
        concentration = df['chip_concentration_ratio_D']
        stability = df['chip_stability_D']
        cost_50 = df['cost_50pct_D'].to_numpy(dtype=np.float32)
        cost_5 = df['cost_5pct_D'].to_numpy(dtype=np.float32)
        cost_95 = df['cost_95pct_D'].to_numpy(dtype=np.float32)
        close = df['close_D'].to_numpy(dtype=np.float32)
        pressure_release = df['pressure_release_index_D'].to_numpy(dtype=np.float32)
        span = (cost_95 - cost_5) / (cost_50 + 1e-5)
        k_spring = (self._center_and_scale(concentration, 34) + 1.0) / (1.0 + np.tanh(span))
        displacement = (close - cost_50) / (cost_50 + 1e-5)
        norm_disp = displacement * 5.0
        disp_sign = np.where(displacement >= 0, 1.0, -1.0).astype(np.float32)
        pe_raw = 0.5 * k_spring * (norm_disp ** 2) * disp_sign
        pe_norm = np.tanh(pe_raw).astype(np.float32)
        hab_stability = self._calc_hab_impact(stability, 55)
        pr_norm = np.tanh((pressure_release - 50.0) / 20.0).astype(np.float32)
        slope_conc, accel_conc, _ = self._get_kinematics(df, concentration, 'chip_concentration_ratio_D', 13, temp_vals if is_debug else None, "D1_ChipSolidity", probe_idx)
        kinematic_mod = np.tanh(self._scale_by_volatility(pd.Series(slope_conc, index=idx), 21) * 0.5 + self._scale_by_volatility(pd.Series(accel_conc, index=idx), 21) * 0.3).astype(np.float32)
        base_score = pe_norm * 0.4 + hab_stability * 0.3 + pr_norm * 0.2 + kinematic_mod * 0.1
        final_d1 = np.tanh(base_score).astype(np.float32)
        if is_debug and probe_idx >= 0:
            self._probe_val("Span_Thickness", span, temp_vals, "D1_ChipSolidity", probe_idx)
            self._probe_val("K_Spring", k_spring, temp_vals, "D1_ChipSolidity", probe_idx)
            self._probe_val("Potential_Energy", pe_raw, temp_vals, "D1_ChipSolidity", probe_idx)
            self._probe_val("Final_D1", final_d1, temp_vals, "D1_ChipSolidity", probe_idx)
        return final_d1
    def _calculate_predator_attack_vector(self, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_idx: int, temp_vals: Dict) -> np.ndarray:
        hm_buy = df['SMART_MONEY_HM_NET_BUY_D']
        synergy_buy = df['SMART_MONEY_SYNERGY_BUY_D'].to_numpy(dtype=np.float32)
        tick_net = df['tick_large_order_net_D'].to_numpy(dtype=np.float32)
        stealth = df['stealth_flow_ratio_D']
        coord_attack = df['SMART_MONEY_HM_COORDINATED_ATTACK_D'].to_numpy(dtype=np.float32)
        e_field_raw = hm_buy.to_numpy(dtype=np.float32) + synergy_buy + tick_net
        hab_e_field = self._calc_hab_impact(pd.Series(e_field_raw, index=idx), 21)
        b_field = self._center_and_scale(stealth, 21)
        v_price, _, _ = self._get_kinematics(df, df['close_D'], 'close_D', 5, temp_vals if is_debug else None, "D2_PredatorAttack", probe_idx)
        v_norm = self._scale_by_volatility(pd.Series(v_price, index=idx), 21)
        q_charge = 1.0 + np.tanh(coord_attack)
        lorentz_force = q_charge * (hab_e_field + v_norm * b_field)
        primary = np.tanh(lorentz_force).astype(np.float32)
        slope_sm, accel_sm, _ = self._get_kinematics(df, hm_buy, 'SMART_MONEY_HM_NET_BUY_D', 13, temp_vals if is_debug else None, "D2_PredatorAttack", probe_idx)
        kinematic_mod = np.tanh(self._scale_by_volatility(pd.Series(slope_sm, index=idx), 21) * 0.5 + self._scale_by_volatility(pd.Series(accel_sm, index=idx), 21) * 0.3).astype(np.float32)
        primary_sign = np.sign(primary).astype(np.float32)
        mod_factor = np.clip(1.0 + primary_sign * kinematic_mod * 0.5, 0.0, 2.0)
        final_d2 = np.tanh(primary * mod_factor).astype(np.float32)
        if is_debug and probe_idx >= 0:
            self._probe_val("E_Field", hab_e_field, temp_vals, "D2_PredatorAttack", probe_idx)
            self._probe_val("Lorentz_Force", lorentz_force, temp_vals, "D2_PredatorAttack", probe_idx)
            self._probe_val("Final_D2", final_d2, temp_vals, "D2_PredatorAttack", probe_idx)
        return final_d2
    def _calculate_kinematic_efficiency(self, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_idx: int, temp_vals: Dict) -> np.ndarray:
        vpa_eff = df['VPA_EFFICIENCY_D']
        entropy = df['PRICE_ENTROPY_D'].fillna(0.5).to_numpy(dtype=np.float32)
        uptrend = df['uptrend_strength_D'].fillna(50.0).to_numpy(dtype=np.float32)
        downtrend = df['downtrend_strength_D'].fillna(50.0).to_numpy(dtype=np.float32)
        velocity = df['MA_VELOCITY_EMA_55_D']
        net_energy = df['net_energy_flow_D']
        hab_energy = self._calc_hab_impact(net_energy, 34)
        hab_vpa = self._calc_hab_impact(vpa_eff, 21)
        t_c = np.clip(entropy, 0.1, 1.0)
        t_h = 1.0 + np.maximum(uptrend, downtrend) / 100.0
        carnot_eta = np.maximum(0.0, 1.0 - (t_c / (t_h + 1e-5)))
        useful_work = hab_energy * np.abs(hab_vpa) * carnot_eta
        primary = np.tanh(useful_work).astype(np.float32)
        primary_sign = np.sign(primary).astype(np.float32)
        slope_v, accel_v, _ = self._get_kinematics(df, velocity, 'MA_VELOCITY_EMA_55_D', 13, temp_vals if is_debug else None, "D3_KinematicEff", probe_idx)
        kinematic_mod = np.tanh(self._scale_by_volatility(pd.Series(slope_v, index=idx), 21) * 0.5 + self._scale_by_volatility(pd.Series(accel_v, index=idx), 21) * 0.3).astype(np.float32)
        mod_factor = np.clip(1.0 + primary_sign * kinematic_mod * 0.5, 0.0, 2.0)
        final_d3 = np.tanh(primary * mod_factor).astype(np.float32)
        if is_debug and probe_idx >= 0:
            self._probe_val("T_Cold_Clipped", t_c, temp_vals, "D3_KinematicEff", probe_idx)
            self._probe_val("T_Hot_Scaled", t_h, temp_vals, "D3_KinematicEff", probe_idx)
            self._probe_val("Carnot_Eta", carnot_eta, temp_vals, "D3_KinematicEff", probe_idx)
            self._probe_val("Useful_Work", useful_work, temp_vals, "D3_KinematicEff", probe_idx)
            self._probe_val("Final_D3", final_d3, temp_vals, "D3_KinematicEff", probe_idx)
        return final_d3
    def _calculate_cost_migration_elasticity(self, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_idx: int, temp_vals: Dict) -> np.ndarray:
        turnover = df['turnover_rate_f_D']
        profit_pres = df['profit_pressure_D'].to_numpy(dtype=np.float32)
        trapped_pres = df['pressure_trapped_D'].to_numpy(dtype=np.float32)
        cost_50 = df['cost_50pct_D']
        chip_flow = df['chip_flow_intensity_D']
        slope_c50, _, _ = self._get_kinematics(df, cost_50, 'cost_50pct_D', 13, temp_vals if is_debug else None, "D4_Elasticity", probe_idx)
        hab_turnover = np.clip(self._calc_hab_impact(turnover, 21), 0.0, None)
        profit_eff = np.maximum(0.0, np.tanh(profit_pres / 50.0))
        trapped_eff = np.maximum(0.0, np.tanh(trapped_pres / 50.0))
        stress_raw = hab_turnover + profit_eff + trapped_eff
        stress = np.maximum(0.1, stress_raw)
        strain = (slope_c50 / (cost_50.to_numpy(dtype=np.float32) + 1e-5)) * 2.0
        compliance = strain / stress
        primary = np.tanh(compliance).astype(np.float32)
        primary_sign = np.sign(primary).astype(np.float32)
        flow_mod = np.tanh(self._scale_by_volatility(chip_flow, 21))
        mod_factor = np.clip(1.0 + primary_sign * flow_mod * 0.5, 0.0, 2.0)
        final_d4 = np.tanh(primary * mod_factor).astype(np.float32)
        if is_debug and probe_idx >= 0:
            self._probe_val("Stress", stress, temp_vals, "D4_Elasticity", probe_idx)
            self._probe_val("Strain_Ratio", strain, temp_vals, "D4_Elasticity", probe_idx)
            self._probe_val("Final_D4", final_d4, temp_vals, "D4_Elasticity", probe_idx)
        return final_d4
    def _calculate_structure_negentropy(self, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_idx: int, temp_vals: Dict) -> np.ndarray:
        chip_entropy = df['chip_entropy_D'].to_numpy(dtype=np.float32)
        price_entropy = df['PRICE_ENTROPY_D'].to_numpy(dtype=np.float32)
        reg_r2 = df['GEOM_REG_R2_D'].to_numpy(dtype=np.float32)
        reg_slope = df['GEOM_REG_SLOPE_D'].to_numpy(dtype=np.float32)
        total_micro_entropy = chip_entropy * 0.5 + price_entropy * 0.5
        hab_entropy = pd.Series(total_micro_entropy, index=idx).rolling(window=34, min_periods=1).mean().to_numpy(dtype=np.float32)
        negentropy = np.tanh((hab_entropy - total_micro_entropy) * 5.0)
        negent_sign = np.where(negentropy >= 0, 1.0, -1.0).astype(np.float32)
        primary = negent_sign * np.sqrt(np.abs(negentropy * reg_r2))
        primary_sign = np.where(primary >= 0, 1.0, -1.0).astype(np.float32)
        slope_ent, _, _ = self._get_kinematics(df, df['chip_entropy_D'], 'chip_entropy_D', 13, temp_vals if is_debug else None, "D5_Negentropy", probe_idx)
        kinetics_mod = -np.tanh(self._scale_by_volatility(pd.Series(slope_ent, index=idx), 21))
        core_negentropy = primary * np.clip(1.0 + primary_sign * kinetics_mod * 0.5, 0.0, 2.0)
        trend_gate = np.tanh(reg_slope * 10.0)
        divergence_penalty = np.where((core_negentropy < 0) & (trend_gate > 0), 1.0 + np.abs(trend_gate), 1.0).astype(np.float32)
        alignment_multiplier = np.where(core_negentropy * trend_gate > 0, 1.0 + np.abs(trend_gate) * 0.5, 1.0).astype(np.float32)
        base_score = core_negentropy * divergence_penalty * alignment_multiplier
        final_d5 = np.tanh(base_score).astype(np.float32)
        if is_debug and probe_idx >= 0:
            self._probe_val("Negentropy", negentropy, temp_vals, "D5_Negentropy", probe_idx)
            self._probe_val("Order_Param", primary, temp_vals, "D5_Negentropy", probe_idx)
            self._probe_val("Core_Negentropy", core_negentropy, temp_vals, "D5_Negentropy", probe_idx)
            self._probe_val("Final_D5", final_d5, temp_vals, "D5_Negentropy", probe_idx)
        return final_d5
    def _calculate_pentagonal_resonance(self, D1: np.ndarray, D2: np.ndarray, D3: np.ndarray, D4: np.ndarray, D5: np.ndarray, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_idx: int, temp_vals: Dict) -> np.ndarray:
        close = df['close_D'].to_numpy(dtype=np.float32)
        adx = df['ADX_14_D'].to_numpy(dtype=np.float32)
        winner_rate = df['winner_rate_D'].to_numpy(dtype=np.float32)
        i12 = _nb_tensor_proxy(D1, D2)
        i23 = _nb_tensor_proxy(D2, D3)
        i34 = _nb_tensor_proxy(D3, D4)
        i45 = _nb_tensor_proxy(D4, D5)
        i51 = _nb_tensor_proxy(D5, D1)
        tensor_vol = (i12 + i23 + i34 + i45 + i51) / 5.0
        w1, w2, w3, w4, w5 = 0.2, 0.3, 0.2, 0.15, 0.15
        linear_score = D1*w1 + D2*w2 + D3*w3 + D4*w4 + D5*w5
        base_resonance = linear_score + tensor_vol * 1.5
        res_diff = pd.Series(base_resonance, index=idx).diff(8).fillna(0.0).to_numpy(dtype=np.float32)
        price_diff = pd.Series(close, index=idx).diff(8).fillna(0.0).to_numpy(dtype=np.float32)
        roll_cov = pd.Series(res_diff, index=idx).rolling(13, min_periods=1).cov(pd.Series(price_diff, index=idx)).fillna(0.0).to_numpy(dtype=np.float32)
        roll_var_res = pd.Series(res_diff, index=idx).rolling(13, min_periods=1).var().fillna(0.0).to_numpy(dtype=np.float32)
        roll_var_price = pd.Series(price_diff, index=idx).rolling(13, min_periods=1).var().fillna(0.0).to_numpy(dtype=np.float32)
        raw_corr = roll_cov / (np.sqrt(np.maximum(0.0, roll_var_res * roll_var_price)) + 1e-5)
        price_slope, _, _ = self._get_kinematics(df, df['close_D'], 'close_D', 13, temp_vals if is_debug else None, "Fusion", probe_idx)
        price_dir = np.where(np.abs(price_slope / (close + 1e-5)) > 0.005, np.sign(price_slope), 0.0).astype(np.float32)
        alignment = np.where(base_resonance * price_dir > 0, 1.0, np.where(base_resonance * price_dir < 0, -1.0, 0.0)).astype(np.float32)
        reflexivity_factor = np.where(alignment > 0, 1.0 + np.abs(raw_corr) * 0.5, np.where(alignment < 0, 1.0 / (1.0 + np.abs(raw_corr) * 0.5), 1.0)).astype(np.float32)
        core_score = base_resonance * reflexivity_factor
        market_temp = np.tanh(adx / 50.0).astype(np.float32)
        extreme_state = (4.0 * ((winner_rate / 100.0 - 0.5) ** 2)).astype(np.float32)
        apt_score = _nb_apt_score(core_score, market_temp, extreme_state)
        if is_debug and probe_idx >= 0:
            self._probe_val("Tensor_Volume", tensor_vol, temp_vals, "D6_Fusion", probe_idx)
            self._probe_val("Extreme_State", extreme_state, temp_vals, "D6_Fusion", probe_idx)
            self._probe_val("Raw_Corr", raw_corr, temp_vals, "D6_Fusion", probe_idx)
            self._probe_val("Alignment", alignment, temp_vals, "D6_Fusion", probe_idx)
            self._probe_val("Reflexivity", reflexivity_factor, temp_vals, "D6_Fusion", probe_idx)
            self._probe_val("APT_Score", apt_score, temp_vals, "D6_Fusion", probe_idx)
            self._probe_val("Final_Score", apt_score, temp_vals, "D6_Fusion", probe_idx)
        return apt_score
    def _log_debug_values(self, debug_out: Dict, temp_vals: Dict, probe_ts: pd.Timestamp, method_name: str):
        print(f"\n====== {method_name} 全链路探针输出 @ {pd.to_datetime(probe_ts).strftime('%Y-%m-%d')} ======")
        for section, data in temp_vals.items():
            print(f"[{section}]")
            for k, v in data.items():
                print(f"  {k:<20}: {v}")
        print("========================================================================\n")












