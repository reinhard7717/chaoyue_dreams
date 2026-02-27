# strategies\trend_following\intelligence\process\calculate_cost_advantage_trend_relationship.py
# 【V11.0.0 · 全息成本资金五维共振计算器】 已升级pro
# strategies\trend_following\intelligence\process\calculate_cost_advantage_trend_relationship.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from strategies.trend_following.utils import get_params_block, get_param_value
from strategies.trend_following.intelligence.process.helper import ProcessIntelligenceHelper

class CalculateCostAdvantageTrendRelationship:
    """
    【V14.0.0 · 物理超平面张量共振免疫版】
    PROCESS_META_COST_ADVANTAGE_TREND
    - 用途: 基于高维物理场模型（谐振子势垒/洛伦兹偏转/卡诺热机/杨氏模量）诊断趋势的爆发力与持续性。
    - 本次修改要点:
      1. [致命修复] 彻底切除 D1 和 D5 中的“极性反噬”漏洞，恢复做空力量向下的摧毁力。
      2. [量纲对齐] 修复卡诺热源(D3)因未归一化导致做功效率恒等 1.0 的计算失真。
      3. [噪音免疫] 绝对价格斜率(D4)强制转为百分比比率应变，防止高价股的导数爆炸。
      4. [死锁攻防] 在共振层(D6)加入方向同向性锁 (Alignment)，阻断纯相关性放大的逆势风险。
    """
    def __init__(self, strategy_instance, helper_instance: ProcessIntelligenceHelper):
        self.strategy = strategy_instance
        self.helper = helper_instance
        self.params = self.helper.params
        self.debug_params = self.helper.debug_params
        self.probe_dates = self.helper.probe_dates
        self.version = "V14.0.0"

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
        return pd.Series(np.tanh(z_score / 3.0), index=series.index)

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
            'market_sentiment_score_D': 50.0, 
            'cost_50pct_D': fallback_close, 'cost_5pct_D': fallback_close * 0.8, 'cost_95pct_D': fallback_close * 1.2,
            'close_D': 10.0, 'chip_entropy_D': 0.5, 'turnover_rate_f_D': 1.0, 'profit_pressure_D': 0.0,
            'pressure_trapped_D': 0.0, 'net_energy_flow_D': 0.0, 'tick_large_order_net_D': 0.0,
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
        print(f" ====== Start {method_name} 启动五维张量物理共振计算，数据形状: {df.shape} ======")
        
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
            
        print(f" ====== End {method_name} 五维张量物理共振计算完成 ======")
        return final_score.astype(np.float32).fillna(0.0)

    def _calculate_chip_barrier_solidity(self, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        concentration = df['chip_concentration_ratio_D']
        stability = df['chip_stability_D']
        cost_50 = df['cost_50pct_D']
        close = df['close_D']
        pressure_release = df['pressure_release_index_D']
        
        k_spring = self._soft_norm(concentration, 34) + 1.0
        displacement = (close - cost_50) / (cost_50 + 1e-8)
        
        # 修复极性反噬：采用位移方向标记保留向下的势能摧毁力
        potential_energy = 0.5 * k_spring * (displacement ** 2) * np.where(displacement >= 0, 1.0, -1.0)
        pe_norm = pd.Series(np.tanh(potential_energy * 10.0), index=idx)
        
        hab_stability = self._calc_hab_impact(stability, 55)
        
        slope_conc, accel_conc, jerk_conc = self._get_kinematics(df, concentration, 'chip_concentration_ratio_D', 13, temp_vals if is_debug and probe_ts in idx else None, "D1_ChipSolidity")
        kinematic_mod = np.tanh(self._soft_norm(slope_conc, 21) * 0.5 + self._soft_norm(accel_conc, 21) * 0.3)
        pr_norm = pd.Series(np.tanh((pressure_release - 50.0) / 20.0), index=idx)
        
        base_score = pe_norm * 0.4 + hab_stability * 0.4 + kinematic_mod * 0.1 + pr_norm * 0.1
        final_d1 = pd.Series(np.tanh(base_score), index=idx)
        
        if is_debug and probe_ts in idx:
            self._probe_val("K_Spring", k_spring.loc[probe_ts], temp_vals, "D1_ChipSolidity")
            self._probe_val("Potential_Energy", pd.Series(potential_energy, index=idx).loc[probe_ts], temp_vals, "D1_ChipSolidity")
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
        
        q_charge = 1.0 + np.tanh(coord_attack)
        lorentz_force = q_charge * (hab_e_field + v_norm * b_field)
        
        slope_sm, accel_sm, jerk_sm = self._get_kinematics(df, hm_buy, 'SMART_MONEY_HM_NET_BUY_D', 13, temp_vals if is_debug and probe_ts in idx else None, "D2_PredatorAttack")
        kinematic_mod = np.tanh(self._soft_norm(slope_sm, 21) * 0.5 + self._soft_norm(accel_sm, 21) * 0.3)
        
        base_score = np.tanh(lorentz_force) * 0.7 + kinematic_mod * 0.3
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
        
        # 修复量纲塌陷：将 0~100 的温源缩放至物理合理区间
        t_c = entropy.clip(0, 1.0)
        t_h = 1.0 + np.maximum(uptrend, downtrend) / 100.0
        carnot_eta = pd.Series(np.maximum(0.0, 1.0 - (t_c / (t_h + 1e-8))), index=idx)
        
        useful_work = hab_energy * hab_vpa.abs() * carnot_eta
        
        slope_v, accel_v, jerk_v = self._get_kinematics(df, velocity, 'MA_VELOCITY_EMA_55_D', 13, temp_vals if is_debug and probe_ts in idx else None, "D3_KinematicEff")
        kinematic_mod = np.tanh(self._soft_norm(slope_v, 21) * 0.5 + self._soft_norm(accel_v, 21) * 0.3)
        
        base_score = np.tanh(useful_work) * 0.6 + kinematic_mod * 0.4
        final_d3 = pd.Series(np.tanh(base_score), index=idx)
        
        if is_debug and probe_ts in idx:
            self._probe_val("T_Hot_Scaled", pd.Series(t_h, index=idx).loc[probe_ts], temp_vals, "D3_KinematicEff")
            self._probe_val("Carnot_Eta", carnot_eta.loc[probe_ts], temp_vals, "D3_KinematicEff")
            self._probe_val("Useful_Work", pd.Series(useful_work, index=idx).loc[probe_ts], temp_vals, "D3_KinematicEff")
            self._probe_val("Final_D3", final_d3.loc[probe_ts], temp_vals, "D3_KinematicEff")
            
        return final_d3

    def _calculate_cost_migration_elasticity(self, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        turnover = df['turnover_rate_f_D']
        profit_pres = df['profit_pressure_D']
        trapped_pres = df['pressure_trapped_D']
        cost_50 = df['cost_50pct_D']
        chip_flow = df['chip_flow_intensity_D']
        
        slope_c50, _, _ = self._get_kinematics(df, cost_50, 'cost_50pct_D', 13, temp_vals if is_debug and probe_ts in idx else None, "D4_Elasticity")
        hab_turnover = self._calc_hab_impact(turnover, 21).clip(0, None)
        
        stress = hab_turnover + np.maximum(0.0, np.tanh(profit_pres / 50.0)) + np.maximum(0.0, np.tanh(trapped_pres / 50.0))
        # 修复斜率未归一化引发的微积分爆炸：强制换算为百分比形变
        strain = (slope_c50 / (cost_50 + 1e-8)) * 100.0
        
        compliance = strain / (stress + 0.1)
        compliance_norm = pd.Series(np.tanh(compliance), index=idx)
        flow_mod = np.tanh(self._soft_norm(chip_flow, 21))
        
        base_score = compliance_norm * 0.6 + flow_mod * 0.4
        final_d4 = pd.Series(np.tanh(base_score), index=idx)
        
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
        
        negentropy = np.tanh((hab_entropy - total_micro_entropy) * 5.0)
        order_param = np.sqrt(np.abs(negentropy * reg_r2)) * np.where(negentropy >= 0, 1.0, -1.0)
        
        slope_ent, _, _ = self._get_kinematics(df, chip_entropy, 'chip_entropy_D', 13, temp_vals if is_debug and probe_ts in idx else None, "D5_Negentropy")
        kinetics_mod = -np.tanh(self._soft_norm(slope_ent, 21))
        trend_gate = pd.Series(np.tanh(reg_slope * 10.0), index=idx)
        
        # 修复极性反噬：采用线性拼接而不是盲目相乘
        base_score = order_param * 0.5 + kinetics_mod * 0.3 + trend_gate * 0.2
        final_d5 = pd.Series(np.tanh(base_score), index=idx)
        
        if is_debug and probe_ts in idx:
            self._probe_val("Negentropy", pd.Series(negentropy, index=idx).loc[probe_ts], temp_vals, "D5_Negentropy")
            self._probe_val("Order_Param", pd.Series(order_param, index=idx).loc[probe_ts], temp_vals, "D5_Negentropy")
            self._probe_val("Final_D5", final_d5.loc[probe_ts], temp_vals, "D5_Negentropy")
            
        return final_d5

    def _calculate_pentagonal_resonance(self, D1: pd.Series, D2: pd.Series, D3: pd.Series, D4: pd.Series, D5: pd.Series, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        close = df['close_D']
        adx = df['ADX_14_D']
        
        def tensor_proxy(a, b): return np.where(a * b >= 0, 1.0, -1.0) * np.sqrt(np.abs(a * b))
        
        i12 = tensor_proxy(D1, D2)
        i23 = tensor_proxy(D2, D3)
        i34 = tensor_proxy(D3, D4)
        i45 = tensor_proxy(D4, D5)
        i51 = tensor_proxy(D5, D1)
        
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
        
        # 引入 Alignment: 仅在张量方向和价格运动同频时给予正向增强，否则赋予惩罚削弱，杜绝逆向反射。
        alignment = np.where(base_resonance * price_slope >= 0, 1.0, -1.0)
        reflexivity_factor = pd.Series(np.where(alignment > 0, 1.0 + np.abs(raw_corr) * 0.5, 1.0 / (1.0 + np.abs(raw_corr) * 0.5)), index=idx)
        
        core_score = base_resonance * reflexivity_factor
        
        market_temp = np.tanh(adx / 50.0)
        dyn_threshold = 0.6 - (market_temp - 0.5) * 0.4
        dyn_gamma = 2.0 + (market_temp - 0.5) * 2.0
        
        gate_val = np.abs(core_score) - dyn_threshold
        apt_score = pd.Series(np.where(gate_val > 0.0, core_score * (1.0 + 0.6 * np.power(gate_val, 1.2) * dyn_gamma), core_score), index=idx)
        final_normalized_score = pd.Series(np.tanh(apt_score), index=idx)
        
        if is_debug and probe_ts in idx:
            self._probe_val("Tensor_Volume", tensor_vol.loc[probe_ts], temp_vals, "D6_Fusion")
            self._probe_val("Raw_Corr", raw_corr.loc[probe_ts], temp_vals, "D6_Fusion")
            self._probe_val("Alignment", pd.Series(alignment, index=idx).loc[probe_ts], temp_vals, "D6_Fusion")
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














