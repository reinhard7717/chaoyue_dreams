# strategies\trend_following\intelligence\process\calculate_process_covert_accumulation.py
# 【V2.12 · 微观订单流与结构共振版】“隐蔽吸筹”专属信号计算引擎 已完成pro
import pandas as pd
import numpy as np
import pandas_ta as ta
from numba import jit
from typing import Dict, List, Optional, Any, Tuple
from strategies.trend_following.utils import get_param_value, _robust_geometric_mean
from strategies.trend_following.intelligence.process.helper import ProcessIntelligenceHelper

class CalculateProcessCovertAccumulation:
    """
    【V9.5.0 · 零点锚定与动态杠杆版】
    PROCESS_META_COVERT_ACCUMULATION
    用途：计算隐蔽吸筹信号，精准识别主力在缩量、恐慌环境下的微观非对称收集行为。
    本次修改要点：
    1. 消除数据孤岛：强力编入被遗忘的 buy_elg_amount_rate_D 与 intraday_cost_center_migration_D。
    2. 算力精准制导：废弃全量微积分遍历，仅对引用的 4 个动力学指标进行定向生成，极大幅度降低耗时。
    3. 零点锚定与防爆：_custom_norm 新增 centered 参数，对双极性指标采用标准差映射，绝对防止 0 点漂移；免疫浮点溢出。
    4. 探针透传与弹性门控：重写 Numba 状态机双轨返回真实锁存状态；废除死板的 clip(0.45) 兜底，引入对数杠杆释放高密度势能。
    """
    def __init__(self, strategy_instance, helper_instance: ProcessIntelligenceHelper):
        self.strategy = strategy_instance
        self.helper = helper_instance

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        print(" ====== CalculateProcessCovertAccumulation V9.5.0 ======")
        is_debug = get_param_value(self.helper.debug_params.get('enabled'), False) and get_param_value(self.helper.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug and self.helper.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.helper.probe_dates]
            mask = pd.to_datetime(df.index).normalize().isin(probe_dates_dt)
            if mask.any(): probe_ts = df.index[mask][-1]

        temp_vals = {}
        c_cfg = self._get_config(config)
        
        raw_sigs = self._validate_and_get_raw_signals(df, temp_vals)
        self._calc_targeted_kinematics_and_hab(df, raw_sigs)
        
        t_vacuum = self._tensor_vol_vacuum(df, raw_sigs, c_cfg, temp_vals)
        t_stealth = self._tensor_micro_stealth(df, raw_sigs, c_cfg, temp_vals)
        den_val = float(temp_vals['stealth_density'].iloc[-1]) if len(temp_vals['stealth_density']) > 0 else 0.0
        stealth_val = float(t_stealth.iloc[-1]) if len(t_stealth) > 0 else 0.0
        print(f"DEBUG_PROBE:ActionSoftened|Density={den_val:.2f}|MicroStealth={stealth_val:.4f}")
        
        t_chip = self._tensor_chip_negentropy(df, raw_sigs, c_cfg, temp_vals)
        t_asym = self._tensor_intraday_asym(df, raw_sigs, c_cfg, temp_vals)
        t_panic = self._tensor_panic_exhaustion(df, raw_sigs, c_cfg, temp_vals)
        
        raw_score = self._fuse_tensors(df.index, t_vacuum, t_stealth, t_chip, t_asym, t_panic, c_cfg, temp_vals)
        
        try:
            final_score = self._apply_signal_latch(raw_score, t_vacuum, t_stealth, t_chip, df.index, temp_vals)
        except Exception as e:
            print(f"ERROR:LatchFailed|Msg={str(e)}")
            final_score = raw_score
            
        final_score = final_score.astype('float32') if not final_score.empty else pd.Series(0.0, index=df.index, dtype='float32')
        temp_vals["final_score"] = final_score
        
        if is_debug and probe_ts:
            print(f"DEBUG_PROBE:CalculateProcessCovertAccumulation|ProbeTS={probe_ts.strftime('%Y-%m-%d')}")
            self._print_debug({}, temp_vals, probe_ts)
            
        print(" ====== ======================== ======")
        return final_score

    def _get_config(self, config: Dict) -> Dict:
        cfg = {}
        for diag in config.get("process_intelligence_params", {}).get("diagnostics", []):
            if diag.get("name") == "PROCESS_META_COVERT_ACCUMULATION":
                cfg = diag.get("covert_accumulation_params", {})
                break
        w = cfg.get("dimension_weights", {"volatility_vacuum": 0.20, "micro_stealth": 0.25, "chip_negentropy": 0.20, "intraday_asymmetry": 0.20, "panic_exhaustion": 0.15})
        print(f"DEBUG_PROBE:CoherencyConfigLoaded|EntropyWeight={w.get('chip_negentropy', 0.20)}")
        return {"dim_weights": w, "tensor_power": cfg.get("tensor_folding_power", 1.2)}

    def _validate_and_get_raw_signals(self, df: pd.DataFrame, temp_vals: Dict) -> Dict[str, pd.Series]:
        req_cols = [
            'BBW_21_2.0_D', 'TURNOVER_STABILITY_INDEX_D', 'PRICE_ENTROPY_D', 'stealth_flow_ratio_D',
            'tick_clustering_index_D', 'hidden_accumulation_intensity_D', 'chip_concentration_ratio_D',
            'chip_stability_D', 'chip_entropy_D', 'INTRADAY_SUPPORT_INTENT_D', 'intraday_accumulation_confidence_D',
            'afternoon_flow_ratio_D', 'market_sentiment_score_D', 'pressure_trapped_D', 'loser_loss_margin_avg_D',
            'high_freq_flow_skewness_D', 'intraday_cost_center_migration_D', 'volume_vs_ma_5_ratio_D', 'ATR_14_D', 'close_D',
            'net_mf_amount_D', 'amount_D', 'buy_elg_amount_rate_D', 'VPA_MF_ADJUSTED_EFF_D'
        ]
        missing = [c for c in req_cols if c not in df.columns]
        if missing:
            temp_vals["__MISSING_COLUMNS__"] = missing
            for c in missing: df[c] = np.nan
            
        sigs = {c: df[c].ffill().fillna(0.0) for c in req_cols}
        
        close_safe = sigs['close_D'].replace(0.0, np.nan)
        norm_atr = sigs['ATR_14_D'] / (close_safe + 0.001)
        temp_vals['stealth_density'] = (sigs['volume_vs_ma_5_ratio_D'] / (norm_atr + 0.001)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        
        amount_safe = sigs['amount_D'].replace(0.0, np.nan)
        sigs['mf_net_ratio_D'] = (sigs['net_mf_amount_D'] / (amount_safe + 1e-5)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        
        temp_vals["原始信号"] = sigs
        print(f"DEBUG_PROBE:RawSignalsExtracted|NetMfRatioLen={len(sigs['mf_net_ratio_D'])}|MigrationLen={len(sigs['intraday_cost_center_migration_D'])}")
        return sigs

    def _threshold_gate(self, s: pd.Series) -> pd.Series:
        eps = s.rolling(21, min_periods=1).std().fillna(1e-4).replace(0.0, 1e-4) * 0.1
        return pd.Series(np.where(s.abs() < eps, 0.0, s * np.tanh(s.abs() / eps)), index=s.index)

    def _custom_norm(self, s: pd.Series, asc: bool = True, method: str = 'sigmoid', centered: bool = False) -> pd.Series:
        clean = s.ffill().fillna(0.0)
        
        if centered:
            # 针对存在物理中轴的指标 (如净流率/导数)，采用标准差映射，杜绝 Min-Max 造成的 0 点漂移
            roll_std = clean.rolling(55, min_periods=1).std().clip(lower=1e-5)
            z = clean / roll_std
            if method == 'sigmoid':
                norm = 1.0 / (1.0 + np.exp(-z))
            elif method == 'power':
                base = (np.tanh(z / 2.0) + 1.0) / 2.0
                norm = np.power(base, 1.5)
            else:
                norm = (np.tanh(z / 2.0) + 1.0) / 2.0
        else:
            rmin = clean.rolling(55, min_periods=1).min()
            rmax = clean.rolling(55, min_periods=1).max()
            diff = rmax - rmin
            # 完美免疫浮点数极小值除零溢出
            diff = np.where(diff <= 1e-6, 1e-5, diff)
            norm = ((clean - rmin) / diff).clip(0.0, 1.0)
            
            if method == 'sigmoid':
                z = (norm - 0.5) * 6.0
                norm = 1.0 / (1.0 + np.exp(-z))
            elif method == 'power':
                norm = np.power(norm, 1.5)
            else:
                norm = (np.tanh((norm - 0.5) * 2.0) + 1.0) / 2.0
                
        if not asc: norm = 1.0 - norm
        return (norm * 0.8 + 0.2).clip(1e-4, 0.9999)

    def _calc_targeted_kinematics_and_hab(self, df: pd.DataFrame, sigs: Dict[str, pd.Series]):
        """【精确定向微积分引擎】砍去无效算力，只针对被张量实际引用的指纹进行高阶导数提取。"""
        # 1. 隐匿资金流 Jerk
        sl_sth = ta.slope(sigs['stealth_flow_ratio_D'], length=8)
        sl_sth_cl = self._threshold_gate(sl_sth.fillna(0.0)) if sl_sth is not None else pd.Series(0.0, index=df.index)
        ac_sth = ta.slope(sl_sth_cl, length=3)
        ac_sth_cl = self._threshold_gate(ac_sth.fillna(0.0)) if ac_sth is not None else pd.Series(0.0, index=df.index)
        jk_sth = ta.slope(ac_sth_cl, length=3)
        df['JERK_8_stealth_flow_ratio_D'] = self._threshold_gate(jk_sth.fillna(0.0)) if jk_sth is not None else pd.Series(0.0, index=df.index)
        
        # 2. 筹码集中度 Jerk
        sl_conc = ta.slope(sigs['chip_concentration_ratio_D'], length=13)
        sl_conc_cl = self._threshold_gate(sl_conc.fillna(0.0)) if sl_conc is not None else pd.Series(0.0, index=df.index)
        ac_conc = ta.slope(sl_conc_cl, length=3)
        ac_conc_cl = self._threshold_gate(ac_conc.fillna(0.0)) if ac_conc is not None else pd.Series(0.0, index=df.index)
        jk_conc = ta.slope(ac_conc_cl, length=3)
        df['JERK_13_chip_concentration_ratio_D'] = self._threshold_gate(jk_conc.fillna(0.0)) if jk_conc is not None else pd.Series(0.0, index=df.index)
        
        # 3. 主力净流率 Slope
        sl_net = ta.slope(sigs['mf_net_ratio_D'], length=5)
        df['SLOPE_5_mf_net_ratio_D'] = self._threshold_gate(sl_net.fillna(0.0)) if sl_net is not None else pd.Series(0.0, index=df.index)
        
        # 4. 筹码熵减 Accel
        sl_ent = ta.slope(sigs['chip_entropy_D'], length=13)
        sl_ent_cl = self._threshold_gate(sl_ent.fillna(0.0)) if sl_ent is not None else pd.Series(0.0, index=df.index)
        ac_ent = ta.slope(sl_ent_cl, length=3)
        df['ACCEL_13_chip_entropy_D'] = self._threshold_gate(ac_ent.fillna(0.0)) if ac_ent is not None else pd.Series(0.0, index=df.index)

        print(f"DEBUG_PROBE:DynamicsCalculated|NetMfRatioSlopeCol={'SLOPE_5_mf_net_ratio_D' in df.columns}")

        # 5. 定向 HAB
        hab_targets = [('hidden_accumulation_intensity_D', 21), ('mf_net_ratio_D', 21), ('pressure_trapped_D', 13)]
        for col, p in hab_targets:
            clean = sigs[col]
            inc = clean.diff().fillna(0.0)
            hab = clean.ewm(span=p, adjust=False).mean()
            df[f'IMPACT_{p}_{col}'] = np.tanh(inc / (hab.abs() + 1e-5))

    def _tensor_vol_vacuum(self, df: pd.DataFrame, sigs: Dict[str, pd.Series], cfg: Dict, temp_vals: Dict) -> pd.Series:
        s1 = self._custom_norm(sigs['BBW_21_2.0_D'], asc=False, method='sigmoid', centered=False)
        s2 = self._custom_norm(sigs['TURNOVER_STABILITY_INDEX_D'], asc=True, method='tanh', centered=False)
        s3 = self._custom_norm(sigs['PRICE_ENTROPY_D'], asc=False, method='power', centered=False)
        t = (s1 * 0.40 + s2 * 0.30 + s3 * 0.30).clip(1e-4, 1.0)
        temp_vals["T_VACUUM"] = t
        return t

    def _tensor_micro_stealth(self, df: pd.DataFrame, sigs: Dict[str, pd.Series], cfg: Dict, temp_vals: Dict) -> pd.Series:
        s1 = self._custom_norm(sigs['stealth_flow_ratio_D'], asc=True, method='power', centered=False)
        s2 = self._custom_norm(sigs['tick_clustering_index_D'], asc=True, method='tanh', centered=False)
        s3 = self._custom_norm(sigs['hidden_accumulation_intensity_D'], asc=True, method='power', centered=False)
        # 高频偏度与主力净流率具有物理 0 点，启用 centered=True
        s4 = self._custom_norm(sigs['high_freq_flow_skewness_D'], asc=True, method='sigmoid', centered=True)
        s5 = self._custom_norm(sigs['mf_net_ratio_D'], asc=True, method='tanh', centered=True)
        s6 = self._custom_norm(sigs['VPA_MF_ADJUSTED_EFF_D'], asc=True, method='tanh', centered=False)
        # 解除孤岛 1
        s7 = self._custom_norm(sigs['buy_elg_amount_rate_D'], asc=True, method='power', centered=False)
        
        hab_impact = self._custom_norm(df.get('IMPACT_21_hidden_accumulation_intensity_D', pd.Series(0.0, index=df.index)), asc=True, method='tanh', centered=True)
        hab_net = self._custom_norm(df.get('IMPACT_21_mf_net_ratio_D', pd.Series(0.0, index=df.index)), asc=True, method='tanh', centered=True)
        jk_sth = self._custom_norm(df.get('JERK_8_stealth_flow_ratio_D', pd.Series(0.0, index=df.index)), asc=True, method='tanh', centered=True)
        
        raw_t = (s1 * 0.10 + s2 * 0.10 + s3 * 0.12 + s4 * 0.10 + s5 * 0.12 + s6 * 0.08 + s7 * 0.12 + hab_impact * 0.08 + hab_net * 0.10 + jk_sth * 0.08).clip(1e-4, 1.0)
        
        den = temp_vals.get('stealth_density', pd.Series(0.0, index=df.index))
        # 废弃硬切，采用平滑且不封顶的物理对数增益补偿密度
        bonus = pd.Series(np.where(den > 15.0, np.log1p(den - 15.0) * 0.08, 0.0), index=df.index)
        t = (raw_t + bonus).clip(1e-4, 1.0)
        temp_vals["T_STEALTH"] = t
        return t

    def _tensor_chip_negentropy(self, df: pd.DataFrame, sigs: Dict[str, pd.Series], cfg: Dict, temp_vals: Dict) -> pd.Series:
        s1 = self._custom_norm(sigs['chip_concentration_ratio_D'], asc=True, method='power', centered=False)
        s2 = self._custom_norm(sigs['chip_stability_D'], asc=True, method='sigmoid', centered=False)
        s3 = self._custom_norm(sigs['chip_entropy_D'], asc=False, method='power', centered=False)
        # 解除孤岛 2
        s4 = self._custom_norm(sigs['intraday_cost_center_migration_D'], asc=True, method='tanh', centered=True)
        
        jk_conc = self._custom_norm(df.get('JERK_13_chip_concentration_ratio_D', pd.Series(0.0, index=df.index)), asc=True, method='tanh', centered=True)
        ac_ent = self._custom_norm(df.get('ACCEL_13_chip_entropy_D', pd.Series(0.0, index=df.index)), asc=False, method='tanh', centered=True)
        
        t = (s1 * 0.25 + s2 * 0.20 + s3 * 0.20 + s4 * 0.15 + jk_conc * 0.10 + ac_ent * 0.10).clip(1e-4, 1.0)
        temp_vals["T_CHIP"] = t
        return t

    def _tensor_intraday_asym(self, df: pd.DataFrame, sigs: Dict[str, pd.Series], cfg: Dict, temp_vals: Dict) -> pd.Series:
        s1 = self._custom_norm(sigs['INTRADAY_SUPPORT_INTENT_D'], asc=True, method='tanh', centered=False)
        s2 = self._custom_norm(sigs['intraday_accumulation_confidence_D'], asc=True, method='sigmoid', centered=False)
        s3 = self._custom_norm(sigs['afternoon_flow_ratio_D'], asc=True, method='power', centered=False)
        t = (s1 * 0.35 + s2 * 0.35 + s3 * 0.30).clip(1e-4, 1.0)
        temp_vals["T_ASYM"] = t
        return t

    def _tensor_panic_exhaustion(self, df: pd.DataFrame, sigs: Dict[str, pd.Series], cfg: Dict, temp_vals: Dict) -> pd.Series:
        s1 = self._custom_norm(sigs['market_sentiment_score_D'], asc=False, method='power', centered=False)
        s2 = self._custom_norm(sigs['pressure_trapped_D'], asc=True, method='sigmoid', centered=False)
        s3 = self._custom_norm(sigs['loser_loss_margin_avg_D'], asc=True, method='power', centered=False)
        hab_impact = self._custom_norm(df.get('IMPACT_13_pressure_trapped_D', pd.Series(0.0, index=df.index)), asc=False, method='tanh', centered=True)
        t = (s1 * 0.30 + s2 * 0.25 + s3 * 0.30 + hab_impact * 0.15).clip(1e-4, 1.0)
        temp_vals["T_PANIC"] = t
        return t

    def _fuse_tensors(self, idx: pd.Index, t1: pd.Series, t2: pd.Series, t3: pd.Series, t4: pd.Series, t5: pd.Series, cfg: Dict, temp_vals: Dict) -> pd.Series:
        w = cfg["dim_weights"]
        prod = (t1 ** w.get('volatility_vacuum', 0.2)) * (t2 ** w.get('micro_stealth', 0.25)) * (t3 ** w.get('chip_negentropy', 0.2)) * (t4 ** w.get('intraday_asymmetry', 0.2)) * (t5 ** w.get('panic_exhaustion', 0.15))
        
        den_val = temp_vals.get('stealth_density', pd.Series(0.0, index=idx))
        folded_raw = np.power(prod, cfg.get("tensor_power", 1.2)).clip(0.0, 1.0)
        
        # 弹性核爆门控：宽容单维度的落后，只要核心维度达到引爆标准即触发开方放大
        is_solid = (den_val > 15.0) & ((t3 > 0.50) | (t2 > 0.50))
        boost_base = (np.log1p(den_val) / 2.0).clip(1.0, 2.5)
        
        folded = pd.Series(np.where(is_solid, np.sqrt(prod * boost_base), folded_raw), index=idx).clip(0.0, 1.0)
        
        temp_vals["RAW_PROD"] = prod
        temp_vals["SOLID"] = is_solid
        temp_vals["FOLDED"] = folded
        
        prod_val = float(prod.iloc[-1]) if len(prod) > 0 else 0.0
        final_val = float(folded.iloc[-1]) if len(folded) > 0 else 0.0
        solid_val = bool(is_solid.iloc[-1]) if len(is_solid) > 0 else False
        print(f"DEBUG_PROBE:FusionIgnited_V9.5.0|Raw={prod_val:.4f}|Final={final_val:.4f}|Solid={solid_val}")
        return folded

    def _apply_signal_latch(self, score: pd.Series, t1: pd.Series, t2: pd.Series, t3: pd.Series, idx: pd.Index, temp_vals: Dict) -> pd.Series:
        comp = pd.concat([t1, t2, t3], axis=1)
        ewd = comp.std(axis=1).fillna(1.0).values
        solid = temp_vals.get("SOLID", pd.Series(False, index=idx)).values
        
        thr = np.where(solid, 0.35, 0.45)
        ent = np.where(solid, 0.35, 0.15)
        
        val = score.fillna(0.0).values.astype(np.float64)
        mask = (val > thr) & (ewd < ent)
        
        roll_trig = pd.Series(mask.astype(np.int8)).rolling(3, min_periods=1).sum().fillna(0).values
        active = (roll_trig >= 2).astype(np.int8)
        
        # Numba 内核现双轨返回数值和真实状态
        latched_vals, latched_states = self._numba_latch_kernel(val, active)
        latched_series = pd.Series(latched_vals, index=idx, dtype='float32').clip(0.0, 1.0)
        
        # 探针修复：打印内部真实锁存记忆状态 (CoreLock)，而非单纯的触发脉冲
        core_lck = bool(latched_states[-1] > 0) if len(latched_states) > 0 else False
        last_score = float(latched_series.iloc[-1]) if len(latched_series) > 0 else 0.0
        print(f"DEBUG_PROBE:LatchFinal_V9.5.0|CoreLock={core_lck}|LastVal={last_score:.4f}")
        return latched_series

    @staticmethod
    @jit(nopython=True, cache=True)
    def _numba_latch_kernel(val: np.ndarray, act: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = len(val)
        out = np.zeros(n, dtype=np.float64)
        states = np.zeros(n, dtype=np.int8)
        decay_locked = 0.96
        decay_free = 0.85
        brk = 0.30
        lck = False
        last = 0.0
        for i in range(n):
            v = val[i]
            if act[i] > 0:
                lck = True
                # 触发时跃迁锁存
                last = min(1.0, max(v * 1.15, last * decay_locked))
            elif lck:
                if v < brk:
                    lck = False
                    last = max(v, last * decay_free)
                else:
                    last = min(1.0, max(v, last * decay_locked))
            else:
                last = max(v, last * decay_free)
            out[i] = last
            states[i] = 1 if lck else 0
        return out, states

    def _print_debug(self, out: Dict, temp: Dict, ts: pd.Timestamp):
        if "__MISSING_COLUMNS__" in temp:
            out[f"  !! [红军告警] 缺失底层组件: {temp['__MISSING_COLUMNS__']}"] = "系统强制触发 0.2 底垫保护。"
        out[f"  -- [相空间五大高阶张量态] @ {ts.strftime('%Y-%m-%d')}"] = ""
        for k in ["T_VACUUM", "T_STEALTH", "T_CHIP", "T_ASYM", "T_PANIC", "FOLDED", "final_score"]:
            if k in temp:
                out[f"     |-- {k:<15}: {temp[k].loc[ts] if ts in temp[k].index else 0.0:.4f}"] = ""
        self.helper._print_debug_output(out)
















