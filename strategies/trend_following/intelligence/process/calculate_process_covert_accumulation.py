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
    【V10.2.0 · 全链路专业探针统合版 (Full-Chain Professional Telemetry Form)】
    PROCESS_META_COVERT_ACCUMULATION
    用途：计算隐蔽吸筹信号，精准识别主力在缩量、恐慌环境下的微观非对称收集行为。
    本次修改要点：
    1. 探针专业化统合：全面废除各计算节点散装的 print 打印，重构建立 _print_full_chain_telemetry 方法，实现全链路统一快照输出，避免高并发日志撕裂。
    2. 时间戳严格对齐：摒弃极易产生“未来函数泄露”风险的 iloc[-1] 截取法。所有监控状态机严格依据探针触发日 (probe_ts) 进行精确提取，确保回溯调试时的数据绝对真实。
    3. 状态机观测全息化：将 Numba 内核生成的内部锁存数组 (CoreLock) 与所有隐蔽指标映射为字典暴露至外部，构建【1.原始数据 -> 2.关键节点 -> 3.五大张量 -> 4.核爆结果】的完美监控流。
    废弃方法说明：无新增废弃，原有的散装 print 打印被全部清理并替换。
    """
    def __init__(self, strategy_instance, helper_instance: ProcessIntelligenceHelper):
        self.strategy = strategy_instance
        self.helper = helper_instance
    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        is_debug = True # get_param_value(self.helper.debug_params.get('enabled'), False) and get_param_value(self.helper.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug and self.helper.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.helper.probe_dates]
            mask = pd.to_datetime(df.index).normalize().isin(probe_dates_dt)
            if mask.any(): probe_ts = df.index[mask][-1]
        temp_vals = {}
        c_cfg = self._get_config(config)
        raw_sigs = self._validate_and_get_raw_signals(df, temp_vals)
        self._calc_targeted_kinematics_and_hab(df, raw_sigs, temp_vals)
        t_vacuum = self._tensor_vol_vacuum(df, raw_sigs, c_cfg, temp_vals)
        t_stealth = self._tensor_micro_stealth(df, raw_sigs, c_cfg, temp_vals)
        t_chip = self._tensor_chip_negentropy(df, raw_sigs, c_cfg, temp_vals)
        t_asym = self._tensor_intraday_asym(df, raw_sigs, c_cfg, temp_vals)
        t_panic = self._tensor_panic_exhaustion(df, raw_sigs, c_cfg, temp_vals)
        raw_score = self._fuse_tensors(df.index, t_vacuum, t_stealth, t_chip, t_asym, t_panic, c_cfg, temp_vals)
        try:
            final_score = self._apply_signal_latch(raw_score, t_vacuum, t_stealth, t_chip, df.index, temp_vals)
        except Exception as e:
            temp_vals["LATCH_ERROR"] = str(e)
            final_score = raw_score
        final_score = final_score.astype('float32') if not final_score.empty else pd.Series(0.0, index=df.index, dtype='float32')
        temp_vals["final_score"] = final_score
        if is_debug and probe_ts is not None:
            self._print_full_chain_telemetry(probe_ts, temp_vals)
        return final_score
    def _get_config(self, config: Dict) -> Dict:
        cfg = {}
        for diag in config.get("process_intelligence_params", {}).get("diagnostics", []):
            if diag.get("name") == "PROCESS_META_COVERT_ACCUMULATION":
                cfg = diag.get("covert_accumulation_params", {})
                break
        w = cfg.get("dimension_weights", {"volatility_vacuum": 0.20, "micro_stealth": 0.25, "chip_negentropy": 0.20, "intraday_asymmetry": 0.20, "panic_exhaustion": 0.15})
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
        return sigs
    def _threshold_gate(self, s: pd.Series) -> pd.Series:
        eps = s.rolling(21, min_periods=1).std().fillna(1e-4).replace(0.0, 1e-4) * 0.1
        return pd.Series(np.where(s.abs() < eps, 0.0, s * np.tanh(s.abs() / eps)), index=s.index)
    def _custom_norm(self, s: pd.Series, asc: bool = True, method: str = 'sigmoid', centered: bool = False) -> pd.Series:
        clean = s.ffill().fillna(0.0)
        if centered:
            roll_std = clean.rolling(55, min_periods=1).std().clip(lower=1e-5)
            z = clean / roll_std
            z = np.clip(z, -20.0, 20.0)
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
            diff = np.where(diff <= 1e-6, 1e-5, diff)
            norm = ((clean - rmin) / diff).clip(0.0, 1.0)
            if method == 'sigmoid':
                z = (norm - 0.5) * 6.0
                z = np.clip(z, -20.0, 20.0)
                norm = 1.0 / (1.0 + np.exp(-z))
            elif method == 'power':
                norm = np.power(norm, 1.5)
            else:
                norm = (np.tanh((norm - 0.5) * 2.0) + 1.0) / 2.0
        if not asc: norm = 1.0 - norm
        return (norm * 0.8 + 0.2).clip(1e-4, 0.9999)
    def _calc_targeted_kinematics_and_hab(self, df: pd.DataFrame, sigs: Dict[str, pd.Series], temp_vals: Dict):
        sl_sth = ta.slope(sigs['stealth_flow_ratio_D'], length=8)
        sl_sth_cl = self._threshold_gate(sl_sth.fillna(0.0)) if sl_sth is not None else pd.Series(0.0, index=df.index)
        ac_sth = ta.slope(sl_sth_cl, length=3)
        ac_sth_cl = self._threshold_gate(ac_sth.fillna(0.0)) if ac_sth is not None else pd.Series(0.0, index=df.index)
        jk_sth = ta.slope(ac_sth_cl, length=3)
        df['JERK_8_stealth_flow_ratio_D'] = self._threshold_gate(jk_sth.fillna(0.0)) if jk_sth is not None else pd.Series(0.0, index=df.index)
        temp_vals['JERK_8_stealth_flow_ratio_D'] = df['JERK_8_stealth_flow_ratio_D']
        sl_conc = ta.slope(sigs['chip_concentration_ratio_D'], length=13)
        sl_conc_cl = self._threshold_gate(sl_conc.fillna(0.0)) if sl_conc is not None else pd.Series(0.0, index=df.index)
        ac_conc = ta.slope(sl_conc_cl, length=3)
        ac_conc_cl = self._threshold_gate(ac_conc.fillna(0.0)) if ac_conc is not None else pd.Series(0.0, index=df.index)
        jk_conc = ta.slope(ac_conc_cl, length=3)
        df['JERK_13_chip_concentration_ratio_D'] = self._threshold_gate(jk_conc.fillna(0.0)) if jk_conc is not None else pd.Series(0.0, index=df.index)
        sl_net = ta.slope(sigs['mf_net_ratio_D'], length=5)
        df['SLOPE_5_mf_net_ratio_D'] = self._threshold_gate(sl_net.fillna(0.0)) if sl_net is not None else pd.Series(0.0, index=df.index)
        temp_vals['SLOPE_5_mf_net_ratio_D'] = df['SLOPE_5_mf_net_ratio_D']
        sl_ent = ta.slope(sigs['chip_entropy_D'], length=13)
        sl_ent_cl = self._threshold_gate(sl_ent.fillna(0.0)) if sl_ent is not None else pd.Series(0.0, index=df.index)
        ac_ent = ta.slope(sl_ent_cl, length=3)
        df['ACCEL_13_chip_entropy_D'] = self._threshold_gate(ac_ent.fillna(0.0)) if ac_ent is not None else pd.Series(0.0, index=df.index)
        hab_targets = [('hidden_accumulation_intensity_D', 21), ('mf_net_ratio_D', 21), ('pressure_trapped_D', 13)]
        for col, p in hab_targets:
            clean = sigs[col]
            inc = clean.diff().fillna(0.0)
            hab = clean.ewm(span=p, adjust=False).mean()
            df[f'IMPACT_{p}_{col}'] = np.tanh(inc / (hab.abs() + 1e-5))
            temp_vals[f'IMPACT_{p}_{col}'] = df[f'IMPACT_{p}_{col}']
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
        s4 = self._custom_norm(sigs['high_freq_flow_skewness_D'], asc=True, method='sigmoid', centered=True)
        s5 = self._custom_norm(sigs['mf_net_ratio_D'], asc=True, method='tanh', centered=True)
        s6 = self._custom_norm(sigs['VPA_MF_ADJUSTED_EFF_D'], asc=True, method='tanh', centered=False)
        s7 = self._custom_norm(sigs['buy_elg_amount_rate_D'], asc=True, method='power', centered=False)
        hab_impact = self._custom_norm(df.get('IMPACT_21_hidden_accumulation_intensity_D', pd.Series(0.0, index=df.index)), asc=True, method='tanh', centered=True)
        hab_net = self._custom_norm(df.get('IMPACT_21_mf_net_ratio_D', pd.Series(0.0, index=df.index)), asc=True, method='tanh', centered=True)
        jk_sth = self._custom_norm(df.get('JERK_8_stealth_flow_ratio_D', pd.Series(0.0, index=df.index)), asc=True, method='tanh', centered=True)
        raw_t = (s1 * 0.10 + s2 * 0.10 + s3 * 0.12 + s4 * 0.10 + s5 * 0.12 + s6 * 0.08 + s7 * 0.12 + hab_impact * 0.08 + hab_net * 0.10 + jk_sth * 0.08).clip(1e-4, 1.0)
        den = temp_vals.get('stealth_density', pd.Series(0.0, index=df.index))
        bonus = pd.Series(np.where(den > 15.0, np.log1p(np.maximum(den - 15.0, 0.0)) * 0.08, 0.0), index=df.index)
        t = (raw_t + bonus).clip(1e-4, 1.0)
        temp_vals["T_STEALTH"] = t
        return t
    def _tensor_chip_negentropy(self, df: pd.DataFrame, sigs: Dict[str, pd.Series], cfg: Dict, temp_vals: Dict) -> pd.Series:
        s1 = self._custom_norm(sigs['chip_concentration_ratio_D'], asc=True, method='power', centered=False)
        s2 = self._custom_norm(sigs['chip_stability_D'], asc=True, method='sigmoid', centered=False)
        s3 = self._custom_norm(sigs['chip_entropy_D'], asc=False, method='power', centered=False)
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
        is_solid = (den_val > 15.0) & (t2 > 0.45) & ((t3 > 0.50) | (t4 > 0.50))
        boost_base = (np.log1p(np.maximum(den_val, 0.0)) / 2.0).clip(1.0, 2.5)
        folded = pd.Series(np.where(is_solid, np.sqrt(prod * boost_base), folded_raw), index=idx).clip(0.0, 1.0)
        temp_vals["RAW_PROD"] = prod
        temp_vals["BOOST_BASE"] = boost_base
        temp_vals["SOLID"] = is_solid
        temp_vals["FOLDED"] = folded
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
        latched_vals, latched_states = self._numba_latch_kernel(val, active)
        latched_series = pd.Series(latched_vals, index=idx, dtype='float32').clip(0.0, 1.0)
        temp_vals["CORE_LOCK_SERIES"] = pd.Series(latched_states, index=idx)
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
    def _print_full_chain_telemetry(self, ts: pd.Timestamp, temp: Dict):
        print(f"\n[TELEMETRY] ====== CalculateProcessCovertAccumulation V10.2.0 ====== | TS: {ts.strftime('%Y-%m-%d')}")
        if "LATCH_ERROR" in temp:
            print(f"  !! [致命告警] 马尔可夫状态机崩溃: {temp['LATCH_ERROR']}")
        if "__MISSING_COLUMNS__" in temp:
            print(f"  !! [静默防御] 缺失底层军械组件: {temp['__MISSING_COLUMNS__']} (已触发 0.2 底垫隔离)")
        def _get_val(grp, k):
            if grp is None:
                s = temp.get(k)
            else:
                s = temp.get(grp, {}).get(k)
            if isinstance(s, pd.Series) and ts in s.index:
                return float(s.loc[ts])
            if isinstance(s, (float, int, bool, np.bool_)):
                return float(s)
            return 0.0
        print(">>> 1. [RAW DATA] 原始数据与复合指纹:")
        print(f"    - close_D                        : {_get_val('原始信号', 'close_D'):.4f}")
        print(f"    - amount_D                       : {_get_val('原始信号', 'amount_D'):.4f}")
        print(f"    - net_mf_amount_D                : {_get_val('原始信号', 'net_mf_amount_D'):.4f}")
        print(f"    - volume_vs_ma_5_ratio_D         : {_get_val('原始信号', 'volume_vs_ma_5_ratio_D'):.4f}")
        print(f"    - stealth_density      [复合指纹] : {_get_val(None, 'stealth_density'):.4f}")
        print(f"    - mf_net_ratio_D       [复合指纹] : {_get_val('原始信号', 'mf_net_ratio_D'):.4f}")
        print(">>> 2. [KINEMATICS] 关键计算节点与势能:")
        print(f"    - SLOPE_5_mf_net_ratio_D         : {_get_val(None, 'SLOPE_5_mf_net_ratio_D'):.4f}")
        print(f"    - JERK_8_stealth_flow_ratio_D    : {_get_val(None, 'JERK_8_stealth_flow_ratio_D'):.4f}")
        print(f"    - IMPACT_21_hidden_accum         : {_get_val(None, 'IMPACT_21_hidden_accumulation_intensity_D'):.4f}")
        print(">>> 3. [TENSOR SPACE] 量子相空间张量映射:")
        print(f"    - T_VACUUM   (波动真空)          : {_get_val(None, 'T_VACUUM'):.4f}")
        print(f"    - T_STEALTH  (微观潜行)          : {_get_val(None, 'T_STEALTH'):.4f}")
        print(f"    - T_CHIP     (筹码熵减)          : {_get_val(None, 'T_CHIP'):.4f}")
        print(f"    - T_ASYM     (日内非对称)        : {_get_val(None, 'T_ASYM'):.4f}")
        print(f"    - T_PANIC    (恐慌衰竭)          : {_get_val(None, 'T_PANIC'):.4f}")
        print(">>> 4. [FUSION & LATCH] 最终分数与状态机锁存:")
        print(f"    - Raw Product (原始连乘)         : {_get_val(None, 'RAW_PROD'):.4f}")
        print(f"    - Resonance Boost (动能乘数)     : {_get_val(None, 'BOOST_BASE'):.4f}")
        print(f"    - Solid State (核爆判定)         : {bool(_get_val(None, 'SOLID'))}")
        print(f"    - Folded Score (张量分)          : {_get_val(None, 'FOLDED'):.4f}")
        print(f"    - Core Lock (状态机内核)         : {bool(_get_val(None, 'CORE_LOCK_SERIES'))}")
        print(f"    - FINAL LATCHED (发信值)         : {_get_val(None, 'final_score'):.4f}")
        print("========================================================================================\n")















