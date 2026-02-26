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
    【V9.1.0 · 量子相空间纠缠坍缩全息观测版】
    PROCESS_META_COVERT_ACCUMULATION
    全面恢复硬核工业探针(DEBUG_PROBE)，底层彻底解决零基陷阱与0值连乘死锁，并大幅提速单向膜特征工程逼近算法。
    """
    def __init__(self, strategy_instance, helper_instance: ProcessIntelligenceHelper):
        self.strategy = strategy_instance
        self.helper = helper_instance
    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.1.0】计算“隐蔽吸筹”专属信号，执行量子相空间张量折叠与硬核探针无条件直传。"""
        print(" ====== CalculateProcessCovertAccumulation V9.1.0 ======")
        is_debug = get_param_value(self.helper.debug_params.get('enabled'), False) and get_param_value(self.helper.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug and self.helper.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.helper.probe_dates]
            mask = pd.to_datetime(df.index).normalize().isin(probe_dates_dt)
            if mask.any(): probe_ts = df.index[mask][-1]
        temp_vals = {}
        c_cfg = self._get_config(config)
        raw_sigs = self._validate_and_get_raw_signals(df, temp_vals)
        self._calc_kinematics(df, raw_sigs)
        self._calc_hab(df, raw_sigs)
        t_vacuum = self._tensor_vol_vacuum(df, raw_sigs, c_cfg, temp_vals)
        t_stealth = self._tensor_micro_stealth(df, raw_sigs, c_cfg, temp_vals)
        den_val = temp_vals['stealth_density'].iloc[-1] if not temp_vals['stealth_density'].empty else 0.0
        stealth_val = t_stealth.iloc[-1] if not t_stealth.empty else 0.0
        print(f"DEBUG_PROBE:ActionSoftened|Density={den_val:.2f}|InstBuySoft={stealth_val:.4f}")
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
        """【V9.1.0】获取张量相空间权重配置，并触发基底探针加载确认。"""
        cfg = {}
        for diag in config.get("process_intelligence_params", {}).get("diagnostics", []):
            if diag.get("name") == "PROCESS_META_COVERT_ACCUMULATION":
                cfg = diag.get("covert_accumulation_params", {})
                break
        w = cfg.get("dimension_weights", {"volatility_vacuum": 0.20, "micro_stealth": 0.25, "chip_negentropy": 0.20, "intraday_asymmetry": 0.20, "panic_exhaustion": 0.15})
        print(f"DEBUG_PROBE:CoherencyConfigLoaded|EntropyWeight={w.get('chip_negentropy', 0.20)}")
        return {"dim_weights": w, "tensor_power": cfg.get("tensor_folding_power", 1.5)}
    def _validate_and_get_raw_signals(self, df: pd.DataFrame, temp_vals: Dict) -> Dict[str, pd.Series]:
        """【V9.1.0】引入基础维度与探测映射阵列，实施严格数据验证防暴并触发信号提取探针。"""
        req_cols = [
            'BBW_21_2.0_D', 'TURNOVER_STABILITY_INDEX_D', 'PRICE_ENTROPY_D', 'stealth_flow_ratio_D',
            'tick_clustering_index_D', 'hidden_accumulation_intensity_D', 'chip_concentration_ratio_D',
            'chip_stability_D', 'chip_entropy_D', 'INTRADAY_SUPPORT_INTENT_D', 'intraday_accumulation_confidence_D',
            'afternoon_flow_ratio_D', 'market_sentiment_score_D', 'pressure_trapped_D', 'loser_loss_margin_avg_D',
            'high_freq_flow_skewness_D', 'intraday_cost_center_migration_D', 'volume_vs_ma_5_ratio_D', 'ATR_14_D', 'close'
        ]
        missing = [c for c in req_cols if c not in df.columns]
        if missing:
            temp_vals["__MISSING_COLUMNS__"] = missing
            for c in missing: df[c] = np.nan
        sigs = {c: df[c].ffill().fillna(0.0) for c in req_cols}
        close_safe = sigs['close'].replace(0, np.nan)
        norm_atr = sigs['ATR_14_D'] / (close_safe + 0.001)
        temp_vals['stealth_density'] = (sigs['volume_vs_ma_5_ratio_D'] / (norm_atr + 0.001)).fillna(0.0)
        temp_vals["原始信号"] = sigs
        print(f"DEBUG_PROBE:RawSignalsExtracted|EntropyLen={len(sigs['chip_entropy_D'])}|MigrationLen={len(sigs['intraday_cost_center_migration_D'])}")
        return sigs
    def _threshold_gate(self, s: pd.Series, eps: float = 1e-4) -> pd.Series:
        """【V9.1.0】物理信号门限函数：拦截分母无穷小导致的微积分假阳性震荡。"""
        return pd.Series(np.where(s.abs() < eps, 0.0, s * np.tanh(s.abs() / eps)), index=s.index)
    def _custom_norm(self, s: pd.Series, asc: bool = True, method: str = 'sigmoid') -> pd.Series:
        """【V9.1.0】单向膜归一化：全向量化 Std 极速替代 Apply 循环离差，完全禁绝 bfill。"""
        clean = s.ffill().fillna(0.0)
        roll_med = clean.rolling(55, min_periods=1).median()
        roll_std = clean.rolling(55, min_periods=1).std().fillna(1e-5).replace(0.0, 1e-5)
        z = (clean - roll_med) / roll_std
        if method == 'sigmoid': norm = 1.0 / (1.0 + np.exp(-z))
        elif method == 'power':
            rmin = clean.rolling(55, min_periods=1).min()
            rmax = clean.rolling(55, min_periods=1).max()
            norm = np.power(((clean - rmin) / (rmax - rmin).replace(0.0, 1e-5)).clip(0,1), 1.5)
        else: norm = (np.tanh(z / 2.0) + 1.0) / 2.0
        if not asc: norm = 1.0 - norm
        return norm.clip(1e-4, 0.9999)
    def _calc_kinematics(self, df: pd.DataFrame, sigs: Dict[str, pd.Series]):
        """【V9.1.0】高阶微积分动力学提取，并挂载 Threshold Gate 防爆。"""
        targets = ['stealth_flow_ratio_D', 'hidden_accumulation_intensity_D', 'chip_concentration_ratio_D', 'chip_entropy_D']
        for c in targets:
            if c not in df.columns: continue
            sm = ta.ema(df[c].ffill().fillna(0.0), length=3)
            if sm is None: sm = df[c].ffill().fillna(0.0)
            for p in [3, 5, 8, 13]:
                sl = ta.slope(sm, length=p)
                df[f'SLOPE_{p}_{c}'] = self._threshold_gate(sl.fillna(0.0)) if sl is not None else pd.Series(0.0, index=df.index)
                ac = ta.slope(df[f'SLOPE_{p}_{c}'], length=3)
                df[f'ACCEL_{p}_{c}'] = self._threshold_gate(ac.fillna(0.0)) if ac is not None else pd.Series(0.0, index=df.index)
                jk = ta.slope(df[f'ACCEL_{p}_{c}'], length=3)
                df[f'JERK_{p}_{c}'] = self._threshold_gate(jk.fillna(0.0)) if jk is not None else pd.Series(0.0, index=df.index)
        print(f"DEBUG_PROBE:DynamicsCalculated|EntropySlopeCol={'SLOPE_5_chip_entropy_D' in df.columns}")
    def _calc_hab(self, df: pd.DataFrame, sigs: Dict[str, pd.Series]):
        """【V9.1.0】历史累积记忆缓冲系统 (Historical Accumulation Buffer)。"""
        targets = ['stealth_flow_ratio_D', 'hidden_accumulation_intensity_D', 'pressure_trapped_D']
        for c in targets:
            if c not in df.columns: continue
            clean = df[c].ffill().fillna(0.0)
            inc = clean.diff().fillna(0.0)
            for p in [13, 21, 34, 55]:
                hab = clean.ewm(span=p, adjust=False).mean()
                df[f'HAB_{p}_{c}'] = hab
                df[f'IMPACT_{p}_{c}'] = np.tanh(inc / (hab.abs() + 1e-5))
    def _tensor_vol_vacuum(self, df: pd.DataFrame, sigs: Dict[str, pd.Series], cfg: Dict, temp_vals: Dict) -> pd.Series:
        """【V9.1.0】张量场：波动真空。"""
        s1 = self._custom_norm(sigs['BBW_21_2.0_D'], asc=False, method='sigmoid')
        s2 = self._custom_norm(sigs['TURNOVER_STABILITY_INDEX_D'], asc=True, method='tanh')
        s3 = self._custom_norm(sigs['PRICE_ENTROPY_D'], asc=False, method='power')
        t = (s1 * 0.4 + s2 * 0.3 + s3 * 0.3).clip(1e-4, 1.0)
        temp_vals["T_VACUUM"] = t
        return t
    def _tensor_micro_stealth(self, df: pd.DataFrame, sigs: Dict[str, pd.Series], cfg: Dict, temp_vals: Dict) -> pd.Series:
        """【V9.1.0】张量场：微观潜行。"""
        s1 = self._custom_norm(sigs['stealth_flow_ratio_D'], asc=True, method='power')
        s2 = self._custom_norm(sigs['tick_clustering_index_D'], asc=True, method='tanh')
        s3 = self._custom_norm(sigs['hidden_accumulation_intensity_D'], asc=True, method='power')
        s4 = self._custom_norm(sigs['high_freq_flow_skewness_D'], asc=True, method='sigmoid')
        hab_impact = self._custom_norm(df.get('IMPACT_21_hidden_accumulation_intensity_D', pd.Series(0.0, index=df.index)), asc=True, method='tanh')
        t = (s1 * 0.25 + s2 * 0.2 + s3 * 0.25 + s4 * 0.15 + hab_impact * 0.15).clip(1e-4, 1.0)
        temp_vals["T_STEALTH"] = t
        return t
    def _tensor_chip_negentropy(self, df: pd.DataFrame, sigs: Dict[str, pd.Series], cfg: Dict, temp_vals: Dict) -> pd.Series:
        """【V9.1.0】张量场：筹码熵减。"""
        s1 = self._custom_norm(sigs['chip_concentration_ratio_D'], asc=True, method='power')
        s2 = self._custom_norm(sigs['chip_stability_D'], asc=True, method='sigmoid')
        s3 = self._custom_norm(sigs['chip_entropy_D'], asc=False, method='power')
        jerk_conc = self._custom_norm(df.get('JERK_13_chip_concentration_ratio_D', pd.Series(0.0, index=df.index)), asc=True, method='tanh')
        t = (s1 * 0.35 + s2 * 0.25 + s3 * 0.25 + jerk_conc * 0.15).clip(1e-4, 1.0)
        temp_vals["T_CHIP"] = t
        return t
    def _tensor_intraday_asym(self, df: pd.DataFrame, sigs: Dict[str, pd.Series], cfg: Dict, temp_vals: Dict) -> pd.Series:
        """【V9.1.0】张量场：日内博弈非对称。"""
        s1 = self._custom_norm(sigs['INTRADAY_SUPPORT_INTENT_D'], asc=True, method='tanh')
        s2 = self._custom_norm(sigs['intraday_accumulation_confidence_D'], asc=True, method='sigmoid')
        s3 = self._custom_norm(sigs['afternoon_flow_ratio_D'], asc=True, method='power')
        t = (s1 * 0.35 + s2 * 0.35 + s3 * 0.30).clip(1e-4, 1.0)
        temp_vals["T_ASYM"] = t
        return t
    def _tensor_panic_exhaustion(self, df: pd.DataFrame, sigs: Dict[str, pd.Series], cfg: Dict, temp_vals: Dict) -> pd.Series:
        """【V9.1.0】张量场：恐慌衰竭。"""
        s1 = self._custom_norm(sigs['market_sentiment_score_D'], asc=False, method='power')
        s2 = self._custom_norm(sigs['pressure_trapped_D'], asc=True, method='sigmoid')
        s3 = self._custom_norm(sigs['loser_loss_margin_avg_D'], asc=True, method='power')
        hab_impact = self._custom_norm(df.get('IMPACT_13_pressure_trapped_D', pd.Series(0.0, index=df.index)), asc=False, method='tanh')
        t = (s1 * 0.3 + s2 * 0.25 + s3 * 0.3 + hab_impact * 0.15).clip(1e-4, 1.0)
        temp_vals["T_PANIC"] = t
        return t
    def _fuse_tensors(self, idx: pd.Index, t1: pd.Series, t2: pd.Series, t3: pd.Series, t4: pd.Series, t5: pd.Series, cfg: Dict, temp_vals: Dict) -> pd.Series:
        """【V9.1.0】张量折叠，恢复点火日志探针心跳。"""
        w = cfg["dim_weights"]
        prod = (t1 ** w.get('volatility_vacuum', 0.2)) * (t2 ** w.get('micro_stealth', 0.25)) * (t3 ** w.get('chip_negentropy', 0.2)) * (t4 ** w.get('intraday_asymmetry', 0.2)) * (t5 ** w.get('panic_exhaustion', 0.15))
        reso = np.where((t2 > 0.6) & (t5 > 0.6), 1.3, 1.0)
        folded = pd.Series(np.power(prod * reso, cfg["tensor_power"]).clip(0.0, 1.0), index=idx)
        temp_vals["RAW_PROD"] = prod
        temp_vals["SOLID"] = pd.Series(reso > 1.0, index=idx)
        temp_vals["FOLDED"] = folded
        prod_val = float(prod.iloc[-1]) if len(prod) > 0 else 0.0
        final_val = float(folded.iloc[-1]) if len(folded) > 0 else 0.0
        solid_val = bool(temp_vals["SOLID"].iloc[-1]) if len(temp_vals["SOLID"]) > 0 else False
        print(f"DEBUG_PROBE:FusionIgnited_V9.1.0|Raw={prod_val:.4f}|Final={final_val:.4f}|Solid={solid_val}")
        return folded
    def _apply_signal_latch(self, score: pd.Series, t1: pd.Series, t2: pd.Series, t3: pd.Series, idx: pd.Index, temp_vals: Dict) -> pd.Series:
        """【V9.1.0】马尔可夫锁存器，抛出锁定状态基底信号探针。"""
        comp = pd.concat([t1, t2, t3], axis=1)
        ewd = comp.std(axis=1).fillna(1.0).values
        solid = (t3 > 0.7).values
        thr = np.where(solid, 0.35, 0.55)
        ent = np.where(solid, 0.25, 0.15)
        val = score.fillna(0.0).values.astype(np.float64)
        mask = (val > thr) & (ewd < ent)
        roll_trig = pd.Series(mask.astype(np.int8)).rolling(3, min_periods=1).sum().fillna(0).values
        active = (roll_trig >= 2).astype(np.int8)
        latched = self._numba_latch_kernel(val, active)
        latched_series = pd.Series(latched, index=idx, dtype='float32').clip(0.0, 1.0)
        locked_val = bool(active[-1] > 0) if len(active) > 0 else False
        last_score = float(latched_series.iloc[-1]) if len(latched_series) > 0 else 0.0
        print(f"DEBUG_PROBE:LatchFinal_V9.1.0|Locked={locked_val}|LastVal={last_score:.4f}")
        return latched_series
    @staticmethod
    @jit(nopython=True, cache=True)
    def _numba_latch_kernel(val: np.ndarray, act: np.ndarray) -> np.ndarray:
        n = len(val)
        out = np.zeros(n, dtype=np.float64)
        decay = 0.985
        brk = 0.30
        lck = False
        last = 0.0
        for i in range(n):
            v = val[i]
            if act[i] > 0:
                lck = True
                last = max(np.tanh(v * 2.0) + 0.1, last * decay)
            elif lck:
                if v < brk:
                    lck, last = False, v
                else: last = max(v, last * decay)
            else: last = v
            out[i] = last
        return out
    def _print_debug(self, out: Dict, temp: Dict, ts: pd.Timestamp):
        """【V9.1.0】全息日志穿透。"""
        if "__MISSING_COLUMNS__" in temp:
            out[f"  !! [红军告警] 缺失底层组件: {temp['__MISSING_COLUMNS__']}"] = "触发底垫保护。"
        out[f"  -- [相空间五大高阶张量态] @ {ts.strftime('%Y-%m-%d')}"] = ""
        for k in ["T_VACUUM", "T_STEALTH", "T_CHIP", "T_ASYM", "T_PANIC", "FOLDED", "final_score"]:
            if k in temp:
                out[f"     |-- {k:<15}: {temp[k].loc[ts] if ts in temp[k].index else 0.0:.4f}"] = ""
        self.helper._print_debug_output(out)









