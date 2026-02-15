import pandas as pd
import numpy as np
from typing import Dict
from strategies.trend_following.utils import get_param_value
from strategies.trend_following.intelligence.process.helper import ProcessIntelligenceHelper

class CalculateCovertAccumulation:
    """
    【PROCESS_META_COVERT_ACCUMULATION - v11.0.0_Ultimate 终极全链路探针全息版】
    说明：
    1. 彻底打破信息孤岛：全面引入高频Tick转移效率、盘整吸筹专有特征、博弈烈度及尾盘定价权。
    2. 抛弃外部归一化，类内独立实现基于滚动 Median/MAD 和 Tanh 软饱和的免疫压缩防御。
    3. 引入 HAB (历史累积记忆缓冲) 系统，剥离量纲影响，侦测真实破窗冲击。
    4. 动态微积分系统 (_calculus_gate) 引入滚动方差阈值，根除零基噪音假信号。
    5. 全息全量探针矩阵：Raw原料 -> 节点特征 -> 维度张量 -> 坍缩结果，打通黑盒监控闭环。
    """
    def __init__(self, strategy_instance, helper_instance: ProcessIntelligenceHelper):
        self.strategy = strategy_instance
        self.helper = helper_instance
        self.debug_params = getattr(self.helper, 'debug_params', {})
        self.probe_dates = getattr(self.helper, 'probe_dates', [])

    def _extract_raw(self, df: pd.DataFrame, col: str) -> pd.Series:
        return self.helper._get_safe_series(df, col, np.nan)

    def _robust_tanh_norm(self, series: pd.Series, window: int = 55, invert: bool = False, k: float = 1.0) -> pd.Series:
        rmed = series.rolling(window, min_periods=1).median()
        rmad = (series - rmed).abs().rolling(window, min_periods=1).mean() + 1e-6
        z_score = (series - rmed) / (rmad * 1.4826)
        norm = 0.5 * (np.tanh(k * z_score) + 1.0)
        clamped = norm.clip(0.0, 1.0).astype(np.float32)
        return (1.0 - clamped) if invert else clamped

    def _calculus_gate(self, df: pd.DataFrame, base_col: str, window: int, derivative: str, invert: bool = False) -> pd.Series:
        base_series = self._extract_raw(df, base_col)
        col_name = f"{derivative.upper()}_{window}_{base_col}"
        raw_deriv = self.helper._get_safe_series(df, col_name, np.nan)
        if raw_deriv.isna().all():
            if derivative.upper() == 'SLOPE':
                raw_deriv = base_series.diff(window)
            elif derivative.upper() == 'ACCEL':
                raw_deriv = base_series.diff(window).diff(window)
            elif derivative.upper() == 'JERK':
                raw_deriv = base_series.diff(window).diff(window).diff(window)
        r_std = base_series.rolling(window, min_periods=1).std() + 1e-6
        gated_momentum = np.tanh(raw_deriv / (r_std * 2.0))
        norm = 0.5 * (gated_momentum + 1.0)
        clamped = norm.clip(0.0, 1.0).astype(np.float32)
        return (1.0 - clamped) if invert else clamped

    def _calculate_hab_impact(self, df: pd.DataFrame, col: str, window: int, invert: bool = False) -> pd.Series:
        val = self._extract_raw(df, col)
        hist_mean = val.abs().rolling(window, min_periods=1).mean() + 1e-6
        impact_ratio = val / hist_mean
        impact_norm = 0.5 * (np.tanh(impact_ratio - 1.0) + 1.0)
        clamped = impact_norm.clip(0.0, 1.0).astype(np.float32)
        return (1.0 - clamped) if invert else clamped

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        method_name = "calculate_covert_accumulation"
        params = config.get('covert_accumulation_params', {})
        is_debug = get_param_value(self.debug_params.get('enabled'), False) and get_param_value(params.get('probe_enabled'), True)
        probe_ts = None
        if is_debug and self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            for date in reversed(df.index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date; break

        r_bbw = self._extract_raw(df, 'BBW_21_2.0_D')
        r_turn = self._extract_raw(df, 'TURNOVER_STABILITY_INDEX_D')
        r_pe = self._extract_raw(df, 'PRICE_ENTROPY_D')
        r_cons = self._extract_raw(df, 'is_consolidating_D')
        r_game = self._extract_raw(df, 'game_intensity_D')
        n_bbw = self._robust_tanh_norm(r_bbw, window=21, invert=True, k=1.5)
        n_turn = self._robust_tanh_norm(r_turn, window=21, k=1.2)
        n_pe = self._robust_tanh_norm(r_pe, window=21, invert=True, k=1.5)
        n_cons = self._robust_tanh_norm(r_cons, window=13, k=1.2)
        n_game = self._robust_tanh_norm(r_game, window=21, invert=True, k=1.2)
        d_vac = ((n_bbw * 0.25 + n_turn * 0.2 + n_pe * 0.2 + n_cons * 0.2 + n_game * 0.15) ** 1.2).astype(np.float32)

        r_stl = self._extract_raw(df, 'stealth_flow_ratio_D')
        r_tick_bal = self._extract_raw(df, 'tick_chip_balance_ratio_D')
        r_tick_eff = self._extract_raw(df, 'tick_chip_transfer_efficiency_D')
        n_stl = self._robust_tanh_norm(r_stl, window=21)
        n_tick_bal = self._robust_tanh_norm(r_tick_bal, window=13, k=1.2)
        n_tick_eff = self._robust_tanh_norm(r_tick_eff, window=13, k=1.2)
        hab_hid = self._calculate_hab_impact(df, 'hidden_accumulation_intensity_D', window=21)
        hab_mf = self._calculate_hab_impact(df, 'net_mf_amount_D', window=34)
        hab_smt = self._calculate_hab_impact(df, 'SMART_MONEY_INST_NET_BUY_D', window=34)
        c_stl_slp = self._calculus_gate(df, 'stealth_flow_ratio_D', window=13, derivative='SLOPE')
        d_stealth = ((n_stl * 0.15 + n_tick_bal * 0.15 + n_tick_eff * 0.1 + hab_hid * 0.2 + hab_mf * 0.15 + hab_smt * 0.15 + c_stl_slp * 0.1) ** 1.1).astype(np.float32)

        r_conc = self._extract_raw(df, 'chip_concentration_ratio_D')
        r_stab = self._extract_raw(df, 'chip_stability_D')
        r_cent = self._extract_raw(df, 'chip_entropy_D')
        r_cons_acc = self._extract_raw(df, 'consolidation_accumulation_score_D')
        n_conc = self._robust_tanh_norm(r_conc, window=34, k=1.2)
        n_stab = self._robust_tanh_norm(r_stab, window=34)
        n_cent = self._robust_tanh_norm(r_cent, window=34, invert=True, k=1.5)
        n_cons_acc = self._robust_tanh_norm(r_cons_acc, window=21, k=1.2)
        hab_c2m = self._calculate_hab_impact(df, 'chip_cost_to_ma21_diff_D', window=21, invert=True)
        c_chp_acc = self._calculus_gate(df, 'chip_concentration_ratio_D', window=21, derivative='ACCEL')
        d_chip = ((n_conc * 0.2 + n_stab * 0.2 + n_cent * 0.15 + n_cons_acc * 0.15 + hab_c2m * 0.15 + c_chp_acc * 0.15) ** 1.1).astype(np.float32)

        r_sup = self._extract_raw(df, 'intraday_support_intent_D')
        r_acc_cf = self._extract_raw(df, 'intraday_accumulation_confidence_D')
        r_vpa_mf = self._extract_raw(df, 'VPA_MF_ADJUSTED_EFF_D')
        r_aft = self._extract_raw(df, 'afternoon_flow_ratio_D')
        r_cls_int = self._extract_raw(df, 'closing_flow_intensity_D')
        n_sup = self._robust_tanh_norm(r_sup, window=13)
        n_acc_cf = self._robust_tanh_norm(r_acc_cf, window=13)
        n_vpa_mf = self._robust_tanh_norm(r_vpa_mf, window=13, k=1.2)
        n_aft = self._robust_tanh_norm(r_aft, window=13)
        n_cls_int = self._robust_tanh_norm(r_cls_int, window=13, k=1.2)
        c_sup_jrk = self._calculus_gate(df, 'intraday_support_intent_D', window=8, derivative='JERK')
        d_intra = ((n_sup * 0.15 + n_acc_cf * 0.15 + n_vpa_mf * 0.2 + n_aft * 0.15 + n_cls_int * 0.2 + c_sup_jrk * 0.15) ** 1.0).astype(np.float32)

        r_snt = self._extract_raw(df, 'market_sentiment_score_D')
        r_acc_sig = self._extract_raw(df, 'accumulation_signal_score_D')
        r_prel = self._extract_raw(df, 'pressure_release_index_D')
        n_snt = self._robust_tanh_norm(r_snt, window=34, invert=True, k=1.5)
        n_acc_sig = self._robust_tanh_norm(r_acc_sig, window=21, k=1.2)
        n_prel = self._robust_tanh_norm(r_prel, window=21, k=1.2)
        hab_trp = self._calculate_hab_impact(df, 'pressure_trapped_D', window=55)
        hab_los = self._calculate_hab_impact(df, 'loser_loss_margin_avg_D', window=34)
        c_snt_slp = self._calculus_gate(df, 'market_sentiment_score_D', window=13, derivative='SLOPE', invert=True)
        d_panic = ((n_snt * 0.2 + n_acc_sig * 0.15 + n_prel * 0.15 + hab_trp * 0.2 + hab_los * 0.15 + c_snt_slp * 0.15) ** 1.2).astype(np.float32)

        w_dim = params.get('dimension_weights', {'volatility_vacuum': 0.20, 'micro_stealth': 0.25, 'chip_negentropy': 0.20, 'intraday_asymmetry': 0.20, 'panic_exhaustion': 0.15})
        tensor_product = ((d_vac ** w_dim.get('volatility_vacuum', 0.20)) * (d_stealth ** w_dim.get('micro_stealth', 0.25)) * (d_chip ** w_dim.get('chip_negentropy', 0.20)) * (d_intra ** w_dim.get('intraday_asymmetry', 0.20)) * (d_panic ** w_dim.get('panic_exhaustion', 0.15)))
        final_score = (tensor_product ** params.get('tensor_folding_power', 1.5)).clip(0.0, 1.0).astype(np.float32)

        if probe_ts is not None and getattr(self.helper, '_print_debug_output', None):
            debug_out = {f"--- {method_name} v11.0.0_Ultimate 全链路探针 @ {probe_ts.strftime('%Y-%m-%d')} ---": ""}
            d_dict = {
                "[RAW]_BBW": r_bbw, "[RAW]_TURN": r_turn, "[RAW]_PE": r_pe, "[RAW]_CONS": r_cons, "[RAW]_GAME": r_game,
                "[NORM]_BBW_INV": n_bbw, "[NORM]_TURN": n_turn, "[NORM]_PE_INV": n_pe, "[NORM]_CONS": n_cons, "[NORM]_GAME_INV": n_game,
                "[RAW]_STL": r_stl, "[RAW]_TICK_BAL": r_tick_bal, "[RAW]_TICK_EFF": r_tick_eff,
                "[NORM]_STL": n_stl, "[NORM]_TICK_BAL": n_tick_bal, "[NORM]_TICK_EFF": n_tick_eff,
                "[HAB]_HID": hab_hid, "[HAB]_MF": hab_mf, "[HAB]_SMT": hab_smt, "[CALC]_STL_SLP": c_stl_slp,
                "[RAW]_CONC": r_conc, "[RAW]_STAB": r_stab, "[RAW]_CENT": r_cent, "[RAW]_CONS_ACC": r_cons_acc,
                "[NORM]_CONC": n_conc, "[NORM]_STAB": n_stab, "[NORM]_CENT_INV": n_cent, "[NORM]_CONS_ACC": n_cons_acc,
                "[HAB]_C2M_INV": hab_c2m, "[CALC]_CHP_ACC": c_chp_acc,
                "[RAW]_SUP": r_sup, "[RAW]_ACC_CF": r_acc_cf, "[RAW]_VPA_MF": r_vpa_mf, "[RAW]_AFT": r_aft, "[RAW]_CLS_INT": r_cls_int,
                "[NORM]_SUP": n_sup, "[NORM]_ACC_CF": n_acc_cf, "[NORM]_VPA_MF": n_vpa_mf, "[NORM]_AFT": n_aft, "[NORM]_CLS_INT": n_cls_int,
                "[CALC]_SUP_JRK": c_sup_jrk,
                "[RAW]_SNT": r_snt, "[RAW]_ACC_SIG": r_acc_sig, "[RAW]_PREL": r_prel,
                "[NORM]_SNT_INV": n_snt, "[NORM]_ACC_SIG": n_acc_sig, "[NORM]_PREL": n_prel,
                "[HAB]_TRP": hab_trp, "[HAB]_LOS": hab_los, "[CALC]_SNT_SLP_INV": c_snt_slp,
                "[DIM]_VAC": d_vac, "[DIM]_STL": d_stealth, "[DIM]_CHP": d_chip, "[DIM]_INTRA": d_intra, "[DIM]_PNC": d_panic,
                "[RES]_TENSOR": tensor_product, "[RES]_FINAL": final_score
            }
            for k, v in d_dict.items():
                debug_out[f"  -> {k}: {v.loc[probe_ts] if probe_ts in v.index else np.nan}"] = ""
            self.helper._print_debug_output(debug_out)

        return final_score