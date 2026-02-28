# strategies\trend_following\intelligence\process\calculate_upthrust_washout.py
# strategies\trend_following\intelligence\process\calculate_upthrust_washout.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from strategies.trend_following.utils import (
    get_params_block, get_param_value, get_adaptive_mtf_normalized_score,
    get_adaptive_mtf_normalized_bipolar_score, normalize_score, _robust_geometric_mean
)
from strategies.trend_following.intelligence.process.helper import ProcessIntelligenceHelper
class CalculateUpthrustWashoutRelationship:
    """
    PROCESS_META_UPTHRUST_WASHOUT
    【V9.0.0 · 时空博弈与强证优先全息版】上冲回落洗盘甄别器
    - 核心职责: 精准识别主力“上冲回落”形态制造虚假抛压、迫使浮筹割肉的隐蔽洗盘行为。
    - 维度升级:
        1. **强证优先 (Strong Evidence Copula)**: 针对主动买盘、下影线强度、权力转移进行局部极大值聚合，暴露单点底牌即锁定洗盘。
        2. **HAB母盘缓冲 (Historical Accumulation Buffer)**: 建立55日主力净买卖股数存量意识，审判当日资金增量与历史母盘的冲击占比。
        3. **非线性张量映射 (Non-linear Tensor Mapping)**: 全局废弃 .clip()，引入 Sigmoid 型软限幅环境，防止边缘极值信息丢失。
        4. **零乘死锁防范 (Deadlock Prevention)**: 采用 robust 几何均值与偏置活性，防止 0 值连乘死锁引发矩阵坍塌。
    - 版本: 9.0.0
    """
    def __init__(self, strategy_instance, helper_instance: ProcessIntelligenceHelper):
        self.strategy = strategy_instance
        self.helper = helper_instance
        self.params = self.helper.params
        self.debug_params = self.helper.debug_params
        self.probe_dates = self.helper.probe_dates
    def _apply_nonlinear_gain(self, x: pd.Series, center: float = 0.0, steepness: float = 1.0) -> pd.Series:
        """
        [V9.0.0 · 非线性张量软边界增益引擎]
        - 消除传统的硬截断 .clip()，采用 Sigmoid 变体进行平滑软压缩，规避极性反噬与梯度丢失。
        """
        return 0.5 * (1.0 + np.tanh((x - center) * steepness))
    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        [V9.0.0 · 全链路执行总控]
        - 融合物理、资金、筹码、取证四层模型。引入微积分加成与 55日 HAB 存量冲击验证。
        """
        method_name = "CalculateUpthrustWashoutRelationship.calculate"
        (open_p, high_p, low_p, close_p, pct_chg, p_ent, slope_p, jerk_p, atr_14, vol_f20, c_mean, p_mig, c_mig, r_f, j_f, smart_n, smart_d, f_acc13, f_acc34, f_acc55, t_acc55, vpa_eff, och_acc, bbp_pos, hm_att, sm_att, p_trap, p_rel, is_ldr, st_ldr, t_hot, t_stab, win, a_win, c_stab, a_stab, c_diff, c_kurt, c_conv, is_multi, p_cnt, c_ent, dw_s, morn, ste, h_lock, skew, h_rel, t_over, clust, vol, cnt, t_net, ma_r, sl_t, sr_ratio, supp_test, cost_mig, acc_supp, tick_abnorm, large_anom, acc_abnorm, net_mf_vol, buy_elg_vol, buy_lg_vol, net_mf_amount, closing_strength, amount, mf_vol_slope, mf_vol_accel, mf_vol_jerk) = self._get_raw_signals(df, method_name)
        df_index = df.index
        core_checks = {'net_mf_vol_D': net_mf_vol, 'net_mf_amount_D': net_mf_amount, 'amount_D': amount}
        missing = [k for k, v in core_checks.items() if v.isna().all()]
        if missing:
            print(f"[{method_name}] ⚠️ 全息探针预警：核心HAB衍生数据 {missing} 缺失，系统将依靠弱维度安全降级运行。")
        fallback_ratio = (high_p - close_p) / high_p.replace(0, 1e-9)
        form_gate = self._apply_nonlinear_gain(fallback_ratio, center=0.03, steepness=50.0)
        active_buy_ratio = (buy_elg_vol + buy_lg_vol) / vol.replace(0, 1e-9)
        ev_active_buy = self._apply_nonlinear_gain(active_buy_ratio, center=0.15, steepness=10.0)
        day_range = (high_p - low_p).replace(0, 1e-9)
        lower_shadow_calc = (np.minimum(open_p, close_p) - low_p) / day_range
        ls_val = np.where(closing_strength.notna(), closing_strength / 100.0, lower_shadow_calc)
        ev_shadow = self._apply_nonlinear_gain(pd.Series(ls_val, index=df_index), center=0.3, steepness=5.0)
        power_transfer = np.where((t_net < 0) & (net_mf_amount > 0), net_mf_amount / t_net.abs().replace(0, 1e-9), 0.0)
        ev_transfer = self._apply_nonlinear_gain(pd.Series(power_transfer, index=df_index), center=0.0, steepness=10.0)
        strong_evidence = np.maximum.reduce([ev_active_buy.values, ev_shadow.values, ev_transfer.values])
        strong_evidence = pd.Series(strong_evidence, index=df_index)
        mf_jerk_scale = mf_vol_jerk.rolling(21, min_periods=1).std().replace(0, 1e-9)
        jerk_bonus = self._apply_nonlinear_gain(mf_vol_jerk / mf_jerk_scale, center=0.0, steepness=2.0)
        strong_evidence = strong_evidence * (1.0 + jerk_bonus * 0.2)
        hab_55_mf_vol = net_mf_vol.rolling(55, min_periods=1).sum()
        impact_intensity = net_mf_vol / (hab_55_mf_vol.abs() + 1e-9)
        dump_severity = np.where((net_mf_vol < 0) & (impact_intensity < -0.1), np.abs(impact_intensity), 0.0)
        hab_retention_gate = 1.0 - self._apply_nonlinear_gain(pd.Series(dump_severity, index=df_index), center=0.2, steepness=10.0)
        k_trap = self._assess_kinematic_trap_physics(close_p, high_p, slope_p, jerk_p, vpa_eff, och_acc, bbp_pos)
        f_res = self._assess_fund_jerk_resonance(pct_chg, j_f, smart_n, smart_d)
        f_split = self._assess_split_order_accumulation(vol, cnt, clust, r_f, t_net)
        f_base = self._assess_multi_cycle_fund_reservoir(f_acc13, f_acc34, f_acc55, t_acc55, hm_att, sm_att)
        f_energy = self._assess_cumulative_energy_decay(r_f, t_net)
        fund_score = np.maximum(f_res, f_split) * f_base * f_energy
        chip_migration = self._assess_chip_holographic_migration(close_p, c_mean, p_mig, c_mig)
        chip_collapse = self._assess_chip_peak_collapse(is_multi, p_cnt)
        chip_order = self._assess_chip_entropy_ordered_collapse(c_ent, p_cnt)
        chip_tension = self._assess_chip_holographic_tension(h_rel, t_over, win, a_win, c_stab, a_stab, c_kurt, c_conv, c_diff)
        chip_lock = self._assess_turnover_lock_stability(t_stab, is_ldr, c_stab)
        chip_score = (chip_migration * 0.25 + chip_collapse * 0.25 + chip_order * 0.2 + chip_tension * 0.15 + chip_lock * 0.15)
        intent_score = self._assess_coordinated_intent_forensics(hm_att, sm_att, p_trap, p_rel)
        ssd_dec = self._assess_stealth_shadow_divergence(morn, ste, h_lock, skew, pct_chg, t_net)
        fractal_manip = self._assess_fractal_manipulation_fingerprint(tick_abnorm, clust, large_anom, acc_abnorm)
        forensic_score = (intent_score * 0.4 + ssd_dec * 0.4 + fractal_manip * 0.2)
        leader_bonus = self._assess_market_leadership_compensation(is_ldr, st_ldr, t_hot)
        atr_norm = (atr_14 / close_p.replace(0, 1e-9)).rolling(60, min_periods=1).rank(pct=True)
        beta_hedge = self._apply_nonlinear_gain(1.2 - atr_norm, center=0.5, steepness=3.0)
        e_scale = p_ent.rolling(60, min_periods=1).mean().replace(0, 1e-9)
        entropy_multi = self._apply_nonlinear_gain(1.2 - (p_ent / e_scale), center=0.0, steepness=3.0)
        sync_gate = np.where(dw_s == 1, 1.1, 0.9)
        core_washout_raw = form_gate * strong_evidence * hab_retention_gate
        context_df = pd.concat([pd.Series(k_trap + 0.1), pd.Series(fund_score + 0.1), pd.Series(chip_score + 0.1), pd.Series(forensic_score + 0.1)], axis=1)
        context_mean = _robust_geometric_mean(context_df) - 0.1
        context_mean = np.maximum(context_mean, 0.0)
        final_raw = core_washout_raw * (context_mean + 0.1) * leader_bonus * beta_hedge * entropy_multi * sync_gate
        final_score = self._apply_nonlinear_gain(final_raw, center=0.2, steepness=4.0)
        debug_ctx = {
            "Final": final_score, "Phys": k_trap, "Fund": fund_score, "Chip": chip_score, "Forens": forensic_score, "Ldr": leader_bonus,
            "Raw": {"FormGate": form_gate, "Evidence": strong_evidence, "HAB_Ret": hab_retention_gate, "JerkP": jerk_p, "Split": f_split, "Reson": f_res, "Base": f_base, "Energy": f_energy, "Mig": chip_migration, "Coll": chip_collapse, "Order": chip_order, "Lock": chip_lock, "SSD": ssd_dec, "Trap": p_trap, "HMAtt": hm_att, "Theme": t_hot}
        }
        self._print_debug_probe(df_index, debug_ctx)
        return final_score.astype(np.float32).fillna(0.0)
    def _print_debug_probe(self, idx: pd.Index, ctx: Dict[str, Any]):
        """
        [V9.0.0 · 全链路深度溯源探针]
        - 揭示从核心门限验证到全息上下文组合计算的全息状态。
        """
        if len(idx) > 0:
            i = -1
            dt = idx[i].strftime('%Y-%m-%d')
            raw = ctx["Raw"]
            print(f"--- [ CalculateUpthrustWashoutRelationship PROBE_V9_FULL_LINK ] {dt} FINAL: {ctx['Final'].iloc[i]:.4f} ---")
            print(f"  [0.Core]     FormGate: {raw['FormGate'].iloc[i]:.4f} | StrongEvid: {raw['Evidence'].iloc[i]:.4f} | HAB_Ret: {raw['HAB_Ret'].iloc[i]:.4f}")
            print(f"  [1.Physics]  Trap: {ctx['Phys'].iloc[i]:.4f} (PriceJerk: {raw['JerkP'].iloc[i]:.2f})")
            print(f"  [2.Funds]    Score: {ctx['Fund'].iloc[i]:.4f} <- (Reson: {raw['Reson'].iloc[i]:.2f}, Split: {raw['Split'].iloc[i]:.2f}, Base: {raw['Base'].iloc[i]:.2f}, Energy: {raw['Energy'].iloc[i]:.2f})")
            print(f"  [3.Chips]    Score: {ctx['Chip'].iloc[i]:.4f} <- (Mig: {raw['Mig'].iloc[i]:.2f}, Coll: {raw['Coll'].iloc[i]:.2f}, Order: {raw['Order'].iloc[i]:.2f}, Lock: {raw['Lock'].iloc[i]:.2f})")
            print(f"  [4.Forens]   Score: {ctx['Forens'].iloc[i]:.4f} <- (SSD: {raw['SSD'].iloc[i]:.2f}, HMAttack: {raw['HMAtt'].iloc[i]}, Trapped: {raw['Trap'].iloc[i]:.2f})")
            print(f"  [5.Strategy] LeaderBonus: {ctx['Ldr'].iloc[i]:.2f}x (ThemeHot: {raw['Theme'].iloc[i]:.1f})")
            print(f"--- ========================================================== ---")
    def _get_raw_signals(self, df: pd.DataFrame, method_name: str) -> Tuple[pd.Series, ...]:
        """
        [V9.0.0 · 全维信号接入终极版]
        - 补足 7 项核心数据，并使用微积分零基陷阱门限拦截噪音。
        """
        open_p = self.helper._get_safe_series(df, 'open_D', np.nan, method_name=method_name)
        high_p = self.helper._get_safe_series(df, 'high_D', np.nan, method_name=method_name)
        low_p = self.helper._get_safe_series(df, 'low_D', np.nan, method_name=method_name)
        close_p = self.helper._get_safe_series(df, 'close_D', np.nan, method_name=method_name)
        pct_chg = self.helper._get_safe_series(df, 'pct_change_D', np.nan, method_name=method_name)
        p_entropy = self.helper._get_safe_series(df, 'PRICE_ENTROPY_D', np.nan, method_name=method_name)
        slope_p = self.helper._get_safe_series(df, 'SLOPE_3_close_D', np.nan, method_name=method_name)
        jerk_p = self.helper._get_safe_series(df, 'JERK_3_close_D', np.nan, method_name=method_name)
        atr_14 = self.helper._get_safe_series(df, 'ATR_14_D', np.nan, method_name=method_name)
        vol_f_20 = self.helper._get_safe_series(df, 'flow_volatility_21d_D', np.nan, method_name=method_name)
        c_mean = self.helper._get_safe_series(df, 'chip_mean_D', np.nan, method_name=method_name)
        peak_mig = self.helper._get_safe_series(df, 'peak_migration_speed_5d_D', np.nan, method_name=method_name)
        conv_mig = self.helper._get_safe_series(df, 'convergence_migration_D', np.nan, method_name=method_name)
        raw_fund = self.helper._get_safe_series(df, 'tick_large_order_net_D', np.nan, method_name=method_name)
        jerk_f = self.helper._get_safe_series(df, 'JERK_3_tick_large_order_net_D', np.nan, method_name=method_name)
        smart_n = self.helper._get_safe_series(df, 'SMART_MONEY_HM_NET_BUY_D', np.nan, method_name=method_name)
        smart_d = self.helper._get_safe_series(df, 'SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D', np.nan, method_name=method_name)
        total_net = self.helper._get_safe_series(df, 'net_amount_D', np.nan, method_name=method_name)
        f_acc13 = raw_fund.rolling(13, min_periods=1).sum()
        f_acc34 = raw_fund.rolling(34, min_periods=1).sum()
        f_acc55 = raw_fund.rolling(55, min_periods=1).sum()
        t_acc55 = total_net.rolling(55, min_periods=1).sum()
        vpa_eff = self.helper._get_safe_series(df, 'VPA_MF_ADJUSTED_EFF_D', np.nan, method_name=method_name)
        och_acc = self.helper._get_safe_series(df, 'OCH_ACCELERATION_D', np.nan, method_name=method_name)
        bbp_pos = self.helper._get_safe_series(df, 'BBP_21_2.0_D', np.nan, method_name=method_name)
        hm_attack = self.helper._get_safe_series(df, 'HM_COORDINATED_ATTACK_D', 0, method_name=method_name)
        sm_attack = self.helper._get_safe_series(df, 'SMART_MONEY_HM_COORDINATED_ATTACK_D', 0, method_name=method_name)
        p_trapped = self.helper._get_safe_series(df, 'pressure_trapped_D', np.nan, method_name=method_name)
        p_release = self.helper._get_safe_series(df, 'pressure_release_index_D', np.nan, method_name=method_name)
        is_leader = self.helper._get_safe_series(df, 'IS_MARKET_LEADER_D', 0, method_name=method_name)
        state_leader = self.helper._get_safe_series(df, 'STATE_MARKET_LEADER_D', 0, method_name=method_name)
        theme_hot = self.helper._get_safe_series(df, 'THEME_HOTNESS_SCORE_D', 0, method_name=method_name)
        turnover_stable = self.helper._get_safe_series(df, 'TURNOVER_STABILITY_INDEX_D', np.nan, method_name=method_name)
        winner = self.helper._get_safe_series(df, 'winner_rate_D', np.nan, method_name=method_name)
        acc_win = self.helper._get_safe_series(df, 'ACCEL_5_winner_rate_D', np.nan, method_name=method_name)
        chip_stab = self.helper._get_safe_series(df, 'chip_stability_D', np.nan, method_name=method_name)
        acc_stab_21 = self.helper._get_safe_series(df, 'ACCEL_21_chip_stability_D', np.nan, method_name=method_name)
        cost_diff = self.helper._get_safe_series(df, 'chip_cost_to_ma21_diff_D', np.nan, method_name=method_name)
        chip_kurt = self.helper._get_safe_series(df, 'chip_kurtosis_D', np.nan, method_name=method_name)
        chip_conv = self.helper._get_safe_series(df, 'chip_convergence_ratio_D', np.nan, method_name=method_name)
        is_multi = self.helper._get_safe_series(df, 'is_multi_peak_D', 0.0, method_name=method_name)
        peak_cnt = self.helper._get_safe_series(df, 'peak_count_D', 1.0, method_name=method_name)
        chip_ent = self.helper._get_safe_series(df, 'chip_entropy_D', np.nan, method_name=method_name)
        dw_sync = self.helper._get_safe_series(df, 'daily_weekly_sync_D', 0, method_name=method_name)
        morning = self.helper._get_safe_series(df, 'morning_flow_ratio_D', np.nan, method_name=method_name)
        stealth = self.helper._get_safe_series(df, 'stealth_flow_ratio_D', np.nan, method_name=method_name)
        high_lock = self.helper._get_safe_series(df, 'intraday_high_lock_ratio_D', np.nan, method_name=method_name)
        skew = self.helper._get_safe_series(df, 'intraday_price_distribution_skewness_D', np.nan, method_name=method_name)
        pres_rel = self.helper._get_safe_series(df, 'pressure_release_index_D', np.nan, method_name=method_name)
        hab_rel = pres_rel.rolling(window=5, min_periods=1).sum()
        turnover = self.helper._get_safe_series(df, 'turnover_rate_D', np.nan, method_name=method_name)
        clustering = self.helper._get_safe_series(df, 'tick_clustering_index_D', np.nan, method_name=method_name)
        volume = self.helper._get_safe_series(df, 'volume_D', np.nan, method_name=method_name)
        trade_count = self.helper._get_safe_series(df, 'trade_count_D', np.nan, method_name=method_name)
        ma_res = self.helper._get_safe_series(df, 'MA_COHERENCE_RESONANCE_D', np.nan, method_name=method_name)
        slope_t = self.helper._get_safe_series(df, 'GEOM_REG_SLOPE_D', np.nan, method_name=method_name)
        sr_ratio = self.helper._get_safe_series(df, 'support_resistance_ratio_D', np.nan, method_name=method_name)
        supp_test = self.helper._get_safe_series(df, 'intraday_support_test_count_D', np.nan, method_name=method_name)
        cost_mig = self.helper._get_safe_series(df, 'intraday_cost_center_migration_D', np.nan, method_name=method_name)
        raw_supp = self.helper._get_safe_series(df, 'support_strength_D', np.nan, method_name=method_name)
        acc_supp = raw_supp.diff(5) / 5.0
        tick_abnorm = self.helper._get_safe_series(df, 'tick_abnormal_volume_ratio_D', np.nan, method_name=method_name)
        large_anom = self.helper._get_safe_series(df, 'large_order_anomaly_D', np.nan, method_name=method_name)
        acc_abnorm = tick_abnorm.diff(5) / 5.0
        net_mf_vol = self.helper._get_safe_series(df, 'net_mf_vol_D', np.nan, method_name=method_name)
        net_mf_amount = self.helper._get_safe_series(df, 'net_mf_amount_D', np.nan, method_name=method_name)
        amount = self.helper._get_safe_series(df, 'amount_D', np.nan, method_name=method_name)
        buy_elg_vol = self.helper._get_safe_series(df, 'buy_elg_vol_D', np.nan, method_name=method_name)
        buy_lg_vol = self.helper._get_safe_series(df, 'buy_lg_vol_D', np.nan, method_name=method_name)
        closing_strength = self.helper._get_safe_series(df, 'CLOSING_STRENGTH_D', np.nan, method_name=method_name)
        mf_vol_slope = net_mf_vol.diff(5) / 5.0
        mf_vol_slope = pd.Series(np.where(mf_vol_slope.abs() < 1e-4, 0.0, mf_vol_slope), index=df.index)
        mf_vol_accel = mf_vol_slope.diff(5) / 5.0
        mf_vol_accel = pd.Series(np.where(mf_vol_accel.abs() < 1e-5, 0.0, mf_vol_accel), index=df.index)
        mf_vol_jerk = mf_vol_accel.diff(5) / 5.0
        mf_vol_jerk = pd.Series(np.where(mf_vol_jerk.abs() < 1e-6, 0.0, mf_vol_jerk), index=df.index)
        return (open_p, high_p, low_p, close_p, pct_chg, p_entropy, slope_p, jerk_p, atr_14, vol_f_20, c_mean, peak_mig, conv_mig, raw_fund, jerk_f, smart_n, smart_d, f_acc13, f_acc34, f_acc55, t_acc55, vpa_eff, och_acc, bbp_pos, hm_attack, sm_attack, p_trapped, p_release, is_leader, state_leader, theme_hot, turnover_stable, winner, acc_win, chip_stab, acc_stab_21, cost_diff, chip_kurt, chip_conv, is_multi, peak_cnt, chip_ent, dw_sync, morning, stealth, high_lock, skew, hab_rel, turnover, clustering, volume, trade_count, total_net, ma_res, slope_t, sr_ratio, supp_test, cost_mig, acc_supp, tick_abnorm, large_anom, acc_abnorm, net_mf_vol, buy_elg_vol, buy_lg_vol, net_mf_amount, closing_strength, amount, mf_vol_slope, mf_vol_accel, mf_vol_jerk)
    def _assess_multi_period_volatility_hedge(self, s5: pd.Series, s20: pd.Series, s60: pd.Series, m5: pd.Series, m20: pd.Series, m60: pd.Series) -> pd.Series:
        r5 = 5.0 * np.tanh(s5 / (m5 * 5.0 + 1e-9))
        r20 = 5.0 * np.tanh(s20 / (m20 * 5.0 + 1e-9))
        r60 = 5.0 * np.tanh(s60 / (m60 * 5.0 + 1e-9))
        tension = (r5 * 0.2 + r20 * 0.4 + r60 * 0.4)
        hedge_factor = self._apply_nonlinear_gain(1.2 - np.tanh(tension / 2.0), center=0.5, steepness=3.0)
        return hedge_factor.fillna(1.0).astype(np.float32)
    def _assess_split_order_accumulation(self, volume: pd.Series, count: pd.Series, clustering: pd.Series, large_net: pd.Series, total_net: pd.Series) -> pd.Series:
        avg_trade_size = volume / (count + 1e-9)
        baseline_size = avg_trade_size.rolling(20, min_periods=1).mean()
        frag_ratio = baseline_size / (avg_trade_size + 1e-9)
        n_frag = self._apply_nonlinear_gain(frag_ratio, center=1.0, steepness=3.0)
        n_clustering = self._apply_nonlinear_gain(clustering, center=0.3, steepness=5.0)
        small_medium_net = total_net - large_net
        sm_scale = small_medium_net.rolling(60, min_periods=1).std().replace(0, 1e-9)
        n_sm_net = self._apply_nonlinear_gain(small_medium_net / sm_scale, center=0.0, steepness=2.0)
        large_mask = np.where(large_net <= 0, 1.2, 0.8)
        stealth_flow = self._apply_nonlinear_gain(n_sm_net * large_mask, center=0.5, steepness=3.0)
        split_score = self._apply_nonlinear_gain(n_frag * 0.3 + n_clustering * 0.3 + stealth_flow * 0.4, center=0.5, steepness=3.0)
        return split_score.fillna(0.0)
    def _assess_chip_entropy_ordered_collapse(self, entropy: pd.Series, peak_cnt: pd.Series) -> pd.Series:
        ent_delta = entropy.diff(5).fillna(0)
        entropy_decay = np.where(ent_delta < 0, ent_delta.abs() / (entropy + 1e-9), 0.0)
        order_score = self._apply_nonlinear_gain(np.tanh(entropy_decay * 5.0) * np.where(peak_cnt <= 2, 1.2, 0.8), center=0.5, steepness=3.0)
        return pd.Series(order_score, index=entropy.index).astype(np.float32)
    def _assess_coordinated_intent_forensics(self, hm_attack: pd.Series, sm_attack: pd.Series, trapped: pd.Series, release: pd.Series) -> pd.Series:
        co_strength = self._apply_nonlinear_gain(hm_attack * 0.4 + sm_attack * 0.6, center=0.5, steepness=3.0)
        t_scale = trapped.rolling(60, min_periods=1).max().replace(0, 1e-9)
        low_pressure_gate = self._apply_nonlinear_gain(1.0 - np.tanh(trapped / t_scale), center=0.5, steepness=4.0)
        intent_score = (co_strength * low_pressure_gate)
        release_bonus = self._apply_nonlinear_gain(release.rolling(5).mean(), center=0.0, steepness=2.0)
        return self._apply_nonlinear_gain(intent_score + release_bonus * 0.3, center=0.5, steepness=2.0).fillna(0.0)
    def _assess_multi_cycle_fund_reservoir(self, f_acc13: pd.Series, f_acc34: pd.Series, f_acc55: pd.Series, t_acc55: pd.Series, hm_attack: pd.Series, smart_attack: pd.Series) -> pd.Series:
        base_55 = self._apply_nonlinear_gain(f_acc55 / f_acc55.rolling(120, min_periods=1).std().replace(0, 1e-9), center=0.0, steepness=2.0)
        resonance = (np.where((f_acc13 > 0) & (f_acc34 > 0), 1.2, 0.8) * np.where(f_acc55 > 0, 1.1, 0.5))
        attack_bonus = self._apply_nonlinear_gain(hm_attack * 0.3 + smart_attack * 0.4, center=0.5, steepness=3.0) + 1.0
        reservoir_score = self._apply_nonlinear_gain(base_55 * resonance * attack_bonus, center=1.0, steepness=2.0)
        penalty = np.where(t_acc55 < 0, 0.6, 1.0)
        return pd.Series(reservoir_score * penalty, index=f_acc55.index).astype(np.float32)
    def _assess_fund_reservoir_buffer(self, daily_large: pd.Series, accum_large: pd.Series, accum_total: pd.Series, f_split: pd.Series, slope_f: pd.Series, accel_f: pd.Series, vpa_eff: pd.Series, volat: pd.Series) -> pd.Series:
        mixed_accum = accum_large * (1.0 - f_split) + accum_total * f_split
        accum_median = mixed_accum.rolling(120, min_periods=1).median().replace(0, 1e-9)
        reservoir_strength = self._apply_nonlinear_gain(mixed_accum / accum_median.abs(), center=0.0, steepness=2.0)
        split_credit = self._apply_nonlinear_gain(f_split.rolling(3, min_periods=1).min(), center=0.5, steepness=3.0) * 0.2
        reservoir_strength = self._apply_nonlinear_gain(reservoir_strength + split_credit, center=0.5, steepness=2.0)
        system_penalty = np.where(accum_total < 0, 0.5, 1.0)
        s_scale = slope_f.rolling(60, min_periods=1).std().replace(0, 1e-9)
        norm_s = np.tanh(slope_f / (s_scale * 1.5))
        a_scale = accel_f.rolling(60, min_periods=1).std().replace(0, 1e-9)
        norm_a = np.tanh(accel_f / (a_scale * 1.5))
        trend_score = (norm_s * 0.6 + norm_a * 0.4 + 1.0) / 2.0
        quality_f = self._apply_nonlinear_gain(vpa_eff.rolling(5, min_periods=1).mean(), center=0.0, steepness=3.0) + 0.5
        dynamic_threshold = self._apply_nonlinear_gain(volat * 2.0, center=0.15, steepness=10.0) * 0.2 + 0.05
        attrition = daily_large.abs() / (mixed_accum.abs() + 1e-9)
        tolerance = self._apply_nonlinear_gain(-(attrition / dynamic_threshold), center=-1.0, steepness=3.0)
        return self._apply_nonlinear_gain(reservoir_strength * trend_score * quality_f * tolerance * system_penalty, center=0.5, steepness=3.0).fillna(0.0)
    def _assess_cumulative_energy_decay(self, raw_f: pd.Series, total_net: pd.Series) -> pd.Series:
        daily_press = np.where((raw_f < 0) & (total_net < 0), (raw_f.abs() / (total_net.abs() + 1e-9)), 0.0)
        daily_press = 5.0 * np.tanh(pd.Series(daily_press, index=raw_f.index) / 5.0)
        cum_press = daily_press.ewm(alpha=0.2, adjust=False).mean()
        energy_decay_factor = self._apply_nonlinear_gain(-cum_press, center=-2.0, steepness=2.0)
        return energy_decay_factor.astype(np.float32)
    def _assess_chip_holographic_tension(self, hab_rel: pd.Series, turnover: pd.Series, winner: pd.Series, acc_win: pd.Series, stability: pd.Series, acc_stab: pd.Series, kurtosis: pd.Series, convergence: pd.Series, cost_diff: pd.Series) -> pd.Series:
        efficiency = self._apply_nonlinear_gain(hab_rel / (turnover + 0.5), center=1.0, steepness=2.0)
        k_scale = kurtosis.rolling(60, min_periods=1).std().replace(0, 1e-9)
        n_kurt = self._apply_nonlinear_gain(kurtosis / k_scale, center=0.0, steepness=2.0)
        thickness_score = self._apply_nonlinear_gain(n_kurt * 0.6 + np.tanh(convergence) * 0.4, center=0.5, steepness=3.0)
        cost_std = cost_diff.rolling(60, min_periods=1).std().replace(0, 1e-9)
        cost_gate = self._apply_nonlinear_gain(cost_diff + 2.0 * cost_std, center=0.0, steepness=10.0)
        w_scale = acc_win.rolling(60, min_periods=1).std().replace(0, 1e-9)
        s_scale = acc_stab.rolling(60, min_periods=1).std().replace(0, 1e-9)
        n_acc_win = np.tanh(acc_win / w_scale)
        n_acc_stab = np.tanh(acc_stab / s_scale)
        stability_gate = self._apply_nonlinear_gain(1.0 + n_acc_stab, center=1.0, steepness=2.0)
        tension = np.where(cost_diff < 0, convergence * 1.2, convergence * 0.8)
        n_tension = self._apply_nonlinear_gain(pd.Series(tension), center=0.0, steepness=3.0)
        final_meta = (efficiency * 0.3 + thickness_score * 0.4 + n_tension * 0.3) * stability_gate * cost_gate
        return final_meta.fillna(0.0).astype(np.float32)
    def _assess_chip_holographic_migration(self, close: pd.Series, c_mean: pd.Series, peak_mig: pd.Series, conv_mig: pd.Series) -> pd.Series:
        fib_list = [5, 13, 21, 34, 55]
        migration_scores = []
        for period in fib_list:
            p_vel = (close.diff(period) / close.shift(period).replace(0, 1e-9)).abs()
            c_vel = (c_mean.diff(period) / c_mean.shift(period).replace(0, 1e-9)).abs()
            m_ratio = 3.0 * np.tanh((p_vel / (c_vel + 0.01)) / 3.0)
            migration_scores.append(m_ratio)
        holographic_mig = pd.concat(migration_scores, axis=1).mean(axis=1)
        c_gate = self._apply_nonlinear_gain(conv_mig.abs().rolling(21, min_periods=1).mean(), center=0.5, steepness=3.0)
        final_mig = self._apply_nonlinear_gain(holographic_mig * (1.0 + np.tanh(peak_mig / 100.0)), center=0.5, steepness=3.0)
        return final_mig.fillna(0.0).astype(np.float32)
    def _assess_stealth_shadow_divergence(self, morning: pd.Series, stealth: pd.Series, high_lock: pd.Series, skew: pd.Series, pct_chg: pd.Series, total_net: pd.Series) -> pd.Series:
        n_morning = self._apply_nonlinear_gain(morning, center=0.4, steepness=3.0)
        net_accel = total_net.diff().diff().rolling(5, min_periods=1).mean().fillna(0)
        n_stealth = self._apply_nonlinear_gain(stealth * 3.0 * (np.tanh(net_accel) + 1.0) / 2.0, center=0.5, steepness=3.0)
        lure_score = (n_morning * 0.4 + n_stealth * 0.6)
        trap_score = self._apply_nonlinear_gain(high_lock, center=0.3, steepness=4.0)
        s_scale = skew.abs().rolling(60, min_periods=1).max().replace(0, 1e-9)
        kill_score = self._apply_nonlinear_gain(-skew / s_scale, center=0.0, steepness=3.0)
        is_drop = (pct_chg < 0).astype(int)
        ssd_final = ((lure_score * 0.5 + trap_score * 0.2 + kill_score * 0.3) * is_drop).fillna(0.0)
        return ssd_final.astype(np.float32)
    def _assess_kinematic_trap_physics(self, close: pd.Series, high: pd.Series, slope: pd.Series, jerk: pd.Series, vpa_eff: pd.Series, och_acc: pd.Series, bbp: pd.Series) -> pd.Series:
        s_scale = slope.rolling(60, min_periods=1).std().replace(0, 1e-9)
        n_slope = self._apply_nonlinear_gain(slope / s_scale, center=0.0, steepness=2.0)
        j_scale = jerk.rolling(60, min_periods=1).std().replace(0, 1e-9)
        v_breaker = self._apply_nonlinear_gain(jerk + 2.5 * j_scale, center=0.0, steepness=10.0)
        exhaustion = self._apply_nonlinear_gain(-jerk / j_scale, center=1.0, steepness=2.0)
        vpa_scale = vpa_eff.abs().rolling(60, min_periods=1).max().replace(0, 1e-9)
        inefficiency = self._apply_nonlinear_gain(-vpa_eff / vpa_scale, center=0.0, steepness=3.0)
        och_scale = och_acc.abs().rolling(60, min_periods=1).max().replace(0, 1e-9)
        structure_fail = self._apply_nonlinear_gain(-och_acc / och_scale, center=0.0, steepness=3.0)
        context_gate = self._apply_nonlinear_gain(bbp, center=0.5, steepness=3.0)
        range_hl = (high - close) / high.replace(0, 1e-9)
        retracement = self._apply_nonlinear_gain(range_hl * 10.0, center=0.5, steepness=4.0)
        base_trap = (n_slope * 0.4 + exhaustion * 0.6)
        quality = (inefficiency * 0.3 + structure_fail * 0.3 + retracement * 0.4)
        return self._apply_nonlinear_gain(base_trap * quality * context_gate * v_breaker, center=0.3, steepness=3.0).fillna(0.0)
    def _assess_fund_energy_divergence(self, raw_f: pd.Series, total_net: pd.Series) -> pd.Series:
        f_ma = raw_f.rolling(5, min_periods=1).mean()
        t_ma = total_net.rolling(5, min_periods=1).mean()
        energy_ratio = (f_ma.abs() / (t_ma.abs() + 1e-9))
        punishment = np.where((raw_f < 0) & (energy_ratio > 2.0), 1.0 / energy_ratio, 1.0)
        return self._apply_nonlinear_gain(pd.Series(punishment, index=raw_f.index), center=0.5, steepness=5.0).fillna(1.0)
    def _assess_fund_jerk_resonance(self, pct_chg: pd.Series, jerk_large: pd.Series, smart_net: pd.Series, smart_div: pd.Series) -> pd.Series:
        j_scale = jerk_large.rolling(60, min_periods=1).std().replace(0, 1e-9)
        panic_impulse = self._apply_nonlinear_gain(-jerk_large / j_scale, center=1.0, steepness=2.0)
        s_scale = smart_net.rolling(60, min_periods=1).std().replace(0, 1e-9)
        absorption = self._apply_nonlinear_gain(smart_net / s_scale, center=0.0, steepness=2.0)
        p_scale = pct_chg.abs().rolling(60, min_periods=1).mean().replace(0, 1e-9) + 1e-5
        elasticity = self._apply_nonlinear_gain(panic_impulse / (pct_chg.abs() / p_scale), center=1.0, steepness=2.0)
        div_bonus = self._apply_nonlinear_gain(smart_div, center=0.0, steepness=2.0)
        resonance_score = self._apply_nonlinear_gain(panic_impulse * 0.4 + absorption * 0.3 + elasticity * 0.2 + div_bonus * 0.1, center=0.5, steepness=3.0)
        validity_gate = np.where(smart_net > 0, 1.0, 0.5) 
        return (resonance_score * validity_gate).fillna(0.0)
    def _assess_chip_peak_collapse(self, is_multi: pd.Series, peak_cnt: pd.Series) -> pd.Series:
        peak_delta = peak_cnt.diff(3).fillna(0)
        collapse_signal = np.where((is_multi.shift(3) == 1) & (peak_delta < 0), peak_delta.abs() / (peak_cnt + 1.0), 0.0)
        stability_bonus = np.where(peak_cnt == 1, 0.3, 0.0)
        final_collapse = self._apply_nonlinear_gain(pd.Series(collapse_signal) + stability_bonus, center=0.5, steepness=3.0)
        return final_collapse.fillna(0.0).astype(np.float32)
    def _assess_structural_stress_test(self, ratio_sr: pd.Series, test_count: pd.Series, cost_mig: pd.Series, acc_supp: pd.Series) -> pd.Series:
        base_solidity = self._apply_nonlinear_gain(ratio_sr, center=0.8, steepness=4.0)
        test_bonus = self._apply_nonlinear_gain(np.log1p(np.maximum(0, test_count)), center=1.0, steepness=2.0)
        resilience = base_solidity * (0.5 + 0.5 * test_bonus)
        c_scale = cost_mig.abs().rolling(60, min_periods=1).mean().replace(0, 1e-9)
        n_mig = cost_mig / c_scale
        gravity_stable = self._apply_nonlinear_gain(n_mig, center=0.0, steepness=2.0)
        a_scale = acc_supp.rolling(60, min_periods=1).std().replace(0, 1e-9)
        n_acc = np.tanh(acc_supp / a_scale)
        boost = self._apply_nonlinear_gain(pd.Series(n_acc), center=0.0, steepness=2.0) + 0.5
        return self._apply_nonlinear_gain(resilience * gravity_stable * boost, center=0.5, steepness=3.0).fillna(0.0)
    def _assess_skewed_deception_narrative(self, morning: pd.Series, stealth: pd.Series, high_lock: pd.Series, skew: pd.Series, pct_chg: pd.Series) -> pd.Series:
        n_morning = self._apply_nonlinear_gain(morning, center=0.4, steepness=3.0)
        n_stealth = self._apply_nonlinear_gain(stealth, center=0.5, steepness=2.0)
        lure_score = (n_morning * 0.6 + n_stealth * 0.4)
        trap_score = self._apply_nonlinear_gain(high_lock, center=0.3, steepness=3.0)
        s_scale = skew.abs().rolling(60, min_periods=1).max().replace(0, 1e-9)
        kill_score = self._apply_nonlinear_gain(-skew / s_scale, center=0.0, steepness=3.0)
        is_drop = (pct_chg < 0).astype(int)
        return ((lure_score * 0.4 + trap_score * 0.3 + kill_score * 0.3) * is_drop).fillna(0.0)
    def _assess_fractal_manipulation_fingerprint(self, abnormal: pd.Series, clustering: pd.Series, anomaly: pd.Series, acc_abnormal: pd.Series) -> pd.Series:
        n_abnormal = self._apply_nonlinear_gain(abnormal, center=0.5, steepness=2.0)
        n_clustering = self._apply_nonlinear_gain(clustering, center=0.3, steepness=3.0)
        n_anomaly = self._apply_nonlinear_gain(anomaly, center=0.0, steepness=2.0)
        a_scale = acc_abnormal.rolling(60, min_periods=1).std().replace(0, 1e-9)
        n_acc = np.tanh(acc_abnormal / a_scale)
        boost = self._apply_nonlinear_gain(pd.Series(n_acc), center=0.0, steepness=2.0) + 0.8
        base_manipulation = (n_abnormal * 0.4 + n_clustering * 0.4 + n_anomaly * 0.2)
        return self._apply_nonlinear_gain(base_manipulation * boost, center=0.5, steepness=3.0).fillna(0.0)
    def _assess_market_leadership_compensation(self, is_leader: pd.Series, state_leader: pd.Series, theme_hot: pd.Series) -> pd.Series:
        leader_identity = ((is_leader == 1) | (state_leader == 1)).astype(float)
        hot_booster = self._apply_nonlinear_gain(theme_hot, center=60.0, steepness=0.1)
        compensation = 1.0 + (leader_identity * (0.2 + hot_booster * 0.1))
        return compensation.fillna(1.0).astype(np.float32)
    def _assess_turnover_lock_stability(self, turnover_stable: pd.Series, is_leader: pd.Series, chip_stab: pd.Series) -> pd.Series:
        t_stable_norm = self._apply_nonlinear_gain(turnover_stable, center=50.0, steepness=0.05)
        leader_gate = np.where(is_leader == 1, 1.3, 1.0)
        c_stable_norm = self._apply_nonlinear_gain(chip_stab, center=0.0, steepness=2.0)
        lock_score = (t_stable_norm * 0.7 + c_stable_norm * 0.3) * leader_gate
        return self._apply_nonlinear_gain(pd.Series(lock_score), center=0.5, steepness=3.0).astype(np.float32)

