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
    【V6.0.0 · 时空博弈全息版】上冲回落洗盘甄别器
    - 核心职责: 在“物理-博弈-结构”三元基础上，叠加“时序欺骗”与“筹码代谢”验证，识别高精度的洗盘信号。
    - 维度升级:
        1. **时域欺骗 (Chronological Deception)**: 利用 `morning_flow_ratio_D` 识别“早盘诱多、尾盘杀跌”的典型洗盘时序特征。
        2. **筹码代谢 (Chip Metabolism)**: 引入 `pressure_release_index_D` 验证洗盘是否有效清洗了浮筹（有效洗盘必须伴随压力释放）。
        3. **异动操控 (Anomaly Manipulation)**: 使用 `tick_abnormal_volume_ratio_D` 确认下跌是由异常大单主导的“人造行情”。
    - 继承逻辑: 保留V5.0.0的物理陷阱(Trap)、逆流背离(Divergence)和结构韧性(Resilience)。
    - 版本: 6.0.0
    """
    def __init__(self, strategy_instance, helper_instance: ProcessIntelligenceHelper):
        self.strategy = strategy_instance
        self.helper = helper_instance
        self.params = self.helper.params
        self.debug_params = self.helper.debug_params
        self.probe_dates = self.helper.probe_dates

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        [V20.0.0 · 全链路溯源总控]
        - 升级: 将所有原始信号(Raw Signals)透传至探针，实现从原料到结果的完整可视。
        - 架构: Trap(4D) -> Resonance(Stratified) -> Meta(Anchor) -> Solidity(Stress) -> Deception(Skew) -> Anomaly(Fractal)
        """
        method_name = "CalculateUpthrustWashoutRelationship.calculate"
        # 1. 全量数据提取
        (open_p, high_p, low_p, close_p, pct_chg, 
         slope_p, accel_p, jerk_p,                                   
         raw_f, jerk_f, smart_n, smart_d, slope_f_21, accel_f_21, f_accum_34, f_volat, 
         vpa_eff, och_acc, bbp_pos,                                  
         test_cnt, cost_mig, sr_ratio, acc_supp, 
         morning, stealth, high_lock, skew,
         pres_rel, hab_rel, winner, acc_win, turnover, chip_stab, cost_diff, 
         abnormal_vol, clustering, order_anomaly, acc_abnormal,
         ma_res, slope_t) = self._get_raw_signals(df, method_name)
        df_index = df.index
        # 2. 物理层 (4D Trap)
        k_trap = self._assess_kinematic_trap_physics(close_p, high_p, slope_p, jerk_p, vpa_eff, och_acc, bbp_pos)
        # 3. 资金层 (Stratified Resonance & HAB)
        f_resonance = self._assess_fund_jerk_resonance(pct_chg, jerk_f, smart_n, smart_d)
        f_hab = self._assess_fund_reservoir_buffer(raw_f, f_accum_34, slope_f_21, accel_f_21, vpa_eff, f_volat)
        fund_score = f_resonance * f_hab
        # 4. 筹码层 (Dynamic Metabolism)
        chip_meta = self._assess_dynamic_chip_metabolism(hab_rel, turnover, winner, acc_win, chip_stab, cost_diff)
        # 5. 防御层 (Structural Stress Test)
        solidity = self._assess_structural_stress_test(sr_ratio, test_cnt, cost_mig, acc_supp)
        # 6. 取证与融合
        chrono_dec = self._assess_skewed_deception_narrative(morning, stealth, high_lock, skew, pct_chg)
        fractal_man = self._assess_fractal_manipulation_fingerprint(abnormal_vol, clustering, order_anomaly, acc_abnormal)
        context = ((slope_t > 0) | (ma_res > 0.6)).astype(int)
        forensics = (chrono_dec * 0.4 + chip_meta * 0.3 + fractal_man * 0.3)
        final_score = (k_trap * fund_score * forensics * solidity * context).clip(0, 1)
        # 7. 全链路探针调用 (传入所有 Raw 数据)
        self._print_debug_probe(df_index, final_score,
                                # 物理层
                                k_trap, slope_p, jerk_p, vpa_eff, och_acc, bbp_pos,
                                # 资金层
                                f_resonance, f_hab, raw_f, jerk_f, smart_n, smart_d, f_accum_34, slope_f_21, accel_f_21, f_volat,
                                # 筹码层
                                chip_meta, hab_rel, turnover, winner, acc_win, chip_stab, cost_diff,
                                # 防御层
                                solidity, sr_ratio, test_cnt, cost_mig, acc_supp,
                                # 取证层
                                chrono_dec, fractal_man, morning, stealth, high_lock, skew, pct_chg,
                                abnormal_vol, clustering, order_anomaly, acc_abnormal)
        return final_score.astype(np.float32).fillna(0.0)

    def _get_raw_signals(self, df: pd.DataFrame, method_name: str) -> Tuple[pd.Series, ...]:
        """
        [V20.1.0 · 纯净原料获取]
        - 核心修复: 将默认填充值从 0.0 改为 np.nan。
        - 目的: 配合 rolling(min_periods=1)，实现"不填充0，NaN不参与计算"的统计逻辑。
        """
        # 基础行情
        open_p = self.helper._get_safe_series(df, 'open_D', np.nan, method_name=method_name)
        high_p = self.helper._get_safe_series(df, 'high_D', np.nan, method_name=method_name)
        low_p = self.helper._get_safe_series(df, 'low_D', np.nan, method_name=method_name)
        close_p = self.helper._get_safe_series(df, 'close_D', np.nan, method_name=method_name)
        pct_chg = self.helper._get_safe_series(df, 'pct_change_D', np.nan, method_name=method_name)
        # 价格运动学
        slope_price = self.helper._get_safe_series(df, 'SLOPE_3_close_D', np.nan, method_name=method_name)
        accel_price = self.helper._get_safe_series(df, 'ACCEL_5_close_D', np.nan, method_name=method_name)
        jerk_price = self.helper._get_safe_series(df, 'JERK_3_close_D', np.nan, method_name=method_name)
        # 资金分层运动学
        raw_fund = self.helper._get_safe_series(df, 'tick_large_order_net_D', np.nan, method_name=method_name)
        jerk_fund = self.helper._get_safe_series(df, 'JERK_3_tick_large_order_net_D', np.nan, method_name=method_name)
        smart_net = self.helper._get_safe_series(df, 'SMART_MONEY_HM_NET_BUY_D', np.nan, method_name=method_name)
        smart_div = self.helper._get_safe_series(df, 'SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D', np.nan, method_name=method_name)
        slope_fund_21 = self.helper._get_safe_series(df, 'SLOPE_21_tick_large_order_net_D', np.nan, method_name=method_name)
        accel_fund_21 = self.helper._get_safe_series(df, 'ACCEL_21_tick_large_order_net_D', np.nan, method_name=method_name)
        # HAB累积: NaN 不参与 sum，min_periods=1 保证只要有一天有数据就有结果
        fund_accum_34 = raw_fund.rolling(window=34, min_periods=1).sum() 
        fund_volat = self.helper._get_safe_series(df, 'flow_volatility_20d_D', np.nan, method_name=method_name)
        # 陷阱结构与环境
        vpa_eff = self.helper._get_safe_series(df, 'VPA_MF_ADJUSTED_EFF_D', np.nan, method_name=method_name)
        och_acc = self.helper._get_safe_series(df, 'OCH_ACCELERATION_D', np.nan, method_name=method_name)
        bbp_pos = self.helper._get_safe_series(df, 'BBP_21_2.0_D', np.nan, method_name=method_name)
        # 结构压力测试
        test_cnt = self.helper._get_safe_series(df, 'intraday_support_test_count_D', np.nan, method_name=method_name)
        cost_mig = self.helper._get_safe_series(df, 'intraday_cost_center_migration_D', np.nan, method_name=method_name)
        sr_ratio = self.helper._get_safe_series(df, 'support_resistance_ratio_D', np.nan, method_name=method_name)
        acc_supp = self.helper._get_safe_series(df, 'ACCEL_5_support_strength_D', np.nan, method_name=method_name)
        # 时空欺骗
        morning = self.helper._get_safe_series(df, 'morning_flow_ratio_D', np.nan, method_name=method_name)
        stealth = self.helper._get_safe_series(df, 'stealth_flow_ratio_D', np.nan, method_name=method_name)
        high_lock = self.helper._get_safe_series(df, 'intraday_high_lock_ratio_D', np.nan, method_name=method_name)
        skew = self.helper._get_safe_series(df, 'intraday_price_distribution_skewness_D', np.nan, method_name=method_name)
        # 筹码代谢
        pres_rel = self.helper._get_safe_series(df, 'pressure_release_index_D', np.nan, method_name=method_name)
        hab_rel = pres_rel.rolling(window=5, min_periods=1).sum()
        winner = self.helper._get_safe_series(df, 'winner_rate_D', np.nan, method_name=method_name)
        acc_win = self.helper._get_safe_series(df, 'ACCEL_5_winner_rate_D', np.nan, method_name=method_name)
        turnover = self.helper._get_safe_series(df, 'turnover_rate_D', np.nan, method_name=method_name)
        chip_stab = self.helper._get_safe_series(df, 'chip_stability_D', np.nan, method_name=method_name)
        cost_diff = self.helper._get_safe_series(df, 'chip_cost_to_ma21_diff_D', np.nan, method_name=method_name)
        # 分形操控
        abnormal_vol = self.helper._get_safe_series(df, 'tick_abnormal_volume_ratio_D', np.nan, method_name=method_name)
        clustering = self.helper._get_safe_series(df, 'tick_clustering_index_D', np.nan, method_name=method_name)
        order_anomaly = self.helper._get_safe_series(df, 'large_order_anomaly_D', np.nan, method_name=method_name)
        acc_abnormal = self.helper._get_safe_series(df, 'ACCEL_5_tick_abnormal_volume_ratio_D', np.nan, method_name=method_name)
        # 趋势辅助
        ma_res = self.helper._get_safe_series(df, 'MA_COHERENCE_RESONANCE_D', np.nan, method_name=method_name)
        slope_trend = self.helper._get_safe_series(df, 'GEOM_REG_SLOPE_D', np.nan, method_name=method_name)
        return (open_p, high_p, low_p, close_p, pct_chg, slope_price, accel_price, jerk_price, 
                raw_fund, jerk_fund, smart_net, smart_div, slope_fund_21, accel_fund_21, fund_accum_34, fund_volat, 
                vpa_eff, och_acc, bbp_pos, 
                test_cnt, cost_mig, sr_ratio, acc_supp, 
                morning, stealth, high_lock, skew,
                pres_rel, hab_rel, winner, acc_win, turnover, chip_stab, cost_diff, 
                abnormal_vol, clustering, order_anomaly, acc_abnormal,
                ma_res, slope_trend)

    def _assess_fund_reservoir_buffer(self, daily: pd.Series, accum_34: pd.Series, slope_f: pd.Series, accel_f: pd.Series, vpa_eff: pd.Series, volat: pd.Series) -> pd.Series:
        accum_median = accum_34.rolling(120, min_periods=1).median().replace(0, 1e-9)
        reservoir_strength = np.tanh(accum_34 / accum_median.abs()).clip(0, 1)
        
        s_scale = slope_f.rolling(60, min_periods=1).std().replace(0, 1e-9)
        norm_s = np.tanh(slope_f / (s_scale * 1.5))
        a_scale = accel_f.rolling(60, min_periods=1).std().replace(0, 1e-9)
        norm_a = np.tanh(accel_f / (a_scale * 1.5))
        trend_score = (norm_s * 0.6 + norm_a * 0.4 + 1.0) / 2.0
        
        quality_f = (vpa_eff.rolling(5, min_periods=1).mean().clip(0, 1) + 0.5).clip(0.5, 1.5)
        dynamic_threshold = (volat * 2.0).clip(0.05, 0.25)
        attrition = daily.abs() / (accum_34.abs() + 1e-9)
        tolerance = (1.0 - (attrition / dynamic_threshold)).clip(0, 1)
        
        return (reservoir_strength * trend_score * quality_f * tolerance).clip(0, 1).fillna(0)

    def _assess_dynamic_chip_metabolism(self, hab_release: pd.Series, turnover: pd.Series, winner: pd.Series, acc_win: pd.Series, stability: pd.Series, cost_diff: pd.Series) -> pd.Series:
        eff_ratio = hab_release / (turnover + 0.5)
        eff_scale = eff_ratio.rolling(60, min_periods=1).max().replace(0, 1e-9)
        efficiency_score = (eff_ratio / eff_scale).clip(0, 1)

        norm_winner = winner.rolling(20, min_periods=1).rank(pct=True).clip(0, 1)
        norm_stab = stability.rolling(20, min_periods=1).rank(pct=True).clip(0, 1)
        integrity_score = (norm_winner * 0.4 + norm_stab * 0.6).clip(0, 1)

        anchor_score = np.where(cost_diff < -0.05, 0.2, 1.0)

        w_scale = acc_win.rolling(60, min_periods=1).std().replace(0, 1e-9)
        n_acc_win = np.tanh(acc_win / w_scale)
        breaker = (1.0 + n_acc_win).clip(0, 1)

        return ((efficiency_score * 0.4 + integrity_score * 0.6) * anchor_score * breaker).fillna(0)

    def _assess_kinematic_trap_physics(self, close: pd.Series, high: pd.Series, slope: pd.Series, jerk: pd.Series, vpa_eff: pd.Series, och_acc: pd.Series, bbp: pd.Series) -> pd.Series:
        # 使用 min_periods=1 确保即使只有1个数据也能计算 std/max
        s_scale = slope.rolling(60, min_periods=1).std().replace(0, 1e-9)
        n_slope = np.tanh(slope / s_scale).clip(0, 1)
        j_scale = jerk.rolling(60, min_periods=1).std().replace(0, 1e-9)
        n_jerk = np.tanh(jerk / j_scale)
        exhaustion = (-n_jerk).clip(0, 1)
        
        vpa_scale = vpa_eff.abs().rolling(60, min_periods=1).max().replace(0, 1e-9)
        inefficiency = (1.0 - np.tanh(vpa_eff / vpa_scale)).clip(0, 1)
        
        och_scale = och_acc.abs().rolling(60, min_periods=1).max().replace(0, 1e-9)
        structure_fail = (-np.tanh(och_acc / och_scale)).clip(0, 1)
        
        context_gate = np.tanh((bbp - 0.5) * 3).clip(0, 1)
        
        range_hl = (high - close) / high.replace(0, 1e-9)
        retracement = (range_hl * 10).clip(0, 1)
        
        base_trap = (n_slope * 0.4 + exhaustion * 0.6)
        quality = (inefficiency * 0.3 + structure_fail * 0.3 + retracement * 0.4)
        
        return (base_trap * quality * context_gate).clip(0, 1).fillna(0)

    def _assess_fund_jerk_resonance(self, pct_chg: pd.Series, jerk_large: pd.Series, smart_net: pd.Series, smart_div: pd.Series) -> pd.Series:
        j_scale = jerk_large.rolling(60, min_periods=1).std().replace(0, 1e-9)
        panic_impulse = (-np.tanh(jerk_large / j_scale)).clip(0, 1)
        
        s_scale = smart_net.rolling(60, min_periods=1).std().replace(0, 1e-9)
        absorption = np.tanh(smart_net / s_scale).clip(0, 1)
        
        # 弹性计算: 注意 pct_chg 可能为 NaN，fillna(0) 防止除以 NaN 导致传播
        p_scale = pct_chg.abs().rolling(60, min_periods=1).mean().replace(0, 1e-9) + 1e-5
        elasticity = (panic_impulse / (pct_chg.abs() / p_scale)).clip(0, 2) / 2.0
        
        div_bonus = (np.tanh(smart_div) + 1.0) / 2.0
        resonance_score = (panic_impulse * 0.4 + absorption * 0.3 + elasticity * 0.2 + div_bonus * 0.1).clip(0, 1)
        validity_gate = np.where(smart_net > 0, 1.0, 0.5) 
        
        return (resonance_score * validity_gate).fillna(0)

    def _assess_structural_stress_test(self, ratio_sr: pd.Series, test_count: pd.Series, cost_mig: pd.Series, acc_supp: pd.Series) -> pd.Series:
        base_solidity = np.tanh(ratio_sr - 0.8).clip(0, 1)
        test_bonus = np.log1p(test_count).clip(0, 2) / 2.0
        resilience = base_solidity * (0.5 + 0.5 * test_bonus)
        
        c_scale = cost_mig.abs().rolling(60, min_periods=1).mean().replace(0, 1e-9)
        n_mig = cost_mig / c_scale
        gravity_stable = (np.tanh(n_mig + 0.5) + 1.0) / 2.0
        
        a_scale = acc_supp.rolling(60, min_periods=1).std().replace(0, 1e-9)
        n_acc = np.tanh(acc_supp / a_scale)
        boost = (1.0 + n_acc * 0.5).clip(0.5, 1.5)
        
        return (resilience * gravity_stable * boost).clip(0, 1).fillna(0)

    def _assess_skewed_deception_narrative(self, morning: pd.Series, stealth: pd.Series, high_lock: pd.Series, skew: pd.Series, pct_chg: pd.Series) -> pd.Series:
        n_morning = np.tanh((morning - 0.4) * 3).clip(0, 1)
        n_stealth = np.tanh(stealth * 2).clip(0, 1)
        lure_score = (n_morning * 0.6 + n_stealth * 0.4)
        
        trap_score = np.tanh(high_lock * 3).clip(0, 1)
        
        s_scale = skew.abs().rolling(60, min_periods=1).max().replace(0, 1e-9)
        kill_score = (-np.tanh(skew / s_scale)).clip(0, 1)
        
        is_drop = (pct_chg < 0).astype(int)
        
        return ((lure_score * 0.4 + trap_score * 0.3 + kill_score * 0.3) * is_drop).fillna(0)

    def _assess_fractal_manipulation_fingerprint(self, abnormal: pd.Series, clustering: pd.Series, anomaly: pd.Series, acc_abnormal: pd.Series) -> pd.Series:
        n_abnormal = np.tanh(abnormal * 2).clip(0, 1)
        n_clustering = np.tanh(clustering * 3).clip(0, 1)
        n_anomaly = np.tanh(anomaly).clip(0, 1)
        
        a_scale = acc_abnormal.rolling(60, min_periods=1).std().replace(0, 1e-9)
        n_acc = np.tanh(acc_abnormal / a_scale)
        boost = (1.0 + n_acc * 0.5).clip(0.8, 1.5)
        
        base_manipulation = (n_abnormal * 0.4 + n_clustering * 0.4 + n_anomaly * 0.2)
        return (base_manipulation * boost).clip(0, 1).fillna(0)

    def _print_debug_probe(self, idx: pd.Index, final: pd.Series,
                           # 物理参数
                           k_trap: pd.Series, slope_p: pd.Series, jerk_p: pd.Series, vpa: pd.Series, och: pd.Series, bbp: pd.Series,
                           # 资金参数
                           f_res: pd.Series, f_hab: pd.Series, raw_f: pd.Series, jerk_f: pd.Series, smart_n: pd.Series, smart_d: pd.Series, 
                           accum_34: pd.Series, slope_f_21: pd.Series, accel_f_21: pd.Series, f_volat: pd.Series,
                           # 筹码参数
                           c_meta: pd.Series, hab_rel: pd.Series, turnover: pd.Series, winner: pd.Series, acc_win: pd.Series, stab: pd.Series, cost_diff: pd.Series,
                           # 防御参数
                           solid: pd.Series, sr_ratio: pd.Series, test_cnt: pd.Series, cost_mig: pd.Series, acc_supp: pd.Series,
                           # 取证参数
                           dec: pd.Series, man: pd.Series, morning: pd.Series, stealth: pd.Series, lock: pd.Series, skew: pd.Series, pct: pd.Series,
                           abnormal: pd.Series, cluster: pd.Series, order_anom: pd.Series, acc_abn: pd.Series):
        """
        [V20.0.0 · 全链路溯源探针]
        - 职责: 提供 [结果] <- [逻辑节点] <- [原始数据] 的三层溯源视图。
        - 目的: 彻底解决数据黑盒，快速定位 nan 或 0 分的根源。
        """
        if len(idx) > 0:
            i = -1
            d_str = idx[i].strftime('%Y-%m-%d')
            print(f"--- [PROBE_V20_FULL_LINK] {d_str} FINAL: {final.iloc[i]:.4f} ---")
            # 1. 物理层
            print(f"  [1.Physics] Node: {k_trap.iloc[i]:.4f}")
            print(f"     > Logic: Kinematics(Slope/Jerk) * VPA * OCH * BBP")
            print(f"     > Raw  : Slope={slope_p.iloc[i]:.4f}, Jerk={jerk_p.iloc[i]:.4f}, VPA={vpa.iloc[i]:.2f}, OCH={och.iloc[i]:.4f}, BBP={bbp.iloc[i]:.2f}")
            # 2. 资金层
            f_total = (f_res * f_hab).iloc[i]
            print(f"  [2.Funds]   Node: {f_total:.4f} (Reson: {f_res.iloc[i]:.4f} * HAB: {f_hab.iloc[i]:.4f})")
            print(f"     > Logic: (PanicJerk vs SmartNet) & (Accum34 & Vector21)")
            print(f"     > Raw(R): PanicJerk={jerk_f.iloc[i]:.2f}, SmartNet={smart_n.iloc[i]:.2f}, SmartDiv={smart_d.iloc[i]:.2f}, DailyF={raw_f.iloc[i]:.0f}")
            print(f"     > Raw(H): Accum34={accum_34.iloc[i]:.0f}, Slope21={slope_f_21.iloc[i]:.2f}, Accel21={accel_f_21.iloc[i]:.2f}, Volat={f_volat.iloc[i]:.4f}")
            # 3. 筹码层
            print(f"  [3.Chips]   Node: {c_meta.iloc[i]:.4f}")
            print(f"     > Logic: Efficiency(HAB/TO) * Integrity(Win/Stab) * Anchor(Cost) * Accel")
            print(f"     > Raw  : HAB_Rel={hab_rel.iloc[i]:.2f}, Turnover={turnover.iloc[i]:.2f}%, Winner={winner.iloc[i]:.2f}, AccWin={acc_win.iloc[i]:.4f}")
            print(f"     > Raw  : Stab={stab.iloc[i]:.2f}, CostDiff={cost_diff.iloc[i]:.4f}")
            # 4. 防御层
            print(f"  [4.Defense] Node: {solid.iloc[i]:.4f}")
            print(f"     > Logic: SR_Ratio * TestCount * CostMig * AccelSupp")
            print(f"     > Raw  : SR_Ratio={sr_ratio.iloc[i]:.2f}, TestCnt={test_cnt.iloc[i]:.1f}, CostMig={cost_mig.iloc[i]:.4f}, AccSupp={acc_supp.iloc[i]:.4f}")
            # 5. 取证层
            print(f"  [5.Forensics] Decept: {dec.iloc[i]:.4f} | Manip: {man.iloc[i]:.4f}")
            print(f"     > Raw(D): Morning={morning.iloc[i]:.2f}, Stealth={stealth.iloc[i]:.2f}, Lock={lock.iloc[i]:.2f}, Skew={skew.iloc[i]:.4f}, Pct={pct.iloc[i]:.2f}%")
            print(f"     > Raw(M): Abnormal={abnormal.iloc[i]:.2f}, Cluster={cluster.iloc[i]:.4f}, OrderAnom={order_anom.iloc[i]:.4f}, AccAbn={acc_abn.iloc[i]:.4f}")






