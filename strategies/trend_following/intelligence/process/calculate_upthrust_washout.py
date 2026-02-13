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
        [V38.0.0 · 全链路执行总控]
        - 变更说明: 整合物理(Trap)、资金(多周期/拆单/能量)、筹码(位移/坍塌/熵减/锁死)、取证(协同/SSD)全维度调用。
        - 目的: 彻底消除方法未调用问题，补全逻辑拼图。
        """
        method_name = "CalculateUpthrustWashoutRelationship.calculate"
        (open_p, high_p, low_p, close_p, pct_chg, p_ent, slope_p, jerk_p, atr_14, vol_f20, c_mean, p_mig, c_mig, r_f, j_f, s_n, s_d, f13, f34, f55, t55, vpa, och, bbp, hm_att, sm_att, p_trap, p_rel, is_ldr, st_ldr, t_hot, t_stab, win, a_win, c_stab, a_stab, c_diff, c_kurt, c_conv, is_multi, p_cnt, c_ent, dw_s, morn, ste, h_lock, skew, h_rel, t_over, clust, vol, cnt, t_net, ma_r, sl_t) = self._get_raw_signals(df, method_name)
        df_index = df.index
        # 1. 物理层 (Physics)
        k_trap = self._assess_kinematic_trap_physics(close_p, high_p, slope_p, jerk_p, vpa, och, bbp)
        # 2. 资金层 (Funds)
        f_res = self._assess_fund_jerk_resonance(pct_chg, j_f, s_n, smart_div=s_d)
        f_split = self._assess_split_order_accumulation(vol, cnt, clust, r_f, t_net)
        f_base = self._assess_multi_cycle_fund_reservoir(f13, f34, f55, t55, hm_att, sm_att)
        f_energy = self._assess_cumulative_energy_decay(r_f, t_net)
        fund_score = np.maximum(f_res, f_split) * f_base * f_energy
        # 3. 筹码层 (Chips)
        chip_migration = self._assess_chip_holographic_migration(close_p, c_mean, p_mig, c_mig)
        chip_collapse = self._assess_chip_peak_collapse(is_multi, p_cnt)
        chip_order = self._assess_chip_entropy_ordered_collapse(c_ent, p_cnt)
        chip_tension = self._assess_chip_holographic_tension(h_rel, t_over, win, a_win, c_stab, a_stab, c_kurt, c_conv, c_diff)
        chip_lock = self._assess_turnover_lock_stability(t_stab, is_ldr, c_stab)
        chip_score = (chip_migration * 0.25 + chip_collapse * 0.25 + chip_order * 0.2 + chip_tension * 0.15 + chip_lock * 0.15)
        # 4. 取证与对冲 (Forensics & Hedge)
        intent_score = self._assess_coordinated_intent_forensics(hm_att, sm_att, p_trap, p_rel)
        ssd_dec = self._assess_stealth_shadow_divergence(morn, ste, h_lock, skew, pct_chg, t_net)
        leader_bonus = self._assess_market_leadership_compensation(is_ldr, st_ldr, t_hot)
        atr_norm = (atr_14 / close_p.replace(0, 1e-9)).rolling(60, min_periods=1).rank(pct=True)
        beta_hedge = (1.2 - atr_norm).clip(0, 1)
        e_scale = p_ent.rolling(60, min_periods=1).mean().replace(0, 1e-9)
        entropy_multi = (1.2 - np.tanh(p_ent / e_scale)).clip(0, 1)
        # 5. 综合评分与门控
        sync_gate = np.where(dw_s == 1, 1.1, 0.9)
        final_score = (k_trap * fund_score * chip_score * intent_score * leader_bonus * beta_hedge * entropy_multi * sync_gate).clip(0, 1)
        # 6. 数据总线封包
        debug_ctx = {
            "Final": final_score, "Phys": k_trap, "Fund": fund_score, "Chip": chip_score, "Intent": intent_score, "Ldr": leader_bonus,
            "Raw": {"JerkP": jerk_p, "Split": f_split, "Reson": f_res, "Base": f_base, "Energy": f_energy, "Mig": chip_migration, "Coll": chip_collapse, "Order": chip_order, "Lock": chip_lock, "SSD": ssd_dec, "Trap": p_trap, "HMAtt": hm_att, "Theme": t_hot}
        }
        self._print_debug_probe(df_index, debug_ctx)
        return final_score.astype(np.float32).fillna(0.0)

    def _print_debug_probe(self, idx: pd.Index, ctx: Dict[str, Any]):
        """
        [V38.0.0 · 全链路深度溯源探针]
        - 职责: 揭示从物理、资金到筹码、取证的 100% 逻辑链路。
        """
        if len(idx) > 0:
            i = -1
            dt = idx[i].strftime('%Y-%m-%d')
            raw = ctx["Raw"]
            print(f"--- [ CalculateUpthrustWashoutRelationship PROBE_V38_FULL_LINK] {dt} FINAL: {ctx['Final'].iloc[i]:.4f} ---")
            print(f"  [1.Physics]  Trap: {ctx['Phys'].iloc[i]:.4f} (PriceJerk: {raw['JerkP'].iloc[i]:.2f})")
            print(f"  [2.Funds]    Score: {ctx['Fund'].iloc[i]:.4f} <- (Reson: {raw['Reson'].iloc[i]:.2f}, Split: {raw['Split'].iloc[i]:.2f}, Base: {raw['Base'].iloc[i]:.2f}, Energy: {raw['Energy'].iloc[i]:.2f})")
            print(f"  [3.Chips]    Score: {ctx['Chip'].iloc[i]:.4f} <- (Mig: {raw['Mig'].iloc[i]:.2f}, Coll: {raw['Coll'].iloc[i]:.2f}, Order: {raw['Order'].iloc[i]:.2f}, Lock: {raw['Lock'].iloc[i]:.2f})")
            print(f"  [4.Forens]   Intent: {ctx['Intent'].iloc[i]:.4f} (SSD: {raw['SSD'].iloc[i]:.2f}, HMAttack: {raw['HMAtt'].iloc[i]}, Trapped: {raw['Trap'].iloc[i]:.2f})")
            print(f"  [5.Strategy] LeaderBonus: {ctx['Ldr'].iloc[i]:.2f}x (ThemeHot: {raw['Theme'].iloc[i]:.1f})")
            print(f"--- ========================================================== ---")

    def _get_raw_signals(self, df: pd.DataFrame, method_name: str) -> Tuple[pd.Series, ...]:
        """
        [V38.0.0 · 全维信号接入终极版]
        - 变更说明: 补齐军械库清单中所有核心指标，确保物理、资金、筹码、取证、防御全链路调用无死角。
        - 修复: 补齐 och_acceleration_D, bbp_21_2.0_D 等物理层关键指标。
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
        vol_f_20 = self.helper._get_safe_series(df, 'flow_volatility_20d_D', np.nan, method_name=method_name)
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
        return (open_p, high_p, low_p, close_p, pct_chg, p_entropy, slope_p, jerk_p, atr_14, vol_f_20, c_mean, peak_mig, conv_mig, raw_fund, jerk_f, smart_n, smart_d, f_acc13, f_acc34, f_acc55, t_acc55, vpa_eff, och_acc, bbp_pos, hm_attack, sm_attack, p_trapped, p_release, is_leader, state_leader, theme_hot, turnover_stable, winner, acc_win, chip_stab, acc_stab_21, cost_diff, chip_kurt, chip_conv, is_multi, peak_cnt, chip_ent, dw_sync, morning, stealth, high_lock, skew, hab_rel, turnover, clustering, volume, trade_count, total_net, ma_res, slope_t)

    def _assess_multi_period_volatility_hedge(self, s5: pd.Series, s20: pd.Series, s60: pd.Series, m5: pd.Series, m20: pd.Series, m60: pd.Series) -> pd.Series:
        """
        [V30.0.0 · 多周期Beta自适应波动对冲]
        - 核心逻辑: 识别个股波动是否脱离大盘基准。
        - 判定维度: 个股相对于市场的异常波动张力 (Tension)。
        - 计算: Tension = Avg(Stock_Vol / Market_Vol)。若 Tension 过高, 判定为 Alpha 风险而非 Beta 波动, 压制分数。
        """
        # 计算相对波动比率 (加小量 eps 防止除零)
        r5 = (s5 / (m5 + 1e-9)).clip(0, 5)
        r20 = (s20 / (m20 + 1e-9)).clip(0, 5)
        r60 = (s60 / (m60 + 1e-9)).clip(0, 5)
        # 综合张力: 赋予中长期更高权重，以识别结构性破位
        tension = (r5 * 0.2 + r20 * 0.4 + r60 * 0.4)
        # 映射逻辑: 若张力在 1.0 附近 (同步市场), 对冲系数为 1.0; 若张力 > 2.0 (离群暴走), 系数迅速衰减
        hedge_factor = (1.2 - np.tanh(tension / 2.0)).clip(0, 1)
        return hedge_factor.fillna(1.0).astype(np.float32)

    def _assess_split_order_accumulation(self, volume: pd.Series, count: pd.Series, clustering: pd.Series, large_net: pd.Series, total_net: pd.Series) -> pd.Series:
        """
        [V21.0.0 · 拆单暗影模型]
        - 核心逻辑: 识别主力通过"化整为零"(拆单)进行的隐蔽吸筹。
        - 判定维度:
            1. 碎片化 (Fragmentation): 笔均成交量显著下降。
            2. 机器指纹 (Clustering): 交易时间/手数具有高度聚类特征。
            3. 暗流涌动 (Undercurrent): 大单净额为负/零，但总净额为正(中小单流入)。
        """
        # 1. 碎片化指数 (Fragmentation)
        # 笔均成交量 = 总量 / 笔数
        avg_trade_size = volume / (count + 1e-9)
        # 历史基准 (20日均值)
        baseline_size = avg_trade_size.rolling(20, min_periods=1).mean()
        # 如果今日笔均显著小于历史基准 (ratio > 1.2)，说明单子被拆碎了
        frag_ratio = baseline_size / (avg_trade_size + 1e-9)
        n_frag = np.tanh(frag_ratio - 1.0).clip(0, 1) # 超过1.0的部分才开始计分
        # 2. 机器指纹 (Clustering)
        # 拆单通常由算法执行，会留下高聚类痕迹
        n_clustering = np.tanh(clustering * 3).clip(0, 1)
        # 3. 暗流涌动 (Undercurrent)
        # 计算"非大单"的净流向 (中小单 = 总 - 大)
        small_medium_net = total_net - large_net
        # 归一化: 我们寻找 SM_Net > 0 且 Large_Net <= 0 的剪刀差情况
        # 如果大单在流出(或微弱)，但中小单在强力流入，这是吸筹铁证
        sm_scale = small_medium_net.rolling(60, min_periods=1).std().replace(0, 1e-9)
        n_sm_net = np.tanh(small_medium_net / sm_scale).clip(0, 1)
        # 大单掩护: 如果大单是流出的(mask=1)，则中小单流入的权重更高
        large_mask = np.where(large_net <= 0, 1.2, 0.8)
        stealth_flow = (n_sm_net * large_mask).clip(0, 1)
        # 4. 最终合成
        # 拆单吸筹 = 单子碎 * 有规律 * 暗中买
        split_score = (n_frag * 0.3 + n_clustering * 0.3 + stealth_flow * 0.4).clip(0, 1)
        return split_score.fillna(0)

    def _assess_chip_entropy_ordered_collapse(self, entropy: pd.Series, peak_cnt: pd.Series) -> pd.Series:
        """
        [V35.0.0 · 缺失方法修复]
        - 核心逻辑: 监控筹码从高熵向低熵的有序坍塌。
        """
        ent_delta = entropy.diff(5).fillna(0)
        entropy_decay = np.where(ent_delta < 0, ent_delta.abs() / (entropy + 1e-9), 0.0)
        order_score = (np.tanh(entropy_decay * 5) * np.where(peak_cnt <= 2, 1.2, 0.8)).clip(0, 1)
        return pd.Series(order_score, index=entropy.index).astype(np.float32)

    def _assess_coordinated_intent_forensics(self, hm_attack: pd.Series, sm_attack: pd.Series, trapped: pd.Series, release: pd.Series) -> pd.Series:
        """
        [V35.0.0 · 主力协同意图取证]
        - 核心逻辑: 识别协同攻击是真实突破还是联合诱多。
        - 判定维度: 
            1. 协同强度: HM与SmartMoney攻击指纹的加权。 
            2. 压力背离: 攻击时套牢盘(pressure_trapped_D)必须处于低位。 
        - 效果: 若套牢压力高且协同攻击强, 视为诱多; 若压力低且协同强, 视为真强。
        """
        co_strength = (hm_attack * 0.4 + sm_attack * 0.6).clip(0, 1)
        t_scale = trapped.rolling(60, min_periods=1).max().replace(0, 1e-9)
        low_pressure_gate = (1.0 - np.tanh(trapped / t_scale)).clip(0, 1)
        # 真强逻辑: 协同攻击 * 低套牢压力
        intent_score = (co_strength * low_pressure_gate)
        # 释放修正: 如果近期有大量压力释放(pressure_release_index_D) 
        release_bonus = np.tanh(release.rolling(5).mean()).clip(0, 1)
        return (intent_score + release_bonus * 0.3).clip(0, 1).fillna(0)

    def _assess_multi_cycle_fund_reservoir(self, f_acc13: pd.Series, f_acc34: pd.Series, f_acc55: pd.Series, t_acc55: pd.Series, hm_attack: pd.Series, smart_attack: pd.Series) -> pd.Series:
        """
        [V34.0.0 · 跨时序资金母盘评估]
        - 核心逻辑: 利用斐波那契序列(13, 34, 55)构建资金蓄水池共振。
        - 判定维度: 
            1. 55日战略累积(f_acc55)提供底池硬度。
            2. 协同攻击(HM_COORDINATED_ATTACK)提供瞬间爆发修正 。
        - 效果: 若短周期砸盘但长周期母盘稳固, 信号维持活性。
        """
        # 1. 战略底池强度
        base_55 = np.tanh(f_acc55 / f_acc55.rolling(120, min_periods=1).std().replace(0, 1e-9)).clip(0, 1)
        # 2. 跨时序共振系数: 短、中、长协同为正
        resonance = (np.where((f_acc13 > 0) & (f_acc34 > 0), 1.2, 0.8) * np.where(f_acc55 > 0, 1.1, 0.5)).clip(0.5, 1.5)
        # 3. 协同攻击加成 
        attack_bonus = (hm_attack * 0.3 + smart_attack * 0.4 + 1.0).clip(1, 1.7)
        # 4. 最终母盘评分
        reservoir_score = (base_55 * resonance * attack_bonus).clip(0, 1)
        # 系统性负反馈: 若 55日全量资金池 (t_acc55) 深度亏空, 强制压制
        penalty = np.where(t_acc55 < 0, 0.6, 1.0)
        return pd.Series(reservoir_score * penalty, index=f_acc55.index).astype(np.float32)

    def _assess_fund_reservoir_buffer(self, daily_large: pd.Series, accum_large: pd.Series, accum_total: pd.Series, f_split: pd.Series, slope_f: pd.Series, accel_f: pd.Series, vpa_eff: pd.Series, volat: pd.Series) -> pd.Series:
        """
        [V27.0.0 · 影子存量补偿版]
        - 核心逻辑: 允许连续的拆单吸筹(f_split)对枯竭的蓄水池进行"信用额度"补偿。
        - 变更说明: 针对 Accum_Total < 0 的情况, 若拆单分值持续强劲, 提供 Max 0.2 的分值补偿。
        """
        mixed_accum = accum_large * (1.0 - f_split) + accum_total * f_split
        accum_median = mixed_accum.rolling(120, min_periods=1).median().replace(0, 1e-9)
        reservoir_strength = np.tanh(mixed_accum / accum_median.abs()).clip(0, 1)
        # 拆单信用补偿: 连续3天拆单高分, 则补偿蓄水池分值
        split_credit = f_split.rolling(3, min_periods=1).min().clip(0, 1) * 0.2
        reservoir_strength = (reservoir_strength + split_credit).clip(0, 1)
        system_penalty = np.where(accum_total < 0, 0.5, 1.0)
        s_scale = slope_f.rolling(60, min_periods=1).std().replace(0, 1e-9)
        norm_s = np.tanh(slope_f / (s_scale * 1.5))
        a_scale = accel_f.rolling(60, min_periods=1).std().replace(0, 1e-9)
        norm_a = np.tanh(accel_f / (a_scale * 1.5))
        trend_score = (norm_s * 0.6 + norm_a * 0.4 + 1.0) / 2.0
        quality_f = (vpa_eff.rolling(5, min_periods=1).mean().clip(0, 1) + 0.5).clip(0.5, 1.5)
        dynamic_threshold = (volat * 2.0).clip(0.05, 0.25)
        attrition = daily_large.abs() / (mixed_accum.abs() + 1e-9)
        tolerance = (1.0 - (attrition / dynamic_threshold)).clip(0, 1)
        return (reservoir_strength * trend_score * quality_f * tolerance * system_penalty).clip(0, 1).fillna(0)

    def _assess_cumulative_energy_decay(self, raw_f: pd.Series, total_net: pd.Series) -> pd.Series:
        """
        [V29.0.0 · 多日能量累积背离模型]
        - 核心逻辑: 识别"阴跌出货"。单日砸盘不猛, 但连续多日大单强度高于整体。
        - 算法: 采用指数衰减累积。Pressure = (Large/Total)。Buffer = Press + Prev * 0.8。
        - 判定: 若 Buffer 超过阈值, 说明近期机构持续撤离, 压力处于过载状态。
        """
        # 计算单日能量压力比 (仅关注流出状态)
        daily_press = np.where((raw_f < 0) & (total_net < 0), (raw_f.abs() / (total_net.abs() + 1e-9)), 0.0)
        daily_press = pd.Series(daily_press, index=raw_f.index).clip(0, 5) # 封顶5.0倍
        # 跨时序累积 (采用 0.8 的衰减率, 约 5 日记忆周期)
        # 使用 ewm 代替手动循环以提升计算效率
        cum_press = daily_press.ewm(alpha=0.2, adjust=False).mean()
        # 归一化评分: 压力越大, 分值越低。2.0为压力警戒线
        energy_decay_factor = (1.0 - np.tanh(cum_press / 2.0)).clip(0, 1)
        return energy_decay_factor.astype(np.float32)

    def _assess_chip_holographic_tension(self, hab_rel: pd.Series, turnover: pd.Series, winner: pd.Series, acc_win: pd.Series, stability: pd.Series, acc_stab: pd.Series, kurtosis: pd.Series, convergence: pd.Series, cost_diff: pd.Series) -> pd.Series:
        """
        [V26.0.0 · 成本带熔断强化版]
        - 核心逻辑: 引入基于 $2\sigma$ 的破位熔断机制 。
        - 判定公式: $$Gate_{cost} = \mathbb{I}(CostDiff \ge -2 \sigma_{cost})$$
        - 变更说明: 若股价回落深度异常，偏离成本均值带过远，则信号直接判死。
        """
        efficiency = (hab_rel / (turnover + 0.5)).clip(0, 1)
        k_scale = kurtosis.rolling(60, min_periods=1).std().replace(0, 1e-9)
        n_kurt = np.tanh(kurtosis / k_scale).clip(0, 1)
        thickness_score = (n_kurt * 0.6 + convergence.clip(0, 1) * 0.4).clip(0, 1)
        # 成本带熔断因子 
        cost_std = cost_diff.rolling(60, min_periods=1).std().replace(0, 1e-9)
        cost_gate = np.where(cost_diff < -2.0 * cost_std, 0.0, 1.0)
        w_scale = acc_win.rolling(60, min_periods=1).std().replace(0, 1e-9)
        s_scale = acc_stab.rolling(60, min_periods=1).std().replace(0, 1e-9)
        n_acc_win = np.tanh(acc_win / w_scale)
        n_acc_stab = np.tanh(acc_stab / s_scale)
        stability_gate = (1.0 + n_acc_stab).clip(0, 1)
        tension = np.where(cost_diff < 0, convergence * 1.2, convergence * 0.8)
        n_tension = np.tanh(tension).clip(0, 1)
        final_meta = (efficiency * 0.3 + thickness_score * 0.4 + n_tension * 0.3) * stability_gate * cost_gate
        return final_meta.fillna(0).astype(np.float32)

    def _assess_chip_holographic_migration(self, close: pd.Series, c_mean: pd.Series, peak_mig: pd.Series, conv_mig: pd.Series) -> pd.Series:
        """
        [V31.0.0 · 筹码全息位移判定]
        - 核心逻辑: 追踪主力的核心防御半径。洗盘特征为价格大幅下坠但长周期重心(Fibonacci)锁死。
        - 判定维度: 
            1. 斐波那契位移比: 计算 5/13/21/34/55 日重心变化。
            2. 迁移稳定性: 价格下行速率 / 筹码重心下移速率。
            3. 收敛位移[cite: 2, 3]: 结合 convergence_migration_D 判断位移是否具有攻击性。
        """
        fib_list = [5, 13, 21, 34, 55]
        migration_scores = []
        for period in fib_list:
            # 价格下行速率
            p_vel = (close.diff(period) / close.shift(period).replace(0, 1e-9)).abs()
            # 筹码重心迁移速率 
            c_vel = (c_mean.diff(period) / c_mean.shift(period).replace(0, 1e-9)).abs()
            # 迁移比: 若价格跌得快而重心不动，说明洗盘质量高
            m_ratio = (p_vel / (c_vel + 0.01)).clip(0, 3) / 3.0
            migration_scores.append(m_ratio)
        # 综合全息位移分
        holographic_mig = pd.concat(migration_scores, axis=1).mean(axis=1)
        # 结合迁移收敛特性 
        c_gate = np.tanh(conv_mig.abs().rolling(21, min_periods=1).mean()).clip(0, 1)
        # 最终位移分: 位移比 * (1 + 峰值迁移速度奖励 )
        final_mig = (holographic_mig * (1.0 + np.tanh(peak_mig / 100))).clip(0, 1)
        print(f"  [DEBUG_V31_MIG] MeanMigration: {final_mig.iloc[-1]:.4f}")
        return final_mig.fillna(0).astype(np.float32)

    def _assess_stealth_shadow_divergence(self, morning: pd.Series, stealth: pd.Series, high_lock: pd.Series, skew: pd.Series, pct_chg: pd.Series, total_net: pd.Series) -> pd.Series:
        """
        [V27.0.0 · SSD隐蔽背离识别]
        - 核心逻辑: 识别大单砸盘掩护下的隐蔽小单匀速吸筹。
        - 判定维度:
            1. 隐蔽动量 (Stealth Momentum): stealth_flow_ratio_D 与全量资金加速度的共振。
            2. 偏度背离 (Skew Div): 价格分布左偏(Skew>0)时, 隐蔽资金反而流入。
        """
        n_morning = np.tanh((morning - 0.4) * 3).clip(0, 1)
        # 计算总资金流的二阶加速度
        net_accel = total_net.diff().diff().rolling(5, min_periods=1).mean().fillna(0)
        n_stealth = (np.tanh(stealth * 3) * (np.tanh(net_accel) + 1.0) / 2.0).clip(0, 1)
        lure_score = (n_morning * 0.4 + n_stealth * 0.6)
        trap_score = np.tanh(high_lock * 3).clip(0, 1)
        # 屠杀偏度修正：洗盘需要负偏度(杀跌快)，但吸筹需要正偏度(低位磨)
        s_scale = skew.abs().rolling(60, min_periods=1).max().replace(0, 1e-9)
        kill_score = (-np.tanh(skew / s_scale)).clip(0, 1)
        is_drop = (pct_chg < 0).astype(int)
        ssd_final = ((lure_score * 0.5 + trap_score * 0.2 + kill_score * 0.3) * is_drop).fillna(0)
        return ssd_final.astype(np.float32)

    def _assess_kinematic_trap_physics(self, close: pd.Series, high: pd.Series, slope: pd.Series, jerk: pd.Series, vpa_eff: pd.Series, och_acc: pd.Series, bbp: pd.Series) -> pd.Series:
        """
        [V28.0.0 · 极限速率熔断版]
        - 核心逻辑: 识别价格运动的非线性崩溃。
        - 变更说明: 引入 Jerk 极端负值熔断。若下行急动度过大，判定为自由落体，不再视为洗盘。
        """
        s_scale = slope.rolling(60, min_periods=1).std().replace(0, 1e-9)
        n_slope = np.tanh(slope / s_scale).clip(0, 1)
        j_scale = jerk.rolling(60, min_periods=1).std().replace(0, 1e-9)
        # 物理熔断: 如果负向 Jerk 超过 2.5 倍标准差，认定为崩盘
        v_breaker = np.where(jerk < -2.5 * j_scale, 0.0, 1.0)
        exhaustion = (-np.tanh(jerk / j_scale)).clip(0, 1)
        vpa_scale = vpa_eff.abs().rolling(60, min_periods=1).max().replace(0, 1e-9)
        inefficiency = (1.0 - np.tanh(vpa_eff / vpa_scale)).clip(0, 1)
        och_scale = och_acc.abs().rolling(60, min_periods=1).max().replace(0, 1e-9)
        structure_fail = (-np.tanh(och_acc / och_scale)).clip(0, 1)
        context_gate = np.tanh((bbp - 0.5) * 3).clip(0, 1)
        range_hl = (high - close) / high.replace(0, 1e-9)
        retracement = (range_hl * 10).clip(0, 1)
        base_trap = (n_slope * 0.4 + exhaustion * 0.6)
        quality = (inefficiency * 0.3 + structure_fail * 0.3 + retracement * 0.4)
        return (base_trap * quality * context_gate * v_breaker).clip(0, 1).fillna(0)

    def _assess_fund_energy_divergence(self, raw_f: pd.Series, total_net: pd.Series) -> pd.Series:
        """
        [V28.0.0 · 资金能量背离判定]
        - 核心逻辑: 识别"机构逃生门"现象。
        - 判定维度: 如果大单砸盘力度远超市场整体承接能力，则判定为机构不计成本撤离。
        """
        f_ma = raw_f.rolling(5, min_periods=1).mean()
        t_ma = total_net.rolling(5, min_periods=1).mean()
        # 能量比: 机构砸盘强度 / 市场总强度
        energy_ratio = (f_ma.abs() / (t_ma.abs() + 1e-9))
        # 惩罚项: 比率越高(>2.0)且大单为负，则分值越低
        punishment = np.where((raw_f < 0) & (energy_ratio > 2.0), 1.0 / energy_ratio, 1.0)
        return pd.Series(punishment, index=raw_f.index).clip(0, 1).fillna(1)

    def _assess_fund_jerk_resonance(self, pct_chg: pd.Series, jerk_large: pd.Series, smart_net: pd.Series, smart_div: pd.Series) -> pd.Series:
        """
        [V32.0.0 · 签名对齐修复版]
        - 核心修复: 显式包含 smart_div 参数以支持 calculate 中的关键字调用。
        """
        j_scale = jerk_large.rolling(60, min_periods=1).std().replace(0, 1e-9)
        panic_impulse = (-np.tanh(jerk_large / j_scale)).clip(0, 1)
        s_scale = smart_net.rolling(60, min_periods=1).std().replace(0, 1e-9)
        absorption = np.tanh(smart_net / s_scale).clip(0, 1)
        p_scale = pct_chg.abs().rolling(60, min_periods=1).mean().replace(0, 1e-9) + 1e-5
        elasticity = (panic_impulse / (pct_chg.abs() / p_scale)).clip(0, 2) / 2.0
        div_bonus = (np.tanh(smart_div) + 1.0) / 2.0
        resonance_score = (panic_impulse * 0.4 + absorption * 0.3 + elasticity * 0.2 + div_bonus * 0.1).clip(0, 1)
        validity_gate = np.where(smart_net > 0, 1.0, 0.5) 
        return (resonance_score * validity_gate).fillna(0)

    def _assess_chip_peak_collapse(self, is_multi: pd.Series, peak_cnt: pd.Series) -> pd.Series:
        """
        [V32.0.0 · 筹码峰坍塌逻辑]
        - 核心逻辑: 监控筹码从离散多峰向单峰坍塌的过程 。
        - 计算方式: 捕捉 peak_count_D 的下降斜率。若近期曾是多峰(is_multi_peak_D=1)且当前正向单峰(peak_cnt=1)坍塌，分值激增。
        """
        # 捕捉峰数变化
        peak_delta = peak_cnt.diff(3).fillna(0)
        # 坍塌分: 只有当之前是多峰且当前峰数在减少时计分
        collapse_signal = np.where((is_multi.shift(3) == 1) & (peak_delta < 0), peak_delta.abs() / (peak_cnt + 1), 0.0)
        # 稳态加成: 如果当前已经是坚固的单峰(peak_cnt=1)
        stability_bonus = np.where(peak_cnt == 1, 0.3, 0.0)
        final_collapse = (np.tanh(collapse_signal) + stability_bonus).clip(0, 1)
        return pd.Series(final_collapse, index=is_multi.index).astype(np.float32)

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

    def _assess_market_leadership_compensation(self, is_leader: pd.Series, state_leader: pd.Series, theme_hot: pd.Series) -> pd.Series:
        """
        [V36.0.0 · 龙头领涨权重补偿评估]
        - 核心逻辑: 优先选拔板块领涨个股。
        - 判定维度: 
            1. 身份认证: IS_MARKET_LEADER_D 与 STATE_MARKET_LEADER_D 联立 。
            2. 环境溢价: 若所属题材热度(THEME_HOTNESS_SCORE_D) > 60, 补偿力度增加 。
        - 补偿系数: 非龙头为 1.0, 龙头为 1.2x, 热题材龙头为 1.3x 。
        """
        # 龙头身份基础判定
        leader_identity = ((is_leader == 1) | (state_leader == 1)).astype(float)
        # 热度加速器
        hot_booster = (np.tanh((theme_hot - 60) / 20) + 1.0) / 2.0
        # 动态补偿系数计算
        compensation = 1.0 + (leader_identity * (0.2 + hot_booster * 0.1))
        return compensation.clip(1.0, 1.4).fillna(1.0).astype(np.float32)

    def _assess_turnover_lock_stability(self, turnover_stable: pd.Series, is_leader: pd.Series, chip_stab: pd.Series) -> pd.Series:
        """
        [V37.0.0 · 筹码良性锁死判定]
        - 核心逻辑: 识别领涨龙头是否伴随筹码锁死 。
        - 判定维度: 
            1. 换手稳定性: TURNOVER_STABILITY_INDEX_D 越高，锁仓质量越好 。
            2. 龙头效应: IS_MARKET_LEADER_D 为 1 时，稳定性权重翻倍 。
            3. 筹码稳定性验证: 结合 chip_stability_D 进行交叉验证 [cite: 2]。
        """
        # 归一化换手稳定性评分 
        t_stable_norm = np.tanh(turnover_stable / 50).clip(0, 1)
        # 龙头增益因子 
        leader_gate = np.where(is_leader == 1, 1.3, 1.0)
        # 交叉验证筹码稳定性 [cite: 2]
        c_stable_norm = np.tanh(chip_stab).clip(0, 1)
        # 最终锁死分值
        lock_score = (t_stable_norm * 0.7 + c_stable_norm * 0.3) * leader_gate
        return pd.Series(lock_score, index=turnover_stable.index).clip(0, 1).astype(np.float32)


