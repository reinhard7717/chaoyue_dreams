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
        [V31.0.0 · 筹码全息位移总控]
        - 变更说明: 修正 V30 数据引用错误，改用 ATR  与全息位移。
        - 逻辑链: 物理熔断 -> 资金蓄水池 -> 全息筹码位移 -> 环境熵校准。
        """
        method_name = "CalculateUpthrustWashoutRelationship.calculate"
        (open_p, high_p, low_p, close_p, pct_chg, p_entropy, slope_p, jerk_p, atr_14, vol_f_10, vol_f_20, c_mean, peak_mig, mig_conv, conv_mig, raw_f, jerk_f, smart_n, f_acc34, t_acc34, vpa_eff, och_acc, bbp_pos, test_cnt, cost_mig, sr_ratio, acc_supp, winner, acc_win, chip_stab, acc_stab_21, cost_diff, chip_kurt, chip_conv, morning, stealth, high_lock, skew, pres_rel, hab_rel, turnover, clustering, order_anomaly, volume, trade_count, total_net, ma_res, slope_t) = self._get_raw_signals(df, method_name)
        df_index = df.index
        k_trap = self._assess_kinematic_trap_physics(close_p, high_p, slope_p, jerk_p, vpa_eff, och_acc, bbp_pos)
        # 资金层保持 V29 逻辑
        f_resonance = self._assess_fund_jerk_resonance(pct_chg, jerk_f, smart_n, smart_d=pd.Series(0, index=df_index))
        f_split = self._assess_split_order_accumulation(volume, trade_count, clustering, raw_f, total_net)
        f_energy = self._assess_cumulative_energy_decay(raw_f, total_net)
        f_hab = self._assess_fund_reservoir_buffer(raw_f, f_acc34, t_acc34, f_split, pd.Series(0, index=df_index), pd.Series(0, index=df_index), vpa_eff, vol_f_20)
        fund_score = np.maximum(f_resonance, f_split) * f_hab * f_energy
        # [V31 核心] 全息筹码位移
        chip_migration = self._assess_chip_holographic_migration(close_p, c_mean, peak_mig, conv_mig)
        chip_tension = self._assess_chip_holographic_tension(hab_rel, turnover, winner, acc_win, chip_stab, acc_stab_21, chip_kurt, chip_conv, cost_diff)
        # [V31 修正] 基于 ATR 的 Beta 对冲 
        atr_norm = (atr_14 / close_p.replace(0, 1e-9)).rolling(60, min_periods=1).rank(pct=True)
        beta_hedge = (1.2 - atr_norm).clip(0, 1)
        # 融合输出
        forensics = (chip_migration * 0.4 + chip_tension * 0.4 + np.tanh(clustering) * 0.2)
        e_scale = p_entropy.rolling(60, min_periods=1).mean().replace(0, 1e-9)
        entropy_multi = (1.2 - np.tanh(p_entropy / e_scale)).clip(0, 1)
        final_score = (k_trap * fund_score * beta_hedge * forensics * entropy_multi).clip(0, 1)
        self._print_debug_probe(df_index, {"Final": final_score, "Mig": chip_migration, "Fund": fund_score, "Beta": beta_hedge, "ATR": atr_14, "CMean": c_mean})
        return final_score.astype(np.float32).fillna(0.0)

    def _print_debug_probe(self, idx: pd.Index, ctx: Dict[str, Any]):
        """
        [V31.0.0 · 全息位移探针]
        - 职责: 揭示筹码重心与价格波动的时空背离。
        """
        if len(idx) > 0:
            i = -1
            dt = idx[i].strftime('%Y-%m-%d')
            print(f"--- [PROBE_V31_MIGRATION] {dt} FINAL: {ctx['Final'].iloc[i]:.4f} ---")
            print(f"  [Holographic] MigrationScore: {ctx['Mig'].iloc[i]:.4f} (ChipMean: {ctx['CMean'].iloc[i]:.2f})")
            print(f"  [Hedge]       BetaFactor: {ctx['Beta'].iloc[i]:.4f} (ATR_14: {ctx['ATR'].iloc[i]:.2f})")
            print(f"  [Foundation]  FundScore: {ctx['Fund'].iloc[i]:.4f}")

    def _get_raw_signals(self, df: pd.DataFrame, method_name: str) -> Tuple[pd.Series, ...]:
        """
        [V31.0.0 · 筹码全息位移数据接入]
        - 核心修复: 移除不存在的 volatility_5d_D 等列，改用清单内存在的 flow_volatility_10d_D  及 ATR_14_D 。
        - 新增列: chip_mean_D , peak_migration_speed_5d_D , migration_convergence_ratio_D , convergence_migration_D.
        """
        open_p = self.helper._get_safe_series(df, 'open_D', np.nan, method_name=method_name)
        high_p = self.helper._get_safe_series(df, 'high_D', np.nan, method_name=method_name)
        low_p = self.helper._get_safe_series(df, 'low_D', np.nan, method_name=method_name)
        close_p = self.helper._get_safe_series(df, 'close_D', np.nan, method_name=method_name)
        pct_chg = self.helper._get_safe_series(df, 'pct_change_D', np.nan, method_name=method_name)
        p_entropy = self.helper._get_safe_series(df, 'PRICE_ENTROPY_D', np.nan, method_name=method_name)
        slope_p = self.helper._get_safe_series(df, 'SLOPE_3_close_D', np.nan, method_name=method_name)
        jerk_p = self.helper._get_safe_series(df, 'JERK_3_close_D', np.nan, method_name=method_name)
        atr_14 = self.helper._get_safe_series(df, 'ATR_14_D', np.nan, method_name=method_name) # 真实波动率 
        vol_f_10 = self.helper._get_safe_series(df, 'flow_volatility_10d_D', np.nan, method_name=method_name) # 资金波动 
        vol_f_20 = self.helper._get_safe_series(df, 'flow_volatility_20d_D', np.nan, method_name=method_name)
        c_mean = self.helper._get_safe_series(df, 'chip_mean_D', np.nan, method_name=method_name) # 筹码重心 
        peak_mig_5 = self.helper._get_safe_series(df, 'peak_migration_speed_5d_D', np.nan, method_name=method_name) # 迁移速度 
        mig_conv = self.helper._get_safe_series(df, 'migration_convergence_ratio_D', np.nan, method_name=method_name) # 迁移收敛比 
        conv_mig = self.helper._get_safe_series(df, 'convergence_migration_D', np.nan, method_name=method_name) # 收敛迁移量 
        raw_fund = self.helper._get_safe_series(df, 'tick_large_order_net_D', np.nan, method_name=method_name)
        jerk_f = self.helper._get_safe_series(df, 'JERK_3_tick_large_order_net_D', np.nan, method_name=method_name)
        smart_n = self.helper._get_safe_series(df, 'SMART_MONEY_HM_NET_BUY_D', np.nan, method_name=method_name)
        f_accum_34 = raw_fund.rolling(window=34, min_periods=1).sum() 
        t_accum_34 = self.helper._get_safe_series(df, 'net_amount_D', np.nan, method_name=method_name).rolling(34, min_periods=1).sum()
        total_net = self.helper._get_safe_series(df, 'net_amount_D', np.nan, method_name=method_name)
        vpa_eff = self.helper._get_safe_series(df, 'VPA_MF_ADJUSTED_EFF_D', np.nan, method_name=method_name)
        och_acc = self.helper._get_safe_series(df, 'OCH_ACCELERATION_D', np.nan, method_name=method_name)
        bbp_pos = self.helper._get_safe_series(df, 'BBP_21_2.0_D', np.nan, method_name=method_name)
        test_cnt = self.helper._get_safe_series(df, 'intraday_support_test_count_D', np.nan, method_name=method_name)
        cost_mig = self.helper._get_safe_series(df, 'intraday_cost_center_migration_D', np.nan, method_name=method_name)
        sr_ratio = self.helper._get_safe_series(df, 'support_resistance_ratio_D', np.nan, method_name=method_name)
        acc_supp = self.helper._get_safe_series(df, 'ACCEL_5_support_strength_D', np.nan, method_name=method_name)
        winner = self.helper._get_safe_series(df, 'winner_rate_D', np.nan, method_name=method_name)
        acc_win = self.helper._get_safe_series(df, 'ACCEL_5_winner_rate_D', np.nan, method_name=method_name)
        chip_stab = self.helper._get_safe_series(df, 'chip_stability_D', np.nan, method_name=method_name)
        acc_stab_21 = self.helper._get_safe_series(df, 'ACCEL_21_chip_stability_D', np.nan, method_name=method_name)
        cost_diff = self.helper._get_safe_series(df, 'chip_cost_to_ma21_diff_D', np.nan, method_name=method_name)
        chip_kurt = self.helper._get_safe_series(df, 'chip_kurtosis_D', np.nan, method_name=method_name)
        chip_conv = self.helper._get_safe_series(df, 'chip_convergence_ratio_D', np.nan, method_name=method_name)
        morning = self.helper._get_safe_series(df, 'morning_flow_ratio_D', np.nan, method_name=method_name)
        stealth = self.helper._get_safe_series(df, 'stealth_flow_ratio_D', np.nan, method_name=method_name)
        high_lock = self.helper._get_safe_series(df, 'intraday_high_lock_ratio_D', np.nan, method_name=method_name)
        skew = self.helper._get_safe_series(df, 'intraday_price_distribution_skewness_D', np.nan, method_name=method_name)
        pres_rel = self.helper._get_safe_series(df, 'pressure_release_index_D', np.nan, method_name=method_name)
        hab_rel = pres_rel.rolling(window=5, min_periods=1).sum()
        turnover = self.helper._get_safe_series(df, 'turnover_rate_D', np.nan, method_name=method_name)
        clustering = self.helper._get_safe_series(df, 'tick_clustering_index_D', np.nan, method_name=method_name)
        order_anomaly = self.helper._get_safe_series(df, 'large_order_anomaly_D', np.nan, method_name=method_name)
        volume = self.helper._get_safe_series(df, 'volume_D', np.nan, method_name=method_name)
        trade_count = self.helper._get_safe_series(df, 'trade_count_D', np.nan, method_name=method_name)
        ma_res = self.helper._get_safe_series(df, 'MA_COHERENCE_RESONANCE_D', np.nan, method_name=method_name)
        slope_t = self.helper._get_safe_series(df, 'GEOM_REG_SLOPE_D', np.nan, method_name=method_name)
        return (open_p, high_p, low_p, close_p, pct_chg, p_entropy, slope_p, jerk_p, atr_14, vol_f_10, vol_f_20, c_mean, peak_mig_5, mig_conv, conv_mig, raw_fund, jerk_f, smart_n, f_accum_34, t_accum_34, vpa_eff, och_acc, bbp_pos, test_cnt, cost_mig, sr_ratio, acc_supp, winner, acc_win, chip_stab, acc_stab_21, cost_diff, chip_kurt, chip_conv, morning, stealth, high_lock, skew, pres_rel, hab_rel, turnover, clustering, order_anomaly, volume, trade_count, total_net, ma_res, slope_t)

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






