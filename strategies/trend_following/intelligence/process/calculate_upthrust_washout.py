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
        [V24.0.0 · 全息张力执行总控]
        - 变更: 将筹码层升级为全息张力模型，接入峰度、收敛度与稳定性加速度。
        """
        method_name = "CalculateUpthrustWashoutRelationship.calculate"
        (open_p, high_p, low_p, close_p, pct_chg, 
         slope_p, accel_p, jerk_p,                                   
         raw_f, jerk_f, smart_n, smart_d, slope_f_21, accel_f_21, f_accum_34, t_accum_34, f_volat, 
         vpa_eff, och_acc, bbp_pos, test_cnt, cost_mig, sr_ratio, acc_supp, 
         morning, stealth, high_lock, skew, pres_rel, hab_rel, winner, acc_win, turnover, 
         chip_stab, acc_stab_21, cost_diff, chip_kurt, chip_skew, chip_conv, # V24 新增
         abnormal_vol, clustering, order_anomaly, acc_abnormal, volume, trade_count, total_net, 
         ma_res, slope_t) = self._get_raw_signals(df, method_name)
        df_index = df.index
        # 1. 物理/资金/防御/取证 逻辑继承 V23
        k_trap = self._assess_kinematic_trap_physics(close_p, high_p, slope_p, jerk_p, vpa_eff, och_acc, bbp_pos)
        f_resonance = self._assess_fund_jerk_resonance(pct_chg, jerk_f, smart_n, smart_d)
        f_split = self._assess_split_order_accumulation(volume, trade_count, clustering, raw_f, total_net)
        f_active = np.maximum(f_resonance, f_split)
        f_hab = self._assess_fund_reservoir_buffer(raw_f, f_accum_34, t_accum_34, f_split, slope_f_21, accel_f_21, vpa_eff, f_volat)
        fund_score = f_active * f_hab
        # 2. 筹码层 [V24 核心升级]
        chip_meta = self._assess_chip_holographic_tension(hab_rel, turnover, winner, acc_win, chip_stab, acc_stab_21, chip_kurt, chip_conv, cost_diff)
        solidity = self._assess_structural_stress_test(sr_ratio, test_cnt, cost_mig, acc_supp)
        chrono_dec = self._assess_skewed_deception_narrative(morning, stealth, high_lock, skew, pct_chg)
        fractal_man = self._assess_fractal_manipulation_fingerprint(abnormal_vol, clustering, order_anomaly, acc_abnormal)
        context = ((slope_t > 0) | (ma_res > 0.6)).astype(int)
        forensics = (chrono_dec * 0.4 + chip_meta * 0.3 + fractal_man * 0.3)
        final_score = (k_trap * fund_score * forensics * solidity * context).clip(0, 1)
        # 3. 总线打包
        debug_context = {
            "Final": final_score, "Physics": {"Node": k_trap},
            "Funds": {"Node": fund_score, "Split": f_split},
            "Chips": {"Node": chip_meta, "Kurt": chip_kurt, "Conv": chip_conv, "AccWin": acc_win, "AccStab": acc_stab_21, "HAB_Rel": hab_rel, "Turnover": turnover},
            "Defense": {"Node": solidity}, "Forensics": {"Dec": chrono_dec, "Skew": skew}
        }
        self._print_debug_probe(df_index, debug_context)
        return final_score.astype(np.float32).fillna(0.0)

    def _print_debug_probe(self, idx: pd.Index, ctx: Dict[str, Any]):
        """
        [V24.0.0 · 全息筹码探针]
        - 职责: 暴露筹码分布的矩特征(峰度)与结构动量。揭示"洗盘"与"崩溃"的微观差异。
        """
        if len(idx) > 0:
            i = -1
            d_str = idx[i].strftime('%Y-%m-%d')
            print(f"--- [PROBE_V24_CHIP_HOLOGRAPHIC] {d_str} FINAL: {ctx['Final'].iloc[i]:.4f} ---")
            # 筹码层深度溯源
            c = ctx["Chips"]
            print(f"  [3.Chips] Node: {c['Node'].iloc[i]:.4f}")
            print(f"     > Structure: Kurtosis={c['Kurt'].iloc[i]:.2f}, Convergence={c['Conv'].iloc[i]:.4f}")
            print(f"     > Kinematics: WinnerAccel={c['AccWin'].iloc[i]:.4f}, StabAccel21={c['AccStab'].iloc[i]:.4f}")
            print(f"     > Metabolism: HAB_Rel={c['HAB_Rel'].iloc[i]:.2f}, TO={c['Turnover'].iloc[i]:.1f}%")
            # 其余维度关键值
            print(f"  [1.Phys] Trap: {ctx['Physics']['Node'].iloc[i]:.2f}")
            print(f"  [2.Fund] Node: {ctx['Funds']['Node'].iloc[i]:.2f} (Split: {ctx['Funds']['Split'].iloc[i]:.2f})")

    def _get_raw_signals(self, df: pd.DataFrame, method_name: str) -> Tuple[pd.Series, ...]:
        """
        [V24.0.0 · 筹码全息数据接入]
        - 核心职责: 提取筹码分布的统计矩指标(峰度/偏度)与收敛特征。
        - 新增列:
            1. chip_kurtosis_D: 筹码峰度，识别筹码是否向主力核心区集中。
            2. chip_skewness_D: 筹码偏度，识别筹码分布的非对称性。
            3. chip_convergence_ratio_D: 筹码收敛率，量化筹码结构的紧凑程度。
            4. ACCEL_21_chip_stability_D: 筹码稳定性的长周期加速度。
        """
        open_p = self.helper._get_safe_series(df, 'open_D', np.nan, method_name=method_name)
        high_p = self.helper._get_safe_series(df, 'high_D', np.nan, method_name=method_name)
        low_p = self.helper._get_safe_series(df, 'low_D', np.nan, method_name=method_name)
        close_p = self.helper._get_safe_series(df, 'close_D', np.nan, method_name=method_name)
        pct_chg = self.helper._get_safe_series(df, 'pct_change_D', np.nan, method_name=method_name)
        volume = self.helper._get_safe_series(df, 'volume_D', np.nan, method_name=method_name)
        trade_count = self.helper._get_safe_series(df, 'trade_count_D', np.nan, method_name=method_name)
        total_net = self.helper._get_safe_series(df, 'net_amount_D', np.nan, method_name=method_name)
        slope_price = self.helper._get_safe_series(df, 'SLOPE_3_close_D', np.nan, method_name=method_name)
        accel_price = self.helper._get_safe_series(df, 'ACCEL_5_close_D', np.nan, method_name=method_name)
        jerk_price = self.helper._get_safe_series(df, 'JERK_3_close_D', np.nan, method_name=method_name)
        raw_fund = self.helper._get_safe_series(df, 'tick_large_order_net_D', np.nan, method_name=method_name)
        jerk_fund = self.helper._get_safe_series(df, 'JERK_3_tick_large_order_net_D', np.nan, method_name=method_name)
        smart_net = self.helper._get_safe_series(df, 'SMART_MONEY_HM_NET_BUY_D', np.nan, method_name=method_name)
        smart_div = self.helper._get_safe_series(df, 'SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D', np.nan, method_name=method_name)
        slope_fund_21 = self.helper._get_safe_series(df, 'SLOPE_21_tick_large_order_net_D', np.nan, method_name=method_name)
        accel_fund_21 = self.helper._get_safe_series(df, 'ACCEL_21_tick_large_order_net_D', np.nan, method_name=method_name)
        fund_accum_34 = raw_fund.rolling(window=34, min_periods=1).sum() 
        total_accum_34 = total_net.rolling(window=34, min_periods=1).sum()
        fund_volat = self.helper._get_safe_series(df, 'flow_volatility_20d_D', np.nan, method_name=method_name)
        vpa_eff = self.helper._get_safe_series(df, 'VPA_MF_ADJUSTED_EFF_D', np.nan, method_name=method_name)
        och_acc = self.helper._get_safe_series(df, 'OCH_ACCELERATION_D', np.nan, method_name=method_name)
        bbp_pos = self.helper._get_safe_series(df, 'BBP_21_2.0_D', np.nan, method_name=method_name)
        test_cnt = self.helper._get_safe_series(df, 'intraday_support_test_count_D', np.nan, method_name=method_name)
        cost_mig = self.helper._get_safe_series(df, 'intraday_cost_center_migration_D', np.nan, method_name=method_name)
        sr_ratio = self.helper._get_safe_series(df, 'support_resistance_ratio_D', np.nan, method_name=method_name)
        acc_supp = self.helper._get_safe_series(df, 'ACCEL_5_support_strength_D', np.nan, method_name=method_name)
        morning = self.helper._get_safe_series(df, 'morning_flow_ratio_D', np.nan, method_name=method_name)
        stealth = self.helper._get_safe_series(df, 'stealth_flow_ratio_D', np.nan, method_name=method_name)
        high_lock = self.helper._get_safe_series(df, 'intraday_high_lock_ratio_D', np.nan, method_name=method_name)
        skew = self.helper._get_safe_series(df, 'intraday_price_distribution_skewness_D', np.nan, method_name=method_name)
        # [筹码全息维度]
        pres_rel = self.helper._get_safe_series(df, 'pressure_release_index_D', np.nan, method_name=method_name)
        hab_rel = pres_rel.rolling(window=5, min_periods=1).sum()
        winner = self.helper._get_safe_series(df, 'winner_rate_D', np.nan, method_name=method_name)
        acc_win = self.helper._get_safe_series(df, 'ACCEL_5_winner_rate_D', np.nan, method_name=method_name)
        turnover = self.helper._get_safe_series(df, 'turnover_rate_D', np.nan, method_name=method_name)
        chip_stab = self.helper._get_safe_series(df, 'chip_stability_D', np.nan, method_name=method_name)
        acc_stab_21 = self.helper._get_safe_series(df, 'ACCEL_21_chip_stability_D', np.nan, method_name=method_name)
        cost_diff = self.helper._get_safe_series(df, 'chip_cost_to_ma21_diff_D', np.nan, method_name=method_name)
        chip_kurt = self.helper._get_safe_series(df, 'chip_kurtosis_D', np.nan, method_name=method_name)
        chip_skew = self.helper._get_safe_series(df, 'chip_skewness_D', np.nan, method_name=method_name)
        chip_conv = self.helper._get_safe_series(df, 'chip_convergence_ratio_D', np.nan, method_name=method_name)
        abnormal_vol = self.helper._get_safe_series(df, 'tick_abnormal_volume_ratio_D', np.nan, method_name=method_name)
        clustering = self.helper._get_safe_series(df, 'tick_clustering_index_D', np.nan, method_name=method_name)
        order_anomaly = self.helper._get_safe_series(df, 'large_order_anomaly_D', np.nan, method_name=method_name)
        acc_abnormal = self.helper._get_safe_series(df, 'ACCEL_5_tick_abnormal_volume_ratio_D', np.nan, method_name=method_name)
        ma_res = self.helper._get_safe_series(df, 'MA_COHERENCE_RESONANCE_D', np.nan, method_name=method_name)
        slope_trend = self.helper._get_safe_series(df, 'GEOM_REG_SLOPE_D', np.nan, method_name=method_name)
        return (open_p, high_p, low_p, close_p, pct_chg, slope_price, accel_price, jerk_price, 
                raw_fund, jerk_fund, smart_net, smart_div, slope_fund_21, accel_fund_21, fund_accum_34, total_accum_34, fund_volat, 
                vpa_eff, och_acc, bbp_pos, test_cnt, cost_mig, sr_ratio, acc_supp, 
                morning, stealth, high_lock, skew, pres_rel, hab_rel, winner, acc_win, turnover, 
                chip_stab, acc_stab_21, cost_diff, chip_kurt, chip_skew, chip_conv, # 返回新增筹码指标
                abnormal_vol, clustering, order_anomaly, acc_abnormal, volume, trade_count, total_net, 
                ma_res, slope_trend)

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
        [V23.0.0 · 双轨混合蓄水池评估]
        - 核心逻辑: 修正"只看大单"导致的拆单误判。若判定为主力拆单吸筹(f_split高)，则切换至全量资金蓄水池(accum_total)。
        - 判定公式: Final_Accum = Accum_Large * (1 - f_split) + Accum_Total * f_split
        - 损耗评估: 损耗基准随视角同步切换，确保评估尺度的一致性。
        """
        # 1. 混合蓄水池水位计算
        # 若拆单概率高，则全量累积的参考权重增加
        mixed_accum = accum_large * (1.0 - f_split) + accum_total * f_split
        accum_median = mixed_accum.rolling(120, min_periods=1).median().replace(0, 1e-9)
        reservoir_strength = np.tanh(mixed_accum / accum_median.abs()).clip(0, 1)
        # 2. 矢量动量与质量
        s_scale = slope_f.rolling(60, min_periods=1).std().replace(0, 1e-9)
        norm_s = np.tanh(slope_f / (s_scale * 1.5))
        a_scale = accel_f.rolling(60, min_periods=1).std().replace(0, 1e-9)
        norm_a = np.tanh(accel_f / (a_scale * 1.5))
        trend_score = (norm_s * 0.6 + norm_a * 0.4 + 1.0) / 2.0
        quality_f = (vpa_eff.rolling(5, min_periods=1).mean().clip(0, 1) + 0.5).clip(0.5, 1.5)
        # 3. 动态损耗评估 (关键修复)
        # 损耗比率 = 当日大单流出 / 混合蓄水池深度
        # 这种设计允许主力在大单出货(假象)但拆单买入(真相)时，损耗率依然保持在安全区间
        dynamic_threshold = (volat * 2.0).clip(0.05, 0.25)
        attrition = daily_large.abs() / (mixed_accum.abs() + 1e-9)
        tolerance = (1.0 - (attrition / dynamic_threshold)).clip(0, 1)
        # 4. 合成 HAB 信号
        hab_score = (reservoir_strength * trend_score * quality_f * tolerance).clip(0, 1)
        return hab_score.fillna(0).astype(np.float32)

    def _assess_chip_holographic_tension(self, hab_rel: pd.Series, turnover: pd.Series, winner: pd.Series, acc_win: pd.Series, 
                                        stability: pd.Series, acc_stab: pd.Series, kurtosis: pd.Series, convergence: pd.Series, cost_diff: pd.Series) -> pd.Series:
        """
        [V24.0.0 · 筹码全息张力判定]
        - 核心逻辑: 识别筹码结构的"向心力"与"张力"。洗盘成功的标志是峰度上升、收敛度提高，且主力稳定性未加速恶化。
        - 判定维度:
            1. 代谢效率 (Efficiency): HAB_Rel / Turnover。
            2. 结构厚度 (Thickness): Kurtosis(峰度) * Convergence(收敛)。峰度越高，主力成本区筹码越厚。
            3. 运动学熔断 (Circuit Breaker): 获利盘加速度(AccWin)与稳定性加速度(AccStab)双重校验。
            4. 弹性张力 (Elasticity): 成本偏差(CostDiff)与收敛度的负相关背离。
        """
        # 1. 代谢效率与结构厚度 [Source 2]
        efficiency = (hab_rel / (turnover + 0.5)).clip(0, 1)
        # 峰度映射：峰度越大，代表单峰越尖锐，筹码越向主力靠拢
        k_scale = kurtosis.rolling(60, min_periods=1).std().replace(0, 1e-9)
        n_kurt = np.tanh(k_urtosis / k_scale).clip(0, 1)
        thickness_score = (n_kurt * 0.6 + convergence.clip(0, 1) * 0.4).clip(0, 1)
        # 2. 运动学安全门控 [Source 2]
        # 获利盘加速度与稳定性加速度必须处于受控状态
        w_scale = acc_win.rolling(60, min_periods=1).std().replace(0, 1e-9)
        s_scale = acc_stab.rolling(60, min_periods=1).std().replace(0, 1e-9)
        n_acc_win = np.tanh(acc_win / w_scale)
        n_acc_stab = np.tanh(acc_stab / s_scale)
        # 若稳定性在长周期(21日)内加速下降，则视为结构性崩溃
        stability_gate = (1.0 + n_acc_stab).clip(0, 1)
        # 3. 弹性张力评分 [Source 2]
        # 逻辑：股价跌破成本(CostDiff < 0)时，如果收敛度提高，说明主力在承接，张力增加
        tension = np.where(cost_diff < 0, convergence * 1.2, convergence * 0.8)
        n_tension = np.tanh(tension).clip(0, 1)
        # 4. 最终合成
        # 逻辑: (效率 * 0.3 + 厚度 * 0.4 + 张力 * 0.3) * 安全门控
        final_meta = (efficiency * 0.3 + thickness_score * 0.4 + n_tension * 0.3) * stability_gate
        return final_meta.fillna(0).astype(np.float32)

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






