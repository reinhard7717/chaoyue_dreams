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
        [V19.0.0 · 全链路白盒总控]
        - 升级: 增强探针数据流，支持从原料到结果的完整溯源。
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
        chip_meta = self._assess_dynamic_chip_metabolism(hab_release, turnover, winner, acc_win, chip_stab, cost_diff)
        # 5. 防御层 (Structural Stress Test)
        solidity = self._assess_structural_stress_test(sr_ratio, test_cnt, cost_mig, acc_supp)
        # 6. 取证与融合
        chrono_dec = self._assess_skewed_deception_narrative(morning, stealth, high_lock, skew, pct_chg)
        fractal_man = self._assess_fractal_manipulation_fingerprint(abnormal_vol, clustering, order_anomaly, acc_abnormal)
        context = ((slope_t > 0) | (ma_res > 0.6)).astype(int)
        forensics = (chrono_dec * 0.4 + chip_meta * 0.3 + fractal_man * 0.3)
        final_score = (k_trap * fund_score * forensics * solidity * context).clip(0, 1)
        # 7. 全链路探针调用
        self._print_debug_probe(df_index, final_score, 
                                k_trap, och_acc, vpa_eff,                   # 物理维度
                                f_resonance, f_hab, smart_n, jerk_f, f_accum_34, # 资金维度
                                chip_meta, hab_rel, turnover,               # 筹码维度
                                solidity, test_cnt,                         # 防御维度
                                chrono_dec, skew, fractal_man, clustering)  # 取证维度
        return final_score.astype(np.float32).fillna(0.0)

    def _get_raw_signals(self, df: pd.DataFrame, method_name: str) -> Tuple[pd.Series, ...]:
        """
        [V19.1.0 · Bug修复版]
        - 修复: 修正 hab_rel 计算时引用的变量名错误 (pres_release -> pres_rel)。
        - 核心职责: 提取描述"诱饵-关门-屠杀"完整欺骗路径及微观操控的日内结构指标。
        """
        # 基础行情
        open_p = self.helper._get_safe_series(df, 'open_D', 0.0, method_name=method_name)
        high_p = self.helper._get_safe_series(df, 'high_D', 0.0, method_name=method_name)
        low_p = self.helper._get_safe_series(df, 'low_D', 0.0, method_name=method_name)
        close_p = self.helper._get_safe_series(df, 'close_D', 0.0, method_name=method_name)
        pct_chg = self.helper._get_safe_series(df, 'pct_change_D', 0.0, method_name=method_name)
        # 价格运动学
        slope_price = self.helper._get_safe_series(df, 'SLOPE_3_close_D', 0.0, method_name=method_name)
        accel_price = self.helper._get_safe_series(df, 'ACCEL_5_close_D', 0.0, method_name=method_name)
        jerk_price = self.helper._get_safe_series(df, 'JERK_3_close_D', 0.0, method_name=method_name)
        # 资金分层运动学
        raw_fund = self.helper._get_safe_series(df, 'tick_large_order_net_D', 0.0, method_name=method_name)
        jerk_fund = self.helper._get_safe_series(df, 'JERK_3_tick_large_order_net_D', 0.0, method_name=method_name)
        smart_net = self.helper._get_safe_series(df, 'SMART_MONEY_HM_NET_BUY_D', 0.0, method_name=method_name)
        smart_div = self.helper._get_safe_series(df, 'SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D', 0.0, method_name=method_name)
        slope_fund_21 = self.helper._get_safe_series(df, 'SLOPE_21_tick_large_order_net_D', 0.0, method_name=method_name)
        accel_fund_21 = self.helper._get_safe_series(df, 'ACCEL_21_tick_large_order_net_D', 0.0, method_name=method_name)
        fund_accum_34 = raw_fund.rolling(window=34, min_periods=1).sum().fillna(0)
        fund_volat = self.helper._get_safe_series(df, 'flow_volatility_20d_D', 0.0, method_name=method_name)
        # 陷阱结构与环境
        vpa_eff = self.helper._get_safe_series(df, 'VPA_MF_ADJUSTED_EFF_D', 0.0, method_name=method_name)
        och_acc = self.helper._get_safe_series(df, 'OCH_ACCELERATION_D', 0.0, method_name=method_name)
        bbp_pos = self.helper._get_safe_series(df, 'BBP_21_2.0_D', 0.0, method_name=method_name)
        # 结构压力测试
        test_cnt = self.helper._get_safe_series(df, 'intraday_support_test_count_D', 0.0, method_name=method_name)
        cost_mig = self.helper._get_safe_series(df, 'intraday_cost_center_migration_D', 0.0, method_name=method_name)
        sr_ratio = self.helper._get_safe_series(df, 'support_resistance_ratio_D', 1.0, method_name=method_name)
        acc_supp = self.helper._get_safe_series(df, 'ACCEL_5_support_strength_D', 0.0, method_name=method_name)
        # 时空欺骗
        morning = self.helper._get_safe_series(df, 'morning_flow_ratio_D', 0.5, method_name=method_name)
        stealth = self.helper._get_safe_series(df, 'stealth_flow_ratio_D', 0.0, method_name=method_name)
        high_lock = self.helper._get_safe_series(df, 'intraday_high_lock_ratio_D', 0.0, method_name=method_name)
        skew = self.helper._get_safe_series(df, 'intraday_price_distribution_skewness_D', 0.0, method_name=method_name)
        # 筹码代谢
        pres_rel = self.helper._get_safe_series(df, 'pressure_release_index_D', 0.0, method_name=method_name)
        # [修复点] 使用 pres_rel 而非 pres_release
        hab_rel = pres_rel.rolling(window=5, min_periods=1).sum().fillna(0)
        winner = self.helper._get_safe_series(df, 'winner_rate_D', 0.0, method_name=method_name)
        acc_win = self.helper._get_safe_series(df, 'ACCEL_5_winner_rate_D', 0.0, method_name=method_name)
        turnover = self.helper._get_safe_series(df, 'turnover_rate_D', 0.0, method_name=method_name)
        chip_stab = self.helper._get_safe_series(df, 'chip_stability_D', 0.0, method_name=method_name)
        cost_diff = self.helper._get_safe_series(df, 'chip_cost_to_ma21_diff_D', 0.0, method_name=method_name)
        # 分形操控
        abnormal_vol = self.helper._get_safe_series(df, 'tick_abnormal_volume_ratio_D', 0.0, method_name=method_name)
        clustering = self.helper._get_safe_series(df, 'tick_clustering_index_D', 0.0, method_name=method_name)
        order_anomaly = self.helper._get_safe_series(df, 'large_order_anomaly_D', 0.0, method_name=method_name)
        acc_abnormal = self.helper._get_safe_series(df, 'ACCEL_5_tick_abnormal_volume_ratio_D', 0.0, method_name=method_name)
        # 趋势辅助
        ma_res = self.helper._get_safe_series(df, 'MA_COHERENCE_RESONANCE_D', 0.0, method_name=method_name)
        slope_trend = self.helper._get_safe_series(df, 'GEOM_REG_SLOPE_D', 0.0, method_name=method_name)
        return (open_p, high_p, low_p, close_p, pct_chg, slope_price, accel_price, jerk_price, 
                raw_fund, jerk_fund, smart_net, smart_div, slope_fund_21, accel_fund_21, fund_accum_34, fund_volat, 
                vpa_eff, och_acc, bbp_pos, 
                test_cnt, cost_mig, sr_ratio, acc_supp, 
                morning, stealth, high_lock, skew,
                pres_rel, hab_rel, winner, acc_win, turnover, chip_stab, cost_diff, 
                abnormal_vol, clustering, order_anomaly, acc_abnormal,
                ma_res, slope_trend)

    def _assess_fund_reservoir_buffer(self, daily: pd.Series, accum_34: pd.Series, slope_f: pd.Series, accel_f: pd.Series, vpa_eff: pd.Series, volat: pd.Series) -> pd.Series:
        """
        [V12.0.0 · 运动学增强型 HAB 系统]
        - 核心逻辑: 洗盘的前提是"蓄水池稳固"。不仅看存量(Accum)，更看存量的动量(Slope)与惯性(Accel)。
        - 数学模型: Buffer = (存量分 * 趋势分 * 质量分) * 损耗容忍度因子。
        - 趋势分判定: 若 Slope > 0 且 Accel > 0，蓄水池处于扩张态，洗盘置信度极高。
        """
        # 1. 内部自适应映射：存量强度 (基于滚动分位数映射)
        accum_median = accum_34.rolling(120).median().replace(0, 1e-9)
        reservoir_strength = np.tanh(accum_34 / accum_median.abs()).clip(0, 1)
        # 2. 蓄水池动态趋势 (Kinematics) [Source 1]
        # 斜率映射：反映近期资金流入速度
        s_scale = slope_f.rolling(60).std().replace(0, 1e-9)
        norm_s = np.tanh(slope_f / (s_scale * 1.5))
        # 加速度映射：反映资金流入的惯性
        a_scale = accel_f.rolling(60).std().replace(0, 1e-9)
        norm_a = np.tanh(accel_f / (a_scale * 1.5))
        # 趋势得分：速度与加速度的加权叠加，捕捉"减速流出"与"加速流入"的差异
        trend_score = (norm_s * 0.6 + norm_a * 0.4 + 1.0) / 2.0
        # 3. 质量因子与损耗控制 [Source 1, 2]
        # VPA 效率越高，资金累积的含金量越高
        quality_f = (vpa_eff.rolling(5).mean().clip(0, 1) + 0.5).clip(0.5, 1.5)
        # 损耗比率：基于资金波动率的动态阈值。若 volat 极高，允许更大的单日流出。
        dynamic_threshold = (volat * 2.0).clip(0.05, 0.25) 
        attrition = daily.abs() / (accum_34.abs() + 1e-9)
        tolerance = (1.0 - (attrition / dynamic_threshold)).clip(0, 1)
        # 4. 最终合成：暴露蓄水池的动态风险
        # 如果趋势分为负（大撤退），即便存量再大，Buffer 也会被剧烈压低
        hab_score = (reservoir_strength * trend_score * quality_f * tolerance).clip(0, 1)
        return hab_score.astype(np.float32)

    def _assess_dynamic_chip_metabolism(self, hab_release: pd.Series, turnover: pd.Series, winner: pd.Series, acc_win: pd.Series, stability: pd.Series, cost_diff: pd.Series) -> pd.Series:
        """
        [V13.0.0 · 动态筹码代谢与成本锚定]
        - 核心逻辑: 真正的洗盘是"低换手下的高压力释放"(高效率) 且 "筹码结构极度稳定"(主力锁仓)。
        - 判定维度:
            1. 代谢效率 (Efficiency): HAB_Release / Turnover。缩量洗盘是最高境界。
            2. 结构稳固 (Integrity): Winner * Stability。剔除虚假的获利盘。
            3. 成本锚定 (Anchor): 股价回落不应击穿成本防线 (Cost_Diff)。
            4. 运动学熔断: Accel_Winner < 0 时强制降分。
        """
        # 1. 代谢效率计算 (Metabolic Efficiency)
        # 逻辑: 分子是累积释放的套牢盘，分母是消耗的换手率。
        # 加上 epsilon 防止除零。
        eff_ratio = hab_release / (turnover + 0.5)
        # 自适应归一化：通过滚动最大值来评估当前的效率水平
        eff_scale = eff_ratio.rolling(60).max().replace(0, 1e-9)
        efficiency_score = (eff_ratio / eff_scale).clip(0, 1)

        # 2. 结构稳固性 (Structural Integrity) [Source 2, 4]
        # 逻辑: 只有当获利盘比例高(Winner) 且 筹码稳定性高(Stability) 时，才是主力控盘
        norm_winner = winner.rolling(20).rank(pct=True).clip(0, 1) # 使用分位数排名
        norm_stab = stability.rolling(20).rank(pct=True).clip(0, 1)
        integrity_score = (norm_winner * 0.4 + norm_stab * 0.6).clip(0, 1)

        # 3. 成本锚定修正 (Cost Anchor) [Source 2]
        # logic: Cost_Diff (价格与成本均线的距离) 应该在合理范围内。
        # 如果跌破成本线太远 (diff < -5%)，说明防线崩溃。
        # 归一化: 我们希望 diff 维持在 -0.05 到 +0.1 之间。
        anchor_score = np.where(cost_diff < -0.05, 0.2, 1.0) # 简单熔断

        # 4. 运动学熔断 (Kinematic Circuit Breaker)
        # 逻辑: 获利盘加速逃离 (acc_win < 0) 是致命信号
        w_scale = acc_win.rolling(60).std().replace(0, 1e-9)
        n_acc_win = np.tanh(acc_win / w_scale)
        # 如果加速为负，系数迅速从 1.0 降至 0.0
        breaker = (1.0 + n_acc_win).clip(0, 1)

        # 5. 最终合成
        # 效率是基础，稳固是核心，锚定是底线，加速度是开关
        final_meta = (efficiency_score * 0.4 + integrity_score * 0.6) * anchor_score * breaker
        return final_meta.astype(np.float32)

    def _assess_kinematic_trap_physics(self, close: pd.Series, high: pd.Series, slope: pd.Series, jerk: pd.Series, vpa_eff: pd.Series, och_acc: pd.Series, bbp: pd.Series) -> pd.Series:
        """
        [V14.0.0 · 四维全息陷阱模型]
        - 核心逻辑: 物理陷阱 = 运动学力竭(Jerk) * VPA低效(Eff) * 日内结构溃败(OCH) * 阻力位环境(BBP)。
        - 升级维度:
            1. Jerk: 引入急动度，捕捉动能的瞬时反转 (Slope>0但Jerk<0)。
            2. VPA: 引入做功效率，识别"放量滞涨"或"诱多量价"。
            3. Context: 引入布林位置(BBP)，只在天花板附近(>0.8)判定Trap。
        """
        # 1. 运动学力竭 (Kinematic Exhaustion)
        # Slope 归一化: 我们需要正向速度(冲高诱多)
        s_scale = slope.rolling(60).std().replace(0, 1e-9)
        n_slope = np.tanh(slope / s_scale).clip(0, 1)
        # Jerk 归一化: 我们需要负向急动度(突发力竭)
        j_scale = jerk.rolling(60).std().replace(0, 1e-9)
        n_jerk = np.tanh(jerk / j_scale)
        exhaustion = (-n_jerk).clip(0, 1) # Jerk越负，力竭越明显
        
        # 2. VPA 效率背离 (Efficiency Divergence) [Source 1]
        # VPA 效率低 (effort > result) 意味着主力在利用高成交量出货或压盘
        # vpa_eff 通常在 -100 到 100 之间，需自适应处理
        vpa_scale = vpa_eff.abs().rolling(60).max().replace(0, 1e-9)
        # 我们寻找 VPA 效率低下的时刻 (Eff < 0 或 极小)
        inefficiency = (1.0 - np.tanh(vpa_eff / vpa_scale)).clip(0, 1)

        # 3. 日内结构溃败 (Intraday Failure) [Source 1]
        # OCH_ACCELERATION < 0 代表 开盘->高点->收盘 的过程中动能衰减
        och_scale = och_acc.abs().rolling(60).max().replace(0, 1e-9)
        structure_fail = (-np.tanh(och_acc / och_scale)).clip(0, 1)

        # 4. 阻力位环境 (Resistance Context) [Source 1]
        # BBP > 0.8 表示股价处于布林通道上轨区域，这里是"诱多"的最佳场所
        # 如果在底部震荡(BBP < 0.5)，则不太可能是 Upthrust Trap
        context_gate = np.tanh((bbp - 0.5) * 3).clip(0, 1)

        # 5. 回落幅度 (Retracement) - 经典定义
        range_hl = (high - close) / high.replace(0, 1e-9)
        retracement = (range_hl * 10).clip(0, 1)

        # 6. 四维融合
        # 基础陷阱分 = 速度 * 力竭
        base_trap = (n_slope * 0.4 + exhaustion * 0.6)
        # 质量加权 = VPA低效 * 结构溃败 * 回落幅度
        quality = (inefficiency * 0.3 + structure_fail * 0.3 + retracement * 0.4)
        
        final_trap = (base_trap * quality * context_gate).clip(0, 1)
        return final_trap.astype(np.float32)

    def _assess_fund_jerk_resonance(self, pct_chg: pd.Series, jerk_large: pd.Series, smart_net: pd.Series, smart_div: pd.Series) -> pd.Series:
        """
        [V15.0.0 · 资金分层共振模型]
        - 核心逻辑: 识别"通用大单恐慌(Jerk < 0)"与"聪明钱从容承接(Smart > 0)"的背离共振。
        - 升级维度:
            1. 分层博弈: 区分 Panic Money (Jerk) 与 Smart Money (Net)。
            2. 弹性系数: 评估股价对资金冲击的抵抗力。
        """
        # 1. 通用大单急动度 (The Panic)
        # 归一化: 我们寻找显著的负向急动度 (抛压突增)
        j_scale = jerk_large.rolling(60).std().replace(0, 1e-9)
        panic_impulse = (-np.tanh(jerk_large / j_scale)).clip(0, 1)
        # 2. 聪明钱承接 (The Absorption) [Source 1]
        # 归一化: 我们寻找正向的主力净买入
        s_scale = smart_net.rolling(60).std().replace(0, 1e-9)
        absorption = np.tanh(smart_net / s_scale).clip(0, 1)
        # 3. 价格弹性 (The Resilience)
        # 逻辑: 抛压很大(Jerk大)但跌幅很小(PctChg小) = 强承接
        # 使用 abs() 处理，关注幅度的不对称性
        p_scale = pct_chg.abs().rolling(60).mean().replace(0, 1e-9) + 1e-5
        # 弹性 = 抛压强度 / 跌幅强度。弹性越高，说明承接盘越硬。
        elasticity = (panic_impulse / (pct_chg.abs() / p_scale)).clip(0, 2) / 2.0
        # 4. 资金背离验证 (The Divergence) [Source 1]
        # smart_div > 0 通常暗示机构与游资的博弈有利于洗盘
        div_bonus = (np.tanh(smart_div) + 1.0) / 2.0
        # 5. 共振合成
        # 核心场景: 市场在恐慌(Panic)，主力在买(Absorption)，价格跌不动(Elasticity)
        resonance_score = (panic_impulse * 0.4 + absorption * 0.3 + elasticity * 0.2 + div_bonus * 0.1).clip(0, 1)
        # 修正: 如果主力也在跑 (Absorption < 0)，则 resonance 归零，避免误判
        validity_gate = np.where(smart_net > 0, 1.0, 0.5) 
        return (resonance_score * validity_gate).astype(np.float32)

    def _assess_structural_stress_test(self, ratio_sr: pd.Series, test_count: pd.Series, cost_mig: pd.Series, acc_supp: pd.Series) -> pd.Series:
        """
        [V16.0.0 · 结构压力测试模型]
        - 核心逻辑: 真正的防线必须经得起炮火洗礼。
        - 判定维度:
            1. 攻防比 (Ratio): 支撑/压力。基础厚度。
            2. 压力测试 (Test): 日内支撑位被测试次数。次数越多且不破，韧性越强。
            3. 重心引力 (Gravity): 成本重心迁移。不能发生崩塌式下移。
            4. 动态增强 (Accel): 支撑强度的加速度。测试中是否在加固工事。
        """
        # 1. 攻防比率 (Static Base) [Source 3]
        # ratio > 1 代表支撑强。归一化到 [0, 1]，中心点为 1.0
        base_solidity = np.tanh(ratio_sr - 0.8).clip(0, 1) # 0.8 为及格线
        
        # 2. 压力测试奖励 (Stress Bonus) [Source 3]
        # test_count 越多越好。比如测试了 5 次，说明主力护盘意愿极强。
        # 使用 log 衰减，避免次数过多导致溢出
        test_bonus = np.log1p(test_count).clip(0, 2) / 2.0
        # 只有在 base_solidity 及格时，测试才有意义。否则是"纸糊的墙被捅了很多次"
        resilience = base_solidity * (0.5 + 0.5 * test_bonus)
        
        # 3. 重心稳定性 (Gravity Stability) [Source 2]
        # cost_mig (成本迁移) 应该是正的(上升)或者微负(抵抗式下跌)。
        # 如果大幅为负 (<-0.5)，说明防线在撤退，即使支撑厚也是且战且退。
        c_scale = cost_mig.abs().rolling(60).mean().replace(0, 1e-9)
        n_mig = cost_mig / c_scale
        # 允许微跌 (-0.5)，但严惩暴跌
        gravity_stable = (np.tanh(n_mig + 0.5) + 1.0) / 2.0
        
        # 4. 动态增强趋势 (Kinematic Boost)
        # 支撑加速度 > 0，说明在测试过程中，主力在不断加单
        a_scale = acc_supp.rolling(60).std().replace(0, 1e-9)
        n_acc = np.tanh(acc_supp / a_scale)
        boost = (1.0 + n_acc * 0.5).clip(0.5, 1.5)
        
        # 5. 最终合成
        # 韧性(经测试的厚度) * 稳定性(重心不崩) * 趋势(动态加固)
        final_solidity = (resilience * gravity_stable * boost).clip(0, 1)
        return final_solidity.astype(np.float32)

    def _assess_skewed_deception_narrative(self, morning: pd.Series, stealth: pd.Series, high_lock: pd.Series, skew: pd.Series, pct_chg: pd.Series) -> pd.Series:
        """
        [V17.0.0 · 时空偏度欺骗模型]
        - 核心逻辑: 洗盘是一个"诱饵-囚禁-屠杀"的完整时空叙事。
        - 判定维度:
            1. 诱饵 (Lure): 早盘放量(Morning) + 隐蔽运作(Stealth)。
            2. 囚禁 (Trap): 日内高位锁定率(HighLock)高，大量筹码在高位成交。
            3. 屠杀 (Kill): 价格分布负偏(Skew < 0)，尾盘快速杀跌。
        """
        # 1. 诱饵系数 (The Lure) [Source 2, 3]
        # 早盘放量且隐蔽资金活跃，说明主力在暗中布局
        n_morning = np.tanh((morning - 0.4) * 3).clip(0, 1)
        n_stealth = np.tanh(stealth * 2).clip(0, 1)
        lure_score = (n_morning * 0.6 + n_stealth * 0.4)
        
        # 2. 囚禁系数 (The Trap) [Source 2]
        # 高位锁定率越高，说明诱多越成功，被套牢的筹码越多
        # high_lock 通常在 0-1 之间
        trap_score = np.tanh(high_lock * 3).clip(0, 1)
        
        # 3. 屠杀偏度 (The Kill) [Source 3]
        # 负偏度代表价格在高位停留久，然后快速下跌。正偏度代表在低位停留久(吸筹)。
        # 洗盘需要负偏度 (Skew < 0)。取反后，值越大越好。
        s_scale = skew.abs().rolling(60).max().replace(0, 1e-9)
        kill_score = (-np.tanh(skew / s_scale)).clip(0, 1)
        
        # 4. 价格确认 (Price Validation)
        # 必须是下跌状态，跌幅越深，欺骗性质越恶劣 (但不能跌停，否则是出货)
        is_drop = (pct_chg < 0).astype(int)
        
        # 5. 最终合成
        # 只有在下跌时，上述欺骗特征才有效
        deception = (lure_score * 0.4 + trap_score * 0.3 + kill_score * 0.3) * is_drop
        return deception.astype(np.float32)

    def _assess_fractal_manipulation_fingerprint(self, abnormal: pd.Series, clustering: pd.Series, anomaly: pd.Series, acc_abnormal: pd.Series) -> pd.Series:
        """
        [V18.0.0 · 分形操控指纹模型]
        - 核心逻辑: 洗盘是人为的，自然恐慌是随机的。利用微观结构的非随机性(聚类/离群)识别操控。
        - 判定维度:
            1. 聚类 (Clustering): Tick级交易的规律性。算法交易的痕迹。
            2. 离群 (Anomaly): 大单分布的统计学异常。人为压盘的痕迹。
            3. 强度 (Volume): 异常成交比率。
            4. 动力 (Kinematics): 异常行为的加速度。
        """
        # 1. 强度指纹 (Volume) [Source 3]
        n_abnormal = np.tanh(abnormal * 2).clip(0, 1)
        # 2. 聚类指纹 (Clustering) [Source 3]
        # 聚类指数越高，说明交易越有组织(非随机散户)
        n_clustering = np.tanh(clustering * 3).clip(0, 1)
        # 3. 离群指纹 (Anomaly) [Source 2]
        # 异常度越高，说明盘口挂单越诡异
        n_anomaly = np.tanh(anomaly).clip(0, 1)
        # 4. 动力增强 (Kinematics)
        # 如果异常成交在加速 (acc > 0)，说明主力在加力
        a_scale = acc_abnormal.rolling(60).std().replace(0, 1e-9)
        n_acc = np.tanh(acc_abnormal / a_scale)
        boost = (1.0 + n_acc * 0.5).clip(0.8, 1.5)
        # 5. 指纹合成
        # 只有当 量大(Volume) + 有组织(Clustering) + 诡异(Anomaly) 同时出现，才是主力操控
        base_manipulation = (n_abnormal * 0.4 + n_clustering * 0.4 + n_anomaly * 0.2)
        final_score = (base_manipulation * boost).clip(0, 1)
        return final_score.astype(np.float32)

    def _print_debug_probe(self, idx: pd.Index, final: pd.Series, 
                           trap: pd.Series, och: pd.Series, vpa: pd.Series,
                           f_res: pd.Series, f_hab: pd.Series, smart: pd.Series, jerk: pd.Series, accum: pd.Series,
                           meta: pd.Series, hab_rel: pd.Series, turnover: pd.Series,
                           solid: pd.Series, test: pd.Series,
                           dec: pd.Series, skew: pd.Series, man: pd.Series, cluster: pd.Series):
        """
        [V19.0.0 · 全链路白盒探针]
        - 职责: 输出从"原料(Raw)"到"节点(Node)"再到"结果(Result)"的完整逻辑链。
        - 格式: [层级] NodeScore <- RawData1, RawData2...
        """
        if len(idx) > 0:
            i = -1 # 取最后一天
            date_str = idx[i].strftime('%Y-%m-%d')
            print(f"--- [PROBE_V19_WHITEBOX] {date_str} FINAL_SCORE: {final.iloc[i]:.4f} ---")
            # 1. 物理层: 陷阱形态是否成立?
            # 逻辑: 陷阱分 <- 日内结构(OCH) + VPA效率(VPA)
            print(f"  [1.Physics]  Trap: {trap.iloc[i]:.4f} <- OCH_Accel: {och.iloc[i]:.4f}, VPA_Eff: {vpa.iloc[i]:.2f}")
            # 2. 资金层: 主力是否在承接? 底仓是否够厚?
            # 逻辑: 资金分 <- (共振分 <- 聪明钱 vs 恐慌急动度) * (HAB分 <- 34日累积)
            f_total = (f_res * f_hab).iloc[i]
            print(f"  [2.Funds]    Score: {f_total:.4f} (Res: {f_res.iloc[i]:.4f} * HAB: {f_hab.iloc[i]:.4f})")
            print(f"               Raw -> SmartNet: {smart.iloc[i]:.2f} vs PanicJerk: {jerk.iloc[i]:.2f} | Accum34: {accum.iloc[i]:.2f}")
            # 3. 筹码层: 洗盘效率高吗?
            # 逻辑: 代谢分 <- 5日累积释放 / 换手率
            print(f"  [3.Chips]    Meta: {meta.iloc[i]:.4f} <- HAB_Rel: {hab_rel.iloc[i]:.2f} / Turnover: {turnover.iloc[i]:.2f}%")
            # 4. 防御层: 支撑经得起考验吗?
            # 逻辑: 稳固分 <- 支撑测试次数
            print(f"  [4.Defense]  Solid: {solid.iloc[i]:.4f} <- TestCount: {test.iloc[i]:.1f}")
            # 5. 取证层: 是诱多吗? 是操控吗?
            # 逻辑: 欺骗分 <- 偏度 | 操控分 <- 聚类
            print(f"  [5.Forensics] Decept: {dec.iloc[i]:.4f} (Skew: {skew.iloc[i]:.4f}) | Manip: {man.iloc[i]:.4f} (Cluster: {cluster.iloc[i]:.4f})")






