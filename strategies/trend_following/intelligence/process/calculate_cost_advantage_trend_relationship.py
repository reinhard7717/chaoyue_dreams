# strategies\trend_following\intelligence\process\calculate_cost_advantage_trend_relationship.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from strategies.trend_following.utils import get_params_block, get_param_value
from strategies.trend_following.intelligence.process.helper import ProcessIntelligenceHelper

class CalculateCostAdvantageTrendRelationship:
    """
    【V9.0.0 · 成本优势趋势运动学计算器】
    PROCESS_META_COST_ADVANTAGE_TREND
    - 核心职责: 基于“军械库”新数据，计算筹码获利结构与趋势运动学的共振关系。
    - 核心模型: 获利盘穿透率 + 聪明钱攻击向量 + 均线相干运动学
    - 数据源: 严格依赖最终军械库清单
    - 版本: 9.0.0 (Refactored for Arsenal Data)
    """
    def __init__(self, strategy_instance, helper_instance: ProcessIntelligenceHelper):
        """初始化处理器，加载探针与配置"""
        self.strategy = strategy_instance
        self.helper = helper_instance
        self.params = self.helper.params
        self.debug_params = self.helper.debug_params
        self.probe_dates = self.helper.probe_dates

    def _initialize_debug_context(self, method_name: str, df: pd.DataFrame) -> Tuple[bool, Optional[pd.Timestamp], Dict, Dict]:
        """初始化调试上下文，定位探针日期"""
        is_debug = get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug and self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            # 精确匹配或寻找最近的前向日期
            for date in reversed(df.index):
                if pd.to_datetime(date).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        debug_output = {}
        temp_vals = {}
        if is_debug and probe_ts:
            print(f"【V9.0探针激活】目标日期: {probe_ts.strftime('%Y-%m-%d')} | 方法: {method_name}")
            debug_output[f"--- {method_name} Probe @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
        return is_debug and (probe_ts is not None), probe_ts, debug_output, temp_vals

    def _probe_val(self, key: str, val: Any, temp_vals: Dict, section: str = "General"):
        """记录探针数据"""
        if section not in temp_vals:
            temp_vals[section] = {}
        temp_vals[section][key] = val

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.5.0 · 成本优势趋势完全体：专属归一化与动力学重构】
        核心逻辑：
        1. 摒弃通用归一化，采用针对信号物理特性的专属数学变换。
        2. 筹码维度：针对百分比数据采用"中心化线性映射"。
        3. 资金维度：针对长尾分布资金数据采用"鲁棒Z-Score + Tanh压缩"。
        4. 趋势维度：针对稀疏异动数据采用"分位数截断映射"。
        5. 高阶动力学：针对高噪导数采用"自适应波动率缩放"。
        """
        method_name = "CalculateCostAdvantageTrend_V9_5"
        print(f"【V9.5.0】启动专属归一化趋势计算，数据形状: {df.shape}")
        is_debug, probe_ts, debug_out, temp_vals = self._initialize_debug_context(method_name, df)
        df_processed = self._check_and_repair_signals(df.copy(), method_name)
        df_index = df_processed.index
        # 1. 基础维度计算 (内置专属归一化)
        # A. 筹码获利动力学
        chip_score = self._calculate_chip_profitability_dynamics(df_processed, df_index, is_debug, probe_ts, temp_vals)
        # B. 聪明钱HAB存量博弈
        smart_money_score = self._calculate_smart_money_hab_dynamics(df_processed, df_index, is_debug, probe_ts, temp_vals)
        # C. 趋势运动与异动聚类
        trend_micro_score = self._calculate_kinematic_anomaly_clustering(df_processed, df_index, is_debug, probe_ts, temp_vals)
        # 2. 高阶动力学维度
        dynamics_score = self._calculate_high_order_dynamics(df_processed, df_index, is_debug, probe_ts, temp_vals)
        # 3. 多维共振融合
        base_resonance = self._calculate_multi_dimension_resonance(
            chip_score, smart_money_score, trend_micro_score, df_processed, df_index, is_debug, probe_ts, temp_vals
        )
        # 4. 动力学爆发修正
        final_score = base_resonance * (0.6 + 0.4 * dynamics_score)
        if is_debug:
            self._probe_val("Base_Resonance", base_resonance.loc[probe_ts], temp_vals, "FinalSynthesis")
            self._probe_val("Dynamics_Multiplier", dynamics_score.loc[probe_ts], temp_vals, "FinalSynthesis")
            self._log_debug_values(debug_out, temp_vals, probe_ts, method_name)
        print(f"【V9.5.0】计算完成。Score Range: [{final_score.min():.3f}, {final_score.max():.3f}]")
        return final_score.astype(np.float32).fillna(0).clip(-1, 1)

    def _calculate_high_order_dynamics(self, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        """【V9.5.0 · 高阶动力学】
        专属归一化：
        1. 导数(Jerk/Accel): 自适应波动率标准化 + Tanh (Adaptive Vol-Std + Tanh)。
        2. 噪音控制: 死寂市场强制归零 (Dead Market Suppression)。
        """
        # 1. 资金动力学
        sm_slope = self.helper._get_safe_series(df, 'SLOPE_13_SMART_MONEY_HM_NET_BUY_D', 0.0, "sm_slope")
        sm_accel = self.helper._get_safe_series(df, 'ACCEL_13_SMART_MONEY_HM_NET_BUY_D', 0.0, "sm_accel")
        # 内置归一化逻辑：Slope/Accel
        # 使用 Slope 自身的波动率
        sm_slope_std = sm_slope.rolling(34).std().replace(0, 1.0)
        sm_accel_std = sm_accel.rolling(34).std().replace(0, 1.0)
        # Z-Score -> Tanh
        sm_slope_norm = np.tanh(sm_slope / (sm_slope_std * 1.5))
        sm_accel_norm = np.tanh(sm_accel / (sm_accel_std * 1.5))
        sm_dynamic_score = sm_slope_norm * 0.5 + sm_accel_norm * 0.5
        # 2. 价格动力学 (重点处理 Jerk)
        p_accel = self.helper._get_safe_series(df, 'ACCEL_13_close_D', 0.0, "p_accel")
        p_jerk = self.helper._get_safe_series(df, 'JERK_13_close_D', 0.0, "p_jerk")
        # Accel 归一化
        p_accel_std = p_accel.rolling(34).std().replace(0, 1.0)
        p_accel_norm = np.tanh(p_accel / (p_accel_std * 1.5))
        # Jerk 归一化：Jerk 极不稳定，需要更大的平滑窗口和更强的压缩
        p_jerk_std = p_jerk.rolling(55).std().replace(0, 1.0)
        # Sensitivity = 2.0 (更不敏感)
        p_jerk_norm = np.tanh(p_jerk / (p_jerk_std * 2.0))
        # 动力学组合：Accel为主，Jerk为早期预警
        p_dynamic_score = p_accel_norm * 0.6 + p_jerk_norm * 0.4
        # 3. 综合动力学
        raw_dynamics = sm_dynamic_score * 0.55 + p_dynamic_score * 0.45
        # 4. 死寂市场抑制 (Dead Market Suppression)
        # 如果 ATR 处于历史最低 10%，说明市场无波动，此时的高阶导数全是噪音
        atr = self.helper._get_safe_series(df, 'ATR_14_D', 0.0, "atr")
        # 计算 ATR 在过去 89 天的分位数
        atr_rank = atr.rolling(89).rank(pct=True)
        # 抑制系数: Rank < 0.1 -> 0, Rank > 0.1 -> 1
        suppression = np.where(atr_rank < 0.1, 0.0, 1.0)
        final_dynamics = raw_dynamics * suppression
        if is_debug and probe_ts:
            self._probe_val("SM_Slope_Norm", sm_slope_norm.loc[probe_ts], temp_vals, "HighOrder")
            self._probe_val("Price_Jerk_Std", p_jerk_std.loc[probe_ts], temp_vals, "HighOrder")
            self._probe_val("Price_Jerk_Norm", p_jerk_norm.loc[probe_ts], temp_vals, "HighOrder")
            self._probe_val("ATR_Rank", atr_rank.loc[probe_ts], temp_vals, "HighOrder")
            self._probe_val("Final_Dynamics", final_dynamics[idx.get_loc(probe_ts)] if probe_ts in idx else 0, temp_vals, "HighOrder")
        return pd.Series(final_dynamics, index=idx).clip(-1, 1)

    def _calculate_chip_profitability_structure(self, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        """
        【维度一：筹码获利结构】
        核心逻辑：分析获利盘比例与成本集中度的关系。
        关键信号：winner_rate_D (获利比例), cost_50pct_D (平均成本), chip_concentration_ratio_D (集中度)
        """
        # A. 获利盘优势 (Profit Advantage)
        # winner_rate_D: 获利盘比例，通常在0-100之间，归一化到[-1, 1]
        # 假设winner_rate_D > 90%为极强(1.0)，< 10%为极弱(-1.0)
        winner_rate = df['winner_rate_D']
        score_profit = (winner_rate - 50) / 40.0 # 50% -> 0, 90% -> 1, 10% -> -1
        score_profit = score_profit.clip(-1, 1)
        # B. 成本集中度 (Cost Concentration)
        # 集中度越高，筹码结构越稳定，爆发力越强。
        # chip_concentration_ratio_D: 越小越集中？通常0.1表示高度集中。需确认方向。
        # 假设值越小越集中(0.05-0.2为优)，值越大越发散。
        # 使用 1 - concentration 进行正向化
        concentration = df['chip_concentration_ratio_D']
        # 归一化：假设0.1为优(1.0)，0.3为差(0.0)
        score_compactness = (0.25 - concentration) * 5.0
        score_compactness = score_compactness.clip(-1, 1)
        # C. 当前价格相对于平均成本的乖离 (Cost Deviation)
        # (Close - Cost50) / Cost50
        cost_50 = df['cost_50pct_D']
        close = df['close_D']
        cost_bias = (close - cost_50) / (cost_50 + 1e-8)
        # 乖离率在 5%~20% 之间最健康，过大(>30%)有获利回吐风险，过小(<0)为套牢
        score_bias = pd.Series(0.0, index=idx)
        score_bias = np.where(cost_bias > 0.3, 0.5, # 乖离过大，分数降低
                     np.where(cost_bias > 0.05, 1.0, # 健康区间
                     np.where(cost_bias > 0, 0.5, # 微利
                     -0.5))) # 套牢
        score_bias = pd.Series(score_bias, index=idx)
        # 综合计算
        total_score = score_profit * 0.4 + score_compactness * 0.3 + score_bias * 0.3
        if is_debug:
            val_p = score_profit.loc[probe_ts]
            val_c = score_compactness.loc[probe_ts]
            val_b = score_bias.loc[probe_ts]
            raw_w = winner_rate.loc[probe_ts]
            raw_con = concentration.loc[probe_ts]
            self._probe_val("Raw_WinnerRate", raw_w, temp_vals, "ChipStructure")
            self._probe_val("Raw_Concentration", raw_con, temp_vals, "ChipStructure")
            self._probe_val("Score_Profit", val_p, temp_vals, "ChipStructure")
            self._probe_val("Score_Compactness", val_c, temp_vals, "ChipStructure")
            self._probe_val("Score_Bias", val_b, temp_vals, "ChipStructure")
            self._probe_val("Total_Chip_Score", total_score.loc[probe_ts], temp_vals, "ChipStructure")
        return total_score

    def _calculate_chip_profitability_dynamics(self, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        """【V9.5.0 · 筹码获利动力学】
        专属归一化：
        1. 获利盘(Bound): 线性中心化映射 (Linear Centered Mapping)。
        2. 动力学(Unbound): 滚动波动率标准化 (Rolling Volatility Standardization)。
        """
        # 1. 存量状态 (Stock State) - 有界数据 [0, 100]
        winner_rate = self.helper._get_safe_series(df, 'winner_rate_D', 50.0, "winner_rate")
        # 逻辑：50是多空分界，90是极强，10是极弱
        # 公式：(Val - 50) / 40 -> 映射到 [-1, 1] 区间，超过90或低于10截断
        state_score = ((winner_rate - 50) / 40.0).clip(-1, 1)
        # 2. 动力学特征 (Dynamics) - 无界导数
        wr_slope = self.helper._get_safe_series(df, 'SLOPE_13_winner_rate_D', 0.0, "wr_slope")
        wr_accel = self.helper._get_safe_series(df, 'ACCEL_13_winner_rate_D', 0.0, "wr_accel")
        # 专属处理：使用 winner_rate 自身的波动率作为基准
        # 计算 winner_rate 的 34日 滚动标准差，作为变化的"单位"
        wr_volatility = winner_rate.rolling(34).std().replace(0, 1.0)
        # Slope Score = Slope / (Volatility * Sensitivity) -> Tanh
        # 1.5 倍标准差的变化视为显著变化
        slope_norm = np.tanh(wr_slope / (wr_volatility * 1.5))
        # 加速度通常比速度小，敏感度需要降低 (分母减小 或 Sensitivity减小)
        accel_norm = np.tanh(wr_accel / (wr_volatility * 0.5))
        # 3. 结构演变判定 (逻辑不变，使用新归一化值)
        lock_up = (state_score > 0.5) & (slope_norm > -0.2)
        distribution = (state_score > 0.5) & (slope_norm < -0.5)
        accumulation = (state_score < -0.5) & (accel_norm > 0.5)
        # 4. 综合评分
        raw_score = state_score * 0.6 + slope_norm * 0.4
        final_score = raw_score
        final_score = np.where(lock_up, raw_score + 0.3, final_score)
        final_score = np.where(distribution, raw_score - 0.8, final_score)
        final_score = np.where(accumulation, raw_score + 0.5, final_score)
        # 5. 熵值辅助 (Entropy) - [0, 1] 概率分布
        entropy = self.helper._get_safe_series(df, 'chip_entropy_D', 0.5, "entropy")
        # 熵越小越好。使用 Sigmoid 反向映射，聚焦于低熵区域
        # (0.3 - x) * 5 -> 当 x=0.1时(0.2*5=1) -> Sigmoid(1)~0.73
        # 简化：线性反转，假设 [0, 1]
        entropy_bonus = (0.7 - entropy).clip(0, 0.3) # 只奖励低熵 (<0.7)
        final_score = (final_score + entropy_bonus).clip(-1, 1)
        if is_debug and probe_ts:
            self._probe_val("WR_Vol_Base", wr_volatility.loc[probe_ts], temp_vals, "ChipDynamics")
            self._probe_val("WR_State_Sc", state_score.loc[probe_ts], temp_vals, "ChipDynamics")
            self._probe_val("WR_Slope_Sc", slope_norm.loc[probe_ts], temp_vals, "ChipDynamics")
            self._probe_val("WR_Accel_Sc", accel_norm.loc[probe_ts], temp_vals, "ChipDynamics")
            self._probe_val("Chip_Final", final_score[idx.get_loc(probe_ts)] if probe_ts in idx else 0, temp_vals, "ChipDynamics")
        return pd.Series(final_score, index=idx)

    def _calculate_smart_money_game(self, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        """
        【维度二：主力博弈动能】
        核心逻辑：识别聪明钱的净买入行为及日内博弈烈度。
        关键信号：SMART_MONEY_HM_NET_BUY_D (鸿蒙净买), game_intensity_D (博弈烈度), net_mf_amount_D (主力净额)
        """
        # A. 聪明钱净买入 (Smart Money Net Buy)
        # 归一化：需要观察量级。使用滚动最大值进行动态缩放。
        sm_net_buy = df['SMART_MONEY_HM_NET_BUY_D']
        # 使用20日滚动绝对值最大值作为分母
        rolling_max_sm = sm_net_buy.abs().rolling(20, min_periods=1).max() + 1.0
        score_sm = sm_net_buy / rolling_max_sm
        score_sm = score_sm.clip(-1, 1)
        # B. 主力净额与强度 (Main Force Flow)
        mf_net = df['net_mf_amount_D']
        rolling_max_mf = mf_net.abs().rolling(20, min_periods=1).max() + 1.0
        score_mf = mf_net / rolling_max_mf
        score_mf = score_mf.clip(-1, 1)
        # C. 博弈烈度 (Game Intensity)
        # game_intensity_D: 假设值越高表示分歧越大或换手越积极。
        # 高博弈 + 价格上涨 = 强势洗盘/突破 (正向)
        # 高博弈 + 价格下跌 = 恐慌出逃 (负向)
        intensity = df['game_intensity_D']
        pct_change = df['pct_change_D']
        # 归一化强度 [0, 1]
        norm_intensity = (intensity - intensity.rolling(60).min()) / (intensity.rolling(60).max() - intensity.rolling(60).min() + 1e-8)
        norm_intensity = norm_intensity.clip(0, 1)
        # 方向性博弈分数
        score_game = np.sign(pct_change) * norm_intensity
        # D. 机构协同性 (Smart Money Synergy) - 如果有的话，这里用SMART_MONEY_SYNERGY_BUY_D
        synergy = df.get('SMART_MONEY_SYNERGY_BUY_D', pd.Series(0, index=idx))
        score_synergy = synergy.clip(0, 1) # 假设是0/1或概率值
        # 综合计算
        # 聪明钱权重最高，因为它代表最敏锐的资金
        total_score = score_sm * 0.4 + score_mf * 0.3 + score_game * 0.2 + score_synergy * 0.1
        if is_debug:
            self._probe_val("Raw_SM_Net", sm_net_buy.loc[probe_ts], temp_vals, "SmartMoney")
            self._probe_val("Score_SM", score_sm.loc[probe_ts], temp_vals, "SmartMoney")
            self._probe_val("Score_MF", score_mf.loc[probe_ts], temp_vals, "SmartMoney")
            self._probe_val("Raw_GameIntensity", intensity.loc[probe_ts], temp_vals, "SmartMoney")
            self._probe_val("Score_Game", score_game.loc[probe_ts], temp_vals, "SmartMoney")
            self._probe_val("Total_MF_Score", total_score.loc[probe_ts], temp_vals, "SmartMoney")
        return total_score

    def _calculate_smart_money_synergy(self, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        """【V9.1.0 · 聪明钱协同攻击维度】
        检测聪明钱的净流入方向及其攻击协同性。
        关键信号: SMART_MONEY_HM_NET_BUY_D, SMART_MONEY_HM_COORDINATED_ATTACK_D
        """
        # 1. 聪明钱净买入强度
        sm_net_buy = self.helper._get_safe_series(df, 'SMART_MONEY_HM_NET_BUY_D', 0.0, "sm_net_buy")
        # 使用动态缩放，避免绝对值影响
        # 过去20天的最大绝对值作为基准
        rolling_max = sm_net_buy.abs().rolling(20, min_periods=1).max().replace(0, 1.0)
        score_sm_flow = sm_net_buy / rolling_max
        score_sm_flow = score_sm_flow.clip(-1, 1)
        # 2. 协同攻击信号 (通常是0或1的布尔信号，或者概率值)
        # 假设是概率值 0~1
        coord_attack = self.helper._get_safe_series(df, 'SMART_MONEY_HM_COORDINATED_ATTACK_D', 0.0, "coord_attack")
        score_attack = coord_attack * 2 - 1 # 0->-1, 1->1 (假设0是没有攻击，1是攻击)
        # 修正：如果 Attack=0，不应该是-1，而是0。Attack=1是强加分。
        # 重新映射：0 -> 0, 1 -> 1.0
        score_attack = coord_attack
        # 3. 资金一致性
        consistency = self.helper._get_safe_series(df, 'flow_consistency_D', 0.5, "consistency")
        score_consistency = (consistency - 0.5) * 2
        # 4. 主力活跃度 (过滤器)
        activity = self.helper._get_safe_series(df, 'main_force_activity_index_D', 0.0, "activity")
        # 活跃度低时，整个分数的权重应降低
        activity_factor = activity.clip(0, 1)
        # 综合计算
        # 基础分：流量(40%) + 一致性(30%)
        raw_score = score_sm_flow * 0.4 + score_consistency * 0.3
        # 攻击信号作为增强乘数
        # 如果有攻击信号，且方向为正，大幅增强
        attack_boost = np.where((score_attack > 0.5) & (raw_score > 0), 0.5, 0.0)
        final_score = (raw_score + attack_boost) * (0.5 + 0.5 * activity_factor)
        if is_debug and probe_ts:
            self._probe_val("SM_Flow_Score", score_sm_flow.loc[probe_ts], temp_vals, "SmartMoneyDimension")
            self._probe_val("Attack_Signal", score_attack.loc[probe_ts], temp_vals, "SmartMoneyDimension")
            self._probe_val("Consistency_Score", score_consistency.loc[probe_ts], temp_vals, "SmartMoneyDimension")
            self._probe_val("Activity_Factor", activity_factor.loc[probe_ts], temp_vals, "SmartMoneyDimension")
            self._probe_val("SM_Final_Score", final_score.loc[probe_ts], temp_vals, "SmartMoneyDimension")
        return final_score.clip(-1, 1)

    def _calculate_smart_money_hab_dynamics(self, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        """【V9.5.0 · 聪明钱HAB存量博弈】
        专属归一化：
        1. 流量(Flow): 滚动最大幅值缩放 (Rolling Abs Max Scaling)。
        2. 存量(HAB): 历史区间定位 (Stochastic Range Positioning)。
        """
        # 1. 流量 (Flow) - 尖峰厚尾数据
        raw_flow = self.helper._get_safe_series(df, 'SMART_MONEY_HM_NET_BUY_D', 0.0, "sm_flow")
        # 专属处理：为了保留爆发性，不使用Sigmoid压缩，而是使用相对强度
        # 分母：过去21天资金流绝对值的最大值 (避免除0)
        flow_capacity = raw_flow.abs().rolling(21).max().replace(0, 1.0)
        # Flow Score: [-1, 1]，保留了资金突变的尖峰特征
        flow_norm = (raw_flow / flow_capacity).clip(-1, 1)
        # 2. 存量 (HAB) - 累积平滑数据
        # 55日累积
        hab_raw = raw_flow.rolling(window=55, min_periods=13).sum()
        # 专属处理：随机指标逻辑 (KDJ中的K值概念)
        # 计算长周期(89日)内的相对位置
        hab_low = hab_raw.rolling(89, min_periods=55).min()
        hab_high = hab_raw.rolling(89, min_periods=55).max()
        hab_range = (hab_high - hab_low).replace(0, 1.0)
        # Position: 0(最低) ~ 1(最高)
        hab_position = (hab_raw - hab_low) / hab_range
        # 映射到 [-1, 1]: (Pos - 0.5) * 2
        hab_score = (hab_position - 0.5) * 2
        # 3. 构建博弈体制 (Regime Classification)
        # A. 底部吸筹: HAB低 (<-0.4) & Flow强 (>0.3)
        mask_accum = (hab_score < -0.4) & (flow_norm > 0.3)
        # B. 拉升持仓: HAB高 (>0.2) & Flow正 (>0.0)
        mask_markup = (hab_score > 0.2) & (flow_norm > 0.0)
        # C. 高位派发: HAB高 (>0.4) & Flow负 (<-0.2)
        mask_dist = (hab_score > 0.4) & (flow_norm < -0.2)
        # D. 阴跌无主: HAB低 (<-0.4) & Flow负 (<-0.1)
        mask_void = (hab_score < -0.4) & (flow_norm < -0.1)
        # 评分矩阵
        regime_score = pd.Series(0.0, index=idx)
        # 中性区域：Flow为主，HAB为辅
        neutral_score = flow_norm * 0.7 + hab_score * 0.3
        regime_score = regime_score.mask(~(mask_accum|mask_markup|mask_dist|mask_void), neutral_score)
        regime_score = regime_score.mask(mask_accum, 1.0) # 强力买入
        regime_score = regime_score.mask(mask_markup, 0.7) # 持仓
        regime_score = regime_score.mask(mask_dist, -0.9) # 强力卖出
        regime_score = regime_score.mask(mask_void, -0.5) # 回避
        # 4. 协同攻击修正
        coord_attack = self.helper._get_safe_series(df, 'SMART_MONEY_HM_COORDINATED_ATTACK_D', 0.0, "coord_attack")
        # Attack 是概率值 0~1，直接作为乘数增强正向信号
        final_score = np.where(regime_score > 0, regime_score * (1 + 0.4 * coord_attack), regime_score)
        if is_debug and probe_ts:
            self._probe_val("Flow_Capacity", flow_capacity.loc[probe_ts], temp_vals, "SmartMoneyHAB")
            self._probe_val("Flow_Norm", flow_norm.loc[probe_ts], temp_vals, "SmartMoneyHAB")
            self._probe_val("HAB_Position", hab_position.loc[probe_ts], temp_vals, "SmartMoneyHAB")
            self._probe_val("Regime_Score", regime_score.loc[probe_ts], temp_vals, "SmartMoneyHAB")
            self._probe_val("SM_Final", final_score[idx.get_loc(probe_ts)] if probe_ts in idx else 0, temp_vals, "SmartMoneyHAB")
        return pd.Series(final_score, index=idx).clip(-1, 1)

    def _calculate_kinematic_trend(self, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        """
        【维度三：趋势运动学】
        核心逻辑：位置(均线相干) -> 速度(MA_VELOCITY) -> 加速度(MA_ACCELERATION)
        关键信号：MA_COHERENCE_RESONANCE_D, MA_VELOCITY_EMA_55_D, MA_ACCELERATION_EMA_55_D
        """
        # A. 均线相干共振 (Coherence/Structure)
        # 代表趋势的整齐度。假设范围 [0, 1] 或 [0, 100]
        coherence = df['MA_COHERENCE_RESONANCE_D']
        # 归一化到 [0, 1]
        norm_coherence = coherence / (coherence.rolling(60).max() + 1e-8)
        # 如果是多头排列，给予正分；这里假设coherence本身只代表整齐度，不分方向
        # 结合 pct_change 或 ma_5 vs ma_55 判断方向
        ma5 = df['MA_5_D']
        ma55 = df['MA_55_D']
        direction = np.where(ma5 >= ma55, 1.0, -1.0)
        score_structure = norm_coherence * direction
        # B. 速度 (Velocity) - 趋势快慢
        velocity = df['MA_VELOCITY_EMA_55_D']
        # 归一化
        score_velocity = velocity / (velocity.abs().rolling(60).max() + 1e-8)
        score_velocity = score_velocity.clip(-1, 1)
        # C. 加速度 (Acceleration) - 趋势转折预警
        accel = df['MA_ACCELERATION_EMA_55_D']
        score_accel = accel / (accel.abs().rolling(60).max() + 1e-8)
        score_accel = score_accel.clip(-1, 1)
        # D. 趋势强度修正
        # 如果 速度>0 且 加速度>0 -> 趋势加速增强 (Weight ++)
        # 如果 速度>0 且 加速度<0 -> 趋势减速滞涨 (Weight --)
        kinematic_score = score_velocity * 0.5 + score_accel * 0.5
        # 综合计算
        total_score = score_structure * 0.4 + kinematic_score * 0.6
        if is_debug:
            self._probe_val("Raw_Coherence", coherence.loc[probe_ts], temp_vals, "TrendKinematics")
            self._probe_val("Score_Structure", score_structure.loc[probe_ts], temp_vals, "TrendKinematics")
            self._probe_val("Raw_Velocity", velocity.loc[probe_ts], temp_vals, "TrendKinematics")
            self._probe_val("Raw_Accel", accel.loc[probe_ts], temp_vals, "TrendKinematics")
            self._probe_val("Kinematic_Score", kinematic_score.loc[probe_ts], temp_vals, "TrendKinematics")
            self._probe_val("Total_Trend_Score", total_score.loc[probe_ts], temp_vals, "TrendKinematics")
        return total_score

    def _calculate_kinematic_anomaly_clustering(self, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        """【V9.5.0 · 趋势运动与异动聚类】
        专属归一化：
        1. 运动学(Kinematics): 滚动最大绝对偏差缩放 (Rolling MAD-like Scaling)。
        2. 异动聚类(Clustering): 历史分位数相对强度 (Historical Quantile Relative Strength)。
        """
        # 1. 宏观运动学 (Kinematics)
        velocity = self.helper._get_safe_series(df, 'MA_VELOCITY_EMA_55_D', 0.0, "velocity")
        accel = self.helper._get_safe_series(df, 'MA_ACCELERATION_EMA_55_D', 0.0, "accel")
        # 专属处理：趋势速度差异巨大，使用过去55天的最大速度作为分母 (Trend Capacity)
        vel_capacity = velocity.abs().rolling(55).max().replace(0, 0.01) # 避免除0
        acc_capacity = accel.abs().rolling(55).max().replace(0, 0.01)
        # 简单的比率归一化，保留线性特征
        vel_norm = (velocity / vel_capacity).clip(-1, 1)
        acc_norm = (accel / acc_capacity).clip(-1, 1)
        kinematic_score = vel_norm * 0.6 + acc_norm * 0.4
        # 2. 微观异动聚类 (Anomaly Clustering)
        raw_anomaly = self.helper._get_safe_series(df, 'large_order_anomaly_D', 0.0, "anomaly")
        # 5日聚类
        cluster_sum = raw_anomaly.rolling(window=5, min_periods=1).sum()
        # 专属处理：异动是稀疏的，均值无意义。
        # 关注当前聚类强度是否超过了过去21天90%的时间
        # Rolling Quantile 0.95 作为"高强度阈值"
        cluster_high = cluster_sum.rolling(21).quantile(0.95).replace(0, 1.0)
        # Cluster Intensity: >1.0 表示突破性异动
        cluster_intensity = (cluster_sum / cluster_high).clip(0, 1.5) # 允许溢出一点
        # 3. 信号合成
        pct_change = self.helper._get_safe_series(df, 'pct_change_D', 0.0, "pct")
        # 只有当价格变动与异动方向一致时，异动才有效
        # 涨 + 强异动 = 强买入验证 (1.0 * 1.2)
        # 跌 + 强异动 = 强卖出验证 (-1.0 * 1.2)
        # 小涨 + 弱异动 = 0
        anomaly_score = np.sign(pct_change) * cluster_intensity
        # 4. 最终得分
        # 如果 Kinematics 强 (>0.5) 且 Anomaly 验证 (>0.8)，给予 Bonus
        base_score = kinematic_score * 0.6 + anomaly_score * 0.4
        is_resonance = (kinematic_score.abs() > 0.5) & (cluster_intensity > 0.8) & (np.sign(kinematic_score) == np.sign(pct_change))
        final_score = np.where(is_resonance, base_score * 1.3, base_score)
        if is_debug and probe_ts:
            self._probe_val("Vel_Capacity", vel_capacity.loc[probe_ts], temp_vals, "TrendKinematics")
            self._probe_val("Vel_Norm", vel_norm.loc[probe_ts], temp_vals, "TrendKinematics")
            self._probe_val("Cluster_High_Q95", cluster_high.loc[probe_ts], temp_vals, "TrendKinematics")
            self._probe_val("Cluster_Intensity", cluster_intensity.loc[probe_ts], temp_vals, "TrendKinematics")
            self._probe_val("Trend_Final", final_score[idx.get_loc(probe_ts)] if probe_ts in idx else 0, temp_vals, "TrendKinematics")
        return pd.Series(final_score, index=idx).clip(-1, 1)

    def _calculate_multi_dimension_resonance(self, chip: pd.Series, sm: pd.Series, trend: pd.Series, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        """【V9.1.0 · 多维共振融合与环境调制】
        融合三大核心维度，并根据市场情绪环境进行动态调制。
        """
        # 1. 核心维度加权
        # 聪明钱 > 趋势 > 筹码
        raw_resonance = sm * 0.4 + trend * 0.35 + chip * 0.25
        # 2. 环境因子获取
        sentiment = self.helper._get_safe_series(df, 'market_sentiment_score_D', 0.5, "sentiment")
        panic = self.helper._get_safe_series(df, 'panic_selling_cascade_D', 0.0, "panic")
        # 3. 动态调制逻辑
        # A. 情绪乘数 (Sentiment Multiplier)
        # 情绪在 0.4~0.7 之间是良性的，过高(>0.8)可能过热，过低(<0.2)冰点
        # 设计一个抛物线函数，中间高两头低？或者顺势放大？
        # 这里采用顺势放大但抑制极值：Sentiment 0.5 -> 1.0, 0.8 -> 1.2, 0.2 -> 0.8
        sentiment_mod = 0.8 + sentiment * 0.4
        # B. 恐慌抑制 (Panic Suppression)
        # 如果恐慌指数 > 0.6，大幅削弱做多信号，增强做空信号(如果有)
        # panic 0 -> 1.0, panic 1 -> 0.5
        panic_mod = 1.0 - (panic * 0.5)
        # 4. 最终合成
        final_score = raw_resonance * sentiment_mod * panic_mod
        if is_debug and probe_ts:
            self._probe_val("Raw_Resonance", raw_resonance.loc[probe_ts], temp_vals, "Resonance")
            self._probe_val("Sentiment_Mod", sentiment_mod.loc[probe_ts], temp_vals, "Resonance")
            self._probe_val("Panic_Mod", panic_mod.loc[probe_ts], temp_vals, "Resonance")
            self._probe_val("Final_Total_Score", final_score.loc[probe_ts], temp_vals, "Resonance")
        return final_score.clip(-1, 1)

    def _synthesize_resonance(self, chip: pd.Series, mf: pd.Series, trend: pd.Series, df: pd.DataFrame, idx: pd.Index, is_debug: bool, probe_ts: pd.Timestamp, temp_vals: Dict) -> pd.Series:
        """
        【多维共振融合】
        核心逻辑：加权融合三个维度，并使用市场情绪进行最终调制。
        """
        # 1. 基础加权
        # 趋势决定方向，资金决定力度，筹码决定阻力
        base_score = trend * 0.4 + mf * 0.35 + chip * 0.25
        # 2. 市场情绪调制 (Sentiment Modulation)
        sentiment = df.get('market_sentiment_score_D', pd.Series(0.5, index=idx))
        # 将情绪归一化到 [0.8, 1.2] 放大系数
        # 假设原始情绪 [0, 1]
        sentiment_mod = 0.8 + (sentiment * 0.4)
        modulated_score = base_score * sentiment_mod
        # 3. 极值处理 (Extremes)
        final_score = modulated_score.clip(-1, 1)
        if is_debug:
            self._probe_val("Base_Score", base_score.loc[probe_ts], temp_vals, "Synthesis")
            self._probe_val("Sentiment_Mod", sentiment_mod.loc[probe_ts], temp_vals, "Synthesis")
            self._probe_val("Final_Score", final_score.loc[probe_ts], temp_vals, "Synthesis")
        return final_score

    def _log_debug_info(self, debug_output: Dict, temp_vals: Dict, probe_ts: pd.Timestamp):
        """格式化输出调试信息"""
        print(f"\n====== {probe_ts.strftime('%Y-%m-%d')} 成本优势趋势诊断报告 ======")
        for section, data in temp_vals.items():
            print(f"[{section}]")
            for k, v in data.items():
                print(f"  {k:<20}: {v:.4f}")
            print("-" * 30)
        print("===================================================\n")

    def _get_required_signals_list(self, mtf_slope_accel_weights: Dict) -> List[str]:
        """【V9.4.0 · 军械库信号依赖清单】
        新增 Winner Rate 的 Slope/Accel 信号，以及异动 HAB 计算依赖。
        """
        required_signals = [
            # 基础维度
            'winner_rate_D', 'chip_entropy_D', 'chip_stability_D', 'cost_50pct_D',
            'SMART_MONEY_HM_NET_BUY_D', 'SMART_MONEY_HM_COORDINATED_ATTACK_D',
            'flow_consistency_D', 'main_force_activity_index_D',
            'MA_VELOCITY_EMA_55_D', 'MA_ACCELERATION_EMA_55_D',
            'MA_COHERENCE_RESONANCE_D', 'large_order_anomaly_D', 'pct_change_D',
            'market_sentiment_score_D', 'panic_selling_cascade_D', 'ATR_14_D',
            'close_D',
            # 筹码动力学 (Lookback = 13)
            'SLOPE_13_winner_rate_D',
            'ACCEL_13_winner_rate_D',
            # 资金动力学 (Lookback = 13)
            'SLOPE_13_SMART_MONEY_HM_NET_BUY_D',
            'ACCEL_13_SMART_MONEY_HM_NET_BUY_D',
            # 价格动力学
            'ACCEL_13_close_D',
            'JERK_13_close_D'
        ]
        return required_signals

    def _check_and_repair_signals(self, df: pd.DataFrame, method_name: str) -> pd.DataFrame:
        """
        【V9.0 信号检查与基础修复】
        仅做最小限度的NaN填充，避免破坏数据真实性
        """
        # 关键列前向填充
        critical_cols = self._get_required_signals_list({})
        for col in critical_cols:
            if col in df.columns:
                df[col] = df[col].ffill().fillna(0)
            else:
                # 缺失列创建全0，并在后续逻辑中被动处理
                df[col] = 0.0
        return df