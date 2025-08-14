# 文件: strategies/trend_following/intelligence/chip_intelligence.py
# 筹码情报模块
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following.utils import get_params_block, get_param_value

class ChipIntelligence:
    def __init__(self, strategy_instance, dynamic_thresholds: Dict):
        """
        初始化筹码情报模块。
        :param strategy_instance: 策略主实例的引用。
        :param dynamic_thresholds: 动态阈值字典。
        """
        self.strategy = strategy_instance
        self.dynamic_thresholds = dynamic_thresholds

    def run_chip_intelligence_command(self, df: pd.DataFrame) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
        """
        【V320.1 数据驱动重构版】筹码情报最高司令部
        - 核心重构: 根据数据层提供的预计算列清单，本方法已完全重构为直接消费这些高性能特征。
                    不再进行任何动态的斜率或加速度计算，显著提升了性能和代码的健壮性。
        """
        print("        -> [筹码情报最高司令部 V320.1 数据驱动重构版] 启动...")
        states = {}
        triggers = {}
        default_series = pd.Series(False, index=df.index)

        p = get_params_block(self.strategy, 'chip_feature_params')
        if not get_param_value(p.get('enabled'), False):
            return states, triggers

        # --- 步骤 1: 校验数据层是否提供了所有必需的预计算列 ---
        required_cols = [
            'concentration_90pct_D',             # 90%筹码集中度 (静态值)
            'SLOPE_5_concentration_90pct_D',     # 5日筹码集中度斜率 (速度)
            'ACCEL_5_concentration_90pct_D',     # 5日筹码集中度加速度
            'ACCEL_21_concentration_90pct_D',    # 21日筹码集中度加速度 (用于风险预警)
            'peak_cost_D',                       # 成本峰价格
            'SLOPE_5_peak_cost_D',               # 5日成本峰斜率
            'ACCEL_5_peak_cost_D'               # 5日成本峰加速度 (用于点火信号)
        ]
        # 如果任一必需列在DataFrame中不存在，则打印警告并安全退出，保证策略不会因数据缺失而崩溃。
        if any(col not in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            print(f"          -> [警告] 缺少筹码分级诊断所需列，跳过。缺失: {missing_cols}")
            return states, triggers

        # --- 步骤 2: 为常用列创建更简洁的变量名，提高代码可读性 ---
        conc_col = 'concentration_90pct_D'
        conc_slope_col = 'SLOPE_5_concentration_90pct_D'
        conc_accel_col = 'ACCEL_5_concentration_90pct_D'
        conc_accel_21d_col = 'ACCEL_21_concentration_90pct_D'
        cost_slope_col = 'SLOPE_5_peak_cost_D'

        # --- 步骤 3: 动态阈值计算 (保留，因为这是策略自适应逻辑的一部分) ---
        p_struct = p.get('structure_params', {})
        steady_gathering_quantile = get_param_value(p_struct.get('steady_gathering_quantile'), 0.40)
        accel_gathering_quantile = get_param_value(p_struct.get('accel_gathering_quantile'), 0.15)
        # 使用滚动分位数计算动态阈值，以适应不同市况和个股波动性
        steady_threshold = df[conc_slope_col].rolling(window=120, min_periods=20).quantile(steady_gathering_quantile)
        accel_threshold = df[conc_slope_col].rolling(window=120, min_periods=20).quantile(accel_gathering_quantile)

        # --- 步骤 4: C/B/B+ 级动态过程诊断 (直接使用预计算列) ---
        # C级: 筹码稳步聚集。直接使用预计算的 'SLOPE_5_concentration_90pct_D' 列进行判断。
        is_steady_gathering = (df[conc_slope_col] < steady_threshold) & (df[conc_slope_col] >= accel_threshold)
        states['CHIP_CONC_STEADY_GATHERING_C'] = is_steady_gathering

        # B级: 筹码加速聚集。直接使用预计算的 'SLOPE_5_concentration_90pct_D' 列进行判断。
        is_accelerated_gathering = df[conc_slope_col] < accel_threshold
        states['CHIP_CONC_ACCELERATED_GATHERING_B'] = is_accelerated_gathering

        # B+级: 筹码聚集强化。直接使用预计算的 'ACCEL_5_concentration_90pct_D' 列。
        # 加速度为负，代表集中趋势在强化 (因为斜率本身是负的，变得更负，所以其导数/加速度也为负)。
        is_intensifying = df[conc_accel_col] < 0
        states['CHIP_CONC_INTENSIFYING_B_PLUS'] = is_accelerated_gathering & is_intensifying
        # if states['CHIP_CONC_INTENSIFYING_B_PLUS'].any():
            # print(f"            -> [情报] 侦测到 {states['CHIP_CONC_INTENSIFYING_B_PLUS'].sum()} 次 B+级“筹码聚集强化”战术信号！")

        # --- 步骤 5: A/S 级静态结果与复合机会诊断 (直接使用预计算列) ---
       # --- A级动态阈值 (新增) ---
        locked_conc_quantile = get_param_value(p_struct.get('locked_concentration_quantile'), 0.10) # 筹码集中度必须优于过去90%的时间
        cost_stability_quantile = get_param_value(p_struct.get('cost_stability_quantile'), 0.20) # 成本波动必须小于过去80%的时间
        
        # 计算动态阈值
        locked_conc_threshold = df[conc_col].rolling(window=120, min_periods=20).quantile(locked_conc_quantile)
        cost_stability_threshold = df[cost_slope_col].abs().rolling(window=120, min_periods=20).quantile(cost_stability_quantile)

        # --- A/S 级静态结果与复合机会诊断 (使用新阈值) ---
        # A级: 筹码锁定稳定。
        is_highly_concentrated_dynamic = df[conc_col] < locked_conc_threshold
        is_cost_peak_stable_dynamic = df[cost_slope_col].abs() < cost_stability_threshold
        states['CHIP_CONC_LOCKED_AND_STABLE_A'] = is_highly_concentrated_dynamic & is_cost_peak_stable_dynamic
        
        # S级: 筹码锁仓突破 (逻辑不变，但基础更可靠)
        is_breakout_candle = self.strategy.atomic_states.get('TRIGGER_BREAKOUT_CANDLE', default_series)
        states['OPP_CHIP_LOCKED_BREAKOUT_S'] = states['CHIP_CONC_LOCKED_AND_STABLE_A'] & is_breakout_candle
        
        # if states['CHIP_CONC_LOCKED_AND_STABLE_A'].any():
        #     print(f"            -> [情报] 侦测到 {states['CHIP_CONC_LOCKED_AND_STABLE_A'].sum()} 次 A级“筹码锁定稳定”机会！")
        # if states['OPP_CHIP_LOCKED_BREAKOUT_S'].any():
        #     print(f"            -> [情报] 侦测到 {states['OPP_CHIP_LOCKED_BREAKOUT_S'].sum()} 次 S级“筹码锁仓突破”王牌机会！")

        # --- 步骤 6: 独立触发器与风险诊断 (直接使用预计算列) ---
        # 点火触发器: 直接使用预计算的 'ACCEL_5_peak_cost_D' 列。
        p_ignition = p.get('ignition_params', {})
        if get_param_value(p_ignition.get('enabled'), True):
            accel_threshold_ignition = get_param_value(p_ignition.get('accel_threshold'), 0.01)
            triggers['TRIGGER_CHIP_IGNITION'] = df.get('ACCEL_5_peak_cost_D', 0) > accel_threshold_ignition

        # 风险诊断1: 长期派发风险
        is_in_high_level_zone = self.strategy.atomic_states.get('CONTEXT_RISK_HIGH_LEVEL_ZONE', default_series)
        worsening_threshold = 1.05
        concentration_21d_ago = df[conc_col].shift(21)
        is_concentration_worsened = df[conc_col] > (concentration_21d_ago * worsening_threshold)
        states['RISK_CONTEXT_LONG_TERM_DISTRIBUTION'] = is_concentration_worsened & is_in_high_level_zone

        # 风险诊断2: 筹码集中趋势恶化拐点。直接使用预计算的 'ACCEL_21_concentration_90pct_D' 列。
        is_worsening_turn = (df[conc_accel_21d_col] > 0) & (df[conc_accel_21d_col].shift(1) <= 0)
        states['RISK_CHIP_CONC_ACCEL_WORSENING'] = is_worsening_turn
        # if is_worsening_turn.any():
        #     print(f"            -> [风险] 侦测到 {is_worsening_turn.sum()} 次“筹码集中趋势恶化”拐点！")
        
        # 整合所有筹码层面的风险信号，形成最终的系统性风险判断
        print("          -> [复合风险合成] 正在整合所有筹码层面的风险信号...")
        chip_risk_1 = self.strategy.atomic_states.get('CHIP_DYN_DIVERGING', default_series)
        chip_risk_2 = self.strategy.atomic_states.get('CHIP_DYN_COST_FALLING', default_series)
        chip_risk_3 = self.strategy.atomic_states.get('CHIP_DYN_WINNER_RATE_COLLAPSING', default_series)
        chip_risk_4 = states.get('RISK_CONTEXT_LONG_TERM_DISTRIBUTION', default_series)
        chip_risk_5 = states.get('RISK_CHIP_CONC_ACCEL_WORSENING', default_series)
        chip_risk_6 = self.strategy.atomic_states.get('RISK_BEHAVIOR_PANIC_FLEEING_S', default_series)
        is_chip_structure_unhealthy = (chip_risk_1 | chip_risk_2 | chip_risk_3 | chip_risk_4 | chip_risk_5 | chip_risk_6)
        states['RISK_CHIP_STRUCTURE_CRITICAL_FAILURE'] = is_chip_structure_unhealthy
        # if is_chip_structure_unhealthy.any():
        #     print(f"            -> [系统风险] 侦测到 {is_chip_structure_unhealthy.sum()} 次“筹码结构严重失效”！")
        
        print("        -> [筹码情报最高司令部 V320.1 数据驱动重构版] 分析完毕。")
        return states, triggers


    def diagnose_dynamic_chip_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V284.0 灵魂注入版】
        - 核心重构: 引入“成本试金石”原则。现在，所有“筹码集中”的判断，都必须
                    满足“成本峰稳定或抬高”这一前置条件。
        - 收益: 从根本上解决了“崩盘式集中”的逻辑陷阱，能够精确区分“建设性吸筹”
                与“毁灭性套牢”，极大提升了筹码分析的可靠性。
        """
        states = {}
        default_series = pd.Series(False, index=df.index)
        base_required_cols = ['concentration_90pct_D', 'peak_cost_D', 'total_winner_rate_D', 'chip_health_score_D']
        base_missing_cols = [col for col in base_required_cols if col not in df.columns]
        if base_missing_cols:
            print(f"            -> [严重警告] 动态筹码分析中心缺少最基础的静态筹码数据: {base_missing_cols}，模块已完全跳过！")
            return states
            
        required_cols = [
            'SLOPE_5_concentration_90pct_D', 'ACCEL_5_concentration_90pct_D',
            'SLOPE_5_peak_cost_D', 'ACCEL_5_peak_cost_D',
            'SLOPE_5_total_winner_rate_D', 'ACCEL_5_total_winner_rate_D',
            'SLOPE_5_chip_health_score_D', 'ACCEL_5_chip_health_score_D'
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"            -> [严重警告] 动态分析中心缺少关键的斜率/加速度数据: {missing_cols}，模块已跳过！")
            return states
            
        is_in_high_level_zone = self.strategy.atomic_states.get('CONTEXT_RISK_HIGH_LEVEL_ZONE', default_series)
        
        # --- 步骤2: 对“筹码集中度”进行动态分析 ---
        # [修改原因] 引入“成本试金石”，区分真假筹码集中。
        # 2.1 定义客观的集中趋势
        is_concentrating_trend = df['SLOPE_5_concentration_90pct_D'] < 0
        
        # 2.2 定义“成本试金石”：成本峰必须稳定或抬高
        # 我们允许非常微小的负斜率，以容忍正常波动
        cost_stability_tolerance = -0.001 
        is_cost_constructive = df['SLOPE_5_peak_cost_D'] >= cost_stability_tolerance
        
        # 2.3 定义“绿色通道”豁免条件
        is_washout_absorption = self.strategy.atomic_states.get('OPP_CONSTRUCTIVE_WASHOUT_ABSORPTION_A', default_series)
        
        # 2.4 最终的“建设性筹码集中” = (常规集中) 或 (豁免的特殊集中)
        states['CHIP_DYN_CONCENTRATING'] = (is_concentrating_trend & is_cost_constructive) | is_washout_absorption

        # 2.4 加速动态 (现在自动继承了“建设性”的前提)
        p_chip = get_params_block(self.strategy, 'chip_feature_params')
        accel_threshold = get_param_value(p_chip.get('accel_concentration_threshold'), -0.001)
        is_accelerating_action = df['ACCEL_5_concentration_90pct_D'] < accel_threshold
        states['CHIP_DYN_S_ACCEL_CONCENTRATING'] = states['CHIP_DYN_CONCENTRATING'] & is_accelerating_action
        
        # 2.5 发散动态 (逻辑不变，因为发散总是坏事)
        is_objective_diverging_action = df['SLOPE_5_concentration_90pct_D'] > 0
        states['CHIP_DYN_OBJECTIVE_DIVERGING'] = is_objective_diverging_action
        states['CHIP_DYN_DIVERGING'] = is_objective_diverging_action & is_in_high_level_zone
        is_accel_diverging_action = df['ACCEL_5_concentration_90pct_D'] > 0
        states['CHIP_DYN_OBJECTIVE_ACCEL_DIVERGING'] = is_accel_diverging_action
        states['CHIP_DYN_ACCEL_DIVERGING'] = is_accel_diverging_action & is_in_high_level_zone
        
        # --- 步骤3: 对“筹码成本”进行动态分析  ---
        states['CHIP_DYN_COST_RISING'] = df['SLOPE_5_peak_cost_D'] > 0
        cost_accel_threshold = self.dynamic_thresholds.get('cost_accel_significant', 0.01)
        states['CHIP_DYN_COST_ACCELERATING'] = df['ACCEL_5_peak_cost_D'] > cost_accel_threshold
        states['CHIP_DYN_COST_FALLING'] = df['SLOPE_5_peak_cost_D'] < 0
        
        # --- 步骤4 & 5  ---
        winner_rate_collapse_threshold = -1.0
        states['CHIP_DYN_WINNER_RATE_COLLAPSING'] = df['SLOPE_5_total_winner_rate_D'] < winner_rate_collapse_threshold
        states['CHIP_DYN_WINNER_RATE_ACCEL_COLLAPSING'] = df['ACCEL_5_total_winner_rate_D'] < 0
        states['CHIP_DYN_HEALTH_IMPROVING'] = df['SLOPE_5_chip_health_score_D'] > 0
        states['CHIP_DYN_HEALTH_DETERIORATING'] = df['SLOPE_5_chip_health_score_D'] < 0
        
        return states

    def diagnose_chip_opportunities(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V341.0 新增】高级筹码“机会”情报诊断模块
        - 核心职责: 识别由高级筹码指标揭示的、结构性的看涨机会。
        """
        # print("        -> [高级筹码机会诊断模块 V341.0] 启动...")
        states = {}
        # --- 机会1: S级 - 筹码断层新生 (结构性重置) ---
        fault_formed_col = 'is_chip_fault_formed_D'
        if fault_formed_col in df.columns:
            states['OPP_CHIP_FAULT_REBIRTH_S'] = df[fault_formed_col]
            # if df[fault_formed_col].any():
            #     print(f"          -> [情报] 侦测到 {df[fault_formed_col].sum()} 次 S级“筹码断层新生”机会！")
        # --- 机会2: A级 - 高利润安全垫 (持股心态稳定) ---
        profit_margin_col = 'winner_profit_margin_D'
        if profit_margin_col in df.columns:
            # 定义：获利盘的平均利润超过20%，代表持股心态极其稳定
            states['CHIP_STATE_HIGH_PROFIT_CUSHION'] = df[profit_margin_col] > 20.0
        # --- 机会3: 获利盘持续上升 (市场情绪积极) ---
        winner_rate_slope_col = 'SLOPE_5_total_winner_rate_D'
        if winner_rate_slope_col in df.columns:
            states['CHIP_DYN_WINNER_RATE_RISING'] = df[winner_rate_slope_col] > 0
        return states


    def diagnose_chip_risks_and_behaviors(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V341.2 加速度增强版】高级筹码“风险与行为”情报诊断模块
        - 核心升级: 引入 turnover_from_winners_ratio_D 的加速度判断，
                    新增 S级“恐慌加速”风险 和 A级“卖盘衰竭”机会，
                    使风险评估体系具备了预测趋势拐点的能力。
        """
        # print("        -> [高级筹码风险与行为诊断模块 V341.2 加速度增强版] 启动...")
        states = {}
        default_series = pd.Series(False, index=df.index)

        # --- 1. 军备检查 ---
        # 检查基础指标、斜率和加速度指标
        required_cols = [
            'turnover_from_winners_ratio_D', 'turnover_from_losers_ratio_D', 'pct_change_D',
            'SLOPE_5_turnover_from_winners_ratio_D', 'ACCEL_5_turnover_from_winners_ratio_D',
            'SLOPE_21_turnover_from_winners_ratio_D', 'ACCEL_21_turnover_from_winners_ratio_D',
            'SLOPE_55_turnover_from_winners_ratio_D', 'ACCEL_55_turnover_from_winners_ratio_D'
        ]
        if any(c not in df.columns for c in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            print(f"          -> [警告] 缺少成交量微观结构或其斜率/加速度数据，模块跳过。缺失: {missing_cols}")
            return {}
            
        is_in_high_zone = self.strategy.atomic_states.get('CONTEXT_RISK_HIGH_LEVEL_ZONE', default_series)

        # --- 风险行为1: 获利盘出逃 (分层动态评估) ---
        # 1.1 获取各周期的抛压趋势 (速度)
        is_fleeing_short_term = df['SLOPE_5_turnover_from_winners_ratio_D'] > 0
        is_fleeing_mid_term = df['SLOPE_21_turnover_from_winners_ratio_D'] > 0
        is_fleeing_long_term = df['SLOPE_55_turnover_from_winners_ratio_D'] > 0

        # 1.2 获取抛压趋势的变化 (加速度) - 我们最关心短期的变化
        is_fleeing_accelerating = df['ACCEL_5_turnover_from_winners_ratio_D'] > 0
        is_fleeing_decelerating = df['ACCEL_5_turnover_from_winners_ratio_D'] < 0

        # 1.3 定义风险等级 (基于速度)
        states['RISK_BEHAVIOR_WINNERS_FLEEING_C'] = is_in_high_zone & is_fleeing_short_term
        states['RISK_BEHAVIOR_WINNERS_FLEEING_B'] = is_in_high_zone & is_fleeing_short_term & is_fleeing_mid_term
        states['RISK_BEHAVIOR_WINNERS_FLEEING_A'] = is_in_high_zone & is_fleeing_short_term & is_fleeing_mid_term & is_fleeing_long_term

        # 1.4 【新增】定义S级风险和A级机会 (基于加速度)
        # S级风险 - 恐慌加速: 中期趋势已在出逃，且短期出逃正在加速，这是最危险的信号。
        states['RISK_BEHAVIOR_PANIC_FLEEING_S'] = states['RISK_BEHAVIOR_WINNERS_FLEEING_B'] & is_fleeing_accelerating
        
        # A级机会 - 卖盘衰竭: 获利盘虽然还在卖(短期斜率>0)，但卖出力度已在减弱(加速度<0)。
        states['OPP_BEHAVIOR_SELLING_EXHAUSTION_A'] = is_fleeing_short_term & is_fleeing_decelerating
        
        # --- 机会行为2: 恐慌盘割肉 (底部反向指标) ---
        # 定义：股价当日大跌（例如超过5%），且成交量主要由套牢盘贡献（例如占比超过50%）
        is_sharp_drop = df['pct_change_D'] < -0.05
        is_panic_selling = df['turnover_from_losers_ratio_D'] > 50.0
        
        states['OPP_BEHAVIOR_PANIC_CAPITULATION_A'] = is_sharp_drop & is_panic_selling

        # 打印情报
        # if states['RISK_BEHAVIOR_PANIC_FLEEING_S'].any():
        #     print(f"          -> [S级战略风险] 侦测到 {states['RISK_BEHAVIOR_PANIC_FLEEING_S'].sum()} 次“获利盘恐慌加速出逃”！")
        # elif states['RISK_BEHAVIOR_WINNERS_FLEEING_A'].any():
        #     print(f"          -> [A级战略风险] 侦测到 {states['RISK_BEHAVIOR_WINNERS_FLEEING_A'].sum()} 次“长期派发”共振！")
        
        # if states['OPP_BEHAVIOR_SELLING_EXHAUSTION_A'].any():
        #     print(f"          -> [A级机会情报] 侦测到 {states['OPP_BEHAVIOR_SELLING_EXHAUSTION_A'].sum()} 次“卖盘衰竭”信号！")

        # --- 风险行为2: 恐慌盘割肉 (加速赶底) ---
        is_sharp_drop = df['pct_change_D'] < -0.05
        is_panic_selling = df['turnover_from_losers_ratio_D'] > 50.0
        states['RISK_BEHAVIOR_PANIC_SELLING'] = is_sharp_drop & is_panic_selling
        # if states['RISK_BEHAVIOR_PANIC_SELLING'].any():
        #     print(f"          -> [机会情报] 侦测到 {states['RISK_BEHAVIOR_PANIC_SELLING'].sum()} 次“恐慌盘割肉”行为(可能见底)！")
            
        return states


    def diagnose_peak_formation_dynamics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V340.0 新增】筹码峰“创世纪”模块
        - 核心职责: 追踪主筹码峰的“政权更迭”，并为其关联上“出生证明”
                    (确立之日的成交量特征)，从而解读其战略意义。
        """
        # print("        -> [筹码峰“创世纪”模块 V340.0] 启动...")
        states = {}
        
        # --- 1. 检查所需数据 ---
        required_cols = ['peak_cost_D', 'volume_D', 'VOL_MA_21_D']
        if any(c not in df.columns for c in required_cols):
            print("          -> [警告] 缺少诊断筹码峰起源所需数据，模块跳过。")
            return {}

        # --- 2. 识别“政权更迭”事件 ---
        # 定义：主峰成本与昨日相比，变化超过1.5%
        is_peak_changed = (df['peak_cost_D'].pct_change().abs() > 0.015)
        
        # --- 3. 状态机：追踪并确认新的主峰 ---
        df['formation_date'] = np.nan
        df['formation_volume_ratio'] = np.nan
        
        in_observation = False
        observation_start_idx = -1
        stability_period = 3 # 需要稳定3天

        for i in range(1, len(df)):
            if is_peak_changed.iloc[i] and not in_observation:
                # 发现潜在的更迭事件，开始观察
                in_observation = True
                observation_start_idx = i
            
            if in_observation:
                # 检查自观察开始以来，主峰是否保持稳定
                observation_window = df['peak_cost_D'].iloc[observation_start_idx : i+1]
                is_stable = (observation_window.std() / observation_window.mean()) < 0.01
                
                if not is_stable:
                    # 如果不稳定，重置观察
                    in_observation = False
                elif (i - observation_start_idx + 1) >= stability_period:
                    # 如果已稳定达到N天，则确认“政权”
                    formation_date = df.index[observation_start_idx]
                    formation_volume_ratio = df.at[formation_date, 'volume_D'] / df.at[formation_date, 'VOL_MA_21_D']
                    
                    # 将“出生证明”赋予从确立日到今天的所有记录
                    for j in range(observation_start_idx, i + 1):
                        df.at[df.index[j], 'formation_date'] = formation_date
                        df.at[df.index[j], 'formation_volume_ratio'] = formation_volume_ratio
                    
                    # 结束本次观察
                    in_observation = False

        # --- 4. 解读“出生证明”，生成战略信号 ---
        # 条件A: 高量形成 (成交量是均量的2倍以上)
        is_high_volume_formation = df['formation_volume_ratio'] > 2.0
        # 条件B: 缩量形成 (成交量低于均量的70%)
        is_low_volume_formation = df['formation_volume_ratio'] < 0.7
        
        # 条件C: 形成于下跌/盘整后 (用长期均线斜率判断)
        is_after_downtrend = df['SLOPE_55_EMA_55_D'].shift(1) <= 0
        # 条件D: 形成于上涨后
        is_after_uptrend = df['SLOPE_55_EMA_55_D'].shift(1) > 0

        # 组合生成最终的原子状态
        states['PEAK_DYN_FORTRESS_SUPPORT'] = is_high_volume_formation & is_after_downtrend
        states['PEAK_DYN_EXHAUSTION_TOP'] = is_high_volume_formation & is_after_uptrend
        states['PEAK_DYN_STEALTH_ACCUMULATION'] = is_low_volume_formation & is_after_downtrend

        return states


    def diagnose_peak_battle_dynamics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V509.0 生产就绪版】主峰攻防战诊断模块
        - 核心逻辑: 采用“准备日+确认日”的两步时序逻辑，识别在主峰激烈换手后得到确认的突破机会。
                    同时，识别在高位区域发生的、有派发嫌疑的放量滞涨风险。
        - 状态: 已移除所有调试探针，代码已净化。
        """
        states = {}
        default_series = pd.Series(False, index=df.index)

        # --- 1. 军备检查 ---
        required_cols = [
            'turnover_at_peak_ratio_D', 'price_to_peak_ratio_D', 'pct_change_D',
            'volume_D', 'VOL_MA_21_D', 'close_D', 'EMA_55_D'
        ]
        if any(c not in df.columns for c in required_cols):
            return {}
        
        # --- 2. 定义参数 ---
        high_battle_threshold = 40.0
        proximity_threshold = 0.03
        volume_multiplier = 1.5
        meaningful_rise_pct = 0.02
        confirmation_lookback_days = 3

        # --- 3. 定义基础条件 ---
        is_battle_intense = df['turnover_at_peak_ratio_D'] > high_battle_threshold
        is_price_at_peak = df['price_to_peak_ratio_D'].between(1 - proximity_threshold, 1 + proximity_threshold)
        is_high_volume = df['volume_D'] > (df['VOL_MA_21_D'] * volume_multiplier)
        is_constructive_context = df['close_D'] > df['EMA_55_D']
        is_in_high_zone = self.strategy.atomic_states.get('CONTEXT_RISK_HIGH_LEVEL_ZONE', default_series)
        is_price_rising_meaningfully = df['pct_change_D'] > meaningful_rise_pct
        is_price_stagnant_or_falling = df['pct_change_D'] < 0.01
        did_not_collapse = df['pct_change_D'] > -0.03

        # --- 4. 机会信号的两步确认 ---
        peak_battle_setup = (
            is_constructive_context &
            is_price_at_peak &
            is_battle_intense &
            is_high_volume &
            did_not_collapse
        )
        final_opportunity_signal = pd.Series(False, index=df.index)
        for i in range(1, confirmation_lookback_days + 1):
            is_recent_setup = peak_battle_setup.shift(i).fillna(False)
            final_opportunity_signal |= (is_price_rising_meaningfully & is_recent_setup)
        
        states['OPP_PEAK_BATTLE_BREAKOUT_A'] = final_opportunity_signal

        # --- 5. 风险信号的定义 ---
        risk_signal = (
            is_in_high_zone &
            is_price_at_peak &
            is_battle_intense &
            is_high_volume &
            is_price_stagnant_or_falling
        )
        states['RISK_PEAK_BATTLE_DISTRIBUTION_A'] = risk_signal

        # (可选) 保留最终的情报报告，作为策略的正常日志输出
        # if final_opportunity_signal.any():
        #     print(f"          -> [A+级机会情报] 侦测到 {final_opportunity_signal.sum()} 次“主峰换手突破(确认后)”！")
        # if risk_signal.any():
        #     print(f"          -> [A级风险情报] 侦测到 {risk_signal.sum()} 次“主峰高位派发嫌疑”！")

        return states


    def diagnose_chip_price_divergence(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】筹码价格顶背离诊断模块
        - 核心职责: 识别“价格创近期新高，但筹码结构却在同步恶化”的经典顶部风险信号。
        """
        states = {}
        default_series = pd.Series(False, index=df.index)
        
        # 1. 定义“价格创近期新高”
        is_new_high = df['close_D'] > df['close_D'].shift(1).rolling(window=20).max()
        
        # 2. 定义“筹码结构恶化”
        #    我们使用“筹码正在发散”作为最核心的恶化指标
        is_chip_diverging = self.strategy.atomic_states.get('CHIP_DYN_DIVERGING', default_series)
        
        # 3. 组合成最终的顶背离风险信号
        states['RISK_CHIP_PRICE_DIVERGENCE'] = is_new_high & is_chip_diverging
        return states

