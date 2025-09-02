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
        - 核心修改: 移除了对资金流的直接依赖，资金流交叉验证逻辑已转移到 CognitiveIntelligence。
        """
        # print("        -> [筹码情报最高司令部 V320.1 数据驱动重构版] 启动...")
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
            'ACCEL_5_peak_cost_D',               # 5日成本峰加速度 (用于点火信号)
            'SLOPE_21_concentration_90pct_D'     # 21日筹码集中度斜率，用于长期派发风险诊断
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
        conc_slope_21d_col = 'SLOPE_21_concentration_90pct_D'

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
        # --- A级动态阈值 ---
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
        # 增加21日筹码集中度斜率为正的条件，直接判断筹码发散趋势
        is_21d_slope_diverging = df[conc_slope_21d_col] > 0
        states['RISK_CONTEXT_LONG_TERM_DISTRIBUTION'] = (is_concentration_worsened | is_21d_slope_diverging) & is_in_high_level_zone

        # 风险诊断2: 筹码集中趋势恶化拐点。直接使用预计算的 'ACCEL_21_concentration_90pct_D' 列。
        is_worsening_turn = (df[conc_accel_21d_col] > 0) & (df[conc_accel_21d_col].shift(1) <= 0)
        states['RISK_CHIP_CONC_ACCEL_WORSENING'] = is_worsening_turn
        # if is_worsening_turn.any():
        #     print(f"            -> [风险] 侦测到 {is_worsening_turn.sum()} 次“筹码集中趋势恶化”拐点！")

        # 风险诊断3: 价格筹码顶背离 (S级风险)
        # 逻辑：价格创近期新高，但筹码集中度却在发散（SLOPE_5_concentration_90pct_D > 0）
        is_price_new_high = df['close_D'] > df['close_D'].rolling(window=20, min_periods=1).max().shift(1)
        is_chip_diverging_short_term = df[conc_slope_col] > 0
        states['RISK_CHIP_PRICE_DIVERGENCE'] = is_price_new_high & is_chip_diverging_short_term & is_in_high_level_zone
        atomic = self.strategy.atomic_states
        # 复合机会 1 (S级): “锁仓点火” - 主力完成锁仓后，开始加速拉升成本，进入主升浪标志。
        # 场景 (静态): 筹码已经锁定且稳定 (CHIP_CONC_LOCKED_AND_STABLE_A)。
        # 剧情 (动态): 成本峰正在“加速”抬升 (CHIP_DYN_COST_ACCELERATING)。
        is_locked_and_stable = states.get('CHIP_CONC_LOCKED_AND_STABLE_A', default_series)
        is_cost_accelerating = atomic.get('CHIP_DYN_COST_ACCELERATING', default_series)
        states['OPP_STRATEGY_LOCKED_FLOAT_IGNITION_S'] = is_locked_and_stable & is_cost_accelerating
        
        # 复合机会 2 (A级): “盘整突破前兆” - 在关键成本区拉锯时，筹码持续被收集，预示向上突破概率大。
        # 场景 (静态): 价格在成本密集区内震荡 (CHIP_STATE_PRICE_IN_PEAK_COST_ZONE)。
        # 剧情 (动态): 筹码发生“共振式”集中 (CHIP_DYN_CONCENTRATING_RESONANCE_A)。
        is_in_battle_zone = atomic.get('CHIP_STATE_PRICE_IN_PEAK_COST_ZONE', default_series)
        is_concentrating_resonance = atomic.get('CHIP_DYN_CONCENTRATING_RESONANCE_A', default_series)
        states['OPP_STRATEGY_BREAKOUT_PRECURSOR_A'] = is_in_battle_zone & is_concentrating_resonance
        
        # 复合风险 1 (S级): “高位派发确认” - 股价脱离成本区很远，且筹码出现系统性派发迹象。
        # 场景 (静态): 价格显著脱离成本区 (CHIP_STATE_PRICE_ABOVE_PEAK_COST)。
        # 剧情 (动态): 筹码发生“共振式”派发 (RISK_CHIP_DIVERGING_RESONANCE_A)。
        is_price_far_above_cost = atomic.get('CHIP_STATE_PRICE_ABOVE_PEAK_COST', default_series)
        is_diverging_resonance = atomic.get('RISK_CHIP_DIVERGING_RESONANCE_A', default_series)
        states['RISK_STRATEGY_DISTRIBUTION_CONFIRMED_S'] = is_price_far_above_cost & is_diverging_resonance
        
        # 复合风险 2 (A级): “主力自救失败” - 价格跌回成本区，但筹码却仍在发散，说明护盘失败。
        # 场景 (静态): 价格在成本密集区内震荡 (CHIP_STATE_PRICE_IN_PEAK_COST_ZONE)。
        # 剧情 (动态): 筹码仍在发散 (CHIP_DYN_DIVERGING)。
        is_diverging_tactical = atomic.get('CHIP_DYN_DIVERGING', default_series)
        states['RISK_STRATEGY_BAILOUT_FAILURE_A'] = is_in_battle_zone & is_diverging_tactical
        
        strategic_scenarios = self.diagnose_strategic_scenarios(df)
        states.update(strategic_scenarios) # 将新生成的战略情景信号合并到总状态中
        
        static_multi_dyn_scenarios = self.diagnose_static_multi_dynamic_scenarios(df)
        states.update(static_multi_dyn_scenarios)
        
        # if states['RISK_CHIP_PRICE_DIVERGENCE'].any():
        #     print(f"            -> [S级风险] 侦测到 {states['RISK_CHIP_PRICE_DIVERGENCE'].sum()} 次“价格筹码顶背离”！")
        
        # --- 步骤 8: 终极风险信号合成 (原步骤7) ---
        # print("          -> [复合风险合成] 正在整合所有筹码层面的风险信号...")
        chip_risk_1 = atomic.get('CHIP_DYN_DIVERGING', default_series)
        chip_risk_2 = atomic.get('CHIP_DYN_COST_FALLING', default_series)
        chip_risk_3 = atomic.get('CHIP_DYN_WINNER_RATE_COLLAPSING', default_series)
        chip_risk_4 = states.get('RISK_CONTEXT_LONG_TERM_DISTRIBUTION', default_series)
        chip_risk_5 = states.get('RISK_CHIP_CONC_ACCEL_WORSENING', default_series)
        chip_risk_6 = atomic.get('RISK_BEHAVIOR_PANIC_FLEEING_S', default_series)
        chip_risk_7 = states.get('RISK_CHIP_PRICE_DIVERGENCE', default_series)
        chip_risk_8 = atomic.get('CHIP_DYN_HEALTH_DETERIORATING', default_series)
        chip_risk_9 = states.get('RISK_STRATEGY_DISTRIBUTION_CONFIRMED_S', default_series)
        chip_risk_10 = states.get('RISK_STRATEGY_BAILOUT_FAILURE_A', default_series)
        chip_risk_11 = states.get('SCENARIO_HIGH_ALTITUDE_EVACUATION_S', default_series)
        chip_risk_12 = states.get('SCENARIO_DAM_CRACKING_B', default_series)
        
        # 最终裁定：只要上述任一高风险信号出现，就认为筹码结构严重失效
        is_chip_structure_unhealthy = (chip_risk_1 | chip_risk_2 | chip_risk_3 | chip_risk_4 | 
                                       chip_risk_5 | chip_risk_6 | chip_risk_7 | chip_risk_8 | 
                                       chip_risk_9 | chip_risk_10 | chip_risk_11 | chip_risk_12)
        states['RISK_CHIP_STRUCTURE_CRITICAL_FAILURE'] = is_chip_structure_unhealthy
        return states, triggers

    def diagnose_dynamic_chip_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V284.2 错误修复与逻辑完善版】
        - 核心修复: 修复了因变量未定义导致的 `NameError` 严重错误。
        - 核心完善: 完整实现了基于短、中、长三周期的筹码动态分析逻辑，
                    补充了缺失的变量定义，使“共振”与“背离”信号的计算准确无误。
        - 收益: 提升了代码的健壮性和可读性，并确保了高级筹码动态信号的正确生成。
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
            'SLOPE_21_concentration_90pct_D',
            'SLOPE_55_concentration_90pct_D',
            'SLOPE_5_peak_cost_D', 'ACCEL_5_peak_cost_D',
            'SLOPE_5_total_winner_rate_D', 'ACCEL_5_total_winner_rate_D',
            'SLOPE_5_chip_health_score_D', 'ACCEL_5_chip_health_score_D'
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"            -> [严重警告] 动态分析中心缺少关键的斜率/加速度数据: {missing_cols}，模块已跳过！")
            return states
            
        # 定义“战场上下文”过滤器，用于区分发散行为的风险等级
        is_in_high_level_zone = self.strategy.atomic_states.get('CONTEXT_RISK_HIGH_LEVEL_ZONE', default_series)
        
        # --- 步骤2: 对“筹码集中度”进行多周期动态分析 ---
        # 2.1 定义短、中、长周期的客观集中/发散趋势
        is_concentrating_short_term = df['SLOPE_5_concentration_90pct_D'] < 0
        is_concentrating_mid_term = df['SLOPE_21_concentration_90pct_D'] < 0
        is_concentrating_long_term = df['SLOPE_55_concentration_90pct_D'] < 0
        is_diverging_short_term = df['SLOPE_5_concentration_90pct_D'] > 0
        is_diverging_mid_term = df['SLOPE_21_concentration_90pct_D'] > 0
        is_diverging_long_term = df['SLOPE_55_concentration_90pct_D'] > 0
        
        # 2.2 定义“成本试金石”：成本峰必须稳定或抬高
        # 我们允许非常微小的负斜率，以容忍正常波动
        cost_stability_tolerance = -0.001 
        is_cost_constructive = df['SLOPE_5_peak_cost_D'] >= cost_stability_tolerance
        
        # 2.3 定义“绿色通道”豁免条件
        is_washout_absorption = self.strategy.atomic_states.get('OPP_CONSTRUCTIVE_WASHOUT_ABSORPTION_A', default_series)
        is_bottoming_phase = self.strategy.atomic_states.get('MA_STATE_BOTTOM_PASSIVATION', default_series) # 获取市场是否处于底部钝化状态
        
        # 2.4 重新定义基础的“建设性筹码集中” (主要看短期)
        states['CHIP_DYN_CONCENTRATING'] = (is_concentrating_short_term & is_cost_constructive) | is_washout_absorption | (is_concentrating_short_term & is_bottoming_phase)
        # 定义“共振式集中”(A级机会)：短、中、长周期均在集中，这是最强的吸筹信号。
        states['CHIP_DYN_CONCENTRATING_RESONANCE_A'] = is_concentrating_short_term & is_concentrating_mid_term & is_concentrating_long_term & is_cost_constructive
        # 定义“背离式吸筹”(B级机会)：长期仍在发散或走平，但短期已逆转为集中，是潜在的底部反转信号。
        is_long_term_not_improving = ~is_concentrating_long_term # 长期趋势未改善
        states['OPP_CHIP_REVERSAL_GATHERING_B'] = is_long_term_not_improving & is_concentrating_short_term & is_cost_constructive
        
        # 2.5 加速动态 (继承基础的“建设性”前提)
        p_chip = get_params_block(self.strategy, 'chip_feature_params')
        accel_threshold = get_param_value(p_chip.get('accel_concentration_threshold'), -0.001)
        is_accelerating_action = df['ACCEL_5_concentration_90pct_D'] < accel_threshold
        states['CHIP_DYN_S_ACCEL_CONCENTRATING'] = states['CHIP_DYN_CONCENTRATING'] & is_accelerating_action
        # 2.6 重新定义发散动态
        # 客观发散行为
        states['CHIP_DYN_OBJECTIVE_DIVERGING'] = is_diverging_short_term
        # 结合高位上下文，生成战术风险信号
        states['CHIP_DYN_DIVERGING'] = is_diverging_short_term & is_in_high_level_zone
        # 定义“共振式派发”(A级风险)：短、中、长周期均在发散，这是最明确的派发信号。
        states['RISK_CHIP_DIVERGING_RESONANCE_A'] = is_diverging_short_term & is_diverging_mid_term & is_diverging_long_term & is_in_high_level_zone
        # 定义“背离式派发”(B级风险)：长期仍在集中，但短期已逆转为发散，是潜在的顶部反转预警。
        is_long_term_still_good = is_concentrating_long_term # 长期趋势看似良好
        states['RISK_CHIP_REVERSAL_DIVERGING_B'] = is_long_term_still_good & is_diverging_short_term & is_in_high_level_zone
        # 加速发散风险
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
        long_ma_col = 'EMA_55_D'
        if fault_formed_col in df.columns:
            is_in_uptrend_context = df['close_D'] > df[long_ma_col]
            states['OPP_CHIP_FAULT_REBIRTH_S'] = df[fault_formed_col] & is_in_uptrend_context
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

        # 1.4 定义S级风险和A级机会 (基于加速度)
        # S级风险 - 恐慌加速: 中期趋势已在出逃，且短期出逃正在加速，这是最危险的信号。
        states['RISK_BEHAVIOR_PANIC_FLEEING_S'] = states['RISK_BEHAVIOR_WINNERS_FLEEING_B'] & is_fleeing_accelerating
        
        # A级机会 - 卖盘衰竭: 获利盘虽然还在卖(短期斜率>0)，但卖出力度已在减弱(加速度<0)。
        # 增加趋势上下文过滤器，确保只在下跌末期或趋势反转初期应用此左侧信号
        is_bottoming_context = self.strategy.atomic_states.get('MA_STATE_BOTTOM_PASSIVATION', default_series)
        is_early_reversal = self.strategy.atomic_states.get('STRUCTURE_EARLY_REVERSAL_B', default_series) # 假设有这个信号
        is_safe_context = is_bottoming_context | is_early_reversal
        
        # --- 机会行为2: 恐慌盘割肉 (底部反向指标) ---
        # 定义：股价当日大跌（例如超过5%），且成交量主要由套牢盘贡献（例如占比超过50%）
        is_sharp_drop = df['pct_change_D'] < -0.05
        is_panic_selling = df['turnover_from_losers_ratio_D'] > 50.0
        
        # 增加上下文过滤器，确保只在底部反向或趋势反转时应用此左侧信号
        states['OPP_BEHAVIOR_PANIC_CAPITULATION_A'] = is_sharp_drop & is_panic_selling & is_safe_context
        states['OPP_BEHAVIOR_SELLING_EXHAUSTION_A'] = is_fleeing_short_term & is_fleeing_decelerating & is_safe_context

        # 打印情报
        # if states['RISK_BEHAVIOR_PANIC_FLEEING_S'].any():
        #     print(f"          -> [S级战略风险] 侦测到 {states['RISK_BEHAVIOR_PANIC_FLEEING_S'].sum()} 次“获利盘恐慌加速出逃”！")
        # elif states['RISK_BEHAVIOR_WINNERS_FLEEING_A'].any():
        #     print(f"          -> [A级战略风险] 侦测到 {states['RISK_BEHAVIOR_WINNERS_FLEEING_A'].sum()} 次“长期派发”共振！")
        
        # if states['OPP_BEHAVIOR_SELLING_EXHAUSTION_A'].any():
        #     print(f"          -> [A级机会情报] 侦测到 {states['OPP_BEHAVIOR_SELLING_EXHAUSTION_A'].sum()} 次“卖盘衰竭”信号！")

        return states

    def diagnose_peak_formation_dynamics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V340.1 战略过滤版】筹码峰“创世纪”模块
        - 核心修复: 为胜率仅10.8%的“隐蔽吸筹”信号增加了“均线底部钝化”的战略环境过滤器。
        - 收益: 根治了在下跌中继中错误识别“吸筹”的致命缺陷，确保只在趋势有反转
                潜力的安全区域内识别此机会。
        """
        # print("        -> [筹码峰“创世纪”模块 V340.1 战略过滤版] 启动...")
        states = {}
        default_series = pd.Series(False, index=df.index) # [新增代码] 增加默认序列
        
        # --- 1. 检查所需数据 ---
        required_cols = ['peak_cost_D', 'volume_D', 'VOL_MA_21_D', 'SLOPE_55_EMA_55_D']
        if any(c not in df.columns for c in required_cols):
            print("          -> [警告] 缺少诊断筹码峰起源所需数据，模块跳过。")
            return {}

        # --- 2. 识别“政权更迭”事件 ---
        is_peak_changed = (df['peak_cost_D'].pct_change().abs() > 0.015)
        
        # --- 3. 使用向量化状态机替代for循环 ---
        stability_period = 3 # 需要稳定3天
        
        # 步骤 3.1: 使用 cumsum 创建事件区块ID，每个ID代表一个潜在的稳定观察期
        change_blocks = is_peak_changed.cumsum()
        
        # 步骤 3.2: 使用 groupby().transform() 并行计算每个区块的属性
        # 计算每个区块的大小（持续天数）
        block_sizes = change_blocks.groupby(change_blocks).transform('size')
        # 计算每个区块内成本峰的相对标准差，判断其稳定性
        block_std = df['peak_cost_D'].groupby(change_blocks).transform('std')
        block_mean = df['peak_cost_D'].groupby(change_blocks).transform('mean')
        is_block_stable = (block_std / block_mean.replace(0, np.nan)).fillna(0) < 0.01
        
        # 步骤 3.3: 确定哪些区块是“已确认形成”的
        is_formation_confirmed = (block_sizes >= stability_period) & is_block_stable
        
        # 步骤 3.4: 获取已确认区块的“出生证明”
        # 获取每个区块的起始日期（即政权更迭日）
        block_start_indices = df.index.to_series().groupby(change_blocks).transform('first')
        # 仅保留已确认区块的起始日期
        formation_dates = block_start_indices.where(is_formation_confirmed)
        
        # 为了获取形成日的成交量比率，创建一个从日期到比率的映射
        unique_formation_dates = formation_dates.dropna().unique()
        if len(unique_formation_dates) > 0:
            formation_day_data = df.loc[unique_formation_dates]
            # 使用 .get() 增加健壮性，防止 VOL_MA_21_D 为0
            ratio_map = (formation_day_data['volume_D'] / formation_day_data['VOL_MA_21_D'].replace(0, np.nan)).to_dict()
            formation_volume_ratios = formation_dates.map(ratio_map)
        else:
            formation_volume_ratios = pd.Series(np.nan, index=df.index)

        # --- 4. 解读“出生证明”，生成战略信号 ---
        is_high_volume_formation = formation_volume_ratios > 2.0
        is_low_volume_formation = formation_volume_ratios < 0.7
        
        # 获取形成前一天的趋势状态
        trend_at_formation = df['SLOPE_55_EMA_55_D'].shift(1).loc[formation_dates.dropna()].to_dict()
        prev_day_trend = formation_dates.map(trend_at_formation)
        
        is_after_downtrend = prev_day_trend <= 0
        is_after_uptrend = prev_day_trend > 0

        # 为“隐蔽吸筹”增加战略环境过滤器，这是本次修复的核心。
        # 过滤器：必须处于“均线底部钝化”状态，代表长期下跌趋势已得到遏制。
        is_bottoming_context = self.strategy.atomic_states.get('MA_STATE_BOTTOM_PASSIVATION', default_series)

        # 组合生成最终的原子状态
        states['PEAK_DYN_FORTRESS_SUPPORT'] = is_high_volume_formation & is_after_downtrend
        states['PEAK_DYN_EXHAUSTION_TOP'] = is_high_volume_formation & is_after_uptrend
        # 注入战略过滤器
        states['PEAK_DYN_STEALTH_ACCUMULATION'] = is_low_volume_formation & is_after_downtrend & is_bottoming_context

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
        had_recent_setup = peak_battle_setup.shift(1).rolling(
            window=confirmation_lookback_days, 
            min_periods=1
        ).max().astype(bool)
        
        final_opportunity_signal = is_price_rising_meaningfully & had_recent_setup
        
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

    def synthesize_prime_chip_opportunity(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】黄金筹码机会合成模块
        - 核心职责: 融合“筹码锁定”的微观结构与“上涨初期”的宏观战场环境，
                      生成一个高确定性的S级机会信号。
        - 收益: 彻底解决了原子筹码信号因缺乏上下文而产生的“崩盘式集中”误判问题。
        """
        # print("        -> [黄金筹码机会合成模块 V1.0] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_series = pd.Series(False, index=df.index)
        # 1. 获取结构条件：筹码必须已经锁定且稳定
        is_chip_locked = atomic.get('CHIP_CONC_LOCKED_AND_STABLE_A', default_series)
        # 2. 获取战场条件：必须处于上涨初期
        is_in_early_stage = atomic.get('CONTEXT_TREND_STAGE_EARLY', default_series)
        # 3. 最终裁定：结构与战场的完美共振
        final_signal = is_chip_locked & is_in_early_stage
        states['CHIP_STRUCTURE_PRIME_OPPORTUNITY_S'] = final_signal
        # if final_signal.any():
        #     print(f"          -> [S级机会确认] 侦测到 {final_signal.sum()} 次“筹码结构黄金机会”！")
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

    def diagnose_chip_dynamics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V320.0 数据驱动版】筹码动态诊断模块
        - 核心重构: 完全重构为直接消费数据层预计算好的“斜率”和“加速度”列。
                    不再进行任何动态计算，显著提升了性能和代码的健壮性。
        """
        # print("        -> [筹码动态诊断模块 V320.0 数据驱动版] 启动...")
        states = {}
        default_series = pd.Series(False, index=df.index)

        # --- 1. 军备检查 ---
        required_cols = [
            'SLOPE_5_concentration_90pct_D', 'ACCEL_5_concentration_90pct_D',
            'SLOPE_5_peak_cost_D', 'ACCEL_5_peak_cost_D',
            'SLOPE_5_total_winner_rate_D', 'ACCEL_21_total_winner_rate_D',
            'SLOPE_5_chip_health_score_D', # 筹码健康度斜率
            'SLOPE_5_winner_profit_margin_D' # 获利盘利润垫斜率
        ]
        if any(col not in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            print(f"          -> [警告] 缺少筹码动态诊断所需列，模块跳过。缺失: {missing_cols}")
            return {}

        # --- 2. 筹码集中/发散动态 (速度与加速度) ---
        # 筹码集中：5日集中度斜率为负 (集中度数值越小代表越集中)
        states['CHIP_DYN_CONCENTRATING'] = df['SLOPE_5_concentration_90pct_D'] < 0
        # 筹码加速集中：5日集中度加速度为负 (集中趋势在强化)
        states['CHIP_DYN_S_ACCEL_CONCENTRATING'] = df['ACCEL_5_concentration_90pct_D'] < 0
        # 筹码发散：5日集中度斜率为正
        states['CHIP_DYN_DIVERGING'] = df['SLOPE_5_concentration_90pct_D'] > 0
        # 筹码加速发散：5日集中度加速度为正
        states['CHIP_DYN_ACCEL_DIVERGING'] = df['ACCEL_5_concentration_90pct_D'] > 0

        # --- 3. 成本动态 (速度与加速度) ---
        # 成本抬升：5日成本峰斜率为正
        states['CHIP_DYN_COST_RISING'] = df['SLOPE_5_peak_cost_D'] > 0
        # 成本加速抬升：5日成本峰加速度为正
        states['CHIP_DYN_COST_ACCELERATING'] = df['ACCEL_5_peak_cost_D'] > 0
        # 成本松动：5日成本峰斜率为负
        states['CHIP_DYN_COST_FALLING'] = df['SLOPE_5_peak_cost_D'] < 0

        # --- 4. 获利盘动态 (速度与加速度) ---
        # 获利盘比例上升：5日获利盘比例斜率为正
        states['CHIP_DYN_WINNER_RATE_RISING'] = df['SLOPE_5_total_winner_rate_D'] > 0
        # 获利盘崩盘：5日获利盘比例斜率为负
        states['CHIP_DYN_WINNER_RATE_COLLAPSING'] = df['SLOPE_5_total_winner_rate_D'] < 0
        # 获利盘加速崩盘：21日获利盘比例加速度为负
        states['CHIP_DYN_WINNER_RATE_ACCEL_COLLAPSING'] = df['ACCEL_21_total_winner_rate_D'] < 0

        # --- 5. 筹码健康度动态 (速度) ---
        # 筹码健康度改善：5日筹码健康度斜率为正 且 筹码集中度在增加
        states['CHIP_DYN_HEALTH_IMPROVING'] = (df['SLOPE_5_chip_health_score_D'] > 0) & states['CHIP_DYN_CONCENTRATING']
        # 筹码健康度恶化：5日筹码健康度斜率为负 且 筹码集中度在发散
        states['CHIP_DYN_HEALTH_DETERIORATING'] = (df['SLOPE_5_chip_health_score_D'] < 0) & states['CHIP_DYN_DIVERGING']
        
        # --- 6. 获利盘利润垫动态 (速度) ---
        # 获利盘利润垫抬升：5日获利盘利润垫斜率为正
        states['CHIP_DYN_WINNER_PROFIT_MARGIN_RISING'] = df['SLOPE_5_winner_profit_margin_D'] > 0
        # 获利盘利润垫收缩：5日获利盘利润垫斜率为负
        states['CHIP_DYN_WINNER_PROFIT_MARGIN_SHRINKING'] = df['SLOPE_5_winner_profit_margin_D'] < 0

        # print("        -> [筹码动态诊断模块 V320.0] 分析完毕。")
        return states

    # “静态筹码结构诊断”方法
    def diagnose_static_chip_structure(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.1 场景扩充版】静态筹码结构诊断模块
        - 核心职责: 诊断当前价格与成本分布的静态关系，以及筹码分布的形态特征。
        - 核心升级 (本次修改): 新增了“价格低于成本峰”的静态场景，为识别深度反转信号提供基础。
        """
        print("            -> [静态筹码结构诊断模块 V1.1] 启动...") # [修改代码行]
        states = {}
        default_series = pd.Series(False, index=df.index)
        
        # --- 1. 军备检查 ---
        required_cols = ['close_D', 'peak_cost_D', 'concentration_70pct_D', 'concentration_90pct_D', 'total_winner_rate_D']
        if any(c not in df.columns for c in required_cols):
            print("              -> [警告] 缺少诊断静态筹码结构所需列，模块跳过。")
            return {}

        # --- 2. 价格与成本峰关系诊断 ---
        profit_cushion_ratio = 1.15 # 价格比成本峰高出15%
        states['CHIP_STATE_PRICE_ABOVE_PEAK_COST'] = df['close_D'] > (df['peak_cost_D'] * profit_cushion_ratio)

        entanglement_ratio = 0.05 # 价格在成本峰上下5%的区域内
        states['CHIP_STATE_PRICE_IN_PEAK_COST_ZONE'] = (df['close_D'] / df['peak_cost_D']).between(1 - entanglement_ratio, 1 + entanglement_ratio)

        # [新增代码块开始]
        # 信号3: 价格显著低于成本区 (大部分持仓者被套牢，市场处于绝望状态)
        loss_zone_ratio = 0.90 # 价格比成本峰低10%
        states['CHIP_STATE_PRICE_BELOW_PEAK_COST'] = df['close_D'] < (df['peak_cost_D'] * loss_zone_ratio)
        # [新增代码块结束]

        # --- 3. 筹码分布形态诊断 ---
        concentration_gap = (df['concentration_90pct_D'] - df['concentration_70pct_D']).replace(0, np.nan)
        compactness_ratio = concentration_gap / df['concentration_70pct_D']
        compact_threshold = 0.3 
        states['CHIP_STRUCTURE_HIGHLY_COMPACT'] = compactness_ratio < compact_threshold

        # --- 4. 获利盘状态诊断 ---
        states['CHIP_STATE_UNIVERSAL_PROFIT'] = df['total_winner_rate_D'] > 95.0
        
        print("            -> [静态筹码结构诊断模块 V1.1] 诊断完毕。") # [修改代码行]
        return states

    # “筹码行为诊断”方法
    def diagnose_chip_behavior_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】筹码行为诊断模块
        - 核心职责: 基于换手率数据，诊断获利盘的抛售意愿和亏损盘的止损行为。
                    这是洞察市场情绪和主力意图的关键。
        - 输出: 一系列描述市场参与者“动态行为”的中性原子信号。
        """
        print("            -> [筹码行为诊断模块 V1.0] 启动...")
        states = {}
        default_series = pd.Series(False, index=df.index)

        # --- 1. 军备检查 ---
        # turnover_from_winners_ratio_D: 获利盘换手率
        # turnover_from_losers_ratio_D: 亏损盘换手率
        required_cols = ['turnover_from_winners_ratio_D', 'turnover_from_losers_ratio_D', 'pct_change_D']
        if any(c not in df.columns for c in required_cols):
            print("              -> [警告] 缺少诊断筹码行为所需列 (如 turnover_from_winners_ratio_D)，模块跳过。")
            return {}

        # --- 2. 获利盘行为诊断 ---
        # 信号1: 获利盘抛压较大 (获利盘换手率高于一个动态阈值，例如60日均值的1.5倍)
        winner_turnover_ma = df['turnover_from_winners_ratio_D'].rolling(60).mean()
        is_high_winner_turnover = df['turnover_from_winners_ratio_D'] > (winner_turnover_ma * 1.8)
        states['CHIP_BEHAVIOR_HIGH_PROFIT_TAKING_PRESSURE'] = is_high_winner_turnover

        # --- 3. 亏损盘行为诊断 ---
        # 信号2: 亏损盘恐慌杀跌/投降 (股价大跌的同时，亏损盘换手率激增)
        is_sharp_drop = df['pct_change_D'] < -0.05 # 股价大跌超过5%
        loser_turnover_ma = df['turnover_from_losers_ratio_D'].rolling(60).mean()
        is_high_loser_turnover = df['turnover_from_losers_ratio_D'] > (loser_turnover_ma * 2.0)
        states['CHIP_BEHAVIOR_LOSER_CAPITULATION'] = is_sharp_drop & is_high_loser_turnover
        
        print("            -> [筹码行为诊断模块 V1.0] 诊断完毕。")
        return states

    def diagnose_strategic_scenarios(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】战略情景诊断模块 (静态 x 多维动态)
        - 核心职责: 融合静态筹码“场景”与短、中、长三维动态筹码“剧情”，生成具备高度战术背景的复合原子信号。
        - 收益: 将分析从“信号”提升至“情景”层面，能够更精准地解读主力意图，区分洗盘与出货、吸筹与反弹。
        """
        # print("        -> [战略情景诊断模块 V1.0] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_series = pd.Series(False, index=df.index)

        # --- 1. 提取基础“场景”（静态信号）---
        # 场景1: 堡垒已成 (筹码高度紧凑，主力高度控盘)
        is_fortress_static = atomic.get('CHIP_STRUCTURE_HIGHLY_COMPACT', default_series)
        # 场景2: 阵地战 (价格在成本密集区拉锯)
        is_battlezone_static = atomic.get('CHIP_STATE_PRICE_IN_PEAK_COST_ZONE', default_series)
        # 场景3: 高空巡航 (价格已大幅脱离成本区，有丰厚利润垫)
        is_high_altitude_static = atomic.get('CHIP_STATE_PRICE_ABOVE_PEAK_COST', default_series)

        # --- 2. 提取各维度“剧情”（动态信号）---
        # 剧情1: 短期动态
        is_concentrating_short = atomic.get('CHIP_DYN_CONCENTRATING', default_series)
        is_diverging_short = atomic.get('CHIP_DYN_DIVERGING', default_series)
        # 新增更多维度的动态“剧情”
        is_cost_accelerating = atomic.get('CHIP_DYN_COST_ACCELERATING', default_series)
        is_cost_falling = atomic.get('CHIP_DYN_COST_FALLING', default_series)
        is_profit_margin_rising = atomic.get('CHIP_DYN_WINNER_PROFIT_MARGIN_RISING', default_series)
        is_profit_margin_shrinking = atomic.get('CHIP_DYN_WINNER_PROFIT_MARGIN_SHRINKING', default_series)
        is_selling_exhausted = atomic.get('OPP_BEHAVIOR_SELLING_EXHAUSTION_A', default_series)
        # 剧情2: 共振动态 (最强趋势)
        is_concentrating_resonance = atomic.get('CHIP_DYN_CONCENTRATING_RESONANCE_A', default_series)
        is_diverging_resonance = atomic.get('RISK_CHIP_DIVERGING_RESONANCE_A', default_series)
        # 剧情3: 背离动态 (拐点信号)
        is_reversal_gathering = atomic.get('OPP_CHIP_REVERSAL_GATHERING_B', default_series)
        is_reversal_diverging = atomic.get('RISK_CHIP_REVERSAL_DIVERGING_B', default_series)

        # --- 3. 融合“场景”与“剧情”，生成全新战略原子信号 ---
        # === 机会情景 ===
        # 情景A1 (S级机会): “主升浪共振” (原A级信号升级)
        # 解读: 主力已高度控盘（静态），且仍在全周期持续吸筹（动态1），同时成本正在加速抬升（动态2），
        #      获利盘的利润也在增加（动态3）。这是最强的上涨共振信号。
        states['SCENARIO_MAIN_WAVE_RESONANCE_S'] = (
            is_fortress_static &
            is_concentrating_resonance &
            is_cost_accelerating &
            is_profit_margin_rising
        )

        # 情景A2 (A级机会): “阵地战转折点” (原B级信号升级)
        # 解读: 在关键成本区拉锯时（静态），出现了底部反转式的吸筹信号（动态1），且卖盘已出现衰竭迹象（动态2）。
        #      这表明空头力量衰竭，多头即将掌控局面，是突破前的黄金坑。
        states['SCENARIO_BATTLEZONE_TURNING_POINT_A'] = (
            is_battlezone_static &
            is_reversal_gathering &
            is_selling_exhausted
        )
        # === 风险情景 ===
        # 情景R1 (S级风险): “高位派发陷阱” (原信号逻辑增强)
        # 解读: 在获利丰厚的高位（静态），出现全周期共振式的派发（动态1），同时获利盘的平均利润开始收缩（动态2）。
        #      这是典型的“明拉暗出”，是极度危险的顶部信号。
        states['SCENARIO_HIGH_ALTITUDE_DISTRIBUTION_TRAP_S'] = (
            is_high_altitude_static &
            is_diverging_resonance &
            is_profit_margin_shrinking
        )

        # 情景R2 (A级风险): “堡垒内部瓦解” (原B级信号升级)
        # 解读: 堡垒看似稳固（静态），但短期已出现派发迹象（动态1），且成本峰开始松动（动态2）。
        #      这表明主力已无心守盘，高控盘成为派发的掩护，风险极高。
        states['SCENARIO_FORTRESS_INTERNAL_COLLAPSE_A'] = (
            is_fortress_static &
            is_reversal_diverging &
            is_cost_falling
        )
        
        # === 中性观察情景 ===
        # 情景N1 (中性): “堡垒下的洗盘”
        # 解读: 主力控盘度很高，短期的筹码发散大概率是洗盘行为，旨在清洗浮筹，可继续观察。
        states['SCENARIO_WASHOUT_BELOW_FORTRESS_N'] = is_fortress_static & is_diverging_short & ~is_diverging_resonance

        return states

    def diagnose_static_multi_dynamic_scenarios(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】静态-多动态交叉验证模块 (补充)
        - 核心职责: 识别更多基于特定静态筹码场景与多维动态行为组合的战术信号。
        """
        print("        -> [静态-多动态交叉验证模块(补充) V1.0] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_series = pd.Series(False, index=df.index)

        # --- 1. 军备检查：确保所有必需的动态列都存在 ---
        # 动态计算缺失的加速度列，如果数据工程层未提供
        if 'SLOPE_5_close_D' in df.columns and 'ACCEL_5_close_D' not in df.columns:
            df['ACCEL_5_close_D'] = df['SLOPE_5_close_D'].diff()
        if 'SLOPE_5_turnover_from_losers_ratio_D' in df.columns and 'ACCEL_5_turnover_from_losers_ratio_D' not in df.columns:
            df['ACCEL_5_turnover_from_losers_ratio_D'] = df['SLOOPE_5_turnover_from_losers_ratio_D'].diff()
        if 'SLOPE_5_turnover_from_winners_ratio_D' in df.columns and 'ACCEL_5_turnover_from_winners_ratio_D' not in df.columns:
            df['ACCEL_5_turnover_from_winners_ratio_D'] = df['SLOPE_5_turnover_from_winners_ratio_D'].diff()

        required_cols = [
            'ACCEL_5_turnover_from_losers_ratio_D', 'SLOPE_5_concentration_90pct_D', 'SLOPE_5_close_D',
            'ACCEL_5_turnover_from_winners_ratio_D', 'ACCEL_5_close_D', 'ACCEL_5_concentration_90pct_D'
        ]
        if any(c not in df.columns for c in required_cols):
            missing_cols = [c for c in required_cols if c not in df.columns]
            print(f"          -> [警告] 静态-多动态交叉验证(补充)模块缺少关键数据: {missing_cols}，模块已跳过！")
            return states
        # --- 2. 生成复合信号 ---
        # === 机会信号 ===
        # 信号1 (S级机会): 深水炸弹·绝望反转
        # 解读: 在最悲观的区域，恐慌抛售已近尾声，同时有新主力在悄悄吸筹，价格也开始响应。
        is_despair_zone_static = atomic.get('CHIP_STATE_PRICE_BELOW_PEAK_COST', default_series)
        is_bleeding_stopping_dyn = df['ACCEL_5_turnover_from_losers_ratio_D'] < 0 # 套牢盘抛售加速度为负（抛压减速）
        is_stealth_buying_start_dyn = df['SLOPE_5_concentration_90pct_D'] < 0 # 筹码开始集中
        is_price_turning_up_dyn = df['SLOPE_5_close_D'] > 0 # 价格开始回升
        states['OPP_CHIP_DEEP_WATER_REVERSAL_S'] = (
            is_despair_zone_static &
            is_bleeding_stopping_dyn &
            is_stealth_buying_start_dyn &
            is_price_turning_up_dyn
        )
        # 信号2 (A级机会): 阵地战·吸筹确认
        # 解读: 在关键的拉锯战中，主力吸筹的意愿和力度都在增强，同时浮筹被有效清洗。
        is_battle_zone_static = atomic.get('CHIP_STATE_PRICE_IN_PEAK_COST_ZONE', default_series)
        is_absorption_accelerating_dyn = df['ACCEL_5_concentration_90pct_D'] < 0 # 筹码加速集中
        is_losers_giving_up_dyn = df['SLOPE_5_turnover_from_losers_ratio_D'] < 0 # 套牢盘换手率下降（惜售或已被洗出）
        states['OPP_CHIP_ACCUMULATION_CONFIRMED_A'] = (
            is_battle_zone_static &
            is_absorption_accelerating_dyn &
            is_losers_giving_up_dyn
        )
        # === 风险信号 ===
        # 信号3 (S级风险): 高位狂欢·亢奋陷阱
        # 解读: 在市场最乐观的时候，主力利用散户的追高情绪，加速派发筹码，同时拉升力度减弱。
        is_euphoria_zone_static = atomic.get('CHIP_STATE_UNIVERSAL_PROFIT', default_series)
        is_winner_selling_accel_dyn = df['ACCEL_5_turnover_from_winners_ratio_D'] > 0 # 获利盘抛售加速
        is_chip_diverging_dyn = df['SLOPE_5_concentration_90pct_D'] > 0 # 筹码开始发散
        is_price_momentum_fading_dyn = df['ACCEL_5_close_D'] < 0 # 价格上涨加速度放缓
        states['RISK_CHIP_EUPHORIA_TRAP_S'] = (
            is_euphoria_zone_static &
            is_winner_selling_accel_dyn &
            is_chip_diverging_dyn &
            is_price_momentum_fading_dyn
        )
        return states








