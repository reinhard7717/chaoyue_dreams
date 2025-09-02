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
        【V320.7 终极完全量化版】筹码情报最高司令部
        - 核心重构: 所有信号，包括基础过程、复合机会、以及最终的风险裁决，
                    均已改造为消费专属的量化得分，标志着本模块的全面量化升级最终完成。
        """
        print("        -> [筹码情报最高司令部 V320.7 终极完全量化版] 启动...")
        states = {}
        triggers = {}
        default_series = pd.Series(False, index=df.index)
        df = self.diagnose_quantitative_chip_scores(df)
        static_states = self.diagnose_static_chip_structure(df)
        states.update(static_states)
        self.strategy.atomic_states.update(static_states)
        dynamic_states = self.diagnose_dynamic_chip_states(df)
        states.update(dynamic_states)
        self.strategy.atomic_states.update(dynamic_states)
        behavior_states = self.diagnose_chip_behavior_states(df)
        states.update(behavior_states)
        self.strategy.atomic_states.update(behavior_states)
        peak_formation_states = self.diagnose_peak_formation_dynamics(df)
        states.update(peak_formation_states)
        opportunity_states = self.diagnose_chip_opportunities(df)
        states.update(opportunity_states)
        risk_behavior_states = self.diagnose_chip_risks_and_behaviors(df)
        states.update(risk_behavior_states)
        peak_battle_states = self.diagnose_peak_battle_dynamics(df)
        states.update(peak_battle_states)
        p = get_params_block(self.strategy, 'chip_feature_params')
        if not get_param_value(p.get('enabled'), False):
            return states, triggers
        # 顶层战略背景
        strategic_score = df['CHIP_SCORE_CONTEXT_STRATEGIC_GATHERING']
        states['CONTEXT_CHIP_STRATEGIC_GATHERING'] = strategic_score > strategic_score.rolling(120).quantile(0.60)
        states['CONTEXT_CHIP_STRATEGIC_DISTRIBUTION'] = strategic_score < strategic_score.rolling(120).quantile(0.40)
        # C/B/B+ 级动态过程
        gathering_score = df['CHIP_SCORE_GATHERING_INTENSITY']
        b_plus_threshold = gathering_score.rolling(120).quantile(0.95)
        b_threshold = gathering_score.rolling(120).quantile(0.85)
        c_threshold = gathering_score.rolling(120).quantile(0.70)
        states['CHIP_CONC_INTENSIFYING_B_PLUS'] = gathering_score > b_plus_threshold
        states['CHIP_CONC_ACCELERATED_GATHERING_B'] = (gathering_score > b_threshold) & (gathering_score <= b_plus_threshold)
        states['CHIP_CONC_STEADY_GATHERING_C'] = (gathering_score > c_threshold) & (gathering_score <= b_threshold)
        # A/S 级静态结果与复合机会
        locked_stable_score = df['CHIP_SCORE_STRUCTURE_LOCKED_STABLE']
        states['CHIP_CONC_LOCKED_AND_STABLE_A'] = locked_stable_score > locked_stable_score.rolling(120).quantile(0.90)
        locked_breakout_score = df['CHIP_SCORE_OPP_LOCKED_BREAKOUT']
        states['OPP_CHIP_LOCKED_BREAKOUT_S'] = locked_breakout_score > locked_breakout_score.rolling(120).quantile(0.95)
        # 独立触发器与风险
        ignition_trigger_score = df['CHIP_SCORE_TRIGGER_IGNITION']
        # 将硬编码的 > 0.98 替换为动态分位数，以保持逻辑一致性
        triggers['TRIGGER_CHIP_IGNITION'] = ignition_trigger_score > ignition_trigger_score.rolling(120).quantile(0.98)
        long_term_risk_score = df['CHIP_SCORE_RISK_LONG_TERM_DISTRIBUTION']
        states['RISK_CONTEXT_LONG_TERM_DISTRIBUTION'] = long_term_risk_score > long_term_risk_score.rolling(120).quantile(0.90)
        worsening_turn_score = df['CHIP_SCORE_RISK_WORSENING_TURN']
        states['RISK_CHIP_CONC_ACCEL_WORSENING'] = worsening_turn_score > worsening_turn_score[worsening_turn_score > 0].rolling(60).quantile(0.80)
        price_divergence_states = self.diagnose_chip_price_divergence(df)
        states.update(price_divergence_states)
        prime_opp_states = self.synthesize_prime_chip_opportunity(df)
        states.update(prime_opp_states)
        # 核心复合信号
        ignition_score = df['CHIP_SCORE_OPP_LOCKED_IGNITION']
        states['OPP_STRATEGY_LOCKED_FLOAT_IGNITION_S'] = ignition_score > ignition_score.rolling(120).quantile(0.95)
        precursor_score = df['CHIP_SCORE_OPP_BREAKOUT_PRECURSOR']
        states['OPP_STRATEGY_BREAKOUT_PRECURSOR_A'] = precursor_score > precursor_score.rolling(120).quantile(0.90)
        dist_confirmed_score = df['CHIP_SCORE_RISK_DISTRIBUTION_CONFIRMED']
        states['RISK_STRATEGY_DISTRIBUTION_CONFIRMED_S'] = dist_confirmed_score > dist_confirmed_score.rolling(120).quantile(0.95)
        bailout_failure_score = df['CHIP_SCORE_RISK_BAILOUT_FAILURE']
        states['RISK_STRATEGY_BAILOUT_FAILURE_A'] = bailout_failure_score > bailout_failure_score.rolling(120).quantile(0.90)
        # 战略情景
        strategic_scenarios = self.diagnose_strategic_scenarios(df)
        states.update(strategic_scenarios)
        static_multi_dyn_scenarios = self.diagnose_static_multi_dynamic_scenarios(df)
        states.update(static_multi_dyn_scenarios)
        final_crossover_states = self.diagnose_static_multi_timeframe_scenarios(df)
        states.update(final_crossover_states)
        # --- 步骤 8: 终极风险信号合成 (唯一的、最终的风险裁决逻辑) ---
        critical_risk_score = df['CHIP_SCORE_RISK_CRITICAL_FAILURE']
        states['RISK_CHIP_STRUCTURE_CRITICAL_FAILURE'] = critical_risk_score > critical_risk_score.rolling(120).quantile(0.90)

        return states, triggers

    def diagnose_dynamic_chip_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.0 终极量化消费版】
        - 核心重构: 本方法不再进行任何布尔运算或阈值判断，而是完全消费由评分模块生成的
                    专属“基础动态得分”，并基于动态分位数将其转换为最终的布尔型原子状态。
        """
        # print("            -> [动态筹码分析中心 V285.2 量化消费版] 启动...")
        states = {}
        # --- 步骤1: 军备检查 ---
        required_scores = [
            'CHIP_SCORE_CONCENTRATION_RESONANCE', 'CHIP_SCORE_CONCENTRATION_DIVERGENCE',
            'CHIP_SCORE_DYN_CONCENTRATING', 'CHIP_SCORE_DYN_DIVERGING',
            'CHIP_SCORE_DYN_S_ACCEL_CONCENTRATING', 'CHIP_SCORE_DYN_ACCEL_DIVERGING',
            'CHIP_SCORE_DYN_COST_RISING', 'CHIP_SCORE_DYN_COST_ACCELERATING', 'CHIP_SCORE_DYN_COST_FALLING',
            'CHIP_SCORE_DYN_WINNER_RATE_COLLAPSING', 'CHIP_SCORE_DYN_WINNER_RATE_ACCEL_COLLAPSING',
            'CHIP_SCORE_DYN_HEALTH_IMPROVING', 'CHIP_SCORE_DYN_HEALTH_DETERIORATING'
        ]
        if any(score not in df.columns for score in required_scores):
            missing = [s for s in required_scores if s not in df.columns]
            print(f"            -> [严重警告] 动态分析中心缺少关键的量化得分数据，模块已跳过！缺失: {missing}")
            return states
            
        # --- 步骤2: 基于量化得分，定义所有动态原子状态 ---
        # 定义一个辅助函数，避免重复代码
        def get_state(score_name: str, quantile: float):
            score = df[score_name]
            return score > score.rolling(120).quantile(quantile)

        # 高级动态
        states['CHIP_DYN_CONCENTRATING_RESONANCE_A'] = get_state('CHIP_SCORE_CONCENTRATION_RESONANCE', 0.90)
        states['OPP_CHIP_REVERSAL_GATHERING_B'] = get_state('CHIP_SCORE_CONCENTRATION_DIVERGENCE', 0.90)
        states['RISK_CHIP_DIVERGING_RESONANCE_A'] = get_state('CHIP_SCORE_CONCENTRATION_RESONANCE', 0.10) == False
        states['RISK_CHIP_REVERSAL_DIVERGING_B'] = get_state('CHIP_SCORE_CONCENTRATION_DIVERGENCE', 0.10) == False
        
        # 基础动态 (quantile=0.5 相当于判断得分是否高于中位数，等价于原来的 > 0)
        states['CHIP_DYN_CONCENTRATING'] = get_state('CHIP_SCORE_DYN_CONCENTRATING', 0.50)
        states['CHIP_DYN_DIVERGING'] = get_state('CHIP_SCORE_DYN_DIVERGING', 0.50)
        states['CHIP_DYN_S_ACCEL_CONCENTRATING'] = get_state('CHIP_SCORE_DYN_S_ACCEL_CONCENTRATING', 0.75) # 加速需要更强的信号
        states['CHIP_DYN_ACCEL_DIVERGING'] = get_state('CHIP_SCORE_DYN_ACCEL_DIVERGING', 0.75)
        
        states['CHIP_DYN_COST_RISING'] = get_state('CHIP_SCORE_DYN_COST_RISING', 0.50)
        states['CHIP_DYN_COST_ACCELERATING'] = get_state('CHIP_SCORE_DYN_COST_ACCELERATING', 0.85) # 加速需要更强的信号
        states['CHIP_DYN_COST_FALLING'] = get_state('CHIP_SCORE_DYN_COST_FALLING', 0.50)
        
        states['CHIP_DYN_WINNER_RATE_COLLAPSING'] = get_state('CHIP_SCORE_DYN_WINNER_RATE_COLLAPSING', 0.80) # 崩盘是严重事件
        states['CHIP_DYN_WINNER_RATE_ACCEL_COLLAPSING'] = get_state('CHIP_SCORE_DYN_WINNER_RATE_ACCEL_COLLAPSING', 0.75)
        
        states['CHIP_DYN_HEALTH_IMPROVING'] = get_state('CHIP_SCORE_DYN_HEALTH_IMPROVING', 0.60)
        states['CHIP_DYN_HEALTH_DETERIORATING'] = get_state('CHIP_SCORE_DYN_HEALTH_DETERIORATING', 0.60)
        
        print("            -> [动态筹码分析中心 V3.0] 分析完毕。")
        return states

    def diagnose_chip_opportunities(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V343.0 终极竣工版】高级筹码“机会”情报诊断模块
        - 核心升级: 所有机会信号均由评分中心内生计算的专属得分驱动，消除所有外部布尔依赖。
        """
        print("        -> [高级筹码机会诊断模块 V343.0] 启动...")
        states = {}
        # --- 1. 军备检查 ---
        required_scores = [
            'CHIP_SCORE_OPP_FAULT_REBIRTH', 'CHIP_SCORE_OPP_PROFIT_CUSHION',
            'CHIP_SCORE_DYN_WINNER_RATE_RISING', 'CHIP_SCORE_OPP_ABSORPTION_INTENSITY'
        ]
        if any(c not in df.columns for c in required_scores):
            missing = [s for s in required_scores if s not in df.columns]
            print(f"          -> [警告] 缺少机会诊断所需得分，模块跳过。缺失: {missing}")
            return {}
        # --- 机会1: S级 - 筹码断层新生 (基于内生得分) ---
        rebirth_score = df['CHIP_SCORE_OPP_FAULT_REBIRTH']
        # 这是一个事件信号，我们捕捉得分显著的时刻
        states['OPP_CHIP_FAULT_REBIRTH_S'] = rebirth_score > rebirth_score[rebirth_score > 0].rolling(60).quantile(0.95)
        # --- 机会2: A级 - 高利润安全垫 (基于得分) ---
        profit_cushion_score = df['CHIP_SCORE_OPP_PROFIT_CUSHION']
        high_cushion_threshold = profit_cushion_score.rolling(120).quantile(0.90)
        states['CHIP_STATE_HIGH_PROFIT_CUSHION'] = profit_cushion_score > high_cushion_threshold
        # --- 机会3: 获利盘持续上升 (基于得分) ---
        rising_score = df['CHIP_SCORE_DYN_WINNER_RATE_RISING']
        states['CHIP_DYN_WINNER_RATE_RISING'] = rising_score > rising_score.rolling(120).quantile(0.5)
        # --- 机会4: A级 - 主峰强力吸筹 (基于得分) ---
        absorption_score = df['CHIP_SCORE_OPP_ABSORPTION_INTENSITY']
        concentrating_score = df['CHIP_SCORE_DYN_CONCENTRATING']
        intense_absorption_score = absorption_score * concentrating_score
        states['OPP_CHIP_INTENSE_ABSORPTION_A'] = intense_absorption_score > intense_absorption_score.rolling(120).quantile(0.85)
        return states

    def diagnose_chip_risks_and_behaviors(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V342.0 量化上下文版】高级筹码“风险与行为”情报诊断模块
        - 核心升级: 机会信号的生成增加了量化的“安全上下文”得分作为过滤器，提高信号质量。
        """
        # print("        -> [高级筹码风险与行为诊断模块 V341.2 加速度增强版] 启动...")
        states = {}
        # --- 1. 军备检查 ---
        required_scores = [
            'CHIP_SCORE_BEHAVIOR_PROFIT_TAKING', 'CHIP_SCORE_BEHAVIOR_SELLING_EXHAUSTION',
            'CHIP_SCORE_BEHAVIOR_PANIC_CAPITULATION', 'CHIP_SCORE_CONTEXT_SAFE',
            'CHIP_SCORE_RISK_PANIC_FLEEING'
        ]
        if any(c not in df.columns for c in required_scores):
            missing = [s for s in required_scores if s not in df.columns]
            print(f"          -> [警告] 缺少风险与行为诊断所需得分，模块跳过。缺失: {missing}")
            return {}
            
        high_zone_score = self.strategy.atomic_states.get('CONTEXT_RISK_HIGH_LEVEL_ZONE', pd.Series(0, index=df.index)).astype(float)

        # --- 2. 基于量化得分定义风险与机会 ---
        profit_taking_score = df['CHIP_SCORE_BEHAVIOR_PROFIT_TAKING']
        exhaustion_score = df['CHIP_SCORE_BEHAVIOR_SELLING_EXHAUSTION']
        capitulation_score = df['CHIP_SCORE_BEHAVIOR_PANIC_CAPITULATION']
        safe_context_score = df['CHIP_SCORE_CONTEXT_SAFE']
        panic_fleeing_score = df['CHIP_SCORE_RISK_PANIC_FLEEING']

        # 2.1 定义风险等级 (基于得分 * 上下文)
        fleeing_score_in_high_zone = profit_taking_score * high_zone_score
        states['RISK_BEHAVIOR_WINNERS_FLEEING_C'] = fleeing_score_in_high_zone > fleeing_score_in_high_zone.rolling(120).quantile(0.60)
        states['RISK_BEHAVIOR_WINNERS_FLEEING_B'] = fleeing_score_in_high_zone > fleeing_score_in_high_zone.rolling(120).quantile(0.75)
        states['RISK_BEHAVIOR_WINNERS_FLEEING_A'] = fleeing_score_in_high_zone > fleeing_score_in_high_zone.rolling(120).quantile(0.90)

        # 2.2 定义S级风险 - 恐慌加速 (基于专属得分)
        states['RISK_BEHAVIOR_PANIC_FLEEING_S'] = panic_fleeing_score > panic_fleeing_score.rolling(120).quantile(0.95)
        
        # 2.3 定义机会信号 (基于得分 * 上下文)
        # A级机会 - 卖盘衰竭
        selling_exhaustion_opp_score = exhaustion_score * safe_context_score
        states['OPP_BEHAVIOR_SELLING_EXHAUSTION_A'] = selling_exhaustion_opp_score > selling_exhaustion_opp_score.rolling(120).quantile(0.90)

        # A级机会 - 恐慌盘割肉
        panic_capitulation_opp_score = capitulation_score * safe_context_score
        states['OPP_BEHAVIOR_PANIC_CAPITULATION_A'] = panic_capitulation_opp_score > panic_capitulation_opp_score.rolling(120).quantile(0.90)
        return states

    def diagnose_peak_formation_dynamics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V342.0 终极量化消费版】筹码峰“创世纪”模块
        - 核心重构: 彻底抛弃所有内部的硬编码阈值和复杂时序逻辑，全面转向消费由评分模块
                    生成的专属“筹码峰创世纪”情景得分，实现与系统其他部分的完美统一。
        """
        # print("        -> [筹码峰“创世纪”模块 V340.1 战略过滤版] 启动...")
        states = {}
        # --- 1. 军备检查 ---
        required_scores = [
            'PEAK_SCORE_OPP_FORTRESS_SUPPORT',
            'PEAK_SCORE_RISK_EXHAUSTION_TOP',
            'PEAK_SCORE_OPP_STEALTH_ACCUMULATION'
        ]
        if any(c not in df.columns for c in required_scores):
            missing = [s for s in required_scores if s not in df.columns]
            print(f"          -> [警告] 筹码峰创世纪模块缺少关键得分数据: {missing}，模块已跳过！")
            return {}
        # --- 2. 基于得分生成最终信号 ---
        # 定义一个辅助函数，用于识别得分序列中的高置信度事件
        def get_event_state(score_name: str, quantile: float):
            score = df[score_name]
            # 仅在得分大于0时进行判断，因为这些是事件驱动的得分
            return (score > 0) & (score > score[score > 0].rolling(60, min_periods=5).quantile(quantile))
        # S级机会: 堡垒支撑
        states['PEAK_DYN_FORTRESS_SUPPORT'] = get_event_state('PEAK_SCORE_OPP_FORTRESS_SUPPORT', 0.90)
        # S级风险: 衰竭顶
        states['PEAK_DYN_EXHAUSTION_TOP'] = get_event_state('PEAK_SCORE_RISK_EXHAUSTION_TOP', 0.90)
        # A级机会: 隐蔽吸筹
        states['PEAK_DYN_STEALTH_ACCUMULATION'] = get_event_state('PEAK_SCORE_OPP_STEALTH_ACCUMULATION', 0.85)
        return states

    def diagnose_peak_battle_dynamics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V6.1 完美闭环版】主峰攻防战诊断模块
        - 核心升级: 风险信号的上下文过滤逻辑已融入评分模块，本模块仅做纯粹的得分消费。
        """
        states = {}
        # --- 1. 军备检查 ---
        required_scores = [
            'CHIP_SCORE_OPP_PEAK_BATTLE_BREAKOUT',
            'CHIP_SCORE_RISK_PEAK_BATTLE_DISTRIBUTION_IN_ZONE'
        ]
        if any(c not in df.columns for c in required_scores):
            missing_cols = [c for c in required_scores if c not in df.columns]
            print(f"          -> [警告] 主峰攻防战模块缺少关键得分数据: {missing_cols}，模块已跳过！")
            return {}
        # --- 2. 基于得分生成最终信号 ---
        breakout_score = df['CHIP_SCORE_OPP_PEAK_BATTLE_BREAKOUT']
        states['OPP_PEAK_BATTLE_BREAKOUT_A'] = breakout_score > breakout_score.rolling(120).quantile(0.90)
        # 消费包含上下文的新得分
        distribution_score = df['CHIP_SCORE_RISK_PEAK_BATTLE_DISTRIBUTION_IN_ZONE']
        states['RISK_PEAK_BATTLE_DISTRIBUTION_A'] = distribution_score > distribution_score.rolling(120).quantile(0.90)
        return states

    def synthesize_prime_chip_opportunity(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 量化消费版】黄金筹码机会合成模块
        - 核心重构: 消费专属的`CHIP_SCORE_PRIME_OPPORTUNITY`得分，该得分已融合了
                    “筹码锁定质量”与“上涨初期”的战场环境，从而更精准地识别高确定性机会。
        """
        # print("        -> [黄金筹码机会合成模块 V1.0] 启动...")
        states = {}
        # --- 1. 军备检查 ---
        required_score = 'CHIP_SCORE_PRIME_OPPORTUNITY'
        if required_score not in df.columns:
            print(f"          -> [警告] 黄金机会模块缺少关键得分数据: {required_score}，模块已跳过！")
            return states
            
        # --- 2. 基于得分生成最终信号 ---
        prime_score = df[required_score]
        # 定义：黄金机会得分高于过去95%的时间，确认为S级机会
        prime_threshold = prime_score.rolling(120).quantile(0.95)
        states['CHIP_STRUCTURE_PRIME_OPPORTUNITY_S'] = prime_score > prime_threshold
        return states

    def diagnose_chip_price_divergence(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 量化消费版】筹码价格顶背离诊断模块
        - 核心重构: 消费专属的`CHIP_SCORE_RISK_PRICE_DIVERGENCE`得分，该得分已量化了
                    “价格新高程度”与“筹码发散强度”的背离程度，使风险识别更灵敏。
        """
        states = {}
        # --- 1. 军备检查 ---
        required_score = 'CHIP_SCORE_RISK_PRICE_DIVERGENCE'
        if required_score not in df.columns:
            print(f"          -> [警告] 价格顶背离模块缺少关键得分数据: {required_score}，模块已跳过！")
            return states
            
        # --- 2. 基于得分生成最终风险信号 ---
        divergence_score = df[required_score]
        # 定义：顶背离风险得分高于过去95%的时间，确认为S级风险
        divergence_threshold = divergence_score.rolling(120).quantile(0.95)
        states['RISK_CHIP_PRICE_DIVERGENCE'] = divergence_score > divergence_threshold
        return states

    # “静态筹码结构诊断”方法
    def diagnose_static_chip_structure(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.1 场景扩充版】静态筹码结构诊断模块
        - 核心职责: 诊断当前价格与成本分布的静态关系，以及筹码分布的形态特征。
        - 核心升级 (本次修改): 新增了“价格低于成本峰”的静态场景，为识别深度反转信号提供基础。
        """
        # print("            -> [静态筹码结构诊断模块 V1.1] 启动...")
        states = {}
        # default_series = pd.Series(False, index=df.index)
        
        # --- 1. 军备检查 ---
        required_scores = [
            'CHIP_SCORE_STATIC_PRICE_MOMENTUM', 'CHIP_SCORE_STATIC_COMPACTNESS', 
            'CHIP_SCORE_STRUCTURE_FORTRESS', 'CHIP_SCORE_SCENARIO_BATTLEZONE_PROXIMITY',
            'CHIP_SCORE_STATIC_PRICE_BELOW_PEAK'
        ]
        if any(s not in df.columns for s in required_scores):
            print("              -> [警告] 缺少诊断静态筹码结构所需得分，模块跳过。")
            return {}
        # --- 2. 价格与成本峰关系诊断 (完全基于得分) ---
        price_momentum_score = df['CHIP_SCORE_STATIC_PRICE_MOMENTUM']
        states['CHIP_STATE_PRICE_ABOVE_PEAK_COST'] = price_momentum_score > price_momentum_score.rolling(120).quantile(0.85)

        proximity_score = df['CHIP_SCORE_SCENARIO_BATTLEZONE_PROXIMITY']
        states['CHIP_STATE_PRICE_IN_PEAK_COST_ZONE'] = proximity_score > proximity_score.rolling(120).quantile(0.90) # 价格非常接近成本
        
        below_peak_score = df['CHIP_SCORE_STATIC_PRICE_BELOW_PEAK']
        states['CHIP_STATE_PRICE_BELOW_PEAK_COST'] = below_peak_score > below_peak_score.rolling(120).quantile(0.85)

        # --- 3. 筹码分布形态诊断 (基于得分) ---
        compactness_score = df['CHIP_SCORE_STATIC_COMPACTNESS']
        states['CHIP_STRUCTURE_HIGHLY_COMPACT'] = compactness_score > compactness_score.rolling(120).quantile(0.85)
        
        # --- 3.5. S级堡垒结构诊断 (基于得分) ---
        fortress_score = df['CHIP_SCORE_STRUCTURE_FORTRESS']
        states['CHIP_STRUCTURE_FORTRESS_S'] = fortress_score > fortress_score.rolling(120).quantile(0.95)

        # --- 4. 获利盘状态诊断 (基于得分) ---
        euphoria_score = self._calculate_normalized_score(df.get('total_winner_rate_D', pd.Series(50, index=df.index)), window=120, ascending=True)
        states['CHIP_STATE_UNIVERSAL_PROFIT'] = euphoria_score > 0.95
        print("            -> [静态筹码结构诊断模块 V1.1] 诊断完毕。")
        return states

    # “筹码行为诊断”方法
    def diagnose_chip_behavior_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 完美闭环版】筹码行为诊断模块
        - 核心升级: 恐慌杀跌的识别逻辑完全基于量化得分，消除了硬编码跌幅阈值。
        """
        print("            -> [筹码行为诊断模块 V2.0] 启动...")
        states = {}
        # --- 1. 军备检查 ---
        required_scores = ['CHIP_SCORE_BEHAVIOR_PROFIT_TAKING_INTENSITY', 'CHIP_SCORE_BEHAVIOR_PANIC_CAPITULATION']
        if any(s not in df.columns for s in required_scores):
            print("              -> [警告] 缺少诊断筹码行为所需得分，模块跳过。")
            return {}
        # --- 2. 获利盘行为诊断 (基于得分) ---
        pressure_score = df['CHIP_SCORE_BEHAVIOR_PROFIT_TAKING_INTENSITY']
        states['CHIP_BEHAVIOR_HIGH_PROFIT_TAKING_PRESSURE'] = pressure_score > pressure_score.rolling(120).quantile(0.85)
        # --- 3. 亏损盘行为诊断 (基于升级后的得分) ---
        capitulation_score = df['CHIP_SCORE_BEHAVIOR_PANIC_CAPITULATION']
        states['CHIP_BEHAVIOR_LOSER_CAPITULATION'] = capitulation_score > capitulation_score.rolling(120).quantile(0.90)
        return states

    def diagnose_strategic_scenarios(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】战略情景诊断模块 (静态 x 多维动态)
        - 核心职责: 融合静态筹码“场景”与短、中、长三维动态筹码“剧情”，生成具备高度战术背景的复合原子信号。
        - 收益: 将分析从“信号”提升至“情景”层面，能够更精准地解读主力意图，区分洗盘与出货、吸筹与反弹。
        """
        # print("        -> [战略情景诊断模块 V1.0] 启动...")
        states = {}
        # --- 1. 军备检查：确保所有必需的“战略情景得分”都存在 ---
        required_scores = [
            'CHIP_SCORE_SCENARIO_MAIN_WAVE_RESONANCE',
            'CHIP_SCORE_SCENARIO_BATTLEZONE_TURNING_POINT',
            'CHIP_SCORE_SCENARIO_HIGH_ALTITUDE_DISTRIBUTION_TRAP',
            'CHIP_SCORE_SCENARIO_FORTRESS_INTERNAL_COLLAPSE'
        ]
        if any(c not in df.columns for c in required_scores):
            missing_cols = [c for c in required_scores if c not in df.columns]
            print(f"          -> [警告] 战略情景模块缺少关键得分数据: {missing_cols}，模块已跳过！")
            return states
            
        # --- 2. 基于情景得分，生成最终的布尔信号 ---
        # === 机会情景 ===
        # S级机会: “主升浪共振”得分极高
        main_wave_score = df['CHIP_SCORE_SCENARIO_MAIN_WAVE_RESONANCE']
        states['SCENARIO_MAIN_WAVE_RESONANCE_S'] = main_wave_score > main_wave_score.rolling(120).quantile(0.95)

        # A级机会: “阵地战转折点”得分较高
        battlezone_score = df['CHIP_SCORE_SCENARIO_BATTLEZONE_TURNING_POINT']
        states['SCENARIO_BATTLEZONE_TURNING_POINT_A'] = battlezone_score > battlezone_score.rolling(120).quantile(0.90)
        
        # === 风险情景 ===
        # S级风险: “高位派发陷阱”得分极高
        trap_score = df['CHIP_SCORE_SCENARIO_HIGH_ALTITUDE_DISTRIBUTION_TRAP']
        states['SCENARIO_HIGH_ALTITUDE_DISTRIBUTION_TRAP_S'] = trap_score > trap_score.rolling(120).quantile(0.95)

        # A级风险: “堡垒内部瓦解”得分较高
        collapse_score = df['CHIP_SCORE_SCENARIO_FORTRESS_INTERNAL_COLLAPSE']
        states['SCENARIO_FORTRESS_INTERNAL_COLLAPSE_A'] = collapse_score > collapse_score.rolling(120).quantile(0.90)
        
        # === 中性观察情景 (基于得分) ===
        washout_score = df['CHIP_SCORE_SCENARIO_WASHOUT_BELOW_FORTRESS']
        states['SCENARIO_WASHOUT_BELOW_FORTRESS_N'] = washout_score > washout_score.rolling(120).quantile(0.80)

        return states

    def diagnose_static_multi_dynamic_scenarios(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】静态-多动态交叉验证模块 (补充)
        - 核心职责: 识别更多基于特定静态筹码场景与多维动态行为组合的战术信号。
        """
        # print("        -> [静态-多动态交叉验证模块(补充) V1.0] 启动...")
        states = {}
        required_scores = [
            'CHIP_SCORE_OPP_DEEP_WATER_REVERSAL',
            'CHIP_SCORE_OPP_ACCUMULATION_CONFIRMED',
            'CHIP_SCORE_RISK_EUPHORIA_TRAP'
        ]
        if any(c not in df.columns for c in required_scores):
            missing_cols = [c for c in required_scores if c not in df.columns]
            print(f"          -> [警告] 静态-多动态交叉验证(补充)模块缺少关键得分数据: {missing_cols}，模块已跳过！")
            return states
            
        # --- 2. 基于情景得分生成最终信号 ---
        # S级机会: 深水炸弹·绝望反转
        reversal_score = df['CHIP_SCORE_OPP_DEEP_WATER_REVERSAL']
        states['OPP_CHIP_DEEP_WATER_REVERSAL_S'] = reversal_score > reversal_score.rolling(120).quantile(0.98) # 极稀有，用98%分位

        # A级机会: 阵地战·吸筹确认
        accumulation_score = df['CHIP_SCORE_OPP_ACCUMULATION_CONFIRMED']
        states['OPP_CHIP_ACCUMULATION_CONFIRMED_A'] = accumulation_score > accumulation_score.rolling(120).quantile(0.90)

        # S级风险: 高位狂欢·亢奋陷阱
        trap_score = df['CHIP_SCORE_RISK_EUPHORIA_TRAP']
        states['RISK_CHIP_EUPHORIA_TRAP_S'] = trap_score > trap_score.rolling(120).quantile(0.95)
        return states

    def diagnose_static_multi_timeframe_scenarios(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.0 量化消费版】静态-多时间维度交叉验证模块 (终极)
        - 核心重构: 全面转向消费由评分模块生成的专属情景得分 (`CHIP_SCORE_*`)，
                    通过动态分位数阈值，识别出最高置信度的S级和A级战术信号。
        """
        # print("        -> [静态-多时间维度交叉验证模块(终极) V2.0] 启动...")
        states = {}
        # --- 1. 军备检查：确保所有必需的“终极情景得分”都存在 ---
        required_scores = [
            'CHIP_SCORE_OPP_BREAKTHROUGH',
            'CHIP_SCORE_RISK_COLLAPSE',
            'CHIP_SCORE_OPP_INFLECTION'
        ]
        if any(c not in df.columns for c in required_scores):
            missing_cols = [c for c in required_scores if c not in df.columns]
            print(f"          -> [警告] 终极模块缺少关键得分数据: {missing_cols}，模块已跳过！")
            return states
            
        # --- 2. 基于情景得分生成最终信号 ---
        # 信号1 (S级机会): “阵地战·协同突破”
        breakthrough_score = df['CHIP_SCORE_OPP_BREAKTHROUGH']
        states['OPP_STATIC_DYN_BREAKTHROUGH_S'] = breakthrough_score > breakthrough_score.rolling(120).quantile(0.95)

        # 信号2 (S级风险): “亢奋顶点·结构瓦解”
        collapse_score = df['CHIP_SCORE_RISK_COLLAPSE']
        states['RISK_STATIC_DYN_COLLAPSE_S'] = collapse_score > collapse_score.rolling(120).quantile(0.95)

        # 信号3 (A级机会): “绝望冰点·周期拐点”
        inflection_score = df['CHIP_SCORE_OPP_INFLECTION']
        states['OPP_STATIC_DYN_INFLECTION_A'] = inflection_score > inflection_score.rolling(120).quantile(0.90)
        return states

    def diagnose_quantitative_chip_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V2.0 量化评分中心版】筹码信号量化评分诊断模块
        - 核心职责: 将所有关键的筹码信号升级为0-1之间的连续得分，成为量化分析的核心枢纽。
        - 核心新增 (本次修改): 新增了“共振得分”和“背离得分”的计算，以量化多周期趋势的一致性与分歧。
        """
        # print("        -> [筹码信号量化评分模块 V2.0] 启动，正在深化信号层次...")
        # 辅助函数，用于安全地从 atomic_states 获取分数
        def _get_atomic_score(name: str, default: float = 0.5) -> pd.Series:
            return self.strategy.atomic_states.get(name, pd.Series(default, index=df.index))
        # --- 1. 多周期集中趋势动能得分 ---
        # 使用_calculate_normalized_score将斜率转换为0-1的得分，斜率越小（负得越多），得分越高
        score_5d = self._calculate_normalized_score(df.get('SLOPE_5_concentration_90pct_D', pd.Series(0, index=df.index)), window=120, ascending=False)
        score_21d = self._calculate_normalized_score(df.get('SLOPE_21_concentration_90pct_D', pd.Series(0, index=df.index)), window=120, ascending=False)
        score_55d = self._calculate_normalized_score(df.get('SLOPE_55_concentration_90pct_D', pd.Series(0, index=df.index)), window=120, ascending=False)
        # --- 2. 共振与背离得分 ---
        weights = {'short': 0.5, 'mid': 0.3, 'long': 0.2}
        df['CHIP_SCORE_CONCENTRATION_RESONANCE'] = (score_5d * weights['short'] + score_21d * weights['mid'] + score_55d * weights['long'])
        df['CHIP_SCORE_CONCENTRATION_DIVERGENCE'] = score_5d - score_55d
        # --- 3. 集中趋势动能得分 (Concentration Momentum Score) ---
        if 'SLOPE_5_concentration_90pct_D' in df.columns:
            df['CHIP_SCORE_CONCENTRATION_MOMENTUM'] = self._calculate_normalized_score(series=df['SLOPE_5_concentration_90pct_D'], window=120, ascending=False)
        # --- 4. 成本支撑强度得分 (Cost Support Score) ---
        if 'SLOPE_5_peak_cost_D' in df.columns:
            df['CHIP_SCORE_COST_SUPPORT_MOMENTUM'] = self._calculate_normalized_score(series=df['SLOPE_5_peak_cost_D'], window=120, ascending=True)
        # --- 5. 结构稳定性得分 (Structural Stability Score) ---
        required_cols = ['peak_control_ratio_D', 'peak_strength_ratio_D']
        if all(c in df.columns for c in required_cols):
            control_score = self._calculate_normalized_score(df['peak_control_ratio_D'], window=120)
            strength_score = self._calculate_normalized_score(df['peak_strength_ratio_D'], window=120)
            # 使用“次峰强度比”来量化“单峰纯度”，替代原有的 is_multi_peak_D 布尔值
            # peak_strength_ratio_D 越小，说明次峰越弱，单峰纯度越高。对于真单峰，该值为NaN，填充为0。
            # ascending=False 意味着值越小，得分越高。
            single_peak_purity_score = self._calculate_normalized_score(
                df.get('peak_strength_ratio_D', 0).fillna(0), 
                window=120, 
                ascending=False
            )
            df['CHIP_SCORE_STRUCTURE_STABILITY'] = (
                0.3 * control_score +
                0.3 * strength_score +
                0.4 * single_peak_purity_score
            )
        # --- 6. 派发风险得分 (Distribution Risk Score) ---
        risk_cols = ['SLOPE_5_concentration_90pct_D', 'SLOPE_5_turnover_from_winners_ratio_D']
        if all(c in df.columns for c in risk_cols):
            divergence_risk = self._calculate_normalized_score(df['SLOPE_5_concentration_90pct_D'], window=120, ascending=True)
            profit_taking_risk = self._calculate_normalized_score(df['SLOPE_5_turnover_from_winners_ratio_D'], window=120, ascending=True)
            df['CHIP_SCORE_DISTRIBUTION_RISK'] = (divergence_risk + profit_taking_risk) / 2
        # --- 7. 行为量化得分 ---
        behavior_cols = [
            'turnover_from_winners_ratio_D', 'SLOPE_5_turnover_from_winners_ratio_D',
            'SLOPE_21_turnover_from_winners_ratio_D', 'SLOPE_55_turnover_from_winners_ratio_D',
            'ACCEL_5_turnover_from_winners_ratio_D', 'turnover_from_losers_ratio_D'
        ]
        if all(c in df.columns for c in behavior_cols):
            # 7.1 获利盘抛压得分 (Profit Taking Pressure Score)
            # 综合评估短、中、长期的抛售趋势
            pressure_5d = self._calculate_normalized_score(df['SLOPE_5_turnover_from_winners_ratio_D'], window=120, ascending=True)
            pressure_21d = self._calculate_normalized_score(df['SLOPE_21_turnover_from_winners_ratio_D'], window=120, ascending=True)
            pressure_55d = self._calculate_normalized_score(df['SLOPE_55_turnover_from_winners_ratio_D'], window=120, ascending=True)
            df['CHIP_SCORE_BEHAVIOR_PROFIT_TAKING'] = (pressure_5d * 0.5 + pressure_21d * 0.3 + pressure_55d * 0.2)
            df['CHIP_SCORE_BEHAVIOR_SELLING_EXHAUSTION'] = self._calculate_normalized_score(df['ACCEL_5_turnover_from_winners_ratio_D'], window=120, ascending=False)
            # 7.3 恐慌盘投降得分 (Panic Capitulation Score) - 升级版
            # a. 量化下跌剧烈程度，替代 is_sharp_drop
            sharp_drop_score = self._calculate_normalized_score(df['pct_change_D'], window=60, ascending=False)
            # b. 结合亏损盘换手率计算最终得分
            loser_turnover_score = self._calculate_normalized_score(df['turnover_from_losers_ratio_D'], window=120, ascending=True)
            df['CHIP_SCORE_BEHAVIOR_PANIC_CAPITULATION'] = loser_turnover_score * sharp_drop_score
        # --- 8. 机会与静态结构得分 ---
        # 8.1 利润安全垫得分 (越高越好)
        if 'winner_profit_margin_D' in df.columns:
            df['CHIP_SCORE_OPP_PROFIT_CUSHION'] = self._calculate_normalized_score(df['winner_profit_margin_D'], window=120, ascending=True)
        # 8.2 主峰吸筹强度得分 (越高越好)
        if 'peak_absorption_intensity_D' in df.columns:
            df['CHIP_SCORE_OPP_ABSORPTION_INTENSITY'] = self._calculate_normalized_score(df['peak_absorption_intensity_D'], window=120, ascending=True)
        # 8.3 价格动能得分 (价格相对成本峰越高，得分越高)
        if 'price_to_peak_ratio_D' in df.columns:
            df['CHIP_SCORE_STATIC_PRICE_MOMENTUM'] = self._calculate_normalized_score(df['price_to_peak_ratio_D'], window=120, ascending=True)
        # 8.4 结构紧凑度得分 (c90-c70的差距越小，得分越高)
        if 'concentration_90pct_D' in df.columns and 'concentration_70pct_D' in df.columns:
            concentration_gap = df['concentration_90pct_D'] - df['concentration_70pct_D']
            df['CHIP_SCORE_STATIC_COMPACTNESS'] = self._calculate_normalized_score(concentration_gap, window=120, ascending=False)
        # --- 9. 战略情景得分 (Scenario Conviction Scores) ---
        # 9.1 创建情景所需的基础得分
        # 看涨背离得分 (短期趋势好于长期)
        bullish_reversal_score = df.get('CHIP_SCORE_CONCENTRATION_DIVERGENCE', pd.Series(0, index=df.index)).clip(lower=0)
        # 看跌背离得分 (短期趋势差于长期)
        bearish_reversal_score = (-df.get('CHIP_SCORE_CONCENTRATION_DIVERGENCE', pd.Series(0, index=df.index))).clip(lower=0)
        # 利润垫抬升得分
        profit_margin_rising_score = self._calculate_normalized_score(df.get('SLOPE_5_winner_profit_margin_D', pd.Series(0, index=df.index)), window=120, ascending=True)
        # 利润垫收缩得分
        profit_margin_shrinking_score = self._calculate_normalized_score(df.get('SLOPE_5_winner_profit_margin_D', pd.Series(0, index=df.index)), window=120, ascending=False)
        # 成本下降得分
        cost_falling_score = self._calculate_normalized_score(df.get('SLOPE_5_peak_cost_D', pd.Series(0, index=df.index)), window=120, ascending=False)
        # 阵地战邻近得分 (价格越接近成本峰，得分越高)
        if 'price_to_peak_ratio_D' in df.columns:
            proximity = 1 - (df['price_to_peak_ratio_D'] - 1).abs()
            df['CHIP_SCORE_SCENARIO_BATTLEZONE_PROXIMITY'] = proximity.clip(lower=0)
        # 9.2 计算复合情景得分 (使用乘法融合各维度得分)
        df['CHIP_SCORE_SCENARIO_MAIN_WAVE_RESONANCE'] = (
            df.get('CHIP_SCORE_STRUCTURE_STABILITY', 0.5) *
            df.get('CHIP_SCORE_CONCENTRATION_RESONANCE', 0.5) *
            df.get('CHIP_SCORE_COST_SUPPORT_MOMENTUM', 0.5) *
            profit_margin_rising_score
        )
        df['CHIP_SCORE_SCENARIO_BATTLEZONE_TURNING_POINT'] = (
            df.get('CHIP_SCORE_SCENARIO_BATTLEZONE_PROXIMITY', 0) *
            bullish_reversal_score *
            df.get('CHIP_SCORE_BEHAVIOR_SELLING_EXHAUSTION', 0.5)
        )
        df['CHIP_SCORE_SCENARIO_HIGH_ALTITUDE_DISTRIBUTION_TRAP'] = (
            df.get('CHIP_SCORE_STATIC_PRICE_MOMENTUM', 0.5) *
            (1 - df.get('CHIP_SCORE_CONCENTRATION_RESONANCE', 0.5)) * # 派发是集中的反面
            profit_margin_shrinking_score
        )
        df['CHIP_SCORE_SCENARIO_FORTRESS_INTERNAL_COLLAPSE'] = (
            df.get('CHIP_SCORE_STRUCTURE_STABILITY', 0.5) *
            bearish_reversal_score *
            cost_falling_score
        )
        # --- 10. 复合状态与交叉验证得分 ---
        # 10.1 筹码锁定稳定得分 (A级状态)
        concentration_score = self._calculate_normalized_score(df.get('concentration_90pct_D', pd.Series(0, index=df.index)), window=120, ascending=False)
        cost_stability_score = self._calculate_normalized_score(df.get('SLOPE_5_peak_cost_D', pd.Series(0, index=df.index)).abs(), window=120, ascending=False)
        df['CHIP_SCORE_STRUCTURE_LOCKED_STABLE'] = concentration_score * cost_stability_score
        # 10.2 行为强度得分 (替代硬编码阈值)
        if 'turnover_from_winners_ratio_D' in df.columns:
            df['CHIP_SCORE_BEHAVIOR_PROFIT_TAKING_INTENSITY'] = self._calculate_normalized_score(df['turnover_from_winners_ratio_D'], window=120, ascending=True)
        if 'turnover_from_losers_ratio_D' in df.columns:
            df['CHIP_SCORE_BEHAVIOR_LOSER_CAPITULATION_INTENSITY'] = self._calculate_normalized_score(df['turnover_from_losers_ratio_D'], window=120, ascending=True)
        # 10.3 静态-多动态交叉验证情景得分
        # S级机会: 深水炸弹·绝望反转
        despair_zone_score = self._calculate_normalized_score(df.get('price_to_peak_ratio_D', pd.Series(1, index=df.index)), window=120, ascending=False)
        bleeding_stopping_score = self._calculate_normalized_score(df.get('ACCEL_5_turnover_from_losers_ratio_D', pd.Series(0, index=df.index)), window=120, ascending=False)
        stealth_buying_score = self._calculate_normalized_score(df.get('SLOPE_5_concentration_90pct_D', pd.Series(0, index=df.index)), window=120, ascending=False)
        price_turning_score = self._calculate_normalized_score(df.get('SLOPE_5_close_D', pd.Series(0, index=df.index)), window=120, ascending=True)
        df['CHIP_SCORE_OPP_DEEP_WATER_REVERSAL'] = despair_zone_score * bleeding_stopping_score * stealth_buying_score * price_turning_score
        # A级机会: 阵地战·吸筹确认
        absorption_accel_score = self._calculate_normalized_score(df.get('ACCEL_5_concentration_90pct_D', pd.Series(0, index=df.index)), window=120, ascending=False)
        losers_giving_up_score = self._calculate_normalized_score(df.get('SLOPE_5_turnover_from_losers_ratio_D', pd.Series(0, index=df.index)), window=120, ascending=False)
        df['CHIP_SCORE_OPP_ACCUMULATION_CONFIRMED'] = df.get('CHIP_SCORE_SCENARIO_BATTLEZONE_PROXIMITY', 0) * absorption_accel_score * losers_giving_up_score
        # S级风险: 高位狂欢·亢奋陷阱
        euphoria_score = self._calculate_normalized_score(df.get('total_winner_rate_D', pd.Series(50, index=df.index)), window=120, ascending=True)
        winner_selling_accel_score = self._calculate_normalized_score(df.get('ACCEL_5_turnover_from_winners_ratio_D', pd.Series(0, index=df.index)), window=120, ascending=True)
        chip_diverging_score = self._calculate_normalized_score(df.get('SLOPE_5_concentration_90pct_D', pd.Series(0, index=df.index)), window=120, ascending=True)
        price_momentum_fading_score = self._calculate_normalized_score(df.get('ACCEL_5_close_D', pd.Series(0, index=df.index)), window=120, ascending=False)
        df['CHIP_SCORE_RISK_EUPHORIA_TRAP'] = euphoria_score * winner_selling_accel_score * chip_diverging_score * price_momentum_fading_score
        # --- 11. 终极交叉验证与复合机会得分 ---
        # 11.1 S级机会: “阵地战·协同突破”得分
        price_accel_score = self._calculate_normalized_score(df.get('ACCEL_5_close_D', pd.Series(0, index=df.index)), window=120, ascending=True)
        mid_term_price_trend_score = self._calculate_normalized_score(df.get('SLOPE_21_close_D', pd.Series(0, index=df.index)), window=120, ascending=True)
        short_term_concentration_score = self._calculate_normalized_score(df.get('SLOPE_5_concentration_90pct_D', pd.Series(0, index=df.index)), window=120, ascending=False)
        df['CHIP_SCORE_OPP_BREAKTHROUGH'] = df.get('CHIP_SCORE_SCENARIO_BATTLEZONE_PROXIMITY', 0) * price_accel_score * mid_term_price_trend_score * short_term_concentration_score
        # 11.2 S级风险: “亢奋顶点·结构瓦解”得分
        short_term_price_trend_score = self._calculate_normalized_score(df.get('SLOPE_5_close_D', pd.Series(0, index=df.index)), window=120, ascending=True)
        mid_term_divergence_score = self._calculate_normalized_score(df.get('SLOPE_21_concentration_90pct_D', pd.Series(0, index=df.index)), window=120, ascending=True)
        short_term_divergence_score = self._calculate_normalized_score(df.get('SLOPE_5_concentration_90pct_D', pd.Series(0, index=df.index)), window=120, ascending=True)
        df['CHIP_SCORE_RISK_COLLAPSE'] = euphoria_score * short_term_price_trend_score * mid_term_divergence_score * short_term_divergence_score
        # 11.3 A级机会: “绝望冰点·周期拐点”得分 (升级版)
        mid_term_price_accel_score = self._calculate_normalized_score(df.get('ACCEL_21_close_D', pd.Series(0, index=df.index)), window=120, ascending=True)
        # 将二元化的 was_in_downtrend_yesterday 替换为连续的 prior_downtrend_strength_score
        prior_downtrend_strength_score = self._calculate_normalized_score(df.get('SLOPE_21_close_D', 0).shift(1), window=60, ascending=False)
        df['CHIP_SCORE_OPP_INFLECTION'] = despair_zone_score * mid_term_price_accel_score * short_term_price_trend_score * prior_downtrend_strength_score
        high_zone_score = _get_atomic_score('COGNITIVE_SCORE_RISK_HIGH_LEVEL_ZONE')
        # 11.4 S级机会: “黄金筹码机会”得分
        # 注意: CONTEXT_TREND_STAGE_EARLY 来自外部模块，这里假设它是一个0/1的Series
        early_stage_score = _get_atomic_score('COGNITIVE_SCORE_TREND_STAGE_EARLY')
        df['CHIP_SCORE_PRIME_OPPORTUNITY'] = df.get('CHIP_SCORE_STRUCTURE_LOCKED_STABLE', 0.5) * early_stage_score
        # 11.5 S级风险: “价格筹码顶背离”得分
        new_high_score = self._calculate_normalized_score(df.get('close_D', pd.Series(0, index=df.index)), window=60, ascending=True) # 用60日价格分位数代表“新高”程度
        df['CHIP_SCORE_RISK_PRICE_DIVERGENCE'] = new_high_score * short_term_divergence_score
        # --- 12. 终极战术情景得分 ---
        # 12.1 S级机会: “锁仓点火”得分
        cost_accelerating_score = self._calculate_normalized_score(df.get('ACCEL_5_peak_cost_D', pd.Series(0, index=df.index)), window=120, ascending=True)
        df['CHIP_SCORE_OPP_LOCKED_IGNITION'] = df.get('CHIP_SCORE_STRUCTURE_LOCKED_STABLE', 0.5) * cost_accelerating_score
        # 12.2 主峰攻防战得分
        battle_intensity_score = self._calculate_normalized_score(df.get('turnover_at_peak_ratio_D', pd.Series(0, index=df.index)), window=120, ascending=True)
        high_volume_score = self._calculate_normalized_score(df.get('volume_D', pd.Series(0, index=df.index)) / df.get('VOL_MA_21_D', pd.Series(1, index=df.index)), window=120, ascending=True)
        # 攻防战“准备”阶段的质量得分
        battle_setup_score = (
            df.get('CHIP_SCORE_SCENARIO_BATTLEZONE_PROXIMITY', 0) *
            battle_intensity_score *
            high_volume_score
        )
        # A级机会: “主峰换手突破”得分 (需要近期的“准备”+当日的“确认”)
        price_rising_meaningfully_score = self._calculate_normalized_score(df.get('pct_change_D', pd.Series(0, index=df.index)), window=120, ascending=True)
        had_recent_setup_score = battle_setup_score.shift(1).rolling(window=3, min_periods=1).max()
        df['CHIP_SCORE_OPP_PEAK_BATTLE_BREAKOUT'] = had_recent_setup_score * price_rising_meaningfully_score
        # A级风险: “主峰高位派发”得分
        price_stagnant_score = self._calculate_normalized_score(df.get('pct_change_D', pd.Series(0, index=df.index)), window=120, ascending=False) # 涨不动得分高
        df['CHIP_SCORE_RISK_PEAK_BATTLE_DISTRIBUTION'] = battle_setup_score * price_stagnant_score
        # --- 13. 核心过程与复合风险量化 ---
        # 13.1 筹码聚集强度得分 (用于C/B/B+级信号)
        concentration_accel_score = self._calculate_normalized_score(df.get('ACCEL_5_concentration_90pct_D', pd.Series(0, index=df.index)), window=120, ascending=False)
        df['CHIP_SCORE_GATHERING_INTENSITY'] = df.get('CHIP_SCORE_CONCENTRATION_MOMENTUM', 0.5) * concentration_accel_score
        # 13.2 长期派发风险得分
        # a. 21日集中度恶化程度得分
        conc_worsened_ratio = df.get('concentration_90pct_D', 1) / df.get('concentration_90pct_D', 1).shift(21)
        conc_worsened_score = self._calculate_normalized_score(conc_worsened_ratio, window=120, ascending=True) # ratio > 1 is bad
        # b. 21日斜率发散得分
        slope_21d_diverging_score = self._calculate_normalized_score(df.get('SLOPE_21_concentration_90pct_D', 0), window=120, ascending=True)
        # c. 高位区风险得分 (假设 CONTEXT_RISK_HIGH_LEVEL_ZONE 是 0/1)
        df['CHIP_SCORE_RISK_LONG_TERM_DISTRIBUTION'] = ((conc_worsened_score + slope_21d_diverging_score) / 2) * high_zone_score
        # 13.3 盘整突破前兆得分 (A级机会)
        df['CHIP_SCORE_OPP_BREAKOUT_PRECURSOR'] = df.get('CHIP_SCORE_SCENARIO_BATTLEZONE_PROXIMITY', 0) * df.get('CHIP_SCORE_CONCENTRATION_RESONANCE', 0.5)
        # 13.4 高位派发确认得分 (S级风险)
        df['CHIP_SCORE_RISK_DISTRIBUTION_CONFIRMED'] = df.get('CHIP_SCORE_STATIC_PRICE_MOMENTUM', 0.5) * (1 - df.get('CHIP_SCORE_CONCENTRATION_RESONANCE', 0.5))
        # 13.5 主力自救失败得分 (A级风险)
        df['CHIP_SCORE_RISK_BAILOUT_FAILURE'] = df.get('CHIP_SCORE_SCENARIO_BATTLEZONE_PROXIMITY', 0) * short_term_divergence_score
        # --- 14. 终极信号量化 (收官) ---
        # 14.1 S级堡垒结构得分
        control_score = self._calculate_normalized_score(df.get('peak_control_ratio_D', 0), window=120)
        strength_score = self._calculate_normalized_score(df.get('peak_strength_ratio_D', 0), window=120)
        single_peak_score = self._calculate_normalized_score(
            df.get('peak_strength_ratio_D', 0).fillna(0), 
            window=120, 
            ascending=False
        )
        df['CHIP_SCORE_STRUCTURE_FORTRESS'] = control_score * strength_score * single_peak_score
        # 14.2 锁仓突破得分
        # a. 突破K线质量分 (综合考虑涨幅和放量)
        breakout_candle_score = price_rising_meaningfully_score * high_volume_score
        # b. 最终得分
        df['CHIP_SCORE_OPP_LOCKED_BREAKOUT'] = df.get('CHIP_SCORE_STRUCTURE_LOCKED_STABLE', 0.5) * breakout_candle_score
        # 14.3 点火信号得分
        df['CHIP_SCORE_TRIGGER_IGNITION'] = self._calculate_normalized_score(df.get('ACCEL_5_peak_cost_D', 0), window=120, ascending=True)
        # 14.4 终极风险裁定得分 (V2.0 状态型升级版)
        winner_rate_collapse_score = self._calculate_normalized_score(df.get('SLOPE_5_total_winner_rate_D', 0), window=120, ascending=False)
        health_deterioration_score = self._calculate_normalized_score(df.get('SLOPE_5_chip_health_score_D', 0), window=120, ascending=False)
        # 将“事件型”的拐点检测，升级为“状态型”的恶化强度量化
        conc_accel_21d = df.get('ACCEL_21_concentration_90pct_D', pd.Series(0, index=df.index))
        # 只保留加速恶化的部分(>0)，将其作为风险强度的原始度量
        worsening_intensity = conc_accel_21d.clip(lower=0)
        # 对这个“恶化强度”进行归一化，得到最终的风险得分
        df['CHIP_SCORE_RISK_WORSENING_TURN'] = self._calculate_normalized_score(worsening_intensity, window=120, ascending=True)
        fleeing_b_score = self._calculate_normalized_score(df.get('CHIP_SCORE_BEHAVIOR_PROFIT_TAKING', 0.5), window=120, ascending=True)
        fleeing_accel_score = (1 - df.get('CHIP_SCORE_BEHAVIOR_SELLING_EXHAUSTION', 0.5))
        df['CHIP_SCORE_RISK_PANIC_FLEEING'] = fleeing_b_score * fleeing_accel_score
        # 重新构建终极风险得分DataFrame
        short_term_divergence_score = self._calculate_normalized_score(df.get('SLOPE_5_concentration_90pct_D', pd.Series(0, index=df.index)), window=120, ascending=True)
        cost_falling_score = self._calculate_normalized_score(df.get('SLOPE_5_peak_cost_D', pd.Series(0, index=df.index)), window=120, ascending=False)
        risk_scores_df = pd.DataFrame({
            'risk_1': short_term_divergence_score,
            'risk_2': cost_falling_score,
            'risk_3': winner_rate_collapse_score,
            'risk_4': df.get('CHIP_SCORE_RISK_LONG_TERM_DISTRIBUTION', 0),
            'risk_5': df.get('CHIP_SCORE_RISK_WORSENING_TURN', 0), # 使用新的、更稳健的“状态型”得分
            'risk_6': df.get('CHIP_SCORE_RISK_PANIC_FLEEING', 0),
            'risk_7': df.get('CHIP_SCORE_RISK_PRICE_DIVERGENCE', 0),
            'risk_8': health_deterioration_score,
            'risk_9': df.get('CHIP_SCORE_RISK_DISTRIBUTION_CONFIRMED', 0),
            'risk_10': df.get('CHIP_SCORE_RISK_BAILOUT_FAILURE', 0),
            'risk_11': df.get('CHIP_SCORE_SCENARIO_HIGH_ALTITUDE_DISTRIBUTION_TRAP', 0),
            'risk_12': df.get('CHIP_SCORE_SCENARIO_FORTRESS_INTERNAL_COLLAPSE', 0)
        })
        df['CHIP_SCORE_RISK_CRITICAL_FAILURE'] = risk_scores_df.max(axis=1)
        # --- 15. 基础动态与上下文得分 (基石) ---
        # 15.1 宏观战略背景得分
        df['CHIP_SCORE_CONTEXT_STRATEGIC_GATHERING'] = score_55d # 直接复用55日集中趋势得分
        # 15.2 基础动态原子得分
        cost_constructive_score = self._calculate_normalized_score(df.get('SLOPE_5_peak_cost_D', 0), window=60, ascending=True)
        washout_score = _get_atomic_score('BEHAVIOR_SCORE_OPP_WASHOUT_ABSORPTION')
        bottoming_score = _get_atomic_score('SCORE_MA_STATE_BOTTOM_PASSIVATION')
        concentrating_score_1 = df.get('CHIP_SCORE_CONCENTRATION_MOMENTUM', 0.5) * cost_constructive_score
        concentrating_score_2 = washout_score
        concentrating_score_3 = df.get('CHIP_SCORE_CONCENTRATION_MOMENTUM', 0.5) * bottoming_score
        df['CHIP_SCORE_DYN_CONCENTRATING'] = pd.concat([concentrating_score_1, concentrating_score_2, concentrating_score_3], axis=1).max(axis=1)
        df['CHIP_SCORE_DYN_DIVERGING'] = (1 - df.get('CHIP_SCORE_CONCENTRATION_MOMENTUM', 0.5)) * high_zone_score
        df['CHIP_SCORE_DYN_S_ACCEL_CONCENTRATING'] = df.get('CHIP_SCORE_DYN_CONCENTRATING', 0.5) * concentration_accel_score
        df['CHIP_SCORE_DYN_ACCEL_DIVERGING'] = self._calculate_normalized_score(df.get('ACCEL_5_concentration_90pct_D', 0), window=120, ascending=True) * high_zone_score
        df['CHIP_SCORE_DYN_COST_RISING'] = df.get('CHIP_SCORE_COST_SUPPORT_MOMENTUM', 0.5)
        df['CHIP_SCORE_DYN_COST_ACCELERATING'] = df.get('CHIP_SCORE_TRIGGER_IGNITION', 0.5)
        df['CHIP_SCORE_DYN_COST_FALLING'] = cost_falling_score
        df['CHIP_SCORE_DYN_WINNER_RATE_RISING'] = self._calculate_normalized_score(df.get('SLOPE_5_total_winner_rate_D', 0), window=120, ascending=True)
        df['CHIP_SCORE_DYN_WINNER_RATE_COLLAPSING'] = winner_rate_collapse_score
        df['CHIP_SCORE_DYN_WINNER_RATE_ACCEL_COLLAPSING'] = self._calculate_normalized_score(df.get('ACCEL_5_total_winner_rate_D', 0), window=120, ascending=False)
        health_improving_score = self._calculate_normalized_score(df.get('SLOPE_5_chip_health_score_D', 0), window=120, ascending=True)
        df['CHIP_SCORE_DYN_HEALTH_IMPROVING'] = health_improving_score * df.get('CHIP_SCORE_DYN_CONCENTRATING', 0.5)
        df['CHIP_SCORE_DYN_HEALTH_DETERIORATING'] = health_deterioration_score * df.get('CHIP_SCORE_DYN_DIVERGING', 0.5)
        # 15.3 上下文过滤器得分
        uptrend_context_score = self._calculate_normalized_score(df.get('close_D', 1) / df.get('EMA_55_D', 1), window=120, ascending=True)
        df['CHIP_SCORE_CONTEXT_UPTREND'] = uptrend_context_score.clip(lower=0)
        early_reversal_score = _get_atomic_score('SCORE_STRUCTURE_EARLY_REVERSAL')
        df['CHIP_SCORE_CONTEXT_SAFE'] = pd.concat([bottoming_score, early_reversal_score], axis=1).max(axis=1)
        # --- 16. 筹码峰“创世纪”过程量化 (V3.0 终极纯化版) ---
        # 16.1 计算基础变化指标
        peak_cost_series = df.get('peak_cost_D', pd.Series(np.nan, index=df.index))
        peak_change_pct = peak_cost_series.pct_change().abs().fillna(0)
        # 16.2 量化“新峰形成”的各个维度
        # a. 变化强度得分
        peak_change_intensity_score = self._calculate_normalized_score(peak_change_pct, window=60, ascending=True)
        # b. 形成过程稳定性得分 (使用近期波动率作为代理)
        peak_cost_volatility = peak_cost_series.rolling(5).std() / peak_cost_series.rolling(5).mean()
        formation_stability_score = self._calculate_normalized_score(peak_cost_volatility.fillna(1), window=60, ascending=False)
        # c. 形成过程持续度得分 (V2.0 纯化版)
        # 废除 is_stable_day 的布尔判断，改为量化“近期平均变化率”
        # 近期平均变化率越低，代表持续稳定性越高
        avg_instability = peak_change_pct.rolling(10, min_periods=3).mean()
        formation_duration_score = self._calculate_normalized_score(avg_instability.fillna(1), window=60, ascending=False)
        # 16.3 合成“新峰形成置信度”得分
        formation_confidence_score = (
            peak_change_intensity_score * 
            formation_stability_score * 
            formation_duration_score
        )
        # 16.4 基于新的置信度得分，计算最终的“创世纪”情景得分
        vol_ratio = df.get('volume_D', 1) / df.get('VOL_MA_21_D', 1)
        high_volume_score = self._calculate_normalized_score(vol_ratio, window=60, ascending=True)
        low_volume_score = 1 - high_volume_score
        prior_trend_slope = df.get('SLOPE_55_EMA_55_D', pd.Series(0, index=df.index)).shift(1)
        prior_downtrend_score = self._calculate_normalized_score(prior_trend_slope, window=60, ascending=False)
        prior_uptrend_score = self._calculate_normalized_score(prior_trend_slope, window=60, ascending=True)
        df['PEAK_SCORE_OPP_FORTRESS_SUPPORT'] = formation_confidence_score * high_volume_score * prior_downtrend_score
        df['PEAK_SCORE_RISK_EXHAUSTION_TOP'] = formation_confidence_score * high_volume_score * prior_uptrend_score
        df['PEAK_SCORE_OPP_STEALTH_ACCUMULATION'] = formation_confidence_score * low_volume_score * prior_downtrend_score * df.get('CHIP_SCORE_CONTEXT_SAFE', 0.5)
        # --- 17. 最终抛光：量化剩余的微观逻辑 ---
        # 17.1 静态价格区间得分
        df['CHIP_SCORE_STATIC_PRICE_BELOW_PEAK'] = self._calculate_normalized_score(df.get('price_to_peak_ratio_D', 1.0), window=120, ascending=False)
        # 17.2 主峰高位派发风险（含高位区上下文）
        df['CHIP_SCORE_RISK_PEAK_BATTLE_DISTRIBUTION_IN_ZONE'] = df.get('CHIP_SCORE_RISK_PEAK_BATTLE_DISTRIBUTION', 0.5) * high_zone_score
        # 17.3 中性场景：堡垒下洗盘得分
        compactness_score = df.get('CHIP_SCORE_STATIC_COMPACTNESS', 0.5)
        diverging_short_score = self._calculate_normalized_score(df.get('SLOPE_5_concentration_90pct_D', 0), window=120, ascending=True)
        not_diverging_resonance_score = df.get('CHIP_SCORE_CONCENTRATION_RESONANCE', 0.5) # 共振分越高，越不可能是共振派发
        df['CHIP_SCORE_SCENARIO_WASHOUT_BELOW_FORTRESS'] = compactness_score * diverging_short_score * not_diverging_resonance_score
        # 17.4 风险拐点事件得分
        conc_accel_21d = df.get('ACCEL_21_concentration_90pct_D', pd.Series(0, index=df.index))
        is_turn_event = (conc_accel_21d > 0) & (conc_accel_21d.shift(1) <= 0)
        # 量化拐点的强度（即当日加速度的大小）
        turn_magnitude = conc_accel_21d.where(is_turn_event, 0)
        df['CHIP_SCORE_RISK_WORSENING_TURN'] = self._calculate_normalized_score(turn_magnitude, window=120, ascending=True)
        # --- 18. 终极竣工：内生化最后的机会信号得分 ---
        # 18.1 量化“筹码断层新生”事件得分，替代 is_chip_fault_formed_D 的布尔逻辑
        # a. 断层强度得分 (强度越大，得分越高)
        fault_strength_score = self._calculate_normalized_score(
            df.get('chip_fault_strength_D', 0).fillna(0), 
            window=120, 
            ascending=True
        )
        # b. 真空区“清澈度”得分 (真空区筹码占比越小，得分越高)
        vacuum_clearance_score = self._calculate_normalized_score(
            df.get('chip_fault_vacuum_percent_D', 100).fillna(100), 
            window=120, 
            ascending=False
        )
        # c. 最终得分是强度、清澈度、以及趋势背景的乘积
        df['CHIP_SCORE_OPP_FAULT_REBIRTH'] = (
            fault_strength_score * 
            vacuum_clearance_score * 
            df.get('CHIP_SCORE_CONTEXT_UPTREND', 0.5)
        )
        return df

    def _calculate_normalized_score(self, series: pd.Series, window: int, ascending: bool = True) -> pd.Series:
        """
        【新增】计算滚动归一化得分的辅助函数。
        使用滚动分位数排名，将一个指标转换为0-1之间的得分。
        :param series: 原始数据Series。
        :param window: 滚动窗口大小。
        :param ascending: 排序方向。True表示值越大得分越高，False反之。
        :return: 归一化后的得分Series。
        """
        # 使用rank(pct=True)计算每个值在滚动窗口内的百分位排名
        # min_periods设为窗口的1/4，以在数据初期尽快产生有效值
        ranked = series.rolling(window=window, min_periods=window // 4).rank(pct=True)
        
        # 根据ascending参数决定最终得分
        # 如果是降序（值越小越好，如筹码集中度），则用1减去排名
        if not ascending:
            score = 1 - ranked
        else:
            score = ranked
            
        # 用0.5（中性值）填充无法计算的NaN值
        return score.fillna(0.5)





