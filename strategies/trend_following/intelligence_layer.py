# 文件: strategies/trend_following/intelligence_layer.py
# 情报层总指挥官 (重构版)
import pandas as pd
import numpy as np
from typing import Dict
from enum import Enum
from scipy.stats import linregress

from strategies.kline_pattern_recognizer import KlinePatternRecognizer
from .utils import get_params_block, get_param_value, create_persistent_state

# --- 从新目录导入所有情报模块 ---
from .intelligence.foundation_intelligence import FoundationIntelligence
from .intelligence.structural_intelligence import StructuralIntelligence
from .intelligence.chip_intelligence import ChipIntelligence
from .intelligence.behavioral_intelligence import BehavioralIntelligence
from .intelligence.cognitive_intelligence import CognitiveIntelligence
from .intelligence.playbook_engine import PlaybookEngine
from .intelligence.fund_flow_intelligence import FundFlowIntelligence
from .intelligence.dynamic_mechanics_engine import DynamicMechanicsEngine
from .intelligence.cyclical_intelligence import CyclicalIntelligence

class MainForceState(Enum):
    """
    定义主力行为序列的各个状态。
    """
    IDLE = 0           # 闲置/观察期
    ACCUMULATING = 1   # 吸筹期
    WASHING = 2        # 洗盘期
    MARKUP = 3         # 拉升期
    DISTRIBUTING = 4   # 派发期
    COLLAPSE = 5       # 崩盘期

class IntelligenceLayer:
    """
    【V400.0 重构版】情报层总指挥官
    - 核心职责: 1. 实例化所有专业化的情报子模块 (如筹码、结构、行为等)。
                2. 按照正确的依赖顺序，编排和调用这些子模块。
                3. 整合所有模块产出的原子状态和触发器，供下游层使用。
    - 设计原则: 高内聚、低耦合。本类只负责“编排”，不负责具体的“诊断”逻辑。
    """
    def __init__(self, strategy_instance):
        """
        初始化情报层总指挥官。
        """
        self.strategy = strategy_instance
        self.kline_params = get_params_block(self.strategy, 'kline_pattern_params')
        self.pattern_recognizer = KlinePatternRecognizer(params=self.kline_params)
        self.strategy.pattern_recognizer = self.pattern_recognizer # 将识别器传递给策略实例，供其他模块使用

        # 计算动态阈值，并传递给需要的子模块
        self.dynamic_thresholds = self._get_dynamic_thresholds(self.strategy.df_indicators)

        # 实例化所有子模块，注入依赖
        self.foundation_intel = FoundationIntelligence(self.strategy)
        self.structural_intel = StructuralIntelligence(self.strategy, self.dynamic_thresholds)
        self.chip_intel = ChipIntelligence(self.strategy, self.dynamic_thresholds)
        self.behavioral_intel = BehavioralIntelligence(self.strategy)
        self.cognitive_intel = CognitiveIntelligence(self.strategy)
        self.playbook_engine = PlaybookEngine(self.strategy)
        self.fund_flow_intel = FundFlowIntelligence(self.strategy)
        self.mechanics_engine = DynamicMechanicsEngine(self.strategy)
        self.cyclical_intel = CyclicalIntelligence(self.strategy)

    def _get_dynamic_thresholds(self, df: pd.DataFrame) -> Dict:
        """
        【V335.2 核心指标版】动态阈值校准中心
        - 核心职责: 为各模块提供自适应标尺，只为最核心、最难被操纵的筹码结构指标提供校准。
        - 调用时机: 在所有模块实例化之前调用。
        """
        # print("        -> [动态阈值校准中心 V335.2] 启动...")
        thresholds = {}
        window = 250 # 使用过去一年的数据作为基准
        # 1. 成本加速度阈值：只相信最顶尖5%的进攻意图
        cost_accel_col = 'ACCEL_5_peak_cost_D'
        if cost_accel_col in df.columns:
            thresholds['cost_accel_significant'] = df[cost_accel_col].rolling(window).quantile(0.95)
        # 2. 筹码集中度加速度阈值：只相信最顶尖5%的吸筹决心
        conc_accel_col = 'ACCEL_5_concentration_90pct_D'
        if conc_accel_col in df.columns:
            thresholds['conc_accel_significant'] = df[conc_accel_col].rolling(window).quantile(0.05)
        # print("        -> [动态阈值校准中心 V335.2] 校准完成。")
        return thresholds

    def run_all_diagnostics(self) -> Dict:
        """
        【V403.1 性能与逻辑修复版】情报层总入口。
        - 核心升级 (本次修改):
          - [逻辑修复] 修正了阶段五中对认知层(cognitive_intel)方法调用的严重逻辑错误。原代码中，每个synthesize_*方法都在原始df上操作，导致后续方法无法获取前面方法新增的列。新代码通过链式调用 (df = method(df)) 修正了数据流，确保了计算的正确性。
          - [性能优化] 将认知层多个返回DataFrame的方法调用串联起来，避免了对策略实例属性 `self.strategy.df_indicators` 的反复、低效的赋值操作，减少了不必要的对象创建和内存开销。
        - 核心重构 (V403.0逻辑保留):
          - 遵循“终极信号”范式，简化了对 `StructuralIntelligence` 的调用。
        - 收益:
          - 保证了情报层计算逻辑的正确性，这是最高优先级的修复。
          - 显著提升了阶段五的执行效率和内存使用效率。
          - 代码结构更清晰，完全适配了新一代的结构层情报引擎。
        """
        # print("--- [情报层总指挥官 V403.1 性能与逻辑修复版] 开始执行所有诊断模块... ---")
        df = self.strategy.df_indicators
        self.strategy.atomic_states = {}
        self.strategy.trigger_events = {}
        # --- 阶段一: 基础层与原子情报诊断 ---
        # print("    - [阶段 1/5] 正在执行基础层与原子情报诊断...")
        self.strategy.atomic_states.update(self._diagnose_strategic_context(df))
        self.strategy.atomic_states.update(self._diagnose_long_term_daily_chip_context(df))
        # ▼▼▼ 调用行业生命周期诊断 ▼▼▼
        self.strategy.atomic_states.update(self._score_industry_lifecycle_context(df))
        # ▼▼▼ 调用KPL题材热度评分 ▼▼▼
        self.strategy.atomic_states.update(self._score_kpl_theme_hotness(df))
        # FoundationIntelligence 现在返回字典，需要更新到 atomic_states
        foundation_states = self.foundation_intel.run_foundation_analysis_command()
        self.strategy.atomic_states.update(foundation_states)

        self.strategy.atomic_states.update(self.fund_flow_intel.diagnose_fund_flow_states(df))
        self.mechanics_engine.run_dynamic_analysis_command()

        df = self.pattern_recognizer.identify_all(df)
        # --- 阶段二: 结构层情报诊断与合成 ---
        # print("    - [阶段 2/5] 正在执行结构层情报诊断与合成...")
        self.strategy.atomic_states.update(self.structural_intel.diagnose_structural_states(df))
        # --- 阶段三: 行为层情报诊断与合成 ---
        # print("    - [阶段 3/5] 正在执行行为层情报诊断与合成...")
        self.strategy.atomic_states.update(self.behavioral_intel.run_behavioral_analysis_command(df))
        self.strategy.atomic_states.update(self.cyclical_intel.run_cyclical_analysis_command(df))
        # --- 阶段四: 筹码层情报诊断与合成 ---
        # print("    - [阶段 4/5] 正在执行筹码层情报诊断与合成...")
        chip_states, chip_triggers = self.chip_intel.run_chip_intelligence_command(df)
        self.strategy.atomic_states.update(chip_states)
        self.strategy.trigger_events.update(chip_triggers)
        # --- 阶段五: 认知层元融合、主力推演与战法生成 ---
        # print("    - [阶段 5/5] 正在执行认知层元融合、主力推演与战法生成...")
        # 修复调用链，将 self.strategy.df_indicators 的反复赋值改为对局部变量 df 的链式调用
        # 5.1 宏观上下文与质量分数合成 (高优先级，被其他认知模块依赖)
        df = self.cognitive_intel.synthesize_trend_quality_score(df)
        df = self.cognitive_intel.synthesize_contextual_zone_scores(df)
        # 5.2 基础认知分数合成
        df = self.cognitive_intel.synthesize_holding_risks(df)
        df = self.cognitive_intel.synthesize_trend_regime_signals(df)
        df = self.cognitive_intel.synthesize_volatility_breakout_signals(df)
        self.strategy.atomic_states.update(self.cognitive_intel.synthesize_chip_fund_flow_synergy(df))
        self.strategy.atomic_states.update(self.cognitive_intel.synthesize_dynamic_offense_states(df))
        # 5.3 高阶风险与机会合成
        df = self.cognitive_intel.synthesize_pullback_states(df)
        df = self.cognitive_intel.synthesize_market_engine_states(df)
        df = self.cognitive_intel.synthesize_divergence_risks(df)
        df = self.cognitive_intel.synthesize_opportunity_risk_scores(df)
        self.strategy.atomic_states.update(self.cognitive_intel.synthesize_topping_behaviors(df))
        df = self.cognitive_intel.synthesize_trend_sustainability_signals(df)
        df = self.cognitive_intel.synthesize_tactical_opportunities(df)
        df = self.cognitive_intel.synthesize_consolidation_breakout_signals(df)
        # 5.4 最终认知合成与主力行为推演
        df = self.cognitive_intel.synthesize_structural_fusion_scores(df)
        df = self.cognitive_intel.synthesize_ultimate_confirmation_scores(df)
        df = self.cognitive_intel.synthesize_ignition_resonance_score(df)
        df = self.cognitive_intel.synthesize_breakdown_resonance_score(df)
        df = self.cognitive_intel.synthesize_reversal_resonance_scores(df)
        df = self.cognitive_intel.synthesize_perfect_storm_signals(df)
        df = self.cognitive_intel.synthesize_cognitive_scores(df)
        self.strategy.atomic_states.update(self.cognitive_intel.diagnose_trend_stage_score(df))
        self.strategy.atomic_states.update(self.cognitive_intel.diagnose_market_structure_states(df))
        # 5.5 生成触发器、战法与交易剧本
        trigger_events = self.playbook_engine.define_trigger_events(df)
        self.strategy.trigger_events.update(trigger_events)
        self.strategy.atomic_states.update(self.cognitive_intel.synthesize_advanced_tactics(df))
        self.strategy.atomic_states.update(self.cognitive_intel.synthesize_prime_tactic(df))
        pullback_enhancements = self.behavioral_intel._diagnose_pullback_enhancement_matrix(df)
        self.strategy.atomic_states.update(self.cognitive_intel._diagnose_pullback_tactics_matrix(df, pullback_enhancements))
        self.strategy.setup_scores, self.strategy.playbook_states = self.playbook_engine.generate_playbook_states(self.strategy.trigger_events)
        squeeze_playbook_states = self.cognitive_intel.synthesize_squeeze_playbooks(df)
        self.strategy.atomic_states.update(squeeze_playbook_states)
        self.strategy.playbook_states.update({k: v for k, v in squeeze_playbook_states.items() if k.startswith('PLAYBOOK_')})
        # 5.6 (调试模块)
        debug_params = get_params_block(self.strategy, 'debug_params')
        if get_param_value(debug_params.get('enable_pullback_decision_log'), False):
            decision_log_df = self.cognitive_intel._create_pullback_decision_log(df, pullback_enhancements)
            final_tactic_days = decision_log_df.filter(like='FINAL_').any(axis=1)
            if final_tactic_days.any():
                print("\n--- [回踩战术决策日志探针] ---")
                display_cols = [col for col in decision_log_df.columns if 'POTENTIAL_' in col or 'FINAL_' in col]
                print("决策日志 (POTENTIAL: 潜在机会, FINAL: 最终决策):")
                print(decision_log_df.loc[final_tactic_days, display_cols])
                print("--- [探针结束] ---\n")
        # 在所有计算完成后，一次性更新策略实例的DataFrame
        self.strategy.df_indicators = df
        
        # --- 最终报告 ---
        # print("--- [情报层总指挥官 V403.1] 所有诊断模块执行完毕。 ---")
        return self.strategy.trigger_events

    def _diagnose_strategic_context(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        周线战略情报转化器
        - 核心职责: 将从周线引擎注入的、连续的战略分数和状态信号，
                    转化为日线引擎各模块可以理解的、离散的布尔型原子状态。
        """
        strategic_states = {}
        default_series = pd.Series(False, index=df.index)

        # 1. 转化战略分数
        score_col = 'strategic_score_W'
        if score_col in df.columns:
            score = df[score_col]
            # 状态: 战略性看涨 (强顺风) - 用于进攻层加分
            strategic_states['CONTEXT_STRATEGIC_BULLISH_W'] = score >= 5
            # 状态: 战略性看跌 (逆风) - 用于判断层增加否决票
            strategic_states['CONTEXT_STRATEGIC_BEARISH_W'] = score < 0
        else:
            # 如果分数不存在，则所有相关状态都为False
            strategic_states['CONTEXT_STRATEGIC_BULLISH_W'] = default_series
            strategic_states['CONTEXT_STRATEGIC_BEARISH_W'] = default_series
            print("    - [周线情报转化-警告] 未找到 'strategic_score_W' 列。")

        # 2. 转化战略顶部风险信号
        topping_col = 'state_node_topping_W'
        if topping_col in df.columns:
            # 状态: 战略性顶部风险 - 用于判断层和离场层的强否决
            strategic_states['CONTEXT_STRATEGIC_TOPPING_RISK_W'] = df[topping_col]
        else:
            strategic_states['CONTEXT_STRATEGIC_TOPPING_RISK_W'] = default_series
            print(f"    - [周线情报转化-警告] 未找到 '{topping_col}' 列。")
            
        # 3. 转化战略点火信号 (可选，用于进攻层)
        ignition_col = 'state_node_ignition_W'
        if ignition_col in df.columns:
            strategic_states['CONTEXT_STRATEGIC_IGNITION_W'] = df[ignition_col]
        else:
            strategic_states['CONTEXT_STRATEGIC_IGNITION_W'] = default_series

        # print(f"    - [周线情报转化] 完成。已生成 {len(strategic_states)} 个战略级原子状态。")
        return strategic_states

    def _diagnose_long_term_daily_chip_context(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.0 智能信号消费版】日线长周期筹码战略上下文诊断模块
        - 核心升级 (本次修改):
          - [逻辑升级] 不再使用简单的斜率来判断筹码集中，而是直接消费由 `chip_intelligence` 模块生成的、
                        经过多维度交叉验证的 `SCORE_CHIP_TRUE_ACCUMULATION` (真实吸筹分)。
          - [风险规避] 同样消费 `SCORE_CHIP_FALSE_ACCUMULATION_RISK` (虚假集中风险分)，并将其转化为
                        `CONTEXT_CHIP_LONG_TERM_DIVERGENCE_D` 信号，用于风险惩罚。
        - 收益:
          - 彻底解决了将顶部派发误判为筹码集中的问题，信号的可靠性实现了质的飞跃。
        """
        states = {}
        atomic_states = self.strategy.atomic_states # 获取原子状态字典
        default_score = pd.Series(0.0, index=df.index)

        # --- 1. 消费“真实吸筹分” ---
        # 直接使用 chip_intelligence 生成的高质量信号
        # 将判断依据从简单的斜率改为消费智能信号
        true_accumulation_score = atomic_states.get('SCORE_CHIP_TRUE_ACCUMULATION', default_score)
        # 当真实吸筹分数高于一个阈值（例如0.5）时，我们才认为这是一个有效的“筹码集中”信号
        states['CONTEXT_CHIP_LONG_TERM_ACCUMULATION_D'] = true_accumulation_score > 0.5
        # “加速集中”可以定义为真实吸筹分的斜率为正
        true_accumulation_slope = true_accumulation_score.diff().fillna(0)
        states['CONTEXT_CHIP_LONG_TERM_ACCEL_ACCUMULATION_D'] = (true_accumulation_score > 0.5) & (true_accumulation_slope > 0)

        # --- 2. 消费“虚假集中风险分” ---
        # 将虚假集中风险分转化为发散信号，用于风险惩罚
        # 将发散的判断依据从简单的斜率改为消费智能风险信号
        false_accumulation_risk_score = atomic_states.get('SCORE_CHIP_FALSE_ACCUMULATION_RISK', default_score)
        states['CONTEXT_CHIP_LONG_TERM_DIVERGENCE_D'] = false_accumulation_risk_score > 0.5
        false_accumulation_risk_slope = false_accumulation_risk_score.diff().fillna(0)
        states['CONTEXT_CHIP_LONG_TERM_ACCEL_DIVERGENCE_D'] = (false_accumulation_risk_score > 0.5) & (false_accumulation_risk_slope > 0)

        # --- 3. 长期筹码健康度趋势 (逻辑保持不变) ---
        is_long_term_health_improving = df.get('SLOPE_21_chip_health_score_D', pd.Series(False, index=df.index)) > 0
        states['CONTEXT_CHIP_LONG_TERM_HEALTH_IMPROVING_D'] = is_long_term_health_improving

        # print(f"            -> [日线长周期筹码战略诊断 V3.0] 已生成 {len(states)} 个基于智能信号的战略状态。") # 修改: 更新版本号
        return states

    def _score_industry_lifecycle_context(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V4.0 数值化消费版】行业生命周期上下文评分模块
        - 核心升级: 直接消费由DAO层计算好的、融合后的数值化行业阶段分数。
        - 收益: 逻辑极大简化，职责更清晰，性能更高。
        """
        scores = {}
        four_layer_params = get_params_block(self.strategy, 'four_layer_scoring_params', {})
        params = four_layer_params.get('industry_lifecycle_scoring_params', {})
        if not params.get('enabled', False):
            print("    - [行业生命周期评分-警告] 模块在配置中被禁用。")
            return {}
        default_series = pd.Series(0.0, index=df.index)
        # 直接从df中获取DAO层计算好的数值化分数
        scores['SCORE_INDUSTRY_MARKUP'] = df.get('industry_markup_score_D', default_series)
        scores['SCORE_INDUSTRY_PREHEAT'] = df.get('industry_preheat_score_D', default_series)
        scores['SCORE_INDUSTRY_STAGNATION'] = df.get('industry_stagnation_score_D', default_series)
        scores['SCORE_INDUSTRY_DOWNTREND'] = df.get('industry_downtrend_score_D', default_series)
        print(f"    - [行业生命周期评分 V4.0] 完成。已直接消费4个融合后的行业阶段数值化分数。")
        return scores

    def _score_kpl_theme_hotness(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        KPL题材热度评分模块
        - 核心职责: 将注入的题材热度原始分转换为最终的策略评分。
        """
        scores = {}
        # 1. 统一从 feature_engineering_params 获取参数
        fe_params = get_params_block(self.strategy, 'feature_engineering_params', {})
        params = fe_params.get('kpl_theme_params', {})
        if not params.get('enabled', False):
            return {}
        required_col = 'THEME_HOTNESS_SCORE_D'
        if required_col not in df.columns:
            print(f"    - [KPL题材热度评分-警告] 缺少依赖列: {required_col}，模块已跳过。")
            return {}
        hotness_score = df[required_col]
        threshold = params.get('score_threshold', 0.5)
        # 示例逻辑：当热度分超过阈值时，我们认为它是一个有效的看涨信号
        # 分数直接使用归一化后的热度分，但只有超过阈值才有效
        final_score = hotness_score.where(hotness_score >= threshold, 0)
        scores['SCORE_THEME_HOTNESS'] = final_score.fillna(0.0)
        # print(f"    - [KPL题材热度评分] 完成。")
        return scores





