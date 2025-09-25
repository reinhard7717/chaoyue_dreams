# 文件: strategies/trend_following/intelligence_layer.py
# 情报层总指挥官 (重构版)
import pandas as pd
from typing import Dict
from .exit_layer import ExitLayer
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
from strategies.kline_pattern_recognizer import KlinePatternRecognizer
from .utils import get_params_block

class IntelligenceLayer:
    """
    【V407.0 · 终极信号适配版】情报层总指挥官
    - 核心职责: 1. 实例化所有专业化的情报子模块。
                2. 按照“原子信号生成 -> 跨域认知融合 -> 战术剧本生成”的顺序，编排和调用这些子模块。
                3. 整合所有模块产出的原子状态和触发器，供下游层使用。
    - 本次修改: 全面适配所有情报引擎的“大一统”重构，确保调用流程和数据流正确无误。
    """
    def __init__(self, strategy_instance):
        """
        初始化情报层总指挥官。
        """
        self.strategy = strategy_instance
        self.kline_params = get_params_block(self.strategy, 'kline_pattern_params')
        self.strategy.pattern_recognizer = KlinePatternRecognizer(params=self.kline_params)

        # 实例化所有子模块，注入依赖
        self.foundation_intel = FoundationIntelligence(self.strategy)
        self.structural_intel = StructuralIntelligence(self.strategy, {}) # dynamic_thresholds 已废弃
        self.chip_intel = ChipIntelligence(self.strategy, {}) # dynamic_thresholds 已废弃
        self.behavioral_intel = BehavioralIntelligence(self.strategy)
        self.fund_flow_intel = FundFlowIntelligence(self.strategy)
        self.mechanics_engine = DynamicMechanicsEngine(self.strategy)
        self.cyclical_intel = CyclicalIntelligence(self.strategy)
        self.cognitive_intel = CognitiveIntelligence(self.strategy)
        self.playbook_engine = PlaybookEngine(self.strategy)
        self.exit_layer = ExitLayer(self.strategy)

    def run_all_diagnostics(self) -> Dict:
        """
        【V410.0 · 依赖重构版】情报层总入口。
        - 核心重构: 调整了情报引擎的调用顺序，将 CyclicalIntelligence 提前到所有其他引擎之前，
                    因为它提供的周期信号是所有引擎计算“情景分”的基础依赖。
        """
        print("--- [情报层总指挥官 V410.0 · 依赖重构版] 开始执行所有诊断模块... ---")
        df = self.strategy.df_indicators
        self.strategy.atomic_states = {}
        self.strategy.trigger_events = {}
        self.strategy.playbook_states = {}
        self.strategy.exit_triggers = pd.DataFrame(index=df.index)

        def update_states(new_states: Dict):
            if isinstance(new_states, dict):
                self.strategy.atomic_states.update(new_states)

        # --- 阶段一: 原子信号生成 (已按依赖关系重构顺序) ---
        print("    - [阶段 1/4] 正在执行原子信号生成...")
        
        # 1. 首先运行周期引擎，生成最基础的宏观周期信号
        print("      -> 正在运行 [周期引擎]...")
        update_states(self.cyclical_intel.run_cyclical_analysis_command(df))
        
        # 2. 运行其他所有情报引擎，它们现在可以安全地消费周期信号
        print("      -> 正在运行 [基础层引擎]...")
        update_states(self.foundation_intel.run_foundation_analysis_command())
        
        print("      -> 正在运行 [筹码引擎]...")
        chip_states, _ = self.chip_intel.run_chip_intelligence_command(df)
        update_states(chip_states)
        
        print("      -> 正在运行 [结构引擎]...")
        update_states(self.structural_intel.diagnose_structural_states(df))
        
        print("      -> 正在运行 [行为引擎]...")
        self.behavioral_intel.run_behavioral_analysis_command() # 此方法内部更新状态
        
        print("      -> 正在运行 [资金流引擎]...")
        update_states(self.fund_flow_intel.diagnose_fund_flow_states(df))
        
        print("      -> 正在运行 [动态力学引擎]...")
        self.mechanics_engine.run_dynamic_analysis_command() # 此方法内部更新状态
        
        # --- 阶段二: 跨域认知融合 ---
        print("    - [阶段 2/4] 正在执行认知层跨域元融合...")
        self.cognitive_intel.synthesize_cognitive_scores(df, pullback_enhancements={})

        # --- 阶段三: 最终战法与剧本生成 ---
        print("    - [阶段 3/4] 正在生成最终战法与剧本...")
        trigger_events = self.playbook_engine.define_trigger_events(df)
        self.strategy.trigger_events.update(trigger_events)
        _, playbook_states = self.playbook_engine.generate_playbook_states(self.strategy.trigger_events)
        self.strategy.playbook_states.update(playbook_states)
        
        # --- 阶段四: 硬性离场信号生成 ---
        print("    - [阶段 4/4] 正在生成硬性离场信号...")
        exit_triggers_df = self.exit_layer.generate_hard_exit_triggers()
        self.strategy.exit_triggers = exit_triggers_df
        
        print("--- [情报层总指挥官 V410.0] 所有诊断模块执行完毕。 ---")
        return self.strategy.trigger_events


















