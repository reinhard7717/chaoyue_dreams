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
        【V408.0 · 依赖修复版】情报层总入口。
        - 核心重构: 调整了情报引擎的调用顺序，将 ChipIntelligence 提前到 StructuralIntelligence 之前，
                    解决了结构层因缺少筹码数据而跳过部分诊断的问题。
        """
        print("--- [情报层总指挥官 V408.0 · 依赖修复版] 开始执行所有诊断模块... ---") # 更新版本号和说明
        df = self.strategy.df_indicators
        self.strategy.atomic_states = {}
        self.strategy.trigger_events = {}
        self.strategy.playbook_states = {}
        self.strategy.exit_triggers = pd.DataFrame(index=df.index)

        def update_states(new_states: Dict):
            if isinstance(new_states, dict):
                self.strategy.atomic_states.update(new_states)

        # --- 阶段一: 原子信号生成 ---
        # 调用所有底层情报引擎，生成所有S+/S/A/B四级终极信号，并注入atomic_states
        print("    - [阶段 1/3] 正在执行原子信号生成...")
        update_states(self.foundation_intel.run_foundation_analysis_command())
        
        # 将 ChipIntelligence 的调用提前
        # 必须先生成筹码数据，才能供其他模块（如结构层）消费
        chip_states, _ = self.chip_intel.run_chip_intelligence_command(df)
        update_states(chip_states)
        
        # 现在 StructuralIntelligence 可以安全地消费由 ChipIntelligence 生成的数据
        update_states(self.structural_intel.diagnose_structural_states(df))
        
        self.behavioral_intel.run_behavioral_analysis_command()
        update_states(self.fund_flow_intel.diagnose_fund_flow_states(df))
        self.mechanics_engine.run_dynamic_analysis_command()
        update_states(self.cyclical_intel.run_cyclical_analysis_command(df))
        
        # --- 阶段二: 跨域认知融合 ---
        # 调用认知层总入口，它会消费阶段一生成的所有原子信号，并生成更高阶的认知信号
        print("    - [阶段 2/3] 正在执行认知层跨域元融合...")
        self.cognitive_intel.synthesize_cognitive_scores(df, pullback_enhancements={})

        # --- 阶段三: 最终战法与剧本生成 ---
        # Playbook引擎消费所有已生成的原子和认知信号，定义最终的触发器和剧本
        print("    - [阶段 3/3] 正在生成最终战法与剧本...")
        trigger_events = self.playbook_engine.define_trigger_events(df)
        self.strategy.trigger_events.update(trigger_events)
        _, playbook_states = self.playbook_engine.generate_playbook_states(self.strategy.trigger_events)
        self.strategy.playbook_states.update(playbook_states)
        
        # --- 阶段四: 硬性离场信号生成 ---
        print("    - [阶段 4/4] 正在生成硬性离场信号...")
        exit_triggers_df = self.exit_layer.generate_hard_exit_triggers()
        self.strategy.exit_triggers = exit_triggers_df
        
        print("--- [情报层总指挥官 V408.0] 所有诊断模块执行完毕。 ---") # 更新版本号
        return self.strategy.trigger_events
