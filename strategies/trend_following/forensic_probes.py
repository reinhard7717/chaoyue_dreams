import pandas as pd
import numpy as np
from typing import Dict
from strategies.trend_following.utils import get_params_block

class ForensicProbes:
    """
    【V2.8 · 垂直切片探针版】法医探针调度中心
    """
    def __init__(self, intelligence_layer_instance):
        """
        【V2.8 · 垂直切片探针版】法医探针调度中心
        - 核心升级: 新增注册 behavioral_probes 和 fund_flow_probes 模块。
        """
        
        from .probes.behavioral_probes import BehavioralProbes
        from .probes.fund_flow_probes import FundFlowProbes
        from .probes.micro_behavior_probes import MicroBehaviorProbes
        from .probes.chip_probes import ChipProbes
        from .probes.cognitive_probes import CognitiveProbes
        from .probes.dynamic_mechanics_probes import DynamicMechanicsProbes
        from .probes.foundation_probes import FoundationProbes
        from .probes.process_probes import ProcessProbes
        from .probes.structural_probes import StructuralProbes
        
        self.behavioral_probes = BehavioralProbes(intelligence_layer_instance)
        self.fund_flow_probes = FundFlowProbes(intelligence_layer_instance)
        self.micro_behavior_probes = MicroBehaviorProbes(intelligence_layer_instance)
        self.chip_probes = ChipProbes(intelligence_layer_instance)
        self.cognitive_probes = CognitiveProbes(intelligence_layer_instance)
        self.dynamic_mechanics_probes = DynamicMechanicsProbes(intelligence_layer_instance)
        self.foundation_probes = FoundationProbes(intelligence_layer_instance)
        self.process_probes = ProcessProbes(intelligence_layer_instance)
        self.structural_probes = StructuralProbes(intelligence_layer_instance)
        
        # 注册新的探针
        self._deploy_liquidity_vacuum_probe = self.behavioral_probes._deploy_liquidity_vacuum_probe
        self._deploy_ff_distribution_resonance_probe = self.fund_flow_probes._deploy_ff_distribution_resonance_probe
        self._deploy_peak_rejection_risk_probe = self.micro_behavior_probes._deploy_peak_rejection_risk_probe
        self._deploy_pressure_transmutation_probe = self.behavioral_probes._deploy_pressure_transmutation_probe
        self._deploy_hephaestus_forge_probe = self.chip_probes._deploy_hephaestus_forge_probe
        self._deploy_chip_resonance_probe = self.chip_probes._deploy_chip_resonance_probe
        self._deploy_liquidity_trap_probe = self.cognitive_probes._deploy_liquidity_trap_probe
        self._deploy_profit_taking_pressure_probe = self.cognitive_probes._deploy_profit_taking_pressure_probe
        self._deploy_comprehensive_top_risk_probe = self.cognitive_probes._deploy_comprehensive_top_risk_probe
        self._deploy_trend_quality_probe = self.cognitive_probes._deploy_trend_quality_probe
        self._deploy_main_force_intent_duel_probe = self.cognitive_probes._deploy_main_force_intent_duel_probe
        self._deploy_structural_health_probe = self.structural_probes._deploy_structural_health_probe
        self._deploy_structural_pillar_fusion_probe = self.structural_probes._deploy_structural_pillar_fusion_probe
        self._deploy_structural_pillar_dissection_probe = self.structural_probes._deploy_structural_pillar_dissection_probe
        self._deploy_ares_chariot_probe = self.dynamic_mechanics_probes._deploy_ares_chariot_probe
        self._deploy_apollos_lyre_probe = self.foundation_probes._deploy_apollos_lyre_probe
        self._deploy_cost_advantage_probe = self.process_probes._deploy_cost_advantage_probe
        self._deploy_themis_scales_probe = self.process_probes._deploy_themis_scales_probe
        self._deploy_process_sync_probe = self.process_probes._deploy_process_sync_probe














