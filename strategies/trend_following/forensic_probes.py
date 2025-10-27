import pandas as pd
import numpy as np
from typing import Dict
from strategies.trend_following.utils import get_params_block

class ForensicProbes:
    """
    【V2.0 · 中央调度版】法医探针调度中心
    - 核心重构: 本模块不再包含任何探针的具体实现。
    - 核心职责: 1. 导入并实例化所有分类探针模块（如 BehavioralProbes, ChipProbes 等）。
                2. 通过方法委托，将所有子模块的探针方法统一暴露给上层调用。
    - 收益: 实现了探针的完全模块化，结构清晰，易于维护和扩展。
    """
    def __init__(self, intelligence_layer_instance):
        """
        【V2.3 · 长期获利盘派发探针版】法医探针调度中心
        """
        from .probes.behavioral_probes import BehavioralProbes
        from .probes.chip_probes import ChipProbes
        from .probes.cognitive_probes import CognitiveProbes
        from .probes.dynamic_mechanics_probes import DynamicMechanicsProbes
        from .probes.foundation_probes import FoundationProbes
        from .probes.fund_flow_probes import FundFlowProbes
        from .probes.process_probes import ProcessProbes
        self.behavioral_probes = BehavioralProbes(intelligence_layer_instance)
        self.chip_probes = ChipProbes(intelligence_layer_instance)
        self.cognitive_probes = CognitiveProbes(intelligence_layer_instance)
        self.dynamic_mechanics_probes = DynamicMechanicsProbes(intelligence_layer_instance)
        self.foundation_probes = FoundationProbes(intelligence_layer_instance)
        self.fund_flow_probes = FundFlowProbes(intelligence_layer_instance)
        self.process_probes = ProcessProbes(intelligence_layer_instance)
        self._deploy_prometheus_torch_probe = self.behavioral_probes._deploy_prometheus_torch_probe
        self._deploy_pressure_transmutation_probe = self.behavioral_probes._deploy_pressure_transmutation_probe
        self._deploy_hephaestus_forge_probe = self.chip_probes._deploy_hephaestus_forge_probe
        self._deploy_chip_resonance_probe = self.chip_probes._deploy_chip_resonance_probe
        self._deploy_thanatos_scythe_probe = self.cognitive_probes._deploy_thanatos_scythe_probe
        self._deploy_liquidity_trap_probe = self.cognitive_probes._deploy_liquidity_trap_probe
        self._deploy_profit_taking_pressure_probe = self.cognitive_probes._deploy_profit_taking_pressure_probe
        self._deploy_comprehensive_top_risk_probe = self.cognitive_probes._deploy_comprehensive_top_risk_probe
        # [代码新增开始]
        self._deploy_main_force_intent_duel_probe = self.cognitive_probes._deploy_main_force_intent_duel_probe
        # [代码新增结束]
        self._deploy_ares_chariot_probe = self.dynamic_mechanics_probes._deploy_ares_chariot_probe
        self._deploy_apollos_lyre_probe = self.foundation_probes._deploy_apollos_lyre_probe
        self._deploy_poseidons_trident_probe = self.fund_flow_probes._deploy_poseidons_trident_probe
        self._deploy_cost_advantage_probe = self.process_probes._deploy_cost_advantage_probe
        self._deploy_themis_scales_probe = self.process_probes._deploy_themis_scales_probe
        self._deploy_process_sync_probe = self.process_probes._deploy_process_sync_probe







