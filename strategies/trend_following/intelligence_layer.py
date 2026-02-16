# 文件: strategies/trend_following/intelligence_layer.py
# 情报层总指挥官 (重构版)
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict
from .structural_defense_layer import StructuralDefenseLayer
# --- 将所有情报模块和工具函数的导入整合到文件顶部 ---
from .intelligence.foundation_intelligence import FoundationIntelligence
from .intelligence.structural_intelligence import StructuralIntelligence
from .intelligence.chip_intelligence import ChipIntelligence
from .intelligence.behavioral_intelligence import BehavioralIntelligence
from .intelligence.cognitive_intelligence import CognitiveIntelligence
from .intelligence.micro_behavior_engine import MicroBehaviorEngine
from .intelligence.fund_flow_intelligence import FundFlowIntelligence
from .intelligence.dynamic_mechanics_engine import DynamicMechanicsEngine
from .intelligence.cyclical_intelligence import CyclicalIntelligence
from .intelligence.pattern_intelligence import PatternIntelligence
from .intelligence.process_intelligence import ProcessIntelligence
from .intelligence.predictive_intelligence import PredictiveIntelligence
from .intelligence.fusion_intelligence import FusionIntelligence
from .intelligence.intraday_behavior_engine import IntradayBehaviorEngine
from strategies.kline_pattern_recognizer import KlinePatternRecognizer
from strategies.trend_following.utils import (
    get_params_block, get_param_value, calculate_context_scores, 
    normalize_score, normalize_to_bipolar, _calculate_gaia_bedrock_support, 
    _calculate_historical_low_support, get_unified_score
)
from .forensic_probes import ForensicProbes

class IntelligenceLayer:
    """
    【V409.0 · 探针免疫与神力重构版】情报层总指挥官
    - 核心职责: 1. 实例化所有专业化的情报子模块。
                2. 按照“原子信号生成 -> 跨域认知融合 -> 战术剧本生成”的顺序，编排和调用这些子模块。
                3. 整合所有模块产出的原子状态和触发器，供下游层使用。
    """
    def __init__(self, strategy_instance):
        """
        【V409.0 · 物理组件复原版】
        - 恢复 self.probes 的初始化，防止外部战术引擎调用时产生属性丢失。
        """
        self.strategy=strategy_instance
        self.kline_params=get_params_block(self.strategy,'kline_pattern_params')
        # self.strategy.pattern_recognizer = KlinePatternRecognizer(params=self.kline_params)
        # self.foundation_intel = FoundationIntelligence(self.strategy)
        # self.structural_intel = StructuralIntelligence(self.strategy, {})
        # self.chip_intel = ChipIntelligence(self.strategy)
        # self.behavioral_intel = BehavioralIntelligence(self.strategy)
        # self.micro_behavior_engine = MicroBehaviorEngine(self.strategy)
        # self.intraday_behavior_engine = IntradayBehaviorEngine(self.strategy)
        # self.fund_flow_intel = FundFlowIntelligence(self.strategy)
        # self.mechanics_engine = DynamicMechanicsEngine(self.strategy)
        # self.pattern_intel = PatternIntelligence(strategy_instance)
        # self.cyclical_intel = CyclicalIntelligence(self.strategy)
        self.process_intel=ProcessIntelligence(self.strategy)
        self.fusion_intel=FusionIntelligence(self.strategy)
        self.cognitive_intel=CognitiveIntelligence(self.strategy)
        self.structural_defense_layer=StructuralDefenseLayer(self.strategy)
        self.predictive_intel=PredictiveIntelligence(self.strategy)
        self.probes=ForensicProbes(self)

    def run_all_diagnostics(self, df: pd.DataFrame) -> Dict:
        """
        【V427.0 · 探针参数唤醒版】
        - 修复使用 get_param_value 获取字典嵌套时的类型转换问题，确保测试环境探针正确开启。
        """
        self.strategy.atomic_states={}
        self.strategy.trigger_events={}
        self.strategy.playbook_states={}
        self.strategy.exit_triggers=pd.DataFrame(index=df.index)
        debug_params=get_params_block(self.strategy,'debug_params',{})
        if hasattr(self,'probes') and self.probes is not None:
            self.probes.should_probe=get_param_value(debug_params.get('enabled'),False)
            probe_dates_list=get_param_value(debug_params.get('probe_dates'),[])
            if not probe_dates_list:
                single_date=get_param_value(debug_params.get('probe_date'),None)
                if single_date:
                    probe_dates_list=[single_date]
            if probe_dates_list and isinstance(probe_dates_list,list):
                self.probes.probe_dates_set={pd.to_datetime(d).date() for d in probe_dates_list if d}
            else:
                self.probes.probe_dates_set=set()
        def update_states(new_states: Dict):
            if isinstance(new_states,dict):
                self.strategy.atomic_states.update(new_states)
        update_states(self.process_intel.run_process_diagnostics(df,task_type_filter=None))
        self._ignite_relational_dynamics_engine(df)
        return self.strategy.atomic_states

    def deploy_forensic_probes(self):
        """
        【V3.0.0 · 鸭子类型柔性探测版】法医探针调度中心
        - 引入 getattr() 与 hasattr() 进行双层防御，完美适配局部模块隔离测试环境免于熔断。
        """
        if not hasattr(self,'probes') or self.probes is None:
            return
        if not getattr(self.probes,'should_probe',False):
            return
        probe_dates_set=getattr(self.probes,'probe_dates_set',set())
        probe_dates_list=list(probe_dates_set)
        if not probe_dates_list:
            return
        print("\n"+"="*30+f" [法医探针部署中心 V3.0.0] 开始对 {len(probe_dates_list)} 个目标日期进行解剖... "+"="*30)
        debug_params=get_params_block(self.strategy,'debug_params',{})
        for probe_date_date in probe_dates_list:
            probe_date=pd.Timestamp(probe_date_date)
            if self.strategy.df_indicators.index.tz is not None:
                try:
                    probe_date=probe_date.tz_localize(self.strategy.df_indicators.index.tz)
                except Exception:
                    try:
                        probe_date=probe_date.tz_convert(self.strategy.df_indicators.index.tz)
                    except Exception as e_conv:
                         continue
            if probe_date not in self.strategy.df_indicators.index:
                continue
            print("\n"+"="*25+f" 正在解剖 {probe_date_date.strftime('%Y-%m-%d')} "+"="*25)
            if get_param_value(debug_params.get('enable_trend_quality_probe'),False) and hasattr(self.probes,'_deploy_trend_quality_probe'):
                self.probes._deploy_trend_quality_probe(probe_date)
            if get_param_value(debug_params.get('enable_liquidity_trap_probe'),False) and hasattr(self.probes,'_deploy_liquidity_trap_probe'):
                self.probes._deploy_liquidity_trap_probe(probe_date)
            if get_param_value(debug_params.get('enable_ff_distribution_resonance_probe'),False) and hasattr(self.probes,'_deploy_ff_distribution_resonance_probe'):
                self.probes._deploy_ff_distribution_resonance_probe(probe_date)
            if get_param_value(debug_params.get('enable_structural_health_probe'),False) and hasattr(self.probes,'_deploy_structural_health_probe'):
                self.probes._deploy_structural_health_probe(probe_date)
            if get_param_value(debug_params.get('enable_structural_pillar_fusion_probe'),False) and hasattr(self.probes,'_deploy_structural_pillar_fusion_probe'):
                self.probes._deploy_structural_pillar_fusion_probe(probe_date)
            if get_param_value(debug_params.get('enable_structural_pillar_dissection_probe'),False) and hasattr(self.probes,'_deploy_structural_pillar_dissection_probe'):
                self.probes._deploy_structural_pillar_dissection_probe(probe_date,pillar_name='structural_stability')
            if get_param_value(debug_params.get('enable_comprehensive_top_risk_probe'),False) and hasattr(self.probes,'_deploy_comprehensive_top_risk_probe'):
                self.probes._deploy_comprehensive_top_risk_probe(probe_date)
            if get_param_value(debug_params.get('enable_euphoric_acceleration_probe'),False) and hasattr(self.probes,'_deploy_euphoric_acceleration_transmutation_probe'):
                self.probes._deploy_euphoric_acceleration_transmutation_probe(probe_date)
            if get_param_value(debug_params.get('enable_chip_lockdown_probe'),False) and hasattr(self.probes,'_deploy_bottom_accumulation_lockdown_probe'):
                self.probes._deploy_bottom_accumulation_lockdown_probe(probe_date)
            if get_param_value(debug_params.get('enable_winner_conviction_probe'),False) and hasattr(self.probes,'_deploy_winner_conviction_probe'):
                self.probes._deploy_winner_conviction_probe(probe_date)
            if get_param_value(debug_params.get('enable_profit_taking_pressure_probe'),False) and hasattr(self.probes,'_deploy_profit_taking_pressure_probe'):
                self.probes._deploy_profit_taking_pressure_probe(probe_date)
            if get_param_value(debug_params.get('enable_lockdown_scramble_probe'),False) and hasattr(self.probes,'_deploy_lockdown_scramble_probe'):
                self.probes._deploy_lockdown_scramble_probe(probe_date)
        print("\n"+"="*35+" [法医探针部署中心] 所有目标解剖完毕 "+"="*35+"\n")

    def _apply_hab_shock(self, series: pd.Series, window: int = 21) -> pd.Series:
        """【V2.0.0 · HAB存量冲击系统】将绝对数值转化为相对历史存量的动能冲击度(Z-Score)"""
        roll_mean=series.rolling(window=window,min_periods=1).mean()
        roll_std=series.rolling(window=window,min_periods=1).std().replace(0,1e-5).fillna(1e-5)
        return ((series-roll_mean)/roll_std).astype(np.float32)

    def _ignite_relational_dynamics_engine(self, df: pd.DataFrame):
        """
        【V3.0.0 · 普罗米修斯神力重构版】
        - 破除孤岛: 引入底层资金一致性与微观筹码收敛度作为张量催化剂。
        - 去线性化: 摒弃绝对值 clip 线性算法。采用微积分防噪、HAB 存量冲击、Tanh 压缩和 Power Law 增益。
        """
        df_index=df.index
        power_transfer=self.strategy.atomic_states.get('PROCESS_META_POWER_TRANSFER',pd.Series(0.0,index=df_index))
        covert_accum=self.strategy.atomic_states.get('PROCESS_META_COVERT_ACCUMULATION',pd.Series(0.0,index=df_index))
        winner_conv=self.strategy.atomic_states.get('PROCESS_META_WINNER_CONVICTION',pd.Series(0.0,index=df_index))
        loser_cap=self.strategy.atomic_states.get('PROCESS_META_LOSER_CAPITULATION',pd.Series(0.0,index=df_index))
        consistency=df['flow_consistency_D'].astype(np.float32) if 'flow_consistency_D' in df.columns else pd.Series(0.0,index=df_index)
        convergence=df['chip_convergence_ratio_D'].astype(np.float32) if 'chip_convergence_ratio_D' in df.columns else pd.Series(0.0,index=df_index)
        power_transfer=np.where(np.abs(power_transfer)<1e-4,0.0,power_transfer)
        covert_accum=np.where(np.abs(covert_accum)<1e-4,0.0,covert_accum)
        power_shock=np.tanh(self._apply_hab_shock(pd.Series(power_transfer,index=df_index),21))
        covert_shock=np.tanh(self._apply_hab_shock(pd.Series(covert_accum,index=df_index),21))
        winner_shock=np.tanh(self._apply_hab_shock(winner_conv,34))
        loser_shock=np.tanh(self._apply_hab_shock(loser_cap,34))
        flow_catalyst=0.5*(1.0+np.tanh(self._apply_hab_shock(consistency,13)))
        chip_catalyst=0.5*(1.0+np.tanh(self._apply_hab_shock(convergence,13)))
        storm_raw=power_shock.clip(lower=0)*loser_shock.clip(lower=0)*flow_catalyst
        stormborn_power=np.tanh(storm_raw**1.5).astype(np.float32)
        self.strategy.atomic_states['SCORE_ATOMIC_STORM_BORN_POWER']=stormborn_power
        still_raw=covert_shock.clip(lower=0)*winner_shock.clip(lower=0)*chip_catalyst
        still_waters_power=np.tanh(still_raw**1.5).astype(np.float32)
        self.strategy.atomic_states['SCORE_ATOMIC_STILL_WATERS_POWER']=still_waters_power
        relational_dynamics_power=np.maximum(stormborn_power,still_waters_power)
        self.strategy.atomic_states['SCORE_ATOMIC_RELATIONAL_DYNAMICS']=relational_dynamics_power.astype(np.float32)





