# 文件: strategies/trend_following/intelligence_layer.py
# 情报层总指挥官 (重构版)
import pandas as pd
import numpy as np
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
from .intelligence.pattern_intelligence import PatternIntelligence
from strategies.trend_following.utils import get_params_block, get_param_value, calculate_context_scores, normalize_score

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
        self.pattern_intel = PatternIntelligence(strategy_instance)
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
        # print("--- [情报层总指挥官 V410.0 · 依赖重构版] 开始执行所有诊断模块... ---")
        df = self.strategy.df_indicators
        self.strategy.atomic_states = {}
        self.strategy.trigger_events = {}
        self.strategy.playbook_states = {}
        self.strategy.exit_triggers = pd.DataFrame(index=df.index)

        def update_states(new_states: Dict):
            if isinstance(new_states, dict):
                self.strategy.atomic_states.update(new_states)

        # --- 阶段一: 原子信号生成 (已按依赖关系重构顺序) ---
        # print("    - [阶段 1/4] 正在执行原子信号生成...")
        
        # 1. 首先运行周期引擎，生成最基础的宏观周期信号
        # print("      -> 正在运行 [周期引擎]...")
        update_states(self.cyclical_intel.run_cyclical_analysis_command(df))
        
        # 2. 运行其他所有情报引擎，它们现在可以安全地消费周期信号
        # print("      -> 正在运行 [基础层引擎]...")
        update_states(self.foundation_intel.run_foundation_analysis_command())
        
        # print("      -> 正在运行 [筹码引擎]...")
        chip_states, _ = self.chip_intel.run_chip_intelligence_command(df)
        update_states(chip_states)
        
        # print("      -> 正在运行 [结构引擎]...")
        update_states(self.structural_intel.diagnose_structural_states(df))
        
        # print("      -> 正在运行 [行为引擎]...")
        self.behavioral_intel.run_behavioral_analysis_command() # 此方法内部更新状态
        
        # print("      -> 正在运行 [资金流引擎]...")
        update_states(self.fund_flow_intel.diagnose_fund_flow_states(df))
        
        # print("      -> 正在运行 [动态力学引擎]...")
        self.mechanics_engine.run_dynamic_analysis_command() # 此方法内部更新状态
        # print("      -> 正在运行 [形态智能引擎]...")
        update_states(self.pattern_intel.run_pattern_analysis_command(df))
        
        # --- 阶段二: 跨域认知融合 ---
        # print("    - [阶段 2/4] 正在执行认知层跨域元融合...")
        self.cognitive_intel.synthesize_cognitive_scores(df, pullback_enhancements={})

        # --- 阶段三: 最终战法与剧本生成 ---
        # print("    - [阶段 3/4] 正在生成最终战法与剧本...")
        trigger_events = self.playbook_engine.define_trigger_events(df)
        self.strategy.trigger_events.update(trigger_events)
        _, playbook_states = self.playbook_engine.generate_playbook_states(self.strategy.trigger_events)
        self.strategy.playbook_states.update(playbook_states)
        
        # --- 阶段四: 硬性离场信号生成 ---
        # print("    - [阶段 4/4] 正在生成硬性离场信号...")
        exit_triggers_df = self.exit_layer.generate_hard_exit_triggers()
        self.strategy.exit_triggers = exit_triggers_df

        self.deploy_forensic_probes()
        
        print("--- [情报层总指挥官 V410.0] 所有诊断模块执行完毕。 ---")
        return self.strategy.trigger_events

    def deploy_forensic_probes(self):
        """
        【V1.1 · 时区校准修复版】法医探针调度中心
        - 核心修复: 解决了因探针日期(tz-naive)与数据索引(tz-aware)时区不匹配导致的探针跳过问题。
        """
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        if not debug_params.get('enabled', False):
            return
            
        probe_date_str = debug_params.get('probe_date')
        if not probe_date_str:
            return
            
        probe_date = pd.to_datetime(probe_date_str)
        
        # 时区校准逻辑
        # 检查数据索引是否为“时区感知”类型
        if self.strategy.df_indicators.index.tz is not None:
            # 如果是，则将“天真”的探针日期本地化到与索引相同的时区
            try:
                probe_date = probe_date.tz_localize(self.strategy.df_indicators.index.tz)
            except Exception as e:
                # 处理重复本地化等异常情况
                print(f"    -> [法医探针] 警告: 在本地化探针日期时发生异常: {e}。可能日期已有时区。")
                # 尝试直接转换时区
                try:
                    probe_date = probe_date.tz_convert(self.strategy.df_indicators.index.tz)
                except Exception as e_conv:
                     print(f"    -> [法医探针] 错误: 转换探针日期时区也失败: {e_conv}。")
                     return

        # 现在，这里的检查是在两个时区类型相同的对象之间进行的
        if probe_date not in self.strategy.df_indicators.index:
            print(f"    -> [法医探针] 警告: 探针日期 {probe_date_str} (校准后: {probe_date}) 不在数据索引中，跳过探针部署。")
            return

        print("\n" + "="*30 + f" [法医探针部署中心 V1.1] 正在解剖 {probe_date_str} " + "="*30) # 更新版本号
        
        # 依次调用所有需要解剖的信号探针
        self._deploy_risk_resonance_probe(probe_date, 'FF')
        self._deploy_risk_resonance_probe(probe_date, 'CHIP')
        self._deploy_risk_resonance_probe(probe_date, 'DYN')
        self._deploy_risk_resonance_probe(probe_date, 'BEHAVIOR')
        self._deploy_risk_resonance_probe(probe_date, 'STRUCTURE')
        self._deploy_risk_resonance_probe(probe_date, 'FOUNDATION')
        
        self._deploy_ultimate_reversal_probe(probe_date, 'BEHAVIOR')
        self._deploy_hard_exit_probe(probe_date)
        self._deploy_dynamic_veto_probe(probe_date)
        
        self._deploy_ignition_resonance_probe(probe_date)
        self._deploy_volatility_breakout_probe(probe_date)
        self._deploy_ultimate_confirmation_probe(probe_date)
        self._deploy_v_reversal_ace_probe(probe_date)
        self._deploy_chip_price_lag_probe(probe_date)
        
        print("="*95 + "\n")

    # 全新的“动态力学否决权”探针
    def _deploy_dynamic_veto_probe(self, probe_date: pd.Timestamp):
        """【探针V1.0】解剖“动态力学一票否决权”信号"""
        print("\n--- [探针] 正在解剖: 【决策】动态力学一票否决权 (AVOID) ---")
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index)

        # 步骤1: 确认否决事实
        dynamic_action = df.get('dynamic_action', pd.Series('HOLD', index=df.index)).get(probe_date, 'HOLD')
        print(f"  - 当日动态力学战术动作: {dynamic_action}")

        if dynamic_action != 'AVOID':
            print("  - [结论] 当日未触发'一票否决'，无需深入解剖。")
            return

        # 步骤2: 追溯否决来源
        risk_expansion_score = atomic.get('SCORE_DYN_BEARISH_RESONANCE_S_PLUS', default_score).get(probe_date, 0)
        veto_threshold = 0.6 # 这是在 _get_dynamic_combat_action 中硬编码的阈值
        print(f"  - 触发原因: 力学看跌共振分 ({risk_expansion_score:.4f}) > 阈值 ({veto_threshold:.2f})")

        # 步骤3: 解剖共振构成
        p_conf = get_params_block(self.strategy, 'dynamic_mechanics_params', {})
        resonance_tf_weights = p_conf.get('resonance_tf_weights', {'short': 0.2, 'medium': 0.5, 'long': 0.3})
        periods = p_conf.get('periods', [1, 5, 13, 21, 55])
        
        # 重新计算 bearish_resonance_health
        overall_health = {}
        health_data = {'bearish_static': [], 'bearish_dynamic': []}
        # 注意：这里为了探针的独立性，重新执行了部分计算逻辑
        for name, calculator in {
            'volatility': self.mechanics_engine._calculate_volatility_health,
            'efficiency': self.mechanics_engine._calculate_efficiency_health,
            'kinetic_energy': self.mechanics_engine._calculate_kinetic_energy_health,
            'inertia': self.mechanics_engine._calculate_inertia_health,
        }.items():
            _, _, s_bear, d_bear = calculator(df, 120, 24, p_conf.get('dynamic_weights', {}), periods)
            health_data['bearish_static'].append(s_bear)
            health_data['bearish_dynamic'].append(d_bear)

        for health_type in health_data:
            overall_health[health_type] = {}
            for p in periods:
                components = [pillar_dict[p].values for pillar_dict in health_data[health_type] if p in pillar_dict]
                overall_health[health_type][p] = pd.Series(np.mean(np.stack(components, axis=0), axis=0), index=df.index)

        bearish_resonance_health = {p: overall_health['bearish_static'][p] * overall_health['bearish_dynamic'][p] for p in periods}
        
        short_force = (bearish_resonance_health.get(1, default_score).get(probe_date, 0.5) * bearish_resonance_health.get(5, default_score).get(probe_date, 0.5))**0.5
        medium_trend = (bearish_resonance_health.get(13, default_score).get(probe_date, 0.5) * bearish_resonance_health.get(21, default_score).get(probe_date, 0.5))**0.5
        long_inertia = bearish_resonance_health.get(55, default_score).get(probe_date, 0.5)
        
        print("  - 看跌共振分由三股力量加权构成:")
        print(f"    - 短期看跌力 (权重 {resonance_tf_weights['short']}): {short_force:.4f}")
        print(f"    - 中期看跌力 (权重 {resonance_tf_weights['medium']}): {medium_trend:.4f}")
        print(f"    - 长期看跌力 (权重 {resonance_tf_weights['long']}): {long_inertia:.4f}")

        # 步骤4 & 5: 找出最差的周期和最差的支柱
        worst_period_force = max([('短期', short_force), ('中期', medium_trend), ('长期', long_inertia)], key=lambda item: item[1])
        print(f"  - 主要矛盾在于【{worst_period_force[0]}看跌力】(分值: {worst_period_force[1]:.4f})")

        # 以最差的周期（例如短期）为例，深入解剖
        p_short = 5 # 以5日为例
        static_health_score = overall_health['bearish_static'][p_short].get(probe_date, 0.5)
        dynamic_health_score = overall_health['bearish_dynamic'][p_short].get(probe_date, 0.5)
        print(f"    -> 该力由 '静态看跌分' 和 '动态看跌分' 相乘得到:")
        print(f"       - {p_short}日静态看跌分: {static_health_score:.4f}")
        print(f"       - {p_short}日动态看跌分: {dynamic_health_score:.4f}")

        # 找出静态和动态中更差的那个，并解剖其支柱
        worst_health_type = 'bearish_static' if static_health_score > dynamic_health_score else 'bearish_dynamic'
        print(f"    -> 其中【{worst_health_type.replace('bearish_', '')}看跌分】更差，解剖其构成支柱:")
        
        pillar_scores = {}
        for i, pillar_name in enumerate(['volatility', 'efficiency', 'kinetic_energy', 'inertia']):
            pillar_score = health_data[worst_health_type][i][p_short].get(probe_date, 0.5)
            pillar_scores[pillar_name] = pillar_score
            print(f"       - {pillar_name.capitalize()} 支柱得分: {pillar_score:.4f}")
            
        root_cause_pillar = max(pillar_scores, key=pillar_scores.get)
        print(f"  - [最终结论] “一票否决”的根源在于【{root_cause_pillar.capitalize()}】支柱的看跌信号过强 (分值: {pillar_scores[root_cause_pillar]:.4f})。")

    # 全新的“硬性离场信号”探针
    def _deploy_hard_exit_probe(self, probe_date: pd.Timestamp):
        """【探针V1.0】解剖“硬性离场”信号"""
        print("\n--- [探针] 正在解剖: 【决策】硬性离场信号 (Sell Signal) ---")
        
        # 步骤1: 检查是否存在硬性离场触发器
        # self.strategy.exit_triggers 是由 ExitLayer 生成的 DataFrame
        if not hasattr(self.strategy, 'exit_triggers') or self.strategy.exit_triggers.empty:
            print("  - [警告] 未在策略实例中找到 'exit_triggers'，无法解剖。")
            return
            
        exit_triggers_today = self.strategy.exit_triggers.loc[probe_date]
        
        if not exit_triggers_today.any():
            print("  - [结论] 当日未触发任何硬性离场信号。'卖出信号'可能来源于其他逻辑。")
            return

        # 步骤2: 逐一解剖触发的离场原因
        print("  - 检测到硬性离场触发器，解剖如下:")
        triggered_reasons = exit_triggers_today[exit_triggers_today].index.tolist()
        
        root_cause = ""
        for reason in triggered_reasons:
            print(f"    - ✅ 触发: {reason}")
            
            # 针对 EXIT_TREND_BROKEN 进行深度解剖
            if reason == 'EXIT_TREND_BROKEN':
                p_pos_mgmt = get_params_block(self.strategy, 'position_management_params')
                p_trailing = p_pos_mgmt.get('trailing_stop', {})
                if get_param_value(p_trailing.get('enabled'), False):
                    model = get_param_value(p_trailing.get('trailing_model'))
                    if model == 'MOVING_AVERAGE':
                        ma_type = get_param_value(p_trailing.get('ma_type'), 'EMA').upper()
                        ma_period = get_param_value(p_trailing.get('ma_period'), 20)
                        ma_col = f'{ma_type}_{ma_period}_D'
                        
                        close_price = self.strategy.df_indicators['close_D'].get(probe_date)
                        ma_value = self.strategy.df_indicators.get(ma_col, pd.Series(np.nan)).get(probe_date)
                        
                        print(f"      -> 触发逻辑: 收盘价 < 移动平均线 ({ma_col})")
                        print(f"      -> 当日收盘价: {close_price:.2f}")
                        print(f"      -> {ma_col} 值: {ma_value:.2f}")
                        print(f"      -> 判定: {close_price:.2f} < {ma_value:.2f} 为 True")
                        root_cause = f"收盘价跌破 {ma_col}"

        if not root_cause and triggered_reasons:
            root_cause = triggered_reasons[0]

        print(f"  - [最终结论] '卖出信号'的直接原因是触发了硬性离场信号【{root_cause}】。")

    # 为“多域点火共振”新增的探针
    def _deploy_ignition_resonance_probe(self, probe_date: pd.Timestamp):
        """【探针V1.0】解剖“多域点火共振”信号"""
        print("\n--- [探针] 正在解剖: 【认知S】多域点火共振 ---")
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=self.strategy.df_indicators.index)
        
        # 获取所有组件的分数
        chip_playbook_ignition = atomic.get('SCORE_CHIP_PLAYBOOK_VACUUM_BREAKOUT', default_score).get(probe_date, 0)
        behavioral_ignition = atomic.get('SCORE_BEHAVIOR_BULLISH_RESONANCE_S_PLUS', default_score).get(probe_date, 0)
        structural_breakout = atomic.get('SCORE_STRUCTURE_BULLISH_RESONANCE_S_PLUS', default_score).get(probe_date, 0)
        mechanics_ignition = atomic.get('SCORE_DYN_BULLISH_RESONANCE_S_PLUS', default_score).get(probe_date, 0)
        chip_consensus_ignition = atomic.get('SCORE_CHIP_BULLISH_RESONANCE_S_PLUS', default_score).get(probe_date, 0)
        fund_flow_ignition = atomic.get('SCORE_FF_BULLISH_RESONANCE_S_PLUS', default_score).get(probe_date, 0)
        volatility_breakout = atomic.get('SCORE_VOL_BREAKOUT_POTENTIAL_S', default_score).get(probe_date, 0)
        fund_flow_conviction_breakout = atomic.get('SCORE_FF_PLAYBOOK_CONVICTION_BREAKOUT', default_score).get(probe_date, 0)

        # 计算通用共振
        general_ignition_resonance = (
            behavioral_ignition * structural_breakout * mechanics_ignition *
            chip_consensus_ignition * fund_flow_ignition * volatility_breakout
        )
        
        # 最终得分
        final_score = np.maximum.reduce([
            chip_playbook_ignition, 
            general_ignition_resonance,
            fund_flow_conviction_breakout
        ])

        print(f"  - 最终得分: {final_score:.4f}")
        print("  - 计算逻辑: 取以下三项的最大值")
        print(f"    1. 筹码剧本点火 (SCORE_CHIP_PLAYBOOK_VACUUM_BREAKOUT): {chip_playbook_ignition:.4f}")
        print(f"    2. 资金信念突破 (SCORE_FF_PLAYBOOK_CONVICTION_BREAKOUT): {fund_flow_conviction_breakout:.4f}")
        print(f"    3. 通用共振相乘: {general_ignition_resonance:.4f}")
        print("      -> 通用共振由以下信号相乘得到:")
        print(f"         - 行为共振: {behavioral_ignition:.4f}")
        print(f"         - 结构共振: {structural_breakout:.4f}")
        print(f"         - 力学共振: {mechanics_ignition:.4f}")
        print(f"         - 筹码共振: {chip_consensus_ignition:.4f}")
        print(f"         - 资金共振: {fund_flow_ignition:.4f}")
        print(f"         - 波动突破: {volatility_breakout:.4f}")
        
        if general_ignition_resonance < 0.01:
            bottleneck = min([
                ('行为', behavioral_ignition), ('结构', structural_breakout), ('力学', mechanics_ignition),
                ('筹码', chip_consensus_ignition), ('资金', fund_flow_ignition), ('波动', volatility_breakout)
            ], key=lambda item: item[1])
            print(f"  - [结论] 得分低是因为“通用共振”接近于0，主要瓶颈在于【{bottleneck[0]}共振】(分值: {bottleneck[1]:.4f})。")
        else:
            print("  - [结论] 通用共振分数正常，但可能低于其他两个剧本分。")

    # 为“终极确认”新增的探针
    def _deploy_ultimate_confirmation_probe(self, probe_date: pd.Timestamp):
        """【探针V1.0】解剖“终极确认”信号"""
        print("\n--- [探针] 正在解剖: 【认知S】终极看涨/底部确认 ---")
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=self.strategy.df_indicators.index)

        # 看涨确认
        fusion_bullish = atomic.get('COGNITIVE_FUSION_BULLISH_RESONANCE_S', default_score).get(probe_date, 0)
        pattern_bullish = atomic.get('SCORE_PATTERN_BULLISH_RESONANCE_S', default_score).get(probe_date, 0)
        final_bullish = fusion_bullish * pattern_bullish
        print(f"  - 终极看涨确认分: {final_bullish:.4f} (融合分 {fusion_bullish:.4f} * 形态分 {pattern_bullish:.4f})")

        # 底部确认
        fusion_bottom = atomic.get('COGNITIVE_FUSION_BOTTOM_REVERSAL_S', default_score).get(probe_date, 0)
        pattern_bottom = atomic.get('SCORE_PATTERN_BOTTOM_REVERSAL_S', default_score).get(probe_date, 0)
        final_bottom = fusion_bottom * pattern_bottom
        print(f"  - 终极底部确认分: {final_bottom:.4f} (融合分 {fusion_bottom:.4f} * 形态分 {pattern_bottom:.4f})")
        
        if pattern_bullish < 0.1 and pattern_bottom < 0.1:
            print("  - [结论] 得分极低的核心原因是【形态分】(pattern_score) 接近于0。")
            print("  - [诊断] 系统中缺少专门生成'PATTERN_...'系列信号的引擎，导致形态分始终为默认的低值。这是一个需要修复的系统性BUG。")
        else:
            print("  - [结论] 形态分正常，请检查融合分。")

    # 为“V型反转王牌”新增的探针
    def _deploy_v_reversal_ace_probe(self, probe_date: pd.Timestamp):
        """【探针V1.0】解剖“V型反转王牌”信号"""
        print("\n--- [探针] 正在解剖: 【战法S++】V型反转王牌 ---")
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=self.strategy.df_indicators.index)
        
        yesterday_date = probe_date - pd.Timedelta(days=1)
        while yesterday_date not in self.strategy.df_indicators.index and yesterday_date > self.strategy.df_indicators.index.min():
            yesterday_date -= pd.Timedelta(days=1)

        was_setup_yesterday = atomic.get('SCORE_SETUP_PANIC_SELLING_S', default_score).get(yesterday_date, 0)
        trigger_today = atomic.get('SCORE_BEHAVIOR_BOTTOM_REVERSAL_S_PLUS', default_score).get(probe_date, 0)
        final_score = was_setup_yesterday * trigger_today

        print(f"  - 最终得分: {final_score:.4f}")
        print(f"  - 计算逻辑: 昨日战备分 * 今日点火分")
        print(f"    - 昨日({yesterday_date.date()})恐慌抛售战备分: {was_setup_yesterday:.4f}")
        print(f"    - 今日({probe_date.date()})行为反转点火分: {trigger_today:.4f}")
        
        if was_setup_yesterday < 0.15: # 假设战备阈值为0.15
            print(f"  - [结论] 得分为0的核心原因是【昨日战备分】过低，未达到恐慌抛售的标准。")
        elif trigger_today < 0.5: # 假设点火阈值为0.5
            print(f"  - [结论] 得分为0的核心原因是【今日点火分】不足，反转形态不够强力。")
        else:
            print("  - [结论] 战备与点火条件均满足，得分正常。")

    # 为“筹码价格滞后”新增的探针
    def _deploy_chip_price_lag_probe(self, probe_date: pd.Timestamp):
        """【探针V1.0】解剖“筹码价格滞后”信号"""
        print("\n--- [探针] 正在解剖: 【战法S】筹码价格滞后 ---")
        atomic = self.strategy.atomic_states
        df = self.strategy.df_indicators
        default_score = pd.Series(0.0, index=df.index)
        
        yesterday_date = probe_date - pd.Timedelta(days=1)
        while yesterday_date not in df.index and yesterday_date > df.index.min():
            yesterday_date -= pd.Timedelta(days=1)

        chip_resonance_score = atomic.get('SCORE_CHIP_BULLISH_RESONANCE_S_PLUS', default_score).get(yesterday_date, 0)
        price_momentum_suppressed_score = normalize_score(df['SLOPE_5_close_D'], df.index, window=60, ascending=False).get(yesterday_date, 0)
        volatility_compression_score = atomic.get('COGNITIVE_SCORE_VOL_COMPRESSION_FUSED', default_score).get(yesterday_date, 0)
        
        setup_score_yesterday = chip_resonance_score * price_momentum_suppressed_score * volatility_compression_score
        trigger_score_today = atomic.get('COGNITIVE_SCORE_EARLY_MOMENTUM_IGNITION_A', default_score).get(probe_date, 0)
        final_score = setup_score_yesterday * trigger_score_today

        print(f"  - 最终得分: {final_score:.4f}")
        print(f"  - 计算逻辑: 昨日战备分 * 今日点火分")
        print(f"    - 今日({probe_date.date()})价格启动点火分: {trigger_score_today:.4f}")
        print(f"    - 昨日({yesterday_date.date()})综合战备分: {setup_score_yesterday:.4f}")
        print("      -> 昨日战备分由以下信号相乘得到:")
        print(f"         - 筹码共振分: {chip_resonance_score:.4f}")
        print(f"         - 价格压制分: {price_momentum_suppressed_score:.4f}")
        print(f"         - 波动压缩分: {volatility_compression_score:.4f}")

        if setup_score_yesterday < 0.2: # 假设战备阈值为0.2
            bottleneck = min([
                ('筹码共振', chip_resonance_score), ('价格压制', price_momentum_suppressed_score), ('波动压缩', volatility_compression_score)
            ], key=lambda item: item[1])
            print(f"  - [结论] 得分低的核心原因是【昨日战备分】不足，主要瓶颈在于【{bottleneck[0]}】(分值: {bottleneck[1]:.4f})。")
        else:
            print("  - [结论] 昨日战备分充足，请检查今日点火分。")

    # 为“波动突破”信号新增的专属探针
    def _deploy_volatility_breakout_probe(self, probe_date: pd.Timestamp):
        """【探针V1.1 · 算法同步版】解剖“波动突破潜力”信号"""
        print("\n--- [探针] 正在解剖: 【认知】波动突破潜力 (多域点火共振的瓶颈) ---")
        atomic = self.strategy.atomic_states
        df = self.strategy.df_indicators
        default_score = pd.Series(0.0, index=df.index)
        
        final_score = atomic.get('SCORE_VOL_BREAKOUT_POTENTIAL_S', default_score).get(probe_date, 0)
        
        # 反推其构成组件
        compression_score = atomic.get('COGNITIVE_SCORE_VOL_COMPRESSION_FUSED', default_score).get(probe_date, 0)
        
        # 匹配新的扩张分计算逻辑
        bbw_slope_score = normalize_score(df.get('SLOPE_5_BBW_21_2.0_D'), df.index, 120, ascending=True).get(probe_date, 0.5)
        bbw_accel_score = normalize_score(df.get('ACCEL_5_BBW_21_2.0_D'), df.index, 120, ascending=True).get(probe_date, 0.5)
        expansion_score = (bbw_slope_score * 0.6 + bbw_accel_score * 0.4)
        
        print(f"  - 最终得分: {final_score:.4f}")
        print(f"  - 计算逻辑: (波动压缩分 * 波动扩张分) ^ 0.5")
        print(f"    - 波动压缩分 (COGNITIVE_SCORE_VOL_COMPRESSION_FUSED): {compression_score:.4f}")
        print(f"    - 波动扩张分 (融合了斜率和加速度): {expansion_score:.4f}")
        print(f"      -> 斜率分: {bbw_slope_score:.4f}")
        print(f"      -> 加速度分: {bbw_accel_score:.4f}")
        
        if expansion_score < 0.5:
            bottleneck = min([('斜率分', bbw_slope_score), ('加速度分', bbw_accel_score)], key=lambda item: item[1])
            print(f"  - [结论] 扩张分偏低的核心瓶颈在于【{bottleneck[0]}】(分值: {bottleneck[1]:.4f})。")
        else:
            print("  - [结论] 扩张分正常。")

    # 全新的终极反转信号探针
    def _deploy_ultimate_reversal_probe(self, probe_date: pd.Timestamp, domain: str):
        """
        【探针V1.3 · 直连引擎版】解剖终极反转信号
        - 核心修复: 不再尝试重构数据，而是直接读取由各情报引擎缓存的 `__<DOMAIN>_overall_health` 数据。
                    这确保了探针的绝对准确性，并解决了之前因无法访问内部状态而报告错误信息的问题。
        """
        domain_upper = domain.upper()
        signal_name = f'SCORE_{domain_upper}_BOTTOM_REVERSAL_S_PLUS'
        print(f"\n--- [探针] 正在解剖: 【终极信号】{signal_name} ---")

        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index)

        # --- 步骤 1: 获取最终得分和其两大组成部分 (逻辑不变) ---
        final_score = atomic.get(signal_name, default_score).get(probe_date, 0.0)
        print(f"  - 当日最终得分: {final_score:.4f}")

        p_conf_domain = get_params_block(self.strategy, f'{domain.lower()}_dynamics_params' if domain_upper == 'BEHAVIOR' else f'{domain.lower()}_ultimate_params', {})
        bonus_factor = get_param_value(p_conf_domain.get('bottom_context_bonus_factor'), 0.5)
        reversal_tf_weights = get_param_value(p_conf_domain.get('reversal_tf_weights'), {'short': 0.6, 'medium': 0.3, 'long': 0.1})
        periods = get_param_value(p_conf_domain.get('periods'), [1, 5, 13, 21, 55])
        
        context_score, _ = calculate_context_scores(df, atomic)
        context_score_today = context_score.get(probe_date, 0.0)
        
        trigger_denominator = 1 + context_score_today * bonus_factor
        trigger_score = final_score / trigger_denominator if trigger_denominator != 0 else 0.0

        print(f"  - 计算逻辑: Trigger Score * (1 + Context Score * Bonus Factor)")
        print(f"    - 底部情景分 (Context): {context_score_today:.4f}")
        print(f"    - 奖励因子 (Bonus Factor): {bonus_factor:.2f}")
        print(f"    - 反推得到的看涨触发分 (Trigger): {trigger_score:.4f}")

        # --- 步骤 2: 解剖 Trigger Score 的构成 (全新逻辑) ---
        # 直接从 atomic_states 读取引擎缓存的 overall_health
        overall_health_cache_key = f'__{domain_upper}_overall_health'
        overall_health = atomic.get(overall_health_cache_key)
        
        if not overall_health:
             print(f"  - [探针错误] 致命错误: 未能在 atomic_states 中找到缓存 '{overall_health_cache_key}'。请确保 {domain} 引擎已正确缓存其内部状态。")
             return

        # 使用新的反转健康度逻辑
        bullish_reversal_health = {p: overall_health['bearish_static'][p].get(probe_date, 0.5) * overall_health['bullish_dynamic'][p].get(probe_date, 0.5) for p in periods}

        short_force = (bullish_reversal_health.get(1, 0.5) * bullish_reversal_health.get(5, 0.5))**0.5
        medium_trend = (bullish_reversal_health.get(13, 0.5) * bullish_reversal_health.get(21, 0.5))**0.5
        long_inertia = bullish_reversal_health.get(55, 0.5)

        print(f"  - 看涨触发分 (Trigger) 由三股力量加权构成:")
        print(f"    - 短期反转力 (权重 {reversal_tf_weights['short']}): {short_force:.4f}")
        print(f"    - 中期反转力 (权重 {reversal_tf_weights['medium']}): {medium_trend:.4f}")
        print(f"    - 长期反转力 (权重 {reversal_tf_weights['long']}): {long_inertia:.4f}")

        # --- 步骤 3 & 4 (全新逻辑) ---
        forces = {'短期': short_force, '中期': medium_trend, '长期': long_inertia}
        main_force_name = max(forces, key=forces.get)
        print(f"  - 主要贡献力量来自【{main_force_name}反转力】(分值: {forces[main_force_name]:.4f})")

        if main_force_name == '短期': p1, p2 = 1, 5
        elif main_force_name == '中期': p1, p2 = 13, 21
        else: p1, p2 = 55, 55
        
        print(f"    -> 该力由 {p1}日 和 {p2}日 的 '反转健康度' 融合得到:")
        print(f"       - {p1}日反转健康度: {bullish_reversal_health.get(p1, 0.5):.4f}")
        print(f"       - {p2}日反转健康度: {bullish_reversal_health.get(p2, 0.5):.4f}")
        
        static_bearish_p1 = overall_health['bearish_static'][p1].get(probe_date, 0.5)
        dynamic_bullish_p1 = overall_health['bullish_dynamic'][p1].get(probe_date, 0.5)
        print(f"    -> {p1}日反转健康度由以下两者相乘得到:")
        print(f"       - {p1}日静态看跌分: {static_bearish_p1:.4f}")
        print(f"       - {p1}日动态看涨分: {dynamic_bullish_p1:.4f}")

        if static_bearish_p1 < 0.5:
             print(f"  - [最终结论] {signal_name} 分数低的核心原因是【静态看跌分】不足，市场并未处于公认的弱势/超卖状态。")
        elif dynamic_bullish_p1 < 0.5:
             print(f"  - [最终结论] {signal_name} 分数低的核心原因是【动态看涨分】不足，未能形成有效的向上攻击力。")
        else:
             print(f"  - [最终结论] {signal_name} 在当日维持高分的根源在于【静态看跌分】和【动态看涨分】同时处于高位。")

    # 彻底重构风险探针，实现“钻透式”解剖
    def _deploy_risk_resonance_probe(self, probe_date: pd.Timestamp, domain: str):
        """
        【探针V4.3 · 健壮性修复版】风险溯源法医探针
        - BUG修复: 彻底修复了调用 BEHAVIOR 和 FOUNDATION 引擎健康度计算器时错误的参数传递和数据类型问题。
        """
        domain_upper = domain.upper()
        signal_name = f'SCORE_{domain_upper}_BEARISH_RESONANCE_S_PLUS'
        print(f"\n--- [风险探针 V4.3] 正在解剖: 【终极风险信号】{signal_name} ---")

        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        
        final_score = atomic.get(signal_name, pd.Series(0.0, index=df.index)).get(probe_date, 0.0)
        print(f"  - 当日最终得分: {final_score:.4f}")

        overall_health_cache_key = f'__{domain_upper}_overall_health'
        overall_health = atomic.get(overall_health_cache_key)
        
        if not overall_health:
            print(f"  - [探针错误] 致命错误: 未能在 atomic_states 中找到缓存 '{overall_health_cache_key}'。")
            return

        period_to_probe = 13
        
        if period_to_probe not in overall_health.get('bearish_static', {}) or period_to_probe not in overall_health.get('bearish_dynamic', {}):
            print(f"  - [探针警告] 在 {domain_upper} 领域的 overall_health 中缺少周期 {period_to_probe} 的数据。")
            return

        s_bear_score = overall_health['bearish_static'][period_to_probe].get(probe_date, 0.0)
        d_bear_score = overall_health['bearish_dynamic'][period_to_probe].get(probe_date, 0.0)
        
        print(f"  - 解剖核心逻辑 (以{period_to_probe}日周期为例): 看跌共振分 ≈ s_bear * d_bear")
        print(f"    - {period_to_probe}日静态看跌分 (s_bear): {s_bear_score:.4f}")
        print(f"    - {period_to_probe}日动态看跌分 (d_bear): {d_bear_score:.4f}")

        bottleneck_type = 's_bear' if s_bear_score < d_bear_score else 'd_bear'
        bottleneck_score = s_bear_score if bottleneck_type == 's_bear' else d_bear_score
        print(f"  - [定位瓶颈] 【{bottleneck_type}】分数({bottleneck_score:.4f}) 更低，是主要问题所在。")
        
        print(f"    -> 开始对 {domain_upper} 领域的【{bottleneck_type}】进行钻透式解剖...")

        engine_map = {
            'FF': self.fund_flow_intel, 'CHIP': self.chip_intel, 'DYN': self.mechanics_engine,
            'BEHAVIOR': self.behavioral_intel, 'STRUCTURE': self.structural_intel, 'FOUNDATION': self.foundation_intel
        }
        engine_instance = engine_map.get(domain_upper)
        if not engine_instance: return

        calc_map = {
            'DYN': [('_calculate_volatility_health', '波动率'), ('_calculate_efficiency_health', '效率'), ('_calculate_kinetic_energy_health', '动能'), ('_calculate_inertia_health', '惯性')],
            'BEHAVIOR': [('_calculate_price_health', '价格'), ('_calculate_volume_health', '成交量'), ('_calculate_kline_pattern_health', 'K线形态')],
            'STRUCTURE': [('_calculate_ma_health', '均线'), ('_calculate_mechanics_health', '力学'), ('_calculate_mtf_health', '多周期'), ('_calculate_pattern_health', '形态')],
            'FOUNDATION': [('_calculate_ema_health', 'EMA'), ('_calculate_rsi_health', 'RSI'), ('_calculate_macd_health', 'MACD'), ('_calculate_cmf_health', 'CMF')]
        }
        
        for calc_func_name, pillar_cn_name in calc_map.get(domain_upper, []):
            try:
                calculator = getattr(engine_instance, calc_func_name)
                
                periods_arg = [period_to_probe]
                # 修复了所有引擎的探针调用逻辑
                if domain_upper == 'BEHAVIOR':
                    atomic_signals_for_behavior = engine_instance._generate_all_atomic_signals(df)
                    if calc_func_name in ['_calculate_price_health', '_calculate_volume_health']:
                        s_bull, d_bull, s_bear, d_bear = calculator(df, 120, 24, {'slope': 0.6, 'accel': 0.4}, periods_arg)
                    else:
                        s_bull, d_bull, s_bear, d_bear = calculator(df, atomic_signals_for_behavior, 120, 24, periods_arg)
                elif domain_upper == 'STRUCTURE':
                    s_bull, d_bull, s_bear, d_bear = calculator(df, periods_arg, 120, {'slope': 0.6, 'accel': 0.4})
                else: # DYN, FOUNDATION, CHIP
                    s_bull, d_bull, s_bear, d_bear = calculator(df, 120, {'slope': 0.6, 'accel': 0.4}, periods_arg)

                pillar_score_series = s_bear.get(period_to_probe) if bottleneck_type == 's_bear' else d_bear.get(period_to_probe)
                if pillar_score_series is None: continue
                pillar_score = pillar_score_series.get(probe_date, 0.0)
                print(f"       - {pillar_cn_name:<12s} 支柱贡献分: {pillar_score:.4f}")

            except Exception as e:
                print(f"       - [探针错误] 解剖支柱 '{pillar_cn_name}' 失败: {e}")
        
        print(f"  - [最终诊断] {domain_upper} 风险分低，根源在于其构成支柱的【{bottleneck_type}】分数，在现有“相对归一化”逻辑下被历史数据“平均化”，无法体现当日的绝对风险。")

    def deploy_nan_forensics_probe(self, nan_date, nan_signal_name: str):
        """
        【V1.0 新增】NaN 值法医探针。
        当检测到 NaN 时，此方法被调用，以追溯并解剖导致问题的信号计算链。
        """
        print("\n" + "="*30 + f" [NaN 法医探针 V1.0 启动] " + "="*30)
        print(f"  - 案发时间: {nan_date.strftime('%Y-%m-%d')}")
        print(f"  - 可疑信号: {nan_signal_name}")
        print("  - 开始进行计算链路回溯解剖...")
        print("-" * 80)

        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states

        def get_val(signal_name, date, source_dict=atomic):
            """安全地获取并打印一个值"""
            val = source_dict.get(signal_name, pd.Series(np.nan, index=df.index)).get(date, np.nan)
            print(f"    -> 读取 '{signal_name}': {val}")
            return val

        def probe_pillar_health(engine_intel, date, period, health_type):
            """通用支柱健康度探针"""
            # 这是一个简化的示例，实际需要根据每个引擎的结构来定制
            # 这里我们假设可以访问到引擎的内部计算方法或存储的中间结果
            # 为了简化，我们直接从 atomic_states 或 df_indicators 中读取最终的指标
            print(f"  ---> 解剖 {health_type} 健康度 (周期 {period})...")
            # 示例：追溯到最原始的 slope 和 accel 指标
            # 这需要根据具体信号的计算逻辑来确定原始指标名
            # 例如，对于 DYN 引擎的波动率健康度
            if "DYN" in nan_signal_name:
                raw_slope = df.get(f'SLOPE_{period}_BBW_21_2.0_D', pd.Series(np.nan)).get(date)
                raw_accel = df.get(f'ACCEL_{period}_BBW_21_2.0_D', pd.Series(np.nan)).get(date)
                print(f"      ----> 原始指标 SLOPE_{period}_BBW_21_2.0_D: {raw_slope}")
                print(f"      ----> 原始指标 ACCEL_{period}_BBW_21_2.0_D: {raw_accel}")
                if pd.isna(raw_slope) or pd.isna(raw_accel):
                    print("      ------> [!!!] 发现源头 NaN！问题可能出在基础指标计算层。")

        # 根据信号名选择解剖路径
        if "CHIP" in nan_signal_name or "DYN" in nan_signal_name or "STRUCTURE" in nan_signal_name or "FOUNDATION" in nan_signal_name:
            print("  --> 检测到筹码层信号，开始解剖 ChipIntelligence...")
            # 简化解剖过程：直接检查构成 overall_health 的所有 pillar health
            # 这是一个示例，实际探针可以做得更精细
            print("  --> 正在检查所有筹码支柱的健康度贡献...")
            for p in self.chip_intel.diagnose_unified_chip_signals.__defaults__[2]: # 获取默认periods
                for ht in ['bullish_static', 'bullish_dynamic', 'bearish_static', 'bearish_dynamic']:
                    # 模拟计算 overall_health 的过程并打印
                    # 此处仅为示意，实际需要更精细的逻辑来重现计算
                    pass
            print("  --> 提示: 请检查 chip_intelligence.py 中各 _calculate_..._health 方法的 normalize_score 输入是否存在NaN。")

        elif "PLAYBOOK" in nan_signal_name or "COGNITIVE" in nan_signal_name or "TACTIC" in nan_signal_name:
            print(f"  --> 检测到认知/战术层信号，开始解剖 {nan_signal_name}...")
            
            if nan_signal_name == 'SCORE_PLAYBOOK_MEAN_REVERSION_GRID_BUY_A':
                print("  ---> 解剖路径: final_score = context * opportunity")
                
                # 1. 解剖 context_is_ranging_market
                print("  -----> 正在解剖 context_is_ranging_market...")
                is_cyclical_regime = get_val('SCORE_CYCLICAL_REGIME', nan_date) > 0.4 # 假设阈值
                is_not_trending_regime = get_val('SCORE_TRENDING_REGIME_FFT', nan_date) < 0.45 # 假设阈值
                context_val = float(is_cyclical_regime and is_not_trending_regime)
                print(f"      - context_is_ranging_market = {context_val}")

                # 2. 解剖 buy_opportunity_score
                print("  -----> 正在解剖 buy_opportunity_score...")
                bbp_val = df.get('BBP_21_2.0_D', pd.Series(np.nan)).get(nan_date)
                print(f"      - 原始指标 BBP_21_2.0_D: {bbp_val}")
                if pd.isna(bbp_val):
                    print("      ------> [!!!] 发现源头 NaN！问题出在 BBP_21_2.0_D 指标计算。")
                opportunity_val = 1 - np.clip(bbp_val, 0, 1) if pd.notna(bbp_val) else np.nan
                print(f"      - buy_opportunity_score = {opportunity_val}")

                # 3. 最终计算
                final_val = context_val * opportunity_val
                print(f"  ---> 最终验算: {context_val} * {opportunity_val} = {final_val}")

        # 可以为其他引擎添加 elif 分支
        # ...

        else:
            print("  --> 未找到特定引擎的解剖路径，执行通用检查...")
            print("  --> 正在检查该信号在 atomic_states 中的值...")
            get_val(nan_signal_name, nan_date)

        print("-" * 80)
        print(f"  - 解剖完毕。请重点关注报告中值为 'NaN' 的步骤，其上一步即为问题源头。")
        print("=" * 80 + "\n")
















