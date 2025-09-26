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
        【探针V1.1 · 健壮性修复版】解剖终极反转信号 (以指定领域为例)
        - 本次修改:
          - [修复] 解决了因 `dynamic_weights` 为 None 导致的 'NoneType' object is not subscriptable 错误。
                   现在会从配置文件中健壮地获取参数，并提供默认值。
          - [重构] 优化了探针逻辑，不再重新执行庞大的终极信号诊断函数，而是直接调用
                   更底层的健康度计算方法，并手动传入所有必需参数，提高了探针的独立性和效率。
        """
        domain_upper = domain.upper()
        signal_name = f'SCORE_{domain_upper}_BOTTOM_REVERSAL_S_PLUS'
        print(f"\n--- [探针] 正在解剖: 【终极信号】{signal_name} ---")

        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index)

        # --- 步骤 1: 获取最终得分和其两大组成部分 ---
        final_score = atomic.get(signal_name, default_score).get(probe_date, 0.0)
        print(f"  - 当日最终得分: {final_score:.4f}")

        # 健壮地获取参数，并提供默认值
        # 尝试从特定领域参数块获取，如果失败，则从通用块获取
        p_conf_domain = get_params_block(self.strategy, f'{domain.lower()}_dynamics_params' if domain_upper == 'BEHAVIOR' else f'{domain.lower()}_ultimate_params', {})
        p_conf_generic = get_params_block(self.strategy, 'trend_quality_params', {}) # 假设通用参数在这里
        
        bonus_factor = get_param_value(p_conf_domain.get('bottom_context_bonus_factor'), 0.5)
        dynamic_weights = get_param_value(p_conf_domain.get('dynamic_weights'), {'slope': 0.6, 'accel': 0.4})
        reversal_tf_weights = get_param_value(p_conf_domain.get('reversal_tf_weights'), {'short': 0.6, 'medium': 0.3, 'long': 0.1})
        periods = get_param_value(p_conf_domain.get('periods'), [1, 5, 13, 21, 55])
        norm_window = get_param_value(p_conf_domain.get('norm_window'), 120)
        
        context_score, _ = calculate_context_scores(df, atomic)
        context_score_today = context_score.get(probe_date, 0.0)
        
        trigger_denominator = 1 + context_score_today * bonus_factor
        trigger_score = final_score / trigger_denominator if trigger_denominator != 0 else 0.0

        print(f"  - 计算逻辑: Trigger Score * (1 + Context Score * Bonus Factor)")
        print(f"    - 底部情景分 (Context): {context_score_today:.4f}")
        print(f"    - 奖励因子 (Bonus Factor): {bonus_factor:.2f}")
        print(f"    - 反推得到的看涨触发分 (Trigger): {trigger_score:.4f}")

        # --- 步骤 2: 解剖 Trigger Score 的构成 ---
        engine_map = {
            'BEHAVIOR': self.behavioral_intel, 'CHIP': self.chip_intel,
            'DYN': self.mechanics_engine, 'FOUNDATION': self.foundation_intel,
            'STRUCTURE': self.structural_intel,
        }
        engine = engine_map.get(domain_upper)
        if not engine:
            print(f"  - [错误] 未找到领域 '{domain}' 对应的引擎实例。")
            return

        # 重构此部分，直接调用健康度计算器，而不是整个终极信号函数
        # 1. 获取所有支柱的健康度
        health_data = {'bullish_dynamic': []}
        if domain_upper == 'BEHAVIOR':
            calculators = {'price': engine._calculate_price_health, 'volume': engine._calculate_volume_health, 'kline': engine._calculate_kline_pattern_health}
            atomic_signals = engine._generate_all_atomic_signals(df) # kline需要
            for name, calculator in calculators.items():
                if name == 'kline':
                    _, d_bull, _, _ = calculator(df, atomic_signals, norm_window, 24, periods)
                else:
                    _, d_bull, _, _ = calculator(df, norm_window, 24, dynamic_weights, periods)
                health_data['bullish_dynamic'].append(d_bull)
        # ... 此处可以为其他 domain 添加 elif ...
        else:
             print(f"  - [警告] 领域 '{domain}' 的详细支柱探针尚未实现。")
             # 即使未实现，也需要一个空的默认值以避免后续错误
             bullish_dynamic_health = {p: 0.5 for p in periods}

        # 2. 融合生成 overall_health['bullish_dynamic']
        overall_bullish_dynamic = {}
        for p in periods:
            components = [pillar_dict[p].values for pillar_dict in health_data['bullish_dynamic'] if p in pillar_dict]
            if components:
                stacked_values = np.stack(components, axis=0)
                # 根据不同引擎的融合逻辑（加权或平均）
                if domain_upper == 'BEHAVIOR':
                    dim_weights = get_param_value(p_conf_domain.get('dimension_weights'), {'price': 0.4, 'volume': 0.3, 'kline': 0.3})
                    weights_array = np.array([dim_weights['price'], dim_weights['volume'], dim_weights['kline']])
                    fused_values = np.sum(stacked_values * weights_array[:, np.newaxis], axis=0)
                else: # 其他引擎默认是平均
                    fused_values = np.mean(stacked_values, axis=0)
                overall_bullish_dynamic[p] = pd.Series(fused_values, index=df.index)
            else:
                overall_bullish_dynamic[p] = pd.Series(0.5, index=df.index)

        bullish_dynamic_health = {p: s.get(probe_date, 0.5) for p, s in overall_bullish_dynamic.items()}

        short_force = (bullish_dynamic_health.get(1, 0.5) * bullish_dynamic_health.get(5, 0.5))**0.5
        medium_trend = (bullish_dynamic_health.get(13, 0.5) * bullish_dynamic_health.get(21, 0.5))**0.5
        long_inertia = bullish_dynamic_health.get(55, 0.5)

        print(f"  - 看涨触发分 (Trigger) 由三股力量加权构成:")
        print(f"    - 短期看涨力 (权重 {reversal_tf_weights['short']}): {short_force:.4f}")
        print(f"    - 中期看涨力 (权重 {reversal_tf_weights['medium']}): {medium_trend:.4f}")
        print(f"    - 长期看涨力 (权重 {reversal_tf_weights['long']}): {long_inertia:.4f}")

        # --- 步骤 3 & 4 (逻辑不变，但现在基于更可靠的数据) ---
        forces = {'短期': short_force, '中期': medium_trend, '长期': long_inertia}
        main_force_name = max(forces, key=forces.get)
        print(f"  - 主要贡献力量来自【{main_force_name}看涨力】(分值: {forces[main_force_name]:.4f})")

        if main_force_name == '短期': p1, p2 = 1, 5
        elif main_force_name == '中期': p1, p2 = 13, 21
        else: p1, p2 = 55, 55
        
        print(f"    -> 该力由 {p1}日 和 {p2}日 的 '动态看涨分' 融合得到:")
        print(f"       - {p1}日动态看涨分: {bullish_dynamic_health.get(p1, 0.5):.4f}")
        print(f"       - {p2}日动态看涨分: {bullish_dynamic_health.get(p2, 0.5):.4f}")

        if domain_upper == 'BEHAVIOR':
            price_score = health_data['bullish_dynamic'][0][p1].get(probe_date, 0.5)
            vol_score = health_data['bullish_dynamic'][1][p1].get(probe_date, 0.5)
            kline_score = health_data['bullish_dynamic'][2][p1].get(probe_date, 0.5)
            
            dim_weights = get_param_value(p_conf_domain.get('dimension_weights'), {'price': 0.4, 'volume': 0.3, 'kline': 0.3})
            print(f"    -> {p1}日动态看涨分由以下维度加权构成:")
            print(f"       - 价格维度 (权重 {dim_weights['price']}): {price_score:.4f}")
            print(f"       - 成交量维度 (权重 {dim_weights['volume']}): {vol_score:.4f}")
            print(f"       - K线形态维度 (权重 {dim_weights['kline']}): {kline_score:.4f}")
            
            root_cause_dim = max({'价格': price_score, '成交量': vol_score, 'K线': kline_score}.items(), key=lambda item: item[1])
            print(f"  - [最终结论] {signal_name} 在当日维持高分的根源在于【{root_cause_dim[0]}维度】的动态看涨分持续强势 (分值: {root_cause_dim[1]:.4f})。")

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
















