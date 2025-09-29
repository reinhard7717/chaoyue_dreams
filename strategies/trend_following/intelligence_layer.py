# 文件: strategies/trend_following/intelligence_layer.py
# 情报层总指挥官 (重构版)
import pandas as pd
import numpy as np
import pandas_ta as ta
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
from .intelligence.process_intelligence import ProcessIntelligence
from strategies.trend_following.utils import get_params_block, get_param_value, calculate_context_scores, normalize_score, normalize_to_bipolar

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
        self.process_intel = ProcessIntelligence(self.strategy)
        self.cognitive_intel = CognitiveIntelligence(self.strategy)
        self.playbook_engine = PlaybookEngine(self.strategy)
        self.exit_layer = ExitLayer(self.strategy)

    def run_all_diagnostics(self) -> Dict:
        """
        【V411.1 · 指挥链修复版】情报层总入口。
        - 核心修复: 确保 IntelligenceLayer 正确接收并更新所有情报引擎返回的信号字典，
                      彻底修复因部分引擎返回结果被忽略而导致的数据流中断问题。
        """
        print("--- [情报层总指挥官 V411.1 · 指挥链修复版] 开始执行所有诊断模块... ---") # 更新版本号
        df = self.strategy.df_indicators
        self.strategy.atomic_states = {}
        self.strategy.trigger_events = {}
        self.strategy.playbook_states = {}
        self.strategy.exit_triggers = pd.DataFrame(index=df.index)

        def update_states(new_states: Dict):
            if isinstance(new_states, dict):
                self.strategy.atomic_states.update(new_states)
            # [代码新增] 增加一个调试打印，用于追踪信号来源
            # else:
            #     print(f"[IntelligenceLayer 警告] update_states 收到非字典类型: {type(new_states)}")

        # --- 阶段一: 基础信号生成 (按依赖关系重构顺序) ---
        # print("    - [阶段 1/5] 正在执行基础信号生成...")
        
        # 1. 首先运行周期引擎
        update_states(self.cyclical_intel.run_cyclical_analysis_command(df))
        
        # --- 阶段 1.5: 基础过程诊断 ---
        # print("    - [阶段 1.5/5] 正在执行基础过程诊断 (分析原始数据)...")
        base_process_states = self.process_intel.run_process_diagnostics(task_type_filter='base')
        update_states(base_process_states)

        # 2. 运行其他所有状态情报引擎
        # print("    - [阶段 2/5] 正在执行状态情报引擎...")
        update_states(self.foundation_intel.run_foundation_analysis_command())
        update_states(self.chip_intel.run_chip_intelligence_command(df))
        update_states(self.structural_intel.diagnose_structural_states(df))
        
        # 使用 update_states 接收 behavioral_intel 返回的战报
        update_states(self.behavioral_intel.run_behavioral_analysis_command())
        
        update_states(self.fund_flow_intel.diagnose_fund_flow_states(df))
        
        # 使用 update_states 接收 mechanics_engine 返回的战报
        update_states(self.mechanics_engine.run_dynamic_analysis_command())
        
        update_states(self.pattern_intel.run_pattern_analysis_command(df))
        
        # --- 阶段 2.5: 战略过程诊断 ---
        # print("    - [阶段 2.5/5] 正在执行战略过程诊断 (分析高阶信号)...")
        strategy_process_states = self.process_intel.run_process_diagnostics(task_type_filter='strategy')
        update_states(strategy_process_states)
        
        # --- 阶段三: 跨域认知融合 ---
        # print("    - [阶段 3/5] 正在执行认知层跨域元融合...")
        self.cognitive_intel.synthesize_cognitive_scores(df, pullback_enhancements={})

        # --- 阶段四: 最终战法与剧本生成 ---
        # print("    - [阶段 4/5] 正在生成最终战法与剧本...")
        trigger_events = self.playbook_engine.define_trigger_events(df)
        self.strategy.trigger_events.update(trigger_events)
        _, playbook_states = self.playbook_engine.generate_playbook_states(self.strategy.trigger_events)
        self.strategy.playbook_states.update(playbook_states)
        
        # --- 阶段五: 硬性离场信号生成 ---
        # print("    - [阶段 5/5] 正在生成硬性离场信号...")
        exit_triggers_df = self.exit_layer.generate_hard_exit_triggers()
        self.strategy.exit_triggers = exit_triggers_df

        self.deploy_forensic_probes()
        
        print("--- [情报层总指挥官 V411.1] 所有诊断模块执行完毕。 ---")
        return self.strategy.trigger_events

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

    def deploy_forensic_probes(self):
        """
        【V1.3 · 终极探针部署版】法医探针调度中心
        - 核心升级: 部署了全新的“终极信号钻透式探针”，用于解剖“信号躺平”问题。
        """
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        if not debug_params.get('enabled', False):
            return
        probe_date_str = debug_params.get('probe_date')
        if not probe_date_str:
            return
        probe_date = pd.to_datetime(probe_date_str)
        # 时区校准逻辑
        if self.strategy.df_indicators.index.tz is not None:
            try:
                probe_date = probe_date.tz_localize(self.strategy.df_indicators.index.tz)
            except Exception:
                try:
                    probe_date = probe_date.tz_convert(self.strategy.df_indicators.index.tz)
                except Exception as e_conv:
                     print(f"    -> [法医探针] 错误: 转换探针日期时区也失败: {e_conv}。")
                     return
        if probe_date not in self.strategy.df_indicators.index:
            print(f"    -> [法医探针] 警告: 探针日期 {probe_date_str} (校准后: {probe_date}) 不在数据索引中，跳过探针部署。")
            return
        print("\n" + "="*30 + f" [法医探针部署中心 V1.3] 正在解剖 {probe_date_str} " + "="*30)
        
        # [代码新增] 部署全新的“终极信号钻透式探针”，专门用于解决“信号躺平”问题
        # 您可以修改这里的参数，来解剖任何一个“躺平”的信号
        self._deploy_ultimate_signal_drill_down_probe(probe_date, domain='CHIP', signal_type='BULLISH_RESONANCE')
        self._deploy_ultimate_signal_drill_down_probe(probe_date, domain='BEHAVIOR', signal_type='BULLISH_RESONANCE')
        # 您也可以解剖看跌信号
        # self._deploy_ultimate_signal_drill_down_probe(probe_date, domain='DYN', signal_type='BEARISH_RESONANCE')
        self._deploy_process_intelligence_probe(probe_date)
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
        【探针V1.4 · 键名校准版】解剖终极反转信号
        - BUG修复: 修正了在访问 `overall_health` 缓存时使用错误键名的问题。
        """
        domain_upper = domain.upper()
        signal_name = f'SCORE_{domain_upper}_BOTTOM_REVERSAL_S_PLUS'
        print(f"\n--- [探针] 正在解剖: 【终极信号】{signal_name} ---")

        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index)

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

        overall_health_cache_key = f'__{domain_upper}_overall_health'
        overall_health = atomic.get(overall_health_cache_key)
        
        if not overall_health:
             print(f"  - [探针错误] 致命错误: 未能在 atomic_states 中找到缓存 '{overall_health_cache_key}'。")
             return

        # 使用正确的键名 's_bear' 和 'd_bull'
        bullish_reversal_health = {p: overall_health['s_bear'][p].get(probe_date, 0.5) * overall_health['d_bull'][p].get(probe_date, 0.5) for p in periods}

        short_force = (bullish_reversal_health.get(1, 0.5) * bullish_reversal_health.get(5, 0.5))**0.5
        medium_trend = (bullish_reversal_health.get(13, 0.5) * bullish_reversal_health.get(21, 0.5))**0.5
        long_inertia = bullish_reversal_health.get(55, 0.5)

        print(f"  - 看涨触发分 (Trigger) 由三股力量加权构成:")
        print(f"    - 短期反转力 (权重 {reversal_tf_weights['short']}): {short_force:.4f}")
        print(f"    - 中期反转力 (权重 {reversal_tf_weights['medium']}): {medium_trend:.4f}")
        print(f"    - 长期反转力 (权重 {reversal_tf_weights['long']}): {long_inertia:.4f}")

        forces = {'短期': short_force, '中期': medium_trend, '长期': long_inertia}
        main_force_name = max(forces, key=forces.get)
        print(f"  - 主要贡献力量来自【{main_force_name}反转力】(分值: {forces[main_force_name]:.4f})")

        if main_force_name == '短期': p1, p2 = 1, 5
        elif main_force_name == '中期': p1, p2 = 13, 21
        else: p1, p2 = 55, 55
        
        print(f"    -> 该力由 {p1}日 和 {p2}日 的 '反转健康度' 融合得到:")
        print(f"       - {p1}日反转健康度: {bullish_reversal_health.get(p1, 0.5):.4f}")
        print(f"       - {p2}日反转健康度: {bullish_reversal_health.get(p2, 0.5):.4f}")
        
        # 使用正确的键名 's_bear' 和 'd_bull'
        static_bearish_p1 = overall_health['s_bear'][p1].get(probe_date, 0.5)
        dynamic_bullish_p1 = overall_health['d_bull'][p1].get(probe_date, 0.5)
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
        【探针V4.4 · 键名校准版】风险溯源法医探针
        - BUG修复: 彻底修复了探针在访问 `overall_health` 缓存时使用错误键名 ('bearish_static' -> 's_bear') 的问题。
        """
        domain_upper = domain.upper()
        signal_name = f'SCORE_{domain_upper}_BEARISH_RESONANCE_S_PLUS'
        print(f"\n--- [风险探针 V4.4] 正在解剖: 【终极风险信号】{signal_name} ---")

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
        
        # 使用正确的键名 's_bear' 和 'd_bear' 来访问缓存
        if period_to_probe not in overall_health.get('s_bear', {}) or period_to_probe not in overall_health.get('d_bear', {}):
            print(f"  - [探针警告] 在 {domain_upper} 领域的 overall_health 中缺少周期 {period_to_probe} 的数据。")
            return

        s_bear_score = overall_health['s_bear'][period_to_probe].get(probe_date, 0.0)
        d_bear_score = overall_health['d_bear'][period_to_probe].get(probe_date, 0.0)
        
        
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
                if domain_upper == 'BEHAVIOR':
                    atomic_signals_for_behavior = engine_instance._generate_all_atomic_signals(df)
                    if calc_func_name in ['_calculate_price_health', '_calculate_volume_health']:
                        s_bull, d_bull, s_bear, d_bear = calculator(df, 120, 24, {'slope': 0.6, 'accel': 0.4}, periods_arg)
                    else:
                        s_bull, d_bull, s_bear, d_bear = calculator(df, atomic_signals_for_behavior, 120, 24, periods_arg)
                elif domain_upper == 'STRUCTURE':
                    s_bull, d_bull, s_bear, d_bear = calculator(df, periods_arg, 120, {'slope': 0.6, 'accel': 0.4})
                else:
                    s_bull, d_bull, s_bear, d_bear = calculator(df, 120, {'slope': 0.6, 'accel': 0.4}, periods_arg)

                pillar_score_series = s_bear.get(period_to_probe) if bottleneck_type == 's_bear' else d_bear.get(period_to_probe)
                if pillar_score_series is None: continue
                pillar_score = pillar_score_series.get(probe_date, 0.0)
                print(f"       - {pillar_cn_name:<12s} 支柱贡献分: {pillar_score:.4f}")

            except Exception as e:
                print(f"       - [探针错误] 解剖支柱 '{pillar_cn_name}' 失败: {e}")
        
        print(f"  - [最终诊断] {domain_upper} 风险分低，根源在于其构成支柱的【{bottleneck_type}】分数，在现有“相对归一化”逻辑下被历史数据“平均化”，无法体现当日的绝对风险。")

    def _deploy_process_intelligence_probe(self, probe_date: pd.Timestamp):
        """
        【探针V2.1.0 · 心电图检测版】为 ProcessIntelligence 引擎定制的钻透式法医探针。
        - 核心升级: 新增“心电图检测”功能，如果输入信号是恒定值，将直接报告根本原因。
        """
        print("\n--- [探针] 正在解剖: 【过程情报引擎 V2.1.0】 ---") # 更新版本号
        
        df = self.strategy.df_indicators
        engine = self.process_intel
        
        for config in engine.diagnostics_config:
            if config.get('type') != 'meta_analysis':
                continue
            
            signal_name = config.get('name')
            signal_a_name = config.get('signal_A')
            signal_b_name = config.get('signal_B')
            
            print(f"\n  -> 正在解剖元分析任务: 【{signal_name}】 ({signal_a_name} vs {signal_b_name})")
            
            # --- 解剖第一维：瞬时关系分 ---
            print("\n     --- [第一维] 解剖当日的“瞬时关系分”(基于量化力学模型) ---")
            
            momentum_a = self.strategy.atomic_states.get(f"_DEBUG_momentum_{signal_a_name}", pd.Series(np.nan)).get(probe_date, np.nan)
            thrust_b = self.strategy.atomic_states.get(f"_DEBUG_thrust_{signal_b_name}", pd.Series(np.nan)).get(probe_date, np.nan)
            
            # [代码新增] --- 心电图检测 ---
            if pd.isna(momentum_a) or pd.isna(thrust_b):
                # 智能数据源选择逻辑
                def get_signal_series(sig_name: str, source_type: str):
                    if source_type == 'atomic_states':
                        return self.strategy.atomic_states.get(sig_name)
                    return df.get(sig_name)

                series_a = get_signal_series(signal_a_name, config.get('source_A', 'df'))
                series_b = get_signal_series(signal_b_name, config.get('source_B', 'df'))

                if series_a is not None and series_a.nunique() <= 1:
                    print(f"       - [!!! 根本原因定位 !!!] 信号 '{signal_a_name}' 在近期是一个恒定值，其变化量为0，导致动量A无法计算(NaN)。")
                    print(f"       - [诊断建议] 请检查生成 '{signal_a_name}' 的情报引擎，是否存在因数据缺失而返回固定默认值的BUG。")
                    continue # 跳到下一个任务的解剖
                if series_b is not None and series_b.nunique() <= 1:
                    print(f"       - [!!! 根本原因定位 !!!] 信号 '{signal_b_name}' 在近期是一个恒定值，其变化量为0，导致推力B无法计算(NaN)。")
                    print(f"       - [诊断建议] 请检查生成 '{signal_b_name}' 的情报引擎，是否存在因数据缺失而返回固定默认值的BUG。")
                    continue # 跳到下一个任务的解剖
            # [代码新增结束]

            relationship_score_today = self.strategy.atomic_states.get(f"PROCESS_ATOMIC_REL_SCORE_{signal_a_name}_VS_{signal_b_name}", pd.Series(np.nan)).get(probe_date, np.nan)
            
            series_a = df.get(signal_a_name, pd.Series(dtype=float))
            series_b = df.get(signal_b_name, pd.Series(dtype=float))
            change_a = ta.percent_return(series_a, length=1).get(probe_date, 0) if not series_a.empty else 0
            change_b = ta.percent_return(series_b, length=1).get(probe_date, 0) if not series_b.empty else 0
            signal_b_factor_k = config.get('signal_b_factor_k', 1.0)

            print(f"       - 原始变化率: {signal_a_name}({change_a:+.2%}), {signal_b_name}({change_b:+.2%})")
            print(f"       - 双极归一化动量: 动量A({momentum_a:.4f}), 推力B({thrust_b:.4f})") # 更新注释
            print(f"       - 力学公式: 关系分 = 动量A * (1 + {signal_b_factor_k} * 推力B)")
            print(f"       - [代入计算]: {momentum_a:.4f} * (1 + {signal_b_factor_k:.1f} * {thrust_b:.4f}) = {relationship_score_today:.4f}")
            
            # --- 解剖第二维：元分析过程 ---
            print(f"\n     --- [第二维] 解剖“关系分”在 {engine.meta_window} 日窗口内的趋势 ---")
            relationship_series = self.strategy.atomic_states.get(f"PROCESS_ATOMIC_REL_SCORE_{signal_a_name}_VS_{signal_b_name}")
            if relationship_series is None:
                print("       - [探针错误] 无法获取“瞬时关系分”序列。")
                continue

            relationship_trend_series = ta.linreg(relationship_series, length=engine.meta_window)
            relationship_trend_today = relationship_trend_series.get(probe_date, np.nan)
            
            relationship_accel_series = ta.linreg(relationship_trend_series, length=engine.meta_window)
            relationship_accel_today = relationship_accel_series.get(probe_date, np.nan)

            print(f"       - “关系分”的趋势 (斜率): {relationship_trend_today:.4f}")
            print(f"       - “关系分”的加速度: {relationship_accel_today:.4f}")

            # 探针的核心升级：使用双极归一化和加权平均法进行重算
            bipolar_trend_strength = normalize_to_bipolar(
                series=relationship_trend_series,
                target_index=df.index,
                window=engine.norm_window,
                sensitivity=engine.bipolar_sensitivity
            ).get(probe_date, np.nan)
            
            bipolar_accel_strength = normalize_to_bipolar(
                series=relationship_accel_series,
                target_index=df.index,
                window=engine.norm_window,
                sensitivity=engine.bipolar_sensitivity
            ).get(probe_date, np.nan)
            
            trend_weight = engine.meta_score_weights[0]
            accel_weight = engine.meta_score_weights[1]
            recalculated_score = (bipolar_trend_strength * trend_weight + bipolar_accel_strength * accel_weight)
            recalculated_score = np.clip(recalculated_score, -1, 1) # 保持与引擎逻辑一致
            
            print(f"       - 趋势强度分 (双极归一化): {bipolar_trend_strength:.4f}")
            print(f"       - 加速度强度分 (双极归一化): {bipolar_accel_strength:.4f}")
            print(f"       - 融合逻辑: (趋势分 * {trend_weight}) + (加速度分 * {accel_weight})")
            print(f"       - [探针重算结果]: {recalculated_score:.4f}")
            # 结束

            print(f"       - [最终信号实际值]: {self.strategy.atomic_states.get(signal_name, pd.Series(np.nan)).get(probe_date, np.nan):.4f}")

    def _deploy_ultimate_signal_drill_down_probe(self, probe_date: pd.Timestamp, domain: str, signal_type: str):
        """
        【探针V1.5 · 动态分统一版】终极信号钻透式法医探针
        - 核心重构: 彻底重写探针的解剖逻辑，以完全适配“静态分 + 动态强度分 (d_intensity)”的全新统一哲学。
        """
        domain_upper = domain.upper()
        signal_name = f'SCORE_{domain_upper}_{signal_type}'
        print(f"\n--- [钻透式探针] 正在对信号【{signal_name}】在【{probe_date.date()}】进行终极解剖 ---")
        
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        
        params_key_map = {
            'CHIP': 'chip_ultimate_params', 'BEHAVIOR': 'behavioral_dynamics_params', 'FF': 'fund_flow_ultimate_params',
            'STRUCTURE': 'structural_ultimate_params', 'DYN': 'dynamic_mechanics_params', 'FOUNDATION': 'foundation_ultimate_params'
        }
        p_conf = get_params_block(self.strategy, params_key_map.get(domain_upper, ''), {})
        periods = get_param_value(p_conf.get('periods'), [1, 5, 13, 21, 55])
        norm_window = get_param_value(p_conf.get('norm_window'), 120)
        
        final_score = atomic.get(signal_name, pd.Series(0.0, index=df.index)).get(probe_date, 0.0)
        print(f"【顶层】最终信号得分: {final_score:.4f}")

        overall_health_cache_key = f'__{domain_upper}_overall_health'
        overall_health = atomic.get(overall_health_cache_key)
        if not overall_health:
            print(f"  - [探针错误] 致命错误: 未能在 atomic_states 中找到缓存 '{overall_health_cache_key}'。解剖终止。")
            return

        print("\n  [链路层 1] 反推 -> 短/中/长 三股力量")
        
        health_components = {}
        # [代码修改] 更新健康度计算逻辑，统一使用 d_intensity
        s_type = 's_bull' if signal_type == 'BULLISH_RESONANCE' else 's_bear'
        if signal_type in ['BULLISH_RESONANCE', 'BEARISH_RESONANCE']:
            health_components = {p: overall_health[s_type].get(p, pd.Series(0.5)) * overall_health['d_intensity'].get(p, pd.Series(0.5)) for p in periods}
        else:
            print(f"  - [探针警告] 未知的信号类型 '{signal_type}'，无法继续解剖。")
            return
            
        default_series = pd.Series(0.5, index=df.index)
        short_force = (health_components.get(1, default_series).get(probe_date, 0.5) * health_components.get(5, default_series).get(probe_date, 0.5))**0.5
        medium_force = (health_components.get(13, default_series).get(probe_date, 0.5) * health_components.get(21, default_series).get(probe_date, 0.5))**0.5
        long_force = health_components.get(55, default_series).get(probe_date, 0.5)
        print(f"    - 短期力: {short_force:.4f}")
        print(f"    - 中期力: {medium_force:.4f}")
        print(f"    - 长期力: {long_force:.4f}")

        period_to_probe = 1
        print(f"\n  [链路层 2] 反推 -> {period_to_probe}日健康度")
        health_score = health_components.get(period_to_probe, default_series).get(probe_date, 0.5)
        
        # [代码修改] 更新解剖逻辑，展示 s_type 和 d_intensity
        s_score = overall_health[s_type][period_to_probe].get(probe_date, 0.5)
        d_intensity_score = overall_health['d_intensity'][period_to_probe].get(probe_date, 0.5)
        print(f"    - {period_to_probe}日健康度 ({health_score:.4f}) = {s_type} ({s_score:.4f}) * d_intensity ({d_intensity_score:.4f})")
        
        print(f"\n  [链路层 3] 反推 -> 构成 {s_type} 和 d_intensity 的各个支柱分数")
        
        engine_map = {
            'CHIP': self.chip_intel, 'BEHAVIOR': self.behavioral_intel, 'FF': self.fund_flow_intel,
            'STRUCTURE': self.structural_intel, 'DYN': self.mechanics_engine, 'FOUNDATION': self.foundation_intel
        }
        calc_map = {
            'CHIP': [('_calculate_quantitative_health', 'quantitative'), ('_calculate_advanced_dynamics_health', 'advanced'), ('_calculate_internal_structure_health', 'internal'), ('_calculate_holder_behavior_health', 'holder'), ('_calculate_fault_health', 'fault')],
            'BEHAVIOR': [('_calculate_price_health', 'price'), ('_calculate_volume_health', 'volume'), ('_calculate_kline_pattern_health', 'kline')],
            'DYN': [('_calculate_volatility_health', 'volatility'), ('_calculate_efficiency_health', 'efficiency'), ('_calculate_kinetic_energy_health', 'momentum'), ('_calculate_inertia_health', 'inertia')],
            'STRUCTURE': [('_calculate_ma_health', 'ma'), ('_calculate_mechanics_health', 'mechanics'), ('_calculate_mtf_health', 'mtf'), ('_calculate_pattern_health', 'pattern')],
            'FOUNDATION': [('_calculate_ema_health', 'ema'), ('_calculate_rsi_health', 'rsi'), ('_calculate_macd_health', 'macd'), ('_calculate_cmf_health', 'cmf')]
        }
        
        engine_instance = engine_map.get(domain_upper)
        pillar_calculators = calc_map.get(domain_upper, [])

        if not engine_instance:
            print(f"  - [探针错误] 未找到领域 '{domain_upper}' 的引擎实例。")
            return

        print(f"    --- 解剖 {s_type} ({s_score:.4f}) ---")
        for calc_func_name, pillar_name in pillar_calculators:
            try:
                calculator = getattr(engine_instance, calc_func_name)
                
                # [代码修改] 更新所有 calculator 的调用和解包逻辑
                if domain_upper == 'BEHAVIOR':
                    atomic_signals_for_behavior = engine_instance._generate_all_atomic_signals(df)
                    min_periods = max(1, norm_window // 5)
                    if calc_func_name == '_calculate_kline_pattern_health':
                        s_bull_pillar, s_bear_pillar, _ = calculator(df, atomic_signals_for_behavior, norm_window, min_periods, [period_to_probe])
                    else:
                        s_bull_pillar, s_bear_pillar, _ = calculator(df, norm_window, min_periods, [period_to_probe])
                elif domain_upper == 'STRUCTURE':
                    s_bull_pillar, s_bear_pillar, _ = calculator(df, [period_to_probe], norm_window, {})
                else: # CHIP, DYN, FOUNDATION
                    s_bull_pillar, s_bear_pillar, _ = calculator(df, norm_window, {}, [period_to_probe])

                pillar_score_series = s_bull_pillar.get(period_to_probe) if s_type == 's_bull' else s_bear_pillar.get(period_to_probe)
                if pillar_score_series is None:
                    print(f"      - {pillar_name} 支柱贡献分: [计算失败，未返回Series]")
                    continue
                
                pillar_s_score = pillar_score_series.get(probe_date, 0.5)
                print(f"      - {pillar_name} 支柱贡献分: {pillar_s_score:.4f}")
                
                # [代码修改] 移除旧的、错误的钻透逻辑
                # if pillar_s_score < 0.2 and domain_upper == 'BEHAVIOR' and pillar_name == 'price':
                #     ... (旧逻辑)

            except Exception as e:
                print(f"       - [探针错误] 解剖支柱 '{pillar_name}' 的 {s_type} 失败: {e}")

        print(f"--- 信号【{signal_name}】解剖完毕 ---")











