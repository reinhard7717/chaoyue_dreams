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
        # self._deploy_ultimate_signal_drill_down_probe(probe_date, domain='CHIP', signal_type='BULLISH_RESONANCE')
        # self._deploy_ultimate_signal_drill_down_probe(probe_date, domain='BEHAVIOR', signal_type='BULLISH_RESONANCE')
        # 您也可以解剖看跌信号
        # self._deploy_ultimate_signal_drill_down_probe(probe_date, domain='DYN', signal_type='BEARISH_RESONANCE')
        # self._deploy_process_intelligence_probe(probe_date)
        print("="*95 + "\n")

    def _deploy_process_intelligence_probe(self, probe_date: pd.Timestamp):
        """
        【探针V2.2.0 · 战略协同解剖版】为 ProcessIntelligence 引擎定制的钻透式法医探针。
        - 核心升级: 新增对 'strategy_sync' 任务类型的解剖能力，使其能正确展示高阶战略信号的分析过程。
        """
        print("\n--- [探针] 正在解剖: 【过程情报引擎 V2.2.0】 ---")
        
        df = self.strategy.df_indicators
        engine = self.process_intel
        
        for config in engine.diagnostics_config:
            signal_name = config.get('name')
            signal_type = config.get('type')
            signal_a_name = config.get('signal_A')
            signal_b_name = config.get('signal_B')
            
            print(f"\n  -> 正在解剖元分析任务: 【{signal_name}】 ({signal_a_name} vs {signal_b_name})")
            
            # 增加对 strategy_sync 任务类型的专属解剖逻辑
            if signal_type == 'strategy_sync':
                print("\n     --- [第一维] 解剖当日的“瞬时关系分”(基于战略信号映射) ---")
                momentum_a = self.strategy.atomic_states.get(f"_DEBUG_momentum_{signal_a_name}", pd.Series(np.nan)).get(probe_date, np.nan)
                thrust_b = self.strategy.atomic_states.get(f"_DEBUG_thrust_{signal_b_name}", pd.Series(np.nan)).get(probe_date, np.nan)
                
                signal_a_val = self.strategy.atomic_states.get(signal_a_name, pd.Series(np.nan)).get(probe_date, np.nan)
                signal_b_val = self.strategy.atomic_states.get(signal_b_name, pd.Series(np.nan)).get(probe_date, np.nan)

                print(f"       - 原始信号值: {signal_a_name}({signal_a_val:.4f}), {signal_b_name}({signal_b_val:.4f})")
                print(f"       - 映射后动量: 动量A({momentum_a:.4f}), 推力B({thrust_b:.4f})")
            else: # 保持对 meta_analysis 的原有解剖逻辑
                print("\n     --- [第一维] 解剖当日的“瞬时关系分”(基于量化力学模型) ---")
                momentum_a = self.strategy.atomic_states.get(f"_DEBUG_momentum_{signal_a_name}", pd.Series(np.nan)).get(probe_date, np.nan)
                thrust_b = self.strategy.atomic_states.get(f"_DEBUG_thrust_{signal_b_name}", pd.Series(np.nan)).get(probe_date, np.nan)
                
                series_a = df.get(signal_a_name, pd.Series(dtype=float))
                series_b = df.get(signal_b_name, pd.Series(dtype=float))
                change_a = ta.percent_return(series_a, length=1).get(probe_date, 0) if not series_a.empty else 0
                change_b = ta.percent_return(series_b, length=1).get(probe_date, 0) if not series_b.empty else 0
                print(f"       - 原始变化率: {signal_a_name}({change_a:+.2%}), {signal_b_name}({change_b:+.2%})")
                print(f"       - 双极归一化动量: 动量A({momentum_a:.4f}), 推力B({thrust_b:.4f})")

            relationship_score_today = self.strategy.atomic_states.get(f"PROCESS_ATOMIC_REL_SCORE_{signal_a_name}_VS_{signal_b_name}", pd.Series(np.nan)).get(probe_date, np.nan)
            signal_b_factor_k = config.get('signal_b_factor_k', 1.0)
            print(f"       - 力学公式: 关系分 = 动量A * (1 + {signal_b_factor_k} * 推力B)")
            print(f"       - [代入计算]: {momentum_a:.4f} * (1 + {signal_b_factor_k:.1f} * {thrust_b:.4f}) = {relationship_score_today:.4f}")
            
            # --- 第二维解剖逻辑保持不变，因为对所有类型都适用 ---
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

            bipolar_trend_strength = normalize_to_bipolar(relationship_trend_series, df.index, engine.norm_window, engine.bipolar_sensitivity).get(probe_date, np.nan)
            bipolar_accel_strength = normalize_to_bipolar(relationship_accel_series, df.index, engine.norm_window, engine.bipolar_sensitivity).get(probe_date, np.nan)
            
            trend_weight = engine.meta_score_weights[0]
            accel_weight = engine.meta_score_weights[1]
            recalculated_score = np.clip((bipolar_trend_strength * trend_weight + bipolar_accel_strength * accel_weight), -1, 1)
            
            print(f"       - 趋势强度分 (双极归一化): {bipolar_trend_strength:.4f}")
            print(f"       - 加速度强度分 (双极归一化): {bipolar_accel_strength:.4f}")
            print(f"       - 融合逻辑: (趋势分 * {trend_weight}) + (加速度分 * {accel_weight})")
            print(f"       - [探针重算结果]: {recalculated_score:.4f}")
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
        # 更新健康度计算逻辑，统一使用 d_intensity
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
        
        # 更新解剖逻辑，展示 s_type 和 d_intensity
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
                
                # 更新所有 calculator 的调用和解包逻辑
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
                
                # 移除旧的、错误的钻透逻辑
                # if pillar_s_score < 0.2 and domain_upper == 'BEHAVIOR' and pillar_name == 'price':
                #     ... (旧逻辑)

            except Exception as e:
                print(f"       - [探针错误] 解剖支柱 '{pillar_name}' 的 {s_type} 失败: {e}")

        print(f"--- 信号【{signal_name}】解剖完毕 ---")











