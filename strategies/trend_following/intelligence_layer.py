# 文件: strategies/trend_following/intelligence_layer.py
# 情报层总指挥官 (重构版)
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict
from .structural_defense_layer import StructuralDefenseLayer
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
        self.structural_defense_layer = StructuralDefenseLayer(self.strategy)

    def run_all_diagnostics(self) -> Dict:
        """
        【V413.0 · 涡轮增压版】情报层总指挥官
        - 核心升级: 在认知层融合的最后阶段，调用新的“涡轮增压”引擎，诊断趋势的级联加速效应。
        """
        print("--- [情报层总指挥官 V413.0 · 涡轮增压版] 开始执行所有诊断模块... ---") # 更新版本号
        df = self.strategy.df_indicators
        self.strategy.atomic_states = {}
        self.strategy.trigger_events = {}
        self.strategy.playbook_states = {}
        self.strategy.exit_triggers = pd.DataFrame(index=df.index)

        def update_states(new_states: Dict):
            if isinstance(new_states, dict):
                self.strategy.atomic_states.update(new_states)

        # --- 阶段一: 基础信号生成 (按依赖关系重构顺序) ---
        print("    - [阶段 1/5] 正在执行周期与基础过程诊断...")
        update_states(self.cyclical_intel.run_cyclical_analysis_command(df))
        base_process_states = self.process_intel.run_process_diagnostics(task_type_filter='base')
        update_states(base_process_states)

        # --- 阶段二: 状态情报与战略过程诊断 ---
        print("    - [阶段 2/5] 正在执行状态情报与战略过程诊断...")
        update_states(self.behavioral_intel.run_behavioral_analysis_command())
        update_states(self.foundation_intel.run_foundation_analysis_command())
        update_states(self.chip_intel.run_chip_intelligence_command(df))
        update_states(self.structural_intel.diagnose_structural_states(df))
        update_states(self.fund_flow_intel.diagnose_fund_flow_states(df))
        update_states(self.mechanics_engine.run_dynamic_analysis_command())
        update_states(self.pattern_intel.run_pattern_analysis_command(df))
        strategy_process_states = self.process_intel.run_process_diagnostics(task_type_filter='strategy')
        update_states(strategy_process_states)
        
        # --- 阶段三: 跨域认知融合 ---
        print("    - [阶段 3/5] 正在执行认知层跨域元融合...")
        self.cognitive_intel.synthesize_cognitive_scores(df, pullback_enhancements={})

        # --- 阶段四: 最终战法与剧本生成 ---
        print("    - [阶段 4/5] 正在生成最终战法与剧本...")
        trigger_events = self.playbook_engine.define_trigger_events(df)
        self.strategy.trigger_events.update(trigger_events)
        _, playbook_states = self.playbook_engine.generate_playbook_states(self.strategy.trigger_events)
        self.strategy.playbook_states.update(playbook_states)
        
        # --- 阶段五: 硬性离场信号生成 ---
        print("    - [阶段 5/5] 正在生成硬性离场信号...")
        exit_triggers_df = self.structural_defense_layer.generate_hard_exit_triggers()
        self.strategy.exit_triggers = exit_triggers_df

        self.deploy_forensic_probes()
        
        print("--- [情报层总指挥官 V413.0] 所有诊断模块执行完毕。 ---")
        return self.strategy.trigger_events

    def deploy_nan_forensics_probe(self, nan_date, nan_signal_name: str):
        """
        【V1.1 · 探针修复版】NaN 值法医探针。
        - 核心修复: 修复了探针在尝试获取 `periods` 列表时因脆弱的内省逻辑而崩溃的问题。
                      现在探针采用与主逻辑相同的、健壮的方式从参数块中读取配置。
        """
        print("\n" + "="*30 + f" [NaN 法医探针 V1.1 启动] " + "="*30)
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
            print(f"  ---> 解剖 {health_type} 健康度 (周期 {period})...")
            if "DYN" in nan_signal_name:
                raw_slope = df.get(f'SLOPE_{period}_BBW_21_2.0_D', pd.Series(np.nan)).get(date)
                raw_accel = df.get(f'ACCEL_{period}_BBW_21_2.0_D', pd.Series(np.nan)).get(date)
                print(f"      ----> 原始指标 SLOPE_{period}_BBW_21_2.0_D: {raw_slope}")
                print(f"      ----> 原始指标 ACCEL_{period}_BBW_21_2.0_D: {raw_accel}")
                if pd.isna(raw_slope) or pd.isna(raw_accel):
                    print("      ------> [!!!] 发现源头 NaN！问题可能出在基础指标计算层。")

        if "CHIP" in nan_signal_name or "DYN" in nan_signal_name or "STRUCTURE" in nan_signal_name or "FOUNDATION" in nan_signal_name:
            print("  --> 检测到终极信号，开始解剖...")
            print("  --> 正在检查所有支柱的健康度贡献...")
            
            # 废除脆弱的__defaults__内省方式
            # 采用与主逻辑相同的、健壮的方式从参数块获取 periods
            p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
            periods_to_probe = get_param_value(p_conf.get('periods', [1, 5, 13, 21, 55]))

            # 使用从参数中安全获取的 periods_to_probe
            for p in periods_to_probe:
                for ht in ['bullish_static', 'bullish_dynamic', 'bearish_static', 'bearish_dynamic']:
                    pass
            print("  --> 提示: 请检查相关情报引擎中各 _calculate_..._health 方法的 normalize_score 输入是否存在NaN。")

        elif "PLAYBOOK" in nan_signal_name or "COGNITIVE" in nan_signal_name or "TACTIC" in nan_signal_name:
            print(f"  --> 检测到认知/战术层信号，开始解剖 {nan_signal_name}...")
            
            if nan_signal_name == 'SCORE_PLAYBOOK_MEAN_REVERSION_GRID_BUY_A':
                print("  ---> 解剖路径: final_score = context * opportunity")
                
                print("  -----> 正在解剖 context_is_ranging_market...")
                is_cyclical_regime = get_val('SCORE_CYCLICAL_REGIME', nan_date) > 0.4
                is_not_trending_regime = get_val('SCORE_TRENDING_REGIME_FFT', nan_date) < 0.45
                context_val = float(is_cyclical_regime and is_not_trending_regime)
                print(f"      - context_is_ranging_market = {context_val}")

                print("  -----> 正在解剖 buy_opportunity_score...")
                bbp_val = df.get('BBP_21_2.0_D', pd.Series(np.nan)).get(nan_date)
                print(f"      - 原始指标 BBP_21_2.0_D: {bbp_val}")
                if pd.isna(bbp_val):
                    print("      ------> [!!!] 发现源头 NaN！问题出在 BBP_21_2.0_D 指标计算。")
                opportunity_val = 1 - np.clip(bbp_val, 0, 1) if pd.notna(bbp_val) else np.nan
                print(f"      - buy_opportunity_score = {opportunity_val}")

                final_val = context_val * opportunity_val
                print(f"  ---> 最终验算: {context_val} * {opportunity_val} = {final_val}")

        else:
            print("  --> 未找到特定引擎的解剖路径，执行通用检查...")
            print("  --> 正在检查该信号在 atomic_states 中的值...")
            get_val(nan_signal_name, nan_date)

        print("-" * 80)
        print(f"  - 解剖完毕。请重点关注报告中值为 'NaN' 的步骤，其上一步即为问题源头。")
        print("=" * 80 + "\n")

    def deploy_forensic_probes(self):
        """
        【V1.7 · 多目标打击版】法医探针调度中心
        - 核心升级: 1. 配置文件中的 'probe_date' 升级为 'probe_dates'，可以接收一个日期列表。
                      2. 探针调度中心现在会遍历列表中的所有日期，并对每个日期执行全套探针解剖。
        - 收益: 极大提升了战役复盘和问题诊断的效率。
        """
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        if not debug_params.get('enabled', False):
            return

        # 从 "probe_date" 升级为 "probe_dates"，并处理列表
        probe_dates_list = debug_params.get('probe_dates')
        
        # 增加向后兼容逻辑，如果只找到旧的 probe_date，则将其转为列表
        if not probe_dates_list:
            single_date = debug_params.get('probe_date')
            if single_date:
                probe_dates_list = [single_date]

        if not probe_dates_list or not isinstance(probe_dates_list, list):
            return
            
        print("\n" + "="*30 + f" [法医探针部署中心 V1.7] 开始对 {len(probe_dates_list)} 个目标日期进行解剖... " + "="*30)

        # 遍历所有需要探查的日期
        for probe_date_str in probe_dates_list:
            if not probe_date_str:
                continue

            probe_date = pd.to_datetime(probe_date_str)
            if self.strategy.df_indicators.index.tz is not None:
                try:
                    probe_date = probe_date.tz_localize(self.strategy.df_indicators.index.tz)
                except Exception:
                    try:
                        probe_date = probe_date.tz_convert(self.strategy.df_indicators.index.tz)
                    except Exception as e_conv:
                         print(f"    -> [法医探针] 错误: 转换探针日期 {probe_date_str} 时区失败: {e_conv}。")
                         continue # 跳过此日期，继续下一个
            
            if probe_date not in self.strategy.df_indicators.index:
                print(f"    -> [法医探针] 警告: 探针日期 {probe_date_str} (校准后: {probe_date}) 不在数据索引中，跳过该日期。")
                continue # 跳过此日期，继续下一个

            # 将打印信息移入循环内，指明当前正在解剖的日期
            print("\n" + "="*25 + f" 正在解剖 {probe_date_str} " + "="*25)
            
            self._deploy_genesis_probe(probe_date)
            self._deploy_turbo_probe(probe_date)
            self._deploy_judgment_day_probe(probe_date)
        
        print("\n" + "="*35 + " [法医探针部署中心] 所有目标解剖完毕 " + "="*35 + "\n")

    def _deploy_judgment_day_probe(self, probe_date: pd.Timestamp):
        """
        【V2.3 · 刻耳柏洛斯之镜版】审判日引擎法医探针
        - 核心革命: 修复了“镜像悖论”。探针的重算逻辑从旧的 max() 升级为与主系统
                      完全同步的“地狱三头犬”主次风险融合算法，确保了验证的准确性。
        """
        print("\n--- [探针] 正在解剖: 【创世纪 VIII · 审判日引擎】 ---")
        atomic = self.strategy.atomic_states
        df = self.strategy.df_indicators
        
        try:
            alert_level_series, alert_reason_series, fused_risks_df = self.strategy.judgment_layer._adjudicate_risk_level()
        except Exception as e:
            print(f"  [错误] 在探针内部调用 _adjudicate_risk_level 时发生异常: {e}。解剖终止。")
            return

        if probe_date not in alert_level_series.index:
            print(f"  [错误] 探针日期 {probe_date} 不在独立计算的风险结果索引中。解剖终止。")
            return
            
        alert_level = alert_level_series.get(probe_date)
        alert_reason = alert_reason_series.get(probe_date)
        
        print(f"\n  [链路层 1] 最终裁决 -> ALERT_LEVEL: {alert_level} ({alert_reason or '无警报'})")

        if probe_date not in fused_risks_df.index:
            print("  [错误] 探针日期不在风险融合数据中。解剖终止。")
            return
            
        print("\n  [链路层 2] 解剖 -> 各审判庭风险强度 (取组内最大值)")
        probe_risk_values = fused_risks_df.loc[probe_date]
        for category, value in probe_risk_values.items():
            print(f"    - {category:<20}: {value:.4f}")

        print("\n  [链路层 2.1] 解剖 -> “天使长”审判庭专项诊断")
        archangel_components = {
            "上冲派发 (Upthrust)": "SCORE_RISK_UPTHRUST_DISTRIBUTION",
            "天地板 (Heaven-Earth)": "SCORE_BOARD_HEAVEN_EARTH",
            "高位回落 (Post-Peak)": "COGNITIVE_SCORE_RISK_POST_PEAK_DOWNTURN"
        }
        component_scores = {}
        for name, signal in archangel_components.items():
            score = atomic.get(signal, pd.Series(0.0, index=df.index)).get(probe_date, 0.0)
            component_scores[name] = score
            print(f"    - {name:<25}: {score:.4f}")
        
        # [代码修改] 升级探针的重算逻辑，与“地狱三头犬”算法完全同步
        p_judge = get_params_block(self.strategy, 'judgment_params', {})
        p_archangel = p_judge.get('archangel_fusion_params', {})
        secondary_risk_discount = get_param_value(p_archangel.get('secondary_risk_discount'), 0.4)

        sorted_scores = sorted(component_scores.values(), reverse=True)
        primary_risk = sorted_scores[0] if sorted_scores else 0.0
        secondary_risk = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        
        recalculated_archangel_score = np.clip(primary_risk + (secondary_risk * secondary_risk_discount), 0, 1)
        actual_archangel_score = probe_risk_values.get('ARCHANGEL_RISK', 0.0)
        
        print(f"    - [探针重算天使长风险]: {primary_risk:.4f} (主) + ({secondary_risk:.4f} (次) * {secondary_risk_discount}) = {recalculated_archangel_score:.4f}")
        print(f"    - [对比]: 实际值 {actual_archangel_score:.4f} vs 重算值 {recalculated_archangel_score:.4f}")

        print("\n  [链路层 3] 验证 -> 警报等级裁决逻辑")
        p_alerts = p_judge.get('alert_level_thresholds', {})
        
        level_3_archangel_threshold = get_param_value(p_alerts.get('level_3_archangel_threshold'), 0.7)
        level_3_threshold = get_param_value(p_alerts.get('level_3_top_reversal'), 0.8)
        level_2_resonance_threshold = get_param_value(p_alerts.get('level_2_bearish_resonance'), 0.7)
        level_2_euphoria_threshold = get_param_value(p_alerts.get('level_2_euphoria_risk'), 0.75)
        level_1_threshold = get_param_value(p_alerts.get('level_1_micro_risk'), 0.6)
        
        print(f"    - Level 3 (天使长) 阈值: > {level_3_archangel_threshold}")
        print(f"    - Level 3 (顶部反转) 阈值: > {level_3_threshold}")
        print(f"    - Level 2 (共振或亢奋) 阈值: > {level_2_resonance_threshold} 或 > {level_2_euphoria_threshold}")
        print(f"    - Level 1 (微观风险) 阈值: > {level_1_threshold}")

        print("\n  [链路层 4] 最终验证")
        recalculated_level = 0
        if probe_risk_values.get('ARCHANGEL_RISK', 0) > level_3_archangel_threshold:
            recalculated_level = 3
        elif probe_risk_values.get('TOP_REVERSAL', 0) > level_3_threshold:
            recalculated_level = 3
        elif (probe_risk_values.get('BEARISH_RESONANCE', 0) > level_2_resonance_threshold) or \
             (probe_risk_values.get('EUPHORIA_RISK', 0) > level_2_euphoria_threshold):
            recalculated_level = 2
        elif probe_risk_values.get('MICRO_RISK', 0) > level_1_threshold:
            recalculated_level = 1
            
        print(f"    - [探针重算]: {recalculated_level}")
        print(f"    - [对比]: 实际值 {alert_level} vs 重算值 {recalculated_level}")
        print("--- 审判日探针解剖完毕 ---")

    def _deploy_turbo_probe(self, probe_date: pd.Timestamp):
        """
        【V2.0 · 指挥家版】“涡轮增压”法医探针
        - 核心升级: 能够清晰地解剖“先锋部队”与“主力部队”的加速分，并展示“协同奖励”的计算过程。
        """
        print("\n--- [探针] 正在解剖: 【创世纪 VII · 指挥家引擎】 ---")
        atomic = self.strategy.atomic_states
        df = self.strategy.df_indicators
        
        def get_val(name, date, default=np.nan):
            first_key = next(iter(atomic), None)
            if first_key is None: return default
            return atomic.get(name, pd.Series(default, index=atomic.get(first_key).index)).get(date, default)

        # --- 步骤 1: 获取最终“涡轮增压”得分 (保持不变) ---
        final_cascade_score = get_val('COGNITIVE_SCORE_TREND_ACCELERATION_CASCADE', probe_date, 0.0)
        print(f"\n  [链路层 1] 解剖 -> 最终得分 (COGNITIVE_SCORE_TREND_ACCELERATION_CASCADE)")
        print(f"    - 【最终融合值】: {final_cascade_score:.4f}")
        print(f"    - [核心公式]: (时序级联分 * 领域级联分)")

        # --- 步骤 2: 解剖“时序级联”分 (保持不变) ---
        temporal_cascade_score = get_val('COGNITIVE_INTERNAL_TEMPORAL_CASCADE', probe_date, 0.0)
        print(f"\n  [链路层 2] 解剖 -> 时序级联分 (Temporal Cascade)")
        print(f"    - 【得分】: {temporal_cascade_score:.4f}")
        # ... (此处省略时序级联的详细解剖，因为它未改变)

        # --- 步骤 3: 解剖“领域级联”分 (全新改造) ---
        domain_cascade_score = get_val('COGNITIVE_INTERNAL_DOMAIN_CASCADE', probe_date, 0.0)
        print(f"\n  [链路层 3] 解剖 -> 领域级联分 (Domain Cascade)")
        print(f"    - 【得分】: {domain_cascade_score:.4f}")
        print(f"    - [核心公式]: vanguard_score * (1 + confirmation_score)")
        
        # 分兵种展示
        vanguard_domains = ['behavior', 'dyn']
        confirmation_domains = ['chip', 'ff', 'structure']
        
        print("\n      --- 先锋部队 (Vanguard) ---")
        vanguard_scores = []
        for domain in vanguard_domains:
            score = get_val(f'COGNITIVE_INTERNAL_ACCEL_{domain.upper()}', probe_date, 0.0)
            vanguard_scores.append(score)
            print(f"        - {domain.capitalize()} 领域加速分: {score:.4f}")
        recalc_vanguard_score = np.linalg.norm(vanguard_scores, ord=2) / np.sqrt(len(vanguard_scores)) if vanguard_scores else 0.0
        print(f"        - [探针重算先锋总分]: {recalc_vanguard_score:.4f}")

        print("\n      --- 主力部队 (Confirmation) ---")
        confirmation_scores = []
        for domain in confirmation_domains:
            score = get_val(f'COGNITIVE_INTERNAL_ACCEL_{domain.upper()}', probe_date, 0.0)
            confirmation_scores.append(score)
            print(f"        - {domain.capitalize()} 领域加速分: {score:.4f}")
        recalc_confirmation_score = np.linalg.norm(confirmation_scores, ord=2) / np.sqrt(len(confirmation_scores)) if confirmation_scores else 0.0
        print(f"        - [探针重算主力总分]: {recalc_confirmation_score:.4f}")

        # --- 步骤 4: 最终验证 (全新改造) ---
        print("\n  [链路层 4] 最终验证")
        recalc_final_domain_score = (recalc_vanguard_score * (1 + recalc_confirmation_score)).clip(0, 1)
        print(f"    - [探针重算领域级联分]: {recalc_vanguard_score:.4f} * (1 + {recalc_confirmation_score:.4f}) = {recalc_final_domain_score:.4f}")
        print(f"    - [对比]: 实际值 {domain_cascade_score:.4f} vs 重算值 {recalc_final_domain_score:.4f}")
        print("--- 指挥家探针解剖完毕 ---")

    def _deploy_genesis_probe(self, probe_date: pd.Timestamp):
        """
        【V2.0 · 双核解剖版】“创世纪”法医探针
        - 核心升级: 能够清晰地解剖“风暴降生”与“静水流深”双核动力源，并展示“强者为王”的最终裁定过程。
        """
        print("\n--- [探针] 正在解剖: 【创世纪 III · 双核驱动引擎】 ---")
        atomic = self.strategy.atomic_states
        
        def get_val(name, date, default=np.nan):
            # 确保即使atomic字典为空也能安全运行
            first_key = next(iter(atomic), None)
            if first_key is None:
                return default
            return atomic.get(name, pd.Series(default, index=atomic.get(first_key).index)).get(date, default)

        # --- 步骤 1: 解剖关系动力分的构成 ---
        print("\n  [链路层 1] 解剖 -> 关系动力分 (SCORE_ATOMIC_RELATIONAL_DYNAMICS)")
        
        # 分别获取两个核心动力源和最终裁定值
        stormborn_power = get_val('SCORE_ATOMIC_STORM_BORN_POWER', probe_date, 0.0)
        still_waters_power = get_val('SCORE_ATOMIC_STILL_WATERS_POWER', probe_date, 0.0)
        relational_power = get_val('SCORE_ATOMIC_RELATIONAL_DYNAMICS', probe_date, 0.0)
        
        print(f"    - 【最终裁定值】: {relational_power:.4f}  <-- (取双核中更强者)")
        
        # 解剖“风暴降生”核心
        print(f"\n    --- 双核之一: “风暴降生”原型 (V反) ---")
        print(f"      - 得分: {stormborn_power:.4f}")
        power_transfer_raw = get_val('PROCESS_META_POWER_TRANSFER', probe_date, 0.0)
        power_transfer_map = np.clip(power_transfer_raw, -1, 1) * 0.5 + 0.5
        loser_capitulation_raw = get_val('PROCESS_META_LOSER_CAPITULATION', probe_date, 0.0)
        loser_capitulation_map = np.clip(loser_capitulation_raw, -1, 1) * 0.5 + 0.5
        print(f"        - 权力转移: raw={power_transfer_raw:.2f} -> mapped={power_transfer_map:.2f}")
        print(f"        - 投降仪式: raw={loser_capitulation_raw:.2f} -> mapped={loser_capitulation_map:.2f}")
        
        # 解剖“静水流深”核心
        print(f"\n    --- 双核之二: “静水流深”原型 (盘整) ---")
        print(f"      - 得分: {still_waters_power:.4f}")
        stealth_accumulation_raw = get_val('PROCESS_META_STEALTH_ACCUMULATION', probe_date, 0.0)
        stealth_accumulation_map = np.clip(stealth_accumulation_raw, -1, 1) * 0.5 + 0.5
        winner_conviction_raw = get_val('PROCESS_META_WINNER_CONVICTION', probe_date, 0.0)
        winner_conviction_map = np.clip(winner_conviction_raw, -1, 1) * 0.5 + 0.5
        print(f"        - 隐秘吸筹: raw={stealth_accumulation_raw:.2f} -> mapped={stealth_accumulation_map:.2f}")
        print(f"        - 赢家信念: raw={winner_conviction_raw:.2f} -> mapped={winner_conviction_map:.2f}")

        # --- 步骤 2: 解剖一个典型的终极信号，看其如何被赋能 ---
        print("\n  [链路层 2] 解剖 -> 典型终极信号 (以 SCORE_BEHAVIOR_BULLISH_RESONANCE 为例)")
        behavior_resonance = get_val('SCORE_BEHAVIOR_BULLISH_RESONANCE', probe_date, 0.0)
        print(f"    - 【最终信号值】: {behavior_resonance:.4f}")
        
        overall_health = atomic.get('__BEHAVIOR_overall_health', {})
        if not overall_health:
            print("    - [探针错误] 无法找到 __BEHAVIOR_overall_health 缓存。")
            return
            
        s_bull_5 = overall_health.get('s_bull', {}).get(5, pd.Series(0.5)).get(probe_date, 0.5)
        d_intensity_5 = overall_health.get('d_intensity', {}).get(5, pd.Series(0.5)).get(probe_date, 0.5)
        
        print("    - [核心公式]: health = np.maximum(s_bull, relational_power) * d_intensity")
        recalc_health_5 = np.maximum(s_bull_5, relational_power) * d_intensity_5
        print(f"      - 5日周期健康度: {recalc_health_5:.4f} = max({s_bull_5:.4f}, {relational_power:.4f}) * {d_intensity_5:.4f}")
        print(f"        - s_bull (静态分): {s_bull_5:.4f}")
        print(f"        - relational_power (关系动力): {relational_power:.4f}  <-- 【权柄交接发生处】")
        print(f"        - d_intensity (动态分): {d_intensity_5:.4f}")

        # --- 步骤 3: 进一步解剖 s_bull 的构成 ---
        print("\n  [链路层 3] 解剖 -> 行为层 s_bull 的构成")
        behavior_engine = self.behavioral_intel
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        pillar_weights = get_param_value(p_conf.get('pillar_weights'), {})
        
        s_bull_pillars = {}
        try:
            price_s_bull, _, _ = behavior_engine._calculate_price_health(self.strategy.df_indicators, 55, 11, [5])
            s_bull_pillars['price'] = price_s_bull[5].get(probe_date, 0.5)
            
            vol_s_bull, _, _ = behavior_engine._calculate_volume_health(self.strategy.df_indicators, 55, 11, [5])
            s_bull_pillars['volume'] = vol_s_bull[5].get(probe_date, 0.5)
            
            kline_s_bull, _, _ = behavior_engine._calculate_kline_pattern_health(self.strategy.df_indicators, atomic, 55, 11, [5])
            s_bull_pillars['kline'] = kline_s_bull[5].get(probe_date, 0.5)
            
            print(f"    - 融合公式: (price^{pillar_weights.get('price',0)}) * (volume^{pillar_weights.get('volume',0)}) * (kline^{pillar_weights.get('kline',0)})")
            for name, val in s_bull_pillars.items():
                print(f"      - {name} 支柱静态分: {val:.4f}")
            
            recalc_s_bull = (s_bull_pillars['price']**pillar_weights.get('price',0) * 
                             s_bull_pillars['volume']**pillar_weights.get('volume',0) * 
                             s_bull_pillars['kline']**pillar_weights.get('kline',0))
            print(f"    - [探针重算 s_bull]: {recalc_s_bull:.4f} (实际值: {s_bull_5:.4f})")

        except Exception as e:
            print(f"    - [探针错误] 无法解剖 s_bull 构成: {e}")
            
        print("--- 创世纪探针解剖完毕 ---")

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











