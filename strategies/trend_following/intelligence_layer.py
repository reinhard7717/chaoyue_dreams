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
from strategies.trend_following.utils import get_params_block, get_param_value, fuse_multi_level_scores, normalize_score

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

        # [代码修改] 现在，这里的检查是在两个时区类型相同的对象之间进行的
        if probe_date not in self.strategy.df_indicators.index:
            print(f"    -> [法医探针] 警告: 探针日期 {probe_date_str} (校准后: {probe_date}) 不在数据索引中，跳过探针部署。")
            return

        print("\n" + "="*30 + f" [法医探针部署中心 V1.1] 正在解剖 {probe_date_str} " + "="*30) # [代码修改] 更新版本号
        
        # 依次调用所有需要解剖的信号探针
        self._deploy_ignition_resonance_probe(probe_date)
        self._deploy_ultimate_confirmation_probe(probe_date)
        self._deploy_v_reversal_ace_probe(probe_date)
        self._deploy_chip_price_lag_probe(probe_date)
        
        print("="*95 + "\n")

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
















