# 文件: strategies/trend_following/intelligence_layer.py
# 情报层总指挥官 (重构版)
import pandas as pd
import numpy as np
from typing import Dict
from enum import Enum
from scipy.stats import linregress

from strategies.kline_pattern_recognizer import KlinePatternRecognizer
from .utils import get_params_block, get_param_value, create_persistent_state

# --- 从新目录导入所有情报模块 ---
from .intelligence.foundation_intelligence import FoundationIntelligence
from .intelligence.structural_intelligence import StructuralIntelligence
from .intelligence.chip_intelligence import ChipIntelligence
from .intelligence.behavioral_intelligence import BehavioralIntelligence
from .intelligence.cognitive_intelligence import CognitiveIntelligence
from .intelligence.playbook_engine import PlaybookEngine
from .intelligence.fund_flow_intelligence import FundFlowIntelligence
from .intelligence.dynamic_mechanics_engine import DynamicMechanicsEngine

class MainForceState(Enum):
    """
    定义主力行为序列的各个状态。
    """
    IDLE = 0           # 闲置/观察期
    ACCUMULATING = 1   # 吸筹期
    WASHING = 2        # 洗盘期
    MARKUP = 3         # 拉升期
    DISTRIBUTING = 4   # 派发期
    COLLAPSE = 5       # 崩盘期

class IntelligenceLayer:
    """
    【V400.0 重构版】情报层总指挥官
    - 核心职责: 1. 实例化所有专业化的情报子模块 (如筹码、结构、行为等)。
                2. 按照正确的依赖顺序，编排和调用这些子模块。
                3. 整合所有模块产出的原子状态和触发器，供下游层使用。
    - 设计原则: 高内聚、低耦合。本类只负责“编排”，不负责具体的“诊断”逻辑。
    """
    def __init__(self, strategy_instance):
        """
        初始化情报层总指挥官。
        """
        self.strategy = strategy_instance
        self.kline_params = get_params_block(self.strategy, 'kline_pattern_params')
        self.pattern_recognizer = KlinePatternRecognizer(params=self.kline_params)
        self.strategy.pattern_recognizer = self.pattern_recognizer # 将识别器传递给策略实例，供其他模块使用

        # 计算动态阈值，并传递给需要的子模块
        self.dynamic_thresholds = self._get_dynamic_thresholds(self.strategy.df_indicators)

        # 实例化所有子模块，注入依赖
        self.foundation_intel = FoundationIntelligence(self.strategy)
        self.structural_intel = StructuralIntelligence(self.strategy, self.dynamic_thresholds)
        self.chip_intel = ChipIntelligence(self.strategy, self.dynamic_thresholds)
        self.behavioral_intel = BehavioralIntelligence(self.strategy)
        self.cognitive_intel = CognitiveIntelligence(self.strategy)
        self.playbook_engine = PlaybookEngine(self.strategy)
        self.fund_flow_intel = FundFlowIntelligence(self.strategy)
        self.mechanics_engine = DynamicMechanicsEngine(self.strategy)

    def _get_dynamic_thresholds(self, df: pd.DataFrame) -> Dict:
        """
        【V335.2 核心指标版】动态阈值校准中心
        - 核心职责: 为各模块提供自适应标尺，只为最核心、最难被操纵的筹码结构指标提供校准。
        - 调用时机: 在所有模块实例化之前调用。
        """
        # print("        -> [动态阈值校准中心 V335.2] 启动...")
        thresholds = {}
        window = 250 # 使用过去一年的数据作为基准

        # 1. 成本加速度阈值：只相信最顶尖5%的进攻意图
        cost_accel_col = 'ACCEL_5_peak_cost_D'
        if cost_accel_col in df.columns:
            thresholds['cost_accel_significant'] = df[cost_accel_col].rolling(window).quantile(0.95)

        # 2. 筹码集中度加速度阈值：只相信最顶尖5%的吸筹决心
        conc_accel_col = 'ACCEL_5_concentration_90pct_D'
        if conc_accel_col in df.columns:
            thresholds['conc_accel_significant'] = df[conc_accel_col].rolling(window).quantile(0.05)
            
        print("        -> [动态阈值校准中心 V335.2] 校准完成。")
        return thresholds

    def run_all_diagnostics(self) -> Dict:
        """
        【V401.1 周线情报适配版】情报层总入口。
        - 核心升级: 新增了对周线战略情报的诊断和转化流程。
        """
        # print("--- [情报层总指挥官 V401.1 周线情报适配版] 开始执行所有诊断模块... ---")
        df = self.strategy.df_indicators
        self.strategy.atomic_states = {}

        # --- 阶段一: 基础K线与板形态识别 ---
        # 此阶段生成所有最基础的原子信号，它们仅依赖于数据工程层提供的指标。
        # print("    - [阶段1/7] 正在执行基础形态识别...")
        # 1.1 基础K线与形态
        df = self.pattern_recognizer.identify_all(df)
        self.strategy.atomic_states.update(self.behavioral_intel.diagnose_kline_patterns(df))
        self.strategy.atomic_states.update(self.behavioral_intel.diagnose_board_patterns(df))
        self.strategy.atomic_states.update(self.behavioral_intel.diagnose_price_volume_atomics(df))
        # 1.2 基础层情报 (波动率, 震荡指标, 资金流)
        self.strategy.atomic_states.update(self.foundation_intel.run_foundation_analysis_command())
        self.strategy.atomic_states.update(self.fund_flow_intel.diagnose_fund_flow_states(df))
        # 1.3 结构层基础情报 (均线, 箱体, 趋势动态)
        self.strategy.atomic_states.update(self.structural_intel.diagnose_ma_states(df))
        self.strategy.atomic_states.update(self.structural_intel.diagnose_box_states(df))
        self.strategy.atomic_states.update(self.structural_intel.diagnose_trend_dynamics(df))
        self.strategy.atomic_states.update(self.structural_intel.diagnose_structural_risks_and_regimes(df))
        # 1.4 筹码层基础情报 (静态, 行为, 动态)
        self.strategy.atomic_states.update(self.chip_intel.diagnose_chip_dynamics(df))
        self.strategy.atomic_states.update(self.chip_intel.diagnose_dynamic_chip_states(df))

        # --- 阶段二: 一级情报合成与上下文诊断 (依赖基础原子) ---
        # 此阶段基于第一阶段的原子信号，生成更复杂的上下文状态和复合信号。
        # print("    - [阶段 2/6] 正在进行一级情报合成与上下文诊断...")
        # 2.1 诊断战场上下文区域 (如高位风险区)
        self.strategy.atomic_states.update(self.cognitive_intel.diagnose_contextual_zones(df))
        self.strategy.atomic_states.update(self.cognitive_intel.diagnose_recent_reversal_context(df))
        # 2.2 诊断筹码平台、结构力学等
        df, platform_states = self.structural_intel.diagnose_platform_states(df)
        self.strategy.atomic_states.update(platform_states)
        self.strategy.atomic_states.update(self.structural_intel.diagnose_structural_mechanics(df))
        # 2.3 运行动态力学引擎
        self.mechanics_engine.run_dynamic_analysis_command()
        # 2.4 运行高级筹码诊断 (依赖上下文和基础筹码信号)
        chip_states, chip_triggers = self.chip_intel.run_chip_intelligence_command(df)
        self.strategy.atomic_states.update(chip_states)
        self.strategy.atomic_states.update(self.chip_intel.diagnose_chip_risks_and_behaviors(df))
        self.strategy.atomic_states.update(self.chip_intel.diagnose_chip_opportunities(df))
        self.strategy.atomic_states.update(self.chip_intel.diagnose_peak_formation_dynamics(df))
        self.strategy.atomic_states.update(self.chip_intel.diagnose_peak_battle_dynamics(df))
        self.strategy.atomic_states.update(self.chip_intel.diagnose_chip_price_divergence(df))
        # 2.5 运行高级行为诊断
        self.strategy.atomic_states.update(self.behavioral_intel.diagnose_behavioral_patterns(df))
        self.strategy.atomic_states.update(self.behavioral_intel.diagnose_post_accumulation_phase(df))
        self.strategy.atomic_states.update(self.behavioral_intel.diagnose_holding_risks(df))
        
        # --- 阶段三: 高级认知与结构合成 (依赖一级合成情报) ---
        # 此阶段进行最高层级的认知判断，定义市场结构、趋势阶段等。
        # print("    - [阶段 3/6] 正在进行高级认知与结构合成...")
        # 3.1 合成筹码与资金流的协同信号
        self.strategy.atomic_states.update(self.cognitive_intel.synthesize_chip_fund_flow_synergy(df))
        # 3.2 合成多时间维度趋势协同信号
        self.strategy.atomic_states.update(self.structural_intel.diagnose_mtf_trend_synergy(df))
        # 3.3 合成顶部行为
        self.strategy.atomic_states.update(self.cognitive_intel.synthesize_topping_behaviors(df))
        # 3.4 评分趋势阶段
        self.strategy.atomic_states.update(self.cognitive_intel.diagnose_trend_stage_score(df))
        # 3.5 合成黄金筹码机会
        self.strategy.atomic_states.update(self.chip_intel.synthesize_prime_chip_opportunity(df))
        # 3.6 诊断市场核心结构 (生成 STRUCTURE_MAIN_UPTREND_WAVE_S 等关键状态)
        self.strategy.atomic_states.update(self.cognitive_intel.diagnose_market_structure_states(df))
        # 3.7 合成复合结构
        self.strategy.atomic_states.update(self.structural_intel.synthesize_composite_structures(df))
        # 3.8 静态-动态融合
        self.strategy.atomic_states.update(self.structural_intel.diagnose_static_dynamic_fusion(df))

        # --- 阶段四: 战术场景诊断 (依赖高层认知状态) ---
        # 此阶段诊断具体的战术场景，如回踩，它们的定性依赖于市场结构等高层状态。
        # print("    - [阶段 4/6] 正在诊断具体战术场景...")
        self.strategy.atomic_states.update(self.behavioral_intel.diagnose_pullback_character(df))
        pullback_enhancements = self.behavioral_intel._diagnose_pullback_enhancement_matrix(df)

        # --- 阶段五: 最终认知合成与主力行为推演 ---
        # print("    - [阶段 5/6] 正在进行最终认知合成与主力行为推演...")
        self.strategy.atomic_states.update(self.cognitive_intel.run_cognitive_synthesis_engine(df))
        self.strategy.atomic_states.update(self.cognitive_intel.synthesize_dynamic_offense_states(df))
        self.strategy.df_indicators = self.cognitive_intel.determine_main_force_behavior_sequence(df)

        # --- 阶段六: 生成触发器、战法与交易剧本 ---
        # print("    - [阶段 6/6] 正在生成触发器、战法与交易剧本...")
        # 6.1 定义所有基础触发器
        trigger_events = self.playbook_engine.define_trigger_events(df)
        trigger_events.update(chip_triggers)
        self.strategy.trigger_events = trigger_events
        # 6.2 基于触发器和原子状态，合成高级战法
        self.strategy.atomic_states.update(self.structural_intel.diagnose_fibonacci_support(df))
        self.strategy.atomic_states.update(self.cognitive_intel.synthesize_advanced_tactics(df))
        self.strategy.atomic_states.update(self.cognitive_intel.synthesize_prime_tactic(df))
        self.strategy.atomic_states.update(self.cognitive_intel._diagnose_pullback_tactics_matrix(df, pullback_enhancements))
        self.strategy.atomic_states.update(self.cognitive_intel._diagnose_lock_chip_reconcentration_tactic(df))
        self.strategy.atomic_states.update(self.cognitive_intel._diagnose_lock_chip_rally_tactic(df))
        # 6.3 生成最终的交易剧本
        self.strategy.setup_scores, self.strategy.playbook_states = self.playbook_engine.generate_playbook_states(trigger_events)
        squeeze_playbooks = self.cognitive_intel.synthesize_squeeze_playbooks(df)
        self.strategy.playbook_states.update(squeeze_playbooks)
        # 6.4 (调试模块)
        debug_params = get_params_block(self.strategy, 'debug_params')
        if get_param_value(debug_params.get('enable_pullback_decision_log'), False):
            decision_log_df = self.cognitive_intel._create_pullback_decision_log(df, pullback_enhancements)
            final_tactic_days = decision_log_df.filter(like='FINAL_').any(axis=1)
            if final_tactic_days.any():
                # print("\n--- [回踩战术决策日志探针] ---")
                display_cols = [col for col in decision_log_df.columns if 'POTENTIAL_' in col or 'FINAL_' in col]
                print("决策日志 (POTENTIAL: 潜在机会, FINAL: 最终决策):")
                print(decision_log_df.loc[final_tactic_days, display_cols])
                print("--- [探针结束] ---\n")
        
        # --- 阶段七: 最终报告 ---
        print("--- [情报层总指挥官 V401.2] 所有诊断模块执行完毕。 ---")
        return trigger_events

    def _diagnose_strategic_context(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        周线战略情报转化器
        - 核心职责: 将从周线引擎注入的、连续的战略分数和状态信号，
                    转化为日线引擎各模块可以理解的、离散的布尔型原子状态。
        """
        strategic_states = {}
        default_series = pd.Series(False, index=df.index)

        # 1. 转化战略分数
        score_col = 'strategic_score_W'
        if score_col in df.columns:
            score = df[score_col]
            # 状态: 战略性看涨 (强顺风) - 用于进攻层加分
            strategic_states['CONTEXT_STRATEGIC_BULLISH_W'] = score >= 5
            # 状态: 战略性看跌 (逆风) - 用于判断层增加否决票
            strategic_states['CONTEXT_STRATEGIC_BEARISH_W'] = score < 0
        else:
            # 如果分数不存在，则所有相关状态都为False
            strategic_states['CONTEXT_STRATEGIC_BULLISH_W'] = default_series
            strategic_states['CONTEXT_STRATEGIC_BEARISH_W'] = default_series
            print("    - [周线情报转化-警告] 未找到 'strategic_score_W' 列。")

        # 2. 转化战略顶部风险信号
        topping_col = 'state_node_topping_W'
        if topping_col in df.columns:
            # 状态: 战略性顶部风险 - 用于判断层和离场层的强否决
            strategic_states['CONTEXT_STRATEGIC_TOPPING_RISK_W'] = df[topping_col]
        else:
            strategic_states['CONTEXT_STRATEGIC_TOPPING_RISK_W'] = default_series
            print(f"    - [周线情报转化-警告] 未找到 '{topping_col}' 列。")
            
        # 3. 转化战略点火信号 (可选，用于进攻层)
        ignition_col = 'state_node_ignition_W'
        if ignition_col in df.columns:
            strategic_states['CONTEXT_STRATEGIC_IGNITION_W'] = df[ignition_col]
        else:
            strategic_states['CONTEXT_STRATEGIC_IGNITION_W'] = default_series

        # print(f"    - [周线情报转化] 完成。已生成 {len(strategic_states)} 个战略级原子状态。")
        return strategic_states

    def _diagnose_long_term_daily_chip_context(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        日线长周期筹码战略上下文诊断模块
        - 核心职责: 直接从日线数据中读取长周期筹码指标，并将其转化为原子状态。
                    这些状态将作为战略层面的上下文，影响日线策略的决策。
        """
        states = {}
        default_series = pd.Series(False, index=df.index)

        # 检查所需的长周期筹码斜率/加速度列
        required_cols = [
            'SLOPE_21_concentration_90pct_D',
            'ACCEL_21_concentration_90pct_D',
            'SLOPE_21_chip_health_score_D'
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"            -> [警告] 日线长周期筹码战略诊断缺少数据: {missing_cols}，模块已跳过。")
            # 临时填充NaN，确保后续代码不报错，但信号将为False
            for col in missing_cols:
                df[col] = np.nan # Ensure column exists to prevent KeyError later

        # --- 1. 长期筹码集中度趋势 ---
        # 21日筹码集中度斜率 < 0 表示长期集中
        is_long_term_concentrating = df.get('SLOPE_21_concentration_90pct_D', default_series) < 0
        states['CONTEXT_CHIP_LONG_TERM_ACCUMULATION_D'] = is_long_term_concentrating

        # 21日筹码集中度斜率 > 0 表示长期发散
        is_long_term_diverging = df.get('SLOPE_21_concentration_90pct_D', default_series) > 0
        states['CONTEXT_CHIP_LONG_TERM_DIVERGENCE_D'] = is_long_term_diverging

        # --- 2. 长期筹码集中度加速度 ---
        # 21日筹码集中度加速度 < 0 表示长期集中加速
        is_long_term_accel_concentrating = df.get('ACCEL_21_concentration_90pct_D', default_series) < 0
        states['CONTEXT_CHIP_LONG_TERM_ACCEL_ACCUMULATION_D'] = is_long_term_accel_concentrating

        # 21日筹码集中度加速度 > 0 表示长期发散加速
        is_long_term_accel_diverging = df.get('ACCEL_21_concentration_90pct_D', default_series) > 0
        states['CONTEXT_CHIP_LONG_TERM_ACCEL_DIVERGENCE_D'] = is_long_term_accel_diverging

        # --- 3. 长期筹码健康度趋势 ---
        # 21日筹码健康分斜率 > 0 表示长期健康度改善
        is_long_term_health_improving = df.get('SLOPE_21_chip_health_score_D', default_series) > 0
        states['CONTEXT_CHIP_LONG_TERM_HEALTH_IMPROVING_D'] = is_long_term_health_improving

        # print(f"            -> [日线长周期筹码战略诊断] 已生成 {len(states)} 个战略级原子状态。")
        return states











