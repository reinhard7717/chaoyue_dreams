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
        【重构版】情报层总入口。
        - 核心职责: 按正确的依赖顺序，依次调用各专业情报模块，并整合其产出。
        """
        # print("--- [情报层总指挥官 V400.0] 开始执行所有诊断模块... ---")
        df = self.strategy.df_indicators
        self.strategy.atomic_states = {}

        # --- 阶段一: 基础K线与板形态识别 ---
        # 此阶段为DataFrame添加基础的形态识别列，是所有后续分析的基础。
        # print("    - [阶段1/5] 正在执行基础形态识别...")
        df = self.pattern_recognizer.identify_all(df)
        self.strategy.atomic_states.update(self.behavioral_intel.diagnose_kline_patterns(df))
        self.strategy.atomic_states.update(self.behavioral_intel.diagnose_board_patterns(df))

        # --- 阶段二: 核心原子状态生成 ---
        # 此阶段生成所有不依赖于其他诊断模块的基础原子状态。
        # print("    - [阶段2/5] 正在生成核心原子状态...")
        # 2.1 基础指标状态 (波动率, 震荡)
        self.strategy.atomic_states.update(self.foundation_intel.diagnose_volatility_states(df))
        self.strategy.atomic_states.update(self.foundation_intel.diagnose_oscillator_states(df))
        # 2.2 结构与趋势状态 (均线, 趋势动态, 斐波那契)
        self.strategy.atomic_states.update(self.structural_intel.diagnose_ma_states(df))
        self.strategy.atomic_states.update(self.structural_intel.diagnose_trend_dynamics(df))
        self.strategy.atomic_states.update(self.structural_intel.diagnose_fibonacci_support(df))
        # 2.3 筹码状态 (最核心的部分)
        chip_states, chip_triggers = self.chip_intel.run_chip_intelligence_command(df)
        self.strategy.atomic_states.update(chip_states)
        self.strategy.atomic_states.update(self.chip_intel.diagnose_dynamic_chip_states(df))
        self.strategy.atomic_states.update(self.chip_intel.diagnose_chip_opportunities(df))
        self.strategy.atomic_states.update(self.chip_intel.diagnose_chip_risks_and_behaviors(df))
        self.strategy.atomic_states.update(self.chip_intel.diagnose_peak_formation_dynamics(df))
        self.strategy.atomic_states.update(self.chip_intel.diagnose_peak_battle_dynamics(df))
        self.strategy.atomic_states.update(self.chip_intel.diagnose_chip_price_divergence(df))

        # --- 阶段三: 复合状态与行为模式诊断 ---
        # 此阶段的方法依赖于阶段二生成的原子状态。
        # print("    - [阶段3/5] 正在诊断复合状态与行为模式...")
        # 3.1 复合结构诊断
        df, structure_states = self.structural_intel.diagnose_market_structure_command(df)
        self.strategy.atomic_states.update(structure_states)
        self.strategy.atomic_states.update(self.structural_intel.diagnose_structural_mechanics(df))
        # 3.2 复合行为诊断
        self.strategy.atomic_states.update(self.behavioral_intel.diagnose_pullback_character(df))
        self.strategy.atomic_states.update(self.behavioral_intel.diagnose_behavioral_patterns(df))
        pullback_enhancements = self.behavioral_intel._diagnose_pullback_enhancement_matrix(df)
        self.strategy.atomic_states.update(self.behavioral_intel.diagnose_post_accumulation_phase(df))
        self.strategy.atomic_states.update(self.behavioral_intel.diagnose_holding_risks(df))

        # --- 阶段四: 顶层认知合成 ---
        # 此阶段是情报的最高层抽象，合成战场上下文、趋势阶段和主力行为序列。
        # print("    - [阶段4/5] 正在执行顶层认知合成...")
        self.strategy.atomic_states.update(self.cognitive_intel.diagnose_contextual_zones(df))
        self.strategy.atomic_states.update(self.cognitive_intel.diagnose_recent_reversal_context(df))
        self.strategy.atomic_states.update(self.cognitive_intel.diagnose_trend_stage_score(df))
        self.strategy.atomic_states.update(self.chip_intel.synthesize_prime_chip_opportunity(df))
        self.strategy.atomic_states.update(self.cognitive_intel.diagnose_market_structure_states(df))
        self.strategy.atomic_states.update(self.cognitive_intel.run_cognitive_synthesis_engine(df))
        self.strategy.atomic_states.update(self.cognitive_intel.synthesize_dynamic_offense_states(df))
        self.strategy.df_indicators = self.cognitive_intel.determine_main_force_behavior_sequence(df)

        # --- 阶段五: 生成最终的触发器与剧本 ---
        # 这是情报层的最终产出，供下游决策层使用。
        # print("    - [阶段5/5] 正在生成触发器与交易剧本...")
        trigger_events = self.playbook_engine.define_trigger_events(df)
        trigger_events.update(chip_triggers) # 合并筹码司令部产出的特殊触发器
        self.strategy.trigger_events = trigger_events # 挂载到strategy实例，供后续模块使用
        # self.strategy.atomic_states.update(self.cognitive_intel._diagnose_lock_chip_reconcentration_tactic(df))
        # self.strategy.atomic_states.update(self.cognitive_intel._diagnose_lock_chip_rally_tactic(df))
        self.strategy.atomic_states.update(self.cognitive_intel.synthesize_prime_tactic(df))
        self.strategy.atomic_states.update(self.cognitive_intel._diagnose_pullback_tactics_matrix(df, pullback_enhancements))
        # [修改原因] 新增战术决策日志探针的调用逻辑，用于深度调试。
        debug_params = get_params_block(self.strategy, 'debug_params')
        if get_param_value(debug_params.get('enable_pullback_decision_log'), False):
            decision_log_df = self.cognitive_intel._create_pullback_decision_log(df, pullback_enhancements)
            
            # 筛选出有最终决策的日子
            final_tactic_days = decision_log_df.filter(like='FINAL_').any(axis=1)
            if final_tactic_days.any():
                print("\n--- [回踩战术决策日志探针] ---")
                # 为了清晰，只显示关键的决策列
                display_cols = [col for col in decision_log_df.columns if 'POTENTIAL_' in col or 'FINAL_' in col]
                print("决策日志 (POTENTIAL: 潜在机会, FINAL: 最终决策):")
                print(decision_log_df.loc[final_tactic_days, display_cols])
                print("--- [探针结束] ---\n")
        self.strategy.setup_scores, self.strategy.playbook_states = self.playbook_engine.generate_playbook_states(trigger_events)

        is_in_squeeze_window = self.strategy.atomic_states.get('VOL_STATE_SQUEEZE_WINDOW', pd.Series(False, index=df.index))
        is_bb_breakout = df['close_D'] > df.get('BBU_21_2.0_D', float('inf'))
        vol_ma_col = 'VOL_MA_21_D'
        if vol_ma_col in df.columns:
            is_volume_confirmed = df['volume_D'] > (df[vol_ma_col] * 1.5)
            trigger_events['VOL_BREAKOUT_FROM_SQUEEZE'] = is_bb_breakout & is_in_squeeze_window.shift(1).fillna(False) & is_volume_confirmed
        else:
            trigger_events['VOL_BREAKOUT_FROM_SQUEEZE'] = is_bb_out & is_in_squeeze_window.shift(1).fillna(False)
        print("--- [情报层总指挥官 V400.0] 所有诊断模块执行完毕。 ---")
        return trigger_events
