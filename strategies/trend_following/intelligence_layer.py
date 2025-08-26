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
        【V401.1 周线情报适配版】情报层总入口。
        - 核心升级: 新增了对周线战略情报的诊断和转化流程。
        """
        # print("--- [情报层总指挥官 V401.1 周线情报适配版] 开始执行所有诊断模块... ---")
        df = self.strategy.df_indicators
        self.strategy.atomic_states = {}

        # --- 阶段一: 基础K线与板形态识别 ---
        # print("    - [阶段1/7] 正在执行基础形态识别...")
        df = self.pattern_recognizer.identify_all(df)
        self.strategy.atomic_states.update(self.behavioral_intel.diagnose_kline_patterns(df))
        self.strategy.atomic_states.update(self.behavioral_intel.diagnose_board_patterns(df))

        # --- 代码修改开始 ---
        # --- 阶段二: 注入并转化周线战略情报 ---
        # print("    - [阶段2/7] 正在注入并转化周线战略情报...")
        self.strategy.atomic_states.update(self._diagnose_strategic_context(df))

        # --- 阶段三: 核心原子状态生成 (第一梯队) ---
        # print("    - [阶段3/7] 正在生成第一梯队原子状态 (无跨模块依赖)...")
        self.strategy.atomic_states.update(self.foundation_intel.diagnose_volatility_states(df))
        self.strategy.atomic_states.update(self.foundation_intel.diagnose_oscillator_states(df))
        self.strategy.atomic_states.update(self.structural_intel.diagnose_ma_states(df))
        self.strategy.atomic_states.update(self.structural_intel.diagnose_trend_dynamics(df))
        self.strategy.atomic_states.update(self.structural_intel.diagnose_fibonacci_support(df))
        chip_states, chip_triggers = self.chip_intel.run_chip_intelligence_command(df)
        self.strategy.atomic_states.update(chip_states)
        self.strategy.atomic_states.update(self.chip_intel.diagnose_dynamic_chip_states(df))
        self.strategy.atomic_states.update(self.chip_intel.diagnose_chip_risks_and_behaviors(df))
        self.strategy.atomic_states.update(self.chip_intel.diagnose_chip_opportunities(df))
        self.strategy.atomic_states.update(self.chip_intel.diagnose_peak_formation_dynamics(df))
        self.strategy.atomic_states.update(self.chip_intel.diagnose_peak_battle_dynamics(df))
        self.strategy.atomic_states.update(self.chip_intel.diagnose_chip_price_divergence(df))

        # --- 阶段四: 核心原子状态生成 (第二梯队 - 依赖第一梯队) ---
        # print("    - [阶段4/7] 正在生成第二梯队原子状态 (依赖第一梯队)...")
        # 将 diagnose_box_states 和 diagnose_platform_states 从旧的“司令部”中解放出来，直接调用
        self.strategy.atomic_states.update(self.structural_intel.diagnose_box_states(df))
        df, platform_states = self.structural_intel.diagnose_platform_states(df)
        self.strategy.atomic_states.update(platform_states)
        self.strategy.atomic_states.update(self.structural_intel.diagnose_structural_mechanics(df))
        self.strategy.atomic_states.update(self.behavioral_intel.diagnose_pullback_character(df))
        self.strategy.atomic_states.update(self.behavioral_intel.diagnose_behavioral_patterns(df))
        pullback_enhancements = self.behavioral_intel._diagnose_pullback_enhancement_matrix(df)
        self.strategy.atomic_states.update(self.behavioral_intel.diagnose_post_accumulation_phase(df))
        self.strategy.atomic_states.update(self.behavioral_intel.diagnose_holding_risks(df))

        # --- 阶段五: 复合与认知合成 ---
        # print("    - [阶段5/7] 正在执行复合与认知合成...")
        # 调用重构后的、只负责合成的 synthesize_composite_structures
        self.strategy.atomic_states.update(self.structural_intel.synthesize_composite_structures(df))
        self.strategy.atomic_states.update(self.cognitive_intel.diagnose_contextual_zones(df))
        self.strategy.atomic_states.update(self.cognitive_intel.diagnose_recent_reversal_context(df))
        self.strategy.atomic_states.update(self.cognitive_intel.diagnose_trend_stage_score(df))
        self.strategy.atomic_states.update(self.chip_intel.synthesize_prime_chip_opportunity(df))
        self.strategy.atomic_states.update(self.cognitive_intel.diagnose_market_structure_states(df))
        self.strategy.atomic_states.update(self.cognitive_intel.run_cognitive_synthesis_engine(df))
        self.strategy.atomic_states.update(self.cognitive_intel.synthesize_dynamic_offense_states(df))
        self.strategy.df_indicators = self.cognitive_intel.determine_main_force_behavior_sequence(df)

        # --- 阶段六: 生成最终的触发器与剧本 ---
        # print("    - [阶段6/7] 正在生成触发器与交易剧本...")
        trigger_events = self.playbook_engine.define_trigger_events(df)
        trigger_events.update(chip_triggers)
        self.strategy.trigger_events = trigger_events
        self.strategy.atomic_states.update(self.cognitive_intel.synthesize_advanced_tactics(df))
        self.strategy.atomic_states.update(self.cognitive_intel.synthesize_prime_tactic(df))
        self.strategy.atomic_states.update(self.cognitive_intel._diagnose_pullback_tactics_matrix(df, pullback_enhancements))
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
        # 步骤1: 生成由 PlaybookEngine 定义的标准剧本
        self.strategy.setup_scores, self.strategy.playbook_states = self.playbook_engine.generate_playbook_states(trigger_events)
        # 步骤2: 调用认知层，生成更复杂的、基于压缩突破的战术剧本
        squeeze_playbooks = self.cognitive_intel.synthesize_squeeze_playbooks(df)
        # 步骤3: 将新生成的战术剧本并入总的剧本状态池，供进攻层统一计分
        self.strategy.playbook_states.update(squeeze_playbooks)

        is_in_squeeze_window = self.strategy.atomic_states.get('VOL_STATE_SQUEEZE_WINDOW', pd.Series(False, index=df.index))
        is_bb_breakout = df['close_D'] > df.get('BBU_21_2.0_D', float('inf'))
        vol_ma_col = 'VOL_MA_21_D'
        if vol_ma_col in df.columns:
            is_volume_confirmed = df['volume_D'] > (df[vol_ma_col] * 1.5)
            trigger_events['VOL_BREAKOUT_FROM_SQUEEZE'] = is_bb_breakout & is_in_squeeze_window.shift(1).fillna(False) & is_volume_confirmed
        else:
            trigger_events['VOL_BREAKOUT_FROM_SQUEEZE'] = is_bb_breakout & is_in_squeeze_window.shift(1).fillna(False)
        
        # --- 阶段七: 最终报告 ---
        # print("--- [情报层总指挥官 V401.1] 所有诊断模块执行完毕。 ---")
        return trigger_events

    def _diagnose_strategic_context(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【新增】周线战略情报转化器
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




