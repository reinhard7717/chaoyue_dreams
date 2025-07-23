# 文件: strategies/trend_following_strategy.py
# 版本: V62.0 - 增强调试日志版
from copy import deepcopy
import json
import logging
from decimal import Decimal
import os
from utils.data_sanitizer import sanitize_for_json
from typing import Any, Dict, List, Optional, Tuple
from scipy.signal import find_peaks
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences
from strategies.kline_pattern_recognizer import KlinePatternRecognizer
from utils.config_loader import load_strategy_config

logger = logging.getLogger(__name__)

# ================================================================================================
# |   最高统帅部 (Supreme Headquarters) -> apply_strategy()                                      |
# ================================================================================================
#           │
#           ├─> 1. 情报总局 (Intelligence General Administration) -> _run_all_diagnostics()
#           │      │
#           │      ├─> 风险情报总局 -> _diagnose_all_risk_signals()
#           │      ├─> 筹码情报总参谋部 -> _diagnose_chip_intelligence()
#           │      ├─> 动态惯性引擎 -> _diagnose_trend_dynamics()
#           │      ├─> 资本动向总参谋部 -> _diagnose_capital_states()
#           │      ├─> 野战部队 (均线) -> _diagnose_ma_states()
#           │      ├─> 工兵部队 (箱体/平台) -> _diagnose_box_states(), _diagnose_platform_states()
#           │      ├─> 基础侦察部队 (K线/盘面) -> _diagnose_kline_patterns(), _diagnose_board_patterns()
#           │      └─> 联合作战司令部 (情报融合) -> _diagnose_market_structure_states(), _diagnose_strategic_setups()
#           │
#           ├─> 2. 参谋部联席会议 (Assessment & Scoring) -> _run_scoring_and_assessment()
#           │      │
#           │      ├─> 进攻方案评估中心 -> _calculate_entry_score()
#           │      │      └─> (下辖) 指挥棒模型 -> _apply_final_score_adjustments()
#           │      └─> 最高风险裁决所 -> _calculate_risk_score()
#           │
#           ├─> 3. 总司令部 (Final Decision Making) -> _make_final_decisions()
#           │      │
#           │      └─> (下辖) 离场指令部 -> _calculate_exit_signals()
#           │
#           ├─> 4. 沙盘推演中心 (Simulation) -> _run_position_management_simulation()
#           │
#           └─> 5. 后勤与战报总署 (Logistics & Reporting)
#                  │
#                  ├─> 战报司令部 -> prepare_db_records()
#                  │      └─> (下辖) _create_db_record_template(), _fill_risk_details()
#                  ├─> 风险量化局 -> _quantify_risk_reasons()
#                  └─> 盘中预警中心 -> generate_intraday_alerts()

# ================================================================================================
# | ★ 军事监察与战地验尸总署 (Inspector General & Field Forensics) - [非作战序列]                |
# |    └─> _deploy_field_coroner_probe(), _probe_entry_score_details(), _probe_risk_score_details()|
# ================================================================================================


class TrendFollowStrategy:
    """
    趋势跟踪策略 (V62.0 - 增强调试日志版)
    - 核心升级: 新增 _format_debug_dates 辅助函数，并全局应用于所有诊断日志，使其能输出具体发生日期，极大提升调试效率。
    """   
    def __init__(self, config: dict):
        self.unified_config = config
        self.strategy_info = self._get_params_block('strategy_info')
        self.scoring_params = self._get_params_block('four_layer_scoring_params')
        self.exit_strategy_params = self._get_params_block('exit_strategy_params')
        self.risk_veto_params = self._get_params_block('risk_veto_params')
        self.debug_params = self._get_params_block('debug_params')
        self.kline_params = self._get_params_block('kline_pattern_params')
        self.pattern_recognizer = KlinePatternRecognizer(params=self.kline_params)
        self._last_score_details_df = None
        self._last_risk_details_df = None

        # 调用方法获取静态蓝图，并将其存储为实例属性，供后续所有方法安全调用
        self.playbook_blueprints = self._get_playbook_blueprints()
        # 在初始化时建立“风险档案室”
        self.risk_playbook_blueprints = self._get_risk_playbook_blueprints()

    #  日志格式化辅助函数 ▼▼▼
    def _format_debug_dates(self, signal_series: pd.Series, display_limit: int = 10) -> str:
        """
        【V92.1 修复版】
        - 核心修复: 恢复了显示具体日期的能力，对调试至关重要。
        - 特性: 当日期不多时全部显示；日期过多时，显示最近的N个并附上总数。
        """
        if not isinstance(signal_series, pd.Series) or signal_series.dtype != bool:
            return ""
        
        active_dates = signal_series.index[signal_series]
        count = len(active_dates)
        
        if count == 0:
            return ""
            
        date_strings = [d.strftime('%Y-%m-%d') for d in active_dates]
        
        if count > display_limit:
            # 如果日期太多，只显示最近的N个并附上总数
            return f" -> 日期: [...{date_strings[-display_limit:]}] (共 {count} 天)"
        else:
            # 否则全部显示
            return f" -> 日期: {date_strings}"

    def _get_param_value(self, param: Any, default: Any = None) -> Any:
        """
        【V40.3 新增】健壮的参数值解析器。
        """
        if isinstance(param, dict) and 'value' in param:
            return param['value']
        if param is not None:
            return param
        return default

    def _get_params_block(self, block_name: str, default_return: Any = None) -> dict:
        """
        【V240.0 通用情报检索版】终极参数块获取器。
        - 核心重构: 废除旧的单一路径查找逻辑，建立一个智能的多路径搜索系统。
        - 新规则:
          1. 首先，在核心战术模块 'strategy_params.trend_follow' 中查找。
          2. 如果未找到，则将搜索范围扩大到整个配置文件的根级别。
          3. 如果仍然未找到，才返回默认值。
        - 收益: 无论参数块在JSON中处于哪个层级，都能被准确找到。
                此方法现在对JSON结构具有极高的适应性和鲁棒性，是此类问题的最终解决方案。
        """
        if default_return is None:
            default_return = {}

        # 搜索路径1: 核心战术模块内部 (strategy_params -> trend_follow)
        trend_follow_params = self.unified_config.get('strategy_params', {}).get('trend_follow', {})
        params = trend_follow_params.get(block_name)

        # 如果在路径1未找到，则启动路径2: 在整个配置文件的根级别进行搜索
        if params is None:
            params = self.unified_config.get(block_name)

        # 最终裁定：如果找到了，就返回；否则返回默认值
        if params is not None:
            return params
        
        return default_return

    def _get_periods_for_timeframe(self, indicator_params: dict, timeframe: str) -> Optional[list]:
        """
        【V5.0 】根据时间周期从指标配置中智能获取正确的'periods'。
        """
        if not indicator_params:
            return None
        if 'configs' in indicator_params and isinstance(indicator_params['configs'], list):
            for config_item in indicator_params['configs']:
                if timeframe in config_item.get('apply_on', []):
                    return config_item.get('periods')
            return None
        elif 'periods' in indicator_params:
            apply_on = indicator_params.get('apply_on', [])
            if not apply_on or timeframe in apply_on:
                return indicator_params.get('periods')
            else:
                return None
        return None

    def _ensure_numeric_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V40.6 智能净化版】数据类型标准化引擎
        """
        converted_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                first_valid_item = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if isinstance(first_valid_item, Decimal):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    converted_cols.append(col)
        if not converted_cols:
            print("      -> 所有数值列类型正常，无需转换。")
        return df

    # 辅助函数，用于定义 CONTEXT_TREND_DETERIORATING
    def _define_context_trend_deteriorating(self, df, atomic_states):
        default_series = pd.Series(False, index=df.index)
        is_in_divergence_window = atomic_states.get('CAPITAL_STATE_DIVERGENCE_WINDOW', default_series)
        is_long_ma_slope_negative = df.get('SLOPE_21_EMA_55_D', 0) < 0
        is_short_ma_slope_negative = df.get('SLOPE_5_EMA_13_D', 0) < 0
        is_long_ma_accel_negative = df.get('ACCEL_21_EMA_55_D', 0) < 0
        is_long_chip_slope_negative = df.get('peak_cost_slope_55d_D', 0) < 0
        unconditional_deterioration = (is_long_ma_slope_negative & is_short_ma_slope_negative & is_long_ma_accel_negative & is_long_chip_slope_negative)
        return unconditional_deterioration & ~is_in_divergence_window


    # 最高统帅部 (Supreme Headquarters)
    # -> 核心入口: apply_strategy()
    def apply_strategy(self, df: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        【V223.0 指挥链重塑版】
        - 核心重构: 将主作战流程严格按照“情报->评估->决策->执行”的指挥链进行重塑。
                    每一个步骤都由一个权责清晰的独立模块完成，确保指挥体系的绝对清晰。
        """
        print("======================================================================")
        print(f"====== 日期: {df.index[-1].date()} | 正在执行【战术引擎 V223.0 指挥链重塑版】 ======")
        print("======================================================================")

        if df is None or df.empty: return pd.DataFrame(), {}
        df = self._ensure_numeric_types(df)

        # --- 步骤1：情报总局 (Intelligence Gathering) ---
        # 运行所有诊断模块，收集所有正面和负面的“原子情报”。
        print("--- [指挥链 1/4] 情报总局：正在收集所有战场情报... ---")
        df, atomic_states, trigger_events = self._run_all_diagnostics(df, params)

        # --- 步骤2：参谋部联席会议 (Assessment & Scoring) ---
        # 对所有情报进行量化评估，形成“进攻价值分”和“战场风险分”。
        print("--- [指挥链 2/4] 参谋部：正在对情报进行量化评估... ---")
        df, score_details_df, risk_details_df = self._run_scoring_and_assessment(df, params, atomic_states, trigger_events)
        self._last_score_details_df = score_details_df
        self._last_risk_details_df = risk_details_df

        # --- 步骤3：总司令部 (Final Decision Making) ---
        # 根据进攻价值和战场风险，下达最终的作战指令（是否进攻、是否撤退）。
        print("--- [指挥链 3/4] 总司令部：正在下达最终作战指令... ---")
        df = self._make_final_decisions(df, params)
        
        # --- 步骤4：沙盘推演 (Position Management Simulation) ---
        # 模拟从建仓到离场的全过程。
        print("--- [指挥链 4/4] 作战推演：正在模拟全程战术动作... ---")
        df = self._run_position_management_simulation(df, params)

        print(f"====== 【战术引擎 V223.0】执行完毕 ======")
        
        # debug_params = self._get_params_block('debug_params')
        # probe_date = self._get_param_value(debug_params.get('probe_date'))
        # if probe_date:
        #     self._deploy_field_coroner_probe(df, probe_date)

        return df, atomic_states


    # 1. 情报总局 (Intelligence General Administration)
    #    -> 核心职责: 统一收集所有战场情报，形成原子状态报告
    #    -> 总指挥: _run_all_diagnostics()
    def _run_all_diagnostics(self, df: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        【V257.0 最终整编版】情报总局
        - 核心重构: 经过V256.0(筹码司令部)和V257.0(认知引擎)两次核心整编，本方法
                      已成为一个高度精炼的、负责调度各大专业化司令部的“总指挥部”。
        - 指挥链条:
          1.  【基础侦察】: 启动K线模式识别器。
          2.  【专业化司令部】: 依次调用平台、筹码、均线、资本等专业司令部，收集原子情报。
          3.  【联合作战分析】: 在各司令部完成基础分析后，启动“筹码-价格”联合分析。
          4.  【认知升华】: 启动“认知综合引擎”，将所有情报升华为高维战术概念。
          5.  【战术触发定义】: 最终定义所有可供决策的“开火信号”。
        - 收益: 指挥链极度清晰，权责分明，彻底消除了情报碎片化和职能交叉的问题。
        """
        print("--- [总指挥] 步骤1: 运行所有诊断模块... ---")
        # 启动基础的K线模式识别器
        df = self.pattern_recognizer.identify_all(df)
        if 'close_D' in df.columns: df['pct_change_D'] = df['close_D'].pct_change()

        # --- 依次调用各大专业化司令部，收集原子情报 ---
        
        # 1. 平台与阵地司令部
        df, platform_states = self._diagnose_platform_states(df, params)
        
        # 2. 筹码情报最高司令部 (已完成一体化整编)
        # 一次性获取所有筹码状态和触发器
        chip_states, chip_triggers = self._run_chip_intelligence_command(df, params)

        # 3. 汇总所有基础原子情报
        atomic_states = {
            **chip_states,                                  # 注入来自筹码最高司令部的情报
            **self._diagnose_ma_states(df, params),         # 均线野战部队
            **self._diagnose_oscillator_states(df, params), # 心理战与市场情绪侦察部
            **self._diagnose_capital_states(df, params),    # 资本动向总参谋部
            **self._diagnose_volatility_states(df, params), # 能量与波动侦察部
            **self._diagnose_box_states(df, params),        # 箱体工兵部队
            **self._diagnose_kline_patterns(df, params),    # 基础K线侦察部队
            **self._diagnose_board_patterns(df, params),    # 盘面特征侦察部队
            **platform_states,                              # 注入来自平台司令部的情报
            **self._diagnose_trend_dynamics(df, params)     # 动态惯性引擎
        }
        
        # --- 在所有基础情报生成后，启动跨部门的联合作战分析 ---
        
        # 4. 筹码-价格行为联合分析部 (跨部门协作)
        atomic_states.update(self._diagnose_chip_price_action(df, atomic_states))
        
        # 5. 市场结构总参谋部
        atomic_states.update(self._diagnose_market_structure_states(df, params, atomic_states))
        
        # --- 启动认知综合引擎，完成从“情报”到“认知”的最终升华 ---
        
        # 6. 认知综合引擎 (已完成一体化整编)
        # 一次性完成所有高维认知合成
        atomic_states.update(self._run_cognitive_synthesis_engine(df, atomic_states))

        # --- 基于所有情报和认知，定义最终的战术触发事件 ---
        
        # 7. 战术触发事件定义中心
        trigger_events = self._define_trigger_events(df, params, atomic_states)
        # 将筹码总参谋部提供的“触发事件”合并到总事件池中
        trigger_events.update(chip_triggers)
        
        # 特殊处理：从波动率压缩中突破的触发事件
        is_in_squeeze_window = atomic_states.get('VOL_STATE_SQUEEZE_WINDOW', pd.Series(False, index=df.index))
        is_bb_breakout = df['close_D'] > df.get('BBU_21_2.0_D', float('inf'))
        trigger_events['VOL_BREAKOUT_FROM_SQUEEZE'] = is_bb_breakout & is_in_squeeze_window.shift(1).fillna(False)

        return df, atomic_states, trigger_events

    # ─> 战术条令与剧本参谋部 (Tactical Doctrine & Playbook Dept.)
    #    -> 核心职责: 负责全军作战计划的理论制定与动态应用。
    #    ├─> 军事档案馆 (Military Archives)
    #    │   -> 核心职责: 存储静态的、不变的作战预案“蓝图”。
    #    │   ├─> _get_playbook_blueprints()
    def _get_playbook_blueprints(self) -> List[Dict]:
        """
        【V83.0 新增】剧本蓝图知识库
        - 职责: 定义所有剧本的静态属性（名称、家族、评分规则等）。
        - 特性: 这是一个纯粹的数据结构，不依赖任何动态数据，可以在初始化时被安全地缓存。
        """
        return [
            # --- 反转/逆势家族 (REVERSAL_CONTRARIAN) ---
            {
                'name': 'ABYSS_GAZE_S', 'cn_name': '【S级】深渊凝视', 'family': 'REVERSAL_CONTRARIAN',
                'type': 'setup', 'score': 320, 'side': 'left', 'comment': 'S级: 市场极度恐慌后的第一个功能性反转。'
            },
            {
                'name': 'CAPITULATION_PIT_REVERSAL', 'cn_name': '【动态】投降坑反转', 'family': 'REVERSAL_CONTRARIAN',
                'type': 'setup_score', 'side': 'left', 'comment': '根据坑的深度和反转K强度动态给分。', 'allow_memory': False,
                'scoring_rules': { 'min_setup_score_to_trigger': 51, 'base_score': 160, 'score_multiplier': 1.0, 'trigger_bonus': {'TRIGGER_REVERSAL_CONFIRMATION_CANDLE': 50}}
            },
            {
                'name': 'CAPITAL_DIVERGENCE_REVERSAL', 'cn_name': '【A-级】资本逆行者', 'family': 'REVERSAL_CONTRARIAN',
                'type': 'setup', 'score': 230, 'side': 'left', 'comment': 'A-级: 在“底背离”机会窗口中出现的反转确认。'
            },
            {
                'name': 'BEAR_TRAP_RALLY', 'cn_name': '【C+级】熊市反弹', 'family': 'REVERSAL_CONTRARIAN',
                'type': 'setup', 'score': 180, 'side': 'left', 'comment': 'C+级: 熊市背景下，价格首次钝化，并由长期趋势企稳信号触发。'
            },
            {
                'name': 'WASHOUT_REVERSAL_A', 'cn_name': '【A级】巨阴洗盘反转', 'family': 'REVERSAL_CONTRARIAN',
                'type': 'setup', 'score': 260, 'side': 'left', 'comment': 'A级: 极端洗盘后的拉升前兆。', 'allow_memory': False
            },
            {
                'name': 'BOTTOM_STABILIZATION_B', 'cn_name': '【B级】底部企稳', 'family': 'REVERSAL_CONTRARIAN',
                'type': 'setup', 'score': 190, 'side': 'left', 'comment': 'B级: 股价严重超卖偏离均线后，出现企稳阳线。'
            },
            # --- 趋势/动能家族 (TREND_MOMENTUM) ---
            {
                'name': 'CHIP_PLATFORM_PULLBACK', 'cn_name': '【S-级】筹码平台回踩', 'family': 'TREND_MOMENTUM',
                'type': 'setup', 'score': 330, 'side': 'right', 
                'comment': 'S-级: 股价回踩由筹码峰形成的、位于趋势线上方的稳固平台，并获得支撑。这是极高质量的“空中加油”信号。'
            },
            {
                'name': 'TREND_EMERGENCE_B_PLUS', 'cn_name': '【B+级】右侧萌芽', 'family': 'TREND_MOMENTUM',
                'type': 'setup', 'score': 210, 'comment': 'B+级: 填补战术真空。在左侧反转后，短期均线走好但尚未站上长线时的首次介入机会。'
            },
            {
                'name': 'DEEP_ACCUMULATION_BREAKOUT', 'cn_name': '【动态】潜龙出海', 'family': 'TREND_MOMENTUM',
                'type': 'setup_score', 'side': 'right', 'comment': '根据深度吸筹分数动态给分。若伴随筹码点火，则确定性更高。',
                'scoring_rules': { 'min_setup_score_to_trigger': 51, 'base_score': 200, 'score_multiplier': 1.5, 'trigger_bonus': {'CHIP_EVENT_IGNITION': 60} }
            },
            {
                'name': 'ENERGY_COMPRESSION_BREAKOUT', 'cn_name': '【动态】能量压缩突破', 'family': 'TREND_MOMENTUM',
                'type': 'precondition_score', 'side': 'right', 'comment': '根据当天前提和历史准备共同给分。若伴随筹码点火，则确定性更高。',
                'scoring_rules': { 'base_score': 150, 'min_score_to_trigger': 180, 'conditions': {'VOL_STATE_SQUEEZE_WINDOW': 50, 'CHIP_STATE_CONCENTRATION_SQUEEZE': 30, 'MA_STATE_CONVERGING': 20}, 'setup_bonus': {'ENERGY_COMPRESSION': 0.2, 'HEALTHY_MARKUP': 0.1}, 'trigger_bonus': {'CHIP_EVENT_IGNITION': 60} }
            },
            {
                'name': 'PLATFORM_SUPPORT_PULLBACK', 
                'cn_name': '【动态】多维支撑回踩', 
                'family': 'TREND_MOMENTUM',
                'type': 'setup_score', 
                'side': 'right', 
                'comment': '根据平台质量(筹码结构、趋势背景)和支撑级别(均线/筹码峰)进行动态评分。',
                'scoring_rules': {
                    'min_setup_score_to_trigger': 50,  # 要求平台质量分至少达到50
                    'base_score': 180,                 # 给予一个稳健的基础分
                    'score_multiplier': 1.2            # 允许根据平台质量分进行加成
                }
            },
            {
                'name': 'HEALTHY_BOX_BREAKOUT', 'cn_name': '【A-级】健康箱体突破', 'family': 'TREND_MOMENTUM',
                'type': 'setup', 'score': 220, 'side': 'right', 'comment': 'A-级: 在一个位于趋势均线上方的健康箱体内盘整后，发生的向上突破。'
            },
            {
                'name': 'HEALTHY_MARKUP_A', 'cn_name': '【A级】健康主升浪', 'family': 'TREND_MOMENTUM',
                'type': 'setup_score', 'side': 'right', 'comment': 'A级: 在主升浪中回踩或延续，根据主升浪健康分动态加成。',
                'scoring_rules': { 
                    'min_setup_score_to_trigger': 60, # 要求主升浪健康分至少达到60
                    'base_score': 240, 
                    'score_multiplier': 1.2, # 允许根据健康分进行加成
                    'conditions': { 'MA_STATE_DIVERGING': 20, 'OSC_STATE_MACD_BULLISH': 15, 'MA_STATE_PRICE_ABOVE_SHORT_MA': 0 } 
                }
            },
            {
                'name': 'N_SHAPE_CONTINUATION_A', 'cn_name': '【A级】N字板接力', 'family': 'TREND_MOMENTUM',
                'type': 'precondition_score', 'side': 'right', 'comment': 'A级: 强势股的经典趋势中继信号。若伴随筹码点火，则确定性更高。',
                'scoring_rules': { 'base_score': 250, 'min_score_to_trigger': 250, 'conditions': {'SETUP_SCORE_N_SHAPE_CONTINUATION_ABOVE_80': 0}, 'trigger_bonus': {'CHIP_EVENT_IGNITION': 60} }
            },
            {
                'name': 'GAP_SUPPORT_PULLBACK_B_PLUS', 'cn_name': '【B+级】缺口支撑回踩', 'family': 'TREND_MOMENTUM',
                'type': 'setup', 'score': 190, 'side': 'right', 'comment': 'B+级: 回踩前期跳空缺口获得支撑并反弹。'
            },
            # --- 特殊事件家族 (SPECIAL_EVENT) ---
            {
                'name': 'EARTH_HEAVEN_BOARD', 'cn_name': '【S+】地天板', 'family': 'SPECIAL_EVENT',
                'type': 'event_driven', 'score': 380, 'side': 'left', 'comment': '市场情绪的极致反转。'
            },
            {
                'name': 'FAULT_REBIRTH_S', 
                'cn_name': '【S级】断层新生', 
                'family': 'SPECIAL_EVENT',
                'type': 'event_driven', 
                'score': 350, 
                'side': 'left', 
                'comment': 'S级: 识别因成本断层导致的筹码结构重置，是极高价值的特殊事件信号。'
            },
        ]

    #    │   └─> _get_risk_playbook_blueprints()
    def _get_risk_playbook_blueprints(self) -> List[Dict]:
        """
        【V202.0 动态防御版】
        - 核心升级: 新增了三个基于动态筹码斜率的风险剧本，用于量化趋势衰竭的风险。
        """
        return [
            # --- 结构性风险 (Structure Risk) ---
            {'name': 'STRUCTURE_BREAKDOWN', 'cn_name': '【结构】关键支撑破位', 'score': 100},
            {'name': 'UPTHRUST_DISTRIBUTION', 'cn_name': '【结构】冲高派发', 'score': 80},
            # --- 动能衰竭风险 (Momentum Exhaustion) ---
            {'name': 'CHIP_EXHAUSTION', 'cn_name': '【动能】筹码成本加速衰竭', 'score': 60},
            {'name': 'CHIP_DIVERGENCE', 'cn_name': '【动能】筹码顶背离', 'score': 70},
            # ▼▼▼ 动态风险剧本 ▼▼▼
            {
                'name': 'PROFIT_EVAPORATION', 'cn_name': '【动态】获利盘蒸发', 'score': 75,
                'comment': '总获利盘斜率转负，市场赚钱效应快速消失，是强烈的离场信号。'
            },
            {
                'name': 'PEAK_WEAKENING', 'cn_name': '【动态】主峰根基动摇', 'score': 55,
                'comment': '主筹码峰的稳定性或占比开始下降，主力阵地可能在瓦解。'
            },
            {
                'name': 'RESISTANCE_BUILDING', 'cn_name': '【动态】上方压力积聚', 'score': 35,
                'comment': '上方套牢盘不减反增，表明进攻受阻，后续突破难度加大。'
            }
        ]

    #    └─> 作战计划推演室 (War Gaming Section)
    #        -> 核心职责: 将静态“蓝图”与实时战况结合，生成可执行的动态作战计划。
    #        -> 对应方法: _get_playbook_definitions()
    def _get_playbook_definitions(self, df: pd.DataFrame, trigger_events: Dict[str, pd.Series], atomic_states: Dict[str, pd.Series]) -> Tuple[List[Dict], Dict[str, pd.Series]]:
        """
        【V224.0 参谋部职能一体化版】
        - 核心重构: 裁撤了独立的 `_calculate_setup_conditions` 部门，将其核心职能
                    ——“战机准备状态评估”，作为前置任务并入本方法。
        - 新流程: 1. 首先，在本方法内部计算所有 SETUP_SCORE。
                   2. 然后，利用这些新鲜出炉的评估分数，继续完成后续的剧本“水合”工作。
        - 输出变更: 此方法现在返回一个元组 (hydrated_playbooks, setup_scores)，确保评估结果能被下游消费。
        """
        print("    - [剧本参谋部 V224.0 一体化版] 启动...")
        default_series = pd.Series(False, index=df.index)

        # --- 步骤1: 战机准备状态评估 (Setup Readiness Assessment) ---
        print("      -> 步骤1/3: 正在进行战机准备状态评估 (Setup Scoring)...")
        setup_scores = {}
        scoring_matrix = self.setup_scoring_matrix
        for setup_name, rules in scoring_matrix.items():
            if not self._get_param_value(rules.get('enabled'), True):
                continue
            
            # --- “投降坑” 专属评分逻辑 ---
            if setup_name == 'CAPITULATION_PIT':
                p_cap_pit = rules
                must_have_score = self._get_param_value(p_cap_pit.get('must_have_score'), 40)
                bonus_score = self._get_param_value(p_cap_pit.get('bonus_score'), 25)
                must_have_conditions = atomic_states.get('OPP_STATE_NEGATIVE_DEVIATION', default_series)
                bonus_conditions_1 = atomic_states.get('CHIP_STATE_LOW_PROFIT', default_series)
                bonus_conditions_2 = atomic_states.get('CHIP_STATE_SCATTERED', default_series)
                base_score = must_have_conditions.astype(int) * must_have_score
                bonus_score_total = (bonus_conditions_1.astype(int) * bonus_score) + (bonus_conditions_2.astype(int) * bonus_score)
                final_score = (base_score + bonus_score_total).where(must_have_conditions, 0)
                setup_scores[f'SETUP_SCORE_{setup_name}'] = final_score
            # --- “平台质量” 专属评分逻辑 ---
            elif setup_name == 'PLATFORM_QUALITY':
                p_quality = rules
                must_have_cond = atomic_states.get('PLATFORM_STATE_STABLE_FORMED', default_series)
                base_score_val = self._get_param_value(p_quality.get('base_score'), 40)
                base_score = must_have_cond.astype(int) * base_score_val
                bonus_score = pd.Series(0.0, index=df.index)
                bonus_rules = p_quality.get('bonus', {})
                for state, score in bonus_rules.items():
                    state_series = atomic_states.get(state, default_series)
                    bonus_score += state_series.astype(int) * score
                setup_scores[f'SETUP_SCORE_{setup_name}'] = (base_score + bonus_score).where(must_have_cond, 0)
            else:
                # --- 其他所有剧本使用通用评分逻辑 ---
                current_score = pd.Series(0.0, index=df.index)
                must_have_rules = rules.get('must_have', {})
                must_have_passed = pd.Series(True, index=df.index)
                for state, score in must_have_rules.items():
                    state_series = atomic_states.get(state, default_series)
                    current_score += state_series * score
                    must_have_passed &= state_series
                
                any_of_rules = rules.get('any_of_must_have', {})
                any_of_passed = pd.Series(False, index=df.index)
                if any_of_rules:
                    any_of_score_component = pd.Series(0.0, index=df.index)
                    for state, score in any_of_rules.items():
                        state_series = atomic_states.get(state, default_series)
                        any_of_score_component.loc[state_series] = score
                        any_of_passed |= state_series
                    current_score += any_of_score_component
                else:
                    any_of_passed = pd.Series(True, index=df.index)

                bonus_rules = rules.get('bonus', {})
                for state, score in bonus_rules.items():
                    state_series = atomic_states.get(state, default_series)
                    current_score += state_series * score
                
                final_validity = must_have_passed & any_of_passed
                setup_scores[f'SETUP_SCORE_{setup_name}'] = current_score.where(final_validity, 0)
        print("      -> 战机准备状态评估完成。")
        # ▲▲▲【代码修改 V224.0】▲▲▲

        # --- 步骤2: 作战计划动态“水合” (Playbook Hydration) ---
        print("      -> 步骤2/3: 正在进行作战计划动态“水合”...")
        hydrated_playbooks = deepcopy(self.playbook_blueprints)
        
        # 从刚刚计算出的 setup_scores 中获取评估分
        score_cap_pit = setup_scores.get('SETUP_SCORE_CAPITULATION_PIT', pd.Series(0, index=df.index))
        score_deep_accum = setup_scores.get('SETUP_SCORE_DEEP_ACCUMULATION', pd.Series(0, index=df.index))
        score_nshape_cont = setup_scores.get('SETUP_SCORE_N_SHAPE_CONTINUATION', pd.Series(0, index=df.index))
        score_gap_support = setup_scores.get('SETUP_SCORE_GAP_SUPPORT_PULLBACK', pd.Series(0, index_df.index))
        score_bottoming_process = setup_scores.get('SETUP_SCORE_BOTTOMING_PROCESS', pd.Series(0, index=df.index))
        score_healthy_markup = setup_scores.get('SETUP_SCORE_HEALTHY_MARKUP', pd.Series(0, index=df.index))
        score_platform_quality = setup_scores.get('SETUP_SCORE_PLATFORM_QUALITY', pd.Series(0, index=df.index))

        # 准备原子状态
        capital_divergence_window = atomic_states.get('CAPITAL_STATE_DIVERGENCE_WINDOW', default_series)
        setup_bottom_passivation = atomic_states.get('MA_STATE_BOTTOM_PASSIVATION', default_series)
        setup_washout_reversal = atomic_states.get('KLINE_STATE_WASHOUT_WINDOW', default_series)
        setup_healthy_box = atomic_states.get('BOX_STATE_HEALTHY_CONSOLIDATION', default_series)
        recent_reversal_context = atomic_states.get('CONTEXT_RECENT_REVERSAL_SIGNAL', default_series)
        ma_short_slope_positive = atomic_states.get('MA_STATE_SHORT_SLOPE_POSITIVE', default_series)
        is_trend_healthy = atomic_states.get('CONTEXT_OVERALL_TREND_HEALTHY', default_series)
        
        # 准备动态布尔条件
        atomic_states['SETUP_SCORE_N_SHAPE_CONTINUATION_ABOVE_80'] = score_nshape_cont > 80
        atomic_states['SETUP_SCORE_HEALTHY_MARKUP_ABOVE_60'] = score_healthy_markup > 60

        # 为每个剧本注入其专属的动态数据 (水合过程)
        for playbook in hydrated_playbooks:
            name = playbook['name']
            
            if name == 'ABYSS_GAZE_S':
                playbook['trigger'] = trigger_events.get('TRIGGER_DOMINANT_REVERSAL', default_series)
                playbook['setup'] = score_cap_pit > 80
            elif name == 'CAPITULATION_PIT_REVERSAL':
                playbook['setup_score_series'] = score_cap_pit
                playbook['trigger'] = (trigger_events.get('TRIGGER_REVERSAL_CONFIRMATION_CANDLE', default_series) | trigger_events.get('TRIGGER_BREAKOUT_CANDLE', default_series))
            elif name == 'CAPITAL_DIVERGENCE_REVERSAL':
                playbook['setup'] = capital_divergence_window
                playbook['trigger'] = trigger_events.get('TRIGGER_REVERSAL_CONFIRMATION_CANDLE', default_series)
            elif name == 'BEAR_TRAP_RALLY':
                playbook['setup'] = setup_bottom_passivation
                playbook['trigger'] = trigger_events.get('TRIGGER_TREND_STABILIZING', default_series)
            elif name == 'WASHOUT_REVERSAL_A':
                playbook['setup'] = setup_washout_reversal
                playbook['trigger'] = trigger_events.get('TRIGGER_BREAKOUT_CANDLE', default_series)
            elif name == 'BOTTOM_STABILIZATION_B':
                playbook['setup'] = score_bottoming_process > 50
                playbook['trigger'] = trigger_events.get('TRIGGER_BREAKOUT_CANDLE', default_series)
            elif name == 'TREND_EMERGENCE_B_PLUS':
                playbook['setup'] = recent_reversal_context & ma_short_slope_positive
                playbook['trigger'] = trigger_events.get('TRIGGER_TREND_CONTINUATION_CANDLE', default_series)
            elif name == 'PLATFORM_SUPPORT_PULLBACK':
                playbook['setup_score_series'] = score_platform_quality
                trigger_ma_rebound = trigger_events.get('TRIGGER_PULLBACK_REBOUND', default_series)
                trigger_chip_rebound = trigger_events.get('TRIGGER_PLATFORM_PULLBACK_REBOUND', default_series)
                playbook['trigger'] = trigger_ma_rebound | trigger_chip_rebound
            elif name == 'HEALTHY_MARKUP_A':
                playbook['setup_score_series'] = score_healthy_markup
                trigger_rebound = trigger_events.get('TRIGGER_PULLBACK_REBOUND', default_series)
                trigger_continuation = trigger_events.get('TRIGGER_TREND_CONTINUATION_CANDLE', default_series)
                playbook['trigger'] = trigger_rebound | trigger_continuation
            elif name == 'HEALTHY_BOX_BREAKOUT':
                playbook['setup'] = setup_healthy_box
                playbook['trigger'] = trigger_events.get('BOX_EVENT_BREAKOUT', default_series)
            elif name == 'GAP_SUPPORT_PULLBACK_B_PLUS':
                playbook['setup'] = (score_gap_support > 60)
                playbook['trigger'] = trigger_events.get('TRIGGER_PULLBACK_REBOUND', default_series)
            elif name == 'CHIP_PLATFORM_PULLBACK':
                setup_platform_formed = atomic_states.get('PLATFORM_STATE_STABLE_FORMED', default_series)
                playbook['setup'] = setup_platform_formed & is_trend_healthy
                playbook['trigger'] = trigger_events.get('TRIGGER_PLATFORM_PULLBACK_REBOUND', default_series)
            elif name == 'ENERGY_COMPRESSION_BREAKOUT':
                playbook['trigger'] = trigger_events.get('TRIGGER_BREAKOUT_CANDLE', default_series)
            elif name == 'DEEP_ACCUMULATION_BREAKOUT':
                playbook['setup_score_series'] = score_deep_accum
                playbook['trigger'] = trigger_events.get('TRIGGER_BREAKOUT_CANDLE', default_series)
            elif name == 'N_SHAPE_CONTINUATION_A':
                playbook['trigger'] = trigger_events.get('TRIGGER_N_SHAPE_BREAKOUT', default_series)
            elif name == 'EARTH_HEAVEN_BOARD':
                playbook['trigger'] = trigger_events.get('TRIGGER_EARTH_HEAVEN_BOARD', default_series)
        
        print("      -> 作战计划动态“水合”完成。")

        # --- 步骤3: 统一交战规则审查 (Unified Rules of Engagement) ---
        print("      -> 步骤3/3: 正在执行统一交战规则审查...")
        is_trend_deteriorating = atomic_states.get('CONTEXT_TREND_DETERIORATING', default_series)
        for playbook in hydrated_playbooks:
            if playbook.get('side') == 'right':
                original_trigger = playbook.get('trigger', default_series)
                playbook['trigger'] = original_trigger & ~is_trend_deteriorating
        print("      -> “统一交战规则”审查完毕，所有右侧进攻性操作已被置于战略监控之下。")

        # ▼▼▼【代码修改 V224.0】: 返回评估分数，确保下游可用 ▼▼▼
        return hydrated_playbooks, setup_scores

    # ─> 战术触发事件定义中心 (Tactical Trigger Definition Center)
    #    -> 核心职责: 识别那些可以作为“开火信号”的瞬时战术事件(Trigger)。
    #    -> 指挥官: _define_trigger_events()
    def _define_trigger_events(self, df: pd.DataFrame, params: dict, atomic_states: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V234.0 最终净化版 - 战术触发事件定义中心】
        - 核心升级: 严格遵循“V234.0 作战条例”，所有参数均从唯一的 trigger_event_params 配置块中获取，
                    确保了配置的单一来源原则，使整个触发体系清晰、健壮、易于维护。
        - 职责: 识别所有可以作为“开火信号”的瞬时战术事件(Trigger)。
        """
        print("        -> [触发事件中心 V234.0] 启动，正在定义所有原子化触发事件...")
        triggers = {}
        default_series = pd.Series(False, index=df.index)
        
        # ▼▼▼【代码修改 V234.0】: 统一从 trigger_event_params 获取所有参数 ▼▼▼
        trigger_params = self._get_params_block('trigger_event_params')
        if not self._get_param_value(trigger_params.get('enabled'), True):
            print("          -> 触发事件引擎被禁用，跳过。")
            return triggers
        # ▲▲▲【代码修改 V234.0】▲▲▲
            
        vol_ma_col = 'VOL_MA_21_D'

        # --- 1. K线形态触发器 (Candlestick Triggers) ---
        # 1.1 【通用级】反转确认阳线
        p_reversal = trigger_params.get('reversal_confirmation_candle', {})
        if self._get_param_value(p_reversal.get('enabled'), True):
            is_green = df['close_D'] > df['open_D']
            is_strong_rally = df['pct_change_D'] > self._get_param_value(p_reversal.get('min_pct_change'), 0.03)
            is_closing_strong = df['close_D'] > (df['high_D'] + df['low_D']) / 2
            triggers['TRIGGER_REVERSAL_CONFIRMATION_CANDLE'] = is_green & is_strong_rally & is_closing_strong

        # 1.2 【精英级】显性反转阳线 (在通用级基础上，要求力量压制前一日)
        p_dominant = trigger_params.get('dominant_reversal_candle', {})
        if self._get_param_value(p_dominant.get('enabled'), True):
            base_reversal_signal = triggers.get('TRIGGER_REVERSAL_CONFIRMATION_CANDLE', default_series)
            today_body_size = df['close_D'] - df['open_D']
            yesterday_body_size = abs(df['close_D'].shift(1) - df['open_D'].shift(1))
            was_yesterday_red = df['close_D'].shift(1) < df['open_D'].shift(1)
            recovery_ratio = self._get_param_value(p_dominant.get('recovery_ratio'), 0.5)
            is_power_recovered = today_body_size >= (yesterday_body_size * recovery_ratio)
            triggers['TRIGGER_DOMINANT_REVERSAL'] = base_reversal_signal & (~was_yesterday_red | is_power_recovered)

        # 1.3 【企稳型】突破阳线 (通常用于底部企稳或平台整理后的首次突破)
        p_breakout_candle = trigger_params.get('breakout_candle', {})
        if self._get_param_value(p_breakout_candle.get('enabled'), True):
            boll_mid_col = 'BBM_21_2.0_D'
            if boll_mid_col in df.columns:
                min_body_ratio = self._get_param_value(p_breakout_candle.get('min_body_ratio'), 0.4)
                body_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
                is_strong_positive_candle = (
                    (df['close_D'] > df['open_D']) &
                    (((df['close_D'] - df['open_D']) / body_range).fillna(1.0) >= min_body_ratio)
                )
                is_breaking_boll_mid = df['close_D'] > df[boll_mid_col]
                triggers['TRIGGER_BREAKOUT_CANDLE'] = is_strong_positive_candle & is_breaking_boll_mid

        # 1.4 【进攻型】能量释放阳线 (强调实体和成交量的双重确认)
        p_energy = trigger_params.get('energy_release', {})
        if self._get_param_value(p_energy.get('enabled'), True) and vol_ma_col in df.columns:
            is_positive_day = df['close_D'] > df['open_D']
            body_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
            body_ratio = (df['close_D'] - df['open_D']) / body_range
            is_strong_body = body_ratio.fillna(1.0) > self._get_param_value(p_energy.get('min_body_ratio'), 0.5)
            volume_ratio = self._get_param_value(p_energy.get('volume_ratio'), 1.5)
            is_volume_spike = df['volume_D'] > df[vol_ma_col] * volume_ratio
            triggers['TRIGGER_ENERGY_RELEASE'] = is_positive_day & is_strong_body & is_volume_spike

        # --- 2. 结构与趋势触发器 (Structure & Trend Triggers) ---
        # 2.1 【经典】放量突破近期高点
        p_vol_breakout = trigger_params.get('volume_spike_breakout', {})
        if self._get_param_value(p_vol_breakout.get('enabled'), True) and vol_ma_col in df.columns:
            volume_ratio = self._get_param_value(p_vol_breakout.get('volume_ratio'), 2.0)
            lookback = self._get_param_value(p_vol_breakout.get('lookback_period'), 20)
            is_volume_spike = df['volume_D'] > df[vol_ma_col] * volume_ratio
            is_price_breakout = df['close_D'] > df['high_D'].shift(1).rolling(lookback).max()
            triggers['TRIGGER_VOLUME_SPIKE_BREAKOUT'] = is_volume_spike & is_price_breakout

        # 2.2 【均线】回踩支撑反弹
        p_ma_rebound = trigger_params.get('pullback_rebound_trigger_params', {})
        if self._get_param_value(p_ma_rebound.get('enabled'), True):
            support_ma_period = self._get_param_value(p_ma_rebound.get('support_ma'), 21)
            support_ma_col = f'EMA_{support_ma_period}_D'
            if support_ma_col in df.columns:
                was_touching_support = df['low_D'].shift(1) <= df[support_ma_col].shift(1)
                is_rebounded_above = df['close_D'] > df[support_ma_col]
                is_positive_day = df['close_D'] > df['open_D']
                triggers['TRIGGER_PULLBACK_REBOUND'] = was_touching_support & is_rebounded_above & is_positive_day

        # 2.3 【筹码】回踩平台反弹 (S级战术动作)
        p_platform_rebound = trigger_params.get('platform_pullback_trigger_params', {})
        if self._get_param_value(p_platform_rebound.get('enabled'), True):
            platform_price_col = 'PLATFORM_PRICE_STABLE'
            if platform_price_col in df.columns:
                proximity_ratio = self._get_param_value(p_platform_rebound.get('proximity_ratio'), 0.01)
                is_touching_platform = df['low_D'] <= df[platform_price_col] * (1 + proximity_ratio)
                is_closing_above = df['close_D'] > df[platform_price_col]
                is_positive_day = df['close_D'] > df['open_D']
                triggers['TRIGGER_PLATFORM_PULLBACK_REBOUND'] = is_touching_platform & is_closing_above & is_positive_day

        # 2.4 【趋势】趋势延续确认K线
        p_cont = trigger_params.get('trend_continuation_candle', {})
        if self._get_param_value(p_cont.get('enabled'), True):
            lookback_period = self._get_param_value(p_cont.get('lookback_period'), 8)
            is_positive_day = df['close_D'] > df['open_D']
            is_new_high = df['close_D'] >= df['high_D'].shift(1).rolling(window=lookback_period).max()
            triggers['TRIGGER_TREND_CONTINUATION_CANDLE'] = is_positive_day & is_new_high

        # --- 3. 复合形态与指标触发器 (Pattern & Indicator Triggers) ---
        # 3.1 N字形态突破 (依赖原子状态)
        p_nshape = self.kline_params.get('n_shape_params', {})
        if self._get_param_value(p_nshape.get('enabled'), True):
            n_shape_consolidation_state = atomic_states.get('KLINE_STATE_N_SHAPE_CONSOLIDATION', default_series)
            consolidation_high = df['high_D'].where(n_shape_consolidation_state, np.nan).ffill()
            is_breaking_consolidation = df['close_D'] > consolidation_high.shift(1)
            triggers['TRIGGER_N_SHAPE_BREAKOUT'] = (df['close_D'] > df['open_D']) & is_breaking_consolidation

        # 3.2 指标金叉 (MACD)
        p_cross = trigger_params.get('indicator_cross_params', {})
        if self._get_param_value(p_cross.get('enabled'), True):
            macd_p = p_cross.get('macd_cross', {})
            if self._get_param_value(macd_p.get('enabled'), True):
                macd_col, signal_col = 'MACD_13_34_8_D', 'MACDs_13_34_8_D'
                if all(c in df.columns for c in [macd_col, signal_col]):
                    is_golden_cross = (df[macd_col] > df[signal_col]) & (df[macd_col].shift(1) <= df[signal_col].shift(1))
                    low_level = self._get_param_value(macd_p.get('low_level'), -0.5)
                    triggers['TRIGGER_MACD_LOW_CROSS'] = is_golden_cross & (df[macd_col] < low_level)

        # --- 4. 从其他诊断模块接收的事件 (Event Reception) ---
        # 这些事件由其他专业部门生成，本部门只负责接收和汇报
        triggers['TRIGGER_BOX_BREAKOUT'] = atomic_states.get('BOX_EVENT_BREAKOUT', default_series)
        triggers['TRIGGER_EARTH_HEAVEN_BOARD'] = atomic_states.get('BOARD_EVENT_EARTH_HEAVEN', default_series)
        triggers['TRIGGER_TREND_STABILIZING'] = atomic_states.get('MA_STATE_D_STABILIZING', default_series)

        # --- 5. 最终安全检查 (Final Safety Check) ---
        # 确保所有触发器都已正确初始化，防止因计算失败导致后续流程出错
        for key in list(triggers.keys()):
            if triggers[key] is None:
                triggers[key] = pd.Series(False, index=df.index)
            else:
                triggers[key] = triggers[key].fillna(False)
                
        print("        -> [触发事件中心 V234.0] 所有触发事件定义完成。")
        return triggers

    # ─> 心理战与市场情绪侦察部 (Psychological Warfare & Sentiment Reconnaissance)
    #    -> 核心职责: 通过震荡指标，侦测市场情绪的“超买”与“超卖”。
    #    -> 指挥官: _diagnose_oscillator_states()
    def _diagnose_oscillator_states(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """【V234.0 最终净化版】震荡指标状态诊断中心"""
        states = {}
        p = self._get_params_block('oscillator_state_params')
        if not self._get_param_value(p.get('enabled'), False): return states
        
        # --- RSI 相关状态 ---
        rsi_col = 'RSI_13_D'
        if rsi_col in df.columns:
            states['OSC_STATE_RSI_OVERBOUGHT'] = df[rsi_col] > self._get_param_value(p.get('rsi_overbought'), 80)
            states['OSC_STATE_RSI_OVERSOLD'] = df[rsi_col] < self._get_param_value(p.get('rsi_oversold'), 25)
        
        # --- MACD 相关状态 ---
        macd_h_col = 'MACDh_13_34_8_D'
        macd_z_col = 'MACD_HIST_ZSCORE_D'
        if macd_h_col in df.columns:
            states['OSC_STATE_MACD_BULLISH'] = df[macd_h_col] > 0
        if macd_z_col in df.columns:
            is_price_higher = df['close_D'] > df['close_D'].rolling(10).max().shift(1)
            is_macd_z_lower = df[macd_z_col] < df[macd_z_col].rolling(10).max().shift(1)
            states['OSC_STATE_MACD_DIVERGENCE'] = is_price_higher & is_macd_z_lower

        # ▼▼▼ BIAS机会状态的诊断 ▼▼▼
        p_bias = p.get('bias_dynamic_threshold', {})
        bias_col = 'BIAS_55_D'
        if bias_col in df.columns:
            window = self._get_param_value(p_bias.get('window'), 120)
            quantile = self._get_param_value(p_bias.get('quantile'), 0.1)
            dynamic_oversold_threshold = df[bias_col].rolling(window=window).quantile(quantile)
            states['OPP_STATE_NEGATIVE_DEVIATION'] = df[bias_col] < dynamic_oversold_threshold

        return states

    # ─> 能量与波动侦察部 (Energy & Volatility Reconnaissance)
    #    -> 核心职责: 侦测市场能量的“积蓄”(压缩)与“释放”(放量)。
    #    -> 指挥官: _diagnose_volatility_states()
    def _diagnose_volatility_states(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """【V224.0 逻辑内嵌版】波动率与成交量状态诊断"""
        states = {}
        p = self._get_params_block('volatility_state_params')
        if not self._get_param_value(p.get('enabled'), False): return states
        
        default_series = pd.Series(False, index=df.index)
        bbw_col = 'BBW_21_2.0_D'
        if bbw_col in df.columns:
            squeeze_threshold = df[bbw_col].rolling(60).quantile(self._get_param_value(p.get('squeeze_percentile'), 0.1))
            # 识别“波动率极度压缩”的单次触发事件
            squeeze_event = (df[bbw_col] < squeeze_threshold) & (df[bbw_col].shift(1) >= squeeze_threshold)
            states['VOL_EVENT_SQUEEZE'] = squeeze_event
        else:
            squeeze_event = default_series

        vol_ma_col = 'VOL_MA_21_D'
        if 'volume_D' in df.columns and vol_ma_col in df.columns:
            states['VOL_STATE_SHRINKING'] = df['volume_D'] < df[vol_ma_col] * self._get_param_value(p.get('shrinking_ratio'), 0.8)

        # 作战单元：建立“能量待爆发”的持续性监控窗口
        p_context = p.get('squeeze_context', {})
        persistence_days = self._get_param_value(p_context.get('persistence_days'), 10)
        
        if vol_ma_col in df.columns:
            # 定义窗口的“打破条件”：成交量显著放大
            volume_break_ratio = self._get_param_value(p_context.get('volume_break_ratio'), 1.5)
            break_condition = df['volume_D'] > df[vol_ma_col] * volume_break_ratio
            
            # --- 原 _create_persistent_state 的核心逻辑开始 ---
            persistent_series = pd.Series(False, index=df.index)
            entry_indices = df.index[squeeze_event] # 使用上面定义的 squeeze_event
            
            for entry_idx in entry_indices:
                window_end_idx = entry_idx + pd.Timedelta(days=persistence_days)
                # 确定实际的窗口范围，不超过数据末尾
                actual_window = df.loc[entry_idx:window_end_idx]
                
                # 在此窗口内查找第一个“打破条件”满足的位置
                break_points = actual_window.index[break_condition[actual_window.index]]
                
                # 如果找到了打破点，则状态持续到打破点
                if not break_points.empty:
                    end_date = break_points[0]
                    persistent_series.loc[entry_idx:end_date] = True
                # 如果没找到，则状态持续整个窗口期
                else:
                    persistent_series.loc[actual_window.index] = True
            
            states['VOL_STATE_SQUEEZE_WINDOW'] = persistent_series
            # --- 原 _create_persistent_state 的核心逻辑结束 ---
        else:
            states['VOL_STATE_SQUEEZE_WINDOW'] = default_series
            
        return states


    # 风险情报总局 (Risk Intelligence Bureau)
    #    -> 核心职责: 汇总所有负面/风险信号。
    #    -> 指挥官: _diagnose_all_risk_signals()
    def _diagnose_all_risk_signals(self, df: pd.DataFrame, params: dict, atomic_states: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V222.0 新增】风险情报总局
        - 核心职责: 统一指挥所有风险侦察部队，收集所有风险信号，形成一份完整的、
                    不包含任何评分的“每日风险简报”。这是风险评估的唯一情报来源。
        """
        print("        -> [风险情报总局 V222.0] 启动，正在汇总所有风险信号...")
        risk_signals = {}

        # 1. 调动“结构风险”侦察部队
        risk_signals.update(self._diagnose_upthrust_distribution(df, params))
        risk_signals.update(self._diagnose_structure_breakdown(df, params))

        # 2. 汇总来自“原子状态”的战略级风险
        # 这些是最高优先级的风险，直接从 atomic_states 中提取
        strategic_risks = [
            'RISK_COST_BASIS_COLLAPSE', 
            'RISK_CONFIDENCE_DETERIORATION',
            'RISK_CAPITAL_STRUCT_BEARISH_DIVERGENCE',
            'RISK_CAPITAL_STRUCT_MAIN_FORCE_DISTRIBUTING',
            'DYN_TREND_WEAKENING_DECELERATING',
            'DYN_TREND_BEARISH_ACCELERATING',
            'CONTEXT_TREND_DETERIORATING',
            'PLATFORM_FAILURE'
        ]
        for risk_name in strategic_risks:
            if risk_name in atomic_states:
                risk_signals[risk_name] = atomic_states[risk_name]

        print(f"        -> [风险情报总局 V222.0] 情报汇总完毕，共监控 {len(risk_signals)} 类风险信号。")
        return risk_signals

    # 冲高回落侦察连: _diagnose_upthrust_distribution()
    def _diagnose_upthrust_distribution(self, df: pd.DataFrame, exit_params: dict) -> pd.Series:
        """
        【V91.2 函数调用修复版】
        - 核心修复: 使用 numpy.maximum 替代错误的 pd.max，以正确计算上影线。
        """
        p = exit_params.get('upthrust_distribution_params', {})
        if not self._get_param_value(p.get('enabled'), False):
            return pd.Series(False, index=df.index)

        overextension_ma_period = self._get_param_value(p.get('overextension_ma_period'), 55)
        overextension_threshold = self._get_param_value(p.get('overextension_threshold'), 0.3)
        upper_shadow_ratio = self._get_param_value(p.get('upper_shadow_ratio'), 0.5)
        high_volume_quantile = self._get_param_value(p.get('high_volume_quantile'), 0.75)
        
        ma_col = f'EMA_{overextension_ma_period}_D'
        vol_ma_col = 'VOL_MA_21_D'
        
        required_cols = ['open_D', 'high_D', 'low_D', 'close_D', 'volume_D', ma_col, vol_ma_col]
        if not all(col in df.columns for col in required_cols):
            print("          -> [警告] 缺少诊断'高位放量长上影'所需列，跳过。")
            return pd.Series(False, index=df.index)

        is_overextended = (df['close_D'] / df[ma_col] - 1) > overextension_threshold
        total_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        
        # ▼▼▼【代码修改 V91.2】: 使用 np.maximum 替代 pd.max ▼▼▼
        upper_shadow = (df['high_D'] - np.maximum(df['open_D'], df['close_D']))
        # ▲▲▲【代码修改 V91.2】▲▲▲
        
        has_long_upper_shadow = (upper_shadow / total_range) >= upper_shadow_ratio
        volume_threshold = df['volume_D'].rolling(window=21).quantile(high_volume_quantile)
        is_high_volume = df['volume_D'] > volume_threshold
        is_weak_close = df['close_D'] < (df['high_D'] + df['low_D']) / 2
        
        signal = is_overextended & has_long_upper_shadow & is_high_volume & is_weak_close
        # print(f"          -> '高位放量长上影派发' 风险诊断完成，共激活 {signal.sum()} 天。{self._format_debug_dates(signal)}")
        return signal

    # 结构破位侦察连: _diagnose_structure_breakdown()
    def _diagnose_structure_breakdown(self, df: pd.DataFrame, exit_params: dict) -> pd.Series:
        """
        诊断“结构性破位”风险 (Structure Breakdown)。
        这是一个非常重要的趋势终结信号。
        """
        p = exit_params.get('structure_breakdown_params', {})
        if not self._get_param_value(p.get('enabled'), False):
            return pd.Series(False, index=df.index)

        # 1. 定义参数
        breakdown_ma_period = self._get_param_value(p.get('breakdown_ma_period'), 21)
        min_pct_change = self._get_param_value(p.get('min_pct_change'), -0.03)
        high_volume_quantile = self._get_param_value(p.get('high_volume_quantile'), 0.75)
        
        ma_col = f'EMA_{breakdown_ma_period}_D'
        
        required_cols = ['open_D', 'close_D', 'pct_change_D', 'volume_D', ma_col]
        if not all(col in df.columns for col in required_cols):
            print("          -> [警告] 缺少诊断'结构性破位'所需列，跳过。")
            return pd.Series(False, index=df.index)

        # 2. 计算各项条件
        # 条件A: 是一根有分量的阴线
        is_decisive_negative_candle = df['pct_change_D'] < min_pct_change
        
        # 条件B: 相对放量
        volume_threshold = df['volume_D'].rolling(window=21).quantile(high_volume_quantile)
        is_high_volume = df['volume_D'] > volume_threshold
        
        # 条件C: 跌破了关键均线
        is_breaking_ma = df['close_D'] < df[ma_col]
        
        # 3. 组合所有条件
        signal = is_decisive_negative_candle & is_high_volume & is_breaking_ma
        
        # print(f"          -> '结构性破位' 风险诊断完成，共激活 {signal.sum()} 天。{self._format_debug_dates(signal)}")
        return signal

    # 筹码情报总参谋部 (Chip Intelligence Dept.)
    #    -> 核心职责: 统一分析所有筹码相关情报。
    #    -> 总参谋长: _diagnose_chip_intelligence()
    def _run_chip_intelligence_command(self, df: pd.DataFrame, params: dict) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
        """
        【V256.0 一体化整编版】筹码情报最高司令部 (Chip Intelligence Supreme Command)
        - 核心重构: 将原有的四个独立筹码诊断方法
                      (_diagnose_chip_intelligence, _diagnose_advanced_chip_structures,
                       _diagnose_chip_alert_conditions, _diagnose_historical_context)
                      整合成一个统一、高效、权责分明的最高指挥部。
        - 作战流程:
          1. 军备检查: 检查所有必需的筹码数据是否到位。
          2. 基础情报分析: 生成所有原子的、动态的筹码状态。
          3. 高维结构解读: 解读筹码健康分等高级指标。
          4. 历史背景政审: 对目标的长期行为进行审查，识别“历史污点”。
          5. 核心戒备等级裁定: 基于所有情报，设定最终的CHIPCON戒备等级。
        - 收益: 极大提升了代码的内聚性、可读性和维护性，消除了情报碎片化，
                确保所有筹码相关的分析都在一个统一的指挥链下完成。
        """
        print("        -> [筹码情报最高司令部 V256.0] 启动，正在执行一体化分析...")
        states = {}
        triggers = {}
        default_series = pd.Series(False, index=df.index)

        # --- 1. 军备检查 (Armory Inspection) ---
        p = self._get_params_block('chip_feature_params')
        if not self._get_param_value(p.get('enabled'), False):
            print("          -> 筹码情报最高司令部被禁用，跳过。")
            return states, triggers

        required_cols = [
            'concentration_90pct_D', 'concentration_90pct_slope_5d_D',
            'concentration_90pct_slope_21d_D', # 历史审查部需要
            'SLOPE_5_peak_cost_D', 'SLOPE_5_total_winner_rate_D',
            'SLOPE_5_peak_stability_D', 'SLOPE_5_peak_percent_D',
            'SLOPE_5_pressure_above_D', 'peak_cost_accel_5d_D',
            'chip_health_score' # 高级结构诊断室需要
        ]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"          -> [严重警告] 缺少筹码诊断所需的列: {missing}。引擎将返回空结果。")
            return states, triggers

        # --- 2. 基础情报分析 (Basic Intelligence Analysis) ---
        # (原 _diagnose_chip_intelligence 的核心逻辑)
        p_struct = p.get('structure_params', {})
        conc_col = 'concentration_90pct_D'
        conc_slope_col = 'concentration_90pct_slope_5d_D'
        
        conc_thresh_abs = self._get_param_value(p_struct.get('high_concentration_threshold'), 0.15)
        slope_tolerance_healthy = self._get_param_value(p_struct.get('slope_tolerance_healthy'), 0.001)
        states['CHIP_STATE_HIGHLY_CONCENTRATED'] = (df[conc_col] < conc_thresh_abs) & (df[conc_slope_col] <= slope_tolerance_healthy)
        
        rapid_concentration_threshold = self._get_param_value(p_struct.get('rapid_concentration_threshold'), -0.005)
        states['CHIP_RAPID_CONCENTRATION'] = df.get(conc_slope_col, 0) < rapid_concentration_threshold

        cost_collapse_threshold = self._get_param_value(p_struct.get('cost_collapse_threshold'), -0.01)
        states['RAW_SIGNAL_COST_COLLAPSE'] = df.get('SLOPE_5_peak_cost_D', 0) < cost_collapse_threshold
        
        winner_rate_collapse_threshold = self._get_param_value(p_struct.get('winner_rate_collapse_threshold'), -1.0)
        states['RAW_SIGNAL_WINNER_RATE_COLLAPSE'] = df.get('SLOPE_5_total_winner_rate_D', 0) < winner_rate_collapse_threshold

        p_ignition = p.get('ignition_params', {})
        if self._get_param_value(p_ignition.get('enabled'), True):
            accel_threshold = self._get_param_value(p_ignition.get('accel_threshold'), 0.01)
            triggers['TRIGGER_CHIP_IGNITION'] = df.get('peak_cost_accel_5d_D', 0) > accel_threshold

        # --- 3. 高维结构解读 (Advanced Structure Interpretation) ---
        # (原 _diagnose_advanced_chip_structures 的逻辑)
        health_score = df.get('chip_health_score')
        if health_score is not None:
            states['CHIP_HEALTH_EXCELLENT'] = health_score > 85

        # --- 4. 历史背景政审 (Historical Context Vetting) ---
        # (原 _diagnose_historical_context 的逻辑)
        long_term_conc_slope_col = 'concentration_90pct_slope_21d_D'
        distribution_threshold = 0.0001 
        states['RISK_CONTEXT_LONG_TERM_DISTRIBUTION'] = df[long_term_conc_slope_col] > distribution_threshold

        # --- 5. 核心戒备等级裁定 (CHIPCON Level Adjudication) ---
        # (原 _diagnose_chip_alert_conditions 的逻辑)
        is_highly_concentrated = states.get('CHIP_STATE_HIGHLY_CONCENTRATED', default_series)
        is_rapidly_concentrating = states.get('CHIP_RAPID_CONCENTRATION', default_series)
        is_cost_rising = df.get('SLOPE_5_peak_cost_D', default_series) > 0
        is_cost_stable = df.get('SLOPE_5_peak_cost_D', default_series).abs() < 0.01
        is_winner_rate_rising = df.get('SLOPE_5_total_winner_rate_D', default_series) > 0
        is_long_term_distributing = states.get('RISK_CONTEXT_LONG_TERM_DISTRIBUTION', default_series)

        # CHIPCON 4: 战备状态 (有潜力，值得关注)
        states['CHIPCON_4_READINESS'] = is_highly_concentrated & is_cost_stable & ~is_long_term_distributing
        # CHIPCON 3: 高度戒备 (即将发动进攻的迹象)
        states['CHIPCON_3_HIGH_ALERT'] = is_highly_concentrated & is_cost_rising & is_winner_rate_rising & ~is_long_term_distributing
        # CHIPCON 2: 战前状态 (进攻迹象非常明显)
        states['CHIPCON_2_PRE_WAR'] = states.get('CHIPCON_3_HIGH_ALERT', default_series) & is_rapidly_concentrating
        # CHIPCON 1: 战争状态 (已确认的、最危险的派发信号)
        states['CHIPCON_1_WAR'] = is_long_term_distributing & (df.get('SLOPE_5_total_winner_rate_D', default_series) < 0)

        print("        -> [筹码情报最高司令部 V256.0] 分析完毕。")
        return states, triggers

    def _diagnose_chip_price_action(self, df: pd.DataFrame, atomic_states: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V228.0 新增】筹码-价格行为联合分析部
        - 核心职责: 将价格的涨跌行为与筹码（尤其是主力资金）的流动进行交叉验证，
                    将单纯的价格行为，升维为包含“主力意图”的复合情报。
        - 作战原则: “无筹码，不决策”。
        """
        print("        -> [联合分析部 V228.0] 启动，正在对价格行为进行筹码深度解析...")
        cpa_states = {} # Chip-Price Action States
        default_series = pd.Series(False, index=df.index)

        # --- 1. 提取基础情报 ---
        # 价格行为情报
        is_price_rising = df.get('pct_change_D', default_series) > 0
        is_price_falling = df.get('pct_change_D', default_series) < 0

        # 主力资金动向情报 (来自筹码总参谋部)
        is_main_force_buying = atomic_states.get('CAPITAL_STRUCT_MAIN_FORCE_ACCUMULATING', default_series)
        is_main_force_selling = atomic_states.get('RISK_CAPITAL_STRUCT_MAIN_FORCE_DISTRIBUTING', default_series)
        
        # 散户资金动向情报
        # 假设散户与主力行为相反，或者直接使用散户数据（如果未来有）
        # 这里我们用主力行为的反面来代表散户的主要动向，简化模型
        is_retail_likely_buying = is_main_force_selling
        is_retail_likely_selling = is_main_force_buying

        # --- 2. 进行联合分析，生成高维复合情报 ---
        
        # 【A级进攻信号】上涨的“质”：主力支撑的上涨
        # 定义：价格上涨，同时主力资金在净流入。这是最健康的上涨模式。
        cpa_states['CPA_RISE_WITH_MAIN_FORCE_SUPPORT'] = is_price_rising & is_main_force_buying
        
        # 【C级风险信号】上涨的“危”：散户追高的上涨
        # 定义：价格上涨，但主力资金在净流出。这可能是拉高出货的危险信号。
        cpa_states['CPA_RISE_WITH_RETAIL_FOMO'] = is_price_rising & is_main_force_selling

        # 【S级风险信号】下跌的“质”：主力出逃的下跌
        # 定义：价格下跌，同时主力资金在净流出。这是最危险的下跌，趋势可能反转。
        cpa_states['CPA_FALL_WITH_MAIN_FORCE_FLEEING'] = is_price_falling & is_main_force_selling

        # 【S级机会信号】下跌的“机”：主力吸筹的下跌
        # 定义：价格下跌，但主力资金在净流入。这是经典的“黄金坑”或“洗盘吸筹”信号。
        cpa_states['CPA_FALL_WITH_MAIN_FORCE_ABSORBING'] = is_price_falling & is_main_force_buying

        print("        -> [联合分析部 V228.0] 深度解析完成。")
        return cpa_states


    # 动态惯性引擎 (Dynamic Momentum Engine)
    #    -> 核心职责: 分析趋势的速度与加速度。
    #    -> 指挥官: _diagnose_trend_dynamics()
    def _diagnose_trend_dynamics(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V174.0 动态惯性引擎】
        - 核心职责: 基于趋势的“斜率”和“加速度”，生成高维度的动态原子状态。
        - 产出: 返回一个包含 DYN_... 信号的字典，供评分引擎使用。
        """
        print("        -> [诊断模块 V174.0] 正在执行动态惯性诊断...")
        dynamics_states = {}
        default_series = pd.Series(False, index=df.index)

        # --- 1. 获取核心的斜率和加速度数据 ---
        # 长期趋势的速度和加速度
        long_slope_col = 'SLOPE_55_EMA_55_D'
        long_accel_col = 'ACCEL_55_EMA_55_D'
        # 短期趋势的速度
        short_slope_col = 'SLOPE_13_EMA_13_D'

        if not all(c in df.columns for c in [long_slope_col, long_accel_col, short_slope_col]):
            print("          -> [错误] 动态惯性诊断缺少必要的斜率/加速度列，跳过。")
            return {}

        long_slope = df[long_slope_col]
        long_accel = df[long_accel_col]
        short_slope = df[short_slope_col]

        # --- 2. 定义基础布尔条件 ---
        is_long_slope_positive = long_slope > 0
        is_long_slope_negative = long_slope < 0
        is_long_accel_positive = long_accel > 0
        is_long_accel_negative = long_accel < 0
        
        is_short_slope_positive = short_slope > 0
        is_short_slope_negative = short_slope < 0
        
        # --- 3. 组合生成高维度动态状态 ---
        # 【S级进攻】健康加速: 长短期趋势同向看涨，且长期趋势在加速
        dynamics_states['DYN_TREND_HEALTHY_ACCELERATING'] = is_long_slope_positive & is_short_slope_positive & is_long_accel_positive
        
        # 【A级进攻】成熟稳定: 长短期趋势同向看涨，但长期趋势已不再加速（减速或匀速）
        dynamics_states['DYN_TREND_MATURE_STABLE'] = is_long_slope_positive & is_short_slope_positive & ~is_long_accel_positive

        # 【B级进攻】底部反转: 长期趋势仍向下，但短期趋势已率先加速向上
        dynamics_states['DYN_TREND_BOTTOM_REVERSING'] = is_long_slope_negative & is_short_slope_positive & (df[short_slope_col] > df[short_slope_col].shift(1))

        # 【S级风险】动能衰减: 长期趋势虽向上，但已开始减速，且短期趋势已逆转
        dynamics_states['DYN_TREND_WEAKENING_DECELERATING'] = is_long_slope_positive & is_long_accel_negative & is_short_slope_negative

        # 【A级风险】下跌加速: 长短期趋势同向看跌，且下跌在加速
        dynamics_states['DYN_TREND_BEARISH_ACCELERATING'] = is_long_slope_negative & is_short_slope_negative & is_long_accel_negative

        # 【B级风险】顶部背离: 价格创近期新高，但长短期斜率均在下降
        is_new_high = df['high_D'] >= df['high_D'].shift(1).rolling(window=10).max()
        is_slope_weakening = (long_slope < long_slope.shift(1)) & (short_slope < short_slope.shift(1))
        dynamics_states['DYN_TREND_TOPPING_DIVERGENCE'] = is_new_high & is_slope_weakening

        # --- 4. 打印调试信息 ---
        # for name, series in dynamics_states.items():
        #     print(f"          -> “{name}” 已定义，激活 {series.sum()} 天。")
            
        return dynamics_states

    # 资本动向总参谋部 (Capital Flow Dept.)
    #    -> 核心职责: 统一分析宏观(CMF)与微观(主力)资金。
    #    -> 总参谋长: _diagnose_capital_states()
    def _diagnose_capital_states(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V219.0 情报一体化版】 - 统一资本动向总参谋部
        - 核心升级: 将原 `_diagnose_capital_structure_states` 的职能（主力/散户资金分析）
                    并入此模块，形成统一的、从宏观到微观的资本分析中心。
        - 作战单元1 (宏观气象站): 保留基于CMF的经典资本状态诊断。
        - 作战单元2 (精锐侦察连): 新增基于主力/散户净流入的、高精度的资本结构诊断。
        """
        print("        -> [诊断模块 V219.0 情报一体化版] 正在执行统一资本诊断...")
        states = {}
        default_series = pd.Series(False, index=df.index)
        
        # --- 作战单元1: 经典资本状态诊断 (基于CMF) ---
        capital_params = self._get_params_block('capital_state_params')
        if self._get_param_value(capital_params.get('enabled'), True):
            cmf_bullish_threshold = self._get_param_value(capital_params.get('cmf_bullish_threshold'), 0.05)
            states['CAPITAL_STATE_INFLOW_CONFIRMED'] = df.get('CMF_21_D', 0) > cmf_bullish_threshold
            
            divergence_context = capital_params.get('divergence_context', {})
            persistence_days = self._get_param_value(divergence_context.get('persistence_days'), 15)
            trend_ma_period = self._get_param_value(divergence_context.get('trend_ma_period'), 55)
            
            price_new_high = df['close_D'] > df['close_D'].shift(1).rolling(window=persistence_days).max()
            cmf_not_new_high = df['CMF_21_D'] < df['CMF_21_D'].shift(1).rolling(window=persistence_days).max()
            is_uptrend = df['close_D'] > df.get(f'EMA_{trend_ma_period}_D', 0)
            
            states['CAPITAL_STATE_DIVERGENCE_WINDOW'] = price_new_high & cmf_not_new_high & is_uptrend

        # --- 作战单元2: 【王牌】新型资本结构诊断 (基于主力/散户资金) ---
        main_force_col = 'main_force_net_inflow_amount_D'
        retail_col = 'retail_net_inflow_volume_D'
        # 检查情报是否送达
        if all(c in df.columns for c in [main_force_col, retail_col]):
            print("          -> [情报确认] 主力/散户资金数据已接收，开始结构分析...")
            # 1. 定义“主力正在吸筹”状态
            states['CAPITAL_STRUCT_MAIN_FORCE_ACCUMULATING'] = df[main_force_col] > 0
            # 2. 定义“主力正在派发”风险状态
            states['RISK_CAPITAL_STRUCT_MAIN_FORCE_DISTRIBUTING'] = df[main_force_col] < 0
            # 3. 定义“黄金坑”：主力吸筹 & 散户割肉
            states['CAPITAL_STRUCT_BULLISH_DIVERGENCE'] = (df[main_force_col] > 0) & (df[retail_col] < 0)
            # 4. 定义“死亡顶”：主力派发 & 散户接盘
            states['RISK_CAPITAL_STRUCT_BEARISH_DIVERGENCE'] = (df[main_force_col] < 0) & (df[retail_col] > 0)
        else:
            print(f"          -> [情报警告] 缺少高精度资金结构数据，跳过结构分析。")

        return states

    # 野战部队 (Field Forces - Trend Analysis)
    #    -> 核心职责: 分析均线趋势。
    #    -> 指挥官: _diagnose_ma_states()
    def _diagnose_ma_states(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V155.0 生命线校准版】
        - 核心修正: 采纳将军建议，将定义核心趋势状态的长期均线(long_ma)基准，
                    从过于迟钝的89日，全面校准为市场公认的55日“生命线”。
        """
        # print("        -> [诊断模块] 正在执行均线状态诊断...")
        states = {}
        default_series = pd.Series(False, index=df.index)
        p = self._get_params_block('ma_state_params')
        if not self._get_param_value(p.get('enabled'), False):
            print("          -> 均线诊断模块被禁用，跳过。")
            return states

        # --- 1. 核心趋势状态诊断 ---
        short_p = self._get_param_value(p.get('short_ma'), 13)
        mid_p = self._get_param_value(p.get('mid_ma'), 34)
        # 【核心修正】将长期均线基准从89日改为55日
        long_p = self._get_param_value(p.get('long_ma'), 55)
        short_ma, mid_ma, long_ma = f'EMA_{short_p}_D', f'EMA_{mid_p}_D', f'EMA_{long_p}_D'
        short_ma_slope_col = f'SLOPE_5_{short_ma}'

        if not all(c in df.columns for c in [short_ma, mid_ma, long_ma]):
            print(f"          -> [警告] 缺少核心均线列({short_ma}, {mid_ma}, {long_ma})，部分均线状态诊断跳过。")
            return states
        if short_ma_slope_col in df.columns:
            states['MA_STATE_SHORT_SLOPE_POSITIVE'] = (df[short_ma_slope_col] > 0)
        else:
            states['MA_STATE_SHORT_SLOPE_POSITIVE'] = default_series

        states['MA_STATE_PRICE_ABOVE_LONG_MA'] = df['close_D'] > df[long_ma]
        states['MA_STATE_STABLE_BULLISH'] = (df[short_ma] > df[mid_ma]) & (df[mid_ma] > df[long_ma])
        states['MA_STATE_SHORT_CROSS_MID'] = (df[short_ma] > df[mid_ma])
        states['MA_STATE_STABLE_BEARISH'] = (df[short_ma] < df[mid_ma]) & (df[mid_ma] < df[long_ma])
        states['MA_STATE_BOTTOM_PASSIVATION'] = states['MA_STATE_STABLE_BEARISH'] & (df['close_D'] > df[short_ma])

        # ... 其他逻辑保持不变 ...
        ma_spread = (df[short_ma] - df[long_ma]) / df[long_ma].replace(0, np.nan)
        ma_spread_zscore = (ma_spread - ma_spread.rolling(60).mean()) / ma_spread.rolling(60).std().replace(0, np.nan)
        states['MA_STATE_CONVERGING'] = ma_spread_zscore < self._get_param_value(p.get('converging_zscore'), -1.0)
        states['MA_STATE_DIVERGING'] = ma_spread_zscore > self._get_param_value(p.get('diverging_zscore'), 1.0)
        lookback_period = 10
        accel_d_col = f'ACCEL_{lookback_period}_{long_ma}'
        if accel_d_col in df.columns:
            states['MA_STATE_D_STABILIZING'] = (df[accel_d_col].shift(1).fillna(0) < 0) & (df[accel_d_col] >= 0)
        else:
            states['MA_STATE_D_STABILIZING'] = default_series
        simulated_w_ma_period = 105
        simulated_w_lookback = 25
        slope_w_simulated_col = f'SLOPE_{simulated_w_lookback}_EMA_{simulated_w_ma_period}_D'
        if slope_w_simulated_col in df.columns:
            states['MA_STATE_W_STABILIZING'] = (df[slope_w_simulated_col].shift(1).fillna(0) < 0) & (df[slope_w_simulated_col] >= 0)
        else:
            states['MA_STATE_W_STABILIZING'] = default_series
        key_support_mas = [21, 34, 55, 89, 144, 233]
        for ma_period in key_support_mas:
            ma_col = f'EMA_{ma_period}_D'
            if ma_col in df.columns:
                is_touching = df['low_D'] <= df[ma_col]
                is_closing_above = df['close_D'] >= df[ma_col]
                state_name = f'MA_STATE_TOUCHING_SUPPORT_{ma_period}'
                states[state_name] = is_touching & is_closing_above
            else:
                states[f'MA_STATE_TOUCHING_SUPPORT_{ma_period}'] = default_series
        p_duck = self._get_params_block('duck_neck_params')
        if self._get_param_value(p_duck.get('enabled'), True):
            duck_short_p = self._get_param_value(p_duck.get('short_ma'), 5)
            duck_mid_p = self._get_param_value(p_duck.get('mid_ma'), 10)
            duck_long_p = self._get_param_value(p_duck.get('long_ma'), 60)
            duck_short_ma, duck_mid_ma, duck_long_ma = f'EMA_{duck_short_p}_D', f'EMA_{duck_mid_p}_D', f'EMA_{duck_long_p}_D'
            if all(c in df.columns for c in [duck_short_ma, duck_mid_ma, duck_long_ma]):
                golden_cross_event = (df[duck_short_ma] > df[duck_mid_ma]) & (df[duck_short_ma].shift(1) <= df[duck_mid_ma].shift(1))
                break_condition = (df[duck_short_ma] < df[duck_mid_ma])
                persistence_days = self._get_param_value(p_duck.get('persistence_days'), 20)
                states['MA_STATE_DUCK_NECK_FORMING'] = self._create_persistent_state(
                    df, entry_event=golden_cross_event, persistence_days=persistence_days, break_condition=break_condition
                )
            else:
                states['MA_STATE_DUCK_NECK_FORMING'] = default_series

        # ▼▼▼ “整体趋势恶化”的定义 ▼▼▼
        # 1. 准备所需情报
        # 检查是否存在资本底背离的豁免窗口
        is_in_divergence_window = states.get('CAPITAL_STATE_DIVERGENCE_WINDOW', default_series)
        # 检查各项战略指标的斜率是否为负
        # a. 长期均线趋势（生命线）
        is_long_ma_slope_negative = df.get('SLOPE_55_EMA_55_D', 0) < 0
        # b. 短期均线趋势（攻击线）
        is_short_ma_slope_negative = df.get('SLOPE_13_EMA_13_D', 0) < 0
        # c. 长期均线加速度（趋势动能）
        is_long_ma_accel_negative = df.get('ACCEL_55_EMA_55_D', 0) < 0
        # d. 长期筹码成本趋势（主力成本）
        is_long_chip_slope_negative = df.get('peak_cost_slope_55d_D', 0) < 0
        # 2. 定义“无条件恶化”状态
        # 必须所有战略指标协同转弱，才是不可逆的恶化
        unconditional_deterioration = (
            is_long_ma_slope_negative & 
            is_short_ma_slope_negative & 
            is_long_ma_accel_negative & 
            is_long_chip_slope_negative
        )
        # 3. 最终裁决
        # 趋势恶化 = “无条件恶化” 且 “不在资本底背离的豁免期内”
        states['CONTEXT_TREND_DETERIORATING'] = unconditional_deterioration & ~is_in_divergence_window

        for key in states:
            if states[key] is None:
                states[key] = default_series
            else:
                states[key] = states[key].fillna(False)
        return states

    # ─> 工兵部队 (Engineer Corps - Structure Analysis)
    #    -> 核心职责: 识别箱体、平台等静态结构。
    #    -> 指挥官: _diagnose_box_states()
    def _diagnose_box_states(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V147.0 实战化地形识别版】
        - 核心重构: 彻底废弃了过于学术化、无法识别平坦整理区的 find_peaks 算法。
        - 新逻辑:
          采用更贴近实战的“振幅比率”法。只要股价在过去N天内的振幅
          (rolling_high - rolling_low) / rolling_low 小于一个阈值(如5%)，
          就认为形成了一个有效的“战术平台”或“箱体”。
        - 收益: 能够精准识别各种形态的盘整区，特别是像06-25至06-27日那种
                极其平坦的“空中加油”平台，从根本上解决了因此类平台无法被识别
                而错失突破信号的问题。
        """
        print("        -> [诊断模块 V147.0] 正在执行箱体状态诊断(实战化地形识别版)...")
        states = {}
        box_params = self._get_params_block('dynamic_box_params')
        if not self._get_param_value(box_params.get('enabled'), False) or df.empty:
            print("          -> 箱体诊断模块被禁用或数据为空，跳过。")
            return states

        # --- 新逻辑参数 ---
        lookback_window = self._get_param_value(box_params.get('lookback_window'), 8)
        max_amplitude_ratio = self._get_param_value(box_params.get('max_amplitude_ratio'), 0.05) # 振幅小于5%

        # --- 步骤1: 计算滚动窗口内的高点、低点和振幅 ---
        rolling_high = df['high_D'].rolling(window=lookback_window).max()
        rolling_low = df['low_D'].rolling(window=lookback_window).min()
        
        # 计算振幅比率，分母使用rolling_low避免除以0，且更符合涨跌幅定义
        amplitude_ratio = (rolling_high - rolling_low) / rolling_low.replace(0, np.nan)

        # --- 步骤2: 识别有效的“箱体”/“平台” ---
        # 只要近期振幅足够小，就认为是一个有效的箱体
        is_valid_box = (amplitude_ratio < max_amplitude_ratio).fillna(False)
        
        # 箱体的顶部和底部就是滚动窗口的高点和低点
        box_top = rolling_high
        box_bottom = rolling_low

        # --- 步骤3: 定义突破与跌破事件 (逻辑不变，但基于新的箱体定义) ---
        was_below_top = df['close_D'].shift(1) <= box_top.shift(1)
        is_above_top = df['close_D'] > box_top
        states['BOX_EVENT_BREAKOUT'] = is_valid_box & is_above_top & was_below_top
        
        was_above_bottom = df['close_D'].shift(1) >= box_bottom.shift(1)
        is_below_bottom = df['close_D'] < box_bottom
        states['BOX_EVENT_BREAKDOWN'] = is_valid_box & is_below_bottom & was_above_bottom
        
        # --- 步骤4: 诊断“健康箱体盘整”状态 (逻辑不变) ---
        ma_params = self._get_params_block('ma_state_params')
        mid_ma_period = self._get_param_value(ma_params.get('mid_ma'), 55)
        mid_ma_col = f"EMA_{mid_ma_period}_D"
        
        is_in_box = (df['close_D'] < box_top) & (df['close_D'] > box_bottom)
        if mid_ma_col in df.columns:
            box_midpoint = (box_top + box_bottom) / 2
            is_box_above_ma = box_midpoint > df[mid_ma_col]
            states['BOX_STATE_HEALTHY_CONSOLIDATION'] = is_valid_box & is_in_box & is_box_above_ma
        else:
            # 如果没有参考均线，则只要在箱体内就算
            states['BOX_STATE_HEALTHY_CONSOLIDATION'] = is_valid_box & is_in_box
        
        # --- 步骤5: 清理与返回 ---
        for key in states:
            if key not in states or states[key] is None:
                states[key] = pd.Series(False, index=df.index)
            else:
                states[key] = states[key].fillna(False)
        
        return states

    #    -> (下辖): 平台引力侦察模块: _diagnose_platform_states()
    def _diagnose_platform_states(self, df: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        【V129.2 健壮部署版 - 筹码平台诊断模块】
        - 核心修复: 修正了对筹码和价格列的引用，不再使用错误的 'CHIP_' 前缀，
                    确保能够正确从数据层获取 'peak_cost_D' 和 'close_D'。
        - 功能增强: 增加了更详细的日志输出和更强的防御性编程，确保在缺少数据时
                    能够优雅地处理并返回标准化的空结果，防止下游模块出错。
        """
        print("        -> [诊断模块 V129.2] 正在执行筹码平台状态诊断...")
        states = {}
        default_series = pd.Series(False, index=df.index)

        # --- 步骤1: 检查核心数据是否存在 ---
        peak_cost_col = 'peak_cost_D'
        close_col = 'close_D'
        long_ma_col = 'EMA_55_D' # 平台必须位于长期均线上方才有意义
        
        required_cols = [peak_cost_col, close_col, long_ma_col]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"          -> [警告] 缺少诊断平台状态所需的核心列: {missing}。模块将返回空结果。")
            # 即使失败，也要确保返回标准化的输出结构，防止下游模块调用失败
            df['PLATFORM_PRICE_STABLE'] = np.nan
            states['PLATFORM_STATE_STABLE_FORMED'] = default_series
            states['PLATFORM_FAILURE'] = default_series
            return df, states

        # --- 步骤2: 定义并计算“稳固平台形成”状态 ---
        # 条件A: 筹码峰成本在短期内高度稳定 (滚动5日的标准差/均值 < 2%)
        is_cost_stable = (df[peak_cost_col].rolling(5).std() / df[peak_cost_col].rolling(5).mean()) < 0.02
        
        # 条件B: 当前价格位于长期趋势均线之上，确保平台处于上升趋势中
        is_above_long_ma = df[close_col] > df[long_ma_col]
        
        # 组合成最终的“稳固平台形成”状态
        stable_formed_series = is_cost_stable & is_above_long_ma
        states['PLATFORM_STATE_STABLE_FORMED'] = stable_formed_series
        
        # --- 步骤3: 将有效的平台价格记录下来，供后续模块使用 ---
        # 只有在平台形成当天，才记录下当天的平台价格，否则为NaN
        df['PLATFORM_PRICE_STABLE'] = df[peak_cost_col].where(stable_formed_series)
        
        # --- 步骤4: 定义并计算“平台破位”风险 ---
        # 条件A: 昨日处于稳固平台之上
        was_on_platform = stable_formed_series.shift(1).fillna(False)
        
        # 条件B: 今日收盘价跌破了昨日的平台价格
        # 使用 ffill() 填充平台价格，以处理平台形成后、破位前的那些天
        stable_platform_price_series = df['PLATFORM_PRICE_STABLE'].ffill()
        is_breaking_down = df[close_col] < stable_platform_price_series.shift(1)
        
        # 组合成最终的“平台破位”风险信号
        platform_failure_series = was_on_platform & is_breaking_down
        states['PLATFORM_FAILURE'] = platform_failure_series

        # --- 步骤5: 打印诊断日志 ---
        print(f"          -> '稳固平台形成' 状态诊断完成，共激活 {stable_formed_series.sum()} 天。")
        print(f"          -> '平台破位' 风险诊断完成，共激活 {platform_failure_series.sum()} 天。")

        return df, states


    # ─> 基础侦察部队 (Basic Reconnaissance Units)
    #    -> 核心职责: 识别最底层的原子形态。
    #    ├─> K线形态侦察连: _diagnose_kline_patterns()
    def _diagnose_kline_patterns(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V60.3 状态持久化修复版】
        """
        # print("        -> [诊断模块] 正在执行K线组合形态诊断...")
        states = {}
        p = self.kline_params
        if not self._get_param_value(p.get('enabled'), False):
            return states
        p_washout = p.get('washout_params', {})
        if self._get_param_value(p_washout.get('enabled'), True):
            lookback = self._get_param_value(p_washout.get('lookback'), 60)
            pct_thresh = self._get_param_value(p_washout.get('pct_change_threshold'), -0.05)
            vol_multiplier = self._get_param_value(p_washout.get('volume_multiplier'), 1.5)
            window_days = self._get_param_value(p_washout.get('window_days'), 15)
            is_large_decline = df['pct_change_D'] <= pct_thresh
            is_high_volume = df['volume_D'] > df['VOL_MA_21_D'] * vol_multiplier
            is_near_low = df['close_D'] <= df['low_D'].rolling(window=lookback).min() * 1.05
            washout_event = is_large_decline & is_high_volume & is_near_low
            states['KLINE_EVENT_WASHOUT'] = washout_event
            
            # signal = states['KLINE_EVENT_WASHOUT']
            # dates_str = self._format_debug_dates(signal)
            # print(f"          -> '巨阴洗盘' 事件诊断完成，发现 {signal.sum()} 天。{dates_str}")
            
            counter = pd.Series(0, index=df.index)
            counter[washout_event] = window_days
            counter = counter.replace(0, np.nan).ffill().fillna(0)
            days_in_window = counter.groupby(washout_event.cumsum()).cumcount()
            states['KLINE_STATE_WASHOUT_WINDOW'] = (days_in_window < window_days) & (counter > 0)
            
            # signal = states['KLINE_STATE_WASHOUT_WINDOW']
            # dates_str = self._format_debug_dates(signal)
            # print(f"          -> '巨阴反转观察窗口' 持久化状态诊断完成，共激活 {signal.sum()} 天。{dates_str}")
            
        p_gap = p.get('gap_support_params', {})
        if self._get_param_value(p_gap.get('enabled'), True):
            gap_up_event = df['low_D'] > df['high_D'].shift(1)
            gap_support_level = df['high_D'].shift(1)
            break_condition = df['low_D'] <= gap_support_level
            persistence_days = self._get_param_value(p_gap.get('persistence_days'), 10)
            states['KLINE_STATE_GAP_SUPPORT_ACTIVE'] = self._create_persistent_state(
                df, entry_event=gap_up_event, persistence_days=persistence_days, break_condition=break_condition
            )
            
            # signal = states['KLINE_STATE_GAP_SUPPORT_ACTIVE']
            # dates_str = self._format_debug_dates(signal)
            # print(f"          -> '缺口支撑有效' 状态诊断完成，共激活 {signal.sum()} 天。{dates_str}")
            
        p_nshape = p.get('n_shape_params', {})
        if self._get_param_value(p_nshape.get('enabled'), True):
            rally_threshold = self._get_param_value(p_nshape.get('rally_threshold'), 0.097)
            consolidation_days_max = self._get_param_value(p_nshape.get('consolidation_days_max'), 3)
            n_shape_start_event = df['pct_change_D'] >= rally_threshold
            rally_open = df['open_D'].where(n_shape_start_event, np.nan)
            rally_high = df['high_D'].where(n_shape_start_event, np.nan)
            rally_open_ffill = rally_open.ffill()
            rally_high_ffill = rally_high.ffill()
            is_after_rally = n_shape_start_event.cumsum() > 0
            is_holding_above_open = df['close_D'] > rally_open_ffill
            is_not_new_rally = ~n_shape_start_event
            potential_consolidation = is_after_rally & is_holding_above_open & is_not_new_rally
            days_since_rally = potential_consolidation.groupby(n_shape_start_event.cumsum()).cumcount() + 1
            states['KLINE_STATE_N_SHAPE_CONSOLIDATION'] = potential_consolidation & (days_since_rally <= consolidation_days_max)
            states['KLINE_N_SHAPE_RALLY_HIGH'] = rally_high_ffill.where(states['KLINE_STATE_N_SHAPE_CONSOLIDATION'])
            
            # signal = states['KLINE_STATE_N_SHAPE_CONSOLIDATION']
            # dates_str = self._format_debug_dates(signal)
            # print(f"          -> 'N字形态整理期' 状态诊断完成，共激活 {signal.sum()} 天。{dates_str}")
            
        # print("        -> [诊断模块] K线组合形态诊断执行完毕。")
        return states

    #    └─> 盘面模式侦察连: _diagnose_board_patterns()
    def _diagnose_board_patterns(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V58.0 诊断模块 - 板形态诊断引擎】
        """
        # print("        -> [诊断模块] 正在执行板形态诊断...")
        states = {}
        p = self._get_params_block('board_pattern_params')
        if not self._get_param_value(p.get('enabled'), False):
            return states
        prev_close = df['close_D'].shift(1)
        limit_up_threshold = self._get_param_value(p.get('limit_up_threshold'), 0.098)
        limit_down_threshold = self._get_param_value(p.get('limit_down_threshold'), -0.098)
        price_buffer = self._get_param_value(p.get('price_buffer'), 0.005)
        limit_up_price = prev_close * (1 + limit_up_threshold)
        limit_down_price = prev_close * (1 + limit_down_threshold)
        is_limit_up_close = df['close_D'] >= limit_up_price * (1 - price_buffer)
        is_limit_up_high = df['high_D'] >= limit_up_price * (1 - price_buffer)
        is_limit_down_low = df['low_D'] <= limit_down_price * (1 + price_buffer)
        states['BOARD_EVENT_EARTH_HEAVEN'] = is_limit_down_low & is_limit_up_close
        
        # signal = states['BOARD_EVENT_EARTH_HEAVEN']
        # dates_str = self._format_debug_dates(signal)
        # print(f"          -> '地天板' 事件诊断完成，发现 {signal.sum()} 天。{dates_str}")
        
        is_limit_down_close = df['close_D'] <= limit_down_price * (1 + price_buffer)
        states['BOARD_EVENT_HEAVEN_EARTH'] = is_limit_up_high & is_limit_down_close
        
        # signal = states['BOARD_EVENT_HEAVEN_EARTH']
        # dates_str = self._format_debug_dates(signal)
        # print(f"          -> '天地板' 事件诊断完成，发现 {signal.sum()} 天。{dates_str}")
        
        return states

    # └─> 联合作战司令部 (Joint Operations Command)
    #    -> 核心职责: 融合多源情报，形成复合战略判断。
    #    ├─> 市场结构分析室: _diagnose_market_structure_states()
    def _diagnose_market_structure_states(self, df: pd.DataFrame, params: dict, atomic_states: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V232.0 A股实战版 - 联合作战司令部】
        - 核心职责: 作为“战局大脑”，融合各部门的原子情报，识别出A股市场最经典的五大战局形态。
        - 作战思想: 聚焦于“趋势、筹码、资金”三位一体的共振与背离，为总司令部提供最高级别的战略洞察。
        """
        print("        -> [联合作战司令部 V232.0 A股实战版] 启动，正在识别核心战局...")
        structure_states = {}
        default_series = pd.Series(False, index=df.index)

        # --- 步骤1：情报总览 (从各部门接收原子情报) ---
        # [趋势部] 均线状态
        is_ma_bullish = atomic_states.get('MA_STATE_STABLE_BULLISH', default_series)
        is_ma_bearish = atomic_states.get('MA_STATE_STABLE_BEARISH', default_series)
        is_ma_converging = atomic_states.get('MA_STATE_CONVERGING', default_series)
        is_price_above_long_ma = atomic_states.get('MA_STATE_PRICE_ABOVE_LONG_MA', default_series)
        is_recent_reversal = atomic_states.get('CONTEXT_RECENT_REVERSAL_SIGNAL', default_series)
        is_ma_short_slope_positive = atomic_states.get('MA_STATE_SHORT_SLOPE_POSITIVE', default_series)

        # [动态部] 趋势动态
        is_dyn_trend_healthy = atomic_states.get('DYN_TREND_HEALTHY_ACCELERATING', default_series)
        is_dyn_trend_weakening = atomic_states.get('DYN_TREND_WEAKENING_DECELERATING', default_series)

        # [筹码部] 新旧筹码情报
        is_chip_concentrating = atomic_states.get('CHIP_RAPID_CONCENTRATION', default_series)
        is_chip_health_good = atomic_states.get('CHIP_HEALTH_GOOD', default_series)
        is_chip_health_excellent = atomic_states.get('CHIP_HEALTH_EXCELLENT', default_series)
        is_chip_health_deteriorating = atomic_states.get('CHIP_HEALTH_DETERIORATING', default_series)

        # [资金部] 资金流情报
        is_fund_flow_consensus_inflow = atomic_states.get('CHIP_FUND_FLOW_CONSENSUS_INFLOW', default_series)
        is_fund_flow_consensus_outflow = atomic_states.get('CHIP_FUND_FLOW_CONSENSUS_OUTFLOW', default_series)
        is_capital_bearish_divergence = atomic_states.get('RISK_CAPITAL_STRUCT_BEARISH_DIVERGENCE', default_series)

        # [波动率部] 波动状态
        is_vol_squeeze = atomic_states.get('VOL_STATE_SQUEEZE_WINDOW', default_series)

        # --- 步骤2：联合裁定 (识别五大经典战局) ---

        # 【战局1: S级主升浪·黄金航道】 - 所有力量的完美共振
        # 定义: 趋势健康加速 + 筹码结构良好 + 资金共识性流入
        structure_states['STRUCTURE_MAIN_UPTREND_WAVE_S'] = (
            is_dyn_trend_healthy &
            is_chip_health_good &
            is_fund_flow_consensus_inflow
        )

        # 【战局2: A级突破前夜·能量压缩】 - 大战前的寂静
        # 定义: 波动率极度压缩 + 筹码正在加速集中 + 均线系统收敛
        structure_states['STRUCTURE_BREAKOUT_EVE_A'] = (
            is_vol_squeeze &
            is_chip_concentrating &
            is_ma_converging
        )

        # 【战局3: B级反转初期·黎明微光】 - 从左侧到右侧的脆弱过渡
        # 定义: 近期出现过反转信号 + 短期均线开始走平或向上
        structure_states['STRUCTURE_EARLY_REVERSAL_B'] = (
            is_recent_reversal &
            is_ma_short_slope_positive
        )

        # 【战局4: S级风险·顶部背离】 - 最危险的诱多陷阱
        # 定义: 资金出现顶背离 或 筹码健康度持续恶化
        structure_states['STRUCTURE_TOPPING_DANGER_S'] = (
            is_capital_bearish_divergence |
            is_chip_health_deteriorating
        )

        # 【战局5: F级禁区·下跌通道】 - 绝对的回避区域
        # 定义: 均线空头排列 + 趋势动能衰减 + 资金共识性流出
        structure_states['STRUCTURE_BEARISH_CHANNEL_F'] = (
            is_ma_bearish &
            is_dyn_trend_weakening &
            is_fund_flow_consensus_outflow
        )

        print("        -> [联合作战司令部 V232.0] 核心战局识别完成。")
        return structure_states

    #    └─> 精英态势研判室: _diagnose_strategic_setups()
    def _run_cognitive_synthesis_engine(self, df: pd.DataFrame, atomic_states: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V257.0 一体化整编版】认知综合引擎 (Cognitive Synthesis Engine)
        - 核心重构: 将原有的四个独立认知层方法
                      (_diagnose_strategic_setups, _diagnose_cognitive_patterns,
                       _diagnose_battlefield_stability, _diagnose_price_action_context)
                      整合成一个统一的、负责将底层情报升华为高维战术概念的“中央处理器”。
        - 作战流程:
          1. 价格行为上下文分析 (Contextual Analysis): 首先定义当前价格行为的性质（如健康上涨、强力突破等）。
          2. 战场稳定性评估 (Stability Assessment): 评估战场的核心要素是否稳定。
          3. 战略布局识别 (Strategic Setup Recognition): 识别是否存在经典的战略布局机会。
          4. 最终认知模式形成 (Cognitive Pattern Formation): 在前三步的基础上，形成最终的、最高维度的认知模式，如“锁仓拉升”。
        - 收益: 建立了一个清晰、连贯的“认知链”，从上下文到稳定性，再到战略布局，最终形成顶层认知。
                极大地提升了代码的内聚性和逻辑的清晰度。
        """
        print("        -> [认知综合引擎 V257.0] 启动，正在进行高维认知合成...")
        cognitive_states = {}
        default_series = pd.Series(False, index=df.index)

        # --- 步骤1: 价格行为上下文分析 (Contextual Analysis) ---
        # (原 _diagnose_price_action_context 的逻辑)
        print("          -> [认知链 1/4] 正在分析价格行为上下文...")
        pct_change_col = 'pct_change_D'
        if pct_change_col in df.columns:
            # 健康上涨 (2% < 涨幅 <= 4%)
            cognitive_states['CONTEXT_HEALTHY_RALLY'] = (df[pct_change_col] > 0.02) & (df[pct_change_col] <= 0.04)
            # 强力突破 (4% < 涨幅 <= 7%)
            cognitive_states['CONTEXT_STRONG_BREAKOUT_RALLY'] = (df[pct_change_col] > 0.04) & (df[pct_change_col] <= 0.07)
            # 爆炸性拉升 (涨幅 > 7%)
            cognitive_states['CONTEXT_EXPLOSIVE_RALLY'] = df[pct_change_col] > 0.07
        
        # --- 步骤2: 战场稳定性评估 (Stability Assessment) ---
        # (原 _diagnose_battlefield_stability 的逻辑)
        print("          -> [认知链 2/4] 正在评估战场核心稳定性...")
        # 假设这些原子状态已由其他司令部提供
        is_cost_stable = atomic_states.get('CHIP_STATE_COST_STABLE', default_series)
        is_winner_rate_stable = atomic_states.get('CHIP_STATE_WINNER_RATE_STABLE', default_series)
        is_peak_stable = atomic_states.get('CHIP_STATE_PEAK_STABLE', default_series)
        cognitive_states['BATTLEFIELD_STABLE'] = is_cost_stable & is_winner_rate_stable & is_peak_stable

        # --- 步骤3: 战略布局识别 (Strategic Setup Recognition) ---
        # (原 _diagnose_strategic_setups 的逻辑)
        print("          -> [认知链 3/4] 正在识别高价值战略布局...")
        # 假设这些原子状态已由其他司令部提供
        is_highly_concentrated = atomic_states.get('CHIP_STATE_HIGHLY_CONCENTRATED', default_series)
        is_winner_rate_low = atomic_states.get('CHIP_STATE_LOW_PROFIT', default_series)
        is_cost_stable_or_rising = atomic_states.get('CHIP_STATE_COST_STABLE_OR_RISING', default_series)
        # “深度吸筹”布局：筹码高度集中 + 场内获利盘极少 + 成本稳定或抬高
        cognitive_states['SETUP_DEEP_ACCUMULATION'] = is_highly_concentrated & is_winner_rate_low & is_cost_stable_or_rising

        # --- 步骤4: 最终认知模式形成 (Cognitive Pattern Formation) ---
        # (原 _diagnose_cognitive_patterns 的逻辑)
        print("          -> [认知链 4/4] 正在形成最终顶层认知模式...")
        
        # 4.1 “锁仓拉升”进攻模式 (COGNITIVE_PATTERN_LOCK_CHIP_RALLY)
        # 情报来源:
        is_healthy_rally = cognitive_states.get('CONTEXT_HEALTHY_RALLY', default_series) # 来自本引擎步骤1
        is_cost_rising_fast = atomic_states.get('CHIP_STATE_COST_RISING_FAST', default_series)
        is_price_detached = atomic_states.get('CHIP_STATE_PRICE_DETACHED_FROM_COST', default_series)
        is_chip_truly_concentrating = atomic_states.get('CHIP_STATE_TRUE_CONCENTRATION', default_series)
        # 最终裁决:
        cognitive_states['COGNITIVE_PATTERN_LOCK_CHIP_RALLY'] = is_healthy_rally & is_cost_rising_fast & ~is_price_detached & is_chip_truly_concentrating

        # 4.2 “突破派发”风险模式 (COGNITIVE_RISK_BREAKOUT_DISTRIBUTION)
        # 情报来源:
        is_strong_rally = cognitive_states.get('CONTEXT_STRONG_BREAKOUT_RALLY', default_series) | cognitive_states.get('CONTEXT_EXPLOSIVE_RALLY', default_series) # 来自本引擎步骤1
        is_main_force_distributing = atomic_states.get('RISK_CAPITAL_STRUCT_MAIN_FORCE_DISTRIBUTING', default_series)
        is_chip_diverging = atomic_states.get('CHIP_STATE_DIVERGENCE', default_series)
        # 最终裁决:
        cognitive_states['COGNITIVE_RISK_BREAKOUT_DISTRIBUTION'] = is_strong_rally & is_main_force_distributing & is_chip_diverging

        print("        -> [认知综合引擎 V257.0] 认知合成完毕。")
        return cognitive_states

    # 2. 参谋部联席会议 (Joint Chiefs of Staff - Assessment & Scoring) 
    #     -> 核心职责: 对情报进行量化评估，计算进攻价值分与战场风险分。
    #     -> 总指挥: _run_scoring_and_assessment()
    def _run_scoring_and_assessment(self, df: pd.DataFrame, params: dict, atomic_states: Dict[str, pd.Series], trigger_events: Dict[str, pd.Series]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        【V233.0 三权分立版 - 参谋部联席会议】
        - 核心改革: 严格执行“进攻”与“风险”评估分离的原则。
        - 新流程: 1. 调用 _calculate_entry_score 计算纯粹的“进攻价值分”。
                   2. 调用 _calculate_risk_score 计算独立的“战场风险分”。
                   3. 将两个独立的分数写入DataFrame，供总司令部决策。
        """
        print("    - [参谋部 V233.0 三权分立版] 启动，正在进行攻防独立评估...")
        
        # 1. 进攻方案评估中心 -> 计算“进攻价值分”
        df, score_details_df = self._calculate_entry_score(df, params, atomic_states, trigger_events)
        
        # 2. 最高风险裁决所 -> 计算“战场风险分”
        df, risk_details_df = self._calculate_risk_score(df, params, atomic_states)

        print("    - [参谋部 V233.0] 攻防独立评估完成。")
        return df, score_details_df, risk_details_df

    # ─> 进攻方案评估中心 (Entry Scoring Center)
    #    -> 核心职责: 计算最终的入场分。
    #    -> 指挥官: _calculate_entry_score()
    def _calculate_entry_score(self, df: pd.DataFrame, params: dict, atomic_states: Dict[str, pd.Series], trigger_events: Dict[str, pd.Series]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        【V233.0 纯进攻版 - 进攻方案评估中心】
        - 核心改革: 此方法现在只负责计算“进攻价值分”，不再处理任何负面或惩罚信号。
        """
        scoring_params = self._get_params_block('four_layer_scoring_params')
        if not self._get_param_value(scoring_params.get('enabled'), True):
            df['entry_score'] = 0
            return df, pd.DataFrame(index=df.index)

        # --- 1. 阵地分与动能分计算 ---
        positional_rules = scoring_params.get('positional_scoring', {}).get('positive_signals', {})
        dynamic_rules = scoring_params.get('dynamic_scoring', {}).get('positive_signals', {})
        
        positional_score = pd.Series(0.0, index=df.index)
        dynamic_score = pd.Series(0.0, index=df.index)
        score_details = {}

        for state, score in positional_rules.items():
            if state in atomic_states:
                signal_series = atomic_states[state] * score
                positional_score += signal_series
                score_details[f"pos_{state}"] = signal_series

        for state, score in dynamic_rules.items():
            if state in atomic_states:
                signal_series = atomic_states[state] * score
                dynamic_score += signal_series
                score_details[f"dyn_{state}"] = signal_series
        
        # --- 2. 触发事件加成 ---
        trigger_score = pd.Series(0.0, index=df.index)
        trigger_rules = scoring_params.get('trigger_events', {})
        for event, score in trigger_rules.items():
            if event in trigger_events:
                event_series = trigger_events[event] * score
                trigger_score += event_series
                score_details[f"trg_{event}"] = event_series

        # --- 3. 计算最终进攻分 (不再有负分项) ---
        df['positional_score'] = positional_score
        df['dynamic_score'] = dynamic_score
        df['trigger_score'] = trigger_score
        df['entry_score'] = positional_score + dynamic_score + trigger_score
        
        score_details_df = pd.DataFrame(score_details)
        return df, score_details_df

    #       └─> 指挥棒模型 (Score Adjustment Module)
    #          -> 核心职责: 对基础分进行最终的乘数加成或削弱。
    #          -> 对应方法: _apply_final_score_adjustments()
    def _apply_final_score_adjustments(self, df: pd.DataFrame, raw_scores: pd.Series, params: dict, atomic_states: Dict[str, pd.Series]) -> Tuple[pd.Series, pd.DataFrame]:
        """
        【V119.4 自适应阈值版】最终得分调整层
        - 核心升级: 引入“自适应绝对阈值”机制。对于波动率这类指标，不再使用固定阈值，
                    而是根据其自身的长期中位数动态计算阈值（例如，低于长期中位数的50%）。
                    这使得策略能更好地适应不同股票和不同市场周期的波动特性。
        """
        print("    - [指挥棒模型 V119.4 自适应阈值版] 启动，开始对原始总分进行最终调整...")
        
        adjustment_params = self._get_params_block('final_score_adjustments')
        if not self._get_param_value(adjustment_params.get('enabled'), False):
            print("      -> 最终得分调整被禁用，返回原始分数。")
            return raw_scores, pd.DataFrame(index=df.index)

        total_multiplier = pd.Series(0.0, index=df.index)
        adjustment_details_df = pd.DataFrame(index=df.index)
        multiplier_rules = adjustment_params.get('multipliers', [])
        default_series = pd.Series(False, index=df.index)

        for rule in multiplier_rules:
            name = rule.get('name')
            cn_name = rule.get('cn_name', name)
            source_col = rule.get('source_column')
            condition = rule.get('condition')
            multiplier_value = rule.get('multiplier_value', 0)

            mask = pd.Series(False, index=df.index)
            if condition == 'is_lowest_percentile':
                if source_col not in df.columns:
                    print(f"      -> [警告] 指挥棒规则'{cn_name}'所需的列'{source_col}'在DataFrame中不存在，跳过。")
                    continue
                
                # 保险一：相对条件 (近期收缩)
                window = rule.get('window', 250)
                percentile = rule.get('percentile', 0.1)
                threshold_relative = df[source_col].rolling(window=window, min_periods=window//2).quantile(percentile)
                mask_relative = df[source_col] < threshold_relative
                
                # 保险二：绝对条件 (现在支持固定或自适应)
                mask_absolute = pd.Series(True, index=df.index) # 默认绝对条件为真
                
                # 检查是否有自适应阈值配置
                adaptive_config = rule.get('adaptive_threshold_config', {})
                if self._get_param_value(adaptive_config.get('enabled'), False):
                    median_window = self._get_param_value(adaptive_config.get('median_window'), 250)
                    multiplier = self._get_param_value(adaptive_config.get('multiplier'), 0.5)
                    
                    # 计算长期中位数作为波动中枢
                    long_term_median = df[source_col].rolling(window=median_window, min_periods=median_window//2).median()
                    # 计算自适应绝对阈值
                    threshold_absolute_adaptive = long_term_median * multiplier
                    
                    mask_absolute = df[source_col] < threshold_absolute_adaptive
                    print(f"      -> 规则'{cn_name}'启用自适应双重保险: 相对分位 & 自适应阈值(<长期中枢*{multiplier})")

                # 如果没有自适应配置，则检查固定阈值 (保持旧逻辑兼容性)
                elif rule.get('absolute_threshold') is not None:
                    absolute_threshold = rule.get('absolute_threshold')
                    mask_absolute = df[source_col] < absolute_threshold
                    print(f"      -> 规则'{cn_name}'启用固定双重保险: 相对分位 & 固定阈值(<{absolute_threshold})")

                # 最终的掩码是两个条件的交集
                mask = mask_relative & mask_absolute
            
            elif condition == 'is_true':
                if source_col not in atomic_states:
                    print(f"      -> [警告] 指挥棒规则'{cn_name}'所需的事件'{source_col}'在atomic_states中不存在，跳过。")
                    continue
                mask = atomic_states.get(source_col, default_series)

            # 应用乘数
            total_multiplier.loc[mask] += multiplier_value
            adjustment_details_df.loc[mask, name] = multiplier_value
            
            if not mask.empty and mask.iloc[-1]:
                print(f"      -> 指挥棒 '{cn_name}' (最新一日): [✓] (乘数: +{multiplier_value*100:.0f}%)")

        adjusted_score = raw_scores * (1 + total_multiplier)
        print("    - [指挥棒模型 V119.4] 调整完成。")
        return adjusted_score, adjustment_details_df

    #       ├─> 基础火力评估室: _calculate_base_scores()
    def _calculate_base_scores(self, df: pd.DataFrame, scoring_params: dict, atomic_states: dict, trigger_events: dict, score_details_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
        """
        【V210.0 作战序列重塑版】
        - 核心升级: 将“惩罚分”彻底剥离，作为一个独立的战斗序列(total_penalty_score)返回。
        - 作战意图: 确保惩罚分不再混入阵地分，从根本上杜绝其被后续的“周线豁免”等
                    逻辑意外抵消或削弱的可能，保证其绝对的、最终的否决权。
        """
        print("      -> [火力层1-3/4] 正在计算“阵地分”、“动能分”与“触发分”...")
        default_series = pd.Series(False, index=df.index)
        
        # 计算阵地分 (纯加分项)
        positional_params = scoring_params.get('positional_scoring', {})
        total_positional_score = pd.Series(0.0, index=df.index)
        for state_name, score in positional_params.get('positive_signals', {}).items():
            mask = atomic_states.get(state_name, default_series)
            total_positional_score.loc[mask] += score
            score_details_df.loc[mask, state_name] = score
            
        # 计算动能分 (纯加分项)
        dynamic_params = scoring_params.get('dynamic_scoring', {})
        total_dynamic_score = pd.Series(0.0, index=df.index)
        for state_name, score in dynamic_params.get('positive_signals', {}).items():
            mask = atomic_states.get(state_name, default_series)
            total_dynamic_score.loc[mask] += score
            score_details_df.loc[mask, state_name] = score

        # 计算触发分 (纯加分项)
        trigger_score_total = pd.Series(0.0, index=df.index)
        trigger_event_scores = scoring_params.get('trigger_events', {})
        for event_name, score in trigger_event_scores.items():
            if event_name == '说明': continue
            mask = trigger_events.get(event_name, default_series)
            trigger_score_total.loc[mask] += score
            score_details_df.loc[mask, event_name] = score

        # ▼▼▼ “认知层”火力计算 ▼▼▼
        print("      -> [王牌火力层] 正在计算“认知模式分”...")
        total_cognitive_score = pd.Series(0.0, index=df.index)
        cognitive_pattern_scores = scoring_params.get('cognitive_pattern_scoring', {})
        for pattern_name, score in cognitive_pattern_scores.items():
            if not pattern_name.startswith('COGNITIVE_PATTERN_'):
                continue
            mask = atomic_states.get(pattern_name, default_series)
            total_cognitive_score.loc[mask] += score
            score_details_df.loc[mask, pattern_name] = score
            if mask.any() and score > 0:
                 status_cognitive = "[✓]" if mask.iloc[-1] else "[✗]"
                 print(f"        -> 认知模式 '{pattern_name}' (最新一日): {status_cognitive} (分值: +{score})")
        
        print("      -> [惩罚层] 正在计算“惩罚分”...")
        total_penalty_score = pd.Series(0.0, index=df.index)
        negative_signals = {
            **positional_params.get('negative_signals', {}),
            **dynamic_params.get('negative_signals', {}),
            **scoring_params.get('penalty_signals', {})
        }
        for state_name, score in negative_signals.items():
            mask = atomic_states.get(state_name, default_series)
            total_penalty_score.loc[mask] += score 
            score_details_df.loc[mask, state_name] = score
            if mask.any() and score < 0:
                 status_penalty = "[✓]" if mask.iloc[-1] else "[✗]"
                 print(f"        -> 惩罚信号 '{state_name}' (最新一日): {status_penalty} (分值: {score})")

        # ▼▼▼ 返回独立的惩罚分，不再将其混入阵地分 ▼▼▼
        return total_positional_score, total_dynamic_score, trigger_score_total, total_cognitive_score, total_penalty_score, score_details_df

    #       ├─> 战略修正室: _apply_weekly_context_modifiers()
    def _apply_weekly_context_modifiers(self, df: pd.DataFrame, positional_score: pd.Series, dynamic_score: pd.Series, trigger_score: pd.Series, scoring_params: dict, atomic_states: dict, score_details_df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        【V212.0 幽灵净化版】
        - 核心重构: 彻底删除了此方法中所有关于“负分分离”和“惩罚豁免”的幽灵代码。
                    由于所有负分已在上游被剥离为独立的“惩罚分”，此模块的职责被
                    净化为：只对正面得分进行周线级别的加权和修正。
        - 作战意图: 彻底根除因代码重构不完全而导致的逻辑短路，确保指挥链的每
                    一个环节都权责清晰、逻辑正确，不再有任何“历史遗留问题”。
        """
        print("      -> [火力层3.5/4] 正在应用“周线精确制导”...")
        default_series = pd.Series(False, index=df.index)

        positional_multiplier = pd.Series(1.0, index=df.index)
        dynamic_multiplier = pd.Series(1.0, index=df.index)
        trigger_multiplier = pd.Series(1.0, index=df.index)
        
        # ▼▼▼ 简化周线指令，只保留对正面火力的加成 ▼▼▼
        main_ascent_mask = df.get('state_node_main_ascent_W', default_series)
        breakout_mask = df.get('playbook_classic_breakout_W', default_series)
        momentum_context_mask = main_ascent_mask | breakout_mask
        if momentum_context_mask.any():
            dynamic_multiplier.loc[momentum_context_mask] *= 1.5
            trigger_multiplier.loc[momentum_context_mask] *= 1.8
            score_details_df.loc[momentum_context_mask, 'CONTEXT_MULT_MOMENTUM'] = 1.5
        
        ignition_mask = df.get('state_node_ignition_W', default_series)
        bottoming_mask = df.get('state_node_bottoming_W', default_series)
        reversal_context_mask = ignition_mask | bottoming_mask
        if reversal_context_mask.any():
            positional_multiplier.loc[reversal_context_mask] *= 1.6
            score_details_df.loc[reversal_context_mask, 'CONTEXT_MULT_POSITIONAL'] = 1.6

        # 应用乘数 (现在直接应用于纯正分)
        adj_total_positional = positional_score * positional_multiplier
        adj_total_dynamic = dynamic_score * dynamic_multiplier
        adj_trigger = trigger_score * trigger_multiplier
        
        # 使用调整后的分数进行混合加权
        weights = scoring_params.get('hybrid_scoring_weights', {})
        weight_pos = weights.get('positional_weight', 0.4)
        weight_dyn = weights.get('dynamic_weight', 0.6)
        weighted_base_score = (adj_total_positional * weight_pos) + (adj_total_dynamic * weight_dyn)
        
        # 叠加上调整后的触发分，得到最终的修正后基础分
        modified_base_score = weighted_base_score + adj_trigger
        
        # ▼▼▼ 调整正面得分的计算方式 ▼▼▼
        # 使用调整后的分数进行混合加权
        weights = scoring_params.get('hybrid_scoring_weights', {})
        weight_pos = weights.get('positional_weight', 0.4)
        weight_dyn = weights.get('dynamic_weight', 0.6)
        weighted_base_score = (positional_score * weight_pos) + (dynamic_score * weight_dyn)
        
        # 叠加上调整后的触发分，再加上独立的、不受影响的认知分
        modified_base_score = weighted_base_score + trigger_score + cognitive_score
        
        return modified_base_score, score_details_df

    #       └─> 火力放大器: _apply_final_score_amplifier()
    def _apply_final_score_amplifier(self, df: pd.DataFrame, modified_base_score: pd.Series, scoring_params: dict, score_details_df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        【新增辅助方法】应用最终的“优势火力”放大器。
        """
        print("      -> [火力层4/4] 正在启动“优势火力”放大器...")
        amp_params = scoring_params.get('chip_dynamics_amplifier', {})
        if not amp_params.get('enabled', False):
            return modified_base_score, score_details_df

        window = self._get_param_value(amp_params.get('lookback_window'), 250)
        
        # 计算兵种1(筹码)的乘数
        conc_slope_col = 'concentration_90pct_slope_5d_D'
        conc_multiplier = pd.Series(1.0, index=df.index)
        if conc_slope_col in df.columns:
            conc_rank = df[conc_slope_col].rolling(window=window, min_periods=window//2).rank(pct=True)
            rules = amp_params.get('concentration_slope_rules', {}).get('tiers', [])
            for rule in sorted(rules, key=lambda x: x['percentile_upper'], reverse=True):
                conc_multiplier[conc_rank <= rule['percentile_upper']] = rule['multiplier']
        
        # 计算兵种2(成本)的乘数
        cost_slope_col = 'SLOPE_5_peak_cost_D'
        cost_multiplier = pd.Series(1.0, index=df.index)
        if cost_slope_col in df.columns:
            cost_rank = df[cost_slope_col].rolling(window=window, min_periods=window//2).rank(pct=True)
            rules = amp_params.get('cost_basis_slope_rules', {}).get('tiers', [])
            for rule in sorted(rules, key=lambda x: x['percentile_lower'], reverse=False):
                cost_multiplier[cost_rank >= rule['percentile_lower']] = rule['multiplier']
        
        # 采用“优势火力”逻辑，取最大值并应用安全限制
        final_multiplier = pd.concat([conc_multiplier, cost_multiplier], axis=1).max(axis=1)
        cap_params = amp_params.get('final_multiplier_cap', {})
        max_mult = cap_params.get('max', 3.0)
        min_mult = cap_params.get('min', 0.4)
        final_multiplier.clip(lower=min_mult, upper=max_mult, inplace=True)
        
        score_details_df['FINAL_MULTIPLIER'] = final_multiplier
        print(f"        -> 火力放大器已激活。最终乘数范围: [{final_multiplier.min():.2f}, {final_multiplier.max():.2f}]")
        
        final_score = modified_base_score * final_multiplier
        return final_score, score_details_df

    # ─> 最高风险裁决所 (Supreme Risk Adjudication)
    #    -> 核心职责: 对风险简报进行量化打分。
    #    -> 首席裁决官: _calculate_risk_score()
    def _calculate_risk_score(self, df: pd.DataFrame, params: dict, atomic_states: Dict[str, pd.Series]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        【V233.0 新增 - 最高风险裁决所】
        - 核心职责: 统一评估所有负面风险信号，计算独立的“战场风险分”。
        """
        scoring_params = self._get_params_block('four_layer_scoring_params')
        risk_rules = scoring_params.get('risk_scoring', {}).get('signals', {})
        
        risk_score = pd.Series(0.0, index=df.index)
        risk_details = {}

        for state, score in risk_rules.items():
            if state in atomic_states:
                # 注意：JSON中的分值是正数，代表风险的严重程度
                risk_series = atomic_states[state] * score
                risk_score += risk_series
                risk_details[f"risk_{state}"] = risk_series
        
        df['risk_score'] = risk_score
        risk_details_df = pd.DataFrame(risk_details)
        return df, risk_details_df

    # 3. 总司令部 (General Headquarters - Final Decision Making)
    #    -> 核心职责: 权衡利弊，下达最终的“进攻”、“撤退”或“否决”指令。
    #    -> 总司令: _make_final_decisions()
    def _make_final_decisions(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        【V223.0 重命名&重组】总司令部
        - 核心职责: 作为指挥链的最终环节，根据“入场分”和“风险分”做出最终决策。
        """
        # 决策1: 风险否决 (Veto) - 是否取消进攻计划？
        veto_params = self.risk_veto_params
        if self._get_param_value(veto_params.get('enabled'), False):
            ratio = self._get_param_value(veto_params.get('risk_tolerance_ratio'), 0.5)
            min_risk = self._get_param_value(veto_params.get('min_absolute_risk_for_veto'), 30)
            
            df['dynamic_veto_threshold'] = df['entry_score'] * ratio
            veto_mask = (df['risk_score'] > df['dynamic_veto_threshold']) & (df['risk_score'] >= min_risk)
            
            if veto_mask.any():
                 print("    -> [总司令部指令] 发现风险过高，部分进攻计划已被否决！")
            df.loc[veto_mask, 'entry_score'] = 0
        
        # 决策2: 生成最终入场信号 - 哪些部队可以进攻？
        score_threshold = self._get_param_value(self._get_params_block('entry_scoring_params').get('score_threshold'), 100)
        df['signal_entry'] = df['entry_score'] >= score_threshold
        
        # 决策3: 生成最终离场信号码 - 哪些部队需要撤退？
        df['exit_signal_code'] = self._calculate_exit_signals(df, params, df['risk_score'])
        
        return df

    #    └─> 离场指令部 (Exit Command)
    #       -> 核心职责: 根据风险分生成具体的撤退信号码。
    #        -> 指挥官: _calculate_exit_signals()
    def _calculate_exit_signals(self, df: pd.DataFrame, params: dict, risk_score: pd.Series) -> pd.Series:
        """
        【V202.11 权限重铸版】出场决策引擎
        - 核心修复: 彻底重构。此引擎现在只负责处理“真正的卖出信号”。
                    它会从配置中获取一个明确的“最低卖出风险分”，只有当风险分
                    超过此值时，才会开始匹配并分配 exit_code。
        """
        exit_strategy_params = self.exit_strategy_params
        threshold_params = exit_strategy_params.get('exit_threshold_params', {})
        
        # ▼▼▼ 设定“最低开火权限”！▼▼▼
        # 只有当风险分达到 exit_threshold_params 中定义的最低 level 时，才触发卖出。
        # 这为“风险预警”留出了明确的安全空间。
        if not threshold_params:
            return pd.Series(0, index=df.index)
        
        min_score_for_exit = min(self._get_param_value(config.get('level')) for config in threshold_params.values())
        
        # 制作一个“开火许可”面具
        fire_permission_mask = risk_score >= min_score_for_exit
        
        # 如果没有任何一天的风险达到最低卖出标准，则直接返回全0，不产生任何卖出信号
        if not fire_permission_mask.any():
            return pd.Series(0, index=df.index)

        levels = sorted(threshold_params.items(), key=lambda item: self._get_param_value(item[1].get('level')), reverse=True)
        
        conditions = []
        choices = []
        
        for level_name, config in levels:
            threshold = self._get_param_value(config.get('level'))
            exit_code = self._get_param_value(config.get('code'))
            # 条件现在必须同时满足：达到阈值 且 拥有“开火许可”
            conditions.append((risk_score >= threshold) & fire_permission_mask)
            choices.append(exit_code)
            
        exit_signal = np.select(conditions, choices, default=0)
        return pd.Series(exit_signal, index=df.index)


    # 4. 沙盘推演中心 (War Gaming Center - Simulation)
    #    -> 核心职责: 模拟从建仓到离场的全过程战术动作。
    #    -> 总指挥: _run_position_management_simulation()
    def _run_position_management_simulation(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        【V190.0 新增】战术持仓管理模拟引擎
        - 核心功能: 模拟一个完整的交易周期，包括入场、持仓、风险预警、动态减仓和最终离场。
        - 输出: 在DataFrame中增加仓位状态、预警级别、交易动作等列，用于最终分析。
        """
        print("\n" + "="*20 + " 【战术持仓管理模拟引擎 V190.0】启动 " + "="*20)
        
        # --- 1. 参数初始化 ---
        sim_params = self._get_params_block('position_management_params')
        if not self._get_param_value(sim_params.get('enabled'), False):
            print("    - 持仓管理模拟被禁用，跳过。")
            return df

        # 从配置中读取减仓比例
        level_2_reduction = self._get_param_value(sim_params.get('level_2_alert_reduction_pct'), 0.3)
        level_3_reduction = self._get_param_value(sim_params.get('level_3_alert_reduction_pct'), 0.5)
        
        # 初始化状态列
        df['position_size'] = 0.0
        df['alert_level'] = 0
        df['alert_reason'] = ''
        df['trade_action'] = ''

        # 初始化状态变量
        in_position = False
        position_size = 0.0
        entry_price = 0.0
        partial_exit_level_2_done = False # 标记L2减仓是否已执行，避免重复减仓

        # --- 2. 逐日迭代，模拟交易状态机 ---
        for row in df.itertuples():
            current_date = row.Index
            
            if not in_position:
                # --- 入场逻辑 ---
                if row.signal_entry:
                    in_position = True
                    position_size = 1.0
                    entry_price = row.close_D
                    partial_exit_level_2_done = False # 重置减仓标记
                    df.loc[current_date, 'trade_action'] = 'ENTRY'
            else:
                # --- 持仓期间的逻辑 ---
                
                # a. 检查硬性离场信号 (最高优先级)
                if row.exit_signal_code > 0:
                    in_position = False
                    position_size = 0.0
                    df.loc[current_date, 'trade_action'] = f'EXIT (Code:{row.exit_signal_code})'
                    df.loc[current_date, 'position_size'] = position_size
                    continue # 当天直接离场，不再执行后续预警判断

                # b. 检查战术预警信号
                alert_level, alert_reason = self._check_tactical_alerts(row, params)
                df.loc[current_date, 'alert_level'] = alert_level
                df.loc[current_date, 'alert_reason'] = alert_reason

                # c. 根据预警级别执行仓位调整
                if alert_level == 3: # 红色预警
                    if position_size > 0:
                        reduction_amount = position_size * level_3_reduction
                        position_size -= reduction_amount
                        df.loc[current_date, 'trade_action'] = f'REDUCE_L3 ({level_3_reduction:.0%})'
                
                elif alert_level == 2 and not partial_exit_level_2_done: # 橙色预警，且尚未执行过
                    if position_size > 0:
                        reduction_amount = position_size * level_2_reduction
                        position_size -= reduction_amount
                        df.loc[current_date, 'trade_action'] = f'REDUCE_L2 ({level_2_reduction:.0%})'
                        partial_exit_level_2_done = True # 标记已执行
                
                # d. 如果没有交易动作，则标记为持仓
                if df.loc[current_date, 'trade_action'] == '':
                    df.loc[current_date, 'trade_action'] = 'HOLD'

            # 更新每日的最终仓位
            df.loc[current_date, 'position_size'] = position_size
        
        print("="*25 + " 【持仓管理模拟】执行完毕 " + "="*25 + "\n")
        return df# ▼▼▼【代码新增 V190.0】: 新增独立的“战术预警”诊断模块 ▼▼▼

    def _check_tactical_alerts(self, row: pd.Series, params: dict) -> Tuple[int, str]:
        """
        【V243.0 统一风险视图版】战术预警中心
        - 核心修复: 彻底根治了风险评估体系的“精神分裂症”。
        - 新规则:
          1. 同时读取 "warning_threshold_params" (预警) 和 "exit_threshold_params" (卖出) 两个配置块。
          2. 将所有风险等级（从低到高）合并到一个统一的列表中。
          3. 根据当日的 risk_score，对照这个完整的风险光谱，返回唯一、正确的预警级别。
        - 收益: 确保了沙盘推演模块能获取到与总司令部一致的、完整的风险认知，
                从根本上杜绝了“幽灵警报”和逻辑矛盾。
        """
        exit_params = self._get_params_block('exit_strategy_params')
        warning_params = exit_params.get('warning_threshold_params', {})
        exit_threshold_params = exit_params.get('exit_threshold_params', {})

        # 如果没有任何风险配置，则不发出任何预警
        if not warning_params and not exit_threshold_params:
            return 0, ''

        # 获取当日的风险分
        risk_score = getattr(row, 'risk_score', 0)
        if risk_score <= 0:
            return 0, ''

        # --- 建立统一的、全谱系的风险等级清单 ---
        all_alerts = []
        # 定义从配置名到数字等级的映射
        level_map = {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}

        # 1. 加载“卖出”级别
        for name, config in exit_threshold_params.items():
            level_name_upper = name.upper()
            if level_name_upper in level_map:
                all_alerts.append({
                    'level_code': level_map[level_name_upper],
                    'threshold': self._get_param_value(config.get('level'), float('inf')),
                    'reason': self._get_param_value(config.get('cn_name'), name)
                })

        # 2. 加载“预警”级别
        for name, config in warning_params.items():
            level_name_upper = name.upper()
            if level_name_upper in level_map:
                all_alerts.append({
                    'level_code': level_map[level_name_upper],
                    'threshold': self._get_param_value(config.get('level'), float('inf')),
                    'reason': self._get_param_value(config.get('cn_name'), name)
                })
        
        # --- 按风险阈值从高到低排序，确保优先匹配最严重的风险 ---
        sorted_alerts = sorted(all_alerts, key=lambda x: x['threshold'], reverse=True)

        # --- 对照全谱系清单，进行最终裁决 ---
        for alert in sorted_alerts:
            if risk_score >= alert['threshold']:
                # 一旦触发最高级别的预警，立即返回
                return alert['level_code'], alert['reason']

        # 如果风险分未达到任何预警阈值，则返回无预警
        return 0, ''

    #  5. 后勤与战报总署 (Logistics & Reporting General Administration)
    #     -> 核心职责: 负责所有战后数据的处理、格式化、记录以及特殊警报的生成。
    #  ─> 战报司令部 (Battle Report Command)
    #    -> 核心职责: 生成标准化的战报记录，供数据库存档。
    #    -> 总指挥: prepare_db_records()
    def prepare_db_records(self, stock_code: str, result_df: pd.DataFrame, params: dict, result_timeframe: str = 'D') -> List[Dict[str, Any]]:
        """
        【V202.16 健壮性修复版】统一战报司令部
        - 核心修复: 解决了 `np.select` 在退出规则配置为空时崩溃的问题。
                    在计算 `exit_severity_level` 之前，增加了一个保护性检查。
                    如果从配置中未找到任何有效的退出规则，则安全地将严重级别设置为0，而不是尝试执行会导致错误的 `np.select`。
        - 业务逻辑: 保持不变。
        """
        required_cols = ['signal_entry', 'exit_signal_code', f'close_{result_timeframe}', 'risk_score']
        if not all(col in result_df.columns for col in required_cols):
            print(f"      -> [错误] prepare_db_records 缺少必要列: {required_cols}")
            return []
        
        df = result_df[(result_df['signal_entry'] == True) | (result_df['risk_score'] > 0)].copy()
        if df.empty: return []

        # --- 准备工作 ---
        strategy_info = self._get_params_block('strategy_info')
        strategy_name = self._get_param_value(strategy_info.get('name'), 'unknown_strategy')
        playbook_map = {p['name']: p.get('cn_name', p['name']) for p in self.playbook_blueprints}
        risk_playbook_map = {p['name']: p.get('cn_name', p['name']) for p in self.risk_playbook_blueprints}
        exit_cfg = params.get('exit_strategy_params', {})
        warning_thresh = exit_cfg.get('warning_threshold_params', {})
        min_warning_level = min([cfg['level'] for cfg in warning_thresh.values() if isinstance(cfg, dict)]) if warning_thresh else 30
        exit_thresh = exit_cfg.get('exit_threshold_params', {})

        # --- 向量化计算 ---
        is_exit = df['exit_signal_code'] > 0
        is_entry = df['signal_entry']
        is_warning = (df['risk_score'] >= min_warning_level) & ~is_exit & ~is_entry
        
        df['stock_code'] = stock_code
        df['timeframe'] = result_timeframe
        df['strategy_name'] = strategy_name
        df['trade_time'] = df.index
        df['close_price'] = df[f'close_{result_timeframe}'].apply(lambda x: float(x) if pd.notna(x) else None)

        df['entry_signal'] = is_entry
        df['is_risk_warning'] = is_warning
        
        df['stable_platform_price'] = np.where(
            is_entry, df['PLATFORM_PRICE_STABLE'].apply(lambda x: float(x) if pd.notna(x) else None), None
        )

        # 步骤1: 过滤出有效的退出规则，排除"说明"等非字典项，确保迭代安全
        valid_exit_rules = {name: cfg for name, cfg in exit_thresh.items() if isinstance(cfg, dict) and 'code' in cfg}
        
        # 步骤2: 增加保护，仅在存在有效规则时才执行np.select
        if valid_exit_rules:
            print(f"    -> 检测到 {len(valid_exit_rules)} 条有效退出规则，正在计算严重等级...")
            conditions = [df['exit_signal_code'] == self._get_param_value(rule.get('code')) for name, rule in valid_exit_rules.items()]
            choices = [3 if name == "CRITICAL" else 2 for name in valid_exit_rules.keys()]
            df['exit_severity_level'] = np.select(conditions, choices, default=0)
        else:
            # 如果没有定义任何退出规则，则将严重级别安全地设置为0
            print("    -> 未检测到有效退出规则，所有信号的严重等级将设为0。")
            df['exit_severity_level'] = 0

        def get_playbooks(timestamp, is_entry_signal):
            df_details = self._last_score_details_df if is_entry_signal else self._last_risk_details_df
            if df_details is None or timestamp not in df_details.index: return [], []
            
            details_row = df_details.loc[timestamp]
            # 注意：这里我们取所有大于0的项，因为认知模式分也是正分
            active_items = details_row[details_row > 0].index
            
            playbooks_en = []
            playbooks_cn = []

            if is_entry_signal:
                # 1. 提取战术剧本 (Playbooks)
                excluded = ('BASE_', 'BONUS_', 'PENALTY_', 'INDUSTRY_', 'WATCHING_SETUP', 'CHIP_PURITY_MULTIPLIER', 'VOLATILITY_SILENCE_MULTIPLIER', 'RISK_SCORE_', 'COGNITIVE_PATTERN_')
                tactical_playbooks_en = [item for item in active_items if not item.startswith(excluded)]
                tactical_playbooks_cn = [playbook_map.get(p, p) for p in tactical_playbooks_en]
                
                # 2. 提取认知模式 (Cognitive Patterns)
                cognitive_patterns_en = [item for item in active_items if item.startswith('COGNITIVE_PATTERN_')]
                # (未来可以为认知模式建立独立的中文名映射)
                cognitive_patterns_cn = [item.replace('COGNITIVE_PATTERN_', '认知模式:') for item in cognitive_patterns_en]

                # 3. 合并，并将认知模式放在最前，以彰显其重要性
                playbooks_en = cognitive_patterns_en + tactical_playbooks_en
                playbooks_cn = cognitive_patterns_cn + tactical_playbooks_cn
            else: # 风险信号逻辑不变
                playbooks_en = [item.replace('RISK_SCORE_', '') for item in active_items]
                playbooks_cn = [risk_playbook_map.get(p, p) for p in playbooks_en]

            return playbooks_en, playbooks_cn

        playbook_results = df.apply(lambda row: get_playbooks(row.name, row['signal_entry']), axis=1)
        df[['triggered_playbooks', 'triggered_playbooks_cn']] = pd.DataFrame(playbook_results.tolist(), index=df.index)

        def build_final_details(row):
            reason = self._quantify_risk_reasons(row)            
            playbook_details = {}
            if self._last_score_details_df is not None and row.name in self._last_score_details_df.index:
                score_row = self._last_score_details_df.loc[row.name].dropna()
                playbook_details['score_components'] = score_row.to_dict()

            if self._last_risk_details_df is not None and row.name in self._last_risk_details_df.index:
                risk_row = self._last_risk_details_df.loc[row.name].dropna()
                playbook_details['risk_components'] = risk_row.to_dict()
            
            return {'reason': reason, 'playbook_details': playbook_details}

        final_details = df.apply(build_final_details, axis=1)
        df[['exit_signal_reason', 'context_snapshot']] = pd.DataFrame(final_details.tolist(), index=df.index)

        final_cols = [
            "stock_code", "trade_time", "timeframe", "strategy_name", "close_price",
            "entry_score", "stable_platform_price", "entry_signal", "is_risk_warning",
            "exit_signal_code", "exit_severity_level", "exit_signal_reason",
            "triggered_playbooks", "triggered_playbooks_cn", "context_snapshot"
        ]
        for col in final_cols:
            if col not in df.columns:
                df[col] = None if col not in ['entry_score', 'exit_signal_code', 'exit_severity_level'] else 0.0

        sanitized_df = df[final_cols].replace({np.nan: None})

        return sanitized_df.to_dict('records')

    #       ├─> 战报模板科 (Template Section)
    #       │   -> 核心职责: 创建空的、标准格式的战报记录。
    #       │   -> 对应方法: _create_db_record_template()
    def _create_db_record_template(self, stock_code, timestamp, timeframe, strategy_name, row) -> Dict:
        """
        【V202.6 健壮性增强版】
        - 核心修复: 对所有从DataFrame中提取的数值类型，都使用 pd.notna() 进行检查，
                    并将 NaN 值转换成数据库可接受的类型（None 或 0.0），从根源上杜绝 `ValidationError`。
        """
        # ▼▼▼【代码修改】对所有数值进行NaN检查和转换 ▼▼▼
        close_price = row.get(f'close_{timeframe}')
        entry_score = row.get('entry_score', 0.0)
        risk_score = row.get('risk_score', 0.0)
        
        return {
            "stock_code": stock_code, "trade_time": timestamp, "timeframe": timeframe,
            "strategy_name": strategy_name, 
            "close_price": float(close_price) if pd.notna(close_price) else None, # 修复点1: close_price
            "entry_score": float(entry_score) if pd.notna(entry_score) else 0.0, # 修复点2: entry_score
            "stable_platform_price": None, # 此处留空，由主流程填充并检查
            "entry_signal": False, "is_risk_warning": False,
            "exit_signal_code": 0, "exit_severity_level": 0, "exit_signal_reason": None,
            "is_pullback_setup": bool(row.get('SETUP_SCORE_PLATFORM_SUPPORT_PULLBACK', 0) > 0),
            "triggered_playbooks": [], "triggered_playbooks_cn": [], 
            "context_snapshot": {
                'close': float(close_price) if pd.notna(close_price) else None,
                'entry_score': float(entry_score) if pd.notna(entry_score) else 0.0,
                'risk_score': float(risk_score) if pd.notna(risk_score) else 0.0, # 修复点3: risk_score
            }
        }
        # ▲▲▲【代码修改】▲▲▲

    #       └─> 风险详情科 (Risk Details Section)
    #           -> 核心职责: 为风险事件填充详细的上下文和原因。
    #           -> 对应方法: _fill_risk_details()
    def _fill_risk_details(self, record: Dict, row: pd.Series, risk_playbook_map: Dict):
        """
        【V202.7 健壮性增强版】
        - 核心修复: 对所有添加到 context_snapshot 的数值，进行 pd.notna() 检查，确保数据清洁。
        """
        timestamp = record['trade_time']
        triggered_risks_en = []
        risk_details_df = getattr(self, '_last_risk_details_df', pd.DataFrame())
        if not risk_details_df.empty and timestamp in risk_details_df.index:
            risk_details_for_day = risk_details_df.loc[timestamp]
            active_risks = risk_details_for_day[risk_details_for_day > 0].index
            triggered_risks_en = [risk.replace('RISK_SCORE_', '') for risk in active_risks]
        
        triggered_risks_cn = [risk_playbook_map.get(risk, risk) for risk in triggered_risks_en]
        reason = ", ".join(triggered_risks_cn) if triggered_risks_cn else "综合风险评分超阈值"
        
        record['exit_signal_reason'] = reason
        record['triggered_playbooks'] = triggered_risks_en
        record['triggered_playbooks_cn'] = triggered_risks_cn
        
        context = record['context_snapshot']
        context['cost_slope_5d'] = float(v) if pd.notna(v := row.get('SLOPE_5_peak_cost_D')) else None
        context['winner_rate_slope_5d'] = float(v) if pd.notna(v := row.get('SLOPE_5_total_winner_rate_D')) else None
        context['peak_stability_slope_5d'] = float(v) if pd.notna(v := row.get('SLOPE_5_peak_stability_D')) else None
        context['pressure_above_slope_5d'] = float(v) if pd.notna(v := row.get('SLOPE_5_pressure_above_D')) else None
        record['context_snapshot'] = sanitize_for_json(context) # sanitize_for_json 仍然保留，用于处理其他JSON不支持的类型

    # ─> 风险量化局 (Risk Quantifier Bureau)
    #    -> 核心职责: 将抽象的风险原因，翻译成直观的“仪表盘”读数。
    #    -> 局长: _quantify_risk_reasons()
    def _quantify_risk_reasons(self, record: pd.Series) -> str:
        """
        【V237.2 情报来源修复版】风险量化局
        - 核心修复: 修正了获取参数的来源，从错误的 self.daily_params 改为
                    正确的 self.unified_config，确保能正确读取到 risk_quantifier_params。
        """
        quantifier_params = self._get_params_block('risk_quantifier_params')
        if not self._get_param_value(quantifier_params.get('enabled'), False):
            return "风险量化未启用"
        risk_scores = []
        for name, params in quantifier_params.items():
            if not isinstance(params, dict) or 'source_metric' not in params:
                continue
            source_metric = params['source_metric']
            if source_metric not in record.index or pd.isna(record[source_metric]):
                continue
            value = record[source_metric]
            direction = params.get('direction', 1)
            center_point = params.get('center_point', 0)
            steepness = params.get('steepness', 1.0)
            cn_name = params.get('cn_name', name)
            # 使用 sigmoid 函数将原始指标值映射到 0-100 的风险分数
            # direction * (value - center_point) 确保了正确的风险方向
            normalized_value = direction * (value - center_point)
            score = 100 / (1 + np.exp(-steepness * normalized_value))
            if score > 20: # 只报告有意义的风险项
                risk_scores.append({'name': cn_name, 'score': int(score), 'value': value})
        if not risk_scores:
            return "无显著量化风险"
        # 按风险分数从高到低排序
        risk_scores.sort(key=lambda x: x['score'], reverse=True)
        # 构建可读的风险描述字符串
        reasons = [f"{item['name']}({item['score']})" for item in risk_scores]
        return "; ".join(reasons)

    # ─> 盘中预警中心 (Intraday Alert Center)
    #    -> 核心职责: 独立于主战略流程，生成分钟级的实时战术警报。
    #    -> 指挥官: generate_intraday_alerts()
    def generate_intraday_alerts(self, daily_df: pd.DataFrame, minute_df: pd.DataFrame, params: dict) -> List[Dict]:
        """
        【V104 终局 · 执行版】
        - 核心革命: 建立一个独立的“战术执行层”，将日线策略的“战略确认”转化为分钟级别的“实时警报”。
        - 工作流程:
          1. 接收已经运行完V103日线策略的 `daily_df`。
          2. 识别出所有“冲高日”(Upthrust Day)。
          3. 在其后的“戒备日”(Alert Day)中，监控分钟线数据。
          4. 将日线级别的确认条件，“翻译”成实时价格的触发条件。
          5. 在条件满足的第一分钟，生成并返回警报。
        """
        print("\n" + "="*60)
        print("====== 开始执行【战术执行层 V104】(分钟级实时警报) ======")

        # --- 步骤0: 加载执行层参数 ---
        exec_params = self._get_params_block('intraday_execution_params')
        if not self._get_param_value(exec_params.get('enabled'), False):
            print("    - [信息] 战术执行层被禁用，跳过。")
            return []

        # --- 步骤1: 从日线策略中，识别出所有的“冲高日” ---
        # 我们需要重新计算V103的冲高日逻辑，以确保独立性
        p_attack = self.exit_strategy_params.get('upthrust_distribution_params', {})
        lookback_period = self._get_param_value(p_attack.get('upthrust_lookback_days'), 5)
        is_upthrust_day = daily_df['high_D'] > daily_df['high_D'].shift(1).rolling(window=lookback_period).max()
        
        upthrust_days_df = daily_df[is_upthrust_day]
        if upthrust_days_df.empty:
            print("    - [信息] 在日线数据中未发现任何“冲高日”，无需启动盘中监控。")
            return []
        
        print(f"    - [战略确认] 发现 {len(upthrust_days_df)} 个潜在的“冲高日”，将在次日启动盘中监控。")

        alerts = []
        # --- 步骤2: 遍历每一个“冲高日”，监控其后一天的分钟行情 ---
        for upthrust_date, upthrust_row in upthrust_days_df.iterrows():
            alert_date = upthrust_date + pd.Timedelta(days=1)
            
            # 获取“戒备日”当天的分钟数据
            alert_day_minute_df = minute_df[minute_df.index.date == alert_date.date()].copy()
            if alert_day_minute_df.empty:
                continue

            # --- 步骤3: 准备实时触发的阈值 ---
            # 阈值1: 戒备日当天的开盘价
            alert_day_open = alert_day_minute_df.iloc[0]['open_M']
            # 阈值2: 冲高日当天的开盘价
            upthrust_day_open = upthrust_row['open_D']

            print(f"      -> [进入戒备] 日期: {alert_date.date()} | 监控启动...")
            print(f"         - 触发阈值1 (低于今日开盘): {alert_day_open:.2f}")
            print(f"         - 触发阈值2 (低于昨日开盘): {upthrust_day_open:.2f}")

            # --- 步骤4: 在分钟线上应用“翻译”后的触发条件 ---
            triggered_minutes = alert_day_minute_df[
                (alert_day_minute_df['close_M'] < alert_day_open) &
                (alert_day_minute_df['close_M'] < upthrust_day_open)
            ]

            if not triggered_minutes.empty:
                # 获取第一次触发警报的分钟
                first_alert_minute = triggered_minutes.iloc[0]
                alert_time = first_alert_minute.name
                alert_price = first_alert_minute['close_M']
                
                alert_info = {
                    "alert_time": alert_time,
                    "alert_price": alert_price,
                    "alert_type": "INTRADAY_UPTHRUST_REJECTION",
                    "cn_name": "【盘中警报】冲高回落结构确认",
                    "reason": f"价格({alert_price:.2f})跌破今日开盘({alert_day_open:.2f})与昨日开盘({upthrust_day_open:.2f})",
                    "daily_setup_date": upthrust_date.date()
                }
                alerts.append(alert_info)
                print(f"         - [警报触发!] 时间: {alert_time.time()} | 价格: {alert_price:.2f} | 原因: {alert_info['reason']}")
        
        print("====== 【战术执行层 V104】执行完毕 ======")
        print("="*60 + "\n")
        return alerts


    #  ★ 军事监察与战地验尸总署 (Inspector General & Field Forensics Administration)
    #     -> 核心职责: (非作战序列) 负责调试、审查与战后复盘，确保作战系统的正确性。
    # ─> 战地验尸官 (Field Coroner)
    #    -> 核心职责: 对特定日期的完整计分流程进行法医级解剖。
    #    -> 首席验尸官: _deploy_field_coroner_probe()
    def _deploy_field_coroner_probe(self, df: pd.DataFrame, probe_date_str: str):
        """
        【V245.0 智能寻路版】战地验尸探针
        - 核心升级: 解决了因指定日期为非交易日而导致验尸失败的问题。
        - 新规则:
          1. 尝试定位指定的 `probe_date`。
          2. 如果找不到，不再直接放弃，而是自动向后搜索，直到找到第一个存在的交易日为止。
          3. 在报告中明确告知指挥部，实际验尸的日期是哪个替代日期。
        - 收益: 极大提升了探针的实战适应性和易用性，确保验尸任务总能成功执行。
        """
        try:
            # ▼▼▼【代码修改 V245.0】: 引入智能寻路逻辑 ▼▼▼
            target_date = pd.to_datetime(probe_date_str, utc=True)
            
            # 智能寻路：如果指定日期不存在，则向后寻找最近的下一个有效交易日
            if target_date not in df.index:
                # 获取所有在指定日期之后（包含）的有效交易日索引
                future_dates = df.index[df.index >= target_date]
                if not future_dates.empty:
                    actual_probe_date = future_dates[0]
                    print(f"\n--- [战地验尸总署-情报] 指定日期 {probe_date_str} 非交易日，自动选用下一个交易日 {actual_probe_date.strftime('%Y-%m-%d')} 进行验尸 ---")
                else:
                    print(f"\n--- [战地验尸总署-警告] 未在数据中找到指定的验尸日期 {probe_date_str} 或任何后续日期 ---")
                    return
            else:
                actual_probe_date = target_date
            
            print("\n" + "="*25 + " [战地验尸总署-探针报告] " + "="*25)
            print(f"  [验尸目标]: {self._get_param_value(self.strategy_info.get('name'))} @ {actual_probe_date.strftime('%Y-%m-%d')}")
            
            # 提取当日的完整数据行
            row = df.loc[actual_probe_date]
            # ▲▲▲【代码修改 V245.0】▲▲▲

            # --- 核心矛盾点审查 ---
            print("\n--- [1. 核心矛盾点审查] ---")
            risk_score = row.get('risk_score', 'N/A')
            exit_signal_code = row.get('exit_signal_code', 'N/A')
            alert_level, alert_reason = self._check_tactical_alerts(row, self.unified_config)
            print(f"  - 当日风险分 (risk_score): {risk_score}")
            print(f"  - 当日离场码 (exit_signal_code): {exit_signal_code}")
            print(f"  - 当日预警级别 (alert_level from _check_tactical_alerts): {alert_level}")
            print(f"  - 当日预警原因 (alert_reason from _check_tactical_alerts): {alert_reason}")

            # --- 风险评估溯源 ---
            print("\n--- [2. 风险评估溯源] ---")
            if self._last_risk_details_df is not None and actual_probe_date in self._last_risk_details_df.index:
                risk_components = self._last_risk_details_df.loc[actual_probe_date].dropna()
                if not risk_components.empty:
                    print("  - 风险分构成:")
                    for k, v in risk_components.to_dict().items():
                        print(f"    - {k}: {v}")
                else:
                    print("  - 当日无任何风险项被激活。")
            else:
                print("  - 未找到当日的风险构成详情。")

            # --- 进攻评估溯源 ---
            print("\n--- [3. 进攻评估溯源] ---")
            entry_score = row.get('entry_score', 'N/A')
            print(f"  - 当日进攻分 (entry_score): {entry_score}")
            if self._last_score_details_df is not None and actual_probe_date in self._last_score_details_df.index:
                score_components = self._last_score_details_df.loc[actual_probe_date].dropna()
                if not score_components.empty:
                    print("  - 进攻分构成:")
                    for k, v in score_components.to_dict().items():
                        print(f"    - {k}: {v}")
                else:
                    print("  - 当日无任何进攻项被激活。")
            else:
                print("  - 未找到当日的进攻构成详情。")
            
            print("\n" + "="*30 + " [验尸报告结束] " + "="*30 + "\n")

        except Exception as e:
            print(f"\n--- [战地验尸总署-严重错误] 探针在执行验尸过程中发生异常: {e} ---")

    # ─> 专项调查组 (Special Investigation Group)
    #    -> 核心职责: 针对“入场分”或“风险分”进行专项调查与复盘。
    #    ├─> 入场分调查员: _probe_entry_score_details()
    def _probe_entry_score_details(
        self, 
        df: pd.DataFrame, 
        probe_dates: List[str], 
        final_score: pd.Series, 
        intermediate_masks: Dict, 
        playbook_definitions: List[Dict], 
        trigger_violent_reversal: pd.Series, 
        setup_scores: Dict[str, pd.Series], 
        context_window: int,
        atomic_states: Dict[str, pd.Series] # 修正1: 增加缺失的 atomic_states 参数
    ):
        """
        【V152.1 “探针后勤补给”修正版】
        - 核心修正: 修复了因函数签名缺少 atomic_states 参数，以及函数内部未定义
                    default_series 而导致的 NameError 崩溃问题。
        """
        print("\n" + "="*25 + " [黑匣子探针启动] " + "="*25)
        
        # 修正2: 在函数内部定义其所需的 default_series
        default_series = pd.Series(False, index=df.index)

        for probe_date_str in probe_dates:
            try:
                probe_ts = pd.to_datetime(probe_date_str).tz_localize('UTC')
                if probe_ts not in df.index:
                    print(f"\n--- [探针信息] 日期 {probe_date_str} 不在当前数据帧的索引中，跳过。 ---")
                    continue
                
                print(f"\n\n--- [探针] 正在剖析日期: {probe_date_str} ---")
                
                print("\n[全局关键变量]")
                print(f"  - 当日收盘价 (close_D): {df.loc[probe_ts, 'close_D']:.2f}")
                ema55_col = 'EMA_55_D'
                if ema55_col in df.columns:
                    print(f"  - 55日均线 ({ema55_col}): {df.loc[probe_ts, ema55_col]:.2f}")
                else:
                    print(f"  - 55日均线 ({ema55_col}): 未找到")
                print(f"  - 暴力反转信号 (trigger_violent_reversal): {trigger_violent_reversal.loc[probe_ts]}")
                
                for playbook in playbook_definitions:
                    name = playbook['name']
                    p_type = playbook.get('type')
                    masks = intermediate_masks.get(name, {})
                    
                    print(f"\n--- 剖析剧本: [{name}] (类型: {p_type}) ---")
                    print(f"  - 触发条件 (trigger_mask): {masks.get('trigger_mask', default_series).loc[probe_ts]}")
                    print(f"  - 站位条件 (side_mask): {masks.get('side_mask', default_series).loc[probe_ts]}")
                    print(f"  - 准备条件 (setup_mask): {masks.get('setup_mask', default_series).loc[probe_ts]}")
                    
                    if p_type == 'setup_score':
                        rules = playbook.get('scoring_rules', {})
                        min_req = rules.get('min_setup_score_to_trigger', 0)
                        # 此处使用 playbook.get('setup_score_series', default_series) 是安全的，因为 playbook 来自 intermediate_masks
                        score_series = masks.get('setup_score_series', default_series)
                        max_score = score_series.rolling(window=context_window, min_periods=1).max().loc[probe_ts]
                        reversal_window_active = atomic_states.get('CONTEXT_VIOLENT_REVERSAL_WINDOW', default_series).loc[probe_ts]
                        print(f"    -> 诊断(setup_score): 要求最低分>{min_req}, 近期最高分是{max_score:.2f}。豁免信号: {reversal_window_active}")

                    print(f"  - [最终决策] 剧本是否激活 (valid_mask): {masks.get('valid_mask', default_series).loc[probe_ts]}")

                print("\n[当日最终得分]")
                print(f"  - 总分 (final_score): {final_score.loc[probe_ts]:.2f}")
                print(f"--- [探针] 日期 {probe_date_str} 剖析完毕 ---")

            except Exception as e:
                print(f"\n--- [探针错误] 在处理日期 {probe_date_str} 时发生错误: {e} ---")
        print("\n" + "="*27 + " [黑匣子探针结束] " + "="*27 + "\n")

    #    └─> 风险分调查员: _probe_risk_score_details()
    def _probe_risk_score_details(self, total_risk_score: pd.Series, risk_details_df: pd.DataFrame, params: dict):
        """
        【V93.1 探针时间限制版】
        - 核心升级: 增加了可配置的时间限制，默认只显示指定日期之后的信息，使日志更聚焦。
        """
        print("\n" + "-"*25 + " 风险剧本探针 (Risk Playbook Probe) " + "-"*24)
        
        # ▼▼▼【代码修改 V93.1】: 增加时间限制逻辑 ▼▼▼
        # 1. 从配置中获取探针的起始日期，并提供一个符合您需求的默认值
        probe_start_date_str = self._get_param_value(self.debug_params.get('probe_start_date'), '2024-12-21')
        print(f"    -> 探针时间范围: 从 {probe_start_date_str} 开始")

        # 2. 筛选出所有有风险分的日期
        key_dates = total_risk_score.index[total_risk_score > 0]
        
        # 3. 应用时间过滤器
        if probe_start_date_str:
            try:
                start_date = pd.to_datetime(probe_start_date_str, utc=True)
                key_dates = key_dates[key_dates >= start_date]
            except Exception as e:
                # 这个异常处理现在不太可能被触发，但保留它是个好习惯
                print(f"      -> [警告] 探针起始日期 '{probe_start_date_str}' 格式错误，将显示所有风险日。错误: {e}")

        if key_dates.empty:
            print(f"    -> 在 {probe_start_date_str} 之后，未发现任何已触发的风险剧本。")
        else:
            risk_playbook_cn_map = {p['name']: p.get('cn_name', p['name']) for p in self.risk_playbook_blueprints}
            
            print(f"    -> 在指定时间范围内发现 {len(key_dates)} 个风险日，详情如下:")
            for date in key_dates:
                total_score = total_risk_score.loc[date]
                print(f"    {date.strftime('%Y-%m-%d')} | 最终风险分: {total_score:.0f}")
                
                triggered_playbooks = risk_details_df.loc[date][risk_details_df.loc[date] > 0]
                for name, score in triggered_playbooks.items():
                    cn_name = risk_playbook_cn_map.get(name, name)
                    print(f"                 -> 触发: '{cn_name}', 风险分 +{score:.0f}")
 

    def _create_persistent_state(
        self, 
        df: pd.DataFrame, 
        entry_event: pd.Series, 
        persistence_days: int, 
        break_condition: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        【V58.0 新增 - 状态机生成器】
        """
        if not entry_event.any():
            return pd.Series(False, index=df.index)
        event_groups = entry_event.cumsum()
        days_since_event = df.groupby(event_groups).cumcount()
        is_within_persistence = days_since_event < persistence_days
        is_active_period = event_groups > 0
        persistent_state = is_active_period & is_within_persistence
        if break_condition is not None:
            break_points = break_condition & persistent_state
            break_groups = break_points.groupby(event_groups).transform('idxmax')
            persistent_state &= (df.index < break_groups) | (break_groups.isna())
        return persistent_state












