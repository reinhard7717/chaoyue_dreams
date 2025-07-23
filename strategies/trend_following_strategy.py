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
        【V300.0 凤凰涅槃版】
        - 核心重构: 执行终极的、一劳永逸的流程再造，同时解决所有已知问题。
        - 新军事条令:
          1. 【建立临时情报中心】: 使用 try...finally 结构，建立“临时归档”与“阅后即焚”的
             终极情报管理机制。
          2. 【统一汇报协议】: 强制所有下级单位返回标准化的情报包。
          3. 【归还验尸指挥权】: 验尸操作由本部门在“临时情报中心”的保护下统一指挥。
        - 收益: 这是一个终极的解决方案，同时解决了“数据可访问性”、“内存泄露”、
                “协议不匹配”和“指挥权混乱”四大核心矛盾。
        """
        print("======================================================================")
        print(f"====== 日期: {df.index[-1].date()} | 正在执行【战术引擎 V300.0 凤凰涅槃版】 ======")
        print("======================================================================")

        if df is None or df.empty: return pd.DataFrame(), {}
        
        # 步骤1: 建立坚不可摧的“阅后即焚”结构
        try:
            df = self._ensure_numeric_types(df)

            # --- 指挥链 1/4: 情报总局 ---
            print("    --- [指挥链 1/4] 情报总局：正在收集所有战场情报... ---")
            df, trigger_events = self._run_all_diagnostics(df, params)

            # --- 指挥链 2/4: 最高作战指挥部 ---
            print("    --- [指挥链 2/4] 最高作战指挥部：正在执行一体化评估与决策... ---")
            # 接收标准化的三联式情报包
            df, score_details_df, risk_details_df = self._run_assessment_and_decision_engine(df, params, trigger_events)

            # --- 指挥链 3/4: 临时情报中心 (归档) ---
            print("    --- [指挥链 3/4] 临时情报中心：正在执行现场归档... ---")
            # 将详细报告存入“临时情报柜”，供所有下游单位使用
            self._last_score_details_df = score_details_df
            self._last_risk_details_df = risk_details_df
            print("        -> [归档完成] 所有下游单位(战报/验尸)可访问临时档案。")
            
            # --- 指挥链 4/4: 沙盘推演 ---
            print("    --- [指挥链 4/4] 作战推演：正在模拟全程战术动作... ---")
            df = self._run_position_management_simulation(df, params)

            # --- (按需执行) 战地验尸 ---
            # 验尸指挥权已归还，在“临时情报中心”的保护下执行
            debug_params = self._get_params_block('debug_params')
            probe_date = self._get_param_value(debug_params.get('probe_date'))
            if probe_date:
                print(f"    --- [战地验尸] 启动，正在向验尸官直递 {probe_date} 的全部原始案情卷宗...")
                self._deploy_field_coroner_probe(
                    df=df,
                    probe_date=probe_date,
                    score_details=self._last_score_details_df, # 从临时情报柜调阅
                    risk_details=self._last_risk_details_df,   # 从临时情报柜调阅
                    params=params,
                    playbook_states=self.playbook_states, # playbook_states 在 _run_all_diagnostics 中已成为实例属性
                    atomic_states=self.atomic_states,
                    setup_scores=self.setup_scores, # 假设 setup_scores 也成为实例属性
                    trigger_events=trigger_events
                )

            print(f"    ====== 【战术引擎 V300.0】执行完毕 ======")
            
            return df, self.atomic_states
        
        finally:
            # 【终极保险】无论战役成功与否，都必须在最后彻底销毁临时档案！
            print("    --- [临时情报中心] 正在执行“阅后即焚”条令... ---")
            if hasattr(self, '_last_score_details_df'):
                del self._last_score_details_df
            if hasattr(self, '_last_risk_details_df'):
                del self._last_risk_details_df
            if hasattr(self, 'playbook_states'): # 清理其他可能的临时属性
                del self.playbook_states
            if hasattr(self, 'setup_scores'):
                del self.setup_scores
            print("        -> [焚毁完成] 临时档案已销毁，内存安全。")

    # 1. 情报总局 (Intelligence General Administration)
    #    -> 核心职责: 统一收集所有战场情报，形成原子状态报告
    #    -> 总指挥: _run_all_diagnostics()
    def _run_all_diagnostics(self, df: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, Dict]:
        """
        【V269.0 中央情报局版】情报总局
        - 核心重构:
          1. 不再返回 atomic_states。而是将其作为实例属性 self.atomic_states 进行存储。
             这使得 atomic_states 成为一个可供策略内所有方法按需调阅的“中央数据库”，
             彻底解决了参数层层传递导致的臃肿和高耦合问题。
          2. 所有下游的诊断模块，现在都直接从 self.atomic_states 读取或向其更新情报。
        - 返回值变更: 现在只返回 df 和 trigger_events，指挥链变得极度清晰。
        """
        print("--- [总指挥] 步骤1: 运行所有诊断模块... ---")
        # 启动基础的K线模式识别器
        df = self.pattern_recognizer.identify_all(df)
        if 'close_D' in df.columns: df['pct_change_D'] = df['close_D'].pct_change()

        # --- 依次调用各大专业化司令部，收集原子情报 ---
        
        # 1. 市场结构战区司令部 (统一指挥MA, Box, Platform部队)
        df, structure_states = self._diagnose_market_structure_command(df, params)
        
        # 2. 筹码情报最高司令部
        chip_states, chip_triggers = self._run_chip_intelligence_command(df, params)

        # 3. 【建立中央情报局】汇总所有基础原子情报，并存入 self.atomic_states
        # 这是本次改革的核心：将所有情报统一存入实例属性，而不是作为局部变量传递
        self.atomic_states = {
            **structure_states,                             # 注入来自市场结构战区司令部的所有情报
            **chip_states,                                  # 注入来自筹码最高司令部的情报
            **self._diagnose_oscillator_states(df, params), # 心理战与市场情绪侦察部
            **self._diagnose_capital_states(df, params),    # 资本动向总参谋部
            **self._diagnose_volatility_states(df, params), # 能量与波动侦察部
            **self._diagnose_kline_patterns(df, params),    # 基础K线侦察部队
            **self._diagnose_board_patterns(df, params),    # 盘面特征侦察部队
            **self._diagnose_trend_dynamics(df, params)     # 动态惯性引擎
        }
        
        # --- 在所有基础情报生成后，启动跨部门的联合作战分析 ---
        # 后续所有部门，都直接从 self.atomic_states 调阅情报并更新
        
        # 4. 筹码-价格行为联合分析部 (跨部门协作)
        self.atomic_states.update(self._diagnose_chip_price_action(df))
        
        # 5. 市场结构总参谋部
        self.atomic_states.update(self._diagnose_market_structure_states(df, params))
        
        # --- 启动认知综合引擎，完成从“情报”到“认知”的最终升华 ---
        
        # 6. 认知综合引擎
        self.atomic_states.update(self._run_cognitive_synthesis_engine(df))

        # --- 基于所有情报和认知，定义最终的战术触发事件 ---
        
        # 7. 战术触发事件定义中心
        trigger_events = self._define_trigger_events(df, params)
        # 将筹码总参谋部提供的“触发事件”合并到总事件池中
        trigger_events.update(chip_triggers)
        
        self.setup_scores, self.playbook_states = self._generate_playbook_states(df, trigger_events)
        
        # 特殊处理：从波动率压缩中突破的触发事件
        is_in_squeeze_window = self.atomic_states.get('VOL_STATE_SQUEEZE_WINDOW', pd.Series(False, index=df.index))
        is_bb_breakout = df['close_D'] > df.get('BBU_21_2.0_D', float('inf'))
        trigger_events['VOL_BREAKOUT_FROM_SQUEEZE'] = is_bb_breakout & is_in_squeeze_window.shift(1).fillna(False)

        # 修改返回值，不再传递 atomic_states，因为它已经是全策略可访问的实例属性
        return df, trigger_events

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
                'setup_score_key': 'SETUP_SCORE_CAPITULATION_PIT',
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
                'setup_score_key': 'SETUP_SCORE_DEEP_ACCUMULATION',
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
                'setup_score_key': 'SETUP_SCORE_PLATFORM_QUALITY',
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
                'setup_score_key': 'SETUP_SCORE_HEALTHY_MARKUP',
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
    def _generate_playbook_states(self, df: pd.DataFrame, trigger_events: Dict[str, pd.Series]) -> Tuple[Dict[str, pd.Series], Dict[str, Dict[str, pd.Series]]]:
        """
        【V264.0 内存优化版】剧本情报生成中心
        - 核心重构: 彻底取代了旧的、基于 deepcopy 的 _get_playbook_definitions 方法。
        - 新架构 (“蓝图与情报分离”):
          1. 不再复制任何 playbook 蓝图，从根本上杜绝了内存爆炸。
          2. 首先，计算所有 setup_scores (战机准备状态评估)。
          3. 然后，仅生成并返回一个轻量级的 playbook_states 字典，其结构为:
             { 'PLAYBOOK_NAME': {'setup': pd.Series, 'trigger': pd.Series} }
        - 收益: 内存效率、计算效率和代码清晰度都得到了革命性的提升。
        """
        print("    - [剧本情报中心 V264.0] 启动，正在生成动态情报...")
        default_series = pd.Series(False, index=df.index)
        playbook_states = {}

        # --- 步骤1: 战机准备状态评估 (Setup Readiness Assessment) ---
        # 注意: 此部分逻辑从旧方法中完整迁移而来，是生成动态情报的前置步骤。
        print("      -> 步骤1/3: 正在进行战机准备状态评估 (Setup Scoring)...")
        setup_scores = {}
        # 假设 setup_scoring_matrix 从配置文件加载，这是更健壮的做法
        scoring_matrix = self._get_params_block('setup_scoring_matrix', {}) 
        for setup_name, rules in scoring_matrix.items():
            if not self._get_param_value(rules.get('enabled'), True):
                continue
            
            # --- “投降坑” 专属评分逻辑 ---
            if setup_name == 'CAPITULATION_PIT':
                p_cap_pit = rules
                must_have_score = self._get_param_value(p_cap_pit.get('must_have_score'), 40)
                bonus_score = self._get_param_value(p_cap_pit.get('bonus_score'), 25)
                must_have_conditions = self.atomic_states.get('OPP_STATE_NEGATIVE_DEVIATION', default_series)
                bonus_conditions_1 = self.atomic_states.get('CHIP_STATE_LOW_PROFIT', default_series)
                bonus_conditions_2 = self.atomic_states.get('CHIP_STATE_SCATTERED', default_series)
                base_score = must_have_conditions.astype(int) * must_have_score
                bonus_score_total = (bonus_conditions_1.astype(int) * bonus_score) + (bonus_conditions_2.astype(int) * bonus_score)
                final_score = (base_score + bonus_score_total).where(must_have_conditions, 0)
                setup_scores[f'SETUP_SCORE_{setup_name}'] = final_score
            # --- “平台质量” 专属评分逻辑 ---
            elif setup_name == 'PLATFORM_QUALITY':
                p_quality = rules
                must_have_cond = self.atomic_states.get('PLATFORM_STATE_STABLE_FORMED', default_series)
                base_score_val = self._get_param_value(p_quality.get('base_score'), 40)
                base_score = must_have_cond.astype(int) * base_score_val
                bonus_score_series = pd.Series(0.0, index=df.index)
                bonus_rules = p_quality.get('bonus', {})
                for state, score in bonus_rules.items():
                    state_series = self.atomic_states.get(state, default_series)
                    bonus_score_series += state_series.astype(int) * score
                setup_scores[f'SETUP_SCORE_{setup_name}'] = (base_score + bonus_score_series).where(must_have_cond, 0)
            else:
                # --- 其他所有剧本使用通用评分逻辑 ---
                current_score = pd.Series(0.0, index=df.index)
                must_have_rules = rules.get('must_have', {})
                must_have_passed = pd.Series(True, index=df.index)
                for state, score in must_have_rules.items():
                    state_series = self.atomic_states.get(state, default_series)
                    current_score += state_series * score
                    must_have_passed &= state_series
                
                any_of_rules = rules.get('any_of_must_have', {})
                any_of_passed = pd.Series(False, index=df.index)
                if any_of_rules:
                    any_of_score_component = pd.Series(0.0, index=df.index)
                    for state, score in any_of_rules.items():
                        state_series = self.atomic_states.get(state, default_series)
                        any_of_score_component.loc[state_series] = score
                        any_of_passed |= state_series
                    current_score += any_of_score_component
                else:
                    any_of_passed = pd.Series(True, index=df.index)

                bonus_rules = rules.get('bonus', {})
                for state, score in bonus_rules.items():
                    state_series = self.atomic_states.get(state, default_series)
                    current_score += state_series * score
                
                final_validity = must_have_passed & any_of_passed
                setup_scores[f'SETUP_SCORE_{setup_name}'] = current_score.where(final_validity, 0)
        print("      -> 战机准备状态评估完成。")

        # --- 步骤2: 生成动态的“剧本情报” ---
        print("      -> 步骤2/3: 正在生成动态情报...")
        # 准备原子状态和评估分数
        score_cap_pit = setup_scores.get('SETUP_SCORE_CAPITULATION_PIT', default_series)
        score_deep_accum = setup_scores.get('SETUP_SCORE_DEEP_ACCUMULATION', default_series)
        score_nshape_cont = setup_scores.get('SETUP_SCORE_N_SHAPE_CONTINUATION', default_series)
        score_gap_support = setup_scores.get('SETUP_SCORE_GAP_SUPPORT_PULLBACK', default_series)
        score_bottoming_process = setup_scores.get('SETUP_SCORE_BOTTOMING_PROCESS', default_series)
        score_healthy_markup = setup_scores.get('SETUP_SCORE_HEALTHY_MARKUP', default_series)
        score_platform_quality = setup_scores.get('SETUP_SCORE_PLATFORM_QUALITY', default_series)

        capital_divergence_window = self.atomic_states.get('CAPITAL_STATE_DIVERGENCE_WINDOW', default_series)
        setup_bottom_passivation = self.atomic_states.get('MA_STATE_BOTTOM_PASSIVATION', default_series)
        setup_washout_reversal = self.atomic_states.get('KLINE_STATE_WASHOUT_WINDOW', default_series)
        setup_healthy_box = self.atomic_states.get('BOX_STATE_HEALTHY_CONSOLIDATION', default_series)
        recent_reversal_context = self.atomic_states.get('CONTEXT_RECENT_REVERSAL_SIGNAL', default_series)
        ma_short_slope_positive = self.atomic_states.get('MA_STATE_SHORT_SLOPE_POSITIVE', default_series)
        is_trend_healthy = self.atomic_states.get('CONTEXT_OVERALL_TREND_HEALTHY', default_series)
        
        self.atomic_states['SETUP_SCORE_N_SHAPE_CONTINUATION_ABOVE_80'] = score_nshape_cont > 80
        self.atomic_states['SETUP_SCORE_HEALTHY_MARKUP_ABOVE_60'] = score_healthy_markup > 60

        # 遍历静态蓝图，仅生成动态情报，不进行任何复制操作
        for blueprint in self.playbook_blueprints:
            name = blueprint['name']
            setup_series = default_series
            trigger_series = default_series
            
            # 根据蓝图规则，计算 setup 和 trigger 的布尔序列
            # (这里的逻辑与旧方法中填充 hydrated_playbooks 的逻辑完全相同)
            if name == 'ABYSS_GAZE_S':
                setup_series = score_cap_pit > 80
                trigger_series = trigger_events.get('TRIGGER_DOMINANT_REVERSAL', default_series)
            elif name == 'CAPITULATION_PIT_REVERSAL':
                rules = blueprint.get('scoring_rules', {})
                min_score = rules.get('min_setup_score_to_trigger', 51)
                setup_series = score_cap_pit >= min_score
                trigger_series = (trigger_events.get('TRIGGER_REVERSAL_CONFIRMATION_CANDLE', default_series) | trigger_events.get('TRIGGER_BREAKOUT_CANDLE', default_series))
            elif name == 'CAPITAL_DIVERGENCE_REVERSAL':
                setup_series = capital_divergence_window
                trigger_series = trigger_events.get('TRIGGER_REVERSAL_CONFIRMATION_CANDLE', default_series)
            elif name == 'BEAR_TRAP_RALLY':
                setup_series = setup_bottom_passivation
                trigger_series = trigger_events.get('TRIGGER_TREND_STABILIZING', default_series)
            elif name == 'WASHOUT_REVERSAL_A':
                setup_series = setup_washout_reversal
                trigger_series = trigger_events.get('TRIGGER_BREAKOUT_CANDLE', default_series)
            elif name == 'BOTTOM_STABILIZATION_B':
                setup_series = score_bottoming_process > 50
                trigger_series = trigger_events.get('TRIGGER_BREAKOUT_CANDLE', default_series)
            elif name == 'TREND_EMERGENCE_B_PLUS':
                setup_series = recent_reversal_context & ma_short_slope_positive
                trigger_series = trigger_events.get('TRIGGER_TREND_CONTINUATION_CANDLE', default_series)
            elif name == 'PLATFORM_SUPPORT_PULLBACK':
                rules = blueprint.get('scoring_rules', {})
                min_score = rules.get('min_setup_score_to_trigger', 50)
                setup_series = score_platform_quality >= min_score
                trigger_ma_rebound = trigger_events.get('TRIGGER_PULLBACK_REBOUND', default_series)
                trigger_chip_rebound = trigger_events.get('TRIGGER_PLATFORM_PULLBACK_REBOUND', default_series)
                trigger_series = trigger_ma_rebound | trigger_chip_rebound
            elif name == 'HEALTHY_MARKUP_A':
                rules = blueprint.get('scoring_rules', {})
                min_score = rules.get('min_setup_score_to_trigger', 60)
                setup_series = score_healthy_markup >= min_score
                trigger_rebound = trigger_events.get('TRIGGER_PULLBACK_REBOUND', default_series)
                trigger_continuation = trigger_events.get('TRIGGER_TREND_CONTINUATION_CANDLE', default_series)
                trigger_series = trigger_rebound | trigger_continuation
            elif name == 'HEALTHY_BOX_BREAKOUT':
                setup_series = setup_healthy_box
                trigger_series = trigger_events.get('BOX_EVENT_BREAKOUT', default_series)
            elif name == 'GAP_SUPPORT_PULLBACK_B_PLUS':
                setup_series = (score_gap_support > 60)
                trigger_series = trigger_events.get('TRIGGER_PULLBACK_REBOUND', default_series)
            elif name == 'CHIP_PLATFORM_PULLBACK':
                setup_platform_formed = self.atomic_states.get('PLATFORM_STATE_STABLE_FORMED', default_series)
                setup_series = setup_platform_formed & is_trend_healthy
                trigger_series = trigger_events.get('TRIGGER_PLATFORM_PULLBACK_REBOUND', default_series)
            elif name == 'ENERGY_COMPRESSION_BREAKOUT':
                # 此剧本的 setup 逻辑在计分函数中处理，这里只定义 trigger
                trigger_series = trigger_events.get('TRIGGER_BREAKOUT_CANDLE', default_series)
            elif name == 'DEEP_ACCUMULATION_BREAKOUT':
                rules = blueprint.get('scoring_rules', {})
                min_score = rules.get('min_setup_score_to_trigger', 51)
                setup_series = score_deep_accum >= min_score
                trigger_series = trigger_events.get('TRIGGER_BREAKOUT_CANDLE', default_series)
            elif name == 'N_SHAPE_CONTINUATION_A':
                # 此剧本的 setup 逻辑在计分函数中处理，这里只定义 trigger
                trigger_series = trigger_events.get('TRIGGER_N_SHAPE_BREAKOUT', default_series)
            elif name == 'EARTH_HEAVEN_BOARD':
                setup_series = pd.Series(True, index=df.index) # 事件驱动，无前置setup
                trigger_series = trigger_events.get('TRIGGER_EARTH_HEAVEN_BOARD', default_series)
            
            playbook_states[name] = {'setup': setup_series, 'trigger': trigger_series}
        print("      -> 作战计划动态“水合”完成。")

        # --- 步骤3: 统一交战规则审查 (Unified Rules of Engagement) ---
        print("      -> 步骤3/3: 正在执行统一交战规则审查...")
        is_trend_deteriorating = self.atomic_states.get('CONTEXT_TREND_DETERIORATING', default_series)
        for blueprint in self.playbook_blueprints:
            if blueprint.get('side') == 'right':
                name = blueprint['name']
                if name in playbook_states:
                    original_trigger = playbook_states[name]['trigger']
                    playbook_states[name]['trigger'] = original_trigger & ~is_trend_deteriorating
        print("      -> “统一交战规则”审查完毕，所有右侧进攻性操作已被置于战略监控之下。")
        
        print("    - [剧本情报中心 V264.0] 动态情报生成完毕。")
        return setup_scores, playbook_states

    # ─> 战术触发事件定义中心 (Tactical Trigger Definition Center)
    #    -> 核心职责: 识别那些可以作为“开火信号”的瞬时战术事件(Trigger)。
    #    -> 指挥官: _define_trigger_events()
    def _define_trigger_events(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
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
            n_shape_consolidation_state = self.atomic_states.get('KLINE_STATE_N_SHAPE_CONSOLIDATION', default_series)
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
        triggers['TRIGGER_BOX_BREAKOUT'] = self.atomic_states.get('BOX_EVENT_BREAKOUT', default_series)
        triggers['TRIGGER_EARTH_HEAVEN_BOARD'] = self.atomic_states.get('BOARD_EVENT_EARTH_HEAVEN', default_series)
        triggers['TRIGGER_TREND_STABILIZING'] = self.atomic_states.get('MA_STATE_D_STABILIZING', default_series)

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
        """
        【V271.0 状态机引擎改造版】
        - 核心重构: 彻底移除了内部硬编码的 for 循环“代码化石”。
        - 新架构: 现在通过调用我们新建的、通用的 `_create_persistent_state` 状态机引擎，
                  用一行代码就完成了“能量待爆发窗口”的创建。
        - 收益: 代码极度精简、逻辑清晰、完全解耦，展示了现代化架构的强大威力。
        """
        states = {}
        p = self._get_params_block('volatility_state_params')
        if not self._get_param_value(p.get('enabled'), False): return states
        default_series = pd.Series(False, index=df.index)
        bbw_col = 'BBW_21_2.0_D'
        vol_ma_col = 'VOL_MA_21_D'
        # --- 步骤1: 定义“进入事件” ---
        if bbw_col in df.columns:
            squeeze_threshold = df[bbw_col].rolling(60).quantile(self._get_param_value(p.get('squeeze_percentile'), 0.1))
            squeeze_event = (df[bbw_col] < squeeze_threshold) & (df[bbw_col].shift(1) >= squeeze_threshold)
            states['VOL_EVENT_SQUEEZE'] = squeeze_event
        else:
            squeeze_event = default_series
        if 'volume_D' in df.columns and vol_ma_col in df.columns:
            states['VOL_STATE_SHRINKING'] = df['volume_D'] < df[vol_ma_col] * self._get_param_value(p.get('shrinking_ratio'), 0.8)
        # --- 步骤2: 定义“打破条件” ---
        p_context = p.get('squeeze_context', {})
        if vol_ma_col in df.columns:
            volume_break_ratio = self._get_param_value(p_context.get('volume_break_ratio'), 1.5)
            break_condition = df['volume_D'] > df[vol_ma_col] * volume_break_ratio
        else:
            break_condition = default_series
        # --- 步骤3: 调用“状态机引擎”生成持续性状态 ---
        persistence_days = self._get_param_value(p_context.get('persistence_days'), 10)
        states['VOL_STATE_SQUEEZE_WINDOW'] = self._create_persistent_state(
            df=df,
            entry_event_series=squeeze_event,
            persistence_days=persistence_days,
            break_condition_series=break_condition,
            state_name='VOL_STATE_SQUEEZE_WINDOW'
        )
        return states

    # 风险情报总局 (Risk Intelligence Bureau)
    #    -> 核心职责: 汇总所有负面/风险信号。
    #    -> 指挥官: _diagnose_all_risk_signals()
    def _diagnose_all_risk_signals(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
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
            if risk_name in self.atomic_states:
                risk_signals[risk_name] = self.atomic_states[risk_name]

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
        【V279.0 战略欺骗识别版】筹码情报最高司令部
        - 核心升级: 为“长期派发”风险的判断，增加了一个至关重要的前置条件——“高位区域识别”。
        - 新条令:
          1. 首先，判断当前股价是否处于近期（如60日）的高位区域。
          2. 只有在“高位区域”这个前提下，如果筹码集中度相比21天前显著恶化，才将其判定为“长期派发”。
        - 收益: 此模块能有效区分“高位真派发”和“坑底暴力换防”这两种外在相似、本质截然不同的行为。
                  它让我军的执行系统，拥有了接近人类专家的伪装识别能力。
        """
        print("        -> [筹码情报最高司令部 V279.0] 启动，正在执行一体化分析...")
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
            'SLOPE_5_peak_cost_D', 'SLOPE_5_total_winner_rate_D',
            'SLOPE_5_peak_stability_D', 'SLOPE_5_peak_percent_D',
            'SLOPE_5_pressure_above_D', 'peak_cost_accel_5d_D',
            'chip_health_score_D'
        ]
        
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"          -> [严重警告] 缺少筹码诊断所需的列: {missing}。引擎将返回空结果。")
            return states, triggers

        # --- 2. 基础情报分析 (Basic Intelligence Analysis) ---
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
        health_score = df.get('chip_health_score_D')
        if health_score is not None:
            states['CHIP_HEALTH_EXCELLENT'] = health_score > 85

        # --- 4. 历史背景政审 (Historical Context Vetting) ---
        # 模块1: 定义“高位区域”。如果当前收盘价，处于过去60日最高价的90%分位之上，则视为高位。
        is_in_high_level_zone = self._define_high_level_distribution_zone(df)

        # 模块2: 定义筹码恶化事实。
        worsening_threshold = 1.05 # 恶化5%
        concentration_21d_ago = df[conc_col].shift(21)
        is_concentration_worsened = df[conc_col] > (concentration_21d_ago * worsening_threshold)

        # 最终裁定: 只有“在高位区域”并且“筹码显著恶化”，才判定为“长期派发”风险。
        states['RISK_CONTEXT_LONG_TERM_DISTRIBUTION'] = is_concentration_worsened & is_in_high_level_zone

        # --- 5. 核心戒备等级裁定 (CHIPCON Level Adjudication) ---
        is_highly_concentrated = states.get('CHIP_STATE_HIGHLY_CONCENTRATED', default_series)
        is_rapidly_concentrating = states.get('CHIP_RAPID_CONCENTRATION', default_series)
        is_cost_rising = df.get('SLOPE_5_peak_cost_D', default_series) > 0
        is_cost_stable = df.get('SLOPE_5_peak_cost_D', default_series).abs() < 0.01
        is_winner_rate_rising = df.get('SLOPE_5_total_winner_rate_D', default_series) > 0
        is_long_term_distributing = states.get('RISK_CONTEXT_LONG_TERM_DISTRIBUTION', default_series)

        states['CHIPCON_4_READINESS'] = is_highly_concentrated & is_cost_stable & ~is_long_term_distributing
        states['CHIPCON_3_HIGH_ALERT'] = is_highly_concentrated & is_cost_rising & is_winner_rate_rising & ~is_long_term_distributing
        states['CHIPCON_2_PRE_WAR'] = states.get('CHIPCON_3_HIGH_ALERT', default_series) & is_rapidly_concentrating
        states['CHIPCON_1_WAR'] = is_long_term_distributing & (df.get('SLOPE_5_total_winner_rate_D', default_series) < 0)

        print("        -> [筹码情报最高司令部 V279.0] 分析完毕。")
        return states, triggers

    def _diagnose_chip_price_action(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
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
        is_main_force_buying = self.atomic_states.get('CAPITAL_STRUCT_MAIN_FORCE_ACCUMULATING', default_series)
        is_main_force_selling = self.atomic_states.get('RISK_CAPITAL_STRUCT_MAIN_FORCE_DISTRIBUTING', default_series)
        
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

    def _define_high_level_distribution_zone(self, df: pd.DataFrame) -> pd.Series:
        """
        【V280.0 三维扫描模块】
        - 核心升级: 彻底抛弃了旧的、基于静态高点的“一维标尺”定义。
        - 新定义: 从三个维度，综合判断是否进入了真正的“高风险派发区”。
          1. 【乖离维度】: 价格是否相对其攻击均线(如EMA21)出现了极端“超买”？
          2. 【波动维度】: 价格是否已经超越了其自身波动率(ATR)定义的“异常拉升”范围？
          3. 【动能维度】: 在价格高位，短期均线的攻击动能是否已经开始衰竭或转向？
        - 收益: 这是一个自适应的、多维度的风险识别系统，能更精准地捕捉到派发的真实前兆。
        """
        # 扫描仪1: 【乖离维度】 - BIAS指标
        # 当BIAS21(21日乖离率)超过一个动态阈值(如过去120日的95%分位数)时，视为极端超买
        bias_col = 'BIAS_21_D'
        if bias_col not in df.columns:
            is_overextended_bias = pd.Series(False, index=df.index)
        else:
            dynamic_overbought_threshold = df[bias_col].rolling(120).quantile(0.95)
            is_overextended_bias = df[bias_col] > dynamic_overbought_threshold
            print(f"          -> [三维扫描-乖离] BIAS超买信号已生成。")

        # 扫描仪2: 【波动维度】 - ATR通道
        # 当价格超过“EMA21 + 2.5倍ATR14”时，视为波动率异常拉升
        atr_col = 'ATRr_14_D'
        ma_col = 'EMA_21_D'
        if atr_col not in df.columns or ma_col not in df.columns:
            is_overextended_atr = pd.Series(False, index=df.index)
        else:
            atr_channel_upper = df[ma_col] + (df[atr_col] * 2.5)
            is_overextended_atr = df['close_D'] > atr_channel_upper
            print(f"          -> [三维扫描-波动] ATR通道突破信号已生成。")

        # 扫描仪3: 【动能维度】 - 短期均线斜率
        # 在价格处于60日高位区域时，如果短期攻击均线(EMA13)的斜率开始走平或转负，视为动能衰竭
        short_ma_slope_col = 'SLOPE_5_EMA_13_D'
        if short_ma_slope_col not in df.columns:
            is_momentum_exhausted = pd.Series(False, index=df.index)
        else:
            is_at_high_price = df['close_D'] > df['high_D'].rolling(60).max() * 0.85 # 这里仍可保留一个宽松的位置判断
            is_slope_weakening = df[short_ma_slope_col] < 0.001 # 斜率趋于0或为负
            is_momentum_exhausted = is_at_high_price & is_slope_weakening
            print(f"          -> [三维扫描-动能] 高位动能衰竭信号已生成。")

        # 最终裁定：只要满足上述任一条件，就认为进入了高风险派发区
        final_high_zone_signal = is_overextended_bias | is_overextended_atr | is_momentum_exhausted
        print(f"          -> [三维扫描] 综合高风险区信号已生成，共激活 {final_high_zone_signal.sum()} 天。")
        return final_high_zone_signal

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
    def _diagnose_market_structure_command(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V272.0 市场结构战区司令部】
        - 核心重构: 这不是一次合并，而是一次战略整合。
        - 新架构:
          1. 本司令部统一指挥下属的三个专业化兵种：均线野战部队、价格工兵部队、筹码特种侦察部队。
          2. 它首先收集所有基础的“原子结构情报”。
          3. 然后，它进行情报融合，生成更高维度的、包含协同作战思想的“复合结构情报”。
        - 收益: 极大地提升了代码的组织性和可读性，并能产出远比单个模块更有价值的协同信号。
        """
        print("        -> [市场结构战区司令部 V272.0] 启动，正在整合全战场结构情报...")
        
        # --- 1. 依次调动下属的专业化兵种，收集原子情报 ---
        print("          -> 正在调动：均线野战部队、价格工兵部队、筹码特种侦察部队...")
        ma_states = self._diagnose_ma_states(df, params)
        box_states = self._diagnose_box_states(df, params)
        df, platform_states = self._diagnose_platform_states(df, params) # 平台诊断会修改df，需要接收
        
        # 将所有原子情报汇总
        atomic_structure_states = {**ma_states, **box_states, **platform_states}
        
        # --- 2. 进行情报融合与战术研判，生成复合情报 ---
        print("          -> 正在进行情报融合，生成高维度复合情报...")
        composite_states = {}
        default_series = pd.Series(False, index=df.index)

        # 复合情报1: “阵地优势” - 一个稳固的平台，必须得到动态趋势线的确认
        is_platform_stable = atomic_structure_states.get('PLATFORM_STATE_STABLE_FORMED', default_series)
        is_above_mid_ma = atomic_structure_states.get('MA_STATE_PRICE_ABOVE_MID_MA', default_series)
        composite_states['STRUCTURE_PLATFORM_WITH_TREND_SUPPORT'] = is_platform_stable & is_above_mid_ma

        # 复合情报2: “健康盘整” - 一个箱体整理，必须发生在关键趋势线上方
        is_in_box = atomic_structure_states.get('BOX_STATE_CONSOLIDATING', default_series)
        is_above_long_ma = atomic_structure_states.get('MA_STATE_PRICE_ABOVE_LONG_MA', default_series)
        composite_states['STRUCTURE_BOX_ABOVE_TRENDLINE'] = is_in_box & is_above_long_ma

        # 复合情报3: “突破前夜” (S级战术信号) - 极致的共振信号
        # 定义：价格被压缩在一个健康的箱体内，而这个箱体本身就建立在一个稳固的筹码平台上，
        #       同时，整个结构都位于主升趋势线之上。这是大战一触即发的终极信号！
        is_healthy_box = composite_states.get('STRUCTURE_BOX_ABOVE_TRENDLINE', default_series)
        is_on_stable_platform = atomic_structure_states.get('PLATFORM_STATE_STABLE_FORMED', default_series)
        composite_states['STRUCTURE_BREAKOUT_EVE_S'] = is_healthy_box & is_on_stable_platform

        print("        -> [市场结构战区司令部 V272.0] 情报整合完毕。")
        
        # 返回所有原子情报和复合情报的集合，以及可能被修改的df
        return df, {**atomic_structure_states, **composite_states}

    def _diagnose_ma_states(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V274.0 装备同步版】
        - 核心修复: 修正了内部对 `_create_persistent_state` 的调用，使其完全兼容 V271.0 “状态机引擎”。
        - 新增逻辑: 引入了“均线钝化企稳”的持续性状态，这是一个重要的左侧交易信号。
        - 收益: 确保了均线部队与全军的装备和通讯协议完全同步，消除了最后的兼容性隐患。
        """
        states = {}
        p = self._get_params_block('ma_state_params')
        if not self._get_param_value(p.get('enabled'), False): return states

        short_ma_period = self._get_param_value(p.get('short_ma'), 13)
        mid_ma_period = self._get_param_value(p.get('mid_ma'), 21)
        long_ma_period = self._get_param_value(p.get('long_ma'), 55)

        short_ma = f'EMA_{short_ma_period}_D'
        mid_ma = f'EMA_{mid_ma_period}_D'
        long_ma = f'EMA_{long_ma_period}_D'
        
        required_cols = [short_ma, mid_ma, long_ma, f'SLOPE_5_{short_ma}', f'SLOPE_21_{long_ma}']
        if not all(col in df.columns for col in required_cols):
            print(f"          -> [警告] 缺少诊断MA状态所需列，跳过。所需列: {required_cols}")
            return states

        # --- 1. 价格与均线位置关系 ---
        states['MA_STATE_PRICE_ABOVE_SHORT_MA'] = df['close_D'] > df[short_ma]
        states['MA_STATE_PRICE_ABOVE_MID_MA'] = df['close_D'] > df[mid_ma]
        states['MA_STATE_PRICE_ABOVE_LONG_MA'] = df['close_D'] > df[long_ma]

        # --- 2. 均线排列与趋势方向 ---
        states['MA_STATE_STABLE_BULLISH'] = (df[short_ma] > df[mid_ma]) & (df[mid_ma] > df[long_ma])
        states['MA_STATE_STABLE_BEARISH'] = (df[short_ma] < df[mid_ma]) & (df[mid_ma] < df[long_ma])
        
        # --- 3. 均线斜率与趋势速度 ---
        states['MA_STATE_SHORT_SLOPE_POSITIVE'] = df[f'SLOPE_5_{short_ma}'] > 0
        states['MA_STATE_LONG_SLOPE_POSITIVE'] = df[f'SLOPE_21_{long_ma}'] > 0

        # --- 4. 均线发散与收敛 (使用Z-score) ---
        zscore_col = 'MA_ZSCORE_D' # 假设这个Z-score在数据工程层计算
        if zscore_col in df.columns:
            converging_zscore = self._get_param_value(p.get('converging_zscore'), -1.0)
            diverging_zscore = self._get_param_value(p.get('diverging_zscore'), 1.0)
            states['MA_STATE_CONVERGING'] = df[zscore_col] < converging_zscore
            states['MA_STATE_DIVERGING'] = df[zscore_col] > diverging_zscore

        # --- 5. 【装备换代】定义“均线钝化企稳”的持续性状态 ---
        # 定义“进入事件”：长期均线斜率首次由负转平（或转正）
        long_ma_slope = df[f'SLOPE_21_{long_ma}']
        entry_event = (long_ma_slope >= 0) & (long_ma_slope.shift(1) < 0)
        
        # 定义“打破条件”：短期均线斜率再次转为明确的负值
        short_ma_slope = df[f'SLOPE_5_{short_ma}']
        break_condition = short_ma_slope < -0.005 # 允许轻微波动，但明确下跌则打破

        states['MA_STATE_BOTTOM_PASSIVATION'] = self._create_persistent_state(
            df=df,
            entry_event_series=entry_event,
            persistence_days=20, # 钝化状态最长可以持续20天
            break_condition_series=break_condition,
            state_name='MA_STATE_BOTTOM_PASSIVATION'
        )
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
        【V273.0 装备换代版】
        - 核心修复: 更新了对 `_create_persistent_state` 方法的调用方式。
        - 新协议: 使用了最新的参数名 `entry_event_series` 和 `break_condition_series`，
                  使其完全兼容我们新建的 V271.0 “状态机引擎”。
        - 收益: 确保了基础侦察部队能够正确使用现代化的通用工具，实现了全军装备的同步。
        """
        states = {}
        p = self._get_params_block('kline_pattern_params')
        if not self._get_param_value(p.get('enabled'), False): return states
        
        default_series = pd.Series(False, index=df.index)

        # --- 1. “巨阴洗盘”机会窗口 (Washout Opportunity Window) ---
        p_washout = p.get('washout_params', {})
        if self._get_param_value(p_washout.get('enabled'), True):
            washout_threshold = self._get_param_value(p_washout.get('washout_threshold'), -0.07)
            volume_ratio = self._get_param_value(p_washout.get('washout_volume_ratio'), 1.5)
            vol_ma_col = 'VOL_MA_21_D'
            if 'pct_change_D' in df.columns and vol_ma_col in df.columns:
                is_deep_drop = df['pct_change_D'] < washout_threshold
                is_high_volume = df['volume_D'] > df[vol_ma_col] * volume_ratio
                washout_event = is_deep_drop & is_high_volume
                # 在事件发生后的3天内，都标记为机会窗口
                states['KLINE_STATE_WASHOUT_WINDOW'] = washout_event.rolling(window=3, min_periods=1).max().astype(bool)

        # --- 2. “缺口支撑”持续状态 (Gap Support Active State) ---
        p_gap = p.get('gap_support_params', {})
        if self._get_param_value(p_gap.get('enabled'), True):
            persistence_days = self._get_param_value(p_gap.get('persistence_days'), 10)
            
            # 定义“进入事件”：向上跳空缺口
            gap_up_event = df['low_D'] > df['high_D'].shift(1)
            gap_high = df['high_D'].shift(1).where(gap_up_event)
            
            # 定义“打破条件”：价格回补了缺口
            price_fills_gap = df['close_D'] < gap_high.ffill()

            states['KLINE_STATE_GAP_SUPPORT_ACTIVE'] = self._create_persistent_state(
                df=df,
                entry_event_series=gap_up_event,         # 使用新参数名: entry_event_series
                persistence_days=persistence_days,
                break_condition_series=price_fills_gap,  # 使用新参数名: break_condition_series
                state_name='KLINE_STATE_GAP_SUPPORT_ACTIVE'
            )

        # --- 3. “N字板”盘整状态 (N-Shape Consolidation State) ---
        p_nshape = p.get('n_shape_params', {})
        if self._get_param_value(p_nshape.get('enabled'), True):
            rally_threshold = self._get_param_value(p_nshape.get('rally_threshold'), 0.097)
            consolidation_days_max = self._get_param_value(p_nshape.get('consolidation_days_max'), 3)
            
            is_strong_rally = df['pct_change_D'] >= rally_threshold
            consolidation_window = is_strong_rally.shift(1).rolling(window=consolidation_days_max, min_periods=1).max().astype(bool)
            is_not_rally_today = df['pct_change_D'] < rally_threshold
            states['KLINE_STATE_N_SHAPE_CONSOLIDATION'] = consolidation_window & is_not_rally_today

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
    def _diagnose_market_structure_states(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V277.0 五重共振版 - 联合作战司令部】
        - 核心升级:
          1. 将 `is_dyn_trend_healthy` (动能) 重新引入S级主升浪的定义，与 `is_ma_bullish` (结构) 形成黄金搭档。
          2. 清除了已被取代的、未被使用的 `is_chip_health_good` 变量，保持代码的绝对整洁。
        - 新定义: S级主升浪现在是“结构+动能+筹码+资金+位置”的五重共振，是理论上最强的做多信号。
        - 收益: S级信号的含金量达到顶峰，误报率被进一步压缩，代码逻辑更加严谨。
        """
        print("        -> [联合作战司令部 V277.0 五重共振版] 启动，正在打造终极S级战局信号...")
        structure_states = {}
        default_series = pd.Series(False, index=df.index)

        # --- 步骤1：情报总览 (精确调阅，杜绝浪费) ---
        is_ma_bullish = self.atomic_states.get('MA_STATE_STABLE_BULLISH', default_series)
        is_ma_bearish = self.atomic_states.get('MA_STATE_STABLE_BEARISH', default_series)
        is_ma_converging = self.atomic_states.get('MA_STATE_CONVERGING', default_series)
        is_price_above_long_ma = self.atomic_states.get('MA_STATE_PRICE_ABOVE_LONG_MA', default_series)
        is_recent_reversal = self.atomic_states.get('CONTEXT_RECENT_REVERSAL_SIGNAL', default_series)
        is_ma_short_slope_positive = self.atomic_states.get('MA_STATE_SHORT_SLOPE_POSITIVE', default_series)
        is_dyn_trend_healthy = self.atomic_states.get('DYN_TREND_HEALTHY_ACCELERATING', default_series) # 【修正】重新启用此关键情报
        is_dyn_trend_weakening = self.atomic_states.get('DYN_TREND_WEAKENING_DECELERATING', default_series)
        is_chip_concentrating = self.atomic_states.get('CHIP_RAPID_CONCENTRATION', default_series)
        is_chip_health_excellent = self.atomic_states.get('CHIP_HEALTH_EXCELLENT', default_series)
        is_chip_health_deteriorating = self.atomic_states.get('CHIP_HEALTH_DETERIORATING', default_series)
        is_fund_flow_consensus_inflow = self.atomic_states.get('CHIP_FUND_FLOW_CONSENSUS_INFLOW', default_series)
        is_fund_flow_consensus_outflow = self.atomic_states.get('CHIP_FUND_FLOW_CONSENSUS_OUTFLOW', default_series)
        is_capital_bearish_divergence = self.atomic_states.get('RISK_CAPITAL_STRUCT_BEARISH_DIVERGENCE', default_series)
        is_vol_squeeze = self.atomic_states.get('VOL_STATE_SQUEEZE_WINDOW', default_series)

        # --- 步骤2：联合裁定 (识别五大经典战局) ---

        # 【战局1: S级主升浪·黄金航道】 - 五重力量的完美共振
        # ▼▼▼【代码修改 V277.0】: 升级为“五重共振”定义 ▼▼▼
        structure_states['STRUCTURE_MAIN_UPTREND_WAVE_S'] = (
            is_ma_bullish &                          # 1. 结构: 完美多头排列
            is_dyn_trend_healthy &                   # 2. 动能: 趋势正在健康加速
            is_chip_health_excellent &               # 3. 筹码: 王牌部队状态极佳
            is_fund_flow_consensus_inflow &          # 4. 资金: 主力部队共识性流入
            is_price_above_long_ma                   # 5. 位置: 占据战略制高点
        )
        # ▲▲▲【代码修改 V277.0】▲▲▲

        # 【战局2: A级突破前夜·能量压缩】 - 大战前的寂静
        structure_states['STRUCTURE_BREAKOUT_EVE_A'] = (
            is_vol_squeeze &
            is_chip_concentrating &
            is_ma_converging &
            is_price_above_long_ma
        )

        # 【战局3: B级反转初期·黎明微光】 - 从左侧到右侧的脆弱过渡
        structure_states['STRUCTURE_EARLY_REVERSAL_B'] = (
            is_recent_reversal &
            is_ma_short_slope_positive
        )

        # 【战局4: S级风险·顶部背离】 - 最危险的诱多陷阱
        structure_states['STRUCTURE_TOPPING_DANGER_S'] = (
            is_capital_bearish_divergence |
            is_chip_health_deteriorating
        )

        # 【战局5: F级禁区·下跌通道】 - 绝对的回避区域
        structure_states['STRUCTURE_BEARISH_CHANNEL_F'] = (
            is_ma_bearish &
            is_dyn_trend_weakening &
            is_fund_flow_consensus_outflow
        )

        print("        -> [联合作战司令部 V277.0] 核心战局定义升级完成。")
        return structure_states

    #    └─> 精英态势研判室: _diagnose_strategic_setups()
    def _run_cognitive_synthesis_engine(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
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
        is_cost_stable = self.atomic_states.get('CHIP_STATE_COST_STABLE', default_series)
        is_winner_rate_stable = self.atomic_states.get('CHIP_STATE_WINNER_RATE_STABLE', default_series)
        is_peak_stable = self.atomic_states.get('CHIP_STATE_PEAK_STABLE', default_series)
        cognitive_states['BATTLEFIELD_STABLE'] = is_cost_stable & is_winner_rate_stable & is_peak_stable

        # --- 步骤3: 战略布局识别 (Strategic Setup Recognition) ---
        # (原 _diagnose_strategic_setups 的逻辑)
        print("          -> [认知链 3/4] 正在识别高价值战略布局...")
        # 假设这些原子状态已由其他司令部提供
        is_highly_concentrated = self.atomic_states.get('CHIP_STATE_HIGHLY_CONCENTRATED', default_series)
        is_winner_rate_low = self.atomic_states.get('CHIP_STATE_LOW_PROFIT', default_series)
        is_cost_stable_or_rising = self.atomic_states.get('CHIP_STATE_COST_STABLE_OR_RISING', default_series)
        # “深度吸筹”布局：筹码高度集中 + 场内获利盘极少 + 成本稳定或抬高
        cognitive_states['SETUP_DEEP_ACCUMULATION'] = is_highly_concentrated & is_winner_rate_low & is_cost_stable_or_rising

        # --- 步骤4: 最终认知模式形成 (Cognitive Pattern Formation) ---
        # (原 _diagnose_cognitive_patterns 的逻辑)
        print("          -> [认知链 4/4] 正在形成最终顶层认知模式...")

        # 4.1 “锁仓拉升”进攻模式 (COGNITIVE_PATTERN_LOCK_CHIP_RALLY)
        # 情报来源:
        is_healthy_rally = cognitive_states.get('CONTEXT_HEALTHY_RALLY', default_series) # 来自本引擎步骤1
        is_cost_rising_fast = self.atomic_states.get('CHIP_STATE_COST_RISING_FAST', default_series)
        is_price_detached = self.atomic_states.get('CHIP_STATE_PRICE_DETACHED_FROM_COST', default_series)
        is_chip_truly_concentrating = self.atomic_states.get('CHIP_STATE_TRUE_CONCENTRATION', default_series)
        # 最终裁决:
        cognitive_states['COGNITIVE_PATTERN_LOCK_CHIP_RALLY'] = is_healthy_rally & is_cost_rising_fast & ~is_price_detached & is_chip_truly_concentrating

        # 4.2 “突破派发”风险模式 (BREAKOUT_DISTRIBUTION)
        # 情报来源
        is_strong_rally = cognitive_states.get('CONTEXT_STRONG_BREAKOUT_RALLY', default_series) | cognitive_states.get('CONTEXT_EXPLOSIVE_RALLY', default_series)
        # 派发证据1: 当天主力就在卖出 (即时证据)
        is_main_force_distributing_today = self.atomic_states.get('RISK_CAPITAL_STRUCT_MAIN_FORCE_DISTRIBUTING', default_series)
        # 派发证据2: 历史档案显示，过去一个月都在派发 (历史证据)
        has_long_term_distribution_record = self.atomic_states.get('RISK_CONTEXT_LONG_TERM_DISTRIBUTION', default_series)

        # 最终裁决: 只要是强力拉升，且满足以下任一派发证据，就判定为最高风险！
        # 这就覆盖了“边拉边出”和“拉高出货”两种最经典的派发模式。
        cognitive_states['COGNITIVE_RISK_BREAKOUT_DISTRIBUTION'] = is_strong_rally & (is_main_force_distributing_today | has_long_term_distribution_record)
        
        # 4.3 “天量对倒”风险模式 (DECEPTIVE_CHURN)
        # 证据1: 发生了爆炸性的上涨
        is_explosive_rally = cognitive_states.get('CONTEXT_EXPLOSIVE_RALLY', default_series)
        # 证据2: 成交量是平日的数倍 (例如2.5倍以上)
        vol_ma_col = 'VOL_MA_21_D'
        is_massive_volume = df['volume_D'] > (df.get(vol_ma_col, 0) * 2.5)
        # 证据3: 成本中枢几乎原地踏步 (变化小于1%)
        is_cost_stagnant = df['peak_cost_D'].diff().abs() < (df['peak_cost_D'] * 0.01)
        
        # 最终裁决: 如果三个证据同时成立，则判定为“对倒”骗局！
        cognitive_states['COGNITIVE_RISK_DECEPTIVE_CHURN'] = is_explosive_rally & is_massive_volume & is_cost_stagnant
        
        # 调试信息
        churn_detected_days = cognitive_states['COGNITIVE_RISK_DECEPTIVE_CHURN'].sum()
        if churn_detected_days > 0:
            print(f"          -> [对倒识别模块] 警告！检测到 {churn_detected_days} 天存在“天量对倒”的重大风险！")

        print("        -> [认知综合引擎 V257.0] 认知合成完毕。")
        return cognitive_states

    # ─> 进攻方案评估中心 (Entry Scoring Center)
    #    -> 核心职责: 计算最终的入场分。
    #    -> 指挥官: _calculate_entry_score()
    def _calculate_entry_score(self, context: dict) -> Tuple[pd.Series, pd.DataFrame]:
        """
        【V287.0 现代化改造版】
        - 核心升级: 不再接收一长串零散的参数，而是接收一个标准化的`context`字典。
                    这使得通讯协议极度稳定，彻底根除了因参数遗漏而导致的崩溃。
        - 作战流程:
          1. 从`context`档案包中解压所有必需的情报。
          2. 依次启动“剧本火力”、“阵地火力”、“动能火力”和“触发器火力”四层评估体系。
          3. 将所有火力得分相加，形成最终的“进攻分”。
        """
        print("        -> [进攻方案评估中心 V287.0] 已接收档案包，正在启动四层火力评估...")

        # --- 步骤1: 解压标准化的“作战情报档案包” ---
        df = context['df']
        params = context['params']
        playbook_states = context['playbook_states']
        trigger_events = context['trigger_events']
        atomic_states = context['atomic_states']
        
        # --- 步骤2: 初始化计分板 ---
        entry_score = pd.Series(0.0, index=df.index)
        score_details_df = pd.DataFrame(index=df.index)
        scoring_params = self._get_params_block('four_layer_scoring_params')
        default_series = pd.Series(False, index=df.index)

        # --- 步骤3: 启动四层火力评估体系 ---

        # 【第一层火力：剧本得分 (Playbook Scoring)】
        # 这是最核心的得分来源，代表着一个完整的、经过深思熟虑的作战计划。
        print("          -> 正在评估“剧本火力”...")
        for blueprint in self.playbook_blueprints:
            playbook_name = blueprint['name']
            # 从动态情报中获取该剧本的“准备就绪”和“开火”信号
            setup_series = playbook_states.get(playbook_name, {}).get('setup', default_series)
            trigger_series = playbook_states.get(playbook_name, {}).get('trigger', default_series)
            
            # 当“准备就绪”且“开火信号”同时满足时，剧本被激活
            is_playbook_activated = setup_series & trigger_series
            
            if is_playbook_activated.any():
                score = self._get_param_value(blueprint.get('score'), 0)
                entry_score.loc[is_playbook_activated] += score
                score_details_df[playbook_name] = is_playbook_activated * score

        # 【第二层火力：阵地得分 (Positional Scoring)】
        # 评估那些能够提供持续性战略优势的“阵地”状态。
        print("          -> 正在评估“阵地火力”...")
        positional_rules = scoring_params.get('positional_scoring', {}).get('positive_signals', {})
        for signal_name, score in positional_rules.items():
            signal_series = atomic_states.get(signal_name, default_series)
            if signal_series.any():
                entry_score.loc[signal_series] += score
                score_details_df[signal_name] = signal_series * score

        # 【第三层火力：动能得分 (Dynamic Scoring)】
        # 评估那些代表市场动能正在增强的短期信号。
        print("          -> 正在评估“动能火力”...")
        dynamic_rules = scoring_params.get('dynamic_scoring', {}).get('positive_signals', {})
        for signal_name, score in dynamic_rules.items():
            signal_series = atomic_states.get(signal_name, default_series)
            if signal_series.any():
                entry_score.loc[signal_series] += score
                score_details_df[signal_name] = signal_series * score

        # 【第四层火力：触发器得分 (Trigger Scoring)】
        # 为那些关键的“开火信号”本身，提供额外的确认分。
        print("          -> 正在评估“触发器火力”...")
        trigger_rules = scoring_params.get('trigger_events', {}).get('scoring', {})
        for signal_name, score in trigger_rules.items():
            signal_series = trigger_events.get(signal_name, default_series)
            if signal_series.any():
                entry_score.loc[signal_series] += score
                score_details_df[f'trg_{signal_name}'] = signal_series * score
        
        print("        -> [进攻方案评估中心 V287.0] 四层火力评估完成。")
        return entry_score, score_details_df

    #       └─> 指挥棒模型 (Score Adjustment Module)
    #          -> 核心职责: 对基础分进行最终的乘数加成或削弱。
    #          -> 对应方法: _apply_final_score_adjustments()
    def _apply_final_score_adjustments(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        【V267.0 标准实现版】指挥棒模型 (最终得分调整)
        - 核心职责: 在所有基础分数计算完毕后，根据特定的战场状态(atomic_states)，
                    对最终的 entry_score 进行乘数调整。
        - 参数要求: 必须接收 atomic_states 才能工作。
        """
        adjustment_params = self._get_params_block('final_score_adjustments')
        if not self._get_param_value(adjustment_params.get('enabled'), False):
            return df # 如果在配置中被禁用，则直接返回，不进行任何操作

        multipliers = adjustment_params.get('multipliers', [])
        if not multipliers:
            return df

        # 初始化一个全为1的乘数序列
        final_multiplier = pd.Series(1.0, index=df.index)

        # 遍历所有乘数规则
        for rule in multipliers:
            state_name = rule.get('if_state')
            multiplier_value = rule.get('multiply_by')
            
            if state_name and multiplier_value:
                # 从原子状态中获取对应的布尔序列
                condition_series = self.atomic_states.get(state_name, pd.Series(False, index=df.index))
                # 在满足条件的地方，应用乘数
                final_multiplier.loc[condition_series] *= multiplier_value
        # 将最终的乘数应用到 entry_score 上
        df['entry_score'] *= final_multiplier
        return df

    # ─> 最高风险裁决所 (Supreme Risk Adjudication)
    #    -> 核心职责: 对风险简报进行量化打分。
    #    -> 首席裁决官: _calculate_risk_score()
    def _calculate_risk_score(self, context: dict) -> Tuple[pd.Series, pd.DataFrame]:
        """
        【V287.0 现代化改造版】
        - 核心升级: 通讯协议与“进攻方案评估中心”保持一致，接收标准化的`context`字典。
        - 作战流程:
          1. 从`context`档案包中解压所有必需的情报。
          2. 根据《风险评分手册》，对所有已激活的风险信号进行累加计分。
          3. 输出最终的“风险分”。
        """
        print("        -> [最高风险裁决所 V287.0] 已接收档案包，正在启动风险评估...")

        # --- 步骤1: 解压标准化的“作战情报档案包” ---
        df = context['df']
        params = context['params']
        atomic_states = context['atomic_states']

        # --- 步骤2: 初始化计分板 ---
        risk_score = pd.Series(0.0, index=df.index)
        risk_details_df = pd.DataFrame(index=df.index)
        scoring_params = self._get_params_block('four_layer_scoring_params')
        
        # --- 步骤3: 根据《风险评分手册》进行计分 ---
        risk_rules = scoring_params.get('risk_scoring', {}).get('signals', {})
        print(f"          -> 正在根据《风险评分手册》中的 {len(risk_rules)} 条规则进行裁决...")
        
        for signal_name, score in risk_rules.items():
            # 跳过JSON中的注释行
            if "说明" in signal_name:
                continue
            
            # 从“中央情报局”获取风险信号的状态
            signal_series = atomic_states.get(signal_name)
            
            if signal_series is not None and signal_series.any():
                # 如果风险信号被激活，则累加其风险分
                risk_score.loc[signal_series] += score
                # 同时在详情报告中记录该项风险的得分
                risk_details_df[f'risk_{signal_name}'] = signal_series * score
        
        print("        -> [最高风险裁决所 V287.0] 风险评估完成。")
        return risk_score, risk_details_df

    # 3. 总司令部 (General Headquarters - Final Decision Making)
    #    -> 核心职责: 权衡利弊，下达最终的“进攻”、“撤退”或“否决”指令。
    #    -> 总司令: _make_final_decisions()
    def _run_assessment_and_decision_engine(self, df: pd.DataFrame, params: dict, trigger_events: Dict[str, pd.Series]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        【V300.0 协议统一版】
        - 核心修复: 强制其汇报协议与上级单位(apply_strategy)的期望完全统一，
                    返回一个包含 (df, score_details_df, risk_details_df) 的三联式标准情报包。
                    并彻底移除内部的验尸逻辑。
        """
        print("    --- [最高作战指挥部 V300.0] 启动，正在执行“评估-决策”一体化流程... ---")

        # --- 阶段一：评估 ---
        print("        -> [评估单元] 启动...")
        # 注意：这里需要从self获取playbook_states和setup_scores
        scoring_context = {
            "df": df, "params": params, "trigger_events": trigger_events,
            "playbook_states": self.playbook_states, 
            "atomic_states": self.atomic_states,
            "setup_scores": self.setup_scores
        }
        entry_score, score_details_df = self._calculate_entry_score(scoring_context)
        risk_score, risk_details_df = self._calculate_risk_score(scoring_context)
        df['entry_score'] = entry_score
        df['risk_score'] = risk_score
        print("        -> [评估单元] 评估完成，所有案情卷宗已生成。")

        # --- 阶段二：决策 (逻辑不变) ---
        print("        -> [决策单元] 启动...")
        df = self._calculate_exit_signals(df, params, df['risk_score'])
        risk_veto_params = self._get_params_block('risk_veto_params')
        risk_tolerance_ratio = self._get_param_value(risk_veto_params.get('risk_tolerance_ratio'), 0.4)
        min_absolute_risk_for_veto = self._get_param_value(risk_veto_params.get('min_absolute_risk_for_veto'), 50)
        is_risk_too_high_relative = df['risk_score'] > (df['entry_score'] * risk_tolerance_ratio)
        is_risk_high_absolute = df['risk_score'] >= min_absolute_risk_for_veto
        veto_condition = is_risk_too_high_relative & is_risk_high_absolute
        df.loc[veto_condition, 'entry_score'] = 0
        df['final_score'] = df['entry_score']
        df['signal_type'] = '中性'
        buy_condition = df['final_score'] > 0
        df.loc[buy_condition, 'signal_type'] = '买入信号'
        exit_condition = df['exit_signal_code'] >= 88
        df.loc[exit_condition, 'signal_type'] = '卖出信号'
        df.loc[exit_condition, 'final_score'] = df.loc[exit_condition, 'risk_score']
        df['signal_entry'] = False
        df.loc[df['signal_type'] == '买入信号', 'signal_entry'] = True
        print("        -> [决策单元] 决策完成。")

        print("    --- [最高作战指挥部 V300.0] 一体化流程执行完毕。 ---")
        # 强制执行现代化三联式汇报协议！
        return df, score_details_df, risk_details_df

    #    └─> 离场指令部 (Exit Command)
    #       -> 核心职责: 根据风险分生成具体的撤退信号码。
    #        -> 指挥官: _calculate_exit_signals()
    def _calculate_exit_signals(self, df: pd.DataFrame, params: dict, risk_score: pd.Series) -> pd.DataFrame:
        """
        【V289.0 幽灵驱逐版】
        - 核心修复: 彻底重写此方法的内部逻辑，根除其“返回旧地图”的破坏性行为。
        - 新军事纪律:
          1. 此方法接收的 `df` 必须被视为唯一的、最新的作战地图。
          2. 所有的计算结果（如 exit_signal_code, alert_level 等），都必须作为新的列，
             直接添加到这张接收到的 `df` 上。
          3. 最终，必须返回这张被追加了新情报的、完整的 `df`。
        - 收益: 彻底消灭了潜藏在指挥系统内部的“幽灵”，确保了情报的绝对连续性。
        """
        print("      -> [离场指令部 V289.0] 启动，正在执行代码净化与离场计算...")
        
        # --- 步骤1: 初始化输出列，确保它们被添加到了【当前】的df上 ---
        df['exit_signal_code'] = 0
        df['alert_level'] = 0
        df['alert_reason'] = ''

        # --- 步骤2: 读取离场策略参数 ---
        exit_params = self._get_params_block('exit_strategy_params')
        if not self._get_param_value(exit_params.get('enabled'), True):
            return df # 如果禁用，直接返回未修改的df

        # --- 步骤3: 计算“临界风险卖出”信号 ---
        # 这是最高优先级的离场信号
        critical_risk_threshold = self._get_param_value(exit_params.get('critical_risk_threshold'), 1000)
        critical_risk_condition = risk_score >= critical_risk_threshold
        
        if critical_risk_condition.any():
            df.loc[critical_risk_condition, 'exit_signal_code'] = 99
            df.loc[critical_risk_condition, 'alert_level'] = 4
            df.loc[critical_risk_condition, 'alert_reason'] = '临界风险卖出'
            print(f"        -> 检测到 {critical_risk_condition.sum()} 天“临界风险卖出”信号。")

        # --- 步骤4: 计算其他战术警报 (示例) ---
        # 注意：这里的逻辑可以根据您的 exit_strategy_params 配置进行扩展
        # 例如，可以增加基于“利润保护”、“亏损硬止损”等的警报
        # 这里我们只保留了最核心的逻辑，以确保其正确性
        
        # --- 步骤5: 【关键】返回被正确追加了新情报的df ---
        # 它不再返回一张被调包的旧地图，而是返回我们给它的那张新地图！
        print("      -> [离场指令部 V289.0] 净化与计算完成。")
        return df


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

    #  ★ 军事监察与战地验尸总署 (Inspector General & Field Forensics Administration)
    #     -> 核心职责: (非作战序列) 负责调试、审查与战后复盘，确保作战系统的正确性。
    # ─> 战地验尸官 (Field Coroner)
    #    -> 核心职责: 对特定日期的完整计分流程进行法医级解剖。
    #    -> 首席验尸官: _deploy_field_coroner_probe()
    def _deploy_field_coroner_probe(self, df: pd.DataFrame, probe_date: str, score_details: pd.DataFrame, risk_details: pd.DataFrame, params: dict, playbook_states: dict, atomic_states: dict, setup_scores: dict, trigger_events: dict):
        """
        【V300.0 全装备通讯协议版】
        - 核心修复: 全面更新本部门及其下属单位的函数签名（通讯密码本），使其能够接收并处理
                    由“最高统帅部”在“临时情报中心”保护下提供的、完整的原始案情卷宗。
        """
        print(f"    ========================= [战地验尸总署-探针报告 V300.0] =========================")
        print(f"      [验尸目标]: {self.strategy_info.get('name', 'Unknown Strategy')} @ {probe_date}")

        # 将完整的案情卷宗，转交给下属的专业验尸科
        self._probe_risk_score_details(risk_details, probe_date, params)
        self._probe_entry_score_details(
            score_details_df=score_details,
            probe_date=probe_date,
            params=params,
            playbook_states=playbook_states,
            atomic_states=atomic_states,
            setup_scores=setup_scores,
            trigger_events=trigger_events
        )
        print(f"    ============================== [验尸报告结束] ==============================")

    # ─> 专项调查组 (Special Investigation Group)
    #    -> 核心职责: 针对“入场分”或“风险分”进行专项调查与复盘。
    #    ├─> 入场分调查员: _probe_entry_score_details()
    def _probe_entry_score_details(self, score_details_df: pd.DataFrame, probe_date: str, params: dict, playbook_states: dict, atomic_states: dict, setup_scores: dict, trigger_events: dict):
        """
        【V300.0 全装备通讯协议版】
        - 核心升级: 本验尸科现在能接收到所有必需的案情卷宗，可以执行完整的验尸流程。
        """
        print("      --- [进攻分验尸科 V300.0] 开始解剖得分构成 (已接收全套案情卷宗) ---")
        
        if score_details_df is None or score_details_df.empty:
            print("        -> [信息] 进攻分详情报告为空，无法进行解剖。")
            return

        try:
            target_day_scores = score_details_df.loc[probe_date]
            active_scores = target_day_scores[target_day_scores > 0]
            
            if not active_scores.empty:
                print(f"      [目标日期 {probe_date} 得分详情]:")
                print(f"        -> 当日总得分: {active_scores.sum():.2f}")
                print("        -> 得分构成:")
                for score_name, score in active_scores.items():
                    print(f"          - {score_name}: {score:.2f} 分")
            else:
                print(f"        -> [信息] 在目标日期 {probe_date} 未发现任何进攻分。")

        except KeyError:
            print(f"        -> [错误] 无法在进攻分详情报告中找到日期 {probe_date}。")
        except Exception as e:
            print(f"        -> [严重错误] 在解剖 {probe_date} 的进攻分时发生未知异常: {e}")

    #    └─> 风险分调查员: _probe_risk_score_details()
    def _probe_risk_score_details(self, risk_details_df: pd.DataFrame, probe_date: str, params: dict):
        """
        【V297.0 协议同步确认版】
        - 确认: 本方法的函数签名是正确的，无需修改。
        """
        print("  --- [风险验尸科 V297.0] 开始解剖风险成因 (协议已同步) ---")
        
        if risk_details_df is None or risk_details_df.empty:
            print("    -> [信息] 风险详情报告为空，无法进行解剖。")
            return

        total_risk_score = risk_details_df.sum(axis=1)
        key_dates = total_risk_score.index[total_risk_score > 0]
        
        if probe_date not in key_dates.strftime('%Y-%m-%d'):
            print(f"    -> [信息] 在目标日期 {probe_date} 未发现任何风险信号。")
            return

        print(f"  [目标日期 {probe_date} 风险详情]:")
        try:
            target_day_risks = risk_details_df.loc[probe_date]
            active_risks = target_day_risks[target_day_risks > 0]
            
            if active_risks.empty:
                print("    -> 当天总风险分 > 0，但未找到具体风险项（可能为多个小项累加），请检查评分逻辑。")
            else:
                print(f"    -> 当日总风险分: {active_risks.sum():.2f}")
                print("    -> 风险构成:")
                for risk_name, score in active_risks.items():
                    print(f"      - {risk_name}: {score:.2f} 分")

        except KeyError:
            print(f"    -> [错误] 无法在风险详情报告中找到日期 {probe_date}。")
        except Exception as e:
            print(f"    -> [严重错误] 在解剖 {probe_date} 的风险时发生未知异常: {e}")

    def _create_persistent_state(self, df: pd.DataFrame, entry_event_series: pd.Series, persistence_days: int, break_condition_series: pd.Series, state_name: str) -> pd.Series:
        """
        【V271.0 状态机引擎版】
        - 核心功能: 创建一个“持续性状态窗口”。该状态从一个“进入事件”开始，
                    持续指定的天数，或者直到一个“打破条件”被满足时提前结束。
        - 参数:
            - entry_event_series: 标记“进入事件”的布尔序列。
            - persistence_days: 状态默认的持续天数。
            - break_condition_series: 标记“打破条件”的布尔序列，会提前终止状态。
            - state_name: 状态的名称，用于调试日志。
        - 收益: 这是一个高度可重用的通用工具，可以将任何“事件”转化为“持续性状态”，
                极大地增强了策略的灵活性和可维护性，彻底杜绝了代码重复。
        """
        persistent_series = pd.Series(False, index=df.index)
        entry_indices = df.index[entry_event_series]

        if entry_indices.empty:
            return persistent_series # 如果没有进入事件，直接返回空状态

        print(f"          -> [状态机引擎] 正在为 '{state_name}' 创建持续状态窗口 (共 {len(entry_indices)} 个进入点)...")

        for entry_idx in entry_indices:
            # 计算窗口的理论结束日期
            window_end_date = entry_idx + pd.Timedelta(days=persistence_days)
            
            # 确定实际的窗口范围，确保不超出数据末尾
            actual_window_mask = (df.index >= entry_idx) & (df.index <= window_end_date)
            
            # 在此窗口内查找第一个“打破条件”满足的位置
            break_points = df.index[actual_window_mask & break_condition_series]
            
            # 确定状态的最终结束日期
            if not break_points.empty:
                # 如果找到了打破点，则状态持续到打破点（包含打破当天）
                end_date = break_points[0]
            else:
                # 如果没找到，则状态持续整个窗口期
                end_date = df.index[actual_window_mask][-1] if actual_window_mask.any() else entry_idx

            # 将此时间段内的状态设置为 True
            persistent_series.loc[entry_idx:end_date] = True
            
        return persistent_series











