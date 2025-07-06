# 文件: strategies/trend_following_strategy.py
# 版本: V21.0 - 适配新架构版
import logging
from services.indicator_services import IndicatorService
from utils.data_sanitizer import sanitize_for_json
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences
from strategies.kline_pattern_recognizer import KlinePatternRecognizer
from utils.config_loader import load_strategy_config

logger = logging.getLogger(__name__)

class TrendFollowStrategy:
    """
    趋势跟踪策略 (V21.0 - 适配新架构版)
    - 核心修改: 移除 IndicatorService 和 asyncio 依赖，变为纯粹的同步计算类。
    - 职责定位: 接收一个包含所有时间框架数据的字典，应用复杂的战术剧本和计分系统，输出最终决策。
    """

    def __init__(self, config: dict):
        """
        【V21.1 构造函数优化版】
        - 移除 self.indicator_service 的实例化，因为本类不应负责数据获取。
        - 移除已过时的 tactical_configs 和 tactical_params 逻辑。
        - 使本类成为一个更纯粹的计算单元。
        """
        # 保存传入的完整战术配置文件
        self.daily_params = config

        # 初始化K线形态识别器
        kline_params = self._get_params_block(self.daily_params, 'kline_pattern_params')
        self.pattern_recognizer = KlinePatternRecognizer(params=kline_params)

        # 初始化用于存储中间信号和分数的字典
        self.signals = {}
        self.scores = {}
        self._last_score_details_df = None # 用于存储详细的得分表，以便后续生成报告

        # 初始化调试参数
        self.debug_params = self._get_params_block(self.daily_params, 'debug_params')
        self.verbose_logging = self.debug_params.get('enabled', False) and self.debug_params.get('verbose_logging', False)
        print("--- [战术策略 TrendFollowStrategy (V21.1 优化版)] 初始化完成。---")

    # 参数解析辅助函数
    def _get_periods_for_timeframe(self, indicator_params: dict, timeframe: str) -> Optional[list]:
        """
        【V5.0 】根据时间周期从指标配置中智能获取正确的'periods'。
        能够处理新旧两种配置格式。

        Args:
            indicator_params (dict): 单个指标的完整配置字典 (例如，params['macd'])。
            timeframe (str): 需要获取参数的时间周期，如 'D', '60', '15'。

        Returns:
            Optional[list]: 找到的周期参数列表，或None。
        """
        if not indicator_params:
            return None

        # 优先处理新的 'configs' 列表格式
        if 'configs' in indicator_params and isinstance(indicator_params['configs'], list):
            for config_item in indicator_params['configs']:
                if timeframe in config_item.get('apply_on', []):
                    return config_item.get('periods')
            # 如果在configs列表中循环后没找到，返回None
            return None
        
        # 向后兼容旧的单一格式
        elif 'periods' in indicator_params:
            # 检查旧格式的apply_on，如果存在且不匹配则返回None
            apply_on = indicator_params.get('apply_on', [])
            if not apply_on or timeframe in apply_on:
                return indicator_params.get('periods')
            else:
                return None
        
        return None

    def _get_params_block(self, params: dict, block_name: str, default_return: Any = None) -> Any:
        """安全地从参数字典中获取一个配置块。"""
        default = default_return if default_return is not None else {}
        return params.get(block_name, default)

    def apply_strategy(self, df_dict: Dict[str, pd.DataFrame], params: dict) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        # print("\n--- [战术策略层 apply_strategy V22.2] 开始执行 ---") # 日常运行时可注释
        df = df_dict.get('D')
        if df is None or df.empty:
            return pd.DataFrame(), {}
        
        timeframe_suffixes = ['_D', '_W', '_M', '_5', '_15', '_30', '_60']
        
        # ▼▼▼ 精准的列名重命名逻辑 ▼▼▼
        rename_map = {
            col: f"{col}_D" for col in df.columns 
            if not any(col.endswith(suffix) for suffix in timeframe_suffixes) 
            and not col.startswith(('VWAP_', 'BASE_', 'playbook_', 'signal_', 'kline_', 'context_'))
        }

        if rename_map:
            df = df.rename(columns=rename_map)
        
        self.signals, self.scores = {}, {}
        df = self.pattern_recognizer.identify_all(df)
        df.loc[:, 'signal_top_divergence'] = self._find_top_divergence_exit(df, params)
        self._analyze_dynamic_box_and_ma_trend(df, params)

        print("    - [信息] 核心计分流程开始...")
        df.loc[:, 'entry_score'], atomic_signals, score_details_df = self._calculate_entry_score(df, df_dict, params)
        self._last_score_details_df = score_details_df
        
        score_threshold = self._get_params_block(params, 'entry_scoring_params').get('score_threshold', 100)
        df.loc[:, 'signal_entry'] = df['entry_score'] >= score_threshold
        
        print("    - [信息] 止盈逻辑判断开始...")
        df.loc[:, 'take_profit_signal'] = self._apply_take_profit_rules(df, df['signal_entry'], df['signal_top_divergence'], params)

        print("\n---【多时间框架协同策略(V22.2 融合版) 逻辑链调试】---")
        entry_signals = df[df['signal_entry']]
        print(f"【最终买入】(得分>{score_threshold})信号总数: {len(entry_signals)}")
        if not entry_signals.empty:
            last_entry_date = entry_signals.index[-1]
            last_entry_score = df.loc[last_entry_date, 'entry_score']
            print(f"  - 最近一次买入: {last_entry_date.date()} (当日总分: {last_entry_score:.2f})")
            
            print(f"  --- 得分小票 for {last_entry_date.date()} ---")
            score_breakdown = score_details_df.loc[last_entry_date].dropna()
            base_score_items = {k: v for k, v in score_breakdown.items() if k.startswith('BASE_') and v > 0}
            tactical_score_items = {k: v for k, v in score_breakdown.items() if not k.startswith('BASE_') and v != 0}
            
            base_total = sum(base_score_items.values())
            print(f"    [战略基础分: {base_total:.2f} pts]")
            if not base_score_items:
                print("      - (无)")
            else:
                for item, score in base_score_items.items():
                    print(f"      - {item}: {score:.2f} pts")

            tactical_total = sum(tactical_score_items.values())
            print(f"    [战术叠加分: {tactical_total:.2f} pts]")
            if not tactical_score_items:
                print("      - (无)")
            else:
                for item, score in tactical_score_items.items():
                    print(f"      - {item}: {score:.2f} pts")
            
            print(f"    ------------------------------------")
            print(f"    [核算总分: {base_total + tactical_total:.2f} pts]")

        return df, atomic_signals

    def prepare_db_records(self, stock_code: str, result_df: pd.DataFrame, atomic_signals: Dict[str, pd.Series], params: dict, result_timeframe: str = 'D') -> List[Dict[str, Any]]:
        """
        【V2.3 参数化修复版】将策略分析结果DataFrame转换为用于数据库存储的字典列表。
        - 新增 result_timeframe 参数，由调用者明确指定要保存的信号周期，解除了对配置文件的硬编码依赖。
        - 使用 sanitize_for_json 工具函数确保所有数据都是JSON兼容的原生Python类型。
        """
        df_with_signals = result_df[
            (result_df['signal_entry'] == True) | (result_df['take_profit_signal'] > 0)
        ].copy()
        if df_with_signals.empty:
            return []
        records = []
        strategy_name = self._get_params_block(params, 'strategy_info').get('name', 'multi_timeframe_collaboration')
        # 解释: 不再从配置文件读取 timeframe，而是直接使用传入的 result_timeframe 参数。
        # 这使得此方法可以被用来准备任何时间周期的信号记录。
        timeframe = result_timeframe
        for timestamp, row in df_with_signals.iterrows():
            # 从计分详情中获取激活的剧本
            triggered_playbooks_list = []
            if self._last_score_details_df is not None and timestamp in self._last_score_details_df.index:
                playbooks = self._last_score_details_df.loc[timestamp]
                triggered_playbooks_list = playbooks[playbooks > 0].index.tolist()
            is_setup_day = 'PULLBACK_SETUP' in triggered_playbooks_list
            # 全面应用 sanitize_for_json
            context_dict = {k: v for k, v in row.items() if pd.notna(v)}
            sanitized_context = sanitize_for_json(context_dict)
            record = {
                # --- 核心字段 ---
                "stock_code": stock_code,
                "trade_time": sanitize_for_json(timestamp),
                "timeframe": timeframe, # 使用了新的、由参数传入的 timeframe
                "strategy_name": strategy_name,
                "close_price": sanitize_for_json(row.get('close_D')),
                "entry_score": sanitize_for_json(row.get('entry_score', 0.0)),
                # --- 细分信号 ---
                "entry_signal": sanitize_for_json(row.get('signal_entry', False)),
                "exit_signal_code": sanitize_for_json(row.get('take_profit_signal', 0)),
                # --- 可查询字段 ---
                "is_long_term_bullish": sanitize_for_json(row.get('context_long_term_bullish', False)),
                "is_mid_term_bullish": sanitize_for_json(row.get('context_mid_term_bullish', False)),
                "is_pullback_setup": is_setup_day,
                "pullback_target_price": sanitize_for_json(row.get('pullback_target_price')),
                # --- 追溯与元数据 ---
                "triggered_playbooks": triggered_playbooks_list,
                "context_snapshot": sanitized_context,
            }
            records.append(record)
        return records

    def _calculate_entry_score(self, df: pd.DataFrame, df_dict: Dict[str, pd.DataFrame], params: dict) -> Tuple[pd.Series, Dict[str, pd.Series], pd.DataFrame]:
        """
        【V22.1 逻辑修正版】计算综合买入得分。
        - 核心修正: 为所有“趋势启动/反转”型剧本松绑，使用宽松前提(tactical_precondition)替代严格前提(strict_precondition)。
        - 重构: 分离了指标信号的计算和前提应用，允许对MACD金叉等信号进行精细化控制。
        """
        atomic_signals = {}
        score_details_df = pd.DataFrame(index=df.index)
        scoring_params = self._get_params_block(params, 'entry_scoring_params')
        points = scoring_params.get('points', {})
        # --- 步骤1: 计算并记录战略背景基础分 ---
        print("    [调试-计分V22.1] 步骤1: 计算并记录周线战略背景基础分...")
        king_signal_col = 'BASE_SIGNAL_BREAKOUT_TRIGGER'
        king_score = points.get('BREAKOUT_TRIGGER_SCORE', 150)
        all_base_score_cols = [king_signal_col]
        if king_signal_col in df.columns and df[king_signal_col].any():
            score_details_df.loc[df[king_signal_col], king_signal_col] = king_score
        playbook_scores_map = self._get_params_block(params, 'playbook_scores', default_return={})
        playbook_cols = [col for col in df.columns if col.startswith('playbook_') and col.endswith('_W')]
        for col in playbook_cols:
            playbook_name = col.replace('playbook_', '').replace('_W', '')
            score = playbook_scores_map.get(playbook_name, 0)
            base_col_name = f"BASE_{playbook_name.upper()}"
            all_base_score_cols.append(base_col_name)
            if score > 0 and df[col].any():
                score_details_df.loc[df[col], base_col_name] = score
        score_details_df.fillna(0, inplace=True)
        if king_signal_col in score_details_df.columns:
            king_signal_mask = (score_details_df[king_signal_col] > 0)
            if king_signal_mask.any():
                other_base_score_cols = [col for col in all_base_score_cols if col != king_signal_col and col in score_details_df.columns]
                score_details_df.loc[king_signal_mask, other_base_score_cols] = 0
        # --- 步骤2: 定义战术信号的前提条件  ---
        base_score = score_details_df.sum(axis=1)
        tactical_precondition = base_score > 0 # 宽松前提：周线有积极信号即可
        strict_precondition = tactical_precondition & df.get('context_mid_term_bullish', pd.Series(False, index=df.index)) # 严格前提：周线多头趋势已确立
        print(f"    [调试-计分V22.1] 步骤2: 定义战术前提... 满足宽松前提天数: {tactical_precondition.sum()}, 满足严格前提天数: {strict_precondition.sum()}")
        
        # --- 步骤3: 计算所有独立的日线战术原子信号 ---
        print("    [调试-计分V22.1] 步骤3: 计算日线战术原子信号...")
        
        # ▼▼▼ 为每个剧本分配合适的前提条件 ▼▼▼
        
        # --- A. 回踩与反转类 (趋势早期或转折点，使用宽松前提) ---
        cond_pullback_ma = self._find_pullback_to_ma_entry(df, tactical_precondition, params)
        cond_pullback_structure = self._find_pullback_to_structure_entry(df, tactical_precondition, params)
        cond_v_reversal = self._find_v_reversal_entry(df, tactical_precondition, params)
        cond_washout_reversal = self._find_washout_reversal_entry(df, tactical_precondition, params)
        
        # --- B. 左侧/底部信号类 (无前提条件) ---
        cond_bottom_divergence = self._find_bottom_divergence_entry(df, params)
        cond_bias_reversal = self._find_bias_reversal_entry(df, params)
        cond_capital_flow_divergence = self._find_capital_flow_divergence_entry(df, params)
        cond_winner_rate_reversal = self._find_winner_rate_reversal_entry(df, params)

        # --- C. 趋势启动/突破类 (关键！从严格前提改为宽松前提) ---
        cond_first_breakout = self._find_first_breakout_entry(df, tactical_precondition, params)
        cond_bb_squeeze_breakout = self._find_bband_squeeze_breakout(df, tactical_precondition, params)
        cond_energy_compression_breakout = self._find_energy_compression_breakout_entry(df, tactical_precondition, params)
        cond_cost_area_reinforcement = self._find_cost_area_reinforcement_entry(df, tactical_precondition, params)
        cond_chip_concentration_breakthrough = self._find_chip_concentration_breakthrough_entry(df, tactical_precondition, params)
        cond_chip_cost_breakthrough = self._find_chip_cost_breakthrough(df, tactical_precondition, params)
        cond_dynamic_box_breakout = self.signals.get('dynamic_box_breakout', pd.Series(False, index=df.index)) & tactical_precondition
        
        # --- D. 趋势延续/加速类 (维持严格前提) ---
        cond_pullback_setup, pullback_target_price = self._find_pullback_setup(df, strict_precondition, params)
        df['pullback_target_price'] = pullback_target_price
        cond_momentum = self._find_momentum_entry(df, strict_precondition, params)
        cond_doji_continuation = self._find_doji_continuation_entry(df, strict_precondition, params)
        cond_old_duck_head = self._find_old_duck_head_entry(df, strict_precondition, params)
        cond_n_shape_relay = self._find_n_shape_relay_entry(df, strict_precondition, params)
        cond_bullish_flag = self._find_bullish_flag_entry(df, strict_precondition, params)
        cond_relative_strength_maverick = self._find_relative_strength_maverick_entry(df, strict_precondition, params)
        cond_ma_acceleration = self._find_ma_acceleration_entry(df, strict_precondition, params)
        cond_chip_pressure_release = self._find_chip_pressure_release(df, strict_precondition, params)
        cond_chip_hurdle_clear = self._find_chip_hurdle_clear_entry(df, strict_precondition, params)
        cond_fib_pullback = self._find_fibonacci_pullback_entry(df, strict_precondition, params)

        # --- E. 指标信号类 (精细化控制前提) ---
        # 先计算原始信号，不加任何前提
        indicator_signals = self._find_indicator_entry(df, params)
        # 再根据指标自身特性，应用不同前提
        cond_dmi_cross = indicator_signals['dmi_cross'] & tactical_precondition # DMI金叉是趋势启动信号，用宽松前提
        cond_macd_low_cross = indicator_signals['macd_low_cross'] & tactical_precondition # 低位金叉是反转信号，用宽松前提
        cond_macd_zero_cross = indicator_signals['macd_zero_cross'] & tactical_precondition # 零轴金叉是趋势确立信号，用宽松前提
        cond_macd_high_cross = indicator_signals['macd_high_cross'] & strict_precondition # 高位金叉是加速信号，用严格前提

        # --- F. 其他确认信号与特殊形态  ---
        cond_cmf_confirm = self._check_cmf_confirmation(df, params)
        cond_vwap_support = self._check_vwap_confirmation(df_dict, params)
        cond_gap_support_state = self._check_upward_gap_support(df, params)
        board_patterns = self._identify_board_patterns(df, params)
        cond_earth_heaven_board, cond_turnover_board, cond_heaven_earth_board = board_patterns.get('earth_heaven_board', pd.Series(False, index=df.index)), board_patterns.get('turnover_board', pd.Series(False, index=df.index)), board_patterns.get('heaven_earth_board', pd.Series(False, index=df.index))
        cond_volume_breakdown = self._find_volume_breakdown_exit(df, params)
        cond_fund_flow_confirm = self._check_fund_flow_confirmation(df, params)
        steady_climb_params = self._get_params_block(params, 'steady_climb_params')
        is_low_volatility = pd.Series(False, index=df.index)
        if steady_climb_params.get('enabled', False):
            atr_period, atr_lookback, atr_percentile = steady_climb_params.get('atr_period', 14), steady_climb_params.get('atr_lookback', 60), steady_climb_params.get('atr_percentile', 0.3)
            atr_col = f'ATRN_{atr_period}_D'
            if atr_col in df.columns: is_low_volatility = df[atr_col] < df[atr_col].rolling(atr_lookback).quantile(atr_percentile)
        is_steady_climb_pullback = (cond_pullback_ma | cond_pullback_structure) & is_low_volatility
        is_normal_pullback = (cond_pullback_ma | cond_pullback_structure) & ~is_low_volatility
        kline_reversal_decent = df.get('kline_c_bullish_engulfing_decent', pd.Series(False, index=df.index)) | df.get('kline_c_piercing_line_decent', pd.Series(False, index=df.index)) | df.get('kline_s_hammer_shape_decent', pd.Series(False, index=df.index)) | df.get('kline_c_tweezer_bottom', pd.Series(False, index=df.index))
        kline_reversal_perfect = df.get('kline_c_bullish_engulfing_perfect', pd.Series(False, index=df.index)) | df.get('kline_c_piercing_line_perfect', pd.Series(False, index=df.index)) | df.get('kline_s_hammer_shape_perfect', pd.Series(False, index=df.index))
        cond_morning_star = df.get('kline_c_morning_star', pd.Series(False, index=df.index))
        cond_three_soldiers = df.get('kline_c_three_white_soldiers', pd.Series(False, index=df.index))
        kline_strong_bearish = df.get('kline_c_evening_star', pd.Series(False, index=df.index)) | df.get('kline_c_bearish_engulfing_decent', pd.Series(False, index=df.index)) | df.get('kline_c_three_black_crows', pd.Series(False, index=df.index)) | df.get('kline_c_dark_cloud_cover_decent', pd.Series(False, index=df.index))

        # ▼▼▼ “日线战术层 - 剧本计算总结”日志块 ▼▼▼
        print("\n---【日线战术层 - 剧本计算总结 V2.1-调试增强】---")
        playbook_summary = {
            "均线加速上涨 (MA_ACCELERATION)": cond_ma_acceleration,
            "筹码集中突破 (CHIP_CONCENTRATION_BREAKTHROUGH)": cond_chip_concentration_breakthrough,
            "成本区增强 (COST_AREA_REINFORCEMENT)": cond_cost_area_reinforcement,
            "投降坑反转 (WINNER_RATE_REVERSAL)": cond_winner_rate_reversal,
            "筹码压力释放 (CHIP_PRESSURE_RELEASE)": cond_chip_pressure_release,
            "筹码关口扫清 (CHIP_HURDLE_CLEAR)": cond_chip_hurdle_clear,
            "筹码成本区突破 (CHIP_COST_BREAKTHROUGH)": cond_chip_cost_breakthrough,
            "常规回踩 (PULLBACK_NORMAL)": is_normal_pullback,
            "稳步回踩 (PULLBACK_STEADY_CLIMB)": is_steady_climb_pullback, # 新增
            "V型反转 (V_SHAPE_REVERSAL)": cond_v_reversal,
            "底部首板 (FIRST_BREAKOUT)": cond_first_breakout,
            "布林收口突破 (BBAND_SQUEEZE_BREAKOUT)": cond_bb_squeeze_breakout,
            "复合底背离 (BOTTOM_DIVERGENCE)": cond_bottom_divergence,
            "资金暗流 (CAPITAL_FLOW_DIVERGENCE)": cond_capital_flow_divergence,
            "老鸭头 (OLD_DUCK_HEAD)": cond_old_duck_head,
            "地天板 (EARTH_HEAVEN_BOARD)": cond_earth_heaven_board,
            "盘整区突破 (CONSOLIDATION_BREAKOUT)": cond_dynamic_box_breakout,
            "上升旗形 (BULLISH_FLAG)": cond_bullish_flag,
            "潜龙在渊 (ENERGY_COMPRESSION_BREAKOUT)": cond_energy_compression_breakout,
            "斐波那契回撤 (PULLBACK_FIBONACCI)": cond_fib_pullback,
            "早晨之星 (KLINE_MORNING_STAR)": cond_morning_star,
            "MACD零轴金叉 (MACD_ZERO_CROSS)": cond_macd_zero_cross,
            "MACD低位金叉 (MACD_LOW_CROSS)": cond_macd_low_cross,
            "DMI金叉 (DMI_CROSS)": cond_dmi_cross,
        }

        # 定义我们关心的调试时间范围
        start_date = pd.to_datetime('2024-09-01').tz_localize('UTC')
        end_date = pd.to_datetime('2024-11-30').tz_localize('UTC')

        for name, condition in playbook_summary.items():
            trigger_count = condition.sum() if hasattr(condition, 'sum') else 0
            print(f"【剧本-{name}】触发天数: {trigger_count}")

            # 只有当剧本至少触发过一次时，才检查特定时间段
            if trigger_count > 0:
                # 筛选出在指定时间范围内触发的日期
                triggered_dates_in_period = condition.index[
                    (condition.index >= start_date) &
                    (condition.index <= end_date) &
                    (condition == True)
                ]

                if not triggered_dates_in_period.empty:
                    # 格式化日期列表以便清晰展示
                    date_list_str = ", ".join([d.strftime('%Y-%m-%d') for d in triggered_dates_in_period])
                    print(f"    -> [24年09-11月触发]: {date_list_str}")
        print("---【日线战术层 - 剧本计算总结结束】---\n")

        # --- 步骤4: 记录所有战术信号得分 ---
        print("    [调试-计分V22.0] 步骤4: 记录日线战术信号得分...")
        def add_score(condition, name, default_score):
            score = points.get(name, {}).get('score', default_score)
            if condition.any():
                score_details_df.loc[condition, name] = score
                print(f"    - [计分-战术分] 剧本 '{name}' 触发 {condition.sum()} 次，计分 {score}。")
            atomic_signals[name] = condition
        add_score(cond_ma_acceleration, 'MA_ACCELERATION', 130)
        add_score(cond_chip_concentration_breakthrough, 'CHIP_CONCENTRATION_BREAKTHROUGH', 180)
        add_score(cond_cost_area_reinforcement, 'COST_AREA_REINFORCEMENT', 160)
        add_score(cond_winner_rate_reversal, 'WINNER_RATE_REVERSAL', 140)
        add_score(cond_chip_pressure_release, 'CHIP_PRESSURE_RELEASE', 150) # 95%线，高分
        add_score(cond_chip_cost_breakthrough, 'CHIP_COST_BREAKTHROUGH', 130)
        add_score(is_steady_climb_pullback, 'PULLBACK_STEADY_CLIMB', 110)
        add_score(cond_chip_hurdle_clear, 'CHIP_HURDLE_CLEAR', 110) # 85%线，分数稍低
        add_score(is_normal_pullback, 'PULLBACK_NORMAL', 100)
        add_score(cond_v_reversal, 'V_SHAPE_REVERSAL', 95)
        add_score(cond_momentum, 'MOMENTUM_BREAKOUT', 70)
        add_score(cond_first_breakout, 'FIRST_BREAKOUT', 90)
        add_score(cond_bb_squeeze_breakout, 'BBAND_SQUEEZE_BREAKOUT', 80)
        add_score(cond_bottom_divergence, 'BOTTOM_DIVERGENCE', 120)
        add_score(cond_bias_reversal, 'BIAS_REVERSAL', 75)
        add_score(cond_washout_reversal, 'WASHOUT_REVERSAL', 115)
        add_score(cond_doji_continuation, 'DOJI_CONTINUATION', 85)
        add_score(cond_old_duck_head, 'OLD_DUCK_HEAD', 120)
        add_score(cond_n_shape_relay, 'N_SHAPE_RELAY', 130)
        add_score(cond_earth_heaven_board, 'EARTH_HEAVEN_BOARD', 200)
        add_score(cond_dynamic_box_breakout, 'CONSOLIDATION_BREAKOUT', 125)
        add_score(cond_bullish_flag, 'BULLISH_FLAG', 110)
        add_score(cond_energy_compression_breakout, 'ENERGY_COMPRESSION_BREAKOUT', 140)
        add_score(cond_capital_flow_divergence, 'CAPITAL_FLOW_DIVERGENCE', 135)
        add_score(cond_relative_strength_maverick, 'RELATIVE_STRENGTH_MAVERICK', 100)
        add_score(cond_fib_pullback, 'PULLBACK_FIBONACCI', points.get('PULLBACK_FIBONACCI', 120))
        add_score(cond_dmi_cross, 'DMI_CROSS', 30)
        add_score(cond_macd_low_cross, 'MACD_LOW_CROSS', 40)
        add_score(cond_macd_zero_cross, 'MACD_ZERO_CROSS', 60)
        add_score(cond_macd_high_cross, 'MACD_HIGH_CROSS', 25)
        add_score(cond_pullback_setup, 'PULLBACK_SETUP', 50)
        add_score(cond_morning_star, 'KLINE_MORNING_STAR', 140)
        add_score(cond_three_soldiers & strict_precondition, 'KLINE_THREE_SOLDIERS', 100)

        # --- 步骤5: 记录协同/冲突规则得分 ---
        print("    [调试-计分V22.0] 步骤5: 记录协同/冲突规则得分...")
        has_positive_score = score_details_df.sum(axis=1) > 0
        add_score(cond_vwap_support & has_positive_score, 'BONUS_VWAP_SUPPORT', points.get('BONUS_VWAP_SUPPORT', 40))
        add_score(cond_cmf_confirm & has_positive_score, 'BONUS_CMF_CONFIRM', points.get('CMF_CONFIRMATION_BONUS', 20))
        add_score(cond_fund_flow_confirm & has_positive_score, 'BONUS_FUND_FLOW_CONFIRM', points.get('FUND_FLOW_CONFIRM_BONUS', 25))
        add_score(cond_turnover_board.shift(1).fillna(False) & has_positive_score, 'BONUS_TURNOVER_BOARD', points.get('TURNOVER_BOARD_BONUS', 45))
        is_any_pullback = is_steady_climb_pullback | is_normal_pullback | cond_v_reversal
        add_score(is_any_pullback & kline_reversal_decent, 'BONUS_PULLBACK_KLINE_DECENT', points.get('PULLBACK_KLINE_DECENT_BONUS', 40))
        add_score(is_any_pullback & kline_reversal_perfect, 'BONUS_PULLBACK_KLINE_PERFECT', points.get('PULLBACK_KLINE_PERFECT_BONUS', 35))
        is_bb_momentum_combo = cond_bb_squeeze_breakout & (cond_momentum | cond_first_breakout)
        add_score(is_bb_momentum_combo, 'BONUS_BB_MOMENTUM_COMBO', points.get('BB_MOMENTUM_COMBO_BONUS', 50))
        is_perfect_entry = is_steady_climb_pullback & cond_macd_zero_cross
        add_score(is_perfect_entry, 'BONUS_STEADY_CLIMB_MACD_ZERO', points.get('STEADY_CLIMB_MACD_ZERO_BONUS', 40))
        current_score = score_details_df.fillna(0).sum(axis=1)
        multiplier_bonus = pd.Series(0.0, index=df.index)
        # 解释: 为了兼容JSON中类似 {"value": 1.2, "说明": "..."} 的格式，
        # 我们先获取原始值，然后判断其是否为字典。如果是，则提取'value'键的值；否则直接使用。
        # 这种方式对所有乘数参数都生效，增强了代码的健壮性。
        # 处理 CMF 乘数
        raw_cmf_multiplier = points.get('CMF_CONFIRMATION_MULTIPLIER', 1.2)
        cmf_multiplier = raw_cmf_multiplier.get('value', 1.2) if isinstance(raw_cmf_multiplier, dict) else raw_cmf_multiplier
        print(f"调试信息: CMF 乘数解析值为: {cmf_multiplier}") # 增加调试信息
        # 处理资金流乘数
        raw_fund_multiplier = points.get('FUND_FLOW_CONFIRM_MULTIPLIER', 1.25)
        fund_multiplier = raw_fund_multiplier.get('value', 1.25) if isinstance(raw_fund_multiplier, dict) else raw_fund_multiplier
        print(f"调试信息: 资金流乘数解析值为: {fund_multiplier}") # 增加调试信息
        # 处理缺口乘数
        raw_gap_multiplier = points.get('GAP_SUPPORT_MULTIPLIER', 1.3)
        gap_multiplier = raw_gap_multiplier.get('value', 1.3) if isinstance(raw_gap_multiplier, dict) else raw_gap_multiplier
        print(f"调试信息: 缺口乘数解析值为: {gap_multiplier}") # 增加调试信息
        multiplier_bonus.loc[cond_cmf_confirm & has_positive_score] += current_score * (cmf_multiplier - 1)
        multiplier_bonus.loc[cond_fund_flow_confirm & has_positive_score] += current_score * (fund_multiplier - 1)
        multiplier_bonus.loc[cond_gap_support_state & has_positive_score] += current_score * (gap_multiplier - 1)
        score_details_df['BONUS_MULTIPLIER'] = multiplier_bonus.where(multiplier_bonus > 0)
        penalty_score = pd.Series(0.0, index=df.index)
        is_reversal_play_conflict = cond_bottom_divergence | cond_bias_reversal
        is_breakout_play_conflict = cond_momentum | cond_first_breakout | cond_bb_squeeze_breakout
        is_conflicting = is_reversal_play_conflict & is_breakout_play_conflict
        # 解释: 再次应用健壮的参数获取模式，以处理JSON中包含说明的字典格式。
        #   1. 先获取原始值，它可能是一个数字，也可能是一个字典。
        #   2. 判断其类型，如果是字典，则提取 'value' 键的值。
        #   3. 最后再进行数学运算。
        raw_penalty = points.get('REVERSAL_BREAKOUT_CONFLICT_PENALTY', 0.8)
        penalty_value = raw_penalty.get('value', 0.8) if isinstance(raw_penalty, dict) else raw_penalty
        print(f"调试信息: 冲突惩罚值解析为: {penalty_value}") # 增加调试信息
        conflict_penalty_rate = 1 - penalty_value
        penalty_score.loc[is_conflicting] -= current_score * conflict_penalty_rate
        score_details_df['PENALTY_CONFLICT'] = penalty_score.where(penalty_score < 0)
        # --- 步骤6: 计算总分并应用最终过滤器 ---
        final_score = score_details_df.fillna(0).sum(axis=1)
        # --- 步骤7: 应用行业强度奖励 ---
        print("    [调试-计分V22.0] 步骤7: 应用行业强度奖励与惩罚...")
        industry_params = self._get_params_block(params, 'industry_context_params', {})
        if industry_params.get('enabled', False) and 'industry_strength_rank_D' in df.columns:
            weak_rank_threshold = industry_params.get('weak_rank_threshold', 0.3)
            weak_penalty_multiplier = industry_params.get('weak_industry_penalty_multiplier', 0.7)
            strength_multiplier_factor = industry_params.get('strength_rank_multiplier', 0.5)
            top_tier_rank_threshold = industry_params.get('top_tier_rank_threshold', 0.9)
            top_tier_bonus = industry_params.get('top_tier_bonus', 30)
            rank_series = df['industry_strength_rank_D']
            has_entry_score = final_score > 0
            multiplier = pd.Series(1.0, index=df.index)
            is_weak = (rank_series < weak_rank_threshold) & has_entry_score
            multiplier.loc[is_weak] = weak_penalty_multiplier
            is_strong = (rank_series >= weak_rank_threshold) & has_entry_score
            multiplier.loc[is_strong] = 1 + (rank_series * strength_multiplier_factor)
            original_score_for_bonus_calc = final_score.copy()
            final_score *= multiplier
            score_change_from_multiplier = final_score - original_score_for_bonus_calc
            score_details_df['INDUSTRY_MULTIPLIER_ADJ'] = score_change_from_multiplier.where(score_change_from_multiplier != 0)
            is_top_tier = (rank_series >= top_tier_rank_threshold) & has_entry_score
            final_score.loc[is_top_tier] += top_tier_bonus
            score_details_df.loc[is_top_tier, 'INDUSTRY_TOP_TIER_BONUS'] = top_tier_bonus
            print(f"    - [行业协同] 已根据行业排名应用奖惩。弱势惩罚({(is_weak).sum()}天), 强势奖励({(is_strong).sum()}天), 龙头奖励({(is_top_tier).sum()}天)。")
        else:
            print("    - [行业协同] 未启用或缺少 'industry_strength_rank_D' 列，跳过此步骤。")
        # 最终风险否决层
        final_score = score_details_df.fillna(0).sum(axis=1)
        final_score.loc[kline_strong_bearish] = 0
        final_score.loc[cond_volume_breakdown] = 0
        final_score.loc[cond_heaven_earth_board] = 0
        # 最终否决：如果一个信号既不满足宽松前提，也不是左侧反转信号，则清零
        is_reversal_play_final = cond_bottom_divergence | cond_bias_reversal | cond_capital_flow_divergence | cond_morning_star | cond_winner_rate_reversal
        final_score = final_score.where(tactical_precondition | is_reversal_play_final, 0)
        print("    [调试-计分V22.1] 计分流程结束。")
        return final_score.round(0), atomic_signals, score_details_df.fillna(0)

    def _get_params_block(self, params: dict, block_name: str, default_return: Any = None) -> dict:
        # 修改行: 增加一个默认返回值，使调用更安全
        if default_return is None:
            default_return = {}
        return params.get('strategy_params', {}).get('trend_follow', {}).get(block_name, default_return)

    def _analyze_dynamic_box_and_ma_trend(self, df: pd.DataFrame, params: dict):
        """
        【V2.18 终极修正版 - 手动应用动态Prominence】
        - 根源修复: 解决了 scipy.find_peaks 无法直接处理动态prominence序列导致的ValueError。
        - 两步法实现:
          1. 先用 find_peaks 找出所有候选波峰/波谷的索引 (不使用prominence参数)。
          2. 再用 peak_prominences 计算这些候选点的实际prominence，并与我们自定义的动态阈值进行手动比较和筛选。
        - 这彻底解决了底层库的限制，确保了动态箱体分析的健壮性和准确性。
        """
        box_params = self._get_params_block(params, 'dynamic_box_params')
        ma_params = self._get_params_block(params, 'mid_term_trend_params')
        
        mid_ma_col = f"EMA_{ma_params.get('mid_ma', 55)}_D"
        slow_ma_col = f"EMA_{ma_params.get('slow_ma', 89)}_D"

        if df.empty or not box_params.get('enabled', False):
            if self.verbose_logging:
                print("调试信息: 动态箱体被禁用或DataFrame为空，执行经典中期趋势判断。")
            if mid_ma_col in df.columns and slow_ma_col in df.columns:
                is_classic_uptrend = (df[mid_ma_col] > df[slow_ma_col]) & (df['close_D'] > df[mid_ma_col])
            else:
                is_classic_uptrend = pd.Series(False, index=df.index)
            df['context_mid_term_bullish'] = is_classic_uptrend
            self.signals['dynamic_box_breakout'] = pd.Series(False, index=df.index)
            self.signals['dynamic_box_breakdown'] = pd.Series(False, index=df.index)
            return
        
        # 1. 计算动态的 prominence 序列 (Series)，而不是标量
        prominence_base = df['close_D']
        peak_prominence_series = prominence_base * box_params.get('peak_prominence', 0.02)
        trough_prominence_series = prominence_base * box_params.get('trough_prominence', 0.02)

                # --- 步骤 2: 【核心修改】手动应用动态 Prominence ---
        
        # 2.1: 找出所有候选波峰，不使用 prominence 参数
        candidate_peak_indices, _ = find_peaks(df['close_D'], distance=box_params.get('peak_distance', 10))
        
        final_peak_indices = []
        if len(candidate_peak_indices) > 0:
            # 2.2: 计算这些候选波峰的实际 prominence
            actual_prominences, _, _ = peak_prominences(df['close_D'], candidate_peak_indices)
            # 2.3: 从我们的动态阈值序列中，提取出这些候选波峰对应的阈值
            custom_thresholds = peak_prominence_series.iloc[candidate_peak_indices]
            # 2.4: 手动比较，筛选出有效的波峰
            valid_peaks_mask = actual_prominences >= custom_thresholds.values
            final_peak_indices = candidate_peak_indices[valid_peaks_mask]

        # 对波谷执行同样的操作
        candidate_trough_indices, _ = find_peaks(-df['close_D'], distance=box_params.get('peak_distance', 10))
        
        final_trough_indices = []
        if len(candidate_trough_indices) > 0:
            actual_prominences, _, _ = peak_prominences(-df['close_D'], candidate_trough_indices)
            custom_thresholds = trough_prominence_series.iloc[candidate_trough_indices]
            valid_troughs_mask = actual_prominences >= custom_thresholds.values
            final_trough_indices = candidate_trough_indices[valid_troughs_mask]

        # --- 步骤 3: 后续逻辑使用最终筛选出的索引 (final_peak_indices, final_trough_indices) ---
        df['last_peak_price'] = np.nan
        if len(final_peak_indices) > 0:
            # 使用 .iloc 和 .columns.get_loc 是最稳健的赋值方式
            df.iloc[final_peak_indices, df.columns.get_loc('last_peak_price')] = df['close_D'].iloc[final_peak_indices]
        df['last_peak_price'].ffill(inplace=True)

        df['last_trough_price'] = np.nan
        if len(final_trough_indices) > 0:
            df.iloc[final_trough_indices, df.columns.get_loc('last_trough_price')] = df['close_D'].iloc[final_trough_indices]
        df['last_trough_price'].ffill(inplace=True)

        # 4. 定义每一天的动态箱体
        box_top = df['last_peak_price']
        box_bottom = df['last_trough_price']
        
        box_height = box_top - box_bottom
        box_midpoint = (box_top + box_bottom) / 2
        box_width_ratio = np.divide(box_height, box_midpoint, out=np.full_like(box_height, np.inf), where=box_midpoint!=0)
        is_valid_box = (box_height > 0) & (box_width_ratio < box_params.get('max_width_ratio', 0.25))

        # 5. 【协同判断】重新定义中期趋势 (context_mid_term_bullish)
        is_classic_uptrend = pd.Series(False, index=df.index)
        if mid_ma_col in df.columns and slow_ma_col in df.columns:
            is_classic_uptrend = (df[mid_ma_col] > df[slow_ma_col]) & (df['close_D'] > df[mid_ma_col])
        
        is_in_box = (df['close_D'] < box_top) & (df['close_D'] > box_bottom)
        is_box_above_ma = box_midpoint > df.get(mid_ma_col, pd.Series(np.inf, index=df.index))
        is_healthy_consolidation = is_valid_box & is_in_box & is_box_above_ma
        
        df['context_mid_term_bullish'] = is_classic_uptrend | is_healthy_consolidation

        # 6. 【信号生成】生成动态箱体的突破和跌破信号
        was_below_top = df['close_D'].shift(1) <= box_top.shift(1)
        breakout_signal = is_valid_box & (df['close_D'] > box_top) & was_below_top
        
        was_above_bottom = df['close_D'].shift(1) >= box_bottom.shift(1)
        breakdown_signal = is_valid_box & (df['close_D'] < box_bottom) & was_above_bottom

        # 7. 将信号存入实例变量，供买卖逻辑使用
        self.signals['dynamic_box_breakout'] = breakout_signal
        self.signals['dynamic_box_breakdown'] = breakdown_signal
        
        # 8. 清理临时列
        df.drop(columns=['last_peak_price', 'last_trough_price'], inplace=True)

    def _check_cmf_confirmation(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【状态确认】检查CMF资金流指标是否为正，以确认买入信号。
        这不是一个直接的买入信号，而是一个用于给其他信号加分或加权的确认状态。
        返回一个布尔序列，表示当天CMF是否处于“资金流入”的确认状态。
        """
        # 从配置中获取参数
        params = self._get_params_block(params, 'cmf_params')
        if not params.get('enabled', True): # 默认启用
            return pd.Series(False, index=df.index)

        # 动态构建CMF列名
        period = params.get('cmf_period', 21) # 默认值也更新为21
        cmf_col = f'CMF_{period}_D'

        # 检查必需的列是否存在，确保健壮性
        if cmf_col not in df.columns:
            # 仅在首次或需要时打印警告，避免刷屏
            if not hasattr(self, '_warned_missing_cmf_col'):
                logger.warning(f"缺少CMF列 '{cmf_col}'，CMF确认功能将不生效。请检查指标计算配置。")
                print(f"columns: {df.columns.to_list()}")
                # 使用一个实例属性来确保警告只打印一次
                self._warned_missing_cmf_col = True
            return pd.Series(False, index=df.index)

        # 获取确认阈值，通常为0
        threshold = params.get('threshold', 0)
        
        # CMF大于阈值（通常为0）表示资金流入，是有效的确认信号
        is_confirmed = df[cmf_col] > threshold
        if self.verbose_logging:
            print(f"调试信息: CMF确认信号触发 {is_confirmed.sum()} 次 (CMF > {threshold})")
        
        return is_confirmed

    def _check_upward_gap_support(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【状态确认】检查近期是否存在未回补的向上跳空缺口。
        这不是一个直接的买入信号，而是一个强烈的趋势确认状态，用于给其他信号加分。
        返回一个布尔序列，表示当天是否处于“缺口支撑”的强势状态下。
        """
        params = self._get_params_block(params, 'gap_support_params')
        if not params.get('enabled', False):
            return pd.Series(False, index=df.index)

        gap_check_days = params.get('gap_check_days', 5)
        
        # 1. 识别所有向上跳空缺口，并记录缺口上沿（支撑位）
        gap_support_level = df['high_D'].shift(1)
        is_gap_up = df['low_D'] > gap_support_level
        
        # 2. 创建一个只在缺口日有值的支撑位序列
        active_support = pd.Series(np.nan, index=df.index)
        active_support.loc[is_gap_up] = gap_support_level[is_gap_up]
        
        # 3. 向前填充支撑位，模拟支撑的持续性，但有时间限制
        active_support = active_support.ffill(limit=gap_check_days)
        
        # 4. 检查从缺口日之后，价格是否始终维持在支撑位之上
        # 使用一个辅助列来判断是否在检查周期内
        is_in_check_period = active_support.notna() & (active_support.diff() != 0).cumsum().ffill().gt(0)
        
        # 检查从缺口日到当前，最低价是否一直高于支撑位
        # 使用 groupby 和 cummin 实现此逻辑
        group = (active_support.diff() != 0).cumsum()
        lowest_since_gap = df.groupby(group)['low_D'].transform('cummin')
        
        is_supported = lowest_since_gap > active_support
        
        return is_supported & is_in_check_period.fillna(False)

    def _identify_board_patterns(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【模块】识别A股特色的各种“板”形态。
        这些是极端市场情绪的体现，拥有高优先级的判断价值。
        返回一个包含各种形态布尔序列的字典。
        """
        params = self._get_params_block(params, 'board_pattern_params')
        if not params.get('enabled', False):
            return {} # 如果禁用，返回空字典

        # 准备基础数据
        prev_close = df['close_D'].shift(1)
        limit_up_threshold = params.get('limit_up_threshold', 0.098)
        limit_down_threshold = params.get('limit_down_threshold', -0.098)
        high_turnover_rate = params.get('high_turnover_rate', 7.0) # 定义高换手率阈值

        # 计算涨跌停价格（近似）
        limit_up_price = prev_close * (1 + limit_up_threshold)
        limit_down_price = prev_close * (1 + limit_down_threshold)

        # 1. 一字板 (Unbroken Board)
        is_limit_up = df['close_D'] >= limit_up_price
        is_limit_down = df['close_D'] <= limit_down_price
        is_one_word_shape = (df['open_D'] == df['high_D']) & (df['high_D'] == df['low_D']) & (df['low_D'] == df['close_D'])
        is_one_word_limit_up = is_limit_up & is_one_word_shape
        
        # 2. 换手板 (Turnover Board)
        is_limit_up_close = (df['close_D'] == df['high_D']) & is_limit_up
        is_opened_during_day = df['open_D'] < df['high_D'] # 盘中开过板
        is_high_turnover = df.get('turnover_rate_D', pd.Series(0, index=df.index)) > high_turnover_rate
        is_turnover_board = is_limit_up_close & is_opened_during_day & is_high_turnover

        # 3. 天地板 (Heaven-Earth Board) - 强力卖出/风险信号
        is_limit_up_high = df['high_D'] >= limit_up_price
        is_limit_down_close = (df['close_D'] == df['low_D']) & is_limit_down
        is_heaven_earth_board = is_limit_up_high & is_limit_down_close

        # 4. 地天板 (Earth-Heaven Board) - 强力买入/反转信号
        is_limit_down_low = df['low_D'] <= limit_down_price
        # is_limit_up_close 已在上面定义
        is_earth_heaven_board = is_limit_down_low & is_limit_up_close

        # 注意：'天地天板'需要分钟级数据才能精确判断，此处基于日线数据不予实现。
        
        return {
            'one_word_limit_up': is_one_word_limit_up,
            'turnover_board': is_turnover_board,
            'heaven_earth_board': is_heaven_earth_board, # 风险信号
            'earth_heaven_board': is_earth_heaven_board, # 机会信号
        }

    # 【模块】主力筹码运作行为识别

    def _find_bullish_flag_entry(self, df: pd.DataFrame, precondition: pd.Series, params: dict) -> pd.Series:
        """
        【剧本】【V1.1 性能重构版】识别“上升三法”（或称旗形整理）。
        此版本重构为完全向量化的实现，消除了性能瓶颈。
        """
        params = self._get_params_block(params, 'bullish_flag_params')
        if not params.get('enabled', False):
            return pd.Series(False, index=df.index)

        pole_threshold = params.get('flagpole_threshold', 0.05)
        max_flag_days = params.get('max_flag_days', 5)
        
        # 重构为向量化实现，避免使用低效的 for 循环
        pct_change = df['close_D'].pct_change()
        is_flagpole = pct_change > pole_threshold

        final_signal = pd.Series(False, index=df.index)

        # 遍历所有可能的旗面天数
        for days in range(1, max_flag_days + 1):
            # 假设今天(T)是突破日，那么旗杆就在 T-days-1 日
            is_pole_day = is_flagpole.shift(days + 1)
            
            # 获取旗杆的相关数据
            pole_open = df['open_D'].shift(days + 1)
            pole_volume = df['volume_D'].shift(days + 1)
            
            # 获取旗面(T-days 到 T-1)和突破日(T)的数据
            flag_period_lows = df['low_D'].shift(1).rolling(window=days).min()
            flag_period_highs = df['high_D'].shift(1).rolling(window=days).max()
            flag_period_volumes = df['volume_D'].shift(1).rolling(window=days).max()

            # 验证旗面条件
            is_supported = flag_period_lows > pole_open
            is_volume_shrunk = flag_period_volumes < pole_volume
            
            # 验证突破日条件
            is_breakout = df['close_D'] > flag_period_highs
            
            # 组合所有条件
            current_signal = (
                is_pole_day &
                is_supported &
                is_volume_shrunk &
                is_breakout
            )
            
            final_signal |= current_signal.fillna(False)

        return final_signal & precondition

    def _find_upthrust_distribution_exit(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【卖出剧本】识别“高位长上影线”，主力隐蔽派发的信号。
        特征：股价处于阶段性高位，当日放出较大成交量，但收盘时留下长长的上影线，
              表明上方抛压沉重，主力拉高出货。
        信号：出现该形态的当天。
        """
        params = self._get_params_block(params, 'upthrust_distribution_params')
        if not params.get('enabled', False):
            return pd.Series(False, index=df.index)

        lookback = params.get('lookback_period', 30)
        upper_shadow_ratio = params.get('upper_shadow_ratio', 0.6) # 上影线占总振幅的比例
        high_vol_quantile = params.get('high_volume_quantile', 0.8) # 成交量分位数

        # 1. 识别是否处于高位
        is_at_highs = df['close_D'] >= df['close_D'].rolling(lookback).quantile(0.9)

        # 2. 识别长上影线
        total_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        upper_shadow = (df['high_D'] - df['close_D']) / total_range
        has_long_upper_shadow = upper_shadow > upper_shadow_ratio

        # 3. 识别高成交量
        is_high_volume = df['volume_D'] > df['volume_D'].rolling(lookback).quantile(high_vol_quantile)

        return is_at_highs & has_long_upper_shadow & is_high_volume

    def _find_volume_breakdown_exit(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【卖出剧本】识别“放量破位”，主力砸盘出货的信号。
        特征：股价跌破关键的中长期支撑均线（如60日线）。
        信号：跌破当天伴随着巨大的成交量，表明是主动性卖出，而非正常回调。
        """
        params = self._get_params_block(params, 'volume_breakdown_params')
        if not params.get('enabled', False):
            return pd.Series(False, index=df.index)

        support_ma_period = params.get('support_ma_period', 60)
        volume_surge_ratio = params.get('volume_surge_ratio', 2.0)
        
        support_ma_col = f"EMA_{support_ma_period}_D"
        vol_ma_col = f"VOL_MA_{params.get('vol_ma_period', 20)}_D"
        if support_ma_col not in df.columns or vol_ma_col not in df.columns:
            return pd.Series(False, index=df.index)

        # 1. 识别破位
        is_breakdown = (df['close_D'] < df[support_ma_col]) & (df['close_D'].shift(1) >= df[support_ma_col].shift(1))
        
        # 2. 识别放量
        is_volume_surge = df['volume_D'] > df[vol_ma_col] * volume_surge_ratio

        return is_breakdown & is_volume_surge

    # 【模块】更隐蔽的主力行为识别 (能量学与博弈论)

    def _find_energy_compression_breakout_entry(self, df: pd.DataFrame, precondition: pd.Series, params: dict) -> pd.Series:
        """
        【剧本】“潜龙在渊” - 识别市场能量极度压缩后的爆发点。
        特征：成交量和波动率（用ATRN衡量）在一段时期内持续萎缩，达到冰点。
        信号：在能量压缩到极致后，出现一根温和放量的中阳线，向上突破。
        """
        params = self._get_params_block(params, 'energy_compression_params')
        if not params.get('enabled', False):
            return pd.Series(False, index=df.index)

        lookback = params.get('lookback_period', 60)
        vol_quantile = params.get('volume_quantile', 0.1) # 成交量处于过去60天10%分位数以下
        atr_quantile = params.get('atr_quantile', 0.1)   # 波动率处于过去60天10%分位数以下
        breakout_vol_ratio = params.get('breakout_volume_ratio', 1.5)
        breakout_pct_change = params.get('breakout_pct_change', 0.03) # 突破日涨幅至少3%

        atr_col = f'ATRN_{params.get("atr_period", 14)}_D'
        vol_ma_col = f'VOL_MA_{params.get("vol_ma_period", 20)}_D'
        if atr_col not in df.columns or vol_ma_col not in df.columns:
            return pd.Series(False, index=df.index)

        # 1. 识别“能量压缩”状态
        is_volume_compressed = df['volume_D'] < df['volume_D'].rolling(lookback).quantile(vol_quantile)
        is_atr_compressed = df[atr_col] < df[atr_col].rolling(lookback).quantile(atr_quantile)
        is_energy_compressed = is_volume_compressed & is_atr_compressed

        # 2. 识别“能量释放”信号
        is_breakout_candle = df['close_D'].pct_change() > breakout_pct_change
        is_breakout_volume = df['volume_D'] > df[vol_ma_col] * breakout_vol_ratio
        is_energy_release = is_breakout_candle & is_breakout_volume

        # 信号：能量释放日的前一天，必须处于“能量压缩”状态
        signal = is_energy_compressed.shift(1).fillna(False) & is_energy_release
        return signal & precondition

    def _find_capital_flow_divergence_entry(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【剧本】【V1.1 逻辑修正版】“资金暗流” - 识别价格与资金指标的底背离。
        特征：股价创出阶段性新低，但OBV(能量潮)指标却未创新低，形成背离。
        信号：在背离形成后的第一个阳线日确认。
        """
        params = self._get_params_block(params, 'capital_flow_divergence_params')
        if not params.get('enabled', False):
            return pd.Series(False, index=df.index)
        
        obv_col = 'OBV_D'
        if obv_col not in df.columns:
            df[obv_col] = (np.sign(df['close_D'].diff()) * df['volume_D']).fillna(0).cumsum()

        lookback = params.get('lookback_period', 30)
        
        price_lows = df['low_D'].rolling(lookback).min()
        is_price_new_low = df['low_D'] == price_lows

        obv_at_price_lows = df[obv_col].where(is_price_new_low)
        last_obv_at_price_lows = obv_at_price_lows.ffill()

        price_makes_new_low = is_price_new_low
        obv_higher_low = df[obv_col] > last_obv_at_price_lows.shift(1)
        
        is_divergence = price_makes_new_low & obv_higher_low
        
        # 修正信号确认逻辑，使其能捕捉到背离当天或次日的阳线信号
        is_green_candle = df['close_D'] > df['open_D']
        # 信号可以发生在背离当天
        signal_today = is_divergence & is_green_candle
        # 或者发生在背离的次日
        signal_next_day = is_divergence.shift(1).fillna(False) & is_green_candle
        # 两者取并集
        signal = signal_today | signal_next_day
        return signal

    def _find_relative_strength_maverick_entry(self, df: pd.DataFrame, precondition: pd.Series, params: dict) -> pd.Series:
        """
        【剧本】“逆市强人” - 识别相对市场表现强势的股票。
        特征：大盘指数下跌，但个股拒绝下跌（横盘或上涨）。
        信号：在大盘企稳或反弹的第一天，这类股票往往率先启动。
        """
        params = self._get_params_block(params, 'relative_strength_params')
        if not params.get('enabled', False):
            return pd.Series(False, index=df.index)

        # 假设df中已合并了大盘指数收盘价，列名为 'index_close'
        index_col = 'index_close'
        if index_col not in df.columns:
            return pd.Series(False, index=df.index)

        lookback = params.get('lookback_period', 5) # 观察期
        
        # 1. 识别“逆市”状态
        is_market_down = (df[index_col].pct_change().rolling(lookback).mean() < 0)
        is_stock_strong = (df['close_D'].pct_change().rolling(lookback).mean() >= 0)
        is_maverick_state = is_market_down & is_stock_strong

        # 2. 识别“大盘企稳”信号
        is_market_rebounding = (df[index_col] > df[index_col].shift(1))

        # 信号：大盘企稳日的前一天，股票处于“逆市强人”状态
        signal = is_maverick_state.shift(1).fillna(False) & is_market_rebounding
        return signal & precondition

    # 专门用于捕捉A股特色的强势启动信号
    def _find_first_breakout_entry(self, df: pd.DataFrame, precondition: pd.Series, params: dict) -> pd.Series:
        """【V2.6 】寻找A股特色的“首板”或强势启动信号。"""
        params = self._get_params_block(params, 'first_breakout_params')
        if not params.get('enabled', False): return pd.Series(False, index=df.index)

        vol_ma_col = f"VOL_MA_{params.get('vol_ma_period', 20)}_D"
        required_cols = ['close_D', 'open_D', 'volume_D', vol_ma_col]
        if not all(col in df.columns for col in required_cols): return pd.Series(False, index=df.index)

        # 条件1: 大阳线或涨停板 (涨幅超过阈值)
        price_increase_ratio = (df['close_D'] - df['open_D']) / df['open_D']
        is_strong_candle = price_increase_ratio > params.get('price_increase_threshold', 0.05)

        # 条件2: 成交量显著放大 (放巨量)
        is_volume_surge = df['volume_D'] > df[vol_ma_col] * params.get('volume_ratio', 2.5)

        # 信号必须在满足大趋势的前提下才有效
        return precondition & is_strong_candle & is_volume_surge

    def _find_pullback_to_ma_entry(self, df: pd.DataFrame, precondition: pd.Series, params: dict) -> pd.Series:
        """
        【V2.20 逻辑与日志修正版】寻找均线回踩买点。
        - 修正了“因果链”逻辑，确保“果”(Effect)是基于严格的过去发生的“因”(Cause)。
        - 增加了对“因”与“果”不能发生在同一天的约束，以捕捉更明确的“先跌后涨”模式。
        - 修正了调试日志，使其不再具有误导性。
        """
        params = self._get_params_block(params, 'pullback_ma_entry_params')
        if not params.get('enabled', False): return pd.Series(False, index=df.index)
        
        support_ma_col = f"EMA_{params.get('support_ma', 20)}_D"
        confirmation_days = params.get('confirmation_days', 2)

        bias_period = self._get_params_block(params, 'bias_reversal_entry_params').get('bias_period', 20)
        bias_col = f"BIAS_{bias_period}_D"

        required_cols = [support_ma_col, 'low_D', 'close_D', 'open_D', bias_col]
        if not all(col in df.columns for col in required_cols):
            return pd.Series(False, index=df.index)

        # 1. 定义“因” (Cause): 健康的回踩触碰支撑
        touches_support = df['low_D'] <= df[support_ma_col]
        bias_lower_bound = params.get('bias_lower_bound', -8.0)
        bias_upper_bound = params.get('bias_upper_bound', 3.0)
        is_healthy_pullback = (df[bias_col] >= bias_lower_bound) & (df[bias_col] <= bias_upper_bound)
        is_cause_day = touches_support & is_healthy_pullback
        
        # 2. 定义“果” (Effect): 确认阳线
        is_effect_day = df['close_D'] > df['open_D']
        
        # 3. 建立“因果”连接：在“果”出现时，回溯【过去】N天内是否存在“因”
        has_recent_cause = is_cause_day.rolling(window=confirmation_days, min_periods=1).sum() > 0
        
        # 4. 生成最终信号：
        # 核心约束：必须是“果”日，且近期有“因”，且“因”和“果”不能是同一天
        # 将 has_recent_cause.shift(1) 应用于最终信号计算，确保因果关系
        final_signal = precondition & is_effect_day & has_recent_cause.shift(1).fillna(False) & ~is_cause_day

        # 修正了调试日志的统计口径，使其更准确、无误导性
        if self.verbose_logging:
            print(f"    [调试-回踩MA]: 长期趋势满足天数: {precondition.sum()} | "
                f"触碰支撑(Cause)天数: {is_cause_day.sum()} | "
                f"确认阳线(Effect)天数: {is_effect_day.sum()} | "
                f"满足'先跌后涨'模式的最终信号: {final_signal.sum()}")

        return final_signal.fillna(False)

    def _find_pullback_to_structure_entry(self, df: pd.DataFrame, precondition: pd.Series, params: dict) -> pd.Series:
        """
        【V2.20 逻辑与日志修正版】寻找前期结构位回踩买点。
        - 修正了“因果链”逻辑，确保“果”(Effect)是基于严格的过去发生的“因”(Cause)。
        - 增加了对“因”与“果”不能发生在同一天的约束。
        - 修正了调试日志，使其不再具有误导性。
        """
        params = self._get_params_block(params, 'pullback_structure_entry_params')
        if not params.get('enabled', False): return pd.Series(False, index=df.index)

        support_col = 'prev_high_support_D'
        confirmation_days = params.get('confirmation_days', 2)

        required_cols = [support_col, 'low_D', 'close_D', 'open_D']
        if not all(col in df.columns for col in required_cols):
            return pd.Series(False, index=df.index)

        # 1. 定义“因” (Cause): 触碰结构支撑
        is_cause_day = df['low_D'] <= df[support_col]
        
        # 2. 定义“果” (Effect): 确认阳线
        is_effect_day = df['close_D'] > df['open_D']
        
        # 3. 建立“因果”连接：在“果”出现时，回溯【过去】N天内是否存在“因”
        has_recent_cause = is_cause_day.rolling(window=confirmation_days, min_periods=1).sum() > 0
        
        # 4. 生成最终信号：
        # 将 has_recent_cause.shift(1) 应用于最终信号计算，确保因果关系
        final_signal = precondition & is_effect_day & has_recent_cause.shift(1).fillna(False) & ~is_cause_day

        # 修正了调试日志的统计口径，使其更准确、无误导性
        if self.verbose_logging:
            print(f"    [调试-回踩结构]: 长期趋势满足天数: {precondition.sum()} | "
                f"触碰支撑(Cause)天数: {is_cause_day.sum()} | "
                f"确认阳线(Effect)天数: {is_effect_day.sum()} | "
                f"满足'先跌后涨'模式的最终信号: {final_signal.sum()}")

        return final_signal.fillna(False)

    def _find_v_reversal_entry(self, df: pd.DataFrame, precondition: pd.Series, params: dict) -> pd.Series:
        """
        【V2.21 形态学重构版】寻找触底当日的“V型反转”买点。
        - 核心逻辑: 从过于严苛的“数学定义”转向经典的“形态学定义”，捕捉“锤子线”神韵。
        - 形态特征: 1.下影线至少是实体的2倍; 2.上影线短于实体; 3.为阳线。
        """
        # 从配置文件中获取参数，如果不存在则使用默认值
        params = self._get_params_block(params, 'v_reversal_entry_params')
        if not params:
            if self.verbose_logging:
                print("    [调试-V型反转-错误]: 无法执行，因为在 config.json 中未找到 'v_reversal_entry_params' 配置块。请确保配置文件已更新。")
            return pd.Series(False, index=df.index)

        if not params.get('enabled', True): return pd.Series(False, index=df.index)
        # 将支撑位检查逻辑解耦，使其更加健壮
        
        # 初始化两个布尔序列，默认为全False
        touches_ma_support = pd.Series(False, index=df.index)
        touches_structure_support = pd.Series(False, index=df.index)

        # 1. 独立检查均线支撑
        support_ma_col = f"EMA_{params.get('support_ma', 20)}_D"
        if support_ma_col in df.columns:
            touches_ma_support = df['low_D'] <= df[support_ma_col]
        else:
            # 仅在首次或需要时打印警告，避免刷屏
            if not hasattr(self, '_warned_missing_ma_support'):
                if self.verbose_logging:
                    print(f"    [调试-V型反转-警告]: 缺少均线支撑列 '{support_ma_col}'，将忽略此支撑类型。")
                self._warned_missing_ma_support = True

        # 2. 独立检查结构支撑
        support_structure_col = 'prev_high_support_D'
        if support_structure_col in df.columns:
            touches_structure_support = df['low_D'] <= df[support_structure_col]
        else:
            if not hasattr(self, '_warned_missing_structure_support'):
                if self.verbose_logging:
                    print(f"    [调试-V型反转-警告]: 缺少结构支撑列 '{support_structure_col}'，将忽略此支撑类型。")
                self._warned_missing_structure_support = True
        
        # 只要触碰到任意一个【存在的】支撑位即可
        touches_any_support = touches_ma_support | touches_structure_support

        # 3. 定义“果” (Effect): V型反转的K线形态 (经典锤子线定义)
        # 检查基础K线数据是否存在
        ohlc_cols = ['low_D', 'high_D', 'open_D', 'close_D']
        if not all(col in df.columns for col in ohlc_cols):
            if self.verbose_logging:
                print(f"    [调试-V型反-错误]: 缺少核心OHLC数据，无法计算K线形态。")
            return pd.Series(False, index=df.index)
            
        body = abs(df['close_D'] - df['open_D']).replace(0, 0.0001) 
        upper_shadow = df['high_D'] - np.maximum(df['open_D'], df['close_D'])
        lower_shadow = np.minimum(df['open_D'], df['close_D']) - df['low_D']

        has_long_lower_shadow = lower_shadow >= (body * params.get('lower_shadow_to_body_ratio', 2.0))
        has_small_upper_shadow = upper_shadow <= (body * params.get('upper_shadow_to_body_ratio', 1.0))
        is_green_candle = df['close_D'] > df['open_D']
        is_v_reversal_shape = has_long_lower_shadow & has_small_upper_shadow & is_green_candle

        # 4. 生成最终信号：满足趋势 + 触碰支撑 + V反形态
        final_signal = precondition & touches_any_support & is_v_reversal_shape

        # 打印详细的调试信息
        if self.verbose_logging:
            print(f"    [调试-V型反转]: 长期趋势满足天数: {precondition.sum()} | "
                f"触碰支撑天数: {touches_any_support.sum()} | "
                f"V反形态(阳线+长下影+短上影): {is_v_reversal_shape.sum()} | "
                f"最终信号: {final_signal.sum()}")
        
        return final_signal.fillna(False)

    def _find_bottom_divergence_entry(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【V4.1 增强调试版】识别“复合底背离”买入信号。
        - 新增: 详细的print语句，用于追踪数据列、参数、中间计算结果，快速定位信号为0的问题。
        """
        # 步骤1: 获取并打印参数
        div_params = self._get_params_block(params, 'bottom_divergence_params')
        if not div_params.get('enabled', False):
            return pd.Series(False, index=df.index)

        # ▼▼▼ 增加详细的调试打印 ▼▼▼
        if self.verbose_logging:
            print("\n--- [调试-复合底背离-启动] ---")

        # 从配置中读取参数
        distance = div_params.get('distance', 5)
        price_prominence = div_params.get('prominence_price', 0.02)
        rsi_col = div_params.get('rsi_col', 'RSI_14_D')
        prominence_rsi = div_params.get('prominence_rsi', 7.5)
        macd_hist_col = div_params.get('macd_hist_col', 'MACD_HIST_ZSCORE_D')
        prominence_macd = div_params.get('prominence_macd_zscore', 0.75)
        confirmation_days = div_params.get('confirmation_days', 3)

        if self.verbose_logging:
            print(f"    [参数检查] distance: {distance}, price_prominence: {price_prominence}")
            print(f"    [参数检查] rsi_col: '{rsi_col}', prominence_rsi: {prominence_rsi}")
            print(f"    [参数检查] macd_hist_col: '{macd_hist_col}', prominence_macd: {prominence_macd}")
            print(f"    [参数检查] confirmation_days: {confirmation_days}")

        # 步骤2: 检查数据列是否存在
        base_cols = ['low_D', 'close_D', 'open_D']
        rsi_col_found = rsi_col in df.columns
        macd_col_found = macd_hist_col in df.columns

        if self.verbose_logging:
            print(f"    [数据列检查] 核心价格列: {'存在' if all(c in df.columns for c in base_cols) else '缺失!'}")
            print(f"    [数据列检查] RSI列 '{rsi_col}': {'存在' if rsi_col_found else '!!! 不存在 !!!'}")
            print(f"    [数据列检查] MACD列 '{macd_hist_col}': {'存在' if macd_col_found else '!!! 不存在 !!!'}")

        if not all(c in df.columns for c in base_cols):
            if self.verbose_logging: print("    [调试终止] 缺少核心价格列，无法执行。")
            return pd.Series(False, index=df.index)
        
        if not rsi_col_found and not macd_col_found:
            if self.verbose_logging: print("    [调试终止] RSI和MACD列均不存在，无法执行。")
            return pd.Series(False, index=df.index)

        # 步骤3: 调用核心算法，计算所有“背离事件”
        divergence_events = pd.Series(False, index=df.index)

        # --- 计算RSI底背离 ---
        if rsi_col_found:
            rsi_trough_params = {"distance": distance, "prominence": price_prominence, "prominence_indicator": prominence_rsi}
            _, rsi_bottom_div = self._find_divergence(df['low_D'], df[rsi_col], {}, rsi_trough_params)
            divergence_events |= rsi_bottom_div
            if self.verbose_logging:
                # ▼▼▼ 增加详细的调试打印 ▼▼▼
                print(f"    [RSI背离检测] 发现 {rsi_bottom_div.sum()} 个RSI背离事件。")

        # --- 计算MACD底背离 ---
        if macd_col_found:
            macd_trough_params = {"distance": distance, "prominence": price_prominence, "prominence_indicator": prominence_macd}
            _, macd_bottom_div = self._find_divergence(df['low_D'], df[macd_hist_col], {}, macd_trough_params)
            divergence_events |= macd_bottom_div
            if self.verbose_logging:
                # ▼▼▼ 增加详细的调试打印 ▼▼▼
                print(f"    [MACD背离检测] 发现 {macd_bottom_div.sum()} 个MACD背离事件。")

        # 步骤4: 应用“因果律”确认逻辑
        total_events = divergence_events.sum()
        if self.verbose_logging:
            # ▼▼▼ 增加详细的调试打印 ▼▼▼
            print(f"    [事件汇总] 总计发现 {total_events} 个复合背离事件。")

        if total_events == 0:
            if self.verbose_logging:
                print("    [调试终止] 未发现任何背离事件，无法生成最终信号。请检查参数是否过于严格或数据本身无此类形态。")
                print("--- [调试-复合底背离-结束] ---\n")
            return pd.Series(False, index=df.index)

        is_green_candle = df['close_D'] > df['open_D']
        has_recent_divergence = divergence_events.rolling(window=confirmation_days, min_periods=1).sum() > 0
        final_signal = is_green_candle & has_recent_divergence.shift(1).fillna(False) & ~divergence_events
        
        # 步骤5: 打印最终结果并返回
        if self.verbose_logging:
            print(f"    [信号确认] 发现 {is_green_candle.sum()} 个阳线。")
            print(f"    [信号确认] 经过'因果律'过滤后，最终生成 {final_signal.sum()} 个买入信号。")
            print("--- [调试-复合底背离-结束] ---\n")
              
        return final_signal.fillna(False)

    def _find_momentum_entry(self, df: pd.DataFrame, precondition: pd.Series, params: dict) -> pd.Series:
        params = self._get_params_block(params, 'momentum_entry_params')
        if not params.get('enabled', False): return pd.Series(False, index=df.index)
        
        vol_ma_col = f"VOL_MA_{params.get('vol_ma_period', 20)}_D"
        required_cols = ['high_D', 'close_D', 'volume_D', vol_ma_col]
        if not all(col in df.columns for col in required_cols): return pd.Series(False, index=df.index)
        
        period_high = df['high_D'].shift(1).rolling(window=params.get('lookback_period', 20)).max()
        is_breakout = df['close_D'] > period_high
        is_volume_surge = df['volume_D'] > df[vol_ma_col] * params.get('volume_ratio', 1.5)
        return precondition & is_breakout & is_volume_surge

    def _find_indicator_entry(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V2.0 重构版】计算所有基于指标的原始买入信号，不应用任何前提条件。
        返回一个包含各种原始信号布尔序列的字典。
        """
        indicator_params = self._get_params_block(params, 'indicator_entry_params')
        if not indicator_params.get('enabled', False):
            return {
                'dmi_cross': pd.Series(False, index=df.index),
                'macd_low_cross': pd.Series(False, index=df.index),
                'macd_zero_cross': pd.Series(False, index=df.index),
                'macd_high_cross': pd.Series(False, index=df.index),
            }

        # --- DMI 金叉 ---
        dmi_params = indicator_params.get('dmi_cross', {})
        dmi_period = self._get_periods_for_timeframe(self.daily_params.get('feature_engineering_params', {}).get('indicators', {}).get('dmi', {}), 'D')
        dmi_cross_signal = pd.Series(False, index=df.index)
        if dmi_params.get('enabled', False) and dmi_period:
            pdi_col, mdi_col = f'DMP_{dmi_period[0]}_D', f'DMN_{dmi_period[0]}_D'
            if pdi_col in df.columns and mdi_col in df.columns:
                dmi_cross_signal = (df[pdi_col] > df[mdi_col]) & (df[pdi_col].shift(1) <= df[mdi_col].shift(1))

        # --- MACD 金叉 ---
        macd_params = indicator_params.get('macd_cross', {})
        macd_periods = self._get_periods_for_timeframe(self.daily_params.get('feature_engineering_params', {}).get('indicators', {}).get('macd', {}), 'D')
        macd_low_cross_signal = pd.Series(False, index=df.index)
        macd_zero_cross_signal = pd.Series(False, index=df.index)
        macd_high_cross_signal = pd.Series(False, index=df.index)
        if macd_params.get('enabled', False) and macd_periods:
            p_fast, p_slow, p_signal = macd_periods
            macd_col = f'MACD_{p_fast}_{p_slow}_{p_signal}_D'
            signal_col = f'MACDs_{p_fast}_{p_slow}_{p_signal}_D'
            if macd_col in df.columns and signal_col in df.columns:
                is_golden_cross = (df[macd_col] > df[signal_col]) & (df[macd_col].shift(1) <= df[signal_col].shift(1))
                
                low_level = macd_params.get('low_level', -0.5)
                high_level = macd_params.get('high_level', 0.5)
                
                macd_low_cross_signal = is_golden_cross & (df[macd_col] < low_level)
                macd_zero_cross_signal = is_golden_cross & (df[macd_col] >= low_level) & (df[macd_col] <= high_level)
                macd_high_cross_signal = is_golden_cross & (df[macd_col] > high_level)

        return {
            'dmi_cross': dmi_cross_signal.fillna(False),
            'macd_low_cross': macd_low_cross_signal.fillna(False),
            'macd_zero_cross': macd_zero_cross_signal.fillna(False),
            'macd_high_cross': macd_high_cross_signal.fillna(False),
        }

    def _find_bband_squeeze_breakout(self, df: pd.DataFrame, precondition: pd.Series, params: dict) -> pd.Series:
        """【V5.0 参数适配版】寻找布林带收缩突破的买点。"""
        bband_params = self._get_params_block(params, 'bband_squeeze_params')
        if not bband_params.get('enabled', False): return pd.Series(False, index=df.index)
        
        fe_params = params.get('feature_engineering_params', {})
        indicators_config = fe_params.get('indicators', {})
        
        # ▼▼▼【修改】: 使用新的辅助函数获取周期参数 ▼▼▼
        # 布林带的参数比较特殊，periods是周期，std是标准差，需要单独处理
        boll_config = indicators_config.get('boll_bands_and_width', {})
        boll_periods_list = self._get_periods_for_timeframe(boll_config, 'D')
        if not boll_periods_list:
            logger.warning("日线BOLL周期未配置，跳过布林带收缩突破检测。")
            return pd.Series(False, index=df.index)
        bb_period = boll_periods_list[0]
        # std 通常不在 periods 里，需要从子配置中单独获取
        bb_std = 2.0 # 默认值
        if 'configs' in boll_config:
            for config_item in boll_config['configs']:
                if 'D' in config_item.get('apply_on', []):
                    bb_std = config_item.get('std_dev', 2.0)
                    break
        else:
            bb_std = boll_config.get('std_dev', 2.0)
        # ▲▲▲【修改】: 修改结束 ▲▲▲

        squeeze_lookback = bband_params.get('squeeze_lookback', 60)

        bbw_col = f'BBW_{bb_period}_{bb_std:.1f}_D'
        bbu_col = f'BBU_{bb_period}_{bb_std:.1f}_D'

        required_cols = [bbw_col, bbu_col, 'close_D']
        if not all(col in df.columns for col in required_cols): return pd.Series(False, index=df.index)

        is_squeeze = df[bbw_col] <= df[bbw_col].rolling(window=squeeze_lookback).min()
        is_breakout = df['close_D'] > df[bbu_col]
        signal = is_breakout & is_squeeze.shift(1).fillna(False)
        return signal & precondition

    # 寻找BIAS超跌反弹买点
    def _find_bias_reversal_entry(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """【V2.10 】寻找BIAS极端超跌后的反弹买点。"""
        params = self._get_params_block(params, 'bias_reversal_entry_params')
        if not params.get('enabled', False): return pd.Series(False, index=df.index)

        bias_col = f"BIAS_{params.get('bias_period', 20)}_D"
        if bias_col not in df.columns: return pd.Series(False, index=df.index)

        # 条件1: BIAS值低于极端阈值，表明严重超卖
        is_extreme_oversold = df[bias_col] < params.get('oversold_threshold', -15.0)
        
        # 条件2: BIAS开始回升（今日BIAS > 昨日BIAS），出现反弹迹象
        is_rebounding = df[bias_col] > df[bias_col].shift(1)

        # 信号：前一日极端超卖，今日开始反弹
        signal = is_extreme_oversold.shift(1).fillna(False) & is_rebounding
        return signal

    # 通用的背离检测逻辑
    def _find_divergence(self, price_series: pd.Series, indicator_series: pd.Series, 
                         find_peaks_params: dict, find_troughs_params: dict) -> Tuple[pd.Series, pd.Series]:
        """
        【V4.0 算法修复版】通用背离检测函数。
        - 核心修复: 解决了prominence参数的“相对/绝对”值混用问题。
        - 动态计算: 能根据价格序列的实际范围，将配置文件中的相对prominence(如0.02)转换为绝对值。
        - 参数兼容: 允许为指标(indicator)单独指定一个绝对的prominence值(prominence_indicator)。
        - 向量化保留: 完全保留了V3.1版本中高效的ffill向量化核心逻辑。
        """
        # 步骤1: 创建一个临时的DataFrame，避免污染原始df
        temp_df = pd.DataFrame({
            'price': price_series,
            'indicator': indicator_series
        })

        # 步骤2: 【核心修复】动态计算并准备 find_peaks 所需的参数
        # ----------------------------------------------------------------
        # 复制参数字典，避免修改传入的原始字典
        peak_params = find_peaks_params.copy()
        trough_params = find_troughs_params.copy()

        # 计算价格序列的波动范围，用于将相对prominence转换为绝对值
        price_range = temp_df['price'].max() - temp_df['price'].min()
        if price_range == 0: price_range = 0.01 # 避免在价格不变时除以零

        # --- 准备价格查找参数 ---
        price_peak_params = peak_params.copy()
        if 'prominence' in price_peak_params:
            # 将价格的相对prominence转换为绝对值
            relative_prom = price_peak_params['prominence']
            price_peak_params['prominence'] = price_range * relative_prom
            if self.verbose_logging:
                print(f"    [调试-背离检测] 价格波峰相对prominence: {relative_prom}, 计算后绝对值: {price_peak_params['prominence']:.4f}")
        price_peak_params.pop('prominence_indicator', None) # 使用 .pop() 安全地移除，如果键不存在也不会报错

        price_trough_params = trough_params.copy()
        if 'prominence' in price_trough_params:
            # 将价格的相对prominence转换为绝对值
            relative_prom = price_trough_params['prominence']
            price_trough_params['prominence'] = price_range * relative_prom
            if self.verbose_logging:
                print(f"    [调试-背离检测] 价格波谷相对prominence: {relative_prom}, 计算后绝对值: {price_trough_params['prominence']:.4f}")
        price_trough_params.pop('prominence_indicator', None) # 使用 .pop() 安全地移除

        # --- 准备指标查找参数 ---
        indicator_peak_params = peak_params.copy()
        if 'prominence_indicator' in indicator_peak_params:
            # 如果为指标指定了独立的prominence，则使用它
            indicator_peak_params['prominence'] = indicator_peak_params.pop('prominence_indicator')
        
        indicator_trough_params = trough_params.copy()
        if 'prominence_indicator' in indicator_trough_params:
            # 如果为指标指定了独立的prominence，则使用它
            indicator_trough_params['prominence'] = indicator_trough_params.pop('prominence_indicator')
        # ----------------------------------------------------------------

        # 步骤3: 使用修复后的参数执行波峰/波谷查找
        # --- 顶背离计算 ---
        price_peaks, _ = find_peaks(temp_df['price'], **price_peak_params)
        indicator_peaks, _ = find_peaks(temp_df['indicator'], **indicator_peak_params)
        
        temp_df['price_at_peak'] = np.nan
        temp_df.iloc[price_peaks, temp_df.columns.get_loc('price_at_peak')] = temp_df['price'].iloc[price_peaks]
        temp_df['indicator_at_peak'] = np.nan
        temp_df.iloc[indicator_peaks, temp_df.columns.get_loc('indicator_at_peak')] = temp_df['indicator'].iloc[indicator_peaks]

        temp_df['last_price_peak'] = temp_df['price_at_peak'].ffill()
        temp_df['last_indicator_peak'] = temp_df['indicator_at_peak'].ffill()

        price_higher_high = temp_df['price_at_peak'] > temp_df['last_price_peak'].shift(1)
        indicator_lower_high = temp_df['indicator_at_peak'] < temp_df['last_indicator_peak'].shift(1)
        top_divergence_signal = temp_df['price_at_peak'].notna() & temp_df['indicator_at_peak'].notna() & price_higher_high & indicator_lower_high

        # --- 底背离计算 ---
        price_troughs, _ = find_peaks(-temp_df['price'], **price_trough_params)
        indicator_troughs, _ = find_peaks(-temp_df['indicator'], **indicator_trough_params)

        temp_df['price_at_trough'] = np.nan
        temp_df.iloc[price_troughs, temp_df.columns.get_loc('price_at_trough')] = temp_df['price'].iloc[price_troughs]
        temp_df['indicator_at_trough'] = np.nan
        temp_df.iloc[indicator_troughs, temp_df.columns.get_loc('indicator_at_trough')] = temp_df['indicator'].iloc[indicator_troughs]

        temp_df['last_price_trough'] = temp_df['price_at_trough'].ffill()
        temp_df['last_indicator_trough'] = temp_df['indicator_at_trough'].ffill()

        price_lower_low = temp_df['price_at_trough'] < temp_df['last_price_trough'].shift(1)
        indicator_higher_low = temp_df['indicator_at_trough'] > temp_df['last_indicator_trough'].shift(1)
        bottom_divergence_signal = temp_df['price_at_trough'].notna() & temp_df['indicator_at_trough'].notna() & price_lower_low & indicator_higher_low
        
        # 步骤4: 返回结果，并确保索引与输入一致
        return top_divergence_signal.set_axis(price_series.index), bottom_divergence_signal.set_axis(price_series.index)
    
    # 寻找复合顶背离卖点
    def _find_top_divergence_exit(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """【V5.0 参数适配版】调用新的辅助函数获取正确的指标周期。"""
        # 简化此函数，使其专注于调用和组合结果
        divergence_params = self._get_params_block(params, 'divergence_params')
        if not divergence_params.get('enabled', False): return pd.Series(False, index=df.index)

        # 动态获取指标列名 
        fe_params = params.get('feature_engineering_params', {})
        indicators_config = fe_params.get('indicators', {})
        macd_config = indicators_config.get('macd', {})
        macd_periods = self._get_periods_for_timeframe(macd_config, 'D')
        if not macd_periods:
            logger.warning("日线MACD周期未配置，跳过顶背离检测。")
            return pd.Series(False, index=df.index)
        fast, slow, signal_line = macd_periods
        macd_hist_col = f"MACDh_{fast}_{slow}_{signal_line}_D"
        
        rsi_config = indicators_config.get('rsi', {})
        rsi_periods_list = self._get_periods_for_timeframe(rsi_config, 'D')
        if not rsi_periods_list:
            logger.warning("日线RSI周期未配置，跳过顶背离检测。")
            return pd.Series(False, index=df.index)
        rsi_period = rsi_periods_list[0]
        rsi_col = f"RSI_{rsi_period}_D"

        if macd_hist_col not in df.columns or rsi_col not in df.columns:
            return pd.Series(False, index=df.index)

        # 定义find_peaks的参数
        find_peaks_params = {
            "distance": params.get('distance', 5),
            "prominence": params.get('prominence_top', 0.1)
        }
        # 底背离参数在此处不需要，传入空字典
        find_troughs_params = {}

        # 调用通用的向量化背离函数
        macd_top_div, _ = self._find_divergence(df['high_D'], df[macd_hist_col], find_peaks_params, find_troughs_params)
        rsi_top_div, _ = self._find_divergence(df['high_D'], df[rsi_col], find_peaks_params, find_troughs_params)

        # 组合信号
        return macd_top_div | rsi_top_div

    def _find_washout_reversal_entry(self, df: pd.DataFrame, precondition: pd.Series, params: dict) -> pd.Series:
        """
        【剧本】【V1.2 逻辑深化版】识别“巨阴洗盘”后的“回马枪”信号。
        放宽了对“反包”的定义，使其更贴近实战。
        """
        params = self._get_params_block(params, 'washout_reversal_params')
        if not params.get('enabled', False):
            return pd.Series(False, index=df.index)

        washout_threshold = params.get('washout_threshold', -0.07)
        vol_ma_period = params.get('vol_ma_period', 20)
        washout_vol_ratio = params.get('washout_volume_ratio', 1.5)
        reversal_rally_threshold = params.get('reversal_rally_threshold', 0.03)
        
        vol_ma_col = f"VOL_MA_{vol_ma_period}_D"
        if vol_ma_col not in df.columns:
            return pd.Series(False, index=df.index)

        # 条件1: 昨天是“放量洗盘日”
        daily_return = df['close_D'].pct_change()
        is_deep_fall_yesterday = daily_return.shift(1) < washout_threshold
        is_volume_surge_yesterday = df['volume_D'].shift(1) > (df[vol_ma_col].shift(1) * washout_vol_ratio)
        is_washout_day = is_deep_fall_yesterday & is_volume_surge_yesterday
        
        # 条件2: 今天是“强势反包日”
        # 放宽反包条件：收盘价高于昨日收盘价即可，更符合实战中的“反包”定义
        is_reversal_cover = df['close_D'] > df['close_D'].shift(1)
        reversal_rally = (df['close_D'] - df['open_D']) / df['open_D'].replace(0, np.nan)
        is_strong_reversal_candle = reversal_rally > reversal_rally_threshold
        is_reversal_day = is_reversal_cover & is_strong_reversal_candle
        
        signal = precondition & is_washout_day & is_reversal_day
        return signal.fillna(False)

    def _find_doji_continuation_entry(self, df: pd.DataFrame, precondition: pd.Series, params: dict) -> pd.Series:
        """
        【剧本】【V1.1 逻辑深化版】识别上涨途中的“十字星”中继信号。
        在上涨趋势中，出现一根缩量十字星（分歧），如果次日能放量突破十字星的高点，
        则意味着分歧转一致，是趋势延续的强信号。
        """
        params = self._get_params_block(params, 'doji_continuation_params')
        if not params.get('enabled', False):
            return pd.Series(False, index=df.index)

        body_ratio_threshold = params.get('body_ratio_threshold', 0.15)
        # 引入成交量比率参数
        confirmation_vol_ratio = params.get('confirmation_volume_ratio', 1.2) # 确认日成交量至少是十字星日的1.2倍
        
        # 计算昨天的K线形态
        prev_body = abs(df['open_D'].shift(1) - df['close_D'].shift(1))
        prev_range = (df['high_D'].shift(1) - df['low_D'].shift(1)).replace(0, np.nan)
        
        # 条件1: 昨天是“十字星”
        is_doji_day = (prev_body / prev_range) < body_ratio_threshold
        
        # 条件2: 今天是“确认日”
        is_price_confirmation = df['close_D'] > df['high_D'].shift(1)
        # 【】确认日必须放量，体现分歧转一致
        is_volume_confirmation = df['volume_D'] > (df['volume_D'].shift(1) * confirmation_vol_ratio)
        is_confirmation_day = is_price_confirmation & is_volume_confirmation
        
        # 最终信号：必须满足大趋势(precondition)，且昨天是十字星，今天是放量确认日
        signal = precondition & is_doji_day & is_confirmation_day
        return signal.fillna(False)

    def _find_old_duck_head_entry(self, df: pd.DataFrame, precondition: pd.Series, params: dict) -> pd.Series:
        """
        【剧本】【V1.1 性能重构版】识别“老鸭头”形态，一个经典的主力深度控盘后的二次启动信号。
        此版本重构为完全向量化的实现，消除了性能瓶颈。
        """
        params = self._get_params_block(params, 'old_duck_head_params')
        if not params.get('enabled', False):
            return pd.Series(False, index=df.index)

        peak_lookback = params.get('peak_lookback', 60)
        neck_ma_period = params.get('neck_ma_period', 60)
        volume_quantile = params.get('volume_quantile', 0.1)
        breakout_volume_ratio = params.get('breakout_volume_ratio', 1.5)
        volume_ma_period = params.get('volume_ma_period', 20)

        neck_ma_col = f"EMA_{neck_ma_period}_D"
        vol_ma_col = f"VOL_MA_{volume_ma_period}_D"
        if neck_ma_col not in df.columns or vol_ma_col not in df.columns:
            return pd.Series(False, index=df.index)

        # 重构为完全向量化的逻辑，避免使用低效的 for 循环
        
        # 步骤1: 找到每个时间点之前的“鸭头顶”高点
        # 使用 rolling().max() 来高效地找到每个点之前的 peak_lookback 周期内的高点
        prev_highs = df['high_D'].shift(1).rolling(window=peak_lookback, min_periods=5).max()

        # 步骤2: 识别“突破日” (鸭嘴部)
        is_breakout_today = df['close_D'] > prev_highs
        is_volume_surge_today = df['volume_D'] > (df[vol_ma_col] * breakout_volume_ratio)
        is_potential_signal_day = is_breakout_today & is_volume_surge_today

        # 步骤3: 识别“鸭颈部”特征 (回调期)
        # 这是一个复杂的部分，我们用一种近似但高效的向量化方法来识别
        # 条件a: 回调期间，最低价始终在颈线(neck_ma_col)之上
        # 我们检查突破日之前的 N 天 (例如10天) 是否满足此条件
        neck_check_days = params.get('neck_check_days', 10)
        neck_supported = (df['low_D'].shift(1).rolling(window=neck_check_days).min() > df[neck_ma_col].shift(1).rolling(window=neck_check_days).max())
        
        # 条件b: 回调期间，出现过极度缩量 (芝麻量)
        volume_threshold = df['volume_D'].shift(1).rolling(window=peak_lookback).quantile(volume_quantile)
        is_volume_shrunk = (df['volume_D'].shift(1).rolling(window=neck_check_days).min() < volume_threshold)

        # 步骤4: 组合所有条件
        # 只有在可能是信号日的日子，我们才需要检查其之前的颈部特征
        final_signal = is_potential_signal_day & neck_supported & is_volume_shrunk
        
        return final_signal.fillna(False) & precondition

    def _find_n_shape_relay_entry(self, df: pd.DataFrame, precondition: pd.Series, params: dict) -> pd.Series:
        """
        【剧本】【V1.2 涨停识别修正版】识别“N字板”接力。
        修正了对“涨停/大阳线”的识别逻辑，使其能正确捕捉包括一字板、T字板在内的所有强势形态。
        """
        params = self._get_params_block(params, 'n_shape_relay_params')
        if not params.get('enabled', False):
            return pd.Series(False, index=df.index)

        rally_threshold = params.get('rally_threshold', 0.097)
        consolidation_days_max = params.get('consolidation_days_max', 3)
        hold_above_key = params.get('hold_above_key', 'open_D')

        # 采用更鲁棒的“拉升日”识别逻辑，并保持向量化实现

        # 步骤1: 识别所有可能的“拉升日” (启动板和接力板)
        # 条件a: K线实体涨幅够大 (捕捉大阳线)
        body_rally = (df['close_D'] - df['open_D']) / df['open_D'].replace(0, np.nan)
        is_strong_body = body_rally > params.get('body_rally_threshold', 0.05) # 参数，默认为5%
        
        # 条件b: 日涨幅够大 (捕捉各类涨停板)
        close_to_close_rally = df['close_D'].pct_change()
        is_limit_up_rally = close_to_close_rally > rally_threshold

        # 最终的“拉升日”定义：满足任一条件即可
        is_rally_day = is_strong_body | is_limit_up_rally

        final_signal = pd.Series(False, index=df.index)

        # 步骤2: 遍历所有可能的整理周期 (1天到consolidation_days_max天)
        for days in range(1, consolidation_days_max + 1):
            # 假设今天(T)是接力板，那么启动板就在 T-days-1 日
            is_second_rally = is_rally_day
            is_first_rally = is_rally_day.shift(days + 1)

            # 获取启动板和整理期的相关数据
            first_rally_high = df['high_D'].shift(days + 1)
            first_rally_support = df[hold_above_key].shift(days + 1)
            first_rally_volume = df['volume_D'].shift(days + 1)

            # 步骤3: 验证整理期 (T-days 到 T-1)
            consolidation_lows = df['low_D'].shift(1).rolling(window=days).min()
            consolidation_volumes = df['volume_D'].shift(1).rolling(window=days).max()

            is_supported = consolidation_lows > first_rally_support
            is_volume_shrunk = consolidation_volumes < first_rally_volume
            
            # 步骤4: 验证接力板(今天)是否突破了启动板的高点
            is_breakout = df['close_D'] > first_rally_high

            # 步骤5: 组合所有条件
            current_signal = (
                is_first_rally & 
                is_supported & 
                is_volume_shrunk & 
                is_second_rally & 
                is_breakout
            )
            
            final_signal |= current_signal.fillna(False)

        return final_signal & precondition

    # 专门用于识别“回撤预备”信号
    def _find_pullback_setup(self, df: pd.DataFrame, precondition: pd.Series, params: dict) -> Tuple[pd.Series, pd.Series]:
        """
        【剧本】识别“回撤预备”信号，用于次日盘中监控。
        返回:
            - is_setup (pd.Series): 当天是否处于回撤预备状态的布尔序列。
            - target_price (pd.Series): 对应的目标回撤价位序列。
        """
        params = self._get_params_block(params, 'pullback_ma_entry_params')
        if not params.get('enabled', False):
            return pd.Series(False, index=df.index), pd.Series(np.nan, index=df.index)

        support_ma_col = f"EMA_{params.get('support_ma', 20)}_D"
        if support_ma_col not in df.columns:
            return pd.Series(False, index=df.index), pd.Series(np.nan, index=df.index)

        # 预备状态的定义：
        # 1. 收盘价在支撑均线之上
        is_above_support = df['close_D'] > df[support_ma_col]
        
        # 2. 收盘价距离支撑均线足够近
        proximity_threshold = params.get('setup_proximity_threshold', 0.03) # 配置项，例如3%
        is_close_to_support = (df['close_D'] / df[support_ma_col] - 1) < proximity_threshold

        # 最终的预备信号：满足大趋势 + 在支撑之上 + 距离支撑足够近
        is_setup = precondition & is_above_support & is_close_to_support
        
        # 目标价位就是当天的支撑均线价格，只在预备日有效
        target_price = df[support_ma_col].where(is_setup)

        return is_setup, target_price

    def _find_chip_cost_breakthrough(self, df: pd.DataFrame, precondition: pd.Series, params: dict) -> pd.Series:
        """【新剧本】【V2.1 列名修复版】筹码成本突破：股价上穿市场平均成本。"""
        params = self._get_params_block(params, 'chip_cost_breakthrough_params')
        close_col = 'close_D'
        weight_avg_col = 'weight_avg_D'
        if not params.get('enabled', False) or close_col not in df.columns or weight_avg_col not in df.columns:
            return pd.Series(False, index=df.index)
        
        signal = (df[close_col] > df[weight_avg_col]) & (df[close_col].shift(1) <= df[weight_avg_col].shift(1))

        return signal & precondition

    def _find_chip_pressure_release(self, df: pd.DataFrame, precondition: pd.Series, params: dict) -> pd.Series:
        """【新剧本】【V2.1 列名修复版】筹码压力释放：股价突破95%的套牢盘成本线。"""
        params = self._get_params_block(params, 'chip_pressure_release_params')
        close_col = 'close_D'
        cost_95pct_col = 'cost_95pct_D'
        if not params.get('enabled', False) or close_col not in df.columns or cost_95pct_col not in df.columns:
            return pd.Series(False, index=df.index)
            
        signal = df[close_col] > df[cost_95pct_col]

        return signal & precondition
    
    # ▼▼▼ 突破85%成本线 ▼▼▼
    def _find_chip_hurdle_clear_entry(self, df: pd.DataFrame, precondition: pd.Series, params: dict) -> pd.Series:
        """【新增剧本】筹码关口扫清：股价突破85%的套牢盘成本线，作为趋势确认信号。"""
        # 注意：这里我们复用 pressure_release 的参数块，或者你可以为其新建一个参数块
        params = self._get_params_block(params, 'chip_pressure_release_params') # 假设复用参数
        close_col = 'close_D'
        cost_85pct_col = 'cost_85pct_D'
        if not params.get('enabled', False) or close_col not in df.columns or cost_85pct_col not in df.columns:
            return pd.Series(False, index=df.index)
        
        signal = (df[close_col] > df[cost_85pct_col]) & (df[close_col].shift(1) <= df[cost_85pct_col].shift(1))

        return signal & precondition

    # ▼▼▼ “成本区增强”剧本(使用精确CYQ数据) ▼▼▼
    def _find_cost_area_reinforcement_entry(self, df: pd.DataFrame, precondition: pd.Series, params: dict) -> pd.Series:
        """
        【剧本】【V1.6-资金验证版】成本区增强
        在V1.5的基础上，加入“主力资金净流入”作为最终的确认条件，确保信号的含金量。
        这要求“放量横盘”必须是由主力主动买入造成的，而非其他类型的换手。
        """
        # 从主配置中获取本剧本的参数块
        params = self._get_params_block(params, 'cost_area_reinforcement_params')
        if not params.get('enabled', False):
            return pd.Series(False, index=df.index)

        # 动态获取成交量均线周期
        fe_params = self.daily_params.get('feature_engineering_params', {})
        vol_ma_config = fe_params.get('indicators', {}).get('vol_ma', {})
        vol_ma_period = vol_ma_config.get('periods', [21])[0]
        vol_ma_col = f"VOL_MA_{vol_ma_period}_D"
        
        weight_avg_col = 'weight_avg_D'
        # ▼▼▼ 增加 net_mf_amount_D 到依赖检查中 ▼▼▼
        required_cols = [weight_avg_col, 'close_D', 'volume_D', vol_ma_col, 'net_mf_amount_D']
        
        # 依赖检查
        if any(col not in df.columns or df[col].isna().all() for col in required_cols):
            # 为了避免日志刷屏，只在第一次或必要时打印
            # print(f"成本区增强剧本依赖缺失，跳过执行。")
            return pd.Series(False, index=df.index)

        # 1. 参数获取
        proximity_threshold = params.get('proximity_threshold', 0.03)
        volume_ratio = params.get('volume_ratio', 1.5)
        sideways_threshold = params.get('sideways_threshold', 0.015)

        # 2. 计算各个条件
        # 条件A: 当前收盘价非常接近市场核心成本区
        main_cost_area = df[weight_avg_col]
        is_price_near_cost_area = (df['close_D'] >= main_cost_area * (1 - proximity_threshold)) & \
                                  (df['close_D'] <= main_cost_area * (1 + proximity_threshold))
        
        # 条件B: 当日成交量显著放大 (有大量换手)
        is_volume_surged = df['volume_D'] > (df[vol_ma_col] * volume_ratio)
        
        # 条件C: 当日收盘价涨跌幅极小，表明是“横盘”换手
        daily_pct_change = df['close_D'].pct_change().abs()
        is_sideways_consolidation = daily_pct_change < sideways_threshold
        
        # ▼▼▼ 增加全新的“资金流”验证条件 ▼▼▼
        # 条件D: 当日主力资金必须是净流入状态
        is_main_force_inflow = df['net_mf_amount_D'] > 0
        
        # ▼▼▼ 在最终信号中加入资金流验证 ▼▼▼
        # 3. 组合最终信号：价格接近成本 + 放量 + 横盘 + 主力净流入
        final_signal = is_price_near_cost_area & is_volume_surged & is_sideways_consolidation & is_main_force_inflow

        # (可选的调试日志) 如果需要再次调试，可以取消下面的注释
        # print("\n--- [调试日志 - 剧本: 成本区增强 (COST_AREA_REINFORCEMENT) V1.6] ---")
        # debug_dates = df[(df.index >= '2024-01-01') & (df.index <= '2024-12-31')].index
        # for date in debug_dates:
        #     if final_signal.get(date, False):
        #         close = df.loc[date, 'close_D']
        #         cost = df.loc[date, weight_avg_col]
        #         vol = df.loc[date, 'volume_D']
        #         vol_ma = df.loc[date, vol_ma_col]
        #         pct_chg = daily_pct_change.loc[date]
        #         net_mf = df.loc[date, 'net_mf_amount_D']
        #         print(f"    - {date.date()}: 信号触发!")
        #         print(f"      -> 数据: 收盘价={close:.2f}, 成本={cost:.2f}, 日涨跌幅={pct_chg:.3%}, 主力净流入={net_mf:,.0f}")
        #         print(f"      -> 条件: 接近成本=True, 放量=True, 横盘=True, 资金流入=True")
        # print(f"  - [最终总结]: 总计满足 [全部条件] 天数: {final_signal.sum()}")
        # print("--- [调试日志结束] ---\n")

        return final_signal.fillna(False)

    # ▼▼▼ “筹码高度集中突破”剧本 ▼▼▼
    def _find_chip_concentration_breakthrough_entry(self, df: pd.DataFrame, precondition: pd.Series, params: dict) -> pd.Series:
        """
        【剧本】【V2.1 列名修复版】识别“筹码高度集中后突破”。
        - 新增: 可通过JSON配置选择使用[5, 95]或[15, 85]等不同百分位来计算集中度。
        """
        # 从JSON获取此剧本的专属配置
        params = self._get_params_block(params, 'chip_concentration_breakthrough_params')
        if not params.get('enabled', False):
            return pd.Series(False, index=df.index)

        # 从配置中读取参数
        threshold = params.get('concentration_threshold', 0.1)
        # 从配置中读取百分位，并提供一个健壮的默认值
        percentiles = params.get('concentration_percentiles', [15, 85])
        if len(percentiles) != 2: # 安全检查
            percentiles = [15, 85]
        
        lower_pct, upper_pct = min(percentiles), max(percentiles)

        # 动态构建依赖的列名
        cost_lower_col = f'cost_{lower_pct}pct' # 修正列名
        cost_upper_col = f'cost_{upper_pct}pct' # 修正列名
        weight_avg_col = 'weight_avg' # 修正列名
        
        required_cols = [cost_lower_col, cost_upper_col, weight_avg_col, 'close_D']
        if not all(col in df.columns for col in required_cols):
            if self.verbose_logging:
                print(f"    [调试-筹码集中-警告]: 缺少必需的筹码列: {required_cols}，剧本跳过。")
            return pd.Series(False, index=df.index)

        # 1. 计算筹码集中度 (Concentration Ratio)
        # 使用加权平均成本作为分母，比收盘价更稳定，更能反映真实成本
        chip_range = df[cost_upper_col] - df[cost_lower_col]
        # 防止分母为0
        concentration_ratio = chip_range / df[weight_avg_col].replace(0, np.nan)

        # 2. 识别“高度集中”状态
        is_concentrated = concentration_ratio < threshold

        # 3. 识别“向上突破”事件
        # 突破的定义是：收盘价向上突破了集中区的上轨
        was_below_upper_cost = df['close_D'].shift(1) <= df[cost_upper_col].shift(1)
        is_breakout = df['close_D'] > df[cost_upper_col]
        
        # 信号：突破日的前一天，必须处于“高度集中”状态
        signal = is_concentrated.shift(1).fillna(False) & is_breakout & was_below_upper_cost

        # 最终信号必须满足外部传入的严格前提条件
        final_signal = signal & precondition

        if self.verbose_logging and final_signal.any():
            print(f"    [调试-筹码集中]: 使用 {lower_pct}/{upper_pct} 百分位，剧本触发 {final_signal.sum()} 次。")

        return final_signal.fillna(False)
    
    # ▼▼▼ “获利盘洗净反转”剧本 ▼▼▼
    def _find_winner_rate_reversal_entry(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【全新剧本】【V1.1 列名修复版】识别“获利盘洗净反转”信号 (或称“投降坑”反转)。
        当获利盘比例极低（市场绝望），随后出现企稳阳线时，视为底部反转信号。
        """
        # 从主配置中获取本剧本的参数块
        params = self._get_params_block(params, 'winner_rate_reversal_params')
        if not params.get('enabled', False):
            return pd.Series(False, index=df.index)
        # 检查必需的CYQ数据列是否存在
        winner_rate_col = 'winner_rate_D'
        close_col = 'close_D'
        open_col = 'open_D'
        required_cols = [winner_rate_col, close_col, open_col]
        if not all(col in df.columns and df[col].notna().any() for col in required_cols):
            if self.verbose_logging:
                print(f"    [调试-获利盘洗净反转]: 跳过，缺少必需的CYQ列: {required_cols}")
            return pd.Series(False, index=df.index)
        # 1. 条件A: 前一日获利盘比例极低（市场已完成“投降式”抛售）
        # 注意：winner_rate的单位是%，所以阈值也要是%
        was_washed_out = df[winner_rate_col].shift(1) < params.get('washout_threshold', 10.0)
        # 2. 条件B: 当日为企稳阳线
        is_reversal_candle = df['close_D'] > df['open_D']
        # 这个信号是典型的左侧交易信号，可以不依赖于严格的上升趋势前提(precondition)
        final_signal = was_washed_out & is_reversal_candle

        if self.verbose_logging and final_signal.any():
            print(f"    [调试-获利盘洗净反转]: 剧本触发 {final_signal.sum()} 次。")
        return final_signal.fillna(False)

    def _find_fibonacci_pullback_entry(self, df: pd.DataFrame, precondition: pd.Series, params: dict) -> pd.Series:
        fib_params = self._get_params_block(params, 'fibonacci_pullback_params')
        if not fib_params.get('enabled', False):
            return pd.Series(False, index=df.index)
        print("    - [计分-战术] 正在执行斐波那契回撤买入剧本...")
        retr_levels = fib_params.get('retracement_levels', [0.618])
        buffer = fib_params.get('pullback_buffer', 0.01)
        final_signal = pd.Series(False, index=df.index)
        for level in retr_levels:
            col_name = f'fib_retr_{str(level).replace(".", "")}_D'
            if col_name not in df.columns:
                continue
            support_price = df[col_name]
            touched_support = df['low_D'] <= support_price * (1 + buffer)
            recovered_above_support = df['close_D'] > support_price
            signal = touched_support & recovered_above_support & precondition
            final_signal |= signal
        return final_signal.fillna(False)

    # ▼▼▼“均线加速上涨”剧本方法 ▼▼▼
    def _find_ma_acceleration_entry(self, df: pd.DataFrame, precondition: pd.Series, params: dict) -> pd.Series:
        """
        【新增剧本】“均线加速” - 识别趋势动量增强的爆发点。
        特征：均线的斜率(一阶导数)为正，且斜率的斜率(二阶导数)也为正。
        信号：在满足趋势前提下，均线加速上涨的当天。
        """
        # 从JSON获取此剧本的专属配置
        params = self._get_params_block(params, 'ma_acceleration_playbook')
        if not params.get('enabled', False):
            return pd.Series(False, index=df.index)

        # 从配置中读取参数，不再硬编码
        ma_period = params.get('ma_period', 21)
        timeframe = params.get('timeframe', 'D')
        confirm_close_above = params.get('confirmation_close_above_ma', True)

        # 构建列名
        ema_col = f'EMA_{ma_period}_{timeframe}'
        close_col = f'close_{timeframe}'
        # 使用独立后缀避免与其他剧本的列名冲突
        slope_col = f'{ema_col}_slope_for_accel'
        accel_col = f'{ema_col}_acceleration'

        # 检查依赖列是否存在
        if ema_col not in df.columns or close_col not in df.columns:
            if self.verbose_logging:
                print(f"    [调试-均线加速-警告]: 缺少必需列 '{ema_col}' 或 '{close_col}'，剧本跳过。")
            return pd.Series(False, index=df.index)

        # 核心计算逻辑
        # 1. 计算斜率 (一阶导数)
        df[slope_col] = df[ema_col].diff(1)
        # 2. 计算加速度 (二阶导数)
        df[accel_col] = df[slope_col].diff(1)

        # 定义信号条件
        # 条件1: 均线在上涨 (斜率 > 0)
        condition1 = df[slope_col] > 0
        # 条件2: 上涨在加速 (加速度 > 0)
        condition2 = df[accel_col] > 0
        
        final_signal = condition1 & condition2

        # 可选的确认条件
        if confirm_close_above:
            condition3 = df[close_col] > df[ema_col]
            final_signal &= condition3
        
        # 最终信号必须满足外部传入的严格前提条件
        final_signal &= precondition

        if self.verbose_logging and final_signal.any():
            print(f"    [调试-均线加速]: 剧本触发 {final_signal.sum()} 次。")

        return final_signal.fillna(False)

    # 引入状态机，重构止盈逻辑以支持连续交易模拟
    def _apply_take_profit_rules(self, df: pd.DataFrame, entry_signals: pd.Series, top_divergence_signals: pd.Series, params: dict) -> pd.Series:
        """
        【V22.0 调试增强版】
        """
        final_tp_signal = pd.Series(0, index=df.index, dtype=int)
        tp_params = self._get_params_block(params, 'take_profit_params')
        
        print("    - [止盈-调试] 步骤1: 计算所有潜在止盈条件...")
        tp_signal_resistance = self._check_resistance_take_profit(df, tp_params.get('resistance_exit', {}))
        tp_signal_trailing = self._check_trailing_stop_take_profit(df, tp_params.get('trailing_stop_exit', {}))
        tp_signal_indicator = self._check_indicator_take_profit(df, tp_params.get('indicator_exit', {}))
        board_patterns = self._identify_board_patterns(df, params)
        tp_signal_heaven_earth = board_patterns.get('heaven_earth_board', pd.Series(False, index=df.index))
        tp_signal_upthrust = self._find_upthrust_distribution_exit(df, params)
        tp_signal_breakdown = self._find_volume_breakdown_exit(df, params)
        box_params = self._get_params_block(params, 'dynamic_box_params')
        tp_signal_box_breakdown = pd.Series(False, index=df.index)
        if box_params.get('breakdown_sell_enabled', True):
            tp_signal_box_breakdown = self.signals.get('dynamic_box_breakdown', pd.Series(False, index=df.index))

        fib_params = self._get_params_block(params, 'fibonacci_analysis_params')
        tp_signal_fib_extension = pd.Series(False, index=df.index)
        if fib_params.get('enabled', False):
            ext_levels = fib_params.get('extension_levels', [1.618])
            if ext_levels:
                first_target_level = ext_levels[0]
                col_name = f'fib_ext_{str(first_target_level).replace(".", "")}_D'
                if col_name in df.columns:
                    target_price = df[col_name]
                    hit_target = (df['high_D'] >= target_price) & target_price.notna()
                    tp_signal_fib_extension[hit_target] = True
        
        # ▼▼▼ 增加详细的止盈条件触发统计 ▼▼▼
        print("    - [止盈-调试] 原始止盈条件触发统计:")
        print(f"      - 阻力位(1): {tp_signal_resistance.sum()} 次")
        print(f"      - 移动止损(2): {tp_signal_trailing.sum()} 次")
        print(f"      - 指标超买(3/5): {(tp_signal_indicator > 0).sum()} 次")
        print(f"      - 顶背离(4): {top_divergence_signals.sum()} 次")
        print(f"      - 天地板(6): {tp_signal_heaven_earth.sum()} 次")
        print(f"      - 高位派发(7): {tp_signal_upthrust.sum()} 次")
        print(f"      - 放量破位(8): {tp_signal_breakdown.sum()} 次")
        print(f"      - 箱体跌破(9): {tp_signal_box_breakdown.sum()} 次")
        print(f"      - 斐波那契扩展(10): {tp_signal_fib_extension.sum()} 次")

        any_tp_signal = (
            tp_signal_fib_extension | tp_signal_breakdown | tp_signal_heaven_earth |
            tp_signal_upthrust | tp_signal_box_breakdown | top_divergence_signals |
            (tp_signal_indicator > 0) | tp_signal_trailing | tp_signal_resistance
        )

        print("    - [止盈-调试] 步骤2: 使用向量化状态机判断持仓...")
        is_new_entry = entry_signals & ~any_tp_signal.shift(1).fillna(False)
        is_exit = any_tp_signal
        trade_block_id = is_new_entry.cumsum()
        first_exit_in_block = is_exit & (is_exit.groupby(trade_block_id).cumcount() == 0)
        reset_points = first_exit_in_block.cumsum()
        active_trade_block_id = trade_block_id - reset_points.shift(1).fillna(0)
        is_holding_vectorized = active_trade_block_id > 0
        print(f"    - [止盈-调试] 识别出 {is_new_entry.sum()} 个交易区块，最终持仓天数共 {is_holding_vectorized.sum()} 天。")

        print("    - [止盈-调试] 步骤3: 应用优先级规则生成最终止盈信号...")
        final_tp_signal.loc[is_holding_vectorized & tp_signal_fib_extension] = 10
        final_tp_signal.loc[(final_tp_signal == 0) & is_holding_vectorized & tp_signal_breakdown] = 8
        final_tp_signal.loc[(final_tp_signal == 0) & is_holding_vectorized & tp_signal_heaven_earth] = 6
        final_tp_signal.loc[(final_tp_signal == 0) & is_holding_vectorized & tp_signal_upthrust] = 7
        final_tp_signal.loc[(final_tp_signal == 0) & is_holding_vectorized & tp_signal_box_breakdown] = 9
        final_tp_signal.loc[(final_tp_signal == 0) & is_holding_vectorized & top_divergence_signals] = 4
        indicator_tp_mask = (final_tp_signal == 0) & is_holding_vectorized & (tp_signal_indicator > 0)
        final_tp_signal.loc[indicator_tp_mask] = tp_signal_indicator[indicator_tp_mask]
        final_tp_signal.loc[(final_tp_signal == 0) & is_holding_vectorized & tp_signal_trailing] = 2
        final_tp_signal.loc[(final_tp_signal == 0) & is_holding_vectorized & tp_signal_resistance] = 1

        print(f"    - [止盈-调试] 止盈逻辑判断完成，共产生 {(final_tp_signal > 0).sum()} 个最终止盈信号。")
        return final_tp_signal

    def _check_resistance_take_profit(self, df: pd.DataFrame, params: dict) -> pd.Series:
        if not params.get('enabled', False): return pd.Series(False, index=df.index)
        resistance_level = df['high_D'].shift(1).rolling(window=params.get('lookback_period', 90)).max()
        signal = df['high_D'] >= resistance_level * (1 - params.get('approach_threshold', 0.02))
        return signal & resistance_level.notna()

    def _check_trailing_stop_take_profit(self, df: pd.DataFrame, params: dict) -> pd.Series:
        if not params.get('enabled', False): return pd.Series(False, index=df.index)
        highest_close_since = df['close_D'].shift(1).rolling(window=params.get('lookback_period', 20)).max()
        stop_price = highest_close_since * (1 - params.get('percentage_pullback', 0.10))
        signal = df['close_D'] < stop_price
        return signal & highest_close_since.notna()

    def _check_indicator_take_profit(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """【V2.10 优化】支持RSI和BIAS两种指标止盈。"""
        if not params.get('enabled', False): return pd.Series(0, index=df.index)
        
        # 初始化返回的信号序列，0表示无信号
        final_signal = pd.Series(0, index=df.index)

        # 检查RSI止盈
        if params.get('use_rsi_exit', False):
            rsi_col = f"RSI_{params.get('rsi_period', 14)}_D"
            if rsi_col in df.columns:
                rsi_signal = df[rsi_col] > params.get('rsi_threshold', 80)
                final_signal[rsi_signal] = 3 # 3 代表RSI超买止盈
        
        # 检查BIAS止盈
        if params.get('use_bias_exit', False):
            bias_col = f"BIAS_{params.get('bias_period', 20)}_D"
            if bias_col in df.columns:
                bias_signal = df[bias_col] > params.get('bias_threshold', 20.0)
                # BIAS信号优先级更高，可以覆盖RSI信号
                final_signal[bias_signal] = 5 # 5 代表BIAS超买止盈

        return final_signal

    def _check_fund_flow_confirmation(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """【新状态】资金流确认：当天主力资金为净流入。"""
        params = self._get_params_block(params, 'fund_flow_confirm_params')
        if not params.get('enabled', False) or 'fund_buy_lg_amount_D' not in df.columns:
            return pd.Series(False, index=df.index)
        
        threshold = params.get('threshold', 0) # 阈值，可以设置为0，代表只要是净流入就算
        # 状态：当日大单净额 > 阈值，并且5日主力净额也是正的，代表趋势性流入
        is_confirmed = (df['fund_buy_lg_amount_D'] > threshold) & (df.get('fund_net_d5_amount_D', 0) > 0)
        return is_confirmed

    def _check_vwap_confirmation(self, df_dict: Dict[str, pd.DataFrame], params: dict) -> pd.Series:
        """
        【V2.5 最终修复版】检查VWAP支撑确认信号。
        """
        # print("    [VWAP确认] 正在执行分钟线VWAP确认逻辑 (V2.5 最终修复版)...") # 可以注释掉常规日志
        vwap_params = self._get_params_block(params, 'vwap_confirmation_params')
        if not vwap_params.get('enabled', False):
            if 'D' in df_dict and not df_dict['D'].empty:
                return pd.Series(False, index=df_dict['D'].index)
            return pd.Series([])

        tf = vwap_params.get('timeframe', '5')
        buffer = vwap_params.get('confirmation_buffer', 0.001)

        df_minute = df_dict.get(tf)
        df_daily = df_dict.get('D')

        if df_minute is None or df_minute.empty or df_daily is None or df_daily.empty:
             return pd.Series(False, index=df_daily.index if df_daily is not None else None)

        # 检查VWAP列是否存在
        vwap_col = f'VWAP_{tf}'
        if vwap_col not in df_minute.columns: vwap_col = 'VWAP_D' # 兼容旧列名
        if vwap_col not in df_minute.columns: return pd.Series(False, index=df_daily.index)

        # 核心逻辑：检查分钟线是否在VWAP之上，然后按天聚合
        # .normalize() 对UTC时区索引同样有效，会将其归一化到当天的零点
        is_above_vwap = df_minute['close'] > df_minute[vwap_col] * (1 + buffer)
        daily_confirmation = is_above_vwap.groupby(is_above_vwap.index.normalize()).any()
        
        # 将聚合后的日线信号映射回原始的日线DataFrame
        final_signal = pd.Series(False, index=df_daily.index)
        
        if not daily_confirmation.empty:
            # 使用 .reindex() 是一种更健壮和简洁的映射方式
            # 它会根据 df_daily 的归一化日期索引，从 daily_confirmation 中查找对应的值
            # 找不到的日期会自动填充为 fill_value (默认为NaN，我们用 .fillna(False) 处理)
            normalized_daily_index = df_daily.index.normalize()
            final_signal = daily_confirmation.reindex(normalized_daily_index).fillna(False)
            # reindex 后索引会变成归一化的日期，需要把它重置回原始的日线索引
            final_signal.index = df_daily.index
        # print(f"    [VWAP确认] 完成，共产生 {final_signal.sum()} 个VWAP支撑信号。") # 可以注释掉常规日志
        return final_signal




