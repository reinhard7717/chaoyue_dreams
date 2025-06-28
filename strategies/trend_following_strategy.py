# 文件: strategies/trend_follow_strategy.py
import asyncio
import json
import logging
from utils.data_sanitizer import sanitize_for_json
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences
from services.indicator_services import IndicatorService
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from strategies.kline_pattern_recognizer import KlinePatternRecognizer
from utils.config_loader import load_strategy_config # 导入我们新的加载函数

logger = logging.getLogger(__name__)

class TrendFollowStrategy:
    """
    趋势跟踪策略 (V2.5 - 多配置适配版)

    核心逻辑:
    1.  **定义长期/中期趋势**: (逻辑不变) 确认股票处于宏观上升趋势中，作为所有信号的前置过滤器。
    2.  **寻找多维买点**: 在满足趋势前提下，并行寻找四种独立的买入机会：
        - **【升级】均线回踩 (高精度)**: 利用更小周期(如60m)的最低价判断是否触及支撑，日线收盘确认。
        - **【升级】结构回踩 (高精度)**: 同上，利用更小周期判断是否触及前期水平支撑。
        - **【新增】动能追击**: 捕捉股价放量突破N日新高，持续走强的信号。
        - **【新增】指标共振**: 捕捉MACD金叉、DMI(ADX)多头排列等技术指标发出的买入信号。
    
    此策略从单一的回调模式，升级为“回调+追击+指标”三位一体的综合趋势跟踪系统。
    """

    def __init__(self,
                 daily_config_path: str = 'config/trend_follow_strategy.json',
                 tactical_configs: Optional[Dict[str, str]] = None):
        """
        【V3.1 修正版】构造函数，加载日线和多套战术周期(分钟线)配置。
        """
        self.indicator_service = IndicatorService()

        # 1. 加载日线配置 (只加载一次)
        self.daily_config_path = daily_config_path
        print(f"--- [策略初始化] 正在加载日线配置: {self.daily_config_path} ---")
        self.daily_params = load_strategy_config(self.daily_config_path)

        # 2. 加载战术配置 (修正了致命的初始化逻辑)
        if tactical_configs is None:
            # 修改行: 使用不含'min'的键，并指向您实际的配置文件名。
            self.tactical_configs = {
                '5': 'config/trend_follow_strategy_5min.json',
                '15': 'config/trend_follow_strategy_15min.json',
                '30': 'config/trend_follow_strategy_30min.json',
                '60': 'config/trend_follow_strategy_60min.json',
            }
        else:
            self.tactical_configs = tactical_configs

        self.tactical_params = {}
        for tf, path in self.tactical_configs.items():
            try:
                # print(f"--- [策略初始化] 正在加载 {tf} 分钟战术配置: {path} ---")
                self.tactical_params[tf] = load_strategy_config(path)
            except FileNotFoundError:
                logger.warning(f"战术配置文件未找到: {path}。周期 '{tf}' 的特定参数将不可用。")
                self.tactical_params[tf] = {} # 提供一个空字典以避免后续错误

        # 3. 初始化其他组件
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        kline_params = self._get_params_block(self.daily_params, 'kline_pattern_params')
        self.pattern_recognizer = KlinePatternRecognizer(params=kline_params)

        self.signals = {}
        self.scores = {}
        self._last_score_details_df = None

        self.debug_params = self._get_params_block(self.daily_params, 'debug_params')
        self.verbose_logging = self.debug_params.get('enabled', False) and self.debug_params.get('verbose_logging', False)

    async def apply_strategy(self, df_dict: Dict[str, pd.DataFrame], params: dict) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        【V15.3 调试增强版】
        - 调用返回得分明细的计分函数，并打印详细的“得分小票”。
        """
        df = df_dict.get('D')
        if df is None or df.empty:
            return pd.DataFrame(), {}
        
        timeframe_suffixes = ['_D', '_W', '_M', '_5', '_15', '_30', '_60']
        rename_map = {col: f"{col}_D" for col in df.columns if not any(col.endswith(suffix) for suffix in timeframe_suffixes)}
        if rename_map:
            df = df.rename(columns=rename_map)
        
        self.signals, self.scores = {}, {}
        df = self.pattern_recognizer.identify_all(df)
        df.loc[:, 'signal_top_divergence'] = self._find_top_divergence_exit(df, params)
        self._analyze_dynamic_box_and_ma_trend(df, params)

        # 调用返回得分明细的计分函数
        df.loc[:, 'entry_score'], atomic_signals, score_details_df = self._calculate_entry_score(df, params)
        
        score_threshold = self._get_params_block(params, 'entry_scoring_params').get('score_threshold', 100)
        df.loc[:, 'signal_entry'] = df['entry_score'] >= score_threshold
        df.loc[:, 'take_profit_signal'] = self._apply_take_profit_rules(df, df['signal_entry'], df['signal_top_divergence'], params)

        # --- 打印详细的“得分小票” ---
        print("\n---【多时间框架协同策略(V15.3 融合版)逻辑链调试】---")
        entry_signals = df[df['signal_entry']]
        print(f"【最终买入】(得分>{score_threshold})信号总数: {len(entry_signals)}")
        if not entry_signals.empty:
            last_entry_date = entry_signals.index[-1]
            last_entry_score = df.loc[last_entry_date, 'entry_score']
            print(f"  - 最近一次买入: {last_entry_date.date()} (当日总分: {last_entry_score})")
            
            # 打印得分构成
            print(f"  --- 得分小票 for {last_entry_date.date()} ---")
            score_breakdown = score_details_df.loc[last_entry_date].dropna()
            base_score_items = {k: v for k, v in score_breakdown.items() if k.startswith('BASE_')}
            tactical_score_items = {k: v for k, v in score_breakdown.items() if not k.startswith('BASE_')}
            
            base_total = sum(base_score_items.values())
            print(f"    [战略基础分: {base_total:.2f} pts]")
            if not base_score_items:
                print("      - (无)")
            for item, score in base_score_items.items():
                print(f"      - {item}: {score:.2f} pts")

            tactical_total = sum(tactical_score_items.values())
            print(f"    [战术叠加分: {tactical_total:.2f} pts]")
            if not tactical_score_items:
                print("      - (无)")
            for item, score in tactical_score_items.items():
                print(f"      - {item}: {score:.2f} pts")
            
            print(f"    ------------------------------------")
            print(f"    [核算总分: {base_total + tactical_total:.2f} pts]")

        return df, atomic_signals

    def prepare_db_records(self, stock_code: str, result_df: pd.DataFrame, atomic_signals: Dict[str, pd.Series], params: dict) -> List[Dict[str, Any]]:
        """
        【V2.2 数据清洗增强版】将策略分析结果DataFrame转换为用于数据库存储的字典列表。
        - 使用 sanitize_for_json 工具函数确保所有数据都是JSON兼容的原生Python类型。
        """
        df_with_signals = result_df[
            (result_df['signal_entry'] == True) | (result_df['take_profit_signal'] > 0)
        ].copy()

        if df_with_signals.empty:
            return []

        records = []
        strategy_name = self._get_params_block(params, 'strategy_info').get('name', 'multi_timeframe_collaboration')
        timeframe = self._get_params_block(params, 'strategy_info').get('timeframe', 'D')

        for timestamp, row in df_with_signals.iterrows():
            # 从计分详情中获取激活的剧本
            triggered_playbooks_list = []
            if self._last_score_details_df is not None and timestamp in self._last_score_details_df.index:
                playbooks = self._last_score_details_df.loc[timestamp]
                triggered_playbooks_list = playbooks[playbooks > 0].index.tolist()

            is_setup_day = 'PULLBACK_SETUP' in triggered_playbooks_list
            
            # ▼▼▼【代码修改】: 全面应用 sanitize_for_json ▼▼▼
            # 解释: 将所有可能来自DataFrame的原始值都通过清洗函数处理，一劳永逸地解决类型问题。
            
            # 1. 清洗 context_snapshot
            context_dict = {k: v for k, v in row.items() if pd.notna(v)}
            sanitized_context = sanitize_for_json(context_dict)

            # 2. 清洗所有其他字段
            record = {
                # --- 核心字段 ---
                "stock_code": stock_code,
                "trade_time": sanitize_for_json(timestamp),
                "timeframe": timeframe,
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
                "triggered_playbooks": triggered_playbooks_list, # 列表本身是安全的
                "context_snapshot": sanitized_context,
            }
            # ▲▲▲【代码修改】: 修改结束 ▲▲▲
            records.append(record)
            
        return records

    def _calculate_entry_score(self, df: pd.DataFrame, params: dict) -> Tuple[pd.Series, Dict[str, pd.Series], pd.DataFrame]:
        """
        【V15.3 调试增强版 - 完整修正版】计算综合买入得分，并返回详细的得分构成。
        - 修正了返回元组数量不匹配的 ValueError。
        - 所有得分项均记录在 score_details_df 中，实现透明的“得分小票”功能。
        """
        atomic_signals = {}
        score_details_df = pd.DataFrame(index=df.index) # 新增：用于存储得分明细
        scoring_params = self._get_params_block(params, 'entry_scoring_params')
        points = scoring_params.get('points', {})

        # --- 步骤1: 计算并记录战略背景基础分 ---
        print("    [调试-计分V15.3] 步骤1: 计算并记录周线战略背景基础分...")
        playbook_scores = self._get_params_block(params, 'playbook_scores', default_return={
            'ma20_turn_up': 100, 'early_uptrend': 80, 'classic_breakout': 50, 
            'ma_uptrend': 30, 'bottom_consolidation': 50
        })
        playbook_cols = [col for col in df.columns if col.startswith('playbook_') and col.endswith('_W')]
        for col in playbook_cols:
            playbook_name = col.replace('playbook_', '').replace('_W', '')
            score = playbook_scores.get(playbook_name, 0)
            if score > 0 and df[col].any():
                base_score_col_name = f"BASE_{playbook_name.upper()}"
                score_details_df.loc[df[col], base_score_col_name] = score

        # 1a. 【全面填充】将所有候选分数（包括王牌和原始剧本）全部写入DataFrame
        all_base_score_cols = []
        
        # 王牌信号
        king_signal_col = 'signal_breakout_trigger_W'
        king_base_col_name = f"BASE_{king_signal_col.replace('_W', '').upper()}"
        all_base_score_cols.append(king_base_col_name)
        king_score = points.get('BREAKOUT_TRIGGER_SCORE', 150)
        if king_signal_col in df.columns and df[king_signal_col].any():
            score_details_df.loc[df[king_signal_col], king_base_col_name] = king_score

        # 原始剧本
        playbook_scores_map = self._get_params_block(params, 'playbook_scores', default_return={
            'ma20_turn_up': 100, 'early_uptrend': 80, 'classic_breakout': 50,
            'ma_uptrend': 30, 'box_breakout': 60
        })
        playbook_cols = [col for col in df.columns if col.startswith('playbook_') and col.endswith('_W')]
        for col in playbook_cols:
            playbook_name = col.replace('playbook_', '').replace('_W', '')
            score = playbook_scores_map.get(playbook_name, 0)
            base_col_name = f"BASE_{playbook_name.upper()}"
            all_base_score_cols.append(base_col_name)
            if score > 0 and df[col].any():
                score_details_df.loc[df[col], base_col_name] = score
        
        score_details_df.fillna(0, inplace=True)

        if king_base_col_name in score_details_df.columns:
            # 列存在，正常生成掩码
            king_signal_mask = (score_details_df[king_base_col_name] > 0)
        else:
            # 列不存在，创建一个安全的、全为 False 的掩码
            print(f"    [调试-计分] 警告: 王牌信号列 '{king_base_col_name}' 未在得分详情中找到，本次计算将不执行互斥清零。")
            king_signal_mask = pd.Series(False, index=df.index)

        # 1b. 【铁腕清零】在王牌信号日，将所有其他基础分清零
        if king_signal_mask.any():
            # 获取除王牌信号外的所有其他基础分列
            other_base_score_cols = [col for col in all_base_score_cols if col != king_base_col_name]
            # 对王牌日，将其他所有基础分强制设为0
            score_details_df.loc[king_signal_mask, other_base_score_cols] = 0
            
            # 1c. 【最终调试日志】打印王牌信号日的最终、纯净分数
            print(f"    [调试-计分V16.1] 检测到 {king_signal_mask.sum()} 次“王牌信号”，执行互斥计分:")
            king_day_scores = score_details_df.loc[king_signal_mask]
            for date, row in king_day_scores.iterrows():
                final_scores = {k: v for k, v in row.to_dict().items() if v > 0}
                print(f"      - 日期: {date.strftime('%Y-%m-%d')}, 最终基础分: {final_scores}")

        # --- 步骤2: 定义战术信号的前提条件 ---
        base_score = score_details_df.sum(axis=1)
        tactical_precondition = base_score > 0
        strict_precondition = tactical_precondition & df.get('context_mid_term_bullish', pd.Series(False, index=df.index))

        # --- 步骤3: 计算所有独立的日线战术原子信号 ---
        print("    [调试-计分V15.3] 步骤2: 计算日线战术原子信号...")
        # (此部分计算逻辑保持不变，仅为完整性展示)
        cond_pullback_ma = self._find_pullback_to_ma_entry(df, tactical_precondition, params)
        cond_pullback_structure = self._find_pullback_to_structure_entry(df, tactical_precondition, params)
        cond_v_reversal = self._find_v_reversal_entry(df, tactical_precondition, params)
        cond_pullback_setup, pullback_target_price = self._find_pullback_setup(df, strict_precondition, params)
        df['pullback_target_price'] = pullback_target_price
        cond_bottom_divergence = self._find_bottom_divergence_entry(df, params)
        cond_bias_reversal = self._find_bias_reversal_entry(df, params)
        cond_washout_reversal = self._find_washout_reversal_entry(df, tactical_precondition, params)
        cond_capital_flow_divergence = self._find_capital_flow_divergence_entry(df, params)
        cond_momentum = self._find_momentum_entry(df, strict_precondition, params)
        cond_first_breakout = self._find_first_breakout_entry(df, strict_precondition, params)
        cond_bb_squeeze_breakout = self._find_bband_squeeze_breakout(df, strict_precondition, params)
        cond_doji_continuation = self._find_doji_continuation_entry(df, strict_precondition, params)
        cond_old_duck_head = self._find_old_duck_head_entry(df, strict_precondition, params)
        cond_n_shape_relay = self._find_n_shape_relay_entry(df, strict_precondition, params)
        cond_bullish_flag = self._find_bullish_flag_entry(df, strict_precondition, params)
        cond_energy_compression_breakout = self._find_energy_compression_breakout_entry(df, strict_precondition, params)
        cond_relative_strength_maverick = self._find_relative_strength_maverick_entry(df, strict_precondition, params)
        cond_dynamic_box_breakout = self.signals.get('dynamic_box_breakout', pd.Series(False, index=df.index)) & strict_precondition
        indicator_signals = self._find_indicator_entry(df, strict_precondition, params)
        cond_dmi_cross, cond_macd_low_cross, cond_macd_zero_cross, cond_macd_high_cross = indicator_signals['dmi_cross'], indicator_signals['macd_low_cross'], indicator_signals['macd_zero_cross'], indicator_signals['macd_high_cross']
        cond_cmf_confirm = self._check_cmf_confirmation(df, params)
        cond_gap_support_state = self._check_upward_gap_support(df, params)
        board_patterns = self._identify_board_patterns(df, params)
        cond_earth_heaven_board, cond_turnover_board, cond_heaven_earth_board = board_patterns.get('earth_heaven_board', pd.Series(False, index=df.index)), board_patterns.get('turnover_board', pd.Series(False, index=df.index)), board_patterns.get('heaven_earth_board', pd.Series(False, index=df.index))
        cond_volume_breakdown = self._find_volume_breakdown_exit(df, params)
        cond_chip_cost_breakthrough = self._find_chip_cost_breakthrough(df, strict_precondition, params)
        cond_chip_pressure_release = self._find_chip_pressure_release(df, strict_precondition, params)
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

        # --- 步骤4: 记录所有战术信号得分 ---
        print("    [调试-计分V15.3] 步骤3: 记录日线战术信号得分...")
        # 辅助函数，用于简化计分逻辑
        def add_score(condition, name, default_score):
            if condition.any():
                score_details_df.loc[condition, name] = points.get(name, default_score)
            atomic_signals[name] = condition

        add_score(cond_chip_pressure_release, 'CHIP_PRESSURE_RELEASE', 150)
        add_score(cond_chip_cost_breakthrough, 'CHIP_COST_BREAKTHROUGH', 130)
        add_score(is_steady_climb_pullback, 'PULLBACK_STEADY_CLIMB', 110)
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
        add_score(cond_dmi_cross, 'DMI_CROSS', 30)
        add_score(cond_macd_low_cross, 'MACD_LOW_CROSS', 40)
        add_score(cond_macd_zero_cross, 'MACD_ZERO_CROSS', 60)
        add_score(cond_macd_high_cross, 'MACD_HIGH_CROSS', 25)
        add_score(cond_pullback_setup, 'PULLBACK_SETUP', 50)
        add_score(cond_morning_star, 'KLINE_MORNING_STAR', 140)
        add_score(cond_three_soldiers & strict_precondition, 'KLINE_THREE_SOLDIERS', 100)

        # --- 步骤5: 记录协同/冲突规则得分 ---
        print("    [调试-计分V15.3] 步骤4: 记录协同/冲突规则得分...")
        has_positive_score = score_details_df.sum(axis=1) > 0

        # 协同奖励（加法）
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

        # 协同奖励（乘法）- 转换为加法项
        current_score = score_details_df.fillna(0).sum(axis=1)
        multiplier_bonus = pd.Series(0.0, index=df.index)
        cmf_multiplier = points.get('CMF_CONFIRMATION_MULTIPLIER', 1.2)
        fund_multiplier = points.get('FUND_FLOW_CONFIRM_MULTIPLIER', 1.25)
        gap_multiplier = points.get('GAP_SUPPORT_MULTIPLIER', 1.3)

        multiplier_bonus.loc[cond_cmf_confirm & has_positive_score] += current_score * (cmf_multiplier - 1)
        multiplier_bonus.loc[cond_fund_flow_confirm & has_positive_score] += current_score * (fund_multiplier - 1)
        multiplier_bonus.loc[cond_gap_support_state & has_positive_score] += current_score * (gap_multiplier - 1)
        score_details_df['BONUS_MULTIPLIER'] = multiplier_bonus.where(multiplier_bonus > 0)

        # 冲突惩罚（乘法）- 转换为减法项
        penalty_score = pd.Series(0.0, index=df.index)
        is_reversal_play_conflict = cond_bottom_divergence | cond_bias_reversal
        is_breakout_play_conflict = cond_momentum | cond_first_breakout | cond_bb_squeeze_breakout
        is_conflicting = is_reversal_play_conflict & is_breakout_play_conflict
        conflict_penalty_rate = 1 - points.get('REVERSAL_BREAKOUT_CONFLICT_PENALTY', 0.8)
        penalty_score.loc[is_conflicting] -= current_score * conflict_penalty_rate
        score_details_df['PENALTY_CONFLICT'] = penalty_score.where(penalty_score < 0)

        # --- 步骤6: 计算总分并应用最终过滤器 ---
        final_score = score_details_df.fillna(0).sum(axis=1)

        # 风险否决
        final_score.loc[kline_strong_bearish] = 0
        final_score.loc[cond_volume_breakdown] = 0
        final_score.loc[cond_heaven_earth_board] = 0

        # 最终过滤器
        is_reversal_play_final = cond_bottom_divergence | cond_bias_reversal | cond_capital_flow_divergence | cond_morning_star
        final_score = final_score.where(tactical_precondition | is_reversal_play_final, 0)

        # --- 步骤7: 正确返回三元组 ---
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
        【新增状态确认】检查CMF资金流指标是否为正，以确认买入信号。
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
        【新增状态确认】检查近期是否存在未回补的向上跳空缺口。
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
        【新增模块】识别A股特色的各种“板”形态。
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

    # 【新增模块】主力筹码运作行为识别

    def _find_consolidation_breakout_entry(self, df: pd.DataFrame, precondition: pd.Series, params: dict) -> pd.Series:
        """
        【新增剧本】识别“底部长期盘整突破”，捕捉主力隐蔽吸筹后的启动点。
        特征：股价经历长时间（如超过3个月）的横盘整理，期间波动率和成交量持续萎缩。
        信号：某天突然以“放量大阳线”的形式，向上突破盘整区间的上沿。
        """
        params = self._get_params_block(params, 'consolidation_breakout_params')
        if not params.get('enabled', False):
            return pd.Series(False, index=df.index)

        lookback = params.get('lookback_period', 60) # 盘整期
        volatility_quantile = params.get('volatility_quantile', 0.2) # 波动率分位数
        volume_quantile = params.get('volume_quantile', 0.2) # 成交量分位数
        breakout_vol_ratio = params.get('breakout_volume_ratio', 2.0)

        # 1. 计算波动率（振幅/收盘价）和成交量
        volatility = (df['high_D'] - df['low_D']) / df['close_D']
        
        # 2. 识别“盘整期”：波动率和成交量均处于近期低位
        is_low_volatility = volatility < volatility.rolling(lookback).quantile(volatility_quantile)
        is_low_volume = df['volume_D'] < df['volume_D'].rolling(lookback).quantile(volume_quantile)
        is_consolidating = is_low_volatility & is_low_volume

        # 3. 识别“突破日”
        consolidation_high = df['high_D'].shift(1).rolling(lookback).max()
        is_breakout = df['close_D'] > consolidation_high
        is_volume_surge = df['volume_D'] > df[f'VOL_MA_{params.get("vol_ma_period", 20)}_D'] * breakout_vol_ratio

        # 信号：突破日的前一天，必须处于“盘整期”状态
        signal = is_consolidating.shift(1).fillna(False) & is_breakout & is_volume_surge
        return signal & precondition

    def _find_bullish_flag_entry(self, df: pd.DataFrame, precondition: pd.Series, params: dict) -> pd.Series:
        """
        【新增剧本】【V1.1 性能重构版】识别“上升三法”（或称旗形整理）。
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
        【新增卖出剧本】识别“高位长上影线”，主力隐蔽派发的信号。
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
        【新增卖出剧本】识别“放量破位”，主力砸盘出货的信号。
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

    # 【新增模块】更隐蔽的主力行为识别 (能量学与博弈论)

    def _find_energy_compression_breakout_entry(self, df: pd.DataFrame, precondition: pd.Series, params: dict) -> pd.Series:
        """
        【新增剧本】“潜龙在渊” - 识别市场能量极度压缩后的爆发点。
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
        【新增剧本】【V1.1 逻辑修正版】“资金暗流” - 识别价格与资金指标的底背离。
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
        【新增剧本】“逆市强人” - 识别相对市场表现强势的股票。
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
        """【V2.6 新增】寻找A股特色的“首板”或强势启动信号。"""
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
        【V3.2 因果修正版】使用 scipy 和 pandas 高级技巧，实现完全向量化的精确底背离检测。
        - 【修复】修正了信号确认逻辑，增加了 .shift(1) 约束，确保“确认阳线”(果)必须发生在“背离事件”(因)之后，
          与其他回踩信号的“因果律”逻辑保持严格一致。
        """
        divergence_params = self._get_params_block(params, 'divergence_params')
        if not divergence_params.get('enabled', False): return pd.Series(False, index=df.index)
        
        # 1. 动态获取指标列名
        fe_params = params.get('feature_engineering_params', {})
        indicators_config = fe_params.get('indicators', {})
        rsi_config = indicators_config.get('rsi', {})
        rsi_period = rsi_config.get('periods', [14])[0]
        rsi_col = f"RSI_{rsi_period}_D"
        macd_config = indicators_config.get('macd', {})
        macd_periods = macd_config.get('periods', [12, 26, 9])
        try:
            fast, slow, signal_line = macd_periods
        except (ValueError, TypeError):
            fast, slow, signal_line = 12, 26, 9
        macd_hist_col = f"MACDh_{fast}_{slow}_{signal_line}_D"
        
        required_cols = ['low_D', 'close_D', 'open_D', rsi_col, macd_hist_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            if self.verbose_logging:
                print(f"    [调试-复合底背离-警告]: 无法执行，缺少必需列: {missing_cols}")
            return pd.Series(False, index=df.index)

        # 2. 使用 find_peaks 精确寻找波谷
        find_troughs_params = {
            "distance": params.get('distance', 5),
            "prominence": params.get('prominence_bottom', 0.1)
        }
        price_troughs, _ = find_peaks(-df['low_D'], **find_troughs_params)
        rsi_troughs, _ = find_peaks(-df[rsi_col], **find_troughs_params)
        macd_troughs, _ = find_peaks(-df[macd_hist_col], **find_troughs_params)

        # 3. 创建只在波谷日有值的稀疏序列
        df['price_at_trough'] = np.nan
        df.iloc[price_troughs, df.columns.get_loc('price_at_trough')] = df['low_D'].iloc[price_troughs]
        
        df['rsi_at_trough'] = np.nan
        df.iloc[rsi_troughs, df.columns.get_loc('rsi_at_trough')] = df[rsi_col].iloc[rsi_troughs]

        df['macd_at_trough'] = np.nan
        df.iloc[macd_troughs, df.columns.get_loc('macd_at_trough')] = df[macd_hist_col].iloc[macd_troughs]

        # 4. 向前填充，创建“状态”序列
        df['last_price_trough'] = df['price_at_trough'].ffill()
        df['last_rsi_trough'] = df['rsi_at_trough'].ffill()
        df['last_macd_trough'] = df['macd_at_trough'].ffill()

        # 5. 向量化判断背离条件
        price_lower_low = df['price_at_trough'] < df['last_price_trough'].shift(1)
        rsi_higher_low = df['rsi_at_trough'] > df['last_rsi_trough'].shift(1)
        macd_higher_low = df['macd_at_trough'] > df['last_macd_trough'].shift(1)

        is_rsi_divergence = df['price_at_trough'].notna() & df['rsi_at_trough'].notna() & price_lower_low & rsi_higher_low
        is_macd_divergence = df['price_at_trough'].notna() & df['macd_at_trough'].notna() & price_lower_low & macd_higher_low
        
        divergence_events = is_rsi_divergence | is_macd_divergence

        # 这是解决“因果律”谬误的关键！
        # 6. 确认信号 (应用 .shift(1) 确保因果关系)
        confirmation_days = params.get('confirmation_days', 3)
        is_green_candle = df['close_D'] > df['open_D']
        # 检查【过去】N天内是否发生过背离事件
        has_recent_divergence = divergence_events.rolling(window=confirmation_days, min_periods=1).sum() > 0
        # 最终信号：必须是确认阳线(果)，且近期发生过背离(因)，且两者不能是同一天
        final_signal = is_green_candle & has_recent_divergence.shift(1).fillna(False) & ~divergence_events
        
        # 7. 清理临时列
        df.drop(columns=['price_at_trough', 'rsi_at_trough', 'macd_at_trough', 
                         'last_price_trough', 'last_rsi_trough', 'last_macd_trough'], inplace=True)

        # 打印调试信息
        if self.verbose_logging:
            print(f"    [调试-复合底背离]: 背离事件天数: {divergence_events.sum()} | "
                f"最终信号: {final_signal.sum()}")
              
        return final_signal
    
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

    def _find_indicator_entry(self, df: pd.DataFrame, precondition: pd.Series, params: dict) -> Dict[str, pd.Series]:
        """【V2.8 MACD深化版】区分低位、零轴、高位金叉，并返回一个信号字典。"""
        params = self._get_params_block(params, 'indicator_entry_params')
        if not params.get('enabled', False): 
            return {
                'dmi_cross': pd.Series(False, index=df.index),
                'macd_low_cross': pd.Series(False, index=df.index),
                'macd_zero_cross': pd.Series(False, index=df.index),
                'macd_high_cross': pd.Series(False, index=df.index),
            }
        
        # 初始化所有信号
        dmi_signal = pd.Series(False, index=df.index)
        macd_low_signal = pd.Series(False, index=df.index)
        macd_zero_signal = pd.Series(False, index=df.index)
        macd_high_signal = pd.Series(False, index=df.index)
        
        # 价格稳定性过滤器 (逻辑不变)
        mid_term_params = self._get_params_block(params, 'mid_term_trend_params')
        key_ma_col = f"EMA_{mid_term_params.get('mid_ma', 60)}_D"
        stability_period = params.get('price_stability_period', 3)
        is_stable_above_ma = pd.Series(True, index=df.index)
        if key_ma_col in df.columns:
            is_stable_above_ma = (df['close_D'].shift(1) > df[key_ma_col].shift(1)).rolling(window=stability_period).min() == 1
        final_precondition = precondition & is_stable_above_ma

        # --- DMI 信号计算 (逻辑不变) ---
        if params.get('use_dmi_adx', False):
            dmi_period = params.get('dmi_period', 14)
            pdi_col, ndi_col, adx_col = f'PDI_{dmi_period}_D', f'NDI_{dmi_period}_D', f'ADX_{dmi_period}_D'
            if all(c in df.columns for c in [pdi_col, ndi_col, adx_col]):
                dmi_cross = (df[pdi_col] > df[ndi_col]) & (df[pdi_col].shift(1) <= df[ndi_col].shift(1))
                dmi_signal = dmi_cross & (df[adx_col] > params.get('adx_threshold', 25)) & final_precondition
        
        # --- MACD 信号分类计算 ---
        if params.get('use_macd_cross', False):
            macd_fast = params.get('macd_fast', 12)
            macd_slow = params.get('macd_slow', 26)
            macd_signal_p = params.get('macd_signal', 9)
            
            dif_col = f"MACD_{macd_fast}_{macd_slow}_{macd_signal_p}_D"
            dea_col = f"MACDs_{macd_fast}_{macd_slow}_{macd_signal_p}_D"
            
            if dif_col in df.columns and dea_col in df.columns:
                # 1. 找到所有基础金叉点
                basic_cross = (df[dif_col] > df[dea_col]) & (df[dif_col].shift(1) <= df[dea_col].shift(1))
                
                # 2. 获取分类阈值
                low_threshold = params.get('macd_low_threshold', -0.5)   # 定义“低位”的DIF值
                high_threshold = params.get('macd_high_threshold', 1.0)  # 定义“高位”的DIF值
                
                # 3. 根据DIF值对金叉进行分类
                # 低位金叉: DIF < low_threshold
                macd_low_signal = basic_cross & (df[dif_col] < low_threshold)
                # 高位金叉: DIF > high_threshold
                macd_high_signal = basic_cross & (df[dif_col] > high_threshold)
                # 零轴金叉: 介于两者之间
                macd_zero_signal = basic_cross & (df[dif_col] >= low_threshold) & (df[dif_col] <= high_threshold)

                # 4. 应用最终的前置条件
                macd_low_signal &= final_precondition
                macd_zero_signal &= final_precondition
                macd_high_signal &= final_precondition

        return {
            'dmi_cross': dmi_signal,
            'macd_low_cross': macd_low_signal,
            'macd_zero_cross': macd_zero_signal,
            'macd_high_cross': macd_high_signal,
        }

    def _find_bband_squeeze_breakout(self, df: pd.DataFrame, precondition: pd.Series, params: dict) -> pd.Series:
        """【V2.5 新增】寻找布林带收缩突破的买点。"""
        params = self._get_params_block(params, 'bband_squeeze_params')
        if not params.get('enabled', False): return pd.Series(False, index=df.index)
        
        bb_period = params.get('bb_period', 20)
        bb_std = params.get('bb_std', 2.0)
        squeeze_lookback = params.get('squeeze_lookback', 60)
        
        bbw_col = f'BBW_{bb_period}_{bb_std:.1f}_D'
        bbu_col = f'BBU_{bb_period}_{bb_std:.1f}_D'
        
        required_cols = [bbw_col, bbu_col, 'close_D']
        if not all(col in df.columns for col in required_cols): return pd.Series(False, index=df.index)

        # 条件1: 布林带宽度处于近期低位 (Squeeze)
        is_squeeze = df[bbw_col] <= df[bbw_col].rolling(window=squeeze_lookback).min()
        
        # 条件2: 价格收盘突破上轨 (Breakout)
        is_breakout = df['close_D'] > df[bbu_col]
        
        # 信号：当天的突破，必须是由前一天的“Squeeze”状态所引发的
        signal = is_breakout & is_squeeze.shift(1).fillna(False)
        return signal & precondition

    # 寻找BIAS超跌反弹买点
    def _find_bias_reversal_entry(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """【V2.10 新增】寻找BIAS极端超跌后的反弹买点。"""
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
    def _find_divergence(self, price_series: pd.Series, indicator_series: pd.Series, find_peaks_params: dict, find_troughs_params: dict) -> Tuple[pd.Series, pd.Series]:
        """
        【V3.1 终极向量化版】通用背离检测函数。
        - 使用与底背离检测相同的稀疏序列+ffill技巧，完全向量化。
        - 能同时高效计算顶背离和底背离。
        """
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # 开始重构通用背离检测
        # 创建一个临时的DataFrame，避免污染原始df
        temp_df = pd.DataFrame({
            'price': price_series,
            'indicator': indicator_series
        })

        # --- 顶背离计算 ---
        price_peaks, _ = find_peaks(temp_df['price'], **find_peaks_params)
        indicator_peaks, _ = find_peaks(temp_df['indicator'], **find_peaks_params)
        
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
        price_troughs, _ = find_peaks(-temp_df['price'], **find_troughs_params)
        indicator_troughs, _ = find_peaks(-temp_df['indicator'], **find_troughs_params)

        temp_df['price_at_trough'] = np.nan
        temp_df.iloc[price_troughs, temp_df.columns.get_loc('price_at_trough')] = temp_df['price'].iloc[price_troughs]
        temp_df['indicator_at_trough'] = np.nan
        temp_df.iloc[indicator_troughs, temp_df.columns.get_loc('indicator_at_trough')] = temp_df['indicator'].iloc[indicator_troughs]

        temp_df['last_price_trough'] = temp_df['price_at_trough'].ffill()
        temp_df['last_indicator_trough'] = temp_df['indicator_at_trough'].ffill()

        price_lower_low = temp_df['price_at_trough'] < temp_df['last_price_trough'].shift(1)
        indicator_higher_low = temp_df['indicator_at_trough'] > temp_df['last_indicator_trough'].shift(1)
        bottom_divergence_signal = temp_df['price_at_trough'].notna() & temp_df['indicator_at_trough'].notna() & price_lower_low & indicator_higher_low
        
        # 添加了至关重要的 return 语句，并返回Series以避免索引问题
        return top_divergence_signal.set_axis(price_series.index), bottom_divergence_signal.set_axis(price_series.index)

    # 寻找复合顶背离卖点
    def _find_top_divergence_exit(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """【V3.1 向量化版】调用新的向量化背离函数来计算顶背离。"""
        # 简化此函数，使其专注于调用和组合结果
        divergence_params = self._get_params_block(params, 'divergence_params')
        if not divergence_params.get('enabled', False): return pd.Series(False, index=df.index)

        # 动态获取指标列名 (逻辑不变)
        fe_params = params.get('feature_engineering_params', {})
        indicators_config = fe_params.get('indicators', {})
        macd_config = indicators_config.get('macd', {})
        macd_periods = macd_config.get('periods', [12, 26, 9])
        try:
            fast, slow, signal_line = macd_periods
        except (ValueError, TypeError):
            fast, slow, signal_line = 12, 26, 9
        macd_hist_col = f"MACDh_{fast}_{slow}_{signal_line}_D"
        
        rsi_config = indicators_config.get('rsi', {})
        rsi_period = rsi_config.get('periods', [14])[0]
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
        【新增剧本】【V1.2 逻辑深化版】识别“巨阴洗盘”后的“回马枪”信号。
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
        【新增剧本】【V1.1 逻辑深化版】识别上涨途中的“十字星”中继信号。
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
        # 【新增】确认日必须放量，体现分歧转一致
        is_volume_confirmation = df['volume_D'] > (df['volume_D'].shift(1) * confirmation_vol_ratio)
        is_confirmation_day = is_price_confirmation & is_volume_confirmation
        
        # 最终信号：必须满足大趋势(precondition)，且昨天是十字星，今天是放量确认日
        signal = precondition & is_doji_day & is_confirmation_day
        return signal.fillna(False)

    def _find_old_duck_head_entry(self, df: pd.DataFrame, precondition: pd.Series, params: dict) -> pd.Series:
        """
        【新增剧本】【V1.1 性能重构版】识别“老鸭头”形态，一个经典的主力深度控盘后的二次启动信号。
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
        【新增剧本】【V1.2 涨停识别修正版】识别“N字板”接力。
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
        is_strong_body = body_rally > params.get('body_rally_threshold', 0.05) # 新增参数，默认为5%
        
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
        【新增剧本】识别“回撤预备”信号，用于次日盘中监控。
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
        proximity_threshold = params.get('setup_proximity_threshold', 0.03) # 新增配置项，例如3%
        is_close_to_support = (df['close_D'] / df[support_ma_col] - 1) < proximity_threshold

        # 最终的预备信号：满足大趋势 + 在支撑之上 + 距离支撑足够近
        is_setup = precondition & is_above_support & is_close_to_support
        
        # 目标价位就是当天的支撑均线价格，只在预备日有效
        target_price = df[support_ma_col].where(is_setup)

        return is_setup, target_price

    def _find_chip_cost_breakthrough(self, df: pd.DataFrame, precondition: pd.Series, params: dict) -> pd.Series:
        """【新剧本】筹码成本突破：股价上穿市场平均成本。"""
        params = self._get_params_block(params, 'chip_cost_breakthrough_params')
        if not params.get('enabled', False) or 'close_D' not in df.columns or 'cyq_weight_avg_D' not in df.columns:
            return pd.Series(False, index=df.index)
        
        # 信号：收盘价从下方首次突破市场平均成本线
        signal = (df['close_D'] > df['cyq_weight_avg_D']) & (df['close_D'].shift(1) <= df['cyq_weight_avg_D'].shift(1))
        return signal & precondition

    def _find_chip_pressure_release(self, df: pd.DataFrame, precondition: pd.Series, params: dict) -> pd.Series:
        """【新剧本】筹码压力释放：股价突破95%的套牢盘成本线。"""
        params = self._get_params_block(params, 'chip_pressure_release_params')
        if not params.get('enabled', False) or 'close_D' not in df.columns or 'cyq_cost_95pct_D' not in df.columns:
            return pd.Series(False, index=df.index)
            
        # 信号：收盘价突破95分位成本线
        signal = df['close_D'] > df['cyq_cost_95pct_D']
        return signal & precondition

    def _score_and_generate_report(self, signal_row: pd.Series, stock_code: str) -> Dict:
        """【V2.0 升级】对最新信号进行评分并生成报告"""
        signal_type = "无明确信号"
        total_score = 0
        main_analysis_parts = []

        if signal_row.get('take_profit_signal', 0) > 0:
            signal_type = "趋势止盈"
            total_score = 0
            main_analysis_parts.append("--- 卖出信号分析 ---")
            tp_type = signal_row['take_profit_signal']
            if tp_type == 1: reason = "原因: 股价已接近前期重要压力位。"
            elif tp_type == 2: reason = "原因: 股价从近期高点回撤，触发移动止盈。"
            elif tp_type == 3: reason = "原因: 技术指标显示市场过热或趋势转弱。"
            else: reason = ""
            main_analysis_parts.append(f"核心发现: **触发止盈条件，建议考虑获利了结或减仓。{reason}**")
        elif signal_row.get('signal_pullback_structure_entry', False):
            signal_type = "结构回踩买入(极高)"
            total_score = 95
            main_analysis_parts.append("--- 趋势跟踪买入信号分析 ---")
            main_analysis_parts.append("核心发现: **在确认的上升趋势中，股价精准回踩前期关键水平支撑位并获得支撑，确认为极高确定性买点！**")
        elif signal_row.get('signal_entry', False):
            signal_type = f"综合买入(得分:{total_score})"
            main_analysis_parts.append(f"--- 综合买入信号分析 (总分: {total_score}) ---")
            main_analysis_parts.append("核心发现: **多个看涨条件共振，形成高置信度买入信号！**")
            # 可以在此处添加更详细的计分项说明
        elif signal_row.get('context_long_term_bullish', False) and signal_row.get('context_mid_term_bullish', False):
            signal_type = "上升趋势(观察)"
            total_score = 70
            main_analysis_parts.append("--- 趋势状态分析 ---")
            main_analysis_parts.append("核心发现: **股票处于健康的上升趋势中，建议密切关注买入机会。**")

        report_text = "\n".join(main_analysis_parts)

        full_report = f"*** 最新信号分析报告 ({stock_code}) ***\n买入信号评分: {total_score} / 100\n信号类型: {signal_type}\n信号日期: {signal_row.name.strftime('%Y-%m-%d')}\n{report_text}"
        return { "analysis_text": full_report, "buy_score": total_score, "signal_type": signal_type }

    def _prepare_db_record(self, signal_row: pd.Series, report_data: dict, stock_code: str) -> Dict:
        """
        【V2.2 升级】从信号行和报告中构造用于数据库的字典
        注意: 此处的列名(如'EMA_20_D')是与数据库表结构对应的，属于接口契约。
        如果数据库列名是固定的，这里的硬编码是合理的。
        如果要实现完全动态，需要数据库表结构也支持更泛化的列名(如'ema_fast', 'ema_mid')。
        当前我们保持与现有数据库结构的兼容性。
        """
        # 动态获取列名，即使数据库字段名是固定的，也从配置中读取参数来查找列
        trend_follow_params = self.params.get('strategy_params', {}).get('trend_follow', {})
        mid_term_params = trend_follow_params.get('mid_term_trend_params', {})
        momentum_params = trend_follow_params.get('momentum_entry_params', {})

        fast_ma_period = mid_term_params.get('fast_ma', 20)
        mid_ma_period = mid_term_params.get('mid_ma', 60)
        vol_ma_period = momentum_params.get('vol_ma_period', 20) # 从任一包含vol_ma_period的参数块读取即可

        ema_20_col = f'EMA_{fast_ma_period}_D'
        ema_60_col = f'EMA_{mid_ma_period}_D'
        vol_ma_20_col = f'VOL_MA_{vol_ma_period}_D'

        return {
            "stock_code": stock_code,
            "trade_time": signal_row.name,
            "signal_entry": bool(signal_row.get('signal_entry', False)),
            "entry_score": int(signal_row.get('entry_score', 0)),
            "signal_take_profit": int(signal_row.get('take_profit_signal', 0)),
            "open_D": signal_row.get('open_D'),
            "high_D": signal_row.get('high_D'),
            "low_D": signal_row.get('low_D'),
            "close_D": signal_row.get('close_D'),
            # 假设数据库列名是固定的，但我们用动态生成的列名从DataFrame中取值
            "EMA_20_D": signal_row.get(ema_20_col),
            "EMA_60_D": signal_row.get(ema_60_col),
            "volume_D": signal_row.get('volume_D'),
            "VOL_MA_20_D": signal_row.get(vol_ma_20_col),
            "buy_score": report_data.get('buy_score', 0),
            "analysis_text": report_data.get('analysis_text', ""),
            "signal_type": report_data.get('signal_type', ""),
        }

    def run_analysis(self, stock_code: str, trade_time: Optional[str] = None, data_df: Optional[pd.DataFrame] = None) -> Tuple[Optional[pd.DataFrame], Optional[List[Dict]]]:
        """
        【V2.5 升级】运行单只股票的【日线】完整策略分析流程。
        - 明确使用日线配置 (self.daily_params) 进行分析。
        - 返回值适配 prepare_db_records 的列表格式。
        """
        if data_df is None:
            # 【修改】明确传递日线配置文件路径
            df_base, _ = self.loop.run_until_complete(self.indicator_service.prepare_daily_centric_dataframe(stock_code=stock_code, params_file=self.daily_config_path, trade_time=trade_time))
        else:
            df_base = data_df.copy()

        if df_base is None or df_base.empty:
            logger.warning(f"为股票 {stock_code} 准备数据失败，分析终止。")
            return None, None
        
        # 【修改】明确传递日线参数集
        final_df, atomic_signals = self.loop.run_until_complete(self.apply_strategy(df_base, params=self.daily_params))

        if final_df is None or final_df.empty: return None, None
        
        analysis_result = None
        latest_row = final_df.iloc[-1]
        
        has_any_signal = latest_row.get('signal_entry', False) or latest_row.get('take_profit_signal', 0) > 0

        if has_any_signal:
            logger.info(f"--- 在 {latest_row.name.strftime('%Y-%m-%d')} 为 {stock_code} 检测到趋势跟踪(V2.5)信号 ---")
            report_data = self._score_and_generate_report(latest_row, stock_code)
            logger.info(f"信号类型: {report_data.get('signal_type')}, 评分: {report_data.get('buy_score')}")
            db_records = self.prepare_db_records(stock_code, final_df, atomic_signals)
        
        if db_records:
            logger.info(f"--- 为 {stock_code} 检测到 {len(db_records)} 条日线级趋势跟踪信号 ---")
        
        return final_df, db_records

    # 引入状态机，重构止盈逻辑以支持连续交易模拟
    def _apply_take_profit_rules(self, df: pd.DataFrame, entry_signals: pd.Series, top_divergence_signals: pd.Series, params: dict) -> pd.Series:
        """
        【V3.3 动态箱体集成版】使用 cumsum 技巧，完全消除状态机循环。
        - 【修正】明确注释了 top_divergence_signals 是从 apply_strategy 传入的参数。
        - 【新增】集成了“跌破动态箱体下轨”作为高优先级卖出信号 (代码9)。
        """
        # top_divergence_signals 是一个布尔序列，由 apply_strategy 方法计算并作为参数传入此函数。
        # 它代表了任何一天是否出现了顶背离信号。
        final_tp_signal = pd.Series(0, index=df.index)
        tp_params = self._get_params_block(params, 'take_profit_params')
        
        # 步骤1: 计算所有潜在的止盈信号，无论是否持仓
        tp_signal_resistance = self._check_resistance_take_profit(df, tp_params.get('resistance_exit', {}))
        tp_signal_trailing = self._check_trailing_stop_take_profit(df, tp_params.get('trailing_stop_exit', {}))
        tp_signal_indicator = self._check_indicator_take_profit(df, tp_params.get('indicator_exit', {}))
        board_patterns = self._identify_board_patterns(df, params)
        tp_signal_heaven_earth = board_patterns.get('heaven_earth_board', pd.Series(False, index=df.index))
        tp_signal_upthrust = self._find_upthrust_distribution_exit(df, params)
        tp_signal_breakdown = self._find_volume_breakdown_exit(df, params)

        # 新增动态箱体跌破作为止盈/止损信号
        box_params = self._get_params_block(params, 'dynamic_box_params')
        tp_signal_box_breakdown = pd.Series(False, index=df.index)
        if box_params.get('breakdown_sell_enabled', True): # 默认启用
            tp_signal_box_breakdown = self.signals.get('dynamic_box_breakdown', pd.Series(False, index=df.index))

        # 步骤2: 将所有止盈信号按优先级合并到一个布尔序列中
        # 优先级: 砸盘(8) > 天地板(6) > 派发(7) > 动态箱体跌破(9) > 顶背离(4) > 指标(5,3) > 移动止损(2) > 阻力位(1)
        any_tp_signal = (
            tp_signal_breakdown |
            tp_signal_heaven_earth |
            tp_signal_upthrust |
            tp_signal_box_breakdown |
            top_divergence_signals |
            (tp_signal_indicator > 0) |
            tp_signal_trailing |
            tp_signal_resistance
        )

        # 步骤3: 使用 cumsum 技巧创建“持仓区块”
        is_new_entry = entry_signals & ~any_tp_signal.shift(1).fillna(False)
        is_exit = any_tp_signal
        trade_block_id = is_new_entry.cumsum()
        first_exit_in_block = is_exit & ~is_exit.duplicated(keep='first')
        reset_points = first_exit_in_block.cumsum()
        active_trade_block_id = trade_block_id - reset_points.shift(1).fillna(0)
        is_holding_vectorized = active_trade_block_id > 0
        
        # 步骤4: 在持仓期内，根据优先级应用止盈规则
        final_tp_signal.loc[is_holding_vectorized & tp_signal_breakdown] = 8
        final_tp_signal.loc[(final_tp_signal == 0) & is_holding_vectorized & tp_signal_heaven_earth] = 6
        final_tp_signal.loc[(final_tp_signal == 0) & is_holding_vectorized & tp_signal_upthrust] = 7
        # 为新的跌破信号分配一个唯一的止盈码(例如9)
        final_tp_signal.loc[(final_tp_signal == 0) & is_holding_vectorized & tp_signal_box_breakdown] = 9
        final_tp_signal.loc[(final_tp_signal == 0) & is_holding_vectorized & top_divergence_signals] = 4
        indicator_tp_mask = (final_tp_signal == 0) & is_holding_vectorized & (tp_signal_indicator > 0)
        final_tp_signal.loc[indicator_tp_mask] = tp_signal_indicator[indicator_tp_mask]
        final_tp_signal.loc[(final_tp_signal == 0) & is_holding_vectorized & tp_signal_trailing] = 2
        final_tp_signal.loc[(final_tp_signal == 0) & is_holding_vectorized & tp_signal_resistance] = 1
        
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



