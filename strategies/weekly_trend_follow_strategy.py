# 文件: strategies/weekly_trend_follow_strategy.py
# 版本: V18.2 - 箱体突破逻辑重构版

import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from services.indicator_services import IndicatorService
from utils.config_loader import load_strategy_config

logger = logging.getLogger(__name__)

class WeeklyTrendFollowStrategy:
    """
    周线趋势跟踪策略 (V21.0 - 适配新架构版)
    - 核心修改: 移除对 IndicatorService 的依赖，变为一个纯粹的同步计算类。
    - 职责定位: 接收一个已包含所有周线指标的DataFrame，应用策略逻辑，并返回带有战略信号的DataFrame。
    """

    def __init__(self, config_path: str = 'config/weekly_trend_follow_strategy.json'):
        """
        初始化周线策略。
        - 移除 IndicatorService 和 asyncio loop 的初始化。
        - 仅加载策略逻辑所需的配置文件。
        """
        # print("--- [周线策略初始化] 正在加载周线策略配置文件... ---")
        # self.indicator_service = IndicatorService() # 不再需要
        self.params = load_strategy_config(config_path)
        self.indicator_cfg = self.params.get('feature_engineering_params', {}).get('indicators', {})
        self.playbook_params = self.params.get('strategy_playbooks', {})
        # print(f"    - 周线策略配置 '{config_path}' 加载完成。")

    def apply_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【核心策略应用函数 V21.0 同步版】
        - 引入信号合成层，将独立的剧本信号提纯为高质量的“突破观察”信号。
        - 变为同步方法，因为它不再执行任何IO操作。
        """
        if df is None or df.empty:
            logger.warning("周线策略输入DataFrame为空，无法应用。")
            return pd.DataFrame()

        # --- 步骤 1: 计算所有基础剧本和诊断信号 ---
        context_df = self._calculate_all_playbooks(df)
        
        print("\n---【周线战略层(V21.0 信号合成与提纯)诊断】---")

        # --- 步骤 2: 信号合成层 (逻辑保持不变) ---
        # 2.1 定义“近期有吸筹”状态 (背景)
        washout_threshold = self.playbook_params.get('washout_score_playbook', {}).get('score_threshold', 2)
        
        accumulation_playbooks = [
            'playbook_early_uptrend_W',
            'playbook_bias_rebound_W',
            'playbook_ma20_turn_up_W'
        ]
        is_accumulation_signal = (
            context_df[accumulation_playbooks].any(axis=1) |
            (context_df['washout_score_W'] >= washout_threshold)
        )
        print(f"    - [吸筹定义] 使用的剧本: {accumulation_playbooks} + 洗盘分数(>={washout_threshold})")

        recent_accumulation_window = 12 
        had_recent_accumulation = is_accumulation_signal.rolling(
            window=recent_accumulation_window, min_periods=1
        ).sum().shift(1) > 0
        context_df['state_had_recent_accumulation_W'] = had_recent_accumulation.fillna(False)

        # 2.2 定义“周线突破事件” (事件)
        is_breakout_event = (
            context_df['playbook_classic_breakout_W'] |
            context_df['playbook_box_breakout_W']
        )
        print(f"    - [突破定义] 使用的剧本: ['playbook_classic_breakout_W', 'playbook_box_breakout_W']")
        context_df['event_is_breakout_week_W'] = is_breakout_event

        # 2.3 合成“突破启动”信号 (状态 + 事件 = 启动)
        is_breakout_initiation = context_df['state_had_recent_accumulation_W'] & context_df['event_is_breakout_week_W']
        signal_breakout_initiation = (is_breakout_initiation) & (is_breakout_initiation.shift(1) == False)
        context_df['signal_breakout_initiation_W'] = signal_breakout_initiation

        # 2.4 应用“无拒绝”过滤器 (风险过滤)
        is_rejection_week = context_df['rejection_signal_W'] < 0
        breakout_event_group = context_df['signal_breakout_initiation_W'].cumsum()
        rejections_in_group_so_far = is_rejection_week.groupby(breakout_event_group).cumsum()
        has_no_rejection_yet = (rejections_in_group_so_far == 0)
        context_df['filter_has_no_rejection_yet_W'] = has_no_rejection_yet

        # 2.5 生成最终的“突破观察”信号 (高质量信号)
        signal_breakout_trigger = is_breakout_initiation & has_no_rejection_yet
        context_df['signal_breakout_trigger_W'] = signal_breakout_trigger

        print(f"【合成-步骤1】近期有吸筹状态(state_had_recent_accumulation_W) 周数: {context_df['state_had_recent_accumulation_W'].sum()}")
        print(f"【合成-步骤2】周线突破事件(event_is_breakout_week_W) 周数: {context_df['event_is_breakout_week_W'].sum()}")
        print(f"【合成-步骤3】突破启动信号(signal_breakout_initiation_W) 周数: {context_df['signal_breakout_initiation_W'].sum()}")
        print(f"【合成-步骤4】至今无拒绝信号(filter_has_no_rejection_yet_W) 周数: {context_df['filter_has_no_rejection_yet_W'].sum()}")
        print(f"【合成-步骤5】最终突破观察信号(signal_breakout_trigger_W) 周数: {context_df['signal_breakout_trigger_W'].sum()}")
        
        return context_df

    def _calculate_all_playbooks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V18.0】计算所有独立的剧本和诊断信号，作为信号合成的原材料。
        """
        print("\n---【周线战略层(V21.0) - 步骤1: 计算所有基础剧本】---")
        playbook_ma20_turn_up = self._playbook_ma20_turn_up(df, self.playbook_params.get('ma20_turn_up_playbook', {}))
        playbook_early_uptrend = self._playbook_early_uptrend(df, self.playbook_params.get('early_uptrend_playbook', {}))
        playbook_classic = self._playbook_classic_breakout(df, self.playbook_params.get('classic_breakout_playbook', {}))
        playbook_ma_uptrend = self._playbook_check_ma_uptrend(df, self.playbook_params.get('ma_uptrend_playbook', {}))
        playbook_box_breakout = self._playbook_box_consolidation_breakout(df, self.playbook_params.get('box_consolidation_breakout_playbook', {}))
        playbook_bias_rebound = self._playbook_oversold_rebound_bias(df, self.playbook_params.get('oversold_rebound_bias_playbook', {}))
        washout_score = self._playbook_calculate_washout_score(df, self.playbook_params.get('washout_score_playbook', {}))
        rejection_signal = self._playbook_check_rejection_filters(df, self.playbook_params.get('rejection_filter_playbook', {}))

        context_df = pd.DataFrame(index=df.index)
        context_df['playbook_ma20_turn_up_W'] = playbook_ma20_turn_up
        context_df['playbook_early_uptrend_W'] = playbook_early_uptrend
        context_df['playbook_classic_breakout_W'] = playbook_classic
        context_df['playbook_ma_uptrend_W'] = playbook_ma_uptrend
        context_df['playbook_box_breakout_W'] = playbook_box_breakout
        context_df['playbook_bias_rebound_W'] = playbook_bias_rebound
        context_df['washout_score_W'] = washout_score
        context_df['rejection_signal_W'] = rejection_signal
        
        print("---【周线战略层(V21.0) - 剧本计算总结】---")
        print(f"【剧本-趋势】稳定上升趋势 最终触发周数: {playbook_ma_uptrend.sum()}")
        print(f"【剧本-箱体】专业箱体突破 最终触发周数: {playbook_box_breakout.sum()}")
        print(f"【剧本-反弹】BIAS超跌反弹 最终触发周数: {playbook_bias_rebound.sum()}")
        print(f"【诊断-洗盘】检测到洗盘行为的周数: {(washout_score > 0).sum()} (最高分: {washout_score.max()})")
        print(f"【诊断-风险】检测到压力位拒绝的周数: {(rejection_signal < 0).sum()}")
        
        return context_df

    def _playbook_ma20_turn_up(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【V18.3 适配健壮诊断版】
        剧本：识别20周均线拐头向上，且股价站上均线。
        """
        playbook_name = 'state_ma20_turn_up_W'
        ma_periods = params.get('ma_periods', [5, 8, 13, 21, 34, 55, 233])
        target_ma_period = 21
        
        ema_col = f'EMA_{target_ma_period}_W'
        slope_col = f'{ema_col}_slope'
        
        close_col = 'close_W' 
        if close_col not in df.columns:
            if 'close' in df.columns:
                close_col = 'close'
                print(f"  [警告] 剧本: {playbook_name} 未找到 '{close_col}'，回退使用 'close'。请检查周线聚合逻辑。")
            else:
                print(f"  [错误] 剧本: {playbook_name} 无法找到收盘价列 ('close_W' 或 'close')。")
                return pd.Series(False, index=df.index, name=playbook_name)

        if ema_col in df.columns:
            df[slope_col] = df[ema_col].diff(1)
        else:
            print(f"  [警告] 剧本: {playbook_name} 未找到指标列 '{ema_col}'，可能数据量不足。")
            df[slope_col] = np.nan

        if all(c in df.columns for c in [slope_col, close_col, ema_col]):
            condition1 = df[slope_col] > 0
            condition2 = df[close_col] > df[ema_col]
            final_signal = condition1 & condition2
            final_signal = final_signal.fillna(False)
        else:
            final_signal = pd.Series(False, index=df.index)

        final_signal.name = playbook_name

        print(f"    - 条件1 (均线斜率>0): {final_signal[condition1.fillna(False)].sum() if 'condition1' in locals() else 0} 次")
        print(f"    - 条件2 (收盘价>均线): {final_signal[condition2.fillna(False)].sum() if 'condition2' in locals() else 0} 次")
        print(f"    - 最终信号 (条件1 & 条件2): {final_signal.sum()} 次")
        
        return final_signal

    def _playbook_early_uptrend(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """【战略剧本-短期核心 V2.1】: 捕捉周线趋势反转的早期“上拐”信号。"""
        print("  [诊断] 剧本: _playbook_early_uptrend")
        if not params.get('enabled', True): return pd.Series(False, index=df.index)

        indicator_cfg = self.params.get('feature_engineering_params', {}).get('indicators', {})
        
        ema_periods = indicator_cfg.get('ema', {}).get('periods', [5, 10, 20, 60])
        short_ma_period = ema_periods[2] if len(ema_periods) > 2 else 10
        mid_ma_period = ema_periods[3] if len(ema_periods) > 3 else 20
        short_ma_col = f'EMA_{short_ma_period}_W'
        mid_ma_col = f'EMA_{mid_ma_period}_W'

        macd_params_raw = indicator_cfg.get('macd', {}).get('periods', [12, 26, 9])
        macd_params = macd_params_raw[0] if isinstance(macd_params_raw[0], list) else macd_params_raw
        p_fast, p_slow, p_signal = macd_params[0], macd_params[1], macd_params[2]
        macd_col = f'MACD_{p_fast}_{p_slow}_{p_signal}_W'
        macd_hist_col = f'MACDh_{p_fast}_{p_slow}_{p_signal}_W'

        required_cols = [short_ma_col, mid_ma_col, macd_col, macd_hist_col, 'close_W']
        if not all(col in df.columns for col in required_cols):
            missing_cols = [c for c in required_cols if c not in df.columns]
            print(f"    - 缺失必要列: {', '.join(missing_cols)}")
            return pd.Series(False, index=df.index)

        ma_slope = df[short_ma_col].diff()
        ma_is_up = ma_slope > 0
        ma_turning_up = (ma_slope > 0) & (ma_slope.shift(1) <= 0)
        price_cross_ma = (df['close_W'] > df[mid_ma_col]) & (df['close_W'].shift(1) <= df[mid_ma_col].shift(1))
        macd_cross_zero_nearby = (df[macd_hist_col] > 0) & (df[macd_hist_col].shift(1) <= 0) & (df[macd_col].abs() < df['close_W'] * 0.05)
        signal = (ma_turning_up | price_cross_ma) & macd_cross_zero_nearby
        in_early_uptrend = (df[short_ma_col] > df[mid_ma_col]) & ma_is_up
        final_signal = (signal | in_early_uptrend).fillna(False)
        
        print(f"    - 子信号1 (拐点信号): {signal.sum()} 次")
        print(f"    - 子信号2 (早期趋势延续): {in_early_uptrend.sum()} 次")
        print(f"    - 最终信号 (子信号1 | 子信号2): {final_signal.sum()} 次")
        return final_signal

    def _playbook_classic_breakout(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """【战略剧本-加速】: 经典高点突破"""
        if not params.get('enabled', True): return pd.Series(False, index=df.index)
        lookback_weeks, volume_multiplier = params.get('lookback_weeks', 26), params.get('volume_multiplier', 1.5)
        if not all(c in df.columns for c in ['high_W', 'volume_W', 'close_W']):
            print("  [诊断] 剧本: _playbook_classic_breakout - 缺失基础K线数据列")
            return pd.Series(False, index=df.index)
        period_high = df['high_W'].shift(1).rolling(window=lookback_weeks).max()
        avg_volume = df['volume_W'].shift(1).rolling(window=lookback_weeks).mean()
        is_price_breakout = df['close_W'] > period_high
        is_volume_breakout = df['volume_W'] > (avg_volume * volume_multiplier)
        return (is_price_breakout & is_volume_breakout).fillna(False)

    def _playbook_check_ma_uptrend(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """【战略剧本-趋势】: 检查均线多头排列"""
        print("  [诊断] 剧本: _playbook_check_ma_uptrend")
        if not params.get('enabled', True): return pd.Series(False, index=df.index)
        
        indicator_cfg = self.params.get('feature_engineering_params', {}).get('indicators', {})
        ema_periods = indicator_cfg.get('ema', {}).get('periods', [10, 20, 60])
        short_ma = ema_periods[1] if len(ema_periods) > 1 else 10
        mid_ma = ema_periods[2] if len(ema_periods) > 2 else 20
        long_ma = ema_periods[4] if len(ema_periods) > 4 else 60
        
        short_col, mid_col, long_col = f'EMA_{short_ma}_W', f'EMA_{mid_ma}_W', f'EMA_{long_ma}_W'
        if not all(c in df.columns for c in [short_col, mid_col, long_col, 'close_W']):
            missing_cols = [c for c in [short_col, mid_col, long_col] if c not in df.columns]
            print(f"    - 缺失必要列: {', '.join(missing_cols)}")
            return pd.Series(False, index=df.index)
        ma_aligned = (df[short_col] > df[mid_col]) & (df[mid_col] > df[long_col])
        price_above_support = df['close_W'] > df[mid_col]
        return (ma_aligned & price_above_support).fillna(False)

    def _playbook_oversold_rebound_bias(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """【战略剧本-反弹 V1.0】: 利用BIAS指标捕捉周线级别的超跌反弹机会。"""
        print("  [诊断] 剧本: _playbook_oversold_rebound_bias")
        if not params.get('enabled', True): return pd.Series(False, index=df.index)

        col_names = self._get_dynamic_col_names()
        bias_col = col_names['bias']
        
        if not self._check_dependencies(df, [bias_col]):
            return pd.Series(False, index=df.index)

        oversold_threshold = params.get('oversold_threshold', -10)
        rebound_trigger = params.get('rebound_trigger', -7)

        was_oversold = (df[bias_col].shift(1) < oversold_threshold)
        is_rebounding = (df[bias_col] > rebound_trigger) & (df[bias_col].shift(1) <= rebound_trigger)

        final_signal = (was_oversold & is_rebounding).fillna(False)
        print(f"    - [BIAS超跌反弹] 最终信号: {final_signal.sum()} 次")
        return final_signal

    def _playbook_calculate_washout_score(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """【诊断剧本-洗盘评分 V1.0】: 量化周线级别的洗盘行为。"""
        print("  [诊断] 剧本: _playbook_calculate_washout_score")
        if not params.get('enabled', False): return pd.Series(0, index=df.index)
        
        washout_score = pd.Series(0, index=df.index)
        support_level = self._get_weekly_support_level(df, params)
        if support_level is None:
            print("    - [洗盘评分] 因无法确定支撑位而跳过。")
            return washout_score

        washout_intraday = (df['low_W'] < support_level) & (df['close_W'] > support_level)
        washout_interday = (df['close_W'] > support_level) & (df['close_W'].shift(1) < support_level.shift(1))
        
        drift_lookback = params.get('drift_lookback_period', 3)
        was_below_recently = (df['close_W'].shift(1) < support_level.shift(1)).rolling(window=drift_lookback, min_periods=1).sum() > 0
        washout_drift = (df['close_W'] > support_level) & was_below_recently
        
        lookback = params.get('bull_trap_lookback_period', 8)
        drop_threshold = params.get('bull_trap_drop_threshold', 0.05)
        recent_peak = df['high_W'].shift(1).rolling(window=lookback).max()
        is_in_trap_zone = df['close_W'] < recent_peak * (1 - drop_threshold)
        is_recovering_from_trap = df['close_W'] > df['close_W'].shift(1)
        washout_bull_trap = is_in_trap_zone & is_recovering_from_trap
        
        avg_period = params.get('volume_avg_period', 20)
        threshold = params.get('volume_contraction_threshold', 0.7)
        avg_volume = df['volume_W'].shift(1).rolling(window=avg_period).mean()
        is_volume_contracted = df['volume_W'] < avg_volume * threshold
        washout_volume_contraction = (washout_interday | washout_drift) & is_volume_contracted.shift(1).fillna(False)
        
        washout_score += washout_intraday.astype(int)
        washout_score += washout_interday.astype(int)
        washout_score += washout_drift.astype(int)
        washout_score += washout_bull_trap.astype(int)
        washout_score += washout_volume_contraction.astype(int)
        
        return washout_score.fillna(0)

    def _get_weekly_support_level(self, df: pd.DataFrame, params: dict) -> Optional[pd.Series]:
        """【V2.0 动态自适应版】辅助函数: 获取周线级别的支撑位。"""
        support_type = params.get('support_type', 'MA')
        print(f"    - [支撑位分析] 使用 '{support_type}' 模式寻找支撑。")
        support_level = pd.Series(np.nan, index=df.index)

        if support_type == 'MA':
            ma_period = params.get('support_ma_period', 21)
            ma_col = f'EMA_{ma_period}_W'
            # print(f"      - [依赖检查] 需要列: '{ma_col}'")
            if ma_col not in df.columns:
                print(f"      - [依赖错误] 致命错误: 列 '{ma_col}' 在DataFrame中不存在！")
                return None
            # print(f"      - [依赖成功] 已找到列: '{ma_col}'")
            support_level = df[ma_col]

        elif support_type == 'BOX':
            boll_period = params.get('boll_period', 20)
            boll_std = params.get('boll_std', 2.0)
            bbw_col = f"BBW_{boll_period}_{float(boll_std)}_W"
            volatility_method = params.get('volatility_method', 'QUANTILE').upper()
            quantile_level = params.get('quantile_level', 0.25)
            static_threshold = params.get('box_volatility_threshold', 0.25)

            required_cols = [bbw_col, 'low_W']
            # print(f"      - [依赖检查] 需要列: {required_cols}")
            if not self._check_dependencies(df, required_cols):
                return None
            # print(f"      - [依赖成功] 已找到所有依赖列。")
            
            threshold = 0.0
            if volatility_method == 'QUANTILE':
                threshold = df[bbw_col].quantile(quantile_level)
                print(f"      - [BOX模式分析] 使用 'QUANTILE' 模式 (level={quantile_level})。")
                print(f"      - [BOX模式分析] 为该股票动态计算出的波动率阈值为: {threshold:.4f}")
            elif volatility_method == 'STATIC':
                threshold = static_threshold
                print(f"      - [BOX模式分析] 使用 'STATIC' 模式，固定阈值: {threshold:.4f}")
            else:
                print(f"      - [BOX模式分析] 错误: 未知的 volatility_method: '{volatility_method}'。")
                return None

            # print(f"      - [BOX模式分析] '{bbw_col}' 列的统计数据:")
            # print(df[bbw_col].describe().to_string())
            
            is_consolidating = df[bbw_col] < threshold
            consolidation_count = is_consolidating.sum()
            # print(f"      - [BOX模式分析] 识别出 {consolidation_count} 个盘整周期 (is_consolidating 为 True 的数量)。")

            if consolidation_count > 0:
                box_period = params.get('box_period', 26)
                box_bottom = df['low_W'].rolling(window=box_period, min_periods=1).min()
                support_level = box_bottom.where(is_consolidating, np.nan)
        
        if support_level.isnull().all():
            print(f"    - [洗盘评分] 警告: 无法计算出任何有效的支撑位（结果全为NaN）。")
            if support_type == 'BOX':
                min_bbw = df[bbw_col].min()
                print(f"    - [策略建议] '{bbw_col}' 的历史最小值 ({min_bbw:.4f}) 可能高于计算出的阈值 ({threshold:.4f})。")
                if volatility_method == 'QUANTILE':
                    print(f"    - [策略建议] 可以考虑在配置文件中适度提高 'quantile_level' 的值 (例如从 {quantile_level} 到 0.3 或 0.4)。")
                else:
                    print(f"    - [策略建议] 可以考虑在配置文件中提高 'box_volatility_threshold' 的值。")
            return None
        
        print(f"    - [支撑位分析] 成功计算出支撑位。")
        return support_level.ffill()

    def _playbook_check_rejection_filters(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """【诊断剧本-风险过滤 V1.0】: 识别均线和箱体压力位的拒绝信号。"""
        print("  [诊断] 剧本: _playbook_check_rejection_filters")
        if not params.get('enabled', True): return pd.Series(0, index=df.index)
        
        ma_period = params.get('ma_period', 21)
        ma_col = f'EMA_{ma_period}_W'
        ma_rejection = self._check_resistance_rejection(df, ma_col, params)

        box_lookback = params.get('box_lookback_period', 52)
        box_resistance_col = f'box_top_{box_lookback}W_resistance'
        df[box_resistance_col] = df['high_W'].shift(1).rolling(window=box_lookback, min_periods=int(box_lookback * 0.8)).max()
        box_rejection = self._check_resistance_rejection(df, box_resistance_col, params)

        final_signal = pd.Series(0, index=df.index)
        final_signal[ma_rejection] -= 1
        final_signal[box_rejection] -= 2
        return final_signal

    def _check_resistance_rejection(self, df: pd.DataFrame, resistance_col: str, params: dict) -> pd.Series:
        """辅助函数: 检查在给定压力列上的拒绝信号"""
        volume_multiplier = params.get('volume_multiplier', 1.5)
        col_names = self._get_dynamic_col_names()
        vol_ma_col = col_names['vol_ma']
        
        if not self._check_dependencies(df, [resistance_col, vol_ma_col, 'open_W', 'high_W', 'close_W', 'volume_W']):
            return pd.Series(False, index=df.index)

        is_near_resistance = df['high_W'] >= df[resistance_col]
        is_long_upper_shadow = (df['high_W'] - df[['open_W', 'close_W']].max(axis=1)) > (df['high_W'] - df['low_W']) * 0.5
        is_high_volume = df['volume_W'] > df[vol_ma_col] * volume_multiplier
        is_closing_lower = df['close_W'] < df[['open_W', 'close_W']].mean(axis=1)

        return (is_near_resistance & is_long_upper_shadow & is_high_volume & is_closing_lower).fillna(False)

    def _playbook_box_consolidation_breakout(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【战略剧本-箱体 V2.4 深度调试版】: 增加详细的日志输出，诊断为何无法产生信号。
        """
        print("  [诊断] 剧本: _playbook_box_consolidation_breakout (V2.4 深度调试版)")
        if not params.get('enabled', True): 
            return pd.Series(False, index=df.index)

        # --- 参数部分保持不变 ---
        quantile_level = params.get('quantile_level', 0.40)
        boll_period = params.get('boll_period', 20)
        boll_std = params.get('boll_std', 2.0)
        box_period = params.get('box_period', 26)
        volume_multiplier = params.get('volume_multiplier', 1.3)
        state_window = params.get('state_window', 5)
        state_threshold = params.get('state_threshold', 2)

        # --- 依赖检查 ---
        bbw_col = f"BBW_{boll_period}_{float(boll_std)}_W"
        vol_ma_period = self.indicator_cfg.get('vol_ma', {}).get('periods', [20])[0]
        vol_ma_col = f"VOL_MA_{vol_ma_period}_W"
        required_cols = ['close_W', 'high_W', 'volume_W', bbw_col, vol_ma_col]
        if not self._check_dependencies(df, required_cols):
            return pd.Series(False, index=df.index)

        # --- 核心计算部分保持不变 ---
        rolling_quantile = df[bbw_col].rolling(window=box_period * 2, min_periods=box_period).quantile(quantile_level)
        is_low_volatility_week = df[bbw_col] < rolling_quantile
        consolidation_state_count = is_low_volatility_week.rolling(window=state_window).sum()
        state_is_in_consolidation = consolidation_state_count >= state_threshold
        box_high = df['high_W'].shift(1).rolling(window=box_period).max()
        is_price_breakout = df['close_W'] > box_high
        was_in_consolidation_state = state_is_in_consolidation.shift(1).fillna(False)
        is_volume_breakout = df['volume_W'] > (df[vol_ma_col].shift(1) * volume_multiplier)
        final_signal = (was_in_consolidation_state & is_price_breakout & is_volume_breakout).fillna(False)

        # ▼▼▼【代码修改】: 增加深度调试日志 ▼▼▼
        # 如果最终没有产生任何信号，就打印最后10周的详细状态，以便分析
        if not final_signal.any():
            print("    - [箱体突破分析] 未找到任何突破信号。打印最后10周的状态进行诊断:")
            
            # 创建一个临时的DataFrame用于展示
            debug_df = pd.DataFrame({
                'Close': df['close_W'],
                'BBW': df[bbw_col],
                'BBW_Quantile': rolling_quantile,
                'Is_Low_Vol': is_low_volatility_week,
                'Consol_Count': consolidation_state_count,
                'In_Consol_State': state_is_in_consolidation,
                'Was_In_Consol': was_in_consolidation_state,
                'Box_High': box_high,
                'Price_Breakout': is_price_breakout,
                'Volume_Breakout': is_volume_breakout,
                'FINAL_SIGNAL': final_signal
            })
            # 使用 to_string() 保证所有列都能显示出来
            print(debug_df.tail(10).to_string())
        else:
            print(f"    - [专业箱体突破] 成功找到 {final_signal.sum()} 个突破信号!")
        # ▲▲▲【代码修改】: 修改结束 ▲▲▲
            
        return final_signal

    def _get_dynamic_col_names(self) -> dict:
        """【V17.0】集中动态构建所有可能用到的列名"""
        ema_periods = self.indicator_cfg.get('ema', {}).get('periods', [5, 8, 13, 21, 34, 55, 233])
        macd_params_raw = self.indicator_cfg.get('macd', {}).get('periods', [12, 26, 9])
        macd_params = macd_params_raw[0] if isinstance(macd_params_raw[0], list) else macd_params_raw
        p_fast, p_slow, p_signal = macd_params
        atr_cfg = self.indicator_cfg.get('atrr', {})
        atr_period = atr_cfg.get('periods', [14])[0] if atr_cfg.get('periods') else 14
        consolidation_cfg = self.indicator_cfg.get('consolidation_period', {})
        consolidation_period = consolidation_cfg.get('periods', [26])[0] if consolidation_cfg.get('periods') else 26
        bias_cfg = self.indicator_cfg.get('bias', {})
        bias_period = bias_cfg.get('periods', [21])[0] if bias_cfg.get('periods') else 21
        bbands_cfg = self.indicator_cfg.get('boll_bands_and_width', {})
        bbands_period = bbands_cfg.get('periods', [20])[0] if bbands_cfg.get('periods') else 20
        bbands_std = bbands_cfg.get('std', bbands_cfg.get('std', 2.0))
        vol_ma_cfg = self.indicator_cfg.get('vol_ma', {})
        vol_ma_period = vol_ma_cfg.get('periods', [20])[0] if vol_ma_cfg.get('periods') else 20

        return {
            'short_ma': f'EMA_{ema_periods[2] if len(ema_periods) > 2 else 10}_W',
            'mid_ma': f'EMA_{ema_periods[3] if len(ema_periods) > 3 else 20}_W',
            'long_ma_trend': f'EMA_{ema_periods[4] if len(ema_periods) > 4 else 60}_W',
            'long_ma_reversal': f'EMA_{next((p for p in ema_periods if p >= 100), 120)}_W',
            'macd': f'MACD_{p_fast}_{p_slow}_{p_signal}_W',
            'macd_hist': f'MACDh_{p_fast}_{p_slow}_{p_signal}_W',
            'atrr': f'ATRr_{atr_period}_W',
            'consolidation_high': 'dynamic_consolidation_high_W',
            'consolidation_low': 'dynamic_consolidation_low_W',
            'consolidation_avg_vol': 'dynamic_consolidation_avg_vol_W',
            'consolidation_duration': 'dynamic_consolidation_duration_W',
            'bias': f'BIAS_{bias_period}_W',
            'bbw': f'BBW_{bbands_period}_{float(bbands_std)}_W',
            'vol_ma': f'VOL_MA_{vol_ma_period}_W'
        }

    def _check_dependencies(self, df: pd.DataFrame, required_cols: List[str]) -> bool:
        """【V17.0】检查数据依赖项，如果缺失则打印日志并返回False"""
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            print(f"    - [依赖检查] 剧本跳过，因缺失必要列: {', '.join(missing_cols)}")
            if any('consolidation' in col for col in missing_cols):
                print("    - [提示] consolidation列缺失通常意味着 'config/...' 文件中未配置 'consolidation_period' 指标。")
            if any('BIAS' in col for col in missing_cols):
                print("    - [提示] BIAS列缺失通常意味着 'config/...' 文件中未配置 'bias' 指标。")
            if any('ATRr' in col for col in missing_cols):
                print("    - [提示] ATRr列缺失通常意味着 'config/...' 文件中未配置 'atrr' 指标。")
            if any('BBW' in col for col in missing_cols):
                print("    - [提示] BBW列缺失通常意味着 'config/...' 文件中未配置 'bbands' 指标。")
            return False
        return True
