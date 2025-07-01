# 文件: strategies/weekly_trend_follow_strategy.py
# 版本: V2.8 - 真正完整终极调试版

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

    def __init__(self, config: dict):
        """
        初始化周线策略。
        """
        self.params = config
        self.indicator_cfg = self.params.get('feature_engineering_params', {}).get('indicators', {})
        self.playbook_params = self.params.get('strategy_playbooks', {})

    def apply_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【核心策略应用函数 V21.0 同步版】
        """
        if df is None or df.empty:
            logger.warning("周线策略输入DataFrame为空，无法应用。")
            return pd.DataFrame()

        # --- 步骤 1: 计算所有基础剧本和诊断信号 ---
        context_df = self._calculate_all_playbooks(df)
        
        print("\n---【周线战略层(V2.8 信号合成与提纯)诊断】---")
        # --- 步骤 2: 信号合成层 (逻辑保持不变) ---
        washout_threshold = self.playbook_params.get('washout_score_playbook', {}).get('score_threshold', 2)
        accumulation_playbooks = ['playbook_early_uptrend_W', 'playbook_bias_rebound_W', 'playbook_ma20_turn_up_W']
        # 解释: TRIX金叉和Coppock底部反转都是非常重要的长期趋势启动或反转信号，将它们视为“吸筹/建仓”阶段的一部分，可以增强我们对趋势早期阶段的识别能力。
        accumulation_playbooks = [
            'playbook_early_uptrend_W', 
            'playbook_bias_rebound_W', 
            'playbook_ma20_turn_up_W',
            'playbook_trix_cross_W',          # 新增TRIX金叉剧本
            'playbook_coppock_reversal_W'   # 新增Coppock反转剧本
        ]
        is_accumulation_signal = (context_df[accumulation_playbooks].any(axis=1) | (context_df['washout_score_W'] >= washout_threshold))
        recent_accumulation_window = 12 
        had_recent_accumulation = is_accumulation_signal.rolling(window=recent_accumulation_window, min_periods=1).sum().shift(1) > 0
        context_df['state_had_recent_accumulation_W'] = had_recent_accumulation.fillna(False)
        is_breakout_event = (context_df['playbook_classic_breakout_W'] | context_df['playbook_box_breakout_W'])
        context_df['event_is_breakout_week_W'] = is_breakout_event
        is_breakout_initiation = context_df['state_had_recent_accumulation_W'] & context_df['event_is_breakout_week_W']
        signal_breakout_initiation = (is_breakout_initiation) & (is_breakout_initiation.shift(1) == False)
        context_df['signal_breakout_initiation_W'] = signal_breakout_initiation
        is_rejection_week = context_df['rejection_signal_W'] < 0
        breakout_event_group = context_df['signal_breakout_initiation_W'].cumsum()
        rejections_in_group_so_far = is_rejection_week.groupby(breakout_event_group).cumsum()
        has_no_rejection_yet = (rejections_in_group_so_far == 0)
        context_df['filter_has_no_rejection_yet_W'] = has_no_rejection_yet
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
        print("\n" + "="*80)
        print(f"---【周线战略层(V2.8 终极调试版) - 检查最新一周: {df.index[-1].date()}】---")
        print("="*80)
        
        playbook_ma20_turn_up = self._playbook_ma20_turn_up(df, self.playbook_params.get('ma20_turn_up_playbook', {}))
        playbook_early_uptrend = self._playbook_early_uptrend(df, self.playbook_params.get('early_uptrend_playbook', {}))
        playbook_classic = self._playbook_classic_breakout(df, self.playbook_params.get('classic_breakout_playbook', {}))
        playbook_ma_uptrend = self._playbook_check_ma_uptrend(df, self.playbook_params.get('ma_uptrend_playbook', {}))
        playbook_box_breakout = self._playbook_box_consolidation_breakout(df, self.playbook_params.get('box_consolidation_breakout_playbook', {}))
        playbook_bias_rebound = self._playbook_oversold_rebound_bias(df, self.playbook_params.get('oversold_rebound_bias_playbook', {}))
        washout_score = self._playbook_calculate_washout_score(df, self.playbook_params.get('washout_score_playbook', {}))
        rejection_signal = self._playbook_check_rejection_filters(df, self.playbook_params.get('rejection_filter_playbook', {}))
        playbook_trix_cross = self._playbook_trix_golden_cross(df, self.playbook_params.get('trix_golden_cross_playbook', {}))
        playbook_coppock_reversal = self._playbook_coppock_bottom_reversal(df, self.playbook_params.get('coppock_reversal_playbook', {}))

        context_df = pd.DataFrame(index=df.index)
        context_df['playbook_ma20_turn_up_W'] = playbook_ma20_turn_up
        context_df['playbook_early_uptrend_W'] = playbook_early_uptrend
        context_df['playbook_classic_breakout_W'] = playbook_classic
        context_df['playbook_ma_uptrend_W'] = playbook_ma_uptrend
        context_df['playbook_box_breakout_W'] = playbook_box_breakout
        context_df['playbook_bias_rebound_W'] = playbook_bias_rebound
        context_df['washout_score_W'] = washout_score
        context_df['rejection_signal_W'] = rejection_signal
        
        print("\n---【周线战略层(V2.8) - 剧本计算总结】---")
        print(f"【剧本-MA20拐头】触发周数: {playbook_ma20_turn_up.sum()}")
        print(f"【剧本-早期趋势】触发周数: {playbook_early_uptrend.sum()}")
        print(f"【剧本-经典突破】触发周数: {playbook_classic.sum()}")
        print(f"【剧本-稳定趋势】触发周数: {playbook_ma_uptrend.sum()}")
        print(f"【剧本-箱体突破】触发周数: {playbook_box_breakout.sum()}")
        print(f"【剧本-BIAS反弹】触发周数: {playbook_bias_rebound.sum()}")
        print(f"【诊断-洗盘】有分数的周数: {(washout_score > 0).sum()} (最高分: {washout_score.max()})")
        print(f"【诊断-风险】有拒绝信号的周数: {(rejection_signal < 0).sum()}")
        
        return context_df

    def _playbook_ma20_turn_up(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """剧本：识别20周均线拐头向上"""
        print("\n--- 剧本检查: [MA20拐头向上] ---")
        if not params.get('enabled', True):
            print("    - 结论: [未启用]")
            return pd.Series(False, index=df.index)

        target_ma_period = 21
        ema_col = f'EMA_{target_ma_period}_W'
        slope_col = f'{ema_col}_slope'
        close_col = 'close_W'
        
        if not self._check_dependencies(df, [ema_col, close_col], log_details=True):
            print(f"    - 结论: [失败] 缺少必要列")
            return pd.Series(False, index=df.index)

        df[slope_col] = df[ema_col].diff(1)
        condition1 = df[slope_col] > 0
        condition2 = df[close_col] > df[ema_col]
        final_signal = condition1 & condition2

        # 调试最新一周
        last = df.iloc[-1]
        c1_last = condition1.iloc[-1]
        c2_last = condition2.iloc[-1]
        print(f"    - 条件1 (均线斜率 > 0): {'[✓]' if c1_last else '[✗]'} (实际值: {last.get(slope_col, float('nan')):.2f})")
        print(f"    - 条件2 (收盘价 > 均线): {'[✓]' if c2_last else '[✗]'} (收盘价: {last.get(close_col, float('nan')):.2f} vs 均线: {last.get(ema_col, float('nan')):.2f})")
        print(f"    - 结论: 最新一周信号为 [{'触发' if final_signal.iloc[-1] else '未触发'}]")
        
        return final_signal.fillna(False)

    def _playbook_early_uptrend(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """剧本：捕捉周线趋势反转的早期“上拐”信号"""
        print("\n--- 剧本检查: [早期上升趋势] ---")
        if not params.get('enabled', True):
            print("    - 结论: [未启用]")
            return pd.Series(False, index=df.index)

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
        
        if not self._check_dependencies(df, required_cols, log_details=True):
            print(f"    - 结论: [失败] 缺少必要列")
            return pd.Series(False, index=df.index)

        ma_slope = df[short_ma_col].diff()
        ma_is_up = ma_slope > 0
        ma_turning_up = (ma_slope > 0) & (ma_slope.shift(1) <= 0)
        price_cross_ma = (df['close_W'] > df[mid_ma_col]) & (df['close_W'].shift(1) <= df[mid_ma_col].shift(1))
        macd_cross_zero_nearby = (df[macd_hist_col] > 0) & (df[macd_hist_col].shift(1) <= 0) & (df[macd_col].abs() < df['close_W'] * 0.05)
        
        signal = (ma_turning_up | price_cross_ma) & macd_cross_zero_nearby
        in_early_uptrend = (df[short_ma_col] > df[mid_ma_col]) & ma_is_up
        final_signal = (signal | in_early_uptrend)

        # 调试最新一周
        s_last = signal.iloc[-1]
        ieu_last = in_early_uptrend.iloc[-1]
        print(f"    - 子信号1 (拐点信号): {'[✓]' if s_last else '[✗]'}")
        print(f"    - 子信号2 (趋势延续): {'[✓]' if ieu_last else '[✗]'}")
        print(f"    - 结论: 最新一周信号为 [{'触发' if final_signal.iloc[-1] else '未触发'}] (逻辑: 拐点 OR 延续)")

        return final_signal.fillna(False)

    def _playbook_classic_breakout(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """剧本：经典高点突破"""
        print("\n--- 剧本检查: [经典高点突破] ---")
        if not params.get('enabled', True):
            print("    - 结论: [未启用]")
            return pd.Series(False, index=df.index)
        
        lookback_weeks, volume_multiplier = params.get('lookback_weeks', 26), params.get('volume_multiplier', 1.5)
        if not self._check_dependencies(df, ['high_W', 'volume_W', 'close_W'], log_details=True):
            print(f"    - 结论: [失败] 缺少必要列")
            return pd.Series(False, index=df.index)

        period_high = df['high_W'].shift(1).rolling(window=lookback_weeks).max()
        avg_volume = df['volume_W'].shift(1).rolling(window=lookback_weeks).mean()
        is_price_breakout = df['close_W'] > period_high
        is_volume_breakout = df['volume_W'] > (avg_volume * volume_multiplier)
        final_signal = is_price_breakout & is_volume_breakout

        # 调试最新一周
        last = df.iloc[-1]
        ph_last = period_high.iloc[-1]
        av_last = avg_volume.iloc[-1]
        pb_last = is_price_breakout.iloc[-1]
        vb_last = is_volume_breakout.iloc[-1]
        print(f"    - 条件1 (价格突破): {'[✓]' if pb_last else '[✗]'} (收盘价: {last.get('close_W', float('nan')):.2f} vs 前{lookback_weeks}周高点: {ph_last:.2f})")
        print(f"    - 条件2 (放量突破): {'[✓]' if vb_last else '[✗]'} (成交量: {last.get('volume_W', 0):.0f} vs 阈值: {(av_last * volume_multiplier):.0f})")
        print(f"    - 结论: 最新一周信号为 [{'触发' if final_signal.iloc[-1] else '未触发'}]")

        return final_signal.fillna(False)

    def _playbook_check_ma_uptrend(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """剧本：检查均线多头排列"""
        print("\n--- 剧本检查: [均线多头排列] ---")
        if not params.get('enabled', True):
            print("    - 结论: [未启用]")
            return pd.Series(False, index=df.index)
        
        indicator_cfg = self.params.get('feature_engineering_params', {}).get('indicators', {})
        ema_periods = indicator_cfg.get('ema', {}).get('periods', [10, 20, 60])
        short_ma = ema_periods[1] if len(ema_periods) > 1 else 10
        mid_ma = ema_periods[2] if len(ema_periods) > 2 else 20
        long_ma = ema_periods[4] if len(ema_periods) > 4 else 60
        short_col, mid_col, long_col = f'EMA_{short_ma}_W', f'EMA_{mid_ma}_W', f'EMA_{long_ma}_W'
        
        if not self._check_dependencies(df, [short_col, mid_col, long_col, 'close_W'], log_details=True):
            print(f"    - 结论: [失败] 缺少必要列")
            return pd.Series(False, index=df.index)

        ma_aligned = (df[short_col] > df[mid_col]) & (df[mid_col] > df[long_col])
        price_above_support = df['close_W'] > df[mid_col]
        final_signal = ma_aligned & price_above_support

        # 调试最新一周
        last = df.iloc[-1]
        ma_last = ma_aligned.iloc[-1]
        pas_last = price_above_support.iloc[-1]
        print(f"    - 条件1 (均线多头): {'[✓]' if ma_last else '[✗]'} (EMA{short_ma}: {last.get(short_col, 0):.2f} > EMA{mid_ma}: {last.get(mid_col, 0):.2f} > EMA{long_ma}: {last.get(long_col, 0):.2f})")
        print(f"    - 条件2 (股价在支撑上): {'[✓]' if pas_last else '[✗]'} (收盘价: {last.get('close_W', 0):.2f} > EMA{mid_ma}: {last.get(mid_col, 0):.2f})")
        print(f"    - 结论: 最新一周信号为 [{'触发' if final_signal.iloc[-1] else '未触发'}]")

        return final_signal.fillna(False)

    def _playbook_oversold_rebound_bias(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """剧本：利用BIAS指标捕捉周线级别的超跌反弹机会"""
        print("\n--- 剧本检查: [BIAS超跌反弹] ---")
        if not params.get('enabled', True):
            print("    - 结论: [未启用]")
            return pd.Series(False, index=df.index)

        bias_period = self.indicator_cfg.get('bias', {}).get('periods', [21])[0]
        bias_col = f'BIAS_{bias_period}_W'
        
        if not self._check_dependencies(df, [bias_col], log_details=True):
            print(f"    - 结论: [失败] 缺少必要列 {bias_col}")
            return pd.Series(False, index=df.index)

        oversold_threshold = params.get('oversold_threshold', -10)
        rebound_trigger = params.get('rebound_trigger', -7)

        was_oversold = (df[bias_col].shift(1) < oversold_threshold)
        is_rebounding = (df[bias_col] > rebound_trigger)
        final_signal = was_oversold & is_rebounding

        # 调试最新一周
        last = df.iloc[-1]
        prev = df.iloc[-2]
        wo_last = was_oversold.iloc[-1]
        ir_last = is_rebounding.iloc[-1]
        print(f"    - 条件1 (上周曾超卖): {'[✓]' if wo_last else '[✗]'} (上周BIAS: {prev.get(bias_col, 0):.2f} < 阈值: {oversold_threshold})")
        print(f"    - 条件2 (本周正反弹): {'[✓]' if ir_last else '[✗]'} (本周BIAS: {last.get(bias_col, 0):.2f} > 阈值: {rebound_trigger})")
        print(f"    - 结论: 最新一周信号为 [{'触发' if final_signal.iloc[-1] else '未触发'}]")

        return final_signal.fillna(False)

    def _playbook_calculate_washout_score(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """诊断剧本：量化周线级别的洗盘行为"""
        print("\n--- 诊断检查: [洗盘行为评分] ---")
        if not params.get('enabled', False):
            print("    - 结论: [未启用]")
            return pd.Series(0, index=df.index)
        
        washout_score = pd.Series(0, index=df.index)
        support_level = self._get_weekly_support_level(df, params)
        if support_level is None:
            print("    - 结论: [失败] 因无法确定支撑位而跳过。")
            return washout_score

        # 计算所有洗盘模式
        washout_intraday = (df['low_W'] < support_level) & (df['close_W'] > support_level)
        washout_interday = (df['close_W'] > support_level) & (df['close_W'].shift(1) < support_level.shift(1))
        was_below_recently = (df['close_W'].shift(1) < support_level.shift(1)).rolling(window=params.get('drift_lookback_period', 3), min_periods=1).sum() > 0
        washout_drift = (df['close_W'] > support_level) & was_below_recently
        recent_peak = df['high_W'].shift(1).rolling(window=params.get('bull_trap_lookback_period', 8)).max()
        is_in_trap_zone = df['close_W'] < recent_peak * (1 - params.get('bull_trap_drop_threshold', 0.05))
        is_recovering_from_trap = df['close_W'] > df['close_W'].shift(1)
        washout_bull_trap = is_in_trap_zone & is_recovering_from_trap
        avg_volume = df['volume_W'].shift(1).rolling(window=params.get('volume_avg_period', 20)).mean()
        is_volume_contracted = df['volume_W'] < avg_volume * params.get('volume_contraction_threshold', 0.7)
        washout_volume_contraction = (washout_interday | washout_drift) & is_volume_contracted.shift(1).fillna(False)
        
        washout_score += washout_intraday.astype(int)
        washout_score += washout_interday.astype(int)
        washout_score += washout_drift.astype(int)
        washout_score += washout_bull_trap.astype(int)
        washout_score += washout_volume_contraction.astype(int)

        # 调试最新一周
        last_support = support_level.iloc[-1]
        print(f"    - 使用的支撑位: {last_support:.2f}")
        print(f"    - 模式1 (日内洗盘): {'[+1分]' if washout_intraday.iloc[-1] else '[+0分]'}")
        print(f"    - 模式2 (日间洗盘): {'[+1分]' if washout_interday.iloc[-1] else '[+0分]'}")
        print(f"    - 模式3 (漂移收复): {'[+1分]' if washout_drift.iloc[-1] else '[+0分]'}")
        print(f"    - 模式4 (诱多陷阱): {'[+1分]' if washout_bull_trap.iloc[-1] else '[+0分]'}")
        print(f"    - 模式5 (缩量确认): {'[+1分]' if washout_volume_contraction.iloc[-1] else '[+0分]'}")
        print(f"    - 结论: 最新一周总得分为 [{washout_score.iloc[-1]}]")
        
        return washout_score.fillna(0)

    def _get_weekly_support_level(self, df: pd.DataFrame, params: dict) -> Optional[pd.Series]:
        """辅助函数: 获取周线级别的支撑位"""
        support_type = params.get('support_type', 'MA')
        support_level = pd.Series(np.nan, index=df.index)

        if support_type == 'MA':
            ma_period = params.get('support_ma_period', 21)
            ma_col = f'EMA_{ma_period}_W'
            if not self._check_dependencies(df, [ma_col], log_details=False): return None
            support_level = df[ma_col]
        elif support_type == 'BOX':
            boll_period, boll_std = params.get('boll_period', 20), params.get('boll_std', 2.0)
            bbw_col = f"BBW_{boll_period}_{float(boll_std)}_W"
            if not self._check_dependencies(df, [bbw_col, 'low_W'], log_details=False): return None
            
            quantile_level = params.get('quantile_level', 0.25)
            threshold = df[bbw_col].quantile(quantile_level)
            is_consolidating = df[bbw_col] < threshold
            if is_consolidating.any():
                box_period = params.get('box_period', 26)
                box_bottom = df['low_W'].rolling(window=box_period, min_periods=1).min()
                support_level = box_bottom.where(is_consolidating, np.nan)
        
        if support_level.isnull().all():
            return None
        
        return support_level.ffill()

    def _playbook_check_rejection_filters(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """诊断剧本：识别均线和箱体压力位的拒绝信号"""
        print("\n--- 诊断检查: [压力位拒绝信号] ---")
        if not params.get('enabled', True):
            print("    - 结论: [未启用]")
            return pd.Series(0, index=df.index)
        
        ma_period = params.get('ma_period', 21)
        ma_col = f'EMA_{ma_period}_W'
        ma_rejection = self._check_resistance_rejection(df, ma_col, params, "均线压力")

        box_lookback = params.get('box_lookback_period', 52)
        box_resistance_col = f'box_top_{box_lookback}W_resistance'
        df[box_resistance_col] = df['high_W'].shift(1).rolling(window=box_lookback, min_periods=int(box_lookback * 0.8)).max()
        box_rejection = self._check_resistance_rejection(df, box_resistance_col, params, "箱顶压力")

        final_signal = pd.Series(0, index=df.index)
        final_signal[ma_rejection] -= 1
        final_signal[box_rejection] -= 2
        
        print(f"    - 结论: 最新一周总得分为 [{final_signal.iloc[-1]}] (均线拒绝-1分, 箱顶拒绝-2分)")
        return final_signal

    def _check_resistance_rejection(self, df: pd.DataFrame, resistance_col: str, params: dict, source_name: str) -> pd.Series:
        """辅助函数: 检查在给定压力列上的拒绝信号"""
        print(f"  - 检查子项: [{source_name}]")
        volume_multiplier = params.get('volume_multiplier', 1.5)
        vol_ma_period = self.indicator_cfg.get('vol_ma', {}).get('periods', [21, 55])[-1]
        vol_ma_col = f'VOL_MA_{vol_ma_period}_W'
        
        required_cols = [resistance_col, vol_ma_col, 'open_W', 'high_W', 'close_W', 'volume_W']
        if not self._check_dependencies(df, required_cols, log_details=True):
            print(f"    - 结论: [失败] 缺少必要列")
            return pd.Series(False, index=df.index)

        is_near_resistance = df['high_W'] >= df[resistance_col]
        is_long_upper_shadow = (df['high_W'] - df[['open_W', 'close_W']].max(axis=1)) > (df['high_W'] - df['low_W']) * 0.5
        is_high_volume = df['volume_W'] > df[vol_ma_col] * volume_multiplier
        is_closing_lower = df['close_W'] < df[['open_W', 'close_W']].mean(axis=1)
        final_signal = (is_near_resistance & is_long_upper_shadow & is_high_volume & is_closing_lower)

        # 调试最新一周
        last = df.iloc[-1]
        c1 = is_near_resistance.iloc[-1]
        c2 = is_long_upper_shadow.iloc[-1]
        c3 = is_high_volume.iloc[-1]
        c4 = is_closing_lower.iloc[-1]
        print(f"    - 条件1 (触及压力): {'[✓]' if c1 else '[✗]'} (最高价: {last.get('high_W', 0):.2f} vs 压力: {last.get(resistance_col, 0):.2f})")
        print(f"    - 条件2 (长上影线): {'[✓]' if c2 else '[✗]'}")
        print(f"    - 条件3 (放出大量): {'[✓]' if c3 else '[✗]'} (成交量: {last.get('volume_W', 0):.0f} vs 阈值: {(last.get(vol_ma_col, 0) * volume_multiplier):.0f})")
        print(f"    - 条件4 (收盘偏低): {'[✓]' if c4 else '[✗]'}")
        print(f"    - 小结: [{source_name}] {'触发' if final_signal.iloc[-1] else '未触发'}")
        
        return final_signal.fillna(False)

    def _playbook_box_consolidation_breakout(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """剧本：专业箱体突破"""
        print("\n--- 剧本检查: [专业箱体突破] (V2.8 终极调试版) ---")
        if not params.get('enabled', True):
            print("    - 结论: [未启用]")
            return pd.Series(False, index=df.index)

        # --- 步骤 1: 参数获取 ---
        quantile_level = params.get('quantile_level', 0.25)
        boll_period = params.get('boll_period', 20)
        boll_std = params.get('boll_std', 2.0)
        box_period = params.get('box_period', 26)
        volume_multiplier = params.get('volume_multiplier', 1.5)
        
        # --- 步骤 2: 依赖检查 ---
        bbw_col = f"BBW_{boll_period}_{float(boll_std)}_W"
        vol_ma_cfg = self.indicator_cfg.get('vol_ma', {})
        vol_ma_period = next((p for p in vol_ma_cfg.get('periods', [21, 55]) if p >= box_period), vol_ma_cfg.get('periods', [21])[-1])
        vol_ma_col = f"VOL_MA_{vol_ma_period}_W"
        required_cols = ['close_W', 'high_W', 'volume_W', bbw_col, vol_ma_col]
        
        if not self._check_dependencies(df, required_cols, log_details=True):
            print("    - 结论: [失败] 依赖检查失败，策略提前退出。")
            return pd.Series(False, index=df.index)

        # --- 步骤 3: 核心计算 ---
        dynamic_bbw_threshold = df[bbw_col].expanding(min_periods=box_period).quantile(quantile_level)
        is_low_volatility_week = df[bbw_col] < dynamic_bbw_threshold
        consolidation_blocks = (is_low_volatility_week != is_low_volatility_week.shift()).cumsum()
        high_in_consolidation = df['high_W'].where(is_low_volatility_week)
        box_high = high_in_consolidation.groupby(consolidation_blocks).transform('max')
        volume_in_consolidation = df['volume_W'].where(is_low_volatility_week)
        box_avg_volume = volume_in_consolidation.groupby(consolidation_blocks).transform('mean')
        
        is_price_breakout = df['close_W'] > box_high.shift(1)
        is_volume_breakout = df['volume_W'] > (box_avg_volume.shift(1) * volume_multiplier)
        was_in_consolidation = is_low_volatility_week.shift(1).fillna(False)
        final_signal = (was_in_consolidation & is_price_breakout & is_volume_breakout)

        # --- 步骤 4: 调试最新一周 ---
        last_idx = -1
        
        # 获取上一周的值
        prev_bbw = df[bbw_col].iloc[last_idx - 1]
        prev_bbw_thresh = dynamic_bbw_threshold.iloc[last_idx - 1]
        prev_box_high = box_high.shift(1).iloc[last_idx]
        prev_box_avg_vol = box_avg_volume.shift(1).iloc[last_idx]

        # 获取当前周的值
        curr_close = df['close_W'].iloc[last_idx]
        curr_vol = df['volume_W'].iloc[last_idx]
        
        # 判断条件
        c1 = was_in_consolidation.iloc[last_idx]
        c2 = is_price_breakout.iloc[last_idx]
        c3 = is_volume_breakout.iloc[last_idx]
        
        print(f"    - 条件1 (前一周处于盘整期): {'[✓]' if c1 else '[✗]'} (前周BBW: {prev_bbw:.4f} vs 动态阈值: {prev_bbw_thresh:.4f})")
        print(f"    - 条件2 (价格突破箱顶): {'[✓]' if c2 else '[✗]'} (本周收盘: {curr_close:.2f} vs 前周箱顶: {prev_box_high:.2f})")
        print(f"    - 条件3 (成交量突破): {'[✓]' if c3 else '[✗]'} (本周成交量: {curr_vol:.0f} vs 阈值: {(prev_box_avg_vol * volume_multiplier):.0f})")
        print(f"    - 结论: 最新一周信号为 [{'触发' if final_signal.iloc[last_idx] else '未触发'}]")
            
        return final_signal.fillna(False)

    def _playbook_trix_golden_cross(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """剧本：识别周线TRIX金叉，一个强大的中长期趋势确认信号。"""
        print("\n--- 剧本检查: [TRIX金叉] ---")
        if not params.get('enabled', True):
            print("    - 结论: [未启用]")
            return pd.Series(False, index=df.index)

        # 从指标配置中动态获取TRIX参数
        trix_cfg = self.indicator_cfg.get('trix', {})
        # 智能解析新版或旧版配置
        trix_periods = None
        if 'configs' in trix_cfg:
            for config_item in trix_cfg['configs']:
                if 'W' in config_item.get('apply_on', []):
                    trix_periods = config_item.get('periods')
                    break
        else: # 兼容旧版
            trix_periods = trix_cfg.get('periods')

        if not trix_periods or len(trix_periods) < 2:
            print("    - 结论: [失败] TRIX周期参数配置不正确。")
            return pd.Series(False, index=df.index)

        trix_len, signal_len = trix_periods[0], trix_periods[1]
        trix_col = f'TRIX_{trix_len}_{signal_len}_W'
        trix_signal_col = f'TRIXs_{trix_len}_{signal_len}_W'

        if not self._check_dependencies(df, [trix_col, trix_signal_col], log_details=True):
            print(f"    - 结论: [失败] 缺少必要的TRIX列。")
            return pd.Series(False, index=df.index)

        # 核心逻辑：TRIX线上穿其信号线
        is_golden_cross = (df[trix_col] > df[trix_signal_col]) & \
                          (df[trix_col].shift(1) <= df[trix_signal_col].shift(1))
        
        # 调试最新一周
        last = df.iloc[-1]
        prev = df.iloc[-2]
        gc_last = is_golden_cross.iloc[-1]
        print(f"    - 条件1 (本周TRIX > 信号线): {'[✓]' if last.get(trix_col, 0) > last.get(trix_signal_col, 0) else '[✗]'} (TRIX: {last.get(trix_col, 0):.2f} vs 信号线: {last.get(trix_signal_col, 0):.2f})")
        print(f"    - 条件2 (上周TRIX <= 信号线): {'[✓]' if prev.get(trix_col, 0) <= prev.get(trix_signal_col, 0) else '[✗]'} (TRIX: {prev.get(trix_col, 0):.2f} vs 信号线: {prev.get(trix_signal_col, 0):.2f})")
        print(f"    - 结论: 最新一周信号为 [{'触发' if gc_last else '未触发'}]")

        return is_golden_cross.fillna(False)

    def _playbook_coppock_bottom_reversal(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """剧本：识别周线Coppock估波曲线从底部反转，一个战略性的长期底部信号。"""
        print("\n--- 剧本检查: [Coppock底部反转] ---")
        if not params.get('enabled', True):
            print("    - 结论: [未启用]")
            return pd.Series(False, index=df.index)

        # 从指标配置中动态获取Coppock参数
        coppock_cfg = self.indicator_cfg.get('coppock', {})
        coppock_periods = None
        if 'configs' in coppock_cfg:
            for config_item in coppock_cfg['configs']:
                if 'W' in config_item.get('apply_on', []):
                    coppock_periods = config_item.get('periods')
                    break
        else: # 兼容旧版
            coppock_periods = coppock_cfg.get('periods')

        if not coppock_periods or len(coppock_periods) < 3:
            print("    - 结论: [失败] Coppock周期参数配置不正确。")
            return pd.Series(False, index=df.index)

        p1, p2, p3 = coppock_periods[0], coppock_periods[1], coppock_periods[2]
        coppock_col = f'COPP_{p1}_{p2}_{p3}_W'

        if not self._check_dependencies(df, [coppock_col], log_details=True):
            print(f"    - 结论: [失败] 缺少必要的Coppock列。")
            return pd.Series(False, index=df.index)

        # 核心逻辑：Coppock曲线在0轴下方，开始拐头向上
        is_turning_up = df[coppock_col] > df[coppock_col].shift(1)
        was_below_zero = df[coppock_col].shift(1) < 0
        final_signal = is_turning_up & was_below_zero

        # 调试最新一周
        last = df.iloc[-1]
        prev = df.iloc[-2]
        fs_last = final_signal.iloc[-1]
        print(f"    - 条件1 (本周Coppock > 上周Coppock): {'[✓]' if last.get(coppock_col, 0) > prev.get(coppock_col, 0) else '[✗]'} (本周: {last.get(coppock_col, 0):.2f} vs 上周: {prev.get(coppock_col, 0):.2f})")
        print(f"    - 条件2 (上周Coppock < 0): {'[✓]' if prev.get(coppock_col, 0) < 0 else '[✗]'} (上周值: {prev.get(coppock_col, 0):.2f})")
        print(f"    - 结论: 最新一周信号为 [{'触发' if fs_last else '未触发'}]")

        return final_signal.fillna(False)

    def _check_dependencies(self, df: pd.DataFrame, cols: list, log_details: bool = False) -> bool:
        """检查DataFrame中是否存在所有必需的列。"""
        missing_cols = [col for col in cols if col not in df.columns]
        if missing_cols:
            if log_details:
                print(f"      - [依赖检查] 失败! 缺少以下必需列: {missing_cols}")
            if not hasattr(self, '_warned_missing_cols_weekly'):
                logger.warning(f"周线策略缺少必需列: {missing_cols}，相关剧本将跳过。")
                self._warned_missing_cols_weekly = True
            return False
        return True

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
