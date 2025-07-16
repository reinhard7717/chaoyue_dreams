# 文件: strategies/weekly_context_engine.py (新文件名，或替换旧文件)
# 版本: V1.0 - 战略上下文引擎

from typing import Dict, Optional
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class WeeklyContextEngine:
    """
    【V1.0】周线战略上下文引擎
    - 职能定位: 不再是独立的策略，而是作为战略参谋部，为日线策略提供
                更高维度的周线级别战略背景信号 (Context)。
    - 输入: 一个包含所有周线指标的 DataFrame。
    - 输出: 一个只包含 'state_..._W', 'playbook_..._W' 等战略信号的 DataFrame。
    """

    def __init__(self, config: dict):
        """
        【V1.1 适配统一指挥版】
        - 核心升级: 重构了配置读取逻辑，使其能够正确解析新的“统一配置文件”。
        - 新逻辑:
          - 从统一配置的顶层获取 'feature_engineering_params'。
          - 从统一配置的 'weekly_context_params' 块中获取自己的行动指令。
        """       
        # 1. 从统一配置中，获取周线引擎专属的逻辑参数块
        self.params = config.get('weekly_context_params', {})
        # 2. 从统一配置的顶层，获取所有指标的定义
        self.indicator_cfg = config.get('feature_engineering_params', {}).get('indicators', {})
        # 3. 从周线专属的逻辑块中，获取剧本定义
        self.playbook_params = self.params.get('strategy_playbooks', {})

    def generate_context(self, df_weekly: pd.DataFrame) -> pd.DataFrame:
        """
        【核心引擎函数】
        接收周线数据，计算所有战略信号，并返回一个纯净的信号DataFrame。
        """
        if df_weekly is None or df_weekly.empty:
            logger.warning("周线上下文引擎输入DataFrame为空，无法生成信号。")
            return pd.DataFrame()

        # --- 步骤 1: 计算所有基础剧本和诊断信号 ---
        # _calculate_all_playbooks 的逻辑与您原代码完全一致
        context_df = self._calculate_all_playbooks(df_weekly)
        
        # --- 步骤 2: 趋势状态分析 (逻辑与您原代码完全一致) ---
        trend_params = self.params.get('trend_analysis_params', {})
        ma_period = trend_params.get('ma_period', 21)
        ma_col = f'EMA_{ma_period}_W'
        
        if ma_col in context_df.columns:
            slope = context_df[ma_col].diff(1)
            acceleration = slope.diff(1)
            context_df['state_trend_accelerating_W'] = (slope > 0) & (acceleration > 0)
            context_df['state_trend_stable_rising_W'] = (slope > 0) & (acceleration <= 0)
            context_df['state_trend_decelerating_fall_W'] = (slope < 0) & (acceleration > 0)
            context_df['state_trend_accelerating_fall_W'] = (slope < 0) & (acceleration < 0)
            context_df['filter_trend_is_healthy_W'] = context_df['state_trend_accelerating_W'] | context_df['state_trend_stable_rising_W']
        else:
            print(f"【趋势分析-警告】缺少核心趋势分析列 {ma_col}，将跳过斜率分析。")
            for col in ['filter_trend_is_healthy_W', 'state_trend_accelerating_W', 'state_trend_stable_rising_W', 'state_trend_decelerating_fall_W']:
                context_df[col] = pd.Series(False, index=context_df.index)
            context_df['filter_trend_is_healthy_W'] = True
        
        # --- 步骤 3: 信号合成层 (引入洗盘豁免) ---
        # print("\n---【合成-右侧信号】---")
        
        # === 3.1 新增：建立“洗盘豁免”状态 ===
        washout_params = self.playbook_params.get('washout_score_playbook', {})
        immunity_threshold = washout_params.get('immunity_score_threshold', 3)
        immunity_window = washout_params.get('immunity_window', 3)
        if 'washout_score_W' in context_df.columns:
            # 检查过去N周内，是否曾出现过一次得分超过豁免阈值的洗盘
            had_recent_strong_washout = (context_df['washout_score_W'].rolling(window=immunity_window).max().shift(1) >= immunity_threshold)
            context_df['state_washout_immunity_W'] = had_recent_strong_washout.fillna(False)
            # print(f"【洗盘豁免】已启用。豁免分数阈值: {immunity_threshold}, 豁免窗口: {immunity_window}周。")
            # print(f"【洗盘豁免】最近一周豁免状态: {'[激活]' if context_df['state_washout_immunity_W'].iloc[-1] else '[未激活]'}")
        else:
            context_df['state_washout_immunity_W'] = pd.Series(False, index=context_df.index)
            print("【洗盘豁免】未启用 (缺少 washout_score_W 列)。")

        # === 3.2 右侧趋势跟踪信号合成 (修改) ===
        accumulation_playbooks = ['playbook_early_uptrend_W', 'playbook_ma20_turn_up_event_W']
        valid_accumulation_playbooks = [p for p in accumulation_playbooks if p in context_df.columns]
        is_accumulation_signal = context_df[valid_accumulation_playbooks].any(axis=1) if valid_accumulation_playbooks else pd.Series(False, index=context_df.index)
        recent_accumulation_window = 12 
        had_recent_accumulation = is_accumulation_signal.rolling(window=recent_accumulation_window, min_periods=1).sum().shift(1) > 0
        context_df['state_had_recent_accumulation_W'] = had_recent_accumulation.fillna(False)
        
        breakout_playbooks = ['playbook_classic_breakout_W', 'playbook_box_breakout_W']
        valid_breakout_playbooks = [p for p in breakout_playbooks if p in context_df.columns]
        is_breakout_event = context_df[valid_breakout_playbooks].any(axis=1) if valid_breakout_playbooks else pd.Series(False, index=context_df.index)
        context_df['event_is_breakout_week_W'] = is_breakout_event
        
        is_breakout_initiation = context_df['state_had_recent_accumulation_W'] & context_df['event_is_breakout_week_W']
        
        # 修改：判断是否出现拒绝信号
        is_rejection_week = context_df.get('rejection_signal_W', pd.Series(0, index=context_df.index)) < 0
        # 核心修改：如果处于洗盘豁免期，则无视拒绝信号
        is_rejection_week_final = is_rejection_week & ~context_df['state_washout_immunity_W']

        breakout_event_group = is_breakout_initiation.cumsum()
        rejections_in_group_so_far = is_rejection_week_final.groupby(breakout_event_group).cumsum()
        has_no_rejection_yet = (rejections_in_group_so_far == 0)
        
        base_trigger = is_breakout_initiation & has_no_rejection_yet & context_df['filter_trend_is_healthy_W']
        ace_trigger = context_df.get('playbook_ace_signal_breakout_trigger_W', pd.Series(False, index=context_df.index))
        context_df['signal_breakout_trigger_W'] = base_trigger | ace_trigger
        
        print(f"【右侧】原始拒绝信号周数: {is_rejection_week.sum()}")
        print(f"【右侧】豁免后有效拒绝周数: {is_rejection_week_final.sum()}")
        print(f"【右侧】最终突破观察信号(signal_breakout_trigger_W)周数: {context_df['signal_breakout_trigger_W'].sum()}")

        # === 3.3 左侧拐点观察信号合成 (逻辑不变) ===
        left_side_params = trend_params.get('left_side_support', {})
        if left_side_params.get('enabled', False):
            bottoming_playbooks = ['playbook_bias_rebound_W', 'playbook_trix_golden_cross_W', 'playbook_coppock_reversal_W']
            valid_bottoming_playbooks = [p for p in bottoming_playbooks if p in context_df.columns]
            is_bottoming_signal = context_df[valid_bottoming_playbooks].any(axis=1) if valid_bottoming_playbooks else pd.Series(False, index=context_df.index)
            washout_score_min = left_side_params.get('washout_score_min', 2)
            if 'washout_score_W' in context_df.columns:
                is_bottoming_signal |= (context_df['washout_score_W'] >= washout_score_min)
            context_df['signal_potential_bottom_zone_W'] = context_df['state_trend_decelerating_fall_W'] & is_bottoming_signal
        else:
            context_df['signal_potential_bottom_zone_W'] = pd.Series(False, index=context_df.index)

        # === 3.4 右侧信号增强 (新增) ===
        # 将Coppock加速信号和TRIX金叉作为“点火期”的增强信号
        ignition_enhancers = [
            'playbook_trix_golden_cross_W',
            'playbook_coppock_accelerating_W' # 新增：使用新的右侧加速信号
        ]
        valid_enhancers = [p for p in ignition_enhancers if p in context_df.columns]
        if valid_enhancers:
            is_ignition_enhancer_event = context_df[valid_enhancers].any(axis=1)
            # 如果有增强信号，也可以触发突破观察（即使没有经典突破形态）
            context_df['signal_breakout_trigger_W'] |= (is_ignition_enhancer_event & context_df['filter_trend_is_healthy_W'])
            # print(f"【右侧】点火增强信号源: {valid_enhancers}")
            # print(f"【右侧】增强后最终突破观察信号周数: {context_df['signal_breakout_trigger_W'].sum()}")

        # --- 步骤 4: 状态节点判读 (修改) ---
        # print("\n---【状态节点判读】---")
        context_df['state_node_bottoming_W'] = context_df['signal_potential_bottom_zone_W']
        
        # 点火期信号源：均线加速拐头、TRIX金叉、Coppock加速
        ignition_signals = [
            'playbook_ma20_turn_up_event_W', 
            'playbook_trix_golden_cross_W',
            'playbook_coppock_accelerating_W' # 修改：使用新的右侧加速信号
        ]
        valid_ignition_signals = [s for s in ignition_signals if s in context_df.columns]
        context_df['state_node_ignition_W'] = context_df[valid_ignition_signals].any(axis=1) if valid_ignition_signals else pd.Series(False, index=context_df.index)
        
        context_df['state_node_main_ascent_W'] = context_df['state_trend_accelerating_W']
        context_df['state_node_topping_W'] = context_df['state_trend_stable_rising_W'] & (context_df.get('rejection_signal_W', 0) < 0)
        # print(f"【状态节点】筑底区: {context_df['state_node_bottoming_W'].sum()}周 | 点火期: {context_df['state_node_ignition_W'].sum()}周 | 主升段: {context_df['state_node_main_ascent_W'].sum()}周 | 滞涨区: {context_df['state_node_topping_W'].sum()}周")

        # --- 步骤 5: 【核心修改】筛选并返回纯净的战略信号 ---
        signal_cols = [col for col in context_df.columns if col.startswith(('state_', 'playbook_', 'signal_', 'filter_', 'event_', 'rejection_', 'washout_'))]
        
        # 确保原始DataFrame的列不会被意外包含
        original_cols = df_weekly.columns
        final_signal_cols = [col for col in signal_cols if col not in original_cols]

        print(f"    - [周线引擎] 已生成 {len(final_signal_cols)} 个周线战略信号。")
        return context_df[final_signal_cols]
    
    def _calculate_all_playbooks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V3.5 命名规范化版】动态遍历JSON配置，并确保输出的剧本名称为大写，与日线策略对齐。
        """
        # print("\n" + "="*80)
        # print(f"---【周线战略层(V3.5 命名规范化) - 检查最新一周: {df.index[-1].date()}】---")
        # print("="*80)
        
        context_df = df.copy()
        
        playbook_map = {
            'ma20_rising_state_playbook': self._playbook_ma20_is_rising,
            'ma20_turn_up_event_playbook': self._playbook_ma20_turn_up_event,
            'early_uptrend_playbook': self._playbook_early_uptrend,
            'classic_breakout_playbook': self._playbook_classic_breakout,
            'ma_uptrend_playbook': self._playbook_check_ma_uptrend,
            'box_consolidation_breakout_playbook': self._playbook_box_consolidation_breakout,
            'oversold_rebound_bias_playbook': self._playbook_oversold_rebound_bias,
            'washout_score_playbook': self._playbook_calculate_washout_score,
            'rejection_filter_playbook': self._playbook_check_rejection_filters,
            'trix_golden_cross_playbook': self._playbook_trix_golden_cross,
            'coppock_reversal_playbook': self._playbook_coppock_reversal,
            'ace_signal_breakout_trigger_playbook': self._playbook_ace_signal_breakout_trigger,
        }

        for playbook_name, params in self.playbook_params.items():
            if playbook_name == "说明": continue

            if playbook_name in playbook_map:
                if params.get('enabled', False):
                    results = playbook_map[playbook_name](df, params)
                    
                    if isinstance(results, dict):
                        for signal_suffix, result_series in results.items():
                            # ▼▼▼【代码修改】: 规范化多信号输出的剧本名称 ▼▼▼
                            # 将 'coppock_stabilizing' 转换为 'PLAYBOOK_COPPOCK_STABILIZING_W'
                            col_name = f"playbook_{signal_suffix.upper()}_W"
                            context_df[col_name] = result_series
                            # print(f"    - [多信号输出模式] 已生成规范化列: '{col_name}'")
                    elif isinstance(results, pd.Series):
                        if 'score' in playbook_name:
                            col_name = 'washout_score_W'
                        elif 'filter' in playbook_name:
                            col_name = 'rejection_signal_W'
                        else:
                            # 将 'ma20_turn_up_event_playbook' 转换为 'PLAYBOOK_MA20_TURN_UP_EVENT_W'
                            base_name = playbook_name.replace('_playbook', '').upper()
                            col_name = f"playbook_{base_name}_W"
                        
                        # print(f"    - [单信号输出模式] 正在为规范化列 '{col_name}' 赋值...")
                        context_df[col_name] = results
                else:
                    # print(f"\n--- 剧本检查: [{params.get('说明', playbook_name)}] ---")
                    print("    - 结论: [未启用]")
            else:
                logger.warning(f"JSON中配置的剧本 '{playbook_name}' 在代码中没有找到对应的实现函数，已跳过。")

        # print("\n---【周线战略层(V3.5) - 剧本计算总结】---")
        # for col in context_df.columns:
        #     if col.startswith('playbook_'):
        #         # label = col.replace('playbook_', '').replace('_W', '').replace('_', ' ').title()
        #         print(f"【剧本-{col}】触发周数: {context_df[col].sum()}")
        # if 'washout_score_W' in context_df.columns:
        #     score = context_df['washout_score_W']
        #     print(f"【诊断-洗盘】有分数的周数: {(score > 0).sum()} (最高分: {score.max()})")
        # if 'rejection_signal_W' in context_df.columns:
        #     rejection = context_df['rejection_signal_W']
        #     print(f"【诊断-风险】有拒绝信号的周数: {(rejection < 0).sum()}")
        
        return context_df

    def _playbook_ma20_is_rising(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【V3.1 升级剧本】: 识别指定周线均线是否处于“有效”上升状态。
        - 核心修改: 引入斜率阈值，过滤掉几乎走平的“伪上涨”状态。
        """
        # print(f"\n--- 剧本检查: [{params.get('说明', '均线处于上升状态')}] ---")
        target_ma_period = params.get('ma_period', 21)
        # 新增斜率阈值参数，要求均线每周至少上涨0.1%才算有效上涨
        slope_threshold_pct = params.get('slope_threshold_pct', 0.1) 
        # print(f"    - 配置参数: ma_period={target_ma_period}, slope_threshold_pct={slope_threshold_pct}%")
        
        ema_col = f'EMA_{target_ma_period}_W'
        close_col = 'close_W'
        if not self._check_dependencies(df, [ema_col, close_col], log_details=True):
            print(f"    - 结论: [失败] 缺少必要列")
            return pd.Series(False, index=df.index)

        # 计算斜率，并进行标准化，使其不受股价绝对值影响
        slope = df[ema_col].diff(1)
        safe_ma = df[ema_col].shift(1).replace(0, 1e-9) # 防止除以零
        normalized_slope_pct = (slope / safe_ma) * 100

        # 条件1: 标准化后的斜率必须大于阈值
        condition1_is_rising_effectively = normalized_slope_pct > slope_threshold_pct
        
        # 条件2: 收盘价在均线之上
        condition2_price_confirm = df[close_col] > df[ema_col]
        
        final_signal = condition1_is_rising_effectively & condition2_price_confirm
        
        # 调试信息
        # last = df.iloc[-1]
        # c1_last = condition1_is_rising_effectively.iloc[-1]
        # c2_last = condition2_price_confirm.iloc[-1]
        
        # print(f"    - 条件1 (均线有效上涨): {'[✓]' if c1_last else '[✗]'} (周涨幅: {normalized_slope_pct.iloc[-1]:.3f}% > 阈值: {slope_threshold_pct}%)")
        # print(f"    - 条件2 (价格确认): {'[✓]' if c2_last else '[✗]'} (收盘价: {last.get(close_col, 0):.2f} vs 均线: {last.get(ema_col, 0):.2f})")
        # print(f"    - 结论: 最新一周信号为 [{'触发' if final_signal.iloc[-1] else '未触发'}]")
        
        return final_signal.fillna(False)

    def _playbook_ma20_turn_up_event(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【V3.1 升级剧本】: 识别指定周线均线“加速拐头向上”的事件。
        - 核心修改: 引入二阶斜率（加速度）作为判断条件。
                     要求拐头不仅是方向改变，还必须是“加速”的，以提高信号质量。
        """
        # print(f"\n--- 剧本检查: [{params.get('说明', '均线拐头向上事件')}] ---")
        target_ma_period = params.get('ma_period', 21)
        # 新增加速度阈值参数，如果JSON中没有，则默认为0
        accel_threshold = params.get('accel_threshold', 0) 
        # print(f"    - 配置参数: ma_period={target_ma_period}, accel_threshold={accel_threshold}")
        
        ema_col = f'EMA_{target_ma_period}_W'
        close_col = 'close_W'
        if not self._check_dependencies(df, [ema_col, close_col], log_details=True):
            print(f"    - 结论: [失败] 缺少必要列")
            return pd.Series(False, index=df.index)

        # 计算斜率（速度）和二阶斜率（加速度）
        slope = df[ema_col].diff(1)
        acceleration = slope.diff(1)

        # 条件1: 发生拐头 (本周斜率为正，上周为负或零)
        slope_is_positive = slope > 0
        slope_was_not_positive = slope.shift(1) <= 0
        condition1_turn_up = slope_is_positive & slope_was_not_positive
        
        # 条件2 (新增): 加速度必须为正，且大于阈值，确保是“有力”的拐头
        condition2_is_accelerating = acceleration > accel_threshold

        # 条件3: 收盘价在均线之上作为确认
        condition3_price_confirm = df[close_col] > df[ema_col]

        final_signal = condition1_turn_up & condition2_is_accelerating & condition3_price_confirm
        
        # 调试信息
        last = df.iloc[-1]
        c1_last = condition1_turn_up.iloc[-1]
        c2_last = condition2_is_accelerating.iloc[-1]
        c3_last = condition3_price_confirm.iloc[-1]
        
        # print(f"    - 条件1 (均线发生拐头): {'[✓]' if c1_last else '[✗]'} (本周斜率: {slope.iloc[-1]:.2f} > 0 AND 上周斜率: {slope.shift(1).iloc[-1]:.2f} <= 0)")
        # print(f"    - 条件2 (拐头正在加速): {'[✓]' if c2_last else '[✗]'} (加速度: {acceleration.iloc[-1]:.2f} > 阈值: {accel_threshold})")
        # print(f"    - 条件3 (价格确认): {'[✓]' if c3_last else '[✗]'} (收盘价: {last.get(close_col, 0):.2f} vs 均线: {last.get(ema_col, 0):.2f})")
        # print(f"    - 结论: 最新一周信号为 [{'触发' if final_signal.iloc[-1] else '未触发'}]")
        
        return final_signal.fillna(False)

    def _playbook_early_uptrend(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """剧本：捕捉周线趋势反转的早期“上拐”信号"""
        # print(f"\n--- 剧本检查: [{params.get('说明', '早期上升趋势')}] ---")
        
        # 从传递的params中读取周期，如果未定义则使用默认值10和20
        short_ma_period = params.get('short_ma', 10)
        mid_ma_period = params.get('mid_ma', 20)
        # print(f"    - 配置参数: short_ma={short_ma_period}, mid_ma={mid_ma_period}")

        # 使用读取到的周期构建列名
        short_ma_col = f'EMA_{short_ma_period}_W'
        mid_ma_col = f'EMA_{mid_ma_period}_W'

        macd_params_raw = self.indicator_cfg.get('macd', {}).get('periods', [12, 26, 9])
        p_fast, p_slow, p_signal = macd_params_raw
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
        s_last = signal.iloc[-1]
        ieu_last = in_early_uptrend.iloc[-1]
        print(f"    - 子信号1 (拐点信号): {'[✓]' if s_last else '[✗]'}")
        print(f"    - 子信号2 (趋势延续): {'[✓]' if ieu_last else '[✗]'}")
        print(f"    - 结论: 最新一周信号为 [{'触发' if final_signal.iloc[-1] else '未触发'}] (逻辑: 拐点 OR 延续)")
        return final_signal.fillna(False)

    def _playbook_classic_breakout(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """剧本：经典高点突破"""
        # print(f"\n--- 剧本检查: [{params.get('说明', '经典高点突破')}] ---")
        lookback_weeks, volume_multiplier = params.get('lookback_weeks', 26), params.get('volume_multiplier', 1.5)
        if not self._check_dependencies(df, ['high_W', 'volume_W', 'close_W'], log_details=True):
            print(f"    - 结论: [失败] 缺少必要列")
            return pd.Series(False, index=df.index)
        period_high = df['high_W'].shift(1).rolling(window=lookback_weeks).max()
        avg_volume = df['volume_W'].shift(1).rolling(window=lookback_weeks).mean()
        is_price_breakout = df['close_W'] > period_high
        is_volume_breakout = df['volume_W'] > (avg_volume * volume_multiplier)
        final_signal = is_price_breakout & is_volume_breakout
        last = df.iloc[-1]
        ph_last = period_high.iloc[-1]
        av_last = avg_volume.iloc[-1]
        pb_last = is_price_breakout.iloc[-1]
        vb_last = is_volume_breakout.iloc[-1]
        # print(f"    - 条件1 (价格突破): {'[✓]' if pb_last else '[✗]'} (收盘价: {last.get('close_W', float('nan')):.2f} vs 前{lookback_weeks}周高点: {ph_last:.2f})")
        # print(f"    - 条件2 (放量突破): {'[✓]' if vb_last else '[✗]'} (成交量: {last.get('volume_W', 0):.0f} vs 阈值: {(av_last * volume_multiplier):.0f})")
        # print(f"    - 结论: 最新一周信号为 [{'触发' if final_signal.iloc[-1] else '未触发'}]")
        return final_signal.fillna(False)

    def _playbook_check_ma_uptrend(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【V3.3 升级剧本】: 检查均线多头排列 (引入容忍区)。
        - 核心修改: 增加 tolerance_pct 参数，允许价格在支撑均线下方有小幅波动，
                     以容忍市场噪音和毛刺，避免因轻微跌破而被错误过滤。
        """
        # print(f"\n--- 剧本检查: [{params.get('说明', '均线多头排列')}] ---")
        short_ma = params.get('short_ma', 13)
        mid_ma = params.get('mid_ma', 21)
        long_ma = params.get('long_ma', 55)
        # 新增：从JSON获取容忍度参数，默认为1%，即允许股价跌破支撑均线1%
        tolerance_pct = params.get('tolerance_pct', 0.01)
        # print(f"    - 配置参数: short={short_ma}, mid={mid_ma}, long={long_ma}, tolerance_pct={tolerance_pct*100}%")

        short_col, mid_col, long_col = f'EMA_{short_ma}_W', f'EMA_{mid_ma}_W', f'EMA_{long_ma}_W'
        if not self._check_dependencies(df, [short_col, mid_col, long_col, 'close_W'], log_details=True):
            print(f"    - 结论: [失败] 缺少必要列")
            return pd.Series(False, index=df.index)

        # 条件1: 均线排列关系不变
        ma_aligned = (df[short_col] > df[mid_col]) & (df[mid_col] > df[long_col])
        
        # 条件2 (修改): 股价在支撑均线的“容忍区”之上
        support_level_with_tolerance = df[mid_col] * (1 - tolerance_pct)
        price_above_support_zone = df['close_W'] > support_level_with_tolerance

        final_signal = ma_aligned & price_above_support_zone
        
        last = df.iloc[-1]
        ma_last = ma_aligned.iloc[-1]
        pas_last = price_above_support_zone.iloc[-1]
        
        # print(f"    - 条件1 (均线多头): {'[✓]' if ma_last else '[✗]'} (EMA{short_ma}: {last.get(short_col, 0):.2f} > EMA{mid_ma}: {last.get(mid_col, 0):.2f} > EMA{long_ma}: {last.get(long_col, 0):.2f})")
        # print(f"    - 条件2 (股价在支撑容忍区上): {'[✓]' if pas_last else '[✗]'} (收盘价: {last.get('close_W', 0):.2f} > 支撑区下轨: {support_level_with_tolerance.iloc[-1]:.2f})")
        # print(f"    - 结论: 最新一周信号为 [{'触发' if final_signal.iloc[-1] else '未触发'}]")
        
        return final_signal.fillna(False)
    
    def _playbook_oversold_rebound_bias(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """剧本：利用BIAS指标捕捉周线级别的超跌反弹机会"""
        # print(f"\n--- 剧本检查: [{params.get('说明', 'BIAS超跌反弹')}] ---")
        bias_period = params.get('bias_period', 20)
        bias_col = f'BIAS_{bias_period}_W'
        if not self._check_dependencies(df, [bias_col], log_details=True):
            print(f"    - 结论: [失败] 缺少必要列 {bias_col}")
            return pd.Series(False, index=df.index)
        oversold_threshold = params.get('oversold_threshold', -15)
        rebound_trigger = params.get('rebound_trigger', -12)
        was_oversold = (df[bias_col].shift(1) < oversold_threshold)
        is_rebounding = (df[bias_col] > rebound_trigger)
        final_signal = was_oversold & is_rebounding
        last = df.iloc[-1]
        prev = df.iloc[-2]
        wo_last = was_oversold.iloc[-1]
        ir_last = is_rebounding.iloc[-1]
        # print(f"    - 条件1 (上周曾超卖): {'[✓]' if wo_last else '[✗]'} (上周BIAS: {prev.get(bias_col, 0):.2f} < 阈值: {oversold_threshold})")
        # print(f"    - 条件2 (本周正反弹): {'[✓]' if ir_last else '[✗]'} (本周BIAS: {last.get(bias_col, 0):.2f} > 阈值: {rebound_trigger})")
        # print(f"    - 结论: 最新一周信号为 [{'触发' if final_signal.iloc[-1] else '未触发'}]")
        return final_signal.fillna(False)

    def _playbook_calculate_washout_score(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """诊断剧本：量化周线级别的洗盘行为"""
        print(f"\n--- 诊断检查: [{params.get('说明', '洗盘行为评分')}] ---")
        washout_score = pd.Series(0, index=df.index)
        support_level = self._get_weekly_support_level(df, params)
        if support_level is None:
            print("    - 结论: [失败] 因无法确定支撑位而跳过。")
            return washout_score
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
        last_support = support_level.iloc[-1]
        # print(f"    - 使用的支撑位: {last_support:.2f}")
        # print(f"    - 模式1 (日内洗盘): {'[+1分]' if washout_intraday.iloc[-1] else '[+0分]'}")
        # print(f"    - 模式2 (日间洗盘): {'[+1分]' if washout_interday.iloc[-1] else '[+0分]'}")
        # print(f"    - 模式3 (漂移收复): {'[+1分]' if washout_drift.iloc[-1] else '[+0分]'}")
        # print(f"    - 模式4 (诱多陷阱): {'[+1分]' if washout_bull_trap.iloc[-1] else '[+0分]'}")
        # print(f"    - 模式5 (缩量确认): {'[+1分]' if washout_volume_contraction.iloc[-1] else '[+0分]'}")
        # print(f"    - 结论: 最新一周总得分为 [{washout_score.iloc[-1]}]")
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
            boll_period = self.playbook_params.get('box_consolidation_breakout_playbook', {}).get('boll_period', 20)
            boll_std = self.playbook_params.get('box_consolidation_breakout_playbook', {}).get('boll_std', 2.0)
            bbw_col = f"BBW_{boll_period}_{float(boll_std)}_W"
            if not self._check_dependencies(df, [bbw_col, 'low_W'], log_details=False): return None
            quantile_level = self.playbook_params.get('box_consolidation_breakout_playbook', {}).get('bbw_quantile', 0.3)
            threshold = df[bbw_col].quantile(quantile_level)
            is_consolidating = df[bbw_col] < threshold
            if is_consolidating.any():
                box_period = self.playbook_params.get('box_consolidation_breakout_playbook', {}).get('box_period', 26)
                box_bottom = df['low_W'].rolling(window=box_period, min_periods=1).min()
                support_level = box_bottom.where(is_consolidating, np.nan)
        if support_level.isnull().all():
            return None
        return support_level.ffill()

    def _playbook_check_rejection_filters(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """诊断剧本：识别均线和箱体压力位的拒绝信号"""
        print(f"\n--- 诊断检查: [{params.get('说明', '压力位拒绝信号')}] ---")
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

    def _playbook_box_consolidation_breakout(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """剧本：专业箱体突破"""
        # print(f"\n--- 剧本检查: [{params.get('说明', '专业箱体突破')}] ---")
        quantile_level = params.get('bbw_quantile', 0.3)
        boll_period = params.get('boll_period', 20)
        boll_std = self.indicator_cfg.get('boll_bands_and_width', {}).get('std_dev', 2.0)
        box_period = params.get('box_period', 26)
        volume_multiplier = params.get('volume_multiplier', 1.5)
        vol_ma_period = params.get('vol_ma_period', 5)
        bbw_col = f"BBW_{boll_period}_{float(boll_std)}_W"
        vol_ma_col = f"VOL_MA_{vol_ma_period}_W"
        required_cols = ['close_W', 'high_W', 'volume_W', bbw_col, vol_ma_col]
        if not self._check_dependencies(df, required_cols, log_details=True):
            print("    - 结论: [失败] 依赖检查失败，策略提前退出。")
            return pd.Series(False, index=df.index)
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
        last_idx = -1
        prev_bbw = df[bbw_col].iloc[last_idx - 1]
        prev_bbw_thresh = dynamic_bbw_threshold.iloc[last_idx - 1]
        prev_box_high = box_high.shift(1).iloc[last_idx]
        prev_box_avg_vol = box_avg_volume.shift(1).iloc[last_idx]
        curr_close = df['close_W'].iloc[last_idx]
        curr_vol = df['volume_W'].iloc[last_idx]
        c1 = was_in_consolidation.iloc[last_idx]
        c2 = is_price_breakout.iloc[last_idx]
        c3 = is_volume_breakout.iloc[last_idx]
        # print(f"    - 条件1 (前一周处于盘整期): {'[✓]' if c1 else '[✗]'} (前周BBW: {prev_bbw:.4f} vs 动态阈值: {prev_bbw_thresh:.4f})")
        # print(f"    - 条件2 (价格突破箱顶): {'[✓]' if c2 else '[✗]'} (本周收盘: {curr_close:.2f} vs 前周箱顶: {prev_box_high:.2f})")
        # print(f"    - 条件3 (成交量突破): {'[✓]' if c3 else '[✗]'} (本周成交量: {curr_vol:.0f} vs 阈值: {(prev_box_avg_vol * volume_multiplier):.0f})")
        # print(f"    - 结论: 最新一周信号为 [{'触发' if final_signal.iloc[last_idx] else '未触发'}]")
        return final_signal.fillna(False)

    def _playbook_trix_golden_cross(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【V3.2 升级剧本】: 识别周线TRIX“强力金叉”。
        - 核心修改: 增加TRIX线自身斜率的判断，要求金叉时必须是“加速向上”的。
        """
        # print(f"\n--- 剧本检查: [{params.get('说明', 'TRIX金叉')}] ---")
        trix_cfg = self.indicator_cfg.get('trix', {})
        trix_periods = next((c.get('periods') for c in trix_cfg.get('configs', []) if 'W' in c.get('apply_on', [])), None)
        if not trix_periods or len(trix_periods) < 2:
            print("    - 结论: [失败] TRIX周期参数配置不正确。")
            return pd.Series(False, index=df.index)
        
        trix_len, signal_len = trix_periods[0], trix_periods[1]
        slope_threshold = params.get('slope_threshold', 0.01) # 从JSON获取斜率阈值
        # print(f"    - 配置参数: trix_len={trix_len}, signal_len={signal_len}, slope_threshold={slope_threshold}")

        trix_col = f'TRIX_{trix_len}_{signal_len}_W'
        trix_signal_col = f'TRIXs_{trix_len}_{signal_len}_W'
        if not self._check_dependencies(df, [trix_col, trix_signal_col], log_details=True):
            print(f"    - 结论: [失败] 缺少必要的TRIX列。")
            return pd.Series(False, index=df.index)

        # 条件1: 经典金叉
        condition1_is_golden_cross = (df[trix_col] > df[trix_signal_col]) & \
                                     (df[trix_col].shift(1) <= df[trix_signal_col].shift(1))
        
        # 条件2 (新增): TRIX线自身斜率必须大于阈值，确保是强力金叉
        trix_slope = df[trix_col].diff(1)
        condition2_is_strong_momentum = trix_slope > slope_threshold

        final_signal = condition1_is_golden_cross & condition2_is_strong_momentum
        
        last = df.iloc[-1]
        c1_last = condition1_is_golden_cross.iloc[-1]
        c2_last = condition2_is_strong_momentum.iloc[-1]
        
        # print(f"    - 条件1 (发生金叉): {'[✓]' if c1_last else '[✗]'}")
        # print(f"    - 条件2 (动能强劲): {'[✓]' if c2_last else '[✗]'} (TRIX斜率: {trix_slope.iloc[-1]:.4f} > 阈值: {slope_threshold})")
        # print(f"    - 结论: 最新一周信号为 [{'触发' if final_signal.iloc[-1] else '未触发'}]")
        
        return final_signal.fillna(False)
    
    def _playbook_coppock_reversal(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V3.4 升级剧本】: Coppock指标形态学 - 分离左侧企稳与右侧加速信号。
        - 核心修改: 不再输出单一信号，而是返回一个包含两个独立信号的字典：
          1. coppock_stabilizing (左侧): 捕捉深水区“跌势衰竭，首次拐头”的瞬间。
          2. coppock_accelerating (右侧): 捕捉拐头后“上涨加速，动能确认”的瞬间。
        """
        # print(f"\n--- 剧本检查: [{params.get('说明', 'Coppock双信号反转')}] ---")
        coppock_cfg = self.indicator_cfg.get('coppock', {})
        coppock_periods = next((c.get('periods') for c in coppock_cfg.get('configs', []) if 'W' in c.get('apply_on', [])), None)
        if not coppock_periods or len(coppock_periods) < 3:
            print("    - 结论: [失败] Coppock周期参数配置不正确。")
            return {}
        
        p1, p2, p3 = coppock_periods[0], coppock_periods[1], coppock_periods[2]
        deep_value_threshold = params.get('deep_value_threshold', -100)
        accel_threshold = params.get('accel_threshold', 10) # 上涨加速度阈值
        # print(f"    - 配置参数: deep_value={deep_value_threshold}, accel_threshold={accel_threshold}")

        coppock_col = f'COPP_{p1}_{p2}_{p3}_W'
        if not self._check_dependencies(df, [coppock_col], log_details=True):
            print(f"    - 结论: [失败] 缺少必要的Coppock列。")
            return {}

        # --- 计算基础变量：斜率(速度)和加速度 ---
        slope = df[coppock_col].diff(1)
        acceleration = slope.diff(1)

        # --- 信号1: 左侧企稳信号 (Coppock Stabilizing) ---
        # 条件1.1: 发生拐头 (斜率由负/零转正)
        is_turning_up = (slope > 0) & (slope.shift(1) <= 0)
        # 条件1.2: 拐头必须发生在深水区
        was_in_deep_zone = df[coppock_col].shift(1) < deep_value_threshold
        signal_stabilizing = is_turning_up & was_in_deep_zone

        # --- 信号2: 右侧加速信号 (Coppock Accelerating) ---
        # 条件2.1: 必须已经处于上升趋势中 (斜率为正)
        is_rising = slope > 0
        # 条件2.2: 加速度首次超过阈值
        is_accelerating = (acceleration > accel_threshold) & (acceleration.shift(1) <= accel_threshold)
        signal_accelerating = is_rising & is_accelerating

        # --- 调试信息 ---
        last_idx = -1
        s_stab_last = signal_stabilizing.iloc[last_idx]
        s_accel_last = signal_accelerating.iloc[last_idx]
        # print(f"    - [左侧信号: 企稳]")
        # print(f"      - 条件1 (深水区拐头): {'[✓]' if s_stab_last else '[✗]'} (上周值: {df[coppock_col].shift(1).iloc[last_idx]:.2f} < {deep_value_threshold} AND 发生拐头)")
        # print(f"    - [右侧信号: 加速]")
        # print(f"      - 条件1 (上涨加速): {'[✓]' if s_accel_last else '[✗]'} (加速度: {acceleration.iloc[last_idx]:.2f} > {accel_threshold} AND 首次满足)")
        # print(f"    - 结论: 左侧信号=[{'触发' if s_stab_last else '未触发'}], 右侧信号=[{'触发' if s_accel_last else '未触发'}]")

        return {
            'coppock_stabilizing': signal_stabilizing.fillna(False),
            'coppock_accelerating': signal_accelerating.fillna(False)
        }
   
    def _playbook_ace_signal_breakout_trigger(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """【新增剧本】: 王牌突破信号，结合年度高点突破、放量和TRIX确认。"""
        # print(f"\n--- 剧本检查: [{params.get('说明', '王牌突破信号')}] ---")
        
        lookback_weeks = params.get('lookback_weeks', 52)
        volume_multiplier = params.get('volume_multiplier', 2.0)
        vol_ma_period = params.get('vol_ma_period', 5)
        trix_confirm = params.get('trix_confirm', True)
        
        vol_ma_col = f'VOL_MA_{vol_ma_period}_W'
        required_cols = ['high_W', 'close_W', 'volume_W', vol_ma_col]
        
        trix_col, trix_signal_col = None, None
        if trix_confirm:
            trix_cfg = self.indicator_cfg.get('trix', {})
            trix_periods = next((c.get('periods') for c in trix_cfg.get('configs', []) if 'W' in c.get('apply_on', [])), None)
            if trix_periods and len(trix_periods) >= 2:
                trix_len, signal_len = trix_periods[0], trix_periods[1]
                trix_col = f'TRIX_{trix_len}_{signal_len}_W'
                trix_signal_col = f'TRIXs_{trix_len}_{signal_len}_W'
                required_cols.extend([trix_col, trix_signal_col])
            else:
                print("    - 警告: TRIX确认已启用，但无法在指标配置中找到有效的周线TRIX参数。")
                trix_confirm = False

        if not self._check_dependencies(df, required_cols, log_details=True):
            print(f"    - 结论: [失败] 缺少必要列")
            return pd.Series(False, index=df.index)

        period_high = df['high_W'].shift(1).rolling(window=lookback_weeks, min_periods=int(lookback_weeks*0.8)).max()
        is_price_breakout = df['close_W'] > period_high

        is_volume_breakout = df['volume_W'] > (df[vol_ma_col] * volume_multiplier)

        is_trix_ok = pd.Series(True, index=df.index)
        if trix_confirm:
            is_trix_ok = df[trix_col] > df[trix_signal_col]

        final_signal = is_price_breakout & is_volume_breakout & is_trix_ok

        last = df.iloc[-1]
        c1 = is_price_breakout.iloc[-1]
        c2 = is_volume_breakout.iloc[-1]
        c3 = is_trix_ok.iloc[-1]
        
        # print(f"    - 条件1 (突破年线): {'[✓]' if c1 else '[✗]'} (收盘价: {last.get('close_W', 0):.2f} vs 前{lookback_weeks}周高点: {period_high.iloc[-1]:.2f})")
        # print(f"    - 条件2 (2倍放量): {'[✓]' if c2 else '[✗]'} (成交量: {last.get('volume_W', 0):.0f} vs 阈值: {(last.get(vol_ma_col, 0) * volume_multiplier):.0f})")
        # if trix_confirm:
        #     print(f"    - 条件3 (TRIX确认): {'[✓]' if c3 else '[✗]'} (TRIX: {last.get(trix_col, 0):.2f} > 信号线: {last.get(trix_signal_col, 0):.2f})")
        
        # print(f"    - 结论: 最新一周信号为 [{'触发' if final_signal.iloc[-1] else '未触发'}]")
        
        return final_signal.fillna(False)

    def _check_resistance_rejection(self, df: pd.DataFrame, resistance_col: str, params: dict, source_name: str) -> pd.Series:
        """辅助函数: 检查在给定压力列上的拒绝信号"""
        print(f"  - 检查子项: [{source_name}]")
        volume_multiplier = params.get('volume_multiplier', 1.5)
        vol_ma_period = self.indicator_cfg.get('vol_ma', {}).get('periods', [5, 20, 55])[-1]
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
        last = df.iloc[-1]
        c1 = is_near_resistance.iloc[-1]
        c2 = is_long_upper_shadow.iloc[-1]
        c3 = is_high_volume.iloc[-1]
        c4 = is_closing_lower.iloc[-1]
        # print(f"    - 条件1 (触及压力): {'[✓]' if c1 else '[✗]'} (最高价: {last.get('high_W', 0):.2f} vs 压力: {last.get(resistance_col, 0):.2f})")
        # print(f"    - 条件2 (长上影线): {'[✓]' if c2 else '[✗]'}")
        # print(f"    - 条件3 (放出大量): {'[✓]' if c3 else '[✗]'} (成交量: {last.get('volume_W', 0):.0f} vs 阈值: {(last.get(vol_ma_col, 0) * volume_multiplier):.0f})")
        # print(f"    - 条件4 (收盘偏低): {'[✓]' if c4 else '[✗]'}")
        # print(f"    - 小结: [{source_name}] {'触发' if final_signal.iloc[-1] else '未触发'}")
        return final_signal.fillna(False)

    def _check_dependencies(self, df: pd.DataFrame, cols: list, log_details: bool = False) -> bool:
        """检查DataFrame中是否存在所有必需的列。"""
        missing_cols = [col for col in cols if col not in df.columns]
        if missing_cols:
            if log_details:
                print(f"      - [依赖检查] 失败! 缺少以下必需列: {missing_cols}")
            if not hasattr(self, '_warned_missing_cols_weekly'):
                self._warned_missing_cols_weekly = set()
            if tuple(missing_cols) not in self._warned_missing_cols_weekly:
                 logger.warning(f"周线策略缺少必需列: {missing_cols}，相关剧本将跳过。")
                 self._warned_missing_cols_weekly.add(tuple(missing_cols))
            return False
        return True
