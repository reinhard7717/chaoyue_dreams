# File: strategies/trend_following/simulation_layer.py
# Version: V509.0 - 盘后引擎精细化交易动作与T+1开盘价入场
# 描述: 模拟层现在更精确地处理盘后引擎的交易逻辑，包括T+1开盘价入场，
#       清仓后重置状态，以及更精细的交易动作判断。

from strategies.trend_following.utils import get_params_block, get_param_value
from typing import Tuple
import pandas as pd
from stock_models.stock_analytics import StrategyDailyScore # 导入 StrategyDailyScore.TradeActionType

class SimulationLayer:
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance
    def run_position_management_simulation(self):
        """
        【V516.0 · 解除武装版】
        - 核心革命: 彻底剥夺模拟层修改 `signal_type` 的权力，确保其绝对服从 JudgmentLayer 的最终裁决。
        - 核心逻辑: 删除了在硬性离场时，模拟层越权修改 `signal_type` 的所有代码。
        - 收益: 根除了因模拟层篡改历史而导致的信号污染问题，恢复了指挥链的绝对权威。
        """
        df = self.strategy.df_indicators.copy() 
        if 'open_D' not in df.columns:
            df['open_D'] = pd.NA
            print("    - 警告: 'open_D' 列不存在，将使用NaN填充。T+1开盘价入场将受影响。")
        sim_params = get_params_block(self.strategy, 'position_management_params')
        if not get_param_value(sim_params.get('enabled'), False):
            df['trade_action'] = StrategyDailyScore.TradeActionType.NO_SIGNAL.value
            self.strategy.df_indicators = df
            return
        p_open_filter = sim_params.get('opening_filter', {})
        opening_filter_enabled = get_param_value(p_open_filter.get('enabled'), True)
        max_opening_gap_pct = get_param_value(p_open_filter.get('max_opening_gap_pct'), 5.0) / 100.0
        p_reduce = sim_params.get('risk_based_reduction', {})
        level_2_reduction = get_param_value(p_reduce.get('level_2_alert_reduction_pct'), 0.3)
        level_3_reduction = get_param_value(p_reduce.get('level_3_alert_reduction_pct'), 0.5)
        p_pyramid = sim_params.get('pyramiding', {})
        pyramiding_enabled = get_param_value(p_pyramid.get('enabled'), False)
        add_size_ratio = get_param_value(p_pyramid.get('add_size_ratio'), 0.5)
        max_pyramid_count = get_param_value(p_pyramid.get('max_pyramid_count'), 2)
        p_stop_loss = sim_params.get('stop_loss', {})
        stop_loss_enabled = get_param_value(p_stop_loss.get('enabled'), False)
        initial_stop_model = get_param_value(p_stop_loss.get('initial_stop_model'))
        atr_multiplier_for_platform = get_param_value(p_stop_loss.get('atr_multiplier_for_platform'), 0.5)
        min_stop_loss_percent = get_param_value(p_stop_loss.get('min_stop_loss_percent'), 4.0) / 100.0
        df['position_size'] = 0.0
        df['alert_level'] = 0
        df['alert_reason'] = ''
        df['trade_action'] = StrategyDailyScore.TradeActionType.NO_SIGNAL.value
        df['current_profit_loss_pct'] = 0.0
        df['entry_price_actual'] = 0.0 
        in_position = False
        current_position_size = 0.0 
        actual_entry_price = 0.0    
        pyramid_count = 0
        last_reduction_level = 0
        stop_loss_price = 0.0
        for i in range(len(df)):
            current_date = df.index[i]
            row = df.iloc[i]
            current_price = row.close_D
            current_signal_type = getattr(row, 'signal_type', '无信号')
            current_dynamic_action = getattr(row, 'dynamic_action', 'HOLD')
            if in_position and actual_entry_price > 0:
                df.loc[current_date, 'current_profit_loss_pct'] = (current_price - actual_entry_price) / actual_entry_price
            else:
                df.loc[current_date, 'current_profit_loss_pct'] = 0.0
            if in_position:
                if i > 0:
                    prev_row = df.iloc[i-1]
                    prev_alert_level = getattr(prev_row, 'alert_level', 0)
                    prev_alert_reason = getattr(prev_row, 'alert_reason', '')
                    if prev_alert_level == 3 and '先知' in prev_alert_reason:
                        print(f"  -> {current_date.date()}: [先知预警离场] T-1日收到高潮衰竭警报，今日开盘执行清仓。")
                        in_position, current_position_size, actual_entry_price, pyramid_count, last_reduction_level, stop_loss_price = False, 0.0, 0.0, 0, 0, 0.0
                        df.loc[current_date, 'trade_action'] = StrategyDailyScore.TradeActionType.PROPHET_EXIT.value
                        # 移除模拟层对 signal_type 的越权修改
                        # df.loc[current_date, 'signal_type'] = '先知离场'
                        df.loc[current_date, 'position_size'] = current_position_size
                        df.loc[current_date, 'entry_price_actual'] = actual_entry_price
                        continue
                if stop_loss_enabled and current_price < stop_loss_price:
                    in_position, current_position_size, actual_entry_price, pyramid_count, last_reduction_level, stop_loss_price = False, 0.0, 0.0, 0, 0, 0.0
                    df.loc[current_date, 'trade_action'] = StrategyDailyScore.TradeActionType.STOP_LOSS_EXIT.value
                    df.loc[current_date, 'position_size'] = current_position_size
                    df.loc[current_date, 'entry_price_actual'] = actual_entry_price
                    continue
                exit_triggers = self.strategy.exit_triggers.loc[current_date]
                if exit_triggers.any():
                    triggered_reasons = exit_triggers[exit_triggers].index.tolist()
                    exit_action = StrategyDailyScore.TradeActionType.RISK_EXIT.value
                    if 'EXIT_PROFIT_PROTECT' in triggered_reasons: exit_action = StrategyDailyScore.TradeActionType.PROFIT_EXIT.value
                    elif 'EXIT_TREND_BROKEN' in triggered_reasons: exit_action = StrategyDailyScore.TradeActionType.TREND_BROKEN_EXIT.value
                    elif 'EXIT_STRATEGY_INVALIDATED' in triggered_reasons: exit_action = StrategyDailyScore.TradeActionType.STRATEGY_INVALIDATED_EXIT.value
                    in_position, current_position_size, actual_entry_price, pyramid_count, last_reduction_level, stop_loss_price = False, 0.0, 0.0, 0, 0, 0.0
                    df.loc[current_date, 'trade_action'] = exit_action
                    # 彻底移除模拟层对 signal_type 的所有越权修改
                    # if exit_action == StrategyDailyScore.TradeActionType.TREND_BROKEN_EXIT.value:
                    #     df.loc[current_date, 'signal_type'] = '趋势破位离场'
                    # elif exit_action == StrategyDailyScore.TradeActionType.STRATEGY_INVALIDATED_EXIT.value:
                    #     df.loc[current_date, 'signal_type'] = '战略失效离场'
                    df.loc[current_date, 'position_size'] = current_position_size
                    df.loc[current_date, 'entry_price_actual'] = actual_entry_price
                    continue
                is_profitable = df.loc[current_date, 'current_profit_loss_pct'] > 0
                is_pyramid_allowed = pyramiding_enabled and row.signal_entry and is_profitable and pyramid_count < max_pyramid_count
                if is_pyramid_allowed:
                    add_amount = 1.0 * add_size_ratio
                    old_total_cost = current_position_size * actual_entry_price
                    new_total_size = current_position_size + add_amount
                    new_total_cost = old_total_cost + (add_amount * current_price)
                    current_position_size = new_total_size
                    actual_entry_price = new_total_cost / new_total_size
                    pyramid_count += 1
                    df.loc[current_date, 'trade_action'] = StrategyDailyScore.TradeActionType.ADD_POSITION.value
                    last_reduction_level = 0
                alert_level, alert_reason = self._check_tactical_alerts(row)
                df.loc[current_date, 'alert_level'], df.loc[current_date, 'alert_reason'] = alert_level, alert_reason
                if alert_level > last_reduction_level:
                    reduction_action = StrategyDailyScore.TradeActionType.REDUCE_POSITION.value
                    if alert_level == 3:
                        current_position_size -= current_position_size * level_3_reduction
                        df.loc[current_date, 'trade_action'], last_reduction_level = reduction_action, 3
                    elif alert_level == 2:
                        current_position_size -= current_position_size * level_2_reduction
                        df.loc[current_date, 'trade_action'], last_reduction_level = reduction_action, 2
                if df.loc[current_date, 'trade_action'] == StrategyDailyScore.TradeActionType.NO_SIGNAL.value: 
                    df.loc[current_date, 'trade_action'] = StrategyDailyScore.TradeActionType.HOLD.value
            else:
                if row.signal_entry:
                    t_plus_1_open = pd.NA
                    if i + 1 < len(df):
                        next_day_index = df.index[i + 1]
                        t_plus_1_open = df.loc[next_day_index, 'open_D']
                    if pd.notna(t_plus_1_open) and t_plus_1_open > 0:
                        opening_gap_pct = (t_plus_1_open - row.close_D) / row.close_D
                        if opening_filter_enabled and opening_gap_pct > max_opening_gap_pct:
                            df.loc[current_date, 'trade_action'] = StrategyDailyScore.TradeActionType.GAP_UP_SKIPPED.value
                            # print(f"  -> {current_date.date()}: [跳过建仓-高开] T+1开盘价 {t_plus_1_open:.2f} (缺口: {opening_gap_pct:+.2%}) 超过容忍度 {max_opening_gap_pct:+.2%}")
                        else:
                            in_position = True
                            current_position_size = 1.0
                            actual_entry_price = t_plus_1_open
                            pyramid_count = 1
                            last_reduction_level = 0
                            if stop_loss_enabled:
                                if initial_stop_model == 'PLATFORM_SUPPORT' and 'PLATFORM_PRICE_STABLE' in df.columns and 'ATR_14_D' in df.columns:
                                    platform_price = df.loc[current_date, 'PLATFORM_PRICE_STABLE']
                                    atr_value = df.loc[current_date, 'ATR_14_D']
                                    if pd.notna(platform_price) and pd.notna(atr_value) and platform_price > 0:
                                        stop_price_from_platform = platform_price - atr_multiplier_for_platform * atr_value
                                        min_stop_price_from_entry = actual_entry_price * (1 - min_stop_loss_percent)
                                        stop_loss_price = max(stop_price_from_platform, min_stop_price_from_entry)
                                    else:
                                        stop_loss_price = actual_entry_price * (1 - min_stop_loss_percent)
                                else:
                                    stop_loss_price = actual_entry_price * (1 - min_stop_loss_percent)
                            if current_signal_type == '先知入场':
                                df.loc[current_date, 'trade_action'] = StrategyDailyScore.TradeActionType.PROPHET_ENTRY.value
                            else:
                                df.loc[current_date, 'trade_action'] = StrategyDailyScore.TradeActionType.INITIAL_ENTRY.value
                    else:
                        df.loc[current_date, 'trade_action'] = StrategyDailyScore.TradeActionType.NO_SIGNAL.value
                else:
                    if current_dynamic_action == 'AVOID': df.loc[current_date, 'trade_action'] = StrategyDailyScore.TradeActionType.AVOID.value
                    elif current_dynamic_action == 'PROCEED_WITH_CAUTION': df.loc[current_date, 'trade_action'] = StrategyDailyScore.TradeActionType.PROCEED_WITH_CAUTION.value
                    elif current_dynamic_action == 'FORCE_ATTACK': df.loc[current_date, 'trade_action'] = StrategyDailyScore.TradeActionType.FORCE_ATTACK.value
                    else: df.loc[current_date, 'trade_action'] = StrategyDailyScore.TradeActionType.NO_SIGNAL.value
            df.loc[current_date, 'position_size'] = current_position_size
            df.loc[current_date, 'entry_price_actual'] = actual_entry_price
        self.strategy.df_indicators = df
    def _check_tactical_alerts(self, row) -> Tuple[int, str]:
        """
        【V2.0 · 审判日协同版】
        - 核心升级: 不再自行计算警报等级，而是直接消费由 JudgmentLayer 的“风险裁决者”生成的
                      权威 ALERT_LEVEL 和 ALERT_REASON。
        - 收益: 实现了决策与执行的完美统一，确保仓位管理严格遵循最高指挥部的风险评级。
        """
        # 直接从每日的行数据中读取由“风险裁决者”生成的权威警报等级和原因
        alert_level = getattr(row, 'alert_level', 0)
        alert_reason = getattr(row, 'alert_reason', '')
        return alert_level, alert_reason







