# File: strategies/trend_following/simulation_layer.py
# Version: V509.0 - 盘后引擎精细化交易动作与T+1开盘价入场
# 描述: 模拟层现在更精确地处理盘后引擎的交易逻辑，包括T+1开盘价入场，
#       清仓后重置状态，以及更精细的交易动作判断。

from .utils import get_params_block, get_param_value
from typing import Tuple
import pandas as pd
from stock_models.stock_analytics import StrategyDailyScore # 导入 StrategyDailyScore.TradeActionType

class SimulationLayer:
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def run_position_management_simulation(self):
        """
        【V510.0 · 开盘过滤器版】
        - 核心升级: 引入“开盘过滤器”，解决盘后信号次日追高买入的问题。
        - 核心逻辑:
          1. 新增 'opening_filter' 配置块，包含 'max_opening_gap_pct' (最大开盘缺口容忍度)。
          2. 在执行“首次建仓”前，计算T+1开盘价相对于T日收盘价的缺口。
          3. 如果缺口超过容忍度，则中止建仓，并将 'trade_action' 标记为新增的 'GAP_UP_SKIPPED' (高开跳过)。
          4. 只有在缺口可接受时，才执行建仓。
        - 收益: 极大提升了策略的实战能力，有效避免了因市场情绪过热导致的追高亏损，使入场点更具优势。
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

        # 读取开盘过滤器参数
        p_open_filter = sim_params.get('opening_filter', {})
        opening_filter_enabled = get_param_value(p_open_filter.get('enabled'), True)
        # 将百分比转换为小数，例如 5.0 -> 0.05
        max_opening_gap_pct = get_param_value(p_open_filter.get('max_opening_gap_pct'), 5.0) / 100.0

        # --- 其他参数读取 ---
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

        # --- 初始化模拟状态列和变量 ---
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

        # --- 核心模拟循环 ---
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

            # --- 1. 持仓状态下的决策 ---
            if in_position:
                # ... (此处省略所有持仓中的止损、止盈、加减仓逻辑，它们保持不变) ...
                # 1.1 检查止损信号
                if stop_loss_enabled and current_price < stop_loss_price:
                    in_position, current_position_size, actual_entry_price, pyramid_count, last_reduction_level, stop_loss_price = False, 0.0, 0.0, 0, 0, 0.0
                    df.loc[current_date, 'trade_action'] = StrategyDailyScore.TradeActionType.STOP_LOSS_EXIT.value
                    df.loc[current_date, 'position_size'] = current_position_size
                    df.loc[current_date, 'entry_price_actual'] = actual_entry_price
                    continue
                # 1.2 检查清仓信号
                exit_triggers = self.strategy.exit_triggers.loc[current_date]
                if exit_triggers.any():
                    triggered_reasons = exit_triggers[exit_triggers].index.tolist()
                    exit_action = StrategyDailyScore.TradeActionType.RISK_EXIT.value
                    if 'EXIT_PROFIT_PROTECT' in triggered_reasons: exit_action = StrategyDailyScore.TradeActionType.PROFIT_EXIT.value
                    elif 'EXIT_TREND_BROKEN' in triggered_reasons: exit_action = StrategyDailyScore.TradeActionType.TREND_BROKEN_EXIT.value
                    in_position, current_position_size, actual_entry_price, pyramid_count, last_reduction_level, stop_loss_price = False, 0.0, 0.0, 0, 0, 0.0
                    df.loc[current_date, 'trade_action'] = exit_action
                    df.loc[current_date, 'position_size'] = current_position_size
                    df.loc[current_date, 'entry_price_actual'] = actual_entry_price
                    continue
                # 1.3 检查加仓信号
                is_profitable = df.loc[current_date, 'current_profit_loss_pct'] > 0
                if pyramiding_enabled and row.signal_entry and is_profitable and pyramid_count < max_pyramid_count:
                    add_amount = 1.0 * add_size_ratio
                    old_total_cost = current_position_size * actual_entry_price
                    new_total_size = current_position_size + add_amount
                    new_total_cost = old_total_cost + (add_amount * current_price)
                    current_position_size = new_total_size
                    actual_entry_price = new_total_cost / new_total_size
                    pyramid_count += 1
                    df.loc[current_date, 'trade_action'] = StrategyDailyScore.TradeActionType.ADD_POSITION.value
                    last_reduction_level = 0
                # 1.4 检查减仓信号
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
                # 1.5 标记为持有
                if df.loc[current_date, 'trade_action'] == StrategyDailyScore.TradeActionType.NO_SIGNAL.value: 
                    df.loc[current_date, 'trade_action'] = StrategyDailyScore.TradeActionType.HOLD.value

            # --- 2. 空仓状态下的决策 ---
            else:
                if row.signal_entry:
                    t_plus_1_open = pd.NA
                    if i + 1 < len(df):
                        next_day_index = df.index[i + 1]
                        t_plus_1_open = df.loc[next_day_index, 'open_D']
                    
                    if pd.notna(t_plus_1_open) and t_plus_1_open > 0:
                        # 在此植入开盘过滤器
                        opening_gap_pct = (t_plus_1_open - row.close_D) / row.close_D
                        
                        if opening_filter_enabled and opening_gap_pct > max_opening_gap_pct:
                            # 如果高开幅度超过容忍度，则跳过建仓
                            df.loc[current_date, 'trade_action'] = StrategyDailyScore.TradeActionType.GAP_UP_SKIPPED.value
                            print(f"  -> {current_date.date()}: [跳过建仓-高开] T+1开盘价 {t_plus_1_open:.2f} (缺口: {opening_gap_pct:+.2%}) 超过容忍度 {max_opening_gap_pct:+.2%}")
                        else:
                            # 缺口在可接受范围内，执行原有的建仓逻辑
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
                            
                            df.loc[current_date, 'trade_action'] = StrategyDailyScore.TradeActionType.INITIAL_ENTRY.value
                            # print(f"  -> {current_date.date()}: [建立仓位] 信号达标，T+1开盘价 {actual_entry_price:.2f} (缺口: {opening_gap_pct:+.2%}) 在容忍范围内。")
                    else:
                        df.loc[current_date, 'trade_action'] = StrategyDailyScore.TradeActionType.NO_SIGNAL.value
                        # print(f"  -> {current_date.date()}: [跳过建仓] 信号达标，但T+1开盘价缺失或无效。")
                else:
                    # ... (空仓无信号时的逻辑不变) ...
                    if current_dynamic_action == 'AVOID': df.loc[current_date, 'trade_action'] = StrategyDailyScore.TradeActionType.AVOID.value
                    elif current_dynamic_action == 'PROCEED_WITH_CAUTION': df.loc[current_date, 'trade_action'] = StrategyDailyScore.TradeActionType.PROCEED_WITH_CAUTION.value
                    elif current_dynamic_action == 'FORCE_ATTACK': df.loc[current_date, 'trade_action'] = StrategyDailyScore.TradeActionType.FORCE_ATTACK.value
                    else: df.loc[current_date, 'trade_action'] = StrategyDailyScore.TradeActionType.NO_SIGNAL.value

            df.loc[current_date, 'position_size'] = current_position_size
            df.loc[current_date, 'entry_price_actual'] = actual_entry_price
        
        self.strategy.df_indicators = df

    def _check_tactical_alerts(self, row) -> Tuple[int, str]:
        exit_params = get_params_block(self.strategy, 'exit_strategy_params')
        warning_params = exit_params.get('warning_threshold_params', {})
        exit_threshold_params = exit_params.get('exit_threshold_params', {})
        if not warning_params and not exit_threshold_params:
            return 0, ''
        risk_score = getattr(row, 'risk_score', 0)
        if risk_score <= 0:
            return 0, ''
        all_alerts = []
        level_map = {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        for name, config in exit_threshold_params.items():
            if name.upper() in level_map:
                all_alerts.append({'level_code': level_map[name.upper()], 'threshold': get_param_value(config.get('level'), float('inf')), 'reason': get_param_value(config.get('cn_name'), name)})
        for name, config in warning_params.items():
            if name.upper() in level_map:
                all_alerts.append({'level_code': level_map[name.upper()], 'threshold': get_param_value(config.get('level'), float('inf')), 'reason': get_param_value(config.get('cn_name'), name)})
        sorted_alerts = sorted(all_alerts, key=lambda x: x['threshold'], reverse=True)
        for alert in sorted_alerts:
            if risk_score >= alert['threshold']:
                return alert['level_code'], alert['reason']
        return 0, ''
