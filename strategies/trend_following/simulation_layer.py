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
        【V509.0 盘后引擎精细化交易动作与T+1开盘价入场】
        - 核心重构: 模拟层现在更精确地处理盘后引擎的交易逻辑。
        - 核心逻辑:
          1. 入场价: 默认使用T+1的开盘价作为入场价。如果T+1开盘价缺失，则当日不建仓。
          2. 清仓后: 彻底重置持仓状态，下一次入场将视为“首次建仓”。
          3. 加仓: 在盈利持仓期间，出现新的买入信号时，执行金字塔加仓，并更新平均成本。
          4. 减仓: 根据风险分自动触发分级减仓。
          5. 交易动作: 输出更精细的交易动作，包括区分止盈/止损/风险/趋势破位清仓，
                       首次建仓/加仓，以及在无明确交易时根据动态力学战术矩阵输出状态。
        """
        print("\n" + "="*20 + " 【战术持仓管理模拟引擎 V509.0】启动 " + "="*20)
        # 修改代码行：使用copy避免SettingWithCopyWarning，并确保操作的是独立副本
        df = self.strategy.df_indicators.copy() 
        
        # 新增代码行：确保 'open_D' 列存在，如果不存在，则创建并填充NaN
        if 'open_D' not in df.columns:
            df['open_D'] = pd.NA
            print("    - 警告: 'open_D' 列不存在，将使用NaN填充。T+1开盘价入场将受影响。")

        sim_params = get_params_block(self.strategy, 'position_management_params')
        if not get_param_value(sim_params.get('enabled'), False):
            print("    - 持仓管理模拟被禁用，跳过。")
            # 修改代码行：即使禁用，也需要初始化 trade_action 列，否则 reporting_layer 会报错
            df['trade_action'] = StrategyDailyScore.TradeActionType.NO_SIGNAL.value
            self.strategy.df_indicators = df # 确保更新回策略实例
            return

        # --- 读取减仓和加仓参数 ---
        p_reduce = sim_params.get('risk_based_reduction', {})
        level_2_reduction = get_param_value(p_reduce.get('level_2_alert_reduction_pct'), 0.3)
        level_3_reduction = get_param_value(p_reduce.get('level_3_alert_reduction_pct'), 0.5)
        
        p_pyramid = sim_params.get('pyramiding', {})
        pyramiding_enabled = get_param_value(p_pyramid.get('enabled'), False)
        add_size_ratio = get_param_value(p_pyramid.get('add_size_ratio'), 0.5)
        max_pyramid_count = get_param_value(p_pyramid.get('max_pyramid_count'), 2)
        
        # --- 初始化模拟状态列和变量 ---
        df['position_size'] = 0.0
        df['alert_level'] = 0
        df['alert_reason'] = ''
        df['trade_action'] = StrategyDailyScore.TradeActionType.NO_SIGNAL.value
        df['current_profit_loss_pct'] = 0.0
        # 新增代码行：记录实际入场价或平均成本
        df['entry_price_actual'] = 0.0 

        # 模拟过程中的实时状态变量
        in_position = False
        current_position_size = 0.0 
        actual_entry_price = 0.0    
        pyramid_count = 0
        last_reduction_level = 0

        # --- 核心模拟循环 ---
        for i in range(len(df)): # 修改代码行：使用索引迭代
            current_date = df.index[i]
            row = df.iloc[i] # 获取当前行数据
            current_price = row.close_D
            
            # 获取当日的信号类型和动态动作
            current_signal_type = getattr(row, 'signal_type', '无信号')
            current_dynamic_action = getattr(row, 'dynamic_action', 'HOLD')

            # 计算当前盈亏百分比
            if in_position and actual_entry_price > 0:
                df.loc[current_date, 'current_profit_loss_pct'] = (current_price - actual_entry_price) / actual_entry_price
            else:
                df.loc[current_date, 'current_profit_loss_pct'] = 0.0

            # --- 1. 持仓状态下的决策 ---
            if in_position:
                # 1.1 检查清仓信号 (三道防线)
                exit_triggers = self.strategy.exit_triggers.loc[current_date]
                if exit_triggers.any():
                    triggered_reasons = exit_triggers[exit_triggers].index.tolist()
                    
                    exit_action = StrategyDailyScore.TradeActionType.RISK_EXIT.value # 默认风险清仓
                    if 'EXIT_PROFIT_PROTECT' in triggered_reasons:
                        exit_action = StrategyDailyScore.TradeActionType.PROFIT_EXIT.value
                    elif 'EXIT_TREND_BROKEN' in triggered_reasons:
                        exit_action = StrategyDailyScore.TradeActionType.TREND_BROKEN_EXIT.value
                    elif 'EXIT_CRITICAL_HIT' in triggered_reasons or 'EXIT_RISK_OVERFLOW' in triggered_reasons:
                        exit_action = StrategyDailyScore.TradeActionType.RISK_EXIT.value
                    # 新增代码行：明确止损清仓
                    elif 'EXIT_STOP_LOSS' in triggered_reasons: 
                        exit_action = StrategyDailyScore.TradeActionType.STOP_LOSS_EXIT.value
                    
                    print(f"  -> {current_date.date()}: [清仓离场] 触发三道防线: {', '.join(triggered_reasons)}")
                    in_position = False
                    current_position_size = 0.0 # 清仓后重置
                    actual_entry_price = 0.0    # 清仓后重置
                    pyramid_count = 0           # 清仓后重置
                    last_reduction_level = 0    # 清仓后重置
                    df.loc[current_date, 'trade_action'] = exit_action
                    df.loc[current_date, 'position_size'] = current_position_size
                    df.loc[current_date, 'entry_price_actual'] = actual_entry_price # 记录清仓后的实际入场价为0
                    continue # 当天清仓后，不再执行其他操作

                # 1.2 检查加仓信号 (金字塔)
                is_profitable = df.loc[current_date, 'current_profit_loss_pct'] > 0
                if pyramiding_enabled and row.signal_entry and is_profitable and pyramid_count < max_pyramid_count:
                    add_amount = 1.0 * add_size_ratio # 假设初始仓位为1.0
                    
                    # 新增代码行：更新平均成本
                    old_total_cost = current_position_size * actual_entry_price
                    new_total_size = current_position_size + add_amount
                    new_total_cost = old_total_cost + (add_amount * current_price) # 加仓价格为当日收盘价
                    
                    current_position_size = new_total_size
                    actual_entry_price = new_total_cost / new_total_size
                    pyramid_count += 1
                    
                    df.loc[current_date, 'trade_action'] = StrategyDailyScore.TradeActionType.ADD_POSITION.value
                    print(f"  -> {current_date.date()}: [乘胜追击] 盈利中出现新买点，执行第 {pyramid_count} 次加仓。新均价: {actual_entry_price:.2f}")
                    last_reduction_level = 0 # 加仓后重置减仓状态

                # 1.3 检查减仓信号 (风险控制)
                alert_level, alert_reason = self._check_tactical_alerts(row)
                df.loc[current_date, 'alert_level'] = alert_level
                df.loc[current_date, 'alert_reason'] = alert_reason

                # 只有在风险升级时才减仓
                if alert_level > last_reduction_level:
                    # 新增代码行：减仓动作统一为 REDUCE_POSITION
                    reduction_action = StrategyDailyScore.TradeActionType.REDUCE_POSITION.value
                    
                    if alert_level == 3: # 高度风险
                        reduction_amount = current_position_size * level_3_reduction
                        current_position_size -= reduction_amount
                        df.loc[current_date, 'trade_action'] = reduction_action
                        last_reduction_level = 3
                        print(f"  -> {current_date.date()}: [风险减仓] 风险升至3级，减仓 {level_3_reduction:.0%}")
                    elif alert_level == 2: # 中度风险
                        reduction_amount = current_position_size * level_2_reduction
                        current_position_size -= reduction_amount
                        df.loc[current_date, 'trade_action'] = reduction_action
                        last_reduction_level = 2
                        print(f"  -> {current_date.date()}: [风险减仓] 风险升至2级，减仓 {level_2_reduction:.0%}")
                
                # 1.4 如果无特殊动作，则标记为持有
                # 修改代码行：检查是否已被其他动作覆盖
                if df.loc[current_date, 'trade_action'] == StrategyDailyScore.TradeActionType.NO_SIGNAL.value: 
                    df.loc[current_date, 'trade_action'] = StrategyDailyScore.TradeActionType.HOLD.value

            # --- 2. 空仓状态下的决策 ---
            else:
                if row.signal_entry:
                    # 盘后引擎，入场价为T+1开盘价
                    t_plus_1_open = pd.NA
                    if i + 1 < len(df):
                        next_day_index = df.index[i + 1]
                        t_plus_1_open = df.loc[next_day_index, 'open_D']
                    
                    # 检查T+1开盘价是否有效
                    if pd.notna(t_plus_1_open) and t_plus_1_open > 0:
                        in_position = True
                        current_position_size = 1.0 # 初始仓位
                        actual_entry_price = t_plus_1_open # 使用T+1开盘价
                        pyramid_count = 1 # 首次建仓也算一次“加仓”
                        last_reduction_level = 0
                        df.loc[current_date, 'trade_action'] = StrategyDailyScore.TradeActionType.INITIAL_ENTRY.value
                        print(f"  -> {current_date.date()}: [建立仓位] 信号分值达标，入场。T+1开盘价: {actual_entry_price:.2f}")
                    else:
                        # 如果T+1开盘价缺失，则不建仓，标记为无信号
                        df.loc[current_date, 'trade_action'] = StrategyDailyScore.TradeActionType.NO_SIGNAL.value
                        print(f"  -> {current_date.date()}: [跳过建仓] 信号分值达标，但T+1开盘价缺失或无效，跳过建仓。")
                else:
                    # 空仓时，根据 dynamic_action 记录更精细的状态
                    if current_dynamic_action == 'AVOID':
                        df.loc[current_date, 'trade_action'] = StrategyDailyScore.TradeActionType.AVOID.value
                    elif current_dynamic_action == 'PROCEED_WITH_CAUTION':
                        df.loc[current_date, 'trade_action'] = StrategyDailyScore.TradeActionType.PROCEED_WITH_CAUTION.value
                    elif current_dynamic_action == 'FORCE_ATTACK':
                        df.loc[current_date, 'trade_action'] = StrategyDailyScore.TradeActionType.FORCE_ATTACK.value
                    else:
                        df.loc[current_date, 'trade_action'] = StrategyDailyScore.TradeActionType.NO_SIGNAL.value

            # 每次循环结束时，更新df中的持仓状态
            df.loc[current_date, 'position_size'] = current_position_size
            df.loc[current_date, 'entry_price_actual'] = actual_entry_price # 记录实际入场价
        
        # 修改代码行：将更新后的DataFrame保存回策略实例
        self.strategy.df_indicators = df 
        print("="*25 + " 【持仓管理模拟】执行完毕 " + "="*25 + "\n")

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
