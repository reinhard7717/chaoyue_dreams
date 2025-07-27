# 文件: strategies/trend_following/simulation_layer.py
# 模拟层
from .utils import get_params_block, get_param_value
from typing import Tuple

class SimulationLayer:
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def run_position_management_simulation(self):
        print("\n" + "="*20 + " 【战术持仓管理模拟引擎 V190.0】启动 " + "="*20)
        df = self.strategy.df_indicators
        sim_params = get_params_block(self.strategy, 'position_management_params')
        if not get_param_value(sim_params.get('enabled'), False):
            print("    - 持仓管理模拟被禁用，跳过。")
            return

        level_2_reduction = get_param_value(sim_params.get('level_2_alert_reduction_pct'), 0.3)
        level_3_reduction = get_param_value(sim_params.get('level_3_alert_reduction_pct'), 0.5)
        
        df['position_size'] = 0.0
        df['alert_level'] = 0
        df['alert_reason'] = ''
        df['trade_action'] = ''

        in_position = False
        position_size = 0.0
        partial_exit_level_2_done = False

        for row in df.itertuples():
            current_date = row.Index
            if not in_position:
                if row.signal_entry:
                    in_position = True
                    position_size = 1.0
                    partial_exit_level_2_done = False
                    df.loc[current_date, 'trade_action'] = 'ENTRY'
            else:
                if row.exit_signal_code > 0:
                    in_position = False
                    position_size = 0.0
                    df.loc[current_date, 'trade_action'] = f'EXIT (Code:{row.exit_signal_code})'
                    df.loc[current_date, 'position_size'] = position_size
                    continue

                alert_level, alert_reason = self._check_tactical_alerts(row)
                df.loc[current_date, 'alert_level'] = alert_level
                df.loc[current_date, 'alert_reason'] = alert_reason

                if alert_level == 3:
                    if position_size > 0:
                        reduction_amount = position_size * level_3_reduction
                        position_size -= reduction_amount
                        df.loc[current_date, 'trade_action'] = f'REDUCE_L3 ({level_3_reduction:.0%})'
                elif alert_level == 2 and not partial_exit_level_2_done:
                    if position_size > 0:
                        reduction_amount = position_size * level_2_reduction
                        position_size -= reduction_amount
                        df.loc[current_date, 'trade_action'] = f'REDUCE_L2 ({level_2_reduction:.0%})'
                        partial_exit_level_2_done = True
                
                if df.loc[current_date, 'trade_action'] == '':
                    df.loc[current_date, 'trade_action'] = 'HOLD'
            df.loc[current_date, 'position_size'] = position_size
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
