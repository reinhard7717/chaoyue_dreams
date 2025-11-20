# 文件: strategies/trend_following/structural_defense_layer.py
# 离场层 -> 结构防御层
import pandas as pd
from typing import Tuple, Dict
from strategies.trend_following.utils import get_params_block, get_param_value

class StructuralDefenseLayer: # 类名从 ExitLayer 修改为 StructuralDefenseLayer
    """
    【V600.0 · 圣殿骑士版】结构防御层
    - 核心哲学: 本模块是策略的最后防线，负责守护策略赖以生存的结构基础。
                  其指令是基于客观事实的、绝对的、拥有最高否决权的硬性离场信号。
    """
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance
    def generate_hard_exit_triggers(self) -> pd.DataFrame:
        """
        【V602.0 · 战神之盾协议版】硬性离场触发器生成器 (最后防线)
        - 核心升级: 战略失效离场信号现在基于从中央获取的“最后防线”(LAST_STAND_LINE)，
                      实现了攻防逻辑的最终统一。
        """
        df = self.strategy.df_indicators
        triggers_df = pd.DataFrame(index=df.index)
        p_pos_mgmt = get_params_block(self.strategy, 'position_management_params')
        p_trailing = p_pos_mgmt.get('trailing_stop', {})
        triggers_df['EXIT_TREND_BROKEN'] = pd.Series(False, index=df.index)
        triggers_df['EXIT_STRATEGY_INVALIDATED'] = pd.Series(False, index=df.index)
        if get_param_value(p_trailing.get('enabled'), False):
            tactical_ma_type = get_param_value(p_trailing.get('tactical_ma_type'), 'EMA').upper()
            tactical_ma_period = get_param_value(p_trailing.get('tactical_ma_period'), 21)
            tactical_ma_col = f'{tactical_ma_type}_{tactical_ma_period}_D'
            if tactical_ma_col in df.columns:
                triggers_df['EXIT_TREND_BROKEN'] = df['close_D'] < df[tactical_ma_col]
            else:
                print(f"    -> [结构防御层-警告] 无法找到战术移动平均线列: {tactical_ma_col}，战术防线未激活。")
            # 不再使用静态的strategic_ma_col，而是从原子状态中获取动态的“最后防线”
            last_stand_line = self.strategy.atomic_states.get('LAST_STAND_LINE')
            strategic_exit_confirmation_days = get_param_value(p_trailing.get('strategic_exit_confirmation_days'), 2)
            if last_stand_line is not None and not last_stand_line.empty:
                is_below_last_stand = df['close_D'] < last_stand_line
                triggers_df['EXIT_STRATEGY_INVALIDATED'] = is_below_last_stand.rolling(
                    window=strategic_exit_confirmation_days, min_periods=strategic_exit_confirmation_days
                ).sum() >= strategic_exit_confirmation_days
            else:
                print(f"    -> [结构防御层-警告] 无法从atomic_states获取'LAST_STAND_LINE'，战略防线未激活。")
        return triggers_df












