# 文件: strategies/trend_following/structural_defense_layer.py
# 离场层 -> 结构防御层
import pandas as pd
from typing import Tuple, Dict
from .utils import get_params_block, get_param_value

class StructuralDefenseLayer: # [代码修改] 类名从 ExitLayer 修改为 StructuralDefenseLayer
    """
    【V600.0 · 圣殿骑士版】结构防御层
    - 核心哲学: 本模块是策略的最后防线，负责守护策略赖以生存的结构基础。
                  其指令是基于客观事实的、绝对的、拥有最高否决权的硬性离场信号。
    """
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def generate_hard_exit_triggers(self) -> pd.DataFrame:
        """
        【V600.0 · 圣殿骑士版】硬性离场触发器生成器 (最后防线)
        - 核心职责: 基于双均线立体防御体系，生成“战术破位”和“战略失效”两种硬性离场信号。
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
                print(f"    -> [结构防御层] 战术防线已激活 (基于 {tactical_ma_col})。")
            else:
                print(f"    -> [结构防御层-警告] 无法找到战术移动平均线列: {tactical_ma_col}，战术防线未激活。")

            strategic_ma_type = get_param_value(p_trailing.get('strategic_ma_type'), 'EMA').upper()
            strategic_ma_period = get_param_value(p_trailing.get('strategic_ma_period'), 55)
            strategic_ma_col = f'{strategic_ma_type}_{strategic_ma_period}_D'

            if strategic_ma_col in df.columns:
                triggers_df['EXIT_STRATEGY_INVALIDATED'] = df['close_D'] < df[strategic_ma_col]
                print(f"    -> [结构防御层] 战略防线已激活 (基于 {strategic_ma_col})。")
            else:
                print(f"    -> [结构防御层-警告] 无法找到战略移动平均线列: {strategic_ma_col}，战略防线未激活。")

        return triggers_df
