# 文件: strategies/trend_following/exit_layer.py
# 离场层
import pandas as pd
from typing import Tuple, Dict # 导入 Tuple 和 Dict
from .utils import get_params_block, get_param_value

class ExitLayer:
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    # 专门负责生成基于规则的、非评分系统的硬性离场信号
    def generate_hard_exit_triggers(self) -> pd.DataFrame:
        """
        【V501.0 · 双均线立体防御版】硬性离场触发器生成器
        - 核心升级: 引入了双均线立体防御体系，区分“战术止损”和“战略清仓”。
          - MA21 (战术线): 触发 'EXIT_TREND_BROKEN'，用于战术性离场，保护利润。
          - MA55 (战略线): 触发 'EXIT_STRATEGY_INVALIDATED'，用于战略性清仓，确认趋势终结。
        - 收益: 实现了战略定力与战术灵活性的完美结合，极大提升了持仓的稳定性和风险控制的精确度。
        """
        df = self.strategy.df_indicators
        triggers_df = pd.DataFrame(index=df.index)
        
        # --- 初始化所有防线 ---
        triggers_df['EXIT_CRITICAL_HIT'] = pd.Series(False, index=df.index)
        triggers_df['EXIT_RISK_OVERFLOW'] = pd.Series(False, index=df.index)
        triggers_df['EXIT_PROFIT_PROTECT'] = pd.Series(False, index=df.index)
        
        # 构建双均线立体防御体系
        p_pos_mgmt = get_params_block(self.strategy, 'position_management_params')
        p_trailing = p_pos_mgmt.get('trailing_stop', {})
        
        # 初始化两条防线为不触发
        triggers_df['EXIT_TREND_BROKEN'] = pd.Series(False, index=df.index) # 战术破位
        triggers_df['EXIT_STRATEGY_INVALIDATED'] = pd.Series(False, index=df.index) # 战略失效

        if get_param_value(p_trailing.get('enabled'), False):
            # 从配置中读取战术和战略均线参数
            tactical_ma_type = get_param_value(p_trailing.get('tactical_ma_type'), 'EMA').upper()
            tactical_ma_period = get_param_value(p_trailing.get('tactical_ma_period'), 21)
            tactical_ma_col = f'{tactical_ma_type}_{tactical_ma_period}_D'
            
            if tactical_ma_col in df.columns:
                # 当日收盘价低于战术移动平均线，触发趋势破位信号
                triggers_df['EXIT_TREND_BROKEN'] = df['close_D'] < df[tactical_ma_col]
                print(f"    -> [离场层] 战术破位监控已激活 (基于 {tactical_ma_col})。")
            else:
                print(f"    -> [离场层-警告] 无法找到战术移动平均线列: {tactical_ma_col}，战术破位监控未激活。")

            # 增加战略生命线（如EMA55）的判断
            strategic_ma_type = get_param_value(p_trailing.get('strategic_ma_type'), 'EMA').upper()
            strategic_ma_period = get_param_value(p_trailing.get('strategic_ma_period'), 55)
            strategic_ma_col = f'{strategic_ma_type}_{strategic_ma_period}_D'

            if strategic_ma_col in df.columns:
                # 当日收盘价低于战略移动平均线，触发战略失效信号
                triggers_df['EXIT_STRATEGY_INVALIDATED'] = df['close_D'] < df[strategic_ma_col]
                print(f"    -> [离场层] 战略失效监控已激活 (基于 {strategic_ma_col})。")
            else:
                print(f"    -> [离场层-警告] 无法找到战略移动平均线列: {strategic_ma_col}，战略失效监控未激活。")

        return triggers_df

