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
        【V500.0 职责净化版】硬性离场触发器生成器
        - 核心职责: 根据“三道防线”原则，生成一个包含所有非评分系统离场原因的布尔型DataFrame。
                      这部分逻辑原先分散在 JudgmentLayer 中，现统一收归于此，职责更清晰。
        """
        df = self.strategy.df_indicators
        triggers_df = pd.DataFrame(index=df.index)
        
        # --- 防线一: 致命一击 (Critical Hit) ---
        # 备注: 原先的致命风险现在统一在WarningLayer计分，这里可以保留一个接口
        #       用于未来可能增加的、不参与计分的、绝对的致命信号。
        triggers_df['EXIT_CRITICAL_HIT'] = pd.Series(False, index=df.index)

        # --- 防线二: 风险溢出 (Risk Overflow) ---
        # 备注: 风险溢出判断现在基于 JudgmentLayer 的最终 risk_penalty_score，
        #       因此这里的逻辑也应迁移或废弃。为保持结构，暂时置空。
        triggers_df['EXIT_RISK_OVERFLOW'] = pd.Series(False, index=df.index)

        # --- 防线三: 趋势破位 (Trend Broken) ---
        # 这是本模块的核心职责：处理技术性止损。
        p_pos_mgmt = get_params_block(self.strategy, 'position_management_params')
        p_trailing = p_pos_mgmt.get('trailing_stop', {})
        triggers_df['EXIT_TREND_BROKEN'] = pd.Series(False, index=df.index) # 默认不触发
        if get_param_value(p_trailing.get('enabled'), False):
            model = get_param_value(p_trailing.get('trailing_model'))
            if model == 'MOVING_AVERAGE':
                ma_type = get_param_value(p_trailing.get('ma_type'), 'EMA').upper()
                ma_period = get_param_value(p_trailing.get('ma_period'), 20)
                ma_col = f'{ma_type}_{ma_period}_D'
                if ma_col in df.columns:
                    # 当日收盘价低于移动平均线，则触发趋势破位信号
                    triggers_df['EXIT_TREND_BROKEN'] = df['close_D'] < df[ma_col]
                else:
                    print(f"    -> [离场层-警告] 无法找到移动平均线列: {ma_col}，趋势破位监控未激活。")

        # --- 防线四: 利润保护 (Profit Protector - 暂未完全实现) ---
        p_judge = get_params_block(self.strategy, 'four_layer_scoring_params').get('judgment_params', {})
        p_protector = p_judge.get('profit_protector', {})
        if get_param_value(p_protector.get('enabled'), False):
            # 此处未来可以实现基于持仓收益回撤的止盈逻辑
            triggers_df['EXIT_PROFIT_PROTECT'] = pd.Series(False, index=df.index)
        else:
            triggers_df['EXIT_PROFIT_PROTECT'] = pd.Series(False, index=df.index)

        return triggers_df

