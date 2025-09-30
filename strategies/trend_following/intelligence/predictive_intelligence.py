# /strategy/intelligence/predictive_intelligence.py
import pandas as pd
import numpy as np
from typing import Dict
from strategies.trend_following.utils import get_params_block, get_param_value

class PredictiveIntelligence:
    """
    【V1.0 · 先知引擎】
    - 核心职责: 专注于生成预测性信号，旨在预判 T+1 的重大风险或机会。
    - 首个模型: “高潮衰竭”模型，用于在崩盘前夜（T日）发出预警。
    """
    def __init__(self, strategy_context):
        self.strategy = strategy_context
        self.params = get_params_block(self.strategy, 'predictive_intelligence_params', {})

    def run_predictive_diagnostics(self) -> Dict[str, pd.Series]:
        """运行所有预测性诊断模型"""
        states = {}
        if not get_param_value(self.params.get('enabled'), True):
            return states
        
        df = self.strategy.df_indicators
        atomic_states = self.strategy.atomic_states
        
        # 调用“高潮衰竭”诊断
        exhaustion_risk = self._diagnose_climactic_exhaustion(df, atomic_states)
        states['PREDICTIVE_RISK_CLIMACTIC_RUN_EXHAUSTION'] = exhaustion_risk.astype(np.float32)
        
        return states

    def _diagnose_climactic_exhaustion(self, df: pd.DataFrame, atomic_states: Dict) -> pd.Series:
        """
        【V1.2 · 神谕扩音版】诊断“高潮衰竭”风险
        - 核心升级: 将kline_weakness_score的计算方式从保守的“几何平均”升级为更具进攻性的“加权算术平均”，
                      赋予“收盘价疲弱”更高的权重，以放大神谕的音量。
        """
        # 1. 获取亢奋信号
        euphoria_score = atomic_states.get('COGNITIVE_SCORE_RISK_EUPHORIA_ACCELERATION', pd.Series(0, index=df.index))
        # 2. 定义天量
        vol_lookback = get_param_value(self.params.get('exhaustion_vol_lookback'), 20)
        is_climax_volume = (df['volume_D'] >= df['volume_D'].rolling(vol_lookback).max() * 0.9).astype(int)
        # 3. 定义冲高回落 (长上影线 + 收盘价偏低)
        high_low_range = df['high_D'] - df['low_D']
        upper_shadow = df['high_D'] - np.maximum(df['open_D'], df['close_D'])
        high_low_range = high_low_range.replace(0, np.nan)
        upper_shadow_ratio = (upper_shadow / high_low_range).fillna(0)
        close_position_in_range = ((df['close_D'] - df['low_D']) / high_low_range).fillna(0.5)
        # 4. 融合计算风险分
        # 条件1: 亢奋程度必须很高
        euphoria_gate = (euphoria_score > get_param_value(self.params.get('exhaustion_euphoria_threshold'), 0.45)).astype(int)
        # 条件2: 必须是天量
        volume_gate = is_climax_volume
        # 条件3: K线形态必须是冲高回落
        upper_shadow_score = np.clip(upper_shadow_ratio * 2, 0, 1)
        weak_close_score = 1 - close_position_in_range
        # [代码修改] 废除保守的几何平均，采用更灵敏的加权算术平均，放大神谕音量
        kline_weakness_score = upper_shadow_score * 0.4 + weak_close_score * 0.6
        # 最终风险分是三者的乘积
        final_risk_score = (euphoria_gate * volume_gate) * kline_weakness_score
        return final_risk_score.clip(0, 1)






