# /strategy/intelligence/predictive_intelligence.py
import pandas as pd
import numpy as np
from typing import Dict, Any
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score

class PredictiveIntelligence:
    """
    【V1.0 · 先知引擎】
    - 核心职责: 专注于生成预测性信号，旨在预判 T+1 的重大风险或机会。
    - 首个模型: “高潮衰竭”模型，用于在崩盘前夜（T日）发出预警。
    """
    def __init__(self, strategy_context):
        self.strategy = strategy_context
        self.params = get_params_block(self.strategy, 'predictive_intelligence_params', {})
    def _get_safe_series(self, df: pd.DataFrame, column_name: str, default_value: Any = 0.0, method_name: str = "未知方法") -> pd.Series:
        """
        安全地从DataFrame获取Series，如果不存在则打印警告并返回默认Series。
        """
        if column_name not in df.columns:
            print(f"    -> [结构情报警告] 方法 '{method_name}' 缺少数据 '{column_name}'，使用默认值 {default_value}。")
            return pd.Series(default_value, index=df.index)
        return df[column_name]
    def run_predictive_diagnostics(self) -> Dict[str, pd.Series]:
        """
        【V2.0 · 德尔菲神谕版】运行所有预测性诊断模型
        - 核心升级: 新增对“恐慌投降反转”机会的预测能力。
        """
        states = {}
        if not get_param_value(self.params.get('enabled'), True):
            return states
        df = self.strategy.df_indicators
        atomic_states = self.strategy.atomic_states
        # 调用“高潮衰竭”风险诊断
        exhaustion_risk = self._diagnose_climactic_exhaustion(df, atomic_states)
        states['PREDICTIVE_RISK_CLIMACTIC_RUN_EXHAUSTION'] = exhaustion_risk.astype(np.float32)
        # 调用全新的“恐慌投降反转”机会诊断
        capitulation_opportunity = self._diagnose_capitulation_reversal(df, atomic_states)
        states['PREDICTIVE_OPP_CAPITULATION_REVERSAL'] = capitulation_opportunity.astype(np.float32)
        return states
    def _diagnose_climactic_exhaustion(self, df: pd.DataFrame, atomic_states: Dict) -> pd.Series:
        """
        【V1.4 · 日间影线版】(已撤销哈迪斯审判协议) 诊断“高潮衰竭”风险
        - 核心升级: 签署“日间影线”协议。上影线的计算基准从当日开盘价改为昨日收盘价，
                      以更精确地衡量价格从高点回落对日间涨幅的侵蚀程度。
        """
        p_conf = get_params_block(self.strategy, 'predictive_intelligence_params', {})
        weights = get_param_value(p_conf.get('trinity_fusion_weights'), {'euphoria': 0.2, 'volume': 0.4, 'kline': 0.4})
        euphoria_score = atomic_states.get('COGNITIVE_SCORE_RISK_EUPHORIA_ACCELERATION', pd.Series(0, index=df.index))
        vol_lookback = get_param_value(p_conf.get('exhaustion_vol_lookback'), 20)
        volume_score = normalize_score(df['volume_D'], df.index, window=vol_lookback, ascending=True)
        high_low_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        # 实施“日间影线”协议
        upper_shadow = df['high_D'] - np.maximum(df['close_D'], df['pre_close_D'])
        upper_shadow_ratio = (upper_shadow / high_low_range).fillna(0)
        close_position_in_range = ((df['close_D'] - df['low_D']) / high_low_range).fillna(0.5)
        upper_shadow_score = np.clip(upper_shadow_ratio * 2, 0, 1)
        weak_close_score = 1 - close_position_in_range
        kline_weakness_score = upper_shadow_score * 0.4 + weak_close_score * 0.6
        final_risk_score = (
            euphoria_score * weights['euphoria'] +
            volume_score * weights['volume'] +
            kline_weakness_score * weights['kline']
        )
        return final_risk_score.clip(0, 1)
    def _diagnose_capitulation_reversal(self, df: pd.DataFrame, atomic_states: Dict) -> pd.Series:
        """
        【V2.0 · 三位一体版】诊断“恐慌投降反转”机会 (入场神谕)
        - 核心革命: 废除旧的、简化的单维模型，严格按照配置文件的 `capitulation_reversal_weights`
                      实现“恐慌”、“衰竭”、“反转”三位一体的加权融合预测模型。
        - 收益: 使得“入场神谕”的预测不再仅仅依赖恐慌程度，而是综合评估了市场是否同时具备
                “卖盘衰竭”和“行为反转”的特征，极大提升了预测信号的可靠性和准确性。
        """
        # 1. 支柱一: 恐慌分 (Panic Score)
        # 获取由 TacticEngine 精心合成的、基于【五维立体模型】的“恐慌战备”分数。
        panic_score = atomic_states.get('SCORE_SETUP_PANIC_SELLING', pd.Series(0.0, index=df.index))
        # 2. 支柱二: 衰竭分 (Exhaustion Score)
        # 获取行为层诊断出的“卖盘衰竭反转”信号，代表卖方力量的枯竭程度。
        exhaustion_score = atomic_states.get('SCORE_BULLISH_EXHAUSTION_REVERSAL', pd.Series(0.0, index=df.index))
        # 3. 支柱三: 反转分 (Reversal Score)
        # 获取行为层诊断出的原子级“反弹/反转”信号，代表买方力量的显性反攻。
        reversal_score = atomic_states.get('SCORE_ATOMIC_REBOUND_REVERSAL', pd.Series(0.0, index=df.index))
        # 4. 三位一体融合 (Trinity Fusion)
        # 从配置中读取三大支柱的权重
        weights = get_param_value(self.params.get('capitulation_reversal_weights'), {'panic': 0.4, 'exhaustion': 0.3, 'reversal': 0.3})
        # 执行加权算术平均融合
        final_opportunity_score = (
            panic_score * weights.get('panic', 0.4) +
            exhaustion_score * weights.get('exhaustion', 0.3) +
            reversal_score * weights.get('reversal', 0.3)
        )
        return final_opportunity_score.clip(0, 1)





