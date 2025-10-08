# /strategy/intelligence/predictive_intelligence.py
import pandas as pd
import numpy as np
from typing import Dict
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
        【V1.3 · 三位一体融合版】(已撤销哈迪斯审判协议) 诊断“高潮衰竭”风险
        - 核心革命: 废除二进制的“门控”逻辑，升级为“三位一体”的模拟信号融合。
                      最终风险分 = (亢奋分 * 权重) + (天量分 * 权重) + (K线疲弱分 * 权重)。
        - 收益: 解决了因单一维度（如亢奋分）未达到极端阈值而导致整个预警系统哑火的致命缺陷，
                使得“先知”的判断更加综合、稳健，能在多个风险因素共振时发出更可靠的警报。
        """
        # [代码撤销] 移除所有与上下文计算相关的代码，回归纯粹的风险计算
        # 1. 支柱一: 亢奋分 (Euphoria Score)
        euphoria_score = atomic_states.get('COGNITIVE_SCORE_RISK_EUPHORIA_ACCELERATION', pd.Series(0, index=df.index))
        # 2. 支柱二: 天量分 (Climax Volume Score) - 从二进制门升级为模拟分数
        vol_lookback = get_param_value(self.params.get('exhaustion_vol_lookback'), 20)
        # 使用 normalize_score 生成0-1之间的模拟天量分
        volume_score = normalize_score(df['volume_D'], df.index, window=vol_lookback, ascending=True)
        # 3. 支柱三: K线疲弱分 (Kline Weakness Score)
        high_low_range = df['high_D'] - df['low_D']
        upper_shadow = df['high_D'] - np.maximum(df['open_D'], df['close_D'])
        high_low_range = high_low_range.replace(0, np.nan)
        upper_shadow_ratio = (upper_shadow / high_low_range).fillna(0)
        close_position_in_range = ((df['close_D'] - df['low_D']) / high_low_range).fillna(0.5)
        upper_shadow_score = np.clip(upper_shadow_ratio * 2, 0, 1)
        weak_close_score = 1 - close_position_in_range
        kline_weakness_score = upper_shadow_score * 0.4 + weak_close_score * 0.6
        # 4. 三位一体融合 (Trinity Fusion)
        # 废除旧的门控逻辑，采用加权算术平均进行融合
        weights = get_param_value(self.params.get('trinity_fusion_weights'), {'euphoria': 0.2, 'volume': 0.4, 'kline': 0.4})
        final_risk_score = (
            euphoria_score * weights['euphoria'] +
            volume_score * weights['volume'] +
            kline_weakness_score * weights['kline']
        )
        # [代码撤销] 移除 contextual_damper 的应用
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





