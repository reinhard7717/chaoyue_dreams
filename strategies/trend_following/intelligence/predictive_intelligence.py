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
        【V1.4 · 哈迪斯审判协议版】诊断“高潮衰竭”风险
        - 核心革命: 签署“哈迪斯审判协议”，为“先知”注入上下文感知能力。
        - 核心逻辑: 引入权威的`bottom_context_score`作为“上下文阻尼器”。
                      最终风险 = 原始风险 * (1 - 底部上下文分数)。
                      这使得模型能够在识别出底部结构时，自动抑制“高潮衰竭”风险，
                      从而区分“顶部派发高潮”与“底部恐慌高潮”。
        """
        # 步骤0: 引入权威的底部上下文分数作为“阻尼器”
        from strategies.trend_following.utils import calculate_context_scores
        bottom_context_score, _ = calculate_context_scores(df, atomic_states)
        contextual_damper = (1.0 - bottom_context_score).clip(0, 1)
        # 1. 支柱一: 亢奋分 (Euphoria Score)
        euphoria_score = atomic_states.get('COGNITIVE_SCORE_RISK_EUPHORIA_ACCELERATION', pd.Series(0, index=df.index))
        # 2. 支柱二: 天量分 (Climax Volume Score) - 从二进制门升级为模拟分数
        vol_lookback = get_param_value(self.params.get('exhaustion_vol_lookback'), 20)
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
        weights = get_param_value(self.params.get('trinity_fusion_weights'), {'euphoria': 0.2, 'volume': 0.4, 'kline': 0.4})
        raw_risk_score = (
            euphoria_score * weights['euphoria'] +
            volume_score * weights['volume'] +
            kline_weakness_score * weights['kline']
        )
        # 步骤5: 应用哈迪斯审判，使用上下文阻尼器进行最终裁决
        final_risk_score = raw_risk_score * contextual_damper
        return final_risk_score.clip(0, 1)

    def _diagnose_capitulation_reversal(self, df: pd.DataFrame, atomic_states: Dict) -> pd.Series:
        """
        【V1.6 · 最终审判版】诊断“恐慌投降反转”机会 (入场神谕)
        - 核心升级: 明确认知到其依赖的 SCORE_SETUP_PANIC_SELLING 信号已进化为五维立体模型。
        - 新核心公式解读: 预测机会 = (价格暴跌 * 成交天量 * 筹码崩溃) × (多维绝望背景) × (结构支撑测试)
        - 收益: 使得未来的法医探针能够基于正确的公式进行解剖，彻底解决逻辑断层问题。
        """
        # 步骤一：获取由 TacticEngine 精心合成的、基于【五维立体模型】的“恐慌战备”分数。
        # 这个分数现在是全系统对“恐慌”的唯一、最高标准定义。
        panic_context_score = atomic_states.get('SCORE_SETUP_PANIC_SELLING', pd.Series(0.0, index=df.index))

        # 步骤二：先知的神谕无比纯粹——今天的“五维恐慌”程度，就是对明天反转机会的预测强度。
        final_opportunity_score = panic_context_score
        
        return final_opportunity_score.clip(0, 1)





