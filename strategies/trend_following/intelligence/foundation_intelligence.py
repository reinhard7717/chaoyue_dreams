# 文件: strategies/trend_following/intelligence/foundation_intelligence.py
# 基础情报模块 (波动率, 震荡指标)
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score, normalize_to_bipolar, bipolar_to_exclusive_unipolar

class FoundationIntelligence:
    def __init__(self, strategy_instance):
        """
        初始化基础情报模块。
        :param strategy_instance: 策略主实例的引用，用于访问df和atomic_states。
        """
        self.strategy = strategy_instance

    def run_foundation_analysis_command(self) -> Dict[str, pd.Series]:
        """
        【V6.0 · 四大公理重构版】基础情报分析总指挥
        - 核心重构: 废弃旧的混合诊断模式，引入基于经典指标的“趋势、摆动、流体、波动”四大公理。
        - 核心流程:
          1. 诊断四大公理，生成纯粹的基础层原子信号。
          2. 融合四大公理，合成终极的基础层共振信号。
        """
        print("启动【V6.0 · 四大公理重构版】基础情报分析...")
        all_states = {}
        p_conf = get_params_block(self.strategy, 'foundation_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            print("基础情报引擎已在配置中禁用，跳过。")
            return {}
        df = self.strategy.df_indicators
        norm_window = get_param_value(p_conf.get('norm_window'), 55)
        # --- 步骤一: 诊断四大公理 ---
        print("工序一: 正在诊断四大基础公理...")
        axiom_trend = self._diagnose_axiom_trend(df, norm_window, p_conf)
        axiom_oscillator = self._diagnose_axiom_oscillator(df, norm_window)
        axiom_flow = self._diagnose_axiom_flow(df, norm_window)
        axiom_volatility = self._diagnose_axiom_volatility(df, norm_window)
        all_states['SCORE_FOUNDATION_AXIOM_TREND'] = axiom_trend
        all_states['SCORE_FOUNDATION_AXIOM_OSCILLATOR'] = axiom_oscillator
        all_states['SCORE_FOUNDATION_AXIOM_FLOW'] = axiom_flow
        all_states['SCORE_FOUNDATION_AXIOM_VOLATILITY'] = axiom_volatility
        # --- 步骤二: 融合四大公理，合成终极信号 ---
        print("工序二: 正在合成终极基础层信号...")
        axiom_weights = get_param_value(p_conf.get('axiom_weights'), {
            'trend': 0.4, 'oscillator': 0.2, 'flow': 0.3, 'volatility': 0.1
        })
        # 构造一个融合了所有公理的原始双极性健康分
        # 注意：波动公理正分代表稳定，对趋势是正面贡献
        bipolar_health = (
            axiom_trend * axiom_weights['trend'] +
            axiom_oscillator * axiom_weights['oscillator'] +
            axiom_flow * axiom_weights['flow'] +
            axiom_volatility * axiom_weights['volatility']
        ).clip(-1, 1)
        # 分解为互斥的单极性共振分
        bullish_resonance, bearish_resonance = bipolar_to_exclusive_unipolar(bipolar_health)
        all_states['SCORE_FOUNDATION_BULLISH_RESONANCE'] = bullish_resonance
        all_states['SCORE_FOUNDATION_BEARISH_RESONANCE'] = bearish_resonance
        print("【V6.0 · 四大公理重构版】基础情报分析完成。")
        return all_states

    def _diagnose_axiom_trend(self, df: pd.DataFrame, norm_window: int, params: dict) -> pd.Series:
        """【V1.0 · 新增】基础公理一：诊断“趋势”"""
        # 证据1: MACD (动能)
        macd_h = df.get('MACDh_13_34_8_D', pd.Series(0.0, index=df.index))
        macd_score = normalize_to_bipolar(macd_h, df.index, norm_window)
        # 证据2: EMA系统 (结构)
        fusion_weights = params.get('ma_health_fusion_weights', {'alignment': 0.5, 'slope': 0.5})
        ma_periods = [5, 13, 21, 55]
        # 结构-排列
        bull_alignment_scores = [(df[f'EMA_{ma_periods[i]}_D'] > df[f'EMA_{ma_periods[i+1]}_D']).astype(float).values for i in range(len(ma_periods) - 1)]
        alignment_score = np.mean(bull_alignment_scores, axis=0) if bull_alignment_scores else np.full(len(df.index), 0.5)
        alignment_bipolar = (pd.Series(alignment_score, index=df.index) - 0.5) * 2
        # 结构-斜率
        slope_scores = [normalize_to_bipolar(df.get(f'SLOPE_{p}_EMA_{p}_D', pd.Series(0.0, index=df.index)), df.index, norm_window).values for p in ma_periods]
        avg_slope_bipolar = pd.Series(np.mean(slope_scores, axis=0), index=df.index)
        # 融合结构分
        structure_score = (
            alignment_bipolar * fusion_weights.get('alignment', 0.5) +
            avg_slope_bipolar * fusion_weights.get('slope', 0.5)
        ).clip(-1, 1)
        # 最终融合：动能与结构的加权
        trend_score = (macd_score * 0.4 + structure_score * 0.6).clip(-1, 1)
        return trend_score.astype(np.float32)

    def _diagnose_axiom_oscillator(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """【V1.0 · 新增】基础公理二：诊断“摆动”"""
        # 核心逻辑: RSI偏离中线50的程度，代表了市场的超买超卖状态
        rsi = df.get('RSI_13_D', pd.Series(50.0, index=df.index))
        # 构造原始双极性序列：RSI - 50
        raw_bipolar_series = rsi - 50.0
        # 使用双极归一化引擎进行最终裁决
        oscillator_score = normalize_to_bipolar(raw_bipolar_series, df.index, window=norm_window, sensitivity=10.0)
        return oscillator_score.astype(np.float32)

    def _diagnose_axiom_flow(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """【V1.0 · 新增】基础公理三：诊断“流体”"""
        # 核心逻辑: CMF本身就是一个以0为中轴的双极性资金流指标
        cmf = df.get('CMF_21_D', pd.Series(0.0, index=df.index))
        # 直接对CMF进行双极归一化
        flow_score = normalize_to_bipolar(cmf, df.index, window=norm_window, sensitivity=0.1)
        return flow_score.astype(np.float32)

    def _diagnose_axiom_volatility(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """【V1.0 · 新增】基础公理四：诊断“波动”"""
        # 核心逻辑: 波动率越低，市场越稳定，对趋势延续越有利，应得正分。
        # 构造原始波动率序列
        bbw = df.get('BBW_21_2.0_D', pd.Series(0.0, index=df.index))
        atr_pct = df.get('ATR_14_D', pd.Series(0.0, index=df.index)) / df['close_D']
        raw_volatility = bbw + atr_pct
        # 构造原始双极性序列：取波动率的负值，使得低波动为正，高波动为负
        raw_bipolar_series = -raw_volatility
        # 使用双极归一化引擎进行最终裁决
        volatility_score = normalize_to_bipolar(raw_bipolar_series, df.index, window=norm_window, sensitivity=1.0)
        return volatility_score.astype(np.float32)









