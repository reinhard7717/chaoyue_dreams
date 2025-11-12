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
        【V6.4 · 纯粹原子版】基础情报分析总指挥
        - 核心升级: 废弃原子层面的“共振”和“领域健康度”信号。
        - 核心职责: 只输出基础领域的原子公理信号、基础背离信号和上下文信号。
        - 移除信号: SCORE_FOUNDATION_BULLISH_RESONANCE, SCORE_FOUNDATION_BEARISH_RESONANCE, BIPOLAR_FOUNDATION_DOMAIN_HEALTH, SCORE_FOUNDATION_BOTTOM_REVERSAL, SCORE_FOUNDATION_TOP_REVERSAL。
        """
        all_states = {}
        p_conf = get_params_block(self.strategy, 'foundation_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            print("基础情报引擎已在配置中禁用，跳过。")
            return {}
        df = self.strategy.df_indicators
        norm_window = get_param_value(p_conf.get('norm_window'), 55)
        # --- 步骤一: 诊断四大公理 ---
        axiom_trend = self._diagnose_axiom_trend(df, norm_window, p_conf)
        axiom_oscillator = self._diagnose_axiom_oscillator(df, norm_window)
        axiom_flow = self._diagnose_axiom_flow(df, norm_window)
        axiom_volatility = self._diagnose_axiom_volatility(df, norm_window)
        axiom_divergence = self._diagnose_axiom_divergence(df, norm_window)
        all_states['SCORE_FOUNDATION_AXIOM_DIVERGENCE'] = axiom_divergence
        all_states['SCORE_FOUNDATION_AXIOM_TREND'] = axiom_trend
        all_states['SCORE_FOUNDATION_AXIOM_OSCILLATOR'] = axiom_oscillator
        all_states['SCORE_FOUNDATION_AXIOM_FLOW'] = axiom_flow
        all_states['SCORE_FOUNDATION_AXIOM_VOLATILITY'] = axiom_volatility
        # --- 步骤二: 计算并发布 CONTEXT_TREND_CONFIRMED 信号 ---
        context_trend_confirmed = self._diagnose_context_trend_confirmed(df, norm_window)
        all_states.update(context_trend_confirmed)
        # 引入基础层面的看涨/看跌背离信号 (保持不变)
        bullish_divergence, bearish_divergence = bipolar_to_exclusive_unipolar(axiom_divergence)
        all_states['SCORE_FOUNDATION_BULLISH_DIVERGENCE'] = bullish_divergence.astype(np.float32)
        all_states['SCORE_FOUNDATION_BEARISH_DIVERGENCE'] = bearish_divergence.astype(np.float32)
        return all_states

    def _diagnose_context_trend_confirmed(self, df: pd.DataFrame, norm_window: int) -> Dict[str, pd.Series]:
        """
        【V1.0 · 新增】诊断内部上下文信号：趋势确认分 (CONTEXT_TREND_CONFIRMED)
        - 核心逻辑: 融合趋势强度(ADX)、方向(PDI/NDI)和健康度(BIAS)，评估上升趋势的确认程度。
        """
        adx_score = normalize_score(df.get('ADX_14_D', pd.Series(0.0, index=df.index)), df.index, norm_window, ascending=True)
        pdi_gt_ndi = (df.get('PDI_14_D', 0) > df.get('NDI_14_D', 0)).astype(float)
        pdi_slope = normalize_score(df.get('SLOPE_5_PDI_14_D', pd.Series(0.0, index=df.index)), df.index, norm_window, ascending=True)
        direction_score = (pdi_gt_ndi * pdi_slope).pow(0.5)
        bias_health_score = 1 - normalize_score(df.get('BIAS_55_D', pd.Series(0.0, index=df.index)).clip(lower=0), df.index, norm_window, ascending=True)
        trend_confirmed = (adx_score * direction_score * bias_health_score).pow(1/3).fillna(0.0)
        return {'CONTEXT_TREND_CONFIRMED': trend_confirmed.astype(np.float32)}

    def _diagnose_axiom_divergence(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V1.0 · 新增】基础公理五：诊断“基础背离”
        - 核心逻辑: 诊断价格趋势与摆动指标（如RSI）之间的背离。
          - 看涨背离：价格创新低但RSI未创新低。
          - 看跌背离：价格创新高但RSI未创新高。
        """
        price_trend = normalize_to_bipolar(df.get('SLOPE_13_close_D', pd.Series(0.0, index=df.index)), df.index, norm_window)
        oscillator_trend = normalize_to_bipolar(df.get('SLOPE_13_RSI_13_D', pd.Series(0.0, index=df.index)), df.index, norm_window)
        divergence_score = (oscillator_trend - price_trend).clip(-1, 1)
        return divergence_score.astype(np.float32)

    def _diagnose_axiom_trend(self, df: pd.DataFrame, norm_window: int, params: dict) -> pd.Series:
        """【V1.0 · 新增】基础公理一：诊断“趋势”"""
        macd_h = df.get('MACDh_13_34_8_D', pd.Series(0.0, index=df.index))
        macd_score = normalize_to_bipolar(macd_h, df.index, norm_window)
        fusion_weights = params.get('ma_health_fusion_weights', {'alignment': 0.5, 'slope': 0.5})
        ma_periods = [5, 13, 21, 55]
        bull_alignment_scores = [(df[f'EMA_{ma_periods[i]}_D'] > df[f'EMA_{ma_periods[i+1]}_D']).astype(float).values for i in range(len(ma_periods) - 1)]
        alignment_score = np.mean(bull_alignment_scores, axis=0) if bull_alignment_scores else np.full(len(df.index), 0.5)
        alignment_bipolar = (pd.Series(alignment_score, index=df.index) - 0.5) * 2
        slope_scores = [normalize_to_bipolar(df.get(f'SLOPE_{p}_EMA_{p}_D', pd.Series(0.0, index=df.index)), df.index, norm_window).values for p in ma_periods]
        avg_slope_bipolar = pd.Series(np.mean(slope_scores, axis=0), index=df.index)
        structure_score = (
            alignment_bipolar * fusion_weights.get('alignment', 0.5) +
            avg_slope_bipolar * fusion_weights.get('slope', 0.5)
        ).clip(-1, 1)
        trend_score = (macd_score * 0.4 + structure_score * 0.6).clip(-1, 1)
        return trend_score.astype(np.float32)

    def _diagnose_axiom_oscillator(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """【V1.0 · 新增】基础公理二：诊断“摆动”"""
        rsi = df.get('RSI_13_D', pd.Series(50.0, index=df.index))
        raw_bipolar_series = rsi - 50.0
        oscillator_score = normalize_to_bipolar(raw_bipolar_series, df.index, window=norm_window, sensitivity=10.0)
        return oscillator_score.astype(np.float32)

    def _diagnose_axiom_flow(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """【V1.0 · 新增】基础公理三：诊断“流体”"""
        cmf = df.get('CMF_21_D', pd.Series(0.0, index=df.index))
        flow_score = normalize_to_bipolar(cmf, df.index, window=norm_window, sensitivity=0.1)
        return flow_score.astype(np.float32)

    def _diagnose_axiom_volatility(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """【V1.0 · 新增】基础公理四：诊断“波动”"""
        bbw = df.get('BBW_21_2.0_D', pd.Series(0.0, index=df.index))
        atr_pct = df.get('ATR_14_D', pd.Series(0.0, index=df.index)) / df['close_D']
        raw_volatility = bbw + atr_pct
        raw_bipolar_series = -raw_volatility
        volatility_score = normalize_to_bipolar(raw_bipolar_series, df.index, window=norm_window, sensitivity=1.0)
        return volatility_score.astype(np.float32)

