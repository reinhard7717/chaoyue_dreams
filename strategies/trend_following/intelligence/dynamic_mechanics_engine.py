# 文件: strategies/trend_following/intelligence/dynamic_mechanics_engine.py
# 动态力学引擎
import pandas as pd
import numpy as np
from typing import Dict
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score, normalize_to_bipolar

class DynamicMechanicsEngine:
    def __init__(self, strategy_instance):
        """
        初始化动态力学引擎。
        :param strategy_instance: 策略主实例的引用，用于访问 df_indicators。
        """
        self.strategy = strategy_instance

    def run_dynamic_analysis_command(self) -> Dict[str, pd.Series]:
        """
        【V5.0 · 四大公理重构版】动态力学引擎总指挥
        - 核心重构: 废弃旧的五支柱模型，引入基于物理学思想的“动量、惯性、波动性、能量”四大公理。
        - 核心流程:
          1. 诊断四大公理，生成纯粹的力学原子信号。
          2. 融合四大公理，合成终极的动态力学健康度。
        """
        all_dynamic_states = {}
        p_conf = get_params_block(self.strategy, 'dynamic_mechanics_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            print("动态力学引擎已在配置中禁用，跳过。")
            return {}
        df = self.strategy.df_indicators
        norm_window = get_param_value(p_conf.get('norm_window'), 55)
        # --- 步骤一: 诊断四大公理 ---
        all_dynamic_states['SCORE_DYN_AXIOM_MOMENTUM'] = self._diagnose_axiom_momentum(df, norm_window)
        all_dynamic_states['SCORE_DYN_AXIOM_INERTIA'] = self._diagnose_axiom_inertia(df, norm_window)
        all_dynamic_states['SCORE_DYN_AXIOM_STABILITY'] = self._diagnose_axiom_stability(df, norm_window)
        all_dynamic_states['SCORE_DYN_AXIOM_ENERGY'] = self._diagnose_axiom_energy(df, norm_window)
        # --- 步骤二: 融合四大公理，合成终极信号 ---
        axiom_weights = get_param_value(p_conf.get('axiom_weights'), {
            'momentum': 0.3, 'inertia': 0.3, 'stability': 0.2, 'energy': 0.2
        })
        bullish_health = (
            all_dynamic_states['SCORE_DYN_AXIOM_MOMENTUM'].clip(lower=0) * axiom_weights['momentum'] +
            all_dynamic_states['SCORE_DYN_AXIOM_INERTIA'].clip(lower=0) * axiom_weights['inertia'] +
            all_dynamic_states['SCORE_DYN_AXIOM_STABILITY'].clip(lower=0) * axiom_weights['stability'] +
            all_dynamic_states['SCORE_DYN_AXIOM_ENERGY'].clip(lower=0) * axiom_weights['energy']
        ).clip(0, 1)
        bearish_health = (
            all_dynamic_states['SCORE_DYN_AXIOM_MOMENTUM'].clip(upper=0).abs() * axiom_weights['momentum'] +
            all_dynamic_states['SCORE_DYN_AXIOM_INERTIA'].clip(upper=0).abs() * axiom_weights['inertia'] +
            all_dynamic_states['SCORE_DYN_AXIOM_STABILITY'].clip(upper=0).abs() * axiom_weights['stability'] +
            all_dynamic_states['SCORE_DYN_AXIOM_ENERGY'].clip(upper=0).abs() * axiom_weights['energy']
        ).clip(0, 1)
        all_dynamic_states['SCORE_DYN_BULLISH_RESONANCE'] = bullish_health.astype(np.float32)
        all_dynamic_states['SCORE_DYN_BEARISH_RESONANCE'] = bearish_health.astype(np.float32)
        return all_dynamic_states

    def _diagnose_axiom_momentum(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """【V1.0 · 新增】力学公理一：诊断“动量”"""
        # 证据1: 价格变化率 (速度)
        roc = df.get('ROC_12_D', pd.Series(0.0, index=df.index))
        # 证据2: MACD动能柱 (加速度)
        macd_h = df.get('MACDh_13_34_8_D', pd.Series(0.0, index=df.index))
        # 归一化为双极性分数
        roc_score = normalize_to_bipolar(roc, df.index, norm_window)
        macd_h_score = normalize_to_bipolar(macd_h, df.index, norm_window)
        # 融合: 速度和加速度的加权平均
        momentum_score = (roc_score * 0.6 + macd_h_score * 0.4).clip(-1, 1)
        return momentum_score.astype(np.float32)

    def _diagnose_axiom_inertia(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V2.1 · 元特征增强版】力学公理二：诊断“惯性”
        - 核心升级: 引入分形维度(FRACTAL_DIMENSION)作为新证据。一个真正的惯性趋势，不仅需要序列记忆(Hurst)，其路径也必须是相对平滑的（低分形维度）。
        """
        # 证据1: ADX (趋势强度)
        adx_strength = normalize_score(df.get('ADX_14_D', pd.Series(0.0, index=df.index)), df.index, norm_window, ascending=True)
        # 证据2: PDI vs NDI (趋势方向)
        adx_direction = (df.get('PDI_14_D', 0) > df.get('NDI_14_D', 0)).astype(float) * 2 - 1 # 映射到 [-1, 1]
        # 证据3: Hurst指数 (序列记忆性)
        hurst_col = next((col for col in df.columns if col.startswith('hurst_')), 'hurst_144d_D')
        hurst = df.get(hurst_col, pd.Series(0.5, index=df.index)).fillna(0.5)
        hurst_bipolar = (hurst - 0.5) * 2 # 映射到 [-1, 1]
        # 证据4 (新增): 分形维度 (路径平滑度) - 分形维度越低，路径越平滑，趋势性越强
        fractal_col = next((col for col in df.columns if col.startswith('FRACTAL_DIMENSION_')), None)
        if fractal_col:
            # 分形维度通常在[1, 2]之间，我们希望它越接近1越好，因此用 (2 - D)
            fractal_dim = df.get(fractal_col, pd.Series(1.5, index=df.index)).fillna(1.5)
            fractal_smoothness = normalize_score(2.0 - fractal_dim, df.index, norm_window)
        else:
            fractal_smoothness = pd.Series(0.5, index=df.index)
        # 融合: 强度 * 方向 * 记忆性 * 平滑度
        raw_bipolar_series = adx_strength * adx_direction * hurst_bipolar * fractal_smoothness
        # 使用双极归一化引擎进行最终裁决
        inertia_score = normalize_to_bipolar(raw_bipolar_series, df.index, window=norm_window, sensitivity=1.0)
        return inertia_score.astype(np.float32)

    def _diagnose_axiom_stability(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V2.1 · 元特征增强版】力学公理三：诊断“稳定性”
        - 核心升级: 引入波动率的不稳定性(VOLATILITY_INSTABILITY_INDEX)作为新证据。一个真正稳定的结构，不仅波动率要低，波动率本身也不能剧烈变化。
        """
        # 证据1: 波动率大小 (越小越好)
        bbw = df.get('BBW_21_2.0_D', pd.Series(0.0, index=df.index))
        atr_pct = df.get('ATR_14_D', pd.Series(0.0, index=df.index)) / df['close_D']
        raw_volatility = bbw + atr_pct
        volatility_level_score = 1 - normalize_score(raw_volatility, df.index, norm_window)
        # 证据2 (新增): 波动率的稳定性 (波动率的波动率，越小越好)
        vol_instability_col = next((col for col in df.columns if col.startswith('VOLATILITY_INSTABILITY_INDEX_')), None)
        if vol_instability_col:
            vol_of_vol = df.get(vol_instability_col, pd.Series(0.0, index=df.index))
            volatility_stability_score = 1 - normalize_score(vol_of_vol, df.index, norm_window)
        else:
            volatility_stability_score = pd.Series(0.5, index=df.index)
        # 融合两大证据
        raw_stability_score = (volatility_level_score * volatility_stability_score).pow(0.5)
        # 转换为双极性分数：高稳定性为正，低稳定性为负
        stability_score = (raw_stability_score * 2 - 1).clip(-1, 1)
        return stability_score.astype(np.float32)

    def _diagnose_axiom_energy(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """【V1.0 · 新增】力学公理四：诊断“能量”"""
        # 证据1: VPA效率 (量价关系)
        vpa = df.get('VPA_EFFICIENCY_D', pd.Series(0.5, index=df.index))
        # 证据2: CMF (资金流量)
        cmf = df.get('CMF_21_D', pd.Series(0.0, index=df.index))
        # 归一化
        vpa_bipolar = (vpa * 2 - 1).clip(-1, 1) # VPA本身在[0,1]区间
        cmf_bipolar = normalize_to_bipolar(cmf, df.index, norm_window)
        # 融合
        energy_score = (vpa_bipolar.abs() * cmf_bipolar.abs()).pow(0.5) * np.sign(vpa_bipolar) * np.sign(cmf_bipolar)
        return energy_score.astype(np.float32).clip(-1, 1)














