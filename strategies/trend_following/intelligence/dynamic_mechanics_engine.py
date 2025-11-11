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
        【V5.5 · 背离公理增强版】动态力学引擎总指挥
        - 核心修复: 修正了输出的共振信号名称，将 'SCORE_DYN_*' 修正为 'SCORE_DYNAMIC_MECHANICS_*'，
                      以严格遵守与融合层的情报供应契约。
        - 探针植入: 新增探针，打印各公理的最终得分，以便追踪融合结果。
        - 【新增】引入力学背离公理。
        """
        p_conf = get_params_block(self.strategy, 'dynamic_mechanics_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            print("-> [指挥覆盖探针] 动态力学引擎在配置中被禁用，跳过分析。")
            return {}
        print("-> [指挥覆盖探针] 动态力学引擎已启用，开始分析...")
        all_dynamic_states = {}
        df = self.strategy.df_indicators
        norm_window = get_param_value(p_conf.get('norm_window'), 55)
        axiom_momentum = self._diagnose_axiom_momentum(df, norm_window)
        axiom_inertia = self._diagnose_axiom_inertia(df, norm_window)
        axiom_stability = self._diagnose_axiom_stability(df, norm_window)
        axiom_energy = self._diagnose_axiom_energy(df, norm_window)
        axiom_divergence = self._diagnose_axiom_divergence(df, norm_window)
        all_dynamic_states['SCORE_DYN_AXIOM_DIVERGENCE'] = axiom_divergence
        all_dynamic_states['SCORE_DYN_AXIOM_MOMENTUM'] = axiom_momentum
        all_dynamic_states['SCORE_DYN_AXIOM_INERTIA'] = axiom_inertia
        all_dynamic_states['SCORE_DYN_AXIOM_STABILITY'] = axiom_stability
        all_dynamic_states['SCORE_DYN_AXIOM_ENERGY'] = axiom_energy
        axiom_weights = get_param_value(p_conf.get('axiom_weights'), {
            'momentum': 0.3, 'inertia': 0.3, 'stability': 0.2, 'energy': 0.2, 'divergence': 0.0 # [代码修改] 新增divergence权重
        })
        bipolar_health = (
            axiom_momentum * axiom_weights['momentum'] +
            axiom_inertia * axiom_weights['inertia'] +
            axiom_stability * axiom_weights['stability'] +
            axiom_energy * axiom_weights['energy']
        ).clip(-1, 1)
        # --- 内部探针 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date = probe_date_naive.tz_localize(df.index.tz) if df.index.tz else probe_date_naive
            if probe_date in df.index:
                print(f"    -> [力学引擎内部探针] @ {probe_date.date()}:")
                print(f"       - 动量公理分 (Momentum): {axiom_momentum.loc[probe_date]:.4f}")
                print(f"       - 惯性公理分 (Inertia): {axiom_inertia.loc[probe_date]:.4f}")
                print(f"       - 稳定公理分 (Stability): {axiom_stability.loc[probe_date]:.4f}")
                print(f"       - 能量公理分 (Energy): {axiom_energy.loc[probe_date]:.4f}")
                print(f"       - 融合健康分 (Bipolar Health): {bipolar_health.loc[probe_date]:.4f}")
        from strategies.trend_following.utils import bipolar_to_exclusive_unipolar
        bullish_resonance, bearish_resonance = bipolar_to_exclusive_unipolar(bipolar_health)
        all_dynamic_states['SCORE_DYNAMIC_MECHANICS_BULLISH_RESONANCE'] = bullish_resonance.astype(np.float32)
        all_dynamic_states['SCORE_DYNAMIC_MECHANICS_BEARISH_RESONANCE'] = bearish_resonance.astype(np.float32)
        # 引入力学层面的看涨/看跌背离信号
        bullish_divergence, bearish_divergence = bipolar_to_exclusive_unipolar(axiom_divergence)
        all_dynamic_states['SCORE_DYNAMIC_MECHANICS_BULLISH_DIVERGENCE'] = bullish_divergence.astype(np.float32)
        all_dynamic_states['SCORE_DYNAMIC_MECHANICS_BEARISH_DIVERGENCE'] = bearish_divergence.astype(np.float32)
        return all_dynamic_states

    def _diagnose_axiom_divergence(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V1.0 · 新增】力学公理五：诊断“力学背离”
        - 核心逻辑: 诊断价格动量与惯性之间的背离。
          - 看涨背离：价格动量减弱（负）但惯性增强（正） -> 预示趋势可能反转向上。
          - 看跌背离：价格动量增强（正）但惯性减弱（负） -> 预示趋势可能反转向下。
        """
        # 证据1: 价格动量 (来自_diagnose_axiom_momentum)
        momentum_score = self._diagnose_axiom_momentum(df, norm_window)
        # 证据2: 惯性 (来自_diagnose_axiom_inertia)
        inertia_score = self._diagnose_axiom_inertia(df, norm_window)
        # 融合：当动量与惯性方向相反时，产生背离信号
        # 看涨背离：动量负（价跌）但惯性正（趋势自我维持能力强）
        # 看跌背离：动量正（价涨）但惯性负（趋势自我维持能力弱）
        # 我们可以用 (inertia_score - momentum_score) 来捕捉这种矛盾
        # 动量正惯性负: (负 - 正) = 负 -> 看跌背离
        # 动量负惯性正: (正 - 负) = 正 -> 看涨背离
        divergence_score = (inertia_score - momentum_score).clip(-1, 1)
        return divergence_score.astype(np.float32)

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
        【V2.2 · 稳健融合重构版】力学公理二：诊断“惯性”
        - 核心重构: 废除脆弱的乘法融合模型，改为更稳健的加权平均模型。
        - 新逻辑: 1. 将趋势强度(ADX)、序列记忆(Hurst)、路径平滑度(Fractal)分别评估为[0,1]的“惯性质量分”。
                   2. 将这三个质量分加权平均，得到一个总体的“惯性质量”。
                   3. 使用趋势方向(PDI/NDI)作为最终的符号，决定惯性是看涨还是看跌。
                   这从根本上解决了因单一维度疲软而导致“一票否决”的系统性风险。
        """
        # 证据1: ADX (趋势强度)，归一化到[0,1]
        adx_strength = normalize_score(df.get('ADX_14_D', pd.Series(0.0, index=df.index)), df.index, norm_window, ascending=True)
        # 证据2: Hurst指数 (序列记忆性)，归一化到[0,1]
        hurst_col = next((col for col in df.columns if col.startswith('hurst_')), 'hurst_144d_D')
        hurst = df.get(hurst_col, pd.Series(0.5, index=df.index)).fillna(0.5)
        # Hurst > 0.5 代表趋势性，分数越高越好
        hurst_quality = normalize_score(hurst, df.index, norm_window, ascending=True)
        # 证据3: 分形维度 (路径平滑度)，归一化到[0,1]
        fractal_col = next((col for col in df.columns if col.startswith('FRACTAL_DIMENSION_')), None)
        if fractal_col:
            # 分形维度越低，路径越平滑，趋势性越强，因此 ascending=False
            fractal_dim = df.get(fractal_col, pd.Series(1.5, index=df.index)).fillna(1.5)
            fractal_smoothness = normalize_score(fractal_dim, df.index, norm_window, ascending=False)
        else:
            fractal_smoothness = pd.Series(0.5, index=df.index)
        # 步骤1: 融合三大质量证据，得到一个[0,1]的“惯性质量分”
        inertia_quality = (
            adx_strength * 0.4 +
            hurst_quality * 0.4 +
            fractal_smoothness * 0.2
        ).clip(0, 1)
        # 步骤2: 获取趋势方向
        adx_direction = (df.get('PDI_14_D', 0) > df.get('NDI_14_D', 0)).astype(float) * 2 - 1
        # 步骤3: 最终惯性分 = 惯性质量 * 趋势方向
        inertia_score = (inertia_quality * adx_direction).clip(-1, 1)
        return inertia_score.astype(np.float32)

    def _diagnose_axiom_stability(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V2.2 · 稳健融合重构版】力学公理三：诊断“稳定性”
        - 核心重构: 废除脆弱的乘法融合模型，改为更稳健的加权平均模型。
        - 新逻辑: 将“波动率水平”和“波动率的稳定性”视为同等重要的两个独立证据，进行加权平均，
                   而不是相乘。这避免了因单一维度暂时表现不佳而完全否定整体稳定性的问题。
        """
        # 证据1: 波动率大小 (越小越好)，归一化到[0,1]
        bbw = df.get('BBW_21_2.0_D', pd.Series(0.0, index=df.index))
        atr_pct = df.get('ATR_14_D', pd.Series(0.0, index=df.index)) / df['close_D'].replace(0, np.nan)
        raw_volatility = (bbw + atr_pct).fillna(0)
        # ascending=False, 波动率越低，分数越高
        volatility_level_score = normalize_score(raw_volatility, df.index, norm_window, ascending=False)
        # 证据2: 波动率的稳定性 (波动率的波动率，越小越好)，归一化到[0,1]
        vol_instability_col = next((col for col in df.columns if col.startswith('VOLATILITY_INSTABILITY_INDEX_')), None)
        if vol_instability_col:
            vol_of_vol = df.get(vol_instability_col, pd.Series(0.0, index=df.index))
            # ascending=False, 波动率的波动率越低，分数越高
            volatility_stability_score = normalize_score(vol_of_vol, df.index, norm_window, ascending=False)
        else:
            volatility_stability_score = pd.Series(0.5, index=df.index)
        # 融合两大证据: 使用加权平均，而不是相乘
        raw_stability_score = (
            volatility_level_score * 0.6 +
            volatility_stability_score * 0.4
        ).clip(0, 1)
        # 转换为双极性分数：高稳定性为正，低稳定性为负
        stability_score = (raw_stability_score * 2 - 1).clip(-1, 1)
        return stability_score.astype(np.float32)

    def _diagnose_axiom_energy(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V1.1 · 稳健融合重构版】力学公理四：诊断“能量”
        - 核心重构: 废除脆弱的乘法融合模型，改为更稳健的加权平均模型。
        - 新逻辑: 将VPA效率和CMF资金流视为两个独立的能量来源，进行加权融合。
                   当两者方向一致时，能量共振增强；方向相反时，能量相互抵消。
        """
        # 证据1: VPA效率 (量价关系)
        vpa = df.get('VPA_EFFICIENCY_D', pd.Series(0.5, index=df.index))
        # 证据2: CMF (资金流量)
        cmf = df.get('CMF_21_D', pd.Series(0.0, index=df.index))
        # 分别归一化为双极性分数
        # VPA本身在[0,1]区间, 映射到[-1,1]
        vpa_bipolar = (vpa * 2 - 1).clip(-1, 1)
        # CMF本身是双极性指标，直接归一化
        cmf_bipolar = normalize_to_bipolar(cmf, df.index, norm_window)
        # 融合: 使用加权平均，而不是相乘
        energy_score = (
            vpa_bipolar * 0.5 +
            cmf_bipolar * 0.5
        ).clip(-1, 1)
        return energy_score.astype(np.float32)














