import pandas as pd
import numpy as np
from typing import Dict, Any
from strategies.trend_following.utils import get_params_block, get_param_value, get_adaptive_mtf_normalized_bipolar_score, get_adaptive_mtf_normalized_score, bipolar_to_exclusive_unipolar

class DynamicMechanicsEngine:
    def __init__(self, strategy_instance):
        """
        初始化动态力学引擎。
        :param strategy_instance: 策略主实例的引用，用于访问 df_indicators。
        """
        self.strategy = strategy_instance

    def _get_safe_series(self, df: pd.DataFrame, column_name: str, default_value: Any = 0.0, method_name: str = "未知方法") -> pd.Series:
        """
        安全地从DataFrame获取Series，如果不存在则打印警告并返回默认Series。
        """
        if column_name not in df.columns:
            print(f"    -> [力学情报警告] 方法 '{method_name}' 缺少数据 '{column_name}'，使用默认值 {default_value}。")
            return pd.Series(default_value, index=df.index)
        return df[column_name]

    def _validate_required_signals(self, df: pd.DataFrame, required_signals: list, method_name: str) -> bool:
        """
        【V1.0 · 战前情报校验】内部辅助方法，用于在方法执行前验证所有必需的数据信号是否存在。
        """
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            # 调整校验信息为“力学情报校验”
            print(f"    -> [力学情报校验] 方法 '{method_name}' 启动失败：缺少核心信号 {missing_signals}。")
            return False
        return True

    def run_dynamic_analysis_command(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V6.1 · 上下文修复版】动态力学引擎总指挥
        - 【修复】修复了 'TrendFollowStrategy' object has no attribute 'current_date' 的错误。
                  将当前日期的获取方式从 self.strategy.current_date 改为从传入的 df.index[-1] 获取，
                  解除了对策略主类属性的依赖，使引擎更加健壮。
        - 新增：获取探针日期，并将其作为上下文传递给各公理诊断方法。
        """
        p_conf = get_params_block(self.strategy, 'dynamic_mechanics_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            print("-> [指挥覆盖探针] 动态力学引擎在配置中被禁用，跳过分析。")
            return {}
        # --- 新增：获取探针配置 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates = debug_params.get('probe_dates', [])
        # --- 修改：从df的索引中获取当前日期，而不是从strategy对象 ---
        current_date_str = df.index[-1].strftime('%Y-%m-%d')
        # --- 修改结束 ---
        is_probe_day = current_date_str in probe_dates
        if is_probe_day:
            print(f"\n--- [力学情报探针] 激活 | 日期: {current_date_str} ---")
        all_dynamic_states = {}
        norm_window = get_param_value(p_conf.get('norm_window'), 55)
        # --- 修改：向下传递探针上下文 ---
        axiom_momentum = self._diagnose_axiom_momentum(df, norm_window, is_probe_day, current_date_str)
        axiom_inertia = self._diagnose_axiom_inertia(df, norm_window, is_probe_day, current_date_str)
        axiom_stability = self._diagnose_axiom_stability(df, norm_window, is_probe_day, current_date_str)
        axiom_energy = self._diagnose_axiom_energy(df, norm_window, is_probe_day, current_date_str)
        axiom_ma_dynamics = self._diagnose_axiom_ma_dynamics(df, norm_window) # 此方法暂不升级，保持原样
        axiom_divergence = self._diagnose_axiom_divergence(df, norm_window, axiom_momentum, axiom_inertia) # 修改：接收计算好的公理
        # --- 修改结束 ---
        all_dynamic_states['SCORE_DYN_AXIOM_DIVERGENCE'] = axiom_divergence
        all_dynamic_states['SCORE_DYN_AXIOM_MOMENTUM'] = axiom_momentum
        all_dynamic_states['SCORE_DYN_AXIOM_INERTIA'] = axiom_inertia
        all_dynamic_states['SCORE_DYN_AXIOM_STABILITY'] = axiom_stability
        all_dynamic_states['SCORE_DYN_AXIOM_ENERGY'] = axiom_energy
        all_dynamic_states['SCORE_DYN_AXIOM_MA_ACCELERATION'] = axiom_ma_dynamics
        bullish_divergence, bearish_divergence = bipolar_to_exclusive_unipolar(axiom_divergence)
        all_dynamic_states['SCORE_DYNAMIC_MECHANICS_BULLISH_DIVERGENCE'] = bullish_divergence.astype(np.float32)
        all_dynamic_states['SCORE_DYNAMIC_MECHANICS_BEARISH_DIVERGENCE'] = bearish_divergence.astype(np.float32)
        return all_dynamic_states

    def _diagnose_axiom_divergence(self, df: pd.DataFrame, norm_window: int, momentum_score: pd.Series, inertia_score: pd.Series) -> pd.Series:
        """
        【V1.2 · 性能优化版】力学公理五：诊断“力学背离”
        - 核心逻辑: 诊断价格动量与惯性之间的背离。
        - 【修改】直接接收已计算好的 momentum_score 和 inertia_score，避免重复计算。
        """
        # 【优化】力学背离本身就是一种关系，其归一化应该基于其自身的动态，而不是简单地使用单一窗口。
        # 这里直接使用计算出的双极性分数，不再进行额外的 normalize_to_bipolar，因为 momentum_score 和 inertia_score 已经是双极性。
        divergence_score = (inertia_score - momentum_score).clip(-1, 1)
        return divergence_score.astype(np.float32)

    def _diagnose_axiom_momentum(self, df: pd.DataFrame, norm_window: int, is_probe_day: bool, current_date_str: str) -> pd.Series:
        """
        【V2.2 · 负值惩罚版】力学公理一：诊断“动量品质”
        - 核心重构: 根据探针反馈，修正品质调节器的核心逻辑。移除对“主力信念”的 .clip(0) 操作，
                      允许负的信念值参与融合计算，从而对整体品质分形成“惩罚”效应。
                      这使得模型能更敏锐地识别主力信念动摇下的虚假动能。
        """
        p_conf_dyn = get_params_block(self.strategy, 'dynamic_mechanics_params', {}) # 获取力学引擎参数
        quality_weights = get_param_value(p_conf_dyn.get('momentum_quality_weights'), {'purity': 0.4, 'conviction': 0.4, 'vitality': 0.2})
        required_signals = [
            'ROC_12_D', 'MACDh_13_34_8_D',
            'upward_impulse_purity_D', 'main_force_conviction_index_D', 'trend_vitality_index_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_momentum"):
            return pd.Series(0.0, index=df.index)
        # 1. 计算原始动量
        roc = self._get_safe_series(df, 'ROC_12_D', 0.0, method_name="_diagnose_axiom_momentum")
        macd_h = self._get_safe_series(df, 'MACDh_13_34_8_D', 0.0, method_name="_diagnose_axiom_momentum")
        p_conf_bhv = get_params_block(self.strategy, 'behavioral_dynamics_params', {}) # 借用行为层的MTF权重配置
        p_mtf = get_param_value(p_conf_bhv.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        roc_score = get_adaptive_mtf_normalized_bipolar_score(roc, df.index, default_weights)
        macd_h_score = get_adaptive_mtf_normalized_bipolar_score(macd_h, df.index, default_weights)
        raw_momentum_score = (roc_score * 0.6 + macd_h_score * 0.4).clip(-1, 1)
        # 2. 计算动量品质调节器 [0, 1]
        raw_purity = self._get_safe_series(df, 'upward_impulse_purity_D', 0.5)
        raw_conviction = self._get_safe_series(df, 'main_force_conviction_index_D', 0.0)
        raw_vitality = self._get_safe_series(df, 'trend_vitality_index_D', 0.5)
        # 2.1 先将原始值加权融合
        composite_quality_raw = (
            raw_purity * quality_weights.get('purity', 0.4) +
            # --- 修改：移除 .clip(0)，让负信念参与计算以形成惩罚 ---
            raw_conviction * quality_weights.get('conviction', 0.4) +
            # --- 修改结束 ---
            raw_vitality * quality_weights.get('vitality', 0.2)
        )
        # 2.2 再对融合后的综合分进行归一化
        momentum_quality_modulator = get_adaptive_mtf_normalized_score(composite_quality_raw, df.index, True, default_weights).clip(0, 1)
        # 3. 最终动量品质分
        final_momentum_score = (raw_momentum_score * momentum_quality_modulator).clip(-1, 1)
        if is_probe_day:
            last_values = {
                "原料-ROC": roc.iloc[-1], "原料-MACDh": macd_h.iloc[-1],
                "品质-纯度": raw_purity.iloc[-1],
                "品质-信念": raw_conviction.iloc[-1],
                "品质-活力": raw_vitality.iloc[-1],
                "节点-原始动量分": raw_momentum_score.iloc[-1],
                "节点-品质综合分(原始)": composite_quality_raw.iloc[-1],
                "节点-品质调节器": momentum_quality_modulator.iloc[-1],
                "结果-最终动量分": final_momentum_score.iloc[-1]
            }
            print(f"  > 动量品质探针: { {k: round(v, 4) if isinstance(v, (int, float)) else v for k, v in last_values.items()} }")
        return final_momentum_score.astype(np.float32)

    def _diagnose_axiom_inertia(self, df: pd.DataFrame, norm_window: int, is_probe_day: bool, current_date_str: str) -> pd.Series:
        """
        【V3.1 · 融合归一版】力学公理二：诊断“结构化惯性”
        - 核心重构: 升级结构加固分的计算逻辑为“先融合，后归一”。先将均线排列和结构杠杆的原始值加权融合成一个
                      “综合结构加固原始分”，再对该综合分进行整体归一化，以保留二者间的相对量级。
        - 新增: 增加探针，用于监测关键计算节点。
        """
        p_conf_dyn = get_params_block(self.strategy, 'dynamic_mechanics_params', {})
        inertia_weights = get_param_value(p_conf_dyn.get('inertia_structural_weights'), {'base_inertia': 0.7, 'structural_reinforcement': 0.3})
        ma_col_base = 'EMA_55'
        timeframe_key = 'D'
        hurst_col = next((col for col in df.columns if col.startswith('hurst_')), 'hurst_144d_D')
        fractal_col = next((col for col in df.columns if col.startswith('FRACTAL_DIMENSION_')), 'FRACTAL_DIMENSION_100d_D')
        required_signals = [
            'ADX_14_D', hurst_col, fractal_col, f'MA_VELOCITY_{ma_col_base}_{timeframe_key}',
            f'MA_ACCELERATION_{ma_col_base}_{timeframe_key}', 'PDI_14_D', 'NDI_14_D',
            'trend_alignment_index_D', 'structural_leverage_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_inertia"):
            return pd.Series(0.0, index=df.index)
        p_conf_bhv = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_bhv.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        # 1. 计算基础惯性品质
        adx_strength = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'ADX_14_D', 0.0), df.index, True, default_weights)
        hurst = self._get_safe_series(df, hurst_col, 0.5).fillna(0.5)
        hurst_quality = get_adaptive_mtf_normalized_score(hurst, df.index, True, default_weights)
        fractal_dim = self._get_safe_series(df, fractal_col, 1.5).fillna(1.5)
        fractal_smoothness = get_adaptive_mtf_normalized_score(fractal_dim, df.index, False, default_weights)
        ma_velocity = get_adaptive_mtf_normalized_score(self._get_safe_series(df, f'MA_VELOCITY_{ma_col_base}_{timeframe_key}', 0.0), df.index, True, default_weights)
        ma_acceleration = get_adaptive_mtf_normalized_score(self._get_safe_series(df, f'MA_ACCELERATION_{ma_col_base}_{timeframe_key}', 0.0), df.index, True, default_weights)
        base_inertia_quality = (adx_strength * 0.3 + hurst_quality * 0.3 + fractal_smoothness * 0.1 + ma_velocity * 0.15 + ma_acceleration * 0.15).clip(0, 1)
        # --- 修改：采用“先融合，后归一” ---
        # 2. 计算结构加固分
        raw_alignment = self._get_safe_series(df, 'trend_alignment_index_D', 0.0)
        raw_leverage = self._get_safe_series(df, 'structural_leverage_D', 0.0)
        # 2.1 先将原始值加权融合
        composite_reinforcement_raw = (raw_alignment * 0.5 + raw_leverage * 0.5)
        # 2.2 再对融合后的综合分进行归一化
        structural_reinforcement_score = get_adaptive_mtf_normalized_bipolar_score(composite_reinforcement_raw, df.index, default_weights).clip(-1, 1)
        # --- 修改结束 ---
        # 3. 融合惯性品质与方向
        total_inertia_quality = (base_inertia_quality * inertia_weights.get('base_inertia', 0.7) + structural_reinforcement_score.clip(0, 1) * inertia_weights.get('structural_reinforcement', 0.3)).clip(0, 1)
        adx_direction = (self._get_safe_series(df, 'PDI_14_D', 0) > self._get_safe_series(df, 'NDI_14_D', 0)).astype(float) * 2 - 1
        final_inertia_score = (total_inertia_quality * adx_direction).clip(-1, 1)
        if is_probe_day:
            last_values = {
                "原料-ADX": self._get_safe_series(df, 'ADX_14_D').iloc[-1], "原料-Hurst": hurst.iloc[-1],
                "结构-排列": raw_alignment.iloc[-1],
                "结构-杠杆": raw_leverage.iloc[-1],
                "节点-基础惯性品质": base_inertia_quality.iloc[-1],
                "节点-结构加固分(原始)": composite_reinforcement_raw.iloc[-1],
                "节点-结构加固分": structural_reinforcement_score.iloc[-1],
                "节点-总惯性品质": total_inertia_quality.iloc[-1],
                "结果-最终惯性分": final_inertia_score.iloc[-1]
            }
            print(f"  > 结构化惯性探针: { {k: round(v, 4) if isinstance(v, (int, float)) else v for k, v in last_values.items()} }")
        return final_inertia_score.astype(np.float32)

    def _diagnose_axiom_stability(self, df: pd.DataFrame, norm_window: int, is_probe_day: bool, current_date_str: str) -> pd.Series:
        """
        【V3.1 · 逻辑修正版】力学公理三：诊断“势能稳定性”
        - 核心重构: 根据探针反馈，修正了原始稳定性分数的计算逻辑。之前高波动被错误地识别为高稳定。
                      新逻辑：1. 计算一个标准的波动率得分（越高越不稳定）。 2. 稳定性 = 1 - 波动率得分。
                      这确保了稳定性与波动率的负相关关系，使其更符合金融直觉。
        - 新增: 增加探针，用于监测关键计算节点。
        """
        p_conf_dyn = get_params_block(self.strategy, 'dynamic_mechanics_params', {})
        potential_weights = get_param_value(p_conf_dyn.get('stability_potential_weights'), {'ma_tension': 0.4, 'structural_tension': 0.4, 'breakout_readiness': 0.2})
        vol_instability_col = next((col for col in df.columns if col.startswith('VOLATILITY_INSTABILITY_INDEX_')), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        required_signals = [
            'BBW_21_2.0_D', 'ATR_14_D', 'close_D', vol_instability_col,
            'MA_POTENTIAL_TENSION_INDEX_D', 'structural_tension_index_D', 'breakout_readiness_score_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_stability"):
            return pd.Series(0.0, index=df.index)
        p_conf_bhv = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_bhv.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        # --- 修改：修正稳定性计算逻辑 ---
        # 1. 计算原始波动率
        bbw = self._get_safe_series(df, 'BBW_21_2.0_D', 0.0)
        atr_pct = self._get_safe_series(df, 'ATR_14_D', 0.0) / self._get_safe_series(df, 'close_D', 1e-9).replace(0, np.nan)
        raw_volatility = (bbw + atr_pct).fillna(0)
        # 2. 计算波动率得分 [0, 1]，越高代表波动越大
        volatility_level_score = get_adaptive_mtf_normalized_score(raw_volatility, df.index, ascending=True, tf_weights=default_weights)
        # 3. 计算原始稳定性得分 [0, 1]，越高代表越稳定 (与波动率得分反向)
        raw_stability_score = 1 - volatility_level_score
        # --- 修改结束 ---
        # 4. 计算势能指数
        ma_tension_score = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'MA_POTENTIAL_TENSION_INDEX_D', 0.5), df.index, True, default_weights)
        structural_tension_score = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'structural_tension_index_D', 0.5), df.index, True, default_weights)
        readiness_score = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'breakout_readiness_score_D', 0.5), df.index, True, default_weights)
        potential_energy_score = (
            ma_tension_score * potential_weights.get('ma_tension', 0.4) +
            structural_tension_score * potential_weights.get('structural_tension', 0.4) +
            readiness_score * potential_weights.get('breakout_readiness', 0.2)
        ).clip(0, 1)
        # 5. 最终势能稳定性分
        # 逻辑: (稳定性*2-1) 将 [0,1] 的稳定性映射到 [-1,1]。高稳定(1)得+1，低稳定(0)得-1。再乘以势能，实现调制。
        final_stability_score = ((raw_stability_score * 2 - 1) * potential_energy_score).clip(-1, 1)
        if is_probe_day:
            last_values = {
                "原料-BBW": bbw.iloc[-1], "原料-ATR%": atr_pct.iloc[-1],
                "势能-均线张力": self._get_safe_series(df, 'MA_POTENTIAL_TENSION_INDEX_D').iloc[-1],
                "势能-结构张力": self._get_safe_series(df, 'structural_tension_index_D').iloc[-1],
                "势能-准备度": self._get_safe_series(df, 'breakout_readiness_score_D').iloc[-1],
                "节点-波动率分(新)": volatility_level_score.iloc[-1],
                "节点-原始稳定分": raw_stability_score.iloc[-1],
                "节点-势能指数": potential_energy_score.iloc[-1],
                "结果-最终稳定分": final_stability_score.iloc[-1]
            }
            print(f"  > 势能稳定性探针: { {k: round(v, 4) if isinstance(v, (int, float)) else v for k, v in last_values.items()} }")
        return final_stability_score.astype(np.float32)

    def _diagnose_axiom_energy(self, df: pd.DataFrame, norm_window: int, is_probe_day: bool, current_date_str: str) -> pd.Series:
        """
        【V2.1 · 融合归一版】力学公理四：诊断“能量真实性”
        - 核心重构: 与其他公理统一，升级为“先融合，后归一”模式。先融合“资金可信度”和“主力流向性”的原始值，
                      计算出“综合能量品质原始分”，再对其进行归一化。最后，将“对倒强度”作为独立的惩罚因子，
                      对归一化后的品质分进行惩罚，逻辑更清晰，结果更可靠。
        """
        p_conf_dyn = get_params_block(self.strategy, 'dynamic_mechanics_params', {})
        quality_weights = get_param_value(p_conf_dyn.get('energy_quality_weights'), {'credibility': 0.5, 'directionality': 0.5}) # 调整权重
        required_signals = [
            'VPA_EFFICIENCY_D', 'CMF_21_D',
            'flow_credibility_index_D', 'main_force_flow_directionality_D', 'wash_trade_intensity_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_energy"):
            return pd.Series(0.0, index=df.index)
        # 1. 计算原始能量
        vpa = self._get_safe_series(df, 'VPA_EFFICIENCY_D', 0.5)
        cmf = self._get_safe_series(df, 'CMF_21_D', 0.0)
        vpa_bipolar = (vpa * 2 - 1).clip(-1, 1)
        p_conf_bhv = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_bhv.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        cmf_bipolar = get_adaptive_mtf_normalized_bipolar_score(cmf, df.index, default_weights)
        raw_energy_score = (vpa_bipolar * 0.5 + cmf_bipolar * 0.5).clip(-1, 1)
        # --- 修改：采用“先融合，后归一”并分离惩罚因子 ---
        # 2. 计算能量品质调节器
        raw_credibility = self._get_safe_series(df, 'flow_credibility_index_D', 0.5)
        raw_directionality = self._get_safe_series(df, 'main_force_flow_directionality_D', 0.5)
        raw_wash_trade = self._get_safe_series(df, 'wash_trade_intensity_D', 0.0)
        # 2.1 先融合核心品质因子的原始值
        composite_quality_raw = (
            raw_credibility * quality_weights.get('credibility', 0.5) +
            raw_directionality * quality_weights.get('directionality', 0.5)
        )
        # 2.2 对融合后的综合分进行归一化，得到基础品质分
        base_quality_score = get_adaptive_mtf_normalized_score(composite_quality_raw, df.index, True, default_weights)
        # 2.3 计算独立的惩罚因子
        wash_trade_penalty_factor = 1 - get_adaptive_mtf_normalized_score(raw_wash_trade, df.index, True, default_weights).clip(0, 1)
        # 2.4 最终调节器 = 基础品质分 * 惩罚因子
        energy_quality_modulator = (base_quality_score * wash_trade_penalty_factor).clip(0, 1)
        # --- 修改结束 ---
        # 3. 最终能量真实性分
        final_energy_score = (raw_energy_score * energy_quality_modulator).clip(-1, 1)
        if is_probe_day:
            last_values = {
                "原料-VPA": vpa.iloc[-1], "原料-CMF": cmf.iloc[-1],
                "品质-可信度": raw_credibility.iloc[-1],
                "品质-流向性": raw_directionality.iloc[-1],
                "品质-对倒": raw_wash_trade.iloc[-1],
                "节点-原始能量分": raw_energy_score.iloc[-1],
                "节点-品质综合分(原始)": composite_quality_raw.iloc[-1],
                "节点-惩罚因子": wash_trade_penalty_factor.iloc[-1],
                "节点-品质调节器": energy_quality_modulator.iloc[-1],
                "结果-最终能量分": final_energy_score.iloc[-1]
            }
            print(f"  > 能量真实性探针: { {k: round(v, 4) if isinstance(v, (int, float)) else v for k, v in last_values.items()} }")
        return final_energy_score.astype(np.float32)

    def _diagnose_axiom_ma_dynamics(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V1.4 · 信号校验增强版】力学公理六：诊断“均线动态”
        - 核心逻辑: 融合均线的速度和加速度，评估趋势的内在变化。
        - 【修复】修正了引用均线速度和加速度列名时，确保其与 `IndicatorService` 中 `merge_results` 方法添加后缀后的列名一致。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `velocity_score` 和 `acceleration_score` 的归一化方式改为多时间维度自适应归一化。
        """
        ma_col_base = 'EMA_55'
        timeframe_key = 'D'
        velocity_col = f'MA_VELOCITY_{ma_col_base}_{timeframe_key}'
        acceleration_col = f'MA_ACCELERATION_{ma_col_base}_{timeframe_key}'
        required_signals = [velocity_col, acceleration_col]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_ma_dynamics"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        velocity_raw = self._get_safe_series(df, velocity_col, 0.0, method_name="_diagnose_axiom_ma_dynamics")
        acceleration_raw = self._get_safe_series(df, acceleration_col, 0.0, method_name="_diagnose_axiom_ma_dynamics")
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {}) # 借用行为层的MTF权重配置
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        velocity_score = get_adaptive_mtf_normalized_bipolar_score(velocity_raw, df_index, default_weights)
        acceleration_score = get_adaptive_mtf_normalized_bipolar_score(acceleration_raw, df_index, default_weights)
        ma_dynamics_score = (velocity_score * 0.6 + acceleration_score * 0.4).clip(-1, 1)
        return ma_dynamics_score.astype(np.float32)

