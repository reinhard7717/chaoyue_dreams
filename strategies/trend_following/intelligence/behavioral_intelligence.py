# 文件: strategies/trend_following/intelligence/behavioral_intelligence.py
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Tuple, Optional, List, Any
from strategies.trend_following.utils import get_params_block, get_param_value, get_adaptive_mtf_normalized_score, get_adaptive_mtf_normalized_bipolar_score, bipolar_to_exclusive_unipolar, get_adaptive_mtf_normalized_score

class BehavioralIntelligence:
    """
    【V28.0 · 结构升维版】
    - 核心升级: 废弃了旧的 _calculate_price_health, _calculate_volume_health, _calculate_kline_pattern_health 方法。
                所有健康度计算已统一由全新的 _calculate_structural_behavior_health 引擎负责。
    """
    def __init__(self, strategy_instance):
        """
        初始化行为与模式识别模块。
        :param strategy_instance: 策略主实例的引用。
        """
        self.strategy = strategy_instance
        self.pattern_recognizer = strategy_instance.pattern_recognizer

    def _get_safe_series(self, df: pd.DataFrame, column_name: str, default_value: Any = 0.0, method_name: str = "未知方法") -> pd.Series:
        """
        安全地从DataFrame获取Series，如果不存在则打印警告并返回默认Series。
        """
        if column_name not in df.columns:
            print(f"    -> [行为情报警告] 方法 '{method_name}' 缺少数据 '{column_name}'，使用默认值 {default_value}。")
            return pd.Series(default_value, index=df.index)
        return df[column_name]

    def _validate_required_signals(self, df: pd.DataFrame, required_signals: list, method_name: str) -> bool:
        """
        【V1.0 · 战前情报校验】内部辅助方法，用于在方法执行前验证所有必需的数据信号是否存在。
        """
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            print(f"    -> [行为情报校验] 方法 '{method_name}' 启动失败：缺少核心信号 {missing_signals}。")
            return False
        return True

    def run_behavioral_analysis_command(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V29.0 · 结构指标升维版】行为情报模块总指挥
        - 核心升级: 新增对 _diagnose_microstructure_intent 方法的调用，引入微观结构意图信号。
        """
        all_behavioral_states = {}
        atomic_signals = self._diagnose_behavioral_axioms(df)
        self.strategy.atomic_states.update(atomic_signals)
        all_behavioral_states.update(atomic_signals)
        # [新增代码块] 调用微观结构意图诊断
        micro_intent_signals = self._diagnose_microstructure_intent(df)
        self.strategy.atomic_states.update(micro_intent_signals)
        all_behavioral_states.update(micro_intent_signals)
        context_new_high_strength = self._diagnose_context_new_high_strength(df)
        self.strategy.atomic_states.update(context_new_high_strength)
        all_behavioral_states.update(context_new_high_strength)
        bullish_divergence, bearish_divergence = bipolar_to_exclusive_unipolar(atomic_signals.get('SCORE_BEHAVIOR_PRICE_VS_VOLUME_DIVERGENCE', pd.Series(0.0, index=df.index)))
        all_behavioral_states['SCORE_BEHAVIOR_BULLISH_DIVERGENCE'] = bullish_divergence.astype(np.float32)
        all_behavioral_states['SCORE_BEHAVIOR_BEARISH_DIVERGENCE'] = bearish_divergence.astype(np.float32)
        for k, v in atomic_signals.items():
            if k not in df.columns:
                df[k] = v
        df_with_dynamics = self._calculate_signal_dynamics(df)
        dynamic_cols = [c for c in df_with_dynamics.columns if c.startswith(('MOMENTUM_', 'POTENTIAL_', 'THRUST_', 'RESONANCE_'))]
        self.strategy.atomic_states.update(df_with_dynamics[dynamic_cols])
        all_behavioral_states.update(df_with_dynamics[dynamic_cols])
        return all_behavioral_states

    def _get_atomic_score(self, df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
        """
        【V1.0】安全地从原子状态库或主数据帧中获取分数。
        - 核心职责: 统一信号获取路径，优先从 self.strategy.atomic_states 获取，
                      若无则从主数据帧 df 获取，最后提供默认值，确保数据流的稳定性。
        """
        if name in self.strategy.atomic_states:
            return self.strategy.atomic_states[name]
        elif name in df.columns:
            return df[name]
        else:
            print(f"     -> [行为情报引擎警告] 信号 '{name}' 不存在，使用默认值 {default}。")
            return pd.Series(default, index=df.index)

    def _get_signal(self, df: pd.DataFrame, signal_name: str, default_value: float = 0.0) -> pd.Series:
        """
        【V1.0】信号获取哨兵方法
        - 核心职责: 安全地从DataFrame获取信号。
        - 预警机制: 如果信号不存在，打印明确的警告信息，并返回一个包含默认值的Series，以防止程序崩溃。
        """
        if signal_name not in df.columns:
            print(f"    -> [行为情报引擎警告] 依赖信号 '{signal_name}' 在数据帧中不存在，将使用默认值 {default_value}。")
            return pd.Series(default_value, index=df.index)
        return df[signal_name]

    def _generate_all_atomic_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.0 · 职责净化版】原子信号中心
        - 核心升级: 遵循“三层金字塔”架构，本方法不再计算跨领域的“趋势健康度”和“绝望度”。
                      这些高级融合逻辑已迁移至 FusionIntelligence。
                      新增对纯净版“行为K线质量分”的计算和发布。
        """
        atomic_signals = {}
        atomic_signals.update(self._diagnose_behavioral_axioms(df))
        day_quality_score = self._calculate_behavioral_day_quality(df)
        atomic_signals['BIPOLAR_BEHAVIORAL_DAY_QUALITY'] = day_quality_score
        battlefield_momentum = day_quality_score.ewm(span=5, adjust=False).mean()
        atomic_signals['SCORE_BEHAVIORAL_BATTLEFIELD_MOMENTUM'] = battlefield_momentum.astype(np.float32)
        self.strategy.atomic_states.update(atomic_signals)
        atomic_signals.update(self._diagnose_upper_shadow_intent(df))
        return atomic_signals

    def _calculate_signal_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V4.3 · 上涨衰竭动态增强与多时间维度归一化版】信号动态计算引擎
        - 核心错误修复: 彻底剥离了对其他情报层终极共振信号的依赖，解决了因执行时序错乱导致的信号获取失败问题。
        - 核心逻辑重构: 遵循“职责分离”原则，本方法现在只聚焦于为【本模块生产的】纯粹行为原子信号注入动态因子（动量、潜力、推力）。
                        不再计算跨领域的 RESONANCE_HEALTH_D 等信号。
        - 【修改】移除对 `SCORE_BEHAVIOR_RISK_UPPER_SHADOW_PRESSURE` 的动态增强。
        - 【优化】将 `momentum`, `potential`, `thrust` 的归一化方式改为多时间维度自适应归一化。
        """
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_dyn = get_param_value(p_conf.get('signal_dynamics_params'), {})
        momentum_span = get_param_value(p_dyn.get('momentum_span'), 5)
        potential_window = get_param_value(p_dyn.get('potential_window'), 120)
        dynamics_df = pd.DataFrame(index=df.index)
        atomic_signals_to_enhance = [
            'SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM',
            'SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM',
            'SCORE_BEHAVIOR_VOLUME_BURST',
            'SCORE_BEHAVIOR_VOLUME_ATROPHY',
            'SCORE_BEHAVIOR_UPWARD_EFFICIENCY',
            'SCORE_BEHAVIOR_DOWNWARD_RESISTANCE',
            'SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL',
            'SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION',
            'SCORE_OPPORTUNITY_LOCKUP_RALLY',
            'SCORE_OPPORTUNITY_SELLING_EXHAUSTION',
            'INTERNAL_BEHAVIOR_STAGNATION_EVIDENCE_RAW', # 新增上涨衰竭原始分的动态增强
            'SCORE_RISK_LIQUIDITY_DRAIN'
        ]
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        for signal_name in atomic_signals_to_enhance:
            if signal_name in self.strategy.atomic_states:
                signal_series = self.strategy.atomic_states[signal_name]
                momentum = signal_series.diff(momentum_span).fillna(0)
                # 【优化】使用多时间维度自适应归一化
                norm_momentum = get_adaptive_mtf_normalized_score(momentum, df.index, ascending=True, tf_weights=default_weights)
                dynamics_df[f'MOMENTUM_{signal_name}'] = norm_momentum.astype(np.float32)
                potential = signal_series.rolling(window=potential_window).mean().fillna(signal_series)
                # 【优化】使用多时间维度自适应归一化
                norm_potential = get_adaptive_mtf_normalized_score(potential, df.index, ascending=True, tf_weights=default_weights)
                dynamics_df[f'POTENTIAL_{signal_name}'] = norm_potential.astype(np.float32)
                thrust = momentum.diff(1).fillna(0)
                # 【优化】使用多时间维度自适应归一化
                norm_thrust = get_adaptive_mtf_normalized_score(thrust, df.index, ascending=True, tf_weights=default_weights)
                dynamics_df[f'THRUST_{signal_name}'] = norm_thrust.astype(np.float32)
            else:
                print(f"     - [警告] 信号 '{signal_name}' 在原子状态库中不存在，跳过动态因子计算。")
        final_df = pd.concat([df, dynamics_df], axis=1)
        return final_df

    def _calculate_behavioral_day_quality(self, df: pd.DataFrame) -> pd.Series:
        """
        【V1.2 · 信号校验增强版】行为K线质量分计算引擎
        - 核心修改: 调用从 utils.py 导入的公共归一化工具。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - [新增] 在方法入口处添加信号校验逻辑。
        """
        required_signals = [
            'closing_price_deviation_score_D', 'real_body_vs_range_ratio_D', 'shadow_dominance_D',
            'VPA_EFFICIENCY_D', 'vwap_control_strength_D', 'intraday_trend_purity_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_calculate_behavioral_day_quality"):
            return pd.Series(0.0, index=df.index)
        print("开始执行【V1.0 · 纯净版】行为K线质量分计算...")
        outcome_core = (self._get_safe_series(df, 'closing_price_deviation_score_D', 0.5, method_name="_calculate_behavioral_day_quality") * 2 - 1).clip(-1, 1)
        body_dominance = self._get_safe_series(df, 'real_body_vs_range_ratio_D', 0.0, method_name="_calculate_behavioral_day_quality")
        shadow_dominance = self._get_safe_series(df, 'shadow_dominance_D', 0.0, method_name="_calculate_behavioral_day_quality")
        pillar1_outcome_score = (outcome_core * 0.7 + outcome_core * body_dominance * 0.1 + shadow_dominance * 0.2).clip(-1, 1)
        vpa_eff_bipolar = get_adaptive_mtf_normalized_bipolar_score(self._get_safe_series(df, 'VPA_EFFICIENCY_D', pd.Series(0.0, index=df.index), method_name="_calculate_behavioral_day_quality"), df.index)
        vwap_ctrl_bipolar = get_adaptive_mtf_normalized_bipolar_score(self._get_safe_series(df, 'vwap_control_strength_D', pd.Series(0.0, index=df.index), method_name="_calculate_behavioral_day_quality"), df.index)
        trend_purity_bipolar = get_adaptive_mtf_normalized_bipolar_score(self._get_safe_series(df, 'intraday_trend_purity_D', pd.Series(0.0, index=df.index), method_name="_calculate_behavioral_day_quality"), df.index)
        bullish_execution = (((vpa_eff_bipolar + 1)/2) * ((vwap_ctrl_bipolar + 1)/2) * ((trend_purity_bipolar + 1)/2)).pow(1/3)
        pillar2_execution_score = (bullish_execution * 2 - 1).clip(-1, 1)
        day_quality_score = (
            pillar1_outcome_score * 0.4 +
            pillar2_execution_score * 0.6
        ).clip(-1, 1)
        print("【纯净版行为K线质量分】计算完成。")
        return day_quality_score.astype(np.float32)

    def _diagnose_behavioral_axioms(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V29.2 · 信号校验增强版】原子信号中心
        - 核心升级: 引入 microstructure_efficiency_index, dip_absorption_power, rally_distribution_pressure 等高级结构指标，
                    对上涨效率、下跌抵抗等核心行为公理进行高保真重构。
        - [新增] 在方法入口处添加信号校验逻辑。
        """
        required_signals = [
            'pct_change_D', 'BIAS_55_D', 'volume_ratio_D', 'microstructure_efficiency_index_D',
            'dip_absorption_power_D', 'vwap_control_strength_D', 'lower_shadow_absorption_strength_D',
            'upper_shadow_selling_pressure_D', 'volume_D', 'rally_distribution_pressure_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_behavioral_axioms"):
            return {}
        states = {}
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        long_term_weights = get_param_value(p_mtf.get('long_term_weights'), {'weights': {21: 0.5, 55: 0.3, 89: 0.2}})
        states['SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM'] = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'pct_change_D', method_name="_diagnose_behavioral_axioms"), df.index, ascending=True, tf_weights=default_weights).astype(np.float32)
        states['SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM'] = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'pct_change_D', method_name="_diagnose_behavioral_axioms").clip(upper=0).abs(), df.index, ascending=True, tf_weights=default_weights).astype(np.float32)
        states['INTERNAL_BEHAVIOR_PRICE_OVEREXTENSION_RAW'] = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'BIAS_55_D', 0.0, method_name="_diagnose_behavioral_axioms"), df.index, ascending=True, tf_weights=long_term_weights).astype(np.float32)
        states['SCORE_BEHAVIOR_VOLUME_BURST'] = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'volume_ratio_D', 1.0, method_name="_diagnose_behavioral_axioms"), df.index, ascending=True, tf_weights=default_weights).astype(np.float32)
        states['SCORE_BEHAVIOR_VOLUME_ATROPHY'] = self._calculate_volume_atrophy(df, default_weights).astype(np.float32)
        upward_efficiency_raw = self._get_safe_series(df, 'microstructure_efficiency_index_D', 0.0, method_name="_diagnose_behavioral_axioms")
        states['SCORE_BEHAVIOR_UPWARD_EFFICIENCY'] = get_adaptive_mtf_normalized_score(upward_efficiency_raw, df.index, ascending=True, tf_weights=default_weights).astype(np.float32)
        downward_resistance_raw = self._get_safe_series(df, 'dip_absorption_power_D', 0.0, method_name="_diagnose_behavioral_axioms")
        states['SCORE_BEHAVIOR_DOWNWARD_RESISTANCE'] = get_adaptive_mtf_normalized_score(downward_resistance_raw, df.index, ascending=True, tf_weights=default_weights).astype(np.float32)
        states['SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL'] = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'vwap_control_strength_D', 0.5, method_name="_diagnose_behavioral_axioms"), df.index, ascending=True, tf_weights=default_weights).astype(np.float32)
        states['SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION'] = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'lower_shadow_absorption_strength_D', 0.0, method_name="_diagnose_behavioral_axioms"), df.index, ascending=True, tf_weights=default_weights).astype(np.float32)
        states['INTERNAL_BEHAVIOR_UPPER_SHADOW_RAW'] = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'upper_shadow_selling_pressure_D', 0.0, method_name="_diagnose_behavioral_axioms"), df.index, ascending=True, tf_weights=default_weights).astype(np.float32)
        price_trend = get_adaptive_mtf_normalized_bipolar_score(self._get_safe_series(df, 'pct_change_D', method_name="_diagnose_behavioral_axioms"), df.index, default_weights)
        volume_trend = get_adaptive_mtf_normalized_bipolar_score(self._get_safe_series(df, 'volume_D', method_name="_diagnose_behavioral_axioms").diff(1), df.index, default_weights)
        divergence_score = (volume_trend - price_trend).clip(-1, 1)
        states['SCORE_BEHAVIOR_PRICE_VS_VOLUME_DIVERGENCE'] = divergence_score.astype(np.float32)
        is_rising = (self._get_safe_series(df, 'pct_change_D', method_name="_diagnose_behavioral_axioms") > 0).astype(float)
        is_falling = (self._get_safe_series(df, 'pct_change_D', method_name="_diagnose_behavioral_axioms") < 0).astype(float)
        states['SCORE_OPPORTUNITY_LOCKUP_RALLY'] = (is_rising * states['SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM'] * states['SCORE_BEHAVIOR_VOLUME_ATROPHY']).pow(1/3).astype(np.float32)
        states['SCORE_OPPORTUNITY_SELLING_EXHAUSTION'] = (is_falling * states['SCORE_BEHAVIOR_VOLUME_ATROPHY'] * states['SCORE_BEHAVIOR_DOWNWARD_RESISTANCE']).pow(1/3).astype(np.float32)
        evidence_components = [
            states['SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM'].clip(0, 1),
            (1 - states['SCORE_BEHAVIOR_UPWARD_EFFICIENCY']).clip(0, 1),
            states['SCORE_BEHAVIOR_VOLUME_BURST'].clip(0, 1),
            (1 - states['SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL']).clip(0, 1)
        ]
        weights_stagnation_evidence = np.array([0.2, 0.3, 0.3, 0.2])
        aligned_evidence_components = [comp.reindex(df.index, fill_value=0.0) for comp in evidence_components]
        safe_evidence_components = [comp + 1e-9 for comp in aligned_evidence_components]
        stagnation_evidence_raw_score = pd.Series(np.prod([comp.values ** w for comp, w in zip(safe_evidence_components, weights_stagnation_evidence)], axis=0), index=df.index)
        states['INTERNAL_BEHAVIOR_STAGNATION_EVIDENCE_RAW'] = (stagnation_evidence_raw_score * is_rising).clip(0, 1).astype(np.float32)
        states['SCORE_RISK_LIQUIDITY_DRAIN'] = (is_falling * states['SCORE_BEHAVIOR_VOLUME_BURST'] * states['SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM']).pow(1/2).astype(np.float32)
        distribution_pressure_raw = self._get_safe_series(df, 'rally_distribution_pressure_D', 0.0, method_name="_diagnose_behavioral_axioms")
        states['SCORE_BEHAVIOR_DISTRIBUTION_PRESSURE'] = get_adaptive_mtf_normalized_score(distribution_pressure_raw, df.index, ascending=True, tf_weights=default_weights).astype(np.float32)
        return states

    def _calculate_volume_atrophy(self, df: pd.DataFrame, tf_weights: Dict) -> pd.Series:
        """
        【V1.1 · 量能萎缩证据增强版】计算 SCORE_BEHAVIOR_VOLUME_ATROPHY 信号。
        - 核心逻辑: 融合三种量能萎缩的强力证据，并引入成交量均线空头排列和萎缩持续性。
          1. 量能一直低于ma_vol_21 (持续性萎缩)
          2. ma_vol_5已经低于ma_vol_21 (结构性萎缩)
          3. 当日量能 < ma_vol_5 < ma_vol_21 (冰点萎缩)
          4. 成交量均线空头排列 (VOL_MA_5_D < VOL_MA_13_D < VOL_MA_21_D)
        - 输出: [0, 1] 的单极性分数，分数越高代表量能萎缩越严重。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        """
        df_index = df.index
        # 获取必要的成交量均线数据
        vol = self._get_safe_series(df, 'volume_D', 0.0, method_name="_calculate_volume_atrophy")
        vol_ma5 = self._get_safe_series(df, 'VOL_MA_5_D', 0.0, method_name="_calculate_volume_atrophy")
        vol_ma13 = self._get_safe_series(df, 'VOL_MA_13_D', 0.0, method_name="_calculate_volume_atrophy") # 新增13日均量
        vol_ma21 = self._get_safe_series(df, 'VOL_MA_21_D', 0.0, method_name="_calculate_volume_atrophy")
        # 确保数据存在，否则返回默认值
        if vol.isnull().all() or vol_ma5.isnull().all() or vol_ma13.isnull().all() or vol_ma21.isnull().all():
            print("    -> [行为情报引擎警告] 缺少成交量或成交量均线数据，无法计算 SCORE_BEHAVIOR_VOLUME_ATROPHY。")
            return pd.Series(0.0, index=df_index)
        # 避免除以零
        vol_ma5_safe = vol_ma5.replace(0, 1e-9)
        vol_ma13_safe = vol_ma13.replace(0, 1e-9)
        vol_ma21_safe = vol_ma21.replace(0, 1e-9)
        # 证据1: 量能一直低于ma_vol_21 (持续性萎缩)
        # 使用 (1 - volume_D / VOL_MA_21_D) 衡量萎缩程度，比值越小，分数越高
        vol_below_ma21_raw = (1 - (vol / vol_ma21_safe)).clip(0, 1)
        vol_below_ma21_score = get_adaptive_mtf_normalized_score(vol_below_ma21_raw, df_index, ascending=True, tf_weights=tf_weights)
        # 证据2: ma_vol_5已经低于ma_vol_21 (结构性萎缩)
        # 使用 (1 - VOL_MA_5_D / VOL_MA_21_D) 衡量萎缩程度
        vol_ma5_below_ma21_raw = (1 - (vol_ma5 / vol_ma21_safe)).clip(0, 1)
        vol_ma5_below_ma21_score = get_adaptive_mtf_normalized_score(vol_ma5_below_ma21_raw, df_index, ascending=True, tf_weights=tf_weights)
        # 证据3: 当日量能 < ma_vol_5 < ma_vol_21 (冰点萎缩)
        # 这是一个布尔条件，直接转换为分数，并可以考虑其持续性
        is_ice_point_atrophy = ((vol < vol_ma5) & (vol_ma5 < vol_ma21)).astype(float)
        # 冰点萎缩的持续性
        ice_point_atrophy_persistence = is_ice_point_atrophy.rolling(window=5, min_periods=1).sum() / 5
        ice_point_atrophy_score = is_ice_point_atrophy * (0.5 + ice_point_atrophy_persistence * 0.5) # 持续越久，分数越高
        # 证据4: 成交量均线空头排列 (VOL_MA_5_D < VOL_MA_13_D < VOL_MA_21_D)
        # 这是一个结构性萎缩的强力证据
        is_ma_bearish_alignment = ((vol_ma5 < vol_ma13) & (vol_ma13 < vol_ma21)).astype(float)
        # 均线空头排列的强度，可以根据均线之间的距离来衡量
        ma_bearish_alignment_strength_raw = (vol_ma21 - vol_ma5) / vol_ma21_safe.replace(0, np.nan)
        ma_bearish_alignment_score = get_adaptive_mtf_normalized_score(ma_bearish_alignment_strength_raw.clip(lower=0), df_index, ascending=True, tf_weights=tf_weights)
        # 融合布尔条件和强度
        ma_bearish_alignment_final_score = is_ma_bearish_alignment * ma_bearish_alignment_score
        # 融合所有证据
        # 赋予冰点萎缩和均线空头排列更高的权重，因为它代表了最强的萎缩信号
        # 使用加权几何平均，确保所有证据都存在时分数才高
        evidence_components = [
            vol_below_ma21_score,
            vol_ma5_below_ma21_score,
            ice_point_atrophy_score, # 使用增强后的冰点萎缩分数
            ma_bearish_alignment_final_score # 新增均线空头排列分数
        ]
        # 调整权重，冰点萎缩和均线空头排列权重更高
        weights = np.array([0.2, 0.2, 0.3, 0.3]) # 调整权重
        aligned_evidence_components = [comp.reindex(df_index, fill_value=0.0) for comp in evidence_components]
        safe_evidence_components = [comp + 1e-9 for comp in aligned_evidence_components] # 避免log(0)
        volume_atrophy_score = pd.Series(np.prod([comp.values ** w for comp, w in zip(safe_evidence_components, weights)], axis=0), index=df_index)
        return volume_atrophy_score.clip(0, 1)

    def _diagnose_context_new_high_strength(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.2 · 信号校验增强版】诊断内部上下文信号：新高强度 (CONTEXT_NEW_HIGH_STRENGTH)
        - 核心逻辑: 融合价格突破、均线斜率和BIAS健康度，评估新高的综合质量。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将所有组成信号的归一化方式改为多时间维度自适应归一化。
        - [新增] 在方法入口处添加信号校验逻辑。
        """
        required_signals = ['pct_change_D', 'SLOPE_5_EMA_55_D', 'BIAS_55_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_context_new_high_strength"):
            return {}
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        long_term_weights = get_param_value(p_mtf.get('long_term_weights'), {'weights': {21: 0.5, 55: 0.3, 89: 0.2}})
        price_breakthrough_score = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'pct_change_D', method_name="_diagnose_context_new_high_strength").clip(lower=0), df.index, ascending=True, tf_weights=default_weights)
        ma_slope_score = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'SLOPE_5_EMA_55_D', pd.Series(0.0, index=df.index), method_name="_diagnose_context_new_high_strength"), df.index, ascending=True, tf_weights=default_weights)
        bias_health_score = 1 - get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'BIAS_55_D', pd.Series(0.0, index=df.index), method_name="_diagnose_context_new_high_strength").clip(lower=0), df.index, ascending=True, tf_weights=long_term_weights)
        new_high_strength = (price_breakthrough_score * ma_slope_score * bias_health_score).pow(1/3).fillna(0.0)
        return {'CONTEXT_NEW_HIGH_STRENGTH': new_high_strength.astype(np.float32)}

    def _resolve_pressure_absorption_dynamics(self, provisional_pressure: pd.Series, intent_diagnosis: pd.Series) -> Dict[str, pd.Series]:
        """
        【V3.3 · 情报校验加固版】压力-承接能量转化模型
        - 核心修改: 调用从 utils.py 导入的公共归一化工具。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `absorption_efficiency` 和 `absorption_control` 的归一化方式改为多时间维度自适应归一化。
        - 【新增】增加战前情报校验，确保所有依赖信号存在。
        """
        states = {}
        df = self.strategy.df_indicators
        # [新增代码块] 战前情报校验
        required_signals = ['VPA_EFFICIENCY_D', 'vwap_control_strength_D']
        if not self._validate_required_signals(df, required_signals, "_resolve_pressure_absorption_dynamics"):
            return {
                'SCORE_RISK_UNRESOLVED_PRESSURE': pd.Series(0.0, index=df.index),
                'SCORE_OPPORTUNITY_PRESSURE_ABSORPTION': pd.Series(0.0, index=df.index)
            }
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        absorption_efficiency = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'VPA_EFFICIENCY_D', pd.Series(0.5, index=df.index), method_name="_resolve_pressure_absorption_dynamics"), df.index, ascending=True, tf_weights=default_weights)
        absorption_control = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'vwap_control_strength_D', pd.Series(0.5, index=df.index), method_name="_resolve_pressure_absorption_dynamics"), df.index, ascending=True, tf_weights=default_weights)
        absorption_intent_factor = (intent_diagnosis.clip(-1, 1) + 1) / 2.0
        absorption_quality_score = (absorption_efficiency * absorption_control * absorption_intent_factor).pow(1/3)
        daily_net_force = absorption_quality_score - provisional_pressure
        battlefield_momentum_score = daily_net_force.ewm(span=3, adjust=False).mean().fillna(0)
        base_risk = provisional_pressure * (1.0 - absorption_quality_score)
        risk_amplifier = 1.0 - battlefield_momentum_score.clip(upper=0)
        final_risk_score = (base_risk * risk_amplifier).clip(0, 1)
        base_opportunity = provisional_pressure * absorption_quality_score
        opportunity_amplifier = 1.0 + battlefield_momentum_score.clip(lower=0)
        trend_health = self.strategy.atomic_states.get('SCORE_TREND_HEALTH', pd.Series(0.5, index=df.index))
        context_modulator = 1.0 + trend_health * 0.5
        final_opportunity_score = (base_opportunity * opportunity_amplifier * context_modulator).clip(0, 1)
        states['SCORE_RISK_UNRESOLVED_PRESSURE'] = final_risk_score.astype(np.float32)
        states['SCORE_OPPORTUNITY_PRESSURE_ABSORPTION'] = final_opportunity_score.astype(np.float32)
        return states

    def _diagnose_microstructure_intent(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.2 · 信号校验增强版】微观结构意图诊断引擎
        - 核心职责: 融合订单流失衡(OFI)与扫单强度，生成一个全新的原子信号 `SCORE_BEHAVIOR_MICROSTRUCTURE_INTENT`，
                    用于捕捉最细微的主力攻击或撤退意图。
        - 数学思想: OFI反映了挂单册上的力量变化，扫单强度反映了直接吃单的决心。两者结合，可以区分是“温和吸筹”还是“暴力抢筹”。
        - [新增] 在方法入口处添加信号校验逻辑。
        """
        required_signals = ['order_book_imbalance_D', 'buy_quote_exhaustion_rate_D', 'sell_quote_exhaustion_rate_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_microstructure_intent"):
            return {}
        states = {}
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        ofi_raw = self._get_safe_series(df, 'order_book_imbalance_D', 0.0, method_name="_diagnose_microstructure_intent")
        buy_sweep_raw = self._get_safe_series(df, 'buy_quote_exhaustion_rate_D', 0.0, method_name="_diagnose_microstructure_intent")
        sell_sweep_raw = self._get_safe_series(df, 'sell_quote_exhaustion_rate_D', 0.0, method_name="_diagnose_microstructure_intent")
        ofi_score = get_adaptive_mtf_normalized_bipolar_score(ofi_raw, df.index, default_weights)
        buy_sweep_score = get_adaptive_mtf_normalized_score(buy_sweep_raw, df.index, ascending=True, tf_weights=default_weights)
        sell_sweep_score = get_adaptive_mtf_normalized_score(sell_sweep_raw, df.index, ascending=True, tf_weights=default_weights)
        bullish_intent = (ofi_score.clip(lower=0) * 0.5 + buy_sweep_score * 0.5)
        bearish_intent = (ofi_score.clip(upper=0).abs() * 0.5 + sell_sweep_score * 0.5)
        micro_intent_score = (bullish_intent - bearish_intent).clip(-1, 1)
        states['SCORE_BEHAVIOR_MICROSTRUCTURE_INTENT'] = micro_intent_score.astype(np.float32)
        return states



