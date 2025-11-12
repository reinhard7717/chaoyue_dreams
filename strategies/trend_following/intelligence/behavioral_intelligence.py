# 文件: strategies/trend_following/intelligence/behavioral_intelligence.py
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Tuple, Optional, List
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_to_bipolar, normalize_score, get_adaptive_mtf_normalized_score, get_adaptive_mtf_normalized_bipolar_score, bipolar_to_exclusive_unipolar

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

    def run_behavioral_analysis_command(self) -> Dict[str, pd.Series]:
        """
        【V5.6 · 原始信号纯粹版】行为情报模块总指挥
        - 核心升级: 废弃原子层面的“共振”和“领域健康度”信号。
        - 核心职责: 只输出行为领域的原子公理信号、行为背离信号和上下文信号。
        - 【修改】将上影线抛压风险的判断逻辑迁移至融合层，行为层只输出原始信号。
        - 移除信号: SCORE_BEHAVIOR_BULLISH_RESONANCE, SCORE_BEHAVIOR_BEARISH_RESONANCE, BIPOLAR_BEHAVIORAL_DOMAIN_HEALTH, SCORE_BEHAVIOR_BOTTOM_REVERSAL, SCORE_BEHAVIOR_TOP_REVERSAL。
        """
        df = self.strategy.df_indicators
        all_behavioral_states = {}
        atomic_signals = self._diagnose_behavioral_axioms(df)
        self.strategy.atomic_states.update(atomic_signals)
        all_behavioral_states.update(atomic_signals)
        context_new_high_strength = self._diagnose_context_new_high_strength(df)
        self.strategy.atomic_states.update(context_new_high_strength)
        all_behavioral_states.update(context_new_high_strength)
        # 引入行为层面的看涨/看跌背离信号 (保持不变)
        bullish_divergence, bearish_divergence = bipolar_to_exclusive_unipolar(atomic_signals.get('SCORE_BEHAVIOR_PRICE_VS_VOLUME_DIVERGENCE', pd.Series(0.0, index=df.index)))
        all_behavioral_states['SCORE_BEHAVIOR_BULLISH_DIVERGENCE'] = bullish_divergence.astype(np.float32)
        all_behavioral_states['SCORE_BEHAVIOR_BEARISH_DIVERGENCE'] = bearish_divergence.astype(np.float32)
        # 【移除行】诊断深度博弈版的上影线抛压风险，现在由融合层处理
        # 【移除行】upper_shadow_risk = self._diagnose_upper_shadow_pressure_risk(df)
        # 【移除行】self.strategy.atomic_states.update(upper_shadow_risk)
        # 【移除行】all_behavioral_states.update(upper_shadow_risk)
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
        【V4.2 · 上涨衰竭动态增强版】信号动态计算引擎
        - 核心错误修复: 彻底剥离了对其他情报层终极共振信号的依赖，解决了因执行时序错乱导致的信号获取失败问题。
        - 核心逻辑重构: 遵循“职责分离”原则，本方法现在只聚焦于为【本模块生产的】纯粹行为原子信号注入动态因子（动量、潜力、推力）。
                        不再计算跨领域的 RESONANCE_HEALTH_D 等信号。
        - 【修改】移除对 `SCORE_BEHAVIOR_RISK_UPPER_SHADOW_PRESSURE` 的动态增强。
        - 【新增】为 `INTERNAL_BEHAVIOR_RALLY_EXHAUSTION_RAW` 提供动态增强。
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
            'SCORE_BEHAVIOR_VOLUME_APATHY',
            'SCORE_BEHAVIOR_UPWARD_EFFICIENCY',
            'SCORE_BEHAVIOR_DOWNWARD_RESISTANCE',
            'SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL',
            'SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION',
            'SCORE_OPPORTUNITY_LOCKUP_RALLY',
            'SCORE_OPPORTUNITY_SELLING_EXHAUSTION',
            'INTERNAL_BEHAVIOR_RALLY_EXHAUSTION_RAW', # 修改行: 新增上涨衰竭原始分的动态增强
            'SCORE_RISK_LIQUIDITY_DRAIN'
        ]
        for signal_name in atomic_signals_to_enhance:
            if signal_name in self.strategy.atomic_states:
                signal_series = self.strategy.atomic_states[signal_name]
                momentum = signal_series.diff(momentum_span).fillna(0)
                norm_momentum = normalize_score(momentum, df.index, potential_window)
                dynamics_df[f'MOMENTUM_{signal_name}'] = norm_momentum.astype(np.float32)
                potential = signal_series.rolling(window=potential_window).mean().fillna(signal_series)
                norm_potential = normalize_score(potential, df.index, potential_window)
                dynamics_df[f'POTENTIAL_{signal_name}'] = norm_potential.astype(np.float32)
                thrust = momentum.diff(1).fillna(0)
                norm_thrust = normalize_score(thrust, df.index, potential_window)
                dynamics_df[f'THRUST_{signal_name}'] = norm_thrust.astype(np.float32)
            else:
                print(f"     - [警告] 信号 '{signal_name}' 在原子状态库中不存在，跳过动态因子计算。")
        final_df = pd.concat([df, dynamics_df], axis=1)
        return final_df

    def _calculate_behavioral_day_quality(self, df: pd.DataFrame) -> pd.Series:
        """
        【V1.1 · 工具归位版】行为K线质量分计算引擎
        - 核心修改: 调用从 utils.py 导入的公共归一化工具。
        """
        print("开始执行【V1.0 · 纯净版】行为K线质量分计算...")
        outcome_core = (df.get('closing_price_deviation_score_D', 0.5) * 2 - 1).clip(-1, 1)
        body_dominance = df.get('real_body_vs_range_ratio_D', 0.0)
        shadow_dominance = df.get('shadow_dominance_D', 0.0)
        pillar1_outcome_score = (outcome_core * 0.7 + outcome_core * body_dominance * 0.1 + shadow_dominance * 0.2).clip(-1, 1)
        vpa_eff_bipolar = get_adaptive_mtf_normalized_bipolar_score(df.get('VPA_EFFICIENCY_D', pd.Series(0.0, index=df.index)), df.index)
        vwap_ctrl_bipolar = get_adaptive_mtf_normalized_bipolar_score(df.get('vwap_control_strength_D', pd.Series(0.0, index=df.index)), df.index)
        trend_purity_bipolar = get_adaptive_mtf_normalized_bipolar_score(df.get('intraday_trend_purity_D', pd.Series(0.0, index=df.index)), df.index)
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
        【V2.7 · 上涨衰竭重构版】行为公理诊断引擎
        - 核心修改: 调用从 utils.py 导入的公共归一化工具。
        - 【新增】引入行为公理五：价量背离。
        - 【修改】将 `SCORE_BEHAVIOR_RISK_PRICE_OVEREXTENSION` 更名为 `INTERNAL_BEHAVIOR_PRICE_OVEREXTENSION_RAW`。
        - 【修改】将 `SCORE_BEHAVIOR_RISK_UPPER_SHADOW_PRESSURE` 更名为 `INTERNAL_BEHAVIOR_UPPER_SHADOW_RAW`。
        - 【重构】移除 `SCORE_RISK_STAGNATION`，新增 `INTERNAL_BEHAVIOR_RALLY_EXHAUSTION_RAW`，并采用更全面的多维度融合逻辑。
        """
        states = {}
        p_behavior = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_behavior.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        long_term_weights = get_param_value(p_mtf.get('long_term_weights'), {'weights': {21: 0.5, 55: 0.3, 89: 0.2}})
        # --- 公理一: 价格行为 (Price Action) ---
        states['SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM'] = get_adaptive_mtf_normalized_score(df['pct_change_D'].clip(lower=0), df.index, ascending=True, tf_weights=default_weights).astype(np.float32)
        states['SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM'] = get_adaptive_mtf_normalized_score(df['pct_change_D'].clip(upper=0).abs(), df.index, ascending=True, tf_weights=default_weights).astype(np.float32)
        states['INTERNAL_BEHAVIOR_PRICE_OVEREXTENSION_RAW'] = get_adaptive_mtf_normalized_score(df.get('BIAS_55_D', 0.0), df.index, ascending=True, tf_weights=long_term_weights).astype(np.float32)
        # --- 公理二: 量能行为 (Volume Action) ---
        states['SCORE_BEHAVIOR_VOLUME_BURST'] = get_adaptive_mtf_normalized_score(df.get('volume_ratio_D', 1.0), df.index, ascending=True, tf_weights=default_weights).astype(np.float32)
        states['SCORE_BEHAVIOR_VOLUME_APATHY'] = get_adaptive_mtf_normalized_score(df.get('turnover_rate_f_D', 10.0), df.index, ascending=False, tf_weights=long_term_weights).astype(np.float32)
        # --- 公理三: 价量关系 (Price-Volume Relation) ---
        states['SCORE_BEHAVIOR_UPWARD_EFFICIENCY'] = get_adaptive_mtf_normalized_score(df.get('VPA_EFFICIENCY_D', 0.5), df.index, ascending=True, tf_weights=default_weights).astype(np.float32)
        states['SCORE_BEHAVIOR_DOWNWARD_RESISTANCE'] = get_adaptive_mtf_normalized_score(df.get('VPA_EFFICIENCY_D', 0.5), df.index, ascending=False, tf_weights=default_weights).astype(np.float32)
        # --- 公理四: 日内形态 (Intraday Form) ---
        states['SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL'] = get_adaptive_mtf_normalized_score(df.get('vwap_control_strength_D', 0.5), df.index, ascending=True, tf_weights=default_weights).astype(np.float32)
        states['SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION'] = get_adaptive_mtf_normalized_score(df.get('lower_shadow_absorption_strength_D', 0.0), df.index, ascending=True, tf_weights=default_weights).astype(np.float32)
        states['INTERNAL_BEHAVIOR_UPPER_SHADOW_RAW'] = get_adaptive_mtf_normalized_score(df.get('upper_shadow_selling_pressure_D', 0.0), df.index, ascending=True, tf_weights=default_weights).astype(np.float32)
        # --- 公理五: 价量背离 (Price-Volume Divergence) ---
        price_trend = normalize_to_bipolar(df['pct_change_D'], df.index, window=55)
        volume_trend = normalize_to_bipolar(df['volume_D'].diff(1), df.index, window=55)
        divergence_score = (volume_trend - price_trend).clip(-1, 1)
        states['SCORE_BEHAVIOR_PRICE_VS_VOLUME_DIVERGENCE'] = divergence_score.astype(np.float32)
        # --- 衍生机会与风险信号 (基于纯粹的原子信号) ---
        is_rising = (df['pct_change_D'] > 0).astype(float)
        states['SCORE_OPPORTUNITY_LOCKUP_RALLY'] = (is_rising * states['SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM'] * states['SCORE_BEHAVIOR_VOLUME_APATHY']).pow(1/3).astype(np.float32)
        is_falling = (df['pct_change_D'] < 0).astype(float)
        states['SCORE_OPPORTUNITY_SELLING_EXHAUSTION'] = (is_falling * states['SCORE_BEHAVIOR_VOLUME_APATHY'] * states['SCORE_BEHAVIOR_DOWNWARD_RESISTANCE']).pow(1/3).astype(np.float32)
        # 修改行: 移除 SCORE_RISK_STAGNATION 的旧计算
        # states['SCORE_RISK_STAGNATION'] = (is_rising * states['SCORE_BEHAVIOR_VOLUME_BURST'] * (1.0 - states['SCORE_BEHAVIOR_UPWARD_EFFICIENCY'])).pow(1/3).astype(np.float32)
        states['SCORE_RISK_LIQUIDITY_DRAIN'] = (is_falling * states['SCORE_BEHAVIOR_VOLUME_BURST'] * states['SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM']).pow(1/2).astype(np.float32)
        # 新增行: 计算 INTERNAL_BEHAVIOR_RALLY_EXHAUSTION_RAW
        states['INTERNAL_BEHAVIOR_RALLY_EXHAUSTION_RAW'] = self._calculate_rally_exhaustion_raw(df, is_rising, default_weights).astype(np.float32)
        return states

    def _calculate_rally_exhaustion_raw(self, df: pd.DataFrame, is_rising: pd.Series, default_weights: Dict) -> pd.Series:
        """
        【V1.0 · 多维度融合版】计算内部行为-上涨衰竭原始分。
        - 核心逻辑: 融合行为、微观行为、资金流和筹码等多维度证据，识别上涨过程中的衰竭迹象。
        - 输出: [0, 1] 的单极性分数，代表上涨衰竭的原始强度。
        """
        print("    -> [行为情报引擎] 正在计算 INTERNAL_BEHAVIOR_RALLY_EXHAUSTION_RAW...")
        # 1. 核心行为证据 (0-1)
        volume_effort = self._get_atomic_score(df, 'SCORE_BEHAVIOR_VOLUME_BURST', 0.0)
        upward_inefficiency = 1.0 - self._get_atomic_score(df, 'SCORE_BEHAVIOR_UPWARD_EFFICIENCY', 0.5) # 0.5为中性，1.0为最差效率
        upper_shadow_raw = self._get_atomic_score(df, 'INTERNAL_BEHAVIOR_UPPER_SHADOW_RAW', 0.0)

        # 2. 微观行为证据 (0-1)
        # SCORE_MICRO_AXIOM_EFFICIENCY 是双极性 [-1, 1]，负值代表低效，取其负向绝对值
        micro_inefficiency = self._get_atomic_score(df, 'SCORE_MICRO_AXIOM_EFFICIENCY', 0.0).clip(upper=0).abs()

        # 3. 资金流证据 (0-1)
        # SCORE_FF_AXIOM_CONSENSUS 是双极性 [-1, 1]，负值代表主力卖出，取其负向绝对值
        mf_selling_pressure = self._get_atomic_score(df, 'SCORE_FF_AXIOM_CONSENSUS', 0.0).clip(upper=0).abs()
        # PROCESS_META_PROFIT_VS_FLOW 是双极性 [-1, 1]，负值代表主力赚钱卖出，取其负向绝对值
        mf_t0_profit_taking = self._get_atomic_score(df, 'PROCESS_META_PROFIT_VS_FLOW', 0.0).clip(upper=0).abs()

        # 4. 筹码结构证据 (0-1)
        # SCORE_CHIP_AXIOM_CONCENTRATION 是双极性 [-1, 1]，负值代表筹码分散，取其负向绝对值
        chip_dispersion = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_CONCENTRATION', 0.0).clip(upper=0).abs()
        # SCORE_CHIP_AXIOM_HOLDER_SENTIMENT 是双极性 [-1, 1]，负值代表信念动摇，取其负向绝对值
        holder_sentiment_decay = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT', 0.0).clip(upper=0).abs()

        # 5. 加权融合 (权重总和为1)
        # 权重分配：核心低效指标和主力资金流出/抛压指标权重较高
        w_vol = 0.10
        w_up_ineff = 0.20
        w_micro_ineff = 0.20
        w_shadow = 0.15
        w_mf_sell = 0.15
        w_chip_disp = 0.10
        w_holder_decay = 0.05
        w_t0_profit = 0.05

        raw_exhaustion_score = (
            w_vol * volume_effort +
            w_up_ineff * upward_inefficiency +
            w_micro_ineff * micro_inefficiency +
            w_shadow * upper_shadow_raw +
            w_mf_sell * mf_selling_pressure +
            w_chip_disp * chip_dispersion +
            w_holder_decay * holder_sentiment_decay +
            w_t0_profit * mf_t0_profit_taking
        )

        # 只有在价格上涨时，才计算上涨衰竭风险，否则为0
        final_raw_score = raw_exhaustion_score * is_rising

        # 归一化到 [0, 1] 范围
        normalized_score = get_adaptive_mtf_normalized_score(final_raw_score, df.index, ascending=True, tf_weights=default_weights)
        print(f"    -> [行为情报引擎] INTERNAL_BEHAVIOR_RALLY_EXHAUSTION_RAW 计算完成，最新分值: {normalized_score.iloc[-1]:.4f}")
        return normalized_score

    def _diagnose_context_new_high_strength(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0】诊断内部上下文信号：新高强度 (CONTEXT_NEW_HIGH_STRENGTH)
        - 核心逻辑: 融合价格突破、均线斜率和BIAS健康度，评估新高的综合质量。
        """
        price_breakthrough_score = normalize_score(df['pct_change_D'].clip(lower=0), df.index, window=55, ascending=True)
        ma_slope_score = normalize_score(df.get('SLOPE_5_EMA_55_D', pd.Series(0.0, index=df.index)), df.index, window=55, ascending=True)
        bias_health_score = 1 - normalize_score(df.get('BIAS_55_D', pd.Series(0.0, index=df.index)).clip(lower=0), df.index, window=55, ascending=True)
        new_high_strength = (price_breakthrough_score * ma_slope_score * bias_health_score).pow(1/3).fillna(0.0)
        return {'CONTEXT_NEW_HIGH_STRENGTH': new_high_strength.astype(np.float32)}

    def _resolve_pressure_absorption_dynamics(self, provisional_pressure: pd.Series, intent_diagnosis: pd.Series) -> Dict[str, pd.Series]:
        """
        【V3.1 · 工具归位版】压力-承接能量转化模型
        - 核心修改: 调用从 utils.py 导入的公共归一化工具。
        """
        states = {}
        df = self.strategy.df_indicators
        absorption_efficiency = get_adaptive_mtf_normalized_score(df.get('VPA_EFFICIENCY_D', pd.Series(0.5, index=df.index)), df.index, ascending=True)
        absorption_control = get_adaptive_mtf_normalized_score(df.get('vwap_control_strength_D', pd.Series(0.5, index=df.index)), df.index, ascending=True)
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
