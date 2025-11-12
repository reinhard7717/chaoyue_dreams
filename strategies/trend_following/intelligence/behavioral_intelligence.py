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
        【V5.5 · 深度博弈版】行为情报模块总指挥
        - 核心升级: 废弃原子层面的“共振”和“领域健康度”信号。
        - 核心职责: 只输出行为领域的原子公理信号、行为背离信号和上下文信号。
        - 【新增】引入深度博弈版的上影线抛压风险诊断。
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
        # 【新增行】诊断深度博弈版的上影线抛压风险
        upper_shadow_risk = self._diagnose_upper_shadow_pressure_risk(df) # 新增行
        self.strategy.atomic_states.update(upper_shadow_risk) # 新增行
        all_behavioral_states.update(upper_shadow_risk) # 新增行
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
        【V4.0 · 职责重塑版】信号动态计算引擎
        - 核心错误修复: 彻底剥离了对其他情报层终极共振信号的依赖，解决了因执行时序错乱导致的信号获取失败问题。
        - 核心逻辑重构: 遵循“职责分离”原则，本方法现在只聚焦于为【本模块生产的】纯粹行为原子信号注入动态因子（动量、潜力、推力）。
                        不再计算跨领域的 RESONANCE_HEALTH_D 等信号。
        """
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_dyn = get_param_value(p_conf.get('signal_dynamics_params'), {})
        momentum_span = get_param_value(p_dyn.get('momentum_span'), 5)
        potential_window = get_param_value(p_dyn.get('potential_window'), 120)
        dynamics_df = pd.DataFrame(index=df.index)
        atomic_signals_to_enhance = [
            'SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM',
            'SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM',
            # 'SCORE_BEHAVIOR_RISK_PRICE_OVEREXTENSION', # 风险信号不直接增强动态
            'SCORE_BEHAVIOR_VOLUME_BURST',
            'SCORE_BEHAVIOR_VOLUME_APATHY',
            'SCORE_BEHAVIOR_UPWARD_EFFICIENCY',
            'SCORE_BEHAVIOR_DOWNWARD_RESISTANCE',
            'SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL',
            'SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION',
            # 'SCORE_BEHAVIOR_RISK_UPPER_SHADOW_PRESSURE', # 风险信号不直接增强动态
            'SCORE_OPPORTUNITY_LOCKUP_RALLY',
            'SCORE_OPPORTUNITY_SELLING_EXHAUSTION',
            'SCORE_RISK_STAGNATION',
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
        【V2.4 · 纯粹原子版】行为公理诊断引擎
        - 核心修改: 调用从 utils.py 导入的公共归一化工具。
        - 【新增】引入行为公理五：价量背离。
        - 【移除】SCORE_BEHAVIOR_RISK_UPPER_SHADOW_PRESSURE 的计算，现在由独立方法处理。
        """
        states = {}
        p_behavior = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_behavior.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        long_term_weights = get_param_value(p_mtf.get('long_term_weights'), {'weights': {21: 0.5, 55: 0.3, 89: 0.2}})
        # --- 公理一: 价格行为 (Price Action) ---
        states['SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM'] = get_adaptive_mtf_normalized_score(df['pct_change_D'].clip(lower=0), df.index, ascending=True, tf_weights=default_weights).astype(np.float32)
        states['SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM'] = get_adaptive_mtf_normalized_score(df['pct_change_D'].clip(upper=0).abs(), df.index, ascending=True, tf_weights=default_weights).astype(np.float32)
        states['SCORE_BEHAVIOR_RISK_PRICE_OVEREXTENSION'] = get_adaptive_mtf_normalized_score(df.get('BIAS_55_D', 0.0), df.index, ascending=True, tf_weights=long_term_weights).astype(np.float32)
        # --- 公理二: 量能行为 (Volume Action) ---
        states['SCORE_BEHAVIOR_VOLUME_BURST'] = get_adaptive_mtf_normalized_score(df.get('volume_ratio_D', 1.0), df.index, ascending=True, tf_weights=default_weights).astype(np.float32)
        states['SCORE_BEHAVIOR_VOLUME_APATHY'] = get_adaptive_mtf_normalized_score(df.get('turnover_rate_f_D', 10.0), df.index, ascending=False, tf_weights=long_term_weights).astype(np.float32)
        # --- 公理三: 价量关系 (Price-Volume Relation) ---
        states['SCORE_BEHAVIOR_UPWARD_EFFICIENCY'] = get_adaptive_mtf_normalized_score(df.get('VPA_EFFICIENCY_D', 0.5), df.index, ascending=True, tf_weights=default_weights).astype(np.float32)
        states['SCORE_BEHAVIOR_DOWNWARD_RESISTANCE'] = get_adaptive_mtf_normalized_score(df.get('VPA_EFFICIENCY_D', 0.5), df.index, ascending=False, tf_weights=default_weights).astype(np.float32)
        # --- 公理四: 日内形态 (Intraday Form) ---
        states['SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL'] = get_adaptive_mtf_normalized_score(df.get('vwap_control_strength_D', 0.5), df.index, ascending=True, tf_weights=default_weights).astype(np.float32)
        states['SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION'] = get_adaptive_mtf_normalized_score(df.get('lower_shadow_absorption_strength_D', 0.0), df.index, ascending=True, tf_weights=default_weights).astype(np.float32)
        # 【移除行】states['SCORE_BEHAVIOR_RISK_UPPER_SHADOW_PRESSURE'] = get_adaptive_mtf_normalized_score(df.get('upper_shadow_selling_pressure_D', 0.0), df.index, ascending=True, tf_weights=default_weights).astype(np.float32)
        # --- 公理五: 价量背离 (Price-Volume Divergence) ---
        price_trend = normalize_to_bipolar(df['pct_change_D'], df.index, window=55)
        volume_trend = normalize_to_bipolar(df['volume_D'].diff(1), df.index, window=55)
        divergence_score = (volume_trend - price_trend).clip(-1, 1)
        states['SCORE_BEHAVIOR_PRICE_VS_VOLUME_DIVERGENCE'] = divergence_score.astype(np.float32)
        # --- 衍生机会与风险信号 (基于纯粹的原子信号) ---
        is_rising = (df['pct_change_D'] > 0).astype(float)
        is_falling = (df['pct_change_D'] < 0).astype(float)
        states['SCORE_OPPORTUNITY_LOCKUP_RALLY'] = (is_rising * states['SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM'] * states['SCORE_BEHAVIOR_VOLUME_APATHY']).pow(1/3).astype(np.float32)
        states['SCORE_OPPORTUNITY_SELLING_EXHAUSTION'] = (is_falling * states['SCORE_BEHAVIOR_VOLUME_APATHY'] * states['SCORE_BEHAVIOR_DOWNWARD_RESISTANCE']).pow(1/3).astype(np.float32)
        states['SCORE_RISK_STAGNATION'] = (is_rising * states['SCORE_BEHAVIOR_VOLUME_BURST'] * (1.0 - states['SCORE_BEHAVIOR_UPWARD_EFFICIENCY'])).pow(1/3).astype(np.float32)
        states['SCORE_RISK_LIQUIDITY_DRAIN'] = (is_falling * states['SCORE_BEHAVIOR_VOLUME_BURST'] * states['SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM']).pow(1/2).astype(np.float32)
        return states

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

    def _diagnose_upper_shadow_pressure_risk(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.1 · 深度博弈版】诊断行为风险：上影线抛压 (SCORE_BEHAVIOR_RISK_UPPER_SHADOW_PRESSURE)
        - 核心逻辑: 综合考虑上影线强度、价格涨跌、主力资金流向、筹码集中度变化、微观欺骗意图和上涨效率，
                    以更全面、精确地判断上影线是真抛压还是洗盘/诱多。
        - 数据来源:
            1. upper_shadow_selling_pressure_D (原始上影线强度)
            2. pct_change_D (当日涨跌幅)
            3. main_force_net_flow_calibrated_D (主力资金净流向)
            4. SCORE_CHIP_AXIOM_CONCENTRATION (筹码集中度，双极性)
            5. SCORE_MICRO_AXIOM_DECEPTION (微观欺骗，双极性)
            6. VPA_EFFICIENCY_D (量价效率)
        - 输出: [0, 1] 的风险分数，分数越高代表风险越大。
        - 【修复】直接从 `self.strategy.atomic_states` 获取跨模块原子信号。
        """
        print("    -- [行为引擎] 诊断上影线抛压风险 (深度博弈版)...")
        # 1. 获取核心参数和信号
        p_behavior = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_behavior.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        norm_window = 55 # 统一归一化窗口
        upper_shadow_raw = self._get_signal(df, 'upper_shadow_selling_pressure_D', 0.0)
        pct_change = self._get_signal(df, 'pct_change_D', 0.0)
        main_force_flow_raw = self._get_signal(df, 'main_force_net_flow_calibrated_D', 0.0)
        vpa_efficiency_raw = self._get_signal(df, 'VPA_EFFICIENCY_D', 0.5)
        # 从 atomic_states 获取双极性信号
        # 修改行: 直接从 self.strategy.atomic_states 获取
        chip_concentration = self.strategy.atomic_states.get('SCORE_CHIP_AXIOM_CONCENTRATION', pd.Series(0.0, index=df.index)) # [-1, 1]
        # 修改行: 直接从 self.strategy.atomic_states 获取
        micro_deception = self.strategy.atomic_states.get('SCORE_MICRO_AXIOM_DECEPTION', pd.Series(0.0, index=df.index)) # [-1, 1]
        # 2. 归一化基础信号
        # 基础上影线强度，越高风险越大，归一化到 [0, 1]
        base_upper_shadow_score = normalize_score(upper_shadow_raw, df.index, norm_window, ascending=True)
        # 3. 价格背景判断
        is_up_day = (pct_change > 0).astype(float)
        is_down_day = (pct_change < 0).astype(float)
        # 4. 主力资金流向影响 (转换为风险放大/缓解因子)
        # 主力资金流出 (负值) 增加风险，流入 (正值) 降低风险
        main_force_flow_bipolar = normalize_to_bipolar(main_force_flow_raw, df.index, norm_window)
        # mf_risk_factor: 流出时为正，流入时为负，范围 [-1, 1]
        mf_risk_factor = -main_force_flow_bipolar
        # 5. 筹码集中度影响 (转换为风险放大/缓解因子)
        # 筹码分散 (负值) 增加风险，集中 (正值) 降低风险
        # chip_risk_factor: 分散时为正，集中时为负，范围 [-1, 1]
        chip_risk_factor = -chip_concentration
        # 6. 微观欺骗意图影响 (转换为风险放大/缓解因子)
        # 伪装派发 (负值) 增加风险，伪装吸筹 (正值) 降低风险
        # deception_risk_factor: 伪装派发时为正，伪装吸筹时为负，范围 [-1, 1]
        deception_risk_factor = -micro_deception
        # 7. 上涨效率影响 (仅在上涨日考虑，转换为风险放大/缓解因子)
        # 上涨效率低 (VPA_EFFICIENCY_D 接近0) 增加风险，效率高 (接近1) 降低风险
        vpa_efficiency_score = normalize_score(vpa_efficiency_raw, df.index, norm_window, ascending=True)
        # vpa_risk_factor: 效率低时为正，效率高时为负，范围 [-1, 1]
        vpa_risk_factor = (1 - vpa_efficiency_score) * is_up_day - vpa_efficiency_score * is_up_day # 仅在上涨日生效
        # 8. 综合修正因子 (加权平均)
        # 权重需要根据实际市场表现进行优化，这里给出初步设定
        w_mf = 0.3 # 主力资金流向权重
        w_chip = 0.25 # 筹码集中度权重
        w_deception = 0.2 # 微观欺骗权重
        w_vpa = 0.15 # 上涨效率权重
        # 综合修正因子，范围大致在 [-1, 1] 之间
        combined_influence = (
            mf_risk_factor * w_mf +
            chip_risk_factor * w_chip +
            deception_risk_factor * w_deception +
            vpa_risk_factor * w_vpa
        ) / (w_mf + w_chip + w_deception + w_vpa) # 归一化权重和
        # 9. 最终风险分数计算
        # 基础风险分数 (0到1) 乘以一个放大/缓解系数 (1 + 修正因子)
        # 修正因子范围 [-1, 1]，所以 (1 + 修正因子) 范围 [0, 2]
        # 这样，如果修正因子为负（缓解风险），最终风险会降低；如果为正（放大风险），最终风险会升高。
        # 额外考虑下跌日的上影线，通常风险更高，给予一个基础放大
        down_day_amplification = is_down_day * 0.5 # 下跌日上影线额外放大0.5倍风险
        final_risk_score = (base_upper_shadow_score * (1 + combined_influence + down_day_amplification)).clip(0, 1)
        return {'SCORE_BEHAVIOR_RISK_UPPER_SHADOW_PRESSURE': final_risk_score.astype(np.float32)}
