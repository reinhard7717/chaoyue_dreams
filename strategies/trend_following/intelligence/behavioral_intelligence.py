# 文件: strategies/trend_following/intelligence/behavioral_intelligence.py
# 行为与模式识别模块
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Tuple, Optional, List
from strategies.trend_following.intelligence.tactic_engine import TacticEngine
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score, get_adaptive_mtf_normalized_score, get_adaptive_mtf_normalized_bipolar_score

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
        # K线形态识别器可能需要在这里初始化或传入
        self.pattern_recognizer = strategy_instance.pattern_recognizer
        self.tactic_engine = TacticEngine(strategy_instance)

    def run_behavioral_analysis_command(self) -> Dict[str, pd.Series]:
        """
        【V5.0 · 职责净化版】行为情报模块总指挥
        - 核心重构: 遵循“三层金字塔”架构，本模块只负责生产纯粹的“行为原子信号”。
          1. 调用 `_diagnose_behavioral_axioms` 基于“价、量、关系、形态”四大公理，提炼核心原子信号。
          2. 调用 `_calculate_signal_dynamics` 为这些原子信号注入动态因子。
          3. 调用 `diagnose_ultimate_behavioral_signals` 将所有信息合成为本领域的终极输出。
        """
        print("启动【V5.0 · 职责净化版】行为情报分析...")
        df = self.strategy.df_indicators
        all_behavioral_states = {}
        # 工序一: 基于行为公理，生产纯粹的原子信号
        print("工序一: 正在生产纯粹的行为原子信号...")
        atomic_signals = self._diagnose_behavioral_axioms(df)
        self.strategy.atomic_states.update(atomic_signals)
        all_behavioral_states.update(atomic_signals)
        # 将新生成的原子信号合并到主DataFrame，为下一步做准备
        for k, v in atomic_signals.items():
            if k not in df.columns:
                df[k] = v
        # 工序二: 为行为信号进行动态赋能
        print("工序二: 正在为行为信号注入动态因子...")
        df_with_dynamics = self._calculate_signal_dynamics(df)
        dynamic_cols = [c for c in df_with_dynamics.columns if c.startswith(('MOMENTUM_', 'POTENTIAL_', 'THRUST_', 'RESONANCE_'))]
        self.strategy.atomic_states.update(df_with_dynamics[dynamic_cols])
        all_behavioral_states.update(df_with_dynamics[dynamic_cols])
        # 工序三: 合成行为领域的终极信号
        print("工序三: 正在合成终极行为信号...")
        ultimate_behavioral_states = self.diagnose_ultimate_behavioral_signals(df_with_dynamics, atomic_signals=self.strategy.atomic_states)
        if ultimate_behavioral_states:
            all_behavioral_states.update(ultimate_behavioral_states)
        print("【V5.0 · 职责净化版】行为情报分析完成。")
        return all_behavioral_states

    def diagnose_ultimate_behavioral_signals(self, df: pd.DataFrame, atomic_signals: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V5.0 · 范式统一版】行为领域终极合成器
        - 核心重构: 废弃旧的健康度/机会度模型，全面转向“双极性健康分 -> 共振分”的标准范式，
                      以解决下游引擎的信号失联问题。
        - 新逻辑:
          1. 分别计算“看涨健康度”与“看跌健康度”。
          2. 合成为一个双极性的“行为健康总分”。
          3. 使用标准工具分裂为互斥的“看涨共振分”和“看跌共振分”。
        """
        # [代码修改开始]
        print("开始执行【V5.0 · 范式统一版】行为领域终极合成器...")
        states = {}
        
        # 1. 计算看涨健康度 (Bullish Health)
        # 证据: 上涨有效率 * 下跌有抵抗 * 日内多头能控盘
        bullish_health = (
            atomic_signals.get('SCORE_BEHAVIOR_UPWARD_EFFICIENCY', pd.Series(0.5, index=df.index)) *
            atomic_signals.get('SCORE_BEHAVIOR_DOWNWARD_RESISTANCE', pd.Series(0.5, index=df.index)) *
            atomic_signals.get('SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL', pd.Series(0.5, index=df.index))
        ).pow(1/3)

        # 2. 计算看跌健康度 (Bearish Health / Risk)
        # 证据: 取“滞涨风险”和“流动性流失风险”中的最大值
        bearish_health = np.maximum(
            atomic_signals.get('SCORE_RISK_STAGNATION', pd.Series(0.0, index=df.index)),
            atomic_signals.get('SCORE_RISK_LIQUIDITY_DRAIN', pd.Series(0.0, index=df.index))
        )

        # 3. 合成双极性行为健康总分
        bipolar_behavioral_health = (bullish_health - bearish_health).clip(-1, 1)

        # 4. 分裂为标准的看涨/看跌共振信号
        from .utils import bipolar_to_exclusive_unipolar
        bullish_resonance, bearish_resonance = bipolar_to_exclusive_unipolar(bipolar_behavioral_health)

        states['SCORE_BEHAVIOR_BULLISH_RESONANCE'] = bullish_resonance.astype(np.float32)
        states['SCORE_BEHAVIOR_BEARISH_RESONANCE'] = bearish_resonance.astype(np.float32)
        
        print("【V5.0 · 范式统一版】行为领域终极合成器诊断完成。")
        return states
        # [代码修改结束]

    # ==============================================================================
    # 以下为新增的原子信号中心和降级的原子诊断引擎
    # ==============================================================================
    def _get_atomic_score(self, df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
        """
        【V1.0 · 新增】安全地从原子状态库或主数据帧中获取分数。
        - 核心职责: 统一信号获取路径，优先从 self.strategy.atomic_states 获取，
                      若无则从主数据帧 df 获取，最后提供默认值，确保数据流的稳定性。
        """
        # [代码新增开始]
        if name in self.strategy.atomic_states:
            return self.strategy.atomic_states[name]
        elif name in df.columns:
            return df[name]
        else:
            # 打印警告信息，便于调试
            print(f"     -> [行为情报引擎警告] 信号 '{name}' 不存在，使用默认值 {default}。")
            return pd.Series(default, index=df.index)
        # [代码新增结束]

    def _generate_all_atomic_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.0 · 职责净化版】原子信号中心
        - 核心升级: 遵循“三层金字塔”架构，本方法不再计算跨领域的“趋势健康度”和“绝望度”。
                      这些高级融合逻辑已迁移至 FusionIntelligence。
                      新增对纯净版“行为K线质量分”的计算和发布。
        """
        atomic_signals = {}
        # 步骤一: 计算纯粹的行为原子信号
        atomic_signals.update(self._diagnose_behavioral_axioms(df))
        # 步骤二: 计算纯粹的行为K线质量分，并基于其EMA计算战场动能
        day_quality_score = self._calculate_behavioral_day_quality(df)
        atomic_signals['BIPOLAR_BEHAVIORAL_DAY_QUALITY'] = day_quality_score
        battlefield_momentum = day_quality_score.ewm(span=5, adjust=False).mean()
        atomic_signals['SCORE_BEHAVIORAL_BATTLEFIELD_MOMENTUM'] = battlefield_momentum.astype(np.float32)
        # 立即发布，供后续引擎使用
        self.strategy.atomic_states.update(atomic_signals)
        # 步骤三: 运行依赖于基础行为信号的诊断引擎
        # 注意：这些引擎现在也必须被净化，只使用行为层信号
        atomic_signals.update(self._diagnose_upper_shadow_intent(df))
        # ... 其他纯行为诊断引擎的调用 ...
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

        # 定义需要计算动态因子的本模块原子信号列表
        # 这些信号必须是由 _diagnose_behavioral_axioms 产出的
        atomic_signals_to_enhance = [
            'SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM',
            'SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM',
            'SCORE_BEHAVIOR_RISK_PRICE_OVEREXTENSION',
            'SCORE_BEHAVIOR_VOLUME_BURST',
            'SCORE_BEHAVIOR_VOLUME_APATHY',
            'SCORE_BEHAVIOR_UPWARD_EFFICIENCY',
            'SCORE_BEHAVIOR_DOWNWARD_RESISTANCE',
            'SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL',
            'SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION',
            'SCORE_BEHAVIOR_RISK_UPPER_SHADOW_PRESSURE',
            'SCORE_OPPORTUNITY_LOCKUP_RALLY',
            'SCORE_OPPORTUNITY_SELLING_EXHAUSTION',
            'SCORE_RISK_STAGNATION',
            'SCORE_RISK_LIQUIDITY_DRAIN'
        ]

        print("  -> 正在为以下行为原子信号计算动态因子:")
        for signal_name in atomic_signals_to_enhance:
            if signal_name in self.strategy.atomic_states:
                print(f"     - {signal_name}")
                signal_series = self.strategy.atomic_states[signal_name]
                
                # 计算动量 (Momentum)
                momentum = signal_series.diff(momentum_span).fillna(0)
                norm_momentum = normalize_score(momentum, df.index, potential_window)
                dynamics_df[f'MOMENTUM_{signal_name}'] = norm_momentum.astype(np.float32)

                # 计算潜力 (Potential) - 长期变化趋势
                potential = signal_series.rolling(window=potential_window).mean().fillna(signal_series)
                norm_potential = normalize_score(potential, df.index, potential_window)
                dynamics_df[f'POTENTIAL_{signal_name}'] = norm_potential.astype(np.float32)

                # 计算推力 (Thrust) - 短期变化加速度
                thrust = momentum.diff(1).fillna(0)
                norm_thrust = normalize_score(thrust, df.index, potential_window)
                dynamics_df[f'THRUST_{signal_name}'] = norm_thrust.astype(np.float32)
            else:
                print(f"     - [警告] 信号 '{signal_name}' 在原子状态库中不存在，跳过动态因子计算。")

        # 将新计算的动态因子合并到主DataFrame中
        final_df = pd.concat([df, dynamics_df], axis=1)
        
        return final_df

    def _calculate_behavioral_day_quality(self, df: pd.DataFrame) -> pd.Series:
        """
        【V1.1 · 工具归位版】行为K线质量分计算引擎
        - 核心修改: 调用从 utils.py 导入的公共归一化工具。
        """
        print("开始执行【V1.0 · 纯净版】行为K线质量分计算...")
        # --- 支柱一: 战役结果 (The Battle's Outcome) - 阵地控制权 ---
        outcome_core = (df.get('closing_price_deviation_score_D', 0.5) * 2 - 1).clip(-1, 1)
        body_dominance = df.get('real_body_vs_range_ratio_D', 0.0)
        shadow_dominance = df.get('shadow_dominance_D', 0.0) # 这是一个[-1, 1]的指标
        pillar1_outcome_score = (outcome_core * 0.7 + outcome_core * body_dominance * 0.1 + shadow_dominance * 0.2).clip(-1, 1)
        # --- 支柱二: 战术执行 (The Tactical Execution) - 操盘效率与决心 ---
        # 调用公共工具函数，并传入 df.index
        vpa_eff_bipolar = get_adaptive_mtf_normalized_bipolar_score(df.get('VPA_EFFICIENCY_D', pd.Series(0.0, index=df.index)), df.index)
        vwap_ctrl_bipolar = get_adaptive_mtf_normalized_bipolar_score(df.get('vwap_control_strength_D', pd.Series(0.0, index=df.index)), df.index)
        trend_purity_bipolar = get_adaptive_mtf_normalized_bipolar_score(df.get('intraday_trend_purity_D', pd.Series(0.0, index=df.index)), df.index)
        bullish_execution = (((vpa_eff_bipolar + 1)/2) * ((vwap_ctrl_bipolar + 1)/2) * ((trend_purity_bipolar + 1)/2)).pow(1/3)
        pillar2_execution_score = (bullish_execution * 2 - 1).clip(-1, 1)
        # --- 最终融合: 两大纯行为支柱加权 ---
        day_quality_score = (
            pillar1_outcome_score * 0.4 +
            pillar2_execution_score * 0.6
        ).clip(-1, 1)
        print("【纯净版行为K线质量分】计算完成。")
        return day_quality_score.astype(np.float32)

    def _diagnose_behavioral_axioms(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.1 · 工具归位版】行为公理诊断引擎
        - 核心修改: 调用从 utils.py 导入的公共归一化工具。
        """
        print("开始执行【V2.0 · 职责净化版 · 行为公理诊断引擎】...")
        states = {}
        p_behavior = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_behavior.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        long_term_weights = get_param_value(p_mtf.get('volatility_weights'), {'weights': {21: 0.5, 55: 0.3, 89: 0.2}})
        # --- 公理一: 价格行为 (Price Action) ---
        price_upward_momentum = get_adaptive_mtf_normalized_score(df['pct_change_D'].clip(lower=0), df.index, ascending=True, tf_weights=default_weights)
        states['SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM'] = price_upward_momentum.astype(np.float32)
        price_downward_momentum = get_adaptive_mtf_normalized_score(df['pct_change_D'].clip(upper=0).abs(), df.index, ascending=True, tf_weights=default_weights)
        states['SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM'] = price_downward_momentum.astype(np.float32)
        bias_risk = get_adaptive_mtf_normalized_score(df.get('BIAS_55_D', 0.0), df.index, ascending=True, tf_weights=long_term_weights)
        states['SCORE_BEHAVIOR_RISK_PRICE_OVEREXTENSION'] = bias_risk.astype(np.float32)
        # --- 公理二: 量能行为 (Volume Action) ---
        volume_burst = get_adaptive_mtf_normalized_score(df.get('volume_ratio_D', 1.0), df.index, ascending=True, tf_weights=default_weights)
        states['SCORE_BEHAVIOR_VOLUME_BURST'] = volume_burst.astype(np.float32)
        volume_apathy = get_adaptive_mtf_normalized_score(df.get('turnover_rate_f_D', 10.0), df.index, ascending=False, tf_weights=long_term_weights)
        states['SCORE_BEHAVIOR_VOLUME_APATHY'] = volume_apathy.astype(np.float32)
        # --- 公理三: 价量关系 (Price-Volume Relation) ---
        upward_efficiency = get_adaptive_mtf_normalized_score(df.get('VPA_EFFICIENCY_D', 0.5), df.index, ascending=True, tf_weights=default_weights)
        states['SCORE_BEHAVIOR_UPWARD_EFFICIENCY'] = upward_efficiency.astype(np.float32)
        downward_resistance = get_adaptive_mtf_normalized_score(df.get('VPA_EFFICIENCY_D', 0.5), df.index, ascending=False, tf_weights=default_weights)
        states['SCORE_BEHAVIOR_DOWNWARD_RESISTANCE'] = downward_resistance.astype(np.float32)
        # --- 公理四: 日内形态 (Intraday Form) ---
        intraday_bull_control = get_adaptive_mtf_normalized_score(df.get('vwap_control_strength_D', 0.5), df.index, ascending=True, tf_weights=default_weights)
        states['SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL'] = intraday_bull_control.astype(np.float32)
        lower_shadow_absorption = get_adaptive_mtf_normalized_score(df.get('lower_shadow_absorption_strength_D', 0.0), df.index, ascending=True, tf_weights=default_weights)
        states['SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION'] = lower_shadow_absorption.astype(np.float32)
        upper_shadow_pressure = get_adaptive_mtf_normalized_score(df.get('upper_shadow_selling_pressure_D', 0.0), df.index, ascending=True, tf_weights=default_weights)
        states['SCORE_BEHAVIOR_RISK_UPPER_SHADOW_PRESSURE'] = upper_shadow_pressure.astype(np.float32)
        # --- 衍生机会与风险信号 (基于纯粹的原子信号) ---
        is_rising = (df['pct_change_D'] > 0).astype(float)
        is_falling = (df['pct_change_D'] < 0).astype(float)
        states['SCORE_OPPORTUNITY_LOCKUP_RALLY'] = (is_rising * price_upward_momentum * volume_apathy).pow(1/3).astype(np.float32)
        states['SCORE_OPPORTUNITY_SELLING_EXHAUSTION'] = (is_falling * volume_apathy * downward_resistance).pow(1/3).astype(np.float32)
        states['SCORE_RISK_STAGNATION'] = (is_rising * volume_burst * (1.0 - upward_efficiency)).pow(1/3).astype(np.float32)
        states['SCORE_RISK_LIQUIDITY_DRAIN'] = (is_falling * volume_burst * price_downward_momentum).pow(1/2).astype(np.float32)
        print("【行为公理诊断引擎】计算完成。")
        return states

    def _resolve_pressure_absorption_dynamics(self, provisional_pressure: pd.Series, intent_diagnosis: pd.Series) -> Dict[str, pd.Series]:
        """
        【V3.1 · 工具归位版】压力-承接能量转化模型
        - 核心修改: 调用从 utils.py 导入的公共归一化工具。
        """
        print("开始执行【V3.0 · 压力-承接能量转化模型】...")
        states = {}
        df = self.strategy.df_indicators
        # --- 步骤一: 评估承接质量 (Absorption Quality) ---
        # 调用公共工具函数，并传入 df.index
        absorption_efficiency = get_adaptive_mtf_normalized_score(df.get('VPA_EFFICIENCY_D', pd.Series(0.5, index=df.index)), df.index, ascending=True)
        absorption_control = get_adaptive_mtf_normalized_score(df.get('vwap_control_strength_D', pd.Series(0.5, index=df.index)), df.index, ascending=True)
        # 证据1.3: 承接意图 (从-1,1映射到0,1)
        absorption_intent_factor = (intent_diagnosis.clip(-1, 1) + 1) / 2.0
        # 融合得到承接质量，体现“三位一体”
        absorption_quality_score = (absorption_efficiency * absorption_control * absorption_intent_factor).pow(1/3)
        # --- 步骤二: 计算博弈动能 (Battlefield Momentum) ---
        # 日度净多头力量 = 承接质量 - 原始压力
        daily_net_force = absorption_quality_score - provisional_pressure
        # 博弈动能 = 日度净多头力量的3日EMA，捕捉力量对比的“势”
        battlefield_momentum_score = daily_net_force.ewm(span=3, adjust=False).mean().fillna(0)
        # --- 步骤三: 裁定最终风险与机会 (能量分配) ---
        # 1. 计算最终风险 (Unresolved Risk)
        # 基础风险 = 原始压力中未被高质量承接的部分
        base_risk = provisional_pressure * (1.0 - absorption_quality_score)
        # 动能放大器: 如果空头势头正盛(动能为负)，则风险加剧
        risk_amplifier = 1.0 - battlefield_momentum_score.clip(upper=0) # 范围 [1, 2]
        final_risk_score = (base_risk * risk_amplifier).clip(0, 1)
        # 2. 计算最终机会 (Absorption Opportunity)
        # 基础机会 = 被高质量吸收的压力，即“被压缩的弹簧”
        base_opportunity = provisional_pressure * absorption_quality_score
        # 动能放大器: 如果多头势头正在逆转(动能为正)，则机会巨大，弹簧开始释放
        opportunity_amplifier = 1.0 + battlefield_momentum_score.clip(lower=0) # 范围 [1, 2]
        # 战略背景调节：在健康趋势中吸收压力，机会更大
        trend_health = self.strategy.atomic_states.get('SCORE_TREND_HEALTH', pd.Series(0.5, index=df.index))
        context_modulator = 1.0 + trend_health * 0.5 # 范围 [1, 1.5]
        final_opportunity_score = (base_opportunity * opportunity_amplifier * context_modulator).clip(0, 1)
        # 统一命名，更清晰地反映信号内涵
        states['SCORE_RISK_UNRESOLVED_PRESSURE'] = final_risk_score.astype(np.float32)
        states['SCORE_OPPORTUNITY_PRESSURE_ABSORPTION'] = final_opportunity_score.astype(np.float32)
        print("【压力-承接能量转化模型】计算完成。")
        return states






