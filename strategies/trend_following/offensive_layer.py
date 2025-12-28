# 文件: strategies/trend_following/offensive_layer.py
# 进攻层
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following.utils import get_params_block, get_param_value

class OffensiveLayer:
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def calculate_entry_score(self, trigger_events: Dict, bottom_context_score: pd.Series, top_context_score: pd.Series) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        """
        【V520.4 · 进攻风险分离与动能质量过滤适配版 & 风险项调试增强版】
        - 核心重构: 将总分计算拆分为总进攻得分 (total_offensive_score) 和总风险惩罚 (total_risk_sum)。
        - 核心增强: 对 SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM 信号引入趋势质量和趋势衰竭风险的动态阻尼器。
        - 调试增强: 针对 SCORE_CHIP_AXIOM_HOLDER_SENTIMENT 信号，增加详细的调试输出，以追踪其贡献值。
        """
        df = self.strategy.df_indicators
        score_details_df = pd.DataFrame(index=df.index)
        score_map = get_params_block(self.strategy, 'score_type_map', {})
        all_available_signals = self.strategy.atomic_states.copy()
        all_available_signals.update(self.strategy.playbook_states)
        total_offensive_score = pd.Series(0.0, index=df.index)
        total_risk_sum = pd.Series(0.0, index=df.index) # 累加所有负向信号的绝对值
        p_context_suppression = get_params_block(self.strategy, 'contextual_suppression_params', {})
        bottom_context_threshold = get_param_value(p_context_suppression.get('bottom_context_threshold'), 0.9)
        top_context_threshold = get_param_value(p_context_suppression.get('top_context_threshold'), 0.9)
        # 获取趋势质量和趋势衰竭风险信号用于动能阻尼器
        trend_quality = all_available_signals.get('FUSION_BIPOLAR_TREND_QUALITY', pd.Series(0.0, index=df.index)).fillna(0.0)
        cognitive_risk_trend_exhaustion = all_available_signals.get('COGNITIVE_RISK_TREND_EXHAUSTION', pd.Series(0.0, index=df.index)).fillna(0.0)
        # 计算动能阻尼器：
        trend_quality_damper = (trend_quality + 1) / 2
        trend_exhaustion_damper = 1 - cognitive_risk_trend_exhaustion * 0.8
        trend_exhaustion_damper = trend_exhaustion_damper.clip(lower=0.2)
        price_momentum_damper = pd.concat([trend_quality_damper, trend_exhaustion_damper], axis=1).min(axis=1).clip(lower=0.1, upper=1.0)
        for signal_name, meta in score_map.items():
            if not isinstance(meta, dict): continue
            signal_series = all_available_signals.get(signal_name)
            if signal_series is None or not isinstance(signal_series, pd.Series):
                continue
            processed_signal_series = signal_series.astype(float)
            scoring_mode = meta.get('scoring_mode', 'unipolar')
            context_role = meta.get('context_role', 'neutral')
            positive_score = abs(meta.get('score', 0))
            penalty_weight = meta.get('penalty_weight', 0)
            bonus_amount_for_signal = pd.Series(0.0, index=df.index)
            if scoring_mode == 'bipolar':
                opportunity_part = processed_signal_series.clip(lower=0)
                # 应用动能阻尼器到 SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM
                if signal_name == 'SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM':
                    opportunity_part *= price_momentum_damper
                    if not df.empty:
                        print(f"    -> [进攻层 Debug] {signal_name} 原始贡献: {(processed_signal_series * positive_score).iloc[-1]:.2f}，应用阻尼器后: {(opportunity_part * positive_score).iloc[-1]:.2f}")
                bonus_amount_for_signal += opportunity_part * positive_score
                risk_part = processed_signal_series.clip(upper=0).abs()
                if context_role == 'top_risk':
                    suppression_factor = bottom_context_score.where(bottom_context_score >= bottom_context_threshold, 0.0)
                    damper = 1.0 - suppression_factor
                    risk_part *= damper
                bonus_amount_for_signal += risk_part * penalty_weight
            else: # unipolar
                unipolar_series = processed_signal_series.clip(lower=0)
                if meta.get('type') == 'risk':
                    if context_role == 'top_risk':
                        suppression_factor = bottom_context_score.where(bottom_context_score >= bottom_context_threshold, 0.0)
                        damper = 1.0 - suppression_factor
                        unipolar_series *= damper
                    bonus_amount_for_signal = unipolar_series * penalty_weight
                else: # opportunity or context
                    if context_role == 'bottom_opportunity':
                        suppression_factor = top_context_score.where(top_context_score >= top_context_threshold, 0.0)
                        damper = 1.0 - suppression_factor
                        unipolar_series *= damper
                    # 应用动能阻尼器到 SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM
                    if signal_name == 'SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM':
                        unipolar_series *= price_momentum_damper
                        if not df.empty:
                            print(f"    -> [进攻层 Debug] {signal_name} 原始贡献: {(processed_signal_series * positive_score).iloc[-1]:.2f}，应用阻尼器后: {(unipolar_series * positive_score).iloc[-1]:.2f}")
                    bonus_amount_for_signal = unipolar_series * positive_score
            # --- 调试增强: 针对 SCORE_CHIP_AXIOM_HOLDER_SENTIMENT 信号 ---
            if signal_name == 'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT' and not df.empty:
                current_date = df.index[-1]
                print(f"    -> [进攻层 Debug] 信号: {signal_name} ({meta.get('cn_name', '')}) @ {current_date.strftime('%Y-%m-%d')}")
                print(f"        - 原始信号值: {signal_series.loc[current_date]:.4f}")
                print(f"        - 惩罚权重 (penalty_weight): {penalty_weight}")
                print(f"        - 计算出的贡献 (bonus_amount_for_signal): {bonus_amount_for_signal.loc[current_date]:.2f}")
                print(f"        - 是否为风险项 (contribution < 0): {bonus_amount_for_signal.loc[current_date] < 0}")
            # --- 调试增强结束 ---
            # 分离正向和负向贡献
            total_offensive_score += bonus_amount_for_signal.clip(lower=0)
            total_risk_sum += bonus_amount_for_signal.clip(upper=0).abs()
            score_details_df[signal_name] = bonus_amount_for_signal
        # 无条件输出调试信息
        if not df.empty:
            print(f"    -> [OffensiveLayer Debug] Date: {df.index[-1].strftime('%Y-%m-%d')}")
            print(f"    -> [OffensiveLayer Debug] Final total_offensive_score: {total_offensive_score.iloc[-1]:.2f}")
            print(f"    -> [OffensiveLayer Debug] Final total_risk_sum: {total_risk_sum.iloc[-1]:.2f}")
            print(f"    -> [OffensiveLayer Debug] Price Momentum Damper: {price_momentum_damper.iloc[-1]:.2f}")
            print(f"    -> [OffensiveLayer Debug] Trend Quality: {trend_quality.iloc[-1]:.2f}")
            print(f"    -> [OffensiveLayer Debug] Cognitive Risk Trend Exhaustion: {cognitive_risk_trend_exhaustion.iloc[-1]:.2f}")
        return total_offensive_score.fillna(0), total_risk_sum.fillna(0), score_details_df.fillna(0)



















