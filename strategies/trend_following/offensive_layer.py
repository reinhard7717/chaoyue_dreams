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
        【V520.3 · 进攻风险分离与动能质量过滤版】
        - 核心重构: 将总分计算拆分为总进攻得分 (total_offensive_score) 和总风险惩罚 (total_risk_sum)。
        - 核心增强: 对 SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM 信号引入趋势质量和趋势衰竭风险的动态阻尼器。
        - 清理: 移除了用于调试的“真理探针”相关代码。
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
        # FUSION_BIPOLAR_TREND_QUALITY: -1到1，正值代表趋势质量好，负值代表差
        # COGNITIVE_RISK_TREND_EXHAUSTION: 0到1，高值代表趋势衰竭风险高
        trend_quality = all_available_signals.get('FUSION_BIPOLAR_TREND_QUALITY', pd.Series(0.0, index=df.index)).fillna(0.0)
        trend_exhaustion_risk = all_available_signals.get('COGNITIVE_RISK_TREND_EXHAUSTION', pd.Series(0.0, index=df.index)).fillna(0.0)
        # 计算动能阻尼器：
        # 1. 趋势质量越差（负值越大），阻尼越强。将 -1到1 映射到 0到1，然后反转。
        #    例如：-1 -> 1 (强阻尼), 0 -> 0.5 (中等阻尼), 1 -> 0 (无阻尼)
        #    (1 - (trend_quality + 1) / 2) 得到 0到1 的阻尼因子，趋势质量越好，阻尼因子越小。
        # 2. 趋势衰竭风险越高，阻尼越强。
        #    (1 - trend_exhaustion_risk) 得到 0到1 的阻尼因子，风险越高，阻尼因子越小。
        # 综合阻尼器：取两者中更强的阻尼效果（即更小的乘数）
        # 初始阻尼器：趋势质量越差，乘数越小；趋势衰竭风险越高，乘数越小。
        # 确保乘数在合理范围，例如 0.1 到 1.0
        trend_quality_damper = (trend_quality + 1) / 2 # 映射到 0到1，1代表趋势好，0代表趋势差
        # trend_exhaustion_damper = 1 - trend_exhaustion_risk # 映射到 0到1，1代表无风险，0代表高风险
        # 调整趋势衰竭风险的阻尼逻辑，使其在风险高时惩罚更重，但不是完全归零
        # 例如，当trend_exhaustion_risk为0时，damper为1；为1时，damper为0.2 (可配置)
        trend_exhaustion_damper = 1 - trend_exhaustion_risk * 0.8 # 0.8是可调参数，控制最大惩罚力度
        trend_exhaustion_damper = trend_exhaustion_damper.clip(lower=0.2) # 确保至少保留20%的得分
        # 最终动能阻尼器：取两者中更严格的（即更小的乘数），并确保不低于一个最小值
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
            penalty_weight = abs(meta.get('penalty_weight', 0))
            if positive_score == 0 and penalty_weight == 0:
                continue
            bonus_amount = pd.Series(0.0, index=df.index)
            if scoring_mode == 'bipolar':
                opportunity_part = processed_signal_series.clip(lower=0)
                if context_role == 'bottom_opportunity':
                    suppression_factor = top_context_score.where(top_context_score >= top_context_threshold, 0.0)
                    damper = 1.0 - suppression_factor
                    opportunity_part *= damper
                # 应用动能阻尼器到 SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM
                if signal_name == 'SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM':
                    opportunity_part *= price_momentum_damper
                    print(f"    -> [进攻层] {signal_name} 原始贡献: {opportunity_part.sum():.2f}，应用阻尼器后: {(opportunity_part * price_momentum_damper).sum():.2f}")
                bonus_amount += opportunity_part * positive_score
                risk_part = processed_signal_series.clip(upper=0).abs()
                if context_role == 'top_risk':
                    suppression_factor = bottom_context_score.where(bottom_context_score >= bottom_context_threshold, 0.0)
                    damper = 1.0 - suppression_factor
                    risk_part *= damper
                bonus_amount -= risk_part * penalty_weight
            else: # unipolar
                unipolar_series = processed_signal_series.clip(lower=0)
                if meta.get('type') == 'risk':
                    if context_role == 'top_risk':
                        suppression_factor = bottom_context_score.where(bottom_context_score >= bottom_context_threshold, 0.0)
                        damper = 1.0 - suppression_factor
                        unipolar_series *= damper
                    bonus_amount -= unipolar_series * penalty_weight
                else: # opportunity or context
                    if context_role == 'bottom_opportunity':
                        suppression_factor = top_context_score.where(top_context_score >= top_context_threshold, 0.0)
                        damper = 1.0 - suppression_factor
                        unipolar_series *= damper
                    # 应用动能阻尼器到 SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM
                    if signal_name == 'SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM':
                        unipolar_series *= price_momentum_damper
                        print(f"    -> [进攻层] {signal_name} 原始贡献: {unipolar_series.sum():.2f}，应用阻尼器后: {(unipolar_series * price_momentum_damper).sum():.2f}")
                    bonus_amount += unipolar_series * positive_score
            # 分离正向和负向贡献
            total_offensive_score += bonus_amount.clip(lower=0)
            total_risk_sum += bonus_amount.clip(upper=0).abs() # 累加负向贡献的绝对值
            score_details_df[signal_name] = bonus_amount
        return total_offensive_score.fillna(0), total_risk_sum.fillna(0), score_details_df.fillna(0)



















