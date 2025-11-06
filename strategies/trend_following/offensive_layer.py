# 文件: strategies/trend_following/offensive_layer.py
# 进攻层
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following.utils import get_params_block, get_param_value

class OffensiveLayer:
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def calculate_entry_score(self, trigger_events: Dict, bottom_context_score: pd.Series, top_context_score: pd.Series) -> Tuple[pd.Series, pd.DataFrame]:
        """
        【V519.0 · 真理探针版】
        - 核心重构: 保持“统一情报总线”的设计。
        - 探针植入: 植入高精度探针，在计分循环内外打印剧本信号的原始值和得分贡献，以彻底查明其是否被激活。
        """
        df = self.strategy.df_indicators
        score_details_df = pd.DataFrame(index=df.index)
        score_map = get_params_block(self.strategy, 'score_type_map', {})
        all_available_signals = self.strategy.atomic_states.copy()
        all_available_signals.update(self.strategy.playbook_states)
        total_score = pd.Series(0.0, index=df.index)
        p_context_suppression = get_params_block(self.strategy, 'contextual_suppression_params', {})
        bottom_context_threshold = get_param_value(p_context_suppression.get('bottom_context_threshold'), 0.9)
        top_context_threshold = get_param_value(p_context_suppression.get('top_context_threshold'), 0.9)
        
        # [代码新增开始]
        # --- 真理探针 1: 检查计分前剧本信号的原始状态 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date = pd.to_datetime(probe_dates_str[0])
            print("\n" + "="*20 + f" [进攻层-真理探针] @ {probe_date.date()} " + "="*20)
            print("--- [探针 1/2] 检查计分前'统一情报总线'中的剧本信号原始值 ---")
            playbook_keys = [k for k in all_available_signals.keys() if 'PLAYBOOK' in k]
            if not playbook_keys:
                print("  -> [探针警告] '统一情报总线'中未发现任何剧本信号！")
            else:
                for key in playbook_keys:
                    signal_series = all_available_signals.get(key)
                    if isinstance(signal_series, pd.Series) and probe_date in signal_series.index:
                        raw_value = signal_series.loc[probe_date]
                        print(f"  -> 信号: {key:<50} | 当日原始值: {raw_value:.4f}")
                    else:
                        print(f"  -> 信号: {key:<50} | [探针警告] 无法在探针日期找到该信号值。")
            print("--- [探针 2/2] 开始计分循环，监控剧本信号的得分贡献 ---")
        # [代码新增结束]

        for signal_name, meta in score_map.items():
            if not isinstance(meta, dict): continue
            signal_series = all_available_signals.get(signal_name)
            if signal_series is None or not isinstance(signal_series, pd.Series):
                continue
            
            # [代码新增开始]
            # --- 真理探针 2: 在循环内部监控特定信号 ---
            is_playbook_signal = 'PLAYBOOK' in signal_name
            # [代码新增结束]

            processed_signal_series = signal_series.astype(float)
            scoring_mode = meta.get('scoring_mode', 'unipolar')
            context_role = meta.get('context_role', 'neutral')
            positive_score = abs(meta.get('score', 0))
            penalty_weight = abs(meta.get('penalty_weight', 0))
            if penalty_weight == 0 and positive_score > 0:
                penalty_weight = positive_score
            if positive_score == 0 and penalty_weight > 0:
                positive_score = penalty_weight
            if positive_score == 0 and penalty_weight == 0:
                continue
            bonus_amount = pd.Series(0.0, index=df.index)
            if scoring_mode == 'bipolar':
                opportunity_part = processed_signal_series.clip(lower=0)
                if context_role == 'bottom_opportunity':
                    suppression_factor = top_context_score.where(top_context_score >= top_context_threshold, 0.0)
                    damper = 1.0 - suppression_factor
                    opportunity_part *= damper
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
                else:
                    if context_role == 'bottom_opportunity':
                        suppression_factor = top_context_score.where(top_context_score >= top_context_threshold, 0.0)
                        damper = 1.0 - suppression_factor
                        unipolar_series *= damper
                    bonus_amount += unipolar_series * positive_score
            
            # [代码新增开始]
            # --- 真理探针 2: 续 ---
            if is_playbook_signal and probe_dates_str:
                probe_date = pd.to_datetime(probe_dates_str[0])
                if probe_date in bonus_amount.index:
                    final_contribution = bonus_amount.loc[probe_date]
                    if final_contribution != 0:
                         print(f"    -> [计分中] 信号: {signal_name:<45} | 得分贡献: {final_contribution:.2f}")
            # [代码新增结束]

            total_score += bonus_amount
            score_details_df[signal_name] = bonus_amount
        return total_score.fillna(0), score_details_df.fillna(0)


















