# 文件: strategies/trend_following/offensive_layer.py
# 进攻层
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from .utils import get_params_block, get_param_value

class OffensiveLayer:
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def calculate_entry_score(self, trigger_events: Dict) -> Tuple[pd.Series, pd.DataFrame]:
        """
        【V501.0 · 终极信号适配版】
        - 核心重构: 全面转向消费由各情报层和认知层产出的、统一范式的终极信号。
        - 计分逻辑: 保持“反转优先”的哲学，但信号源已全面更新。
        """
        print("        -> [进攻方案评估中心 V501.0 · 终极信号适配版] 启动...")
        df = self.strategy.df_indicators
        score_details_df = pd.DataFrame(index=df.index)
        scoring_params = get_params_block(self.strategy, 'four_layer_scoring_params')
        if not get_param_value(scoring_params.get('enabled'), True):
            return pd.Series(0.0, index=df.index), score_details_df
        
        # --- 步骤 1: 计算“反转进攻分” ---
        reversal_params = scoring_params.get('reversal_offense_scoring', {})
        reversal_score, score_details_df = self._calculate_weighted_score(
            reversal_params.get('positive_signals', {}),
            score_details_df
        )
        
        # --- 步骤 2: 计算“共振进攻分” ---
        resonance_params = scoring_params.get('resonance_offense_scoring', {})
        resonance_score, score_details_df = self._calculate_weighted_score(
            resonance_params.get('positive_signals', {}),
            score_details_df
        )

        # --- 步骤 3: 计算“剧本协同分” ---
        playbook_params = scoring_params.get('playbook_synergy_scoring', {})
        playbook_score, score_details_df = self._calculate_weighted_score(
            playbook_params.get('positive_signals', {}),
            score_details_df
        )
        
        # --- 步骤 4: 计算“触发器加分” ---
        trigger_params = scoring_params.get('trigger_event_scoring', {})
        trigger_score, score_details_df = self._calculate_weighted_score(
            trigger_params.get('positive_signals', {}),
            score_details_df
        )

        # --- 步骤 5: 合成总进攻分 ---
        entry_score = (reversal_score + resonance_score + playbook_score + trigger_score).fillna(0).astype(int)
        
        score_details_df['SCORE_REVERSAL_OFFENSE'] = reversal_score
        score_details_df['SCORE_RESONANCE_OFFENSE'] = resonance_score
        score_details_df['SCORE_PLAYBOOK_SYNERGY'] = playbook_score
        score_details_df['SCORE_TRIGGER'] = trigger_score

        return entry_score, score_details_df.fillna(0)

    def _calculate_weighted_score(self, signals_config: Dict, score_details_df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        【V501.0 · 终极信号适配版】向量化加权分数计算辅助函数
        - 核心重构: 简化了信号源的获取逻辑，现在统一从 atomic_states 和 playbook_states 中获取。
        """
        atomic_states = self.strategy.atomic_states
        playbook_states = self.strategy.playbook_states
        df_index = self.strategy.df_indicators.index
        default_series = pd.Series(0.0, index=df_index)
        
        total_score = pd.Series(0.0, index=df_index)

        for signal_name, score_config in signals_config.items():
            if signal_name.startswith("说明"):
                continue
            
            score_value = score_config.get('score', 0) if isinstance(score_config, dict) else score_config

            # 统一从 atomic_states 或 playbook_states 获取信号
            signal_series = atomic_states.get(signal_name, playbook_states.get(signal_name))

            if signal_series is not None and not signal_series.empty:
                bonus_amount = signal_series.astype(float) * score_value
                total_score += bonus_amount
                if signal_name not in score_details_df.columns:
                    score_details_df[signal_name] = 0.0
                score_details_df[signal_name] += bonus_amount
        
        return total_score, score_details_df
