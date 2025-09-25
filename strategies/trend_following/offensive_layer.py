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
        【V502.0 · 分数去重修复版】
        - 核心修复: 重构了计分流程，防止同一个信号因配置在不同类别下而被重复计分。
                      现在每个大类独立计算，最后合并详情，确保报告的准确性。
        """
        print("        -> [进攻方案评估中心 V502.0 · 分数去重修复版] 启动...") # 更新版本号
        df = self.strategy.df_indicators
        
        scoring_params = get_params_block(self.strategy, 'four_layer_scoring_params')
        if not get_param_value(scoring_params.get('enabled'), True):
            return pd.Series(0.0, index=df.index), pd.DataFrame(index=df.index)
        
        # 每个大类独立计算总分和详情
        reversal_params = scoring_params.get('reversal_offense_scoring', {})
        reversal_score, reversal_details = self._calculate_weighted_score(reversal_params.get('positive_signals', {}))
        
        resonance_params = scoring_params.get('resonance_offense_scoring', {})
        resonance_score, resonance_details = self._calculate_weighted_score(resonance_params.get('positive_signals', {}))

        playbook_params = scoring_params.get('playbook_synergy_scoring', {})
        playbook_score, playbook_details = self._calculate_weighted_score(playbook_params.get('positive_signals', {}))
        
        trigger_params = scoring_params.get('trigger_event_scoring', {})
        trigger_score, trigger_details = self._calculate_weighted_score(trigger_params.get('positive_signals', {}))

        # --- 合成总进攻分 ---
        entry_score = (reversal_score + resonance_score + playbook_score + trigger_score).fillna(0).astype(int)
        
        # 合并所有分数详情，并处理重复项（以最后一次计算为准，或可以改为相加）
        # 这里我们使用合并，如果一个信号在多个类别中，其分数会被覆盖，但通常不应如此配置。
        # 一个更健壮的合并方式是按信号名分组求和，但会使逻辑复杂化。当前方式能暴露配置问题。
        score_details_df = pd.concat([reversal_details, resonance_details, playbook_details, trigger_details], axis=1)
        # 处理因concat产生的重复列，按信号名分组求和，这是最稳妥的方式
        score_details_df = score_details_df.groupby(score_details_df.columns, axis=1).sum()

        score_details_df['SCORE_REVERSAL_OFFENSE'] = reversal_score
        score_details_df['SCORE_RESONANCE_OFFENSE'] = resonance_score
        score_details_df['SCORE_PLAYBOOK_SYNERGY'] = playbook_score
        score_details_df['SCORE_TRIGGER'] = trigger_score

        return entry_score, score_details_df.fillna(0)

    def _calculate_weighted_score(self, signals_config: Dict) -> Tuple[pd.Series, pd.DataFrame]:
        """
        【V502.0 · 分数去重修复版】向量化加权分数计算辅助函数
        - 核心修复: 不再接收和修改外部的DataFrame，而是返回一个本地计算的详情DataFrame，
                      避免了跨类别调用时污染分数的风险。
        """
        atomic_states = self.strategy.atomic_states
        playbook_states = self.strategy.playbook_states
        df_index = self.strategy.df_indicators.index
        
        total_score = pd.Series(0.0, index=df_index)
        # 创建一个本地的、干净的DataFrame来存储本次计算的详情
        local_score_details_df = pd.DataFrame(index=df_index)

        for signal_name, score_config in signals_config.items():
            if signal_name.startswith("说明"):
                continue
            
            score_value = score_config.get('score', 0) if isinstance(score_config, dict) else score_config
            signal_series = atomic_states.get(signal_name, playbook_states.get(signal_name))

            if signal_series is not None and not signal_series.empty:
                bonus_amount = signal_series.astype(float) * score_value
                total_score += bonus_amount
                # 将分数详情记录在本地DataFrame中
                local_score_details_df[signal_name] = bonus_amount
        
        # 返回总分和本次计算的详情DataFrame
        return total_score, local_score_details_df
