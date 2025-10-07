# 文件: strategies/trend_following/offensive_layer.py
# 进攻层
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from .utils import get_params_block, get_param_value

class OffensiveLayer:
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def calculate_entry_score(self, trigger_events: Dict, bottom_context_score: pd.Series, top_context_score: pd.Series) -> Tuple[pd.Series, pd.DataFrame]:
        """
        【V515.1 · 时区统一法案版】
        - 核心修正: 签署“时区统一法案”，在“最终验尸”探针中强制统一时间戳的时区信息，确保探针能够命中目标。
        - 收益: 修复了因时区不匹配导致探针完全失效的致命BUG，让分数构成重见天日。
        """
        df = self.strategy.df_indicators
        score_details_df = pd.DataFrame(index=df.index)
        score_map = get_params_block(self.strategy, 'score_type_map', {})
        atomic_states = self.strategy.atomic_states
        playbook_states = self.strategy.playbook_states
        total_score = pd.Series(0.0, index=df.index)
        p_context_suppression = get_params_block(self.strategy, 'contextual_suppression_params', {})
        bottom_context_threshold = get_param_value(p_context_suppression.get('bottom_context_threshold'), 0.9)
        top_context_threshold = get_param_value(p_context_suppression.get('top_context_threshold'), 0.9)
        # [代码修改] 应用“时区统一法案”，确保探针能够命中目标
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        probe_dates_naive = [pd.to_datetime(d) for d in probe_dates_str]
        probe_dates = []
        if df.index.tz is not None:
            # 如果 df.index 具有时区信息，则将探针日期本地化到相同的时区
            for d in probe_dates_naive:
                try:
                    probe_dates.append(d.tz_localize(df.index.tz))
                except TypeError: # 如果日期已经有时区，则尝试转换
                    probe_dates.append(d.tz_convert(df.index.tz))
        else:
            # 如果 df.index 没有时区信息，则保持探针日期为 naive
            probe_dates = probe_dates_naive
        # [代码修改] 最终验尸探针 - 打印表头
        for date in probe_dates:
            if date in df.index:
                print(f"\n" + "="*30 + f" [最终验尸 @ {date.date()}] " + "="*30)
                print(f"  {'信号名称':<50} {'贡献分数':>15}")
                print(f"  {'-'*50} {'-'*15}")
        for signal_name, meta in score_map.items():
            if not isinstance(meta, dict): continue
            score_value = meta.get('score', 0)
            if score_value != 0:
                signal_series = atomic_states.get(signal_name, playbook_states.get(signal_name))
                if signal_series is not None and not signal_series.empty:
                    processed_signal_series = signal_series.astype(float)
                    context_role = meta.get('context_role', 'neutral')
                    if context_role == 'top_risk' and score_value < 0:
                        suppression_factor = bottom_context_score.where(bottom_context_score >= bottom_context_threshold, 0.0)
                        damper = 1.0 - suppression_factor
                        processed_signal_series *= damper
                    elif context_role == 'bottom_opportunity' and score_value > 0:
                        suppression_factor = top_context_score.where(top_context_score >= top_context_threshold, 0.0)
                        damper = 1.0 - suppression_factor
                        processed_signal_series *= damper
                    scoring_mode = meta.get('scoring_mode', 'bipolar')
                    if scoring_mode == 'unipolar':
                        processed_signal_series = processed_signal_series.clip(lower=0)
                    bonus_amount = processed_signal_series * score_value
                    total_score += bonus_amount
                    score_details_df[signal_name] = bonus_amount
                    # [代码修改] 在循环内部进行验尸打印
                    for date in probe_dates:
                        if date in bonus_amount.index:
                            contribution = bonus_amount.loc[date]
                            if pd.notna(contribution) and contribution != 0:
                                print(f"  {signal_name:<50} {contribution:>15.2f}")
        # [代码修改] 最终验尸探针 - 打印表尾
        for date in probe_dates:
             if date in df.index:
                print(f"  {'-'*50} {'-'*15}")
                print(f"  {'总计':<50} {total_score.loc[date]:>15.2f}")
                print("="*77 + "\n")
        # 保持浮点数以便观察，最终的取整在 judgment_layer 中完成
        return total_score.fillna(0), score_details_df.fillna(0)


















