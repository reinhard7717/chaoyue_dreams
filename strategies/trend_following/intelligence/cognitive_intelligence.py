# 文件: strategies/trend_following/intelligence/cognitive_intelligence.py
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any
from enum import Enum
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score, normalize_to_bipolar, is_limit_up

class CognitiveIntelligence:
    """
    - 核心重构: 废弃旧的、分散的信号合成方法，引入统一的“贝叶斯战术推演”框架。
    - 核心思想: 将A股的复杂博弈场景抽象为一系列“战术剧本”。引擎不再是简单地叠加信号，
                  而是基于融合层提供的“战场态势”（先验信念），结合原子层的“微观证据”（似然度），
                  通过贝叶斯推演，计算出每个战术剧本上演的“后验概率”（最终信号分）。
    - 收益: 使认知层的每一个判断都有清晰的数学逻辑和博弈论基础，直指A股本质。
    """
    def __init__(self, strategy_instance, dynamic_thresholds: Dict = None):
        """
        初始化认知情报模块。
        :param strategy_instance: 策略主实例的引用。
        :param dynamic_thresholds: 动态阈值字典。
        """
        self.strategy = strategy_instance
        # 修改开始 - 清理冗余探针，并确保 debug_params 和 dynamic_thresholds 的正确加载
        # 模仿 StructuralIntelligence 的 debug_params 加载方式
        # 假设 get_params_block 能够从 self.strategy 对象中正确解析出配置
        debug_params_config = get_params_block(self.strategy, 'debug_params', {})
        self.probe_dates_list_str = debug_params_config.get('probe_dates', [])
        self.debug_enabled = debug_params_config.get('enabled', {}).get('value', False)
        # 如果 dynamic_thresholds 没有被传递，则从配置中加载
        if dynamic_thresholds is None:
            # 确保 full_config_dict_for_dynamic_thresholds 是 self.strategy.params
            full_config_dict_for_dynamic_thresholds = {}
            if hasattr(self.strategy, 'params') and isinstance(self.strategy.params, dict):
                full_config_dict_for_dynamic_thresholds = self.strategy.params
            elif hasattr(self.strategy, 'config') and isinstance(self.strategy.config, dict):
                full_config_dict_for_dynamic_thresholds = self.strategy.config
            self.dynamic_thresholds = get_params_block(full_config_dict_for_dynamic_thresholds, 'strategy_params.trend_follow.dynamic_thresholds', {})
        else:
            self.dynamic_thresholds = dynamic_thresholds
        # 修改结束

    def _get_safe_series(self, df: pd.DataFrame, column_name: str, default_value: Any = 0.0, method_name: str = "未知方法") -> pd.Series:
        """
        安全地从DataFrame获取Series，如果不存在则打印警告并返回默认Series。
        【V27.1 · 返回值修复版】修复了在信号不存在时错误返回索引而非Series的问题。
        """
        if column_name not in df.columns:
            print(f"    -> [CognitiveIntelligence情报警告] 方法 '{method_name}' 缺少数据 '{column_name}'，使用默认值 {default_value}。")
            return pd.Series(default_value, index=df.index) # 移除了末尾的 .index
        return df[column_name]

    def _get_fused_score(self, df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
        """
        【V1.4 · 返回值修复版】安全地从原子状态库中获取由融合层提供的态势分数。
        - 【V1.4 修复】修复了在信号不存在时错误返回索引而非Series的问题。
        """
        if name in self.strategy.atomic_states:
            score = self.strategy.atomic_states[name]
            return score
        else:
            print(f"    -> [认知层警告] 融合态势信号 '{name}' 不存在，无法作为证据！返回默认值 {default}。")
            return pd.Series(default, index=df.index) # 移除了末尾的 .index

    def _get_atomic_score(self, df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
        """
        【V2.3 · 返回值修复版】安全地从原子状态库或主数据帧中获取信号。
        - 【V2.3 修复】修复了在信号不存在时错误返回索引而非Series的问题。
        """
        score = None
        if name in self.strategy.atomic_states:
            score = self.strategy.atomic_states[name]
        elif name in df.columns:
            score = df[name]
        else:
            print(f"    -> [认知层警告] 原子信号 '{name}' 不存在，无法作为证据！返回默认值 {default}。")
            score = pd.Series(default, index=df.index)
        # 修改开始 - 使用实例属性 self.debug_enabled 和 self.probe_dates_list_str
        if self.debug_enabled and self.probe_dates_list_str:
            if not df.empty:
                df_index_tz = df.index.tz
                for date_str in self.probe_dates_list_str:
                    try:
                        probe_date_naive = pd.to_datetime(date_str)
                        probe_date_for_loop = probe_date_naive.tz_localize(df_index_tz) if df_index_tz else probe_date_naive
                        if probe_date_for_loop is not None and probe_date_for_loop in df.index:
                            if isinstance(score, pd.Series):
                                print(f"    -> [DEBUG _get_atomic_score] 信号 '{name}' 在 {probe_date_for_loop.strftime('%Y-%m-%d')} 原始值: {score.loc[probe_date_for_loop]:.4f}")
                            else:
                                print(f"    -> [DEBUG _get_atomic_score] 信号 '{name}' 在 {probe_date_for_loop.strftime('%Y-%m-%d')} 原始值: {score:.4f} (非Series)")
                    except Exception:
                        pass
        # 修改结束
        return score

    def _get_playbook_score(self, df: pd.DataFrame, signal_name: str, default_value: float = 0.0) -> pd.Series:
        """
        安全地从 playbook_states 获取剧本信号分数。
        【V27.1 · 返回值修复版】修复了在信号不存在时错误返回索引而非Series的问题。
        """
        score = self.strategy.playbook_states.get(signal_name)
        if score is None:
            print(f"    -> [认知层警告] 剧本信号 '{signal_name}' 不存在，无法作为证据！返回默认值 {default_value}。")
            return pd.Series(default_value, index=df.index) # 移除了末尾的 .index
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df.index.tz) if df.index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df.index:
                if isinstance(score, pd.Series):
                    print(f"    -> [DEBUG _get_playbook_score] 信号 '{signal_name}' 原始值: {score.loc[probe_date_for_loop]:.4f}")
                else:
                    print(f"    -> [DEBUG _get_playbook_score] 信号 '{signal_name}' 原始值: {score:.4f}")
        return score

    def synthesize_cognitive_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        cognitive_scores = pd.DataFrame(index=df.index)
        cognitive_intelligence_config = get_params_block(self.strategy, 'cognitive_intelligence_params', {})
        playbooks_config = cognitive_intelligence_config.get('playbooks', {})
        if playbooks_config.get('cognitive_playbook_suppressive_accumulation_params'):
            cognitive_scores["COGNITIVE_PLAYBOOK_SUPPRESSIVE_ACCUMULATION"] = self._calculate_suppressive_accumulation(df)
        if playbooks_config.get('cognitive_playbook_chasing_accumulation_params'):
            cognitive_scores["COGNITIVE_PLAYBOOK_CHASING_ACCUMULATION"] = self._calculate_chasing_accumulation(df)
        if playbooks_config.get('cognitive_playbook_capitulation_reversal_params'):
            cognitive_scores["COGNITIVE_PLAYBOOK_CAPITULATION_REVERSAL"] = self._calculate_capitulation_reversal(df)
        if playbooks_config.get('cognitive_risk_distribution_at_high_params'):
            cognitive_scores["COGNITIVE_RISK_DISTRIBUTION_AT_HIGH"] = self._calculate_distribution_at_high(df)
        if playbooks_config.get('cognitive_playbook_leading_dragon_awakening_params'):
            cognitive_scores["COGNITIVE_PLAYBOOK_LEADING_DRAGON_AWAKENING"] = self._calculate_leading_dragon_awakening(df)
        # 修改开始 - 新增 COGNITIVE_PLAYBOOK_ENERGY_COMPRESSION 的调用
        if playbooks_config.get('cognitive_playbook_energy_compression_params'):
            cognitive_scores["COGNITIVE_PLAYBOOK_ENERGY_COMPRESSION"] = self._calculate_energy_compression(df)
        # 修改结束
        return cognitive_scores

    def _calculate_suppressive_accumulation(self, df: pd.DataFrame) -> pd.Series:
        method_name = "COGNITIVE_PLAYBOOK_SUPPRESSIVE_ACCUMULATION"
        print(f"  -> [认知层] 正在计算 {method_name}...")
        # 确定配置根字典
        full_config_dict = {}
        if hasattr(self.strategy, 'params') and isinstance(self.strategy.params, dict):
            full_config_dict = self.strategy.params
        elif hasattr(self.strategy, 'config') and isinstance(self.strategy.config, dict):
            full_config_dict = self.strategy.config
        else:
            pass 
        # 修正参数加载路径，使用 get_params_block 获取顶层块，然后使用 .get() 获取嵌套块
        cognitive_intelligence_config = get_params_block(self.strategy, 'cognitive_intelligence_params', {})
        params = cognitive_intelligence_config.get('playbooks', {}).get('cognitive_playbook_suppressive_accumulation_params', {})
        # 修改开始 - 移除 enabled 状态检查
        # if not get_param_value(params.get('enabled'), False):
        #     return pd.Series(0.0, index=df.index, dtype=np.float32)
        # 修改结束
        suppression_weights = get_param_value(params.get('suppression_weights'), {})
        accumulation_weights = get_param_value(params.get('accumulation_weights'), {})
        contradiction_weights = get_param_value(params.get('contradiction_weights'), {})
        context_modulator_weights = get_param_value(params.get('context_modulator_weights'), {})
        final_fusion_exponent = get_param_value(params.get('final_fusion_exponent'), 2.0)
        min_activation_threshold = get_param_value(params.get('min_activation_threshold'), 0.1)
        norm_window = get_param_value(params.get('norm_window'), 55)
        # 移除对 SCORE_CONTEXT_DEEP_BOTTOM_ZONE 的依赖，并调整权重
        if 'SCORE_CONTEXT_DEEP_BOTTOM_ZONE' in context_modulator_weights:
            removed_weight = context_modulator_weights.pop('SCORE_CONTEXT_DEEP_BOTTOM_ZONE')
            remaining_total_weight = sum(v for k, v in context_modulator_weights.items() if k != 'description' and isinstance(v, (int, float)))
            if remaining_total_weight > 0:
                for k in context_modulator_weights:
                    if k != 'description' and isinstance(context_modulator_weights[k], (int, float)):
                        context_modulator_weights[k] += context_modulator_weights[k] / remaining_total_weight * removed_weight
            else:
                context_modulator_weights = {}
        current_total_context_weight = sum(v for k, v in context_modulator_weights.items() if k != 'description' and isinstance(v, (int, float)))
        if current_total_context_weight > 0 and abs(current_total_context_weight - 1.0) > 1e-6:
            context_modulator_weights = {k: v / current_total_context_weight for k, v in context_modulator_weights.items() if k != 'description' and isinstance(v, (int, float))}
        all_required_signals = set()
        all_required_signals.update(suppression_weights.keys())
        all_required_signals.update(accumulation_weights.keys())
        all_required_signals.update(contradiction_weights.keys())
        all_required_signals.update(context_modulator_weights.keys())
        fetched_signals = {}
        for signal_name in all_required_signals:
            if signal_name == 'description':
                continue
            fetched_signals[signal_name] = self._get_atomic_score(df, signal_name, default=0.0)
            if not isinstance(fetched_signals[signal_name], pd.Series):
                fetched_signals[signal_name] = pd.Series(fetched_signals[signal_name], index=df.index)
            else:
                fetched_signals[signal_name] = fetched_signals[signal_name].reindex(df.index).fillna(0.0)
        suppression_score_components = pd.Series(0.0, index=df.index)
        accumulation_score_components = pd.Series(0.0, index=df.index)
        contradiction_score_components = pd.Series(0.0, index=df.index)
        context_modulator_score_components = pd.Series(0.0, index=df.index)
        total_suppression_weight = sum(v for k, v in suppression_weights.items() if k != 'description' and isinstance(v, (int, float)))
        if total_suppression_weight > 0:
            for signal_name, weight in suppression_weights.items():
                if signal_name == 'description':
                    continue
                raw_signal = fetched_signals[signal_name]
                if "PRICE_DOWNWARD_MOMENTUM" in signal_name or \
                   "DISTRIBUTION_INTENT" in signal_name or \
                   "STAGNATION_EVIDENCE_RAW" in signal_name or \
                   "FUSION_RISK_DISTRIBUTION_PRESSURE" in signal_name:
                    signal_score = raw_signal.clip(lower=0)
                elif "FF_AXIOM_CONSENSUS" in signal_name:
                    signal_score = raw_signal.clip(upper=0).abs()
                elif "TREND_FORM" in signal_name or \
                     "LIQUIDITY_TIDE" in signal_name:
                    signal_score = (1 - raw_signal).clip(lower=0)
                else:
                    signal_score = raw_signal.clip(lower=0)
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                suppression_score_components += normalized_signal_score * weight
            suppression_score = suppression_score_components / total_suppression_weight
        else:
            suppression_score = pd.Series(0.0, index=df.index)
        total_accumulation_weight = sum(v for k, v in accumulation_weights.items() if k != 'description' and isinstance(v, (int, float)))
        if total_accumulation_weight > 0:
            for signal_name, weight in accumulation_weights.items():
                if signal_name == 'description':
                    continue
                raw_signal = fetched_signals[signal_name]
                if "STEALTH_ACCUMULATION" in signal_name or \
                   "PANIC_WASHOUT_ACCUMULATION" in signal_name or \
                   "DECEPTIVE_ACCUMULATION" in signal_name or \
                   "ABSORPTION_ECHO" in signal_name:
                    signal_score = raw_signal.clip(lower=0)
                elif "FF_AXIOM_CONVICTION" in signal_name:
                    signal_score = raw_signal.clip(upper=0).abs()
                else:
                    signal_score = raw_signal.clip(lower=0)
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                accumulation_score_components += normalized_signal_score * weight
            accumulation_score = accumulation_score_components / total_accumulation_weight
        else:
            accumulation_score = pd.Series(0.0, index=df.index)
        total_contradiction_weight = sum(v for k, v in contradiction_weights.items() if k != 'description' and isinstance(v, (int, float)))
        if total_contradiction_weight > 0:
            for signal_name, weight in contradiction_weights.items():
                if signal_name == 'description':
                    continue
                raw_signal = fetched_signals[signal_name]
                if "BULLISH_DIVERGENCE" in signal_name or \
                   "CHIP_AXIOM_DIVERGENCE" in signal_name or \
                   "FUND_FLOW_BULLISH_DIVERGENCE" in signal_name or \
                   "PRICE_VS_RETAIL_CAPITULATION" in signal_name or \
                   "PROFIT_VS_FLOW" in signal_name:
                    signal_score = raw_signal.clip(upper=0).abs()
                else:
                    signal_score = raw_signal.clip(lower=0)
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                contradiction_score_components += normalized_signal_score * weight
            contradiction_score = contradiction_score_components / total_contradiction_weight
        else:
            contradiction_score = pd.Series(0.0, index=df.index)
        total_context_weight = sum(v for k, v in context_modulator_weights.items() if k != 'description' and isinstance(v, (int, float)))
        if total_context_weight > 0:
            for signal_name, weight in context_modulator_weights.items():
                if signal_name == 'description':
                    continue
                raw_signal = fetched_signals[signal_name]
                signal_score = raw_signal.clip(lower=0)
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                context_modulator_score_components += normalized_signal_score * weight
            context_modulator = context_modulator_score_components / total_context_weight
        else:
            context_modulator = pd.Series(1.0, index=df.index)
        epsilon = 1e-6
        fused_score_raw = (
            (suppression_score + epsilon) *
            (accumulation_score + epsilon) *
            (contradiction_score + epsilon)
        )**(1/3)
        final_score = (fused_score_raw * context_modulator)**final_fusion_exponent
        final_score = final_score.clip(0, 1)
        final_score = final_score.where(final_score >= min_activation_threshold, 0.0)
        print(f"  -> {method_name} 计算完成。")
        return final_score.astype(np.float32)

    def _calculate_chasing_accumulation(self, df: pd.DataFrame) -> pd.Series:
        method_name = "COGNITIVE_PLAYBOOK_CHASING_ACCUMULATION"
        print(f"  -> [认知层] 正在计算 {method_name}...")
        cognitive_intelligence_config = get_params_block(self.strategy, 'cognitive_intelligence_params', {})
        params = cognitive_intelligence_config.get('playbooks', {}).get('cognitive_playbook_chasing_accumulation_params', {})
        chasing_weights = get_param_value(params.get('chasing_weights'), {})
        accumulation_quality_weights = get_param_value(params.get('accumulation_quality_weights'), {})
        context_weights = get_param_value(params.get('context_weights'), {})
        risk_filter_weights = get_param_value(params.get('risk_filter_weights'), {})
        confirmation_contradiction_weights = get_param_value(params.get('confirmation_contradiction_weights'), {})
        purity_penalty_sensitivity = get_param_value(params.get('purity_penalty_sensitivity'), 0.5)
        synergy_threshold = get_param_value(params.get('synergy_threshold'), 0.6)
        synergy_bonus_factor = get_param_value(params.get('synergy_bonus_factor'), 0.2)
        conflict_penalty_factor = get_param_value(params.get('conflict_penalty_factor'), 0.3)
        dynamic_context_modulator_signal = get_param_value(params.get('dynamic_context_modulator_signal'), "SCORE_FOUNDATION_AXIOM_MARKET_CONSTITUTION")
        dynamic_context_sensitivity = get_param_value(params.get('dynamic_context_sensitivity'), 0.5)
        final_fusion_exponent = get_param_value(params.get('final_fusion_exponent'), 2.0)
        min_activation_threshold = get_param_value(params.get('min_activation_threshold'), 0.1)
        norm_window = get_param_value(params.get('norm_window'), 55)
        all_required_signals = set()
        all_required_signals.update(chasing_weights.keys())
        all_required_signals.update(accumulation_quality_weights.keys())
        all_required_signals.update(context_weights.keys())
        all_required_signals.update(risk_filter_weights.keys())
        all_required_signals.update(confirmation_contradiction_weights.keys())
        all_required_signals.add("SCORE_BEHAVIOR_DECEPTION_INDEX")
        all_required_signals.add("SCORE_CHIP_RISK_DISTRIBUTION_WHISPER")
        all_required_signals.add(dynamic_context_modulator_signal)
        fetched_signals = {}
        for signal_name in all_required_signals:
            if signal_name == 'description':
                continue
            fetched_signals[signal_name] = self._get_atomic_score(df, signal_name, default=0.0)
            if not isinstance(fetched_signals[signal_name], pd.Series):
                fetched_signals[signal_name] = pd.Series(fetched_signals[signal_name], index=df.index)
            else:
                fetched_signals[signal_name] = fetched_signals[signal_name].reindex(df.index).fillna(0.0)
        chasing_score_components = pd.Series(0.0, index=df.index)
        accumulation_quality_score_components = pd.Series(0.0, index=df.index)
        context_score_components = pd.Series(0.0, index=df.index)
        risk_filter_score_components = pd.Series(0.0, index=df.index)
        confirmation_contradiction_score_components = pd.Series(0.0, index=df.index)
        deception_index = fetched_signals.get("SCORE_BEHAVIOR_DECEPTION_INDEX", pd.Series(0.0, index=df.index))
        distribution_whisper = fetched_signals.get("SCORE_CHIP_RISK_DISTRIBUTION_WHISPER", pd.Series(0.0, index=df.index))
        deception_penalty = deception_index.clip(upper=0).abs() * purity_penalty_sensitivity
        distribution_penalty = distribution_whisper * purity_penalty_sensitivity
        market_constitution_score = fetched_signals.get(dynamic_context_modulator_signal, pd.Series(0.5, index=df.index))
        normalized_market_constitution = normalize_score(market_constitution_score, df.index, norm_window, ascending=True)
        chasing_dynamic_modulator = 1 + (normalized_market_constitution - 0.5) * dynamic_context_sensitivity * 2
        accumulation_dynamic_modulator = 1 - (normalized_market_constitution - 0.5) * dynamic_context_sensitivity * 2
        chasing_dynamic_modulator = chasing_dynamic_modulator.clip(0.5, 1.5)
        accumulation_dynamic_modulator = accumulation_dynamic_modulator.clip(0.5, 1.5)
        total_chasing_weight = sum(v for k, v in chasing_weights.items() if k != 'description' and isinstance(v, (int, float)))
        if total_chasing_weight > 0:
            for signal_name, weight in chasing_weights.items():
                if signal_name == 'description':
                    continue
                raw_signal = fetched_signals[signal_name]
                signal_score = raw_signal.clip(lower=0)
                if "PRICE_UPWARD_MOMENTUM" in signal_name or "VOLUME_BURST" in signal_name or "UPWARD_EFFICIENCY" in signal_name or \
                   "MICRO_STRATEGY_SHOCK_AND_AWE" in signal_name or "INTRADAY_OFFENSIVE_PURITY" in signal_name or \
                   "PROCESS_META_BREAKOUT_ACCELERATION" in signal_name or "PROCESS_META_MAIN_FORCE_RALLY_INTENT" in signal_name or \
                   "SCORE_PATTERN_AXIOM_BREAKOUT" in signal_name:
                    signal_score = signal_score * (1 - deception_penalty) * (1 - distribution_penalty)
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                chasing_score_components += normalized_signal_score * weight
            chasing_score = (chasing_score_components / total_chasing_weight) * chasing_dynamic_modulator
        else:
            chasing_score = pd.Series(0.0, index=df.index)
        total_accumulation_quality_weight = sum(v for k, v in accumulation_quality_weights.items() if k != 'description' and isinstance(v, (int, float)))
        if total_accumulation_quality_weight > 0:
            for signal_name, weight in accumulation_quality_weights.items():
                if signal_name == 'description':
                    continue
                raw_signal = fetched_signals[signal_name]
                if "SCORE_CHIP_STRATEGIC_POSTURE" in signal_name or \
                   "SCORE_FF_AXIOM_CONSENSUS" in signal_name or \
                   "SCORE_FF_AXIOM_CONVICTION" in signal_name or \
                   "PROCESS_META_COST_ADVANTAGE_TREND" in signal_name or \
                   "PROCESS_META_PROFIT_VS_FLOW" in signal_name:
                    signal_score = raw_signal.clip(lower=0)
                else:
                    signal_score = raw_signal.clip(lower=0)
                if "SCORE_CHIP_COHERENT_DRIVE" in signal_name or "PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY" in signal_name or \
                   "PROCESS_META_FUND_FLOW_ACCUMULATION_INFLECTION_INTENT" in signal_name or "SCORE_CHIP_TACTICAL_EXCHANGE" in signal_name:
                    signal_score = signal_score * (1 - distribution_penalty)
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                accumulation_quality_score_components += normalized_signal_score * weight
            accumulation_quality_score = (accumulation_quality_score_components / total_accumulation_quality_weight) * accumulation_dynamic_modulator
        else:
            accumulation_quality_score = pd.Series(0.0, index=df.index)
        total_context_weight = sum(v for k, v in context_weights.items() if k != 'description' and isinstance(v, (int, float)))
        if total_context_weight > 0:
            for signal_name, weight in context_weights.items():
                if signal_name == 'description':
                    continue
                raw_signal = fetched_signals[signal_name]
                if "FUSION_BIPOLAR_TREND_QUALITY" in signal_name or \
                   "FUSION_BIPOLAR_FUND_FLOW_TREND" in signal_name or \
                   "FUSION_BIPOLAR_CHIP_TREND" in signal_name or \
                   "FUSION_BIPOLAR_LIQUIDITY_DYNAMICS" in signal_name:
                    signal_score = raw_signal.clip(lower=0)
                else:
                    signal_score = raw_signal.clip(lower=0)
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                context_score_components += normalized_signal_score * weight
            context_score = context_score_components / total_context_weight
        else:
            context_score = pd.Series(0.0, index=df.index)
        total_risk_filter_weight = sum(v for k, v in risk_filter_weights.items() if k != 'description' and isinstance(v, (int, float)))
        if total_risk_filter_weight > 0:
            for signal_name, weight in risk_filter_weights.items():
                if signal_name == 'description':
                    continue
                raw_signal = fetched_signals[signal_name]
                if "SCORE_BEHAVIOR_DECEPTION_INDEX" in signal_name or \
                   "SCORE_FF_AXIOM_DIVERGENCE" in signal_name or \
                   "SCORE_BEHAVIOR_BEARISH_DIVERGENCE" in signal_name or \
                   "SCORE_FUND_FLOW_BEARISH_DIVERGENCE" in signal_name:
                    signal_score = raw_signal.clip(upper=0).abs()
                else:
                    signal_score = raw_signal.clip(lower=0)
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                risk_filter_score_components += normalized_signal_score * weight
            risk_filter_score = risk_filter_score_components / total_risk_filter_weight
        else:
            risk_filter_score = pd.Series(0.0, index=df.index)
        total_confirmation_contradiction_weight = sum(v for k, v in confirmation_contradiction_weights.items() if k != 'description' and isinstance(v, (int, float)))
        if total_confirmation_contradiction_weight > 0:
            for signal_name, weight in confirmation_contradiction_weights.items():
                if signal_name == 'description':
                    continue
                raw_signal = fetched_signals[signal_name]
                if "PROCESS_META_PRICE_VS_RETAIL_CAPITULATION" in signal_name:
                    signal_score = raw_signal.clip(upper=0).abs()
                else:
                    signal_score = raw_signal.clip(lower=0)
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                confirmation_contradiction_score_components += normalized_signal_score * weight
            confirmation_contradiction_score = confirmation_contradiction_score_components / total_confirmation_contradiction_weight
        else:
            confirmation_contradiction_score = pd.Series(0.0, index=df.index)
        synergy_factor = pd.Series(1.0, index=df.index)
        synergy_condition = (chasing_score > synergy_threshold) & \
                            (accumulation_quality_score > synergy_threshold) & \
                            (risk_filter_score < (1 - synergy_threshold))
        synergy_factor.loc[synergy_condition] += synergy_bonus_factor
        conflict_condition = (chasing_score > synergy_threshold) & \
                             (risk_filter_score > synergy_threshold)
        synergy_factor.loc[conflict_condition] -= conflict_penalty_factor
        synergy_factor = synergy_factor.clip(0.5, 1.5)
        epsilon = 1e-6
        fused_score_raw = (
            (chasing_score + epsilon) *
            (accumulation_quality_score + epsilon) *
            (context_score + epsilon) *
            (confirmation_contradiction_score + epsilon)
        )**(1/4)
        fused_score_raw = fused_score_raw * synergy_factor
        risk_adjusted_fused_score = fused_score_raw * (1 - risk_filter_score.clip(0, 1))
        final_score = (risk_adjusted_fused_score)**final_fusion_exponent
        final_score = final_score.clip(0, 1)
        final_score = final_score.where(final_score >= min_activation_threshold, 0.0)
        print(f"  -> {method_name} 计算完成。")
        return final_score.astype(np.float32)

    def _calculate_capitulation_reversal(self, df: pd.DataFrame) -> pd.Series:
        method_name = "COGNITIVE_PLAYBOOK_CAPITULATION_REVERSAL"
        print(f"  -> [认知层] 正在计算 {method_name}...")
        cognitive_intelligence_config = get_params_block(self.strategy, 'cognitive_intelligence_params', {})
        params = cognitive_intelligence_config.get('playbooks', {}).get('cognitive_playbook_capitulation_reversal_params', {})
        panic_evidence_weights = get_param_value(params.get('panic_evidence_weights'), {})
        absorption_quality_weights = get_param_value(params.get('absorption_quality_weights'), {})
        reversal_intent_weights = get_param_value(params.get('reversal_intent_weights'), {})
        context_reinforcement_weights = get_param_value(params.get('context_reinforcement_weights'), {})
        risk_filter_weights = get_param_value(params.get('risk_filter_weights'), {})
        confirmation_contradiction_weights = get_param_value(params.get('confirmation_contradiction_weights'), {})
        purity_penalty_sensitivity = get_param_value(params.get('purity_penalty_sensitivity'), 0.5)
        synergy_threshold = get_param_value(params.get('synergy_threshold'), 0.6)
        synergy_bonus_factor = get_param_value(params.get('synergy_bonus_factor'), 0.2)
        conflict_penalty_factor = get_param_value(params.get('conflict_penalty_factor'), 0.3)
        dynamic_context_modulator_signal = get_param_value(params.get('dynamic_context_modulator_signal'), "SCORE_CYCLICAL_HURST_REVERSION_REGIME")
        dynamic_context_sensitivity = get_param_value(params.get('dynamic_context_sensitivity'), 0.5)
        final_fusion_exponent = get_param_value(params.get('final_fusion_exponent'), 2.0)
        min_activation_threshold = get_param_value(params.get('min_activation_threshold'), 0.1)
        norm_window = get_param_value(params.get('norm_window'), 55)
        all_required_signals = set()
        all_required_signals.update(panic_evidence_weights.keys())
        all_required_signals.update(absorption_quality_weights.keys())
        all_required_signals.update(reversal_intent_weights.keys())
        all_required_signals.update(context_reinforcement_weights.keys())
        all_required_signals.update(risk_filter_weights.keys())
        all_required_signals.update(confirmation_contradiction_weights.keys())
        all_required_signals.add("SCORE_BEHAVIOR_DECEPTION_INDEX")
        all_required_signals.add("SCORE_CHIP_RISK_DISTRIBUTION_WHISPER")
        all_required_signals.add(dynamic_context_modulator_signal)
        all_required_signals.discard("SLOPE_5_SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM")
        all_required_signals.discard("ACCEL_5_SCORE_BEHAVIOR_ABSORPTION_STRENGTH")
        all_required_signals.add("SCORE_CHIP_COHERENT_DRIVE")
        all_required_signals.add("SCORE_BEHAVIOR_BULLISH_DIVERGENCE_QUALITY")
        fetched_signals = {}
        for signal_name in all_required_signals:
            if signal_name == 'description':
                continue
            fetched_signals[signal_name] = self._get_atomic_score(df, signal_name, default=0.0)
            if not isinstance(fetched_signals[signal_name], pd.Series):
                fetched_signals[signal_name] = pd.Series(fetched_signals[signal_name], index=df.index)
            else:
                fetched_signals[signal_name] = fetched_signals[signal_name].reindex(df.index).fillna(0.0)
        panic_evidence_score_components = pd.Series(0.0, index=df.index)
        absorption_quality_score_components = pd.Series(0.0, index=df.index)
        reversal_intent_score_components = pd.Series(0.0, index=df.index)
        context_reinforcement_score_components = pd.Series(0.0, index=df.index)
        risk_filter_score_components = pd.Series(0.0, index=df.index)
        confirmation_contradiction_score_components = pd.Series(0.0, index=df.index)
        deception_index = fetched_signals.get("SCORE_BEHAVIOR_DECEPTION_INDEX", pd.Series(0.0, index=df.index))
        distribution_whisper = fetched_signals.get("SCORE_CHIP_RISK_DISTRIBUTION_WHISPER", pd.Series(0.0, index=df.index))
        deception_penalty_for_panic = deception_index.clip(lower=0) * purity_penalty_sensitivity
        distribution_penalty_for_absorption = distribution_whisper * purity_penalty_sensitivity
        market_context_score = fetched_signals.get(dynamic_context_modulator_signal, pd.Series(0.5, index=df.index))
        normalized_market_context = normalize_score(market_context_score, df.index, norm_window, ascending=True)
        panic_dynamic_modulator = 1 + (normalized_market_context - 0.5) * dynamic_context_sensitivity * 2
        absorption_dynamic_modulator = 1 + (normalized_market_context - 0.5) * dynamic_context_sensitivity * 2
        reversal_dynamic_modulator = 1 + (normalized_market_context - 0.5) * dynamic_context_sensitivity * 2
        panic_dynamic_modulator = panic_dynamic_modulator.clip(0.5, 1.5)
        absorption_dynamic_modulator = absorption_dynamic_modulator.clip(0.5, 1.5)
        reversal_dynamic_modulator = reversal_dynamic_modulator.clip(0.5, 1.5)
        total_panic_evidence_weight = sum(v for k, v in panic_evidence_weights.items() if k != 'description' and isinstance(v, (int, float)))
        if total_panic_evidence_weight > 0:
            for signal_name, weight in panic_evidence_weights.items():
                if signal_name == 'description':
                    continue
                raw_signal = fetched_signals[signal_name]
                if "SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM" in signal_name:
                    signal_score = raw_signal.clip(upper=0).abs()
                elif "SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM" in signal_name or \
                     "FUSION_RISK_STAGNATION" in signal_name or \
                     "FUSION_RISK_DISTRIBUTION_PRESSURE" in signal_name or \
                     "PREDICTIVE_OPP_CAPITULATION_REVERSAL" in signal_name or \
                     "PROCESS_META_LOSER_CAPITULATION" in signal_name or \
                     "SCORE_BEHAVIOR_VOLUME_ATROPHY" in signal_name:
                    signal_score = raw_signal.clip(lower=0)
                else:
                    signal_score = raw_signal.clip(lower=0)
                if "PRICE_DOWNWARD_MOMENTUM" in signal_name or "FUSION_RISK" in signal_name:
                    signal_score = signal_score * (1 - deception_penalty_for_panic)
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                panic_evidence_score_components += normalized_signal_score * weight
            panic_score = (panic_evidence_score_components / total_panic_evidence_weight) * panic_dynamic_modulator
        else:
            panic_score = pd.Series(0.0, index=df.index)
        total_absorption_quality_weight = sum(v for k, v in absorption_quality_weights.items() if k != 'description' and isinstance(v, (int, float)))
        if total_absorption_quality_weight > 0:
            for signal_name, weight in absorption_quality_weights.items():
                if signal_name == 'description':
                    continue
                raw_signal = fetched_signals[signal_name]
                if "SCORE_BEHAVIOR_OFFENSIVE_ABSORPTION_INTENT" in signal_name or \
                   "SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION" in signal_name or \
                   "SCORE_CHIP_OPP_ABSORPTION_ECHO" in signal_name or \
                   "SCORE_BEHAVIOR_ABSORPTION_STRENGTH" in signal_name or \
                   "PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY" in signal_name or \
                   "SCORE_CHIP_TACTICAL_EXCHANGE" in signal_name or \
                   "SCORE_CHIP_COHERENT_DRIVE" in signal_name:
                    signal_score = raw_signal.clip(lower=0)
                else:
                    signal_score = raw_signal.clip(lower=0)
                if "SCORE_CHIP_OPP_ABSORPTION_ECHO" in signal_name or "SCORE_BEHAVIOR_ABSORPTION_STRENGTH" in signal_name or \
                   "PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY" in signal_name or "SCORE_CHIP_TACTICAL_EXCHANGE" in signal_name or \
                   "SCORE_CHIP_COHERENT_DRIVE" in signal_name:
                    signal_score = signal_score * (1 - distribution_penalty_for_absorption)
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                absorption_quality_score_components += normalized_signal_score * weight
            absorption_score = (absorption_quality_score_components / total_absorption_quality_weight) * absorption_dynamic_modulator
        else:
            absorption_score = pd.Series(0.0, index=df.index)
        total_reversal_intent_weight = sum(v for k, v in reversal_intent_weights.items() if k != 'description' and isinstance(v, (int, float)))
        if total_reversal_intent_weight > 0:
            for signal_name, weight in reversal_intent_weights.items():
                if signal_name == 'description':
                    continue
                raw_signal = fetched_signals[signal_name]
                if "SCORE_INTRADAY_CONVICTION_REVERSAL" in signal_name or \
                   "SCORE_MICRO_HARMONY_INFLECTION" in signal_name or \
                   "PROCESS_META_FUND_FLOW_BOTTOM_REVERSAL" in signal_name or \
                   "PROCESS_META_CHIP_BOTTOM_REVERSAL" in signal_name or \
                   "PROCESS_META_BEHAVIOR_BOTTOM_REVERSAL" in signal_name or \
                   "SCORE_CHIP_HARMONY_INFLECTION" in signal_name or \
                   "SCORE_FOUNDATION_HARMONY_INFLECTION" in signal_name or \
                   "SCORE_BEHAVIOR_BULLISH_DIVERGENCE_QUALITY" in signal_name:
                    signal_score = raw_signal.clip(lower=0)
                else:
                    signal_score = raw_signal.clip(lower=0)
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                reversal_intent_score_components += normalized_signal_score * weight
            reversal_intent_score = (reversal_intent_score_components / total_reversal_intent_weight) * reversal_dynamic_modulator
        else:
            reversal_intent_score = pd.Series(0.0, index=df.index)
        total_context_reinforcement_weight = sum(v for k, v in context_reinforcement_weights.items() if k != 'description' and isinstance(v, (int, float)))
        if total_context_reinforcement_weight > 0:
            for signal_name, weight in context_reinforcement_weights.items():
                if signal_name == 'description':
                    continue
                raw_signal = fetched_signals[signal_name]
                if "SCORE_FOUNDATION_AXIOM_MARKET_TENSION" in signal_name or \
                   "SCORE_STRUCT_AXIOM_TENSION" in signal_name or \
                   "SCORE_FOUNDATION_AXIOM_LIQUIDITY_TIDE" in signal_name or \
                   "SCORE_STRUCT_AXIOM_STABILITY" in signal_name or \
                   "SCORE_FOUNDATION_AXIOM_MARKET_CONSTITUTION" in signal_name or \
                   "SCORE_CYCLICAL_HURST_REVERSION_REGIME" in signal_name:
                    signal_score = raw_signal.clip(lower=0)
                else:
                    signal_score = raw_signal.clip(lower=0)
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                context_reinforcement_score_components += normalized_signal_score * weight
            context_score = context_reinforcement_score_components / total_context_reinforcement_weight
        else:
            context_score = pd.Series(0.0, index=df.index)
        total_risk_filter_weight = sum(v for k, v in risk_filter_weights.items() if k != 'description' and isinstance(v, (int, float)))
        if total_risk_filter_weight > 0:
            for signal_name, weight in risk_filter_weights.items():
                if signal_name == 'description':
                    continue
                raw_signal = fetched_signals[signal_name]
                if "SCORE_BEHAVIOR_BEARISH_DIVERGENCE" in signal_name or \
                   "SCORE_CHIP_RISK_DISTRIBUTION_WHISPER" in signal_name or \
                   "FUSION_RISK_DISTRIBUTION_PRESSURE" in signal_name:
                    signal_score = raw_signal.clip(lower=0)
                elif "SCORE_FF_AXIOM_DIVERGENCE" in signal_name or \
                     "SCORE_STRUCT_AXIOM_TREND_FORM" in signal_name:
                    signal_score = raw_signal.clip(upper=0).abs()
                elif "SCORE_BEHAVIOR_DECEPTION_INDEX" in signal_name:
                    signal_score = raw_signal.clip(lower=0)
                else:
                    signal_score = raw_signal.clip(lower=0)
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                risk_filter_score_components += normalized_signal_score * weight
            risk_filter_score = risk_filter_score_components / total_risk_filter_weight
        else:
            risk_filter_score = pd.Series(0.0, index=df.index)
        total_confirmation_contradiction_weight = sum(v for k, v in confirmation_contradiction_weights.items() if k != 'description' and isinstance(v, (int, float)))
        if total_confirmation_contradiction_weight > 0:
            for signal_name, weight in confirmation_contradiction_weights.items():
                if signal_name == 'description':
                    continue
                raw_signal = fetched_signals[signal_name]
                if "PROCESS_META_PRICE_VS_RETAIL_CAPITULATION" in signal_name or \
                   "SCORE_BEHAVIOR_BULLISH_DIVERGENCE_QUALITY" in signal_name or \
                   "SCORE_FUND_FLOW_BULLISH_DIVERGENCE" in signal_name:
                    signal_score = raw_signal.clip(lower=0)
                else:
                    signal_score = raw_signal.clip(lower=0)
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                confirmation_contradiction_score_components += normalized_signal_score * weight
            confirmation_score = confirmation_contradiction_score_components / total_confirmation_contradiction_weight
        else:
            confirmation_score = pd.Series(0.0, index=df.index)
        synergy_factor = pd.Series(1.0, index=df.index)
        synergy_condition = (panic_score > synergy_threshold) & \
                            (absorption_score > synergy_threshold) & \
                            (reversal_intent_score > synergy_threshold) & \
                            (risk_filter_score < (1 - synergy_threshold))
        synergy_factor.loc[synergy_condition] += synergy_bonus_factor
        conflict_condition = ((panic_score > synergy_threshold) & (absorption_score < (1 - synergy_threshold))) | \
                             (risk_filter_score > synergy_threshold)
        synergy_factor.loc[conflict_condition] -= conflict_penalty_factor
        synergy_factor = synergy_factor.clip(0.5, 1.5)
        epsilon = 1e-6
        fused_score_raw = (
            (panic_score + epsilon) *
            (absorption_score + epsilon) *
            (reversal_intent_score + epsilon) *
            (context_score + epsilon) *
            (confirmation_score + epsilon)
        )**(1/5)
        fused_score_raw = fused_score_raw * synergy_factor
        risk_adjusted_fused_score = fused_score_raw * (1 - risk_filter_score.clip(0, 1))
        final_score = (risk_adjusted_fused_score)**final_fusion_exponent
        final_score = final_score.clip(0, 1)
        final_score = final_score.where(final_score >= min_activation_threshold, 0.0)
        print(f"  -> {method_name} 计算完成。")
        return final_score.astype(np.float32)

    def _calculate_distribution_at_high(self, df: pd.DataFrame) -> pd.Series:
        method_name = "COGNITIVE_RISK_DISTRIBUTION_AT_HIGH"
        print(f"  -> [认知层] 正在计算 {method_name}...")
        cognitive_intelligence_config = get_params_block(self.strategy, 'cognitive_intelligence_params', {})
        params = cognitive_intelligence_config.get('playbooks', {}).get('cognitive_risk_distribution_at_high_params', {})
        high_level_context_weights = get_param_value(params.get('high_level_context_weights'), {})
        distribution_behavior_weights = get_param_value(params.get('distribution_behavior_weights'), {})
        chip_fund_flow_risk_weights = get_param_value(params.get('chip_fund_flow_risk_weights'), {})
        structural_trend_exhaustion_weights = get_param_value(params.get('structural_trend_exhaustion_weights'), {})
        deception_amplifier_sensitivity = get_param_value(params.get('deception_amplifier_sensitivity'), 0.5)
        synergy_threshold = get_param_value(params.get('synergy_threshold'), 0.6)
        synergy_bonus_factor = get_param_value(params.get('synergy_bonus_factor'), 0.2)
        conflict_penalty_factor = get_param_value(params.get('conflict_penalty_factor'), 0.3)
        dynamic_context_modulator_signal = get_param_value(params.get('dynamic_context_modulator_signal'), "SCORE_FOUNDATION_AXIOM_MARKET_CONSTITUTION")
        dynamic_context_sensitivity = get_param_value(params.get('dynamic_context_sensitivity'), 0.5)
        final_fusion_exponent = get_param_value(params.get('final_fusion_exponent'), 2.0)
        min_activation_threshold = get_param_value(params.get('min_activation_threshold'), 0.1)
        norm_window = get_param_value(params.get('norm_window'), 55)
        all_required_signals = set()
        all_required_signals.update(high_level_context_weights.keys())
        all_required_signals.update(distribution_behavior_weights.keys())
        all_required_signals.update(chip_fund_flow_risk_weights.keys())
        all_required_signals.update(structural_trend_exhaustion_weights.keys())
        all_required_signals.add("SCORE_BEHAVIOR_DECEPTION_INDEX")
        all_required_signals.add("market_sentiment_score_D")
        all_required_signals.add(dynamic_context_modulator_signal)
        fetched_signals = {}
        for signal_name in all_required_signals:
            if signal_name == 'description':
                continue
            fetched_signals[signal_name] = self._get_atomic_score(df, signal_name, default=0.0)
            if not isinstance(fetched_signals[signal_name], pd.Series):
                fetched_signals[signal_name] = pd.Series(fetched_signals[signal_name], index=df.index)
            else:
                fetched_signals[signal_name] = fetched_signals[signal_name].reindex(df.index).fillna(0.0)
        high_level_context_score_components = pd.Series(0.0, index=df.index)
        distribution_behavior_score_components = pd.Series(0.0, index=df.index)
        chip_fund_flow_risk_score_components = pd.Series(0.0, index=df.index)
        structural_trend_exhaustion_score_components = pd.Series(0.0, index=df.index)
        deception_index = fetched_signals.get("SCORE_BEHAVIOR_DECEPTION_INDEX", pd.Series(0.0, index=df.index))
        market_sentiment = fetched_signals.get("market_sentiment_score_D", pd.Series(0.5, index=df.index))
        deception_amplifier = 1 + deception_index.clip(lower=0) * deception_amplifier_sensitivity
        market_constitution_score = fetched_signals.get(dynamic_context_modulator_signal, pd.Series(0.5, index=df.index))
        normalized_market_constitution = normalize_score(market_constitution_score, df.index, norm_window, ascending=False)
        dynamic_risk_modulator = 1 + (normalized_market_constitution - 0.5) * dynamic_context_sensitivity * 2
        dynamic_risk_modulator = dynamic_risk_modulator.clip(0.5, 1.5)
        total_high_level_context_weight = sum(v for k, v in high_level_context_weights.items() if k != 'description' and isinstance(v, (int, float)))
        if total_high_level_context_weight > 0:
            for signal_name, weight in high_level_context_weights.items():
                if signal_name == 'description':
                    continue
                raw_signal = fetched_signals[signal_name]
                if "FUSION_BIPOLAR_PRICE_OVEREXTENSION_INTENT" in signal_name or \
                   "SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM" in signal_name or \
                   "SCORE_FOUNDATION_AXIOM_RELATIVE_STRENGTH" in signal_name:
                    signal_score = raw_signal.clip(lower=0)
                elif "SCORE_STRUCT_AXIOM_MTF_COHESION" in signal_name:
                    signal_score = raw_signal.clip(upper=0).abs()
                else:
                    signal_score = raw_signal.clip(lower=0)
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                high_level_context_score_components += normalized_signal_score * weight
            high_level_context_score = (high_level_context_score_components / total_high_level_context_weight) * dynamic_risk_modulator
        else:
            high_level_context_score = pd.Series(0.0, index=df.index)
        total_distribution_behavior_weight = sum(v for k, v in distribution_behavior_weights.items() if k != 'description' and isinstance(v, (int, float)))
        if total_distribution_behavior_weight > 0:
            for signal_name, weight in distribution_behavior_weights.items():
                if signal_name == 'description':
                    continue
                raw_signal = fetched_signals[signal_name]
                if "SCORE_BEHAVIOR_DISTRIBUTION_INTENT" in signal_name or \
                   "SCORE_BEHAVIOR_BEARISH_DIVERGENCE" in signal_name:
                    signal_score = raw_signal.clip(lower=0)
                elif "PROCESS_META_MAIN_FORCE_RALLY_INTENT" in signal_name or \
                     "PROCESS_META_PROFIT_VS_FLOW" in signal_name:
                    signal_score = raw_signal.clip(upper=0).abs()
                else:
                    signal_score = raw_signal.clip(lower=0)
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                distribution_behavior_score_components += normalized_signal_score * weight
            distribution_behavior_score = (distribution_behavior_score_components / total_distribution_behavior_weight) * dynamic_risk_modulator
        else:
            distribution_behavior_score = pd.Series(0.0, index=df.index)
        total_chip_fund_flow_risk_weight = sum(v for k, v in chip_fund_flow_risk_weights.items() if k != 'description' and isinstance(v, (int, float)))
        if total_chip_fund_flow_risk_weight > 0:
            for signal_name, weight in chip_fund_flow_risk_weights.items():
                if signal_name == 'description':
                    continue
                raw_signal = fetched_signals[signal_name]
                if "SCORE_CHIP_RISK_DISTRIBUTION_WHISPER" in signal_name or \
                   "SCORE_FUND_FLOW_BEARISH_DIVERGENCE" in signal_name or \
                   "FUSION_RISK_DISTRIBUTION_PRESSURE" in signal_name:
                    signal_score = raw_signal.clip(lower=0)
                elif "SCORE_CHIP_AXIOM_DIVERGENCE" in signal_name:
                    signal_score = raw_signal.clip(upper=0).abs()
                else:
                    signal_score = raw_signal.clip(lower=0)
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                chip_fund_flow_risk_score_components += normalized_signal_score * weight
            chip_fund_flow_risk_score = (chip_fund_flow_risk_score_components / total_chip_fund_flow_risk_weight) * dynamic_risk_modulator
        else:
            chip_fund_flow_risk_score = pd.Series(0.0, index=df.index)
        total_structural_trend_exhaustion_weight = sum(v for k, v in structural_trend_exhaustion_weights.items() if k != 'description' and isinstance(v, (int, float)))
        if total_structural_trend_exhaustion_weight > 0:
            for signal_name, weight in structural_trend_exhaustion_weights.items():
                if signal_name == 'description':
                    continue
                raw_signal = fetched_signals[signal_name]
                if "FUSION_RISK_STAGNATION" in signal_name or \
                   "PROCESS_META_WINNER_CONVICTION_DECAY" in signal_name or \
                   "PROCESS_META_PRICE_VS_MOMENTUM_DIVERGENCE" in signal_name:
                    signal_score = raw_signal.clip(lower=0)
                elif "FUSION_BIPOLAR_TREND_QUALITY" in signal_name or \
                     "SCORE_DYN_AXIOM_MOMENTUM" in signal_name or \
                     "SCORE_STRUCT_STRATEGIC_POSTURE" in signal_name:
                    signal_score = raw_signal.clip(upper=0).abs()
                else:
                    signal_score = raw_signal.clip(lower=0)
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                structural_trend_exhaustion_score_components += normalized_signal_score * weight
            structural_trend_exhaustion_score = (structural_trend_exhaustion_score_components / total_structural_trend_exhaustion_weight) * dynamic_risk_modulator
        else:
            structural_trend_exhaustion_score = pd.Series(0.0, index=df.index)
        synergy_factor = pd.Series(1.0, index=df.index)
        synergy_condition = (high_level_context_score > synergy_threshold) & \
                            (distribution_behavior_score > synergy_threshold) & \
                            (chip_fund_flow_risk_score > synergy_threshold) & \
                            (structural_trend_exhaustion_score > synergy_threshold) & \
                            (market_sentiment > synergy_threshold)
        synergy_factor.loc[synergy_condition] += synergy_bonus_factor
        conflict_condition = ((high_level_context_score > synergy_threshold) & (distribution_behavior_score < (1 - synergy_threshold))) | \
                             ((high_level_context_score > synergy_threshold) & (structural_trend_exhaustion_score < (1 - synergy_threshold)))
        synergy_factor.loc[conflict_condition] -= conflict_penalty_factor
        synergy_factor = synergy_factor.clip(0.5, 1.5)
        epsilon = 1e-6
        fused_risk_raw = (
            (high_level_context_score + epsilon) *
            (distribution_behavior_score + epsilon) *
            (chip_fund_flow_risk_score + epsilon) *
            (structural_trend_exhaustion_score + epsilon)
        )**(1/4)
        fused_risk_raw = fused_risk_raw * deception_amplifier * synergy_factor
        final_score = (fused_risk_raw)**final_fusion_exponent
        final_score = final_score.clip(0, 1)
        final_score = final_score.where(final_score >= min_activation_threshold, 0.0)
        print(f"  -> {method_name} 计算完成。")
        return final_score.astype(np.float32)

    def _calculate_leading_dragon_awakening(self, df: pd.DataFrame) -> pd.Series:
        method_name = "COGNITIVE_PLAYBOOK_LEADING_DRAGON_AWAKENING"
        print(f"  -> [认知层] 正在计算 {method_name}...")
        cognitive_intelligence_config = get_params_block(self.strategy, 'cognitive_intelligence_params', {})
        params = cognitive_intelligence_config.get('playbooks', {}).get('cognitive_playbook_leading_dragon_awakening_params', {})
        dormancy_breakout_weights = get_param_value(params.get('dormancy_breakout_weights'), {})
        main_force_offensive_weights = get_param_value(params.get('main_force_offensive_weights'), {})
        chip_fund_flow_resonance_weights = get_param_value(params.get('chip_fund_flow_resonance_weights'), {})
        market_context_reinforcement_weights = get_param_value(params.get('market_context_reinforcement_weights'), {})
        risk_filter_weights = get_param_value(params.get('risk_filter_weights'), {})
        synergy_threshold = get_param_value(params.get('synergy_threshold'), 0.6)
        synergy_bonus_factor = get_param_value(params.get('synergy_bonus_factor'), 0.2)
        conflict_penalty_factor = get_param_value(params.get('conflict_penalty_factor'), 0.3)
        dynamic_context_modulator_signal = get_param_value(params.get('dynamic_context_modulator_signal'), "FUSION_BIPOLAR_TREND_QUALITY")
        dynamic_context_sensitivity = get_param_value(params.get('dynamic_context_sensitivity'), 0.5)
        final_fusion_exponent = get_param_value(params.get('final_fusion_exponent'), 2.0)
        min_activation_threshold = get_param_value(params.get('min_activation_threshold'), 0.1)
        norm_window = get_param_value(params.get('norm_window'), 55)
        all_required_signals = set()
        all_required_signals.update(dormancy_breakout_weights.keys())
        all_required_signals.update(main_force_offensive_weights.keys())
        all_required_signals.update(chip_fund_flow_resonance_weights.keys())
        all_required_signals.update(market_context_reinforcement_weights.keys())
        all_required_signals.update(risk_filter_weights.keys())
        all_required_signals.add("SCORE_BEHAVIOR_DECEPTION_INDEX")
        all_required_signals.add("SCORE_CHIP_RISK_DISTRIBUTION_WHISPER")
        all_required_signals.add("FUSION_RISK_DISTRIBUTION_PRESSURE")
        all_required_signals.add(dynamic_context_modulator_signal)
        all_required_signals.add("SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM")
        fetched_signals = {}
        for signal_name in all_required_signals:
            if signal_name == 'description':
                continue
            fetched_signals[signal_name] = self._get_atomic_score(df, signal_name, default=0.0)
            if not isinstance(fetched_signals[signal_name], pd.Series):
                fetched_signals[signal_name] = pd.Series(fetched_signals[signal_name], index=df.index)
            else:
                fetched_signals[signal_name] = fetched_signals[signal_name].reindex(df.index).fillna(0.0)
        market_trend_quality_score = fetched_signals.get(dynamic_context_modulator_signal, pd.Series(0.5, index=df.index))
        normalized_market_trend_quality = normalize_score(market_trend_quality_score, df.index, norm_window, ascending=True)
        dynamic_modulator = 1 + (normalized_market_trend_quality - 0.5) * dynamic_context_sensitivity * 2
        dynamic_modulator = dynamic_modulator.clip(0.5, 1.5)
        dormancy_breakout_score_components = pd.Series(0.0, index=df.index)
        main_force_offensive_score_components = pd.Series(0.0, index=df.index)
        chip_fund_flow_resonance_score_components = pd.Series(0.0, index=df.index)
        market_context_reinforcement_score_components = pd.Series(0.0, index=df.index)
        risk_filter_score_components = pd.Series(0.0, index=df.index)
        total_dormancy_breakout_weight = sum(v for k, v in dormancy_breakout_weights.items() if k != 'description' and isinstance(v, (int, float)))
        if total_dormancy_breakout_weight > 0:
            for signal_name, weight in dormancy_breakout_weights.items():
                if signal_name == 'description':
                    continue
                raw_signal = fetched_signals[signal_name]
                if "SCORE_PATTERN_AXIOM_BREAKOUT" in signal_name or \
                   "SCORE_BEHAVIOR_VOLUME_BURST" in signal_name or \
                   "SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM" in signal_name or \
                   "SCORE_MICRO_HARMONY_INFLECTION" in signal_name or \
                   "PROCESS_META_COST_ADVANTAGE_TREND" in signal_name:
                    signal_score = raw_signal.clip(lower=0)
                else:
                    signal_score = raw_signal.clip(lower=0)
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                dormancy_breakout_score_components += normalized_signal_score * weight
            dormancy_breakout_score = (dormancy_breakout_score_components / total_dormancy_breakout_weight) * dynamic_modulator
        else:
            dormancy_breakout_score = pd.Series(0.0, index=df.index)
        total_main_force_offensive_weight = sum(v for k, v in main_force_offensive_weights.items() if k != 'description' and isinstance(v, (int, float)))
        if total_main_force_offensive_weight > 0:
            for signal_name, weight in main_force_offensive_weights.items():
                if signal_name == 'description':
                    continue
                raw_signal = fetched_signals[signal_name]
                if "SCORE_BEHAVIOR_OFFENSIVE_ABSORPTION_INTENT" in signal_name or \
                   "PROCESS_META_MAIN_FORCE_RALLY_INTENT" in signal_name or \
                   "SCORE_INTRADAY_OFFENSIVE_PURITY" in signal_name or \
                   "PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY" in signal_name or \
                   "SCORE_CHIP_TACTICAL_EXCHANGE" in signal_name:
                    signal_score = raw_signal.clip(lower=0)
                else:
                    signal_score = raw_signal.clip(lower=0)
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                main_force_offensive_score_components += normalized_signal_score * weight
            main_force_offensive_score = (main_force_offensive_score_components / total_main_force_offensive_weight) * dynamic_modulator
        else:
            main_force_offensive_score = pd.Series(0.0, index=df.index)
        total_chip_fund_flow_resonance_weight = sum(v for k, v in chip_fund_flow_resonance_weights.items() if k != 'description' and isinstance(v, (int, float)))
        if total_chip_fund_flow_resonance_weight > 0:
            for signal_name, weight in chip_fund_flow_resonance_weights.items():
                if signal_name == 'description':
                    continue
                raw_signal = fetched_signals[signal_name]
                if "SCORE_CHIP_COHERENT_DRIVE" in signal_name or \
                   "SCORE_FUND_FLOW_BULLISH_DIVERGENCE" in signal_name or \
                   "SCORE_CHIP_HARMONY_INFLECTION" in signal_name or \
                   "PROCESS_META_FUND_FLOW_ACCUMULATION_INFLECTION_INTENT" in signal_name:
                    signal_score = raw_signal.clip(lower=0)
                elif "FUSION_BIPOLAR_FUND_FLOW_TREND" in signal_name:
                    signal_score = raw_signal.clip(lower=0)
                else:
                    signal_score = raw_signal.clip(lower=0)
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                chip_fund_flow_resonance_score_components += normalized_signal_score * weight
            chip_fund_flow_resonance_score = (chip_fund_flow_resonance_score_components / total_chip_fund_flow_resonance_weight) * dynamic_modulator
        else:
            chip_fund_flow_resonance_score = pd.Series(0.0, index=df.index)
        total_market_context_reinforcement_weight = sum(v for k, v in market_context_reinforcement_weights.items() if k != 'description' and isinstance(v, (int, float)))
        if total_market_context_reinforcement_weight > 0:
            for signal_name, weight in market_context_reinforcement_weights.items():
                if signal_name == 'description':
                    continue
                raw_signal = fetched_signals[signal_name]
                if "SCORE_FOUNDATION_AXIOM_MARKET_CONSTITUTION" in signal_name or \
                   "SCORE_FOUNDATION_AXIOM_LIQUIDITY_TIDE" in signal_name or \
                   "SCORE_STRUCT_LEADERSHIP_POTENTIAL" in signal_name or \
                   "SCORE_FOUNDATION_AXIOM_RELATIVE_STRENGTH" in signal_name or \
                   "SCORE_INTRADAY_DOMINANCE_CONSENSUS" in signal_name:
                    signal_score = raw_signal.clip(lower=0)
                else:
                    signal_score = raw_signal.clip(lower=0)
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                market_context_reinforcement_score_components += normalized_signal_score * weight
            market_context_reinforcement_score = (market_context_reinforcement_score_components / total_market_context_reinforcement_weight) * dynamic_modulator
        else:
            market_context_reinforcement_score = pd.Series(0.0, index=df.index)
        total_risk_filter_weight = sum(v for k, v in risk_filter_weights.items() if k != 'description' and isinstance(v, (int, float)))
        if total_risk_filter_weight > 0:
            for signal_name, weight in risk_filter_weights.items():
                if signal_name == 'description':
                    continue
                raw_signal = fetched_signals[signal_name]
                if "SCORE_BEHAVIOR_DECEPTION_INDEX" in signal_name or \
                   "SCORE_CHIP_RISK_DISTRIBUTION_WHISPER" in signal_name or \
                   "FUSION_RISK_DISTRIBUTION_PRESSURE" in signal_name or \
                   "SCORE_BEHAVIOR_BEARISH_DIVERGENCE" in signal_name:
                    signal_score = raw_signal.clip(lower=0)
                elif "SCORE_FF_AXIOM_DIVERGENCE" in signal_name:
                    signal_score = raw_signal.clip(upper=0).abs()
                else:
                    signal_score = raw_signal.clip(lower=0)
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                risk_filter_score_components += normalized_signal_score * weight
            risk_filter_score = risk_filter_score_components / total_risk_filter_weight
        else:
            risk_filter_score = pd.Series(0.0, index=df.index)
        synergy_factor = pd.Series(1.0, index=df.index)
        sentiment_pendulum = fetched_signals.get("SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM", pd.Series(0.0, index=df.index))
        synergy_condition = (dormancy_breakout_score > synergy_threshold) & \
                            (main_force_offensive_score > synergy_threshold) & \
                            (chip_fund_flow_resonance_score > synergy_threshold) & \
                            (market_context_reinforcement_score > synergy_threshold) & \
                            (risk_filter_score < (1 - synergy_threshold)) & \
                            (sentiment_pendulum < synergy_threshold)
        synergy_factor.loc[synergy_condition] += synergy_bonus_factor
        conflict_condition = ((main_force_offensive_score > synergy_threshold) & (risk_filter_score > synergy_threshold)) | \
                             ((dormancy_breakout_score > synergy_threshold) & (chip_fund_flow_resonance_score < (1 - synergy_threshold)))
        synergy_factor.loc[conflict_condition] -= conflict_penalty_factor
        synergy_factor = synergy_factor.clip(0.5, 1.5)
        epsilon = 1e-6
        fused_score_raw = (
            (dormancy_breakout_score + epsilon) *
            (main_force_offensive_score + epsilon) *
            (chip_fund_flow_resonance_score + epsilon) *
            (market_context_reinforcement_score + epsilon)
        )**(1/4)
        fused_score_raw = fused_score_raw * synergy_factor
        risk_adjusted_fused_score = fused_score_raw * (1 - risk_filter_score.clip(0, 1))
        final_score = (risk_adjusted_fused_score)**final_fusion_exponent
        final_score = final_score.clip(0, 1)
        final_score = final_score.where(final_score >= min_activation_threshold, 0.0)
        print(f"  -> {method_name} 计算完成。")
        return final_score.astype(np.float32)

    def _calculate_energy_compression(self, df: pd.DataFrame) -> pd.Series:
        method_name = "COGNITIVE_PLAYBOOK_ENERGY_COMPRESSION"
        print(f"  -> [认知层] 正在计算 {method_name}...")
        cognitive_intelligence_config = get_params_block(self.strategy, 'cognitive_intelligence_params', {})
        params = cognitive_intelligence_config.get('playbooks', {}).get('cognitive_playbook_energy_compression_params', {})
        if self.debug_enabled:
            print(f"    -> [探针] {method_name} 加载的原始参数 (params): {params}")
        volatility_compression_weights = get_param_value(params.get('volatility_compression_weights'), {})
        volume_atrophy_weights = get_param_value(params.get('volume_atrophy_weights'), {})
        main_force_control_weights = get_param_value(params.get('main_force_control_weights'), {})
        pre_breakout_indicators_weights = get_param_value(params.get('pre_breakout_indicators_weights'), {})
        risk_filter_weights = get_param_value(params.get('risk_filter_weights'), {})
        synergy_threshold = get_param_value(params.get('synergy_threshold'), 0.6)
        synergy_bonus_factor = get_param_value(params.get('synergy_bonus_factor'), 0.2)
        conflict_penalty_factor = get_param_value(params.get('conflict_penalty_factor'), 0.3)
        dynamic_context_modulator_signal = get_param_value(params.get('dynamic_context_modulator_signal'), "SCORE_CYCLICAL_HURST_REVERSION_REGIME")
        dynamic_context_sensitivity = get_param_value(params.get('dynamic_context_sensitivity'), 0.5)
        final_fusion_exponent = get_param_value(params.get('final_fusion_exponent'), 2.0)
        min_activation_threshold = get_param_value(params.get('min_activation_threshold'), 0.1)
        norm_window = get_param_value(params.get('norm_window'), 55)
        probe_dates_to_print = []
        if self.debug_enabled and self.probe_dates_list_str:
            if not df.empty:
                df_index_tz = df.index.tz
                for date_str in self.probe_dates_list_str:
                    try:
                        probe_date_naive = pd.to_datetime(date_str)
                        if df_index_tz is not None and probe_date_naive.tz is None:
                            current_probe_date = probe_date_naive.tz_localize(df_index_tz)
                        elif df_index_tz is None and probe_date_naive.tz is not None:
                            current_probe_date = probe_date_naive.tz_convert(None)
                        else:
                            current_probe_date = probe_date_naive
                        if current_probe_date in df.index:
                            probe_dates_to_print.append(current_probe_date)
                        else:
                            print(f"    -> [探针警告] 探测日期 '{date_str}' (时区校准后: {current_probe_date}) 不在DataFrame索引中，跳过。")
                    except Exception as e:
                        print(f"    -> [探针警告] 无法解析或处理探针日期 '{date_str}': {e}")
        if probe_dates_to_print:
            print(f"    -> [探针] 准备为以下日期输出详细信息: {[d.strftime('%Y-%m-%d') for d in probe_dates_to_print]}")
        else:
            if self.debug_enabled:
                print(f"    -> [探针] 未找到有效的探测日期或调试未启用，将跳过详细探针输出。请检查 debug_params['probe_dates'] 和数据范围。")
        all_required_signals = set()
        all_required_signals.update(volatility_compression_weights.keys())
        all_required_signals.update(volume_atrophy_weights.keys())
        all_required_signals.update(main_force_control_weights.keys())
        all_required_signals.update(pre_breakout_indicators_weights.keys())
        all_required_signals.update(risk_filter_weights.keys())
        all_required_signals.add("SCORE_BEHAVIOR_DECEPTION_INDEX")
        all_required_signals.add("SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM")
        all_required_signals.add(dynamic_context_modulator_signal)
        fetched_signals = {}
        for signal_name in all_required_signals:
            if signal_name == 'description':
                continue
            fetched_signals[signal_name] = self._get_atomic_score(df, signal_name, default=0.0)
            if not isinstance(fetched_signals[signal_name], pd.Series):
                fetched_signals[signal_name] = pd.Series(fetched_signals[signal_name], index=df.index)
            else:
                fetched_signals[signal_name] = fetched_signals[signal_name].reindex(df.index).fillna(0.0)
        hurst_reversion_score = fetched_signals.get(dynamic_context_modulator_signal, pd.Series(0.5, index=df.index))
        normalized_hurst_reversion = normalize_score(hurst_reversion_score, df.index, norm_window, ascending=True)
        dynamic_modulator = 1 + (normalized_hurst_reversion - 0.5) * dynamic_context_sensitivity * 2
        dynamic_modulator = dynamic_modulator.clip(0.5, 1.5)
        volatility_compression_score_components = pd.Series(0.0, index=df.index)
        volume_atrophy_score_components = pd.Series(0.0, index=df.index)
        main_force_control_score_components = pd.Series(0.0, index=df.index)
        pre_breakout_indicators_score_components = pd.Series(0.0, index=df.index)
        risk_filter_score_components = pd.Series(0.0, index=df.index)
        if self.debug_enabled:
            print(f"    -> [探针] 开始计算波动率压缩证据分数...")
            for p_date in probe_dates_to_print:
                print(f"      - [探针 {p_date.strftime('%Y-%m-%d')}] 动态调制器 (Dynamic Modulator): {dynamic_modulator.loc[p_date]:.4f}")
        total_volatility_compression_weight = sum(v for k, v in volatility_compression_weights.items() if k != 'description' and isinstance(v, (int, float)))
        if total_volatility_compression_weight > 0:
            if self.debug_enabled:
                print(f"    -> [探针] 波动率压缩证据总权重: {total_volatility_compression_weight:.4f}")
            for signal_name, weight in volatility_compression_weights.items():
                if signal_name == 'description':
                    continue
                raw_signal = fetched_signals[signal_name]
                if "SCORE_STRUCT_AXIOM_TENSION" in signal_name or \
                   "SCORE_DYN_AXIOM_STABILITY" in signal_name:
                    signal_score = raw_signal.clip(lower=0)
                elif "BBW_21_2.0_D" in signal_name or \
                     "ATR_14_D" in signal_name:
                    signal_score = normalize_score(raw_signal, df.index, norm_window, ascending=False)
                else:
                    signal_score = raw_signal.clip(lower=0)
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                volatility_compression_score_components += normalized_signal_score * weight
                if self.debug_enabled:
                    for p_date in probe_dates_to_print:
                        print(f"      - [探针 {p_date.strftime('%Y-%m-%d')}] 波动率压缩信号 '{signal_name}' (权重: {weight:.2f}) 原始值: {raw_signal.loc[p_date]:.4f}, 转换后值: {signal_score.loc[p_date]:.4f}, 归一化后: {normalized_signal_score.loc[p_date]:.4f}, 加权贡献: {(normalized_signal_score.loc[p_date] * weight):.4f}")
            volatility_compression_score = (volatility_compression_score_components / total_volatility_compression_weight) * dynamic_modulator
        else:
            volatility_compression_score = pd.Series(0.0, index=df.index)
        if self.debug_enabled:
            for p_date in probe_dates_to_print:
                print(f"    -> [探针 {p_date.strftime('%Y-%m-%d')}] 综合波动率压缩证据分数 (Volatility Compression Evidence Score): {volatility_compression_score.loc[p_date]:.4f}")
        if self.debug_enabled:
            print(f"    -> [探针] 开始计算量能萎缩证据分数...")
        total_volume_atrophy_weight = sum(v for k, v in volume_atrophy_weights.items() if k != 'description' and isinstance(v, (int, float)))
        if total_volume_atrophy_weight > 0:
            if self.debug_enabled:
                print(f"    -> [探针] 量能萎缩证据总权重: {total_volume_atrophy_weight:.4f}")
            for signal_name, weight in volume_atrophy_weights.items():
                if signal_name == 'description':
                    continue
                raw_signal = fetched_signals[signal_name]
                if "SCORE_BEHAVIOR_VOLUME_ATROPHY" in signal_name:
                    signal_score = raw_signal.clip(lower=0)
                elif "VOL_MA_5_D" in signal_name or \
                     "VOL_MA_13_D" in signal_name:
                    signal_score = normalize_score(raw_signal, df.index, norm_window, ascending=False)
                else:
                    signal_score = raw_signal.clip(lower=0)
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                volume_atrophy_score_components += normalized_signal_score * weight
                if self.debug_enabled:
                    for p_date in probe_dates_to_print:
                        print(f"      - [探针 {p_date.strftime('%Y-%m-%d')}] 量能萎缩信号 '{signal_name}' (权重: {weight:.2f}) 原始值: {raw_signal.loc[p_date]:.4f}, 转换后值: {signal_score.loc[p_date]:.4f}, 归一化后: {normalized_signal_score.loc[p_date]:.4f}, 加权贡献: {(normalized_signal_score.loc[p_date] * weight):.4f}")
            volume_atrophy_score = (volume_atrophy_score_components / total_volume_atrophy_weight) * dynamic_modulator
        else:
            volume_atrophy_score = pd.Series(0.0, index=df.index)
        if self.debug_enabled:
            for p_date in probe_dates_to_print:
                print(f"    -> [探针 {p_date.strftime('%Y-%m-%d')}] 综合量能萎缩证据分数 (Volume Atrophy Evidence Score): {volume_atrophy_score.loc[p_date]:.4f}")
        if self.debug_enabled:
            print(f"    -> [探针] 开始计算主力控盘迹象分数...")
        total_main_force_control_weight = sum(v for k, v in main_force_control_weights.items() if k != 'description' and isinstance(v, (int, float)))
        if total_main_force_control_weight > 0:
            if self.debug_enabled:
                print(f"    -> [探针] 主力控盘迹象总权重: {total_main_force_control_weight:.4f}")
            for signal_name, weight in main_force_control_weights.items():
                if signal_name == 'description':
                    continue
                raw_signal = fetched_signals[signal_name]
                if "PROCESS_META_MAIN_FORCE_CONTROL" in signal_name or \
                   "SCORE_CHIP_AXIOM_HOLDER_SENTIMENT" in signal_name or \
                   "SCORE_FF_AXIOM_CONSENSUS" in signal_name or \
                   "SCORE_MICRO_STRATEGY_COST_CONTROL" in signal_name or \
                   "SCORE_STRUCT_BREAKOUT_READINESS" in signal_name:
                    signal_score = raw_signal.clip(lower=0)
                else:
                    signal_score = raw_signal.clip(lower=0)
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                main_force_control_score_components += normalized_signal_score * weight
                if self.debug_enabled:
                    for p_date in probe_dates_to_print:
                        print(f"      - [探针 {p_date.strftime('%Y-%m-%d')}] 主力控盘信号 '{signal_name}' (权重: {weight:.2f}) 原始值: {raw_signal.loc[p_date]:.4f}, 转换后值: {signal_score.loc[p_date]:.4f}, 归一化后: {normalized_signal_score.loc[p_date]:.4f}, 加权贡献: {(normalized_signal_score.loc[p_date] * weight):.4f}")
            main_force_control_score = (main_force_control_score_components / total_main_force_control_weight) * dynamic_modulator
        else:
            main_force_control_score = pd.Series(0.0, index=df.index)
        if self.debug_enabled:
            for p_date in probe_dates_to_print:
                print(f"    -> [探针 {p_date.strftime('%Y-%m-%d')}] 综合主力控盘迹象分数 (Main Force Control Signs Score): {main_force_control_score.loc[p_date]:.4f}")
        if self.debug_enabled:
            print(f"    -> [探针] 开始计算爆发前兆分数...")
        total_pre_breakout_indicators_weight = sum(v for k, v in pre_breakout_indicators_weights.items() if k != 'description' and isinstance(v, (int, float)))
        if total_pre_breakout_indicators_weight > 0:
            if self.debug_enabled:
                print(f"    -> [探针] 爆发前兆总权重: {total_pre_breakout_indicators_weight:.4f}")
            for signal_name, weight in pre_breakout_indicators_weights.items():
                if signal_name == 'description':
                    continue
                raw_signal = fetched_signals[signal_name]
                if "SCORE_STRUCT_AXIOM_ENVIRONMENT" in signal_name or \
                   "SCORE_FOUNDATION_AXIOM_MARKET_TENSION" in signal_name or \
                   "SCORE_CHIP_AXIOM_HISTORICAL_POTENTIAL" in signal_name or \
                   "SCORE_MICRO_HARMONY_INFLECTION" in signal_name or \
                   "PROCESS_META_ACCUMULATION_INFLECTION" in signal_name:
                    signal_score = raw_signal.clip(lower=0)
                else:
                    signal_score = raw_signal.clip(lower=0)
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                pre_breakout_indicators_score_components += normalized_signal_score * weight
                if self.debug_enabled:
                    for p_date in probe_dates_to_print:
                        print(f"      - [探针 {p_date.strftime('%Y-%m-%d')}] 爆发前兆信号 '{signal_name}' (权重: {weight:.2f}) 原始值: {raw_signal.loc[p_date]:.4f}, 转换后值: {signal_score.loc[p_date]:.4f}, 归一化后: {normalized_signal_score.loc[p_date]:.4f}, 加权贡献: {(normalized_signal_score.loc[p_date] * weight):.4f}")
            pre_breakout_indicators_score = (pre_breakout_indicators_score_components / total_pre_breakout_indicators_weight) * dynamic_modulator
        else:
            pre_breakout_indicators_score = pd.Series(0.0, index=df.index)
        if self.debug_enabled:
            for p_date in probe_dates_to_print:
                print(f"    -> [探针 {p_date.strftime('%Y-%m-%d')}] 综合爆发前兆分数 (Pre-Breakout Indicators Score): {pre_breakout_indicators_score.loc[p_date]:.4f}")
        if self.debug_enabled:
            print(f"    -> [探针] 开始计算风险过滤分数...")
        total_risk_filter_weight = sum(v for k, v in risk_filter_weights.items() if k != 'description' and isinstance(v, (int, float)))
        if total_risk_filter_weight > 0:
            if self.debug_enabled:
                print(f"    -> [探针] 风险过滤总权重: {total_risk_filter_weight:.4f}")
            for signal_name, weight in risk_filter_weights.items():
                if signal_name == 'description':
                    continue
                raw_signal = fetched_signals[signal_name]
                if "SCORE_BEHAVIOR_DECEPTION_INDEX" in signal_name or \
                   "SCORE_RISK_BREAKOUT_FAILURE_CASCADE" in signal_name or \
                   "FUSION_RISK_STAGNATION" in signal_name or \
                   "FUSION_RISK_DISTRIBUTION_PRESSURE" in signal_name:
                    signal_score = raw_signal.clip(lower=0)
                elif "SCORE_FF_AXIOM_DIVERGENCE" in signal_name:
                    signal_score = raw_signal.clip(upper=0).abs()
                else:
                    signal_score = raw_signal.clip(lower=0)
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                risk_filter_score_components += normalized_signal_score * weight
                if self.debug_enabled:
                    for p_date in probe_dates_to_print:
                        print(f"      - [探针 {p_date.strftime('%Y-%m-%d')}] 风险信号 '{signal_name}' (权重: {weight:.2f}) 原始值: {raw_signal.loc[p_date]:.4f}, 转换后值: {signal_score.loc[p_date]:.4f}, 归一化后: {normalized_signal_score.loc[p_date]:.4f}, 加权贡献: {(normalized_signal_score.loc[p_date] * weight):.4f}")
            risk_filter_score = risk_filter_score_components / total_risk_filter_weight
        else:
            risk_filter_score = pd.Series(0.0, index=df.index)
        if self.debug_enabled:
            for p_date in probe_dates_to_print:
                print(f"    -> [探针 {p_date.strftime('%Y-%m-%d')}] 综合风险过滤分数 (Risk Filter Score): {risk_filter_score.loc[p_date]:.4f}")
        synergy_factor = pd.Series(1.0, index=df.index)
        sentiment_pendulum = fetched_signals.get("SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM", pd.Series(0.0, index=df.index))
        synergy_condition = (volatility_compression_score > synergy_threshold) & \
                            (volume_atrophy_score > synergy_threshold) & \
                            (main_force_control_score > synergy_threshold) & \
                            (pre_breakout_indicators_score > synergy_threshold) & \
                            (risk_filter_score < (1 - synergy_threshold)) & \
                            (sentiment_pendulum < synergy_threshold) # 悲观情绪下，压缩爆发更具价值
        synergy_factor.loc[synergy_condition] += synergy_bonus_factor
        conflict_condition = ((volatility_compression_score > synergy_threshold) & (main_force_control_score < (1 - synergy_threshold))) | \
                             ((volume_atrophy_score > synergy_threshold) & (risk_filter_score > synergy_threshold))
        synergy_factor.loc[conflict_condition] -= conflict_penalty_factor
        synergy_factor = synergy_factor.clip(0.5, 1.5)
        if self.debug_enabled:
            for p_date in probe_dates_to_print:
                print(f"    -> [探针 {p_date.strftime('%Y-%m-%d')}] 协同因子 (Synergy Factor): {synergy_factor.loc[p_date]:.4f}")
        epsilon = 1e-6
        fused_score_raw = (
            (volatility_compression_score + epsilon) *
            (volume_atrophy_score + epsilon) *
            (main_force_control_score + epsilon) *
            (pre_breakout_indicators_score + epsilon)
        )**(1/4)
        fused_score_raw = fused_score_raw * synergy_factor
        risk_adjusted_fused_score = fused_score_raw * (1 - risk_filter_score.clip(0, 1))
        final_score = (risk_adjusted_fused_score)**final_fusion_exponent
        final_score = final_score.clip(0, 1)
        if self.debug_enabled:
            for p_date in probe_dates_to_print:
                print(f"    -> [探针 {p_date.strftime('%Y-%m-%d')}] final_score (clip后, where前): {final_score.loc[p_date]:.4f}")
                print(f"    -> [探针 {p_date.strftime('%Y-%m-%d')}] min_activation_threshold: {min_activation_threshold:.4f}")
                print(f"    -> [探针 {p_date.strftime('%Y-%m-%d')}] final_score >= min_activation_threshold: {(final_score.loc[p_date] >= min_activation_threshold)}")
        final_score = final_score.where(final_score >= min_activation_threshold, 0.0)
        if self.debug_enabled:
            for p_date in probe_dates_to_print:
                print(f"    -> [探针 {p_date.strftime('%Y-%m-%d')}] 最终融合原始分数 (Fused Score Raw): {fused_score_raw.loc[p_date]:.4f}")
                print(f"    -> [探针 {p_date.strftime('%Y-%m-%d')}] 风险调整后融合分数 (Risk Adjusted Fused Score): {risk_adjusted_fused_score.loc[p_date]:.4f}")
                print(f"    -> [探针 {p_date.strftime('%Y-%m-%d')}] 最终剧本分数 (Final Playbook Score): {final_score.loc[p_date]:.4f}")
        print(f"  -> {method_name} 计算完成。")
        return final_score.astype(np.float32)











