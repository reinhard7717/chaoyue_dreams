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

    def synthesize_cognitive_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        修改思路：
        1.  在现有方法中，调用新编写的 _calculate_suppressive_accumulation 方法。
        2.  将计算结果存储到 self.strategy.playbook_states 字典中。
        """
        cognitive_scores = {}
        # 修改开始
        # 调用主力打压吸筹剧本计算方法
        cognitive_scores["COGNITIVE_PLAYBOOK_SUPPRESSIVE_ACCUMULATION"] = self._calculate_suppressive_accumulation(df) # 修改行
        # 修改结束

        # ... (其他认知剧本的计算，如果存在) ...

        return cognitive_scores

    def _calculate_suppressive_accumulation(self, df: pd.DataFrame) -> pd.Series:
        method_name = "COGNITIVE_PLAYBOOK_SUPPRESSIVE_ACCUMULATION"
        print(f"  -> [认知层] 正在计算 {method_name}...")

        full_config_dict = {}
        if hasattr(self.strategy, 'params') and isinstance(self.strategy.params, dict):
            full_config_dict = self.strategy.params
        elif hasattr(self.strategy, 'config') and isinstance(self.strategy.config, dict):
            full_config_dict = self.strategy.config
        else:
            if self.debug_enabled:
                print(f"    -> [探针警告] self.strategy.params 和 self.strategy.config 都不存在或不是字典类型。参数加载可能失败。")
        
        if self.debug_enabled:
            print(f"    -> [探针] _calculate_suppressive_accumulation: full_config_dict (self.strategy.params) 顶层键: {list(full_config_dict.keys())}")
            strategy_params_block = full_config_dict.get('strategy_params', {})
            print(f"    -> [探针] _calculate_suppressive_accumulation: 'strategy_params' 结果: {list(strategy_params_block.keys()) if isinstance(strategy_params_block, dict) else strategy_params_block}")
            trend_follow_params_block = strategy_params_block.get('trend_follow', {})
            print(f"    -> [探针] _calculate_suppressive_accumulation: 'strategy_params.trend_follow' 结果: {list(trend_follow_params_block.keys()) if isinstance(trend_follow_params_block, dict) else trend_follow_params_block}")
            cognitive_intel_params_block = trend_follow_params_block.get('cognitive_intelligence_params', {})
            print(f"    -> [探针] _calculate_suppressive_accumulation: 'strategy_params.trend_follow.cognitive_intelligence_params' 结果: {list(cognitive_intel_params_block.keys()) if isinstance(cognitive_intel_params_block, dict) else cognitive_intel_params_block}")
            playbook_params_block = cognitive_intel_params_block.get('cognitive_playbook_suppressive_accumulation_params', {})
            print(f"    -> [探针] _calculate_suppressive_accumulation: 'strategy_params.trend_follow.cognitive_intelligence_params.cognitive_playbook_suppressive_accumulation_params' 结果: {playbook_params_block}")

        cognitive_intelligence_config = get_params_block(self.strategy, 'cognitive_intelligence_params', {})
        params = cognitive_intelligence_config.get('cognitive_playbook_suppressive_accumulation_params', {})

        if self.debug_enabled:
            print(f"    -> [探针] {method_name} 加载的原始参数 (params): {params}")

        suppression_weights = get_param_value(params.get('suppression_weights'), {})
        accumulation_weights = get_param_value(params.get('accumulation_weights'), {})
        contradiction_weights = get_param_value(params.get('contradiction_weights'), {})
        context_modulator_weights = get_param_value(params.get('context_modulator_weights'), {})
        final_fusion_exponent = get_param_value(params.get('final_fusion_exponent'), 2.0)
        min_activation_threshold = get_param_value(params.get('min_activation_threshold'), 0.1)
        norm_window = get_param_value(params.get('norm_window'), 55)

        # 修改开始 - 移除对 SCORE_CONTEXT_DEEP_BOTTOM_ZONE 的依赖，并调整权重
        # 原始权重：{'SCORE_CONTEXT_DEEP_BOTTOM_ZONE': 0.4, 'SCORE_STRUCT_AXIOM_TENSION': 0.3, 'SCORE_FOUNDATION_AXIOM_MARKET_TENSION': 0.3}
        # 移除 SCORE_CONTEXT_DEEP_BOTTOM_ZONE 后，将 0.4 的权重按比例分配给剩余两个信号
        if 'SCORE_CONTEXT_DEEP_BOTTOM_ZONE' in context_modulator_weights:
            removed_weight = context_modulator_weights.pop('SCORE_CONTEXT_DEEP_BOTTOM_ZONE')
            # 修改行 - 过滤掉非数字值
            remaining_total_weight = sum(v for k, v in context_modulator_weights.items() if k != 'description' and isinstance(v, (int, float)))
            if remaining_total_weight > 0:
                # 按比例增加剩余信号的权重
                for k in context_modulator_weights:
                    if k != 'description' and isinstance(context_modulator_weights[k], (int, float)): # 确保只处理数字权重
                        context_modulator_weights[k] += context_modulator_weights[k] / remaining_total_weight * removed_weight
            else: # 如果没有剩余信号，则清空
                context_modulator_weights = {}
        # 确保权重总和为1，如果不是，则重新归一化
        # 修改行 - 过滤掉非数字值
        current_total_context_weight = sum(v for k, v in context_modulator_weights.items() if k != 'description' and isinstance(v, (int, float)))
        if current_total_context_weight > 0 and abs(current_total_context_weight - 1.0) > 1e-6:
            context_modulator_weights = {k: v / current_total_context_weight for k, v in context_modulator_weights.items() if k != 'description' and isinstance(v, (int, float))}
        # 修改结束

        if self.debug_enabled:
            print(f"    -> [探针] suppression_weights: {suppression_weights}")
            print(f"    -> [探针] accumulation_weights: {accumulation_weights}")
            print(f"    -> [探针] contradiction_weights: {contradiction_weights}")
            print(f"    -> [探针] context_modulator_weights: {context_modulator_weights}")

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
            print(f"    -> [探针] 未找到有效的探测日期或调试未启用，将跳过详细探针输出。请检查 debug_params['probe_dates'] 和数据范围。")

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

        # 1. 计算打压证据分数 (Suppression Evidence Score)
        if probe_dates_to_print:
            print(f"    -> [探针] 开始计算打压证据分数...")
        total_suppression_weight = sum(v for k, v in suppression_weights.items() if k != 'description' and isinstance(v, (int, float)))
        if probe_dates_to_print:
            print(f"    -> [探针] 打压证据总权重: {total_suppression_weight:.4f}")
        if total_suppression_weight > 0:
            for signal_name, weight in suppression_weights.items():
                if signal_name == 'description':
                    continue
                raw_signal = fetched_signals[signal_name]
                if "PRICE_DOWNWARD_MOMENTUM" in signal_name:
                    signal_score = raw_signal.clip(upper=0).abs()
                elif "DISTRIBUTION_INTENT" in signal_name or "STAGNATION_EVIDENCE_RAW" in signal_name or "DISTRIBUTION_PRESSURE" in signal_name:
                    signal_score = raw_signal.clip(lower=0)
                elif "TREND_FORM" in signal_name or "LIQUIDITY_TIDE" in signal_name or "FF_AXIOM_CONSENSUS" in signal_name:
                    signal_score = raw_signal.clip(upper=0).abs()
                else:
                    signal_score = raw_signal
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                suppression_score_components += normalized_signal_score * weight
                for p_date in probe_dates_to_print:
                    print(f"      - [探针 {p_date.strftime('%Y-%m-%d')}] 打压信号 '{signal_name}' (权重: {weight:.2f}) 原始值: {raw_signal.loc[p_date]:.4f}, 归一化后: {normalized_signal_score.loc[p_date]:.4f}, 加权贡献: {(normalized_signal_score.loc[p_date] * weight):.4f}")
            suppression_score = suppression_score_components / total_suppression_weight
        else:
            suppression_score = pd.Series(0.0, index=df.index)
        for p_date in probe_dates_to_print:
            print(f"    -> [探针 {p_date.strftime('%Y-%m-%d')}] 综合打压分数 (Suppression Score): {suppression_score.loc[p_date]:.4f}")

        # 2. 计算吸筹证据分数 (Accumulation Evidence Score)
        if probe_dates_to_print:
            print(f"    -> [探针] 开始计算吸筹证据分数...")
        total_accumulation_weight = sum(v for k, v in accumulation_weights.items() if k != 'description' and isinstance(v, (int, float)))
        if total_accumulation_weight > 0:
            print(f"    -> [探针] 吸筹证据总权重: {total_accumulation_weight:.4f}")
            for signal_name, weight in accumulation_weights.items():
                if signal_name == 'description':
                    continue
                raw_signal = fetched_signals[signal_name]
                if "STEALTH_ACCUMULATION" in signal_name or "PANIC_WASHOUT_ACCUMULATION" in signal_name or "DECEPTIVE_ACCUMULATION" in signal_name or "ABSORPTION_ECHO" in signal_name:
                    signal_score = raw_signal.clip(lower=0)
                elif "FF_AXIOM_CONVICTION" in signal_name:
                    signal_score = raw_signal.clip(upper=0).abs()
                else:
                    signal_score = raw_signal
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                accumulation_score_components += normalized_signal_score * weight
                for p_date in probe_dates_to_print:
                    print(f"      - [探针 {p_date.strftime('%Y-%m-%d')}] 吸筹信号 '{signal_name}' (权重: {weight:.2f}) 原始值: {raw_signal.loc[p_date]:.4f}, 归一化后: {normalized_signal_score.loc[p_date]:.4f}, 加权贡献: {(normalized_signal_score.loc[p_date] * weight):.4f}")
            accumulation_score = accumulation_score_components / total_accumulation_weight
        else:
            accumulation_score = pd.Series(0.0, index=df.index)
        for p_date in probe_dates_to_print:
            print(f"    -> [探针 {p_date.strftime('%Y-%m-%d')}] 综合吸筹分数 (Accumulation Score): {accumulation_score.loc[p_date]:.4f}")

        # 3. 计算矛盾证据分数 (Contradiction Evidence Score)
        if probe_dates_to_print:
            print(f"    -> [探针] 开始计算矛盾证据分数...")
        total_contradiction_weight = sum(v for k, v in contradiction_weights.items() if k != 'description' and isinstance(v, (int, float)))
        if total_contradiction_weight > 0:
            print(f"    -> [探针] 矛盾证据总权重: {total_contradiction_weight:.4f}")
            for signal_name, weight in contradiction_weights.items():
                if signal_name == 'description':
                    continue
                raw_signal = fetched_signals[signal_name]
                if "BULLISH_DIVERGENCE" in signal_name or "CHIP_AXIOM_DIVERGENCE" in signal_name or "FUND_FLOW_BULLISH_DIVERGENCE" in signal_name:
                    signal_score = raw_signal.clip(lower=0)
                elif "PRICE_VS_RETAIL_CAPITULATION" in signal_name or "PROFIT_VS_FLOW" in signal_name:
                    signal_score = raw_signal.clip(upper=0).abs()
                else:
                    signal_score = raw_signal
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                contradiction_score_components += normalized_signal_score * weight
                for p_date in probe_dates_to_print:
                    print(f"      - [探针 {p_date.strftime('%Y-%m-%d')}] 矛盾信号 '{signal_name}' (权重: {weight:.2f}) 原始值: {raw_signal.loc[p_date]:.4f}, 归一化后: {normalized_signal_score.loc[p_date]:.4f}, 加权贡献: {(normalized_signal_score.loc[p_date] * weight):.4f}")
            contradiction_score = contradiction_score_components / total_contradiction_weight
        else:
            contradiction_score = pd.Series(0.0, index=df.index)
        for p_date in probe_dates_to_print:
            print(f"    -> [探针 {p_date.strftime('%Y-%m-%d')}] 综合矛盾分数 (Contradiction Score): {contradiction_score.loc[p_date]:.4f}")

        # 4. 计算情境调节器 (Context Modulators)
        if probe_dates_to_print:
            print(f"    -> [探针] 开始计算情境调节器分数...")
        total_context_weight = sum(v for k, v in context_modulator_weights.items() if k != 'description' and isinstance(v, (int, float)))
        if total_context_weight > 0:
            print(f"    -> [探针] 情境调节器总权重: {total_context_weight:.4f}")
            for signal_name, weight in context_modulator_weights.items():
                if signal_name == 'description':
                    continue
                raw_signal = fetched_signals[signal_name]
                signal_score = raw_signal.clip(lower=0)
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                context_modulator_score_components += normalized_signal_score * weight
                for p_date in probe_dates_to_print:
                    print(f"      - [探针 {p_date.strftime('%Y-%m-%d')}] 情境信号 '{signal_name}' (权重: {weight:.2f}) 原始值: {raw_signal.loc[p_date]:.4f}, 归一化后: {normalized_signal_score.loc[p_date]:.4f}, 加权贡献: {(normalized_signal_score.loc[p_date] * weight):.4f}")
            context_modulator = context_modulator_score_components / total_context_weight
        else:
            context_modulator = pd.Series(1.0, index=df.index)
        for p_date in probe_dates_to_print:
            print(f"    -> [探针 {p_date.strftime('%Y-%m-%d')}] 综合情境调节器 (Context Modulator): {context_modulator.loc[p_date]:.4f}")

        epsilon = 1e-6
        fused_score_raw = (
            (suppression_score + epsilon) *
            (accumulation_score + epsilon) *
            (contradiction_score + epsilon)
        )**(1/3)

        final_score = (fused_score_raw * context_modulator)**final_fusion_exponent
        final_score = final_score.clip(0, 1)
        final_score = final_score.where(final_score >= min_activation_threshold, 0.0)

        for p_date in probe_dates_to_print:
            print(f"    -> [探针 {p_date.strftime('%Y-%m-%d')}] 最终融合原始分数 (Fused Score Raw): {fused_score_raw.loc[p_date]:.4f}")
            print(f"    -> [探针 {p_date.strftime('%Y-%m-%d')}] 最终剧本分数 (Final Playbook Score): {final_score.loc[p_date]:.4f}")

        print(f"  -> {method_name} 计算完成。")
        return final_score.astype(np.float32)




