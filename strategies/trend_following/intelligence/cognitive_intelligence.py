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
    def __init__(self, strategy_instance, dynamic_thresholds: Dict):
        """
        初始化认知情报模块。
        :param strategy_instance: 策略主实例的引用。
        :param dynamic_thresholds: 动态阈值字典。
        """
        self.strategy = strategy_instance
        self.dynamic_thresholds = dynamic_thresholds

        # 修改开始 - 在 __init__ 中统一加载 debug_params，并存储为实例属性
        full_config_dict = {}
        if hasattr(self.strategy, 'params') and isinstance(self.strategy.params, dict):
            full_config_dict = self.strategy.params
        elif hasattr(self.strategy, 'config') and isinstance(self.strategy.config, dict): # 备用方案
            full_config_dict = self.strategy.config
        else:
            print(f"    -> [认知层警告] 策略实例没有 'params' 或 'config' 属性，或它们不是字典。调试参数加载可能失败。")

        # 从正确的配置源和路径加载 debug_params
        debug_params_config = get_params_block(full_config_dict, 'strategy_params.trend_follow.debug_params', {})
        self.probe_dates_list_str = debug_params_config.get('probe_dates', [])
        self.debug_enabled = debug_params_config.get('enabled', {}).get('value', False)
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
        """
        修改思路：
        1.  修正 `get_params_block` 的调用，使用 `self.strategy.params` 作为配置源，并使用完整的点分隔路径来正确加载嵌套的参数配置。
        2.  增加 `self.strategy.params` 结构探针，以确认其内容。
        3.  移除方法开头对 `enabled` 参数的检查，确保该方法总是执行其计算逻辑。
        4.  增强探针输出逻辑，确保在每个探测日期，详细打印构成“打压”、“吸筹”、“矛盾”和“情境调节器”
            这四大类分数的每一个原始信号值及其归一化后的值。
        5.  在方法开始时，通过 `_get_atomic_score` 预先检查所有所需信号的存在性，并获取其Series，
            确保后续计算的健壮性。
        6.  在方法开始处，增加对 `params` 字典以及各个 `weights` 字典内容的打印，以诊断配置加载问题。
        7.  从score_type_map中选择非COGNITIVE_*的信号作为输入，分为“打压证据”、“吸筹证据”和“矛盾证据”三大类。
        8.  对每类证据进行加权融合，得到各自的综合分数。
        9.  引入“情境调节器”，如深度底部区域和结构张力，对最终分数进行放大。
        10. 使用乘法模型将三类核心证据和情境调节器进行非线性融合，并通过指数放大，以捕捉剧本的共振效应。
        """
        method_name = "COGNITIVE_PLAYBOOK_SUPPRESSIVE_ACCUMULATION"
        print(f"  -> [认知层] 正在计算 {method_name}...")

        # 修改开始 - 确定配置根字典
        full_config_dict = {}
        if hasattr(self.strategy, 'params') and isinstance(self.strategy.params, dict):
            full_config_dict = self.strategy.params
        elif hasattr(self.strategy, 'config') and isinstance(self.strategy.config, dict):
            full_config_dict = self.strategy.config
        else:
            if self.debug_enabled: # 只有在调试启用时才打印警告
                print(f"    -> [探针警告] self.strategy.params 和 self.strategy.config 都不存在或不是字典类型。参数加载可能失败。")
        # 修改结束

        # 修改开始 - 修正参数加载路径，从 full_config_dict 中获取
        params = get_params_block(full_config_dict, 'strategy_params.trend_follow.cognitive_intelligence_params.cognitive_playbook_suppressive_accumulation_params', {})
        # 修改结束

        # 修改开始 - 使用实例属性 self.debug_enabled
        if self.debug_enabled:
            print(f"    -> [探针] self.strategy.params 类型: {type(full_config_dict)}")
            print(f"    -> [探针] self.strategy.params 顶层键: {list(full_config_dict.keys())}")
            if 'strategy_params' in full_config_dict:
                print(f"    -> [探针] self.strategy.params['strategy_params'] 顶层键: {list(full_config_dict['strategy_params'].keys())}")
            print(f"    -> [探针] {method_name} 加载的原始参数 (params): {params}")
        # 修改结束

        suppression_weights = get_param_value(params.get('suppression_weights'), {})
        accumulation_weights = get_param_value(params.get('accumulation_weights'), {})
        contradiction_weights = get_param_value(params.get('contradiction_weights'), {})
        context_modulator_weights = get_param_value(params.get('context_modulator_weights'), {})
        final_fusion_exponent = get_param_value(params.get('final_fusion_exponent'), 2.0)
        min_activation_threshold = get_param_value(params.get('min_activation_threshold'), 0.1)
        norm_window = get_param_value(params.get('norm_window'), 55)

        # 修改开始 - 使用实例属性 self.debug_enabled
        if self.debug_enabled:
            print(f"    -> [探针] suppression_weights: {suppression_weights}")
            print(f"    -> [探针] accumulation_weights: {accumulation_weights}")
            print(f"    -> [探针] contradiction_weights: {contradiction_weights}")
            print(f"    -> [探针] context_modulator_weights: {context_modulator_weights}")
        # 修改结束

        # --- 探针：获取所有需要探测的日期 ---
        probe_dates_to_print = []
        # 修改开始 - 使用实例属性 self.debug_enabled 和 self.probe_dates_list_str
        if self.debug_enabled and self.probe_dates_list_str:
            if not df.empty:
                df_index_tz = df.index.tz
                for date_str in self.probe_dates_list_str:
                    try:
                        current_probe_date = pd.to_datetime(date_str)
                        if df_index_tz is not None and current_probe_date.tz is None:
                            current_probe_date = current_probe_date.tz_localize(df_index_tz)
                        elif df_index_tz is None and current_probe_date.tz is not None:
                            current_probe_date = current_probe_date.tz_convert(None)
                        
                        if current_probe_date in df.index:
                            probe_dates_to_print.append(current_probe_date)
                        else:
                            print(f"    -> [探针警告] 探测日期 '{date_str}' (时区校准后: {current_probe_date}) 不在DataFrame索引中，跳过。")
                    except Exception as e:
                        print(f"    -> [探针警告] 无法解析或处理探针日期 '{date_str}': {e}")
        # 修改结束
        
        if probe_dates_to_print:
            print(f"    -> [探针] 准备为以下日期输出详细信息: {[d.strftime('%Y-%m-%d') for d in probe_dates_to_print]}")
        else:
            print(f"    -> [探针] 未找到有效的探测日期或调试未启用，将跳过详细探针输出。请检查 debug_params['probe_dates'] 和数据范围。")


        # --- 预先获取所有需要的信号，并存储在字典中，确保存在性 ---
        all_required_signals = set()
        all_required_signals.update(suppression_weights.keys())
        all_required_signals.update(accumulation_weights.keys())
        all_required_signals.update(contradiction_weights.keys())
        all_required_signals.update(context_modulator_weights.keys())

        fetched_signals = {}
        for signal_name in all_required_signals:
            fetched_signals[signal_name] = self._get_atomic_score(df, signal_name, default=0.0)
            if not isinstance(fetched_signals[signal_name], pd.Series):
                fetched_signals[signal_name] = pd.Series(fetched_signals[signal_name], index=df.index)
            else:
                fetched_signals[signal_name] = fetched_signals[signal_name].reindex(df.index).fillna(0.0)


        # 初始化所有分数为0，确保DataFrame操作的安全性
        suppression_score_components = pd.Series(0.0, index=df.index)
        accumulation_score_components = pd.Series(0.0, index=df.index)
        contradiction_score_components = pd.Series(0.0, index=df.index)
        context_modulator_score_components = pd.Series(0.0, index=df.index)


        # 1. 计算打压证据分数 (Suppression Evidence Score)
        if probe_dates_to_print:
            print(f"    -> [探针] 开始计算打压证据分数...")
        total_suppression_weight = sum(suppression_weights.values())
        if probe_dates_to_print:
            print(f"    -> [探针] 打压证据总权重: {total_suppression_weight:.4f}")
        if total_suppression_weight > 0:
            for signal_name, weight in suppression_weights.items():
                raw_signal = fetched_signals[signal_name]
                if "PRICE_DOWNWARD_MOMENTUM" in signal_name or "TREND_FORM" in signal_name or "LIQUIDITY_TIDE" in signal_name:
                    signal_score = raw_signal.clip(upper=0).abs()
                elif "DISTRIBUTION_PRESSURE" in signal_name or "DISTRIBUTION_INT" in signal_name or "STAGNATION_EVIDENCE_RAW" in signal_name:
                    signal_score = raw_signal.clip(lower=0)
                else:
                    signal_score = raw_signal
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                suppression_score_components += normalized_signal_score * weight
                for p_date in probe_dates_to_print:
                    print(f"      - [探针 {p_date.strftime('%Y-%m-%d')}] 打压信号 '{signal_name}' (权重: {weight:.2f}) 原始值: {raw_signal.loc[p_date]:.4f}, 归一化后: {normalized_signal_score.loc[p_date]:.4f}")
            suppression_score = suppression_score_components / total_suppression_weight
        else:
            suppression_score = pd.Series(0.0, index=df.index)
        for p_date in probe_dates_to_print:
            print(f"    -> [探针 {p_date.strftime('%Y-%m-%d')}] 综合打压分数 (Suppression Score): {suppression_score.loc[p_date]:.4f}")

        # 2. 计算吸筹证据分数 (Accumulation Evidence Score)
        if probe_dates_to_print:
            print(f"    -> [探针] 开始计算吸筹证据分数...")
        total_accumulation_weight = sum(accumulation_weights.values())
        if probe_dates_to_print:
            print(f"    -> [探针] 吸筹证据总权重: {total_accumulation_weight:.4f}")
        if total_accumulation_weight > 0:
            for signal_name, weight in accumulation_weights.items():
                raw_signal = fetched_signals[signal_name]
                if "CONVICTION" in signal_name or "COHERENT_DRIVE" in signal_name:
                    signal_score = raw_signal.clip(lower=0)
                else:
                    signal_score = raw_signal
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                accumulation_score_components += normalized_signal_score * weight
                for p_date in probe_dates_to_print:
                    print(f"      - [探针 {p_date.strftime('%Y-%m-%d')}] 吸筹信号 '{signal_name}' (权重: {weight:.2f}) 原始值: {raw_signal.loc[p_date]:.4f}, 归一化后: {normalized_signal_score.loc[p_date]:.4f}")
            accumulation_score = accumulation_score_components / total_accumulation_weight
        else:
            accumulation_score = pd.Series(0.0, index=df.index)
        for p_date in probe_dates_to_print:
            print(f"    -> [探针 {p_date.strftime('%Y-%m-%d')}] 综合吸筹分数 (Accumulation Score): {accumulation_score.loc[p_date]:.4f}")

        # 3. 计算矛盾证据分数 (Contradiction Evidence Score)
        if probe_dates_to_print:
            print(f"    -> [探针] 开始计算矛盾证据分数...")
        total_contradiction_weight = sum(contradiction_weights.values())
        if probe_dates_to_print:
            print(f"    -> [探针] 矛盾证据总权重: {total_contradiction_weight:.4f}")
        if total_contradiction_weight > 0:
            for signal_name, weight in contradiction_weights.items():
                raw_signal = fetched_signals[signal_name]
                signal_score = raw_signal.clip(lower=0)
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                contradiction_score_components += normalized_signal_score * weight
                for p_date in probe_dates_to_print:
                    print(f"      - [探针 {p_date.strftime('%Y-%m-%d')}] 矛盾信号 '{signal_name}' (权重: {weight:.2f}) 原始值: {raw_signal.loc[p_date]:.4f}, 归一化后: {normalized_signal_score.loc[p_date]:.4f}")
            contradiction_score = contradiction_score_components / total_contradiction_weight
        else:
            contradiction_score = pd.Series(0.0, index=df.index)
        for p_date in probe_dates_to_print:
            print(f"    -> [探针 {p_date.strftime('%Y-%m-%d')}] 综合矛盾分数 (Contradiction Score): {contradiction_score.loc[p_date]:.4f}")

        # 4. 计算情境调节器 (Context Modulators)
        if probe_dates_to_print:
            print(f"    -> [探针] 开始计算情境调节器分数...")
        total_context_weight = sum(context_modulator_weights.values())
        if probe_dates_to_print:
            print(f"    -> [探针] 情境调节器总权重: {total_context_weight:.4f}")
        if total_context_weight > 0:
            for signal_name, weight in context_modulator_weights.items():
                raw_signal = fetched_signals[signal_name]
                normalized_signal_score = normalize_score(raw_signal, df.index, norm_window, ascending=True)
                context_modulator_score_components += normalized_signal_score * weight
                for p_date in probe_dates_to_print:
                    print(f"      - [探针 {p_date.strftime('%Y-%m-%d')}] 情境信号 '{signal_name}' (权重: {weight:.2f}) 原始值: {raw_signal.loc[p_date]:.4f}, 归一化后: {normalized_signal_score.loc[p_date]:.4f}")
            context_modulator = context_modulator_score_components / total_context_weight
        else:
            context_modulator = pd.Series(1.0, index=df.index)
        for p_date in probe_dates_to_print:
            print(f"    -> [探针 {p_date.strftime('%Y-%m-%d')}] 综合情境调节器 (Context Modulator): {context_modulator.loc[p_date]:.4f}")

        # 5. 最终融合 (Final Fusion)
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




