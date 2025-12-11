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
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance
        self.min_evidence_threshold = 1e-9 # 最小证据阈值，避免对数运算错误
        self.norm_window = 55 # 统一归一化窗口，可根据需要调整

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
        if name in self.strategy.atomic_states:
            return self.strategy.atomic_states[name]
        elif name in df.columns:
            return df[name]
        else:
            print(f"    -> [认知层警告] 原子信号 '{name}' 不存在，无法作为证据！返回默认值 {default}。")
            return pd.Series(default, index=df.index) # 移除了末尾的 .index

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
        1.  定义“主力打压吸筹”剧本的核心要素：价格弱势/打压、主力吸筹行为、以及两者之间的矛盾（即在打压中吸筹）。
        2.  从score_type_map中选择非COGNITIVE_*的信号作为输入，分为“打压证据”、“吸筹证据”和“矛盾证据”三大类。
        3.  对每类证据进行加权融合，得到各自的综合分数。
        4.  引入“情境调节器”，如深度底部区域和结构张力，对最终分数进行放大。
        5.  使用乘法模型将三类核心证据和情境调节器进行非线性融合，并通过指数放大，以捕捉剧本的共振效应。
        6.  加入详细的探针输出，以便检查和调试每个关键计算节点的值。
        """
        method_name = "COGNITIVE_PLAYBOOK_SUPPRESSIVE_ACCUMULATION"
        print(f"  -> [认知层] 正在计算 {method_name}...")

        params = get_params_block(self.strategy, 'cognitive_playbook_suppressive_accumulation_params', {})
        if not get_param_value(params.get('enabled'), False):
            print(f"    -> {method_name} 未启用，返回0。")
            return pd.Series(0.0, index=df.index)

        suppression_weights = get_param_value(params.get('suppression_weights'), {})
        accumulation_weights = get_param_value(params.get('accumulation_weights'), {})
        contradiction_weights = get_param_value(params.get('contradiction_weights'), {})
        context_modulator_weights = get_param_value(params.get('context_modulator_weights'), {})
        final_fusion_exponent = get_param_value(params.get('final_fusion_exponent'), 2.0)
        min_activation_threshold = get_param_value(params.get('min_activation_threshold'), 0.1)
        norm_window = get_param_value(params.get('norm_window'), 55)

        # 初始化所有分数为0，确保DataFrame操作的安全性
        suppression_score_components = pd.Series(0.0, index=df.index)
        accumulation_score_components = pd.Series(0.0, index=df.index)
        contradiction_score_components = pd.Series(0.0, index=df.index)
        context_modulator_score_components = pd.Series(0.0, index=df.index)

        # --- 探针：原始信号值 ---
        # 获取当前探测日期，如果未设置则不进行详细打印
        probe_date_str = get_params_block(self.strategy, 'debug_params', {}).get('probe_dates', [None])[0]
        probe_date = pd.to_datetime(probe_date_str) if probe_date_str else None
        if probe_date is not None and probe_date in df.index:
            print(f"    -> [探针] {method_name} 原始信号值 (日期: {probe_date.strftime('%Y-%m-%d')}):")

        # 1. 计算打压证据分数 (Suppression Evidence Score)
        total_suppression_weight = sum(suppression_weights.values())
        if total_suppression_weight > 0:
            for signal_name, weight in suppression_weights.items():
                raw_signal = self._get_atomic_score(df, signal_name, default=0.0)
                # 针对双极性信号，负值代表打压，需要转换为正向分数
                if "PRICE_DOWNWARD_MOMENTUM" in signal_name or "TREND_FORM" in signal_name or "LIQUIDITY_TIDE" in signal_name:
                    # 这些信号负值代表打压/弱势，将其绝对值作为打压强度
                    signal_score = raw_signal.clip(upper=0).abs()
                elif "DISTRIBUTION_PRESSURE" in signal_name or "DISTRIBUTION_INT" in signal_name or "STAGNATION_EVIDENCE_RAW" in signal_name:
                    # 这些信号正值代表打压/风险
                    signal_score = raw_signal.clip(lower=0)
                else:
                    signal_score = raw_signal # 默认直接使用

                # 归一化到 [0, 1] 范围
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                suppression_score_components += normalized_signal_score * weight
                if probe_date is not None and probe_date in df.index:
                    print(f"      - 原始信号 '{signal_name}': {raw_signal.loc[probe_date]:.4f}, 归一化后: {normalized_signal_score.loc[probe_date]:.4f}")
            suppression_score = suppression_score_components / total_suppression_weight
        else:
            suppression_score = pd.Series(0.0, index=df.index)
        if probe_date is not None and probe_date in df.index:
            print(f"    -> [探针] 综合打压分数 (Suppression Score): {suppression_score.loc[probe_date]:.4f}")

        # 2. 计算吸筹证据分数 (Accumulation Evidence Score)
        total_accumulation_weight = sum(accumulation_weights.values())
        if total_accumulation_weight > 0:
            for signal_name, weight in accumulation_weights.items():
                raw_signal = self._get_atomic_score(df, signal_name, default=0.0)
                # 针对双极性信号，正值代表吸筹
                if "CONVICTION" in signal_name or "COHERENT_DRIVE" in signal_name:
                    signal_score = raw_signal.clip(lower=0)
                else:
                    signal_score = raw_signal # 默认直接使用

                # 归一化到 [0, 1] 范围
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                accumulation_score_components += normalized_signal_score * weight
                if probe_date is not None and probe_date in df.index:
                    print(f"      - 原始信号 '{signal_name}': {raw_signal.loc[probe_date]:.4f}, 归一化后: {normalized_signal_score.loc[probe_date]:.4f}")
            accumulation_score = accumulation_score_components / total_accumulation_weight
        else:
            accumulation_score = pd.Series(0.0, index=df.index)
        if probe_date is not None and probe_date in df.index:
            print(f"    -> [探针] 综合吸筹分数 (Accumulation Score): {accumulation_score.loc[probe_date]:.4f}")

        # 3. 计算矛盾证据分数 (Contradiction Evidence Score)
        # 矛盾证据是指在价格弱势/打压背景下，主力仍在积极吸筹的信号
        total_contradiction_weight = sum(contradiction_weights.values())
        if total_contradiction_weight > 0:
            for signal_name, weight in contradiction_weights.items():
                raw_signal = self._get_atomic_score(df, signal_name, default=0.0)
                # 这些信号的正值代表矛盾/看涨背离
                signal_score = raw_signal.clip(lower=0)
                # 归一化到 [0, 1] 范围
                normalized_signal_score = normalize_score(signal_score, df.index, norm_window, ascending=True)
                contradiction_score_components += normalized_signal_score * weight
                if probe_date is not None and probe_date in df.index:
                    print(f"      - 原始信号 '{signal_name}': {raw_signal.loc[probe_date]:.4f}, 归一化后: {normalized_signal_score.loc[probe_date]:.4f}")
            contradiction_score = contradiction_score_components / total_contradiction_weight
        else:
            contradiction_score = pd.Series(0.0, index=df.index)
        if probe_date is not None and probe_date in df.index:
            print(f"    -> [探针] 综合矛盾分数 (Contradiction Score): {contradiction_score.loc[probe_date]:.4f}")

        # 4. 计算情境调节器 (Context Modulators)
        total_context_weight = sum(context_modulator_weights.values())
        if total_context_weight > 0:
            for signal_name, weight in context_modulator_weights.items():
                raw_signal = self._get_atomic_score(df, signal_name, default=0.0)
                # 情境调节器通常是正向的，直接归一化
                normalized_signal_score = normalize_score(raw_signal, df.index, norm_window, ascending=True)
                context_modulator_score_components += normalized_signal_score * weight
                if probe_date is not None and probe_date in df.index:
                    print(f"      - 原始信号 '{signal_name}': {raw_signal.loc[probe_date]:.4f}, 归一化后: {normalized_signal_score.loc[probe_date]:.4f}")
            context_modulator = context_modulator_score_components / total_context_weight
        else:
            context_modulator = pd.Series(1.0, index=df.index) # 默认不调节
        if probe_date is not None and probe_date in df.index:
            print(f"    -> [探针] 综合情境调节器 (Context Modulator): {context_modulator.loc[probe_date]:.4f}")

        # 5. 最终融合 (Final Fusion)
        # 使用几何平均的变体，确保所有要素都存在时才能获得高分
        # (S * A * C)^(1/3) * M ^ E
        # 为了避免0值导致整个乘积为0，对每个分数进行小幅抬升
        epsilon = 1e-6
        fused_score_raw = (
            (suppression_score + epsilon) *
            (accumulation_score + epsilon) *
            (contradiction_score + epsilon)
        )**(1/3)

        # 应用情境调节器和指数放大
        final_score = (fused_score_raw * context_modulator)**final_fusion_exponent

        # 确保分数在 [0, 1] 范围内，并应用最小激活阈值
        final_score = final_score.clip(0, 1)
        final_score = final_score.where(final_score >= min_activation_threshold, 0.0)

        if probe_date is not None and probe_date in df.index:
            print(f"    -> [探针] 最终融合原始分数 (Fused Score Raw): {fused_score_raw.loc[probe_date]:.4f}")
            print(f"    -> [探针] 最终剧本分数 (Final Playbook Score): {final_score.loc[probe_date]:.4f}")

        print(f"  -> {method_name} 计算完成。")
        return final_score.astype(np.float32)




