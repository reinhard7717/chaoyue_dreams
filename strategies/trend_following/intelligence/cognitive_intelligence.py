# 文件: strategies/trend_following/intelligence/cognitive_intelligence.py
# 顶层认知合成模块
import pandas as pd
import numpy as np
from typing import Dict
from enum import Enum
from strategies.trend_following.utils import get_params_block, get_param_value
from strategies.trend_following.intelligence.micro_behavior_engine import MicroBehaviorEngine
from strategies.trend_following.intelligence.tactic_engine import TacticEngine

class MainForceState(Enum):
    """
    定义主力行为序列的各个状态。
    """
    IDLE = 0           # 闲置/观察期
    ACCUMULATING = 1   # 吸筹期
    WASHING = 2        # 洗盘期
    MARKUP = 3         # 拉升期
    DISTRIBUTING = 4   # 派发期
    COLLAPSE = 5       # 崩盘期

class CognitiveIntelligence:
    def __init__(self, strategy_instance):
        """
        初始化顶层认知合成模块。
        :param strategy_instance: 策略主实例的引用。
        """
        self.strategy = strategy_instance
        self.micro_behavior_engine = MicroBehaviorEngine(strategy_instance)
        self.tactic_engine = TacticEngine(strategy_instance)

    def _fuse_multi_level_scores(self, df: pd.DataFrame, base_name: str, weights: Dict[str, float] = None) -> pd.Series:
        """
        【V1.2 S+适配版】融合S+/S/A/B等多层置信度分数的辅助函数。
        - 核心升级 (本次修改):
          - [新增] 增加了对 'S_PLUS' 级别的识别与支持，并赋予其最高权重。
          - [健壮性] 修复了当只找到无等级单一信号时，未进行reindex的潜在bug。
        - 收益: 能够完整地消费包含S+级的全系列信号，确保最高置信度的信息被正确利用。
        """
        # 增加 S_PLUS 等级并调整权重
        if weights is None:
            weights = {'S_PLUS': 1.5, 'S': 1.0, 'A': 0.6, 'B': 0.3}
        
        total_score = pd.Series(0.0, index=df.index)
        total_weight = 0.0
        
        # 明确迭代顺序，确保 S_PLUS 被正确解析
        for level in ['S_PLUS', 'S', 'A', 'B']:
            if level not in weights: continue
            weight = weights[level]
            score_name = f"SCORE_{base_name}_{level}"
            if score_name in self.strategy.atomic_states:
                score_series = self.strategy.atomic_states[score_name]
                if len(score_series) > 0:
                    total_score += score_series.reindex(df.index).fillna(0.0) * weight
                    total_weight += weight
        
        if total_weight == 0:
            single_score_name = f"SCORE_{base_name}"
            if single_score_name in self.strategy.atomic_states:
                # 对单一信号也使用reindex和fillna，保证返回的Series索引正确且无NaN
                return self.strategy.atomic_states[single_score_name].reindex(df.index).fillna(0.5)
            return pd.Series(0.5, index=df.index)
            
        return (total_score / total_weight).clip(0, 1)

    def _get_atomic_score(self, df: pd.DataFrame, name: str, default=0.0) -> pd.Series:
        """
        【健壮性修复版】安全地从原子状态库中获取分数。
        - 核心修复: 移除对 self.strategy.df.index 的依赖，改为使用传入的 df.index。
                      这可以防止因状态不同步导致返回一个空Series。
        :param df: 当前正在处理的数据帧，用于获取正确的索引。
        :param name: 要获取的分数名称。
        :param default: 如果分数不存在，使用的默认值。
        :return: 一个与df索引长度一致的 pandas Series。
        """
        return self.strategy.atomic_states.get(name, pd.Series(default, index=df.index))

    def _normalize_score(self, series: pd.Series, window: int = 120, ascending: bool = True, default=0.5) -> pd.Series:
        """
        辅助函数：将一个Series进行滚动窗口排名归一化，生成0-1分。
        - 从其他情报模块迁移而来，保持架构一致性。
        :param series: 原始数据Series。
        :param window: 归一化滚动窗口。
        :param ascending: 归一化方向，True表示值越大分数越高。
        :param default: 填充NaN的默认值。
        :return: 归一化后的0-1分数Series。
        """
        if series is None or series.empty:
            # 如果输入为空，根据情况返回一个填充了默认值的Series
            # 假设 self.strategy.df_indicators.index 是可用的主索引
            return pd.Series(default, index=self.strategy.df_indicators.index)
        min_periods = max(1, window // 5)
        rank = series.rolling(window=window, min_periods=min_periods).rank(pct=True)
        score = rank if ascending else 1 - rank
        return score.fillna(default).astype(np.float32)

    def synthesize_fused_risk_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.2 智能信号消费版】风险元融合模块
        - 核心升级 (本次修改):
          - [配置驱动] 无需修改代码，本方法会自动消费配置文件中新增的 `SCORE_CHIP_FALSE_ACCUMULATION_RISK` 信号，
                        并将其纳入“筹码剧本风险”的计算中。
        - 收益:
          - 实现了对“顶部派发式假集中”这一核心风险的量化评估和融合。
        """
        # print("        -> [风险元融合模块 V3.2 智能信号消费版] 启动...") # 修改: 更新版本号
        states = {}
        p_fused_risk = get_params_block(self.strategy, 'fused_risk_scoring')
        if not get_param_value(p_fused_risk.get('enabled'), True):
            states['COGNITIVE_FUSED_RISK_SCORE'] = pd.Series(0.0, index=df.index, dtype=np.float32)
            return states
        risk_categories = p_fused_risk.get('risk_categories', {})
        p_dynamic_weighting = p_fused_risk.get('dynamic_weighting_params', {})
        base_weights = p_dynamic_weighting.get('base_weights', {})
        context_adjustments = p_dynamic_weighting.get('context_adjustments', {})
        p_fusion_params = p_fused_risk.get('intra_dimension_fusion_params', {})
        secondary_risk_discount = p_fusion_params.get('secondary_risk_discount', 0.3)
        p_resonance = p_fused_risk.get('resonance_penalty_params', {})
        fused_dimension_scores = {}
        default_series = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 维度内融合 ---
        for category_name, signals in risk_categories.items():
            if category_name == "说明": continue
            category_signal_scores = []
            for signal_name, signal_params in signals.items():
                if signal_name == "说明": continue
                # 检查原子状态库中是否存在该信号
                if signal_name not in self.strategy.atomic_states:
                    # print(f"          -> [风险融合警告] 缺少上游信号: {signal_name}，跳过。")
                    continue
                atomic_score = self._get_atomic_score(df, signal_name, 0.0)
                is_inverse = signal_params.get('inverse', False)
                processed_score = 1.0 - atomic_score if is_inverse else atomic_score
                weight = signal_params.get('weight', 1.0)
                final_signal_score = processed_score * weight
                category_signal_scores.append(final_signal_score)
            if category_signal_scores:
                stacked_scores = np.stack([s.values for s in category_signal_scores], axis=1)
                sorted_scores = np.sort(stacked_scores, axis=1)
                primary_risk_values = sorted_scores[:, -1]
                secondary_risk_values = sorted_scores[:, -2] if sorted_scores.shape[1] > 1 else 0
                dimension_risk_values = primary_risk_values + secondary_risk_values * secondary_risk_discount
                dimension_risk_score = pd.Series(dimension_risk_values, index=df.index)
                fused_dimension_scores[category_name] = dimension_risk_score
                states[f'FUSED_RISK_SCORE_{category_name.upper()}'] = dimension_risk_score.astype(np.float32)
            else:
                fused_dimension_scores[category_name] = default_series.copy()
        # --- 2. 维度间融合 ---
        total_fused_risk_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        is_early_stage = self.strategy.atomic_states.get('CONTEXT_TREND_STAGE_EARLY', pd.Series(False, index=df.index))
        is_late_stage = self.strategy.atomic_states.get('CONTEXT_TREND_STAGE_LATE', pd.Series(False, index=df.index))
        for category_name, weight in base_weights.items():
            if category_name in fused_dimension_scores:
                current_weight = pd.Series(weight, index=df.index)
                if get_param_value(p_dynamic_weighting.get('enabled'), True):
                    early_adjustments = context_adjustments.get("CONTEXT_TREND_STAGE_EARLY", {})
                    if category_name in early_adjustments:
                        adjustment_factor = early_adjustments[category_name]
                        current_weight = current_weight.where(~is_early_stage, current_weight * adjustment_factor)
                    late_adjustments = context_adjustments.get("CONTEXT_TREND_STAGE_LATE", {})
                    if category_name in late_adjustments:
                        adjustment_factor = late_adjustments[category_name]
                        current_weight = current_weight.where(~is_late_stage, current_weight * adjustment_factor)
                total_fused_risk_score += fused_dimension_scores[category_name] * current_weight
        # --- 3. 风险共振惩罚 ---
        if get_param_value(p_resonance.get('enabled'), True):
            core_dims = get_param_value(p_resonance.get('core_risk_dimensions'), [])
            min_dims = get_param_value(p_resonance.get('min_dimensions_for_resonance'), 2)
            threshold = get_param_value(p_resonance.get('risk_score_threshold'), 150)
            penalty_multiplier = get_param_value(p_resonance.get('penalty_multiplier'), 1.2)
            high_risk_dimension_count = pd.Series(0, index=df.index)
            for dim in core_dims:
                if dim in fused_dimension_scores:
                    high_risk_dimension_count += (fused_dimension_scores[dim] > threshold).astype(int)
            is_resonance_triggered = (high_risk_dimension_count >= min_dims)
            total_fused_risk_score = total_fused_risk_score.where(~is_resonance_triggered, total_fused_risk_score * penalty_multiplier)
            states['FUSED_RISK_RESONANCE_PENALTY_ACTIVE'] = is_resonance_triggered
        states['COGNITIVE_FUSED_RISK_SCORE'] = total_fused_risk_score.astype(np.float32)
        return states

    def synthesize_prime_chip_opportunity(self, df: pd.DataFrame) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
        """
        【V2.1 数值化信号升级版】黄金筹码机会元融合模块
        - 核心职责: 将多个独立的、描述筹码结构健康度的S/A/B三级数值评分，通过
                      “维度内置信度加权”和“维度间重要性加权”两步，融合成一个顶层的、
                      能精细度量机会“成色”的元分数。
        - 核心升级: 将原先生成的布尔型机会信号 'CHIP_STRUCTURE_PRIME_OPPORTUNITY_S'，
                      升级为数值型信号 'SIGNAL_PRIME_CHIP_OPPORTUNITY_S'。
                      新信号的值为“元分数”与一个固定阈值的差值，直接反映机会的强度。
        - 收益: 实现了从“信号有无”到“机会质量”的升维，为决策提供了更平滑、更鲁棒的依据。
        """
        print("        -> [黄金筹码机会元融合模块 V2.1 数值化信号升级版] 启动...")
        states = {}
        new_scores = {}
        atomic = self.strategy.atomic_states
        p_module = get_params_block(self.strategy, 'prime_chip_opportunity_params_v2', {})
        if not get_param_value(p_module.get('enabled'), True):
            return states, new_scores
        # --- 1. 加载参数：维度权重与置信度权重 ---
        dim_weights = get_param_value(p_module.get('dimension_weights'), {
            'structure_health': 0.35, 'core_holder': 0.30, 'net_support': 0.20, 'cost_structure': 0.15
        })
        conf_weights = get_param_value(p_module.get('confidence_weights'), {
            'S': 1.0, 'A': 0.6, 'B': 0.3
        })
        total_conf_weight = sum(conf_weights.values())
        # --- 2. 军备检查：获取所有S/A/B三级评分 ---
        score_map = {
            'structure_health': ['SCORE_STRUCTURE_BULLISH_RESONANCE_S', 'SCORE_STRUCTURE_BULLISH_RESONANCE_A', 'SCORE_STRUCTURE_BULLISH_RESONANCE_B'],
            'core_holder': ['SCORE_CORE_HOLDER_BULLISH_RESONANCE_S', 'SCORE_CORE_HOLDER_BULLISH_RESONANCE_A', 'SCORE_CORE_HOLDER_BULLISH_RESONANCE_B'],
            'net_support': [None, 'SCORE_NET_SUPPORT_BULLISH_RESONANCE_A', 'SCORE_NET_SUPPORT_BULLISH_RESONANCE_B'], # Net Support 没有S级
            'cost_structure': ['SCORE_COST_STRUCTURE_BULLISH_RESONANCE_S', 'SCORE_COST_STRUCTURE_BULLISH_RESONANCE_A', 'SCORE_COST_STRUCTURE_BULLISH_RESONANCE_B']
        }
        all_required_scores = [s for group in score_map.values() for s in group if s]
        missing_scores = [s for s in all_required_scores if s not in atomic]
        if missing_scores:
            print(f"          -> [警告] synthesize_prime_chip_opportunity黄金机会融合模块缺少上游分数: {missing_scores}，模块已跳过。")
            return states, new_scores
        # --- 3. 维度内融合：计算每个维度的综合强度分 ---
        fused_dimension_scores = {}
        default_series = pd.Series(0.0, index=df.index, dtype=np.float32)
        for dim_name, (s_score_name, a_score_name, b_score_name) in score_map.items():
            s_score = atomic.get(s_score_name, default_series) if s_score_name else default_series
            a_score = atomic.get(a_score_name, default_series) if a_score_name else default_series
            b_score = atomic.get(b_score_name, default_series) if b_score_name else default_series
            # 置信度加权求和，并归一化
            fused_score = (s_score * conf_weights['S'] + a_score * conf_weights['A'] + b_score * conf_weights['B']) / total_conf_weight
            fused_dimension_scores[dim_name] = fused_score
        # --- 4. 维度间融合：计算最终的“黄金机会”元分数 ---
        final_prime_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        for dim_name, weight in dim_weights.items():
            final_prime_score += fused_dimension_scores[dim_name] * weight
        new_scores['CHIP_SCORE_PRIME_OPPORTUNITY_S'] = final_prime_score.clip(0, 1)
        # --- 5. 基于元分数，生成数值化机会信号 ---
        # 获取用于生成信号的阈值参数，原参数名 'prime_score_threshold_for_bool' 已优化
        threshold = get_param_value(p_module.get('prime_score_threshold'), 0.7)
        # 计算数值化信号：元分数 - 阈值。正值代表机会，值越大机会越强
        prime_opportunity_numerical_signal = new_scores['CHIP_SCORE_PRIME_OPPORTUNITY_S'] - threshold
        # 使用新的命名规范 SIGNAL_...，并将其添加到 states 字典中，替换原布尔信号
        states['SIGNAL_PRIME_CHIP_OPPORTUNITY_S'] = prime_opportunity_numerical_signal.astype(np.float32)
        print("        -> [黄金筹码机会元融合模块 V2.1 数值化信号升级版] 计算完毕。")
        return states, new_scores


    def synthesize_market_engine_states(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V2.2 性能优化版】市场引擎状态融合模块
        - 核心升级 (本次修改):
          - [性能优化] 将原有的 `pd.concat([...]).max(axis=1)` 逻辑，重构为使用 `np.maximum.reduce`。
        - 核心重构 (V2.1逻辑保留):
          - [信号适配] 将“引擎失速”的判断依据升级为消费更通用的“顶部反转”融合分数。
        - 收益:
          - 通过避免创建临时DataFrame，显著降低了内存占用并提升了计算速度。
          - 统一了对引擎失效的判断标准，信号源更可靠。
        """
        # print("        -> [市场引擎状态融合模块 V2.2 性能优化版] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 提取引擎失效的多个症状分数 ---
        engine_stalling_score = self._fuse_multi_level_scores(df, 'BEHAVIOR_TOP_REVERSAL')
        vpa_stagnation_score = self._get_atomic_score(df, 'SCORE_RISK_VPA_STAGNATION', 0.0)
        bearish_divergence_score = self._get_atomic_score(df, 'SCORE_RISK_MTF_RSI_DIVERGENCE_S', 0.0)
        # --- 2. 定义触发的战场环境分数 ---
        danger_zone_score = atomic.get('COGNITIVE_SCORE_RISK_HIGH_LEVEL_ZONE', default_score)
        # --- 3. 最终裁定 ---
        # 1. 将所有症状分数的底层NumPy数组收集到一个列表中
        symptom_scores_list = [
            engine_stalling_score.values,
            vpa_stagnation_score.values,
            bearish_divergence_score.values
        ]
        # 2. 使用np.maximum.reduce高效计算元素级最大值
        max_symptom_values = np.maximum.reduce(symptom_scores_list)
        # 3. 将结果包装回一个Pandas Series
        max_symptom_score_series = pd.Series(max_symptom_values, index=df.index).fillna(0.0)
        final_risk_score = danger_zone_score * max_symptom_score_series
        states['COGNITIVE_SCORE_ENGINE_FAILURE_S'] = final_risk_score.astype(np.float32)
        # --- 4. 更新原子状态库 ---
        self.strategy.atomic_states.update(states)
        # print("        -> [市场引擎状态融合模块 V2.2] 计算完毕。")
        return df

    def synthesize_contextual_zone_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V2.8 性能优化版】战场上下文评分模块
        - 核心升级 (本次修改):
          - [性能优化] 将原有的 `pd.DataFrame(risk_scores).max(axis=1)` 逻辑，重构为使用 `np.maximum.reduce`。
        - 核心升级 (V2.7逻辑保留):
          - [信号适配] 全面审查并更新了所有筹码和行为维度的风险信号消费。
        - 收益:
          - 通过避免创建大型临时DataFrame，显著降低了内存占用并提升了计算速度。
          - 确保了评分基于最新的终极信号。
        """
        # print("        -> [战场上下文评分模块 V2.8 性能优化版] 启动...")
        risk_scores = {
            'bias': self._get_atomic_score(df, 'SCORE_BIAS_OVERBOUGHT_EXTENT', 0.0),
            'exhaustion': np.maximum(
                self._get_atomic_score(df, 'SCORE_RISK_MOMENTUM_EXHAUSTION', 0.0).values,
                self._get_atomic_score(df, 'SCORE_RISK_PROFIT_EXHAUSTION_S', 0.0).values
            ),
            'deceptive_rally': self._get_atomic_score(df, 'CHIP_SCORE_FUSED_DECEPTIVE_RALLY', 0.0),
            'chip_top_reversal': self._fuse_multi_level_scores(df, 'TOP_REVERSAL'),
            'structural_weakness': self._get_atomic_score(df, 'SCORE_MTF_STRUCTURAL_WEAKNESS_RISK_S', 0.0),
            'engine_stalling': self._fuse_multi_level_scores(df, 'BEHAVIOR_TOP_REVERSAL'),
            'volume_spike_down': self._get_atomic_score(df, 'SCORE_VOL_PRICE_PANIC_DOWN_RISK', 0.0),
            'chip_divergence': self._fuse_multi_level_scores(df, 'FALLING_RESONANCE'),
            'chip_fault': self._fuse_multi_level_scores(df, 'FAULT_RISK_TOP_REVERSAL'),
            'structural_fault': self._fuse_multi_level_scores(df, 'STRUCTURE_BEARISH_RESONANCE'),
            'fund_flow_reversal': self._get_atomic_score(df, 'FF_SCORE_REVERSAL_TOP_HIGH', 0.0),
            'mechanics_reversal': self._fuse_multi_level_scores(df, 'MECHANICS_TOP_REVERSAL'),
            'retail_frenzy': self._get_atomic_score(df, 'FF_SCORE_RETAIL_RESONANCE_FRENZY_HIGH', 0.0),
            'ultimate_breakdown': self._get_atomic_score(df, 'COGNITIVE_SCORE_BREAKDOWN_RESONANCE_S', 0.0),
            'ultimate_confirmation': self._get_atomic_score(df, 'COGNITIVE_ULTIMATE_BEARISH_CONFIRMATION_S', 0.0),
            'price_deviation_risk': self._get_atomic_score(df, 'SCORE_RISK_PRICE_DEVIATION_S', 0.0),
            'structural_fault_risk': self._get_atomic_score(df, 'SCORE_RISK_STRUCTURAL_FAULT_S', 0.0),
        }
        # 将 exhaustion 的 numpy 数组转换为 pandas Series
        risk_scores['exhaustion'] = pd.Series(risk_scores['exhaustion'], index=df.index)
        # 1. 将所有风险分数的底层NumPy数组收集到一个列表中
        risk_values_list = [score.values for score in risk_scores.values()]
        # 2. 使用np.maximum.reduce计算最大值，这比创建DataFrame快得多
        max_risk_values = np.maximum.reduce(risk_values_list)
        # 3. 将结果包装回一个Pandas Series
        high_level_zone_score = pd.Series(max_risk_values, index=df.index)
        df['COGNITIVE_SCORE_RISK_HIGH_LEVEL_ZONE'] = high_level_zone_score
        self.strategy.atomic_states['COGNITIVE_SCORE_RISK_HIGH_LEVEL_ZONE'] = df['COGNITIVE_SCORE_RISK_HIGH_LEVEL_ZONE']
        high_risk_threshold = high_level_zone_score.rolling(120).quantile(0.85)
        is_in_high_level_zone = high_level_zone_score > high_risk_threshold
        self.strategy.atomic_states['CONTEXT_RISK_HIGH_LEVEL_ZONE'] = is_in_high_level_zone
        # print("        -> [战场上下文评分模块 V2.8] 计算完毕。")
        return df

    def diagnose_trend_stage_score(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V401.16 性能优化版】趋势阶段评分模块
        - 核心升级 (本次修改):
          - [性能优化] 将 `pd.concat([...]).max(axis=1)` 逻辑重构为使用 `np.maximum`，避免创建临时DataFrame。
        - 核心升级 (V401.15逻辑保留):
          - [信号适配] 修正了风险维度解析逻辑，使其能正确消费终极信号。
        - 收益:
          - 提升了计算效率并降低了内存占用。
          - 确保了对上涨末期风险的评估能够纳入最高质量的信号。
        """
        # print("        -> [趋势阶段评分模块 V401.16 性能优化版] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        p_trend_stage_scoring = get_params_block(self.strategy, 'trend_stage_scoring_params')
        if not get_param_value(p_trend_stage_scoring.get('enabled'), True):
            states['COGNITIVE_SCORE_CONTEXT_LATE_STAGE'] = pd.Series(0.0, index=df.index, dtype=np.float32)
            states['CONTEXT_TREND_STAGE_EARLY'] = pd.Series(False, index=df.index)
            # print("        -> [趋势阶段评分模块] 已在配置中禁用，跳过计算。")
            return states
        signal_definitions = get_param_value(p_trend_stage_scoring.get('weights'), {})
        # --- 1. 计算“上涨初期”的量化分数 (Early Stage Score) ---
        ascent_structure_score = atomic.get('COGNITIVE_SCORE_ACCUMULATION_BREAKOUT_S', default_score)
        yearly_high = df['high_D'].rolling(250, min_periods=60).max()
        yearly_low = df['low_D'].rolling(250, min_periods=60).min()
        price_range = (yearly_high - yearly_low).replace(0, 1e-9) # 使用一个极小值代替np.nan
        price_position_score = 1 - ((df['close_D'] - yearly_low) / price_range)
        price_position_score = price_position_score.clip(0, 1).fillna(0.5)
        early_stage_score = pd.Series(
            np.maximum(ascent_structure_score.values, price_position_score.values),
            index=df.index
        )
        df['COGNITIVE_SCORE_TREND_STAGE_EARLY'] = early_stage_score
        self.strategy.atomic_states['COGNITIVE_SCORE_TREND_STAGE_EARLY'] = df['COGNITIVE_SCORE_TREND_STAGE_EARLY']
        states['CONTEXT_TREND_STAGE_EARLY'] = early_stage_score > 0.6
        # --- 2. 计算“上涨末期”的原始风险累积分 ---
        vpa_stagnation_score = self._get_atomic_score(df, 'SCORE_RISK_VPA_STAGNATION', 0.0)
        vpa_volume_accelerating_score = self._get_atomic_score(df, 'SCORE_RISK_VPA_VOLUME_ACCELERATING', 0.0)
        vpa_efficiency_decline_score = self._get_atomic_score(df, 'SCORE_RISK_VPA_EFFICIENCY_DECLINING', 0.0)
        vpa_risk_score_arr = np.maximum.reduce([
            vpa_stagnation_score.values,
            vpa_volume_accelerating_score.values,
            vpa_efficiency_decline_score.values
        ])
        vpa_risk_score_series = pd.Series(vpa_risk_score_arr, index=df.index)
        risk_dimension_scores = []
        fuse_list = ["TOP_REVERSAL", "FALLING_RESONANCE", "STRUCTURE_BEARISH_RESONANCE", "MTF_BEARISH_RESONANCE", "MECHANICS_TOP_REVERSAL", "PATTERN_TOP_REVERSAL"]
        for name, weight in signal_definitions.items():
            if name == "vpa_risk_score_series":
                risk_dimension_scores.append(vpa_risk_score_series * weight)
            elif name in fuse_list:
                risk_dimension_scores.append(self._fuse_multi_level_scores(df, name) * weight)
            else:
                risk_dimension_scores.append(self._get_atomic_score(df, name, 0.0) * weight)
        score_components = [s.to_numpy(dtype=np.float32) for s in risk_dimension_scores]
        late_stage_raw_score = pd.Series(np.add.reduce(score_components), index=df.index, dtype=np.float32)
        late_stage_raw_score = late_stage_raw_score.fillna(0.0)
        start_threshold = get_param_value(p_trend_stage_scoring.get('late_stage_start_threshold'), 160)
        full_threshold = get_param_value(p_trend_stage_scoring.get('late_stage_full_threshold'), 300)
        scaling_range = full_threshold - start_threshold
        scaling_range = max(scaling_range, 1)
        late_stage_prob_score = ((late_stage_raw_score - start_threshold) / scaling_range).clip(0, 1)
        states['COGNITIVE_SCORE_CONTEXT_LATE_STAGE'] = late_stage_prob_score.astype(np.float32)
        states['CONTEXT_TREND_STAGE_LATE'] = late_stage_prob_score > 0.5
        # print("        -> [趋势阶段评分模块 V401.16] 计算完毕。")
        return states

    def synthesize_tactical_opportunities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.3 终极力学信号适配版】战术机会与潜在风险合成模块
        - 核心重构 (本次修改):
          - [信号适配] 将对旧版力学信号 (`SCORE_FV_*`) 的引用，全面升级为消费
                        `DynamicMechanicsEngine` V3.0 产出的终极信号 (`SCORE_DYN_*`)。
        - 收益: 确保战术机会的判断基于最新、最可靠的七维动态力学信号。
        """
        # print("        -> [战术机会与潜在风险合成模块 V1.3 终极力学信号适配版] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 战术 1: 缺口回补支撑机会 ---
        gap_support_strength_score = atomic.get('SCORE_GAP_SUPPORT_ACTIVE', default_score)
        healthy_pullback_score = atomic.get('COGNITIVE_SCORE_PULLBACK_HEALTHY_S', default_score)
        states['COGNITIVE_SCORE_OPP_GAP_SUPPORT_PULLBACK'] = (gap_support_strength_score * healthy_pullback_score).astype(np.float32)
        # --- 战术 2: 斐波那契关键位反弹确认机会 ---
        fib_rebound_score = self._fuse_multi_level_scores(df, 'FIB_REBOUND')
        bottom_reversal_confirmation = atomic.get('COGNITIVE_SCORE_BOTTOM_REVERSAL_RESONANCE_S', default_score)
        states['COGNITIVE_SCORE_OPP_FIB_REBOUND_CONFIRMED'] = (fib_rebound_score * bottom_reversal_confirmation).astype(np.float32)
        # --- 战术 3: 纯粹进攻性动能机会 ---
        # 将旧信号替换为新版S级进攻共振信号
        pure_offensive_momentum = self._fuse_multi_level_scores(df, 'DYN_BULLISH_RESONANCE')
        trend_quality = atomic.get('COGNITIVE_SCORE_TREND_QUALITY', default_score)
        states['COGNITIVE_SCORE_OPP_PURE_MOMENTUM'] = (pure_offensive_momentum * trend_quality).astype(np.float32)
        # --- 战术 4: 混乱扩张风险 ---
        # 将旧信号替换为新版S级风险扩张信号
        chaotic_expansion_risk = self._fuse_multi_level_scores(df, 'DYN_BEARISH_RESONANCE')
        high_level_zone_risk = atomic.get('COGNITIVE_SCORE_RISK_HIGH_LEVEL_ZONE', default_score)
        states['COGNITIVE_SCORE_RISK_CHAOTIC_EXPANSION'] = (chaotic_expansion_risk * high_level_zone_risk).astype(np.float32)
        self.strategy.atomic_states.update(states)
        # print(f"        -> [战术机会与潜在风险合成模块 V1.3] 计算完毕，新增 {len(states)} 个信号。")
        return df

    def synthesize_trend_sustainability_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.3 行为信号升级版】趋势可持续性与衰竭诊断模块
        - 核心重构 (本次修改):
          - [信号适配] 将对旧的、间接的认知层反转信号的引用，升级为直接消费由 `BehavioralIntelligence`
                        V4.0 生成的、更基础、更可靠的多层次行为反转信号。
        - 收益: 显著提升了趋势可持续性判断的准确性和信号源的纯净度。
        """
        # print("        -> [趋势可持续性诊断模块 V1.3 行为信号升级版] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 提取核心上游信号 ---
        trend_quality = atomic.get('COGNITIVE_SCORE_TREND_QUALITY', default_score)
        # 直接消费来自行为层的、经过多级融合的、更高质量的反转信号
        top_reversal_potential = self._fuse_multi_level_scores(df, 'BEHAVIOR_TOP_REVERSAL')
        bottom_reversal_potential = self._fuse_multi_level_scores(df, 'BEHAVIOR_BOTTOM_REVERSAL')
        high_level_zone_context = atomic.get('COGNITIVE_SCORE_RISK_HIGH_LEVEL_ZONE', default_score)
        oversold_context = atomic.get('SCORE_RSI_OVERSOLD_EXTENT', default_score)
        # --- 2. 计算“上升趋势可持续性”评分 ---
        # --- 获取“虚假集中风险”分数 ---
        false_accumulation_risk = atomic.get('SCORE_CHIP_FALSE_ACCUMULATION_RISK', default_score)
        # --- 在原有基础上，额外乘以 (1 - 虚假集中风险分) ---
        states['COGNITIVE_SCORE_TREND_SUSTAINABILITY_UP'] = (
            trend_quality * 
            (1 - top_reversal_potential) * 
            (1 - false_accumulation_risk)
        ).astype(np.float32)
        # --- 3. 计算“上升趋势衰竭”风险评分 ---
        trend_fatigue_risk = np.maximum(top_reversal_potential.values, false_accumulation_risk.values)
        states['COGNITIVE_SCORE_TREND_FATIGUE_RISK'] = (pd.Series(trend_fatigue_risk, index=df.index) * high_level_zone_context).astype(np.float32)
        # --- 4. 计算“下跌趋势衰竭”机会评分 ---
        states['COGNITIVE_SCORE_TREND_FATIGUE_OPP'] = (bottom_reversal_potential * oversold_context).astype(np.float32)
        # --- 5. 更新原子状态库 ---
        self.strategy.atomic_states.update(states)
        # print("        -> [趋势可持续性诊断模块 V1.3] 计算完毕。")
        return df

    def synthesize_consolidation_breakout_signals(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V1.0 新增】盘整突破机会合成模块
        - 核心职责: 消费结构层识别出的“盘整中继模式”数值化评分，并结合
                      “放量”或“强阳线”等点火信号，生成一个经过确认的、
                      更高质量的顶层认知突破机会分数。
        - 核心逻辑: 最终分数 = 昨日高质量盘整得分 * 今日点火信号强度
        - 收益: 将未被利用的底层盘整信号转化为顶层战术情报，丰富了策略的决策依据。
        """
        # print("        -> [盘整突破机会合成模块 V1.0] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        triggers = self.strategy.trigger_events
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        default_series = pd.Series(False, index=df.index)
        # --- 1. 提取并融合“盘整”战备(Setup)信号 ---
        # 使用辅助函数融合S/A/B三级盘整模式分数，得到综合的“战备质量分”
        # 更新对辅助函数的调用
        consolidation_setup_score = self._fuse_multi_level_scores(df, 'PATTERN_CONSOLIDATION')
        # --- 2. 提取并融合“点火”(Trigger)信号 ---
        # 点火源1: 放量突破
        volume_ignition_score = atomic.get('SCORE_VOL_PRICE_IGNITION_UP', default_score)
        # 点火源2: 显性反转K线 (如大阳线)
        reversal_candle_trigger = triggers.get('TRIGGER_DOMINANT_REVERSAL', default_series).astype(float)
        # 取最强的点火信号作为当日的点火强度
        trigger_score = np.maximum(volume_ignition_score.values, reversal_candle_trigger.values)
        trigger_series = pd.Series(trigger_score, index=df.index)
        # --- 3. 融合生成认知层“盘整突破机会”分数 ---
        # 逻辑: 昨日战备就绪(高质量盘整) * 今日点火 = 突破机会
        final_score_series = consolidation_setup_score.shift(1).fillna(0.0) * trigger_series
        states['COGNITIVE_SCORE_CONSOLIDATION_BREAKOUT_OPP_A'] = final_score_series.astype(np.float32)
        # --- 4. 更新原子状态库 ---
        self.strategy.atomic_states.update(states)
        # print("        -> [盘整突破机会合成模块 V1.0] 计算完毕。")
        return df

    def synthesize_breakdown_resonance_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.7 终极资金流信号适配版】多域崩溃共振分数合成模块
        - 核心升级: [信号适配] 将对资金流维度信号的消费，从旧的 'FF_SCORE_SEPTAFECTA_RESONANCE_DOWN_HIGH'
                    适配为消费由 FundFlowIntelligence V19.0 生成的 'FF_BEARISH_RESONANCE' 终极信号。
        """
        # print("        -> [多域崩溃共振分数合成模块 V1.7 终极资金流信号适配版] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 提取七大领域的核心“崩溃”分数 ---
        # 领域1 (行为): 消费最新的多层次看跌共振信号
        behavioral_breakdown = self._fuse_multi_level_scores(df, 'BEHAVIOR_BEARISH_RESONANCE')
        # 领域2 (力学): 结构力学看跌共振分
        mechanics_breakdown = self._fuse_multi_level_scores(df, 'DYN_BEARISH_RESONANCE')
        # 领域3 (筹码): 筹码看跌共振分
        chip_breakdown = self._fuse_multi_level_scores(df, 'FALLING_RESONANCE')
        # 领域4 (资金流): 七位一体看跌共振分
        fund_flow_breakdown = self._fuse_multi_level_scores(df, 'FF_BEARISH_RESONANCE') # 适配新的终极资金流看跌共振信号
        # 领域5 (波动率): S级波动率崩溃分
        volatility_breakdown = self._fuse_multi_level_scores(df, 'VOL_BREAKDOWN')
        # 领域6 (终极结构): 融合了结构元信号与形态元信号的最高级别确认分
        structural_confirmation = atomic.get('COGNITIVE_ULTIMATE_BEARISH_CONFIRMATION_S', default_score)
        # 领域7 (ATR波幅): ATR扩张衰竭风险分
        atr_exhaustion = self._get_atomic_score(df, 'SCORE_ATR_EXPANSION_EXHAUSTION_RISK', default_score)
        # --- 2. 交叉验证：生成“多域崩溃共振”元分数 ---
        breakdown_resonance_score = (
            behavioral_breakdown * mechanics_breakdown * chip_breakdown *
            fund_flow_breakdown * volatility_breakdown * structural_confirmation *
            atr_exhaustion
        ).astype(np.float32)
        states['COGNITIVE_SCORE_BREAKDOWN_RESONANCE_S'] = breakdown_resonance_score
        # --- 3. 更新原子状态库 ---
        self.strategy.atomic_states.update(states)
        # print("        -> [多域崩溃共振分数合成模块 V1.7 终极资金流信号适配版] 计算完毕。")
        return df

    def synthesize_trend_regime_signals(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V2.3 终极资金流信号适配版】趋势政权融合模块
        - 核心升级: [信号适配] 将对资金流维度信号的消费，从旧的 'FF_SCORE_SEPTAFECTA_*'
                    适配为消费由 FundFlowIntelligence V19.0 生成的 'FF_*_RESONANCE' 终极信号。
        """
        # print("        -> [趋势政权融合模块 V2.3 终极资金流信号适配版] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 提取核心原子分数 ---
        trending_regime_score = atomic.get('SCORE_TRENDING_REGIME', default_score)
        # 使用辅助函数融合S/A/B三级均线共振分数
        bullish_confluence_score = self._fuse_multi_level_scores(df, 'MA_BULLISH_RESONANCE')
        bearish_confluence_score = self._fuse_multi_level_scores(df, 'MA_BEARISH_RESONANCE')
        # 从布尔信号升级为数值化评分
        rsi_bullish_accel_score = atomic.get('SCORE_RSI_BULLISH_ACCEL', default_score)
        rsi_bearish_accel_score = atomic.get('SCORE_RSI_BEARISH_ACCEL_RISK', default_score)
        # 提取资金流最高级别确认信号
        fund_flow_ignition_score = self._fuse_multi_level_scores(df, 'FF_BULLISH_RESONANCE') # 适配新的终极资金流看涨共振信号
        fund_flow_breakdown_score = self._fuse_multi_level_scores(df, 'FF_BEARISH_RESONANCE') # 适配新的终极资金流看跌共振信号
        # --- 2. 融合生成S级认知分数 ---
        states['COGNITIVE_SCORE_TREND_REGIME_IGNITION_S'] = (
            trending_regime_score * bullish_confluence_score * rsi_bullish_accel_score * fund_flow_ignition_score
        ).astype(np.float32)
        states['COGNITIVE_SCORE_TREND_REGIME_BREAKDOWN_S'] = (
            trending_regime_score * bearish_confluence_score * rsi_bearish_accel_score * fund_flow_breakdown_score
        ).astype(np.float32)
        # --- 3. 更新原子状态库 ---
        self.strategy.atomic_states.update(states)
        # print("        -> [趋势政权融合模块 V2.3 终极资金流信号适配版] 计算完毕。")
        return df

    def synthesize_volatility_breakout_signals(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V2.5 终极信号适配版】波动率突破融合模块
        - 核心升级: [信号适配] 将对筹码维度的机会和风险信号的消费，全面升级为
                    ChipIntelligence 产出的 'RISING_RESONANCE' 和 'FALLING_RESONANCE' 终极信号。
        - 收益: 确保波动率突破的确认基于最可靠的筹码情报。
        """
        # print("        -> [波动率突破融合模块 V2.5 终极信号适配版] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 提取核心原子分数 ---
        # 使用新的融合函数来生成环境分
        compression_setup_score = self._fuse_multi_level_scores(df, 'VOL_COMPRESSION')
        expansion_setup_score = self._fuse_multi_level_scores(df, 'VOL_EXPANSION')
        # 动态信号 (Trigger)
        norm_window = 120
        min_periods = 24
        score_vol_expanding = df['SLOPE_5_BBW_21_2.0_D'].rolling(norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        score_vol_accelerating = df['ACCEL_5_BBW_21_2.0_D'].rolling(norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        score_price_trending_up = df['SLOPE_5_close_D'].rolling(norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        score_price_accelerating_up = df['ACCEL_5_close_D'].rolling(norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        score_price_trending_down = 1 - score_price_trending_up
        score_price_accelerating_down = 1 - score_price_accelerating_up
        # 确认信号 (Confirmation)
        volume_up_confirm_score = atomic.get('SCORE_VOL_PRICE_IGNITION_UP', default_score)
        volume_down_confirm_score = atomic.get('SCORE_VOL_PRICE_PANIC_DOWN_RISK', default_score)
        chip_opportunity_score = self._fuse_multi_level_scores(df, 'RISING_RESONANCE') # 适配新的终极上升共振信号
        chip_bearish_score = self._fuse_multi_level_scores(df, 'FALLING_RESONANCE') # 适配新的终极下跌共振信号
        # --- 2. 上升共振 (Breakout) 分数合成 ---
        trigger_b_bullish = score_vol_expanding * score_price_trending_up
        states['COGNITIVE_SCORE_VOL_BREAKOUT_B'] = (compression_setup_score * trigger_b_bullish).astype(np.float32)
        trigger_a_bullish = trigger_b_bullish * volume_up_confirm_score
        states['COGNITIVE_SCORE_VOL_BREAKOUT_A'] = (compression_setup_score * trigger_a_bullish).astype(np.float32)
        trigger_s_bullish = trigger_a_bullish * score_vol_accelerating * score_price_accelerating_up * chip_opportunity_score
        states['COGNITIVE_SCORE_VOL_BREAKOUT_S'] = (compression_setup_score * trigger_s_bullish).astype(np.float32)
        # --- 3. 下跌共振 (Breakdown) 分数合成 ---
        trigger_b_bearish = score_vol_expanding * score_price_trending_down
        states['COGNITIVE_SCORE_VOL_BREAKDOWN_B'] = (expansion_setup_score * trigger_b_bearish).astype(np.float32)
        trigger_a_bearish = trigger_b_bearish * volume_down_confirm_score
        states['COGNITIVE_SCORE_VOL_BREAKDOWN_A'] = (expansion_setup_score * trigger_a_bearish).astype(np.float32)
        trigger_s_bearish = trigger_a_bearish * score_vol_accelerating * score_price_accelerating_down * chip_bearish_score
        states['COGNITIVE_SCORE_VOL_BREAKDOWN_S'] = (expansion_setup_score * trigger_s_bearish).astype(np.float32)
        # --- 4. 更新原子状态库 ---
        self.strategy.atomic_states.update(states)
        # print("        -> [波动率突破融合模块 V2.5 终极信号适配版] 计算完毕。")
        return df

    def synthesize_structural_fusion_scores(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V2.4 终极结构层信号适配版】结构化元信号融合模块
        - 核心升级 (本次修改):
          - [信号适配] 将对MA、MTF、力学等多个结构层信号的消费，统一升级为消费
                        由 StructuralIntelligence V2.0 生成的 'STRUCTURE_*' 终极信号。
          - [逻辑简化] 融合维度从四个（Foundation, Mechanics, Behavior）简化为三个（Foundation, Structure, Behavior）。
        - 收益: 融合逻辑更清晰，信号源质量更高，避免了结构层内部的重复计算和潜在矛盾。
        """
        # print("        -> [结构化元信号融合模块 V2.4 终极结构层信号适配版] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 获取各领域最高置信度的S级信号或融合分数 ---
        # 领域1 (Foundation)
        foundation_bullish = self._fuse_multi_level_scores(df, 'FOUNDATION_BULLISH_RESONANCE')
        foundation_bearish = self._fuse_multi_level_scores(df, 'FOUNDATION_BEARISH_RESONANCE')
        foundation_bottom = self._fuse_multi_level_scores(df, 'FOUNDATION_BOTTOM_REVERSAL')
        foundation_top = self._fuse_multi_level_scores(df, 'FOUNDATION_TOP_REVERSAL')
        # 领域2 (Structure) - 消费统一的结构层终极信号
        structure_bullish = self._fuse_multi_level_scores(df, 'STRUCTURE_BULLISH_RESONANCE') # 替换 mechanics_bullish
        structure_bearish = self._fuse_multi_level_scores(df, 'STRUCTURE_BEARISH_RESONANCE') # 替换 mechanics_bearish
        structure_bottom = self._fuse_multi_level_scores(df, 'STRUCTURE_BOTTOM_REVERSAL') # 替换 mechanics_bottom
        structure_top = self._fuse_multi_level_scores(df, 'STRUCTURE_TOP_REVERSAL') # 替换 mechanics_top
        # 领域3 (Behavior)
        behavior_bullish = self._fuse_multi_level_scores(df, 'BEHAVIOR_BULLISH_RESONANCE')
        behavior_bearish = self._fuse_multi_level_scores(df, 'BEHAVIOR_BEARISH_RESONANCE')
        behavior_bottom = self._fuse_multi_level_scores(df, 'BEHAVIOR_BOTTOM_REVERSAL')
        behavior_top = self._fuse_multi_level_scores(df, 'BEHAVIOR_TOP_REVERSAL')
        # --- 2. 三域联合作战：融合生成元信号 ---
        # 确保所有Series都有相同的索引
        all_scores = [
            foundation_bullish, foundation_bearish, foundation_bottom, foundation_top,
            structure_bullish, structure_bearish, structure_bottom, structure_top, # 使用新的 structure_* 信号
            behavior_bullish, behavior_bearish, behavior_bottom, behavior_top
        ]
        for i, score in enumerate(all_scores):
            if not isinstance(score, pd.Series) or score.index.empty:
                all_scores[i] = default_score.copy() # 使用copy避免多处引用同一个对象
            else:
                all_scores[i] = score.reindex(df.index).fillna(0.0)
        # 重新解包以确保安全
        (foundation_bullish, foundation_bearish, foundation_bottom, foundation_top,
         structure_bullish, structure_bearish, structure_bottom, structure_top, # 使用新的 structure_* 信号
         behavior_bullish, behavior_bearish, behavior_bottom, behavior_top) = all_scores
        # 更新融合逻辑为三域融合
        states['COGNITIVE_FUSION_BULLISH_RESONANCE_S'] = (foundation_bullish * structure_bullish * behavior_bullish).astype(np.float32) # 使用 structure_bullish
        states['COGNITIVE_FUSION_BEARISH_RESONANCE_S'] = (foundation_bearish * structure_bearish * behavior_bearish).astype(np.float32) # 使用 structure_bearish
        states['COGNITIVE_FUSION_BOTTOM_REVERSAL_S'] = (foundation_bottom * structure_bottom * behavior_bottom).astype(np.float32) # 使用 structure_bottom
        states['COGNITIVE_FUSION_TOP_REVERSAL_S'] = (foundation_top * structure_top * behavior_top).astype(np.float32) # 使用 structure_top
        self.strategy.atomic_states.update(states)
        # print("        -> [结构化元信号融合模块 V2.4 终极结构层信号适配版] 计算完毕。")
        return df

    def synthesize_ultimate_confirmation_scores(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V1.2 架构统一版】终极确认融合模块
        - 核心职责: 寻找“连续性共振元信号”与“离散性模式信号”同时发生的最高置信度信号。
        - 本次升级 (V1.2):
          - [架构统一] 废除了对形态信号的硬编码直接获取方式，改为使用标准的 `_fuse_multi_level_scores` 辅助函数进行融合。
        - 收益: 极大地提升了模块的鲁棒性，即使只有A/B级形态信号也能正常计算，并与其他模块的调用方式保持架构一致。
        """
        # print("        -> [终极确认融合模块 V1.2 架构统一版] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        required_fusion_signals = [
            'COGNITIVE_FUSION_BULLISH_RESONANCE_S', 'COGNITIVE_FUSION_BEARISH_RESONANCE_S',
            'COGNITIVE_FUSION_BOTTOM_REVERSAL_S', 'COGNITIVE_FUSION_TOP_REVERSAL_S'
        ]
        # 检查第一类依赖信号
        missing_fusion_signals = [s for s in required_fusion_signals if s not in atomic]
        if missing_fusion_signals:
            print(f"          -> [警告] 终极确认融合缺少核心[融合元]信号: {sorted(missing_fusion_signals)}，模块已跳过！")
            return df
        default_series = pd.Series(0.0, index=df.index, dtype=np.float32)
        fusion_bullish = atomic.get('COGNITIVE_FUSION_BULLISH_RESONANCE_S', default_series)
        fusion_bearish = atomic.get('COGNITIVE_FUSION_BEARISH_RESONANCE_S', default_series)
        fusion_bottom = atomic.get('COGNITIVE_FUSION_BOTTOM_REVERSAL_S', default_series)
        fusion_top = atomic.get('COGNITIVE_FUSION_TOP_REVERSAL_S', default_series)
        pattern_bullish = self._fuse_multi_level_scores(df, 'PATTERN_BULLISH_RESONANCE')
        pattern_bearish = self._fuse_multi_level_scores(df, 'PATTERN_BEARISH_RESONANCE')
        pattern_bottom = self._fuse_multi_level_scores(df, 'PATTERN_BOTTOM_REVERSAL')
        pattern_top = self._fuse_multi_level_scores(df, 'PATTERN_TOP_REVERSAL')
        states['COGNITIVE_ULTIMATE_BULLISH_CONFIRMATION_S'] = (fusion_bullish * pattern_bullish).astype(np.float32)
        states['COGNITIVE_ULTIMATE_BEARISH_CONFIRMATION_S'] = (fusion_bearish * pattern_bearish).astype(np.float32)
        states['COGNITIVE_ULTIMATE_BOTTOM_CONFIRMATION_S'] = (fusion_bottom * pattern_bottom).astype(np.float32)
        states['COGNITIVE_ULTIMATE_TOP_CONFIRMATION_S'] = (fusion_top * pattern_top).astype(np.float32)
        self.strategy.atomic_states.update(states)
        # print("        -> [终极确认融合模块 V1.2] 计算完毕。")
        return df

    def synthesize_ignition_resonance_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V2.0 王牌信号增强版】多域点火共振分数合成模块
        - 核心升级 (本次修改):
          - [新增王牌信号] 将资金流层的“信念突破”剧本 (`SCORE_FF_PLAYBOOK_CONVICTION_BREAKOUT`) 作为一个高优先级的独立点火源。
          - [融合优化] 最终点火分 = MAX(常规多域共振, 信念突破分)。确保这种高质量信号不会被其他领域的弱信号稀释。
        - 收益: 极大提升了对最强、最健康突破行情的捕捉能力。
        """
        # print("        -> [多域点火共振分数合成模块 V2.0 王牌信号增强版] 启动...") # 修改: 更新版本号
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        
        # --- 1. 提取各领域的核心“点火”分数 ---
        chip_playbook_ignition = self._get_atomic_score(df, 'SCORE_CHIP_PLAYBOOK_VACUUM_BREAKOUT', 0.0)
        chip_consensus_ignition = self._fuse_multi_level_scores(df, 'CHIP_BULLISH_RESONANCE')
        behavioral_ignition = self._fuse_multi_level_scores(df, 'BEHAVIOR_BULLISH_RESONANCE')
        structural_breakout = self._fuse_multi_level_scores(df, 'STRUCTURE_BULLISH_RESONANCE')
        mechanics_ignition = self._fuse_multi_level_scores(df, 'DYN_BULLISH_RESONANCE')
        volatility_breakout = self._fuse_multi_level_scores(df, 'VOL_BREAKOUT')
        
        # --- 提取新的“王牌”资金流点火信号 ---
        # 旧的资金流信号
        fund_flow_ignition_old = self._fuse_multi_level_scores(df, 'FF_BULLISH_RESONANCE')
        # 新的、更高质量的“信念突破”信号
        fund_flow_conviction_breakout = self._get_atomic_score(df, 'SCORE_FF_PLAYBOOK_CONVICTION_BREAKOUT', 0.0)
        # --- 2. 交叉验证：生成“多域点火共振”元分数 ---
        # 常规共振分
        general_ignition_resonance = (
            behavioral_ignition * structural_breakout * mechanics_ignition *
            chip_consensus_ignition * fund_flow_ignition_old * volatility_breakout
        )
        # --- 使用 np.maximum 融合，确保王牌信号的最高优先级 ---
        # 最终点火分 = MAX(筹码真空突破, 常规多域共振, 资金信念突破)
        ignition_resonance_score = np.maximum.reduce([
            chip_playbook_ignition.values, 
            general_ignition_resonance.values,
            fund_flow_conviction_breakout.values
        ]).astype(np.float32)

        states['COGNITIVE_SCORE_IGNITION_RESONANCE_S'] = pd.Series(ignition_resonance_score, index=df.index)
        self.strategy.atomic_states.update(states)
        # print("        -> [多域点火共振分数合成模块 V2.0] 计算完毕。") # 修改: 更新版本号
        return df

    def synthesize_reversal_resonance_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V2.5 融合算法升级版】多域反转共振分数合成模块
        - 核心升级 (本次修改):
          - [算法升级] 将原有的“乘法融合”（几何平均）方式，升级为“加权平均融合”（算术平均）。
          - [配置驱动] 新增从配置文件读取各领域权重的逻辑，使融合策略可配置。
        - 收益:
          - 降低了融合过程对单个输入信号极端值的敏感度，避免了单一错误信号被过度放大，使融合结果更稳健。
          - 提升了策略的灵活性和可维护性。
        """
        # print("        -> [多域反转共振分数合成模块 V2.5 融合算法升级版] 启动...") # 修改: 更新版本号
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 从配置中读取权重，并使用加权平均 ---
        p = get_params_block(self.strategy, 'reversal_resonance_params', {})
        # 如果配置不存在，则使用默认的等权重
        bottom_weights = get_param_value(p.get('bottom_resonance_weights'), {'mechanics': 0.4, 'chip': 0.4, 'foundation': 0.2})
        top_weights = get_param_value(p.get('top_resonance_weights'), {'mechanics': 0.4, 'chip': 0.4, 'foundation': 0.2})
        # --- 1. 合成“底部反转共振”分数 ---
        # 提取各领域信号
        mechanics_bottom_score = self._fuse_multi_level_scores(df, 'DYN_BOTTOM_REVERSAL')
        chip_bottom_score = self._fuse_multi_level_scores(df, 'CHIP_BOTTOM_REVERSAL') # 注意：这里消费的是我们刚刚改造好的、包含位置上下文的信号
        foundation_bottom_score = self._fuse_multi_level_scores(df, 'FOUNDATION_BOTTOM_REVERSAL')
        behavior_bottom_score = self._fuse_multi_level_scores(df, 'BEHAVIOR_BOTTOM_REVERSAL') # 同样消费改造后的信号
        # 使用加权平均进行融合
        total_bottom_score = pd.Series(0.0, index=df.index)
        total_bottom_weight = 0.0
        # 将各领域分数放入字典，方便按权重key调用
        bottom_sources = {
            'mechanics': mechanics_bottom_score,
            'chip': chip_bottom_score,
            'foundation': foundation_bottom_score,
            'behavior': behavior_bottom_score
        }
        for domain, weight in bottom_weights.items():
            if domain in bottom_sources and weight > 0:
                total_bottom_score += bottom_sources[domain] * weight
                total_bottom_weight += weight
        if total_bottom_weight > 0:
            bottom_reversal_score = total_bottom_score / total_bottom_weight
        else:
            bottom_reversal_score = default_score
        states['COGNITIVE_SCORE_BOTTOM_REVERSAL_RESONANCE_S'] = bottom_reversal_score.astype(np.float32)
        # --- 2. 合成“顶部反转共振”分数 (对称逻辑) ---
        mechanics_top_score = self._fuse_multi_level_scores(df, 'DYN_TOP_REVERSAL')
        chip_top_score = self._fuse_multi_level_scores(df, 'CHIP_TOP_REVERSAL')
        foundation_top_score = self._fuse_multi_level_scores(df, 'FOUNDATION_TOP_REVERSAL')
        behavior_top_score = self._fuse_multi_level_scores(df, 'BEHAVIOR_TOP_REVERSAL')
        total_top_score = pd.Series(0.0, index=df.index)
        total_top_weight = 0.0
        top_sources = {
            'mechanics': mechanics_top_score,
            'chip': chip_top_score,
            'foundation': foundation_top_score,
            'behavior': behavior_top_score
        }
        for domain, weight in top_weights.items():
            if domain in top_sources and weight > 0:
                total_top_score += top_sources[domain] * weight
                total_top_weight += weight
        if total_top_weight > 0:
            top_reversal_score = total_top_score / total_top_weight
        else:
            top_reversal_score = default_score
        states['COGNITIVE_SCORE_TOP_REVERSAL_RESONANCE_S'] = top_reversal_score.astype(np.float32)

        self.strategy.atomic_states.update(states)
        # print("        -> [多域反转共振分数合成模块 V2.5] 计算完毕。") # 修改: 更新版本号
        return df

    def synthesize_divergence_risks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V2.2 行为信号升级版】多维背离风险融合模块
        - 核心重构 (本次修改):
          - [信号适配] 将“引擎背离”的判断依据，从旧的、特定的信号，升级为消费
                        `BehavioralIntelligence` V4.0产出的、更通用的“顶部反转”融合分数。
        - 收益: 统一了对顶部风险的判断标准，信号源更可靠。
        """
        # print("        -> [多维背离风险融合模块 V2.2 行为信号升级版] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 提取各维度的背离风险分数 ---
        oscillator_divergence_score = np.maximum(atomic.get('SCORE_RISK_MTF_RSI_DIVERGENCE_S', default_score).values, atomic.get('SCORE_MACD_BEARISH_DIVERGENCE_RISK', default_score).values)
        price_momentum_divergence_score = self._get_atomic_score(df, 'SCORE_MTF_TOP_DIVERGENCE_A', 0.0)
        exhaustion_divergence_score = self._get_atomic_score(df, 'SCORE_MTF_TOP_DIVERGENCE_S', 0.0)
        # 将旧的、特定的行为背离信号，替换为新的、通用的顶部反转融合分
        engine_divergence_score = self._fuse_multi_level_scores(df, 'BEHAVIOR_TOP_REVERSAL')
        # --- 2. 定义触发的战场环境分数 ---
        danger_zone_score = atomic.get('COGNITIVE_SCORE_RISK_HIGH_LEVEL_ZONE', default_score)
        # --- 3. 最终裁定 ---
        max_divergence_score = np.maximum.reduce([oscillator_divergence_score, price_momentum_divergence_score.values, exhaustion_divergence_score.values, engine_divergence_score.values])
        final_risk_score = danger_zone_score * pd.Series(max_divergence_score, index=df.index)
        states['COGNITIVE_SCORE_MULTI_DIMENSIONAL_DIVERGENCE_S'] = final_risk_score.astype(np.float32)
        # --- 4. 更新原子状态库 ---
        self.strategy.atomic_states.update(states)
        # print("        -> [多维背离风险融合模块 V2.2] 计算完毕。")
        return df

    def synthesize_pullback_states(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V2.3 终极信号适配版】认知层回踩状态合成模块
        - 核心升级: [信号适配] 将“筹码稳定性”的评估，从消费旧信号升级为消费
                    ChipIntelligence 产出的 'FALLING_RESONANCE' 终极信号。
        - 收益: 对回踩期间筹码稳定性的判断更加精确和可靠。
        """
        # print("        -> [认知层回踩状态合成模块 V2.3 终极信号适配版] 启动...")
        states = {}
        # --- 1. 定义通用回踩条件和环境分数 ---
        is_pullback_day = (df['pct_change_D'] < 0).astype(float)
        constructive_context_score = self._get_atomic_score(df, 'COGNITIVE_SCORE_TREND_QUALITY', 0.0)
        # --- 2. 合成“健康回踩”分数 (Healthy Pullback Score) ---
        gentle_drop_score = (1 - (df['pct_change_D'].abs() / 0.05)).clip(0, 1).fillna(0.0)
        shrinking_volume_score = self._get_atomic_score(df, 'SCORE_VOL_WEAKENING_DROP', 0.0) 
        # 相位分数在-1(波谷)到+1(波峰)之间。我们希望在接近波谷时分数高。
        # (1 - phase) / 2 将其映射到 0(波峰) 到 1(波谷)
        cycle_trough_score = (1 - self._get_atomic_score(df, 'DOMINANT_CYCLE_PHASE', 0.0).fillna(0.0)) / 2.0 
        winner_holding_tight_score = 1.0 - self._fuse_multi_level_scores(df, 'TOP_REVERSAL') # 获利盘稳定度应由顶部反转风险评估
        chip_stable_score = 1.0 - self._fuse_multi_level_scores(df, 'FALLING_RESONANCE') # 适配新的终极下跌共振信号
        healthy_pullback_score = (
            is_pullback_day * constructive_context_score *
            gentle_drop_score * shrinking_volume_score *
            winner_holding_tight_score * chip_stable_score *
            (1 + cycle_trough_score * 0.5) # 周期确认提供最多50%的加成
        )
        states['COGNITIVE_SCORE_PULLBACK_HEALTHY_S'] = healthy_pullback_score.astype(np.float32)
        # --- 3. 合成“打压式回踩”分数 (Suppressive Pullback Score) ---
        significant_drop_score = (df['pct_change_D'].abs() / 0.07).clip(0, 1).fillna(0.0)
        # 此处应消费行为层的恐慌抛售信号，而非筹码层的
        panic_selling_score = self._fuse_multi_level_scores(df, 'BEHAVIOR_BEARISH_RESONANCE')
        suppressive_pullback_score = (
            is_pullback_day * constructive_context_score *
            significant_drop_score * panic_selling_score * winner_holding_tight_score
        )
        states['COGNITIVE_SCORE_PULLBACK_SUPPRESSIVE_S'] = suppressive_pullback_score.astype(np.float32)
        # --- 4. 更新原子状态库 ---
        self.strategy.atomic_states.update(states)
        # print("        -> [认知层回踩状态合成模块 V2.3 终极信号适配版] 计算完毕。")
        return df

    def synthesize_holding_risks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.1 数值化升级版】认知层持仓风险合成模块
        - 核心职责: 基于顶层融合的“趋势质量分”的动态变化，诊断持仓健康度是否出现“失速”风险。
        - 本次升级: 将原有的布尔信号升级为数值化的“失速风险分”，通过量化趋势质量分的
                    下降幅度，更精确地度量风险的严重程度。
        """
        # print("        -> [认知层持仓风险合成模块 V1.1 数值化升级版] 启动...")
        states = {}
        # 更新对辅助函数的调用
        # 使用更高维度的“趋势质量分”来判断健康度
        trend_quality_score = self._get_atomic_score(df, 'COGNITIVE_SCORE_TREND_QUALITY', 0.5)
        # 计算质量分的下降幅度作为风险源
        quality_decline = (trend_quality_score.shift(1) - trend_quality_score).clip(0)
        # 对下降幅度进行归一化，生成0-1的风险分
        # 假设质量分单日最大合理降幅为0.3，以此为基准归一化
        stalling_risk_score = (quality_decline / 0.3).clip(0, 1).fillna(0.0)
        states['COGNITIVE_SCORE_HOLD_RISK_HEALTH_STALLING'] = stalling_risk_score.astype(np.float32)
        # “失速”定义为：昨日仍在改善，今日不再改善，且风险分超过动态阈值
        is_improving = trend_quality_score > trend_quality_score.shift(1)
        was_improving = is_improving.shift(1).fillna(False)
        is_not_improving_now = ~is_improving
        # 使用滚动85%分位数作为动态阈值，避免硬编码
        stalling_threshold = stalling_risk_score.rolling(120).quantile(0.85)
        is_significant_stalling = stalling_risk_score > stalling_threshold
        states['HOLD_RISK_HEALTH_STALLING'] = was_improving & is_not_improving_now & is_significant_stalling
        self.strategy.atomic_states.update(states)
        # print("        -> [认知层持仓风险合成模块 V1.1 数值化升级版] 计算完毕。") 
        return df

    def synthesize_opportunity_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V2.0 职责净化版】顶层机会与风险评分模块
        - 核心重构: 移除了所有本地计算逻辑，使其职责回归纯粹的“元融合”。
                      现在直接消费由 ChipIntelligence 生成的高质量“行为-筹码”融合分数。
        - 收益: 严格遵循了分层架构原则，认知层只负责顶层融合，不再执行底层计算。
        """
        # print("        -> [顶层机会风险评分模块 V2.0 职责净化版] 启动...")
        atomic = self.strategy.atomic_states
        # --- 1. 合成“认知层·底部反转机会”得分 ---
        # 直接消费来自 ChipIntelligence 的高质量融合分
        washout_absorption_score = atomic.get('CHIP_SCORE_FUSED_WASHOUT_ABSORPTION', pd.Series(0.5, index=df.index))
        # 将其与行为层的“反转K线”信号进行最终融合
        reversal_candle_score = atomic.get('TRIGGER_DOMINANT_REVERSAL', pd.Series(False, index=df.index)).astype(float)
        # 最终认知分 = 融合分 * (1 + K线确认带来的额外加成)
        cognitive_bottom_reversal_score = (washout_absorption_score * (1 + reversal_candle_score * 0.2)).clip(0, 1)
        df['COGNITIVE_SCORE_OPP_BOTTOM_REVERSAL'] = cognitive_bottom_reversal_score
        self.strategy.atomic_states['COGNITIVE_SCORE_OPP_BOTTOM_REVERSAL'] = df['COGNITIVE_SCORE_OPP_BOTTOM_REVERSAL']
        # --- 2. 合成“认知层·顶部派发风险”得分 ---
        # 直接消费来自 ChipIntelligence 的高质量融合分
        deceptive_rally_score = atomic.get('CHIP_SCORE_FUSED_DECEPTIVE_RALLY', pd.Series(0.5, index=df.index))
        # 将其与认知层的“高位危险区”上下文进行最终融合
        high_zone_score = atomic.get('COGNITIVE_SCORE_RISK_HIGH_LEVEL_ZONE', pd.Series(0.5, index=df.index))
        # 最终认知分 = 诱多派发分 * 危险区上下文分
        cognitive_top_distribution_score = deceptive_rally_score * high_zone_score
        df['COGNITIVE_SCORE_RISK_TOP_DISTRIBUTION'] = cognitive_top_distribution_score
        self.strategy.atomic_states['COGNITIVE_SCORE_RISK_TOP_DISTRIBUTION'] = df['COGNITIVE_SCORE_RISK_TOP_DISTRIBUTION']
        # print("        -> [顶层机会风险评分模块 V2.0 职责净化版] 计算完毕。")
        return df

    def synthesize_trend_quality_score(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V2.1 调用修复版】趋势质量融合评分模块
        - 核心升级 (V2.0):
          - [精细加权] 废除旧的、单一的筹码健康分。现在消费7个独立的“筹码支柱健康分”，并根据不同支柱对趋势质量的重要性赋予不同权重。
          - [信号适配] 将其他领域的信号消费全面适配为最新的终极信号。
        - 本次修复 (V2.1):
          - [调用修复] 修正了 `get_param_value` 的错误调用，应使用标准的字典 `.get()` 方法来获取权重配置，解决了 TypeError。
        - 收益: 对趋势质量的评估更加精细和准确，能区分“控盘驱动的趋势”和“情绪驱动的趋势”。
        """
        # print("        -> [趋势质量融合评分模块 V2.1 调用修复版] 启动...")
        # --- 1. 提取各领域的核心健康度评分 ---
        behavior_health_score = 1.0 - self._fuse_multi_level_scores(df, 'BEHAVIOR_TOP_REVERSAL')
        fund_flow_health_score = self._fuse_multi_level_scores(df, 'FF_BULLISH_RESONANCE')
        structural_health_score = self._fuse_multi_level_scores(df, 'STRUCTURE_BULLISH_RESONANCE')
        mechanics_health_score = self._fuse_multi_level_scores(df, 'DYN_BULLISH_RESONANCE')
        # 原有的Hurst指数趋势分
        regime_health_score_hurst = self._get_atomic_score(df, 'SCORE_TRENDING_REGIME')
        # 新的FFT趋势分
        regime_health_score_fft = self._get_atomic_score(df, 'SCORE_TRENDING_REGIME_FFT')
        # 将两者融合，例如取平均值或加权平均，这里取平均
        regime_health_score = (regime_health_score_hurst + regime_health_score_fft) / 2.0
        # 消费独立的筹码支柱分，并进行加权
        p_chip_pillars = get_params_block(self.strategy, 'trend_quality_params', {}).get('chip_pillar_weights', {})
        chip_health_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        total_chip_weight = 0.0
        for pillar_name, weight in p_chip_pillars.items():
            pillar_score = self._get_atomic_score(df, f'SCORE_CHIP_PILLAR_{pillar_name.upper()}_HEALTH', 0.0)
            chip_health_score += pillar_score * weight
            total_chip_weight += weight
        if total_chip_weight > 0:
            chip_health_score /= total_chip_weight # 归一化
        # --- 2. 定义各维度权重 ---
        p = get_params_block(self.strategy, 'trend_quality_params', {})
        weights = p.get('domain_weights', {})
        # --- 3. 加权融合生成最终的趋势质量分 ---
        trend_quality_score = (
            behavior_health_score * weights.get('behavior', 0.20) +
            chip_health_score * weights.get('chip', 0.30) + # 增加筹码的权重
            fund_flow_health_score * weights.get('fund_flow', 0.15) +
            structural_health_score * weights.get('structural', 0.15) + 
            mechanics_health_score * weights.get('mechanics', 0.10) +
            regime_health_score * weights.get('regime', 0.10)
        )
        df['COGNITIVE_SCORE_TREND_QUALITY'] = trend_quality_score
        self.strategy.atomic_states['COGNITIVE_SCORE_TREND_QUALITY'] = df['COGNITIVE_SCORE_TREND_QUALITY']
        # print("        -> [趋势质量融合评分模块 V2.1] 计算完毕。")
        return df

    def diagnose_market_structure_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V280.0 终极结构层信号适配版】 - 联合作战司令部
        - 核心升级: [信号适配] 将对均线、力学等结构维度信号的消费，全面适配为消费由
                    StructuralIntelligence V2.0 生成的 'STRUCTURE_*' 终极信号。
        """
        # print("        -> [联合作战司令部 V280.0 终极结构层信号适配版] 启动，正在分析战场核心结构...")
        structure_states = {}
        default_series = pd.Series(False, index=df.index)
        atomic = self.strategy.atomic_states
        # --- 1. 提取核心终极信号 ---
        structural_bullish_score = self._fuse_multi_level_scores(df, 'STRUCTURE_BULLISH_RESONANCE') # 消费终极结构看涨信号
        structural_bearish_score = self._fuse_multi_level_scores(df, 'STRUCTURE_BEARISH_RESONANCE') # 消费终极结构看跌信号
        structural_top_reversal_score = self._fuse_multi_level_scores(df, 'STRUCTURE_TOP_REVERSAL') # 消费终极结构顶部反转信号
        chip_concentrating_score = self._fuse_multi_level_scores(df, 'RISING_RESONANCE')
        chip_diverging_score = self._fuse_multi_level_scores(df, 'FALLING_RESONANCE')
        behavioral_bearish_score = self._fuse_multi_level_scores(df, 'BEHAVIOR_BEARISH_RESONANCE')
        risk_late_stage_score_raw = self.strategy.atomic_states.get('CONTEXT_TREND_LATE_STAGE_SCORE', pd.Series(0, index=df.index))
        risk_late_stage_score = (risk_late_stage_score_raw / 600).clip(0, 1)
        prime_opportunity_score = self._get_atomic_score(df, 'CHIP_SCORE_PRIME_OPPORTUNITY_S', 0.0)
        # --- 2. 定义核心结构状态 ---
        # 主升浪: 结构看涨 & 筹码集中
        base_uptrend_score = (structural_bullish_score * chip_concentrating_score) # 使用 structural_bullish_score
        structure_states['SCORE_STRUCTURE_MAIN_UPTREND_WAVE_S'] = base_uptrend_score.astype(np.float32)
        structure_states['STRUCTURE_MAIN_UPTREND_WAVE_S'] = base_uptrend_score > 0.4
        # 堡垒主升浪: 主升浪 + 黄金筹码机会
        structure_states['SCORE_STRUCTURE_FORTRESS_UPTREND_S_PLUS'] = (base_uptrend_score * prime_opportunity_score).astype(np.float32)
        structure_states['STRUCTURE_FORTRESS_UPTREND_S_PLUS'] = structure_states['SCORE_STRUCTURE_FORTRESS_UPTREND_S_PLUS'] > 0.5
        # 突破前夜: 黄金筹码机会 & 波动率压缩 & 结构看涨
        fused_squeeze_score = self._fuse_multi_level_scores(df, 'VOL_COMPRESSION')
        prime_setup_score = prime_opportunity_score * fused_squeeze_score * structural_bullish_score # 使用 structural_bullish_score
        structure_states['SCORE_SETUP_PRIME_STRUCTURE_S'] = prime_setup_score.astype(np.float32)
        structure_states['SETUP_PRIME_STRUCTURE_S'] = prime_setup_score > 0.6
        # 早期反转: 行为反转 & 均线斜率转正 (保留原有逻辑，因为它依赖更底层的EMA斜率)
        is_recent_reversal = atomic.get('BEHAVIOR_CONTEXT_RECENT_REVERSAL_SIGNAL', default_series)
        is_ma_short_slope_positive = df.get('SLOPE_5_EMA_5_D', pd.Series(0, index=df.index)) > 0
        structure_states['STRUCTURE_EARLY_REVERSAL_B'] = is_recent_reversal & is_ma_short_slope_positive
        # 顶部危险: 结构看跌/反转 或 行为看跌 或 上涨末期
        topping_danger_score = np.maximum.reduce([
            structural_bearish_score.values, 
            structural_top_reversal_score.values, 
            behavioral_bearish_score.values,
            risk_late_stage_score.values
        ]) # 使用新的结构层终极信号
        structure_states['SCORE_STRUCTURE_TOPPING_DANGER_S'] = pd.Series(topping_danger_score, index=df.index, dtype=np.float32)
        # 下跌通道: 结构看跌 & 筹码发散
        bearish_channel_score = structural_bearish_score * chip_diverging_score # 使用 structural_bearish_score
        structure_states['SCORE_STRUCTURE_BEARISH_CHANNEL_F'] = bearish_channel_score.astype(np.float32)
        structure_states['STRUCTURE_BEARISH_CHANNEL_F'] = bearish_channel_score > 0.5
        # print("        -> [联合作战司令部 V280.0] 核心战局定义升级完成。")
        return structure_states

    def synthesize_topping_behaviors(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V332.0 风险归因修正版】顶部行为合成模块
        - 核心重构 (本次修改):
          - [风险归因] 将“拉升背离”的判断依据，从旧的、混杂的筹码信号，修正为纯粹消费
                        `BehavioralIntelligence` 产出的“顶部反转”与“看跌共振”融合分数。
        - 收益: 模块职责更清晰，风险归因更准确，信号逻辑与方法名完全匹配。
        """
        # print("        -> [顶部行为合成模块 V332.0 风险归因修正版] 启动...") 
        states = {}
        default_series = pd.Series(False, index=df.index)
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        atomic = self.strategy.atomic_states
        required_states = ['COGNITIVE_SCORE_RISK_HIGH_LEVEL_ZONE']
        if any(s not in atomic for s in required_states):
            print("          -> [警告] 缺少合成“顶部行为”所需情报，模块跳过。")
            return {}
        # --- 1. 定义顶部风险行为 ---
        # 将判断依据从筹码信号，修正为更纯粹、更可靠的行为信号
        # 融合“顶部反转”和“看跌共振”两大行为风险
        topping_reversal_risk = self._fuse_multi_level_scores(df, 'BEHAVIOR_TOP_REVERSAL')
        bearish_resonance_risk = self._fuse_multi_level_scores(df, 'BEHAVIOR_BEARISH_RESONANCE')
        behavioral_topping_risk = np.maximum(topping_reversal_risk, bearish_resonance_risk)
        is_rallying_score = self._get_atomic_score(df, 'SCORE_PRICE_POSITION_IN_RECENT_RANGE', 0.0)
        # 重命名信号，使其更准确地反映逻辑
        states['SCORE_ACTION_RISK_DECEPTIVE_RALLY'] = (is_rallying_score * behavioral_topping_risk).astype(np.float32)
        states['ACTION_RISK_DECEPTIVE_RALLY'] = states['SCORE_ACTION_RISK_DECEPTIVE_RALLY'] > 0.6
        stagnation_score = self._get_atomic_score(df, 'SCORE_RISK_VPA_STAGNATION', 0.0)
        states['SCORE_ACTION_RISK_RALLY_STAGNATION'] = stagnation_score.astype(np.float32)
        states['ACTION_RISK_RALLY_STAGNATION'] = stagnation_score > 0.7
        # --- 2. 定义S+级确认派发风险 ---
        danger_zone_score = atomic.get('COGNITIVE_SCORE_RISK_HIGH_LEVEL_ZONE', default_score)
        # 使用新的行为风险分进行融合
        distributing_action_score = states.get('SCORE_ACTION_RISK_DECEPTIVE_RALLY', default_score)
        states['SCORE_S_PLUS_CONFIRMED_DISTRIBUTION'] = (danger_zone_score * distributing_action_score).astype(np.float32)
        states['RISK_S_PLUS_CONFIRMED_DISTRIBUTION'] = states['SCORE_S_PLUS_CONFIRMED_DISTRIBUTION'] > 0.5
        # --- 3. 定义其他行为状态 ---
        is_rallying = df['pct_change_D'] > 0.02
        is_strategic_distribution = atomic.get('CONTEXT_CHIP_STRATEGIC_DISTRIBUTION', default_series)
        states['RISK_STRATEGIC_DISTRIBUTION_RALLY_TRAP_S'] = is_rallying & is_strategic_distribution
        if states['RISK_STRATEGIC_DISTRIBUTION_RALLY_TRAP_S'].any():
            print(f"          -> [S级战略风险] 侦测到 {states['RISK_STRATEGIC_DISTRIBUTION_RALLY_TRAP_S'].sum()} 次“战略派发背景下的诱多陷阱”！")
        concentrating_score = self._fuse_multi_level_scores(df, 'RISING_RESONANCE') # 适配新的终极上升共振信号
        is_concentrating = concentrating_score > 0.6
        is_in_danger_zone = atomic.get('CONTEXT_RISK_HIGH_LEVEL_ZONE', default_series)
        states['RALLY_STATE_HEALTHY_LOCKED'] = is_rallying & is_concentrating & ~is_in_danger_zone
        return states

    def synthesize_chip_fund_flow_synergy(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.0 深度博弈版】筹码与资金流协同合成模块
        - 核心升级 (本次修改):
          - [逻辑升维] 不再使用通用的共振信号，而是直接消费筹码层和资金流层产出的“深度博弈剧本”信号。
          - [新范式] 协同吸筹 = 真实筹码吸筹 * 资金隐蔽吸筹。
          - [新范式] 协同派发 = 虚假筹码集中 * 资金高位派发。
        - 收益: 对“协同”的定义从模糊的统计相关性，升级为精确的行为模式匹配，信号质量实现质的飞跃。
        """
        # print("        -> [筹码与资金流协同合成模块 V3.0 深度博弈版] 启动...") # 修改: 更新版本号
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)

        # --- 1. 交叉验证：生成“协同吸筹”分数 ---
        # 使用更高质量的“真实吸筹”和“隐蔽吸筹”剧本进行融合
        chip_true_accumulation_score = atomic.get('SCORE_CHIP_TRUE_ACCUMULATION', default_score)
        fund_flow_stealth_accumulation_score = atomic.get('SCORE_FF_PLAYBOOK_STEALTH_ACCUMULATION', default_score)
        synergy_accumulation_score = (chip_true_accumulation_score * fund_flow_stealth_accumulation_score).astype(np.float32)
        states['COGNITIVE_SCORE_CHIP_FUND_FLOW_ACCUMULATION_S'] = synergy_accumulation_score

        # --- 2. 交叉验证：生成“协同派发”风险分数 ---
        # 使用更高质量的“虚假集中风险”和“高位派发风险”剧本进行融合
        chip_false_accumulation_risk = atomic.get('SCORE_CHIP_FALSE_ACCUMULATION_RISK', default_score)
        fund_flow_top_distribution_risk = atomic.get('SCORE_FF_RISK_TOP_DISTRIBUTION', default_score)
        synergy_distribution_score = (chip_false_accumulation_risk * fund_flow_top_distribution_risk).astype(np.float32)
        states['COGNITIVE_SCORE_CHIP_FUND_FLOW_DISTRIBUTION_S'] = synergy_distribution_score
        
        # --- 3. 为了兼容性，保留旧的布尔信号 ---
        states['CHIP_FUND_FLOW_ACCUMULATION_CONFIRMED_A'] = synergy_accumulation_score > 0.6
        states['CHIP_FUND_FLOW_ACCUMULATION_STRONG_S'] = synergy_accumulation_score > 0.8
        states['RISK_CHIP_FUND_FLOW_DISTRIBUTION_A'] = synergy_distribution_score > 0.7
        # print("        -> [筹码与资金流协同合成模块 V3.0] 合成完毕。") # 修改: 更新版本号
        return states

    def synthesize_perfect_storm_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.1 数值化升级版】完美风暴信号合成模块
        - 核心职责: 寻找“行为层确认信号”与“结构层融合信号”同时发生的最高置信度信号。
        - 本次升级: 将原有的布尔逻辑升级为数值化评分，能够更精确地量化“完美风暴”的强度。
        - 新增信号:
          - COGNITIVE_SCORE_PERFECT_STORM_TOP_S_PLUS: 完美风暴顶部风险分 (S++级)。
          - COGNITIVE_SCORE_PERFECT_STORM_BOTTOM_S_PLUS: 完美风暴底部机会分 (S++级)。
        """
        # print("        -> [完美风暴信号合成模块 V1.1 数值化升级版] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        triggers = self.strategy.trigger_events
        default_series = pd.Series(False, index=df.index)
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. S++级情报融合：定义“完美风暴”顶部风险 (数值化) ---
        # 逻辑: S+级“确认派发”风险分 * 顶层“融合顶背离”结构分 = 完美风暴风险强度
        cognitive_fusion_top_reversal_score = atomic.get('COGNITIVE_FUSION_TOP_REVERSAL_S', default_score)
        confirmed_distribution_score = atomic.get('SCORE_S_PLUS_CONFIRMED_DISTRIBUTION', default_score) # 消费数值化分数
        perfect_storm_top_score = (confirmed_distribution_score * cognitive_fusion_top_reversal_score).astype(np.float32) # 分数相乘
        states['COGNITIVE_SCORE_PERFECT_STORM_TOP_S_PLUS'] = perfect_storm_top_score
        states['COGNITIVE_RISK_PERFECT_STORM_TOP_S_PLUS'] = perfect_storm_top_score > 0.5 # 基于分数生成兼容性布尔信号
        if states['COGNITIVE_RISK_PERFECT_STORM_TOP_S_PLUS'].any():
            print(f"          -> [S++级顶级风险] 侦测到 {states['COGNITIVE_RISK_PERFECT_STORM_TOP_S_PLUS'].sum()} 次“完美风暴”顶部风险信号！")
        # --- 2. S++级情报融合：定义“完美风暴”底部机会 (数值化) ---
        # 逻辑: “显性反转K线”强度分 * 顶层“融合底背离”结构分 = 完美风暴机会强度
        cognitive_fusion_bottom_reversal_score = atomic.get('COGNITIVE_FUSION_BOTTOM_REVERSAL_S', default_score)
        reversal_candle_score = triggers.get('TRIGGER_DOMINANT_REVERSAL', default_series).astype(float) # 将布尔触发器转为数值分 (0.0 或 1.0)
        perfect_storm_bottom_score = (reversal_candle_score * cognitive_fusion_bottom_reversal_score).astype(np.float32) # 分数相乘
        states['COGNITIVE_SCORE_PERFECT_STORM_BOTTOM_S_PLUS'] = perfect_storm_bottom_score
        states['COGNITIVE_OPP_PERFECT_STORM_BOTTOM_S_PLUS'] = perfect_storm_bottom_score > 0.5 # 基于分数生成兼容性布尔信号
        if states['COGNITIVE_OPP_PERFECT_STORM_BOTTOM_S_PLUS'].any():
            print(f"          -> [S++级顶级机会] 侦测到 {states['COGNITIVE_OPP_PERFECT_STORM_BOTTOM_S_PLUS'].sum()} 次“完美风暴”底部机会信号！")
        # --- 3. 更新原子状态库 ---
        self.strategy.atomic_states.update(states)
        # print("        -> [完美风暴信号合成模块 V1.1 数值化升级版] 计算完毕。") 
        return df

    def synthesize_dynamic_offense_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.1 信号源适配版】协同进攻动能合成模块
        - 核心升级: [信号适配] 将对筹码维度信号的消费，从旧的 'CHIP_BULLISH_RESONANCE'
                    适配为消费由 ChipIntelligence.diagnose_cross_validation_signals
                    生成的、更高质量的 'RISING_RESONANCE' 终极信号。
        """
        # print("        -> [协同进攻动能合成模块 V2.1 信号源适配版] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # 1. 获取五大领域的核心S级看涨分数
        mechanics_score = atomic.get('SCORE_DYN_BULLISH_RESONANCE_S', default_score)
        chip_score = self._fuse_multi_level_scores(df, 'RISING_RESONANCE', {'S_PLUS': 1.2, 'S': 1.0, 'A': 0.0, 'B': 0.0}) # 适配新的终极上升共振信号, 且仅使用S/S+级
        structure_score = self._fuse_multi_level_scores(df, 'MA_BULLISH_RESONANCE', {'S': 1.0, 'A': 0.0, 'B': 0.0}) # 仅使用S级
        fund_flow_score = atomic.get('FF_SCORE_SEPTAFECTA_RESONANCE_UP_HIGH', default_score)
        regime_score = atomic.get('SCORE_TRENDING_REGIME', default_score)
        # 2. 计算“协同进攻”元分数
        synergistic_offense_score = (
            mechanics_score * chip_score * structure_score * fund_flow_score * regime_score
        ).astype(np.float32)
        states['COGNITIVE_SCORE_SYNERGISTIC_OFFENSE_S'] = synergistic_offense_score
        self.strategy.atomic_states['COGNITIVE_SCORE_SYNERGISTIC_OFFENSE_S'] = synergistic_offense_score
        # 3. 获取战略过滤器：是否处于上涨末期
        late_stage_score = self.strategy.atomic_states.get('CONTEXT_TREND_LATE_STAGE_SCORE', pd.Series(0, index=df.index))
        p_trend_stage = get_params_block(self.strategy, 'trend_stage_params', {})
        max_score_for_offense = get_param_value(p_trend_stage.get('dynamic_offense_max_late_stage_score'), 30)
        is_in_safe_stage = late_stage_score < max_score_for_offense
        # 4. 最终裁定：发动协同进攻，且【处于安全阶段】
        # 基于新的元分数和阈值生成最终信号
        offense_threshold = get_param_value(p_trend_stage.get('synergistic_offense_threshold'), 0.3)
        is_synergistic_offense = synergistic_offense_score > offense_threshold
        final_signal = is_synergistic_offense & is_in_safe_stage
        states['DYN_AGGRESSIVE_OFFENSE_A'] = final_signal
        return states

    def _create_pullback_decision_log(self, df: pd.DataFrame, enhancements: Dict) -> pd.DataFrame:
        """
        【V2.1 信号修复与逻辑同步版】战术决策日志探针
        - 核心重构: 全面修复所有失效和过时的信号引用，并使其逻辑与
                      `_diagnose_pullback_tactics_matrix` V7.3 版本完全同步。
        - 核心修复 (本次修改): 修复了对 `COGNITIVE_OPP_ACCUMULATION_BREAKOUT_S` 的引用，
                        更新为消费 `STRUCTURE_MAIN_UPTREND_WAVE_S`。
        - 收益: 确保日志探针能100%准确地反映最终战术的决策过程。
        """
        log_data = {}
        atomic = self.strategy.atomic_states
        triggers = self.strategy.trigger_events
        default_series = pd.Series(False, index=df.index)
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 记录所有输入条件 (与主函数同步) ---
        # 1.1 战场环境
        lookback_window = 15
        ascent_start_event = atomic.get('STRUCTURE_MAIN_UPTREND_WAVE_S', default_series)
        cruise_start_event = atomic.get('TACTIC_LOCK_CHIP_RECONCENTRATION_S_PLUS', default_series)
        log_data['is_in_ascent_window'] = ascent_start_event.rolling(window=lookback_window, min_periods=1).max().astype(bool)
        log_data['is_in_cruise_window'] = cruise_start_event.rolling(window=lookback_window, min_periods=1).max().astype(bool)
        # 1.2 回踩性质 (昨日)
        p_pullback = get_params_block(self.strategy, 'pullback_tactics_params', {})
        healthy_threshold = get_param_value(p_pullback.get('healthy_pullback_score_threshold'), 0.3)
        suppressive_threshold = get_param_value(p_pullback.get('suppressive_pullback_score_threshold'), 0.3)
        log_data['score_healthy_pullback_yesterday'] = atomic.get('COGNITIVE_SCORE_PULLBACK_HEALTHY_S', default_score).shift(1).fillna(0.0)
        log_data['was_healthy_pullback'] = log_data['score_healthy_pullback_yesterday'] > healthy_threshold
        log_data['score_suppressive_pullback_yesterday'] = atomic.get('COGNITIVE_SCORE_PULLBACK_SUPPRESSIVE_S', default_score).shift(1).fillna(0.0)
        log_data['was_suppressive_pullback'] = log_data['score_suppressive_pullback_yesterday'] > suppressive_threshold
        # 1.3 确认信号 (今日)
        log_data['is_reversal_confirmed_today'] = triggers.get('TRIGGER_DOMINANT_REVERSAL', default_series)
        # 1.4 安全过滤器
        late_stage_score = self.strategy.atomic_states.get('CONTEXT_TREND_LATE_STAGE_SCORE', pd.Series(0, index=df.index))
        p_trend_stage = get_params_block(self.strategy, 'trend_stage_params', {})
        max_score_for_pullback = get_param_value(p_trend_stage.get('pullback_s_plus_max_late_stage_score'), 50)
        log_data['is_in_safe_stage'] = late_stage_score < max_score_for_pullback
        # --- 2. 记录所有潜在战法 (不考虑互斥) ---
        log_data['POTENTIAL_S+++'] = log_data['is_in_cruise_window'] & log_data['was_suppressive_pullback'] & log_data['is_reversal_confirmed_today']
        log_data['POTENTIAL_S+'] = log_data['is_in_cruise_window'] & log_data['was_healthy_pullback'] & log_data['is_reversal_confirmed_today'] & log_data['is_in_safe_stage']
        log_data['POTENTIAL_A+'] = log_data['is_in_ascent_window'] & log_data['was_suppressive_pullback'] & log_data['is_reversal_confirmed_today']
        log_data['POTENTIAL_A'] = log_data['is_in_ascent_window'] & log_data['was_healthy_pullback'] & log_data['is_reversal_confirmed_today']
        # --- 3. 记录最终的互斥决策结果 ---
        final_s_triple_plus = log_data['POTENTIAL_S+++']
        final_s_plus = log_data['POTENTIAL_S+'] & ~final_s_triple_plus
        is_cruise_decision = final_s_triple_plus | final_s_plus
        final_a_plus = log_data['POTENTIAL_A+'] & ~is_cruise_decision
        final_a = log_data['POTENTIAL_A'] & ~is_cruise_decision & ~final_a_plus
        log_data['FINAL_S+++'] = final_s_triple_plus
        log_data['FINAL_S+'] = final_s_plus
        log_data['FINAL_A+'] = final_a_plus
        log_data['FINAL_A'] = final_a
        return pd.DataFrame(log_data)

    def synthesize_industry_synergy_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】行业-个股协同元融合引擎
        - 核心职责: 将行业生命周期状态与个股的关键认知信号进行交叉验证，生成更高维度的战术信号。
        - 核心逻辑:
          - 协同进攻 = 行业景气周期 (预热/主升) * 个股进攻信号 (点火/突破)
          - 协同风险 = 行业衰退周期 (滞涨/下跌) * 个股风险信号 (崩溃/派发)
        - 收益: 创造出具备“戴维斯双击”效应的超高质量信号，极大提升策略的胜率和赔率。
        """
        # print("        -> [行业-个股协同元融合引擎 V1.0] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 获取上游的行业与个股核心信号 ---
        # 行业景气信号 (加权融合)
        score_markup = atomic.get('SCORE_INDUSTRY_MARKUP', default_score)
        score_preheat = atomic.get('SCORE_INDUSTRY_PREHEAT', default_score)
        industry_bullish_score = np.maximum(score_markup, score_preheat) # 取两者中的最大值作为行业看涨强度
        # 行业衰退信号 (加权融合)
        score_stagnation = atomic.get('SCORE_INDUSTRY_STAGNATION', default_score)
        score_downtrend = atomic.get('SCORE_INDUSTRY_DOWNTREND', default_score)
        industry_bearish_score = np.maximum(score_stagnation, score_downtrend) # 取两者中的最大值作为行业看跌强度
        # 个股进攻信号
        stock_ignition_score = atomic.get('COGNITIVE_SCORE_IGNITION_RESONANCE_S', default_score)
        stock_breakout_score = self._fuse_multi_level_scores(df, 'VOL_BREAKOUT')
        stock_bullish_score = np.maximum(stock_ignition_score, stock_breakout_score) # 取两者最大值作为个股进攻强度
        # 个股风险信号
        stock_breakdown_score = atomic.get('COGNITIVE_SCORE_BREAKDOWN_RESONANCE_S', default_score)
        stock_distribution_score = atomic.get('COGNITIVE_SCORE_RISK_TOP_DISTRIBUTION', default_score)
        stock_bearish_score = np.maximum(stock_breakdown_score, stock_distribution_score) # 取两者最大值作为个股风险强度
        # --- 2. 生成“协同进攻”元融合信号 ---
        # 逻辑: 行业景气 * 个股进攻 = 协同进攻强度
        synergy_offense_score = pd.Series(industry_bullish_score, index=df.index) * pd.Series(stock_bullish_score, index=df.index)
        states['COGNITIVE_SCORE_INDUSTRY_SYNERGY_OFFENSE_S'] = synergy_offense_score.astype(np.float32)
        # --- 3. 生成“协同风险”元融合信号 ---
        # 逻辑: 行业衰退 * 个股风险 = 协同风险强度
        synergy_risk_score = pd.Series(industry_bearish_score, index=df.index) * pd.Series(stock_bearish_score, index=df.index)
        states['COGNITIVE_SCORE_INDUSTRY_SYNERGY_RISK_S'] = synergy_risk_score.astype(np.float32)
        print(f"        -> [行业-个股协同元融合引擎 V1.0] 完成，生成了2个S级协同信号。")
        return states

    def synthesize_cognitive_scores(self, df: pd.DataFrame, pullback_enhancements: Dict) -> pd.DataFrame:
        """
        【V2.2 架构修复版】顶层认知总分合成模块
        - 核心重构 (本次修改):
          - [架构调整] 本方法现在作为总指挥，按逻辑顺序调用新拆分出的 `MicroBehaviorEngine` 和 `TacticEngine`。
          - [职责净化] 移除了对具体微观行为和战术的直接计算，这些逻辑已下沉到新的子引擎中。
          - [流程优化] 确保子引擎生成的信号能被后续的认知融合模块正确消费。
          - [战术扩充] 新增对“均值回归”策略信号的合成调用。
          - [依赖注入] 接收 `pullback_enhancements` 并传递给战术引擎，修复数据流。
          - [代码内聚] 将回踩战术的调试日志逻辑移入此方法。
        """
        # 代码修改：更新方法签名和版本号
        print("        -> [顶层认知总分合成模块 V2.2 架构修复版] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)

        # --- 步骤 1: 调用微观行为引擎，生成深层行为模式信号 ---
        micro_behavior_states = self.micro_behavior_engine.run_micro_behavior_synthesis(df)
        states.update(micro_behavior_states)
        self.strategy.atomic_states.update(micro_behavior_states) # 关键：立即更新原子状态库，供后续模块使用

        # --- 步骤 2: 调用战术引擎，生成具体战术信号 ---
        # 代码修改：将 pullback_enhancements 传递给战术引擎
        tactic_states = self.tactic_engine.run_tactic_synthesis(df, pullback_enhancements)
        states.update(tactic_states)
        self.strategy.atomic_states.update(tactic_states) # 关键：再次更新原子状态库
        # 代码新增：将战术引擎产出的剧本状态也更新到策略实例中
        self.strategy.playbook_states.update({k: v for k, v in tactic_states.items() if k.startswith('PLAYBOOK_')})

        # --- 步骤 3: 执行本模块剩余的核心认知融合任务 ---
        # 注意：这些方法现在可以消费由子引擎生成的、更丰富的信号
        industry_synergy_states = self.synthesize_industry_synergy_signals(df)
        states.update(industry_synergy_states)
        self.strategy.atomic_states.update(industry_synergy_states)
        # 代码新增：调用新增的均值回归信号合成模块
        mean_reversion_states = self.synthesize_mean_reversion_signals(df)
        states.update(mean_reversion_states)
        self.strategy.atomic_states.update(mean_reversion_states)
        # --- 步骤 4: 汇总所有S级的“机会”类认知分数 ---
        bullish_scores = [
            atomic.get('COGNITIVE_SCORE_IGNITION_RESONANCE_S', default_score).values,
            atomic.get('COGNITIVE_SCORE_BOTTOM_REVERSAL_RESONANCE_S', default_score).values,
            atomic.get('COGNITIVE_SCORE_CLASSIC_PATTERN_OPP_S', default_score).values,
            atomic.get('COGNITIVE_SCORE_BEARISH_EXHAUSTION_OPP_S', default_score).values,
            atomic.get('COGNITIVE_SCORE_CONSOLIDATION_BREAKOUT_OPP_A', default_score).values,
            atomic.get('COGNITIVE_SCORE_PERFECT_STORM_BOTTOM_S_PLUS', default_score).values,
            states.get('COGNITIVE_SCORE_INDUSTRY_SYNERGY_OFFENSE_S', default_score).values,
            states.get('COGNITIVE_SCORE_OPP_POWER_SHIFT_TO_MAIN_FORCE', default_score).values,
            states.get('COGNITIVE_SCORE_OPP_MAIN_FORCE_CONVICTION_STRENGTHENING', default_score).values,
            states.get('COGNITIVE_SCORE_OPP_POST_REVERSAL_RESONANCE_A_PLUS', default_score).values,
        ]
        cognitive_bullish_score = np.maximum.reduce(bullish_scores)
        states['COGNITIVE_BULLISH_SCORE'] = pd.Series(cognitive_bullish_score, index=df.index, dtype=np.float32)

        # --- 步骤 5: 调用风险元融合模块，并汇总所有“风险”类认知分数 ---
        fused_risk_states = self.synthesize_fused_risk_scores(df)
        states.update(fused_risk_states)
        cognitive_bearish_score_series = states.get('COGNITIVE_FUSED_RISK_SCORE', default_score)
        industry_synergy_risk_score = states.get('COGNITIVE_SCORE_INDUSTRY_SYNERGY_RISK_S', default_score)
        microstructure_conviction_risk_score = states.get('COGNITIVE_SCORE_RISK_MAIN_FORCE_CONVICTION_WEAKENING', default_score)
        microstructure_power_shift_risk_score = states.get('COGNITIVE_SCORE_RISK_POWER_SHIFT_TO_RETAIL', default_score)
        euphoric_risk_score = states.get('COGNITIVE_SCORE_RISK_EUPHORIC_ACCELERATION', default_score)
        
        final_bearish_score = np.maximum.reduce([
            cognitive_bearish_score_series.values,
            industry_synergy_risk_score.values,
            microstructure_conviction_risk_score.values,
            microstructure_power_shift_risk_score.values,
            euphoric_risk_score.values
        ])
        states['COGNITIVE_BEARISH_SCORE'] = pd.Series(final_bearish_score, index=df.index, dtype=np.float32)

        # --- 步骤 6: 移动调试日志逻辑到此 ---
        # 代码新增：将调试日志逻辑从 IntelligenceLayer 移入，保持内聚
        debug_params = get_params_block(self.strategy, 'debug_params')
        if get_param_value(debug_params.get('enable_pullback_decision_log'), False):
            decision_log_df = self._create_pullback_decision_log(df, pullback_enhancements)
            final_tactic_days = decision_log_df.filter(like='FINAL_').any(axis=1)
            if final_tactic_days.any():
                print("\n--- [回踩战术决策日志探针] ---")
                display_cols = [col for col in decision_log_df.columns if 'POTENTIAL_' in col or 'FINAL_' in col]
                print("决策日志 (POTENTIAL: 潜在机会, FINAL: 最终决策):")
                print(decision_log_df.loc[final_tactic_days, display_cols])
                print("--- [探针结束] ---\n")

        # --- 步骤 7: 更新原子状态库并返回 ---
        self.strategy.atomic_states.update(states)
        print("        -> [顶层认知总分合成模块 V2.2] 模块化重构完成。") # 代码修改：更新版本号
        return df

    def synthesize_mean_reversion_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】均值回归网格交易策略信号合成模块
        - 核心目标: 在识别出的震荡市中，捕捉价格触及统计下轨的买入机会。
        - 核心逻辑: 机会分 = 震荡市环境 * 价格触及布林带下轨程度
        - 产出信号: SCORE_PLAYBOOK_MEAN_REVERSION_GRID_BUY_A - 一个A级的网格交易买入机会分。
        """
        print("        -> [均值回归信号合成模块 V1.0] 启动...")
        states = {}
        p = get_params_block(self.strategy, 'mean_reversion_grid_params', {})
        if not get_param_value(p.get('enabled'), True):
            return states
        # 1. 定义震荡市环境 (Context)
        cyclical_regime_threshold = get_param_value(p.get('cyclical_regime_threshold'), 0.4)
        trending_regime_threshold = get_param_value(p.get('trending_regime_threshold'), 0.45)
        
        is_cyclical_regime = self._get_atomic_score(df, 'SCORE_CYCLICAL_REGIME') > cyclical_regime_threshold
        is_not_trending_regime = self._get_atomic_score(df, 'SCORE_TRENDING_REGIME_FFT') < trending_regime_threshold
        context_is_ranging_market = (is_cyclical_regime & is_not_trending_regime).astype(float)
        states['CONTEXT_RANGING_MARKET'] = context_is_ranging_market.astype(np.float32)
        # 2. 定义买入机会 (Buy Opportunity)
        # BBP (布林带百分比) 是一个很好的指标。BBP < 0 意味着收盘价低于下轨。
        bbp = df.get('BBP_21_2.0_D', pd.Series(0.5, index=df.index))
        # 当价格低(BBP低)时，分数高。(1 - bbp) 将 [0, 1] 映射到 [1, 0]。我们将其裁剪以处理带外的价格。
        buy_opportunity_score = (1 - bbp.clip(0, 1)).astype(np.float32)
        states['SCORE_OPP_MEAN_REVERSION_BUY'] = buy_opportunity_score
        # 3. 融合生成最终剧本分数
        final_playbook_score = context_is_ranging_market * buy_opportunity_score
        states['SCORE_PLAYBOOK_MEAN_REVERSION_GRID_BUY_A'] = final_playbook_score.astype(np.float32)
        print(f"          - [均值回归] 完成, 识别到 {(final_playbook_score > 0.5).sum()} 个潜在网格买点。")
        return states






