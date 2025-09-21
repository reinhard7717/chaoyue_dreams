# 文件: strategies/trend_following/intelligence/cognitive_intelligence.py
# 顶层认知合成模块
import pandas as pd
import numpy as np
from typing import Dict
from enum import Enum
from strategies.trend_following.utils import get_params_block, get_param_value

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

    def synthesize_early_momentum_ignition(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】早期动能点火诊断模块 (东风初起)
        - 核心目标: 识别“万事俱备”之后，动能“刚刚启动”的精确时点，避免追高。
        - 核心逻辑: 融合多个“早期”和“温和”的动能信号，如波动率拐点、MACD低位金叉、
                      价格温和放量等，形成一个综合的“早期点火分”。
        - 产出信号:
          - `COGNITIVE_SCORE_EARLY_MOMENTUM_IGNITION_A`: A级早期动能点火信号，可用于替代或补充
                                                         现有激进的动能信号。
        """
        print("        -> [早期动能点火诊断模块 V1.0] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 提取多个“早期”或“温和”的动能信号 ---
        # 信号1: 波动率拐点 (从压缩到扩张的转折点)
        vol_tipping_point_score = atomic.get('SCORE_VOL_TIPPING_POINT_BOTTOM_OPP', default_score)
        # 信号2: MACD低位金叉 (趋势反转的早期信号)
        # 使用B级和A级信号，避免在高位金叉时介入
        macd_reversal_score = np.maximum(
            atomic.get('SCORE_MACD_BOTTOM_REVERSAL_B', default_score).values,
            atomic.get('SCORE_MACD_BOTTOM_REVERSAL_A', default_score).values
        )
        macd_reversal_series = pd.Series(macd_reversal_score, index=df.index)
        # 信号3: 价格温和上涨 (涨幅在1%到4%之间，避免涨停)
        pct_change = df['pct_change_D']
        # 使用三角形函数对涨幅进行评分，在2.5%附近得分最高，在0%和5%附近为0
        gentle_rally_score = np.maximum(0, 1 - np.abs(pct_change - 0.025) / 0.025).fillna(0)
        # 信号4: 成交量温和放大 (相对于21日均量放大1.2到2.5倍)
        volume_ratio = df['volume_D'] / df.get('VOL_MA_21_D', df['volume_D']).replace(0, np.nan)
        # 使用梯形函数评分
        vol_score1 = (volume_ratio - 1.2) / (1.8 - 1.2) # 从1.2到1.8线性增长
        vol_score2 = (3.0 - volume_ratio) / (3.0 - 1.8) # 从1.8到3.0线性下降
        gentle_volume_score = np.minimum(vol_score1, vol_score2).clip(0, 1).fillna(0)
        # 信号5: 价格正向加速度 (确认开始启动)
        price_accel_score = self._normalize_score(df['ACCEL_1_close_D'].clip(lower=0), default=0.0)
        # --- 2. 融合生成“早期动能点火”分数 ---
        # 采用几何平均值融合，要求多个信号同时存在
        # 将Series转换为NumPy数组进行计算
        score_components = [
            vol_tipping_point_score.values,
            macd_reversal_series.values,
            gentle_rally_score.values,
            gentle_volume_score.values,
            price_accel_score.values
        ]
        # 为避免0值导致整个结果为0，给每个分量加上一个极小值
        epsilon = 1e-9
        prod_scores = np.prod([arr + epsilon for arr in score_components], axis=0)
        final_score_arr = prod_scores**(1.0 / len(score_components))
        final_score = pd.Series(final_score_arr, index=df.index, dtype=np.float32)
        states['COGNITIVE_SCORE_EARLY_MOMENTUM_IGNITION_A'] = final_score
        print(f"        -> [早期动能点火诊断模块 V1.0] 计算完毕。")
        return states

    def synthesize_chip_price_lag_playbook(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】“筹码共振-价格滞后”战术剧本
        - 核心目标: 捕捉“万事俱备，只欠东风”的黄金买点。
        - 战备状态 (Setup): 筹码高度共振 + 价格动能被压制 + 波动率压缩。
        - 点火触发 (Trigger): 价格出现温和的启动迹象。
        - 剧本逻辑: 昨日战备就绪，今日点火触发。
        """
        print("        -> [筹码共振-价格滞后剧本 V1.0] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 定义“战备状态”评分 (Setup Score) ---
        # 战备条件1: 筹码高度共振 (使用S级或S+级信号)
        chip_resonance_score = self._fuse_multi_level_scores(df, 'CHIP_BULLISH_RESONANCE', {'S_PLUS': 1.2, 'S': 1.0})
        # 战备条件2: 价格动能被压制 (5日价格斜率处于历史低位)
        price_momentum_suppressed_score = self._normalize_score(df['SLOPE_5_close_D'], ascending=False)
        # 战备条件3: 波动率压缩 (使用S级压缩信号)
        volatility_compression_score = atomic.get('SCORE_VOL_COMPRESSION_S', default_score)
        # 融合生成战备分
        setup_score = (chip_resonance_score * price_momentum_suppressed_score * volatility_compression_score).astype(np.float32)
        states['SCORE_SETUP_CHIP_RESONANCE_READY_S'] = setup_score
        states['SETUP_CHIP_RESONANCE_READY_S'] = setup_score > 0.6 # 布尔信号用于逻辑判断
        # --- 2. 定义“点火触发”评分 (Trigger Score) ---
        # 直接复用“早期动能点火”的逻辑，因为它完美符合“温和启动”的定义
        trigger_score = atomic.get('COGNITIVE_SCORE_EARLY_MOMENTUM_IGNITION_A', default_score)
        states['SCORE_TRIGGER_GENTLE_PRICE_LIFT_A'] = trigger_score
        states['TRIGGER_GENTLE_PRICE_LIFT_A'] = trigger_score > 0.4 # 布尔信号用于逻辑判断
            
        print("        -> [筹码共振-价格滞后剧本 V1.0] 计算完毕。")
        return states

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

# 文件: strategies/trend_following/intelligence/cognitive_intelligence.py

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
        gentle_drop_score = (1 - (df['pct_change_D'].abs() / 0.05)).clip(0, 1)
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
        significant_drop_score = (df['pct_change_D'].abs() / 0.07).clip(0, 1)
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

    def _diagnose_lock_chip_reconcentration_tactic(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.5 信号源修复版】锁仓再集中S+战法诊断模块
        - 核心重构: (V2.0) 将战法的“准备状态”从有缺陷的A级信号，升级为经过战场环境过滤的
                      S级“筹码结构黄金机会”信号。
        - 本次升级: 【数值化】将原有的布尔逻辑升级为“战备分 * 点火分”的数值化评分体系。
        - 核心修复 (V2.5): 修复了对 `SCORE_DYN_OVERALL_BULLISH_MOMENTUM_S` 这个不存在信号的引用，
                        替换为消费由 `DynamicMechanicsEngine` 生成的、逻辑最相近的
                        `SCORE_DYN_BULLISH_RESONANCE_S` 终极信号。
        """
        # print("        -> [S+战法诊断] 正在扫描“锁仓再集中(V2.3 王牌重铸数值化增强版)”...") 
        states = {}
        atomic = self.strategy.atomic_states
        triggers = self.strategy.trigger_events
        default_series = pd.Series(False, index=df.index)
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 定义“准备状态”评分 (Setup Score) ---
        setup_score = atomic.get('CHIP_SCORE_PRIME_OPPORTUNITY_S', default_score)
        # --- 2. 定义“点火事件”评分 (Ignition Score) ---
        trigger_chip_ignition_score = triggers.get('TRIGGER_CHIP_IGNITION', default_series).astype(float)
        energy_release_score = atomic.get('SCORE_DYN_BULLISH_RESONANCE_S', default_score) 
        cost_accel_score = atomic.get('SCORE_PLATFORM_COST_ACCEL', default_score)
        squeeze_breakout_score = atomic.get('COGNITIVE_SCORE_VOL_BREAKOUT_S', default_score) 
        ignition_trigger_score_arr = np.maximum.reduce([
            trigger_chip_ignition_score.values,
            energy_release_score.values, 
            cost_accel_score.values,
            squeeze_breakout_score.values 
        ])
        ignition_trigger_score = pd.Series(ignition_trigger_score_arr, index=df.index)
        # --- 3. 最终裁定：昨日“准备就绪”(分) * 今日“点火”(分) = 最终战法分 ---
        was_setup_yesterday_score = setup_score.shift(1).fillna(0.0)
        final_tactic_score = (was_setup_yesterday_score * ignition_trigger_score).astype(np.float32)
        states['COGNITIVE_SCORE_LOCK_CHIP_RECONCENTRATION_S_PLUS'] = final_tactic_score
        final_tactic_signal = final_tactic_score > 0.5
        states['TACTIC_LOCK_CHIP_RECONCENTRATION_S_PLUS'] = final_tactic_signal
        if final_tactic_signal.any():
            print(f"          -> [S+级战法确认] 侦测到 {final_tactic_signal.sum()} 次“锁仓再集中”的最终拉升信号！")
        return states

    def _diagnose_lock_chip_rally_tactic(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.4 信号源适配版】锁筹拉升S级战法诊断模块
        - 核心升级: [信号适配] 将对筹码维度信号的消费，适配为消费由 ChipIntelligence.diagnose_cross_validation_signals
                    生成的、更高质量的 'RISING_RESONANCE' 和 'FALLING_RESONANCE' 终极信号。
        """
        # print("        -> [S级战法诊断] 正在扫描“锁筹拉升(V2.4 信号源适配版)”...")
        states = {}
        atomic = self.strategy.atomic_states
        default_series = pd.Series(False, index=df.index)
        # --- 1. 获取参数  ---
        p = get_params_block(self.strategy, 'lock_chip_rally_params', {})
        require_concentration = get_param_value(p.get('require_continuous_concentration'), True)
        terminate_on_stalling = get_param_value(p.get('terminate_on_health_stalling'), True)
        # 为数值化分数定义阈值
        divergence_threshold = get_param_value(p.get('divergence_threshold'), 0.7)
        concentration_threshold = get_param_value(p.get('concentration_threshold'), 0.6)
        # --- 2. 定义“点火”事件  ---
        ignition_event = atomic.get('TACTIC_LOCK_CHIP_RECONCENTRATION_S_PLUS', default_series)
        # --- 3. 定义“硬性熄火”条件  ---
        # 使用融合后的数值分代替废弃的 CHIP_DYN_OBJECTIVE_DIVERGING
        is_diverging = self._fuse_multi_level_scores(df, 'FALLING_RESONANCE') > divergence_threshold # 适配新的终极下跌共振信号
        is_late_stage = atomic.get('CONTEXT_TREND_STAGE_LATE', default_series)
        is_ma_broken = self._get_atomic_score(df, 'SCORE_MA_HEALTH', 1.0) < 0.4 # 修复失效的 MA_STATE_STABLE_BULLISH 信号，升级为基于均线健康分的判断
        is_health_stalling = atomic.get('COGNITIVE_HOLD_RISK_HEALTH_STALLING', default_series)
        hard_termination_condition = is_diverging | is_late_stage | is_ma_broken
        if terminate_on_stalling:
            hard_termination_condition |= is_health_stalling
        # --- 4. 定义“软性巡航”条件  ---
        # 使用融合后的数值分代替废弃的 CHIP_DYN_CONCENTRATING
        is_cruise_condition_met = self._fuse_multi_level_scores(df, 'RISING_RESONANCE') > concentration_threshold if require_concentration else pd.Series(True, index=df.index) # 适配新的终极上升共振信号
        # --- 5. 构建带“容错机制”的状态机  ---
        n = len(df)
        # 步骤5.1: 将所有需要在循环中访问的Pandas Series一次性转换为NumPy数组
        hard_term_arr = hard_termination_condition.to_numpy(dtype=bool)
        cruise_cond_arr = is_cruise_condition_met.to_numpy(dtype=bool)
        ignition_arr = ignition_event.to_numpy(dtype=bool)
        # 步骤5.2: 初始化一个NumPy数组来存储状态结果
        rally_state_arr = np.full(n, False, dtype=bool)
        # 步骤5.3: 在高性能的NumPy循环中执行状态机逻辑
        cruise_warning_active = False # 引入“健康预警”状态标志
        for i in range(1, n):
            # 检查硬性熄火条件，这是最高优先级
            if hard_term_arr[i]:
                rally_state_arr[i] = False
                cruise_warning_active = False
                continue
            # 如果昨天处于巡航状态
            if rally_state_arr[i-1]:
                # 检查软性巡航条件
                if cruise_cond_arr[i]:
                    # 条件满足，继续健康巡航，并解除预警
                    rally_state_arr[i] = True
                    cruise_warning_active = False
                else:
                    # 条件不满足，检查是否已在预警状态
                    if cruise_warning_active:
                        # 已经预警过一次，这是连续第二次失败，终止巡航
                        rally_state_arr[i] = False
                        cruise_warning_active = False
                    else:
                        # 这是第一次失败，进入预警状态，但巡航继续
                        rally_state_arr[i] = True
                        cruise_warning_active = True
            # 如果今天有点火信号，则开启巡航
            elif ignition_arr[i]:
                rally_state_arr[i] = True
                cruise_warning_active = False # 新的巡航开始时，总是健康的
        # 步骤5.4: 将计算结果转换回Pandas Series
        is_in_rally_state = pd.Series(rally_state_arr, index=df.index)
        final_tactic_signal = is_in_rally_state & ~hard_termination_condition
        states['TACTIC_LOCK_CHIP_RALLY_S'] = final_tactic_signal
        # if final_tactic_signal.any():
            # print(f"          -> [S级持仓确认] 侦测到 {final_tactic_signal.sum()} 天处于“健康锁筹拉升”巡航状态！")
        return states

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

    def synthesize_prime_tactic(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.4 融合函数升级版】终极战法合成模块
        - 核心职责: (原有注释)
        - 本次升级 (V2.4): 
          - [逻辑深化] 使用新增的 `_fuse_multi_level_scores` 辅助函数来融合S/A/B三级
                        波动率压缩信号，使得对“极致压缩”的判断更平滑、更鲁棒。
        - 收益: 战法对市场状态的感知更精确，避免了因S级信号的微小波动而错失机会。
        """
        # print("        -> [终极战法合成模块 V2.4 融合函数升级版] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_series = pd.Series(False, index=df.index)
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 定义S级“黄金阵地” (Prime Setup) ---
        is_prime_chip_structure = self._get_atomic_score(df, 'CHIP_SCORE_PRIME_OPPORTUNITY_S', 0.0) > 0.7
        # 使用新的融合函数来生成“极致压缩”的判断条件
        fused_compression_score = self._fuse_multi_level_scores(df, 'VOL_COMPRESSION')
        is_extreme_squeeze = fused_compression_score > 0.9
        has_energy_advantage = self._fuse_multi_level_scores(df, 'MECHANICS_BULLISH_RESONANCE') > 0.7
        condition_sum = (
            is_prime_chip_structure.astype(int) +
            is_extreme_squeeze.astype(int) +
            has_energy_advantage.astype(int)
        )
        setup_s_plus_plus = (condition_sum == 3)
        states['SETUP_PRIME_STRUCTURE_S_PLUS_PLUS'] = setup_s_plus_plus
        setup_s_plus = (condition_sum == 2)
        states['SETUP_PRIME_STRUCTURE_S_PLUS'] = setup_s_plus
        # --- 2. 获取S级“突破冲锋号” ---
        ignition_resonance_score = atomic.get('COGNITIVE_SCORE_IGNITION_RESONANCE_S', default_score)
        trigger_prime_breakout_s = ignition_resonance_score > 0.6
        # --- 3. 定义战略环境过滤器 ---
        is_in_early_stage_today = atomic.get('CONTEXT_TREND_STAGE_EARLY', default_series)
        # --- 4. 【终极裁定】生成王牌战法 (已注入战略智慧) ---
        is_triggered_today = trigger_prime_breakout_s
        was_setup_s_plus_plus_yesterday = setup_s_plus_plus.shift(1).fillna(False)
        final_tactic_s_plus_plus = was_setup_s_plus_plus_yesterday & is_triggered_today & is_in_early_stage_today
        states['TACTIC_PRIME_STRUCTURE_BREAKOUT_S_PLUS_PLUS'] = final_tactic_s_plus_plus
        was_setup_s_plus_yesterday = setup_s_plus.shift(1).fillna(False)
        final_tactic_s_plus = was_setup_s_plus_yesterday & is_triggered_today & is_in_early_stage_today & ~final_tactic_s_plus_plus
        states['TACTIC_PRIME_STRUCTURE_BREAKOUT_S_PLUS'] = final_tactic_s_plus
        # if final_tactic_s_plus_plus.any():
        #     print(f"          -> [S++级王牌战法] 侦测到 {final_tactic_s_plus_plus.sum()} 次“终极结构突破”机会！")
        # if final_tactic_s_plus.any():
        #     print(f"          -> [S+级王牌战法] 侦测到 {final_tactic_s_plus.sum()} 次“次级结构突破”机会！")
        return states

    def _diagnose_pullback_tactics_matrix(self, df: pd.DataFrame, enhancements: Dict) -> Dict[str, pd.Series]:
        """
        【V7.4 信号源修复版】回踩战术诊断模块
        - 核心升级: 为 S+ 级“巡航回踩确认”战法增加了“非上涨末期”的前置条件。
        - 本次升级: [信号修复] 修复了对“蓄势突破”信号的引用。原信号 `STRUCTURAL_OPP_ACCUMULATION_BREAKOUT_S` 已失效，
                      现已更新为消费逻辑最相近的 `STRUCTURE_MAIN_UPTREND_WAVE_S` 信号，以恢复对“初升浪”阶段的正确判断。
        """
        # print("        -> [回踩战术矩阵 V7.3 信号源更新版] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        triggers = self.strategy.trigger_events
        default_series = pd.Series(False, index=df.index)
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 提取核心情报  ---
        # 战场环境
        lookback_window = 15
        ascent_start_event = atomic.get('STRUCTURE_MAIN_UPTREND_WAVE_S', default_series)
        cruise_start_event = atomic.get('TACTIC_LOCK_CHIP_RECONCENTRATION_S_PLUS', default_series)
        is_in_ascent_window = ascent_start_event.rolling(window=lookback_window, min_periods=1).max().astype(bool)
        is_in_cruise_window = cruise_start_event.rolling(window=lookback_window, min_periods=1).max().astype(bool)
        # 获取回踩分数阈值
        p_pullback = get_params_block(self.strategy, 'pullback_tactics_params', {})
        healthy_threshold = get_param_value(p_pullback.get('healthy_pullback_score_threshold'), 0.3)
        suppressive_threshold = get_param_value(p_pullback.get('suppressive_pullback_score_threshold'), 0.3)
        # 回踩性质 (昨日) - 使用数值分和阈值判断
        was_healthy_pullback = (atomic.get('COGNITIVE_SCORE_PULLBACK_HEALTHY_S', default_score).shift(1).fillna(0.0) > healthy_threshold)
        was_suppressive_pullback = (atomic.get('COGNITIVE_SCORE_PULLBACK_SUPPRESSIVE_S', default_score).shift(1).fillna(0.0) > suppressive_threshold)
        # 统一确认信号 (今日)
        is_reversal_confirmed = triggers.get('TRIGGER_DOMINANT_REVERSAL', default_series)
        #：获取“上涨末期”上下文状态
        late_stage_score = self.strategy.atomic_states.get('CONTEXT_TREND_LATE_STAGE_SCORE', pd.Series(0, index=df.index))
        # 从配置中读取此战法能容忍的最高风险分数
        p_trend_stage = get_params_block(self.strategy, 'trend_stage_params', {})
        max_score_for_pullback = get_param_value(p_trend_stage.get('pullback_s_plus_max_late_stage_score'), 50)
        # 定义是否处于安全区域
        is_in_safe_stage = late_stage_score < max_score_for_pullback
        # --- 2. 【新范式】按优先级生成唯一的战术信号  ---
        # 优先级 1 (S+++ 王牌): 巡航期 + 打压回踩(昨日) + 显性反转(今日) -> 经典的“黄金坑”V型反转
        s_triple_plus_signal = is_in_cruise_window & was_suppressive_pullback & is_reversal_confirmed
        states['TACTIC_CRUISE_PIT_REVERSAL_S_TRIPLE_PLUS'] = s_triple_plus_signal
        # 优先级 2 (S+): 巡航期 + 健康回踩(昨日) + 显性反转(今日) + 【非上涨末期】
        s_plus_signal = is_in_cruise_window & was_healthy_pullback & is_reversal_confirmed & is_in_safe_stage & ~s_triple_plus_signal
        states['TACTIC_CRUISE_PULLBACK_REVERSAL_S_PLUS'] = s_plus_signal
        # 优先级 3 (A+): 初升浪期 + 打压回踩(昨日) + 显性反转(今日)
        a_plus_signal = is_in_ascent_window & was_suppressive_pullback & is_reversal_confirmed & ~is_in_cruise_window
        states['TACTIC_ASCENT_PIT_REVERSAL_A_PLUS'] = a_plus_signal
        # 优先级 4 (A): 初升浪期 + 健康回踩(昨日) + 显性反转(今日)
        a_signal = is_in_ascent_window & was_healthy_pullback & is_reversal_confirmed & ~is_in_cruise_window & ~a_plus_signal
        states['TACTIC_ASCENT_PULLBACK_REVERSAL_A'] = a_signal
        # --- 3. 打印日志 (适配新战法名称)  ---
        tactic_name_map = {
            "CRUISE_PIT_REVERSAL": "巡航黄金坑V反(王牌)",
            "CRUISE_PULLBACK_REVERSAL": "巡航常规回踩确认",
            "ASCENT_PIT_REVERSAL": "初升浪黄金坑V反",
            "ASCENT_PULLBACK_REVERSAL": "初升浪常规回踩确认"
        }
        grade_map = {
            "S_TRIPLE_PLUS": "S+++", "S_PLUS": "S+", "A_PLUS": "A+", "A": "A"
        }
        for name, series in states.items():
            if series.any():
                matched_grade_key = ""
                for grade_key in sorted(grade_map.keys(), key=len, reverse=True):
                    if name.endswith(f"_{grade_key}"):
                        matched_grade_key = grade_key
                        break
                if matched_grade_key:
                    tactic_key_part = name.replace("TACTIC_", "").replace(f"_{matched_grade_key}", "")
                    cn_tactic = tactic_name_map.get(tactic_key_part, tactic_key_part)
                    cn_grade = grade_map.get(matched_grade_key, "")
                    print(f"          -> [{cn_grade}级战法] 侦测到 {series.sum()} 次“{cn_tactic}”机会！")
                else:
                    print(f"          -> [战法确认] 侦测到 {series.sum()} 次“{name}”机会！")
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

    def synthesize_advanced_tactics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.4 逻辑修复版】高级战法合成模块
        - 核心职责: 合成那些需要复杂时序逻辑的高级战法。
        - 本次升级: [修复] 新增对 `_diagnose_lock_chip_rally_tactic` 的调用，
                    修复了“锁筹拉升”战法从未被执行的逻辑缺陷。
        """
        # print("        -> [高级战法合成模块 V1.4 逻辑修复版] 启动...") 
        states = {}
        # --- 战法1: 【战法S+】断层新生·主升浪 ---
        states.update(self._diagnose_lock_chip_reconcentration_tactic(df))
        states.update(self._diagnose_lock_chip_rally_tactic(df))
        # --- 战法2: 【战法S+】断层新生·主升浪 (原战法1) ---
        atomic = self.strategy.atomic_states
        triggers = self.strategy.trigger_events
        default_series = pd.Series(False, index=df.index)
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        weights = {'S': 1.0, 'A': 0.6, 'B': 0.3}
        total_weight = sum(weights.values())
        # 更新对辅助函数的调用
        capitulation_s = self._get_atomic_score(df, 'SCORE_CAPITULATION_BOTTOM_RESONANCE_S', 0.0)
        capitulation_a = self._get_atomic_score(df, 'SCORE_CAPITULATION_BOTTOM_REVERSAL_A', 0.0)
        capitulation_b = self._get_atomic_score(df, 'SCORE_CAPITULATION_BOTTOM_REVERSAL_B', 0.0)
        fault_event_score = (capitulation_s * weights['S'] + capitulation_a * weights['A'] + capitulation_b * weights['B']) / total_weight
        confirmation_trigger_score_arr = np.maximum(
            triggers.get('TRIGGER_DOMINANT_REVERSAL', default_series).astype(float).values,
            triggers.get('TRIGGER_CHIP_IGNITION', default_series).astype(float).values
        )
        confirmation_trigger_score = pd.Series(confirmation_trigger_score_arr, index=df.index)
        main_uptrend_score = atomic.get('SCORE_STRUCTURE_MAIN_UPTREND_WAVE_S', default_score)
        fault_window_score = fault_event_score.rolling(window=3, min_periods=1).max()
        final_tactic_score = (fault_window_score * confirmation_trigger_score * main_uptrend_score).astype(np.float32)
        states['COGNITIVE_SCORE_FAULT_REBIRTH_ASCENT_S_PLUS'] = final_tactic_score
        p_advanced = get_params_block(self.strategy, 'advanced_tactics_params', {})
        final_signal_threshold = get_param_value(p_advanced.get('fault_rebirth_threshold'), 0.4)
        final_signal = final_tactic_score > final_signal_threshold
        states['TACTIC_FAULT_REBIRTH_ASCENT_S_PLUS'] = final_signal
        if final_signal.any():
            print(f"          -> [S+级战法重构版] 侦测到 {final_signal.sum()} 次“断层新生·主升浪”机会！")
        return states

    def synthesize_squeeze_playbooks(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.5 信号发布增强版】压缩突破战术剧本合成模块
        - 本次升级 (V1.5):
          - [信号发布] 将内部使用的 `vol_compression_score` 正式发布为
                        `COGNITIVE_SCORE_VOL_COMPRESSION_FUSED` 原子状态，供其他模块消费。
        - 收益: 解决了 PlaybookEngine 跨模块调用的架构问题，修复了因此引发的 AttributeError。
        """
        # print("        -> [压缩突破战术剧本合成模块 V1.5 信号发布增强版] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 剧本1: S+级 - 极致压缩·暴力突破 (数值化) ---
        # 战备分(昨日):
        # 将对已废弃信号的引用，升级为消费融合后的多层次分数
        vol_compression_score = self._fuse_multi_level_scores(df, 'VOL_COMPRESSION')
        # ▼▼▼ 将融合后的分数发布为原子状态 ▼▼▼
        states['COGNITIVE_SCORE_VOL_COMPRESSION_FUSED'] = vol_compression_score.astype(np.float32)
        # ▲▲▲ 新增结束 ▲▲▲
        setup_extreme_squeeze_score = vol_compression_score.shift(1).fillna(0.0)
        # 确认分(今日):
        trigger_explosive_breakout_score = atomic.get('SCORE_SQUEEZE_BREAKOUT_OPP_S', default_score)
        # 最终剧本分:
        score_s_plus = (setup_extreme_squeeze_score * trigger_explosive_breakout_score).astype(np.float32)
        states['SCORE_PLAYBOOK_EXTREME_SQUEEZE_EXPLOSION_S_PLUS'] = score_s_plus 
        states['PLAYBOOK_EXTREME_SQUEEZE_EXPLOSION_S_PLUS'] = score_s_plus > 0.7
        # --- 剧本2: S级 - 突破前夜·决战冲锋 (数值化) ---
        # 战备分(昨日):
        platform_quality_score = atomic.get('SCORE_PLATFORM_QUALITY_S', default_score)
        breakout_eve_score = (platform_quality_score * vol_compression_score)
        setup_breakout_eve_score = breakout_eve_score.shift(1).fillna(0.0)
        # 确认分(今日):
        trigger_prime_breakout_score = atomic.get('COGNITIVE_SCORE_IGNITION_RESONANCE_S', default_score)
        # 最终剧本分:
        score_s = (setup_breakout_eve_score * trigger_prime_breakout_score).astype(np.float32)
        states['SCORE_PLAYBOOK_BREAKOUT_EVE_S'] = score_s
        states['PLAYBOOK_BREAKOUT_EVE_S'] = score_s > 0.6
        # --- 剧本3: A级 - 常规压缩·确认突破 (数值化) ---
        # 战备分(昨日):
        setup_normal_squeeze_score = vol_compression_score.shift(1).fillna(0.0)
        # 确认分(今日):
        trigger_grinding_advance_score = atomic.get('COGNITIVE_SCORE_VOL_BREAKOUT_A', default_score)
        trigger_any_breakout_score = np.maximum(trigger_explosive_breakout_score.values, trigger_grinding_advance_score.values) # 确保在numpy层面操作
        # 最终剧本分 (注意：需要排除掉更高级别的S+剧本，保证信号互斥)
        score_a = (setup_normal_squeeze_score * pd.Series(trigger_any_breakout_score, index=df.index)).astype(np.float32)
        states['SCORE_PLAYBOOK_NORMAL_SQUEEZE_BREAKOUT_A'] = score_a
        states['PLAYBOOK_NORMAL_SQUEEZE_BREAKOUT_A'] = (score_a > 0.5) & ~states['PLAYBOOK_EXTREME_SQUEEZE_EXPLOSION_S_PLUS']
        return states

    def synthesize_industry_synergy_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】行业-个股协同元融合引擎
        - 核心职责: 将行业生命周期状态与个股的关键认知信号进行交叉验证，生成更高维度的战术信号。
        - 核心逻辑:
          - 协同进攻 = 行业景气周期 (预热/主升) * 个股进攻信号 (点火/突破)
          - 协同风险 = 行业衰退周期 (滞涨/下跌) * 个股风险信号 (崩溃/派发)
        - 收益: 创造出具备“戴维斯双击”效应的超高质量信号，极大提升策略的胜率和赔率。
        """
        print("        -> [行业-个股协同元融合引擎 V1.0] 启动...")
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

    def synthesize_cognitive_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.2 逻辑修正版】顶层认知总分合成模块
        - 核心职责: 融合所有顶层的、跨领域的认知机会分与风险分。
        - 本次升级:
          - [逻辑修正] 新增对“早期动能点火”信号的合成与调用，以解决追高问题。
          - [BUG修复] 将对“完美风暴”信号的引用从布尔型升级为数值化评分，
                    确保最终总分能正确反映其强度。
        """
        print("        -> [顶层认知总分合成模块 V1.2 逻辑修正版] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 合成“早期动能点火”信号 ---
        early_momentum_states = self.synthesize_early_momentum_ignition(df) # 调用早期动能诊断模块
        states.update(early_momentum_states) # 将新信号加入states
        # 调用“筹码共振-价格滞后”剧本诊断模块
        chip_price_lag_states = self.synthesize_chip_price_lag_playbook(df)
        states.update(chip_price_lag_states)
        industry_synergy_states = self.synthesize_industry_synergy_signals(df)
        states.update(industry_synergy_states)
        # 调用“伪装散户吸筹”诊断引擎
        deceptive_flow_states = self.diagnose_deceptive_retail_flow(df)
        states.update(deceptive_flow_states)
        # --- 运行微观结构动态诊断引擎 ---
        microstructure_dynamic_states = self.synthesize_microstructure_dynamics(df)
        states.update(microstructure_dynamic_states)
        # --- 1. 汇总所有S级的“机会”类认知分数 ---
        bullish_scores = [
            atomic.get('COGNITIVE_SCORE_IGNITION_RESONANCE_S', default_score).values,
            atomic.get('COGNITIVE_SCORE_BOTTOM_REVERSAL_RESONANCE_S', default_score).values,
            atomic.get('COGNITIVE_SCORE_CLASSIC_PATTERN_OPP_S', default_score).values,
            atomic.get('COGNITIVE_SCORE_BEARISH_EXHAUSTION_OPP_S', default_score).values,
            atomic.get('COGNITIVE_SCORE_CONSOLIDATION_BREAKOUT_OPP_A', default_score).values,
            atomic.get('COGNITIVE_SCORE_PERFECT_STORM_BOTTOM_S_PLUS', default_score).values,
            states.get('COGNITIVE_SCORE_INDUSTRY_SYNERGY_OFFENSE_S', default_score).values,
            states.get('COGNITIVE_SCORE_OPP_POWER_SHIFT_TO_MAIN_FORCE', default_score).values,
            states.get('COGNITIVE_SCORE_OPP_MAIN_FORCE_CONVICTION_STRENGTHENING', default_score).values, # “主力信念加强”机会信号
        ]
        cognitive_bullish_score = np.maximum.reduce(bullish_scores)
        states['COGNITIVE_BULLISH_SCORE'] = pd.Series(cognitive_bullish_score, index=df.index, dtype=np.float32)
        # --- 2. 调用风险元融合模块 ---
        fused_risk_states = self.synthesize_fused_risk_scores(df)
        states.update(fused_risk_states) # 将所有融合后的风险分（包括各维度分和总分）加入states
        # --- 3. 更新顶层认知熊市总分 ---
        # 直接使用融合后的风险总分作为顶层熊市分
        # 注意：COGNITIVE_FUSED_RISK_SCORE 是加权后的分数，可能超过1，这里需要归一化或直接使用
        # 暂时直接使用，下游计分系统需要能处理大于1的风险分
        cognitive_bearish_score_series = states.get('COGNITIVE_FUSED_RISK_SCORE', default_score)
        industry_synergy_risk_score = states.get('COGNITIVE_SCORE_INDUSTRY_SYNERGY_RISK_S', default_score)
        microstructure_conviction_risk_score = states.get('COGNITIVE_SCORE_RISK_MAIN_FORCE_CONVICTION_WEAKENING', default_score)
        # “主导权向散户转移”风险信号
        microstructure_power_shift_risk_score = states.get('COGNITIVE_SCORE_RISK_POWER_SHIFT_TO_RETAIL', default_score)
        final_bearish_score = np.maximum.reduce([
            cognitive_bearish_score_series.values, 
            industry_synergy_risk_score.values,
            microstructure_conviction_risk_score.values,
            microstructure_power_shift_risk_score.values
        ])
        states['COGNITIVE_BEARISH_SCORE'] = pd.Series(final_bearish_score, index=df.index, dtype=np.float32)
        # --- 4. 更新原子状态库 ---
        self.strategy.atomic_states.update(states)
        print("        -> [顶层认知总分合成模块 V1.2 风险源升级版] 计算完毕。") # 修改: 更新版本号
        return df

    def diagnose_deceptive_retail_flow(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 VPA增强版】伪装散户吸筹诊断引擎 (主力分单行为识别)
        - 架构归属: 从 ChipIntelligence 迁移至 CognitiveIntelligence，因为它融合了筹码、资金、价格、量价四大维度。
        - 核心增强: 新增对 VPA 效率的判断，形成四维交叉验证，极大提升信号置信度。
        - 核心逻辑:
          1. 资金流表象: 散户资金持续净流入。
          2. 筹码结构结果: 筹码持续集中。
          3. 价格环境: 股价波动被压制。
          4. 量价效率佐证 (VPA): 成交量很大，但价格波动很小，证明交易未用于推升价格。
        - 产出: SCORE_COGNITIVE_DECEPTIVE_RETAIL_ACCUMULATION_S - 一个高置信度的、识别主力隐蔽吸筹的S级认知信号。
        """
        print("        -> [伪装散户吸筹诊断引擎 V2.0 VPA增强版] 启动...")
        states = {}
        p = get_params_block(self.strategy, 'deceptive_flow_params', {})
        if not get_param_value(p.get('enabled'), True):
            return states

        # --- 1. 军备检查 ---
        required_cols = [
            'retail_net_flow_consensus_D',      # 资金流表象
            'SLOPE_5_concentration_90pct_D',    # 筹码结构结果
            'SLOPE_5_close_D',                  # 价格环境
            'VPA_EFFICIENCY_D'                  # 量价效率佐证 (VPA)
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"          -> [严重警告] 伪装散户吸筹诊断引擎缺少关键数据: {missing_cols}，模块已跳过！")
            return states

        # --- 2. 核心要素数值化 (归一化) ---
        norm_window = get_param_value(p.get('norm_window'), 120)

        # 条件1: 散户资金持续净流入 (值越大，分数越高)
        retail_inflow_score = self._normalize_score(df['retail_net_flow_consensus_D'].clip(lower=0), norm_window, ascending=True)

        # 条件2: 筹码集中度持续提升 (斜率为负且越小，分数越高)
        chip_concentration_score = self._normalize_score(df['SLOPE_5_concentration_90pct_D'], norm_window, ascending=False)

        # 条件3: 价格波动被压制 (价格斜率的绝对值越小，分数越高)
        price_suppression_score = self._normalize_score(df['SLOPE_5_close_D'].abs(), norm_window, ascending=False)

        # 新增条件4: VPA效率低下 (VPA效率值越小，分数越高)
        vpa_inefficiency_score = self._normalize_score(df['VPA_EFFICIENCY_D'], norm_window, ascending=False)

        # --- 3. 融合生成最终信号 ---
        # 四维交叉验证，只有当四个条件同时满足时，分数才会高
        final_score = (
            retail_inflow_score *
            chip_concentration_score *
            price_suppression_score *
            vpa_inefficiency_score
        ).astype(np.float32)
        
        states['SCORE_COGNITIVE_DECEPTIVE_RETAIL_ACCUMULATION_S'] = final_score

        if (final_score > 0.85).any():
             print(f"          -> [S级认知信号] 侦测到 {(final_score > 0.85).sum()} 次高度疑似“伪装散户吸筹”的博弈行为！")

        print("        -> [伪装散户吸筹诊断引擎 V2.0] 计算完毕。")
        return states

    def synthesize_microstructure_dynamics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 完全对称版】市场微观结构动态诊断引擎
        - 核心升级 (本次修改):
          - [对称实现] 补全了所有机会和风险的镜像信号，现在能同时诊断四种状态：
            1. 机会：主导权向主力转移
            2. 风险：主导权向散户转移 (新增)
            3. 机会：主力信念在加强 (新增)
            4. 风险：主力信念在瓦解
        - 收益: 实现了对市场微观结构变化的完全对称、无死角的监控。
        """
        # print("        -> [市场微观结构动态诊断引擎 V2.0 完全对称版] 启动...") # 修改: 更新版本号
        states = {}
        norm_window = 120

        # --- 1. 诊断“市场主导权”的转移方向 ---
        # 机会: 主导权向主力转移 (交易颗粒度和集中度加速提升)
        granularity_momentum_up = self._normalize_score(df.get('SLOPE_5_avg_order_value_D'), norm_window, ascending=True)
        granularity_accel_up = self._normalize_score(df.get('ACCEL_5_avg_order_value_D'), norm_window, ascending=True)
        dominance_momentum_up = self._normalize_score(df.get('SLOPE_5_trade_concentration_index_D'), norm_window, ascending=True)
        dominance_accel_up = self._normalize_score(df.get('ACCEL_5_trade_concentration_index_D'), norm_window, ascending=True)
        power_shift_to_main_force_score = (
            granularity_momentum_up * granularity_accel_up *
            dominance_momentum_up * dominance_accel_up
        ).astype(np.float32)
        states['COGNITIVE_SCORE_OPP_POWER_SHIFT_TO_MAIN_FORCE'] = power_shift_to_main_force_score
        # --- 风险的镜像信号 ---
        # 风险: 主导权向散户转移 (交易颗粒度和集中度加速下降)
        granularity_momentum_down = self._normalize_score(df.get('SLOPE_5_avg_order_value_D'), norm_window, ascending=False)
        granularity_accel_down = self._normalize_score(df.get('ACCEL_5_avg_order_value_D'), norm_window, ascending=False)
        dominance_momentum_down = self._normalize_score(df.get('SLOPE_5_trade_concentration_index_D'), norm_window, ascending=False)
        dominance_accel_down = self._normalize_score(df.get('ACCEL_5_trade_concentration_index_D'), norm_window, ascending=False)
        power_shift_to_retail_risk = (
            granularity_momentum_down * granularity_accel_down *
            dominance_momentum_down * dominance_accel_down
        ).astype(np.float32)
        states['COGNITIVE_SCORE_RISK_POWER_SHIFT_TO_RETAIL'] = power_shift_to_retail_risk

        # --- 2. 诊断“主力信念”的动态变化 ---
        # 风险: 主力信念在瓦解 (信念比率加速下降)
        conviction_momentum_weakening = self._normalize_score(df.get('SLOPE_5_main_force_conviction_ratio_D'), norm_window, ascending=False)
        conviction_accel_weakening = self._normalize_score(df.get('ACCEL_5_main_force_conviction_ratio_D'), norm_window, ascending=False)
        conviction_weakening_risk = (conviction_momentum_weakening * conviction_accel_weakening).astype(np.float32)
        states['COGNITIVE_SCORE_RISK_MAIN_FORCE_CONVICTION_WEAKENING'] = conviction_weakening_risk
        # --- 机会的镜像信号 ---
        # 机会: 主力信念在加强 (信念比率加速上升)
        conviction_momentum_strengthening = self._normalize_score(df.get('SLOPE_5_main_force_conviction_ratio_D'), norm_window, ascending=True)
        conviction_accel_strengthening = self._normalize_score(df.get('ACCEL_5_main_force_conviction_ratio_D'), norm_window, ascending=True)
        conviction_strengthening_opp = (conviction_momentum_strengthening * conviction_accel_strengthening).astype(np.float32)
        states['COGNITIVE_SCORE_OPP_MAIN_FORCE_CONVICTION_STRENGTHENING'] = conviction_strengthening_opp

        # print("        -> [市场微观结构动态诊断引擎 V2.0] 计算完毕。") # 修改: 更新版本号
        return states









