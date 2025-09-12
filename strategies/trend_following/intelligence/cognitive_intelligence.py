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

    def synthesize_fused_risk_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.0 配置驱动重构版】风险元融合模块
        - 核心重构 (本次修改):
          - [NameError修复] 彻底废弃了方法内硬编码的风险定义，现在完全从配置文件 `fused_risk_scoring` 块中读取 `risk_categories`、`dynamic_weighting_params` 等所有参数，从根源上解决了 `risk_categories` 未定义的错误。
          - [TypeError修复] 修正了维度内融合的致命逻辑错误，现在能正确解析每个信号的 `weight` 和 `inverse` 属性，并进行正确的数值计算。
          - [健壮性提升] 为所有从配置中读取的参数添加了默认值，使模块功能更加稳固。
        - 收益: 实现了完全由配置文件驱动的、逻辑正确且运行稳定的新一代风险融合引擎。
        """
        # print("        -> [风险元融合模块 V3.0 配置驱动重构版] 启动...")
        states = {}
        p_fused_risk = get_params_block(self.strategy, 'fused_risk_scoring')
        if not get_param_value(p_fused_risk.get('enabled'), True):
            print("        -> [风险元融合模块] 已在配置中禁用，跳过计算。")
            states['COGNITIVE_FUSED_RISK_SCORE'] = pd.Series(0.0, index=df.index, dtype=np.float32)
            return states

        risk_categories = p_fused_risk.get('risk_categories', {})
        p_dynamic_weighting = p_fused_risk.get('dynamic_weighting_params', {})
        base_weights = p_dynamic_weighting.get('base_weights', {})
        context_adjustments = p_dynamic_weighting.get('context_adjustments', {})
        # 假设 intra_dimension_fusion_params 与 resonance_penalty_params 也在 p_fused_risk 下
        p_fusion_params = p_fused_risk.get('intra_dimension_fusion_params', {})
        secondary_risk_discount = p_fusion_params.get('secondary_risk_discount', 0.3)
        p_resonance = p_fused_risk.get('resonance_penalty_params', {})
        fused_dimension_scores = {}
        default_series = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 维度内融合：【深化升级】采用“主次融合”逻辑 ---
        # print("          -> 步骤1: 执行维度内风险主次融合...")
        for category_name, signals in risk_categories.items():
            if category_name == "说明": continue
            category_signal_scores = []
            for signal_name, signal_params in signals.items(): # signal_params 是一个字典，如 {"weight": 1.0, "inverse": true}
                if signal_name == "说明": continue
                atomic_score = self._get_atomic_score(df, signal_name, 0.0)
                # 正确处理 inverse 逻辑
                is_inverse = signal_params.get('inverse', False)
                if is_inverse:
                    processed_score = 1.0 - atomic_score
                else:
                    processed_score = atomic_score
                # 正确应用 weight
                weight = signal_params.get('weight', 1.0)
                final_signal_score = processed_score * weight
                final_signal_score.name = signal_name # 为Series命名，便于后续排序
                category_signal_scores.append(final_signal_score)
            if category_signal_scores:
                category_df = pd.concat(category_signal_scores, axis=1)
                sorted_scores = np.sort(category_df.values, axis=1)
                primary_risk = pd.Series(sorted_scores[:, -1], index=df.index)
                secondary_risk = pd.Series(sorted_scores[:, -2] if sorted_scores.shape[1] > 1 else 0, index=df.index)
                dimension_risk_score = primary_risk + secondary_risk * secondary_risk_discount
                fused_dimension_scores[category_name] = dimension_risk_score
                states[f'FUSED_RISK_SCORE_{category_name.upper()}'] = dimension_risk_score.astype(np.float32)
            else:
                fused_dimension_scores[category_name] = default_series.copy()
        # --- 2. 维度间融合：【深化升级】应用“动态风险加权” ---
        # print("          -> 步骤2: 应用市场阶段进行动态风险加权...")
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
        # --- 3. 风险共振惩罚：【深化升级】对协同风险施加额外惩罚 ---
        # print("          -> 步骤3: 检测风险共振并施加惩罚...")
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
        # print(f"        -> [风险元融合模块 V3.0] 计算完毕，生成了 {len(states)} 个结构化风险信号。")
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
        # --- 战术 1: 缺口回补支撑机会 (逻辑不变) ---
        gap_support_strength_score = atomic.get('SCORE_GAP_SUPPORT_ACTIVE', default_score)
        healthy_pullback_score = atomic.get('COGNITIVE_SCORE_PULLBACK_HEALTHY_S', default_score)
        states['COGNITIVE_SCORE_OPP_GAP_SUPPORT_PULLBACK'] = (gap_support_strength_score * healthy_pullback_score).astype(np.float32)
        # --- 战术 2: 斐波那契关键位反弹确认机会 (逻辑不变) ---
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
        # --- 2. 计算“上升趋势可持续性”评分 (逻辑不变) ---
        states['COGNITIVE_SCORE_TREND_SUSTAINABILITY_UP'] = (trend_quality * (1 - top_reversal_potential)).astype(np.float32)
        # --- 3. 计算“上升趋势衰竭”风险评分 (逻辑不变) ---
        states['COGNITIVE_SCORE_TREND_FATIGUE_RISK'] = (top_reversal_potential * high_level_zone_context).astype(np.float32)
        # --- 4. 计算“下跌趋势衰竭”机会评分 (逻辑不变) ---
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
        【V1.7 终极结构层信号适配版】多域点火共振分数合成模块
        - 核心升级: [信号适配] 将对结构层突破信号的消费，从旧的 'SCORE_STRUCTURAL_ACCUMULATION_BREAKOUT_S'
                    适配为消费由 StructuralIntelligence V2.0 生成的 'STRUCTURE_BULLISH_RESONANCE' 终极信号。
        """
        # print("        -> [多域点火共振分数合成模块 V1.7 终极结构层信号适配版] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 提取八大领域的核心“点火”分数 ---
        # 领域1 (筹码剧本 - 最高优先级):
        chip_playbook_ignition = self._get_atomic_score(df, 'SCORE_CHIP_PLAYBOOK_VACUUM_BREAKOUT', 0.0)
        # 领域2 (筹码共识 - 次级优先级):
        chip_consensus_ignition = self._fuse_multi_level_scores(df, 'CHIP_BULLISH_RESONANCE')
        # 其他领域信号
        behavioral_ignition = self._fuse_multi_level_scores(df, 'BEHAVIOR_BULLISH_RESONANCE')
        structural_breakout = self._fuse_multi_level_scores(df, 'STRUCTURE_BULLISH_RESONANCE')
        mechanics_ignition = self._fuse_multi_level_scores(df, 'DYN_BULLISH_RESONANCE')
        fund_flow_ignition = self._fuse_multi_level_scores(df, 'FF_BULLISH_RESONANCE')
        volatility_breakout = self._fuse_multi_level_scores(df, 'VOL_BREAKOUT')
        # --- 2. 交叉验证：生成“多域点火共振”元分数 ---
        general_ignition_resonance = (
            behavioral_ignition * structural_breakout * mechanics_ignition *
            chip_consensus_ignition * fund_flow_ignition * volatility_breakout
        )
        # 使用 np.maximum 确保我们捕捉到最强的信号：要么是特定剧本发生，要么是普遍的共振
        ignition_resonance_score = np.maximum(chip_playbook_ignition, general_ignition_resonance).astype(np.float32)
        states['COGNITIVE_SCORE_IGNITION_RESONANCE_S'] = ignition_resonance_score
        self.strategy.atomic_states.update(states)
        # print("        -> [多域点火共振分数合成模块 V2.0] 计算完毕。")
        return df

    def synthesize_reversal_resonance_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V2.3 终极基础层信号适配版】多域反转共振分数合成模块
        - 核心升级: [信号适配] 将对均线、形态等基础/结构层反转信号的消费，
                    统一升级为消费由 FoundationIntelligence V3.0 生成的 'FOUNDATION_BOTTOM_REVERSAL'
                    和 'FOUNDATION_TOP_REVERSAL' 终极信号。
        - 收益: 确保了反转共振的判断基于最可靠、经过四重交叉验证的基础层共识信号。
        """
        # print("        -> [多域反转共振分数合成模块 V2.3 终极基础层信号适配版] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        p = get_params_block(self.strategy, 'reversal_resonance_params', {})
        bottom_weights = get_param_value(p.get('bottom_trigger_weights'), {})
        top_weights = get_param_value(p.get('top_trigger_weights'), {})
        # --- 1. 合成“底部反转共振”分数 ---
        oversold_context = np.maximum(atomic.get('SCORE_RSI_OVERSOLD_EXTENT', default_score).values, atomic.get('SCORE_BIAS_OVERSOLD_EXTENT', default_score).values)
        mechanics_bottom_setup = self._fuse_multi_level_scores(df, 'DYN_BOTTOM_REVERSAL')
        chip_ultimate_bottom_reversal = self._fuse_multi_level_scores(df, 'BOTTOM_REVERSAL')
        # 将旧的 ma_bottom_setup 替换为更可靠的基础层共识信号
        foundation_bottom_setup = self._fuse_multi_level_scores(df, 'FOUNDATION_BOTTOM_REVERSAL')
        reversal_setup_score = (mechanics_bottom_setup * chip_ultimate_bottom_reversal * foundation_bottom_setup) # 使用新的 foundation_bottom_setup
        # 将 pattern_trigger 和 foundation_trigger 统一为 foundation_trigger
        # foundation_trigger 现在代表了基础层和形态层的共同反转信号
        foundation_trigger = self._fuse_multi_level_scores(df, 'FOUNDATION_BOTTOM_REVERSAL')
        behavioral_trigger = self._fuse_multi_level_scores(df, 'BEHAVIOR_BOTTOM_REVERSAL')
        # 更新 trigger_scores 字典
        trigger_scores = {'foundation': foundation_trigger, 'behavioral': behavioral_trigger}
        final_trigger_score = pd.Series(0.0, index=df.index)
        total_weight_bottom = sum(bottom_weights.values())
        if total_weight_bottom > 0:
            for key, weight in bottom_weights.items():
                final_trigger_score += trigger_scores.get(key, default_score) * weight
            final_trigger_score /= total_weight_bottom
        bottom_reversal_score = pd.Series(oversold_context, index=df.index) * reversal_setup_score * final_trigger_score
        states['COGNITIVE_SCORE_BOTTOM_REVERSAL_RESONANCE_S'] = bottom_reversal_score.astype(np.float32)
        # --- 2. 合成“顶部反转共振”分数 (对称逻辑) ---
        overbought_context = np.maximum(atomic.get('SCORE_RSI_OVERBOUGHT_EXTENT', default_score).values, atomic.get('SCORE_BIAS_OVERBOUGHT_EXTENT', default_score).values)
        mechanics_top_setup = self._fuse_multi_level_scores(df, 'DYN_TOP_REVERSAL')
        chip_ultimate_top_reversal = self._fuse_multi_level_scores(df, 'TOP_REVERSAL')
        # 将旧的 ma_top_setup 替换为更可靠的基础层共识信号
        foundation_top_setup = self._fuse_multi_level_scores(df, 'FOUNDATION_TOP_REVERSAL')
        top_setup_score = (mechanics_top_setup * chip_ultimate_top_reversal * foundation_top_setup) # 使用新的 foundation_top_setup
        # 将 pattern_top_trigger 和 foundation_top_trigger 统一为 foundation_top_trigger
        foundation_top_trigger = self._fuse_multi_level_scores(df, 'FOUNDATION_TOP_REVERSAL')
        behavioral_top_trigger = self._fuse_multi_level_scores(df, 'BEHAVIOR_TOP_REVERSAL')
        # 更新 top_trigger_scores 字典
        top_trigger_scores = {'foundation': foundation_top_trigger, 'behavioral': behavioral_top_trigger}
        final_top_trigger_score = pd.Series(0.0, index=df.index)
        total_weight_top = sum(top_weights.values())
        if total_weight_top > 0:
            for key, weight in top_weights.items():
                final_top_trigger_score += top_trigger_scores.get(key, default_score) * weight
            final_top_trigger_score /= total_weight_top
        top_reversal_score = pd.Series(overbought_context, index=df.index) * top_setup_score * final_top_trigger_score
        states['COGNITIVE_SCORE_TOP_REVERSAL_RESONANCE_S'] = top_reversal_score.astype(np.float32)
        self.strategy.atomic_states.update(states)
        # print("        -> [多域反转共振分数合成模块 V2.3 终极基础层信号适配版] 计算完毕。")
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
        # --- 2. 定义触发的战场环境分数 (逻辑不变) ---
        danger_zone_score = atomic.get('COGNITIVE_SCORE_RISK_HIGH_LEVEL_ZONE', default_score)
        # --- 3. 最终裁定 (逻辑不变) ---
        max_divergence_score = np.maximum.reduce([oscillator_divergence_score, price_momentum_divergence_score.values, exhaustion_divergence_score.values, engine_divergence_score.values])
        final_risk_score = danger_zone_score * pd.Series(max_divergence_score, index=df.index)
        states['COGNITIVE_SCORE_MULTI_DIMENSIONAL_DIVERGENCE_S'] = final_risk_score.astype(np.float32)
        # --- 4. 更新原子状态库 ---
        self.strategy.atomic_states.update(states)
        # print("        -> [多维背离风险融合模块 V2.2] 计算完毕。")
        return df

    def synthesize_market_engine_states(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V2.1 行为信号升级版】市场引擎状态融合模块
        - 核心重构 (本次修改):
          - [信号适配] 将“引擎失速”的判断依据，从旧的、特定的信号，升级为消费
                        `BehavioralIntelligence` V4.0产出的、更通用的“顶部反转”融合分数。
        - 收益: 统一了对引擎失效的判断标准，信号源更可靠。
        """
        # print("        -> [市场引擎状态融合模块 V2.1 行为信号升级版] 启动...") 
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 提取引擎失效的多个症状分数 ---
        # 将旧的、特定的引擎失速信号，替换为新的、通用的顶部反转融合分
        engine_stalling_score = self._fuse_multi_level_scores(df, 'BEHAVIOR_TOP_REVERSAL')
        vpa_stagnation_score = self._get_atomic_score(df, 'SCORE_RISK_VPA_STAGNATION', 0.0)
        bearish_divergence_score = self._get_atomic_score(df, 'SCORE_RISK_MTF_RSI_DIVERGENCE_S', 0.0)
        # --- 2. 定义触发的战场环境分数 ---
        danger_zone_score = atomic.get('COGNITIVE_SCORE_RISK_HIGH_LEVEL_ZONE', default_score)
        # --- 3. 最终裁定 ---
        max_symptom_score_series = pd.concat([
            engine_stalling_score,
            vpa_stagnation_score,
            bearish_divergence_score
        ], axis=1).max(axis=1).fillna(0.0)
        final_risk_score = danger_zone_score * max_symptom_score_series
        states['COGNITIVE_SCORE_ENGINE_FAILURE_S'] = final_risk_score.astype(np.float32)
        # --- 4. 更新原子状态库 ---
        self.strategy.atomic_states.update(states)
        # print("        -> [市场引擎状态融合模块 V2.1] 计算完毕。") 
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
        cycle_trough_score = (1 - self._get_atomic_score(df, 'DOMINANT_CYCLE_PHASE', 0.0)) / 2.0
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

    def synthesize_contextual_zone_scores(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V2.7 终极信号适配版】战场上下文评分模块
        - 核心升级: [信号适配] 全面审查并更新了所有筹码和行为维度的风险信号消费，
                    确保其完全基于最新的终极信号。
        """
        # print("        -> [战场上下文评分模块 V2.7 终极信号适配版] 启动...")
        risk_scores = {
            'bias': self._get_atomic_score(df, 'SCORE_BIAS_OVERBOUGHT_EXTENT', 0.0),
            'exhaustion': np.maximum(
                self._get_atomic_score(df, 'SCORE_RISK_MOMENTUM_EXHAUSTION', 0.0).values,
                self._get_atomic_score(df, 'SCORE_RISK_PROFIT_EXHAUSTION_S', 0.0).values
            ),
            'deceptive_rally': self._get_atomic_score(df, 'CHIP_SCORE_FUSED_DECEPTIVE_RALLY', 0.0),
            'chip_top_reversal': self._fuse_multi_level_scores(df, 'TOP_REVERSAL'), # 已适配
            'structural_weakness': self._get_atomic_score(df, 'SCORE_MTF_STRUCTURAL_WEAKNESS_RISK_S', 0.0),
            'engine_stalling': self._fuse_multi_level_scores(df, 'BEHAVIOR_TOP_REVERSAL'), # 适配行为层终极信号
            'volume_spike_down': self._get_atomic_score(df, 'SCORE_VOL_PRICE_PANIC_DOWN_RISK', 0.0),
            'chip_divergence': self._fuse_multi_level_scores(df, 'FALLING_RESONANCE'), # 已适配
            'chip_fault': self._fuse_multi_level_scores(df, 'FAULT_RISK_TOP_REVERSAL'), # 保留，此为特定结构风险
            'structural_fault': self._fuse_multi_level_scores(df, 'STRUCTURE_BEARISH_RESONANCE'), # 保留，此为特定结构风险
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
        risk_df = pd.DataFrame(risk_scores)
        high_level_zone_score = risk_df.max(axis=1)
        df['COGNITIVE_SCORE_RISK_HIGH_LEVEL_ZONE'] = high_level_zone_score
        self.strategy.atomic_states['COGNITIVE_SCORE_RISK_HIGH_LEVEL_ZONE'] = df['COGNITIVE_SCORE_RISK_HIGH_LEVEL_ZONE']
        high_risk_threshold = high_level_zone_score.rolling(120).quantile(0.85)
        is_in_high_level_zone = high_level_zone_score > high_risk_threshold
        self.strategy.atomic_states['CONTEXT_RISK_HIGH_LEVEL_ZONE'] = is_in_high_level_zone
        # print("        -> [战场上下文评分模块 V2.7 终极信号适配版] 计算完毕。")
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

    def diagnose_trend_stage_score(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V401.15 信号源适配版】趋势阶段评分模块
        - 核心升级: [信号适配] 修正了风险维度解析逻辑，使其能正确识别并消费
                    'TOP_REVERSAL' 等终极信号。
        - 收益: 确保了对上涨末期风险的评估，能够纳入来自筹码情报模块的最高质量信号。
        """
        # print("        -> [趋势阶段评分模块 V401.15 信号源适配版] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        p_trend_stage_scoring = get_params_block(self.strategy, 'trend_stage_scoring_params')
        if not get_param_value(p_trend_stage_scoring.get('enabled'), True):
            states['COGNITIVE_SCORE_CONTEXT_LATE_STAGE'] = pd.Series(0.0, index=df.index, dtype=np.float32)
            states['CONTEXT_TREND_STAGE_EARLY'] = pd.Series(False, index=df.index)
            print("        -> [趋势阶段评分模块] 已在配置中禁用，跳过计算。")
            return states
        signal_definitions = get_param_value(p_trend_stage_scoring.get('weights'), {})
        # --- 1. 计算“上涨初期”的量化分数 (Early Stage Score) ---
        ascent_structure_score = atomic.get('COGNITIVE_SCORE_ACCUMULATION_BREAKOUT_S', default_score)
        yearly_high = df['high_D'].rolling(250, min_periods=60).max()
        yearly_low = df['low_D'].rolling(250, min_periods=60).min()
        price_range = (yearly_high - yearly_low).replace(0, np.nan)
        price_position_score = 1 - ((df['close_D'] - yearly_low) / price_range)
        price_position_score = price_position_score.clip(0, 1).fillna(0.5)
        early_stage_score = pd.concat([ascent_structure_score, price_position_score], axis=1).max(axis=1)
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
        # 更新此列表以正确识别新的终极信号名
        fuse_list = ["TOP_REVERSAL", "FALLING_RESONANCE", "STRUCTURE_BEARISH_RESONANCE", "MTF_BEARISH_RESONANCE", "MECHANICS_TOP_REVERSAL", "PATTERN_TOP_REVERSAL"]
        for name, weight in signal_definitions.items():
            if name == "vpa_risk_score_series":
                risk_dimension_scores.append(vpa_risk_score_series * weight)
            elif name in fuse_list: # 使用新的列表进行判断
                risk_dimension_scores.append(self._fuse_multi_level_scores(df, name) * weight)
            else:
                risk_dimension_scores.append(self._get_atomic_score(df, name, 0.0) * weight)
        score_components = [s.to_numpy(dtype=np.float32) for s in risk_dimension_scores]
        late_stage_raw_score = pd.Series(np.add.reduce(score_components), index=df.index, dtype=np.float32)
        late_stage_raw_score = late_stage_raw_score.fillna(0.0)
        
        start_threshold = get_param_value(p_trend_stage_scoring.get('late_stage_start_threshold'), 160)
        full_threshold = get_param_value(p_trend_stage_scoring.get('late_stage_full_threshold'), 300)
        scaling_range = full_threshold - start_threshold
        scaling_range = max(scaling_range, 1) # 避免除以零
        # 计算0-1的平滑概率分
        late_stage_prob_score = ((late_stage_raw_score - start_threshold) / scaling_range).clip(0, 1)
        states['COGNITIVE_SCORE_CONTEXT_LATE_STAGE'] = late_stage_prob_score.astype(np.float32)
        # 为了兼容旧的布尔状态，可以基于新的概率分生成
        states['CONTEXT_TREND_STAGE_LATE'] = late_stage_prob_score > 0.5 

        # print("        -> [趋势阶段评分模块 V401.15 信号源适配版] 计算完毕。")
        return states

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
        # --- 3. 定义其他行为状态 (逻辑不变) ---
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
        【V2.2 终极资金流信号适配版】筹码与资金流协同合成模块
        - 核心升级: [信号适配] 将对资金流维度信号的消费，从旧的 'FF_SCORE_SEPTAFECTA_*'
                    适配为消费由 FundFlowIntelligence V19.0 生成的 'FF_*_RESONANCE' 终极信号。
        """
        # print("        -> [筹码与资金流协同合成模块 V2.2 终极资金流信号适配版] 启动...")
        states = {}
        # --- 1. 获取核心筹码与资金流的数值化融合分数 ---
        chip_bullish_score = self._fuse_multi_level_scores(df, 'RISING_RESONANCE')
        chip_bearish_score = self._fuse_multi_level_scores(df, 'FALLING_RESONANCE')
        # 获取资金流最高质量的终极融合分数
        fund_flow_bullish_score = self._fuse_multi_level_scores(df, 'FF_BULLISH_RESONANCE') # 适配新的终极资金流看涨共振信号
        fund_flow_bearish_score = self._fuse_multi_level_scores(df, 'FF_BEARISH_RESONANCE') # 适配新的终极资金流看跌共振信号
        # --- 2. 交叉验证：生成协同吸筹分数 ---
        synergy_accumulation_score = (chip_bullish_score * fund_flow_bullish_score).astype(np.float32)
        states['COGNITIVE_SCORE_CHIP_FUND_FLOW_ACCUMULATION_S'] = synergy_accumulation_score
        # --- 3. 交叉验证：生成协同派发风险分数 ---
        synergy_distribution_score = (chip_bearish_score * fund_flow_bearish_score).astype(np.float32)
        states['COGNITIVE_SCORE_CHIP_FUND_FLOW_DISTRIBUTION_S'] = synergy_distribution_score
        # --- 4. 为了兼容性，可以基于分数生成旧的布尔信号 ---
        states['CHIP_FUND_FLOW_ACCUMULATION_CONFIRMED_A'] = synergy_accumulation_score > 0.6
        states['CHIP_FUND_FLOW_ACCUMULATION_STRONG_S'] = synergy_accumulation_score > 0.8
        states['RISK_CHIP_FUND_FLOW_DISTRIBUTION_A'] = synergy_distribution_score > 0.7
        # print("        -> [筹码与资金流协同合成模块 V2.2 终极资金流信号适配版] 合成完毕。")
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
        【V2.3 王牌重铸数值化增强版】锁仓再集中S+战法诊断模块
        - 核心重构: (V2.0) 将战法的“准备状态”从有缺陷的A级信号，升级为经过战场环境过滤的
                      S级“筹码结构黄金机会”信号。
        - 本次升级: 【数值化】将原有的布尔逻辑升级为“战备分 * 点火分”的数值化评分体系，
                    能够更精确地量化战法机会的质量。
        - 核心修复 (V2.3): 修复了对 TRIGGER_ENERGY_RELEASE 和 FRACTAL_OPP_SQUEEZE_BREAKOUT_CONFIRMED
                        两个失效信号的引用，替换为更高质量的数值化评分。
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
        # 修复失效引用: 将 TRIGGER_ENERGY_RELEASE 替换为更高维度的力学元融合分数
        energy_release_score = atomic.get('SCORE_DYN_OVERALL_BULLISH_MOMENTUM_S', default_score) 
        cost_accel_score = atomic.get('SCORE_PLATFORM_COST_ACCEL', default_score)
        # 修复失效引用: 将 FRACTAL_OPP_SQUEEZE_BREAKOUT_CONFIRMED 替换为协同情报中心的压缩突破分
        squeeze_breakout_score = atomic.get('SCORE_SQUEEZE_BREAKOUT_OPP_S', default_score) 
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
        【V7.3 信号源更新版】回踩战术诊断模块
        - 核心升级: 为 S+ 级“巡航回踩确认”战法增加了“非上涨末期”的前置条件。
        - 本次升级: [信号修复] 更新了对“蓄势突破”信号的引用，现在消费来自 StructuralIntelligence
                    的 `STRUCTURAL_OPP_ACCUMULATION_BREAKOUT_S` 信号。
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
        ascent_start_event = atomic.get('STRUCTURAL_OPP_ACCUMULATION_BREAKOUT_S', default_series)
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
        【V2.0 信号修复与逻辑同步版】战术决策日志探针
        - 核心重构: 全面修复所有失效和过时的信号引用，并使其逻辑与
                      `_diagnose_pullback_tactics_matrix` V7.2 版本完全同步。
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
        ascent_start_event = atomic.get('COGNITIVE_OPP_ACCUMULATION_BREAKOUT_S', default_series)
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
        【V1.4 融合函数升级版】压缩突破战术剧本合成模块
        - 收益: 剧本的“战备”判断基于更可靠、更平滑的“压缩度”评分，提升了战术的准确性。
        """
        # print("        -> [压缩突破战术剧本合成模块 V1.4 融合函数升级版] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 剧本1: S+级 - 极致压缩·暴力突破 (数值化) ---
        # 战备分(昨日):
        # 将对已废弃信号的引用，升级为消费融合后的多层次分数
        vol_compression_score = self._fuse_multi_level_scores(df, 'VOL_COMPRESSION')
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

    def synthesize_cognitive_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.1 数值化升级版】顶层认知总分合成模块
        - 核心职责: 融合所有顶层的、跨领域的认知机会分与风险分。
        - 本次升级: [修复] 将对“完美风暴”信号的引用从布尔型升级为数值化评分，
                    确保最终总分能正确反映其强度。
        """
        # print("        -> [顶层认知总分合成模块 V1.1 数值化升级版] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 汇总所有S级的“机会”类认知分数 ---
        bullish_scores = [
            atomic.get('COGNITIVE_SCORE_IGNITION_RESONANCE_S', default_score).values,
            atomic.get('COGNITIVE_SCORE_BOTTOM_REVERSAL_RESONANCE_S', default_score).values,
            atomic.get('COGNITIVE_SCORE_CLASSIC_PATTERN_OPP_S', default_score).values,
            atomic.get('COGNITIVE_SCORE_BEARISH_EXHAUSTION_OPP_S', default_score).values,
            atomic.get('COGNITIVE_SCORE_CONSOLIDATION_BREAKOUT_OPP_A', default_score).values,
            atomic.get('COGNITIVE_SCORE_PERFECT_STORM_BOTTOM_S_PLUS', default_score).values,
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
        states['COGNITIVE_BEARISH_SCORE'] = cognitive_bearish_score_series
        # --- 4. 更新原子状态库 ---
        self.strategy.atomic_states.update(states)
        # print("        -> [顶层认知总分合成模块 V1.2 风险源升级版] 计算完毕。")
        return df


