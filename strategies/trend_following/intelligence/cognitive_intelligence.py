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
        【V2.0 动态加权与情景感知版】风险元融合模块
        - 核心职责: 消费配置文件中按维度组织的风险信号，生成结构化的风险态势评估。
        - 核心升级 (V2.0):
          1.  【动态加权】: 根据市场阶段(上涨初期/末期)动态调整各风险维度的权重。
          2.  【风险共振】: 当多个核心维度同时高风险时，对总风险分施加额外惩罚。
          3.  【主次融合】: 优化维度内融合逻辑，从“取最大”升级为“主要风险+次要风险*折扣”，感知风险累积。
        """
        print("        -> [风险元融合模块 V2.0 动态加权与情景感知版] 启动...") # // 更新版本号和描述
        states = {}
        p_fused_risk = get_params_block(self.strategy, 'fused_risk_scoring')
        if not get_param_value(p_fused_risk.get('enabled'), True):
            print("        -> [风险元融合模块] 已在配置中禁用，跳过计算。")
            states['COGNITIVE_FUSED_RISK_SCORE'] = pd.Series(0.0, index=df.index, dtype=np.float32)
            return states

        risk_categories = get_param_value(p_fused_risk.get('risk_categories'), {})
        
        # 加载新的参数块
        p_dynamic_weighting = get_params_block(p_fused_risk, 'dynamic_weighting_params')
        p_resonance = get_params_block(p_fused_risk, 'resonance_penalty_params')
        p_intra_fusion = get_params_block(p_fused_risk, 'intra_dimension_fusion_params')

        base_weights = get_param_value(p_dynamic_weighting.get('base_weights'), {})
        context_adjustments = get_param_value(p_dynamic_weighting.get('context_adjustments'), {})
        secondary_risk_discount = get_param_value(p_intra_fusion.get('secondary_risk_discount_factor'), 0.3)

        fused_dimension_scores = {}
        default_series = pd.Series(0.0, index=df.index, dtype=np.float32)

        # --- 1. 维度内融合：【深化升级】采用“主次融合”逻辑 ---
        print("          -> 步骤1: 执行维度内风险主次融合...")
        for category_name, signals in risk_categories.items():
            if category_name == "说明": continue
            
            category_signal_scores = []
            for signal_name, base_score in signals.items():
                if signal_name == "说明": continue
                
                atomic_score = self._get_atomic_score(df, signal_name, 0.0)
                final_signal_score = atomic_score * base_score
                final_signal_score.name = signal_name # 为Series命名，便于后续排序
                category_signal_scores.append(final_signal_score)

            if category_signal_scores:
                # 将所有信号分数合并到一个DataFrame中，便于计算主次风险
                category_df = pd.concat(category_signal_scores, axis=1)
                
                # 计算每一行的最大值（主要风险）和次大值（次要风险）
                sorted_scores = np.sort(category_df.values, axis=1)
                primary_risk = pd.Series(sorted_scores[:, -1], index=df.index)
                secondary_risk = pd.Series(sorted_scores[:, -2] if sorted_scores.shape[1] > 1 else 0, index=df.index)
                
                # 应用主次融合逻辑
                dimension_risk_score = primary_risk + secondary_risk * secondary_risk_discount
                
                fused_dimension_scores[category_name] = dimension_risk_score
                states[f'FUSED_RISK_SCORE_{category_name.upper()}'] = dimension_risk_score.astype(np.float32)
            else:
                fused_dimension_scores[category_name] = default_series.copy()

        # --- 2. 维度间融合：【深化升级】应用“动态风险加权” ---
        print("          -> 步骤2: 应用市场阶段进行动态风险加权...")
        total_fused_risk_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        
        # 获取市场阶段上下文
        is_early_stage = self.strategy.atomic_states.get('CONTEXT_TREND_STAGE_EARLY', pd.Series(False, index=df.index))
        is_late_stage = self.strategy.atomic_states.get('CONTEXT_TREND_STAGE_LATE', pd.Series(False, index=df.index))

        for category_name, weight in base_weights.items():
            if category_name in fused_dimension_scores:
                # 获取当前维度的基础权重
                current_weight = pd.Series(weight, index=df.index)
                
                # 根据市场阶段动态调整权重
                if get_param_value(p_dynamic_weighting.get('enabled'), True):
                    # 在上涨初期，调整权重
                    if "CONTEXT_TREND_STAGE_EARLY" in context_adjustments and category_name in context_adjustments["CONTEXT_TREND_STAGE_EARLY"]:
                        adjustment_factor = context_adjustments["CONTEXT_TREND_STAGE_EARLY"][category_name]
                        current_weight = current_weight.where(~is_early_stage, current_weight * adjustment_factor)

                    # 在上涨末期，调整权重
                    if "CONTEXT_TREND_STAGE_LATE" in context_adjustments and category_name in context_adjustments["CONTEXT_TREND_STAGE_LATE"]:
                        adjustment_factor = context_adjustments["CONTEXT_TREND_STAGE_LATE"][category_name]
                        current_weight = current_weight.where(~is_late_stage, current_weight * adjustment_factor)

                total_fused_risk_score += fused_dimension_scores[category_name] * current_weight

        # --- 3. 风险共振惩罚：【深化升级】对协同风险施加额外惩罚 ---
        print("          -> 步骤3: 检测风险共振并施加惩罚...")
        if get_param_value(p_resonance.get('enabled'), True):
            core_dims = get_param_value(p_resonance.get('core_risk_dimensions'), [])
            min_dims = get_param_value(p_resonance.get('min_dimensions_for_resonance'), 2)
            threshold = get_param_value(p_resonance.get('risk_score_threshold'), 150)
            penalty_multiplier = get_param_value(p_resonance.get('penalty_multiplier'), 1.2)
            # 计算有多少个核心维度的风险超过了阈值
            high_risk_dimension_count = pd.Series(0, index=df.index)
            for dim in core_dims:
                if dim in fused_dimension_scores:
                    high_risk_dimension_count += (fused_dimension_scores[dim] > threshold).astype(int)
            # 判断是否触发共振条件
            is_resonance_triggered = (high_risk_dimension_count >= min_dims)
            # 对触发共振的日子，应用惩罚乘数
            total_fused_risk_score = total_fused_risk_score.where(~is_resonance_triggered, total_fused_risk_score * penalty_multiplier)
            states['FUSED_RISK_RESONANCE_PENALTY_ACTIVE'] = is_resonance_triggered # 增加一个状态信号便于观察

        states['COGNITIVE_FUSED_RISK_SCORE'] = total_fused_risk_score.astype(np.float32)
        
        print(f"        -> [风险元融合模块 V2.0] 计算完毕，生成了 {len(states)} 个结构化风险信号。")
        return states

    def _fuse_multi_level_scores(self, df: pd.DataFrame, base_name: str, weights: Dict[str, float] = None) -> pd.Series:
        """
        【V1.1 健壮性修复版】融合S/A/B等多层置信度分数。
        - 核心修复: 修复了当只找到无等级的单一信号时，未进行reindex就返回的潜在bug，确保所有返回路径的索引都与输入df对齐。
        - :param df: 当前正在处理的数据帧，用于获取正确的索引。
        - :param base_name: 分数的基础名称 (例如 'MA_BULLISH_RESONANCE').
        - :param weights: 一个字典，定义了 'S', 'A', 'B' 等级的权重。
        - :return: 融合后的分数 (pd.Series).
        """
        if weights is None:
            weights = {'S': 1.0, 'A': 0.6, 'B': 0.3}
        
        total_score = pd.Series(0.0, index=df.index)
        total_weight = 0.0
        
        for level, weight in weights.items():
            score_name = f"SCORE_{base_name}_{level}"
            if score_name in self.strategy.atomic_states:
                score_series = self.strategy.atomic_states[score_name]
                if len(score_series) > 0:
                    # 使用reindex安全地对齐和相加
                    total_score += score_series.reindex(df.index).fillna(0.0) * weight
                    total_weight += weight
        
        if total_weight == 0:
            single_score_name = f"SCORE_{base_name}"
            if single_score_name in self.strategy.atomic_states:
                # 对单一信号也使用reindex和fillna，保证返回的Series索引正确且无NaN，这是修复的关键
                return self.strategy.atomic_states[single_score_name].reindex(df.index).fillna(0.5)
            return pd.Series(0.5, index=df.index)
            
        return (total_score / total_weight).clip(0, 1)

    def synthesize_tactical_opportunities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.2 DynamicMechanics适配版】战术机会与潜在风险合成模块
        - 核心重构 (本次修改):
          - [信号适配] 修复了对 DynamicMechanicsEngine 旧版信号的引用。
          - 将 `SCORE_FV_PURE_OFFENSIVE_MOMENTUM` 替换为新版引擎中逻辑对应的最高置信度信号 `SCORE_FV_OFFENSIVE_RESONANCE_S`。
          - 将 `SCORE_FV_CHAOTIC_EXPANSION_RISK` 替换为新版引擎中逻辑对应的最高置信度信号 `SCORE_FV_RISK_EXPANSION_S`。
        - 收益: 确保战术机会的判断基于最新、最可靠的力学信号。
        """
        print("        -> [战术机会与潜在风险合成模块 V1.2 DynamicMechanics适配版] 启动...")
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
        # --- 将旧信号替换为新版S级进攻共振信号 ---
        pure_offensive_momentum = atomic.get('SCORE_FV_OFFENSIVE_RESONANCE_S', default_score)
        trend_quality = atomic.get('COGNITIVE_SCORE_TREND_QUALITY', default_score)
        states['COGNITIVE_SCORE_OPP_PURE_MOMENTUM'] = (pure_offensive_momentum * trend_quality).astype(np.float32)
        # --- 战术 4: 混乱扩张风险 ---
        # --- 将旧信号替换为新版S级风险扩张信号 ---
        chaotic_expansion_risk = atomic.get('SCORE_FV_RISK_EXPANSION_S', default_score)
        high_level_zone_risk = atomic.get('COGNITIVE_SCORE_RISK_HIGH_LEVEL_ZONE', default_score)
        states['COGNITIVE_SCORE_RISK_CHAOTIC_EXPANSION'] = (chaotic_expansion_risk * high_level_zone_risk).astype(np.float32)
        self.strategy.atomic_states.update(states)
        print(f"        -> [战术机会与潜在风险合成模块 V1.2] 计算完毕，新增 {len(states)} 个信号。")
        return df

    def synthesize_trend_sustainability_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.3 行为信号升级版】趋势可持续性与衰竭诊断模块
        - 核心重构 (本次修改):
          - [信号适配] 将对旧的、间接的认知层反转信号的引用，升级为直接消费由 `BehavioralIntelligence`
                        V4.0 生成的、更基础、更可靠的多层次行为反转信号。
        - 收益: 显著提升了趋势可持续性判断的准确性和信号源的纯净度。
        """
        print("        -> [趋势可持续性诊断模块 V1.3 行为信号升级版] 启动...")
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
        print("        -> [趋势可持续性诊断模块 V1.3] 计算完毕。")
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
        print("        -> [盘整突破机会合成模块 V1.0] 计算完毕。")
        return df

    def synthesize_breakdown_resonance_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.4 行为信号升级版】多域崩溃共振分数合成模块
        - 核心职责: (原有注释)
        - 本次升级: [信号适配] 将对单一S级行为信号的引用，升级为消费由 `BehavioralIntelligence`
                    V4.0 生成的、包含S+级的多层次“看跌共振”融合分数。
        - 收益: 显著提升了“行为”维度在崩溃共振分析中的信号质量和可靠性。
        """
        print("        -> [多域崩溃共振分数合成模块 V1.4 行为信号升级版] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 提取七大领域的核心“崩溃”分数 ---
        # 领域1 (行为): 消费最新的多层次看跌共振信号
        behavioral_breakdown = self._fuse_multi_level_scores(df, 'BEHAVIOR_BEARISH_RESONANCE')
        # 领域2 (力学): 结构力学看跌共振分
        mechanics_breakdown = self._fuse_multi_level_scores(df, 'MECHANICS_BEARISH_RESONANCE')
        # 领域3 (筹码): 筹码看跌共振分
        chip_breakdown = self._fuse_multi_level_scores(df, 'CHIP_BEARISH_RESONANCE')
        # 领域4 (资金流): 七位一体看跌共振分
        fund_flow_breakdown = self._get_atomic_score(df, 'FF_SCORE_SEPTAFECTA_RESONANCE_DOWN_HIGH', default_score)
        # 领域5 (波动率): S级波动率崩溃分
        volatility_breakdown = self._fuse_multi_level_scores(df, 'VOL_BREAKDOWN') # 同样升级为融合信号
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
        print("        -> [多域崩溃共振分数合成模块 V1.4 行为信号升级版] 计算完毕。")
        return df

    def synthesize_trend_regime_signals(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V2.2 资金流对称增强版】趋势政权融合模块
        - 核心职责: 融合“趋势政权”、“均线共振”、“RSI动能”以及“资金流共振”四大领域的数值化评分，
                      生成更高质量的、量化的认知层趋势信号。
        - 本次升级: 【对称修复】为看跌信号增加了对应的资金流最高级别确认信号，实现了看涨与看跌信号的对称交叉验证。
        """
        # print("        -> [趋势政权融合模块 V2.2] 启动...") 
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 提取核心原子分数 ---
        # 从布尔信号升级为数值化评分
        trending_regime_score = atomic.get('SCORE_TRENDING_REGIME', default_score)
        # 更新对辅助函数的调用
        # 使用辅助函数融合S/A/B三级均线共振分数
        bullish_confluence_score = self._fuse_multi_level_scores(df, 'MA_BULLISH_RESONANCE')
        bearish_confluence_score = self._fuse_multi_level_scores(df, 'MA_BEARISH_RESONANCE')
        # 从布尔信号升级为数值化评分
        rsi_bullish_accel_score = atomic.get('SCORE_RSI_BULLISH_ACCEL', default_score)
        rsi_bearish_accel_score = atomic.get('SCORE_RSI_BEARISH_ACCEL_RISK', default_score)
        # 提取资金流最高级别确认信号
        fund_flow_ignition_score = atomic.get('FF_SCORE_SEPTAFECTA_RESONANCE_UP_HIGH', default_score)
        fund_flow_breakdown_score = atomic.get('FF_SCORE_SEPTAFECTA_RESONANCE_DOWN_HIGH', default_score)
        # --- 2. 融合生成S级认知分数 ---
        # 逻辑从布尔 AND 升级为分数相乘
        states['COGNITIVE_SCORE_TREND_REGIME_IGNITION_S'] = (
            trending_regime_score * bullish_confluence_score * rsi_bullish_accel_score * fund_flow_ignition_score
        ).astype(np.float32)
        # 为看跌信号增加资金流确认，实现对称
        states['COGNITIVE_SCORE_TREND_REGIME_BREAKDOWN_S'] = (
            trending_regime_score * bearish_confluence_score * rsi_bearish_accel_score * fund_flow_breakdown_score
        ).astype(np.float32)
        # --- 3. 更新原子状态库 ---
        self.strategy.atomic_states.update(states)
        print("        -> [趋势政权融合模块 V2.2] 计算完毕。") 
        return df

    def synthesize_volatility_breakout_signals(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V2.3 融合函数升级版】波动率突破融合模块
        - 核心职责: (原有注释)
        - 本次升级: 【逻辑深化】使用新增的 `_fuse_multi_level_scores` 辅助函数来融合S/A/B三级
                    波动率信号，生成更平滑、更鲁棒的“环境分”。
        - 收益: 突破信号的判断不再依赖于单一的S级信号，而是基于一个连续变化的“压缩度”或“扩张度”评分。
        """
        print("        -> [波动率突破融合模块 V2.3 融合函数升级版] 启动...") 
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
        chip_opportunity_score = atomic.get('CHIP_SCORE_PRIME_OPPORTUNITY_S', default_score)
        chip_bearish_score = self._fuse_multi_level_scores(df, 'CHIP_BEARISH_RESONANCE')
        
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
        print("        -> [波动率突破融合模块 V2.3 融合函数升级版] 计算完毕。") 
        return df

    def synthesize_structural_fusion_scores(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V2.0 四域融合版】结构化元信号融合模块
        - 核心升级 (本次修改):
          - [维度增强] 将“行为(Behavior)”作为第四个核心领域，与“均线(MA)”、“力学(Mechanics)”、“多时间框架(MTF)”
                        进行融合，形成真正的“四域联合作战”元信号。
        - 收益: 融合了市场的“行为表象”与底层的“结构支撑”，元信号的可靠性与实战价值大幅提升。
        """
        print("        -> [结构化元信号融合模块 V2.0 四域融合版] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        # 在信号源中全面加入 'BEHAVIOR' 维度的最高置信度信号
        signal_sources = {
            'bullish_resonance': ['SCORE_MA_BULLISH_RESONANCE_S', 'SCORE_MECHANICS_BULLISH_RESONANCE_S', 'SCORE_MTF_BULLISH_RESONANCE_S', 'SCORE_BEHAVIOR_BULLISH_RESONANCE_S_PLUS'],
            'bearish_resonance': ['SCORE_MA_BEARISH_RESONANCE_S', 'SCORE_MECHANICS_BEARISH_RESONANCE_S', 'SCORE_MTF_BEARISH_RESONANCE_S', 'SCORE_BEHAVIOR_BEARISH_RESONANCE_S_PLUS'],
            'bottom_reversal': ['SCORE_MA_BOTTOM_REVERSAL_S', 'SCORE_MECHANICS_BOTTOM_REVERSAL_S', 'SCORE_MTF_BOTTOM_REVERSAL_S', 'SCORE_BEHAVIOR_BOTTOM_REVERSAL_S'],
            'top_reversal': ['SCORE_MA_TOP_REVERSAL_S', 'SCORE_MECHANICS_TOP_REVERSAL_S', 'SCORE_MTF_TOP_REVERSAL_S', 'SCORE_BEHAVIOR_TOP_REVERSAL_S']
        }
        all_required_signals = [sig for group in signal_sources.values() for sig in group]
        # 检查时，允许S+信号不存在，增加兼容性
        signals_to_check = [s for s in all_required_signals if '_S_PLUS' not in s]
        if any(s not in atomic for s in signals_to_check):
            print("          -> [警告] 结构化元信号融合缺少核心上游S级分数，模块已跳过！")
            return df
        default_series = pd.Series(0.0, index=df.index, dtype=np.float32)
        def fuse_scores(source_keys: list) -> pd.Series:
            # 使用 _get_atomic_score 保证即使信号不存在也不会报错
            scores_to_fuse = [self._get_atomic_score(df, key, 0.0) for key in source_keys]
            fused_values = np.prod(np.array([s.values for s in scores_to_fuse]), axis=0)
            return pd.Series(fused_values, index=df.index, dtype=np.float32)
        states['COGNITIVE_FUSION_BULLISH_RESONANCE_S'] = fuse_scores(signal_sources['bullish_resonance'])
        states['COGNITIVE_FUSION_BEARISH_RESONANCE_S'] = fuse_scores(signal_sources['bearish_resonance'])
        states['COGNITIVE_FUSION_BOTTOM_REVERSAL_S'] = fuse_scores(signal_sources['bottom_reversal'])
        states['COGNITIVE_FUSION_TOP_REVERSAL_S'] = fuse_scores(signal_sources['top_reversal'])
        self.strategy.atomic_states.update(states)
        print("        -> [结构化元信号融合模块 V2.0 四域融合版] 计算完毕。")
        return df

    def synthesize_ultimate_confirmation_scores(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V1.0 新增 & 逻辑迁移】终极确认融合模块
        - 核心职责: 承接原 StructuralIntelligence 中的 `diagnose_ultimate_confirmation_scores` 逻辑。
                      寻找“连续性共振元信号”与“离散性模式信号”同时发生的最高置信度信号。
        - 收益: 在认知层完成最高级别的“完美风暴”信号融合。
        """
        # print("        -> [终极确认融合模块 V1.0] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        required_fusion_signals = [
            'COGNITIVE_FUSION_BULLISH_RESONANCE_S', 'COGNITIVE_FUSION_BEARISH_RESONANCE_S',
            'COGNITIVE_FUSION_BOTTOM_REVERSAL_S', 'COGNITIVE_FUSION_TOP_REVERSAL_S'
        ]
        required_pattern_signals = [
            'SCORE_PATTERN_BULLISH_RESONANCE_S', 'SCORE_PATTERN_BEARISH_RESONANCE_S',
            'SCORE_PATTERN_BOTTOM_REVERSAL_S', 'SCORE_PATTERN_TOP_REVERSAL_S'
        ]
        all_required_signals = required_fusion_signals + required_pattern_signals
        if any(s not in atomic for s in all_required_signals):
            print("          -> [警告] 终极确认融合缺少核心上游S级分数，模块已跳过！")
            return df
        default_series = pd.Series(0.0, index=df.index, dtype=np.float32)
        fusion_bullish = atomic.get('COGNITIVE_FUSION_BULLISH_RESONANCE_S', default_series)
        fusion_bearish = atomic.get('COGNITIVE_FUSION_BEARISH_RESONANCE_S', default_series)
        fusion_bottom = atomic.get('COGNITIVE_FUSION_BOTTOM_REVERSAL_S', default_series)
        fusion_top = atomic.get('COGNITIVE_FUSION_TOP_REVERSAL_S', default_series)
        pattern_bullish = atomic.get('SCORE_PATTERN_BULLISH_RESONANCE_S', default_series)
        pattern_bearish = atomic.get('SCORE_PATTERN_BEARISH_RESONANCE_S', default_series)
        pattern_bottom = atomic.get('SCORE_PATTERN_BOTTOM_REVERSAL_S', default_series)
        pattern_top = atomic.get('SCORE_PATTERN_TOP_REVERSAL_S', default_series)
        states['COGNITIVE_ULTIMATE_BULLISH_CONFIRMATION_S'] = (fusion_bullish * pattern_bullish).astype(np.float32)
        states['COGNITIVE_ULTIMATE_BEARISH_CONFIRMATION_S'] = (fusion_bearish * pattern_bearish).astype(np.float32)
        states['COGNITIVE_ULTIMATE_BOTTOM_CONFIRMATION_S'] = (fusion_bottom * pattern_bottom).astype(np.float32)
        states['COGNITIVE_ULTIMATE_TOP_CONFIRMATION_S'] = (fusion_top * pattern_top).astype(np.float32)
        self.strategy.atomic_states.update(states)
        print("        -> [终极确认融合模块 V1.0] 计算完毕。")
        return df

    def synthesize_ignition_resonance_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.4 行为维度增强版】多域点火共振分数合成模块
        - 核心职责: (原有注释)
        - 本次升级: [维度增强] 新增了对“行为”维度的交叉验证，消费由 `BehavioralIntelligence`
                    V4.0 生成的、包含S+级的多层次“看涨共振”融合分数。
        - 收益: 补全了点火信号的关键一环，确保了只有在价量行为健康时，才能触发最高级别的点火共振。
        """
        print("        -> [多域点火共振分数合成模块 V1.4 行为维度增强版] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 提取八大领域的核心“点火”分数 ---
        # 领域1 (行为): 消费最新的多层次看涨共振信号
        behavioral_ignition = self._fuse_multi_level_scores(df, 'BEHAVIOR_BULLISH_RESONANCE')
        # 领域2 (结构): 蓄势突破机会分
        structural_breakout = atomic.get('SCORE_STRUCTURAL_ACCUMULATION_BREAKOUT_S', default_score)
        # 领域3 (力学): 整体力学健康度元分数
        mechanics_ignition = atomic.get('SCORE_DYN_OVERALL_BULLISH_MOMENTUM_S', default_score)
        # 领域4 (筹码): 黄金筹码机会元分数
        chip_opportunity = atomic.get('CHIP_SCORE_PRIME_OPPORTUNITY_S', default_score)
        # 领域5 (资金流): 七位一体看涨共振分
        fund_flow_ignition = atomic.get('FF_SCORE_SEPTAFECTA_RESONANCE_UP_HIGH', default_score)
        # 领域6 (波动率): S级波动率突破分
        volatility_breakout = self._fuse_multi_level_scores(df, 'VOL_BREAKOUT') # 同样升级为融合信号
        # 领域7 (终极结构): 最高级别确认分
        structural_confirmation = atomic.get('COGNITIVE_ULTIMATE_BULLISH_CONFIRMATION_S', default_score)
        # 领域8 (ATR波幅): ATR扩张点火机会分
        atr_ignition = self._get_atomic_score(df, 'SCORE_ATR_EXPANSION_IGNITION_OPP', default_score)
        # --- 2. 交叉验证：生成“多域点火共振”元分数 ---
        # 将新增的行为维度加入融合
        ignition_resonance_score = (
            behavioral_ignition * structural_breakout * mechanics_ignition *
            chip_opportunity * fund_flow_ignition * volatility_breakout *
            structural_confirmation * atr_ignition
        ).astype(np.float32)
        states['COGNITIVE_SCORE_IGNITION_RESONANCE_S'] = ignition_resonance_score
        # --- 3. 更新原子状态库 ---
        self.strategy.atomic_states.update(states)
        print("        -> [多域点火共振分数合成模块 V1.4 行为维度增强版] 计算完毕。")
        return df

    def synthesize_reversal_resonance_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V2.1 行为信号升级版】多域反转共振分数合成模块
        - 核心重构 (本次修改):
          - [信号适配] 将行为触发器从旧的、特定的信号，升级为消费由 `BehavioralIntelligence`
                        V4.0 生成的、更通用的多层次“顶部/底部反转”融合分数。
        - 收益: 使得反转共振的判断逻辑更统一，信号源更可靠。
        """
        print("        -> [多域反转共振分数合成模块 V2.1 行为信号升级版] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        p = get_params_block(self.strategy, 'reversal_resonance_params', {})
        bottom_weights = get_param_value(p.get('bottom_trigger_weights'), {})
        top_weights = get_param_value(p.get('top_trigger_weights'), {})
        
        # --- 1. 合成“底部反转共振”分数 ---
        oversold_context = np.maximum(atomic.get('SCORE_RSI_OVERSOLD_EXTENT', default_score).values, atomic.get('SCORE_BIAS_OVERSOLD_EXTENT', default_score).values)
        mechanics_bottom_setup = self._fuse_multi_level_scores(df, 'DYN_BOTTOM_REVERSAL')
        chip_bottom_setup = self._fuse_multi_level_scores(df, 'BOTTOM_REVERSAL')
        ma_bottom_setup = self._fuse_multi_level_scores(df, 'EMA_BOTTOM_REVERSAL')
        reversal_setup_score = (mechanics_bottom_setup * chip_bottom_setup * ma_bottom_setup)
        
        pattern_trigger = self._fuse_multi_level_scores(df, 'PATTERN_BOTTOM_REVERSAL')
        foundation_trigger = self._fuse_multi_level_scores(df, 'FOUNDATION_BOTTOM_REVERSAL')
        # 消费最新的多层次行为反转信号
        behavioral_trigger = self._fuse_multi_level_scores(df, 'BEHAVIOR_BOTTOM_REVERSAL')
        
        trigger_scores = {'pattern': pattern_trigger, 'foundation': foundation_trigger, 'behavioral': behavioral_trigger}
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
        chip_top_setup = self._fuse_multi_level_scores(df, 'TOP_REVERSAL')
        ma_top_setup = self._fuse_multi_level_scores(df, 'EMA_TOP_REVERSAL')
        top_setup_score = (mechanics_top_setup * chip_top_setup * ma_top_setup)

        pattern_top_trigger = self._fuse_multi_level_scores(df, 'PATTERN_TOP_REVERSAL')
        foundation_top_trigger = self._fuse_multi_level_scores(df, 'FOUNDATION_TOP_REVERSAL')
        # 消费最新的多层次行为反转信号
        behavioral_top_trigger = self._fuse_multi_level_scores(df, 'BEHAVIOR_TOP_REVERSAL')

        top_trigger_scores = {'pattern': pattern_top_trigger, 'foundation': foundation_top_trigger, 'behavioral': behavioral_top_trigger}
        final_top_trigger_score = pd.Series(0.0, index=df.index)
        total_weight_top = sum(top_weights.values())
        if total_weight_top > 0:
            for key, weight in top_weights.items():
                final_top_trigger_score += top_trigger_scores.get(key, default_score) * weight
            final_top_trigger_score /= total_weight_top

        top_reversal_score = pd.Series(overbought_context, index=df.index) * top_setup_score * final_top_trigger_score
        states['COGNITIVE_SCORE_TOP_REVERSAL_RESONANCE_S'] = top_reversal_score.astype(np.float32)
        
        self.strategy.atomic_states.update(states)
        print("        -> [多域反转共振分数合成模块 V2.1] 计算完毕。")
        return df

    def synthesize_divergence_risks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V2.2 行为信号升级版】多维背离风险融合模块
        - 核心重构 (本次修改):
          - [信号适配] 将“引擎背离”的判断依据，从旧的、特定的信号，升级为消费
                        `BehavioralIntelligence` V4.0产出的、更通用的“顶部反转”融合分数。
        - 收益: 统一了对顶部风险的判断标准，信号源更可靠。
        """
        print("        -> [多维背离风险融合模块 V2.2 行为信号升级版] 启动...")
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
        print("        -> [多维背离风险融合模块 V2.2] 计算完毕。")
        return df

    def synthesize_market_engine_states(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V2.1 行为信号升级版】市场引擎状态融合模块
        - 核心重构 (本次修改):
          - [信号适配] 将“引擎失速”的判断依据，从旧的、特定的信号，升级为消费
                        `BehavioralIntelligence` V4.0产出的、更通用的“顶部反转”融合分数。
        - 收益: 统一了对引擎失效的判断标准，信号源更可靠。
        """
        print("        -> [市场引擎状态融合模块 V2.1 行为信号升级版] 启动...") 
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
        print("        -> [市场引擎状态融合模块 V2.1] 计算完毕。") 
        return df

    def synthesize_pullback_states(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V2.2 信号融合增强版】认知层回踩状态合成模块
        - 核心职责: 融合行为层与筹码层的【分数】，生成高质量的、量化的“回踩性质”认知分数。
        - 收益: 实现了对回踩“健康度”和“打压强度”的连续度量，为下游战术提供更精细的输入。
        - 本次升级: 增强了对“健康回踩”中筹码稳定性的评估，从消费单一S级信号升级为
                    融合S/A/B三级置信度分数，评估更鲁棒。
        """
        # print("        -> [认知层回踩状态合成模块 V2.2] 启动...") 
        states = {}
        # --- 1. 定义通用回踩条件和环境分数 ---
        is_pullback_day = (df['pct_change_D'] < 0).astype(float)
        # 更新对辅助函数的调用
        constructive_context_score = self._get_atomic_score(df, 'COGNITIVE_SCORE_TREND_QUALITY', 0.0)
        # --- 2. 合成“健康回踩”分数 (Healthy Pullback Score) ---
        # 所有判断升级为0-1的连续分数
        gentle_drop_score = (1 - (df['pct_change_D'].abs() / 0.05)).clip(0, 1) # 跌幅越小，分数越高
        shrinking_volume_score = self._get_atomic_score(df, 'SCORE_VOL_WEAKENING_DROP', 0.0) 
        winner_holding_tight_score = 1.0 - self._fuse_multi_level_scores(df, 'PROFIT_TAKING_TOP_REVERSAL')
        chip_stable_score = 1.0 - self._fuse_multi_level_scores(df, 'CHIP_BEARISH_RESONANCE')
        healthy_pullback_score = (
            is_pullback_day * constructive_context_score *
            gentle_drop_score * shrinking_volume_score *
            winner_holding_tight_score * chip_stable_score
        )
        states['COGNITIVE_SCORE_PULLBACK_HEALTHY_S'] = healthy_pullback_score.astype(np.float32)
        # --- 3. 合成“打压式回踩”分数 (Suppressive Pullback Score) ---
        # 所有判断升级为0-1的连续分数
        significant_drop_score = (df['pct_change_D'].abs() / 0.07).clip(0, 1) # 跌幅越大，分数越高
        weights = {'S': 1.0, 'A': 0.6, 'B': 0.3}
        total_weight = sum(weights.values())
        capitulation_score_b = self._get_atomic_score(df, 'SCORE_CAPITULATION_BOTTOM_REVERSAL_B', 0.0)
        capitulation_score_a = self._get_atomic_score(df, 'SCORE_CAPITULATION_BOTTOM_REVERSAL_A', 0.0)
        capitulation_score_s = self._get_atomic_score(df, 'SCORE_CAPITULATION_BOTTOM_RESONANCE_S', 0.0)
        panic_selling_score = (
            capitulation_score_s * weights['S'] +
            capitulation_score_a * weights['A'] +
            capitulation_score_b * weights['B']
        ) / total_weight
        suppressive_pullback_score = (
            is_pullback_day * constructive_context_score *
            significant_drop_score * panic_selling_score * winner_holding_tight_score
        )
        states['COGNITIVE_SCORE_PULLBACK_SUPPRESSIVE_S'] = suppressive_pullback_score.astype(np.float32)
        # --- 4. 更新原子状态库 ---
        self.strategy.atomic_states.update(states)
        print("        -> [认知层回踩状态合成模块 V2.2] 计算完毕。") 
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
        print("        -> [认知层持仓风险合成模块 V1.1 数值化升级版] 计算完毕。") 
        return df

    def synthesize_contextual_zone_scores(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V2.5 终极风险增强版】战场上下文评分模块
        - 核心职责: 将多个维度的原子风险【分数】融合成一个顶层的“高位危险区”综合风险分。
        - 核心升级: 修复了所有失效的上游信号引用，并引入了更多维度的风险源，
                      使得危险区的评估更加全面和准确。
        - 本次升级: 【维度增强】新增了对“多域崩溃共振”、“终极看跌确认”、“价格偏离”、“结构断层”
                    等多个顶级风险信号的融合，确保危险区评分能捕捉到最致命的风险。
        """
        # print("        -> [战场上下文评分模块 V2.5 终极风险增强版] 启动...")
        # --- 1. 量化“高位危险区”得分 ---
        # 修复并扩展了用于融合的风险信号源
        # 更新对辅助函数的调用
        risk_scores = {
            'bias': self._get_atomic_score(df, 'SCORE_BIAS_OVERBOUGHT_EXTENT', 0.0),
            # 融合动能衰竭与利润衰竭两大维度
            'exhaustion': np.maximum(
                self._get_atomic_score(df, 'SCORE_RISK_MOMENTUM_EXHAUSTION', 0.0).values,
                self._get_atomic_score(df, 'SCORE_RISK_PROFIT_EXHAUSTION_S', 0.0).values
            ),
            'chip_decay': self._get_atomic_score(df, 'SCORE_RISK_CHIP_STRUCTURE_DECAY', 0.0),
            'churn': self._get_atomic_score(df, 'SCORE_RISK_DYNAMIC_DECEPTIVE_CHURN', 0.0),
            'deceptive_rally': self._get_atomic_score(df, 'CHIP_SCORE_FUSED_DECEPTIVE_RALLY', 0.0),
            'chip_top_reversal': self._fuse_multi_level_scores(df, 'CHIP_TOP_REVERSAL'), 
            'structural_weakness': self._get_atomic_score(df, 'SCORE_MTF_STRUCTURAL_WEAKNESS_RISK_S', 0.0),
            'engine_stalling': self._get_atomic_score(df, 'SCORE_BEHAVIOR_ENGINE_STALLING_RISK_S', 0.0),
            'volume_spike_down': self._get_atomic_score(df, 'SCORE_VOL_PRICE_PANIC_DOWN_RISK', 0.0),
            'chip_divergence': self._fuse_multi_level_scores(df, 'CHIP_BEARISH_RESONANCE'),
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
        risk_df = pd.DataFrame(risk_scores)
        # 使用最大值作为最终风险分，代表最严重的风险维度
        high_level_zone_score = risk_df.max(axis=1)
        df['COGNITIVE_SCORE_RISK_HIGH_LEVEL_ZONE'] = high_level_zone_score
        self.strategy.atomic_states['COGNITIVE_SCORE_RISK_HIGH_LEVEL_ZONE'] = df['COGNITIVE_SCORE_RISK_HIGH_LEVEL_ZONE']
        # 基于新的分数，生成兼容性的布尔信号
        # 当综合风险分超过其85%分位数时，定义为高风险区
        high_risk_threshold = high_level_zone_score.rolling(120).quantile(0.85)
        is_in_high_level_zone = high_level_zone_score > high_risk_threshold
        self.strategy.atomic_states['CONTEXT_RISK_HIGH_LEVEL_ZONE'] = is_in_high_level_zone
        print("        -> [战场上下文评分模块 V2.5 终极风险增强版] 计算完毕。")
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
        print("        -> [顶层机会风险评分模块 V2.0 职责净化版] 计算完毕。")
        return df

    def synthesize_trend_quality_score(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V1.5 行为信号升级版】趋势质量融合评分模块
        - 核心职责: (原有注释)
        - 本次升级: [信号适配] 将“行为健康度”的评估，从基于旧的“引擎失速”信号，
                    升级为基于最新的多层次“顶部反转”融合分数。
        - 收益: 对趋势质量的判断更精确，一个没有顶部反转风险的趋势才是高质量的趋势。
        """
        print("        -> [趋势质量融合评分模块 V1.5 行为信号升级版] 启动...") 
        # --- 1. 提取各领域的核心健康度评分 ---
        # 行为健康度: 基于最新的多层次“顶部反转”融合分数。反转风险低则健康。
        behavior_risk_score = self._fuse_multi_level_scores(df, 'BEHAVIOR_TOP_REVERSAL')
        behavior_health_score = 1.0 - behavior_risk_score
        # 筹码健康度
        chip_health_score = self._fuse_multi_level_scores(df, 'CHIP_BULLISH_RESONANCE') 
        # 资金流健康度
        fund_flow_health_score = self._get_atomic_score(df, 'FF_SCORE_SEPTAFECTA_RESONANCE_UP_HIGH')
        # 结构健康度
        structural_health_score = self._get_atomic_score(df, 'SCORE_MA_HEALTH') 
        # 市场情绪健康度
        market_health_score = self._get_atomic_score(df, 'SCORE_MKT_HEALTH_S') 
        # 力学健康度
        mechanics_health_score = self._get_atomic_score(df, 'SCORE_DYN_OVERALL_BULLISH_MOMENTUM_S') 
        # 市场政权健康度
        regime_health_score = self._get_atomic_score(df, 'SCORE_TRENDING_REGIME')
        # --- 2. 定义各维度权重 ---
        p = get_params_block(self.strategy, 'trend_quality_params', {})
        weights = {
            'behavior': get_param_value(p.get('behavior_weight'), 0.20), # 提升行为权重
            'chip': get_param_value(p.get('chip_weight'), 0.20),
            'fund_flow': get_param_value(p.get('fund_flow_weight'), 0.15), 
            'structural': get_param_value(p.get('structural_weight'), 0.15), 
            'market': get_param_value(p.get('market_weight'), 0.10), 
            'mechanics': get_param_value(p.get('mechanics_weight'), 0.10), # 调整权重
            'regime': get_param_value(p.get('regime_weight'), 0.10)
        }
        # --- 3. 加权融合生成最终的趋势质量分 ---
        trend_quality_score = (
            behavior_health_score * weights['behavior'] +
            chip_health_score * weights['chip'] +
            fund_flow_health_score * weights['fund_flow'] +
            structural_health_score * weights['structural'] + 
            market_health_score * weights['market'] +
            mechanics_health_score * weights['mechanics'] +
            regime_health_score * weights['regime']
        )
        df['COGNITIVE_SCORE_TREND_QUALITY'] = trend_quality_score
        self.strategy.atomic_states['COGNITIVE_SCORE_TREND_QUALITY'] = df['COGNITIVE_SCORE_TREND_QUALITY']
        print("        -> [趋势质量融合评分模块 V1.5 行为信号升级版] 计算完毕。") 
        return df

    def diagnose_trend_stage_score(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V401.13 配置化改造版】趋势阶段评分模块
        - 核心修正: 修复了VPA风险融合逻辑，确保所有相关风险信号都被正确消费。
        - 核心改造: 移除了硬编码的风险权重，改为从JSON配置文件中动态加载，并增加了模块开关。
        """
        # print("        -> [趋势阶段评分模块 V401.13 配置化改造版] 启动...") # 代码更新版本号和描述
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        p_trend_stage_scoring = get_params_block(self.strategy, 'trend_stage_scoring_params')
        if not get_param_value(p_trend_stage_scoring.get('enabled'), True):
            # 如果模块被禁用，则返回0分，避免影响下游逻辑
            states['CONTEXT_TREND_LATE_STAGE_SCORE'] = pd.Series(0, index=df.index, dtype=int)
            states['CONTEXT_TREND_STAGE_LATE'] = pd.Series(False, index=df.index)
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
        # --- 2. 计算“上涨末期”的量化分数 (Late Stage Score) ---
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
        for name, weight in signal_definitions.items():
            if name == "vpa_risk_score_series":
                risk_dimension_scores.append(vpa_risk_score_series * weight)
            elif name in ["CHIP_TOP_REVERSAL", "STRUCTURE_BEARISH_RESONANCE", "MTF_BEARISH_RESONANCE", "MECHANICS_TOP_REVERSAL", "PATTERN_TOP_REVERSAL"]:
                risk_dimension_scores.append(self._fuse_multi_level_scores(df, name) * weight)
            else:
                risk_dimension_scores.append(self._get_atomic_score(df, name, 0.0) * weight)
        score_components = [s.to_numpy(dtype=np.float32) for s in risk_dimension_scores]
        late_stage_score_arr = np.add.reduce(score_components)
        late_stage_score_arr = np.nan_to_num(late_stage_score_arr, nan=0.0, posinf=0.0, neginf=0.0)
        late_stage_score_arr_int = late_stage_score_arr.astype(int)
        late_stage_score = pd.Series(late_stage_score_arr_int, index=df.index, dtype=int)
        states['CONTEXT_TREND_LATE_STAGE_SCORE'] = late_stage_score
        states['CONTEXT_TREND_STAGE_LATE'] = late_stage_score >= 160
        print("        -> [趋势阶段评分模块 V401.13 配置化改造版] 计算完毕。")
        return states

    def diagnose_market_structure_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V279.0 危险感知增强版】 - 联合作战司令部
        - 核心升级 (本次修改):
          - [维度增强] 在“顶部危险”的诊断中，加入了对 `BehavioralIntelligence` V4.0
                        产出的“行为看跌共振”融合分数的判断。
        - 收益: 使得对顶部结构的危险性评估更加全面，能更早地捕捉到价量行为层面的恶化迹象。
        """
        print("        -> [联合作战司令部 V279.0 危险感知增强版] 启动，正在分析战场核心结构...")
        structure_states = {}
        default_series = pd.Series(False, index=df.index)
        atomic = self.strategy.atomic_states
        ma_bullish_score = self._fuse_multi_level_scores(df, 'MA_BULLISH_RESONANCE')
        dyn_trend_healthy_score = self._get_atomic_score(df, 'SCORE_DYN_BULLISH_RESONANCE_S', 0.0)
        chip_concentrating_score = self._fuse_multi_level_scores(df, 'CHIP_BULLISH_RESONANCE')
        is_price_above_long_ma = (df['close_D'] > df['EMA_55_D']).astype(float)
        ma_bearish_score = self._fuse_multi_level_scores(df, 'MA_BEARISH_RESONANCE')
        dyn_trend_weakening_score = self._get_atomic_score(df, 'SCORE_MTF_TOP_DIVERGENCE_S', 0.0)
        chip_diverging_score = self._fuse_multi_level_scores(df, 'CHIP_BEARISH_RESONANCE')
        risk_chip_failure_score = self._fuse_multi_level_scores(df, 'STRUCTURE_BEARISH_RESONANCE')
        risk_late_stage_score_raw = self.strategy.atomic_states.get('CONTEXT_TREND_LATE_STAGE_SCORE', pd.Series(0, index=df.index))
        risk_late_stage_score = (risk_late_stage_score_raw / 600).clip(0, 1)
        prime_opportunity_score = self._get_atomic_score(df, 'CHIP_SCORE_PRIME_OPPORTUNITY_S', 0.0)
        base_uptrend_score = (ma_bullish_score * dyn_trend_healthy_score * chip_concentrating_score * is_price_above_long_ma)
        structure_states['SCORE_STRUCTURE_MAIN_UPTREND_WAVE_S'] = base_uptrend_score.astype(np.float32)
        structure_states['SCORE_STRUCTURE_FORTRESS_UPTREND_S_PLUS'] = (base_uptrend_score * prime_opportunity_score).astype(np.float32)
        structure_states['STRUCTURE_MAIN_UPTREND_WAVE_S'] = structure_states['SCORE_STRUCTURE_MAIN_UPTREND_WAVE_S'] > 0.4
        structure_states['STRUCTURE_FORTRESS_UPTREND_S_PLUS'] = structure_states['SCORE_STRUCTURE_FORTRESS_UPTREND_S_PLUS'] > 0.5
        fused_squeeze_score = self._fuse_multi_level_scores(df, 'VOL_COMPRESSION')
        has_energy_advantage_score = self._get_atomic_score(df, 'SCORE_MECHANICS_BULLISH_RESONANCE_S', 0.0)
        prime_setup_score = prime_opportunity_score * fused_squeeze_score * has_energy_advantage_score
        structure_states['SCORE_SETUP_PRIME_STRUCTURE_S'] = prime_setup_score.astype(np.float32)
        structure_states['SETUP_PRIME_STRUCTURE_S'] = prime_setup_score > 0.6
        is_recent_reversal = atomic.get('BEHAVIOR_CONTEXT_RECENT_REVERSAL_SIGNAL', default_series)
        is_ma_short_slope_positive = df.get('SLOPE_5_EMA_5_D', pd.Series(0, index=df.index)) > 0
        structure_states['STRUCTURE_EARLY_REVERSAL_B'] = is_recent_reversal & is_ma_short_slope_positive
        confirmed_distribution_score = self._get_atomic_score(df, 'SCORE_S_PLUS_CONFIRMED_DISTRIBUTION', 0.0)
        mechanics_top_reversal_score = self._fuse_multi_level_scores(df, 'MECHANICS_TOP_REVERSAL')
        pattern_top_reversal_score = self._fuse_multi_level_scores(df, 'PATTERN_TOP_REVERSAL')
        # 引入最新的行为看跌共振信号
        behavioral_bearish_score = self._fuse_multi_level_scores(df, 'BEHAVIOR_BEARISH_RESONANCE')
        # 将新的行为风险加入“顶部危险”的判断
        topping_danger_score = np.maximum.reduce([risk_chip_failure_score.values, risk_late_stage_score.values, confirmed_distribution_score.values, mechanics_top_reversal_score.values, pattern_top_reversal_score.values, behavioral_bearish_score.values])
        structure_states['SCORE_STRUCTURE_TOPPING_DANGER_S'] = pd.Series(topping_danger_score, index=df.index, dtype=np.float32)
        bearish_channel_score = ma_bearish_score * dyn_trend_weakening_score * chip_diverging_score
        structure_states['SCORE_STRUCTURE_BEARISH_CHANNEL_F'] = bearish_channel_score.astype(np.float32)
        structure_states['STRUCTURE_BEARISH_CHANNEL_F'] = bearish_channel_score > 0.5
        print("        -> [联合作战司令部 V279.0] 核心战局定义升级完成。")
        return structure_states

    def synthesize_topping_behaviors(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V332.0 风险归因修正版】顶部行为合成模块
        - 核心重构 (本次修改):
          - [风险归因] 将“拉升背离”的判断依据，从旧的、混杂的筹码信号，修正为纯粹消费
                        `BehavioralIntelligence` V4.0产出的“顶部反转”与“看跌共振”融合分数。
        - 收益: 模块职责更清晰，风险归因更准确，信号逻辑与方法名完全匹配。
        """
        print("        -> [顶部行为合成模块 V332.0 风险归因修正版] 启动...") 
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
        
        concentrating_score = self._fuse_multi_level_scores(df, 'CHIP_BULLISH_RESONANCE')
        is_concentrating = concentrating_score > 0.6
        is_in_danger_zone = atomic.get('CONTEXT_RISK_HIGH_LEVEL_ZONE', default_series)
        states['RALLY_STATE_HEALTHY_LOCKED'] = is_rallying & is_concentrating & ~is_in_danger_zone
        return states

    def synthesize_chip_fund_flow_synergy(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 数值化重构版】筹码与资金流协同合成模块
        - 核心重构: 废除所有已失效的布尔信号，全面升级为消费现代化的数值化融合分数。
        - 核心逻辑: 将筹码的“共振分”与资金流的“七位一体共振分”相乘，生成更高置信度的协同信号。
        - 收益: 信号质量与鲁棒性大幅提升，能更精确地量化“吸筹”与“派发”的协同强度。
        """
        # print("        -> [筹码与资金流协同合成模块 V2.0 数值化重构版] 启动...")
        states = {}
        # --- 1. 获取核心筹码与资金流的数值化融合分数 ---
        # 更新对辅助函数的调用
        # 使用 fuse_multi_level_scores 融合多级置信度，获得更平滑的筹码分数
        chip_bullish_score = self._fuse_multi_level_scores(df, 'CHIP_BULLISH_RESONANCE')
        chip_bearish_score = self._fuse_multi_level_scores(df, 'CHIP_BEARISH_RESONANCE')
        # 获取资金流最高质量的“七位一体”融合分数
        fund_flow_bullish_score = self._get_atomic_score(df, 'FF_SCORE_SEPTAFECTA_RESONANCE_UP_HIGH', 0.5)
        fund_flow_bearish_score = self._get_atomic_score(df, 'FF_SCORE_SEPTAFECTA_RESONANCE_DOWN_HIGH', 0.5)
        # --- 2. 交叉验证：生成协同吸筹分数 ---
        # 逻辑: 协同吸筹强度 = 筹码看涨分 * 资金流看涨分
        synergy_accumulation_score = (chip_bullish_score * fund_flow_bullish_score).astype(np.float32)
        states['COGNITIVE_SCORE_CHIP_FUND_FLOW_ACCUMULATION_S'] = synergy_accumulation_score
        # --- 3. 交叉验证：生成协同派发风险分数 ---
        # 逻辑: 协同派发风险 = 筹码看跌分 * 资金流看跌分
        synergy_distribution_score = (chip_bearish_score * fund_flow_bearish_score).astype(np.float32)
        states['COGNITIVE_SCORE_CHIP_FUND_FLOW_DISTRIBUTION_S'] = synergy_distribution_score
        # --- 4. 为了兼容性，可以基于分数生成旧的布尔信号 ---
        states['CHIP_FUND_FLOW_ACCUMULATION_CONFIRMED_A'] = synergy_accumulation_score > 0.6
        states['CHIP_FUND_FLOW_ACCUMULATION_STRONG_S'] = synergy_accumulation_score > 0.8
        states['RISK_CHIP_FUND_FLOW_DISTRIBUTION_A'] = synergy_distribution_score > 0.7
        print("        -> [筹码与资金流协同合成模块 V2.0 数值化重构版] 合成完毕。")
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
        print("        -> [完美风暴信号合成模块 V1.1 数值化升级版] 计算完毕。") 
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
        print("        -> [S+战法诊断] 正在扫描“锁仓再集中(V2.3 王牌重铸数值化增强版)”...") 
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
        【V2.3 信号升级与容错巡航版】锁筹拉升S级战法诊断模块
        - 核心升级: 引入“容错机制”。允许“筹码持续集中”的巡航条件出现一次短暂中断，
                    如果连续两天中断，才终止巡航。
        - 信号修复: 修复了对 CHIP_DYN_OBJECTIVE_DIVERGING 和 CHIP_DYN_CONCENTRATING
                    等废弃信号的引用，升级为消费多级置信度融合分数。
        """
        # print("        -> [S级战法诊断] 正在扫描“锁筹拉升(V2.3 信号升级与容错巡航版)”...") 
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
        # 更新对辅助函数的调用
        # 使用融合后的数值分代替废弃的 CHIP_DYN_OBJECTIVE_DIVERGING
        is_diverging = self._fuse_multi_level_scores(df, 'CHIP_BEARISH_RESONANCE') > divergence_threshold
        is_late_stage = atomic.get('CONTEXT_TREND_STAGE_LATE', default_series)
        is_ma_broken = self._get_atomic_score(df, 'SCORE_MA_HEALTH', 1.0) < 0.4 # 修复失效的 MA_STATE_STABLE_BULLISH 信号，升级为基于均线健康分的判断
        is_health_stalling = atomic.get('COGNITIVE_HOLD_RISK_HEALTH_STALLING', default_series)
        hard_termination_condition = is_diverging | is_late_stage | is_ma_broken
        if terminate_on_stalling:
            hard_termination_condition |= is_health_stalling
        # --- 4. 定义“软性巡航”条件  ---
        # 使用融合后的数值分代替废弃的 CHIP_DYN_CONCENTRATING
        is_cruise_condition_met = self._fuse_multi_level_scores(df, 'CHIP_BULLISH_RESONANCE') > concentration_threshold if require_concentration else pd.Series(True, index=df.index)
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
        if final_tactic_signal.any():
            print(f"          -> [S级持仓确认] 侦测到 {final_tactic_signal.sum()} 天处于“健康锁筹拉升”巡航状态！")
        return states

    def synthesize_dynamic_offense_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 数值化重构版】协同进攻动能合成模块
        - 核心重构: 废除了所有已失效的原子信号，重构为一个基于多领域核心看涨分数
                      交叉验证的数值化评分体系。
        - 核心逻辑: 1. 将力学、筹码、结构、资金流、市场状态五大领域的核心S级看涨分数相乘，
                         生成一个顶层的“协同进攻”元分数。
                      2. 使用“趋势阶段”上下文对该信号进行战略过滤，防止在上涨末期追高。
        - 收益: 创造了一个更高质量、更安全的A级动能信号。
        """
        # print("        -> [协同进攻动能合成模块 V2.0] 启动...") 
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)

        # 更新对辅助函数的调用
        # 1. 获取五大领域的核心S级看涨分数
        mechanics_score = atomic.get('SCORE_DYN_BULLISH_RESONANCE_S', default_score)
        chip_score = self._fuse_multi_level_scores(df, 'CHIP_BULLISH_RESONANCE', {'S': 1.0, 'A': 0.0, 'B': 0.0}) # 仅使用S级
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
        print("        -> [终极战法合成模块 V2.4 融合函数升级版] 启动...")
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
        if final_tactic_s_plus_plus.any():
            print(f"          -> [S++级王牌战法] 侦测到 {final_tactic_s_plus_plus.sum()} 次“终极结构突破”机会！")
        if final_tactic_s_plus.any():
            print(f"          -> [S+级王牌战法] 侦测到 {final_tactic_s_plus.sum()} 次“次级结构突破”机会！")
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
        print("        -> [压缩突破战术剧本合成模块 V1.4 融合函数升级版] 启动...")
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
        print("        -> [顶层认知总分合成模块 V1.2 风险源升级版] 计算完毕。")
        return df


