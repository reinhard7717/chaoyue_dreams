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
        self._get_atomic_score = lambda name, default=0.5: self.strategy.atomic_states.get(name, pd.Series(default, index=self.strategy.df.index))

    def _fuse_multi_level_scores(self, base_name: str, weights: Dict[str, float] = None) -> pd.Series:
        """
        【新增辅助函数】融合S/A/B等多层置信度分数。
        - 逻辑: 根据给定的权重，将 'SCORE_..._S', 'SCORE_..._A', 'SCORE_..._B' 等分数
                加权融合成一个单一的综合分数。
        - :param base_name: 分数的基础名称 (例如 'MA_BULLISH_RESONANCE').
        - :param weights: 一个字典，定义了 'S', 'A', 'B' 等级的权重。
        - :return: 融合后的分数 (pd.Series).
        """
        if weights is None:
            weights = {'S': 1.0, 'A': 0.6, 'B': 0.3}
        
        total_score = pd.Series(0.0, index=self.strategy.df.index)
        total_weight = 0.0
        
        # 动态地获取并加权S/A/B等级的分数
        for level, weight in weights.items():
            score_name = f"SCORE_{base_name}_{level}"
            if score_name in self.strategy.atomic_states:
                score_series = self.strategy.atomic_states[score_name]
                total_score += score_series * weight
                total_weight += weight
        
        # 如果没有找到任何等级的分数，返回一个中性分数
        if total_weight == 0:
            # 尝试获取没有等级的单一分数
            single_score_name = f"SCORE_{base_name}"
            if single_score_name in self.strategy.atomic_states:
                return self.strategy.atomic_states[single_score_name]
            return pd.Series(0.5, index=self.strategy.df.index)
            
        # 归一化处理
        return (total_score / total_weight).clip(0, 1)

    def run_cognitive_synthesis_engine(self, df: pd.DataFrame) -> Dict[str, pd.Series]: 
        """
        【V337.14 终极修复与增强版】认知综合引擎
            - 【依赖修复】(V337.13) 新增对 synthesize_topping_behaviors 的调用，修复完美风暴信号的上游依赖。
            - 【信号增强】(V337.14) 增强了“近期派发压力”的信号融合维度，并新增“趋势可持续性”信号。
        """
        print("        -> [认知综合引擎 V337.14 终极修复与增强版] 启动，正在合成顶层风险上下文...") 
        # --- 重构模块调用顺序以修复依赖关系 ---
        # 阶段1: 生成基础的、无内部依赖的认知层信号
        df = self.synthesize_behavioral_risks(df)
        df = self.synthesize_holding_risks(df)
        df = self.synthesize_washout_intelligence(df)
        df = self.synthesize_breakdown_risk(df)
        df = self.synthesize_accumulation_breakout_opportunity(df)
        df = self.synthesize_classic_pattern_opportunity(df)
        df = self.synthesize_trend_regime_signals(df)
        df = self.synthesize_volatility_breakout_signals(df)
        df = self.synthesize_reversal_potential_signals(df) 
        df = self.synthesize_consolidation_breakout_signals(df)
        # 阶段2: 基于基础信号，进行第一层融合（结构元信号）
        df = self.synthesize_structural_fusion_scores(df)
        # 阶段3: 基于结构元信号，进行第二层融合（终极确认信号）
        df = self.synthesize_ultimate_confirmation_scores(df)
        # 阶段4: 基于终极确认信号，生成顶级的“完美风暴”共振分数
        df = self.synthesize_ignition_resonance_score(df) 
        df = self.synthesize_breakdown_resonance_score(df) 
        df = self.synthesize_reversal_resonance_scores(df) 
        # 阶段5: 生成依赖广泛输入的宏观质量与状态分数
        df = self.synthesize_trend_quality_score(df)
        df = self.synthesize_pullback_states(df)
        # 阶段6: 此处生成战场上下文，因为它依赖阶段4生成的顶级共振风险分
        df = self.synthesize_contextual_zone_scores(df)
        # 阶段7: 生成依赖战场上下文的其他风险与机会分数
        df = self.synthesize_market_engine_states(df)
        df = self.synthesize_divergence_risks(df)
        df = self.synthesize_opportunity_risk_scores(df)
        df = self.synthesize_topping_behaviors(df)
        df = self.synthesize_perfect_storm_signals(df) 
        # 阶段8: 调用新增的趋势衰竭信号模块
        df = self.synthesize_trend_exhaustion_signals(df)
        # [新增行] 阶段8.5: 调用新增的趋势可持续性信号模块
        df = self.synthesize_trend_sustainability_signals(df)
        # 阶段9: 调用完整实现的顶层认知总分合成模块
        df = self.synthesize_cognitive_scores(df)
        cognitive_states = {}
        default_series = pd.Series(False, index=df.index)
        atomic = self.strategy.atomic_states
        # --- 认知链 1/2: 识别“突破派发”风险 ---
        # 将判断依据从单一的A级分升级为融合后的综合分，提升准确性
        is_strong_rally = df['pct_change_D'] > 0.03
        is_chip_diverging_score = self._fuse_multi_level_scores('CHIP_BEARISH_RESONANCE') # 使用多层置信度融合分数
        cognitive_states['COGNITIVE_RISK_BREAKOUT_DISTRIBUTION'] = is_strong_rally & (is_chip_diverging_score > 0.7)
        # --- 认知链 2/2: 汇总“近期派发压力”上下文 ---
        distribution_score = np.maximum.reduce([
            self._fuse_multi_level_scores('CHIP_TOP_REVERSAL').values,
            self._fuse_multi_level_scores('CHIP_BEARISH_RESONANCE').values,
            self._get_atomic_score('SCORE_BEHAVIOR_PANIC_SELLING_RISK_S', 0.0).values,
            self._get_atomic_score('SCORE_MTF_PROFIT_CUSHION_EROSION_RISK_S', 0.0).values,
            self._get_atomic_score('CHIP_SCORE_FUSED_DECEPTIVE_RALLY', 0.0).values,
            self._get_atomic_score('SCORE_VOL_PRICE_PANIC_DOWN_RISK', 0.0).values,
            self._get_atomic_score('SCORE_BEHAVIOR_WINNERS_FLEEING_A', 0.0).values,
            self._get_atomic_score('COGNITIVE_SCORE_MULTI_DIMENSIONAL_DIVERGENCE_S', 0.0).values,
            self._get_atomic_score('COGNITIVE_SCORE_TOP_REVERSAL_RESONANCE_S', 0.0).values,
            self._get_atomic_score('FF_SCORE_SEPTAFECTA_RESONANCE_DOWN_HIGH', 0.0).values,
            self._get_atomic_score('COGNITIVE_SCORE_BREAKDOWN_RESONANCE_S', 0.0).values, # [新增行] 增加崩溃共振风险
            self._get_atomic_score('COGNITIVE_ULTIMATE_BEARISH_CONFIRMATION_S', 0.0).values, # [新增行] 增加终极看跌确认风险
            self._get_atomic_score('COGNITIVE_SCORE_BULLISH_EXHAUSTION_RISK_S', 0.0).values, # [新增行] 增加看涨衰竭风险
            self._get_atomic_score('COGNITIVE_SCORE_ENGINE_FAILURE_S', 0.0).values, # [新增行] 增加引擎失效风险
        ])
        distribution_series = pd.Series(distribution_score, index=df.index)
        p_dist = get_params_block(self.strategy, 'distribution_context_params', {})
        lookback = get_param_value(p_dist.get('lookback_days'), 10)
        # 将布尔事件升级为对近期最高风险分的判断
        # 如果近期最高派发风险分超过阈值，则认为存在压力
        recent_max_score = distribution_series.rolling(window=lookback, min_periods=1).max()
        cognitive_states['CONTEXT_RECENT_DISTRIBUTION_PRESSURE'] = recent_max_score > 0.7
        print("        -> [认知综合引擎 V337.14 终极修复与增强版] 顶层风险上下文合成完毕。") # [修改行] 更新版本号
        return cognitive_states

    def synthesize_trend_sustainability_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.0 新增】趋势可持续性与衰竭诊断模块
        - 核心职责: 融合“趋势质量”与“反转潜力”两大高维信号，生成具备前瞻性的
                      “趋势可持续性”与“趋势衰竭”认知评分。
        - 核心逻辑:
          - 可持续性: 趋势质量高 且 反转潜力低。
          - 衰竭风险/机会: 反转潜力高 且 市场处于危险/超卖区域。
        - 收益: 创造了全新的信号维度，用于预判趋势的健康度和潜在拐点。
        """
        print("        -> [趋势可持续性诊断模块 V1.0] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 提取核心上游信号 ---
        trend_quality = atomic.get('COGNITIVE_SCORE_TREND_QUALITY', default_score)
        top_reversal_potential = atomic.get('COGNITIVE_SCORE_TOP_REVERSAL_POTENTIAL_A', default_score)
        bottom_reversal_potential = atomic.get('COGNITIVE_SCORE_BOTTOM_REVERSAL_POTENTIAL_A', default_score)
        high_level_zone_context = atomic.get('COGNITIVE_SCORE_RISK_HIGH_LEVEL_ZONE', default_score)
        oversold_context = atomic.get('SCORE_RSI_OVERSOLD_EXTENT', default_score)
        # --- 2. 计算“上升趋势可持续性”评分 ---
        # 逻辑: 趋势质量越高，且顶部反转潜力越低，则上升趋势越可持续
        states['COGNITIVE_SCORE_TREND_SUSTAINABILITY_UP'] = (trend_quality * (1 - top_reversal_potential)).astype(np.float32)
        # --- 3. 计算“上升趋势衰竭”风险评分 ---
        # 逻辑: 顶部反转潜力越高，且处于高位危险区，则衰竭风险越大
        states['COGNITIVE_SCORE_TREND_FATIGUE_RISK'] = (top_reversal_potential * high_level_zone_context).astype(np.float32)
        # --- 4. 计算“下跌趋势衰竭”机会评分 (对称逻辑) ---
        # 逻辑: 底部反转潜力越高，且处于超卖区，则衰竭机会越大
        states['COGNITIVE_SCORE_TREND_FATIGUE_OPP'] = (bottom_reversal_potential * oversold_context).astype(np.float32)
        # --- 5. 更新原子状态库 ---
        self.strategy.atomic_states.update(states)
        print("        -> [趋势可持续性诊断模块 V1.0] 计算完毕。")
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
        print("        -> [盘整突破机会合成模块 V1.0] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        triggers = self.strategy.trigger_events
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        default_series = pd.Series(False, index=df.index)
        # --- 1. 提取并融合“盘整”战备(Setup)信号 ---
        # 使用辅助函数融合S/A/B三级盘整模式分数，得到综合的“战备质量分”
        consolidation_setup_score = self._fuse_multi_level_scores('PATTERN_CONSOLIDATION')
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

    def synthesize_reversal_potential_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.0 新增】反转潜力信号合成模块
        - 核心职责: 融合“K线收盘位置分”与“连续涨跌天数”两大原子信号，
                      生成更高维度的、具有预判性的“顶部/底部反转潜力”认知分数。
        - 核心逻辑:
          - 顶部反转潜力: 连续上涨天数越多，且当日收盘越弱（收盘价在K线位置越低），则潜力分越高。
          - 底部反转潜力: 连续下跌天数越多，且当日收盘越强（收盘价在K线位置越高），则潜力分越高。
        - 收益: 丰富了策略对趋势末端力竭和潜在拐点的早期识别能力。
        """
        print("        -> [反转潜力信号合成模块 V1.0] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 提取并归一化核心原子信号 ---
        # 连续涨跌天数，假设超过5天为强持续，进行归一化
        up_streak_score = (atomic.get('COUNT_CONSECUTIVE_UP_STREAK', default_score) / 5.0).clip(0, 1)
        down_streak_score = (atomic.get('COUNT_CONSECUTIVE_DOWN_STREAK', default_score) / 5.0).clip(0, 1)
        # K线收盘位置分 (0-1)，分数越高代表收盘越强劲
        close_position_score = atomic.get('SCORE_PRICE_POSITION_IN_RANGE', default_score)
        # --- 2. 融合生成“顶部反转潜力”风险分数 ---
        # 逻辑: 连涨天数越多 * 当日收盘越弱(1 - close_position_score) = 顶部反转潜力越大
        top_reversal_potential = up_streak_score * (1 - close_position_score)
        states['COGNITIVE_SCORE_TOP_REVERSAL_POTENTIAL_A'] = top_reversal_potential.astype(np.float32)
        # --- 3. 融合生成“底部反转潜力”机会分数 ---
        # 逻辑: 连跌天数越多 * 当日收盘越强(close_position_score) = 底部反转潜力越大
        bottom_reversal_potential = down_streak_score * close_position_score
        states['COGNITIVE_SCORE_BOTTOM_REVERSAL_POTENTIAL_A'] = bottom_reversal_potential.astype(np.float32)
        # --- 4. 更新原子状态库 ---
        self.strategy.atomic_states.update(states)
        print("        -> [反转潜力信号合成模块 V1.0] 计算完毕。")
        return df

    def synthesize_trend_exhaustion_signals(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V1.0 新增】趋势衰竭信号合成模块
        - 核心职责: 消费此前未被利用的“连涨/连跌天数”原子信号，并结合
                      高位风险、超卖环境等顶层上下文，生成更高维度的
                      “趋势衰竭风险”与“恐慌衰竭机会”认知分数。
        - 收益: 丰富了策略对市场极端情绪和趋势末端行为的捕捉能力。
        """
        print("        -> [趋势衰竭信号合成模块 V1.0] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 提取并归一化连涨/连跌天数 ---
        # 假设连涨/跌超过10天为极限，进行归一化处理
        up_streak_score = (atomic.get('COUNT_CONSECUTIVE_UP_STREAK', default_score) / 10.0).clip(0, 1)
        down_streak_score = (atomic.get('COUNT_CONSECUTIVE_DOWN_STREAK', default_score) / 10.0).clip(0, 1)
        # --- 2. 提取相关的风险与机会上下文分数 ---
        high_level_risk_context = atomic.get('COGNITIVE_SCORE_RISK_HIGH_LEVEL_ZONE', default_score)
        divergence_risk_context = atomic.get('COGNITIVE_SCORE_MULTI_DIMENSIONAL_DIVERGENCE_S', default_score)
        oversold_opp_context = atomic.get('SCORE_RSI_OVERSOLD_EXTENT', default_score)
        capitulation_opp_context = atomic.get('SCORE_BEHAVIOR_CAPITULATION_EXHAUSTION_OPP_A', default_score)
        # --- 3. 融合生成“看涨衰竭风险”分数 ---
        # 逻辑: 连涨天数越多 * 高位风险越大 * 背离风险越大 = 衰竭风险越高
        bullish_exhaustion_score = up_streak_score * high_level_risk_context * divergence_risk_context
        states['COGNITIVE_SCORE_BULLISH_EXHAUSTION_RISK_S'] = bullish_exhaustion_score.astype(np.float32)
        # --- 4. 融合生成“看跌衰竭(恐慌底)机会”分数 ---
        # 逻辑: 连跌天数越多 * 超卖程度越深 * 恐慌盘涌出迹象越明显 = 衰竭机会越大
        bearish_exhaustion_score = down_streak_score * oversold_opp_context * capitulation_opp_context
        states['COGNITIVE_SCORE_BEARISH_EXHAUSTION_OPP_S'] = bearish_exhaustion_score.astype(np.float32)
        # --- 5. 更新原子状态库 ---
        self.strategy.atomic_states.update(states)
        print("        -> [趋势衰竭信号合成模块 V1.0] 计算完毕。")
        return df

    def synthesize_classic_pattern_opportunity(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V1.0 新增】经典形态机会融合模块
        - 核心职责: 消费行为层识别出的“老鸭头”、“N字板”等经典A股形态的数值化评分，
                      并结合当前市场的趋势质量，生成一个经过环境过滤的、更高质量的
                      顶层认知机会分数。
        - 收益: 将未被利用的底层形态信号转化为顶层战术情报，丰富了策略的决策依据。
        """
        print("        -> [经典形态机会融合模块 V1.0] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 提取来自行为层的经典形态原子分数 ---
        # 源1: 老鸭头形态形成中
        old_duck_head_score = atomic.get('SCORE_OLD_DUCK_HEAD_FORMING', default_score)
        # 源2: N字板盘整中
        n_shape_score = atomic.get('SCORE_N_SHAPE_CONSOLIDATION', default_score)
        # --- 2. 提取宏观环境过滤器 ---
        # 使用趋势质量分作为过滤器，确保形态出现在健康的趋势环境中
        trend_quality_context = atomic.get('COGNITIVE_SCORE_TREND_QUALITY', default_score)
        # --- 3. 融合生成认知层“经典形态机会”分数 ---
        # 逻辑: 取所有形态源中的最高分，并乘以环境分进行过滤
        base_pattern_score = np.maximum(old_duck_head_score.values, n_shape_score.values)
        final_score_series = pd.Series(base_pattern_score, index=df.index) * trend_quality_context
        states['COGNITIVE_SCORE_CLASSIC_PATTERN_OPP_S'] = final_score_series.astype(np.float32)
        # --- 4. 更新原子状态库 ---
        self.strategy.atomic_states.update(states)
        print("        -> [经典形态机会融合模块 V1.0] 计算完毕。")
        return df

    def synthesize_breakdown_resonance_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.2 七位一体增强版】多域崩溃共振分数合成模块
        - 核心职责: 融合来自认知、力学、筹码、资金流、波动率、终极结构、ATR波幅七大领域的顶级
                      “崩溃/破位”信号，生成一个置信度极高的“完美风暴”式看跌共振分数。
        - 本次升级: 新增了对“ATR扩张衰竭风险”信号的融合，形成了七位一体的顶级风险信号，
                    进一步提升了信号的可靠性。
        """
        print("        -> [多域崩溃共振分数合成模块 V1.2 七位一体增强版] 启动...") 
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 提取七大领域的核心S级“崩溃”分数 ---
        # 领域1 (认知): 结构性破位风险分
        cognitive_breakdown = atomic.get('COGNITIVE_SCORE_STRUCTURE_BREAKDOWN_S', default_score)
        # 领域2 (力学): 结构力学看跌共振分 (融合S/A/B三级)
        mechanics_breakdown = self._fuse_multi_level_scores('MECHANICS_BEARISH_RESONANCE')
        # 领域3 (筹码): 筹码看跌共振分 (融合S/A/B三级)
        chip_breakdown = self._fuse_multi_level_scores('CHIP_BEARISH_RESONANCE')
        # 领域4 (资金流): 七位一体看跌共振分
        fund_flow_breakdown = self._get_atomic_score('FF_SCORE_SEPTAFECTA_RESONANCE_DOWN_HIGH', default_score)
        # 领域5 (波动率): S级波动率崩溃分
        volatility_breakdown = atomic.get('COGNITIVE_SCORE_VOL_BREAKDOWN_S', default_score)
        # 领域6 (终极结构): 融合了结构元信号与形态元信号的最高级别确认分 
        structural_confirmation = atomic.get('COGNITIVE_ULTIMATE_BEARISH_CONFIRMATION_S', default_score) 
        # 领域7 (ATR波幅): ATR扩张衰竭风险分 
        atr_exhaustion = self._get_atomic_score('SCORE_ATR_EXPANSION_EXHAUSTION_RISK', default_score) 
        # --- 2. 交叉验证：生成“多域崩溃共振”元分数 ---
        # 逻辑: 只有当所有领域的信号都一致看跌时，分数才会高。
        breakdown_resonance_score = (
            cognitive_breakdown * mechanics_breakdown * chip_breakdown *
            fund_flow_breakdown * volatility_breakdown * structural_confirmation *
            atr_exhaustion
        ).astype(np.float32)
        states['COGNITIVE_SCORE_BREAKDOWN_RESONANCE_S'] = breakdown_resonance_score
        # --- 3. 更新原子状态库 ---
        self.strategy.atomic_states.update(states)
        print("        -> [多域崩溃共振分数合成模块 V1.2 七位一体增强版] 计算完毕。") 
        return df

    def synthesize_trend_regime_signals(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V2.1 资金流增强版】趋势政权融合模块
        - 核心职责: 融合“趋势政权”、“均线共振”、“RSI动能”以及“资金流共振”四大领域的数值化评分，
                      生成更高质量的、量化的认知层趋势信号。
        - 本次升级: 【维度增强】新增了对资金流最高级别确认信号的融合，实现了跨领域（波动率+结构+震荡+资金流）的交叉验证，信号更可靠。
        """
        print("        -> [趋势政权融合模块 V2.1] 启动...") 
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 提取核心原子分数 ---
        # 从布尔信号升级为数值化评分
        trending_regime_score = atomic.get('SCORE_TRENDING_REGIME', default_score)
        # 使用辅助函数融合S/A/B三级均线共振分数
        bullish_confluence_score = self._fuse_multi_level_scores('MA_BULLISH_RESONANCE')
        bearish_confluence_score = self._fuse_multi_level_scores('MA_BEARISH_RESONANCE')
        # 从布尔信号升级为数值化评分
        rsi_bullish_accel_score = atomic.get('SCORE_RSI_BULLISH_ACCEL', default_score)
        rsi_bearish_accel_score = atomic.get('SCORE_RSI_BEARISH_ACCEL_RISK', default_score)
        # 提取资金流最高级别确认信号
        fund_flow_ignition_score = atomic.get('FF_SCORE_SEPTAFECTA_RESONANCE_UP_HIGH', default_score)
        # --- 2. 融合生成S级认知分数 ---
        # 逻辑从布尔 AND 升级为分数相乘
        states['COGNITIVE_SCORE_TREND_REGIME_IGNITION_S'] = (
            trending_regime_score * bullish_confluence_score * rsi_bullish_accel_score * fund_flow_ignition_score # 增加资金流确认
        ).astype(np.float32)
        # 看跌信号逻辑保持不变，因为资金流信号不对称
        states['COGNITIVE_SCORE_TREND_REGIME_BREAKDOWN_S'] = (
            trending_regime_score * bearish_confluence_score * rsi_bearish_accel_score
        ).astype(np.float32)
        # --- 3. 更新原子状态库 ---
        self.strategy.atomic_states.update(states)
        print("        -> [趋势政权融合模块 V2.1] 计算完毕。") 
        return df

    def synthesize_volatility_breakout_signals(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V2.1 筹码确认增强版】波动率突破融合模块
        - 核心职责: 消费波动率、价格、成交量的数值化评分，生成经过交叉验证的
                      认知层“突破”与“崩溃”分数。
        - 本次升级: 【维度增强】为S级信号增加了筹码维度的交叉验证，看涨突破需结合“黄金筹码机会”，看跌则结合“筹码看跌共振”，显著提升信号质量。
        """
        print("        -> [波动率突破融合模块 V2.1] 启动...") 
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 提取核心原子分数 ---
        # 环境信号 (Setup) 从布尔升级为数值分
        compression_setup_score = atomic.get('SCORE_VOL_COMPRESSION_LEVEL', default_score)
        expansion_setup_score = atomic.get('SCORE_VOL_EXPANSION_LEVEL', default_score)
        # 动态信号 (Trigger) 从布尔升级为归一化数值分
        norm_window = 120
        min_periods = 24
        score_vol_expanding = df['SLOPE_5_BBW_21_2.0_D'].rolling(norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        score_vol_accelerating = df['ACCEL_5_BBW_21_2.0_D'].rolling(norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        score_price_trending_up = df['SLOPE_5_close_D'].rolling(norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        score_price_accelerating_up = df['ACCEL_5_close_D'].rolling(norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        score_price_trending_down = 1 - score_price_trending_up
        score_price_accelerating_down = 1 - score_price_accelerating_up
        # 确认信号 (Confirmation) 从布尔升级为数值分
        volume_up_confirm_score = atomic.get('SCORE_VOL_PRICE_IGNITION_UP', default_score)
        volume_down_confirm_score = atomic.get('SCORE_VOL_PRICE_PANIC_DOWN_RISK', default_score)
        # 提取筹码维度确认信号
        chip_opportunity_score = atomic.get('CHIP_SCORE_PRIME_OPPORTUNITY_S', default_score)
        chip_bearish_score = self._fuse_multi_level_scores('CHIP_BEARISH_RESONANCE')
        # --- 2. 上升共振 (Breakout) 分数合成 ---
        # 逻辑从布尔 AND 升级为分数相乘
        trigger_b_bullish = score_vol_expanding * score_price_trending_up
        states['COGNITIVE_SCORE_VOL_BREAKOUT_B'] = (compression_setup_score * trigger_b_bullish).astype(np.float32)
        trigger_a_bullish = trigger_b_bullish * volume_up_confirm_score
        states['COGNITIVE_SCORE_VOL_BREAKOUT_A'] = (compression_setup_score * trigger_a_bullish).astype(np.float32)
        trigger_s_bullish = trigger_a_bullish * score_vol_accelerating * score_price_accelerating_up * chip_opportunity_score
        states['COGNITIVE_SCORE_VOL_BREAKOUT_S'] = (compression_setup_score * trigger_s_bullish).astype(np.float32)
        # --- 3. 下跌共振 (Breakdown) 分数合成 ---
        # 逻辑从布尔 AND 升级为分数相乘
        trigger_b_bearish = score_vol_expanding * score_price_trending_down
        states['COGNITIVE_SCORE_VOL_BREAKDOWN_B'] = (expansion_setup_score * trigger_b_bearish).astype(np.float32)
        trigger_a_bearish = trigger_b_bearish * volume_down_confirm_score
        states['COGNITIVE_SCORE_VOL_BREAKDOWN_A'] = (expansion_setup_score * trigger_a_bearish).astype(np.float32)
        trigger_s_bearish = trigger_a_bearish * score_vol_accelerating * score_price_accelerating_down * chip_bearish_score
        states['COGNITIVE_SCORE_VOL_BREAKDOWN_S'] = (expansion_setup_score * trigger_s_bearish).astype(np.float32)
        # --- 4. 更新原子状态库 ---
        self.strategy.atomic_states.update(states)
        print("        -> [波动率突破融合模块 V2.1] 计算完毕。") 
        return df

    def synthesize_structural_fusion_scores(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V1.0 新增 & 逻辑迁移】结构化元信号融合模块
        - 核心职责: 承接原 StructuralIntelligence 中的 `diagnose_fusion_scores` 逻辑。
                      融合来自均线、力学、MTF三大领域的S级信号，生成顶层的“联合作战”元信号。
        - 收益: 遵循分层架构，将顶层融合统一收归认知层。
        """
        print("        -> [结构化元信号融合模块 V1.0] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        signal_sources = {
            'bullish_resonance': ['SCORE_MA_BULLISH_RESONANCE_S', 'SCORE_MECHANICS_BULLISH_RESONANCE_S', 'SCORE_MTF_BULLISH_RESONANCE_S'],
            'bearish_resonance': ['SCORE_MA_BEARISH_RESONANCE_S', 'SCORE_MECHANICS_BEARISH_RESONANCE_S', 'SCORE_MTF_BEARISH_RESONANCE_S'],
            'bottom_reversal': ['SCORE_MA_BOTTOM_REVERSAL_S', 'SCORE_MECHANICS_BOTTOM_REVERSAL_S', 'SCORE_MTF_BOTTOM_REVERSAL_S'],
            'top_reversal': ['SCORE_MA_TOP_REVERSAL_S', 'SCORE_MECHANICS_TOP_REVERSAL_S', 'SCORE_MTF_TOP_REVERSAL_S']
        }
        all_required_signals = [sig for group in signal_sources.values() for sig in group]
        if any(s not in atomic for s in all_required_signals):
            print("          -> [警告] 结构化元信号融合缺少核心上游S级分数，模块已跳过！")
            return df
        default_series = pd.Series(0.0, index=df.index, dtype=np.float32)
        def fuse_scores(source_keys: list) -> pd.Series:
            scores_to_fuse = [atomic.get(key, default_series) for key in source_keys]
            fused_values = np.prod(np.array([s.values for s in scores_to_fuse]), axis=0)
            return pd.Series(fused_values, index=df.index, dtype=np.float32)
        states['COGNITIVE_FUSION_BULLISH_RESONANCE_S'] = fuse_scores(signal_sources['bullish_resonance'])
        states['COGNITIVE_FUSION_BEARISH_RESONANCE_S'] = fuse_scores(signal_sources['bearish_resonance'])
        states['COGNITIVE_FUSION_BOTTOM_REVERSAL_S'] = fuse_scores(signal_sources['bottom_reversal'])
        states['COGNITIVE_FUSION_TOP_REVERSAL_S'] = fuse_scores(signal_sources['top_reversal'])
        self.strategy.atomic_states.update(states)
        print("        -> [结构化元信号融合模块 V1.0] 计算完毕。")
        return df

    def synthesize_ultimate_confirmation_scores(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V1.0 新增 & 逻辑迁移】终极确认融合模块
        - 核心职责: 承接原 StructuralIntelligence 中的 `diagnose_ultimate_confirmation_scores` 逻辑。
                      寻找“连续性共振元信号”与“离散性模式信号”同时发生的最高置信度信号。
        - 收益: 在认知层完成最高级别的“完美风暴”信号融合。
        """
        print("        -> [终极确认融合模块 V1.0] 启动...")
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

    def synthesize_accumulation_breakout_opportunity(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V1.1 信号升级版】蓄势突破机会融合模块
        - 核心职责: 遵循分层架构，融合来自结构层(箱体、平台)的多个突破信号，
                      生成一个统一的、更高质量的“蓄势突破”认知机会。
        - 本次升级: 将对箱体突破S/A/B三级信号的融合方式从“取最大值”升级为
                    使用 _fuse_multi_level_scores 进行加权融合，提升信号质量。
        """
        print("        -> [蓄势突破机会融合模块 V1.1] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_series = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 提取来自结构层的多个突破源信号 ---
        # 源1: 箱体突破 (从取最大值升级为加权融合)
        box_breakout_score = self._fuse_multi_level_scores('BOX_BREAKOUT').values # 使用辅助函数融合S/A/B三级分数
        # 源2: 平台突破
        platform_breakout_score = atomic.get('SCORE_OPP_PLATFORM_BREAKOUT_S', default_series).values
        # --- 2. 融合生成认知层“蓄势突破”分数与信号 ---
        # 逻辑: 取所有突破源中的最高分作为最终的认知分数
        final_score_arr = np.maximum(box_breakout_score, platform_breakout_score)
        final_score_series = pd.Series(final_score_arr, index=df.index, dtype=np.float32)
        states['COGNITIVE_SCORE_ACCUMULATION_BREAKOUT_S'] = final_score_series
        # 生成布尔信号，用于替代旧的 POST_ACCUMULATION_ASCENT_C
        p = get_params_block(self.strategy, 'cognitive_fusion_params', {})
        breakout_threshold = get_param_value(p.get('accumulation_breakout_threshold'), 0.3)
        final_signal = final_score_series > breakout_threshold
        states['COGNITIVE_OPP_ACCUMULATION_BREAKOUT_S'] = final_signal
        # --- 3. 更新原子状态库 ---
        self.strategy.atomic_states.update(states)
        print("        -> [蓄势突破机会融合模块 V1.1] 计算完毕。")
        return df

    def synthesize_ignition_resonance_score(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V1.2 七位一体增强版】多域点火共振分数合成模块
        - 核心职责: 融合来自认知、力学、筹码、资金流、波动率、终极结构、ATR波幅七大领域的顶级
                      “点火/突破”信号，生成一个置信度极高的“完美风暴”式看涨共振分数。
        - 本次升级: 新增了对“ATR扩张点火机会”信号的融合，形成了七位一体的顶级机会信号，
                    确保进攻信号经过最严格的交叉验证。
        """
        print("        -> [多域点火共振分数合成模块 V1.2 七位一体增强版] 启动...") 
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 提取七大领域的核心S级“点火”分数 ---  # [修改行] 从六大领域升级为七大
        # 领域1 (认知): 蓄势突破机会分 (已融合箱体和平台)
        cognitive_breakout = atomic.get('COGNITIVE_SCORE_ACCUMULATION_BREAKOUT_S', default_score)
        # 领域2 (力学): 整体力学健康度元分数 (融合了力矢量、微观、MTF、行为)
        mechanics_ignition = atomic.get('SCORE_DYN_OVERALL_BULLISH_MOMENTUM_S', default_score)
        # 领域3 (筹码): 黄金筹码机会元分数 (融合了结构、核心持仓、净支撑、成本)
        chip_opportunity = atomic.get('CHIP_SCORE_PRIME_OPPORTUNITY_S', default_score)
        # 领域4 (资金流): 七位一体看涨共振分 (最高质量的资金流信号)
        fund_flow_ignition = atomic.get('FF_SCORE_SEPTAFECTA_RESONANCE_UP_HIGH', default_score)
        # 领域5 (波动率): S级波动率突破分 (最高质量的波动率信号)
        volatility_breakout = atomic.get('COGNITIVE_SCORE_VOL_BREAKOUT_S', default_score)
        # 领域6 (终极结构): 融合了结构元信号与形态元信号的最高级别确认分 
        structural_confirmation = atomic.get('COGNITIVE_ULTIMATE_BULLISH_CONFIRMATION_S', default_score) 
        # 领域7 (ATR波幅): ATR扩张点火机会分 
        atr_ignition = self._get_atomic_score('SCORE_ATR_EXPANSION_IGNITION_OPP', default_score) 
        # --- 2. 交叉验证：生成“多域点火共振”元分数 ---
        # 逻辑: 只有当所有领域的信号都一致看涨时，分数才会高。
        ignition_resonance_score = (
            cognitive_breakout * mechanics_ignition * chip_opportunity *
            fund_flow_ignition * volatility_breakout * structural_confirmation *
            atr_ignition # [修改行] 增加 atr_ignition
        ).astype(np.float32)
        states['COGNITIVE_SCORE_IGNITION_RESONANCE_S'] = ignition_resonance_score
        # --- 3. 更新原子状态库 ---
        self.strategy.atomic_states.update(states)
        print("        -> [多域点火共振分数合成模块 V1.2 七位一体增强版] 计算完毕。") 
        return df

    def synthesize_reversal_resonance_scores(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V1.5 板块形态增强版】多域反转共振分数合成模块
        - 核心职责: 遵循“环境+战备+信号”三段式逻辑，融合来自基础层、力学、筹码、形态、
                      资金流、震荡指标六大领域的反转信号，生成高质量的顶部和底部反转共振分数。
        """
        print("        -> [多域反转共振分数合成模块 V1.5 板块形态增强版] 启动...") 
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 新增: 从参数获取触发信号权重 ---
        p = get_params_block(self.strategy, 'reversal_resonance_params', {}) 
        # 增加 board_pattern 权重
        bottom_weights = get_param_value(p.get('bottom_trigger_weights'), {'pattern': 0.15, 'fund_flow': 0.20, 'oscillator': 0.1, 'behavioral': 0.15, 'chip': 0.15, 'volatility': 0.1, 'mechanics': 0.1, 'board_pattern': 0.05}) 
        # 增加 board_pattern 权重
        top_weights = get_param_value(p.get('top_trigger_weights'), {'pattern': 0.15, 'fund_flow': 0.20, 'oscillator': 0.1, 'behavioral': 0.15, 'chip': 0.15, 'volatility': 0.1, 'mechanics': 0.1, 'board_pattern': 0.05}) 
        # --- 1. 合成“底部反转共振”分数 ---
        # 1.1 环境分 (Context): 市场处于超卖状态
        oversold_context = np.maximum(
            atomic.get('SCORE_RSI_OVERSOLD_EXTENT', default_score).values,
            atomic.get('SCORE_BIAS_OVERSOLD_EXTENT', default_score).values
        )
        # 1.2 战备分 (Setup): 力学和筹码结构出现反转迹象
        mechanics_bottom_setup = self._fuse_multi_level_scores('MECHANICS_BOTTOM_REVERSAL')
        chip_bottom_setup = self._fuse_multi_level_scores('CHIP_BOTTOM_REVERSAL')
        reversal_setup_score = (mechanics_bottom_setup + chip_bottom_setup) / 2
        # 1.3 信号分 (Trigger): 形态学、资金流、震荡指标等出现确认信号
        pattern_trigger = self._fuse_multi_level_scores('PATTERN_BOTTOM_REVERSAL')
        # 构建复合资金流底部反转触发信号，融合七大引擎
        fund_flow_bottom_triggers = [
            self._get_atomic_score('FF_SCORE_REVERSAL_BOTTOM_HIGH', default_score).values,
            self._get_atomic_score('FF_SCORE_STRUCTURE_REVERSAL_BOTTOM_HIGH', default_score).values,
            self._get_atomic_score('FF_SCORE_CONFLICT_REVERSAL_BOTTOM_HIGH', default_score).values,
            self._get_atomic_score('FF_SCORE_CMF_REVERSAL_BOTTOM_HIGH', default_score).values,
            self._get_atomic_score('FF_SCORE_XL_REVERSAL_BOTTOM_HIGH', default_score).values,
            self._get_atomic_score('FF_SCORE_RETAIL_REVERSAL_BOTTOM_FISHING', default_score).values,
            self._get_atomic_score('FF_SCORE_INTENSITY_REVERSAL_BOTTOM_HIGH', default_score).values,
        ]
        fund_flow_trigger = pd.Series(np.maximum.reduce(fund_flow_bottom_triggers), index=df.index)
        oscillator_trigger = self._get_atomic_score('SCORE_MACD_BULLISH_DIVERGENCE_OPP', default_score)
        behavioral_trigger = self._get_atomic_score('SCORE_BEHAVIOR_CAPITULATION_EXHAUSTION_OPP_A', default_score) 
        chip_capitulation_trigger = self._get_atomic_score('SCORE_LONG_TERM_CAPITULATION_BOTTOM_REVERSAL_S', default_score) 
        volatility_trigger = self._get_atomic_score('SCORE_VOL_TIPPING_POINT_BOTTOM_OPP', default_score) 
        mechanics_trigger = self._get_atomic_score('SCORE_MTF_BOTTOM_INFLECTION_OPP_A', default_score) 
        board_pattern_trigger = self._get_atomic_score('SCORE_BOARD_EARTH_HEAVEN', default_score)
        #  使用加权平均融合更多维度的触发信号 
        total_weight_bottom = sum(bottom_weights.values()) 
        final_trigger_score = ( 
            pattern_trigger * bottom_weights['pattern'] + fund_flow_trigger * bottom_weights['fund_flow'] + 
            oscillator_trigger * bottom_weights['oscillator'] + behavioral_trigger * bottom_weights['behavioral'] + 
            chip_capitulation_trigger * bottom_weights['chip'] + volatility_trigger * bottom_weights['volatility'] + 
            mechanics_trigger * bottom_weights['mechanics'] +
            board_pattern_trigger * bottom_weights['board_pattern'] # 融合地天板信号
        ) / total_weight_bottom if total_weight_bottom > 0 else default_score 
        # 1.4 最终融合
        bottom_reversal_score = pd.Series(oversold_context, index=df.index) * reversal_setup_score * final_trigger_score 
        states['COGNITIVE_SCORE_BOTTOM_REVERSAL_RESONANCE_S'] = bottom_reversal_score.astype(np.float32)
        # --- 2. 合成“顶部反转共振”分数 (对称逻辑) ---
        # 2.1 环境分 (Context): 市场处于超买状态
        overbought_context = np.maximum(
            atomic.get('SCORE_RSI_OVERBOUGHT_EXTENT', default_score).values,
            atomic.get('SCORE_BIAS_OVERBOUGHT_EXTENT', default_score).values
        )
        # 2.2 战备分 (Setup): 力学和筹码结构出现顶部迹象
        mechanics_top_setup = self._fuse_multi_level_scores('MECHANICS_TOP_REVERSAL')
        chip_top_setup = self._fuse_multi_level_scores('CHIP_TOP_REVERSAL')
        top_setup_score = (mechanics_top_setup + chip_top_setup) / 2
        # 2.3 信号分 (Trigger): 形态学、资金流、震荡指标等出现确认信号
        pattern_top_trigger = self._fuse_multi_level_scores('PATTERN_TOP_REVERSAL')
        # 构建复合资金流顶部反转触发信号，融合七大引擎
        fund_flow_top_triggers = [
            self._get_atomic_score('FF_SCORE_REVERSAL_TOP_HIGH', default_score).values,
            self._get_atomic_score('FF_SCORE_STRUCTURE_REVERSAL_TOP_HIGH', default_score).values,
            self._get_atomic_score('FF_SCORE_CONFLICT_REVERSAL_TOP_HIGH', default_score).values,
            self._get_atomic_score('FF_SCORE_CMF_REVERSAL_TOP_HIGH', default_score).values,
            self._get_atomic_score('FF_SCORE_XL_REVERSAL_TOP_HIGH', default_score).values,
            self._get_atomic_score('FF_SCORE_RETAIL_REVERSAL_TOP_SELLING', default_score).values,
            self._get_atomic_score('FF_SCORE_INTENSITY_REVERSAL_TOP_HIGH', default_score).values,
        ]
        fund_flow_top_trigger = pd.Series(np.maximum.reduce(fund_flow_top_triggers), index=df.index)
        oscillator_top_trigger = self._get_atomic_score('SCORE_MACD_BEARISH_DIVERGENCE_RISK', default_score)
        behavioral_top_trigger = self._get_atomic_score('SCORE_BEHAVIOR_PANIC_SELLING_RISK_S', default_score) 
        chip_instability_trigger = self._fuse_multi_level_scores('LONG_TERM_INSTABILITY_TOP_REVERSAL') 
        volatility_top_trigger = self._get_atomic_score('SCORE_VOL_TIPPING_POINT_TOP_RISK', default_score) 
        mechanics_top_trigger = self._get_atomic_score('SCORE_DYN_EXHAUSTION_DIVERGENCE_RISK_S', default_score) 
        board_pattern_top_trigger = self._get_atomic_score('SCORE_BOARD_HEAVEN_EARTH', default_score) # 增加天地板信号
        #  使用加权平均融合更多维度的触发信号 
        total_weight_top = sum(top_weights.values()) 
        final_top_trigger_score = ( 
            pattern_top_trigger * top_weights['pattern'] + fund_flow_top_trigger * top_weights['fund_flow'] + 
            oscillator_top_trigger * top_weights['oscillator'] + behavioral_top_trigger * top_weights['behavioral'] + 
            chip_instability_trigger * top_weights['chip'] + volatility_top_trigger * top_weights['volatility'] + 
            mechanics_top_trigger * top_weights['mechanics'] +
            board_pattern_top_trigger * top_weights['board_pattern'] # 融合天地板信号
        ) / total_weight_top if total_weight_top > 0 else default_score 
        # 2.4 最终融合
        top_reversal_score = pd.Series(overbought_context, index=df.index) * top_setup_score * final_top_trigger_score 
        states['COGNITIVE_SCORE_TOP_REVERSAL_RESONANCE_S'] = top_reversal_score.astype(np.float32)
        # --- 3. 更新原子状态库 ---
        self.strategy.atomic_states.update(states)
        print("        -> [多域反转共振分数合成模块 V1.5 板块形态增强版] 计算完毕。") 
        return df

    def synthesize_breakdown_risk(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V2.0 数值化升级版】结构性破位风险融合模块
        - 核心职责: 融合“均线破位”、“放量杀跌”和“恐慌大跌”等多个维度的数值化评分，
                      生成一个统一的、量化的“结构性破位”风险分数。
        - 收益: 风险评估从“是否破位”升级为“破位强度有多大”，决策依据更充分。
        """
        # print("        -> [结构性破位风险融合模块 V2.0] 启动...") 
        states = {}
        # --- 1. 提取来自不同情报源的原子分数 ---
        # 升级为消费数值化评分
        ma_broken_score = self._get_atomic_score('SCORE_BEHAVIOR_MA_BREAKDOWN', 0.0) 
        volume_spike_down_score = self._get_atomic_score('SCORE_VOL_PRICE_PANIC_DOWN_RISK', 0.0)
        sharp_drop_score = self._get_atomic_score('SCORE_KLINE_SHARP_DROP', 0.0) 
        # --- 2. 融合生成认知层“结构性破位”风险分数 ---
        # 逻辑从布尔运算升级为数值融合
        core_risk = ma_broken_score * volume_spike_down_score
        panic_risk = ma_broken_score * sharp_drop_score
        # 取两者中的最大值作为最终风险分
        final_score = np.maximum(core_risk, panic_risk)
        states['COGNITIVE_SCORE_STRUCTURE_BREAKDOWN_S'] = final_score.astype(np.float32)
        # --- 3. 更新原子状态库 ---
        self.strategy.atomic_states.update(states)
        print("        -> [结构性破位风险融合模块 V2.0] 计算完毕。") 
        return df

    def synthesize_washout_intelligence(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V2.0 数值化升级版】洗盘情报融合模块
        - 核心职责: 融合来自行为层和基础层的“洗盘”相关【分数】，生成一个更高质量、
                      量化洗盘强度的认知层分数。
        - 收益: 信号从“是否洗盘”升级为“洗盘强度多大”，为后续决策提供更精细的依据。
        """
        # print("        -> [洗盘情报融合模块 V2.0] 启动...") 
        states = {}
        # --- 1. 提取来自不同情报源的原子分数 ---
        # 从布尔事件升级为数值化分数
        behavioral_washout_score = self._get_atomic_score('SCORE_BEHAVIOR_WASHOUT', 0.0)
        foundation_washout_score = self._get_atomic_score('SCORE_VOL_PRICE_PANIC_DOWN_RISK', 0.0)
        # --- 2. 融合生成认知层“洗盘强度分” ---
        # 逻辑从布尔 OR 升级为取最大分
        cognitive_washout_score = np.maximum(behavioral_washout_score, foundation_washout_score)
        states['COGNITIVE_SCORE_WASHOUT_INTENSITY'] = cognitive_washout_score.astype(np.float32)
        # --- 3. 生成认知层“洗盘机会窗口”持续状态 ---
        # 基于分数生成布尔事件，逻辑更清晰
        washout_event_threshold = 0.7 # 定义一个合理的阈值来触发窗口期
        cognitive_washout_event = cognitive_washout_score > washout_event_threshold
        washout_window = cognitive_washout_event.rolling(window=3, min_periods=1).max().astype(bool)
        states['COGNITIVE_STATE_WASHOUT_WINDOW'] = washout_window
        # --- 4. 更新原子状态库 ---
        self.strategy.atomic_states.update(states)
        print("        -> [洗盘情报融合模块 V2.0] 计算完毕。") 
        return df

    def synthesize_divergence_risks(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V2.0 数值化升级版】多维背离风险融合模块
        - 核心职责: 融合来自基础层和力学层的多种顶背离【风险分数】，生成一个统一的、
                      更高维度的认知层风险分数。
        - 收益: 将离散的背离警报升级为连续的风险仪表盘，能更精确地度量顶部风险的累积过程。
        """
        # print("        -> [多维背离风险融合模块 V2.0] 启动...") 
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)

        # --- 1. 提取各维度的背离风险分数 ---
        # 修复所有失效引用，并统一使用数值化评分
        oscillator_divergence_score = np.maximum(
            atomic.get('SCORE_RISK_MTF_RSI_DIVERGENCE_S', default_score).values,
            atomic.get('SCORE_MACD_BEARISH_DIVERGENCE_RISK', default_score).values
        )
        price_momentum_divergence_score = self._get_atomic_score('SCORE_DYN_DIVERGENCE_RISK_A', 0.0)
        exhaustion_divergence_score = self._get_atomic_score('SCORE_DYN_EXHAUSTION_DIVERGENCE_RISK_S', 0.0)
        engine_divergence_score = self._get_atomic_score('SCORE_BEHAVIOR_TOP_DIVERGENCE_RISK_S_PLUS', 0.0)

        # --- 2. 定义触发的战场环境分数 ---
        # 使用数值化的危险区上下文分数
        danger_zone_score = atomic.get('COGNITIVE_SCORE_RISK_HIGH_LEVEL_ZONE', default_score)

        # --- 3. 最终裁定 ---
        # 逻辑从布尔运算升级为数值融合
        # 将所有维度的背离风险取最大值，代表最突出的背离风险
        max_divergence_score = np.maximum.reduce([
            oscillator_divergence_score,
            price_momentum_divergence_score.values,
            exhaustion_divergence_score.values,
            engine_divergence_score.values
        ])
        # 最终风险分 = 危险区上下文 * 最强背离风险
        final_risk_score = danger_zone_score * pd.Series(max_divergence_score, index=df.index)
        states['COGNITIVE_SCORE_MULTI_DIMENSIONAL_DIVERGENCE_S'] = final_risk_score.astype(np.float32)

        # --- 4. 更新原子状态库 ---
        self.strategy.atomic_states.update(states)
        print("        -> [多维背离风险融合模块 V2.0] 计算完毕。") 
        return df

    def synthesize_market_engine_states(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V2.0 数值化升级版】市场引擎状态融合模块
        - 核心职责: 融合多个与“市场引擎效率”相关的原子【风险分数】，生成一个顶层的、
                      量化的“引擎失效”认知风险分数。
        - 收益: 移除了硬编码阈值，通过顶层分数融合提升了判断的准确性和鲁棒性。
        """
        # print("        -> [市场引擎状态融合模块 V2.0] 启动...") 
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)

        # --- 1. 提取引擎失效的多个症状分数 ---
        # 直接使用分数，不再转换为布尔值
        engine_stalling_score = self._get_atomic_score('SCORE_BEHAVIOR_ENGINE_STALLING_RISK_S', 0.0)
        vpa_stagnation_score = self._get_atomic_score('SCORE_RISK_VPA_STAGNATION', 0.0) # 假设此分数存在
        bearish_divergence_score = self._get_atomic_score('SCORE_RISK_MTF_RSI_DIVERGENCE_S', 0.0)

        # --- 2. 定义触发的战场环境分数 ---
        # 使用数值化的危险区上下文分数
        danger_zone_score = atomic.get('COGNITIVE_SCORE_RISK_HIGH_LEVEL_ZONE', default_score)

        # --- 3. 最终裁定 ---
        # 逻辑从布尔运算升级为数值融合
        # 将所有症状分数取最大值，代表最主要的引擎失效风险
        max_symptom_score = np.maximum.reduce([
            engine_stalling_score.values,
            vpa_stagnation_score.values,
            bearish_divergence_score.values
        ])
        # 最终风险分 = 危险区上下文 * 最强引擎失效症状
        final_risk_score = danger_zone_score * pd.Series(max_symptom_score, index=df.index)
        states['COGNITIVE_SCORE_ENGINE_FAILURE_S'] = final_risk_score.astype(np.float32)

        # --- 4. 更新原子状态库 ---
        self.strategy.atomic_states.update(states)
        print("        -> [市场引擎状态融合模块 V2.0] 计算完毕。") 
        return df

    def synthesize_pullback_states(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V2.2 信号融合增强版】认知层回踩状态合成模块
        - 核心职责: 融合行为层与筹码层的【分数】，生成高质量的、量化的“回踩性质”认知分数。
        - 收益: 实现了对回踩“健康度”和“打压强度”的连续度量，为下游战术提供更精细的输入。
        - 本次升级: 增强了对“健康回踩”中筹码稳定性的评估，从消费单一S级信号升级为
                    融合S/A/B三级置信度分数，评估更鲁棒。
        """
        print("        -> [认知层回踩状态合成模块 V2.2] 启动...") 
        states = {}
        # --- 1. 定义通用回踩条件和环境分数 ---
        is_pullback_day = (df['pct_change_D'] < 0).astype(float)
        constructive_context_score = self._get_atomic_score('COGNITIVE_SCORE_TREND_QUALITY', 0.0) # 使用趋势质量分
        # --- 2. 合成“健康回踩”分数 (Healthy Pullback Score) ---
        # 所有判断升级为0-1的连续分数
        gentle_drop_score = (1 - (df['pct_change_D'].abs() / 0.05)).clip(0, 1) # 跌幅越小，分数越高
        shrinking_volume_score = self._get_atomic_score('SCORE_VOL_WEAKENING_DROP', 0.0) 
        winner_holding_tight_score = 1.0 - self._fuse_multi_level_scores('PROFIT_TAKING_TOP_REVERSAL') # 升级为融合S/A/B三级分数，更准确地评估抛压
        chip_stable_score = 1.0 - self._fuse_multi_level_scores('CHIP_BEARISH_RESONANCE') # [修改行] 从消费单一S级信号升级为融合S/A/B三级分数，评估更鲁棒
        healthy_pullback_score = (
            is_pullback_day * constructive_context_score *
            gentle_drop_score * shrinking_volume_score *
            winner_holding_tight_score * chip_stable_score
        )
        states['COGNITIVE_SCORE_PULLBACK_HEALTHY_S'] = healthy_pullback_score.astype(np.float32)
        # --- 3. 合成“打压式回踩”分数 (Suppressive Pullback Score) ---
        # 所有判断升级为0-1的连续分数
        significant_drop_score = (df['pct_change_D'].abs() / 0.07).clip(0, 1) # 跌幅越大，分数越高
        # 手动融合S/A/B三级分数，因为S级信号名特殊(RESONANCE)，无法使用通用函数
        capitulation_b = self._get_atomic_score('SCORE_CAPITULATION_BOTTOM_REVERSAL_B', 0.0)
        capitulation_a = self._get_atomic_score('SCORE_CAPITULATION_BOTTOM_REVERSAL_A', 0.0)
        capitulation_s = self._get_atomic_score('SCORE_CAPITULATION_BOTTOM_RESONANCE_S', 0.0)
        # 使用与 _fuse_multi_level_scores 相同的默认权重
        weights = {'S': 1.0, 'A': 0.6, 'B': 0.3}
        total_weight = sum(weights.values())
        panic_selling_score = (capitulation_s * weights['S'] + capitulation_a * weights['A'] + capitulation_b * weights['B']) / total_weight
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
        【V1.0 新增】认知层持仓风险合成模块
        - 核心职责: 基于顶层融合的“趋势质量分”的动态变化，诊断持仓健康度是否出现“失速”风险。
        - 收益: 提供一个更宏观、更可靠的早期预警信号，取代了原先基于单一筹码指标的判断。
        """
        print("        -> [认知层持仓风险合成模块 V1.0] 启动...")
        states = {}
        # 使用更高维度的“趋势质量分”来判断健康度
        trend_quality_score = self._get_atomic_score('COGNITIVE_SCORE_TREND_QUALITY')
        # 定义“改善”为质量分高于前一日
        is_improving = trend_quality_score > trend_quality_score.shift(1)
        was_improving = is_improving.shift(1).fillna(False)
        is_not_improving_now = ~is_improving
        # “失速”定义为：昨日仍在改善，今日不再改善
        states['COGNITIVE_HOLD_RISK_HEALTH_STALLING'] = was_improving & is_not_improving_now
        self.strategy.atomic_states.update(states)
        print("        -> [认知层持仓风险合成模块 V1.0] 计算完毕。")
        return df

    def synthesize_contextual_zone_scores(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V2.4 终极风险增强版】战场上下文评分模块
        - 核心职责: 将多个维度的原子风险【分数】融合成一个顶层的“高位危险区”综合风险分。
        - 核心升级: 修复了所有失效的上游信号引用，并引入了更多维度的风险源，
                      使得危险区的评估更加全面和准确。
        - 本次升级: 【维度增强】新增了对“多域崩溃共振”和“终极看跌确认”两大顶级风险信号的融合，确保危险区评分能捕捉到最致命的风险。
        """
        print("        -> [战场上下文评分模块 V2.4 终极风险增强版] 启动...") 
        # --- 1. 量化“高位危险区”得分 ---
        # 修复并扩展了用于融合的风险信号源
        risk_scores = {
            'bias': self._get_atomic_score('SCORE_BIAS_OVERBOUGHT_EXTENT', 0.0),
            'exhaustion': self._get_atomic_score('SCORE_RISK_MOMENTUM_EXHAUSTION', 0.0),
            'chip_decay': self._get_atomic_score('SCORE_RISK_CHIP_STRUCTURE_DECAY', 0.0),
            'churn': self._get_atomic_score('SCORE_RISK_DYNAMIC_DECEPTIVE_CHURN', 0.0),
            'deceptive_rally': self._get_atomic_score('CHIP_SCORE_FUSED_DECEPTIVE_RALLY', 0.0),
            'chip_top_reversal': self._fuse_multi_level_scores('CHIP_TOP_REVERSAL'), 
            'structural_weakness': self._get_atomic_score('SCORE_MTF_STRUCTURAL_WEAKNESS_RISK_S', 0.0),
            'engine_stalling': self._get_atomic_score('SCORE_BEHAVIOR_ENGINE_STALLING_RISK_S', 0.0),
            'volume_spike_down': self._get_atomic_score('SCORE_VOL_PRICE_PANIC_DOWN_RISK', 0.0),
            'chip_divergence': self._fuse_multi_level_scores('CHIP_BEARISH_RESONANCE'),
            'chip_fault': self._fuse_multi_level_scores('FAULT_RISK_TOP_REVERSAL'),
            'structural_fault': self._fuse_multi_level_scores('STRUCTURE_BEARISH_RESONANCE'),
            'fund_flow_reversal': self._get_atomic_score('FF_SCORE_REVERSAL_TOP_HIGH', 0.0),
            'mechanics_reversal': self._fuse_multi_level_scores('MECHANICS_TOP_REVERSAL'),
            'retail_frenzy': self._get_atomic_score('FF_SCORE_RETAIL_RESONANCE_FRENZY_HIGH', 0.0), 
            'ultimate_breakdown': self._get_atomic_score('COGNITIVE_SCORE_BREAKDOWN_RESONANCE_S', 0.0), # 增加终极崩溃共振风险
            'ultimate_confirmation': self._get_atomic_score('COGNITIVE_ULTIMATE_BEARISH_CONFIRMATION_S', 0.0), # 增加终极看跌确认风险
        }
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
        print("        -> [战场上下文评分模块 V2.4 终极风险增强版] 计算完毕。") 
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
        【V1.4 政权维度增强版】趋势质量融合评分模块
        - 核心职责: 融合行为、筹码、资金流、结构、市场情绪、动态力学、市场政权七大情报源的核心健康度指标，
                      生成一个顶层的、量化的“趋势质量”评分 (0-1)。
        - 本次升级: 【维度增强】新增了对“市场政权(SCORE_TRENDING_REGIME)”的融合，使趋势质量评估包含了对趋势持续性的考量。
        """
        print("        -> [趋势质量融合评分模块 V1.4 政权维度增强版] 启动...") 
        # --- 1. 提取各领域的核心健康度评分 ---
        # 行为健康度: 基于市场引擎效率。效率高(失速风险低)则健康。
        behavior_risk_score = self._get_atomic_score('SCORE_BEHAVIOR_ENGINE_STALLING_RISK_S', default=0.5)
        behavior_health_score = 1.0 - behavior_risk_score
        # 筹码健康度: 基于筹码的看涨共振信号。共振越强，健康度越高。
        chip_health_score = self._fuse_multi_level_scores('CHIP_BULLISH_RESONANCE') 
        # 资金流健康度: 基于“七位一体”的看涨共振信号。共振越强，健康度越高。
        fund_flow_health_score = self._get_atomic_score('FF_SCORE_SEPTAFECTA_RESONANCE_UP_HIGH')
        # 结构健康度: 基于均线系统的排列、动能和加速度。 
        structural_health_score = self._get_atomic_score('SCORE_MA_HEALTH') 
        # 市场情绪健康度: 基于获利盘比例的动态变化。 
        market_health_score = self._get_atomic_score('SCORE_MKT_HEALTH_S') 
        # 力学健康度: 基于动态力学引擎的顶层元融合分数。 
        mechanics_health_score = self._get_atomic_score('SCORE_DYN_OVERALL_BULLISH_MOMENTUM_S') 
        # 市场政权健康度: 基于Hurst指数，衡量趋势的持续性
        regime_health_score = self._get_atomic_score('SCORE_TRENDING_REGIME')
        # --- 2. 定义各维度权重 ---
        p = get_params_block(self.strategy, 'trend_quality_params', {})
        weights = {
            'behavior': get_param_value(p.get('behavior_weight'), 0.10), 
            'chip': get_param_value(p.get('chip_weight'), 0.20), # 调整权重
            'fund_flow': get_param_value(p.get('fund_flow_weight'), 0.15), 
            'structural': get_param_value(p.get('structural_weight'), 0.15), 
            'market': get_param_value(p.get('market_weight'), 0.10), 
            'mechanics': get_param_value(p.get('mechanics_weight'), 0.20), # 调整权重
            'regime': get_param_value(p.get('regime_weight'), 0.10) # 增加政权维度权重
        }
        # --- 3. 加权融合生成最终的趋势质量分 ---
        trend_quality_score = (
            behavior_health_score * weights['behavior'] +
            chip_health_score * weights['chip'] +
            fund_flow_health_score * weights['fund_flow'] +
            structural_health_score * weights['structural'] + 
            market_health_score * weights['market'] +
            mechanics_health_score * weights['mechanics'] +
            regime_health_score * weights['regime'] # 融合政权健康度
        )
        df['COGNITIVE_SCORE_TREND_QUALITY'] = trend_quality_score
        self.strategy.atomic_states['COGNITIVE_SCORE_TREND_QUALITY'] = df['COGNITIVE_SCORE_TREND_QUALITY']
        print("        -> [趋势质量融合评分模块 V1.4 政权维度增强版] 计算完毕。") 
        return df

    def synthesize_behavioral_risks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V2.1 命名规范化版】行为风险融合模块
        - 核心职责: 将来自不同领域的纯粹原子信号进行交叉验证，生成更高维度的、
                      经过情景过滤的认知层风险信号。
        - 核心升级: 将原有的布尔逻辑升级为数值化评分，能够更精确地量化风险程度。
        - 本次升级: 将输出信号重命名为'SCORE_...'前缀，以符合数值化评分的命名规范。
        """
        # print("        -> [行为风险融合模块 V2.1 命名规范化版] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        # --- 融合信号1: 高位获利盘出逃风险 (A级) ---
        # 全面升级为数值化评分，信号更平滑
        # 情报1 (筹码): 获利盘兑现意愿强烈 (消费数值分)
        profit_taking_score = self._get_atomic_score('SCORE_CHIP_PROFIT_TAKING_INTENSITY', 0.0)
        # 情报2 (行为): 价格处于近期高位区间 (消费数值分，替代失效的 PRICE_STATE_NEAR_HIGH_RANGE)
        price_position_score = self._get_atomic_score('SCORE_PRICE_POSITION_IN_RECENT_RANGE', 0.0)
        # 最终裁定: 风险分 = 获利盘兑现分 * 价格位置分
        # 重命名信号以符合数值化评分的规范
        states['SCORE_BEHAVIOR_WINNERS_FLEEING_A'] = (profit_taking_score * price_position_score).astype(np.float32)
        # --- 更新原子状态库 ---
        self.strategy.atomic_states.update(states)
        print("        -> [行为风险融合模块 V2.1 命名规范化版] 计算完毕。") 
        return df

    def diagnose_recent_reversal_context(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】近期反转上下文诊断模块
        - 核心职责: 识别近期（如3天内）是否发生过关键的反转触发事件。
        """
        states = {}
        default_series = pd.Series(False, index=df.index)
        # 我们认为“显性反转阳线”是最高质量的反转信号
        is_reversal_trigger = self.strategy.atomic_states.get('TRIGGER_DOMINANT_REVERSAL', default_series)
        # 使用滚动窗口检查过去3天内是否出现过该信号
        had_recent_reversal = is_reversal_trigger.rolling(window=3, min_periods=1).max().astype(bool)
        states['CONTEXT_RECENT_REVERSAL_SIGNAL'] = had_recent_reversal
        return states

    def diagnose_trend_stage_score(self, df: pd.DataFrame) -> Dict[str, pd.Series]: 
        """
        【V401.4 终极确认增强版】趋势阶段评分模块
        - 核心升级:
          1. 全面修复了所有失效的原子信号引用，使其与现代化情报层对齐。
          2. 将所有风险维度的判断从布尔型升级为基于0-1的数值化评分。
          3. 引入多级置信度分数融合，更精确地量化每个风险维度的强度。
        - 本次升级: 【维度增强】新增了对“终极看跌确认”信号的融合，这是最高级别的风险信号，能极大地提升对上涨末期风险的识别准确性。
        """
        print("        -> [趋势阶段评分模块 V401.4 终极确认增强版] 启动...") 
        states = {}
        atomic = self.strategy.atomic_states
        # --- 1. 计算“上涨初期”的量化分数 (Early Stage Score) ---
        ascent_structure_score = atomic.get('COGNITIVE_OPP_ACCUMULATION_BREAKOUT_S', pd.Series(False, index=df.index)).astype(float)
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
        # 重新定义风险维度，并全面数值化，移除硬编码阈值判断
        risk_dimension_scores = [
            # ... (维度 1-22 无修改)
            self._get_atomic_score('SCORE_BIAS_OVERBOUGHT_EXTENT', 0.0) * 25,
            self._get_atomic_score('SCORE_RISK_MOMENTUM_EXHAUSTION', 0.0) * 25,
            self._get_atomic_score('SCORE_ACTION_RISK_RALLY_WITH_DIVERGENCE', 0.0) * 25,
            np.maximum(
                self._get_atomic_score('SCORE_RISK_VPA_STAGNATION', 0.0).values,
                self._get_atomic_score('SCORE_RISK_VPA_VOLUME_ACCELERATING', 0.0).values
            ) * 25,
            self._get_atomic_score('SCORE_VOL_EXPANSION_LEVEL', 0.0) * 25,
            self._get_atomic_score('COGNITIVE_SCORE_RISK_TOP_DISTRIBUTION', 0.0) * 40,
            self._get_atomic_score('SCORE_BEHAVIOR_PANIC_SELLING_RISK_S', 0.0) * 25,
            self._fuse_multi_level_scores('CHIP_TOP_REVERSAL') * 30,
            self._fuse_multi_level_scores('STRUCTURE_BEARISH_RESONANCE') * 35,
            self._get_atomic_score('SCORE_MTF_PROFIT_CUSHION_EROSION_RISK_S', 0.0) * 25,
            self._get_atomic_score('SCORE_BEHAVIOR_ENGINE_STALLING_RISK_S', 0.0) * 25,
            self._get_atomic_score('SCORE_RISK_MTF_RSI_DIVERGENCE_S', 0.0) * 20,
            self._get_atomic_score('CHIP_SCORE_FUSED_DECEPTIVE_RALLY', 0.0) * 35,
            self._get_atomic_score('SCORE_MTF_STRUCTURAL_WEAKNESS_RISK_S', 0.0) * 35,
            self._get_atomic_score('SCORE_MACD_BEARISH_CONFLUENCE_RISK', 0.0) * 20,
            self._get_atomic_score('SCORE_VOL_PRICE_PANIC_DOWN_RISK', 0.0) * 30,
            self._get_atomic_score('FF_SCORE_RETAIL_RESONANCE_FRENZY_HIGH', 0.0) * 25,
            self._fuse_multi_level_scores('MTF_BEARISH_RESONANCE') * 35,
            self._get_atomic_score('FF_SCORE_CONFLICT_REVERSAL_TOP_HIGH', 0.0) * 30,
            self._get_atomic_score('FF_SCORE_INTENSITY_REVERSAL_TOP_HIGH', 0.0) * 30,
            self._fuse_multi_level_scores('MECHANICS_TOP_REVERSAL') * 35,
            self._fuse_multi_level_scores('PATTERN_TOP_REVERSAL') * 35,
            self._get_atomic_score('SCORE_DYN_EXHAUSTION_DIVERGENCE_RISK_S', 0.0) * 30, 
            self._get_atomic_score('SCORE_FV_CHAOTIC_EXPANSION_RISK', 0.0) * 30, 
            # 维度 25: 终极看跌确认风险 (最高级别的融合风险信号) 
            self._get_atomic_score('COGNITIVE_ULTIMATE_BEARISH_CONFIRMATION_S', 0.0) * 50, # 给予最高权重
        ]
        # 使用列表推导式和np.add.reduce进行向量化求和
        score_components = [s.to_numpy(dtype=np.float32) for s in risk_dimension_scores]
        late_stage_score_arr = np.add.reduce(score_components)
        late_stage_score = pd.Series(late_stage_score_arr, index=df.index, dtype=int)
        states['CONTEXT_TREND_LATE_STAGE_SCORE'] = late_stage_score
        states['CONTEXT_TREND_STAGE_LATE'] = late_stage_score >= 320
        print("        -> [趋势阶段评分模块 V401.4 终极确认增强版] 计算完毕。") 
        return states

    def diagnose_market_structure_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]: 
        """
        【V278.6 信号融合增强版】 - 联合作战司令部
        - 核心重构: 全面修复所有失效的原子信号引用，并升级为基于数值化评分的判断逻辑。
        - 本次升级: 【信号升级】将“顶部危险评分”中的单一S级筹码结构风险，升级为
                    消费融合后的S/A/B三级置信度分数，提升了风险评估的一致性与准确性。
        """
        print("        -> [联合作战司令部 V278.6 信号融合增强版] 启动，正在分析战场核心结构...") # 更新版本号和说明
        structure_states = {}
        default_series = pd.Series(False, index=df.index)
        atomic = self.strategy.atomic_states
        # --- 步骤1：情报总览 (全面升级为数值化评分) ---
        ma_bullish_score = self._fuse_multi_level_scores('MA_BULLISH_RESONANCE') 
        dyn_trend_healthy_score = self._get_atomic_score('SCORE_DYN_BULLISH_RESONANCE_S', 0.0)
        chip_concentrating_score = self._fuse_multi_level_scores('CHIP_BULLISH_RESONANCE') 
        is_price_above_long_ma = (df['close_D'] > df['EMA_55_D']).astype(float) # 转为浮点数用于乘法
        ma_bearish_score = self._fuse_multi_level_scores('MA_BEARISH_RESONANCE') 
        dyn_trend_weakening_score = self._get_atomic_score('SCORE_DYN_EXHAUSTION_DIVERGENCE_RISK_S', 0.0)
        chip_diverging_score = self._fuse_multi_level_scores('CHIP_BEARISH_RESONANCE') 
        risk_chip_failure_score = self._fuse_multi_level_scores('STRUCTURE_BEARISH_RESONANCE') # 从单一S级信号升级为融合S/A/B三级分数
        risk_late_stage_score_raw = self.strategy.atomic_states.get('CONTEXT_TREND_LATE_STAGE_SCORE', pd.Series(0, index=df.index))
        # 将上涨末期分数归一化到0-1，假设最大值为600
        risk_late_stage_score = (risk_late_stage_score_raw / 600).clip(0, 1)
        prime_opportunity_score = self._get_atomic_score('CHIP_SCORE_PRIME_OPPORTUNITY_S', 0.0)
        # --- 步骤2：联合裁定 (基于数值化评分) ---
        # 主升浪评分
        base_uptrend_score = (ma_bullish_score * dyn_trend_healthy_score * chip_concentrating_score * is_price_above_long_ma)
        structure_states['SCORE_STRUCTURE_MAIN_UPTREND_WAVE_S'] = base_uptrend_score.astype(np.float32)
        structure_states['SCORE_STRUCTURE_FORTRESS_UPTREND_S_PLUS'] = (base_uptrend_score * prime_opportunity_score).astype(np.float32)
        # 兼容旧版布尔信号
        structure_states['STRUCTURE_MAIN_UPTREND_WAVE_S'] = structure_states['SCORE_STRUCTURE_MAIN_UPTREND_WAVE_S'] > 0.4
        structure_states['STRUCTURE_FORTRESS_UPTREND_S_PLUS'] = structure_states['SCORE_STRUCTURE_FORTRESS_UPTREND_S_PLUS'] > 0.5
        # “黄金阵地”评分
        is_extreme_squeeze_score = self._get_atomic_score('SCORE_VOL_COMPRESSION_LEVEL', 0.0)
        has_energy_advantage_score = self._get_atomic_score('SCORE_MECHANICS_BULLISH_RESONANCE_S', 0.0)
        prime_setup_score = prime_opportunity_score * is_extreme_squeeze_score * has_energy_advantage_score
        structure_states['SCORE_SETUP_PRIME_STRUCTURE_S'] = prime_setup_score.astype(np.float32)
        # 兼容旧版布尔信号
        structure_states['SETUP_PRIME_STRUCTURE_S'] = prime_setup_score > 0.6 # (0.9 * 0.9 * 0.7) approx
        # 反转初期信号修复 (保持布尔逻辑，因为是事件驱动)
        is_recent_reversal = atomic.get('CONTEXT_RECENT_REVERSAL_SIGNAL', default_series)
        is_ma_short_slope_positive = df.get('SLOPE_5_EMA_5_D', pd.Series(0, index=df.index)) > 0
        structure_states['STRUCTURE_EARLY_REVERSAL_B'] = is_recent_reversal & is_ma_short_slope_positive
        # 顶部危险评分
        confirmed_distribution_score = self._get_atomic_score('SCORE_S_PLUS_CONFIRMED_DISTRIBUTION', 0.0) # 消费新的数值化分数
        mechanics_top_reversal_score = self._fuse_multi_level_scores('MECHANICS_TOP_REVERSAL') # 获取力学顶部反转分
        pattern_top_reversal_score = self._fuse_multi_level_scores('PATTERN_TOP_REVERSAL') # 获取形态顶部反转分
        topping_danger_score = np.maximum.reduce([
            risk_chip_failure_score.values, # 此处现在消费的是融合后的分数
            risk_late_stage_score.values,
            confirmed_distribution_score.values, # 现在是数值化分数
            mechanics_top_reversal_score.values, # 融合力学风险
            pattern_top_reversal_score.values, # 融合形态风险
        ])
        structure_states['SCORE_STRUCTURE_TOPPING_DANGER_S'] = pd.Series(topping_danger_score, index=df.index, dtype=np.float32)
        # 兼容旧版布尔信号
        structure_states['STRUCTURE_TOPPING_DANGER_S'] = structure_states['SCORE_STRUCTURE_TOPPING_DANGER_S'] > 0.6
        # 下跌通道评分 
        bearish_channel_score = ma_bearish_score * dyn_trend_weakening_score * chip_diverging_score
        structure_states['SCORE_STRUCTURE_BEARISH_CHANNEL_F'] = bearish_channel_score.astype(np.float32)
        # 兼容旧版布尔信号
        structure_states['STRUCTURE_BEARISH_CHANNEL_F'] = bearish_channel_score > 0.5
        print("        -> [联合作战司令部 V278.6 信号融合增强版] 核心战局定义升级完成。") # 更新版本号
        return structure_states

    def synthesize_topping_behaviors(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V331.5 数值化升级版】顶部行为合成模块
        - 核心重构: 职责被简化为“行为合成”。它消费已有的“战场上下文”和
                    “筹码动态”情报，将其融合成顶层的战术信号。
        - 本次升级: 【信号升级】新增了数值化的 `SCORE_S_PLUS_CONFIRMED_DISTRIBUTION` 信号。
                    它通过将“高位危险区得分”与“拉升派发风险分”相乘，取代了原有的
                    布尔逻辑，实现了对“确认派发”风险的精确量化。
        """
        print("        -> [顶部行为合成模块 V331.5 数值化升级版] 启动...") 
        states = {}
        default_series = pd.Series(False, index=df.index)
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        atomic = self.strategy.atomic_states # 增加atomic引用
        # --- 1. 军备检查  ---
        # 检查依赖的顶层上下文信号
        required_states = [
            'COGNITIVE_SCORE_RISK_HIGH_LEVEL_ZONE', # 依赖数值分
            'CONTEXT_CHIP_STRATEGIC_DISTRIBUTION'
        ]
        if any(s not in self.strategy.atomic_states for s in required_states):
            print("          -> [警告] 缺少合成“顶部行为”所需情报，模块跳过。")
            return {}
        # --- 2. 评估“天时”：识别当天的危险拉升行为 (升级为数值化) ---
        is_rallying_score = self._get_atomic_score('SCORE_PRICE_POSITION_IN_RECENT_RANGE', 0.0) # 使用近期(20日)价格位置分，更符合“高位”上下文
        # 2.1 拉升出货 (核心风险行为)
        # 使用多级置信度融合分数代替废弃的 CHIP_DYN_DIVERGING
        diverging_score = self._fuse_multi_level_scores('CHIP_BEARISH_RESONANCE')
        # 生成数值化的风险分数
        states['SCORE_ACTION_RISK_RALLY_WITH_DIVERGENCE'] = (is_rallying_score * diverging_score).astype(np.float32)
        # 基于数值分生成兼容性的布尔信号
        states['ACTION_RISK_RALLY_WITH_DIVERGENCE'] = states['SCORE_ACTION_RISK_RALLY_WITH_DIVERGENCE'] > 0.6
        # 2.2 天量滞涨 
        is_huge_volume = df['volume_D'] > df['VOL_MA_21_D'] * 2.5
        is_stagnant = df['pct_change_D'] < 0.01
        states['ACTION_RISK_RALLY_STAGNATION'] = is_huge_volume & is_stagnant
        # --- 3. 【S+级情报融合】：在危险战区确认派发行为 (全面数值化) ---
        danger_zone_score = self.strategy.atomic_states.get('COGNITIVE_SCORE_RISK_HIGH_LEVEL_ZONE', default_score) # 获取数值分
        distributing_action_score = states.get('SCORE_ACTION_RISK_RALLY_WITH_DIVERGENCE', default_score) # 获取数值分
        # 生成数值化的S+级风险分
        states['SCORE_S_PLUS_CONFIRMED_DISTRIBUTION'] = (danger_zone_score * distributing_action_score).astype(np.float32)
        # 基于新的数值分生成兼容性的布尔信号
        states['RISK_S_PLUS_CONFIRMED_DISTRIBUTION'] = states['SCORE_S_PLUS_CONFIRMED_DISTRIBUTION'] > 0.5 # 假设阈值为0.5
        # --- 3.5 【S级战略风险融合】：在战略派发背景下出现的任何拉升都是陷阱  ---
        is_rallying = df['pct_change_D'] > 0.02 # 此处保留原始的拉升定义
        is_strategic_distribution = self.strategy.atomic_states.get('CONTEXT_CHIP_STRATEGIC_DISTRIBUTION', default_series)
        states['RISK_STRATEGIC_DISTRIBUTION_RALLY_TRAP_S'] = is_rallying & is_strategic_distribution
        if states['RISK_STRATEGIC_DISTRIBUTION_RALLY_TRAP_S'].any():
            print(f"          -> [S级战略风险] 侦测到 {states['RISK_STRATEGIC_DISTRIBUTION_RALLY_TRAP_S'].sum()} 次“战略派发背景下的诱多陷阱”！")
        # --- 4. 重新定义“健康锁筹拉升” (增加保险丝)  ---
        # 使用多级置信度融合分数代替废弃的 CHIP_DYN_CONCENTRATING
        concentrating_score = self._fuse_multi_level_scores('CHIP_BULLISH_RESONANCE')
        is_concentrating = concentrating_score > 0.6
        is_in_danger_zone = self.strategy.atomic_states.get('CONTEXT_RISK_HIGH_LEVEL_ZONE', default_series) # 重新获取布尔信号
        states['RALLY_STATE_HEALTHY_LOCKED'] = is_rallying & is_concentrating & ~is_in_danger_zone
        return states

    def determine_main_force_behavior_sequence(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V304.4 信号融合增强版】
        - 核心修复: 全面修正了此状态机对上游原子信号的引用，将其替换为基于现代化的
                    【数值化分数】的阈值判断。解决了因引用失效信号导致的逻辑崩溃问题。
        - 本次升级: 将关键的“吸筹”状态判断条件，从单一A级信号升级为多层置信度融合分数。
        """
        print("    --- [战略推演单元 V304.4 信号融合增强版] 启动... ---") 
        atomic = self.strategy.atomic_states
        default_series_bool = pd.Series(False, index=df.index)
        # 全面替换为基于现代化数值分数的阈值判断
        conditions = {
            # 吸筹/建仓信号 (Accumulation Signals)
            'is_concentrating_resonance': (self._fuse_multi_level_scores('CHIP_BULLISH_RESONANCE') > 0.7),
            'is_reversal_gathering': (self._fuse_multi_level_scores('CHIP_BOTTOM_REVERSAL') > 0.7),
            'is_sustained_inflow': (self._get_atomic_score('FF_SCORE_SEPTAFECTA_RESONANCE_UP_HIGH') > 0.6),
            'is_concentrating': (self._fuse_multi_level_scores('CHIP_BULLISH_RESONANCE') > 0.6), # 升级为消费融合分数
            # 拉升/突破信号 (Markup/Breakout Signals)
            # 修复失效信号引用，升级为更高质量的 CHIP_STRUCTURE_PRIME_OPPORTUNITY_S
            'is_prime_opportunity': atomic.get('CHIP_STRUCTURE_PRIME_OPPORTUNITY_S', default_series_bool),
            'is_markup_breakout': (self._get_atomic_score('COGNITIVE_SCORE_ACCUMULATION_BREAKOUT_S') > 0.6),
            'is_engine_ignition': (self._get_atomic_score('SCORE_DYN_BULLISH_RESONANCE_S') > 0.8),
            'is_macd_golden_cross': (self._get_atomic_score('SCORE_MACD_BULLISH_CONFLUENCE') > 0.7),
            'is_volume_spike_up': (self._get_atomic_score('SCORE_VOL_PRICE_IGNITION_UP') > 0.7),
            'is_trending_regime': (self._get_atomic_score('SCORE_TRENDING_REGIME') > 0.8),
            # 派发/顶部信号 (Distribution/Topping Signals)
            'is_distributing': (self._fuse_multi_level_scores('CHIP_TOP_REVERSAL') > 0.7),
            'is_diverging_resonance': (self._fuse_multi_level_scores('CHIP_BEARISH_RESONANCE') > 0.7),
            'is_stagnation': (self._get_atomic_score('SCORE_RISK_VPA_STAGNATION') > 0.8),
            'is_engine_stalling': (self._get_atomic_score('SCORE_BEHAVIOR_ENGINE_STALLING_RISK_S') > 0.8),
            'is_macd_death_cross': (self._get_atomic_score('SCORE_MACD_BEARISH_CONFLUENCE_RISK') > 0.7),
            'is_overextended_risk': (self._fuse_multi_level_scores('MTF_BEARISH_RESONANCE') > 0.8),
            # 风险/下跌信号 (Risk/Decline Signals)
            # 升级过时的布尔信号为基于数值分的判断
            'is_sharp_drop': (self._get_atomic_score('SCORE_KLINE_SHARP_DROP', 0.0) > 0.8),
            'is_volume_spike_down': (self._get_atomic_score('SCORE_VOL_PRICE_PANIC_DOWN_RISK') > 0.7),
            'is_below_long_ma': (df['close_D'] < df['EMA_55_D']),
            # 其他状态信号
            'is_sideways': (df.get('SLOPE_5_close_D', pd.Series(0, index=df.index)).abs() < 0.01),
        }
        # 将所有Series一次性转换为NumPy数组
        for key in conditions:
            conditions[key] = conditions[key].to_numpy(dtype=bool)
        n = len(df)
        main_force_state_arr = np.full(n, MainForceState.IDLE.value, dtype=int)
        for i in range(1, n):
            prev_state = MainForceState(main_force_state_arr[i-1])
            current_state = prev_state
            if prev_state == MainForceState.IDLE:
                if conditions['is_concentrating_resonance'][i] or conditions['is_reversal_gathering'][i] or conditions['is_sustained_inflow'][i]: 
                    current_state = MainForceState.ACCUMULATING
            elif prev_state == MainForceState.ACCUMULATING:
                # 使用 is_prime_opportunity 替代 is_fortress_built
                if conditions['is_prime_opportunity'][i] or conditions['is_markup_breakout'][i] or conditions['is_engine_ignition'][i] or conditions['is_macd_golden_cross'][i] or conditions['is_volume_spike_up'][i] or conditions['is_trending_regime'][i]: 
                    current_state = MainForceState.MARKUP
                elif conditions['is_diverging_resonance'][i]:
                    current_state = MainForceState.DISTRIBUTING
                elif conditions['is_sharp_drop'][i] and conditions['is_concentrating'][i]: 
                    current_state = MainForceState.WASHING
            elif prev_state == MainForceState.WASHING:
                # 使用 is_prime_opportunity 替代 is_fortress_built
                if conditions['is_prime_opportunity'][i] or conditions['is_markup_breakout'][i] or conditions['is_engine_ignition'][i] or conditions['is_macd_golden_cross'][i] or conditions['is_volume_spike_up'][i] or conditions['is_trending_regime'][i]: 
                    current_state = MainForceState.MARKUP
                elif not conditions['is_concentrating'][i]: 
                    current_state = MainForceState.DISTRIBUTING
                elif conditions['is_sideways'][i] and conditions['is_concentrating'][i]: 
                    current_state = MainForceState.ACCUMULATING
            elif prev_state == MainForceState.MARKUP:
                if conditions['is_distributing'][i] or conditions['is_diverging_resonance'][i] or conditions['is_stagnation'][i] or conditions['is_engine_stalling'][i] or conditions['is_macd_death_cross'][i] or conditions['is_volume_spike_down'][i] or conditions['is_overextended_risk'][i]: 
                    current_state = MainForceState.DISTRIBUTING
                elif conditions['is_sharp_drop'][i] and conditions['is_concentrating'][i]: 
                    current_state = MainForceState.WASHING
            elif prev_state == MainForceState.DISTRIBUTING:
                if conditions['is_below_long_ma'][i]: current_state = MainForceState.COLLAPSE
            elif prev_state == MainForceState.COLLAPSE:
                if conditions['is_sideways'][i] and not conditions['is_below_long_ma'][i]: current_state = MainForceState.IDLE
            main_force_state_arr[i] = current_state.value
        df['main_force_state'] = main_force_state_arr
        print("    --- [战略推演单元 V304.4 信号融合增强版] 主力行为序列已生成。 ---") 
        return df

    def synthesize_chip_fund_flow_synergy(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 数值化重构版】筹码与资金流协同合成模块
        - 核心重构: 废除所有已失效的布尔信号，全面升级为消费现代化的数值化融合分数。
        - 核心逻辑: 将筹码的“共振分”与资金流的“七位一体共振分”相乘，生成更高置信度的协同信号。
        - 收益: 信号质量与鲁棒性大幅提升，能更精确地量化“吸筹”与“派发”的协同强度。
        """
        print("        -> [筹码与资金流协同合成模块 V2.0 数值化重构版] 启动...") # 方法块
        states = {}
        # --- 1. 获取核心筹码与资金流的数值化融合分数 ---
        # 使用 fuse_multi_level_scores 融合多级置信度，获得更平滑的筹码分数
        chip_bullish_score = self._fuse_multi_level_scores('CHIP_BULLISH_RESONANCE')
        chip_bearish_score = self._fuse_multi_level_scores('CHIP_BEARISH_RESONANCE')
        # 获取资金流最高质量的“七位一体”融合分数
        fund_flow_bullish_score = self._get_atomic_score('FF_SCORE_SEPTAFECTA_RESONANCE_UP_HIGH', 0.5)
        fund_flow_bearish_score = self._get_atomic_score('FF_SCORE_SEPTAFECTA_RESONANCE_DOWN_HIGH', 0.5)
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
        print("        -> [完美风暴信号合成模块 V1.1 数值化升级版] 启动...") 
        states = {}
        atomic = self.strategy.atomic_states
        triggers = self.strategy.trigger_events
        default_series = pd.Series(False, index=df.index)
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. S++级情报融合：定义“完美风暴”顶部风险 (数值化) ---
        # 逻辑: S+级“确认派发”风险分 * 顶层“融合顶背离”结构分 = 完美风暴风险强度
        cognitive_fusion_top_reversal_score = atomic.get('COGNITIVE_FUSION_TOP_REVERSAL_S', default_score)
        confirmed_distribution_score = atomic.get('SCORE_S_PLUS_CONFIRMED_DISTRIBUTION', default_score) # [修改行] 消费数值化分数
        perfect_storm_top_score = (confirmed_distribution_score * cognitive_fusion_top_reversal_score).astype(np.float32) # [修改行] 分数相乘
        states['COGNITIVE_SCORE_PERFECT_STORM_TOP_S_PLUS'] = perfect_storm_top_score # [修改行] 新增数值化信号
        states['COGNITIVE_RISK_PERFECT_STORM_TOP_S_PLUS'] = perfect_storm_top_score > 0.5 # [修改行] 基于分数生成兼容性布尔信号
        if states['COGNITIVE_RISK_PERFECT_STORM_TOP_S_PLUS'].any():
            print(f"          -> [S++级顶级风险] 侦测到 {states['COGNITIVE_RISK_PERFECT_STORM_TOP_S_PLUS'].sum()} 次“完美风暴”顶部风险信号！")
        # --- 2. S++级情报融合：定义“完美风暴”底部机会 (数值化) ---
        # 逻辑: “显性反转K线”强度分 * 顶层“融合底背离”结构分 = 完美风暴机会强度
        cognitive_fusion_bottom_reversal_score = atomic.get('COGNITIVE_FUSION_BOTTOM_REVERSAL_S', default_score)
        reversal_candle_score = triggers.get('TRIGGER_DOMINANT_REVERSAL', default_series).astype(float) # [修改行] 将布尔触发器转为数值分 (0.0 或 1.0)
        perfect_storm_bottom_score = (reversal_candle_score * cognitive_fusion_bottom_reversal_score).astype(np.float32) # [修改行] 分数相乘
        states['COGNITIVE_SCORE_PERFECT_STORM_BOTTOM_S_PLUS'] = perfect_storm_bottom_score # [修改行] 新增数值化信号
        states['COGNITIVE_OPP_PERFECT_STORM_BOTTOM_S_PLUS'] = perfect_storm_bottom_score > 0.5 # [修改行] 基于分数生成兼容性布尔信号
        if states['COGNITIVE_OPP_PERFECT_STORM_BOTTOM_S_PLUS'].any():
            print(f"          -> [S++级顶级机会] 侦测到 {states['COGNITIVE_OPP_PERFECT_STORM_BOTTOM_S_PLUS'].sum()} 次“完美风暴”底部机会信号！")
        # --- 3. 更新原子状态库 ---
        self.strategy.atomic_states.update(states)
        print("        -> [完美风暴信号合成模块 V1.1 数值化升级版] 计算完毕。") # [修改行] 更新版本号
        return df

    def _diagnose_lock_chip_reconcentration_tactic(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.2 王牌重铸数值化版】锁仓再集中S+战法诊断模块
        - 核心重构: (V2.0) 将战法的“准备状态”从有缺陷的A级信号，升级为经过战场环境过滤的
                      S级“筹码结构黄金机会”信号。
        - 本次升级: 【数值化】将原有的布尔逻辑升级为“战备分 * 点火分”的数值化评分体系，
                    能够更精确地量化战法机会的质量。
        """
        print("        -> [S+战法诊断] 正在扫描“锁仓再集中(V2.2 王牌重铸数值化版)”...") 
        states = {}
        atomic = self.strategy.atomic_states
        triggers = self.strategy.trigger_events
        default_series = pd.Series(False, index=df.index)
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 定义“准备状态”评分 (Setup Score) ---
        setup_score = atomic.get('CHIP_SCORE_PRIME_OPPORTUNITY_S', default_score)
        # --- 2. 定义“点火事件”评分 (Ignition Score) ---
        trigger_chip_ignition_score = triggers.get('TRIGGER_CHIP_IGNITION', default_series).astype(float)
        trigger_energy_release_score = triggers.get('TRIGGER_ENERGY_RELEASE', default_series).astype(float)
        cost_accel_score = atomic.get('SCORE_PLATFORM_COST_ACCEL', default_score)
        fractal_opp_score = triggers.get('FRACTAL_OPP_SQUEEZE_BREAKOUT_CONFIRMED', default_series).astype(float)
        ignition_trigger_score_arr = np.maximum.reduce([
            trigger_chip_ignition_score.values,
            trigger_energy_release_score.values,
            cost_accel_score.values,
            fractal_opp_score.values
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
        # 使用融合后的数值分代替废弃的 CHIP_DYN_OBJECTIVE_DIVERGING
        is_diverging = self._fuse_multi_level_scores('CHIP_BEARISH_RESONANCE') > divergence_threshold
        is_late_stage = atomic.get('CONTEXT_TREND_STAGE_LATE', default_series)
        is_ma_broken = self._get_atomic_score('SCORE_MA_HEALTH', 1.0) < 0.4 # 修复失效的 MA_STATE_STABLE_BULLISH 信号，升级为基于均线健康分的判断
        is_health_stalling = atomic.get('COGNITIVE_HOLD_RISK_HEALTH_STALLING', default_series)
        hard_termination_condition = is_diverging | is_late_stage | is_ma_broken
        if terminate_on_stalling:
            hard_termination_condition |= is_health_stalling
        # --- 4. 定义“软性巡航”条件  ---
        # 使用融合后的数值分代替废弃的 CHIP_DYN_CONCENTRATING
        is_cruise_condition_met = self._fuse_multi_level_scores('CHIP_BULLISH_RESONANCE') > concentration_threshold if require_concentration else pd.Series(True, index=df.index)
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

        # 1. 获取五大领域的核心S级看涨分数
        mechanics_score = atomic.get('SCORE_DYN_BULLISH_RESONANCE_S', default_score)
        chip_score = self._fuse_multi_level_scores('CHIP_BULLISH_RESONANCE', {'S': 1.0, 'A': 0.0, 'B': 0.0}) # 仅使用S级
        structure_score = self._fuse_multi_level_scores('MA_BULLISH_RESONANCE', {'S': 1.0, 'A': 0.0, 'B': 0.0}) # 仅使用S级
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
        【V2.0 战略过滤版】终极战法合成模块
        - 核心修复: 为S++和S+级王牌战法增加了“必须处于上涨初期”的战略环境过滤器。
        - 收益: 解决了该战法在上涨末期被“力竭性突破”欺骗的致命缺陷，
                确保我们的王牌武器只在战役的“点火阶段”投入，而不是在“高潮出货”阶段。
        """
        # print("        -> [终极战法合成模块 V2.0 战略过滤版] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        triggers = self.strategy.trigger_events
        default_series = pd.Series(False, index=df.index)
        # --- 1. 定义S级“黄金阵地” (Prime Setup) ---
        is_prime_chip_structure = atomic.get('CHIP_STRUCTURE_PRIME_OPPORTUNITY_S', default_series)
        is_extreme_squeeze = self._get_atomic_score('SCORE_VOL_COMPRESSION_LEVEL', 0.0) > 0.9 # 修复失效的 VOL_STATE_EXTREME_SQUEEZE 信号，升级为基于波动率压缩分的判断
        has_energy_advantage = self._get_atomic_score('SCORE_MECHANICS_BULLISH_RESONANCE_S', 0.0) > 0.7 # 修复失效的 MECHANICS_ENERGY_ADVANTAGE 信号，升级为基于力学共振分的判断
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
        trigger_prime_breakout_s = triggers.get('TRIGGER_PRIME_BREAKOUT_S', default_series)
        # --- 3. 定义战略环境过滤器 ---
        # 这是本次修复的核心。确保终极战法在触发的当天，战场环境依然是安全的“上涨初期”。
        # 这解决了因突破当天状态变化而导致战法在错误时机触发的致命逻辑陷阱。
        is_in_early_stage_today = atomic.get('CONTEXT_TREND_STAGE_EARLY', default_series)
        # --- 4. 【终极裁定】生成王牌战法 (已注入战略智慧) ---
        is_triggered_today = trigger_prime_breakout_s
        # 4.1 生成 S++ 战法
        was_setup_s_plus_plus_yesterday = setup_s_plus_plus.shift(1).fillna(False)
        # 最终裁定 = 昨日S++级准备就绪 AND 今日发动S级总攻 AND 【今日必须仍处于上涨初期】
        final_tactic_s_plus_plus = was_setup_s_plus_plus_yesterday & is_triggered_today & is_in_early_stage_today
        states['TACTIC_PRIME_STRUCTURE_BREAKOUT_S_PLUS_PLUS'] = final_tactic_s_plus_plus
        # 4.2 生成 S+ 战法
        was_setup_s_plus_yesterday = setup_s_plus.shift(1).fillna(False)
        # 最终裁定 = 昨日S+级准备就绪 AND 今日发动S级总攻 AND 【今日必须仍处于上涨初期】 (且不与S++重叠)
        final_tactic_s_plus = was_setup_s_plus_yesterday & is_triggered_today & is_in_early_stage_today & ~final_tactic_s_plus_plus
        states['TACTIC_PRIME_STRUCTURE_BREAKOUT_S_PLUS'] = final_tactic_s_plus
        if final_tactic_s_plus_plus.any():
            print(f"          -> [S++级王牌战法] 侦测到 {final_tactic_s_plus_plus.sum()} 次“终极结构突破”机会！")
        if final_tactic_s_plus.any():
            print(f"          -> [S+级王牌战法] 侦测到 {final_tactic_s_plus.sum()} 次“次级结构突破”机会！")
        return states

    def _diagnose_pullback_tactics_matrix(self, df: pd.DataFrame, enhancements: Dict) -> Dict[str, pd.Series]:
        """
        【V7.2 信号升级版】回踩战术诊断模块
        - 核心升级: 为 S+ 级“巡航回踩确认”战法增加了“非上涨末期”的前置条件。
        - 信号修复: 修复了对回踩性质信号的引用，从布尔型升级为消费数值化分数。
        """
        # print("        -> [回踩战术矩阵 V7.2 信号升级版] 启动...") 
        states = {}
        atomic = self.strategy.atomic_states
        triggers = self.strategy.trigger_events
        default_series = pd.Series(False, index=df.index)
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)

        # --- 1. 提取核心情报  ---
        # 战场环境
        lookback_window = 15
        ascent_start_event = atomic.get('COGNITIVE_OPP_ACCUMULATION_BREAKOUT_S', default_series)
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
        【V1.3 数值化升级版】高级战法合成模块
        - 核心职责: 合成那些需要复杂时序逻辑的高级战法。
        - 本次升级: 将原有的布尔逻辑升级为“窗口期质量分 * 确认信号分 * 环境分”的
                    数值化评分体系，能够更精确地量化战法机会的质量。
        """
        print("        -> [高级战法合成模块 V1.3 数值化升级版] 启动...") 
        states = {}
        atomic = self.strategy.atomic_states
        triggers = self.strategy.trigger_events
        default_series = pd.Series(False, index=df.index)
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 战法1: 【战法S+】断层新生·主升浪 (数值化) ---
        # 1.1 核心事件评分: 手动融合S/A/B三级恐慌抛售反转分
        capitulation_b = self._get_atomic_score('SCORE_CAPITULATION_BOTTOM_REVERSAL_B', 0.0)
        capitulation_a = self._get_atomic_score('SCORE_CAPITULATION_BOTTOM_REVERSAL_A', 0.0)
        capitulation_s = self._get_atomic_score('SCORE_CAPITULATION_BOTTOM_RESONANCE_S', 0.0)
        weights = {'S': 1.0, 'A': 0.6, 'B': 0.3}
        total_weight = sum(weights.values())
        fault_event_score = (capitulation_s * weights['S'] + capitulation_a * weights['A'] + capitulation_b * weights['B']) / total_weight
        # 1.2 确认信号评分: 融合强力阳线与筹码点火
        confirmation_trigger_score_arr = np.maximum(
            triggers.get('TRIGGER_DOMINANT_REVERSAL', default_series).astype(float).values,
            triggers.get('TRIGGER_CHIP_IGNITION', default_series).astype(float).values
        )
        confirmation_trigger_score = pd.Series(confirmation_trigger_score_arr, index=df.index)
        # 1.3 状态过滤评分: 主升浪黄金航道质量分
        main_uptrend_score = atomic.get('SCORE_STRUCTURE_MAIN_UPTREND_WAVE_S', default_score)
        # 1.4 时序逻辑评分: 在断层事件发生后的3天窗口期内，寻找最高质量的事件
        fault_window_score = fault_event_score.rolling(window=3, min_periods=1).max()
        # 1.5 最终裁定: (窗口期质量分) * (今日确认分) * (全程环境分)
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
        【V1.2 数值化升级版】压缩突破战术剧本合成模块
        - 核心职责: 严格执行“战备(Setup) + 确认(Trigger)”的战术剧本逻辑。
        - 本次升级: 将原有的布尔逻辑升级为“战备分 * 点火分”的数值化评分体系，
                    能够更精确地量化战术剧本的质量。
        """
        print("        -> [压缩突破战术剧本合成模块 V1.2 数值化升级版] 启动...") 
        states = {}
        atomic = self.strategy.atomic_states
        triggers = self.strategy.trigger_events
        default_series = pd.Series(False, index=df.index)
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 剧本1: S+级 - 极致压缩·暴力突破 (数值化) ---
        # 战备分(昨日):
        vol_compression_score = atomic.get('SCORE_VOL_COMPRESSION_LEVEL', default_score)
        setup_extreme_squeeze_score = vol_compression_score.shift(1).fillna(0.0)
        # 确认分(今日):
        trigger_explosive_breakout_score = triggers.get('TRIGGER_EXPLOSIVE_BREAKOUT_S', default_series).astype(float)
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
        trigger_prime_breakout_score = triggers.get('TRIGGER_PRIME_BREAKOUT_S', default_series).astype(float)
        # 最终剧本分:
        score_s = (setup_breakout_eve_score * trigger_prime_breakout_score).astype(np.float32)
        states['SCORE_PLAYBOOK_BREAKOUT_EVE_S'] = score_s
        states['PLAYBOOK_BREAKOUT_EVE_S'] = score_s > 0.6
        # --- 剧本3: A级 - 常规压缩·确认突破 (数值化) ---
        # 战备分(昨日):
        setup_normal_squeeze_score = vol_compression_score.shift(1).fillna(0.0)
        # 确认分(今日):
        trigger_grinding_advance_score = triggers.get('TRIGGER_GRINDING_ADVANCE_A', default_series).astype(float)
        trigger_any_breakout_score = np.maximum(trigger_explosive_breakout_score, trigger_grinding_advance_score)
        # 最终剧本分 (注意：需要排除掉更高级别的S+剧本，保证信号互斥)
        score_a = (setup_normal_squeeze_score * pd.Series(trigger_any_breakout_score, index=df.index)).astype(np.float32)
        states['SCORE_PLAYBOOK_NORMAL_SQUEEZE_BREAKOUT_A'] = score_a
        states['PLAYBOOK_NORMAL_SQUEEZE_BREAKOUT_A'] = (score_a > 0.5) & ~states['PLAYBOOK_EXTREME_SQUEEZE_EXPLOSION_S_PLUS']
        return states

    def synthesize_cognitive_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.0 新增】顶层认知总分合成模块
        - 核心职责: 融合所有顶层的、跨领域的认知机会分与风险分，生成最终的
                      “认知看涨总分”与“认知看跌总分”，作为策略决策的最高依据。
        - 核心逻辑:
          - 看涨总分: 取所有S级“机会”类信号的最大值，代表当天最强的看涨理由。
          - 看跌总分: 取所有S级“风险”类信号的最大值，代表当天最强的看跌理由。
        - 收益: 为策略提供了两个最终的、可直接用于比较和决策的顶层仪表盘。
        """
        print("        -> [顶层认知总分合成模块 V1.0] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 汇总所有S级的“机会”类认知分数 ---
        bullish_scores = [
            atomic.get('COGNITIVE_SCORE_IGNITION_RESONANCE_S', default_score).values,
            atomic.get('COGNITIVE_SCORE_BOTTOM_REVERSAL_RESONANCE_S', default_score).values,
            atomic.get('COGNITIVE_SCORE_CLASSIC_PATTERN_OPP_S', default_score).values,
            atomic.get('COGNITIVE_SCORE_BEARISH_EXHAUSTION_OPP_S', default_score).values,
            atomic.get('COGNITIVE_SCORE_CONSOLIDATION_BREAKOUT_OPP_A', default_score).values, # 虽然是A级，但性质是机会
            atomic.get('COGNITIVE_OPP_PERFECT_STORM_BOTTOM_S_PLUS', default_score).astype(float).values,
        ]
        cognitive_bullish_score = np.maximum.reduce(bullish_scores)
        states['COGNITIVE_BULLISH_SCORE'] = pd.Series(cognitive_bullish_score, index=df.index, dtype=np.float32)
        # --- 2. 汇总所有S级的“风险”类认知分数 ---
        bearish_scores = [
            atomic.get('COGNITIVE_SCORE_BREAKDOWN_RESONANCE_S', default_score).values,
            atomic.get('COGNITIVE_SCORE_TOP_REVERSAL_RESONANCE_S', default_score).values,
            atomic.get('COGNITIVE_SCORE_MULTI_DIMENSIONAL_DIVERGENCE_S', default_score).values,
            atomic.get('COGNITIVE_SCORE_ENGINE_FAILURE_S', default_score).values,
            atomic.get('COGNITIVE_SCORE_BULLISH_EXHAUSTION_RISK_S', default_score).values,
            atomic.get('COGNITIVE_RISK_PERFECT_STORM_TOP_S_PLUS', default_score).astype(float).values,
            atomic.get('COGNITIVE_SCORE_RISK_TOP_DISTRIBUTION', default_score).values,
        ]
        cognitive_bearish_score = np.maximum.reduce(bearish_scores)
        states['COGNITIVE_BEARISH_SCORE'] = pd.Series(cognitive_bearish_score, index=df.index, dtype=np.float32)
        # --- 3. 更新原子状态库 ---
        self.strategy.atomic_states.update(states)
        print("        -> [顶层认知总分合成模块 V1.0] 计算完毕。")
        return df





