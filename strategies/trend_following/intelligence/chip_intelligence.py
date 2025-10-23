# 文件: strategies/trend_following/intelligence/chip_intelligence.py
# 筹码情报模块
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following.utils import transmute_health_to_ultimate_signals, get_params_block, get_param_value, normalize_score, normalize_to_bipolar, calculate_holographic_dynamics

class ChipIntelligence:
    def __init__(self, strategy_instance, dynamic_thresholds: Dict):
        """
        初始化筹码情报模块。
        :param strategy_instance: 策略主实例的引用。
        :param dynamic_thresholds: 动态阈值字典。
        """
        self.strategy = strategy_instance
        self.dynamic_thresholds = dynamic_thresholds

    def run_chip_intelligence_command(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V602.0 · 分层印证版】筹码情报最高司令部
        - 核心升级: 引入1日周期，并建立“分层动态印证”框架，废除固定的55日宏观背景。
        - 指挥流程: 1. 为每个核心公理，在所有时间级别上进行诊断，每个级别都由其更长一级趋势进行印证。
                      2. 在终极信号合成器中，对多维度的分数进行“周期内融合”和“跨周期共振”两步锻造。
        """
        all_chip_states = {}
        # 增加1日周期，实现最灵敏的战术变化捕捉
        periods = [1, 5, 13, 21, 55]
        
        concentration_scores = self._diagnose_concentration_dynamics(df, periods)
        all_chip_states['SCORE_CHIP_MTF_CONCENTRATION'] = concentration_scores
        accumulation_scores = self._diagnose_main_force_action(df, periods)
        all_chip_states['SCORE_CHIP_MTF_ACCUMULATION'] = accumulation_scores
        power_transfer_scores = self._diagnose_power_transfer(df, periods)
        all_chip_states['SCORE_CHIP_MTF_POWER_TRANSFER'] = power_transfer_scores
        ultimate_signals = self._synthesize_ultimate_signals(
            concentration_scores,
            accumulation_scores,
            power_transfer_scores
        )
        all_chip_states.update(ultimate_signals)
        accumulation_potential_states = self.diagnose_accumulation_playbooks(df)
        all_chip_states.update(accumulation_potential_states)
        capitulation_potential_states = self.diagnose_capitulation_reversal_potential(df)
        all_chip_states.update(capitulation_potential_states)
        return all_chip_states

    def _synthesize_ultimate_signals(self, df: pd.DataFrame, concentration: Dict[int, pd.Series], accumulation: Dict[int, pd.Series], power_transfer: Dict[int, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V3.0 · 上下文感知版】终极信号合成器
        - 核心升级: 引入全局上下文！为底部/顶部反转信号和看跌共振信号，注入由 utils.calculate_context_scores
                      计算的 bottom_context_score 和 top_context_score，实现筹码行为与市场大环境的协同判断。
        """
        states = {}
        periods = sorted(concentration.keys())
        tf_weights = {1: 0.1, 5: 0.4, 13: 0.3, 21: 0.15, 55: 0.05}
        norm_window = 55
        # [代码新增开始] 获取全局上下文分数
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        p_context = get_param_value(p_conf.get('context_modulation_params'), {})
        bottom_context_factor = get_param_value(p_context.get('bottom_context_factor'), 0.5)
        top_context_factor = get_param_value(p_context.get('top_context_factor'), 0.5)
        self.strategy.atomic_states['strategy_instance_ref'] = self.strategy
        bottom_context_score, top_context_score = calculate_context_scores(df, self.strategy.atomic_states)
        del self.strategy.atomic_states['strategy_instance_ref']
        # [代码新增结束]
        bullish_scores_by_period = {}
        bearish_scores_by_period = {}
        for p in periods:
            bullish_scores_by_period[p] = (concentration[p] + accumulation[p] + power_transfer[p]) / 3.0
            bearish_scores_by_period[p] = ((1 - concentration[p]) + (1 - accumulation[p]) + (1 - power_transfer[p])) / 3.0
        bullish_resonance = pd.Series(0.0, index=self.strategy.df_indicators.index)
        bearish_resonance = pd.Series(0.0, index=self.strategy.df_indicators.index)
        total_weight = sum(tf_weights.get(p, 0) for p in periods)
        if total_weight > 0:
            for p in periods:
                weight = tf_weights.get(p, 0) / total_weight
                bullish_resonance += bullish_scores_by_period[p] * weight
                bearish_resonance += bearish_scores_by_period[p] * weight
        states['SCORE_CHIP_BULLISH_RESONANCE'] = bullish_resonance.fillna(0).clip(0, 1).astype(np.float32)
        # [代码修改开始] 使用 top_context_score 增强看跌共振信号
        final_bearish_resonance = (bearish_resonance * (1 + top_context_score * top_context_factor)).clip(0, 1)
        states['SCORE_CHIP_BEARISH_RESONANCE'] = final_bearish_resonance.fillna(0).clip(0, 1).astype(np.float32)
        # [代码修改结束]
        bottom_reversal_scores = {}
        top_reversal_scores = {}
        for p in periods:
            context_p = periods[periods.index(p) + 1] if periods.index(p) + 1 < len(periods) else p
            bottom_reversal_scores[p] = self._calculate_holographic_divergence(bullish_scores_by_period[p], p, context_p, norm_window)
            top_reversal_scores[p] = self._calculate_holographic_divergence(bearish_scores_by_period[p], p, context_p, norm_window)
        bottom_reversal_divergence = pd.Series(0.0, index=self.strategy.df_indicators.index)
        top_reversal_divergence = pd.Series(0.0, index=self.strategy.df_indicators.index)
        if total_weight > 0:
            for p in periods:
                weight = tf_weights.get(p, 0) / total_weight
                bottom_reversal_divergence += bottom_reversal_scores[p] * weight
                top_reversal_divergence += top_reversal_scores[p] * weight
        # [代码修改开始] 使用 bottom_context_score 增强底部反转信号
        final_bottom_reversal = (bottom_reversal_divergence * (1 + bottom_context_score * bottom_context_factor)).clip(0, 1)
        states['SCORE_CHIP_BOTTOM_REVERSAL'] = final_bottom_reversal.clip(0, 1).astype(np.float32)
        # [代码修改结束]
        # [代码修改开始] 使用 top_context_score 增强顶部反转信号
        final_top_reversal = (top_reversal_divergence * (1 + top_context_score * top_context_factor)).clip(0, 1)
        states['SCORE_CHIP_TOP_REVERSAL'] = final_top_reversal.clip(0, 1).astype(np.float32)
        # [代码修改结束]
        tactical_reversal = (bullish_resonance * 0.5).astype(np.float32)
        states['SCORE_CHIP_TACTICAL_REVERSAL'] = tactical_reversal
        p = 5
        cost_divergence_score = normalize_score(df.get(f'SLOPE_{p}_cost_divergence_D'), df.index, norm_window, ascending=True)
        loser_turnover_up = normalize_score(df.get(f'SLOPE_{p}_turnover_from_losers_ratio_D'), df.index, norm_window, ascending=True)
        transfer_to_main_force_evidence = (cost_divergence_score * loser_turnover_up)**0.5
        cost_convergence_score = normalize_score(df.get(f'SLOPE_{p}_cost_divergence_D'), df.index, norm_window, ascending=False)
        loser_turnover_down = normalize_score(df.get(f'SLOPE_{p}_turnover_from_losers_ratio_D'), df.index, norm_window, ascending=False)
        transfer_to_retail_evidence = (cost_convergence_score * loser_turnover_down)**0.5
        transfer_snapshot = (transfer_to_main_force_evidence - transfer_to_retail_evidence).astype(np.float32)
        distribution_strength = (transfer_snapshot.clip(-1, 0) * -1).astype(np.float32)
        hades_trap_score = (states['SCORE_CHIP_BOTTOM_REVERSAL'] * distribution_strength).clip(0, 1)
        states['SCORE_CHIP_HADES_TRAP'] = hades_trap_score.astype(np.float32)
        return states

    def _diagnose_concentration_dynamics(self, df: pd.DataFrame, periods: list) -> Dict[int, pd.Series]:
        """
        【V5.0 · 三维全息版】核心公理一：诊断筹码“聚散”的动态
        - 核心升级: 引入“三维全息”分析范式，将“状态”、“动态”、“势（上下文）”三者融合。
        - 新范式: 最终证据分 = (战术状态 * 战术动态 * 上下文状态 * 上下文动态) ^ 0.25。
        """
        scores = {}
        sorted_periods = sorted(periods)
        for i, p in enumerate(sorted_periods):
            context_p = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p
            # [代码修改开始] 引入状态、动态、上下文的三维融合
            # --- 看涨证据 ---
            bullish_evidence_static = df.get(f'concentration_increase_by_support_D', 0) + df.get(f'concentration_increase_by_chasing_D', 0)
            bullish_evidence_dynamic = df.get(f'concentration_increase_by_support_slope_{p}d_D', 0) + df.get(f'concentration_increase_by_chasing_slope_{p}d_D', 0)
            # 战术层 (p)
            tactical_bullish_static_score = normalize_score(bullish_evidence_static, df.index, p, ascending=True)
            tactical_bullish_dynamic_score = normalize_score(bullish_evidence_dynamic, df.index, p, ascending=True)
            # 上下文层 (context_p)
            context_bullish_static_score = normalize_score(bullish_evidence_static, df.index, context_p, ascending=True)
            context_bullish_dynamic_score = normalize_score(bullish_evidence_dynamic, df.index, context_p, ascending=True)
            # 三维融合
            tactical_bullish_quality = (tactical_bullish_static_score * tactical_bullish_dynamic_score * context_bullish_static_score * context_bullish_dynamic_score)**0.25
            # --- 看跌证据 ---
            bearish_evidence_static = df.get(f'concentration_decrease_by_distribution_D', 0) + df.get(f'concentration_decrease_by_capitulation_D', 0)
            bearish_evidence_dynamic = df.get(f'concentration_decrease_by_distribution_slope_{p}d_D', 0) + df.get(f'concentration_decrease_by_capitulation_slope_{p}d_D', 0)
            # 战术层 (p)
            tactical_bearish_static_score = normalize_score(bearish_evidence_static, df.index, p, ascending=True)
            tactical_bearish_dynamic_score = normalize_score(bearish_evidence_dynamic, df.index, p, ascending=True)
            # 上下文层 (context_p)
            context_bearish_static_score = normalize_score(bearish_evidence_static, df.index, context_p, ascending=True)
            context_bearish_dynamic_score = normalize_score(bearish_evidence_dynamic, df.index, context_p, ascending=True)
            # 三维融合
            tactical_bearish_quality = (tactical_bearish_static_score * tactical_bearish_dynamic_score * context_bearish_static_score * context_bearish_dynamic_score)**0.25
            # 生成双极快照分
            concentration_quality_snapshot = (tactical_bullish_quality - tactical_bearish_quality).astype(np.float32)
            # [代码修改结束]
            holographic_divergence = self._calculate_holographic_divergence(concentration_quality_snapshot, 1, p, p * 2)
            dynamic_concentration_score = self._perform_chip_relational_meta_analysis(
                df, concentration_quality_snapshot, p, holographic_divergence
            )
            scores[p] = dynamic_concentration_score
        return scores

    def _diagnose_main_force_action(self, df: pd.DataFrame, periods: list) -> Dict[int, pd.Series]:
        """
        【V5.0 · 三维全息版】核心公理二：诊断主力“吸筹与派发”
        - 核心升级: 引入“三维全息”分析范式，将“状态”、“动态”、“势（上下文）”三者融合。
        - 新范式: 最终证据分 = (战术状态 * 战术动态 * 上下文状态 * 上下文动态) ^ 0.25。
        """
        scores = {}
        sorted_periods = sorted(periods)
        for i, p in enumerate(sorted_periods):
            context_p = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p
            # [代码修改开始] 引入状态、动态、上下文的三维融合
            # --- 吸筹证据 ---
            accumulation_evidence_static = df.get('main_force_suppressive_accumulation_D', 0) + df.get('main_force_chasing_accumulation_D', 0)
            accumulation_evidence_dynamic = df.get(f'main_force_suppressive_accumulation_slope_{p}d_D', 0) + df.get(f'main_force_chasing_accumulation_slope_{p}d_D', 0)
            # 战术层
            tactical_accumulation_static_score = normalize_score(accumulation_evidence_static, df.index, p, ascending=True)
            tactical_accumulation_dynamic_score = normalize_score(accumulation_evidence_dynamic, df.index, p, ascending=True)
            # 上下文层
            context_accumulation_static_score = normalize_score(accumulation_evidence_static, df.index, context_p, ascending=True)
            context_accumulation_dynamic_score = normalize_score(accumulation_evidence_dynamic, df.index, context_p, ascending=True)
            # 三维融合
            accumulation_evidence = (tactical_accumulation_static_score * tactical_accumulation_dynamic_score * context_accumulation_static_score * context_accumulation_dynamic_score)**0.25
            # --- 派发证据 ---
            distribution_evidence_static = df.get('main_force_rally_distribution_D', 0) + df.get('main_force_capitulation_distribution_D', 0)
            distribution_evidence_dynamic = df.get(f'main_force_rally_distribution_slope_{p}d_D', 0) + df.get(f'main_force_capitulation_distribution_slope_{p}d_D', 0)
            # 战术层
            tactical_distribution_static_score = normalize_score(distribution_evidence_static, df.index, p, ascending=True)
            tactical_distribution_dynamic_score = normalize_score(distribution_evidence_dynamic, df.index, p, ascending=True)
            # 上下文层
            context_distribution_static_score = normalize_score(distribution_evidence_static, df.index, context_p, ascending=True)
            context_distribution_dynamic_score = normalize_score(distribution_evidence_dynamic, df.index, context_p, ascending=True)
            # 三维融合
            distribution_evidence = (tactical_distribution_static_score * tactical_distribution_dynamic_score * context_distribution_static_score * context_distribution_dynamic_score)**0.25
            # [代码修改结束]
            action_snapshot = (accumulation_evidence - distribution_evidence).astype(np.float32)
            holographic_divergence = self._calculate_holographic_divergence(action_snapshot, 1, p, p * 2)
            dynamic_action_score = self._perform_chip_relational_meta_analysis(
                df, action_snapshot, p, holographic_divergence
            )
            scores[p] = dynamic_action_score
        return scores

    def _diagnose_power_transfer(self, df: pd.DataFrame, periods: list) -> Dict[int, pd.Series]:
        """
        【V4.0 · 三维全息版】核心公理三：诊断筹码“转移方向”
        - 核心升级: 引入“三维全息”分析范式，将“状态”、“动态”、“势（上下文）”三者融合。
        - 新范式: 最终证据分 = (战术状态 * 战术动态 * 上下文状态 * 上下文动态) ^ 0.25。
        """
        scores = {}
        sorted_periods = sorted(periods)
        for i, p in enumerate(sorted_periods):
            context_p = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p
            # [代码修改开始] 引入状态、动态、上下文的三维融合
            # --- 向主力转移的证据 ---
            transfer_to_main_force_static = df.get('short_term_capitulation_ratio_D', 0) + df.get('long_term_despair_selling_ratio_D', 0)
            transfer_to_main_force_dynamic = df.get(f'short_term_capitulation_ratio_slope_{p}d_D', 0) + df.get(f'long_term_despair_selling_ratio_slope_{p}d_D', 0)
            # 战术层
            tactical_transfer_to_main_static_score = normalize_score(transfer_to_main_force_static, df.index, p, ascending=True)
            tactical_transfer_to_main_dynamic_score = normalize_score(transfer_to_main_force_dynamic, df.index, p, ascending=True)
            # 上下文层
            context_transfer_to_main_static_score = normalize_score(transfer_to_main_force_static, df.index, context_p, ascending=True)
            context_transfer_to_main_dynamic_score = normalize_score(transfer_to_main_force_dynamic, df.index, context_p, ascending=True)
            # 三维融合
            transfer_to_main_force_evidence = (tactical_transfer_to_main_static_score * tactical_transfer_to_main_dynamic_score * context_transfer_to_main_static_score * context_transfer_to_main_dynamic_score)**0.25
            # --- 向散户转移的证据 ---
            transfer_to_retail_static = df.get('short_term_profit_taking_ratio_D', 0) + df.get('long_term_chips_unlocked_ratio_D', 0)
            transfer_to_retail_dynamic = df.get(f'short_term_profit_taking_ratio_slope_{p}d_D', 0) + df.get(f'long_term_chips_unlocked_ratio_slope_{p}d_D', 0)
            # 战术层
            tactical_transfer_to_retail_static_score = normalize_score(transfer_to_retail_static, df.index, p, ascending=True)
            tactical_transfer_to_retail_dynamic_score = normalize_score(transfer_to_retail_dynamic, df.index, p, ascending=True)
            # 上下文层
            context_transfer_to_retail_static_score = normalize_score(transfer_to_retail_static, df.index, context_p, ascending=True)
            context_transfer_to_retail_dynamic_score = normalize_score(transfer_to_retail_dynamic, df.index, context_p, ascending=True)
            # 三维融合
            transfer_to_retail_evidence = (tactical_transfer_to_retail_static_score * tactical_transfer_to_retail_dynamic_score * context_transfer_to_retail_static_score * context_transfer_to_retail_dynamic_score)**0.25
            # [代码修改结束]
            transfer_snapshot = (transfer_to_main_force_evidence - transfer_to_retail_evidence).astype(np.float32)
            holographic_divergence = self._calculate_holographic_divergence(transfer_snapshot, 1, p, p * 2)
            dynamic_transfer_score = self._perform_chip_relational_meta_analysis(
                df, transfer_snapshot, p, holographic_divergence
            )
            scores[p] = dynamic_transfer_score
        return scores

    def _perform_chip_relational_meta_analysis(self, df: pd.DataFrame, snapshot_score: pd.Series, meta_window: int, holographic_divergence_score: pd.Series) -> pd.Series:
        """
        【V6.0 · 阿瑞斯之怒协议版】筹码专用的关系元分析核心引擎
        - 核心革命: 废除“冥王之眼”乘法模型，全面升级为与微观行为引擎一致的“阿瑞斯之怒”加法模型。
                      最终得分 = (状态*权重) + (速度*权重) + (加速度*权重) + (背离*权重)
        - 升级意义: 新模型更侧重于动态变化，即使状态分较低，只要速度、加速度或背离足够强，也能产生高分，
                      从而更敏锐地捕捉到趋势的“拐点”。
        """
        # 全面升级为“阿瑞斯之怒”加法模型
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        p_meta = get_param_value(p_conf.get('relational_meta_analysis_params'), {})
        # 新的加法模型权重
        w_state = get_param_value(p_meta.get('state_weight'), 0.2) # 降低状态权重
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.3)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.3)
        w_holographic = get_param_value(p_meta.get('holographic_weight'), 0.2) # 将背离分作为第四维度
        norm_window = 55
        bipolar_sensitivity = 1.0
        # 维度一：状态分 (State Score) - 范围 [0, 1]
        state_score = snapshot_score.clip(0, 1)
        # 维度二：速度分 (Velocity Score) - 范围 [-1, 1]
        relationship_trend = snapshot_score.diff(meta_window).fillna(0)
        velocity_score = normalize_to_bipolar(relationship_trend, df.index, norm_window, bipolar_sensitivity)
        # 维度三：加速度分 (Acceleration Score) - 范围 [-1, 1]
        relationship_accel = relationship_trend.diff(meta_window).fillna(0)
        acceleration_score = normalize_to_bipolar(relationship_accel, df.index, norm_window, bipolar_sensitivity)
        # 维度四：全息背离分 (Holographic Divergence Score) - 范围 [-1, 1]
        # 确保背离分也是双极性的
        holographic_score = holographic_divergence_score.clip(-1, 1)
        # 终极融合：从乘法调制升级为四维加法赋权
        final_score = (
            state_score * w_state +
            velocity_score.clip(0, 1) * w_velocity + # 看涨信号只取正向速度
            acceleration_score.clip(0, 1) * w_acceleration + # 看涨信号只取正向加速度
            holographic_score.clip(0, 1) * w_holographic # 看涨信号只取正向背离
        ).clip(0, 1)
        
        return final_score.astype(np.float32)

    def _calculate_holographic_divergence(self, series: pd.Series, short_p: int, long_p: int, norm_window: int) -> pd.Series:
        """
        【V1.0 · 新增 · 冥王之眼】全息背离计算引擎
        - 战略意义: 洞察多时间维度的“结构性背离”，输出一个[-1, 1]的双极性背离分数。
        - 正分: 看涨背离 (短期趋势强于长期趋势)。
        - 负分: 看跌背离 (短期趋势弱于长期趋势)。
        """
        # 新增方法
        # 维度一：速度背离 (短期斜率 vs 长期斜率)
        slope_short = series.diff(short_p).fillna(0)
        slope_long = series.diff(long_p).fillna(0)
        velocity_divergence = slope_short - slope_long
        velocity_divergence_score = normalize_to_bipolar(velocity_divergence, series.index, norm_window)
        # 维度二：加速度背离 (短期加速度 vs 长期加速度)
        accel_short = slope_short.diff(short_p).fillna(0)
        accel_long = slope_long.diff(long_p).fillna(0)
        acceleration_divergence = accel_short - accel_long
        acceleration_divergence_score = normalize_to_bipolar(acceleration_divergence, series.index, norm_window)
        # 融合：速度背离和加速度背离的加权平均
        final_divergence_score = (velocity_divergence_score * 0.6 + acceleration_divergence_score * 0.4).clip(-1, 1)
        return final_divergence_score.astype(np.float32)

    def _calculate_ma_trend_context(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V1.1 · 赫尔墨斯之翼优化版】计算均线趋势上下文分数
        - 性能优化: 全程使用Numpy数组进行计算，避免了多个中间Pandas Series的创建和开销，
                      显著提升了计算速度和内存效率。
        - 核心逻辑: 保持“均线排列健康度 * 价格位置健康度”的融合逻辑不变。
        """
        ma_cols = [f'EMA_{p}_D' for p in periods]
        if not all(col in df.columns for col in ma_cols):
            return pd.Series(0.5, index=df.index, dtype=np.float32)

        # 将所有需要的Series一次性转换为Numpy数组
        ma_values = np.stack([df[col].values for col in ma_cols], axis=0)
        close_values = df['close_D'].values

        # 1. 计算均线排列健康度 (Alignment Health)
        # 比较相邻均线的大小关系 (short > long)，结果为布尔数组
        alignment_bools = ma_values[:-1] > ma_values[1:]
        # 沿均线轴计算看涨排列的比例
        alignment_health = np.mean(alignment_bools, axis=0)

        # 2. 计算价格位置健康度 (Position Health)
        # 比较收盘价与所有均线的大小关系 (close > ma)
        position_bools = close_values > ma_values
        # 沿均线轴计算价格在均线上方的比例
        position_health = np.mean(position_bools, axis=0)

        # 3. 融合得到最终的趋势上下文分数
        # 使用Numpy进行高效的几何平均计算
        ma_context_score_values = np.sqrt(alignment_health * position_health)
        
        return pd.Series(ma_context_score_values, index=df.index, dtype=np.float32)

    # ==============================================================================
    # 以下为保留的、具有特殊战术意义的“剧本”诊断模块
    # ==============================================================================

    def diagnose_accumulation_playbooks(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.0 · 归因重构版】诊断“吸筹”相关的战术剧本
        - 核心升级: 重新定义“拉升吸筹”为“主力未派发的散户追涨”，更精确地捕捉主升浪初期的关键特征。
        """
        states = {}
        periods = [1, 5, 13, 21, 55]
        sorted_periods = sorted(periods)
        rally_scores_by_period = {}
        for i, p_tactical in enumerate(sorted_periods):
            p_context = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p_tactical
            # [代码修改开始] 使用新的归因指标重构“拉升吸筹”逻辑
            # 战术层
            tactical_retail_chasing = normalize_score(df.get('retail_chasing_accumulation_D', 0), df.index, p_tactical, ascending=True)
            tactical_main_force_not_distributing = 1.0 - normalize_score(df.get('main_force_rally_distribution_D', 0), df.index, p_tactical, ascending=True)
            # 上下文层
            context_retail_chasing = normalize_score(df.get('retail_chasing_accumulation_D', 0), df.index, p_context, ascending=True)
            context_main_force_not_distributing = 1.0 - normalize_score(df.get('main_force_rally_distribution_D', 0), df.index, p_context, ascending=True)
            # [代码修改结束]
            fused_retail_chasing = (tactical_retail_chasing * context_retail_chasing)**0.5
            fused_main_force_not_distributing = (tactical_main_force_not_distributing * context_main_force_not_distributing)**0.5
            rally_snapshot_score = (fused_retail_chasing * fused_main_force_not_distributing)**0.5
            holographic_divergence = self._calculate_holographic_divergence(rally_snapshot_score, p_tactical, p_context, p_context * 2)
            rally_scores_by_period[p_tactical] = self._perform_chip_relational_meta_analysis(df, rally_snapshot_score, p_tactical, holographic_divergence)
        tf_weights = {1: 0.1, 5: 0.4, 13: 0.3, 21: 0.15, 55: 0.05}
        final_fused_score = pd.Series(0.0, index=df.index)
        total_weight = sum(tf_weights.get(p, 0) for p in periods)
        if total_weight > 0:
            for p_tactical in periods:
                weight = tf_weights.get(p_tactical, 0) / total_weight
                final_fused_score += rally_scores_by_period.get(p_tactical, 0.0) * weight
        # [代码新增开始] 融合新的“真实吸筹”信号
        suppressive_accumulation = normalize_score(df.get('main_force_suppressive_accumulation_D', 0), df.index, 55, ascending=True)
        true_accumulation_score = np.maximum(final_fused_score, suppressive_accumulation)
        states['SCORE_CHIP_TRUE_ACCUMULATION'] = true_accumulation_score.clip(0, 1).astype(np.float32)
        # [代码新增结束]
        states['SCORE_CHIP_PB_RALLY_ACCUMULATION'] = final_fused_score.clip(0, 1).astype(np.float32)
        return states

    def diagnose_capitulation_reversal_potential(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V4.0 · 分层印证版】诊断“恐慌投降反转”的潜力
        - 核心升级: 引入“分层动态印证”框架。对“深度套牢”、“低位价格”、“输家换手”三大核心证据进行多时间维度的分层验证。
        """
        states = {}
        required_cols = ['total_loser_rate_D', 'close_D', 'turnover_from_losers_ratio_D']
        if any(col not in df.columns for col in required_cols):
            states['SCORE_CHIP_CONTEXT_CAPITULATION_POTENTIAL'] = pd.Series(0.0, index=df.index)
            return states
        # 引入分层印证框架
        periods = [5, 13, 21, 55] # 恐慌信号不宜使用过短周期
        sorted_periods = sorted(periods)
        capitulation_scores_by_period = {}
        bearish_ma_context = 1 - self._calculate_ma_trend_context(df, [5, 13, 21, 55])
        for i, p_tactical in enumerate(sorted_periods):
            p_context = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p_tactical
            # 战术层
            tactical_deep_cap = normalize_score(df['total_loser_rate_D'], df.index, p_tactical, ascending=True)
            tactical_price_lows = 1.0 - normalize_score(df['close_D'], df.index, window=p_tactical, ascending=True)
            tactical_loser_turnover = normalize_score(df['turnover_from_losers_ratio_D'], df.index, p_tactical, ascending=True)
            # 上下文层
            context_deep_cap = normalize_score(df['total_loser_rate_D'], df.index, p_context, ascending=True)
            context_price_lows = 1.0 - normalize_score(df['close_D'], df.index, window=p_context, ascending=True)
            context_loser_turnover = normalize_score(df['turnover_from_losers_ratio_D'], df.index, p_context, ascending=True)
            # 融合
            fused_deep_cap = (tactical_deep_cap * context_deep_cap)**0.5
            fused_price_lows = (tactical_price_lows * context_price_lows)**0.5
            fused_loser_turnover = (tactical_loser_turnover * context_loser_turnover)**0.5
            # 生成快照分
            snapshot_score = (fused_deep_cap * fused_price_lows * fused_loser_turnover * bearish_ma_context).astype(np.float32)
            # 为快照分计算其结构性背离
            holographic_divergence = self._calculate_holographic_divergence(snapshot_score, p_tactical, p_context, p_context * 2)
            # 对每个周期的快照分进行元分析
            capitulation_scores_by_period[p_tactical] = self._perform_chip_relational_meta_analysis(df, snapshot_score, p_tactical, holographic_divergence)
        # 跨周期融合
        tf_weights = {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}
        final_fused_score = pd.Series(0.0, index=df.index)
        total_weight = sum(tf_weights.get(p, 0) for p in periods)
        if total_weight > 0:
            for p_tactical in periods:
                weight = tf_weights.get(p_tactical, 0) / total_weight
                final_fused_score += capitulation_scores_by_period.get(p_tactical, 0.0) * weight
        states['SCORE_CHIP_CONTEXT_CAPITULATION_POTENTIAL'] = final_fused_score.clip(0, 1).astype(np.float32)
        
        return states


