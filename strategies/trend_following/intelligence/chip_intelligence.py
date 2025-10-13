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
        【V601.0 · 全息共振版】筹码情报最高司令部
        - 核心升级: 引入多时间级别(5,13,21,55)分析，构建“全息共振引擎”。
        - 指挥流程: 1. 为每个核心公理，在所有时间级别上进行诊断，生成多维度的动态分。
                      2. 在终极信号合成器中，对多维度的分数进行“周期内融合”和“跨周期共振”两步锻造。
        """
        # 修改开始: 引入多时间级别分析
        all_chip_states = {}
        periods = [5, 13, 21, 55] # 定义分析的时间级别
        # 步骤 1: 诊断三大核心公理，生成多维度的核心动态分
        concentration_scores = self._diagnose_concentration_dynamics(df, periods)
        all_chip_states['SCORE_CHIP_MTF_CONCENTRATION'] = concentration_scores
        accumulation_scores = self._diagnose_main_force_action(df, periods)
        all_chip_states['SCORE_CHIP_MTF_ACCUMULATION'] = accumulation_scores
        power_transfer_scores = self._diagnose_power_transfer(df, periods)
        all_chip_states['SCORE_CHIP_MTF_POWER_TRANSFER'] = power_transfer_scores
        # 步骤 2: 基于多维核心公理，合成具备共振效应的终极信号
        ultimate_signals = self._synthesize_ultimate_signals(
            concentration_scores,
            accumulation_scores,
            power_transfer_scores
        )
        all_chip_states.update(ultimate_signals)
        # 步骤 3: 保留独立的、具有特殊战术意义的剧本诊断 (不受影响)
        accumulation_potential_states = self.diagnose_accumulation_playbooks(df)
        all_chip_states.update(accumulation_potential_states)
        capitulation_potential_states = self.diagnose_capitulation_reversal_potential(df)
        all_chip_states.update(capitulation_potential_states)
        return all_chip_states
        # 修改结束

    def _synthesize_ultimate_signals(self, concentration: Dict[int, pd.Series], accumulation: Dict[int, pd.Series], power_transfer: Dict[int, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.1 · 全息共振版】终极信号合成器 (共振熔炉)
        - 核心升级: 接收多维度的核心公理字典，执行“周期内融合”与“跨周期共振”两步锻造。
        """
        # 修改开始: 升级为两步融合逻辑
        states = {}
        periods = sorted(concentration.keys())
        # 定义跨周期融合的权重
        tf_weights = {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1} # 短期动态更重要
        # --- 看涨共振 (Bullish Resonance) ---
        bullish_scores_by_period = {}
        for p in periods:
            # 步骤一：周期内融合
            score = (concentration[p] * accumulation[p].clip(0, 1) * power_transfer[p].clip(0, 1))**(1/3)
            bullish_scores_by_period[p] = score
        # 步骤二：跨周期共振 (加权几何平均)
        bullish_resonance = pd.Series(1.0, index=self.strategy.df_indicators.index)
        total_weight = sum(tf_weights.get(p, 0) for p in periods)
        for p in periods:
            weight = tf_weights.get(p, 0) / total_weight
            bullish_resonance *= (bullish_scores_by_period[p] ** weight)
        states['SCORE_CHIP_BULLISH_RESONANCE'] = bullish_resonance.fillna(0).astype(np.float32)
        # --- 看跌共振 (Bearish Resonance) ---
        bearish_scores_by_period = {}
        for p in periods:
            # 步骤一：周期内融合
            score = ((1 - concentration[p]) * abs(accumulation[p].clip(-1, 0)) * abs(power_transfer[p].clip(-1, 0)))**(1/3)
            bearish_scores_by_period[p] = score
        # 步骤二：跨周期共振 (加权几何平均)
        bearish_resonance = pd.Series(1.0, index=self.strategy.df_indicators.index)
        for p in periods:
            weight = tf_weights.get(p, 0) / total_weight
            bearish_resonance *= (bearish_scores_by_period[p] ** weight)
        states['SCORE_CHIP_BEARISH_RESONANCE'] = bearish_resonance.fillna(0).astype(np.float32)
        # --- 底部/顶部反转信号 (基于共振信号的加速度) ---
        bottom_reversal = self._perform_chip_relational_meta_analysis(self.strategy.df_indicators, bullish_resonance)
        states['SCORE_CHIP_BOTTOM_REVERSAL'] = bottom_reversal.astype(np.float32)
        top_reversal = self._perform_chip_relational_meta_analysis(self.strategy.df_indicators, bearish_resonance)
        states['SCORE_CHIP_TOP_REVERSAL'] = top_reversal.astype(np.float32)
        # --- 战术反转 (Tactical Reversal) ---
        tactical_reversal = (bullish_resonance * 0.5).astype(np.float32)
        states['SCORE_CHIP_TACTICAL_REVERSAL'] = tactical_reversal
        return states
        # 修改结束

    def _diagnose_concentration_dynamics(self, df: pd.DataFrame, periods: list) -> Dict[int, pd.Series]:
        """
        【V1.2 · 元分析贯穿版】核心公理一：诊断筹码“聚散”的动态
        - 核心修正: 确保输入给元分析引擎的“快照分”是纯粹的静态指标融合，移除了斜率指标，
                      让动态分析完全由元分析引擎负责。
        """
        # 修改开始: 净化快照分，使其成为纯粹的静态指标融合
        scores = {}
        norm_window = 120
        # 1. 计算纯粹的静态“集中度快照分”
        # 融合了“90%集中度”、“70%集中度”和“单峰稳定性”，全面评估筹码的聚集状态
        conc_90_score = normalize_score(df.get('concentration_90pct_D'), df.index, norm_window, ascending=False)
        conc_70_score = normalize_score(df.get('concentration_70pct_D'), df.index, norm_window, ascending=False)
        # 使用静态的稳定性指标，而非其斜率
        stability_score = normalize_score(df.get('peak_stability_D'), df.index, norm_window, ascending=True)
        # 使用几何平均融合，确保所有因子都有效
        concentration_snapshot = (conc_90_score * conc_70_score * stability_score)**(1/3)
        # 2. 对快照分进行关系元分析，捕捉其动态变化
        # 注意：由于快照分不依赖于周期 p，所以所有周期的动态分析结果是相同的。
        # 这是符合逻辑的，因为“集中度”本身是一个全局状态，其动态变化不应因观察窗口而异。
        dynamic_concentration_score = self._perform_chip_relational_meta_analysis(df, concentration_snapshot)
        for p in periods:
            scores[p] = dynamic_concentration_score
        return scores
        # 修改结束

    def _diagnose_main_force_action(self, df: pd.DataFrame, periods: list) -> Dict[int, pd.Series]:
        """
        【V1.3 · 元分析贯穿版】核心公理二：诊断主力“吸筹与派发”
        - 核心重构: 将“吸筹”和“派发”证据融合为一个双极性的“行动快照分”，然后将此快照分送入
                      关系元分析引擎，从而对“主力行动”本身进行三维动态分析。
        """
        # 修改开始: 部署元分析贯穿协议
        scores = {}
        norm_window = 120
        for p in periods:
            # 步骤 1: 构建“吸筹”证据链
            conc_slope_up = normalize_score(df.get(f'SLOPE_{p}_concentration_90pct_D'), df.index, norm_window, ascending=True)
            winner_turnover_down = normalize_score(df.get(f'SLOPE_{p}_turnover_from_winners_ratio_D'), df.index, norm_window, ascending=False)
            trade_conc_up = normalize_score(df.get(f'SLOPE_{p}_trade_concentration_index_D'), df.index, norm_window, ascending=True)
            accumulation_evidence = (conc_slope_up * winner_turnover_down * trade_conc_up)**(1/3)
            # 步骤 2: 构建“派发”证据链
            conc_slope_down = normalize_score(df.get(f'SLOPE_{p}_concentration_90pct_D'), df.index, norm_window, ascending=False)
            winner_turnover_up = normalize_score(df.get(f'SLOPE_{p}_turnover_from_winners_ratio_D'), df.index, norm_window, ascending=True)
            trade_conc_down = normalize_score(df.get(f'SLOPE_{p}_trade_concentration_index_D'), df.index, norm_window, ascending=False)
            distribution_evidence = (conc_slope_down * winner_turnover_up * trade_conc_down)**(1/3)
            # 步骤 3: 生成双极性的“行动快照分”
            action_snapshot = (accumulation_evidence - distribution_evidence).astype(np.float32)
            # 步骤 4: 对“行动快照分”进行关系元分析，得到最终的动态分数
            dynamic_action_score = self._perform_chip_relational_meta_analysis(df, action_snapshot)
            scores[p] = dynamic_action_score
        return scores
        # 修改结束

    def _diagnose_power_transfer(self, df: pd.DataFrame, periods: list) -> Dict[int, pd.Series]:
        """
        【V1.3 · 元分析贯穿版】核心公理三：诊断筹码“转移方向”
        - 核心重构: 将“向主力转移”和“向散户转移”的证据融合为一个双极性的“转移快照分”，然后将此快照分
                      送入关系元分析引擎，从而对“权力转移”本身进行三维动态分析。
        """
        # 修改开始: 部署元分析贯穿协议
        scores = {}
        norm_window = 120
        for p in periods:
            # 步骤 1: 构建“筹码向主力转移”的证据
            cost_divergence_score = normalize_score(df.get(f'SLOPE_{p}_cost_divergence_D'), df.index, norm_window, ascending=True)
            loser_turnover_up = normalize_score(df.get(f'SLOPE_{p}_turnover_from_losers_ratio_D'), df.index, norm_window, ascending=True)
            transfer_to_main_force_evidence = (cost_divergence_score * loser_turnover_up)**0.5
            # 步骤 2: 构建“筹码向散户转移”的证据
            cost_convergence_score = normalize_score(df.get(f'SLOPE_{p}_cost_divergence_D'), df.index, norm_window, ascending=False)
            loser_turnover_down = normalize_score(df.get(f'SLOPE_{p}_turnover_from_losers_ratio_D'), df.index, norm_window, ascending=False)
            transfer_to_retail_evidence = (cost_convergence_score * loser_turnover_down)**0.5
            # 步骤 3: 生成双极性的“转移快照分”
            transfer_snapshot = (transfer_to_main_force_evidence - transfer_to_retail_evidence).astype(np.float32)
            # 步骤 4: 对“转移快照分”进行关系元分析，得到最终的动态分数
            dynamic_transfer_score = self._perform_chip_relational_meta_analysis(df, transfer_snapshot)
            scores[p] = dynamic_transfer_score
        return scores
        # 修改结束

    def _perform_chip_relational_meta_analysis(self, df: pd.DataFrame, snapshot_score: pd.Series) -> pd.Series:
        """
        【V2.0 · 阿瑞斯之怒协议版】筹码专用的关系元分析核心引擎
        - 核心革命: 响应“重变化、轻状态”的哲学，从“状态 * (1 + 动态)”的乘法模型，升级为
                      “(状态*权重) + (速度*权重) + (加速度*权重)”的加法模型。
        - 核心目标: 即使静态分很低，只要动态（尤其是加速度）足够强，也能产生高分，真正捕捉“拐点”。
        """
        # 引入新的权重体系和加法融合模型
        # 从配置中获取新的加法模型权重
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        p_meta = get_param_value(p_conf.get('relational_meta_analysis_params'), {})
        # 新的权重体系，直接作用于最终分数，而非杠杆
        w_state = get_param_value(p_meta.get('state_weight'), 0.3)
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.3)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.4) # 赋予加速度最高权重
        # 核心参数
        norm_window = 55
        meta_window = 5
        bipolar_sensitivity = 1.0
        # 第一维度：状态分 (State Score) - 范围 [0, 1]
        state_score = snapshot_score.clip(0, 1)
        # 第二维度：速度分 (Velocity Score) - 范围 [-1, 1]
        relationship_trend = snapshot_score.diff(meta_window).fillna(0)
        velocity_score = normalize_to_bipolar(
            series=relationship_trend, target_index=df.index,
            window=norm_window, sensitivity=bipolar_sensitivity
        )
        # 第三维度：加速度分 (Acceleration Score) - 范围 [-1, 1]
        relationship_accel = relationship_trend.diff(meta_window).fillna(0)
        acceleration_score = normalize_to_bipolar(
            series=relationship_accel, target_index=df.index,
            window=norm_window, sensitivity=bipolar_sensitivity
        )
        # 终极融合：从乘法调制升级为加法赋权
        # 旧的乘法模型: dynamic_leverage = 1 + (velocity_score * w_velocity) + (acceleration_score * w_acceleration)
        # 旧的乘法模型: final_score = (state_score * dynamic_leverage).clip(0, 1)
        # 新的加法模型:
        final_score = (
            state_score * w_state +
            velocity_score * w_velocity +
            acceleration_score * w_acceleration
        ).clip(0, 1) # clip确保分数在[0, 1]范围内
        return final_score.astype(np.float32)

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
        【V5.1 · 赫尔墨斯之翼优化版】主力吸筹模式与风险诊断引擎
        - 性能优化: 确保所有中间和最终的Series都显式转换为float32。
        - 核心逻辑: 保持基于“关系元分析”的拉升吸筹与打压吸筹的诊断范式不变。
        """
        states = {}
        norm_window = 120
        
        # 获取均线趋势上下文，作为判断拉升或打压的背景
        ma_context_score = self._calculate_ma_trend_context(df, [5, 13, 21, 55])

        # --- 拉升吸筹 (Rally Accumulation) ---
        # 核心关系：在上升趋势中(ma_context高)，筹码依然在集中，且获利盘惜售。
        chip_concentration_score = normalize_score(df.get('concentration_90pct_D'), df.index, norm_window, ascending=False)
        winner_conviction_score = normalize_score(df.get('turnover_from_winners_ratio_D'), df.index, norm_window, ascending=False)
        rally_snapshot_score = (ma_context_score * chip_concentration_score * winner_conviction_score)
        rally_accumulation_score = self._perform_chip_relational_meta_analysis(df, rally_snapshot_score)
        states['SCORE_CHIP_PLAYBOOK_RALLY_ACCUMULATION'] = rally_accumulation_score # 元分析函数已确保是float32

        # --- 打压吸筹 (Suppress Accumulation) ---
        # 核心关系：在下跌或盘整趋势中(ma_context低)，筹码逆势集中。
        price_weakness_score = 1.0 - ma_context_score
        suppress_snapshot_score = (price_weakness_score * chip_concentration_score)
        suppress_accumulation_score = self._perform_chip_relational_meta_analysis(df, suppress_snapshot_score)
        states['SCORE_CHIP_PLAYBOOK_SUPPRESS_ACCUMULATION'] = suppress_accumulation_score # 元分析函数已确保是float32
        
        # --- 真实吸筹 (True Accumulation) ---
        # 融合两种吸筹信号，取更强的一种作为最终的真实吸筹信号
        # 使用Numpy.maximum进行高效融合
        true_accumulation_score = np.maximum(rally_accumulation_score.values, suppress_accumulation_score.values)
        states['SCORE_CHIP_TRUE_ACCUMULATION'] = pd.Series(true_accumulation_score, index=df.index, dtype=np.float32)
        
        return states

    def diagnose_capitulation_reversal_potential(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.0 · 权责净化版】诊断“恐慌投降反转”的潜力
        - 核心革命: 不再生成剧本信号，而是生成一个更底层的“潜力/上下文”信号，供上层战术引擎消费。
        """
        states = {}
        p = get_params_block(self.strategy, 'capitulation_reversal_params', {})
        norm_window = get_param_value(p.get('norm_window'), 120)
        
        required_cols = ['total_loser_rate_D', 'close_D', 'turnover_from_losers_ratio_D']
        if any(col not in df.columns for col in required_cols):
            return states

        # 步骤一：构建“恐慌投降关系”的瞬时快照分 (逻辑不变)
        deep_capitulation_score = normalize_score(df['total_loser_rate_D'], df.index, norm_window, ascending=True)
        price_at_lows_score = 1.0 - normalize_score(df['close_D'], df.index, window=250, ascending=True)
        loser_turnover_score = normalize_score(df['turnover_from_losers_ratio_D'], df.index, norm_window, ascending=True)
        bearish_ma_context = 1 - self._calculate_ma_trend_context(df, [5, 13, 21, 55])
        snapshot_score = (deep_capitulation_score * price_at_lows_score * loser_turnover_score * bearish_ma_context).astype(np.float32)

        # 步骤二：对“恐慌投降关系”进行元分析 (逻辑不变)
        final_score = self._perform_chip_relational_meta_analysis(df, snapshot_score)
        
        # 输出信号的命名和语义发生根本性改变：从“剧本”降级为“潜力”
        states['SCORE_CHIP_CONTEXT_CAPITULATION_POTENTIAL'] = final_score
        return states

