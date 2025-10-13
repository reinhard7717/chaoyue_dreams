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
        # 引入多时间级别分析
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

    def _synthesize_ultimate_signals(self, concentration: Dict[int, pd.Series], accumulation: Dict[int, pd.Series], power_transfer: Dict[int, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.5 · 冥王之眼版】终极信号合成器
        - 核心升级: 重新定义“反转”信号。
                      - 底部反转 = “看涨共振”信号发生看涨背离 (短期走强)。
                      - 顶部反转 = “看跌共振”信号发生看涨背离 (短期加速恶化)。
        """
        # 使用全息背离引擎重新定义反转
        states = {}
        periods = sorted(concentration.keys())
        tf_weights = {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}
        norm_window = 55
        # --- 看涨/看跌共振 (逻辑不变) ---
        bullish_scores_by_period = {}
        for p in periods:
            score = (concentration[p] + accumulation[p] + power_transfer[p]) / 3.0
            bullish_scores_by_period[p] = score
        bullish_resonance = pd.Series(0.0, index=self.strategy.df_indicators.index)
        total_weight = sum(tf_weights.get(p, 0) for p in periods)
        if total_weight > 0:
            for p in periods:
                weight = tf_weights.get(p, 0) / total_weight
                bullish_resonance += bullish_scores_by_period[p] * weight
        states['SCORE_CHIP_BULLISH_RESONANCE'] = bullish_resonance.fillna(0).clip(0,1).astype(np.float32)
        bearish_scores_by_period = {}
        for p in periods:
            score = ((1 - concentration[p]) + (1 - accumulation[p]) + (1 - power_transfer[p])) / 3.0
            bearish_scores_by_period[p] = score
        bearish_resonance = pd.Series(0.0, index=self.strategy.df_indicators.index)
        if total_weight > 0:
            for p in periods:
                weight = tf_weights.get(p, 0) / total_weight
                bearish_resonance += bearish_scores_by_period[p] * weight
        states['SCORE_CHIP_BEARISH_RESONANCE'] = bearish_resonance.fillna(0).clip(0,1).astype(np.float32)
        # --- 底部/顶部反转信号 (基于共振信号的结构性背离) ---
        # 底部反转 = 看涨共振信号的看涨背离 (短期5日 vs 长期21日)
        bottom_reversal_divergence = self._calculate_holographic_divergence(bullish_resonance, 5, 21, norm_window)
        states['SCORE_CHIP_BOTTOM_REVERSAL'] = bottom_reversal_divergence.clip(0, 1).astype(np.float32) # 只取看涨背离部分
        # 顶部反转 = 看跌共振信号的看涨背离 (即看跌趋势在加速恶化)
        top_reversal_divergence = self._calculate_holographic_divergence(bearish_resonance, 5, 21, norm_window)
        states['SCORE_CHIP_TOP_REVERSAL'] = top_reversal_divergence.clip(0, 1).astype(np.float32) # 只取看涨背离部分
        # --- 战术反转 (Tactical Reversal) ---
        tactical_reversal = (bullish_resonance * 0.5).astype(np.float32)
        states['SCORE_CHIP_TACTICAL_REVERSAL'] = tactical_reversal
        return states


    def _diagnose_concentration_dynamics(self, df: pd.DataFrame, periods: list) -> Dict[int, pd.Series]:
        """
        【V1.7 · 冥王之眼版】核心公理一：诊断筹码“聚散”的动态
        - 核心升级: 调用“冥王之眼”引擎，计算快照分在(1, p)周期上的结构性背离。
        """
        # 计算并传入结构性背离分
        scores = {}
        norm_window = 120
        # 1. 计算静态快照分
        conc_90_score = normalize_score(df.get('concentration_90pct_D'), df.index, norm_window, ascending=False)
        conc_70_score = normalize_score(df.get('concentration_70pct_D'), df.index, norm_window, ascending=False)
        stability_score = normalize_score(df.get('peak_stability_D'), df.index, norm_window, ascending=True)
        concentration_snapshot = (conc_90_score * conc_70_score * stability_score)**(1/3)
        # 2. 对快照分进行关系元分析
        for p in periods:
            # 计算当前周期p与最短周期1之间的结构性背离
            holographic_divergence = self._calculate_holographic_divergence(concentration_snapshot, 1, p, norm_window)
            dynamic_concentration_score = self._perform_chip_relational_meta_analysis(
                df, concentration_snapshot, p, holographic_divergence
            )
            scores[p] = dynamic_concentration_score
        return scores


    def _diagnose_main_force_action(self, df: pd.DataFrame, periods: list) -> Dict[int, pd.Series]:
        """
        【V1.6 · 冥王之眼版】核心公理二：诊断主力“吸筹与派发”
        - 核心升级: 调用“冥王之眼”引擎，为每个周期的行动快照分计算其在(1, p)周期上的结构性背离，
                      并将这个关键的背离分送入元分析引擎。
        """
        # 计算并传入结构性背离分
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
            # 步骤 4 (新增): 计算当前周期p与最短周期1之间的结构性背离
            holographic_divergence = self._calculate_holographic_divergence(action_snapshot, 1, p, norm_window)
            # 步骤 5 (升级): 对“行动快照分”进行关系元分析，传入背离分
            dynamic_action_score = self._perform_chip_relational_meta_analysis(
                df, action_snapshot, p, holographic_divergence
            )
            scores[p] = dynamic_action_score
        return scores

    def _diagnose_power_transfer(self, df: pd.DataFrame, periods: list) -> Dict[int, pd.Series]:
        """
        【V1.6 · 冥王之眼版】核心公理三：诊断筹码“转移方向”
        - 核心升级: 调用“冥王之眼”引擎，为每个周期的转移快照分计算其在(1, p)周期上的结构性背离，
                      并将这个关键的背离分送入元分析引擎。
        """
        # 计算并传入结构性背离分
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
            # 步骤 4 (新增): 计算当前周期p与最短周期1之间的结构性背离
            holographic_divergence = self._calculate_holographic_divergence(transfer_snapshot, 1, p, norm_window)
            # 步骤 5 (升级): 对“转移快照分”进行关系元分析，传入背离分
            dynamic_transfer_score = self._perform_chip_relational_meta_analysis(
                df, transfer_snapshot, p, holographic_divergence
            )
            scores[p] = dynamic_transfer_score
        return scores

    def _perform_chip_relational_meta_analysis(self, df: pd.DataFrame, snapshot_score: pd.Series, meta_window: int, holographic_divergence_score: pd.Series) -> pd.Series:
        """
        【V5.0 · 冥王之眼版】筹码专用的关系元分析核心引擎
        - 核心革命: 签署“冥王之眼”协议，引入双极性的“背离杠杆”。
                      最终得分 = 基础分 * (1 + 一阶动态杠杆) * (1 + 背离杠杆)
        """
        # 消化双极性背离分
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        p_meta = get_param_value(p_conf.get('relational_meta_analysis_params'), {})
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.3)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.4)
        w_holographic = get_param_value(p_meta.get('holographic_weight'), 0.5)
        norm_window = 55
        bipolar_sensitivity = 1.0
        # 维度一：状态分
        state_score = snapshot_score.clip(0, 1)
        # 维度二：一阶动态 (速度与加速度)
        relationship_trend = snapshot_score.diff(meta_window).fillna(0)
        velocity_score = normalize_to_bipolar(relationship_trend, df.index, norm_window, bipolar_sensitivity)
        relationship_accel = relationship_trend.diff(meta_window).fillna(0)
        acceleration_score = normalize_to_bipolar(relationship_accel, df.index, norm_window, bipolar_sensitivity)
        # 杠杆一：一阶动态杠杆
        dynamic_leverage = 1 + (velocity_score * w_velocity) + (acceleration_score * w_acceleration)
        # 杠杆二：背离杠杆 (holographic_divergence_score 是 [-1, 1] 的双极性分数)
        holographic_leverage = 1 + (holographic_divergence_score * w_holographic)
        # 冥王之眼：基础分被双重杠杆撬动
        final_score = (state_score * dynamic_leverage * holographic_leverage).clip(0, 1)
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
        【V1.2 · 冥王之眼同步版】诊断“吸筹”相关的战术剧本
        - 核心修复: 在调用元分析引擎时，补上缺失的“全息背离分”参数，与“冥王之眼”协议保持一致。
        """
        # 修改开始: 补上缺失的“全息背离分”参数
        states = {}
        norm_window = 120
        # 剧本一：“拉升吸筹” (Rally Accumulation)
        # 证据链：价格上涨，但换手率下降，且筹码集中度在提升
        price_up_score = normalize_score(df.get('SLOPE_5_close'), df.index, norm_window, ascending=True)
        turnover_down_score = normalize_score(df.get('SLOPE_5_turnover_rate_f'), df.index, norm_window, ascending=False)
        concentration_up_score = normalize_score(df.get('SLOPE_5_concentration_90pct_D'), df.index, norm_window, ascending=True)
        # 快照分
        rally_snapshot_score = (price_up_score * turnover_down_score * concentration_up_score)**(1/3)
        # 新增行: 为快照分计算其在(1, 5)周期上的结构性背离
        holographic_divergence = self._calculate_holographic_divergence(rally_snapshot_score, 1, 5, norm_window)
        # 动态分 (元分析)
        rally_accumulation_score = self._perform_chip_relational_meta_analysis(
            df, rally_snapshot_score, 5, holographic_divergence
        ) # 修改行: 传入新增的背离分参数
        states['SCORE_CHIP_PB_RALLY_ACCUMULATION'] = rally_accumulation_score.astype(np.float32)
        # 更多剧本可以在此添加...
        return states
        # 修改结束

    def diagnose_capitulation_reversal_potential(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.2 · 冥王之眼同步版】诊断“恐慌投降反转”的潜力
        - 核心修复: 在调用元分析引擎时，补上缺失的“全息背离分”参数，与“冥王之眼”协议保持一致。
        """
        # 修改开始: 补上缺失的“全息背离分”参数
        states = {}
        p = get_params_block(self.strategy, 'capitulation_reversal_params', {})
        norm_window = get_param_value(p.get('norm_window'), 120)
        meta_window = 5 # 定义元分析窗口以保持一致性
        required_cols = ['total_loser_rate_D', 'close_D', 'turnover_from_losers_ratio_D']
        if any(col not in df.columns for col in required_cols):
            states['SCORE_CHIP_CONTEXT_CAPITULATION_POTENTIAL'] = pd.Series(0.0, index=df.index)
            return states
        # 步骤一：构建“恐慌投降关系”的瞬时快照分
        deep_capitulation_score = normalize_score(df['total_loser_rate_D'], df.index, norm_window, ascending=True)
        price_at_lows_score = 1.0 - normalize_score(df['close_D'], df.index, window=250, ascending=True)
        loser_turnover_score = normalize_score(df['turnover_from_losers_ratio_D'], df.index, norm_window, ascending=True)
        bearish_ma_context = 1 - self._calculate_ma_trend_context(df, [5, 13, 21, 55])
        snapshot_score = (deep_capitulation_score * price_at_lows_score * loser_turnover_score * bearish_ma_context).astype(np.float32)
        # 步骤二 (新增): 为快照分计算其在(1, 5)周期上的结构性背离
        holographic_divergence = self._calculate_holographic_divergence(snapshot_score, 1, meta_window, norm_window)
        # 步骤三 (升级): 对“恐慌投降关系”进行元分析，传入背离分
        final_score = self._perform_chip_relational_meta_analysis(
            df, snapshot_score, meta_window, holographic_divergence
        )
        states['SCORE_CHIP_CONTEXT_CAPITULATION_POTENTIAL'] = final_score
        return states
        # 修改结束

