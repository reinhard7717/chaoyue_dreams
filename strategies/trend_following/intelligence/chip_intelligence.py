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

    def _synthesize_ultimate_signals(self, concentration: Dict[int, pd.Series], accumulation: Dict[int, pd.Series], power_transfer: Dict[int, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.6 · 哈迪斯陷阱版】终极信号合成器
        - 核心升级: 新增“哈迪斯陷阱” (Hades' Trap) 诊断模块。
                      专门用于识别“技术性底部反转”与“行为性主力派发”同时发生的致命陷阱。
                      陷阱分 = 底部反转强度 * 短期派发强度
        """
        # 使用全息背离引擎重新定义反转
        states = {}
        periods = sorted(concentration.keys())
        tf_weights = {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}
        norm_window = 55
        # --- 看涨/看跌共振 ---
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
        # --- 底部/顶部反转信号 ---
        bottom_reversal_divergence = self._calculate_holographic_divergence(bullish_resonance, 5, 21, norm_window)
        states['SCORE_CHIP_BOTTOM_REVERSAL'] = bottom_reversal_divergence.clip(0, 1).astype(np.float32)
        top_reversal_divergence = self._calculate_holographic_divergence(bearish_resonance, 5, 21, norm_window)
        states['SCORE_CHIP_TOP_REversal'] = top_reversal_divergence.clip(0, 1).astype(np.float32)
        # --- 战术反转 ---
        tactical_reversal = (bullish_resonance * 0.5).astype(np.float32)
        states['SCORE_CHIP_TACTICAL_REVERSAL'] = tactical_reversal
        # 部署“哈迪斯陷阱”诊断模块
        # 步骤1: 即时计算5日周期的“权力转移”快照分，作为短期派发的直接证据
        df = self.strategy.df_indicators
        p = 5 # 使用最短周期5日来捕捉当日行为
        cost_divergence_score = normalize_score(df.get(f'SLOPE_{p}_cost_divergence_D'), df.index, norm_window, ascending=True)
        loser_turnover_up = normalize_score(df.get(f'SLOPE_{p}_turnover_from_losers_ratio_D'), df.index, norm_window, ascending=True)
        transfer_to_main_force_evidence = (cost_divergence_score * loser_turnover_up)**0.5
        cost_convergence_score = normalize_score(df.get(f'SLOPE_{p}_cost_divergence_D'), df.index, norm_window, ascending=False)
        loser_turnover_down = normalize_score(df.get(f'SLOPE_{p}_turnover_from_losers_ratio_D'), df.index, norm_window, ascending=False)
        transfer_to_retail_evidence = (cost_convergence_score * loser_turnover_down)**0.5
        transfer_snapshot = (transfer_to_main_force_evidence - transfer_to_retail_evidence).astype(np.float32)
        # 步骤2: 将派发行为(-1到0)映射为派发强度(0到1)
        distribution_strength = (transfer_snapshot.clip(-1, 0) * -1).astype(np.float32)
        # 步骤3: 融合“反转幻象”与“派发事实”，铸造“哈迪斯陷阱分”
        # 陷阱分 = 底部反转信号强度 * 当日派发强度
        hades_trap_score = (states['SCORE_CHIP_BOTTOM_REVERSAL'] * distribution_strength).clip(0, 1)
        states['SCORE_CHIP_HADES_TRAP'] = hades_trap_score.astype(np.float32)

        return states

    def _diagnose_concentration_dynamics(self, df: pd.DataFrame, periods: list) -> Dict[int, pd.Series]:
        """
        【V2.1 · 分层印证版】核心公理一：诊断筹码“聚散”的动态
        - 核心升级: 引入“分层动态印证框架”。每个战术周期的集中度变化，都由其紧邻的更长一级周期趋势进行印证。
        - 印证链: 1日由5日印证，5日由13日印证，以此类推，形成动态共振。
        """
        scores = {}
        # 引入动态的、分层的上下文印证框架
        sorted_periods = sorted(periods)
        for i, p in enumerate(sorted_periods):
            # 确定当前战术周期p的上下文周期context_p，形成 1->5, 5->13, ... 的分层对比关系
            context_p = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p
            # 步骤1: 计算上下文层(context_p)的“集中度质量”基准分
            context_conc_90 = normalize_score(df.get('concentration_90pct_D'), df.index, window=context_p, ascending=False)
            context_conc_70 = normalize_score(df.get('concentration_70pct_D'), df.index, window=context_p, ascending=False)
            context_stability = normalize_score(df.get('peak_stability_D'), df.index, window=context_p, ascending=True)
            context_control = normalize_score(df.get('peak_control_ratio_D'), df.index, window=context_p, ascending=True)
            context_turnover_risk = normalize_score(df.get('turnover_at_peak_ratio_D'), df.index, window=context_p, ascending=True)
            context_non_risk = 1.0 - context_turnover_risk
            context_concentration_quality = (context_conc_90 * context_conc_70 * context_stability * context_control * context_non_risk)**(1/5)
            # 步骤2: 计算战术层(p)的“集中度质量”分
            tactical_conc_90 = normalize_score(df.get('concentration_90pct_D'), df.index, window=p, ascending=False)
            tactical_conc_70 = normalize_score(df.get('concentration_70pct_D'), df.index, window=p, ascending=False)
            tactical_stability = normalize_score(df.get('peak_stability_D'), df.index, window=p, ascending=True)
            tactical_control = normalize_score(df.get('peak_control_ratio_D'), df.index, window=p, ascending=True)
            tactical_turnover_risk = normalize_score(df.get('turnover_at_peak_ratio_D'), df.index, window=p, ascending=True)
            tactical_non_risk = 1.0 - tactical_turnover_risk
            tactical_concentration_quality = (tactical_conc_90 * tactical_conc_70 * tactical_stability * tactical_control * tactical_non_risk)**(1/5)
            # 步骤3: 融合上下文与战术层，形成共振快照分
            concentration_quality_snapshot = (tactical_concentration_quality * context_concentration_quality)**0.5
            # 步骤4: 对共振快照分进行关系元分析
            holographic_divergence = self._calculate_holographic_divergence(concentration_quality_snapshot, 1, p, p * 2)
            dynamic_concentration_score = self._perform_chip_relational_meta_analysis(
                df, concentration_quality_snapshot, p, holographic_divergence
            )
            scores[p] = dynamic_concentration_score
        
        return scores

    def _diagnose_main_force_action(self, df: pd.DataFrame, periods: list) -> Dict[int, pd.Series]:
        """
        【V2.1 · 分层印证版】核心公理二：诊断主力“吸筹与派发”
        - 核心升级: 引入“分层动态印证框架”。短期的吸筹/派发行为，由其紧邻的更长一级趋势进行印证。
        - 印证链: 1日行为由5日趋势印证，5日行为由13日趋势印证，以此类推。
        """
        scores = {}
        main_force_urgency_score = self.strategy.atomic_states.get('PROCESS_META_MAIN_FORCE_URGENCY', pd.Series(0.5, index=df.index))
        # 引入动态的、分层的上下文印证框架
        sorted_periods = sorted(periods)
        for i, p in enumerate(sorted_periods):
            # 确定当前战术周期p的上下文周期context_p
            context_p = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p
            # 步骤1: 计算上下文层(context_p)的“主力行动”基准分
            context_conc_slope_up = normalize_score(df.get(f'SLOPE_{context_p}_concentration_90pct_D'), df.index, window=context_p, ascending=True)
            context_winner_turnover_down = normalize_score(df.get(f'SLOPE_{context_p}_turnover_from_winners_ratio_D'), df.index, window=context_p, ascending=False)
            context_trade_conc_up = normalize_score(df.get(f'SLOPE_{context_p}_trade_concentration_index_D'), df.index, window=context_p, ascending=True)
            context_accumulation_evidence = (context_conc_slope_up * context_winner_turnover_down * context_trade_conc_up * main_force_urgency_score)**(1/4)
            context_conc_slope_down = normalize_score(df.get(f'SLOPE_{context_p}_concentration_90pct_D'), df.index, window=context_p, ascending=False)
            context_winner_turnover_up = normalize_score(df.get(f'SLOPE_{context_p}_turnover_from_winners_ratio_D'), df.index, window=context_p, ascending=True)
            context_trade_conc_down = normalize_score(df.get(f'SLOPE_{context_p}_trade_concentration_index_D'), df.index, window=context_p, ascending=False)
            context_losing_money = normalize_score(df.get(f'SLOPE_{context_p}_main_force_intraday_profit_D'), df.index, window=context_p, ascending=False)
            context_distribution_evidence = (context_conc_slope_down * context_winner_turnover_up * context_trade_conc_down * context_losing_money)**(1/4)
            # 步骤2: 计算战术层(p)的“吸筹/派发”证据分
            tactical_conc_slope_up = normalize_score(df.get(f'SLOPE_{p}_concentration_90pct_D'), df.index, window=p, ascending=True)
            tactical_winner_turnover_down = normalize_score(df.get(f'SLOPE_{p}_turnover_from_winners_ratio_D'), df.index, window=p, ascending=False)
            tactical_trade_conc_up = normalize_score(df.get(f'SLOPE_{p}_trade_concentration_index_D'), df.index, window=p, ascending=True)
            tactical_accumulation_evidence = (tactical_conc_slope_up * tactical_winner_turnover_down * tactical_trade_conc_up * main_force_urgency_score)**(1/4)
            tactical_conc_slope_down = normalize_score(df.get(f'SLOPE_{p}_concentration_90pct_D'), df.index, window=p, ascending=False)
            tactical_winner_turnover_up = normalize_score(df.get(f'SLOPE_{p}_turnover_from_winners_ratio_D'), df.index, window=p, ascending=True)
            tactical_trade_conc_down = normalize_score(df.get(f'SLOPE_{p}_trade_concentration_index_D'), df.index, window=p, ascending=False)
            tactical_losing_money = normalize_score(df.get(f'SLOPE_{p}_main_force_intraday_profit_D'), df.index, window=p, ascending=False)
            tactical_distribution_evidence = (tactical_conc_slope_down * tactical_winner_turnover_up * tactical_trade_conc_down * tactical_losing_money)**(1/4)
            # 步骤3: 融合上下文与战术层，形成共振
            accumulation_evidence = (tactical_accumulation_evidence * context_accumulation_evidence)**0.5
            distribution_evidence = (tactical_distribution_evidence * context_distribution_evidence)**0.5
            # 步骤4: 生成双极性的“行动快照分”
            action_snapshot = (accumulation_evidence - distribution_evidence).astype(np.float32)
            # 步骤5: 对共振快照分进行关系元分析
            holographic_divergence = self._calculate_holographic_divergence(action_snapshot, 1, p, p * 2)
            dynamic_action_score = self._perform_chip_relational_meta_analysis(
                df, action_snapshot, p, holographic_divergence
            )
            scores[p] = dynamic_action_score
        
        return scores

    def _diagnose_power_transfer(self, df: pd.DataFrame, periods: list) -> Dict[int, pd.Series]:
        """
        【V1.9 · 分层印证版】核心公理三：诊断筹码“转移方向”
        - 核心升级: 引入“分层动态印证框架”。短期的筹码转移方向，由其紧邻的更长一级趋势进行印证，形成共振。
        - 印证链: 1日转移由5日趋势印证，5日转移由13日趋势印证，以此类推。
        """
        scores = {}
        # 引入动态的、分层的上下文印证框架
        sorted_periods = sorted(periods)
        for i, p in enumerate(sorted_periods):
            # 确定当前战术周期p的上下文周期context_p
            context_p = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p
            # 步骤1: 计算上下文层(context_p)的“权力转移”基准分
            context_cost_divergence = normalize_score(df.get(f'SLOPE_{context_p}_cost_divergence_D'), df.index, window=context_p, ascending=True)
            context_loser_turnover_up = normalize_score(df.get(f'SLOPE_{context_p}_turnover_from_losers_ratio_D'), df.index, window=context_p, ascending=True)
            context_transfer_to_main_force = (context_cost_divergence * context_loser_turnover_up)**0.5
            context_cost_convergence = normalize_score(df.get(f'SLOPE_{context_p}_cost_divergence_D'), df.index, window=context_p, ascending=False)
            context_loser_turnover_down = normalize_score(df.get(f'SLOPE_{context_p}_turnover_from_losers_ratio_D'), df.index, window=context_p, ascending=False)
            context_transfer_to_retail = (context_cost_convergence * context_loser_turnover_down)**0.5
            # 步骤2: 计算战术层(p)的“权力转移”证据分
            tactical_cost_divergence = normalize_score(df.get(f'SLOPE_{p}_cost_divergence_D'), df.index, window=p, ascending=True)
            tactical_loser_turnover_up = normalize_score(df.get(f'SLOPE_{p}_turnover_from_losers_ratio_D'), df.index, window=p, ascending=True)
            tactical_transfer_to_main_force = (tactical_cost_divergence * tactical_loser_turnover_up)**0.5
            tactical_cost_convergence = normalize_score(df.get(f'SLOPE_{p}_cost_divergence_D'), df.index, window=p, ascending=False)
            tactical_loser_turnover_down = normalize_score(df.get(f'SLOPE_{p}_turnover_from_losers_ratio_D'), df.index, window=p, ascending=False)
            tactical_transfer_to_retail = (tactical_cost_convergence * tactical_loser_turnover_down)**0.5
            # 步骤3: 融合上下文与战术层，形成共振
            transfer_to_main_force_evidence = (tactical_transfer_to_main_force * context_transfer_to_main_force)**0.5
            transfer_to_retail_evidence = (tactical_transfer_to_retail * context_transfer_to_retail)**0.5
            # 步骤4: 生成双极性的“转移快照分”
            transfer_snapshot = (transfer_to_main_force_evidence - transfer_to_retail_evidence).astype(np.float32)
            # 步骤5: 对共振快照分进行关系元分析
            holographic_divergence = self._calculate_holographic_divergence(transfer_snapshot, 1, p, p * 2)
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
        # 补上缺失的“全息背离分”参数
        states = {}
        norm_window = 120
        # 剧本一：“拉升吸筹” (Rally Accumulation)
        # 证据链：价格上涨，但换手率下降，且筹码集中度在提升
        price_up_score = normalize_score(df.get('SLOPE_5_close'), df.index, norm_window, ascending=True)
        turnover_down_score = normalize_score(df.get('SLOPE_5_turnover_rate_f'), df.index, norm_window, ascending=False)
        concentration_up_score = normalize_score(df.get('SLOPE_5_concentration_90pct_D'), df.index, norm_window, ascending=True)
        # 快照分
        rally_snapshot_score = (price_up_score * turnover_down_score * concentration_up_score)**(1/3)
        # 为快照分计算其在(1, 5)周期上的结构性背离
        holographic_divergence = self._calculate_holographic_divergence(rally_snapshot_score, 1, 5, norm_window)
        # 动态分 (元分析)
        rally_accumulation_score = self._perform_chip_relational_meta_analysis(
            df, rally_snapshot_score, 5, holographic_divergence
        ) # 传入新增的背离分参数
        states['SCORE_CHIP_PB_RALLY_ACCUMULATION'] = rally_accumulation_score.astype(np.float32)
        # 更多剧本可以在此添加...
        return states

    def diagnose_capitulation_reversal_potential(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.2 · 冥王之眼同步版】诊断“恐慌投降反转”的潜力
        - 核心修复: 在调用元分析引擎时，补上缺失的“全息背离分”参数，与“冥王之眼”协议保持一致。
        """
        # 补上缺失的“全息背离分”参数
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


