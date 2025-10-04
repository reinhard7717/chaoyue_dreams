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
        【V502.0 · 权责净化版】筹码情报最高司令部
        - 核心升级: 剥离所有剧本(Playbook)的定义职责，回归情报生产者的本源。
        """
        all_chip_states = {}
        
        # 步骤 1: 执行统一的终极信号引擎 (不变)
        unified_states = self.diagnose_unified_chip_signals(df)
        all_chip_states.update(unified_states)

        # “吸筹剧本”和“投降反转剧本”的诊断逻辑被重构为潜力诊断，不再输出PLAYBOOK信号
        accumulation_potential_states = self.diagnose_accumulation_playbooks(df)
        all_chip_states.update(accumulation_potential_states)

        capitulation_potential_states = self.diagnose_capitulation_reversal_potential(df)
        all_chip_states.update(capitulation_potential_states)

        return all_chip_states

    def diagnose_unified_chip_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V22.1 · 赫尔墨斯之翼优化版】筹码终极信号诊断引擎
        - 性能优化: 1. 将普通几何平均的计算统一为更数值稳定的log-exp形式。
                      2. 确保所有Numpy和Pandas操作都输出为内存效率更高的float32类型。
        - 核心逻辑: 保持动态权重融合的核心逻辑不变，确保在部分数据支柱缺失时系统依然稳健。
        """
        states = {}
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True): return states
        
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        pillar_weights = get_param_value(p_conf.get('pillar_weights'), {})
        periods = get_param_value(p_synthesis.get('periods'), [1, 5, 13, 21, 55])
        norm_window = get_param_value(p_synthesis.get('norm_window'), 55)
        
        # 步骤一：计算所有子维度的健康度
        health_data = { 's_bull': [], 's_bear': [], 'd_intensity': [] } 
        calculators = {
            'quantitative': self._calculate_quantitative_health,
            'advanced': self._calculate_advanced_dynamics_health,
            'internal': self._calculate_internal_structure_health,
            'holder': self._calculate_holder_behavior_health,
            'fault': self._calculate_fault_health,
        }
        for name, calculator in calculators.items():
            # 注意：此处的dynamic_weights参数在优化后已不再需要，为保持签名兼容性暂时保留
            s_bull, s_bear, d_intensity = calculator(df, norm_window, {}, periods)
            health_data['s_bull'].append((name, s_bull)) 
            health_data['s_bear'].append((name, s_bear)) 
            health_data['d_intensity'].append((name, d_intensity))
            
        # 步骤二：对每个健康度类型进行多维度融合
        overall_health = {}
        use_equal_weights = not pillar_weights or sum(pillar_weights.values()) == 0
        
        for health_type, health_sources_with_names in health_data.items():
            overall_health[health_type] = {}
            for p in periods:
                # 动态构建当前周期有效的支柱列表和权重列表
                valid_pillars = []
                valid_weights = []
                for name, pillar_dict in health_sources_with_names:
                    if p in pillar_dict:
                        valid_pillars.append(pillar_dict[p].values)
                        valid_weights.append(pillar_weights.get(name, 0))

                if not valid_pillars:
                    overall_health[health_type][p] = pd.Series(0.5, index=df.index, dtype=np.float32)
                    continue

                stacked_values = np.stack(valid_pillars, axis=0)
                # 为避免log(0)错误，给所有值增加一个极小量
                safe_stacked_values = np.maximum(stacked_values, 1e-9)
                
                if use_equal_weights:
                    # 使用log-exp方式计算几何平均，以提高数值稳定性并统一代码路径
                    fused_values = np.exp(np.mean(np.log(safe_stacked_values), axis=0))
                else:
                    weights_array = np.array(valid_weights)
                    total_weight = weights_array.sum()
                    if total_weight > 0:
                        normalized_weights = weights_array / total_weight
                        # 计算加权几何平均
                        fused_values = np.exp(np.sum(np.log(safe_stacked_values) * normalized_weights[:, np.newaxis], axis=0))
                    else:
                        # 权重和为0时，退化为数值稳定的标准几何平均
                        fused_values = np.exp(np.mean(np.log(safe_stacked_values), axis=0))

                overall_health[health_type][p] = pd.Series(fused_values, index=df.index, dtype=np.float32)

        self.strategy.atomic_states['__CHIP_overall_health'] = overall_health
        
        # 步骤三：调用终极信号合成引擎
        ultimate_signals = transmute_health_to_ultimate_signals(
            df=df,
            atomic_states=self.strategy.atomic_states,
            overall_health=overall_health,
            params=p_synthesis,
            domain_prefix="CHIP"
        )
        states.update(ultimate_signals)
        return states

    # ==============================================================================
    # 以下为重构后的健康度组件计算器，现在返回四维健康度
    # ==============================================================================

    def _calculate_quantitative_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict[int, pd.Series], Dict[int, pd.Series], Dict[int, pd.Series]]:
        """
        【V6.0 · 德尔斐神谕协议版】计算基础量化维度的三维健康度
        - 核心修正: 签署“德尔斐神谕协议”，剥离 ma_context_score 对 s_bull/s_bear 的污染。
        """
        s_bull, s_bear, d_intensity = {}, {}, {}
        required_cols = ['concentration_90pct_D', 'chip_health_score_D']
        if any(col not in df.columns for col in required_cols):
            default_series = pd.Series(0.5, index=df.index, dtype=np.float32)
            for p in periods:
                s_bull[p] = s_bear[p] = d_intensity[p] = default_series
            return s_bull, s_bear, d_intensity
        static_bull_conc = normalize_score(df['concentration_90pct_D'], df.index, norm_window, ascending=False)
        static_bull_health = normalize_score(df['chip_health_score_D'], df.index, norm_window, ascending=True)
        chip_static_bull = np.sqrt(static_bull_conc * static_bull_health)
        static_bear_conc = 1.0 - static_bull_conc
        static_bear_health = 1.0 - static_bull_health
        chip_static_bear = np.sqrt(static_bear_conc * static_bear_health)
        # bullish_snapshot_score 和 bearish_snapshot_score 现在是纯粹的静态分，不再与 ma_context_score 相乘
        bullish_snapshot_score = chip_static_bull.astype(np.float32)
        bearish_snapshot_score = chip_static_bear.astype(np.float32)
        # 对纯粹的静态分进行元分析，得到动态强度
        unified_d_intensity = self._perform_chip_relational_meta_analysis(df, bullish_snapshot_score)
        for p in periods:
            s_bull[p] = bullish_snapshot_score
            s_bear[p] = bearish_snapshot_score
            d_intensity[p] = unified_d_intensity
        return s_bull, s_bear, d_intensity

    def _calculate_advanced_dynamics_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict[int, pd.Series], Dict[int, pd.Series], Dict[int, pd.Series]]:
        """
        【V6.0 · 德尔斐神谕协议版】计算高级动态维度的三维健康度
        - 核心修正: 签署“德尔斐神谕协议”，剥离 ma_context_score 对 s_bull/s_bear 的污染。
        """
        s_bull, s_bear, d_intensity = {}, {}, {}
        required_cols = ['peak_control_ratio_D', 'peak_strength_ratio_D', 'peak_stability_D', 'is_multi_peak_D']
        if any(col not in df.columns for col in required_cols):
            default_series = pd.Series(0.5, index=df.index, dtype=np.float32)
            for p in periods:
                s_bull[p] = default_series
                s_bear[p] = default_series
                d_intensity[p] = default_series
            return s_bull, s_bear, d_intensity
        is_multi_peak_series = df.get('is_multi_peak_D', pd.Series(0.0, index=df.index)).astype(float)
        chip_static_bull = (normalize_score(df.get('peak_control_ratio_D'), df.index, norm_window) * normalize_score(df.get('peak_strength_ratio_D'), df.index, norm_window) * normalize_score(df.get('peak_stability_D'), df.index, norm_window))**(1/3)
        chip_static_bear = (normalize_score(df.get('peak_control_ratio_D'), df.index, norm_window, ascending=False) * normalize_score(df.get('peak_strength_ratio_D'), df.index, norm_window, ascending=False) * normalize_score(df.get('peak_stability_D'), df.index, norm_window, ascending=False) * is_multi_peak_series)**(1/4)
        # bullish_snapshot_score 和 bearish_snapshot_score 现在是纯粹的静态分
        bullish_snapshot_score = chip_static_bull.astype(np.float32)
        bearish_snapshot_score = chip_static_bear.astype(np.float32)
        unified_d_intensity = self._perform_chip_relational_meta_analysis(df, bullish_snapshot_score)
        for p in periods:
            s_bull[p] = bullish_snapshot_score
            s_bear[p] = bearish_snapshot_score
            d_intensity[p] = unified_d_intensity
        return s_bull, s_bear, d_intensity

    def _calculate_internal_structure_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict[int, pd.Series], Dict[int, pd.Series], Dict[int, pd.Series]]:
        """
        【V6.0 · 德尔斐神谕协议版】计算内部结构维度的三维健康度
        - 核心修正: 签署“德尔斐神谕协议”，剥离 ma_context_score 对 s_bull/s_bear 的污染。
        """
        s_bull, s_bear, d_intensity = {}, {}, {}
        required_cols = ['concentration_70pct_D', 'support_below_D', 'pressure_above_D']
        if any(col not in df.columns for col in required_cols):
            default_series = pd.Series(0.5, index=df.index, dtype=np.float32)
            for p in periods:
                s_bull[p] = default_series
                s_bear[p] = default_series
                d_intensity[p] = default_series
            return s_bull, s_bear, d_intensity
        chip_static_bull = (normalize_score(df.get('concentration_70pct_D'), df.index, norm_window, ascending=False) * (normalize_score(df.get('support_below_D'), df.index, norm_window) * normalize_score(df.get('pressure_above_D'), df.index, norm_window, ascending=False))**0.5)**0.5
        chip_static_bear = (normalize_score(df.get('concentration_70pct_D'), df.index, norm_window, ascending=True) * (normalize_score(df.get('support_below_D'), df.index, norm_window, ascending=False) * normalize_score(df.get('pressure_above_D'), df.index, norm_window, ascending=True))**0.5)**0.5
        # bullish_snapshot_score 和 bearish_snapshot_score 现在是纯粹的静态分
        bullish_snapshot_score = chip_static_bull.astype(np.float32)
        bearish_snapshot_score = chip_static_bear.astype(np.float32)
        unified_d_intensity = self._perform_chip_relational_meta_analysis(df, bullish_snapshot_score)
        for p in periods:
            s_bull[p] = bullish_snapshot_score
            s_bear[p] = bearish_snapshot_score
            d_intensity[p] = unified_d_intensity
        return s_bull, s_bear, d_intensity

    def _calculate_holder_behavior_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict[int, pd.Series], Dict[int, pd.Series], Dict[int, pd.Series]]:
        """
        【V6.0 · 德尔斐神谕协议版】计算持仓者行为维度的三维健康度
        - 核心修正: 签署“德尔斐神谕协议”，剥离 ma_context_score 对 s_bull/s_bear 的污染。
        """
        s_bull, s_bear, d_intensity = {}, {}, {}
        required_cols = ['cost_divergence_D', 'winner_profit_margin_D', 'total_winner_rate_D', 'turnover_from_winners_ratio_D']
        if any(col not in df.columns for col in required_cols):
            default_series = pd.Series(0.5, index=df.index, dtype=np.float32)
            for p in periods:
                s_bull[p] = default_series
                s_bear[p] = default_series
                d_intensity[p] = default_series
            return s_bull, s_bear, d_intensity
        chip_static_bull = (normalize_score(df.get('cost_divergence_D'), df.index, norm_window) * normalize_score(df.get('winner_profit_margin_D'), df.index, norm_window) * normalize_score(df.get('total_winner_rate_D'), df.index, norm_window) * normalize_score(df.get('turnover_from_winners_ratio_D'), df.index, norm_window, ascending=False))**(1/4)
        chip_static_bear = (normalize_score(df.get('cost_divergence_D'), df.index, norm_window, ascending=False) * normalize_score(df.get('winner_profit_margin_D'), df.index, norm_window, ascending=False) * normalize_score(df.get('total_winner_rate_D'), df.index, norm_window, ascending=False) * normalize_score(df.get('turnover_from_winners_ratio_D'), df.index, norm_window, ascending=True))**(1/4)
        # bullish_snapshot_score 和 bearish_snapshot_score 现在是纯粹的静态分
        bullish_snapshot_score = chip_static_bull.astype(np.float32)
        bearish_snapshot_score = chip_static_bear.astype(np.float32)
        unified_d_intensity = self._perform_chip_relational_meta_analysis(df, bullish_snapshot_score)
        for p in periods:
            s_bull[p] = bullish_snapshot_score
            s_bear[p] = bearish_snapshot_score
            d_intensity[p] = unified_d_intensity
        return s_bull, s_bear, d_intensity

    def _calculate_fault_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict[int, pd.Series], Dict[int, pd.Series], Dict[int, pd.Series]]:
        """
        【V5.0 · 德尔斐神谕协议版】计算筹码断层维度的三维健康度
        - 核心修正: 签署“德尔斐神谕协议”，剥离 ma_context_score 对 s_bull/s_bear 的污染。
        """
        s_bull, s_bear, d_intensity = {}, {}, {}
        required_cols = ['chip_fault_strength_D', 'chip_fault_vacuum_percent_D']
        if any(col not in df.columns for col in required_cols):
            default_series = pd.Series(0.5, index=df.index, dtype=np.float32)
            for p in periods:
                s_bull[p] = default_series
                s_bear[p] = default_series
                d_intensity[p] = default_series
            return s_bull, s_bear, d_intensity
        chip_static_bull = (normalize_score(df.get('chip_fault_strength_D'), df.index, norm_window, ascending=False) * normalize_score(df.get('chip_fault_vacuum_percent_D'), df.index, norm_window, ascending=False))**0.5
        chip_static_bear = (normalize_score(df.get('chip_fault_strength_D'), df.index, norm_window, ascending=True) * normalize_score(df.get('chip_fault_vacuum_percent_D'), df.index, norm_window, ascending=True))**0.5
        # bullish_snapshot_score 和 bearish_snapshot_score 现在是纯粹的静态分
        bullish_snapshot_score = chip_static_bull.astype(np.float32)
        bearish_snapshot_score = chip_static_bear.astype(np.float32)
        unified_d_intensity = self._perform_chip_relational_meta_analysis(df, bullish_snapshot_score)
        for p in periods:
            s_bull[p] = bullish_snapshot_score
            s_bear[p] = bearish_snapshot_score
            d_intensity[p] = unified_d_intensity
        return s_bull, s_bear, d_intensity

    def _perform_chip_relational_meta_analysis(self, df: pd.DataFrame, snapshot_score: pd.Series) -> pd.Series:
        """
        【V1.0 · 新增】筹码专用的关系元分析核心引擎 (赫拉织布机V2)
        - 核心逻辑: 实现“状态 * (1 + 动态杠杆)”的动态价值调制范式。
        """
        # 从配置中获取动态杠杆权重
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        p_meta = get_param_value(p_conf.get('relational_meta_analysis_params'), {})
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.6)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.4)

        # 核心参数
        norm_window = 55
        meta_window = 5
        bipolar_sensitivity = 1.0

        # 第一维度：状态分 (State Score)
        state_score = snapshot_score.clip(0, 1)

        # 第二维度：速度分 (Velocity Score)
        relationship_trend = snapshot_score.diff(meta_window).fillna(0)
        velocity_score = normalize_to_bipolar(
            series=relationship_trend, target_index=df.index,
            window=norm_window, sensitivity=bipolar_sensitivity
        )

        # 第三维度：加速度分 (Acceleration Score)
        relationship_accel = relationship_trend.diff(meta_window).fillna(0)
        acceleration_score = normalize_to_bipolar(
            series=relationship_accel, target_index=df.index,
            window=norm_window, sensitivity=bipolar_sensitivity
        )

        # 终极融合：动态价值调制
        dynamic_leverage = 1 + (velocity_score * w_velocity) + (acceleration_score * w_acceleration)
        final_score = (state_score * dynamic_leverage).clip(0, 1)
        
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

    def _diagnose_setup_capitulation_ready(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """【V1.2 · 重构修复版】诊断“恐慌已弥漫”的战备(Setup)状态 (战术模块，予以保留)"""
        states = {}
        required_col = 'total_loser_rate_D'
        if required_col not in df.columns:
            print(f"        -> [筹码情报-恐慌战备诊断] 警告: 缺少关键数据列 '{required_col}'，模块已跳过！")
            return states
        p = get_params_block(self.strategy, 'capitulation_reversal_params', {})
        norm_window = get_param_value(p.get('norm_window'), 120)
        
        # 修正对 normalize_score 的调用，这是导致错误的根源
        deep_capitulation_score = normalize_score(df['total_loser_rate_D'], df.index, norm_window, ascending=True)
        
        long_term_window = 250
        min_periods_long = long_term_window // 4
        
        # 修正对 normalize_score 的调用
        rank_score = normalize_score(df['close_D'], df.index, window=long_term_window, ascending=False)
        
        rolling_low = df['low_D'].rolling(window=long_term_window, min_periods=min_periods_long).min()
        rolling_high = df['high_D'].rolling(window=long_term_window, min_periods=min_periods_long).max()
        price_range = (rolling_high - rolling_low).replace(0, 1e-9)
        position_in_range = (df['close_D'] - rolling_low) / price_range
        range_score = 1.0 - position_in_range.clip(0, 1)
        price_pos_score = np.maximum(rank_score, range_score.fillna(0.5))
        setup_score = (deep_capitulation_score * price_pos_score).astype(np.float32)
        states['SCORE_SETUP_CAPITULATION_READY'] = setup_score
        return states

    def _diagnose_trigger_capitulation_fire(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """【V1.1 · 重构修复版】诊断“卖压出清”的点火(Trigger)行为 (战术模块，予以保留)"""
        states = {}
        required_cols = ['turnover_from_losers_ratio_D', 'ACCEL_5_turnover_from_losers_ratio_D']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"        -> [筹码情报-卖压出清诊断] 警告: 缺少关键数据列 {missing_cols}，模块已跳过！")
            return states
        p = get_params_block(self.strategy, 'capitulation_reversal_params', {})
        norm_window = get_param_value(p.get('norm_window'), 120)
        
        # 修正对 normalize_score 的调用
        relative_turnover_score = normalize_score(df['turnover_from_losers_ratio_D'], df.index, norm_window, ascending=True)
        
        k = get_param_value(p.get('logistic_k', 0.1))
        x0 = get_param_value(p.get('logistic_x0', 50.0))
        absolute_turnover_score = 1 / (1 + np.exp(-k * (df['turnover_from_losers_ratio_D'] - x0)))
        loser_turnover_score = np.maximum(relative_turnover_score, absolute_turnover_score)
        
        # 修正对 normalize_score 的调用
        loser_turnover_accel_score = normalize_score(df['ACCEL_5_turnover_from_losers_ratio_D'], df.index, norm_window, ascending=True)
        
        trigger_score = (loser_turnover_score * loser_turnover_accel_score).astype(np.float32)
        states['SCORE_TRIGGER_CAPITULATION_FIRE'] = trigger_score
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

