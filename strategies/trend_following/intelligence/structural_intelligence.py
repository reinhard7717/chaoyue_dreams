# 文件: strategies/trend_following/intelligence/structural_intelligence.py
# 结构情报模块 (均线, 箱体, 平台, 趋势)
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following.utils import get_params_block, get_param_value, create_persistent_state, normalize_score, calculate_context_scores

class StructuralIntelligence:
    def __init__(self, strategy_instance, dynamic_thresholds: Dict):
        """
        初始化结构情报模块。
        :param strategy_instance: 策略主实例的引用。
        :param dynamic_thresholds: 动态阈值字典。
        """
        self.strategy = strategy_instance
        self.dynamic_thresholds = dynamic_thresholds

    def diagnose_structural_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 终极信号版】结构情报分析总指挥
        - 核心重构: 遵循终极信号范式，本模块不再返回一堆零散的原子信号。
                      现在只调用唯一的终极信号引擎 `diagnose_ultimate_structural_signals`，
                      并将其产出的16个S+/S/A/B级信号作为本模块的最终输出。
        - 收益: 架构与其他情报模块完全统一，极大提升了信号质量和架构清晰度。
        """
        # print("      -> [结构情报分析总指挥 V2.0 终极信号版] 启动...")
        # 直接调用终极信号引擎，并将其结果作为本模块的唯一输出
        ultimate_structural_states = self.diagnose_ultimate_structural_signals(df)
        # print(f"      -> [结构情报分析总指挥 V2.0] 分析完毕，共生成 {len(ultimate_structural_states)} 个终极结构信号。")
        return ultimate_structural_states

    def diagnose_ultimate_structural_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V8.0 · 信号净化版】终极结构信号诊断模块
        - 核心重构: 废除S/A/B分级，只输出唯一的、归一化的终极信号。
                      信号名不再包含 _S_PLUS 后缀，实现命名的终极简化。
        - 健壮性加固: 统一了多周期力计算中的默认值类型，确保在数据缺失时使用 Series 而非 float。
        """
        states = {}
        p_conf = get_params_block(self.strategy, 'structural_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True): return states
        exponent = get_param_value(p_conf.get('final_score_exponent'), 1.0)
        
        resonance_tf_weights = {'short': 0.2, 'medium': 0.5, 'long': 0.3}
        reversal_tf_weights = {'short': 0.6, 'medium': 0.3, 'long': 0.1}
        dynamic_weights = {'slope': 0.6, 'accel': 0.4}
        periods = get_param_value(p_conf.get('periods', [1, 5, 13, 21, 55]))
        norm_window = get_param_value(p_conf.get('norm_window'), 120)
        bottom_context_bonus_factor = get_param_value(p_conf.get('bottom_context_bonus_factor'), 0.5)
        top_context_bonus_factor = get_param_value(p_conf.get('top_context_bonus_factor'), 0.8)

        bottom_context_score, top_context_score = calculate_context_scores(df, self.strategy.atomic_states)
        
        health_data = { 's_bull': [], 'd_bull': [], 's_bear': [], 'd_bear': [] } 
        calculators = { 'ma': self._calculate_ma_health, 'mechanics': self._calculate_mechanics_health, 'mtf': self._calculate_mtf_health, 'pattern': self._calculate_pattern_health }
        for name, calculator in calculators.items():
            s_bull, d_bull, s_bear, d_bear = calculator(df, periods, norm_window, dynamic_weights)
            health_data['s_bull'].append(s_bull) 
            health_data['d_bull'].append(d_bull) 
            health_data['s_bear'].append(s_bear) 
            health_data['d_bear'].append(d_bear) 
        
        overall_health = {}
        
        for health_type, health_sources in [ ('s_bull', health_data['s_bull']), ('d_bull', health_data['d_bull']), ('s_bear', health_data['s_bear']), ('d_bear', health_data['d_bear']) ]:
        
            overall_health[health_type] = {}
            for p in periods:
                components_for_period = [pillar_dict[p].values for pillar_dict in health_sources if p in pillar_dict]
                if components_for_period:
                    stacked_values = np.stack(components_for_period, axis=0)
                    fused_values = np.prod(stacked_values, axis=0) ** (1.0 / stacked_values.shape[0])
                    overall_health[health_type][p] = pd.Series(fused_values, index=df.index, dtype=np.float32)
                else:
                    overall_health[health_type][p] = pd.Series(0.5, index=df.index, dtype=np.float32)

        self.strategy.atomic_states['__STRUCTURE_overall_health'] = overall_health

        # 创建一个标准的默认Series，用于健壮性处理
        default_series = pd.Series(0.5, index=df.index, dtype=np.float32)

        bullish_resonance_health = {p: overall_health['s_bull'][p] * overall_health['d_bull'][p] for p in periods}
        # 使用 default_series 替换浮点数 0.5
        bullish_short_force_res = (bullish_resonance_health.get(1, default_series) * bullish_resonance_health.get(5, default_series))**0.5
        bullish_medium_trend_res = (bullish_resonance_health.get(13, default_series) * bullish_resonance_health.get(21, default_series))**0.5
        bullish_long_inertia_res = bullish_resonance_health.get(55, default_series)
        overall_bullish_resonance = (
            (bullish_short_force_res ** resonance_tf_weights['short']) *
            (bullish_medium_trend_res ** resonance_tf_weights['medium']) *
            (bullish_long_inertia_res ** resonance_tf_weights['long'])
        )
        
        bullish_reversal_health = {p: overall_health['s_bear'][p] * overall_health['d_bull'][p] for p in periods}
        # 使用 default_series 替换浮点数 0.5
        bullish_short_force_rev = (bullish_reversal_health.get(1, default_series) * bullish_reversal_health.get(5, default_series))**0.5
        bullish_medium_trend_rev = (bullish_reversal_health.get(13, default_series) * bullish_reversal_health.get(21, default_series))**0.5
        bullish_long_inertia_rev = bullish_reversal_health.get(55, default_series)
        overall_bullish_reversal_trigger = (
            (bullish_short_force_rev ** reversal_tf_weights['short']) *
            (bullish_medium_trend_rev ** reversal_tf_weights['medium']) *
            (bullish_long_inertia_rev ** reversal_tf_weights['long'])
        )
        final_bottom_reversal_score = (overall_bullish_reversal_trigger * (1 + bottom_context_score * bottom_context_bonus_factor)).clip(0, 1)

        bearish_resonance_health = {p: overall_health['s_bear'][p] * overall_health['d_bear'][p] for p in periods}
        # 使用 default_series 替换浮点数 0.5
        bearish_short_force_res = (bearish_resonance_health.get(1, default_series) * bearish_resonance_health.get(5, default_series))**0.5
        bearish_medium_trend_res = (bearish_resonance_health.get(13, default_series) * bearish_resonance_health.get(21, default_series))**0.5
        bearish_long_inertia_res = bearish_resonance_health.get(55, default_series)
        overall_bearish_resonance = (
            (bearish_short_force_res ** resonance_tf_weights['short']) *
            (bearish_medium_trend_res ** resonance_tf_weights['medium']) *
            (bearish_long_inertia_res ** resonance_tf_weights['long'])
        )

        bearish_reversal_health = {p: overall_health['s_bull'][p] * overall_health['d_bear'][p] for p in periods}
        # 使用 default_series 替换浮点数 0.5
        bearish_short_force_rev = (bearish_reversal_health.get(1, default_series) * bearish_reversal_health.get(5, default_series))**0.5
        bearish_medium_trend_rev = (bearish_reversal_health.get(13, default_series) * bearish_reversal_health.get(21, default_series))**0.5
        bearish_long_inertia_rev = bearish_reversal_health.get(55, default_series)
        overall_bearish_reversal_trigger = (
            (bearish_short_force_rev ** reversal_tf_weights['short']) *
            (bearish_medium_trend_rev ** reversal_tf_weights['medium']) *
            (bearish_long_inertia_rev ** reversal_tf_weights['long'])
        )
        final_top_reversal_score = (overall_bearish_reversal_trigger * (1 + top_context_score * top_context_bonus_factor)).clip(0, 1)
        
        # 信号命名净化：废除S/A/B分级，只使用唯一的、归一化的终极信号名
        final_signal_map = {
            'SCORE_STRUCTURE_BULLISH_RESONANCE': (overall_bullish_resonance ** exponent),
            'SCORE_STRUCTURE_BOTTOM_REVERSAL': (final_bottom_reversal_score ** exponent),
            'SCORE_STRUCTURE_BEARISH_RESONANCE': (overall_bearish_resonance ** exponent),
            'SCORE_STRUCTURE_TOP_REVERSAL': (final_top_reversal_score ** exponent)
        }

        for signal_name, score in final_signal_map.items():
            # 只生成唯一的、归一化的信号，其名称不包含任何等级后缀
            states[signal_name] = score.astype(np.float32)
        
        return states

    # ==============================================================================
    # 以下为重构后的健康度组件计算器，现在返回四维健康度
    # ==============================================================================

    def _calculate_ma_health(self, df: pd.DataFrame, periods: list, norm_window: int, dynamic_weights: Dict) -> Tuple[Dict, Dict, Dict, Dict]:
        """【V2.1 · 静态逻辑重构版】计算MA支柱的四维健康度"""
        s_bull, d_bull, s_bear, d_bear = {}, {}, {}, {}

        # 将静态分计算移出循环，使用所有均线的平均对齐度作为静态分
        ma_periods = [5, 10, 20, 60, 120] # 使用一组固定的均线来评估整体结构
        bull_alignment_scores = []
        bear_alignment_scores = []
        for i in range(len(ma_periods) - 1):
            short_col = f'EMA_{ma_periods[i]}_D'
            long_col = f'EMA_{ma_periods[i+1]}_D'
            if short_col in df and long_col in df:
                bull_alignment_scores.append((df[short_col] > df[long_col]).astype(float))
                bear_alignment_scores.append((df[short_col] < df[long_col]).astype(float))
        
        if bull_alignment_scores:
            static_bull_score = pd.DataFrame(bull_alignment_scores).mean().fillna(0.5)
            static_bear_score = pd.DataFrame(bear_alignment_scores).mean().fillna(0.5)
        else:
            static_bull_score = pd.Series(0.5, index=df.index)
            static_bear_score = pd.Series(0.5, index=df.index)

        for p in periods:
            # 为所有周期分配同一个、真正的静态分
            s_bull[p] = static_bull_score
            s_bear[p] = static_bear_score
            
            static_col = f'EMA_{p}_D' if p > 1 else 'close_D'
            slope_col = f'SLOPE_{p}_{static_col}'
            accel_col = f'ACCEL_{p}_{static_col}'
            
            slope_score = normalize_score(df.get(slope_col), df.index, norm_window, ascending=True)
            accel_score = normalize_score(df.get(accel_col), df.index, norm_window, ascending=True)
            d_bull[p] = (slope_score * accel_score)**0.5
            
            slope_score_neg = normalize_score(df.get(slope_col), df.index, norm_window, ascending=False)
            accel_score_neg = normalize_score(df.get(accel_col), df.index, norm_window, ascending=False)
            d_bear[p] = (slope_score_neg * accel_score_neg)**0.5
        return s_bull, d_bull, s_bear, d_bear

    def _calculate_mechanics_health(self, df: pd.DataFrame, periods: list, norm_window: int, dynamic_weights: Dict) -> Tuple[Dict, Dict, Dict, Dict]:
        """【V2.1 · 终极哲学统一版】计算力学支柱的四维健康度"""
        s_bull, d_bull, s_bear, d_bear = {}, {}, {}, {}
        static_bull_energy = normalize_score(df.get('energy_ratio_D'), df.index, norm_window, ascending=True)
        static_bear_energy = normalize_score(df.get('energy_ratio_D'), df.index, norm_window, ascending=False)
        for p in periods:
            s_bull[p] = static_bull_energy
            s_bear[p] = static_bear_energy
            
            # 根除所有动态分计算中的加法
            cost_slope_bull = normalize_score(df.get(f'SLOPE_{p}_peak_cost_D'), df.index, norm_window, ascending=True)
            conc_lock_bull = normalize_score(df.get(f'SLOPE_{p}_concentration_90pct_D'), df.index, norm_window, ascending=False)
            d_bull[p] = (cost_slope_bull * conc_lock_bull)**0.5
            
            cost_slope_bear = normalize_score(df.get(f'SLOPE_{p}_peak_cost_D'), df.index, norm_window, ascending=False)
            conc_lock_bear = normalize_score(df.get(f'SLOPE_{p}_concentration_90pct_D'), df.index, norm_window, ascending=True)
            d_bear[p] = (cost_slope_bear * conc_lock_bear)**0.5
            
        return s_bull, d_bull, s_bear, d_bear

    def _calculate_mtf_health(self, df: pd.DataFrame, periods: list, norm_window: int, dynamic_weights: Dict) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        【V2.1 · 性能优化版】计算MTF(多时间框架)支柱的四维健康度
        - 本次优化:
          - [效率] 简化了 `get_avg_score` 辅助函数，使用列表推导式和 `np.mean` 提高效率。
          - [效率] 将静态分和动态分的计算移出循环，一次性生成后通过字典推导式赋值给所有周期，
                    避免了不必要的重复计算。
        """
        # 静态健康度：周线结构对齐度
        weekly_cols = [col for col in df.columns if 'EMA' in col and col.endswith('_W')]
        if len(weekly_cols) > 1:
            # 使用列表推导式和np.mean进行向量化计算
            bull_align_matrix = np.stack([(df[weekly_cols[i]] > df[weekly_cols[i+1]]).values for i in range(len(weekly_cols)-1)], axis=0)
            bear_align_matrix = np.stack([(df[weekly_cols[i]] < df[weekly_cols[i+1]]).values for i in range(len(weekly_cols)-1)], axis=0)
            static_bull_score = pd.Series(np.mean(bull_align_matrix, axis=0), index=df.index, dtype=np.float32)
            static_bear_score = pd.Series(np.mean(bear_align_matrix, axis=0), index=df.index, dtype=np.float32)
        else:
            static_bull_score = static_bear_score = pd.Series(0.5, index=df.index, dtype=np.float32)

        # 简化的 get_avg_score 辅助函数
        def get_avg_score(cols: list[str], asc: bool) -> pd.Series:
            if not cols: return pd.Series(0.5, index=df.index, dtype=np.float32)
            # 使用列表推导式和np.mean进行向量化计算
            scores_matrix = np.stack([normalize_score(df.get(c), df.index, norm_window, ascending=asc).values for c in cols], axis=0)
            return pd.Series(np.mean(scores_matrix, axis=0), index=df.index, dtype=np.float32)

        weekly_slope_cols = [c for c in df.columns if 'SLOPE' in c and c.endswith('_W')]
        weekly_accel_cols = [c for c in df.columns if 'ACCEL' in c and c.endswith('_W')]
        
        dynamic_bull_score = (get_avg_score(weekly_slope_cols, True) * get_avg_score(weekly_accel_cols, True))**0.5
        dynamic_bear_score = (get_avg_score(weekly_slope_cols, False) * get_avg_score(weekly_accel_cols, False))**0.5

        # 使用字典推导式一次性生成所有周期的分数，避免在循环中重复赋值
        s_bull = {p: static_bull_score for p in periods}
        s_bear = {p: static_bear_score for p in periods}
        d_bull = {p: dynamic_bull_score for p in periods}
        d_bear = {p: dynamic_bear_score for p in periods}
        
        return s_bull, d_bull, s_bear, d_bear

    def _calculate_pattern_health(self, df: pd.DataFrame, periods: list, norm_window: int, dynamic_weights: Dict) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        【V2.1 · 性能优化版】计算形态支柱的四维健康度
        - 本次优化:
          - [效率] 将静态分和动态分的计算移出循环，一次性生成后通过字典推导式赋值给所有周期，
                    避免了不必要的重复计算。
        """
        # 静态健康度：是否存在吸筹/盘整形态 vs 派发形态
        is_accumulation = df.get('is_accumulation_D', 0).astype(float)
        is_consolidation = df.get('is_consolidation_D', 0).astype(float)
        is_distribution = df.get('is_distribution_D', 0).astype(float)
        static_bull_score = pd.Series(np.maximum(is_accumulation, is_consolidation), index=df.index).replace(0, 0.5)
        static_bear_score = pd.Series(is_distribution, index=df.index).replace(0, 0.5)

        # 动态健康度：是否存在突破 vs 破位事件
        is_breakthrough = df.get('is_breakthrough_D', 0).astype(float)
        is_breakdown = df.get('is_breakdown_D', 0).astype(float)
        dynamic_bull_score = pd.Series(is_breakthrough, index=df.index).replace(0, 0.5)
        dynamic_bear_score = pd.Series(is_breakdown, index=df.index).replace(0, 0.5)

        # 使用字典推导式一次性生成所有周期的分数，避免在循环中重复赋值
        s_bull = {p: static_bull_score for p in periods}
        s_bear = {p: static_bear_score for p in periods}
        d_bull = {p: dynamic_bull_score for p in periods}
        d_bear = {p: dynamic_bear_score for p in periods}
        
        return s_bull, d_bull, s_bear, d_bear






















