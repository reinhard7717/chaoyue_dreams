# 文件: strategies/trend_following/intelligence/structural_intelligence.py
# 结构情报模块 (均线, 箱体, 平台, 趋势)
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following.utils import transmute_health_to_ultimate_signals, get_params_block, get_param_value, normalize_score, normalize_to_bipolar

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
        【V17.0 · 关系元分析版】
        - 核心革命: 1. 引入关系元分析，重构四大支柱健康度计算。
                      2. 使用加权几何平均融合四大支柱，体现不同支柱的重要性。
        """
        states = {}
        p_conf = get_params_block(self.strategy, 'structural_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True): return states
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        periods = get_param_value(p_synthesis.get('periods'), [1, 5, 13, 21, 55])
        norm_window = get_param_value(p_synthesis.get('norm_window'), 55)
        # 获取支柱权重
        pillar_weights = get_param_value(p_conf.get('pillar_weights'), {'ma': 0.25, 'mechanics': 0.25, 'mtf': 0.25, 'pattern': 0.25})
        # 准备存储健康度和权重的容器
        health_data = { 's_bull': [], 's_bear': [], 'd_intensity': [] }
        pillar_names_in_order = ['ma', 'mechanics', 'mtf', 'pattern']
        weights_in_order = [pillar_weights.get(name, 0.25) for name in pillar_names_in_order]
        calculators = {
            'ma': self._calculate_ma_health,
            'mechanics': self._calculate_mechanics_health,
            'mtf': self._calculate_mtf_health,
            'pattern': self._calculate_pattern_health
        }
        # 循环计算并存储每个支柱的健康度
        for name in pillar_names_in_order:
            calculator = calculators[name]
            s_bull, s_bear, d_intensity = calculator(df, periods, norm_window)
            health_data['s_bull'].append(s_bull)
            health_data['s_bear'].append(s_bear)
            health_data['d_intensity'].append(d_intensity)
        overall_health = {}
        for health_type, health_sources in health_data.items():
            overall_health[health_type] = {}
            for p in periods:
                components_for_period = [pillar_dict[p].values for pillar_dict in health_sources if p in pillar_dict]
                if components_for_period:
                    stacked_values = np.stack(components_for_period, axis=0)
                    # 实现加权几何平均
                    # weights_array = np.array(weights_in_order).reshape(-1, 1)
                    # weighted_prod = np.prod(stacked_values ** weights_array, axis=0)
                    # fused_values = weighted_prod # 权重和为1，无需再开方
                    # 修正：权重和可能不为1，需要归一化
                    total_weight = sum(weights_in_order)
                    if total_weight > 0:
                        normalized_weights = np.array(weights_in_order) / total_weight
                        weights_array = normalized_weights.reshape(-1, 1)
                        fused_values = np.prod(stacked_values ** weights_array, axis=0)
                    else:
                        fused_values = np.prod(stacked_values, axis=0) ** (1.0 / stacked_values.shape[0])
                    overall_health[health_type][p] = pd.Series(fused_values, index=df.index, dtype=np.float32)
                else:
                    overall_health[health_type][p] = pd.Series(0.5, index=df.index, dtype=np.float32)
        self.strategy.atomic_states['__STRUCTURE_overall_health'] = overall_health
        ultimate_signals = transmute_health_to_ultimate_signals(
            df=df,
            atomic_states=self.strategy.atomic_states,
            overall_health=overall_health,
            params=p_synthesis,
            domain_prefix="STRUCTURE"
        )
        states.update(ultimate_signals)
        return states

    # ==============================================================================
    # 以下为重构后的健康度组件计算器，现在返回四维健康度
    # ==============================================================================
    def _calculate_ma_health(self, df: pd.DataFrame, periods: list, norm_window: int) -> Tuple[Dict, Dict, Dict]:
        """【V4.0 · 关系元分析版】计算MA支柱的三维健康度"""
        s_bull, s_bear, d_intensity = {}, {}, {}
        p_conf = get_params_block(self.strategy, 'structural_ultimate_params', {})
        fusion_weights = get_param_value(p_conf.get('ma_health_fusion_weights'), {'alignment': 0.1, 'slope': 0.2, 'accel': 0.2, 'relational': 0.5})
        ma_periods = [5, 13, 21, 55]
        # 步骤一：计算原始的、纯粹的MA健康度分数
        bull_alignment_scores, bear_alignment_scores = [], []
        for i in range(len(ma_periods) - 1):
            short_col, long_col = f'EMA_{ma_periods[i]}_D', f'EMA_{ma_periods[i+1]}_D'
            if short_col in df and long_col in df:
                bull_alignment_scores.append((df[short_col] > df[long_col]).astype(float))
                bear_alignment_scores.append((df[short_col] < df[long_col]).astype(float))
        alignment_score = pd.DataFrame(bull_alignment_scores).mean().fillna(0.5) if bull_alignment_scores else pd.Series(0.5, index=df.index)
        static_bear_score = pd.DataFrame(bear_alignment_scores).mean().fillna(0.5) if bear_alignment_scores else pd.Series(0.5, index=df.index)
        slope_health_scores, accel_health_scores, relational_health_scores = [], [], []
        for p in ma_periods:
            slope_col = f'SLOPE_{p}_EMA_{p}_D' if p != 1 else f'SLOPE_1_close_D'
            accel_col = f'ACCEL_{p}_EMA_{p}_D' if p != 1 else f'ACCEL_1_close_D'
            if slope_col in df.columns:
                slope_health_scores.append((normalize_to_bipolar(df[slope_col], df.index, norm_window) + 1) / 2.0)
            if accel_col in df.columns:
                accel_health_scores.append((normalize_to_bipolar(df[accel_col], df.index, norm_window) + 1) / 2.0)
        for short_p, long_p in [(5, 21), (13, 55)]:
            spread_accel = (df[f'EMA_{short_p}_D'] - df[f'EMA_{long_p}_D']).diff(3).diff(3).fillna(0)
            relational_health_scores.append((normalize_to_bipolar(spread_accel, df.index, norm_window) + 1) / 2.0)
        avg_slope_health = pd.concat(slope_health_scores, axis=1).mean(axis=1).fillna(0.5) if slope_health_scores else pd.Series(0.5, index=df.index)
        avg_accel_health = pd.concat(accel_health_scores, axis=1).mean(axis=1).fillna(0.5) if accel_health_scores else pd.Series(0.5, index=df.index)
        avg_relational_health = pd.concat(relational_health_scores, axis=1).mean(axis=1).fillna(0.5) if relational_health_scores else pd.Series(0.5, index=df.index)
        raw_ma_health_score = (
            alignment_score * fusion_weights.get('alignment', 0.1) +
            avg_slope_health * fusion_weights.get('slope', 0.2) +
            avg_accel_health * fusion_weights.get('accel', 0.2) +
            avg_relational_health * fusion_weights.get('relational', 0.5)
        )
        # 步骤二：MA健康度本身就是与均线结构的关系，直接作为快照分
        snapshot_score = raw_ma_health_score
        # 步骤三：对快照分进行关系元分析，得到最终的动态调制分数
        unified_d_intensity = self._perform_structural_relational_meta_analysis(df, snapshot_score)
        for p in periods:
            s_bull[p] = snapshot_score
            s_bear[p] = static_bear_score # 看跌分保持简单，只看排列
            d_intensity[p] = unified_d_intensity
        return s_bull, s_bear, d_intensity

    def _calculate_mechanics_health(self, df: pd.DataFrame, periods: list, norm_window: int) -> Tuple[Dict, Dict, Dict]:
        """【V4.0 · 关系元分析版】计算力学支柱的三维健康度"""
        s_bull, s_bear, d_intensity = {}, {}, {}
        # 步骤一：计算原始的、纯粹的力学健康度分数
        raw_mechanics_score = normalize_score(df.get('energy_ratio_D'), df.index, norm_window, ascending=True)
        # 步骤二：获取均线趋势上下文分数
        ma_context_score = self._calculate_ma_trend_context(df, [5, 13, 21, 55])
        # 步骤三：构建融合了趋势上下文的“瞬时关系快照分”
        snapshot_score = raw_mechanics_score * ma_context_score
        # 步骤四：对快照分进行关系元分析，得到最终的动态调制分数
        unified_d_intensity = self._perform_structural_relational_meta_analysis(df, snapshot_score)
        # 看跌分也应体现关系
        bear_snapshot_score = normalize_score(df.get('energy_ratio_D'), df.index, norm_window, ascending=False) * (1 - ma_context_score)
        for p in periods:
            s_bull[p] = snapshot_score
            s_bear[p] = bear_snapshot_score
            d_intensity[p] = unified_d_intensity
        return s_bull, s_bear, d_intensity

    def _calculate_mtf_health(self, df: pd.DataFrame, periods: list, norm_window: int) -> Tuple[Dict, Dict, Dict]:
        """【V3.0 · 关系元分析版】计算MTF(多时间框架)支柱的三维健康度"""
        s_bull, s_bear, d_intensity = {}, {}, {}
        # 步骤一：计算原始的、纯粹的MTF健康度分数 (周线结构)
        weekly_cols = [col for col in df.columns if 'EMA' in col and col.endswith('_W')]
        if len(weekly_cols) > 1:
            bull_align_matrix = np.stack([(df[weekly_cols[i]] > df[weekly_cols[i+1]]).values for i in range(len(weekly_cols)-1)], axis=0)
            bear_align_matrix = np.stack([(df[weekly_cols[i]] < df[weekly_cols[i+1]]).values for i in range(len(weekly_cols)-1)], axis=0)
            raw_mtf_bull_score = pd.Series(np.mean(bull_align_matrix, axis=0), index=df.index, dtype=np.float32)
            raw_mtf_bear_score = pd.Series(np.mean(bear_align_matrix, axis=0), index=df.index, dtype=np.float32)
        else:
            raw_mtf_bull_score = raw_mtf_bear_score = pd.Series(0.5, index=df.index, dtype=np.float32)
        # 步骤二：获取均线趋势上下文分数 (日线结构)
        ma_context_score = self._calculate_ma_trend_context(df, [5, 13, 21, 55])
        # 步骤三：构建“瞬时关系快照分”(周线与日线结构的共振)
        snapshot_score = raw_mtf_bull_score * ma_context_score
        # 步骤四：对快照分进行关系元分析
        unified_d_intensity = self._perform_structural_relational_meta_analysis(df, snapshot_score)
        bear_snapshot_score = raw_mtf_bear_score * (1 - ma_context_score)
        for p in periods:
            s_bull[p] = snapshot_score
            s_bear[p] = bear_snapshot_score
            d_intensity[p] = unified_d_intensity
        return s_bull, s_bear, d_intensity

    def _calculate_pattern_health(self, df: pd.DataFrame, periods: list, norm_window: int) -> Tuple[Dict, Dict, Dict]:
        """【V3.0 · 关系元分析版】计算形态支柱的三维健康度"""
        s_bull, s_bear, d_intensity = {}, {}, {}
        # 步骤一：计算原始的、纯粹的形态分数
        is_accumulation = df.get('is_accumulation_D', 0).astype(float)
        is_consolidation = df.get('is_consolidation_D', 0).astype(float)
        is_distribution = df.get('is_distribution_D', 0).astype(float)
        raw_pattern_bull_score = pd.Series(np.maximum(is_accumulation, is_consolidation), index=df.index)
        raw_pattern_bear_score = pd.Series(is_distribution, index=df.index)
        # 步骤二：获取均线趋势上下文分数
        ma_context_score = self._calculate_ma_trend_context(df, [5, 13, 21, 55])
        # 步骤三：构建“瞬时关系快照分”
        snapshot_score = raw_pattern_bull_score * ma_context_score
        # 步骤四：对快照分进行关系元分析
        unified_d_intensity = self._perform_structural_relational_meta_analysis(df, snapshot_score)
        bear_snapshot_score = raw_pattern_bear_score * (1 - ma_context_score)
        for p in periods:
            s_bull[p] = snapshot_score
            s_bear[p] = bear_snapshot_score
            d_intensity[p] = unified_d_intensity
        return s_bull, s_bear, d_intensity

    def _perform_structural_relational_meta_analysis(self, df: pd.DataFrame, snapshot_score: pd.Series) -> pd.Series:
        """
        【V1.0 · 新增】结构专用的关系元分析核心引擎 (赫拉织布机V2)
        - 核心逻辑: 实现“状态 * (1 + 动态杠杆)”的动态价值调制范式。
        """
        # 从配置中获取动态杠杆权重
        p_conf = get_params_block(self.strategy, 'structural_ultimate_params', {})
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
        【V1.0 · 新增】计算均线趋势上下文分数
        - 核心逻辑: 评估短期、中期、长期均线的排列和价格位置，输出一个统一的趋势健康分。
        """
        # 确保所有需要的均线都存在
        ma_cols = [f'EMA_{p}_D' for p in periods]
        if not all(col in df.columns for col in ma_cols):
            return pd.Series(0.5, index=df.index)
        # 均线排列健康度
        alignment_scores = []
        for i in range(len(periods) - 1):
            short_ma = df[f'EMA_{periods[i]}_D']
            long_ma = df[f'EMA_{periods[i+1]}_D']
            alignment_scores.append((short_ma > long_ma).astype(float))
        alignment_health = np.mean(alignment_scores, axis=0) if alignment_scores else np.full(len(df.index), 0.5)
        # 价格位置健康度 (价格应在所有均线之上)
        position_scores = [(df['close_D'] > df[col]).astype(float) for col in ma_cols]
        position_health = np.mean(position_scores, axis=0) if position_scores else np.full(len(df.index), 0.5)
        # 融合得到最终的趋势上下文分数
        ma_context_score = pd.Series((alignment_health * position_health)**0.5, index=df.index)
        return ma_context_score.astype(np.float32)





















