# 文件: strategies/trend_following/intelligence/structural_intelligence.py
# 结构情报模块 (均线, 箱体, 平台, 趋势)
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score, normalize_to_bipolar, bipolar_to_exclusive_unipolar

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
        【V21.1 · 深度调试版】结构情报的终极信号诊断与合成
        - 核心重构: 废弃对通用函数 transmute_health_to_ultimate_signals 的调用，引入“四象限动态分析法”，
                      彻底解决信号命名与逻辑混乱的问题，确保与筹码、资金流、行为模块的哲学统一。
        """
        states = {}
        p_conf = get_params_block(self.strategy, 'structural_ultimate_params', {})
        periods = get_param_value(get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {}).get('periods'), [1, 5, 13, 21, 55])
        norm_window = get_param_value(p_conf.get('norm_window'), 55)
        pillar_weights = get_param_value(p_conf.get('pillar_weights'), {})
        resonance_tf_weights = get_param_value(p_conf.get('resonance_tf_weights'), {'short': 0.2, 'medium': 0.5, 'long': 0.3})
        # 步骤一：计算每个支柱的健康度
        ti_s_bull, ti_s_bear, _, daily_bipolar = self._calculate_trend_integrity_health(df, periods, norm_window)
        mtf_s_bull, mtf_s_bear, _ = self._calculate_mtf_cohesion_health(df, periods, norm_window, daily_bipolar)
        bp_s_bull, bp_s_bear, _ = self._calculate_breakout_potential_health(df, periods, norm_window)
        ss_s_bull, ss_s_bear, _ = self._calculate_structural_stability_health(df, periods, norm_window)
        # 步骤二：计算各支柱的双极性净值
        default_series = pd.Series(0.0, index=df.index, dtype=np.float32)
        pillar_bipolar_scores = {
            'trend_integrity': {p: ti_s_bull.get(p, default_series) - ti_s_bear.get(p, default_series) for p in periods},
            'mtf_cohesion': {p: mtf_s_bull.get(p, default_series) - mtf_s_bear.get(p, default_series) for p in periods},
            'breakout_potential': {p: bp_s_bull.get(p, default_series) - bp_s_bear.get(p, default_series) for p in periods},
            'structural_stability': {p: ss_s_bull.get(p, default_series) - ss_s_bear.get(p, default_series) for p in periods}
        }
        # 步骤三：融合得到最终的、跨周期的双极性总健康分
        period_groups = {'short': [p for p in periods if p <= 5], 'medium': [p for p in periods if 5 < p <= 21], 'long': [p for p in periods if p > 21]}
        final_bipolar_health = pd.Series(0.0, index=df.index, dtype=np.float64)
        total_tf_weight = sum(resonance_tf_weights.values())
        if total_tf_weight > 0:
            for tf_name, weight in resonance_tf_weights.items():
                group_periods = period_groups.get(tf_name, [])
                if not group_periods: continue
                period_fused_scores = []
                for p in group_periods:
                    period_score = pd.Series(0.0, index=df.index, dtype=np.float32)
                    total_pillar_weight = sum(pillar_weights.values())
                    if total_pillar_weight > 0:
                        for pillar_name, p_weight in pillar_weights.items():
                            period_score += pillar_bipolar_scores.get(pillar_name, {}).get(p, default_series) * (p_weight / total_pillar_weight)
                    period_fused_scores.append(period_score)
                if period_fused_scores:
                    avg_group_score = sum(period_fused_scores) / len(period_fused_scores)
                    final_bipolar_health += avg_group_score * (weight / total_tf_weight)
        final_bipolar_health = final_bipolar_health.clip(-1, 1).astype(np.float32)
        # 步骤四：分离为纯粹的看涨/看跌健康分，并计算静态共振信号
        bullish_health, bearish_health = bipolar_to_exclusive_unipolar(final_bipolar_health)
        states['SCORE_STRUCTURE_BULLISH_RESONANCE'] = bullish_health
        states['SCORE_STRUCTURE_BEARISH_RESONANCE'] = bearish_health
        # 步骤五：计算四象限动态信号
        bull_divergence = self._calculate_holographic_divergence_structural(bullish_health, 5, 21, norm_window)
        bullish_acceleration = bull_divergence.clip(0, 1)
        top_reversal = (bull_divergence.clip(-1, 0) * -1)
        bear_divergence = self._calculate_holographic_divergence_structural(bearish_health, 5, 21, norm_window)
        bearish_acceleration = bear_divergence.clip(0, 1)
        bottom_reversal = (bear_divergence.clip(-1, 0) * -1)
        # 步骤六：赋值给命名准确的终极信号
        states['SCORE_STRUCTURE_BULLISH_ACCELERATION'] = bullish_acceleration.astype(np.float32)
        states['SCORE_STRUCTURE_TOP_REVERSAL'] = top_reversal.astype(np.float32)
        states['SCORE_STRUCTURE_BEARISH_ACCELERATION'] = bearish_acceleration.astype(np.float32)
        states['SCORE_STRUCTURE_BOTTOM_REVERSAL'] = bottom_reversal.astype(np.float32)
        # 步骤七：重铸战术反转信号
        states['SCORE_STRUCTURE_TACTICAL_REVERSAL'] = (bullish_health * top_reversal).clip(0, 1).astype(np.float32)
        return states

    def _calculate_trend_integrity_health(self, df: pd.DataFrame, periods: list, norm_window: int) -> Tuple[Dict, Dict, Dict, pd.Series]:
        """
        【V3.2 · 统一命名版】支柱一：趋势完整性
        - 核心修复: 调用重命名后的 `_perform_relational_meta_analysis`。
        """
        s_bull, s_bear, d_intensity = {}, {}, {}
        p_conf = get_params_block(self.strategy, 'structural_ultimate_params', {})
        fusion_weights = get_param_value(p_conf.get('ma_health_fusion_weights'), {})
        ma_periods = [5, 13, 21, 55]
        required_cols = [f'MA_{p}_D' for p in ma_periods]
        if not all(col in df.columns for col in required_cols):
            default_series = pd.Series(0.5, index=df.index, dtype=np.float32)
            bipolar_default = pd.Series(0.0, index=df.index, dtype=np.float32)
            for p in periods:
                s_bull[p], s_bear[p], d_intensity[p] = default_series.copy(), default_series.copy(), default_series.copy()
            return s_bull, s_bear, d_intensity, bipolar_default
        ma_values = np.stack([df[col].values for col in required_cols], axis=0)
        bull_alignment = np.mean([(df[f'MA_{ma_periods[i]}_D'] > df[f'MA_{ma_periods[i+1]}_D']).values for i in range(len(ma_periods) - 1)], axis=0)
        bear_alignment = np.mean([(df[f'MA_{ma_periods[i]}_D'] < df[f'MA_{ma_periods[i+1]}_D']).values for i in range(len(ma_periods) - 1)], axis=0)
        slope_cols = [f'SLOPE_{p}_MA_{p}_D' for p in ma_periods if f'SLOPE_{p}_MA_{p}_D' in df.columns]
        bull_velocity = np.mean([normalize_score(df[col], df.index, norm_window, ascending=True).values for col in slope_cols], axis=0) if slope_cols else 0.5
        bear_velocity = np.mean([normalize_score(df[col], df.index, norm_window, ascending=False).values for col in slope_cols], axis=0) if slope_cols else 0.5
        accel_cols = [f'ACCEL_{p}_MA_{p}_D' for p in ma_periods if f'ACCEL_{p}_MA_{p}_D' in df.columns]
        bull_acceleration = np.mean([normalize_score(df[col], df.index, norm_window, ascending=True).values for col in accel_cols], axis=0) if accel_cols else 0.5
        bear_acceleration = np.mean([normalize_score(df[col], df.index, norm_window, ascending=False).values for col in accel_cols], axis=0) if accel_cols else 0.5
        ma_std = np.std(ma_values / df['close_D'].values[:, np.newaxis].T, axis=0)
        bull_relational = 1.0 - normalize_score(pd.Series(ma_std, index=df.index), df.index, norm_window, ascending=True).values
        bear_relational = normalize_score(pd.Series(ma_std, index=df.index), df.index, norm_window, ascending=True).values
        meta_dynamics_cols = ['SLOPE_5_MA_55_D', 'SLOPE_13_MA_89_D']
        valid_meta_cols = [col for col in meta_dynamics_cols if col in df.columns]
        bull_meta_dynamics = np.mean([normalize_score(df[col], df.index, norm_window, ascending=True).values for col in valid_meta_cols], axis=0) if valid_meta_cols else 0.5
        bear_meta_dynamics = np.mean([normalize_score(df[col], df.index, norm_window, ascending=False).values for col in valid_meta_cols], axis=0) if valid_meta_cols else 0.5
        bull_score_values = (
            bull_alignment * fusion_weights.get('alignment', 0.15) +
            bull_velocity * fusion_weights.get('slope', 0.15) +
            bull_acceleration * fusion_weights.get('accel', 0.2) +
            bull_relational * fusion_weights.get('relational', 0.25) +
            bull_meta_dynamics * fusion_weights.get('meta_dynamics', 0.25)
        )
        bear_score_values = (
            bear_alignment * fusion_weights.get('alignment', 0.15) +
            bear_velocity * fusion_weights.get('slope', 0.15) +
            bear_acceleration * fusion_weights.get('accel', 0.2) +
            bear_relational * fusion_weights.get('relational', 0.25) +
            bear_meta_dynamics * fusion_weights.get('meta_dynamics', 0.25)
        )
        bipolar_snapshot = pd.Series(bull_score_values - bear_score_values, index=df.index, dtype=np.float32).clip(-1, 1)
        
        final_dynamic_score = self._perform_relational_meta_analysis(df, bipolar_snapshot)
        
        final_bull_score, final_bear_score = bipolar_to_exclusive_unipolar(final_dynamic_score)
        unified_d_intensity = pd.Series(1.0, index=df.index, dtype=np.float32)
        for p in periods:
            s_bull[p] = final_bull_score
            s_bear[p] = final_bear_score
            d_intensity[p] = unified_d_intensity
        return s_bull, s_bear, d_intensity, bipolar_snapshot

    def _calculate_mtf_cohesion_health(self, df: pd.DataFrame, periods: list, norm_window: int, daily_bipolar_snapshot: pd.Series) -> Tuple[Dict, Dict, Dict]:
        """
        【V3.2 · 统一命名版】支柱二：多时间框架协同
        - 核心修复: 调用重命名后的 `_perform_relational_meta_analysis`。
        """
        s_bull, s_bear, d_intensity = {}, {}, {}
        ma_periods_w = [5, 13, 21, 55]
        required_cols_w = [f'MA_{p}_W' for p in ma_periods_w]
        if not all(col in df.columns for col in required_cols_w):
            weekly_bipolar_snapshot = pd.Series(0.0, index=df.index, dtype=np.float32)
        else:
            weekly_alignment_bull = np.mean([(df[f'MA_{ma_periods_w[i]}_W'] > df[f'MA_{ma_periods_w[i+1]}_W']).values for i in range(len(ma_periods_w) - 1)], axis=0)
            weekly_alignment_bear = np.mean([(df[f'MA_{ma_periods_w[i]}_W'] < df[f'MA_{ma_periods_w[i+1]}_W']).values for i in range(len(ma_periods_w) - 1)], axis=0)
            weekly_slope_cols = [f'SLOPE_{p}_MA_{p}_W' for p in ma_periods_w if f'SLOPE_{p}_MA_{p}_W' in df.columns]
            weekly_velocity_bull = np.mean([normalize_score(df[col], df.index, norm_window, ascending=True).values for col in weekly_slope_cols], axis=0) if weekly_slope_cols else 0.5
            weekly_velocity_bear = np.mean([normalize_score(df[col], df.index, norm_window, ascending=False).values for col in weekly_slope_cols], axis=0) if weekly_slope_cols else 0.5
            weekly_bull_health = weekly_alignment_bull * 0.5 + weekly_velocity_bull * 0.5
            weekly_bear_health = weekly_alignment_bear * 0.5 + weekly_velocity_bear * 0.5
            weekly_bipolar_snapshot = pd.Series(weekly_bull_health - weekly_bear_health, index=df.index, dtype=np.float32).clip(-1, 1)
        fused_bipolar_snapshot = (daily_bipolar_snapshot * 0.7 + weekly_bipolar_snapshot * 0.3)
        
        final_dynamic_score = self._perform_relational_meta_analysis(df, fused_bipolar_snapshot)
        
        final_bull_score, final_bear_score = bipolar_to_exclusive_unipolar(final_dynamic_score)
        unified_d_intensity = pd.Series(1.0, index=df.index, dtype=np.float32)
        for p in periods:
            s_bull[p] = final_bull_score
            s_bear[p] = final_bear_score
            d_intensity[p] = unified_d_intensity
        return s_bull, s_bear, d_intensity

    def _calculate_breakout_potential_health(self, df: pd.DataFrame, periods: list, norm_window: int) -> Tuple[Dict, Dict, Dict]:
        """
        【V2.1 · 统一命名版】支柱三：结构突破潜力
        - 核心修复: 调用重命名后的 `_perform_relational_meta_analysis`。
        """
        s_bull, s_bear, d_intensity = {}, {}, {}
        classic_breakout_score = (df['close_D'] > df.get('dynamic_consolidation_high_D', np.inf)).astype(float)
        classic_breakdown_score = (df['close_D'] < df.get('dynamic_consolidation_low_D', -np.inf)).astype(float)
        vol_compression_score = (
            normalize_score(df.get('price_cv_60d_D', pd.Series(1, index=df.index)), df.index, norm_window, ascending=False) *
            normalize_score(df.get('BBW_21_2.0_D', pd.Series(1, index=df.index)), df.index, norm_window, ascending=False)
        )**0.5
        vol_expansion_bull_score = vol_compression_score * (df.get('SLOPE_5_BBW_21_2.0_D', pd.Series(0, index=df.index)).clip(lower=0) > 0).astype(float)
        vol_expansion_bear_score = vol_compression_score * (df.get('SLOPE_5_BBW_21_2.0_D', pd.Series(0, index=df.index)).clip(lower=0) > 0).astype(float)
        fault_breakthrough_score = normalize_score(df.get('fault_breakthrough_intensity_D', pd.Series(0, index=df.index)), df.index, norm_window)
        bull_snapshot_score = np.maximum.reduce([
            classic_breakout_score,
            vol_expansion_bull_score.values,
            fault_breakthrough_score.values
        ]).astype(np.float32)
        bear_snapshot_score = classic_breakdown_score.astype(np.float32)
        bipolar_snapshot = pd.Series(bull_snapshot_score - bear_snapshot_score, index=df.index).clip(-1, 1)
        
        final_dynamic_score = self._perform_relational_meta_analysis(df, bipolar_snapshot)
        
        final_bull_score, final_bear_score = bipolar_to_exclusive_unipolar(final_dynamic_score)
        unified_d_intensity = pd.Series(1.0, index=df.index, dtype=np.float32)
        for p in periods:
            s_bull[p] = final_bull_score
            s_bear[p] = final_bear_score
            d_intensity[p] = unified_d_intensity
        return s_bull, s_bear, d_intensity

    def _calculate_structural_stability_health(self, df: pd.DataFrame, periods: list, norm_window: int) -> Tuple[Dict, Dict, Dict]:
        """
        【V2.0 · 堡垒完整度 vs 围城压力版】支柱四：结构稳定性
        - 核心重构: 废除旧的、过于简单的计算模型。引入全新的“堡垒完整度 vs 围城压力”物理战况模型。
        - 看涨分量 (堡垒完整度): 融合了支撑距离、支撑带成交量、已实现的支撑强度、主峰防守强度和主力支撑强度，
                          从静态和动态两个维度评估下方支撑的坚固程度。
        - 看跌分量 (围城压力): 融合了压力距离、压力带成交量、已实现的压力强度和主力派发压力，
                          从静态和动态两个维度评估上方压力的沉重程度。
        - 收益: 极大提升了该支柱的智能性和鲁棒性，使其能更真实地反映结构层面的多空攻防态势。
        """
        
        s_bull, s_bear, d_intensity = {}, {}, {}
        default_series = pd.Series(0.0, index=df.index, dtype=np.float32)

        # 1. 看涨分量: 堡垒完整度 (Fortress Integrity)
        # 静态城墙厚度 (Static Fortress Thickness)
        static_support_dist = normalize_score(df.get('support_below_D', default_series), df.index, norm_window)
        static_support_vol = normalize_score(df.get('support_below_volume_D', default_series), df.index, norm_window)
        static_fortress_score = (static_support_dist * static_support_vol)**0.5

        # 动态防御行动 (Dynamic Defense Actions)
        realized_support = normalize_score(df.get('realized_support_intensity_D', default_series), df.index, norm_window)
        peak_defense = normalize_score(df.get('peak_defense_intensity_D', default_series), df.index, norm_window)
        main_force_support = normalize_score(df.get('main_force_support_strength_D', default_series), df.index, norm_window)
        dynamic_defense_score = (realized_support * peak_defense * main_force_support)**(1/3)

        # 融合看涨分
        bull_snapshot_score = (static_fortress_score * 0.4 + dynamic_defense_score * 0.6)

        # 2. 看跌分量: 围城压力 (Siege Pressure)
        # 静态兵力规模 (Static Siege Force)
        static_pressure_dist = normalize_score(df.get('pressure_above_D', default_series), df.index, norm_window, ascending=False) # 距离越近，分数越高
        static_pressure_vol = normalize_score(df.get('pressure_above_volume_D', default_series), df.index, norm_window)
        static_siege_score = (static_pressure_dist * static_pressure_vol)**0.5

        # 动态攻击行动 (Dynamic Assault Actions)
        realized_pressure = normalize_score(df.get('realized_pressure_intensity_D', default_series), df.index, norm_window)
        main_force_pressure = normalize_score(df.get('main_force_distribution_pressure_D', default_series), df.index, norm_window)
        dynamic_assault_score = (realized_pressure * main_force_pressure)**0.5

        # 融合看跌分
        bear_snapshot_score = (static_siege_score * 0.4 + dynamic_assault_score * 0.6)

        # 3. 生成双极性快照并进行元分析
        bipolar_snapshot = (bull_snapshot_score - bear_snapshot_score).clip(-1, 1)
        final_dynamic_score = self._perform_relational_meta_analysis(df, bipolar_snapshot)

        # 4. 最终转换与输出
        final_bull_score, final_bear_score = bipolar_to_exclusive_unipolar(final_dynamic_score)
        unified_d_intensity = pd.Series(1.0, index=df.index, dtype=np.float32)
        for p in periods:
            s_bull[p] = final_bull_score
            s_bear[p] = final_bear_score
            d_intensity[p] = unified_d_intensity
        return s_bull, s_bear, d_intensity

    def _calculate_holographic_divergence_structural(self, series: pd.Series, short_p: int, long_p: int, norm_window: int) -> pd.Series:
        """
        【V1.0 · 新增】结构层专用的全息背离计算引擎
        - 战略意义: 洞察多时间维度的“结构性背离”，输出一个[-1, 1]的双极性背离分数。
        """
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

    def _perform_relational_meta_analysis(self, df: pd.DataFrame, snapshot_score: pd.Series) -> pd.Series:
        """
        【V4.1 · 加速度校准版】结构专用的关系元分析核心引擎
        - 核心修复: 修正了加速度计算的致命逻辑错误，应为 relationship_trend.diff(1)。
        """
        p_conf = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_meta = get_param_value(p_conf.get('relational_meta_analysis_params'), {})
        w_state = get_param_value(p_meta.get('state_weight'), 0.3)
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.3)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.4)
        norm_window = 55
        meta_window = 5
        bipolar_sensitivity = 1.0
        relationship_trend = snapshot_score.diff(meta_window).fillna(0)
        velocity_score = normalize_to_bipolar(
            series=relationship_trend, target_index=df.index,
            window=norm_window, sensitivity=bipolar_sensitivity
        )

        # 致命错误修复：加速度是速度(trend)的一阶导数，应使用 diff(1)
        relationship_accel = relationship_trend.diff(1).fillna(0)
        
        acceleration_score = normalize_to_bipolar(
            series=relationship_accel, target_index=df.index,
            window=norm_window, sensitivity=bipolar_sensitivity
        )
        bullish_state = snapshot_score.clip(0, 1)
        bullish_velocity = velocity_score.clip(0, 1)
        bullish_acceleration = acceleration_score.clip(0, 1)
        total_bullish_force = (
            bullish_state * w_state +
            bullish_velocity * w_velocity +
            bullish_acceleration * w_acceleration
        )
        bearish_state = (snapshot_score.clip(-1, 0) * -1)
        bearish_velocity = (velocity_score.clip(-1, 0) * -1)
        bearish_acceleration = (acceleration_score.clip(-1, 0) * -1)
        total_bearish_force = (
            bearish_state * w_state +
            bearish_velocity * w_velocity +
            bearish_acceleration * w_acceleration
        )
        net_force = (total_bullish_force - total_bearish_force).clip(-1, 1)
        final_score = np.where(snapshot_score >= 0, net_force.clip(lower=0), net_force.clip(upper=0))
        return pd.Series(final_score, index=df.index, dtype=np.float32)



        





















