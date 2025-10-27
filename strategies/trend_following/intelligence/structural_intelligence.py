# 文件: strategies/trend_following/intelligence/structural_intelligence.py
# 结构情报模块 (均线, 箱体, 平台, 趋势)
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following.utils import transmute_health_to_ultimate_signals, get_params_block, get_param_value, normalize_score, normalize_to_bipolar, bipolar_to_exclusive_unipolar

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
        【V19.0 · 结构势能重构版】
        - 核心重构: 对四大支柱进行全面升级，引入筹码结构、波动性压缩、筹码断层等更高维度的分析。
        - 支柱一/二: 将均线基准从 EMA 全面切换到 MA，以获取更稳定的结构信号。
        - 支柱三: 升级为“多维突破势能引擎”，融合经典箱体、波动性压缩、筹码断层三大突破模式。
        - 支柱四: 废除旧的“形态确认”，替换为全新的“结构稳定性”支柱，基于筹码峰的支撑与压力进行评估。
        """
        states = {}
        p_conf = get_params_block(self.strategy, 'structural_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True): return states
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        periods = get_param_value(p_synthesis.get('periods'), [1, 5, 13, 21, 55])
        norm_window = get_param_value(p_synthesis.get('norm_window'), 55)
        pillar_weights = get_param_value(p_conf.get('pillar_weights'), {})
        health_data = { 's_bull': [], 's_bear': [], 'd_intensity': [] }
        # [代码修改开始]
        # 更新支柱名称以匹配新的方法
        pillar_names_in_order = ['trend_integrity', 'mtf_cohesion', 'breakout_potential', 'structural_stability']
        # [代码修改结束]
        weights_in_order = [pillar_weights.get(name, 0.25) for name in pillar_names_in_order]
        ti_s_bull, ti_s_bear, ti_d_intensity, daily_bipolar_snapshot = self._calculate_trend_integrity_health(df, periods, norm_window)
        health_data['s_bull'].append(ti_s_bull)
        health_data['s_bear'].append(ti_s_bear)
        health_data['d_intensity'].append(ti_d_intensity)
        mtf_s_bull, mtf_s_bear, mtf_d_intensity = self._calculate_mtf_cohesion_health(df, periods, norm_window, daily_bipolar_snapshot)
        health_data['s_bull'].append(mtf_s_bull)
        health_data['s_bear'].append(mtf_s_bear)
        health_data['d_intensity'].append(mtf_d_intensity)
        bp_s_bull, bp_s_bear, bp_d_intensity = self._calculate_breakout_potential_health(df, periods, norm_window)
        health_data['s_bull'].append(bp_s_bull)
        health_data['s_bear'].append(bp_s_bear)
        health_data['d_intensity'].append(bp_d_intensity)
        # [代码修改开始]
        # 调用新的“结构稳定性”支柱方法
        ss_s_bull, ss_s_bear, ss_d_intensity = self._calculate_structural_stability_health(df, periods, norm_window)
        health_data['s_bull'].append(ss_s_bull)
        health_data['s_bear'].append(ss_s_bear)
        health_data['d_intensity'].append(ss_d_intensity)
        # [代码修改结束]
        overall_health = {}
        for health_type, health_sources in health_data.items():
            overall_health[health_type] = {}
            for p in periods:
                total_weight = sum(weights_in_order)
                if total_weight > 0:
                    weighted_scores = [
                        pillar_dict[p].fillna(0.5) * (weights_in_order[i] / total_weight)
                        for i, pillar_dict in enumerate(health_sources) if p in pillar_dict
                    ]
                    fused_score = sum(weighted_scores) if weighted_scores else pd.Series(0.5, index=df.index)
                else:
                    components = [pillar_dict[p].fillna(0.5) for pillar_dict in health_sources if p in pillar_dict]
                    fused_score = sum(components) / len(components) if components else pd.Series(0.5, index=df.index)
                overall_health[health_type][p] = fused_score.astype(np.float32)
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
        # [代码修改开始]
        final_dynamic_score = self._perform_relational_meta_analysis(df, bipolar_snapshot)
        # [代码修改结束]
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
        # [代码修改开始]
        final_dynamic_score = self._perform_relational_meta_analysis(df, fused_bipolar_snapshot)
        # [代码修改结束]
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
        # [代码修改开始]
        final_dynamic_score = self._perform_relational_meta_analysis(df, bipolar_snapshot)
        # [代码修改结束]
        final_bull_score, final_bear_score = bipolar_to_exclusive_unipolar(final_dynamic_score)
        unified_d_intensity = pd.Series(1.0, index=df.index, dtype=np.float32)
        for p in periods:
            s_bull[p] = final_bull_score
            s_bear[p] = final_bear_score
            d_intensity[p] = unified_d_intensity
        return s_bull, s_bear, d_intensity

    def _calculate_structural_stability_health(self, df: pd.DataFrame, periods: list, norm_window: int) -> Tuple[Dict, Dict, Dict]:
        """
        【V1.1 · 统一命名版】支柱四：结构稳定性
        - 核心修复: 调用重命名后的 `_perform_relational_meta_analysis`。
        """
        s_bull, s_bear, d_intensity = {}, {}, {}
        peak_stability = normalize_score(df.get('peak_stability_D', pd.Series(0, index=df.index)), df.index, norm_window)
        peak_control = normalize_score(df.get('peak_control_ratio_D', pd.Series(0, index=df.index)), df.index, norm_window)
        support_below = normalize_score(df.get('support_below_D', pd.Series(0, index=df.index)), df.index, norm_window)
        bull_snapshot_score = (peak_stability * peak_control * support_below)**(1/3)
        bear_snapshot_score = normalize_score(df.get('pressure_above_D', pd.Series(0, index=df.index)), df.index, norm_window)
        bipolar_snapshot = (bull_snapshot_score - bear_snapshot_score).clip(-1, 1)
        # [代码修改开始]
        final_dynamic_score = self._perform_relational_meta_analysis(df, bipolar_snapshot)
        # [代码修改结束]
        final_bull_score, final_bear_score = bipolar_to_exclusive_unipolar(final_dynamic_score)
        unified_d_intensity = pd.Series(1.0, index=df.index, dtype=np.float32)
        for p in periods:
            s_bull[p] = final_bull_score
            s_bear[p] = final_bear_score
            d_intensity[p] = unified_d_intensity
        return s_bull, s_bear, d_intensity

    def _perform_relational_meta_analysis(self, df: pd.DataFrame, snapshot_score: pd.Series) -> pd.Series:
        """
        【V4.0 · 状态主导协议版】结构专用的关系元分析核心引擎
        - 核心重构: 废除旧名 `_perform_structural_relational_meta_analysis`，统一命名。
        - 核心修复: 植入“状态主导协议”。在计算出最终动态分后增加一道“护栏”：
                      如果原始快照分是正，则最终结果最低为0，绝不允许被负向动态拖入负值区。
                      反之亦然。这从根本上解决了“动态压制”问题。
        """
        # [代码修改开始]
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
        relationship_accel = relationship_trend.diff(meta_window).fillna(0)
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
        # 植入“状态主导协议”护栏
        final_score = np.where(snapshot_score >= 0, net_force.clip(lower=0), net_force.clip(upper=0))
        return pd.Series(final_score, index=df.index, dtype=np.float32)
        # [代码修改结束]


        





















