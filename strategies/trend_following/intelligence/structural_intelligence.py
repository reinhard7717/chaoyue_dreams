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
        【V18.3 · 权重校准版】
        - 核心修复: 修正了四大支柱融合时的权重计算错误。现在确保每个支柱的贡献是基于其
                      在总权重中的正确比例，彻底修复了融合结果不正确的问题。
        - 优化: 将融合逻辑重构为更清晰、更安全的函数式风格（列表推导式 + sum）。
        """
        states = {}
        p_conf = get_params_block(self.strategy, 'structural_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True): return states
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        periods = get_param_value(p_synthesis.get('periods'), [1, 5, 13, 21, 55])
        norm_window = get_param_value(p_synthesis.get('norm_window'), 55)
        pillar_weights = get_param_value(p_conf.get('pillar_weights'), {})
        health_data = { 's_bull': [], 's_bear': [], 'd_intensity': [] }
        pillar_names_in_order = ['trend_integrity', 'mtf_cohesion', 'breakout_potential', 'pattern_confirmation']
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
        pc_s_bull, pc_s_bear, pc_d_intensity = self._calculate_pattern_health(df, periods, norm_window)
        health_data['s_bull'].append(pc_s_bull)
        health_data['s_bear'].append(pc_s_bear)
        health_data['d_intensity'].append(pc_d_intensity)
        overall_health = {}
        for health_type, health_sources in health_data.items():
            overall_health[health_type] = {}
            for p in periods:
                # [代码修改开始]
                total_weight = sum(weights_in_order)
                if total_weight > 0:
                    # 使用列表推导式创建一个包含所有加权分数的列表
                    weighted_scores = [
                        pillar_dict[p].fillna(0.5) * (weights_in_order[i] / total_weight)
                        for i, pillar_dict in enumerate(health_sources) if p in pillar_dict
                    ]
                    # 使用sum()对列表中的所有Series求和
                    if weighted_scores:
                        fused_score = sum(weighted_scores)
                    else:
                        fused_score = pd.Series(0.5, index=df.index)
                else:
                    components = [pillar_dict[p].fillna(0.5) for pillar_dict in health_sources if p in pillar_dict]
                    if components:
                        fused_score = sum(components) / len(components)
                    else:
                        fused_score = pd.Series(0.5, index=df.index)
                overall_health[health_type][p] = fused_score.astype(np.float32)
                # [代码修改结束]
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
        【V2.1 · 信号链路版】支柱一：趋势完整性
        - 核心升级: 增加返回其计算出的双极性快照分 `bipolar_snapshot`，为下游支柱提供正确的信号输入。
        """
        # [代码修改开始]
        s_bull, s_bear, d_intensity = {}, {}, {}
        p_conf = get_params_block(self.strategy, 'structural_ultimate_params', {})
        neutral_zone_threshold = get_param_value(p_conf.get('neutral_zone_threshold'), 0.1)
        fusion_weights = get_param_value(p_conf.get('ma_health_fusion_weights'), {})
        ma_periods = [5, 13, 21, 55]
        required_cols = [f'EMA_{p}_D' for p in ma_periods]
        if not all(col in df.columns for col in required_cols):
            default_series = pd.Series(0.5, index=df.index, dtype=np.float32)
            bipolar_default = pd.Series(0.0, index=df.index, dtype=np.float32)
            for p in periods:
                s_bull[p], s_bear[p], d_intensity[p] = default_series.copy(), default_series.copy(), default_series.copy()
            return s_bull, s_bear, d_intensity, bipolar_default
        ma_values = np.stack([df[col].values for col in required_cols], axis=0)
        bull_alignment = np.mean([(df[f'EMA_{ma_periods[i]}_D'] > df[f'EMA_{ma_periods[i+1]}_D']).values for i in range(len(ma_periods) - 1)], axis=0)
        bear_alignment = np.mean([(df[f'EMA_{ma_periods[i]}_D'] < df[f'EMA_{ma_periods[i+1]}_D']).values for i in range(len(ma_periods) - 1)], axis=0)
        slope_cols = [f'SLOPE_{p}_EMA_{p}_D' for p in ma_periods if f'SLOPE_{p}_EMA_{p}_D' in df.columns]
        bull_velocity = np.mean([normalize_score(df[col], df.index, norm_window, ascending=True).values for col in slope_cols], axis=0) if slope_cols else 0.5
        bear_velocity = np.mean([normalize_score(df[col], df.index, norm_window, ascending=False).values for col in slope_cols], axis=0) if slope_cols else 0.5
        accel_cols = [f'ACCEL_{p}_EMA_{p}_D' for p in ma_periods if f'ACCEL_{p}_EMA_{p}_D' in df.columns]
        bull_acceleration = np.mean([normalize_score(df[col], df.index, norm_window, ascending=True).values for col in accel_cols], axis=0) if accel_cols else 0.5
        bear_acceleration = np.mean([normalize_score(df[col], df.index, norm_window, ascending=False).values for col in accel_cols], axis=0) if accel_cols else 0.5
        ma_std = np.std(ma_values / df['close_D'].values[:, np.newaxis].T, axis=0)
        bull_relational = 1.0 - normalize_score(pd.Series(ma_std, index=df.index), df.index, norm_window, ascending=True).values
        bear_relational = normalize_score(pd.Series(ma_std, index=df.index), df.index, norm_window, ascending=True).values
        meta_dynamics_cols = ['SLOPE_5_EMA_55_D', 'SLOPE_13_EMA_89_D']
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
        final_dynamic_score = self._perform_structural_relational_meta_analysis(df, bipolar_snapshot)
        final_bull_score, final_bear_score = bipolar_to_exclusive_unipolar(final_dynamic_score, neutral_zone_threshold)
        unified_d_intensity = pd.Series(1.0, index=df.index, dtype=np.float32)
        for p in periods:
            s_bull[p] = final_bull_score
            s_bear[p] = final_bear_score
            d_intensity[p] = unified_d_intensity
        return s_bull, s_bear, d_intensity, bipolar_snapshot
        # [代码修改结束]

    def _calculate_mtf_cohesion_health(self, df: pd.DataFrame, periods: list, norm_window: int, daily_bipolar_snapshot: pd.Series) -> Tuple[Dict, Dict, Dict]:
        """
        【V2.1 · 双极性加权融合版】支柱二：多时间框架协同
        - 核心重构: 融合逻辑从惩罚性的乘法改为更稳健的加权平均。现在将日线和周线的双极性快照分
                      进行加权融合，能更合理地反映综合趋势，避免因周期稍有不配合就导致信号崩溃。
        """
        # [代码修改开始]
        s_bull, s_bear, d_intensity = {}, {}, {}
        ma_periods_w = [5, 13, 21, 55]
        required_cols_w = [f'EMA_{p}_W' for p in ma_periods_w]
        if not all(col in df.columns for col in required_cols_w):
            weekly_bipolar_snapshot = pd.Series(0.0, index=df.index, dtype=np.float32)
        else:
            weekly_alignment_bull = np.mean([(df[f'EMA_{ma_periods_w[i]}_W'] > df[f'EMA_{ma_periods_w[i+1]}_W']).values for i in range(len(ma_periods_w) - 1)], axis=0)
            weekly_alignment_bear = np.mean([(df[f'EMA_{ma_periods_w[i]}_W'] < df[f'EMA_{ma_periods_w[i+1]}_W']).values for i in range(len(ma_periods_w) - 1)], axis=0)
            weekly_slope_cols = [f'SLOPE_{p}_EMA_{p}_W' for p in ma_periods_w if f'SLOPE_{p}_EMA_{p}_W' in df.columns]
            weekly_velocity_bull = np.mean([normalize_score(df[col], df.index, norm_window, ascending=True).values for col in weekly_slope_cols], axis=0) if weekly_slope_cols else 0.5
            weekly_velocity_bear = np.mean([normalize_score(df[col], df.index, norm_window, ascending=False).values for col in weekly_slope_cols], axis=0) if weekly_slope_cols else 0.5
            weekly_bull_health = weekly_alignment_bull * 0.5 + weekly_velocity_bull * 0.5
            weekly_bear_health = weekly_alignment_bear * 0.5 + weekly_velocity_bear * 0.5
            weekly_bipolar_snapshot = pd.Series(weekly_bull_health - weekly_bear_health, index=df.index, dtype=np.float32).clip(-1, 1)
        # --- 使用加权平均融合日线和周线的双极性快照分 ---
        fused_bipolar_snapshot = (daily_bipolar_snapshot * 0.7 + weekly_bipolar_snapshot * 0.3)
        final_dynamic_score = self._perform_structural_relational_meta_analysis(df, fused_bipolar_snapshot)
        p_conf = get_params_block(self.strategy, 'structural_ultimate_params', {})
        neutral_zone_threshold = get_param_value(p_conf.get('neutral_zone_threshold'), 0.1)
        final_bull_score, final_bear_score = bipolar_to_exclusive_unipolar(final_dynamic_score, neutral_zone_threshold)
        unified_d_intensity = pd.Series(1.0, index=df.index, dtype=np.float32)
        for p in periods:
            s_bull[p] = final_bull_score
            s_bear[p] = final_bear_score
            d_intensity[p] = unified_d_intensity
        return s_bull, s_bear, d_intensity
        # [代码修改结束]

    def _calculate_breakout_potential_health(self, df: pd.DataFrame, periods: list, norm_window: int) -> Tuple[Dict, Dict, Dict]:
        """
        【V1.1 · 赫利俄斯敕令版】支柱三：结构突破潜力 (赫淮斯托斯的火山熔炉)
        - 核心革命: 遵循“赫利俄斯敕令”，对双极性快照分执行关系元分析，得到最终动态分，再派生s_bull/s_bear。
        """
        s_bull, s_bear, d_intensity = {}, {}, {}
        score_breakout = (df['close_D'] > df.get('dynamic_consolidation_high_D', np.inf)).astype(float)
        score_breakdown = (df['close_D'] < df.get('dynamic_consolidation_low_D', -np.inf)).astype(float)
        bbw_slope = df.get('SLOPE_5_BBW_21_2.0_D', pd.Series(0, index=df.index))
        atr_slope = df.get('SLOPE_5_ATR_14_D', pd.Series(0, index=df.index))
        energy_expansion_score = (normalize_score(bbw_slope.clip(lower=0), df.index, norm_window) * normalize_score(atr_slope.clip(lower=0), df.index, norm_window))**0.5
        bull_snapshot_score = (score_breakout * energy_expansion_score).astype(np.float32)
        bear_snapshot_score = (score_breakdown * energy_expansion_score).astype(np.float32)
        # 遵循“赫利俄斯敕令”
        # 1. 计算双极性快照分
        bipolar_snapshot = (bull_snapshot_score - bear_snapshot_score).clip(-1, 1)
        # 2. 对双极性快照分执行关系元分析，得到最终的动态健康分
        final_dynamic_score = self._perform_structural_relational_meta_analysis(df, bipolar_snapshot)
        # 3. 从最终动态分中互斥地派生出 s_bull 和 s_bear
        final_bull_score = final_dynamic_score.clip(0, 1).astype(np.float32)
        final_bear_score = (final_dynamic_score.clip(-1, 0) * -1).astype(np.float32)
        # 4. 将 d_intensity 降级为无意义的占位符
        unified_d_intensity = pd.Series(1.0, index=df.index, dtype=np.float32)
        for p in periods:
            s_bull[p] = final_bull_score
            s_bear[p] = final_bear_score
            d_intensity[p] = unified_d_intensity
        
        return s_bull, s_bear, d_intensity

    def _calculate_pattern_health(self, df: pd.DataFrame, periods: list, norm_window: int) -> Tuple[Dict, Dict, Dict]:
        """
        【V5.1 · 赫利俄斯敕令版】支柱四：形态确认 (阿波罗的竖琴)
        - 核心革命: 遵循“赫利俄斯敕令”，对双极性快照分执行关系元分析，得到最终动态分，再派生s_bull/s_bear。
        """
        s_bull, s_bear, d_intensity = {}, {}, {}
        bullish_resonance = self.strategy.atomic_states.get('SCORE_PATTERN_BULLISH_RESONANCE', pd.Series(0.5, index=df.index))
        bottom_reversal = self.strategy.atomic_states.get('SCORE_PATTERN_BOTTOM_REVERSAL', pd.Series(0.5, index=df.index))
        bearish_resonance = self.strategy.atomic_states.get('SCORE_PATTERN_BEARISH_RESONANCE', pd.Series(0.5, index=df.index))
        top_reversal = self.strategy.atomic_states.get('SCORE_PATTERN_TOP_REVERSAL', pd.Series(0.5, index=df.index))
        bull_snapshot_score = np.maximum(bullish_resonance, bottom_reversal)
        bear_snapshot_score = np.maximum(bearish_resonance, top_reversal)
        # 遵循“赫利俄斯敕令”
        # 1. 计算双极性快照分
        bipolar_snapshot = (bull_snapshot_score - bear_snapshot_score).clip(-1, 1)
        # 2. 对双极性快照分执行关系元分析，得到最终的动态健康分
        final_dynamic_score = self._perform_structural_relational_meta_analysis(df, bipolar_snapshot)
        # 3. 从最终动态分中互斥地派生出 s_bull 和 s_bear
        final_bull_score = final_dynamic_score.clip(0, 1).astype(np.float32)
        final_bear_score = (final_dynamic_score.clip(-1, 0) * -1).astype(np.float32)
        # 4. 将 d_intensity 降级为无意义的占位符
        unified_d_intensity = pd.Series(1.0, index=df.index, dtype=np.float32)
        for p in periods:
            s_bull[p] = final_bull_score
            s_bear[p] = final_bear_score
            d_intensity[p] = unified_d_intensity
        
        return s_bull, s_bear, d_intensity

    def _perform_structural_relational_meta_analysis(self, df: pd.DataFrame, snapshot_score: pd.Series) -> pd.Series:
        """
        【V3.0 · 阿瑞斯之怒协议版】结构专用的关系元分析核心引擎
        - 核心革命: 签署“阿瑞斯之怒”协议，废除旧的、错误的单极性剪裁逻辑。
                      引入“双通道”处理，分别计算看涨和看跌力量，最终输出一个[-1, 1]的净值分数，
                      彻底修复了因丢弃负向信息而导致信号失效的致命BUG。
        """
        p_conf = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_meta = get_param_value(p_conf.get('relational_meta_analysis_params'), {})
        w_state = get_param_value(p_meta.get('state_weight'), 0.3)
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.3)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.4)
        norm_window = 55
        meta_window = 5
        bipolar_sensitivity = 1.0
        # 维度二：速度分 (Velocity Score) - 范围 [-1, 1]
        relationship_trend = snapshot_score.diff(meta_window).fillna(0)
        velocity_score = normalize_to_bipolar(
            series=relationship_trend, target_index=df.index,
            window=norm_window, sensitivity=bipolar_sensitivity
        )
        # 维度三：加速度分 (Acceleration Score) - 范围 [-1, 1]
        relationship_accel = relationship_trend.diff(meta_window).fillna(0)
        acceleration_score = normalize_to_bipolar(
            series=relationship_accel, target_index=df.index,
            window=norm_window, sensitivity=bipolar_sensitivity
        )
        # --- 看涨力量评估 (Bullish Force) ---
        bullish_state = snapshot_score.clip(0, 1)
        bullish_velocity = velocity_score.clip(0, 1)
        bullish_acceleration = acceleration_score.clip(0, 1)
        total_bullish_force = (
            bullish_state * w_state +
            bullish_velocity * w_velocity +
            bullish_acceleration * w_acceleration
        )
        # --- 看跌力量评估 (Bearish Force) ---
        bearish_state = (snapshot_score.clip(-1, 0) * -1)
        bearish_velocity = (velocity_score.clip(-1, 0) * -1)
        bearish_acceleration = (acceleration_score.clip(-1, 0) * -1)
        total_bearish_force = (
            bearish_state * w_state +
            bearish_velocity * w_velocity +
            bearish_acceleration * w_acceleration
        )
        # --- 净值裁决 (Net Value Adjudication) ---
        final_score = (total_bullish_force - total_bearish_force).clip(-1, 1)
        return final_score.astype(np.float32)


        





















