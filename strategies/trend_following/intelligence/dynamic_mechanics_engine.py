# 文件: strategies/trend_following/intelligence/dynamic_mechanics_engine.py
# 动态力学引擎
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following.utils import transmute_health_to_ultimate_signals, get_params_block, get_param_value, normalize_score, normalize_to_bipolar

class DynamicMechanicsEngine:
    def __init__(self, strategy_instance):
        """
        初始化动态力学引擎。
        :param strategy_instance: 策略主实例的引用，用于访问 df_indicators。
        """
        self.strategy = strategy_instance

    def run_dynamic_analysis_command(self) -> Dict[str, pd.Series]: # 修正返回类型注解，并移除 -> None
        """
        【V4.1 · 协议统一版】动态力学引擎总指挥
        - 核心重构: 不再返回None，而是返回一个包含所有生成信号的字典，遵循标准汇报协议。
        """
        # print("    -> [动态力学引擎总指挥 V4.1 · 协议统一版] 启动...") # 更新版本号
        ultimate_dynamic_states = self.diagnose_ultimate_dynamic_mechanics_signals(self.strategy.df_indicators)
        if ultimate_dynamic_states:
            # self.strategy.atomic_states.update(ultimate_dynamic_states) # IntelligenceLayer会做这个
            # print(f"    -> [动态力学引擎总指挥 V4.1] 分析完毕，共生成 {len(ultimate_dynamic_states)} 个终极动态力学信号。")
            # 返回包含所有状态的单一字典
            return ultimate_dynamic_states
        return {} # 如果没有生成信号，返回一个空字典

    def diagnose_ultimate_dynamic_mechanics_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V16.1 · 五大支柱版】动态力学终极信号诊断引擎
        - 火力升级: 新增“能量跃迁”作为第五大支柱，用于捕捉市场从盘整到趋势的“相变”时刻。
        """
        states = {}
        p_conf = get_params_block(self.strategy, 'dynamic_mechanics_params', {})
        if not get_param_value(p_conf.get('enabled'), True): return states
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        pillar_weights = get_param_value(p_conf.get('pillar_weights'), {})
        periods = get_param_value(p_synthesis.get('periods'), [1, 5, 13, 21, 55])
        sorted_periods = sorted(periods)
        norm_window = get_param_value(p_synthesis.get('norm_window'), 55)
        ma_health_score = self._calculate_ma_health(df, p_conf, norm_window)
        overall_health = {'s_bull': {}, 's_bear': {}, 'd_intensity': {}}
        vol_bull_snapshot = (
            normalize_score(df.get('BBW_21_2.0_D'), df.index, norm_window, ascending=False) *
            normalize_score(df.get('ATR_14_D'), df.index, norm_window, ascending=False)
        )**0.5
        vol_bear_snapshot = (
            normalize_score(df.get('BBW_21_2.0_D'), df.index, norm_window, ascending=True) *
            normalize_score(df.get('ATR_14_D'), df.index, norm_window, ascending=True)
        )**0.5
        eff_bull_snapshot = (
            normalize_score(df.get('VPA_EFFICIENCY_D'), df.index, norm_window, ascending=True) *
            normalize_score(df.get('intraday_trend_efficiency_D'), df.index, norm_window, ascending=True)
        )**0.5
        eff_bear_snapshot = (
            normalize_score(df.get('VPA_EFFICIENCY_D'), df.index, norm_window, ascending=False) *
            normalize_score(df.get('intraday_trend_efficiency_D'), df.index, norm_window, ascending=False)
        )**0.5
        mom_bull_snapshot = (
            normalize_score(df.get('ROC_12_D'), df.index, norm_window, ascending=True) *
            normalize_score(df.get('MACDh_13_34_8_D'), df.index, norm_window, ascending=True)
        )**0.5
        mom_bear_snapshot = (
            normalize_score(df.get('ROC_12_D'), df.index, norm_window, ascending=False) *
            normalize_score(df.get('MACDh_13_34_8_D'), df.index, norm_window, ascending=False)
        )**0.5
        adx_strength = normalize_score(df.get('ADX_14_D'), df.index, norm_window)
        adx_direction = (df.get('PDI_14_D', 0) > df.get('NDI_14_D', 0)).astype(float)
        hurst_strength = normalize_score(df.get('hurst_120d_D'), df.index, norm_window)
        ine_bull_snapshot = (adx_strength * adx_direction * hurst_strength)**(1/3)
        ine_bear_snapshot = (
            normalize_score(df.get('ADX_14_D'), df.index, norm_window, ascending=False) *
            normalize_score(df.get('hurst_120d_D'), df.index, norm_window, ascending=False)
        )**0.5
        # 新增第五支柱：能量跃迁
        energy_bull_snapshot = normalize_score(df.get('energy_ratio_D'), df.index, norm_window, ascending=True)
        energy_bear_snapshot = normalize_score(df.get('energy_ratio_D'), df.index, norm_window, ascending=False)
        
        weight_keys = list(pillar_weights.keys())
        weights_array = np.array([pillar_weights.get(name, 1.0/len(weight_keys)) for name in weight_keys])
        weights_array /= weights_array.sum()
        # 将新支柱加入融合列表
        bull_snapshots = [vol_bull_snapshot, eff_bull_snapshot, mom_bull_snapshot, ine_bull_snapshot, energy_bull_snapshot]
        bear_snapshots = [vol_bear_snapshot, eff_bear_snapshot, mom_bear_snapshot, ine_bear_snapshot, energy_bear_snapshot]

        stacked_bull = np.stack([s.fillna(0.5).values for s in bull_snapshots], axis=0)
        stacked_bear = np.stack([s.fillna(0.5).values for s in bear_snapshots], axis=0)
        fused_bull_snapshot = pd.Series(np.prod(stacked_bull ** weights_array[:, np.newaxis], axis=0), index=df.index)
        fused_bear_snapshot = pd.Series(np.prod(stacked_bear ** weights_array[:, np.newaxis], axis=0), index=df.index)
        for i, p in enumerate(sorted_periods):
            context_p = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p
            final_bull_health = self._perform_dynamic_relational_meta_analysis(df, fused_bull_snapshot, p, context_p)
            overall_health['s_bull'][p] = (final_bull_health * ma_health_score).astype(np.float32)
            final_bear_health = self._perform_dynamic_relational_meta_analysis(df, fused_bear_snapshot, p, context_p)
            overall_health['s_bear'][p] = (final_bear_health * (1 - ma_health_score)).astype(np.float32)
            overall_health['d_intensity'][p] = final_bull_health.astype(np.float32)
        self.strategy.atomic_states['__DYN_overall_health'] = overall_health
        ultimate_signals = transmute_health_to_ultimate_signals(
            df=df,
            atomic_states=self.strategy.atomic_states,
            overall_health=overall_health,
            params=p_synthesis,
            domain_prefix="DYN"
        )
        states.update(ultimate_signals)
        return states

    # ==============================================================================
    # 以下为重构后的健康度组件计算器
    # ==============================================================================
    def _calculate_ma_health(self, df: pd.DataFrame, params: dict, norm_window: int) -> pd.Series:
        """
        【V2.0 · 五维元动力版】“赫尔墨斯的商神杖”五维均线健康度评估引擎
        - 核心升级: 引入第五维度“元动力(Meta-Dynamics)”，利用跨周期导数（如SLOPE_5_EMA_55_D）
                      来评估长期趋势的短期变化率，从而获得预判趋势拐点的领先信号。
        """
        # 增加 'meta_dynamics' 权重
        p_ma_health = get_param_value(params.get('ma_health_fusion_weights'), {})
        weights = {
            'alignment': get_param_value(p_ma_health.get('alignment'), 0.15),
            'slope': get_param_value(p_ma_health.get('slope'), 0.15),
            'accel': get_param_value(p_ma_health.get('accel'), 0.2),
            'relational': get_param_value(p_ma_health.get('relational'), 0.25),
            'meta_dynamics': get_param_value(p_ma_health.get('meta_dynamics'), 0.25)
        }

        ma_periods = [5, 13, 21, 55]
        ma_cols = [f'EMA_{p}_D' for p in ma_periods]
        if not all(col in df.columns for col in ma_cols):
            return pd.Series(0.5, index=df.index, dtype=np.float32)
        ma_values = np.stack([df[col].values for col in ma_cols], axis=0)
        # 维度1: 排列健康度
        alignment_bools = ma_values[:-1] > ma_values[1:]
        alignment_health = np.mean(alignment_bools, axis=0) if alignment_bools.size > 0 else np.full(len(df.index), 0.5)
        # 维度2: 速度健康度
        slope_cols = [f'SLOPE_{p}_EMA_{p}_D' for p in ma_periods]
        if all(col in df.columns for col in slope_cols):
            slope_values = np.stack([df[col].values for col in slope_cols], axis=0)
            slope_health = np.mean(normalize_score(pd.Series(slope_values.flatten()), df.index, norm_window).values.reshape(slope_values.shape), axis=0)
        else:
            slope_health = np.full(len(df.index), 0.5)
        # 维度3: 加速度健康度
        accel_cols = [f'ACCEL_{p}_EMA_{p}_D' for p in ma_periods]
        if all(col in df.columns for col in accel_cols):
            accel_values = np.stack([df[col].values for col in accel_cols], axis=0)
            accel_health = np.mean(normalize_score(pd.Series(accel_values.flatten()), df.index, norm_window).values.reshape(accel_values.shape), axis=0)
        else:
            accel_health = np.full(len(df.index), 0.5)
        # 维度4: 关系健康度
        ma_std = np.std(ma_values / df['close_D'].values[:, np.newaxis].T, axis=0)
        relational_health = 1.0 - normalize_score(pd.Series(ma_std, index=df.index), df.index, norm_window, ascending=True)
        # 维度5: 元动力健康度 (Meta-Dynamics Health) - 跨周期导数
        meta_dynamics_cols = [
            'SLOPE_5_EMA_55_D', 'SLOPE_13_EMA_89_D', 'SLOPE_21_EMA_144_D'
        ]
        valid_meta_cols = [col for col in meta_dynamics_cols if col in df.columns]
        if valid_meta_cols:
            meta_values = np.stack([normalize_score(df[col], df.index, norm_window) for col in valid_meta_cols], axis=0)
            meta_dynamics_health = np.mean(meta_values, axis=0)
        else:
            meta_dynamics_health = np.full(len(df.index), 0.5)
        
        # 将新维度加入最终融合
        scores = np.stack([alignment_health, slope_health, accel_health, relational_health, meta_dynamics_health], axis=0)
        weights_array = np.array(list(weights.values()))
        weights_array /= weights_array.sum()
        final_score_values = np.prod(scores ** weights_array[:, np.newaxis], axis=0)

        return pd.Series(final_score_values, index=df.index, dtype=np.float32)

    def _perform_dynamic_relational_meta_analysis(self, df: pd.DataFrame, snapshot_score: pd.Series, tactical_p: int, context_p: int) -> pd.Series:
        """
        【V3.0 · 双层印证版】动态力学专用的关系元分析核心引擎
        - 核心升级: 接收 tactical_p 和 context_p，在两个时间层级上独立计算速度和加速度，然后融合，实现双层动态印证。
        - 保持不变: 核心的“阿瑞斯之怒”加法模型逻辑保持不变。
        """
        # 引入双层动态印证
        p_conf = get_params_block(self.strategy, 'dynamic_mechanics_params', {})
        p_meta = p_conf.get('relational_meta_analysis_params', {})
        w_state = get_param_value(p_meta.get('state_weight'), 0.3)
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.3)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.4)
        norm_window = 55
        bipolar_sensitivity = 1.0
        # 维度一：状态分 (State Score) - 直接使用快照分
        state_score = snapshot_score.clip(0, 1)
        # 维度二：速度分 (Velocity Score) - 双层印证
        tactical_trend = snapshot_score.diff(tactical_p).fillna(0)
        tactical_velocity = normalize_to_bipolar(tactical_trend, df.index, norm_window, bipolar_sensitivity)
        context_trend = snapshot_score.diff(context_p).fillna(0)
        context_velocity = normalize_to_bipolar(context_trend, df.index, norm_window, bipolar_sensitivity)
        velocity_score = (tactical_velocity * context_velocity)**0.5 * np.sign(tactical_velocity) # 融合后保留方向
        # 维度三：加速度分 (Acceleration Score) - 双层印证
        tactical_accel = tactical_trend.diff(tactical_p).fillna(0)
        tactical_acceleration = normalize_to_bipolar(tactical_accel, df.index, norm_window, bipolar_sensitivity)
        context_accel = context_trend.diff(context_p).fillna(0)
        context_acceleration = normalize_to_bipolar(context_accel, df.index, norm_window, bipolar_sensitivity)
        acceleration_score = (tactical_acceleration * context_acceleration)**0.5 * np.sign(tactical_acceleration) # 融合后保留方向
        # 终极融合：阿瑞斯之怒加法模型
        final_score = (
            state_score * w_state +
            velocity_score * w_velocity +
            acceleration_score * w_acceleration
        ).clip(0, 1)
        
        return final_score.astype(np.float32)
        
















