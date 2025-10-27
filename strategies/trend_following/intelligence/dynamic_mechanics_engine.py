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
        【V16.3 · 算术平均融合版】动态力学终极信号诊断引擎
        - 核心重构: 废除脆弱的“几何平均数”，换用更具韧性的“加权算术平均数”来融合五大支柱。
                      此修改确保了单个支柱的暂时性疲软不会导致整个力学快照分崩溃为零。
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
        vol_bull_snapshot = (normalize_score(df.get('BBW_21_2.0_D'), df.index, norm_window, ascending=False) * normalize_score(df.get('ATR_14_D'), df.index, norm_window, ascending=False))**0.5
        vol_bear_snapshot = (normalize_score(df.get('BBW_21_2.0_D'), df.index, norm_window, ascending=True) * normalize_score(df.get('ATR_14_D'), df.index, norm_window, ascending=True))**0.5
        eff_bull_snapshot = (normalize_score(df.get('VPA_EFFICIENCY_D'), df.index, norm_window, ascending=True) * normalize_score(df.get('intraday_trend_efficiency_D'), df.index, norm_window, ascending=True))**0.5
        eff_bear_snapshot = (normalize_score(df.get('VPA_EFFICIENCY_D'), df.index, norm_window, ascending=False) * normalize_score(df.get('intraday_trend_efficiency_D'), df.index, norm_window, ascending=False))**0.5
        mom_bull_snapshot = (normalize_score(df.get('ROC_12_D'), df.index, norm_window, ascending=True) * normalize_score(df.get('MACDh_13_34_8_D'), df.index, norm_window, ascending=True))**0.5
        mom_bear_snapshot = (normalize_score(df.get('ROC_12_D'), df.index, norm_window, ascending=False) * normalize_score(df.get('MACDh_13_34_8_D'), df.index, norm_window, ascending=False))**0.5
        adx_strength = normalize_score(df.get('ADX_14_D'), df.index, norm_window)
        adx_direction = (df.get('PDI_14_D', 0) > df.get('NDI_14_D', 0)).astype(float)
        hurst_strength = normalize_score(df.get('hurst_120d_D'), df.index, norm_window)
        ine_bull_snapshot = (adx_strength * adx_direction * hurst_strength)**(1/3)
        ine_bear_snapshot = (normalize_score(df.get('ADX_14_D'), df.index, norm_window, ascending=False) * normalize_score(df.get('hurst_120d_D'), df.index, norm_window, ascending=False))**0.5
        energy_bull_snapshot = normalize_score(df.get('energy_ratio_D'), df.index, norm_window, ascending=True)
        energy_bear_snapshot = normalize_score(df.get('energy_ratio_D'), df.index, norm_window, ascending=False)
            # --- 使用加权算术平均数进行融合 ---
        bull_snapshots = {
            'volatility': vol_bull_snapshot, 'efficiency': eff_bull_snapshot, 'momentum': mom_bull_snapshot,
            'inertia': ine_bull_snapshot, 'energy_transition': energy_bull_snapshot
        }
        bear_snapshots = {
            'volatility': vol_bear_snapshot, 'efficiency': eff_bear_snapshot, 'momentum': mom_bear_snapshot,
            'inertia': ine_bear_snapshot, 'energy_transition': energy_bear_snapshot
        }
        fused_bull_snapshot = pd.Series(0.0, index=df.index, dtype=np.float64)
        fused_bear_snapshot = pd.Series(0.0, index=df.index, dtype=np.float64)
        total_weight = sum(pillar_weights.values())
        if total_weight > 0:
            for name, weight in pillar_weights.items():
                fused_bull_snapshot += bull_snapshots.get(name, pd.Series(0.5, index=df.index)).fillna(0.5) * (weight / total_weight)
                fused_bear_snapshot += bear_snapshots.get(name, pd.Series(0.5, index=df.index)).fillna(0.5) * (weight / total_weight)
        else:
            fused_bull_snapshot = pd.Series(0.5, index=df.index)
            fused_bear_snapshot = pd.Series(0.5, index=df.index)
            bipolar_mechanics_snapshot = (fused_bull_snapshot - fused_bear_snapshot).clip(-1, 1)
        modulated_bipolar_snapshot = bipolar_mechanics_snapshot * ma_health_score
        for i, p in enumerate(sorted_periods):
            context_p = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p
            final_bipolar_health = self._perform_dynamic_relational_meta_analysis(df, modulated_bipolar_snapshot, p, context_p)
            overall_health['s_bull'][p] = final_bipolar_health.clip(0, 1).astype(np.float32)
            overall_health['s_bear'][p] = (final_bipolar_health.clip(-1, 0) * -1).astype(np.float32)
            overall_health['d_intensity'][p] = final_bipolar_health.clip(0, 1).astype(np.float32)
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
        【V3.0 · 算术平均版】“赫尔墨斯的商神杖”五维均线健康度评估引擎
        - 核心重构: 废除脆弱的“几何平均数(np.prod)”，换用更具韧性的“加权算术平均数(Σ(score*weight))”。
                      此修改确保了单个维度的暂时性疲软不会导致整个评估系统崩溃，彻底根除“零值传染病”的一个源头。
        """
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
        # --- 五维健康度计算 ---
        alignment_health = np.mean(ma_values[:-1] > ma_values[1:], axis=0) if ma_values.shape[0] > 1 else np.full(len(df.index), 0.5)
        slope_cols = [f'SLOPE_{p}_EMA_{p}_D' for p in ma_periods if f'SLOPE_{p}_EMA_{p}_D' in df.columns]
        slope_health = np.mean([normalize_score(df[col], df.index, norm_window).values for col in slope_cols], axis=0) if slope_cols else np.full(len(df.index), 0.5)
        accel_cols = [f'ACCEL_{p}_EMA_{p}_D' for p in ma_periods if f'ACCEL_{p}_EMA_{p}_D' in df.columns]
        accel_health = np.mean([normalize_score(df[col], df.index, norm_window).values for col in accel_cols], axis=0) if accel_cols else np.full(len(df.index), 0.5)
        ma_std = np.std(ma_values / df['close_D'].values[:, np.newaxis].T, axis=0)
        relational_health = 1.0 - normalize_score(pd.Series(ma_std, index=df.index), df.index, norm_window, ascending=True).values
        meta_dynamics_cols = ['SLOPE_5_EMA_55_D', 'SLOPE_13_EMA_89_D', 'SLOPE_21_EMA_144_D']
        valid_meta_cols = [col for col in meta_dynamics_cols if col in df.columns]
        meta_dynamics_health = np.mean([normalize_score(df[col], df.index, norm_window).values for col in valid_meta_cols], axis=0) if valid_meta_cols else np.full(len(df.index), 0.5)
        # --- 使用加权算术平均数进行融合 ---
        total_score = pd.Series(0.0, index=df.index, dtype=np.float64)
        total_weight = 0.0
        health_components = {
            'alignment': alignment_health, 'slope': slope_health, 'accel': accel_health,
            'relational': relational_health, 'meta_dynamics': meta_dynamics_health
        }
        for name, score in health_components.items():
            weight = weights.get(name, 0)
            if weight > 0:
                total_score += score * weight
                total_weight += weight
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = pd.Series(0.5, index=df.index, dtype=np.float32)
        return final_score.astype(np.float32)
    
    def _perform_dynamic_relational_meta_analysis(self, df: pd.DataFrame, snapshot_score: pd.Series, tactical_p: int, context_p: int) -> pd.Series:
        """
        【V3.1 · 双子座回响版】动态力学专用的关系元分析核心引擎
        - 核心革命: 签署“双子座的回响”协议，从筹码引擎引入成熟的双极性评估逻辑。
                      1. [一体两面] 分别计算看涨力量(Bullish Force)和看跌力量(Bearish Force)。
                      2. [净值裁决] 最终得分 = 看涨力量 - 看跌力量，输出一个[-1, 1]的双极性净值分数。
        - 升级意义: 修复了引擎“单极性失明”的致命BUG，使其能正确评估负面动态。
        """
        p_conf = get_params_block(self.strategy, 'dynamic_mechanics_params', {})
        p_meta = p_conf.get('relational_meta_analysis_params', {})
        w_state = get_param_value(p_meta.get('state_weight'), 0.3)
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.3)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.4)
        norm_window = 55
        bipolar_sensitivity = 1.0
        # 实施“双子座的回响”协议
        # 维度二：速度分 (Velocity Score) - 双层印证
        tactical_trend = snapshot_score.diff(tactical_p).fillna(0)
        tactical_velocity = normalize_to_bipolar(tactical_trend, df.index, norm_window, bipolar_sensitivity)
        context_trend = snapshot_score.diff(context_p).fillna(0)
        context_velocity = normalize_to_bipolar(context_trend, df.index, norm_window, bipolar_sensitivity)
        velocity_score = (tactical_velocity.abs() * context_velocity.abs())**0.5 * np.sign(tactical_velocity) # 融合后保留方向
        # 维度三：加速度分 (Acceleration Score) - 双层印证
        tactical_accel = tactical_trend.diff(tactical_p).fillna(0)
        tactical_acceleration = normalize_to_bipolar(tactical_accel, df.index, norm_window, bipolar_sensitivity)
        context_accel = context_trend.diff(context_p).fillna(0)
        context_acceleration = normalize_to_bipolar(context_accel, df.index, norm_window, bipolar_sensitivity)
        acceleration_score = (tactical_acceleration.abs() * context_acceleration.abs())**0.5 * np.sign(tactical_acceleration) # 融合后保留方向
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
        
















