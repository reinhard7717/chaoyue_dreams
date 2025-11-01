# 文件: strategies/trend_following/intelligence/dynamic_mechanics_engine.py
# 动态力学引擎
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score, normalize_to_bipolar

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
        【V17.0 · 四象限重构版】动态力学终极信号诊断引擎
        - 核心重构: 废弃对通用函数 transmute_health_to_ultimate_signals 的调用，引入“四象限动态分析法”，
                      彻底解决信号命名与逻辑混乱的问题，确保与所有情报模块的哲学统一。
        """
        states = {}
        p_conf = get_params_block(self.strategy, 'dynamic_mechanics_params', {})
        if not get_param_value(p_conf.get('enabled'), True): return states
        

        # 步骤一：计算各支柱的静态快照分
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        pillar_weights = get_param_value(p_conf.get('pillar_weights'), {})
        norm_window = get_param_value(p_synthesis.get('norm_window'), 55)
        
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
        
        bull_snapshots = {'volatility': vol_bull_snapshot, 'efficiency': eff_bull_snapshot, 'momentum': mom_bull_snapshot, 'inertia': ine_bull_snapshot, 'energy_transition': energy_bull_snapshot}
        bear_snapshots = {'volatility': vol_bear_snapshot, 'efficiency': eff_bear_snapshot, 'momentum': mom_bear_snapshot, 'inertia': ine_bear_snapshot, 'energy_transition': energy_bear_snapshot}
        
        # 步骤二：融合得到静态的共振信号 (零阶动态)
        fused_bull_snapshot = pd.Series(0.0, index=df.index, dtype=np.float64)
        fused_bear_snapshot = pd.Series(0.0, index=df.index, dtype=np.float64)
        total_weight = sum(pillar_weights.values())
        if total_weight > 0:
            for name, weight in pillar_weights.items():
                fused_bull_snapshot += bull_snapshots.get(name, pd.Series(0.5, index=df.index)).fillna(0.5) * (weight / total_weight)
                fused_bear_snapshot += bear_snapshots.get(name, pd.Series(0.5, index=df.index)).fillna(0.5) * (weight / total_weight)
        
        ma_health_score = self._calculate_ma_health(df, p_conf, norm_window)
        bullish_resonance = (fused_bull_snapshot * ma_health_score).clip(0, 1)
        bearish_resonance = (fused_bear_snapshot * (1 - ma_health_score)).clip(0, 1)
        
        states['SCORE_DYN_BULLISH_RESONANCE'] = bullish_resonance.astype(np.float32)
        states['SCORE_DYN_BEARISH_RESONANCE'] = bearish_resonance.astype(np.float32)
        
        # 步骤三：计算四象限动态信号 (一阶和二阶动态)
        bull_divergence = self._calculate_holographic_divergence_dyn(bullish_resonance, 5, 21, norm_window)
        bullish_acceleration = bull_divergence.clip(0, 1)
        top_reversal = (bull_divergence.clip(-1, 0) * -1)
        
        bear_divergence = self._calculate_holographic_divergence_dyn(bearish_resonance, 5, 21, norm_window)
        bearish_acceleration = bear_divergence.clip(0, 1)
        bottom_reversal = (bear_divergence.clip(-1, 0) * -1)
        
        # 步骤四：赋值给命名准确的终极信号
        states['SCORE_DYN_BULLISH_ACCELERATION'] = bullish_acceleration.astype(np.float32)
        states['SCORE_DYN_TOP_REVERSAL'] = top_reversal.astype(np.float32)
        states['SCORE_DYN_BEARISH_ACCELERATION'] = bearish_acceleration.astype(np.float32)
        states['SCORE_DYN_BOTTOM_REVERSAL'] = bottom_reversal.astype(np.float32)
        
        # 步骤五：重铸战术反转信号
        states['SCORE_DYN_TACTICAL_REVERSAL'] = (bullish_resonance * top_reversal).clip(0, 1).astype(np.float32)
        
        
        return states

    def _calculate_holographic_divergence_dyn(self, series: pd.Series, short_p: int, long_p: int, norm_window: int) -> pd.Series:
        """
        【V1.0 · 新增】力学层专用的全息背离计算引擎
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
        【V4.0 · 状态主导协议版】动态力学专用的关系元分析核心引擎
        - 核心修复: 植入“状态主导协议”，并调整默认权重为状态主导，解决“动态压制”问题。
        """
        
        p_conf = get_params_block(self.strategy, 'dynamic_mechanics_params', {})
        p_meta = p_conf.get('relational_meta_analysis_params', {})
        # 权重调整为状态主导
        w_state = get_param_value(p_meta.get('state_weight'), 0.6)
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.2)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.2)
        norm_window = 55
        bipolar_sensitivity = 1.0
        tactical_trend = snapshot_score.diff(tactical_p).fillna(0)
        tactical_velocity = normalize_to_bipolar(tactical_trend, df.index, norm_window, bipolar_sensitivity)
        context_trend = snapshot_score.diff(context_p).fillna(0)
        context_velocity = normalize_to_bipolar(context_trend, df.index, norm_window, bipolar_sensitivity)
        velocity_score = (tactical_velocity.abs() * context_velocity.abs())**0.5 * np.sign(tactical_velocity)
        tactical_accel = tactical_trend.diff(tactical_p).fillna(0)
        tactical_acceleration = normalize_to_bipolar(tactical_accel, df.index, norm_window, bipolar_sensitivity)
        context_accel = context_trend.diff(context_p).fillna(0)
        context_acceleration = normalize_to_bipolar(context_accel, df.index, norm_window, bipolar_sensitivity)
        acceleration_score = (tactical_acceleration.abs() * context_acceleration.abs())**0.5 * np.sign(tactical_acceleration)
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
        

















