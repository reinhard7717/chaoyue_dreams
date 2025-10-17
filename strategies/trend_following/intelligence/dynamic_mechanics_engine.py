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
        【V14.0 · 分层印证版】动态力学终极信号诊断引擎
        - 核心升级: 全面采纳“分层动态印证”框架。在主循环中为每个周期 p，对四大力学支柱进行“战术周期 p vs 上下文周期 context_p”的动态共振计算。
        - 架构重构: 废除旧的 _calculate_*_health 辅助方法，将所有逻辑内聚到本方法中，使诊断流程更清晰、更强大。
        """
        states = {}
        p_conf = get_params_block(self.strategy, 'dynamic_mechanics_params', {})
        if not get_param_value(p_conf.get('enabled'), True): return states
        # 全面重构为分层印证框架
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        pillar_weights = get_param_value(p_conf.get('pillar_weights'), {})
        periods = get_param_value(p_synthesis.get('periods'), [1, 5, 13, 21, 55])
        sorted_periods = sorted(periods)
        norm_window = get_param_value(p_synthesis.get('norm_window'), 55)
        ma_health_score = self._calculate_ma_health(df, p_conf, norm_window)
        overall_health = {'s_bull': {}, 's_bear': {}, 'd_intensity': {}}
        # 主循环：为每个周期 p 计算其最终的、经过分层印证的健康度
        for i, p in enumerate(sorted_periods):
            context_p = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p
            # --- 为当前周期 p，计算四大力学支柱的原始快照分 ---
            # 1. 波动率快照 (看涨=收缩, 看跌=扩张)
            vol_bull_snapshot = normalize_score(df.get('BBW_21_2.0_D'), df.index, norm_window, ascending=False)
            vol_bear_snapshot = normalize_score(df.get('BBW_21_2.0_D'), df.index, norm_window, ascending=True)
            # 2. 效率快照 (看涨=高效, 看跌=低效)
            eff_bull_snapshot = normalize_score(df.get('VPA_EFFICIENCY_D'), df.index, norm_window)
            eff_bear_snapshot = normalize_score(df.get('VPA_EFFICIENCY_D'), df.index, norm_window, ascending=False)
            # 3. 动能快照 (看涨=动能强, 看跌=动能弱)
            mom_bull_snapshot = normalize_score(df.get('ATR_14_D'), df.index, norm_window)
            mom_bear_snapshot = normalize_score(df.get('ATR_14_D'), df.index, norm_window, ascending=False)
            # 4. 惯性快照 (看涨=ADX强且PDI>NDI, 看跌=ADX弱)
            adx_strength = normalize_score(df.get('ADX_14_D'), df.index, norm_window)
            adx_direction = (df.get('PDI_14_D', 0) > df.get('NDI_14_D', 0)).astype(float)
            ine_bull_snapshot = (adx_strength * adx_direction)
            ine_bear_snapshot = normalize_score(df.get('ADX_14_D'), df.index, norm_window, ascending=False)
            # --- 融合四大支柱的快照分 ---
            weight_keys = list(pillar_weights.keys())
            weights_array = np.array([pillar_weights.get(name, 0.25) for name in weight_keys])
            weights_array /= weights_array.sum()
            bull_snapshots = [vol_bull_snapshot, eff_bull_snapshot, mom_bull_snapshot, ine_bull_snapshot]
            bear_snapshots = [vol_bear_snapshot, eff_bear_snapshot, mom_bear_snapshot, ine_bear_snapshot]
            stacked_bull = np.stack([s.values for s in bull_snapshots], axis=0)
            stacked_bear = np.stack([s.values for s in bear_snapshots], axis=0)
            fused_bull_snapshot = pd.Series(np.prod(stacked_bull ** weights_array[:, np.newaxis], axis=0), index=df.index)
            fused_bear_snapshot = pd.Series(np.prod(stacked_bear ** weights_array[:, np.newaxis], axis=0), index=df.index)
            # --- 对融合后的快照分进行双层动态印证 ---
            # 看涨健康度 s_bull
            final_bull_health = self._perform_dynamic_relational_meta_analysis(df, fused_bull_snapshot, p, context_p)
            overall_health['s_bull'][p] = (final_bull_health * ma_health_score).astype(np.float32)
            # 看跌健康度 s_bear
            final_bear_health = self._perform_dynamic_relational_meta_analysis(df, fused_bear_snapshot, p, context_p)
            overall_health['s_bear'][p] = (final_bear_health * (1 - ma_health_score)).astype(np.float32)
            # 动态强度 d_intensity (使用看涨健康度的最终结果作为强度的代理)
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
        【V1.0 · 新增】“赫尔墨斯的商神杖”四维均线健康度评估引擎
        - 核心职责: 严格按照 ma_health_fusion_weights 配置，计算并融合均线健康度的四大维度。
        - 四大维度:
          1. 排列 (Alignment): 均线是否呈多头排列。
          2. 斜率 (Slope): 均线趋势的强度。
          3. 加速度 (Acceleration): 均线趋势的加速能力。
          4. 关系 (Relational): 均线族的收敛/发散状态。
        """
        p_ma_health = get_param_value(params.get('ma_health_fusion_weights'), {})
        weights = {
            'alignment': get_param_value(p_ma_health.get('alignment'), 0.15),
            'slope': get_param_value(p_ma_health.get('slope'), 0.15),
            'accel': get_param_value(p_ma_health.get('accel'), 0.2),
            'relational': get_param_value(p_ma_health.get('relational'), 0.5)
        }
        
        ma_periods = [5, 13, 21, 55]
        ma_cols = [f'EMA_{p}_D' for p in ma_periods]
        if not all(col in df.columns for col in ma_cols):
            return pd.Series(0.5, index=df.index, dtype=np.float32)

        ma_values = np.stack([df[col].values for col in ma_cols], axis=0)
        
        # 维度1: 排列健康度 (Alignment Health)
        alignment_bools = ma_values[:-1] > ma_values[1:]
        alignment_health = np.mean(alignment_bools, axis=0) if alignment_bools.size > 0 else np.full(len(df.index), 0.5)

        # 维度2: 斜率健康度 (Slope Health)
        slope_cols = [f'SLOPE_5_{col}' for col in ma_cols]
        if all(col in df.columns for col in slope_cols):
            slope_values = np.stack([df[col].values for col in slope_cols], axis=0)
            slope_health = np.mean(normalize_score(pd.Series(slope_values.flatten()), df.index, norm_window).values.reshape(slope_values.shape), axis=0)
        else:
            slope_health = np.full(len(df.index), 0.5)

        # 维度3: 加速度健康度 (Acceleration Health)
        accel_cols = [f'ACCEL_5_{col}' for col in ma_cols]
        if all(col in df.columns for col in accel_cols):
            accel_values = np.stack([df[col].values for col in accel_cols], axis=0)
            accel_health = np.mean(normalize_score(pd.Series(accel_values.flatten()), df.index, norm_window).values.reshape(accel_values.shape), axis=0)
        else:
            accel_health = np.full(len(df.index), 0.5)

        # 维度4: 关系健康度 (Relational Health) - 衡量均线收敛性
        # 使用均线值的标准差作为发散度的代理指标，标准差越小，收敛性越好
        ma_std = np.std(ma_values / df['close_D'].values[:, np.newaxis].T, axis=0)
        relational_health = 1.0 - normalize_score(pd.Series(ma_std, index=df.index), df.index, norm_window, ascending=True)

        # 最终融合：加权几何平均
        scores = np.stack([alignment_health, slope_health, accel_health, relational_health], axis=0)
        weights_array = np.array(list(weights.values()))
        weights_array /= weights_array.sum() # 归一化

        final_score_values = np.prod(scores ** weights_array[:, np.newaxis], axis=0)
        
        return pd.Series(final_score_values, index=df.index, dtype=np.float32)

    def _calculate_volatility_health(self, df: pd.DataFrame, norm_window: int, periods: list, ma_context_score: pd.Series) -> Tuple[Dict, Dict, Dict]:
        """
        【V4.0 · 德尔斐神谕协议版】计算波动率(BBW)维度的三维健康度
        - 核心修正: 签署“德尔斐神谕协议”，剥离 ma_context_score 对 s_bull/s_bear 的污染。
        """
        s_bull, s_bear, d_intensity = {}, {}, {}
        if 'BBW_21_2.0_D' not in df.columns:
            default_series = pd.Series(0.5, index=df.index, dtype=np.float32)
            for p in periods:
                s_bull[p] = default_series.copy()
                s_bear[p] = default_series.copy()
                d_intensity[p] = default_series.copy()
            return s_bull, s_bear, d_intensity
        bbw_series = df['BBW_21_2.0_D']
        mechanic_static_bull = normalize_score(bbw_series, df.index, norm_window, ascending=False)
        mechanic_static_bear = normalize_score(bbw_series, df.index, norm_window, ascending=True)
        # bullish_snapshot_score 和 bearish_snapshot_score 现在是纯粹的静态分
        bullish_snapshot_score = mechanic_static_bull.astype(np.float32)
        bearish_snapshot_score = mechanic_static_bear.astype(np.float32)
        unified_d_intensity = self._perform_dynamic_relational_meta_analysis(df, bullish_snapshot_score)
        for p in periods:
            s_bull[p] = bullish_snapshot_score
            s_bear[p] = bearish_snapshot_score
            d_intensity[p] = unified_d_intensity
        return s_bull, s_bear, d_intensity

    def _calculate_efficiency_health(self, df: pd.DataFrame, norm_window: int, periods: list, ma_context_score: pd.Series) -> Tuple[Dict, Dict, Dict]:
        """
        【V4.0 · 德尔斐神谕协议版】计算效率(VPA)维度的三维健康度
        - 核心修正: 签署“德尔斐神谕协议”，剥离 ma_context_score 对 s_bull/s_bear 的污染。
        """
        s_bull, s_bear, d_intensity = {}, {}, {}
        if 'VPA_EFFICIENCY_D' not in df.columns:
            default_series = pd.Series(0.5, index=df.index, dtype=np.float32)
            for p in periods:
                s_bull[p] = default_series.copy()
                s_bear[p] = default_series.copy()
                d_intensity[p] = default_series.copy()
            return s_bull, s_bear, d_intensity
        vpa_series = df['VPA_EFFICIENCY_D']
        mechanic_static_bull = normalize_score(vpa_series, df.index, norm_window)
        mechanic_static_bear = normalize_score(vpa_series, df.index, norm_window, ascending=False)
        # bullish_snapshot_score 和 bearish_snapshot_score 现在是纯粹的静态分
        bullish_snapshot_score = mechanic_static_bull.astype(np.float32)
        bearish_snapshot_score = mechanic_static_bear.astype(np.float32)
        unified_d_intensity = self._perform_dynamic_relational_meta_analysis(df, bullish_snapshot_score)
        for p in periods:
            s_bull[p] = bullish_snapshot_score
            s_bear[p] = bearish_snapshot_score
            d_intensity[p] = unified_d_intensity
        return s_bull, s_bear, d_intensity

    def _calculate_kinetic_energy_health(self, df: pd.DataFrame, norm_window: int, periods: list, ma_context_score: pd.Series) -> Tuple[Dict, Dict, Dict]:
        """
        【V4.0 · 德尔斐神谕协议版】计算动能(ATR)维度的三维健康度
        - 核心修正: 签署“德尔斐神谕协议”，剥离 ma_context_score 对 s_bull/s_bear 的污染。
        """
        s_bull, s_bear, d_intensity = {}, {}, {}
        if 'ATR_14_D' not in df.columns:
            default_series = pd.Series(0.5, index=df.index, dtype=np.float32)
            for p in periods:
                s_bull[p] = default_series.copy()
                s_bear[p] = default_series.copy()
                d_intensity[p] = default_series.copy()
            return s_bull, s_bear, d_intensity
        atr_series = df['ATR_14_D']
        mechanic_static_bull = normalize_score(atr_series, df.index, norm_window)
        mechanic_static_bear = normalize_score(atr_series, df.index, norm_window, ascending=False)
        # bullish_snapshot_score 和 bearish_snapshot_score 现在是纯粹的静态分
        bullish_snapshot_score = mechanic_static_bull.astype(np.float32)
        bearish_snapshot_score = mechanic_static_bear.astype(np.float32)
        unified_d_intensity = self._perform_dynamic_relational_meta_analysis(df, bullish_snapshot_score)
        for p in periods:
            s_bull[p] = bullish_snapshot_score
            s_bear[p] = bearish_snapshot_score
            d_intensity[p] = unified_d_intensity
        return s_bull, s_bear, d_intensity

    def _calculate_inertia_health(self, df: pd.DataFrame, norm_window: int, periods: list, ma_context_score: pd.Series) -> Tuple[Dict, Dict, Dict]:
        """
        【V4.0 · 德尔斐神谕协议版】计算惯性(ADX)维度的三维健康度
        - 核心修正: 签署“德尔斐神谕协议”，剥离 ma_context_score 对 s_bull/s_bear 的污染。
        """
        s_bull, s_bear, d_intensity = {}, {}, {}
        required_cols = ['ADX_14_D', 'PDI_14_D', 'NDI_14_D']
        if any(col not in df.columns for col in required_cols):
            default_series = pd.Series(0.5, index=df.index, dtype=np.float32)
            for p in periods:
                s_bull[p] = default_series.copy()
                s_bear[p] = default_series.copy()
                d_intensity[p] = default_series.copy()
            return s_bull, s_bear, d_intensity
        adx_strength = normalize_score(df['ADX_14_D'], df.index, norm_window)
        adx_direction = (df['PDI_14_D'] > df['NDI_14_D']).astype(float)
        mechanic_static_bull = (adx_strength * adx_direction)
        mechanic_static_bear = normalize_score(df['ADX_14_D'], df.index, norm_window, ascending=False)
        # bullish_snapshot_score 和 bearish_snapshot_score 现在是纯粹的静态分
        bullish_snapshot_score = mechanic_static_bull.astype(np.float32)
        bearish_snapshot_score = mechanic_static_bear.astype(np.float32)
        unified_d_intensity = self._perform_dynamic_relational_meta_analysis(df, bullish_snapshot_score)
        for p in periods:
            s_bull[p] = bullish_snapshot_score
            s_bear[p] = bearish_snapshot_score
            d_intensity[p] = unified_d_intensity
        return s_bull, s_bear, d_intensity

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
        
















