# 文件: strategies/trend_following/intelligence/dynamic_mechanics_engine.py
# 动态力学引擎
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following.utils import transmute_health_to_ultimate_signals, get_params_block, get_param_value, normalize_score, calculate_holographic_dynamics, normalize_to_bipolar

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
        【V12.0 · 圣杯契约版】
        - 核心革命: 不再读取本地的、重复的合成参数，而是从最高指挥部获取唯一的“圣杯”配置
                      (`ultimate_signal_synthesis_params`)，并将其传递给中央合成引擎。
        """
        states = {}
        p_conf = get_params_block(self.strategy, 'dynamic_mechanics_params', {})
        if not get_param_value(p_conf.get('enabled'), True): return states
        # 获取中央“圣杯”配置
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        dynamic_weights = {'slope': 0.6, 'accel': 0.4}
        pillar_weights = get_param_value(p_conf.get('pillar_weights'), {})
        periods = get_param_value(p_synthesis.get('periods'), [1, 5, 13, 21, 55])
        norm_window = get_param_value(p_synthesis.get('norm_window'), 55)
        health_data = { 's_bull': [], 's_bear': [], 'd_intensity': [] } 
        calculators = {
            'volatility': self._calculate_volatility_health,
            'efficiency': self._calculate_efficiency_health,
            'momentum': self._calculate_kinetic_energy_health,
            'inertia': self._calculate_inertia_health,
        }
        for name, calculator in calculators.items():
            s_bull, s_bear, d_intensity = calculator(df, norm_window, dynamic_weights, periods)
            health_data['s_bull'].append(s_bull) 
            health_data['s_bear'].append(s_bear) 
            health_data['d_intensity'].append(d_intensity)
        overall_health = {}
        weights_array = np.array([pillar_weights.get(name, 0.25) for name in calculators.keys()])
        for health_type, health_sources in [
            ('s_bull', health_data['s_bull']),
            ('s_bear', health_data['s_bear']),
            ('d_intensity', health_data['d_intensity'])
        ]:
            overall_health[health_type] = {}
            for p in periods:
                if not health_sources: continue
                valid_pillars = [pillar_dict[p].values for pillar_dict in health_sources if p in pillar_dict]
                if not valid_pillars: continue
                stacked_values = np.stack(valid_pillars, axis=0)
                fused_values = np.prod(stacked_values ** weights_array[:, np.newaxis], axis=0)
                overall_health[health_type][p] = pd.Series(fused_values, index=df.index, dtype=np.float32)
        self.strategy.atomic_states['__DYN_overall_health'] = overall_health
        # 传入唯一的“圣杯”配置
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

    def _calculate_volatility_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict, Dict, Dict]:
        """【V3.0 · 关系元分析版】计算波动率(BBW)维度的三维健康度"""
        s_bull, s_bear, d_intensity = {}, {}, {}
        
        # 增加数据检查和韧性返回
        required_cols = ['BBW_21_2.0_D']
        if any(col not in df.columns for col in required_cols):
            default_series = pd.Series(0.5, index=df.index, dtype=np.float32)
            for p in periods:
                s_bull[p] = default_series
                s_bear[p] = default_series
                d_intensity[p] = default_series
            return s_bull, s_bear, d_intensity

        # 步骤一：计算原始的、纯粹的力学静态健康度
        # 健康的上涨需要收敛的波动率（低BBW）
        mechanic_static_bull = normalize_score(df.get('BBW_21_2.0_D'), df.index, norm_window, ascending=False)
        mechanic_static_bear = normalize_score(df.get('BBW_21_2.0_D'), df.index, norm_window, ascending=True)

        # 步骤二：获取均线趋势上下文分数
        ma_context_score = self._calculate_ma_trend_context(df, [5, 13, 21, 55])

        # 步骤三：构建融合了趋势上下文的“瞬时关系快照分”
        bullish_snapshot_score = (mechanic_static_bull * ma_context_score)
        bearish_snapshot_score = (mechanic_static_bear * (1 - ma_context_score))

        # 步骤四：对快照分进行关系元分析，得到最终的动态强度分
        unified_d_intensity = self._perform_dynamic_relational_meta_analysis(df, bullish_snapshot_score)

        # 步骤五：更新输出
        for p in periods:
            s_bull[p] = bullish_snapshot_score
            s_bear[p] = bearish_snapshot_score
            d_intensity[p] = unified_d_intensity

        return s_bull, s_bear, d_intensity

    def _calculate_efficiency_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict, Dict, Dict]:
        """【V3.0 · 关系元分析版】计算效率(VPA)维度的三维健康度"""
        s_bull, s_bear, d_intensity = {}, {}, {}
        
        # 增加数据检查和韧性返回
        required_cols = ['VPA_EFFICIENCY_D']
        if any(col not in df.columns for col in required_cols):
            default_series = pd.Series(0.5, index=df.index, dtype=np.float32)
            for p in periods:
                s_bull[p] = default_series
                s_bear[p] = default_series
                d_intensity[p] = default_series
            return s_bull, s_bear, d_intensity

        # 步骤一：计算原始的、纯粹的力学静态健康度
        # 健康的上涨需要高效率
        mechanic_static_bull = normalize_score(df.get('VPA_EFFICIENCY_D'), df.index, norm_window)
        mechanic_static_bear = normalize_score(df.get('VPA_EFFICIENCY_D'), df.index, norm_window, ascending=False)

        # 步骤二：获取均线趋势上下文分数
        ma_context_score = self._calculate_ma_trend_context(df, [5, 13, 21, 55])

        # 步骤三：构建融合了趋势上下文的“瞬时关系快照分”
        bullish_snapshot_score = (mechanic_static_bull * ma_context_score)
        bearish_snapshot_score = (mechanic_static_bear * (1 - ma_context_score))

        # 步骤四：对快照分进行关系元分析，得到最终的动态强度分
        unified_d_intensity = self._perform_dynamic_relational_meta_analysis(df, bullish_snapshot_score)

        # 步骤五：更新输出
        for p in periods:
            s_bull[p] = bullish_snapshot_score
            s_bear[p] = bearish_snapshot_score
            d_intensity[p] = unified_d_intensity
            
        return s_bull, s_bear, d_intensity

    def _calculate_kinetic_energy_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict, Dict, Dict]:
        """【V3.0 · 关系元分析版】计算动能(ATR)维度的三维健康度"""
        s_bull, s_bear, d_intensity = {}, {}, {}
        
        # 增加数据检查和韧性返回
        required_cols = ['ATR_14_D']
        if any(col not in df.columns for col in required_cols):
            default_series = pd.Series(0.5, index=df.index, dtype=np.float32)
            for p in periods:
                s_bull[p] = default_series
                s_bear[p] = default_series
                d_intensity[p] = default_series
            return s_bull, s_bear, d_intensity

        # 步骤一：计算原始的、纯粹的力学静态健康度
        # 健康的上涨需要充足的动能（高ATR）
        mechanic_static_bull = normalize_score(df.get('ATR_14_D'), df.index, norm_window)
        mechanic_static_bear = normalize_score(df.get('ATR_14_D'), df.index, norm_window, ascending=False)

        # 步骤二：获取均线趋势上下文分数
        ma_context_score = self._calculate_ma_trend_context(df, [5, 13, 21, 55])

        # 步骤三：构建融合了趋势上下文的“瞬时关系快照分”
        bullish_snapshot_score = (mechanic_static_bull * ma_context_score)
        bearish_snapshot_score = (mechanic_static_bear * (1 - ma_context_score))

        # 步骤四：对快照分进行关系元分析，得到最终的动态强度分
        unified_d_intensity = self._perform_dynamic_relational_meta_analysis(df, bullish_snapshot_score)

        # 步骤五：更新输出
        for p in periods:
            s_bull[p] = bullish_snapshot_score
            s_bear[p] = bearish_snapshot_score
            d_intensity[p] = unified_d_intensity

        return s_bull, s_bear, d_intensity

    def _calculate_inertia_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict, Dict, Dict]:
        """【V3.0 · 关系元分析版】计算惯性(ADX)维度的三维健康度"""
        s_bull, s_bear, d_intensity = {}, {}, {}
        
        # 增加数据检查和韧性返回
        required_cols = ['ADX_14_D', 'PDI_14_D', 'NDI_14_D']
        if any(col not in df.columns for col in required_cols):
            default_series = pd.Series(0.5, index=df.index, dtype=np.float32)
            for p in periods:
                s_bull[p] = default_series
                s_bear[p] = default_series
                d_intensity[p] = default_series
            return s_bull, s_bear, d_intensity

        # 步骤一：计算原始的、纯粹的力学静态健康度
        # 健康的上涨需要强大的惯性（高ADX）且方向向上（PDI > NDI）
        adx_strength = normalize_score(df.get('ADX_14_D'), df.index, norm_window)
        adx_direction = (df.get('PDI_14_D') > df.get('NDI_14_D')).astype(float)
        mechanic_static_bull = (adx_strength * adx_direction)
        mechanic_static_bear = normalize_score(df.get('ADX_14_D'), df.index, norm_window, ascending=False)

        # 步骤二：获取均线趋势上下文分数
        ma_context_score = self._calculate_ma_trend_context(df, [5, 13, 21, 55])

        # 步骤三：构建融合了趋势上下文的“瞬时关系快照分”
        bullish_snapshot_score = (mechanic_static_bull * ma_context_score)
        bearish_snapshot_score = (mechanic_static_bear * (1 - ma_context_score))

        # 步骤四：对快照分进行关系元分析，得到最终的动态强度分
        unified_d_intensity = self._perform_dynamic_relational_meta_analysis(df, bullish_snapshot_score)

        # 步骤五：更新输出
        for p in periods:
            s_bull[p] = bullish_snapshot_score
            s_bear[p] = bearish_snapshot_score
            d_intensity[p] = unified_d_intensity
            
        return s_bull, s_bear, d_intensity

    def _perform_dynamic_relational_meta_analysis(self, df: pd.DataFrame, snapshot_score: pd.Series) -> pd.Series:
        """
        【V1.0 · 新增】动态力学专用的关系元分析核心引擎 (赫拉织布机V2)
        - 核心逻辑: 实现“状态 * (1 + 动态杠杆)”的动态价值调制范式。
        """
        # 从配置中获取动态杠杆权重
        p_conf = get_params_block(self.strategy, 'dynamic_mechanics_params', {})
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















