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
        - 优化说明: 1. 将均线趋势上下文(`ma_context_score`)的计算提前，避免在循环中重复计算4次，显著提升效率。
                      2. 使用Numpy向量化计算加权几何平均数，高效融合各维度健康度。
        """
        states = {}
        p_conf = get_params_block(self.strategy, 'dynamic_mechanics_params', {})
        if not get_param_value(p_conf.get('enabled'), True): return states
        
        # --- 1. 获取核心配置参数 ---
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        pillar_weights = get_param_value(p_conf.get('pillar_weights'), {})
        periods = get_param_value(p_synthesis.get('periods'), [1, 5, 13, 21, 55])
        norm_window = get_param_value(p_synthesis.get('norm_window'), 55)

        # 提前计算均线趋势上下文分数，避免在各健康度计算器中重复执行。
        # 这是本方法最核心的性能优化点。
        ma_context_score = self._calculate_ma_trend_context(df, [5, 13, 21, 55])

        # --- 2. 计算各维度的三维健康度 ---
        health_data = { 's_bull': [], 's_bear': [], 'd_intensity': [] } 
        calculators = {
            'volatility': self._calculate_volatility_health,
            'efficiency': self._calculate_efficiency_health,
            'momentum': self._calculate_kinetic_energy_health,
            'inertia': self._calculate_inertia_health,
        }
        
        # 循环调用各维度计算器，并传入预先计算好的 ma_context_score
        for name, calculator in calculators.items():
            # 注意：原 `dynamic_weights` 参数在子函数中并未使用，因此从调用中移除
            s_bull, s_bear, d_intensity = calculator(df, norm_window, periods, ma_context_score)
            health_data['s_bull'].append(s_bull) 
            health_data['s_bear'].append(s_bear) 
            health_data['d_intensity'].append(d_intensity)

        # --- 3. 融合各维度健康度，得到整体健康度 ---
        overall_health = {}
        weight_keys = list(calculators.keys())
        weights_array = np.array([pillar_weights.get(name, 0.25) for name in weight_keys])
        weights_array /= weights_array.sum() # 权重归一化，确保总权重为1

        for health_type, health_sources in health_data.items():
            overall_health[health_type] = {}
            for p in periods:
                if not health_sources: continue
                valid_pillars = [pillar_dict[p].values for pillar_dict in health_sources if p in pillar_dict]
                if not valid_pillars: continue
                
                stacked_values = np.stack(valid_pillars, axis=0)
                # 使用加权几何平均数进行融合。该算法能有效惩罚任何一个维度的短板，更能体现“健康”的综合性。
                # 公式: G = (s1^w1 * s2^w2 * ... * sn^wn)
                fused_values = np.prod(stacked_values ** weights_array[:, np.newaxis], axis=0)
                overall_health[health_type][p] = pd.Series(fused_values, index=df.index, dtype=np.float32)
        
        self.strategy.atomic_states['__DYN_overall_health'] = overall_health

        # --- 4. 调用中央合成引擎，生成最终信号 ---
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

    def _perform_dynamic_relational_meta_analysis(self, df: pd.DataFrame, snapshot_score: pd.Series) -> pd.Series:
        """
        【V1.0 · 新增】动态力学专用的关系元分析核心引擎 (赫拉织布机V2)
        - 核心逻辑: 实现“状态 * (1 + 动态杠杆)”的动态价值调制范式。
                      即最终分数不仅取决于当前状态好不好，还取决于它变好的速度和加速度。
        - 优化说明: 全程使用Pandas/Numpy向量化操作，计算效率高。
        """
        # --- 1. 获取参数 ---
        p_conf = get_params_block(self.strategy, 'dynamic_mechanics_params', {})
        p_meta = p_conf.get('relational_meta_analysis_params', {})
        w_velocity = p_meta.get('velocity_weight', 0.6)
        w_acceleration = p_meta.get('acceleration_weight', 0.4)
        norm_window = 55
        meta_window = 5
        bipolar_sensitivity = 1.0

        # --- 2. 计算三维动态要素 ---
        # 第一维度：状态分 (State Score) - 当前快照分值
        state_score = snapshot_score.clip(0, 1)

        # 第二维度：速度分 (Velocity Score) - 快照分的变化趋势（一阶导数）
        # 使用 .diff() 高效计算变化量
        relationship_trend = snapshot_score.diff(meta_window).fillna(0)
        velocity_score = normalize_to_bipolar(
            series=relationship_trend, target_index=df.index,
            window=norm_window, sensitivity=bipolar_sensitivity
        )

        # 第三维度：加速度分 (Acceleration Score) - 变化趋势的趋势（二阶导数）
        relationship_accel = relationship_trend.diff(meta_window).fillna(0)
        acceleration_score = normalize_to_bipolar(
            series=relationship_accel, target_index=df.index,
            window=norm_window, sensitivity=bipolar_sensitivity
        )

        # --- 3. 终极融合：动态价值调制 ---
        # 动态杠杆 = 1 + (加权速度) + (加权加速度)。
        # 当速度和加速度为正时，杠杆>1，放大当前状态分；为负时，杠杆<1，削弱当前状态分。
        dynamic_leverage = 1 + (velocity_score * w_velocity) + (acceleration_score * w_acceleration)
        final_score = (state_score * dynamic_leverage).clip(0, 1)
        
        return final_score.astype(np.float32)

    def _calculate_ma_trend_context(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V1.0 · 新增】计算均线趋势上下文分数
        - 核心逻辑: 综合评估均线排列和价格位置，输出一个统一的[0, 1]区间的趋势健康分。
                      分数越高，代表均线多头排列越整齐，且价格处于强势位置。
        - 优化说明: 使用Numpy进行向量化计算，避免循环，效率高。
        """
        # 确保所有需要的均线列都存在于DataFrame中
        ma_cols = [f'EMA_{p}_D' for p in periods]
        if not all(col in df.columns for col in ma_cols):
            # 如果缺少任何一根均线，则无法判断趋势，返回中性值0.5
            return pd.Series(0.5, index=df.index, dtype=np.float32)

        # --- 1. 均线排列健康度 (Alignment Health) ---
        # 评估均线是否呈多头排列（短期均线在长期均线之上）。
        # 使用列表推导式和np.mean高效计算。
        alignment_scores = [
            (df[f'EMA_{periods[i]}_D'] > df[f'EMA_{periods[i+1]}_D']).astype(float).values
            for i in range(len(periods) - 1)
        ]
        # 对所有排列关系的分数取平均，得到总的排列健康度
        alignment_health = np.mean(alignment_scores, axis=0) if alignment_scores else np.full(len(df.index), 0.5)

        # --- 2. 价格位置健康度 (Position Health) ---
        # 评估当前价格是否在所有关键均线之上，代表强势特征。
        position_scores = [(df['close_D'] > df[col]).astype(float).values for col in ma_cols]
        # 对所有位置关系的分数取平均，得到总的位置健康度
        position_health = np.mean(position_scores, axis=0) if position_scores else np.full(len(df.index), 0.5)

        # --- 3. 融合得到最终分数 ---
        # 使用几何平均数融合排列健康度和位置健康度，要求两者俱佳。
        ma_context_score_values = (alignment_health * position_health)**0.5
        
        return pd.Series(ma_context_score_values, index=df.index, dtype=np.float32)















