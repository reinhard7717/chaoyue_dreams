# 文件: strategies/trend_following/intelligence/foundation_intelligence.py
# 基础情报模块 (波动率, 震荡指标)
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score, normalize_to_bipolar, bipolar_to_exclusive_unipolar

class FoundationIntelligence:
    def __init__(self, strategy_instance):
        """
        初始化基础情报模块。
        :param strategy_instance: 策略主实例的引用，用于访问df和atomic_states。
        """
        self.strategy = strategy_instance

    def run_foundation_analysis_command(self) -> Dict[str, pd.Series]:
        """
        【V5.2 · 筹码断层感知版】基础情报分析总指挥
        - 核心升级: 新增调用 `diagnose_vpa_risks`，补完对VPA相关风险的独立诊断能力。
        - 新增功能(V5.2): 新增调用 `diagnose_chip_fault_dynamics`，利用筹码断层数据识别潜在突破机会。
        """
        df = self.strategy.df_indicators
        all_states = {}
        ma_context_score = self._calculate_ma_trend_context(df, [5, 13, 21, 55])
        unified_states = self.diagnose_unified_foundation_signals(df, ma_context_score)
        all_states.update(unified_states)
        all_states.update(self.diagnose_volatility_intelligence(df, ma_context_score))
        all_states.update(self.diagnose_classic_indicators_atomics(df, ma_context_score))
        tactical_ultimate_states = self.diagnose_tactical_foundation_signals(df)
        all_states.update(tactical_ultimate_states)
        vpa_risk_states = self.diagnose_vpa_risks(df)
        all_states.update(vpa_risk_states)
        # 调用新增的筹码断层诊断引擎
        chip_fault_states = self.diagnose_chip_fault_dynamics(df)
        all_states.update(chip_fault_states)
        return all_states

    def diagnose_unified_foundation_signals(self, df: pd.DataFrame, ma_context_score: pd.Series) -> Dict[str, pd.Series]:
        """
        【V16.0 · 四象限重构版】
        - 核心重构: 废弃对通用函数 transmute_health_to_ultimate_signals 的调用，引入“四象限动态分析法”，
                      彻底解决信号命名与逻辑混乱的问题，确保与所有情报模块的哲学统一。
        """
        states = {}
        p_conf = get_params_block(self.strategy, 'foundation_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True): return states
        

        # 步骤一：计算各支柱的静态快照分并融合
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        norm_window = get_param_value(p_synthesis.get('norm_window'), 55)
        pillar_weights = get_param_value(p_conf.get('pillar_weights'), {})
        
        ema_snapshot = self._calculate_ema_health(df, norm_window, [])
        rsi_snapshot = self._calculate_rsi_health(df, norm_window, [], ma_context_score)
        macd_snapshot = self._calculate_macd_health(df, norm_window, [], ma_context_score)
        cmf_snapshot = self._calculate_cmf_health(df, norm_window, [], ma_context_score)
        
        snapshots = {'ema': ema_snapshot, 'rsi': rsi_snapshot, 'macd': macd_snapshot, 'cmf': cmf_snapshot}
        weight_keys = list(snapshots.keys())
        weights_array = np.array([pillar_weights.get(name, 1.0/len(weight_keys)) for name in weight_keys])
        weights_array /= weights_array.sum()
        
        stacked_snapshots = np.stack([s.fillna(0.0).values for s in snapshots.values()], axis=0)
        fused_bipolar_snapshot = pd.Series(
            np.sum(stacked_snapshots * weights_array[:, np.newaxis], axis=0),
            index=df.index, dtype=np.float32
        ).clip(-1, 1)
        
        # 步骤二：分离为纯粹的看涨/看跌健康分，并计算静态共振信号
        bullish_resonance, bearish_resonance = bipolar_to_exclusive_unipolar(fused_bipolar_snapshot)
        states['SCORE_FOUNDATION_BULLISH_RESONANCE'] = bullish_resonance.astype(np.float32)
        states['SCORE_FOUNDATION_BEARISH_RESONANCE'] = bearish_resonance.astype(np.float32)
        
        # 步骤三：计算四象限动态信号
        bull_divergence = self._calculate_holographic_divergence_foundation(bullish_resonance, 5, 21, norm_window)
        bullish_acceleration = bull_divergence.clip(0, 1)
        top_reversal = (bull_divergence.clip(-1, 0) * -1)
        
        bear_divergence = self._calculate_holographic_divergence_foundation(bearish_resonance, 5, 21, norm_window)
        bearish_acceleration = bear_divergence.clip(0, 1)
        bottom_reversal = (bear_divergence.clip(-1, 0) * -1)
        
        # 步骤四：赋值给命名准确的终极信号
        states['SCORE_FOUNDATION_BULLISH_ACCELERATION'] = bullish_acceleration.astype(np.float32)
        states['SCORE_FOUNDATION_TOP_REVERSAL'] = top_reversal.astype(np.float32)
        states['SCORE_FOUNDATION_BEARISH_ACCELERATION'] = bearish_acceleration.astype(np.float32)
        states['SCORE_FOUNDATION_BOTTOM_REVERSAL'] = bottom_reversal.astype(np.float32)
        
        # 步骤五：重铸战术反转信号
        states['SCORE_FOUNDATION_TACTICAL_REVERSAL'] = (bullish_resonance * top_reversal).clip(0, 1).astype(np.float32)
        
        
        return states

    def diagnose_tactical_foundation_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.5 · 职责净化版】战术终极信号合成引擎
        - 核心净化: 彻底移除了所有与“抛压”、“吸收”、“反转”相关的逻辑。
                      这些职责已完全移交至`behavioral_intelligence`模块。
        - 职责定位: 本方法现在只负责处理纯粹的基础层战术信号，如波动率和量价点火。
        """
        states = {}
        # 彻底移除所有与抛压分析相关的逻辑
        compression_state = self.strategy.atomic_states.get('SCORE_VOL_COMPRESSION_STATE', pd.Series(0.5, index=df.index))
        compression_dynamic = self.strategy.atomic_states.get('SCORE_VOL_COMPRESSION_DYNAMIC', pd.Series(0.5, index=df.index))
        states['SCORE_FOUNDATION_VOL_COMPRESSION_OPP'] = (compression_state * compression_dynamic).astype(np.float32)
        expansion_risk_state = self.strategy.atomic_states.get('SCORE_VOL_EXPANSION_RISK_STATE', pd.Series(0.5, index=df.index))
        expansion_risk_dynamic = self.strategy.atomic_states.get('SCORE_VOL_EXPANSION_RISK_DYNAMIC', pd.Series(0.5, index=df.index))
        states['SCORE_FOUNDATION_VOL_EXPANSION_RISK'] = (expansion_risk_state * expansion_risk_dynamic).astype(np.float32)
        ignition_state = self.strategy.atomic_states.get('SCORE_VOL_PRICE_IGNITION_STATE', pd.Series(0.5, index=df.index))
        ignition_dynamic = self.strategy.atomic_states.get('SCORE_VOL_PRICE_IGNITION_DYNAMIC', pd.Series(0.5, index=df.index))
        states['SCORE_FOUNDATION_IGNITION_CONFIRMATION'] = (ignition_state * ignition_dynamic).astype(np.float32)
        
        return states

    def diagnose_vpa_risks(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 · 分层印证版】VPA风险诊断引擎
        - 核心升级: 引入“分层动态印证”框架。对“效率下降”和“成交量加速”风险的计算进行多时间维度的分层验证。
        """
        states = {}
        # 引入分层印证框架
        periods = [1, 5, 13, 21, 55]
        sorted_periods = sorted(periods)
        tf_weights = {1: 0.1, 5: 0.4, 13: 0.3, 21: 0.15, 55: 0.05}
        # --- 1. VPA效率下降风险 (分层计算) ---
        efficiency_decline_scores = {}
        vpa_efficiency_series = df.get('VPA_EFFICIENCY_D', pd.Series(0.5, index=df.index))
        for i, p_tactical in enumerate(sorted_periods):
            p_context = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p_tactical
            tactical_score = normalize_score(vpa_efficiency_series, df.index, p_tactical, ascending=False)
            context_score = normalize_score(vpa_efficiency_series, df.index, p_context, ascending=False)
            efficiency_decline_scores[p_tactical] = (tactical_score * context_score)**0.5
        # 跨周期融合
        final_efficiency_risk = pd.Series(0.0, index=df.index)
        total_weight = sum(tf_weights.values())
        if total_weight > 0:
            for p in periods:
                final_efficiency_risk += efficiency_decline_scores.get(p, 0.0) * (tf_weights.get(p, 0) / total_weight)
        states['SCORE_RISK_VPA_EFFICIENCY_DECLINING'] = final_efficiency_risk.clip(0, 1).astype(np.float32)
        # --- 2. VPA成交量加速风险 (分层计算) ---
        volume_accel_scores = {}
        volume_accel_series = df.get('ACCEL_5_volume_D', pd.Series(0.0, index=df.index))
        for i, p_tactical in enumerate(sorted_periods):
            p_context = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p_tactical
            tactical_score = normalize_score(volume_accel_series, df.index, p_tactical, ascending=True)
            context_score = normalize_score(volume_accel_series, df.index, p_context, ascending=True)
            volume_accel_scores[p_tactical] = (tactical_score * context_score)**0.5
        # 跨周期融合
        final_volume_accel_risk = pd.Series(0.0, index=df.index)
        if total_weight > 0:
            for p in periods:
                final_volume_accel_risk += volume_accel_scores.get(p, 0.0) * (tf_weights.get(p, 0) / total_weight)
        states['SCORE_RISK_VPA_VOLUME_ACCELERATING'] = final_volume_accel_risk.clip(0, 1).astype(np.float32)
        
        return states

    def diagnose_chip_fault_dynamics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 · 分层印证版】筹码断层动态诊断引擎
        - 核心升级: 引入“分层动态印证”框架。对“断层强度”和“真空范围”的评估进行多时间维度的分层验证。
        """
        states = {}
        # 引入分层印证框架
        periods = [5, 13, 21, 55] # 筹码断层使用稍长周期更稳定
        sorted_periods = sorted(periods)
        tf_weights = {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}
        breakout_potential_scores = {}
        fault_strength_series = df.get('chip_fault_strength_D')
        vacuum_percent_series = df.get('chip_fault_vacuum_percent_D')
        if fault_strength_series is None or vacuum_percent_series is None:
            states['SCORE_FOUNDATION_CHIP_FAULT_BREAKOUT'] = pd.Series(0.0, index=df.index, dtype=np.float32)
            return states
        for i, p_tactical in enumerate(sorted_periods):
            p_context = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p_tactical
            # 断层强度分层验证
            tactical_strength = normalize_score(fault_strength_series, df.index, p_tactical, ascending=True)
            context_strength = normalize_score(fault_strength_series, df.index, p_context, ascending=True)
            fused_strength = (tactical_strength * context_strength)**0.5
            # 真空范围分层验证
            tactical_vacuum = normalize_score(vacuum_percent_series, df.index, p_tactical, ascending=True)
            context_vacuum = normalize_score(vacuum_percent_series, df.index, p_context, ascending=True)
            fused_vacuum = (tactical_vacuum * context_vacuum)**0.5
            # 融合生成当期快照分
            breakout_potential_scores[p_tactical] = (fused_strength * fused_vacuum)**0.5
        # 跨周期融合
        final_breakout_potential = pd.Series(0.0, index=df.index)
        total_weight = sum(tf_weights.values())
        if total_weight > 0:
            for p in periods:
                final_breakout_potential += breakout_potential_scores.get(p, 0.0) * (tf_weights.get(p, 0) / total_weight)
        states['SCORE_FOUNDATION_CHIP_FAULT_BREAKOUT'] = final_breakout_potential.clip(0, 1).astype(np.float32)
        
        return states

    # ==============================================================================
    # 以下为重构后的健康度组件计算器，现在返回四维健康度
    # ==============================================================================
    def _calculate_ema_health(self, df: pd.DataFrame, norm_window: int, periods: list) -> pd.Series:
        """
        【V7.0 · 阿波罗审判版】计算EMA维度的双极性快照分
        - 核心重构: 废除“赫利俄斯敕令”，不再执行元分析。
                      回归本源，仅负责计算并返回一个蕴含五维信息的、范围在[-1, 1]的双极性快照分。
        """
        p_conf = get_params_block(self.strategy, 'foundation_ultimate_params', {})
        fusion_weights = p_conf.get('ma_health_fusion_weights', {
            'alignment': 0.15, 'slope': 0.15, 'accel': 0.2, 'relational': 0.25, 'meta_dynamics': 0.25
        })
        ma_periods = [5, 13, 21, 55]
        required_cols = [f'EMA_{p}_D' for p in ma_periods] + [f'SLOPE_{p}_EMA_{p}_D' for p in ma_periods] + [f'ACCEL_{p}_EMA_{p}_D' for p in ma_periods]
        if not all(col in df.columns for col in required_cols):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        
        bull_alignment_scores = [(df[f'EMA_{ma_periods[i]}_D'] > df[f'EMA_{ma_periods[i+1]}_D']).astype(float).values for i in range(len(ma_periods) - 1)]
        alignment_score = np.mean(bull_alignment_scores, axis=0) if bull_alignment_scores else np.full(len(df.index), 0.5)
        
        slope_scores = [normalize_to_bipolar(df[f'SLOPE_{p}_EMA_{p}_D'], df.index, norm_window).values for p in ma_periods]
        accel_scores = [normalize_to_bipolar(df[f'ACCEL_{p}_EMA_{p}_D'], df.index, norm_window).values for p in ma_periods]
        
        relational_scores = []
        for short_p, long_p in [(5, 21), (13, 55)]:
            spread_accel = (df[f'EMA_{short_p}_D'] - df[f'EMA_{long_p}_D']).diff(3).diff(3).fillna(0)
            relational_scores.append(normalize_to_bipolar(spread_accel, df.index, norm_window).values)
            
        meta_dynamics_cols = ['SLOPE_5_EMA_55_D', 'SLOPE_13_EMA_89_D', 'SLOPE_21_EMA_144_D']
        valid_meta_cols = [col for col in meta_dynamics_cols if col in df.columns]
        meta_scores = [normalize_to_bipolar(df[col], df.index, norm_window).values for col in valid_meta_cols] if valid_meta_cols else [np.full(len(df.index), 0.0)]

        # 将所有维度转换为[-1, 1]的双极性分数
        alignment_bipolar = (pd.Series(alignment_score, index=df.index) - 0.5) * 2
        avg_slope_bipolar = pd.Series(np.mean(slope_scores, axis=0), index=df.index)
        avg_accel_bipolar = pd.Series(np.mean(accel_scores, axis=0), index=df.index)
        avg_relational_bipolar = pd.Series(np.mean(relational_scores, axis=0), index=df.index)
        avg_meta_bipolar = pd.Series(np.mean(meta_scores, axis=0), index=df.index)

        # 使用加法模型融合双极性分数
        bipolar_snapshot = (
            alignment_bipolar * fusion_weights.get('alignment', 0.15) +
            avg_slope_bipolar * fusion_weights.get('slope', 0.15) +
            avg_accel_bipolar * fusion_weights.get('accel', 0.2) +
            avg_relational_bipolar * fusion_weights.get('relational', 0.25) +
            avg_meta_bipolar * fusion_weights.get('meta_dynamics', 0.25)
        )
        return bipolar_snapshot.clip(-1, 1).astype(np.float32)

    def _calculate_rsi_health(self, df: pd.DataFrame, norm_window: int, periods: list, ma_context_score: pd.Series) -> pd.Series:
        """
        【V7.0 · 阿波罗审判版】计算RSI维度的双极性快照分
        - 核心重构: 仅返回一个[-1, 1]的双极性快照分。
        """
        if 'RSI_13_D' not in df.columns:
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        return normalize_to_bipolar(df['RSI_13_D'], df.index, norm_window)

    def _calculate_macd_health(self, df: pd.DataFrame, norm_window: int, periods: list, ma_context_score: pd.Series) -> pd.Series:
        """
        【V7.0 · 阿波罗审判版】计算MACD维度的双极性快照分
        - 核心重构: 仅返回一个[-1, 1]的双极性快照分。
        """
        if 'MACDh_13_34_8_D' not in df.columns:
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        return normalize_to_bipolar(df['MACDh_13_34_8_D'], df.index, norm_window)

    def _calculate_cmf_health(self, df: pd.DataFrame, norm_window: int, periods: list, ma_context_score: pd.Series) -> pd.Series:
        """
        【V7.0 · 阿波罗审判版】计算CMF维度的双极性快照分
        - 核心重构: 仅返回一个[-1, 1]的双极性快照分。
        """
        if 'CMF_21_D' not in df.columns:
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        return normalize_to_bipolar(df['CMF_21_D'], df.index, norm_window)

    def _perform_foundation_relational_meta_analysis(self, df: pd.DataFrame, snapshot_score: pd.Series, meta_window: int) -> pd.Series:
        """
        【V2.3 · 加速度校准版】基础情报专用的关系元分析核心引擎
        - 核心修复: 修正了“加速度”计算的致命逻辑错误。加速度是速度的一阶导数，
                      因此其计算应为 relationship_trend.diff(1)，而不是错误的 diff(meta_window)。
        """
        p_conf = get_params_block(self.strategy, 'foundation_ultimate_params', {})
        p_meta = get_param_value(p_conf.get('relational_meta_analysis_params'), {})
        w_state = get_param_value(p_meta.get('state_weight'), 0.3)
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.3)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.4)
        norm_window = 55
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
        final_score = (total_bullish_force - total_bearish_force).clip(-1, 1)
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

    def _calculate_holographic_divergence_foundation(self, series: pd.Series, short_p: int, long_p: int, norm_window: int) -> pd.Series:
        """
        【V1.0 · 新增】基础层专用的全息背离计算引擎
        - 战略意义: 洞察多时间维度的“结构性背离”，输出一个[-1, 1]的双极性背离分数。
        """
        # [代码新增开始]
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
        # [代码新增结束]

    # ==============================================================================
    # 以下为保留的、具有特殊战术意义的模块
    # ==============================================================================

    def diagnose_volatility_intelligence(self, df: pd.DataFrame, ma_context_score: pd.Series) -> Dict[str, pd.Series]:
        """
        【V8.1 · 哥白尼革命同步版】波动率统一情报中心
        - 核心升级: 同步“哥白尼革命”，在调用元分析时传入正确的周期参数 p_tactical。
        """
        states = {}
        if 'BBW_21_2.0_D' not in df.columns or 'hurst_120d_D' not in df.columns:
            return {}
        periods = [1, 5, 13, 21, 55]
        sorted_periods = sorted(periods)
        tf_weights = {1: 0.1, 5: 0.4, 13: 0.3, 21: 0.15, 55: 0.05}
        compression_state_scores = {}
        compression_dynamic_scores = {}
        expansion_risk_state_scores = {}
        expansion_risk_dynamic_scores = {}
        bbw_series = df['BBW_21_2.0_D']
        for i, p_tactical in enumerate(sorted_periods):
            p_context = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p_tactical
            tactical_comp = normalize_score(bbw_series, df.index, p_tactical, ascending=False)
            context_comp = normalize_score(bbw_series, df.index, p_context, ascending=False)
            fused_compression = (tactical_comp * context_comp)**0.5
            state_score_p = (fused_compression * ma_context_score).astype(np.float32)
            compression_state_scores[p_tactical] = state_score_p
            # 同步“哥白尼革命”，传入 meta_window
            current_meta_window = max(1, p_tactical)
            compression_dynamic_scores[p_tactical] = self._perform_foundation_relational_meta_analysis(df, state_score_p, meta_window=current_meta_window)
            
            tactical_exp = normalize_score(bbw_series, df.index, p_tactical, ascending=True)
            context_exp = normalize_score(bbw_series, df.index, p_context, ascending=True)
            fused_expansion = (tactical_exp * context_exp)**0.5
            risk_state_score_p = (fused_expansion * (1 - ma_context_score)).astype(np.float32)
            expansion_risk_state_scores[p_tactical] = risk_state_score_p
            # 同步“哥白尼革命”，传入 meta_window
            expansion_risk_dynamic_scores[p_tactical] = self._perform_foundation_relational_meta_analysis(df, risk_state_score_p, meta_window=current_meta_window)
            
        def fuse_across_periods(scores_dict):
            final_score = pd.Series(0.0, index=df.index)
            total_weight = sum(tf_weights.values())
            if total_weight > 0:
                for p in periods:
                    final_score += scores_dict.get(p, 0.0) * (tf_weights.get(p, 0) / total_weight)
            return final_score.clip(0, 1)
        states['SCORE_VOL_COMPRESSION_STATE'] = fuse_across_periods(compression_state_scores)
        states['SCORE_VOL_COMPRESSION_DYNAMIC'] = fuse_across_periods(compression_dynamic_scores)
        states['SCORE_VOL_EXPANSION_RISK_STATE'] = fuse_across_periods(expansion_risk_state_scores)
        states['SCORE_VOL_EXPANSION_RISK_DYNAMIC'] = fuse_across_periods(expansion_risk_dynamic_scores)
        hurst_score = normalize_score(df['hurst_120d_D'], df.index, 120)
        states['SCORE_TRENDING_REGIME'] = hurst_score
        return states

    def diagnose_classic_indicators_atomics(self, df: pd.DataFrame, ma_context_score: pd.Series) -> Dict[str, pd.Series]:
        """
        【V3.5 · 职责净化版】经典指标原子信号诊断
        - 核心净化: 彻底移除了所有与“抛压”相关的计算逻辑，该职责已移交`behavioral_intelligence`。
        """
        states = {}
        p = get_params_block(self.strategy, 'classic_indicator_params')
        if not get_param_value(p.get('enabled'), True): return states
        required_cols = ['close_D', 'open_D']
        if not all(col in df.columns for col in required_cols):
            return {}
        periods = [1, 5, 13, 21, 55]
        sorted_periods = sorted(periods)
        tf_weights = {1: 0.1, 5: 0.4, 13: 0.3, 21: 0.15, 55: 0.05}
        ignition_state_scores = {}
        ignition_dynamic_scores = {}
        # 移除所有与抛压相关的变量和计算
        candle_body_up = (df['close_D'] - df['open_D']).clip(lower=0)
        for i, p_tactical in enumerate(sorted_periods):
            p_context = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p_tactical
            tactical_price_up = normalize_score(candle_body_up, df.index, p_tactical)
            context_price_up = normalize_score(candle_body_up, df.index, p_context)
            fused_price_up = (tactical_price_up * context_price_up)**0.5
            vol_slope_series = df.get(f'SLOPE_{p_tactical}_volume_D', pd.Series(0.0, index=df.index)).clip(lower=0)
            vol_accel_series = df.get(f'ACCEL_{p_tactical}_volume_D', pd.Series(0.0, index=df.index)).clip(lower=0)
            tactical_vol_slope = normalize_score(vol_slope_series, df.index, p_tactical)
            tactical_vol_accel = normalize_score(vol_accel_series, df.index, p_tactical)
            context_vol_slope = normalize_score(vol_slope_series, df.index, p_context)
            context_vol_accel = normalize_score(vol_accel_series, df.index, p_context)
            fused_vol_slope = (tactical_vol_slope * context_vol_slope)**0.5
            fused_vol_accel = (tactical_vol_accel * context_vol_accel)**0.5
            fused_volume_igniting = fused_vol_slope * fused_vol_accel
            ignition_snapshot = (fused_price_up * fused_volume_igniting * ma_context_score).astype(np.float32)
            ignition_state_scores[p_tactical] = ignition_snapshot
            current_meta_window = max(1, p_tactical)
            ignition_dynamic_scores[p_tactical] = self._perform_foundation_relational_meta_analysis(df, ignition_snapshot, meta_window=current_meta_window)
        def fuse_across_periods(scores_dict):
            final_score = pd.Series(0.0, index=df.index)
            total_weight = sum(tf_weights.values())
            if total_weight > 0:
                for p in periods:
                    final_score += scores_dict.get(p, 0.0) * (tf_weights.get(p, 0) / total_weight)
            return final_score.clip(0, 1)
        states['SCORE_VOL_PRICE_IGNITION_STATE'] = fuse_across_periods(ignition_state_scores)
        states['SCORE_VOL_PRICE_IGNITION_DYNAMIC'] = fuse_across_periods(ignition_dynamic_scores)
        
        return states










