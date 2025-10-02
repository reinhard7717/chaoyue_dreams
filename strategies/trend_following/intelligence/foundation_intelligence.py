# 文件: strategies/trend_following/intelligence/foundation_intelligence.py
# 基础情报模块 (波动率, 震荡指标)
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following.utils import transmute_health_to_ultimate_signals, get_params_block, get_param_value, calculate_holographic_dynamics, normalize_score, normalize_to_bipolar

class FoundationIntelligence:
    def __init__(self, strategy_instance):
        """
        初始化基础情报模块。
        :param strategy_instance: 策略主实例的引用，用于访问df和atomic_states。
        """
        self.strategy = strategy_instance

    def run_foundation_analysis_command(self) -> Dict[str, pd.Series]:
        """
        【V5.0 · 战术升维版】基础情报分析总指挥
        - 核心重构 (本次修改):
          - [新增步骤] 在生成所有原子信号后，调用新的 `diagnose_tactical_foundation_signals` 引擎，
                      将战术原子信号融合成新的战术终极信号。
        """
        df = self.strategy.df_indicators
        all_states = {}

        # 步骤 1: 执行唯一的、统一的终极信号引擎
        unified_states = self.diagnose_unified_foundation_signals(df)
        all_states.update(unified_states)

        # 步骤 2: 执行具有特殊战术意义的模块，生成战术原子信号
        # 这些模块现在生成的是 _STATE 和 _DYNAMIC 原子信号
        all_states.update(self.diagnose_volatility_intelligence(df))
        all_states.update(self.diagnose_classic_indicators_atomics(df))
        
        # 步骤 3: 调用新的战术终极信号合成引擎
        # 这个引擎会从 atomic_states 中读取刚刚生成的战术原子信号，并将其融合成终极信号
        tactical_ultimate_states = self.diagnose_tactical_foundation_signals(df)
        all_states.update(tactical_ultimate_states)
        
        return all_states

    def diagnose_unified_foundation_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V14.0 · 圣杯契约版】
        - 核心革命: 不再读取本地的、重复的合成参数，而是从最高指挥部获取唯一的“圣杯”配置
                      (`ultimate_signal_synthesis_params`)，并将其传递给中央合成引擎。
        """
        states = {}
        p_conf = get_params_block(self.strategy, 'foundation_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True): return states
        # 获取中央“圣杯”配置
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        periods = get_param_value(p_synthesis.get('periods'), [1, 5, 13, 21, 55])
        norm_window = get_param_value(p_synthesis.get('norm_window'), 55)
        dynamic_weights = {'slope': 0.6, 'accel': 0.4}
        health_data = { 's_bull': [], 's_bear': [], 'd_intensity': [] } 
        calculators = { 'ema': self._calculate_ema_health, 'rsi': self._calculate_rsi_health, 'macd': self._calculate_macd_health, 'cmf': self._calculate_cmf_health }
        for name, calculator in calculators.items():
            s_bull, s_bear, d_intensity = calculator(df, norm_window, dynamic_weights, periods)
            health_data['s_bull'].append(s_bull) 
            health_data['s_bear'].append(s_bear) 
            health_data['d_intensity'].append(d_intensity)
        overall_health = {}
        for health_type, health_sources in [ ('s_bull', health_data['s_bull']), ('s_bear', health_data['s_bear']), ('d_intensity', health_data['d_intensity']) ]:
            overall_health[health_type] = {}
            for p in periods:
                components_for_period = [pillar_dict[p].values for pillar_dict in health_sources if p in pillar_dict]
                if components_for_period:
                    stacked_values = np.stack(components_for_period, axis=0)
                    fused_values = np.prod(stacked_values, axis=0) ** (1.0 / stacked_values.shape[0])
                    overall_health[health_type][p] = pd.Series(fused_values, index=df.index, dtype=np.float32)
                else:
                    overall_health[health_type][p] = pd.Series(0.5, index=df.index, dtype=np.float32)
        self.strategy.atomic_states['__FOUNDATION_overall_health'] = overall_health
        # 传入唯一的“圣杯”配置
        ultimate_signals = transmute_health_to_ultimate_signals(
            df=df,
            atomic_states=self.strategy.atomic_states,
            overall_health=overall_health,
            params=p_synthesis,
            domain_prefix="FOUNDATION"
        )
        states.update(ultimate_signals)
        return states

    def diagnose_tactical_foundation_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 · 新增】战术终极信号合成引擎
        - 核心逻辑: 将来自特殊战术模块的“状态分”和“动态分”原子信号，融合成可直接决策的战术终极信号。
        """
        states = {}
        
        # 1. 波动率压缩机会信号 (Volatility Compression Opportunity)
        # 核心关系：一个好的“压缩状态”，如果其“压缩动态”也在增强，则构成一个强烈的突破前夕机会信号。
        compression_state = self.strategy.atomic_states.get('SCORE_VOL_COMPRESSION_STATE', pd.Series(0.5, index=df.index))
        compression_dynamic = self.strategy.atomic_states.get('SCORE_VOL_COMPRESSION_DYNAMIC', pd.Series(0.5, index=df.index))
        states['SCORE_FOUNDATION_VOL_COMPRESSION_OPP'] = (compression_state * compression_dynamic).astype(np.float32)

        # 2. 波动率扩张风险信号 (Volatility Expansion Risk)
        # 核心关系：一个危险的“扩张风险状态”，如果其“风险动态”还在恶化，则构成一个强烈的顶部或破位风险信号。
        expansion_risk_state = self.strategy.atomic_states.get('SCORE_VOL_EXPANSION_RISK_STATE', pd.Series(0.5, index=df.index))
        expansion_risk_dynamic = self.strategy.atomic_states.get('SCORE_VOL_EXPANSION_RISK_DYNAMIC', pd.Series(0.5, index=df.index))
        states['SCORE_FOUNDATION_VOL_EXPANSION_RISK'] = (expansion_risk_state * expansion_risk_dynamic).astype(np.float32)

        # 3. 量价点火确认信号 (Volume-Price Ignition Confirmation)
        # 核心关系：一个良好的“量价点火状态”，如果其“点火动态”也在增强，则构成一个强烈的上涨确认信号。
        ignition_state = self.strategy.atomic_states.get('SCORE_VOL_PRICE_IGNITION_STATE', pd.Series(0.5, index=df.index))
        ignition_dynamic = self.strategy.atomic_states.get('SCORE_VOL_PRICE_IGNITION_DYNAMIC', pd.Series(0.5, index=df.index))
        states['SCORE_FOUNDATION_IGNITION_CONFIRMATION'] = (ignition_state * ignition_dynamic).astype(np.float32)

        # 4. 恐慌抛售风险信号 (Panic Selling Risk)
        # 核心关系：一个危险的“恐慌抛售状态”，如果其“恐慌动态”还在恶化，则构成一个强烈的下跌风险信号。
        panic_risk_state = self.strategy.atomic_states.get('SCORE_VOL_PRICE_PANIC_RISK_STATE', pd.Series(0.5, index=df.index))
        panic_risk_dynamic = self.strategy.atomic_states.get('SCORE_VOL_PRICE_PANIC_RISK_DYNAMIC', pd.Series(0.5, index=df.index))
        states['SCORE_FOUNDATION_PANIC_SELLING_RISK'] = (panic_risk_state * panic_risk_dynamic).astype(np.float32)

        return states

    # ==============================================================================
    # 以下为重构后的健康度组件计算器，现在返回四维健康度
    # ==============================================================================
    def _calculate_ema_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict, Dict, Dict]:
        """【V5.0 · 关系元分析版】计算EMA维度的三维健康度"""
        s_bull, s_bear, d_intensity = {}, {}, {}
        p_conf = get_params_block(self.strategy, 'foundation_ultimate_params', {})
        fusion_weights = get_param_value(p_conf.get('ma_health_fusion_weights'), {'alignment': 0.1, 'slope': 0.2, 'accel': 0.2, 'relational': 0.5})
        ma_periods = [5, 13, 21, 55]
        
        # 步骤一：计算四维融合的“瞬时关系快照分” (s_bull)
        bull_alignment_scores, bear_alignment_scores = [], []
        for i in range(len(ma_periods) - 1):
            short_col, long_col = f'EMA_{ma_periods[i]}_D', f'EMA_{ma_periods[i+1]}_D'
            if short_col in df and long_col in df:
                bull_alignment_scores.append((df[short_col] > df[long_col]).astype(float))
                bear_alignment_scores.append((df[short_col] < df[long_col]).astype(float))
        alignment_score = pd.DataFrame(bull_alignment_scores).mean().fillna(0.5) if bull_alignment_scores else pd.Series(0.5, index=df.index)
        static_bear_score = pd.DataFrame(bear_alignment_scores).mean().fillna(0.5) if bear_alignment_scores else pd.Series(0.5, index=df.index)

        slope_health_scores, accel_health_scores, relational_health_scores = [], [], []
        for p in ma_periods:
            slope_col = f'SLOPE_{p}_EMA_{p}_D' if p != 1 else f'SLOPE_1_close_D'
            accel_col = f'ACCEL_{p}_EMA_{p}_D' if p != 1 else f'ACCEL_1_close_D'
            if slope_col in df.columns: slope_health_scores.append((normalize_to_bipolar(df[slope_col], df.index, norm_window) + 1) / 2.0)
            if accel_col in df.columns: accel_health_scores.append((normalize_to_bipolar(df[accel_col], df.index, norm_window) + 1) / 2.0)
        for short_p, long_p in [(5, 21), (13, 55)]:
            short_ma_col, long_ma_col = f'EMA_{short_p}_D', f'EMA_{long_p}_D'
            if short_ma_col in df.columns and long_ma_col in df.columns:
                spread_accel = (df[short_ma_col] - df[long_ma_col]).diff(3).diff(3).fillna(0)
                relational_health_scores.append((normalize_to_bipolar(spread_accel, df.index, norm_window) + 1) / 2.0)

        avg_slope_health = pd.concat(slope_health_scores, axis=1).mean(axis=1).fillna(0.5) if slope_health_scores else pd.Series(0.5, index=df.index)
        avg_accel_health = pd.concat(accel_health_scores, axis=1).mean(axis=1).fillna(0.5) if accel_health_scores else pd.Series(0.5, index=df.index)
        avg_relational_health = pd.concat(relational_health_scores, axis=1).mean(axis=1).fillna(0.5) if relational_health_scores else pd.Series(0.5, index=df.index)

        static_bull_score = (
            alignment_score * fusion_weights.get('alignment', 0.1) +
            avg_slope_health * fusion_weights.get('slope', 0.2) +
            avg_accel_health * fusion_weights.get('accel', 0.2) +
            avg_relational_health * fusion_weights.get('relational', 0.5)
        )

        # 步骤二：对“瞬时关系快照分”进行元分析，得到动态强度分 (d_intensity)
        unified_d_intensity = self._perform_foundation_relational_meta_analysis(df, static_bull_score)

        # 步骤三：更新输出
        for p in periods:
            s_bull[p] = static_bull_score
            s_bear[p] = static_bear_score
            d_intensity[p] = unified_d_intensity
        
        return s_bull, s_bear, d_intensity

    def _calculate_rsi_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict, Dict, Dict]:
        """【V5.0 · 关系元分析版】计算RSI维度的三维健康度"""
        s_bull, s_bear, d_intensity = {}, {}, {}
        
        # 步骤一：计算原始的、纯粹的指标静态健康度
        indicator_static_bull = normalize_score(df.get('RSI_13_D'), df.index, norm_window, ascending=True)
        indicator_static_bear = normalize_score(df.get('RSI_13_D'), df.index, norm_window, ascending=False)

        # 步骤二：获取均线趋势上下文分数
        ma_context_score = self._calculate_ma_trend_context(df, [5, 13, 21, 55])

        # 步骤三：构建融合了趋势上下文的“瞬时关系快照分”
        bullish_snapshot_score = (indicator_static_bull * ma_context_score)
        bearish_snapshot_score = (indicator_static_bear * (1 - ma_context_score))

        # 步骤四：对快照分进行关系元分析，得到最终的动态强度分
        unified_d_intensity = self._perform_foundation_relational_meta_analysis(df, bullish_snapshot_score)

        # 步骤五：更新输出
        for p in periods:
            s_bull[p] = bullish_snapshot_score
            s_bear[p] = bearish_snapshot_score
            d_intensity[p] = unified_d_intensity
        
        return s_bull, s_bear, d_intensity

    def _calculate_macd_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict, Dict, Dict]:
        """【V5.0 · 关系元分析版】计算MACD维度的三维健康度"""
        s_bull, s_bear, d_intensity = {}, {}, {}
        
        # 步骤一：计算原始的、纯粹的指标静态健康度
        indicator_static_bull = normalize_score(df.get('MACDh_13_34_8_D'), df.index, norm_window, ascending=True)
        indicator_static_bear = normalize_score(df.get('MACDh_13_34_8_D'), df.index, norm_window, ascending=False)

        # 步骤二：获取均线趋势上下文分数
        ma_context_score = self._calculate_ma_trend_context(df, [5, 13, 21, 55])

        # 步骤三：构建融合了趋势上下文的“瞬时关系快照分”
        bullish_snapshot_score = (indicator_static_bull * ma_context_score)
        bearish_snapshot_score = (indicator_static_bear * (1 - ma_context_score))

        # 步骤四：对快照分进行关系元分析，得到最终的动态强度分
        unified_d_intensity = self._perform_foundation_relational_meta_analysis(df, bullish_snapshot_score)

        # 步骤五：更新输出
        for p in periods:
            s_bull[p] = bullish_snapshot_score
            s_bear[p] = bearish_snapshot_score
            d_intensity[p] = unified_d_intensity
        
        return s_bull, s_bear, d_intensity

    def _calculate_cmf_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict, Dict, Dict]:
        """【V5.0 · 关系元分析版】计算CMF维度的三维健康度"""
        s_bull, s_bear, d_intensity = {}, {}, {}
        
        # 步骤一：计算原始的、纯粹的指标静态健康度
        indicator_static_bull = normalize_score(df.get('CMF_21_D'), df.index, norm_window, ascending=True)
        indicator_static_bear = normalize_score(df.get('CMF_21_D'), df.index, norm_window, ascending=False)

        # 步骤二：获取均线趋势上下文分数
        ma_context_score = self._calculate_ma_trend_context(df, [5, 13, 21, 55])

        # 步骤三：构建融合了趋势上下文的“瞬时关系快照分”
        bullish_snapshot_score = (indicator_static_bull * ma_context_score)
        bearish_snapshot_score = (indicator_static_bear * (1 - ma_context_score))

        # 步骤四：对快照分进行关系元分析，得到最终的动态强度分
        unified_d_intensity = self._perform_foundation_relational_meta_analysis(df, bullish_snapshot_score)

        # 步骤五：更新输出
        for p in periods:
            s_bull[p] = bullish_snapshot_score
            s_bear[p] = bearish_snapshot_score
            d_intensity[p] = unified_d_intensity
        
        return s_bull, s_bear, d_intensity

    def _perform_foundation_relational_meta_analysis(self, df: pd.DataFrame, snapshot_score: pd.Series) -> pd.Series:
        """
        【V1.0 · 新增】基础情报专用的关系元分析核心引擎 (赫拉织布机V2)
        - 核心逻辑: 实现“状态 * (1 + 动态杠杆)”的动态价值调制范式。
        """
        # 从配置中获取动态杠杆权重
        p_conf = get_params_block(self.strategy, 'foundation_ultimate_params', {})
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

    # ==============================================================================
    # 以下为保留的、具有特殊战术意义的模块
    # ==============================================================================

    def diagnose_volatility_intelligence(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V7.0 · 关系元分析版】波动率统一情报中心
        - 核心革命: 废除旧的、基于孤立导数的信号，重构为“状态分”和“动态分”的新范式。
        """
        states = {}
        norm_window = 120
        
        # 获取均线趋势上下文
        ma_context_score = self._calculate_ma_trend_context(df, [5, 13, 21, 55])

        # 步骤一：构建“波动率压缩”的瞬时关系快照分 (状态分)
        # 核心关系：在强势趋势背景下(ma_context高)，波动率受到压缩(BBW低)。
        compression_score_raw = normalize_score(df.get('BBW_21_2.0_D'), df.index, norm_window, ascending=False)
        compression_state_score = (compression_score_raw * ma_context_score)
        states['SCORE_VOL_COMPRESSION_STATE'] = compression_state_score.astype(np.float32)

        # 步骤二：对“波动率压缩关系”进行元分析，得到动态分
        compression_dynamic_score = self._perform_foundation_relational_meta_analysis(df, compression_state_score)
        states['SCORE_VOL_COMPRESSION_DYNAMIC'] = compression_dynamic_score.astype(np.float32)

        # 步骤三：构建“波动率扩张风险”的瞬时关系快照分 (状态分)
        # 核心关系：在弱势趋势背景下(ma_context低)，波动率正在扩张(BBW高)。
        expansion_score_raw = normalize_score(df.get('BBW_21_2.0_D'), df.index, norm_window, ascending=True)
        expansion_risk_state_score = (expansion_score_raw * (1 - ma_context_score))
        states['SCORE_VOL_EXPANSION_RISK_STATE'] = expansion_risk_state_score.astype(np.float32)

        # 步骤四：对“波动率扩张风险关系”进行元分析，得到动态分
        expansion_risk_dynamic_score = self._perform_foundation_relational_meta_analysis(df, expansion_risk_state_score)
        states['SCORE_VOL_EXPANSION_RISK_DYNAMIC'] = expansion_risk_dynamic_score.astype(np.float32)
        
        # Hurst指数保持不变，作为独立的宏观状态判断
        hurst_score = normalize_score(df.get('hurst_120d_D'), df.index, norm_window)
        states['SCORE_TRENDING_REGIME'] = hurst_score

        return states

    def diagnose_classic_indicators_atomics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 · 关系元分析版】经典指标原子信号诊断
        - 核心革命: 将“量价点火”信号升维，引入均线趋势上下文，并用关系元分析捕捉其动态。
        """
        states = {}
        p = get_params_block(self.strategy, 'classic_indicator_params')
        if not get_param_value(p.get('enabled'), True): return states
        
        norm_window = 120
        
        # 获取均线趋势上下文
        ma_context_score = self._calculate_ma_trend_context(df, [5, 13, 21, 55])

        # 步骤一：构建“量价齐升点火”的瞬时关系快照分 (状态分)
        # 核心关系：在强势趋势背景下(ma_context高)，出现阳线实体，且成交量加速放大。
        candle_body_up = (df.get('close_D', 0) - df.get('open_D', 0)).clip(lower=0)
        score_price_up_strength = normalize_score(candle_body_up, df.index, norm_window)
        score_vol_slope_up = normalize_score(df.get('SLOPE_5_volume_D', pd.Series(0.5, index=df.index)).clip(lower=0), df.index, norm_window)
        score_vol_accel_up = normalize_score(df.get('ACCEL_5_volume_D', pd.Series(0.5, index=df.index)).clip(lower=0), df.index, norm_window)
        score_volume_igniting = score_vol_slope_up * score_vol_accel_up
        ignition_state_score = (score_price_up_strength * score_volume_igniting * ma_context_score)
        states['SCORE_VOL_PRICE_IGNITION_STATE'] = ignition_state_score.astype(np.float32)

        # 步骤二：对“量价点火关系”进行元分析，得到动态分
        ignition_dynamic_score = self._perform_foundation_relational_meta_analysis(df, ignition_state_score)
        states['SCORE_VOL_PRICE_IGNITION_DYNAMIC'] = ignition_dynamic_score.astype(np.float32)

        # 步骤三：构建“放量恐慌下跌”的瞬时关系快照分 (状态分)
        # 核心关系：在弱势趋势背景下(ma_context低)，出现阴线实体，且成交量加速放大。
        candle_body_down = (df.get('open_D', 0) - df.get('close_D', 0)).clip(lower=0)
        score_price_down_strength = normalize_score(candle_body_down, df.index, norm_window)
        panic_state_score = (score_price_down_strength * score_volume_igniting * (1 - ma_context_score))
        states['SCORE_VOL_PRICE_PANIC_RISK_STATE'] = panic_state_score.astype(np.float32)

        # 步骤四：对“放量恐慌关系”进行元分析，得到动态分
        panic_dynamic_score = self._perform_foundation_relational_meta_analysis(df, panic_state_score)
        states['SCORE_VOL_PRICE_PANIC_RISK_DYNAMIC'] = panic_dynamic_score.astype(np.float32)

        return states










