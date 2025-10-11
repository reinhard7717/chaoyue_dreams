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
        【V5.1 · VPA风险感知版】基础情报分析总指挥
        - 核心升级: 新增调用 `diagnose_vpa_risks`，补完对VPA相关风险的独立诊断能力。
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
        # 调用新的VPA风险诊断引擎
        vpa_risk_states = self.diagnose_vpa_risks(df)
        all_states.update(vpa_risk_states)
        return all_states

    def diagnose_vpa_risks(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 · 新增】VPA风险诊断引擎
        - 核心职责: 补完对VPA相关风险的独立诊断能力，生成可直接消费的风险信号。
        """
        states = {}
        norm_window = 120
        
        # 1. VPA效率下降风险
        # 效率指标本身越低风险越高，因此使用 ascending=False
        vpa_efficiency_decline_risk = normalize_score(df.get('VPA_EFFICIENCY_D', pd.Series(0.5, index=df.index)), df.index, norm_window, ascending=False)
        states['SCORE_RISK_VPA_EFFICIENCY_DECLINING'] = vpa_efficiency_decline_risk.astype(np.float32)
        
        # 2. VPA成交量加速风险
        # 成交量加速度越大，风险越高
        vpa_volume_accelerating_risk = normalize_score(df.get('ACCEL_5_volume_D', pd.Series(0.0, index=df.index)), df.index, norm_window, ascending=True)
        states['SCORE_RISK_VPA_VOLUME_ACCELERATING'] = vpa_volume_accelerating_risk.astype(np.float32)
        
        return states

    def diagnose_unified_foundation_signals(self, df: pd.DataFrame, ma_context_score: pd.Series) -> Dict[str, pd.Series]:
        """
        【V14.1 · 权重激活版】
        - 核心修复: 激活 pillar_weights 配置，使用加权几何平均数融合各支柱健康度，确保配置生效。
        """
        states = {}
        p_conf = get_params_block(self.strategy, 'foundation_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True): return states
        
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        periods = get_param_value(p_synthesis.get('periods'), [1, 5, 13, 21, 55])
        norm_window = get_param_value(p_synthesis.get('norm_window'), 55)
        
        # 读取支柱权重配置
        pillar_weights = get_param_value(p_conf.get('pillar_weights'), {})
        
        health_data = { 's_bull': [], 's_bear': [], 'd_intensity': [] } 
        calculators = {
            'ema': lambda: self._calculate_ema_health(df, norm_window, periods),
            'rsi': lambda: self._calculate_rsi_health(df, norm_window, periods, ma_context_score),
            'macd': lambda: self._calculate_macd_health(df, norm_window, periods, ma_context_score),
            'cmf': lambda: self._calculate_cmf_health(df, norm_window, periods, ma_context_score)
        }
        
        # 提前构建权重数组
        weight_keys = list(calculators.keys())
        weights_array = np.array([pillar_weights.get(name, 0.25) for name in weight_keys])
        weights_array /= weights_array.sum() # 权重归一化

        for name, calculator in calculators.items():
            s_bull, s_bear, d_intensity = calculator()
            health_data['s_bull'].append(s_bull) 
            health_data['s_bear'].append(s_bear) 
            health_data['d_intensity'].append(d_intensity)

        overall_health = {}
        for health_type, health_sources in health_data.items():
            overall_health[health_type] = {}
            for p in periods:
                components_for_period = [pillar_dict[p].values for pillar_dict in health_sources if p in pillar_dict]
                if components_for_period:
                    stacked_values = np.stack(components_for_period, axis=0)
                    # 使用加权几何平均数进行融合，确保权重配置生效
                    fused_values = np.prod(stacked_values ** weights_array[:, np.newaxis], axis=0)
                    overall_health[health_type][p] = pd.Series(fused_values, index=df.index, dtype=np.float32)
                else:
                    overall_health[health_type][p] = pd.Series(0.5, index=df.index, dtype=np.float32)
        
        self.strategy.atomic_states['__FOUNDATION_overall_health'] = overall_health
        
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
    def _calculate_ema_health(self, df: pd.DataFrame, norm_window: int, periods: list) -> Tuple[Dict, Dict, Dict]:
        """
        【V5.0 · 关系元分析版】计算EMA维度的三维健康度
        - 核心逻辑: 从均线排列、斜率、加速度、关系四个维度综合评估趋势健康度。
        - 优化说明: 1. 使用Numpy向量化操作替换原先低效的`pd.DataFrame.mean()`和`pd.concat().mean()`，显著提升性能。
                      2. 增加对所需指标列的健壮性检查。
        """
        s_bull, s_bear, d_intensity = {}, {}, {}
        p_conf = get_params_block(self.strategy, 'foundation_ultimate_params', {})
        fusion_weights = p_conf.get('ma_health_fusion_weights', {'alignment': 0.1, 'slope': 0.2, 'accel': 0.2, 'relational': 0.5})
        ma_periods = [5, 13, 21, 55]
        
        # 检查所需列，如果缺失则返回默认值
        required_cols = [f'EMA_{p}_D' for p in ma_periods] + [f'SLOPE_{p}_EMA_{p}_D' for p in ma_periods if p != 1] + [f'ACCEL_{p}_EMA_{p}_D' for p in ma_periods if p != 1]
        if not all(col in df.columns for col in required_cols):
            default_series = pd.Series(0.5, index=df.index, dtype=np.float32)
            for p in periods:
                s_bull[p], s_bear[p], d_intensity[p] = default_series.copy(), default_series.copy(), default_series.copy()
            return s_bull, s_bear, d_intensity

        # --- 步骤一：计算四维融合的“瞬时关系快照分” (s_bull) ---
        # 1.1 均线排列健康度
        bull_alignment_scores = [(df[f'EMA_{ma_periods[i]}_D'] > df[f'EMA_{ma_periods[i+1]}_D']).astype(float).values for i in range(len(ma_periods) - 1)]
        bear_alignment_scores = [(df[f'EMA_{ma_periods[i]}_D'] < df[f'EMA_{ma_periods[i+1]}_D']).astype(float).values for i in range(len(ma_periods) - 1)]
        # 使用Numpy高效计算均值
        alignment_score = np.mean(bull_alignment_scores, axis=0) if bull_alignment_scores else np.full(len(df.index), 0.5)
        static_bear_score = np.mean(bear_alignment_scores, axis=0) if bear_alignment_scores else np.full(len(df.index), 0.5)

        # 1.2 斜率、加速度、关系健康度
        slope_health_scores = [((normalize_to_bipolar(df[f'SLOPE_{p}_EMA_{p}_D' if p != 1 else 'SLOPE_1_close_D'], df.index, norm_window) + 1) / 2.0).values for p in ma_periods]
        accel_health_scores = [((normalize_to_bipolar(df[f'ACCEL_{p}_EMA_{p}_D' if p != 1 else 'ACCEL_1_close_D'], df.index, norm_window) + 1) / 2.0).values for p in ma_periods]
        relational_health_scores = []
        for short_p, long_p in [(5, 21), (13, 55)]:
            spread_accel = (df[f'EMA_{short_p}_D'] - df[f'EMA_{long_p}_D']).diff(3).diff(3).fillna(0)
            relational_health_scores.append(((normalize_to_bipolar(spread_accel, df.index, norm_window) + 1) / 2.0).values)
        
        # 使用Numpy高效计算均值，避免创建和合并DataFrame
        avg_slope_health = np.mean(slope_health_scores, axis=0) if slope_health_scores else np.full(len(df.index), 0.5)
        avg_accel_health = np.mean(accel_health_scores, axis=0) if accel_health_scores else np.full(len(df.index), 0.5)
        avg_relational_health = np.mean(relational_health_scores, axis=0) if relational_health_scores else np.full(len(df.index), 0.5)

        # 1.3 加权融合四维度分数
        static_bull_score_values = (
            alignment_score * fusion_weights.get('alignment', 0.1) +
            avg_slope_health * fusion_weights.get('slope', 0.2) +
            avg_accel_health * fusion_weights.get('accel', 0.2) +
            avg_relational_health * fusion_weights.get('relational', 0.5)
        )
        static_bull_score = pd.Series(static_bull_score_values, index=df.index, dtype=np.float32)
        static_bear_score = pd.Series(static_bear_score, index=df.index, dtype=np.float32)

        # --- 步骤二：对“瞬时关系快照分”进行元分析，得到动态强度分 (d_intensity) ---
        unified_d_intensity = self._perform_foundation_relational_meta_analysis(df, static_bull_score)

        # --- 步骤三：更新输出 ---
        for p in periods:
            s_bull[p] = static_bull_score
            s_bear[p] = static_bear_score
            d_intensity[p] = unified_d_intensity
        
        return s_bull, s_bear, d_intensity

    def _calculate_rsi_health(self, df: pd.DataFrame, norm_window: int, periods: list, ma_context_score: pd.Series) -> Tuple[Dict, Dict, Dict]:
        """
        【V6.0 · 德尔斐神谕协议版】计算RSI维度的三维健康度
        - 核心修正: 签署“德尔斐神谕协议”，剥离 ma_context_score 对 s_bull/s_bear 的污染。
        """
        s_bull, s_bear, d_intensity = {}, {}, {}
        if 'RSI_13_D' not in df.columns:
            default_series = pd.Series(0.5, index=df.index, dtype=np.float32)
            for p in periods:
                s_bull[p], s_bear[p], d_intensity[p] = default_series.copy(), default_series.copy(), default_series.copy()
            return s_bull, s_bear, d_intensity
        indicator_static_bull = normalize_score(df['RSI_13_D'], df.index, norm_window, ascending=True)
        indicator_static_bear = normalize_score(df['RSI_13_D'], df.index, norm_window, ascending=False)
        # bullish_snapshot_score 和 bearish_snapshot_score 现在是纯粹的静态分
        bullish_snapshot_score = indicator_static_bull.astype(np.float32)
        bearish_snapshot_score = indicator_static_bear.astype(np.float32)
        unified_d_intensity = self._perform_foundation_relational_meta_analysis(df, bullish_snapshot_score)
        for p in periods:
            s_bull[p] = bullish_snapshot_score
            s_bear[p] = bearish_snapshot_score
            d_intensity[p] = unified_d_intensity
        return s_bull, s_bear, d_intensity

    def _calculate_macd_health(self, df: pd.DataFrame, norm_window: int, periods: list, ma_context_score: pd.Series) -> Tuple[Dict, Dict, Dict]:
        """
        【V6.0 · 德尔斐神谕协议版】计算MACD维度的三维健康度
        - 核心修正: 签署“德尔斐神谕协议”，剥离 ma_context_score 对 s_bull/s_bear 的污染。
        """
        s_bull, s_bear, d_intensity = {}, {}, {}
        if 'MACDh_13_34_8_D' not in df.columns:
            default_series = pd.Series(0.5, index=df.index, dtype=np.float32)
            for p in periods:
                s_bull[p], s_bear[p], d_intensity[p] = default_series.copy(), default_series.copy(), default_series.copy()
            return s_bull, s_bear, d_intensity
        indicator_static_bull = normalize_score(df['MACDh_13_34_8_D'], df.index, norm_window, ascending=True)
        indicator_static_bear = normalize_score(df['MACDh_13_34_8_D'], df.index, norm_window, ascending=False)
        # bullish_snapshot_score 和 bearish_snapshot_score 现在是纯粹的静态分
        bullish_snapshot_score = indicator_static_bull.astype(np.float32)
        bearish_snapshot_score = indicator_static_bear.astype(np.float32)
        unified_d_intensity = self._perform_foundation_relational_meta_analysis(df, bullish_snapshot_score)
        for p in periods:
            s_bull[p] = bullish_snapshot_score
            s_bear[p] = bearish_snapshot_score
            d_intensity[p] = unified_d_intensity
        return s_bull, s_bear, d_intensity

    def _calculate_cmf_health(self, df: pd.DataFrame, norm_window: int, periods: list, ma_context_score: pd.Series) -> Tuple[Dict, Dict, Dict]:
        """
        【V6.0 · 德尔斐神谕协议版】计算CMF维度的三维健康度
        - 核心修正: 签署“德尔斐神谕协议”，剥离 ma_context_score 对 s_bull/s_bear 的污染。
        """
        s_bull, s_bear, d_intensity = {}, {}, {}
        if 'CMF_21_D' not in df.columns:
            default_series = pd.Series(0.5, index=df.index, dtype=np.float32)
            for p in periods:
                s_bull[p], s_bear[p], d_intensity[p] = default_series.copy(), default_series.copy(), default_series.copy()
            return s_bull, s_bear, d_intensity
        indicator_static_bull = normalize_score(df['CMF_21_D'], df.index, norm_window, ascending=True)
        indicator_static_bear = normalize_score(df['CMF_21_D'], df.index, norm_window, ascending=False)
        # bullish_snapshot_score 和 bearish_snapshot_score 现在是纯粹的静态分
        bullish_snapshot_score = indicator_static_bull.astype(np.float32)
        bearish_snapshot_score = indicator_static_bear.astype(np.float32)
        unified_d_intensity = self._perform_foundation_relational_meta_analysis(df, bullish_snapshot_score)
        for p in periods:
            s_bull[p] = bullish_snapshot_score
            s_bear[p] = bearish_snapshot_score
            d_intensity[p] = unified_d_intensity
        return s_bull, s_bear, d_intensity

    def _perform_foundation_relational_meta_analysis(self, df: pd.DataFrame, snapshot_score: pd.Series) -> pd.Series:
        """
        【V2.0 · 阿瑞斯之怒协议版】基础情报专用的关系元分析核心引擎
        - 核心革命: 响应“重变化、轻状态”的哲学，从“状态 * (1 + 动态)”的乘法模型，升级为
                      “(状态*权重) + (速度*权重) + (加速度*权重)”的加法模型。
        - 核心目标: 即使静态分很低，只要动态（尤其是加速度）足够强，也能产生高分，真正捕捉“拐点”。
        """
        # 引入新的权重体系和加法融合模型
        # 从配置中获取新的加法模型权重
        p_conf = get_params_block(self.strategy, 'foundation_ultimate_params', {})
        p_meta = get_param_value(p_conf.get('relational_meta_analysis_params'), {})
        # 新的权重体系，直接作用于最终分数，而非杠杆
        w_state = get_param_value(p_meta.get('state_weight'), 0.3)
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.3)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.4) # 赋予加速度最高权重
        # 核心参数
        norm_window = 55
        meta_window = 5
        bipolar_sensitivity = 1.0
        # 第一维度：状态分 (State Score) - 范围 [0, 1]
        state_score = snapshot_score.clip(0, 1)
        # 第二维度：速度分 (Velocity Score) - 范围 [-1, 1]
        relationship_trend = snapshot_score.diff(meta_window).fillna(0)
        velocity_score = normalize_to_bipolar(
            series=relationship_trend, target_index=df.index,
            window=norm_window, sensitivity=bipolar_sensitivity
        )
        # 第三维度：加速度分 (Acceleration Score) - 范围 [-1, 1]
        relationship_accel = relationship_trend.diff(meta_window).fillna(0)
        acceleration_score = normalize_to_bipolar(
            series=relationship_accel, target_index=df.index,
            window=norm_window, sensitivity=bipolar_sensitivity
        )
        # 终极融合：从乘法调制升级为加法赋权
        # 旧的乘法模型: final_score = (state_score * (1 + (velocity_score * w_velocity) + (acceleration_score * w_acceleration))).clip(0, 1)
        # 新的加法模型:
        final_score = (
            state_score * w_state +
            velocity_score * w_velocity +
            acceleration_score * w_acceleration
        ).clip(0, 1) # clip确保分数在[0, 1]范围内
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

    def diagnose_volatility_intelligence(self, df: pd.DataFrame, ma_context_score: pd.Series) -> Dict[str, pd.Series]:
        """
        【V7.0 · 关系元分析版】波动率统一情报中心
        - 核心革命: 废除旧的、基于孤立导数的信号，重构为“状态分”和“动态分”的新范式。
        - 优化说明: 接收预计算的`ma_context_score`，避免重复计算。增加健壮性检查。
        """
        states = {}
        norm_window = 120
        
        # 检查所需指标列是否存在
        if 'BBW_21_2.0_D' not in df.columns or 'hurst_120d_D' not in df.columns:
            return {} # 如果关键指标缺失，则不生成任何信号

        # 步骤一：构建“波动率压缩”的瞬时关系快照分 (状态分)
        # 核心关系：在强势趋势背景下(ma_context高)，波动率受到压缩(BBW低)。
        compression_score_raw = normalize_score(df['BBW_21_2.0_D'], df.index, norm_window, ascending=False)
        # 直接使用传入的 ma_context_score
        compression_state_score = (compression_score_raw * ma_context_score).astype(np.float32)
        states['SCORE_VOL_COMPRESSION_STATE'] = compression_state_score

        # 步骤二：对“波动率压缩关系”进行元分析，得到动态分
        compression_dynamic_score = self._perform_foundation_relational_meta_analysis(df, compression_state_score)
        states['SCORE_VOL_COMPRESSION_DYNAMIC'] = compression_dynamic_score

        # 步骤三：构建“波动率扩张风险”的瞬时关系快照分 (状态分)
        # 核心关系：在弱势趋势背景下(ma_context低)，波动率正在扩张(BBW高)。
        expansion_score_raw = normalize_score(df['BBW_21_2.0_D'], df.index, norm_window, ascending=True)
        # 直接使用传入的 ma_context_score
        expansion_risk_state_score = (expansion_score_raw * (1 - ma_context_score)).astype(np.float32)
        states['SCORE_VOL_EXPANSION_RISK_STATE'] = expansion_risk_state_score

        # 步骤四：对“波动率扩张风险关系”进行元分析，得到动态分
        expansion_risk_dynamic_score = self._perform_foundation_relational_meta_analysis(df, expansion_risk_state_score)
        states['SCORE_VOL_EXPANSION_RISK_DYNAMIC'] = expansion_risk_dynamic_score
        
        # Hurst指数保持不变，作为独立的宏观状态判断
        hurst_score = normalize_score(df['hurst_120d_D'], df.index, norm_window)
        states['SCORE_TRENDING_REGIME'] = hurst_score

        return states

    def diagnose_classic_indicators_atomics(self, df: pd.DataFrame, ma_context_score: pd.Series) -> Dict[str, pd.Series]:
        """
        【V2.0 · 关系元分析版】经典指标原子信号诊断
        - 核心革命: 将“量价点火”信号升维，引入均线趋势上下文，并用关系元分析捕捉其动态。
        - 优化说明: 接收预计算的`ma_context_score`，避免重复计算。增加健壮性检查。
        """
        states = {}
        p = get_params_block(self.strategy, 'classic_indicator_params')
        if not get_param_value(p.get('enabled'), True): return states
        
        norm_window = 120
        
        # 检查所需指标列是否存在
        required_cols = ['close_D', 'open_D', 'SLOPE_5_volume_D', 'ACCEL_5_volume_D']
        if not all(col in df.columns for col in required_cols):
            return {} # 如果关键指标缺失，则不生成任何信号

        # --- 量价齐升点火信号 ---
        # 步骤一：构建“量价齐升点火”的瞬时关系快照分 (状态分)
        # 核心关系：在强势趋势背景下(ma_context高)，出现阳线实体，且成交量加速放大。
        candle_body_up = (df['close_D'] - df['open_D']).clip(lower=0)
        score_price_up_strength = normalize_score(candle_body_up, df.index, norm_window)
        score_vol_slope_up = normalize_score(df['SLOPE_5_volume_D'].clip(lower=0), df.index, norm_window)
        score_vol_accel_up = normalize_score(df['ACCEL_5_volume_D'].clip(lower=0), df.index, norm_window)
        score_volume_igniting = score_vol_slope_up * score_vol_accel_up
        # 直接使用传入的 ma_context_score
        ignition_state_score = (score_price_up_strength * score_volume_igniting * ma_context_score).astype(np.float32)
        states['SCORE_VOL_PRICE_IGNITION_STATE'] = ignition_state_score

        # 步骤二：对“量价点火关系”进行元分析，得到动态分
        ignition_dynamic_score = self._perform_foundation_relational_meta_analysis(df, ignition_state_score)
        states['SCORE_VOL_PRICE_IGNITION_DYNAMIC'] = ignition_dynamic_score

        # --- 放量恐慌下跌信号 ---
        # 步骤三：构建“放量恐慌下跌”的瞬时关系快照分 (状态分)
        # 核心关系：在弱势趋势背景下(ma_context低)，出现阴线实体，且成交量加速放大。
        candle_body_down = (df['open_D'] - df['close_D']).clip(lower=0)
        score_price_down_strength = normalize_score(candle_body_down, df.index, norm_window)
        # 直接使用传入的 ma_context_score
        panic_state_score = (score_price_down_strength * score_volume_igniting * (1 - ma_context_score)).astype(np.float32)
        states['SCORE_VOL_PRICE_PANIC_RISK_STATE'] = panic_state_score

        # 步骤四：对“放量恐慌关系”进行元分析，得到动态分
        panic_dynamic_score = self._perform_foundation_relational_meta_analysis(df, panic_state_score)
        states['SCORE_VOL_PRICE_PANIC_RISK_DYNAMIC'] = panic_dynamic_score

        return states










