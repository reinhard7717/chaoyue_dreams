import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from strategies.trend_following.utils import get_params_block, get_param_value, get_adaptive_mtf_normalized_bipolar_score, get_adaptive_mtf_normalized_score, bipolar_to_exclusive_unipolar

class FoundationIntelligence:
    def __init__(self, strategy_instance):
        """
        初始化基础情报模块。
        :param strategy_instance: 策略主实例的引用，用于访问df和atomic_states。
        """
        self.strategy = strategy_instance

    def _get_safe_series(self, df: pd.DataFrame, column_name: str, default_value: Any = 0.0, method_name: str = "未知方法") -> pd.Series:
        """
        安全地从DataFrame获取Series，如果不存在则打印警告并返回默认Series。
        """
        if column_name not in df.columns:
            print(f"    -> [基础情报警告] 方法 '{method_name}' 缺少数据 '{column_name}'，使用默认值 {default_value}。")
            return pd.Series(default_value, index=df.index)
        return df[column_name]

    def _validate_required_signals(self, df: pd.DataFrame, required_signals: list, method_name: str) -> bool:
        """
        【V1.0 · 战前情报校验】内部辅助方法，用于在方法执行前验证所有必需的数据信号是否存在。
        """
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            # 调整校验信息为“基础情报校验”
            print(f"    -> [基础情报校验] 方法 '{method_name}' 启动失败：缺少核心信号 {missing_signals}。")
            return False
        return True

    def run_foundation_analysis_command(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V8.0 · 相对强度公理版】基础情报分析总指挥
        - 核心新增: 引入第五大公理——“相对强度公理”，旨在衡量股票相对于其板块和大盘的
                      强弱，为识别“真龙头”提供核心依据。
        """
        print("启动【V8.0 · 相对强度公理版】基础情报分析...")
        all_states = {}
        p_conf = get_params_block(self.strategy, 'foundation_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            print("基础情报引擎已在配置中禁用，跳过。")
            return {}
        axiom_constitution = self._diagnose_axiom_market_constitution(df, p_conf)
        axiom_pendulum = self._diagnose_axiom_sentiment_pendulum(df)
        axiom_tide = self._diagnose_axiom_liquidity_tide(df)
        axiom_tension = self._diagnose_axiom_market_tension(df)
        # [新增代码行] 调用新增的相对强度公理诊断方法
        axiom_relative_strength = self._diagnose_axiom_relative_strength(df)
        all_states['SCORE_FOUNDATION_AXIOM_MARKET_CONSTITUTION'] = axiom_constitution
        all_states['SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM'] = axiom_pendulum
        all_states['SCORE_FOUNDATION_AXIOM_LIQUIDITY_TIDE'] = axiom_tide
        all_states['SCORE_FOUNDATION_AXIOM_MARKET_TENSION'] = axiom_tension
        # [新增代码行] 将新的公理分数添加到状态字典
        all_states['SCORE_FOUNDATION_AXIOM_RELATIVE_STRENGTH'] = axiom_relative_strength
        context_trend_confirmed = self._diagnose_context_trend_confirmed(df)
        all_states.update(context_trend_confirmed)
        print(f"【V8.0 · 相对强度公理版】分析完成，生成 {len(all_states)} 个基础原子信号。")
        return all_states

    def _diagnose_context_trend_confirmed(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 · 健康度重构版】诊断内部上下文信号：趋势确认分 (CONTEXT_TREND_CONFIRMED)
        - 核心重构: 废弃了基于BIAS的、存在逻辑缺陷的“健康度”评估。
        - 核心升级: 引入“突破质量分”和“趋势活力指数”重新定义趋势健康度，使其能够
                      真正识别并奖励强势突破，而非惩罚。
        """
        print("    -> [基础层] 正在诊断“趋势确认”上下文...")
        # 更新依赖信号，用突破质量和趋势活力替换BIAS
        required_signals = [
            'ADX_14_D', 'PDI_14_D', 'NDI_14_D', 'SLOPE_5_PDI_14_D',
            'breakout_quality_score_D', 'trend_vitality_index_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_context_trend_confirmed"):
            return {}
        df_index = df.index
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        # 1. 趋势强度 (ADX) - 逻辑不变
        adx_score = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'ADX_14_D', 0.0, method_name="_diagnose_context_trend_confirmed"), df.index, ascending=True, tf_weights=default_weights)
        # 2. 趋势方向 (PDI/NDI) - 逻辑不变
        pdi_gt_ndi = (self._get_safe_series(df, 'PDI_14_D', 0, method_name="_diagnose_context_trend_confirmed") > self._get_safe_series(df, 'NDI_14_D', 0, method_name="_diagnose_context_trend_confirmed")).astype(float)
        pdi_slope = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'SLOPE_5_PDI_14_D', 0.0, method_name="_diagnose_context_trend_confirmed"), df.index, ascending=True, tf_weights=default_weights)
        direction_score = (pdi_gt_ndi * pdi_slope).pow(0.5)
        # 重新定义趋势健康度
        # 3. 趋势健康度 (质地) - 全新逻辑
        breakout_quality = self._get_safe_series(df, 'breakout_quality_score_D', 0.0, method_name="_diagnose_context_trend_confirmed")
        breakout_quality_score = get_adaptive_mtf_normalized_score(breakout_quality, df.index, ascending=True, tf_weights=default_weights)
        trend_vitality = self._get_safe_series(df, 'trend_vitality_index_D', 0.0, method_name="_diagnose_context_trend_confirmed")
        trend_vitality_score = get_adaptive_mtf_normalized_score(trend_vitality, df.index, ascending=True, tf_weights=default_weights)
        trend_health_score = (breakout_quality_score * trend_vitality_score).pow(0.5)
        # 4. 最终融合
        trend_confirmed = (adx_score * direction_score * trend_health_score).pow(1/3).fillna(0.0)
        # 更新探针内容
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df_index:
                print(f"    -> [趋势确认探针] @ {probe_date_for_loop.date()}:")
                print(f"       - adx_score (强度): {adx_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - direction_score (方向): {direction_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - trend_health_score (健康度): {trend_health_score.loc[probe_date_for_loop]:.4f} (突破质量: {breakout_quality_score.loc[probe_date_for_loop]:.4f}, 趋势活力: {trend_vitality_score.loc[probe_date_for_loop]:.4f})")
                print(f"       - final_trend_confirmed: {trend_confirmed.loc[probe_date_for_loop]:.4f}")
        return {'CONTEXT_TREND_CONFIRMED': trend_confirmed.astype(np.float32)}

    def _diagnose_axiom_market_constitution(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【V2.0 · 升维版】基础公理一：诊断“市场体质”
        - 核心逻辑: 融合均线结构、换手率健康度和趋势信念，评估市场“体质”。
        - A股特性: 健康的上涨（体质强）应是结构稳固、换手温和、信念坚定的，区别于高换手的“亢奋式”虚胖拉升。
        """
        print("    -> [基础层] 正在诊断“市场体质”公理...")
        required_signals = [
            'MACDh_13_34_8_D', 'SLOPE_5_DMA_D', 'turnover_rate_f_D', 'trend_conviction_ratio_D'
        ]
        ma_periods = [5, 13, 21, 55]
        required_signals.extend([f'EMA_{p}_D' for p in ma_periods])
        required_signals.extend([f'SLOPE_{p}_EMA_{p}_D' for p in ma_periods])
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_market_constitution"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        # 1. 均线结构分 (骨架)
        macd_h = self._get_safe_series(df, 'MACDh_13_34_8_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        macd_score = get_adaptive_mtf_normalized_bipolar_score(macd_h, df_index, default_weights)
        fusion_weights = params.get('ma_health_fusion_weights', {'alignment': 0.5, 'slope': 0.5})
        bull_alignment_scores = [(self._get_safe_series(df, f'EMA_{ma_periods[i]}_D', method_name="_diagnose_axiom_market_constitution") > self._get_safe_series(df, f'EMA_{ma_periods[i+1]}_D', method_name="_diagnose_axiom_market_constitution")).astype(float).values for i in range(len(ma_periods) - 1)]
        alignment_score = np.mean(bull_alignment_scores, axis=0) if bull_alignment_scores else np.full(len(df_index), 0.5)
        alignment_bipolar = (pd.Series(alignment_score, index=df_index) - 0.5) * 2
        slope_scores = [get_adaptive_mtf_normalized_bipolar_score(self._get_safe_series(df, f'SLOPE_{p}_EMA_{p}_D', 0.0, method_name="_diagnose_axiom_market_constitution"), df_index, default_weights).values for p in ma_periods]
        avg_slope_bipolar = pd.Series(np.mean(slope_scores, axis=0), index=df_index)
        dma_slope = self._get_safe_series(df, 'SLOPE_5_DMA_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        dma_slope_score = get_adaptive_mtf_normalized_bipolar_score(dma_slope, df_index, default_weights)
        structure_score = (alignment_bipolar * fusion_weights.get('alignment', 0.5) + avg_slope_bipolar * fusion_weights.get('slope', 0.5)).clip(-1, 1)
        base_trend_score = (macd_score * 0.3 + structure_score * 0.5 + dma_slope_score * 0.2).clip(-1, 1)
        # 2. 换手率健康度 (新陈代谢) - 换手率越低越健康，故 ascending=False
        turnover_rate = self._get_safe_series(df, 'turnover_rate_f_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        turnover_health_score = get_adaptive_mtf_normalized_score(turnover_rate, df_index, ascending=False, tf_weights=default_weights)
        # 3. 趋势信念 (意志力)
        trend_conviction = self._get_safe_series(df, 'trend_conviction_ratio_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        conviction_score = get_adaptive_mtf_normalized_score(trend_conviction, df_index, ascending=True, tf_weights=default_weights)
        # 4. 融合
        # 当趋势向上时，用健康度和信念进行加权；当趋势向下时，主要由趋势本身决定
        health_modulator = (turnover_health_score * 0.6 + conviction_score * 0.4).clip(0, 1)
        constitution_score = base_trend_score.copy()
        bullish_mask = base_trend_score > 0
        constitution_score[bullish_mask] = (base_trend_score[bullish_mask] * health_modulator[bullish_mask]).pow(0.5)
        # [新增] 调试探针
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df_index:
                print(f"    -> [市场体质探针] @ {probe_date_for_loop.date()}:")
                print(f"       - base_trend_score: {base_trend_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - turnover_health_score: {turnover_health_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - conviction_score: {conviction_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - health_modulator: {health_modulator.loc[probe_date_for_loop]:.4f}")
                print(f"       - final_constitution_score: {constitution_score.loc[probe_date_for_loop]:.4f}")
        return constitution_score.clip(-1, 1).astype(np.float32)

    def _diagnose_axiom_sentiment_pendulum(self, df: pd.DataFrame) -> pd.Series:
        """
        【V2.0 · 升维版】基础公理二：诊断“市场情绪钟摆”
        - 核心逻辑: 融合RSI、散户恐慌指数与FOMO指数，诊断市场情绪的两极状态。
        - A股特性: 直接利用刻画散户恐慌与FOMO的指标，比单纯的RSI超买超卖更具实战意义。
        """
        print("    -> [基础层] 正在诊断“情绪钟摆”公理...")
        required_signals = ['RSI_13_D', 'retail_panic_surrender_index_D', 'retail_fomo_premium_index_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_sentiment_pendulum"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        # 1. RSI作为基础位置
        rsi = self._get_safe_series(df, 'RSI_13_D', 50.0, method_name="_diagnose_axiom_sentiment_pendulum")
        rsi_score = get_adaptive_mtf_normalized_bipolar_score(rsi - 50.0, df_index, default_weights, sensitivity=10.0)
        # 2. 恐慌与FOMO作为两极的加强器
        panic_index = self._get_safe_series(df, 'retail_panic_surrender_index_D', 0.0, method_name="_diagnose_axiom_sentiment_pendulum")
        panic_score = get_adaptive_mtf_normalized_score(panic_index, df_index, ascending=True, tf_weights=default_weights)
        fomo_index = self._get_safe_series(df, 'retail_fomo_premium_index_D', 0.0, method_name="_diagnose_axiom_sentiment_pendulum")
        fomo_score = get_adaptive_mtf_normalized_score(fomo_index, df_index, ascending=True, tf_weights=default_weights)
        # 3. 融合
        pendulum_score = rsi_score + (fomo_score * 0.5) - (panic_score * 0.5)
        # [新增] 调试探针
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df_index:
                print(f"    -> [情绪钟摆探针] @ {probe_date_for_loop.date()}:")
                print(f"       - rsi_score: {rsi_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - panic_score: {panic_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - fomo_score: {fomo_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - final_pendulum_score: {pendulum_score.loc[probe_date_for_loop]:.4f}")
        return pendulum_score.clip(-1, 1).astype(np.float32)

    def _diagnose_axiom_liquidity_tide(self, df: pd.DataFrame) -> pd.Series:
        """
        【V2.0 · 升维版】基础公理三：诊断“流动性潮汐”
        - 核心逻辑: 融合CMF(方向)、成交额趋势(能量)与换手率趋势(活跃度)。
        - A股特性: 真实的上涨需要“水涨船高”，即资金流入需要伴随市场关注度和活跃度的同步提升。
        """
        print("    -> [基础层] 正在诊断“流动性潮汐”公理...")
        required_signals = ['CMF_21_D', 'SLOPE_5_amount_D', 'SLOPE_5_turnover_rate_f_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_liquidity_tide"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        # 1. 潮汐方向 (CMF)
        cmf = self._get_safe_series(df, 'CMF_21_D', 0.0, method_name="_diagnose_axiom_liquidity_tide")
        direction_score = get_adaptive_mtf_normalized_bipolar_score(cmf, df_index, default_weights, sensitivity=0.5)
        # 2. 潮汐能量 (成交额趋势)
        amount_slope = self._get_safe_series(df, 'SLOPE_5_amount_D', 0.0, method_name="_diagnose_axiom_liquidity_tide")
        energy_score = get_adaptive_mtf_normalized_bipolar_score(amount_slope, df_index, default_weights)
        # 3. 潮汐活跃度 (换手率趋势)
        turnover_slope = self._get_safe_series(df, 'SLOPE_5_turnover_rate_f_D', 0.0, method_name="_diagnose_axiom_liquidity_tide")
        activity_score = get_adaptive_mtf_normalized_bipolar_score(turnover_slope, df_index, default_weights)
        # 4. 融合
        tide_score = (direction_score * 0.5 + energy_score * 0.3 + activity_score * 0.2)
        # [新增] 调试探针
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df_index:
                print(f"    -> [流动性潮汐探针] @ {probe_date_for_loop.date()}:")
                print(f"       - direction_score (CMF): {direction_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - energy_score (Amount Slope): {energy_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - activity_score (Turnover Slope): {activity_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - final_tide_score: {tide_score.loc[probe_date_for_loop]:.4f}")
        return tide_score.clip(-1, 1).astype(np.float32)

    def _diagnose_axiom_market_tension(self, df: pd.DataFrame) -> pd.Series:
        """
        【V2.0 · 升维版】基础公理四：诊断“市场张力”
        - 核心逻辑: 融合BBW(收敛度)、均线压缩率和均线张力指数，评估市场能量的压缩状态。
        - A股特性: 暴风雨前的宁静。高张力状态（正分）往往是下一轮大级别行情的前兆。
        """
        print("    -> [基础层] 正在诊断“市场张力”公理...")
        required_signals = ['BBW_21_2.0_D', 'MA_POTENTIAL_COMPRESSION_RATE_D', 'MA_POTENTIAL_TENSION_INDEX_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_market_tension"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        # 1. 波动收敛度 (BBW越小，分数越高)
        bbw = self._get_safe_series(df, 'BBW_21_2.0_D', 0.0, method_name="_diagnose_axiom_market_tension")
        squeeze_score = get_adaptive_mtf_normalized_score(bbw, df_index, ascending=False, tf_weights=default_weights)
        # 2. 均线压缩率
        compression_rate = self._get_safe_series(df, 'MA_POTENTIAL_COMPRESSION_RATE_D', 0.0, method_name="_diagnose_axiom_market_tension")
        compression_score = get_adaptive_mtf_normalized_score(compression_rate, df_index, ascending=True, tf_weights=default_weights)
        # 3. 均线张力
        tension_index = self._get_safe_series(df, 'MA_POTENTIAL_TENSION_INDEX_D', 0.0, method_name="_diagnose_axiom_market_tension")
        tension_score = get_adaptive_mtf_normalized_score(tension_index, df_index, ascending=True, tf_weights=default_weights)
        # 4. 融合 (高张力是正分)
        tension_final_score = (squeeze_score * 0.4 + compression_score * 0.3 + tension_score * 0.3)
        # [新增] 调试探针
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df_index:
                print(f"    -> [市场张力探针] @ {probe_date_for_loop.date()}:")
                print(f"       - squeeze_score (BBW): {squeeze_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - compression_score: {compression_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - tension_score: {tension_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - final_tension_score: {tension_final_score.loc[probe_date_for_loop]:.4f}")
        return tension_final_score.clip(0, 1).astype(np.float32)

    def _diagnose_axiom_relative_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        【V1.0 · 新增】基础公理五：诊断“相对强度”
        - 核心逻辑: 融合价格和资金流的相对强度，评估股票在“空间”维度上的领涨或跟跌属性。
        - A股特性: “强者恒强”。此模型旨在第一时间锁定板块和市场中的“领航舰”。
        """
        print("    -> [基础层] 正在诊断“相对强度”公理...")
        required_signals = [
            'pct_change_D', 'industry_pct_change_D', 'index_pct_change_D',
            'main_force_net_flow_calibrated_D', 'industry_main_force_net_flow_avg_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_relative_strength"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        # 1. 相对价格强度 (Price RS)
        stock_pct = self._get_safe_series(df, 'pct_change_D', 0.0)
        industry_pct = self._get_safe_series(df, 'industry_pct_change_D', 0.0)
        index_pct = self._get_safe_series(df, 'index_pct_change_D', 0.0)
        # 计算超额收益，并对累积超额收益进行归一化
        excess_return_vs_industry = (stock_pct - industry_pct).rolling(window=21).sum()
        excess_return_vs_index = (stock_pct - index_pct).rolling(window=21).sum()
        price_rs_score = (
            get_adaptive_mtf_normalized_bipolar_score(excess_return_vs_industry, df_index, default_weights) * 0.6 +
            get_adaptive_mtf_normalized_bipolar_score(excess_return_vs_index, df_index, default_weights) * 0.4
        )
        # 2. 相对资金强度 (Flow RS)
        stock_flow = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0)
        industry_flow = self._get_safe_series(df, 'industry_main_force_net_flow_avg_D', 0.0)
        excess_flow = (stock_flow - industry_flow).rolling(window=21).sum()
        flow_rs_score = get_adaptive_mtf_normalized_bipolar_score(excess_flow, df_index, default_weights)
        # 3. 融合
        relative_strength_score = (price_rs_score * 0.7 + flow_rs_score * 0.3)
        # [新增] 调试探针
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df_index:
                print(f"    -> [相对强度探针] @ {probe_date_for_loop.date()}:")
                print(f"       - price_rs_score (价格): {price_rs_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - flow_rs_score (资金): {flow_rs_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - final_relative_strength_score: {relative_strength_score.loc[probe_date_for_loop]:.4f}")
        return relative_strength_score.clip(-1, 1).astype(np.float32)




