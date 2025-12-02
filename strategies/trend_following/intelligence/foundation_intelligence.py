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
        【V10.0 · 和谐拐点版】基础情报分析总指挥
        - 核心新增: 引入终极机会信号 SCORE_FOUNDATION_HARMONY_INFLECTION，通过对顶层
                      战略态势进行二阶求导，捕捉趋势启动的拐点。
        """
        print("启动【V10.0 · 和谐拐点版】基础情报分析...") # 修改: 更新版本号和描述
        all_states = {}
        p_conf = get_params_block(self.strategy, 'foundation_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            print("基础情报引擎已在配置中禁用，跳过。")
            return {}
        axiom_constitution = self._diagnose_axiom_market_constitution(df, p_conf)
        axiom_pendulum = self._diagnose_axiom_sentiment_pendulum(df)
        axiom_tide = self._diagnose_axiom_liquidity_tide(df)
        axiom_tension = self._diagnose_axiom_market_tension(df)
        axiom_relative_strength = self._diagnose_axiom_relative_strength(df)
        all_states['SCORE_FOUNDATION_AXIOM_MARKET_CONSTITUTION'] = axiom_constitution
        all_states['SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM'] = axiom_pendulum
        all_states['SCORE_FOUNDATION_AXIOM_LIQUIDITY_TIDE'] = axiom_tide
        all_states['SCORE_FOUNDATION_AXIOM_MARKET_TENSION'] = axiom_tension
        all_states['SCORE_FOUNDATION_AXIOM_RELATIVE_STRENGTH'] = axiom_relative_strength
        strategic_posture = self._synthesize_strategic_posture(
            p_conf,
            axiom_constitution,
            axiom_relative_strength,
            axiom_tide,
            axiom_pendulum,
            axiom_tension
        )
        all_states['SCORE_FOUNDATION_STRATEGIC_POSTURE'] = strategic_posture
        # 新增: 调用和谐拐点诊断方法
        harmony_inflection = self._diagnose_harmony_inflection(p_conf, strategic_posture)
        all_states['SCORE_FOUNDATION_HARMONY_INFLECTION'] = harmony_inflection
        context_trend_confirmed = self._diagnose_context_trend_confirmed(df)
        all_states.update(context_trend_confirmed)
        # 修改: 更新日志输出
        print(f"【V10.0 · 和谐拐点版】分析完成，生成 {len(all_states)} 个基础信号 (含1个顶层及1个拐点信号)。")
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
        【V3.0 · 韧性诊断版】基础公理一：诊断“市场体质”
        - 核心逻辑: 融合均线结构(骨架)、换手健康度(新陈代谢)、趋势信念(意志力)以及新增的“下跌承接力(免疫韧性)”。
        - A股特性: 健康的上涨不仅结构稳固、换手温和，更应具备强大的下跌抵抗能力。此升级旨在识别这种“抗揍”的健康体质。
        """
        print("    -> [基础层] 正在诊断“市场体质”公理 (V3.0 · 韧性诊断版)...") # 修改: 更新版本号和描述
        # 修改: 新增 dip_absorption_power_D 作为韧性指标
        required_signals = [
            'MACDh_13_34_8_D', 'SLOPE_5_DMA_D', 'turnover_rate_f_D', 'trend_alignment_index_D',
            'dip_absorption_power_D'
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
        # 1. 均线结构分 (骨架) - 逻辑不变
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
        # 2. 换手率健康度 (新陈代谢) - 逻辑不变
        turnover_rate = self._get_safe_series(df, 'turnover_rate_f_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        turnover_health_score = get_adaptive_mtf_normalized_score(turnover_rate, df_index, ascending=False, tf_weights=default_weights)
        # 3. 趋势信念 (意志力) - 逻辑不变
        trend_conviction = self._get_safe_series(df, 'trend_alignment_index_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        conviction_score = get_adaptive_mtf_normalized_score(trend_conviction, df_index, ascending=True, tf_weights=default_weights)
        # 新增: 4. 免疫韧性 (下跌承接力)
        resilience = self._get_safe_series(df, 'dip_absorption_power_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        resilience_score = get_adaptive_mtf_normalized_score(resilience, df_index, ascending=True, tf_weights=default_weights)
        # 5. 融合 - 修改: 在健康度调节器中加入韧性评分
        health_modulator = (turnover_health_score * 0.4 + conviction_score * 0.3 + resilience_score * 0.3).clip(0, 1)
        constitution_score = base_trend_score.copy()
        bullish_mask = base_trend_score > 0
        constitution_score[bullish_mask] = (base_trend_score[bullish_mask] * health_modulator[bullish_mask]).pow(0.5)
        # 修改: 更新调试探针
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
                print(f"       - resilience_score (新增): {resilience_score.loc[probe_date_for_loop]:.4f} (原始值: {resilience.loc[probe_date_for_loop]:.2f})")
                print(f"       - health_modulator: {health_modulator.loc[probe_date_for_loop]:.4f}")
                print(f"       - final_constitution_score: {constitution_score.loc[probe_date_for_loop]:.4f}")
        return constitution_score.clip(-1, 1).astype(np.float32)

    def _diagnose_axiom_sentiment_pendulum(self, df: pd.DataFrame) -> pd.Series:
        """
        【V3.0 · 诡道甄别版】基础公理二：诊断“市场情绪钟摆”
        - 核心逻辑: 融合RSI、散户恐慌/FOMO指数，并引入“博弈欺骗指数”作为核心调节器。
        - A股特性: 情绪常常是陷阱。此升级旨在利用欺骗指数来甄别情绪的真实性，过滤主力诱多或诱空行为。
        """
        print("    -> [基础层] 正在诊断“情绪钟摆”公理 (V3.0 · 诡道甄别版)...")
        required_signals = ['RSI_13_D', 'retail_panic_surrender_index_D', 'retail_fomo_premium_index_D', 'deception_index_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_sentiment_pendulum"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        rsi = self._get_safe_series(df, 'RSI_13_D', 50.0, method_name="_diagnose_axiom_sentiment_pendulum")
        rsi_score = get_adaptive_mtf_normalized_bipolar_score(rsi - 50.0, df_index, default_weights, sensitivity=10.0)
        panic_index = self._get_safe_series(df, 'retail_panic_surrender_index_D', 0.0, method_name="_diagnose_axiom_sentiment_pendulum")
        panic_score = get_adaptive_mtf_normalized_score(panic_index, df_index, ascending=True, tf_weights=default_weights)
        fomo_index = self._get_safe_series(df, 'retail_fomo_premium_index_D', 0.0, method_name="_diagnose_axiom_sentiment_pendulum")
        fomo_score = get_adaptive_mtf_normalized_score(fomo_index, df_index, ascending=True, tf_weights=default_weights)
        base_pendulum_score = (rsi_score + (fomo_score * 0.5) - (panic_score * 0.5)).clip(-1, 1)
        deception_index = self._get_safe_series(df, 'deception_index_D', 0.0, method_name="_diagnose_axiom_sentiment_pendulum")
        # 修改: 将诡道调节器的惩罚因子从 0.5 提升至 0.75，增强压制力
        reality_check_modulator = 1 - (base_pendulum_score * deception_index.clip(-1, 1) < 0) * np.abs(deception_index.clip(-1, 1)) * 0.75
        pendulum_score = base_pendulum_score * reality_check_modulator
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
                print(f"       - base_pendulum_score: {base_pendulum_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - deception_index (原始值): {deception_index.loc[probe_date_for_loop]:.4f}")
                # 修改: 在探针中说明惩罚因子已调整
                print(f"       - reality_check_modulator (诡道调节器 @ 惩罚因子0.75): {reality_check_modulator.loc[probe_date_for_loop]:.4f}")
                print(f"       - final_pendulum_score: {pendulum_score.loc[probe_date_for_loop]:.4f}")
        return pendulum_score.clip(-1, 1).astype(np.float32)

    def _diagnose_axiom_liquidity_tide(self, df: pd.DataFrame) -> pd.Series:
        """
        【V3.0 · 品质过滤版】基础公理三：诊断“流动性潮汐”
        - 核心逻辑: 融合CMF(方向)、成交额趋势(能量)与换手率趋势(活跃度)，并引入“对倒强度”作为品质过滤器。
        - A股特性: 成交量可以作假。此升级旨在通过惩罚对倒行为，还原真实的流动性状态。
        """
        print("    -> [基础层] 正在诊断“流动性潮汐”公理 (V3.0 · 品质过滤版)...") # 修改: 更新版本号和描述
        # 修改: 新增 wash_trade_intensity_D 作为品质过滤器
        required_signals = ['CMF_21_D', 'SLOPE_5_amount_D', 'SLOPE_5_turnover_rate_f_D', 'wash_trade_intensity_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_liquidity_tide"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        # 1. 潮汐方向 (CMF) - 逻辑不变
        cmf = self._get_safe_series(df, 'CMF_21_D', 0.0, method_name="_diagnose_axiom_liquidity_tide")
        direction_score = get_adaptive_mtf_normalized_bipolar_score(cmf, df_index, default_weights, sensitivity=0.5)
        # 2. 潮汐能量 (成交额趋势) - 逻辑不变
        amount_slope = self._get_safe_series(df, 'SLOPE_5_amount_D', 0.0, method_name="_diagnose_axiom_liquidity_tide")
        energy_score = get_adaptive_mtf_normalized_bipolar_score(amount_slope, df_index, default_weights)
        # 3. 潮汐活跃度 (换手率趋势) - 逻辑不变
        turnover_slope = self._get_safe_series(df, 'SLOPE_5_turnover_rate_f_D', 0.0, method_name="_diagnose_axiom_liquidity_tide")
        activity_score = get_adaptive_mtf_normalized_bipolar_score(turnover_slope, df_index, default_weights)
        # 4. 融合基础流动性分
        base_tide_score = (direction_score * 0.5 + energy_score * 0.3 + activity_score * 0.2)
        # 新增: 5. 流动性品质调节器 (对倒强度)
        wash_trade = self._get_safe_series(df, 'wash_trade_intensity_D', 0.0, method_name="_diagnose_axiom_liquidity_tide")
        # 对倒强度越高，品质越差，惩罚越大
        wash_trade_penalty = get_adaptive_mtf_normalized_score(wash_trade, df_index, ascending=True, tf_weights=default_weights)
        quality_modulator = (1 - wash_trade_penalty * 0.75).clip(0, 1) # 最多惩罚75%
        # 6. 最终融合
        tide_score = base_tide_score * quality_modulator
        # 修改: 更新调试探针
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
                print(f"       - base_tide_score: {base_tide_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - wash_trade_penalty (对倒惩罚): {wash_trade_penalty.loc[probe_date_for_loop]:.4f} (原始值: {wash_trade.loc[probe_date_for_loop]:.2f})")
                print(f"       - quality_modulator (品质调节器): {quality_modulator.loc[probe_date_for_loop]:.4f}")
                print(f"       - final_tide_score: {tide_score.loc[probe_date_for_loop]:.4f}")
        return tide_score.clip(-1, 1).astype(np.float32)

    def _diagnose_axiom_market_tension(self, df: pd.DataFrame) -> pd.Series:
        """
        【V3.0 · 意图方向版】基础公理四：诊断“市场张力”
        - 核心逻辑: 在融合BBW、均线压缩率等张力指标的基础上，引入“主力姿态指数”作为方向调节器。
        - A股特性: 盘整末端的方向选择，关键看主力意图。此升级旨在为“张力”赋予方向，预判突破概率。
        """
        print("    -> [基础层] 正在诊断“市场张力”公理 (V3.0 · 意图方向版)...") # 修改: 更新版本号和描述
        # 修改: 新增 main_force_posture_index_D 作为方向调节器
        required_signals = ['BBW_21_2.0_D', 'MA_POTENTIAL_COMPRESSION_RATE_D', 'MA_POTENTIAL_TENSION_INDEX_D', 'main_force_posture_index_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_market_tension"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        # 1. 波动收敛度 (BBW越小，分数越高) - 逻辑不变
        bbw = self._get_safe_series(df, 'BBW_21_2.0_D', 0.0, method_name="_diagnose_axiom_market_tension")
        squeeze_score = get_adaptive_mtf_normalized_score(bbw, df_index, ascending=False, tf_weights=default_weights)
        # 2. 均线压缩率 - 逻辑不变
        compression_rate = self._get_safe_series(df, 'MA_POTENTIAL_COMPRESSION_RATE_D', 0.0, method_name="_diagnose_axiom_market_tension")
        compression_score = get_adaptive_mtf_normalized_score(compression_rate, df_index, ascending=True, tf_weights=default_weights)
        # 3. 均线张力 - 逻辑不变
        tension_index = self._get_safe_series(df, 'MA_POTENTIAL_TENSION_INDEX_D', 0.0, method_name="_diagnose_axiom_market_tension")
        tension_score = get_adaptive_mtf_normalized_score(tension_index, df_index, ascending=True, tf_weights=default_weights)
        # 4. 融合为无方向的张力强度分
        unipolar_tension_score = (squeeze_score * 0.4 + compression_score * 0.3 + tension_score * 0.3).clip(0, 1)
        # 新增: 5. 主力意图作为方向调节器
        main_force_posture = self._get_safe_series(df, 'main_force_posture_index_D', 0.0, method_name="_diagnose_axiom_market_tension")
        directional_bias = get_adaptive_mtf_normalized_bipolar_score(main_force_posture, df_index, default_weights)
        # 6. 最终融合: 张力强度 * 意图方向
        tension_final_score = unipolar_tension_score * directional_bias
        # 修改: 更新调试探针
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df_index:
                print(f"    -> [市场张力探针] @ {probe_date_for_loop.date()}:")
                print(f"       - unipolar_tension_score (张力强度): {unipolar_tension_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - directional_bias (主力意图): {directional_bias.loc[probe_date_for_loop]:.4f} (原始值: {main_force_posture.loc[probe_date_for_loop]:.2f})")
                print(f"       - final_tension_score (最终得分): {tension_final_score.loc[probe_date_for_loop]:.4f}")
        return tension_final_score.clip(-1, 1).astype(np.float32)

    def _diagnose_axiom_relative_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        【V2.0 · 动量增强版】基础公理五：诊断“相对强度”
        - 核心逻辑: 融合个股在行业内的“强度排名(状态)”与“排名变化趋势(动量)”。
        - A股特性: 市场的焦点是动态变化的。此升级旨在捕捉从强到更强的“领涨龙头”，而非仅仅是静态的“强者”。
        """
        print("    -> [基础层] 正在诊断“相对强度”公理 (V2.0 · 动量增强版)...") # 修改: 更新版本号和描述
        # 修改: 新增 industry_rank_slope_D 用于动量评估
        required_signals = [
            'industry_strength_rank_D', 'industry_rank_slope_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_relative_strength"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        # 1. 强度状态分 (当前排名)
        industry_rank = self._get_safe_series(df, 'industry_strength_rank_D', 0.5, method_name="_diagnose_axiom_relative_strength")
        state_score = (industry_rank - 0.5) * 2
        # 新增: 2. 强度动量分 (排名变化趋势)
        rank_slope = self._get_safe_series(df, 'industry_rank_slope_D', 0.0, method_name="_diagnose_axiom_relative_strength")
        momentum_score = get_adaptive_mtf_normalized_bipolar_score(rank_slope, df_index, default_weights)
        # 3. 融合: 状态与动量加权
        relative_strength_score = (state_score * 0.6 + momentum_score * 0.4)
        # 修改: 更新调试探针
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df_index:
                print(f"    -> [相对强度探针] @ {probe_date_for_loop.date()}:")
                print(f"       - state_score (静态排名分): {state_score.loc[probe_date_for_loop]:.4f} (原始排名: {industry_rank.loc[probe_date_for_loop]:.2f})")
                print(f"       - momentum_score (排名动量分): {momentum_score.loc[probe_date_for_loop]:.4f} (原始斜率: {rank_slope.loc[probe_date_for_loop]:.4f})")
                print(f"       - final_relative_strength_score (融合后): {relative_strength_score.loc[probe_date_for_loop]:.4f}")
        return relative_strength_score.clip(-1, 1).astype(np.float32)

    def _diagnose_harmony_inflection(self, params: dict, strategic_posture: pd.Series) -> pd.Series:
        """
        【V1.0 · 新增】诊断“和谐拐点”
        - 核心逻辑: 对“战略态势”进行二阶求导，捕捉其加速改善的瞬间。
        """
        print("    -> [基础层] 正在诊断“和谐拐点”机会信号...")
        p_conf = params.get('harmony_inflection_params', {})
        velocity_period = p_conf.get('velocity_period', 3)
        acceleration_period = p_conf.get('acceleration_period', 2)
        df_index = strategic_posture.index
        # 1. 计算速度 (一阶导数)
        velocity = strategic_posture.diff(periods=1).rolling(window=velocity_period, min_periods=1).mean()
        # 2. 计算加速度 (二阶导数)
        acceleration = velocity.diff(periods=1).rolling(window=acceleration_period, min_periods=1).mean()
        # 3. 归一化
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        short_term_weights = get_param_value(p_mtf.get('short_term_weights'), {'weights': {3: 0.5, 5: 0.3, 8: 0.2}})
        velocity_norm = get_adaptive_mtf_normalized_score(velocity.fillna(0), df_index, ascending=True, tf_weights=short_term_weights)
        acceleration_norm = get_adaptive_mtf_normalized_score(acceleration.fillna(0), df_index, ascending=True, tf_weights=short_term_weights)
        # 4. 应用“双正”门控逻辑并融合
        gate = (velocity_norm > 0) & (acceleration_norm > 0)
        inflection_score = ((velocity_norm * acceleration_norm).pow(0.5) * gate).fillna(0.0)
        # 探针
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df_index:
                print(f"    -> [和谐拐点探针] @ {probe_date_for_loop.date()}:")
                print(f"       - 战略态势分: {strategic_posture.loc[probe_date_for_loop]:.4f}")
                print(f"       - 速度 (原始): {velocity.loc[probe_date_for_loop]:.4f}, (归一化): {velocity_norm.loc[probe_date_for_loop]:.4f}")
                print(f"       - 加速度 (原始): {acceleration.loc[probe_date_for_loop]:.4f}, (归一化): {acceleration_norm.loc[probe_date_for_loop]:.4f}")
                print(f"       - '双正'门控是否开启: {gate.loc[probe_date_for_loop]}")
                print(f"       - 最终和谐拐点分: {inflection_score.loc[probe_date_for_loop]:.4f}")
        return inflection_score.clip(0, 1).astype(np.float32)

    def _synthesize_strategic_posture(
        self,
        params: dict,
        constitution: pd.Series,
        relative_strength: pd.Series,
        liquidity: pd.Series,
        sentiment: pd.Series,
        tension: pd.Series
    ) -> pd.Series:
        """
        【V1.0 · 新增】顶层融合：合成“基础层战略态势”
        - 核心逻辑: 对五大基础公理进行加权融合，形成统一的顶层判断。
        """
        print("    -> [基础层] 正在合成“战略态势”顶层信号...")
        weights = params.get('strategic_posture_weights', {
            "constitution": 0.30, "relative_strength": 0.25, "liquidity": 0.20,
            "sentiment": 0.15, "tension": 0.10
        })
        w_c = weights.get("constitution", 0.30)
        w_rs = weights.get("relative_strength", 0.25)
        w_l = weights.get("liquidity", 0.20)
        w_s = weights.get("sentiment", 0.15)
        w_t = weights.get("tension", 0.10)
        strategic_posture = (
            constitution * w_c +
            relative_strength * w_rs +
            liquidity * w_l +
            sentiment * w_s +
            tension * w_t
        )
        # 探针
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            df_index = constitution.index
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df_index:
                print(f"    -> [战略态势探针] @ {probe_date_for_loop.date()}:")
                print(f"       - 体质贡献: {constitution.loc[probe_date_for_loop]:.4f} * {w_c} = {constitution.loc[probe_date_for_loop] * w_c:.4f}")
                print(f"       - 强度贡献: {relative_strength.loc[probe_date_for_loop]:.4f} * {w_rs} = {relative_strength.loc[probe_date_for_loop] * w_rs:.4f}")
                print(f"       - 流动性贡献: {liquidity.loc[probe_date_for_loop]:.4f} * {w_l} = {liquidity.loc[probe_date_for_loop] * w_l:.4f}")
                print(f"       - 情绪贡献: {sentiment.loc[probe_date_for_loop]:.4f} * {w_s} = {sentiment.loc[probe_date_for_loop] * w_s:.4f}")
                print(f"       - 张力贡献: {tension.loc[probe_date_for_loop]:.4f} * {w_t} = {tension.loc[probe_date_for_loop] * w_t:.4f}")
                print(f"       - 最终战略态势分: {strategic_posture.loc[probe_date_for_loop]:.4f}")
        return strategic_posture.clip(-1, 1).astype(np.float32)



