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
        【V11.0 · 环境共振版】基础情报分析总指挥
        - 核心新增: 引入“环境共振调节器”，将个股信号与市场/板块/主题环境耦合，
                      实现对顶层信号的宏观上下文校准。
        """
        print("启动【V11.0 · 环境共振版】基础情报分析...")
        all_states = {}
        p_conf = get_params_block(self.strategy, 'foundation_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            print("基础情报引擎已在配置中禁用，跳过。")
            return {}
        environmental_modulator = self._calculate_environmental_modulator(df, p_conf)
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
            axiom_tension, # 修正变量名，从 tension -> axiom_tension
            environmental_modulator
        )
        all_states['SCORE_FOUNDATION_STRATEGIC_POSTURE'] = strategic_posture
        harmony_inflection = self._diagnose_harmony_inflection(p_conf, strategic_posture, environmental_modulator)
        all_states['SCORE_FOUNDATION_HARMONY_INFLECTION'] = harmony_inflection
        context_trend_confirmed = self._diagnose_context_trend_confirmed(df)
        all_states.update(context_trend_confirmed)
        print(f"【V11.0 · 环境共振版】分析完成，生成 {len(all_states)} 个基础信号 (含1个顶层及1个拐点信号)。")
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
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
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
        return {'CONTEXT_TREND_CONFIRMED': trend_confirmed.astype(np.float32)}

    def _diagnose_axiom_market_constitution(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【V3.0 · 韧性诊断版】基础公理一：诊断“市场体质”
        - 核心逻辑: 融合均线结构(骨架)、换手健康度(新陈代谢)、趋势信念(意志力)以及新增的“下跌承接力(免疫韧性)”。
        - A股特性: 健康的上涨不仅结构稳固、换手温和，更应具备强大的下跌抵抗能力。此升级旨在识别这种“抗揍”的健康体质。
        """
        print("    -> [基础层] 正在诊断“市场体质”公理 (V3.0 · 韧性诊断版)...")
        # 新增 dip_absorption_power_D 作为韧性指标
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
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
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
        # 5. 融合 - 在健康度调节器中加入韧性评分
        health_modulator = (turnover_health_score * 0.4 + conviction_score * 0.3 + resilience_score * 0.3).clip(0, 1)
        constitution_score = base_trend_score.copy()
        bullish_mask = base_trend_score > 0
        constitution_score[bullish_mask] = (base_trend_score[bullish_mask] * health_modulator[bullish_mask]).pow(0.5)
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
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        rsi = self._get_safe_series(df, 'RSI_13_D', 50.0, method_name="_diagnose_axiom_sentiment_pendulum")
        rsi_score = get_adaptive_mtf_normalized_bipolar_score(rsi - 50.0, df_index, default_weights, sensitivity=10.0)
        panic_index = self._get_safe_series(df, 'retail_panic_surrender_index_D', 0.0, method_name="_diagnose_axiom_sentiment_pendulum")
        panic_score = get_adaptive_mtf_normalized_score(panic_index, df_index, ascending=True, tf_weights=default_weights)
        fomo_index = self._get_safe_series(df, 'retail_fomo_premium_index_D', 0.0, method_name="_diagnose_axiom_sentiment_pendulum")
        fomo_score = get_adaptive_mtf_normalized_score(fomo_index, df_index, ascending=True, tf_weights=default_weights)
        base_pendulum_score = (rsi_score + (fomo_score * 0.5) - (panic_score * 0.5)).clip(-1, 1)
        deception_index = self._get_safe_series(df, 'deception_index_D', 0.0, method_name="_diagnose_axiom_sentiment_pendulum")
        # 将诡道调节器的惩罚因子从 0.5 提升至 0.75，增强压制力
        reality_check_modulator = 1 - (base_pendulum_score * deception_index.clip(-1, 1) < 0) * np.abs(deception_index.clip(-1, 1)) * 0.75
        pendulum_score = base_pendulum_score * reality_check_modulator
        return pendulum_score.clip(-1, 1).astype(np.float32)

    def _diagnose_axiom_liquidity_tide(self, df: pd.DataFrame) -> pd.Series:
        """
        【V3.0 · 品质过滤版】基础公理三：诊断“流动性潮汐”
        - 核心逻辑: 融合CMF(方向)、成交额趋势(能量)与换手率趋势(活跃度)，并引入“对倒强度”作为品质过滤器。
        - A股特性: 成交量可以作假。此升级旨在通过惩罚对倒行为，还原真实的流动性状态。
        """
        print("    -> [基础层] 正在诊断“流动性潮汐”公理 (V3.0 · 品质过滤版)...")
        # 新增 wash_trade_intensity_D 作为品质过滤器
        required_signals = ['CMF_21_D', 'SLOPE_5_amount_D', 'SLOPE_5_turnover_rate_f_D', 'wash_trade_intensity_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_liquidity_tide"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
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
        return tide_score.clip(-1, 1).astype(np.float32)

    def _diagnose_axiom_market_tension(self, df: pd.DataFrame) -> pd.Series:
        """
        【V3.0 · 意图方向版】基础公理四：诊断“市场张力”
        - 核心逻辑: 在融合BBW、均线压缩率等张力指标的基础上，引入“主力姿态指数”作为方向调节器。
        - A股特性: 盘整末端的方向选择，关键看主力意图。此升级旨在为“张力”赋予方向，预判突破概率。
        """
        print("    -> [基础层] 正在诊断“市场张力”公理 (V3.0 · 意图方向版)...")
        # 新增 main_force_posture_index_D 作为方向调节器
        required_signals = ['BBW_21_2.0_D', 'MA_POTENTIAL_COMPRESSION_RATE_D', 'MA_POTENTIAL_TENSION_INDEX_D', 'main_force_posture_index_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_market_tension"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
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
        return tension_final_score.clip(-1, 1).astype(np.float32)

    def _diagnose_axiom_relative_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        【V3.0 · 深度洞察版】基础公理五：诊断“相对强度”
        - 核心逻辑: 融合个股在行业内的“强度排名(状态)”与“排名变化趋势(动量)”，并引入排名稳定性与加速度，
                      通过非线性融合和协同奖励，更精准捕捉“从强到更强”的领涨龙头。
        - A股特性: 市场的焦点是动态变化的。此升级旨在捕捉从强到更强的“领涨龙头”，而非仅仅是静态的“强者”。
        """
        print("    -> [基础层] 正在诊断“相对强度”公理 (V3.0 · 深度洞察版)...") # 修改: 更新版本描述
        # 获取相对强度公理的专属参数
        p_conf_rs = get_params_block(self.strategy, 'foundation_ultimate_params', {}).get('relative_strength_params', {}) # 新增: 获取相对强度专属参数
        # 获取行为动态参数中的MTF归一化默认权重
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        # 获取探针启用状态
        probe_enabled = get_param_value(p_conf_rs.get('enable_probe'), False) # 新增: 获取探针启用状态

        df_index = df.index

        # 获取参数
        state_weights = p_conf_rs.get('state_weights', {'current_rank': 0.7, 'rank_stability': 0.3}) # 新增: 状态分融合权重
        momentum_weights = p_conf_rs.get('momentum_weights', {'rank_slope': 0.6, 'rank_acceleration': 0.4}) # 新增: 动量分融合权重
        fusion_weights = p_conf_rs.get('fusion_weights', {'enhanced_state': 0.6, 'enhanced_momentum': 0.4}) # 新增: 最终融合权重
        synergy_bonus_factor = get_param_value(p_conf_rs.get('synergy_bonus_factor'), 0.2) # 新增: 协同奖励因子
        synergy_threshold = get_param_value(p_conf_rs.get('synergy_threshold'), 0.2) # 新增: 协同阈值
        rank_stability_window = get_param_value(p_conf_rs.get('rank_stability_window'), 5) # 新增: 排名稳定性窗口
        rank_acceleration_window = get_param_value(p_conf_rs.get('rank_acceleration_window'), 5) # 新增: 排名加速度窗口

        # 1. 校验所需信号 (动态构建信号名称)
        required_signals = [ # 修改: 动态构建所需信号列表
            'industry_strength_rank_D',
            f'SLOPE_{rank_stability_window}_industry_strength_rank_D',
            f'ACCEL_{rank_acceleration_window}_industry_strength_rank_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_relative_strength"):
            return pd.Series(0.0, index=df.index)

        # --- 原始数据获取与探针输出 ---
        industry_rank_raw = self._get_safe_series(df, 'industry_strength_rank_D', 0.5, method_name="_diagnose_axiom_relative_strength")
        # 修改: 使用配置中的窗口期获取斜率和加速度
        industry_rank_slope_raw = self._get_safe_series(df, f'SLOPE_{rank_stability_window}_industry_strength_rank_D', 0.0, method_name="_diagnose_axiom_relative_strength")
        industry_rank_accel_raw = self._get_safe_series(df, f'ACCEL_{rank_acceleration_window}_industry_strength_rank_D', 0.0, method_name="_diagnose_axiom_relative_strength")

        if probe_enabled: # 新增: 探针输出原始数据
            print(f"    -> [探针] 原始行业强度排名 (industry_rank_raw) 尾部: {industry_rank_raw.tail().to_dict()}")
            print(f"    -> [探针] 原始行业排名斜率 (industry_rank_slope_raw) 尾部: {industry_rank_slope_raw.tail().to_dict()}")
            print(f"    -> [探针] 原始行业排名加速度 (industry_rank_accel_raw) 尾部: {industry_rank_accel_raw.tail().to_dict()}")

        # --- 1. 强度状态分 (当前排名与稳定性) ---
        # 1.1 当前排名归一化到 [-1, 1]
        state_score_normalized = (industry_rank_raw - 0.5) * 2 # 逻辑不变
        # 1.2 排名稳定性分 (斜率绝对值越小，稳定性越高，分数越高)
        # 使用 rank_stability_window 对应的斜率来衡量稳定性
        rank_stability_score = get_adaptive_mtf_normalized_score( # 新增: 排名稳定性分计算
            industry_rank_slope_raw.abs(), df_index, default_weights, ascending=False
        )
        # 1.3 融合强化状态分
        enhanced_state_score = ( # 新增: 强化状态分融合
            state_score_normalized * state_weights.get('current_rank', 0.7) +
            rank_stability_score * state_weights.get('rank_stability', 0.3)
        ).clip(-1, 1)

        if probe_enabled: # 新增: 探针输出强化状态分中间结果
            print(f"    -> [探针] 状态分 (state_score_normalized) 尾部: {state_score_normalized.tail().to_dict()}")
            print(f"    -> [探针] 排名稳定性分 (rank_stability_score) 尾部: {rank_stability_score.tail().to_dict()}")
            print(f"    -> [探针] 强化状态分 (enhanced_state_score) 尾部: {enhanced_state_score.tail().to_dict()}")

        # --- 2. 强度动量分 (排名斜率与加速度) ---
        # 2.1 排名斜率归一化到 [-1, 1]
        momentum_score_normalized = get_adaptive_mtf_normalized_bipolar_score( # 逻辑不变
            industry_rank_slope_raw, df_index, default_weights
        )
        # 2.2 排名加速度分归一化到 [-1, 1]
        # 使用 rank_acceleration_window 对应的加速度
        rank_acceleration_score = get_adaptive_mtf_normalized_bipolar_score( # 新增: 排名加速度分计算
            industry_rank_accel_raw, df_index, default_weights
        )
        # 2.3 融合强化动量分
        enhanced_momentum_score = ( # 新增: 强化动量分融合
            momentum_score_normalized * momentum_weights.get('rank_slope', 0.6) +
            rank_acceleration_score * momentum_weights.get('rank_acceleration', 0.4)
        ).clip(-1, 1)

        if probe_enabled: # 新增: 探针输出强化动量分中间结果
            print(f"    -> [探针] 动量分 (momentum_score_normalized) 尾部: {momentum_score_normalized.tail().to_dict()}")
            print(f"    -> [探针] 排名加速度分 (rank_acceleration_score) 尾部: {rank_acceleration_score.tail().to_dict()}")
            print(f"    -> [探针] 强化动量分 (enhanced_momentum_score) 尾部: {enhanced_momentum_score.tail().to_dict()}")

        # --- 3. 最终融合: 强化状态与动量，并引入协同奖励 ---
        relative_strength_score = ( # 修改: 使用强化后的状态分和动量分进行融合
            enhanced_state_score * fusion_weights.get('enhanced_state', 0.6) +
            enhanced_momentum_score * fusion_weights.get('enhanced_momentum', 0.4)
        )

        # 协同奖励: 当强化状态分和强化动量分都为正且超过阈值时，给予额外奖励
        synergy_condition = (enhanced_state_score > synergy_threshold) & (enhanced_momentum_score > synergy_threshold) # 新增: 协同条件判断
        if synergy_condition.any(): # 新增: 协同奖励逻辑
            # 奖励因子基于两者平均强度，并乘以配置的奖励系数
            synergy_boost = (enhanced_state_score[synergy_condition] + enhanced_momentum_score[synergy_condition]) / 2 * synergy_bonus_factor
            relative_strength_score.loc[synergy_condition] = (
                relative_strength_score.loc[synergy_condition] * (1 + synergy_boost)
            )

        # 确保最终分数在 [-1, 1] 范围内
        relative_strength_score = relative_strength_score.clip(-1, 1).astype(np.float32)

        if probe_enabled: # 新增: 探针输出最终结果
            print(f"    -> [探针] 最终相对强度分 (relative_strength_score) 尾部: {relative_strength_score.tail().to_dict()}")
        print("    -> [基础层] “相对强度”公理诊断完成。")
        return relative_strength_score

    def _diagnose_harmony_inflection(self, params: dict, strategic_posture: pd.Series, modulator: pd.Series) -> pd.Series: # 接收调节器
        """
        【V2.0 · 环境共振版】诊断“和谐拐点”
        - 核心逻辑: 对“战略态势”进行二阶求导，并应用环境共振调节器。
        """
        print("    -> [基础层] 正在诊断“和谐拐点”机会信号...")
        p_conf = params.get('harmony_inflection_params', {})
        velocity_period = p_conf.get('velocity_period', 3)
        acceleration_period = p_conf.get('acceleration_period', 2)
        df_index = strategic_posture.index
        velocity = strategic_posture.diff(periods=1).rolling(window=velocity_period, min_periods=1).mean()
        acceleration = velocity.diff(periods=1).rolling(window=acceleration_period, min_periods=1).mean()
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        short_term_weights = get_param_value(p_mtf.get('short_term'), {'3': 0.5, '5': 0.3, '8': 0.2})
        velocity_norm = get_adaptive_mtf_normalized_score(velocity.fillna(0), df_index, ascending=True, tf_weights=short_term_weights)
        acceleration_norm = get_adaptive_mtf_normalized_score(acceleration.fillna(0), df_index, ascending=True, tf_weights=short_term_weights)
        gate = (velocity_norm > 0) & (acceleration_norm > 0)
        raw_inflection_score = ((velocity_norm * acceleration_norm).pow(0.5) * gate).fillna(0.0) # 变量重命名为 raw_
        # 新增: 应用环境调节器
        inflection_score = raw_inflection_score * modulator
        return inflection_score.clip(0, 1).astype(np.float32)

    def _calculate_environmental_modulator(self, df: pd.DataFrame, params: dict) -> pd.Series: # 增加df参数
        """
        【V1.0 · 新增】计算“环境共振调节器”
        - 核心逻辑: 融合市场趋势代理、板块强度、主题热度，生成一个[0.75, 1.25]区间的调节器。
        """
        print("    -> [基础层] 正在计算“环境共振调节器”...")
        p_conf = params.get('environmental_modulator_params', {})
        if not p_conf.get('enabled', True):
            return pd.Series(1.0, index=df.index)
        required_signals = ['SLOPE_55_close_D', 'industry_strength_rank_D', 'THEME_HOTNESS_SCORE_D']
        if not self._validate_required_signals(df, required_signals, "_calculate_environmental_modulator"):
            return pd.Series(1.0, index=df.index)
        df_index = df.index
        weights = p_conf.get('weights', {})
        w_mkt = weights.get('market_proxy', 0.3)
        w_sec = weights.get('sector_strength', 0.4)
        w_thm = weights.get('theme_hotness', 0.3)
        bonus_factor = p_conf.get('bonus_factor', 0.25)
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        market_proxy_raw = self._get_safe_series(df, 'SLOPE_55_close_D', 0.0, "_calculate_environmental_modulator")
        market_proxy_score = get_adaptive_mtf_normalized_bipolar_score(market_proxy_raw, df_index, default_weights)
        sector_strength_raw = self._get_safe_series(df, 'industry_strength_rank_D', 0.5, "_calculate_environmental_modulator")
        sector_strength_score = (sector_strength_raw - 0.5) * 2
        theme_hotness_raw = self._get_safe_series(df, 'THEME_HOTNESS_SCORE_D', 0.0, "_calculate_environmental_modulator")
        theme_hotness_score = get_adaptive_mtf_normalized_score(theme_hotness_raw, df_index, ascending=True, tf_weights=default_weights)
        env_score = (market_proxy_score * w_mkt + sector_strength_score * w_sec + theme_hotness_score * w_thm).clip(-1, 1)
        modulator = 1.0 + (env_score * bonus_factor)
        return modulator.astype(np.float32)

    def _synthesize_strategic_posture(
        self,
        params: dict,
        constitution: pd.Series,
        relative_strength: pd.Series,
        liquidity: pd.Series,
        sentiment: pd.Series,
        tension: pd.Series,
        modulator: pd.Series # 接收调节器
    ) -> pd.Series:
        """
        【V2.0 · 环境共振版】顶层融合：合成“基础层战略态势”
        - 核心逻辑: 对五大公理进行加权融合，并应用环境共振调节器。
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
        raw_strategic_posture = ( # 变量重命名为 raw_
            constitution * w_c +
            relative_strength * w_rs +
            liquidity * w_l +
            sentiment * w_s +
            tension * w_t
        )
        # 新增: 应用环境调节器
        strategic_posture = raw_strategic_posture * modulator
        return strategic_posture.clip(-1, 1).astype(np.float32)



