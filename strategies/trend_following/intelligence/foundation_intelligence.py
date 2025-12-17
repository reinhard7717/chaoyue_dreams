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
        【V5.1 · 数据鲁棒性强化版】基础公理一：诊断“市场体质”
        - 核心逻辑: 融合均线骨架质量(MSQ)、筹码新陈代谢健康度(TH)、市场信念强度(MCS)、免疫韧性与抗压能力(IR)，
                      以及价格行为品质(PAQ)和筹码博弈深度(CGD)六大维度。
                      通过更精细的原始数据和优化的融合逻辑，全面捕捉A股市场的“体质”，识别具备高质量、深层博弈优势的标的。
                      引入“短板效应”惩罚和“协同奖励”机制，并新增对原始数据有效性的动态权重调整，以增强鲁棒性。
        - A股特性: 健康的上涨不仅结构稳固、换手温和，更应具备强大的下跌抵抗能力、主力资金的真实信念，
                    以及高质量的价格行为和健康的筹码分布。此升级旨在识别这种“抗揍”、有“内生动力”、
                    “行为纯粹”且“筹码稳固”的健康体质。
        """
        print("    -> [基础层] 正在诊断“市场体质”公理 (V5.1 · 数据鲁棒性强化版)...")
        # 获取市场体质公理的专属参数
        p_conf_mc = get_params_block(self.strategy, 'foundation_ultimate_params', {}).get('market_constitution_params', {})
        # 获取行为动态参数中的MTF归一化默认权重
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        # 获取探针启用状态
        probe_enabled = get_param_value(p_conf_mc.get('enable_probe'), True)
        df_index = df.index
        # 获取参数
        ma_periods = p_conf_mc.get('ma_periods', [5, 13, 21, 55])
        msq_weights = p_conf_mc.get('ma_structure_quality_weights', {'macd_score': 0.2, 'alignment_score': 0.3, 'slope_score': 0.3, 'dma_slope_score': 0.1, 'orderliness_score': 0.05, 'tension_score': 0.05})
        th_weights = p_conf_mc.get('turnover_health_weights', {'turnover_rate': 0.4, 'volume_burstiness': 0.2, 'constructive_turnover': 0.3, 'volume_structure_skew': 0.1})
        mcs_weights = p_conf_mc.get('market_conviction_strength_weights', {'trend_alignment': 0.3, 'main_force_conviction': 0.3, 'flow_credibility': 0.2, 'main_force_flow_directionality': 0.1, 'main_force_cost_advantage': 0.1})
        ir_weights_base = p_conf_mc.get('immune_resilience_weights', {'dip_absorption': 0.3, 'pressure_rejection': 0.2, 'chip_health': 0.2, 'capitulation_absorption': 0.1, 'winner_stability': 0.1, 'loser_pain': 0.1})
        paq_weights = p_conf_mc.get('price_action_quality_weights', {'upward_impulse_purity': 0.3, 'thrust_efficiency': 0.3, 'closing_acceptance_type': 0.2, 'volume_structure_skew': 0.2})
        cgd_weights = p_conf_mc.get('chip_game_depth_weights', {'winner_stability': 0.3, 'loser_pain': 0.2, 'main_force_flow_directionality': 0.3, 'main_force_cost_advantage': 0.2})
        final_fusion_weights = p_conf_mc.get('final_fusion_weights', {'msq': 0.2, 'th': 0.15, 'mcs': 0.2, 'ir': 0.15, 'paq': 0.15, 'cgd': 0.15})
        short_board_penalty_threshold = get_param_value(p_conf_mc.get('short_board_penalty_threshold'), 0.2)
        short_board_penalty_factor = get_param_value(p_conf_mc.get('short_board_penalty_factor'), 0.5)
        synergy_bonus_threshold = get_param_value(p_conf_mc.get('synergy_bonus_threshold'), 0.8)
        synergy_bonus_factor = get_param_value(p_conf_mc.get('synergy_bonus_factor'), 0.2)
        # 新增: 数据鲁棒性参数
        capitulation_absorption_min_value_threshold = get_param_value(p_conf_mc.get('capitulation_absorption_min_value_threshold'), 0.1)
        capitulation_absorption_check_window = get_param_value(p_conf_mc.get('capitulation_absorption_check_window'), 21)
        # 1. 校验所需信号 (动态构建信号名称)
        required_signals = [
            'MACDh_13_34_8_D', 'SLOPE_5_DMA_D', 'turnover_rate_f_D', 'trend_alignment_index_D',
            'dip_absorption_power_D',
            'MA_POTENTIAL_ORDERLINESS_SCORE_D',
            'MA_POTENTIAL_TENSION_INDEX_D',
            'volume_burstiness_index_D',
            'constructive_turnover_ratio_D',
            'main_force_conviction_index_D',
            'flow_credibility_index_D',
            'pressure_rejection_strength_D',
            'chip_health_score_D',
            'capitulation_absorption_index_D',
            'upward_impulse_purity_D',
            'thrust_efficiency_score_D',
            'closing_acceptance_type_D',
            'volume_structure_skew_D',
            'winner_stability_index_D',
            'loser_pain_index_D',
            'main_force_flow_directionality_D',
            'main_force_cost_advantage_D'
        ]
        required_signals.extend([f'EMA_{p}_D' for p in ma_periods])
        required_signals.extend([f'SLOPE_{p}_EMA_{p}_D' for p in ma_periods])
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_market_constitution"):
            return pd.Series(0.0, index=df.index)
        # --- 原始数据获取 ---
        macd_h_raw = self._get_safe_series(df, 'MACDh_13_34_8_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        dma_slope_raw = self._get_safe_series(df, 'SLOPE_5_DMA_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        turnover_rate_raw = self._get_safe_series(df, 'turnover_rate_f_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        trend_alignment_raw = self._get_safe_series(df, 'trend_alignment_index_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        dip_absorption_raw = self._get_safe_series(df, 'dip_absorption_power_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        ma_orderliness_raw = self._get_safe_series(df, 'MA_POTENTIAL_ORDERLINESS_SCORE_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        ma_tension_raw = self._get_safe_series(df, 'MA_POTENTIAL_TENSION_INDEX_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        volume_burstiness_raw = self._get_safe_series(df, 'volume_burstiness_index_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        constructive_turnover_raw = self._get_safe_series(df, 'constructive_turnover_ratio_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        main_force_conviction_raw = self._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        flow_credibility_raw = self._get_safe_series(df, 'flow_credibility_index_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        pressure_rejection_raw = self._get_safe_series(df, 'pressure_rejection_strength_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        chip_health_raw = self._get_safe_series(df, 'chip_health_score_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        capitulation_absorption_raw = self._get_safe_series(df, 'capitulation_absorption_index_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        upward_impulse_purity_raw = self._get_safe_series(df, 'upward_impulse_purity_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        thrust_efficiency_score_raw = self._get_safe_series(df, 'thrust_efficiency_score_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        closing_acceptance_type_raw = self._get_safe_series(df, 'closing_acceptance_type_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        volume_structure_skew_raw = self._get_safe_series(df, 'volume_structure_skew_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        winner_stability_index_raw = self._get_safe_series(df, 'winner_stability_index_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        loser_pain_index_raw = self._get_safe_series(df, 'loser_pain_index_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        main_force_flow_directionality_raw = self._get_safe_series(df, 'main_force_flow_directionality_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        main_force_cost_advantage_raw = self._get_safe_series(df, 'main_force_cost_advantage_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        # --- 1. 均线骨架质量 (MA Structure Quality - MSQ) ---
        macd_score = get_adaptive_mtf_normalized_bipolar_score(macd_h_raw, df_index, default_weights)
        
        bull_alignment_scores = []
        for i in range(len(ma_periods) - 1):
            ema_current = self._get_safe_series(df, f'EMA_{ma_periods[i]}_D', 0.0, method_name="_diagnose_axiom_market_constitution")
            ema_next = self._get_safe_series(df, f'EMA_{ma_periods[i+1]}_D', 0.0, method_name="_diagnose_axiom_market_constitution")
            bull_alignment_scores.append((ema_current > ema_next).astype(float).values)
        alignment_score = np.mean(bull_alignment_scores, axis=0) if bull_alignment_scores else np.full(len(df_index), 0.5)
        alignment_bipolar = (pd.Series(alignment_score, index=df_index) - 0.5) * 2
        slope_scores = []
        for p in ma_periods:
            slope_raw = self._get_safe_series(df, f'SLOPE_{p}_EMA_{p}_D', 0.0, method_name="_diagnose_axiom_market_constitution")
            slope_scores.append(get_adaptive_mtf_normalized_bipolar_score(slope_raw, df_index, default_weights).values)
        avg_slope_bipolar = pd.Series(np.mean(slope_scores, axis=0), index=df_index)
        dma_slope_score = get_adaptive_mtf_normalized_bipolar_score(dma_slope_raw, df_index, default_weights)
        ma_orderliness_score = get_adaptive_mtf_normalized_bipolar_score(ma_orderliness_raw, df_index, default_weights)
        ma_tension_score = get_adaptive_mtf_normalized_bipolar_score(ma_tension_raw, df_index, default_weights)
        ma_structure_quality_score = (
            macd_score * msq_weights.get('macd_score', 0.2) +
            alignment_bipolar * msq_weights.get('alignment_score', 0.3) +
            avg_slope_bipolar * msq_weights.get('slope_score', 0.3) +
            dma_slope_score * msq_weights.get('dma_slope_score', 0.1) +
            ma_orderliness_score * msq_weights.get('orderliness_score', 0.05) +
            ma_tension_score * msq_weights.get('tension_score', 0.05)
        ).clip(-1, 1)
        # --- 2. 筹码新陈代谢健康度 (Turnover Health - TH) ---
        turnover_health_score_v3 = get_adaptive_mtf_normalized_score(turnover_rate_raw, df_index, ascending=False, tf_weights=default_weights)
        volume_burstiness_score = get_adaptive_mtf_normalized_score(volume_burstiness_raw, df_index, default_weights, ascending=False)
        constructive_turnover_score = get_adaptive_mtf_normalized_score(constructive_turnover_raw, df_index, default_weights, ascending=True)
        volume_structure_skew_score_th = get_adaptive_mtf_normalized_bipolar_score(volume_structure_skew_raw, df_index, default_weights)
        turnover_health_score = (
            turnover_health_score_v3 * th_weights.get('turnover_rate', 0.4) +
            volume_burstiness_score * th_weights.get('volume_burstiness', 0.2) +
            constructive_turnover_score * th_weights.get('constructive_turnover', 0.3) +
            (volume_structure_skew_score_th + 1) / 2 * th_weights.get('volume_structure_skew', 0.1)
        ).clip(0, 1)
        # --- 3. 市场信念强度 (Market Conviction Strength - MCS) ---
        conviction_score_v3 = get_adaptive_mtf_normalized_score(trend_alignment_raw, df_index, ascending=True, tf_weights=default_weights)
        main_force_conviction_score = get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, default_weights)
        flow_credibility_score = get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, default_weights, ascending=True)
        main_force_flow_directionality_score = get_adaptive_mtf_normalized_bipolar_score(main_force_flow_directionality_raw, df_index, default_weights)
        main_force_cost_advantage_score = get_adaptive_mtf_normalized_bipolar_score(main_force_cost_advantage_raw, df_index, default_weights)
        market_conviction_strength_score = (
            (conviction_score_v3 * 2 - 1) * mcs_weights.get('trend_alignment', 0.3) +
            main_force_conviction_score * mcs_weights.get('main_force_conviction', 0.3) +
            (flow_credibility_score * 2 - 1) * mcs_weights.get('flow_credibility', 0.2) +
            main_force_flow_directionality_score * mcs_weights.get('main_force_flow_directionality', 0.1) +
            main_force_cost_advantage_score * mcs_weights.get('main_force_cost_advantage', 0.1)
        ).clip(-1, 1)
        # --- 4. 免疫韧性与抗压能力 (Immune Resilience - IR) ---
        # 动态调整 ir_weights
        ir_weights = ir_weights_base.copy()
        # 检查 capitulation_absorption_index_D 的有效性
        cap_abs_max_in_window = capitulation_absorption_raw.rolling(window=capitulation_absorption_check_window, min_periods=1).max()
        is_cap_abs_valid = (cap_abs_max_in_window > capitulation_absorption_min_value_threshold)
        # 创建一个Series来存储动态权重
        dynamic_ir_weights = pd.DataFrame(index=df_index, columns=ir_weights.keys(), dtype=float)
        for key, value in ir_weights.items():
            dynamic_ir_weights[key] = value
        # 对无效的日期，将capitulation_absorption的权重重新分配
        invalid_mask = ~is_cap_abs_valid
        if invalid_mask.any():
            cap_abs_weight = dynamic_ir_weights.loc[invalid_mask, 'capitulation_absorption'].fillna(0)
            # 将权重分配给 dip_absorption 和 pressure_rejection
            # 确保分配的权重不会导致其他权重超过1
            remaining_weight_sum = dynamic_ir_weights.loc[invalid_mask, ['dip_absorption', 'pressure_rejection', 'chip_health', 'winner_stability', 'loser_pain']].sum(axis=1)
            # 优先分配给 dip_absorption 和 pressure_rejection
            # 计算可分配的比例，避免超过1
            distribute_ratio_da = dynamic_ir_weights.loc[invalid_mask, 'dip_absorption'] / (dynamic_ir_weights.loc[invalid_mask, 'dip_absorption'] + dynamic_ir_weights.loc[invalid_mask, 'pressure_rejection'] + 1e-9)
            distribute_ratio_pr = dynamic_ir_weights.loc[invalid_mask, 'pressure_rejection'] / (dynamic_ir_weights.loc[invalid_mask, 'dip_absorption'] + dynamic_ir_weights.loc[invalid_mask, 'pressure_rejection'] + 1e-9)
            # 确保分母不为0
            distribute_ratio_da = distribute_ratio_da.fillna(0.5)
            distribute_ratio_pr = distribute_ratio_pr.fillna(0.5)
            dynamic_ir_weights.loc[invalid_mask, 'dip_absorption'] += cap_abs_weight * distribute_ratio_da
            dynamic_ir_weights.loc[invalid_mask, 'pressure_rejection'] += cap_abs_weight * distribute_ratio_pr
            dynamic_ir_weights.loc[invalid_mask, 'capitulation_absorption'] = 0.0 # 将无效指标的权重设为0
        dip_absorption_score = get_adaptive_mtf_normalized_score(dip_absorption_raw, df_index, ascending=True, tf_weights=default_weights)
        pressure_rejection_score = get_adaptive_mtf_normalized_score(pressure_rejection_raw, df_index, default_weights, ascending=True)
        chip_health_score = get_adaptive_mtf_normalized_score(chip_health_raw, df_index, default_weights, ascending=True)
        capitulation_absorption_score = get_adaptive_mtf_normalized_score(capitulation_absorption_raw, df_index, default_weights, ascending=True)
        winner_stability_score = get_adaptive_mtf_normalized_score(winner_stability_index_raw, df_index, default_weights, ascending=True)
        loser_pain_score = get_adaptive_mtf_normalized_score(loser_pain_index_raw, df_index, default_weights, ascending=False)
        immune_resilience_score = (
            dip_absorption_score * dynamic_ir_weights['dip_absorption'] +
            pressure_rejection_score * dynamic_ir_weights['pressure_rejection'] +
            chip_health_score * dynamic_ir_weights['chip_health'] +
            capitulation_absorption_score * dynamic_ir_weights['capitulation_absorption'] +
            winner_stability_score * dynamic_ir_weights['winner_stability'] +
            loser_pain_score * dynamic_ir_weights['loser_pain']
        ).clip(0, 1)
        # --- 5. 价格行为品质 (Price Action Quality - PAQ) ---
        upward_impulse_purity_score = get_adaptive_mtf_normalized_score(upward_impulse_purity_raw, df_index, default_weights, ascending=True)
        thrust_efficiency_score = get_adaptive_mtf_normalized_score(thrust_efficiency_score_raw, df_index, default_weights, ascending=True)
        closing_acceptance_type_score = get_adaptive_mtf_normalized_score(closing_acceptance_type_raw, df_index, default_weights, ascending=True)
        volume_structure_skew_score_paq = get_adaptive_mtf_normalized_bipolar_score(volume_structure_skew_raw, df_index, default_weights)
        price_action_quality_score = (
            upward_impulse_purity_score * paq_weights.get('upward_impulse_purity', 0.3) +
            thrust_efficiency_score * paq_weights.get('thrust_efficiency', 0.3) +
            (closing_acceptance_type_score * 2 - 1) * paq_weights.get('closing_acceptance_type', 0.2) +
            volume_structure_skew_score_paq * paq_weights.get('volume_structure_skew', 0.2)
        ).clip(-1, 1)
        # --- 6. 筹码博弈深度 (Chip Game Depth - CGD) ---
        winner_stability_score_cgd = get_adaptive_mtf_normalized_score(winner_stability_index_raw, df_index, default_weights, ascending=True)
        loser_pain_score_cgd = get_adaptive_mtf_normalized_score(loser_pain_index_raw, df_index, default_weights, ascending=False)
        main_force_flow_directionality_score_cgd = get_adaptive_mtf_normalized_bipolar_score(main_force_flow_directionality_raw, df_index, default_weights)
        main_force_cost_advantage_score_cgd = get_adaptive_mtf_normalized_bipolar_score(main_force_cost_advantage_raw, df_index, default_weights)
        chip_game_depth_score = (
            winner_stability_score_cgd * cgd_weights.get('winner_stability', 0.3) +
            (loser_pain_score_cgd * 2 - 1) * cgd_weights.get('loser_pain', 0.2) +
            main_force_flow_directionality_score_cgd * cgd_weights.get('main_force_flow_directionality', 0.3) +
            main_force_cost_advantage_score_cgd * cgd_weights.get('main_force_cost_advantage', 0.2)
        ).clip(-1, 1)
        # --- 7. 最终融合: 六大核心维度 + 非线性调节 ---
        # 将所有维度归一化到 [0, 1] 范围，以便进行短板效应和协同奖励
        msq_unipolar = (ma_structure_quality_score + 1) / 2
        mcs_unipolar = (market_conviction_strength_score + 1) / 2
        paq_unipolar = (price_action_quality_score + 1) / 2
        cgd_unipolar = (chip_game_depth_score + 1) / 2
        # 核心维度加权融合，注意将 [0,1] 的分数转换为 [-1,1] 再进行融合
        raw_constitution_score = (
            ma_structure_quality_score * final_fusion_weights.get('msq', 0.2) +
            (turnover_health_score * 2 - 1) * final_fusion_weights.get('th', 0.15) +
            market_conviction_strength_score * final_fusion_weights.get('mcs', 0.2) +
            (immune_resilience_score * 2 - 1) * final_fusion_weights.get('ir', 0.15) +
            price_action_quality_score * final_fusion_weights.get('paq', 0.15) +
            chip_game_depth_score * final_fusion_weights.get('cgd', 0.15)
        ).clip(-1, 1)
        # 短板效应惩罚: 如果任何一个核心维度（归一化到[0,1]）低于阈值，则进行惩罚
        min_score_components = pd.DataFrame({
            'msq': msq_unipolar,
            'th': turnover_health_score,
            'mcs': mcs_unipolar,
            'ir': immune_resilience_score,
            'paq': paq_unipolar,
            'cgd': cgd_unipolar
        }).min(axis=1)
        short_board_penalty = pd.Series(1.0, index=df_index)
        penalty_condition = min_score_components < short_board_penalty_threshold
        if penalty_condition.any():
            penalty_amount = (short_board_penalty_threshold - min_score_components[penalty_condition]) / short_board_penalty_threshold * short_board_penalty_factor
            short_board_penalty.loc[penalty_condition] = (1 - penalty_amount).clip(lower=0.0)
        # 协同奖励: 如果所有核心维度（归一化到[0,1]）都高于阈值，则给予奖励
        synergy_condition = (msq_unipolar > synergy_bonus_threshold) & \
                            (turnover_health_score > synergy_bonus_threshold) & \
                            (mcs_unipolar > synergy_bonus_threshold) & \
                            (immune_resilience_score > synergy_bonus_threshold) & \
                            (paq_unipolar > synergy_bonus_threshold) & \
                            (cgd_unipolar > synergy_bonus_threshold)
        
        synergy_bonus = pd.Series(0.0, index=df_index)
        if synergy_condition.any():
            avg_high_score = pd.DataFrame({
                'msq': msq_unipolar[synergy_condition],
                'th': turnover_health_score[synergy_condition],
                'mcs': mcs_unipolar[synergy_condition],
                'ir': immune_resilience_score[synergy_condition],
                'paq': paq_unipolar[synergy_condition],
                'cgd': cgd_unipolar[synergy_condition]
            }).mean(axis=1)
            synergy_bonus.loc[synergy_condition] = (avg_high_score - synergy_bonus_threshold) / (1 - synergy_bonus_threshold) * synergy_bonus_factor
        # 应用非线性调节
        constitution_score = raw_constitution_score * short_board_penalty * (1 + synergy_bonus)
        # 确保最终分数在 [-1, 1] 范围内
        constitution_score = constitution_score.clip(-1, 1).astype(np.float32)
        print("    -> [基础层] “市场体质”公理诊断完成。")
        return constitution_score

    def _diagnose_axiom_sentiment_pendulum(self, df: pd.DataFrame) -> pd.Series:
        """
        【V5.0 · 情绪结构与诡道共振版】基础公理二：诊断“市场情绪钟摆”
        - 核心逻辑: 融合市场整体情绪、散户恐慌/FOMO、RSI动量，并通过“诡道精算器”进行非对称调节，
                      同时引入“情境放大器”进行宏观上下文校准。旨在穿透表层情绪，识别真实的市场心理状态。
        - 核心升级:
            1. 情绪核心重构：引入“情绪结构品质”，融合情绪活跃度、情绪一致性，更细致刻画情绪质地。
            2. 诡道精算器增强：引入“诡道行为确认”，利用量价行为交叉验证诱多/诱空，提升甄别精度。
            3. 情境放大器深化：引入“市场结构稳定性”和“行业情绪共振”，提供更全面的宏观情境校准。
        - A股特性: 情绪常常是陷阱。此升级旨在利用更精细的诡道甄别和情境感知，过滤主力诱多或诱空行为，
                      并捕捉情绪在不同市场环境下的真实价值。
        """
        print("    -> [基础层] 正在诊断“情绪钟摆”公理 (V5.0 · 情绪结构与诡道共振版)...")
        # 获取情绪钟摆公理的专属参数
        p_conf_sp = get_params_block(self.strategy, 'foundation_ultimate_params', {}).get('sentiment_pendulum_params', {})
        probe_enabled = get_param_value(p_conf_sp.get('enable_probe'), True)
        # 获取行为动态参数中的MTF归一化默认权重
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        df_index = df.index
        # 1. 校验所需信号
        required_signals = [
            'RSI_13_D', 'retail_panic_surrender_index_D', 'retail_fomo_premium_index_D',
            'market_sentiment_score_D',
            'deception_lure_long_intensity_D', 'deception_lure_short_intensity_D', 'wash_trade_intensity_D',
            'VOLATILITY_INSTABILITY_INDEX_21d_D', 'main_force_conviction_index_D',
            # V5.0 新增原始数据 (情绪核心)
            'turnover_rate_f_D', 'volume_burstiness_index_D', 'PRICE_VOLUME_ENTROPY_D',
            'SLOPE_5_RSI_13_D', 'ACCEL_5_RSI_13_D',
            # V5.0 新增原始数据 (诡道精算器)
            'closing_acceptance_type_D', 'volume_structure_skew_D', 'upward_impulse_purity_D', 'thrust_efficiency_score_D',
            # V5.0 新增原始数据 (情境放大器)
            'BBW_21_2.0_D', 'MA_POTENTIAL_TENSION_INDEX_D', 'industry_strength_rank_D', 'SLOPE_5_industry_strength_rank_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_sentiment_pendulum"):
            return pd.Series(0.0, index=df.index)
        # --- 原始数据获取 ---
        rsi_raw = self._get_safe_series(df, 'RSI_13_D', 50.0, method_name="_diagnose_axiom_sentiment_pendulum")
        retail_panic_raw = self._get_safe_series(df, 'retail_panic_surrender_index_D', 0.0, method_name="_diagnose_axiom_sentiment_pendulum")
        retail_fomo_raw = self._get_safe_series(df, 'retail_fomo_premium_index_D', 0.0, method_name="_diagnose_axiom_sentiment_pendulum")
        market_sentiment_raw = self._get_safe_series(df, 'market_sentiment_score_D', 0.0, method_name="_diagnose_axiom_sentiment_pendulum")
        deception_lure_long_raw = self._get_safe_series(df, 'deception_lure_long_intensity_D', 0.0, method_name="_diagnose_axiom_sentiment_pendulum")
        deception_lure_short_raw = self._get_safe_series(df, 'deception_lure_short_intensity_D', 0.0, method_name="_diagnose_axiom_sentiment_pendulum")
        wash_trade_raw = self._get_safe_series(df, 'wash_trade_intensity_D', 0.0, method_name="_diagnose_axiom_sentiment_pendulum")
        volatility_instability_raw = self._get_safe_series(df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', 0.0, method_name="_diagnose_axiom_sentiment_pendulum")
        main_force_conviction_raw = self._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name="_diagnose_axiom_sentiment_pendulum")
        # V5.0 新增原始数据
        turnover_rate_raw = self._get_safe_series(df, 'turnover_rate_f_D', 0.0, method_name="_diagnose_axiom_sentiment_pendulum")
        volume_burstiness_raw = self._get_safe_series(df, 'volume_burstiness_index_D', 0.0, method_name="_diagnose_axiom_sentiment_pendulum")
        price_volume_entropy_raw = self._get_safe_series(df, 'PRICE_VOLUME_ENTROPY_D', 0.0, method_name="_diagnose_axiom_sentiment_pendulum")
        rsi_slope_raw = self._get_safe_series(df, 'SLOPE_5_RSI_13_D', 0.0, method_name="_diagnose_axiom_sentiment_pendulum")
        rsi_accel_raw = self._get_safe_series(df, 'ACCEL_5_RSI_13_D', 0.0, method_name="_diagnose_axiom_sentiment_pendulum")
        closing_acceptance_type_raw = self._get_safe_series(df, 'closing_acceptance_type_D', 0.0, method_name="_diagnose_axiom_sentiment_pendulum")
        volume_structure_skew_raw = self._get_safe_series(df, 'volume_structure_skew_D', 0.0, method_name="_diagnose_axiom_sentiment_pendulum")
        upward_impulse_purity_raw = self._get_safe_series(df, 'upward_impulse_purity_D', 0.0, method_name="_diagnose_axiom_sentiment_pendulum")
        thrust_efficiency_score_raw = self._get_safe_series(df, 'thrust_efficiency_score_D', 0.0, method_name="_diagnose_axiom_sentiment_pendulum")
        bbw_raw = self._get_safe_series(df, 'BBW_21_2.0_D', 0.0, method_name="_diagnose_axiom_sentiment_pendulum")
        ma_potential_tension_raw = self._get_safe_series(df, 'MA_POTENTIAL_TENSION_INDEX_D', 0.0, method_name="_diagnose_axiom_sentiment_pendulum")
        industry_strength_rank_raw = self._get_safe_series(df, 'industry_strength_rank_D', 0.5, method_name="_diagnose_axiom_sentiment_pendulum")
        industry_strength_rank_slope_raw = self._get_safe_series(df, 'SLOPE_5_industry_strength_rank_D', 0.0, method_name="_diagnose_axiom_sentiment_pendulum")
        # --- 2. 情绪核心 (Sentiment Core) ---
        sc_weights = p_conf_sp.get('sentiment_core_weights', {'market_sentiment': 0.3, 'retail_fomo': 0.15, 'retail_panic': 0.15, 'rsi_momentum': 0.2, 'sentiment_structure_quality': 0.2})
        market_sentiment_score = get_adaptive_mtf_normalized_bipolar_score(market_sentiment_raw, df_index, default_weights)
        retail_fomo_score = get_adaptive_mtf_normalized_score(retail_fomo_raw, df_index, default_weights, ascending=True)
        retail_panic_score = get_adaptive_mtf_normalized_score(retail_panic_raw, df_index, default_weights, ascending=True)
        rsi_momentum_score = get_adaptive_mtf_normalized_bipolar_score(rsi_raw - 50.0, df_index, default_weights, sensitivity=10.0)
        retail_fomo_bipolar = retail_fomo_score * 2 - 1
        retail_panic_bipolar = -(retail_panic_score * 2 - 1)
        # V5.0 新增: 情绪结构品质 (Sentiment Structure Quality)
        # 修改开始
        ssq_weights = p_conf_sp.get('sentiment_structure_quality_weights', {
            'activity_overall': 0.4, 'consistency_overall': 0.3, 'rsi_dynamics_overall': 0.3,
            'activity_sub_weights': {'turnover': 0.5, 'burstiness': 0.5},
            'rsi_dynamics_sub_weights': {'slope': 0.5, 'accel': 0.5}
        })
        # 定义 turnover_rate_score 和 volume_burstiness_score
        turnover_rate_score = get_adaptive_mtf_normalized_score(turnover_rate_raw, df_index, default_weights, ascending=True)
        volume_burstiness_score = get_adaptive_mtf_normalized_score(volume_burstiness_raw, df_index, default_weights, ascending=True)
        # 情绪活跃度: 换手率和成交量爆发性
        sentiment_activity_score = (
            turnover_rate_score * ssq_weights.get('activity_sub_weights', {}).get('turnover', 0.5) +
            volume_burstiness_score * ssq_weights.get('activity_sub_weights', {}).get('burstiness', 0.5)
        ).clip(0, 1)
        # 修改结束
        # 情绪一致性: 价量熵 (负向)
        price_volume_entropy_score = get_adaptive_mtf_normalized_score(price_volume_entropy_raw, df_index, default_weights, ascending=False)
        sentiment_consistency_score = price_volume_entropy_score.clip(0, 1)
        # RSI动态分: RSI斜率和加速度
        rsi_slope_score = get_adaptive_mtf_normalized_bipolar_score(rsi_slope_raw, df_index, default_weights)
        rsi_accel_score = get_adaptive_mtf_normalized_bipolar_score(rsi_accel_raw, df_index, default_weights)
        rsi_dynamics_score = (
            rsi_slope_score * ssq_weights.get('rsi_dynamics_sub_weights', {}).get('slope', 0.5) +
            rsi_accel_score * ssq_weights.get('rsi_dynamics_sub_weights', {}).get('accel', 0.5)
        ).clip(-1, 1)
        sentiment_structure_quality_score = (
            sentiment_activity_score * ssq_weights.get('activity_overall', 0.4) +
            sentiment_consistency_score * ssq_weights.get('consistency_overall', 0.3) +
            (rsi_dynamics_score + 1) / 2 * ssq_weights.get('rsi_dynamics_overall', 0.3)
        ).clip(0, 1)
        sentiment_core = (
            market_sentiment_score * sc_weights.get('market_sentiment', 0.3) +
            retail_fomo_bipolar * sc_weights.get('retail_fomo', 0.15) +
            retail_panic_bipolar * sc_weights.get('retail_panic', 0.15) +
            rsi_momentum_score * sc_weights.get('rsi_momentum', 0.2) +
            (sentiment_structure_quality_score * 2 - 1) * sc_weights.get('sentiment_structure_quality', 0.2)
        ).clip(-1, 1)
        # --- 3. 诡道精算器 (Deception Modulator) ---
        dm_params = p_conf_sp.get('deception_modulator_params', {})
        lure_long_penalty_factor = get_param_value(dm_params.get('lure_long_penalty_factor'), 0.7)
        lure_short_bonus_factor = get_param_value(dm_params.get('lure_short_bonus_factor'), 0.5)
        wash_trade_credibility_reduction_factor = get_param_value(dm_params.get('wash_trade_credibility_reduction_factor'), 0.5)
        deception_confirmation_weight = get_param_value(dm_params.get('deception_confirmation_weight'), 0.3)
        lure_long_score = get_adaptive_mtf_normalized_score(deception_lure_long_raw, df_index, default_weights, ascending=True)
        lure_short_score = get_adaptive_mtf_normalized_score(deception_lure_short_raw, df_index, default_weights, ascending=True)
        wash_trade_score = get_adaptive_mtf_normalized_score(wash_trade_raw, df_index, default_weights, ascending=True)
        deception_modulator = pd.Series(1.0, index=df_index)
        # V5.0 新增: 诡道行为确认 (Deception Behavior Confirmation)
        dbc_weights = p_conf_sp.get('deception_behavior_confirmation_weights', {'closing_acceptance': 0.3, 'volume_skew': 0.3, 'upward_purity': 0.2, 'thrust_efficiency': 0.2})
        closing_acceptance_score = get_adaptive_mtf_normalized_score(closing_acceptance_type_raw, df_index, default_weights, ascending=True)
        volume_structure_skew_score = get_adaptive_mtf_normalized_bipolar_score(volume_structure_skew_raw, df_index, default_weights)
        upward_impulse_purity_score = get_adaptive_mtf_normalized_score(upward_impulse_purity_raw, df_index, default_weights, ascending=True)
        thrust_efficiency_score = get_adaptive_mtf_normalized_score(thrust_efficiency_score_raw, df_index, default_weights, ascending=True)
        # 诡道行为确认分 (0-1)
        # 诱多确认：收盘接受度越低，偏度越负，纯度越低，效率越低，越确认诱多
        # 诱空确认：收盘接受度越高，偏度越正，纯度越高，效率越高，越确认诱空
        # 这里构建一个双极性的确认分，正向确认诱多，负向确认诱空
        deception_behavior_confirmation_score = (
            (1 - closing_acceptance_score) * dbc_weights.get('closing_acceptance', 0.3) +
            (1 - (volume_structure_skew_score + 1) / 2) * dbc_weights.get('volume_skew', 0.3) +
            (1 - upward_impulse_purity_score) * dbc_weights.get('upward_purity', 0.2) +
            (1 - thrust_efficiency_score) * dbc_weights.get('thrust_efficiency', 0.2)
        ).clip(0, 1)
        # 诱多惩罚: 如果情绪核心为正 (贪婪) 且有诱多行为，则惩罚，并由诡道行为确认分强化
        bull_trap_condition = (sentiment_core > 0) & (lure_long_score > 0)
        deception_modulator.loc[bull_trap_condition] -= lure_long_score.loc[bull_trap_condition] * lure_long_penalty_factor * (1 + deception_behavior_confirmation_score.loc[bull_trap_condition] * deception_confirmation_weight)
        # 诱空奖励: 如果情绪核心为负 (恐慌) 且有诱空行为，则奖励 (视为潜在底部机会)，并由诡道行为确认分强化
        bear_trap_opportunity_condition = (sentiment_core < 0) & (lure_short_score > 0)
        # 诱空时，诡道行为确认分应该反向作用，即行为越“好”，越确认诱空是机会
        # 所以这里使用 (1 - deception_behavior_confirmation_score) 来表示行为的“好”
        deception_modulator.loc[bear_trap_opportunity_condition] += lure_short_score.loc[bear_trap_opportunity_condition] * lure_short_bonus_factor * (1 + (1 - deception_behavior_confirmation_score.loc[bear_trap_opportunity_condition]) * deception_confirmation_weight)
        # 对倒削弱情绪可信度
        deception_modulator *= (1 - wash_trade_score * wash_trade_credibility_reduction_factor)
        deception_modulator = deception_modulator.clip(0.0, 2.0)
        # --- 4. 情境放大器 (Contextual Amplifier) ---
        ca_params = p_conf_sp.get('contextual_amplification_params', {})
        volatility_weight = get_param_value(ca_params.get('volatility_weight', 0.2))
        main_force_conviction_weight = get_param_value(ca_params.get('main_force_conviction_weight', 0.3))
        market_structure_stability_weight = get_param_value(ca_params.get('market_structure_stability_weight', 0.3))
        sector_sentiment_resonance_weight = get_param_value(ca_params.get('sector_sentiment_resonance_weight', 0.2))
        base_amplifier_factor = get_param_value(ca_params.get('base_amplifier_factor'), 0.25)
        volatility_score = get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, default_weights, ascending=True)
        main_force_conviction_score = get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, default_weights)
        # V5.0 新增: 市场结构稳定性 (Market Structure Stability)
        bbw_score = get_adaptive_mtf_normalized_score(bbw_raw, df_index, default_weights, ascending=False)
        ma_potential_tension_score = get_adaptive_mtf_normalized_score(ma_potential_tension_raw, df_index, default_weights, ascending=True)
        market_structure_stability_score = (bbw_score * 0.5 + ma_potential_tension_score * 0.5).clip(0, 1)
        # V5.0 新增: 行业情绪共振 (Sector Sentiment Resonance)
        industry_strength_rank_score = (industry_strength_rank_raw - 0.5) * 2
        industry_strength_rank_slope_score = get_adaptive_mtf_normalized_bipolar_score(industry_strength_rank_slope_raw, df_index, default_weights)
        sector_sentiment_resonance_score = (industry_strength_rank_score * 0.5 + industry_strength_rank_slope_score * 0.5).clip(-1, 1)
        context_amplifier_score = (
            volatility_score * volatility_weight +
            main_force_conviction_score * main_force_conviction_weight +
            (market_structure_stability_score * 2 - 1) * market_structure_stability_weight +
            sector_sentiment_resonance_score * sector_sentiment_resonance_weight
        ).clip(-1, 1)
        context_multiplier = 1 + context_amplifier_score * base_amplifier_factor
        context_multiplier = context_multiplier.clip(0.5, 1.5)
        # --- 5. 最终情绪钟摆分数 ---
        sentiment_pendulum_score = (sentiment_core * deception_modulator * context_multiplier).clip(-1, 1)
        print("    -> [基础层] “情绪钟摆”公理诊断完成。")
        return sentiment_pendulum_score.astype(np.float32)

    def _diagnose_axiom_liquidity_tide(self, df: pd.DataFrame) -> pd.Series:
        """
        【V5.0 · 微观结构与动态博弈版】基础公理三：诊断“流动性潮汐”
        - 核心逻辑: 在融合流动性方向、能量、活跃度与结构品质的基础上，深度引入“微观流动性健康度”和“流动性韧性”两大维度，
                      并采用情境自适应的动态权重与非线性融合，同时深化“诡道情境调制器”的动态与非对称特性。
        - A股特性: 成交量可以作假。此升级旨在穿透表层流动性数据，还原市场真实的、由多空双方真金白银博弈产生的流动性状态，
                      评估其内在品质、微观结构、持久性及主力意图。
        """
        print("    -> [基础层] 正在诊断“流动性潮汐”公理 (V5.0 · 微观结构与动态博弈版)...")
        # 获取流动性潮汐公理的专属参数
        p_conf_lt = get_params_block(self.strategy, 'foundation_ultimate_params', {}).get('liquidity_tide_params', {})
        probe_enabled = get_param_value(p_conf_lt.get('enable_probe'), True)
        # 获取行为动态参数中的MTF归一化默认权重
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        df_index = df.index
        # 1. 校验所需信号
        required_signals = [
            'CMF_21_D', 'SLOPE_5_amount_D', 'SLOPE_5_turnover_rate_f_D', 'wash_trade_intensity_D',
            'main_force_flow_directionality_D', 'volume_burstiness_index_D', 'SLOPE_5_volume_D',
            'turnover_rate_f_D', 'PRICE_VOLUME_ENTROPY_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D',
            'flow_credibility_index_D', 'market_sentiment_score_D', 'main_force_conviction_index_D',
            # V5.0 新增原始数据 (微观流动性健康度)
            'bid_ask_spread_D', 'market_depth_ratio_D', 'order_flow_imbalance_D',
            # V5.0 新增原始数据 (流动性韧性)
            'liquidity_persistence_index_D', 'volume_profile_density_D', 'vwap_deviation_D',
            # V5.0 新增原始数据 (动态权重情境因子)
            'SLOPE_55_close_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_liquidity_tide"):
            return pd.Series(0.0, index=df.index)
        # --- 原始数据获取 ---
        cmf_raw = self._get_safe_series(df, 'CMF_21_D', 0.0, method_name="_diagnose_axiom_liquidity_tide")
        amount_slope_raw = self._get_safe_series(df, 'SLOPE_5_amount_D', 0.0, method_name="_diagnose_axiom_liquidity_tide")
        turnover_slope_raw = self._get_safe_series(df, 'SLOPE_5_turnover_rate_f_D', 0.0, method_name="_diagnose_axiom_liquidity_tide")
        wash_trade_raw = self._get_safe_series(df, 'wash_trade_intensity_D', 0.0, method_name="_diagnose_axiom_liquidity_tide")
        main_force_flow_directionality_raw = self._get_safe_series(df, 'main_force_flow_directionality_D', 0.0, method_name="_diagnose_axiom_liquidity_tide")
        volume_burstiness_raw = self._get_safe_series(df, 'volume_burstiness_index_D', 0.0, method_name="_diagnose_axiom_liquidity_tide")
        volume_slope_raw = self._get_safe_series(df, 'SLOPE_5_volume_D', 0.0, method_name="_diagnose_axiom_liquidity_tide")
        turnover_rate_level_raw = self._get_safe_series(df, 'turnover_rate_f_D', 0.0, method_name="_diagnose_axiom_liquidity_tide")
        price_volume_entropy_raw = self._get_safe_series(df, 'PRICE_VOLUME_ENTROPY_D', 0.0, method_name="_diagnose_axiom_liquidity_tide")
        volatility_instability_raw = self._get_safe_series(df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', 0.0, method_name="_diagnose_axiom_liquidity_tide")
        flow_credibility_raw = self._get_safe_series(df, 'flow_credibility_index_D', 0.0, method_name="_diagnose_axiom_liquidity_tide")
        market_sentiment_raw = self._get_safe_series(df, 'market_sentiment_score_D', 0.0, method_name="_diagnose_axiom_liquidity_tide")
        main_force_conviction_raw = self._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name="_diagnose_axiom_liquidity_tide")
        # V5.0 新增原始数据
        bid_ask_spread_raw = self._get_safe_series(df, 'bid_ask_spread_D', 0.0, method_name="_diagnose_axiom_liquidity_tide")
        market_depth_ratio_raw = self._get_safe_series(df, 'market_depth_ratio_D', 0.0, method_name="_diagnose_axiom_liquidity_tide")
        order_flow_imbalance_raw = self._get_safe_series(df, 'order_flow_imbalance_D', 0.0, method_name="_diagnose_axiom_liquidity_tide")
        liquidity_persistence_index_raw = self._get_safe_series(df, 'liquidity_persistence_index_D', 0.0, method_name="_diagnose_axiom_liquidity_tide")
        volume_profile_density_raw = self._get_safe_series(df, 'volume_profile_density_D', 0.0, method_name="_diagnose_axiom_liquidity_tide")
        vwap_deviation_raw = self._get_safe_series(df, 'vwap_deviation_D', 0.0, method_name="_diagnose_axiom_liquidity_tide")
        # 动态权重情境因子
        market_trend_slope_raw = self._get_safe_series(df, 'SLOPE_55_close_D', 0.0, method_name="_diagnose_axiom_liquidity_tide")
        if probe_enabled:
            print(f"    -> [探针] 原始数据: CMF_21_D 尾部: {cmf_raw.tail().to_dict()}")
            print(f"    -> [探针] 原始数据: SLOPE_5_amount_D 尾部: {amount_slope_raw.tail().to_dict()}")
            print(f"    -> [探针] 原始数据: SLOPE_5_turnover_rate_f_D 尾部: {turnover_slope_raw.tail().to_dict()}")
            print(f"    -> [探针] 原始数据: wash_trade_intensity_D 尾部: {wash_trade_raw.tail().to_dict()}")
            print(f"    -> [探针] 原始数据: main_force_flow_directionality_D 尾部: {main_force_flow_directionality_raw.tail().to_dict()}")
            print(f"    -> [探针] 原始数据: volume_burstiness_index_D 尾部: {volume_burstiness_raw.tail().to_dict()}")
            print(f"    -> [探针] 原始数据: SLOPE_5_volume_D 尾部: {volume_slope_raw.tail().to_dict()}")
            print(f"    -> [探针] 原始数据: turnover_rate_f_D 尾部: {turnover_rate_level_raw.tail().to_dict()}")
            print(f"    -> [探针] 原始数据: PRICE_VOLUME_ENTROPY_D 尾部: {price_volume_entropy_raw.tail().to_dict()}")
            print(f"    -> [探针] 原始数据: VOLATILITY_INSTABILITY_INDEX_21d_D 尾部: {volatility_instability_raw.tail().to_dict()}")
            print(f"    -> [探针] 原始数据: flow_credibility_index_D 尾部: {flow_credibility_raw.tail().to_dict()}")
            print(f"    -> [探针] 原始数据: market_sentiment_score_D 尾部: {market_sentiment_raw.tail().to_dict()}")
            print(f"    -> [探针] 原始数据: main_force_conviction_index_D 尾部: {main_force_conviction_raw.tail().to_dict()}")
            print(f"    -> [探针] 原始数据: bid_ask_spread_D 尾部: {bid_ask_spread_raw.tail().to_dict()}")
            print(f"    -> [探针] 原始数据: market_depth_ratio_D 尾部: {market_depth_ratio_raw.tail().to_dict()}")
            print(f"    -> [探针] 原始数据: order_flow_imbalance_D 尾部: {order_flow_imbalance_raw.tail().to_dict()}")
            print(f"    -> [探针] 原始数据: liquidity_persistence_index_D 尾部: {liquidity_persistence_index_raw.tail().to_dict()}")
            print(f"    -> [探针] 原始数据: volume_profile_density_D 尾部: {volume_profile_density_raw.tail().to_dict()}")
            print(f"    -> [探针] 原始数据: vwap_deviation_D 尾部: {vwap_deviation_raw.tail().to_dict()}")
            print(f"    -> [探针] 原始数据: SLOPE_55_close_D 尾部: {market_trend_slope_raw.tail().to_dict()}")
        # --- 2. 潮汐方向 (Tide Direction) ---
        td_weights = p_conf_lt.get('tide_direction_weights', {'cmf': 0.4, 'main_force_flow_directionality': 0.6})
        cmf_score = get_adaptive_mtf_normalized_bipolar_score(cmf_raw, df_index, default_weights, sensitivity=0.5)
        main_force_flow_directionality_score = get_adaptive_mtf_normalized_bipolar_score(main_force_flow_directionality_raw, df_index, default_weights)
        direction_score = (
            cmf_score * td_weights.get('cmf', 0.4) +
            main_force_flow_directionality_score * td_weights.get('main_force_flow_directionality', 0.6)
        ).clip(-1, 1)
        if probe_enabled:
            print(f"    -> [探针] 关键计算节点: 潮汐方向得分 (direction_score) 尾部: {direction_score.tail().to_dict()}")
        # --- 3. 潮汐能量 (Tide Energy) ---
        te_weights = p_conf_lt.get('tide_energy_weights', {'amount_slope': 0.4, 'volume_burstiness': 0.3, 'volume_slope': 0.3})
        amount_slope_score = get_adaptive_mtf_normalized_bipolar_score(amount_slope_raw, df_index, default_weights)
        volume_burstiness_score = get_adaptive_mtf_normalized_score(volume_burstiness_raw, df_index, default_weights, ascending=True)
        volume_slope_score = get_adaptive_mtf_normalized_bipolar_score(volume_slope_raw, df_index, default_weights)
        energy_score = (
            amount_slope_score * te_weights.get('amount_slope', 0.4) +
            (volume_burstiness_score * 2 - 1) * te_weights.get('volume_burstiness', 0.3) + # 转换为双极性
            volume_slope_score * te_weights.get('volume_slope', 0.3)
        ).clip(-1, 1)
        if probe_enabled:
            print(f"    -> [探针] 关键计算节点: 潮汐能量得分 (energy_score) 尾部: {energy_score.tail().to_dict()}")
        # --- 4. 潮汐活跃度 (Tide Activity) ---
        ta_weights = p_conf_lt.get('tide_activity_weights', {'turnover_slope': 0.5, 'turnover_rate_level': 0.5})
        turnover_slope_score = get_adaptive_mtf_normalized_bipolar_score(turnover_slope_raw, df_index, default_weights)
        turnover_rate_level_score = get_adaptive_mtf_normalized_score(turnover_rate_level_raw, df_index, default_weights, ascending=True)
        activity_score = (
            turnover_slope_score * ta_weights.get('turnover_slope', 0.5) +
            (turnover_rate_level_score * 2 - 1) * ta_weights.get('turnover_rate_level', 0.5) # 转换为双极性
        ).clip(-1, 1)
        if probe_enabled:
            print(f"    -> [探针] 关键计算节点: 潮汐活跃度得分 (activity_score) 尾部: {activity_score.tail().to_dict()}")
        # --- 5. 潮汐结构品质 (Tide Structural Quality) ---
        tsq_weights = p_conf_lt.get('tide_structural_quality_weights', {'price_volume_entropy': 0.4, 'volatility_instability': 0.3, 'flow_credibility': 0.3})
        price_volume_entropy_score = get_adaptive_mtf_normalized_score(price_volume_entropy_raw, df_index, default_weights, ascending=False)
        volatility_instability_score = get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, default_weights, ascending=False)
        flow_credibility_score = get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, default_weights, ascending=True)
        structural_quality_score = (
            (price_volume_entropy_score * 2 - 1) * tsq_weights.get('price_volume_entropy', 0.4) +
            (volatility_instability_score * 2 - 1) * tsq_weights.get('volatility_instability', 0.3) +
            (flow_credibility_score * 2 - 1) * tsq_weights.get('flow_credibility', 0.3)
        ).clip(-1, 1)
        if probe_enabled:
            print(f"    -> [探针] 关键计算节点: 潮汐结构品质得分 (structural_quality_score) 尾部: {structural_quality_score.tail().to_dict()}")
        # --- 6. V5.0 新增: 微观流动性健康度 (Micro-Liquidity Health - MLH) ---
        mlh_weights = p_conf_lt.get('micro_liquidity_health_weights', {'bid_ask_spread': 0.4, 'market_depth_ratio': 0.3, 'order_flow_imbalance': 0.3})
        # 买卖价差越小越好 (负向)
        bid_ask_spread_score = get_adaptive_mtf_normalized_score(bid_ask_spread_raw, df_index, default_weights, ascending=False)
        # 市场深度比越大越好 (正向)
        market_depth_ratio_score = get_adaptive_mtf_normalized_score(market_depth_ratio_raw, df_index, default_weights, ascending=True)
        # 订单流失衡绝对值越小越好 (双极性，但我们希望其接近0，所以取1-abs(score))
        order_flow_imbalance_score = get_adaptive_mtf_normalized_bipolar_score(order_flow_imbalance_raw, df_index, default_weights)
        order_flow_balance_score = 1 - order_flow_imbalance_score.abs() # 转换为0-1，平衡度越高分数越高
        micro_liquidity_health_score = (
            (bid_ask_spread_score * 2 - 1) * mlh_weights.get('bid_ask_spread', 0.4) +
            (market_depth_ratio_score * 2 - 1) * mlh_weights.get('market_depth_ratio', 0.3) +
            (order_flow_balance_score * 2 - 1) * mlh_weights.get('order_flow_imbalance', 0.3)
        ).clip(-1, 1)
        if probe_enabled:
            print(f"    -> [探针] 关键计算节点: 买卖价差得分 (bid_ask_spread_score) 尾部: {bid_ask_spread_score.tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: 市场深度比得分 (market_depth_ratio_score) 尾部: {market_depth_ratio_score.tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: 订单流平衡度得分 (order_flow_balance_score) 尾部: {order_flow_balance_score.tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: 微观流动性健康度得分 (micro_liquidity_health_score) 尾部: {micro_liquidity_health_score.tail().to_dict()}")
        # --- 7. V5.0 新增: 流动性韧性 (Liquidity Resilience - LR) ---
        lr_weights = p_conf_lt.get('liquidity_resilience_weights', {'liquidity_persistence': 0.4, 'volume_profile_density': 0.3, 'vwap_deviation': 0.3})
        # 流动性持久性指数越高越好 (正向)
        liquidity_persistence_score = get_adaptive_mtf_normalized_score(liquidity_persistence_index_raw, df_index, default_weights, ascending=True)
        # 成交量分布密度越高越好 (正向)
        volume_profile_density_score = get_adaptive_mtf_normalized_score(volume_profile_density_raw, df_index, default_weights, ascending=True)
        # VWAP偏离度绝对值越小越好 (负向)
        vwap_deviation_score = get_adaptive_mtf_normalized_score(vwap_deviation_raw.abs(), df_index, default_weights, ascending=False)
        liquidity_resilience_score = (
            (liquidity_persistence_score * 2 - 1) * lr_weights.get('liquidity_persistence', 0.4) +
            (volume_profile_density_score * 2 - 1) * lr_weights.get('volume_profile_density', 0.3) +
            (vwap_deviation_score * 2 - 1) * lr_weights.get('vwap_deviation', 0.3)
        ).clip(-1, 1)
        if probe_enabled:
            print(f"    -> [探针] 关键计算节点: 流动性持久性得分 (liquidity_persistence_score) 尾部: {liquidity_persistence_score.tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: 成交量分布密度得分 (volume_profile_density_score) 尾部: {volume_profile_density_score.tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: VWAP偏离度得分 (vwap_deviation_score) 尾部: {vwap_deviation_score.tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: 流动性韧性得分 (liquidity_resilience_score) 尾部: {liquidity_resilience_score.tail().to_dict()}")
        # --- 8. 动态权重分配 (Dynamic Weight Allocation) ---
        dynamic_weights_params = p_conf_lt.get('dynamic_weight_allocation_params', {})
        base_dimension_weights = dynamic_weights_params.get('base_dimension_weights', {
            'direction': 0.15, 'energy': 0.15, 'activity': 0.15,
            'structural_quality': 0.2, 'micro_health': 0.2, 'resilience': 0.15
        })
        context_modulators = dynamic_weights_params.get('context_modulators', {})
        # 情境因子归一化
        volatility_instability_score_ctx = get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, default_weights, ascending=True)
        market_sentiment_score_ctx = get_adaptive_mtf_normalized_bipolar_score(market_sentiment_raw, df_index, default_weights)
        main_force_conviction_score_ctx = get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, default_weights)
        market_trend_slope_score_ctx = get_adaptive_mtf_normalized_bipolar_score(market_trend_slope_raw, df_index, default_weights)
        # 计算动态权重
        dynamic_weights = {}
        for dim, base_weight in base_dimension_weights.items():
            adjusted_weight = pd.Series(base_weight, index=df_index)
            for ctx_name, ctx_config in context_modulators.items():
                ctx_signal = None
                if ctx_name == 'volatility_instability':
                    ctx_signal = volatility_instability_score_ctx
                elif ctx_name == 'market_sentiment':
                    ctx_signal = market_sentiment_score_ctx
                elif ctx_name == 'main_force_conviction':
                    ctx_signal = main_force_conviction_score_ctx
                elif ctx_name == 'market_trend_slope':
                    ctx_signal = market_trend_slope_score_ctx
                if ctx_signal is not None:
                    # 使用 Sigmoid 函数将情境因子映射到 [-1, 1] 的调整系数，再乘以敏感度
                    # (ctx_signal + 1) / 2 将双极性信号映射到 [0, 1]
                    # 2 * ((ctx_signal + 1) / 2 - 0.5) = ctx_signal
                    # 调整因子 = tanh(ctx_signal * sensitivity)
                    # 调整后的权重 = base_weight * (1 + adjustment_factor * impact_factor)
                    sensitivity = ctx_config.get('sensitivity', 0.5)
                    impact_factor = ctx_config.get('impact_factor', 0.2)
                    # 针对每个维度配置的调整方向和强度
                    dim_specific_impact = ctx_config.get('dimension_impact', {}).get(dim, 0.0)
                    if dim_specific_impact != 0.0:
                        adjustment = np.tanh(ctx_signal * sensitivity) * impact_factor * dim_specific_impact
                        adjusted_weight *= (1 + adjustment)
            dynamic_weights[dim] = adjusted_weight.clip(0.01, 1.0) # 确保权重非负且不过大
        # 归一化动态权重，使其和为1
        total_dynamic_weight = sum(dynamic_weights.values())
        for dim in dynamic_weights:
            dynamic_weights[dim] /= total_dynamic_weight
        if probe_enabled:
            print(f"    -> [探针] 关键计算节点: 波动不稳定性情境得分 (volatility_instability_score_ctx) 尾部: {volatility_instability_score_ctx.tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: 市场情绪情境得分 (market_sentiment_score_ctx) 尾部: {market_sentiment_score_ctx.tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: 主力信念情境得分 (main_force_conviction_score_ctx) 尾部: {main_force_conviction_score_ctx.tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: 市场趋势斜率情境得分 (market_trend_slope_score_ctx) 尾部: {market_trend_slope_score_ctx.tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: 动态权重 (direction) 尾部: {dynamic_weights['direction'].tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: 动态权重 (energy) 尾部: {dynamic_weights['energy'].tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: 动态权重 (activity) 尾部: {dynamic_weights['activity'].tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: 动态权重 (structural_quality) 尾部: {dynamic_weights['structural_quality'].tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: 动态权重 (micro_health) 尾部: {dynamic_weights['micro_health'].tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: 动态权重 (resilience) 尾部: {dynamic_weights['resilience'].tail().to_dict()}")
        # --- 9. 基础流动性分 (融合六大维度，使用动态权重和非线性融合) ---
        # 将所有维度分数转换为 [0, 1] 范围，以便进行几何平均或幂平均
        direction_score_unipolar = (direction_score + 1) / 2
        energy_score_unipolar = (energy_score + 1) / 2
        activity_score_unipolar = (activity_score + 1) / 2
        structural_quality_score_unipolar = (structural_quality_score + 1) / 2
        micro_liquidity_health_score_unipolar = (micro_liquidity_health_score + 1) / 2
        liquidity_resilience_score_unipolar = (liquidity_resilience_score + 1) / 2
        # 避免 log(0) 错误，将0值替换为极小值
        direction_score_unipolar = direction_score_unipolar.replace(0, 1e-9)
        energy_score_unipolar = energy_score_unipolar.replace(0, 1e-9)
        activity_score_unipolar = activity_score_unipolar.replace(0, 1e-9)
        structural_quality_score_unipolar = structural_quality_score_unipolar.replace(0, 1e-9)
        micro_liquidity_health_score_unipolar = micro_liquidity_health_score_unipolar.replace(0, 1e-9)
        liquidity_resilience_score_unipolar = liquidity_resilience_score_unipolar.replace(0, 1e-9)
        # 使用加权几何平均进行非线性融合
        log_sum = (
            np.log(direction_score_unipolar) * dynamic_weights['direction'] +
            np.log(energy_score_unipolar) * dynamic_weights['energy'] +
            np.log(activity_score_unipolar) * dynamic_weights['activity'] +
            np.log(structural_quality_score_unipolar) * dynamic_weights['structural_quality'] +
            np.log(micro_liquidity_health_score_unipolar) * dynamic_weights['micro_health'] +
            np.log(liquidity_resilience_score_unipolar) * dynamic_weights['resilience']
        )
        base_tide_score_unipolar = np.exp(log_sum)
        base_tide_score = (base_tide_score_unipolar * 2 - 1).clip(-1, 1) # 转换回双极性
        if probe_enabled:
            print(f"    -> [探针] 关键计算节点: 基础流动性得分 (base_tide_score) 尾部: {base_tide_score.tail().to_dict()}")
        # --- 10. 诡道情境调制深化 (Deception Contextual Modulation) ---
        dcm_params = p_conf_lt.get('deception_context_modulator_params', {})
        base_penalty_factor = get_param_value(dcm_params.get('base_penalty_factor'), 0.5)
        sentiment_sensitivity = get_param_value(dcm_params.get('sentiment_sensitivity'), 0.3)
        conviction_sensitivity = get_param_value(dcm_params.get('conviction_sensitivity'), 0.4)
        positive_tide_penalty_multiplier = get_param_value(dcm_params.get('positive_tide_penalty_multiplier'), 1.2)
        negative_tide_bonus_multiplier = get_param_value(dcm_params.get('negative_tide_bonus_multiplier'), 0.8)
        wash_trade_score = get_adaptive_mtf_normalized_score(wash_trade_raw, df_index, default_weights, ascending=True)
        market_sentiment_score = get_adaptive_mtf_normalized_bipolar_score(market_sentiment_raw, df_index, default_weights)
        main_force_conviction_score = get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, default_weights)
        # 动态惩罚因子：当市场情绪乐观 (market_sentiment_score > 0) 且主力信念不足 (main_force_conviction_score < 0) 时，惩罚更重
        # 情绪越正，惩罚因子越大；主力信念越负（越不坚定），惩罚因子越大
        dynamic_penalty_factor = base_penalty_factor + \
                                 market_sentiment_score.clip(lower=0) * sentiment_sensitivity + \
                                 (1 - main_force_conviction_score.clip(upper=0).abs()) * conviction_sensitivity
        dynamic_penalty_factor = dynamic_penalty_factor.clip(0.1, 1.5) # 确保在合理范围
        deception_modulator = pd.Series(1.0, index=df_index)
        # 非对称调制
        # 如果是涨潮 (base_tide_score > 0)，高对倒意味着虚假繁荣，惩罚更重
        positive_tide_mask = base_tide_score > 0
        deception_modulator.loc[positive_tide_mask] = (1 - wash_trade_score.loc[positive_tide_mask] * dynamic_penalty_factor.loc[positive_tide_mask] * positive_tide_penalty_multiplier).clip(0, 1)
        # 如果是退潮 (base_tide_score <= 0)，高对倒可能意味着洗盘，惩罚适当减轻
        negative_tide_mask = base_tide_score <= 0
        deception_modulator.loc[negative_tide_mask] = (1 - wash_trade_score.loc[negative_tide_mask] * dynamic_penalty_factor.loc[negative_tide_mask] * negative_tide_bonus_multiplier).clip(0, 1)
        if probe_enabled:
            print(f"    -> [探针] 关键计算节点: 动态惩罚因子 (dynamic_penalty_factor) 尾部: {dynamic_penalty_factor.tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: 诡道情境调制器 (deception_modulator) 尾部: {deception_modulator.tail().to_dict()}")
        # --- 11. 最终流动性潮汐分数 ---
        tide_score = (base_tide_score * deception_modulator).clip(-1, 1)
        if probe_enabled:
            print(f"    -> [探针] 最终结果: 流动性潮汐分 (tide_score) 尾部: {tide_score.tail().to_dict()}")
        print("    -> [基础层] “流动性潮汐”公理诊断完成。")
        return tide_score.astype(np.float32)

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
        【V4.1 · 数据鲁棒性与调节器精细化版】基础公理五：诊断“相对强度”
        - 核心逻辑: 在融合个股“状态”与“动量”的基础上，引入“行业领导力质量”、“相对强度信念”和“价格行为对齐”
                      三大维度进行情境感知与品质校准，通过非线性乘性调节器，更精准捕捉具备高质量、可持续性的领涨龙头。
                      新增对行业领导力质量的“激活门控”，避免因原始数据长期为零而产生虚假信号。
        - A股特性: 市场的焦点是动态变化的。此升级旨在捕捉从强到更强的“领涨龙头”，而非仅仅是静态的“强者”。
        """
        print("    -> [基础层] 正在诊断“相对强度”公理 (V4.1 · 数据鲁棒性与调节器精细化版)...")
        # 获取相对强度公理的专属参数
        p_conf_rs = get_params_block(self.strategy, 'foundation_ultimate_params', {}).get('relative_strength_params', {})
        # 获取行为动态参数中的MTF归一化默认权重
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        # 获取探针启用状态
        probe_enabled = get_param_value(p_conf_rs.get('enable_probe'), False) # 保持此行，但配置中会设为False
        df_index = df.index
        # 获取参数
        state_weights = p_conf_rs.get('state_weights', {'current_rank': 0.7, 'rank_stability': 0.3})
        momentum_weights = p_conf_rs.get('momentum_weights', {'rank_slope': 0.6, 'rank_acceleration': 0.4})
        fusion_weights = p_conf_rs.get('fusion_weights', {'enhanced_state': 0.6, 'enhanced_momentum': 0.4})
        synergy_bonus_factor = get_param_value(p_conf_rs.get('synergy_bonus_factor'), 0.2)
        synergy_threshold = get_param_value(p_conf_rs.get('synergy_threshold'), 0.2)
        rank_stability_window = get_param_value(p_conf_rs.get('rank_stability_window'), 5)
        rank_acceleration_window = get_param_value(p_conf_rs.get('rank_acceleration_window'), 5)
        # 新增: 情境与品质校准参数
        slq_weights = p_conf_rs.get('sector_leadership_quality_weights', {'leader_score': 0.4, 'markup_score': 0.3, 'theme_hotness': 0.3})
        rsc_weights = p_conf_rs.get('relative_strength_conviction_weights', {'main_force_conviction': 0.4, 'flow_credibility': 0.3, 'wash_trade_penalty': 0.2, 'deception_penalty': 0.1})
        paa_weights = p_conf_rs.get('price_action_alignment_weights', {'breakout_quality': 0.4, 'trend_vitality': 0.3, 'upward_impulse_purity': 0.3})
        slq_mod_factor = get_param_value(p_conf_rs.get('slq_mod_factor'), 0.3)
        rsc_mod_factor = get_param_value(p_conf_rs.get('rsc_mod_factor'), 0.5)
        paa_mod_factor = get_param_value(p_conf_rs.get('paa_mod_factor'), 0.4)
        slq_activation_threshold = get_param_value(p_conf_rs.get('slq_activation_threshold'), 0.1)
        slq_activation_window = get_param_value(p_conf_rs.get('slq_activation_window'), 21)
        # 1. 校验所需信号 (动态构建信号名称)
        required_signals = [
            'industry_strength_rank_D',
            f'SLOPE_{rank_stability_window}_industry_strength_rank_D',
            f'ACCEL_{rank_acceleration_window}_industry_strength_rank_D',
            # 新增: 行业领导力质量所需信号
            'industry_leader_score_D',
            'industry_markup_score_D',
            'THEME_HOTNESS_SCORE_D',
            # 新增: 相对强度信念所需信号
            'main_force_conviction_index_D',
            'flow_credibility_index_D',
            'wash_trade_intensity_D',
            'deception_index_D',
            # 新增: 价格行为对齐所需信号
            'breakout_quality_score_D',
            'trend_vitality_index_D',
            'upward_impulse_purity_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_relative_strength"):
            return pd.Series(0.0, index=df.index)
        # --- 原始数据获取 ---
        industry_rank_raw = self._get_safe_series(df, 'industry_strength_rank_D', 0.5, method_name="_diagnose_axiom_relative_strength")
        industry_rank_slope_raw = self._get_safe_series(df, f'SLOPE_{rank_stability_window}_industry_strength_rank_D', 0.0, method_name="_diagnose_axiom_relative_strength")
        industry_rank_accel_raw = self._get_safe_series(df, f'ACCEL_{rank_acceleration_window}_industry_strength_rank_D', 0.0, method_name="_diagnose_axiom_relative_strength")
        
        # 新增: 获取情境与品质校准的原始数据
        industry_leader_score_raw = self._get_safe_series(df, 'industry_leader_score_D', 0.0, method_name="_diagnose_axiom_relative_strength")
        industry_markup_score_raw = self._get_safe_series(df, 'industry_markup_score_D', 0.0, method_name="_diagnose_axiom_relative_strength")
        theme_hotness_score_raw = self._get_safe_series(df, 'THEME_HOTNESS_SCORE_D', 0.0, method_name="_diagnose_axiom_relative_strength")
        main_force_conviction_raw = self._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name="_diagnose_axiom_relative_strength")
        flow_credibility_raw = self._get_safe_series(df, 'flow_credibility_index_D', 0.0, method_name="_diagnose_axiom_relative_strength")
        wash_trade_intensity_raw = self._get_safe_series(df, 'wash_trade_intensity_D', 0.0, method_name="_diagnose_axiom_relative_strength")
        deception_index_raw = self._get_safe_series(df, 'deception_index_D', 0.0, method_name="_diagnose_axiom_relative_strength")
        breakout_quality_raw = self._get_safe_series(df, 'breakout_quality_score_D', 0.0, method_name="_diagnose_axiom_relative_strength")
        trend_vitality_raw = self._get_safe_series(df, 'trend_vitality_index_D', 0.0, method_name="_diagnose_axiom_relative_strength")
        upward_impulse_purity_raw = self._get_safe_series(df, 'upward_impulse_purity_D', 0.0, method_name="_diagnose_axiom_relative_strength")

        # --- 1. 核心相对强度 (CRS) ---
        # 1.1 强度状态分 (当前排名与稳定性)
        state_score_normalized = (industry_rank_raw - 0.5) * 2
        rank_stability_score = get_adaptive_mtf_normalized_score(
            industry_rank_slope_raw.abs(), df_index, default_weights, ascending=False
        )
        enhanced_state_score = (
            state_score_normalized * state_weights.get('current_rank', 0.7) +
            rank_stability_score * state_weights.get('rank_stability', 0.3)
        ).clip(-1, 1)
        # 1.2 强度动量分 (排名斜率与加速度)
        momentum_score_normalized = get_adaptive_mtf_normalized_bipolar_score(
            industry_rank_slope_raw, df_index, default_weights
        )
        rank_acceleration_score = get_adaptive_mtf_normalized_bipolar_score(
            industry_rank_accel_raw, df_index, default_weights
        )
        enhanced_momentum_score = (
            momentum_score_normalized * momentum_weights.get('rank_slope', 0.6) +
            rank_acceleration_score * momentum_weights.get('rank_acceleration', 0.4)
        ).clip(-1, 1)
        # 1.3 基础融合与协同奖励
        core_relative_strength_score = (
            enhanced_state_score * fusion_weights.get('enhanced_state', 0.6) +
            enhanced_momentum_score * fusion_weights.get('enhanced_momentum', 0.4)
        )
        synergy_condition = (enhanced_state_score > synergy_threshold) & (enhanced_momentum_score > synergy_threshold)
        if synergy_condition.any():
            synergy_boost = (enhanced_state_score[synergy_condition] + enhanced_momentum_score[synergy_condition]) / 2 * synergy_bonus_factor
            core_relative_strength_score.loc[synergy_condition] = (
                core_relative_strength_score.loc[synergy_condition] * (1 + synergy_boost)
            )

        # --- 2. 行业领导力质量 (SLQ) ---
        # 行业领导力得分 (0-1), 行业溢价得分 (0-1), 主题热度得分 (0-1)
        slq_raw_inputs_avg = (industry_leader_score_raw + industry_markup_score_raw + theme_hotness_score_raw) / 3
        slq_active_mask = slq_raw_inputs_avg.rolling(window=slq_activation_window, min_periods=1).mean() > slq_activation_threshold
        sector_leadership_quality_score = (
            get_adaptive_mtf_normalized_score(industry_leader_score_raw, df_index, default_weights, ascending=True) * slq_weights.get('leader_score', 0.4) +
            get_adaptive_mtf_normalized_score(industry_markup_score_raw, df_index, default_weights, ascending=True) * slq_weights.get('markup_score', 0.3) +
            get_adaptive_mtf_normalized_score(theme_hotness_score_raw, df_index, default_weights, ascending=True) * slq_weights.get('theme_hotness', 0.3)
        ).clip(0, 1)
        
        # 应用激活门控
        sector_leadership_quality_score = sector_leadership_quality_score.where(slq_active_mask, 0.0)

        # --- 3. 相对强度信念 (RSC) ---
        # 主力信念 (双极), 资金流可信度 (0-1), 对倒强度 (0-1, 负向), 欺骗指数 (双极, 负向)
        main_force_conviction_score = get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, default_weights)
        flow_credibility_score = get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, default_weights, ascending=True)
        wash_trade_penalty_score = get_adaptive_mtf_normalized_score(wash_trade_intensity_raw, df_index, default_weights, ascending=True)
        deception_penalty_score = get_adaptive_mtf_normalized_bipolar_score(deception_index_raw, df_index, default_weights)
        relative_strength_conviction_score = (
            main_force_conviction_score * rsc_weights.get('main_force_conviction', 0.4) +
            flow_credibility_score * rsc_weights.get('flow_credibility', 0.3) -
            wash_trade_penalty_score * rsc_weights.get('wash_trade_penalty', 0.2) -
            deception_penalty_score.clip(lower=0) * rsc_weights.get('deception_penalty', 0.1)
        ).clip(-1, 1)

        # --- 4. 价格行为对齐 (PAA) ---
        # 突破质量 (0-1), 趋势活力 (0-1), 上涨脉冲纯度 (0-1)
        price_action_alignment_score = (
            get_adaptive_mtf_normalized_score(breakout_quality_raw, df_index, default_weights, ascending=True) * paa_weights.get('breakout_quality', 0.4) +
            get_adaptive_mtf_normalized_score(trend_vitality_raw, df_index, default_weights, ascending=True) * paa_weights.get('trend_vitality', 0.3) +
            get_adaptive_mtf_normalized_score(upward_impulse_purity_raw, df_index, default_weights, ascending=True) * paa_weights.get('upward_impulse_purity', 0.3)
        ).clip(0, 1)

        # --- 5. 最终融合: 核心相对强度 + 情境与品质校准 ---
        slq_modulator = 1 + (sector_leadership_quality_score * 2 - 1) * slq_mod_factor
        rsc_modulator = 1 + relative_strength_conviction_score * rsc_mod_factor
        paa_modulator = 1 + (price_action_alignment_score * 2 - 1) * paa_mod_factor
        slq_modulator = slq_modulator.clip(lower=0.5)
        rsc_modulator = rsc_modulator.clip(lower=0.5)
        paa_modulator = paa_modulator.clip(lower=0.5)
        relative_strength_score = (
            core_relative_strength_score * slq_modulator * rsc_modulator * paa_modulator
        )
        # 确保最终分数在 [-1, 1] 范围内
        relative_strength_score = relative_strength_score.clip(-1, 1).astype(np.float32)
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



