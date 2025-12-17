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
        【V4.0 · 深度体质诊断版】基础公理一：诊断“市场体质”
        - 核心逻辑: 融合均线骨架质量(MSQ)、筹码新陈代谢健康度(TH)、市场信念强度(MCS)以及免疫韧性与抗压能力(IR)。
                      通过多维度、更精细的原始数据，并优化融合逻辑，以更全面、更准确地评估市场体质。
                      引入“短板效应”惩罚和“协同奖励”机制，使评估更贴近A股博弈特性。
        - A股特性: 健康的上涨不仅结构稳固、换手温和，更应具备强大的下跌抵抗能力和主力资金的真实信念。
                    此升级旨在识别这种“抗揍”且有“内生动力”的健康体质。
        """
        print("    -> [基础层] 正在诊断“市场体质”公理 (V4.0 · 深度体质诊断版)...") # 修改: 更新版本描述
        # 获取市场体质公理的专属参数
        p_conf_mc = get_params_block(self.strategy, 'foundation_ultimate_params', {}).get('market_constitution_params', {})
        # 获取行为动态参数中的MTF归一化默认权重
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        # 获取探针启用状态
        probe_enabled = get_param_value(p_conf_mc.get('enable_probe'), True) # 启用探针

        df_index = df.index

        # 获取参数
        ma_periods = p_conf_mc.get('ma_periods', [5, 13, 21, 55])
        msq_weights = p_conf_mc.get('ma_structure_quality_weights', {'macd_score': 0.2, 'alignment_score': 0.3, 'slope_score': 0.3, 'dma_slope_score': 0.1, 'orderliness_score': 0.05, 'tension_score': 0.05})
        th_weights = p_conf_mc.get('turnover_health_weights', {'turnover_rate': 0.5, 'volume_burstiness': 0.2, 'constructive_turnover': 0.3})
        mcs_weights = p_conf_mc.get('market_conviction_strength_weights', {'trend_alignment': 0.4, 'main_force_conviction': 0.3, 'flow_credibility': 0.3})
        ir_weights = p_conf_mc.get('immune_resilience_weights', {'dip_absorption': 0.4, 'pressure_rejection': 0.2, 'chip_health': 0.2, 'capitulation_absorption': 0.2})
        final_fusion_weights = p_conf_mc.get('final_fusion_weights', {'msq': 0.3, 'th': 0.2, 'mcs': 0.3, 'ir': 0.2})
        short_board_penalty_threshold = get_param_value(p_conf_mc.get('short_board_penalty_threshold'), 0.2)
        short_board_penalty_factor = get_param_value(p_conf_mc.get('short_board_penalty_factor'), 0.5)
        synergy_bonus_threshold = get_param_value(p_conf_mc.get('synergy_bonus_threshold'), 0.8)
        synergy_bonus_factor = get_param_value(p_conf_mc.get('synergy_bonus_factor'), 0.2)

        # 1. 校验所需信号 (动态构建信号名称)
        required_signals = [
            'MACDh_13_34_8_D', 'SLOPE_5_DMA_D', 'turnover_rate_f_D', 'trend_alignment_index_D',
            'dip_absorption_power_D',
            # 新增 MSQ 所需信号
            'MA_POTENTIAL_ORDERLINESS_SCORE_D',
            'MA_POTENTIAL_TENSION_INDEX_D',
            # 新增 TH 所需信号
            'volume_burstiness_index_D',
            'constructive_turnover_ratio_D',
            # 新增 MCS 所需信号
            'main_force_conviction_index_D',
            'flow_credibility_index_D',
            # 新增 IR 所需信号
            'pressure_rejection_strength_D',
            'chip_health_score_D',
            'capitulation_absorption_index_D'
        ]
        required_signals.extend([f'EMA_{p}_D' for p in ma_periods])
        required_signals.extend([f'SLOPE_{p}_EMA_{p}_D' for p in ma_periods])

        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_market_constitution"):
            return pd.Series(0.0, index=df.index)

        # --- 原始数据获取 ---
        # 沿用 V3.0 原始数据
        macd_h_raw = self._get_safe_series(df, 'MACDh_13_34_8_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        dma_slope_raw = self._get_safe_series(df, 'SLOPE_5_DMA_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        turnover_rate_raw = self._get_safe_series(df, 'turnover_rate_f_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        trend_alignment_raw = self._get_safe_series(df, 'trend_alignment_index_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        dip_absorption_raw = self._get_safe_series(df, 'dip_absorption_power_D', 0.0, method_name="_diagnose_axiom_market_constitution")

        # 新增原始数据
        ma_orderliness_raw = self._get_safe_series(df, 'MA_POTENTIAL_ORDERLINESS_SCORE_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        ma_tension_raw = self._get_safe_series(df, 'MA_POTENTIAL_TENSION_INDEX_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        volume_burstiness_raw = self._get_safe_series(df, 'volume_burstiness_index_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        constructive_turnover_raw = self._get_safe_series(df, 'constructive_turnover_ratio_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        main_force_conviction_raw = self._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        flow_credibility_raw = self._get_safe_series(df, 'flow_credibility_index_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        pressure_rejection_raw = self._get_safe_series(df, 'pressure_rejection_strength_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        chip_health_raw = self._get_safe_series(df, 'chip_health_score_D', 0.0, method_name="_diagnose_axiom_market_constitution")
        capitulation_absorption_raw = self._get_safe_series(df, 'capitulation_absorption_index_D', 0.0, method_name="_diagnose_axiom_market_constitution")

        if probe_enabled:
            print(f"    -> [探针] 原始数据: MACDh_13_34_8_D 尾部: {macd_h_raw.tail().to_dict()}")
            print(f"    -> [探针] 原始数据: SLOPE_5_DMA_D 尾部: {dma_slope_raw.tail().to_dict()}")
            print(f"    -> [探针] 原始数据: turnover_rate_f_D 尾部: {turnover_rate_raw.tail().to_dict()}")
            print(f"    -> [探针] 原始数据: trend_alignment_index_D 尾部: {trend_alignment_raw.tail().to_dict()}")
            print(f"    -> [探针] 原始数据: dip_absorption_power_D 尾部: {dip_absorption_raw.tail().to_dict()}")
            print(f"    -> [探针] 原始数据: MA_POTENTIAL_ORDERLINESS_SCORE_D 尾部: {ma_orderliness_raw.tail().to_dict()}")
            print(f"    -> [探针] 原始数据: MA_POTENTIAL_TENSION_INDEX_D 尾部: {ma_tension_raw.tail().to_dict()}")
            print(f"    -> [探针] 原始数据: volume_burstiness_index_D 尾部: {volume_burstiness_raw.tail().to_dict()}")
            print(f"    -> [探针] 原始数据: constructive_turnover_ratio_D 尾部: {constructive_turnover_raw.tail().to_dict()}")
            print(f"    -> [探针] 原始数据: main_force_conviction_index_D 尾部: {main_force_conviction_raw.tail().to_dict()}")
            print(f"    -> [探针] 原始数据: flow_credibility_index_D 尾部: {flow_credibility_raw.tail().to_dict()}")
            print(f"    -> [探针] 原始数据: pressure_rejection_strength_D 尾部: {pressure_rejection_raw.tail().to_dict()}")
            print(f"    -> [探针] 原始数据: chip_health_score_D 尾部: {chip_health_raw.tail().to_dict()}")
            print(f"    -> [探针] 原始数据: capitulation_absorption_index_D 尾部: {capitulation_absorption_raw.tail().to_dict()}")
            for p in ma_periods:
                print(f"    -> [探针] 原始数据: EMA_{p}_D 尾部: {self._get_safe_series(df, f'EMA_{p}_D', 0.0, method_name='_diagnose_axiom_market_constitution').tail().to_dict()}")
                print(f"    -> [探针] 原始数据: SLOPE_{p}_EMA_{p}_D 尾部: {self._get_safe_series(df, f'SLOPE_{p}_EMA_{p}_D', 0.0, method_name='_diagnose_axiom_market_constitution').tail().to_dict()}")

        # --- 1. 均线骨架质量 (MA Structure Quality - MSQ) ---
        macd_score = get_adaptive_mtf_normalized_bipolar_score(macd_h_raw, df_index, default_weights)
        
        # 均线排列得分
        bull_alignment_scores = []
        for i in range(len(ma_periods) - 1):
            ema_current = self._get_safe_series(df, f'EMA_{ma_periods[i]}_D', 0.0, method_name="_diagnose_axiom_market_constitution")
            ema_next = self._get_safe_series(df, f'EMA_{ma_periods[i+1]}_D', 0.0, method_name="_diagnose_axiom_market_constitution")
            bull_alignment_scores.append((ema_current > ema_next).astype(float).values)
        alignment_score = np.mean(bull_alignment_scores, axis=0) if bull_alignment_scores else np.full(len(df_index), 0.5)
        alignment_bipolar = (pd.Series(alignment_score, index=df_index) - 0.5) * 2 # 归一化到 [-1, 1]

        # 均线斜率得分
        slope_scores = []
        for p in ma_periods:
            slope_raw = self._get_safe_series(df, f'SLOPE_{p}_EMA_{p}_D', 0.0, method_name="_diagnose_axiom_market_constitution")
            slope_scores.append(get_adaptive_mtf_normalized_bipolar_score(slope_raw, df_index, default_weights).values)
        avg_slope_bipolar = pd.Series(np.mean(slope_scores, axis=0), index=df_index)

        dma_slope_score = get_adaptive_mtf_normalized_bipolar_score(dma_slope_raw, df_index, default_weights)

        # 新增: 均线有序性得分 (0-1)
        ma_orderliness_score = get_adaptive_mtf_normalized_score(ma_orderliness_raw, df_index, default_weights, ascending=True)
        # 新增: 均线张力得分 (0-1)
        ma_tension_score = get_adaptive_mtf_normalized_score(ma_tension_raw, df_index, default_weights, ascending=True)

        ma_structure_quality_score = (
            macd_score * msq_weights.get('macd_score', 0.2) +
            alignment_bipolar * msq_weights.get('alignment_score', 0.3) +
            avg_slope_bipolar * msq_weights.get('slope_score', 0.3) +
            dma_slope_score * msq_weights.get('dma_slope_score', 0.1) +
            ma_orderliness_score * msq_weights.get('orderliness_score', 0.05) * 2 - 1 + # 归一化到 [-1, 1]
            ma_tension_score * msq_weights.get('tension_score', 0.05) * 2 - 1 # 归一化到 [-1, 1]
        ).clip(-1, 1)

        if probe_enabled:
            print(f"    -> [探针] 关键计算节点: MACD得分 (macd_score) 尾部: {macd_score.tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: 均线排列得分 (alignment_bipolar) 尾部: {alignment_bipolar.tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: 均线斜率得分 (avg_slope_bipolar) 尾部: {avg_slope_bipolar.tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: DMA斜率得分 (dma_slope_score) 尾部: {dma_slope_score.tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: 均线有序性得分 (ma_orderliness_score) 尾部: {ma_orderliness_score.tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: 均线张力得分 (ma_tension_score) 尾部: {ma_tension_score.tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: 均线骨架质量分 (ma_structure_quality_score) 尾部: {ma_structure_quality_score.tail().to_dict()}")

        # --- 2. 筹码新陈代谢健康度 (Turnover Health - TH) ---
        # 换手率健康度 (越低越健康，但不能过低)
        turnover_health_score = get_adaptive_mtf_normalized_score(turnover_rate_raw, df_index, ascending=False, tf_weights=default_weights)
        # 新增: 成交量爆发性 (越低越健康)
        volume_burstiness_score = get_adaptive_mtf_normalized_score(volume_burstiness_raw, df_index, default_weights, ascending=False)
        # 新增: 建设性换手率 (越高越健康)
        constructive_turnover_score = get_adaptive_mtf_normalized_score(constructive_turnover_raw, df_index, default_weights, ascending=True)

        turnover_health_score = (
            turnover_health_score * th_weights.get('turnover_rate', 0.5) +
            volume_burstiness_score * th_weights.get('volume_burstiness', 0.2) +
            constructive_turnover_score * th_weights.get('constructive_turnover', 0.3)
        ).clip(0, 1)

        if probe_enabled:
            print(f"    -> [探针] 关键计算节点: 换手率健康度得分 (turnover_health_score) 尾部: {turnover_health_score.tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: 成交量爆发性得分 (volume_burstiness_score) 尾部: {volume_burstiness_score.tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: 建设性换手率得分 (constructive_turnover_score) 尾部: {constructive_turnover_score.tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: 筹码新陈代谢健康度分 (turnover_health_score) 尾部: {turnover_health_score.tail().to_dict()}")

        # --- 3. 市场信念强度 (Market Conviction Strength - MCS) ---
        conviction_score_v3 = get_adaptive_mtf_normalized_score(trend_alignment_raw, df_index, ascending=True, tf_weights=default_weights)
        # 新增: 主力信念 (双极)
        main_force_conviction_score = get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, default_weights)
        # 新增: 资金流可信度 (0-1)
        flow_credibility_score = get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, default_weights, ascending=True)

        market_conviction_strength_score = (
            conviction_score_v3 * mcs_weights.get('trend_alignment', 0.4) +
            main_force_conviction_score * mcs_weights.get('main_force_conviction', 0.3) +
            flow_credibility_score * mcs_weights.get('flow_credibility', 0.3) * 2 - 1 # 归一化到 [-1, 1]
        ).clip(-1, 1)

        if probe_enabled:
            print(f"    -> [探针] 关键计算节点: 趋势一致性得分 (conviction_score_v3) 尾部: {conviction_score_v3.tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: 主力信念得分 (main_force_conviction_score) 尾部: {main_force_conviction_score.tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: 资金流可信度得分 (flow_credibility_score) 尾部: {flow_credibility_score.tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: 市场信念强度分 (market_conviction_strength_score) 尾部: {market_conviction_strength_score.tail().to_dict()}")

        # --- 4. 免疫韧性与抗压能力 (Immune Resilience - IR) ---
        dip_absorption_score = get_adaptive_mtf_normalized_score(dip_absorption_raw, df_index, ascending=True, tf_weights=default_weights)
        # 新增: 压力拒绝强度 (越高越好)
        pressure_rejection_score = get_adaptive_mtf_normalized_score(pressure_rejection_raw, df_index, default_weights, ascending=True)
        # 新增: 筹码健康度 (越高越好)
        chip_health_score = get_adaptive_mtf_normalized_score(chip_health_raw, df_index, default_weights, ascending=True)
        # 新增: 投降吸收指数 (越高越好)
        capitulation_absorption_score = get_adaptive_mtf_normalized_score(capitulation_absorption_raw, df_index, default_weights, ascending=True)

        immune_resilience_score = (
            dip_absorption_score * ir_weights.get('dip_absorption', 0.4) +
            pressure_rejection_score * ir_weights.get('pressure_rejection', 0.2) +
            chip_health_score * ir_weights.get('chip_health', 0.2) +
            capitulation_absorption_score * ir_weights.get('capitulation_absorption', 0.2)
        ).clip(0, 1)

        if probe_enabled:
            print(f"    -> [探针] 关键计算节点: 下跌承接力得分 (dip_absorption_score) 尾部: {dip_absorption_score.tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: 压力拒绝强度得分 (pressure_rejection_score) 尾部: {pressure_rejection_score.tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: 筹码健康度得分 (chip_health_score) 尾部: {chip_health_score.tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: 投降吸收指数得分 (capitulation_absorption_score) 尾部: {capitulation_absorption_score.tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: 免疫韧性与抗压能力分 (immune_resilience_score) 尾部: {immune_resilience_score.tail().to_dict()}")

        # --- 5. 最终融合: 四大核心维度 + 非线性调节 ---
        # 将所有维度归一化到 [0, 1] 范围，以便进行短板效应和协同奖励
        msq_unipolar = (ma_structure_quality_score + 1) / 2
        mcs_unipolar = (market_conviction_strength_score + 1) / 2

        raw_constitution_score = (
            ma_structure_quality_score * final_fusion_weights.get('msq', 0.3) +
            turnover_health_score * final_fusion_weights.get('th', 0.2) * 2 - 1 + # TH是[0,1]，转为[-1,1]
            market_conviction_strength_score * final_fusion_weights.get('mcs', 0.3) +
            immune_resilience_score * final_fusion_weights.get('ir', 0.2) * 2 - 1 # IR是[0,1]，转为[-1,1]
        ).clip(-1, 1)

        # 短板效应惩罚: 如果任何一个核心维度（归一化到[0,1]）低于阈值，则进行惩罚
        min_score_components = pd.DataFrame({
            'msq': msq_unipolar,
            'th': turnover_health_score,
            'mcs': mcs_unipolar,
            'ir': immune_resilience_score
        }).min(axis=1)

        short_board_penalty = pd.Series(1.0, index=df_index)
        penalty_condition = min_score_components < short_board_penalty_threshold
        if penalty_condition.any():
            # 惩罚因子与短板程度成正比，并由 short_board_penalty_factor 控制强度
            penalty_amount = (short_board_penalty_threshold - min_score_components[penalty_condition]) / short_board_penalty_threshold * short_board_penalty_factor
            short_board_penalty.loc[penalty_condition] = (1 - penalty_amount).clip(lower=0.0) # 惩罚因子至少为0

        # 协同奖励: 如果所有核心维度（归一化到[0,1]）都高于阈值，则给予奖励
        synergy_condition = (msq_unipolar > synergy_bonus_threshold) & \
                            (turnover_health_score > synergy_bonus_threshold) & \
                            (mcs_unipolar > synergy_bonus_threshold) & \
                            (immune_resilience_score > synergy_bonus_threshold)
        
        synergy_bonus = pd.Series(0.0, index=df_index)
        if synergy_condition.any():
            # 奖励因子与平均表现成正比，并由 synergy_bonus_factor 控制强度
            avg_high_score = pd.DataFrame({
                'msq': msq_unipolar[synergy_condition],
                'th': turnover_health_score[synergy_condition],
                'mcs': mcs_unipolar[synergy_condition],
                'ir': immune_resilience_score[synergy_condition]
            }).mean(axis=1)
            synergy_bonus.loc[synergy_condition] = (avg_high_score - synergy_bonus_threshold) / (1 - synergy_bonus_threshold) * synergy_bonus_factor

        # 应用非线性调节
        constitution_score = raw_constitution_score * short_board_penalty * (1 + synergy_bonus)

        # 确保最终分数在 [-1, 1] 范围内
        constitution_score = constitution_score.clip(-1, 1).astype(np.float32)

        if probe_enabled:
            print(f"    -> [探针] 关键计算节点: 原始体质分 (raw_constitution_score) 尾部: {raw_constitution_score.tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: 最小分量 (min_score_components) 尾部: {min_score_components.tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: 短板惩罚条件 (penalty_condition) 尾部: {penalty_condition.tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: 短板惩罚因子 (short_board_penalty) 尾部: {short_board_penalty.tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: 协同奖励条件 (synergy_condition) 尾部: {synergy_condition.tail().to_dict()}")
            print(f"    -> [探针] 关键计算节点: 协同奖励因子 (synergy_bonus) 尾部: {synergy_bonus.tail().to_dict()}")
            print(f"    -> [探针] 最终结果: 市场体质分 (constitution_score) 尾部: {constitution_score.tail().to_dict()}")
        print("    -> [基础层] “市场体质”公理诊断完成。")
        return constitution_score


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



