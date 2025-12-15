import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, Union
from strategies.trend_following import utils
from strategies.trend_following.utils import (
    get_params_block, get_param_value, get_adaptive_mtf_normalized_score, 
    get_adaptive_mtf_normalized_bipolar_score, get_robust_bipolar_normalized_score, normalize_score
)

class ChipIntelligence:
    def __init__(self, strategy_instance):
        """
        【V2.1 · 依赖注入版】
        - 核心升级: 新增 self.bipolar_sensitivity 属性，从策略配置中读取归一化所需的敏感度参数。
                     解决了在调用外部归一化工具时缺少依赖参数的问题。
        """
        self.strategy = strategy_instance
        self.params = get_params_block(self.strategy, 'chip_intelligence_params', {})
        self.score_type_map = get_params_block(self.strategy, 'score_type_map', {})
        # 注入双极归一化所需的敏感度参数
        process_params = get_params_block(self.strategy, 'process_intelligence_params', {})
        self.bipolar_sensitivity = get_param_value(process_params.get('bipolar_sensitivity'), 1.0)

    def _get_safe_series(self, df: pd.DataFrame, data_source: Union[pd.DataFrame, Dict[str, pd.Series]], column_name: str, default_value: Any = 0.0, method_name: str = "未知方法") -> pd.Series:
        """
        【V2.0 · 上下文修复版】安全地从DataFrame或字典中获取Series，如果不存在则打印警告并返回默认Series。
        - 核心修复: 接收 df 参数，并使用其索引创建默认 Series，确保上下文一致。
        """
        df_index = df.index # 使用传入的 df.index
        if isinstance(data_source, pd.DataFrame):
            if column_name not in data_source.columns:
                print(f"    -> [筹码情报警告] 方法 '{method_name}' 缺少DataFrame数据 '{column_name}'，使用默认值 {default_value}。")
                return pd.Series(default_value, index=df_index)
            return data_source[column_name]
        elif isinstance(data_source, dict):
            if column_name not in data_source:
                print(f"    -> [筹码情报警告] 方法 '{method_name}' 缺少字典数据 '{column_name}'，使用默认值 {default_value}。")
                return pd.Series(default_value, index=df_index)
            series = data_source[column_name]
            if isinstance(series, pd.Series):
                return series.reindex(df_index, fill_value=default_value)
            else:
                return pd.Series(series, index=df_index)
        else:
            print(f"    -> [筹码情报警告] 方法 '{method_name}' 接收到未知数据源类型 {type(data_source)}，无法获取 '{column_name}'，使用默认值 {default_value}。")
            return pd.Series(default_value, index=df_index)

    def _validate_required_signals(self, df: pd.DataFrame, required_signals: list, method_name: str) -> bool:
        """
        【V1.0 · 战前情报校验】内部辅助方法，用于在方法执行前验证所有必需的数据信号是否存在。
        """
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            print(f"    -> [筹码情报校验] 方法 '{method_name}' 启动失败：缺少核心信号 {missing_signals}。")
            return False
        return True

    def run_chip_intelligence_command(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V19.0 · 诡道反吸版】筹码情报总指挥
        - 核心升维: 升级“吸筹回声”信号到 V2.0，严格遵循纯筹码原则，深度融入诡道博弈特性。
        """
        print("启动【V19.0 · 诡道反吸版】筹码情报分析...")
        all_chip_states = {}
        periods = [5, 13, 21, 55]
        holder_sentiment_scores = self._diagnose_axiom_holder_sentiment(df, periods)
        divergence_scores = self._diagnose_axiom_divergence(df, periods)
        all_chip_states['SCORE_CHIP_AXIOM_HOLDER_SENTIMENT'] = holder_sentiment_scores
        all_chip_states['SCORE_CHIP_AXIOM_DIVERGENCE'] = divergence_scores
        strategic_posture = self._diagnose_strategic_posture(df)
        all_chip_states['SCORE_CHIP_STRATEGIC_POSTURE'] = strategic_posture
        battlefield_geography = self._diagnose_battlefield_geography(df)
        all_chip_states['SCORE_CHIP_BATTLEFIELD_GEOGRAPHY'] = battlefield_geography
        chip_trend_momentum_scores = self._diagnose_axiom_trend_momentum(df, periods, strategic_posture, battlefield_geography, holder_sentiment_scores)
        all_chip_states['SCORE_CHIP_AXIOM_TREND_MOMENTUM'] = chip_trend_momentum_scores
        print(f"    -> [筹码情报校验] 计算“筹码趋势动量(SCORE_CHIP_AXIOM_TREND_MOMENTUM)” 分数：{chip_trend_momentum_scores.mean():.4f}")
        historical_potential = self._diagnose_axiom_historical_potential(df)
        all_chip_states['SCORE_CHIP_AXIOM_HISTORICAL_POTENTIAL'] = historical_potential
        print(f"    -> [筹码情报校验] 计算“筹码历史潜力(SCORE_CHIP_AXIOM_HISTORICAL_POTENTIAL)” 分数：{historical_potential.mean():.4f}")
        absorption_echo = self._diagnose_absorption_echo(df, divergence_scores)
        all_chip_states['SCORE_CHIP_OPP_ABSORPTION_ECHO'] = absorption_echo
        distribution_whisper = self._diagnose_distribution_whisper(df, divergence_scores)
        all_chip_states['SCORE_CHIP_RISK_DISTRIBUTION_WHISPER'] = distribution_whisper
        coherent_drive = self._diagnose_structural_consensus(df, battlefield_geography, holder_sentiment_scores)
        all_chip_states['SCORE_CHIP_COHERENT_DRIVE'] = coherent_drive
        print(f"    -> [筹码情报校验] 计算“筹码一致驱动(SCORE_CHIP_COHERENT_DRIVE)” 分数：{coherent_drive.mean():.4f}")
        tactical_exchange = self._diagnose_tactical_exchange(df, battlefield_geography)
        all_chip_states['SCORE_CHIP_TACTICAL_EXCHANGE'] = tactical_exchange
        # [修改代码行] 传递 holder_sentiment_scores
        strategic_tactical_harmony = self._diagnose_strategic_tactical_harmony(df, strategic_posture, tactical_exchange, holder_sentiment_scores)
        all_chip_states['SCORE_CHIP_STRATEGIC_TACTICAL_HARMONY'] = strategic_tactical_harmony
        harmony_inflection = self._diagnose_harmony_inflection(df, strategic_tactical_harmony)
        all_chip_states['SCORE_CHIP_HARMONY_INFLECTION'] = harmony_inflection
        print(f"【V19.0 · 诡道反吸版】分析完成，生成 {len(all_chip_states)} 个筹码原子信号。")
        return all_chip_states

    def _diagnose_strategic_posture(self, df: pd.DataFrame) -> pd.Series:
        """
        【V9.1 · 诡道情境自适应版】诊断主力的综合战略态势。
        - 核心升级1: 诡道博弈深度融合与情境调制：引入主力信念和筹码健康度作为情境，动态调整欺骗指数和对倒强度的影响，实现非对称调制，更精准识别和应对主力诡道博弈。
        - 核心升级2: 动态权重自适应：根据筹码波动不稳定性、筹码健康度斜率等情境因子，动态调整基础态势、速度和加速度的融合权重，使信号自适应市场动态。
        - 核心升级3: 维度间非线性互动增强：引入“协同/冲突”因子，评估阵型部署、指挥官决心、战场控制各维度之间的非线性互动，提高信号的敏感性和准确性。
        - 核心升级4: 全局情境调制器：引入筹码健康度、市场情绪作为全局调制器，对最终战略态势分数进行校准，提高信号在不同市场情境下的可靠性。
        - 核心升级5: 新增筹码指标整合：
            - 诱多/诱空欺骗强度 (`deception_lure_long_intensity_D`, `deception_lure_short_intensity_D`) 进一步精细化诡道调制。
            - 主力成本区买卖意图 (`mf_cost_zone_buy_intent_D`, `mf_cost_zone_sell_intent_D`) 增强指挥官决心维度。
            - 隐蔽派发信号 (`covert_distribution_signal_D`) 作为负向调制器。
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        print("    -> [筹码层] 正在诊断“战略态势 (V9.1 · 诡道情境自适应版)”...")
        # --- 探针: 原始输入 ---
        df_index = df.index
        probe_date = None
        # --- 参数加载 ---
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        sp_params = get_param_value(p_conf.get('strategic_posture_params'), {})
        # V9.0 诡道博弈深度融合参数
        deception_fusion_weights = get_param_value(sp_params.get('deception_fusion_weights'), {"bear_trap_positive": 0.6, "bull_trap_negative": 0.2, "wash_trade_negative": 0.2})
        deception_context_mod_enabled = get_param_value(sp_params.get('deception_context_mod_enabled'), True)
        deception_conviction_threshold = get_param_value(sp_params.get('deception_conviction_threshold'), 0.2)
        deception_health_threshold = get_param_value(sp_params.get('deception_health_threshold'), 0.5)
        deception_boost_factor = get_param_value(sp_params.get('deception_boost_factor'), 0.5)
        deception_penalty_factor = get_param_value(sp_params.get('deception_penalty_factor'), 0.7)
        wash_trade_penalty_factor = get_param_value(sp_params.get('wash_trade_penalty_factor'), 0.3)
        # [新增代码行] V9.1 新增欺骗指标参数
        deception_lure_long_penalty_factor = get_param_value(sp_params.get('deception_lure_long_penalty_factor'), 0.3)
        deception_lure_short_boost_factor = get_param_value(sp_params.get('deception_lure_short_boost_factor'), 0.3)
        # [新增代码行] V9.1 新增主力成本区意图参数
        mf_cost_zone_buy_intent_weight = get_param_value(sp_params.get('mf_cost_zone_buy_intent_weight'), 0.1)
        mf_cost_zone_sell_intent_weight = get_param_value(sp_params.get('mf_cost_zone_sell_intent_weight'), 0.1)
        # [新增代码行] V9.1 新增隐蔽派发信号参数
        covert_distribution_penalty_factor = get_param_value(sp_params.get('covert_distribution_penalty_factor'), 0.2)
        # V9.0 动态权重自适应参数
        dynamic_fusion_weights_base = get_param_value(sp_params.get('dynamic_fusion_weights_base'), {'base_score': 0.6, 'velocity': 0.2, 'acceleration': 0.2})
        dynamic_weight_modulator_signal_1_name = get_param_value(sp_params.get('dynamic_weight_modulator_signal_1'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        dynamic_weight_modulator_signal_2_name = get_param_value(sp_params.get('dynamic_weight_modulator_signal_2'), 'SLOPE_5_chip_health_score_D')
        dynamic_weight_sensitivity_volatility = get_param_value(sp_params.get('dynamic_weight_sensitivity_volatility'), 0.4)
        dynamic_weight_sensitivity_health_slope = get_param_value(sp_params.get('dynamic_weight_sensitivity_health_slope'), 0.3)
        # V9.0 维度间非线性互动增强参数
        inter_dimension_interaction_enabled = get_param_value(sp_params.get('inter_dimension_interaction_enabled'), True)
        synergy_bonus_factor = get_param_value(sp_params.get('synergy_bonus_factor'), 0.15)
        conflict_penalty_factor = get_param_value(sp_params.get('conflict_penalty_factor'), 0.2)
        # V9.0 全局情境调制器参数
        global_context_modulator_enabled = get_param_value(sp_params.get('global_context_modulator_enabled'), True)
        global_context_signal_1_name = get_param_value(sp_params.get('global_context_signal_1'), 'chip_health_score_D')
        global_context_signal_2_name = get_param_value(sp_params.get('global_context_signal_2'), 'market_sentiment_score_D')
        global_context_sensitivity_health = get_param_value(sp_params.get('global_context_sensitivity_health'), 0.5)
        global_context_sensitivity_sentiment = get_param_value(sp_params.get('global_context_sensitivity_sentiment'), 0.3)
        smoothing_ema_span = get_param_value(sp_params.get('smoothing_ema_span'), 5)
        # --- 信号依赖校验 ---
        required_signals = [
            'cost_gini_coefficient_D', 'covert_accumulation_signal_D', 'peak_exchange_purity_D',
            'main_force_cost_advantage_D', 'control_solidity_index_D', 'SLOPE_5_main_force_conviction_index_D',
            'floating_chip_cleansing_efficiency_D', 'dominant_peak_solidity_D',
            'deception_index_D', 'wash_trade_intensity_D',
            'main_force_conviction_index_D', 'chip_health_score_D', # V9.0 新增诡道情境依赖
            dynamic_weight_modulator_signal_1_name, dynamic_weight_modulator_signal_2_name, # V9.0 新增动态权重依赖
            global_context_signal_1_name, global_context_signal_2_name, # V9.0 新增全局情境依赖
            # [新增代码行] V9.1 新增筹码指标
            'deception_lure_long_intensity_D', 'deception_lure_short_intensity_D',
            'mf_cost_zone_buy_intent_D', 'mf_cost_zone_sell_intent_D', 'covert_distribution_signal_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_strategic_posture"):
            return pd.Series(0.0, index=df.index)
        # --- 原始数据获取 (用于探针和计算) ---
        cost_gini_coefficient_raw = self._get_safe_series(df, df, 'cost_gini_coefficient_D', 0.5, method_name="_diagnose_strategic_posture")
        covert_accumulation_raw = self._get_safe_series(df, df, 'covert_accumulation_signal_D', 0.0, method_name="_diagnose_strategic_posture")
        peak_exchange_purity_raw = self._get_safe_series(df, df, 'peak_exchange_purity_D', 0.0, method_name="_diagnose_strategic_posture")
        main_force_cost_advantage_raw = self._get_safe_series(df, df, 'main_force_cost_advantage_D', 0.0, method_name="_diagnose_strategic_posture")
        control_solidity_index_raw = self._get_safe_series(df, df, 'control_solidity_index_D', 0.0, method_name="_diagnose_strategic_posture")
        conviction_slope_raw = self._get_safe_series(df, df, 'SLOPE_5_main_force_conviction_index_D', 0.0, method_name="_diagnose_strategic_posture")
        deception_index_raw = self._get_safe_series(df, df, 'deception_index_D', 0.0, method_name="_diagnose_strategic_posture")
        wash_trade_intensity_raw = self._get_safe_series(df, df, 'wash_trade_intensity_D', 0.0, method_name="_diagnose_strategic_posture")
        cleansing_efficiency_raw = self._get_safe_series(df, df, 'floating_chip_cleansing_efficiency_D', 0.0, method_name="_diagnose_strategic_posture")
        dominant_peak_solidity_raw = self._get_safe_series(df, df, 'dominant_peak_solidity_D', 0.5, method_name="_diagnose_strategic_posture")
        main_force_conviction_raw = self._get_safe_series(df, df, 'main_force_conviction_index_D', 0.0, method_name="_diagnose_strategic_posture")
        chip_health_raw = self._get_safe_series(df, df, 'chip_health_score_D', 0.0, method_name="_diagnose_strategic_posture")
        volatility_instability_raw = self._get_safe_series(df, df, dynamic_weight_modulator_signal_1_name, 0.0, method_name="_diagnose_strategic_posture")
        chip_health_slope_raw = self._get_safe_series(df, df, dynamic_weight_modulator_signal_2_name, 0.0, method_name="_diagnose_strategic_posture")
        market_sentiment_raw = self._get_safe_series(df, df, global_context_signal_2_name, 0.0, method_name="_diagnose_strategic_posture")
        # [新增代码行] V9.1 获取新增筹码指标
        deception_lure_long_intensity_raw = self._get_safe_series(df, df, 'deception_lure_long_intensity_D', 0.0, method_name="_diagnose_strategic_posture")
        deception_lure_short_intensity_raw = self._get_safe_series(df, df, 'deception_lure_short_intensity_D', 0.0, method_name="_diagnose_strategic_posture")
        mf_cost_zone_buy_intent_raw = self._get_safe_series(df, df, 'mf_cost_zone_buy_intent_D', 0.0, method_name="_diagnose_strategic_posture")
        mf_cost_zone_sell_intent_raw = self._get_safe_series(df, df, 'mf_cost_zone_sell_intent_D', 0.0, method_name="_diagnose_strategic_posture")
        covert_distribution_signal_raw = self._get_safe_series(df, df, 'covert_distribution_signal_D', 0.0, method_name="_diagnose_strategic_posture")
        # --- 维度1: 阵型部署 (Formation Deployment) ---
        concentration_level = 1 - cost_gini_coefficient_raw
        level_score = get_adaptive_mtf_normalized_bipolar_score(concentration_level, df_index, tf_weights)
        efficiency_score = (
            get_adaptive_mtf_normalized_bipolar_score(covert_accumulation_raw, df_index, tf_weights).add(1)/2 *
            get_adaptive_mtf_normalized_bipolar_score(peak_exchange_purity_raw, df_index, tf_weights).add(1)/2
        ).pow(0.5) * 2 - 1
        formation_deployment_score = (level_score.add(1)/2 * efficiency_score.add(1)/2).pow(0.5) * 2 - 1
        # --- 维度2: 指挥官决心 (Commander's Resolve) ---
        advantage_score = get_adaptive_mtf_normalized_bipolar_score(main_force_cost_advantage_raw, df_index, tf_weights)
        solidity_score = get_adaptive_mtf_normalized_bipolar_score(control_solidity_index_raw, df_index, tf_weights)
        intent_score = get_adaptive_mtf_normalized_bipolar_score(conviction_slope_raw, df_index, tf_weights)
        # [新增代码行] V9.1 整合主力成本区买卖意图
        norm_mf_cost_zone_buy_intent = get_adaptive_mtf_normalized_score(mf_cost_zone_buy_intent_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_mf_cost_zone_sell_intent = get_adaptive_mtf_normalized_score(mf_cost_zone_sell_intent_raw, df_index, ascending=True, tf_weights=tf_weights)
        # V9.0 诡道博弈深度融合与情境调制
        norm_deception_index = get_adaptive_mtf_normalized_bipolar_score(deception_index_raw, df_index, tf_weights)
        norm_wash_trade_intensity = get_adaptive_mtf_normalized_score(wash_trade_intensity_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_main_force_conviction = get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, tf_weights)
        norm_chip_health = get_adaptive_mtf_normalized_score(chip_health_raw, df_index, ascending=True, tf_weights=tf_weights)
        # [新增代码行] V9.1 归一化诱多/诱空欺骗强度
        norm_deception_lure_long = get_adaptive_mtf_normalized_score(deception_lure_long_intensity_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_deception_lure_short = get_adaptive_mtf_normalized_score(deception_lure_short_intensity_raw, df_index, ascending=True, tf_weights=tf_weights)
        deception_modulator = pd.Series(1.0, index=df_index)
        if deception_context_mod_enabled:
            # 情境：主力信念强且筹码健康
            strong_conviction_healthy_chip_mask = (norm_main_force_conviction > deception_conviction_threshold) & \
                                                  (norm_chip_health > deception_health_threshold)
            # 情境：主力信念弱或筹码不健康
            weak_conviction_unhealthy_chip_mask = (norm_main_force_conviction < -deception_conviction_threshold) | \
                                                   (norm_chip_health < (1 - deception_health_threshold))
            # 诱空（负向欺骗）在强信念健康筹码情境下，可能被视为善意洗盘，增强决心
            bear_trap_boost_mask = strong_conviction_healthy_chip_mask & (norm_deception_index < 0)
            # [修改代码行] 整合诱空欺骗强度
            deception_modulator.loc[bear_trap_boost_mask] = 1 + (norm_deception_index.loc[bear_trap_boost_mask].abs() * deception_boost_factor + \
                                                                 norm_deception_lure_short.loc[bear_trap_boost_mask] * deception_lure_short_boost_factor)
            # 诱多（正向欺骗）在任何情境下都惩罚，尤其在弱信念不健康筹码情境下更严厉
            bull_trap_penalty_mask = (norm_deception_index > 0)
            # [修改代码行] 整合诱多欺骗强度
            deception_modulator.loc[bull_trap_penalty_mask] = 1 - (norm_deception_index.loc[bull_trap_penalty_mask] * deception_penalty_factor + \
                                                                   norm_deception_lure_long.loc[bull_trap_penalty_mask] * deception_lure_long_penalty_factor)
            # 在弱信念不健康筹码情境下，诱多惩罚加倍
            deception_modulator.loc[bull_trap_penalty_mask & weak_conviction_unhealthy_chip_mask] = \
                deception_modulator.loc[bull_trap_penalty_mask & weak_conviction_unhealthy_chip_mask] - \
                (norm_deception_index.loc[bull_trap_penalty_mask & weak_conviction_unhealthy_chip_mask] * deception_penalty_factor + \
                 norm_deception_lure_long.loc[bull_trap_penalty_mask & weak_conviction_unhealthy_chip_mask] * deception_lure_long_penalty_factor)
            # 对倒行为始终惩罚，惩罚力度受信念和健康度影响
            wash_trade_penalty_mod = norm_wash_trade_intensity * wash_trade_penalty_factor
            # 在强信念健康筹码情境下，对倒惩罚略轻
            deception_modulator.loc[strong_conviction_healthy_chip_mask] = \
                deception_modulator.loc[strong_conviction_healthy_chip_mask] * (1 - wash_trade_penalty_mod.loc[strong_conviction_healthy_chip_mask] * 0.5)
            # 在弱信念不健康筹码情境下，对倒惩罚更重
            deception_modulator.loc[weak_conviction_unhealthy_chip_mask] = \
                deception_modulator.loc[weak_conviction_unhealthy_chip_mask] * (1 - wash_trade_penalty_mod.loc[weak_conviction_unhealthy_chip_mask] * 1.5)
            # 其他情况正常惩罚
            deception_modulator.loc[~(strong_conviction_healthy_chip_mask | weak_conviction_unhealthy_chip_mask)] = \
                deception_modulator.loc[~(strong_conviction_healthy_chip_mask | weak_conviction_unhealthy_chip_mask)] * (1 - wash_trade_penalty_mod.loc[~(strong_conviction_healthy_chip_mask | weak_conviction_unhealthy_chip_mask)])
            deception_modulator = deception_modulator.clip(0.1, 2.0) # 限制调制范围
        # 将诡道调制器融入指挥官决心
        commanders_resolve_score = (
            (advantage_score.add(1)/2) * (solidity_score.add(1)/2) *
            (intent_score.clip(lower=-1, upper=1).add(1)/2)
        ).pow(1/3) * 2 - 1 # 基础融合
        # [新增代码行] V9.1 整合主力成本区买卖意图到指挥官决心
        commanders_resolve_score = commanders_resolve_score + \
                                   (norm_mf_cost_zone_buy_intent * mf_cost_zone_buy_intent_weight) - \
                                   (norm_mf_cost_zone_sell_intent * mf_cost_zone_sell_intent_weight)
        commanders_resolve_score = commanders_resolve_score.clip(-1, 1) # 确保分数在 [-1, 1] 范围内
        commanders_resolve_score = commanders_resolve_score * deception_modulator.pow(np.sign(commanders_resolve_score)) # 诡道调制
        # --- 维度3: 战场控制 (Battlefield Control) ---
        cleansing_score = get_adaptive_mtf_normalized_bipolar_score(cleansing_efficiency_raw, df_index, tf_weights)
        peak_solidity_score = get_adaptive_mtf_normalized_bipolar_score(dominant_peak_solidity_raw, df_index, tf_weights)
        battlefield_control_score = (cleansing_score.add(1)/2 * peak_solidity_score.add(1)/2).pow(0.5) * 2 - 1
        # --- 基础融合 (含维度间非线性互动) ---
        base_strategic_posture_score = (
            (commanders_resolve_score.add(1)/2).pow(0.5) *
            (formation_deployment_score.add(1)/2).pow(0.3) *
            (battlefield_control_score.add(1)/2).pow(0.2)
        ).pow(1/(0.5+0.3+0.2)) * 2 - 1
        # [新增代码行] V9.1 整合隐蔽派发信号
        norm_covert_distribution_signal = get_adaptive_mtf_normalized_score(covert_distribution_signal_raw, df_index, ascending=True, tf_weights=tf_weights)
        base_strategic_posture_score = base_strategic_posture_score * (1 - norm_covert_distribution_signal * covert_distribution_penalty_factor)
        base_strategic_posture_score = base_strategic_posture_score.clip(-1, 1) # 确保分数在 [-1, 1] 范围内
        # V9.0 维度间非线性互动增强
        if inter_dimension_interaction_enabled:
            # 协同/冲突因子：评估各维度方向一致性
            synergy_factor = pd.Series(0.0, index=df_index)
            # 积极协同：三个维度都为正
            positive_synergy_mask = (formation_deployment_score > 0) & (commanders_resolve_score > 0) & (battlefield_control_score > 0)
            synergy_factor.loc[positive_synergy_mask] = synergy_bonus_factor
            # 消极协同：三个维度都为负
            negative_synergy_mask = (formation_deployment_score < 0) & (commanders_resolve_score < 0) & (battlefield_control_score < 0)
            synergy_factor.loc[negative_synergy_mask] = -synergy_bonus_factor
            # 冲突：例如，指挥官决心积极但阵型部署消极
            conflict_mask = ((formation_deployment_score > 0) & (commanders_resolve_score < 0)) | \
                            ((formation_deployment_score < 0) & (commanders_resolve_score > 0)) | \
                            ((battlefield_control_score > 0) & (commanders_resolve_score < 0)) | \
                            ((battlefield_control_score < 0) & (commanders_resolve_score > 0))
            synergy_factor.loc[conflict_mask] = -conflict_penalty_factor
            base_strategic_posture_score = np.tanh(base_strategic_posture_score + synergy_factor) # 使用tanh平滑融合协同/冲突因子
        # --- 时间序列分析 (Strategic Dynamics) ---
        smoothed_base_score = base_strategic_posture_score.ewm(span=smoothing_ema_span, adjust=False).mean()
        velocity = smoothed_base_score.diff(1).fillna(0)
        acceleration = velocity.diff(1).fillna(0)
        norm_velocity = get_adaptive_mtf_normalized_bipolar_score(velocity, df_index, tf_weights)
        norm_acceleration = get_adaptive_mtf_normalized_bipolar_score(acceleration, df_index, tf_weights)
        # V9.0 动态权重自适应
        dynamic_base_weight = pd.Series(dynamic_fusion_weights_base.get('base_score', 0.6), index=df_index)
        dynamic_velocity_weight = pd.Series(dynamic_fusion_weights_base.get('velocity', 0.2), index=df_index)
        dynamic_acceleration_weight = pd.Series(dynamic_fusion_weights_base.get('acceleration', 0.2), index=df_index)
        norm_volatility_instability = get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=True, tf_weights=tf_weights) # 波动性越高，值越大
        norm_chip_health_slope = get_adaptive_mtf_normalized_bipolar_score(chip_health_slope_raw, df_index, tf_weights) # 健康度斜率，正值代表改善
        # 波动性高或健康度恶化时，增加速度和加速度权重，降低基础分权重
        # 波动性低或健康度改善时，增加基础分权重，降低速度和加速度权重
        mod_factor = (norm_volatility_instability * dynamic_weight_sensitivity_volatility) - \
                     (norm_chip_health_slope.clip(upper=0).abs() * dynamic_weight_sensitivity_health_slope) # 负向健康度斜率增加动态权重
        dynamic_base_weight = dynamic_base_weight * (1 - mod_factor)
        dynamic_velocity_weight = dynamic_velocity_weight * (1 + mod_factor * 0.5)
        dynamic_acceleration_weight = dynamic_acceleration_weight * (1 + mod_factor * 0.5)
        # 归一化动态权重
        sum_dynamic_weights = dynamic_base_weight + dynamic_velocity_weight + dynamic_acceleration_weight
        dynamic_base_weight = dynamic_base_weight / sum_dynamic_weights
        dynamic_velocity_weight = dynamic_velocity_weight / sum_dynamic_weights
        dynamic_acceleration_weight = dynamic_acceleration_weight / sum_dynamic_weights
        final_score_unmodulated = (
            (base_strategic_posture_score.add(1)/2).pow(dynamic_base_weight) *
            (norm_velocity.add(1)/2).pow(dynamic_velocity_weight) *
            (norm_acceleration.add(1)/2).pow(dynamic_acceleration_weight)
        ).pow(1 / (dynamic_base_weight + dynamic_velocity_weight + dynamic_acceleration_weight)) * 2 - 1
        # V9.0 全局情境调制器
        final_score = final_score_unmodulated
        if global_context_modulator_enabled:
            norm_global_chip_health = get_adaptive_mtf_normalized_score(chip_health_raw, df_index, ascending=True, tf_weights=tf_weights)
            norm_market_sentiment = get_adaptive_mtf_normalized_score(market_sentiment_raw, df_index, ascending=True, tf_weights=tf_weights)
            global_modulator_effect = (
                (1 + norm_global_chip_health * global_context_sensitivity_health) *
                (1 + norm_market_sentiment * global_context_sensitivity_sentiment)
            ).clip(0.5, 1.5) # 限制调制范围
            final_score = final_score * global_modulator_effect
        return final_score.clip(-1, 1).fillna(0.0).astype(np.float32)

    def _diagnose_battlefield_geography(self, df: pd.DataFrame) -> pd.Series:
        """
        【V9.1 · 诡道地形判别版】诊断筹码的战场地形，旨在提供一个双极的、具备诡道过滤和情境自适应能力的信号。
        - 核心升级1: 核心地形优势量化：重新定义地形优势为“支撑强度 - 阻力强度”，直接输出双极分数 [-1, 1]，正值代表地形有利，负值代表地形不利。
        - 核心升级2: 最小阻力路径动态调制：路径效率（真空区大小与穿越效率）不再简单相乘，而是作为非线性调制因子，放大或削弱核心地形优势。
        - 核心升级3: 动态演化趋势强化：地形趋势变化（支撑与阻力斜率之差）作为乘数，对地形优势进行非线性强化，引入前瞻性。
        - 核心升级4: 诡道地形过滤与惩罚：引入欺骗指数和筹码故障幅度作为诡道因子，对地形优势进行过滤和惩罚，例如在有利地形伴随诱多或虚假支撑时进行惩罚，在不利地形伴随诱空洗盘或虚假阻力时进行缓解。
        - 核心升级5: 情境感知与自适应权重：引入筹码健康度、筹码波动不稳定性等情境因子，动态调整各维度的融合权重，使模型在不同市场环境下自适应地调整对地形特征的关注重点。
        - 核心升级6: 新增筹码指标整合：
            - 向上/向下脉冲强度 (`upward_impulse_strength_D`, `downward_impulse_strength_D`) 增强支撑/阻力强度。
            - 主力成本区买卖意图 (`mf_cost_zone_buy_intent_D`, `mf_cost_zone_sell_intent_D`) 进一步强化支撑/阻力。
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        print("    -> [筹码层] 正在诊断“战场地形 (V9.1 · 诡道地形判别版)”...")
        df_index = df.index
        probe_date = self.strategy.debug_params.get('target_date') if self.strategy.debug_params.get('enable_mfca_probe') else None
        # --- 参数加载 ---
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        bg_params = get_param_value(p_conf.get('battlefield_geography_params'), {}) # 新增参数块
        # V9.0 最小阻力路径动态调制参数
        path_efficiency_mod_factor = get_param_value(bg_params.get('path_efficiency_mod_factor'), 0.5)
        path_efficiency_non_linear_exponent = get_param_value(bg_params.get('path_efficiency_non_linear_exponent'), 1.5)
        # V9.0 动态演化趋势强化参数
        dynamic_evolution_mod_factor = get_param_value(bg_params.get('dynamic_evolution_mod_factor'), 0.3)
        dynamic_evolution_non_linear_exponent = get_param_value(bg_params.get('dynamic_evolution_non_linear_exponent'), 1.2)
        # V9.0 诡道地形过滤与惩罚参数
        deception_signal_name = get_param_value(bg_params.get('deception_signal'), 'deception_index_D')
        chip_fault_signal_name = get_param_value(bg_params.get('chip_fault_signal'), 'chip_fault_magnitude_D')
        deception_penalty_sensitivity = get_param_value(bg_params.get('deception_penalty_sensitivity'), 0.6)
        chip_fault_penalty_sensitivity = get_param_value(bg_params.get('chip_fault_penalty_sensitivity'), 0.4)
        deception_mitigation_sensitivity = get_param_value(bg_params.get('deception_mitigation_sensitivity'), 0.3)
        # V9.0 情境感知与自适应权重参数
        context_modulator_signal_1_name = get_param_value(bg_params.get('context_modulator_signal_1'), 'chip_health_score_D')
        context_modulator_signal_2_name = get_param_value(bg_params.get('context_modulator_signal_2'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        context_modulator_sensitivity_health = get_param_value(bg_params.get('context_modulator_sensitivity_health'), 0.4)
        context_modulator_sensitivity_volatility = get_param_value(bg_params.get('context_modulator_sensitivity_volatility'), 0.3)
        # [修改代码行] V9.1 新增指标参数
        upward_impulse_strength_weight = get_param_value(bg_params.get('upward_impulse_strength_weight'), 0.1)
        downward_impulse_strength_weight = get_param_value(bg_params.get('downward_impulse_strength_weight'), 0.1)
        mf_cost_zone_buy_intent_weight = get_param_value(bg_params.get('mf_cost_zone_buy_intent_weight'), 0.1)
        mf_cost_zone_sell_intent_weight = get_param_value(bg_params.get('mf_cost_zone_sell_intent_weight'), 0.1)
        # --- 信号依赖校验 ---
        required_signals = [
            'dominant_peak_solidity_D', 'support_validation_strength_D', 'chip_fault_blockage_ratio_D',
            'pressure_rejection_strength_D', 'vacuum_zone_magnitude_D', 'vacuum_traversal_efficiency_D',
            'SLOPE_5_support_validation_strength_D', 'SLOPE_5_pressure_rejection_strength_D',
            deception_signal_name, chip_fault_signal_name,
            context_modulator_signal_1_name, context_modulator_signal_2_name,
            # [修改代码行] V9.1 新增筹码指标
            'upward_impulse_strength_D', 'downward_impulse_strength_D',
            'mf_cost_zone_buy_intent_D', 'mf_cost_zone_sell_intent_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_battlefield_geography"):
            return pd.Series(0.0, index=df.index)
        # --- 原始数据获取 ---
        peak_solidity = self._get_safe_series(df, df, 'dominant_peak_solidity_D', 0.5, method_name="_diagnose_battlefield_geography")
        support_validation = self._get_safe_series(df, df, 'support_validation_strength_D', 0.0, method_name="_diagnose_battlefield_geography")
        fault_blockage = self._get_safe_series(df, df, 'chip_fault_blockage_ratio_D', 0.5, method_name="_diagnose_battlefield_geography")
        pressure_rejection = self._get_safe_series(df, df, 'pressure_rejection_strength_D', 0.5, method_name="_diagnose_battlefield_geography")
        vacuum_magnitude = self._get_safe_series(df, df, 'vacuum_zone_magnitude_D', 0.0, method_name="_diagnose_battlefield_geography")
        vacuum_efficiency = self._get_safe_series(df, df, 'vacuum_traversal_efficiency_D', 0.0, method_name="_diagnose_battlefield_geography")
        support_trend_raw = self._get_safe_series(df, df, 'SLOPE_5_support_validation_strength_D', 0.0, method_name="_diagnose_battlefield_geography")
        resistance_trend_raw = self._get_safe_series(df, df, 'SLOPE_5_pressure_rejection_strength_D', 0.0, method_name="_diagnose_battlefield_geography")
        deception_raw = self._get_safe_series(df, df, deception_signal_name, 0.0, method_name="_diagnose_battlefield_geography")
        chip_fault_raw = self._get_safe_series(df, df, chip_fault_signal_name, 0.0, method_name="_diagnose_battlefield_geography")
        chip_health_raw = self._get_safe_series(df, df, context_modulator_signal_1_name, 0.0, method_name="_diagnose_battlefield_geography")
        volatility_instability_raw = self._get_safe_series(df, df, context_modulator_signal_2_name, 0.0, method_name="_diagnose_battlefield_geography")
        # [修改代码行] V9.1 获取新增筹码指标
        upward_impulse_strength_raw = self._get_safe_series(df, df, 'upward_impulse_strength_D', 0.0, method_name="_diagnose_battlefield_geography")
        downward_impulse_strength_raw = self._get_safe_series(df, df, 'downward_impulse_strength_D', 0.0, method_name="_diagnose_battlefield_geography")
        mf_cost_zone_buy_intent_raw = self._get_safe_series(df, df, 'mf_cost_zone_buy_intent_D', 0.0, method_name="_diagnose_battlefield_geography")
        mf_cost_zone_sell_intent_raw = self._get_safe_series(df, df, 'mf_cost_zone_sell_intent_D', 0.0, method_name="_diagnose_battlefield_geography")
        if probe_date: print(f"        -> [探针] {probe_date} 原始信号: dominant_peak_solidity_D={peak_solidity.asof(probe_date):.4f}, support_validation_strength_D={support_validation.asof(probe_date):.4f}, chip_fault_blockage_ratio_D={fault_blockage.asof(probe_date):.4f}, pressure_rejection_strength_D={pressure_rejection.asof(probe_date):.4f}, vacuum_zone_magnitude_D={vacuum_magnitude.asof(probe_date):.4f}, vacuum_traversal_efficiency_D={vacuum_efficiency.asof(probe_date):.4f}, SLOPE_5_support_validation_strength_D={support_trend_raw.asof(probe_date):.4f}, SLOPE_5_pressure_rejection_strength_D={resistance_trend_raw.asof(probe_date):.4f}, {deception_signal_name}={deception_raw.asof(probe_date):.4f}, {chip_fault_signal_name}={chip_fault_raw.asof(probe_date):.4f}, {context_modulator_signal_1_name}={chip_health_raw.asof(probe_date):.4f}, {context_modulator_signal_2_name}={volatility_instability_raw.asof(probe_date):.4f}, upward_impulse_strength_D={upward_impulse_strength_raw.asof(probe_date):.4f}, downward_impulse_strength_D={downward_impulse_strength_raw.asof(probe_date):.4f}, mf_cost_zone_buy_intent_D={mf_cost_zone_buy_intent_raw.asof(probe_date):.4f}, mf_cost_zone_sell_intent_D={mf_cost_zone_sell_intent_raw.asof(probe_date):.4f}")
        # --- 1. 下方支撑 (Support Strength) ---
        solidity_score = get_adaptive_mtf_normalized_score(peak_solidity, df_index, tf_weights)
        validation_score = get_adaptive_mtf_normalized_score(support_validation, df_index, tf_weights)
        support_strength_score = (solidity_score * validation_score).pow(0.5)
        # [修改代码行] V9.1 整合向上脉冲强度和主力成本区买入意图
        norm_upward_impulse_strength = get_adaptive_mtf_normalized_score(upward_impulse_strength_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_mf_cost_zone_buy_intent = get_adaptive_mtf_normalized_score(mf_cost_zone_buy_intent_raw, df_index, ascending=True, tf_weights=tf_weights)
        support_strength_score = support_strength_score * (1 + norm_upward_impulse_strength * upward_impulse_strength_weight + \
                                                              norm_mf_cost_zone_buy_intent * mf_cost_zone_buy_intent_weight)
        support_strength_score = support_strength_score.clip(0, 1)
        if probe_date: print(f"        -> [探针] {probe_date} 支撑强度分数: {support_strength_score.asof(probe_date):.4f}")
        # --- 2. 上方阻力 (Resistance Strength) ---
        blockage_score = get_adaptive_mtf_normalized_score(fault_blockage, df_index, tf_weights)
        rejection_score = get_adaptive_mtf_normalized_score(pressure_rejection, df_index, tf_weights)
        resistance_strength_score = (blockage_score * rejection_score).pow(0.5)
        # [修改代码行] V9.1 整合向下脉冲强度和主力成本区卖出意图
        norm_downward_impulse_strength = get_adaptive_mtf_normalized_score(downward_impulse_strength_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_mf_cost_zone_sell_intent = get_adaptive_mtf_normalized_score(mf_cost_zone_sell_intent_raw, df_index, ascending=True, tf_weights=tf_weights)
        resistance_strength_score = resistance_strength_score * (1 + norm_downward_impulse_strength * downward_impulse_strength_weight + \
                                                                    norm_mf_cost_zone_sell_intent * mf_cost_zone_sell_intent_weight)
        resistance_strength_score = resistance_strength_score.clip(0, 1)
        if probe_date: print(f"        -> [探针] {probe_date} 阻力强度分数: {resistance_strength_score.asof(probe_date):.4f}")
        # --- 3. 核心地形优势量化 (Core Terrain Advantage Quantification - Bipolar) ---
        # 正值代表地形有利 (支撑强于阻力)，负值代表地形不利 (阻力强于支撑)
        base_terrain_advantage_score = support_strength_score - resistance_strength_score
        if probe_date: print(f"        -> [探针] {probe_date} 基础地形优势分数: {base_terrain_advantage_score.asof(probe_date):.4f}")
        # --- 4. 最小阻力路径动态调制 (Dynamic Path of Least Resistance Modulation) ---
        norm_vacuum_magnitude = get_adaptive_mtf_normalized_score(vacuum_magnitude, df_index, tf_weights)
        norm_vacuum_efficiency = get_adaptive_mtf_normalized_score(vacuum_efficiency, df_index, tf_weights)
        path_efficiency = (norm_vacuum_magnitude * norm_vacuum_efficiency).pow(0.5) # 归一化到 [0, 1]
        # 路径效率越高，对地形优势的放大作用越强
        path_modulation_factor = (1 + path_efficiency * path_efficiency_mod_factor).pow(path_efficiency_non_linear_exponent)
        if probe_date: print(f"        -> [探针] {probe_date} 路径效率调制因子: {path_modulation_factor.asof(probe_date):.4f}")
        # --- 5. 动态演化趋势强化 (Dynamic Evolution Trend Reinforcement) ---
        norm_support_trend = get_adaptive_mtf_normalized_bipolar_score(support_trend_raw, df_index, tf_weights) # 探针输出位置调整
        norm_resistance_trend = get_adaptive_mtf_normalized_bipolar_score(resistance_trend_raw, df_index, tf_weights) # 探针输出位置调整
        # 地形优势变化趋势 (正值代表趋势改善，负值代表趋势恶化)
        terrain_advantage_change = norm_support_trend - norm_resistance_trend
        # 趋势改善时，对地形优势进行放大；趋势恶化时，对地形优势进行削弱
        dynamic_evolution_modulator = (1 + terrain_advantage_change * dynamic_evolution_mod_factor).pow(dynamic_evolution_non_linear_exponent)
        dynamic_evolution_modulator = dynamic_evolution_modulator.clip(0.5, 1.5) # 限制调制范围
        if probe_date: print(f"        -> [探针] {probe_date} 动态演化调制器: {dynamic_evolution_modulator.asof(probe_date):.4f}")
        # --- 6. 诡道地形过滤与惩罚 (Deceptive Terrain Filtering & Penalty) ---
        norm_deception = get_adaptive_mtf_normalized_bipolar_score(deception_raw, df_index, tf_weights) # 归一化到 [-1, 1]
        norm_chip_fault = get_adaptive_mtf_normalized_bipolar_score(chip_fault_raw, df_index, tf_weights) # 归一化到 [-1, 1]
        deception_filter_factor = pd.Series(1.0, index=df_index)
        # 地形有利 (base_terrain_advantage_score > 0) 且伴随诱多 (norm_deception > 0) 或虚假支撑 (norm_chip_fault > 0) -> 惩罚
        bull_trap_penalty_mask = (base_terrain_advantage_score > 0) & ((norm_deception > 0) | (norm_chip_fault > 0))
        deception_filter_factor.loc[bull_trap_penalty_mask] = \
            1 - ((norm_deception.loc[bull_trap_penalty_mask].clip(lower=0) * deception_penalty_sensitivity) + \
                 (norm_chip_fault.loc[bull_trap_penalty_mask].clip(lower=0) * chip_fault_penalty_sensitivity)).clip(0, 1)
        # 地形不利 (base_terrain_advantage_score < 0) 且伴随诱空洗盘 (norm_deception < 0) 或虚假阻力 (norm_chip_fault < 0) -> 缓解
        bear_trap_mitigation_mask = (base_terrain_advantage_score < 0) & ((norm_deception < 0) | (norm_chip_fault < 0))
        deception_filter_factor.loc[bear_trap_mitigation_mask] = \
            1 + ((norm_deception.loc[bear_trap_mitigation_mask].abs().clip(lower=0) * deception_mitigation_sensitivity) + \
                 (norm_chip_fault.loc[bear_trap_mitigation_mask].abs().clip(lower=0) * deception_mitigation_sensitivity)).clip(0, 0.5) # 限制缓解幅度
        deception_filter_factor = deception_filter_factor.clip(0.1, 2.0) # 限制过滤因子范围
        if probe_date: print(f"        -> [探针] {probe_date} 诡道过滤因子: {deception_filter_factor.asof(probe_date):.4f}")
        # --- 7. 情境感知与自适应权重 (Contextual Awareness & Adaptive Weighting) ---
        norm_chip_health = get_adaptive_mtf_normalized_score(chip_health_raw, df_index, ascending=True, tf_weights=tf_weights) # 归一化到 [0, 1]
        norm_volatility_instability = get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=False, tf_weights=tf_weights) # 波动性越高，值越小 (负向影响)
        # 融合情境调制器：健康度越高，波动性越低，情境因子越接近1，地形信号越可靠
        context_modulator = (
            (1 + norm_chip_health * context_modulator_sensitivity_health) *
            (1 + norm_volatility_instability * context_modulator_sensitivity_volatility)
        ).clip(0.5, 1.5) # 限制调制范围
        if probe_date: print(f"        -> [探针] {probe_date} 情境调制器: {context_modulator.asof(probe_date):.4f}")
        # --- 最终融合 ---
        # 核心地形优势 * 路径调制 * 动态演化调制 * 诡道过滤 * 情境调制
        final_score = base_terrain_advantage_score * path_modulation_factor * dynamic_evolution_modulator * deception_filter_factor * context_modulator
        if probe_date: print(f"        -> [探针] {probe_date} 最终战场地形分数: {final_score.asof(probe_date):.4f}")
        return final_score.clip(-1, 1).fillna(0.0).astype(np.float32)

    def _diagnose_axiom_holder_sentiment(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V8.2 · 动态韧性感知版】筹码公理三：诊断“持仓信念韧性”
        - 核心升级1: 信念内核深度化。在V7.8基础上，引入总赢家比例、总输家比例、赢家输家动量、赢家稳定性短期斜率、输家痛苦指数短期加速度和赢家输家动量短期斜率，更全面刻画信念的动态变化。
        - 核心升级2: 压力测试精细化。在V7.8基础上，引入开盘缺口防守强度、控制坚实度指数、订单簿清算率、微观价格冲击不对称性、支撑验证强度短期斜率、投降吸收指数短期加速度和主动买盘支撑短期斜率，更精准评估主力承受压力的能力。
        - 核心升级3: 杂质削弱多维感知。在V7.8基础上，引入上影线抛压、反弹派发压力、散户狂热溢价、赢家平均利润率短期斜率和散户狂热溢价短期加速度，更全面识别削弱信念的杂质。
        - 核心升级4: 诡道因子智能化。在V7.8基础上，引入欺骗指数和对倒强度，更智能地判断诡道意图并进行调制。
        - 核心升级5: 情境调制器扩展。引入波动率不稳定性指数、市场情绪分数和资金流可信度指数，提供更丰富的宏观情境感知。
        - 核心升级6: 新增筹码指标整合：
            - 信念流买入/卖出强度 (`conviction_flow_buy_intensity_D`, `conviction_flow_sell_intensity_D`) 增强信念内核维度，并以乘法形式融入。
        - 代码优化: 移除所有调试探针，优化部分计算逻辑，提高运行效率。
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        print("    -> [筹码层] 正在诊断“持仓信念韧性 (V8.2 · 动态韧性感知版)”...")
        df_index = df.index
        probe_date = self.strategy.debug_params.get('target_date') if self.strategy.debug_params.get('enable_mfca_probe') else None
        required_signals = [
            'winner_stability_index_D', 'loser_pain_index_D', 'active_buying_support_D',
            'support_validation_strength_D', 'winner_concentration_90pct_D',
            'winner_profit_margin_avg_D', 'capitulation_absorption_index_D',
            'SLOPE_55_winner_concentration_90pct_D',
            'chip_fatigue_index_D', 'chip_fault_magnitude_D', 'chip_health_score_D',
            'total_winner_rate_D', 'total_loser_rate_D', 'winner_loser_momentum_D',
            'SLOPE_5_winner_stability_index_D', 'ACCEL_5_loser_pain_index_D', 'SLOPE_5_winner_loser_momentum_D',
            'opening_gap_defense_strength_D', 'control_solidity_index_D', 'order_book_clearing_rate_D',
            'micro_price_impact_asymmetry_D', 'SLOPE_5_support_validation_strength_D',
            'ACCEL_5_capitulation_absorption_index_D', 'SLOPE_5_active_buying_support_D',
            'upper_shadow_selling_pressure_D', 'rally_distribution_pressure_D', 'retail_fomo_premium_index_D',
            'SLOPE_5_winner_profit_margin_avg_D', 'ACCEL_5_retail_fomo_premium_index_D',
            'deception_index_D', 'wash_trade_intensity_D',
            'VOLATILITY_INSTABILITY_INDEX_21d_D', 'market_sentiment_score_D', 'flow_credibility_index_D',
            'main_force_conviction_index_D',
            # [修改代码行] V8.1 新增筹码指标
            'conviction_flow_buy_intensity_D', 'conviction_flow_sell_intensity_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_holder_sentiment"):
            return pd.Series(0.0, index=df.index)
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        holder_sentiment_params = get_param_value(p_conf.get('holder_sentiment_params'), {})
        # --- 参数加载 ---
        sentiment_trend_modulator_signal_name = get_param_value(holder_sentiment_params.get('sentiment_trend_modulator_signal_name'), 'SLOPE_55_winner_concentration_90pct_D')
        sentiment_trend_mod_factor = get_param_value(holder_sentiment_params.get('sentiment_trend_mod_factor'), 0.5)
        panic_reward_modulator_signal_name = get_param_value(holder_sentiment_params.get('panic_reward_modulator_signal_name'), 'chip_fatigue_index_D')
        panic_reward_mod_tanh_factor = get_param_value(holder_sentiment_params.get('panic_reward_mod_tanh_factor'), 1.0)
        panic_reward_mod_factor = get_param_value(holder_sentiment_params.get('panic_reward_mod_factor'), 1.0)
        capitulation_base_reward_multiplier = get_param_value(holder_sentiment_params.get('capitulation_base_reward_multiplier'), 0.3)
        impurity_non_linear_enabled = get_param_value(holder_sentiment_params.get('impurity_non_linear_enabled'), True)
        fomo_tanh_factor = get_param_value(holder_sentiment_params.get('fomo_tanh_factor'), 1.0)
        fomo_sentiment_sensitivity = get_param_value(holder_sentiment_params.get('fomo_sentiment_sensitivity'), 0.5)
        profit_taking_tanh_factor = get_param_value(holder_sentiment_params.get('profit_taking_tanh_factor'), 1.0)
        profit_taking_sentiment_sensitivity = get_param_value(holder_sentiment_params.get('profit_taking_sentiment_sensitivity'), 0.5)
        deception_factor_enabled = get_param_value(holder_sentiment_params.get('deception_factor_enabled'), True)
        deception_signal_name = get_param_value(holder_sentiment_params.get('deception_signal_name'), 'chip_fault_magnitude_D')
        deception_impact_factor = get_param_value(holder_sentiment_params.get('deception_impact_factor'), 0.2)
        positive_deception_penalty_enabled = get_param_value(holder_sentiment_params.get('positive_deception_penalty_enabled'), True)
        positive_deception_impact_factor = get_param_value(holder_sentiment_params.get('positive_deception_impact_factor'), 0.15)
        impurity_context_modulation_enabled = get_param_value(holder_sentiment_params.get('impurity_context_modulation_enabled'), True)
        impurity_context_modulator_signal_name = get_param_value(holder_sentiment_params.get('impurity_context_modulator_signal_name'), 'chip_health_score_D')
        impurity_context_overbought_amp_factor = get_param_value(holder_sentiment_params.get('impurity_context_overbought_amp_factor'), 0.5)
        impurity_context_oversold_damp_factor = get_param_value(holder_sentiment_params.get('impurity_context_oversold_damp_factor'), 0.2)
        dynamic_fusion_enabled = get_param_value(holder_sentiment_params.get('dynamic_fusion_enabled'), True)
        min_pressure_weight = get_param_value(holder_sentiment_params.get('min_pressure_weight'), 0.3)
        max_pressure_weight = get_param_value(holder_sentiment_params.get('max_pressure_weight'), 0.7)
        impurity_fusion_exponent_base = get_param_value(holder_sentiment_params.get('impurity_fusion_exponent_base'), 0.7)
        impurity_fusion_exponent_sensitivity = get_param_value(holder_sentiment_params.get('impurity_fusion_exponent_sensitivity'), 0.5)
        fomo_concentration_optimal_target = get_param_value(holder_sentiment_params.get('fomo_concentration_optimal_target'), 0.5)
        profit_taking_threshold = get_param_value(holder_sentiment_params.get('profit_taking_threshold'), 5.0)
        # V8.0 新增参数
        belief_core_weights = get_param_value(holder_sentiment_params.get('belief_core_weights'), {
            'winner_stability': 0.2, 'loser_pain': 0.2, 'total_winner_rate': 0.1, 'total_loser_rate': 0.1,
            'winner_loser_momentum': 0.1, 'winner_stability_slope': 0.1, 'loser_pain_accel': 0.1,
            'winner_loser_momentum_slope': 0.1
        })
        pressure_test_weights = get_param_value(holder_sentiment_params.get('pressure_test_weights'), {
            'active_buying_support': 0.15, 'support_validation': 0.15, 'capitulation_absorption': 0.15,
            'opening_gap_defense_strength': 0.1, 'control_solidity': 0.1, 'order_book_clearing_rate': 0.08,
            'micro_price_impact_asymmetry': 0.07, 'support_validation_slope': 0.08,
            'capitulation_absorption_accel': 0.07, 'active_buying_support_slope': 0.05
        })
        impurity_weights = get_param_value(holder_sentiment_params.get('impurity_weights'), {
            'fomo_concentration': 0.2, 'profit_taking_margin': 0.2, 'upper_shadow_selling_pressure': 0.15,
            'rally_distribution_pressure': 0.15, 'retail_fomo_premium': 0.1,
            'winner_profit_margin_slope': 0.1, 'retail_fomo_premium_accel': 0.1
        })
        deception_modulator_weights = get_param_value(holder_sentiment_params.get('deception_modulator_weights'), {
            'deception_index_boost': 0.5, 'wash_trade_penalty': 0.3
        })
        context_modulator_weights = get_param_value(holder_sentiment_params.get('context_modulator_weights'), {
            'volatility_instability': 0.3, 'market_sentiment': 0.4, 'flow_credibility': 0.3
        })
        deception_modulator_params = get_param_value(holder_sentiment_params.get('deception_modulator_params'), {'boost_factor': 0.6, 'penalty_factor': 0.4, 'conviction_threshold': 0.2, 'deception_index_weight': 0.5})
        df_index = df.index
        # --- 原始数据获取 ---
        winner_stability = self._get_safe_series(df, df, 'winner_stability_index_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        loser_pain = self._get_safe_series(df, df, 'loser_pain_index_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        total_winner_rate_raw = self._get_safe_series(df, df, 'total_winner_rate_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        total_loser_rate_raw = self._get_safe_series(df, df, 'total_loser_rate_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        winner_loser_momentum_raw = self._get_safe_series(df, df, 'winner_loser_momentum_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        slope_5_winner_stability_raw = self._get_safe_series(df, df, 'SLOPE_5_winner_stability_index_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        accel_5_loser_pain_raw = self._get_safe_series(df, df, 'ACCEL_5_loser_pain_index_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        slope_5_winner_loser_momentum_raw = self._get_safe_series(df, df, 'SLOPE_5_winner_loser_momentum_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        absorption_power = self._get_safe_series(df, df, 'active_buying_support_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        defense_intent = self._get_safe_series(df, df, 'support_validation_strength_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        capitulation_absorption = self._get_safe_series(df, df, 'capitulation_absorption_index_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        opening_gap_defense_strength_raw = self._get_safe_series(df, df, 'opening_gap_defense_strength_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        control_solidity_raw = self._get_safe_series(df, df, 'control_solidity_index_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        order_book_clearing_rate_raw = self._get_safe_series(df, df, 'order_book_clearing_rate_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        micro_price_impact_asymmetry_raw = self._get_safe_series(df, df, 'micro_price_impact_asymmetry_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        slope_5_support_validation_raw = self._get_safe_series(df, df, 'SLOPE_5_support_validation_strength_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        accel_5_capitulation_absorption_raw = self._get_safe_series(df, df, 'ACCEL_5_capitulation_absorption_index_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        slope_5_active_buying_support_raw = self._get_safe_series(df, df, 'SLOPE_5_active_buying_support_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        fomo_index_raw = self._get_safe_series(df, df, 'winner_concentration_90pct_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        profit_taking_quality_raw = self._get_safe_series(df, df, 'winner_profit_margin_avg_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        upper_shadow_selling_pressure_raw = self._get_safe_series(df, df, 'upper_shadow_selling_pressure_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        rally_distribution_pressure_raw = self._get_safe_series(df, df, 'rally_distribution_pressure_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        retail_fomo_premium_raw = self._get_safe_series(df, df, 'retail_fomo_premium_index_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        slope_5_winner_profit_margin_raw = self._get_safe_series(df, df, 'SLOPE_5_winner_profit_margin_avg_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        accel_5_retail_fomo_premium_raw = self._get_safe_series(df, df, 'ACCEL_5_retail_fomo_premium_index_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        chip_fatigue_raw = self._get_safe_series(df, df, 'chip_fatigue_index_D', 0.0, method_name="_diagnose_axiom_holder_sentiment") # 提前获取
        deception_raw = self._get_safe_series(df, df, deception_signal_name, 0.0, method_name="_diagnose_axiom_holder_sentiment")
        deception_index_raw = self._get_safe_series(df, df, 'deception_index_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        wash_trade_intensity_raw = self._get_safe_series(df, df, 'wash_trade_intensity_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        main_force_conviction_raw = self._get_safe_series(df, df, 'main_force_conviction_index_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        volatility_instability_raw = self._get_safe_series(df, df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        market_sentiment_raw = self._get_safe_series(df, df, 'market_sentiment_score_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        flow_credibility_raw = self._get_safe_series(df, df, 'flow_credibility_index_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        # [修改代码行] V8.1 获取新增信念流强度指标
        conviction_flow_buy_intensity_raw = self._get_safe_series(df, df, 'conviction_flow_buy_intensity_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        conviction_flow_sell_intensity_raw = self._get_safe_series(df, df, 'conviction_flow_sell_intensity_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        if probe_date: print(f"        -> [探针] {probe_date} 原始信号: winner_stability_index_D={winner_stability.asof(probe_date):.4f}, loser_pain_index_D={loser_pain.asof(probe_date):.4f}, total_winner_rate_D={total_winner_rate_raw.asof(probe_date):.4f}, total_loser_rate_D={total_loser_rate_raw.asof(probe_date):.4f}, winner_loser_momentum_D={winner_loser_momentum_raw.asof(probe_date):.4f}, SLOPE_5_winner_stability_index_D={slope_5_winner_stability_raw.asof(probe_date):.4f}, ACCEL_5_loser_pain_index_D={accel_5_loser_pain_raw.asof(probe_date):.4f}, SLOPE_5_winner_loser_momentum_D={slope_5_winner_loser_momentum_raw.asof(probe_date):.4f}, active_buying_support_D={absorption_power.asof(probe_date):.4f}, support_validation_strength_D={defense_intent.asof(probe_date):.4f}, capitulation_absorption_index_D={capitulation_absorption.asof(probe_date):.4f}, opening_gap_defense_strength_D={opening_gap_defense_strength_raw.asof(probe_date):.4f}, control_solidity_index_D={control_solidity_raw.asof(probe_date):.4f}, order_book_clearing_rate_D={order_book_clearing_rate_raw.asof(probe_date):.4f}, micro_price_impact_asymmetry_D={micro_price_impact_asymmetry_raw.asof(probe_date):.4f}, SLOPE_5_support_validation_strength_D={slope_5_support_validation_raw.asof(probe_date):.4f}, ACCEL_5_capitulation_absorption_index_D={accel_5_capitulation_absorption_raw.asof(probe_date):.4f}, SLOPE_5_active_buying_support_D={slope_5_active_buying_support_raw.asof(probe_date):.4f}, winner_concentration_90pct_D={fomo_index_raw.asof(probe_date):.4f}, winner_profit_margin_avg_D={profit_taking_quality_raw.asof(probe_date):.4f}, upper_shadow_selling_pressure_D={upper_shadow_selling_pressure_raw.asof(probe_date):.4f}, rally_distribution_pressure_D={rally_distribution_pressure_raw.asof(probe_date):.4f}, retail_fomo_premium_index_D={retail_fomo_premium_raw.asof(probe_date):.4f}, SLOPE_5_winner_profit_margin_avg_D={slope_5_winner_profit_margin_raw.asof(probe_date):.4f}, ACCEL_5_retail_fomo_premium_index_D={accel_5_retail_fomo_premium_raw.asof(probe_date):.4f}, chip_fatigue_index_D={chip_fatigue_raw.asof(probe_date):.4f}, deception_index_D={deception_index_raw.asof(probe_date):.4f}, wash_trade_intensity_D={wash_trade_intensity_raw.asof(probe_date):.4f}, main_force_conviction_index_D={main_force_conviction_raw.asof(probe_date):.4f}, VOLATILITY_INSTABILITY_INDEX_21d_D={volatility_instability_raw.asof(probe_date):.4f}, market_sentiment_score_D={market_sentiment_raw.asof(probe_date):.4f}, flow_credibility_index_D={flow_credibility_raw.asof(probe_date):.4f}, conviction_flow_buy_intensity_D={conviction_flow_buy_intensity_raw.asof(probe_date):.4f}, conviction_flow_sell_intensity_D={conviction_flow_sell_intensity_raw.asof(probe_date):.4f}")
        # --- 维度1: 信念内核 (Belief Core Score) ---
        norm_winner_stability = get_adaptive_mtf_normalized_bipolar_score(winner_stability, df_index, tf_weights)
        norm_loser_pain = get_adaptive_mtf_normalized_bipolar_score(loser_pain, df_index, tf_weights)
        norm_total_winner_rate = get_adaptive_mtf_normalized_score(total_winner_rate_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_total_loser_rate = get_adaptive_mtf_normalized_score(total_loser_rate_raw, df_index, ascending=False, tf_weights=tf_weights)
        norm_winner_loser_momentum = get_adaptive_mtf_normalized_bipolar_score(winner_loser_momentum_raw, df_index, tf_weights)
        norm_slope_5_winner_stability = get_adaptive_mtf_normalized_bipolar_score(slope_5_winner_stability_raw, df_index, tf_weights)
        norm_accel_5_loser_pain = get_adaptive_mtf_normalized_bipolar_score(accel_5_loser_pain_raw, df_index, tf_weights)
        norm_slope_5_winner_loser_momentum = get_adaptive_mtf_normalized_bipolar_score(slope_5_winner_loser_momentum_raw, df_index, tf_weights)
        sentiment_trend_raw = self._get_safe_series(df, df, sentiment_trend_modulator_signal_name, 0.0, method_name="_diagnose_axiom_holder_sentiment")
        normalized_sentiment_trend = get_adaptive_mtf_normalized_score(sentiment_trend_raw, df_index, tf_weights=tf_weights, ascending=True)
        x = (normalized_sentiment_trend * sentiment_trend_mod_factor).clip(-0.4, 0.4)
        dynamic_stability_weight = 0.5 + x
        dynamic_pain_weight = 0.5 - x
        belief_core_numeric_weights = {k: v for k, v in belief_core_weights.items() if isinstance(v, (int, float))}
        # [修改代码行] V8.2 整合信念流强度到信念内核的融合计算中
        norm_conviction_flow_buy_intensity = get_adaptive_mtf_normalized_score(conviction_flow_buy_intensity_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_conviction_flow_sell_intensity = get_adaptive_mtf_normalized_score(conviction_flow_sell_intensity_raw, df_index, ascending=True, tf_weights=tf_weights)
        # Ensure all components are positive for geometric mean, then scale back to bipolar
        # For bipolar scores, convert to [0,1] range for geometric mean, then back to [-1,1]
        comp_winner_stability = (norm_winner_stability.add(1)/2).pow(belief_core_numeric_weights.get('winner_stability', 0.15) * dynamic_stability_weight)
        comp_loser_pain = (norm_loser_pain.add(1)/2).pow(belief_core_numeric_weights.get('loser_pain', 0.15) * dynamic_pain_weight)
        comp_total_winner_rate = norm_total_winner_rate.pow(belief_core_numeric_weights.get('total_winner_rate', 0.08))
        comp_total_loser_rate = (1 - norm_total_loser_rate).pow(belief_core_numeric_weights.get('total_loser_rate', 0.08))
        comp_winner_loser_momentum = (norm_winner_loser_momentum.add(1)/2).pow(belief_core_numeric_weights.get('winner_loser_momentum', 0.08))
        comp_winner_stability_slope = (norm_slope_5_winner_stability.add(1)/2).pow(belief_core_numeric_weights.get('winner_stability_slope', 0.08))
        comp_loser_pain_accel = (norm_accel_5_loser_pain.add(1)/2).pow(belief_core_numeric_weights.get('loser_pain_accel', 0.08))
        comp_winner_loser_momentum_slope = (norm_slope_5_winner_loser_momentum.add(1)/2).pow(belief_core_numeric_weights.get('winner_loser_momentum_slope', 0.08))
        # [修改代码行] V8.2 信念流买入强度正向贡献
        comp_conviction_flow_buy = norm_conviction_flow_buy_intensity.pow(belief_core_numeric_weights.get('conviction_flow_buy', 0.1))
        # [修改代码行] V8.2 信念流卖出强度负向贡献 (1-score 形式)
        comp_conviction_flow_sell = (1 - norm_conviction_flow_sell_intensity).pow(belief_core_numeric_weights.get('conviction_flow_sell', 0.1))
        # Sum of all weights, including new ones, for normalization exponent
        total_belief_core_weight = sum(belief_core_numeric_weights.values())
        belief_core_score = (
            comp_winner_stability *
            comp_loser_pain *
            comp_total_winner_rate *
            comp_total_loser_rate *
            comp_winner_loser_momentum *
            comp_winner_stability_slope *
            comp_loser_pain_accel *
            comp_winner_loser_momentum_slope *
            comp_conviction_flow_buy * # [修改代码行] V8.2 融入信念流买入
            comp_conviction_flow_sell # [修改代码行] V8.2 融入信念流卖出
        ).pow(1 / total_belief_core_weight) * 2 - 1
        belief_core_score = belief_core_score.clip(-1, 1) # 确保分数在 [-1, 1] 范围内
        if probe_date: print(f"        -> [探针] {probe_date} 信念内核分数: {belief_core_score.asof(probe_date):.4f}")
        # --- 维度2: 压力测试 (Pressure Test Score) ---
        norm_absorption_power = get_adaptive_mtf_normalized_bipolar_score(absorption_power, df_index, tf_weights)
        norm_defense_intent = get_adaptive_mtf_normalized_bipolar_score(defense_intent, df_index, tf_weights)
        norm_capitulation_absorption = get_adaptive_mtf_normalized_score(capitulation_absorption, df_index, tf_weights=tf_weights)
        norm_opening_gap_defense_strength = get_adaptive_mtf_normalized_score(opening_gap_defense_strength_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_control_solidity = get_adaptive_mtf_normalized_score(control_solidity_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_order_book_clearing_rate = get_adaptive_mtf_normalized_score(order_book_clearing_rate_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_micro_price_impact_asymmetry = get_adaptive_mtf_normalized_score(micro_price_impact_asymmetry_raw, df_index, ascending=False, tf_weights=tf_weights)
        norm_slope_5_support_validation = get_adaptive_mtf_normalized_score(slope_5_support_validation_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_accel_5_capitulation_absorption = get_adaptive_mtf_normalized_score(accel_5_capitulation_absorption_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_slope_5_active_buying_support = get_adaptive_mtf_normalized_score(slope_5_active_buying_support_raw, df_index, ascending=True, tf_weights=tf_weights)
        base_pressure_test_numeric_weights = {k: v for k, v in pressure_test_weights.items() if isinstance(v, (int, float))}
        total_base_pressure_test_weight = sum(base_pressure_test_numeric_weights.values())
        base_pressure_score = (
            (norm_absorption_power.add(1)/2).pow(base_pressure_test_numeric_weights.get('active_buying_support', 0.15)) *
            (norm_defense_intent.add(1)/2).pow(base_pressure_test_numeric_weights.get('support_validation', 0.15)) *
            norm_capitulation_absorption.pow(base_pressure_test_numeric_weights.get('capitulation_absorption', 0.15)) *
            norm_opening_gap_defense_strength.pow(base_pressure_test_numeric_weights.get('opening_gap_defense_strength', 0.1)) *
            norm_control_solidity.pow(base_pressure_test_numeric_weights.get('control_solidity', 0.1)) *
            norm_order_book_clearing_rate.pow(base_pressure_test_numeric_weights.get('order_book_clearing_rate', 0.08)) *
            norm_micro_price_impact_asymmetry.pow(base_pressure_test_numeric_weights.get('micro_price_impact_asymmetry', 0.07)) *
            norm_slope_5_support_validation.pow(base_pressure_test_numeric_weights.get('support_validation_slope', 0.08)) *
            norm_accel_5_capitulation_absorption.pow(base_pressure_test_numeric_weights.get('capitulation_absorption_accel', 0.07)) *
            norm_slope_5_active_buying_support.pow(base_pressure_test_numeric_weights.get('active_buying_support_slope', 0.05))
        ).pow(1 / total_base_pressure_test_weight) * 2 - 1
        panic_modulator_raw = self._get_safe_series(df, df, panic_reward_modulator_signal_name, 0.0, method_name="_diagnose_axiom_holder_sentiment")
        normalized_panic_modulator = get_adaptive_mtf_normalized_score(panic_modulator_raw, df_index, tf_weights=tf_weights, ascending=True)
        panic_reward_adjustment_factor = np.tanh(normalized_panic_modulator * panic_reward_mod_tanh_factor) * panic_reward_mod_factor
        dynamic_capitulation_reward_multiplier = capitulation_base_reward_multiplier * (1 + panic_reward_adjustment_factor)
        dynamic_capitulation_reward_multiplier = dynamic_capitulation_reward_multiplier.clip(0.1, 0.8)
        capitulation_bonus = norm_capitulation_absorption * dynamic_capitulation_reward_multiplier
        deception_impact = pd.Series(0.0, index=df.index)
        deception_raw = self._get_safe_series(df, df, deception_signal_name, 0.0, method_name="_diagnose_axiom_holder_sentiment")
        if deception_factor_enabled:
            negative_deception = deception_raw.clip(upper=0).abs()
            normalized_negative_deception = get_adaptive_mtf_normalized_score(negative_deception, df_index, tf_weights)
            deception_impact = normalized_negative_deception * deception_impact_factor
        pressure_test_score = base_pressure_score * (1 + capitulation_bonus + deception_impact)
        pressure_test_score = pressure_test_score.clip(-1, 1)
        if probe_date: print(f"        -> [探针] {probe_date} 压力测试分数: {pressure_test_score.asof(probe_date):.4f}")
        # --- 动态融合信念内核与压力测试 ---
        s_belief_core = belief_core_score.add(1)/2
        s_pressure_test = pressure_test_score.add(1)/2
        dynamic_belief_core_weight = pd.Series(0.5, index=df.index)
        dynamic_pressure_test_weight = pd.Series(0.5, index=df.index)
        if dynamic_fusion_enabled:
            dynamic_pressure_test_weight = min_pressure_weight + (max_pressure_weight - min_pressure_weight) * normalized_panic_modulator
            dynamic_belief_core_weight = 1.0 - dynamic_pressure_test_weight
        conviction_base = (s_belief_core.pow(dynamic_belief_core_weight) * s_pressure_test.pow(dynamic_pressure_test_weight))
        if probe_date: print(f"        -> [探针] {probe_date} 动态融合信念基础分数: {conviction_base.asof(probe_date):.4f}")
        # --- 诡道因子 (Deception Factors) ---
        positive_deception_penalty = pd.Series(0.0, index=df.index)
        if positive_deception_penalty_enabled:
            positive_deception_raw = deception_raw.clip(lower=0)
            normalized_positive_deception = get_adaptive_mtf_normalized_score(positive_deception_raw, df_index, tf_weights)
            positive_deception_penalty = normalized_positive_deception * positive_deception_impact_factor
            conviction_base = conviction_base * (1 - positive_deception_penalty)
            conviction_base = conviction_base.clip(0, 1)
        # V8.0 新增：更智能的诡道因子调制
        norm_deception_index_bipolar = get_adaptive_mtf_normalized_bipolar_score(deception_index_raw, df_index, tf_weights)
        norm_wash_trade_intensity = get_adaptive_mtf_normalized_score(wash_trade_intensity_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_main_force_conviction_bipolar = get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, tf_weights)
        # 负向欺骗（诱空）且主力信念坚定 -> 增强信念
        deception_boost_mask = (norm_deception_index_bipolar < 0) & (norm_main_force_conviction_bipolar > deception_modulator_params.get('conviction_threshold', 0.2))
        conviction_base.loc[deception_boost_mask] = conviction_base.loc[deception_boost_mask] * (1 + norm_deception_index_bipolar.loc[deception_boost_mask].abs() * deception_modulator_weights.get('deception_index_boost', 0.5))
        # 正向欺骗（诱多）且主力信念动摇 -> 削弱信念
        deception_penalty_mask = (norm_deception_index_bipolar > 0) & (norm_main_force_conviction_bipolar < -deception_modulator_params.get('conviction_threshold', 0.2))
        conviction_base.loc[deception_penalty_mask] = conviction_base.loc[deception_penalty_mask] * (1 - norm_deception_index_bipolar.loc[deception_penalty_mask] * deception_modulator_weights.get('deception_index_boost', 0.5))
        # 对倒强度作为杂质削弱信念
        conviction_base = conviction_base * (1 - norm_wash_trade_intensity * deception_modulator_weights.get('wash_trade_penalty', 0.3))
        conviction_base = conviction_base.clip(0, 1)
        if probe_date: print(f"        -> [探针] {probe_date} 诡道调制后信念基础分数: {conviction_base.asof(probe_date):.4f}")
        # --- 杂质削弱 (Impurity Score) ---
        fomo_deviation = (fomo_index_raw - fomo_concentration_optimal_target).abs()
        norm_fomo_deviation = get_adaptive_mtf_normalized_score(fomo_deviation, df_index, tf_weights=tf_weights)
        profit_taking_quality_thresholded = (profit_taking_quality_raw - profit_taking_threshold).clip(lower=0)
        norm_profit_taking_quality = get_adaptive_mtf_normalized_score(profit_taking_quality_thresholded, df_index, tf_weights=tf_weights)
        norm_upper_shadow_selling_pressure = get_adaptive_mtf_normalized_score(upper_shadow_selling_pressure_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_rally_distribution_pressure = get_adaptive_mtf_normalized_score(rally_distribution_pressure_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_retail_fomo_premium = get_adaptive_mtf_normalized_score(retail_fomo_premium_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_slope_5_winner_profit_margin = get_adaptive_mtf_normalized_score(slope_5_winner_profit_margin_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_accel_5_retail_fomo_premium = get_adaptive_mtf_normalized_score(accel_5_retail_fomo_premium_raw, df_index, ascending=True, tf_weights=tf_weights)
        fomo_effect = pd.Series(0.0, index=df.index)
        profit_taking_effect = pd.Series(0.0, index=df.index)
        other_impurity_effect = pd.Series(0.0, index=df.index)
        final_impurity_effect = pd.Series(0.0, index=df.index)
        if impurity_non_linear_enabled:
            current_sentiment_strength = conviction_base.abs()
            normalized_sentiment_strength = normalize_score(current_sentiment_strength, df_index, window=21, ascending=True)
            context_adjustment_factor = pd.Series(1.0, index=df.index)
            # --- 情境调制器 (Contextual Modulators) ---
            norm_volatility_instability = get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=True, tf_weights=tf_weights)
            norm_market_sentiment = get_adaptive_mtf_normalized_score(market_sentiment_raw, df_index, ascending=True, tf_weights=tf_weights)
            norm_flow_credibility = get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, ascending=True, tf_weights=tf_weights)
            context_modulator_numeric_weights = {k: v for k, v in context_modulator_weights.items() if isinstance(v, (int, float))}
            total_context_modulator_weight = sum(context_modulator_numeric_weights.values())
            # 融合情境调制器，作为对杂质影响的额外调整
            if total_context_modulator_weight > 0:
                fused_context_modulator = (
                    norm_volatility_instability.pow(context_modulator_numeric_weights.get('volatility_instability', 0.3)) *
                    norm_market_sentiment.pow(context_modulator_numeric_weights.get('market_sentiment', 0.4)) *
                    norm_flow_credibility.pow(context_modulator_numeric_weights.get('flow_credibility', 0.3))
                ).pow(1 / total_context_modulator_weight)
                context_adjustment_factor = context_adjustment_factor * (1 + (fused_context_modulator - 0.5) * 0.5)
            if impurity_context_modulation_enabled:
                context_modulator_raw = self._get_safe_series(df, df, impurity_context_modulator_signal_name, 0.0, method_name="_diagnose_axiom_holder_sentiment")
                normalized_context_modulator = get_adaptive_mtf_normalized_score(context_modulator_raw, df_index, tf_weights=tf_weights, ascending=True)
                overbought_mask = normalized_context_modulator > 0.7
                oversold_mask = normalized_context_modulator < 0.3
                context_adjustment_factor.loc[overbought_mask] = context_adjustment_factor.loc[overbought_mask] * (1 + (normalized_context_modulator.loc[overbought_mask] - 0.7) * impurity_context_overbought_amp_factor / 0.3)
                context_adjustment_factor.loc[oversold_mask] = context_adjustment_factor.loc[oversold_mask] * (1 - (0.3 - normalized_context_modulator.loc[oversold_mask]) * impurity_context_oversold_damp_factor / 0.3)
            # FOMO 效应
            dynamic_fomo_tanh_factor = fomo_tanh_factor * (1 + normalized_sentiment_strength * fomo_sentiment_sensitivity)
            dynamic_fomo_tanh_factor = dynamic_fomo_tanh_factor * context_adjustment_factor
            dynamic_fomo_tanh_factor = dynamic_fomo_tanh_factor.clip(0.5, 3.0)
            fomo_effect = np.tanh(norm_fomo_deviation * dynamic_fomo_tanh_factor)
            # 利润兑现效应
            dynamic_profit_taking_tanh_factor = profit_taking_tanh_factor * (1 + normalized_sentiment_strength * profit_taking_sentiment_sensitivity)
            dynamic_profit_taking_tanh_factor = dynamic_profit_taking_tanh_factor * context_adjustment_factor
            dynamic_profit_taking_tanh_factor = dynamic_profit_taking_tanh_factor.clip(0.5, 3.0)
            profit_taking_effect = np.tanh(norm_profit_taking_quality * dynamic_profit_taking_tanh_factor)
            # V8.0 新增：其他杂质效应
            other_impurity_numeric_weights = {k: v for k, v in impurity_weights.items() if isinstance(v, (int, float)) and k not in ['fomo_concentration', 'profit_taking_margin']}
            total_other_impurity_weight = sum(other_impurity_numeric_weights.values())
            if total_other_impurity_weight > 0:
                other_impurity_score = (
                    norm_upper_shadow_selling_pressure.pow(other_impurity_numeric_weights.get('upper_shadow_selling_pressure', 0.15)) *
                    norm_rally_distribution_pressure.pow(other_impurity_numeric_weights.get('rally_distribution_pressure', 0.15)) *
                    norm_retail_fomo_premium.pow(other_impurity_numeric_weights.get('retail_fomo_premium', 0.1)) *
                    norm_slope_5_winner_profit_margin.pow(other_impurity_numeric_weights.get('winner_profit_margin_slope', 0.1)) *
                    norm_accel_5_retail_fomo_premium.pow(other_impurity_numeric_weights.get('retail_fomo_premium_accel', 0.1))
                ).pow(1 / total_other_impurity_weight)
                other_impurity_effect = np.tanh(other_impurity_score * context_adjustment_factor)
            dynamic_impurity_fusion_exponent = impurity_fusion_exponent_base * (1 - normalized_sentiment_strength * impurity_fusion_exponent_sensitivity)
            dynamic_impurity_fusion_exponent = dynamic_impurity_fusion_exponent.clip(0.1, 1.0)
            # 融合所有杂质效应
            final_impurity_effect = 1 - ((1 - fomo_effect) * (1 - profit_taking_effect) * (1 - other_impurity_effect)).pow(dynamic_impurity_fusion_exponent)
        else:
            final_impurity_effect = pd.Series(0.0, index=df.index)
        if probe_date: print(f"        -> [探针] {probe_date} 最终杂质效应: {final_impurity_effect.asof(probe_date):.4f}")
        final_score = (conviction_base * (1 - final_impurity_effect)) * 2 - 1
        if probe_date: print(f"        -> [探针] {probe_date} 最终持仓信念韧性分数: {final_score.asof(probe_date):.4f}")
        return final_score.clip(-1, 1).fillna(0.0).astype(np.float32)

    def _diagnose_axiom_trend_momentum(self, df: pd.DataFrame, periods: list, strategic_posture: pd.Series, battlefield_geography: pd.Series, holder_sentiment: pd.Series) -> pd.Series:
        """
        【V7.2 · 战略推力引擎版】筹码公理六：诊断“结构性推力”
        - 核心升级1: 引擎功率动态权重。引入筹码健康度趋势作为调制器，动态调整静态基础分与动态变化率的融合权重。
        - 核心升级2: 燃料品质诡道调制。引入筹码故障幅度作为负向调制器，削弱被“诱多”等诡道污染的燃料品质，并使协同奖励情境感知。
        - 核心升级3: 喷管效率多维深化。融合真空区大小、真空区趋势和穿越效率，更全面评估最小阻力路径。
        - 核心升级4: 最终融合动态权重。引入战略态势作为情境调制器，动态调整引擎功率、燃料品质、喷管效率的融合权重。
        - 核心升级5: 新增筹码指标整合：
            - 向上脉冲强度 (`upward_impulse_strength_D`) 增强燃料品质维度。
        - 升级: 优化 synergy_bonus 计算，引入平滑激活函数，避免硬性截断。
        - 升级: 增强最终融合动态权重的情境感知，引入多情境调制器进行综合调整。
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        print("    -> [筹码层] 正在诊断“结构性推力 (V7.2 · 战略推力引擎版)”...")
        df_index = df.index
        probe_date = self.strategy.debug_params.get('target_date') if self.strategy.debug_params.get('enable_mfca_probe') else None
        required_signals = [
            'main_force_conviction_index_D', 'vacuum_zone_magnitude_D', 'upward_impulse_purity_D',
            'chip_health_score_D', 'chip_fault_magnitude_D', 'SLOPE_5_vacuum_zone_magnitude_D',
            'vacuum_traversal_efficiency_D',
            # [修改代码行] V7.2 新增筹码指标
            'upward_impulse_strength_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_trend_momentum"):
            return pd.Series(0.0, index=df.index)
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        trend_momentum_params = get_param_value(p_conf.get('trend_momentum_params'), {})
        health_weights = get_param_value(trend_momentum_params.get('health_weights'), {'posture': 0.4, 'geography': 0.4, 'sentiment': 0.2})
        engine_power_dynamic_weight_modulator_signal_name = get_param_value(trend_momentum_params.get('engine_power_dynamic_weight_modulator_signal_name'), 'SLOPE_5_chip_health_score_D')
        engine_power_dynamic_weight_sensitivity = get_param_value(trend_momentum_params.get('engine_power_dynamic_weight_sensitivity'), 0.5)
        static_engine_power_base_weight = get_param_value(trend_momentum_params.get('static_engine_power_base_weight'), 0.5)
        dynamic_engine_power_base_weight = get_param_value(trend_momentum_params.get('dynamic_engine_power_base_weight'), 0.5)
        fuel_purity_deception_penalty_factor = get_param_value(trend_momentum_params.get('fuel_purity_deception_penalty_factor'), 0.3)
        synergy_bonus_base = get_param_value(trend_momentum_params.get('synergy_bonus_base'), 0.25)
        synergy_bonus_context_modulator_signal_name = get_param_value(trend_momentum_params.get('synergy_bonus_context_modulator_signal_name'), 'chip_health_score_D')
        synergy_bonus_context_sensitivity = get_param_value(trend_momentum_params.get('synergy_bonus_context_sensitivity'), 0.5)
        synergy_activation_threshold = get_param_value(trend_momentum_params.get('synergy_activation_threshold'), 0.0)
        nozzle_efficiency_weights = get_param_value(trend_momentum_params.get('nozzle_efficiency_weights'), {'magnitude': 0.5, 'trend': 0.3, 'traversal': 0.2})
        final_fusion_dynamic_weights_enabled = get_param_value(trend_momentum_params.get('final_fusion_dynamic_weights_enabled'), True)
        final_fusion_weights_base = get_param_value(trend_momentum_params.get('final_fusion_weights_base'), {'engine': 0.33, 'fuel': 0.33, 'nozzle': 0.34})
        final_fusion_weights_sensitivity = get_param_value(trend_momentum_params.get('final_fusion_weights_sensitivity'), {'engine': 0.5, 'fuel': 0.5, 'nozzle': 0.5})
        final_fusion_context_modulators_config = get_param_value(trend_momentum_params.get('final_fusion_context_modulators'), {
            'strategic_posture': {'signal': "strategic_posture", 'weight': 0.5, 'sensitivity': 0.5},
            'battlefield_geography': {'signal': "battlefield_geography", 'weight': 0.3, 'sensitivity': 0.3},
            'holder_sentiment': {'signal': "holder_sentiment", 'weight': 0.2, 'sensitivity': 0.2}
        })
        # [修改代码行] V7.2 新增向上脉冲强度权重
        upward_impulse_strength_weight = get_param_value(trend_momentum_params.get('upward_impulse_strength_weight'), 0.2)
        df_index = df.index
        # [修改代码块] 创建信号映射字典
        signal_map = {
            "strategic_posture": strategic_posture,
            "battlefield_geography": battlefield_geography,
            "holder_sentiment": holder_sentiment
        }
        # --- 1. 引擎功率 (Engine Power) ---
        # 静态引擎功率：由战略态势、战场地形、持仓信念韧性融合而成
        static_engine_power = (
            strategic_posture * health_weights['posture'] +
            battlefield_geography * health_weights['geography'] +
            holder_sentiment * health_weights['sentiment']
        )
        if probe_date: print(f"        -> [探针] {probe_date} 静态引擎功率: {static_engine_power.asof(probe_date):.4f}")
        # 动态调整静态和动态引擎功率的融合权重
        health_score_slope_raw = self._get_safe_series(df, df, engine_power_dynamic_weight_modulator_signal_name, 0.0, method_name="_diagnose_axiom_trend_momentum")
        norm_health_score_slope = get_adaptive_mtf_normalized_bipolar_score(health_score_slope_raw, df_index, tf_weights)
        dynamic_weight_mod = (norm_health_score_slope * engine_power_dynamic_weight_sensitivity).clip(-0.5, 0.5)
        current_static_weight = (static_engine_power_base_weight - dynamic_weight_mod).clip(0.1, 0.9)
        current_dynamic_weight = (dynamic_engine_power_base_weight + dynamic_weight_mod).clip(0.1, 0.9)
        sum_current_weights = current_static_weight + current_dynamic_weight
        current_static_weight = current_static_weight / sum_current_weights
        current_dynamic_weight = current_dynamic_weight / sum_current_weights
        if probe_date: print(f"        -> [探针] {probe_date} 引擎功率动态权重: Static={current_static_weight.asof(probe_date):.4f}, Dynamic={current_dynamic_weight.asof(probe_date):.4f}")
        # 动态引擎功率：由静态引擎功率的速度和加速度组成
        slope = static_engine_power.diff(1).fillna(0)
        accel = slope.diff(1).fillna(0)
        norm_slope = get_adaptive_mtf_normalized_bipolar_score(slope, df_index, tf_weights)
        norm_accel = get_adaptive_mtf_normalized_bipolar_score(accel, df_index, tf_weights)
        dynamic_engine_power = (norm_slope.add(1)/2 * norm_accel.clip(lower=-1, upper=1).add(1)/2).pow(0.5) * 2 - 1
        # 融合静态和动态引擎功率
        engine_power_score = static_engine_power * current_static_weight + dynamic_engine_power * current_dynamic_weight
        if probe_date: print(f"        -> [探针] {probe_date} 引擎功率分数: {engine_power_score.asof(probe_date):.4f}")
        # --- 2. 燃料品质 (Fuel Quality) ---
        conviction_raw = self._get_safe_series(df, df, 'main_force_conviction_index_D', 0.0, method_name="_diagnose_axiom_trend_momentum")
        impulse_purity_raw = self._get_safe_series(df, df, 'upward_impulse_purity_D', 0.0, method_name="_diagnose_axiom_trend_momentum")
        # [修改代码行] V7.2 获取向上脉冲强度
        upward_impulse_strength_raw = self._get_safe_series(df, df, 'upward_impulse_strength_D', 0.0, method_name="_diagnose_axiom_trend_momentum")
        conviction_score = get_adaptive_mtf_normalized_bipolar_score(conviction_raw, df_index, tf_weights)
        purity_score = get_adaptive_mtf_normalized_bipolar_score(impulse_purity_raw, df_index, tf_weights)
        # [修改代码行] V7.2 归一化向上脉冲强度
        norm_upward_impulse_strength = get_adaptive_mtf_normalized_score(upward_impulse_strength_raw, df_index, ascending=True, tf_weights=tf_weights)
        # 基础燃料品质：由主力信念和上涨脉冲纯度融合
        base_fuel_quality = ((conviction_score.add(1)/2) * (purity_score.add(1)/2)).pow(0.5) * 2 - 1
        # [修改代码行] V7.2 整合向上脉冲强度到燃料品质
        base_fuel_quality = base_fuel_quality * (1 + norm_upward_impulse_strength * upward_impulse_strength_weight)
        base_fuel_quality = base_fuel_quality.clip(-1, 1) # 确保分数在 [-1, 1] 范围内
        if probe_date: print(f"        -> [探针] {probe_date} 基础燃料品质分数 (含向上脉冲强度): {base_fuel_quality.asof(probe_date):.4f}")
        # 诡道调制：筹码故障（诱多）对燃料品质的惩罚
        chip_fault_raw = self._get_safe_series(df, df, 'chip_fault_magnitude_D', 0.0, method_name="_diagnose_axiom_trend_momentum")
        norm_chip_fault = get_adaptive_mtf_normalized_score(chip_fault_raw.abs(), df_index, ascending=True, tf_weights=tf_weights)
        deception_penalty = pd.Series(0.0, index=df_index)
        positive_fault_mask = chip_fault_raw > 0 # 正向筹码故障通常代表诱多
        deception_penalty.loc[positive_fault_mask] = norm_chip_fault.loc[positive_fault_mask] * fuel_purity_deception_penalty_factor
        fuel_quality_score_after_deception = base_fuel_quality * (1 - deception_penalty.clip(0, 1))
        if probe_date: print(f"        -> [探针] {probe_date} 诡道调制后燃料品质分数: {fuel_quality_score_after_deception.asof(probe_date):.4f}")
        # 协同奖励：当主力信念和脉冲纯度都高时，给予额外奖励
        synergy_context_raw = self._get_safe_series(df, df, synergy_bonus_context_modulator_signal_name, 0.0, method_name="_diagnose_axiom_trend_momentum")
        norm_synergy_context = get_adaptive_mtf_normalized_score(synergy_context_raw, df_index, ascending=True, tf_weights=tf_weights)
        dynamic_synergy_bonus_factor = synergy_bonus_base * (1 + norm_synergy_context * synergy_bonus_context_sensitivity)
        dynamic_synergy_bonus_factor = dynamic_synergy_bonus_factor.clip(0.1, 0.5)
        conviction_norm = conviction_score.add(1) / 2
        purity_norm = purity_score.add(1) / 2
        synergy_potential = (conviction_norm * purity_norm).pow(0.5)
        synergy_activation = ((synergy_potential - synergy_activation_threshold) / (1 - synergy_activation_threshold)).clip(0, 1)
        synergy_bonus = synergy_activation * dynamic_synergy_bonus_factor
        fuel_quality_score = fuel_quality_score_after_deception + synergy_bonus
        if probe_date: print(f"        -> [探针] {probe_date} 燃料品质分数 (含协同奖励): {fuel_quality_score.asof(probe_date):.4f}")
        # --- 3. 喷管效率 (Nozzle Efficiency) ---
        vacuum_magnitude_raw = self._get_safe_series(df, df, 'vacuum_zone_magnitude_D', 0.0, method_name="_diagnose_axiom_trend_momentum")
        vacuum_trend_raw = self._get_safe_series(df, df, 'SLOPE_5_vacuum_zone_magnitude_D', 0.0, method_name="_diagnose_axiom_trend_momentum")
        vacuum_traversal_raw = self._get_safe_series(df, df, 'vacuum_traversal_efficiency_D', 0.0, method_name="_diagnose_axiom_trend_momentum")
        norm_vacuum_magnitude = get_adaptive_mtf_normalized_bipolar_score(vacuum_magnitude_raw, df_index, tf_weights)
        norm_vacuum_trend = get_adaptive_mtf_normalized_bipolar_score(vacuum_trend_raw, df_index, tf_weights)
        norm_traversal_efficiency = get_adaptive_mtf_normalized_bipolar_score(vacuum_traversal_raw, df_index, tf_weights)
        nozzle_efficiency_score = (
            norm_vacuum_magnitude * nozzle_efficiency_weights.get('magnitude', 0.5) +
            norm_vacuum_trend * nozzle_efficiency_weights.get('trend', 0.3) +
            norm_traversal_efficiency * nozzle_efficiency_weights.get('traversal', 0.2)
        ).clip(-1, 1)
        if probe_date: print(f"        -> [探针] {probe_date} 喷管效率分数: {nozzle_efficiency_score.asof(probe_date):.4f}")
        # --- 4. 最终融合 (Final Fusion) ---
        engine_score_normalized = engine_power_score.add(1)/2
        fuel_score_normalized = fuel_quality_score.clip(-1, 1).add(1)/2
        nozzle_score_normalized = nozzle_efficiency_score.add(1)/2
        # 动态调整最终融合权重
        final_engine_weight = pd.Series(final_fusion_weights_base.get('engine', 0.33), index=df_index)
        final_fuel_weight = pd.Series(final_fusion_weights_base.get('fuel', 0.33), index=df_index)
        final_nozzle_weight = pd.Series(final_fusion_weights_base.get('nozzle', 0.34), index=df_index)
        if final_fusion_dynamic_weights_enabled:
            context_modulator_components = []
            total_context_weight = 0.0
            for ctx_name, ctx_config in final_fusion_context_modulators_config.items():
                signal_key = ctx_config.get('signal')
                signal_series = signal_map.get(signal_key)
                weight = ctx_config.get('weight', 0.0)
                sensitivity = ctx_config.get('sensitivity', 0.0)
                if signal_series is not None and weight > 0:
                    norm_signal = get_adaptive_mtf_normalized_bipolar_score(signal_series, df_index, tf_weights)
                    context_modulator_components.append(norm_signal * weight * sensitivity)
                    total_context_weight += weight * sensitivity
            if context_modulator_components and total_context_weight > 0:
                context_fusion_modulator = sum(context_modulator_components) / total_context_weight
                normalized_fusion_modulator = context_fusion_modulator
            else:
                normalized_fusion_modulator = pd.Series(0.0, index=df_index)
            if probe_date: print(f"        -> [探针] {probe_date} 最终融合情境调制器: {normalized_fusion_modulator.asof(probe_date):.4f}")
            # 根据情境调制器调整各维度的权重
            engine_mod = normalized_fusion_modulator * final_fusion_weights_sensitivity.get('engine', 0.5)
            fuel_mod = normalized_fusion_modulator * final_fusion_weights_sensitivity.get('fuel', 0.5)
            nozzle_mod = -normalized_fusion_modulator * final_fusion_weights_sensitivity.get('nozzle', 0.5) # 喷管效率在情境好时权重降低，情境差时权重增加
            final_engine_weight = (final_fusion_weights_base.get('engine', 0.33) + engine_mod).clip(0.1, 0.6)
            final_fuel_weight = (final_fusion_weights_base.get('fuel', 0.33) + fuel_mod).clip(0.1, 0.6)
            final_nozzle_weight = (final_fusion_weights_base.get('nozzle', 0.34) + nozzle_mod).clip(0.1, 0.6)
            # 重新归一化动态权重，使其和为1
            sum_dynamic_fusion_weights = final_engine_weight + final_fuel_weight + final_nozzle_weight
            final_engine_weight = final_engine_weight / sum_dynamic_fusion_weights
            final_fuel_weight = final_fuel_weight / sum_dynamic_fusion_weights
            final_nozzle_weight = final_nozzle_weight / sum_dynamic_fusion_weights
            if probe_date: print(f"        -> [探针] {probe_date} 最终融合动态权重: Engine={final_engine_weight.asof(probe_date):.4f}, Fuel={final_fuel_weight.asof(probe_date):.4f}, Nozzle={final_nozzle_weight.asof(probe_date):.4f}")
        # 最终融合得分
        final_score = (
            engine_score_normalized.pow(final_engine_weight) *
            fuel_score_normalized.pow(final_fuel_weight) *
            nozzle_score_normalized.pow(final_nozzle_weight)
        ).pow(1 / (final_engine_weight + final_fuel_weight + final_nozzle_weight)) * 2 - 1
        if probe_date: print(f"        -> [探针] {probe_date} 最终结构性推力分数: {final_score.asof(probe_date):.4f}")
        return final_score.clip(-1, 1).fillna(0.0).astype(np.float32)

    def _diagnose_axiom_divergence(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V7.2 · 情境自适应张力版】筹码公理五：诊断“价筹张力”
        - 核心数学升级1: 将“主力共谋验证”从依赖资金流信号升级为更纯粹、更稳健的“主力筹码意图验证”模型。
                          该模型直接评估1)主力筹码信念是否与背离方向一致(同谋), 2)主力信念强度是否足够大(兵力)。
                          只有当两者都满足时，才确认为一次高置信度的“战术性背离”，并给予显著加成。
        - 核心数学升级2: “筹码趋势”的多元化解读。引入赢家集中度与赢家/输家动量共同构建复合筹码趋势，更全面捕捉筹码结构与价格的分歧。
        - 核心数学升级3: “持续性”的优化。将持续性量化为分歧方向的一致性累积，而非波动性，更准确反映张力积蓄。
        - 核心数学升级4: “能量注入”的筹码化。替换通用成交量为建设性换手率，更精准反映筹码层面的活跃度与质量。
        - 核心数学升级5: “诡道双向调制”。引入筹码故障幅度对分歧强度进行情境调制，根据故障与分歧方向的匹配关系，动态地放大或削弱价筹张力信号。
        - 核心数学升级6: “情境自适应放大器”。引入筹码健康度作为情境调制器，动态调整张力强度和主力意图验证的放大倍数。
        - 核心数学升级7: “非线性放大控制”。对放大项引入tanh变换，使其增长更平滑，并有饱和上限，防止过度放大。
        - 核心数学升级8: “动态复合筹码趋势权重”。引入筹码波动不稳定性指数作为调制器，自适应调整复合筹码趋势中动量和集中度的权重。
        """
        required_signals = [
            'winner_loser_momentum_D', 'winner_concentration_90pct_D', 'SLOPE_5_close_D',
            'constructive_turnover_ratio_D', 'main_force_conviction_index_D', 'chip_fault_magnitude_D',
            'chip_health_score_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_divergence"):
            return pd.Series(0.0, index=df.index)
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        divergence_params = get_param_value(p_conf.get('divergence_params'), {})
        chip_trend_momentum_weight_base = get_param_value(divergence_params.get('chip_trend_momentum_weight'), 0.6)
        chip_trend_concentration_weight_base = get_param_value(divergence_params.get('chip_trend_concentration_weight'), 0.4)
        tension_magnitude_amplifier_base = get_param_value(divergence_params.get('tension_magnitude_amplifier'), 1.5)
        chip_intent_factor_amplifier_base = get_param_value(divergence_params.get('chip_intent_factor_amplifier'), 0.5)
        deception_modulator_impact_clip = get_param_value(divergence_params.get('deception_modulator_impact_clip'), 0.5)
        conflict_bonus = get_param_value(divergence_params.get('conflict_bonus'), 0.5)
        deception_modulator_reinforce_factor = get_param_value(divergence_params.get('deception_modulator_reinforce_factor'), 0.5)
        contextual_amplification_enabled = get_param_value(divergence_params.get('contextual_amplification_enabled'), True)
        context_modulator_signal_name = get_param_value(divergence_params.get('context_modulator_signal_name'), 'chip_health_score_D')
        context_sensitivity_tension = get_param_value(divergence_params.get('context_sensitivity_tension'), 0.5)
        context_sensitivity_intent = get_param_value(divergence_params.get('context_sensitivity_intent'), 0.5)
        non_linear_amplification_enabled = get_param_value(divergence_params.get('non_linear_amplification_enabled'), True)
        non_linear_amp_tanh_factor = get_param_value(divergence_params.get('non_linear_amp_tanh_factor'), 1.0)
        dynamic_chip_trend_weights_enabled = get_param_value(divergence_params.get('dynamic_chip_trend_weights_enabled'), True)
        chip_trend_weight_modulator_signal_name = get_param_value(divergence_params.get('chip_trend_weight_modulator_signal_name'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        chip_trend_weight_mod_sensitivity = get_param_value(divergence_params.get('chip_trend_weight_mod_sensitivity'), 0.5)
        df_index = df.index
        dynamic_momentum_weight = pd.Series(chip_trend_momentum_weight_base, index=df_index)
        dynamic_concentration_weight = pd.Series(chip_trend_concentration_weight_base, index=df_index)
        if dynamic_chip_trend_weights_enabled:
            chip_trend_modulator_raw = self._get_safe_series(df, df, chip_trend_weight_modulator_signal_name, 0.0, method_name="_diagnose_axiom_divergence")
            normalized_chip_trend_modulator = get_adaptive_mtf_normalized_score(chip_trend_modulator_raw, df_index, tf_weights=tf_weights, ascending=True)
            dynamic_momentum_weight = chip_trend_momentum_weight_base * (1 + normalized_chip_trend_modulator * chip_trend_weight_mod_sensitivity)
            dynamic_concentration_weight = chip_trend_concentration_weight_base * (1 - normalized_chip_trend_modulator * chip_trend_weight_mod_sensitivity)
            sum_dynamic_weights = dynamic_momentum_weight + dynamic_concentration_weight
            dynamic_momentum_weight = (dynamic_momentum_weight / sum_dynamic_weights).clip(0.1, 0.9)
            dynamic_concentration_weight = (dynamic_concentration_weight / sum_dynamic_weights).clip(0.1, 0.9)
        chip_momentum_raw = self._get_safe_series(df, df, 'winner_loser_momentum_D', 0.0, method_name="_diagnose_axiom_divergence")
        chip_concentration_raw = self._get_safe_series(df, df, 'winner_concentration_90pct_D', 0.0, method_name="_diagnose_axiom_divergence")
        norm_chip_momentum = get_adaptive_mtf_normalized_bipolar_score(chip_momentum_raw, df_index, tf_weights)
        norm_chip_concentration = get_adaptive_mtf_normalized_bipolar_score(chip_concentration_raw, df_index, tf_weights)
        composite_chip_trend = (
            norm_chip_momentum * dynamic_momentum_weight +
            norm_chip_concentration * dynamic_concentration_weight
        )
        price_trend_raw = self._get_safe_series(df, df, 'SLOPE_5_close_D', 0.0, method_name="_diagnose_axiom_divergence")
        norm_price_trend = get_adaptive_mtf_normalized_bipolar_score(price_trend_raw, df_index, tf_weights)
        disagreement_vector = composite_chip_trend - norm_price_trend
        persistence_raw = np.sign(disagreement_vector).rolling(window=13, min_periods=5).sum().fillna(0)
        norm_persistence = get_adaptive_mtf_normalized_score(persistence_raw.abs(), df_index, tf_weights=tf_weights)
        constructive_turnover_raw = self._get_safe_series(df, df, 'constructive_turnover_ratio_D', 0.0, method_name="_diagnose_axiom_divergence")
        norm_constructive_turnover = get_adaptive_mtf_normalized_score(constructive_turnover_raw, df_index, tf_weights=tf_weights)
        energy_injection = norm_constructive_turnover * disagreement_vector.abs()
        tension_magnitude = (norm_persistence * energy_injection).pow(0.5)
        mf_chip_conviction_raw = self._get_safe_series(df, df, 'main_force_conviction_index_D', 0.0)
        norm_mf_chip_conviction = get_adaptive_mtf_normalized_bipolar_score(mf_chip_conviction_raw, df_index, tf_weights)
        is_aligned = (np.sign(disagreement_vector) * np.sign(norm_mf_chip_conviction)) > 0
        intent_strength = norm_mf_chip_conviction.abs()
        chip_intent_verification_score = is_aligned * intent_strength
        dynamic_tension_amplifier = pd.Series(tension_magnitude_amplifier_base, index=df_index)
        dynamic_chip_intent_factor_amplifier = pd.Series(chip_intent_factor_amplifier_base, index=df_index)
        if contextual_amplification_enabled:
            context_modulator_raw = self._get_safe_series(df, df, context_modulator_signal_name, 0.0, method_name="_diagnose_axiom_divergence")
            normalized_context = get_adaptive_mtf_normalized_score(context_modulator_raw, df_index, tf_weights=tf_weights, ascending=True)
            dynamic_tension_amplifier = tension_magnitude_amplifier_base * (1 + normalized_context * context_sensitivity_tension)
            dynamic_chip_intent_factor_amplifier = chip_intent_factor_amplifier_base * (1 + normalized_context * context_sensitivity_intent)
            dynamic_tension_amplifier = dynamic_tension_amplifier.clip(tension_magnitude_amplifier_base * 0.5, tension_magnitude_amplifier_base * 2.0)
            dynamic_chip_intent_factor_amplifier = dynamic_chip_intent_factor_amplifier.clip(chip_intent_factor_amplifier_base * 0.5, chip_intent_factor_amplifier_base * 2.0)
        tension_amplification_term = tension_magnitude * dynamic_tension_amplifier
        chip_intent_amplification_term = chip_intent_verification_score * dynamic_chip_intent_factor_amplifier
        if non_linear_amplification_enabled:
            tension_amplification_term = np.tanh(tension_amplification_term * non_linear_amp_tanh_factor)
            chip_intent_amplification_term = np.tanh(chip_intent_amplification_term * non_linear_amp_tanh_factor)
        chip_intent_factor = 1.0 + chip_intent_amplification_term
        chip_fault_raw = self._get_safe_series(df, df, 'chip_fault_magnitude_D', 0.0)
        norm_chip_fault = get_adaptive_mtf_normalized_score(chip_fault_raw.abs(), df_index, ascending=True, tf_weights=tf_weights)
        divergence_sign = np.sign(disagreement_vector)
        fault_sign = np.sign(chip_fault_raw)
        deception_modulator_factor = pd.Series(1.0, index=df_index)
        align_mask = (divergence_sign == fault_sign)
        deception_modulator_factor.loc[align_mask] = 1 - norm_chip_fault.loc[align_mask] * deception_modulator_impact_clip
        oppose_mask = (divergence_sign != fault_sign)
        deception_modulator_factor.loc[oppose_mask] = 1 + norm_chip_fault.loc[oppose_mask] * deception_modulator_reinforce_factor
        deception_modulator_factor = deception_modulator_factor.clip(0.1, 2.0)
        base_final_score = disagreement_vector * (1 + tension_amplification_term) * chip_intent_factor * deception_modulator_factor
        conflict_mask = (np.sign(composite_chip_trend) * np.sign(norm_price_trend) < 0)
        conflict_amplifier = pd.Series(1.0, index=df_index)
        conflict_amplifier.loc[conflict_mask] = 1.0 + conflict_bonus
        safe_base_score = base_final_score.clip(-0.999, 0.999)
        final_score = np.tanh(np.arctanh(safe_base_score) * conflict_amplifier)
        return final_score.clip(-1, 1).fillna(0.0).astype(np.float32)

    def _diagnose_structural_consensus(self, df: pd.DataFrame, cost_structure_scores: pd.Series, holder_sentiment_scores: pd.Series) -> pd.Series:
        """
        【V7.18 · 最终分数敏感度动态版 (生产就绪版)】诊断筹码同调驱动力
        一个基于“引擎-传动”思想的终极信号，旨在量化筹码结构对上涨意愿的真实转化效率。
        它将“持股心态”视为提供上涨意愿的引擎，将“成本结构”视为决定能量损耗的传动系统。
        核心升级:
        - 筹码健康度 `chip_health_score_D` 作为非线性调制参数（amplification_power, dampening_power）的动态调节器。
        - 筹码健康度对幂指数的敏感度根据另一个筹码层面的信号（例如 `VOLATILITY_INSTABILITY_INDEX_21d_D` 筹码波动性）进行动态调整。
        - 筹码结构分数 `cost_structure_scores` 对情绪驱动力的调制强度，根据持股心态 `holder_sentiment_scores` 的正负方向，进行非对称的非线性动态调整。
        - 情绪与筹码结构之间的耦合强度也实现了动态调整。
        - 筹码健康度调制敏感度引入了非对称性。
        - 动态中性阈值使得判断情绪和筹码结构是看涨/看跌或顺风/逆风的“中性”界限，将根据筹码健康度动态调整。
        - 情绪激活阈值使得持股心态的原始强度在参与驱动力计算之前，会根据其与动态中性阈值的相对关系进行“激活”或“去激活”处理。
        - 情绪强度对筹码结构调制效果的动态缩放，激活后的情绪强度将动态缩放筹码结构分数对驱动力的最终影响。
        - 结构强度对幂指数的自适应调整，amplification_power 和 dampening_power 将根据最终用于调制的筹码结构分数的绝对强度进行进一步的动态调整。
        - 结构强度对幂指数自适应调整的敏感度动态调制，使得模型在不同市场环境下对筹码结构信号的反应更加精细和智能。
        - 结构强度对幂指数自适应调整的非对称非线性映射，为正向和负向结构强度引入独立的 tanh 因子和可选的偏移量。
        - 最终分数敏感度的动态调整，final_score 的饱和速度将根据市场环境进行动态调整。
        高分代表市场不仅想涨，而且其内部筹码结构健康且具备高效转化这种意愿的能力。
        """
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        coherent_drive_params = get_param_value(p_conf.get('coherent_drive_params'), {})
        base_amplification_power = get_param_value(coherent_drive_params.get('amplification_power'), 1.2)
        base_dampening_power = get_param_value(coherent_drive_params.get('dampening_power'), 1.5)
        chip_health_modulation_enabled = get_param_value(coherent_drive_params.get('chip_health_modulation_enabled'), True)
        default_chip_health_sensitivity_amp = get_param_value(coherent_drive_params.get('chip_health_sensitivity_amp'), 0.5)
        default_chip_health_sensitivity_damp = get_param_value(coherent_drive_params.get('chip_health_sensitivity_damp'), 0.5)
        chip_health_mtf_norm_params = get_param_value(coherent_drive_params.get('chip_health_mtf_norm_params'), {})
        chip_health_tanh_factor_amp = get_param_value(coherent_drive_params.get('chip_health_tanh_factor_amp'), 1.0)
        chip_health_tanh_factor_damp = get_param_value(coherent_drive_params.get('chip_health_tanh_factor_damp'), 1.0)
        chip_health_sensitivity_modulation_enabled = get_param_value(coherent_drive_params.get('chip_health_sensitivity_modulation_enabled'), False)
        chip_sensitivity_modulator_signal_name = get_param_value(coherent_drive_params.get('chip_sensitivity_modulator_signal_name'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        chip_sensitivity_mod_norm_window = get_param_value(coherent_drive_params.get('chip_sensitivity_mod_norm_window'), 21)
        chip_sensitivity_mod_factor_amp = get_param_value(coherent_drive_params.get('chip_sensitivity_mod_factor_amp'), 1.0)
        chip_sensitivity_mod_factor_damp = get_param_value(coherent_drive_params.get('chip_sensitivity_mod_factor_damp'), 1.0)
        chip_sensitivity_mod_tanh_factor_amp = get_param_value(coherent_drive_params.get('chip_sensitivity_mod_tanh_factor_amp'), 1.0)
        chip_sensitivity_mod_tanh_factor_damp = get_param_value(coherent_drive_params.get('chip_sensitivity_mod_tanh_factor_damp'), 1.0)
        cost_structure_asymmetric_impact_enabled = get_param_value(coherent_drive_params.get('cost_structure_asymmetric_impact_enabled'), False)
        cost_structure_impact_base_factor_bullish = get_param_value(coherent_drive_params.get('cost_structure_impact_base_factor_bullish'), 1.0)
        cost_structure_impact_base_factor_bearish = get_param_value(coherent_drive_params.get('cost_structure_impact_base_factor_bearish'), 1.0)
        cost_structure_impact_sentiment_sensitivity_bullish = get_param_value(coherent_drive_params.get('cost_structure_impact_sentiment_sensitivity_bullish'), 1.0)
        cost_structure_impact_sentiment_sensitivity_bearish = get_param_value(coherent_drive_params.get('cost_structure_impact_sentiment_sensitivity_bearish'), 1.0)
        cost_structure_impact_sentiment_tanh_factor_bullish = get_param_value(coherent_drive_params.get('cost_structure_impact_sentiment_tanh_factor_bullish'), 1.0)
        cost_structure_impact_sentiment_tanh_factor_bearish = get_param_value(coherent_drive_params.get('cost_structure_impact_sentiment_tanh_factor_bearish'), 1.0)
        sentiment_cost_structure_coupling_enabled = get_param_value(coherent_drive_params.get('sentiment_cost_structure_coupling_enabled'), False)
        sentiment_coupling_base_factor = get_param_value(coherent_drive_params.get('sentiment_coupling_base_factor'), 1.0)
        sentiment_coupling_tanh_factor = get_param_value(coherent_drive_params.get('sentiment_coupling_tanh_factor'), 1.0)
        sentiment_coupling_sensitivity = get_param_value(coherent_drive_params.get('sentiment_coupling_sensitivity'), 1.0)
        chip_health_asymmetric_sensitivity_enabled = get_param_value(coherent_drive_params.get('chip_health_asymmetric_sensitivity_enabled'), False)
        chip_health_sensitivity_amp_positive_health = get_param_value(coherent_drive_params.get('chip_health_sensitivity_amp_positive_health'), 0.5)
        chip_health_sensitivity_amp_negative_health = get_param_value(coherent_drive_params.get('chip_health_sensitivity_amp_negative_health'), 0.5)
        chip_health_sensitivity_damp_positive_health = get_param_value(coherent_drive_params.get('chip_health_sensitivity_damp_positive_health'), 0.5)
        chip_health_sensitivity_damp_negative_health = get_param_value(coherent_drive_params.get('chip_health_sensitivity_damp_negative_health'), 0.5)
        dynamic_neutrality_thresholds_enabled = get_param_value(coherent_drive_params.get('dynamic_neutrality_thresholds_enabled'), False)
        sentiment_neutrality_base_threshold = get_param_value(coherent_drive_params.get('sentiment_neutrality_base_threshold'), 0.0)
        sentiment_neutrality_chip_health_sensitivity = get_param_value(coherent_drive_params.get('sentiment_neutrality_chip_health_sensitivity'), 0.1)
        cost_structure_neutrality_base_threshold = get_param_value(coherent_drive_params.get('cost_structure_neutrality_base_threshold'), 0.0)
        cost_structure_neutrality_chip_health_sensitivity = get_param_value(coherent_drive_params.get('cost_structure_neutrality_chip_health_sensitivity'), 0.1)
        sentiment_activation_enabled = get_param_value(coherent_drive_params.get('sentiment_activation_enabled'), False)
        sentiment_activation_tanh_factor = get_param_value(coherent_drive_params.get('sentiment_activation_tanh_factor'), 1.0)
        sentiment_activation_strength = get_param_value(coherent_drive_params.get('sentiment_activation_strength'), 1.0)
        structure_modulation_strength_enabled = get_param_value(coherent_drive_params.get('structure_modulation_strength_enabled'), False)
        structure_modulation_base_strength = get_param_value(coherent_drive_params.get('structure_modulation_base_strength'), 1.0)
        structure_modulation_sentiment_tanh_factor = get_param_value(coherent_drive_params.get('structure_modulation_sentiment_tanh_factor'), 1.0)
        structure_modulation_sentiment_sensitivity = get_param_value(coherent_drive_params.get('structure_modulation_sentiment_sensitivity'), 1.0)
        structural_power_adjustment_enabled = get_param_value(coherent_drive_params.get('structural_power_adjustment_enabled'), False)
        default_structural_power_sensitivity_amp = get_param_value(coherent_drive_params.get('structural_power_sensitivity_amp'), 0.5)
        default_structural_power_sensitivity_damp = get_param_value(coherent_drive_params.get('structural_power_sensitivity_damp'), 0.5)
        default_structural_power_tanh_factor_amp = get_param_value(coherent_drive_params.get('structural_power_tanh_factor_amp'), 1.0)
        default_structural_power_tanh_factor_damp = get_param_value(coherent_drive_params.get('structural_power_tanh_factor_damp'), 1.0)
        structural_power_sensitivity_modulation_enabled = get_param_value(coherent_drive_params.get('structural_power_sensitivity_modulation_enabled'), False)
        structural_power_modulator_signal_name = get_param_value(coherent_drive_params.get('structural_power_modulator_signal_name'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        structural_power_mod_norm_window = get_param_value(coherent_drive_params.get('structural_power_mod_norm_window'), 21)
        structural_power_mod_factor_amp = get_param_value(coherent_drive_params.get('structural_power_mod_factor_amp'), 1.0)
        structural_power_mod_factor_damp = get_param_value(coherent_drive_params.get('structural_power_mod_factor_damp'), 1.0)
        structural_power_mod_tanh_factor_amp = get_param_value(coherent_drive_params.get('structural_power_mod_tanh_factor_amp'), 1.0)
        structural_power_mod_tanh_factor_damp = get_param_value(coherent_drive_params.get('structural_power_mod_tanh_factor_damp'), 1.0)
        structural_power_asymmetric_tanh_enabled = get_param_value(coherent_drive_params.get('structural_power_asymmetric_tanh_enabled'), False)
        structural_power_tanh_factor_positive_structure = get_param_value(coherent_drive_params.get('structural_power_tanh_factor_positive_structure'), 1.0)
        structural_power_tanh_factor_negative_structure = get_param_value(coherent_drive_params.get('structural_power_tanh_factor_negative_structure'), 1.0)
        structural_power_offset_positive_structure = get_param_value(coherent_drive_params.get('structural_power_offset_positive_structure'), 0.0)
        structural_power_offset_negative_structure = get_param_value(coherent_drive_params.get('structural_power_offset_negative_structure'), 0.0)
        final_score_sensitivity_modulation_enabled = get_param_value(coherent_drive_params.get('final_score_sensitivity_modulation_enabled'), False)
        final_score_modulator_signal_name = get_param_value(coherent_drive_params.get('final_score_modulator_signal_name'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        final_score_mod_norm_window = get_param_value(coherent_drive_params.get('final_score_mod_norm_window'), 21)
        final_score_mod_factor = get_param_value(coherent_drive_params.get('final_score_mod_factor'), 1.0)
        final_score_mod_tanh_factor = get_param_value(coherent_drive_params.get('final_score_mod_tanh_factor'), 1.0)
        final_score_base_sensitivity_multiplier = get_param_value(coherent_drive_params.get('final_score_base_sensitivity_multiplier'), 2.0)
        amplification_power = pd.Series(base_amplification_power, index=df.index)
        dampening_power = pd.Series(base_dampening_power, index=df.index)
        modulation_factor = pd.Series(1.0, index=df.index)
        current_chip_health_score_raw = pd.Series(0.0, index=df.index)
        normalized_chip_health = pd.Series(0.0, index=df.index)
        dynamic_chip_health_sensitivity_amp = pd.Series(default_chip_health_sensitivity_amp, index=df.index)
        dynamic_chip_health_sensitivity_damp = pd.Series(default_chip_health_sensitivity_damp, index=df.index)
        dynamic_cost_structure_impact_factor_bullish = pd.Series(cost_structure_impact_base_factor_bullish, index=df.index)
        dynamic_cost_structure_impact_factor_bearish = pd.Series(cost_structure_impact_base_factor_bearish, index=df.index)
        dynamic_coupling_factor = pd.Series(sentiment_coupling_base_factor, index=df.index)
        final_cost_structure_for_modulation = pd.Series(0.0, index=df.index)
        dynamic_sentiment_neutrality_threshold = pd.Series(sentiment_neutrality_base_threshold, index=df.index)
        dynamic_cost_structure_neutrality_threshold = pd.Series(cost_structure_neutrality_base_threshold, index=df.index)
        activated_holder_sentiment_scores = holder_sentiment_scores.copy()
        dynamic_structure_modulation_strength = pd.Series(structure_modulation_base_strength, index=df.index)
        final_cost_structure_for_modulation_scaled = pd.Series(0.0, index=df.index)
        dynamic_structural_power_sensitivity_amp = pd.Series(default_structural_power_sensitivity_amp, index=df.index)
        dynamic_structural_power_sensitivity_damp = pd.Series(default_structural_power_sensitivity_damp, index=df.index)
        dynamic_final_score_sensitivity_multiplier = pd.Series(final_score_base_sensitivity_multiplier, index=df.index)
        if chip_health_modulation_enabled:
            current_chip_health_score_raw = self._get_safe_series(df, df, 'chip_health_score_D', 0.0, method_name="_diagnose_structural_consensus")
            normalized_chip_health = get_adaptive_mtf_normalized_bipolar_score(
                current_chip_health_score_raw, 
                df.index, 
                tf_weights=chip_health_mtf_norm_params.get('weights', {}),
                sensitivity=chip_health_mtf_norm_params.get('sensitivity', 2.0)
            )
            base_amp_sensitivity_series = pd.Series(default_chip_health_sensitivity_amp, index=df.index)
            base_damp_sensitivity_series = pd.Series(default_chip_health_sensitivity_damp, index=df.index)
            if chip_health_asymmetric_sensitivity_enabled:
                positive_health_mask = normalized_chip_health > 0
                negative_health_mask = normalized_chip_health < 0
                base_amp_sensitivity_series.loc[positive_health_mask] = chip_health_sensitivity_amp_positive_health
                base_amp_sensitivity_series.loc[negative_health_mask] = chip_health_sensitivity_amp_negative_health
                base_damp_sensitivity_series.loc[positive_health_mask] = chip_health_sensitivity_damp_positive_health
                base_damp_sensitivity_series.loc[negative_health_mask] = chip_health_sensitivity_damp_negative_health
            if chip_health_sensitivity_modulation_enabled:
                modulator_signal_raw = self._get_safe_series(df, df, chip_sensitivity_modulator_signal_name, 0.0, method_name="_diagnose_structural_consensus")
                normalized_modulator_signal = normalize_score(
                    modulator_signal_raw,
                    df.index,
                    window=chip_sensitivity_mod_norm_window,
                    ascending=True
                )
                modulator_bipolar = (normalized_modulator_signal * 2) - 1
                non_linear_modulator_effect_amp = np.tanh(modulator_bipolar * chip_sensitivity_mod_tanh_factor_amp)
                non_linear_modulator_effect_damp = np.tanh(modulator_bipolar * chip_sensitivity_mod_tanh_factor_damp)
                dynamic_chip_health_sensitivity_amp = base_amp_sensitivity_series * (1 + non_linear_modulator_effect_amp * chip_sensitivity_mod_factor_amp)
                dynamic_chip_health_sensitivity_damp = base_damp_sensitivity_series * (1 + non_linear_modulator_effect_damp * chip_sensitivity_mod_factor_damp)
                dynamic_chip_health_sensitivity_amp = dynamic_chip_health_sensitivity_amp.clip(base_amp_sensitivity_series * 0.1, base_amp_sensitivity_series * 2.0)
                dynamic_chip_health_sensitivity_damp = dynamic_chip_health_sensitivity_damp.clip(base_damp_sensitivity_series * 0.1, base_damp_sensitivity_series * 2.0) # [修改代码行] 修正变量名
            else:
                dynamic_chip_health_sensitivity_amp = base_amp_sensitivity_series
                dynamic_chip_health_sensitivity_damp = base_damp_sensitivity_series
            modulated_chip_health_amp = np.tanh(normalized_chip_health * chip_health_tanh_factor_amp)
            modulated_chip_health_damp = np.tanh(normalized_chip_health * chip_health_tanh_factor_damp)
            amplification_power = base_amplification_power * (1 + modulated_chip_health_amp * dynamic_chip_health_sensitivity_amp)
            dampening_power = base_dampening_power * (1 - modulated_chip_health_damp * dynamic_chip_health_sensitivity_damp)
            amplification_power = amplification_power.clip(0.5, 2.0) 
            dampening_power = dampening_power.clip(0.5, 2.0) 
        if dynamic_neutrality_thresholds_enabled:
            dynamic_sentiment_neutrality_threshold = sentiment_neutrality_base_threshold + (normalized_chip_health * sentiment_neutrality_chip_health_sensitivity)
            dynamic_cost_structure_neutrality_threshold = cost_structure_neutrality_base_threshold + (normalized_chip_health * cost_structure_neutrality_chip_health_sensitivity)
            dynamic_sentiment_neutrality_threshold = dynamic_sentiment_neutrality_threshold.clip(-0.2, 0.2)
            dynamic_cost_structure_neutrality_threshold = dynamic_cost_structure_neutrality_threshold.clip(-0.2, 0.2)
        if sentiment_activation_enabled:
            positive_active_mask = holder_sentiment_scores > dynamic_sentiment_neutrality_threshold
            negative_active_mask = holder_sentiment_scores < -dynamic_sentiment_neutrality_threshold
            neutral_mask = ~(positive_active_mask | negative_active_mask)
            activated_holder_sentiment_scores.loc[positive_active_mask] = \
                holder_sentiment_scores.loc[positive_active_mask] - dynamic_sentiment_neutrality_threshold.loc[positive_active_mask]
            activated_holder_sentiment_scores.loc[negative_active_mask] = \
                holder_sentiment_scores.loc[negative_active_mask] + dynamic_sentiment_neutrality_threshold.loc[negative_active_mask]
            activated_holder_sentiment_scores.loc[neutral_mask] = 0.0
            activated_holder_sentiment_scores = np.tanh(activated_holder_sentiment_scores * sentiment_activation_tanh_factor) * sentiment_activation_strength
        if cost_structure_asymmetric_impact_enabled:
            positive_sentiment_mask = holder_sentiment_scores > 0
            if positive_sentiment_mask.any():
                positive_sentiment_strength = holder_sentiment_scores[positive_sentiment_mask]
                normalized_positive_sentiment_tanh = np.tanh(positive_sentiment_strength * cost_structure_impact_sentiment_tanh_factor_bullish)
                dynamic_cost_structure_impact_factor_bullish.loc[positive_sentiment_mask] = \
                    cost_structure_impact_base_factor_bullish * (1 + (normalized_positive_sentiment_tanh - 0.5) * cost_structure_impact_sentiment_sensitivity_bullish)
                dynamic_cost_structure_impact_factor_bullish = dynamic_cost_structure_impact_factor_bullish.clip(0.1, 2.0)
            negative_sentiment_mask = holder_sentiment_scores < 0
            if negative_sentiment_mask.any():
                negative_sentiment_strength = holder_sentiment_scores[negative_sentiment_mask].abs()
                normalized_negative_sentiment_tanh = np.tanh(negative_sentiment_strength * cost_structure_impact_sentiment_tanh_factor_bearish)
                dynamic_cost_structure_impact_factor_bearish.loc[negative_sentiment_mask] = \
                    cost_structure_impact_base_factor_bearish * (1 + (normalized_negative_sentiment_tanh - 0.5) * cost_structure_impact_sentiment_sensitivity_bearish)
                dynamic_cost_structure_impact_factor_bearish = dynamic_cost_structure_impact_factor_bearish.clip(0.1, 2.0)
        selected_dynamic_cost_structure_impact_factor = pd.Series(1.0, index=df.index)
        selected_dynamic_cost_structure_impact_factor.loc[holder_sentiment_scores > 0] = dynamic_cost_structure_impact_factor_bullish.loc[holder_sentiment_scores > 0]
        selected_dynamic_cost_structure_impact_factor.loc[holder_sentiment_scores < 0] = dynamic_cost_structure_impact_factor_bearish.loc[holder_sentiment_scores < 0]
        adjusted_cost_structure_scores = cost_structure_scores * selected_dynamic_cost_structure_impact_factor
        if sentiment_cost_structure_coupling_enabled:
            abs_holder_sentiment = holder_sentiment_scores.abs()
            sentiment_tanh_modulated = np.tanh(abs_holder_sentiment * sentiment_coupling_tanh_factor)
            dynamic_coupling_factor = sentiment_coupling_base_factor * (1 + sentiment_tanh_modulated * sentiment_coupling_sensitivity)
            dynamic_coupling_factor = dynamic_coupling_factor.clip(0.1, 2.0)
        final_cost_structure_for_modulation = adjusted_cost_structure_scores * dynamic_coupling_factor
        if structure_modulation_strength_enabled:
            abs_activated_sentiment = activated_holder_sentiment_scores.abs()
            sentiment_tanh_modulated_for_structure = np.tanh(abs_activated_sentiment * structure_modulation_sentiment_tanh_factor)
            dynamic_structure_modulation_strength = structure_modulation_base_strength * (1 + sentiment_tanh_modulated_for_structure * structure_modulation_sentiment_sensitivity)
            dynamic_structure_modulation_strength = dynamic_structure_modulation_strength.clip(0.1, 2.0)
        final_cost_structure_for_modulation_scaled = final_cost_structure_for_modulation * dynamic_structure_modulation_strength
        if structural_power_sensitivity_modulation_enabled:
            structural_power_modulator_signal_raw = self._get_safe_series(df, df, structural_power_modulator_signal_name, 0.0, method_name="_diagnose_structural_consensus")
            structural_power_normalized_modulator_signal = normalize_score(
                structural_power_modulator_signal_raw,
                df.index,
                window=structural_power_mod_norm_window,
                ascending=True
            )
            structural_power_modulator_bipolar = (structural_power_normalized_modulator_signal * 2) - 1
            structural_power_non_linear_modulator_effect_amp = np.tanh(structural_power_modulator_bipolar * structural_power_mod_tanh_factor_amp)
            structural_power_non_linear_modulator_effect_damp = np.tanh(structural_power_modulator_bipolar * structural_power_mod_tanh_factor_damp)
            dynamic_structural_power_sensitivity_amp = default_structural_power_sensitivity_amp * (1 + structural_power_non_linear_modulator_effect_amp * structural_power_mod_factor_amp)
            dynamic_structural_power_sensitivity_damp = default_structural_power_sensitivity_damp * (1 + structural_power_non_linear_modulator_effect_damp * structural_power_mod_factor_damp)
            dynamic_structural_power_sensitivity_amp = dynamic_structural_power_sensitivity_amp.clip(default_structural_power_sensitivity_amp * 0.1, default_structural_power_sensitivity_amp * 2.0)
            dynamic_structural_power_sensitivity_damp = dynamic_structural_power_sensitivity_damp.clip(default_structural_power_sensitivity_damp * 0.1, default_structural_power_sensitivity_damp * 2.0)
        else:
            dynamic_structural_power_sensitivity_amp = pd.Series(default_structural_power_sensitivity_amp, index=df.index)
            dynamic_structural_power_sensitivity_damp = pd.Series(default_structural_power_sensitivity_damp, index=df.index)
        if structural_power_adjustment_enabled:
            positive_structure_mask = final_cost_structure_for_modulation_scaled > 0
            negative_structure_mask = final_cost_structure_for_modulation_scaled < 0
            if positive_structure_mask.any():
                positive_structure_strength = final_cost_structure_for_modulation_scaled[positive_structure_mask]
                if structural_power_asymmetric_tanh_enabled:
                    boost_amp = np.tanh((positive_structure_strength + structural_power_offset_positive_structure) * structural_power_tanh_factor_positive_structure) * dynamic_structural_power_sensitivity_amp.loc[positive_structure_mask]
                else:
                    boost_amp = np.tanh(positive_structure_strength * default_structural_power_tanh_factor_amp) * dynamic_structural_power_sensitivity_amp.loc[positive_structure_mask]
                amplification_power.loc[positive_structure_mask] = amplification_power.loc[positive_structure_mask] * (1 + boost_amp)
            if negative_structure_mask.any():
                negative_structure_strength = final_cost_structure_for_modulation_scaled[negative_structure_mask].abs()
                if structural_power_asymmetric_tanh_enabled:
                    boost_damp = np.tanh((negative_structure_strength + structural_power_offset_negative_structure) * structural_power_tanh_factor_negative_structure) * dynamic_structural_power_sensitivity_damp.loc[negative_structure_mask]
                else:
                    boost_damp = np.tanh(negative_structure_strength * default_structural_power_tanh_factor_damp) * dynamic_structural_power_sensitivity_damp.loc[negative_structure_mask]
                dampening_power.loc[negative_structure_mask] = dampening_power.loc[negative_structure_mask] * (1 + boost_damp)
            amplification_power = amplification_power.clip(0.5, 3.0)
            dampening_power = dampening_power.clip(0.5, 3.0)
        bullish_mask = holder_sentiment_scores > dynamic_sentiment_neutrality_threshold
        bearish_mask = holder_sentiment_scores < -dynamic_sentiment_neutrality_threshold
        bullish_tailwind_mask = bullish_mask & (final_cost_structure_for_modulation_scaled > dynamic_cost_structure_neutrality_threshold)
        modulation_factor.loc[bullish_tailwind_mask] = (1 + final_cost_structure_for_modulation_scaled.loc[bullish_tailwind_mask]) ** amplification_power.loc[bullish_tailwind_mask]
        bullish_headwind_mask = bullish_mask & (final_cost_structure_for_modulation_scaled < -dynamic_cost_structure_neutrality_threshold)
        modulation_factor.loc[bullish_headwind_mask] = (1 - final_cost_structure_for_modulation_scaled.loc[bullish_headwind_mask].abs()) ** dampening_power.loc[bullish_headwind_mask]
        bearish_tailwind_mask = bearish_mask & (final_cost_structure_for_modulation_scaled < -dynamic_cost_structure_neutrality_threshold)
        modulation_factor.loc[bearish_tailwind_mask] = (1 + final_cost_structure_for_modulation_scaled.loc[bearish_tailwind_mask].abs()) ** amplification_power.loc[bearish_tailwind_mask]
        bearish_headwind_mask = bearish_mask & (final_cost_structure_for_modulation_scaled > dynamic_cost_structure_neutrality_threshold)
        modulation_factor.loc[bearish_headwind_mask] = (1 - final_cost_structure_for_modulation_scaled.loc[bearish_headwind_mask]) ** dampening_power.loc[bearish_headwind_mask]
        coherent_drive_raw = activated_holder_sentiment_scores * modulation_factor
        if final_score_sensitivity_modulation_enabled:
            final_score_modulator_signal_raw = self._get_safe_series(df, df, final_score_modulator_signal_name, 0.0, method_name="_diagnose_structural_consensus")
            final_score_normalized_modulator_signal = normalize_score(
                final_score_modulator_signal_raw,
                df.index,
                window=final_score_mod_norm_window,
                ascending=True
            )
            final_score_modulator_bipolar = (final_score_normalized_modulator_signal * 2) - 1
            final_score_non_linear_modulator_effect = np.tanh(final_score_modulator_bipolar * final_score_mod_tanh_factor)
            dynamic_final_score_sensitivity_multiplier = final_score_base_sensitivity_multiplier * (1 + final_score_non_linear_modulator_effect * final_score_mod_factor)
            dynamic_final_score_sensitivity_multiplier = dynamic_final_score_sensitivity_multiplier.clip(final_score_base_sensitivity_multiplier * 0.5, final_score_base_sensitivity_multiplier * 2.0)
        else:
            dynamic_final_score_sensitivity_multiplier = pd.Series(final_score_base_sensitivity_multiplier, index=df.index)
        final_score = np.tanh(coherent_drive_raw * (self.bipolar_sensitivity * dynamic_final_score_sensitivity_multiplier))
        print(f"    -> [筹码情报校验] 计算“筹码一致驱动” 分数：{final_score.mean():.4f}")
        return final_score.astype(np.float32)

    def _diagnose_absorption_echo(self, df: pd.DataFrame, divergence_scores: pd.Series) -> pd.Series:
        """
        【V5.1 · 诡道反吸强化版】吸筹回声探针
        - 核心升级1: 恐慌声源精细化。在V4.0基础上，引入总输家比例短期加速度、散户恐慌投降指数短期斜率、结构性紧张指数短期加速度，更精准捕捉恐慌蔓延。
        - 核心升级2: 逆流介质强化。在V4.0基础上，引入浮动筹码清洗效率短期斜率、订单簿清算率短期加速度、微观价格冲击不对称性短期斜率、VWAP控制强度短期斜率、VWAP穿越强度短期加速度，更全面评估承接能力。
        - 核心升级3: 主力回声深化。在V4.0基础上，引入隐蔽吸筹信号短期加速度、压制式吸筹强度短期斜率、主力成本优势短期加速度、主力资金流向方向性短期斜率、主力VPOC短期加速度、智能资金净买入短期斜率，更细致刻画主力吸筹意图。
        - 核心升级4: 诡道背景调制智能化。优化诡道调制逻辑，引入“诱空反吸”的增强机制，即当出现“诱空”式欺骗且主力信念坚定，则增强吸筹回声信号；同时细化对“诱多”式欺骗和对倒行为的惩罚。
        - 核心升级5: 情境调制器引入。引入资金流可信度指数、结构性紧张指数、筹码健康度作为最终分数的调制器，提供更丰富的宏观情境感知。
        - 核心升级6: 新增筹码指标整合：
            - 支持性派发强度 (`supportive_distribution_intensity_D`) 作为负向调制器。
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        print("    -> [筹码情报校验] 正在诊断“吸筹回声” (V5.1 · 诡道反吸强化版)...")
        df_index = df.index
        probe_date = self.strategy.debug_params.get('target_date') if self.strategy.debug_params.get('enable_mfca_probe') else None
        required_signals = [
            'retail_panic_surrender_index_D', 'loser_pain_index_D', 'chip_fatigue_index_D',
            'structural_tension_index_D', 'panic_selling_cascade_D', 'total_loser_rate_D',
            'loser_loss_margin_avg_D', 'SLOPE_5_loser_pain_index_D', 'ACCEL_5_chip_fatigue_index_D',
            'VOLATILITY_INSTABILITY_INDEX_21d_D', 'capitulation_absorption_index_D',
            'floating_chip_cleansing_efficiency_D', 'support_validation_strength_D',
            'main_force_execution_alpha_D', 'active_buying_support_D', 'opening_gap_defense_strength_D',
            'control_solidity_index_D', 'SLOPE_5_support_validation_strength_D',
            'ACCEL_5_main_force_execution_alpha_D', 'order_book_clearing_rate_D',
            'covert_accumulation_signal_D', 'suppressive_accumulation_intensity_D',
            'main_force_cost_advantage_D', 'peak_control_transfer_D', 'main_force_conviction_index_D',
            'main_force_net_flow_calibrated_D', 'main_force_flow_directionality_D', 'main_force_vpoc_D',
            'main_force_activity_ratio_D', 'SLOPE_5_covert_accumulation_signal_D',
            'ACCEL_5_main_force_conviction_index_D', 'SMART_MONEY_HM_NET_BUY_D',
            'chip_fault_magnitude_D', 'deception_index_D', 'wash_trade_intensity_D',
            'chip_health_score_D', 'main_force_conviction_index_D',
            'ACCEL_5_total_loser_rate_D', 'SLOPE_5_retail_panic_surrender_index_D', 'ACCEL_5_structural_tension_index_D',
            'SLOPE_5_floating_chip_cleansing_efficiency_D', 'ACCEL_5_order_book_clearing_rate_D',
            'SLOPE_5_micro_price_impact_asymmetry_D', 'SLOPE_5_vwap_control_strength_D', 'ACCEL_5_vwap_crossing_intensity_D',
            'ACCEL_5_covert_accumulation_signal_D', 'SLOPE_5_suppressive_accumulation_intensity_D',
            'ACCEL_5_main_force_cost_advantage_D', 'SLOPE_5_main_force_flow_directionality_D',
            'ACCEL_5_main_force_vpoc_D', 'SLOPE_5_SMART_MONEY_HM_NET_BUY_D',
            'flow_credibility_index_D',
            # [修改代码行] V5.1 新增筹码指标
            'supportive_distribution_intensity_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_absorption_echo"):
            return pd.Series(0.0, index=df.index)
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        absorption_echo_params = get_param_value(p_conf.get('absorption_echo_params'), {})
        # --- 参数加载 ---
        panic_source_weights = get_param_value(absorption_echo_params.get('panic_source_weights'), {})
        panic_context_threshold = get_param_value(absorption_echo_params.get('panic_context_threshold'), 0.3)
        counter_flow_medium_weights = get_param_value(absorption_echo_params.get('counter_flow_medium_weights'), {})
        main_force_echo_weights = get_param_value(absorption_echo_params.get('main_force_echo_weights'), {})
        deception_modulator_params = get_param_value(absorption_echo_params.get('deception_modulator_params'), {})
        final_fusion_exponent = get_param_value(absorption_echo_params.get('final_fusion_exponent'), 0.25)
        context_modulator_weights = get_param_value(absorption_echo_params.get('context_modulator_weights'), {})
        # [修改代码行] V5.1 新增支持性派发强度权重
        supportive_distribution_penalty_factor = get_param_value(absorption_echo_params.get('supportive_distribution_penalty_factor'), 0.2)
        df_index = df.index
        # --- 原始数据获取 ---
        retail_panic_surrender_raw = self._get_safe_series(df, df, 'retail_panic_surrender_index_D', 0.0, method_name="_diagnose_absorption_echo")
        loser_pain_raw = self._get_safe_series(df, df, 'loser_pain_index_D', 0.0, method_name="_diagnose_absorption_echo")
        chip_fatigue_raw = self._get_safe_series(df, df, 'chip_fatigue_index_D', 0.0, method_name="_diagnose_absorption_echo")
        structural_tension_raw = self._get_safe_series(df, df, 'structural_tension_index_D', 0.0, method_name="_diagnose_absorption_echo")
        panic_selling_cascade_raw = self._get_safe_series(df, df, 'panic_selling_cascade_D', 0.0, method_name="_diagnose_absorption_echo")
        total_loser_rate_raw = self._get_safe_series(df, df, 'total_loser_rate_D', 0.0, method_name="_diagnose_absorption_echo")
        loser_loss_margin_avg_raw = self._get_safe_series(df, df, 'loser_loss_margin_avg_D', 0.0, method_name="_diagnose_absorption_echo")
        slope_5_loser_pain_raw = self._get_safe_series(df, df, 'SLOPE_5_loser_pain_index_D', 0.0, method_name="_diagnose_absorption_echo")
        accel_5_chip_fatigue_raw = self._get_safe_series(df, df, 'ACCEL_5_chip_fatigue_index_D', 0.0, method_name="_diagnose_absorption_echo")
        volatility_instability_raw = self._get_safe_series(df, df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', 0.0, method_name="_diagnose_absorption_echo")
        accel_5_total_loser_rate_raw = self._get_safe_series(df, df, 'ACCEL_5_total_loser_rate_D', 0.0, method_name="_diagnose_absorption_echo")
        slope_5_retail_panic_surrender_raw = self._get_safe_series(df, df, 'SLOPE_5_retail_panic_surrender_index_D', 0.0, method_name="_diagnose_absorption_echo")
        accel_5_structural_tension_raw = self._get_safe_series(df, df, 'ACCEL_5_structural_tension_index_D', 0.0, method_name="_diagnose_absorption_echo")
        divergence_bullish_raw = divergence_scores
        capitulation_absorption_raw = self._get_safe_series(df, df, 'capitulation_absorption_index_D', 0.0, method_name="_diagnose_absorption_echo")
        cleansing_efficiency_raw = self._get_safe_series(df, df, 'floating_chip_cleansing_efficiency_D', 0.0, method_name="_diagnose_absorption_echo")
        support_validation_raw = self._get_safe_series(df, df, 'support_validation_strength_D', 0.0, method_name="_diagnose_absorption_echo")
        main_force_execution_alpha_raw = self._get_safe_series(df, df, 'main_force_execution_alpha_D', 0.0, method_name="_diagnose_absorption_echo")
        active_buying_support_raw = self._get_safe_series(df, df, 'active_buying_support_D', 0.0, method_name="_diagnose_absorption_echo")
        opening_gap_defense_strength_raw = self._get_safe_series(df, df, 'opening_gap_defense_strength_D', 0.0, method_name="_diagnose_absorption_echo")
        control_solidity_raw = self._get_safe_series(df, df, 'control_solidity_index_D', 0.0, method_name="_diagnose_absorption_echo")
        slope_5_support_validation_raw = self._get_safe_series(df, df, 'SLOPE_5_support_validation_strength_D', 0.0, method_name="_diagnose_absorption_echo")
        accel_5_main_force_execution_alpha_raw = self._get_safe_series(df, df, 'ACCEL_5_main_force_execution_alpha_D', 0.0, method_name="_diagnose_absorption_echo")
        order_book_clearing_rate_raw = self._get_safe_series(df, df, 'order_book_clearing_rate_D', 0.0, method_name="_diagnose_absorption_echo")
        slope_5_floating_chip_cleansing_raw = self._get_safe_series(df, df, 'SLOPE_5_floating_chip_cleansing_efficiency_D', 0.0, method_name="_diagnose_absorption_echo")
        accel_5_order_book_clearing_raw = self._get_safe_series(df, df, 'ACCEL_5_order_book_clearing_rate_D', 0.0, method_name="_diagnose_absorption_echo")
        micro_price_impact_asymmetry_raw = self._get_safe_series(df, df, 'micro_price_impact_asymmetry_D', 0.0, method_name="_diagnose_absorption_echo")
        slope_5_micro_price_impact_asymmetry_raw = self._get_safe_series(df, df, 'SLOPE_5_micro_price_impact_asymmetry_D', 0.0, method_name="_diagnose_absorption_echo")
        vwap_control_strength_raw = self._get_safe_series(df, df, 'vwap_control_strength_D', 0.0, method_name="_diagnose_absorption_echo")
        slope_5_vwap_control_strength_raw = self._get_safe_series(df, df, 'SLOPE_5_vwap_control_strength_D', 0.0, method_name="_diagnose_absorption_echo")
        vwap_crossing_intensity_raw = self._get_safe_series(df, df, 'vwap_crossing_intensity_D', 0.0, method_name="_diagnose_absorption_echo")
        accel_5_vwap_crossing_intensity_raw = self._get_safe_series(df, df, 'ACCEL_5_vwap_crossing_intensity_D', 0.0, method_name="_diagnose_absorption_echo")
        covert_accumulation_raw = self._get_safe_series(df, df, 'covert_accumulation_signal_D', 0.0, method_name="_diagnose_absorption_echo")
        suppressive_accumulation_raw = self._get_safe_series(df, df, 'suppressive_accumulation_intensity_D', 0.0, method_name="_diagnose_absorption_echo")
        main_force_cost_advantage_raw = self._get_safe_series(df, df, 'main_force_cost_advantage_D', 0.0, method_name="_diagnose_absorption_echo")
        peak_control_transfer_raw = self._get_safe_series(df, df, 'peak_control_transfer_D', 0.0, method_name="_diagnose_absorption_echo")
        main_force_conviction_raw = self._get_safe_series(df, df, 'main_force_conviction_index_D', 0.0, method_name="_diagnose_absorption_echo")
        main_force_net_flow_calibrated_raw = self._get_safe_series(df, df, 'main_force_net_flow_calibrated_D', 0.0, method_name="_diagnose_absorption_echo")
        main_force_flow_directionality_raw = self._get_safe_series(df, df, 'main_force_flow_directionality_D', 0.0, method_name="_diagnose_absorption_echo")
        main_force_vpoc_raw = self._get_safe_series(df, df, 'main_force_vpoc_D', 0.0, method_name="_diagnose_absorption_echo")
        main_force_activity_ratio_raw = self._get_safe_series(df, df, 'main_force_activity_ratio_D', 0.0, method_name="_diagnose_absorption_echo")
        slope_5_covert_accumulation_raw = self._get_safe_series(df, df, 'SLOPE_5_covert_accumulation_signal_D', 0.0, method_name="_diagnose_absorption_echo")
        accel_5_main_force_conviction_raw = self._get_safe_series(df, df, 'ACCEL_5_main_force_conviction_index_D', 0.0, method_name="_diagnose_absorption_echo")
        smart_money_net_buy_raw = self._get_safe_series(df, df, 'SMART_MONEY_HM_NET_BUY_D', 0.0, method_name="_diagnose_absorption_echo")
        accel_5_covert_accumulation_raw = self._get_safe_series(df, df, 'ACCEL_5_covert_accumulation_signal_D', 0.0, method_name="_diagnose_absorption_echo")
        slope_5_suppressive_accumulation_raw = self._get_safe_series(df, df, 'SLOPE_5_suppressive_accumulation_intensity_D', 0.0, method_name="_diagnose_absorption_echo")
        accel_5_main_force_cost_advantage_raw = self._get_safe_series(df, df, 'ACCEL_5_main_force_cost_advantage_D', 0.0, method_name="_diagnose_absorption_echo")
        slope_5_main_force_flow_directionality_raw = self._get_safe_series(df, df, 'SLOPE_5_main_force_flow_directionality_D', 0.0, method_name="_diagnose_absorption_echo")
        accel_5_main_force_vpoc_raw = self._get_safe_series(df, df, 'ACCEL_5_main_force_vpoc_D', 0.0, method_name="_diagnose_absorption_echo")
        slope_5_smart_money_net_buy_raw = self._get_safe_series(df, df, 'SLOPE_5_SMART_MONEY_HM_NET_BUY_D', 0.0, method_name="_diagnose_absorption_echo")
        chip_fault_magnitude_raw = self._get_safe_series(df, df, 'chip_fault_magnitude_D', 0.0, method_name="_diagnose_absorption_echo")
        deception_index_raw = self._get_safe_series(df, df, 'deception_index_D', 0.0, method_name="_diagnose_absorption_echo")
        wash_trade_intensity_raw = self._get_safe_series(df, df, 'wash_trade_intensity_D', 0.0, method_name="_diagnose_absorption_echo")
        chip_health_score_raw = self._get_safe_series(df, df, 'chip_health_score_D', 0.0, method_name="_diagnose_absorption_echo")
        flow_credibility_raw = self._get_safe_series(df, df, 'flow_credibility_index_D', 0.0, method_name="_diagnose_absorption_echo")
        # [修改代码行] V5.1 获取支持性派发强度
        supportive_distribution_intensity_raw = self._get_safe_series(df, df, 'supportive_distribution_intensity_D', 0.0, method_name="_diagnose_absorption_echo")
        if probe_date: print(f"        -> [探针] {probe_date} 原始信号: retail_panic_surrender_index_D={retail_panic_surrender_raw.asof(probe_date):.4f}, loser_pain_index_D={loser_pain_raw.asof(probe_date):.4f}, chip_fatigue_index_D={chip_fatigue_raw.asof(probe_date):.4f}, structural_tension_index_D={structural_tension_raw.asof(probe_date):.4f}, panic_selling_cascade_D={panic_selling_cascade_raw.asof(probe_date):.4f}, total_loser_rate_D={total_loser_rate_raw.asof(probe_date):.4f}, loser_loss_margin_avg_D={loser_loss_margin_avg_raw.asof(probe_date):.4f}, SLOPE_5_loser_pain_index_D={slope_5_loser_pain_raw.asof(probe_date):.4f}, ACCEL_5_chip_fatigue_index_D={accel_5_chip_fatigue_raw.asof(probe_date):.4f}, VOLATILITY_INSTABILITY_INDEX_21d_D={volatility_instability_raw.asof(probe_date):.4f}, ACCEL_5_total_loser_rate_D={accel_5_total_loser_rate_raw.asof(probe_date):.4f}, SLOPE_5_retail_panic_surrender_index_D={slope_5_retail_panic_surrender_raw.asof(probe_date):.4f}, ACCEL_5_structural_tension_index_D={accel_5_structural_tension_raw.asof(probe_date):.4f}, capitulation_absorption_index_D={capitulation_absorption_raw.asof(probe_date):.4f}, floating_chip_cleansing_efficiency_D={cleansing_efficiency_raw.asof(probe_date):.4f}, support_validation_strength_D={support_validation_raw.asof(probe_date):.4f}, main_force_execution_alpha_D={main_force_execution_alpha_raw.asof(probe_date):.4f}, active_buying_support_D={active_buying_support_raw.asof(probe_date):.4f}, opening_gap_defense_strength_D={opening_gap_defense_strength_raw.asof(probe_date):.4f}, control_solidity_index_D={control_solidity_raw.asof(probe_date):.4f}, SLOPE_5_support_validation_strength_D={slope_5_support_validation_raw.asof(probe_date):.4f}, ACCEL_5_main_force_execution_alpha_D={accel_5_main_force_execution_alpha_raw.asof(probe_date):.4f}, order_book_clearing_rate_D={order_book_clearing_rate_raw.asof(probe_date):.4f}, SLOPE_5_floating_chip_cleansing_efficiency_D={slope_5_floating_chip_cleansing_raw.asof(probe_date):.4f}, ACCEL_5_order_book_clearing_rate_D={accel_5_order_book_clearing_raw.asof(probe_date):.4f}, micro_price_impact_asymmetry_D={micro_price_impact_asymmetry_raw.asof(probe_date):.4f}, SLOPE_5_micro_price_impact_asymmetry_D={slope_5_micro_price_impact_asymmetry_raw.asof(probe_date):.4f}, vwap_control_strength_D={vwap_control_strength_raw.asof(probe_date):.4f}, SLOPE_5_vwap_control_strength_D={slope_5_vwap_control_strength_raw.asof(probe_date):.4f}, vwap_crossing_intensity_D={vwap_crossing_intensity_raw.asof(probe_date):.4f}, ACCEL_5_vwap_crossing_intensity_D={accel_5_vwap_crossing_intensity_raw.asof(probe_date):.4f}, covert_accumulation_signal_D={covert_accumulation_raw.asof(probe_date):.4f}, suppressive_accumulation_intensity_D={suppressive_accumulation_raw.asof(probe_date):.4f}, main_force_cost_advantage_D={main_force_cost_advantage_raw.asof(probe_date):.4f}, peak_control_transfer_D={peak_control_transfer_raw.asof(probe_date):.4f}, main_force_conviction_index_D={main_force_conviction_raw.asof(probe_date):.4f}, main_force_net_flow_calibrated_D={main_force_net_flow_calibrated_raw.asof(probe_date):.4f}, main_force_flow_directionality_D={main_force_flow_directionality_raw.asof(probe_date):.4f}, main_force_vpoc_D={main_force_vpoc_raw.asof(probe_date):.4f}, main_force_activity_ratio_D={main_force_activity_ratio_raw.asof(probe_date):.4f}, SLOPE_5_covert_accumulation_signal_D={slope_5_covert_accumulation_raw.asof(probe_date):.4f}, ACCEL_5_main_force_conviction_index_D={accel_5_main_force_conviction_raw.asof(probe_date):.4f}, SMART_MONEY_HM_NET_BUY_D={smart_money_net_buy_raw.asof(probe_date):.4f}, ACCEL_5_covert_accumulation_signal_D={accel_5_covert_accumulation_raw.asof(probe_date):.4f}, SLOPE_5_suppressive_accumulation_intensity_D={slope_5_suppressive_accumulation_raw.asof(probe_date):.4f}, ACCEL_5_main_force_cost_advantage_D={accel_5_main_force_cost_advantage_raw.asof(probe_date):.4f}, SLOPE_5_main_force_flow_directionality_D={slope_5_main_force_flow_directionality_raw.asof(probe_date):.4f}, ACCEL_5_main_force_vpoc_D={accel_5_main_force_vpoc_raw.asof(probe_date):.4f}, SLOPE_5_SMART_MONEY_HM_NET_BUY_D={slope_5_smart_money_net_buy_raw.asof(probe_date):.4f}, chip_fault_magnitude_D={chip_fault_magnitude_raw.asof(probe_date):.4f}, deception_index_D={deception_index_raw.asof(probe_date):.4f}, wash_trade_intensity_D={wash_trade_intensity_raw.asof(probe_date):.4f}, chip_health_score_D={chip_health_score_raw.asof(probe_date):.4f}, flow_credibility_index_D={flow_credibility_raw.asof(probe_date):.4f}, supportive_distribution_intensity_D={supportive_distribution_intensity_raw.asof(probe_date):.4f}")
        # --- 维度1: 恐慌声源 (Panic Source Score) ---
        norm_retail_panic_surrender = get_adaptive_mtf_normalized_score(retail_panic_surrender_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_loser_pain = get_adaptive_mtf_normalized_score(loser_pain_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_chip_fatigue = get_adaptive_mtf_normalized_score(chip_fatigue_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_structural_tension_negative = get_adaptive_mtf_normalized_score(structural_tension_raw, df_index, ascending=False, tf_weights=tf_weights)
        norm_panic_selling_cascade = get_adaptive_mtf_normalized_score(panic_selling_cascade_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_total_loser_rate = get_adaptive_mtf_normalized_score(total_loser_rate_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_loser_loss_margin_avg = get_adaptive_mtf_normalized_score(loser_loss_margin_avg_raw, df_index, ascending=False, tf_weights=tf_weights)
        norm_slope_5_loser_pain = get_adaptive_mtf_normalized_score(slope_5_loser_pain_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_accel_5_chip_fatigue = get_adaptive_mtf_normalized_score(accel_5_chip_fatigue_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_volatility_instability = get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_accel_5_total_loser_rate = get_adaptive_mtf_normalized_score(accel_5_total_loser_rate_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_slope_5_retail_panic_surrender = get_adaptive_mtf_normalized_score(slope_5_retail_panic_surrender_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_accel_5_structural_tension = get_adaptive_mtf_normalized_score(accel_5_structural_tension_raw, df_index, ascending=True, tf_weights=tf_weights)
        panic_source_numeric_weights = {k: v for k, v in panic_source_weights.items() if isinstance(v, (int, float))}
        total_panic_source_weight = sum(panic_source_numeric_weights.values())
        panic_source_score = (
            norm_retail_panic_surrender.pow(panic_source_numeric_weights.get('retail_panic_surrender', 0.15)) *
            norm_loser_pain.pow(panic_source_numeric_weights.get('loser_pain', 0.15)) *
            norm_chip_fatigue.pow(panic_source_numeric_weights.get('chip_fatigue', 0.1)) *
            norm_structural_tension_negative.pow(panic_source_numeric_weights.get('structural_tension_negative', 0.1)) *
            norm_panic_selling_cascade.pow(panic_source_numeric_weights.get('panic_selling_cascade', 0.1)) *
            norm_total_loser_rate.pow(panic_source_numeric_weights.get('total_loser_rate', 0.08)) *
            norm_loser_loss_margin_avg.pow(panic_source_numeric_weights.get('loser_loss_margin_avg', 0.08)) *
            norm_slope_5_loser_pain.pow(panic_source_numeric_weights.get('loser_pain_slope', 0.08)) *
            norm_accel_5_chip_fatigue.pow(panic_source_numeric_weights.get('chip_fatigue_accel', 0.08)) *
            norm_volatility_instability.pow(panic_source_numeric_weights.get('volatility_instability', 0.08)) *
            norm_accel_5_total_loser_rate.pow(panic_source_numeric_weights.get('total_loser_rate_accel', 0.05)) *
            norm_slope_5_retail_panic_surrender.pow(panic_source_numeric_weights.get('retail_panic_surrender_slope', 0.05)) *
            norm_accel_5_structural_tension.pow(panic_source_numeric_weights.get('structural_tension_accel', 0.05))
        ).pow(1 / total_panic_source_weight)
        if probe_date: print(f"        -> [探针] {probe_date} 恐慌声源分数: {panic_source_score.asof(probe_date):.4f}")
        is_panic_context = panic_source_score > panic_context_threshold
        # --- 维度2: 逆流介质 (Counter Flow Medium Score) ---
        norm_divergence_bullish = get_adaptive_mtf_normalized_score(divergence_bullish_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_capitulation_absorption = get_adaptive_mtf_normalized_score(capitulation_absorption_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_cleansing_efficiency = get_adaptive_mtf_normalized_score(cleansing_efficiency_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_support_validation = get_adaptive_mtf_normalized_score(support_validation_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_main_force_execution_alpha = get_adaptive_mtf_normalized_score(main_force_execution_alpha_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_active_buying_support = get_adaptive_mtf_normalized_score(active_buying_support_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_opening_gap_defense_strength = get_adaptive_mtf_normalized_score(opening_gap_defense_strength_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_control_solidity = get_adaptive_mtf_normalized_score(control_solidity_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_support_validation_slope = get_adaptive_mtf_normalized_score(slope_5_support_validation_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_main_force_execution_alpha_accel = get_adaptive_mtf_normalized_score(accel_5_main_force_execution_alpha_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_order_book_clearing_rate = get_adaptive_mtf_normalized_score(order_book_clearing_rate_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_slope_5_floating_chip_cleansing = get_adaptive_mtf_normalized_score(slope_5_floating_chip_cleansing_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_accel_5_order_book_clearing = get_adaptive_mtf_normalized_score(accel_5_order_book_clearing_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_micro_price_impact_asymmetry = get_adaptive_mtf_normalized_score(micro_price_impact_asymmetry_raw, df_index, ascending=False, tf_weights=tf_weights)
        norm_slope_5_micro_price_impact_asymmetry = get_adaptive_mtf_normalized_score(slope_5_micro_price_impact_asymmetry_raw, df_index, ascending=False, tf_weights=tf_weights)
        norm_vwap_control_strength = get_adaptive_mtf_normalized_score(vwap_control_strength_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_slope_5_vwap_control_strength = get_adaptive_mtf_normalized_score(slope_5_vwap_control_strength_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_vwap_crossing_intensity = get_adaptive_mtf_normalized_score(vwap_crossing_intensity_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_accel_5_vwap_crossing_intensity = get_adaptive_mtf_normalized_score(accel_5_vwap_crossing_intensity_raw, df_index, ascending=True, tf_weights=tf_weights)
        counter_flow_medium_numeric_weights = {k: v for k, v in counter_flow_medium_weights.items() if isinstance(v, (int, float))}
        total_counter_flow_medium_weight = sum(counter_flow_medium_numeric_weights.values())
        counter_flow_medium_score = (
            norm_divergence_bullish.pow(counter_flow_medium_numeric_weights.get('divergence_bullish', 0.15)) *
            norm_capitulation_absorption.pow(counter_flow_medium_numeric_weights.get('capitulation_absorption', 0.1)) *
            norm_cleansing_efficiency.pow(counter_flow_medium_numeric_weights.get('cleansing_efficiency', 0.1)) *
            norm_support_validation.pow(counter_flow_medium_numeric_weights.get('support_validation', 0.1)) *
            norm_main_force_execution_alpha.pow(counter_flow_medium_numeric_weights.get('main_force_execution_alpha', 0.1)) *
            norm_active_buying_support.pow(counter_flow_medium_numeric_weights.get('active_buying_support', 0.1)) *
            norm_opening_gap_defense_strength.pow(counter_flow_medium_numeric_weights.get('opening_gap_defense_strength', 0.08)) *
            norm_control_solidity.pow(counter_flow_medium_numeric_weights.get('control_solidity', 0.08)) *
            norm_support_validation_slope.pow(counter_flow_medium_numeric_weights.get('support_validation_slope', 0.08)) *
            norm_main_force_execution_alpha_accel.pow(counter_flow_medium_numeric_weights.get('main_force_execution_alpha_accel', 0.05)) *
            norm_order_book_clearing_rate.pow(counter_flow_medium_numeric_weights.get('order_book_clearing_rate', 0.06)) *
            norm_slope_5_floating_chip_cleansing.pow(counter_flow_medium_numeric_weights.get('cleansing_efficiency_slope', 0.05)) *
            norm_accel_5_order_book_clearing.pow(counter_flow_medium_numeric_weights.get('order_book_clearing_accel', 0.05)) *
            norm_micro_price_impact_asymmetry.pow(counter_flow_medium_numeric_weights.get('micro_impact_asymmetry', 0.05)) *
            norm_slope_5_micro_price_impact_asymmetry.pow(counter_flow_medium_numeric_weights.get('micro_impact_asymmetry_slope', 0.03)) *
            norm_vwap_control_strength.pow(counter_flow_medium_numeric_weights.get('vwap_control_strength', 0.05)) *
            norm_slope_5_vwap_control_strength.pow(counter_flow_medium_numeric_weights.get('vwap_control_strength_slope', 0.03)) *
            norm_vwap_crossing_intensity.pow(counter_flow_medium_numeric_weights.get('vwap_crossing_intensity', 0.03)) *
            norm_accel_5_vwap_crossing_intensity.pow(counter_flow_medium_numeric_weights.get('vwap_crossing_intensity_accel', 0.02))
        ).pow(1 / total_counter_flow_medium_weight)
        if probe_date: print(f"        -> [探针] {probe_date} 逆流介质分数: {counter_flow_medium_score.asof(probe_date):.4f}")
        # --- 维度3: 主力回声 (Main Force Echo Score) ---
        norm_covert_accumulation = get_adaptive_mtf_normalized_score(covert_accumulation_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_suppressive_accumulation = get_adaptive_mtf_normalized_score(suppressive_accumulation_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_main_force_cost_advantage = get_adaptive_mtf_normalized_score(main_force_cost_advantage_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_peak_control_transfer = get_adaptive_mtf_normalized_score(peak_control_transfer_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_main_force_conviction_positive = get_adaptive_mtf_normalized_score(main_force_conviction_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_main_force_net_flow_positive = get_adaptive_mtf_normalized_score(main_force_net_flow_calibrated_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_main_force_flow_directionality_positive = get_adaptive_mtf_normalized_score(main_force_flow_directionality_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_main_force_vpoc = get_adaptive_mtf_normalized_score(main_force_vpoc_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_main_force_activity_ratio = get_adaptive_mtf_normalized_score(main_force_activity_ratio_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_slope_5_covert_accumulation = get_adaptive_mtf_normalized_score(slope_5_covert_accumulation_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_accel_5_main_force_conviction = get_adaptive_mtf_normalized_score(accel_5_main_force_conviction_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_smart_money_net_buy_positive = get_adaptive_mtf_normalized_score(smart_money_net_buy_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_accel_5_covert_accumulation = get_adaptive_mtf_normalized_score(accel_5_covert_accumulation_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_slope_5_suppressive_accumulation = get_adaptive_mtf_normalized_score(slope_5_suppressive_accumulation_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_accel_5_main_force_cost_advantage = get_adaptive_mtf_normalized_score(accel_5_main_force_cost_advantage_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_slope_5_main_force_flow_directionality = get_adaptive_mtf_normalized_score(slope_5_main_force_flow_directionality_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_accel_5_main_force_vpoc = get_adaptive_mtf_normalized_score(accel_5_main_force_vpoc_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_slope_5_smart_money_net_buy = get_adaptive_mtf_normalized_score(slope_5_smart_money_net_buy_raw, df_index, ascending=True, tf_weights=tf_weights)
        main_force_echo_numeric_weights = {k: v for k, v in main_force_echo_weights.items() if isinstance(v, (int, float))}
        total_main_force_echo_weight = sum(main_force_echo_numeric_weights.values())
        main_force_echo_score = (
            norm_covert_accumulation.pow(main_force_echo_numeric_weights.get('covert_accumulation', 0.1)) *
            norm_suppressive_accumulation.pow(main_force_echo_numeric_weights.get('suppressive_accumulation', 0.08)) *
            norm_main_force_cost_advantage.pow(main_force_echo_numeric_weights.get('cost_advantage', 0.08)) *
            norm_peak_control_transfer.pow(main_force_echo_numeric_weights.get('peak_control_transfer', 0.08)) *
            norm_main_force_conviction_positive.pow(main_force_echo_numeric_weights.get('main_force_conviction_positive', 0.08)) *
            norm_main_force_net_flow_positive.pow(main_force_echo_numeric_weights.get('main_force_net_flow_positive', 0.1)) *
            norm_main_force_flow_directionality_positive.pow(main_force_echo_numeric_weights.get('main_force_flow_directionality_positive', 0.08)) *
            norm_main_force_vpoc.pow(main_force_echo_numeric_weights.get('main_force_vpoc', 0.08)) *
            norm_main_force_activity_ratio.pow(main_force_echo_numeric_weights.get('main_force_activity_ratio', 0.08)) *
            norm_slope_5_covert_accumulation.pow(main_force_echo_numeric_weights.get('covert_accumulation_slope', 0.08)) *
            norm_accel_5_main_force_conviction.pow(main_force_echo_numeric_weights.get('main_force_conviction_accel', 0.08)) *
            norm_smart_money_net_buy_positive.pow(main_force_echo_numeric_weights.get('smart_money_net_buy_positive', 0.08)) *
            norm_accel_5_covert_accumulation.pow(main_force_echo_numeric_weights.get('covert_accumulation_accel', 0.05)) *
            norm_slope_5_suppressive_accumulation.pow(main_force_echo_numeric_weights.get('suppressive_accumulation_slope', 0.05)) *
            norm_accel_5_main_force_cost_advantage.pow(main_force_echo_numeric_weights.get('cost_advantage_accel', 0.05)) *
            norm_slope_5_main_force_flow_directionality.pow(main_force_echo_numeric_weights.get('flow_directionality_slope', 0.05)) *
            norm_accel_5_main_force_vpoc.pow(main_force_echo_numeric_weights.get('main_force_vpoc_accel', 0.05)) *
            norm_slope_5_smart_money_net_buy.pow(main_force_echo_numeric_weights.get('smart_money_net_buy_slope', 0.05))
        ).pow(1 / total_main_force_echo_weight)
        if probe_date: print(f"        -> [探针] {probe_date} 主力回声分数: {main_force_echo_score.asof(probe_date):.4f}")
        # --- 诡道背景调制 (Deception Modulator) ---
        deception_modulator = pd.Series(1.0, index=df_index)
        norm_chip_fault_magnitude_bipolar = get_adaptive_mtf_normalized_bipolar_score(chip_fault_magnitude_raw, df_index, tf_weights)
        norm_deception_index_bipolar = get_adaptive_mtf_normalized_bipolar_score(deception_index_raw, df_index, tf_weights)
        norm_wash_trade_intensity = get_adaptive_mtf_normalized_score(wash_trade_intensity_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_main_force_conviction_bipolar = get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, tf_weights)
        # [修改代码行] V5.1 归一化支持性派发强度
        norm_supportive_distribution_intensity = get_adaptive_mtf_normalized_score(supportive_distribution_intensity_raw, df_index, ascending=True, tf_weights=tf_weights)
        conviction_threshold = deception_modulator_params.get('conviction_threshold', 0.2)
        deception_index_boost_weight = deception_modulator_params.get('deception_index_boost_weight', 0.5)
        deception_index_penalty_weight = deception_modulator_params.get('deception_index_penalty_weight', 0.5)
        wash_trade_penalty_weight = deception_modulator_params.get('wash_trade_penalty_weight', 0.3)
        deception_boost_mask = (norm_deception_index_bipolar < 0) & (norm_main_force_conviction_bipolar > conviction_threshold)
        deception_modulator.loc[deception_boost_mask] = deception_modulator.loc[deception_boost_mask] * (1 + norm_deception_index_bipolar.loc[deception_boost_mask].abs() * deception_index_boost_weight)
        deception_penalty_mask = (norm_deception_index_bipolar > 0) & (norm_main_force_conviction_bipolar < -conviction_threshold)
        deception_modulator.loc[deception_penalty_mask] = deception_modulator.loc[deception_penalty_mask] * (1 - norm_deception_index_bipolar.loc[deception_penalty_mask] * deception_index_penalty_weight)
        deception_modulator = deception_modulator * (1 - norm_wash_trade_intensity * wash_trade_penalty_weight)
        deception_modulator = deception_modulator * (1 - norm_chip_fault_magnitude_bipolar.clip(lower=0) * deception_modulator_params.get('penalty_factor', 0.4))
        # [修改代码行] V5.1 整合支持性派发强度作为负向调制
        deception_modulator = deception_modulator * (1 - norm_supportive_distribution_intensity * supportive_distribution_penalty_factor)
        deception_modulator = deception_modulator.clip(0.1, 2.0)
        if probe_date: print(f"        -> [探针] {probe_date} 诡道背景调制器: {deception_modulator.asof(probe_date):.4f}")
        # --- 情境调制器 (Contextual Modulators) ---
        norm_flow_credibility = get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_structural_tension = get_adaptive_mtf_normalized_score(structural_tension_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_chip_health_score = get_adaptive_mtf_normalized_score(chip_health_score_raw, df_index, ascending=True, tf_weights=tf_weights)
        context_modulator_numeric_weights = {k: v for k, v in context_modulator_weights.items() if isinstance(v, (int, float))}
        total_context_modulator_weight = sum(context_modulator_numeric_weights.values())
        context_modulator = pd.Series(1.0, index=df.index)
        if total_context_modulator_weight > 0:
            fused_context_modulator_raw = (
                norm_flow_credibility.pow(context_modulator_numeric_weights.get('flow_credibility', 0.4)) *
                (1 - norm_structural_tension).pow(context_modulator_numeric_weights.get('structural_tension_inverse', 0.3)) *
                norm_chip_health_score.pow(context_modulator_numeric_weights.get('chip_health', 0.3))
            ).pow(1 / total_context_modulator_weight)
            context_modulator = 1 + (fused_context_modulator_raw - 0.5) * 0.5
        context_modulator = context_modulator.clip(0.5, 1.5)
        if probe_date: print(f"        -> [探针] {probe_date} 情境调制器: {context_modulator.asof(probe_date):.4f}")
        # --- 最终融合 ---
        base_score = pd.Series(0.0, index=df.index)
        valid_mask = is_panic_context
        if valid_mask.any():
            base_score.loc[valid_mask] = (
                counter_flow_medium_score.loc[valid_mask].pow(0.5) *
                main_force_echo_score.loc[valid_mask].pow(0.5)
            )
        final_score = base_score * deception_modulator * context_modulator
        final_score = final_score.pow(final_fusion_exponent)
        if probe_date: print(f"        -> [探针] {probe_date} 最终吸筹回声分数: {final_score.asof(probe_date):.4f}")
        return final_score.clip(0.0, 1.0).fillna(0.0).astype(np.float32)

    def _diagnose_distribution_whisper(self, df: pd.DataFrame, divergence_score: pd.Series) -> pd.Series:
        """
        【V4.0 · 深度高频诡道派发版】诊断“派发诡影”信号
        - 核心升级1: 狂热背景深度化。在V3.0基础上，引入总赢家比例、赢家输家动量及其短期斜率，更全面刻画市场狂热和筹码结构膨胀。
        - 核心升级2: 背离诡影精细化。在V3.0基础上，引入主峰利润率、主峰坚实度、上影线抛压、压力拒绝强度及其短期斜率，评估主力派发动力、筹码结构松动和承接力减弱。
        - 核心升级3: 主力抽离多维度验证。在V3.0基础上，引入主力净流量校准、主力滑点指数及其短期加速度、反弹派发压力、控制坚实度、对手盘枯竭和智能资金净买入负向，多角度验证主力隐蔽、坚决派发。
        - 核心升级4: 诡道背景调制强化。引入欺骗指数，结合筹码故障幅度与主力信念指数，更智能地判断诡道意图并进行调制。
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        required_signals = [
            'retail_fomo_premium_index_D', 'winner_profit_margin_avg_D', 'THEME_HOTNESS_SCORE_D', 'market_sentiment_score_D', 'winner_concentration_90pct_D',
            'total_winner_rate_D', 'winner_loser_momentum_D', 'SLOPE_5_winner_loser_momentum_D', # V4.0 新增
            'dispersal_by_distribution_D', 'profit_taking_flow_ratio_D', 'chip_fault_magnitude_D',
            'cost_structure_skewness_D', 'winner_stability_index_D', 'chip_fault_blockage_ratio_D',
            'dominant_peak_profit_margin_D', 'dominant_peak_solidity_D', 'upper_shadow_selling_pressure_D', 'pressure_rejection_strength_D', # V4.0 新增
            'SLOPE_5_dominant_peak_solidity_D', 'SLOPE_5_pressure_rejection_strength_D', # V4.0 新增
            'covert_accumulation_signal_D', 'wash_trade_intensity_D', 'main_force_conviction_index_D', 'retail_flow_dominance_index_D',
            'main_force_net_flow_calibrated_D', 'main_force_slippage_index_D', 'rally_distribution_pressure_D', 'control_solidity_index_D', # V4.0 新增
            'counterparty_exhaustion_index_D', 'SMART_MONEY_HM_NET_BUY_D', # V4.0 新增
            'SLOPE_5_main_force_net_flow_calibrated_D', 'ACCEL_5_main_force_slippage_index_D', # V4.0 新增
            'deception_index_D' # V4.0 新增
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_distribution_whisper"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        distribution_whisper_params = get_param_value(p_conf.get('distribution_whisper_params'), {})
        fomo_backdrop_weights = get_param_value(distribution_whisper_params.get('fomo_backdrop_weights'), {'retail_fomo_premium': 0.2, 'winner_profit_margin': 0.2, 'theme_hotness': 0.15, 'market_sentiment_positive': 0.1, 'winner_concentration_negative': 0.1, 'total_winner_rate': 0.15, 'winner_loser_momentum': 0.05, 'winner_loser_momentum_slope': 0.05}) # V4.0 权重调整
        fomo_context_threshold = get_param_value(distribution_whisper_params.get('fomo_context_threshold'), 0.3)
        divergence_shadow_weights = get_param_value(distribution_whisper_params.get('divergence_shadow_weights'), {'divergence_bearish': 0.15, 'distribution_intensity': 0.15, 'chip_fault_magnitude': 0.1, 'cost_structure_negative': 0.1, 'winner_stability_negative': 0.1, 'chip_fault_blockage': 0.1, 'dominant_peak_profit_margin': 0.1, 'dominant_peak_solidity_negative': 0.05, 'upper_shadow_selling_pressure': 0.05, 'pressure_rejection_strength_negative': 0.05, 'dominant_peak_solidity_slope_negative': 0.025, 'pressure_rejection_strength_slope_negative': 0.025}) # V4.0 权重调整
        main_force_retreat_weights = get_param_value(distribution_whisper_params.get('main_force_retreat_weights'), {'profit_taking_flow': 0.15, 'dispersal_by_distribution': 0.15, 'covert_accumulation_negative': 0.1, 'wash_trade_intensity': 0.1, 'main_force_conviction_negative': 0.1, 'retail_flow_dominance': 0.1, 'main_force_net_flow_negative': 0.1, 'main_force_slippage': 0.05, 'rally_distribution_pressure': 0.05, 'control_solidity_negative': 0.05, 'counterparty_exhaustion': 0.025, 'smart_money_net_buy_negative': 0.025, 'main_force_net_flow_slope_negative': 0.025, 'main_force_slippage_accel': 0.025}) # V4.0 权重调整
        deception_modulator_params = get_param_value(distribution_whisper_params.get('deception_modulator_params'), {'boost_factor': 0.6, 'penalty_factor': 0.4, 'conviction_threshold': 0.2, 'deception_index_weight': 0.5}) # V4.0 权重调整
        final_fusion_exponent = get_param_value(distribution_whisper_params.get('final_fusion_exponent'), 0.25)
        # --- 维度1: 狂热背景 (FOMO Backdrop - Pure Chip FOMO) ---
        retail_fomo_premium_raw = self._get_safe_series(df, df, 'retail_fomo_premium_index_D', 0.0, method_name="_diagnose_distribution_whisper")
        winner_profit_margin_raw = self._get_safe_series(df, df, 'winner_profit_margin_avg_D', 0.0, method_name="_diagnose_distribution_whisper")
        theme_hotness_raw = self._get_safe_series(df, df, 'THEME_HOTNESS_SCORE_D', 0.0, method_name="_diagnose_distribution_whisper")
        market_sentiment_raw = self._get_safe_series(df, df, 'market_sentiment_score_D', 0.0, method_name="_diagnose_distribution_whisper")
        winner_concentration_raw = self._get_safe_series(df, df, 'winner_concentration_90pct_D', 0.0, method_name="_diagnose_distribution_whisper")
        total_winner_rate_raw = self._get_safe_series(df, df, 'total_winner_rate_D', 0.0, method_name="_diagnose_distribution_whisper") # V4.0 新增
        winner_loser_momentum_raw = self._get_safe_series(df, df, 'winner_loser_momentum_D', 0.0, method_name="_diagnose_distribution_whisper") # V4.0 新增
        slope_5_winner_loser_momentum_raw = self._get_safe_series(df, df, 'SLOPE_5_winner_loser_momentum_D', 0.0, method_name="_diagnose_distribution_whisper") # V4.0 新增
        norm_retail_fomo_premium = get_adaptive_mtf_normalized_score(retail_fomo_premium_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_winner_profit_margin = get_adaptive_mtf_normalized_score(winner_profit_margin_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_theme_hotness = get_adaptive_mtf_normalized_score(theme_hotness_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_market_sentiment_positive = get_adaptive_mtf_normalized_bipolar_score(market_sentiment_raw, df_index, tf_weights).clip(0, 1)
        norm_winner_concentration_negative = get_adaptive_mtf_normalized_score(winner_concentration_raw, df_index, ascending=False, tf_weights=tf_weights)
        norm_total_winner_rate = get_adaptive_mtf_normalized_score(total_winner_rate_raw, df_index, ascending=True, tf_weights=tf_weights) # V4.0 新增
        norm_winner_loser_momentum = get_adaptive_mtf_normalized_score(winner_loser_momentum_raw, df_index, ascending=True, tf_weights=tf_weights) # V4.0 新增
        norm_slope_5_winner_loser_momentum = get_adaptive_mtf_normalized_score(slope_5_winner_loser_momentum_raw, df_index, ascending=True, tf_weights=tf_weights) # V4.0 新增
        fomo_backdrop_numeric_weights = {k: v for k, v in fomo_backdrop_weights.items() if isinstance(v, (int, float))}
        total_fomo_backdrop_weight = sum(fomo_backdrop_numeric_weights.values())
        fomo_backdrop_score = (
            norm_retail_fomo_premium.pow(fomo_backdrop_numeric_weights.get('retail_fomo_premium', 0.2)) *
            norm_winner_profit_margin.pow(fomo_backdrop_numeric_weights.get('winner_profit_margin', 0.2)) *
            norm_theme_hotness.pow(fomo_backdrop_numeric_weights.get('theme_hotness', 0.15)) *
            norm_market_sentiment_positive.pow(fomo_backdrop_numeric_weights.get('market_sentiment_positive', 0.1)) *
            norm_winner_concentration_negative.pow(fomo_backdrop_numeric_weights.get('winner_concentration_negative', 0.1)) *
            norm_total_winner_rate.pow(fomo_backdrop_numeric_weights.get('total_winner_rate', 0.15)) * # V4.0 新增
            norm_winner_loser_momentum.pow(fomo_backdrop_numeric_weights.get('winner_loser_momentum', 0.05)) * # V4.0 新增
            norm_slope_5_winner_loser_momentum.pow(fomo_backdrop_numeric_weights.get('winner_loser_momentum_slope', 0.05)) # V4.0 新增
        ).pow(1 / total_fomo_backdrop_weight)
        is_fomo_context = fomo_backdrop_score > fomo_context_threshold
        # --- 维度2: 背离诡影 (Divergence Shadow - Chip-centric Distribution Evidence) ---
        norm_divergence_bearish = divergence_score.clip(-1, 0).abs()
        dispersal_by_distribution_raw = self._get_safe_series(df, df, 'dispersal_by_distribution_D', 0.0, method_name="_diagnose_distribution_whisper")
        chip_fault_magnitude_raw = self._get_safe_series(df, df, 'chip_fault_magnitude_D', 0.0, method_name="_diagnose_distribution_whisper")
        cost_structure_skewness_raw = self._get_safe_series(df, df, 'cost_structure_skewness_D', 0.0, method_name="_diagnose_distribution_whisper")
        winner_stability_raw = self._get_safe_series(df, df, 'winner_stability_index_D', 0.0, method_name="_diagnose_distribution_whisper")
        chip_fault_blockage_raw = self._get_safe_series(df, df, 'chip_fault_blockage_ratio_D', 0.0, method_name="_diagnose_distribution_whisper")
        dominant_peak_profit_margin_raw = self._get_safe_series(df, df, 'dominant_peak_profit_margin_D', 0.0, method_name="_diagnose_distribution_whisper") # V4.0 新增
        dominant_peak_solidity_raw = self._get_safe_series(df, df, 'dominant_peak_solidity_D', 0.0, method_name="_diagnose_distribution_whisper") # V4.0 新增
        upper_shadow_selling_pressure_raw = self._get_safe_series(df, df, 'upper_shadow_selling_pressure_D', 0.0, method_name="_diagnose_distribution_whisper") # V4.0 新增
        pressure_rejection_strength_raw = self._get_safe_series(df, df, 'pressure_rejection_strength_D', 0.0, method_name="_diagnose_distribution_whisper") # V4.0 新增
        slope_5_dominant_peak_solidity_raw = self._get_safe_series(df, df, 'SLOPE_5_dominant_peak_solidity_D', 0.0, method_name="_diagnose_distribution_whisper") # V4.0 新增
        slope_5_pressure_rejection_strength_raw = self._get_safe_series(df, df, 'SLOPE_5_pressure_rejection_strength_D', 0.0, method_name="_diagnose_distribution_whisper") # V4.0 新增
        norm_dispersal_by_distribution = get_adaptive_mtf_normalized_score(dispersal_by_distribution_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_chip_fault_magnitude_for_shadow = get_adaptive_mtf_normalized_score(chip_fault_magnitude_raw.abs(), df_index, ascending=True, tf_weights=tf_weights)
        norm_cost_structure_negative = get_adaptive_mtf_normalized_score(cost_structure_skewness_raw, df_index, ascending=False, tf_weights=tf_weights)
        norm_winner_stability_negative = get_adaptive_mtf_normalized_score(winner_stability_raw, df_index, ascending=False, tf_weights=tf_weights)
        norm_chip_fault_blockage = get_adaptive_mtf_normalized_score(chip_fault_blockage_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_dominant_peak_profit_margin = get_adaptive_mtf_normalized_score(dominant_peak_profit_margin_raw, df_index, ascending=True, tf_weights=tf_weights) # V4.0 新增
        norm_dominant_peak_solidity_negative = get_adaptive_mtf_normalized_score(dominant_peak_solidity_raw, df_index, ascending=False, tf_weights=tf_weights) # V4.0 新增
        norm_upper_shadow_selling_pressure = get_adaptive_mtf_normalized_score(upper_shadow_selling_pressure_raw, df_index, ascending=True, tf_weights=tf_weights) # V4.0 新增
        norm_pressure_rejection_strength_negative = get_adaptive_mtf_normalized_score(pressure_rejection_strength_raw, df_index, ascending=False, tf_weights=tf_weights) # V4.0 新增
        norm_slope_5_dominant_peak_solidity_negative = get_adaptive_mtf_normalized_score(slope_5_dominant_peak_solidity_raw, df_index, ascending=False, tf_weights=tf_weights) # V4.0 新增
        norm_slope_5_pressure_rejection_strength_negative = get_adaptive_mtf_normalized_score(slope_5_pressure_rejection_strength_raw, df_index, ascending=False, tf_weights=tf_weights) # V4.0 新增
        divergence_shadow_numeric_weights = {k: v for k, v in divergence_shadow_weights.items() if isinstance(v, (int, float))}
        total_divergence_shadow_weight = sum(divergence_shadow_numeric_weights.values())
        divergence_shadow_score = (
            norm_divergence_bearish.pow(divergence_shadow_numeric_weights.get('divergence_bearish', 0.15)) *
            norm_dispersal_by_distribution.pow(divergence_shadow_numeric_weights.get('distribution_intensity', 0.15)) *
            norm_chip_fault_magnitude_for_shadow.pow(divergence_shadow_numeric_weights.get('chip_fault_magnitude', 0.1)) *
            norm_cost_structure_negative.pow(divergence_shadow_numeric_weights.get('cost_structure_negative', 0.1)) *
            norm_winner_stability_negative.pow(divergence_shadow_numeric_weights.get('winner_stability_negative', 0.1)) *
            norm_chip_fault_blockage.pow(divergence_shadow_numeric_weights.get('chip_fault_blockage', 0.1)) *
            norm_dominant_peak_profit_margin.pow(divergence_shadow_numeric_weights.get('dominant_peak_profit_margin', 0.1)) * # V4.0 新增
            norm_dominant_peak_solidity_negative.pow(divergence_shadow_numeric_weights.get('dominant_peak_solidity_negative', 0.05)) * # V4.0 新增
            norm_upper_shadow_selling_pressure.pow(divergence_shadow_numeric_weights.get('upper_shadow_selling_pressure', 0.05)) * # V4.0 新增
            norm_pressure_rejection_strength_negative.pow(divergence_shadow_numeric_weights.get('pressure_rejection_strength_negative', 0.05)) * # V4.0 新增
            norm_slope_5_dominant_peak_solidity_negative.pow(divergence_shadow_numeric_weights.get('dominant_peak_solidity_slope_negative', 0.025)) * # V4.0 新增
            norm_slope_5_pressure_rejection_strength_negative.pow(divergence_shadow_numeric_weights.get('pressure_rejection_strength_slope_negative', 0.025)) # V4.0 新增
        ).pow(1 / total_divergence_shadow_weight)
        # --- 维度3: 主力抽离 (Main Force Retreat - Pure Chip Distribution Evidence) ---
        profit_taking_flow_ratio_raw = self._get_safe_series(df, df, 'profit_taking_flow_ratio_D', 0.0, method_name="_diagnose_distribution_whisper")
        # dispersal_by_distribution_raw 已经获取过
        covert_accumulation_raw = self._get_safe_series(df, df, 'covert_accumulation_signal_D', 0.0, method_name="_diagnose_distribution_whisper")
        wash_trade_intensity_raw = self._get_safe_series(df, df, 'wash_trade_intensity_D', 0.0, method_name="_diagnose_distribution_whisper")
        main_force_conviction_raw = self._get_safe_series(df, df, 'main_force_conviction_index_D', 0.0, method_name="_diagnose_distribution_whisper")
        retail_flow_dominance_raw = self._get_safe_series(df, df, 'retail_flow_dominance_index_D', 0.0, method_name="_diagnose_distribution_whisper")
        main_force_net_flow_calibrated_raw = self._get_safe_series(df, df, 'main_force_net_flow_calibrated_D', 0.0, method_name="_diagnose_distribution_whisper") # V4.0 新增
        main_force_slippage_raw = self._get_safe_series(df, df, 'main_force_slippage_index_D', 0.0, method_name="_diagnose_distribution_whisper") # V4.0 新增
        rally_distribution_pressure_raw = self._get_safe_series(df, df, 'rally_distribution_pressure_D', 0.0, method_name="_diagnose_distribution_whisper") # V4.0 新增
        control_solidity_raw = self._get_safe_series(df, df, 'control_solidity_index_D', 0.0, method_name="_diagnose_distribution_whisper") # V4.0 新增
        counterparty_exhaustion_raw = self._get_safe_series(df, df, 'counterparty_exhaustion_index_D', 0.0, method_name="_diagnose_distribution_whisper") # V4.0 新增
        smart_money_net_buy_raw = self._get_safe_series(df, df, 'SMART_MONEY_HM_NET_BUY_D', 0.0, method_name="_diagnose_distribution_whisper") # V4.0 新增
        slope_5_main_force_net_flow_calibrated_raw = self._get_safe_series(df, df, 'SLOPE_5_main_force_net_flow_calibrated_D', 0.0, method_name="_diagnose_distribution_whisper") # V4.0 新增
        accel_5_main_force_slippage_raw = self._get_safe_series(df, df, 'ACCEL_5_main_force_slippage_index_D', 0.0, method_name="_diagnose_distribution_whisper") # V4.0 新增
        norm_profit_taking_flow_ratio = get_adaptive_mtf_normalized_score(profit_taking_flow_ratio_raw, df_index, ascending=True, tf_weights=tf_weights)
        # norm_dispersal_by_distribution 已经获取过
        norm_covert_accumulation_negative = get_adaptive_mtf_normalized_score(covert_accumulation_raw, df_index, ascending=False, tf_weights=tf_weights)
        norm_wash_trade_intensity = get_adaptive_mtf_normalized_score(wash_trade_intensity_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_main_force_conviction_negative = get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, tf_weights).clip(-1, 0).abs()
        norm_retail_flow_dominance = get_adaptive_mtf_normalized_score(retail_flow_dominance_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_main_force_net_flow_negative = get_adaptive_mtf_normalized_score(main_force_net_flow_calibrated_raw, df_index, ascending=False, tf_weights=tf_weights) # V4.0 新增
        norm_main_force_slippage = get_adaptive_mtf_normalized_score(main_force_slippage_raw, df_index, ascending=True, tf_weights=tf_weights) # V4.0 新增
        norm_rally_distribution_pressure = get_adaptive_mtf_normalized_score(rally_distribution_pressure_raw, df_index, ascending=True, tf_weights=tf_weights) # V4.0 新增
        norm_control_solidity_negative = get_adaptive_mtf_normalized_score(control_solidity_raw, df_index, ascending=False, tf_weights=tf_weights) # V4.0 新增
        norm_counterparty_exhaustion = get_adaptive_mtf_normalized_score(counterparty_exhaustion_raw, df_index, ascending=True, tf_weights=tf_weights) # V4.0 新增
        norm_smart_money_net_buy_negative = get_adaptive_mtf_normalized_score(smart_money_net_buy_raw, df_index, ascending=False, tf_weights=tf_weights) # V4.0 新增
        norm_slope_5_main_force_net_flow_negative = get_adaptive_mtf_normalized_score(slope_5_main_force_net_flow_calibrated_raw, df_index, ascending=False, tf_weights=tf_weights) # V4.0 新增
        norm_accel_5_main_force_slippage = get_adaptive_mtf_normalized_score(accel_5_main_force_slippage_raw, df_index, ascending=True, tf_weights=tf_weights) # V4.0 新增
        main_force_retreat_numeric_weights = {k: v for k, v in main_force_retreat_weights.items() if isinstance(v, (int, float))}
        total_main_force_retreat_weight = sum(main_force_retreat_numeric_weights.values())
        main_force_retreat_score = (
            norm_profit_taking_flow_ratio.pow(main_force_retreat_numeric_weights.get('profit_taking_flow', 0.15)) *
            norm_dispersal_by_distribution.pow(main_force_retreat_numeric_weights.get('dispersal_by_distribution', 0.15)) *
            norm_covert_accumulation_negative.pow(main_force_retreat_numeric_weights.get('covert_accumulation_negative', 0.1)) *
            norm_wash_trade_intensity.pow(main_force_retreat_numeric_weights.get('wash_trade_intensity', 0.1)) *
            norm_main_force_conviction_negative.pow(main_force_retreat_numeric_weights.get('main_force_conviction_negative', 0.1)) *
            norm_retail_flow_dominance.pow(main_force_retreat_numeric_weights.get('retail_flow_dominance', 0.1)) *
            norm_main_force_net_flow_negative.pow(main_force_retreat_numeric_weights.get('main_force_net_flow_negative', 0.1)) * # V4.0 新增
            norm_main_force_slippage.pow(main_force_retreat_numeric_weights.get('main_force_slippage', 0.05)) * # V4.0 新增
            norm_rally_distribution_pressure.pow(main_force_retreat_numeric_weights.get('rally_distribution_pressure', 0.05)) * # V4.0 新增
            norm_control_solidity_negative.pow(main_force_retreat_numeric_weights.get('control_solidity_negative', 0.05)) * # V4.0 新增
            norm_counterparty_exhaustion.pow(main_force_retreat_numeric_weights.get('counterparty_exhaustion', 0.025)) * # V4.0 新增
            norm_smart_money_net_buy_negative.pow(main_force_retreat_numeric_weights.get('smart_money_net_buy_negative', 0.025)) * # V4.0 新增
            norm_slope_5_main_force_net_flow_negative.pow(main_force_retreat_numeric_weights.get('main_force_net_flow_slope_negative', 0.025)) * # V4.0 新增
            norm_accel_5_main_force_slippage.pow(main_force_retreat_numeric_weights.get('main_force_slippage_accel', 0.025)) # V4.0 新增
        ).pow(1 / total_main_force_retreat_weight)
        # --- 维度4: 诡道背景调制 (Deception Context Modulation) ---
        # chip_fault_magnitude_raw 已经获取过
        norm_chip_fault_magnitude_bipolar = get_adaptive_mtf_normalized_bipolar_score(chip_fault_magnitude_raw, df_index, tf_weights)
        norm_main_force_conviction_bipolar = get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, tf_weights)
        deception_index_raw = self._get_safe_series(df, df, 'deception_index_D', 0.0, method_name="_diagnose_distribution_whisper") # V4.0 新增
        norm_deception_index_bipolar = get_adaptive_mtf_normalized_bipolar_score(deception_index_raw, df_index, tf_weights) # V4.0 新增
        deception_modulator = pd.Series(1.0, index=df_index)
        # 诱多派发 (正向筹码故障 + 主力信念动摇) -> 增强信号
        deceptive_bullish_and_weak_conviction_mask = (norm_chip_fault_magnitude_bipolar > 0) & \
                                                     (norm_main_force_conviction_bipolar < -deception_modulator_params.get('conviction_threshold', 0.2))
        deception_modulator.loc[deceptive_bullish_and_weak_conviction_mask] = 1 + norm_chip_fault_magnitude_bipolar.loc[deceptive_bullish_and_weak_conviction_mask] * deception_modulator_params.get('boost_factor', 0.6)
        # 诱空吸筹 (负向筹码故障 + 主力信念坚定) -> 惩罚信号 (与派发逻辑相悖)
        induced_panic_and_conviction_mask = (norm_chip_fault_magnitude_bipolar < 0) & \
                                            (norm_main_force_conviction_bipolar > deception_modulator_params.get('conviction_threshold', 0.2))
        deception_modulator.loc[induced_panic_and_conviction_mask] = 1 - norm_chip_fault_magnitude_bipolar.loc[induced_panic_and_conviction_mask].abs() * deception_modulator_params.get('penalty_factor', 0.4)
        # V4.0 新增：结合欺骗指数进行调制
        # 如果欺骗指数为正（诱多），且主力信念动摇，则进一步增强派发信号
        deception_index_boost_mask = (norm_deception_index_bipolar > 0) & \
                                     (norm_main_force_conviction_bipolar < -deception_modulator_params.get('conviction_threshold', 0.2))
        deception_modulator.loc[deception_index_boost_mask] = deception_modulator.loc[deception_index_boost_mask] + \
                                                              norm_deception_index_bipolar.loc[deception_index_boost_mask] * deception_modulator_params.get('deception_index_weight', 0.5)
        # 如果欺骗指数为负（诱空），且主力信念坚定，则削弱派发信号
        deception_index_penalty_mask = (norm_deception_index_bipolar < 0) & \
                                       (norm_main_force_conviction_bipolar > deception_modulator_params.get('conviction_threshold', 0.2))
        deception_modulator.loc[deception_index_penalty_mask] = deception_modulator.loc[deception_index_penalty_mask] - \
                                                                norm_deception_index_bipolar.loc[deception_index_penalty_mask].abs() * deception_modulator_params.get('deception_index_weight', 0.5)
        deception_modulator = deception_modulator.clip(0.1, 2.0)
        # --- 最终融合 ---
        base_score = (
            fomo_backdrop_score.pow(final_fusion_exponent) *
            divergence_shadow_score.pow(final_fusion_exponent) *
            main_force_retreat_score.pow(final_fusion_exponent)
        ).pow(1 / (3 * final_fusion_exponent))
        final_score = (base_score * deception_modulator) * is_fomo_context
        return final_score.clip(0, 1).fillna(0.0).astype(np.float32)

    def _diagnose_axiom_historical_potential(self, df: pd.DataFrame) -> pd.Series:
        """
        【V5.0 · 势能博弈临界版】筹码公理六：诊断“筹码势能”
        - 核心升级1: 主力吸筹质量 (MF_AQ)。引入“吸筹效率的非对称性”，结合主力成本优势和筹码健康度动态调整吸筹模式权重，并考虑主力执行效率和非对称摩擦指数等高频聚合信号。
        - 核心升级2: 筹码结构张力 (CST)。引入“结构临界点识别”，结合赢家/输家集中度斜率预判结构转折，并考虑结构张力指数和结构熵变。
        - 核心升级3: 势能转化效率 (PCE)。引入“阻力位博弈强度”，评估关键阻力位和支撑位的博弈激烈程度，并考虑订单簿清算率和微观价格冲击不对称性等微观层面的阻力消化能力。
        - 核心升级4: 诡道博弈调制 (DGM)。引入“诡道博弈的非对称影响”，对诱多/诱空施加不同敏感度的调制，并考虑散户恐慌和主力信念对诡道博弈有效性的影响。
        - 核心升级5: 情境自适应权重 (ACW)。引入“市场情绪与流动性情境”，增加市场情绪分数和资金流可信度指数作为情境调制器。
        """
        required_signals = [
            'covert_accumulation_signal_D', 'suppressive_accumulation_intensity_D',
            'main_force_cost_advantage_D', 'floating_chip_cleansing_efficiency_D',
            'chip_health_score_D', 'dominant_peak_solidity_D',
            'SLOPE_5_cost_structure_skewness_D', 'SLOPE_5_peak_separation_ratio_D',
            'vacuum_zone_magnitude_D', 'vacuum_traversal_efficiency_D',
            'VOLATILITY_INSTABILITY_INDEX_21d_D', 'chip_fatigue_index_D',
            'chip_fault_magnitude_D',
            'winner_stability_index_D', 'loser_pain_index_D',
            'active_selling_pressure_D', 'capitulation_absorption_index_D',
            'deception_index_D', 'wash_trade_intensity_D', 'main_force_flow_directionality_D',
            'main_force_execution_alpha_D', 'asymmetric_friction_index_D',
            'SLOPE_5_winner_concentration_90pct_D', 'SLOPE_5_loser_concentration_90pct_D',
            'structural_tension_index_D', 'structural_entropy_change_D',
            'pressure_rejection_strength_D', 'support_validation_strength_D',
            'order_book_clearing_rate_D', 'micro_price_impact_asymmetry_D',
            'retail_panic_surrender_index_D', 'main_force_conviction_index_D',
            'market_sentiment_score_D', 'flow_credibility_index_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_historical_potential"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        historical_potential_params = get_param_value(p_conf.get('historical_potential_params'), {})
        mf_aq_weights = get_param_value(historical_potential_params.get('mf_aq_weights'), {
            'covert_accumulation': 0.25, 'suppressive_accumulation': 0.15,
            'cost_advantage': 0.25, 'cleansing_efficiency': 0.15, 'deception_purity_factor': 0.1,
            'execution_alpha': 0.05, 'friction_index': 0.05
        })
        mf_aq_asymmetry_params = get_param_value(historical_potential_params.get('mf_aq_asymmetry_params'), {
            'cost_advantage_threshold': 0.0, 'chip_health_threshold': 0.0,
            'covert_weight_boost': 0.2, 'suppressive_weight_boost': 0.1
        })
        cst_weights = get_param_value(historical_potential_params.get('cst_weights'), {
            'chip_health': 0.2, 'peak_solidity': 0.2,
            'cost_skewness_slope': 0.1, 'peak_separation_slope': 0.1, 'structural_elasticity': 0.15,
            'concentration_slope_divergence': 0.15, 'structural_tension': 0.05, 'structural_entropy': 0.05
        })
        pce_weights = get_param_value(historical_potential_params.get('pce_weights'), {
            'vacuum_magnitude': 0.3, 'vacuum_efficiency': 0.3, 'resistance_absorption': 0.2,
            'resistance_game_strength_weight': 0.2,
            'order_book_clearing_rate': 0.05, 'micro_price_impact_asymmetry': 0.05
        })
        dgm_weights = get_param_value(historical_potential_params.get('dgm_weights'), {
            'deception_impact': 0.4, 'wash_trade_penalty': 0.2, 'flow_directionality_boost': 0.1,
            'retail_panic_impact': 0.15, 'main_force_conviction_impact': 0.15
        })
        dgm_asymmetry_params = get_param_value(historical_potential_params.get('dgm_asymmetry_params'), {
            'bull_trap_penalty_factor': 1.5, 'bear_trap_bonus_factor': 1.2
        })
        final_fusion_weights = get_param_value(historical_potential_params.get('final_fusion_weights'), {
            'mf_aq': 0.35, 'cst': 0.3, 'pce': 0.35
        })
        context_modulator_signals = get_param_value(historical_potential_params.get('context_modulator_signals'), {
            'volatility_instability': {'signal_name': 'VOLATILITY_INSTABILITY_INDEX_21d_D', 'weight': 0.3, 'ascending': False},
            'chip_fatigue': {'signal_name': 'chip_fatigue_index_D', 'weight': 0.2, 'ascending': False},
            'market_sentiment': {'signal_name': 'market_sentiment_score_D', 'weight': 0.3, 'ascending': True},
            'flow_credibility': {'signal_name': 'flow_credibility_index_D', 'weight': 0.2, 'ascending': True}
        })
        context_modulator_sensitivity = get_param_value(historical_potential_params.get('context_modulator_sensitivity'), 0.5)
        dgm_modulator_sensitivity = get_param_value(historical_potential_params.get('dgm_modulator_sensitivity'), 0.8)
        chip_health_raw = self._get_safe_series(df, df, 'chip_health_score_D', 0.0, method_name="_diagnose_axiom_historical_potential")
        norm_chip_health = get_adaptive_mtf_normalized_bipolar_score(chip_health_raw, df_index, tf_weights)
        # --- A. 主力吸筹质量 (Main Force Accumulation Quality - MF_AQ) ---
        covert_accumulation_raw = self._get_safe_series(df, df, 'covert_accumulation_signal_D', 0.0, method_name="_diagnose_axiom_historical_potential")
        suppressive_accumulation_raw = self._get_safe_series(df, df, 'suppressive_accumulation_intensity_D', 0.0, method_name="_diagnose_axiom_historical_potential")
        main_force_cost_advantage_raw = self._get_safe_series(df, df, 'main_force_cost_advantage_D', 0.0, method_name="_diagnose_axiom_historical_potential")
        floating_chip_cleansing_efficiency_raw = self._get_safe_series(df, df, 'floating_chip_cleansing_efficiency_D', 0.0, method_name="_diagnose_axiom_historical_potential")
        chip_fault_magnitude_raw = self._get_safe_series(df, df, 'chip_fault_magnitude_D', 0.0, method_name="_diagnose_axiom_historical_potential")
        main_force_execution_alpha_raw = self._get_safe_series(df, df, 'main_force_execution_alpha_D', 0.0, method_name="_diagnose_axiom_historical_potential")
        asymmetric_friction_index_raw = self._get_safe_series(df, df, 'asymmetric_friction_index_D', 0.0, method_name="_diagnose_axiom_historical_potential")
        norm_covert_accumulation = get_adaptive_mtf_normalized_score(covert_accumulation_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_suppressive_accumulation = get_adaptive_mtf_normalized_score(suppressive_accumulation_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_main_force_cost_advantage = get_adaptive_mtf_normalized_bipolar_score(main_force_cost_advantage_raw, df_index, tf_weights)
        norm_floating_chip_cleansing_efficiency = get_adaptive_mtf_normalized_score(floating_chip_cleansing_efficiency_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_chip_fault_magnitude = get_adaptive_mtf_normalized_bipolar_score(chip_fault_magnitude_raw, df_index, tf_weights)
        norm_main_force_execution_alpha = get_adaptive_mtf_normalized_score(main_force_execution_alpha_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_asymmetric_friction_index = get_adaptive_mtf_normalized_score(asymmetric_friction_index_raw, df_index, ascending=False, tf_weights=tf_weights)
        deception_purity_adjustment = pd.Series(1.0, index=df_index)
        deception_purity_adjustment = 1 + (norm_chip_fault_magnitude * -1) * mf_aq_weights.get('deception_purity_factor', 0.1)
        deception_purity_adjustment = deception_purity_adjustment.clip(0.5, 1.5)
        dynamic_covert_weight = pd.Series(mf_aq_weights.get('covert_accumulation', 0.25), index=df_index)
        dynamic_suppressive_weight = pd.Series(mf_aq_weights.get('suppressive_accumulation', 0.15), index=df_index)
        low_health_low_cost_advantage_mask = (norm_chip_health < mf_aq_asymmetry_params.get('chip_health_threshold', 0.0)) & \
                                             (norm_main_force_cost_advantage < mf_aq_asymmetry_params.get('cost_advantage_threshold', 0.0))
        dynamic_covert_weight.loc[low_health_low_cost_advantage_mask] += mf_aq_asymmetry_params.get('covert_weight_boost', 0.2)
        dynamic_suppressive_weight.loc[low_health_low_cost_advantage_mask] -= mf_aq_asymmetry_params.get('suppressive_weight_boost', 0.1)
        base_mf_aq_total_weight = mf_aq_weights.get('covert_accumulation', 0.25) + mf_aq_weights.get('suppressive_accumulation', 0.15) + \
                                  mf_aq_weights.get('cost_advantage', 0.25) + mf_aq_weights.get('cleansing_efficiency', 0.15) + \
                                  mf_aq_weights.get('execution_alpha', 0.05) + mf_aq_weights.get('friction_index', 0.05)
        sum_dynamic_weights_mf_aq = dynamic_covert_weight + dynamic_suppressive_weight + \
                                    mf_aq_weights.get('cost_advantage', 0.25) + mf_aq_weights.get('cleansing_efficiency', 0.15) + \
                                    mf_aq_weights.get('execution_alpha', 0.05) + mf_aq_weights.get('friction_index', 0.05)
        mf_aq_score = (
            (norm_covert_accumulation * dynamic_covert_weight) +
            (norm_suppressive_accumulation * dynamic_suppressive_weight) +
            ((norm_main_force_cost_advantage.add(1)/2) * mf_aq_weights.get('cost_advantage', 0.25)) +
            (norm_floating_chip_cleansing_efficiency * mf_aq_weights.get('cleansing_efficiency', 0.15)) +
            (norm_main_force_execution_alpha * mf_aq_weights.get('execution_alpha', 0.05)) +
            (norm_asymmetric_friction_index * mf_aq_weights.get('friction_index', 0.05))
        ) / sum_dynamic_weights_mf_aq.replace(0, 1e-6) * base_mf_aq_total_weight
        mf_aq_score = mf_aq_score.clip(0, 1)
        # --- B. 筹码结构张力 (Chip Structure Tension - CST) ---
        dominant_peak_solidity_raw = self._get_safe_series(df, df, 'dominant_peak_solidity_D', 0.0, method_name="_diagnose_axiom_historical_potential")
        cost_structure_skewness_slope_raw = self._get_safe_series(df, df, 'SLOPE_5_cost_structure_skewness_D', 0.0, method_name="_diagnose_axiom_historical_potential")
        peak_separation_ratio_slope_raw = self._get_safe_series(df, df, 'SLOPE_5_peak_separation_ratio_D', 0.0, method_name="_diagnose_axiom_historical_potential")
        winner_stability_raw = self._get_safe_series(df, df, 'winner_stability_index_D', 0.0, method_name="_diagnose_axiom_historical_potential")
        loser_pain_raw = self._get_safe_series(df, df, 'loser_pain_index_D', 0.0, method_name="_diagnose_axiom_historical_potential")
        winner_concentration_slope_raw = self._get_safe_series(df, df, 'SLOPE_5_winner_concentration_90pct_D', 0.0, method_name="_diagnose_axiom_historical_potential")
        loser_concentration_slope_raw = self._get_safe_series(df, df, 'SLOPE_5_loser_concentration_90pct_D', 0.0, method_name="_diagnose_axiom_historical_potential")
        structural_tension_raw = self._get_safe_series(df, df, 'structural_tension_index_D', 0.0, method_name="_diagnose_axiom_historical_potential")
        structural_entropy_change_raw = self._get_safe_series(df, df, 'structural_entropy_change_D', 0.0, method_name="_diagnose_axiom_historical_potential")
        norm_dominant_peak_solidity = get_adaptive_mtf_normalized_score(dominant_peak_solidity_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_cost_structure_skewness_slope = get_adaptive_mtf_normalized_bipolar_score(cost_structure_skewness_slope_raw, df_index, tf_weights)
        norm_peak_separation_ratio_slope = get_adaptive_mtf_normalized_bipolar_score(peak_separation_ratio_slope_raw, df_index, tf_weights)
        norm_winner_stability = get_adaptive_mtf_normalized_score(winner_stability_raw, df_index, ascending=False, tf_weights=tf_weights)
        norm_loser_pain = get_adaptive_mtf_normalized_score(loser_pain_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_winner_concentration_slope = get_adaptive_mtf_normalized_bipolar_score(winner_concentration_slope_raw, df_index, tf_weights)
        norm_loser_concentration_slope = get_adaptive_mtf_normalized_bipolar_score(loser_concentration_slope_raw, df_index, tf_weights)
        norm_structural_tension = get_adaptive_mtf_normalized_score(structural_tension_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_structural_entropy_change = get_adaptive_mtf_normalized_score(structural_entropy_change_raw, df_index, ascending=False, tf_weights=tf_weights)
        structural_elasticity_score = (norm_winner_stability * 0.5 + norm_loser_pain * 0.5).clip(0, 1)
        concentration_slope_divergence = pd.Series(0.0, index=df_index)
        concentration_slope_divergence = (norm_winner_concentration_slope - norm_loser_concentration_slope).clip(-1, 1)
        cst_score = (
            (norm_chip_health.add(1)/2) * cst_weights.get('chip_health', 0.2) +
            norm_dominant_peak_solidity * cst_weights.get('peak_solidity', 0.2) +
            (1 - (norm_cost_structure_skewness_slope.add(1)/2)) * cst_weights.get('cost_skewness_slope', 0.1) +
            (1 - (norm_peak_separation_ratio_slope.add(1)/2)) * cst_weights.get('peak_separation_slope', 0.1) +
            structural_elasticity_score * cst_weights.get('structural_elasticity', 0.15) +
            (concentration_slope_divergence.add(1)/2) * cst_weights.get('concentration_slope_divergence', 0.15) +
            norm_structural_tension * cst_weights.get('structural_tension', 0.05) +
            norm_structural_entropy_change * cst_weights.get('structural_entropy', 0.05)
        ).clip(0, 1)
        # --- C. 势能转化效率 (Potential Conversion Efficiency - PCE) ---
        vacuum_zone_magnitude_raw = self._get_safe_series(df, df, 'vacuum_zone_magnitude_D', 0.0, method_name="_diagnose_axiom_historical_potential")
        vacuum_traversal_efficiency_raw = self._get_safe_series(df, df, 'vacuum_traversal_efficiency_D', 0.0, method_name="_diagnose_axiom_historical_potential")
        active_selling_pressure_raw = self._get_safe_series(df, df, 'active_selling_pressure_D', 0.0, method_name="_diagnose_axiom_historical_potential")
        capitulation_absorption_raw = self._get_safe_series(df, df, 'capitulation_absorption_index_D', 0.0, method_name="_diagnose_axiom_historical_potential")
        pressure_rejection_strength_raw = self._get_safe_series(df, df, 'pressure_rejection_strength_D', 0.0, method_name="_diagnose_axiom_historical_potential")
        support_validation_strength_raw = self._get_safe_series(df, df, 'support_validation_strength_D', 0.0, method_name="_diagnose_axiom_historical_potential")
        order_book_clearing_rate_raw = self._get_safe_series(df, df, 'order_book_clearing_rate_D', 0.0, method_name="_diagnose_axiom_historical_potential")
        micro_price_impact_asymmetry_raw = self._get_safe_series(df, df, 'micro_price_impact_asymmetry_D', 0.0, method_name="_diagnose_axiom_historical_potential")
        norm_vacuum_zone_magnitude = get_adaptive_mtf_normalized_score(vacuum_zone_magnitude_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_vacuum_traversal_efficiency = get_adaptive_mtf_normalized_score(vacuum_traversal_efficiency_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_active_selling_pressure = get_adaptive_mtf_normalized_score(active_selling_pressure_raw, df_index, ascending=False, tf_weights=tf_weights)
        norm_capitulation_absorption = get_adaptive_mtf_normalized_score(capitulation_absorption_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_pressure_rejection_strength = get_adaptive_mtf_normalized_score(pressure_rejection_strength_raw, df_index, ascending=False, tf_weights=tf_weights)
        norm_support_validation_strength = get_adaptive_mtf_normalized_score(support_validation_strength_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_order_book_clearing_rate = get_adaptive_mtf_normalized_score(order_book_clearing_rate_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_micro_price_impact_asymmetry = get_adaptive_mtf_normalized_score(micro_price_impact_asymmetry_raw.abs(), df_index, ascending=False, tf_weights=tf_weights)
        resistance_absorption_score = (norm_active_selling_pressure * 0.5 + norm_capitulation_absorption * 0.5).clip(0, 1)
        resistance_game_strength = (norm_pressure_rejection_strength * 0.5 + norm_support_validation_strength * 0.5).clip(0, 1)
        pce_score = (
            norm_vacuum_zone_magnitude * pce_weights.get('vacuum_magnitude', 0.3) +
            norm_vacuum_traversal_efficiency * pce_weights.get('vacuum_efficiency', 0.3) +
            resistance_absorption_score * pce_weights.get('resistance_absorption', 0.2) +
            resistance_game_strength * pce_weights.get('resistance_game_strength_weight', 0.2) +
            norm_order_book_clearing_rate * pce_weights.get('order_book_clearing_rate', 0.05) +
            norm_micro_price_impact_asymmetry * pce_weights.get('micro_price_impact_asymmetry', 0.05)
        ).clip(0, 1)
        # --- D. 诡道博弈调制 (Deceptive Game Modulator - DGM) ---
        deception_index_raw = self._get_safe_series(df, df, 'deception_index_D', 0.0, method_name="_diagnose_axiom_historical_potential")
        wash_trade_intensity_raw = self._get_safe_series(df, df, 'wash_trade_intensity_D', 0.0, method_name="_diagnose_axiom_historical_potential")
        main_force_flow_directionality_raw = self._get_safe_series(df, df, 'main_force_flow_directionality_D', 0.0, method_name="_diagnose_axiom_historical_potential")
        retail_panic_surrender_raw = self._get_safe_series(df, df, 'retail_panic_surrender_index_D', 0.0, method_name="_diagnose_axiom_historical_potential")
        main_force_conviction_raw = self._get_safe_series(df, df, 'main_force_conviction_index_D', 0.0, method_name="_diagnose_axiom_historical_potential")
        norm_deception_index = get_adaptive_mtf_normalized_bipolar_score(deception_index_raw, df_index, tf_weights)
        norm_wash_trade_intensity = get_adaptive_mtf_normalized_score(wash_trade_intensity_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_main_force_flow_directionality = get_adaptive_mtf_normalized_bipolar_score(main_force_flow_directionality_raw, df_index, tf_weights)
        norm_retail_panic_surrender = get_adaptive_mtf_normalized_score(retail_panic_surrender_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_main_force_conviction = get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, tf_weights)
        dgm_score = pd.Series(0.0, index=df_index)
        bull_trap_mask = (norm_deception_index > 0) & (norm_main_force_flow_directionality < 0)
        dgm_score.loc[bull_trap_mask] -= (norm_deception_index.loc[bull_trap_mask] * norm_main_force_flow_directionality.loc[bull_trap_mask].abs()) * dgm_weights.get('deception_impact', 0.4) * dgm_asymmetry_params.get('bull_trap_penalty_factor', 1.5)
        bear_trap_absorption_mask = (norm_deception_index < 0) & (norm_main_force_flow_directionality > 0)
        dgm_score.loc[bear_trap_absorption_mask] += (norm_deception_index.loc[bear_trap_absorption_mask].abs() * norm_main_force_flow_directionality.loc[bear_trap_absorption_mask]) * dgm_weights.get('deception_impact', 0.4) * dgm_asymmetry_params.get('bear_trap_bonus_factor', 1.2)
        dgm_score -= norm_wash_trade_intensity * dgm_weights.get('wash_trade_penalty', 0.2)
        positive_flow_boost_mask = (norm_main_force_flow_directionality > 0) & (~bull_trap_mask)
        dgm_score.loc[positive_flow_boost_mask] += norm_main_force_flow_directionality.loc[positive_flow_boost_mask] * dgm_weights.get('flow_directionality_boost', 0.1)
        dgm_score += norm_retail_panic_surrender * dgm_weights.get('retail_panic_impact', 0.15)
        dgm_score += (norm_main_force_conviction.abs()) * dgm_weights.get('main_force_conviction_impact', 0.15)
        dgm_score = dgm_score.clip(-1, 1)
        # --- E. 情境自适应权重 (Adaptive Contextual Weights - ACW) ---
        context_modulator_components = []
        total_context_weight = 0.0
        for ctx_key, ctx_config in context_modulator_signals.items():
            signal_name = ctx_config.get('signal_name')
            weight = ctx_config.get('weight', 0.0)
            ascending = ctx_config.get('ascending', True)
            if signal_name and weight > 0:
                raw_signal = self._get_safe_series(df, df, signal_name, 0.0, method_name="_diagnose_axiom_historical_potential")
                norm_signal = get_adaptive_mtf_normalized_score(raw_signal, df_index, ascending=ascending, tf_weights=tf_weights)
                context_modulator_components.append(norm_signal * weight)
                total_context_weight += weight
        if context_modulator_components and total_context_weight > 0:
            combined_context_modulator = sum(context_modulator_components) / total_context_weight
        else:
            combined_context_modulator = pd.Series(0.5, index=df_index)
        dynamic_final_fusion_weights = {
            'mf_aq': final_fusion_weights.get('mf_aq', 0.35) * (1 + combined_context_modulator * context_modulator_sensitivity),
            'cst': final_fusion_weights.get('cst', 0.3) * (1 + combined_context_modulator * context_modulator_sensitivity),
            'pce': final_fusion_weights.get('pce', 0.35) * (1 + combined_context_modulator * context_modulator_sensitivity)
        }
        sum_dynamic_weights = sum(dynamic_final_fusion_weights.values())
        normalized_dynamic_weights = {k: v / sum_dynamic_weights for k, v in dynamic_final_fusion_weights.items()}
        # --- 最终融合 ---
        base_potential_score = (
            mf_aq_score * normalized_dynamic_weights.get('mf_aq', 0.35) +
            cst_score * normalized_dynamic_weights.get('cst', 0.3) +
            pce_score * normalized_dynamic_weights.get('pce', 0.35)
        ).clip(0, 1)
        dgm_multiplier = 1 + dgm_score * dgm_modulator_sensitivity
        dgm_multiplier = dgm_multiplier.clip(0.1, 2.0)
        final_potential_score = (base_potential_score * dgm_multiplier).clip(0, 1)
        return final_potential_score.clip(0, 1).fillna(0.0).astype(np.float32)

    def _diagnose_tactical_exchange(self, df: pd.DataFrame, battlefield_geography: pd.Series) -> pd.Series:
        """
        【V6.0 · 筹码脉动版】诊断战术换手博弈的质量与意图
        - 核心升级1: 筹码“微观结构”与“订单流执行效率”评估。引入意图执行质量，作为意图维度的一个重要组成部分。
        - 核心升级2: 筹码“多峰结构”与“共振/冲突”分析。引入筹码峰动态，作为质量维度的一个新组成部分。
        - 核心升级3: 筹码“情绪”与“行为模式”识别。引入筹码行为模式强度，作为意图或质量维度的调制器。
        - 核心升级4: 非线性融合的“自学习”与“情境权重矩阵”。升级元调制器，使其能够更精细地调整融合权重。
        """
        required_signals = [
            'peak_control_transfer_D', 'floating_chip_cleansing_efficiency_D',
            'suppressive_accumulation_intensity_D', 'gathering_by_chasing_D', 'gathering_by_support_D',
            'chip_fault_magnitude_D', 'main_force_conviction_index_D',
            'retail_panic_surrender_index_D', 'loser_pain_index_D', 'winner_profit_margin_avg_D',
            'peak_exchange_purity_D',
            'SLOPE_5_winner_concentration_90pct_D', 'SLOPE_5_cost_structure_skewness_D', 'SLOPE_5_peak_separation_ratio_D',
            'winner_loser_momentum_D', 'chip_health_score_D',
            'capitulation_absorption_index_D', 'upward_impulse_purity_D', 'profit_realization_quality_D',
            'chip_fatigue_index_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D',
            'dominant_peak_solidity_D', 'SLOPE_5_dominant_peak_solidity_D',
            'total_loser_rate_D', 'total_winner_rate_D',
            'SLOPE_5_total_loser_rate_D', 'SLOPE_5_total_winner_rate_D',
            'volume_D',
            'winner_stability_index_D',
            'active_buying_support_D', 'active_selling_pressure_D', 'micro_price_impact_asymmetry_D',
            'order_book_clearing_rate_D', 'flow_credibility_index_D',
            'secondary_peak_cost_D', 'dominant_peak_volume_ratio_D',
            'main_force_activity_ratio_D', 'main_force_flow_directionality_D',
            'SLOPE_5_main_force_activity_ratio_D', 'SLOPE_5_main_force_flow_directionality_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_tactical_exchange"):
            return pd.Series(0.0, index=df.index)
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        tactical_exchange_params = get_param_value(p_conf.get('tactical_exchange_params'), {})
        intent_weights = get_param_value(tactical_exchange_params.get('intent_weights'), {'control_transfer': 0.3, 'cleansing_efficiency': 0.2, 'accumulation_intent': 0.3, 'intent_execution_quality': 0.2})
        deception_arbitration_power = get_param_value(tactical_exchange_params.get('deception_arbitration_power'), 2.0)
        deception_impact_sensitivity = get_param_value(tactical_exchange_params.get('deception_impact_sensitivity'), 0.5)
        deception_context_modulator_signal_name = get_param_value(tactical_exchange_params.get('deception_context_modulator_signal_name'), 'chip_health_score_D')
        deception_context_sensitivity = get_param_value(tactical_exchange_params.get('deception_context_sensitivity'), 0.3)
        deception_outcome_weights = get_param_value(tactical_exchange_params.get('deception_outcome_weights'), {'effectiveness': 0.6, 'cost': 0.4})
        deception_outcome_effectiveness_threshold = get_param_value(tactical_exchange_params.get('deception_outcome_effectiveness_threshold'), 0.3)
        deception_outcome_cost_threshold = get_param_value(tactical_exchange_params.get('deception_outcome_cost_threshold'), 0.3)
        intent_execution_quality_params = get_param_value(tactical_exchange_params.get('intent_execution_quality_params'), {})
        intent_execution_quality_slope_period = get_param_value(intent_execution_quality_params.get('slope_period'), 5)
        quality_weights = get_param_value(tactical_exchange_params.get('quality_weights'), {'bullish_absorption': 0.15, 'bullish_purity': 0.15, 'bearish_distribution': 0.15, 'exchange_purity': 0.15, 'structural_optimization': 0.1, 'psychological_pressure_absorption': 0.1, 'exchange_efficiency': 0.05, 'chip_peak_dynamics': 0.15})
        quality_context_signal_name = get_param_value(tactical_exchange_params.get('quality_context_signal_name'), 'winner_loser_momentum_D')
        structural_optimization_slope_period = get_param_value(tactical_exchange_params.get('structural_optimization_slope_period'), 5)
        psychological_pressure_absorption_slope_period = get_param_value(tactical_exchange_params.get('psychological_pressure_absorption_slope_period'), 5)
        chip_peak_dynamics_params = get_param_value(tactical_exchange_params.get('chip_peak_dynamics_params'), {})
        chip_peak_dynamics_slope_period = get_param_value(chip_peak_dynamics_params.get('slope_period'), 5)
        chip_behavioral_pattern_intensity_params = get_param_value(tactical_exchange_params.get('chip_behavioral_pattern_intensity_params'), {})
        chip_behavioral_pattern_intensity_slope_period = get_param_value(chip_behavioral_pattern_intensity_params.get('slope_period'), 5)
        chip_behavioral_pattern_intensity_modulator_factor = get_param_value(chip_behavioral_pattern_intensity_params.get('modulator_factor'), 0.2)
        environment_weights = get_param_value(tactical_exchange_params.get('environment_weights'), {'geography': 0.3, 'chip_fatigue': 0.2, 'chip_stability': 0.2, 'dominant_peak_health': 0.15, 'chip_patience_and_stability': 0.15})
        chip_fatigue_impact_factor = get_param_value(tactical_exchange_params.get('chip_fatigue_impact_factor'), 0.5)
        chip_stability_modulator_signal_name = get_param_value(tactical_exchange_params.get('chip_stability_modulator_signal_name'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        chip_stability_sensitivity = get_param_value(tactical_exchange_params.get('chip_stability_sensitivity'), 0.5)
        dominant_peak_health_slope_period = get_param_value(tactical_exchange_params.get('dominant_peak_health_slope_period'), 5)
        rhythm_persistence_slope_period = get_param_value(tactical_exchange_params.get('rhythm_persistence_slope_period'), 5)
        rhythm_persistence_sensitivity = get_param_value(tactical_exchange_params.get('rhythm_persistence_sensitivity'), 0.5)
        final_fusion_weights = get_param_value(tactical_exchange_params.get('final_fusion_weights'), {'intent': 0.35, 'quality': 0.35, 'environment': 0.2, 'rhythm_persistence': 0.1})
        meta_modulator_weights = get_param_value(tactical_exchange_params.get('meta_modulator_weights'), {'chip_health': 0.25, 'volatility_instability': 0.25, 'main_force_conviction': 0.25, 'main_force_activity': 0.15, 'flow_credibility': 0.1})
        meta_modulator_sensitivity = get_param_value(tactical_exchange_params.get('meta_modulator_sensitivity'), 0.5)
        df_index = df.index
        # --- 维度1: 换手意图 (Exchange Intent) - 纯筹码化与诡道深化 ---
        control_transfer_raw = self._get_safe_series(df, df, 'peak_control_transfer_D', method_name="_diagnose_tactical_exchange")
        cleansing_efficiency_raw = self._get_safe_series(df, df, 'floating_chip_cleansing_efficiency_D', method_name="_diagnose_tactical_exchange")
        norm_control_transfer = get_adaptive_mtf_normalized_bipolar_score(control_transfer_raw, df_index, tf_weights)
        norm_cleansing_efficiency = get_adaptive_mtf_normalized_score(cleansing_efficiency_raw, df_index, ascending=True, tf_weights=tf_weights)
        suppressive_accum_raw = self._get_safe_series(df, df, 'suppressive_accumulation_intensity_D', method_name="_diagnose_tactical_exchange")
        gathering_chasing_raw = self._get_safe_series(df, df, 'gathering_by_chasing_D', method_name="_diagnose_tactical_exchange")
        gathering_support_raw = self._get_safe_series(df, df, 'gathering_by_support_D', method_name="_diagnose_tactical_exchange")
        norm_suppressive_accum = get_adaptive_mtf_normalized_score(suppressive_accum_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_gathering_chasing = get_adaptive_mtf_normalized_score(gathering_chasing_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_gathering_support = get_adaptive_mtf_normalized_score(gathering_support_raw, df_index, ascending=True, tf_weights=tf_weights)
        accumulation_intent_score = (norm_suppressive_accum * 0.4 + norm_gathering_chasing * 0.3 + norm_gathering_support * 0.3)
        active_buying_support_raw = self._get_safe_series(df, df, 'active_buying_support_D', method_name="_diagnose_tactical_exchange")
        active_selling_pressure_raw = self._get_safe_series(df, df, 'active_selling_pressure_D', method_name="_diagnose_tactical_exchange")
        micro_price_impact_asymmetry_raw = self._get_safe_series(df, df, 'micro_price_impact_asymmetry_D', method_name="_diagnose_tactical_exchange")
        order_book_clearing_rate_raw = self._get_safe_series(df, df, 'order_book_clearing_rate_D', method_name="_diagnose_tactical_exchange")
        flow_credibility_index_raw = self._get_safe_series(df, df, 'flow_credibility_index_D', method_name="_diagnose_tactical_exchange")
        norm_active_buying_support = get_adaptive_mtf_normalized_score(active_buying_support_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_active_selling_pressure = get_adaptive_mtf_normalized_score(active_selling_pressure_raw, df_index, ascending=False, tf_weights=tf_weights)
        norm_micro_price_impact_asymmetry = get_adaptive_mtf_normalized_bipolar_score(micro_price_impact_asymmetry_raw, df_index, tf_weights)
        norm_order_book_clearing_rate = get_adaptive_mtf_normalized_score(order_book_clearing_rate_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_flow_credibility_index = get_adaptive_mtf_normalized_score(flow_credibility_index_raw, df_index, ascending=True, tf_weights=tf_weights)
        intent_execution_quality_score = (
            norm_active_buying_support * intent_execution_quality_params.get('buying_support_weight', 0.3) +
            norm_active_selling_pressure * intent_execution_quality_params.get('selling_pressure_weight', 0.2) +
            (1 - norm_micro_price_impact_asymmetry.abs()) * intent_execution_quality_params.get('price_impact_weight', 0.2) +
            norm_order_book_clearing_rate * intent_execution_quality_params.get('clearing_rate_weight', 0.15) +
            norm_flow_credibility_index * intent_execution_quality_params.get('flow_credibility_weight', 0.15)
        ).clip(0, 1)
        base_intent_score = (
            norm_control_transfer * intent_weights.get('control_transfer', 0.3) +
            norm_cleansing_efficiency * intent_weights.get('cleansing_efficiency', 0.2) +
            accumulation_intent_score * intent_weights.get('accumulation_intent', 0.3) +
            intent_execution_quality_score * intent_weights.get('intent_execution_quality', 0.2)
        ).clip(-1, 1)
        chip_fault_raw = self._get_safe_series(df, df, 'chip_fault_magnitude_D', method_name="_diagnose_tactical_exchange")
        mf_conviction_raw = self._get_safe_series(df, df, 'main_force_conviction_index_D', method_name="_diagnose_tactical_exchange")
        norm_chip_fault = get_adaptive_mtf_normalized_score(chip_fault_raw.abs(), df_index, ascending=True, tf_weights=tf_weights)
        norm_mf_conviction = get_adaptive_mtf_normalized_bipolar_score(mf_conviction_raw, df_index, tf_weights)
        chip_deception_direction = np.sign(norm_mf_conviction)
        retail_panic_surrender_raw = self._get_safe_series(df, df, 'retail_panic_surrender_index_D', method_name="_diagnose_tactical_exchange")
        loser_pain_raw = self._get_safe_series(df, df, 'loser_pain_index_D', method_name="_diagnose_tactical_exchange")
        winner_profit_margin_avg_raw = self._get_safe_series(df, df, 'winner_profit_margin_avg_D', method_name="_diagnose_tactical_exchange")
        norm_retail_panic_surrender = get_adaptive_mtf_normalized_score(retail_panic_surrender_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_loser_pain = get_adaptive_mtf_normalized_score(loser_pain_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_winner_profit_margin_avg = get_adaptive_mtf_normalized_score(winner_profit_margin_avg_raw, df_index, ascending=True, tf_weights=tf_weights)
        deception_effectiveness_score = pd.Series(0.0, index=df_index)
        deception_cost_score = pd.Series(0.0, index=df_index)
        induce_bear_mask = chip_deception_direction > 0
        deception_effectiveness_score.loc[induce_bear_mask] = (norm_retail_panic_surrender.loc[induce_bear_mask] + norm_loser_pain.loc[induce_bear_mask]) / 2
        deception_cost_score.loc[induce_bear_mask] = norm_suppressive_accum.loc[induce_bear_mask]
        induce_bull_mask = chip_deception_direction < 0
        deception_effectiveness_score.loc[induce_bull_mask] = norm_winner_profit_margin_avg.loc[induce_bull_mask]
        deception_cost_score.loc[induce_bull_mask] = (1 - get_adaptive_mtf_normalized_score(self._get_safe_series(df, df, 'profit_realization_quality_D', method_name="_diagnose_tactical_exchange"), df_index, ascending=True, tf_weights=tf_weights)).loc[induce_bull_mask]
        deception_quality_modulator = (
            deception_outcome_weights.get('effectiveness', 0.6) * deception_effectiveness_score.clip(0, 1) +
            deception_outcome_weights.get('cost', 0.4) * deception_cost_score.clip(0, 1)
        )
        high_quality_deception_mask = (deception_effectiveness_score > deception_outcome_effectiveness_threshold) & (deception_cost_score > deception_outcome_cost_threshold)
        deception_quality_modulator.loc[~high_quality_deception_mask] *= 0.5
        chip_deception_score_refined = norm_chip_fault * chip_deception_direction * (1 + deception_quality_modulator.clip(0, 1))
        chip_deception_score_refined = chip_deception_score_refined.clip(-1, 1)
        deception_context_modulator_raw = self._get_safe_series(df, df, deception_context_modulator_signal_name, method_name="_diagnose_tactical_exchange")
        norm_deception_context = get_adaptive_mtf_normalized_score(deception_context_modulator_raw, df_index, ascending=True, tf_weights=tf_weights)
        dynamic_deception_impact_sensitivity = deception_impact_sensitivity * (1 - norm_deception_context * deception_context_sensitivity)
        dynamic_deception_impact_sensitivity = dynamic_deception_impact_sensitivity.clip(0.1, 1.0)
        arbitration_weight = (norm_chip_fault * dynamic_deception_impact_sensitivity).pow(deception_arbitration_power).clip(0, 1)
        intent_score = base_intent_score * (1 - arbitration_weight) + chip_deception_score_refined * arbitration_weight
        intent_score = intent_score.clip(-1, 1)
        main_force_activity_raw = self._get_safe_series(df, df, 'main_force_activity_ratio_D', method_name="_diagnose_tactical_exchange")
        main_force_flow_directionality_raw = self._get_safe_series(df, df, 'main_force_flow_directionality_D', method_name="_diagnose_tactical_exchange")
        norm_main_force_activity = get_adaptive_mtf_normalized_score(main_force_activity_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_main_force_flow_directionality = get_adaptive_mtf_normalized_bipolar_score(main_force_flow_directionality_raw, df_index, tf_weights)
        chip_behavioral_pattern_intensity_score = (norm_main_force_activity * 0.6 + norm_main_force_flow_directionality.abs() * 0.4).clip(0, 1)
        intent_score = intent_score * (1 + chip_behavioral_pattern_intensity_score * chip_behavioral_pattern_intensity_modulator_factor)
        intent_score = intent_score.clip(-1, 1)
        # --- 维度2: 换手质量 (Exchange Quality) - 纯筹码化与情境自适应 ---
        chip_momentum_raw = self._get_safe_series(df, df, quality_context_signal_name, method_name="_diagnose_tactical_exchange")
        norm_chip_momentum_context = get_adaptive_mtf_normalized_bipolar_score(chip_momentum_raw, df_index, tf_weights)
        absorption_idx_raw = self._get_safe_series(df, df, 'capitulation_absorption_index_D', method_name="_diagnose_tactical_exchange")
        impulse_purity_raw = self._get_safe_series(df, df, 'upward_impulse_purity_D', method_name="_diagnose_tactical_exchange")
        profit_quality_raw = self._get_safe_series(df, df, 'profit_realization_quality_D', method_name="_diagnose_tactical_exchange")
        norm_absorption = get_adaptive_mtf_normalized_score(absorption_idx_raw, df_index, tf_weights)
        norm_impulse_purity = get_adaptive_mtf_normalized_score(impulse_purity_raw, df_index, tf_weights)
        norm_profit_realization = get_adaptive_mtf_normalized_score(profit_quality_raw, df_index, ascending=False, tf_weights=tf_weights)
        dynamic_bullish_quality_weight = (norm_chip_momentum_context.add(1)/2) * 0.5 + 0.5
        dynamic_bearish_quality_weight = (1 - norm_chip_momentum_context.add(1)/2) * 0.5 + 0.5
        bullish_quality_score = (
            norm_absorption * quality_weights.get('bullish_absorption', 0.15) +
            norm_impulse_purity * quality_weights.get('bullish_purity', 0.15)
        ) * dynamic_bullish_quality_weight
        bearish_quality_score = norm_profit_realization * quality_weights.get('bearish_distribution', 0.15) * dynamic_bearish_quality_weight
        peak_exchange_purity_raw = self._get_safe_series(df, df, 'peak_exchange_purity_D', method_name="_diagnose_tactical_exchange")
        exchange_purity_score = get_adaptive_mtf_normalized_score(peak_exchange_purity_raw, df_index, ascending=True, tf_weights=tf_weights)
        slope_wc_raw = self._get_safe_series(df, df, f'SLOPE_{structural_optimization_slope_period}_winner_concentration_90pct_D', method_name="_diagnose_tactical_exchange")
        slope_css_raw = self._get_safe_series(df, df, f'SLOPE_{structural_optimization_slope_period}_cost_structure_skewness_D', method_name="_diagnose_tactical_exchange")
        slope_psr_raw = self._get_safe_series(df, df, f'SLOPE_{structural_optimization_slope_period}_peak_separation_ratio_D', method_name="_diagnose_tactical_exchange")
        norm_slope_wc = get_adaptive_mtf_normalized_score(slope_wc_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_slope_css = get_adaptive_mtf_normalized_score(slope_css_raw, df_index, ascending=False, tf_weights=tf_weights)
        norm_slope_psr = get_adaptive_mtf_normalized_score(slope_psr_raw, df_index, ascending=False, tf_weights=tf_weights)
        structural_optimization_score = (norm_slope_wc + norm_slope_css + norm_slope_psr) / 3
        structural_optimization_score = structural_optimization_score.clip(0, 1)
        total_loser_rate_raw = self._get_safe_series(df, df, 'total_loser_rate_D', method_name="_diagnose_tactical_exchange")
        total_winner_rate_raw = self._get_safe_series(df, df, 'total_winner_rate_D', method_name="_diagnose_tactical_exchange")
        slope_loser_rate_raw = self._get_safe_series(df, df, f'SLOPE_{psychological_pressure_absorption_slope_period}_total_loser_rate_D', method_name="_diagnose_tactical_exchange")
        slope_winner_rate_raw = self._get_safe_series(df, df, f'SLOPE_{psychological_pressure_absorption_slope_period}_total_winner_rate_D', method_name="_diagnose_tactical_exchange")
        norm_total_loser_rate = get_adaptive_mtf_normalized_score(total_loser_rate_raw, df_index, ascending=False, tf_weights=tf_weights)
        norm_total_winner_rate = get_adaptive_mtf_normalized_score(total_winner_rate_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_slope_loser_rate = get_adaptive_mtf_normalized_bipolar_score(slope_loser_rate_raw, df_index, tf_weights, sensitivity=1.0)
        norm_slope_winner_rate = get_adaptive_mtf_normalized_bipolar_score(slope_winner_rate_raw, df_index, tf_weights, sensitivity=1.0)
        loser_absorption_quality = norm_absorption * (1 - norm_slope_loser_rate.clip(upper=0).abs())
        winner_resilience_quality = norm_profit_realization * (1 - norm_slope_winner_rate.clip(lower=0))
        psychological_pressure_absorption_score = (loser_absorption_quality + winner_resilience_quality) / 2
        psychological_pressure_absorption_score = psychological_pressure_absorption_score.clip(0, 1)
        volume_raw = self._get_safe_series(df, df, 'volume_D', method_name="_diagnose_tactical_exchange")
        norm_volume = get_adaptive_mtf_normalized_score(volume_raw, df_index, ascending=True, tf_weights=tf_weights)
        exchange_efficiency_score = structural_optimization_score / (norm_volume.replace(0, 1e-6))
        exchange_efficiency_score = exchange_efficiency_score.clip(0, 1)
        secondary_peak_cost_raw = self._get_safe_series(df, df, 'secondary_peak_cost_D', method_name="_diagnose_tactical_exchange")
        dominant_peak_volume_ratio_raw = self._get_safe_series(df, df, 'dominant_peak_volume_ratio_D', method_name="_diagnose_tactical_exchange")
        norm_secondary_peak_cost = get_adaptive_mtf_normalized_score(secondary_peak_cost_raw, df_index, ascending=False, tf_weights=tf_weights)
        norm_dominant_peak_volume_ratio = get_adaptive_mtf_normalized_score(dominant_peak_volume_ratio_raw, df_index, ascending=True, tf_weights=tf_weights)
        chip_peak_dynamics_score = (norm_secondary_peak_cost * chip_peak_dynamics_params.get('secondary_cost_weight', 0.5) +
                                    norm_dominant_peak_volume_ratio * chip_peak_dynamics_params.get('secondary_volume_weight', 0.5)).clip(0, 1)
        quality_score = (
            bullish_quality_score * (1 - dynamic_bearish_quality_weight) +
            bearish_quality_score * (1 - dynamic_bullish_quality_weight) +
            exchange_purity_score * quality_weights.get('exchange_purity', 0.15) +
            structural_optimization_score * quality_weights.get('structural_optimization', 0.1) +
            psychological_pressure_absorption_score * quality_weights.get('psychological_pressure_absorption', 0.1) +
            exchange_efficiency_score * quality_weights.get('exchange_efficiency', 0.05) +
            chip_peak_dynamics_score * quality_weights.get('chip_peak_dynamics', 0.15)
        ).clip(-1, 1)
        quality_score = quality_score * (1 + chip_behavioral_pattern_intensity_score * chip_behavioral_pattern_intensity_modulator_factor)
        quality_score = quality_score.clip(-1, 1)
        # --- 维度3: 换手环境 (Exchange Context) - 纯筹码化与情境自适应 ---
        chip_fatigue_raw = self._get_safe_series(df, df, 'chip_fatigue_index_D', method_name="_diagnose_tactical_exchange")
        norm_chip_fatigue = get_adaptive_mtf_normalized_score(chip_fatigue_raw, df_index, ascending=True, tf_weights=tf_weights)
        chip_stability_modulator_raw = self._get_safe_series(df, df, chip_stability_modulator_signal_name, method_name="_diagnose_tactical_exchange")
        norm_chip_stability_modulator = get_adaptive_mtf_normalized_score(chip_stability_modulator_raw, df_index, ascending=False, tf_weights=tf_weights)
        chip_health_raw = self._get_safe_series(df, df, 'chip_health_score_D', method_name="_diagnose_tactical_exchange")
        norm_chip_health = get_adaptive_mtf_normalized_bipolar_score(chip_health_raw, df_index, tf_weights)
        dynamic_chip_fatigue_impact = norm_chip_fatigue * chip_fatigue_impact_factor
        dynamic_chip_stability_bonus = norm_chip_stability_modulator * chip_stability_sensitivity
        dominant_peak_solidity_raw = self._get_safe_series(df, df, 'dominant_peak_solidity_D', method_name="_diagnose_tactical_exchange")
        slope_dps_raw = self._get_safe_series(df, df, f'SLOPE_{dominant_peak_health_slope_period}_dominant_peak_solidity_D', method_name="_diagnose_tactical_exchange")
        norm_dps = get_adaptive_mtf_normalized_score(dominant_peak_solidity_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_slope_dps = get_adaptive_mtf_normalized_bipolar_score(slope_dps_raw, df_index, tf_weights)
        dominant_peak_health_score = (norm_dps * 0.7 + norm_slope_dps * 0.3).clip(0, 1)
        winner_stability_index_raw = self._get_safe_series(df, df, 'winner_stability_index_D', method_name="_diagnose_tactical_exchange")
        norm_winner_stability = get_adaptive_mtf_normalized_score(winner_stability_index_raw, df_index, ascending=True, tf_weights=tf_weights)
        chip_patience_ratio = norm_gathering_support / (norm_gathering_support + norm_gathering_chasing + 1e-6)
        chip_patience_score = get_adaptive_mtf_normalized_score(chip_patience_ratio, df_index, ascending=True, tf_weights=tf_weights)
        chip_patience_and_stability_score = (norm_winner_stability * 0.5 + chip_patience_score * 0.5).clip(0, 1)
        context_score = (
            battlefield_geography * environment_weights.get('geography', 0.3) -
            dynamic_chip_fatigue_impact * environment_weights.get('chip_fatigue', 0.2) +
            dynamic_chip_stability_bonus * environment_weights.get('chip_stability', 0.2) +
            dominant_peak_health_score * environment_weights.get('dominant_peak_health', 0.15) +
            chip_patience_and_stability_score * environment_weights.get('chip_patience_and_stability', 0.15)
        ).clip(-1, 1)
        # --- 维度4: 换手节奏与持续性 (Exchange Rhythm & Persistence) ---
        rhythm_intent_slope = base_intent_score.diff(rhythm_persistence_slope_period).fillna(0)
        rhythm_quality_slope = quality_score.diff(rhythm_persistence_slope_period).fillna(0)
        norm_rhythm_intent_slope = get_adaptive_mtf_normalized_bipolar_score(rhythm_intent_slope, df_index, tf_weights)
        norm_rhythm_quality_slope = get_adaptive_mtf_normalized_bipolar_score(rhythm_quality_slope, df_index, tf_weights)
        rhythm_and_persistence_score = (norm_rhythm_intent_slope + norm_rhythm_quality_slope) / 2
        rhythm_and_persistence_score = (rhythm_and_persistence_score * rhythm_persistence_sensitivity).clip(-1, 1)
        # --- 最终融合 ---
        volatility_instability_raw = self._get_safe_series(df, df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', method_name="_diagnose_tactical_exchange")
        norm_volatility_instability = get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=False, tf_weights=tf_weights)
        main_force_conviction_abs_raw = self._get_safe_series(df, df, 'main_force_conviction_index_D', method_name="_diagnose_tactical_exchange").abs()
        norm_main_force_conviction = get_adaptive_mtf_normalized_score(main_force_conviction_abs_raw, df_index, ascending=True, tf_weights=tf_weights)
        main_force_activity_abs_raw = self._get_safe_series(df, df, 'main_force_activity_ratio_D', method_name="_diagnose_tactical_exchange").abs()
        norm_main_force_activity_meta = get_adaptive_mtf_normalized_score(main_force_activity_abs_raw, df_index, ascending=True, tf_weights=tf_weights)
        flow_credibility_index_meta_raw = self._get_safe_series(df, df, 'flow_credibility_index_D', method_name="_diagnose_tactical_exchange")
        norm_flow_credibility_index_meta = get_adaptive_mtf_normalized_score(flow_credibility_index_meta_raw, df_index, ascending=True, tf_weights=tf_weights)
        market_context_meta_modulator = (
            norm_chip_health * meta_modulator_weights.get('chip_health', 0.25) +
            norm_volatility_instability * meta_modulator_weights.get('volatility_instability', 0.25) +
            norm_main_force_conviction * meta_modulator_weights.get('main_force_conviction', 0.25) +
            norm_main_force_activity_meta * meta_modulator_weights.get('main_force_activity', 0.15) +
            norm_flow_credibility_index_meta * meta_modulator_weights.get('flow_credibility', 0.1)
        ).clip(0, 1)
        dynamic_final_fusion_weights = {
            'intent': final_fusion_weights.get('intent', 0.35) * (1 + market_context_meta_modulator * meta_modulator_sensitivity),
            'quality': final_fusion_weights.get('quality', 0.35) * (1 + market_context_meta_modulator * meta_modulator_sensitivity),
            'environment': final_fusion_weights.get('environment', 0.2) * (1 + market_context_meta_modulator * meta_modulator_sensitivity),
            'rhythm_persistence': final_fusion_weights.get('rhythm_persistence', 0.1) * (1 + market_context_meta_modulator * meta_modulator_sensitivity)
        }
        sum_dynamic_weights = sum(dynamic_final_fusion_weights.values())
        normalized_dynamic_weights = {k: v / sum_dynamic_weights for k, v in dynamic_final_fusion_weights.items()}
        final_score = (
            intent_score * normalized_dynamic_weights.get('intent', 0.35) +
            quality_score * normalized_dynamic_weights.get('quality', 0.35) +
            context_score * normalized_dynamic_weights.get('environment', 0.2) +
            rhythm_and_persistence_score * normalized_dynamic_weights.get('rhythm_persistence', 0.1)
        ).clip(-1, 1)
        return final_score.clip(-1, 1).fillna(0.0).astype(np.float32)

    def _diagnose_strategic_tactical_harmony(self, df: pd.DataFrame, strategic_posture: pd.Series, tactical_exchange: pd.Series, holder_sentiment_scores: pd.Series) -> pd.Series: # [修改代码行] 增加 holder_sentiment_scores 参数
        """
        【V3.0 · 诡道微观共振版】诊断战略与战术的和谐度
        - 核心升级1: 战术执行微观深化。引入高频微观筹码行为（如日内筹码流平衡、订单簿压力等）作为“当日战术执行”的更精细化输入，提升战术评估的颗粒度和准确性。
        - 核心升级2: 动态权重调制精细化。战略与战术的融合权重不再固定，而是根据筹码波动不稳定性、筹码疲劳指数等筹码层情境因子动态调整，以适应不同市场阶段的侧重点。
        - 核心升级3: 和谐因子情境纯筹码化。和谐度因子的情境调制器严格限定为筹码层信号（如持仓信念韧性、价筹张力），确保信号的纯粹性。
        - 核心升级4: 冲突情境诡道深化。明确识别战略与战术方向完全背离的“冲突区”，并引入诡道因子（如欺骗指数、对倒强度）进行调制，对伴随欺骗的冲突施加更严厉惩罚。
        - 核心升级5: 趋势一致性品质校准。当战略与战术在同一方向上高度协同并具备足够强度时，引入筹码品质因子（如筹码健康度、主力信念指数）进行校准，确保奖励的是高质量、可持续的趋势。
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        # --- 探针: 原始输入 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        df_index = df.index
        # --- 参数加载 ---
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        harmony_params = get_param_value(p_conf.get('strategic_tactical_harmony_params'), {})
        # V3.0 动态权重参数
        strategic_weight_base = get_param_value(harmony_params.get('strategic_weight_base'), 0.6)
        tactical_weight_base = get_param_value(harmony_params.get('tactical_weight_base'), 0.4)
        dynamic_weight_modulator_signal_name = get_param_value(harmony_params.get('dynamic_weight_modulator_signal'), 'VOLATILITY_INSTABILITY_INDEX_21d_D') # 更改为筹码波动不稳定性
        dynamic_weight_sensitivity = get_param_value(harmony_params.get('dynamic_weight_sensitivity'), 0.3) # 调整敏感度
        # V3.0 和谐因子参数
        harmony_non_linear_exponent = get_param_value(harmony_params.get('harmony_non_linear_exponent'), 2.5) # 提高非线性惩罚
        harmony_context_modulator_signal_name = get_param_value(harmony_params.get('harmony_context_modulator_signal'), 'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT') # 更改为纯筹码信号
        harmony_context_sensitivity = get_param_value(harmony_params.get('harmony_context_sensitivity'), 0.4) # 调整敏感度
        # V3.0 冲突惩罚参数
        conflict_threshold = get_param_value(harmony_params.get('conflict_threshold'), 0.6) # 提高冲突阈值
        conflict_penalty_factor = get_param_value(harmony_params.get('conflict_penalty_factor'), 0.7) # 提高惩罚力度
        deception_modulator_signal_name = get_param_value(harmony_params.get('deception_modulator_signal'), 'deception_index_D') # 引入欺骗指数
        deception_penalty_sensitivity = get_param_value(harmony_params.get('deception_penalty_sensitivity'), 0.5)
        # V3.0 趋势奖励参数
        trend_alignment_threshold = get_param_value(harmony_params.get('trend_alignment_threshold'), 0.75) # 提高奖励阈值
        trend_bonus_factor = get_param_value(harmony_params.get('trend_bonus_factor'), 0.15) # 提高奖励力度
        quality_calibrator_signal_name = get_param_value(harmony_params.get('quality_calibrator_signal'), 'chip_health_score_D') # 引入品质校准
        quality_calibration_sensitivity = get_param_value(harmony_params.get('quality_calibration_sensitivity'), 0.5)
        high_harmony_threshold = get_param_value(harmony_params.get('high_harmony_threshold'), 0.8)
        # --- 信号依赖校验 ---
        required_signals = [
            dynamic_weight_modulator_signal_name,
            deception_modulator_signal_name,
            quality_calibrator_signal_name
        ]
        # 如果 harmony_context_modulator_signal_name 不是 SCORE_CHIP_AXIOM_HOLDER_SENTIMENT，则也需要从 df 中获取
        if harmony_context_modulator_signal_name != 'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT':
            required_signals.append(harmony_context_modulator_signal_name)
        if not self._validate_required_signals(df, required_signals, "_diagnose_strategic_tactical_harmony"):
            return pd.Series(0.0, index=df.index)
        # --- 1. 动态权重融合 (Dynamic Weight Fusion) ---
        # 使用筹码波动不稳定性作为调制器
        dynamic_weight_modulator_raw = self._get_safe_series(df, df, dynamic_weight_modulator_signal_name, 0.0, method_name="_diagnose_strategic_tactical_harmony")
        # 归一化到 [0, 1]，波动性越高，值越大
        norm_dynamic_weight_modulator = get_adaptive_mtf_normalized_score(dynamic_weight_modulator_raw, df_index, ascending=True, tf_weights=tf_weights)
        # 根据筹码波动不稳定性调整战略和战术权重
        # 波动性越高 (norm_dynamic_weight_modulator 越大)，市场越不稳定，短期战术执行的重要性被放大，战略权重降低
        # 波动性越低 (norm_dynamic_weight_modulator 越小)，市场越稳定，长期战略意图的权重越高，战术权重降低
        dynamic_strategic_weight = strategic_weight_base * (1 - norm_dynamic_weight_modulator * dynamic_weight_sensitivity)
        dynamic_tactical_weight = tactical_weight_base * (1 + norm_dynamic_weight_modulator * dynamic_weight_sensitivity)
        # 归一化动态权重，使其和为1
        sum_dynamic_weights = dynamic_strategic_weight + dynamic_tactical_weight
        dynamic_strategic_weight = dynamic_strategic_weight / sum_dynamic_weights
        dynamic_tactical_weight = dynamic_tactical_weight / sum_dynamic_weights
        # 计算以动态权重为基础的意图分
        base_intent_score = strategic_posture * dynamic_strategic_weight + tactical_exchange * dynamic_tactical_weight
        # --- 2. 和谐因子非线性增强 (Non-linear Harmony Factor Enhancement) ---
        raw_difference = (strategic_posture - tactical_exchange).abs() / 2 # 归一化到 [0, 1]
        # 应用非线性变换，放大差异的影响
        non_linear_diff = raw_difference.pow(harmony_non_linear_exponent)
        # 基础和谐因子
        harmony_factor = (1 - non_linear_diff).clip(lower=0)
        # 情境化和谐度调制 (使用持仓信念韧性作为调制器，纯筹码信号)
        # [修改代码行] 从传入的 holder_sentiment_scores 中获取，而不是从 df 中获取
        if harmony_context_modulator_signal_name == 'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT':
            harmony_context_modulator_raw = holder_sentiment_scores
        else:
            harmony_context_modulator_raw = self._get_safe_series(df, df, harmony_context_modulator_signal_name, 0.0, method_name="_diagnose_strategic_tactical_harmony")
        norm_harmony_context = get_adaptive_mtf_normalized_bipolar_score(harmony_context_modulator_raw, df_index, tf_weights) # 归一化到 [-1, 1]
        # 持仓信念韧性越强（norm_harmony_context 越大），市场对分歧的容忍度越高，和谐度因子被放大
        # 持仓信念韧性越弱（norm_harmony_context 越小），市场对分歧越敏感，和谐度因子被削弱
        context_modulation_effect = (norm_harmony_context * harmony_context_sensitivity).clip(-0.5, 0.5) # 限制调制效果
        harmony_factor = harmony_factor * (1 + context_modulation_effect)
        harmony_factor = harmony_factor.clip(0, 1) # 确保在 [0, 1] 范围内
        # --- 3. 冲突情境识别与惩罚 (Conflict Context Recognition & Penalty) ---
        conflict_penalty_factor_adjusted = pd.Series(1.0, index=df_index)
        # 识别战略与战术方向完全背离且强度足够的情境
        strong_bullish_strategic_bearish_tactical = (strategic_posture > conflict_threshold) & (tactical_exchange < -conflict_threshold)
        strong_bearish_strategic_bullish_tactical = (strategic_posture < -conflict_threshold) & (tactical_exchange > conflict_threshold)
        conflict_mask = strong_bullish_strategic_bearish_tactical | strong_bearish_strategic_bullish_tactical
        # 引入诡道因子进行调制
        deception_raw = self._get_safe_series(df, df, deception_modulator_signal_name, 0.0, method_name="_diagnose_strategic_tactical_harmony")
        norm_deception = get_adaptive_mtf_normalized_bipolar_score(deception_raw, df_index, tf_weights) # 归一化到 [-1, 1]
        # 如果冲突伴随着明显的“诱多”式欺骗 (norm_deception > 0)，惩罚更严厉
        # 如果冲突伴随着“诱空”式欺骗 (norm_deception < 0)，且主力信念坚定，则可能减弱惩罚 (此处简化为不增强惩罚)
        deception_impact = pd.Series(0.0, index=df_index)
        deception_impact.loc[conflict_mask & (norm_deception > 0)] = norm_deception.loc[conflict_mask & (norm_deception > 0)] * deception_penalty_sensitivity
        # 在冲突情境下施加额外惩罚，并考虑欺骗影响
        conflict_penalty_factor_adjusted.loc[conflict_mask] = 1 - (conflict_penalty_factor + deception_impact.loc[conflict_mask]).clip(0, 1)
        conflict_penalty_factor_adjusted = conflict_penalty_factor_adjusted.clip(0, 1) # 确保惩罚因子在 [0, 1] 范围内
        # --- 4. 趋势一致性奖励 (Trend Alignment Bonus) ---
        alignment_bonus = pd.Series(0.0, index=df_index)
        # 识别战略与战术高度协同且强度足够的情境
        bullish_alignment_mask = (strategic_posture > trend_alignment_threshold) & \
                                 (tactical_exchange > trend_alignment_threshold) & \
                                 (harmony_factor > high_harmony_threshold)
        bearish_alignment_mask = (strategic_posture < -trend_alignment_threshold) & \
                                 (tactical_exchange < -trend_alignment_threshold) & \
                                 (harmony_factor > high_harmony_threshold)
        # 引入筹码品质因子进行校准
        quality_calibrator_raw = self._get_safe_series(df, df, quality_calibrator_signal_name, 0.0, method_name="_diagnose_strategic_tactical_harmony")
        norm_quality_calibrator = get_adaptive_mtf_normalized_score(quality_calibrator_raw, df_index, ascending=True, tf_weights=tf_weights) # 归一化到 [0, 1]
        # 只有当品质因子较高时才给予奖励，否则削弱奖励
        calibrated_bonus_factor = trend_bonus_factor * (1 + norm_quality_calibrator * quality_calibration_sensitivity)
        calibrated_bonus_factor = calibrated_bonus_factor.clip(0, trend_bonus_factor * 2) # 限制奖励上限
        # 给予奖励或惩罚
        alignment_bonus.loc[bullish_alignment_mask] = calibrated_bonus_factor.loc[bullish_alignment_mask]
        alignment_bonus.loc[bearish_alignment_mask] = -calibrated_bonus_factor.loc[bearish_alignment_mask]
        # --- 最终融合 ---
        # 基础意图分 * 和谐因子 * 冲突惩罚 + 趋势奖励
        final_score = base_intent_score * harmony_factor * conflict_penalty_factor_adjusted + alignment_bonus
        return final_score.clip(-1, 1).fillna(0.0).astype(np.float32)

    def _diagnose_harmony_inflection(self, df: pd.DataFrame, harmony_score: pd.Series) -> pd.Series:
        """
        【V3.0 · 诡道确认版】诊断战略与战术和谐度的动态转折点，旨在构建一个诡道拐点判别与确认系统。
        - 核心升级1: 动态阈值自适应：和谐度所处区间（低位、中位、高位）的判断阈值不再固定，而是根据市场波动性或筹码健康度动态调整，提高对拐点“位置”判断的适应性。
        - 核心升级2: 非对称拐点动能融合：采用更复杂的非线性函数融合速度和加速度，并允许正向和负向拐点使用不同的融合参数，以更精细地量化拐点背后的真实动能，并反映市场情绪的非对称性。
        - 核心升级3: 诡道博弈过滤与惩罚：引入欺骗指数、对倒强度等诡道因子作为调制器，识别并惩罚伴随诱多欺骗的正向拐点，或适度削弱伴随诱空洗盘的负向拐点，提高信号真实性。
        - 核心升级4: 拐点延续性确认奖励：引入短期延续性检查机制，如果拐点方向在后续几天得到确认，则给予额外奖励，增强信号可靠性和鲁棒性。
        - 核心升级5: 增强情境调制器：除了筹码健康度和波动性，再引入主力信念指数作为情境调制器，更全面评估拐点信号在不同市场参与者意图下的可靠性。
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        df_index = df.index
        # --- 参数加载 ---
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        inflection_params = get_param_value(p_conf.get('harmony_inflection_params'), {})
        # V3.0 广义拐点定义参数
        velocity_period = get_param_value(inflection_params.get('velocity_period'), 1)
        acceleration_period = get_param_value(inflection_params.get('acceleration_period'), 1)
        positive_strength_tanh_factor = get_param_value(inflection_params.get('positive_strength_tanh_factor'), 1.5)
        negative_strength_tanh_factor = get_param_value(inflection_params.get('negative_strength_tanh_factor'), 1.5)
        # V3.0 动态阈值自适应参数
        base_low_harmony_threshold = get_param_value(inflection_params.get('base_low_harmony_threshold'), 0.2)
        base_high_harmony_threshold = get_param_value(inflection_params.get('base_high_harmony_threshold'), 0.8)
        threshold_modulator_signal_name = get_param_value(inflection_params.get('threshold_modulator_signal'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        threshold_modulator_sensitivity = get_param_value(inflection_params.get('threshold_modulator_sensitivity'), 0.2)
        low_harmony_boost_factor = get_param_value(inflection_params.get('low_harmony_boost_factor'), 1.5)
        high_harmony_boost_factor = get_param_value(inflection_params.get('high_harmony_boost_factor'), 1.5)
        mid_harmony_neutral_factor = get_param_value(inflection_params.get('mid_harmony_neutral_factor'), 1.0)
        # V3.0 诡道博弈过滤与惩罚参数
        deception_signal_name = get_param_value(inflection_params.get('deception_signal'), 'deception_index_D')
        wash_trade_signal_name = get_param_value(inflection_params.get('wash_trade_signal'), 'wash_trade_intensity_D')
        deception_penalty_sensitivity = get_param_value(inflection_params.get('deception_penalty_sensitivity'), 0.7)
        wash_trade_mitigation_sensitivity = get_param_value(inflection_params.get('wash_trade_mitigation_sensitivity'), 0.3)
        # V3.0 拐点延续性确认奖励参数
        persistence_period = get_param_value(inflection_params.get('persistence_period'), 2)
        persistence_bonus_factor = get_param_value(inflection_params.get('persistence_bonus_factor'), 0.1)
        # V3.0 增强情境调制器参数
        context_modulator_signal_1_name = get_param_value(inflection_params.get('context_modulator_signal_1'), 'chip_health_score_D')
        context_modulator_signal_2_name = get_param_value(inflection_params.get('context_modulator_signal_2'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        context_modulator_signal_3_name = get_param_value(inflection_params.get('context_modulator_signal_3'), 'main_force_conviction_index_D') # 新增主力信念
        context_modulator_sensitivity_health = get_param_value(inflection_params.get('context_modulator_sensitivity_health'), 0.5)
        context_modulator_sensitivity_volatility = get_param_value(inflection_params.get('context_modulator_sensitivity_volatility'), 0.3)
        context_modulator_sensitivity_conviction = get_param_value(inflection_params.get('context_modulator_sensitivity_conviction'), 0.4)
        # --- 信号依赖校验 ---
        required_signals = [
            threshold_modulator_signal_name,
            deception_signal_name,
            wash_trade_signal_name,
            context_modulator_signal_1_name,
            context_modulator_signal_2_name,
            context_modulator_signal_3_name
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_harmony_inflection"):
            return pd.Series(0.0, index=df.index)
        # --- 1. 计算速度与加速度 (保留方向信息) ---
        harmony_velocity = harmony_score.diff(velocity_period).fillna(0)
        harmony_acceleration = harmony_velocity.diff(acceleration_period).fillna(0)
        # 使用双极归一化，保留方向和强度
        norm_velocity = get_adaptive_mtf_normalized_bipolar_score(harmony_velocity, df_index, tf_weights)
        norm_acceleration = get_adaptive_mtf_normalized_bipolar_score(harmony_acceleration, df_index, tf_weights)
        # --- 2. 非对称拐点动能融合 (Asymmetric Inflection Momentum Fusion) ---
        # 底部拐点强度 (速度负 -> 正, 或速度负但加速度正)
        positive_inflection_strength = pd.Series(0.0, index=df_index)
        positive_inflection_mask = ((norm_velocity.shift(1) < 0) & (norm_velocity >= 0)) | \
                                   ((norm_velocity < 0) & (norm_acceleration > 0)) | \
                                   ((norm_velocity >= 0) & (norm_acceleration > 0)) # 速度和加速度都为正也算正向动能
        positive_inflection_strength.loc[positive_inflection_mask] = \
            np.tanh((norm_velocity.loc[positive_inflection_mask].clip(lower=0) + norm_acceleration.loc[positive_inflection_mask].clip(lower=0)) * positive_strength_tanh_factor)
        # 顶部拐点强度 (速度正 -> 负, 或速度正但加速度负)
        negative_inflection_strength = pd.Series(0.0, index=df_index)
        negative_inflection_mask = ((norm_velocity.shift(1) > 0) & (norm_velocity <= 0)) | \
                                   ((norm_velocity > 0) & (norm_acceleration < 0)) | \
                                   ((norm_velocity <= 0) & (norm_acceleration < 0)) # 速度和加速度都为负也算负向动能
        negative_inflection_strength.loc[negative_inflection_mask] = \
            np.tanh((norm_velocity.loc[negative_inflection_mask].abs().clip(lower=0) + norm_acceleration.loc[negative_inflection_mask].abs().clip(lower=0)) * negative_strength_tanh_factor)
        # 综合拐点强度 (保留方向)
        inflection_strength = positive_inflection_strength - negative_inflection_strength
        # --- 3. 动态阈值自适应 (Dynamic Threshold Adaptation) ---
        threshold_modulator_raw = self._get_safe_series(df, df, threshold_modulator_signal_name, 0.0, method_name="_diagnose_harmony_inflection")
        # 归一化到 [0, 1]，波动性越高，值越大
        norm_threshold_modulator = get_adaptive_mtf_normalized_score(threshold_modulator_raw, df_index, ascending=True, tf_weights=tf_weights)
        # 动态调整阈值：波动性越高，阈值范围越宽 (对拐点判断更宽松)
        dynamic_low_harmony_threshold = base_low_harmony_threshold * (1 - norm_threshold_modulator * threshold_modulator_sensitivity)
        dynamic_high_harmony_threshold = base_high_harmony_threshold * (1 + norm_threshold_modulator * threshold_modulator_sensitivity)
        dynamic_low_harmony_threshold = dynamic_low_harmony_threshold.clip(0.05, 0.3) # 限制合理范围
        dynamic_high_harmony_threshold = dynamic_high_harmony_threshold.clip(0.7, 0.95) # 限制合理范围
        # --- 4. 动态位置敏感度 (Dynamic Position Sensitivity) ---
        position_sensitivity_factor = pd.Series(mid_harmony_neutral_factor, index=df_index)
        # 低和谐度区域 (0 到 dynamic_low_harmony_threshold)
        low_harmony_zone_mask = harmony_score < dynamic_low_harmony_threshold
        position_sensitivity_factor.loc[low_harmony_zone_mask & (inflection_strength > 0)] = low_harmony_boost_factor
        position_sensitivity_factor.loc[low_harmony_zone_mask & (inflection_strength < 0)] = 1 / low_harmony_boost_factor # 削弱负向拐点
        # 高和谐度区域 (dynamic_high_harmony_threshold 到 1)
        high_harmony_zone_mask = harmony_score > dynamic_high_harmony_threshold
        position_sensitivity_factor.loc[high_harmony_zone_mask & (inflection_strength < 0)] = high_harmony_boost_factor # 增强负向拐点
        position_sensitivity_factor.loc[high_harmony_zone_mask & (inflection_strength > 0)] = 1 / high_harmony_boost_factor # 削弱正向拐点
        # --- 5. 诡道博弈过滤与惩罚 (Deceptive Game Filtering & Penalty) ---
        deception_raw = self._get_safe_series(df, df, deception_signal_name, 0.0, method_name="_diagnose_harmony_inflection")
        norm_deception = get_adaptive_mtf_normalized_bipolar_score(deception_raw, df_index, tf_weights) # 归一化到 [-1, 1]
        wash_trade_raw = self._get_safe_series(df, df, wash_trade_signal_name, 0.0, method_name="_diagnose_harmony_inflection")
        norm_wash_trade = get_adaptive_mtf_normalized_score(wash_trade_raw, df_index, ascending=True, tf_weights=tf_weights) # 归一化到 [0, 1]
        deception_modulator = pd.Series(1.0, index=df_index)
        # 诱多陷阱惩罚：正向拐点伴随高欺骗指数
        bull_trap_mask = (inflection_strength > 0) & (norm_deception > 0)
        deception_modulator.loc[bull_trap_mask] = 1 - (norm_deception.loc[bull_trap_mask] * deception_penalty_sensitivity).clip(0, 1)
        # 诱空洗盘缓解：负向拐点伴随高对倒强度 (可能不是真实恶化)
        bear_trap_mitigation_mask = (inflection_strength < 0) & (norm_wash_trade > 0)
        deception_modulator.loc[bear_trap_mitigation_mask] = 1 + (norm_wash_trade.loc[bear_trap_mitigation_mask] * wash_trade_mitigation_sensitivity).clip(0, 0.5) # 限制缓解幅度
        # 应用诡道调制
        inflection_strength_modulated = inflection_strength * deception_modulator
        # --- 6. 拐点延续性确认奖励 (Inflection Persistence Confirmation Bonus) ---
        persistence_bonus = pd.Series(0.0, index=df_index)
        # 计算拐点方向的短期延续性
        # 如果当前是正向拐点，且过去 persistence_period 天内，inflection_strength 大部分时间为正
        positive_persistence_mask = (inflection_strength_modulated > 0) & \
                                    (inflection_strength_modulated.rolling(window=persistence_period, min_periods=1).mean() > 0)
        persistence_bonus.loc[positive_persistence_mask] = persistence_bonus_factor
        # 如果当前是负向拐点，且过去 persistence_period 天内，inflection_strength 大部分时间为负
        negative_persistence_mask = (inflection_strength_modulated < 0) & \
                                    (inflection_strength_modulated.rolling(window=persistence_period, min_periods=1).mean() < 0)
        persistence_bonus.loc[negative_persistence_mask] = -persistence_bonus_factor
        # --- 7. 增强情境调制器 (Enhanced Contextual Modulators) ---
        # 筹码健康度 (chip_health_score_D)
        chip_health_raw = self._get_safe_series(df, df, context_modulator_signal_1_name, 0.0, method_name="_diagnose_harmony_inflection")
        norm_chip_health = get_adaptive_mtf_normalized_score(chip_health_raw, df_index, ascending=True, tf_weights=tf_weights) # 归一化到 [0, 1]
        # 筹码波动不稳定性 (VOLATILITY_INSTABILITY_INDEX_21d_D)
        volatility_instability_raw = self._get_safe_series(df, df, context_modulator_signal_2_name, 0.0, method_name="_diagnose_harmony_inflection")
        norm_volatility_instability = get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=False, tf_weights=tf_weights) # 波动性越高，值越小 (负向影响)
        # 主力信念指数 (main_force_conviction_index_D)
        main_force_conviction_raw = self._get_safe_series(df, df, context_modulator_signal_3_name, 0.0, method_name="_diagnose_harmony_inflection")
        norm_main_force_conviction = get_adaptive_mtf_normalized_score(main_force_conviction_raw.abs(), df_index, ascending=True, tf_weights=tf_weights) # 绝对值越高，信念越强
        # 融合情境调制器
        # 健康度越高，波动性越低，主力信念越强，情境因子越接近1，拐点信号越可靠
        context_modulator = (
            (1 + norm_chip_health * context_modulator_sensitivity_health) *
            (1 + norm_volatility_instability * context_modulator_sensitivity_volatility) *
            (1 + norm_main_force_conviction * context_modulator_sensitivity_conviction)
        ).clip(0.5, 2.0) # 限制调制范围
        # --- 最终融合 ---
        # 拐点强度 (诡道调制后) * 动态位置敏感度 * 情境调制器 + 延续性奖励
        final_score = (inflection_strength_modulated * position_sensitivity_factor * context_modulator) + persistence_bonus
        return final_score.clip(-1, 1).fillna(0.0).astype(np.float32)



