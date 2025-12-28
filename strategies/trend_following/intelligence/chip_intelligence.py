import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, Union
from strategies.trend_following import utils
from strategies.trend_following.utils import (
    get_params_block, get_param_value, get_adaptive_mtf_normalized_score, load_external_json_config,
    get_adaptive_mtf_normalized_bipolar_score, _robust_geometric_mean, normalize_score
)

class ChipIntelligence:
    def __init__(self, strategy_instance):
        """
        【V2.3 · 外部配置加载版】
        - 核心升级: `chip_ultimate_params` 现在从外部文件 `config/intelligence/chip.json` 加载，
                     解决了配置块转移的问题，并确保了模块化。
        - 核心修复: 注入 debug_params，并预处理 probe_dates，解决 AttributeError。
        """
        self.strategy = strategy_instance
        self.score_type_map = get_params_block(self.strategy, 'score_type_map', {})
        process_params = get_params_block(self.strategy, 'process_intelligence_params', {})
        self.bipolar_sensitivity = get_param_value(process_params.get('bipolar_sensitivity'), 1.0)
        # 从外部文件加载 chip_ultimate_params
        # loaded_chip_config 应该直接是 chip.json 的内容
        loaded_chip_config = load_external_json_config("config/intelligence/chip.json", {})
        # 直接从加载的配置中获取 chip_ultimate_params 块，而不是通过 get_params_block
        self.chip_ultimate_params = loaded_chip_config.get('chip_ultimate_params', {})
        self.debug_params = get_params_block(self.strategy, 'debug_params', {})
        self.should_probe = self.debug_params.get('should_probe', False)
        self.probe_dates_set = {pd.to_datetime(d).date() for d in self.debug_params.get('probe_dates', [])}

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

    def _get_all_required_signals(self, df: pd.DataFrame, required_signals: list, method_name: str) -> Dict[str, pd.Series]:
        """
        高效地从DataFrame中获取所有必需的Series，如果不存在则打印警告并返回默认Series。
        """
        df_index = df.index
        signals_data = {}
        for signal_name in required_signals:
            if signal_name not in df.columns:
                print(f"    -> [筹码情报警告] 方法 '{method_name}' 缺少DataFrame数据 '{signal_name}'，使用默认值 0.0。")
                signals_data[signal_name] = pd.Series(0.0, index=df_index)
            else:
                signals_data[signal_name] = df[signal_name]
        return signals_data

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
        - 新增功能: 整合5个新的筹码信号诊断方法，并为所有核心信号添加校验打印。
        """
        print("启动【V19.0 · 诡道反吸版】筹码情报分析...")
        all_chip_states = {}
        periods = [5, 13, 21, 55]
        # 借用行为层的MTF权重配置
        p_behavior_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_behavior_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        # 调用并记录持仓信念韧性信号
        holder_sentiment_scores = self._diagnose_axiom_holder_sentiment(df, periods)
        all_chip_states['SCORE_CHIP_AXIOM_HOLDER_SENTIMENT'] = holder_sentiment_scores
        print(f"    -> [筹码情报校验] 计算“持仓信念韧性(SCORE_CHIP_AXIOM_HOLDER_SENTIMENT)” 分数：{holder_sentiment_scores.mean():.4f}")
        # 调用并记录价筹张力信号
        divergence_scores = self._diagnose_axiom_divergence(df, periods)
        all_chip_states['SCORE_CHIP_AXIOM_DIVERGENCE'] = divergence_scores
        print(f"    -> [筹码情报校验] 计算“价筹张力(SCORE_CHIP_AXIOM_DIVERGENCE)” 分数：{divergence_scores.mean():.4f}")
        # 调用并记录战略态势信号
        strategic_posture = self._diagnose_strategic_posture(df)
        all_chip_states['SCORE_CHIP_STRATEGIC_POSTURE'] = strategic_posture
        print(f"    -> [筹码情报校验] 计算“战略态势(SCORE_CHIP_STRATEGIC_POSTURE)” 分数：{strategic_posture.mean():.4f}")
        # 调用并记录战场地形信号
        battlefield_geography = self._diagnose_battlefield_geography(df)
        all_chip_states['SCORE_CHIP_BATTLEFIELD_GEOGRAPHY'] = battlefield_geography
        print(f"    -> [筹码情报校验] 计算“战场地形(SCORE_CHIP_BATTLEFIELD_GEOGRAPHY)” 分数：{battlefield_geography.mean():.4f}")
        # 调用并记录筹码趋势动量信号
        chip_trend_momentum_scores = self._diagnose_axiom_trend_momentum(df, periods, strategic_posture, battlefield_geography, holder_sentiment_scores)
        all_chip_states['SCORE_CHIP_AXIOM_TREND_MOMENTUM'] = chip_trend_momentum_scores
        print(f"    -> [筹码情报校验] 计算“筹码趋势动量(SCORE_CHIP_AXIOM_TREND_MOMENTUM)” 分数：{chip_trend_momentum_scores.mean():.4f}")
        # 调用并记录筹码历史潜力信号
        historical_potential = self._diagnose_axiom_historical_potential(df)
        all_chip_states['SCORE_CHIP_AXIOM_HISTORICAL_POTENTIAL'] = historical_potential
        print(f"    -> [筹码情报校验] 计算“筹码历史潜力(SCORE_CHIP_AXIOM_HISTORICAL_POTENTIAL)” 分数：{historical_potential.mean():.4f}")
        # 调用并记录吸筹回声信号
        absorption_echo = self._diagnose_absorption_echo(df, divergence_scores)
        all_chip_states['SCORE_CHIP_OPP_ABSORPTION_ECHO'] = absorption_echo
        print(f"    -> [筹码情报校验] 计算“吸筹回声(SCORE_CHIP_OPP_ABSORPTION_ECHO)” 分数：{absorption_echo.mean():.4f}")
        # 调用并记录派发诡影信号
        distribution_whisper = self._diagnose_distribution_whisper(df, divergence_scores)
        all_chip_states['SCORE_CHIP_RISK_DISTRIBUTION_WHISPER'] = distribution_whisper
        print(f"    -> [筹码情报校验] 计算“派发诡影(SCORE_CHIP_RISK_DISTRIBUTION_WHISPER)” 分数：{distribution_whisper.mean():.4f}")
        # 调用并记录筹码一致驱动信号
        coherent_drive = self._diagnose_structural_consensus(df, battlefield_geography, holder_sentiment_scores)
        all_chip_states['SCORE_CHIP_COHERENT_DRIVE'] = coherent_drive
        print(f"    -> [筹码情报校验] 计算“筹码一致驱动(SCORE_CHIP_COHERENT_DRIVE)” 分数：{coherent_drive.mean():.4f}")
        # 调用并记录战术换手博弈信号
        tactical_exchange = self._diagnose_tactical_exchange(df, battlefield_geography)
        all_chip_states['SCORE_CHIP_TACTICAL_EXCHANGE'] = tactical_exchange
        print(f"    -> [筹码情报校验] 计算“战术换手博弈(SCORE_CHIP_TACTICAL_EXCHANGE)” 分数：{tactical_exchange.mean():.4f}")
        # 调用并记录战略战术和谐度信号
        strategic_tactical_harmony = self._diagnose_strategic_tactical_harmony(df, strategic_posture, tactical_exchange, holder_sentiment_scores)
        all_chip_states['SCORE_CHIP_STRATEGIC_TACTICAL_HARMONY'] = strategic_tactical_harmony
        print(f"    -> [筹码情报校验] 计算“战略战术和谐度(SCORE_CHIP_STRATEGIC_TACTICAL_HARMONY)” 分数：{strategic_tactical_harmony.mean():.4f}")
        # 调用并记录和谐拐点信号
        harmony_inflection = self._diagnose_harmony_inflection(df, strategic_tactical_harmony)
        all_chip_states['SCORE_CHIP_HARMONY_INFLECTION'] = harmony_inflection
        print(f"    -> [筹码情报校验] 计算“和谐拐点(SCORE_CHIP_HARMONY_INFLECTION)” 分数：{harmony_inflection.mean():.4f}")
        # --- 调用新的筹码信号诊断方法 ---
        # 调用并记录散户筹码脆弱性指数信号
        retail_vulnerability = self._diagnose_chip_retail_vulnerability(df)
        all_chip_states['SCORE_CHIP_RETAIL_VULNERABILITY'] = retail_vulnerability
        print(f"    -> [筹码情报校验] 计算“散户筹码脆弱性指数(SCORE_CHIP_RETAIL_VULNERABILITY)” 分数：{retail_vulnerability.mean():.4f}")
        # 调用并记录主力成本区攻防意图信号
        main_force_cost_intent = self._diagnose_chip_main_force_cost_intent(df)
        all_chip_states['SCORE_CHIP_MAIN_FORCE_COST_INTENT'] = main_force_cost_intent
        print(f"    -> [筹码情报校验] 计算“主力成本区攻防意图(SCORE_CHIP_MAIN_FORCE_COST_INTENT)” 分数：{main_force_cost_intent.mean():.4f}")
        # 调用并记录筹码空心化风险信号
        hollowing_out_risk = self._diagnose_chip_hollowing_out_risk(df)
        all_chip_states['SCORE_CHIP_HOLLOWING_OUT_RISK'] = hollowing_out_risk
        print(f"    -> [筹码情报校验] 计算“筹码空心化风险(SCORE_CHIP_HOLLOWING_OUT_RISK)” 分数：{hollowing_out_risk.mean():.4f}")
        # 调用并记录换手纯度与成本优化信号
        turnover_purity_cost_optimization = self._diagnose_chip_turnover_purity_cost_optimization(df)
        all_chip_states['SCORE_CHIP_TURNOVER_PURITY_COST_OPTIMIZATION'] = turnover_purity_cost_optimization
        print(f"    -> [筹码情报校验] 计算“换手纯度与成本优化(SCORE_CHIP_TURNOVER_PURITY_COST_OPTIMIZATION)” 分数：{turnover_purity_cost_optimization.mean():.4f}")
        # 调用并记录筹码绝望与诱惑区信号
        despair_temptation_zones = self._diagnose_chip_despair_temptation_zones(df)
        all_chip_states['SCORE_CHIP_DESPAIR_TEMPTATION_ZONES'] = despair_temptation_zones
        print(f"    -> [筹码情报校验] 计算“筹码绝望与诱惑区(SCORE_CHIP_DESPAIR_TEMPTATION_ZONES)” 分数：{despair_temptation_zones.mean():.4f}")
        # 更新最终生成的筹码原子信号数量
        print(f"【V19.0 · 诡道反吸版】分析完成，生成 {len(all_chip_states)} 个筹码原子信号。")
        return all_chip_states

    def _diagnose_strategic_posture(self, df: pd.DataFrame) -> pd.Series:
        """
        【V9.1 · 诡道情境自适应版】诊断主力的综合战略态势。
        - 核心升级1: 诡道博弈深度融合与情境调制：引入主力信念和筹码健康度作为情境，动态调整欺骗指数和对倒强度的影响，实现非对称调制，更精准识别和应对主力诡道博弈。
        - 核心升级2: 动态权重自适应：根据筹码波动不稳定性、筹码健康度斜率等情境因子，动态调整基础态势、速度和加速度的融合权重，使信号自适应市场动态。
        - 核心升级3: 维度间非线性互动增强：引入“协同/冲突”因子，评估阵型部署、指挥官决心、战场控制各维度之间的非线性互动，提高信号的敏感性和准确性。
        - 核心升级4: 全局情境调制器：引入筹码健康度、市场情绪作为全局调制器，对最终战略态态势分数进行校准，提高信号在不同市场情境下的可靠性。
        - 核心升级5: 新增筹码指标整合：
            - 诱多/诱空欺骗强度 (`deception_lure_long_intensity_D`, `deception_lure_short_intensity_D`) 进一步精细化诡道调制。
            - 主力成本区买卖意图 (`mf_cost_zone_buy_intent_D`, `mf_cost_zone_sell_intent_D`) 增强指挥官决心维度。
            - 隐蔽派发信号 (`covert_distribution_signal_D`) 作为负向调制器。
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        df_index = df.index
        p_conf = self.chip_ultimate_params
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        sp_params = get_param_value(p_conf.get('strategic_posture_params'), {})
        deception_fusion_weights = get_param_value(sp_params.get('deception_fusion_weights'), {"bear_trap_positive": 0.6, "bull_trap_negative": 0.2, "wash_trade_negative": 0.2})
        deception_context_mod_enabled = get_param_value(sp_params.get('deception_context_mod_enabled'), True)
        deception_conviction_threshold = get_param_value(sp_params.get('deception_conviction_threshold'), 0.2)
        deception_health_threshold = get_param_value(sp_params.get('deception_health_threshold'), 0.5)
        deception_boost_factor = get_param_value(sp_params.get('deception_boost_factor'), 0.5)
        deception_penalty_factor = get_param_value(sp_params.get('deception_penalty_factor'), 0.7)
        wash_trade_penalty_factor = get_param_value(sp_params.get('wash_trade_penalty_factor'), 0.3)
        deception_lure_long_penalty_factor = get_param_value(sp_params.get('deception_lure_long_penalty_factor'), 0.3)
        deception_lure_short_boost_factor = get_param_value(sp_params.get('deception_lure_short_boost_factor'), 0.3)
        mf_cost_zone_buy_intent_weight = get_param_value(sp_params.get('mf_cost_zone_buy_intent_weight'), 0.1)
        mf_cost_zone_sell_intent_weight = get_param_value(sp_params.get('mf_cost_zone_sell_intent_weight'), 0.1)
        covert_distribution_penalty_factor = get_param_value(sp_params.get('covert_distribution_penalty_factor'), 0.2)
        dynamic_fusion_weights_base = get_param_value(sp_params.get('dynamic_fusion_weights_base'), {'base_score': 0.6, 'velocity': 0.2, 'acceleration': 0.2})
        dynamic_weight_modulator_signal_1_name = get_param_value(sp_params.get('dynamic_weight_modulator_signal_1'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        dynamic_weight_modulator_signal_2_name = get_param_value(sp_params.get('dynamic_weight_modulator_signal_2'), 'SLOPE_5_chip_health_score_D')
        dynamic_weight_sensitivity_volatility = get_param_value(sp_params.get('dynamic_weight_sensitivity_volatility'), 0.4)
        dynamic_weight_sensitivity_health_slope = get_param_value(sp_params.get('dynamic_weight_sensitivity_health_slope'), 0.3)
        inter_dimension_interaction_enabled = get_param_value(sp_params.get('inter_dimension_interaction_enabled'), True)
        synergy_bonus_factor = get_param_value(sp_params.get('synergy_bonus_factor'), 0.15)
        conflict_penalty_factor = get_param_value(sp_params.get('conflict_penalty_factor'), 0.2)
        global_context_modulator_enabled = get_param_value(sp_params.get('global_context_modulator_enabled'), True)
        global_context_signal_1_name = get_param_value(sp_params.get('global_context_signal_1'), 'chip_health_score_D')
        global_context_signal_2_name = get_param_value(sp_params.get('global_context_signal_2'), 'market_sentiment_score_D')
        global_context_sensitivity_health = get_param_value(sp_params.get('global_context_sensitivity_health'), 0.5)
        global_context_sensitivity_sentiment = get_param_value(sp_params.get('global_context_sensitivity_sentiment'), 0.3)
        smoothing_ema_span = get_param_value(sp_params.get('smoothing_ema_span'), 5)
        required_signals = [
            'cost_gini_coefficient_D', 'covert_accumulation_signal_D', 'peak_exchange_purity_D',
            'main_force_cost_advantage_D', 'control_solidity_index_D', 'SLOPE_5_main_force_conviction_index_D',
            'floating_chip_cleansing_efficiency_D', 'dominant_peak_solidity_D',
            'deception_index_D', 'wash_trade_intensity_D',
            'main_force_conviction_index_D', 'chip_health_score_D',
            dynamic_weight_modulator_signal_1_name, dynamic_weight_modulator_signal_2_name,
            global_context_signal_1_name, global_context_signal_2_name,
            'deception_lure_long_intensity_D', 'deception_lure_short_intensity_D',
            'mf_cost_zone_buy_intent_D', 'mf_cost_zone_sell_intent_D', 'covert_distribution_signal_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_strategic_posture"):
            return pd.Series(0.0, index=df.index)
        signals_data = self._get_all_required_signals(df, required_signals, "_diagnose_strategic_posture")
        # --- 调试信息构建 ---
        is_debug_enabled = self.should_probe
        probe_ts = None
        if is_debug_enabled and self.probe_dates_set:
            for date in reversed(df_index):
                if date.date() in self.probe_dates_set:
                    probe_ts = date
                    break
        debug_info_tuple = False # (is_debug_enabled, probe_ts, "_diagnose_strategic_posture")
        # --- 原始数据获取 ---
        cost_gini_coefficient_raw = signals_data['cost_gini_coefficient_D']
        covert_accumulation_raw = signals_data['covert_accumulation_signal_D']
        peak_exchange_purity_raw = signals_data['peak_exchange_purity_D']
        main_force_cost_advantage_raw = signals_data['main_force_cost_advantage_D']
        control_solidity_index_raw = signals_data['control_solidity_index_D']
        conviction_slope_raw = signals_data['SLOPE_5_main_force_conviction_index_D']
        deception_index_raw = signals_data['deception_index_D']
        wash_trade_intensity_raw = signals_data['wash_trade_intensity_D']
        cleansing_efficiency_raw = signals_data['floating_chip_cleansing_efficiency_D']
        dominant_peak_solidity_raw = signals_data['dominant_peak_solidity_D']
        main_force_conviction_raw = signals_data['main_force_conviction_index_D']
        chip_health_raw = signals_data['chip_health_score_D']
        volatility_instability_raw = signals_data[dynamic_weight_modulator_signal_1_name]
        chip_health_slope_raw = signals_data[dynamic_weight_modulator_signal_2_name]
        market_sentiment_raw = signals_data[global_context_signal_2_name]
        deception_lure_long_intensity_raw = signals_data['deception_lure_long_intensity_D']
        deception_lure_short_intensity_raw = signals_data['deception_lure_short_intensity_D']
        mf_cost_zone_buy_intent_raw = signals_data['mf_cost_zone_buy_intent_D']
        mf_cost_zone_sell_intent_raw = signals_data['mf_cost_zone_sell_intent_D']
        covert_distribution_signal_raw = signals_data['covert_distribution_signal_D']
        # --- 1. 阵型部署 (Formation Deployment) ---
        concentration_level = 1 - cost_gini_coefficient_raw
        level_score = get_adaptive_mtf_normalized_bipolar_score(concentration_level, df_index, tf_weights)
        norm_covert_accumulation = get_adaptive_mtf_normalized_bipolar_score(covert_accumulation_raw, df_index, tf_weights)
        norm_peak_exchange_purity = get_adaptive_mtf_normalized_bipolar_score(peak_exchange_purity_raw, df_index, tf_weights)
        efficiency_score = (
            (norm_covert_accumulation.add(1)/2) *
            (norm_peak_exchange_purity.add(1)/2)
        ).pow(0.5) * 2 - 1
        formation_deployment_score = ((level_score.add(1)/2) * (efficiency_score.add(1)/2)).pow(0.5) * 2 - 1
        # --- 2. 指挥官决心 (Commanders Resolve) ---
        advantage_score = get_adaptive_mtf_normalized_bipolar_score(main_force_cost_advantage_raw, df_index, tf_weights)
        solidity_score = get_adaptive_mtf_normalized_bipolar_score(control_solidity_index_raw, df_index, tf_weights)
        intent_score = get_adaptive_mtf_normalized_bipolar_score(conviction_slope_raw, df_index, tf_weights)
        norm_mf_cost_zone_buy_intent = get_adaptive_mtf_normalized_score(mf_cost_zone_buy_intent_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_mf_cost_zone_sell_intent = get_adaptive_mtf_normalized_score(mf_cost_zone_sell_intent_raw, df_index, ascending=True, tf_weights=tf_weights)
        commanders_resolve_score = (
            (advantage_score.add(1)/2) * (solidity_score.add(1)/2) *
            (intent_score.clip(lower=-1, upper=1).add(1)/2)
        ).pow(1/3) * 2 - 1
        commanders_resolve_score = commanders_resolve_score + \
                                   (norm_mf_cost_zone_buy_intent * mf_cost_zone_buy_intent_weight) - \
                                   (norm_mf_cost_zone_sell_intent * mf_cost_zone_sell_intent_weight)
        commanders_resolve_score = commanders_resolve_score.clip(-1, 1)
        # --- 诡道情境调制 (Deception Context Modulation) ---
        norm_deception_index = get_adaptive_mtf_normalized_bipolar_score(deception_index_raw, df_index, tf_weights)
        norm_wash_trade_intensity = get_adaptive_mtf_normalized_score(wash_trade_intensity_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_main_force_conviction = get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, tf_weights)
        norm_chip_health = get_adaptive_mtf_normalized_score(chip_health_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_deception_lure_long = get_adaptive_mtf_normalized_score(deception_lure_long_intensity_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_deception_lure_short = get_adaptive_mtf_normalized_score(deception_lure_short_intensity_raw, df_index, ascending=True, tf_weights=tf_weights)
        deception_modulator = pd.Series(1.0, index=df_index)
        if deception_context_mod_enabled:
            strong_conviction_healthy_chip_mask = (norm_main_force_conviction > deception_conviction_threshold) & \
                                                  (norm_chip_health > deception_health_threshold)
            weak_conviction_unhealthy_chip_mask = (norm_main_force_conviction < -deception_conviction_threshold) | \
                                                   (norm_chip_health < (1 - deception_health_threshold))
            # 诱空增强 (主力信念强，筹码健康，且有诱空信号或负向欺骗)
            bear_trap_boost_mask = strong_conviction_healthy_chip_mask & ((norm_deception_index < 0) | (norm_deception_lure_short > 0))
            deception_modulator.loc[bear_trap_boost_mask] = deception_modulator.loc[bear_trap_boost_mask] * (1 + (norm_deception_index.loc[bear_trap_boost_mask].abs() * deception_boost_factor + \
                                                                 norm_deception_lure_short.loc[bear_trap_boost_mask] * deception_lure_short_boost_factor))
            # 诱多惩罚 (有诱多信号或正向欺骗)
            bull_trap_penalty_mask = (norm_deception_index > 0) | (norm_deception_lure_long > 0)
            deception_modulator.loc[bull_trap_penalty_mask] = deception_modulator.loc[bull_trap_penalty_mask] * (1 - (norm_deception_index.loc[bull_trap_penalty_mask].clip(lower=0) * deception_penalty_factor + \
                                                                   norm_deception_lure_long.loc[bull_trap_penalty_mask] * deception_lure_long_penalty_factor))
            # 对倒惩罚
            wash_trade_penalty_mod = norm_wash_trade_intensity * wash_trade_penalty_factor
            # 主力强时，对倒惩罚减半 (可能为洗盘)
            deception_modulator.loc[strong_conviction_healthy_chip_mask] = \
                deception_modulator.loc[strong_conviction_healthy_chip_mask] * (1 - wash_trade_penalty_mod.loc[strong_conviction_healthy_chip_mask] * 0.5)
            # 主力弱时，对倒惩罚加倍 (可能为出货)
            deception_modulator.loc[weak_conviction_unhealthy_chip_mask] = \
                deception_modulator.loc[weak_conviction_unhealthy_chip_mask] * (1 - wash_trade_penalty_mod.loc[weak_conviction_unhealthy_chip_mask] * 1.5)
            # 其他情况正常惩罚
            deception_modulator.loc[~(strong_conviction_healthy_chip_mask | weak_conviction_unhealthy_chip_mask)] = \
                deception_modulator.loc[~(strong_conviction_healthy_chip_mask | weak_conviction_unhealthy_chip_mask)] * (1 - wash_trade_penalty_mod.loc[~(strong_conviction_healthy_chip_mask | weak_conviction_unhealthy_chip_mask)])
            deception_modulator = deception_modulator.clip(0.1, 2.0)
        commanders_resolve_score = commanders_resolve_score * deception_modulator.pow(np.sign(commanders_resolve_score))
        # --- 3. 战场控制 (Battlefield Control) ---
        cleansing_score = get_adaptive_mtf_normalized_bipolar_score(cleansing_efficiency_raw, df_index, tf_weights)
        peak_solidity_score = get_adaptive_mtf_normalized_bipolar_score(dominant_peak_solidity_raw, df_index, tf_weights)
        battlefield_control_score = ((cleansing_score.add(1)/2) * (peak_solidity_score.add(1)/2)).pow(0.5) * 2 - 1
        # --- 基础战略态势融合 ---
        base_strategic_posture_score = (
            (commanders_resolve_score.add(1)/2).pow(0.5) *
            (formation_deployment_score.add(1)/2).pow(0.3) *
            (battlefield_control_score.add(1)/2).pow(0.2)
        ).pow(1/(0.5+0.3+0.2)) * 2 - 1
        # --- 隐蔽派发信号惩罚 ---
        norm_covert_distribution_signal = get_adaptive_mtf_normalized_score(covert_distribution_signal_raw, df_index, ascending=True, tf_weights=tf_weights)
        base_strategic_posture_score = base_strategic_posture_score * (1 - norm_covert_distribution_signal * covert_distribution_penalty_factor)
        base_strategic_posture_score = base_strategic_posture_score.clip(-1, 1)
        # --- 维度间非线性互动增强 (Inter-Dimension Interaction) ---
        if inter_dimension_interaction_enabled:
            synergy_factor = pd.Series(0.0, index=df_index)
            # 正向协同：所有维度都为正
            positive_synergy_mask = (formation_deployment_score > 0) & (commanders_resolve_score > 0) & (battlefield_control_score > 0)
            synergy_factor.loc[positive_synergy_mask] = synergy_bonus_factor
            # 负向协同：所有维度都为负
            negative_synergy_mask = (formation_deployment_score < 0) & (commanders_resolve_score < 0) & (battlefield_control_score < 0)
            synergy_factor.loc[negative_synergy_mask] = -synergy_bonus_factor
            # 冲突：任意两个维度方向相反
            conflict_mask = ((formation_deployment_score > 0) & (commanders_resolve_score < 0)) | \
                            ((formation_deployment_score < 0) & (commanders_resolve_score > 0)) | \
                            ((battlefield_control_score > 0) & (commanders_resolve_score < 0)) | \
                            ((battlefield_control_score < 0) & (commanders_resolve_score > 0)) | \
                            ((formation_deployment_score > 0) & (battlefield_control_score < 0)) | \
                            ((formation_deployment_score < 0) & (battlefield_control_score > 0))
            synergy_factor.loc[conflict_mask] = -conflict_penalty_factor
            # 将协同/冲突因子非线性地作用于基础分数
            base_strategic_posture_score = np.tanh(base_strategic_posture_score + synergy_factor)
        # --- 动态权重自适应 (Dynamic Weight Adaptation) ---
        smoothed_base_score = base_strategic_posture_score.ewm(span=smoothing_ema_span, adjust=False).mean()
        velocity = smoothed_base_score.diff(1).fillna(0)
        acceleration = velocity.diff(1).fillna(0)
        norm_velocity = get_adaptive_mtf_normalized_bipolar_score(velocity, df_index, tf_weights)
        norm_acceleration = get_adaptive_mtf_normalized_bipolar_score(acceleration, df_index, tf_weights)
        dynamic_base_weight = pd.Series(dynamic_fusion_weights_base.get('base_score', 0.6), index=df_index)
        dynamic_velocity_weight = pd.Series(dynamic_fusion_weights_base.get('velocity', 0.2), index=df_index)
        dynamic_acceleration_weight = pd.Series(dynamic_fusion_weights_base.get('acceleration', 0.2), index=df_index)
        norm_volatility_instability = get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_chip_health_slope = get_adaptive_mtf_normalized_bipolar_score(chip_health_slope_raw, df_index, tf_weights)
        mod_factor = (norm_volatility_instability * dynamic_weight_sensitivity_volatility) - \
                     (norm_chip_health_slope.clip(upper=0).abs() * dynamic_weight_sensitivity_health_slope) # 筹码健康度斜率负向时，增强动态权重
        dynamic_base_weight = dynamic_base_weight * (1 - mod_factor)
        dynamic_velocity_weight = dynamic_velocity_weight * (1 + mod_factor * 0.5)
        dynamic_acceleration_weight = dynamic_acceleration_weight * (1 + mod_factor * 0.5)
        sum_dynamic_weights = dynamic_base_weight + dynamic_velocity_weight + dynamic_acceleration_weight
        dynamic_base_weight = dynamic_base_weight / sum_dynamic_weights
        dynamic_velocity_weight = dynamic_velocity_weight / sum_dynamic_weights
        dynamic_acceleration_weight = dynamic_acceleration_weight / sum_dynamic_weights
        final_score_unmodulated = (
            (base_strategic_posture_score.add(1)/2).pow(dynamic_base_weight) *
            (norm_velocity.add(1)/2).pow(dynamic_velocity_weight) *
            (norm_acceleration.add(1)/2).pow(dynamic_acceleration_weight)
        ).pow(1 / (dynamic_base_weight + dynamic_velocity_weight + dynamic_acceleration_weight)) * 2 - 1
        # --- 全局情境调制器 (Global Context Modulator) ---
        final_score = final_score_unmodulated
        if global_context_modulator_enabled:
            norm_global_chip_health = get_adaptive_mtf_normalized_score(chip_health_raw, df_index, ascending=True, tf_weights=tf_weights)
            norm_market_sentiment = get_adaptive_mtf_normalized_score(market_sentiment_raw, df_index, ascending=True, tf_weights=tf_weights) # 市场情绪正向归一化
            global_modulator_effect = (
                (1 + norm_global_chip_health * global_context_sensitivity_health) *
                (1 + norm_market_sentiment * global_context_sensitivity_sentiment)
            ).clip(0.5, 1.5) # 限制调制范围，防止过度放大或缩小
            final_score = final_score * global_modulator_effect
        final_score = final_score.clip(-1, 1).fillna(0.0).astype(np.float32)
        print(f"    -> [筹码层] 计算完成 '筹码最终融合' 分数: {final_score.iloc[-1]}")
        return final_score

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
        df_index = df.index
        p_conf = self.chip_ultimate_params
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        bg_params = get_param_value(p_conf.get('battlefield_geography_params'), {})
        path_efficiency_mod_factor = get_param_value(bg_params.get('path_efficiency_mod_factor'), 0.5)
        path_efficiency_non_linear_exponent = get_param_value(bg_params.get('path_efficiency_non_linear_exponent'), 1.5)
        dynamic_evolution_mod_factor = get_param_value(bg_params.get('dynamic_evolution_mod_factor'), 0.3)
        dynamic_evolution_non_linear_exponent = get_param_value(bg_params.get('dynamic_evolution_non_linear_exponent'), 1.2)
        deception_signal_name = get_param_value(bg_params.get('deception_signal'), 'deception_index_D')
        chip_fault_signal_name = get_param_value(bg_params.get('chip_fault_signal'), 'chip_fault_magnitude_D')
        deception_penalty_sensitivity = get_param_value(bg_params.get('deception_penalty_sensitivity'), 0.6)
        chip_fault_penalty_sensitivity = get_param_value(bg_params.get('chip_fault_penalty_sensitivity'), 0.4)
        deception_mitigation_sensitivity = get_param_value(bg_params.get('deception_mitigation_sensitivity'), 0.3)
        context_modulator_signal_1_name = get_param_value(bg_params.get('context_modulator_signal_1'), 'chip_health_score_D')
        context_modulator_signal_2_name = get_param_value(bg_params.get('context_modulator_signal_2'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        context_modulator_sensitivity_health = get_param_value(bg_params.get('context_modulator_sensitivity_health'), 0.4)
        context_modulator_sensitivity_volatility = get_param_value(bg_params.get('context_modulator_sensitivity_volatility'), 0.3)
        upward_impulse_strength_weight = get_param_value(bg_params.get('upward_impulse_strength_weight'), 0.1)
        downward_impulse_strength_weight = get_param_value(bg_params.get('downward_impulse_strength_weight'), 0.1)
        mf_cost_zone_buy_intent_weight = get_param_value(bg_params.get('mf_cost_zone_buy_intent_weight'), 0.1)
        mf_cost_zone_sell_intent_weight = get_param_value(bg_params.get('mf_cost_zone_sell_intent_weight'), 0.1)
        required_signals = [
            'dominant_peak_solidity_D', 'support_validation_strength_D', 'chip_fault_blockage_ratio_D',
            'pressure_rejection_strength_D', 'vacuum_zone_magnitude_D', 'vacuum_traversal_efficiency_D',
            'SLOPE_5_support_validation_strength_D', 'SLOPE_5_pressure_rejection_strength_D',
            deception_signal_name, chip_fault_signal_name,
            context_modulator_signal_1_name, context_modulator_signal_2_name,
            'upward_impulse_strength_D', 'downward_impulse_strength_D',
            'mf_cost_zone_buy_intent_D', 'mf_cost_zone_sell_intent_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_battlefield_geography"):
            return pd.Series(0.0, index=df.index)
        signals_data = self._get_all_required_signals(df, required_signals, "_diagnose_battlefield_geography")
        # --- 调试信息构建 ---
        is_debug_enabled = self.should_probe
        probe_ts = None
        if is_debug_enabled and self.probe_dates_set:
            for date in reversed(df_index):
                if date.date() in self.probe_dates_set:
                    probe_ts = date
                    break
        debug_info_tuple = False # (is_debug_enabled, probe_ts, "_diagnose_battlefield_geography")
        # --- 原始数据获取 ---
        peak_solidity = signals_data['dominant_peak_solidity_D']
        support_validation = signals_data['support_validation_strength_D']
        fault_blockage = signals_data['chip_fault_blockage_ratio_D']
        pressure_rejection = signals_data['pressure_rejection_strength_D']
        vacuum_magnitude = signals_data['vacuum_zone_magnitude_D']
        vacuum_efficiency = signals_data['vacuum_traversal_efficiency_D']
        support_trend_raw = signals_data['SLOPE_5_support_validation_strength_D']
        resistance_trend_raw = signals_data['SLOPE_5_pressure_rejection_strength_D']
        deception_raw = signals_data[deception_signal_name]
        chip_fault_raw = signals_data[chip_fault_signal_name]
        chip_health_raw = signals_data[context_modulator_signal_1_name]
        volatility_instability_raw = signals_data[context_modulator_signal_2_name]
        upward_impulse_strength_raw = signals_data['upward_impulse_strength_D']
        downward_impulse_strength_raw = signals_data['downward_impulse_strength_D']
        mf_cost_zone_buy_intent_raw = signals_data['mf_cost_zone_buy_intent_D']
        mf_cost_zone_sell_intent_raw = signals_data['mf_cost_zone_sell_intent_D']
        # --- 1. 支撑强度 (Support Strength) ---
        solidity_score = get_adaptive_mtf_normalized_score(peak_solidity, df_index, tf_weights)
        validation_score = get_adaptive_mtf_normalized_score(support_validation, df_index, tf_weights)
        support_strength_score = (solidity_score * validation_score).pow(0.5)
        norm_upward_impulse_strength = get_adaptive_mtf_normalized_score(upward_impulse_strength_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_mf_cost_zone_buy_intent = get_adaptive_mtf_normalized_score(mf_cost_zone_buy_intent_raw, df_index, ascending=True, tf_weights=tf_weights)
        support_strength_score = support_strength_score * (1 + norm_upward_impulse_strength * upward_impulse_strength_weight + \
                                                              norm_mf_cost_zone_buy_intent * mf_cost_zone_buy_intent_weight)
        support_strength_score = support_strength_score.clip(0, 1)
        # --- 2. 阻力强度 (Resistance Strength) ---
        blockage_score = get_adaptive_mtf_normalized_score(fault_blockage, df_index, tf_weights)
        rejection_score = get_adaptive_mtf_normalized_score(pressure_rejection, df_index, tf_weights)
        resistance_strength_score = (blockage_score * rejection_score).pow(0.5)
        norm_downward_impulse_strength = get_adaptive_mtf_normalized_score(downward_impulse_strength_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_mf_cost_zone_sell_intent = get_adaptive_mtf_normalized_score(mf_cost_zone_sell_intent_raw, df_index, ascending=True, tf_weights=tf_weights)
        resistance_strength_score = resistance_strength_score * (1 + norm_downward_impulse_strength * downward_impulse_strength_weight + \
                                                                    norm_mf_cost_zone_sell_intent * mf_cost_zone_sell_intent_weight)
        resistance_strength_score = resistance_strength_score.clip(0, 1)
        # --- 3. 核心地形优势 (Core Terrain Advantage) ---
        base_terrain_advantage_score = support_strength_score - resistance_strength_score
        # --- 4. 最小阻力路径动态调制 (Minimum Resistance Path Dynamic Modulation) ---
        norm_vacuum_magnitude = get_adaptive_mtf_normalized_score(vacuum_magnitude, df_index, tf_weights)
        norm_vacuum_efficiency = get_adaptive_mtf_normalized_score(vacuum_efficiency, df_index, tf_weights)
        path_efficiency = (norm_vacuum_magnitude * norm_vacuum_efficiency).pow(0.5)
        path_modulation_factor = (1 + path_efficiency * path_efficiency_mod_factor).pow(path_efficiency_non_linear_exponent)
        # --- 5. 动态演化趋势强化 (Dynamic Evolution Trend Reinforcement) ---
        norm_support_trend = get_adaptive_mtf_normalized_bipolar_score(support_trend_raw, df_index, tf_weights)
        norm_resistance_trend = get_adaptive_mtf_normalized_bipolar_score(resistance_trend_raw, df_index, tf_weights)
        terrain_advantage_change = norm_support_trend - norm_resistance_trend
        dynamic_evolution_modulator = (1 + terrain_advantage_change * dynamic_evolution_mod_factor).pow(dynamic_evolution_non_linear_exponent)
        dynamic_evolution_modulator = dynamic_evolution_modulator.clip(0.5, 1.5)
        # --- 6. 诡道地形过滤与惩罚 (Deceptive Terrain Filtering & Penalty) ---
        norm_deception = get_adaptive_mtf_normalized_bipolar_score(deception_raw, df_index, tf_weights)
        norm_chip_fault = get_adaptive_mtf_normalized_bipolar_score(chip_fault_raw, df_index, tf_weights)
        deception_filter_factor = pd.Series(1.0, index=df_index)
        # 有利地形伴随诱多或虚假支撑时惩罚 (base_terrain_advantage_score > 0 且有正向欺骗或正向故障)
        bull_trap_penalty_mask = (base_terrain_advantage_score > 0) & ((norm_deception > 0) | (norm_chip_fault > 0))
        deception_filter_factor.loc[bull_trap_penalty_mask] = \
            1 - ((norm_deception.loc[bull_trap_penalty_mask].clip(lower=0) * deception_penalty_sensitivity) + \
                 (norm_chip_fault.loc[bull_trap_penalty_mask].clip(lower=0) * chip_fault_penalty_sensitivity)).clip(0, 1)
        # 不利地形伴随诱空洗盘或虚假阻力时缓解 (base_terrain_advantage_score < 0 且有负向欺骗或负向故障)
        bear_trap_mitigation_mask = (base_terrain_advantage_score < 0) & ((norm_deception < 0) | (norm_chip_fault < 0))
        deception_filter_factor.loc[bear_trap_mitigation_mask] = \
            1 + ((norm_deception.loc[bear_trap_mitigation_mask].abs().clip(lower=0) * deception_mitigation_sensitivity) + \
                 (norm_chip_fault.loc[bear_trap_mitigation_mask].abs().clip(lower=0) * deception_mitigation_sensitivity)).clip(0, 0.5)
        deception_filter_factor = deception_filter_factor.clip(0.1, 2.0)
        # --- 7. 情境感知与自适应权重 (Context-Aware Adaptive Weighting) ---
        norm_chip_health = get_adaptive_mtf_normalized_score(chip_health_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_volatility_instability = get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=False, tf_weights=tf_weights) # 低波动性 = 高健康度
        context_modulator = (
            (1 + norm_chip_health * context_modulator_sensitivity_health) *
            (1 + norm_volatility_instability * context_modulator_sensitivity_volatility)
        ).clip(0.5, 1.5)
        # --- 最终融合 ---
        final_score = base_terrain_advantage_score * path_modulation_factor * dynamic_evolution_modulator * deception_filter_factor * context_modulator
        final_score = final_score.clip(-1, 1).fillna(0.0).astype(np.float32)
        print(f"    -> [筹码层] 计算完成 '筹码最终融合' 分数: {final_score.iloc[-1]}")
        return final_score

    def _diagnose_axiom_holder_sentiment(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V9.1 · 恐慌动态修正版】筹码公理三：诊断“持仓信念韧性”
        - 核心升级1: 纯筹码指标强化。严格遵循纯筹码原则，将全局市场情绪替换为筹码主力信念，并新增恐慌买入吸收贡献、低吸吸收强度等纯筹码指标。
        - 核心升级2: 诡道反噬机制深化。在“杂质削弱”维度中，引入诱多/诱空欺骗强度，根据主力信念动态放大或削弱杂质影响，实现“诡道反噬”。
        - 核心升级3: 韧性重构机制引入。在“杂质削弱”维度中，引入筹码健康度斜率和结构性紧张指数，动态评估筹码结构在压力下的自我修复或恶化加速，实现“韧性重构”。
        - 核心升级4: 压力测试精细化。在V8.2基础上，新增恐慌买入吸收贡献和低吸吸收强度，更全面评估主力在恐慌和下跌中的承接能力。
        - 核心升级5: 全局情境调制器优化。将全局市场情绪替换为筹码主力信念，使情境调制更聚焦于筹码层面。
        - 核心修复: 修正 `panic_source_score` 对恐慌动态的判断，当散户恐慌快速消退时，降低其贡献。
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        method_name = "_diagnose_axiom_holder_sentiment"
        df_index = df.index
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
            'VOLATILITY_INSTABILITY_INDEX_21d_D', 'flow_credibility_index_D',
            'main_force_conviction_index_D',
            'conviction_flow_buy_intensity_D', 'conviction_flow_sell_intensity_D',
            'deception_lure_long_intensity_D', 'deception_lure_short_intensity_D',
            'panic_buy_absorption_contribution_D', 'dip_buy_absorption_strength_D',
            'structural_tension_index_D', 'SLOPE_5_chip_health_score_D',
            'SLOPE_5_retail_panic_surrender_index_D' # 新增所需信号
        ]
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index)
        signals_data = self._get_all_required_signals(df, required_signals, method_name)
        p_conf = self.chip_ultimate_params
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        holder_sentiment_params = get_param_value(p_conf.get('holder_sentiment_params'), {})
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
        belief_core_weights = {k: v for k, v in get_param_value(holder_sentiment_params.get('belief_core_weights'), {}).items() if isinstance(v, (int, float))}
        pressure_test_weights = {k: v for k, v in get_param_value(holder_sentiment_params.get('pressure_test_weights'), {}).items() if isinstance(v, (int, float))}
        impurity_weights = {k: v for k, v in get_param_value(holder_sentiment_params.get('impurity_weights'), {}).items() if isinstance(v, (int, float))}
        deception_modulator_params = get_param_value(holder_sentiment_params.get('deception_modulator_params'), {'boost_factor': 0.6, 'penalty_factor': 0.4, 'conviction_threshold': 0.2, 'deception_index_weight': 0.5})
        deception_modulator_weights = {k: v for k, v in get_param_value(holder_sentiment_params.get('deception_modulator_weights'), {}).items() if isinstance(v, (int, float))}
        context_modulator_weights = {k: v for k, v in get_param_value(holder_sentiment_params.get('context_modulator_weights'), {}).items() if isinstance(v, (int, float))}
        impurity_deception_mod_enabled = get_param_value(holder_sentiment_params.get('impurity_deception_mod_enabled'), True)
        deception_lure_long_impurity_amp_factor = get_param_value(holder_sentiment_params.get('deception_lure_long_impurity_amp_factor'), 0.3)
        deception_lure_short_impurity_damp_factor = get_param_value(holder_sentiment_params.get('deception_lure_short_impurity_damp_factor'), 0.2)
        impurity_resilience_mod_enabled = get_param_value(holder_sentiment_params.get('impurity_resilience_mod_enabled'), True)
        chip_health_slope_impurity_damp_factor = get_param_value(holder_sentiment_params.get('chip_health_slope_impurity_damp_factor'), 0.2)
        structural_tension_impurity_amp_factor = get_param_value(holder_sentiment_params.get('structural_tension_impurity_amp_factor'), 0.2)
        global_context_modulator_enabled = get_param_value(holder_sentiment_params.get('global_context_modulator_enabled'), True)
        global_context_sensitivity_health = get_param_value(holder_sentiment_params.get('global_context_sensitivity_health'), 0.5)
        global_context_sensitivity_conviction = get_param_value(holder_sentiment_params.get('global_context_sensitivity_conviction'), 0.3)
        # 新增参数
        panic_slope_dampening_enabled = get_param_value(holder_sentiment_params.get('panic_slope_dampening_enabled'), True)
        panic_slope_dampening_sensitivity = get_param_value(holder_sentiment_params.get('panic_slope_dampening_sensitivity'), 0.5)
        # --- 调试信息构建 ---
        is_debug_enabled = self.should_probe
        probe_ts = None
        if is_debug_enabled and self.probe_dates_set:
            for date in reversed(df_index):
                if date.date() in self.probe_dates_set:
                    probe_ts = date
                    break
        debug_info_tuple = (is_debug_enabled, probe_ts, method_name)
        # if is_debug_enabled and probe_ts and probe_ts in df_index:
        #     print(f"      [探针] {method_name} 启动 @ {probe_ts.strftime('%Y-%m-%d')}")
        #     print(f"        - 原始数据 (winner_stability_index_D): {signals_data['winner_stability_index_D'].loc[probe_ts]:.4f}")
        #     print(f"        - 原始数据 (loser_pain_index_D): {signals_data['loser_pain_index_D'].loc[probe_ts]:.4f}")
        #     print(f"        - 原始数据 (chip_health_score_D): {signals_data['chip_health_score_D'].loc[probe_ts]:.4f}")
        #     print(f"        - 原始数据 (deception_index_D): {signals_data['deception_index_D'].loc[probe_ts]:.4f}")
        #     print(f"        - 原始数据 (deception_lure_long_intensity_D): {signals_data['deception_lure_long_intensity_D'].loc[probe_ts]:.4f}")
        #     print(f"        - 原始数据 (deception_lure_short_intensity_D): {signals_data['deception_lure_short_intensity_D'].loc[probe_ts]:.4f}")
        #     print(f"        - 原始数据 (SLOPE_5_chip_health_score_D): {signals_data['SLOPE_5_chip_health_score_D'].loc[probe_ts]:.4f}")
        #     print(f"        - 原始数据 (structural_tension_index_D): {signals_data['structural_tension_index_D'].loc[probe_ts]:.4f}")
        #     print(f"        - 原始数据 (SLOPE_5_retail_panic_surrender_index_D): {signals_data['SLOPE_5_retail_panic_surrender_index_D'].loc[probe_ts]:.4f}") # 新增探针
        # --- 原始数据获取 ---
        chip_health_raw = signals_data['chip_health_score_D']
        winner_stability = signals_data['winner_stability_index_D']
        loser_pain = signals_data['loser_pain_index_D']
        total_winner_rate_raw = signals_data['total_winner_rate_D']
        total_loser_rate_raw = signals_data['total_loser_rate_D']
        winner_loser_momentum_raw = signals_data['winner_loser_momentum_D']
        slope_5_winner_stability_raw = signals_data['SLOPE_5_winner_stability_index_D']
        accel_5_loser_pain_raw = signals_data['ACCEL_5_loser_pain_index_D']
        slope_5_winner_loser_momentum_raw = signals_data['SLOPE_5_winner_loser_momentum_D']
        absorption_power = signals_data['active_buying_support_D']
        defense_intent = signals_data['support_validation_strength_D']
        capitulation_absorption = signals_data['capitulation_absorption_index_D']
        opening_gap_defense_strength_raw = signals_data['opening_gap_defense_strength_D']
        control_solidity_raw = signals_data['control_solidity_index_D']
        order_book_clearing_rate_raw = signals_data['order_book_clearing_rate_D']
        micro_price_impact_asymmetry_raw = signals_data['micro_price_impact_asymmetry_D']
        slope_5_support_validation_raw = signals_data['SLOPE_5_support_validation_strength_D']
        accel_5_capitulation_absorption_raw = signals_data['ACCEL_5_capitulation_absorption_index_D']
        slope_5_active_buying_support_raw = signals_data['SLOPE_5_active_buying_support_D']
        fomo_index_raw = signals_data['winner_concentration_90pct_D']
        profit_taking_quality_raw = signals_data['winner_profit_margin_avg_D']
        upper_shadow_selling_pressure_raw = signals_data['upper_shadow_selling_pressure_D']
        rally_distribution_pressure_raw = signals_data['rally_distribution_pressure_D']
        retail_fomo_premium_raw = signals_data['retail_fomo_premium_index_D']
        slope_5_winner_profit_margin_raw = signals_data['SLOPE_5_winner_profit_margin_avg_D']
        accel_5_retail_fomo_premium_raw = signals_data['ACCEL_5_retail_fomo_premium_index_D']
        chip_fatigue_raw = signals_data['chip_fatigue_index_D']
        deception_raw = signals_data[deception_signal_name]
        deception_index_raw = signals_data['deception_index_D']
        wash_trade_intensity_raw = signals_data['wash_trade_intensity_D']
        main_force_conviction_raw = signals_data['main_force_conviction_index_D']
        volatility_instability_raw = signals_data['VOLATILITY_INSTABILITY_INDEX_21d_D']
        flow_credibility_raw = signals_data['flow_credibility_index_D']
        conviction_flow_buy_intensity_raw = signals_data['conviction_flow_buy_intensity_D']
        conviction_flow_sell_intensity_raw = signals_data['conviction_flow_sell_intensity_D']
        deception_lure_long_intensity_raw = signals_data['deception_lure_long_intensity_D']
        deception_lure_short_intensity_raw = signals_data['deception_lure_short_intensity_D']
        panic_buy_absorption_contribution_raw = signals_data['panic_buy_absorption_contribution_D']
        dip_buy_absorption_strength_raw = signals_data['dip_buy_absorption_strength_D']
        structural_tension_raw = signals_data['structural_tension_index_D']
        slope_5_chip_health_raw = signals_data['SLOPE_5_chip_health_score_D']
        slope_5_retail_panic_surrender_raw = signals_data['SLOPE_5_retail_panic_surrender_index_D'] # 获取新增信号
        # --- 1. 信念核心 (Belief Core) ---
        norm_winner_stability = get_adaptive_mtf_normalized_bipolar_score(winner_stability, df_index, tf_weights)
        norm_loser_pain = get_adaptive_mtf_normalized_bipolar_score(loser_pain, df_index, tf_weights)
        norm_total_winner_rate = get_adaptive_mtf_normalized_score(total_winner_rate_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_total_loser_rate = get_adaptive_mtf_normalized_score(total_loser_rate_raw, df_index, ascending=False, tf_weights=tf_weights)
        norm_winner_loser_momentum = get_adaptive_mtf_normalized_bipolar_score(winner_loser_momentum_raw, df_index, tf_weights)
        norm_slope_5_winner_stability = get_adaptive_mtf_normalized_bipolar_score(slope_5_winner_stability_raw, df_index, tf_weights)
        norm_accel_5_loser_pain = get_adaptive_mtf_normalized_bipolar_score(accel_5_loser_pain_raw, df_index, tf_weights)
        norm_slope_5_winner_loser_momentum = get_adaptive_mtf_normalized_bipolar_score(slope_5_winner_loser_momentum_raw, df_index, tf_weights)
        norm_conviction_flow_buy_intensity = get_adaptive_mtf_normalized_score(conviction_flow_buy_intensity_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_conviction_flow_sell_intensity = get_adaptive_mtf_normalized_score(conviction_flow_sell_intensity_raw, df_index, ascending=True, tf_weights=tf_weights)
        sentiment_trend_raw = signals_data[sentiment_trend_modulator_signal_name]
        normalized_sentiment_trend = get_adaptive_mtf_normalized_score(sentiment_trend_raw, df_index, tf_weights=tf_weights, ascending=True)
        x = (normalized_sentiment_trend * sentiment_trend_mod_factor).clip(-0.4, 0.4)
        dynamic_stability_weight = 0.5 + x
        dynamic_pain_weight = 0.5 - x
        belief_core_numeric_weights = {k: v for k, v in belief_core_weights.items() if isinstance(v, (int, float))}
        total_belief_core_weight = sum(belief_core_numeric_weights.values())
        # 使用 _robust_geometric_mean 进行融合
        belief_core_components = {
            'winner_stability': (norm_winner_stability + 1) / 2,
            'loser_pain': (norm_loser_pain + 1) / 2,
            'total_winner_rate': norm_total_winner_rate,
            'total_loser_rate': (1 - norm_total_loser_rate), # 负向指标，反转为正向健康度
            'winner_loser_momentum': (norm_winner_loser_momentum + 1) / 2,
            'winner_stability_slope': (norm_slope_5_winner_stability + 1) / 2,
            'loser_pain_accel': (norm_accel_5_loser_pain + 1) / 2,
            'winner_loser_momentum_slope': (norm_slope_5_winner_loser_momentum + 1) / 2,
            'conviction_flow_buy': norm_conviction_flow_buy_intensity,
            'conviction_flow_sell': (1 - norm_conviction_flow_sell_intensity) # 负向指标，反转为正向健康度
        }
        belief_core_component_weights = {
            'winner_stability': belief_core_numeric_weights.get('winner_stability', 0.15) * dynamic_stability_weight,
            'loser_pain': belief_core_numeric_weights.get('loser_pain', 0.15) * dynamic_pain_weight,
            'total_winner_rate': belief_core_numeric_weights.get('total_winner_rate', 0.08),
            'total_loser_rate': belief_core_numeric_weights.get('total_loser_rate', 0.08),
            'winner_loser_momentum': belief_core_numeric_weights.get('winner_loser_momentum', 0.08),
            'winner_stability_slope': belief_core_numeric_weights.get('winner_stability_slope', 0.08),
            'loser_pain_accel': belief_core_numeric_weights.get('loser_pain_accel', 0.08),
            'winner_loser_momentum_slope': belief_core_numeric_weights.get('winner_loser_momentum_slope', 0.08),
            'conviction_flow_buy': belief_core_numeric_weights.get('conviction_flow_buy', 0.1),
            'conviction_flow_sell': belief_core_numeric_weights.get('conviction_flow_sell', 0.1)
        }
        belief_core_score_unipolar = _robust_geometric_mean(belief_core_components, belief_core_component_weights, df_index)
        belief_core_score = (belief_core_score_unipolar * 2 - 1).clip(-1, 1)
        # if is_debug_enabled and probe_ts and probe_ts in df_index:
        #     print(f"        - 中间节点 (norm_winner_stability): {norm_winner_stability.loc[probe_ts]:.4f}")
        #     print(f"        - 中间节点 (norm_loser_pain): {norm_loser_pain.loc[probe_ts]:.4f}")
        #     print(f"        - 中间节点 (norm_accel_5_loser_pain): {norm_accel_5_loser_pain.loc[probe_ts]:.4f}")
        #     print(f"        - 中间节点 (belief_core_score): {belief_core_score.loc[probe_ts]:.4f}")
        # --- 2. 压力测试 (Pressure Test) ---
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
        norm_panic_buy_absorption_contribution = get_adaptive_mtf_normalized_score(panic_buy_absorption_contribution_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_dip_buy_absorption_strength = get_adaptive_mtf_normalized_score(dip_buy_absorption_strength_raw, df_index, ascending=True, tf_weights=tf_weights)
        base_pressure_test_numeric_weights = {k: v for k, v in pressure_test_weights.items() if isinstance(v, (int, float))}
        total_base_pressure_test_weight = sum(base_pressure_test_numeric_weights.values())
        pressure_test_components = {
            'absorption_power': (norm_absorption_power + 1) / 2,
            'defense_intent': (norm_defense_intent + 1) / 2,
            'capitulation_absorption': norm_capitulation_absorption,
            'opening_gap_defense_strength': norm_opening_gap_defense_strength,
            'control_solidity': norm_control_solidity,
            'order_book_clearing_rate': norm_order_book_clearing_rate,
            'micro_price_impact_asymmetry': norm_micro_price_impact_asymmetry,
            'support_validation_slope': norm_slope_5_support_validation,
            'capitulation_absorption_accel': norm_accel_5_capitulation_absorption,
            'active_buying_support_slope': norm_slope_5_active_buying_support,
            'panic_buy_absorption_contribution': norm_panic_buy_absorption_contribution,
            'dip_buy_absorption_strength': norm_dip_buy_absorption_strength
        }
        pressure_test_score_unipolar = _robust_geometric_mean(pressure_test_components, base_pressure_test_numeric_weights, df_index)
        base_pressure_score = (pressure_test_score_unipolar * 2 - 1).clip(-1, 1)
        panic_modulator_raw = signals_data[panic_reward_modulator_signal_name]
        normalized_panic_modulator = get_adaptive_mtf_normalized_score(panic_modulator_raw, df_index, tf_weights=tf_weights, ascending=True)
        panic_reward_adjustment_factor = np.tanh(normalized_panic_modulator * panic_reward_mod_tanh_factor) * panic_reward_mod_factor
        dynamic_capitulation_reward_multiplier = capitulation_base_reward_multiplier * (1 + panic_reward_adjustment_factor)
        dynamic_capitulation_reward_multiplier = dynamic_capitulation_reward_multiplier.clip(0.1, 0.8)
        capitulation_bonus = norm_capitulation_absorption * dynamic_capitulation_reward_multiplier
        deception_impact = pd.Series(0.0, index=df_index)
        deception_raw = signals_data[deception_signal_name]
        if deception_factor_enabled:
            negative_deception = deception_raw.clip(upper=0).abs()
            normalized_negative_deception = get_adaptive_mtf_normalized_score(negative_deception, df_index, tf_weights)
            deception_impact = normalized_negative_deception * deception_impact_factor
        # --- 修正：恐慌消退速度对 panic_source_score 的调制 ---
        panic_slope_dampening_factor = pd.Series(1.0, index=df_index)
        if panic_slope_dampening_enabled:
            norm_panic_slope = get_adaptive_mtf_normalized_bipolar_score(slope_5_retail_panic_surrender_raw, df_index, tf_weights)
            # 如果恐慌斜率为负（恐慌消退），则降低 panic_source_score
            panic_slope_dampening_factor = (1 - norm_panic_slope.clip(upper=0).abs() * panic_slope_dampening_sensitivity).clip(0.1, 1.0)
            # if is_debug_enabled and probe_ts and probe_ts in df_index:
            #     print(f"        - 中间节点 (norm_panic_slope): {norm_panic_slope.loc[probe_ts]:.4f}")
            #     print(f"        - 中间节点 (panic_slope_dampening_factor): {panic_slope_dampening_factor.loc[probe_ts]:.4f}")
        pressure_test_score = base_pressure_score * (1 + capitulation_bonus + deception_impact) * panic_slope_dampening_factor # 应用调制因子
        pressure_test_score = pressure_test_score.clip(-1, 1)
        # if is_debug_enabled and probe_ts and probe_ts in df_index:
        #     print(f"        - 中间节点 (norm_capitulation_absorption): {norm_capitulation_absorption.loc[probe_ts]:.4f}")
        #     print(f"        - 中间节点 (norm_accel_5_capitulation_absorption): {norm_accel_5_capitulation_absorption.loc[probe_ts]:.4f}")
        #     print(f"        - 中间节点 (pressure_test_score): {pressure_test_score.loc[probe_ts]:.4f}")
        # --- 3. 杂质削弱 (Impurity Attenuation) ---
        s_belief_core = belief_core_score.add(1)/2
        s_pressure_test = pressure_test_score.add(1)/2
        dynamic_belief_core_weight = pd.Series(0.5, index=df_index)
        dynamic_pressure_test_weight = pd.Series(0.5, index=df_index)
        if dynamic_fusion_enabled:
            dynamic_pressure_test_weight = min_pressure_weight + (max_pressure_weight - min_pressure_weight) * normalized_panic_modulator
            dynamic_belief_core_weight = 1.0 - dynamic_pressure_test_weight
        conviction_base_unipolar = (s_belief_core.pow(dynamic_belief_core_weight) * s_pressure_test.pow(dynamic_pressure_test_weight))
        positive_deception_penalty = pd.Series(0.0, index=df_index)
        if positive_deception_penalty_enabled:
            positive_deception_raw = deception_raw.clip(lower=0)
            normalized_positive_deception = get_adaptive_mtf_normalized_score(positive_deception_raw, df_index, tf_weights)
            positive_deception_penalty = normalized_positive_deception * positive_deception_impact_factor
            conviction_base_unipolar = conviction_base_unipolar * (1 - positive_deception_penalty)
            conviction_base_unipolar = conviction_base_unipolar.clip(0, 1)
        # --- 诡道反噬机制深化 ---
        norm_deception_index_bipolar = get_adaptive_mtf_normalized_bipolar_score(deception_index_raw, df_index, tf_weights)
        norm_wash_trade_intensity = get_adaptive_mtf_normalized_score(wash_trade_intensity_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_main_force_conviction_bipolar = get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, tf_weights)
        conviction_threshold = deception_modulator_params.get('conviction_threshold', 0.2)
        # 诱空反吸增强 (负向欺骗且主力信念强)
        deception_boost_mask = (norm_deception_index_bipolar < 0) & (norm_main_force_conviction_bipolar > conviction_threshold)
        conviction_base_unipolar.loc[deception_boost_mask] = conviction_base_unipolar.loc[deception_boost_mask] * (1 + norm_deception_index_bipolar.loc[deception_boost_mask].abs() * deception_modulator_weights.get('deception_index_boost', 0.5))
        # 诱多惩罚 (正向欺骗且主力信念弱)
        deception_penalty_mask = (norm_deception_index_bipolar > 0) & (norm_main_force_conviction_bipolar < -conviction_threshold)
        conviction_base_unipolar.loc[deception_penalty_mask] = conviction_base_unipolar.loc[deception_penalty_mask] * (1 - norm_deception_index_bipolar.loc[deception_penalty_mask] * deception_modulator_weights.get('deception_index_penalty', 0.5))
        # 对倒惩罚
        conviction_base_unipolar = conviction_base_unipolar * (1 - norm_wash_trade_intensity * deception_modulator_weights.get('wash_trade_penalty', 0.3))
        conviction_base_unipolar = conviction_base_unipolar.clip(0, 1)
        # --- 杂质因子计算 ---
        norm_fomo_deviation = get_adaptive_mtf_normalized_score((fomo_index_raw - fomo_concentration_optimal_target).abs(), df_index, tf_weights=tf_weights)
        profit_taking_quality_thresholded = (profit_taking_quality_raw - profit_taking_threshold).clip(lower=0)
        norm_profit_taking_quality = get_adaptive_mtf_normalized_score(profit_taking_quality_thresholded, df_index, tf_weights=tf_weights)
        norm_upper_shadow_selling_pressure = get_adaptive_mtf_normalized_score(upper_shadow_selling_pressure_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_rally_distribution_pressure = get_adaptive_mtf_normalized_score(rally_distribution_pressure_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_retail_fomo_premium = get_adaptive_mtf_normalized_score(retail_fomo_premium_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_slope_5_winner_profit_margin = get_adaptive_mtf_normalized_score(slope_5_winner_profit_margin_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_accel_5_retail_fomo_premium = get_adaptive_mtf_normalized_score(accel_5_retail_fomo_premium_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_deception_lure_long_intensity = get_adaptive_mtf_normalized_score(deception_lure_long_intensity_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_deception_lure_short_intensity = get_adaptive_mtf_normalized_score(deception_lure_short_intensity_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_slope_5_chip_health = get_adaptive_mtf_normalized_bipolar_score(slope_5_chip_health_raw, df_index, tf_weights)
        norm_structural_tension = get_adaptive_mtf_normalized_score(structural_tension_raw, df_index, ascending=True, tf_weights=tf_weights)
        fomo_effect = pd.Series(0.0, index=df_index)
        profit_taking_effect = pd.Series(0.0, index=df_index)
        other_impurity_effect = pd.Series(0.0, index=df_index)
        final_impurity_effect = pd.Series(0.0, index=df_index)
        if impurity_non_linear_enabled:
            current_sentiment_strength = (conviction_base_unipolar * 2 - 1).abs() # 转换为 [-1,1] 再取绝对值
            normalized_sentiment_strength = normalize_score(current_sentiment_strength, df_index, windows=21, ascending=True)
            context_adjustment_factor = pd.Series(1.0, index=df_index)
            norm_volatility_instability = get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=True, tf_weights=tf_weights)
            norm_flow_credibility = get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, ascending=True, tf_weights=tf_weights)
            norm_chip_health_for_context = get_adaptive_mtf_normalized_score(chip_health_raw, df_index, ascending=True, tf_weights=tf_weights)
            context_modulator_numeric_weights = {k: v for k, v in context_modulator_weights.items() if isinstance(v, (int, float))}
            total_context_modulator_weight = sum(context_modulator_numeric_weights.values())
            if total_context_modulator_weight > 0:
                fused_context_modulator = (
                    norm_volatility_instability.pow(context_modulator_numeric_weights.get('volatility_instability', 0.4)) *
                    norm_flow_credibility.pow(context_modulator_numeric_weights.get('flow_credibility', 0.3)) *
                    norm_chip_health_for_context.pow(context_modulator_numeric_weights.get('chip_health', 0.3))
                ).pow(1 / total_context_modulator_weight)
                context_adjustment_factor = context_adjustment_factor * (1 + (fused_context_modulator - 0.5) * 0.5)
            if impurity_context_modulation_enabled:
                context_modulator_raw = signals_data[impurity_context_modulator_signal_name]
                normalized_context_modulator = get_adaptive_mtf_normalized_score(context_modulator_raw, df_index, tf_weights=tf_weights, ascending=True)
                overbought_mask = normalized_context_modulator > 0.7
                oversold_mask = normalized_context_modulator < 0.3
                context_adjustment_factor.loc[overbought_mask] = context_adjustment_factor.loc[overbought_mask] * (1 + (normalized_context_modulator.loc[overbought_mask] - 0.7) * impurity_context_overbought_amp_factor / 0.3)
                context_adjustment_factor.loc[oversold_mask] = context_adjustment_factor.loc[oversold_mask] * (1 - (0.3 - normalized_context_modulator.loc[oversold_mask]) * impurity_context_oversold_damp_factor / 0.3)
            dynamic_fomo_tanh_factor = fomo_tanh_factor * (1 + normalized_sentiment_strength * fomo_sentiment_sensitivity)
            dynamic_fomo_tanh_factor = dynamic_fomo_tanh_factor * context_adjustment_factor
            dynamic_fomo_tanh_factor = dynamic_fomo_tanh_factor.clip(0.5, 3.0)
            fomo_effect = np.tanh(norm_fomo_deviation * dynamic_fomo_tanh_factor)
            dynamic_profit_taking_tanh_factor = profit_taking_tanh_factor * (1 + normalized_sentiment_strength * profit_taking_sentiment_sensitivity)
            dynamic_profit_taking_tanh_factor = dynamic_profit_taking_tanh_factor * context_adjustment_factor
            dynamic_profit_taking_tanh_factor = dynamic_profit_taking_tanh_factor.clip(0.5, 3.0)
            profit_taking_effect = np.tanh(norm_profit_taking_quality * dynamic_profit_taking_tanh_factor)
            other_impurity_numeric_weights = {k: v for k, v in impurity_weights.items() if isinstance(v, (int, float)) and k not in ['fomo_concentration', 'profit_taking_margin', 'deception_lure_long', 'deception_lure_short', 'chip_health_slope', 'structural_tension']}
            total_other_impurity_weight = sum(other_impurity_numeric_weights.values())
            if total_other_impurity_weight > 0:
                other_impurity_components = {
                    'upper_shadow_selling_pressure': norm_upper_shadow_selling_pressure,
                    'rally_distribution_pressure': norm_rally_distribution_pressure,
                    'retail_fomo_premium': norm_retail_fomo_premium,
                    'winner_profit_margin_slope': norm_slope_5_winner_profit_margin,
                    'retail_fomo_premium_accel': norm_accel_5_retail_fomo_premium
                }
                other_impurity_score = _robust_geometric_mean(other_impurity_components, other_impurity_numeric_weights, df_index)
                other_impurity_effect = np.tanh(other_impurity_score * context_adjustment_factor)
            # --- 诡道反噬机制深化 (诱多/诱空欺骗强度) ---
            impurity_deception_modulator = pd.Series(1.0, index=df_index)
            if impurity_deception_mod_enabled:
                # 诱多增强杂质 (正向欺骗)
                impurity_deception_modulator = impurity_deception_modulator * (1 + norm_deception_lure_long_intensity * deception_lure_long_impurity_amp_factor)
                # 诱空减弱杂质 (负向欺骗，且主力信念强，视为洗盘)
                deception_lure_short_damp_mask = (norm_deception_lure_short_intensity > 0) & (norm_main_force_conviction_bipolar > conviction_threshold)
                impurity_deception_modulator.loc[deception_lure_short_damp_mask] = impurity_deception_modulator.loc[deception_lure_short_damp_mask] * (1 - norm_deception_lure_short_intensity.loc[deception_lure_short_damp_mask] * deception_lure_short_impurity_damp_factor)
            # --- 韧性重构机制引入 (筹码健康度斜率和结构性紧张指数) ---
            impurity_resilience_modulator = pd.Series(1.0, index=df_index)
            if impurity_resilience_mod_enabled:
                # 筹码健康度斜率负向时，杂质减弱 (筹码结构在改善)
                impurity_resilience_modulator = impurity_resilience_modulator * (1 - norm_slope_5_chip_health.clip(lower=0) * chip_health_slope_impurity_damp_factor)
                # 结构性紧张指数高时，杂质增强 (结构不稳定，容易出问题)
                impurity_resilience_modulator = impurity_resilience_modulator * (1 + norm_structural_tension * structural_tension_impurity_amp_factor)
            dynamic_impurity_fusion_exponent = impurity_fusion_exponent_base * (1 - normalized_sentiment_strength * impurity_fusion_exponent_sensitivity)
            dynamic_impurity_fusion_exponent = dynamic_impurity_fusion_exponent.clip(0.1, 1.0)
            # 融合所有杂质效应
            final_impurity_effect = 1 - ((1 - fomo_effect) * (1 - profit_taking_effect) * (1 - other_impurity_effect)).pow(dynamic_impurity_fusion_exponent)
            final_impurity_effect = final_impurity_effect * impurity_deception_modulator * impurity_resilience_modulator
            final_impurity_effect = final_impurity_effect.clip(0, 1)
        else:
            final_impurity_effect = pd.Series(0.0, index=df_index)
        # if is_debug_enabled and probe_ts and probe_ts in df_index:
        #     print(f"        - 中间节点 (final_impurity_effect): {final_impurity_effect.loc[probe_ts]:.4f}")
        # --- 4. 全局情境调制器 (Global Context Modulator) ---
        global_modulator_effect = pd.Series(1.0, index=df_index)
        if global_context_modulator_enabled:
            norm_global_chip_health = get_adaptive_mtf_normalized_score(chip_health_raw, df_index, ascending=True, tf_weights=tf_weights)
            norm_global_main_force_conviction = get_adaptive_mtf_normalized_score(main_force_conviction_raw.abs(), df_index, ascending=True, tf_weights=tf_weights)
            global_modulator_effect = (
                (1 + norm_global_chip_health * global_context_sensitivity_health) *
                (1 + norm_global_main_force_conviction * global_context_sensitivity_conviction)
            ).clip(0.5, 1.5)
        # if is_debug_enabled and probe_ts and probe_ts in df_index:
        #     print(f"        - 中间节点 (global_modulator_effect): {global_modulator_effect.loc[probe_ts]:.4f}")
        # --- 最终融合 ---
        # 将信念核心和压力测试融合后的分数，再减去杂质效应
        final_score = (conviction_base_unipolar * (1 - final_impurity_effect)) * 2 - 1
        final_score = final_score * global_modulator_effect
        final_score = final_score.clip(-1, 1).fillna(0.0).astype(np.float32)
        print(f"    -> [筹码层] 计算完成 筹码公理三：诊断“持仓信念韧性” 分数: {final_score.loc[probe_ts] if probe_ts and probe_ts in df_index else final_score.iloc[-1]:.4f}")
        return final_score

    def _diagnose_axiom_trend_momentum(self, df: pd.DataFrame, periods: list, strategic_posture: pd.Series, battlefield_geography: pd.Series, holder_sentiment: pd.Series) -> pd.Series:
        """
        【V7.7 · 战略推力引擎强化与燃料品质深度修正版】筹码公理六：诊断“结构性推力”
        - 核心升级1: 引擎功率动态权重。引入筹码健康度趋势作为调制器，动态调整静态基础分与动态变化率的融合权重。
        - 核心升级2: 燃料品质诡道调制。引入筹码故障幅度作为负向调制器，削弱被“诱多”等诡道污染的燃料品质，并使协同奖励情境感知。
        - 核心升级3: 喷管效率多维深化。融合真空区大小、真空区趋势和穿越效率，更全面评估最小阻力路径。
        - 核心升级4: 最终融合动态权重。引入战略态势作为情境调制器，动态调整引擎功率、燃料品质、喷管效率的融合权重。
        - 核心升级5: 新增筹码指标整合：
            - 向上脉冲强度 (`upward_impulse_strength_D`) 增强燃料品质维度。
        - 升级: 优化 synergy_bonus 计算，引入平滑激活函数，避免硬性截断。
        - 升级: 增强最终融合动态权重的情境感知，引入多情境调制器进行综合调整。
        - 核心修复1: 修正 `engine_power_score` 的动态权重调制逻辑，使其在筹码健康度斜率负向时，增加静态权重。
        - 核心修复2: 增强 `fuel_quality_score` 中 `deception_penalty` 对正向筹码故障的惩罚力度，并取消诱多情境下的协同奖励。
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        method_name = "_diagnose_axiom_trend_momentum"
        print(f"    -> [筹码层] 正在诊断“{method_name}” (V7.7 · 战略推力引擎强化与燃料品质深度修正版)”...")
        df_index = df.index
        required_signals = [
            'main_force_conviction_index_D', 'vacuum_zone_magnitude_D', 'upward_impulse_purity_D',
            'chip_health_score_D', 'chip_fault_magnitude_D', 'SLOPE_5_vacuum_zone_magnitude_D',
            'vacuum_traversal_efficiency_D',
            'upward_impulse_strength_D',
            'SLOPE_5_chip_health_score_D'
        ]
        p_conf = self.chip_ultimate_params
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
        upward_impulse_strength_weight = get_param_value(trend_momentum_params.get('upward_impulse_strength_weight'), 0.2)
        # Ensure modulator signals are in required_signals
        if engine_power_dynamic_weight_modulator_signal_name not in required_signals:
            required_signals.append(engine_power_dynamic_weight_modulator_signal_name)
        if synergy_bonus_context_modulator_signal_name not in required_signals:
            required_signals.append(synergy_bonus_context_modulator_signal_name)
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index)
        signals_data = self._get_all_required_signals(df, required_signals, method_name)
        # --- 调试信息构建 ---
        is_debug_enabled = self.should_probe
        probe_ts = None
        if is_debug_enabled and self.probe_dates_set:
            for date in reversed(df_index):
                if date.date() in self.probe_dates_set:
                    probe_ts = date
                    break
        debug_info_tuple = (is_debug_enabled, probe_ts, method_name)
        # if is_debug_enabled and probe_ts and probe_ts in df_index:
        #     print(f"      [探针] {method_name} 启动 @ {probe_ts.strftime('%Y-%m-%d')}")
        #     print(f"        - 原始数据 (strategic_posture): {strategic_posture.loc[probe_ts]:.4f}")
        #     print(f"        - 原始数据 (battlefield_geography): {battlefield_geography.loc[probe_ts]:.4f}")
        #     print(f"        - 原始数据 (holder_sentiment): {holder_sentiment.loc[probe_ts]:.4f}")
        #     print(f"        - 原始数据 (main_force_conviction_index_D): {signals_data['main_force_conviction_index_D'].loc[probe_ts]:.4f}")
        #     print(f"        - 原始数据 (upward_impulse_purity_D): {signals_data['upward_impulse_purity_D'].loc[probe_ts]:.4f}")
        #     print(f"        - 原始数据 (chip_fault_magnitude_D): {signals_data['chip_fault_magnitude_D'].loc[probe_ts]:.4f}")
        #     print(f"        - 原始数据 (vacuum_zone_magnitude_D): {signals_data['vacuum_zone_magnitude_D'].loc[probe_ts]:.4f}")
        #     print(f"        - 原始数据 (SLOPE_5_vacuum_zone_magnitude_D): {signals_data['SLOPE_5_vacuum_zone_magnitude_D'].loc[probe_ts]:.4f}")
        #     print(f"        - 原始数据 (vacuum_traversal_efficiency_D): {signals_data['vacuum_traversal_efficiency_D'].loc[probe_ts]:.4f}")
        #     print(f"        - 原始数据 (SLOPE_5_chip_health_score_D): {signals_data['SLOPE_5_chip_health_score_D'].loc[probe_ts]:.4f}")
        #     print(f"        - 原始数据 (upward_impulse_strength_D): {signals_data['upward_impulse_strength_D'].loc[probe_ts]:.4f}")
        #     print(f"        - 原始数据 (chip_health_score_D): {signals_data['chip_health_score_D'].loc[probe_ts]:.4f}")
        # --- 原始数据获取 ---
        signal_map = {
            "strategic_posture": strategic_posture,
            "battlefield_geography": battlefield_geography,
            "holder_sentiment": holder_sentiment
        }
        health_score_slope_raw = signals_data[engine_power_dynamic_weight_modulator_signal_name]
        conviction_raw = signals_data['main_force_conviction_index_D']
        impulse_purity_raw = signals_data['upward_impulse_purity_D']
        upward_impulse_strength_raw = signals_data['upward_impulse_strength_D']
        chip_fault_raw = signals_data['chip_fault_magnitude_D']
        synergy_context_raw = signals_data[synergy_bonus_context_modulator_signal_name]
        vacuum_magnitude_raw = signals_data['vacuum_zone_magnitude_D']
        vacuum_trend_raw = signals_data['SLOPE_5_vacuum_zone_magnitude_D']
        vacuum_traversal_raw = signals_data['vacuum_traversal_efficiency_D']
        # --- 1. 引擎功率 (Engine Power) ---
        static_engine_power = (
            strategic_posture * health_weights['posture'] +
            battlefield_geography * health_weights['geography'] +
            holder_sentiment * health_weights['sentiment']
        )
        norm_health_score_slope = get_adaptive_mtf_normalized_bipolar_score(health_score_slope_raw, df_index, tf_weights)
        # 修正：当筹码健康度斜率为负时，增加静态权重，降低动态权重
        dynamic_weight_mod = (norm_health_score_slope * engine_power_dynamic_weight_sensitivity)
        current_static_weight = (static_engine_power_base_weight - dynamic_weight_mod).clip(0.1, 0.9)
        current_dynamic_weight = (dynamic_engine_power_base_weight + dynamic_weight_mod).clip(0.1, 0.9)
        # 重新归一化，确保总和为1
        sum_current_weights = current_static_weight + current_dynamic_weight
        current_static_weight = current_static_weight / sum_current_weights
        current_dynamic_weight = current_dynamic_weight / sum_current_weights
        slope = static_engine_power.diff(1).fillna(0)
        accel = slope.diff(1).fillna(0)
        norm_slope = get_adaptive_mtf_normalized_bipolar_score(slope, df_index, tf_weights)
        norm_accel = get_adaptive_mtf_normalized_bipolar_score(accel, df_index, tf_weights)
        dynamic_engine_power = ((norm_slope.add(1)/2) * (norm_accel.clip(lower=-1, upper=1).add(1)/2)).pow(0.5) * 2 - 1
        engine_power_score = static_engine_power * current_static_weight + dynamic_engine_power * current_dynamic_weight
        # if is_debug_enabled and probe_ts and probe_ts in df_index:
        #     print(f"        - 中间节点 (static_engine_power): {static_engine_power.loc[probe_ts]:.4f}")
        #     print(f"        - 中间节点 (norm_health_score_slope): {norm_health_score_slope.loc[probe_ts]:.4f}")
        #     print(f"        - 中间节点 (dynamic_weight_mod): {dynamic_weight_mod.loc[probe_ts]:.4f}")
        #     print(f"        - 中间节点 (current_static_weight): {current_static_weight.loc[probe_ts]:.4f}")
        #     print(f"        - 中间节点 (current_dynamic_weight): {current_dynamic_weight.loc[probe_ts]:.4f}")
        #     print(f"        - 中间节点 (engine_power_score): {engine_power_score.loc[probe_ts]:.4f}")
        # --- 2. 燃料品质 (Fuel Quality) ---
        conviction_score = get_adaptive_mtf_normalized_bipolar_score(conviction_raw, df_index, tf_weights)
        purity_score = get_adaptive_mtf_normalized_bipolar_score(impulse_purity_raw, df_index, tf_weights)
        norm_upward_impulse_strength = get_adaptive_mtf_normalized_score(upward_impulse_strength_raw, df_index, ascending=True, tf_weights=tf_weights)
        base_fuel_quality = ((conviction_score.add(1)/2) * (purity_score.add(1)/2)).pow(0.5) * 2 - 1
        base_fuel_quality = base_fuel_quality * (1 + norm_upward_impulse_strength * upward_impulse_strength_weight)
        base_fuel_quality = base_fuel_quality.clip(-1, 1)
        norm_chip_fault = get_adaptive_mtf_normalized_score(chip_fault_raw.abs(), df_index, ascending=True, tf_weights=tf_weights)
        deception_penalty = pd.Series(0.0, index=df_index)
        positive_fault_mask = chip_fault_raw > 0 # 筹码故障为正，视为负面影响 (诱多)
        # 修正：增强对正向筹码故障的惩罚力度
        deception_penalty.loc[positive_fault_mask] = norm_chip_fault.loc[positive_fault_mask] * fuel_purity_deception_penalty_factor * 4.0 # 惩罚因子进一步加倍
        fuel_quality_score_after_deception = base_fuel_quality - deception_penalty.clip(0, 1) # 直接减去惩罚
        # --- 协同奖励情境感知 ---
        norm_synergy_context = get_adaptive_mtf_normalized_score(synergy_context_raw, df_index, ascending=True, tf_weights=tf_weights)
        dynamic_synergy_bonus_factor = synergy_bonus_base * (1 + norm_synergy_context * synergy_bonus_context_sensitivity)
        dynamic_synergy_bonus_factor = dynamic_synergy_bonus_factor.clip(0.1, 0.5)
        conviction_norm = (conviction_score + 1) / 2
        purity_norm = (purity_score + 1) / 2
        synergy_potential = (conviction_norm * purity_norm).pow(0.5)
        # 修正：当存在正向筹码故障（诱多）时，取消协同奖励
        synergy_bonus = pd.Series(0.0, index=df_index)
        synergy_activation_mask = ~positive_fault_mask # 只有在没有诱多故障时才激活协同奖励
        synergy_activation = (1 / (1 + np.exp(-(synergy_potential.loc[synergy_activation_mask] - synergy_activation_threshold) * 10))).clip(0, 1) # Sigmoid-like activation
        synergy_bonus.loc[synergy_activation_mask] = synergy_activation * dynamic_synergy_bonus_factor.loc[synergy_activation_mask]
        fuel_quality_score = fuel_quality_score_after_deception + synergy_bonus
        fuel_quality_score = fuel_quality_score.clip(-1, 1)
        # if is_debug_enabled and probe_ts and probe_ts in df_index:
        #     print(f"        - 中间节点 (base_fuel_quality): {base_fuel_quality.loc[probe_ts]:.4f}")
        #     print(f"        - 中间节点 (norm_chip_fault): {norm_chip_fault.loc[probe_ts]:.4f}")
        #     print(f"        - 中间节点 (deception_penalty): {deception_penalty.loc[probe_ts]:.4f}")
        #     print(f"        - 中间节点 (synergy_bonus): {synergy_bonus.loc[probe_ts]:.4f}") # 新增探针
        #     print(f"        - 中间节点 (fuel_quality_score): {fuel_quality_score.loc[probe_ts]:.4f}")
        # --- 3. 喷管效率 (Nozzle Efficiency) ---
        norm_vacuum_magnitude = get_adaptive_mtf_normalized_bipolar_score(vacuum_magnitude_raw, df_index, tf_weights)
        norm_vacuum_trend = get_adaptive_mtf_normalized_bipolar_score(vacuum_trend_raw, df_index, tf_weights)
        norm_traversal_efficiency = get_adaptive_mtf_normalized_bipolar_score(vacuum_traversal_raw, df_index, tf_weights)
        nozzle_efficiency_score = (
            norm_vacuum_magnitude * nozzle_efficiency_weights.get('magnitude', 0.5) +
            norm_vacuum_trend * nozzle_efficiency_weights.get('trend', 0.3) +
            norm_traversal_efficiency * nozzle_efficiency_weights.get('traversal', 0.2)
        ).clip(-1, 1)
        # if is_debug_enabled and probe_ts and probe_ts in df_index:
        #     print(f"        - 中间节点 (nozzle_efficiency_score): {nozzle_efficiency_score.loc[probe_ts]:.4f}")
        # --- 4. 最终融合动态权重 (Final Fusion Dynamic Weights) ---
        engine_score_normalized = (engine_power_score + 1) / 2
        fuel_score_normalized = (fuel_quality_score + 1) / 2
        nozzle_score_normalized = (nozzle_efficiency_score + 1) / 2
        final_engine_weight = pd.Series(final_fusion_weights_base.get('engine', 0.33), index=df_index)
        final_fuel_weight = pd.Series(final_fusion_weights_base.get('fuel', 0.33), index=df_index)
        final_nozzle_weight = pd.Series(final_fusion_weights_base.get('nozzle', 0.34), index=df_index)
        if final_fusion_dynamic_weights_enabled:
            context_modulator_components = []
            total_context_weight = 0.0
            for ctx_name, ctx_config in final_fusion_context_modulators_config.items():
                signal_key = ctx_config.get('signal')
                signal_series = signal_map.get(signal_key) # 从传入的 signal_map 获取
                weight = ctx_config.get('weight', 0.0)
                sensitivity = ctx_config.get('sensitivity', 0.0)
                if signal_series is not None and weight > 0:
                    norm_signal = get_adaptive_mtf_normalized_bipolar_score(signal_series, df_index, tf_weights)
                    context_modulator_components.append(norm_signal * weight * sensitivity)
                    total_context_weight += weight * sensitivity
            if context_modulator_components and total_context_weight > 0:
                context_fusion_modulator = sum(context_modulator_components) / total_context_weight
                normalized_fusion_modulator = context_fusion_modulator # 已经归一化到 [-1, 1]
            else:
                normalized_fusion_modulator = pd.Series(0.0, index=df_index)
            # 根据情境调制器调整权重
            engine_mod = normalized_fusion_modulator * final_fusion_weights_sensitivity.get('engine', 0.5)
            fuel_mod = normalized_fusion_modulator * final_fusion_weights_sensitivity.get('fuel', 0.5)
            nozzle_mod = -normalized_fusion_modulator * final_fusion_weights_sensitivity.get('nozzle', 0.5) # 负向调制，当情境有利时，喷管权重降低
            final_engine_weight = (final_fusion_weights_base.get('engine', 0.33) + engine_mod).clip(0.1, 0.6)
            final_fuel_weight = (final_fusion_weights_base.get('fuel', 0.33) + fuel_mod).clip(0.1, 0.6)
            final_nozzle_weight = (final_fusion_weights_base.get('nozzle', 0.34) + nozzle_mod).clip(0.1, 0.6)
            # 重新归一化权重，确保总和为1
            sum_dynamic_fusion_weights = final_engine_weight + final_fuel_weight + final_nozzle_weight
            final_engine_weight = final_engine_weight / sum_dynamic_fusion_weights
            final_fuel_weight = final_fuel_weight / sum_dynamic_fusion_weights
            final_nozzle_weight = final_nozzle_weight / sum_dynamic_fusion_weights
        # --- 最终融合 ---
        final_score_unipolar = (
            engine_score_normalized.pow(final_engine_weight) *
            fuel_score_normalized.pow(final_fuel_weight) *
            nozzle_score_normalized.pow(final_nozzle_weight)
        ).pow(1 / (final_engine_weight + final_fuel_weight + final_nozzle_weight)) # 几何平均
        final_score = (final_score_unipolar * 2 - 1).clip(-1, 1)
        final_score = final_score.clip(-1, 1).fillna(0.0).astype(np.float32)
        print(f"    -> [筹码层] 计算完成 '{method_name}' 分数: {final_score.loc[probe_ts] if probe_ts and probe_ts in df_index else final_score.iloc[-1]:.4f}")
        return final_score

    def _diagnose_axiom_divergence(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V7.3 · 诡道反噬强化版】筹码公理五：诊断“价筹张力”
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
        - 核心修复: 增强 `deception_modulator_factor` 的惩罚机制，当出现“诱空”且主力资金流出时，大幅降低分数。
        """
        method_name = "_diagnose_axiom_divergence"
        df_index = df.index
        required_signals = [
            'winner_loser_momentum_D', 'winner_concentration_90pct_D', 'SLOPE_5_close_D',
            'constructive_turnover_ratio_D', 'main_force_conviction_index_D', 'chip_fault_magnitude_D',
            'chip_health_score_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D',
            'main_force_flow_directionality_D', 'deception_index_D' # 新增所需信号
        ]
        p_conf = self.chip_ultimate_params
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        divergence_params = get_param_value(p_conf.get('divergence_params'), {})
        chip_trend_momentum_weight_base = get_param_value(divergence_params.get('chip_trend_momentum_weight'), 0.6)
        chip_trend_concentration_weight_base = get_param_value(divergence_params.get('chip_trend_concentration_weight'), 0.4)
        tension_magnitude_amplifier_base = get_param_value(divergence_params.get('tension_magnitude_amplifier'), 1.5)
        chip_intent_factor_amplifier_base = get_param_value(divergence_params.get('chip_intent_factor_amplifier'), 0.5)
        deception_modulator_impact_clip = get_param_value(divergence_params.get('deception_modulator_impact_clip'), 0.5)
        deception_modulator_reinforce_factor = get_param_value(divergence_params.get('deception_modulator_reinforce_factor'), 0.5)
        conflict_bonus = get_param_value(divergence_params.get('conflict_bonus'), 0.5)
        contextual_amplification_enabled = get_param_value(divergence_params.get('contextual_amplification_enabled'), True)
        context_modulator_signal_name = get_param_value(divergence_params.get('context_modulator_signal_name'), 'chip_health_score_D')
        context_sensitivity_tension = get_param_value(divergence_params.get('context_sensitivity_tension'), 0.5)
        context_sensitivity_intent = get_param_value(divergence_params.get('context_sensitivity_intent'), 0.5)
        non_linear_amplification_enabled = get_param_value(divergence_params.get('non_linear_amplification_enabled'), True)
        non_linear_amp_tanh_factor = get_param_value(divergence_params.get('non_linear_amp_tanh_factor'), 1.0)
        dynamic_chip_trend_weights_enabled = get_param_value(divergence_params.get('dynamic_chip_trend_weights_enabled'), True)
        chip_trend_weight_modulator_signal_name = get_param_value(divergence_params.get('chip_trend_weight_modulator_signal_name'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        chip_trend_weight_mod_sensitivity = get_param_value(divergence_params.get('chip_trend_weight_mod_sensitivity'), 0.5)
        # 新增参数
        bearish_deception_penalty_factor = get_param_value(divergence_params.get('bearish_deception_penalty_factor'), 0.8)
        if chip_trend_weight_modulator_signal_name not in required_signals:
            required_signals.append(chip_trend_weight_modulator_signal_name)
        if context_modulator_signal_name not in required_signals:
            required_signals.append(context_modulator_signal_name)
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index)
        signals_data = self._get_all_required_signals(df, required_signals, method_name)
        # --- 调试信息构建 ---
        is_debug_enabled = self.should_probe
        probe_ts = None
        if is_debug_enabled and self.probe_dates_set:
            for date in reversed(df_index):
                if date.date() in self.probe_dates_set:
                    probe_ts = date
                    break
        debug_info_tuple = (is_debug_enabled, probe_ts, method_name)
        # if is_debug_enabled and probe_ts and probe_ts in df_index:
        #     print(f"      [探针] {method_name} 启动 @ {probe_ts.strftime('%Y-%m-%d')}")
        #     print(f"        - 原始数据 (winner_loser_momentum_D): {signals_data['winner_loser_momentum_D'].loc[probe_ts]:.4f}")
        #     print(f"        - 原始数据 (winner_concentration_90pct_D): {signals_data['winner_concentration_90pct_D'].loc[probe_ts]:.4f}")
        #     print(f"        - 原始数据 (SLOPE_5_close_D): {signals_data['SLOPE_5_close_D'].loc[probe_ts]:.4f}")
        #     print(f"        - 原始数据 (constructive_turnover_ratio_D): {signals_data['constructive_turnover_ratio_D'].loc[probe_ts]:.4f}")
        #     print(f"        - 原始数据 (main_force_conviction_index_D): {signals_data['main_force_conviction_index_D'].loc[probe_ts]:.4f}")
        #     print(f"        - 原始数据 (chip_fault_magnitude_D): {signals_data['chip_fault_magnitude_D'].loc[probe_ts]:.4f}")
        #     print(f"        - 原始数据 (chip_health_score_D): {signals_data['chip_health_score_D'].loc[probe_ts]:.4f}")
        #     print(f"        - 原始数据 (VOLATILITY_INSTABILITY_INDEX_21d_D): {signals_data['VOLATILITY_INSTABILITY_INDEX_21d_D'].loc[probe_ts]:.4f}")
        #     print(f"        - 原始数据 (main_force_flow_directionality_D): {signals_data['main_force_flow_directionality_D'].loc[probe_ts]:.4f}") # 新增探针
        #     print(f"        - 原始数据 (deception_index_D): {signals_data['deception_index_D'].loc[probe_ts]:.4f}") # 新增探针
        # --- 原始数据获取 ---
        chip_momentum_raw = signals_data['winner_loser_momentum_D']
        chip_concentration_raw = signals_data['winner_concentration_90pct_D']
        price_trend_raw = signals_data['SLOPE_5_close_D']
        constructive_turnover_raw = signals_data['constructive_turnover_ratio_D']
        mf_chip_conviction_raw = signals_data['main_force_conviction_index_D']
        chip_fault_raw = signals_data['chip_fault_magnitude_D']
        chip_health_raw = signals_data['chip_health_score_D']
        chip_trend_modulator_raw = signals_data[chip_trend_weight_modulator_signal_name]
        context_modulator_raw = signals_data[context_modulator_signal_name]
        main_force_flow_directionality_raw = signals_data['main_force_flow_directionality_D'] # 获取新增信号
        deception_index_raw = signals_data['deception_index_D'] # 获取新增信号
        # --- 1. 复合筹码趋势 (Composite Chip Trend) ---
        dynamic_momentum_weight = pd.Series(chip_trend_momentum_weight_base, index=df_index)
        dynamic_concentration_weight = pd.Series(chip_trend_concentration_weight_base, index=df_index)
        if dynamic_chip_trend_weights_enabled:
            normalized_chip_trend_modulator = get_adaptive_mtf_normalized_score(chip_trend_modulator_raw, df_index, tf_weights=tf_weights, ascending=True)
            dynamic_momentum_weight = chip_trend_momentum_weight_base * (1 + normalized_chip_trend_modulator * chip_trend_weight_mod_sensitivity)
            dynamic_concentration_weight = chip_trend_concentration_weight_base * (1 - normalized_chip_trend_modulator * chip_trend_weight_mod_sensitivity)
            sum_dynamic_weights = dynamic_momentum_weight + dynamic_concentration_weight
            dynamic_momentum_weight = (dynamic_momentum_weight / sum_dynamic_weights).clip(0.1, 0.9)
            dynamic_concentration_weight = (dynamic_concentration_weight / sum_dynamic_weights).clip(0.1, 0.9)
        norm_chip_momentum = get_adaptive_mtf_normalized_bipolar_score(chip_momentum_raw, df_index, tf_weights)
        norm_chip_concentration = get_adaptive_mtf_normalized_bipolar_score(chip_concentration_raw, df_index, tf_weights)
        composite_chip_trend = (
            norm_chip_momentum * dynamic_momentum_weight +
            norm_chip_concentration * dynamic_concentration_weight
        )
        # if is_debug_enabled and probe_ts and probe_ts in df_index:
        #     print(f"        - 中间节点 (composite_chip_trend): {composite_chip_trend.loc[probe_ts]:.4f}")
        # --- 2. 价筹分歧 (Price-Chip Disagreement) ---
        norm_price_trend = get_adaptive_mtf_normalized_bipolar_score(price_trend_raw, df_index, tf_weights)
        disagreement_vector = composite_chip_trend - norm_price_trend
        # if is_debug_enabled and probe_ts and probe_ts in df_index:
        #     print(f"        - 中间节点 (norm_price_trend): {norm_price_trend.loc[probe_ts]:.4f}")
        #     print(f"        - 中间节点 (disagreement_vector): {disagreement_vector.loc[probe_ts]:.4f}")
        # --- 3. 持续性与能量注入 (Persistence & Energy Injection) ---
        persistence_raw = np.sign(disagreement_vector).rolling(window=13, min_periods=5).sum().fillna(0)
        norm_persistence = get_adaptive_mtf_normalized_score(persistence_raw.abs(), df_index, tf_weights=tf_weights)
        norm_constructive_turnover = get_adaptive_mtf_normalized_score(constructive_turnover_raw, df_index, tf_weights=tf_weights)
        energy_injection = norm_constructive_turnover * disagreement_vector.abs()
        tension_magnitude = (norm_persistence * energy_injection).pow(0.5)
        # if is_debug_enabled and probe_ts and probe_ts in df_index:
        #     print(f"        - 中间节点 (tension_magnitude): {tension_magnitude.loc[probe_ts]:.4f}")
        # --- 4. 主力筹码意图验证 (Main Force Chip Intent Verification) ---
        norm_mf_chip_conviction = get_adaptive_mtf_normalized_bipolar_score(mf_chip_conviction_raw, df_index, tf_weights)
        is_aligned = (np.sign(disagreement_vector) * np.sign(norm_mf_chip_conviction)) > 0
        intent_strength = norm_mf_chip_conviction.abs()
        chip_intent_verification_score = is_aligned * intent_strength
        # if is_debug_enabled and probe_ts and probe_ts in df_index:
        #     print(f"        - 中间节点 (norm_mf_chip_conviction): {norm_mf_chip_conviction.loc[probe_ts]:.4f}")
        #     print(f"        - 中间节点 (chip_intent_verification_score): {chip_intent_verification_score.loc[probe_ts]:.4f}")
        # --- 5. 情境自适应放大器 (Context-Adaptive Amplifier) ---
        dynamic_tension_amplifier = pd.Series(tension_magnitude_amplifier_base, index=df_index)
        dynamic_chip_intent_factor_amplifier = pd.Series(chip_intent_factor_amplifier_base, index=df_index)
        if contextual_amplification_enabled:
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
        # if is_debug_enabled and probe_ts and probe_ts in df_index:
        #     print(f"        - 中间节点 (tension_amplification_term): {tension_amplification_term.loc[probe_ts]:.4f}")
        #     print(f"        - 中间节点 (chip_intent_factor): {chip_intent_factor.loc[probe_ts]:.4f}")
        # --- 6. 诡道双向调制 (Deceptive Bidirectional Modulation) ---
        norm_chip_fault = get_adaptive_mtf_normalized_score(chip_fault_raw.abs(), df_index, ascending=True, tf_weights=tf_weights)
        divergence_sign = np.sign(disagreement_vector)
        fault_sign = np.sign(chip_fault_raw) # 筹码故障的正负代表方向
        deception_modulator_factor = pd.Series(1.0, index=df_index)
        # 修正：增强对“诱空”且主力流出情境的惩罚
        norm_deception_index = get_adaptive_mtf_normalized_bipolar_score(deception_index_raw, df_index, tf_weights)
        norm_main_force_flow_directionality = get_adaptive_mtf_normalized_bipolar_score(main_force_flow_directionality_raw, df_index, tf_weights)
        # 诱空且主力流出：强惩罚
        bearish_deception_and_mf_out_mask = (norm_deception_index < 0) & (norm_main_force_flow_directionality < 0)
        deception_modulator_factor.loc[bearish_deception_and_mf_out_mask] = \
            deception_modulator_factor.loc[bearish_deception_and_mf_out_mask] * \
            (1 - norm_deception_index.loc[bearish_deception_and_mf_out_mask].abs() * \
                 norm_main_force_flow_directionality.loc[bearish_deception_and_mf_out_mask].abs() * \
                 bearish_deception_penalty_factor)
        # 故障方向与分歧方向一致时，惩罚 (例如，价跌筹码故障为负，分歧为负，则惩罚)
        align_mask = (divergence_sign == fault_sign) & (~bearish_deception_and_mf_out_mask) # 排除已处理的强惩罚情境
        deception_modulator_factor.loc[align_mask] = 1 - norm_chip_fault.loc[align_mask] * deception_modulator_impact_clip
        # 故障方向与分歧方向相反时，增强 (例如，价跌筹码故障为正，分歧为负，则增强，视为洗盘)
        oppose_mask = (divergence_sign != fault_sign) & (~bearish_deception_and_mf_out_mask) # 排除已处理的强惩罚情境
        deception_modulator_factor.loc[oppose_mask] = 1 + norm_chip_fault.loc[oppose_mask] * deception_modulator_reinforce_factor
        deception_modulator_factor = deception_modulator_factor.clip(0.01, 2.0) # 惩罚可以更深
        # if is_debug_enabled and probe_ts and probe_ts in df_index:
        #     print(f"        - 中间节点 (deception_modulator_factor): {deception_modulator_factor.loc[probe_ts]:.4f}")
        # --- 7. 最终融合 ---
        base_final_score = disagreement_vector * (1 + tension_amplification_term) * chip_intent_factor * deception_modulator_factor
        # 冲突奖励：当价筹趋势方向相反时，给予额外奖励，放大分歧信号
        conflict_mask = (np.sign(composite_chip_trend) * np.sign(norm_price_trend) < 0)
        conflict_amplifier = pd.Series(1.0, index=df_index)
        conflict_amplifier.loc[conflict_mask] = 1.0 + conflict_bonus
        # 使用 tanh 确保最终分数在 [-1, 1] 范围内，并防止过度放大
        safe_base_score = base_final_score.clip(-0.999, 0.999) # 避免 arctanh(+-1) 为 inf
        final_score = np.tanh(np.arctanh(safe_base_score) * conflict_amplifier)
        final_score = final_score.clip(-1, 1).fillna(0.0).astype(np.float32)
        print(f"    -> [筹码层] 计算完成 筹码公理五：诊断“价筹张力” 分数: {final_score.loc[probe_ts] if probe_ts and probe_ts in df_index else final_score.iloc[-1]:.4f}")
        return final_score

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
        df_index = df.index
        p_conf = self.chip_ultimate_params
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
        chip_sensitivity_mod_tanh_factor_amp = get_param_value(coherent_drive_params.get('chip_health_sensitivity_mod_tanh_factor_amp'), 1.0)
        chip_sensitivity_mod_tanh_factor_damp = get_param_value(coherent_drive_params.get('chip_health_sensitivity_mod_tanh_factor_damp'), 1.0)
        cost_structure_asymmetric_impact_enabled = get_param_value(coherent_drive_params.get('cost_structure_asymmetric_impact_enabled'), False)
        cost_structure_impact_base_factor_bullish = get_param_value(coherent_drive_params.get('cost_structure_impact_base_factor_bullish'), 1.0)
        cost_structure_impact_base_factor_bearish = get_param_value(coherent_drive_params.get('cost_structure_impact_base_factor_bearish'), 1.0)
        cost_structure_impact_sentiment_sensitivity_bullish = get_param_value(coherent_drive_params.get('cost_structure_impact_sentiment_sensitivity_bullish'), 1.0)
        cost_structure_impact_sentiment_tanh_factor_bullish = get_param_value(coherent_drive_params.get('cost_structure_impact_sentiment_tanh_factor_bullish'), 1.0)
        cost_structure_impact_sentiment_sensitivity_bearish = get_param_value(coherent_drive_params.get('cost_structure_impact_sentiment_sensitivity_bearish'), 1.0)
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
        required_signals = [
            'chip_health_score_D',
        ]
        if chip_sensitivity_modulator_signal_name not in required_signals:
            required_signals.append(chip_sensitivity_modulator_signal_name)
        if final_score_modulator_signal_name not in required_signals:
            required_signals.append(final_score_modulator_signal_name)
        if not self._validate_required_signals(df, required_signals, "_diagnose_structural_consensus"):
            return pd.Series(0.0, index=df.index)
        signals_data = self._get_all_required_signals(df, required_signals, "_diagnose_structural_consensus")
        if chip_health_modulation_enabled:
            current_chip_health_score_raw = signals_data['chip_health_score_D']
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
                modulator_signal_raw = signals_data[chip_sensitivity_modulator_signal_name]
                normalized_modulator_signal = normalize_score(
                    modulator_signal_raw,
                    df_index,
                    windows=chip_sensitivity_mod_norm_window,
                    ascending=True
                )
                modulator_bipolar = (normalized_modulator_signal * 2) - 1
                non_linear_modulator_effect_amp = np.tanh(modulator_bipolar * chip_sensitivity_mod_tanh_factor_amp)
                non_linear_modulator_effect_damp = np.tanh(modulator_bipolar * chip_sensitivity_mod_tanh_factor_damp)
                dynamic_chip_health_sensitivity_amp = base_amp_sensitivity_series * (1 + non_linear_modulator_effect_amp * chip_sensitivity_mod_factor_amp)
                dynamic_chip_health_sensitivity_damp = base_damp_sensitivity_series * (1 + non_linear_modulator_effect_damp * chip_sensitivity_mod_factor_damp)
                dynamic_chip_health_sensitivity_amp = dynamic_chip_health_sensitivity_amp.clip(base_amp_sensitivity_series * 0.1, base_amp_sensitivity_series * 2.0)
                dynamic_chip_health_sensitivity_damp = dynamic_chip_health_sensitivity_damp.clip(base_damp_sensitivity_series * 0.1, base_damp_sensitivity_series * 2.0)
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
        if structural_power_adjustment_enabled:
            positive_structure_mask = final_cost_structure_for_modulation_scaled > 0
            negative_structure_mask = final_cost_structure_for_modulation_scaled < 0
            if structural_power_asymmetric_tanh_enabled:
                if positive_structure_mask.any():
                    positive_structure_strength = final_cost_structure_for_modulation_scaled[positive_structure_mask]
                    boost_amp = np.tanh((positive_structure_strength + structural_power_offset_positive_structure) * structural_power_tanh_factor_positive_structure) * dynamic_structural_power_sensitivity_amp.loc[positive_structure_mask]
                    amplification_power.loc[positive_structure_mask] = amplification_power.loc[positive_structure_mask] * (1 + boost_amp)
                if negative_structure_mask.any():
                    negative_structure_strength = final_cost_structure_for_modulation_scaled[negative_structure_mask].abs()
                    boost_damp = np.tanh((negative_structure_strength + structural_power_offset_negative_structure) * structural_power_tanh_factor_negative_structure) * dynamic_structural_power_sensitivity_damp.loc[negative_structure_mask]
                    dampening_power.loc[negative_structure_mask] = dampening_power.loc[negative_structure_mask] * (1 + boost_damp)
            else:
                if positive_structure_mask.any():
                    positive_structure_strength = final_cost_structure_for_modulation_scaled[positive_structure_mask]
                    boost_amp = np.tanh(positive_structure_strength * default_structural_power_tanh_factor_amp) * dynamic_structural_power_sensitivity_amp.loc[positive_structure_mask]
                    amplification_power.loc[positive_structure_mask] = amplification_power.loc[positive_structure_mask] * (1 + boost_amp)
                if negative_structure_mask.any():
                    negative_structure_strength = final_cost_structure_for_modulation_scaled[negative_structure_mask].abs()
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
            final_score_modulator_signal_raw = signals_data[final_score_modulator_signal_name]
            final_score_normalized_modulator_signal = normalize_score(
                final_score_modulator_signal_raw,
                df_index,
                windows=final_score_mod_norm_window,
                ascending=True
            )
            final_score_modulator_bipolar = (final_score_normalized_modulator_signal * 2) - 1
            final_score_non_linear_modulator_effect = np.tanh(final_score_modulator_bipolar * final_score_mod_tanh_factor)
            dynamic_final_score_sensitivity_multiplier = final_score_base_sensitivity_multiplier * (1 + final_score_non_linear_modulator_effect * final_score_mod_factor)
            dynamic_final_score_sensitivity_multiplier = dynamic_final_score_sensitivity_multiplier.clip(final_score_base_sensitivity_multiplier * 0.5, final_score_base_sensitivity_multiplier * 2.0)
        else:
            dynamic_final_score_sensitivity_multiplier = pd.Series(final_score_base_sensitivity_multiplier, index=df.index)
        final_score = np.tanh(coherent_drive_raw * (self.bipolar_sensitivity * dynamic_final_score_sensitivity_multiplier))
        print(f"    -> [筹码层] 计算完成 '筹码同调驱动力' 分数: {final_score.iloc[-1]}")
        return final_score.astype(np.float32)

    def _diagnose_absorption_echo(self, df: pd.DataFrame, divergence_scores: pd.Series) -> pd.Series:
        """
        【V5.3 · 诡道反吸强化与恐慌动态修正版 & 牛市陷阱惩罚版】吸筹回声探针
        - 核心升级1: 恐慌声源精细化。在V4.0基础上，引入总输家比例短期加速度、散户恐慌投降指数短期斜率、结构性紧张指数短期加速度，更精准捕捉恐慌蔓延。
        - 核心升级2: 逆流介质强化。在V4.0基础上，引入浮动筹码清洗效率短期斜率、订单簿清算率短期加速度、微观价格冲击不对称性短期斜率、VWAP控制强度短期斜率、VWAP穿越强度短期加速度，更全面评估承接能力。
        - 核心升级3: 主力回声深化。在V4.0基础上，引入隐蔽吸筹信号短期加速度、压制式吸筹强度短期斜率、主力成本优势短期加速度、主力资金流向方向性短期斜率、主力VPOC短期加速度、智能资金净买入短期斜率，更细致刻画主力吸筹意图。
        - 核心升级4: 诡道背景调制智能化。优化诡道调制逻辑，引入“诱空反吸”的增强机制，即当出现“诱空”式欺骗且主力信念坚定，则增强吸筹回声信号；同时细化对“诱多”式欺骗和对倒行为的惩罚。
        - 核心升级5: 情境调制器引入。引入资金流可信度指数、结构性紧张指数、筹码健康度作为最终分数的调制器，提供更丰富的宏观情境感知。
        - 核心升级6: 新增筹码指标整合：
            - 支持性派发强度 (`supportive_distribution_intensity_D`) 作为负向调制器。
        - 核心修复1: 修正 `panic_source_score` 对恐慌动态的判断，当散户恐慌快速消退时，降低其贡献。
        - 核心修复2: 调整 `deception_modulator` 逻辑，当 `norm_deception_index_bipolar` 为负时，应增强 `absorption_echo` 信号。
        - **新增业务逻辑：引入“牛市陷阱情境惩罚”，在近期大幅下跌后伴随正向欺骗时，大幅降低吸筹回声的得分。**
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        method_name = "_diagnose_absorption_echo"
        print(f"    -> [筹码情报校验] 正在诊断“{method_name}” (V5.3 · 诡道反吸强化与恐慌动态修正版 & 牛市陷阱惩罚版)...")
        df_index = df.index
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
            'supportive_distribution_intensity_D',
            'vwap_control_strength_D', 'vwap_crossing_intensity_D', 'micro_price_impact_asymmetry_D',
            'pct_change_D' # 新增牛市陷阱检测所需信号
        ]
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index)
        signals_data = self._get_all_required_signals(df, required_signals, method_name)
        p_conf = self.chip_ultimate_params
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        absorption_echo_params = get_param_value(p_conf.get('absorption_echo_params'), {})
        panic_source_weights = get_param_value(absorption_echo_params.get('panic_source_weights'), {})
        panic_context_threshold = get_param_value(absorption_echo_params.get('panic_context_threshold'), 0.3)
        counter_flow_medium_weights = get_param_value(absorption_echo_params.get('counter_flow_medium_weights'), {})
        main_force_echo_weights = get_param_value(absorption_echo_params.get('main_force_echo_weights'), {})
        deception_modulator_params = get_param_value(absorption_echo_params.get('deception_modulator_params'), {})
        final_fusion_exponent = get_param_value(absorption_echo_params.get('final_fusion_exponent'), 0.25)
        context_modulator_weights = get_param_value(absorption_echo_params.get('context_modulator_weights'), {})
        supportive_distribution_penalty_factor = get_param_value(absorption_echo_params.get('supportive_distribution_penalty_factor'), 0.2)
        panic_slope_dampening_enabled = get_param_value(absorption_echo_params.get('panic_slope_dampening_enabled'), True)
        panic_slope_dampening_sensitivity = get_param_value(absorption_echo_params.get('panic_slope_dampening_sensitivity'), 0.5)
        deception_boost_factor_negative = get_param_value(absorption_echo_params.get('deception_boost_factor_negative'), 0.5)
        # --- 调试信息构建 ---
        is_debug_enabled = self.should_probe
        probe_ts = None
        if is_debug_enabled and self.probe_dates_set:
            for date in reversed(df_index):
                if date.date() in self.probe_dates_set:
                    probe_ts = date
                    break
        debug_info_tuple = (is_debug_enabled, probe_ts, method_name)
        # --- 原始数据获取 ---
        retail_panic_surrender_raw = signals_data['retail_panic_surrender_index_D']
        loser_pain_raw = signals_data['loser_pain_index_D']
        chip_fatigue_raw = signals_data['chip_fatigue_index_D']
        structural_tension_raw = signals_data['structural_tension_index_D']
        panic_selling_cascade_raw = signals_data['panic_selling_cascade_D']
        total_loser_rate_raw = signals_data['total_loser_rate_D']
        loser_loss_margin_avg_raw = signals_data['loser_loss_margin_avg_D']
        slope_5_loser_pain_raw = signals_data['SLOPE_5_loser_pain_index_D']
        accel_5_chip_fatigue_raw = signals_data['ACCEL_5_chip_fatigue_index_D']
        volatility_instability_raw = signals_data['VOLATILITY_INSTABILITY_INDEX_21d_D']
        accel_5_total_loser_rate_raw = signals_data['ACCEL_5_total_loser_rate_D']
        slope_5_retail_panic_surrender_raw = signals_data['SLOPE_5_retail_panic_surrender_index_D']
        accel_5_structural_tension_raw = signals_data['ACCEL_5_structural_tension_index_D']
        divergence_bullish_raw = divergence_scores # 从参数传入
        capitulation_absorption_raw = signals_data['capitulation_absorption_index_D']
        cleansing_efficiency_raw = signals_data['floating_chip_cleansing_efficiency_D']
        support_validation_raw = signals_data['support_validation_strength_D']
        main_force_execution_alpha_raw = signals_data['main_force_execution_alpha_D']
        active_buying_support_raw = signals_data['active_buying_support_D']
        opening_gap_defense_strength_raw = signals_data['opening_gap_defense_strength_D']
        control_solidity_raw = signals_data['control_solidity_index_D']
        slope_5_support_validation_raw = signals_data['SLOPE_5_support_validation_strength_D']
        accel_5_main_force_execution_alpha_raw = signals_data['ACCEL_5_main_force_execution_alpha_D']
        order_book_clearing_rate_raw = signals_data['order_book_clearing_rate_D']
        slope_5_floating_chip_cleansing_raw = signals_data['SLOPE_5_floating_chip_cleansing_efficiency_D']
        accel_5_order_book_clearing_raw = signals_data['ACCEL_5_order_book_clearing_rate_D']
        micro_price_impact_asymmetry_raw = signals_data['micro_price_impact_asymmetry_D']
        slope_5_micro_price_impact_asymmetry_raw = signals_data['SLOPE_5_micro_price_impact_asymmetry_D']
        vwap_control_strength_raw = signals_data['vwap_control_strength_D']
        slope_5_vwap_control_strength_raw = signals_data['SLOPE_5_vwap_control_strength_D']
        vwap_crossing_intensity_raw = signals_data['vwap_crossing_intensity_D']
        accel_5_vwap_crossing_intensity_raw = signals_data['ACCEL_5_vwap_crossing_intensity_D']
        covert_accumulation_raw = signals_data['covert_accumulation_signal_D']
        suppressive_accumulation_raw = signals_data['suppressive_accumulation_intensity_D']
        main_force_cost_advantage_raw = signals_data['main_force_cost_advantage_D']
        peak_control_transfer_raw = signals_data['peak_control_transfer_D']
        main_force_conviction_raw = signals_data['main_force_conviction_index_D']
        main_force_net_flow_calibrated_raw = signals_data['main_force_net_flow_calibrated_D']
        main_force_flow_directionality_raw = signals_data['main_force_flow_directionality_D']
        main_force_vpoc_raw = signals_data['main_force_vpoc_D']
        main_force_activity_ratio_raw = signals_data['main_force_activity_ratio_D']
        slope_5_covert_accumulation_raw = signals_data['SLOPE_5_covert_accumulation_signal_D']
        accel_5_main_force_conviction_raw = signals_data['ACCEL_5_main_force_conviction_index_D']
        smart_money_net_buy_raw = signals_data['SMART_MONEY_HM_NET_BUY_D']
        accel_5_covert_accumulation_raw = signals_data['ACCEL_5_covert_accumulation_signal_D']
        slope_5_suppressive_accumulation_raw = signals_data['SLOPE_5_suppressive_accumulation_intensity_D']
        accel_5_main_force_cost_advantage_raw = signals_data['ACCEL_5_main_force_cost_advantage_D']
        slope_5_main_force_flow_directionality_raw = signals_data['SLOPE_5_main_force_flow_directionality_D']
        accel_5_main_force_vpoc_raw = signals_data['ACCEL_5_main_force_vpoc_D']
        slope_5_smart_money_net_buy_raw = signals_data['SLOPE_5_SMART_MONEY_HM_NET_BUY_D']
        chip_fault_magnitude_raw = signals_data['chip_fault_magnitude_D']
        deception_index_raw = signals_data['deception_index_D']
        wash_trade_intensity_raw = signals_data['wash_trade_intensity_D']
        chip_health_score_raw = signals_data['chip_health_score_D']
        flow_credibility_raw = signals_data['flow_credibility_index_D']
        supportive_distribution_intensity_raw = signals_data['supportive_distribution_intensity_D']
        # --- 1. 恐慌声源 (Panic Source) ---
        norm_retail_panic_surrender = get_adaptive_mtf_normalized_score(retail_panic_surrender_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_loser_pain = get_adaptive_mtf_normalized_score(loser_pain_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_chip_fatigue = get_adaptive_mtf_normalized_score(chip_fatigue_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_structural_tension_negative = get_adaptive_mtf_normalized_score(structural_tension_raw, df_index, ascending=False, tf_weights=tf_weights) # 结构紧张度低，恐慌声源强
        norm_panic_selling_cascade = get_adaptive_mtf_normalized_score(panic_selling_cascade_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_total_loser_rate = get_adaptive_mtf_normalized_score(total_loser_rate_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_loser_loss_margin_avg = get_adaptive_mtf_normalized_score(loser_loss_margin_avg_raw, df_index, ascending=False, tf_weights=tf_weights) # 亏损幅度小，恐慌声源强
        norm_slope_5_loser_pain = get_adaptive_mtf_normalized_score(slope_5_loser_pain_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_accel_5_chip_fatigue = get_adaptive_mtf_normalized_score(accel_5_chip_fatigue_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_volatility_instability = get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_accel_5_total_loser_rate = get_adaptive_mtf_normalized_score(accel_5_total_loser_rate_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_slope_5_retail_panic_surrender = get_adaptive_mtf_normalized_score(slope_5_retail_panic_surrender_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_accel_5_structural_tension = get_adaptive_mtf_normalized_score(accel_5_structural_tension_raw, df_index, ascending=True, tf_weights=tf_weights)
        panic_source_numeric_weights = {k: v for k, v in panic_source_weights.items() if isinstance(v, (int, float))}
        panic_source_score = _robust_geometric_mean(
            {
                'retail_panic_surrender': norm_retail_panic_surrender,
                'loser_pain': norm_loser_pain,
                'chip_fatigue': norm_chip_fatigue,
                'structural_tension_negative': norm_structural_tension_negative,
                'panic_selling_cascade': norm_panic_selling_cascade,
                'total_loser_rate': norm_total_loser_rate,
                'loser_loss_margin_avg': norm_loser_loss_margin_avg,
                'loser_pain_slope': norm_slope_5_loser_pain,
                'chip_fatigue_accel': norm_accel_5_chip_fatigue,
                'volatility_instability': norm_volatility_instability,
                'total_loser_rate_accel': norm_accel_5_total_loser_rate,
                'retail_panic_surrender_slope': norm_slope_5_retail_panic_surrender,
                'structural_tension_accel': norm_accel_5_structural_tension
            },
            panic_source_numeric_weights, df_index
        )
        panic_slope_dampening_factor = pd.Series(1.0, index=df_index)
        if panic_slope_dampening_enabled:
            norm_panic_slope = get_adaptive_mtf_normalized_bipolar_score(slope_5_retail_panic_surrender_raw, df_index, tf_weights)
            panic_slope_dampening_factor = (1 - norm_panic_slope.clip(upper=0).abs() * panic_slope_dampening_sensitivity).clip(0.1, 1.0)
        panic_source_score_modulated = panic_source_score * panic_slope_dampening_factor
        is_panic_context = panic_source_score_modulated > panic_context_threshold
        # --- 2. 逆流介质 (Counter-Flow Medium) ---
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
        norm_micro_price_impact_asymmetry = get_adaptive_mtf_normalized_score(micro_price_impact_asymmetry_raw, df_index, ascending=False, tf_weights=tf_weights) # 不对称性低，承接力强
        norm_slope_5_micro_price_impact_asymmetry = get_adaptive_mtf_normalized_score(slope_5_micro_price_impact_asymmetry_raw, df_index, ascending=False, tf_weights=tf_weights)
        norm_vwap_control_strength = get_adaptive_mtf_normalized_score(vwap_control_strength_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_slope_5_vwap_control_strength = get_adaptive_mtf_normalized_score(slope_5_vwap_control_strength_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_vwap_crossing_intensity = get_adaptive_mtf_normalized_score(vwap_crossing_intensity_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_accel_5_vwap_crossing_intensity = get_adaptive_mtf_normalized_score(accel_5_vwap_crossing_intensity_raw, df_index, ascending=True, tf_weights=tf_weights)
        counter_flow_medium_numeric_weights = {k: v for k, v in counter_flow_medium_weights.items() if isinstance(v, (int, float))}
        counter_flow_medium_score = _robust_geometric_mean(
            {
                'divergence_bullish': norm_divergence_bullish,
                'capitulation_absorption': norm_capitulation_absorption,
                'cleansing_efficiency': norm_cleansing_efficiency,
                'support_validation': norm_support_validation,
                'main_force_execution_alpha': norm_main_force_execution_alpha,
                'active_buying_support': norm_active_buying_support,
                'opening_gap_defense_strength': norm_opening_gap_defense_strength,
                'control_solidity': norm_control_solidity,
                'support_validation_slope': norm_support_validation_slope,
                'main_force_execution_alpha_accel': norm_main_force_execution_alpha_accel,
                'order_book_clearing_rate': norm_order_book_clearing_rate,
                'cleansing_efficiency_slope': norm_slope_5_floating_chip_cleansing,
                'order_book_clearing_accel': norm_accel_5_order_book_clearing,
                'micro_impact_asymmetry': norm_micro_price_impact_asymmetry,
                'micro_impact_asymmetry_slope': norm_slope_5_micro_price_impact_asymmetry,
                'vwap_control_strength': norm_vwap_control_strength,
                'vwap_control_strength_slope': norm_slope_5_vwap_control_strength,
                'vwap_crossing_intensity': norm_vwap_crossing_intensity,
                'vwap_crossing_intensity_accel': norm_accel_5_vwap_crossing_intensity
            },
            counter_flow_medium_numeric_weights, df_index
        )
        # --- 3. 主力回声 (Main Force Echo) ---
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
        main_force_echo_score = _robust_geometric_mean(
            {
                'covert_accumulation': norm_covert_accumulation,
                'suppressive_accumulation': norm_suppressive_accumulation,
                'cost_advantage': norm_main_force_cost_advantage,
                'peak_control_transfer': norm_peak_control_transfer,
                'main_force_conviction_positive': norm_main_force_conviction_positive,
                'main_force_net_flow_positive': norm_main_force_net_flow_positive,
                'main_force_flow_directionality_positive': norm_main_force_flow_directionality_positive,
                'main_force_vpoc': norm_main_force_vpoc,
                'main_force_activity_ratio': norm_main_force_activity_ratio,
                'covert_accumulation_slope': norm_slope_5_covert_accumulation,
                'main_force_conviction_accel': norm_accel_5_main_force_conviction,
                'smart_money_net_buy_positive': norm_smart_money_net_buy_positive,
                'covert_accumulation_accel': norm_accel_5_covert_accumulation,
                'suppressive_accumulation_slope': norm_slope_5_suppressive_accumulation,
                'cost_advantage_accel': norm_accel_5_main_force_cost_advantage,
                'flow_directionality_slope': norm_slope_5_main_force_flow_directionality,
                'main_force_vpoc_accel': norm_accel_5_main_force_vpoc,
                'smart_money_net_buy_slope': norm_slope_5_smart_money_net_buy
            },
            main_force_echo_numeric_weights, df_index
        )
        # --- 4. 诡道背景调制智能化 (Intelligent Deception Context Modulation) ---
        deception_modulator = pd.Series(1.0, index=df_index)
        norm_chip_fault_magnitude_bipolar = get_adaptive_mtf_normalized_bipolar_score(chip_fault_magnitude_raw, df_index, tf_weights)
        norm_deception_index_bipolar = get_adaptive_mtf_normalized_bipolar_score(deception_index_raw, df_index, tf_weights)
        norm_wash_trade_intensity = get_adaptive_mtf_normalized_score(wash_trade_intensity_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_main_force_conviction_bipolar = get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, tf_weights)
        norm_supportive_distribution_intensity = get_adaptive_mtf_normalized_score(supportive_distribution_intensity_raw, df_index, ascending=True, tf_weights=tf_weights)
        conviction_threshold = deception_modulator_params.get('conviction_threshold', 0.2)
        deception_index_boost_weight = deception_modulator_params.get('deception_index_boost_weight', 0.5)
        deception_index_penalty_weight = deception_modulator_params.get('deception_index_penalty_weight', 0.7) # 提高敏感度
        wash_trade_penalty_weight = deception_modulator_params.get('wash_trade_penalty_weight', 0.3)
        # 修正：诱空反吸增强：负向欺骗时增强吸筹信号
        deception_boost_mask = (norm_deception_index_bipolar < 0)
        deception_modulator.loc[deception_boost_mask] = deception_modulator.loc[deception_boost_mask] * (1 + norm_deception_index_bipolar.loc[deception_boost_mask].abs() * deception_boost_factor_negative)
        # 诱多惩罚：正向欺骗且主力信念弱
        deception_penalty_mask = (norm_deception_index_bipolar > 0) & (norm_main_force_conviction_bipolar < -conviction_threshold)
        deception_modulator.loc[deception_penalty_mask] = deception_modulator.loc[deception_penalty_mask] * (1 - norm_deception_index_bipolar.loc[deception_penalty_mask] * deception_index_penalty_weight)
        # 对倒惩罚
        deception_modulator = deception_modulator * (1 - norm_wash_trade_intensity * wash_trade_penalty_weight)
        # 筹码故障惩罚
        deception_modulator = deception_modulator * (1 - norm_chip_fault_magnitude_bipolar.clip(lower=0) * deception_modulator_params.get('penalty_factor', 0.4))
        # 支持性派发惩罚
        deception_modulator = deception_modulator * (1 - norm_supportive_distribution_intensity * supportive_distribution_penalty_factor)
        deception_modulator = deception_modulator.clip(0.1, 2.0)
        # --- 5. 情境调制器引入 (Context Modulator Introduction) ---
        norm_flow_credibility = get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_structural_tension = get_adaptive_mtf_normalized_score(structural_tension_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_chip_health_score = get_adaptive_mtf_normalized_score(chip_health_score_raw, df_index, ascending=True, tf_weights=tf_weights)
        context_modulator_numeric_weights = {k: v for k, v in context_modulator_weights.items() if isinstance(v, (int, float))}
        total_context_modulator_weight = sum(context_modulator_numeric_weights.values())
        context_modulator = pd.Series(1.0, index=df_index)
        if total_context_modulator_weight > 0:
            fused_context_modulator_raw = (
                norm_flow_credibility.pow(context_modulator_numeric_weights.get('flow_credibility', 0.4)) *
                (1 - norm_structural_tension).pow(context_modulator_numeric_weights.get('structural_tension_inverse', 0.3)) * # 结构紧张度低，情境有利
                norm_chip_health_score.pow(context_modulator_numeric_weights.get('chip_health', 0.3))
            ).pow(1 / total_context_modulator_weight)
            context_modulator = 1 + (fused_context_modulator_raw - 0.5) * 0.5 # 将 [0,1] 映射到 [0.75, 1.25] 左右
        context_modulator = context_modulator.clip(0.5, 1.5)
        # --- 最终融合 ---
        base_score = pd.Series(0.0, index=df_index)
        valid_mask = is_panic_context # 只有在恐慌背景下才计算吸筹回声
        if valid_mask.any():
            base_score.loc[valid_mask] = (
                counter_flow_medium_score.loc[valid_mask].pow(0.5) *
                main_force_echo_score.loc[valid_mask].pow(0.5)
            )
        final_score = base_score * deception_modulator * context_modulator
        final_score = final_score.pow(final_fusion_exponent) # 非线性放大
        # --- 应用牛市陷阱情境惩罚 ---
        bull_trap_penalty = self._calculate_bull_trap_context_penalty(df)
        final_score = final_score * bull_trap_penalty
        final_score = final_score.clip(0.0, 1.0).fillna(0.0).astype(np.float32)
        print(f"    -> [筹码层] 计算完成 '吸筹回声探针' 分数: {final_score.loc[probe_ts] if probe_ts and probe_ts in df_index else final_score.iloc[-1]:.4f}")
        return final_score

    def _diagnose_distribution_whisper(self, df: pd.DataFrame, divergence_score: pd.Series) -> pd.Series:
        """
        【V4.0 · 深度高频诡道派发版】诊断“派发诡影”信号
        - 核心升级1: 狂热背景深度化。在V3.0基础上，引入总赢家比例、总输家比例、赢家输家动量及其短期斜率，更全面刻画市场狂热和筹码结构膨胀。
        - 核心升级2: 背离诡影精细化。在V3.0基础上，引入主峰利润率、主峰坚实度、上影线抛压、压力拒绝强度及其短期斜率，评估主力派发动力、筹码结构松动和承接力减弱。
        - 核心升级3: 主力抽离多维度验证。在V3.0基础上，引入主力净流量校准、主力滑点指数及其短期加速度、反弹派发压力、控制坚实度、对手盘枯竭和智能资金净买入负向，多角度验证主力隐蔽、坚决派发。
        - 核心升级4: 诡道背景调制强化。引入欺骗指数，结合筹码故障幅度与主力信念指数，更智能地判断诡道意图并进行调制。
        - 探针增强: 详细输出所有原始数据、归一化数据、各维度子分数、动态权重、最终分数，以便于检查和调试。
        """
        df_index = df.index
        required_signals = [
            'retail_fomo_premium_index_D', 'winner_profit_margin_avg_D', 'THEME_HOTNESS_SCORE_D', 'market_sentiment_score_D', 'winner_concentration_90pct_D',
            'total_winner_rate_D', 'winner_loser_momentum_D', 'SLOPE_5_winner_loser_momentum_D',
            'dispersal_by_distribution_D', 'profit_taking_flow_ratio_D', 'chip_fault_magnitude_D',
            'cost_structure_skewness_D', 'winner_stability_index_D', 'chip_fault_blockage_ratio_D',
            'dominant_peak_profit_margin_D', 'dominant_peak_solidity_D', 'upper_shadow_selling_pressure_D', 'pressure_rejection_strength_D',
            'SLOPE_5_dominant_peak_solidity_D', 'SLOPE_5_pressure_rejection_strength_D',
            'covert_accumulation_signal_D', 'wash_trade_intensity_D', 'main_force_conviction_index_D', 'retail_flow_dominance_index_D',
            'main_force_net_flow_calibrated_D', 'main_force_slippage_index_D', 'rally_distribution_pressure_D', 'control_solidity_index_D',
            'counterparty_exhaustion_index_D', 'SMART_MONEY_HM_NET_BUY_D',
            'SLOPE_5_main_force_net_flow_calibrated_D', 'ACCEL_5_main_force_slippage_index_D',
            'deception_index_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_distribution_whisper"):
            return pd.Series(0.0, index=df.index)
        signals_data = self._get_all_required_signals(df, required_signals, "_diagnose_distribution_whisper")
        p_conf = self.chip_ultimate_params
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        distribution_whisper_params = get_param_value(p_conf.get('distribution_whisper_params'), {})
        fomo_backdrop_weights = get_param_value(distribution_whisper_params.get('fomo_backdrop_weights'), {'retail_fomo_premium': 0.2, 'winner_profit_margin': 0.2, 'theme_hotness': 0.15, 'market_sentiment_positive': 0.1, 'winner_concentration_negative': 0.1, 'total_winner_rate': 0.15, 'winner_loser_momentum': 0.05, 'winner_loser_momentum_slope': 0.05})
        fomo_context_threshold = get_param_value(distribution_whisper_params.get('fomo_context_threshold'), 0.3)
        divergence_shadow_weights = get_param_value(distribution_whisper_params.get('divergence_shadow_weights'), {'divergence_bearish': 0.15, 'distribution_intensity': 0.15, 'chip_fault_magnitude': 0.1, 'cost_structure_negative': 0.1, 'winner_stability_negative': 0.1, 'chip_fault_blockage': 0.1, 'dominant_peak_profit_margin': 0.1, 'dominant_peak_solidity_negative': 0.05, 'upper_shadow_selling_pressure': 0.05, 'pressure_rejection_strength_negative': 0.05, 'dominant_peak_solidity_slope_negative': 0.025, 'pressure_rejection_strength_slope_negative': 0.025})
        main_force_retreat_weights = get_param_value(distribution_whisper_params.get('main_force_retreat_weights'), {'profit_taking_flow': 0.15, 'dispersal_by_distribution': 0.15, 'covert_accumulation_negative': 0.1, 'wash_trade_intensity': 0.1, 'main_force_conviction_negative': 0.1, 'retail_flow_dominance': 0.1, 'main_force_net_flow_negative': 0.1, 'main_force_slippage': 0.05, 'rally_distribution_pressure': 0.05, 'control_solidity_negative': 0.05, 'counterparty_exhaustion': 0.025, 'smart_money_net_buy_negative': 0.025, 'main_force_net_flow_slope_negative': 0.025, 'main_force_slippage_accel': 0.025})
        deception_modulator_params = get_param_value(distribution_whisper_params.get('deception_modulator_params'), {'boost_factor': 0.6, 'penalty_factor': 0.4, 'conviction_threshold': 0.2, 'deception_index_weight': 0.5})
        final_fusion_exponent = get_param_value(distribution_whisper_params.get('final_fusion_exponent'), 0.25)
        # --- 调试信息构建 ---
        is_debug_enabled = self.should_probe
        probe_ts = None
        if is_debug_enabled and self.probe_dates_set:
            for date in reversed(df_index):
                if date.date() in self.probe_dates_set:
                    probe_ts = date
                    break
        debug_info_tuple = False # (is_debug_enabled, probe_ts, "_diagnose_distribution_whisper")
        # --- 原始数据获取 ---
        retail_fomo_premium_raw = signals_data['retail_fomo_premium_index_D']
        winner_profit_margin_raw = signals_data['winner_profit_margin_avg_D']
        theme_hotness_raw = signals_data['THEME_HOTNESS_SCORE_D']
        market_sentiment_raw = signals_data['market_sentiment_score_D']
        winner_concentration_raw = signals_data['winner_concentration_90pct_D']
        total_winner_rate_raw = signals_data['total_winner_rate_D']
        winner_loser_momentum_raw = signals_data['winner_loser_momentum_D']
        slope_5_winner_loser_momentum_raw = signals_data['SLOPE_5_winner_loser_momentum_D']
        dispersal_by_distribution_raw = signals_data['dispersal_by_distribution_D']
        chip_fault_magnitude_raw = signals_data['chip_fault_magnitude_D']
        cost_structure_skewness_raw = signals_data['cost_structure_skewness_D']
        winner_stability_raw = signals_data['winner_stability_index_D']
        chip_fault_blockage_raw = signals_data['chip_fault_blockage_ratio_D']
        dominant_peak_profit_margin_raw = signals_data['dominant_peak_profit_margin_D']
        dominant_peak_solidity_raw = signals_data['dominant_peak_solidity_D']
        upper_shadow_selling_pressure_raw = signals_data['upper_shadow_selling_pressure_D']
        pressure_rejection_strength_raw = signals_data['pressure_rejection_strength_D']
        slope_5_dominant_peak_solidity_raw = signals_data['SLOPE_5_dominant_peak_solidity_D']
        slope_5_pressure_rejection_strength_raw = signals_data['SLOPE_5_pressure_rejection_strength_D']
        profit_taking_flow_ratio_raw = signals_data['profit_taking_flow_ratio_D']
        covert_accumulation_raw = signals_data['covert_accumulation_signal_D']
        wash_trade_intensity_raw = signals_data['wash_trade_intensity_D']
        main_force_conviction_raw = signals_data['main_force_conviction_index_D']
        retail_flow_dominance_raw = signals_data['retail_flow_dominance_index_D']
        main_force_net_flow_calibrated_raw = signals_data['main_force_net_flow_calibrated_D']
        main_force_slippage_raw = signals_data['main_force_slippage_index_D']
        rally_distribution_pressure_raw = signals_data['rally_distribution_pressure_D']
        control_solidity_raw = signals_data['control_solidity_index_D']
        counterparty_exhaustion_raw = signals_data['counterparty_exhaustion_index_D']
        smart_money_net_buy_raw = signals_data['SMART_MONEY_HM_NET_BUY_D']
        slope_5_main_force_net_flow_calibrated_raw = signals_data['SLOPE_5_main_force_net_flow_calibrated_D']
        accel_5_main_force_slippage_raw = signals_data['ACCEL_5_main_force_slippage_index_D']
        deception_index_raw = signals_data['deception_index_D']
        # --- 1. 狂热背景 (FOMO Backdrop) ---
        norm_retail_fomo_premium = get_adaptive_mtf_normalized_score(retail_fomo_premium_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_winner_profit_margin = get_adaptive_mtf_normalized_score(winner_profit_margin_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_theme_hotness = get_adaptive_mtf_normalized_score(theme_hotness_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_market_sentiment_positive = get_adaptive_mtf_normalized_bipolar_score(market_sentiment_raw, df_index, tf_weights).clip(0, 1)
        norm_winner_concentration_negative = get_adaptive_mtf_normalized_score(winner_concentration_raw, df_index, ascending=False, tf_weights=tf_weights)
        norm_total_winner_rate = get_adaptive_mtf_normalized_score(total_winner_rate_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_winner_loser_momentum = get_adaptive_mtf_normalized_score(winner_loser_momentum_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_slope_5_winner_loser_momentum = get_adaptive_mtf_normalized_score(slope_5_winner_loser_momentum_raw, df_index, ascending=True, tf_weights=tf_weights)
        fomo_backdrop_numeric_weights = {k: v for k, v in fomo_backdrop_weights.items() if isinstance(v, (int, float))}
        fomo_backdrop_score = _robust_geometric_mean(
            {
                'retail_fomo_premium': norm_retail_fomo_premium,
                'winner_profit_margin': norm_winner_profit_margin,
                'theme_hotness': norm_theme_hotness,
                'market_sentiment_positive': norm_market_sentiment_positive,
                'winner_concentration_negative': norm_winner_concentration_negative,
                'total_winner_rate': norm_total_winner_rate,
                'winner_loser_momentum': norm_winner_loser_momentum,
                'winner_loser_momentum_slope': norm_slope_5_winner_loser_momentum
            },
            fomo_backdrop_numeric_weights, df_index
        )
        is_fomo_context = fomo_backdrop_score > fomo_context_threshold
        # --- 2. 背离诡影 (Divergence Shadow) ---
        norm_divergence_bearish = divergence_score.clip(-1, 0).abs() # 取负向背离的绝对值
        norm_dispersal_by_distribution = get_adaptive_mtf_normalized_score(dispersal_by_distribution_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_chip_fault_magnitude_for_shadow = get_adaptive_mtf_normalized_score(chip_fault_magnitude_raw.abs(), df_index, ascending=True, tf_weights=tf_weights)
        norm_cost_structure_negative = get_adaptive_mtf_normalized_score(cost_structure_skewness_raw, df_index, ascending=False, tf_weights=tf_weights)
        norm_winner_stability_negative = get_adaptive_mtf_normalized_score(winner_stability_raw, df_index, ascending=False, tf_weights=tf_weights)
        norm_chip_fault_blockage = get_adaptive_mtf_normalized_score(chip_fault_blockage_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_dominant_peak_profit_margin = get_adaptive_mtf_normalized_score(dominant_peak_profit_margin_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_dominant_peak_solidity_negative = get_adaptive_mtf_normalized_score(dominant_peak_solidity_raw, df_index, ascending=False, tf_weights=tf_weights)
        norm_upper_shadow_selling_pressure = get_adaptive_mtf_normalized_score(upper_shadow_selling_pressure_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_pressure_rejection_strength_negative = get_adaptive_mtf_normalized_score(pressure_rejection_strength_raw, df_index, ascending=False, tf_weights=tf_weights)
        norm_slope_5_dominant_peak_solidity_negative = get_adaptive_mtf_normalized_score(slope_5_dominant_peak_solidity_raw, df_index, ascending=False, tf_weights=tf_weights)
        norm_slope_5_pressure_rejection_strength_negative = get_adaptive_mtf_normalized_score(slope_5_pressure_rejection_strength_raw, df_index, ascending=False, tf_weights=tf_weights)
        divergence_shadow_numeric_weights = {k: v for k, v in divergence_shadow_weights.items() if isinstance(v, (int, float))}
        divergence_shadow_score = _robust_geometric_mean(
            {
                'divergence_bearish': norm_divergence_bearish,
                'distribution_intensity': norm_dispersal_by_distribution,
                'chip_fault_magnitude': norm_chip_fault_magnitude_for_shadow,
                'cost_structure_negative': norm_cost_structure_negative,
                'winner_stability_negative': norm_winner_stability_negative,
                'chip_fault_blockage': norm_chip_fault_blockage,
                'dominant_peak_profit_margin': norm_dominant_peak_profit_margin,
                'dominant_peak_solidity_negative': norm_dominant_peak_solidity_negative,
                'upper_shadow_selling_pressure': norm_upper_shadow_selling_pressure,
                'pressure_rejection_strength_negative': norm_pressure_rejection_strength_negative,
                'dominant_peak_solidity_slope_negative': norm_slope_5_dominant_peak_solidity_negative,
                'pressure_rejection_strength_slope_negative': norm_slope_5_pressure_rejection_strength_negative
            },
            divergence_shadow_numeric_weights, df_index
        )
        # --- 3. 主力抽离 (Main Force Retreat) ---
        norm_profit_taking_flow_ratio = get_adaptive_mtf_normalized_score(profit_taking_flow_ratio_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_covert_accumulation_negative = get_adaptive_mtf_normalized_score(covert_accumulation_raw, df_index, ascending=False, tf_weights=tf_weights)
        norm_wash_trade_intensity = get_adaptive_mtf_normalized_score(wash_trade_intensity_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_main_force_conviction_negative = get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, tf_weights).clip(-1, 0).abs()
        norm_retail_flow_dominance = get_adaptive_mtf_normalized_score(retail_flow_dominance_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_main_force_net_flow_negative = get_adaptive_mtf_normalized_score(main_force_net_flow_calibrated_raw, df_index, ascending=False, tf_weights=tf_weights)
        norm_main_force_slippage = get_adaptive_mtf_normalized_score(main_force_slippage_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_rally_distribution_pressure = get_adaptive_mtf_normalized_score(rally_distribution_pressure_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_control_solidity_negative = get_adaptive_mtf_normalized_score(control_solidity_raw, df_index, ascending=False, tf_weights=tf_weights)
        norm_counterparty_exhaustion = get_adaptive_mtf_normalized_score(counterparty_exhaustion_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_smart_money_net_buy_negative = get_adaptive_mtf_normalized_score(smart_money_net_buy_raw, df_index, ascending=False, tf_weights=tf_weights)
        norm_slope_5_main_force_net_flow_negative = get_adaptive_mtf_normalized_score(slope_5_main_force_net_flow_calibrated_raw, df_index, ascending=False, tf_weights=tf_weights)
        norm_accel_5_main_force_slippage = get_adaptive_mtf_normalized_score(accel_5_main_force_slippage_raw, df_index, ascending=True, tf_weights=tf_weights)
        main_force_retreat_numeric_weights = {k: v for k, v in main_force_retreat_weights.items() if isinstance(v, (int, float))}
        main_force_retreat_score = _robust_geometric_mean(
            {
                'profit_taking_flow': norm_profit_taking_flow_ratio,
                'dispersal_by_distribution': norm_dispersal_by_distribution,
                'covert_accumulation_negative': norm_covert_accumulation_negative,
                'wash_trade_intensity': norm_wash_trade_intensity,
                'main_force_conviction_negative': norm_main_force_conviction_negative,
                'retail_flow_dominance': norm_retail_flow_dominance,
                'main_force_net_flow_negative': norm_main_force_net_flow_negative,
                'main_force_slippage': norm_main_force_slippage,
                'rally_distribution_pressure': norm_rally_distribution_pressure,
                'control_solidity_negative': norm_control_solidity_negative,
                'counterparty_exhaustion': norm_counterparty_exhaustion,
                'smart_money_net_buy_negative': norm_smart_money_net_buy_negative,
                'main_force_net_flow_slope_negative': norm_slope_5_main_force_net_flow_negative,
                'main_force_slippage_accel': norm_accel_5_main_force_slippage
            },
            main_force_retreat_numeric_weights, df_index
        )
        # --- 4. 诡道背景调制强化 (Deception Context Modulation Reinforcement) ---
        norm_chip_fault_magnitude_bipolar = get_adaptive_mtf_normalized_bipolar_score(chip_fault_magnitude_raw, df_index, tf_weights)
        norm_main_force_conviction_bipolar = get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, tf_weights)
        norm_deception_index_bipolar = get_adaptive_mtf_normalized_bipolar_score(deception_index_raw, df_index, tf_weights)
        deception_modulator = pd.Series(1.0, index=df_index)
        conviction_threshold = deception_modulator_params.get('conviction_threshold', 0.2)
        # 欺骗性看涨（主力诱多）且主力信念弱时，放大派发风险
        deceptive_bullish_and_weak_conviction_mask = (norm_chip_fault_magnitude_bipolar > 0) & \
                                                     (norm_main_force_conviction_bipolar < -conviction_threshold)
        deception_modulator.loc[deceptive_bullish_and_weak_conviction_mask] = 1 + norm_chip_fault_magnitude_bipolar.loc[deceptive_bullish_and_weak_conviction_mask] * deception_modulator_params.get('boost_factor', 0.6)
        # 欺骗性看跌（主力诱空）且主力信念强时，减弱派发风险（可能是洗盘）
        induced_panic_and_conviction_mask = (norm_chip_fault_magnitude_bipolar < 0) & \
                                            (norm_main_force_conviction_bipolar > conviction_threshold)
        deception_modulator.loc[induced_panic_and_conviction_mask] = 1 - norm_chip_fault_magnitude_bipolar.loc[induced_panic_and_conviction_mask].abs() * deception_modulator_params.get('penalty_factor', 0.4)
        # 欺骗指数正向且主力信念弱时，放大派发风险
        deception_index_boost_mask = (norm_deception_index_bipolar > 0) & \
                                     (norm_main_force_conviction_bipolar < -conviction_threshold)
        deception_modulator.loc[deception_index_boost_mask] = deception_modulator.loc[deception_index_boost_mask] + \
                                                              norm_deception_index_bipolar.loc[deception_index_boost_mask] * deception_modulator_params.get('deception_index_weight', 0.5)
        # 欺骗指数负向且主力信念强时，减弱派发风险
        deception_index_penalty_mask = (norm_deception_index_bipolar < 0) & \
                                       (norm_main_force_conviction_bipolar > conviction_threshold)
        deception_modulator.loc[deception_index_penalty_mask] = deception_modulator.loc[deception_index_penalty_mask] - \
                                                                norm_deception_index_bipolar.loc[deception_index_penalty_mask].abs() * deception_modulator_params.get('deception_index_weight', 0.5)
        deception_modulator = deception_modulator.clip(0.1, 2.0)
        # --- 最终融合 ---
        base_score = (
            fomo_backdrop_score.pow(final_fusion_exponent) *
            divergence_shadow_score.pow(final_fusion_exponent) *
            main_force_retreat_score.pow(final_fusion_exponent)
        ).pow(1 / (3 * final_fusion_exponent)) # 几何平均
        final_score = (base_score * deception_modulator) * is_fomo_context # 只有在FOMO背景下才计算派发诡影
        final_score = final_score.clip(0, 1).fillna(0.0).astype(np.float32)
        print(f"    -> [筹码层] 计算完成 '“派发诡影”信号' 分数: {final_score.iloc[-1]}")
        return final_score

    def _diagnose_axiom_historical_potential(self, df: pd.DataFrame) -> pd.Series:
        """
        【V5.8 · 势能博弈临界强化与诡道惩罚深度修正版 & 牛市陷阱惩罚版】筹码公理六：诊断“筹码势能”
        - 核心升级1: 主力吸筹质量 (MF_AQ)。引入“吸筹效率的非对称性”，结合主力成本优势和筹码健康度动态调整吸筹模式权重，并考虑主力执行效率和非对称摩擦指数等高频聚合信号。
        - 核心升级2: 筹码结构张力 (CST)。引入“结构临界点识别”，结合赢家/输家集中度斜率预判结构转折，并考虑结构张力指数和结构熵变。
        - 核心升级3: 势能转化效率 (PCE)。引入“阻力位博弈强度”，评估关键阻力位和支撑位的博弈激烈程度，并考虑订单簿清算率和微观价格冲击不对称性等微观层面的阻力消化能力。
        - 核心升级4: 诡道博弈调制 (DGM)。引入“诡道博弈的非对称影响”，对诱多/诱空施加不同敏感度的调制，并考虑散户恐慌和主力信念对诡道博弈有效性的影响。
        - 核心升级5: 情境自适应权重 (ACW)。引入“市场情绪与流动性情境”，增加市场情绪分数和资金流可信度指数作为情境调制器。
        - 核心修复: 增强 `dgm_score` 的惩罚机制，当出现“诱空”且主力资金流出时，强制将分数设置为极低负值，并确保其优先级，修正主力流出判断逻辑。
        - 纯筹码化：移除对资金类信号（如主力资金流向、资金流可信度）的依赖，替换为纯筹码类指标。
        - **新增业务逻辑：引入“牛市陷阱情境惩罚”，在近期大幅下跌后伴随正向欺骗时，大幅降低筹码势能的得分。**
        """
        method_name = "_diagnose_axiom_historical_potential"
        df_index = df.index
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
            'deception_index_D', 'wash_trade_intensity_D',
            'main_force_execution_alpha_D', 'asymmetric_friction_index_D',
            'SLOPE_5_winner_concentration_90pct_D', 'SLOPE_5_loser_concentration_90pct_D',
            'structural_tension_index_D', 'structural_entropy_change_D',
            'pressure_rejection_strength_D', 'support_validation_strength_D',
            'order_book_clearing_rate_D', 'micro_price_impact_asymmetry_D',
            'retail_panic_surrender_index_D', 'main_force_conviction_index_D',
            'market_sentiment_score_D',
            'conviction_flow_buy_intensity_D', 'conviction_flow_sell_intensity_D',
            'pct_change_D' # 新增牛市陷阱检测所需信号
        ]
        p_conf = self.chip_ultimate_params
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
            'bull_trap_penalty_factor': 1.5, 'bear_trap_bonus_factor': 1.2,
            'bull_trap_ll_penalty_factor': 0.5, 'bear_trap_ls_bonus_factor': 0.5
        })
        final_fusion_weights = get_param_value(historical_potential_params.get('final_fusion_weights'), {
            'mf_aq': 0.35, 'cst': 0.3, 'pce': 0.35
        })
        context_modulator_signals = get_param_value(historical_potential_params.get('context_modulator_signals'), {
            'volatility_instability': {'signal_name': 'VOLATILITY_INSTABILITY_INDEX_21d_D', 'weight': 0.3, 'ascending': False},
            'chip_fatigue': {'signal_name': 'chip_fatigue_index_D', 'weight': 0.2, 'ascending': False},
            'market_sentiment': {'signal_name': 'market_sentiment_score_D', 'weight': 0.3, 'ascending': True},
            'flow_credibility': {'signal_name': 'flow_credibility_index_D', 'weight': 0.2, 'ascending': True} # 待替换
        })
        context_modulator_sensitivity = get_param_value(historical_potential_params.get('context_modulator_sensitivity'), 0.5)
        dgm_modulator_sensitivity = get_param_value(historical_potential_params.get('dgm_modulator_sensitivity'), 0.8)
        bearish_deception_and_mf_out_penalty_factor = get_param_value(historical_potential_params.get('bearish_deception_and_mf_out_penalty_factor'), 1.0)
        # Ensure context modulator signals are in required_signals
        # 替换 flow_credibility_index_D 为 winner_stability_index_D
        context_modulator_signals['flow_credibility']['signal_name'] = 'winner_stability_index_D'
        context_modulator_signals['flow_credibility']['ascending'] = True # 赢家稳定性高，可信度高
        for ctx_key, ctx_config in context_modulator_signals.items():
            signal_name = ctx_config.get('signal_name')
            if signal_name and signal_name not in required_signals:
                required_signals.append(signal_name)
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index)
        signals_data = self._get_all_required_signals(df, required_signals, method_name)
        # --- 调试信息构建 ---
        is_debug_enabled = self.should_probe
        probe_ts = None
        if is_debug_enabled and self.probe_dates_set:
            for date in reversed(df_index):
                if date.date() in self.probe_dates_set:
                    probe_ts = date
                    break
        debug_info_tuple = (is_debug_enabled, probe_ts, method_name)
        # --- 原始数据获取 ---
        chip_health_raw = signals_data['chip_health_score_D']
        norm_chip_health = get_adaptive_mtf_normalized_bipolar_score(chip_health_raw, df_index, tf_weights)
        covert_accumulation_raw = signals_data['covert_accumulation_signal_D']
        suppressive_accumulation_raw = signals_data['suppressive_accumulation_intensity_D']
        main_force_cost_advantage_raw = signals_data['main_force_cost_advantage_D']
        floating_chip_cleansing_efficiency_raw = signals_data['floating_chip_cleansing_efficiency_D']
        chip_fault_magnitude_raw = signals_data['chip_fault_magnitude_D']
        main_force_execution_alpha_raw = signals_data['main_force_execution_alpha_D']
        asymmetric_friction_index_raw = signals_data['asymmetric_friction_index_D']
        conviction_flow_buy_raw = signals_data['conviction_flow_buy_intensity_D'] # 获取新增信号
        conviction_flow_sell_raw = signals_data['conviction_flow_sell_intensity_D'] # 获取新增信号
        winner_stability_raw = signals_data['winner_stability_index_D'] # 获取新增信号
        # --- 1. 主力吸筹质量 (MF_AQ) ---
        norm_covert_accumulation = get_adaptive_mtf_normalized_score(covert_accumulation_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_suppressive_accumulation = get_adaptive_mtf_normalized_score(suppressive_accumulation_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_main_force_cost_advantage = get_adaptive_mtf_normalized_bipolar_score(main_force_cost_advantage_raw, df_index, tf_weights)
        norm_floating_chip_cleansing_efficiency = get_adaptive_mtf_normalized_score(floating_chip_cleansing_efficiency_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_chip_fault_magnitude = get_adaptive_mtf_normalized_bipolar_score(chip_fault_magnitude_raw, df_index, tf_weights)
        norm_main_force_execution_alpha = get_adaptive_mtf_normalized_score(main_force_execution_alpha_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_asymmetric_friction_index = get_adaptive_mtf_normalized_score(asymmetric_friction_index_raw, df_index, ascending=False, tf_weights=tf_weights)
        deception_purity_adjustment = pd.Series(1.0, index=df_index)
        deception_purity_adjustment = 1 + (norm_chip_fault_magnitude * -1) * mf_aq_weights.get('deception_purity_factor', 0.1) # 筹码故障负向时，纯度增加
        deception_purity_adjustment = deception_purity_adjustment.clip(0.5, 1.5)
        dynamic_covert_weight = pd.Series(mf_aq_weights.get('covert_accumulation', 0.25), index=df_index)
        dynamic_suppressive_weight = pd.Series(mf_aq_weights.get('suppressive_accumulation', 0.15), index=df_index)
        # 吸筹效率的非对称性：当筹码健康度低且主力成本优势不明显时，隐蔽吸筹权重增加，压制吸筹权重降低
        low_health_low_cost_advantage_mask = (norm_chip_health < mf_aq_asymmetry_params.get('chip_health_threshold', 0.0)) & \
                                             (norm_main_force_cost_advantage < mf_aq_asymmetry_params.get('cost_advantage_threshold', 0.0))
        dynamic_covert_weight.loc[low_health_low_cost_advantage_mask] += mf_aq_asymmetry_params.get('covert_weight_boost', 0.2)
        dynamic_suppressive_weight.loc[low_health_low_cost_advantage_mask] -= mf_aq_asymmetry_params.get('suppressive_weight_boost', 0.1)
        # 重新计算总权重，确保加权平均的正确性
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
        ) / sum_dynamic_weights_mf_aq.replace(0, 1e-6) * base_mf_aq_total_weight # 确保总权重归一化
        mf_aq_score = mf_aq_score * deception_purity_adjustment # 诡道纯度调整
        mf_aq_score = mf_aq_score.clip(0, 1)
        # --- 2. 筹码结构张力 (CST) ---
        dominant_peak_solidity_raw = signals_data['dominant_peak_solidity_D']
        cost_structure_skewness_slope_raw = signals_data['SLOPE_5_cost_structure_skewness_D']
        peak_separation_ratio_slope_raw = signals_data['SLOPE_5_peak_separation_ratio_D']
        winner_stability_raw = signals_data['winner_stability_index_D']
        loser_pain_raw = signals_data['loser_pain_index_D']
        winner_concentration_slope_raw = signals_data['SLOPE_5_winner_concentration_90pct_D']
        loser_concentration_slope_raw = signals_data['SLOPE_5_loser_concentration_90pct_D']
        structural_tension_raw = signals_data['structural_tension_index_D']
        structural_entropy_change_raw = signals_data['structural_entropy_change_D']
        norm_dominant_peak_solidity = get_adaptive_mtf_normalized_score(dominant_peak_solidity_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_cost_structure_skewness_slope = get_adaptive_mtf_normalized_bipolar_score(cost_structure_skewness_slope_raw, df_index, tf_weights)
        norm_peak_separation_ratio_slope = get_adaptive_mtf_normalized_bipolar_score(peak_separation_ratio_slope_raw, df_index, tf_weights)
        norm_winner_stability = get_adaptive_mtf_normalized_score(winner_stability_raw, df_index, ascending=False, tf_weights=tf_weights) # 赢家稳定性低，结构张力高
        norm_loser_pain = get_adaptive_mtf_normalized_score(loser_pain_raw, df_index, ascending=True, tf_weights=tf_weights) # 输家痛苦高，结构张力高
        norm_winner_concentration_slope = get_adaptive_mtf_normalized_bipolar_score(winner_concentration_slope_raw, df_index, tf_weights)
        norm_loser_concentration_slope = get_adaptive_mtf_normalized_bipolar_score(loser_concentration_slope_raw, df_index, tf_weights)
        norm_structural_tension = get_adaptive_mtf_normalized_score(structural_tension_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_structural_entropy_change = get_adaptive_mtf_normalized_score(structural_entropy_change_raw, df_index, ascending=False, tf_weights=tf_weights) # 结构熵变低，结构张力高
        structural_elasticity_score = (norm_winner_stability * 0.5 + norm_loser_pain * 0.5).clip(0, 1)
        concentration_slope_divergence = (norm_winner_concentration_slope - norm_loser_concentration_slope).clip(-1, 1)
        cst_components = {
            'chip_health': (norm_chip_health + 1) / 2,
            'peak_solidity': norm_dominant_peak_solidity,
            'cost_skewness_slope': (1 - (norm_cost_structure_skewness_slope + 1) / 2), # 负向斜率代表结构张力
            'peak_separation_slope': (1 - (norm_peak_separation_ratio_slope + 1) / 2), # 负向斜率代表结构张力
            'structural_elasticity': structural_elasticity_score,
            'concentration_slope_divergence': (concentration_slope_divergence + 1) / 2,
            'structural_tension': norm_structural_tension,
            'structural_entropy': norm_structural_entropy_change
        }
        cst_score = _robust_geometric_mean(cst_components, cst_weights, df_index)
        cst_score = cst_score.clip(0, 1)
        # --- 3. 势能转化效率 (PCE) ---
        vacuum_zone_magnitude_raw = signals_data['vacuum_zone_magnitude_D']
        vacuum_traversal_efficiency_raw = signals_data['vacuum_traversal_efficiency_D']
        active_selling_pressure_raw = signals_data['active_selling_pressure_D']
        capitulation_absorption_raw = signals_data['capitulation_absorption_index_D']
        pressure_rejection_strength_raw = signals_data['pressure_rejection_strength_D']
        support_validation_strength_raw = signals_data['support_validation_strength_D']
        order_book_clearing_rate_raw = signals_data['order_book_clearing_rate_D']
        micro_price_impact_asymmetry_raw = signals_data['micro_price_impact_asymmetry_D']
        norm_vacuum_zone_magnitude = get_adaptive_mtf_normalized_score(vacuum_zone_magnitude_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_vacuum_traversal_efficiency = get_adaptive_mtf_normalized_score(vacuum_traversal_efficiency_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_active_selling_pressure = get_adaptive_mtf_normalized_score(active_selling_pressure_raw, df_index, ascending=False, tf_weights=tf_weights) # 卖压低，转化效率高
        norm_capitulation_absorption = get_adaptive_mtf_normalized_score(capitulation_absorption_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_pressure_rejection_strength = get_adaptive_mtf_normalized_score(pressure_rejection_strength_raw, df_index, ascending=False, tf_weights=tf_weights) # 阻力拒绝强度低，转化效率高
        norm_support_validation_strength = get_adaptive_mtf_normalized_score(support_validation_strength_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_order_book_clearing_rate = get_adaptive_mtf_normalized_score(order_book_clearing_rate_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_micro_price_impact_asymmetry = get_adaptive_mtf_normalized_score(micro_price_impact_asymmetry_raw.abs(), df_index, ascending=False, tf_weights=tf_weights) # 价格冲击不对称性低，转化效率高
        resistance_absorption_score = (norm_active_selling_pressure * 0.5 + norm_capitulation_absorption * 0.5).clip(0, 1)
        resistance_game_strength = (norm_pressure_rejection_strength * 0.5 + norm_support_validation_strength * 0.5).clip(0, 1)
        pce_components = {
            'vacuum_magnitude': norm_vacuum_zone_magnitude,
            'vacuum_efficiency': norm_vacuum_traversal_efficiency,
            'resistance_absorption': resistance_absorption_score,
            'resistance_game_strength': resistance_game_strength,
            'order_book_clearing_rate': norm_order_book_clearing_rate,
            'micro_price_impact_asymmetry': norm_micro_price_impact_asymmetry
        }
        pce_score = _robust_geometric_mean(pce_components, pce_weights, df_index)
        pce_score = pce_score.clip(0, 1)
        # --- 4. 诡道博弈调制 (DGM) ---
        deception_index_raw = signals_data['deception_index_D']
        wash_trade_intensity_raw = signals_data['wash_trade_intensity_D']
        retail_panic_surrender_raw = signals_data['retail_panic_surrender_index_D']
        main_force_conviction_raw = signals_data['main_force_conviction_index_D']
        conviction_flow_buy_raw = signals_data['conviction_flow_buy_intensity_D'] # 新增
        conviction_flow_sell_raw = signals_data['conviction_flow_sell_intensity_D'] # 新增
        norm_deception_index = get_adaptive_mtf_normalized_bipolar_score(deception_index_raw, df_index, tf_weights)
        norm_wash_trade_intensity = get_adaptive_mtf_normalized_score(wash_trade_intensity_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_retail_panic_surrender = get_adaptive_mtf_normalized_score(retail_panic_surrender_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_main_force_conviction = get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, tf_weights)
        # 构建纯筹码流向代理
        norm_conviction_flow_buy = get_adaptive_mtf_normalized_score(conviction_flow_buy_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_conviction_flow_sell = get_adaptive_mtf_normalized_score(conviction_flow_sell_raw, df_index, ascending=True, tf_weights=tf_weights)
        # 代理筹码流向：买入信念流 - 卖出信念流，归一化到双极
        chip_flow_directionality_proxy = (norm_conviction_flow_buy - norm_conviction_flow_sell).clip(-1, 1)
        # 初始化 dgm_score
        dgm_score_base = pd.Series(0.0, index=df_index)
        # 诱多惩罚：当有正向欺骗且筹码流向为负时，惩罚
        bull_trap_mask = (norm_deception_index > 0) & (chip_flow_directionality_proxy < 0)
        # 强化诱多惩罚：使用 bull_trap_penalty_factor 放大惩罚
        dgm_score_base.loc[bull_trap_mask] -= (norm_deception_index.loc[bull_trap_mask] * chip_flow_directionality_proxy.loc[bull_trap_mask].abs()) * dgm_weights.get('deception_impact', 0.4) * dgm_asymmetry_params.get('bull_trap_penalty_factor', 1.5)
        # 诱空奖励：当有负向欺骗且筹码流向为正时，奖励
        bear_trap_absorption_mask = (norm_deception_index < 0) & (chip_flow_directionality_proxy > 0)
        dgm_score_base.loc[bear_trap_absorption_mask] += (norm_deception_index.loc[bear_trap_absorption_mask].abs() * chip_flow_directionality_proxy.loc[bear_trap_absorption_mask]) * dgm_weights.get('deception_impact', 0.4) * dgm_asymmetry_params.get('bear_trap_bonus_factor', 1.2)
        dgm_score_base -= norm_wash_trade_intensity * dgm_weights.get('wash_trade_penalty', 0.2) # 对倒惩罚
        positive_flow_boost_mask = (chip_flow_directionality_proxy > 0) & (~bull_trap_mask)
        dgm_score_base.loc[positive_flow_boost_mask] += chip_flow_directionality_proxy.loc[positive_flow_boost_mask] * dgm_weights.get('flow_directionality_boost', 0.1)
        dgm_score_base += norm_retail_panic_surrender * dgm_weights.get('retail_panic_impact', 0.15) # 散户恐慌增加势能
        dgm_score_base += (norm_main_force_conviction.abs()) * dgm_weights.get('main_force_conviction_impact', 0.15) # 主力信念强度增加势能
        # 修正：诱空且筹码流出：强制设置为极低负值，并确保优先级
        bearish_deception_and_mf_out_mask = (norm_deception_index < 0) & (chip_flow_directionality_proxy < 0)
        # 使用 np.where 确保强制惩罚的优先级
        dgm_score = np.where(bearish_deception_and_mf_out_mask, -0.9, dgm_score_base)
        dgm_score = pd.Series(dgm_score, index=df_index).clip(-1, 1)
        # --- 5. 情境自适应权重 (ACW) ---
        context_modulator_components = []
        total_context_weight = 0.0
        for ctx_key, ctx_config in context_modulator_signals.items():
            signal_name = ctx_config.get('signal_name')
            weight = ctx_config.get('weight', 0.0)
            ascending = ctx_config.get('ascending', True)
            if signal_name and weight > 0:
                raw_signal = signals_data[signal_name]
                norm_signal = get_adaptive_mtf_normalized_score(raw_signal, df_index, ascending=ascending, tf_weights=tf_weights)
                context_modulator_components.append(norm_signal * weight)
                total_context_weight += weight
        if context_modulator_components and total_context_weight > 0:
            combined_context_modulator = sum(context_modulator_components) / total_context_weight
        else:
            combined_context_modulator = pd.Series(0.5, index=df_index) # 默认中性
        # 动态调整最终融合权重
        dynamic_final_fusion_weights = {
            'mf_aq': final_fusion_weights.get('mf_aq', 0.35) * (1 + combined_context_modulator * context_modulator_sensitivity),
            'cst': final_fusion_weights.get('cst', 0.3) * (1 + combined_context_modulator * context_modulator_sensitivity),
            'pce': final_fusion_weights.get('pce', 0.35) * (1 + combined_context_modulator * context_modulator_sensitivity)
        }
        sum_dynamic_weights = sum(dynamic_final_fusion_weights.values())
        normalized_dynamic_weights = {k: v / sum_dynamic_weights for k, v in dynamic_final_fusion_weights.items()}
        base_potential_score = (
            mf_aq_score * normalized_dynamic_weights.get('mf_aq', 0.35) +
            cst_score * normalized_dynamic_weights.get('cst', 0.3) +
            pce_score * normalized_dynamic_weights.get('pce', 0.35)
        ).clip(0, 1)
        # DGM 乘性调制
        dgm_multiplier = 1 + dgm_score * dgm_modulator_sensitivity
        dgm_multiplier = dgm_multiplier.clip(0.01, 2.0) # 限制调制范围，惩罚可以更深
        final_potential_score = (base_potential_score * dgm_multiplier).clip(0, 1)
        # --- 应用牛市陷阱情境惩罚 ---
        bull_trap_penalty = self._calculate_bull_trap_context_penalty(df)
        final_score = final_potential_score * bull_trap_penalty
        final_score = final_score.clip(0, 1).fillna(0.0).astype(np.float32)
        print(f"    -> [筹码层] 计算完成 '“筹码势能”' 分数: {final_score.loc[probe_ts] if probe_ts and probe_ts in df_index else final_score.iloc[-1]:.4f}")
        return final_score

    def _diagnose_tactical_exchange(self, df: pd.DataFrame, battlefield_geography: pd.Series) -> pd.Series:
        """
        【V6.0 · 筹码脉动版】诊断战术换手博弈的质量与意图
        - 核心升级1: 筹码“微观结构”与“订单流执行效率”评估。引入意图执行质量，作为意图维度的一个重要组成部分。
        - 核心升级2: 筹码“多峰结构”与“共振/冲突”分析。引入筹码峰动态，作为质量维度的一个新组成部分。
        - 核心升级3: 筹码“情绪”与“行为模式”识别。引入筹码行为模式强度，作为意图或质量维度的调制器。
        - 核心升级4: 非线性融合的“自学习”与“情境权重矩阵”。升级元调制器，使其能够更精细地调整融合权重。
        """
        df_index = df.index
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
        p_conf = self.chip_ultimate_params
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
        quality_weights = get_param_value(tactical_exchange_params.get('quality_weights'), {'bullish_absorption': 0.15, 'bullish_purity': 0.15, 'bearish_distribution': 0.15, 'exchange_purity': 0.15, 'structural_optimization': 0.1, 'psychological_pressure_absorption': 0.1, 'exchange_efficiency': 0.05, 'chip_peak_dynamics': 0.15})
        quality_context_signal_name = get_param_value(tactical_exchange_params.get('quality_context_signal_name'), 'winner_loser_momentum_D')
        structural_optimization_slope_period = get_param_value(tactical_exchange_params.get('structural_optimization_slope_period'), 5)
        psychological_pressure_absorption_slope_period = get_param_value(tactical_exchange_params.get('psychological_pressure_absorption_slope_period'), 5)
        chip_peak_dynamics_params = get_param_value(tactical_exchange_params.get('chip_peak_dynamics_params'), {})
        chip_behavioral_pattern_intensity_params = get_param_value(tactical_exchange_params.get('chip_behavioral_pattern_intensity_params'), {})
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
        # Add dynamic signals to required_signals if not already present
        if deception_context_modulator_signal_name not in required_signals:
            required_signals.append(deception_context_modulator_signal_name)
        if quality_context_signal_name not in required_signals:
            required_signals.append(quality_context_signal_name)
        if chip_stability_modulator_signal_name not in required_signals:
            required_signals.append(chip_stability_modulator_signal_name)
        # Add slope signals for structural_optimization_slope_period and psychological_pressure_absorption_slope_period
        slope_wc_signal = f'SLOPE_{structural_optimization_slope_period}_winner_concentration_90pct_D'
        if slope_wc_signal not in required_signals: required_signals.append(slope_wc_signal)
        slope_css_signal = f'SLOPE_{structural_optimization_slope_period}_cost_structure_skewness_D'
        if slope_css_signal not in required_signals: required_signals.append(slope_css_signal)
        slope_psr_signal = f'SLOPE_{structural_optimization_slope_period}_peak_separation_ratio_D'
        if slope_psr_signal not in required_signals: required_signals.append(slope_psr_signal)
        slope_loser_rate_signal = f'SLOPE_{psychological_pressure_absorption_slope_period}_total_loser_rate_D'
        if slope_loser_rate_signal not in required_signals: required_signals.append(slope_loser_rate_signal)
        slope_winner_rate_signal = f'SLOPE_{psychological_pressure_absorption_slope_period}_total_winner_rate_D'
        if slope_winner_rate_signal not in required_signals: required_signals.append(slope_winner_rate_signal)
        slope_dps_signal = f'SLOPE_{dominant_peak_health_slope_period}_dominant_peak_solidity_D'
        if slope_dps_signal not in required_signals: required_signals.append(slope_dps_signal)
        if not self._validate_required_signals(df, required_signals, "_diagnose_tactical_exchange"):
            return pd.Series(0.0, index=df.index)
        signals_data = self._get_all_required_signals(df, required_signals, "_diagnose_tactical_exchange")
        # --- 调试信息构建 ---
        is_debug_enabled = self.should_probe
        probe_ts = None
        if is_debug_enabled and self.probe_dates_set:
            for date in reversed(df_index):
                if date.date() in self.probe_dates_set:
                    probe_ts = date
                    break
        debug_info_tuple = False # (is_debug_enabled, probe_ts, "_diagnose_tactical_exchange")
        # --- 原始数据获取 ---
        control_transfer_raw = signals_data['peak_control_transfer_D']
        cleansing_efficiency_raw = signals_data['floating_chip_cleansing_efficiency_D']
        suppressive_accum_raw = signals_data['suppressive_accumulation_intensity_D']
        gathering_chasing_raw = signals_data['gathering_by_chasing_D']
        gathering_support_raw = signals_data['gathering_by_support_D']
        chip_fault_raw = signals_data['chip_fault_magnitude_D']
        mf_conviction_raw = signals_data['main_force_conviction_index_D']
        retail_panic_surrender_raw = signals_data['retail_panic_surrender_index_D']
        loser_pain_raw = signals_data['loser_pain_index_D']
        winner_profit_margin_avg_raw = signals_data['winner_profit_margin_avg_D']
        peak_exchange_purity_raw = signals_data['peak_exchange_purity_D']
        slope_wc_raw = signals_data[f'SLOPE_{structural_optimization_slope_period}_winner_concentration_90pct_D']
        slope_css_raw = signals_data[f'SLOPE_{structural_optimization_slope_period}_cost_structure_skewness_D']
        slope_psr_raw = signals_data[f'SLOPE_{structural_optimization_slope_period}_peak_separation_ratio_D']
        winner_loser_momentum_raw = signals_data['winner_loser_momentum_D']
        chip_health_raw = signals_data['chip_health_score_D']
        capitulation_absorption_raw = signals_data['capitulation_absorption_index_D']
        upward_impulse_purity_raw = signals_data['upward_impulse_purity_D']
        profit_realization_quality_raw = signals_data['profit_realization_quality_D']
        chip_fatigue_raw = signals_data['chip_fatigue_index_D']
        volatility_instability_raw = signals_data['VOLATILITY_INSTABILITY_INDEX_21d_D']
        dominant_peak_solidity_raw = signals_data['dominant_peak_solidity_D']
        slope_dps_raw = signals_data[f'SLOPE_{dominant_peak_health_slope_period}_dominant_peak_solidity_D']
        total_loser_rate_raw = signals_data['total_loser_rate_D']
        total_winner_rate_raw = signals_data['total_winner_rate_D']
        slope_loser_rate_raw = signals_data[f'SLOPE_{psychological_pressure_absorption_slope_period}_total_loser_rate_D']
        slope_winner_rate_raw = signals_data[f'SLOPE_{psychological_pressure_absorption_slope_period}_total_winner_rate_D']
        volume_raw = signals_data['volume_D']
        winner_stability_index_raw = signals_data['winner_stability_index_D']
        active_buying_support_raw = signals_data['active_buying_support_D']
        active_selling_pressure_raw = signals_data['active_selling_pressure_D']
        micro_price_impact_asymmetry_raw = signals_data['micro_price_impact_asymmetry_D']
        order_book_clearing_rate_raw = signals_data['order_book_clearing_rate_D']
        flow_credibility_index_raw = signals_data['flow_credibility_index_D']
        secondary_peak_cost_raw = signals_data['secondary_peak_cost_D']
        dominant_peak_volume_ratio_raw = signals_data['dominant_peak_volume_ratio_D']
        main_force_activity_ratio_raw = signals_data['main_force_activity_ratio_D']
        main_force_flow_directionality_raw = signals_data['main_force_flow_directionality_D']
        slope_5_main_force_activity_ratio_raw = signals_data['SLOPE_5_main_force_activity_ratio_D']
        slope_5_main_force_flow_directionality_raw = signals_data['SLOPE_5_main_force_flow_directionality_D']
        deception_context_modulator_raw = signals_data[deception_context_modulator_signal_name]
        quality_context_raw = signals_data[quality_context_signal_name]
        chip_stability_modulator_raw = signals_data[chip_stability_modulator_signal_name]
        # --- 维度1: 换手意图 (Exchange Intent) - 纯筹码化与诡道深化 ---
        norm_control_transfer = get_adaptive_mtf_normalized_bipolar_score(control_transfer_raw, df_index, tf_weights)
        norm_cleansing_efficiency = get_adaptive_mtf_normalized_score(cleansing_efficiency_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_suppressive_accum = get_adaptive_mtf_normalized_score(suppressive_accum_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_gathering_chasing = get_adaptive_mtf_normalized_score(gathering_chasing_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_gathering_support = get_adaptive_mtf_normalized_score(gathering_support_raw, df_index, ascending=True, tf_weights=tf_weights)
        accumulation_intent_score = (norm_suppressive_accum * 0.4 + norm_gathering_chasing * 0.3 + norm_gathering_support * 0.3)
        # 意图执行质量
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
        # 诡道仲裁与调制
        chip_fault_raw = signals_data['chip_fault_magnitude_D']
        mf_conviction_raw = signals_data['main_force_conviction_index_D']
        norm_chip_fault = get_adaptive_mtf_normalized_score(chip_fault_raw.abs(), df_index, ascending=True, tf_weights=tf_weights)
        norm_mf_conviction = get_adaptive_mtf_normalized_bipolar_score(mf_conviction_raw, df_index, tf_weights)
        chip_deception_direction = np.sign(norm_mf_conviction) # 主力信念方向作为欺骗方向判断依据
        retail_panic_surrender_raw = signals_data['retail_panic_surrender_index_D']
        loser_pain_raw = signals_data['loser_pain_index_D']
        winner_profit_margin_avg_raw = signals_data['winner_profit_margin_avg_D']
        norm_retail_panic_surrender = get_adaptive_mtf_normalized_score(retail_panic_surrender_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_loser_pain = get_adaptive_mtf_normalized_score(loser_pain_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_winner_profit_margin_avg = get_adaptive_mtf_normalized_score(winner_profit_margin_avg_raw, df_index, ascending=True, tf_weights=tf_weights)
        deception_effectiveness_score = pd.Series(0.0, index=df_index)
        deception_cost_score = pd.Series(0.0, index=df_index)
        # 诱空效果评估 (主力诱导散户恐慌抛售)
        induce_bear_mask = chip_deception_direction > 0 # 主力信念为正，但有欺骗
        deception_effectiveness_score.loc[induce_bear_mask] = (norm_retail_panic_surrender.loc[induce_bear_mask] + norm_loser_pain.loc[induce_bear_mask]) / 2
        deception_cost_score.loc[induce_bear_mask] = norm_suppressive_accum.loc[induce_bear_mask] # 压制式吸筹作为成本
        # 诱多效果评估 (主力诱导散户追涨接盘)
        induce_bull_mask = chip_deception_direction < 0 # 主力信念为负，但有欺骗
        deception_effectiveness_score.loc[induce_bull_mask] = norm_winner_profit_margin_avg.loc[induce_bull_mask] # 赢家利润率作为效果
        profit_realization_quality_raw = signals_data['profit_realization_quality_D']
        deception_cost_score.loc[induce_bull_mask] = (1 - get_adaptive_mtf_normalized_score(profit_realization_quality_raw, df_index, ascending=True, tf_weights=tf_weights)).loc[induce_bull_mask] # 利润兑现质量低，成本高
        deception_quality_modulator = (
            deception_outcome_weights.get('effectiveness', 0.6) * deception_effectiveness_score.clip(0, 1) +
            deception_outcome_weights.get('cost', 0.4) * deception_cost_score.clip(0, 1)
        )
        # 只有高质量的欺骗才会被放大
        high_quality_deception_mask = (deception_effectiveness_score > deception_outcome_effectiveness_threshold) & (deception_cost_score > deception_outcome_cost_threshold)
        deception_quality_modulator.loc[~high_quality_deception_mask] *= 0.5 # 低质量欺骗效果减半
        chip_deception_score_refined = norm_chip_fault * chip_deception_direction * (1 + deception_quality_modulator.clip(0, 1))
        chip_deception_score_refined = chip_deception_score_refined.clip(-1, 1)
        deception_context_modulator_raw = signals_data[deception_context_modulator_signal_name]
        norm_deception_context = get_adaptive_mtf_normalized_score(deception_context_modulator_raw, df_index, ascending=True, tf_weights=tf_weights)
        dynamic_deception_impact_sensitivity = deception_impact_sensitivity * (1 - norm_deception_context * deception_context_sensitivity)
        dynamic_deception_impact_sensitivity = dynamic_deception_impact_sensitivity.clip(0.1, 1.0)
        arbitration_weight = (norm_chip_fault * dynamic_deception_impact_sensitivity).pow(deception_arbitration_power).clip(0, 1)
        # 意图分数 = 基础意图 * (1 - 仲裁权重) + 诡道分数 * 仲裁权重
        intent_score = base_intent_score * (1 - arbitration_weight) + chip_deception_score_refined * arbitration_weight
        intent_score = intent_score.clip(-1, 1)
        # 筹码行为模式强度调制
        norm_main_force_activity = get_adaptive_mtf_normalized_score(main_force_activity_ratio_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_main_force_flow_directionality = get_adaptive_mtf_normalized_bipolar_score(main_force_flow_directionality_raw, df_index, tf_weights)
        chip_behavioral_pattern_intensity_score = (norm_main_force_activity * 0.6 + norm_main_force_flow_directionality.abs() * 0.4).clip(0, 1)
        intent_score = intent_score * (1 + chip_behavioral_pattern_intensity_score * chip_behavioral_pattern_intensity_modulator_factor)
        intent_score = intent_score.clip(-1, 1)
        # --- 维度2: 换手质量 (Exchange Quality) - 纯筹码化与情境自适应 ---
        chip_momentum_raw = signals_data[quality_context_signal_name]
        norm_chip_momentum_context = get_adaptive_mtf_normalized_bipolar_score(chip_momentum_raw, df_index, tf_weights)
        # 看涨质量
        absorption_idx_raw = signals_data['capitulation_absorption_index_D']
        impulse_purity_raw = signals_data['upward_impulse_purity_D']
        norm_absorption = get_adaptive_mtf_normalized_score(absorption_idx_raw, df_index, tf_weights)
        norm_impulse_purity = get_adaptive_mtf_normalized_score(impulse_purity_raw, df_index, tf_weights)
        dynamic_bullish_quality_weight = (norm_chip_momentum_context.add(1)/2) * 0.5 + 0.5 # 动量越强，看涨质量权重越高
        bullish_quality_score = (
            norm_absorption * quality_weights.get('bullish_absorption', 0.15) +
            norm_impulse_purity * quality_weights.get('bullish_purity', 0.15)
        ) * dynamic_bullish_quality_weight
        # 看跌质量
        profit_quality_raw = signals_data['profit_realization_quality_D']
        norm_profit_realization = get_adaptive_mtf_normalized_score(profit_quality_raw, df_index, ascending=False, tf_weights=tf_weights)
        dynamic_bearish_quality_weight = (1 - norm_chip_momentum_context.add(1)/2) * 0.5 + 0.5 # 动量越弱，看跌质量权重越高
        bearish_quality_score = norm_profit_realization * quality_weights.get('bearish_distribution', 0.15) * dynamic_bearish_quality_weight
        # 换手纯度
        peak_exchange_purity_raw = signals_data['peak_exchange_purity_D']
        exchange_purity_score = get_adaptive_mtf_normalized_score(peak_exchange_purity_raw, df_index, ascending=True, tf_weights=tf_weights)
        # 结构优化
        slope_wc_raw = signals_data[f'SLOPE_{structural_optimization_slope_period}_winner_concentration_90pct_D']
        slope_css_raw = signals_data[f'SLOPE_{structural_optimization_slope_period}_cost_structure_skewness_D']
        slope_psr_raw = signals_data[f'SLOPE_{structural_optimization_slope_period}_peak_separation_ratio_D']
        norm_slope_wc = get_adaptive_mtf_normalized_score(slope_wc_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_slope_css = get_adaptive_mtf_normalized_score(slope_css_raw, df_index, ascending=False, tf_weights=tf_weights)
        norm_slope_psr = get_adaptive_mtf_normalized_score(slope_psr_raw, df_index, ascending=False, tf_weights=tf_weights)
        structural_optimization_score = (norm_slope_wc + norm_slope_css + norm_slope_psr) / 3
        structural_optimization_score = structural_optimization_score.clip(0, 1)
        # 心理压力吸收
        slope_loser_rate_raw = signals_data[f'SLOPE_{psychological_pressure_absorption_slope_period}_total_loser_rate_D']
        slope_winner_rate_raw = signals_data[f'SLOPE_{psychological_pressure_absorption_slope_period}_total_winner_rate_D']
        norm_loser_absorption_quality = get_adaptive_mtf_normalized_bipolar_score(slope_loser_rate_raw, df_index, tf_weights, sensitivity=1.0) # 输家比例下降，吸收质量高
        norm_winner_resilience_quality = get_adaptive_mtf_normalized_bipolar_score(slope_winner_rate_raw, df_index, tf_weights, sensitivity=1.0) # 赢家比例上升，韧性强
        psychological_pressure_absorption_score = (norm_loser_absorption_quality.clip(lower=0) + norm_winner_resilience_quality.clip(upper=0).abs()) / 2 # 负向的赢家比例变化代表韧性
        psychological_pressure_absorption_score = psychological_pressure_absorption_score.clip(0, 1)
        # 换手效率
        volume_raw = signals_data['volume_D']
        norm_volume = get_adaptive_mtf_normalized_score(volume_raw, df_index, ascending=True, tf_weights=tf_weights)
        exchange_efficiency_score = structural_optimization_score / (norm_volume.replace(0, 1e-6)) # 结构优化程度 / 成交量
        exchange_efficiency_score = exchange_efficiency_score.clip(0, 1)
        # 筹码峰动态
        secondary_peak_cost_raw = signals_data['secondary_peak_cost_D']
        dominant_peak_volume_ratio_raw = signals_data['dominant_peak_volume_ratio_D']
        norm_secondary_peak_cost = get_adaptive_mtf_normalized_score(secondary_peak_cost_raw, df_index, ascending=False, tf_weights=tf_weights) # 次峰成本低，结构健康
        norm_dominant_peak_volume_ratio = get_adaptive_mtf_normalized_score(dominant_peak_volume_ratio_raw, df_index, ascending=True, tf_weights=tf_weights) # 主峰成交量占比高，结构健康
        chip_peak_dynamics_score = (norm_secondary_peak_cost * chip_peak_dynamics_params.get('secondary_cost_weight', 0.5) +
                                    norm_dominant_peak_volume_ratio * chip_peak_dynamics_params.get('secondary_volume_weight', 0.5)).clip(0, 1)
        quality_score = (
            bullish_quality_score * (1 - dynamic_bearish_quality_weight) + # 看涨质量受看跌权重抑制
            bearish_quality_score * (1 - dynamic_bullish_quality_weight) + # 看跌质量受看涨权重抑制
            exchange_purity_score * quality_weights.get('exchange_purity', 0.15) +
            structural_optimization_score * quality_weights.get('structural_optimization', 0.1) +
            psychological_pressure_absorption_score * quality_weights.get('psychological_pressure_absorption', 0.1) +
            exchange_efficiency_score * quality_weights.get('exchange_efficiency', 0.05) +
            chip_peak_dynamics_score * quality_weights.get('chip_peak_dynamics', 0.15)
        ).clip(-1, 1)
        quality_score = quality_score * (1 + chip_behavioral_pattern_intensity_score * chip_behavioral_pattern_intensity_modulator_factor)
        quality_score = quality_score.clip(-1, 1)
        # --- 维度3: 换手环境 (Exchange Context) - 纯筹码化与情境自适应 ---
        chip_fatigue_raw = signals_data['chip_fatigue_index_D']
        norm_chip_fatigue = get_adaptive_mtf_normalized_score(chip_fatigue_raw, df_index, ascending=True, tf_weights=tf_weights)
        chip_stability_modulator_raw = signals_data[chip_stability_modulator_signal_name]
        norm_chip_stability_modulator = get_adaptive_mtf_normalized_score(chip_stability_modulator_raw, df_index, ascending=False, tf_weights=tf_weights) # 波动性低，稳定性高
        chip_health_raw = signals_data['chip_health_score_D']
        norm_chip_health = get_adaptive_mtf_normalized_bipolar_score(chip_health_raw, df_index, tf_weights)
        dynamic_chip_fatigue_impact = norm_chip_fatigue * chip_fatigue_impact_factor
        dynamic_chip_stability_bonus = norm_chip_stability_modulator * chip_stability_sensitivity
        dominant_peak_solidity_raw = signals_data['dominant_peak_solidity_D']
        slope_dps_raw = signals_data[f'SLOPE_{dominant_peak_health_slope_period}_dominant_peak_solidity_D']
        norm_dps = get_adaptive_mtf_normalized_score(dominant_peak_solidity_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_slope_dps = get_adaptive_mtf_normalized_bipolar_score(slope_dps_raw, df_index, tf_weights)
        dominant_peak_health_score = (norm_dps * 0.7 + norm_slope_dps * 0.3).clip(0, 1)
        winner_stability_index_raw = signals_data['winner_stability_index_D']
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
        rhythm_intent_slope = intent_score.diff(rhythm_persistence_slope_period).fillna(0)
        rhythm_quality_slope = quality_score.diff(rhythm_persistence_slope_period).fillna(0)
        norm_rhythm_intent_slope = get_adaptive_mtf_normalized_bipolar_score(rhythm_intent_slope, df_index, tf_weights)
        norm_rhythm_quality_slope = get_adaptive_mtf_normalized_bipolar_score(rhythm_quality_slope, df_index, tf_weights)
        rhythm_and_persistence_score = (norm_rhythm_intent_slope + norm_rhythm_quality_slope) / 2
        rhythm_and_persistence_score = (rhythm_and_persistence_score * rhythm_persistence_sensitivity).clip(-1, 1)
        # --- 最终融合 ---
        volatility_instability_raw = signals_data['VOLATILITY_INSTABILITY_INDEX_21d_D']
        norm_volatility_instability = get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=False, tf_weights=tf_weights)
        main_force_conviction_abs_raw = signals_data['main_force_conviction_index_D'].abs()
        norm_main_force_conviction = get_adaptive_mtf_normalized_score(main_force_conviction_abs_raw, df_index, ascending=True, tf_weights=tf_weights)
        main_force_activity_abs_raw = signals_data['main_force_activity_ratio_D'].abs()
        norm_main_force_activity_meta = get_adaptive_mtf_normalized_score(main_force_activity_abs_raw, df_index, ascending=True, tf_weights=tf_weights)
        flow_credibility_index_meta_raw = signals_data['flow_credibility_index_D']
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
        print(f"    -> [筹码层] 计算完成 '战术换手博弈的质量与意图' 分数: {final_score.iloc[-1]}")
        return final_score.clip(-1, 1).fillna(0.0).astype(np.float32)

    def _diagnose_strategic_tactical_harmony(self, df: pd.DataFrame, strategic_posture: pd.Series, tactical_exchange: pd.Series, holder_sentiment_scores: pd.Series) -> pd.Series:
        """
        【V3.0 · 诡道微观共振版】诊断战略与战术的和谐度
        - 核心升级1: 战术执行微观深化。引入高频微观筹码行为（如日内筹码流平衡、订单簿压力等）作为“当日战术执行”的更精细化输入，提升战术评估的颗粒度和准确性。
        - 核心升级2: 动态权重调制精细化。战略与战术的融合权重不再固定，而是根据筹码波动不稳定性、筹码疲劳指数等筹码层情境因子动态调整，以适应不同市场阶段的侧重点。
        - 核心升级3: 和谐因子情境纯筹码化。和谐度因子的情境调制器严格限定为筹码层信号（如持仓信念韧性、价筹张力），确保信号的纯粹性。
        - 核心升级4: 冲突情境诡道深化。明确识别战略与战术方向完全背离的“冲突区”，并引入诡道因子（如欺骗指数、对倒强度）进行调制，对伴随欺骗的冲突施加更严厉惩罚。
        - 核心升级5: 趋势一致性品质校准。当战略与战术在同一方向上高度协同并具备足够强度时，引入筹码品质因子（如筹码健康度、主力信念指数）进行校准，确保奖励的是高质量、可持续的趋势。
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        df_index = df.index
        p_conf = self.chip_ultimate_params
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        harmony_params = get_param_value(p_conf.get('strategic_tactical_harmony_params'), {})
        strategic_weight_base = get_param_value(harmony_params.get('strategic_weight_base'), 0.6)
        tactical_weight_base = get_param_value(harmony_params.get('tactical_weight_base'), 0.4)
        dynamic_weight_modulator_signal_name = get_param_value(harmony_params.get('dynamic_weight_modulator_signal'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        dynamic_weight_sensitivity = get_param_value(harmony_params.get('dynamic_weight_sensitivity'), 0.3)
        harmony_non_linear_exponent = get_param_value(harmony_params.get('harmony_non_linear_exponent'), 2.5)
        harmony_context_modulator_signal_name = get_param_value(harmony_params.get('harmony_context_modulator_signal'), 'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT')
        harmony_context_sensitivity = get_param_value(harmony_params.get('harmony_context_sensitivity'), 0.4)
        conflict_threshold = get_param_value(harmony_params.get('conflict_threshold'), 0.6)
        conflict_penalty_factor = get_param_value(harmony_params.get('conflict_penalty_factor'), 0.7)
        deception_modulator_signal_name = get_param_value(harmony_params.get('deception_modulator_signal'), 'deception_index_D')
        deception_penalty_sensitivity = get_param_value(harmony_params.get('deception_penalty_sensitivity'), 0.5)
        trend_alignment_threshold = get_param_value(harmony_params.get('trend_alignment_threshold'), 0.75)
        trend_bonus_factor = get_param_value(harmony_params.get('trend_bonus_factor'), 0.15)
        quality_calibrator_signal_name = get_param_value(harmony_params.get('quality_calibrator_signal'), 'chip_health_score_D')
        quality_calibration_sensitivity = get_param_value(harmony_params.get('quality_calibration_sensitivity'), 0.5)
        high_harmony_threshold = get_param_value(harmony_params.get('high_harmony_threshold'), 0.8)
        required_signals = [
            dynamic_weight_modulator_signal_name,
            deception_modulator_signal_name,
            quality_calibrator_signal_name
        ]
        if harmony_context_modulator_signal_name != 'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT':
            required_signals.append(harmony_context_modulator_signal_name)
        if not self._validate_required_signals(df, required_signals, "_diagnose_strategic_tactical_harmony"):
            return pd.Series(0.0, index=df.index)
        signals_data = self._get_all_required_signals(df, required_signals, "_diagnose_strategic_tactical_harmony")
        # --- 调试信息构建 ---
        is_debug_enabled = self.should_probe
        probe_ts = None
        if is_debug_enabled and self.probe_dates_set:
            for date in reversed(df_index):
                if date.date() in self.probe_dates_set:
                    probe_ts = date
                    break
        debug_info_tuple = False # (is_debug_enabled, probe_ts, "_diagnose_strategic_tactical_harmony")
        # --- 原始数据获取 ---
        dynamic_weight_modulator_raw = signals_data[dynamic_weight_modulator_signal_name]
        deception_raw = signals_data[deception_modulator_signal_name]
        quality_calibrator_raw = signals_data[quality_calibrator_signal_name]
        if harmony_context_modulator_signal_name == 'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT':
            harmony_context_modulator_raw = holder_sentiment_scores
        else:
            harmony_context_modulator_raw = signals_data[harmony_context_modulator_signal_name]
        # --- 1. 动态权重融合 (Dynamic Weight Fusion) ---
        norm_dynamic_weight_modulator = get_adaptive_mtf_normalized_score(dynamic_weight_modulator_raw, df_index, ascending=True, tf_weights=tf_weights)
        dynamic_strategic_weight = strategic_weight_base * (1 - norm_dynamic_weight_modulator * dynamic_weight_sensitivity)
        dynamic_tactical_weight = tactical_weight_base * (1 + norm_dynamic_weight_modulator * dynamic_weight_sensitivity)
        sum_dynamic_weights = dynamic_strategic_weight + dynamic_tactical_weight
        dynamic_strategic_weight = dynamic_strategic_weight / sum_dynamic_weights
        dynamic_tactical_weight = dynamic_tactical_weight / sum_dynamic_weights
        base_intent_score = strategic_posture * dynamic_strategic_weight + tactical_exchange * dynamic_tactical_weight
        # --- 2. 和谐因子非线性增强 (Non-linear Harmony Factor Enhancement) ---
        raw_difference = (strategic_posture - tactical_exchange).abs() / 2
        non_linear_diff = raw_difference.pow(harmony_non_linear_exponent)
        harmony_factor = (1 - non_linear_diff).clip(lower=0)
        norm_harmony_context = get_adaptive_mtf_normalized_bipolar_score(harmony_context_modulator_raw, df_index, tf_weights)
        context_modulation_effect = (norm_harmony_context * harmony_context_sensitivity).clip(-0.5, 0.5)
        harmony_factor = harmony_factor * (1 + context_modulation_effect)
        harmony_factor = harmony_factor.clip(0, 1)
        # --- 3. 冲突情境识别与惩罚 (Conflict Context Recognition & Penalty) ---
        conflict_penalty_factor_adjusted = pd.Series(1.0, index=df_index)
        strong_bullish_strategic_bearish_tactical = (strategic_posture > conflict_threshold) & (tactical_exchange < -conflict_threshold)
        strong_bearish_strategic_bullish_tactical = (strategic_posture < -conflict_threshold) & (tactical_exchange > conflict_threshold)
        conflict_mask = strong_bullish_strategic_bearish_tactical | strong_bearish_strategic_bullish_tactical
        norm_deception = get_adaptive_mtf_normalized_bipolar_score(deception_raw, df_index, tf_weights)
        deception_impact = pd.Series(0.0, index=df_index)
        # 冲突区内，如果伴随欺骗，则惩罚加重
        deception_impact.loc[conflict_mask & (norm_deception > 0)] = norm_deception.loc[conflict_mask & (norm_deception > 0)] * deception_penalty_sensitivity
        conflict_penalty_factor_adjusted.loc[conflict_mask] = 1 - (conflict_penalty_factor + deception_impact.loc[conflict_mask]).clip(0, 1)
        conflict_penalty_factor_adjusted = conflict_penalty_factor_adjusted.clip(0, 1)
        # --- 4. 趋势一致性奖励 (Trend Alignment Bonus) ---
        alignment_bonus = pd.Series(0.0, index=df_index)
        bullish_alignment_mask = (strategic_posture > trend_alignment_threshold) & \
                                 (tactical_exchange > trend_alignment_threshold) & \
                                 (harmony_factor > high_harmony_threshold)
        bearish_alignment_mask = (strategic_posture < -trend_alignment_threshold) & \
                                 (tactical_exchange < -trend_alignment_threshold) & \
                                 (harmony_factor > high_harmony_threshold)
        norm_quality_calibrator = get_adaptive_mtf_normalized_score(quality_calibrator_raw, df_index, ascending=True, tf_weights=tf_weights)
        calibrated_bonus_factor = trend_bonus_factor * (1 + norm_quality_calibrator * quality_calibration_sensitivity)
        calibrated_bonus_factor = calibrated_bonus_factor.clip(0, trend_bonus_factor * 2) # 限制奖励上限
        alignment_bonus.loc[bullish_alignment_mask] = calibrated_bonus_factor.loc[bullish_alignment_mask]
        alignment_bonus.loc[bearish_alignment_mask] = -calibrated_bonus_factor.loc[bearish_alignment_mask]
        # --- 最终融合 ---
        final_score = base_intent_score * harmony_factor * conflict_penalty_factor_adjusted + alignment_bonus
        final_score = final_score.clip(-1, 1).fillna(0.0).astype(np.float32)
        print(f"    -> [筹码层] 计算完成 '主力成本区攻防意图' 分数: {final_score.iloc[-1]}")
        return final_score

    def _diagnose_harmony_inflection(self, df: pd.DataFrame, harmony_score: pd.Series) -> pd.Series:
        """
        【V3.4 · 诡道确认强化与参数深度修正版 & 牛市陷阱惩罚版】诊断战略与战术和谐度的动态转折点，旨在构建一个诡道拐点判别与确认系统。
        - 核心升级1: 动态阈值自适应：和谐度所处区间（低位、中位、高位）的判断阈值不再固定，而是根据市场波动性或筹码健康度动态调整，提高对拐点“位置”判断的适应性。
        - 核心升级2: 非对称拐点动能融合：采用更复杂的非线性函数融合速度和加速度，并允许正向和负向拐点使用不同的融合参数，以更精细地量化拐点背后的真实动能，并反映市场情绪的非对称性。
        - 核心升级3: 诡道博弈过滤与惩罚：引入欺骗指数、对倒强度等诡道因子作为调制器，识别并惩罚伴随诱多欺骗的正向拐点，或适度削弱伴随诱空洗盘的负向拐点，提高信号真实性。
        - 核心升级4: 拐点延续性确认奖励：引入短期延续性检查机制，如果拐点方向在后续几天得到确认，则给予额外奖励，增强信号可靠性和鲁棒性。
        - 核心升级5: 增强情境调制器：除了筹码健康度和波动性，再引入主力信念指数作为情境调制器，更全面评估拐点信号在不同市场参与者意图下的可靠性。
        - 核心修复: 确保 `deception_modulator` 在 `norm_deception` 为负时，能够正确增强正向拐点信号，并增加增强敏感度。
        - **新增业务逻辑：引入“牛市陷阱情境惩罚”，在近期大幅下跌后伴随正向欺骗时，大幅降低和谐拐点的得分。**
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        method_name = "_diagnose_harmony_inflection"
        df_index = df.index
        p_conf = self.chip_ultimate_params
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        inflection_params = get_param_value(p_conf.get('harmony_inflection_params'), {})
        velocity_period = get_param_value(inflection_params.get('velocity_period'), 1)
        acceleration_period = get_param_value(inflection_params.get('acceleration_period'), 1)
        positive_strength_tanh_factor = get_param_value(inflection_params.get('positive_strength_tanh_factor'), 1.5)
        negative_strength_tanh_factor = get_param_value(inflection_params.get('negative_strength_tanh_factor'), 1.5)
        base_low_harmony_threshold = get_param_value(inflection_params.get('base_low_harmony_threshold'), 0.2)
        base_high_harmony_threshold = get_param_value(inflection_params.get('base_high_harmony_threshold'), 0.8)
        threshold_modulator_signal_name = get_param_value(inflection_params.get('threshold_modulator_signal'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        threshold_modulator_sensitivity = get_param_value(inflection_params.get('threshold_modulator_sensitivity'), 0.2)
        low_harmony_boost_factor = get_param_value(inflection_params.get('low_harmony_boost_factor'), 1.5)
        high_harmony_boost_factor = get_param_value(inflection_params.get('high_harmony_boost_factor'), 1.5)
        mid_harmony_neutral_factor = get_param_value(inflection_params.get('mid_harmony_neutral_factor'), 1.0)
        deception_signal_name = get_param_value(inflection_params.get('deception_signal'), 'deception_index_D')
        wash_trade_signal_name = get_param_value(inflection_params.get('wash_trade_signal'), 'wash_trade_intensity_D')
        deception_penalty_sensitivity = get_param_value(inflection_params.get('deception_penalty_sensitivity'), 0.7) # 提高敏感度
        wash_trade_mitigation_sensitivity = get_param_value(inflection_params.get('wash_trade_mitigation_sensitivity'), 0.3)
        persistence_period = get_param_value(inflection_params.get('persistence_period'), 2)
        persistence_bonus_factor = get_param_value(inflection_params.get('persistence_bonus_factor'), 0.1)
        context_modulator_signal_1_name = get_param_value(inflection_params.get('context_modulator_signal_1'), 'chip_health_score_D')
        context_modulator_signal_2_name = get_param_value(inflection_params.get('context_modulator_signal_2'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        context_modulator_signal_3_name = get_param_value(inflection_params.get('context_modulator_signal_3'), 'main_force_conviction_index_D')
        context_modulator_sensitivity_health = get_param_value(inflection_params.get('context_modulator_sensitivity_health'), 0.5)
        context_modulator_sensitivity_volatility = get_param_value(inflection_params.get('context_modulator_sensitivity_volatility'), 0.3)
        context_modulator_sensitivity_conviction = get_param_value(inflection_params.get('context_modulator_sensitivity_conviction'), 0.4)
        deception_boost_factor_negative = get_param_value(inflection_params.get('deception_boost_factor_negative'), 0.5)
        required_signals = [
            threshold_modulator_signal_name,
            deception_signal_name,
            wash_trade_signal_name,
            context_modulator_signal_1_name,
            context_modulator_signal_2_name,
            context_modulator_signal_3_name,
            'pct_change_D' # 新增牛市陷阱检测所需信号
        ]
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index)
        signals_data = self._get_all_required_signals(df, required_signals, method_name)
        # --- 调试信息构建 ---
        is_debug_enabled = self.should_probe
        probe_ts = None
        if is_debug_enabled and self.probe_dates_set:
            for date in reversed(df_index):
                if date.date() in self.probe_dates_set:
                    probe_ts = date
                    break
        debug_info_tuple = (is_debug_enabled, probe_ts, method_name)
        # --- 原始数据获取 ---
        threshold_modulator_raw = signals_data[threshold_modulator_signal_name]
        deception_raw = signals_data[deception_signal_name]
        wash_trade_raw = signals_data[wash_trade_signal_name]
        chip_health_raw = signals_data[context_modulator_signal_1_name]
        volatility_instability_raw = signals_data[context_modulator_signal_2_name]
        main_force_conviction_raw = signals_data[context_modulator_signal_3_name]
        # --- 1. 计算速度与加速度 (保留方向信息) ---
        harmony_velocity = harmony_score.diff(velocity_period).fillna(0)
        harmony_acceleration = harmony_velocity.diff(acceleration_period).fillna(0)
        norm_velocity = get_adaptive_mtf_normalized_bipolar_score(harmony_velocity, df_index, tf_weights)
        norm_acceleration = get_adaptive_mtf_normalized_bipolar_score(harmony_acceleration, df_index, tf_weights)
        # --- 2. 非对称拐点动能融合 (Asymmetric Inflection Momentum Fusion) ---
        positive_inflection_strength = pd.Series(0.0, index=df_index)
        # 拐点判断逻辑：速度从负转正，或速度为负但加速度为正（即将转正），或速度为正且加速度为正
        positive_inflection_mask = ((norm_velocity.shift(1) < 0) & (norm_velocity >= 0)) | \
                                   ((norm_velocity < 0) & (norm_acceleration > 0)) | \
                                   ((norm_velocity >= 0) & (norm_acceleration > 0))
        positive_inflection_strength.loc[positive_inflection_mask] = \
            np.tanh((norm_velocity.loc[positive_inflection_mask].clip(lower=0) + norm_acceleration.loc[positive_inflection_mask].clip(lower=0)) * positive_strength_tanh_factor)
        negative_inflection_strength = pd.Series(0.0, index=df_index)
        # 拐点判断逻辑：速度从正转负，或速度为正但加速度为负（即将转负），或速度为负且加速度为负
        negative_inflection_mask = ((norm_velocity.shift(1) > 0) & (norm_velocity <= 0)) | \
                                   ((norm_velocity > 0) & (norm_acceleration < 0)) | \
                                   ((norm_velocity <= 0) & (norm_acceleration < 0))
        negative_inflection_strength.loc[negative_inflection_mask] = \
            np.tanh((norm_velocity.loc[negative_inflection_mask].abs().clip(lower=0) + norm_acceleration.loc[negative_inflection_mask].abs().clip(lower=0)) * negative_strength_tanh_factor)
        inflection_strength = positive_inflection_strength - negative_inflection_strength
        # --- 3. 动态阈值自适应 (Dynamic Threshold Adaptation) ---
        norm_threshold_modulator = get_adaptive_mtf_normalized_score(threshold_modulator_raw, df_index, ascending=True, tf_weights=tf_weights)
        dynamic_low_harmony_threshold = base_low_harmony_threshold * (1 - norm_threshold_modulator * threshold_modulator_sensitivity)
        dynamic_high_harmony_threshold = base_high_harmony_threshold * (1 + norm_threshold_modulator * threshold_modulator_sensitivity)
        dynamic_low_harmony_threshold = dynamic_low_harmony_threshold.clip(0.05, 0.3) # 限制阈值范围
        dynamic_high_harmony_threshold = dynamic_high_harmony_threshold.clip(0.7, 0.95)
        # --- 4. 动态位置敏感度 (Dynamic Position Sensitivity) ---
        position_sensitivity_factor = pd.Series(mid_harmony_neutral_factor, index=df_index)
        low_harmony_zone_mask = harmony_score < dynamic_low_harmony_threshold
        position_sensitivity_factor.loc[low_harmony_zone_mask & (inflection_strength > 0)] = low_harmony_boost_factor # 低位正向拐点，放大
        position_sensitivity_factor.loc[low_harmony_zone_mask & (inflection_strength < 0)] = 1 / low_harmony_boost_factor # 低位负向拐点，缩小
        high_harmony_zone_mask = harmony_score > dynamic_high_harmony_threshold
        position_sensitivity_factor.loc[high_harmony_zone_mask & (inflection_strength < 0)] = high_harmony_boost_factor # 高位负向拐点，放大
        position_sensitivity_factor.loc[high_harmony_zone_mask & (inflection_strength > 0)] = 1 / high_harmony_boost_factor # 高位正向拐点，缩小
        # --- 5. 诡道博弈过滤与惩罚 (Deceptive Game Filtering & Penalty) ---
        norm_deception = get_adaptive_mtf_normalized_bipolar_score(deception_raw, df_index, tf_weights)
        norm_wash_trade = get_adaptive_mtf_normalized_score(wash_trade_raw, df_index, ascending=True, tf_weights=tf_weights)
        deception_modulator = pd.Series(1.0, index=df_index)
        # 修正：诱空反吸增强：负向欺骗时增强正向拐点信号
        deception_boost_mask = (inflection_strength > 0) & (norm_deception < 0)
        deception_modulator.loc[deception_boost_mask] = 1 + (norm_deception.loc[deception_boost_mask].abs() * deception_boost_factor_negative * 1.5).clip(0, 1) # 增加敏感度
        # 伴随诱多欺骗的正向拐点，惩罚
        bull_trap_mask = (inflection_strength > 0) & (norm_deception > 0) & (~deception_boost_mask) # 排除已处理的增强情境
        deception_modulator.loc[bull_trap_mask] = 1 - (norm_deception.loc[bull_trap_mask] * deception_penalty_sensitivity).clip(0, 1)
        # 伴随诱空洗盘的负向拐点，缓解（视为洗盘）
        bear_trap_mitigation_mask = (inflection_strength < 0) & (norm_wash_trade > 0)
        deception_modulator.loc[bear_trap_mitigation_mask] = 1 + (norm_wash_trade.loc[bear_trap_mitigation_mask] * wash_trade_mitigation_sensitivity).clip(0, 0.5)
        inflection_strength_modulated = inflection_strength * deception_modulator
        # --- 6. 拐点延续性确认奖励 (Inflection Persistence Confirmation Bonus) ---
        persistence_bonus = pd.Series(0.0, index=df_index)
        # 正向拐点在后续几天得到确认
        positive_persistence_mask = (inflection_strength_modulated > 0) & \
                                    (inflection_strength_modulated.rolling(window=persistence_period, min_periods=1).mean() > 0)
        persistence_bonus.loc[positive_persistence_mask] = persistence_bonus_factor
        # 负向拐点在后续几天得到确认
        negative_persistence_mask = (inflection_strength_modulated < 0) & \
                                    (inflection_strength_modulated.rolling(window=persistence_period, min_periods=1).mean() < 0)
        persistence_bonus.loc[negative_persistence_mask] = -persistence_bonus_factor
        # --- 7. 增强情境调制器 (Enhanced Contextual Modulators) ---
        chip_health_raw = signals_data[context_modulator_signal_1_name]
        norm_chip_health = get_adaptive_mtf_normalized_score(chip_health_raw, df_index, ascending=True, tf_weights=tf_weights)
        volatility_instability_raw = signals_data[context_modulator_signal_2_name]
        norm_volatility_instability = get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=False, tf_weights=tf_weights) # 低波动性，情境有利
        main_force_conviction_raw = signals_data[context_modulator_signal_3_name]
        norm_main_force_conviction = get_adaptive_mtf_normalized_score(main_force_conviction_raw.abs(), df_index, ascending=True, tf_weights=tf_weights) # 主力信念强，情境有利
        context_modulator = (
            (1 + norm_chip_health * context_modulator_sensitivity_health) *
            (1 + norm_volatility_instability * context_modulator_sensitivity_volatility) *
            (1 + norm_main_force_conviction * context_modulator_sensitivity_conviction)
        ).clip(0.5, 2.0)
        # --- 最终融合 ---
        final_score = (inflection_strength_modulated * position_sensitivity_factor * context_modulator) + persistence_bonus
        # --- 应用牛市陷阱情境惩罚 ---
        bull_trap_penalty = self._calculate_bull_trap_context_penalty(df)
        final_score = final_score * bull_trap_penalty
        final_score = final_score.clip(-1, 1).fillna(0.0).astype(np.float32)
        print(f"    -> [筹码层] 计算完成 '战略与战术的和谐度' 分数: {final_score.loc[probe_ts] if probe_ts and probe_ts in df_index else final_score.iloc[-1]:.4f}")
        return final_score

    def _diagnose_chip_retail_vulnerability(self, df: pd.DataFrame) -> pd.Series:
        """
        【V2.0 · 诡道诱导版】散户筹码脆弱性指数
        量化散户持仓的集中度、平均成本与当前价格的偏离程度，以及其在市场波动下的潜在抛压。
        高分代表散户筹码结构高度不稳定，易受主力诱导而产生恐慌或盲目追涨行为。
        - 核心升级1: 引入“散户筹码结构脆弱性”维度，评估散户持仓的集中度、分散度及流量主导。
        - 核心升级2: 引入“散户行为极端化”维度，评估散户情绪和行为的非理性程度。
        - 核心升级3: 引入“主力诱导情境”维度，评估主力是否在制造有利于诱导散户的情境。
        - 核心升级4: 引入情境调制器，根据市场波动性和情绪动态调整最终分数。
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        df_index = df.index
        df_dates_set = set(df_index.date)
        probe_dates_in_df = sorted([d for d in self.probe_dates_set if d in df_dates_set])
        should_probe_overall = self.should_probe and bool(probe_dates_in_df)
        p_conf = self.chip_ultimate_params
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        rv_params = get_param_value(p_conf.get('chip_retail_vulnerability_params'), {})
        structure_fragility_weights = get_param_value(rv_params.get('structure_fragility_weights'), {})
        behavior_extremism_weights = get_param_value(rv_params.get('behavior_extremism_weights'), {})
        inducement_context_weights = get_param_value(rv_params.get('inducement_context_weights'), {})
        final_fusion_weights = get_param_value(rv_params.get('final_fusion_weights'), {})
        contextual_modulator_enabled = get_param_value(rv_params.get('contextual_modulator_enabled'), True)
        context_modulator_weights = get_param_value(rv_params.get('context_modulator_weights'), {})
        context_modulator_sensitivity = get_param_value(rv_params.get('context_modulator_sensitivity'), 0.5)
        final_exponent = get_param_value(rv_params.get('final_exponent'), 2.0)
        required_signals = [
            'total_winner_rate_D', 'total_loser_rate_D', 'retail_fomo_premium_index_D',
            'panic_buy_absorption_contribution_D', 'winner_concentration_90pct_D', 'loser_concentration_90pct_D',
            'cost_gini_coefficient_D', 'retail_flow_dominance_index_D', 'retail_net_flow_calibrated_D',
            'deception_index_D', 'wash_trade_intensity_D', 'main_force_conviction_index_D',
            'VOLATILITY_INSTABILITY_INDEX_21d_D', 'market_sentiment_score_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_chip_retail_vulnerability"):
            return pd.Series(0.0, index=df.index)
        signals_data = self._get_all_required_signals(df, required_signals, "_diagnose_chip_retail_vulnerability")
        # --- 调试信息构建 ---
        is_debug_enabled = self.should_probe
        probe_ts = None
        if is_debug_enabled and self.probe_dates_set:
            for date in reversed(df_index):
                if date.date() in self.probe_dates_set:
                    probe_ts = date
                    break
        debug_info_tuple = False # (is_debug_enabled, probe_ts, "_diagnose_chip_retail_vulnerability")
        # --- 原始数据获取 ---
        winner_concentration_raw = signals_data['winner_concentration_90pct_D']
        loser_concentration_raw = signals_data['loser_concentration_90pct_D']
        cost_gini_coefficient_raw = signals_data['cost_gini_coefficient_D']
        retail_flow_dominance_raw = signals_data['retail_flow_dominance_index_D']
        retail_fomo_premium_index_raw = signals_data['retail_fomo_premium_index_D']
        panic_buy_absorption_contribution_raw = signals_data['panic_buy_absorption_contribution_D']
        retail_net_flow_calibrated_raw = signals_data['retail_net_flow_calibrated_D']
        total_winner_rate_raw = signals_data['total_winner_rate_D']
        total_loser_rate_raw = signals_data['total_loser_rate_D']
        deception_index_raw = signals_data['deception_index_D']
        wash_trade_intensity_raw = signals_data['wash_trade_intensity_D']
        main_force_conviction_raw = signals_data['main_force_conviction_index_D']
        volatility_instability_raw = signals_data['VOLATILITY_INSTABILITY_INDEX_21d_D']
        market_sentiment_raw = signals_data['market_sentiment_score_D']
        # --- 维度1: 散户筹码结构脆弱性 (Retail Chip Structure Fragility) ---
        norm_winner_concentration_inverse = get_adaptive_mtf_normalized_score(winner_concentration_raw, df_index, ascending=False, tf_weights=tf_weights) # 赢家集中度低，脆弱性高
        norm_loser_concentration_inverse = get_adaptive_mtf_normalized_score(loser_concentration_raw, df_index, ascending=False, tf_weights=tf_weights) # 输家集中度低，脆弱性高
        norm_cost_gini_coefficient_inverse = get_adaptive_mtf_normalized_score(cost_gini_coefficient_raw, df_index, ascending=False, tf_weights=tf_weights) # 基尼系数低，筹码分散，脆弱性高
        norm_retail_flow_dominance = get_adaptive_mtf_normalized_score(retail_flow_dominance_raw, df_index, ascending=True, tf_weights=tf_weights) # 散户主导，脆弱性高
        structure_fragility_score = _robust_geometric_mean(
            {
                'winner_concentration_inverse': norm_winner_concentration_inverse,
                'loser_concentration_inverse': norm_loser_concentration_inverse,
                'cost_gini_coefficient_inverse': norm_cost_gini_coefficient_inverse,
                'retail_flow_dominance': norm_retail_flow_dominance
            },
            structure_fragility_weights, df_index
        )
        # --- 维度2: 散户行为极端化 (Retail Behavior Extremism) ---
        norm_retail_fomo_premium = get_adaptive_mtf_normalized_score(retail_fomo_premium_index_raw, df_index, ascending=True, tf_weights=tf_weights) # FOMO高，极端化高
        norm_panic_buy_absorption_inverse = get_adaptive_mtf_normalized_score(panic_buy_absorption_contribution_raw, df_index, ascending=False, tf_weights=tf_weights) # 恐慌买入吸收低，极端化高
        norm_retail_net_flow_abs = get_adaptive_mtf_normalized_bipolar_score(retail_net_flow_calibrated_raw, df_index, tf_weights=tf_weights).abs() # 散户净流量绝对值高，极端化高
        norm_total_winner_rate = get_adaptive_mtf_normalized_score(total_winner_rate_raw, df_index, ascending=True, tf_weights=tf_weights) # 赢家比例高，极端化高
        norm_total_loser_rate = get_adaptive_mtf_normalized_score(total_loser_rate_raw, df_index, ascending=True, tf_weights=tf_weights) # 输家比例高，极端化高
        behavior_extremism_score = _robust_geometric_mean(
            {
                'retail_fomo_premium': norm_retail_fomo_premium,
                'panic_buy_absorption_inverse': norm_panic_buy_absorption_inverse,
                'retail_net_flow_abs': norm_retail_net_flow_abs,
                'total_winner_rate': norm_total_winner_rate,
                'total_loser_rate': norm_total_loser_rate
            },
            behavior_extremism_weights, df_index
        )
        # --- 维度3: 主力诱导情境 (Main Force Inducement Context) ---
        norm_deception_index_positive = get_adaptive_mtf_normalized_bipolar_score(deception_index_raw, df_index, tf_weights=tf_weights).clip(lower=0) # 正向欺骗，诱导情境高
        norm_wash_trade_intensity = get_adaptive_mtf_normalized_score(wash_trade_intensity_raw, df_index, ascending=True, tf_weights=tf_weights) # 对倒强度高，诱导情境高
        norm_main_force_conviction_negative_abs = get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, tf_weights=tf_weights).clip(upper=0).abs() # 主力信念负向，诱导情境高
        inducement_context_score = _robust_geometric_mean(
            {
                'deception_index_positive': norm_deception_index_positive,
                'wash_trade_intensity': norm_wash_trade_intensity,
                'main_force_conviction_negative_abs': norm_main_force_conviction_negative_abs
            },
            inducement_context_weights, df_index
        )
        # --- 初始融合 ---
        initial_vulnerability_score = _robust_geometric_mean(
            {
                'structure_fragility': structure_fragility_score,
                'behavior_extremism': behavior_extremism_score,
                'inducement_context': inducement_context_score
            },
            final_fusion_weights, df_index
        )
        # --- 情境调制器 (Contextual Modulator) ---
        final_score = initial_vulnerability_score
        if contextual_modulator_enabled:
            norm_volatility_instability = get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=True, tf_weights=tf_weights) # 波动性高，脆弱性高
            norm_market_sentiment_extreme = get_adaptive_mtf_normalized_bipolar_score(market_sentiment_raw, df_index, tf_weights=tf_weights).abs() # 市场情绪极端，脆弱性高
            modulator = _robust_geometric_mean(
                {
                    'volatility_instability': norm_volatility_instability,
                    'market_sentiment_extreme': norm_market_sentiment_extreme
                },
                context_modulator_weights, df_index
            )
            modulator = 1 + (modulator - 0.5) * context_modulator_sensitivity # 将 [0,1] 映射到 [1-sens/2, 1+sens/2]
            modulator = modulator.clip(0.5, 1.5) # 限制调制范围
            final_score = final_score * modulator
        # --- 最终非线性放大 ---
        final_score = np.tanh(final_score * final_exponent)
        final_score = final_score.clip(0, 1).fillna(0.0).astype(np.float32)
        print(f"    -> [筹码层] 计算完成 '散户筹码脆弱性指数' 分数: {final_score.iloc[-1]}")
        return final_score

    def _diagnose_chip_main_force_cost_intent(self, df: pd.DataFrame) -> pd.Series:
        """
        【V2.1 · 净流量方向修正版】主力成本区攻防意图
        诊断主力资金在其核心持仓成本区域（或关键筹码峰区域）进行主动买入或卖出的强度。
        正分代表主力在其成本区下方或附近积极承接，显示出强烈的防守或吸筹意图；
        负分代表主力在其成本区上方或附近主动派发，显示出减仓或打压意图。
        - 核心升级1: 动态成本区定义：引入 `dominant_peak_cost_D` 作为核心成本区，并根据 `BBW_21_2.0_D` 和 `chip_concentration_90pct_D` 动态调整成本区间的容忍度。
        - 核心升级2: 增强净流量质量：结合 `main_force_activity_ratio_D` 和 `main_force_flow_directionality_D` 调制 `net_conviction_flow`，评估资金流质量。
        - 核心升级3: 非线性价格偏离放大：采用 `tanh` 函数对价格偏离因子进行非线性放大，更敏感地捕捉主力意图。
        - 核心升级4: 情境化成本区内逻辑：在成本区内，结合 `SLOPE_5_chip_health_score_D` 和 `SLOPE_5_main_force_conviction_index_D` 动态调整意图强度。
        - 核心升级5: 诡道调制：引入 `deception_index_D` 和 `wash_trade_intensity_D` 作为调制器，对主力意图进行增强或削弱。
        - 核心升级6: 宏观情境调制：引入 `chip_health_score_D` 和 `VOLATILITY_INSTABILITY_INDEX_21d_D` 作为全局情境调制器，对最终分数进行校准。
        - 核心升级7: 动态权重融合：根据市场波动性和情绪，动态调整不同意图场景（低于、高于、在成本区内）的融合权重。
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        df_index = df.index
        df_dates_set = set(df_index.date)
        probe_dates_in_df = sorted([d for d in self.probe_dates_set if d in df_dates_set])
        should_probe_overall = self.should_probe and bool(probe_dates_in_df)
        p_conf = self.chip_ultimate_params
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        mfci_params = get_param_value(p_conf.get('main_force_cost_intent_params'), {})
        cost_zone_tolerance_base = get_param_value(mfci_params.get('cost_zone_tolerance_base'), 0.02)
        dynamic_tolerance_mod_enabled = get_param_value(mfci_params.get('dynamic_tolerance_mod_enabled'), True)
        dynamic_tolerance_mod_signal = get_param_value(mfci_params.get('dynamic_tolerance_mod_signal'), 'BBW_21_2.0_D')
        dynamic_tolerance_sensitivity = get_param_value(mfci_params.get('dynamic_tolerance_sensitivity'), 0.5)
        price_deviation_tanh_factor = get_param_value(mfci_params.get('price_deviation_tanh_factor'), 5.0)
        in_zone_intent_base_multiplier = get_param_value(mfci_params.get('in_zone_intent_base_multiplier'), 0.5)
        in_zone_health_slope_sensitivity = get_param_value(mfci_params.get('in_zone_health_slope_sensitivity'), 0.3)
        deception_mod_enabled = get_param_value(mfci_params.get('deception_mod_enabled'), True)
        deception_boost_factor = get_param_value(mfci_params.get('deception_boost_factor'), 0.5)
        deception_penalty_factor = get_param_value(mfci_params.get('deception_penalty_factor'), 0.7)
        wash_trade_penalty_factor = get_param_value(mfci_params.get('wash_trade_penalty_factor'), 0.3)
        global_context_mod_enabled = get_param_value(mfci_params.get('global_context_mod_enabled'), True)
        global_context_sensitivity_health = get_param_value(mfci_params.get('global_context_sensitivity_health'), 0.5)
        global_context_sensitivity_volatility = get_param_value(mfci_params.get('global_context_sensitivity_volatility'), 0.3)
        dynamic_fusion_weights_enabled = get_param_value(mfci_params.get('dynamic_fusion_weights_enabled'), True)
        dynamic_fusion_weights_base = get_param_value(mfci_params.get('dynamic_fusion_weights_base'), {'below_vpoc': 0.4, 'above_vpoc': 0.4, 'in_vpoc': 0.2})
        dynamic_weight_mod_signal = get_param_value(mfci_params.get('dynamic_weight_mod_signal'), 'market_sentiment_score_D')
        dynamic_weight_sensitivity = get_param_value(mfci_params.get('dynamic_weight_sensitivity'), 0.3)
        required_signals = [
            'close_D', 'vpoc_D', 'dominant_peak_cost_D', 'main_force_conviction_index_D',
            'conviction_flow_buy_intensity_D', 'conviction_flow_sell_intensity_D',
            'main_force_activity_ratio_D', 'main_force_flow_directionality_D',
            'deception_index_D', 'wash_trade_intensity_D', 'chip_health_score_D',
            'SLOPE_5_chip_health_score_D', 'SLOPE_5_main_force_conviction_index_D',
            'BBW_21_2.0_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D', 'market_sentiment_score_D'
        ]
        if dynamic_tolerance_mod_enabled and dynamic_tolerance_mod_signal not in required_signals:
            required_signals.append(dynamic_tolerance_mod_signal)
        if dynamic_fusion_weights_enabled and dynamic_weight_mod_signal not in required_signals:
            required_signals.append(dynamic_weight_mod_signal)
        if not self._validate_required_signals(df, required_signals, "_diagnose_chip_main_force_cost_intent"):
            return pd.Series(0.0, index=df.index)
        signals_data = self._get_all_required_signals(df, required_signals, "_diagnose_chip_main_force_cost_intent")
        # --- 调试信息构建 ---
        is_debug_enabled = self.should_probe
        probe_ts = None
        if is_debug_enabled and self.probe_dates_set:
            for date in reversed(df_index):
                if date.date() in self.probe_dates_set:
                    probe_ts = date
                    break
        debug_info_tuple = False # (is_debug_enabled, probe_ts, "_diagnose_chip_main_force_cost_intent")
        # --- 原始数据获取 ---
        close_raw = signals_data['close_D']
        vpoc_raw = signals_data['vpoc_D']
        dominant_peak_cost_raw = signals_data['dominant_peak_cost_D']
        main_force_conviction_raw = signals_data['main_force_conviction_index_D']
        conviction_flow_buy_raw = signals_data['conviction_flow_buy_intensity_D']
        conviction_flow_sell_raw = signals_data['conviction_flow_sell_intensity_D']
        main_force_activity_ratio_raw = signals_data['main_force_activity_ratio_D']
        main_force_flow_directionality_raw = signals_data['main_force_flow_directionality_D']
        deception_index_raw = signals_data['deception_index_D']
        wash_trade_intensity_raw = signals_data['wash_trade_intensity_D']
        chip_health_score_raw = signals_data['chip_health_score_D']
        slope_5_chip_health_raw = signals_data['SLOPE_5_chip_health_score_D']
        slope_5_main_force_conviction_raw = signals_data['SLOPE_5_main_force_conviction_index_D']
        volatility_instability_raw = signals_data['VOLATILITY_INSTABILITY_INDEX_21d_D']
        market_sentiment_raw = signals_data['market_sentiment_score_D']
        dynamic_tolerance_mod_raw = signals_data[dynamic_tolerance_mod_signal] if dynamic_tolerance_mod_enabled else None
        dynamic_weight_mod_raw = signals_data[dynamic_weight_mod_signal] if dynamic_fusion_weights_enabled else None
        # --- 1. 动态成本区定义 (Dynamic Cost Zone Definition) ---
        cost_center = dominant_peak_cost_raw.fillna(vpoc_raw)
        dynamic_cost_zone_tolerance = pd.Series(cost_zone_tolerance_base, index=df_index)
        if dynamic_tolerance_mod_enabled and dynamic_tolerance_mod_raw is not None:
            norm_dynamic_tolerance_mod = get_adaptive_mtf_normalized_score(dynamic_tolerance_mod_raw, df_index, ascending=True, tf_weights=tf_weights)
            dynamic_cost_zone_tolerance = cost_zone_tolerance_base * (1 + norm_dynamic_tolerance_mod * dynamic_tolerance_sensitivity)
            dynamic_cost_zone_tolerance = dynamic_cost_zone_tolerance.clip(cost_zone_tolerance_base * 0.5, cost_zone_tolerance_base * 2.0)
        upper_bound = cost_center * (1 + dynamic_cost_zone_tolerance)
        lower_bound = cost_center * (1 - dynamic_cost_zone_tolerance)
        # --- 2. 增强净流量质量 (Enhanced Net Flow Quality) ---
        net_conviction_flow = conviction_flow_buy_raw - conviction_flow_sell_raw
        norm_positive_flow = get_adaptive_mtf_normalized_score(net_conviction_flow.clip(lower=0), df_index, ascending=True, tf_weights=tf_weights)
        norm_negative_flow = get_adaptive_mtf_normalized_score(net_conviction_flow.clip(upper=0).abs(), df_index, ascending=True, tf_weights=tf_weights)
        norm_net_conviction_flow_directional = norm_positive_flow - norm_negative_flow
        norm_main_force_activity = get_adaptive_mtf_normalized_score(main_force_activity_ratio_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_main_force_flow_directionality = get_adaptive_mtf_normalized_bipolar_score(main_force_flow_directionality_raw, df_index, tf_weights=tf_weights)
        flow_quality_modulator = (norm_main_force_activity * 0.5 + norm_main_force_flow_directionality.abs() * 0.5).clip(0, 1)
        net_conviction_flow_quality = norm_net_conviction_flow_directional * (1 + flow_quality_modulator * 0.5)
        net_conviction_flow_quality = net_conviction_flow_quality.clip(-1, 1)
        # --- 3. 非线性价格偏离放大 (Non-linear Price Deviation Amplification) ---
        norm_main_force_conviction = get_adaptive_mtf_normalized_score(main_force_conviction_raw.abs(), df_index, ascending=True, tf_weights=tf_weights)
        price_deviation_factor_buy = (cost_center - close_raw) / cost_center.replace(0, np.nan)
        price_deviation_factor_buy = np.tanh(price_deviation_factor_buy.clip(0, 0.1) * price_deviation_tanh_factor) # 价格低于成本中心越多，买入意图越强
        price_deviation_factor_sell = (close_raw - cost_center) / cost_center.replace(0, np.nan)
        price_deviation_factor_sell = np.tanh(price_deviation_factor_sell.clip(0, 0.1) * price_deviation_tanh_factor) # 价格高于成本中心越多，卖出意图越强
        # --- 4. 情境化成本区内逻辑 (Contextual Logic within Cost Zone) ---
        norm_slope_5_chip_health = get_adaptive_mtf_normalized_bipolar_score(slope_5_chip_health_raw, df_index, tf_weights=tf_weights)
        norm_slope_5_main_force_conviction = get_adaptive_mtf_normalized_bipolar_score(slope_5_main_force_conviction_raw, df_index, tf_weights=tf_weights)
        in_zone_intent_modulator = (norm_slope_5_chip_health * 0.5 + norm_slope_5_main_force_conviction * 0.5).clip(-1, 1)
        # --- 5. 诡道调制 (Deception Modulation) ---
        deception_modulator = pd.Series(1.0, index=df_index)
        if deception_mod_enabled:
            norm_deception_index_bipolar = get_adaptive_mtf_normalized_bipolar_score(deception_index_raw, df_index, tf_weights=tf_weights)
            norm_wash_trade_intensity = get_adaptive_mtf_normalized_score(wash_trade_intensity_raw, df_index, ascending=True, tf_weights=tf_weights)
            # 诱空增强 (净流量为正且有负向欺骗)
            bear_trap_boost_mask = (net_conviction_flow_quality > 0) & (norm_deception_index_bipolar < 0)
            deception_modulator.loc[bear_trap_boost_mask] = deception_modulator.loc[bear_trap_boost_mask] * (1 + norm_deception_index_bipolar.loc[bear_trap_boost_mask].abs() * deception_boost_factor)
            # 诱多增强 (净流量为负且有正向欺骗)
            bull_trap_boost_mask = (net_conviction_flow_quality < 0) & (norm_deception_index_bipolar > 0)
            deception_modulator.loc[bull_trap_boost_mask] = deception_modulator.loc[bull_trap_boost_mask] * (1 + norm_deception_index_bipolar.loc[bull_trap_boost_mask] * deception_boost_factor)
            # 诱多惩罚 (净流量为正且有正向欺骗)
            bull_trap_penalty_mask = (net_conviction_flow_quality > 0) & (norm_deception_index_bipolar > 0)
            deception_modulator.loc[bull_trap_penalty_mask] = deception_modulator.loc[bull_trap_penalty_mask] * (1 - norm_deception_index_bipolar.loc[bull_trap_penalty_mask] * deception_penalty_factor)
            # 对倒惩罚
            deception_modulator = deception_modulator * (1 - norm_wash_trade_intensity * wash_trade_penalty_factor)
            deception_modulator = deception_modulator.clip(0.1, 2.0)
        # --- 6. 宏观情境调制 (Global Context Modulation) ---
        global_context_modulator = pd.Series(1.0, index=df_index)
        if global_context_mod_enabled:
            norm_chip_health = get_adaptive_mtf_normalized_score(chip_health_score_raw, df_index, ascending=True, tf_weights=tf_weights)
            norm_volatility_instability = get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=False, tf_weights=tf_weights)
            global_context_modulator = (
                (1 + norm_chip_health * global_context_sensitivity_health) *
                (1 + norm_volatility_instability * global_context_sensitivity_volatility)
            ).clip(0.5, 1.5)
        # --- 7. 动态权重融合 (Dynamic Fusion Weights) ---
        dynamic_weight_below_vpoc = pd.Series(dynamic_fusion_weights_base.get('below_vpoc', 0.4), index=df_index)
        dynamic_weight_above_vpoc = pd.Series(dynamic_fusion_weights_base.get('above_vpoc', 0.4), index=df_index)
        dynamic_weight_in_vpoc = pd.Series(dynamic_fusion_weights_base.get('in_vpoc', 0.2), index=df_index)
        if dynamic_fusion_weights_enabled and dynamic_weight_mod_raw is not None:
            norm_dynamic_weight_mod = get_adaptive_mtf_normalized_bipolar_score(dynamic_weight_mod_raw, df_index, tf_weights=tf_weights)
            mod_factor = norm_dynamic_weight_mod * dynamic_weight_sensitivity
            # 情绪看涨时，增加 below_vpoc 权重，减少 above_vpoc 权重
            dynamic_weight_below_vpoc = (dynamic_weight_below_vpoc - mod_factor).clip(0.1, 0.7)
            dynamic_weight_above_vpoc = (dynamic_weight_above_vpoc + mod_factor).clip(0.1, 0.7)
            dynamic_weight_in_vpoc = (dynamic_weight_in_vpoc - mod_factor.abs() * 0.5).clip(0.05, 0.4) # 情绪极端时，in_vpoc 权重降低
            sum_dynamic_weights = dynamic_weight_below_vpoc + dynamic_weight_above_vpoc + dynamic_weight_in_vpoc
            dynamic_weight_below_vpoc /= sum_dynamic_weights
            dynamic_weight_above_vpoc /= sum_dynamic_weights
            dynamic_weight_in_vpoc /= sum_dynamic_weights
        # --- 8. 计算主力成本区意图 (Main Force Cost Intent Calculation) ---
        main_force_cost_intent_raw = pd.Series(0.0, index=df_index)
        # 价格低于成本区 (Below VPOC)
        mask_below_vpoc = (close_raw < lower_bound)
        intent_below_vpoc = (
            net_conviction_flow_quality.loc[mask_below_vpoc].clip(lower=0) * # 净流量为正
            norm_main_force_conviction.loc[mask_below_vpoc] *
            (1 + price_deviation_factor_buy.loc[mask_below_vpoc]) * # 价格偏离越远，意图越强
            dynamic_weight_below_vpoc.loc[mask_below_vpoc]
        )
        main_force_cost_intent_raw.loc[mask_below_vpoc] = intent_below_vpoc
        # 价格高于成本区 (Above VPOC)
        mask_above_vpoc = (close_raw > upper_bound)
        intent_above_vpoc = -( # 负分代表派发意图
            net_conviction_flow_quality.loc[mask_above_vpoc].clip(upper=0).abs() * # 净流量为负
            norm_main_force_conviction.loc[mask_above_vpoc] *
            (1 + price_deviation_factor_sell.loc[mask_above_vpoc]) * # 价格偏离越远，意图越强
            dynamic_weight_above_vpoc.loc[mask_above_vpoc]
        )
        main_force_cost_intent_raw.loc[mask_above_vpoc] = intent_above_vpoc
        # 价格在成本区内 (In VPOC)
        mask_in_vpoc = (close_raw >= lower_bound) & (close_raw <= upper_bound)
        intent_in_vpoc = (
            net_conviction_flow_quality.loc[mask_in_vpoc] * # 净流量方向决定意图方向
            norm_main_force_conviction.loc[mask_in_vpoc] *
            (in_zone_intent_base_multiplier + in_zone_intent_modulator.loc[mask_in_vpoc].clip(-0.5, 0.5) * in_zone_health_slope_sensitivity) * # 成本区内，健康度斜率和主力信念斜率调制
            dynamic_weight_in_vpoc.loc[mask_in_vpoc]
        )
        main_force_cost_intent_raw.loc[mask_in_vpoc] = intent_in_vpoc
        # --- 最终融合 ---
        final_score = main_force_cost_intent_raw * deception_modulator * global_context_modulator
        final_score = final_score.clip(-1, 1).fillna(0.0).astype(np.float32)
        print(f"    -> [筹码层] 计算完成 '主力成本区攻防意图' 分数: {final_score.iloc[-1]}")
        return final_score

    def _diagnose_chip_hollowing_out_risk(self, df: pd.DataFrame) -> pd.Series:
        """
        【V3.0 · 深度结构与诡道情境版】筹码空心化风险
        评估筹码结构中，主力核心持仓的稳定性与数量，以及高位套牢盘或短期获利盘的比例。
        高分代表主力核心筹码正在流失，市场筹码结构出现“空心化”迹象，即大部分筹码由不稳定资金在高位持有，
        一旦下跌容易引发连锁抛售。
        - 核心升级1: 引入四大核心维度：核心筹码分散与弱化、派发压力与获利了结、主力意图与诡道、市场情境与脆弱性。
        - 核心升级2: 动态融合权重：根据市场波动性和情绪动态调整四大维度的融合权重，使信号自适应市场环境。
        - 核心升级3: 诡道放大机制：主力意图与诡道维度对最终分数进行乘性放大，增强对欺骗性风险的识别。
        - 核心升级4: 引入更多筹码相关原始数据，并结合其斜率，更全面地捕捉空心化风险的动态演变。
        - 探针增强: 详细输出所有原始数据、归一化数据、各维度子分数、动态权重、最终分数，以便于检查和调试。
        """
        df_index = df.index
        p_conf = self.chip_ultimate_params
        hollow_params = get_param_value(p_conf.get('chip_hollowing_out_risk_params'), {})
        probe_enabled = get_param_value(hollow_params.get('probe_enabled'), False)
        should_probe_overall = self.should_probe and probe_enabled
        df_dates_set = set(df_index.date)
        probe_dates_in_df = sorted([d for d in self.probe_dates_set if d in df_dates_set])
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        dispersion_weakness_weights = get_param_value(hollow_params.get('dispersion_weakness_weights'), {})
        distribution_pressure_weights = get_param_value(hollow_params.get('distribution_pressure_weights'), {})
        main_force_deception_weights = get_param_value(hollow_params.get('main_force_deception_weights'), {})
        market_vulnerability_weights = get_param_value(hollow_params.get('market_vulnerability_weights'), {})
        final_fusion_weights_base = get_param_value(hollow_params.get('final_fusion_weights'), {})
        dynamic_fusion_modulator_params = get_param_value(hollow_params.get('dynamic_fusion_modulator_params'), {})
        deception_amplification_factor = get_param_value(hollow_params.get('deception_amplification_factor'), 1.5)
        non_linear_exponent = get_param_value(hollow_params.get('non_linear_exponent'), 2.0)
        required_signals = [
            'winner_concentration_90pct_D', 'loser_concentration_90pct_D', 'cost_gini_coefficient_D',
            'dominant_peak_solidity_D', 'peak_separation_ratio_D', 'SLOPE_5_winner_concentration_90pct_D',
            'SLOPE_5_chip_health_score_D', 'total_winner_rate_D', 'winner_profit_margin_avg_D',
            'rally_distribution_pressure_D', 'profit_taking_flow_ratio_D', 'upper_shadow_selling_pressure_D',
            'covert_distribution_signal_D', 'SLOPE_5_rally_distribution_pressure_D',
            'deception_index_D', 'wash_trade_intensity_D', 'main_force_conviction_index_D',
            'main_force_net_flow_calibrated_D', 'main_force_cost_advantage_D',
            'retail_fomo_premium_index_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D',
            'market_sentiment_score_D', 'flow_credibility_index_D', 'structural_tension_index_D'
        ]
        if get_param_value(dynamic_fusion_modulator_params.get('enabled'), False):
            mod_signal_1 = get_param_value(dynamic_fusion_modulator_params.get('modulator_signal_1'))
            mod_signal_2 = get_param_value(dynamic_fusion_modulator_params.get('modulator_signal_2'))
            if mod_signal_1 and mod_signal_1 not in required_signals: required_signals.append(mod_signal_1)
            if mod_signal_2 and mod_signal_2 not in required_signals: required_signals.append(mod_signal_2)
        if not self._validate_required_signals(df, required_signals, "_diagnose_chip_hollowing_out_risk"):
            return pd.Series(0.0, index=df.index)
        signals_data = self._get_all_required_signals(df, required_signals, "_diagnose_chip_hollowing_out_risk")
        # --- 调试信息构建 ---
        is_debug_enabled = self.should_probe and probe_enabled
        probe_ts = None
        if is_debug_enabled and self.probe_dates_set:
            for date in reversed(df_index):
                if date.date() in self.probe_dates_set:
                    probe_ts = date
                    break
        debug_info_tuple = False # (is_debug_enabled, probe_ts, "_diagnose_chip_hollowing_out_risk")
        # --- 原始数据获取 ---
        # Dimension 1: Dispersion & Weakness
        winner_concentration_raw = signals_data['winner_concentration_90pct_D']
        loser_concentration_raw = signals_data['loser_concentration_90pct_D']
        cost_gini_coefficient_raw = signals_data['cost_gini_coefficient_D']
        dominant_peak_solidity_raw = signals_data['dominant_peak_solidity_D']
        peak_separation_ratio_raw = signals_data['peak_separation_ratio_D']
        slope_winner_concentration_raw = signals_data['SLOPE_5_winner_concentration_90pct_D']
        slope_chip_health_raw = signals_data['SLOPE_5_chip_health_score_D']
        # Dimension 2: Distribution Pressure
        total_winner_rate_raw = signals_data['total_winner_rate_D']
        winner_profit_margin_avg_raw = signals_data['winner_profit_margin_avg_D']
        rally_distribution_pressure_raw = signals_data['rally_distribution_pressure_D']
        profit_taking_flow_ratio_raw = signals_data['profit_taking_flow_ratio_D']
        upper_shadow_selling_pressure_raw = signals_data['upper_shadow_selling_pressure_D']
        covert_distribution_signal_raw = signals_data['covert_distribution_signal_D']
        slope_rally_distribution_pressure_raw = signals_data['SLOPE_5_rally_distribution_pressure_D']
        # Dimension 3: Main Force Deception
        deception_index_raw = signals_data['deception_index_D']
        wash_trade_intensity_raw = signals_data['wash_trade_intensity_D']
        main_force_conviction_raw = signals_data['main_force_conviction_index_D']
        main_force_net_flow_calibrated_raw = signals_data['main_force_net_flow_calibrated_D']
        main_force_cost_advantage_raw = signals_data['main_force_cost_advantage_D']
        # Dimension 4: Market Context & Vulnerability
        retail_fomo_premium_index_raw = signals_data['retail_fomo_premium_index_D']
        volatility_instability_raw = signals_data['VOLATILITY_INSTABILITY_INDEX_21d_D']
        market_sentiment_raw = signals_data['market_sentiment_score_D']
        flow_credibility_raw = signals_data['flow_credibility_index_D']
        structural_tension_raw = signals_data['structural_tension_index_D']
        # --- 1. 核心筹码分散与弱化 (Core Chip Dispersion & Weakness) ---
        norm_winner_concentration_inverse = get_adaptive_mtf_normalized_score(winner_concentration_raw, df_index, ascending=False, tf_weights=tf_weights) # 赢家集中度低，风险高
        norm_loser_concentration_high_price = get_adaptive_mtf_normalized_score(loser_concentration_raw, df_index, ascending=True, tf_weights=tf_weights) # 输家集中在高价区，风险高
        norm_cost_gini_coefficient_inverse = get_adaptive_mtf_normalized_score(cost_gini_coefficient_raw, df_index, ascending=False, tf_weights=tf_weights) # 基尼系数低，筹码分散，风险高
        norm_dominant_peak_solidity_inverse = get_adaptive_mtf_normalized_score(dominant_peak_solidity_raw, df_index, ascending=False, tf_weights=tf_weights) # 主峰坚实度低，风险高
        norm_peak_separation_ratio = get_adaptive_mtf_normalized_score(peak_separation_ratio_raw, df_index, ascending=True, tf_weights=tf_weights) # 峰分离比高，风险高
        norm_winner_concentration_slope_inverse = get_adaptive_mtf_normalized_score(slope_winner_concentration_raw, df_index, ascending=False, tf_weights=tf_weights) # 赢家集中度斜率下降，风险高
        norm_chip_health_score_inverse_slope = get_adaptive_mtf_normalized_score(slope_chip_health_raw, df_index, ascending=False, tf_weights=tf_weights) # 筹码健康度斜率下降，风险高
        dispersion_weakness_score = _robust_geometric_mean(
            {
                'winner_concentration_inverse': norm_winner_concentration_inverse,
                'loser_concentration_high_price': norm_loser_concentration_high_price,
                'cost_gini_coefficient_inverse': norm_cost_gini_coefficient_inverse,
                'dominant_peak_solidity_inverse': norm_dominant_peak_solidity_inverse,
                'peak_separation_ratio': norm_peak_separation_ratio,
                'winner_concentration_slope_inverse': norm_winner_concentration_slope_inverse,
                'chip_health_score_inverse_slope': norm_chip_health_score_inverse_slope
            },
            dispersion_weakness_weights, df_index
        )
        # --- 2. 派发压力与获利了结 (Distribution Pressure & Profit-Taking) ---
        norm_total_winner_rate = get_adaptive_mtf_normalized_score(total_winner_rate_raw, df_index, ascending=True, tf_weights=tf_weights) # 赢家比例高，压力高
        norm_winner_profit_margin_avg = get_adaptive_mtf_normalized_score(winner_profit_margin_avg_raw, df_index, ascending=True, tf_weights=tf_weights) # 平均利润高，压力高
        norm_rally_distribution_pressure = get_adaptive_mtf_normalized_score(rally_distribution_pressure_raw, df_index, ascending=True, tf_weights=tf_weights) # 反弹派发压力高，压力高
        norm_profit_taking_flow_ratio = get_adaptive_mtf_normalized_score(profit_taking_flow_ratio_raw, df_index, ascending=True, tf_weights=tf_weights) # 获利了结流量高，压力高
        norm_upper_shadow_selling_pressure = get_adaptive_mtf_normalized_score(upper_shadow_selling_pressure_raw, df_index, ascending=True, tf_weights=tf_weights) # 上影线抛压高，压力高
        norm_covert_distribution_signal = get_adaptive_mtf_normalized_score(covert_distribution_signal_raw, df_index, ascending=True, tf_weights=tf_weights) # 隐蔽派发高，压力高
        norm_rally_distribution_pressure_slope = get_adaptive_mtf_normalized_score(slope_rally_distribution_pressure_raw, df_index, ascending=True, tf_weights=tf_weights) # 派发压力斜率上升，压力高
        distribution_pressure_score = _robust_geometric_mean(
            {
                'total_winner_rate': norm_total_winner_rate,
                'winner_profit_margin_avg': norm_winner_profit_margin_avg,
                'rally_distribution_pressure': norm_rally_distribution_pressure,
                'profit_taking_flow_ratio': norm_profit_taking_flow_ratio,
                'upper_shadow_selling_pressure': norm_upper_shadow_selling_pressure,
                'covert_distribution_signal': norm_covert_distribution_signal,
                'rally_distribution_pressure_slope': norm_rally_distribution_pressure_slope
            },
            distribution_pressure_weights, df_index
        )
        # --- 3. 主力意图与诡道 (Main Force Intent & Deception) ---
        norm_deception_index_positive = get_adaptive_mtf_normalized_bipolar_score(deception_index_raw, df_index, tf_weights=tf_weights).clip(lower=0) # 正向欺骗，风险高
        norm_wash_trade_intensity = get_adaptive_mtf_normalized_score(wash_trade_intensity_raw, df_index, ascending=True, tf_weights=tf_weights) # 对倒强度高，风险高
        norm_main_force_conviction_negative = get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, tf_weights=tf_weights).clip(upper=0).abs() # 主力信念负向，风险高
        norm_main_force_net_flow_negative = get_adaptive_mtf_normalized_score(main_force_net_flow_calibrated_raw, df_index, ascending=False, tf_weights=tf_weights) # 主力净流出，风险高
        norm_main_force_cost_advantage_negative = get_adaptive_mtf_normalized_bipolar_score(main_force_cost_advantage_raw, df_index, tf_weights=tf_weights).clip(upper=0).abs() # 主力成本优势负向，风险高
        main_force_deception_score = _robust_geometric_mean(
            {
                'deception_index_positive': norm_deception_index_positive,
                'wash_trade_intensity': norm_wash_trade_intensity,
                'main_force_conviction_negative': norm_main_force_conviction_negative,
                'main_force_net_flow_negative': norm_main_force_net_flow_negative,
                'main_force_cost_advantage_negative': norm_main_force_cost_advantage_negative
            },
            main_force_deception_weights, df_index
        )
        # --- 4. 市场情境与脆弱性 (Market Context & Vulnerability) ---
        norm_retail_fomo_premium_index = get_adaptive_mtf_normalized_score(retail_fomo_premium_index_raw, df_index, ascending=True, tf_weights=tf_weights) # 散户FOMO高，脆弱性高
        norm_volatility_instability_index = get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=True, tf_weights=tf_weights) # 波动性高，脆弱性高
        norm_market_sentiment_extreme = get_adaptive_mtf_normalized_bipolar_score(market_sentiment_raw, df_index, tf_weights=tf_weights).abs() # 市场情绪极端，脆弱性高
        norm_flow_credibility_inverse = get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, ascending=False, tf_weights=tf_weights) # 资金流可信度低，脆弱性高
        norm_structural_tension_index = get_adaptive_mtf_normalized_score(structural_tension_raw, df_index, ascending=True, tf_weights=tf_weights) # 结构紧张度高，脆弱性高
        market_vulnerability_score = _robust_geometric_mean(
            {
                'retail_fomo_premium_index': norm_retail_fomo_premium_index,
                'volatility_instability_index': norm_volatility_instability_index,
                'market_sentiment_extreme': norm_market_sentiment_extreme,
                'flow_credibility_inverse': norm_flow_credibility_inverse,
                'structural_tension_index': norm_structural_tension_index
            },
            market_vulnerability_weights, df_index
        )
        # --- 动态融合权重 ---
        dynamic_fusion_weights = final_fusion_weights_base.copy()
        if get_param_value(dynamic_fusion_modulator_params.get('enabled'), False):
            mod_signal_1_name = get_param_value(dynamic_fusion_modulator_params.get('modulator_signal_1'))
            mod_signal_2_name = get_param_value(dynamic_fusion_modulator_params.get('modulator_signal_2'))
            sensitivity_volatility = get_param_value(dynamic_fusion_modulator_params.get('sensitivity_volatility'))
            sensitivity_sentiment = get_param_value(dynamic_fusion_modulator_params.get('sensitivity_sentiment'))
            norm_mod_signal_1 = get_adaptive_mtf_normalized_score(signals_data[mod_signal_1_name], df_index, ascending=True, tf_weights=tf_weights)
            norm_mod_signal_2 = get_adaptive_mtf_normalized_bipolar_score(signals_data[mod_signal_2_name], df_index, tf_weights=tf_weights)
            current_dispersion_weight = pd.Series(final_fusion_weights_base.get('dispersion_weakness', 0.0), index=df_index)
            current_distribution_weight = pd.Series(final_fusion_weights_base.get('distribution_pressure', 0.0), index=df_index)
            current_deception_weight = pd.Series(final_fusion_weights_base.get('main_force_deception', 0.0), index=df_index)
            current_vulnerability_weight = pd.Series(final_fusion_weights_base.get('market_vulnerability', 0.0), index=df_index)
            # 波动性对权重的影响
            volatility_impact_weights = get_param_value(dynamic_fusion_modulator_params.get('volatility_impact_weights'), {})
            current_dispersion_weight += norm_mod_signal_1 * sensitivity_volatility * volatility_impact_weights.get('dispersion_weakness', 0.0)
            current_distribution_weight += norm_mod_signal_1 * sensitivity_volatility * volatility_impact_weights.get('distribution_pressure', 0.0)
            current_deception_weight += norm_mod_signal_1 * sensitivity_volatility * volatility_impact_weights.get('main_force_deception', 0.0)
            current_vulnerability_weight += norm_mod_signal_1 * sensitivity_volatility * volatility_impact_weights.get('market_vulnerability', 0.0)
            # 情绪对权重的影响
            sentiment_impact_weights = get_param_value(dynamic_fusion_modulator_params.get('sentiment_impact_weights'), {})
            current_dispersion_weight += norm_mod_signal_2 * sensitivity_sentiment * sentiment_impact_weights.get('dispersion_weakness', 0.0)
            current_distribution_weight += norm_mod_signal_2 * sensitivity_sentiment * sentiment_impact_weights.get('distribution_pressure', 0.0)
            current_deception_weight += norm_mod_signal_2 * sensitivity_sentiment * sentiment_impact_weights.get('main_force_deception', 0.0)
            current_vulnerability_weight += norm_mod_signal_2 * sensitivity_sentiment * sentiment_impact_weights.get('market_vulnerability', 0.0)
            # 裁剪并重新归一化权重
            current_dispersion_weight = current_dispersion_weight.clip(0.05, 0.5)
            current_distribution_weight = current_distribution_weight.clip(0.05, 0.5)
            current_deception_weight = current_deception_weight.clip(0.05, 0.5)
            current_vulnerability_weight = current_vulnerability_weight.clip(0.05, 0.5)
            sum_dynamic_weights = current_dispersion_weight + current_distribution_weight + current_deception_weight + current_vulnerability_weight
            dynamic_fusion_weights['dispersion_weakness'] = current_dispersion_weight / sum_dynamic_weights
            dynamic_fusion_weights['distribution_pressure'] = current_distribution_weight / sum_dynamic_weights
            dynamic_fusion_weights['main_force_deception'] = current_deception_weight / sum_dynamic_weights
            dynamic_fusion_weights['market_vulnerability'] = current_vulnerability_weight / sum_dynamic_weights
        else:
            dynamic_fusion_weights = {k: pd.Series(v, index=df_index) for k, v in final_fusion_weights_base.items()}
        # --- 最终融合 ---
        hollowing_out_risk_score = (
            dispersion_weakness_score * dynamic_fusion_weights['dispersion_weakness'] +
            distribution_pressure_score * dynamic_fusion_weights['distribution_pressure'] +
            main_force_deception_score * dynamic_fusion_weights['main_force_deception'] +
            market_vulnerability_score * dynamic_fusion_weights['market_vulnerability']
        )
        # 诡道放大机制
        deception_amplifier = 1 + main_force_deception_score * deception_amplification_factor
        hollowing_out_risk_score = hollowing_out_risk_score * deception_amplifier
        final_score = np.tanh(hollowing_out_risk_score.clip(0, 1) ** non_linear_exponent).clip(0, 1).fillna(0.0).astype(np.float32)
        print(f"    -> [筹码层] 计算完成 '筹码空心化风险' 分数: {final_score.iloc[-1]}")
        return final_score

    def _diagnose_chip_turnover_purity_cost_optimization(self, df: pd.DataFrame) -> pd.Series:
        """
        【筹码】换手纯度与成本优化
        评估换手过程中，筹码从高成本、不稳定持仓向低成本、稳定持仓转移的效率和纯度。
        高分代表换手是健康的，有助于优化筹码结构，降低整体持仓成本，为后续上涨奠定基础；
        低分或负分代表换手是恶性的，筹码从低成本向高成本转移，或伴随大量对倒和虚假交易。
        """
        df_index = df.index
        required_signals = [
            'wash_trade_intensity_D', 'conviction_flow_buy_intensity_D', 'conviction_flow_sell_intensity_D',
            'winner_profit_margin_avg_D', 'loser_pain_index_D', 'turnover_rate_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_chip_turnover_purity_cost_optimization"):
            return pd.Series(0.0, index=df.index)
        signals_data = self._get_all_required_signals(df, required_signals, "_diagnose_chip_turnover_purity_cost_optimization")
        p_conf = self.chip_ultimate_params
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # --- 调试信息构建 ---
        is_debug_enabled = self.should_probe
        probe_ts = None
        if is_debug_enabled and self.probe_dates_set:
            for date in reversed(df_index):
                if date.date() in self.probe_dates_set:
                    probe_ts = date
                    break
        debug_info_tuple = False # (is_debug_enabled, probe_ts, "_diagnose_chip_turnover_purity_cost_optimization")
        # 获取原始信号
        wash_trade_intensity_raw = signals_data['wash_trade_intensity_D']
        conviction_flow_buy_raw = signals_data['conviction_flow_buy_intensity_D']
        conviction_flow_sell_raw = signals_data['conviction_flow_sell_intensity_D']
        winner_profit_margin_avg_raw = signals_data['winner_profit_margin_avg_D']
        loser_pain_index_raw = signals_data['loser_pain_index_D']
        turnover_rate_raw = signals_data['turnover_rate_D']
        # 归一化各项指标
        norm_wash_trade_intensity = get_adaptive_mtf_normalized_score(wash_trade_intensity_raw, df_index, ascending=False, tf_weights=tf_weights) # 对倒强度低，纯度高
        net_conviction_flow = conviction_flow_buy_raw - conviction_flow_sell_raw
        norm_net_conviction_flow = get_adaptive_mtf_normalized_bipolar_score(net_conviction_flow, df_index, tf_weights=tf_weights) # 净信念流向
        norm_winner_profit_margin_avg = get_adaptive_mtf_normalized_score(winner_profit_margin_avg_raw, df_index, ascending=False, tf_weights=tf_weights) # 赢家平均利润低，成本优化好
        norm_loser_pain_index = get_adaptive_mtf_normalized_score(loser_pain_index_raw, df_index, ascending=True, tf_weights=tf_weights) # 输家痛苦高，成本优化好
        norm_turnover_rate = get_adaptive_mtf_normalized_score(turnover_rate_raw, df_index, ascending=True, tf_weights=tf_weights) # 换手率高，强度高
        # 纯度因子 (0到1，1为纯净)
        purity_factor = (1 - norm_wash_trade_intensity)
        # 成本优化因子 (0到1，1为优化好)
        cost_optimization_factor = (norm_winner_profit_margin_avg + norm_loser_pain_index) / 2
        # 换手质量因子 (双极，-1到1)
        # 纯度 * 成本优化 * 净信念流向
        turnover_quality_factor = purity_factor * cost_optimization_factor * norm_net_conviction_flow
        # 结合换手率强度进行调制
        turnover_purity_cost_optimization = turnover_quality_factor * (1 + norm_turnover_rate * 0.5) # 换手率越高，调制效果越强
        final_score = turnover_purity_cost_optimization.clip(-1, 1).fillna(0.0).astype(np.float32)
        print(f"    -> [筹码层] 计算完成 '换手纯度与成本优化' 分数: {final_score.iloc[-1]}")
        return final_score

    def _diagnose_chip_despair_temptation_zones(self, df: pd.DataFrame) -> pd.Series:
        """
        【筹码】筹码绝望与诱惑区
        识别当前筹码分布中，散户或弱势资金处于极端亏损（绝望区）或极端浮盈（诱惑区）的价格区间。
        正分代表诱惑区风险（主力派发），负分代表绝望区机会（主力吸筹）。
        """
        df_index = df.index
        required_signals = [
            'loser_pain_index_D', 'total_loser_rate_D', 'panic_buy_absorption_contribution_D',
            'retail_fomo_premium_index_D', 'winner_profit_margin_avg_D', 'total_winner_rate_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_chip_despair_temptation_zones"):
            return pd.Series(0.0, index=df.index)
        signals_data = self._get_all_required_signals(df, required_signals, "_diagnose_chip_despair_temptation_zones")
        p_conf = self.chip_ultimate_params
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # --- 调试信息构建 ---
        is_debug_enabled = self.should_probe
        probe_ts = None
        if is_debug_enabled and self.probe_dates_set:
            for date in reversed(df_index):
                if date.date() in self.probe_dates_set:
                    probe_ts = date
                    break
        debug_info_tuple = False # (is_debug_enabled, probe_ts, "_diagnose_chip_despair_temptation_zones")
        # 获取原始信号
        loser_pain_index_raw = signals_data['loser_pain_index_D']
        total_loser_rate_raw = signals_data['total_loser_rate_D']
        panic_buy_absorption_contribution_raw = signals_data['panic_buy_absorption_contribution_D']
        retail_fomo_premium_index_raw = signals_data['retail_fomo_premium_index_D']
        winner_profit_margin_avg_raw = signals_data['winner_profit_margin_avg_D']
        total_winner_rate_raw = signals_data['total_winner_rate_D'] # 重新获取，因为上面被覆盖了
        # 归一化绝望区相关指标 (高值代表更绝望)
        norm_loser_pain_index = get_adaptive_mtf_normalized_score(loser_pain_index_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_total_loser_rate = get_adaptive_mtf_normalized_score(total_loser_rate_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_panic_buy_absorption_contribution = get_adaptive_mtf_normalized_score(panic_buy_absorption_contribution_raw, df_index, ascending=False, tf_weights=tf_weights) # 吸收贡献低，绝望区强度高
        # 归一化诱惑区相关指标 (高值代表更诱惑)
        norm_retail_fomo_premium_index = get_adaptive_mtf_normalized_score(retail_fomo_premium_index_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_winner_profit_margin_avg = get_adaptive_mtf_normalized_score(winner_profit_margin_avg_raw, df_index, ascending=True, tf_weights=tf_weights)
        norm_total_winner_rate_temptation = get_adaptive_mtf_normalized_score(total_winner_rate_raw, df_index, ascending=True, tf_weights=tf_weights) # 赢家比例高，诱惑区强度高
        # 计算绝望区强度 (0到1)
        despair_strength = (
            norm_loser_pain_index.pow(0.4) *
            norm_total_loser_rate.pow(0.3) *
            norm_panic_buy_absorption_contribution.pow(0.3)
        ).pow(1 / 1.0) # 权重和为1
        # 计算诱惑区强度 (0到1)
        temptation_strength = (
            norm_retail_fomo_premium_index.pow(0.4) *
            norm_winner_profit_margin_avg.pow(0.3) *
            norm_total_winner_rate_temptation.pow(0.3)
        ).pow(1 / 1.0) # 权重和为1
        # 结合成双极分数：正分代表诱惑区风险，负分代表绝望区机会
        despair_temptation_score = temptation_strength - despair_strength
        # 进一步非线性放大，并映射到 [-1, 1]
        final_score = np.tanh(despair_temptation_score * 2) # 放大因子2
        final_score = final_score.clip(-1, 1).fillna(0.0).astype(np.float32)
        print(f"    -> [筹码层] 计算完成 '筹码绝望与诱惑区' 分数: {final_score.iloc[-1]}")
        return final_score

    def _calculate_bull_trap_context_penalty(self, df: pd.DataFrame) -> pd.Series:
        """
        【V1.0 · 牛市陷阱情境惩罚】计算在近期大幅下跌后，伴随欺骗性反弹情境下的惩罚因子。
        - 核心逻辑: 检测近期是否存在大幅下跌，同时当前是否存在正向欺骗信号。
                    结合市场波动性作为情境调制器，动态调整惩罚强度。
        - 返回值: 一个 Series，值为 0 到 1 之间。1 表示无惩罚，0 表示完全惩罚。
        """
        df_index = df.index
        p_conf = self.chip_ultimate_params
        bt_params = get_param_value(p_conf.get('bull_trap_detection_params'), {})
        if not get_param_value(bt_params.get('enabled'), False):
            return pd.Series(1.0, index=df_index, dtype=np.float32)
        recent_sharp_drop_window = get_param_value(bt_params.get('recent_sharp_drop_window'), 3)
        min_sharp_drop_pct = get_param_value(bt_params.get('min_sharp_drop_pct'), -0.05)
        deception_penalty_multiplier = get_param_value(bt_params.get('deception_penalty_multiplier'), 2.5)
        context_modulator_signal_name = get_param_value(bt_params.get('context_modulator_signal'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        context_modulator_sensitivity = get_param_value(bt_params.get('context_modulator_sensitivity'), 0.7)
        required_signals = ['pct_change_D', 'deception_index_D', context_modulator_signal_name]
        if not self._validate_required_signals(df, required_signals, "_calculate_bull_trap_context_penalty"):
            return pd.Series(1.0, index=df_index, dtype=np.float32)
        signals_data = self._get_all_required_signals(df, required_signals, "_calculate_bull_trap_context_penalty")
        pct_change_raw = signals_data['pct_change_D'] / 100 # 转换为小数
        deception_index_raw = signals_data['deception_index_D']
        context_modulator_raw = signals_data[context_modulator_signal_name]
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # 1. 检测近期大幅下跌
        # 计算过去 N 天的最低跌幅
        min_pct_change_in_window = pct_change_raw.rolling(window=recent_sharp_drop_window, min_periods=1).min()
        has_recent_sharp_drop = (min_pct_change_in_window <= min_sharp_drop_pct)
        # 2. 检测当前是否存在正向欺骗
        norm_deception_index_bipolar = get_adaptive_mtf_normalized_bipolar_score(deception_index_raw, df_index, tf_weights)
        has_positive_deception = (norm_deception_index_bipolar > 0)
        # 3. 结合情境调制器
        norm_context_modulator = get_adaptive_mtf_normalized_score(context_modulator_raw, df_index, ascending=True, tf_weights=tf_weights)
        # 波动性越高，惩罚越敏感
        dynamic_penalty_sensitivity = 1 + norm_context_modulator * context_modulator_sensitivity
        # 4. 计算惩罚因子
        penalty_factor = pd.Series(1.0, index=df_index, dtype=np.float32)
        bull_trap_condition = has_recent_sharp_drop & has_positive_deception
        if bull_trap_condition.any():
            # 惩罚强度 = 欺骗指数 * 惩罚乘数 * 动态敏感度
            penalty_strength = norm_deception_index_bipolar.loc[bull_trap_condition] * deception_penalty_multiplier * dynamic_penalty_sensitivity.loc[bull_trap_condition]
            # 将惩罚强度映射到 0 到 1 之间，1 - 惩罚强度
            penalty_factor.loc[bull_trap_condition] = (1 - penalty_strength).clip(0.0, 1.0)
        print(f"    -> [筹码层] 计算牛市陷阱情境惩罚，平均惩罚因子: {penalty_factor.mean():.4f}")
        return penalty_factor


