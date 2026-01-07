# 文件: strategies/trend_following/intelligence/process/calculate_main_force_rally_intent.py
import json
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional, Any, Tuple

from strategies.trend_following.utils import (
    get_params_block, get_param_value, get_adaptive_mtf_normalized_score, 
    is_limit_up, get_adaptive_mtf_normalized_bipolar_score, 
    normalize_score, _robust_geometric_mean
)
from strategies.trend_following.intelligence.process.helper import ProcessIntelligenceHelper

class CalculateMainForceRallyIntent:
    """
    【V11.3 · 承接强化与风险情境化版】计算“主力拉升意图”的专属关系分数。
    - 核心强化: 显著增强“派发吸收强度”的积极作用，使其更直接地贡献于看涨意图，并提高其在攻击性中的权重。
    - 核心优化: 进一步情境化“派发风险”，提高“派发吸收强度反向”在派发风险中的权重，以更有效地对冲派发烈度。
    - 核心调整: 进一步调整风险惩罚的非线性函数参数，使其在面对中等风险时更加温和。
    - 核心升级: 引入“派发吸收强度”指标，对冲“派发烈度”带来的负面影响，更全面反映市场承接能力。
    - 核心优化: 引入“派发情境衰减器”，根据当日涨幅动态削弱“派发烈度”对看跌意图的贡献。
    - 核心升级: 严格限制仅使用数据层提供的原始指标，移除所有情报层生成的SCORE_FOUNDATION_AXIOM_*信号。
    - 动态权重机制精细化: 综合市场稳定性、市场情绪（基于原始零售/主力行为）、流动性（基于原始订单簿/资金流）等原始情境因子，自适应调整各维度权重。
    - 非线性融合优化: 基础看涨意图的加权幂平均幂次p根据原始情境因子动态调整。
    - 风险审判深化: 引入更多微观风险信号，并优化非线性风险惩罚函数。
    - 攻击性、控制力、障碍清除维度进一步补充微观结构和行为信号。
    - 情境调节器扩展，融入更多行为心理和市场情绪因子（基于原始数据）。
    - 新增历史记忆与上下文机制：引入长期主力资金累计记忆、筹码集中度稳定性记忆和长期趋势强度上下文，以更全面地评估市场状态。
    - 调整长期主力资金累计记忆和长期趋势强度的周期为21日，以更好地捕捉中短期趋势。
    - 探针优化: 增加详细的调试输出，覆盖所有新增的原始数据、归一化过程、中间计算节点和最终结果。
    """
    def __init__(self, strategy_instance, process_intelligence_helper_instance: ProcessIntelligenceHelper):
        """
        初始化 CalculateMainForceRallyIntent。
        参数:
            strategy_instance: 策略实例，用于访问全局配置和原子状态。
            process_intelligence_helper_instance: ProcessIntelligenceHelper 的实例，用于调用辅助方法。
        """
        self.strategy = strategy_instance
        self.helper = process_intelligence_helper_instance
        # 从 helper 实例获取调试参数和探针日期
        self.debug_params = self.helper.debug_params
        self.probe_dates = self.helper.probe_dates

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        计算“主力拉升意图”的专属关系分数。
        """
        method_name = "_calculate_main_force_rally_intent"
        is_debug_enabled_for_method, probe_ts, debug_output, _temp_debug_values = self._setup_debug_info(df, method_name)

        params = self._get_parameters(config)
        actual_mtf_weights = params["actual_mtf_weights"]
        mtf_slope_accel_weights = params["mtf_slope_accel_weights"]
        historical_context_params = params["historical_context_params"]
        rally_intent_synthesis_params = params["rally_intent_synthesis_params"]

        df_index = df.index
        is_limit_up_day = df.apply(lambda row: is_limit_up(row), axis=1)

        # 1. 校验所需信号
        required_signals = self._get_required_signals_list(mtf_slope_accel_weights, historical_context_params)
        if not self.helper._validate_required_signals(df, required_signals, method_name):
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。"] = ""
                self.helper._print_debug_output(debug_output)
            return pd.Series(0.0, index=df.index, dtype=np.float32)

        # 2. 获取原始信号
        raw_signals = self._get_raw_signals(df, method_name, get_param_value(historical_context_params.get('long_term_trend_slope_period'), 21))
        _temp_debug_values["原始信号值"] = raw_signals
        _temp_debug_values["派生信号值"] = {"is_limit_up_day": is_limit_up_day}

        # 3. 计算MTF融合信号
        mtf_signals = self._calculate_mtf_fused_signals(df, mtf_slope_accel_weights, df_index, method_name)
        _temp_debug_values["MTF融合信号"] = mtf_signals

        # 4. 计算历史上下文
        historical_context = self._calculate_historical_context(df, df_index, raw_signals, historical_context_params, is_debug_enabled_for_method, probe_ts, _temp_debug_values)

        # 5. 归一化原始信号
        normalized_signals = self._normalize_raw_signals(df_index, raw_signals, method_name)
        _temp_debug_values["归一化处理"] = normalized_signals

        # 6. 构建代理信号
        proxy_signals = self._construct_proxy_signals(df_index, mtf_signals, normalized_signals, config)
        _temp_debug_values["代理信号"] = proxy_signals

        # 7. 计算动态权重
        dynamic_weights = self._calculate_dynamic_weights(df_index, normalized_signals, proxy_signals, mtf_signals, _temp_debug_values)

        # 8. 计算攻击性
        aggressiveness_score = self._calculate_aggressiveness_score(df_index, mtf_signals, normalized_signals, dynamic_weights, _temp_debug_values)
        self.strategy.atomic_states["_DEBUG_rally_aggressiveness"] = aggressiveness_score

        # 9. 计算控制力
        control_score = self._calculate_control_score(df_index, mtf_signals, normalized_signals, historical_context, _temp_debug_values)
        self.strategy.atomic_states["_DEBUG_rally_control"] = control_score

        # 10. 计算障碍清除
        obstacle_clearance_score = self._calculate_obstacle_clearance_score(df_index, mtf_signals, normalized_signals, _temp_debug_values)
        self.strategy.atomic_states["_DEBUG_rally_obstacle_clearance"] = obstacle_clearance_score

        # 11. 合成基础看涨意图
        bullish_intent = self._synthesize_bullish_intent(df_index, aggressiveness_score, control_score, obstacle_clearance_score, mtf_signals, dynamic_weights, historical_context, rally_intent_synthesis_params, _temp_debug_values)
        self.strategy.atomic_states["_DEBUG_rally_bullish_intent"] = bullish_intent

        # 12. 计算看跌意图
        bearish_score = self._calculate_bearish_intent(df_index, raw_signals, mtf_signals, normalized_signals, historical_context, _temp_debug_values)
        self.strategy.atomic_states["_DEBUG_rally_bearish_score"] = bearish_score

        # 13. 风险审判模块
        total_risk_penalty = self._adjudicate_risk(df_index, raw_signals, mtf_signals, normalized_signals, dynamic_weights, rally_intent_synthesis_params, _temp_debug_values)
        self.strategy.atomic_states["_DEBUG_rally_total_risk_penalty"] = total_risk_penalty

        # 14. 最终意图合成
        penalized_bullish_part = bullish_intent * (1 - total_risk_penalty)
        final_rally_intent = (penalized_bullish_part + bearish_score).clip(-1, 1)
        final_rally_intent = final_rally_intent.mask(is_limit_up_day & (total_risk_penalty > 0.5), final_rally_intent * (1 - total_risk_penalty))
        final_rally_intent = final_rally_intent.mask(is_limit_up_day & (final_rally_intent < 0), 0.0)
        _temp_debug_values["最终意图合成"] = {
            "penalized_bullish_part": penalized_bullish_part,
            "final_rally_intent_before_mod": final_rally_intent
        }

        # 15. 应用情境调节器
        final_rally_intent = self._apply_contextual_modulators(df_index, final_rally_intent, proxy_signals, mtf_signals, _temp_debug_values)

        # 16. 输出调试信息
        self._output_debug_info(is_debug_enabled_for_method, probe_ts, debug_output, _temp_debug_values, final_rally_intent, method_name)

        return final_rally_intent.astype(np.float32)

    def _setup_debug_info(self, df: pd.DataFrame, method_name: str) -> Tuple[bool, Optional[pd.Timestamp], Dict, Dict]:
        """
        设置调试信息，判断是否启用调试和探针日期。
        """
        is_debug_enabled_for_method = get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug_enabled_for_method and self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            for date in reversed(df.index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        if probe_ts is None:
            is_debug_enabled_for_method = False
        debug_output = {}
        _temp_debug_values = {} # 临时存储所有中间计算结果的原始值 (无条件收集)
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算主力拉升意图..."] = ""
        return is_debug_enabled_for_method, probe_ts, debug_output, _temp_debug_values

    def _get_parameters(self, config: Dict) -> Dict:
        """
        获取所有必要的配置参数。
        """
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        mtf_slope_accel_weights = config.get('mtf_slope_accel_weights', {"slope_periods": {"5": 0.6, "13": 0.4}, "accel_periods": {"5": 0.7, "13": 0.3}})
        historical_context_params = config.get('historical_context_params', {})
        rally_intent_synthesis_params = config.get('rally_intent_synthesis_params', {})
        return {
            "actual_mtf_weights": actual_mtf_weights,
            "mtf_slope_accel_weights": mtf_slope_accel_weights,
            "historical_context_params": historical_context_params,
            "rally_intent_synthesis_params": rally_intent_synthesis_params
        }

    def _get_raw_signals(self, df: pd.DataFrame, method_name: str, long_term_trend_slope_period: int) -> Dict[str, pd.Series]:
        """
        获取所有原始数据信号。
        """
        raw_signals = {
            'pct_change': self.helper._get_safe_series(df, 'pct_change_D', 0.0, method_name=method_name),
            'main_force_net_flow': self.helper._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name=method_name),
            'main_force_slippage': self.helper._get_safe_series(df, 'main_force_slippage_index_D', 0.0, method_name=method_name),
            'upward_impulse_purity': self.helper._get_safe_series(df, 'upward_impulse_purity_D', 0.0, method_name=method_name),
            'volume_ratio': self.helper._get_safe_series(df, 'volume_ratio_D', 1.0, method_name=method_name),
            'control_solidity': self.helper._get_safe_series(df, 'control_solidity_index_D', 0.0, method_name=method_name),
            'cost_advantage': self.helper._get_safe_series(df, 'main_force_cost_advantage_D', 0.0, method_name=method_name),
            'concentration_slope': self.helper._get_safe_series(df, f'SLOPE_5_winner_concentration_90pct_D', 0.0, method_name=method_name),
            'peak_solidity': self.helper._get_safe_series(df, 'dominant_peak_solidity_D', 0.0, method_name=method_name),
            'active_buying_support': self.helper._get_safe_series(df, 'active_buying_support_D', 0.0, method_name=method_name),
            'pressure_rejection': self.helper._get_safe_series(df, 'pressure_rejection_strength_D', 0.0, method_name=method_name),
            'profit_realization_quality': self.helper._get_safe_series(df, 'profit_realization_quality_D', 0.5, method_name=method_name),
            'distribution_at_peak_intensity': self.helper._get_safe_series(df, 'distribution_at_peak_intensity_D', 0.0, method_name=method_name),
            'upper_shadow_selling_pressure': self.helper._get_safe_series(df, 'upper_shadow_selling_pressure_D', 0.0, method_name=method_name),
            'flow_credibility': self.helper._get_safe_series(df, 'flow_credibility_index_D', 0.0, method_name=method_name),
            'chip_health': self.helper._get_safe_series(df, 'chip_health_score_D', 0.0, method_name=method_name),
            'retail_fomo': self.helper._get_safe_series(df, 'retail_fomo_premium_index_D', 0.0, method_name=method_name),
            'close_price': self.helper._get_safe_series(df, 'close_D', 0.0, method_name=method_name),
            'prev_day_pct_change': self.helper._get_safe_series(df, 'pct_change_D', 0.0, method_name=method_name).shift(1).fillna(0),
            'slope_21_close': self.helper._get_safe_series(df, 'SLOPE_21_close_D', 0.0, method_name=method_name),
            'accel_21_close': self.helper._get_safe_series(df, 'ACCEL_21_close_D', 0.0, method_name=method_name),
            'slope_34_close': self.helper._get_safe_series(df, 'SLOPE_34_close_D', 0.0, method_name=method_name),
            'accel_34_close': self.helper._get_safe_series(df, 'ACCEL_34_close_D', 0.0, method_name=method_name),
            'buy_sweep_intensity': self.helper._get_safe_series(df, 'buy_sweep_intensity_D', 0.0, method_name=method_name),
            'main_force_buy_ofi': self.helper._get_safe_series(df, 'main_force_buy_ofi_D', 0.0, method_name=method_name),
            'main_force_t0_buy_efficiency': self.helper._get_safe_series(df, 'main_force_t0_buy_efficiency_D', 0.0, method_name=method_name),
            'order_book_imbalance': self.helper._get_safe_series(df, 'order_book_imbalance_D', 0.0, method_name=method_name),
            'micro_price_impact_asymmetry': self.helper._get_safe_series(df, 'micro_price_impact_asymmetry_D', 0.0, method_name=method_name),
            'constructive_turnover': self.helper._get_safe_series(df, 'constructive_turnover_ratio_D', 0.0, method_name=method_name),
            'upward_impulse_strength': self.helper._get_safe_series(df, 'upward_impulse_strength_D', 0.0, method_name=method_name),
            'vwap_buy_control_strength': self.helper._get_safe_series(df, 'vwap_buy_control_strength_D', 0.0, method_name=method_name),
            'mf_cost_zone_buy_intent': self.helper._get_safe_series(df, 'mf_cost_zone_buy_intent_D', 0.0, method_name=method_name),
            'chip_fault_blockage_ratio': self.helper._get_safe_series(df, 'chip_fault_blockage_ratio_D', 0.0, method_name=method_name),
            'vacuum_traversal_efficiency': self.helper._get_safe_series(df, 'vacuum_traversal_efficiency_D', 0.0, method_name=method_name),
            'vacuum_zone_magnitude': self.helper._get_safe_series(df, 'vacuum_zone_magnitude_D', 0.0, method_name=method_name),
            'dip_buy_absorption_strength': self.helper._get_safe_series(df, 'dip_buy_absorption_strength_D', 0.0, method_name=method_name),
            'rally_buy_support_weakness': self.helper._get_safe_series(df, 'rally_buy_support_weakness_D', 0.0, method_name=method_name),
            'covert_distribution': self.helper._get_safe_series(df, 'covert_distribution_signal_D', 0.0, method_name=method_name),
            'deception_lure_short': self.helper._get_safe_series(df, 'deception_lure_short_intensity_D', 0.0, method_name=method_name),
            'rally_distribution_pressure': self.helper._get_safe_series(df, 'rally_distribution_pressure_D', 0.0, method_name=method_name),
            'exhaustion_risk': self.helper._get_safe_series(df, 'exhaustion_risk_index_D', 0.0, method_name=method_name),
            'asymmetric_friction': self.helper._get_safe_series(df, 'asymmetric_friction_index_D', 0.0, method_name=method_name),
            'volatility_expansion': self.helper._get_safe_series(df, 'volatility_expansion_ratio_D', 0.0, method_name=method_name),
            'market_sentiment': self.helper._get_safe_series(df, 'market_sentiment_score_D', 0.0, method_name=method_name),
            'structural_tension': self.helper._get_safe_series(df, 'structural_tension_index_D', 0.0, method_name=method_name),
            'trend_vitality': self.helper._get_safe_series(df, 'trend_vitality_index_D', 0.0, method_name=method_name),
            'liquidity_authenticity': self.helper._get_safe_series(df, 'liquidity_authenticity_score_D', 0.0, method_name=method_name),
            'order_book_clearing_rate': self.helper._get_safe_series(df, 'order_book_clearing_rate_D', 0.0, method_name=method_name),
            'sell_sweep_intensity': self.helper._get_safe_series(df, 'sell_sweep_intensity_D', 0.0, method_name=method_name),
            'main_force_flow_gini': self.helper._get_safe_series(df, 'main_force_flow_gini_D', 0.0, method_name=method_name),
            'microstructure_efficiency': self.helper._get_safe_series(df, 'microstructure_efficiency_index_D', 0.0, method_name=method_name),
            'imbalance_effectiveness': self.helper._get_safe_series(df, 'imbalance_effectiveness_D', 0.0, method_name=method_name),
            'auction_showdown': self.helper._get_safe_series(df, 'auction_showdown_score_D', 0.0, method_name=method_name),
            'closing_conviction': self.helper._get_safe_series(df, 'closing_conviction_score_D', 0.0, method_name=method_name),
            'intraday_energy_density': self.helper._get_safe_series(df, 'intraday_energy_density_D', 0.0, method_name=method_name),
            'intraday_thrust_purity': self.helper._get_safe_series(df, 'intraday_thrust_purity_D', 0.0, method_name=method_name),
            'price_thrust_divergence': self.helper._get_safe_series(df, 'price_thrust_divergence_D', 0.0, method_name=method_name),
            'trend_efficiency_ratio': self.helper._get_safe_series(df, 'trend_efficiency_ratio_D', 0.0, method_name=method_name),
            'loser_concentration_90pct': self.helper._get_safe_series(df, 'loser_concentration_90pct_D', 0.0, method_name=method_name),
            'winner_loser_momentum': self.helper._get_safe_series(df, 'winner_loser_momentum_D', 0.0, method_name=method_name),
            'cost_structure_skewness': self.helper._get_safe_series(df, 'cost_structure_skewness_D', 0.0, method_name=method_name),
            'cost_gini_coefficient': self.helper._get_safe_series(df, 'cost_gini_coefficient_D', 0.0, method_name=method_name),
            'mf_vpoc_premium': self.helper._get_safe_series(df, 'mf_vpoc_premium_D', 0.0, method_name=method_name),
            'character_score': self.helper._get_safe_series(df, 'character_score_D', 0.0, method_name=method_name),
            'signal_conviction_score': self.helper._get_safe_series(df, 'signal_conviction_score_D', 0.0, method_name=method_name),
            'touch_conviction_score': self.helper._get_safe_series(df, 'touch_conviction_score_D', 0.0, method_name=method_name),
            'gathering_by_chasing': self.helper._get_safe_series(df, 'gathering_by_chasing_D', 0.0, method_name=method_name),
            'gathering_by_support': self.helper._get_safe_series(df, 'gathering_by_support_D', 0.0, method_name=method_name),
            'volatility_instability': self.helper._get_safe_series(df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', 0.0, method_name=method_name),
            'adx': self.helper._get_safe_series(df, 'ADX_14_D', 0.0, method_name=method_name),
            'buy_flow_efficiency': self.helper._get_safe_series(df, 'buy_flow_efficiency_index_D', 0.0, method_name=method_name),
            'sell_flow_efficiency': self.helper._get_safe_series(df, 'sell_flow_efficiency_index_D', 0.0, method_name=method_name),
            'auction_closing_position': self.helper._get_safe_series(df, 'auction_closing_position_D', 0.0, method_name=method_name),
            'auction_impact_score': self.helper._get_safe_series(df, 'auction_impact_score_D', 0.0, method_name=method_name),
            'auction_intent_signal': self.helper._get_safe_series(df, 'auction_intent_signal_D', 0.0, method_name=method_name),
            'order_book_liquidity_supply': self.helper._get_safe_series(df, 'order_book_liquidity_supply_D', 0.0, method_name=method_name),
            'liquidity_slope': self.helper._get_safe_series(df, 'liquidity_slope_D', 0.0, method_name=method_name),
            'peak_mass_transfer_rate': self.helper._get_safe_series(df, 'peak_mass_transfer_rate_D', 0.0, method_name=method_name),
            'mf_cost_zone_defense_intent': self.helper._get_safe_series(df, 'mf_cost_zone_defense_intent_D', 0.0, method_name=method_name),
            'bid_side_liquidity': self.helper._get_safe_series(df, 'bid_side_liquidity_D', 0.0, method_name=method_name),
            'ask_side_liquidity': self.helper._get_safe_series(df, 'ask_side_liquidity_D', 0.0, method_name=method_name),
            'retail_panic_surrender': self.helper._get_safe_series(df, 'retail_panic_surrender_index_D', 0.0, method_name=method_name),
            'main_force_activity_ratio': self.helper._get_safe_series(df, 'main_force_activity_ratio_D', 0.0, method_name=method_name),
            'main_force_conviction_index': self.helper._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name=method_name),
            'main_force_execution_alpha': self.helper._get_safe_series(df, 'main_force_execution_alpha_D', 0.0, method_name=method_name),
            'main_force_flow_directionality': self.helper._get_safe_series(df, 'main_force_flow_directionality_D', 0.0, method_name=method_name),
            'main_force_on_peak_buy_flow': self.helper._get_safe_series(df, 'main_force_on_peak_buy_flow_D', 0.0, method_name=method_name),
            'main_force_on_peak_sell_flow': self.helper._get_safe_series(df, 'main_force_on_peak_sell_flow_D', 0.0, method_name=method_name),
            'main_force_t0_efficiency': self.helper._get_safe_series(df, 'main_force_t0_efficiency_D', 0.0, method_name=method_name),
            'main_force_t0_sell_efficiency': self.helper._get_safe_series(df, 'main_force_t0_sell_efficiency_D', 0.0, method_name=method_name),
            'main_force_vwap_down_guidance': self.helper._get_safe_series(df, 'main_force_vwap_down_guidance_D', 0.0, method_name=method_name),
            'main_force_vwap_up_guidance': self.helper._get_safe_series(df, 'main_force_vwap_up_guidance_D', 0.0, method_name=method_name),
            'market_impact_cost': self.helper._get_safe_series(df, 'market_impact_cost_D', 0.0, method_name=method_name),
            'opening_buy_strength': self.helper._get_safe_series(df, 'opening_buy_strength_D', 0.0, method_name=method_name),
            'opening_sell_strength': self.helper._get_safe_series(df, 'opening_sell_strength_D', 0.0, method_name=method_name),
            'closing_strength_index': self.helper._get_safe_series(df, 'closing_strength_index_D', 0.0, method_name=method_name),
            'total_buy_amount_calibrated': self.helper._get_safe_series(df, 'total_buy_amount_calibrated_D', 0.0, method_name=method_name),
            'total_sell_amount_calibrated': self.helper._get_safe_series(df, 'total_sell_amount_calibrated_D', 0.0, method_name=method_name),
            'wash_trade_intensity': self.helper._get_safe_series(df, 'wash_trade_intensity_D', 0.0, method_name=method_name),
            'winner_profit_margin_avg': self.helper._get_safe_series(df, 'winner_profit_margin_avg_D', 0.0, method_name=method_name),
            'loser_loss_margin_avg': self.helper._get_safe_series(df, 'loser_loss_margin_avg_D', 0.0, method_name=method_name),
            'total_winner_rate': self.helper._get_safe_series(df, 'total_winner_rate_D', 0.0, method_name=method_name),
            'total_loser_rate': self.helper._get_safe_series(df, 'total_loser_rate_D', 0.0, method_name=method_name),
            'impulse_quality_ratio': self.helper._get_safe_series(df, 'impulse_quality_ratio_D', 0.0, method_name=method_name),
            'thrust_efficiency_score': self.helper._get_safe_series(df, 'thrust_efficiency_score_D', 0.0, method_name=method_name),
            'platform_conviction_score': self.helper._get_safe_series(df, 'platform_conviction_score_D', 0.0, method_name=method_name),
            'platform_high': self.helper._get_safe_series(df, 'platform_high_D', 0.0, method_name=method_name),
            'platform_low': self.helper._get_safe_series(df, 'platform_low_D', 0.0, method_name=method_name),
            'breakout_quality_score': self.helper._get_safe_series(df, 'breakout_quality_score_D', 0.0, method_name=method_name),
            'breakout_readiness_score': self.helper._get_safe_series(df, 'breakout_readiness_score_D', 0.0, method_name=method_name),
            'breakthrough_conviction_score': self.helper._get_safe_series(df, 'breakthrough_conviction_score_D', 0.0, method_name=method_name),
            'defense_solidity_score': self.helper._get_safe_series(df, 'defense_solidity_score_D', 0.0, method_name=method_name),
            'support_validation_strength': self.helper._get_safe_series(df, 'support_validation_strength_D', 0.0, method_name=method_name),
            'covert_accumulation_signal': self.helper._get_safe_series(df, 'covert_accumulation_signal_D', 0.0, method_name=method_name),
            'suppressive_accumulation_intensity': self.helper._get_safe_series(df, 'suppressive_accumulation_intensity_D', 0.0, method_name=method_name),
            'deception_index': self.helper._get_safe_series(df, 'deception_index_D', 0.0, method_name=method_name),
            'deception_lure_long_intensity': self.helper._get_safe_series(df, 'deception_lure_long_intensity_D', 0.0, method_name=method_name),
            'equilibrium_compression_index': self.helper._get_safe_series(df, 'equilibrium_compression_index_D', 0.0, method_name=method_name),
            'final_charge_intensity': self.helper._get_safe_series(df, 'final_charge_intensity_D', 0.0, method_name=method_name),
            'floating_chip_cleansing_efficiency': self.helper._get_safe_series(df, 'floating_chip_cleansing_efficiency_D', 0.0, method_name=method_name),
            'hidden_accumulation_intensity': self.helper._get_safe_series(df, 'hidden_accumulation_intensity_D', 0.0, method_name=method_name),
            'internal_accumulation_intensity': self.helper._get_safe_series(df, 'internal_accumulation_intensity_D', 0.0, method_name=method_name),
            'intraday_posture_score': self.helper._get_safe_series(df, 'intraday_posture_score_D', 0.0, method_name=method_name),
            'opening_gap_defense_strength': self.helper._get_safe_series(df, 'opening_gap_defense_strength_D', 0.0, method_name=method_name),
            'panic_buy_absorption_contribution': self.helper._get_safe_series(df, 'panic_buy_absorption_contribution_D', 0.0, method_name=method_name),
            'panic_sell_volume_contribution': self.helper._get_safe_series(df, 'panic_sell_volume_contribution_D', 0.0, method_name=method_name),
            'panic_selling_cascade': self.helper._get_safe_series(df, 'panic_selling_cascade_D', 0.0, method_name=method_name),
            'peak_control_transfer': self.helper._get_safe_series(df, 'peak_control_transfer_D', 0.0, method_name=method_name),
            'peak_separation_ratio': self.helper._get_safe_series(df, 'peak_separation_ratio_D', 0.0, method_name=method_name),
            'price_reversion_velocity': self.helper._get_safe_series(df, 'price_reversion_velocity_D', 0.0, method_name=method_name),
            'pullback_depth_ratio': self.helper._get_safe_series(df, 'pullback_depth_ratio_D', 0.0, method_name=method_name),
            'quality_score': self.helper._get_safe_series(df, 'quality_score_D', 0.0, method_name=method_name),
            'reversal_conviction_rate': self.helper._get_safe_series(df, 'reversal_conviction_rate_D', 0.0, method_name=method_name),
            'reversal_power_index': self.helper._get_safe_series(df, 'reversal_power_index_D', 0.0, method_name=method_name),
            'reversal_recovery_rate': self.helper._get_safe_series(df, 'reversal_recovery_rate_D', 0.0, method_name=method_name),
            'risk_reward_profile': self.helper._get_safe_series(df, 'risk_reward_profile_D', 0.0, method_name=method_name),
            'shock_conviction_score': self.helper._get_safe_series(df, 'shock_conviction_score_D', 0.0, method_name=method_name),
            'strategic_phase_score': self.helper._get_safe_series(df, 'strategic_phase_score_D', 0.0, method_name=method_name),
            'structural_entropy_change': self.helper._get_safe_series(df, 'structural_entropy_change_D', 0.0, method_name=method_name),
            'structural_leverage': self.helper._get_safe_series(df, 'structural_leverage_D', 0.0, method_name=method_name),
            'structural_node_count': self.helper._get_safe_series(df, 'structural_node_count_D', 0.0, method_name=method_name),
            'structural_potential_score': self.helper._get_safe_series(df, 'structural_potential_score_D', 0.0, method_name=method_name),
            'support_validation_score': self.helper._get_safe_series(df, 'support_validation_score_D', 0.0, method_name=method_name),
            'supportive_distribution_intensity': self.helper._get_safe_series(df, 'supportive_distribution_intensity_D', 0.0, method_name=method_name),
            'trend_acceleration_score': self.helper._get_safe_series(df, 'trend_acceleration_score_D', 0.0, method_name=method_name),
            'trend_alignment_index': self.helper._get_safe_series(df, 'trend_alignment_index_D', 0.0, method_name=method_name),
            'trend_asymmetry_index': self.helper._get_safe_series(df, 'trend_asymmetry_index_D', 0.0, method_name=method_name),
            'trend_conviction_score': self.helper._get_safe_series(df, 'trend_conviction_score_D', 0.0, method_name=method_name),
            'value_area_migration': self.helper._get_safe_series(df, 'value_area_migration_D', 0.0, method_name=method_name),
            'value_area_overlap_pct': self.helper._get_safe_series(df, 'value_area_overlap_pct_D', 0.0, method_name=method_name),
            'volatility_asymmetry_index': self.helper._get_safe_series(df, 'volatility_asymmetry_index_D', 0.0, method_name=method_name),
            'volume_burstiness_index': self.helper._get_safe_series(df, 'volume_burstiness_index_D', 0.0, method_name=method_name),
            'volume_structure_skew': self.helper._get_safe_series(df, 'volume_structure_skew_D', 0.0, method_name=method_name),
            'vpin_score': self.helper._get_safe_series(df, 'vpin_score_D', 0.0, method_name=method_name),
            'vwap_control_strength': self.helper._get_safe_series(df, 'vwap_control_strength_D', 0.0, method_name=method_name),
            'vwap_cross_down_intensity': self.helper._get_safe_series(df, 'vwap_cross_down_intensity_D', 0.0, method_name=method_name),
            'vwap_cross_up_intensity': self.helper._get_safe_series(df, 'vwap_cross_up_intensity_D', 0.0, method_name=method_name),
            'vwap_crossing_intensity': self.helper._get_safe_series(df, 'vwap_crossing_intensity_D', 0.0, method_name=method_name),
            'vwap_mean_reversion_corr': self.helper._get_safe_series(df, 'vwap_mean_reversion_corr_D', 0.0, method_name=method_name),
            'vwap_sell_control_strength': self.helper._get_safe_series(df, 'vwap_sell_control_strength_D', 0.0, method_name=method_name),
            'winner_stability_index': self.helper._get_safe_series(df, 'winner_stability_index_D', 0.0, method_name=method_name),
            'winner_concentration_90pct': self.helper._get_safe_series(df, 'winner_concentration_90pct_D', 0.0, method_name=method_name),
            'long_term_trend_slope': self.helper._get_safe_series(df, f'SLOPE_{long_term_trend_slope_period}_close_D', 0.0, method_name=method_name),
            'absorption_of_distribution_intensity': self.helper._get_safe_series(df, 'absorption_of_distribution_intensity_D', 0.0, method_name=method_name)
        }
        return raw_signals

    def _calculate_mtf_fused_signals(self, df: pd.DataFrame, mtf_slope_accel_weights: Dict, df_index: pd.Index, method_name: str) -> Dict[str, pd.Series]:
        """
        计算所有MTF融合信号。
        """
        mtf_signals = {
            'mtf_price_trend': self.helper._get_mtf_slope_accel_score(df, 'close_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True),
            'mtf_mf_net_flow': self.helper._get_mtf_slope_accel_score(df, 'main_force_net_flow_calibrated_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True),
            'mtf_upper_shadow_pressure': self.helper._get_mtf_slope_accel_score(df, 'upper_shadow_selling_pressure_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_retail_fomo': self.helper._get_mtf_slope_accel_score(df, 'retail_fomo_premium_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_buy_sweep_intensity': self.helper._get_mtf_slope_accel_score(df, 'buy_sweep_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_main_force_buy_ofi': self.helper._get_mtf_slope_accel_score(df, 'main_force_buy_ofi_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_main_force_t0_buy_efficiency': self.helper._get_mtf_slope_accel_score(df, 'main_force_t0_buy_efficiency_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_upward_impulse_strength': self.helper._get_mtf_slope_accel_score(df, 'upward_impulse_strength_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_vwap_buy_control_strength': self.helper._get_mtf_slope_accel_score(df, 'vwap_buy_control_strength_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_mf_cost_zone_buy_intent': self.helper._get_mtf_slope_accel_score(df, 'mf_cost_zone_buy_intent_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_chip_fault_blockage_ratio': self.helper._get_mtf_slope_accel_score(df, 'chip_fault_blockage_ratio_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_vacuum_traversal_efficiency': self.helper._get_mtf_slope_accel_score(df, 'vacuum_traversal_efficiency_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_dip_buy_absorption_strength': self.helper._get_mtf_slope_accel_score(df, 'dip_buy_absorption_strength_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_rally_buy_support_weakness': self.helper._get_mtf_slope_accel_score(df, 'rally_buy_support_weakness_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_covert_distribution': self.helper._get_mtf_slope_accel_score(df, 'covert_distribution_signal_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_deception_lure_short': self.helper._get_mtf_slope_accel_score(df, 'deception_lure_short_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_rally_distribution_pressure': self.helper._get_mtf_slope_accel_score(df, 'rally_distribution_pressure_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_exhaustion_risk': self.helper._get_mtf_slope_accel_score(df, 'exhaustion_risk_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_asymmetric_friction': self.helper._get_mtf_slope_accel_score(df, 'asymmetric_friction_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_volatility_expansion': self.helper._get_mtf_slope_accel_score(df, 'volatility_expansion_ratio_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_market_sentiment': self.helper._get_mtf_slope_accel_score(df, 'market_sentiment_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True),
            'mtf_structural_tension': self.helper._get_mtf_slope_accel_score(df, 'structural_tension_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_trend_vitality': self.helper._get_mtf_slope_accel_score(df, 'trend_vitality_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_liquidity_authenticity': self.helper._get_mtf_slope_accel_score(df, 'liquidity_authenticity_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_order_book_clearing_rate': self.helper._get_mtf_slope_accel_score(df, 'order_book_clearing_rate_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_sell_sweep_intensity': self.helper._get_mtf_slope_accel_score(df, 'sell_sweep_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_main_force_flow_gini': self.helper._get_mtf_slope_accel_score(df, 'main_force_flow_gini_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_microstructure_efficiency': self.helper._get_mtf_slope_accel_score(df, 'microstructure_efficiency_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_imbalance_effectiveness': self.helper._get_mtf_slope_accel_score(df, 'imbalance_effectiveness_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_auction_showdown': self.helper._get_mtf_slope_accel_score(df, 'auction_showdown_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_closing_conviction': self.helper._get_mtf_slope_accel_score(df, 'closing_conviction_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_intraday_energy_density': self.helper._get_mtf_slope_accel_score(df, 'intraday_energy_density_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_intraday_thrust_purity': self.helper._get_mtf_slope_accel_score(df, 'intraday_thrust_purity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_price_thrust_divergence': self.helper._get_mtf_slope_accel_score(df, 'price_thrust_divergence_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True),
            'mtf_trend_efficiency_ratio': self.helper._get_mtf_slope_accel_score(df, 'trend_efficiency_ratio_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_loser_concentration_90pct': self.helper._get_mtf_slope_accel_score(df, 'loser_concentration_90pct_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_winner_loser_momentum': self.helper._get_mtf_slope_accel_score(df, 'winner_loser_momentum_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True),
            'mtf_cost_structure_skewness': self.helper._get_mtf_slope_accel_score(df, 'cost_structure_skewness_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True),
            'mtf_cost_gini_coefficient': self.helper._get_mtf_slope_accel_score(df, 'cost_gini_coefficient_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_mf_vpoc_premium': self.helper._get_mtf_slope_accel_score(df, 'mf_vpoc_premium_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True),
            'mtf_character_score': self.helper._get_mtf_slope_accel_score(df, 'character_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_signal_conviction_score': self.helper._get_mtf_slope_accel_score(df, 'signal_conviction_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_touch_conviction_score': self.helper._get_mtf_slope_accel_score(df, 'touch_conviction_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_gathering_by_chasing': self.helper._get_mtf_slope_accel_score(df, 'gathering_by_chasing_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_gathering_by_support': self.helper._get_mtf_slope_accel_score(df, 'gathering_by_support_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_buy_flow_efficiency': self.helper._get_mtf_slope_accel_score(df, 'buy_flow_efficiency_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_sell_flow_efficiency': self.helper._get_mtf_slope_accel_score(df, 'sell_flow_efficiency_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_auction_closing_position': self.helper._get_mtf_slope_accel_score(df, 'auction_closing_position_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True),
            'mtf_auction_impact_score': self.helper._get_mtf_slope_accel_score(df, 'auction_impact_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_auction_intent_signal': self.helper._get_mtf_slope_accel_score(df, 'auction_intent_signal_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_order_book_liquidity_supply': self.helper._get_mtf_slope_accel_score(df, 'order_book_liquidity_supply_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_liquidity_slope': self.helper._get_mtf_slope_accel_score(df, 'liquidity_slope_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True),
            'mtf_peak_mass_transfer_rate': self.helper._get_mtf_slope_accel_score(df, 'peak_mass_transfer_rate_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_mf_cost_zone_defense_intent': self.helper._get_mtf_slope_accel_score(df, 'mf_cost_zone_defense_intent_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_bid_side_liquidity': self.helper._get_mtf_slope_accel_score(df, 'bid_side_liquidity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_ask_side_liquidity': self.helper._get_mtf_slope_accel_score(df, 'ask_side_liquidity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_retail_panic_surrender': self.helper._get_mtf_slope_accel_score(df, 'retail_panic_surrender_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_main_force_activity_ratio': self.helper._get_mtf_slope_accel_score(df, 'main_force_activity_ratio_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_main_force_conviction_index': self.helper._get_mtf_slope_accel_score(df, 'main_force_conviction_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_main_force_execution_alpha': self.helper._get_mtf_slope_accel_score(df, 'main_force_execution_alpha_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True),
            'mtf_main_force_flow_directionality': self.helper._get_mtf_slope_accel_score(df, 'main_force_flow_directionality_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True),
            'mtf_main_force_on_peak_buy_flow': self.helper._get_mtf_slope_accel_score(df, 'main_force_on_peak_buy_flow_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_main_force_on_peak_sell_flow': self.helper._get_mtf_slope_accel_score(df, 'main_force_on_peak_sell_flow_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_main_force_t0_efficiency': self.helper._get_mtf_slope_accel_score(df, 'main_force_t0_efficiency_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_main_force_t0_sell_efficiency': self.helper._get_mtf_slope_accel_score(df, 'main_force_t0_sell_efficiency_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_main_force_vwap_down_guidance': self.helper._get_mtf_slope_accel_score(df, 'main_force_vwap_down_guidance_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True),
            'mtf_main_force_vwap_up_guidance': self.helper._get_mtf_slope_accel_score(df, 'main_force_vwap_up_guidance_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True),
            'mtf_market_impact_cost': self.helper._get_mtf_slope_accel_score(df, 'market_impact_cost_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_opening_buy_strength': self.helper._get_mtf_slope_accel_score(df, 'opening_buy_strength_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_opening_sell_strength': self.helper._get_mtf_slope_accel_score(df, 'opening_sell_strength_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_closing_strength_index': self.helper._get_mtf_slope_accel_score(df, 'closing_strength_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_total_buy_amount_calibrated': self.helper._get_mtf_slope_accel_score(df, 'total_buy_amount_calibrated_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_total_sell_amount_calibrated': self.helper._get_mtf_slope_accel_score(df, 'total_sell_amount_calibrated_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_wash_trade_intensity': self.helper._get_mtf_slope_accel_score(df, 'wash_trade_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_winner_profit_margin_avg': self.helper._get_mtf_slope_accel_score(df, 'winner_profit_margin_avg_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True),
            'mtf_loser_loss_margin_avg': self.helper._get_mtf_slope_accel_score(df, 'loser_loss_margin_avg_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True),
            'mtf_total_winner_rate': self.helper._get_mtf_slope_accel_score(df, 'total_winner_rate_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_total_loser_rate': self.helper._get_mtf_slope_accel_score(df, 'total_loser_rate_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_impulse_quality_ratio': self.helper._get_mtf_slope_accel_score(df, 'impulse_quality_ratio_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_thrust_efficiency_score': self.helper._get_mtf_slope_accel_score(df, 'thrust_efficiency_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_platform_conviction_score': self.helper._get_mtf_slope_accel_score(df, 'platform_conviction_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_platform_high': self.helper._get_mtf_slope_accel_score(df, 'platform_high_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_platform_low': self.helper._get_mtf_slope_accel_score(df, 'platform_low_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_breakout_quality_score': self.helper._get_mtf_slope_accel_score(df, 'breakout_quality_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_breakout_readiness_score': self.helper._get_mtf_slope_accel_score(df, 'breakout_readiness_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_breakthrough_conviction_score': self.helper._get_mtf_slope_accel_score(df, 'breakthrough_conviction_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_defense_solidity_score': self.helper._get_mtf_slope_accel_score(df, 'defense_solidity_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_support_validation_strength': self.helper._get_mtf_slope_accel_score(df, 'support_validation_strength_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_covert_accumulation_signal': self.helper._get_mtf_slope_accel_score(df, 'covert_accumulation_signal_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_suppressive_accumulation_intensity': self.helper._get_mtf_slope_accel_score(df, 'suppressive_accumulation_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_deception_index': self.helper._get_mtf_slope_accel_score(df, 'deception_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_deception_lure_long_intensity': self.helper._get_mtf_slope_accel_score(df, 'deception_lure_long_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_equilibrium_compression_index': self.helper._get_mtf_slope_accel_score(df, 'equilibrium_compression_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_final_charge_intensity': self.helper._get_mtf_slope_accel_score(df, 'final_charge_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_floating_chip_cleansing_efficiency': self.helper._get_mtf_slope_accel_score(df, 'floating_chip_cleansing_efficiency_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_hidden_accumulation_intensity': self.helper._get_mtf_slope_accel_score(df, 'hidden_accumulation_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_internal_accumulation_intensity': self.helper._get_mtf_slope_accel_score(df, 'internal_accumulation_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_intraday_posture_score': self.helper._get_mtf_slope_accel_score(df, 'intraday_posture_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_opening_gap_defense_strength': self.helper._get_mtf_slope_accel_score(df, 'opening_gap_defense_strength_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_panic_buy_absorption_contribution': self.helper._get_mtf_slope_accel_score(df, 'panic_buy_absorption_contribution_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_panic_sell_volume_contribution': self.helper._get_mtf_slope_accel_score(df, 'panic_sell_volume_contribution_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_panic_selling_cascade': self.helper._get_mtf_slope_accel_score(df, 'panic_selling_cascade_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_peak_control_transfer': self.helper._get_mtf_slope_accel_score(df, 'peak_control_transfer_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_peak_separation_ratio': self.helper._get_mtf_slope_accel_score(df, 'peak_separation_ratio_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_price_reversion_velocity': self.helper._get_mtf_slope_accel_score(df, 'price_reversion_velocity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True),
            'mtf_pullback_depth_ratio': self.helper._get_mtf_slope_accel_score(df, 'pullback_depth_ratio_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_quality_score': self.helper._get_mtf_slope_accel_score(df, 'quality_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_reversal_conviction_rate': self.helper._get_mtf_slope_accel_score(df, 'reversal_conviction_rate_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_reversal_power_index': self.helper._get_mtf_slope_accel_score(df, 'reversal_power_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_reversal_recovery_rate': self.helper._get_mtf_slope_accel_score(df, 'reversal_recovery_rate_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_risk_reward_profile': self.helper._get_mtf_slope_accel_score(df, 'risk_reward_profile_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True),
            'mtf_shock_conviction_score': self.helper._get_mtf_slope_accel_score(df, 'shock_conviction_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_strategic_phase_score': self.helper._get_mtf_slope_accel_score(df, 'strategic_phase_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_structural_entropy_change': self.helper._get_mtf_slope_accel_score(df, 'structural_entropy_change_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True),
            'mtf_structural_leverage': self.helper._get_mtf_slope_accel_score(df, 'structural_leverage_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_structural_node_count': self.helper._get_mtf_slope_accel_score(df, 'structural_node_count_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_structural_potential_score': self.helper._get_mtf_slope_accel_score(df, 'structural_potential_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_support_validation_score': self.helper._get_mtf_slope_accel_score(df, 'support_validation_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_supportive_distribution_intensity': self.helper._get_mtf_slope_accel_score(df, 'supportive_distribution_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_trend_acceleration_score': self.helper._get_mtf_slope_accel_score(df, 'trend_acceleration_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_trend_alignment_index': self.helper._get_mtf_slope_accel_score(df, 'trend_alignment_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_trend_asymmetry_index': self.helper._get_mtf_slope_accel_score(df, 'trend_asymmetry_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True),
            'mtf_trend_conviction_score': self.helper._get_mtf_slope_accel_score(df, 'trend_conviction_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_value_area_migration': self.helper._get_mtf_slope_accel_score(df, 'value_area_migration_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True),
            'mtf_value_area_overlap_pct': self.helper._get_mtf_slope_accel_score(df, 'value_area_overlap_pct_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_volatility_asymmetry_index': self.helper._get_mtf_slope_accel_score(df, 'volatility_asymmetry_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True),
            'mtf_volume_burstiness_index': self.helper._get_mtf_slope_accel_score(df, 'volume_burstiness_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_volume_structure_skew': self.helper._get_mtf_slope_accel_score(df, 'volume_structure_skew_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True),
            'mtf_vpin_score': self.helper._get_mtf_slope_accel_score(df, 'vpin_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_vwap_control_strength': self.helper._get_mtf_slope_accel_score(df, 'vwap_control_strength_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_vwap_cross_down_intensity': self.helper._get_mtf_slope_accel_score(df, 'vwap_cross_down_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_vwap_cross_up_intensity': self.helper._get_mtf_slope_accel_score(df, 'vwap_cross_up_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_vwap_crossing_intensity': self.helper._get_mtf_slope_accel_score(df, 'vwap_crossing_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_vwap_mean_reversion_corr': self.helper._get_mtf_slope_accel_score(df, 'vwap_mean_reversion_corr_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True),
            'mtf_vwap_sell_control_strength': self.helper._get_mtf_slope_accel_score(df, 'vwap_sell_control_strength_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_winner_stability_index': self.helper._get_mtf_slope_accel_score(df, 'winner_stability_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            'mtf_absorption_of_distribution_intensity': self.helper._get_mtf_slope_accel_score(df, 'absorption_of_distribution_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        }
        return mtf_signals

    def _calculate_historical_context(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], params: Dict, is_debug_enabled_for_method: bool, probe_ts: Optional[pd.Timestamp], _temp_debug_values: Dict) -> Dict[str, pd.Series]:
        """
        计算历史记忆与上下文机制信号。
        """
        hc_enabled = get_param_value(params.get('enabled'), False)
        cumulative_mf_flow_window = get_param_value(params.get('cumulative_mf_flow_window'), 21)
        cumulative_mf_flow_modulator_factor = get_param_value(params.get('cumulative_mf_flow_modulator_factor'), 0.1)
        chip_concentration_stability_window = get_param_value(params.get('chip_concentration_stability_window'), 55)
        chip_concentration_stability_modulator_factor = get_param_value(params.get('chip_concentration_stability_modulator_factor'), 0.05)
        long_term_trend_modulator_factor = get_param_value(params.get('long_term_trend_modulator_factor'), 0.15)
        hc_mtf_weights_medium = get_param_value(params.get('mtf_weights_medium'), {"21": 0.4, "34": 0.3, "55": 0.3})

        mtf_cumulative_mf_flow = pd.Series(0.0, index=df_index, dtype=np.float32)
        mtf_chip_concentration_stability = pd.Series(0.5, index=df_index, dtype=np.float32)
        mtf_long_term_trend_strength = pd.Series(0.0, index=df_index, dtype=np.float32)

        if hc_enabled:
            cumulative_mf_flow_long = raw_signals['main_force_net_flow'].rolling(window=cumulative_mf_flow_window, min_periods=int(cumulative_mf_flow_window * 0.5)).sum()
            mtf_cumulative_mf_flow = get_adaptive_mtf_normalized_score(
                cumulative_mf_flow_long, df_index, hc_mtf_weights_medium, ascending=True,
                debug_info=(is_debug_enabled_for_method, probe_ts, "mtf_cumulative_mf_flow")
            )
            rolling_std_winner_concentration = raw_signals['winner_concentration_90pct'].rolling(window=chip_concentration_stability_window, min_periods=int(chip_concentration_stability_window * 0.5)).std().replace(0, np.nan)
            chip_concentration_stability_raw = (1 / rolling_std_winner_concentration).fillna(0)
            mtf_chip_concentration_stability = get_adaptive_mtf_normalized_score(
                chip_concentration_stability_raw, df_index, hc_mtf_weights_medium, ascending=True,
                debug_info=(is_debug_enabled_for_method, probe_ts, "mtf_chip_concentration_stability")
            )
            mtf_long_term_trend_strength = get_adaptive_mtf_normalized_score(
                raw_signals['long_term_trend_slope'], df_index, hc_mtf_weights_medium, ascending=True,
                debug_info=(is_debug_enabled_for_method, probe_ts, "mtf_long_term_trend_strength")
            )
        
        _temp_debug_values["MTF融合信号"].update({
            "mtf_cumulative_mf_flow": mtf_cumulative_mf_flow,
            "mtf_chip_concentration_stability": mtf_chip_concentration_stability,
            "mtf_long_term_trend_strength": mtf_long_term_trend_strength
        })
        return {
            "mtf_cumulative_mf_flow": mtf_cumulative_mf_flow,
            "cumulative_mf_flow_modulator_factor": cumulative_mf_flow_modulator_factor,
            "mtf_chip_concentration_stability": mtf_chip_concentration_stability,
            "chip_concentration_stability_modulator_factor": chip_concentration_stability_modulator_factor,
            "mtf_long_term_trend_strength": mtf_long_term_trend_strength,
            "long_term_trend_modulator_factor": long_term_trend_modulator_factor,
            "hc_enabled": hc_enabled
        }

    def _normalize_raw_signals(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], method_name: str) -> Dict[str, pd.Series]:
        """
        归一化所有原始信号。
        """
        normalized_signals = {
            'price_impact_norm': self.helper._normalize_series(raw_signals['main_force_slippage'], df_index, bipolar=True),
            'impulse_purity_norm': self.helper._normalize_series(raw_signals['upward_impulse_purity'], df_index, bipolar=True),
            'volume_ratio_norm': self.helper._normalize_series(raw_signals['volume_ratio'] - 1.0, df_index, bipolar=True),
            'control_solidity_norm': self.helper._normalize_series(raw_signals['control_solidity'], df_index, bipolar=True),
            'cost_advantage_norm': self.helper._normalize_series(raw_signals['cost_advantage'], df_index, bipolar=True),
            'concentration_slope_norm': self.helper._normalize_series(raw_signals['concentration_slope'], df_index, bipolar=True),
            'peak_solidity_norm': self.helper._normalize_series(raw_signals['peak_solidity'], df_index, bipolar=True),
            'buying_support_norm': self.helper._normalize_series(raw_signals['active_buying_support'], df_index, bipolar=True),
            'pressure_rejection_norm': self.helper._normalize_series(raw_signals['pressure_rejection'], df_index, bipolar=True),
            'profit_absorption_norm': self.helper._normalize_series((1 - raw_signals['profit_realization_quality']) - 0.5, df_index, bipolar=True),
            'flow_credibility_norm': self.helper._normalize_series(raw_signals['flow_credibility'], df_index, bipolar=False),
            'chip_health_norm': self.helper._normalize_series(raw_signals['chip_health'], df_index, bipolar=False),
            'retail_fomo_norm': self.helper._normalize_series(raw_signals['retail_fomo'], df_index, bipolar=False),
            'order_book_imbalance_positive_norm': self.helper._normalize_series(raw_signals['order_book_imbalance'].clip(lower=0), df_index, bipolar=False),
            'micro_price_impact_asymmetry_positive_norm': self.helper._normalize_series(raw_signals['micro_price_impact_asymmetry'].clip(lower=0), df_index, bipolar=False),
            'constructive_turnover_norm': self.helper._normalize_series(raw_signals['constructive_turnover'], df_index, bipolar=False),
            'chip_fault_blockage_ratio_inverted_norm': self.helper._normalize_series(raw_signals['chip_fault_blockage_ratio'], df_index, bipolar=False, ascending=False),
            'vacuum_zone_magnitude_norm': self.helper._normalize_series(raw_signals['vacuum_zone_magnitude'], df_index, bipolar=False),
            'rally_buy_support_weakness_inverted_norm': self.helper._normalize_series(raw_signals['rally_buy_support_weakness'], df_index, bipolar=False, ascending=False),
            'order_book_clearing_rate_norm': self.helper._normalize_series(raw_signals['order_book_clearing_rate'], df_index, bipolar=False),
            'sell_sweep_intensity_inverted_norm': self.helper._normalize_series(raw_signals['sell_sweep_intensity'], df_index, bipolar=False, ascending=False),
            'main_force_flow_gini_inverted_norm': self.helper._normalize_series(raw_signals['main_force_flow_gini'], df_index, bipolar=False, ascending=False),
            'microstructure_efficiency_norm': self.helper._normalize_series(raw_signals['microstructure_efficiency'], df_index, bipolar=False),
            'imbalance_effectiveness_norm': self.helper._normalize_series(raw_signals['imbalance_effectiveness'], df_index, bipolar=False),
            'auction_showdown_norm': self.helper._normalize_series(raw_signals['auction_showdown'], df_index, bipolar=False),
            'closing_conviction_norm': self.helper._normalize_series(raw_signals['closing_conviction'], df_index, bipolar=False),
            'intraday_energy_density_norm': self.helper._normalize_series(raw_signals['intraday_energy_density'], df_index, bipolar=False),
            'intraday_thrust_purity_norm': self.helper._normalize_series(raw_signals['intraday_thrust_purity'], df_index, bipolar=False),
            'price_thrust_divergence_norm': self.helper._normalize_series(raw_signals['price_thrust_divergence'], df_index, bipolar=True),
            'trend_efficiency_ratio_norm': self.helper._normalize_series(raw_signals['trend_efficiency_ratio'], df_index, bipolar=False),
            'loser_concentration_90pct_norm': self.helper._normalize_series(raw_signals['loser_concentration_90pct'], df_index, bipolar=False),
            'winner_loser_momentum_norm': self.helper._normalize_series(raw_signals['winner_loser_momentum'], df_index, bipolar=True),
            'cost_structure_skewness_norm': self.helper._normalize_series(raw_signals['cost_structure_skewness'], df_index, bipolar=True),
            'cost_gini_coefficient_norm': self.helper._normalize_series(raw_signals['cost_gini_coefficient'], df_index, bipolar=False),
            'mf_vpoc_premium_norm': self.helper._normalize_series(raw_signals['mf_vpoc_premium'], df_index, bipolar=True),
            'character_score_norm': self.helper._normalize_series(raw_signals['character_score'], df_index, bipolar=False),
            'signal_conviction_score_norm': self.helper._normalize_series(raw_signals['signal_conviction_score'], df_index, bipolar=False),
            'touch_conviction_score_norm': self.helper._normalize_series(raw_signals['touch_conviction_score'], df_index, bipolar=False),
            'gathering_by_chasing_norm': self.helper._normalize_series(raw_signals['gathering_by_chasing'], df_index, bipolar=False),
            'gathering_by_support_norm': self.helper._normalize_series(raw_signals['gathering_by_support'], df_index, bipolar=False),
            'volatility_instability_norm': self.helper._normalize_series(raw_signals['volatility_instability'], df_index, bipolar=False),
            'adx_norm': self.helper._normalize_series(raw_signals['adx'], df_index, bipolar=False),
            'buy_flow_efficiency_norm': self.helper._normalize_series(raw_signals['buy_flow_efficiency'], df_index, bipolar=False),
            'sell_flow_efficiency_norm': self.helper._normalize_series(raw_signals['sell_flow_efficiency'], df_index, bipolar=False),
            'auction_closing_position_norm': self.helper._normalize_series(raw_signals['auction_closing_position'], df_index, bipolar=True),
            'auction_impact_score_norm': self.helper._normalize_series(raw_signals['auction_impact_score'], df_index, bipolar=False),
            'auction_intent_signal_norm': self.helper._normalize_series(raw_signals['auction_intent_signal'], df_index, bipolar=False),
            'order_book_liquidity_supply_norm': self.helper._normalize_series(raw_signals['order_book_liquidity_supply'], df_index, bipolar=False),
            'liquidity_slope_norm': self.helper._normalize_series(raw_signals['liquidity_slope'], df_index, bipolar=True),
            'peak_mass_transfer_rate_norm': self.helper._normalize_series(raw_signals['peak_mass_transfer_rate'], df_index, bipolar=False),
            'mf_cost_zone_defense_intent_norm': self.helper._normalize_series(raw_signals['mf_cost_zone_defense_intent'], df_index, bipolar=False),
            'bid_side_liquidity_norm': self.helper._normalize_series(raw_signals['bid_side_liquidity'], df_index, bipolar=False),
            'ask_side_liquidity_norm': self.helper._normalize_series(raw_signals['ask_side_liquidity'], df_index, bipolar=False),
            'retail_panic_surrender_norm': self.helper._normalize_series(raw_signals['retail_panic_surrender'], df_index, bipolar=False),
            'main_force_activity_ratio_norm': self.helper._normalize_series(raw_signals['main_force_activity_ratio'], df_index, bipolar=False),
            'main_force_conviction_index_norm': self.helper._normalize_series(raw_signals['main_force_conviction_index'], df_index, bipolar=False),
            'main_force_execution_alpha_norm': self.helper._normalize_series(raw_signals['main_force_execution_alpha'], df_index, bipolar=True),
            'main_force_flow_directionality_norm': self.helper._normalize_series(raw_signals['main_force_flow_directionality'], df_index, bipolar=True),
            'main_force_on_peak_buy_flow_norm': self.helper._normalize_series(raw_signals['main_force_on_peak_buy_flow'], df_index, bipolar=False),
            'main_force_on_peak_sell_flow_norm': self.helper._normalize_series(raw_signals['main_force_on_peak_sell_flow'], df_index, bipolar=False),
            'main_force_t0_efficiency_norm': self.helper._normalize_series(raw_signals['main_force_t0_efficiency'], df_index, bipolar=False),
            'main_force_t0_sell_efficiency_norm': self.helper._normalize_series(raw_signals['main_force_t0_sell_efficiency'], df_index, bipolar=False),
            'main_force_vwap_down_guidance_norm': self.helper._normalize_series(raw_signals['main_force_vwap_down_guidance'], df_index, bipolar=True),
            'main_force_vwap_up_guidance_norm': self.helper._normalize_series(raw_signals['main_force_vwap_up_guidance'], df_index, bipolar=True),
            'market_impact_cost_norm': self.helper._normalize_series(raw_signals['market_impact_cost'], df_index, bipolar=False),
            'opening_buy_strength_norm': self.helper._normalize_series(raw_signals['opening_buy_strength'], df_index, bipolar=False),
            'opening_sell_strength_norm': self.helper._normalize_series(raw_signals['opening_sell_strength'], df_index, bipolar=False),
            'closing_strength_index_norm': self.helper._normalize_series(raw_signals['closing_strength_index'], df_index, bipolar=False),
            'total_buy_amount_calibrated_norm': self.helper._normalize_series(raw_signals['total_buy_amount_calibrated'], df_index, bipolar=False),
            'total_sell_amount_calibrated_norm': self.helper._normalize_series(raw_signals['total_sell_amount_calibrated'], df_index, bipolar=False),
            'wash_trade_intensity_norm': self.helper._normalize_series(raw_signals['wash_trade_intensity'], df_index, bipolar=False),
            'winner_profit_margin_avg_norm': self.helper._normalize_series(raw_signals['winner_profit_margin_avg'], df_index, bipolar=True),
            'loser_loss_margin_avg_norm': self.helper._normalize_series(raw_signals['loser_loss_margin_avg'], df_index, bipolar=True),
            'total_winner_rate_norm': self.helper._normalize_series(raw_signals['total_winner_rate'], df_index, bipolar=False),
            'total_loser_rate_norm': self.helper._normalize_series(raw_signals['total_loser_rate'], df_index, bipolar=False),
            'impulse_quality_ratio_norm': self.helper._normalize_series(raw_signals['impulse_quality_ratio'], df_index, bipolar=False),
            'thrust_efficiency_score_norm': self.helper._normalize_series(raw_signals['thrust_efficiency_score'], df_index, bipolar=False),
            'platform_conviction_score_norm': self.helper._normalize_series(raw_signals['platform_conviction_score'], df_index, bipolar=False),
            'platform_high_norm': self.helper._normalize_series(raw_signals['platform_high'], df_index, bipolar=False),
            'platform_low_norm': self.helper._normalize_series(raw_signals['platform_low'], df_index, bipolar=False),
            'breakout_quality_score_norm': self.helper._normalize_series(raw_signals['breakout_quality_score'], df_index, bipolar=False),
            'breakout_readiness_score_norm': self.helper._normalize_series(raw_signals['breakout_readiness_score'], df_index, bipolar=False),
            'breakthrough_conviction_score_norm': self.helper._normalize_series(raw_signals['breakthrough_conviction_score'], df_index, bipolar=False),
            'defense_solidity_score_norm': self.helper._normalize_series(raw_signals['defense_solidity_score'], df_index, bipolar=False),
            'support_validation_strength_norm': self.helper._normalize_series(raw_signals['support_validation_strength'], df_index, bipolar=False),
            'covert_accumulation_signal_norm': self.helper._normalize_series(raw_signals['covert_accumulation_signal'], df_index, bipolar=False),
            'suppressive_accumulation_intensity_norm': self.helper._normalize_series(raw_signals['suppressive_accumulation_intensity'], df_index, bipolar=False),
            'deception_index_norm': self.helper._normalize_series(raw_signals['deception_index'], df_index, bipolar=False),
            'deception_lure_long_intensity_norm': self.helper._normalize_series(raw_signals['deception_lure_long_intensity'], df_index, bipolar=False),
            'equilibrium_compression_index_norm': self.helper._normalize_series(raw_signals['equilibrium_compression_index'], df_index, bipolar=False),
            'final_charge_intensity_norm': self.helper._normalize_series(raw_signals['final_charge_intensity'], df_index, bipolar=False),
            'floating_chip_cleansing_efficiency_norm': self.helper._normalize_series(raw_signals['floating_chip_cleansing_efficiency'], df_index, bipolar=False),
            'hidden_accumulation_intensity_norm': self.helper._normalize_series(raw_signals['hidden_accumulation_intensity'], df_index, bipolar=False),
            'internal_accumulation_intensity_norm': self.helper._normalize_series(raw_signals['internal_accumulation_intensity'], df_index, bipolar=False),
            'intraday_posture_score_norm': self.helper._normalize_series(raw_signals['intraday_posture_score'], df_index, bipolar=False),
            'opening_gap_defense_strength_norm': self.helper._normalize_series(raw_signals['opening_gap_defense_strength'], df_index, bipolar=False),
            'panic_buy_absorption_contribution_norm': self.helper._normalize_series(raw_signals['panic_buy_absorption_contribution'], df_index, bipolar=False),
            'panic_sell_volume_contribution_norm': self.helper._normalize_series(raw_signals['panic_sell_volume_contribution'], df_index, bipolar=False),
            'panic_selling_cascade_norm': self.helper._normalize_series(raw_signals['panic_selling_cascade'], df_index, bipolar=False),
            'peak_control_transfer_norm': self.helper._normalize_series(raw_signals['peak_control_transfer'], df_index, bipolar=False),
            'peak_separation_ratio_norm': self.helper._normalize_series(raw_signals['peak_separation_ratio'], df_index, bipolar=False),
            'price_reversion_velocity_norm': self.helper._normalize_series(raw_signals['price_reversion_velocity'], df_index, bipolar=True),
            'pullback_depth_ratio_norm': self.helper._normalize_series(raw_signals['pullback_depth_ratio'], df_index, bipolar=False),
            'quality_score_norm': self.helper._normalize_series(raw_signals['quality_score'], df_index, bipolar=False),
            'reversal_conviction_rate_norm': self.helper._normalize_series(raw_signals['reversal_conviction_rate'], df_index, bipolar=False),
            'reversal_power_index_norm': self.helper._normalize_series(raw_signals['reversal_power_index'], df_index, bipolar=False),
            'reversal_recovery_rate_norm': self.helper._normalize_series(raw_signals['reversal_recovery_rate'], df_index, bipolar=False),
            'risk_reward_profile_norm': self.helper._normalize_series(raw_signals['risk_reward_profile'], df_index, bipolar=True),
            'shock_conviction_score_norm': self.helper._normalize_series(raw_signals['shock_conviction_score'], df_index, bipolar=False),
            'strategic_phase_score_norm': self.helper._normalize_series(raw_signals['strategic_phase_score'], df_index, bipolar=False),
            'structural_entropy_change_norm': self.helper._normalize_series(raw_signals['structural_entropy_change'], df_index, bipolar=True),
            'structural_leverage_norm': self.helper._normalize_series(raw_signals['structural_leverage'], df_index, bipolar=False),
            'structural_node_count_norm': self.helper._normalize_series(raw_signals['structural_node_count'], df_index, bipolar=False),
            'structural_potential_score_norm': self.helper._normalize_series(raw_signals['structural_potential_score'], df_index, bipolar=False),
            'support_validation_score_norm': self.helper._normalize_series(raw_signals['support_validation_score'], df_index, bipolar=False),
            'supportive_distribution_intensity_norm': self.helper._normalize_series(raw_signals['supportive_distribution_intensity'], df_index, bipolar=False),
            'trend_acceleration_score_norm': self.helper._normalize_series(raw_signals['trend_acceleration_score'], df_index, bipolar=False),
            'trend_alignment_index_norm': self.helper._normalize_series(raw_signals['trend_alignment_index'], df_index, bipolar=False),
            'trend_asymmetry_index_norm': self.helper._normalize_series(raw_signals['trend_asymmetry_index'], df_index, bipolar=True),
            'trend_conviction_score_norm': self.helper._normalize_series(raw_signals['trend_conviction_score'], df_index, bipolar=False),
            'value_area_migration_norm': self.helper._normalize_series(raw_signals['value_area_migration'], df_index, bipolar=True),
            'value_area_overlap_pct_norm': self.helper._normalize_series(raw_signals['value_area_overlap_pct'], df_index, bipolar=False),
            'volatility_asymmetry_index_norm': self.helper._normalize_series(raw_signals['volatility_asymmetry_index'], df_index, bipolar=True),
            'volume_burstiness_index_norm': self.helper._normalize_series(raw_signals['volume_burstiness_index'], df_index, bipolar=False),
            'volume_structure_skew_norm': self.helper._normalize_series(raw_signals['volume_structure_skew'], df_index, bipolar=True),
            'vpin_score_norm': self.helper._normalize_series(raw_signals['vpin_score'], df_index, bipolar=False),
            'vwap_control_strength_norm': self.helper._normalize_series(raw_signals['vwap_control_strength'], df_index, bipolar=False),
            'vwap_cross_down_intensity_norm': self.helper._normalize_series(raw_signals['vwap_cross_down_intensity'], df_index, bipolar=False),
            'vwap_cross_up_intensity_norm': self.helper._normalize_series(raw_signals['vwap_cross_up_intensity'], df_index, bipolar=False),
            'vwap_crossing_intensity_norm': self.helper._normalize_series(raw_signals['vwap_crossing_intensity'], df_index, bipolar=False),
            'vwap_mean_reversion_corr_norm': self.helper._normalize_series(raw_signals['vwap_mean_reversion_corr'], df_index, bipolar=True),
            'vwap_sell_control_strength_norm': self.helper._normalize_series(raw_signals['vwap_sell_control_strength'], df_index, bipolar=False),
            'winner_stability_index_norm': self.helper._normalize_series(raw_signals['winner_stability_index'], df_index, bipolar=False),
            'absorption_of_distribution_intensity_norm': self.helper._normalize_series(raw_signals['absorption_of_distribution_intensity'], df_index, bipolar=False)
        }
        return normalized_signals

    def _construct_proxy_signals(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], config: Dict) -> Dict[str, pd.Series]:
        """
        构建代理信号。
        """
        rs_modulator_proxy = (mtf_signals['mtf_price_trend'].clip(lower=0) * 0.4 + mtf_signals['mtf_trend_vitality'] * 0.3 + mtf_signals['mtf_breakout_quality_score'] * 0.3).clip(0,1)
        rs_modulator = (1 + rs_modulator_proxy * config.get('relative_strength_amplifier', 0.0))
        
        capital_modulator_proxy = (mtf_signals['mtf_mf_net_flow'].clip(lower=0) * 0.2 + normalized_signals['main_force_flow_gini_inverted_norm'] * 0.15 + mtf_signals['mtf_main_force_buy_ofi'] * 0.15 +
                                   mtf_signals['mtf_main_force_t0_buy_efficiency'] * 0.1 + normalized_signals['main_force_activity_ratio_norm'] * 0.1 + normalized_signals['main_force_conviction_index_norm'] * 0.1 +
                                   mtf_signals['mtf_main_force_execution_alpha'].clip(lower=0) * 0.1 + mtf_signals['mtf_main_force_flow_directionality'].clip(lower=0) * 0.1).clip(0,1)
        capital_modulator = (1 + capital_modulator_proxy * config.get('capital_signature_modulator_weight', 0.0))
        
        market_sentiment_proxy = (mtf_signals['mtf_market_sentiment'].clip(lower=0) * 0.4 + mtf_signals['mtf_retail_fomo'] * 0.3 + (1 - mtf_signals['mtf_retail_panic_surrender']) * 0.3).clip(0,1)
        
        liquidity_tide_proxy = (mtf_signals['mtf_order_book_liquidity_supply'] * 0.3 + mtf_signals['mtf_liquidity_slope'].clip(lower=0) * 0.3 +
                                mtf_signals['mtf_bid_side_liquidity'] * 0.2 + (1 - mtf_signals['mtf_ask_side_liquidity']) * 0.2).clip(0,1)
        return {
            "rs_modulator": rs_modulator,
            "capital_modulator": capital_modulator,
            "market_sentiment_proxy": market_sentiment_proxy,
            "liquidity_tide_proxy": liquidity_tide_proxy
        }

    def _calculate_dynamic_weights(self, df_index: pd.Index, normalized_signals: Dict[str, pd.Series], proxy_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> Dict[str, pd.Series]:
        """
        计算动态权重。
        """
        # 市场情境因子：波动率、趋势强度、市场情绪、流动性潮汐
        # 市场稳定性得分：低波动率 + 强趋势
        market_stability_score = (1 - normalized_signals['volatility_instability_norm']) * normalized_signals['adx_norm']
        market_stability_norm = self.helper._normalize_series(market_stability_score, df_index, bipolar=False)
        # 市场情绪得分 (0-1范围)
        market_sentiment_norm_unipolar = proxy_signals['market_sentiment_proxy']
        # 流动性潮汐得分 (0-1范围)
        liquidity_tide_norm_unipolar = proxy_signals['liquidity_tide_proxy']
        # 综合情境得分 (0-1范围)
        overall_context_score = (market_stability_norm * 0.4 + market_sentiment_norm_unipolar * 0.3 + liquidity_tide_norm_unipolar * 0.3).clip(0,1)
        # 定义基础权重 (可配置)
        base_weights = {
            "aggressiveness": 0.3,
            "control": 0.3,
            "obstacle_clearance": 0.2,
            "risk": 0.2
        }
        # 根据综合情境得分动态调整权重
        dynamic_weights = {}
        for key, base_w in base_weights.items():
            if key in ["aggressiveness", "control"]:
                # 综合情境越好，权重增加越多
                dynamic_weights[key] = base_w * (1 + overall_context_score * 0.4 - (1 - overall_context_score) * 0.1)
            elif key == "obstacle_clearance":
                # 综合情境越好，权重略增
                dynamic_weights[key] = base_w * (1 + overall_context_score * 0.1 - (1 - overall_context_score) * 0.05)
            elif key == "risk":
                # 综合情境越好，风险权重降低越多
                dynamic_weights[key] = base_w * (1 - overall_context_score * 0.4 + (1 - overall_context_score) * 0.1)
            dynamic_weights[key] = dynamic_weights[key].clip(0.05, 0.5) # 限制权重范围
        # 归一化动态权重，确保和为1
        total_dynamic_weight = pd.Series(0.0, index=df_index, dtype=np.float32)
        for key in dynamic_weights:
            total_dynamic_weight += dynamic_weights[key]
        # 避免除以零
        total_dynamic_weight = total_dynamic_weight.replace(0, 1e-9)
        for key in dynamic_weights:
            dynamic_weights[key] = dynamic_weights[key] / total_dynamic_weight
        _temp_debug_values["动态权重"] = {
            "market_stability_score": market_stability_score,
            "market_stability_norm": market_stability_norm,
            "market_sentiment_norm_unipolar": market_sentiment_norm_unipolar,
            "liquidity_tide_norm_unipolar": liquidity_tide_norm_unipolar,
            "overall_context_score": overall_context_score,
            "dynamic_weights_aggressiveness": dynamic_weights["aggressiveness"],
            "dynamic_weights_control": dynamic_weights["control"],
            "dynamic_weights_obstacle_clearance": dynamic_weights["obstacle_clearance"],
            "dynamic_weights_risk": dynamic_weights["risk"]
        }
        return dynamic_weights

    def _calculate_aggressiveness_score(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], dynamic_weights: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        计算攻击性分数。
        """
        aggressiveness_components = {
            "mtf_price_trend": mtf_signals['mtf_price_trend'].clip(lower=0),
            "mtf_mf_net_flow": mtf_signals['mtf_mf_net_flow'].clip(lower=0),
            "price_impact_norm": normalized_signals['price_impact_norm'].clip(lower=0),
            "impulse_purity_norm": normalized_signals['impulse_purity_norm'].clip(lower=0),
            "volume_ratio_norm": normalized_signals['volume_ratio_norm'].clip(lower=0),
            "flow_credibility_norm": normalized_signals['flow_credibility_norm'],
            "chip_health_norm": normalized_signals['chip_health_norm'],
            "mtf_buy_sweep_intensity": mtf_signals['mtf_buy_sweep_intensity'],
            "mtf_main_force_buy_ofi": mtf_signals['mtf_main_force_buy_ofi'],
            "mtf_main_force_t0_buy_efficiency": mtf_signals['mtf_main_force_t0_buy_efficiency'],
            "order_book_imbalance_positive_norm": normalized_signals['order_book_imbalance_positive_norm'],
            "micro_price_impact_asymmetry_positive_norm": normalized_signals['micro_price_impact_asymmetry_positive_norm'],
            "constructive_turnover_norm": normalized_signals['constructive_turnover_norm'],
            "mtf_upward_impulse_strength": mtf_signals['mtf_upward_impulse_strength'],
            "mtf_order_book_clearing_rate": mtf_signals['mtf_order_book_clearing_rate'],
            "sell_sweep_intensity_inverted_norm": normalized_signals['sell_sweep_intensity_inverted_norm'],
            "microstructure_efficiency_norm": normalized_signals['microstructure_efficiency_norm'],
            "imbalance_effectiveness_norm": normalized_signals['imbalance_effectiveness_norm'],
            "mtf_auction_showdown": mtf_signals['mtf_auction_showdown'],
            "mtf_closing_conviction": mtf_signals['mtf_closing_conviction'],
            "mtf_intraday_energy_density": mtf_signals['mtf_intraday_energy_density'],
            "mtf_intraday_thrust_purity": mtf_signals['mtf_intraday_thrust_purity'],
            "mtf_buy_flow_efficiency": mtf_signals['mtf_buy_flow_efficiency'],
            "mtf_sell_flow_efficiency_inverted": (1 - mtf_signals['mtf_sell_flow_efficiency']), # 卖方效率低，买方攻击性强
            "mtf_auction_closing_position_positive": mtf_signals['mtf_auction_closing_position'].clip(lower=0), # 集合竞价收盘位置偏高
            "mtf_auction_impact_score": mtf_signals['mtf_auction_impact_score'],
            "mtf_auction_intent_signal": mtf_signals['mtf_auction_intent_signal'],
            "mtf_opening_buy_strength": mtf_signals['mtf_opening_buy_strength'],
            "mtf_closing_strength_index": mtf_signals['mtf_closing_strength_index'],
            "mtf_total_buy_amount_calibrated": mtf_signals['mtf_total_buy_amount_calibrated'],
            "mtf_main_force_vwap_up_guidance": mtf_signals['mtf_main_force_vwap_up_guidance'].clip(lower=0),
            "mtf_impulse_quality_ratio": mtf_signals['mtf_impulse_quality_ratio'],
            "mtf_thrust_efficiency_score": mtf_signals['mtf_thrust_efficiency_score'],
            "mtf_breakout_quality_score": mtf_signals['mtf_breakout_quality_score'],
            "mtf_breakout_readiness_score": mtf_signals['mtf_breakout_readiness_score'],
            "mtf_breakthrough_conviction_score": mtf_signals['mtf_breakthrough_conviction_score'],
            "mtf_final_charge_intensity": mtf_signals['mtf_final_charge_intensity'],
            "mtf_hidden_accumulation_intensity": mtf_signals['mtf_hidden_accumulation_intensity'],
            "mtf_internal_accumulation_intensity": mtf_signals['mtf_internal_accumulation_intensity'],
            "mtf_intraday_posture_score": mtf_signals['mtf_intraday_posture_score'],
            "mtf_panic_buy_absorption_contribution": mtf_signals['mtf_panic_buy_absorption_contribution'],
            "mtf_reversal_power_index": mtf_signals['mtf_reversal_power_index'],
            "mtf_trend_acceleration_score": mtf_signals['mtf_trend_acceleration_score'],
            "mtf_trend_conviction_score": mtf_signals['mtf_trend_conviction_score'],
            "mtf_vwap_cross_up_intensity": mtf_signals['mtf_vwap_cross_up_intensity'],
            "mtf_absorption_of_distribution_intensity": mtf_signals['mtf_absorption_of_distribution_intensity'] # 新增：派发吸收强度作为攻击性的一部分
        }
        aggressiveness_weights = {
            "mtf_price_trend": 0.04, "mtf_mf_net_flow": 0.04, "price_impact_norm": 0.03,
            "impulse_purity_norm": 0.03, "volume_ratio_norm": 0.02, "flow_credibility_norm": 0.02,
            "chip_health_norm": 0.02, "mtf_buy_sweep_intensity": 0.05, "mtf_main_force_buy_ofi": 0.05,
            "mtf_main_force_t0_buy_efficiency": 0.05, "order_book_imbalance_positive_norm": 0.03,
            "micro_price_impact_asymmetry_positive_norm": 0.03, "constructive_turnover_norm": 0.03,
            "mtf_upward_impulse_strength": 0.03, "mtf_order_book_clearing_rate": 0.02,
            "sell_sweep_intensity_inverted_norm": 0.02, "microstructure_efficiency_norm": 0.02,
            "imbalance_effectiveness_norm": 0.02, "mtf_auction_showdown": 0.01,
            "mtf_closing_conviction": 0.01, "mtf_intraday_energy_density": 0.01,
            "mtf_intraday_thrust_purity": 0.01, "mtf_buy_flow_efficiency": 0.02,
            "mtf_sell_flow_efficiency_inverted": 0.01, "mtf_auction_closing_position_positive": 0.01,
            "mtf_auction_impact_score": 0.01, "mtf_auction_intent_signal": 0.01,
            "mtf_opening_buy_strength": 0.02, "mtf_closing_strength_index": 0.02,
            "mtf_total_buy_amount_calibrated": 0.02, "mtf_main_force_vwap_up_guidance": 0.02,
            "mtf_impulse_quality_ratio": 0.02, "mtf_thrust_efficiency_score": 0.02,
            "mtf_breakout_quality_score": 0.02, "mtf_breakout_readiness_score": 0.02,
            "mtf_breakthrough_conviction_score": 0.02, "mtf_final_charge_intensity": 0.02,
            "mtf_hidden_accumulation_intensity": 0.02, "mtf_internal_accumulation_intensity": 0.02,
            "mtf_intraday_posture_score": 0.02, "mtf_panic_buy_absorption_contribution": 0.02,
            "mtf_reversal_power_index": 0.02, "mtf_trend_acceleration_score": 0.02,
            "mtf_trend_conviction_score": 0.02, "mtf_vwap_cross_up_intensity": 0.02,
            "mtf_absorption_of_distribution_intensity": 0.05 # 提高权重，从0.03调整为0.05
        }
        aggressiveness_score = _robust_geometric_mean(aggressiveness_components, aggressiveness_weights, df_index).clip(0, 1)
        _temp_debug_values["攻击性"] = {
            "aggressiveness_score": aggressiveness_score
        }
        return aggressiveness_score

    def _calculate_control_score(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], historical_context: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        计算控制力分数。
        """
        control_components = {
            "control_solidity_norm": normalized_signals['control_solidity_norm'].clip(lower=0),
            "cost_advantage_norm": normalized_signals['cost_advantage_norm'].clip(lower=0),
            "concentration_slope_norm": normalized_signals['concentration_slope_norm'].clip(lower=0),
            "peak_solidity_norm": normalized_signals['peak_solidity_norm'].clip(lower=0),
            "mtf_vwap_buy_control_strength": mtf_signals['mtf_vwap_buy_control_strength'],
            "mtf_mf_cost_zone_buy_intent": mtf_signals['mtf_mf_cost_zone_buy_intent'],
            "chip_fault_blockage_ratio_inverted_norm": normalized_signals['chip_fault_blockage_ratio_inverted_norm'],
            "main_force_flow_gini_inverted_norm": normalized_signals['main_force_flow_gini_inverted_norm'],
            "mtf_cost_structure_skewness_positive": mtf_signals['mtf_cost_structure_skewness'].clip(lower=0), # 成本结构偏度正向
            "cost_gini_coefficient_norm": normalized_signals['cost_gini_coefficient_norm'],
            "mtf_mf_vpoc_premium_positive": mtf_signals['mtf_mf_vpoc_premium'].clip(lower=0), # VPOC溢价正向
            "mtf_mf_cost_zone_defense_intent_inverted": (1 - mtf_signals['mtf_mf_cost_zone_defense_intent']), # 成本区防守意图低，控盘强
            "mtf_main_force_activity_ratio": mtf_signals['mtf_main_force_activity_ratio'],
            "mtf_main_force_conviction_index": mtf_signals['mtf_main_force_conviction_index'],
            "mtf_main_force_execution_alpha": mtf_signals['mtf_main_force_execution_alpha'].clip(lower=0),
            "mtf_main_force_flow_directionality": mtf_signals['mtf_main_force_flow_directionality'].clip(lower=0),
            "mtf_main_force_on_peak_buy_flow": mtf_signals['mtf_main_force_on_peak_buy_flow'],
            "mtf_main_force_t0_efficiency": mtf_signals['mtf_main_force_t0_efficiency'],
            "mtf_main_force_vwap_up_guidance": mtf_signals['mtf_main_force_vwap_up_guidance'].clip(lower=0),
            "mtf_platform_conviction_score": mtf_signals['mtf_platform_conviction_score'],
            "mtf_peak_control_transfer": mtf_signals['mtf_peak_control_transfer'],
            "mtf_vwap_control_strength": mtf_signals['mtf_vwap_control_strength'],
            "mtf_winner_stability_index": mtf_signals['mtf_winner_stability_index']
        }
        control_weights = {
            "control_solidity_norm": 0.08, "cost_advantage_norm": 0.07, "concentration_slope_norm": 0.07,
            "peak_solidity_norm": 0.06, "mtf_vwap_buy_control_strength": 0.06,
            "mtf_mf_cost_zone_buy_intent": 0.06, "chip_fault_blockage_ratio_inverted_norm": 0.04,
            "main_force_flow_gini_inverted_norm": 0.04, "mtf_cost_structure_skewness_positive": 0.04,
            "cost_gini_coefficient_norm": 0.04, "mtf_mf_vpoc_premium_positive": 0.04,
            "mtf_mf_cost_zone_defense_intent_inverted": 0.05, "mtf_main_force_activity_ratio": 0.04,
            "mtf_main_force_conviction_index": 0.04, "mtf_main_force_execution_alpha": 0.04,
            "mtf_main_force_flow_directionality": 0.04, "mtf_main_force_on_peak_buy_flow": 0.03,
            "mtf_main_force_t0_efficiency": 0.03, "mtf_main_force_vwap_up_guidance": 0.03,
            "mtf_platform_conviction_score": 0.03, "mtf_peak_control_transfer": 0.03,
            "mtf_vwap_control_strength": 0.03, "mtf_winner_stability_index": 0.03
        }
        control_score = _robust_geometric_mean(control_components, control_weights, df_index).clip(0, 1)
        # V11.1: 应用筹码集中度稳定性调节器
        if historical_context['hc_enabled']:
            control_score = (control_score * (1 + historical_context['mtf_chip_concentration_stability'] * historical_context['chip_concentration_stability_modulator_factor'])).clip(0, 1)
        _temp_debug_values["控制力"] = {
            "control_score": control_score
        }
        return control_score

    def _calculate_obstacle_clearance_score(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        计算障碍清除分数。
        """
        obstacle_clearance_components = {
            "buying_support_norm": normalized_signals['buying_support_norm'].clip(lower=0),
            "pressure_rejection_norm": normalized_signals['pressure_rejection_norm'].clip(lower=0),
            "profit_absorption_norm": normalized_signals['profit_absorption_norm'].clip(lower=0),
            "mtf_vacuum_traversal_efficiency": mtf_signals['mtf_vacuum_traversal_efficiency'],
            "vacuum_zone_magnitude_norm": normalized_signals['vacuum_zone_magnitude_norm'],
            "mtf_dip_buy_absorption_strength": mtf_signals['mtf_dip_buy_absorption_strength'],
            "rally_buy_support_weakness_inverted_norm": normalized_signals['rally_buy_support_weakness_inverted_norm'],
            "mtf_price_thrust_divergence_positive": mtf_signals['mtf_price_thrust_divergence'].clip(lower=0), # 价格推力正向
            "mtf_trend_efficiency_ratio": mtf_signals['mtf_trend_efficiency_ratio'],
            "mtf_order_book_liquidity_supply": mtf_signals['mtf_order_book_liquidity_supply'],
            "mtf_liquidity_slope_positive": mtf_signals['mtf_liquidity_slope'].clip(lower=0), # 流动性斜率正向
            "bid_side_liquidity_norm": normalized_signals['bid_side_liquidity_norm'],
            "ask_side_liquidity_inverted_norm": (1 - normalized_signals['ask_side_liquidity_norm']),
            "mtf_defense_solidity_score": mtf_signals['mtf_defense_solidity_score'],
            "mtf_support_validation_strength": mtf_signals['mtf_support_validation_strength'],
            "mtf_floating_chip_cleansing_efficiency": mtf_signals['mtf_floating_chip_cleansing_efficiency'],
            "mtf_opening_gap_defense_strength": mtf_signals['mtf_opening_gap_defense_strength'],
            "mtf_peak_separation_ratio": mtf_signals['mtf_peak_separation_ratio'],
            "mtf_pullback_depth_ratio_inverted": (1 - mtf_signals['mtf_pullback_depth_ratio']),
            "mtf_support_validation_score": mtf_signals['mtf_support_validation_score'],
            "mtf_value_area_migration_positive": mtf_signals['mtf_value_area_migration'].clip(lower=0),
            "mtf_value_area_overlap_pct": mtf_signals['mtf_value_area_overlap_pct']
        }
        obstacle_clearance_weights = {
            "buying_support_norm": 0.08, "pressure_rejection_norm": 0.07, "profit_absorption_norm": 0.07,
            "mtf_vacuum_traversal_efficiency": 0.06, "vacuum_zone_magnitude_norm": 0.03,
            "mtf_dip_buy_absorption_strength": 0.08, "rally_buy_support_weakness_inverted_norm": 0.03,
            "mtf_price_thrust_divergence_positive": 0.03, "mtf_trend_efficiency_ratio": 0.06,
            "mtf_order_book_liquidity_supply": 0.05, "mtf_liquidity_slope_positive": 0.04,
            "bid_side_liquidity_norm": 0.04, "ask_side_liquidity_inverted_norm": 0.03,
            "mtf_defense_solidity_score": 0.04, "mtf_support_validation_strength": 0.04,
            "mtf_floating_chip_cleansing_efficiency": 0.03, "mtf_opening_gap_defense_strength": 0.03,
            "mtf_peak_separation_ratio": 0.03, "mtf_pullback_depth_ratio_inverted": 0.03,
            "mtf_support_validation_score": 0.03, "mtf_value_area_migration_positive": 0.03,
            "mtf_value_area_overlap_pct": 0.03
        }
        obstacle_clearance_score = _robust_geometric_mean(obstacle_clearance_components, obstacle_clearance_weights, df_index).clip(0, 1)
        _temp_debug_values["障碍清除"] = {
            "obstacle_clearance_score": obstacle_clearance_score
        }
        return obstacle_clearance_score

    def _synthesize_bullish_intent(self, df_index: pd.Index, aggressiveness_score: pd.Series, control_score: pd.Series, obstacle_clearance_score: pd.Series, mtf_signals: Dict[str, pd.Series], dynamic_weights: Dict[str, pd.Series], historical_context: Dict[str, pd.Series], rally_intent_synthesis_params: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        合成基础看涨意图。
        """
        # V11.1: 引入长期趋势强度上下文调节器
        long_term_trend_strength_modulator = (1 + historical_context['mtf_long_term_trend_strength'] * historical_context['long_term_trend_modulator_factor']) # 长期趋势越强，看涨意图越受加成

        # 攻击性、控制力、障碍清除的加权平均
        bullish_intent_base = (
            (aggressiveness_score * dynamic_weights["aggressiveness"] +
             control_score * dynamic_weights["control"] +
             obstacle_clearance_score * dynamic_weights["obstacle_clearance"]) /
            (dynamic_weights["aggressiveness"] + dynamic_weights["control"] + dynamic_weights["obstacle_clearance"])
        )
        # V11.3: 强化 mtf_absorption_of_distribution_intensity 的积极作用，直接贡献于看涨意图
        # 承接强度越高，看涨意图越强，这里使用一个较小的系数，避免过度放大
        bullish_intent = (bullish_intent_base + mtf_signals['mtf_absorption_of_distribution_intensity'] * 0.1).clip(0, 1)

        # V11.1: 应用长期趋势强度调节
        bullish_intent = (bullish_intent * long_term_trend_strength_modulator).clip(0, 1)

        # 幂平均，放大高分，抑制低分
        power_mean_exponent = get_param_value(rally_intent_synthesis_params.get('power_mean_exponent'), 2.0)
        bullish_intent = bullish_intent.pow(power_mean_exponent)
        _temp_debug_values["基础看涨意图"] = {
            "power_mean_exponent": power_mean_exponent,
            "bullish_intent_base": bullish_intent_base, # 增加基础看涨意图的调试输出
            "bullish_intent": bullish_intent
        }
        return bullish_intent

    def _calculate_bearish_intent(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], historical_context: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        计算看跌意图。
        """
        # V11.1: 引入主力资金累计记忆调节器
        # 历史累计资金流入强劲 (mtf_cumulative_mf_flow > 0)，削弱看跌意图
        mf_flow_memory_anti_bearish_modulator = (1 - historical_context['mtf_cumulative_mf_flow'].clip(lower=0) * historical_context['cumulative_mf_flow_modulator_factor']).clip(0, 1)

        # 看跌意图的计算
        distribution_intensity_norm = normalized_signals['distribution_intensity_norm']
        upper_shadow_selling_pressure_norm = normalized_signals['upper_shadow_selling_pressure_norm']
        flow_credibility_norm_inverted = (1 - normalized_signals['flow_credibility_norm']) # 信用度低，看跌
        
        # V11.2: 引入派发情境衰减器 (Distribution Context Dampener)
        # 当日涨幅越大，对派发强度的看跌解读越弱
        # 使用 tanh 函数将涨幅映射到 [0, 1] 范围内的衰减因子
        # 例如，pct_change = 7% -> tanh(0.07 * 10) = tanh(0.7) = 0.6，衰减因子为 1 - 0.6 = 0.4
        # pct_change = 0% -> tanh(0) = 0，衰减因子为 1
        # pct_change = -5% -> tanh(-0.5) = -0.46，衰减因子为 1 - (-0.46) = 1.46 (负涨幅反而增强派发风险)
        # 确保衰减因子在合理范围，例如 [0.1, 1.5]
        distribution_dampener = (1 - np.tanh(raw_signals['pct_change'] / 100 * 10)).clip(0.1, 1.5) # 涨幅越大，dampener越小，削弱派发影响

        # 修正 bearish_score 的计算逻辑：它应该是一个正值，表示看跌意图的强度，然后乘以 -1 转换为双极性。
        # 并且 mf_flow_memory_anti_bearish_modulator 应该削弱看跌意图，而不是使其变得更负。
        bearish_score_raw = (
            (distribution_intensity_norm * distribution_dampener * 0.4 + # 应用衰减器
             upper_shadow_selling_pressure_norm * 0.3 +
             flow_credibility_norm_inverted * 0.3)
        ).clip(0, 1) # 确保原始看跌分数在 [0, 1] 之间

        # V11.1: 应用主力资金累计记忆调节，削弱看跌意图
        bearish_score_modulated = (bearish_score_raw * mf_flow_memory_anti_bearish_modulator).clip(0, 1)
        
        # 转换为负值，表示看跌
        bearish_score = -bearish_score_modulated
        _temp_debug_values["看跌意图"] = {
            "distribution_dampener": distribution_dampener, # 增加衰减器调试输出
            "bearish_score_raw": bearish_score_raw, # 增加原始看跌分数的调试输出
            "mf_flow_memory_anti_bearish_modulator": mf_flow_memory_anti_bearish_modulator, # 增加调节器的调试输出
            "bearish_score_modulated": bearish_score_modulated, # 增加调节后的看跌分数的调试输出
            "bearish_score": bearish_score
        }
        return bearish_score

    def _adjudicate_risk(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], dynamic_weights: Dict[str, pd.Series], rally_intent_synthesis_params: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        风险审判模块。
        """
        # 6.1. 派发风险 (Distribution Risk)
        mf_outflow_divergence = mtf_signals['mtf_mf_net_flow'].clip(upper=0).abs()
        distribution_risk_components = {
            "distribution_intensity_norm": normalized_signals['distribution_intensity_norm'],
            "mtf_upper_shadow_pressure": mtf_signals['mtf_upper_shadow_pressure'],
            "mf_outflow_divergence": mf_outflow_divergence,
            "mtf_retail_fomo": mtf_signals['mtf_retail_fomo'],
            "mtf_covert_distribution": mtf_signals['mtf_covert_distribution'],
            "mtf_deception_lure_short": mtf_signals['mtf_deception_lure_short'],
            "mtf_rally_distribution_pressure": mtf_signals['mtf_rally_distribution_pressure'],
            "mtf_exhaustion_risk": mtf_signals['mtf_exhaustion_risk'],
            "mtf_asymmetric_friction": mtf_signals['mtf_asymmetric_friction'],
            "mtf_volatility_expansion": mtf_signals['mtf_volatility_expansion'],
            "mtf_loser_concentration_90pct": mtf_signals['mtf_loser_concentration_90pct'],
            "mtf_winner_loser_momentum_negative": mtf_signals['mtf_winner_loser_momentum'].clip(upper=0).abs(), # 赢家动量减弱，输家动量增强
            "mtf_sell_flow_efficiency": mtf_signals['mtf_sell_flow_efficiency'], # 卖方效率高，派发风险高
            "mtf_sell_sweep_intensity": mtf_signals['mtf_sell_sweep_intensity'], # 卖出扫单强度高，派发风险高
            "mtf_peak_mass_transfer_rate": mtf_signals['mtf_peak_mass_transfer_rate'], # 筹码转移率高，派发风险高
            "mtf_main_force_on_peak_sell_flow": mtf_signals['mtf_main_force_on_peak_sell_flow'],
            "mtf_main_force_t0_sell_efficiency": mtf_signals['mtf_main_force_t0_sell_efficiency'],
            "mtf_main_force_vwap_down_guidance": mtf_signals['mtf_main_force_vwap_down_guidance'].clip(upper=0).abs(),
            "mtf_market_impact_cost": mtf_signals['mtf_market_impact_cost'],
            "mtf_opening_sell_strength": mtf_signals['mtf_opening_sell_strength'],
            "mtf_total_sell_amount_calibrated": mtf_signals['mtf_total_sell_amount_calibrated'],
            "mtf_wash_trade_intensity": mtf_signals['mtf_wash_trade_intensity'],
            "mtf_winner_profit_margin_avg_negative": mtf_signals['mtf_winner_profit_margin_avg'].clip(upper=0).abs(),
            "mtf_loser_loss_margin_avg_positive": mtf_signals['mtf_loser_loss_margin_avg'].clip(lower=0),
            "mtf_total_winner_rate_inverted": (1 - mtf_signals['mtf_total_winner_rate']),
            "mtf_total_loser_rate": mtf_signals['mtf_total_loser_rate'],
            "mtf_deception_index": mtf_signals['mtf_deception_index'],
            "mtf_deception_lure_long_intensity": mtf_signals['mtf_deception_lure_long_intensity'],
            "mtf_panic_sell_volume_contribution": mtf_signals['mtf_panic_sell_volume_contribution'],
            "mtf_panic_selling_cascade": mtf_signals['mtf_panic_selling_cascade'],
            "mtf_price_reversion_velocity_negative": mtf_signals['mtf_price_reversion_velocity'].clip(upper=0).abs(),
            "mtf_pullback_depth_ratio": mtf_signals['mtf_pullback_depth_ratio'],
            "mtf_risk_reward_profile_negative": mtf_signals['mtf_risk_reward_profile'].clip(upper=0).abs(),
            "mtf_shock_conviction_score": mtf_signals['mtf_shock_conviction_score'],
            "mtf_structural_entropy_change_positive": mtf_signals['mtf_structural_entropy_change'].clip(lower=0),
            "mtf_structural_leverage": mtf_signals['mtf_structural_leverage'],
            "mtf_structural_node_count": mtf_signals['mtf_structural_node_count'],
            "mtf_structural_potential_score": mtf_signals['mtf_structural_potential_score'],
            "mtf_supportive_distribution_intensity": mtf_signals['mtf_supportive_distribution_intensity'],
            "mtf_trend_asymmetry_index_negative": mtf_signals['mtf_trend_asymmetry_index'].clip(upper=0).abs(),
            "mtf_value_area_migration_negative": mtf_signals['mtf_value_area_migration'].clip(upper=0).abs(),
            "mtf_volatility_asymmetry_index_positive": mtf_signals['mtf_volatility_asymmetry_index'].clip(lower=0),
            "mtf_volume_burstiness_index": mtf_signals['mtf_volume_burstiness_index'],
            "mtf_volume_structure_skew_positive": mtf_signals['mtf_volume_structure_skew'].clip(lower=0),
            "mtf_vpin_score": mtf_signals['mtf_vpin_score'],
            "mtf_vwap_cross_down_intensity": mtf_signals['mtf_vwap_cross_down_intensity'],
            "mtf_vwap_sell_control_strength": mtf_signals['mtf_vwap_sell_control_strength'],
            "mtf_absorption_of_distribution_intensity_inverted": (1 - mtf_signals['mtf_absorption_of_distribution_intensity']) # 新增：派发吸收强度反向，承接越强，风险越低
        }
        distribution_risk_weights = {
            "distribution_intensity_norm": 0.03, "mtf_upper_shadow_pressure": 0.03,
            "mf_outflow_divergence": 0.03, "mtf_retail_fomo": 0.02,
            "mtf_covert_distribution": 0.03, "mtf_deception_lure_short": 0.03,
            "mtf_rally_distribution_pressure": 0.03, "mtf_exhaustion_risk": 0.02,
            "mtf_asymmetric_friction": 0.02, "mtf_volatility_expansion": 0.02,
            "mtf_loser_concentration_90pct": 0.02, "mtf_winner_loser_momentum_negative": 0.02,
            "mtf_sell_flow_efficiency": 0.02, "mtf_sell_sweep_intensity": 0.02,
            "mtf_peak_mass_transfer_rate": 0.02, "mtf_main_force_on_peak_sell_flow": 0.02,
            "mtf_main_force_t0_sell_efficiency": 0.02, "mtf_main_force_vwap_down_guidance": 0.02,
            "mtf_market_impact_cost": 0.02, "mtf_opening_sell_strength": 0.02,
            "mtf_total_sell_amount_calibrated": 0.02, "mtf_wash_trade_intensity": 0.02,
            "mtf_winner_profit_margin_avg_negative": 0.02, "mtf_loser_loss_margin_avg_positive": 0.02,
            "mtf_total_winner_rate_inverted": 0.02, "mtf_total_loser_rate": 0.02,
            "mtf_deception_index": 0.02, "mtf_deception_lure_long_intensity": 0.02,
            "mtf_panic_sell_volume_contribution": 0.02, "mtf_panic_selling_cascade": 0.02,
            "mtf_price_reversion_velocity_negative": 0.02, "mtf_pullback_depth_ratio": 0.02,
            "mtf_risk_reward_profile_negative": 0.02, "mtf_shock_conviction_score": 0.02,
            "mtf_structural_entropy_change_positive": 0.02, "mtf_structural_leverage": 0.02,
            "mtf_structural_node_count": 0.02, "mtf_structural_potential_score": 0.02,
            "mtf_supportive_distribution_intensity": 0.02, "mtf_trend_asymmetry_index_negative": 0.02,
            "mtf_value_area_migration_negative": 0.02, "mtf_volatility_asymmetry_index_positive": 0.02,
            "mtf_volume_burstiness_index": 0.02, "mtf_volume_structure_skew_positive": 0.02,
            "mtf_vpin_score": 0.02, "mtf_vwap_cross_down_intensity": 0.02,
            "mtf_vwap_sell_control_strength": 0.02,
            "mtf_absorption_of_distribution_intensity_inverted": 0.05 # 提高权重，从0.03调整为0.05
        }
        distribution_risk_score = _robust_geometric_mean(distribution_risk_components, distribution_risk_weights, df_index).clip(0, 1)
        _temp_debug_values["派发风险"] = {
            "distribution_risk_score": distribution_risk_score
        }
        # 6.2. 前置下跌风险 (Pre-Drop Risk) - 深度情境感知
        pre_5day_pct_change = raw_signals['close_price'].pct_change(periods=5).shift(1).fillna(0)
        pre_13day_pct_change = raw_signals['close_price'].pct_change(periods=13).shift(1).fillna(0)
        norm_pre_drop_5d = self.helper._normalize_series(pre_5day_pct_change.clip(upper=0).abs(), df_index, bipolar=False)
        norm_pre_drop_13d = self.helper._normalize_series(pre_13day_pct_change.clip(upper=0).abs(), df_index, bipolar=False)
        single_day_drop_risk = self.helper._normalize_series(raw_signals['prev_day_pct_change'].clip(upper=0).abs(), df_index, bipolar=False)
        norm_slope_21_neg = self.helper._normalize_series(raw_signals['slope_21_close'].clip(upper=0).abs(), df_index, bipolar=False)
        norm_accel_21_neg = self.helper._normalize_series(raw_signals['accel_21_close'].clip(upper=0).abs(), df_index, bipolar=False)
        norm_slope_34_neg = self.helper._normalize_series(raw_signals['slope_34_close'].clip(upper=0).abs(), df_index, bipolar=False)
        norm_accel_34_neg = self.helper._normalize_series(raw_signals['accel_34_close'].clip(upper=0).abs(), df_index, bipolar=False)
        medium_term_downtrend_strength = (norm_slope_21_neg * 0.3 + norm_accel_21_neg * 0.2 + 
                                          norm_slope_34_neg * 0.3 + norm_accel_34_neg * 0.2).clip(0, 1)
        high_21d = raw_signals['close_price'].rolling(window=21).max()
        fall_from_peak_21d = (1 - raw_signals['close_price'] / high_21d).clip(lower=0).fillna(0)
        norm_fall_from_peak_21d = self.helper._normalize_series(fall_from_peak_21d, df_index, bipolar=False)
        pre_drop_risk_components = {
            "single_day_drop_risk": single_day_drop_risk,
            "norm_pre_drop_5d": norm_pre_drop_5d,
            "norm_pre_drop_13d": norm_pre_drop_13d,
            "medium_term_downtrend_strength": medium_term_downtrend_strength,
            "norm_fall_from_peak_21d": norm_fall_from_peak_21d,
            "mtf_price_thrust_divergence_negative": mtf_signals['mtf_price_thrust_divergence'].clip(upper=0).abs(), # 价格推力负向
            "mtf_trend_efficiency_ratio_inverted": (1 - mtf_signals['mtf_trend_efficiency_ratio']), # 趋势效率低，风险高
            "mtf_loser_concentration_90pct": mtf_signals['mtf_loser_concentration_90pct'], # 输家集中度高，风险高
            "mtf_main_force_vwap_down_guidance_negative": mtf_signals['mtf_main_force_vwap_down_guidance'].clip(upper=0).abs(),
            "mtf_platform_high": mtf_signals['mtf_platform_high'],
            "mtf_platform_low_inverted": (1 - mtf_signals['mtf_platform_low']),
            "mtf_pullback_depth_ratio": mtf_signals['mtf_pullback_depth_ratio'],
            "mtf_reversal_conviction_rate_inverted": (1 - mtf_signals['mtf_reversal_conviction_rate']),
            "mtf_reversal_recovery_rate_inverted": (1 - mtf_signals['mtf_reversal_recovery_rate']),
            "mtf_risk_reward_profile_negative": mtf_signals['mtf_risk_reward_profile'].clip(upper=0).abs(),
            "mtf_shock_conviction_score": mtf_signals['mtf_shock_conviction_score'],
            "mtf_structural_entropy_change_negative": mtf_signals['mtf_structural_entropy_change'].clip(upper=0).abs(),
            "mtf_structural_leverage": mtf_signals['mtf_structural_leverage'],
            "mtf_structural_node_count": mtf_signals['mtf_structural_node_count'],
            "mtf_structural_potential_score": mtf_signals['mtf_structural_potential_score'],
            "mtf_trend_alignment_index_inverted": (1 - mtf_signals['mtf_trend_alignment_index']),
            "mtf_trend_asymmetry_index_negative": mtf_signals['mtf_trend_asymmetry_index'].clip(upper=0).abs(),
            "mtf_vwap_cross_down_intensity": mtf_signals['mtf_vwap_cross_down_intensity'],
            "mtf_vwap_mean_reversion_corr_negative": mtf_signals['mtf_vwap_mean_reversion_corr'].clip(upper=0).abs()
        }
        pre_drop_risk_weights = {
            "single_day_drop_risk": 0.08, "norm_pre_drop_5d": 0.08, "norm_pre_drop_13d": 0.05,
            "medium_term_downtrend_strength": 0.15, "norm_fall_from_peak_21d": 0.06,
            "mtf_price_thrust_divergence_negative": 0.06, "mtf_trend_efficiency_ratio_inverted": 0.05,
            "mtf_loser_concentration_90pct": 0.05, "mtf_main_force_vwap_down_guidance_negative": 0.05,
            "mtf_platform_high": 0.04, "mtf_platform_low_inverted": 0.04,
            "mtf_pullback_depth_ratio": 0.04, "mtf_reversal_conviction_rate_inverted": 0.04,
            "mtf_reversal_recovery_rate_inverted": 0.04, "mtf_risk_reward_profile_negative": 0.04,
            "mtf_shock_conviction_score": 0.04, # 保持冲击信念的风险贡献，但后续会调整其在上涨行情中的解读
            "mtf_structural_entropy_change_negative": 0.04, "mtf_structural_leverage": 0.04,
            "mtf_structural_node_count": 0.04, "mtf_structural_potential_score": 0.04,
            "mtf_trend_alignment_index_inverted": 0.04, "mtf_trend_asymmetry_index_negative": 0.04,
            "mtf_vwap_cross_down_intensity": 0.04,
            "mtf_vwap_mean_reversion_corr_negative": 0.04
        }
        pre_drop_risk_factor = _robust_geometric_mean(pre_drop_risk_components, pre_drop_risk_weights, df_index).clip(0, 1) * 0.7 # 整体风险因子权重
        _temp_debug_values["前置下跌风险"] = {
            "pre_5day_pct_change": pre_5day_pct_change,
            "pre_13day_pct_change": pre_13day_pct_change,
            "norm_pre_drop_5d": norm_pre_drop_5d,
            "norm_pre_drop_13d": norm_pre_drop_13d,
            "single_day_drop_risk": single_day_drop_risk,
            "norm_slope_21_neg": norm_slope_21_neg,
            "norm_accel_21_neg": norm_accel_21_neg,
            "norm_slope_34_neg": norm_slope_34_neg,
            "norm_accel_34_neg": norm_accel_34_neg,
            "medium_term_downtrend_strength": medium_term_downtrend_strength,
            "high_21d": high_21d,
            "fall_from_peak_21d": fall_from_peak_21d,
            "norm_fall_from_peak_21d": norm_fall_from_peak_21d,
            "pre_drop_risk_factor": pre_drop_risk_factor
        }
        # 6.3. 综合风险惩罚因子 - V10.0 非线性惩罚优化
        # 风险权重根据动态权重调整
        risk_sensitivity = get_param_value(rally_intent_synthesis_params.get('risk_sensitivity'), 5.0)
        sigmoid_center = get_param_value(rally_intent_synthesis_params.get('sigmoid_center'), 0.3)
        total_risk_penalty_raw = (distribution_risk_score * dynamic_weights["risk"] + pre_drop_risk_factor * (1 - dynamic_weights["risk"])).clip(0, 1)
        # 应用Sigmoid函数进行非线性惩罚
        # V11.3: 进一步调整 risk_sensitivity 和 sigmoid_center
        total_risk_penalty = 1 / (1 + np.exp(risk_sensitivity * (total_risk_penalty_raw - sigmoid_center)))
        # 归一化为惩罚因子，高风险对应高惩罚 (1-sigmoid_output)
        total_risk_penalty = (1 - total_risk_penalty).clip(0, 1)
        _temp_debug_values["综合风险惩罚因子"] = {
            "total_risk_penalty_raw": total_risk_penalty_raw,
            "total_risk_penalty": total_risk_penalty
        }
        return total_risk_penalty

    def _apply_contextual_modulators(self, df_index: pd.Index, final_rally_intent: pd.Series, proxy_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        应用情境调节器。
        """
        # rs_modulator 和 capital_modulator 已在上方使用代理信号计算
        # V10.0 情境调节器 (基于原始MTF信号)
        market_sentiment_modulator = (1 + mtf_signals['mtf_market_sentiment'] * 0.1) # 市场情绪越好，意图越强
        structural_tension_modulator = (1 - mtf_signals['mtf_structural_tension'] * 0.1) # 结构张力越大，意图越弱
        trend_vitality_modulator = (1 + mtf_signals['mtf_trend_vitality'] * 0.1) # 趋势活力越强，意图越强
        liquidity_authenticity_modulator = (1 + mtf_signals['mtf_liquidity_authenticity'] * 0.05) # 流动性越真实，意图越强
        # V10.0 行为心理调节器 (基于原始MTF信号)
        character_score_modulator = (1 + mtf_signals['mtf_character_score'] * 0.05)
        signal_conviction_modulator = (1 + mtf_signals['mtf_signal_conviction_score'] * 0.05)
        touch_conviction_modulator = (1 + mtf_signals['mtf_touch_conviction_score'] * 0.05)
        gathering_by_chasing_modulator = (1 + mtf_signals['mtf_gathering_by_chasing'] * 0.05)
        gathering_by_support_modulator = (1 + mtf_signals['mtf_gathering_by_support'] * 0.05)
        # V10.0 微观结构调节器 (基于原始MTF信号)
        microstructure_efficiency_modulator = (1 + mtf_signals['mtf_microstructure_efficiency'] * 0.05)
        imbalance_effectiveness_modulator = (1 + mtf_signals['mtf_imbalance_effectiveness'] * 0.05)
        auction_intent_modulator = (1 + mtf_signals['mtf_auction_intent_signal'] * 0.05) # 集合竞价意图
        final_rally_intent = (final_rally_intent * proxy_signals['rs_modulator'] * proxy_signals['capital_modulator'] *
                              market_sentiment_modulator * structural_tension_modulator *
                              trend_vitality_modulator * liquidity_authenticity_modulator *
                              character_score_modulator * signal_conviction_modulator *
                              touch_conviction_modulator * gathering_by_chasing_modulator *
                              gathering_by_support_modulator * microstructure_efficiency_modulator *
                              imbalance_effectiveness_modulator * auction_intent_modulator).clip(-1, 1)
        _temp_debug_values["相对强度和资本属性调节"] = {
            "rs_modulator": proxy_signals['rs_modulator'],
            "capital_modulator": proxy_signals['capital_modulator'],
            "market_sentiment_modulator": market_sentiment_modulator,
            "structural_tension_modulator": structural_tension_modulator,
            "trend_vitality_modulator": trend_vitality_modulator,
            "liquidity_authenticity_modulator": liquidity_authenticity_modulator,
            "character_score_modulator": character_score_modulator,
            "signal_conviction_modulator": signal_conviction_modulator,
            "touch_conviction_modulator": touch_conviction_modulator,
            "gathering_by_chasing_modulator": gathering_by_chasing_modulator,
            "gathering_by_support_modulator": gathering_by_support_modulator,
            "microstructure_efficiency_modulator": microstructure_efficiency_modulator,
            "imbalance_effectiveness_modulator": imbalance_effectiveness_modulator,
            "auction_intent_modulator": auction_intent_modulator,
            "final_rally_intent": final_rally_intent
        }
        return final_rally_intent

    def _output_debug_info(self, is_debug_enabled_for_method: bool, probe_ts: Optional[pd.Timestamp], debug_output: Dict, _temp_debug_values: Dict, final_rally_intent: pd.Series, method_name: str):
        """
        统一输出调试信息。
        """
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
            for sig_name, series in _temp_debug_values["原始信号值"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        '{sig_name}': {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 派生信号值 ---"] = ""
            for sig_name, series in _temp_debug_values["派生信号值"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        '{sig_name}': {val}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- MTF融合信号 ---"] = ""
            for key, series in _temp_debug_values["MTF融合信号"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 归一化处理 ---"] = ""
            for key, series in _temp_debug_values["归一化处理"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 代理信号 ---"] = ""
            for key, series in _temp_debug_values["代理信号"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 动态权重 ---"] = ""
            for key, series in _temp_debug_values["动态权重"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 攻击性 ---"] = ""
            for key, series in _temp_debug_values["攻击性"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 控制力 ---"] = ""
            for key, series in _temp_debug_values["控制力"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 障碍清除 ---"] = ""
            for key, series in _temp_debug_values["障碍清除"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 基础看涨意图 ---"] = ""
            for key, value in _temp_debug_values["基础看涨意图"].items(): # 这里的value可能是float
                if isinstance(value, pd.Series):
                    val = value.loc[probe_ts] if probe_ts in value.index else np.nan
                    debug_output[f"        {key}: {val:.4f}"] = ""
                else: # 如果是标量，直接输出
                    debug_output[f"        {key}: {value:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 看跌意图 ---"] = ""
            for key, series in _temp_debug_values["看跌意图"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 派发风险 ---"] = ""
            # 增加对 distribution_risk_components 的调试输出
            if "distribution_risk_components_debug" in _temp_debug_values:
                debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 派发风险组件 ---"] = ""
                for comp_name, comp_series in _temp_debug_values["distribution_risk_components_debug"].items():
                    val = comp_series.loc[probe_ts] if probe_ts in comp_series.index else np.nan
                    debug_output[f"        {comp_name}: {val:.4f}"] = ""
            for key, series in _temp_debug_values["派发风险"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 前置下跌风险 ---"] = ""
            for key, series in _temp_debug_values["前置下跌风险"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 综合风险惩罚因子 ---"] = ""
            for key, series in _temp_debug_values["综合风险惩罚因子"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 最终意图合成 ---"] = ""
            for key, series in _temp_debug_values["最终意图合成"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 相对强度和资本属性调节 ---"] = ""
            for key, series in _temp_debug_values["相对强度和资本属性调节"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 主力拉升意图诊断完成，最终分值: {final_rally_intent.loc[probe_ts]:.4f}"] = ""
            for key, value in debug_output.items():
                if value:
                    print(f"{key}: {value}")
                else:
                    print(key)

    def _get_required_signals_list(self, mtf_slope_accel_weights: Dict, historical_context_params: Dict) -> List[str]:
        """
        根据配置动态生成所需的信号列表。
        """
        long_term_trend_slope_period = get_param_value(historical_context_params.get('long_term_trend_slope_period'), 21)
        required_signals = [
            'pct_change_D', 'main_force_net_flow_calibrated_D', 'main_force_slippage_index_D',
            'upward_impulse_purity_D', 'volume_ratio_D', 'control_solidity_index_D',
            'main_force_cost_advantage_D', 'SLOPE_5_winner_concentration_90pct_D',
            'dominant_peak_solidity_D', 'active_buying_support_D', 'pressure_rejection_strength_D',
            'profit_realization_quality_D', 
            'distribution_at_peak_intensity_D', 'upper_shadow_selling_pressure_D', 
            'flow_credibility_index_D', 'chip_health_score_D', 'retail_fomo_premium_index_D',
            'SLOPE_21_close_D', 'ACCEL_21_close_D', 'SLOPE_34_close_D', 'ACCEL_34_close_D',
            'buy_sweep_intensity_D', 'main_force_buy_ofi_D', 'main_force_t0_buy_efficiency_D',
            'order_book_imbalance_D', 'micro_price_impact_asymmetry_D', 'constructive_turnover_ratio_D',
            'upward_impulse_strength_D', 'vwap_buy_control_strength_D', 'mf_cost_zone_buy_intent_D',
            'chip_fault_blockage_ratio_D', 'vacuum_traversal_efficiency_D', 'vacuum_zone_magnitude_D',
            'dip_buy_absorption_strength_D', 'rally_buy_support_weakness_D', 'covert_distribution_signal_D',
            'deception_lure_short_intensity_D', 'rally_distribution_pressure_D', 'exhaustion_risk_index_D',
            'asymmetric_friction_index_D', 'volatility_expansion_ratio_D', 'market_sentiment_score_D',
            'structural_tension_index_D', 'trend_vitality_index_D', 'liquidity_authenticity_score_D',
            'order_book_clearing_rate_D', 'sell_sweep_intensity_D', 'main_force_flow_gini_D',
            'microstructure_efficiency_index_D', 'imbalance_effectiveness_D', 'auction_showdown_score_D',
            'closing_conviction_score_D', 'intraday_energy_density_D', 'intraday_thrust_purity_D',
            'price_thrust_divergence_D', 'trend_efficiency_ratio_D', 'loser_concentration_90pct_D',
            'winner_loser_momentum_D', 'cost_structure_skewness_D', 'cost_gini_coefficient_D',
            'mf_vpoc_premium_D', 'character_score_D', 'signal_conviction_score_D',
            'touch_conviction_score_D', 'gathering_by_chasing_D', 'gathering_by_support_D',
            'VOLATILITY_INSTABILITY_INDEX_21d_D', 'ADX_14_D', # 用于动态权重
            'buy_flow_efficiency_index_D', 'sell_flow_efficiency_index_D', # 订单流效率
            'auction_closing_position_D', 'auction_impact_score_D', 'auction_intent_signal_D', # 集合竞价
            'order_book_liquidity_supply_D', 'liquidity_slope_D', # 流动性深度
            'peak_mass_transfer_rate_D', # 筹码转移
            'mf_cost_zone_defense_intent_D', # 成本区防守
            'bid_side_liquidity_D', 'ask_side_liquidity_D', # 订单簿流动性
            'retail_panic_surrender_index_D', # 零售恐慌投降
            'main_force_activity_ratio_D', 'main_force_conviction_index_D', # 主力活跃度与信念
            'main_force_execution_alpha_D', 'main_force_flow_directionality_D', # 主力执行效率与流向
            'main_force_on_peak_buy_flow_D', 'main_force_on_peak_sell_flow_D', # 主力峰值买卖流
            'main_force_t0_efficiency_D', 'main_force_t0_sell_efficiency_D', # 主力T0效率
            'main_force_vwap_down_guidance_D', 'main_force_vwap_up_guidance_D', # 主力VWAP引导
            'market_impact_cost_D', # 市场冲击成本
            'opening_buy_strength_D', 'opening_sell_strength_D', # 开盘强度
            'closing_strength_index_D', # 收盘强度
            'total_buy_amount_calibrated_D', 'total_sell_amount_calibrated_D', # 总买卖金额
            'wash_trade_intensity_D', # 洗盘强度
            'winner_profit_margin_avg_D', 'loser_loss_margin_avg_D', # 赢家利润，输家亏损
            'total_winner_rate_D', 'total_loser_rate_D', # 赢家输家比例
            'impulse_quality_ratio_D', 'thrust_efficiency_score_D', # 脉冲质量与推力效率
            'platform_conviction_score_D', 'platform_high_D', 'platform_low_D', # 平台信念与高低
            'breakout_quality_score_D', 'breakout_readiness_score_D', 'breakthrough_conviction_score_D', # 突破相关
            'defense_solidity_score_D', 'support_validation_strength_D', # 防守与支撑
            'covert_accumulation_signal_D', 'suppressive_accumulation_intensity_D', # 隐蔽吸筹
            'deception_index_D', 'deception_lure_long_intensity_D', # 欺骗
            'equilibrium_compression_index_D', 'final_charge_intensity_D', # 均衡压缩与最终冲刺
            'floating_chip_cleansing_efficiency_D', # 浮筹清洗效率
            'hidden_accumulation_intensity_D', 'internal_accumulation_intensity_D', # 隐藏吸筹
            'intraday_posture_score_D', 'opening_gap_defense_strength_D', # 日内姿态
            'panic_buy_absorption_contribution_D', 'panic_sell_volume_contribution_D', 'panic_selling_cascade_D', # 恐慌
            'peak_control_transfer_D', 'peak_separation_ratio_D', # 峰值控制
            'price_reversion_velocity_D', 'pullback_depth_ratio_D', # 价格回归与回调
            'quality_score_D', 'reversal_conviction_rate_D', 'reversal_power_index_D', 'reversal_recovery_rate_D', # 反转
            'risk_reward_profile_D', 'shock_conviction_score_D', # 风险回报与冲击
            'strategic_phase_score_D', 'structural_entropy_change_D', 'structural_leverage_D', 'structural_node_count_D', 'structural_potential_score_D', # 结构
            'support_validation_score_D', 'supportive_distribution_intensity_D', # 支撑与派发
            'trend_acceleration_score_D', 'trend_alignment_index_D', 'trend_asymmetry_index_D', 'trend_conviction_score_D', # 趋势
            'value_area_migration_D', 'value_area_overlap_pct_D', # 价值区域
            'volatility_asymmetry_index_D', 'volume_burstiness_index_D', 'volume_structure_skew_D', # 波动率与成交量
            'vpin_score_D', 'vwap_control_strength_D', 'vwap_cross_down_intensity_D', 'vwap_cross_up_intensity_D', 'vwap_crossing_intensity_D', 'vwap_mean_reversion_corr_D', 'vwap_sell_control_strength_D', # VWAP
            'winner_stability_index_D', # 赢家稳定性
            'winner_concentration_90pct_D', # 用于筹码集中度稳定性
            f'SLOPE_{long_term_trend_slope_period}_close_D', # 长期趋势斜率，现在是21日
            'absorption_of_distribution_intensity_D' # 新增：派发吸收强度
        ]
        # 动态添加MTF斜率和加速度信号到required_signals
        base_signals_for_mtf = [s.replace('_D', '') for s in required_signals if not s.startswith(('SLOPE_', 'ACCEL_')) and s.endswith('_D')]
        for base_sig_name in base_signals_for_mtf:
            for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
                required_signals.append(f'SLOPE_{period_str}_{base_sig_name}_D')
            for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
                required_signals.append(f'ACCEL_{period_str}_{base_sig_name}_D')
        return required_signals

