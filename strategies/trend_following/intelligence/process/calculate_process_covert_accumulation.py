# strategies\trend_following\intelligence\process\calculate_process_covert_accumulation.py
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

class CalculateProcessCovertAccumulation:
    def __init__(self, strategy_instance, helper_instance: ProcessIntelligenceHelper):
            self.strategy = strategy_instance
            self.helper = helper_instance

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.11 · 信号丰富与共振机制版】计算“隐蔽吸筹”的专属信号。
        - 核心升级: 引入新的原始信号，新增主力吸筹共振机制，提供更全面和协同的隐蔽吸筹判断。
        """
        method_name = "_calculate_process_covert_accumulation"
        # --- 调试信息构建 ---
        is_debug_enabled_for_method = get_param_value(self.helper.debug_params.get('enabled'), False) and get_param_value(self.helper.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug_enabled_for_method and self.helper.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.helper.probe_dates]
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
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算隐蔽吸筹..."] = ""
        
        # 1. 获取配置参数
        fusion_weights, market_context_weights, covert_action_weights, chip_optimization_weights, \
        price_weakness_slope_window, low_volatility_bbw_window, mtf_slope_accel_weights, \
        neutral_range_threshold, cumulative_flow_windows, cumulative_flow_weights, \
        cumulative_acc_windows, cumulative_acc_weights, \
        daily_mf_flow_weight, cumulative_mf_flow_weight, daily_acc_weight, cumulative_acc_weight, \
        new_raw_signals_weights, main_force_accumulation_resonance_weight = self._get_covert_accumulation_config(config)
        
        df_index = df.index

        # 2. 校验并获取原始信号 (此方法内部会先计算所有派生信号)
        raw_signals = self._validate_and_get_raw_signals(df, method_name, price_weakness_slope_window, low_volatility_bbw_window, mtf_slope_accel_weights, is_debug_enabled_for_method, probe_ts, _temp_debug_values, cumulative_flow_windows, cumulative_acc_windows)
        if raw_signals is None:
            return pd.Series(0.0, index=df.index)

        # 3. 计算维度一：市场背景 (Market Context)
        market_context_score = self._calculate_market_context_score(df, df_index, raw_signals, mtf_slope_accel_weights, market_context_weights, price_weakness_slope_window, low_volatility_bbw_window, method_name, _temp_debug_values, neutral_range_threshold)

        # 4. 计算维度二：隐蔽行动 (Covert Action)
        covert_action_score = self._calculate_covert_action_score(df, df_index, raw_signals, mtf_slope_accel_weights, covert_action_weights, method_name, _temp_debug_values, cumulative_flow_windows, cumulative_flow_weights, cumulative_acc_windows, cumulative_acc_weights, daily_mf_flow_weight, cumulative_mf_flow_weight, daily_acc_weight, cumulative_acc_weight, new_raw_signals_weights, main_force_accumulation_resonance_weight)

        # 5. 计算维度三：筹码优化 (Chip Optimization)
        chip_optimization_score = self._calculate_chip_optimization_score(df, df_index, raw_signals, mtf_slope_accel_weights, chip_optimization_weights, method_name, _temp_debug_values)
        
        # 6. 最终合成：三维融合
        final_score = self._fuse_final_score(df_index, market_context_score, covert_action_score, chip_optimization_score, fusion_weights, _temp_debug_values)
        _temp_debug_values["final_score"] = final_score # 存储最终分数用于调试输出

        # --- 统一输出调试信息 ---
        if is_debug_enabled_for_method and probe_ts:
            self._print_debug_info(debug_output, _temp_debug_values, method_name, probe_ts)
            
        return final_score

    def _get_covert_accumulation_config(self, config: Dict) -> Tuple[Dict, Dict, Dict, Dict, int, int, Dict, float, List[int], Dict, List[int], Dict, float, float, float, float, Dict, float]:
        """
        【V2.11 · 信号丰富与共振机制版】获取隐蔽吸筹计算所需的所有配置参数。
        - 核心修改: 引入新的原始信号权重，新增主力吸筹共振信号权重，并更新隐蔽行动的默认权重。
        """
        covert_accum_params = get_param_value(self.helper.params.get('covert_accumulation_params'), {})
        fusion_weights = get_param_value(covert_accum_params.get('fusion_weights'), {"market_context": 0.3, "covert_action": 0.4, "chip_optimization": 0.3})
        market_context_weights = get_param_value(covert_accum_params.get('market_context_weights'), {
            "retail_panic": 0.1, "price_weakness": 0.1, "low_volatility": 0.1,
            "sentiment_pendulum_inverted": 0.1, "tension_inverted": 0.1, "market_sentiment_inverted": 0.05,
            "volatility_instability_inverted": 0.05, "equilibrium_compression": 0.1,
            "price_volume_entropy_inverted": 0.05, "fractal_dimension_inverted": 0.05,
            "hurst_inverted": 0.05, "is_consolidating": 0.05, "dynamic_consolidation_duration": 0.05,
            "volume_burstiness_inverted": 0.05, "market_impact_cost_inverted": 0.05,
            "liquidity_authenticity": 0.05, "order_book_imbalance_neutral": 0.05,
            "micro_price_impact_asymmetry_neutral": 0.05
        })
        # 新增的原始信号权重
        new_raw_signals_weights = get_param_value(covert_accum_params.get('new_raw_signals_weights'), {
            "ask_side_liquidity_inverted": 0.03,
            "mf_level5_buy_ofi": 0.05,
            "mf_buy_execution_alpha": 0.05,
            "upper_shadow_selling_pressure_inverted": 0.03,
            "smart_money_inst_net_buy": 0.05,
            "microstructure_efficiency": 0.03
        })
        # 主力吸筹共振信号权重
        main_force_accumulation_resonance_weight = get_param_value(covert_accum_params.get('main_force_accumulation_resonance_weight'), 0.1)

        covert_action_weights = get_param_value(covert_accum_params.get('covert_action_weights'), {
            "suppressive_accum": 0.08, "contextualized_main_force_flow": 0.15, "deception_lure_long": 0.1,
            "stealth_ops": 0.1, "contextualized_hidden_accumulation": 0.15, "chip_historical_potential": 0.05,
            "mf_buy_ofi": 0.05, "mf_cost_advantage": 0.05, "mf_flow_slope": 0.05,
            "suppressive_accum_slope": 0.05, "internal_accumulation": 0.05, "gathering_by_support": 0.05,
            "main_force_flow_gini": 0.05,
            "mf_t0_buy_efficiency": 0.05,
            "buy_quote_exhaustion_inverted": 0.05, "bid_side_liquidity": 0.05,
            "net_lg_amount": 0.05, "dip_buy_absorption": 0.05, "mf_slippage_inverted": 0.05,
            "micro_impact_elasticity_inverted": 0.05,
            # 新增信号的默认权重
            "ask_side_liquidity_inverted": new_raw_signals_weights["ask_side_liquidity_inverted"],
            "mf_level5_buy_ofi": new_raw_signals_weights["mf_level5_buy_ofi"],
            "mf_buy_execution_alpha": new_raw_signals_weights["mf_buy_execution_alpha"],
            "upper_shadow_selling_pressure_inverted": new_raw_signals_weights["upper_shadow_selling_pressure_inverted"],
            "smart_money_inst_net_buy": new_raw_signals_weights["smart_money_inst_net_buy"],
            "microstructure_efficiency": new_raw_signals_weights["microstructure_efficiency"],
            "main_force_accumulation_resonance": main_force_accumulation_resonance_weight # 主力吸筹共振信号
        })
        chip_optimization_weights = get_param_value(covert_accum_params.get('chip_optimization_weights'), {
            "chip_fatigue": 0.15, "loser_pain": 0.15, "holder_sentiment_inverted": 0.1,
            "turnover_purity_cost_opt": 0.1, "floating_chip_cleansing": 0.1, "total_loser_rate": 0.05,
            "winner_concentration_inverted": 0.05, "loser_concentration": 0.05,
            "cost_dispersion_inverted": 0.05, "dominant_peak_solidity": 0.05,
            "chip_health": 0.05, "lower_shadow_absorption": 0.05,
            "panic_sell_contribution": 0.05, "profit_realization_inverted": 0.05
        })
        price_weakness_slope_window = get_param_value(covert_accum_params.get('price_weakness_slope_window'), 5)
        low_volatility_bbw_window = get_param_value(covert_accum_params.get('low_volatility_bbw_window'), 21)
        mtf_slope_accel_weights = config.get('mtf_slope_accel_weights', {"slope_periods": {"5": 0.6, "13": 0.4}, "accel_periods": {"5": 0.7, "13": 0.3}})
        neutral_range_threshold = get_param_value(covert_accum_params.get('neutral_range_threshold'), 0.1)

        cumulative_flow_windows = get_param_value(covert_accum_params.get('cumulative_flow_windows'), [13, 21])
        cumulative_flow_weights = get_param_value(covert_accum_params.get('cumulative_flow_weights'), {"13": 0.6, "21": 0.4})
        cumulative_acc_windows = get_param_value(covert_accum_params.get('cumulative_acc_windows'), [13, 21])
        cumulative_acc_weights = get_param_value(covert_accum_params.get('cumulative_acc_weights'), {"13": 0.6, "21": 0.4})

        daily_mf_flow_weight = get_param_value(covert_accum_params.get('daily_mf_flow_weight'), 0.4)
        cumulative_mf_flow_weight = get_param_value(covert_accum_params.get('cumulative_mf_flow_weight'), 0.6)
        daily_acc_weight = get_param_value(covert_accum_params.get('daily_acc_weight'), 0.4)
        cumulative_acc_weight = get_param_value(covert_accum_params.get('cumulative_acc_weight'), 0.6)

        return fusion_weights, market_context_weights, covert_action_weights, chip_optimization_weights, \
               price_weakness_slope_window, low_volatility_bbw_window, mtf_slope_accel_weights, \
               neutral_range_threshold, cumulative_flow_windows, cumulative_flow_weights, \
               cumulative_acc_windows, cumulative_acc_weights, \
               daily_mf_flow_weight, cumulative_mf_flow_weight, daily_acc_weight, cumulative_acc_weight, \
               new_raw_signals_weights, main_force_accumulation_resonance_weight

    def _calculate_derived_signals(self, df: pd.DataFrame, mtf_slope_accel_weights: Dict, cumulative_flow_windows: List[int], cumulative_acc_windows: List[int]):
        """
        【V2.11 · 信号丰富与共振机制版】计算所有派生信号（累积求和、斜率、加速度）并添加到DataFrame中。
        - 核心修改: 增加了新引入原始信号的MTF斜率和加速度计算。
        """
        # 1. 计算累积求和信号
        mf_flow_base = 'main_force_net_flow_calibrated_D'
        hidden_acc_base = 'hidden_accumulation_intensity_D'
        suppressive_acc_base = 'suppressive_accumulation_intensity_D'
        # 确保原始列存在，否则无法计算滚动和
        if mf_flow_base in df.columns:
            for window in cumulative_flow_windows:
                col_name = f'{mf_flow_base}_{window}d_sum'
                if col_name not in df.columns:
                    df[col_name] = df[mf_flow_base].rolling(window, min_periods=1).sum()
        if hidden_acc_base in df.columns:
            for window in cumulative_acc_windows:
                col_name_hidden = f'{hidden_acc_base}_{window}d_sum'
                if col_name_hidden not in df.columns:
                    df[col_name_hidden] = df[hidden_acc_base].rolling(window, min_periods=1).sum()
        if suppressive_acc_base in df.columns:
            for window in cumulative_acc_windows:
                col_name_suppressive = f'{suppressive_acc_base}_{window}d_sum'
                if col_name_suppressive not in df.columns:
                    df[col_name_suppressive] = df[suppressive_acc_base].rolling(window, min_periods=1).sum()

        # 2. 定义所有需要计算MTF斜率和加速度的基准信号
        mtf_base_signals_for_calculation = [
            'close_D', # For price_weakness
            'suppressive_accumulation_intensity_D',
            'main_force_net_flow_calibrated_D',
            'deception_lure_long_intensity_D',
            'hidden_accumulation_intensity_D',
            'main_force_buy_ofi_D',
            'main_force_cost_advantage_D',
            'retail_panic_surrender_index_D',
            'BBW_21_2.0_D',
            'market_sentiment_score_D',
            'VOLATILITY_INSTABILITY_INDEX_21d_D',
            'chip_fatigue_index_D',
            'loser_pain_index_D',
            'floating_chip_cleansing_efficiency_D',
            'total_loser_rate_D',
            'structural_tension_index_D',
            'covert_accumulation_signal_D',
            'structural_potential_score_D',
            'winner_stability_index_D',
            'constructive_turnover_ratio_D',
            'equilibrium_compression_index_D',
            'price_volume_entropy_D',
            'FRACTAL_DIMENSION_89d_D',
            'HURST_144d_D',
            'is_consolidating_D',
            'dynamic_consolidation_duration_D',
            'volume_burstiness_index_D',
            'market_impact_cost_D',
            'liquidity_authenticity_score_D',
            'order_book_imbalance_D',
            'micro_price_impact_asymmetry_D',
            'internal_accumulation_intensity_D',
            'gathering_by_support_D',
            'main_force_flow_gini_D',
            'main_force_t0_buy_efficiency_D',
            'buy_quote_exhaustion_rate_D',
            'bid_side_liquidity_D',
            'net_lg_amount_calibrated_D',
            'dip_buy_absorption_strength_D',
            'main_force_slippage_index_D',
            'micro_impact_elasticity_D',
            'winner_concentration_90pct_D',
            'loser_concentration_90pct_D',
            'cost_dispersion_index_D',
            'dominant_peak_solidity_D',
            'chip_health_score_D',
            'lower_shadow_absorption_strength_D',
            'panic_sell_volume_contribution_D',
            'profit_realization_quality_D',
            # 新增的原始信号
            'ask_side_liquidity_D',
            'main_force_level5_buy_ofi_D',
            'main_force_buy_execution_alpha_D',
            'upper_shadow_selling_pressure_D',
            'SMART_MONEY_INST_NET_BUY_D',
            'microstructure_efficiency_index_D'
        ]
        # 添加累积求和信号到MTF基准信号列表
        for window in cumulative_flow_windows:
            mtf_base_signals_for_calculation.append(f'{mf_flow_base}_{window}d_sum')
        for window in cumulative_acc_windows:
            mtf_base_signals_for_calculation.append(f'{hidden_acc_base}_{window}d_sum')
            mtf_base_signals_for_calculation.append(f'{suppressive_acc_base}_{window}d_sum')

        # 3. 计算所有MTF斜率和加速度信号并添加到df
        for base_sig in mtf_base_signals_for_calculation:
            if base_sig not in df.columns:
                continue # 如果基准信号不存在，则跳过其斜率/加速度计算
            
            # 计算斜率
            for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
                period = int(period_str)
                slope_col_name = f'SLOPE_{period}_{base_sig}'
                if slope_col_name not in df.columns:
                    df[slope_col_name] = ta.slope(df[base_sig], length=period)
            
            # 计算加速度 (斜率的斜率)
            if mtf_slope_accel_weights.get('slope_periods'):
                first_slope_period_str = list(mtf_slope_accel_weights['slope_periods'].keys())[0]
                base_slope_col = f'SLOPE_{first_slope_period_str}_{base_sig}'
                if base_slope_col in df.columns:
                    for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
                        period = int(period_str)
                        accel_col_name = f'ACCEL_{period}_{base_sig}'
                        if accel_col_name not in df.columns:
                            df[accel_col_name] = ta.slope(df[base_slope_col], length=period)
                else:
                    # 如果基准斜率不存在，则加速度也无法计算
                    for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
                        df[f'ACCEL_{period_str}_{base_sig}'] = np.nan

    def _validate_and_get_raw_signals(self, df: pd.DataFrame, method_name: str, price_weakness_slope_window: int, low_volatility_bbw_window: int, mtf_slope_accel_weights: Dict, is_debug_enabled_for_method: bool, probe_ts: Optional[pd.Timestamp], _temp_debug_values: Dict, cumulative_flow_windows: List[int], cumulative_acc_windows: List[int]) -> Optional[Dict[str, pd.Series]]:
        """
        【V2.11 · 信号丰富与共振机制版】校验所需信号并获取所有原始数据。
        - 核心修改: 增加了新引入原始信号的校验和获取。
        """
        # 0. 动态计算所有派生信号 (累积求和、斜率、加速度)
        self._calculate_derived_signals(df, mtf_slope_accel_weights, cumulative_flow_windows, cumulative_acc_windows)

        # 1. 定义所有需要校验的原始信号列名
        required_df_columns = [
            'retail_panic_surrender_index_D', f'SLOPE_{price_weakness_slope_window}_close_D', f'BBW_{low_volatility_bbw_window}_2.0_D',
            'suppressive_accumulation_intensity_D', 'main_force_net_flow_calibrated_D', 'deception_index_D',
            'chip_fatigue_index_D', 'loser_pain_index_D',
            'deception_lure_long_intensity_D', 'deception_lure_short_intensity_D',
            'hidden_accumulation_intensity_D',
            'market_sentiment_score_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D',
            'main_force_buy_ofi_D', 'main_force_cost_advantage_D',
            'floating_chip_cleansing_efficiency_D', 'total_loser_rate_D',
            'structural_tension_index_D',
            'covert_accumulation_signal_D',
            'structural_potential_score_D',
            'winner_stability_index_D',
            'constructive_turnover_ratio_D',
            'equilibrium_compression_index_D',
            'price_volume_entropy_D',
            'FRACTAL_DIMENSION_89d_D',
            'HURST_144d_D',
            'is_consolidating_D',
            'dynamic_consolidation_duration_D',
            'volume_burstiness_index_D',
            'market_impact_cost_D',
            'liquidity_authenticity_score_D',
            'order_book_imbalance_D',
            'micro_price_impact_asymmetry_D',
            'internal_accumulation_intensity_D',
            'gathering_by_support_D',
            'main_force_flow_gini_D',
            'main_force_t0_buy_efficiency_D',
            'buy_quote_exhaustion_rate_D',
            'bid_side_liquidity_D',
            'net_lg_amount_calibrated_D',
            'dip_buy_absorption_strength_D',
            'main_force_slippage_index_D',
            'micro_impact_elasticity_D',
            'winner_concentration_90pct_D',
            'loser_concentration_90pct_D',
            'cost_dispersion_index_D',
            'dominant_peak_solidity_D',
            'chip_health_score_D',
            'lower_shadow_absorption_strength_D',
            'panic_sell_volume_contribution_D',
            'profit_realization_quality_D',
            # 新增的原始信号
            'ask_side_liquidity_D',
            'main_force_level5_buy_ofi_D',
            'main_force_buy_execution_alpha_D',
            'upper_shadow_selling_pressure_D',
            'SMART_MONEY_INST_NET_BUY_D',
            'microstructure_efficiency_index_D'
        ]

        # 添加累积求和信号的列名
        mf_flow_base = 'main_force_net_flow_calibrated_D'
        hidden_acc_base = 'hidden_accumulation_intensity_D'
        suppressive_acc_base = 'suppressive_accumulation_intensity_D'
        for window in cumulative_flow_windows:
            required_df_columns.append(f'{mf_flow_base}_{window}d_sum')
        for window in cumulative_acc_windows:
            required_df_columns.append(f'{hidden_acc_base}_{window}d_sum')
            required_df_columns.append(f'{suppressive_acc_base}_{window}d_sum')

        # 添加所有MTF斜率和加速度信号的列名
        mtf_base_signals_for_required_check = [
            'close_D', # For price_weakness
            'suppressive_accumulation_intensity_D',
            'main_force_net_flow_calibrated_D',
            'deception_lure_long_intensity_D',
            'hidden_accumulation_intensity_D',
            'main_force_buy_ofi_D',
            'main_force_cost_advantage_D',
            'retail_panic_surrender_index_D',
            'BBW_21_2.0_D',
            'market_sentiment_score_D',
            'VOLATILITY_INSTABILITY_INDEX_21d_D',
            'chip_fatigue_index_D',
            'loser_pain_index_D',
            'floating_chip_cleansing_efficiency_D',
            'total_loser_rate_D',
            'structural_tension_index_D',
            'covert_accumulation_signal_D',
            'structural_potential_score_D',
            'winner_stability_index_D',
            'constructive_turnover_ratio_D',
            'equilibrium_compression_index_D',
            'price_volume_entropy_D',
            'FRACTAL_DIMENSION_89d_D',
            'HURST_144d_D',
            'dynamic_consolidation_duration_D',
            'volume_burstiness_index_D',
            'market_impact_cost_D',
            'liquidity_authenticity_score_D',
            'order_book_imbalance_D',
            'micro_price_impact_asymmetry_D',
            'internal_accumulation_intensity_D',
            'gathering_by_support_D',
            'main_force_flow_gini_D',
            'main_force_t0_buy_efficiency_D',
            'buy_quote_exhaustion_rate_D',
            'bid_side_liquidity_D',
            'net_lg_amount_calibrated_D',
            'dip_buy_absorption_strength_D',
            'main_force_slippage_index_D',
            'micro_impact_elasticity_D',
            'winner_concentration_90pct_D',
            'loser_concentration_90pct_D',
            'cost_dispersion_index_D',
            'dominant_peak_solidity_D',
            'chip_health_score_D',
            'lower_shadow_absorption_strength_D',
            'panic_sell_volume_contribution_D',
            'profit_realization_quality_D',
            # 新增的原始信号
            'ask_side_liquidity_D',
            'main_force_level5_buy_ofi_D',
            'main_force_buy_execution_alpha_D',
            'upper_shadow_selling_pressure_D',
            'SMART_MONEY_INST_NET_BUY_D',
            'microstructure_efficiency_index_D'
        ]
        for base_sig in mtf_base_signals_for_required_check:
            for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
                required_df_columns.append(f'SLOPE_{period_str}_{base_sig}')
            for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
                required_df_columns.append(f'ACCEL_{period_str}_{base_sig}')
        
        all_required_signals = required_df_columns
        if not self.helper._validate_required_signals(df, all_required_signals, method_name):
            if is_debug_enabled_for_method and probe_ts:
                debug_output = {f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。": ""}
                self.helper._print_debug_output(debug_output)
            return None

        # 2. 获取所有原始信号 (包括新计算的累积求和信号)
        raw_signals = {
            'retail_panic_raw': self.helper._get_safe_series(df, 'retail_panic_surrender_index_D', 0.0, method_name=method_name),
            'price_weakness_slope_raw': self.helper._get_safe_series(df, f'SLOPE_{price_weakness_slope_window}_close_D', 0.0, method_name=method_name),
            'bbw_raw': self.helper._get_safe_series(df, f'BBW_{low_volatility_bbw_window}_2.0_D', 0.0, method_name=method_name),
            'suppressive_accum_raw': self.helper._get_safe_series(df, 'suppressive_accumulation_intensity_D', 0.0, method_name=method_name),
            'main_force_flow_raw': self.helper._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name=method_name),
            'deception_raw': self.helper._get_safe_series(df, 'deception_index_D', 0.0, method_name=method_name),
            'chip_fatigue_raw': self.helper._get_safe_series(df, 'chip_fatigue_index_D', 0.0, method_name=method_name),
            'loser_pain_raw': self.helper._get_safe_series(df, 'loser_pain_index_D', 0.0, method_name=method_name),
            'deception_lure_long_raw': self.helper._get_safe_series(df, 'deception_lure_long_intensity_D', 0.0, method_name=method_name),
            'deception_lure_short_raw': self.helper._get_safe_series(df, 'deception_lure_short_intensity_D', 0.0, method_name=method_name),
            'hidden_accumulation_intensity_raw': self.helper._get_safe_series(df, 'hidden_accumulation_intensity_D', 0.0, method_name=method_name),
            'market_sentiment_raw': self.helper._get_safe_series(df, 'market_sentiment_score_D', 0.0, method_name=method_name),
            'volatility_instability_raw': self.helper._get_safe_series(df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', 0.0, method_name=method_name),
            'mf_buy_ofi_raw': self.helper._get_safe_series(df, 'main_force_buy_ofi_D', 0.0, method_name=method_name),
            'mf_cost_advantage_raw': self.helper._get_safe_series(df, 'main_force_cost_advantage_D', 0.0, method_name=method_name),
            'floating_chip_cleansing_raw': self.helper._get_safe_series(df, 'floating_chip_cleansing_efficiency_D', 0.0, method_name=method_name),
            'total_loser_rate_raw': self.helper._get_safe_series(df, 'total_loser_rate_D', 0.0, method_name=method_name),
            'structural_tension_index_raw': self.helper._get_safe_series(df, 'structural_tension_index_D', 0.0, method_name=method_name),
            'covert_accumulation_signal_raw': self.helper._get_safe_series(df, 'covert_accumulation_signal_D', 0.0, method_name=method_name),
            'structural_potential_score_raw': self.helper._get_safe_series(df, 'structural_potential_score_D', 0.0, method_name=method_name),
            'winner_stability_index_raw': self.helper._get_safe_series(df, 'winner_stability_index_D', 0.0, method_name=method_name),
            'constructive_turnover_ratio_raw': self.helper._get_safe_series(df, 'constructive_turnover_ratio_D', 0.0, method_name=method_name),
            'equilibrium_compression_raw': self.helper._get_safe_series(df, 'equilibrium_compression_index_D', 0.0, method_name=method_name),
            'price_volume_entropy_raw': self.helper._get_safe_series(df, 'price_volume_entropy_D', 0.0, method_name=method_name),
            'fractal_dimension_raw': self.helper._get_safe_series(df, 'FRACTAL_DIMENSION_89d_D', 0.0, method_name=method_name),
            'hurst_raw': self.helper._get_safe_series(df, 'HURST_144d_D', 0.0, method_name=method_name),
            'is_consolidating_raw': self.helper._get_safe_series(df, 'is_consolidating_D', 0.0, method_name=method_name),
            'dynamic_consolidation_duration_raw': self.helper._get_safe_series(df, 'dynamic_consolidation_duration_D', 0.0, method_name=method_name),
            'volume_burstiness_raw': self.helper._get_safe_series(df, 'volume_burstiness_index_D', 0.0, method_name=method_name),
            'market_impact_cost_raw': self.helper._get_safe_series(df, 'market_impact_cost_D', 0.0, method_name=method_name),
            'liquidity_authenticity_raw': self.helper._get_safe_series(df, 'liquidity_authenticity_score_D', 0.0, method_name=method_name),
            'order_book_imbalance_raw': self.helper._get_safe_series(df, 'order_book_imbalance_D', 0.0, method_name=method_name),
            'micro_price_impact_asymmetry_raw': self.helper._get_safe_series(df, 'micro_price_impact_asymmetry_D', 0.0, method_name=method_name),
            'internal_accumulation_raw': self.helper._get_safe_series(df, 'internal_accumulation_intensity_D', 0.0, method_name=method_name),
            'gathering_by_support_raw': self.helper._get_safe_series(df, 'gathering_by_support_D', 0.0, method_name=method_name),
            'main_force_flow_gini_raw': self.helper._get_safe_series(df, 'main_force_flow_gini_D', 0.0, method_name=method_name),
            'mf_t0_buy_efficiency_raw': self.helper._get_safe_series(df, 'main_force_t0_buy_efficiency_D', 0.0, method_name=method_name),
            'buy_quote_exhaustion_raw': self.helper._get_safe_series(df, 'buy_quote_exhaustion_rate_D', 0.0, method_name=method_name),
            'bid_side_liquidity_raw': self.helper._get_safe_series(df, 'bid_side_liquidity_D', 0.0, method_name=method_name),
            'net_lg_amount_raw': self.helper._get_safe_series(df, 'net_lg_amount_calibrated_D', 0.0, method_name=method_name),
            'dip_buy_absorption_raw': self.helper._get_safe_series(df, 'dip_buy_absorption_strength_D', 0.0, method_name=method_name),
            'main_force_slippage_raw': self.helper._get_safe_series(df, 'main_force_slippage_index_D', 0.0, method_name=method_name),
            'micro_impact_elasticity_raw': self.helper._get_safe_series(df, 'micro_impact_elasticity_D', 0.0, method_name=method_name),
            'winner_concentration_raw': self.helper._get_safe_series(df, 'winner_concentration_90pct_D', 0.0, method_name=method_name),
            'loser_concentration_raw': self.helper._get_safe_series(df, 'loser_concentration_90pct_D', 0.0, method_name=method_name),
            'cost_dispersion_raw': self.helper._get_safe_series(df, 'cost_dispersion_index_D', 0.0, method_name=method_name),
            'dominant_peak_solidity_raw': self.helper._get_safe_series(df, 'dominant_peak_solidity_D', 0.0, method_name=method_name),
            'chip_health_raw': self.helper._get_safe_series(df, 'chip_health_score_D', 0.0, method_name=method_name),
            'lower_shadow_absorption_raw': self.helper._get_safe_series(df, 'lower_shadow_absorption_strength_D', 0.0, method_name=method_name),
            'panic_sell_volume_contribution_raw': self.helper._get_safe_series(df, 'panic_sell_volume_contribution_D', 0.0, method_name=method_name),
            'profit_realization_quality_raw': self.helper._get_safe_series(df, 'profit_realization_quality_D', 0.0, method_name=method_name),
            # 新增的原始信号
            'ask_side_liquidity_raw': self.helper._get_safe_series(df, 'ask_side_liquidity_D', 0.0, method_name=method_name),
            'mf_level5_buy_ofi_raw': self.helper._get_safe_series(df, 'main_force_level5_buy_ofi_D', 0.0, method_name=method_name),
            'mf_buy_execution_alpha_raw': self.helper._get_safe_series(df, 'main_force_buy_execution_alpha_D', 0.0, method_name=method_name),
            'upper_shadow_selling_pressure_raw': self.helper._get_safe_series(df, 'upper_shadow_selling_pressure_D', 0.0, method_name=method_name),
            'smart_money_inst_net_buy_raw': self.helper._get_safe_series(df, 'SMART_MONEY_INST_NET_BUY_D', 0.0, method_name=method_name),
            'microstructure_efficiency_raw': self.helper._get_safe_series(df, 'microstructure_efficiency_index_D', 0.0, method_name=method_name)
        }

        # 获取累积信号的原始数据 (这些列现在已在df中)
        for window in cumulative_flow_windows:
            raw_signals[f'cumulative_mf_flow_{window}d_raw'] = self.helper._get_safe_series(df, f'{mf_flow_base}_{window}d_sum', 0.0, method_name=method_name)
        for window in cumulative_acc_windows:
            raw_signals[f'cumulative_hidden_acc_{window}d_raw'] = self.helper._get_safe_series(df, f'{hidden_acc_base}_{window}d_sum', 0.0, method_name=method_name)
            raw_signals[f'cumulative_suppressive_acc_{window}d_raw'] = self.helper._get_safe_series(df, f'{suppressive_acc_base}_{window}d_sum', 0.0, method_name=method_name)

        _temp_debug_values["原始信号值"] = {k: v for k, v in raw_signals.items()}
        return raw_signals

    def _calculate_market_context_score(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_slope_accel_weights: Dict, market_context_weights: Dict, price_weakness_slope_window: int, low_volatility_bbw_window: int, method_name: str, _temp_debug_values: Dict, neutral_range_threshold: float) -> pd.Series:
        """
        【V2.6 · 原始数据层适配版】计算隐蔽吸筹的市场背景分数。
        - 核心修改: 引入更多市场背景信号，并调整部分信号的归一化逻辑。
        """
        retail_panic_score = self.helper._normalize_series(raw_signals['retail_panic_raw'], df_index, bipolar=False)
        mtf_price_weakness_score = self.helper._get_mtf_slope_accel_score(df, f'close_D', mtf_slope_accel_weights, df_index, method_name, ascending=False, bipolar=False)
        low_volatility_score = self.helper._normalize_series(raw_signals['bbw_raw'], df_index, ascending=False)
        # 替换 SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM
        sentiment_pendulum_inverted_score = (1 - self.helper._normalize_series(raw_signals['market_sentiment_raw'], df_index, bipolar=True).clip(lower=0))
        # 替换 SCORE_STRUCT_AXIOM_TENSION
        tension_inverted_score = (1 - self.helper._normalize_series(raw_signals['structural_tension_index_raw'], df_index, bipolar=False))
        market_sentiment_inverted_score = self.helper._normalize_series(raw_signals['market_sentiment_raw'], df_index, ascending=False)
        volatility_instability_inverted_score = self.helper._normalize_series(raw_signals['volatility_instability_raw'], df_index, ascending=False)
        # 新增市场背景信号
        equilibrium_compression_score = self.helper._normalize_series(raw_signals['equilibrium_compression_raw'], df_index, bipolar=False)
        price_volume_entropy_inverted_score = self.helper._normalize_series(raw_signals['price_volume_entropy_raw'], df_index, ascending=False)
        fractal_dimension_inverted_score = self.helper._normalize_series(raw_signals['fractal_dimension_raw'], df_index, ascending=False) # 奖励非趋势性
        hurst_inverted_score = self.helper._normalize_series(raw_signals['hurst_raw'], df_index, ascending=False) # 奖励均值回归
        is_consolidating_score = self.helper._normalize_series(raw_signals['is_consolidating_raw'], df_index, bipolar=False)
        dynamic_consolidation_duration_score = self.helper._normalize_series(raw_signals['dynamic_consolidation_duration_raw'], df_index, bipolar=False)
        volume_burstiness_inverted_score = self.helper._normalize_series(raw_signals['volume_burstiness_raw'], df_index, ascending=False)
        market_impact_cost_inverted_score = self.helper._normalize_series(raw_signals['market_impact_cost_raw'], df_index, ascending=False)
        liquidity_authenticity_score = self.helper._normalize_series(raw_signals['liquidity_authenticity_raw'], df_index, bipolar=False)
        # 订单簿不平衡和微观价格冲击不对称性，奖励中性（接近0）
        order_book_imbalance_neutral_score = (1 - self.helper._normalize_series(raw_signals['order_book_imbalance_raw'].abs(), df_index, bipolar=False)).clip(0,1)
        micro_price_impact_asymmetry_neutral_score = (1 - self.helper._normalize_series(raw_signals['micro_price_impact_asymmetry_raw'].abs(), df_index, bipolar=False)).clip(0,1)
        market_context_scores_dict = {
            "retail_panic": retail_panic_score,
            "price_weakness": mtf_price_weakness_score,
            "low_volatility": low_volatility_score,
            "sentiment_pendulum_inverted": sentiment_pendulum_inverted_score,
            "tension_inverted": tension_inverted_score,
            "market_sentiment_inverted": market_sentiment_inverted_score,
            "volatility_instability_inverted": volatility_instability_inverted_score,
            "equilibrium_compression": equilibrium_compression_score,
            "price_volume_entropy_inverted": price_volume_entropy_inverted_score,
            "fractal_dimension_inverted": fractal_dimension_inverted_score,
            "hurst_inverted": hurst_inverted_score,
            "is_consolidating": is_consolidating_score,
            "dynamic_consolidation_duration": dynamic_consolidation_duration_score,
            "volume_burstiness_inverted": volume_burstiness_inverted_score,
            "market_impact_cost_inverted": market_impact_cost_inverted_score,
            "liquidity_authenticity": liquidity_authenticity_score,
            "order_book_imbalance_neutral": order_book_imbalance_neutral_score,
            "micro_price_impact_asymmetry_neutral": micro_price_impact_asymmetry_neutral_score
        }
        market_context_score = _robust_geometric_mean(market_context_scores_dict, market_context_weights, df_index)
        _temp_debug_values["市场背景"] = {
            "retail_panic_score": retail_panic_score,
            "mtf_price_weakness_score": mtf_price_weakness_score,
            "low_volatility_score": low_volatility_score,
            "sentiment_pendulum_inverted_score": sentiment_pendulum_inverted_score,
            "tension_inverted_score": tension_inverted_score,
            "market_sentiment_inverted_score": market_sentiment_inverted_score,
            "volatility_instability_inverted_score": volatility_instability_inverted_score,
            "equilibrium_compression_score": equilibrium_compression_score,
            "price_volume_entropy_inverted_score": price_volume_entropy_inverted_score,
            "fractal_dimension_inverted_score": fractal_dimension_inverted_score,
            "hurst_inverted_score": hurst_inverted_score,
            "is_consolidating_score": is_consolidating_score,
            "dynamic_consolidation_duration_score": dynamic_consolidation_duration_score,
            "volume_burstiness_inverted_score": volume_burstiness_inverted_score,
            "market_impact_cost_inverted_score": market_impact_cost_inverted_score,
            "liquidity_authenticity_score": liquidity_authenticity_score,
            "order_book_imbalance_neutral_score": order_book_imbalance_neutral_score,
            "micro_price_impact_asymmetry_neutral_score": micro_price_impact_asymmetry_neutral_score,
            "market_context_score": market_context_score
        }
        return market_context_score

    def _calculate_covert_action_score(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_slope_accel_weights: Dict, covert_action_weights: Dict, method_name: str, _temp_debug_values: Dict, cumulative_flow_windows: List[int], cumulative_flow_weights: Dict, cumulative_acc_windows: List[int], cumulative_acc_weights: Dict, daily_mf_flow_weight: float, cumulative_mf_flow_weight: float, daily_acc_weight: float, cumulative_acc_weight: float, new_raw_signals_weights: Dict, main_force_accumulation_resonance_weight: float) -> pd.Series:
        """
        【V2.11 · 信号丰富与共振机制版】计算隐蔽行动分数。
        - 核心修改: 引入新的原始信号及其归一化分数，并新增主力吸筹共振信号。
        """
        mtf_suppressive_accum_score = self.helper._get_mtf_slope_accel_score(df, 'suppressive_accumulation_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_main_force_flow_score = self.helper._get_mtf_slope_accel_score(df, 'main_force_net_flow_calibrated_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True) # 保持双极性，正为流入，负为流出
        mtf_deception_lure_long_score = self.helper._get_mtf_slope_accel_score(df, 'deception_lure_long_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        stealth_ops_normalized = self.helper._normalize_series(raw_signals['covert_accumulation_signal_raw'], df_index, bipolar=False)
        mtf_hidden_accumulation_intensity = self.helper._get_mtf_slope_accel_score(df, 'hidden_accumulation_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        chip_historical_potential_normalized = self.helper._normalize_series(raw_signals['structural_potential_score_raw'].clip(lower=0), df_index, bipolar=False)
        mtf_mf_buy_ofi_normalized = self.helper._get_mtf_slope_accel_score(df, 'main_force_buy_ofi_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_mf_cost_advantage_normalized = self.helper._get_mtf_slope_accel_score(df, 'main_force_cost_advantage_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_mf_flow_slope_normalized = self.helper._get_mtf_slope_accel_score(df, 'main_force_net_flow_calibrated_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True).clip(lower=0)
        mtf_suppressive_accum_slope_normalized = self.helper._get_mtf_slope_accel_score(df, 'suppressive_accumulation_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True).clip(lower=0)
        internal_accumulation_score = self.helper._normalize_series(raw_signals['internal_accumulation_raw'], df_index, bipolar=False)
        gathering_by_support_score = self.helper._normalize_series(raw_signals['gathering_by_support_raw'], df_index, bipolar=False)
        main_force_flow_gini_score = self.helper._normalize_series(raw_signals['main_force_flow_gini_raw'], df_index, bipolar=False)
        mf_t0_buy_efficiency_score = self.helper._normalize_series(raw_signals['mf_t0_buy_efficiency_raw'], df_index, bipolar=False)
        buy_quote_exhaustion_inverted_score = self.helper._normalize_series(raw_signals['buy_quote_exhaustion_raw'], df_index, ascending=False)
        bid_side_liquidity_score = self.helper._normalize_series(raw_signals['bid_side_liquidity_raw'], df_index, bipolar=False)
        net_lg_amount_score = self.helper._normalize_series(raw_signals['net_lg_amount_raw'], df_index, bipolar=False)
        dip_buy_absorption_score = self.helper._normalize_series(raw_signals['dip_buy_absorption_raw'], df_index, bipolar=False)
        main_force_slippage_inverted_score = self.helper._normalize_series(raw_signals['main_force_slippage_raw'], df_index, ascending=False)
        micro_impact_elasticity_inverted_score = self.helper._normalize_series(raw_signals['micro_impact_elasticity_raw'], df_index, ascending=False)

        # --- 新增原始信号的归一化分数 ---
        ask_side_liquidity_inverted_score = self.helper._normalize_series(raw_signals['ask_side_liquidity_raw'], df_index, ascending=False)
        mf_level5_buy_ofi_score = self.helper._normalize_series(raw_signals['mf_level5_buy_ofi_raw'], df_index, bipolar=False)
        mf_buy_execution_alpha_score = self.helper._normalize_series(raw_signals['mf_buy_execution_alpha_raw'], df_index, bipolar=False)
        upper_shadow_selling_pressure_inverted_score = self.helper._normalize_series(raw_signals['upper_shadow_selling_pressure_raw'], df_index, ascending=False)
        smart_money_inst_net_buy_score = self.helper._normalize_series(raw_signals['smart_money_inst_net_buy_raw'], df_index, bipolar=False)
        microstructure_efficiency_score = self.helper._normalize_series(raw_signals['microstructure_efficiency_raw'], df_index, bipolar=False)

        # --- 累积资金流和吸筹强度计算 ---
        cumulative_mf_flow_scores = []
        for window in cumulative_flow_windows:
            cumulative_mf_flow_raw = raw_signals[f'cumulative_mf_flow_{window}d_raw']
            cumulative_mf_flow_scores.append(self.helper._normalize_series(cumulative_mf_flow_raw, df_index, bipolar=True))
        
        cumulative_mf_flow_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        total_flow_weight_cum = sum(cumulative_flow_weights.values())
        if total_flow_weight_cum > 0:
            for i, window in enumerate(cumulative_flow_windows):
                weight = cumulative_flow_weights.get(str(window), 0.0)
                cumulative_mf_flow_score += cumulative_mf_flow_scores[i] * (weight / total_flow_weight_cum)
        
        blended_mf_flow_score = (mtf_main_force_flow_score * daily_mf_flow_weight + cumulative_mf_flow_score * cumulative_mf_flow_weight) / (daily_mf_flow_weight + cumulative_mf_flow_weight)
        contextualized_main_force_flow_score = (blended_mf_flow_score + 1) / 2

        # 累积隐蔽吸筹强度
        cumulative_hidden_acc_scores = []
        for window in cumulative_acc_windows:
            cumulative_hidden_acc_raw = raw_signals[f'cumulative_hidden_acc_{window}d_raw']
            cumulative_hidden_acc_scores.append(self.helper._normalize_series(cumulative_hidden_acc_raw, df_index, bipolar=False))
        
        cumulative_hidden_acc_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        total_acc_weight_cum = sum(cumulative_acc_weights.values())
        if total_acc_weight_cum > 0:
            for i, window in enumerate(cumulative_acc_windows):
                weight = cumulative_acc_weights.get(str(window), 0.0)
                cumulative_hidden_acc_score += cumulative_hidden_acc_scores[i] * (weight / total_acc_weight_cum)

        blended_hidden_acc_score = (mtf_hidden_accumulation_intensity * daily_acc_weight + cumulative_hidden_acc_score * cumulative_acc_weight) / (daily_acc_weight + cumulative_acc_weight)
        contextualized_hidden_accumulation_score = blended_hidden_acc_score

        # --- 主力吸筹共振信号 ---
        main_force_accumulation_resonance_base_signals = [
            'main_force_net_flow_calibrated_D',
            'hidden_accumulation_intensity_D',
            'main_force_buy_ofi_D'
        ]
        main_force_accumulation_resonance_score_bipolar = self.helper._get_mtf_resonance_score(
            df, main_force_accumulation_resonance_base_signals, mtf_slope_accel_weights, df_index, method_name
        )
        # 将双极性共振分数映射到 [0, 1] 范围，以便与几何平均融合
        main_force_accumulation_resonance_score = (main_force_accumulation_resonance_score_bipolar + 1) / 2


        covert_action_scores_dict = {
            "suppressive_accum": mtf_suppressive_accum_score,
            "contextualized_main_force_flow": contextualized_main_force_flow_score,
            "deception_lure_long": mtf_deception_lure_long_score,
            "stealth_ops": stealth_ops_normalized,
            "contextualized_hidden_accumulation": contextualized_hidden_accumulation_score,
            "chip_historical_potential": chip_historical_potential_normalized,
            "mf_buy_ofi": mtf_mf_buy_ofi_normalized,
            "mf_cost_advantage": mtf_mf_cost_advantage_normalized,
            "mf_flow_slope": mtf_mf_flow_slope_normalized,
            "suppressive_accum_slope": mtf_suppressive_accum_slope_normalized,
            "internal_accumulation": internal_accumulation_score,
            "gathering_by_support": gathering_by_support_score,
            "main_force_flow_gini": main_force_flow_gini_score,
            "mf_t0_buy_efficiency": mf_t0_buy_efficiency_score,
            "buy_quote_exhaustion_inverted": buy_quote_exhaustion_inverted_score,
            "bid_side_liquidity": bid_side_liquidity_score,
            "net_lg_amount": net_lg_amount_score,
            "dip_buy_absorption": dip_buy_absorption_score,
            "main_force_slippage_inverted": main_force_slippage_inverted_score,
            "micro_impact_elasticity_inverted": micro_impact_elasticity_inverted_score,
            # 新增信号
            "ask_side_liquidity_inverted": ask_side_liquidity_inverted_score,
            "mf_level5_buy_ofi": mf_level5_buy_ofi_score,
            "mf_buy_execution_alpha": mf_buy_execution_alpha_score,
            "upper_shadow_selling_pressure_inverted": upper_shadow_selling_pressure_inverted_score,
            "smart_money_inst_net_buy": smart_money_inst_net_buy_score,
            "microstructure_efficiency": microstructure_efficiency_score,
            # 主力吸筹共振信号
            "main_force_accumulation_resonance": main_force_accumulation_resonance_score
        }
        covert_action_score = _robust_geometric_mean(covert_action_scores_dict, covert_action_weights, df_index)
        _temp_debug_values["隐蔽行动"] = {
            "mtf_suppressive_accum_score": mtf_suppressive_accum_score,
            "mtf_main_force_flow_score": mtf_main_force_flow_score,
            "cumulative_mf_flow_score": cumulative_mf_flow_score,
            "blended_mf_flow_score": blended_mf_flow_score,
            "contextualized_main_force_flow_score": contextualized_main_force_flow_score,
            "mtf_deception_lure_long_score": mtf_deception_lure_long_score,
            "stealth_ops_normalized": stealth_ops_normalized,
            "mtf_hidden_accumulation_intensity": mtf_hidden_accumulation_intensity,
            "cumulative_hidden_acc_score": cumulative_hidden_acc_score,
            "blended_hidden_acc_score": blended_hidden_acc_score,
            "contextualized_hidden_accumulation_score": contextualized_hidden_accumulation_score,
            "chip_historical_potential_normalized": chip_historical_potential_normalized,
            "mtf_mf_buy_ofi_normalized": mtf_mf_buy_ofi_normalized,
            "mtf_mf_cost_advantage_normalized": mtf_mf_cost_advantage_normalized,
            "mtf_mf_flow_slope_normalized": mtf_mf_flow_slope_normalized,
            "suppressive_accum_slope_normalized": mtf_suppressive_accum_slope_normalized,
            "internal_accumulation_score": internal_accumulation_score,
            "gathering_by_support_score": gathering_by_support_score,
            "main_force_flow_gini_score": main_force_flow_gini_score,
            "mf_t0_buy_efficiency_score": mf_t0_buy_efficiency_score,
            "buy_quote_exhaustion_inverted_score": buy_quote_exhaustion_inverted_score,
            "bid_side_liquidity_score": bid_side_liquidity_score,
            "net_lg_amount_score": net_lg_amount_score,
            "dip_buy_absorption_score": dip_buy_absorption_score,
            "main_force_slippage_inverted_score": main_force_slippage_inverted_score,
            "micro_impact_elasticity_inverted_score": micro_impact_elasticity_inverted_score,
            # 新增信号
            "ask_side_liquidity_inverted_score": ask_side_liquidity_inverted_score,
            "mf_level5_buy_ofi_score": mf_level5_buy_ofi_score,
            "mf_buy_execution_alpha_score": mf_buy_execution_alpha_score,
            "upper_shadow_selling_pressure_inverted_score": upper_shadow_selling_pressure_inverted_score,
            "smart_money_inst_net_buy_score": smart_money_inst_net_buy_score,
            "microstructure_efficiency_score": microstructure_efficiency_score,
            # 主力吸筹共振信号
            "main_force_accumulation_resonance_score_bipolar": main_force_accumulation_resonance_score_bipolar,
            "main_force_accumulation_resonance_score": main_force_accumulation_resonance_score,
            "covert_action_score": covert_action_score
        }
        return covert_action_score

    def _calculate_chip_optimization_score(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_slope_accel_weights: Dict, chip_optimization_weights: Dict, method_name: str, _temp_debug_values: Dict) -> pd.Series:
        """
        【V2.6 · 原始数据层适配版】计算筹码优化分数。
        - 核心修改: 引入更多筹码优化信号，并调整部分信号的归一化逻辑。
        """
        mtf_chip_fatigue_score = self.helper._get_mtf_slope_accel_score(df, 'chip_fatigue_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_loser_pain_score = self.helper._get_mtf_slope_accel_score(df, 'loser_pain_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        # 替换 SCORE_CHIP_AXIOM_HOLDER_SENTIMENT
        holder_sentiment_inverted_score = (1 - self.helper._normalize_series(raw_signals['winner_stability_index_raw'], df_index, bipolar=False)).clip(0, 1)
        # 替换 SCORE_CHIP_TURNOVER_PURITY_COST_OPTIMIZATION
        turnover_purity_cost_opt_normalized = self.helper._normalize_series(raw_signals['constructive_turnover_ratio_raw'].clip(lower=0), df_index, bipolar=False)
        mtf_floating_chip_cleansing_normalized = self.helper._get_mtf_slope_accel_score(df, 'floating_chip_cleansing_efficiency_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_total_loser_rate_normalized = self.helper._get_mtf_slope_accel_score(df, 'total_loser_rate_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        # 新增筹码优化信号
        winner_concentration_inverted_score = self.helper._normalize_series(raw_signals['winner_concentration_raw'], df_index, ascending=False)
        loser_concentration_score = self.helper._normalize_series(raw_signals['loser_concentration_raw'], df_index, bipolar=False)
        cost_dispersion_inverted_score = self.helper._normalize_series(raw_signals['cost_dispersion_raw'], df_index, ascending=False)
        dominant_peak_solidity_score = self.helper._normalize_series(raw_signals['dominant_peak_solidity_raw'], df_index, bipolar=False)
        chip_health_score = self.helper._normalize_series(raw_signals['chip_health_raw'], df_index, bipolar=False)
        lower_shadow_absorption_score = self.helper._normalize_series(raw_signals['lower_shadow_absorption_raw'], df_index, bipolar=False)
        panic_sell_volume_contribution_score = self.helper._normalize_series(raw_signals['panic_sell_volume_contribution_raw'], df_index, bipolar=False)
        profit_realization_inverted_score = self.helper._normalize_series(raw_signals['profit_realization_quality_raw'], df_index, ascending=False)
        chip_optimization_scores_dict = {
            "chip_fatigue": mtf_chip_fatigue_score,
            "loser_pain": mtf_loser_pain_score,
            "holder_sentiment_inverted": holder_sentiment_inverted_score,
            "turnover_purity_cost_opt": turnover_purity_cost_opt_normalized,
            "floating_chip_cleansing": mtf_floating_chip_cleansing_normalized,
            "total_loser_rate": mtf_total_loser_rate_normalized,
            "winner_concentration_inverted": winner_concentration_inverted_score,
            "loser_concentration": loser_concentration_score,
            "cost_dispersion_inverted": cost_dispersion_inverted_score,
            "dominant_peak_solidity": dominant_peak_solidity_score,
            "chip_health": chip_health_score,
            "lower_shadow_absorption": lower_shadow_absorption_score,
            "panic_sell_contribution": panic_sell_volume_contribution_score,
            "profit_realization_inverted": profit_realization_inverted_score
        }
        chip_optimization_score = _robust_geometric_mean(chip_optimization_scores_dict, chip_optimization_weights, df_index)
        _temp_debug_values["筹码优化"] = {
            "mtf_chip_fatigue_score": mtf_chip_fatigue_score,
            "mtf_loser_pain_score": mtf_loser_pain_score,
            "holder_sentiment_inverted_score": holder_sentiment_inverted_score,
            "turnover_purity_cost_opt_normalized": turnover_purity_cost_opt_normalized,
            "mtf_floating_chip_cleansing_normalized": mtf_floating_chip_cleansing_normalized,
            "mtf_total_loser_rate_normalized": mtf_total_loser_rate_normalized,
            "winner_concentration_inverted_score": winner_concentration_inverted_score,
            "loser_concentration_score": loser_concentration_score,
            "cost_dispersion_inverted_score": cost_dispersion_inverted_score,
            "dominant_peak_solidity_score": dominant_peak_solidity_score,
            "chip_health_score": chip_health_score,
            "lower_shadow_absorption_score": lower_shadow_absorption_score,
            "panic_sell_volume_contribution_score": panic_sell_volume_contribution_score,
            "profit_realization_inverted_score": profit_realization_inverted_score,
            "chip_optimization_score": chip_optimization_score
        }
        return chip_optimization_score

    def _fuse_final_score(self, df_index: pd.Index, market_context_score: pd.Series, covert_action_score: pd.Series, chip_optimization_score: pd.Series, fusion_weights: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        将三个维度分数进行最终融合。
        """
        final_fusion_scores_dict = {
            "market_context": market_context_score,
            "covert_action": covert_action_score,
            "chip_optimization": chip_optimization_score
        }
        covert_accumulation_score = _robust_geometric_mean(final_fusion_scores_dict, fusion_weights, df_index)
        _temp_debug_values["最终合成"] = {
            "covert_accumulation_score": covert_accumulation_score
        }
        return covert_accumulation_score.clip(0, 1).astype(np.float32)

    def _print_debug_info(self, debug_output: Dict, _temp_debug_values: Dict, method_name: str, probe_ts: pd.Timestamp):
        """
        【V2.11 · 信号丰富与共振机制版】统一打印隐蔽吸筹计算的调试信息。
        - 核心修改: 确保能够正确打印新加入的原始信号和共振信号的调试信息。
        """
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
        for key, value in _temp_debug_values["原始信号值"].items():
            if isinstance(value, pd.Series):
                val = value.loc[probe_ts] if probe_ts in value.index else np.nan
                debug_output[f"        '{key}': {val:.4f}"] = ""
            else:
                debug_output[f"        '{key}': {value}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 市场背景 ---"] = ""
        for key, series in _temp_debug_values["市场背景"].items():
            if isinstance(series, pd.Series):
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            else:
                debug_output[f"        {key}: {series}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 隐蔽行动 ---"] = ""
        for key, series in _temp_debug_values["隐蔽行动"].items():
            if isinstance(series, pd.Series):
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            else:
                debug_output[f"        {key}: {series}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 筹码优化 ---"] = ""
        for key, series in _temp_debug_values["筹码优化"].items():
            if isinstance(series, pd.Series):
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            else:
                debug_output[f"        {key}: {series}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 最终合成 ---"] = ""
        for key, series in _temp_debug_values["最终合成"].items():
            if isinstance(series, pd.Series):
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            else:
                debug_output[f"        {key}: {series}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 隐蔽吸筹诊断完成，最终分值: {_temp_debug_values['final_score'].loc[probe_ts]:.4f}"] = ""
        self.helper._print_debug_output(debug_output)
