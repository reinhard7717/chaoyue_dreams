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
    """
    计算“隐蔽吸筹”的专属信号。PROCESS_META_COVERT_ACCUMULATION
    """
    def __init__(self, strategy_instance, helper_instance: ProcessIntelligenceHelper):
            self.strategy = strategy_instance
            self.helper = helper_instance

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.12 · 微观订单流与结构共振版】计算“隐蔽吸筹”的专属信号。
        - 核心升级: 引入更多微观订单流和市场结构信号，新增微观订单流共振机制，提供更全面和协同的隐蔽吸筹判断。
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
        new_raw_signals_weights, main_force_accumulation_resonance_weight, \
        new_raw_signals_weights_v2, covert_order_flow_resonance_weight = self._get_covert_accumulation_config(config)
        df_index = df.index
        # 2. 校验并获取原始信号 (此方法内部会先计算所有派生信号)
        raw_signals = self._validate_and_get_raw_signals(df, method_name, price_weakness_slope_window, low_volatility_bbw_window, mtf_slope_accel_weights, is_debug_enabled_for_method, probe_ts, _temp_debug_values, cumulative_flow_windows, cumulative_acc_windows)
        if raw_signals is None:
            return pd.Series(0.0, index=df.index)
        # 3. 计算维度一：市场背景 (Market Context)
        market_context_score = self._calculate_market_context_score(df, df_index, raw_signals, mtf_slope_accel_weights, market_context_weights, price_weakness_slope_window, low_volatility_bbw_window, method_name, _temp_debug_values, neutral_range_threshold)
        # 4. 计算维度二：隐蔽行动 (Covert Action)
        covert_action_score = self._calculate_covert_action_score(df, df_index, raw_signals, mtf_slope_accel_weights, covert_action_weights, method_name, _temp_debug_values, cumulative_flow_windows, cumulative_flow_weights, cumulative_acc_windows, cumulative_acc_weights, daily_mf_flow_weight, cumulative_mf_flow_weight, daily_acc_weight, cumulative_acc_weight, new_raw_signals_weights, main_force_accumulation_resonance_weight, new_raw_signals_weights_v2, covert_order_flow_resonance_weight)
        # 5. 计算维度三：筹码优化 (Chip Optimization)
        chip_optimization_score = self._calculate_chip_optimization_score(df, df_index, raw_signals, mtf_slope_accel_weights, chip_optimization_weights, method_name, _temp_debug_values)
        # 6. 最终合成：三维融合
        final_score = self._fuse_final_score(df_index, market_context_score, covert_action_score, chip_optimization_score, fusion_weights, _temp_debug_values)
        _temp_debug_values["final_score"] = final_score # 存储最终分数用于调试输出
        # --- 统一输出调试信息 ---
        # if is_debug_enabled_for_method and probe_ts:
        #     self._print_debug_info(debug_output, _temp_debug_values, method_name, probe_ts)
        return final_score

    def _get_covert_accumulation_config(self, config: Dict) -> Tuple[Dict, Dict, Dict, Dict, int, int, Dict, float, List[int], Dict, List[int], Dict, float, float, float, float, Dict, float, Dict, float]:
        """
        【V2.12 · 微观订单流与结构共振版】获取隐蔽吸筹计算所需的所有配置参数。
        - 核心修改: 引入新的原始信号权重，新增微观订单流共振信号权重，并更新隐蔽行动和市场背景的默认权重。
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
            "micro_price_impact_asymmetry_neutral": 0.05,
            # 新增市场背景信号权重
            "order_book_liquidity_supply": 0.05,
            "is_high_potential_consolidation": 0.05
        })
        # 新增的原始信号权重 (V2.11)
        new_raw_signals_weights = get_param_value(covert_accum_params.get('new_raw_signals_weights'), {
            "ask_side_liquidity_inverted": 0.03,
            "mf_level5_buy_ofi": 0.05,
            "mf_buy_execution_alpha": 0.05,
            "upper_shadow_selling_pressure_inverted": 0.03,
            "smart_money_inst_net_buy": 0.05,
            "microstructure_efficiency": 0.03
        })
        # 新增的原始信号权重 (V2.12)
        new_raw_signals_weights_v2 = get_param_value(covert_accum_params.get('new_raw_signals_weights_v2'), {
            "buy_flow_efficiency": 0.05,
            "sell_flow_efficiency_inverted": 0.03,
            "main_force_vwap_up_guidance": 0.05,
            "observed_large_order_size_avg_inverted": 0.03
        })
        # 主力吸筹共振信号权重
        main_force_accumulation_resonance_weight = get_param_value(covert_accum_params.get('main_force_accumulation_resonance_weight'), 0.1)
        # 微观订单流共振信号权重
        covert_order_flow_resonance_weight = get_param_value(covert_accum_params.get('covert_order_flow_resonance_weight'), 0.08)
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
            # V2.11新增信号的默认权重
            "ask_side_liquidity_inverted": new_raw_signals_weights["ask_side_liquidity_inverted"],
            "mf_level5_buy_ofi": new_raw_signals_weights["mf_level5_buy_ofi"],
            "mf_buy_execution_alpha": new_raw_signals_weights["mf_buy_execution_alpha"],
            "upper_shadow_selling_pressure_inverted": new_raw_signals_weights["upper_shadow_selling_pressure_inverted"],
            "smart_money_inst_net_buy": new_raw_signals_weights["smart_money_inst_net_buy"],
            "microstructure_efficiency": new_raw_signals_weights["microstructure_efficiency"],
            "main_force_accumulation_resonance": main_force_accumulation_resonance_weight,
            # V2.12新增信号的默认权重
            "buy_flow_efficiency": new_raw_signals_weights_v2["buy_flow_efficiency"],
            "sell_flow_efficiency_inverted": new_raw_signals_weights_v2["sell_flow_efficiency_inverted"],
            "main_force_vwap_up_guidance": new_raw_signals_weights_v2["main_force_vwap_up_guidance"],
            "observed_large_order_size_avg_inverted": new_raw_signals_weights_v2["observed_large_order_size_avg_inverted"],
            "covert_order_flow_resonance": covert_order_flow_resonance_weight
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
               new_raw_signals_weights, main_force_accumulation_resonance_weight, \
               new_raw_signals_weights_v2, covert_order_flow_resonance_weight

    def _calculate_derived_signals(self, df: pd.DataFrame, mtf_slope_accel_weights: Dict, cumulative_flow_windows: List[int], cumulative_acc_windows: List[int]):
        """
        【V4.0 · 高阶导数物理建模版】基于斐波那契窗口计算斜率、加速度与加加速度。
        - 逻辑：通过 pandas_ta 计算三阶导数 JERK，捕捉主力吸筹从“线性增长”到“非线性爆发”的临界点。
        """
        fib_windows = [5, 8, 13, 21, 34, 55]
        # 核心监控指标：资金流与结构压缩
        core_base_signals = [
            'stealth_flow_ratio_D', 
            'SMART_MONEY_INST_NET_BUY_D', 
            'MA_POTENTIAL_COMPRESSION_RATE_D',
            'VPA_MF_ADJUSTED_EFF_D',
            'chip_stability_D'
        ]
        for base in core_base_signals:
            if base not in df.columns:
                continue
            for period in fib_windows:
                # 1. 一阶导数：斜率 (Velocity)
                slope_col = f'SLOPE_{period}_{base}'
                if slope_col not in df.columns:
                    df[slope_col] = ta.slope(df[base], length=period)
                # 2. 二阶导数：加速度 (Acceleration)
                accel_col = f'ACCEL_{period}_{base}'
                if accel_col not in df.columns and slope_col in df.columns:
                    df[accel_col] = ta.slope(df[slope_col], length=period)
                # 3. 三阶导数：加加速度 (Jerk/Jolt)
                jerk_col = f'JERK_{period}_{base}'
                if jerk_col not in df.columns and accel_col in df.columns:
                    df[jerk_col] = ta.slope(df[accel_col], length=period)
        # 兼容旧逻辑的累积计算
        for base in ['stealth_flow_ratio_D', 'SMART_MONEY_INST_NET_BUY_D']:
            if base in df.columns:
                for window in cumulative_flow_windows:
                    col_name = f'{base}_{window}d_sum'
                    if col_name not in df.columns:
                        df[col_name] = df[base].rolling(window, min_periods=1).sum()

    def _validate_and_get_raw_signals(self, df: pd.DataFrame, method_name: str, price_weakness_slope_window: int, low_volatility_bbw_window: int, mtf_slope_accel_weights: Dict, is_debug_enabled_for_method: bool, probe_ts: Optional[pd.Timestamp], _temp_debug_values: Dict, cumulative_flow_windows: List[int], cumulative_acc_windows: List[int]) -> Optional[Dict[str, pd.Series]]:
        """
        【V4.0 · 高阶导数物理建模版】校验并获取包含高阶导数的全量信号。
        - 逻辑：动态加载针对斐波那契窗口生成的 SLOPE/ACCEL/JERK 特征。
        """
        self._calculate_derived_signals(df, mtf_slope_accel_weights, cumulative_flow_windows, cumulative_acc_windows)
        fib_windows = [5, 8, 13, 21, 34, 55]
        core_indicators = ['stealth_flow_ratio_D', 'SMART_MONEY_INST_NET_BUY_D', 'MA_POTENTIAL_COMPRESSION_RATE_D']
        required_columns = [
            'IS_ROUNDING_BOTTOM_D', 'afternoon_flow_ratio_D', 'closing_flow_intensity_D',
            'long_term_chip_ratio_D', 'chip_stability_D', 'VPA_MF_ADJUSTED_EFF_D'
        ]
        # 动态添加高阶导数校验
        for base in core_indicators:
            for period in fib_windows:
                required_columns.extend([f'SLOPE_{period}_{base}', f'ACCEL_{period}_{base}', f'JERK_{period}_{base}'])
        if not self.helper._validate_required_signals(df, required_columns, method_name):
            return None
        raw_signals = {col: df[col] for col in required_columns if col in df.columns}
        # 映射至语义化字典以便后续计算
        semantic_signals = {
            'stealth_flow_main': df['stealth_flow_ratio_D'],
            'inst_buy_main': df['SMART_MONEY_INST_NET_BUY_D'],
            'compression_main': df['MA_POTENTIAL_COMPRESSION_RATE_D'],
            'afternoon_flow': df['afternoon_flow_ratio_D'],
            'closing_intensity': df['closing_flow_intensity_D'],
            'chip_stability': df['chip_stability_D'],
            'mf_efficiency': df['VPA_MF_ADJUSTED_EFF_D'],
            'long_term_chip': df['long_term_chip_ratio_D']
        }
        # 存储高阶导数引用
        for base in core_indicators:
            for period in fib_windows:
                semantic_signals[f'jerk_{base}_{period}'] = df[f'JERK_{period}_{base}']
                semantic_signals[f'accel_{base}_{period}'] = df[f'ACCEL_{period}_{base}']
                semantic_signals[f'slope_{base}_{period}'] = df[f'SLOPE_{period}_{base}']
        _temp_debug_values["semantic_signals"] = semantic_signals
        return semantic_signals

    def _calculate_market_context_score(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_slope_accel_weights: Dict, market_context_weights: Dict, price_weakness_slope_window: int, low_volatility_bbw_window: int, method_name: str, _temp_debug_values: Dict, neutral_range_threshold: float) -> pd.Series:
        """
        【V3.1 · 多维深度探测版】计算市场背景分数。
        - 逻辑：增加形态学权重，利用圆弧底识别和金坑确认（Golden Pit）定位高胜率吸筹结构。
        """
        scores = {
            "sentiment_extreme": self.helper._normalize_series(raw_signals['emo_extreme'], df_index, bipolar=False),
            "vol_compression": self.helper._normalize_series(raw_signals['vol_bbw'], df_index, ascending=False),
            "rounding_bottom": self.helper._normalize_series(raw_signals['rounding_bottom'], df_index, bipolar=False),
            "golden_pit": self.helper._normalize_series(raw_signals['golden_pit'], df_index, bipolar=False),
            "arc_curvature": self.helper._normalize_series(raw_signals['arc_curvature'], df_index, bipolar=False),
            "is_consolidating": self.helper._normalize_series(raw_signals['ma_compression'], df_index, bipolar=False),
            "structural_tension": (1 - self.helper._normalize_series(raw_signals['structural_tension'], df_index))
        }
        market_context_score = _robust_geometric_mean(scores, market_context_weights, df_index)
        _temp_debug_values["market_context_score"] = market_context_score
        return market_context_score

    def _calculate_covert_action_score(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_slope_accel_weights: Dict, covert_action_weights: Dict, method_name: str, _temp_debug_values: Dict, **kwargs) -> pd.Series:
        """
        【V4.0 · 高阶导数物理建模版】集成 JERK 突变特征的隐蔽行动评分。
        - 逻辑：将物理学中的“冲量”概念引入资金流分析，识别主力建仓的瞬时放量突变。
        """
        # 计算核心指标的多周期 JERK 综合分
        stealth_jerk_score = pd.Series(0.0, index=df_index)
        fib_windows = [5, 8, 13, 21, 34, 55]
        for p in fib_windows:
            stealth_jerk_score += self.helper._normalize_series(raw_signals[f'jerk_stealth_flow_ratio_D_{p}'], df_index)
        stealth_jerk_score /= len(fib_windows)
        # 计算机构买入的加速度分
        inst_buy_accel_score = pd.Series(0.0, index=df_index)
        for p in fib_windows:
            inst_buy_accel_score += self.helper._normalize_series(raw_signals[f'accel_SMART_MONEY_INST_NET_BUY_D_{p}'], df_index, bipolar=True)
        inst_buy_accel_score = (inst_buy_accel_score / len(fib_windows) + 1) / 2
        scores = {
            "stealth_flow_jerk": stealth_jerk_score,
            "inst_buy_accel": inst_buy_accel_score,
            "stealth_flow_slope": self.helper._normalize_series(raw_signals['slope_stealth_flow_ratio_D_13'], df_index),
            "afternoon_bias": self.helper._normalize_series(raw_signals['afternoon_flow'], df_index),
            "closing_intensity": self.helper._normalize_series(raw_signals['closing_intensity'], df_index),
            "flow_consistency": self.helper._normalize_series(df['flow_consistency_D'], df_index),
            "mf_efficiency": self.helper._normalize_series(raw_signals['mf_efficiency'], df_index),
            "accumulation_signal": self.helper._normalize_series(df['accumulation_score_D'], df_index),
            "inst_net_buy": self.helper._normalize_series(raw_signals['inst_buy_main'], df_index, bipolar=True)
        }
        covert_action_score = _robust_geometric_mean(scores, covert_action_weights, df_index)
        _temp_debug_values["covert_action_score"] = covert_action_score
        return covert_action_score

    def _calculate_chip_optimization_score(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_slope_accel_weights: Dict, chip_optimization_weights: Dict, method_name: str, _temp_debug_values: Dict) -> pd.Series:
        """
        【V3.1 · 多维深度探测版】计算筹码优化分数。
        - 逻辑：关注筹码成熟度。当长线筹码比例提升且现价处于均线密集成本区（Proximity）时，反转动力最强。
        """
        scores = {
            "chip_stability": self.helper._normalize_series(raw_signals['chip_stability'], df_index, bipolar=False),
            "long_term_ratio": self.helper._normalize_series(raw_signals['long_term_chip'], df_index, bipolar=False),
            "cost_ma_proximity": (1 - self.helper._normalize_series(raw_signals['cost_ma_diff'].abs(), df_index)),
            "chip_concentration": self.helper._normalize_series(raw_signals['chip_concentration'], df_index, ascending=False)
        }
        chip_optimization_score = _robust_geometric_mean(scores, chip_optimization_weights, df_index)
        _temp_debug_values["chip_optimization_score"] = chip_optimization_score
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
        【V2.12 · 微观订单流与结构共振版】统一打印隐蔽吸筹计算的调试信息。
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
