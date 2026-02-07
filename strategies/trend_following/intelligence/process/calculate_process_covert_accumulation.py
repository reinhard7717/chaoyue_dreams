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
    计算“隐蔽吸筹”的专属信号。
    PROCESS_META_COVERT_ACCUMULATION
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
        【V5.0 · 物理高阶导数全量版】计算基于斐波那契窗口的 SLOPE, ACCEL, JERK。
        - 逻辑：针对资金流和结构指标计算三阶物理导数，识别吸筹动能的非线性爆发。
        - 版本：5.0.0
        """
        fib_windows = [5, 8, 13, 21, 34, 55]
        derivative_targets = ['stealth_flow_ratio_D', 'SMART_MONEY_INST_NET_BUY_D', 'MA_POTENTIAL_COMPRESSION_RATE_D', 'VPA_MF_ADJUSTED_EFF_D', 'chip_stability_D']
        for base in derivative_targets:
            if base not in df.columns:
                continue
            for period in fib_windows:
                s_col = f'SLOPE_{period}_{base}'
                if s_col not in df.columns:
                    df[s_col] = ta.slope(df[base], length=period)
                a_col = f'ACCEL_{period}_{base}'
                if a_col not in df.columns and s_col in df.columns:
                    df[a_col] = ta.slope(df[s_col], length=period)
                j_col = f'JERK_{period}_{base}'
                if j_col not in df.columns and a_col in df.columns:
                    df[j_col] = ta.slope(df[a_col], length=period)
        for base in ['stealth_flow_ratio_D', 'SMART_MONEY_INST_NET_BUY_D']:
            if base in df.columns:
                for window in cumulative_flow_windows:
                    c_col = f'{base}_{window}d_sum'
                    if c_col not in df.columns:
                        df[c_col] = df[base].rolling(window, min_periods=1).sum()

    def _validate_and_get_raw_signals(self, df: pd.DataFrame, method_name: str, price_weakness_slope_window: int, low_volatility_bbw_window: int, mtf_slope_accel_weights: Dict, is_debug_enabled_for_method: bool, probe_ts: Optional[pd.Timestamp], _temp_debug_values: Dict, cumulative_flow_windows: List[int], cumulative_acc_windows: List[int]) -> Optional[Dict[str, pd.Series]]:
        """
        【V5.0 · 严密数据预检版】执行全量列存在性检查并构建语义映射。
        - 逻辑：在计算前强制验证军械库清单原始列，并动态注入高阶物理导数 Key。
        - 版本：5.0.0
        """
        essential_cols = [
            'IS_EMOTIONAL_EXTREME_D', 'BBW_21_2.0_D', 'MA_POTENTIAL_COMPRESSION_RATE_D',
            'IS_ROUNDING_BOTTOM_D', 'IS_GOLDEN_PIT_D', 'GEOM_ARC_CURVATURE_D',
            'afternoon_flow_ratio_D', 'closing_flow_intensity_D', 'flow_consistency_D',
            'long_term_chip_ratio_D', 'chip_stability_D', 'VPA_MF_ADJUSTED_EFF_D',
            'stealth_flow_ratio_D', 'SMART_MONEY_INST_NET_BUY_D', 'accumulation_score_D',
            'MA_POTENTIAL_TENSION_INDEX_D', 'market_sentiment_score_D', 'chip_concentration_ratio_D',
            'chip_convergence_ratio_D', 'winner_rate_D', 'chip_cost_to_ma21_diff_D', 'buy_lg_amount_rate_D'
        ]
        missing_cols = [c for c in essential_cols if c not in df.columns]
        if missing_cols:
            return None
        self._calculate_derived_signals(df, mtf_slope_accel_weights, cumulative_flow_windows, cumulative_acc_windows)
        raw_signals = {
            'emo_extreme': df['IS_EMOTIONAL_EXTREME_D'],
            'vol_bbw': df['BBW_21_2.0_D'],
            'ma_compression': df['MA_POTENTIAL_COMPRESSION_RATE_D'],
            'rounding_bottom': df['IS_ROUNDING_BOTTOM_D'],
            'golden_pit': df['IS_GOLDEN_PIT_D'],
            'arc_curvature': df['GEOM_ARC_CURVATURE_D'],
            'afternoon_flow': df['afternoon_flow_ratio_D'],
            'closing_intensity': df['closing_flow_intensity_D'],
            'flow_consistency': df['flow_consistency_D'],
            'long_term_chip': df['long_term_chip_ratio_D'],
            'chip_stability': df['chip_stability_D'],
            'mf_efficiency': df['VPA_MF_ADJUSTED_EFF_D'],
            'stealth_flow': df['stealth_flow_ratio_D'],
            'inst_buy': df['SMART_MONEY_INST_NET_BUY_D'],
            'acc_score': df['accumulation_score_D'],
            'structural_tension': df['MA_POTENTIAL_TENSION_INDEX_D'],
            'market_sentiment': df['market_sentiment_score_D'],
            'chip_concentration': df['chip_concentration_ratio_D'],
            'chip_convergence': df['chip_convergence_ratio_D'],
            'winner_rate': df['winner_rate_D'],
            'cost_ma_diff': df['chip_cost_to_ma21_diff_D'],
            'lg_buy_rate': df['buy_lg_amount_rate_D']
        }
        fib_windows = [5, 8, 13, 21, 34, 55]
        derivative_bases = ['stealth_flow_ratio_D', 'SMART_MONEY_INST_NET_BUY_D', 'MA_POTENTIAL_COMPRESSION_RATE_D']
        for base in derivative_bases:
            for p in fib_windows:
                for prefix in ['SLOPE', 'ACCEL', 'JERK']:
                    col_name = f'{prefix}_{p}_{base}'
                    col_key = f'{prefix.lower()}_{base}_{p}'
                    raw_signals[col_key] = df[col_name] if col_name in df.columns else pd.Series(0.0, index=df.index)
        return raw_signals

    def _calculate_market_context_score(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_slope_accel_weights: Dict, market_context_weights: Dict, price_weakness_slope_window: int, low_volatility_bbw_window: int, method_name: str, _temp_debug_values: Dict, neutral_range_threshold: float) -> pd.Series:
        """
        【V4.1 · 物理高阶导数修正版】计算市场背景评分。
        - 逻辑深化：利用修复后的 emo_extreme 捕捉情绪冰点，并结合均线压缩加速度识别横盘末端的结构爆发力。
        """
        # 引入均线压缩的加速度 (21日周期) 识别结构紧凑度变化
        accel_compression = raw_signals.get('accel_MA_POTENTIAL_COMPRESSION_RATE_D_21', pd.Series(0.0, index=df_index))
        scores = {
            "sentiment_extreme": self.helper._normalize_series(raw_signals['emo_extreme'], df_index, bipolar=False),
            "vol_compression": self.helper._normalize_series(raw_signals['vol_bbw'], df_index, ascending=False),
            "rounding_bottom": self.helper._normalize_series(raw_signals['rounding_bottom'], df_index, bipolar=False),
            "golden_pit": self.helper._normalize_series(raw_signals['golden_pit'], df_index, bipolar=False),
            "arc_curvature": self.helper._normalize_series(raw_signals['arc_curvature'], df_index, bipolar=False),
            "is_consolidating": self.helper._normalize_series(raw_signals['ma_compression'], df_index, bipolar=False),
            "structural_tension": (1 - self.helper._normalize_series(raw_signals['structural_tension'], df_index)),
            "compression_accel": self.helper._normalize_series(accel_compression, df_index, bipolar=True)
        }
        # 更新权重配置以匹配新增的 compression_accel (若配置中未定义则通过 helper 容错)
        market_context_score = _robust_geometric_mean(scores, market_context_weights, df_index)
        _temp_debug_values["market_context_v41"] = scores
        return market_context_score

    def _calculate_covert_action_score(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_slope_accel_weights: Dict, covert_action_weights: Dict, method_name: str, _temp_debug_values: Dict, cumulative_flow_windows: List[int], cumulative_flow_weights: Dict, cumulative_acc_windows: List[int], cumulative_acc_weights: Dict, daily_mf_flow_weight: float, cumulative_mf_flow_weight: float, daily_acc_weight: float, cumulative_acc_weight: float, new_raw_signals_weights: Dict, main_force_accumulation_resonance_weight: float, new_raw_signals_weights_v2: Dict, covert_order_flow_resonance_weight: float) -> pd.Series:
        """
        【V5.0 · 物理突变修复版】计算隐蔽行动分数。
        - 逻辑：对齐 19 个位置参数。集成斐波那契 JERK 评分，识别资金流的瞬时爆发。
        - 版本：5.0.0
        """
        fib_windows = [5, 8, 13, 21, 34, 55]
        jerk_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        for p in fib_windows:
            key = f'jerk_stealth_flow_ratio_D_{p}'
            if key in raw_signals:
                jerk_score += self.helper._normalize_series(raw_signals[key], df_index, bipolar=True)
        jerk_score = (jerk_score / len(fib_windows) + 1) / 2
        scores = {
            "stealth_ops": self.helper._normalize_series(raw_signals['stealth_flow'], df_index, bipolar=False),
            "inst_net_buy": self.helper._normalize_series(raw_signals['inst_buy'], df_index, bipolar=True),
            "stealth_flow_jerk": jerk_score,
            "afternoon_bias": self.helper._normalize_series(raw_signals['afternoon_flow'], df_index, bipolar=False),
            "closing_intensity": self.helper._normalize_series(raw_signals['closing_intensity'], df_index, bipolar=False),
            "mf_efficiency": self.helper._normalize_series(raw_signals['mf_efficiency'], df_index, bipolar=False),
            "flow_consistency": self.helper._normalize_series(raw_signals['flow_consistency'], df_index, bipolar=False),
            "contextualized_hidden_accumulation": self.helper._normalize_series(raw_signals['acc_score'], df_index, bipolar=False)
        }
        current_weights = covert_action_weights.copy()
        if "stealth_flow_jerk" not in current_weights:
            current_weights["stealth_flow_jerk"] = 0.12
        covert_action_score = _robust_geometric_mean(scores, current_weights, df_index)
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
