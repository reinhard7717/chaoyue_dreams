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

class CalculateSplitOrderAccumulation:
    """
    【V4.0.2 · 拆单吸筹强度 · 探针强化与问题暴露版】
    - 核心修正: 严格区分原始数据及其MTF衍生与原子信号。原子信号的趋势通过直接diff()计算，
                避免在df中查找不存在的MTF衍生列。
    - 核心升级: 引入动态效率基准线，增强价格行为捕捉，精细化欺诈意图识别，MTF核心信号增强，
                情境自适应权重调整，非线性融合强化，趋势动量diff()化。
    - 探针强化: 增加关键中间计算节点的详细探针，特别是针对_robust_geometric_mean的输入和输出，
                以及最终pow()操作的精确值，以暴露潜在的计算偏差或bug。
    """
    def __init__(self, strategy_instance, helper: ProcessIntelligenceHelper):
        """
        初始化拆单吸筹强度计算器。
        参数:
            strategy_instance: 策略实例，用于访问全局配置和原子状态。
            helper: ProcessIntelligenceHelper 实例，用于访问辅助方法。
        """
        self.strategy = strategy_instance
        self.helper = helper
        # 从 helper 获取参数
        self.params = self.helper.params
        self.debug_params = self.helper.debug_params
        self.probe_dates = self.helper.probe_dates
        # 获取MTF权重配置
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        self.actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # 默认的MTF斜率/加速度权重，可在config中覆盖
        self.default_mtf_slope_accel_weights = {"slope_periods": {"5": 0.6, "13": 0.4}, "accel_periods": {"5": 0.7, "13": 0.3}}

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V4.0.3 · 拆单吸筹强度 · 探针激活版】计算“拆单吸筹强度”的专属信号。
        - 核心修正: 确保调试信息 (is_debug_enabled_for_method, probe_ts) 被正确传递到
                    _calculate_holographic_validation 方法，从而激活 _robust_geometric_mean 内部的探针。
        - 核心升级: 引入动态效率基准线，增强价格行为捕捉，精细化欺诈意图识别，MTF核心信号增强，
                    情境自适应权重调整，非线性融合强化，趋势动量diff()化。
        - 探针强化: 增加关键中间计算节点的详细探针，特别是针对_robust_geometric_mean的输入和输出，
                    以及最终pow()操作的精确值，以暴露潜在的计算偏差或bug。
        """
        method_name = "calculate_split_order_accumulation"
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
        _temp_debug_values = {}
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算拆单吸筹强度..."] = ""
        df_index = df.index
        mtf_slope_accel_weights = config.get('mtf_slope_accel_weights', self.default_mtf_slope_accel_weights)
        # 1. 获取并归一化所有信号
        raw_signals, normalized_signals, mtf_signals, context_signals = self._get_and_normalize_signals(df, mtf_slope_accel_weights, method_name)
        _temp_debug_values["原始信号值"] = raw_signals
        _temp_debug_values["归一化处理"] = normalized_signals
        _temp_debug_values["MTF信号"] = mtf_signals
        _temp_debug_values["情境信号"] = context_signals
        # 2. 动态效率基准线
        dynamic_efficiency_baseline, baseline_debug_values = self._calculate_dynamic_efficiency_baseline(context_signals, df_index, config)
        _temp_debug_values["动态效率基准线"] = baseline_debug_values
        _temp_debug_values["动态效率基准线"]["dynamic_efficiency_baseline"] = dynamic_efficiency_baseline
        # 3. 计算初步分数 (dynamic_preliminary_score)
        dynamic_preliminary_score, preliminary_debug_values = self._calculate_preliminary_score(normalized_signals, mtf_signals, context_signals, df_index, config)
        _temp_debug_values["初步计算"] = preliminary_debug_values
        _temp_debug_values["初步计算"]["dynamic_preliminary_score"] = dynamic_preliminary_score
        # 4. 计算全息验证分数 (holographic_validation_score)
        # 传递 is_debug_enabled_for_method 和 probe_ts
        # 修复：将 mtf_signals 传递给 _calculate_holographic_validation
        holographic_validation_score, holographic_debug_values = self._calculate_holographic_validation(df, raw_signals, normalized_signals, mtf_signals, context_signals, df_index, config, is_debug_enabled_for_method, probe_ts)
        _temp_debug_values["全息验证"] = holographic_debug_values
        _temp_debug_values["全息验证"]["holographic_validation_score"] = holographic_validation_score
        # 5. 应用质效校准并计算最终分数
        final_score, final_score_debug_values = self._apply_quality_efficiency_calibration(dynamic_preliminary_score, holographic_validation_score, dynamic_efficiency_baseline, probe_ts)
        _temp_debug_values["最终分数"] = final_score_debug_values
        _temp_debug_values["最终分数"]["final_score"] = final_score
        # --- 统一输出调试信息 ---
        if is_debug_enabled_for_method and probe_ts:
            self._print_debug_info(method_name, probe_ts, debug_output, _temp_debug_values, final_score)
        return final_score.astype(np.float32)

    def _get_and_normalize_signals(self, df: pd.DataFrame, mtf_slope_accel_weights: Dict, method_name: str) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series], Dict[str, pd.Series], Dict[str, pd.Series]]:
        """
        【V4.5 · 流动性与执行质量增强版】
        获取并归一化所有拆单吸筹强度计算所需的原始信号。
        - 核心增强: 引入新的流动性复合信号 (data_liquidity_outcome)。
        - 核心优化: 增强订单流复合信号 (data_order_flow_outcome) 纳入执行质量指标。
        - 新增: 针对新增原始指标计算移动平均、斜率和加速度。
        返回: (raw_signals, normalized_signals, mtf_signals, context_signals)
        """
        df_index = df.index
        # 明确列出所有需要从 df 中获取的原始数据层信号
        raw_df_columns = [
            'hidden_accumulation_intensity_D', 'SLOPE_5_close_D', 'SLOPE_13_close_D', 'SLOPE_21_close_D',
            'ACCEL_5_close_D', 'ACCEL_13_close_D', 'deception_index_D',
            'upward_impulse_purity_D', 'main_force_net_flow_calibrated_D', 'SMART_MONEY_INST_NET_BUY_D',
            'conviction_flow_index_D', 'chip_health_score_D', 'structural_potential_score_D',
            'MA_POTENTIAL_ORDERLINESS_SCORE_D', 'control_solidity_index_D', 'THEME_HOTNESS_SCORE_D',
            'VOLATILITY_INSTABILITY_INDEX_21d_D', 'ADX_14_D', 'is_consolidating_D',
            'dynamic_consolidation_duration_D', 'deception_lure_long_intensity_D', 'deception_lure_short_intensity_D',
            'trend_vitality_index_D', 'close_D',
            # 订单流相关原始指标 (已存在，但新增执行质量)
            'main_force_buy_ofi_D', 'main_force_sell_ofi_D', 'buy_flow_efficiency_index_D',
            'sell_flow_efficiency_index_D', 'order_book_imbalance_D',
            'main_force_execution_alpha_D', 'main_force_slippage_index_D', # 新增执行质量
            # 筹码结构相关原始指标 (已存在)
            'cost_dispersion_index_D', 'winner_concentration_90pct_D', 'loser_concentration_90pct_D',
            'chip_fatigue_index_D',
            # 新增流动性相关原始指标
            'bid_side_liquidity_D', 'ask_side_liquidity_D', 'order_book_liquidity_supply_D'
        ]
        # 动态添加数据层原始信号的MTF斜率和加速度到required_signals
        base_signals_for_mtf_from_df = [
            'hidden_accumulation_intensity_D', 'close_D', 'deception_index_D',
            'THEME_HOTNESS_SCORE_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D', 'ADX_14_D',
            'deception_lure_long_intensity_D', 'deception_lure_short_intensity_D',
            'trend_vitality_index_D', 'main_force_net_flow_calibrated_D', 'chip_health_score_D',
            # 订单流和筹码结构信号 (已存在)
            'main_force_buy_ofi_D', 'main_force_sell_ofi_D', 'buy_flow_efficiency_index_D',
            'sell_flow_efficiency_index_D', 'order_book_imbalance_D',
            'main_force_execution_alpha_D', 'main_force_slippage_index_D', # 新增执行质量
            'cost_dispersion_index_D', 'winner_concentration_90pct_D', 'loser_concentration_90pct_D',
            'chip_fatigue_index_D',
            # 新增流动性信号
            'bid_side_liquidity_D', 'ask_side_liquidity_D', 'order_book_liquidity_supply_D'
        ]
        required_signals = list(raw_df_columns)
        for base_sig in base_signals_for_mtf_from_df:
            for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
                required_signals.append(f'SLOPE_{period_str}_{base_sig}')
            for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
                required_signals.append(f'ACCEL_{period_str}_{base_sig}')
        # 获取MTF协同性所需的基础信号列表
        mtf_cohesion_base_signals = get_param_value(self.params.get('mtf_cohesion_base_signals'), [])
        required_signals.extend(mtf_cohesion_base_signals) # 将协同性基础信号也加入到校验列表
        if not self.helper._validate_required_signals(df, required_signals, method_name):
            print(f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。")
            return {}, {}, {}, {}
        # 原始数据获取 (全部来自数据层)
        raw_signals_dict = {}
        for col in raw_df_columns:
            raw_signals_dict[col] = self.helper._get_safe_series(df, col, 0.0, method_name=method_name)
        
        # 引入历史周期性考量：计算关键信号的5日、13日和21日移动平均
        raw_signals_dict['hidden_accumulation_intensity_MA5_D'] = raw_signals_dict['hidden_accumulation_intensity_D'].rolling(5).mean().fillna(0.0)
        raw_signals_dict['hidden_accumulation_intensity_MA13_D'] = raw_signals_dict['hidden_accumulation_intensity_D'].rolling(13).mean().fillna(0.0)
        raw_signals_dict['hidden_accumulation_intensity_MA21_D'] = raw_signals_dict['hidden_accumulation_intensity_D'].rolling(21).mean().fillna(0.0)
        raw_signals_dict['main_force_net_flow_calibrated_MA5_D'] = raw_signals_dict['main_force_net_flow_calibrated_D'].rolling(5).mean().fillna(0.0)
        raw_signals_dict['main_force_net_flow_calibrated_MA13_D'] = raw_signals_dict['main_force_net_flow_calibrated_D'].rolling(13).mean().fillna(0.0)
        raw_signals_dict['main_force_net_flow_calibrated_MA21_D'] = raw_signals_dict['main_force_net_flow_calibrated_D'].rolling(21).mean().fillna(0.0)
        raw_signals_dict['chip_health_score_MA5_D'] = raw_signals_dict['chip_health_score_D'].rolling(5).mean().fillna(0.0)
        raw_signals_dict['chip_health_score_MA13_D'] = raw_signals_dict['chip_health_score_D'].rolling(13).mean().fillna(0.0)
        raw_signals_dict['structural_potential_score_MA5_D'] = raw_signals_dict['structural_potential_score_D'].rolling(5).mean().fillna(0.0)
        raw_signals_dict['structural_potential_score_MA13_D'] = raw_signals_dict['structural_potential_score_D'].rolling(13).mean().fillna(0.0)
        raw_signals_dict['THEME_HOTNESS_SCORE_MA5_D'] = raw_signals_dict['THEME_HOTNESS_SCORE_D'].rolling(5).mean().fillna(0.0)
        raw_signals_dict['THEME_HOTNESS_SCORE_MA13_D'] = raw_signals_dict['THEME_HOTNESS_SCORE_D'].rolling(13).mean().fillna(0.0)
        # 订单流和筹码结构相关MA (已存在)
        raw_signals_dict['main_force_buy_ofi_MA5_D'] = raw_signals_dict['main_force_buy_ofi_D'].rolling(5).mean().fillna(0.0)
        raw_signals_dict['main_force_buy_ofi_MA13_D'] = raw_signals_dict['main_force_buy_ofi_D'].rolling(13).mean().fillna(0.0)
        raw_signals_dict['buy_flow_efficiency_index_MA5_D'] = raw_signals_dict['buy_flow_efficiency_index_D'].rolling(5).mean().fillna(0.0)
        raw_signals_dict['order_book_imbalance_MA5_D'] = raw_signals_dict['order_book_imbalance_D'].rolling(5).mean().fillna(0.0)
        raw_signals_dict['cost_dispersion_index_MA5_D'] = raw_signals_dict['cost_dispersion_index_D'].rolling(5).mean().fillna(0.0)
        raw_signals_dict['winner_concentration_90pct_MA5_D'] = raw_signals_dict['winner_concentration_90pct_D'].rolling(5).mean().fillna(0.0)
        raw_signals_dict['loser_concentration_90pct_MA5_D'] = raw_signals_dict['loser_concentration_90pct_D'].rolling(5).mean().fillna(0.0)
        raw_signals_dict['chip_fatigue_index_MA5_D'] = raw_signals_dict['chip_fatigue_index_D'].rolling(5).mean().fillna(0.0)
        # 新增：执行质量相关MA
        raw_signals_dict['main_force_execution_alpha_MA5_D'] = raw_signals_dict['main_force_execution_alpha_D'].rolling(5).mean().fillna(0.0)
        raw_signals_dict['main_force_slippage_index_MA5_D'] = raw_signals_dict['main_force_slippage_index_D'].rolling(5).mean().fillna(0.0)
        # 新增：流动性相关MA
        raw_signals_dict['bid_side_liquidity_MA5_D'] = raw_signals_dict['bid_side_liquidity_D'].rolling(5).mean().fillna(0.0)
        raw_signals_dict['ask_side_liquidity_MA5_D'] = raw_signals_dict['ask_side_liquidity_D'].rolling(5).mean().fillna(0.0)
        raw_signals_dict['order_book_liquidity_supply_MA5_D'] = raw_signals_dict['order_book_liquidity_supply_D'].rolling(5).mean().fillna(0.0)
        # 归一化处理 (针对原始信号或其简单衍生)
        normalized_signals = {}
        context_signals = {}
        # 基础归一化信号
        normalized_signals["normalized_score"] = (raw_signals_dict['hidden_accumulation_intensity_D'] / 100).clip(0, 1)
        normalized_signals["price_trend_norm"] = self.helper._normalize_series(raw_signals_dict['SLOPE_5_close_D'], df_index, bipolar=True)
        normalized_signals["SLOPE_13_close_D"] = self.helper._normalize_series(raw_signals_dict['SLOPE_13_close_D'], df_index, bipolar=True)
        normalized_signals["SLOPE_21_close_D"] = self.helper._normalize_series(raw_signals_dict['SLOPE_21_close_D'], df_index, bipolar=True)
        normalized_signals["upward_purity"] = self.helper._normalize_series(raw_signals_dict['upward_impulse_purity_D'], df_index, bipolar=False)
        normalized_signals["deception_norm"] = self.helper._normalize_series(raw_signals_dict['deception_index_D'], df_index, bipolar=True)
        # 构建新的数据层复合“原子信号”
        # 1. 数据层资金流 (data_flow_outcome) - 保持不变
        norm_mf_net_flow = self.helper._normalize_series(raw_signals_dict['main_force_net_flow_calibrated_D'], df_index, bipolar=True)
        norm_mf_net_flow_ma5 = self.helper._normalize_series(raw_signals_dict['main_force_net_flow_calibrated_MA5_D'], df_index, bipolar=True)
        norm_mf_net_flow_ma13 = self.helper._normalize_series(raw_signals_dict['main_force_net_flow_calibrated_MA13_D'], df_index, bipolar=True)
        norm_hidden_acc_intensity = self.helper._normalize_series(raw_signals_dict['hidden_accumulation_intensity_D'], df_index, bipolar=False)
        norm_hidden_acc_intensity_ma5 = self.helper._normalize_series(raw_signals_dict['hidden_accumulation_intensity_MA5_D'], df_index, bipolar=False)
        norm_hidden_acc_intensity_ma13 = self.helper._normalize_series(raw_signals_dict['hidden_accumulation_intensity_MA13_D'], df_index, bipolar=False)
        norm_smart_money_net_buy = self.helper._normalize_series(raw_signals_dict['SMART_MONEY_INST_NET_BUY_D'], df_index, bipolar=True)
        norm_conviction_flow = self.helper._normalize_series(raw_signals_dict['conviction_flow_index_D'], df_index, bipolar=False)
        flow_weights = get_param_value(self.params.get('data_flow_composite_weights'), {
            'daily_mf_net_flow': 0.2, 'mf_net_flow_5d_avg': 0.15, 'mf_net_flow_13d_avg': 0.15,
            'daily_hidden_acc_intensity': 0.1, 'hidden_acc_5d_avg': 0.1, 'hidden_acc_13d_avg': 0.1,
            'smart_money_net_buy': 0.1, 'conviction_flow': 0.1
        })
        data_flow_components = {
            "daily_mf_net_flow": norm_mf_net_flow,
            "mf_net_flow_5d_avg": norm_mf_net_flow_ma5,
            "mf_net_flow_13d_avg": norm_mf_net_flow_ma13,
            "daily_hidden_acc_intensity": norm_hidden_acc_intensity,
            "hidden_acc_5d_avg": norm_hidden_acc_intensity_ma5,
            "hidden_acc_13d_avg": norm_hidden_acc_intensity_ma13,
            "smart_money_net_buy": norm_smart_money_net_buy,
            "conviction_flow": norm_conviction_flow,
        }
        normalized_signals["data_flow_outcome"] = _robust_geometric_mean(data_flow_components, flow_weights, df_index).fillna(0.0)
        # 2. 数据层筹码结构 (data_structure_outcome) - 保持不变
        norm_chip_health = self.helper._normalize_series(raw_signals_dict['chip_health_score_D'], df_index, bipolar=False)
        norm_chip_health_ma5 = self.helper._normalize_series(raw_signals_dict['chip_health_score_MA5_D'], df_index, bipolar=False)
        norm_chip_health_ma13 = self.helper._normalize_series(raw_signals_dict['chip_health_score_MA13_D'], df_index, bipolar=False)
        norm_structural_potential = self.helper._normalize_series(raw_signals_dict['structural_potential_score_D'], df_index, bipolar=False)
        norm_structural_potential_ma5 = self.helper._normalize_series(raw_signals_dict['structural_potential_score_MA5_D'], df_index, bipolar=False)
        norm_structural_potential_ma13 = self.helper._normalize_series(raw_signals_dict['structural_potential_score_MA13_D'], df_index, bipolar=False)
        norm_control_solidity = self.helper._normalize_series(raw_signals_dict['control_solidity_index_D'], df_index, bipolar=False)
        structure_weights = get_param_value(self.params.get('data_structure_composite_weights'), {
            'daily_chip_health': 0.25, 'chip_health_5d_avg': 0.15, 'chip_health_13d_avg': 0.15,
            'daily_structural_potential': 0.15, 'structural_potential_5d_avg': 0.1, 'structural_potential_13d_avg': 0.1,
            'control_solidity': 0.1
        })
        data_structure_components = {
            "daily_chip_health": norm_chip_health,
            "chip_health_5d_avg": norm_chip_health_ma5,
            "chip_health_13d_avg": norm_chip_health_ma13,
            "daily_structural_potential": norm_structural_potential,
            "structural_potential_5d_avg": norm_structural_potential_ma5,
            "structural_potential_13d_avg": norm_structural_potential_ma13,
            "control_solidity": norm_control_solidity,
        }
        normalized_signals["data_structure_outcome"] = _robust_geometric_mean(data_structure_components, structure_weights, df_index).fillna(0.0)
        # 3. 数据层动态潜力 (data_potential_outcome) - 保持不变
        norm_ma_potential_orderliness = self.helper._normalize_series(raw_signals_dict['MA_POTENTIAL_ORDERLINESS_SCORE_D'], df_index, bipolar=False)
        norm_trend_vitality = self.helper._normalize_series(raw_signals_dict['trend_vitality_index_D'], df_index, bipolar=False)
        
        potential_weights = get_param_value(self.params.get('data_potential_composite_weights'), {
            'ma_potential_orderliness': 0.4, 'structural_potential': 0.3, 'trend_vitality': 0.3
        })
        data_potential_components = {
            "ma_potential_orderliness": norm_ma_potential_orderliness,
            "structural_potential": norm_structural_potential, # 复用上面的归一化结构潜力
            "trend_vitality": norm_trend_vitality,
        }
        normalized_signals["data_potential_outcome"] = _robust_geometric_mean(data_potential_components, potential_weights, df_index).fillna(0.0)
        # 4. 数据层订单流结果 (data_order_flow_outcome) - 增强
        norm_mf_buy_ofi = self.helper._normalize_series(raw_signals_dict['main_force_buy_ofi_D'], df_index, bipolar=True)
        norm_mf_buy_ofi_ma5 = self.helper._normalize_series(raw_signals_dict['main_force_buy_ofi_MA5_D'], df_index, bipolar=True)
        norm_mf_buy_ofi_ma13 = self.helper._normalize_series(raw_signals_dict['main_force_buy_ofi_MA13_D'], df_index, bipolar=True)
        norm_buy_flow_efficiency = self.helper._normalize_series(raw_signals_dict['buy_flow_efficiency_index_D'], df_index, bipolar=False)
        norm_order_book_imbalance = self.helper._normalize_series(raw_signals_dict['order_book_imbalance_D'], df_index, bipolar=True)
        norm_mf_exec_alpha = self.helper._normalize_series(raw_signals_dict['main_force_execution_alpha_D'], df_index, bipolar=True) # 新增
        norm_mf_exec_alpha_ma5 = self.helper._normalize_series(raw_signals_dict['main_force_execution_alpha_MA5_D'], df_index, bipolar=True) # 新增
        norm_mf_slippage = self.helper._normalize_series(raw_signals_dict['main_force_slippage_index_D'], df_index, bipolar=False, ascending=False) # 滑点越低越好 # 新增
        norm_mf_slippage_ma5 = self.helper._normalize_series(raw_signals_dict['main_force_slippage_index_MA5_D'], df_index, bipolar=False, ascending=False) # 新增
        order_flow_weights = get_param_value(self.params.get('data_order_flow_composite_weights'), {
            'mf_buy_ofi': 0.2, 'mf_buy_ofi_5d_avg': 0.15, 'mf_buy_ofi_13d_avg': 0.1,
            'buy_flow_efficiency': 0.15, 'order_book_imbalance': 0.1,
            'mf_exec_alpha': 0.1, 'mf_exec_alpha_5d_avg': 0.05,
            'mf_slippage_inverted': 0.1, 'mf_slippage_inverted_5d_avg': 0.05
        })
        data_order_flow_components = {
            "mf_buy_ofi": norm_mf_buy_ofi,
            "mf_buy_ofi_5d_avg": norm_mf_buy_ofi_ma5,
            "mf_buy_ofi_13d_avg": norm_mf_buy_ofi_ma13,
            "buy_flow_efficiency": norm_buy_flow_efficiency,
            "order_book_imbalance": norm_order_book_imbalance,
            "mf_exec_alpha": norm_mf_exec_alpha, # 新增
            "mf_exec_alpha_5d_avg": norm_mf_exec_alpha_ma5, # 新增
            "mf_slippage_inverted": norm_mf_slippage, # 新增
            "mf_slippage_inverted_5d_avg": norm_mf_slippage_ma5 # 新增
        }
        normalized_signals["data_order_flow_outcome"] = _robust_geometric_mean(data_order_flow_components, order_flow_weights, df_index).fillna(0.0)
        # 5. 数据层筹码结构结果 (data_chip_structure_outcome) - 保持不变
        norm_cost_dispersion = self.helper._normalize_series(raw_signals_dict['cost_dispersion_index_D'], df_index, bipolar=False, ascending=False) # 成本分散度越低越好
        norm_cost_dispersion_ma5 = self.helper._normalize_series(raw_signals_dict['cost_dispersion_index_MA5_D'], df_index, bipolar=False, ascending=False)
        norm_winner_concentration = self.helper._normalize_series(raw_signals_dict['winner_concentration_90pct_D'], df_index, bipolar=False)
        norm_loser_concentration = self.helper._normalize_series(raw_signals_dict['loser_concentration_90pct_D'], df_index, bipolar=False, ascending=False) # 亏损筹码集中度越低越好
        norm_chip_fatigue = self.helper._normalize_series(raw_signals_dict['chip_fatigue_index_D'], df_index, bipolar=False, ascending=False) # 筹码疲劳度越低越好
        chip_structure_weights = get_param_value(self.params.get('data_chip_structure_composite_weights'), {
            'cost_dispersion': 0.3, 'cost_dispersion_5d_avg': 0.2,
            'winner_concentration': 0.2, 'loser_concentration': 0.1,
            'chip_fatigue': 0.2
        })
        data_chip_structure_components = {
            "cost_dispersion": norm_cost_dispersion,
            "cost_dispersion_5d_avg": norm_cost_dispersion_ma5,
            "winner_concentration": norm_winner_concentration,
            "loser_concentration": norm_loser_concentration,
            "chip_fatigue": norm_chip_fatigue,
        }
        normalized_signals["data_chip_structure_outcome"] = _robust_geometric_mean(data_chip_structure_components, chip_structure_weights, df_index).fillna(0.0)
        # 6. 新增：数据层流动性结果 (data_liquidity_outcome)
        norm_bid_liquidity = self.helper._normalize_series(raw_signals_dict['bid_side_liquidity_D'], df_index, bipolar=False)
        norm_bid_liquidity_ma5 = self.helper._normalize_series(raw_signals_dict['bid_side_liquidity_MA5_D'], df_index, bipolar=False)
        norm_ask_liquidity_inverted = self.helper._normalize_series(raw_signals_dict['ask_side_liquidity_D'], df_index, bipolar=False, ascending=False) # 卖盘流动性越高，吸筹难度越大，所以反向
        norm_ask_liquidity_inverted_ma5 = self.helper._normalize_series(raw_signals_dict['ask_side_liquidity_MA5_D'], df_index, bipolar=False, ascending=False)
        norm_liquidity_supply = self.helper._normalize_series(raw_signals_dict['order_book_liquidity_supply_D'], df_index, bipolar=False)
        norm_liquidity_supply_ma5 = self.helper._normalize_series(raw_signals_dict['order_book_liquidity_supply_MA5_D'], df_index, bipolar=False)
        
        liquidity_weights = get_param_value(self.params.get('data_liquidity_composite_weights'), {
            'bid_liquidity': 0.3, 'bid_liquidity_5d_avg': 0.2,
            'ask_liquidity_inverted': 0.2, 'ask_liquidity_inverted_5d_avg': 0.1,
            'liquidity_supply': 0.1, 'liquidity_supply_5d_avg': 0.1
        })
        data_liquidity_components = {
            "bid_liquidity": norm_bid_liquidity,
            "bid_liquidity_5d_avg": norm_bid_liquidity_ma5,
            "ask_liquidity_inverted": norm_ask_liquidity_inverted,
            "ask_liquidity_inverted_5d_avg": norm_ask_liquidity_inverted_ma5,
            "liquidity_supply": norm_liquidity_supply,
            "liquidity_supply_5d_avg": norm_liquidity_supply_ma5,
        }
        normalized_signals["data_liquidity_outcome"] = _robust_geometric_mean(data_liquidity_components, liquidity_weights, df_index).fillna(0.0)

        # --- 计算数据层复合信号的斜率和加速度，用于RDI分析 ---
        rdi_periods = [5, 13, 21]
        composite_signals_for_rdi = {
            "data_flow_outcome": normalized_signals["data_flow_outcome"],
            "data_structure_outcome": normalized_signals["data_structure_outcome"],
            "data_potential_outcome": normalized_signals["data_potential_outcome"],
            "data_order_flow_outcome": normalized_signals["data_order_flow_outcome"],
            "data_chip_structure_outcome": normalized_signals["data_chip_structure_outcome"],
            "data_liquidity_outcome": normalized_signals["data_liquidity_outcome"] # 新增
        }
        for sig_name, sig_series in composite_signals_for_rdi.items():
            for period in rdi_periods:
                # 计算斜率
                slope_series = self.helper._calculate_slope_series(sig_series, period)
                normalized_signals[f'SLOPE_{period}_{sig_name}'] = self.helper._normalize_series(slope_series, df_index, bipolar=True)
                # 计算加速度
                accel_series = self.helper._calculate_accel_series(sig_series, period)
                normalized_signals[f'ACCEL_{period}_{sig_name}'] = self.helper._normalize_series(accel_series, df_index, bipolar=True)
        # MTF信号 (仅针对数据层原始信号)
        mtf_signals = {
            "mtf_hidden_accumulation_intensity": self.helper._get_mtf_slope_accel_score(df, 'hidden_accumulation_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            "mtf_price_trend": self.helper._get_mtf_slope_accel_score(df, 'close_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True, periods=[5, 13, 21]),
            "mtf_deception_index": self.helper._get_mtf_slope_accel_score(df, 'deception_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True),
            "mtf_price_trend_5": self.helper._get_mtf_slope_accel_score(df, 'close_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True, periods=[5]),
            "mtf_price_trend_13": self.helper._get_mtf_slope_accel_score(df, 'close_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True, periods=[13]),
            "mtf_price_trend_21": self.helper._get_mtf_slope_accel_score(df, 'close_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True, periods=[21]),
            # MTF协同性分数
            "mtf_cohesion_score": self.helper._get_mtf_cohesion_score(df, mtf_cohesion_base_signals, mtf_slope_accel_weights, df_index, method_name)
        }
        # 情境信号 (使用新的数据层情绪信号)
        norm_theme_hotness = self.helper._normalize_series(raw_signals_dict['THEME_HOTNESS_SCORE_D'], df_index, bipolar=False)
        norm_theme_hotness_ma5 = self.helper._normalize_series(raw_signals_dict['THEME_HOTNESS_SCORE_MA5_D'], df_index, bipolar=False)
        norm_theme_hotness_ma13 = self.helper._normalize_series(raw_signals_dict['THEME_HOTNESS_SCORE_MA13_D'], df_index, bipolar=False)
        
        sentiment_weights = get_param_value(self.params.get('data_sentiment_composite_weights'), {
            'daily_theme_hotness': 0.5, 'theme_hotness_5d_avg': 0.3, 'theme_hotness_13d_avg': 0.2
        })
        data_sentiment_components = {
            "daily_theme_hotness": norm_theme_hotness,
            "theme_hotness_5d_avg": norm_theme_hotness_ma5,
            "theme_hotness_13d_avg": norm_theme_hotness_ma13,
        }
        context_signals["market_sentiment_norm"] = _robust_geometric_mean(data_sentiment_components, sentiment_weights, df_index).fillna(0.0)
        context_signals["volatility_instability_norm"] = self.helper._normalize_series(raw_signals_dict['VOLATILITY_INSTABILITY_INDEX_21d_D'], df_index, bipolar=False)
        context_signals["adx_norm"] = self.helper._normalize_series(raw_signals_dict['ADX_14_D'], df_index, bipolar=False)
        context_signals["is_consolidating_norm"] = self.helper._normalize_series(raw_signals_dict['is_consolidating_D'], df_index, bipolar=False)
        context_signals["dynamic_consolidation_duration_norm"] = self.helper._normalize_series(raw_signals_dict['dynamic_consolidation_duration_D'], df_index, bipolar=False)
        context_signals["deception_lure_long_norm"] = self.helper._normalize_series(raw_signals_dict['deception_lure_long_intensity_D'], df_index, bipolar=False)
        context_signals["deception_lure_short_norm"] = self.helper._normalize_series(raw_signals_dict['deception_lure_short_intensity_D'], df_index, bipolar=False)
        context_signals["trend_vitality_norm"] = self.helper._normalize_series(raw_signals_dict['trend_vitality_index_D'], df_index, bipolar=False)
        raw_signals = raw_signals_dict
        return raw_signals, normalized_signals, mtf_signals, context_signals

    def _calculate_holographic_validation(self, df: pd.DataFrame, raw_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], context_signals: Dict[str, pd.Series], df_index: pd.Index, config: Dict, is_debug_enabled_for_method: bool, probe_ts: Optional[pd.Timestamp]) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """
        【V4.5 · 流动性与执行质量增强版】
        计算全息验证分数。
        - 核心增强: 引入 `data_liquidity_outcome` 复合信号，并全面集成到状态和趋势融合中。
        - 核心优化: 实施情境自适应RDI权重调整，根据市场趋势强度和波动率动态调整奖励/惩罚因子。
        """
        holographic_debug_values = {}
        params = config.get('holographic_validation_params', {})
        # 调整默认权重以包含新的复合信号
        state_fusion_weights = get_param_value(params.get('state_fusion_weights'), {"flow": 0.2, "structure": 0.2, "potential": 0.2, "order_flow": 0.15, "chip_structure": 0.15, "liquidity": 0.1})
        trend_fusion_weights = get_param_value(params.get('trend_fusion_weights'), {"flow_trend": 0.2, "structure_trend": 0.2, "potential_trend": 0.2, "order_flow_trend": 0.15, "chip_structure_trend": 0.15, "liquidity_trend": 0.1})
        overall_fusion_weights = get_param_value(params.get('overall_fusion_weights'), {"state": 0.6, "trend": 0.4})
        
        # 使用新的数据层复合信号作为“原子信号”
        data_flow_outcome = normalized_signals["data_flow_outcome"]
        data_structure_outcome = normalized_signals["data_structure_outcome"]
        data_potential_outcome = normalized_signals["data_potential_outcome"]
        data_order_flow_outcome = normalized_signals["data_order_flow_outcome"]
        data_chip_structure_outcome = normalized_signals["data_chip_structure_outcome"]
        data_liquidity_outcome = normalized_signals["data_liquidity_outcome"] # 新增
        # 情境自适应权重调整
        market_sentiment_norm = context_signals["market_sentiment_norm"]
        volatility_instability_norm = context_signals["volatility_instability_norm"]
        adx_norm = context_signals["adx_norm"] # 趋势强度
        
        sentiment_factor = (market_sentiment_norm + 1) / 2 # [0, 1]
        volatility_factor = 1 - volatility_instability_norm # [0, 1] (稳定性)
        trend_strength_factor = adx_norm # [0, 1]
        
        # 动态调整 state_fusion_weights
        dynamic_state_fusion_weights = state_fusion_weights.copy()
        # 现有信号的动态调整
        dynamic_state_fusion_weights["flow"] = dynamic_state_fusion_weights["flow"] * (1 - sentiment_factor * 0.2) * (1 + volatility_factor * 0.1) * (1 - trend_strength_factor * 0.1)
        dynamic_state_fusion_weights["structure"] = dynamic_state_fusion_weights["structure"] * (1 + sentiment_factor * 0.2) * (1 + volatility_factor * 0.1) * (1 + trend_strength_factor * 0.1)
        dynamic_state_fusion_weights["potential"] = dynamic_state_fusion_weights["potential"] * (1 + sentiment_factor * 0.1) * (1 + volatility_factor * 0.05) * (1 + trend_strength_factor * 0.05)
        # 新增信号的动态调整
        dynamic_state_fusion_weights["order_flow"] = dynamic_state_fusion_weights["order_flow"] * (1 + sentiment_factor * 0.1) * (1 + volatility_factor * 0.1) * (1 + trend_strength_factor * 0.1)
        dynamic_state_fusion_weights["chip_structure"] = dynamic_state_fusion_weights["chip_structure"] * (1 + sentiment_factor * 0.1) * (1 + volatility_factor * 0.05) * (1 + trend_strength_factor * 0.05)
        dynamic_state_fusion_weights["liquidity"] = dynamic_state_fusion_weights["liquidity"] * (1 + sentiment_factor * 0.05) * (1 + volatility_factor * 0.1) * (1 + trend_strength_factor * 0.05) # 新增流动性动态权重
        # 归一化动态权重
        total_dynamic_weight = sum(dynamic_state_fusion_weights.values())
        total_dynamic_weight = total_dynamic_weight.replace(0, 1e-9)
        for k in dynamic_state_fusion_weights:
            dynamic_state_fusion_weights[k] /= total_dynamic_weight
        holographic_debug_values["dynamic_state_fusion_weights"] = dynamic_state_fusion_weights
        # 非线性融合 holographic_state_score
        holographic_state_components = {
            "flow": data_flow_outcome,
            "structure": data_structure_outcome,
            "potential": data_potential_outcome,
            "order_flow": data_order_flow_outcome,
            "chip_structure": data_chip_structure_outcome,
            "liquidity": data_liquidity_outcome # 新增
        }
        
        holographic_state_components_pre_gm = {k: (v + 1) / 2 if v.min() < 0 else v for k, v in holographic_state_components.items()}
        holographic_debug_values["holographic_state_components_pre_gm_values"] = holographic_state_components_pre_gm
        
        debug_info_for_state_gm = (is_debug_enabled_for_method, probe_ts, "holographic_state_score_GM")
        holographic_state_score = _robust_geometric_mean(
            holographic_state_components_pre_gm, # 确保输入为正
            dynamic_state_fusion_weights,
            df_index
        )
        holographic_debug_values["holographic_state_score_components"] = holographic_state_components
        holographic_debug_values["holographic_state_score"] = holographic_state_score
        # 趋势动量 diff() 化 (针对数据层复合信号)
        flow_trend_raw = data_flow_outcome.diff(3).fillna(0)
        structure_trend_raw = data_structure_outcome.diff(3).fillna(0)
        potential_trend_raw = data_potential_outcome.diff(3).fillna(0)
        order_flow_trend_raw = data_order_flow_outcome.diff(3).fillna(0)
        chip_structure_trend_raw = data_chip_structure_outcome.diff(3).fillna(0)
        liquidity_trend_raw = data_liquidity_outcome.diff(3).fillna(0) # 新增
        flow_trend = self.helper._normalize_series(flow_trend_raw, df_index, bipolar=True)
        structure_trend = self.helper._normalize_series(structure_trend_raw, df_index, bipolar=True)
        potential_trend = self.helper._normalize_series(potential_trend_raw, df_index, bipolar=True)
        order_flow_trend = self.helper._normalize_series(order_flow_trend_raw, df_index, bipolar=True)
        chip_structure_trend = self.helper._normalize_series(chip_structure_trend_raw, df_index, bipolar=True)
        liquidity_trend = self.helper._normalize_series(liquidity_trend_raw, df_index, bipolar=True) # 新增
        holographic_debug_values["flow_trend_raw"] = flow_trend_raw
        holographic_debug_values["structure_trend_raw"] = structure_trend_raw
        holographic_debug_values["potential_trend_raw"] = potential_trend_raw
        holographic_debug_values["order_flow_trend_raw"] = order_flow_trend_raw
        holographic_debug_values["chip_structure_trend_raw"] = chip_structure_trend_raw
        holographic_debug_values["liquidity_trend_raw"] = liquidity_trend_raw # 新增
        holographic_trend_components = {
            "flow_trend": flow_trend,
            "structure_trend": structure_trend,
            "potential_trend": potential_trend,
            "order_flow_trend": order_flow_trend,
            "chip_structure_trend": chip_structure_trend,
            "liquidity_trend": liquidity_trend # 新增
        }
        debug_info_for_trend_gm = (is_debug_enabled_for_method, probe_ts, "holographic_trend_score_GM")
        holographic_trend_score = _robust_geometric_mean(
            {k: (v + 1) / 2 if v.min() < 0 else v for k, v in holographic_trend_components.items()}, # 确保输入为正
            trend_fusion_weights,
            df_index
        )
        holographic_debug_values["holographic_trend_score_components"] = holographic_trend_components
        holographic_debug_values["holographic_trend_score"] = holographic_trend_score
        # --- 计算 RDI 信号并应用奖励/惩罚 ---
        rdi_signals = self._calculate_rdi_signals(normalized_signals, df_index, config)
        holographic_debug_values["rdi_signals"] = rdi_signals
        rdi_params = config.get('rdi_params', {})
        base_resonance_reward_factor = get_param_value(rdi_params.get('resonance_reward_factor'), 0.1)
        base_divergence_penalty_factor = get_param_value(rdi_params.get('divergence_penalty_factor'), 0.15)
        base_inflection_reward_factor = get_param_value(rdi_params.get('inflection_reward_factor'), 0.05)
        base_cohesion_reward_factor = get_param_value(rdi_params.get('cohesion_reward_factor'), 0.1)
        # 情境自适应RDI权重调整
        # 趋势越强 (ADX高)，共振和拐点奖励可能更显著，背离惩罚更重
        # 波动率越低 (稳定性高)，共振和协同性奖励可能更显著
        adx_mod_factor = (1 + adx_norm) / 2 # ADX越高，因子越大 (1.0 to 1.5)
        vol_mod_factor = (1 + volatility_factor) / 2 # 波动率越低，因子越大 (0.5 to 1.0)
        resonance_reward_factor = base_resonance_reward_factor * adx_mod_factor * vol_mod_factor
        divergence_penalty_factor = base_divergence_penalty_factor * adx_mod_factor * (1 + (1 - vol_mod_factor)) # 波动率高时，惩罚更重
        inflection_reward_factor = base_inflection_reward_factor * adx_mod_factor
        cohesion_reward_factor = base_cohesion_reward_factor * vol_mod_factor
        holographic_debug_values["dynamic_rdi_factors"] = {
            "resonance_reward_factor": resonance_reward_factor.loc[probe_ts] if probe_ts in resonance_reward_factor.index else np.nan,
            "divergence_penalty_factor": divergence_penalty_factor.loc[probe_ts] if probe_ts in divergence_penalty_factor.index else np.nan,
            "inflection_reward_factor": inflection_reward_factor.loc[probe_ts] if probe_ts in inflection_reward_factor.index else np.nan,
            "cohesion_reward_factor": cohesion_reward_factor.loc[probe_ts] if probe_ts in cohesion_reward_factor.index else np.nan,
        }
        # 应用共振奖励
        # 调整 total_positive_resonance 的计算，包含新的复合信号
        total_positive_resonance = (rdi_signals["overall_positive_resonance"] + 
                                    rdi_signals["positive_flow_resonance"] + 
                                    rdi_signals["positive_structure_resonance"] + 
                                    rdi_signals["positive_potential_resonance"] +
                                    rdi_signals["positive_order_flow_resonance"] +
                                    rdi_signals["positive_chip_structure_resonance"] +
                                    rdi_signals["positive_liquidity_resonance"]) / 7 # 新增
        holographic_state_score = holographic_state_score * (1 + total_positive_resonance * resonance_reward_factor)
        holographic_trend_score = holographic_trend_score * (1 + total_positive_resonance * resonance_reward_factor)
        # 应用背离惩罚
        # 调整 total_divergence_penalty 的计算，包含新的复合信号
        total_divergence_penalty = (rdi_signals["price_flow_divergence"] + 
                                    rdi_signals["price_structure_divergence"] + 
                                    rdi_signals["internal_divergence"] +
                                    rdi_signals["price_order_flow_divergence"] +
                                    rdi_signals["price_chip_structure_divergence"] +
                                    rdi_signals["order_flow_chip_structure_divergence"] +
                                    rdi_signals["price_liquidity_divergence"]) / 7 # 新增
        holographic_state_score = holographic_state_score * (1 - total_divergence_penalty * divergence_penalty_factor)
        holographic_trend_score = holographic_trend_score * (1 - total_divergence_penalty * divergence_penalty_factor)
        # 应用拐点奖励 (主要影响趋势)
        # 调整 total_inflection_reward 的计算，包含新的复合信号
        total_inflection_reward = (rdi_signals["flow_inflection"] + 
                                   rdi_signals["structure_inflection"] + 
                                   rdi_signals["potential_inflection"] +
                                   rdi_signals["order_flow_inflection"] +
                                   rdi_signals["chip_structure_inflection"] +
                                   rdi_signals["liquidity_inflection"]) / 6 # 新增
        holographic_trend_score = holographic_trend_score * (1 + total_inflection_reward * inflection_reward_factor)
        # 应用MTF协同性奖励
        mtf_cohesion_score = mtf_signals["mtf_cohesion_score"] 
        holographic_debug_values["mtf_cohesion_score"] = mtf_cohesion_score
        # 协同性分数现在是双极性的，这里根据其正负来决定是奖励还是惩罚
        holographic_state_score = holographic_state_score * (1 + mtf_cohesion_score * cohesion_reward_factor)
        holographic_trend_score = holographic_trend_score * (1 + mtf_cohesion_score * cohesion_reward_factor)
        # 确保分数仍在合理范围
        holographic_state_score = holographic_state_score.clip(0, 1)
        holographic_trend_score = holographic_trend_score.clip(0, 1)
        # 整体融合
        overall_holographic_components = {
            "state": holographic_state_score,
            "trend": holographic_trend_score
        }
        debug_info_for_overall_gm = (is_debug_enabled_for_method, probe_ts, "holographic_validation_score_GM")
        holographic_validation_score = _robust_geometric_mean(
            {k: (v + 1) / 2 if v.min() < 0 else v for k, v in overall_holographic_components.items()}, # 确保输入为正
            overall_fusion_weights,
            df_index
        )
        # 将几何平均结果映射回 [-1, 1] 区间，通过乘以原始状态和趋势的平均符号
        weighted_avg_direction = (data_flow_outcome * dynamic_state_fusion_weights["flow"] + 
                                  data_structure_outcome * dynamic_state_fusion_weights["structure"] + 
                                  data_potential_outcome * dynamic_state_fusion_weights["potential"] +
                                  data_order_flow_outcome * dynamic_state_fusion_weights["order_flow"] +
                                  data_chip_structure_outcome * dynamic_state_fusion_weights["chip_structure"] +
                                  data_liquidity_outcome * dynamic_state_fusion_weights["liquidity"]) # 新增
        holographic_validation_score = holographic_validation_score * weighted_avg_direction.apply(np.sign).replace(0, 1) # 确保符号一致，0时视为正向
        return holographic_validation_score.clip(-1, 1), holographic_debug_values

    def _calculate_rdi_signals(self, normalized_signals: Dict[str, pd.Series], df_index: pd.Index, config: Dict) -> Dict[str, pd.Series]:
        """
        【V4.5 · 流动性与执行质量增强版】
        计算共振 (Resonance)、背离 (Divergence) 和拐点 (Inflection) 信号。
        这些信号将用于对全息验证分数进行奖励或惩罚。
        - 新增: 针对 `data_liquidity_outcome` 复合信号的RDI计算。
        - 优化: 调整了聚合RDI信号的计算，以包含所有新的复合信号。
        """
        rdi_signals = {}
        params = config.get('rdi_params', {})
        rdi_periods = [5, 13, 21]
        # 获取数据层复合信号
        data_flow_outcome = normalized_signals["data_flow_outcome"]
        data_structure_outcome = normalized_signals["data_structure_outcome"]
        data_potential_outcome = normalized_signals["data_potential_outcome"]
        data_order_flow_outcome = normalized_signals["data_order_flow_outcome"]
        data_chip_structure_outcome = normalized_signals["data_chip_structure_outcome"]
        data_liquidity_outcome = normalized_signals["data_liquidity_outcome"] # 新增
        # 获取价格趋势信号
        price_slope_5 = normalized_signals["price_trend_norm"]
        price_slope_13 = normalized_signals["SLOPE_13_close_D"]
        price_slope_21 = normalized_signals["SLOPE_21_close_D"]
        # --- 共振信号 (Resonance) ---
        # 1. 资金流共振
        flow_slopes = {
            f'SLOPE_{p}_data_flow_outcome': normalized_signals[f'SLOPE_{p}_data_flow_outcome'] for p in rdi_periods
        }
        positive_flow_resonance_components = [s.clip(lower=0) for s in flow_slopes.values()]
        rdi_signals["positive_flow_resonance"] = _robust_geometric_mean(
            {f'flow_slope_{p}': s for p, s in zip(rdi_periods, positive_flow_resonance_components)},
            get_param_value(params.get('flow_resonance_weights'), {"flow_slope_5": 0.4, "flow_slope_13": 0.3, "flow_slope_21": 0.3}),
            df_index
        )
        # 2. 结构共振
        structure_slopes = {
            f'SLOPE_{p}_data_structure_outcome': normalized_signals[f'SLOPE_{p}_data_structure_outcome'] for p in rdi_periods
        }
        positive_structure_resonance_components = [s.clip(lower=0) for s in structure_slopes.values()]
        rdi_signals["positive_structure_resonance"] = _robust_geometric_mean(
            {f'structure_slope_{p}': s for p, s in zip(rdi_periods, positive_structure_resonance_components)},
            get_param_value(params.get('structure_resonance_weights'), {"structure_slope_5": 0.4, "structure_slope_13": 0.3, "structure_slope_21": 0.3}),
            df_index
        )
        # 3. 潜力共振
        potential_slopes = {
            f'SLOPE_{p}_data_potential_outcome': normalized_signals[f'SLOPE_{p}_data_potential_outcome'] for p in rdi_periods
        }
        positive_potential_resonance_components = [s.clip(lower=0) for s in potential_slopes.values()]
        rdi_signals["positive_potential_resonance"] = _robust_geometric_mean(
            {f'potential_slope_{p}': s for p, s in zip(rdi_periods, positive_potential_resonance_components)},
            get_param_value(params.get('potential_resonance_weights'), {"potential_slope_5": 0.4, "potential_slope_13": 0.3, "potential_slope_21": 0.3}),
            df_index
        )
        # 4. 订单流共振
        order_flow_slopes = {
            f'SLOPE_{p}_data_order_flow_outcome': normalized_signals[f'SLOPE_{p}_data_order_flow_outcome'] for p in rdi_periods
        }
        positive_order_flow_resonance_components = [s.clip(lower=0) for s in order_flow_slopes.values()]
        rdi_signals["positive_order_flow_resonance"] = _robust_geometric_mean(
            {f'order_flow_slope_{p}': s for p, s in zip(rdi_periods, positive_order_flow_resonance_components)},
            get_param_value(params.get('order_flow_resonance_weights'), {"order_flow_slope_5": 0.4, "order_flow_slope_13": 0.3, "order_flow_slope_21": 0.3}),
            df_index
        )
        # 5. 筹码结构共振
        chip_structure_slopes = {
            f'SLOPE_{p}_data_chip_structure_outcome': normalized_signals[f'SLOPE_{p}_data_chip_structure_outcome'] for p in rdi_periods
        }
        positive_chip_structure_resonance_components = [s.clip(lower=0) for s in chip_structure_slopes.values()]
        rdi_signals["positive_chip_structure_resonance"] = _robust_geometric_mean(
            {f'chip_structure_slope_{p}': s for p, s in zip(rdi_periods, positive_chip_structure_resonance_components)},
            get_param_value(params.get('chip_structure_resonance_weights'), {"chip_structure_slope_5": 0.4, "chip_structure_slope_13": 0.3, "chip_structure_slope_21": 0.3}),
            df_index
        )
        # 6. 新增：流动性共振
        liquidity_slopes = {
            f'SLOPE_{p}_data_liquidity_outcome': normalized_signals[f'SLOPE_{p}_data_liquidity_outcome'] for p in rdi_periods
        }
        positive_liquidity_resonance_components = [s.clip(lower=0) for s in liquidity_slopes.values()]
        rdi_signals["positive_liquidity_resonance"] = _robust_geometric_mean(
            {f'liquidity_slope_{p}': s for p, s in zip(rdi_periods, positive_liquidity_resonance_components)},
            get_param_value(params.get('liquidity_resonance_weights'), {"liquidity_slope_5": 0.4, "liquidity_slope_13": 0.3, "liquidity_slope_21": 0.3}),
            df_index
        )
        # 7. 价格与所有复合信号共振
        price_all_composite_resonance_components = {
            "price_slope_5": price_slope_5.clip(lower=0),
            "flow_outcome": data_flow_outcome.clip(lower=0),
            "structure_outcome": data_structure_outcome.clip(lower=0),
            "potential_outcome": data_potential_outcome.clip(lower=0),
            "order_flow_outcome": data_order_flow_outcome.clip(lower=0),
            "chip_structure_outcome": data_chip_structure_outcome.clip(lower=0),
            "liquidity_outcome": data_liquidity_outcome.clip(lower=0) # 新增
        }
        rdi_signals["overall_positive_resonance"] = _robust_geometric_mean(
            price_all_composite_resonance_components,
            get_param_value(params.get('overall_resonance_weights'), {"price_slope_5": 0.15, "flow_outcome": 0.15, "structure_outcome": 0.15, "potential_outcome": 0.15, "order_flow_outcome": 0.15, "chip_structure_outcome": 0.15, "liquidity_outcome": 0.1}), # 调整默认权重
            df_index
        )
        # --- 背离信号 (Divergence) ---
        # 1. 价格与资金流背离
        rdi_signals["price_flow_divergence"] = self.helper._normalize_series(
            (price_slope_5 * -1 * data_flow_outcome).clip(lower=0),
            df_index, bipolar=False
        )
        # 2. 价格与结构背离
        rdi_signals["price_structure_divergence"] = self.helper._normalize_series(
            (price_slope_5 * -1 * data_structure_outcome).clip(lower=0),
            df_index, bipolar=False
        )
        # 3. 复合信号内部背离 (保持不变，但可以考虑扩展)
        rdi_signals["internal_divergence"] = self.helper._normalize_series(
            (data_flow_outcome * -1 * data_structure_outcome).clip(lower=0) +
            (data_flow_outcome * -1 * data_potential_outcome).clip(lower=0) +
            (data_structure_outcome * -1 * data_potential_outcome).clip(lower=0),
            df_index, bipolar=False
        )
        # 4. 价格与订单流背离
        rdi_signals["price_order_flow_divergence"] = self.helper._normalize_series(
            (price_slope_5 * -1 * data_order_flow_outcome).clip(lower=0),
            df_index, bipolar=False
        )
        # 5. 价格与筹码结构背离
        rdi_signals["price_chip_structure_divergence"] = self.helper._normalize_series(
            (price_slope_5 * -1 * data_chip_structure_outcome).clip(lower=0),
            df_index, bipolar=False
        )
        # 6. 订单流与筹码结构背离
        rdi_signals["order_flow_chip_structure_divergence"] = self.helper._normalize_series(
            (data_order_flow_outcome * -1 * data_chip_structure_outcome).clip(lower=0),
            df_index, bipolar=False
        )
        # 7. 新增：价格与流动性背离
        rdi_signals["price_liquidity_divergence"] = self.helper._normalize_series(
            (price_slope_5 * -1 * data_liquidity_outcome).clip(lower=0),
            df_index, bipolar=False
        )
        # --- 拐点信号 (Inflection) ---
        # 1. 资金流拐点
        flow_accel_5 = normalized_signals['ACCEL_5_data_flow_outcome']
        flow_slope_5 = normalized_signals['SLOPE_5_data_flow_outcome']
        rdi_signals["flow_inflection"] = self.helper._normalize_series(
            (flow_accel_5.clip(lower=0) * (flow_slope_5.shift(1) < 0).astype(float)).fillna(0),
            df_index, bipolar=False
        )
        # 2. 结构拐点
        structure_accel_5 = normalized_signals['ACCEL_5_data_structure_outcome']
        structure_slope_5 = normalized_signals['SLOPE_5_data_structure_outcome']
        rdi_signals["structure_inflection"] = self.helper._normalize_series(
            (structure_accel_5.clip(lower=0) * (structure_slope_5.shift(1) < 0).astype(float)).fillna(0),
            df_index, bipolar=False
        )
        # 3. 潜力拐点
        potential_accel_5 = normalized_signals['ACCEL_5_data_potential_outcome']
        potential_slope_5 = normalized_signals['SLOPE_5_data_potential_outcome']
        rdi_signals["potential_inflection"] = self.helper._normalize_series(
            (potential_accel_5.clip(lower=0) * (potential_slope_5.shift(1) < 0).astype(float)).fillna(0),
            df_index, bipolar=False
        )
        # 4. 订单流拐点
        order_flow_accel_5 = normalized_signals['ACCEL_5_data_order_flow_outcome']
        order_flow_slope_5 = normalized_signals['SLOPE_5_data_order_flow_outcome']
        rdi_signals["order_flow_inflection"] = self.helper._normalize_series(
            (order_flow_accel_5.clip(lower=0) * (order_flow_slope_5.shift(1) < 0).astype(float)).fillna(0),
            df_index, bipolar=False
        )
        # 5. 筹码结构拐点
        chip_structure_accel_5 = normalized_signals['ACCEL_5_data_chip_structure_outcome']
        chip_structure_slope_5 = normalized_signals['SLOPE_5_data_chip_structure_outcome']
        rdi_signals["chip_structure_inflection"] = self.helper._normalize_series(
            (chip_structure_accel_5.clip(lower=0) * (chip_structure_slope_5.shift(1) < 0).astype(float)).fillna(0),
            df_index, bipolar=False
        )
        # 6. 新增：流动性拐点
        liquidity_accel_5 = normalized_signals['ACCEL_5_data_liquidity_outcome']
        liquidity_slope_5 = normalized_signals['SLOPE_5_data_liquidity_outcome']
        rdi_signals["liquidity_inflection"] = self.helper._normalize_series(
            (liquidity_accel_5.clip(lower=0) * (liquidity_slope_5.shift(1) < 0).astype(float)).fillna(0),
            df_index, bipolar=False
        )
        return rdi_signals

    def _calculate_dynamic_efficiency_baseline(self, context_signals: Dict[str, pd.Series], df_index: pd.Index, config: Dict) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """
        【V4.1 · 数据层强化与周期性考量版】
        计算动态效率基准线。
        - 核心修正: 使用数据层复合情绪信号作为市场情绪输入。
        """
        baseline_debug_values = {}
        params = config.get('dynamic_efficiency_baseline_params', {})
        base_baseline = get_param_value(params.get('base_baseline'), 0.15)
        sentiment_impact = get_param_value(params.get('sentiment_impact'), 0.1)
        volatility_impact = get_param_value(params.get('volatility_impact'), 0.05)
        consolidation_impact = get_param_value(params.get('consolidation_impact'), 0.05)
        # 使用新的数据层复合情绪信号
        market_sentiment_norm = context_signals["market_sentiment_norm"]
        volatility_instability_norm = context_signals["volatility_instability_norm"]
        is_consolidating_norm = context_signals["is_consolidating_norm"]
        # 情绪越积极，基准线越高 (要求更高)
        # 波动率越低 (稳定性越高)，基准线越高 (要求更高)
        # 处于盘整期，基准线可以适当降低 (吸筹难度大)
        dynamic_baseline = base_baseline + \
                           (market_sentiment_norm.clip(lower=0) * sentiment_impact) - \
                           (volatility_instability_norm * volatility_impact) - \
                           (is_consolidating_norm * consolidation_impact)
        dynamic_baseline = dynamic_baseline.clip(0.05, 0.3) # 限制基准线范围
        baseline_debug_values["base_baseline"] = base_baseline
        baseline_debug_values["sentiment_impact_term"] = market_sentiment_norm.clip(lower=0) * sentiment_impact
        baseline_debug_values["volatility_impact_term"] = volatility_instability_norm * volatility_impact
        baseline_debug_values["consolidation_impact_term"] = is_consolidating_norm * consolidation_impact
        return dynamic_baseline, baseline_debug_values

    def _calculate_preliminary_score(self, normalized_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], context_signals: Dict[str, pd.Series], df_index: pd.Index, config: Dict) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """
        【V4.1 · 数据层强化与周期性考量版】
        计算初步的拆单吸筹强度分数。
        - 核心修正: 使用数据层复合潜力信号作为稳定性输入。
        """
        preliminary_debug_values = {}
        params = config.get('preliminary_score_params', {})
        price_suppression_weights = get_param_value(params.get('price_suppression_weights'), {"price_trend_negative": 0.3, "is_consolidating": 0.3, "consolidation_duration": 0.2, "upward_purity_inverted": 0.2})
        strategic_context_weights = get_param_value(params.get('strategic_context_weights'), {"stability": 0.4, "deception_lure_long_inverted": 0.3, "deception_lure_short": 0.3})
        fusion_weights = get_param_value(params.get('fusion_weights'), {"mtf_intensity": 0.4, "price_suppression": 0.3, "strategic_context": 0.3})
        momentum_weight = get_param_value(params.get('momentum_weight'), 0.3)
        normalized_score = normalized_signals["normalized_score"]
        price_trend_norm = normalized_signals["price_trend_norm"]
        upward_purity = normalized_signals["upward_purity"]
        # 使用新的数据层复合潜力信号
        potential_outcome = normalized_signals["data_potential_outcome"]
        deception_lure_long_norm = context_signals["deception_lure_long_norm"]
        deception_lure_short_norm = context_signals["deception_lure_short_norm"]
        mtf_hidden_accumulation_intensity = mtf_signals["mtf_hidden_accumulation_intensity"]
        is_consolidating_norm = context_signals["is_consolidating_norm"]
        dynamic_consolidation_duration_norm = context_signals["dynamic_consolidation_duration_norm"]
        # 增强价格行为捕捉 (price_suppression_factor)
        price_trend_negative = price_trend_norm.clip(upper=0).abs() # 价格下跌的强度
        upward_purity_inverted = 1 - upward_purity # 上涨纯度越低越好
        price_suppression_components = {
            "price_trend_negative": price_trend_negative,
            "is_consolidating": is_consolidating_norm,
            "consolidation_duration": dynamic_consolidation_duration_norm,
            "upward_purity_inverted": upward_purity_inverted
        }
        price_suppression_factor = _robust_geometric_mean(price_suppression_components, price_suppression_weights, df_index).clip(0, 1)
        preliminary_debug_values["price_trend_negative"] = price_trend_negative
        preliminary_debug_values["upward_purity_inverted"] = upward_purity_inverted
        preliminary_debug_values["price_suppression_factor_components"] = price_suppression_components
        preliminary_debug_values["price_suppression_factor"] = price_suppression_factor
        # 精细化欺诈意图识别 (strategic_context_factor)
        deception_lure_long_inverted = 1 - deception_lure_long_norm # 诱多越少越好
        deception_lure_short = deception_lure_short_norm # 诱空越多越好 (主力震仓)
        strategic_context_components = {
            "stability": potential_outcome, # 使用新的数据层复合潜力信号
            "deception_lure_long_inverted": deception_lure_long_inverted,
            "deception_lure_short": deception_lure_short
        }
        strategic_context_factor = _robust_geometric_mean(strategic_context_components, strategic_context_weights, df_index).clip(0, 1)
        preliminary_debug_values["deception_lure_long_inverted"] = deception_lure_long_inverted
        preliminary_debug_values["deception_lure_short"] = deception_lure_short
        preliminary_debug_values["strategic_context_factor_components"] = strategic_context_components
        preliminary_debug_values["strategic_context_factor"] = strategic_context_factor
        # MTF 核心信号增强
        preliminary_components = {
            "mtf_intensity": mtf_hidden_accumulation_intensity,
            "price_suppression": price_suppression_factor,
            "strategic_context": strategic_context_factor
        }
        preliminary_score = _robust_geometric_mean(preliminary_components, fusion_weights, df_index).fillna(0.0)
        preliminary_debug_values["preliminary_score_components"] = preliminary_components
        preliminary_debug_values["preliminary_score"] = preliminary_score
        tactical_momentum_score = self.helper._normalize_series(preliminary_score.diff(1).fillna(0), df_index, bipolar=False)
        preliminary_debug_values["tactical_momentum_score"] = tactical_momentum_score
        dynamic_preliminary_score = (preliminary_score * (1 - momentum_weight) + tactical_momentum_score * momentum_weight).clip(0, 1)
        return dynamic_preliminary_score, preliminary_debug_values

    def _apply_quality_efficiency_calibration(self, dynamic_preliminary_score: pd.Series, holographic_validation_score: pd.Series, dynamic_efficiency_baseline: pd.Series, probe_ts: Optional[pd.Timestamp]) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """
        应用质效校准并计算最终分数。
        """
        final_score_debug_values = {}
        calibrated_holographic_score = holographic_validation_score - dynamic_efficiency_baseline
        final_score_debug_values["calibrated_holographic_score"] = calibrated_holographic_score
        # 确保 quality_efficiency_modulator 始终为正，且在合理范围内
        # 如果 calibrated_holographic_score 很高，modulator 应该小 (放大 preliminary_score)
        # 如果 calibrated_holographic_score 很低，modulator 应该大 (惩罚 preliminary_score)
        quality_efficiency_modulator = (1 - calibrated_holographic_score).clip(0.1, 2.0)
        final_score_debug_values["quality_efficiency_modulator"] = quality_efficiency_modulator
        final_score = dynamic_preliminary_score.pow(quality_efficiency_modulator).clip(0, 1).fillna(0.0)
        # --- 探针强化: 打印pow()操作的精确值 ---
        if probe_ts is not None and probe_ts in dynamic_preliminary_score.index and probe_ts in quality_efficiency_modulator.index:
            final_score_debug_values["dynamic_preliminary_score_at_probe_ts"] = dynamic_preliminary_score.loc[probe_ts]
            final_score_debug_values["quality_efficiency_modulator_at_probe_ts"] = quality_efficiency_modulator.loc[probe_ts]
            final_score_debug_values["calculated_pow_result_at_probe_ts"] = dynamic_preliminary_score.loc[probe_ts] ** quality_efficiency_modulator.loc[probe_ts]
        return final_score, final_score_debug_values

    def _print_debug_info(self, method_name: str, probe_ts: pd.Timestamp, debug_output: Dict, _temp_debug_values: Dict, final_score: pd.Series):
        """
        统一打印调试信息。
        """
        for section, values in _temp_debug_values.items():
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- {section} ---"] = ""
            for key, series_or_value in values.items():
                if isinstance(series_or_value, pd.Series):
                    val = series_or_value.loc[probe_ts] if probe_ts in series_or_value.index else np.nan
                    debug_output[f"        {key}: {val:.4f}"] = ""
                elif isinstance(series_or_value, dict): # 处理嵌套字典，例如 preliminary_score_components
                    debug_output[f"        {key}:"] = ""
                    for sub_key, sub_series in series_or_value.items():
                        if isinstance(sub_series, pd.Series):
                            sub_val = sub_series.loc[probe_ts] if probe_ts in sub_series.index else np.nan
                            debug_output[f"          {sub_key}: {sub_val:.4f}"] = ""
                        else:
                            debug_output[f"          {sub_key}: {sub_series}"] = ""
                else:
                    debug_output[f"        {key}: {series_or_value}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 拆单吸筹强度诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
        self.helper._print_debug_output(debug_output)





