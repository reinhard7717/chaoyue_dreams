# 文件: strategies/trend_following/intelligence/process/calculate_cost_advantage_trend_relationship.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

from strategies.trend_following.utils import (
    get_params_block, get_param_value, get_adaptive_mtf_normalized_score,
    get_adaptive_mtf_normalized_bipolar_score, normalize_score, _robust_geometric_mean
)
from strategies.trend_following.intelligence.process.helper import ProcessIntelligenceHelper

class CalculateCostAdvantageTrendRelationship:
    """
    【V1.0.0 · 成本优势趋势关系计算器】
    PROCESS_META_COST_ADVANTAGE_TREND
    - 核心职责: 计算“成本优势趋势”的专属关系分数。
    - 版本: 1.0.0
    """
    def __init__(self, strategy_instance, helper_instance: ProcessIntelligenceHelper):
        """
        初始化 CalculateCostAdvantageTrendRelationship 处理器。
        参数:
            strategy_instance: 策略实例，用于访问全局配置和原子状态。
            helper_instance (ProcessIntelligenceHelper): 过程情报辅助工具实例。
        """
        self.strategy = strategy_instance
        self.helper = helper_instance
        self.params = self.helper.params # 获取ProcessIntelligence的参数
        self.debug_params = self.helper.debug_params
        self.probe_dates = self.helper.probe_dates

    def _initialize_debug_context(self, method_name: str, df: pd.DataFrame) -> Tuple[bool, Optional[pd.Timestamp], Dict, Dict]:
        """
        【V1.0.0 · 调试上下文初始化】
        - 核心职责: 初始化调试相关的变量，包括是否启用调试、探针时间戳、调试输出字典和临时调试值字典。
        - 版本: 1.0.0
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
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算成本优势趋势关系..."] = ""
        return is_debug_enabled_for_method, probe_ts, debug_output, _temp_debug_values

    def _log_debug_values(self, debug_output: Dict, _temp_debug_values: Dict, probe_ts: pd.Timestamp, method_name: str):
        """
        【V1.0.0 · 调试值日志输出】
        - 核心职责: 统一输出调试信息到控制台。
        - 版本: 1.0.0
        """
        for section, values_dict in _temp_debug_values.items():
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- {section} ---"] = ""
            for key, value in values_dict.items():
                if isinstance(value, dict):
                    debug_output[f"        {key}:"] = ""
                    for sub_key, sub_series in value.items():
                        val = sub_series.loc[probe_ts] if probe_ts in sub_series.index else np.nan
                        debug_output[f"          {sub_key}: {val:.4f}"] = ""
                elif isinstance(value, pd.Series):
                    val = value.loc[probe_ts] if probe_ts in value.index else np.nan
                    debug_output[f"        '{key}': {val:.4f}"] = ""
                else:
                    debug_output[f"        '{key}': {value}"] = "" # For non-series values
        for key, value in debug_output.items():
            if value:
                print(f"{key}: {value}")
            else:
                print(key)

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V4.2 · 修复AttributeError】计算成本优势趋势。
        - 核心修复: 解决 'TrendFollowStrategy' object has no attribute 'df' 错误，正确传递 df 参数。
        - 核心修复: 将缺失的 'market_stability_score_D' 替换为数据层已有的 'MA_POTENTIAL_ORDERLINESS_SCORE_D'。
        - 核心升级: 引入情境自适应权重、动态指数和信号交互项。
        - 核心新增: 引入更多微观结构和订单流信号，增强主力行为的验证。
        - 版本: 4.2
        """
        method_name = "CalculateCostAdvantageTrendRelationship"
        is_debug_enabled_for_method, probe_ts, debug_output, _temp_debug_values = self._initialize_debug_context(method_name, df)
        # 1. 获取MTF配置
        _, _, _, mtf_slope_accel_weights = self._get_mtf_configs(config)
        # 2. 获取所需信号列表并进行验证
        required_signals = self._get_required_signals_list(mtf_slope_accel_weights)
        # 补充情境调制和微观信号到 required_signals
        required_signals.extend([
            'VOLATILITY_INSTABILITY_INDEX_21d_D', 'ADX_14_D', 'market_sentiment_score_D',
            'liquidity_authenticity_score_D', 'MA_POTENTIAL_ORDERLINESS_SCORE_D', 'microstructure_efficiency_index_D',
            'main_force_buy_execution_alpha_D', 'main_force_sell_execution_alpha_D',
            'micro_price_impact_asymmetry_D'
        ])
        if not self.helper._validate_required_signals(df, required_signals, method_name):
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。"] = ""
                self.helper._print_debug_output(debug_output)
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        # 3. 获取原始数据和MTF融合信号
        fetched_signals = self._fetch_raw_and_mtf_signals(df, df_index, mtf_slope_accel_weights, method_name, _temp_debug_values)
        # 4. 归一化处理
        normalized_signals = self._normalize_all_signals(df, df_index, fetched_signals, mtf_slope_accel_weights, method_name, _temp_debug_values)
        # 5. 计算动态权重
        dynamic_weights = self._calculate_dynamic_weights(normalized_signals, config, df_index, method_name, _temp_debug_values)
        # 6. 计算各象限分数
        Q1_final = self._calculate_q1_healthy_rally(fetched_signals, normalized_signals, dynamic_weights, _temp_debug_values)
        Q2_final = self._calculate_q2_bearish_distribution(fetched_signals, normalized_signals, dynamic_weights, _temp_debug_values)
        Q3_final = self._calculate_q3_golden_pit(fetched_signals, normalized_signals, df_index, dynamic_weights, _temp_debug_values)
        Q4_final = self._calculate_q4_bull_trap(fetched_signals, normalized_signals, dynamic_weights, _temp_debug_values)
        # 7. 计算交互项
        interaction_score = self._calculate_interaction_terms(fetched_signals, normalized_signals, config, df_index, _temp_debug_values)
        # --- 最终融合 ---
        # 基础融合分数
        base_fusion_score = (Q1_final + Q2_final + Q3_final + Q4_final)
        # 叠加交互项
        final_score_with_interaction = base_fusion_score + interaction_score
        # 8. 计算动态指数
        dynamic_exponent = self._calculate_dynamic_exponent(fetched_signals, config, df_index, _temp_debug_values)
        # 应用动态指数进行非线性放大/平滑
        # 将分数映射到 [0, 1] 区间，应用指数，再映射回 [-1, 1]
        # 对于 [-1, 1] 的分数，可以先映射到 [0, 2]，应用指数，再映射回 [-1, 1]
        final_score_normalized_for_exponent = (final_score_with_interaction + 1) / 2 # 映射到 [0, 1]
        final_score_exponentiated = final_score_normalized_for_exponent.pow(dynamic_exponent)
        final_score = (final_score_exponentiated * 2 - 1).clip(-1, 1) # 映射回 [-1, 1]
        _temp_debug_values["最终融合"] = {
            "base_fusion_score": base_fusion_score,
            "final_score_with_interaction": final_score_with_interaction,
            "final_score": final_score
        }
        # --- 统一输出调试信息 ---
        if is_debug_enabled_for_method and probe_ts:
            self._log_debug_values(debug_output, _temp_debug_values, probe_ts, method_name)
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 成本优势趋势关系诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
            self.helper._print_debug_output(debug_output)
        return final_score.astype(np.float32)

    def _get_mtf_configs(self, config: Dict) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        【V1.0.0 · MTF配置提取】
        - 核心职责: 从配置中提取所有MTF相关的权重配置。
        - 版本: 1.0.0
        """
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        mtf_slope_accel_weights = config.get('mtf_slope_accel_weights', {"slope_periods": {"5": 0.6, "13": 0.4}, "accel_periods": {"5": 0.7, "13": 0.3}})
        return p_conf_structural_ultimate, p_mtf, actual_mtf_weights, mtf_slope_accel_weights

    def _get_required_signals_list(self, mtf_slope_accel_weights: Dict) -> List[str]:
        """
        【V1.0.0 · 必需信号列表构建】
        - 核心职责: 构建所有必需的原始信号和MTF派生信号的列表。
        - 版本: 1.0.0
        """
        required_signals = [
            'pct_change_D', 'main_force_cost_advantage_D', 'main_force_conviction_index_D',
            'upward_impulse_purity_D', 'suppressive_accumulation_intensity_D',
            'SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION', 'distribution_at_peak_intensity_D',
            'active_selling_pressure_D', 'profit_taking_flow_ratio_D', 'active_buying_support_D',
            'main_force_net_flow_calibrated_D', 'flow_credibility_index_D'
        ]
        # 动态添加MTF斜率和加速度信号到required_signals
        for base_sig in ['close_D', 'main_force_cost_advantage_D', 'main_force_conviction_index_D',
                         'upward_impulse_purity_D', 'distribution_at_peak_intensity_D', 'active_selling_pressure_D']:
            for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
                required_signals.append(f'SLOPE_{period_str}_{base_sig}')
            for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
                required_signals.append(f'ACCEL_{period_str}_{base_sig}')
        return required_signals

    def _fetch_raw_and_mtf_signals(self, df: pd.DataFrame, df_index: pd.Index, mtf_slope_accel_weights: Dict, method_name: str, _temp_debug_values: Dict) -> Dict[str, pd.Series]:
        """
        【V1.1.1 · 原始及MTF信号获取 - 情境指标替换版】
        - 核心职责: 获取所有原始信号和MTF融合信号，并将其存储到调试字典中。
        - 核心修复: 将缺失的 'market_stability_score_D' 替换为 'MA_POTENTIAL_ORDERLINESS_SCORE_D'。
        - 版本: 1.1.1
        """
        fetched_signals = {}
        # 价格变化和成本优势变化改为MTF融合信号
        fetched_signals['mtf_price_change'] = self.helper._get_mtf_slope_accel_score(df, 'close_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        fetched_signals['mtf_ca_change'] = self.helper._get_mtf_slope_accel_score(df, 'main_force_cost_advantage_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        fetched_signals['main_force_conviction'] = self.helper._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name=method_name)
        fetched_signals['upward_impulse_purity'] = self.helper._get_safe_series(df, 'upward_impulse_purity_D', 0.0, method_name=method_name)
        fetched_signals['suppressive_accum'] = self.helper._get_safe_series(df, 'suppressive_accumulation_intensity_D', 0.0, method_name=method_name)
        fetched_signals['lower_shadow_absorb'] = self.helper._get_atomic_score(df, 'SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION', 0.0)
        fetched_signals['distribution_intensity'] = self.helper._get_safe_series(df, 'distribution_at_peak_intensity_D', 0.0, method_name=method_name)
        fetched_signals['active_selling'] = self.helper._get_safe_series(df, 'active_selling_pressure_D', 0.0, method_name=method_name)
        fetched_signals['profit_taking_flow'] = self.helper._get_safe_series(df, 'profit_taking_flow_ratio_D', 0.0, method_name=method_name)
        fetched_signals['active_buying_support'] = self.helper._get_safe_series(df, 'active_buying_support_D', 0.0, method_name=method_name)
        fetched_signals['main_force_net_flow'] = self.helper._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name=method_name)
        fetched_signals['flow_credibility'] = self.helper._get_safe_series(df, 'flow_credibility_index_D', 0.0, method_name=method_name)
        fetched_signals['close_price'] = self.helper._get_safe_series(df, 'close_D', 0.0, method_name=method_name) # 用于计算前置下跌
        # --- 新增情境调制信号 ---
        fetched_signals['volatility_instability'] = self.helper._get_safe_series(df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', 0.0, method_name=method_name)
        fetched_signals['adx_trend_strength'] = self.helper._get_safe_series(df, 'ADX_14_D', 0.0, method_name=method_name)
        fetched_signals['market_sentiment'] = self.helper._get_safe_series(df, 'market_sentiment_score_D', 0.0, method_name=method_name)
        fetched_signals['liquidity_authenticity'] = self.helper._get_safe_series(df, 'liquidity_authenticity_score_D', 0.0, method_name=method_name)
        fetched_signals['ma_potential_orderliness_score'] = self.helper._get_safe_series(df, 'MA_POTENTIAL_ORDERLINESS_SCORE_D', 0.0, method_name=method_name) # 替换为MA_POTENTIAL_ORDERLINESS_SCORE_D
        fetched_signals['microstructure_efficiency'] = self.helper._get_safe_series(df, 'microstructure_efficiency_index_D', 0.0, method_name=method_name)
        # --- 新增微观结构和订单流信号 ---
        fetched_signals['main_force_buy_execution_alpha'] = self.helper._get_safe_series(df, 'main_force_buy_execution_alpha_D', 0.0, method_name=method_name)
        fetched_signals['main_force_sell_execution_alpha'] = self.helper._get_safe_series(df, 'main_force_sell_execution_alpha_D', 0.0, method_name=method_name)
        fetched_signals['micro_price_impact_asymmetry'] = self.helper._get_safe_series(df, 'micro_price_impact_asymmetry_D', 0.0, method_name=method_name)
        _temp_debug_values["原始信号值"] = {k: v for k, v in fetched_signals.items()}
        return fetched_signals

    def _normalize_all_signals(self, df: pd.DataFrame, df_index: pd.Index, fetched_signals: Dict[str, pd.Series], mtf_slope_accel_weights: Dict, method_name: str, _temp_debug_values: Dict) -> Dict[str, pd.Series]:
        """
        【V1.1.2 · 信号归一化处理 - 修复AttributeError】
        - 核心职责: 对所有必要的信号进行归一化处理，并将其存储到调试字典中。
        - 核心修复: 传入 df 参数，并替换 self.strategy.df，解决 AttributeError。
        - 版本: 1.1.2
        """
        normalized_signals = {}
        normalized_signals['mtf_main_force_conviction'] = self.helper._get_mtf_slope_accel_score(df, 'main_force_conviction_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        normalized_signals['mtf_upward_purity'] = self.helper._get_mtf_slope_accel_score(df, 'upward_impulse_purity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        normalized_signals['suppressive_accum_norm'] = self.helper._normalize_series(fetched_signals['suppressive_accum'], df_index, bipolar=False)
        normalized_signals['mtf_distribution_intensity'] = self.helper._get_mtf_slope_accel_score(df, 'distribution_at_peak_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        normalized_signals['mtf_active_selling'] = self.helper._get_mtf_slope_accel_score(df, 'active_selling_pressure_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        normalized_signals['profit_taking_flow_norm'] = self.helper._normalize_series(fetched_signals['profit_taking_flow'], df_index, bipolar=False)
        normalized_signals['active_buying_support_inverted_norm'] = 1 - self.helper._normalize_series(fetched_signals['active_buying_support'], df_index, bipolar=False) # 买盘虚弱度
        normalized_signals['main_force_net_flow_outflow_norm'] = self.helper._normalize_series(fetched_signals['main_force_net_flow'].clip(upper=0).abs(), df_index, bipolar=False) # 主力净流出风险
        normalized_signals['flow_credibility_norm'] = self.helper._normalize_series(fetched_signals['flow_credibility'], df_index, bipolar=False)
        # --- 新增情境调制信号归一化 ---
        normalized_signals['volatility_inverse_norm'] = 1 - self.helper._normalize_series(fetched_signals['volatility_instability'], df_index, bipolar=False) # 波动率越低，分数越高
        normalized_signals['trend_strength_inverse_norm'] = 1 - self.helper._normalize_series(fetched_signals['adx_trend_strength'], df_index, bipolar=False) # 趋势强度越低，分数越高
        normalized_signals['sentiment_neutrality_norm'] = 1 - self.helper._normalize_series(fetched_signals['market_sentiment'].abs(), df_index, bipolar=False) # 情绪越中性，分数越高
        normalized_signals['liquidity_authenticity_score_norm'] = self.helper._normalize_series(fetched_signals['liquidity_authenticity'], df_index, bipolar=False)
        normalized_signals['ma_potential_orderliness_score_norm'] = self.helper._normalize_series(fetched_signals['ma_potential_orderliness_score'], df_index, bipolar=False) # 替换为MA_POTENTIAL_ORDERLINESS_SCORE_D
        normalized_signals['microstructure_efficiency_index_norm'] = self.helper._normalize_series(fetched_signals['microstructure_efficiency'], df_index, bipolar=False)
        # --- 新增微观结构和订单流信号归一化 ---
        normalized_signals['main_force_buy_execution_alpha_norm'] = self.helper._normalize_series(fetched_signals['main_force_buy_execution_alpha'], df_index, bipolar=False)
        normalized_signals['main_force_sell_execution_alpha_norm'] = self.helper._normalize_series(fetched_signals['main_force_sell_execution_alpha'], df_index, bipolar=False)
        normalized_signals['micro_price_impact_asymmetry_norm'] = self.helper._normalize_series(fetched_signals['micro_price_impact_asymmetry'], df_index, bipolar=False)
        _temp_debug_values["归一化处理"] = {k: v for k, v in normalized_signals.items()}
        return normalized_signals

    def _calculate_q1_healthy_rally(self, fetched_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], dynamic_weights: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V1.1.1 · Q1健康上涨计算 - 动态权重与微观增强版】
        - 核心职责: 计算Q1（价涨 & 优扩）象限的分数。
        - 核心新增: 引入动态权重和主力买入执行Alpha作为确认项。
        - 核心修复: 解决 Series 比较的 ValueError。
        - 版本: 1.1.1
        """
        Q1_base = (fetched_signals['mtf_price_change'].clip(lower=0) * fetched_signals['mtf_ca_change'].clip(lower=0)).pow(0.5)
        # 确认：主力信念、上涨纯度、资金流可信度、主力买入执行Alpha
        Q1_confirm_components = {
            'mtf_main_force_conviction': normalized_signals['mtf_main_force_conviction'].clip(lower=0),
            'mtf_upward_purity': normalized_signals['mtf_upward_purity'],
            'flow_credibility_norm': normalized_signals['flow_credibility_norm'],
            'main_force_buy_execution_alpha_norm': normalized_signals['main_force_buy_execution_alpha_norm']
        }
        Q1_confirm_weights_series = dynamic_weights['Q1_confirmation_weights'] # 这是一个字典，值是 Series
        
        # 计算加权和
        weighted_sum = pd.Series(0.0, index=Q1_base.index, dtype=np.float32)
        for k, component_series in Q1_confirm_components.items():
            weight_series = Q1_confirm_weights_series.get(k, pd.Series(0.0, index=Q1_base.index, dtype=np.float32))
            weighted_sum += component_series * weight_series
        
        # 计算所有权重的总和 (Series)
        sum_of_weights_series = pd.Series(0.0, index=Q1_base.index, dtype=np.float32)
        for weight_series in Q1_confirm_weights_series.values():
            sum_of_weights_series += weight_series
        
        # 避免除以零，将总和为零的元素替换为 NaN
        sum_of_weights_series_safe = sum_of_weights_series.replace(0, np.nan)
        
        # 执行除法，并用0填充 NaN
        Q1_confirm = (weighted_sum / sum_of_weights_series_safe).fillna(0)
        
        Q1_final = (Q1_base * Q1_confirm).clip(0, 1)
        _temp_debug_values["Q1: 价涨 & 优扩"] = {
            "Q1_base": Q1_base,
            "Q1_confirm": Q1_confirm,
            "Q1_final": Q1_final
        }
        return Q1_final

    def _calculate_q2_bearish_distribution(self, fetched_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], dynamic_weights: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V1.1.1 · Q2派发下跌计算 - 动态权重与微观增强版】
        - 核心职责: 计算Q2（价跌 & 优缩）象限的分数。
        - 核心新增: 引入动态权重和主力卖出执行Alpha作为确认项。
        - 核心修复: 解决 Series 比较的 ValueError。
        - 版本: 1.1.1
        """
        Q2_base = (fetched_signals['mtf_price_change'].clip(upper=0).abs() * fetched_signals['mtf_ca_change'].clip(upper=0).abs()).pow(0.5)
        # 确认：利润兑现流量、主动卖压、行为派发意图 (使用MTF信号)、主力卖出执行Alpha
        Q2_confirm_components = {
            'profit_taking_flow_norm': normalized_signals['profit_taking_flow_norm'],
            'mtf_active_selling': normalized_signals['mtf_active_selling'],
            'mtf_distribution_intensity': normalized_signals['mtf_distribution_intensity'],
            'main_force_sell_execution_alpha_norm': normalized_signals['main_force_sell_execution_alpha_norm']
        }
        Q2_confirm_weights_series = dynamic_weights['Q2_confirmation_weights']
        
        weighted_sum = pd.Series(0.0, index=Q2_base.index, dtype=np.float32)
        for k, component_series in Q2_confirm_components.items():
            weight_series = Q2_confirm_weights_series.get(k, pd.Series(0.0, index=Q2_base.index, dtype=np.float32))
            weighted_sum += component_series * weight_series
        
        sum_of_weights_series = pd.Series(0.0, index=Q2_base.index, dtype=np.float32)
        for weight_series in Q2_confirm_weights_series.values():
            sum_of_weights_series += weight_series
        
        sum_of_weights_series_safe = sum_of_weights_series.replace(0, np.nan)
        
        Q2_distribution_evidence = (weighted_sum / sum_of_weights_series_safe).fillna(0)
        
        Q2_final = (Q2_base * Q2_distribution_evidence * -1).clip(-1, 0)
        _temp_debug_values["Q2: 价跌 & 优缩"] = {
            "Q2_base": Q2_base,
            "Q2_distribution_evidence": Q2_distribution_evidence,
            "Q2_final": Q2_final
        }
        return Q2_final

    def _calculate_q3_golden_pit(self, fetched_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], df_index: pd.Index, dynamic_weights: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V1.1.1 · Q3黄金坑计算 - 动态权重与微观增强版】
        - 核心职责: 计算Q3（价跌 & 优扩）象限的分数。
        - 核心新增: 引入动态权重和流动性真实性分数作为确认项。
        - 核心修复: 解决 Series 比较的 ValueError。
        - 版本: 1.1.1
        """
        Q3_base = (fetched_signals['mtf_price_change'].clip(upper=0).abs() * fetched_signals['mtf_ca_change'].clip(lower=0)).pow(0.5)
        # 确认：隐蔽吸筹、下影线吸收、资金流可信度、流动性真实性
        Q3_confirm_components = {
            'suppressive_accum_norm': normalized_signals['suppressive_accum_norm'],
            'lower_shadow_absorb': fetched_signals['lower_shadow_absorb'],
            'flow_credibility_norm': normalized_signals['flow_credibility_norm'],
            'liquidity_authenticity_score_norm': normalized_signals['liquidity_authenticity_score_norm']
        }
        Q3_confirm_weights_series = dynamic_weights['Q3_confirmation_weights']
        
        weighted_sum = pd.Series(0.0, index=Q3_base.index, dtype=np.float32)
        for k, component_series in Q3_confirm_components.items():
            weight_series = Q3_confirm_weights_series.get(k, pd.Series(0.0, index=Q3_base.index, dtype=np.float32))
            weighted_sum += component_series * weight_series
        
        sum_of_weights_series = pd.Series(0.0, index=Q3_base.index, dtype=np.float32)
        for weight_series in Q3_confirm_weights_series.values():
            sum_of_weights_series += weight_series
        
        sum_of_weights_series_safe = sum_of_weights_series.replace(0, np.nan)
        
        Q3_confirm = (weighted_sum / sum_of_weights_series_safe).fillna(0)
        
        # 前置下跌上下文，如果前几日有深跌，则增加黄金坑的权重
        pre_5day_pct_change = fetched_signals['close_price'].pct_change(periods=5).shift(1).fillna(0)
        norm_pre_drop_5d = self.helper._normalize_series(pre_5day_pct_change.clip(upper=0).abs(), df_index, bipolar=False)
        pre_drop_context_bonus = norm_pre_drop_5d * 0.5
        Q3_final = (Q3_base * Q3_confirm * (1 + pre_drop_context_bonus)).clip(0, 1)
        _temp_debug_values["Q3: 价跌 & 优扩"] = {
            "Q3_base": Q3_base,
            "Q3_confirm": Q3_confirm,
            "pre_5day_pct_change": pre_5day_pct_change,
            "norm_pre_drop_5d": norm_pre_drop_5d,
            "pre_drop_context_bonus": pre_drop_context_bonus,
            "Q3_final": Q3_final
        }
        return Q3_final

    def _calculate_q4_bull_trap(self, fetched_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], dynamic_weights: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V1.1.1 · Q4牛市陷阱计算 - 动态权重与微观增强版】
        - 核心职责: 计算Q4（价涨 & 优缩）象限的分数。
        - 核心新增: 引入动态权重和微观价格冲击不对称性作为确认项。
        - 核心修复: 解决 Series 比较的 ValueError。
        - 版本: 1.1.1
        """
        Q4_base = (fetched_signals['mtf_price_change'].clip(lower=0) * fetched_signals['mtf_ca_change'].clip(upper=0).abs()).pow(0.5)
        # 确认：派发强度、买盘虚弱度、主力资金净流出 (使用MTF信号)、微观价格冲击不对称性
        Q4_confirm_components = {
            'mtf_distribution_intensity': normalized_signals['mtf_distribution_intensity'],
            'active_buying_support_inverted_norm': normalized_signals['active_buying_support_inverted_norm'],
            'main_force_net_flow_outflow_norm': normalized_signals['main_force_net_flow_outflow_norm'],
            'micro_price_impact_asymmetry_norm': normalized_signals['micro_price_impact_asymmetry_norm']
        }
        Q4_confirm_weights_series = dynamic_weights['Q4_confirmation_weights']
        
        weighted_sum = pd.Series(0.0, index=Q4_base.index, dtype=np.float32)
        for k, component_series in Q4_confirm_components.items():
            weight_series = Q4_confirm_weights_series.get(k, pd.Series(0.0, index=Q4_base.index, dtype=np.float32))
            weighted_sum += component_series * weight_series
        
        sum_of_weights_series = pd.Series(0.0, index=Q4_base.index, dtype=np.float32)
        for weight_series in Q4_confirm_weights_series.values():
            sum_of_weights_series += weight_series
        
        sum_of_weights_series_safe = sum_of_weights_series.replace(0, np.nan)
        
        Q4_trap_evidence = (weighted_sum / sum_of_weights_series_safe).fillna(0)
        
        Q4_final = (Q4_base * Q4_trap_evidence * -1).clip(-1, 0)
        _temp_debug_values["Q4: 价涨 & 优缩"] = {
            "Q4_base": Q4_base,
            "mf_outflow_risk": normalized_signals['main_force_net_flow_outflow_norm'],
            "Q4_trap_evidence": Q4_trap_evidence,
            "Q4_final": Q4_final
        }
        return Q4_final

    def _calculate_dynamic_weights(self, normalized_signals: Dict[str, pd.Series], config: Dict, df_index: pd.Index, method_name: str, _temp_debug_values: Dict) -> Dict[str, pd.Series]:
        """
        【V1.0.1 · 动态权重计算 - 情境指标替换版】
        - 核心职责: 根据情境调制信号计算动态权重调制因子。
        - 核心修复: 将缺失的 'market_stability_score_D' 替换为 'MA_POTENTIAL_ORDERLINESS_SCORE_D'。
        - 版本: 1.0.1
        """
        context_modulator_weights = config.get('context_modulator_weights', {})
        if not context_modulator_weights:
            # 如果没有配置动态权重，则返回原始配置中的固定权重
            return {
                'Q1_confirmation_weights': config.get('Q1_confirmation_weights', {}),
                'Q2_confirmation_weights': config.get('Q2_confirmation_weights', {}),
                'Q3_confirmation_weights': config.get('Q3_confirmation_weights', {}),
                'Q4_confirmation_weights': config.get('Q4_confirmation_weights', {})
            }
        # 基础调制因子，所有情境信号的加权平均
        modulator_components = {
            'volatility_inverse': normalized_signals['volatility_inverse_norm'],
            'trend_strength_inverse': normalized_signals['trend_strength_inverse_norm'],
            'sentiment_neutrality': normalized_signals['sentiment_neutrality_norm'],
            'liquidity_authenticity_score': normalized_signals['liquidity_authenticity_score_norm'],
            'ma_potential_orderliness_score': normalized_signals['ma_potential_orderliness_score_norm'], # 替换为ma_potential_orderliness_score_norm
            'microstructure_efficiency_index': normalized_signals['microstructure_efficiency_index_norm']
        }
        base_modulator = pd.Series(0.0, index=df_index, dtype=np.float32)
        total_modulator_weight = 0.0
        for signal_name, weight in context_modulator_weights.items():
            if signal_name in modulator_components:
                base_modulator += modulator_components[signal_name] * weight
                total_modulator_weight += weight
        if total_modulator_weight > 0:
            base_modulator /= total_modulator_weight
        else:
            base_modulator = pd.Series(0.5, index=df_index, dtype=np.float32) # 默认中性调制
        # 将 base_modulator 映射到 [0.8, 1.2] 范围，以避免过度调整
        modulator_factor = (base_modulator * 0.4 + 0.8).clip(0.8, 1.2) # 0.5 -> 1.0, 0 -> 0.8, 1 -> 1.2
        dynamic_weights = {}
        for q_key in ['Q1_confirmation_weights', 'Q2_confirmation_weights', 'Q3_confirmation_weights', 'Q4_confirmation_weights']:
            base_weights = config.get(q_key, {})
            current_dynamic_weights = {}
            for signal_name, weight in base_weights.items():
                # 简单地将基础权重乘以调制因子，可以根据需要设计更复杂的调整逻辑
                current_dynamic_weights[signal_name] = pd.Series(weight, index=df_index, dtype=np.float32) * modulator_factor
            dynamic_weights[q_key] = current_dynamic_weights
        _temp_debug_values["动态权重调制"] = {
            "base_modulator": base_modulator,
            "modulator_factor": modulator_factor
        }
        return dynamic_weights

    def _calculate_interaction_terms(self, fetched_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], config: Dict, df_index: pd.Index, _temp_debug_values: Dict) -> pd.Series:
        """
        【V1.0.0 · 交互项计算】
        - 核心职责: 计算信号间的交互项，捕捉协同效应。
        - 版本: 1.0.0
        """
        interaction_terms_weights = config.get('interaction_terms_weights', {})
        interaction_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        # Q1: 主力信念与资金流可信度协同
        if 'Q1_mf_conviction_flow_credibility_synergy' in interaction_terms_weights:
            synergy = normalized_signals['mtf_main_force_conviction'].clip(lower=0) * normalized_signals['flow_credibility_norm']
            interaction_score += synergy * interaction_terms_weights['Q1_mf_conviction_flow_credibility_synergy']
        # Q3: 打压吸筹与下影线吸收协同
        if 'Q3_suppressive_absorb_synergy' in interaction_terms_weights:
            synergy = normalized_signals['suppressive_accum_norm'] * fetched_signals['lower_shadow_absorb']
            interaction_score += synergy * interaction_terms_weights['Q3_suppressive_absorb_synergy']
        # Q4: 派发强度与主力净流出协同
        if 'Q4_distribution_mf_outflow_synergy' in interaction_terms_weights:
            synergy = normalized_signals['mtf_distribution_intensity'] * normalized_signals['main_force_net_flow_outflow_norm']
            interaction_score += synergy * interaction_terms_weights['Q4_distribution_mf_outflow_synergy']
        _temp_debug_values["交互项"] = {
            "interaction_score": interaction_score
        }
        return interaction_score

    def _calculate_dynamic_exponent(self, fetched_signals: Dict[str, pd.Series], config: Dict, df_index: pd.Index, _temp_debug_values: Dict) -> pd.Series:
        """
        【V1.0.0 · 动态指数计算】
        - 核心职责: 根据市场波动率计算动态指数调制因子。
        - 版本: 1.0.0
        """
        dynamic_exponent_params = config.get('dynamic_exponent_modulator_weights', {})
        if not dynamic_exponent_params.get('enabled', False):
            return pd.Series(dynamic_exponent_params.get('base_exponent', 1.0), index=df_index, dtype=np.float32)
        modulator_signal_name = dynamic_exponent_params.get('modulator_signal')
        sensitivity = dynamic_exponent_params.get('sensitivity', 0.5)
        base_exponent = dynamic_exponent_params.get('base_exponent', 1.0)
        min_exponent = dynamic_exponent_params.get('min_exponent', 1.0)
        max_exponent = dynamic_exponent_params.get('max_exponent', 2.0)
        modulator_signal = self.helper._get_safe_series(self.strategy.df, modulator_signal_name, 0.0, method_name="dynamic_exponent")
        # 归一化调制信号到 [0, 1]
        normalized_modulator = self.helper._normalize_series(modulator_signal, df_index, bipolar=False)
        # 根据归一化调制信号调整指数
        # 例如，波动率越高，指数越大，放大信号；波动率越低，指数越小，平滑信号
        dynamic_exponent = base_exponent + (normalized_modulator - 0.5) * sensitivity * (max_exponent - min_exponent) * 2
        dynamic_exponent = dynamic_exponent.clip(min_exponent, max_exponent)
        _temp_debug_values["动态指数调制"] = {
            "modulator_signal": modulator_signal,
            "normalized_modulator": normalized_modulator,
            "dynamic_exponent": dynamic_exponent
        }
        return dynamic_exponent
