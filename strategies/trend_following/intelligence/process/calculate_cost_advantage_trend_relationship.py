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
        【V1.0.2 · 调试上下文初始化 - 探针日期查找增强版】
        - 核心修复: 增强探针日期查找逻辑，确保找到有效探针日期
        - 核心优化: 添加详细的探针查找日志
        - 核心新增: 探针日期有效性验证
        - 版本: 1.0.2
        """
        # 检查调试参数
        is_debug_enabled_for_method = get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)
        
        print(f"【调试初始化】方法: {method_name}, 调试启用: {is_debug_enabled_for_method}")
        print(f"【调试初始化】探针日期列表: {self.probe_dates}")
        
        probe_ts = None
        
        if is_debug_enabled_for_method and self.probe_dates:
            # 将探针日期转换为datetime格式
            probe_dates_dt = []
            for d in self.probe_dates:
                try:
                    probe_dates_dt.append(pd.to_datetime(d).normalize())
                except Exception as e:
                    print(f"【调试初始化】探针日期转换错误 {d}: {e}")
            
            print(f"【调试初始化】转换后的探针日期: {probe_dates_dt}")
            
            # 从后往前查找匹配的日期
            for date in reversed(df.index):
                date_normalized = pd.to_datetime(date).tz_localize(None).normalize()
                if date_normalized in probe_dates_dt:
                    probe_ts = date
                    print(f"【调试初始化】找到匹配的探针日期: {probe_ts.strftime('%Y-%m-%d')}")
                    break
            
            if probe_ts is None:
                print(f"【调试初始化】未找到匹配的探针日期，尝试查找最近的有效日期")
                # 如果没有精确匹配，尝试查找最近的日期
                if probe_dates_dt:
                    # 获取最新的探针日期
                    latest_probe_date = max(probe_dates_dt)
                    # 查找数据集中不超过该日期的最近日期
                    for date in reversed(df.index):
                        date_normalized = pd.to_datetime(date).tz_localize(None).normalize()
                        if date_normalized <= latest_probe_date:
                            probe_ts = date
                            print(f"【调试初始化】使用最近的探针日期: {probe_ts.strftime('%Y-%m-%d')}")
                            break
        
        if probe_ts is None:
            print(f"【调试初始化】未设置有效探针日期，调试输出将被限制")
            is_debug_enabled_for_method = False
        
        debug_output = {}
        _temp_debug_values = {}
        
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算成本优势趋势关系..."] = ""
            print(f"【调试初始化】成功初始化调试上下文，探针时间: {probe_ts.strftime('%Y-%m-%d')}")
        else:
            print(f"【调试初始化】调试未启用或无有效探针日期")
        
        return is_debug_enabled_for_method, probe_ts, debug_output, _temp_debug_values

    def _log_debug_values(self, debug_output: Dict, _temp_debug_values: Dict, probe_ts: pd.Timestamp, method_name: str):
        """
        【V1.0.2 · 调试值日志输出 - 多时间维度探针增强版】
        - 核心新增: 添加多时间维度分析的探针输出
        - 核心优化: 确保各时间维度的信号值都能被记录和查看
        - 核心优化: 增强嵌套字典的输出格式
        - 版本: 1.0.2
        """
        print(f"【探针输出】开始输出调试信息，共有 {len(_temp_debug_values)} 个调试模块")
        
        for section, values_dict in _temp_debug_values.items():
            print(f"【探针输出】处理模块: {section}, 包含 {len(values_dict) if isinstance(values_dict, dict) else 1} 个值")
            
            if isinstance(values_dict, dict):
                debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- {section} ---"] = ""
                
                for key, value in values_dict.items():
                    try:
                        if isinstance(value, dict):
                            debug_output[f"        {key}:"] = ""
                            for sub_key, sub_value in value.items():
                                if isinstance(sub_value, pd.Series):
                                    val = sub_value.loc[probe_ts] if probe_ts in sub_value.index else np.nan
                                    debug_output[f"          {sub_key}: {val:.4f}"] = ""
                                elif isinstance(sub_value, (int, float, np.float32, np.float64)):
                                    debug_output[f"          {sub_key}: {sub_value:.4f}"] = ""
                                else:
                                    debug_output[f"          {sub_key}: {sub_value}"] = ""
                        elif isinstance(value, pd.Series):
                            val = value.loc[probe_ts] if probe_ts in value.index else np.nan
                            debug_output[f"        '{key}': {val:.4f}"] = ""
                        elif isinstance(value, (int, float, np.float32, np.float64)):
                            debug_output[f"        '{key}': {value:.4f}"] = ""
                        else:
                            debug_output[f"        '{key}': {value}"] = ""
                    except Exception as e:
                        debug_output[f"        '{key}': [输出错误: {e}]"] = ""
            else:
                debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- {section}: {values_dict} ---"] = ""
        
        # 特别处理多时间维度融合详情
        if "多时间维度融合详情" in _temp_debug_values:
            multi_time_details = _temp_debug_values["多时间维度融合详情"]
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 多时间维度融合统计 ---"] = ""
            
            for key, value in multi_time_details.items():
                if isinstance(value, dict):
                    debug_output[f"        {key}:"] = ""
                    for sub_key, sub_value in value.items():
                        debug_output[f"          {sub_key}: {sub_value}"] = ""
                elif isinstance(value, pd.Series):
                    val = value.loc[probe_ts] if probe_ts in value.index else np.nan
                    debug_output[f"        '{key}': {val:.4f}"] = ""
                else:
                    debug_output[f"        '{key}': {value}"] = ""
        
        # 输出所有调试信息
        print(f"【探针输出】开始打印调试信息到控制台")
        for key, value in debug_output.items():
            if value:
                print(f"{key}: {value}")
            else:
                print(key)

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V5.0 · 修复探针重复输出与计算逻辑优化版】计算成本优势趋势。
        - 核心修复: 移除重复的探针输出调用，解决信息重复打印问题
        - 核心修复: 检查各象限计算逻辑，确保合理性
        - 核心优化: 清理冗余调试代码，提升运行效率
        - 版本: 5.0
        """
        method_name = "CalculateCostAdvantageTrendRelationship"
        print(f"【开始计算】{method_name}，数据形状: {df.shape}")
        print(f"【多时间维度】使用斐波那契数列时间维度: 5, 13, 21, 55日进行趋势分析")
        
        # 初始化调试上下文
        is_debug_enabled_for_method, probe_ts, debug_output, _temp_debug_values = self._initialize_debug_context(method_name, df)
        
        print(f"【计算状态】调试启用: {is_debug_enabled_for_method}, 探针时间: {probe_ts.strftime('%Y-%m-%d') if probe_ts else '无'}")
        
        # 1. 获取MTF配置（包含多时间维度权重）
        _, _, _, mtf_slope_accel_weights = self._get_mtf_configs(config)
        print(f"【多时间维度配置】斜率周期权重: {mtf_slope_accel_weights.get('slope_periods', {})}")
        print(f"【多时间维度配置】加速度周期权重: {mtf_slope_accel_weights.get('accel_periods', {})}")
        
        # 2. 获取所需信号列表并进行快速验证
        required_signals = self._get_required_signals_list(mtf_slope_accel_weights)
        
        # 快速验证：只检查最近55天数据中是否存在必要信号
        recent_df = df.tail(55) if len(df) > 55 else df
        if not self.helper._validate_required_signals(recent_df, required_signals, method_name):
            print(f"【快速验证失败】最近55天缺少核心信号，返回默认值")
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        
        print(f"【快速验证通过】最近55天所有必需信号都存在")
        
        df_index = df.index
        
        # 3. 获取原始数据和MTF融合信号（直接使用斜率和加速度信号）
        fetched_signals = self._fetch_raw_and_mtf_signals(df, df_index, mtf_slope_accel_weights, method_name, _temp_debug_values)
        
        # 4. 计算高级协同效应
        try:
            advanced_synergy_score = self._calculate_advanced_synergy(fetched_signals, df, df_index, _temp_debug_values)
            advanced_synergy_score = advanced_synergy_score.fillna(0.25)
        except Exception as e:
            print(f"【协同效应错误】计算异常: {e}，使用保守值0.25")
            advanced_synergy_score = pd.Series(0.25, index=df_index)
        
        # 5. 归一化处理（使用多时间维度融合信号）
        normalized_signals = self._normalize_all_signals(df, df_index, fetched_signals, mtf_slope_accel_weights, method_name, _temp_debug_values)
        
        # 6. 计算动态权重
        dynamic_weights = self._calculate_dynamic_weights(normalized_signals, config, df_index, method_name, _temp_debug_values)
        
        # 7. 计算各象限分数
        print(f"【开始计算象限分数】基于多时间维度趋势分析")
        Q1_final = self._calculate_q1_healthy_rally(fetched_signals, normalized_signals, dynamic_weights, _temp_debug_values)
        Q2_final = self._calculate_q2_bearish_distribution(fetched_signals, normalized_signals, dynamic_weights, _temp_debug_values)
        Q3_final = self._calculate_q3_golden_pit(fetched_signals, normalized_signals, df_index, dynamic_weights, _temp_debug_values)
        Q4_final = self._calculate_q4_bull_trap(fetched_signals, normalized_signals, dynamic_weights, _temp_debug_values)
        
        # 8. 计算交互项
        interaction_score = self._calculate_interaction_terms(fetched_signals, normalized_signals, config, df_index, _temp_debug_values)
        interaction_score = interaction_score.fillna(0)
        
        # 9. 计算高级协同效应增强项
        synergy_enhancement = advanced_synergy_score * 0.2
        
        # --- 最终融合 ---
        base_fusion_score = (Q1_final + Q2_final + Q3_final + Q4_final).fillna(0)
        final_score_with_interaction = (base_fusion_score + interaction_score + synergy_enhancement).fillna(0)
        
        # 10. 计算动态指数
        dynamic_exponent = self._calculate_dynamic_exponent(fetched_signals, config, df, df_index, _temp_debug_values)
        
        # 应用动态指数进行非线性放大/平滑
        final_score_normalized_for_exponent = ((final_score_with_interaction + 1) / 2).clip(0, 1).fillna(0.5)
        final_score_exponentiated = final_score_normalized_for_exponent.pow(dynamic_exponent)
        final_score = (final_score_exponentiated * 2 - 1).clip(-1, 1)
        
        # 最终检查：确保没有NaN
        final_score = final_score.fillna(0)
        
        # 记录最终融合探针信息
        _temp_debug_values["最终融合"] = {
            "Q1_final": Q1_final,
            "Q2_final": Q2_final,
            "Q3_final": Q3_final,
            "Q4_final": Q4_final,
            "base_fusion_score": base_fusion_score,
            "interaction_score": interaction_score,
            "advanced_synergy_score": advanced_synergy_score,
            "synergy_enhancement": synergy_enhancement,
            "final_score_with_interaction": final_score_with_interaction,
            "dynamic_exponent": dynamic_exponent,
            "final_score_normalized_for_exponent": final_score_normalized_for_exponent,
            "final_score_exponentiated": final_score_exponentiated,
            "final_score": final_score
        }
        
        # 输出探针信息 - 仅调用一次_log_debug_values，避免重复输出
        if is_debug_enabled_for_method and probe_ts:
            print(f"【开始输出详细探针信息】")
            self._log_debug_values(debug_output, _temp_debug_values, probe_ts, method_name)
            # 不再调用self.helper._print_debug_output，避免重复输出
        else:
            # 输出关键统计信息
            print(f"【关键统计】各象限分数: Q1={Q1_final.mean():.4f}, Q2={Q2_final.mean():.4f}, Q3={Q3_final.mean():.4f}, Q4={Q4_final.mean():.4f}")
            print(f"【关键统计】多时间维度融合: 价格变化均值={fetched_signals['mtf_price_change'].mean():.4f}, 成本优势变化均值={fetched_signals['mtf_ca_change'].mean():.4f}")
        
        # 输出最近55天的统计信息
        if len(final_score) > 55:
            recent_final = final_score.tail(55)
        else:
            recent_final = final_score
            
        print(f"【最终结果】最近55天分数范围: [{recent_final.min():.4f}, {recent_final.max():.4f}]，均值: {recent_final.mean():.4f}")
        print(f"【最终结果】最近55天正分数比例: {(recent_final > 0).sum() / len(recent_final):.1%}")
        print(f"【最终结果】最近55天负分数比例: {(recent_final < 0).sum() / len(recent_final):.1%}")
        
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
        【V1.1.0 · 必需信号列表构建 - 修复信号缺失问题】
        - 核心修复: 将缺失的 'SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION' 替换为 'lower_shadow_absorption_strength_D'
        - 核心优化: 确保所有信号都在数据层中存在
        - 版本: 1.1.0
        """
        required_signals = [
            'pct_change_D', 'main_force_cost_advantage_D', 'main_force_conviction_index_D',
            'upward_impulse_purity_D', 'suppressive_accumulation_intensity_D',
            'lower_shadow_absorption_strength_D', 'distribution_at_peak_intensity_D',  # 替换为数据层存在的信号
            'active_selling_pressure_D', 'profit_taking_flow_ratio_D', 'active_buying_support_D',
            'main_force_net_flow_calibrated_D', 'flow_credibility_index_D',
            'close_D',  # 明确添加close_D，用于价格变化计算
            'VOLATILITY_INSTABILITY_INDEX_21d_D', 'ADX_14_D', 'market_sentiment_score_D',
            'liquidity_authenticity_score_D', 'MA_POTENTIAL_ORDERLINESS_SCORE_D', 'microstructure_efficiency_index_D',
            'main_force_buy_execution_alpha_D', 'main_force_sell_execution_alpha_D',
            'micro_price_impact_asymmetry_D'
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
        【V1.1.7 · 修复斜率和加速度信号归一化版】
        - 核心修复: 对斜率和加速度信号进行归一化处理，确保在合理范围内
        - 核心修复: 修复多时间维度融合后的值超出范围问题
        - 核心优化: 添加信号值范围检查和截断处理
        - 版本: 1.1.7
        """
        fetched_signals = {}
        
        print(f"【信号获取】使用多时间维度斜率和加速度信号，配置权重: {mtf_slope_accel_weights}")
        
        # 获取配置中的斜率和加速度权重
        slope_periods_weights = mtf_slope_accel_weights.get('slope_periods', {})
        accel_periods_weights = mtf_slope_accel_weights.get('accel_periods', {})
        
        # 1. 价格变化的多时间维度融合信号
        price_slope_components = {}
        price_accel_components = {}
        
        for period_str, weight in slope_periods_weights.items():
            slope_signal_name = f'SLOPE_{period_str}_close_D'
            if slope_signal_name in df.columns:
                raw_slope = self.helper._get_safe_series(df, slope_signal_name, 0.0, method_name=method_name)
                # 对斜率信号进行归一化，确保在合理范围内
                norm_slope = self.helper._normalize_series(raw_slope, df_index, bipolar=True)
                price_slope_components[period_str] = {
                    'series': norm_slope,
                    'weight': weight
                }
                print(f"【价格斜率】使用 {period_str}日斜率信号，权重: {weight}, 归一化范围: [{norm_slope.min():.4f}, {norm_slope.max():.4f}]")
            else:
                print(f"【价格斜率警告】{slope_signal_name} 不存在")
        
        for period_str, weight in accel_periods_weights.items():
            accel_signal_name = f'ACCEL_{period_str}_close_D'
            if accel_signal_name in df.columns:
                raw_accel = self.helper._get_safe_series(df, accel_signal_name, 0.0, method_name=method_name)
                # 对加速度信号进行归一化
                norm_accel = self.helper._normalize_series(raw_accel, df_index, bipolar=True)
                price_accel_components[period_str] = {
                    'series': norm_accel,
                    'weight': weight
                }
                print(f"【价格加速度】使用 {period_str}日加速度信号，权重: {weight}, 归一化范围: [{norm_accel.min():.4f}, {norm_accel.max():.4f}]")
            else:
                print(f"【价格加速度警告】{accel_signal_name} 不存在")
        
        # 计算加权融合的斜率和加速度
        mtf_price_slope = pd.Series(0.0, index=df_index)
        total_slope_weight = 0.0
        for period_str, component in price_slope_components.items():
            mtf_price_slope += component['series'] * component['weight']
            total_slope_weight += component['weight']
        
        mtf_price_accel = pd.Series(0.0, index=df_index)
        total_accel_weight = 0.0
        for period_str, component in price_accel_components.items():
            mtf_price_accel += component['series'] * component['weight']
            total_accel_weight += component['weight']
        
        # 归一化加权融合
        if total_slope_weight > 0:
            mtf_price_slope = mtf_price_slope / total_slope_weight
        if total_accel_weight > 0:
            mtf_price_accel = mtf_price_accel / total_accel_weight
        
        # 综合斜率和加速度得到价格变化信号，确保在[-1,1]范围内
        mtf_price_change_raw = (mtf_price_slope + mtf_price_accel) / 2
        fetched_signals['mtf_price_change'] = mtf_price_change_raw.clip(-1, 1)
        
        print(f"【价格融合】最终价格变化信号范围: [{fetched_signals['mtf_price_change'].min():.4f}, {fetched_signals['mtf_price_change'].max():.4f}]")
        
        # 2. 成本优势变化的多时间维度融合信号
        ca_slope_components = {}
        ca_accel_components = {}
        
        for period_str, weight in slope_periods_weights.items():
            slope_signal_name = f'SLOPE_{period_str}_main_force_cost_advantage_D'
            if slope_signal_name in df.columns:
                raw_slope = self.helper._get_safe_series(df, slope_signal_name, 0.0, method_name=method_name)
                norm_slope = self.helper._normalize_series(raw_slope, df_index, bipolar=True)
                ca_slope_components[period_str] = {
                    'series': norm_slope,
                    'weight': weight
                }
                print(f"【成本优势斜率】使用 {period_str}日斜率信号，权重: {weight}, 归一化范围: [{norm_slope.min():.4f}, {norm_slope.max():.4f}]")
            else:
                print(f"【成本优势斜率警告】{slope_signal_name} 不存在")
        
        for period_str, weight in accel_periods_weights.items():
            accel_signal_name = f'ACCEL_{period_str}_main_force_cost_advantage_D'
            if accel_signal_name in df.columns:
                raw_accel = self.helper._get_safe_series(df, accel_signal_name, 0.0, method_name=method_name)
                norm_accel = self.helper._normalize_series(raw_accel, df_index, bipolar=True)
                ca_accel_components[period_str] = {
                    'series': norm_accel,
                    'weight': weight
                }
                print(f"【成本优势加速度】使用 {period_str}日加速度信号，权重: {weight}, 归一化范围: [{norm_accel.min():.4f}, {norm_accel.max():.4f}]")
            else:
                print(f"【成本优势加速度警告】{accel_signal_name} 不存在")
        
        # 计算加权融合的斜率和加速度
        mtf_ca_slope = pd.Series(0.0, index=df_index)
        total_ca_slope_weight = 0.0
        for period_str, component in ca_slope_components.items():
            mtf_ca_slope += component['series'] * component['weight']
            total_ca_slope_weight += component['weight']
        
        mtf_ca_accel = pd.Series(0.0, index=df_index)
        total_ca_accel_weight = 0.0
        for period_str, component in ca_accel_components.items():
            mtf_ca_accel += component['series'] * component['weight']
            total_ca_accel_weight += component['weight']
        
        # 归一化加权融合
        if total_ca_slope_weight > 0:
            mtf_ca_slope = mtf_ca_slope / total_ca_slope_weight
        if total_ca_accel_weight > 0:
            mtf_ca_accel = mtf_ca_accel / total_ca_accel_weight
        
        # 综合斜率和加速度得到成本优势变化信号
        mtf_ca_change_raw = (mtf_ca_slope + mtf_ca_accel) / 2
        fetched_signals['mtf_ca_change'] = mtf_ca_change_raw.clip(-1, 1)
        
        print(f"【成本优势融合】最终成本优势变化信号范围: [{fetched_signals['mtf_ca_change'].min():.4f}, {fetched_signals['mtf_ca_change'].max():.4f}]")
        
        # 3. 获取其他原始信号
        signal_configs = [
            ('main_force_conviction_index_D', 'main_force_conviction', 0.0),
            ('upward_impulse_purity_D', 'upward_impulse_purity', 0.0),
            ('suppressive_accumulation_intensity_D', 'suppressive_accum', 0.0),
            ('lower_shadow_absorption_strength_D', 'lower_shadow_absorb', 0.0),
            ('distribution_at_peak_intensity_D', 'distribution_intensity', 0.0),
            ('active_selling_pressure_D', 'active_selling', 0.0),
            ('profit_taking_flow_ratio_D', 'profit_taking_flow', 0.0),
            ('active_buying_support_D', 'active_buying_support', 0.0),
            ('main_force_net_flow_calibrated_D', 'main_force_net_flow', 0.0),
            ('flow_credibility_index_D', 'flow_credibility', 0.0),
            ('close_D', 'close_price', 0.0),
            ('VOLATILITY_INSTABILITY_INDEX_21d_D', 'volatility_instability', 0.0),
            ('ADX_14_D', 'adx_trend_strength', 0.0),
            ('market_sentiment_score_D', 'market_sentiment', 0.0),
            ('liquidity_authenticity_score_D', 'liquidity_authenticity', 0.0),
            ('MA_POTENTIAL_ORDERLINESS_SCORE_D', 'ma_potential_orderliness_score', 0.0),
            ('microstructure_efficiency_index_D', 'microstructure_efficiency', 0.0),
            ('main_force_buy_execution_alpha_D', 'main_force_buy_execution_alpha', 0.0),
            ('main_force_sell_execution_alpha_D', 'main_force_sell_execution_alpha', 0.0),
            ('micro_price_impact_asymmetry_D', 'micro_price_impact_asymmetry', 0.0),
        ]
        
        for df_col_name, signal_name, default_value in signal_configs:
            signal_series = self.helper._get_safe_series(df, df_col_name, default_value, method_name=method_name)
            fetched_signals[signal_name] = signal_series
        
        # 4. 快速信号质量检查 - 只检查最近55天
        print(f"【快速质量检查】开始检查最近55天的信号质量...")
        signal_quality_reports = {}
        critical_issues = []
        
        for signal_name, signal_series in fetched_signals.items():
            quality_report = self._check_signal_quality(signal_series, signal_name, df_index, recent_days=55)
            signal_quality_reports[signal_name] = quality_report
            
            # 记录严重问题
            if quality_report["quality_level"] == "POOR":
                critical_issues.append(f"{signal_name}(NaN比例:{quality_report['nan_ratio']:.1%})")
        
        # 输出质量总结
        if critical_issues:
            print(f"【质量警告】发现{len(critical_issues)}个信号质量较差: {', '.join(critical_issues)}")
        else:
            print(f"【质量检查通过】所有信号最近55天质量合格")
        
        # 5. 记录多时间维度融合详情
        _temp_debug_values["原始信号值"] = {k: v for k, v in fetched_signals.items()}
        _temp_debug_values["信号质量报告"] = signal_quality_reports
        _temp_debug_values["多时间维度融合详情"] = {
            "价格斜率组件": {k: v['weight'] for k, v in price_slope_components.items()},
            "价格加速度组件": {k: v['weight'] for k, v in price_accel_components.items()},
            "成本优势斜率组件": {k: v['weight'] for k, v in ca_slope_components.items()},
            "成本优势加速度组件": {k: v['weight'] for k, v in ca_accel_components.items()},
            "mtf_price_slope": mtf_price_slope,
            "mtf_price_accel": mtf_price_accel,
            "mtf_ca_slope": mtf_ca_slope,
            "mtf_ca_accel": mtf_ca_accel,
            "mtf_price_change_range": f"[{fetched_signals['mtf_price_change'].min():.4f}, {fetched_signals['mtf_price_change'].max():.4f}]",
            "mtf_ca_change_range": f"[{fetched_signals['mtf_ca_change'].min():.4f}, {fetched_signals['mtf_ca_change'].max():.4f}]",
        }
        
        return fetched_signals

    def _normalize_all_signals(self, df: pd.DataFrame, df_index: pd.Index, fetched_signals: Dict[str, pd.Series], mtf_slope_accel_weights: Dict, method_name: str, _temp_debug_values: Dict) -> Dict[str, pd.Series]:
        """
        【V1.1.4 · 信号归一化处理 - 直接使用MTF斜率和加速度信号版】
        - 核心修改: 直接使用数据层中的MTF斜率和加速度信号进行归一化处理
        - 核心新增: 为每个需要MTF处理的信号创建多时间维度融合版本
        - 版本: 1.1.4
        """
        normalized_signals = {}
        
        print(f"【信号归一化】开始多时间维度信号归一化处理")
        
        # 获取配置中的斜率和加速度权重
        slope_periods_weights = mtf_slope_accel_weights.get('slope_periods', {})
        accel_periods_weights = mtf_slope_accel_weights.get('accel_periods', {})
        
        # 1. 主力信念指数的多时间维度融合
        mf_conviction_slope_components = {}
        mf_conviction_accel_components = {}
        
        for period_str, weight in slope_periods_weights.items():
            slope_signal_name = f'SLOPE_{period_str}_main_force_conviction_index_D'
            if slope_signal_name in df.columns:
                mf_conviction_slope_components[period_str] = {
                    'series': self.helper._get_safe_series(df, slope_signal_name, 0.0, method_name=method_name),
                    'weight': weight
                }
        
        for period_str, weight in accel_periods_weights.items():
            accel_signal_name = f'ACCEL_{period_str}_main_force_conviction_index_D'
            if accel_signal_name in df.columns:
                mf_conviction_accel_components[period_str] = {
                    'series': self.helper._get_safe_series(df, accel_signal_name, 0.0, method_name=method_name),
                    'weight': weight
                }
        
        # 计算加权融合
        mtf_mf_conviction_slope = pd.Series(0.0, index=df_index)
        total_slope_weight = 0.0
        for period_str, component in mf_conviction_slope_components.items():
            mtf_mf_conviction_slope += component['series'] * component['weight']
            total_slope_weight += component['weight']
        
        mtf_mf_conviction_accel = pd.Series(0.0, index=df_index)
        total_accel_weight = 0.0
        for period_str, component in mf_conviction_accel_components.items():
            mtf_mf_conviction_accel += component['series'] * component['weight']
            total_accel_weight += component['weight']
        
        if total_slope_weight > 0:
            mtf_mf_conviction_slope = mtf_mf_conviction_slope / total_slope_weight
        if total_accel_weight > 0:
            mtf_mf_conviction_accel = mtf_mf_conviction_accel / total_accel_weight
        
        normalized_signals['mtf_main_force_conviction'] = (mtf_mf_conviction_slope + mtf_mf_conviction_accel) / 2
        
        # 2. 上涨纯度多时间维度融合
        upward_purity_slope_components = {}
        upward_purity_accel_components = {}
        
        for period_str, weight in slope_periods_weights.items():
            slope_signal_name = f'SLOPE_{period_str}_upward_impulse_purity_D'
            if slope_signal_name in df.columns:
                upward_purity_slope_components[period_str] = {
                    'series': self.helper._get_safe_series(df, slope_signal_name, 0.0, method_name=method_name),
                    'weight': weight
                }
        
        for period_str, weight in accel_periods_weights.items():
            accel_signal_name = f'ACCEL_{period_str}_upward_impulse_purity_D'
            if accel_signal_name in df.columns:
                upward_purity_accel_components[period_str] = {
                    'series': self.helper._get_safe_series(df, accel_signal_name, 0.0, method_name=method_name),
                    'weight': weight
                }
        
        mtf_upward_purity_slope = pd.Series(0.0, index=df_index)
        total_upward_slope_weight = 0.0
        for period_str, component in upward_purity_slope_components.items():
            mtf_upward_purity_slope += component['series'] * component['weight']
            total_upward_slope_weight += component['weight']
        
        mtf_upward_purity_accel = pd.Series(0.0, index=df_index)
        total_upward_accel_weight = 0.0
        for period_str, component in upward_purity_accel_components.items():
            mtf_upward_purity_accel += component['series'] * component['weight']
            total_upward_accel_weight += component['weight']
        
        if total_upward_slope_weight > 0:
            mtf_upward_purity_slope = mtf_upward_purity_slope / total_upward_slope_weight
        if total_upward_accel_weight > 0:
            mtf_upward_purity_accel = mtf_upward_purity_accel / total_upward_accel_weight
        
        normalized_signals['mtf_upward_purity'] = (mtf_upward_purity_slope + mtf_upward_purity_accel) / 2
        
        # 3. 派发强度多时间维度融合
        distribution_slope_components = {}
        distribution_accel_components = {}
        
        for period_str, weight in slope_periods_weights.items():
            slope_signal_name = f'SLOPE_{period_str}_distribution_at_peak_intensity_D'
            if slope_signal_name in df.columns:
                distribution_slope_components[period_str] = {
                    'series': self.helper._get_safe_series(df, slope_signal_name, 0.0, method_name=method_name),
                    'weight': weight
                }
        
        for period_str, weight in accel_periods_weights.items():
            accel_signal_name = f'ACCEL_{period_str}_distribution_at_peak_intensity_D'
            if accel_signal_name in df.columns:
                distribution_accel_components[period_str] = {
                    'series': self.helper._get_safe_series(df, accel_signal_name, 0.0, method_name=method_name),
                    'weight': weight
                }
        
        mtf_distribution_slope = pd.Series(0.0, index=df_index)
        total_dist_slope_weight = 0.0
        for period_str, component in distribution_slope_components.items():
            mtf_distribution_slope += component['series'] * component['weight']
            total_dist_slope_weight += component['weight']
        
        mtf_distribution_accel = pd.Series(0.0, index=df_index)
        total_dist_accel_weight = 0.0
        for period_str, component in distribution_accel_components.items():
            mtf_distribution_accel += component['series'] * component['weight']
            total_dist_accel_weight += component['weight']
        
        if total_dist_slope_weight > 0:
            mtf_distribution_slope = mtf_distribution_slope / total_dist_slope_weight
        if total_dist_accel_weight > 0:
            mtf_distribution_accel = mtf_distribution_accel / total_dist_accel_weight
        
        normalized_signals['mtf_distribution_intensity'] = (mtf_distribution_slope + mtf_distribution_accel) / 2
        
        # 4. 主动卖压多时间维度融合
        active_selling_slope_components = {}
        active_selling_accel_components = {}
        
        for period_str, weight in slope_periods_weights.items():
            slope_signal_name = f'SLOPE_{period_str}_active_selling_pressure_D'
            if slope_signal_name in df.columns:
                active_selling_slope_components[period_str] = {
                    'series': self.helper._get_safe_series(df, slope_signal_name, 0.0, method_name=method_name),
                    'weight': weight
                }
        
        for period_str, weight in accel_periods_weights.items():
            accel_signal_name = f'ACCEL_{period_str}_active_selling_pressure_D'
            if accel_signal_name in df.columns:
                active_selling_accel_components[period_str] = {
                    'series': self.helper._get_safe_series(df, accel_signal_name, 0.0, method_name=method_name),
                    'weight': weight
                }
        
        mtf_active_selling_slope = pd.Series(0.0, index=df_index)
        total_active_slope_weight = 0.0
        for period_str, component in active_selling_slope_components.items():
            mtf_active_selling_slope += component['series'] * component['weight']
            total_active_slope_weight += component['weight']
        
        mtf_active_selling_accel = pd.Series(0.0, index=df_index)
        total_active_accel_weight = 0.0
        for period_str, component in active_selling_accel_components.items():
            mtf_active_selling_accel += component['series'] * component['weight']
            total_active_accel_weight += component['weight']
        
        if total_active_slope_weight > 0:
            mtf_active_selling_slope = mtf_active_selling_slope / total_active_slope_weight
        if total_active_accel_weight > 0:
            mtf_active_selling_accel = mtf_active_selling_accel / total_active_accel_weight
        
        normalized_signals['mtf_active_selling'] = (mtf_active_selling_slope + mtf_active_selling_accel) / 2
        
        # 5. 其他非MTF信号的归一化处理
        normalized_signals['suppressive_accum_norm'] = self.helper._normalize_series(fetched_signals['suppressive_accum'], df_index, bipolar=False)
        normalized_signals['profit_taking_flow_norm'] = self.helper._normalize_series(fetched_signals['profit_taking_flow'], df_index, bipolar=False)
        normalized_signals['active_buying_support_inverted_norm'] = 1 - self.helper._normalize_series(fetched_signals['active_buying_support'], df_index, bipolar=False)
        normalized_signals['main_force_net_flow_outflow_norm'] = self.helper._normalize_series(fetched_signals['main_force_net_flow'].clip(upper=0).abs(), df_index, bipolar=False)
        normalized_signals['flow_credibility_norm'] = self.helper._normalize_series(fetched_signals['flow_credibility'], df_index, bipolar=False)
        
        # 情境调制信号归一化
        normalized_signals['volatility_inverse_norm'] = 1 - self.helper._normalize_series(fetched_signals['volatility_instability'], df_index, bipolar=False)
        normalized_signals['trend_strength_inverse_norm'] = 1 - self.helper._normalize_series(fetched_signals['adx_trend_strength'], df_index, bipolar=False)
        normalized_signals['sentiment_neutrality_norm'] = 1 - self.helper._normalize_series(fetched_signals['market_sentiment'].abs(), df_index, bipolar=False)
        normalized_signals['liquidity_authenticity_score_norm'] = self.helper._normalize_series(fetched_signals['liquidity_authenticity'], df_index, bipolar=False)
        normalized_signals['ma_potential_orderliness_score_norm'] = self.helper._normalize_series(fetched_signals['ma_potential_orderliness_score'], df_index, bipolar=False)
        normalized_signals['microstructure_efficiency_index_norm'] = self.helper._normalize_series(fetched_signals['microstructure_efficiency'], df_index, bipolar=False)
        
        # 微观结构和订单流信号归一化
        normalized_signals['main_force_buy_execution_alpha_norm'] = self.helper._normalize_series(fetched_signals['main_force_buy_execution_alpha'], df_index, bipolar=False)
        normalized_signals['main_force_sell_execution_alpha_norm'] = self.helper._normalize_series(fetched_signals['main_force_sell_execution_alpha'], df_index, bipolar=False)
        normalized_signals['micro_price_impact_asymmetry_norm'] = self.helper._normalize_series(fetched_signals['micro_price_impact_asymmetry'], df_index, bipolar=False)
        
        # 记录归一化详情
        _temp_debug_values["归一化处理"] = {k: v for k, v in normalized_signals.items()}
        _temp_debug_values["多时间维度归一化详情"] = {
            "主力信念斜率组件": len(mf_conviction_slope_components),
            "主力信念加速度组件": len(mf_conviction_accel_components),
            "上涨纯度斜率组件": len(upward_purity_slope_components),
            "上涨纯度加速度组件": len(upward_purity_accel_components),
            "派发强度斜率组件": len(distribution_slope_components),
            "派发强度加速度组件": len(distribution_accel_components),
            "主动卖压斜率组件": len(active_selling_slope_components),
            "主动卖压加速度组件": len(active_selling_accel_components),
        }
        
        print(f"【信号归一化完成】已处理{len(normalized_signals)}个归一化信号")
        
        return normalized_signals

    def _calculate_q1_healthy_rally(self, fetched_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], dynamic_weights: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V1.1.3 · Q1健康上涨计算 - 修复值溢出与范围控制版】
        - 核心修复: 确保Q1_base计算在合理范围[0,1]内
        - 核心修复: 防止Q1_final超出clip上限
        - 核心优化: 添加计算范围检查和安全处理
        - 版本: 1.1.3
        """
        # 确保输入信号在[-1,1]范围内
        price_change_clipped = fetched_signals['mtf_price_change'].clip(-1, 1)
        ca_change_clipped = fetched_signals['mtf_ca_change'].clip(-1, 1)
        
        # Q1基础计算：只考虑正的价格变化和成本优势变化
        price_positive = price_change_clipped.clip(lower=0)
        ca_positive = ca_change_clipped.clip(lower=0)
        
        # 使用几何平均而不是简单相乘，避免值过大
        Q1_base = (price_positive * ca_positive).pow(0.5)
        
        # 确保基础值在[0,1]范围内
        Q1_base = Q1_base.clip(0, 1)
        
        # 确认：主力信念、上涨纯度、资金流可信度、主力买入执行Alpha
        Q1_confirm_components = {
            'mtf_main_force_conviction': normalized_signals['mtf_main_force_conviction'].clip(lower=0),
            'mtf_upward_purity': normalized_signals['mtf_upward_purity'].clip(0, 1),
            'flow_credibility_norm': normalized_signals['flow_credibility_norm'].clip(0, 1),
            'main_force_buy_execution_alpha_norm': normalized_signals['main_force_buy_execution_alpha_norm'].clip(0, 1)
        }
        
        Q1_confirm_weights_series = dynamic_weights['Q1_confirmation_weights']
        
        # 计算加权和
        weighted_sum = pd.Series(0.0, index=Q1_base.index, dtype=np.float32)
        for k, component_series in Q1_confirm_components.items():
            weight_series = Q1_confirm_weights_series.get(k, pd.Series(0.0, index=Q1_base.index, dtype=np.float32))
            weighted_sum += component_series * weight_series
        
        # 计算所有权重的总和
        sum_of_weights_series = pd.Series(0.0, index=Q1_base.index, dtype=np.float32)
        for weight_series in Q1_confirm_weights_series.values():
            sum_of_weights_series += weight_series
        
        # 避免除以零
        sum_of_weights_series_safe = sum_of_weights_series.replace(0, np.nan)
        
        # 执行除法，并用0填充NaN
        Q1_confirm = (weighted_sum / sum_of_weights_series_safe).fillna(0)
        
        # 确保确认值在[0,1]范围内
        Q1_confirm = Q1_confirm.clip(0, 1)
        
        # 最终计算，确保在[0,1]范围内
        Q1_final = (Q1_base * Q1_confirm).clip(0, 1)
        
        # 探针输出
        _temp_debug_values["Q1: 价涨 & 优扩"] = {
            "Q1_base": Q1_base,
            "Q1_confirm": Q1_confirm,
            "Q1_final": Q1_final,
            "price_change_range": f"[{price_change_clipped.min():.4f}, {price_change_clipped.max():.4f}]",
            "ca_change_range": f"[{ca_change_clipped.min():.4f}, {ca_change_clipped.max():.4f}]",
            "price_positive_range": f"[{price_positive.min():.4f}, {price_positive.max():.4f}]",
            "ca_positive_range": f"[{ca_positive.min():.4f}, {ca_positive.max():.4f}]"
        }
        
        # 输出统计信息
        print(f"【Q1计算】Q1_base均值: {Q1_base.mean():.4f}, 范围: [{Q1_base.min():.4f}, {Q1_base.max():.4f}]")
        print(f"【Q1计算】Q1_confirm均值: {Q1_confirm.mean():.4f}, Q1_final均值: {Q1_final.mean():.4f}")
        
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
        【V1.1.2 · Q3黄金坑计算 - 修复异常值与信号质量增强版】
        - 核心修复: 检查lower_shadow_absorb信号质量，处理异常值
        - 核心优化: 确保Q3_confirm计算在合理范围内[0,1]
        - 核心新增: 添加信号质量检查和数据清理逻辑
        - 版本: 1.1.2
        """
        Q3_base = (fetched_signals['mtf_price_change'].clip(upper=0).abs() * fetched_signals['mtf_ca_change'].clip(lower=0)).pow(0.5)
        # 检查lower_shadow_absorb信号质量
        lower_shadow_absorb_series = fetched_signals['lower_shadow_absorb']
        if lower_shadow_absorb_series is None or lower_shadow_absorb_series.isna().all():
            print(f"【Q3警告】lower_shadow_absorb信号全部为NaN，使用0替代")
            lower_shadow_absorb_norm = pd.Series(0.0, index=df_index)
        else:
            # 归一化处理，确保在[0,1]范围内
            lower_shadow_absorb_norm = self.helper._normalize_series(lower_shadow_absorb_series, df_index, bipolar=False)
            # 检查是否有异常值
            if (lower_shadow_absorb_norm > 10).any():
                print(f"【Q3警告】lower_shadow_absorb归一化值存在异常(>10)，进行截断处理")
                lower_shadow_absorb_norm = lower_shadow_absorb_norm.clip(0, 1)
        # 确认：隐蔽吸筹、下影线吸收、资金流可信度、流动性真实性
        Q3_confirm_components = {
            'suppressive_accum_norm': normalized_signals['suppressive_accum_norm'],
            'lower_shadow_absorb': lower_shadow_absorb_norm,  # 使用处理后的归一化值
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
        # 确保Q3_confirm在合理范围内
        Q3_confirm = Q3_confirm.clip(0, 1)
        # 前置下跌上下文，如果前几日有深跌，则增加黄金坑的权重
        pre_5day_pct_change = fetched_signals['close_price'].pct_change(periods=5).shift(1).fillna(0)
        norm_pre_drop_5d = self.helper._normalize_series(pre_5day_pct_change.clip(upper=0).abs(), df_index, bipolar=False)
        pre_drop_context_bonus = norm_pre_drop_5d * 0.5
        Q3_final = (Q3_base * Q3_confirm * (1 + pre_drop_context_bonus)).clip(0, 1)
        # 探针输出
        _temp_debug_values["Q3: 价跌 & 优扩"] = {
            "Q3_base": Q3_base,
            "Q3_confirm": Q3_confirm,
            "pre_5day_pct_change": pre_5day_pct_change,
            "norm_pre_drop_5d": norm_pre_drop_5d,
            "pre_drop_context_bonus": pre_drop_context_bonus,
            "Q3_final": Q3_final
        }
        # 输出统计信息
        print(f"【Q3计算】Q3_base均值: {Q3_base.mean():.4f}, Q3_confirm均值: {Q3_confirm.mean():.4f}, 范围: [{Q3_confirm.min():.4f}, {Q3_confirm.max():.4f}]")
        print(f"【Q3计算】前置下跌均值: {norm_pre_drop_5d.mean():.4f}, 最终分数均值: {Q3_final.mean():.4f}")
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

    def _calculate_dynamic_exponent(self, fetched_signals: Dict[str, pd.Series], config: Dict, df: pd.DataFrame, df_index: pd.Index, _temp_debug_values: Dict) -> pd.Series:
        """
        【V1.1.0 · 动态指数计算 - 修复AttributeError与探针增强版】
        - 核心修复: 移除对 self.strategy.df 的依赖，使用传入的 df 参数
        - 核心优化: 增加探针输出，便于调试动态指数计算过程
        - 核心优化: 添加指数计算的安全性检查
        - 版本: 1.1.0
        """
        dynamic_exponent_params = config.get('dynamic_exponent_modulator_weights', {})
        if not dynamic_exponent_params.get('enabled', False):
            base_exponent = dynamic_exponent_params.get('base_exponent', 1.0)
            print(f"【动态指数】动态指数计算未启用，使用固定指数: {base_exponent}")
            return pd.Series(base_exponent, index=df_index, dtype=np.float32)
        modulator_signal_name = dynamic_exponent_params.get('modulator_signal', 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        sensitivity = dynamic_exponent_params.get('sensitivity', 0.5)
        base_exponent = dynamic_exponent_params.get('base_exponent', 1.5)
        min_exponent = dynamic_exponent_params.get('min_exponent', 1.0)
        max_exponent = dynamic_exponent_params.get('max_exponent', 2.0)
        # 修复：直接从 df 中获取调制信号，而不是使用 self.strategy.df
        modulator_signal = self.helper._get_safe_series(df, modulator_signal_name, 0.0, method_name="dynamic_exponent")
        # 检查调制信号是否存在
        if modulator_signal is None or modulator_signal.isna().all():
            print(f"【动态指数警告】调制信号 '{modulator_signal_name}' 不存在或全部为NaN，使用默认指数")
            return pd.Series(base_exponent, index=df_index, dtype=np.float32)
        # 归一化调制信号到 [0, 1]
        normalized_modulator = self.helper._normalize_series(modulator_signal, df_index, bipolar=False)
        # 根据归一化调制信号调整指数
        # 调制信号越高，指数越大，放大信号；调制信号越低，指数越小，平滑信号
        dynamic_exponent = base_exponent + (normalized_modulator - 0.5) * sensitivity * (max_exponent - min_exponent) * 2
        dynamic_exponent = dynamic_exponent.clip(min_exponent, max_exponent)
        # 探针输出
        _temp_debug_values["动态指数调制"] = {
            "modulator_signal_name": modulator_signal_name,
            "modulator_signal_raw": modulator_signal,
            "normalized_modulator": normalized_modulator,
            "dynamic_exponent_raw": dynamic_exponent,
            "dynamic_exponent_clipped": dynamic_exponent.clip(min_exponent, max_exponent),
            "statistics": {
                "调制信号均值": modulator_signal.mean(),
                "归一化调制均值": normalized_modulator.mean(),
                "动态指数均值": dynamic_exponent.mean(),
                "动态指数最小值": dynamic_exponent.min(),
                "动态指数最大值": dynamic_exponent.max()
            }
        }
        print(f"【动态指数】调制信号: {modulator_signal_name}, 均值: {modulator_signal.mean():.4f}, 归一化均值: {normalized_modulator.mean():.4f}")
        print(f"【动态指数】计算值范围: [{dynamic_exponent.min():.4f}, {dynamic_exponent.max():.4f}], 均值: {dynamic_exponent.mean():.4f}")
        return dynamic_exponent.clip(min_exponent, max_exponent)

    def _calculate_advanced_synergy(self, fetched_signals: Dict[str, pd.Series], df: pd.DataFrame, df_index: pd.Index, _temp_debug_values: Dict) -> pd.Series:
        """
        【V1.0.5 · 高级协同效应计算 - 复合流动性替代与探针增强版】
        - 核心修复: 使用复合流动性指标替代BID_LIQUIDITY_FRACTAL
        - 核心优化: 构建流动性健康度指数，结合多个流动性相关指标
        - 核心新增: 详细的替代方案探针输出
        - 版本: 1.0.5
        """
        print(f"【协同效应】开始计算高级协同效应，检查所需信号...")
        
        # 1. 获取分形维数信号
        if 'FRACTAL_DIMENSION_89d_D' not in df.columns:
            print(f"【协同效应】FRACTAL_DIMENSION_89d_D信号不存在，使用默认值1.5")
            fractal_dimension = pd.Series(1.5, index=df_index)
        else:
            fractal_dimension = self.helper._get_safe_series(df, 'FRACTAL_DIMENSION_89d_D', 1.5, method_name="advanced_synergy")
        
        # 2. 获取赫斯特指数信号 - 检查可能的列名变体
        hurst_exponent = None
        hurst_column_names = ['HURST_144d_D', 'HURST_144d_d', 'hurst_144d_D', 'HURST_EXPONENT_144d_D']
        
        for col_name in hurst_column_names:
            if col_name in df.columns:
                print(f"【协同效应】找到赫斯特指数信号: {col_name}")
                hurst_exponent = self.helper._get_safe_series(df, col_name, 0.5, method_name="advanced_synergy")
                break
        
        if hurst_exponent is None:
            print(f"【协同效应】所有赫斯特指数信号均不存在，使用默认值0.5")
            hurst_exponent = pd.Series(0.5, index=df_index)
        
        # 3. 构建复合流动性健康度指数（替代BID_LIQUIDITY_FRACTAL）
        print(f"【协同效应】构建复合流动性健康度指数替代BID_LIQUIDITY_FRACTAL...")
        liquidity_components = {}
        
        # 可用流动性指标列表（按优先级排序）
        liquidity_indicators = [
            ('order_book_liquidity_supply_D', '订单簿流动性供给', 0.3),
            ('bid_side_liquidity_D', '买方流动性深度', 0.25),
            ('liquidity_authenticity_score_D', '流动性真实性', 0.2),
            ('liquidity_slope_D', '流动性斜率', 0.15),
            ('microstructure_efficiency_index_D', '微观结构效率', 0.1)
        ]
        
        available_indicators = []
        liquidity_scores = []
        
        for col_name, indicator_name, weight in liquidity_indicators:
            if col_name in df.columns:
                indicator_series = self.helper._get_safe_series(df, col_name, np.nan, method_name="advanced_synergy")
                if indicator_series is not None and not indicator_series.isna().all():
                    # 归一化到[0,1]范围
                    normalized_indicator = self.helper._normalize_series(indicator_series, df_index, bipolar=False)
                    liquidity_scores.append(normalized_indicator * weight)
                    available_indicators.append(indicator_name)
                    print(f"【协同效应】使用 {indicator_name} (权重: {weight})")
                else:
                    print(f"【协同效应】{indicator_name} 信号质量差，跳过")
            else:
                print(f"【协同效应】{indicator_name} 信号不存在")
        
        # 计算复合流动性分数
        if liquidity_scores:
            composite_liquidity_score = pd.Series(0.0, index=df_index)
            for score in liquidity_scores:
                composite_liquidity_score += score.fillna(0)
            
            # 如果使用了部分指标，重新调整权重
            if len(liquidity_scores) > 0:
                liquidity_fractal_score = composite_liquidity_score / sum([w for _, _, w in liquidity_indicators if w])
                liquidity_fractal_score = liquidity_fractal_score.clip(0, 1).fillna(0.5)
                print(f"【协同效应】使用 {len(available_indicators)} 个流动性指标: {', '.join(available_indicators)}")
            else:
                print(f"【协同效应】无可用流动性指标，使用中性值0.5")
                liquidity_fractal_score = pd.Series(0.5, index=df_index)
        else:
            print(f"【协同效应】无可用流动性指标，使用中性值0.5")
            liquidity_fractal_score = pd.Series(0.5, index=df_index)
        
        # 4. 计算分形效率：分形维数越接近1.5，市场效率越高
        fractal_efficiency = 1.0 - (fractal_dimension - 1.5).abs() / 0.5
        fractal_efficiency = fractal_efficiency.clip(0, 1).fillna(0.5)
        
        # 5. 计算市场记忆效应：赫斯特指数越接近1，持久性越强
        market_memory = (hurst_exponent - 0.5).abs() * 2
        market_memory = market_memory.clip(0, 1).fillna(0.5)
        
        # 6. 计算多尺度协同效应
        fractal_synergy = fractal_efficiency * market_memory * liquidity_fractal_score
        fractal_synergy = fractal_synergy.fillna(0.25)
        
        # 7. 探针输出 - 详细的替代方案信息
        if len(fractal_synergy) > 55:
            recent_synergy = fractal_synergy.tail(55)
        else:
            recent_synergy = fractal_synergy
        
        # 记录详细的探针信息
        _temp_debug_values["高级协同效应"] = {
            "分形维数": fractal_dimension,
            "赫斯特指数": hurst_exponent,
            "分形效率": fractal_efficiency,
            "市场记忆": market_memory,
            "流动性分数": liquidity_fractal_score,
            "协同分数": fractal_synergy,
            "统计信息": {
                "fractal_synergy_recent_mean": recent_synergy.mean(),
                "fractal_synergy_recent_std": recent_synergy.std(),
                "fractal_efficiency_mean": fractal_efficiency.mean(),
                "market_memory_mean": market_memory.mean(),
                "liquidity_fractal_score_mean": liquidity_fractal_score.mean(),
                "可用流动性指标数": len(available_indicators),
                "可用流动性指标": available_indicators
            }
        }
        
        print(f"【高级协同效应】分形效率均值: {fractal_efficiency.mean():.4f}, 市场记忆均值: {market_memory.mean():.4f}")
        print(f"【高级协同效应】流动性分数均值: {liquidity_fractal_score.mean():.4f}, 协同分数均值: {fractal_synergy.mean():.4f}")
        print(f"【高级协同效应】使用 {len(available_indicators)} 个流动性指标替代BID_LIQUIDITY_FRACTAL")
        
        return fractal_synergy

    def _enhance_with_market_regime(self, df: pd.DataFrame, final_score: pd.Series, df_index: pd.Index, _temp_debug_values: Dict) -> pd.Series:
        """
        【V1.0.0 · 市场状态增强 - 基于市场阶段和结构潜力】
        - 核心新增: 根据市场阶段（趋势/震荡）和结构潜力调整最终分数
        - 核心目标: 在不同市场环境下优化信号的可靠性
        - 数据来源: 使用数据层中的 MARKET_PHASE_D, structural_potential_score_D, trend_conviction_score_D
        - 版本: 1.0.0
        """
        # 获取市场状态指标
        market_phase = self.helper._get_safe_series(df, 'MARKET_PHASE_D', 0.0, method_name="market_regime")
        structural_potential = self.helper._get_safe_series(df, 'structural_potential_score_D', 0.5, method_name="market_regime")
        trend_conviction = self.helper._get_safe_series(df, 'trend_conviction_score_D', 0.5, method_name="market_regime")
        # 归一化处理
        market_phase_norm = self.helper._normalize_series(market_phase, df_index, bipolar=False)
        structural_potential_norm = self.helper._normalize_series(structural_potential, df_index, bipolar=False)
        trend_conviction_norm = self.helper._normalize_series(trend_conviction, df_index, bipolar=False)
        # 计算市场状态因子
        # 趋势市场下，增强趋势信号的权重；震荡市场下，降低趋势信号的权重
        trend_market_factor = market_phase_norm  # 假设市场阶段指标越大代表趋势越强
        # 结构潜力因子：结构潜力越高，信号可靠性越高
        structure_factor = structural_potential_norm
        # 趋势信念因子：趋势信念越强，信号可靠性越高
        conviction_factor = trend_conviction_norm
        # 综合市场状态因子
        market_regime_factor = (trend_market_factor * 0.4 + structure_factor * 0.3 + conviction_factor * 0.3).clip(0.5, 1.5)
        # 应用市场状态因子
        enhanced_score = final_score * market_regime_factor
        enhanced_score = enhanced_score.clip(-1, 1)
        # 探针输出
        _temp_debug_values["市场状态增强"] = {
            "market_phase_raw": market_phase,
            "structural_potential_raw": structural_potential,
            "trend_conviction_raw": trend_conviction,
            "market_phase_norm": market_phase_norm,
            "structural_potential_norm": structural_potential_norm,
            "trend_conviction_norm": trend_conviction_norm,
            "trend_market_factor": trend_market_factor,
            "structure_factor": structure_factor,
            "conviction_factor": conviction_factor,
            "market_regime_factor": market_regime_factor,
            "final_score_before": final_score,
            "enhanced_score": enhanced_score
        }
        print(f"【市场状态增强】市场阶段均值: {market_phase.mean():.4f}，结构潜力均值: {structural_potential.mean():.4f}")
        print(f"【市场状态增强】趋势信念均值: {trend_conviction.mean():.4f}，市场状态因子均值: {market_regime_factor.mean():.4f}")
        return enhanced_score

    def _check_signal_quality(self, signal_series: pd.Series, signal_name: str, df_index: pd.Index, recent_days: int = 55) -> Dict[str, Any]:
        """
        【V1.2.0 · 信号质量检查 - 优化警告阈值与关键信号聚焦】
        - 核心优化: 降低警告阈值，避免过多警告信息干扰
        - 核心优化: 只对关键信号输出详细警告
        - 核心新增: 区分关键信号和非关键信号，采取不同的质量要求
        - 版本: 1.2.0
        """
        # 定义关键信号列表 - 这些信号对策略至关重要
        critical_signals = [
            'mtf_price_change', 'mtf_ca_change', 'main_force_conviction',
            'main_force_net_flow', 'flow_credibility', 'close_price'
        ]
        
        # 只检查最近recent_days天的数据
        if len(signal_series) > recent_days:
            recent_signal = signal_series.tail(recent_days)
            total_count = recent_days
        else:
            recent_signal = signal_series
            total_count = len(signal_series)
        
        # 快速质量检查 - 只统计关键指标
        nan_count = recent_signal.isna().sum()
        zero_count = (recent_signal == 0).sum() if not recent_signal.isna().all() else 0
        
        quality_report = {
            "signal_name": signal_name,
            "is_critical": signal_name in critical_signals,
            "recent_days_checked": total_count,
            "nan_count": nan_count,
            "nan_ratio": nan_count / total_count if total_count > 0 else 1.0,
            "zero_count": zero_count,
            "zero_ratio": zero_count / total_count if total_count > 0 else 0,
            "recent_mean": recent_signal.mean() if not recent_signal.isna().all() and total_count > 0 else np.nan,
            "recent_std": recent_signal.std() if not recent_signal.isna().all() and total_count > 0 else np.nan,
        }
        
        # 质量评级 - 根据信号重要性调整阈值
        if signal_name in critical_signals:
            # 关键信号使用更严格的标准
            if quality_report["nan_ratio"] > 0.3:
                quality_report["quality_level"] = "POOR"
                quality_report["suggestion"] = f"关键信号缺失严重，影响策略准确性"
            elif quality_report["nan_ratio"] > 0.1:
                quality_report["quality_level"] = "MEDIUM"
                quality_report["suggestion"] = f"关键信号部分缺失"
            else:
                quality_report["quality_level"] = "GOOD"
                quality_report["suggestion"] = "关键信号质量良好"
        else:
            # 非关键信号使用较宽松的标准
            if quality_report["nan_ratio"] > 0.5:
                quality_report["quality_level"] = "POOR"
                quality_report["suggestion"] = f"信号缺失较多"
            elif quality_report["nan_ratio"] > 0.2:
                quality_report["quality_level"] = "MEDIUM"
                quality_report["suggestion"] = f"信号部分缺失"
            else:
                quality_report["quality_level"] = "GOOD"
                quality_report["suggestion"] = "信号质量可接受"
        
        # 只输出关键信号的质量警告，或质量特别差的信号
        if signal_name in critical_signals and quality_report["quality_level"] != "GOOD":
            print(f"【关键信号质量】{signal_name}: {quality_report['quality_level']}, 最近{recent_days}天NaN比例: {quality_report['nan_ratio']:.1%}")
        elif quality_report["quality_level"] == "POOR":
            print(f"【信号质量警告】{signal_name}: {quality_report['quality_level']}, 最近{recent_days}天NaN比例: {quality_report['nan_ratio']:.1%}")
        
        return quality_report




