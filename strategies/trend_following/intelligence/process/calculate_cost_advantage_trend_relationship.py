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
        【V1.0.1 · 调试值日志输出 - 完整性修复版】
        - 核心修复: 确保所有探针信息都能输出，包括嵌套字典和Series
        - 核心优化: 添加探针输出的完整性检查，确保不会遗漏任何信息
        - 核心新增: 支持更多数据类型的输出格式
        - 版本: 1.0.1
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
        
        # 输出所有调试信息
        print(f"【探针输出】开始打印调试信息到控制台")
        for key, value in debug_output.items():
            if value:
                print(f"{key}: {value}")
            else:
                print(key)

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V4.5 · 斐波那契时间窗口全面优化版】计算成本优势趋势。
        - 核心优化: 全面采用55天斐波那契时间窗口进行质量检查和探针输出
        - 核心优化: 大幅减少计算开销，提高运行效率
        - 核心优化: 探针输出只关注最近55天，提高可读性
        - 版本: 4.5
        """
        method_name = "CalculateCostAdvantageTrendRelationship"
        print(f"【开始计算】{method_name}，数据形状: {df.shape}，使用55天斐波那契窗口进行快速检查")
        
        is_debug_enabled_for_method, probe_ts, debug_output, _temp_debug_values = self._initialize_debug_context(method_name, df)
        
        # 1. 获取MTF配置
        _, _, _, mtf_slope_accel_weights = self._get_mtf_configs(config)
        
        # 2. 获取所需信号列表并进行快速验证
        required_signals = self._get_required_signals_list(mtf_slope_accel_weights)
        
        # 快速验证：只检查最近55天数据中是否存在必要信号
        recent_df = df.tail(55) if len(df) > 55 else df
        if not self.helper._validate_required_signals(recent_df, required_signals, method_name):
            print(f"【快速验证失败】最近55天缺少核心信号，返回默认值")
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        
        print(f"【快速验证通过】最近55天所有必需信号都存在")
        
        df_index = df.index
        
        # 3. 获取原始数据和MTF融合信号（使用55天窗口快速检查）
        fetched_signals = self._fetch_raw_and_mtf_signals(df, df_index, mtf_slope_accel_weights, method_name, _temp_debug_values)
        
        # 4. 计算高级协同效应
        try:
            advanced_synergy_score = self._calculate_advanced_synergy(fetched_signals, df, df_index, _temp_debug_values)
            advanced_synergy_score = advanced_synergy_score.fillna(0.5)
        except Exception as e:
            print(f"【协同效应错误】计算异常: {e}，使用中性值0.5")
            advanced_synergy_score = pd.Series(0.5, index=df_index)
        
        # 5. 归一化处理
        normalized_signals = self._normalize_all_signals(df, df_index, fetched_signals, mtf_slope_accel_weights, method_name, _temp_debug_values)
        
        # 6. 计算动态权重
        dynamic_weights = self._calculate_dynamic_weights(normalized_signals, config, df_index, method_name, _temp_debug_values)
        
        # 7. 计算各象限分数
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
        
        # 输出最近55天的统计信息（更有参考价值）
        if len(final_score) > 55:
            recent_final = final_score.tail(55)
        else:
            recent_final = final_score
            
        print(f"【最终结果】最近55天分数范围: [{recent_final.min():.4f}, {recent_final.max():.4f}]，均值: {recent_final.mean():.4f}")
        print(f"【最终结果】最近55天正分数比例: {(recent_final > 0).sum() / len(recent_final):.1%}")
        
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
        【V1.1.4 · 原始及MTF信号获取 - 斐波那契时间窗口优化版】
        - 核心优化: 使用55天窗口进行快速信号质量检查，大幅提高效率
        - 核心优化: 只对近期数据进行详细统计，减少计算开销
        - 核心优化: 保留完整数据用于计算，仅质量检查使用55天窗口
        - 版本: 1.1.4
        """
        fetched_signals = {}
        
        print(f"【信号获取】使用55天斐波那契窗口进行信号质量快速检查")
        
        # 价格变化和成本优势变化改为MTF融合信号
        fetched_signals['mtf_price_change'] = self.helper._get_mtf_slope_accel_score(df, 'close_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        fetched_signals['mtf_ca_change'] = self.helper._get_mtf_slope_accel_score(df, 'main_force_cost_advantage_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        
        # 定义需要获取的信号列表及其默认值
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
        
        # 获取所有信号
        for df_col_name, signal_name, default_value in signal_configs:
            signal_series = self.helper._get_safe_series(df, df_col_name, default_value, method_name=method_name)
            fetched_signals[signal_name] = signal_series
        
        # 快速信号质量检查 - 只检查最近55天
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
        
        _temp_debug_values["原始信号值"] = {k: v for k, v in fetched_signals.items()}
        _temp_debug_values["信号质量报告"] = signal_quality_reports
        
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
        【V1.0.3 · 高级协同效应计算 - 修复列名错误与信号缺失处理】
        - 核心修复: 修正列名 HURST_144d_d 为 HURST_144d_D
        - 核心优化: 增强信号存在性检查，提供更明确的错误信息
        - 核心优化: 当关键信号缺失时使用替代计算方案
        - 版本: 1.0.3
        """
        print(f"【协同效应】开始计算高级协同效应，检查所需信号...")
        
        # 检查分形维数信号
        if 'FRACTAL_DIMENSION_89d_D' not in df.columns:
            print(f"【协同效应错误】FRACTAL_DIMENSION_89d_D 信号不存在，使用替代计算")
            fractal_dimension = pd.Series(1.5, index=df_index)
        else:
            fractal_dimension = self.helper._get_safe_series(df, 'FRACTAL_DIMENSION_89d_D', 1.5, method_name="advanced_synergy")
        
        # 检查赫斯特指数信号 - 修正列名
        if 'HURST_144d_D' not in df.columns:
            print(f"【协同效应错误】HURST_144d_D 信号不存在，检查替代名称...")
            # 尝试可能的列名变体
            possible_names = ['HURST_144d_D', 'HURST_144d_d', 'HURST_144_D', 'hurst_144d_D']
            hurst_series = None
            for name in possible_names:
                if name in df.columns:
                    print(f"【协同效应】找到替代信号: {name}")
                    hurst_series = self.helper._get_safe_series(df, name, 0.5, method_name="advanced_synergy")
                    break
            
            if hurst_series is None:
                print(f"【协同效应警告】所有赫斯特指数信号均不存在，使用默认值0.5")
                hurst_exponent = pd.Series(0.5, index=df_index)
            else:
                hurst_exponent = hurst_series
        else:
            hurst_exponent = self.helper._get_safe_series(df, 'HURST_144d_D', 0.5, method_name="advanced_synergy")
        
        # 快速检查信号质量 - 只检查最近55天
        fractal_quality = self._check_signal_quality(fractal_dimension, 'FRACTAL_DIMENSION_89d_D', df_index, recent_days=55)
        hurst_quality = self._check_signal_quality(hurst_exponent, 'HURST_144d_D', df_index, recent_days=55)
        
        if fractal_quality["quality_level"] == "POOR":
            print(f"【协同效应】分形维数信号质量差，使用默认值1.5")
            fractal_dimension = pd.Series(1.5, index=df_index)
        
        if hurst_quality["quality_level"] == "POOR":
            print(f"【协同效应】赫斯特指数信号质量差，使用默认值0.5")
            hurst_exponent = pd.Series(0.5, index=df_index)
        
        # 检查bid_liquidity_fractal信号是否存在
        if 'BID_LIQUIDITY_FRACTAL_DIMENSION_89d_D' not in df.columns:
            print(f"【协同效应】BID_LIQUIDITY_FRACTAL_DIMENSION_89d_D 信号不存在，使用流动性真实性分数作为替代")
            # 使用流动性真实性分数作为替代
            if 'liquidity_authenticity' in fetched_signals:
                liquidity_authenticity = fetched_signals['liquidity_authenticity']
                liquidity_fractal_score = self.helper._normalize_series(liquidity_authenticity, df_index, bipolar=False)
            else:
                liquidity_fractal_score = pd.Series(0.5, index=df_index)
                print(f"【协同效应】无替代信号可用，使用中性值0.5")
        else:
            bid_liquidity_fractal = self.helper._get_safe_series(df, 'BID_LIQUIDITY_FRACTAL_DIMENSION_89d_D', np.nan, method_name="advanced_synergy")
            # 快速检查最近55天质量
            if bid_liquidity_fractal is not None:
                liquidity_quality = self._check_signal_quality(bid_liquidity_fractal, 'BID_LIQUIDITY_FRACTAL', df_index, recent_days=55)
                if liquidity_quality["quality_level"] == "POOR":
                    print(f"【协同效应】买方流动性分形信号质量差，使用中性值0.5")
                    liquidity_fractal_score = pd.Series(0.5, index=df_index)
                else:
                    liquidity_fractal_score = (bid_liquidity_fractal - 1.0).abs() / 2.0
                    liquidity_fractal_score = (1.0 - liquidity_fractal_score).clip(0, 1)
                    liquidity_fractal_score = liquidity_fractal_score.fillna(0.5)
            else:
                print(f"【协同效应】买方流动性分形信号获取失败，使用中性值0.5")
                liquidity_fractal_score = pd.Series(0.5, index=df_index)
        
        # 计算分形效率：分形维数越接近1.5，市场效率越高
        fractal_efficiency = 1.0 - (fractal_dimension - 1.5).abs() / 0.5
        fractal_efficiency = fractal_efficiency.clip(0, 1).fillna(0.5)
        
        # 计算市场记忆效应：赫斯特指数越接近1，持久性越强
        market_memory = (hurst_exponent - 0.5).abs() * 2
        market_memory = market_memory.clip(0, 1).fillna(0.5)
        
        # 计算多尺度协同效应
        fractal_synergy = fractal_efficiency * market_memory * liquidity_fractal_score
        fractal_synergy = fractal_synergy.fillna(0.25)  # 使用中性偏保守的值
        
        # 探针输出 - 只输出最近55天的统计信息
        if len(fractal_synergy) > 55:
            recent_synergy = fractal_synergy.tail(55)
        else:
            recent_synergy = fractal_synergy
            
        _temp_debug_values["高级协同效应"] = {
            "fractal_synergy_recent_mean": recent_synergy.mean(),
            "fractal_synergy_recent_std": recent_synergy.std(),
            "fractal_efficiency_mean": fractal_efficiency.mean(),
            "market_memory_mean": market_memory.mean(),
            "liquidity_fractal_score_mean": liquidity_fractal_score.mean(),
        }
        
        print(f"【高级协同效应】分形效率均值: {fractal_efficiency.mean():.4f}, 市场记忆均值: {market_memory.mean():.4f}")
        print(f"【高级协同效应】流动性分数均值: {liquidity_fractal_score.mean():.4f}, 协同分数均值: {fractal_synergy.mean():.4f}")
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




