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
        【V4.4 · 全面NaN处理与鲁棒性增强版】计算成本优势趋势。
        - 核心修复: 彻底解决NaN传播问题，确保最终结果有效
        - 核心优化: 增强错误处理和恢复机制
        - 核心新增: 详细的NaN检查和清理逻辑
        - 版本: 4.4
        """
        method_name = "CalculateCostAdvantageTrendRelationship"
        print(f"【开始计算】{method_name}，数据形状: {df.shape}，索引范围: {df.index[0]} 到 {df.index[-1]}")
        is_debug_enabled_for_method, probe_ts, debug_output, _temp_debug_values = self._initialize_debug_context(method_name, df)
        # 1. 获取MTF配置
        _, _, _, mtf_slope_accel_weights = self._get_mtf_configs(config)
        print(f"【MTF配置】斜率周期权重: {mtf_slope_accel_weights.get('slope_periods', {})}，加速度周期权重: {mtf_slope_accel_weights.get('accel_periods', {})}")
        # 2. 获取所需信号列表并进行验证
        required_signals = self._get_required_signals_list(mtf_slope_accel_weights)
        print(f"【信号验证】需要验证 {len(required_signals)} 个信号")
        if not self.helper._validate_required_signals(df, required_signals, method_name):
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。"] = ""
                self.helper._print_debug_output(debug_output)
            print(f"【信号验证失败】缺少核心信号，返回默认值")
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        print(f"【信号验证通过】所有必需信号都存在")
        df_index = df.index
        # 3. 获取原始数据和MTF融合信号（增强版）
        fetched_signals = self._fetch_raw_and_mtf_signals(df, df_index, mtf_slope_accel_weights, method_name, _temp_debug_values)
        # 4. 计算高级协同效应，添加异常处理
        try:
            advanced_synergy_score = self._calculate_advanced_synergy(fetched_signals, df, df_index, _temp_debug_values)
            # 检查协同效应分数是否有NaN
            if advanced_synergy_score is None or advanced_synergy_score.isna().all():
                print(f"【协同效应警告】高级协同效应计算失败，使用中性值0.5")
                advanced_synergy_score = pd.Series(0.5, index=df_index)
            else:
                advanced_synergy_score = advanced_synergy_score.fillna(0.5)
        except Exception as e:
            print(f"【协同效应错误】计算高级协同效应时发生异常: {e}，使用中性值0.5")
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
        # 检查各象限分数是否有NaN
        for q_name, q_series in [("Q1", Q1_final), ("Q2", Q2_final), ("Q3", Q3_final), ("Q4", Q4_final)]:
            if q_series.isna().any():
                print(f"【{q_name}警告】存在NaN值，使用0填充")
                q_series = q_series.fillna(0)
        print(f"【象限分数】Q1均值: {Q1_final.mean():.4f}, Q2均值: {Q2_final.mean():.4f}, Q3均值: {Q3_final.mean():.4f}, Q4均值: {Q4_final.mean():.4f}")
        # 8. 计算交互项
        interaction_score = self._calculate_interaction_terms(fetched_signals, normalized_signals, config, df_index, _temp_debug_values)
        if interaction_score is None or interaction_score.isna().all():
            print(f"【交互项警告】交互项计算失败，使用0替代")
            interaction_score = pd.Series(0.0, index=df_index)
        # 9. 计算高级协同效应增强项
        synergy_enhancement = advanced_synergy_score * 0.2
        # --- 最终融合 ---
        # 基础融合分数
        base_fusion_score = (Q1_final + Q2_final + Q3_final + Q4_final)
        # 检查基础融合分数是否有NaN
        if base_fusion_score.isna().any():
            print(f"【基础融合警告】基础融合分数存在NaN，使用0填充")
            base_fusion_score = base_fusion_score.fillna(0)
        # 叠加交互项和协同效应
        final_score_with_interaction = base_fusion_score + interaction_score + synergy_enhancement
        # 检查最终分数是否有NaN
        if final_score_with_interaction.isna().any():
            print(f"【最终分数警告】最终分数存在NaN，使用基础融合分数替代")
            final_score_with_interaction = base_fusion_score.fillna(0)
        # 10. 计算动态指数
        dynamic_exponent = self._calculate_dynamic_exponent(fetched_signals, config, df, df_index, _temp_debug_values)
        # 应用动态指数进行非线性放大/平滑
        final_score_normalized_for_exponent = (final_score_with_interaction + 1) / 2
        # 检查是否有NaN
        if final_score_normalized_for_exponent.isna().any():
            print(f"【指数归一化警告】归一化分数存在NaN，使用0.5填充")
            final_score_normalized_for_exponent = final_score_normalized_for_exponent.fillna(0.5)
        final_score_exponentiated = final_score_normalized_for_exponent.pow(dynamic_exponent)
        final_score = (final_score_exponentiated * 2 - 1).clip(-1, 1)
        # 最终检查：确保没有NaN
        if final_score.isna().any():
            print(f"【最终结果警告】最终分数存在NaN，使用0填充")
            final_score = final_score.fillna(0)
        # 探针：记录所有关键计算节点
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
        # 输出详细探针信息
        if is_debug_enabled_for_method and probe_ts:
            self._log_debug_values(debug_output, _temp_debug_values, probe_ts, method_name)
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 成本优势趋势关系诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
            self.helper._print_debug_output(debug_output)
        # 输出最终统计信息
        print(f"【最终结果】最终分数范围: [{final_score.min():.4f}, {final_score.max():.4f}]，均值: {final_score.mean():.4f}，标准差: {final_score.std():.4f}")
        print(f"【最终结果】正分数比例: {(final_score > 0).sum() / len(final_score):.2%}，负分数比例: {(final_score < 0).sum() / len(final_score):.2%}")
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
        【V1.1.3 · 原始及MTF信号获取 - 信号质量增强版】
        - 核心修复: 增强信号质量检查和异常处理
        - 核心优化: 对缺失严重的信号进行替代或跳过处理
        - 核心新增: 信号质量评估和报告
        - 版本: 1.1.3
        """
        fetched_signals = {}
        signal_quality_reports = {}
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
        # 获取所有信号并检查质量
        for df_col_name, signal_name, default_value in signal_configs:
            signal_series = self.helper._get_safe_series(df, df_col_name, default_value, method_name=method_name)
            fetched_signals[signal_name] = signal_series
            # 检查信号质量
            quality_report = self._check_signal_quality(signal_series, signal_name, df_index)
            signal_quality_reports[signal_name] = quality_report
            # 如果信号质量差，输出警告
            if quality_report["quality_level"] in ["POOR", "MEDIUM"]:
                print(f"【信号质量警告】{signal_name}: {quality_report['quality_level']}, NaN比例: {quality_report['nan_ratio']:.2%}, 建议: {quality_report['suggestion']}")
        # 对于缺失严重的信号，尝试使用替代信号
        # 例如，如果main_force_conviction缺失严重，可以使用main_force_net_flow作为替代
        if signal_quality_reports.get('main_force_conviction', {}).get('nan_ratio', 0) > 0.5:
            print(f"【信号替代】main_force_conviction缺失严重，尝试使用main_force_net_flow的归一化值作为替代")
            net_flow_norm = self.helper._normalize_series(fetched_signals['main_force_net_flow'], df_index, bipolar=True)
            fetched_signals['main_force_conviction'] = net_flow_norm * 10  # 缩放因子
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
        【V1.0.1 · 高级协同效应计算 - 修复NaN问题与稳健性增强版】
        - 核心修复: 检查bid_liquidity_fractal信号是否存在，处理NaN问题
        - 核心优化: 当信号不存在时使用替代方案，确保计算稳健性
        - 核心新增: 添加详细的探针输出和异常处理
        - 版本: 1.0.1
        """
        # 获取分形市场指标
        fractal_dimension = self.helper._get_safe_series(df, 'FRACTAL_DIMENSION_89d_D', 0.0, method_name="advanced_synergy")
        hurst_exponent = self.helper._get_safe_series(df, 'HURST_144d_d', 0.5, method_name="advanced_synergy")
        # 检查bid_liquidity_fractal信号是否存在
        bid_liquidity_fractal = self.helper._get_safe_series(df, 'BID_LIQUIDITY_FRACTAL_DIMENSION_89d_D', np.nan, method_name="advanced_synergy")
        # 探针：输出信号基本信息
        print(f"【高级协同效应】分形维数形状: {fractal_dimension.shape}, NaN数量: {fractal_dimension.isna().sum()}")
        print(f"【高级协同效应】赫斯特指数形状: {hurst_exponent.shape}, NaN数量: {hurst_exponent.isna().sum()}")
        print(f"【高级协同效应】买方流动性分形形状: {bid_liquidity_fractal.shape if bid_liquidity_fractal is not None else 'None'}, NaN数量: {bid_liquidity_fractal.isna().sum() if bid_liquidity_fractal is not None else 'N/A'}")
        # 计算分形效率：分形维数越接近1.5，市场效率越高
        fractal_efficiency = 1.0 - (fractal_dimension - 1.5).abs() / 0.5
        fractal_efficiency = fractal_efficiency.clip(0, 1)
        # 计算市场记忆效应：赫斯特指数越接近1，持久性越强
        market_memory = (hurst_exponent - 0.5).abs() * 2
        market_memory = market_memory.clip(0, 1)
        # 计算流动性分形特征 - 处理NaN情况
        if bid_liquidity_fractal is not None and not bid_liquidity_fractal.isna().all():
            liquidity_fractal_score = (bid_liquidity_fractal - 1.0).abs() / 2.0
            liquidity_fractal_score = (1.0 - liquidity_fractal_score).clip(0, 1)
            # 填充NaN值为0.5（中性）
            liquidity_fractal_score = liquidity_fractal_score.fillna(0.5)
        else:
            print(f"【高级协同效应警告】买方流动性分形信号不存在或全部为NaN，使用中性值0.5替代")
            liquidity_fractal_score = pd.Series(0.5, index=df_index)
        # 计算多尺度协同效应
        fractal_synergy = fractal_efficiency * market_memory * liquidity_fractal_score
        # 确保没有NaN值
        fractal_synergy = fractal_synergy.fillna(0.5)
        # 探针输出
        _temp_debug_values["高级协同效应"] = {
            "fractal_dimension": fractal_dimension,
            "hurst_exponent": hurst_exponent,
            "bid_liquidity_fractal": bid_liquidity_fractal,
            "fractal_efficiency": fractal_efficiency,
            "market_memory": market_memory,
            "liquidity_fractal_score": liquidity_fractal_score,
            "fractal_synergy": fractal_synergy
        }
        print(f"【高级协同效应】分形维数均值: {fractal_dimension.mean():.4f}, 赫斯特指数均值: {hurst_exponent.mean():.4f}")
        print(f"【高级协同效应】分形效率均值: {fractal_efficiency.mean():.4f}, 市场记忆均值: {market_memory.mean():.4f}")
        print(f"【高级协同效应】流动性分形分数均值: {liquidity_fractal_score.mean():.4f}, 协同分数均值: {fractal_synergy.mean():.4f}")
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

    def _check_signal_quality(self, signal_series: pd.Series, signal_name: str, df_index: pd.Index) -> Dict[str, Any]:
        """
        【V1.0.0 · 信号质量检查】
        - 核心职责: 检查信号的数据质量，包括NaN比例、异常值等
        - 核心新增: 提供信号质量报告和修复建议
        - 版本: 1.0.0
        """
        quality_report = {
            "signal_name": signal_name,
            "total_count": len(signal_series),
            "nan_count": signal_series.isna().sum(),
            "nan_ratio": signal_series.isna().sum() / len(signal_series),
            "zero_count": (signal_series == 0).sum() if not signal_series.isna().all() else 0,
            "zero_ratio": (signal_series == 0).sum() / len(signal_series) if not signal_series.isna().all() else 0,
            "mean": signal_series.mean() if not signal_series.isna().all() else np.nan,
            "std": signal_series.std() if not signal_series.isna().all() else np.nan,
            "min": signal_series.min() if not signal_series.isna().all() else np.nan,
            "max": signal_series.max() if not signal_series.isna().all() else np.nan,
        }
        # 质量评级
        if quality_report["nan_ratio"] > 0.5:
            quality_report["quality_level"] = "POOR"
            quality_report["suggestion"] = "信号缺失严重，考虑使用替代信号或跳过"
        elif quality_report["nan_ratio"] > 0.2:
            quality_report["quality_level"] = "MEDIUM"
            quality_report["suggestion"] = "信号部分缺失，建议进行插值处理"
        else:
            quality_report["quality_level"] = "GOOD"
            quality_report["suggestion"] = "信号质量良好"
        return quality_report





