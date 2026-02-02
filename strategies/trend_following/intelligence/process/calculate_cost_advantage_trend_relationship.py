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
        """【V8.0 · 幻方量化A股趋势判断核心算法 - 筹码成本优势趋势关系】
        - 核心重构: 基于幻方量化A股经验的完全重构
        - 核心维度: 筹码成本结构 + 主力行为博弈 + 多时间框架共振
        - 核心指标: 筹码集中度变化、主力成本博弈、趋势结构完整性
        - 核心逻辑: 识别趋势健康度与反转博弈信号
        - 版本: 8.0"""
        method_name = "CalculateCostAdvantageTrendRelationship_V8"
        print(f"【V8.0幻方量化】开始计算A股趋势判断核心信号")
        print(f"【V8.0数据状态】数据形状: {df.shape}, 时间范围: {df.index[0]} 到 {df.index[-1]}")
        # 1. 数据质量检查 - 直接使用已有的_check_and_repair_signals方法
        print(f"【V8.0计算】步骤0: 数据质量检查与修复")
        df_processed = self._check_and_repair_signals(df.copy(), method_name)
        df_index = df_processed.index
        # 2. 计算三大核心维度分数
        print(f"【V8.0计算】步骤1/3: 计算筹码成本结构维度")
        chip_structure_score = self._calculate_chip_structure_dimension(df_processed, df_index)
        print(f"【V8.0计算】步骤2/3: 计算主力行为博弈维度")
        main_force_score = self._calculate_main_force_dimension(df_processed, df_index)
        print(f"【V8.0计算】步骤3/3: 计算趋势结构完整性维度")
        trend_structure_score = self._calculate_trend_structure_dimension(df_processed, df_index)
        # 3. 多维度共振分析
        print(f"【V8.0计算】多维共振分析")
        resonance_score = self._calculate_multi_dimension_resonance(
            chip_structure_score, main_force_score, trend_structure_score, df_index
        )
        # 4. 市场状态调制
        print(f"【V8.0计算】市场状态调制")
        market_state_modulator = self._calculate_market_state_modulator(df_processed, df_index)
        modulated_score = resonance_score * market_state_modulator
        # 5. 反转博弈增强
        print(f"【V8.0计算】反转博弈增强")
        reversal_enhancement = self._calculate_reversal_game_enhancement(df_processed, df_index)
        final_score = modulated_score + reversal_enhancement * 0.3
        # 6. 最终处理
        final_score = final_score.clip(-1, 1).astype(np.float32)
        final_score = final_score.fillna(0)
        # 探针输出
        print(f"【V8.0幻方量化】计算完成")
        print(f"  筹码结构维度: [{chip_structure_score.min():.3f}, {chip_structure_score.max():.3f}]")
        print(f"  主力行为维度: [{main_force_score.min():.3f}, {main_force_score.max():.3f}]")
        print(f"  趋势结构维度: [{trend_structure_score.min():.3f}, {trend_structure_score.max():.3f}]")
        print(f"  多维共振分数: {resonance_score.mean():.3f}")
        print(f"  市场状态调制: {market_state_modulator.mean():.3f}")
        print(f"  最终信号范围: [{final_score.min():.3f}, {final_score.max():.3f}]")
        return final_score

    def _calculate_chip_structure_dimension(self, df: pd.DataFrame, df_index: pd.Index) -> pd.Series:
        """【V8.0 · 筹码成本结构维度 - 幻方量化A股专用】
        - 核心逻辑: 分析不同成本区间的筹码分布与变化
        - 关键指标: 筹码集中度、成本区间迁移、筹码稳定性
        - 输出: [-1, 1]区间，正值表示筹码结构健康，负值表示结构恶化
        - 版本: 8.0"""
        print(f"【筹码维度】开始计算筹码成本结构")
        # 1. 筹码集中度分析
        concentration_components = []
        concentration_signals = ['chip_concentration_ratio_D', 'peak_concentration_D', 'concentration_comprehensive_D']
        for signal in concentration_signals:
            if signal in df.columns:
                series = self.helper._get_safe_series(df, signal, 0.5, "chip_concentration")
                concentration_components.append(self.helper._normalize_series(series, df_index, bipolar=False))
        if concentration_components:
            concentration_score = pd.Series(0.0, index=df_index)
            for comp in concentration_components:
                concentration_score += comp
            concentration_score = concentration_score / len(concentration_components)
        else:
            concentration_score = pd.Series(0.5, index=df_index)
        print(f"【筹码维度】集中度分数均值: {concentration_score.mean():.3f}")
        # 2. 成本区间迁移分析
        cost_migration_score = pd.Series(0.0, index=df_index)
        cost_signals = ['cost_5pct_D', 'cost_15pct_D', 'cost_50pct_D', 'cost_85pct_D', 'cost_95pct_D']
        valid_cost_signals = [s for s in cost_signals if s in df.columns]
        if len(valid_cost_signals) >= 3:
            # 计算成本区间宽度
            cost_ranges = []
            for i in range(len(valid_cost_signals)-1):
                high_signal = valid_cost_signals[i+1]
                low_signal = valid_cost_signals[i]
                if high_signal in df.columns and low_signal in df.columns:
                    high = self.helper._get_safe_series(df, high_signal, 0, "cost_migration")
                    low = self.helper._get_safe_series(df, low_signal, 0, "cost_migration")
                    cost_range = high - low
                    cost_ranges.append(self.helper._normalize_series(cost_range, df_index, bipolar=False))
            if cost_ranges:
                # 成本区间收窄表示筹码集中，拓宽表示分散
                range_stability = pd.Series(0.0, index=df_index)
                for rng in cost_ranges:
                    range_stability += (1 - rng)  # 收窄为正值
                cost_migration_score = range_stability / len(cost_ranges)
        # 3. 筹码稳定性分析
        stability_score = pd.Series(0.5, index=df_index)
        stability_signals = ['chip_stability_D', 'chip_stability_change_5d_D', 'MA_POTENTIAL_ORDERLINESS_SCORE_D']
        stability_components = []
        for signal in stability_signals:
            if signal in df.columns:
                series = self.helper._get_safe_series(df, signal, 0.5, "chip_stability")
                stability_components.append(self.helper._normalize_series(series, df_index, bipolar=False))
        if stability_components:
            stability_score = pd.Series(0.0, index=df_index)
            for comp in stability_components:
                stability_score += comp
            stability_score = stability_score / len(stability_components)
        # 4. 综合筹码结构分数
        chip_structure_score = (
            concentration_score * 0.4 +
            cost_migration_score * 0.3 +
            stability_score * 0.3
        )
        # 转换为[-1, 1]区间
        chip_structure_score = (chip_structure_score - 0.5) * 2
        print(f"【筹码维度】综合分数均值: {chip_structure_score.mean():.3f}")
        return chip_structure_score.clip(-1, 1)

    def _calculate_main_force_dimension(self, df: pd.DataFrame, df_index: pd.Index) -> pd.Series:
        """【V8.0 · 主力行为博弈维度 - 幻方量化A股专用】
        - 核心逻辑: 分析主力资金在不同成本区间的博弈行为
        - 关键指标: 主力成本优势、主力资金流向、博弈强度
        - 输出: [-1, 1]区间，正值表示主力积极，负值表示主力撤离
        - 版本: 8.0"""
        print(f"【主力维度】开始计算主力行为博弈")
        # 1. 主力成本优势分析
        cost_advantage_score = pd.Series(0.0, index=df_index)
        if 'main_force_cost_advantage_D' in df.columns:
            mf_cost_advantage = self.helper._get_safe_series(df, 'main_force_cost_advantage_D', 0, "mf_cost")
            # 归一化到[-1, 1]
            cost_advantage_score = mf_cost_advantage / 100  # 假设成本优势以百分点表示
            cost_advantage_score = cost_advantage_score.clip(-1, 1)
        print(f"【主力维度】成本优势分数均值: {cost_advantage_score.mean():.3f}")
        # 2. 主力资金流向分析
        flow_score = pd.Series(0.0, index=df_index)
        flow_signals = [
            'main_force_net_flow_calibrated_D',
            'SMART_MONEY_HM_NET_BUY_D',
            'SMART_MONEY_INST_NET_BUY_D',
            'order_flow_imbalance_score_D'
        ]
        flow_components = []
        for signal in flow_signals:
            if signal in df.columns:
                series = self.helper._get_safe_series(df, signal, 0, "mf_flow")
                # 简单归一化
                normalized = series / (series.abs().max() + 1e-8)
                flow_components.append(normalized)
        if flow_components:
            flow_score = pd.Series(0.0, index=df_index)
            for comp in flow_components:
                flow_score += comp
            flow_score = flow_score / len(flow_components)
        # 3. 主力博弈强度分析
        game_intensity_score = pd.Series(0.0, index=df_index)
        intensity_signals = [
            'game_intensity_D',
            'auction_showdown_score_D',
            'counterparty_exhaustion_index_D',
            'large_order_anomaly_D'
        ]
        intensity_components = []
        for signal in intensity_signals:
            if signal in df.columns:
                series = self.helper._get_safe_series(df, signal, 0, "game_intensity")
                intensity_components.append(self.helper._normalize_series(series, df_index, bipolar=False))
        if intensity_components:
            game_intensity_score = pd.Series(0.0, index=df_index)
            for comp in intensity_components:
                game_intensity_score += comp
            game_intensity_score = game_intensity_score / len(intensity_components)
            # 博弈强度高时分数为正，但过高可能表示分歧大
            game_intensity_score = game_intensity_score * 2 - 1
        # 4. 主力行为一致性分析
        consistency_score = pd.Series(0.0, index=df_index)
        if 'main_force_conviction_index_D' in df.columns:
            conviction = self.helper._get_safe_series(df, 'main_force_conviction_index_D', 0.5, "conviction")
            consistency_score = (conviction - 0.5) * 2
        # 5. 综合主力行为分数
        main_force_score = (
            cost_advantage_score * 0.3 +
            flow_score * 0.3 +
            game_intensity_score * 0.2 +
            consistency_score * 0.2
        )
        print(f"【主力维度】综合分数均值: {main_force_score.mean():.3f}")
        return main_force_score.clip(-1, 1)

    def _calculate_trend_structure_dimension(self, df: pd.DataFrame, df_index: pd.Index) -> pd.Series:
        """【V8.0 · 趋势结构完整性维度 - 幻方量化A股专用】
        - 核心逻辑: 分析价格趋势的完整性和可持续性
        - 关键指标: 趋势强度、结构完整性、突破质量
        - 输出: [-1, 1]区间，正值表示趋势健康，负值表示趋势脆弱
        - 版本: 8.0"""
        print(f"【趋势维度】开始计算趋势结构完整性")
        # 1. 趋势强度分析
        trend_strength_score = pd.Series(0.0, index=df_index)
        strength_signals = [
            'ADX_14_D',
            'trend_conviction_score_D',
            'uptrend_strength_D',
            'downtrend_strength_D',
            'trendline_validity_score_D'
        ]
        strength_components = []
        for signal in strength_signals:
            if signal in df.columns:
                series = self.helper._get_safe_series(df, signal, 0.5, "trend_strength")
                strength_components.append(self.helper._normalize_series(series, df_index, bipolar=False))
        if strength_components:
            trend_strength_score = pd.Series(0.0, index=df_index)
            for comp in strength_components:
                trend_strength_score += comp
            trend_strength_score = trend_strength_score / len(strength_components)
        # 2. 趋势方向分析（基于价格与均线关系）
        trend_direction_score = pd.Series(0.0, index=df_index)
        price_ma_ratios = []
        ma_periods = [5, 13, 21, 34, 55]
        for period in ma_periods:
            price_signal = f'price_vs_ma_{period}_ratio_D'
            if price_signal in df.columns:
                ratio = self.helper._get_safe_series(df, price_signal, 1.0, f"price_ma_{period}")
                # 价格在均线上方为正值
                direction = (ratio - 1.0) * 2  # 将[0.5, 1.5]映射到[-1, 1]
                price_ma_ratios.append(direction.clip(-1, 1))
        if price_ma_ratios:
            trend_direction_score = pd.Series(0.0, index=df_index)
            # 短期权重高，长期权重低
            weights = [0.35, 0.25, 0.2, 0.15, 0.05][:len(price_ma_ratios)]
            for i, ratio in enumerate(price_ma_ratios[:len(weights)]):
                trend_direction_score += ratio * weights[i]
        # 3. 结构完整性分析
        structure_score = pd.Series(0.5, index=df_index)
        structure_signals = [
            'structural_potential_score_D',
            'breakout_quality_score_D',
            'platform_conviction_score_D',
            'consolidation_strength_D'
        ]
        structure_components = []
        for signal in structure_signals:
            if signal in df.columns:
                series = self.helper._get_safe_series(df, signal, 0.5, "structure")
                structure_components.append(self.helper._normalize_series(series, df_index, bipolar=False))
        if structure_components:
            structure_score = pd.Series(0.0, index=df_index)
            for comp in structure_components:
                structure_score += comp
            structure_score = structure_score / len(structure_components)
        # 4. 动量与加速度分析
        momentum_score = pd.Series(0.0, index=df_index)
        momentum_signals = ['ROC_13_D', 'MACDh_13_34_8_D', 'MA_VELOCITY_EMA_55_D']
        momentum_components = []
        for signal in momentum_signals:
            if signal in df.columns:
                series = self.helper._get_safe_series(df, signal, 0, "momentum")
                # 简单归一化
                if series.abs().max() > 0:
                    normalized = series / (series.abs().max() + 1e-8)
                    momentum_components.append(normalized)
        if momentum_components:
            momentum_score = pd.Series(0.0, index=df_index)
            for comp in momentum_components:
                momentum_score += comp
            momentum_score = momentum_score / len(momentum_components)
        # 5. 综合趋势结构分数
        trend_structure_score = (
            trend_strength_score * 0.25 +
            trend_direction_score * 0.3 +
            structure_score * 0.25 +
            momentum_score * 0.2
        )
        # 转换为[-1, 1]区间
        trend_structure_score = trend_structure_score * 2 - 1
        print(f"【趋势维度】综合分数均值: {trend_structure_score.mean():.3f}")
        return trend_structure_score.clip(-1, 1)

    def _calculate_multi_dimension_resonance(self, chip_score: pd.Series, main_force_score: pd.Series, 
                                             trend_score: pd.Series, df_index: pd.Index) -> pd.Series:
        """【V8.0 · 多维度共振分析 - 幻方量化A股专用】
        - 核心逻辑: 分析三个维度的协同与背离关系
        - 关键原则: 三维共振 > 二维共振 > 单维强势
        - 输出: [-1, 1]区间，正值表示强共振看涨，负值表示强共振看跌
        - 版本: 8.0"""
        print(f"【共振分析】开始多维度共振计算")
        # 1. 计算各维度符号
        chip_sign = np.sign(chip_score)
        main_force_sign = np.sign(main_force_score)
        trend_sign = np.sign(trend_score)
        # 2. 计算共振强度
        resonance_strength = pd.Series(0.0, index=df_index)
        for i in range(len(df_index)):
            signs = [chip_sign.iloc[i], main_force_sign.iloc[i], trend_sign.iloc[i]]
            positive_count = sum(1 for s in signs if s > 0)
            negative_count = sum(1 for s in signs if s < 0)
            # 三维共振（最强信号）
            if positive_count == 3:
                resonance_strength.iloc[i] = 1.0
            elif negative_count == 3:
                resonance_strength.iloc[i] = -1.0
            # 二维共振
            elif positive_count == 2:
                resonance_strength.iloc[i] = 0.5
            elif negative_count == 2:
                resonance_strength.iloc[i] = -0.5
            # 单维强势或三维分歧
            else:
                # 计算各维度绝对值的加权平均
                abs_values = [
                    abs(chip_score.iloc[i]),
                    abs(main_force_score.iloc[i]),
                    abs(trend_score.iloc[i])
                ]
                # 单维强势情况
                max_abs = max(abs_values)
                if max_abs > 0.7:  # 单维度特别强势
                    dominant_index = abs_values.index(max_abs)
                    dominant_sign = signs[dominant_index]
                    resonance_strength.iloc[i] = dominant_sign * 0.3
                else:
                    resonance_strength.iloc[i] = 0.0
        # 3. 计算各维度强度均值作为共振强度调整
        abs_scores = pd.DataFrame({
            'chip': chip_score.abs(),
            'main_force': main_force_score.abs(),
            'trend': trend_score.abs()
        })
        avg_strength = abs_scores.mean(axis=1)
        # 4. 最终共振分数 = 共振方向 * 平均强度
        resonance_score = resonance_strength * avg_strength
        print(f"【共振分析】共振分数均值: {resonance_score.mean():.3f}")
        print(f"  三维共振天数: {(resonance_strength.abs() == 1.0).sum()}")
        print(f"  二维共振天数: {(resonance_strength.abs() == 0.5).sum()}")
        return resonance_score.clip(-1, 1)

    def _calculate_market_state_modulator(self, df: pd.DataFrame, df_index: pd.Index) -> pd.Series:
        """【V8.0 · 市场状态调制器 - 幻方量化A股专用】
        - 核心逻辑: 根据市场状态调整信号强度
        - 关键状态: 趋势市场加强，震荡市场减弱，极端市场谨慎
        - 输出: [0.5, 1.5]调制因子
        - 版本: 8.0"""
        print(f"【市场调制】开始计算市场状态调制器")
        # 1. 趋势状态判断
        trend_state = pd.Series(1.0, index=df_index)
        if 'MARKET_PHASE_D' in df.columns:
            market_phase = self.helper._get_safe_series(df, 'MARKET_PHASE_D', 0.5, "market_phase")
            # 假设MARKET_PHASE_D已在[0,1]范围
            # 趋势市场(>0.7)增强，震荡市场(~0.5)中性，反转市场(<0.3)减弱
            trend_state = 0.8 + market_phase * 0.4  # [0,1] -> [0.8, 1.2]
        # 2. 波动性调整
        volatility_mod = pd.Series(1.0, index=df_index)
        if 'VOLATILITY_INSTABILITY_INDEX_21d_D' in df.columns:
            volatility = self.helper._get_safe_series(df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', 0.5, "volatility")
            vol_norm = self.helper._normalize_series(volatility, df_index, bipolar=False)
            # 中等波动性最理想，过高或过低都减弱信号
            volatility_mod = 1.2 - abs(vol_norm - 0.5) * 0.4
        # 3. 成交量支持度
        volume_mod = pd.Series(1.0, index=df_index)
        if 'volume_vs_ma_5_ratio_D' in df.columns and 'volume_vs_ma_21_ratio_D' in df.columns:
            vol_ma5 = self.helper._get_safe_series(df, 'volume_vs_ma_5_ratio_D', 1.0, "vol_ma5")
            vol_ma21 = self.helper._get_safe_series(df, 'volume_vs_ma_21_ratio_D', 1.0, "vol_ma21")
            # 成交量在均线上方为良好状态
            volume_support = (vol_ma5 + vol_ma21) / 2
            volume_mod = 0.9 + volume_support * 0.2  # [0.5, 1.5] -> [1.0, 1.2]
        # 4. 市场情绪调整
        sentiment_mod = pd.Series(1.0, index=df_index)
        if 'market_sentiment_score_D' in df.columns:
            sentiment = self.helper._get_safe_series(df, 'market_sentiment_score_D', 0.0, "sentiment")
            sentiment_norm = (sentiment.clip(-1, 1) + 1) / 2  # [-1,1] -> [0,1]
            # 情绪过热或过冷都减弱信号
            sentiment_mod = 1.2 - abs(sentiment_norm - 0.5) * 0.4
        # 5. 综合调制因子
        modulator = (trend_state * 0.4 + 
                     volatility_mod * 0.25 + 
                     volume_mod * 0.2 + 
                     sentiment_mod * 0.15)
        # 限制在合理范围
        modulator = modulator.clip(0.5, 1.5)
        print(f"【市场调制】调制因子均值: {modulator.mean():.3f}")
        return modulator

    def _calculate_reversal_game_enhancement(self, df: pd.DataFrame, df_index: pd.Index) -> pd.Series:
        """【V8.0 · 反转博弈增强 - 幻方量化A股专用】
        - 核心逻辑: 识别趋势末端的主力博弈行为，增强反转信号
        - 关键场景: 上涨末端派发、下跌末端吸筹
        - 输出: [-0.5, 0.5]增强项
        - 版本: 8.0"""
        print(f"【反转增强】开始计算反转博弈增强")
        enhancement = pd.Series(0.0, index=df_index)
        # 1. 上涨末端派发检测（负增强）
        distribution_enhancement = pd.Series(0.0, index=df_index)
        if all(signal in df.columns for signal in ['close_D', 'distribution_at_peak_intensity_D', 'profit_pressure_D']):
            close = self.helper._get_safe_series(df, 'close_D', 0, "close")
            distribution = self.helper._get_safe_series(df, 'distribution_at_peak_intensity_D', 0, "distribution")
            profit_pressure = self.helper._get_safe_series(df, 'profit_pressure_D', 0, "profit_pressure")
            # 价格上涨但派发增强
            price_high = close > close.rolling(20).mean() * 1.1
            distribution_high = distribution > 0.7
            pressure_high = profit_pressure > 0.7
            bearish_reversal = price_high & (distribution_high | pressure_high)
            distribution_enhancement = -0.3 * bearish_reversal.astype(float)
        # 2. 下跌末端吸筹检测（正增强）
        accumulation_enhancement = pd.Series(0.0, index=df_index)
        if all(signal in df.columns for signal in ['close_D', 'suppressive_accumulation_intensity_D', 'lower_shadow_absorption_strength_D']):
            close = self.helper._get_safe_series(df, 'close_D', 0, "close")
            accumulation = self.helper._get_safe_series(df, 'suppressive_accumulation_intensity_D', 0, "accumulation")
            lower_shadow = self.helper._get_safe_series(df, 'lower_shadow_absorption_strength_D', 0, "lower_shadow")
            # 价格低位但吸筹增强
            price_low = close < close.rolling(20).mean() * 0.9
            accumulation_high = accumulation > 0.7
            shadow_strong = lower_shadow > 0.7
            bullish_reversal = price_low & (accumulation_high | shadow_strong)
            accumulation_enhancement = 0.3 * bullish_reversal.astype(float)
        # 3. 背离增强
        divergence_enhancement = pd.Series(0.0, index=df_index)
        if all(signal in df.columns for signal in ['close_D', 'main_force_cost_advantage_D', 'chip_rsi_divergence_D']):
            close = self.helper._get_safe_series(df, 'close_D', 0, "close")
            cost_advantage = self.helper._get_safe_series(df, 'main_force_cost_advantage_D', 0, "cost_advantage")
            divergence = self.helper._get_safe_series(df, 'chip_rsi_divergence_D', 0, "divergence")
            # 价格新高但成本优势下降（看跌背离）
            price_new_high = close == close.rolling(20).max()
            cost_declining = cost_advantage < cost_advantage.rolling(5).mean()
            bearish_divergence = price_new_high & cost_declining
            # 价格新低但成本优势上升（看涨背离）
            price_new_low = close == close.rolling(20).min()
            cost_rising = cost_advantage > cost_advantage.rolling(5).mean()
            bullish_divergence = price_new_low & cost_rising
            divergence_enhancement = (
                -0.2 * bearish_divergence.astype(float) +
                0.2 * bullish_divergence.astype(float)
            )
        # 4. 综合增强
        enhancement = (
            distribution_enhancement +
            accumulation_enhancement +
            divergence_enhancement
        )
        print(f"【反转增强】增强项均值: {enhancement.mean():.3f}")
        return enhancement.clip(-0.5, 0.5)





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
        """【V6.0 · 必需信号列表全面增强版】
        - 核心新增: 添加市场结构、流动性、微观结构、情绪等四大维度信号
        - 核心优化: 增加信号冗余度，为每个核心逻辑提供2-3个备选信号
        - 核心修复: 确保所有信号在数据层中存在
        - 版本: 6.0"""
        required_signals = [
            # 价格与趋势维度
            'close_D', 'pct_change_D', 'trendline_slope_D', 'trendline_validity_score_D',
            'MTF_TRENDLINE_SLOPE_D', 'MTF_TRENDLINE_VALIDITY_SCORE_D', 'trend_conviction_score_D',
            'ADX_14_D', 'DMA_D', 'MACD_13_34_8_D', 'MACDh_13_34_8_D',
            # 成本优势与主力资金维度
            'main_force_cost_advantage_D', 'main_force_conviction_index_D', 
            'main_force_net_flow_calibrated_D', 'main_force_buy_amount_calibrated_D',
            'main_force_sell_amount_calibrated_D', 'main_force_buy_ofi_D', 'main_force_sell_ofi_D',
            'main_force_level5_ofi_D', 'mf_cost_zone_buy_intent_D', 'mf_cost_zone_sell_intent_D',
            # 订单流与微观结构
            'order_book_imbalance_D', 'bid_side_liquidity_D', 'ask_side_liquidity_D',
            'buy_order_book_clearing_rate_D', 'sell_order_book_clearing_rate_D',
            'microstructure_efficiency_index_D', 'micro_price_impact_asymmetry_D',
            'main_force_buy_execution_alpha_D', 'main_force_sell_execution_alpha_D',
            'order_book_liquidity_supply_D', 'liquidity_slope_D',
            # 行为金融与市场情绪
            'market_sentiment_score_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D',
            'panic_selling_cascade_D', 'retail_panic_surrender_index_D',
            'retail_fomo_premium_index_D', 'retail_flow_dominance_index_D',
            # 市场结构与形态识别
            'MARKET_PHASE_D', 'structural_potential_score_D', 'trend_conviction_score_D',
            'breakout_quality_score_D', 'breakout_readiness_score_D',
            'IS_ACCUMULATION_D', 'IS_DISTRIBUTION_D', 'IS_TREND_REVERSAL_D',
            # 成交量与资金流
            'volume_D', 'amount_D', 'turnover_rate_D', 'flow_credibility_index_D',
            'buy_flow_efficiency_index_D', 'sell_flow_efficiency_index_D',
            'constructive_turnover_ratio_D', 'volume_profile_entropy_D',
            # 支撑阻力与关键价位
            'dominant_peak_cost_D', 'cost_50pct_D', 'cost_85pct_D', 'cost_15pct_D',
            'VWAP_D', 'VPOC_D', 'main_force_vpoc_D',
            # 备选信号（当主信号缺失时使用）
            'price_vs_ma_5_ratio_D', 'price_vs_ma_13_ratio_D', 'price_vs_ma_21_ratio_D',
            'volume_vs_ma_5_ratio_D', 'volume_vs_ma_13_ratio_D', 'volume_vs_ma_21_ratio_D',
            'RSI_13_D', 'BBP_21_2.0_D', 'ATR_14_D',
        ]
        # 动态添加MTF斜率和加速度信号
        for base_sig in ['close_D', 'main_force_cost_advantage_D', 'main_force_conviction_index_D',
                         'main_force_net_flow_calibrated_D', 'order_book_imbalance_D']:
            for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
                slope_signal = f'SLOPE_{period_str}_{base_sig}'
                if slope_signal.replace('close_D', 'pct_change_D') in required_signals:
                    continue
                required_signals.append(slope_signal)
            for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
                accel_signal = f'ACCEL_{period_str}_{base_sig}'
                if accel_signal.replace('close_D', 'pct_change_D') in required_signals:
                    continue
                required_signals.append(accel_signal)
        print(f"【V6.0信号列表】共{len(required_signals)}个信号，覆盖8个维度")
        return required_signals

    def _check_and_repair_signals(self, df: pd.DataFrame, method_name: str) -> pd.DataFrame:
        """【V7.0 · 信号NaN检查与修复简化版 - 基于军械库清单体系】
        - 核心简化: 只专注于NaN值的检测与修复，不再进行复杂的信号替代
        - 核心优化: 使用高效插值方法，针对不同信号类型采用不同策略
        - 核心新增: 详细的NaN修复统计和探针输出
        - 核心修复: 确保所有信号都有有效数据，避免计算中断
        - 版本: 7.0"""
        print(f"【V7.0信号修复】开始检查{len(df.columns)}个信号的NaN情况")
        print(f"【V7.0信号修复】数据形状: {df.shape}, 时间范围: {df.index[0]} 到 {df.index[-1]}")
        # 关键信号列表 - 这些信号对成本优势趋势计算至关重要
        critical_signals_v7 = [
            'close_D', 'main_force_cost_advantage_D', 'main_force_conviction_index_D',
            'main_force_net_flow_calibrated_D', 'flow_credibility_index_D',
            'volume_D', 'amount_D', 'turnover_rate_D', 'ADX_14_D', 'RSI_13_D',
            'VWAP_D', 'VPOC_D', 'cost_50pct_D', 'chip_concentration_ratio_D',
            'order_flow_imbalance_score_D', 'market_sentiment_score_D'
        ]
        # 修复统计信息
        repair_stats = {
            'total_signals': len(df.columns),
            'signals_checked': 0,
            'signals_repaired': 0,
            'critical_repaired': 0,
            'total_nan_filled': 0,
            'repair_methods_used': {},
            'critical_signals_status': {}
        }
        # 修复方法计数器
        repair_methods_count = {
            'linear_interpolation': 0,
            'forward_fill': 0,
            'backward_fill': 0,
            'rolling_mean': 0,
            'constant_fill': 0
        }
        # 第一步：检查并修复关键信号
        print(f"【V7.0信号修复】第一步：检查{len(critical_signals_v7)}个关键信号")
        for signal in critical_signals_v7:
            if signal not in df.columns:
                print(f"【V7.0信号修复警告】关键信号{signal}不在数据中，创建默认序列")
                df[signal] = pd.Series(0.0, index=df.index)
                repair_stats['critical_signals_status'][signal] = 'created_default'
                continue
                
            signal_series = df[signal]
            nan_count = signal_series.isna().sum()
            total_count = len(signal_series)
            nan_ratio = nan_count / total_count if total_count > 0 else 1.0
            repair_stats['critical_signals_status'][signal] = {
                'nan_count': nan_count,
                'nan_ratio': nan_ratio,
                'original_mean': signal_series.mean(),
                'repaired': False
            }
            # 如果NaN比例过高，需要修复
            if nan_ratio > 0.05:  # 5%阈值
                print(f"【V7.0信号修复】关键信号{signal}: NaN比例{nan_ratio:.1%}，开始修复")
                
                # 根据信号类型选择修复策略
                original_dtype = signal_series.dtype
                repaired_series = signal_series.copy()
                
                # 策略1：对于数值型信号，使用线性插值
                if np.issubdtype(original_dtype, np.number):
                    # 首先尝试线性插值
                    repaired_series = repaired_series.interpolate(method='linear', limit_direction='both')
                    remaining_nan = repaired_series.isna().sum()
                    
                    # 如果还有NaN，使用前向填充
                    if remaining_nan > 0:
                        repaired_series = repaired_series.ffill()
                        remaining_nan = repaired_series.isna().sum()
                        
                        # 如果还有NaN，使用后向填充
                        if remaining_nan > 0:
                            repaired_series = repaired_series.bfill()
                    
                    repair_methods_count['linear_interpolation'] += 1
                    method_used = 'linear_interpolation'
                else:
                    # 对于非数值型信号，使用众数填充或前向填充
                    if signal_series.dtype == object:
                        # 尝试找到最常见的非NaN值
                        mode_values = signal_series.mode()
                        if len(mode_values) > 0:
                            fill_value = mode_values[0]
                            repaired_series = signal_series.fillna(fill_value)
                            method_used = 'mode_fill'
                        else:
                            repaired_series = signal_series.ffill().bfill()
                            method_used = 'forward_backward_fill'
                    else:
                        repaired_series = signal_series.ffill().bfill()
                        method_used = 'forward_backward_fill'
                
                # 检查修复效果
                final_nan_count = repaired_series.isna().sum()
                if final_nan_count == 0:
                    df[signal] = repaired_series
                    repair_stats['critical_signals_status'][signal]['repaired'] = True
                    repair_stats['critical_signals_status'][signal]['method_used'] = method_used
                    repair_stats['critical_signals_status'][signal]['final_nan_count'] = 0
                    repair_stats['critical_repaired'] += 1
                    repair_stats['total_nan_filled'] += nan_count
                    print(f"  → 成功修复，清除{nan_count}个NaN值")
                else:
                    print(f"  → 修复失败，仍有{final_nan_count}个NaN值")
        # 第二步：批量处理其他信号的NaN值
        print(f"【V7.0信号修复】第二步：批量处理其他信号的NaN值")
        all_signals = list(df.columns)
        non_critical_signals = [sig for sig in all_signals if sig not in critical_signals_v7]
        for signal in non_critical_signals:
            signal_series = df[signal]
            nan_count = signal_series.isna().sum()
            if nan_count > 0:
                # 只对有一定数量NaN的信号进行修复
                nan_ratio = nan_count / len(signal_series)
                if nan_ratio > 0.1:  # 10%阈值
                    # 简单的前向后向填充
                    repaired_series = signal_series.ffill().bfill()
                    remaining_nan = repaired_series.isna().sum()
                    
                    if remaining_nan == 0:
                        df[signal] = repaired_series
                        repair_stats['signals_repaired'] += 1
                        repair_stats['total_nan_filled'] += nan_count
                        repair_methods_count['forward_fill'] += 1
        # 第三步：确保没有全NaN的列
        print(f"【V7.0信号修复】第三步：检查全NaN列")
        all_nan_columns = []
        for column in df.columns:
            if df[column].isna().all():
                all_nan_columns.append(column)
        if all_nan_columns:
            print(f"【V7.0信号修复警告】发现{len(all_nan_columns)}个全NaN列: {all_nan_columns[:10]}{'...' if len(all_nan_columns) > 10 else ''}")
            for column in all_nan_columns:
                # 根据列名判断填充值类型
                if any(keyword in column for keyword in ['score', 'ratio', 'index', 'pct']):
                    df[column] = 0.5  # 分数/比例类用中性值
                elif any(keyword in column for keyword in ['close', 'price', 'high', 'low', 'open']):
                    if 'close_D' in df.columns and not df['close_D'].isna().all():
                        df[column] = df['close_D']  # 价格类用收盘价
                    else:
                        df[column] = 0.0
                else:
                    df[column] = 0.0  # 其他用0
        # 第四步：最终检查
        print(f"【V7.0信号修复】第四步：最终检查")
        final_nan_report = {}
        for signal in critical_signals_v7:
            if signal in df.columns:
                nan_count = df[signal].isna().sum()
                if nan_count > 0:
                    final_nan_report[signal] = nan_count
        if final_nan_report:
            print(f"【V7.0信号修复警告】修复后仍有{len(final_nan_report)}个关键信号存在NaN")
            for signal, count in list(final_nan_report.items())[:5]:
                print(f"  {signal}: {count}个NaN")
            if len(final_nan_report) > 5:
                print(f"  ... 还有{len(final_nan_report)-5}个信号")
        else:
            print(f"【V7.0信号修复】所有关键信号已无NaN值")
        # 更新修复统计
        repair_stats['signals_checked'] = len(all_signals)
        repair_stats['repair_methods_used'] = repair_methods_count
        # 计算整体NaN比例
        total_cells = df.shape[0] * df.shape[1]
        total_nan_after = df.isna().sum().sum()
        overall_nan_ratio = total_nan_after / total_cells if total_cells > 0 else 0
        # 探针输出
        print(f"【V7.0信号修复完成】统计信息:")
        print(f"  检查信号数: {repair_stats['signals_checked']}")
        print(f"  修复信号数: {repair_stats['signals_repaired']} (关键信号: {repair_stats['critical_repaired']})")
        print(f"  填充NaN总数: {repair_stats['total_nan_filled']}")
        print(f"  修复方法分布: {repair_methods_count}")
        print(f"  整体NaN比例: {overall_nan_ratio:.4%}")
        # 关键信号修复状态
        critical_repair_summary = {}
        for signal, status in repair_stats['critical_signals_status'].items():
            if isinstance(status, dict) and status.get('repaired', False):
                critical_repair_summary[signal] = {
                    'nan_ratio_before': status.get('nan_ratio', 0),
                    'method': status.get('method_used', 'unknown')
                }
        print(f"  关键信号修复详情: {len(critical_repair_summary)}个信号被修复")
        for signal, info in list(critical_repair_summary.items())[:3]:
            print(f"    {signal}: NaN比例{info['nan_ratio_before']:.1%} -> 使用{info['method']}")
        if len(critical_repair_summary) > 3:
            print(f"    ... 还有{len(critical_repair_summary)-3}个关键信号")
        return df


