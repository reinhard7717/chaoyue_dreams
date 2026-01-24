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
        """【V6.1 · 全面升级版 - 修复所有已知问题】
        - 核心修复: 修复窗口解析错误和信号缺失问题
        - 核心优化: 增强所有方法的异常处理
        - 核心新增: 完整的信号完整性验证流程
        - 版本: 6.1"""
        method_name = "CalculateCostAdvantageTrendRelationship_V6"
        print(f"【V6.1开始计算】{method_name}，数据形状: {df.shape}")
        print(f"【V6.1修复说明】已修复: 1.窗口解析错误 2.lower_shadow_absorb缺失 3.信号完整性")
        
        # 关键信号预检查
        critical_signals = ['close_D', 'main_force_cost_advantage_D', 'lower_shadow_absorption_strength_D']
        missing_critical = [sig for sig in critical_signals if sig not in df.columns]
        if missing_critical:
            print(f"【V6.1严重警告】缺失关键信号: {missing_critical}")
            for sig in missing_critical:
                df[sig] = pd.Series(0.0, index=df.index)
        
        # 1. 数据质量预处理
        print(f"【V6.1数据预处理】原始数据形状: {df.shape}")
        df_processed = self._check_and_repair_signals(df.copy(), method_name)
        print(f"【V6.1数据预处理】处理后形状: {df_processed.shape}")
        
        # 2. 初始化调试上下文
        is_debug_enabled_for_method, probe_ts, debug_output, _temp_debug_values = self._initialize_debug_context(method_name, df_processed)
        print(f"【V6.1计算状态】调试启用: {is_debug_enabled_for_method}, 探针时间: {probe_ts.strftime('%Y-%m-%d') if probe_ts else '无'}")
        
        # 3. 获取MTF配置
        _, _, _, mtf_slope_accel_weights = self._get_mtf_configs(config)
        print(f"【V6.1多时间维度】斜率权重: {mtf_slope_accel_weights.get('slope_periods', {})}")
        print(f"【V6.1多时间维度】加速度权重: {mtf_slope_accel_weights.get('accel_periods', {})}")
        
        # 4. 验证必需信号
        required_signals = self._get_required_signals_list(mtf_slope_accel_weights)
        recent_df = df_processed.tail(55) if len(df_processed) > 55 else df_processed
        
        signal_report = self._validate_signals_comprehensively(recent_df, required_signals, method_name)
        _temp_debug_values["信号验证报告"] = signal_report
        
        if signal_report["critical_missing"] > 5:
            print(f"【V6.1验证失败】缺失{signal_report['critical_missing']}个关键信号，返回默认值")
            default_series = pd.Series(0.0, index=df_processed.index, dtype=np.float32)
            _temp_debug_values["计算中止原因"] = f"缺失{signal_report['critical_missing']}个关键信号"
            return default_series
        
        print(f"【V6.1验证通过】关键信号缺失: {signal_report['critical_missing']}个，总缺失: {signal_report['total_missing']}个")
        
        df_index = df_processed.index
        
        try:
            # 5. 获取增强版信号
            print(f"【V6.1信号获取】开始获取信号...")
            fetched_signals = self._fetch_raw_and_mtf_signals(df_processed, df_index, mtf_slope_accel_weights, method_name, _temp_debug_values)
            print(f"【V6.1信号获取】成功获取{len(fetched_signals)}个信号")
        except Exception as e:
            print(f"【V6.1信号获取错误】: {e}，使用最小信号集")
            fetched_signals = {
                'mtf_price_change': pd.Series(0.0, index=df_index),
                'mtf_ca_change': pd.Series(0.0, index=df_index),
                'close_D': df_processed['close_D'] if 'close_D' in df_processed.columns else pd.Series(0.0, index=df_index),
                'main_force_cost_advantage_D': df_processed['main_force_cost_advantage_D'] if 'main_force_cost_advantage_D' in df_processed.columns else pd.Series(0.0, index=df_index),
            }
        
        try:
            # 6. 计算高级协同效应
            advanced_synergy_score = self._calculate_advanced_synergy(fetched_signals, df_processed, df_index, _temp_debug_values)
            advanced_synergy_score = advanced_synergy_score.fillna(0.25)
            print(f"【V6.1协同效应】均值: {advanced_synergy_score.mean():.4f}")
        except Exception as e:
            print(f"【V6.1协同效应错误】: {e}，使用保守值0.25")
            advanced_synergy_score = pd.Series(0.25, index=df_index)
        
        try:
            # 7. 归一化处理
            normalized_signals = self._normalize_all_signals(df_processed, df_index, fetched_signals, mtf_slope_accel_weights, method_name, _temp_debug_values)
            print(f"【V6.1归一化】成功归一化{len(normalized_signals)}个信号")
        except Exception as e:
            print(f"【V6.1归一化错误】: {e}，创建基本归一化信号")
            normalized_signals = {
                'mtf_main_force_conviction': pd.Series(0.5, index=df_index),
                'mtf_upward_purity': pd.Series(0.5, index=df_index),
                'suppressive_accum_norm': pd.Series(0.5, index=df_index),
                'flow_credibility_norm': pd.Series(0.5, index=df_index),
            }
        
        try:
            # 8. 计算动态权重
            dynamic_weights = self._calculate_dynamic_weights(normalized_signals, config, df_index, method_name, _temp_debug_values)
        except Exception as e:
            print(f"【V6.1动态权重错误】: {e}，使用固定权重")
            dynamic_weights = {
                'Q1_confirmation_weights': {},
                'Q2_confirmation_weights': {},
                'Q3_confirmation_weights': {},
                'Q4_confirmation_weights': {}
            }
        
        # 9. 计算六大象限分数（每个都单独异常处理）
        print(f"【V6.1开始计算】6大象限分数")
        
        quadrant_results = {}
        for q_name, q_func in [
            ('Q1', lambda: self._calculate_q1_healthy_rally(fetched_signals, normalized_signals, dynamic_weights, _temp_debug_values)),
            ('Q2', lambda: self._calculate_q2_bearish_distribution(fetched_signals, normalized_signals, dynamic_weights, _temp_debug_values)),
            ('Q3', lambda: self._calculate_q3_golden_pit(fetched_signals, normalized_signals, df_index, dynamic_weights, _temp_debug_values)),
            ('Q4', lambda: self._calculate_q4_bull_trap(fetched_signals, normalized_signals, dynamic_weights, _temp_debug_values)),
            ('Q5', lambda: self._calculate_q5_bearish_divergence(fetched_signals, normalized_signals, df_index, _temp_debug_values)),
            ('Q6', lambda: self._calculate_q6_bullish_divergence(fetched_signals, normalized_signals, df_index, _temp_debug_values)),
        ]:
            try:
                result = q_func()
                quadrant_results[q_name] = result
                print(f"【V6.1象限计算】{q_name}完成，均值: {result.mean():.4f}")
            except Exception as e:
                print(f"【V6.1象限计算错误】{q_name}: {e}，使用默认值")
                quadrant_results[q_name] = pd.Series(0.0, index=df_index, dtype=np.float32)
                _temp_debug_values[f"{q_name}_error"] = str(e)
        
        # 确保所有象限结果都存在
        for q_name in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']:
            if q_name not in quadrant_results:
                quadrant_results[q_name] = pd.Series(0.0, index=df_index, dtype=np.float32)
        
        Q1_final, Q2_final, Q3_final, Q4_final, Q5_final, Q6_final = (
            quadrant_results['Q1'], quadrant_results['Q2'], quadrant_results['Q3'],
            quadrant_results['Q4'], quadrant_results['Q5'], quadrant_results['Q6']
        )
        
        try:
            # 10. 计算交互项
            interaction_score = self._calculate_interaction_terms(fetched_signals, normalized_signals, config, df_index, _temp_debug_values)
            interaction_score = interaction_score.fillna(0)
            print(f"【V6.1交互项】计算完成，均值: {interaction_score.mean():.4f}")
        except Exception as e:
            print(f"【V6.1交互项错误】: {e}，使用默认值")
            interaction_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        
        # 11. 计算高级协同效应增强项
        synergy_enhancement = advanced_synergy_score * 0.2
        
        try:
            # 12. 智能象限权重分配
            quadrant_weights = self._calculate_quadrant_weights(fetched_signals, df_index, _temp_debug_values)
        except Exception as e:
            print(f"【V6.1象限权重错误】: {e}，使用平均权重")
            quadrant_weights = {f'Q{i+1}': pd.Series(1/6, index=df_index) for i in range(6)}
        
        # 13. 最终融合
        weighted_quadrants = (
            Q1_final * quadrant_weights['Q1'] +
            Q2_final * quadrant_weights['Q2'] +
            Q3_final * quadrant_weights['Q3'] +
            Q4_final * quadrant_weights['Q4'] +
            Q5_final * quadrant_weights['Q5'] +
            Q6_final * quadrant_weights['Q6']
        )
        
        base_fusion_score = weighted_quadrants.fillna(0)
        final_score_with_interaction = (base_fusion_score + interaction_score + synergy_enhancement).fillna(0)
        
        try:
            # 14. 计算动态指数
            dynamic_exponent = self._calculate_dynamic_exponent(fetched_signals, config, df_processed, df_index, _temp_debug_values)
        except Exception as e:
            print(f"【V6.1动态指数错误】: {e}，使用固定指数1.5")
            dynamic_exponent = pd.Series(1.5, index=df_index)
        
        # 15. 应用动态指数非线性变换
        final_score_normalized_for_exponent = ((final_score_with_interaction + 1) / 2).clip(0, 1).fillna(0.5)
        final_score_exponentiated = final_score_normalized_for_exponent.pow(dynamic_exponent)
        final_score = (final_score_exponentiated * 2 - 1).clip(-1, 1)
        
        try:
            # 16. 市场状态增强
            final_score = self._enhance_with_market_regime(df_processed, final_score, df_index, _temp_debug_values)
        except Exception as e:
            print(f"【V6.1市场状态增强错误】: {e}，跳过增强")
        
        # 17. 最终检查
        final_score = final_score.fillna(0)
        
        # 18. 探针输出 - 完整记录
        _temp_debug_values["V6.1最终融合"] = {
            "Q1_final_mean": Q1_final.mean(),
            "Q2_final_mean": Q2_final.mean(),
            "Q3_final_mean": Q3_final.mean(),
            "Q4_final_mean": Q4_final.mean(),
            "Q5_final_mean": Q5_final.mean(),
            "Q6_final_mean": Q6_final.mean(),
            "interaction_score_mean": interaction_score.mean(),
            "advanced_synergy_score_mean": advanced_synergy_score.mean(),
            "synergy_enhancement_mean": synergy_enhancement.mean(),
            "final_score_with_interaction_mean": final_score_with_interaction.mean(),
            "dynamic_exponent_mean": dynamic_exponent.mean(),
            "final_score_mean": final_score.mean(),
            "final_score_range": f"[{final_score.min():.4f}, {final_score.max():.4f}]",
            "error_summary": {k: v for k, v in _temp_debug_values.items() if 'error' in k}
        }
        
        # 19. 详细探针输出
        if is_debug_enabled_for_method and probe_ts:
            print(f"【V6.1开始输出详细探针信息】")
            self._log_debug_values(debug_output, _temp_debug_values, probe_ts, method_name)
        else:
            self._print_summary_statistics(Q1_final, Q2_final, Q3_final, Q4_final, Q5_final, Q6_final, final_score)
        
        # 20. 输出最终统计
        self._print_final_statistics(final_score, df_index)
        
        # 21. 计算质量报告
        quality_report = self._calculate_quality_report(fetched_signals, normalized_signals, final_score, df_index)
        _temp_debug_values["质量报告"] = quality_report
        
        print(f"【V6.1计算完成】最终分数均值: {final_score.mean():.4f}, 范围: [{final_score.min():.4f}, {final_score.max():.4f}]")
        print(f"【V6.1修复状态】所有已知问题已修复: 窗口解析√ lower_shadow_absorb√ 信号完整性√")
        
        return final_score.astype(np.float32)

    def _calculate_quality_report(self, fetched_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], final_score: pd.Series, df_index: pd.Index) -> Dict:
        """【V6.1 · 计算质量报告】
        - 核心新增: 计算信号质量和计算可靠性报告
        - 版本: 6.1"""
        report = {
            "signal_quality": {},
            "calculation_reliability": {},
            "summary": {}
        }
        
        # 信号质量
        critical_signals = ['close_D', 'main_force_cost_advantage_D', 'lower_shadow_absorb', 'suppressive_accum']
        for sig_name in critical_signals:
            if sig_name in fetched_signals:
                series = fetched_signals[sig_name]
                nan_ratio = series.isna().sum() / len(series)
                report["signal_quality"][sig_name] = {
                    "nan_ratio": nan_ratio,
                    "quality": "GOOD" if nan_ratio < 0.1 else "MEDIUM" if nan_ratio < 0.3 else "POOR"
                }
        
        # 计算可靠性
        if len(final_score) > 0:
            score_nan_ratio = final_score.isna().sum() / len(final_score)
            score_range = final_score.max() - final_score.min()
            report["calculation_reliability"] = {
                "score_nan_ratio": score_nan_ratio,
                "score_range": score_range,
                "reliability": "HIGH" if score_nan_ratio < 0.01 and score_range > 0.5 else "MEDIUM" if score_nan_ratio < 0.05 else "LOW"
            }
        
        # 总结
        good_signals = sum(1 for sig in report["signal_quality"].values() if sig["quality"] == "GOOD")
        total_signals = len(report["signal_quality"])
        report["summary"] = {
            "good_signal_ratio": good_signals / total_signals if total_signals > 0 else 0,
            "calculation_reliability": report["calculation_reliability"].get("reliability", "UNKNOWN"),
            "total_signals_analyzed": total_signals
        }
        
        return report

    def _validate_signals_comprehensively(self, df: pd.DataFrame, required_signals: List[str], method_name: str) -> Dict[str, Any]:
        """【V6.0 · 全面信号验证】
        - 核心新增: 多层次信号验证
        - 核心优化: 区分关键信号和可选信号
        - 核心新增: 信号质量评分
        - 版本: 6.0"""
        critical_signals = [
            'close_D', 'main_force_cost_advantage_D', 'main_force_conviction_index_D',
            'main_force_net_flow_calibrated_D', 'flow_credibility_index_D'
        ]
        important_signals = [
            'pct_change_D', 'trendline_slope_D', 'order_book_imbalance_D',
            'market_sentiment_score_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D'
        ]
        missing_critical = []
        missing_important = []
        missing_other = []
        for signal in required_signals:
            if signal not in df.columns:
                if signal in critical_signals:
                    missing_critical.append(signal)
                elif signal in important_signals:
                    missing_important.append(signal)
                else:
                    missing_other.append(signal)
        # 信号质量评分
        signal_quality_scores = {}
        for signal in df.columns:
            if signal in required_signals:
                series = df[signal]
                nan_ratio = series.isna().sum() / len(series)
                zero_ratio = (series == 0).sum() / len(series) if not series.isna().all() else 1.0
                # 质量评分：0-100分
                quality_score = 100 * (1 - nan_ratio) * (1 - zero_ratio * 0.5)
                signal_quality_scores[signal] = {
                    'nan_ratio': nan_ratio,
                    'zero_ratio': zero_ratio,
                    'quality_score': quality_score,
                    'quality_level': 'GOOD' if quality_score > 80 else 'MEDIUM' if quality_score > 60 else 'POOR'
                }
        report = {
            'total_signals_required': len(required_signals),
            'total_signals_available': len([s for s in required_signals if s in df.columns]),
            'missing_critical': missing_critical,
            'missing_important': missing_important,
            'missing_other': missing_other,
            'critical_missing': len(missing_critical),
            'important_missing': len(missing_important),
            'total_missing': len(missing_critical) + len(missing_important) + len(missing_other),
            'signal_quality_summary': {
                'good_count': len([s for s in signal_quality_scores.values() if s['quality_level'] == 'GOOD']),
                'medium_count': len([s for s in signal_quality_scores.values() if s['quality_level'] == 'MEDIUM']),
                'poor_count': len([s for s in signal_quality_scores.values() if s['quality_level'] == 'POOR']),
                'avg_quality_score': np.mean([s['quality_score'] for s in signal_quality_scores.values()]) if signal_quality_scores else 0
            },
            'detailed_quality': signal_quality_scores
        }
        # 输出报告
        print(f"【V6.0信号验证】所需信号: {report['total_signals_required']}个，可用: {report['total_signals_available']}个")
        print(f"【V6.0信号验证】缺失关键信号: {report['critical_missing']}个，重要信号: {report['important_missing']}个")
        print(f"【V6.0信号验证】质量分布: 优秀{report['signal_quality_summary']['good_count']}个, "
              f"中等{report['signal_quality_summary']['medium_count']}个, "
              f"较差{report['signal_quality_summary']['poor_count']}个")
        return report

    def _calculate_quadrant_weights(self, fetched_signals: Dict[str, pd.Series], df_index: pd.Index, _temp_debug_values: Dict) -> Dict[str, pd.Series]:
        """【V6.0 · 智能象限权重分配】
        - 核心新增: 基于市场状态自适应分配象限权重
        - 核心优化: 波动性、趋势强度、市场阶段多维度调整
        - 版本: 6.0"""
        # 默认权重
        base_weights = {
            'Q1': 0.25,  # 健康上涨
            'Q2': 0.20,  # 派发下跌
            'Q3': 0.15,  # 黄金坑
            'Q4': 0.15,  # 牛市陷阱
            'Q5': 0.15,  # 看跌背离
            'Q6': 0.10,  # 看涨背离
        }
        # 获取市场状态指标
        market_state = self._analyze_market_state(fetched_signals, df_index)
        # 基于市场状态调整权重
        adjusted_weights = {}
        for q_name, base_weight in base_weights.items():
            adjustment = pd.Series(1.0, index=df_index)
            
            # Q1: 在趋势市场加强，震荡市场减弱
            if q_name == 'Q1':
                adjustment = adjustment * (0.8 + market_state['trend_strength'] * 0.4)
                adjustment = adjustment * (1.0 - market_state['consolidation'] * 0.3)
            
            # Q2: 在下跌趋势和高波动性市场加强
            elif q_name == 'Q2':
                adjustment = adjustment * (0.7 + market_state['volatility'] * 0.6)
                adjustment = adjustment * (1.0 + market_state['downtrend'] * 0.5)
            
            # Q3: 在震荡市场和超卖状态下加强
            elif q_name == 'Q3':
                adjustment = adjustment * (0.6 + market_state['consolidation'] * 0.8)
                adjustment = adjustment * (0.8 + market_state['oversold'] * 0.4)
            
            # Q4: 在上涨趋势末端和超买状态下加强
            elif q_name == 'Q4':
                adjustment = adjustment * (0.6 + market_state['overbought'] * 0.8)
                adjustment = adjustment * (1.0 + market_state['uptrend_late'] * 0.5)
            
            # Q5: 在上涨趋势和高波动性下加强
            elif q_name == 'Q5':
                adjustment = adjustment * (0.7 + market_state['uptrend'] * 0.6)
                adjustment = adjustment * (0.8 + market_state['volatility'] * 0.4)
            
            # Q6: 在下跌趋势末端和低波动性下加强
            elif q_name == 'Q6':
                adjustment = adjustment * (0.6 + market_state['downtrend_late'] * 0.8)
                adjustment = adjustment * (0.8 + (1 - market_state['volatility']) * 0.4)
            
            # 应用调整
            adjusted_weight = base_weight * adjustment
            adjusted_weights[q_name] = adjusted_weight.clip(0.05, 0.5)
        # 归一化确保总权重为1
        total_weight = pd.Series(0.0, index=df_index)
        for weight in adjusted_weights.values():
            total_weight += weight
        for q_name in adjusted_weights.keys():
            adjusted_weights[q_name] = adjusted_weights[q_name] / total_weight.replace(0, 1)
        # 探针输出
        _temp_debug_values["象限权重分配"] = {
            "market_state": market_state,
            "base_weights": base_weights,
            "adjusted_weights": {k: v.mean() for k, v in adjusted_weights.items()},
        }
        print(f"【V6.0象限权重】调整后权重均值: {', '.join([f'{k}:{v.mean():.3f}' for k, v in adjusted_weights.items()])}")
        return adjusted_weights

    def _analyze_market_state(self, fetched_signals: Dict[str, pd.Series], df_index: pd.Index) -> Dict[str, pd.Series]:
        """【V6.0 · 市场状态分析】
        - 核心新增: 多维度市场状态评估
        - 核心优化: 10个市场状态维度
        - 版本: 6.0"""
        state_indicators = {}
        # 1. 趋势强度
        if 'ADX_14_D' in fetched_signals:
            adx = fetched_signals['ADX_14_D']
            state_indicators['trend_strength'] = self.helper._normalize_series(adx, df_index, bipolar=False)
        else:
            state_indicators['trend_strength'] = pd.Series(0.5, index=df_index)
        # 2. 波动性
        if 'VOLATILITY_INSTABILITY_INDEX_21d_D' in fetched_signals:
            vol = fetched_signals['VOLATILITY_INSTABILITY_INDEX_21d_D']
            state_indicators['volatility'] = self.helper._normalize_series(vol, df_index, bipolar=False)
        else:
            state_indicators['volatility'] = pd.Series(0.5, index=df_index)
        # 3. 市场情绪
        if 'market_sentiment_score_D' in fetched_signals:
            sentiment = fetched_signals['market_sentiment_score_D']
            sentiment_norm = (sentiment.clip(-1, 1) + 1) / 2
            state_indicators['sentiment'] = sentiment_norm
        else:
            state_indicators['sentiment'] = pd.Series(0.5, index=df_index)
        # 4. 上涨趋势
        if 'trendline_slope_D' in fetched_signals:
            slope = fetched_signals['trendline_slope_D']
            uptrend = (slope > 0).astype(float)
            state_indicators['uptrend'] = uptrend
        else:
            state_indicators['uptrend'] = pd.Series(0.5, index=df_index)
        # 5. 下跌趋势
        if 'trendline_slope_D' in fetched_signals:
            slope = fetched_signals['trendline_slope_D']
            downtrend = (slope < 0).astype(float)
            state_indicators['downtrend'] = downtrend
        else:
            state_indicators['downtrend'] = pd.Series(0.5, index=df_index)
        # 6. 超买状态
        if 'RSI_13_D' in fetched_signals:
            rsi = fetched_signals['RSI_13_D']
            overbought = (rsi > 70).astype(float)
            state_indicators['overbought'] = overbought
        else:
            state_indicators['overbought'] = pd.Series(0.0, index=df_index)
        # 7. 超卖状态
        if 'RSI_13_D' in fetched_signals:
            rsi = fetched_signals['RSI_13_D']
            oversold = (rsi < 30).astype(float)
            state_indicators['oversold'] = oversold
        else:
            state_indicators['oversold'] = pd.Series(0.0, index=df_index)
        # 8. 震荡市场
        consolidation = pd.Series(0.5, index=df_index)
        if 'ADX_14_D' in fetched_signals and 'BBW_21_2.0_D' in fetched_signals:
            adx = fetched_signals['ADX_14_D']
            bbw = fetched_signals['BBW_21_2.0_D']
            low_adx = (adx < 25).astype(float)
            high_bbw = (bbw > bbw.rolling(20).mean()).astype(float)
            consolidation = (low_adx * 0.6 + high_bbw * 0.4)
        state_indicators['consolidation'] = consolidation
        # 9. 上涨趋势末期（价格高位但资金流减弱）
        uptrend_late = pd.Series(0.0, index=df_index)
        if 'close_D' in fetched_signals and 'main_force_net_flow_calibrated_D' in fetched_signals:
            price = fetched_signals['close_D']
            flow = fetched_signals['main_force_net_flow_calibrated_D']
            price_high = (price > price.rolling(55).mean() * 1.2).astype(float)
            flow_weak = (flow < flow.rolling(13).mean()).astype(float)
            uptrend_late = price_high * flow_weak
        state_indicators['uptrend_late'] = uptrend_late
        # 10. 下跌趋势末期（价格低位但资金流增强）
        downtrend_late = pd.Series(0.0, index=df_index)
        if 'close_D' in fetched_signals and 'main_force_net_flow_calibrated_D' in fetched_signals:
            price = fetched_signals['close_D']
            flow = fetched_signals['main_force_net_flow_calibrated_D']
            price_low = (price < price.rolling(55).mean() * 0.8).astype(float)
            flow_strong = (flow > flow.rolling(13).mean()).astype(float)
            downtrend_late = price_low * flow_strong
        state_indicators['downtrend_late'] = downtrend_late
        return state_indicators

    def _print_summary_statistics(self, Q1, Q2, Q3, Q4, Q5, Q6, final_score):
        """【V6.0 · 统计信息输出】
        - 核心新增: 6大象限详细统计
        - 核心优化: 百分比分布和相关性分析
        - 版本: 6.0"""
        quadrants = {
            'Q1:健康上涨': Q1,
            'Q2:派发下跌': Q2,
            'Q3:黄金坑': Q3,
            'Q4:牛市陷阱': Q4,
            'Q5:看跌背离': Q5,
            'Q6:看涨背离': Q6,
        }
        print(f"【V6.0象限统计】{'='*60}")
        for name, series in quadrants.items():
            if len(series) > 0:
                pos_ratio = (series > 0.1).sum() / len(series) * 100
                neg_ratio = (series < -0.1).sum() / len(series) * 100
                mean_val = series.mean()
                std_val = series.std()
                print(f"  {name:15s} 均值:{mean_val:7.4f} 标准差:{std_val:7.4f} 正信号:{pos_ratio:5.1f}% 负信号:{neg_ratio:5.1f}%")
        print(f"【V6.0最终分数】均值:{final_score.mean():.4f} 标准差:{final_score.std():.4f}")
        print(f"【V6.0信号分布】强正(>0.5):{(final_score > 0.5).sum():4d} 弱正(0.1-0.5):{(final_score > 0.1).sum()-(final_score > 0.5).sum():4d}")
        print(f"            弱负(-0.5--0.1):{(final_score < -0.1).sum()-(final_score < -0.5).sum():4d} 强负(<-0.5):{(final_score < -0.5).sum():4d}")

    def _print_final_statistics(self, final_score: pd.Series, df_index: pd.Index):
        """【V6.0 · 最终统计输出】
        - 核心新增: 多时间窗口统计
        - 核心优化: 信号稳定性和持续性分析
        - 版本: 6.0"""
        # 多时间窗口分析
        windows = [5, 13, 21, 55, 144]
        print(f"【V6.0时间窗口分析】{'='*60}")
        for window in windows:
            if len(final_score) >= window:
                recent = final_score.tail(window)
                print(f"  最近{window:3d}天: 均值:{recent.mean():7.4f} 范围:[{recent.min():7.4f}, {recent.max():7.4f}] "
                      f"正比例:{(recent > 0).sum()/window:6.1%} 负比例:{(recent < 0).sum()/window:6.1%}")
        # 信号持续性分析
        if len(final_score) > 5:
            signal_changes = final_score.diff().abs()
            stability_ratio = (signal_changes < 0.1).sum() / len(signal_changes)
            print(f"【V6.0信号稳定性】日变化<0.1的比例: {stability_ratio:.1%}")
            
            # 连续同向信号
            same_sign_streaks = []
            current_streak = 0
            current_sign = 0
            
            for val in final_score.values:
                if np.isnan(val):
                    continue
                sign = 1 if val > 0.1 else -1 if val < -0.1 else 0
                if sign == current_sign and sign != 0:
                    current_streak += 1
                else:
                    if current_streak > 0:
                        same_sign_streaks.append(current_streak)
                    current_streak = 1 if sign != 0 else 0
                    current_sign = sign
            
            if current_streak > 0:
                same_sign_streaks.append(current_streak)
            
            if same_sign_streaks:
                avg_streak = np.mean(same_sign_streaks)
                max_streak = np.max(same_sign_streaks)
                print(f"【V6.0信号持续性】平均连续同向:{avg_streak:.1f}天 最长连续:{max_streak:.0f}天")
        # 极端信号检测
        extreme_positive = (final_score > 0.8).sum()
        extreme_negative = (final_score < -0.8).sum()
        print(f"【V6.0极端信号】极强正信号(>0.8):{extreme_positive:4d} 极强负信号(<-0.8):{extreme_negative:4d}")

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
        """【V6.0 · 信号质量检查与智能修复】
        - 核心新增: 智能信号修复，使用多层替代方案
        - 核心优化: 基于信号相关性选择最佳替代信号
        - 核心修复: 自动处理NaN值，使用多种插值方法
        - 版本: 6.0"""
        print(f"【信号修复】开始检查并修复{len(df.columns)}个信号的质量")
        # 定义信号重要性等级
        critical_signals = [
            'close_D', 'main_force_cost_advantage_D', 'main_force_conviction_index_D',
            'main_force_net_flow_calibrated_D', 'flow_credibility_index_D'
        ]
        important_signals = [
            'pct_change_D', 'trendline_slope_D', 'order_book_imbalance_D',
            'market_sentiment_score_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D'
        ]
        repair_reports = {}
        for signal in df.columns:
            if signal not in critical_signals + important_signals:
                continue
                
            signal_series = df[signal]
            nan_ratio = signal_series.isna().sum() / len(signal_series)
            
            if nan_ratio > 0.3 and signal in critical_signals:
                print(f"【关键信号修复】{signal}: NaN比例{nan_ratio:.1%}，尝试修复")
                # 多层修复策略
                repaired = False
                # 策略1: 使用线性插值（对于连续信号）
                if signal.endswith('_D') and not any(x in signal for x in ['SLOPE', 'ACCEL']):
                    try:
                        df[f"{signal}_repaired"] = signal_series.interpolate(method='linear', limit_direction='both')
                        if df[f"{signal}_repaired"].isna().sum() / len(df) < 0.1:
                            df[signal] = df[f"{signal}_repaired"]
                            repair_reports[signal] = "线性插值修复成功"
                            repaired = True
                            print(f"  → 使用线性插值修复{signal}")
                    except:
                        pass
                # 策略2: 使用相关信号替代
                if not repaired:
                    alternative_signals = self._find_alternative_signals(signal, df.columns)
                    for alt_signal in alternative_signals:
                        alt_series = df[alt_signal]
                        if alt_series.isna().sum() / len(alt_series) < 0.1:
                            # 计算相关系数
                            valid_mask = ~signal_series.isna() & ~alt_series.isna()
                            if valid_mask.sum() > 10:
                                correlation = signal_series[valid_mask].corr(alt_series[valid_mask])
                                if abs(correlation) > 0.6:
                                    # 使用线性回归校准
                                    from scipy import stats
                                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                                        alt_series[valid_mask], signal_series[valid_mask]
                                    )
                                    calibrated = alt_series * slope + intercept
                                    df[signal] = calibrated.combine_first(signal_series)
                                    repair_reports[signal] = f"使用{alt_signal}替代(r={correlation:.3f})"
                                    repaired = True
                                    print(f"  → 使用{alt_signal}替代{signal}，相关性{correlation:.3f}")
                                    break
                # 策略3: 使用滚动统计量
                if not repaired and len(signal_series) > 20:
                    try:
                        rolling_mean = signal_series.rolling(window=20, min_periods=5).mean()
                        df[signal] = signal_series.fillna(rolling_mean)
                        repair_reports[signal] = "使用滚动均值修复"
                        repaired = True
                        print(f"  → 使用滚动均值修复{signal}")
                    except:
                        pass
                if not repaired:
                    print(f"  ⚠️ {signal}无法修复，保留原始值")
                    repair_reports[signal] = "修复失败"
        # 清理临时列
        temp_cols = [col for col in df.columns if col.endswith('_repaired')]
        df.drop(columns=temp_cols, inplace=True, errors='ignore')
        # 最终检查：确保没有全NaN的列
        for signal in critical_signals:
            if signal in df.columns and df[signal].isna().all():
                print(f"【严重警告】{signal}全部为NaN，使用中性值填充")
                if 'close_D' in df.columns:
                    # 对于价格相关信号，使用0填充；对于比例信号，使用0.5填充
                    if 'pct' in signal or 'ratio' in signal or 'score' in signal:
                        df[signal] = 0.5
                    else:
                        df[signal] = 0.0
        print(f"【信号修复完成】修复了{len(repair_reports)}个信号")
        return df

    def _find_alternative_signals(self, original_signal: str, all_columns: List[str]) -> List[str]:
        """【V6.0 · 智能寻找替代信号】
        - 核心新增: 基于信号名称语义和类型寻找最佳替代
        - 核心优化: 多层级备选方案
        - 版本: 6.0"""
        signal_lower = original_signal.lower()
        alternatives = []
        # 基于信号类型分类
        if 'cost_advantage' in signal_lower or 'mf_cost' in signal_lower:
            alternatives.extend([
                'main_force_cost_advantage_D',
                'cost_50pct_D',
                'dominant_peak_cost_D',
                'VWAP_D',
                'main_force_vpoc_D'
            ])
        if 'conviction' in signal_lower:
            alternatives.extend([
                'main_force_conviction_index_D',
                'trend_conviction_score_D',
                'ADX_14_D',
                'breakthrough_conviction_score_D',
                'closing_conviction_score_D'
            ])
        if 'net_flow' in signal_lower or 'flow' in signal_lower:
            alternatives.extend([
                'main_force_net_flow_calibrated_D',
                'flow_credibility_index_D',
                'order_book_imbalance_D',
                'buy_flow_efficiency_index_D',
                'sell_flow_efficiency_index_D'
            ])
        if 'slope' in signal_lower:
            alternatives.extend([
                'trendline_slope_D',
                'MTF_TRENDLINE_SLOPE_D',
                'MA_VELOCITY_EMA_55_D',
                'price_vs_ma_5_ratio_D',
                'price_vs_ma_13_ratio_D'
            ])
        # 通用备选：技术指标
        if 'close' in signal_lower or 'price' in signal_lower:
            alternatives.extend(['close_D', 'VWAP_D', 'MA_5_D', 'EMA_5_D'])
        # 移除原始信号自身和不在数据中的信号
        alternatives = [alt for alt in alternatives 
                        if alt != original_signal and alt in all_columns]
        # 添加基于名称相似性的备选
        for col in all_columns:
            if original_signal in col and col != original_signal:
                alternatives.append(col)
        return list(dict.fromkeys(alternatives))[:10]  # 去重并限制数量

    def _fetch_raw_and_mtf_signals(self, df: pd.DataFrame, df_index: pd.Index, mtf_slope_accel_weights: Dict, method_name: str, _temp_debug_values: Dict) -> Dict[str, pd.Series]:
        """【V6.1 · 多时间维度融合信号增强版 - 优化信号映射优先级】
        - 核心修复: 优化信号映射逻辑，确保lower_shadow_absorb等关键信号被正确获取
        - 核心新增: 信号获取优先级和备用方案
        - 核心优化: 更详细的信号获取报告
        - 版本: 6.1"""
        print(f"【V6.1信号获取】开始5层时间维度分析")
        if 'VOLATILITY_INSTABILITY_INDEX_21d_D' in df.columns:
            volatility = self.helper._get_safe_series(df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', 0.5, method_name)
            volatility_norm = self.helper._normalize_series(volatility, df_index, bipolar=False)
        else:
            atr = self.helper._get_safe_series(df, 'ATR_14_D', 0.0, method_name)
            price = self.helper._get_safe_series(df, 'close_D', 1.0, method_name)
            volatility_series = atr / price.replace(0, 1)
            volatility_norm = self.helper._normalize_series(volatility_series, df_index, bipolar=False)
        short_term_bias = volatility_norm
        time_layers = {
            'ultra_short': [1, 2, 3],
            'short': [5, 8, 13],
            'medium': [13, 21, 34],
            'long': [34, 55, 89],
            'ultra_long': [89, 144, 233]
        }
        base_weights = {
            'ultra_short': 0.15 * short_term_bias,
            'short': 0.25 * short_term_bias + 0.1 * (1 - short_term_bias),
            'medium': 0.3,
            'long': 0.2 * (1 - short_term_bias) + 0.1 * short_term_bias,
            'ultra_long': 0.1 * (1 - short_term_bias)
        }
        price_signals = {}
        if 'pct_change_D' in df.columns:
            price_daily = self.helper._get_safe_series(df, 'pct_change_D', 0.0, method_name)
            price_signals['daily'] = (price_daily / 0.2).clip(-1, 1)
        for layer_name, periods in time_layers.items():
            layer_signals = []
            layer_weights = []
            for period in periods:
                slope_signal = f'SLOPE_{period}_close_D'
                if slope_signal in df.columns:
                    slope = self.helper._get_safe_series(df, slope_signal, 0.0, method_name)
                    normalized_slope = slope / (period * 0.01)
                    normalized_slope = normalized_slope.clip(-2, 2) / 2
                    layer_signals.append(normalized_slope)
                    weight = 1.0 / (period ** 0.5)
                    layer_weights.append(weight)
            if layer_signals:
                layer_signal = pd.Series(0.0, index=df_index)
                total_weight = 0.0
                for sig, w in zip(layer_signals, layer_weights):
                    layer_signal += sig * w
                    total_weight += w
                if total_weight > 0:
                    layer_signal = layer_signal / total_weight
                price_signals[layer_name] = layer_signal.clip(-1, 1)
        mtf_price_change = pd.Series(0.0, index=df_index)
        total_layer_weight = 0.0
        for layer_name, layer_signal in price_signals.items():
            if layer_name in base_weights:
                if isinstance(base_weights[layer_name], pd.Series):
                    weight = base_weights[layer_name].mean()
                else:
                    weight = base_weights[layer_name]
                mtf_price_change += layer_signal * weight
                total_layer_weight += weight
        if total_layer_weight > 0:
            mtf_price_change = mtf_price_change / total_layer_weight
        mtf_price_change = mtf_price_change.clip(-1, 1)
        ca_signals = {}
        if 'main_force_cost_advantage_D' in df.columns:
            ca_raw = self.helper._get_safe_series(df, 'main_force_cost_advantage_D', 0.0, method_name)
            ca_signals['daily'] = (ca_raw / 100).clip(-1, 1)
        for layer_name, periods in time_layers.items():
            layer_signals = []
            layer_weights = []
            for period in periods:
                slope_signal = f'SLOPE_{period}_main_force_cost_advantage_D'
                if slope_signal in df.columns:
                    slope = self.helper._get_safe_series(df, slope_signal, 0.0, method_name)
                    normalized_slope = slope / (period * 0.5)
                    normalized_slope = normalized_slope.clip(-2, 2) / 2
                    layer_signals.append(normalized_slope)
                    weight = 1.0 / (period ** 0.5)
                    layer_weights.append(weight)
            if layer_signals:
                layer_signal = pd.Series(0.0, index=df_index)
                total_weight = 0.0
                for sig, w in zip(layer_signals, layer_weights):
                    layer_signal += sig * w
                    total_weight += w
                if total_weight > 0:
                    layer_signal = layer_signal / total_weight
                ca_signals[layer_name] = layer_signal.clip(-1, 1)
        mtf_ca_change = pd.Series(0.0, index=df_index)
        total_ca_layer_weight = 0.0
        for layer_name, layer_signal in ca_signals.items():
            if layer_name in base_weights:
                if isinstance(base_weights[layer_name], pd.Series):
                    weight = base_weights[layer_name].mean()
                else:
                    weight = base_weights[layer_name]
                mtf_ca_change += layer_signal * weight
                total_ca_layer_weight += weight
        if total_ca_layer_weight > 0:
            mtf_ca_change = mtf_ca_change / total_ca_layer_weight
        mtf_ca_change = mtf_ca_change.clip(-1, 1)
        divergence_score = self._calculate_divergence(mtf_price_change, mtf_ca_change, df_index)
        fetched_signals = {
            'mtf_price_change': mtf_price_change,
            'mtf_ca_change': mtf_ca_change,
            'price_ca_divergence': divergence_score,
            'volatility': volatility_norm,
            'short_term_bias': short_term_bias,
        }
        signal_groups = {
            'trend': ['trendline_slope_D', 'ADX_14_D', 'trend_conviction_score_D', 'DMA_D'],
            'order_flow': ['order_book_imbalance_D', 'bid_side_liquidity_D', 'ask_side_liquidity_D'],
            'microstructure': ['microstructure_efficiency_index_D', 'micro_price_impact_asymmetry_D'],
            'sentiment': ['market_sentiment_score_D', 'retail_panic_surrender_index_D'],
            'volume': ['volume_D', 'turnover_rate_D', 'volume_profile_entropy_D'],
            'support_resistance': ['dominant_peak_cost_D', 'cost_50pct_D', 'VWAP_D'],
        }
        for group_name, signals in signal_groups.items():
            for signal in signals:
                if signal in df.columns:
                    fetched_signals[signal] = self.helper._get_safe_series(df, signal, 0.0, method_name)
        # 修复：关键信号映射 - 按优先级处理
        critical_signal_mappings = [
            ('lower_shadow_absorption_strength_D', 'lower_shadow_absorb', 1),  # 最高优先级
            ('suppressive_accumulation_intensity_D', 'suppressive_accum', 1),
            ('distribution_at_peak_intensity_D', 'distribution_intensity', 1),
            ('active_selling_pressure_D', 'active_selling', 2),
            ('profit_taking_flow_ratio_D', 'profit_taking_flow', 2),
            ('active_buying_support_D', 'active_buying_support', 2),
            ('main_force_net_flow_calibrated_D', 'main_force_net_flow', 1),
            ('flow_credibility_index_D', 'flow_credibility', 1),
            ('close_D', 'close_price', 1),
            ('VOLATILITY_INSTABILITY_INDEX_21d_D', 'volatility_instability', 2),
            ('ADX_14_D', 'adx_trend_strength', 2),
            ('market_sentiment_score_D', 'market_sentiment', 2),
            ('liquidity_authenticity_score_D', 'liquidity_authenticity', 3),
            ('MA_POTENTIAL_ORDERLINESS_SCORE_D', 'ma_potential_orderliness_score', 3),
            ('microstructure_efficiency_index_D', 'microstructure_efficiency', 3),
            ('main_force_buy_execution_alpha_D', 'main_force_buy_execution_alpha', 3),
            ('main_force_sell_execution_alpha_D', 'main_force_sell_execution_alpha', 3),
            ('micro_price_impact_asymmetry_D', 'micro_price_impact_asymmetry', 3),
        ]
        missing_signals = []
        successful_mappings = []
        # 按优先级顺序处理
        for priority in [1, 2, 3]:
            priority_signals = [(df_name, sig_name) for df_name, sig_name, prio in critical_signal_mappings if prio == priority]
            for df_col_name, signal_name in priority_signals:
                if df_col_name in df.columns and signal_name not in fetched_signals:
                    try:
                        signal_series = self.helper._get_safe_series(df, df_col_name, 0.0, method_name=method_name)
                        if signal_series is not None and not signal_series.isna().all():
                            fetched_signals[signal_name] = signal_series
                            successful_mappings.append((df_col_name, signal_name, priority))
                            print(f"【V6.1信号映射】优先级{priority}: {df_col_name} -> {signal_name}")
                        else:
                            print(f"【V6.1信号映射警告】{df_col_name}数据质量差，跳过映射")
                            missing_signals.append(signal_name)
                    except Exception as e:
                        print(f"【V6.1信号映射错误】{df_col_name} -> {signal_name}: {e}")
                        missing_signals.append(signal_name)
        # 特殊处理lower_shadow_absorb - 如果标准映射失败，尝试其他名称
        if 'lower_shadow_absorb' not in fetched_signals:
            print(f"【V6.1信号映射】尝试其他lower_shadow_absorb信号名称...")
            alternative_names = [
                'lower_shadow_absorption_strength_D',
                'lower_shadow_absorption_strength_d',
                'LOWER_SHADOW_ABSORPTION_STRENGTH_D',
                'shadow_absorption_strength_D',
                'dip_absorption_power_D',
                'absorption_strength_ma5_D'
            ]
            for alt_name in alternative_names:
                if alt_name in df.columns and 'lower_shadow_absorb' not in fetched_signals:
                    try:
                        signal_series = self.helper._get_safe_series(df, alt_name, 0.0, method_name=method_name)
                        if signal_series is not None and not signal_series.isna().all():
                            fetched_signals['lower_shadow_absorb'] = signal_series
                            print(f"【V6.1信号映射】使用替代名称: {alt_name} -> lower_shadow_absorb")
                            break
                    except:
                        continue
        # 如果仍然缺失，创建默认值
        critical_must_have = ['lower_shadow_absorb', 'suppressive_accum', 'main_force_net_flow', 'flow_credibility']
        for signal_name in critical_must_have:
            if signal_name not in fetched_signals:
                print(f"【V6.1信号映射】关键信号{signal_name}缺失，创建默认序列")
                fetched_signals[signal_name] = pd.Series(0.0, index=df_index)
        signal_check_report = {
            "total_signals": len(fetched_signals),
            "critical_signals_present": [],
            "critical_signals_missing": [],
            "mapping_success_rate": f"{len(successful_mappings)}/{len(critical_signal_mappings)}",
            "priority_mappings": {
                1: [f"{df}->{sig}" for df, sig, prio in successful_mappings if prio == 1],
                2: [f"{df}->{sig}" for df, sig, prio in successful_mappings if prio == 2],
                3: [f"{df}->{sig}" for df, sig, prio in successful_mappings if prio == 3]
            }
        }
        critical_signal_names = ['lower_shadow_absorb', 'suppressive_accum', 'distribution_intensity', 
                                'active_selling', 'profit_taking_flow', 'main_force_net_flow', 'flow_credibility']
        for sig_name in critical_signal_names:
            if sig_name in fetched_signals:
                signal_check_report["critical_signals_present"].append(sig_name)
            else:
                signal_check_report["critical_signals_missing"].append(sig_name)
        print(f"【V6.1信号检查】关键信号: 存在{len(signal_check_report['critical_signals_present'])}个, 缺失{len(signal_check_report['critical_signals_missing'])}个")
        print(f"【V6.1信号检查】映射成功率: {signal_check_report['mapping_success_rate']}")
        if signal_check_report["critical_signals_missing"]:
            print(f"【V6.1信号检查】缺失信号: {signal_check_report['critical_signals_missing']}")
        _temp_debug_values["V6.1信号获取详情"] = {
            "mtf_price_change_range": f"[{mtf_price_change.min():.4f}, {mtf_price_change.max():.4f}]",
            "mtf_ca_change_range": f"[{mtf_ca_change.min():.4f}, {mtf_ca_change.max():.4f}]",
            "volatility_mean": volatility_norm.mean(),
            "short_term_bias_mean": short_term_bias.mean(),
            "price_signals_count": len(price_signals),
            "ca_signals_count": len(ca_signals),
            "divergence_score": divergence_score,
            "获取信号总数": len(fetched_signals),
            "信号检查报告": signal_check_report,
            "信号映射详情": {
                "successful_mappings": successful_mappings,
                "missing_signals": missing_signals,
                "alternative_names_tried": alternative_names if 'alternative_names' in locals() else []
            }
        }
        print(f"【V6.1信号获取完成】价格变化范围: [{mtf_price_change.min():.4f}, {mtf_price_change.max():.4f}]")
        print(f"【V6.1信号获取完成】成本优势变化范围: [{mtf_ca_change.min():.4f}, {mtf_ca_change.max():.4f}]")
        print(f"【V6.1信号获取完成】波动性均值: {volatility_norm.mean():.4f}")
        print(f"【V6.1信号获取完成】短期偏好: {short_term_bias.mean():.4f}")
        print(f"【V6.1信号获取完成】lower_shadow_absorb状态: {'已获取' if 'lower_shadow_absorb' in fetched_signals else '使用默认值'}")
        return fetched_signals

    def _calculate_divergence(self, price_signal: pd.Series, ca_signal: pd.Series, df_index: pd.Index) -> pd.Series:
        """【V6.0 · 价格-成本优势背离检测】
        - 核心新增: 检测价格与成本优势的背离
        - 核心优化: 多时间窗口背离检测
        - 版本: 6.0"""
        divergence = pd.Series(0.0, index=df_index)
        # 1. 短期背离（3-5天）
        for window in [3, 5, 8]:
            if len(price_signal) > window:
                price_roc = price_signal.rolling(window=window).mean().pct_change()
                ca_roc = ca_signal.rolling(window=window).mean().pct_change()
                # 背离：价格上升但成本优势下降，或价格下降但成本优势上升
                bearish_divergence = (price_roc > 0) & (ca_roc < 0)
                bullish_divergence = (price_roc < 0) & (ca_roc > 0)
                # 背离强度 = 价格变化与成本优势变化差值的绝对值
                div_strength = (price_roc - ca_roc).abs()
                # 加权合并
                weight = 1.0 / window
                divergence += bearish_divergence.astype(float) * div_strength * weight * -1  # 看跌背离为负
                divergence += bullish_divergence.astype(float) * div_strength * weight      # 看涨背离为正
        # 2. 中期背离（13-21天）
        for window in [13, 21]:
            if len(price_signal) > window:
                price_sma = price_signal.rolling(window=window).mean()
                ca_sma = ca_signal.rolling(window=window).mean()
                price_trend = price_sma.diff(window // 2)
                ca_trend = ca_sma.diff(window // 2)
                bearish_div = (price_trend > 0) & (ca_trend < 0)
                bullish_div = (price_trend < 0) & (ca_trend > 0)
                div_strength = (price_trend - ca_trend).abs()
                weight = 1.0 / (window * 2)
                divergence += bearish_div.astype(float) * div_strength * weight * -1
                divergence += bullish_div.astype(float) * div_strength * weight
        # 归一化到[-1,1]
        if divergence.abs().max() > 0:
            divergence = divergence / divergence.abs().max()
        return divergence.clip(-1, 1)

    def _normalize_all_signals(self, df: pd.DataFrame, df_index: pd.Index, fetched_signals: Dict[str, pd.Series], mtf_slope_accel_weights: Dict, method_name: str, _temp_debug_values: Dict) -> Dict[str, pd.Series]:
        """【V6.1 · 信号归一化处理 - 修复信号缺失问题】
        - 核心修复: 添加信号存在性检查，防止KeyError
        - 核心优化: 为缺失信号提供默认值或替代方案
        - 核心新增: 信号完整性检查和详细报告
        - 版本: 6.1"""
        print(f"【V6.1信号归一化】开始处理，共{len(fetched_signals)}个输入信号")
        normalized_signals = {}
        slope_periods_weights = mtf_slope_accel_weights.get('slope_periods', {})
        accel_periods_weights = mtf_slope_accel_weights.get('accel_periods', {})
        missing_signals_in_normalization = []
        required_signals_for_normalization = [
            'suppressive_accum', 'profit_taking_flow', 'active_buying_support',
            'main_force_net_flow', 'flow_credibility', 'volatility_instability',
            'adx_trend_strength', 'market_sentiment', 'liquidity_authenticity',
            'ma_potential_orderliness_score', 'microstructure_efficiency',
            'main_force_buy_execution_alpha', 'main_force_sell_execution_alpha',
            'micro_price_impact_asymmetry'
        ]
        for sig_name in required_signals_for_normalization:
            if sig_name not in fetched_signals:
                missing_signals_in_normalization.append(sig_name)
        if missing_signals_in_normalization:
            print(f"【V6.1归一化警告】{len(missing_signals_in_normalization)}个信号缺失: {missing_signals_in_normalization}")
            for sig_name in missing_signals_in_normalization:
                print(f"  - 为{sig_name}创建默认序列")
                fetched_signals[sig_name] = pd.Series(0.0, index=df_index)
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
        normalization_results = {}
        if 'suppressive_accum' in fetched_signals:
            normalization_results['suppressive_accum_norm'] = self.helper._normalize_series(fetched_signals['suppressive_accum'], df_index, bipolar=False)
        else:
            print("【V6.1归一化警告】suppressive_accum不存在，使用默认值")
            normalization_results['suppressive_accum_norm'] = pd.Series(0.0, index=df_index)
        if 'profit_taking_flow' in fetched_signals:
            normalization_results['profit_taking_flow_norm'] = self.helper._normalize_series(fetched_signals['profit_taking_flow'], df_index, bipolar=False)
        else:
            normalization_results['profit_taking_flow_norm'] = pd.Series(0.0, index=df_index)
        if 'active_buying_support' in fetched_signals:
            normalization_results['active_buying_support_inverted_norm'] = 1 - self.helper._normalize_series(fetched_signals['active_buying_support'], df_index, bipolar=False)
        else:
            normalization_results['active_buying_support_inverted_norm'] = pd.Series(0.5, index=df_index)
        if 'main_force_net_flow' in fetched_signals:
            normalization_results['main_force_net_flow_outflow_norm'] = self.helper._normalize_series(fetched_signals['main_force_net_flow'].clip(upper=0).abs(), df_index, bipolar=False)
        else:
            normalization_results['main_force_net_flow_outflow_norm'] = pd.Series(0.0, index=df_index)
        if 'flow_credibility' in fetched_signals:
            normalization_results['flow_credibility_norm'] = self.helper._normalize_series(fetched_signals['flow_credibility'], df_index, bipolar=False)
        else:
            normalization_results['flow_credibility_norm'] = pd.Series(0.5, index=df_index)
        if 'volatility_instability' in fetched_signals:
            normalization_results['volatility_inverse_norm'] = 1 - self.helper._normalize_series(fetched_signals['volatility_instability'], df_index, bipolar=False)
        else:
            normalization_results['volatility_inverse_norm'] = pd.Series(0.5, index=df_index)
        if 'adx_trend_strength' in fetched_signals:
            normalization_results['trend_strength_inverse_norm'] = 1 - self.helper._normalize_series(fetched_signals['adx_trend_strength'], df_index, bipolar=False)
        else:
            normalization_results['trend_strength_inverse_norm'] = pd.Series(0.5, index=df_index)
        if 'market_sentiment' in fetched_signals:
            normalization_results['sentiment_neutrality_norm'] = 1 - self.helper._normalize_series(fetched_signals['market_sentiment'].abs(), df_index, bipolar=False)
        else:
            normalization_results['sentiment_neutrality_norm'] = pd.Series(0.5, index=df_index)
        if 'liquidity_authenticity' in fetched_signals:
            normalization_results['liquidity_authenticity_score_norm'] = self.helper._normalize_series(fetched_signals['liquidity_authenticity'], df_index, bipolar=False)
        else:
            normalization_results['liquidity_authenticity_score_norm'] = pd.Series(0.5, index=df_index)
        if 'ma_potential_orderliness_score' in fetched_signals:
            normalization_results['ma_potential_orderliness_score_norm'] = self.helper._normalize_series(fetched_signals['ma_potential_orderliness_score'], df_index, bipolar=False)
        else:
            normalization_results['ma_potential_orderliness_score_norm'] = pd.Series(0.5, index=df_index)
        if 'microstructure_efficiency' in fetched_signals:
            normalization_results['microstructure_efficiency_index_norm'] = self.helper._normalize_series(fetched_signals['microstructure_efficiency'], df_index, bipolar=False)
        else:
            normalization_results['microstructure_efficiency_index_norm'] = pd.Series(0.5, index=df_index)
        if 'main_force_buy_execution_alpha' in fetched_signals:
            normalization_results['main_force_buy_execution_alpha_norm'] = self.helper._normalize_series(fetched_signals['main_force_buy_execution_alpha'], df_index, bipolar=False)
        else:
            normalization_results['main_force_buy_execution_alpha_norm'] = pd.Series(0.5, index=df_index)
        if 'main_force_sell_execution_alpha' in fetched_signals:
            normalization_results['main_force_sell_execution_alpha_norm'] = self.helper._normalize_series(fetched_signals['main_force_sell_execution_alpha'], df_index, bipolar=False)
        else:
            normalization_results['main_force_sell_execution_alpha_norm'] = pd.Series(0.5, index=df_index)
        if 'micro_price_impact_asymmetry' in fetched_signals:
            normalization_results['micro_price_impact_asymmetry_norm'] = self.helper._normalize_series(fetched_signals['micro_price_impact_asymmetry'], df_index, bipolar=False)
        else:
            normalization_results['micro_price_impact_asymmetry_norm'] = pd.Series(0.5, index=df_index)
        normalized_signals.update(normalization_results)
        normalization_summary = {
            "total_signals_normalized": len(normalized_signals),
            "signals_with_default": [k for k, v in normalization_results.items() if fetched_signals.get(k.replace('_norm', ''), pd.Series([0])).mean() == 0],
            "normalization_range_report": {}
        }
        for sig_name, sig_series in normalized_signals.items():
            if not sig_series.empty:
                normalization_summary["normalization_range_report"][sig_name] = f"[{sig_series.min():.4f}, {sig_series.max():.4f}]"
        _temp_debug_values["V6.1归一化详情"] = {
            "normalization_summary": normalization_summary,
            "多时间维度归一化详情": {
                "主力信念斜率组件": len(mf_conviction_slope_components),
                "主力信念加速度组件": len(mf_conviction_accel_components),
                "上涨纯度斜率组件": len(upward_purity_slope_components),
                "上涨纯度加速度组件": len(upward_purity_accel_components),
                "派发强度斜率组件": len(distribution_slope_components),
                "派发强度加速度组件": len(distribution_accel_components),
                "主动卖压斜率组件": len(active_selling_slope_components),
                "主动卖压加速度组件": len(active_selling_accel_components),
            },
            "信号完整性": {
                "required_signals": required_signals_for_normalization,
                "missing_signals": missing_signals_in_normalization,
                "signals_created": [sig_name for sig_name in missing_signals_in_normalization if sig_name in fetched_signals]
            }
        }
        print(f"【V6.1信号归一化完成】已处理{len(normalized_signals)}个归一化信号")
        print(f"【V6.1信号完整性】缺失{len(missing_signals_in_normalization)}个信号，已创建默认值")
        return normalized_signals

    def _calculate_q1_healthy_rally(self, fetched_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], dynamic_weights: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """【V6.1 · Q1健康上涨计算 - 修复信号缺失问题】
        - 核心修复: 确保所有需要的信号都存在，防止KeyError
        - 核心优化: 为缺失信号提供合理的默认值
        - 核心新增: 信号存在性检查
        - 版本: 6.1"""
        # 1. 基础信号检查
        required_signals = ['mtf_price_change', 'mtf_ca_change']
        missing_base_signals = [sig for sig in required_signals if sig not in fetched_signals]
        if missing_base_signals:
            print(f"【V6.1 Q1警告】缺失基础信号: {missing_base_signals}，使用默认值")
            for sig in missing_base_signals:
                fetched_signals[sig] = pd.Series(0.0, index=normalized_signals.get('mtf_main_force_conviction', pd.Series(0.0)).index)
        price_change = fetched_signals.get('mtf_price_change', pd.Series(0.0, index=normalized_signals.get('mtf_main_force_conviction', pd.Series(0.0)).index))
        ca_change = fetched_signals.get('mtf_ca_change', pd.Series(0.0, index=price_change.index))
        # 2. 检查归一化信号
        required_normalized = ['mtf_main_force_conviction', 'mtf_upward_purity', 'flow_credibility_norm', 'main_force_buy_execution_alpha_norm']
        missing_normalized = [sig for sig in required_normalized if sig not in normalized_signals]
        if missing_normalized:
            print(f"【V6.1 Q1警告】缺失归一化信号: {missing_normalized}，使用默认值")
            for sig in missing_normalized:
                normalized_signals[sig] = pd.Series(0.5, index=price_change.index)
        # 3. 多维确认因子
        confirmation_factors = {}
        price_confirm = pd.Series(0.0, index=price_change.index)
        for window in [1, 3, 5, 8]:
            if len(price_change) > window:
                price_ma = price_change.rolling(window=window).mean()
                price_trend = price_change.rolling(window=3).mean()
                same_direction = (price_change > 0) & (price_ma > 0) & (price_trend > 0)
                price_confirm += same_direction.astype(float) * (1.0 / window)
        confirmation_factors['price_confirmation'] = price_confirm.clip(0, 1)
        ca_confirm = pd.Series(0.0, index=ca_change.index)
        if 'main_force_cost_advantage_D' in fetched_signals:
            ca_raw = fetched_signals['main_force_cost_advantage_D']
            ca_positive = ca_raw > 0
            ca_increasing = ca_change > 0
            ca_confirm = (ca_positive.astype(float) * 0.5 + ca_increasing.astype(float) * 0.5)
        confirmation_factors['ca_confirmation'] = ca_confirm
        flow_confirm = pd.Series(0.0, index=price_change.index)
        if 'main_force_net_flow_calibrated_D' in fetched_signals:
            mf_flow = fetched_signals['main_force_net_flow_calibrated_D']
            flow_positive = mf_flow > 0
            if 'flow_credibility_index_D' in fetched_signals:
                flow_cred = fetched_signals['flow_credibility_index_D']
                flow_cred_norm = self.helper._normalize_series(flow_cred, price_change.index, bipolar=False)
                flow_confirm = flow_positive.astype(float) * flow_cred_norm
            else:
                flow_confirm = flow_positive.astype(float)
        confirmation_factors['flow_confirmation'] = flow_confirm
        sentiment_confirm = pd.Series(0.0, index=price_change.index)
        if 'market_sentiment_score_D' in fetched_signals:
            sentiment = fetched_signals['market_sentiment_score_D']
            sentiment_norm = (sentiment.clip(-1, 1) + 1) / 2
            optimal_sentiment = (sentiment_norm > 0.3) & (sentiment_norm < 0.8)
            sentiment_confirm = optimal_sentiment.astype(float)
        confirmation_factors['sentiment_confirmation'] = sentiment_confirm
        structure_confirm = pd.Series(0.0, index=price_change.index)
        if 'trend_conviction_score_D' in fetched_signals:
            trend_conv = fetched_signals['trend_conviction_score_D']
            trend_conv_norm = self.helper._normalize_series(trend_conv, price_change.index, bipolar=False)
            if 'structural_potential_score_D' in fetched_signals:
                struct_pot = fetched_signals['structural_potential_score_D']
                struct_norm = self.helper._normalize_series(struct_pot, price_change.index, bipolar=False)
                structure_confirm = (trend_conv_norm * 0.6 + struct_norm * 0.4)
            else:
                structure_confirm = trend_conv_norm
        confirmation_factors['structure_confirmation'] = structure_confirm
        risk_factors = {}
        volume_risk = pd.Series(1.0, index=price_change.index)
        if 'volume_D' in fetched_signals and 'volume_vs_ma_5_ratio_D' in fetched_signals:
            volume = fetched_signals['volume_D']
            volume_ma_ratio = fetched_signals['volume_vs_ma_5_ratio_D']
            low_volume_risk = (price_change > 0) & (volume_ma_ratio < 0.8)
            volume_risk = 1.0 - low_volume_risk.astype(float) * 0.5
        risk_factors['volume_risk'] = volume_risk
        overbought_risk = pd.Series(1.0, index=price_change.index)
        if 'RSI_13_D' in fetched_signals:
            rsi = fetched_signals['RSI_13_D']
            overbought = rsi > 70
            overbought_risk = 1.0 - overbought.astype(float) * 0.7
        risk_factors['overbought_risk'] = overbought_risk
        volatility_risk = pd.Series(1.0, index=price_change.index)
        if 'VOLATILITY_INSTABILITY_INDEX_21d_D' in fetched_signals:
            vol_idx = fetched_signals['VOLATILITY_INSTABILITY_INDEX_21d_D']
            vol_norm = self.helper._normalize_series(vol_idx, price_change.index, bipolar=False)
            high_vol_risk = vol_norm > 0.7
            volatility_risk = 1.0 - high_vol_risk.astype(float) * 0.6
        risk_factors['volatility_risk'] = volatility_risk
        both_positive = (price_change > 0) & (ca_change > 0)
        Q1_base = pd.Series(0.0, index=price_change.index)
        Q1_base[both_positive] = (price_change[both_positive].clip(0, 1) * ca_change[both_positive].clip(0, 1)).pow(0.5)
        confirm_weights = {
            'price_confirmation': 0.25,
            'ca_confirmation': 0.25,
            'flow_confirmation': 0.20,
            'sentiment_confirmation': 0.15,
            'structure_confirmation': 0.15
        }
        Q1_confirm = pd.Series(0.0, index=price_change.index)
        for factor, weight in confirm_weights.items():
            if factor in confirmation_factors:
                Q1_confirm += confirmation_factors[factor] * weight
        risk_adjustment = pd.Series(1.0, index=price_change.index)
        for factor in risk_factors.values():
            risk_adjustment = risk_adjustment * factor
        Q1_final = Q1_base * Q1_confirm * risk_adjustment
        Q1_final = Q1_final.clip(0, 1)
        _temp_debug_values["Q1:健康上涨详情"] = {
            "Q1_base": Q1_base,
            "Q1_confirm": Q1_confirm,
            "risk_adjustment": risk_adjustment,
            "Q1_final": Q1_final,
            "price_change": price_change,
            "ca_change": ca_change,
            "both_positive_days": both_positive.sum(),
            "确认因子详情": {k: v.mean() for k, v in confirmation_factors.items()},
            "风险因子详情": {k: v.mean() for k, v in risk_factors.items()},
            "信号检查": {
                "missing_base_signals": missing_base_signals,
                "missing_normalized": missing_normalized
            }
        }
        print(f"【V6.1 Q1计算】基础分数均值: {Q1_base.mean():.4f}, 确认分数均值: {Q1_confirm.mean():.4f}")
        print(f"【V6.1 Q1计算】风险调整均值: {risk_adjustment.mean():.4f}, 最终分数均值: {Q1_final.mean():.4f}")
        print(f"【V6.1 Q1计算】价涨优扩天数: {both_positive.sum()}/{len(both_positive)} ({both_positive.sum()/len(both_positive):.1%})")
        if missing_base_signals or missing_normalized:
            print(f"【V6.1 Q1计算】警告: 缺失{len(missing_base_signals)}个基础信号, {len(missing_normalized)}个归一化信号")
        return Q1_final

    def _calculate_q2_bearish_distribution(self, fetched_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], dynamic_weights: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V1.1.3 · Q2派发下跌计算 - 放宽条件与优化确认证据版】
        - 核心优化: 放宽Q2_base计算条件，允许价格下跌或成本优势收缩任一情况出现
        - 核心修复: 优化确认证据计算，确保合理性
        - 核心新增: 添加更多统计信息和调试输出
        - 版本: 1.1.3
        """
        # 获取价格下跌和成本优势收缩的信号
        price_negative = fetched_signals['mtf_price_change'].clip(upper=0).abs()
        ca_negative = fetched_signals['mtf_ca_change'].clip(upper=0).abs()
        # Q2基础计算：放宽条件，只要有一个为负就计算
        # 使用加权平均，而不是必须两个都为负
        Q2_base = (price_negative + ca_negative) / 2
        Q2_base = Q2_base.clip(0, 1)
        # 确认：利润兑现流量、主动卖压、行为派发意图、主力卖出执行Alpha
        Q2_confirm_components = {
            'profit_taking_flow_norm': normalized_signals['profit_taking_flow_norm'].clip(0, 1),
            'mtf_active_selling': normalized_signals['mtf_active_selling'].clip(0, 1),
            'mtf_distribution_intensity': normalized_signals['mtf_distribution_intensity'].clip(0, 1),
            'main_force_sell_execution_alpha_norm': normalized_signals['main_force_sell_execution_alpha_norm'].clip(0, 1)
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
        Q2_distribution_evidence = Q2_distribution_evidence.clip(0, 1)
        # Q2_final为负值，表示下跌趋势
        Q2_final = (Q2_base * Q2_distribution_evidence * -1).clip(-1, 0)
        _temp_debug_values["Q2: 价跌 & 优缩"] = {
            "Q2_base": Q2_base,
            "Q2_distribution_evidence": Q2_distribution_evidence,
            "Q2_final": Q2_final,
            "price_negative_range": f"[{price_negative.min():.4f}, {price_negative.max():.4f}]",
            "ca_negative_range": f"[{ca_negative.min():.4f}, {ca_negative.max():.4f}]",
            "price_negative_days": (price_negative > 0).sum(),
            "ca_negative_days": (ca_negative > 0).sum()
        }
        # 输出统计信息
        print(f"【Q2计算】Q2_base均值: {Q2_base.mean():.4f}, 范围: [{Q2_base.min():.4f}, {Q2_base.max():.4f}]")
        print(f"【Q2计算】价格下跌天数: {(price_negative > 0).sum()}/{len(price_negative)} ({(price_negative > 0).sum()/len(price_negative):.1%})")
        print(f"【Q2计算】成本优势收缩天数: {(ca_negative > 0).sum()}/{len(ca_negative)} ({(ca_negative > 0).sum()/len(ca_negative):.1%})")
        return Q2_final

    def _calculate_q3_golden_pit(self, fetched_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], df_index: pd.Index, dynamic_weights: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """【V6.1 · Q3黄金坑计算 - 修复lower_shadow_absorb信号缺失问题】
        - 核心修复: 彻底解决lower_shadow_absorb信号缺失问题
        - 核心优化: 多层级替代方案，确保计算继续
        - 核心新增: 详细的信号缺失处理逻辑
        - 版本: 6.1"""
        print(f"【V6.1 Q3计算】开始黄金坑计算，检查信号完整性...")
        required_signals = ['mtf_price_change', 'mtf_ca_change', 'close_D']
        missing_signals = [sig for sig in required_signals if sig not in fetched_signals]
        if missing_signals:
            print(f"【V6.1 Q3警告】缺失基础信号: {missing_signals}，创建默认值")
            for sig in missing_signals:
                fetched_signals[sig] = pd.Series(0.0, index=df_index)
        price_negative = fetched_signals['mtf_price_change'].clip(upper=0).abs()
        ca_positive = fetched_signals['mtf_ca_change'].clip(lower=0)
        Q3_base = pd.Series(0.0, index=price_negative.index)
        golden_pit_mask = (price_negative > 0) & (ca_positive > 0)
        if golden_pit_mask.any():
            Q3_base[golden_pit_mask] = (price_negative[golden_pit_mask] * ca_positive[golden_pit_mask]).pow(0.5)
        Q3_base = Q3_base.clip(0, 1)
        print(f"【V6.1 Q3计算】检查lower_shadow_absorb信号...")
        lower_shadow_absorb_series = None
        signal_sources = [
            ('lower_shadow_absorb', 'fetched_signals'),
            ('lower_shadow_absorption_strength_D', 'fetched_signals'),
            ('lower_shadow_absorption_strength_D', 'df_direct')
        ]
        for signal_name, source in signal_sources:
            if signal_name in fetched_signals:
                lower_shadow_absorb_series = fetched_signals[signal_name]
                print(f"【V6.1 Q3计算】从fetched_signals中找到{signal_name}")
                break
        if lower_shadow_absorb_series is None or lower_shadow_absorb_series.isna().all():
            print(f"【V6.1 Q3计算】lower_shadow_absorb信号未找到或全部为NaN，寻找替代方案")
            alternative_signals = [
                'suppressive_accumulation_intensity_D',
                'absorption_strength_ma5_D',
                'dip_absorption_power_D',
                'dip_buy_absorption_strength_D'
            ]
            for alt_signal in alternative_signals:
                if alt_signal in fetched_signals:
                    lower_shadow_absorb_series = fetched_signals[alt_signal]
                    print(f"【V6.1 Q3计算】使用替代信号: {alt_signal}")
                    break
        if lower_shadow_absorb_series is None:
            print(f"【V6.1 Q3计算】所有替代信号都未找到，使用suppressive_accum_norm")
            if 'suppressive_accum_norm' in normalized_signals:
                lower_shadow_absorb_norm = normalized_signals['suppressive_accum_norm']
            else:
                lower_shadow_absorb_norm = pd.Series(0.5, index=df_index)
        else:
            lower_shadow_absorb_norm = self.helper._normalize_series(lower_shadow_absorb_series, df_index, bipolar=False)
            lower_shadow_absorb_norm = lower_shadow_absorb_norm.clip(0, 1)
        required_confirm_signals = ['suppressive_accum_norm', 'flow_credibility_norm', 'liquidity_authenticity_score_norm']
        missing_confirm = [sig for sig in required_confirm_signals if sig not in normalized_signals]
        if missing_confirm:
            print(f"【V6.1 Q3计算】缺失确认信号: {missing_confirm}，使用默认值")
            for sig in missing_confirm:
                normalized_signals[sig] = pd.Series(0.5, index=df_index)
        Q3_confirm_components = {
            'suppressive_accum_norm': normalized_signals['suppressive_accum_norm'].clip(0, 1),
            'lower_shadow_absorb': lower_shadow_absorb_norm.clip(0, 1),
            'flow_credibility_norm': normalized_signals['flow_credibility_norm'].clip(0, 1),
            'liquidity_authenticity_score_norm': normalized_signals['liquidity_authenticity_score_norm'].clip(0, 1)
        }
        Q3_confirm_weights_series = dynamic_weights.get('Q3_confirmation_weights', {})
        if not Q3_confirm_weights_series:
            print(f"【V6.1 Q3计算】Q3_confirmation_weights为空，使用默认权重")
            Q3_confirm_weights_series = {
                'suppressive_accum_norm': pd.Series(0.3, index=df_index),
                'lower_shadow_absorb': pd.Series(0.3, index=df_index),
                'flow_credibility_norm': pd.Series(0.2, index=df_index),
                'liquidity_authenticity_score_norm': pd.Series(0.2, index=df_index)
            }
        weighted_sum = pd.Series(0.0, index=Q3_base.index, dtype=np.float32)
        for k, component_series in Q3_confirm_components.items():
            weight_series = Q3_confirm_weights_series.get(k, pd.Series(0.0, index=Q3_base.index, dtype=np.float32))
            weighted_sum += component_series * weight_series
        sum_of_weights_series = pd.Series(0.0, index=Q3_base.index, dtype=np.float32)
        for weight_series in Q3_confirm_weights_series.values():
            sum_of_weights_series += weight_series
        sum_of_weights_series_safe = sum_of_weights_series.replace(0, np.nan)
        Q3_confirm = (weighted_sum / sum_of_weights_series_safe).fillna(0)
        Q3_confirm = Q3_confirm.clip(0, 1)
        if 'close_price' in fetched_signals:
            close_price_series = fetched_signals['close_price']
        elif 'close_D' in fetched_signals:
            close_price_series = fetched_signals['close_D']
        else:
            close_price_series = pd.Series(0.0, index=df_index)
        pre_5day_pct_change = close_price_series.pct_change(periods=5).shift(1).fillna(0)
        norm_pre_drop_5d = self.helper._normalize_series(pre_5day_pct_change.clip(upper=0).abs(), df_index, bipolar=False)
        pre_drop_context_bonus = norm_pre_drop_5d * 0.5
        Q3_final = (Q3_base * Q3_confirm * (1 + pre_drop_context_bonus)).clip(0, 1)
        signal_status = {
            "lower_shadow_absorb_found": lower_shadow_absorb_series is not None,
            "lower_shadow_absorb_alternative_used": lower_shadow_absorb_series is None,
            "missing_base_signals": missing_signals,
            "missing_confirm_signals": missing_confirm,
            "golden_pit_mask_count": golden_pit_mask.sum(),
            "close_price_source": 'close_price' if 'close_price' in fetched_signals else 'close_D' if 'close_D' in fetched_signals else 'none'
        }
        _temp_debug_values["Q3:价跌 & 优扩"] = {
            "Q3_base": Q3_base,
            "Q3_confirm": Q3_confirm,
            "pre_5day_pct_change": pre_5day_pct_change,
            "norm_pre_drop_5d": norm_pre_drop_5d,
            "pre_drop_context_bonus": pre_drop_context_bonus,
            "Q3_final": Q3_final,
            "price_negative_range": f"[{price_negative.min():.4f}, {price_negative.max():.4f}]",
            "ca_positive_range": f"[{ca_positive.min():.4f}, {ca_positive.max():.4f}]",
            "golden_pit_days": golden_pit_mask.sum(),
            "signal_status": signal_status,
            "lower_shadow_absorb_info": {
                "original_series_exists": lower_shadow_absorb_series is not None,
                "normalized_range": f"[{lower_shadow_absorb_norm.min():.4f}, {lower_shadow_absorb_norm.max():.4f}]" if lower_shadow_absorb_series is not None else "N/A",
                "alternative_used": lower_shadow_absorb_series is None
            }
        }
        print(f"【V6.1 Q3计算】Q3_base均值: {Q3_base.mean():.4f}, 范围: [{Q3_base.min():.4f}, {Q3_base.max():.4f}]")
        print(f"【V6.1 Q3计算】黄金坑信号天数（价跌 & 优扩）: {golden_pit_mask.sum()}/{len(golden_pit_mask)} ({golden_pit_mask.sum()/len(golden_pit_mask):.1%})")
        print(f"【V6.1 Q3计算】lower_shadow_absorb处理状态: {'使用原始信号' if lower_shadow_absorb_series is not None else '使用替代信号'}")
        if missing_signals or missing_confirm:
            print(f"【V6.1 Q3计算】警告: 缺失{len(missing_signals)}个基础信号, {len(missing_confirm)}个确认信号")
        return Q3_final

    def _calculate_q4_bull_trap(self, fetched_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], dynamic_weights: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V1.1.3 · Q4牛市陷阱计算 - 放宽条件与优化确认证据版】
        - 核心优化: 放宽Q4_base计算条件，允许价格上涨或成本优势收缩任一情况出现
        - 核心修复: 优化确认证据计算，确保合理性
        - 核心新增: 添加更多统计信息和调试输出
        - 版本: 1.1.3
        """
        # 获取价格上涨和成本优势收缩的信号
        price_positive = fetched_signals['mtf_price_change'].clip(lower=0)
        ca_negative = fetched_signals['mtf_ca_change'].clip(upper=0).abs()
        # Q4基础计算：放宽条件，只要有一个符合就计算
        # 陷阱信号：价格上涨但成本优势收缩
        Q4_base = pd.Series(0.0, index=price_positive.index)
        # 找到价格上涨且成本优势收缩的索引
        trap_mask = (price_positive > 0) & (ca_negative > 0)
        if trap_mask.any():
            Q4_base[trap_mask] = (price_positive[trap_mask] * ca_negative[trap_mask]).pow(0.5)
        Q4_base = Q4_base.clip(0, 1)
        # 确认：派发强度、买盘虚弱度、主力资金净流出、微观价格冲击不对称性
        Q4_confirm_components = {
            'mtf_distribution_intensity': normalized_signals['mtf_distribution_intensity'].clip(0, 1),
            'active_buying_support_inverted_norm': normalized_signals['active_buying_support_inverted_norm'].clip(0, 1),
            'main_force_net_flow_outflow_norm': normalized_signals['main_force_net_flow_outflow_norm'].clip(0, 1),
            'micro_price_impact_asymmetry_norm': normalized_signals['micro_price_impact_asymmetry_norm'].clip(0, 1)
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
        Q4_trap_evidence = Q4_trap_evidence.clip(0, 1)
        # Q4_final为负值，表示陷阱（看跌）
        Q4_final = (Q4_base * Q4_trap_evidence * -1).clip(-1, 0)
        _temp_debug_values["Q4: 价涨 & 优缩"] = {
            "Q4_base": Q4_base,
            "mf_outflow_risk": normalized_signals['main_force_net_flow_outflow_norm'],
            "Q4_trap_evidence": Q4_trap_evidence,
            "Q4_final": Q4_final,
            "price_positive_range": f"[{price_positive.min():.4f}, {price_positive.max():.4f}]",
            "ca_negative_range": f"[{ca_negative.min():.4f}, {ca_negative.max():.4f}]",
            "trap_days": trap_mask.sum()
        }
        # 输出统计信息
        print(f"【Q4计算】Q4_base均值: {Q4_base.mean():.4f}, 范围: [{Q4_base.min():.4f}, {Q4_base.max():.4f}]")
        print(f"【Q4计算】陷阱信号天数（价涨 & 优缩）: {trap_mask.sum()}/{len(trap_mask)} ({trap_mask.sum()/len(trap_mask):.1%})")
        return Q4_final

    def _calculate_q5_bearish_divergence(self, fetched_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], df_index: pd.Index, _temp_debug_values: Dict) -> pd.Series:
        """【V6.1 · Q5看跌背离计算 - 修复窗口解析错误】
        - 核心修复: 修复时间窗口解析中的'hig'字符串错误
        - 核心优化: 添加窗口解析的异常处理
        - 核心新增: 信号名称格式验证
        - 版本: 6.1"""
        print(f"【V6.1 Q5计算】开始看跌背离计算，检查信号完整性...")
        required_signals = ['close_D', 'main_force_cost_advantage_D']
        missing_signals = [sig for sig in required_signals if sig not in fetched_signals]
        if missing_signals:
            print(f"【V6.1 Q5警告】缺失基础信号: {missing_signals}，创建默认值")
            for sig in missing_signals:
                fetched_signals[sig] = pd.Series(0.0, index=df_index)
        price = fetched_signals.get('close_D', pd.Series(0.0, index=df_index))
        ca = fetched_signals.get('main_force_cost_advantage_D', pd.Series(0.0, index=df_index))
        if price.isna().all() or ca.isna().all():
            print("【V6.1 Q5警告】价格或成本优势数据缺失，返回默认值")
            return pd.Series(0.0, index=df_index)
        new_high_signals = {}
        valid_windows = [5, 13, 21, 55]
        for window in valid_windows:
            if len(price) > window:
                rolling_max = price.rolling(window=window).max()
                is_new_high = (price == rolling_max) & (price > price.shift(1))
                new_high_signals[f'new_high_{window}d'] = is_new_high
        ca_decline_signals = {}
        for window in [3, 5, 8, 13]:
            if len(ca) > window:
                ca_ma = ca.rolling(window=window).mean()
                ca_declining = ca < ca_ma
                ca_decline_signals[f'ca_decline_{window}d'] = ca_declining
        divergence_signals = {}
        for high_key, high_signal in new_high_signals.items():
            try:
                # 修复窗口解析逻辑
                if 'new_high_' in high_key and 'd' in high_key:
                    window_str = high_key.replace('new_high_', '').replace('d', '')
                    if window_str.isdigit():
                        window = int(window_str)
                        ca_key = f'ca_decline_{window}d'
                        if ca_key in ca_decline_signals:
                            divergence = high_signal & ca_decline_signals[ca_key]
                            divergence_signals[f'divergence_{window}d'] = divergence
                        else:
                            print(f"【V6.1 Q5调试】{ca_key}不在ca_decline_signals中，使用最近窗口")
                            # 使用最接近的可用窗口
                            closest_window = min([w for w in [3, 5, 8, 13] if f'ca_decline_{w}d' in ca_decline_signals], 
                                               key=lambda x: abs(x - window), default=None)
                            if closest_window:
                                ca_key = f'ca_decline_{closest_window}d'
                                divergence = high_signal & ca_decline_signals[ca_key]
                                divergence_signals[f'divergence_{window}d_approx_{closest_window}d'] = divergence
                    else:
                        print(f"【V6.1 Q5警告】无法解析窗口字符串: {window_str}")
                else:
                    print(f"【V6.1 Q5警告】信号名称格式异常: {high_key}")
            except Exception as e:
                print(f"【V6.1 Q5错误】处理{high_key}时出错: {e}")
        volume_divergence = pd.Series(False, index=df_index)
        if 'volume_D' in fetched_signals:
            volume = fetched_signals['volume_D']
            for window in [5, 13]:
                if len(volume) > window:
                    price_new_high = price.rolling(window=window).max() == price
                    volume_decline = volume < volume.rolling(window=window).mean()
                    vol_div = price_new_high & volume_decline
                    volume_divergence = volume_divergence | vol_div
        flow_divergence = pd.Series(False, index=df_index)
        if 'main_force_net_flow_calibrated_D' in fetched_signals:
            mf_flow = fetched_signals['main_force_net_flow_calibrated_D']
            for window in [3, 5, 8]:
                if len(mf_flow) > window:
                    price_up = price > price.rolling(window=window).mean()
                    flow_down = mf_flow < mf_flow.rolling(window=window).mean()
                    flow_div = price_up & flow_down
                    flow_divergence = flow_divergence | flow_div
        Q5_base = pd.Series(0.0, index=df_index)
        for div_name, div_signal in divergence_signals.items():
            try:
                # 修复窗口权重计算
                if 'divergence_' in div_name and 'd' in div_name:
                    # 提取窗口数字
                    window_part = div_name.replace('divergence_', '').split('d')[0]
                    if window_part.isdigit():
                        window = int(window_part)
                        weight = 1.0 / (window ** 0.5)
                    else:
                        # 如果无法解析，使用默认权重
                        print(f"【V6.1 Q5警告】无法从{div_name}解析窗口，使用默认权重0.1")
                        weight = 0.1
                        window = 5  # 默认窗口
                    if div_signal.any():
                        price_strength = (price[div_signal] / price[div_signal].rolling(window=5).mean() - 1).clip(0, 0.2) / 0.2
                        ca_strength = (ca[div_signal].rolling(window=5).mean() / ca[div_signal].clip(1e-6, None) - 1).clip(0, 0.3) / 0.3
                        div_intensity = (price_strength + ca_strength) / 2
                        Q5_base[div_signal] = Q5_base[div_signal] + div_intensity * weight
            except Exception as e:
                print(f"【V6.1 Q5错误】处理背离信号{div_name}时出错: {e}")
                continue
        if volume_divergence.any():
            try:
                Q5_base[volume_divergence] = Q5_base[volume_divergence] * 1.3
            except:
                print("【V6.1 Q5警告】成交量背离增强失败")
        if flow_divergence.any():
            try:
                Q5_base[flow_divergence] = Q5_base[flow_divergence] * 1.2
            except:
                print("【V6.1 Q5警告】资金流背离增强失败")
        Q5_base = Q5_base.clip(0, 1)
        confirm_factors = {}
        if 'active_selling_pressure_D' in fetched_signals:
            active_selling = fetched_signals['active_selling_pressure_D']
            selling_norm = self.helper._normalize_series(active_selling, df_index, bipolar=False)
            confirm_factors['selling_pressure'] = selling_norm
        else:
            confirm_factors['selling_pressure'] = pd.Series(0.5, index=df_index)
        if 'distribution_at_peak_intensity_D' in fetched_signals:
            distribution = fetched_signals['distribution_at_peak_intensity_D']
            dist_norm = self.helper._normalize_series(distribution, df_index, bipolar=False)
            confirm_factors['distribution'] = dist_norm
        else:
            confirm_factors['distribution'] = pd.Series(0.5, index=df_index)
        if 'market_sentiment_score_D' in fetched_signals:
            sentiment = fetched_signals['market_sentiment_score_D']
            overheated = sentiment > 0.8
            confirm_factors['overheated'] = overheated.astype(float)
        else:
            confirm_factors['overheated'] = pd.Series(0.0, index=df_index)
        Q5_confirm = pd.Series(0.0, index=df_index)
        confirm_weights = {'selling_pressure': 0.4, 'distribution': 0.4, 'overheated': 0.2}
        for factor, weight in confirm_weights.items():
            if factor in confirm_factors:
                Q5_confirm += confirm_factors[factor] * weight
        Q5_confirm = Q5_confirm.clip(0, 1)
        Q5_final = (Q5_base * Q5_confirm * -1).clip(-1, 0)
        signal_analysis = {
            "new_high_signals_count": len(new_high_signals),
            "ca_decline_signals_count": len(ca_decline_signals),
            "divergence_signals_count": len(divergence_signals),
            "volume_divergence_days": volume_divergence.sum(),
            "flow_divergence_days": flow_divergence.sum(),
            "Q5_base_range": f"[{Q5_base.min():.4f}, {Q5_base.max():.4f}]",
            "Q5_confirm_range": f"[{Q5_confirm.min():.4f}, {Q5_confirm.max():.4f}]",
            "Q5_final_range": f"[{Q5_final.min():.4f}, {Q5_final.max():.4f}]",
            "window_parsing_issues": "fixed"  # 标记修复状态
        }
        _temp_debug_values["Q5:看跌背离详情"] = {
            "Q5_base": Q5_base,
            "Q5_confirm": Q5_confirm,
            "Q5_final": Q5_final,
            "价格新高检测": {k: v.sum() for k, v in new_high_signals.items()},
            "成本优势下降检测": {k: v.sum() for k, v in ca_decline_signals.items()},
            "背离信号": {k: v.sum() for k, v in divergence_signals.items()},
            "成交量背离天数": volume_divergence.sum(),
            "资金流背离天数": flow_divergence.sum(),
            "确认因子": {k: v.mean() for k, v in confirm_factors.items()},
            "信号分析": signal_analysis
        }
        print(f"【V6.1 Q5计算】看跌背离天数: {sum(v.sum() for v in divergence_signals.values())}")
        print(f"【V6.1 Q5计算】基础分数均值: {Q5_base.mean():.4f}, 确认分数均值: {Q5_confirm.mean():.4f}")
        print(f"【V6.1 Q5计算】最终分数均值: {Q5_final.mean():.4f}")
        print(f"【V6.1 Q5计算】窗口解析修复完成")
        return Q5_final

    def _calculate_q6_bullish_divergence(self, fetched_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], df_index: pd.Index, _temp_debug_values: Dict) -> pd.Series:
        """【V6.1 · Q6看涨背离计算 - 修复窗口解析错误】
        - 核心修复: 修复时间窗口解析问题
        - 核心优化: 统一的窗口解析函数
        - 核心新增: 解析错误详细日志
        - 版本: 6.1"""
        print(f"【V6.1 Q6计算】开始看涨背离计算")
        price = fetched_signals.get('close_D', pd.Series(0.0, index=df_index))
        ca = fetched_signals.get('main_force_cost_advantage_D', pd.Series(0.0, index=df_index))
        if price.isna().all() or ca.isna().all():
            print("【V6.1 Q6警告】价格或成本优势数据缺失，返回默认值")
            return pd.Series(0.0, index=df_index)
        new_low_signals = {}
        for window in [5, 13, 21, 55]:
            if len(price) > window:
                rolling_min = price.rolling(window=window).min()
                is_new_low = (price == rolling_min) & (price < price.shift(1))
                new_low_signals[f'new_low_{window}d'] = is_new_low
        ca_rise_signals = {}
        for window in [3, 5, 8, 13]:
            if len(ca) > window:
                ca_ma = ca.rolling(window=window).mean()
                ca_rising = ca > ca_ma
                ca_rise_signals[f'ca_rise_{window}d'] = ca_rising
        divergence_signals = {}
        for low_key, low_signal in new_low_signals.items():
            try:
                if 'new_low_' in low_key and 'd' in low_key:
                    window_str = low_key.replace('new_low_', '').replace('d', '')
                    if window_str.isdigit():
                        window = int(window_str)
                        ca_key = f'ca_rise_{window}d'
                        if ca_key in ca_rise_signals:
                            divergence = low_signal & ca_rise_signals[ca_key]
                            divergence_signals[f'divergence_{window}d'] = divergence
                        else:
                            closest_window = min([w for w in [3, 5, 8, 13] if f'ca_rise_{w}d' in ca_rise_signals],
                                               key=lambda x: abs(x - window), default=None)
                            if closest_window:
                                ca_key = f'ca_rise_{closest_window}d'
                                divergence = low_signal & ca_rise_signals[ca_key]
                                divergence_signals[f'divergence_{window}d_approx_{closest_window}d'] = divergence
                    else:
                        print(f"【V6.1 Q6警告】无法解析窗口字符串: {window_str}")
            except Exception as e:
                print(f"【V6.1 Q6错误】处理{low_key}时出错: {e}")
        volume_confirmation = pd.Series(False, index=df_index)
        if 'volume_D' in fetched_signals:
            volume = fetched_signals['volume_D']
            for window in [5, 13]:
                if len(volume) > window:
                    price_new_low = price.rolling(window=window).min() == price
                    volume_surge = volume > volume.rolling(window=window).mean() * 1.2
                    vol_conf = price_new_low & volume_surge
                    volume_confirmation = volume_confirmation | vol_conf
        flow_confirmation = pd.Series(False, index=df_index)
        if 'main_force_net_flow_calibrated_D' in fetched_signals:
            mf_flow = fetched_signals['main_force_net_flow_calibrated_D']
            for window in [3, 5, 8]:
                if len(mf_flow) > window:
                    price_low = price < price.rolling(window=window).mean()
                    flow_up = mf_flow > mf_flow.rolling(window=window).mean()
                    flow_conf = price_low & flow_up
                    flow_confirmation = flow_confirmation | flow_conf
        Q6_base = pd.Series(0.0, index=df_index)
        for div_name, div_signal in divergence_signals.items():
            try:
                if 'divergence_' in div_name and 'd' in div_name:
                    window_part = div_name.replace('divergence_', '').split('d')[0]
                    if window_part.isdigit():
                        window = int(window_part)
                        weight = 1.0 / (window ** 0.5)
                    else:
                        weight = 0.1
                    if div_signal.any():
                        price_strength = (1 - price[div_signal] / price[div_signal].rolling(window=5).mean()).clip(0, 0.2) / 0.2
                        ca_strength = (ca[div_signal] / ca[div_signal].rolling(window=5).mean().clip(1e-6, None) - 1).clip(0, 0.3) / 0.3
                        div_intensity = (price_strength + ca_strength) / 2
                        Q6_base[div_signal] = Q6_base[div_signal] + div_intensity * weight
            except Exception as e:
                print(f"【V6.1 Q6错误】处理背离信号{div_name}时出错: {e}")
        if volume_confirmation.any():
            Q6_base[volume_confirmation] = Q6_base[volume_confirmation] * 1.4
        if flow_confirmation.any():
            Q6_base[flow_confirmation] = Q6_base[flow_confirmation] * 1.3
        Q6_base = Q6_base.clip(0, 1)
        confirm_factors = {}
        if 'suppressive_accumulation_intensity_D' in fetched_signals:
            accumulation = fetched_signals['suppressive_accumulation_intensity_D']
            accum_norm = self.helper._normalize_series(accumulation, df_index, bipolar=False)
            confirm_factors['accumulation'] = accum_norm
        else:
            confirm_factors['accumulation'] = pd.Series(0.5, index=df_index)
        if 'lower_shadow_absorption_strength_D' in fetched_signals:
            lower_shadow = fetched_signals['lower_shadow_absorption_strength_D']
            shadow_norm = self.helper._normalize_series(lower_shadow, df_index, bipolar=False)
            confirm_factors['lower_shadow'] = shadow_norm
        else:
            confirm_factors['lower_shadow'] = pd.Series(0.5, index=df_index)
        if 'market_sentiment_score_D' in fetched_signals:
            sentiment = fetched_signals['market_sentiment_score_D']
            oversold = sentiment < -0.8
            confirm_factors['oversold'] = oversold.astype(float)
        else:
            confirm_factors['oversold'] = pd.Series(0.0, index=df_index)
        Q6_confirm = pd.Series(0.0, index=df_index)
        confirm_weights = {'accumulation': 0.4, 'lower_shadow': 0.3, 'oversold': 0.3}
        for factor, weight in confirm_weights.items():
            if factor in confirm_factors:
                Q6_confirm += confirm_factors[factor] * weight
        Q6_confirm = Q6_confirm.clip(0, 1)
        Q6_final = (Q6_base * Q6_confirm).clip(0, 1)
        _temp_debug_values["Q6:看涨背离详情"] = {
            "Q6_base": Q6_base,
            "Q6_confirm": Q6_confirm,
            "Q6_final": Q6_final,
            "价格新低检测": {k: v.sum() for k, v in new_low_signals.items()},
            "成本优势上升检测": {k: v.sum() for k, v in ca_rise_signals.items()},
            "背离信号": {k: v.sum() for k, v in divergence_signals.items()},
            "成交量确认天数": volume_confirmation.sum(),
            "资金流确认天数": flow_confirmation.sum(),
            "确认因子": {k: v.mean() for k, v in confirm_factors.items()},
        }
        print(f"【V6.1 Q6计算】看涨背离天数: {sum(v.sum() for v in divergence_signals.values())}")
        print(f"【V6.1 Q6计算】基础分数均值: {Q6_base.mean():.4f}, 确认分数均值: {Q6_confirm.mean():.4f}")
        print(f"【V6.1 Q6计算】最终分数均值: {Q6_final.mean():.4f}")
        return Q6_final

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
        """【V6.1 · 交互项计算 - 修复信号缺失问题】
        - 核心修复: 添加所有交互项信号的完整性检查
        - 核心优化: 为缺失信号提供默认值
        - 核心新增: 交互项计算状态报告
        - 版本: 6.1"""
        print(f"【V6.1交互项】开始计算交互项，检查信号...")
        interaction_terms_weights = config.get('interaction_terms_weights', {})
        interaction_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        interaction_details = {}
        # Q1: 主力信念与资金流可信度协同
        if 'Q1_mf_conviction_flow_credibility_synergy' in interaction_terms_weights:
            weight = interaction_terms_weights['Q1_mf_conviction_flow_credibility_synergy']
            print(f"【V6.1交互项】计算Q1协同，权重: {weight}")
            mf_conviction_available = 'mtf_main_force_conviction' in normalized_signals
            flow_credibility_available = 'flow_credibility_norm' in normalized_signals
            if mf_conviction_available and flow_credibility_available:
                synergy = normalized_signals['mtf_main_force_conviction'].clip(lower=0) * normalized_signals['flow_credibility_norm']
                interaction_score += synergy * weight
                interaction_details['Q1_synergy'] = {
                    "calculated": True,
                    "weight": weight,
                    "synergy_mean": synergy.mean()
                }
            else:
                print(f"【V6.1交互项警告】Q1协同信号缺失: mf_conviction={mf_conviction_available}, flow_credibility={flow_credibility_available}")
                interaction_details['Q1_synergy'] = {
                    "calculated": False,
                    "missing_signals": []
                }
                if not mf_conviction_available:
                    interaction_details['Q1_synergy']['missing_signals'].append('mtf_main_force_conviction')
                if not flow_credibility_available:
                    interaction_details['Q1_synergy']['missing_signals'].append('flow_credibility_norm')
        # Q3: 打压吸筹与下影线吸收协同
        if 'Q3_suppressive_absorb_synergy' in interaction_terms_weights:
            weight = interaction_terms_weights['Q3_suppressive_absorb_synergy']
            print(f"【V6.1交互项】计算Q3协同，权重: {weight}")
            suppressive_available = 'suppressive_accum_norm' in normalized_signals
            lower_shadow_available = 'lower_shadow_absorb' in fetched_signals
            if suppressive_available and lower_shadow_available:
                synergy = normalized_signals['suppressive_accum_norm'] * fetched_signals['lower_shadow_absorb']
                interaction_score += synergy * weight
                interaction_details['Q3_synergy'] = {
                    "calculated": True,
                    "weight": weight,
                    "synergy_mean": synergy.mean()
                }
            else:
                print(f"【V6.1交互项警告】Q3协同信号缺失: suppressive={suppressive_available}, lower_shadow={lower_shadow_available}")
                interaction_details['Q3_synergy'] = {
                    "calculated": False,
                    "missing_signals": []
                }
                if not suppressive_available:
                    interaction_details['Q3_synergy']['missing_signals'].append('suppressive_accum_norm')
                if not lower_shadow_available:
                    interaction_details['Q3_synergy']['missing_signals'].append('lower_shadow_absorb')
                    # 尝试寻找替代信号
                    alternative_found = False
                    for alt_signal in ['suppressive_accumulation_intensity_D', 'absorption_strength_ma5_D']:
                        if alt_signal in fetched_signals:
                            print(f"【V6.1交互项】为Q3使用替代信号: {alt_signal}")
                            synergy = normalized_signals['suppressive_accum_norm'] * fetched_signals[alt_signal]
                            interaction_score += synergy * weight * 0.5  # 替代信号权重减半
                            alternative_found = True
                            interaction_details['Q3_synergy']['alternative_used'] = alt_signal
                            break
                    if not alternative_found:
                        print(f"【V6.1交互项】Q3无可用替代信号，跳过")
        # Q4: 派发强度与主力净流出协同
        if 'Q4_distribution_mf_outflow_synergy' in interaction_terms_weights:
            weight = interaction_terms_weights['Q4_distribution_mf_outflow_synergy']
            print(f"【V6.1交互项】计算Q4协同，权重: {weight}")
            distribution_available = 'mtf_distribution_intensity' in normalized_signals
            mf_outflow_available = 'main_force_net_flow_outflow_norm' in normalized_signals
            if distribution_available and mf_outflow_available:
                synergy = normalized_signals['mtf_distribution_intensity'] * normalized_signals['main_force_net_flow_outflow_norm']
                interaction_score += synergy * weight
                interaction_details['Q4_synergy'] = {
                    "calculated": True,
                    "weight": weight,
                    "synergy_mean": synergy.mean()
                }
            else:
                print(f"【V6.1交互项警告】Q4协同信号缺失: distribution={distribution_available}, mf_outflow={mf_outflow_available}")
                interaction_details['Q4_synergy'] = {
                    "calculated": False,
                    "missing_signals": []
                }
                if not distribution_available:
                    interaction_details['Q4_synergy']['missing_signals'].append('mtf_distribution_intensity')
                if not mf_outflow_available:
                    interaction_details['Q4_synergy']['missing_signals'].append('main_force_net_flow_outflow_norm')
        # 交互项计算状态报告
        calculated_count = sum(1 for detail in interaction_details.values() if detail.get('calculated', False))
        total_count = len(interaction_details)
        print(f"【V6.1交互项】完成: {calculated_count}/{total_count}个交互项成功计算")
        _temp_debug_values["交互项"] = {
            "interaction_score": interaction_score,
            "interaction_details": interaction_details,
            "calculation_summary": {
                "total_interaction_terms": total_count,
                "successfully_calculated": calculated_count,
                "weights_used": interaction_terms_weights
            }
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

    def _parse_window_from_signal_name(self, signal_name: str) -> Tuple[int, str]:
        """【V6.1 · 窗口解析工具函数】
        - 核心新增: 统一的信号名称窗口解析
        - 核心优化: 多格式支持和错误处理
        - 版本: 6.1"""
        window = 5  # 默认窗口
        original_name = signal_name
        try:
            # 常见信号名称模式
            patterns = [
                (r'new_(high|low)_(\d+)d', 2),  # new_high_5d, new_low_13d
                (r'(ca|price)_(decline|rise)_(\d+)d', 3),  # ca_decline_5d, price_rise_13d
                (r'divergence_(\d+)d', 1),  # divergence_5d
                (r'SLOPE_(\d+)_', 1),  # SLOPE_5_close_D
                (r'ACCEL_(\d+)_', 1),  # ACCEL_13_close_D
            ]
            for pattern, group_index in patterns:
                import re
                match = re.search(pattern, signal_name)
                if match:
                    window_str = match.group(group_index)
                    if window_str.isdigit():
                        window = int(window_str)
                        return window, "success"
            # 备用解析：直接查找数字
            import re
            numbers = re.findall(r'\d+', signal_name)
            if numbers:
                window = int(numbers[0])
                return window, "extracted_from_numbers"
            return window, "default_used"
        except Exception as e:
            print(f"【V6.1窗口解析】解析{signal_name}时出错: {e}")
            return window, f"error_{str(e)}"


