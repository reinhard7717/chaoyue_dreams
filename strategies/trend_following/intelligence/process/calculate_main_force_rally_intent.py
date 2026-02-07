# 文件: strategies/trend_following/intelligence/process/calculate_main_force_rally_intent.py
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

class CalculateMainForceRallyIntent:
    """
    PROCESS_META_MAIN_FORCE_RALLY_INTENT
    【V2.0 · 全面重构版】基于最终军械库清单计算"主力拉升意图"
    核心特点：
    1. 完全使用新清单数据，摒弃所有旧信号
    2. 重新设计攻击性、控制力、障碍清除三大维度
    3. 简化信号处理流程，提升计算效率
    4. 增强探针功能，详细输出每个计算步骤
    """
    def __init__(self, strategy_instance, process_intelligence_helper_instance: ProcessIntelligenceHelper):
        self.strategy = strategy_instance
        self.helper = process_intelligence_helper_instance
        self.debug_params = self.helper.debug_params
        self.probe_dates = self.helper.probe_dates
        self._probe_output = []  # 探针输出缓冲区

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V8.0 · 逻辑精简版】主计算流程：移除缺失的Flag列映射，确保策略启动的兼容性
        """
        self._probe_output = []
        params = self._get_parameters(config)
        # 触发自检，此时已不包含缺失的三个Flag列
        if not self._check_data_integrity(df):
            self._probe_print("警告: 发现基础数据列不完整，将尝试使用安全值填充。")
        raw_signals = self._get_raw_signals(df)
        is_limit_up_day = df.apply(lambda row: is_limit_up(row), axis=1)
        df_index = df.index
        normalized_signals = self._normalize_raw_signals(df_index, raw_signals)
        mtf_signals = self._calculate_mtf_fused_signals(df, raw_signals, params['mtf_slope_accel_weights'], df_index)
        historical_context = self._calculate_historical_context(df, df_index, raw_signals, params['historical_context_params'])
        proxy_signals = self._construct_proxy_signals(df_index, mtf_signals, normalized_signals, config)
        dynamic_weights = self._calculate_dynamic_weights(df_index, normalized_signals, proxy_signals, mtf_signals)
        aggressiveness_score = self._calculate_aggressiveness_score(df_index, mtf_signals, normalized_signals, dynamic_weights)
        control_score = self._calculate_control_score(df_index, mtf_signals, normalized_signals, historical_context)
        obstacle_clearance_score = self._calculate_obstacle_clearance_score(df_index, mtf_signals, normalized_signals)
        bullish_intent = self._synthesize_bullish_intent(
            df_index, aggressiveness_score, control_score, obstacle_clearance_score,
            mtf_signals, normalized_signals, dynamic_weights, historical_context,
            params['rally_intent_synthesis_params']
        )
        bearish_score = self._calculate_bearish_intent(df_index, raw_signals, mtf_signals, normalized_signals, historical_context)
        total_risk_penalty = self._adjudicate_risk(df_index, raw_signals, mtf_signals, normalized_signals, dynamic_weights, aggressiveness_score, params['rally_intent_synthesis_params'])
        penalized_bullish_part = bullish_intent * (1 - total_risk_penalty)
        final_rally_intent = (penalized_bullish_part + bearish_score).clip(-1, 1)
        final_rally_intent = self._apply_contextual_modulators(df_index, final_rally_intent, proxy_signals, mtf_signals)
        final_rally_intent = final_rally_intent.mask(is_limit_up_day & (total_risk_penalty > 0.5), final_rally_intent * (1 - total_risk_penalty))
        final_rally_intent = final_rally_intent.mask(is_limit_up_day & (final_rally_intent < 0), 0.0)
        self.strategy.atomic_states["_DEBUG_rally_aggressiveness"] = aggressiveness_score
        self.strategy.atomic_states["_DEBUG_rally_control"] = control_score
        self.strategy.atomic_states["_DEBUG_rally_obstacle_clearance"] = obstacle_clearance_score
        self.strategy.atomic_states["_DEBUG_rally_bullish_intent"] = bullish_intent
        self.strategy.atomic_states["_DEBUG_rally_bearish_score"] = bearish_score
        self.strategy.atomic_states["_DEBUG_rally_total_risk_penalty"] = total_risk_penalty
        if self._is_probe_enabled(df):
            self._output_probe_info(df_index, final_rally_intent)
        return final_rally_intent.astype(np.float32)

    def _probe_print(self, message: str):
        """
        【V2.0】探针打印方法
        """
        self._probe_output.append(message)
        if get_param_value(self.debug_params.get('enabled'), False):
            print(f"[PROBE] {message}")

    def _is_probe_enabled(self, df: pd.DataFrame) -> bool:
        """
        【V2.0】检查探针是否启用
        """
        is_debug_enabled = get_param_value(self.debug_params.get('enabled'), False)
        should_probe = get_param_value(self.debug_params.get('should_probe'), False)
        return is_debug_enabled and should_probe and self.probe_dates

    def _get_parameters(self, config: Dict) -> Dict:
        """
        【V2.0】获取所有必要的配置参数
        """
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        mtf_slope_accel_weights = config.get('mtf_slope_accel_weights', {"slope_periods": {"5": 0.6, "13": 0.4}, "accel_periods": {"5": 0.7, "13": 0.3}})
        historical_context_params = config.get('historical_context_params', {})
        rally_intent_synthesis_params = config.get('rally_intent_synthesis_params', {})
        return {
            "actual_mtf_weights": actual_mtf_weights,
            "mtf_slope_accel_weights": mtf_slope_accel_weights,
            "historical_context_params": historical_context_params,
            "rally_intent_synthesis_params": rally_intent_synthesis_params
        }

    def _get_required_column_map(self) -> Dict[str, str]:
        """
        【V8.0 · 映射清单精简版】移除数据层不提供的 'IS_BREAKOUT_D' 等Flag列
        """
        col_map = {
            'close': 'close_D', 'high': 'high_D', 'low': 'low_D', 'open': 'open_D',
            'pct_change': 'pct_change_D', 'pre_close': 'pre_close_D',
            'up_limit': 'up_limit_D', 'down_limit': 'down_limit_D',
            'absolute_change_strength': 'absolute_change_strength_D',
            'volume': 'volume_D', 'volume_ratio': 'volume_ratio_D',
            'turnover_rate': 'turnover_rate_D', 'turnover_rate_f': 'turnover_rate_f_D',
            'net_amount': 'net_amount_D', 'net_amount_rate': 'net_amount_rate_D',
            'net_amount_ratio': 'net_amount_ratio_D', 'total_net_amount_5d': 'total_net_amount_5d_D',
            'ADX': 'ADX_14_D', 'RSI': 'RSI_13_D', 'MACD': 'MACD_13_34_8_D',
            'MACDh': 'MACDh_13_34_8_D', 'MACDs': 'MACDs_13_34_8_D',
            'BIAS_5': 'BIAS_5_D', 'BIAS_13': 'BIAS_13_D', 'BIAS_21': 'BIAS_21_D',
            'BBP': 'BBP_21_2.0_D', 'BBW': 'BBW_21_2.0_D', 'ATR': 'ATR_14_D',
            'CMF': 'CMF_21_D', 'ROC': 'ROC_13_D',
            'chip_concentration_ratio': 'chip_concentration_ratio_D',
            'chip_convergence_ratio': 'chip_convergence_ratio_D',
            'chip_divergence_ratio': 'chip_divergence_ratio_D',
            'chip_entropy': 'chip_entropy_D',
            'chip_flow_direction': 'chip_flow_direction_D',
            'chip_flow_intensity': 'chip_flow_intensity_D',
            'chip_stability': 'chip_stability_D',
            'net_mf_amount': 'net_mf_amount_D', 'net_mf_vol': 'net_mf_vol_D',
            'buy_elg_amount': 'buy_elg_amount_D', 'sell_elg_amount': 'sell_elg_amount_D',
            'buy_lg_amount': 'buy_lg_amount_D', 'sell_lg_amount': 'sell_lg_amount_D',
            'flow_acceleration': 'flow_acceleration_D', 'flow_consistency': 'flow_consistency_D',
            'flow_intensity': 'flow_intensity_D', 'flow_momentum_5d': 'flow_momentum_5d_D',
            'flow_stability': 'flow_stability_D', 'inflow_persistence': 'inflow_persistence_D',
            'market_sentiment': 'market_sentiment_score_D', 'industry_breadth': 'industry_breadth_score_D',
            'industry_leader': 'industry_leader_score_D', 'industry_strength_rank': 'industry_strength_rank_D',
            'breakout_quality': 'breakout_quality_score_D', 'breakout_confidence': 'breakout_confidence_D',
            'breakout_potential': 'breakout_potential_D',
            'uptrend_strength': 'uptrend_strength_D', 'downtrend_strength': 'downtrend_strength_D',
            'trend_confirmation': 'trend_confirmation_score_D',
            'accumulation_score': 'accumulation_score_D', 'distribution_score': 'distribution_score_D',
            'behavior_accumulation': 'behavior_accumulation_D', 'behavior_distribution': 'behavior_distribution_D'
        }
        kinematic_bases = {
            'price_trend': 'close_D', 'volume_trend': 'volume_D',
            'net_amount_trend': 'net_amount_D', 'flow_intensity': 'flow_intensity_D',
            'chip_concentration': 'chip_concentration_ratio_D'
        }
        for signal_name, col_name in kinematic_bases.items():
            for p in [5, 13, 21, 55]:
                for metric in ['SLOPE', 'ACCEL', 'JERK']:
                    col_map[f'{metric}_{p}_{signal_name}'] = f'{metric}_{p}_{col_name}'
        return col_map

    def _check_data_integrity(self, df: pd.DataFrame) -> bool:
        """
        【V6.0 · 数据完整性守卫】检查DataFrame是否包含所有必需的军械库列
        """
        required_columns = set(self._get_required_column_map().values())
        existing_columns = set(df.columns)
        missing_columns = required_columns - existing_columns
        if missing_columns:
            self._probe_print(f"!!! 数据完整性异常 !!! 缺失关键列: {sorted(list(missing_columns))}")
            return False
        self._probe_print("数据完整性检查通过，所有必需列均已就绪。")
        return True

    def _get_raw_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V8.0 · 动态加载版】基于精简后的映射清单自动加载原始信号
        """
        raw_signals = {}
        method_name = "_get_raw_signals"
        column_map = self._get_required_column_map()
        for signal_name, col_name in column_map.items():
            raw_signals[signal_name] = self.helper._get_safe_series(
                df, col_name, 0.0, method_name=method_name
            )
        return raw_signals

    def _calculate_mtf_fused_signals(self, df: pd.DataFrame, raw_signals: Dict[str, pd.Series], mtf_slope_accel_weights: Dict, df_index: pd.Index) -> Dict[str, pd.Series]:
        """
        【V7.0 · 指标接驳版】移除物理计算逻辑，直接接驳数据层预计算的S/A/J指标
        """
        mtf_signals = {}
        # 1. 提取接驳信号：遍历raw_signals，将所有运动学指标（S/A/J）直接注入mtf_signals上下文
        for internal_key, series in raw_signals.items():
            if any(prefix in internal_key for prefix in ['SLOPE_', 'ACCEL_', 'JERK_']):
                mtf_signals[internal_key] = series
        # 2. 计算传统MTF融合评分
        core_fusion_signals = {
            'price_trend': 'close_D', 'volume_trend': 'volume_D',
            'net_amount_trend': 'net_amount_D', 'ADX_trend': 'ADX_14_D',
            'RSI_trend': 'RSI_13_D', 'MACD_trend': 'MACD_13_34_8_D',
            'chip_concentration_trend': 'chip_concentration_ratio_D',
            'market_sentiment_trend': 'market_sentiment_score_D',
            'breakout_quality_trend': 'breakout_quality_score_D',
            'accumulation_trend': 'accumulation_score_D'
        }
        for signal_name, column_name in core_fusion_signals.items():
            mtf_signals[f'mtf_{signal_name}'] = self.helper._get_mtf_slope_accel_score(
                df, column_name, mtf_slope_accel_weights, df_index, 
                "_calculate_mtf_fused_signals", bipolar=True
            )
        return mtf_signals

    def _calculate_historical_context(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], params: Dict) -> Dict[str, pd.Series]:
        """
        【V3.0 · 历史上下文深度重构版】
        基于幻方量化A股交易经验，深度重构历史上下文计算
        核心理念：
        1. 多维度记忆融合：价格记忆、资金记忆、筹码记忆、情绪记忆
        2. 非线性衰减权重：近期数据权重更高，但保留关键历史拐点记忆
        3. 周期自适应：根据市场状态动态调整记忆周期
        4. 相位同步检测：识别主力行为与价格趋势的相位关系
        """
        self._probe_print("=== 历史上下文计算开始 ===")
        # 参数配置
        hc_enabled = get_param_value(params.get('enabled'), True)
        if not hc_enabled:
            self._probe_print("历史上下文功能已禁用")
            return self._get_empty_context(df_index)
        # 1. 价格记忆（Price Memory） - 识别趋势强度与持续性
        price_memory = self._calculate_price_memory(df_index, raw_signals, params)
        # 2. 资金记忆（Capital Memory） - 主力资金行为的持续性分析
        capital_memory = self._calculate_capital_memory(df_index, raw_signals, params)
        # 3. 筹码记忆（Chip Memory） - 筹码结构的稳定性与演变
        chip_memory = self._calculate_chip_memory(df_index, raw_signals, params)
        # 4. 情绪记忆（Sentiment Memory） - 市场情绪的持续性特征
        sentiment_memory = self._calculate_sentiment_memory(df_index, raw_signals, params)
        # 5. 综合记忆融合（使用加权几何平均增强信号一致性）
        integrated_memory = self._fuse_integrated_memory(
            price_memory, capital_memory, chip_memory, sentiment_memory, params
        )
        # 6. 相位同步检测（Phase Synchronization Detection）
        phase_sync = self._detect_phase_synchronization(
            df_index, price_memory, capital_memory, integrated_memory
        )
        # 7. 记忆质量评估（Memory Quality Assessment）
        memory_quality = self._assess_memory_quality(
            df_index, price_memory, capital_memory, chip_memory, sentiment_memory
        )
        context = {
            "price_memory": price_memory,
            "capital_memory": capital_memory,
            "chip_memory": chip_memory,
            "sentiment_memory": sentiment_memory,
            "integrated_memory": integrated_memory,
            "phase_sync": phase_sync,
            "memory_quality": memory_quality,
            "hc_enabled": hc_enabled,
            "dynamic_memory_period": self._calculate_dynamic_period(df_index, raw_signals)
        }
        self._probe_print("=== 历史上下文计算完成 ===")
        return context

    def _calculate_price_memory(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], params: Dict) -> Dict[str, pd.Series]:
        """
        【V3.0】价格记忆深度计算
        核心理念：价格具有记忆效应，近期价格行为对当前影响更大
        数学模型：指数加权记忆衰减 + 趋势结构识别
        """
        # 参数
        memory_period = get_param_value(params.get('price_memory_period'), 34)
        decay_factor = get_param_value(params.get('price_memory_decay'), 0.94)
        # 1. 趋势强度记忆（加权指数衰减）
        uptrend_strength = raw_signals.get('uptrend_strength', pd.Series(0.0, index=df_index))
        downtrend_strength = raw_signals.get('downtrend_strength', pd.Series(0.0, index=df_index))
        net_trend_strength = uptrend_strength - downtrend_strength
        # 指数加权移动平均（EWMA）赋予近期更高权重
        trend_memory_ewma = net_trend_strength.ewm(alpha=1-decay_factor, adjust=False).mean()
        # 2. 价格动量记忆（自适应周期）
        momentum_memory = self._calculate_adaptive_momentum_memory(
            raw_signals.get('close', pd.Series(0.0, index=df_index)),
            df_index, memory_period
        )
        # 3. 波动率记忆（GARCH模型简化版）
        volatility_memory = self._calculate_volatility_memory(
            raw_signals.get('pct_change', pd.Series(0.0, index=df_index)),
            df_index, memory_period
        )
        # 4. 支撑阻力记忆（关键价格水平记忆）
        support_resistance_memory = self._calculate_support_resistance_memory(
            raw_signals, df_index, memory_period
        )
        # 综合价格记忆（使用模糊逻辑融合）
        price_memory_score = (
            trend_memory_ewma * 0.35 +
            momentum_memory * 0.25 +
            (1 - volatility_memory) * 0.20 +  # 低波动有利
            support_resistance_memory * 0.20
        ).clip(0, 1)
        return {
            "trend_memory": trend_memory_ewma,
            "momentum_memory": momentum_memory,
            "volatility_memory": volatility_memory,
            "support_resistance_memory": support_resistance_memory,
            "integrated_price_memory": price_memory_score
        }

    def _calculate_capital_memory(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], params: Dict) -> Dict[str, pd.Series]:
        """
        【V3.0】资金记忆深度计算（完整版）
        核心理念：主力资金行为具有持续性，大单资金流向是先行指标
        数学模型：资金流向量合成 + 持续性检测 + 异常检测
        """
        self._probe_print("  开始计算资金记忆...")
        # 参数
        memory_period = get_param_value(params.get('capital_memory_period'), 21)
        # 1. 多级别资金流合成（向量合成法）
        capital_vectors = []
        # 特大单净流向（权重最高）
        if 'buy_elg_amount' in raw_signals and 'sell_elg_amount' in raw_signals:
            elg_net = (raw_signals['buy_elg_amount'] - raw_signals['sell_elg_amount'])
            elg_vector = elg_net.rolling(window=5, min_periods=3).mean()  # 5日平滑
            capital_vectors.append(("elg", elg_vector, 0.35))
            self._probe_print(f"  特大单净流向计算完成，样本数: {len(elg_vector.dropna())}")
        # 大单净流向
        if 'buy_lg_amount' in raw_signals and 'sell_lg_amount' in raw_signals:
            lg_net = (raw_signals['buy_lg_amount'] - raw_signals['sell_lg_amount'])
            lg_vector = lg_net.rolling(window=3, min_periods=2).mean()  # 3日平滑
            capital_vectors.append(("lg", lg_vector, 0.30))
            self._probe_print(f"  大单净流向计算完成，样本数: {len(lg_vector.dropna())}")
        # 中单净流向
        if 'buy_md_amount' in raw_signals and 'sell_md_amount' in raw_signals:
            md_net = (raw_signals['buy_md_amount'] - raw_signals['sell_md_amount'])
            md_vector = md_net.rolling(window=2, min_periods=1).mean()  # 2日平滑
            capital_vectors.append(("md", md_vector, 0.20))
            self._probe_print(f"  中单净流向计算完成，样本数: {len(md_vector.dropna())}")
        # 小单净流向（反向指标，权重为负）
        if 'buy_sm_amount' in raw_signals and 'sell_sm_amount' in raw_signals:
            sm_net = (raw_signals['buy_sm_amount'] - raw_signals['sell_sm_amount'])
            sm_vector = -sm_net.rolling(window=1).mean()  # 当日，反向
            capital_vectors.append(("sm", sm_vector, 0.15))
            self._probe_print(f"  小单净流向计算完成（反向指标）")
        # 向量合成
        composite_capital_flow = pd.Series(0.0, index=df_index)
        for name, vector, weight in capital_vectors:
            # 归一化处理
            vector_norm = self._normalize_capital_vector(vector, df_index)
            composite_capital_flow += vector_norm * weight
            self._probe_print(f"  资金向量 '{name}' 权重: {weight:.2f}, 均值: {vector_norm.mean():.4f}")
        # 2. 资金持续性检测（Hurst指数简化版）
        self._probe_print("  计算资金持续性...")
        persistence_score = self._calculate_capital_persistence(
            composite_capital_flow, df_index, memory_period
        )
        # 3. 资金异常检测（Z-score异常检测）
        self._probe_print("  检测资金异常...")
        anomaly_score = self._detect_capital_anomaly(
            composite_capital_flow, df_index, memory_period
        )
        # 4. 资金效率记忆（资金推动价格上涨的效率）
        self._probe_print("  计算资金效率...")
        efficiency_memory = self._calculate_capital_efficiency(
            composite_capital_flow,
            raw_signals.get('pct_change', pd.Series(0.0, index=df_index)),
            df_index, memory_period
        )
        # 综合资金记忆
        capital_memory_score = (
            composite_capital_flow.clip(-1, 1) * 0.40 +
            persistence_score * 0.25 +
            (1 - anomaly_score) * 0.20 +  # 异常越低越好
            efficiency_memory * 0.15
        ).clip(-1, 1)
        self._probe_print(f"  资金记忆计算完成:")
        self._probe_print(f"    - 综合资金流均值: {composite_capital_flow.mean():.4f}")
        self._probe_print(f"    - 持续性得分均值: {persistence_score.mean():.4f}")
        self._probe_print(f"    - 异常得分均值: {anomaly_score.mean():.4f}")
        self._probe_print(f"    - 效率记忆均值: {efficiency_memory.mean():.4f}")
        self._probe_print(f"    - 综合资金记忆均值: {capital_memory_score.mean():.4f}")
        return {
            "composite_capital_flow": composite_capital_flow,
            "persistence_score": persistence_score,
            "anomaly_score": anomaly_score,
            "efficiency_memory": efficiency_memory,
            "integrated_capital_memory": capital_memory_score
        }

    def _calculate_chip_memory(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], params: Dict) -> Dict[str, pd.Series]:
        """
        【V3.0】筹码记忆深度计算
        核心理念：筹码结构演变反映市场参与者的成本分布变化
        数学模型：筹码熵变分析 + 集中度迁移 + 稳定性检测
        """
        memory_period = get_param_value(params.get('chip_memory_period'), 55)
        # 1. 筹码熵变记忆（信息熵变化反映筹码混乱度）
        entropy_memory = self._calculate_chip_entropy_memory(
            raw_signals, df_index, memory_period
        )
        # 2. 筹码集中度迁移记忆
        concentration_migration = self._calculate_concentration_migration(
            raw_signals.get('chip_concentration_ratio', pd.Series(0.0, index=df_index)),
            df_index, memory_period
        )
        # 3. 筹码稳定性记忆（马尔可夫稳定性检测）
        stability_memory = self._calculate_chip_stability_memory(
            raw_signals, df_index, memory_period
        )
        # 4. 筹码压力记忆（获利盘与套牢盘记忆）
        pressure_memory = self._calculate_chip_pressure_memory(
            raw_signals, df_index, memory_period
        )
        # 综合筹码记忆（低熵+高集中度+高稳定性+低压力=良好筹码结构）
        chip_memory_score = (
            (1 - entropy_memory) * 0.30 +  # 低熵有利
            concentration_migration * 0.25 +
            stability_memory * 0.25 +
            (1 - pressure_memory) * 0.20  # 低压力有利
        ).clip(0, 1)
        return {
            "entropy_memory": entropy_memory,
            "concentration_migration": concentration_migration,
            "stability_memory": stability_memory,
            "pressure_memory": pressure_memory,
            "integrated_chip_memory": chip_memory_score
        }

    def _calculate_sentiment_memory(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], params: Dict) -> Dict[str, pd.Series]:
        """
        【V3.0】情绪记忆深度计算
        核心理念：市场情绪具有惯性和均值回归特性
        数学模型：情绪动量 + 情绪分歧度 + 情绪极端检测
        """
        memory_period = get_param_value(params.get('sentiment_memory_period'), 13)
        # 1. 情绪动量记忆（一阶差分动量）
        sentiment_momentum = self._calculate_sentiment_momentum(
            raw_signals.get('market_sentiment', pd.Series(0.0, index=df_index)),
            df_index, memory_period
        )
        # 2. 情绪分歧度记忆（波动率反映分歧）
        sentiment_divergence = self._calculate_sentiment_divergence(
            raw_signals.get('market_sentiment', pd.Series(0.0, index=df_index)),
            df_index, memory_period
        )
        # 3. 情绪极端记忆（Z-score检测极端情绪）
        sentiment_extreme = self._detect_sentiment_extreme(
            raw_signals.get('market_sentiment', pd.Series(0.0, index=df_index)),
            df_index, memory_period
        )
        # 4. 情绪一致性记忆（多指标情绪一致性）
        sentiment_consistency = self._calculate_sentiment_consistency(
            raw_signals, df_index, memory_period
        )
        # 综合情绪记忆（适度动量+低分歧+非极端+高一致性=健康情绪）
        sentiment_memory_score = (
            sentiment_momentum.abs().clip(0, 0.5) * 0.30 +  # 适度动量
            (1 - sentiment_divergence) * 0.25 +  # 低分歧
            (1 - sentiment_extreme) * 0.25 +  # 非极端
            sentiment_consistency * 0.20
        ).clip(0, 1)
        return {
            "sentiment_momentum": sentiment_momentum,
            "sentiment_divergence": sentiment_divergence,
            "sentiment_extreme": sentiment_extreme,
            "sentiment_consistency": sentiment_consistency,
            "integrated_sentiment_memory": sentiment_memory_score
        }

    def _fuse_integrated_memory(self, price_memory: Dict, capital_memory: Dict, 
                                chip_memory: Dict, sentiment_memory: Dict, params: Dict) -> pd.Series:
        """
        【V3.0】综合记忆融合算法
        核心理念：多维度记忆的加权几何平均，强调一致性
        数学模型：加权几何平均 + 一致性检测 + 置信度加权
        """
        # 提取各维度核心记忆
        price_score = price_memory.get("integrated_price_memory", pd.Series(0.5, index=price_memory["trend_memory"].index))
        capital_score = capital_memory.get("integrated_capital_memory", pd.Series(0.0, index=capital_memory["composite_capital_flow"].index)).clip(0, 1)
        chip_score = chip_memory.get("integrated_chip_memory", pd.Series(0.5, index=chip_memory["entropy_memory"].index))
        sentiment_score = sentiment_memory.get("integrated_sentiment_memory", pd.Series(0.5, index=sentiment_memory["sentiment_momentum"].index))
        # 动态权重调整（基于市场状态）
        market_state = self._detect_market_state(price_score, capital_score, chip_score, sentiment_score)
        # 不同市场状态下的权重配置
        if market_state == "trending_up":
            weights = {"price": 0.35, "capital": 0.30, "chip": 0.20, "sentiment": 0.15}
        elif market_state == "trending_down":
            weights = {"price": 0.30, "capital": 0.35, "chip": 0.25, "sentiment": 0.10}
        elif market_state == "consolidating":
            weights = {"price": 0.25, "capital": 0.25, "chip": 0.30, "sentiment": 0.20}
        else:  # "reversing"
            weights = {"price": 0.20, "capital": 0.35, "chip": 0.25, "sentiment": 0.20}
        # 加权几何平均（增强一致性要求）
        # 防止零值
        eps = 1e-9
        price_adj = price_score.clip(eps, 1)
        capital_adj = capital_score.clip(eps, 1)
        chip_adj = chip_score.clip(eps, 1)
        sentiment_adj = sentiment_score.clip(eps, 1)
        # 几何平均：exp(sum(weight * log(score)))
        log_avg = (
            np.log(price_adj) * weights["price"] +
            np.log(capital_adj) * weights["capital"] +
            np.log(chip_adj) * weights["chip"] +
            np.log(sentiment_adj) * weights["sentiment"]
        )
        integrated_memory = np.exp(log_avg)
        # 一致性增强因子（所有维度同向时增强）
        consistency_factor = self._calculate_memory_consistency(
            price_score, capital_score, chip_score, sentiment_score
        )
        # 应用一致性增强
        integrated_memory_enhanced = integrated_memory * (1 + consistency_factor * 0.3)
        return integrated_memory_enhanced.clip(0, 1)

    def _detect_phase_synchronization(self, df_index: pd.Index, price_memory: Dict, 
                                     capital_memory: Dict, integrated_memory: pd.Series) -> pd.Series:
        """
        【V3.0】相位同步检测算法
        核心理念：检测价格趋势与资金流向的相位关系，领先滞后分析
        数学模型：Hilbert变换相位差 + 交叉相关分析
        """
        # 简化版：使用一阶差分的相关性分析
        price_trend = price_memory.get("trend_memory", pd.Series(0.0, index=df_index))
        capital_flow = capital_memory.get("composite_capital_flow", pd.Series(0.0, index=df_index))
        # 1. 一阶差分（近似导数）
        price_diff = price_trend.diff().fillna(0)
        capital_diff = capital_flow.diff().fillna(0)
        # 2. 滚动窗口相关性（21日窗口）
        window = 21
        phase_correlation = price_diff.rolling(window=window).corr(capital_diff)
        # 3. 领先滞后分析（交叉相关）
        lead_lag_score = pd.Series(0.0, index=df_index)
        for i in range(window, len(df_index)):
            if i < window:
                continue
            # 计算不同滞后期的相关性
            max_corr = 0
            best_lag = 0
            for lag in range(-5, 6):  # ±5日滞后
                if i + lag < window or i + lag >= len(df_index):
                    continue
                # 对齐序列
                price_segment = price_trend.iloc[i-window:i]
                capital_segment = capital_flow.iloc[i-window+lag:i+lag] if lag >= 0 else capital_flow.iloc[i-window:i].shift(-lag)
                # 计算相关性
                corr = price_segment.corr(capital_segment)
                if abs(corr) > abs(max_corr):
                    max_corr = corr
                    best_lag = lag
            # 资金领先为正，价格领先为负
            lead_lag_score.iloc[i] = best_lag / 5.0  # 归一化到[-1, 1]
        # 4. 相位同步综合评分
        # 高正相关+资金领先=强相位同步（看涨）
        # 高负相关+价格领先=弱相位同步（看跌或背离）
        phase_sync_score = (
            phase_correlation.fillna(0) * 0.6 +
            lead_lag_score * 0.4
        ).clip(-1, 1)
        return phase_sync_score

    def _assess_memory_quality(self, df_index: pd.Index, price_memory: Dict, 
                              capital_memory: Dict, chip_memory: Dict, 
                              sentiment_memory: Dict) -> pd.Series:
        """
        【V3.0】记忆质量评估算法
        核心理念：评估各维度记忆信号的清晰度和可靠性
        数学模型：信噪比估计 + 稳定性检测 + 一致性评估
        """
        # 1. 信噪比估计（信号强度/噪声强度）
        price_snr = self._estimate_snr(
            price_memory.get("trend_memory", pd.Series(0.0, index=df_index))
        )
        capital_snr = self._estimate_snr(
            capital_memory.get("composite_capital_flow", pd.Series(0.0, index=df_index))
        )
        chip_snr = self._estimate_snr(
            chip_memory.get("integrated_chip_memory", pd.Series(0.5, index=df_index))
        )
        sentiment_snr = self._estimate_snr(
            sentiment_memory.get("integrated_sentiment_memory", pd.Series(0.5, index=df_index))
        )
        # 2. 稳定性评分（低波动=高稳定性）
        price_stability = 1 - price_memory.get("volatility_memory", pd.Series(0.5, index=df_index))
        capital_stability = 1 - capital_memory.get("anomaly_score", pd.Series(0.5, index=df_index))
        chip_stability = chip_memory.get("stability_memory", pd.Series(0.5, index=df_index))
        sentiment_stability = 1 - sentiment_memory.get("sentiment_divergence", pd.Series(0.5, index=df_index))
        # 3. 一致性评估（各维度间相关性）
        consistency_score = self._calculate_memory_consistency(
            price_memory.get("integrated_price_memory", pd.Series(0.5, index=df_index)),
            capital_memory.get("integrated_capital_memory", pd.Series(0.5, index=df_index)).clip(0, 1),
            chip_memory.get("integrated_chip_memory", pd.Series(0.5, index=df_index)),
            sentiment_memory.get("integrated_sentiment_memory", pd.Series(0.5, index=df_index))
        )
        # 4. 综合记忆质量评分
        memory_quality_score = (
            (price_snr + capital_snr + chip_snr + sentiment_snr) / 4 * 0.4 +
            (price_stability + capital_stability + chip_stability + sentiment_stability) / 4 * 0.4 +
            consistency_score * 0.2
        ).clip(0, 1)
        return memory_quality_score

    def _calculate_adaptive_momentum_memory(self, close_series: pd.Series, df_index: pd.Index, base_period: int) -> pd.Series:
        """
        【V3.1 · 紧急修复版】自适应动量记忆计算
        修复说明：解决Pandas EWM不支持动态span导致的ValueError crash问题。
        改为使用固定周期计算基础RSI，并在结果层应用波动率调节。
        """
        # 1. 计算波动率因子 (保留原逻辑用于后处理)
        returns = close_series.pct_change().fillna(0)
        # 使用固定窗口计算波动率
        volatility = returns.rolling(window=20).std().fillna(0.01)
        # 2. 计算RSI (使用固定base_period，避免Series ambiguous错误)
        # 确保base_period为有效整数
        safe_period = max(2, int(base_period))
        gains = returns.where(returns > 0, 0)
        losses = -returns.where(returns < 0, 0)
        # 使用固定周期的指数加权移动平均
        avg_gain = gains.ewm(span=safe_period, adjust=False).mean()
        avg_loss = losses.ewm(span=safe_period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        # 归一化到[0, 1]
        rsi_norm = (rsi / 100).clip(0, 1)
        # 3. 动量加速度（二阶差分）
        momentum_accel = rsi_norm.diff().diff().fillna(0)
        # 4. 应用自适应调节 (替代原有的动态周期)
        # 逻辑：高波动率时(volatility高)，信号置信度略降；低波动率时置信度高
        # 构建调节因子：波动率越低，因子越接近1.1；波动率越高，因子越接近0.9
        adaptive_modulator = 1.0 + (0.02 - volatility).clip(-0.05, 0.05)
        # 综合动量记忆
        momentum_memory = (rsi_norm * 0.7 + (0.5 + momentum_accel * 0.5) * 0.3)
        momentum_memory = (momentum_memory * adaptive_modulator).clip(0, 1)
        return momentum_memory

    def _calculate_volatility_memory(self, returns_series: pd.Series, df_index: pd.Index, memory_period: int) -> pd.Series:
        """
        【V3.0】波动率记忆计算（GARCH简化版）
        数学模型：EWMA波动率 + 波动率聚集效应
        """
        # EWMA波动率（RiskMetrics方法）
        lambda_factor = 0.94
        squared_returns = returns_series ** 2
        # 初始化
        vol_memory = pd.Series(0.0, index=df_index)
        if len(vol_memory) > 0:
            vol_memory.iloc[0] = squared_returns.iloc[:min(30, len(squared_returns))].mean()
        # 递归计算
        for i in range(1, len(vol_memory)):
            vol_memory.iloc[i] = (lambda_factor * vol_memory.iloc[i-1] + 
                                 (1 - lambda_factor) * squared_returns.iloc[i-1])
        # 年化波动率并归一化（假设年化波动率在10%-50%之间）
        annualized_vol = np.sqrt(vol_memory * 252)
        vol_norm = ((annualized_vol - 0.1) / 0.4).clip(0, 1)
        return vol_norm

    def _calculate_capital_persistence(self, capital_flow: pd.Series, df_index: pd.Index, period: int) -> pd.Series:
        """
        【V3.0】资金持续性检测（Hurst指数简化版）
        数学模型：重标极差分析（R/S）简化版本
        """
        persistence_scores = pd.Series(0.5, index=df_index)
        for i in range(period, len(df_index)):
            if i < period:
                continue
            # 取最近period日数据
            segment = capital_flow.iloc[i-period:i]
            if len(segment) < 10:
                persistence_scores.iloc[i] = 0.5
                continue
            # 计算均值
            mean_val = segment.mean()
            # 计算累积离差
            deviations = segment - mean_val
            cumulative_dev = deviations.cumsum()
            # 计算极差
            R = cumulative_dev.max() - cumulative_dev.min()
            # 计算标准差
            S = segment.std()
            # 避免除零
            if S == 0:
                persistence_scores.iloc[i] = 0.5
                continue
            # R/S比率
            rs_ratio = R / S
            # 简化Hurst指数估计
            # log(R/S) ≈ H * log(n)
            n = len(segment)
            if rs_ratio > 0 and n > 1:
                H = np.log(rs_ratio) / np.log(n)
            else:
                H = 0.5
            # H值映射到[0, 1]
            # H>0.5: 持续性，H<0.5: 反持续性，H=0.5: 随机
            persistence_score = (H - 0.3) / 0.4  # 映射到[0,1]，0.3-0.7范围
            persistence_scores.iloc[i] = persistence_score.clip(0, 1)
        return persistence_scores

    def _calculate_memory_consistency(self, *memory_series) -> pd.Series:
        """
        【V3.0】记忆一致性计算
        数学模型：多序列相关性 + 方向一致性
        """
        if len(memory_series) < 2:
            return pd.Series(1.0, index=memory_series[0].index)
        # 1. 方向一致性（符号相同比例）
        direction_consistency = pd.Series(1.0, index=memory_series[0].index)
        for i in range(len(direction_consistency)):
            signs = []
            for series in memory_series:
                if i < len(series):
                    # 计算方向（与均值的比较）
                    if i >= 10:
                        mean_val = series.iloc[max(0, i-10):i].mean()
                        sign = 1 if series.iloc[i] > mean_val else -1 if series.iloc[i] < mean_val else 0
                        signs.append(sign)
            if len(signs) >= 2:
                # 计算方向一致比例
                positive_count = sum(1 for s in signs if s > 0)
                negative_count = sum(1 for s in signs if s < 0)
                total_count = len(signs)
                if total_count > 0:
                    max_uniform = max(positive_count, negative_count)
                    direction_consistency.iloc[i] = max_uniform / total_count
        # 2. 相关性一致性（滚动窗口相关性）
        correlation_consistency = pd.Series(1.0, index=memory_series[0].index)
        if len(memory_series) >= 2:
            window = 20
            for i in range(window, len(correlation_consistency)):
                if i < window:
                    continue
                correlations = []
                # 计算所有序列对的相关性
                for j in range(len(memory_series)):
                    for k in range(j+1, len(memory_series)):
                        series1_seg = memory_series[j].iloc[i-window:i]
                        series2_seg = memory_series[k].iloc[i-window:i]
                        if len(series1_seg) >= 5 and len(series2_seg) >= 5:
                            corr = series1_seg.corr(series2_seg)
                            if not np.isnan(corr):
                                correlations.append(abs(corr))  # 取绝对值
                if correlations:
                    correlation_consistency.iloc[i] = np.mean(correlations)
        # 综合一致性
        consistency_score = (direction_consistency * 0.6 + correlation_consistency * 0.4).clip(0, 1)
        return consistency_score

    def _normalize_capital_vector(self, capital_series: pd.Series, df_index: pd.Index) -> pd.Series:
        """
        【V3.0】资金向量归一化方法
        核心理念：将原始资金流数据归一化到[-1, 1]范围，保留方向信息
        数学模型：Robust Z-score归一化，使用滚动窗口统计
        """
        if capital_series.empty:
            return pd.Series(0.0, index=df_index)
        # 使用滚动窗口统计（21日窗口）
        window = 21
        min_periods = int(window * 0.7)  # 70%的最小观测值
        # 计算滚动均值和标准差
        rolling_mean = capital_series.rolling(window=window, min_periods=min_periods).mean()
        rolling_std = capital_series.rolling(window=window, min_periods=min_periods).std()
        # 避免除零，设置最小标准差
        min_std = rolling_std.abs().quantile(0.1)  # 取10%分位数作为最小标准差
        if min_std == 0:
            min_std = 1e-9
        rolling_std = rolling_std.clip(lower=min_std)
        # Robust Z-score归一化：z = (x - mean) / std
        z_scores = (capital_series - rolling_mean) / rolling_std
        # 限制极端值（使用tanh函数将Z-score映射到[-1, 1]）
        normalized = np.tanh(z_scores * 0.5)  # 0.5为缩放因子，控制映射敏感度
        # 前向填充NaN值
        normalized = normalized.ffill().fillna(0.0)
        return normalized

    def _detect_capital_anomaly(self, capital_flow: pd.Series, df_index: pd.Index, memory_period: int) -> pd.Series:
        """
        【V3.0】资金异常检测方法
        核心理念：使用统计方法检测资金流的异常波动
        数学模型：IQR异常检测 + 历史分位数比较
        """
        anomaly_scores = pd.Series(0.0, index=df_index)
        # 使用滚动窗口检测异常
        window = min(60, memory_period * 3)  # 足够长的窗口以获得稳定统计
        for i in range(window, len(df_index)):
            if i < window:
                anomaly_scores.iloc[i] = 0.0
                continue
            # 获取历史窗口数据
            historical_data = capital_flow.iloc[i-window:i]
            # 1. IQR方法检测异常（适用于非正态分布）
            Q1 = historical_data.quantile(0.25)
            Q3 = historical_data.quantile(0.75)
            IQR = Q3 - Q1
            # 避免IQR过小
            if IQR < 1e-9:
                IQR = historical_data.std()
                if IQR < 1e-9:
                    IQR = 1e-9
            # 计算当前值与历史分布的偏差
            current_value = capital_flow.iloc[i]
            # 下界和上界
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # 2. 计算Z-score异常
            historical_mean = historical_data.mean()
            historical_std = historical_data.std()
            if historical_std < 1e-9:
                historical_std = 1e-9
            z_score = abs((current_value - historical_mean) / historical_std)
            # 3. 计算历史分位数异常
            percentile_rank = (historical_data <= current_value).sum() / len(historical_data)
            # 距离中位数的偏差（0.5为中间）
            quantile_deviation = abs(percentile_rank - 0.5) * 2  # 映射到[0, 1]
            # 4. 综合异常评分
            # 如果超出IQR边界，则异常程度较高
            is_outlier = (current_value < lower_bound) or (current_value > upper_bound)
            # 异常评分组合
            anomaly_score = (
                (1.0 if is_outlier else 0.0) * 0.4 +  # IQR边界异常权重
                min(z_score / 3.0, 1.0) * 0.4 +       # Z-score异常（3σ对应1.0）
                quantile_deviation * 0.2              # 分位数异常
            ).clip(0, 1)
            anomaly_scores.iloc[i] = anomaly_score
        # 前向填充初始值
        anomaly_scores = anomaly_scores.ffill().fillna(0.0)
        return anomaly_scores

    def _calculate_capital_efficiency(self, capital_flow: pd.Series, price_change: pd.Series, 
                                     df_index: pd.Index, memory_period: int) -> pd.Series:
        """
        【V3.0】资金效率计算方法
        核心理念：评估单位资金推动价格上涨的效率
        数学模型：资金-价格弹性系数 + 领先滞后效率
        """
        efficiency_scores = pd.Series(0.5, index=df_index)  # 默认中等效率
        # 确保数据对齐
        if len(capital_flow) != len(price_change):
            min_len = min(len(capital_flow), len(price_change))
            capital_flow = capital_flow.iloc[:min_len]
            price_change = price_change.iloc[:min_len]
        # 窗口大小
        window = min(20, memory_period)
        for i in range(window, len(df_index)):
            if i < window:
                efficiency_scores.iloc[i] = 0.5
                continue
            # 1. 计算资金-价格弹性（简单线性回归斜率）
            capital_window = capital_flow.iloc[i-window:i].values
            price_window = price_change.iloc[i-window:i].values
            # 避免零变化
            if np.std(capital_window) < 1e-9 or np.std(price_window) < 1e-9:
                efficiency_scores.iloc[i] = 0.5
                continue
            # 计算相关系数
            correlation = np.corrcoef(capital_window, price_window)[0, 1]
            if np.isnan(correlation):
                correlation = 0
            # 2. 计算单位资金推动的价格变化（弹性系数）
            # 标准化后计算斜率
            capital_normalized = (capital_window - np.mean(capital_window)) / (np.std(capital_window) + 1e-9)
            price_normalized = (price_window - np.mean(price_window)) / (np.std(price_window) + 1e-9)
            # 简单线性回归：斜率 = cov(x,y) / var(x)
            covariance = np.cov(capital_normalized, price_normalized)[0, 1]
            variance = np.var(capital_normalized)
            if variance < 1e-9:
                slope = 0
            else:
                slope = covariance / variance
            # 3. 计算领先滞后效率（资金是否领先于价格）
            # 使用交叉相关找到最大相关性的滞后
            max_corr = 0
            best_lag = 0
            for lag in range(-3, 4):  # ±3天滞后
                if lag <= 0:
                    # 资金领先（负滞后）
                    capital_shifted = capital_window[:len(capital_window)+lag] if lag < 0 else capital_window
                    price_shifted = price_window[-lag:] if lag < 0 else price_window
                else:
                    # 价格领先（正滞后）
                    capital_shifted = capital_window[lag:]
                    price_shifted = price_window[:len(price_window)-lag]
                # 确保长度一致
                min_len = min(len(capital_shifted), len(price_shifted))
                if min_len < 5:
                    continue
                capital_shifted = capital_shifted[:min_len]
                price_shifted = price_shifted[:min_len]
                corr = np.corrcoef(capital_shifted, price_shifted)[0, 1]
                if not np.isnan(corr) and abs(corr) > abs(max_corr):
                    max_corr = corr
                    best_lag = lag
            # 资金领先（负滞后）为高效，价格领先（正滞后）为低效
            lag_efficiency = max(0, -best_lag) / 3.0  # 映射到[0, 1]，资金领先最多3天
            # 4. 综合效率评分
            # 相关系数（方向一致性）
            correlation_score = (correlation + 1) / 2  # 映射到[0, 1]
            # 斜率（单位资金推动力）
            slope_score = np.tanh(slope * 0.5) * 0.5 + 0.5  # 映射到[0, 1]
            # 综合效率
            efficiency_score = (
                correlation_score * 0.4 +      # 方向一致性
                slope_score * 0.3 +            # 推动力度
                lag_efficiency * 0.3           # 领先滞后
            ).clip(0, 1)
            efficiency_scores.iloc[i] = efficiency_score
        # 前向填充
        efficiency_scores = efficiency_scores.ffill().fillna(0.5)
        return efficiency_scores

    def _calculate_chip_entropy_memory(self, raw_signals: Dict[str, pd.Series], df_index: pd.Index, memory_period: int) -> pd.Series:
        """
        【V3.0】筹码熵变记忆计算
        核心理念：使用信息熵衡量筹码分布的混乱程度
        数学模型：信息熵计算 + 熵变趋势
        """
        entropy_scores = pd.Series(0.5, index=df_index)
        # 获取筹码相关信号
        chip_signals = [
            ('chip_concentration_ratio', 0.4),
            ('chip_convergence_ratio', 0.3),
            ('chip_divergence_ratio', 0.3)
        ]
        for i in range(memory_period, len(df_index)):
            if i < memory_period:
                entropy_scores.iloc[i] = 0.5
                continue
            entropy_components = []
            for signal_name, weight in chip_signals:
                if signal_name not in raw_signals:
                    continue
                signal_series = raw_signals[signal_name]
                # 获取窗口数据
                window_data = signal_series.iloc[i-memory_period:i]
                if len(window_data) < 5:
                    continue
                # 离散化：将数据分成5个区间
                try:
                    # 使用分位数进行离散化
                    bins = np.quantile(window_data, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                    # 确保边界值唯一
                    bins = np.unique(bins)
                    if len(bins) < 2:
                        continue
                    # 计算当前值所在区间的概率分布
                    hist, _ = np.histogram(window_data, bins=bins)
                    # 转换为概率
                    prob = hist / len(window_data)
                    prob = prob[prob > 0]  # 只保留正概率
                    # 计算信息熵：H = -sum(p * log2(p))
                    if len(prob) > 0:
                        entropy = -np.sum(prob * np.log2(prob))
                        # 归一化：最大熵为log2(n)
                        max_entropy = np.log2(len(prob))
                        if max_entropy > 0:
                            normalized_entropy = entropy / max_entropy
                            entropy_components.append(normalized_entropy * weight)
                except:
                    continue
            if entropy_components:
                # 加权平均熵
                total_weight = sum(weight for _, weight in chip_signals if _ in raw_signals)
                if total_weight > 0:
                    entropy_score = sum(entropy_components) / total_weight
                    entropy_scores.iloc[i] = entropy_score
                else:
                    entropy_scores.iloc[i] = 0.5
        entropy_scores = entropy_scores.ffill().fillna(0.5)
        return entropy_scores

    def _calculate_concentration_migration(self, concentration_series: pd.Series, df_index: pd.Index, memory_period: int) -> pd.Series:
        """
        【V3.0】筹码集中度迁移记忆计算
        核心理念：跟踪筹码集中度的变化方向和速度
        数学模型：趋势斜率 + 动量分析
        """
        if concentration_series.empty:
            return pd.Series(0.5, index=df_index)
        migration_scores = pd.Series(0.5, index=df_index)
        # 使用滚动窗口计算趋势
        window = min(13, memory_period // 2)
        for i in range(window, len(df_index)):
            if i < window:
                migration_scores.iloc[i] = 0.5
                continue
            # 获取窗口数据
            window_data = concentration_series.iloc[i-window:i]
            if len(window_data) < 3:
                migration_scores.iloc[i] = 0.5
                continue
            # 1. 计算线性趋势斜率
            x = np.arange(len(window_data))
            y = window_data.values
            # 线性回归
            A = np.vstack([x, np.ones(len(x))]).T
            try:
                slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
            except:
                slope = 0
            # 2. 计算动量（近期变化）
            recent_change = window_data.iloc[-1] - window_data.iloc[0]
            momentum = recent_change / (window_data.max() - window_data.min() + 1e-9)
            # 3. 计算加速度（斜率变化）
            if i >= window * 2:
                prev_window = concentration_series.iloc[i-window*2:i-window]
                if len(prev_window) >= 3:
                    x_prev = np.arange(len(prev_window))
                    y_prev = prev_window.values
                    A_prev = np.vstack([x_prev, np.ones(len(x_prev))]).T
                    try:
                        slope_prev, _ = np.linalg.lstsq(A_prev, y_prev, rcond=None)[0]
                        acceleration = slope - slope_prev
                    except:
                        acceleration = 0
                else:
                    acceleration = 0
            else:
                acceleration = 0
            # 4. 综合迁移评分（集中度上升为正，下降为负）
            # 归一化处理
            slope_score = np.tanh(slope * 10) * 0.5 + 0.5  # 映射到[0, 1]
            momentum_score = momentum * 0.5 + 0.5  # 映射到[0, 1]
            accel_score = np.tanh(acceleration * 5) * 0.5 + 0.5  # 映射到[0, 1]
            migration_score = (
                slope_score * 0.5 +
                momentum_score * 0.3 +
                accel_score * 0.2
            )
            migration_scores.iloc[i] = migration_score.clip(0, 1)
        migration_scores = migration_scores.ffill().fillna(0.5)
        return migration_scores

    def _get_empty_context(self, df_index: pd.Index) -> Dict[str, pd.Series]:
        """
        【V3.0】返回空的历史上下文（禁用时使用）
        """
        empty_series = pd.Series(0.5, index=df_index)
        return {
            "price_memory": {
                "trend_memory": empty_series,
                "momentum_memory": empty_series,
                "volatility_memory": empty_series,
                "support_resistance_memory": empty_series,
                "integrated_price_memory": empty_series
            },
            "capital_memory": {
                "composite_capital_flow": empty_series,
                "persistence_score": empty_series,
                "anomaly_score": empty_series,
                "efficiency_memory": empty_series,
                "integrated_capital_memory": empty_series
            },
            "chip_memory": {
                "entropy_memory": empty_series,
                "concentration_migration": empty_series,
                "stability_memory": empty_series,
                "pressure_memory": empty_series,
                "integrated_chip_memory": empty_series
            },
            "sentiment_memory": {
                "sentiment_momentum": empty_series,
                "sentiment_divergence": empty_series,
                "sentiment_extreme": empty_series,
                "sentiment_consistency": empty_series,
                "integrated_sentiment_memory": empty_series
            },
            "integrated_memory": empty_series,
            "phase_sync": empty_series,
            "memory_quality": empty_series,
            "hc_enabled": False,
            "dynamic_memory_period": pd.Series(21, index=df_index)
        }

    def _normalize_raw_signals(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V2.0】归一化原始信号 - 基于新指标
        """
        normalized_signals = {}
        # 价格相关信号归一化
        price_signals_to_normalize = ['pct_change', 'absolute_change_strength']
        for signal in price_signals_to_normalize:
            normalized_signals[f'{signal}_norm'] = self.helper._normalize_series(
                raw_signals[signal], df_index, bipolar=True
            )
        # 成交量相关信号归一化
        volume_signals_to_normalize = ['volume_ratio', 'turnover_rate', 'net_amount_ratio']
        for signal in volume_signals_to_normalize:
            normalized_signals[f'{signal}_norm'] = self.helper._normalize_series(
                raw_signals[signal], df_index, bipolar=False
            )
        # 技术指标归一化
        technical_signals_to_normalize = ['ADX', 'RSI', 'MACD', 'MACDh', 'MACDs']
        for signal in technical_signals_to_normalize:
            normalized_signals[f'{signal}_norm'] = self.helper._normalize_series(
                raw_signals[signal], df_index, bipolar=True
            )
        # 筹码信号归一化
        chip_signals_to_normalize = ['chip_concentration_ratio', 'chip_convergence_ratio', 
                                    'chip_stability', 'chip_flow_intensity']
        for signal in chip_signals_to_normalize:
            normalized_signals[f'{signal}_norm'] = self.helper._normalize_series(
                raw_signals[signal], df_index, bipolar=False
            )
        # 资金流信号归一化
        flow_signals_to_normalize = ['flow_acceleration', 'flow_consistency', 'flow_intensity',
                                    'flow_momentum_5d', 'flow_stability', 'inflow_persistence']
        for signal in flow_signals_to_normalize:
            normalized_signals[f'{signal}_norm'] = self.helper._normalize_series(
                raw_signals[signal], df_index, bipolar=True
            )
        # 市场情绪归一化
        sentiment_signals_to_normalize = ['market_sentiment', 'industry_breadth', 
                                         'industry_leader', 'industry_strength_rank']
        for signal in sentiment_signals_to_normalize:
            normalized_signals[f'{signal}_norm'] = self.helper._normalize_series(
                raw_signals[signal], df_index, bipolar=True
            )
        # 突破相关信号归一化
        breakout_signals_to_normalize = ['breakout_quality', 'breakout_confidence', 'breakout_potential']
        for signal in breakout_signals_to_normalize:
            normalized_signals[f'{signal}_norm'] = self.helper._normalize_series(
                raw_signals[signal], df_index, bipolar=False
            )
        # 趋势信号归一化
        trend_signals_to_normalize = ['uptrend_strength', 'downtrend_strength', 'trend_confirmation']
        for signal in trend_signals_to_normalize:
            normalized_signals[f'{signal}_norm'] = self.helper._normalize_series(
                raw_signals[signal], df_index, bipolar=True
            )
        # 行为模式归一化
        behavior_signals_to_normalize = ['accumulation_score', 'distribution_score',
                                        'behavior_accumulation', 'behavior_distribution']
        for signal in behavior_signals_to_normalize:
            normalized_signals[f'{signal}_norm'] = self.helper._normalize_series(
                raw_signals[signal], df_index, bipolar=True
            )
        return normalized_signals

    def _construct_proxy_signals(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], 
                                normalized_signals: Dict[str, pd.Series], config: Dict) -> Dict[str, pd.Series]:
        """
        【V4.0 · 代理信号深度重构版】
        基于幻方量化A股交易经验，深度重构代理信号构建系统
        核心理念：
        1. 多维度代理信号：相对强度、资本属性、市场情绪、流动性、波动性、风险偏好
        2. 非线性合成：使用几何平均或Sigmoid函数增强信号一致性
        3. 自适应权重：根据市场状态动态调整各维度权重
        4. 领先滞后关系：识别各代理信号的领先滞后特性
        5. 信号质量评估：评估每个代理信号的可靠性和稳定性
        数学模型创新：
        1. 相对强度指数（RSI扩展版）
        2. 资本效率因子（CEF）
        3. 情绪复合指数（SCI）
        4. 流动性分层模型（LTM）
        5. 波动性调整因子（VAF）
        6. 风险偏好指数（RPI）
        """
        self._probe_print("=== 代理信号构建开始 ===")
        # 1. 相对强度代理信号（增强版）
        rs_proxy = self._calculate_enhanced_rs_proxy(df_index, mtf_signals, normalized_signals, config)
        # 2. 资本属性代理信号（多维度资本分析）
        capital_proxy = self._calculate_enhanced_capital_proxy(df_index, mtf_signals, normalized_signals, config)
        # 3. 市场情绪代理信号（复合情绪指数）
        sentiment_proxy = self._calculate_enhanced_sentiment_proxy(df_index, mtf_signals, normalized_signals, config)
        # 4. 流动性代理信号（分层流动性分析）
        liquidity_proxy = self._calculate_enhanced_liquidity_proxy(df_index, mtf_signals, normalized_signals, config)
        # 5. 波动性代理信号（自适应波动率）
        volatility_proxy = self._calculate_enhanced_volatility_proxy(df_index, mtf_signals, normalized_signals, config)
        # 6. 风险偏好代理信号（市场风险偏好）
        risk_preference_proxy = self._calculate_enhanced_risk_preference_proxy(df_index, mtf_signals, normalized_signals, config)
        # 7. 综合信号质量评估
        signal_quality = self._assess_signal_quality(
            rs_proxy, capital_proxy, sentiment_proxy, 
            liquidity_proxy, volatility_proxy, risk_preference_proxy
        )
        # 8. 动态权重合成（根据信号质量和市场状态）
        final_proxy_signals = self._dynamic_weighted_synthesis(
            rs_proxy, capital_proxy, sentiment_proxy, 
            liquidity_proxy, volatility_proxy, risk_preference_proxy,
            signal_quality, config
        )
        self._probe_print("=== 代理信号构建完成 ===")
        return final_proxy_signals

    def _calculate_enhanced_rs_proxy(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], 
                                    normalized_signals: Dict[str, pd.Series], config: Dict) -> Dict[str, pd.Series]:
        """
        【V4.0】增强版相对强度代理信号
        核心理念：相对强度不仅看价格趋势，还要看动量、加速、结构等多个维度
        数学模型：多维度RSI扩展 + 结构强度 + 动量扩散
        数据层需要新增：
        1. 行业相对强度排名
        2. 板块轮动速度
        3. 相对成交额变化
        """
        rs_components = {}
        # 1. 价格趋势相对强度（MTF趋势）
        price_trend_strength = mtf_signals.get('mtf_price_trend', pd.Series(0.0, index=df_index)).clip(lower=0)
        # 2. 动量相对强度（自适应RSI）
        rsi_strength = self._calculate_adaptive_rsi_strength(
            normalized_signals.get('RSI_norm', pd.Series(0.5, index=df_index)),
            df_index
        )
        # 3. 加速相对强度（价格加速度）
        acceleration_strength = self._calculate_price_acceleration_strength(
            mtf_signals.get('mtf_price_trend', pd.Series(0.0, index=df_index)),
            df_index
        )
        # 4. 结构相对强度（突破、支撑、阻力）
        structural_strength = self._calculate_structural_strength(
            normalized_signals, mtf_signals, df_index
        )
        # 5. 成交量相对强度（量价配合）
        volume_strength = self._calculate_volume_relative_strength(
            normalized_signals.get('volume_ratio_norm', pd.Series(0.5, index=df_index)),
            price_trend_strength,
            df_index
        )
        # 6. 资金流相对强度（资金推动效率）
        capital_flow_strength = self._calculate_capital_flow_relative_strength(
            normalized_signals.get('net_amount_ratio_norm', pd.Series(0.5, index=df_index)),
            price_trend_strength,
            df_index
        )
        # 综合相对强度（使用加权几何平均）
        rs_weights = {
            "price_trend": 0.25,
            "rsi": 0.20,
            "acceleration": 0.15,
            "structural": 0.15,
            "volume": 0.15,
            "capital_flow": 0.10
        }
        rs_components_values = {
            "price_trend": price_trend_strength.clip(0, 1),
            "rsi": rsi_strength.clip(0, 1),
            "acceleration": acceleration_strength.clip(0, 1),
            "structural": structural_strength.clip(0, 1),
            "volume": volume_strength.clip(0, 1),
            "capital_flow": capital_flow_strength.clip(0, 1)
        }
        # 使用加权几何平均（强调一致性）
        rs_proxy = self._weighted_geometric_mean(rs_components_values, rs_weights, df_index)
        # 动态调节器：根据市场状态调整强度
        market_state = self._detect_market_state_for_rs(rs_components_values, df_index)
        rs_modulator = self._calculate_rs_modulator(market_state, config)
        # 最终相对强度代理信号
        enhanced_rs_proxy = (rs_proxy * rs_modulator).clip(0, 1)
        self._probe_print(f"相对强度代理信号构建完成，均值: {enhanced_rs_proxy.mean():.4f}")
        return {
            "raw_rs_proxy": rs_proxy,
            "enhanced_rs_proxy": enhanced_rs_proxy,
            "rs_modulator": rs_modulator,
            "rs_components": rs_components_values,
            "market_state": pd.Series(market_state, index=df_index)
        }

    def _calculate_enhanced_capital_proxy(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], 
                                         normalized_signals: Dict[str, pd.Series], config: Dict) -> Dict[str, pd.Series]:
        """
        【V4.0】增强版资本属性代理信号
        核心理念：资本属性包括资金流向、资金效率、资金结构、资金持续性
        数学模型：资本效率因子（CEF） + 资金结构指数 + 资金持续性
        数据层需要新增：
        1. 资金流日内分布（上午/下午）
        2. 资金流的板块集中度
        3. 大单冲击成本
        4. 资金流的方向持续性
        """
        capital_components = {}
        # 1. 多级别资金流合成（基于原始日级资金数据）
        # 使用向量合成法，但加入更多维度
        capital_flow_composite = self._calculate_multi_level_capital_flow(
            normalized_signals, df_index, config
        )
        # 2. 资本效率因子（CEF）：单位资金推动价格上涨的效率
        capital_efficiency = self._calculate_capital_efficiency_factor(
            capital_flow_composite,
            mtf_signals.get('mtf_price_trend', pd.Series(0.0, index=df_index)),
            normalized_signals.get('volume_ratio_norm', pd.Series(0.5, index=df_index)),
            df_index
        )
        # 3. 资金结构指数（FSI）：资金流向的集中度和稳定性
        fund_structure_index = self._calculate_fund_structure_index(
            normalized_signals, df_index
        )
        # 4. 资金持续性指数（FPI）：资金流向的持续时间和强度
        fund_persistence_index = self._calculate_fund_persistence_index(
            capital_flow_composite, df_index
        )
        # 5. 资金领先滞后关系（FLR）：资金流向领先价格的程度
        fund_lead_lag_ratio = self._calculate_fund_lead_lag_ratio(
            capital_flow_composite,
            mtf_signals.get('mtf_price_trend', pd.Series(0.0, index=df_index)),
            df_index
        )
        # 6. 资金异常检测（FAD）：检测异常资金流动
        fund_anomaly_detection = self._detect_fund_anomalies(
            capital_flow_composite, df_index
        )
        # 综合资本属性代理信号
        capital_weights = {
            "flow_composite": 0.25,
            "efficiency": 0.20,
            "structure": 0.20,
            "persistence": 0.15,
            "lead_lag": 0.10,
            "anomaly": 0.10  # 负向指标
        }
        capital_components_values = {
            "flow_composite": capital_flow_composite.clip(-1, 1) * 0.5 + 0.5,  # 映射到[0,1]
            "efficiency": capital_efficiency.clip(0, 1),
            "structure": fund_structure_index.clip(0, 1),
            "persistence": fund_persistence_index.clip(0, 1),
            "lead_lag": fund_lead_lag_ratio.clip(0, 1),
            "anomaly": (1 - fund_anomaly_detection).clip(0, 1)  # 异常越低越好
        }
        # 使用非线性合成（Sigmoid加权）
        capital_proxy = self._nonlinear_synthesis(capital_components_values, capital_weights, df_index)
        # 资本属性调节器：根据市场流动性调整
        liquidity_state = self._assess_market_liquidity_state(normalized_signals, df_index)
        capital_modulator = self._calculate_capital_modulator(liquidity_state, config)
        # 最终资本属性代理信号
        enhanced_capital_proxy = (capital_proxy * capital_modulator).clip(0, 1)
        self._probe_print(f"资本属性代理信号构建完成，均值: {enhanced_capital_proxy.mean():.4f}")
        return {
            "raw_capital_proxy": capital_proxy,
            "enhanced_capital_proxy": enhanced_capital_proxy,
            "capital_modulator": capital_modulator,
            "capital_components": capital_components_values,
            "liquidity_state": pd.Series(liquidity_state, index=df_index)
        }

    def _calculate_enhanced_sentiment_proxy(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], 
                                           normalized_signals: Dict[str, pd.Series], config: Dict) -> Dict[str, pd.Series]:
        """
        【V4.0】增强版市场情绪代理信号
        核心理念：市场情绪是多维度的，包括贪婪恐惧、一致性、极端性、传染性
        数学模型：情绪复合指数（SCI） + 情绪分歧度 + 情绪极端性
        数据层需要新增：
        1. 社交媒体情绪指数
        2. 新闻情绪分析
        3. 机构调研热度
        4. 投资者情绪调查数据
        """
        sentiment_components = {}
        # 1. 基础市场情绪（来自数据层）
        base_sentiment = normalized_signals.get('market_sentiment_norm', pd.Series(0.5, index=df_index))
        # 2. 情绪动量（情绪变化速度）
        sentiment_momentum = self._calculate_sentiment_momentum(
            base_sentiment, df_index
        )
        # 3. 情绪分歧度（市场参与者情绪差异）
        sentiment_divergence = self._calculate_sentiment_divergence_index(
            normalized_signals, df_index
        )
        # 4. 情绪极端性（情绪处于极端状态的程度）
        sentiment_extremity = self._calculate_sentiment_extremity_index(
            base_sentiment, df_index
        )
        # 5. 情绪传染性（情绪在板块间的传播）
        sentiment_contagion = self._calculate_sentiment_contagion_index(
            normalized_signals, df_index
        )
        # 6. 情绪稳定性（情绪的波动和反转）
        sentiment_stability = self._calculate_sentiment_stability_index(
            base_sentiment, df_index
        )
        # 综合情绪代理信号
        sentiment_weights = {
            "base": 0.25,
            "momentum": 0.20,
            "divergence": 0.15,
            "extremity": 0.15,
            "contagion": 0.15,
            "stability": 0.10
        }
        sentiment_components_values = {
            "base": base_sentiment.clip(0, 1),
            "momentum": (sentiment_momentum * 0.5 + 0.5).clip(0, 1),  # 映射到[0,1]
            "divergence": (1 - sentiment_divergence).clip(0, 1),  # 分歧越低越好
            "extremity": (1 - sentiment_extremity).clip(0, 1),  # 极端性越低越好
            "contagion": sentiment_contagion.clip(0, 1),
            "stability": sentiment_stability.clip(0, 1)
        }
        # 使用模糊逻辑合成
        sentiment_proxy = self._fuzzy_logic_synthesis(sentiment_components_values, sentiment_weights, df_index)
        # 情绪调节器：根据市场阶段调整
        market_phase = self._identify_market_phase(sentiment_components_values, df_index)
        sentiment_modulator = self._calculate_sentiment_modulator(market_phase, config)
        # 最终市场情绪代理信号
        enhanced_sentiment_proxy = (sentiment_proxy * sentiment_modulator).clip(0, 1)
        self._probe_print(f"市场情绪代理信号构建完成，均值: {enhanced_sentiment_proxy.mean():.4f}")
        return {
            "raw_sentiment_proxy": sentiment_proxy,
            "enhanced_sentiment_proxy": enhanced_sentiment_proxy,
            "sentiment_modulator": sentiment_modulator,
            "sentiment_components": sentiment_components_values,
            "market_phase": pd.Series(market_phase, index=df_index)
        }

    def _calculate_enhanced_liquidity_proxy(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], 
                                           normalized_signals: Dict[str, pd.Series], config: Dict) -> Dict[str, pd.Series]:
        """
        【V4.0】增强版流动性代理信号
        核心理念：流动性包括成交量、换手率、订单簿深度、买卖不平衡度
        数学模型：流动性分层模型（LTM） + 流动性风险溢价
        数据层需要新增：
        1. 订单簿不平衡度（买一卖一量比）
        2. 订单簿深度（各档位挂单量）
        3. 大单冲击成本
        4. 流动性分层数据（不同价格区间的流动性）
        """
        liquidity_components = {}
        # 1. 成交量流动性（传统流动性指标）
        volume_liquidity = normalized_signals.get('volume_ratio_norm', pd.Series(0.5, index=df_index))
        # 2. 换手率流动性（筹码交换速度）
        turnover_liquidity = normalized_signals.get('turnover_rate_norm', pd.Series(0.5, index=df_index))
        # 3. 订单簿流动性（买卖不平衡度）
        orderbook_liquidity = self._calculate_orderbook_liquidity_index(
            normalized_signals, df_index
        )
        # 4. 冲击成本流动性（大单交易对价格的影响）
        impact_cost_liquidity = self._calculate_impact_cost_liquidity(
            normalized_signals, df_index
        )
        # 5. 流动性分层（不同价格区间的流动性差异）
        layered_liquidity = self._calculate_layered_liquidity_index(
            normalized_signals, df_index
        )
        # 6. 流动性风险溢价（流动性不足时的风险补偿）
        liquidity_risk_premium = self._calculate_liquidity_risk_premium(
            volume_liquidity, turnover_liquidity, df_index
        )
        # 综合流动性代理信号（流动性越好，得分越高）
        liquidity_weights = {
            "volume": 0.25,
            "turnover": 0.20,
            "orderbook": 0.20,
            "impact_cost": 0.15,
            "layered": 0.10,
            "risk_premium": 0.10  # 负向指标
        }
        liquidity_components_values = {
            "volume": volume_liquidity.clip(0, 1),
            "turnover": turnover_liquidity.clip(0, 1),
            "orderbook": orderbook_liquidity.clip(0, 1),
            "impact_cost": (1 - impact_cost_liquidity).clip(0, 1),  # 冲击成本越低越好
            "layered": layered_liquidity.clip(0, 1),
            "risk_premium": (1 - liquidity_risk_premium).clip(0, 1)  # 风险溢价越低越好
        }
        # 使用最小二乘优化合成
        liquidity_proxy = self._optimized_synthesis(liquidity_components_values, liquidity_weights, df_index)
        # 流动性调节器：根据市场波动调整
        market_volatility = normalized_signals.get('volatility_instability_norm', pd.Series(0.5, index=df_index))
        liquidity_modulator = self._calculate_liquidity_modulator(market_volatility, config)
        # 最终流动性代理信号
        enhanced_liquidity_proxy = (liquidity_proxy * liquidity_modulator).clip(0, 1)
        self._probe_print(f"流动性代理信号构建完成，均值: {enhanced_liquidity_proxy.mean():.4f}")
        return {
            "raw_liquidity_proxy": liquidity_proxy,
            "enhanced_liquidity_proxy": enhanced_liquidity_proxy,
            "liquidity_modulator": liquidity_modulator,
            "liquidity_components": liquidity_components_values,
            "market_volatility": market_volatility
        }

    def _calculate_enhanced_volatility_proxy(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], 
                                            normalized_signals: Dict[str, pd.Series], config: Dict) -> Dict[str, pd.Series]:
        """
        【V4.0】增强版波动性代理信号
        核心理念：波动性包括历史波动、隐含波动、波动率微笑、波动率聚集
        数学模型：波动性调整因子（VAF） + 波动率结构模型
        数据层需要新增：
        1. 历史波动率（不同周期）
        2. 波动率偏度（上涨波动 vs 下跌波动）
        3. 波动率期限结构
        4. 波动率聚集效应数据
        """
        volatility_components = {}
        # 1. 历史波动率（基于收益率）
        historical_volatility = self._calculate_historical_volatility_index(
            normalized_signals.get('pct_change_norm', pd.Series(0.0, index=df_index)),
            df_index
        )
        # 2. 波动率偏度（上涨波动与下跌波动的不对称性）
        volatility_skew = self._calculate_volatility_skew_index(
            normalized_signals.get('pct_change_norm', pd.Series(0.0, index=df_index)),
            df_index
        )
        # 3. 波动率聚集（GARCH效应）
        volatility_clustering = self._calculate_volatility_clustering_index(
            normalized_signals.get('pct_change_norm', pd.Series(0.0, index=df_index)),
            df_index
        )
        # 4. 波动率微笑（不同行权价的波动率差异）
        volatility_smile = self._calculate_volatility_smile_index(
            normalized_signals, df_index
        )
        # 5. 波动率期限结构（不同到期日的波动率）
        volatility_term_structure = self._calculate_volatility_term_structure_index(
            normalized_signals, df_index
        )
        # 6. 波动率风险溢价（预期波动率与实际波动率差异）
        volatility_risk_premium = self._calculate_volatility_risk_premium_index(
            historical_volatility, df_index
        )
        # 综合波动性代理信号（适度的波动性最好）
        volatility_weights = {
            "historical": 0.25,
            "skew": 0.20,
            "clustering": 0.15,
            "smile": 0.15,
            "term_structure": 0.15,
            "risk_premium": 0.10
        }
        # 波动性得分：不是越低越好，而是适度最好（倒U型）
        volatility_components_values = {
            "historical": self._inverse_u_transform(historical_volatility, 0.2, 0.5),  # 最优波动率20%-50%
            "skew": (1 - abs(volatility_skew)).clip(0, 1),  # 偏度越小越好
            "clustering": (1 - volatility_clustering).clip(0, 1),  # 聚集效应越低越好
            "smile": (1 - volatility_smile).clip(0, 1),  # 微笑效应越小越好
            "term_structure": volatility_term_structure.clip(0, 1),
            "risk_premium": (1 - volatility_risk_premium).clip(0, 1)  # 风险溢价越低越好
        }
        # 使用贝叶斯合成
        volatility_proxy = self._bayesian_synthesis(volatility_components_values, volatility_weights, df_index)
        # 波动性调节器：根据市场阶段调整
        market_regime = self._identify_volatility_regime(volatility_components_values, df_index)
        volatility_modulator = self._calculate_volatility_modulator(market_regime, config)
        # 最终波动性代理信号
        enhanced_volatility_proxy = (volatility_proxy * volatility_modulator).clip(0, 1)
        self._probe_print(f"波动性代理信号构建完成，均值: {enhanced_volatility_proxy.mean():.4f}")
        return {
            "raw_volatility_proxy": volatility_proxy,
            "enhanced_volatility_proxy": enhanced_volatility_proxy,
            "volatility_modulator": volatility_modulator,
            "volatility_components": volatility_components_values,
            "market_regime": pd.Series(market_regime, index=df_index)
        }

    def _calculate_enhanced_risk_preference_proxy(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], 
                                                normalized_signals: Dict[str, pd.Series], config: Dict) -> Dict[str, pd.Series]:
        """
        【V4.0】增强版风险偏好代理信号
        核心理念：市场风险偏好决定资金流向和资产价格
        数学模型：风险偏好指数（RPI） + 风险规避程度 + 风险转移
        数据层需要新增：
        1. 高风险资产 vs 低风险资产表现
        2. 信用利差变化
        3. 期权偏度指数
        4. 避险资产资金流向
        """
        risk_preference_components = {}
        # 1. 风险资产表现（高风险板块 vs 低风险板块）
        risky_asset_performance = self._calculate_risky_asset_performance_index(
            normalized_signals, df_index
        )
        # 2. 风险规避程度（避险资产资金流向）
        risk_aversion_degree = self._calculate_risk_aversion_degree_index(
            normalized_signals, df_index
        )
        # 3. 风险转移（资金从避险资产流向风险资产）
        risk_transfer = self._calculate_risk_transfer_index(
            normalized_signals, df_index
        )
        # 4. 风险定价（风险溢价水平）
        risk_pricing = self._calculate_risk_pricing_index(
            normalized_signals, df_index
        )
        # 5. 风险情绪（投资者对风险的容忍度）
        risk_sentiment = self._calculate_risk_sentiment_index(
            normalized_signals, df_index
        )
        # 6. 风险传导（风险在资产间的传播）
        risk_contagion = self._calculate_risk_contagion_index(
            normalized_signals, df_index
        )
        # 综合风险偏好代理信号（风险偏好高=得分高）
        risk_weights = {
            "risky_performance": 0.25,
            "aversion_degree": 0.20,  # 负向指标
            "risk_transfer": 0.20,
            "risk_pricing": 0.15,
            "risk_sentiment": 0.10,
            "risk_contagion": 0.10  # 负向指标
        }
        risk_preference_components_values = {
            "risky_performance": risky_asset_performance.clip(0, 1),
            "aversion_degree": (1 - risk_aversion_degree).clip(0, 1),  # 风险规避越低越好
            "risk_transfer": risk_transfer.clip(0, 1),
            "risk_pricing": risk_pricing.clip(0, 1),
            "risk_sentiment": risk_sentiment.clip(0, 1),
            "risk_contagion": (1 - risk_contagion).clip(0, 1)  # 风险传导越低越好
        }
        # 使用神经网络启发式合成
        risk_preference_proxy = self._neural_inspired_synthesis(risk_preference_components_values, risk_weights, df_index)
        # 风险偏好调节器：根据经济周期调整
        economic_cycle = self._identify_economic_cycle(risk_preference_components_values, df_index)
        risk_preference_modulator = self._calculate_risk_preference_modulator(economic_cycle, config)
        # 最终风险偏好代理信号
        enhanced_risk_preference_proxy = (risk_preference_proxy * risk_preference_modulator).clip(0, 1)
        self._probe_print(f"风险偏好代理信号构建完成，均值: {enhanced_risk_preference_proxy.mean():.4f}")
        return {
            "raw_risk_preference_proxy": risk_preference_proxy,
            "enhanced_risk_preference_proxy": enhanced_risk_preference_proxy,
            "risk_preference_modulator": risk_preference_modulator,
            "risk_preference_components": risk_preference_components_values,
            "economic_cycle": pd.Series(economic_cycle, index=df_index)
        }

    def _dynamic_weighted_synthesis(self, rs_proxy: Dict, capital_proxy: Dict, sentiment_proxy: Dict,
                                   liquidity_proxy: Dict, volatility_proxy: Dict, risk_preference_proxy: Dict,
                                   signal_quality: pd.Series, config: Dict) -> Dict[str, pd.Series]:
        """
        【V4.0】动态权重合成算法
        核心理念：根据市场状态和信号质量动态调整各代理信号的权重
        数学模型：马尔可夫决策过程 + 动态规划优化
        策略：
        1. 趋势市：强调相对强度和资本属性
        2. 震荡市：强调波动性和流动性
        3. 转折市：强调市场情绪和风险偏好
        4. 危机市：强调风险偏好和波动性
        """
        # 提取增强版代理信号
        rs_enhanced = rs_proxy.get("enhanced_rs_proxy", pd.Series(0.5, index=signal_quality.index))
        capital_enhanced = capital_proxy.get("enhanced_capital_proxy", pd.Series(0.5, index=signal_quality.index))
        sentiment_enhanced = sentiment_proxy.get("enhanced_sentiment_proxy", pd.Series(0.5, index=signal_quality.index))
        liquidity_enhanced = liquidity_proxy.get("enhanced_liquidity_proxy", pd.Series(0.5, index=signal_quality.index))
        volatility_enhanced = volatility_proxy.get("enhanced_volatility_proxy", pd.Series(0.5, index=signal_quality.index))
        risk_preference_enhanced = risk_preference_proxy.get("enhanced_risk_preference_proxy", pd.Series(0.5, index=signal_quality.index))
        # 检测市场状态
        market_state = self._detect_comprehensive_market_state(
            rs_enhanced, capital_enhanced, sentiment_enhanced,
            liquidity_enhanced, volatility_enhanced, risk_preference_enhanced
        )
        # 动态权重矩阵（根据市场状态）
        weight_matrices = {
            "trending_up": {
                "rs": 0.30, "capital": 0.25, "sentiment": 0.15,
                "liquidity": 0.10, "volatility": 0.10, "risk_preference": 0.10
            },
            "trending_down": {
                "rs": 0.15, "capital": 0.20, "sentiment": 0.20,
                "liquidity": 0.15, "volatility": 0.20, "risk_preference": 0.10
            },
            "consolidating": {
                "rs": 0.15, "capital": 0.15, "sentiment": 0.20,
                "liquidity": 0.20, "volatility": 0.20, "risk_preference": 0.10
            },
            "reversing_up": {
                "rs": 0.20, "capital": 0.20, "sentiment": 0.25,
                "liquidity": 0.15, "volatility": 0.10, "risk_preference": 0.10
            },
            "reversing_down": {
                "rs": 0.15, "capital": 0.15, "sentiment": 0.25,
                "liquidity": 0.15, "volatility": 0.20, "risk_preference": 0.10
            },
            "crisis": {
                "rs": 0.10, "capital": 0.10, "sentiment": 0.20,
                "liquidity": 0.15, "volatility": 0.25, "risk_preference": 0.20
            }
        }
        # 应用信号质量调整权重（高质量信号权重增加）
        quality_adjusted_weights = {}
        for state, weights in weight_matrices.items():
            quality_adjusted_weights[state] = {}
            for key, weight in weights.items():
                # 信号质量影响因子：质量越高，权重增加越多
                quality_factor = 1 + signal_quality * 0.3  # 最高增加30%
                quality_adjusted_weights[state][key] = weight * quality_factor
        # 按日期动态合成
        final_proxy_signals = {}
        rs_modulator_values = []
        capital_modulator_values = []
        for i in range(len(signal_quality)):
            current_state = market_state.iloc[i] if i < len(market_state) else "consolidating"
            current_weights = quality_adjusted_weights.get(current_state, weight_matrices["consolidating"])
            # 提取当前日的各代理信号值
            rs_val = rs_enhanced.iloc[i] if i < len(rs_enhanced) else 0.5
            capital_val = capital_enhanced.iloc[i] if i < len(capital_enhanced) else 0.5
            sentiment_val = sentiment_enhanced.iloc[i] if i < len(sentiment_enhanced) else 0.5
            liquidity_val = liquidity_enhanced.iloc[i] if i < len(liquidity_enhanced) else 0.5
            volatility_val = volatility_enhanced.iloc[i] if i < len(volatility_enhanced) else 0.5
            risk_preference_val = risk_preference_enhanced.iloc[i] if i < len(risk_preference_enhanced) else 0.5
            # 提取权重
            rs_weight = current_weights["rs"].iloc[i] if isinstance(current_weights["rs"], pd.Series) else current_weights["rs"]
            capital_weight = current_weights["capital"].iloc[i] if isinstance(current_weights["capital"], pd.Series) else current_weights["capital"]
            sentiment_weight = current_weights["sentiment"].iloc[i] if isinstance(current_weights["sentiment"], pd.Series) else current_weights["sentiment"]
            liquidity_weight = current_weights["liquidity"].iloc[i] if isinstance(current_weights["liquidity"], pd.Series) else current_weights["liquidity"]
            volatility_weight = current_weights["volatility"].iloc[i] if isinstance(current_weights["volatility"], pd.Series) else current_weights["volatility"]
            risk_preference_weight = current_weights["risk_preference"].iloc[i] if isinstance(current_weights["risk_preference"], pd.Series) else current_weights["risk_preference"]
            # 加权平均
            weighted_sum = (
                rs_val * rs_weight +
                capital_val * capital_weight +
                sentiment_val * sentiment_weight +
                liquidity_val * liquidity_weight +
                volatility_val * volatility_weight +
                risk_preference_val * risk_preference_weight
            )
            total_weight = (
                rs_weight + capital_weight + sentiment_weight +
                liquidity_weight + volatility_weight + risk_preference_weight
            )
            if total_weight > 0:
                final_value = weighted_sum / total_weight
            else:
                final_value = 0.5
            # 存储最终代理信号
            if i == 0:
                final_proxy_signals = {
                    "rs_modulator": pd.Series(index=signal_quality.index, dtype=float),
                    "capital_modulator": pd.Series(index=signal_quality.index, dtype=float),
                    "market_sentiment_proxy": pd.Series(index=signal_quality.index, dtype=float),
                    "liquidity_tide_proxy": pd.Series(index=signal_quality.index, dtype=float),
                    "volatility_proxy": pd.Series(index=signal_quality.index, dtype=float),
                    "risk_preference_proxy": pd.Series(index=signal_quality.index, dtype=float),
                    "comprehensive_proxy": pd.Series(index=signal_quality.index, dtype=float),
                    "market_state": pd.Series(index=signal_quality.index, dtype=str)
                }
            final_proxy_signals["rs_modulator"].iloc[i] = rs_proxy.get("rs_modulator", pd.Series(1.0, index=signal_quality.index)).iloc[i] if i < len(rs_proxy.get("rs_modulator", pd.Series(1.0, index=signal_quality.index))) else 1.0
            final_proxy_signals["capital_modulator"].iloc[i] = capital_proxy.get("capital_modulator", pd.Series(1.0, index=signal_quality.index)).iloc[i] if i < len(capital_proxy.get("capital_modulator", pd.Series(1.0, index=signal_quality.index))) else 1.0
            final_proxy_signals["market_sentiment_proxy"].iloc[i] = sentiment_val
            final_proxy_signals["liquidity_tide_proxy"].iloc[i] = liquidity_val
            final_proxy_signals["volatility_proxy"].iloc[i] = volatility_val
            final_proxy_signals["risk_preference_proxy"].iloc[i] = risk_preference_val
            final_proxy_signals["comprehensive_proxy"].iloc[i] = final_value
            final_proxy_signals["market_state"].iloc[i] = current_state
        self._probe_print(f"动态权重合成完成，综合代理信号均值: {final_proxy_signals['comprehensive_proxy'].mean():.4f}")
        return final_proxy_signals

    def _assess_signal_quality(self, rs_proxy: Dict, capital_proxy: Dict, sentiment_proxy: Dict,
                              liquidity_proxy: Dict, volatility_proxy: Dict, risk_preference_proxy: Dict) -> pd.Series:
        """
        【V4.0】代理信号质量评估方法
        核心理念：综合评估各代理信号的可靠性、稳定性、一致性和预测能力
        数学模型：多维度质量评分系统 + 自适应权重分配
        质量评估维度：
        1. 信号稳定性（低波动性 = 高质量）
        2. 信噪比（清晰信号 vs 噪声）
        3. 一致性（与其他相关信号的一致性）
        4. 预测有效性（对未来价格变动的预测能力）
        5. 极端值频率（异常值越少越好）
        6. 信号完整性（数据缺失率）
        返回：综合信号质量评分，范围[0, 1]，1表示最高质量
        """
        self._probe_print("=== 开始评估代理信号质量 ===")
        # 提取各代理信号的时间序列（使用增强版信号）
        rs_signal = rs_proxy.get("enhanced_rs_proxy", pd.Series(0.5, index=rs_proxy.get("raw_rs_proxy", pd.Series()).index))
        capital_signal = capital_proxy.get("enhanced_capital_proxy", pd.Series(0.5, index=capital_proxy.get("raw_capital_proxy", pd.Series()).index))
        sentiment_signal = sentiment_proxy.get("enhanced_sentiment_proxy", pd.Series(0.5, index=sentiment_proxy.get("raw_sentiment_proxy", pd.Series()).index))
        liquidity_signal = liquidity_proxy.get("enhanced_liquidity_proxy", pd.Series(0.5, index=liquidity_proxy.get("raw_liquidity_proxy", pd.Series()).index))
        volatility_signal = volatility_proxy.get("enhanced_volatility_proxy", pd.Series(0.5, index=volatility_proxy.get("raw_volatility_proxy", pd.Series()).index))
        risk_preference_signal = risk_preference_proxy.get("enhanced_risk_preference_proxy", pd.Series(0.5, index=risk_preference_proxy.get("raw_risk_preference_proxy", pd.Series()).index))
        # 确保所有信号有相同的索引和长度
        all_signals = {
            "rs": rs_signal,
            "capital": capital_signal,
            "sentiment": sentiment_signal,
            "liquidity": liquidity_signal,
            "volatility": volatility_signal,
            "risk_preference": risk_preference_signal
        }
        # 1. 计算各信号的个体质量指标
        individual_qualities = {}
        for signal_name, signal_series in all_signals.items():
            if signal_series.empty:
                individual_qualities[signal_name] = pd.Series(0.5, index=signal_series.index)
                continue
            signal_quality = self._calculate_individual_signal_quality(signal_series, signal_name)
            individual_qualities[signal_name] = signal_quality
            self._probe_print(f"  {signal_name}信号质量均值: {signal_quality.mean():.4f}")
        # 2. 计算信号间的一致性质量
        consistency_quality = self._calculate_signal_consistency_quality(all_signals)
        # 3. 计算信号的预测有效性（滞后相关性分析）
        predictive_quality = self._calculate_predictive_quality(all_signals)
        # 4. 综合信号质量（加权合成）
        # 动态权重：根据各信号的历史表现调整权重
        dynamic_weights = self._calculate_dynamic_quality_weights(individual_qualities, consistency_quality, predictive_quality)
        # 合成综合信号质量
        comprehensive_quality = pd.Series(0.0, index=rs_signal.index)
        for i in range(len(comprehensive_quality)):
            if i < 20:  # 前20天数据不足，使用默认值
                comprehensive_quality.iloc[i] = 0.7
                continue
            # 计算各质量维度的加权平均
            weighted_sum = 0
            total_weight = 0
            # 个体信号质量贡献
            for signal_name in individual_qualities:
                if signal_name in dynamic_weights:
                    signal_weight = dynamic_weights[signal_name].iloc[i] if isinstance(dynamic_weights[signal_name], pd.Series) else dynamic_weights[signal_name]
                    signal_quality_val = individual_qualities[signal_name].iloc[i] if i < len(individual_qualities[signal_name]) else 0.5
                    weighted_sum += signal_quality_val * signal_weight
                    total_weight += signal_weight
            # 一致性质量贡献
            consistency_weight = 0.2  # 固定权重
            consistency_val = consistency_quality.iloc[i] if i < len(consistency_quality) else 0.5
            weighted_sum += consistency_val * consistency_weight
            total_weight += consistency_weight
            # 预测有效性贡献
            predictive_weight = 0.3  # 固定权重
            predictive_val = predictive_quality.iloc[i] if i < len(predictive_quality) else 0.5
            weighted_sum += predictive_val * predictive_weight
            total_weight += predictive_weight
            if total_weight > 0:
                comprehensive_quality.iloc[i] = weighted_sum / total_weight
            else:
                comprehensive_quality.iloc[i] = 0.5
        # 平滑处理
        comprehensive_quality = comprehensive_quality.rolling(window=5, min_periods=3).mean().fillna(0.7)
        self._probe_print(f"综合信号质量评估完成，均值: {comprehensive_quality.mean():.4f}")
        return comprehensive_quality.clip(0, 1)

    def _calculate_individual_signal_quality(self, signal_series: pd.Series, signal_name: str) -> pd.Series:
        """
        【V4.0】计算单个信号的个体质量指标
        评估维度：
        1. 稳定性（波动率倒数）
        2. 信噪比（趋势强度/噪声强度）
        3. 极端值频率（异常值比例）
        4. 数据完整性（缺失值比例）
        """
        if signal_series.empty:
            return pd.Series(0.5, index=signal_series.index)
        quality_scores = pd.Series(0.5, index=signal_series.index)
        for i in range(len(signal_series)):
            if i < 20:  # 需要足够的数据来计算质量
                quality_scores.iloc[i] = 0.5
                continue
            # 1. 稳定性评分（基于滚动窗口波动率）
            window_data = signal_series.iloc[max(0, i-20):i]
            if len(window_data) >= 10:
                volatility = window_data.std()
                if volatility > 0:
                    # 波动率越低，稳定性越高（倒U型，适中最好）
                    # 对于大多数代理信号，0.1-0.3的波动率是理想的
                    optimal_volatility = 0.2
                    stability_score = 1.0 - min(abs(volatility - optimal_volatility) / optimal_volatility, 1.0)
                else:
                    stability_score = 0.5
            else:
                stability_score = 0.5
            # 2. 信噪比评分（趋势强度 vs 噪声强度）
            if len(window_data) >= 15:
                # 计算趋势（线性回归斜率）
                x = np.arange(len(window_data))
                y = window_data.values
                try:
                    slope, _ = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y, rcond=None)[0]
                    # 趋势强度的绝对值
                    trend_strength = abs(slope) * 100  # 放大到合理范围
                    # 噪声强度（去趋势后的残差标准差）
                    residuals = y - (slope * x + _)
                    noise_strength = np.std(residuals)
                    if noise_strength > 0:
                        snr = trend_strength / noise_strength
                        # SNR在1-3之间为理想范围
                        snr_score = min(snr / 3.0, 1.0)
                    else:
                        snr_score = 1.0 if trend_strength > 0 else 0.5
                except:
                    snr_score = 0.5
            else:
                snr_score = 0.5
            # 3. 极端值频率评分
            if len(window_data) >= 20:
                # 使用IQR方法检测极端值
                Q1 = window_data.quantile(0.25)
                Q3 = window_data.quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = window_data[(window_data < lower_bound) | (window_data > upper_bound)]
                    outlier_ratio = len(outliers) / len(window_data)
                    # 极端值越少越好
                    extreme_score = 1.0 - outlier_ratio
                else:
                    extreme_score = 1.0
            else:
                extreme_score = 0.5
            # 4. 数据完整性评分
            # 计算窗口内缺失值比例
            if len(window_data) > 0:
                missing_ratio = window_data.isna().sum() / len(window_data)
                completeness_score = 1.0 - missing_ratio
            else:
                completeness_score = 1.0
            # 综合个体质量评分
            # 权重分配：稳定性30%，信噪比30%，极端值20%，完整性20%
            individual_quality = (
                stability_score * 0.3 +
                snr_score * 0.3 +
                extreme_score * 0.2 +
                completeness_score * 0.2
            )
            quality_scores.iloc[i] = individual_quality
        # 前向填充并平滑
        quality_scores = quality_scores.ffill().fillna(0.5)
        quality_scores = quality_scores.rolling(window=5, min_periods=3).mean().fillna(0.5)
        return quality_scores.clip(0, 1)

    def _calculate_signal_consistency_quality(self, all_signals: Dict[str, pd.Series]) -> pd.Series:
        """
        【V4.0】计算信号间的一致性质量
        评估维度：
        1. 方向一致性（各信号方向相同的比例）
        2. 幅度一致性（变化幅度相关性）
        3. 时序一致性（领先滞后关系的稳定性）
        """
        # 获取第一个信号的索引作为基准
        first_signal = next(iter(all_signals.values()))
        if first_signal.empty:
            return pd.Series(0.5, index=first_signal.index)
        consistency_scores = pd.Series(0.5, index=first_signal.index)
        for i in range(len(consistency_scores)):
            if i < 30:  # 需要足够的数据
                consistency_scores.iloc[i] = 0.5
                continue
            # 获取最近30天的数据窗口
            window_start = max(0, i-30)
            window_end = i
            # 收集各信号在窗口内的数据
            window_signals = {}
            for signal_name, signal_series in all_signals.items():
                if len(signal_series) > window_end:
                    window_data = signal_series.iloc[window_start:window_end]
                    if len(window_data) >= 10:
                        window_signals[signal_name] = window_data
            if len(window_signals) < 2:
                consistency_scores.iloc[i] = 0.5
                continue
            # 1. 方向一致性评分
            direction_scores = []
            for day in range(max(0, window_end-5), window_end):  # 最近5天
                if day >= len(first_signal):
                    continue
                directions = []
                for signal_name, signal_series in window_signals.items():
                    if day < len(signal_series):
                        # 计算方向（与前一天比较）
                        if day > 0 and (day-1) < len(signal_series):
                            change = signal_series.iloc[day] - signal_series.iloc[day-1]
                            direction = 1 if change > 0 else (-1 if change < 0 else 0)
                            directions.append(direction)
                if len(directions) >= 2:
                    # 计算方向一致比例
                    positive_count = sum(1 for d in directions if d > 0)
                    negative_count = sum(1 for d in directions if d < 0)
                    max_uniform = max(positive_count, negative_count)
                    direction_consistency = max_uniform / len(directions) if len(directions) > 0 else 0
                    direction_scores.append(direction_consistency)
            direction_avg = np.mean(direction_scores) if direction_scores else 0.5
            # 2. 幅度一致性评分（相关系数）
            correlation_scores = []
            signal_names = list(window_signals.keys())
            for j in range(len(signal_names)):
                for k in range(j+1, len(signal_names)):
                    signal1 = window_signals[signal_names[j]]
                    signal2 = window_signals[signal_names[k]]
                    if len(signal1) >= 10 and len(signal2) >= 10:
                        # 确保长度一致
                        min_len = min(len(signal1), len(signal2))
                        signal1_trimmed = signal1.iloc[:min_len]
                        signal2_trimmed = signal2.iloc[:min_len]
                        # 计算相关系数
                        corr = signal1_trimmed.corr(signal2_trimmed)
                        if not np.isnan(corr):
                            correlation_scores.append(abs(corr))  # 取绝对值
            correlation_avg = np.mean(correlation_scores) if correlation_scores else 0.5
            # 3. 时序一致性评分（领先滞后关系的稳定性）
            timing_scores = []
            # 简化实现：计算信号变化的同步性
            for j in range(len(signal_names)):
                for k in range(j+1, len(signal_names)):
                    signal1 = window_signals[signal_names[j]]
                    signal2 = window_signals[signal_names[k]]
                    if len(signal1) >= 10 and len(signal2) >= 10:
                        # 计算一阶差分的相关性（同步性）
                        diff1 = signal1.diff().dropna()
                        diff2 = signal2.diff().dropna()
                        if len(diff1) >= 5 and len(diff2) >= 5:
                            min_len = min(len(diff1), len(diff2))
                            diff1_trimmed = diff1.iloc[:min_len]
                            diff2_trimmed = diff2.iloc[:min_len]
                            
                            sync_corr = diff1_trimmed.corr(diff2_trimmed)
                            if not np.isnan(sync_corr):
                                timing_scores.append(abs(sync_corr))
            timing_avg = np.mean(timing_scores) if timing_scores else 0.5
            # 综合一致性评分
            consistency_score = (
                direction_avg * 0.4 +
                correlation_avg * 0.4 +
                timing_avg * 0.2
            )
            consistency_scores.iloc[i] = consistency_score
        # 平滑处理
        consistency_scores = consistency_scores.ffill().fillna(0.5)
        consistency_scores = consistency_scores.rolling(window=5, min_periods=3).mean().fillna(0.5)
        return consistency_scores.clip(0, 1)

    def _calculate_predictive_quality(self, all_signals: Dict[str, pd.Series]) -> pd.Series:
        """
        【V4.0】计算信号的预测有效性质量
        评估维度：信号对未来价格变动的预测能力
        数学模型：滞后相关性分析 + 信息系数（IC）
        """
        # 注意：这里需要价格数据，假设我们有一个全局的价格序列
        # 由于在上下文中没有价格数据，我们将使用相对强度信号作为代理
        # 简化实现：使用信号自身的变化来评估预测能力
        # 在实际应用中，应该使用信号与未来收益的相关性
        predictive_scores = pd.Series(0.5, index=next(iter(all_signals.values())).index)
        for i in range(len(predictive_scores)):
            if i < 40:  # 需要足够的历史数据
                predictive_scores.iloc[i] = 0.5
                continue
            # 使用最近20天的数据评估预测能力
            window_start = max(0, i-20)
            window_end = i
            # 收集各信号的预测能力评分
            signal_predictive_scores = []
            for signal_name, signal_series in all_signals.items():
                if len(signal_series) <= window_end:
                    continue
                window_data = signal_series.iloc[window_start:window_end]
                if len(window_data) < 10:
                    continue
                # 计算信号变化的预测能力
                # 简化：假设信号的趋势变化具有一定的持续性
                # 实际中应该计算信号与未来n日收益的相关性
                # 计算信号的自相关性（滞后1期）
                if len(window_data) >= 10:
                    autocorr = window_data.autocorr(lag=1)
                    if not np.isnan(autocorr):
                        # 适度的正自相关表示有一定的预测能力
                        # 但过高的自相关可能意味着信号过于平滑，反应迟钝
                        if autocorr > 0:
                            # 0.1-0.4的自相关为理想范围
                            if autocorr < 0.1:
                                pred_score = autocorr / 0.1
                            elif autocorr > 0.4:
                                pred_score = 1.0 - (autocorr - 0.4) / 0.6
                            else:
                                pred_score = 1.0
                        else:
                            pred_score = 0.0
                        signal_predictive_scores.append(pred_score)
            # 计算平均预测能力
            if signal_predictive_scores:
                predictive_score = np.mean(signal_predictive_scores)
            else:
                predictive_score = 0.5
            predictive_scores.iloc[i] = predictive_score
        # 平滑处理
        predictive_scores = predictive_scores.ffill().fillna(0.5)
        predictive_scores = predictive_scores.rolling(window=10, min_periods=5).mean().fillna(0.5)
        return predictive_scores.clip(0, 1)

    def _calculate_dynamic_quality_weights(self, individual_qualities: Dict[str, pd.Series], 
                                         consistency_quality: pd.Series, 
                                         predictive_quality: pd.Series) -> Dict[str, pd.Series]:
        """
        【V4.0】计算动态质量权重
        核心理念：根据各信号的历史表现动态调整其在质量评估中的权重
        表现好的信号获得更高权重，表现差的信号权重降低
        """
        # 初始化权重
        dynamic_weights = {}
        # 获取时间索引
        if individual_qualities:
            index = next(iter(individual_qualities.values())).index
        else:
            return {}
        # 为每个信号创建权重序列
        for signal_name in individual_qualities:
            dynamic_weights[signal_name] = pd.Series(1.0 / len(individual_qualities), index=index)
        # 如果只有1-2个信号，直接返回等权重
        if len(individual_qualities) <= 2:
            return dynamic_weights
        # 动态调整权重（基于最近的表现）
        for i in range(len(index)):
            if i < 30:  # 前30天使用等权重
                continue
            # 计算各信号最近20天的平均质量
            recent_qualities = {}
            total_quality = 0
            for signal_name, quality_series in individual_qualities.items():
                if i < len(quality_series):
                    recent_window = quality_series.iloc[max(0, i-20):i]
                    recent_avg = recent_window.mean() if len(recent_window) > 0 else 0.5
                    recent_qualities[signal_name] = recent_avg
                    total_quality += recent_avg
                else:
                    recent_qualities[signal_name] = 0.5
                    total_quality += 0.5
            # 计算归一化权重
            if total_quality > 0:
                for signal_name in individual_qualities:
                    if signal_name in recent_qualities:
                        # 权重与质量成正比
                        normalized_weight = recent_qualities[signal_name] / total_quality
                        # 限制权重范围：0.05 - 0.4
                        normalized_weight = max(0.05, min(normalized_weight, 0.4))
                        dynamic_weights[signal_name].iloc[i] = normalized_weight
                    else:
                        dynamic_weights[signal_name].iloc[i] = 1.0 / len(individual_qualities)
        return dynamic_weights

    def _detect_comprehensive_market_state(self, rs_signal: pd.Series, capital_signal: pd.Series,
                                          sentiment_signal: pd.Series, liquidity_signal: pd.Series,
                                          volatility_signal: pd.Series, risk_preference_signal: pd.Series) -> pd.Series:
        """
        【V4.0】综合市场状态检测方法
        核心理念：基于多个代理信号综合判断市场状态
        市场状态分类：
        1. trending_up（趋势上涨）
        2. trending_down（趋势下跌）
        3. consolidating（震荡整理）
        4. reversing_up（反转上涨）
        5. reversing_down（反转下跌）
        6. crisis（危机模式）
        """
        market_states = pd.Series("consolidating", index=rs_signal.index)
        for i in range(len(market_states)):
            if i < 30:  # 需要足够的历史数据
                market_states.iloc[i] = "consolidating"
                continue
            # 获取当前信号值
            rs_val = rs_signal.iloc[i] if i < len(rs_signal) else 0.5
            capital_val = capital_signal.iloc[i] if i < len(capital_signal) else 0.5
            sentiment_val = sentiment_signal.iloc[i] if i < len(sentiment_signal) else 0.5
            liquidity_val = liquidity_signal.iloc[i] if i < len(liquidity_signal) else 0.5
            volatility_val = volatility_signal.iloc[i] if i < len(volatility_signal) else 0.5
            risk_val = risk_preference_signal.iloc[i] if i < len(risk_preference_signal) else 0.5
            # 获取最近20天的信号值用于趋势判断
            window_start = max(0, i-20)
            rs_window = rs_signal.iloc[window_start:i] if i < len(rs_signal) else pd.Series()
            capital_window = capital_signal.iloc[window_start:i] if i < len(capital_signal) else pd.Series()
            sentiment_window = sentiment_signal.iloc[window_start:i] if i < len(sentiment_signal) else pd.Series()
            # 计算趋势
            rs_trend = self._calculate_trend_direction(rs_window)
            capital_trend = self._calculate_trend_direction(capital_window)
            sentiment_trend = self._calculate_trend_direction(sentiment_window)
            # 计算信号强度
            rs_strength = abs(rs_trend)
            capital_strength = abs(capital_trend)
            sentiment_strength = abs(sentiment_trend)
            # 规则1：危机模式检测
            # 高波动性 + 低风险偏好 + 低流动性
            if volatility_val > 0.7 and risk_val < 0.3 and liquidity_val < 0.3:
                market_states.iloc[i] = "crisis"
                continue
            # 规则2：趋势上涨检测
            # 相对强度强 + 资金流入 + 情绪积极
            if rs_val > 0.7 and capital_val > 0.6 and sentiment_val > 0.6:
                if rs_trend > 0.1 and capital_trend > 0.1:  # 趋势向上
                    market_states.iloc[i] = "trending_up"
                    continue
            # 规则3：趋势下跌检测
            # 相对强度弱 + 资金流出 + 情绪消极
            if rs_val < 0.3 and capital_val < 0.4 and sentiment_val < 0.4:
                if rs_trend < -0.1 and capital_trend < -0.1:  # 趋势向下
                    market_states.iloc[i] = "trending_down"
                    continue
            # 规则4：反转上涨检测
            # 相对强度开始转强 + 资金开始流入 + 情绪改善
            if rs_trend > 0.2 and capital_trend > 0.2 and sentiment_trend > 0.2:
                # 但当前值还不太高
                if rs_val < 0.6 and capital_val < 0.6:
                    market_states.iloc[i] = "reversing_up"
                    continue
            # 规则5：反转下跌检测
            # 相对强度开始转弱 + 资金开始流出 + 情绪恶化
            if rs_trend < -0.2 and capital_trend < -0.2 and sentiment_trend < -0.2:
                # 但当前值还不太低
                if rs_val > 0.4 and capital_val > 0.4:
                    market_states.iloc[i] = "reversing_down"
                    continue
            # 默认状态：震荡整理
            market_states.iloc[i] = "consolidating"
        return market_states

    def _calculate_trend_direction(self, series: pd.Series) -> float:
        """
        【V4.0】计算序列的趋势方向
        返回：趋势斜率，正数表示上升趋势，负数表示下降趋势
        """
        if series.empty or len(series) < 5:
            return 0.0
        try:
            x = np.arange(len(series))
            y = series.values
            # 线性回归
            A = np.vstack([x, np.ones(len(x))]).T
            slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
            # 归一化到[-1, 1]范围
            # 假设最大斜率不超过0.05每天
            normalized_slope = slope / 0.05
            normalized_slope = max(-1.0, min(normalized_slope, 1.0))
            return normalized_slope
        except:
            return 0.0

    def _weighted_geometric_mean(self, components: Dict[str, pd.Series], weights: Dict[str, float], df_index: pd.Index) -> pd.Series:
        """
        【V4.0】加权几何平均
        核心理念：几何平均强调一致性，所有因子都需要表现良好
        公式：exp(Σ(weight_i * ln(component_i + epsilon)))
        """
        result = pd.Series(0.0, index=df_index)
        for i in range(len(result)):
            if i < 10:  # 需要足够的数据
                result.iloc[i] = 0.5
                continue
            log_sum = 0.0
            total_weight = 0.0
            for comp_name, weight in weights.items():
                if comp_name in components:
                    comp_series = components[comp_name]
                    if i < len(comp_series):
                        comp_val = comp_series.iloc[i]
                        # 避免零或负值
                        comp_val = max(0.001, comp_val)
                        log_sum += weight * np.log(comp_val)
                        total_weight += weight
            if total_weight > 0:
                result.iloc[i] = np.exp(log_sum / total_weight)
            else:
                result.iloc[i] = 0.5
        return result.clip(0, 1)

    def _nonlinear_synthesis(self, components: Dict[str, pd.Series], weights: Dict[str, float], df_index: pd.Index) -> pd.Series:
        """
        【V4.0】非线性合成（Sigmoid加权）
        核心理念：使用Sigmoid函数处理非线性关系
        公式：Σ(weight_i * sigmoid(component_i))
        """
        result = pd.Series(0.0, index=df_index)
        for i in range(len(result)):
            if i < 10:
                result.iloc[i] = 0.5
                continue
            weighted_sum = 0.0
            total_weight = 0.0
            for comp_name, weight in weights.items():
                if comp_name in components:
                    comp_series = components[comp_name]
                    if i < len(comp_series):
                        comp_val = comp_series.iloc[i]
                        # Sigmoid变换
                        sigmoid_val = 1 / (1 + np.exp(-(comp_val - 0.5) * 10))
                        weighted_sum += weight * sigmoid_val
                        total_weight += weight
            if total_weight > 0:
                result.iloc[i] = weighted_sum / total_weight
            else:
                result.iloc[i] = 0.5
        return result.clip(0, 1)

    def _calculate_dynamic_weights(self, df_index: pd.Index, normalized_signals: Dict[str, pd.Series], 
                                  proxy_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V4.0 · 动态权重深度重构版】
        基于幻方量化A股交易经验，深度重构动态权重计算
        核心理念：
        1. 多维度市场状态诊断：趋势、波动、情绪、流动性、资金、筹码、风险
        2. 非线性权重映射：使用S型函数进行非线性权重调整
        3. 相位适应性：根据市场不同阶段（启动、主升、调整、反转）动态调整权重
        4. 风险平价优化：考虑各维度对最终信号贡献的边际效用
        5. 记忆增强：结合历史状态进行平滑过渡
        """
        self._probe_print("=== 动态权重计算开始 ===")
        # 1. 计算七维度市场状态因子
        market_state_factors = self._calculate_market_state_factors(df_index, normalized_signals, mtf_signals)
        # 2. 市场阶段识别
        market_phase = self._identify_market_phase(df_index, market_state_factors, normalized_signals)
        # 3. 维度间相关性分析
        dimension_correlations = self._analyze_dimension_correlations(df_index, normalized_signals)
        # 4. 计算基础权重（基于市场阶段和状态）
        base_weights = self._calculate_base_weights(df_index, market_phase, market_state_factors)
        # 5. 应用非线性映射（Sigmoid函数调整）
        adjusted_weights = self._apply_nonlinear_weight_mapping(df_index, base_weights, market_state_factors)
        # 6. 风险平价优化（基于维度波动率）
        risk_parity_weights = self._apply_risk_parity_optimization(df_index, adjusted_weights, normalized_signals)
        # 7. 记忆平滑（避免权重剧烈变化）
        smoothed_weights = self._apply_memory_smoothing(df_index, risk_parity_weights)
        # 8. 最终归一化确保和为1
        final_weights = self._normalize_weights(df_index, smoothed_weights)
        # 探针输出
        self._probe_print(f"市场阶段分布: {market_phase.value_counts().to_dict()}")
        self._probe_print(f"最终权重均值 - 攻击性: {final_weights['aggressiveness'].mean():.4f}, "
                         f"控制力: {final_weights['control'].mean():.4f}, "
                         f"障碍清除: {final_weights['obstacle_clearance'].mean():.4f}, "
                         f"风险: {final_weights['risk'].mean():.4f}")
        self._probe_print("=== 动态权重计算完成 ===")
        return final_weights

    def _calculate_market_state_factors(self, df_index: pd.Index, 
                                       normalized_signals: Dict[str, pd.Series],
                                       mtf_signals: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V4.0】七维度市场状态因子计算
        核心理念：综合七个维度全面评估市场状态
        数学模型：主成分分析思想 + 加权综合评分
        """
        factors = {}
        # 1. 趋势状态因子（权重25%）
        trend_components = {
            'mtf_price_trend': mtf_signals.get('mtf_price_trend', pd.Series(0.5, index=df_index)),
            'uptrend_strength_norm': normalized_signals.get('uptrend_strength_norm', pd.Series(0.5, index=df_index)),
            'downtrend_strength_norm': 1 - normalized_signals.get('downtrend_strength_norm', pd.Series(0.5, index=df_index)),
            'trend_confirmation_norm': normalized_signals.get('trend_confirmation_norm', pd.Series(0.5, index=df_index)),
            'ADX_norm': normalized_signals.get('ADX_norm', pd.Series(0.5, index=df_index))
        }
        factors['trend_state'] = self._weighted_geometric_mean(trend_components, 
                                                             {'mtf_price_trend': 0.3, 
                                                              'uptrend_strength_norm': 0.25,
                                                              'downtrend_strength_norm': 0.2,
                                                              'trend_confirmation_norm': 0.15,
                                                              'ADX_norm': 0.1})
        # 2. 波动状态因子（权重15%）
        volatility_components = {
            'ATR_norm': normalized_signals.get('ATR_norm', pd.Series(0.5, index=df_index)),
            'BBW_norm': normalized_signals.get('BBW_norm', pd.Series(0.5, index=df_index)),
            'chip_entropy_norm': normalized_signals.get('chip_entropy_norm', pd.Series(0.5, index=df_index)),
            'flow_volatility_norm': normalized_signals.get('flow_volatility_norm', pd.Series(0.5, index=df_index))
        }
        # 高波动对应低因子值（1-波动）
        factors['volatility_state'] = 1 - self._weighted_geometric_mean(volatility_components, 
                                                                      {'ATR_norm': 0.4,
                                                                       'BBW_norm': 0.3,
                                                                       'chip_entropy_norm': 0.2,
                                                                       'flow_volatility_norm': 0.1})
        # 3. 情绪状态因子（权重15%）
        sentiment_components = {
            'market_sentiment_norm': normalized_signals.get('market_sentiment_norm', pd.Series(0.5, index=df_index)),
            'industry_leader_norm': normalized_signals.get('industry_leader_norm', pd.Series(0.5, index=df_index)),
            'industry_breadth_norm': normalized_signals.get('industry_breadth_norm', pd.Series(0.5, index=df_index)),
            'accumulation_score_norm': normalized_signals.get('accumulation_score_norm', pd.Series(0.5, index=df_index))
        }
        factors['sentiment_state'] = self._weighted_geometric_mean(sentiment_components,
                                                                 {'market_sentiment_norm': 0.4,
                                                                  'industry_leader_norm': 0.3,
                                                                  'industry_breadth_norm': 0.2,
                                                                  'accumulation_score_norm': 0.1})
        # 4. 流动性状态因子（权重15%）
        liquidity_components = {
            'volume_ratio_norm': normalized_signals.get('volume_ratio_norm', pd.Series(0.5, index=df_index)),
            'turnover_rate_norm': normalized_signals.get('turnover_rate_norm', pd.Series(0.5, index=df_index)),
            'net_amount_ratio_norm': normalized_signals.get('net_amount_ratio_norm', pd.Series(0.5, index=df_index)),
            'flow_intensity_norm': normalized_signals.get('flow_intensity_norm', pd.Series(0.5, index=df_index))
        }
        factors['liquidity_state'] = self._weighted_geometric_mean(liquidity_components,
                                                                 {'volume_ratio_norm': 0.35,
                                                                  'turnover_rate_norm': 0.25,
                                                                  'net_amount_ratio_norm': 0.25,
                                                                  'flow_intensity_norm': 0.15})
        # 5. 资金状态因子（权重15%）
        capital_components = {
            'net_mf_amount_norm': normalized_signals.get('net_mf_amount_norm', pd.Series(0.5, index=df_index)),
            'buy_elg_amount_norm': normalized_signals.get('buy_elg_amount_norm', pd.Series(0.5, index=df_index)),
            'buy_lg_amount_norm': normalized_signals.get('buy_lg_amount_norm', pd.Series(0.5, index=df_index)),
            'flow_acceleration_norm': normalized_signals.get('flow_acceleration_norm', pd.Series(0.5, index=df_index)),
            'flow_persistence_norm': normalized_signals.get('flow_persistence_norm', pd.Series(0.5, index=df_index))
        }
        factors['capital_state'] = self._weighted_geometric_mean(capital_components,
                                                               {'net_mf_amount_norm': 0.3,
                                                                'buy_elg_amount_norm': 0.25,
                                                                'buy_lg_amount_norm': 0.2,
                                                                'flow_acceleration_norm': 0.15,
                                                                'flow_persistence_norm': 0.1})
        # 6. 筹码状态因子（权重10%）
        chip_components = {
            'chip_concentration_norm': normalized_signals.get('chip_concentration_ratio_norm', pd.Series(0.5, index=df_index)),
            'chip_convergence_norm': normalized_signals.get('chip_convergence_ratio_norm', pd.Series(0.5, index=df_index)),
            'chip_stability_norm': normalized_signals.get('chip_stability_norm', pd.Series(0.5, index=df_index)),
            'chip_flow_direction_norm': normalized_signals.get('chip_flow_direction_norm', pd.Series(0.5, index=df_index))
        }
        factors['chip_state'] = self._weighted_geometric_mean(chip_components,
                                                            {'chip_concentration_norm': 0.35,
                                                             'chip_convergence_norm': 0.3,
                                                             'chip_stability_norm': 0.25,
                                                             'chip_flow_direction_norm': 0.1})
        # 7. 风险状态因子（权重5%）
        risk_components = {
            'breakout_risk_norm': 1 - normalized_signals.get('breakout_risk_warning_norm', pd.Series(0.5, index=df_index)),
            'reversal_risk_norm': 1 - normalized_signals.get('reversal_warning_score_norm', pd.Series(0.5, index=df_index)),
            'distribution_score_norm': 1 - normalized_signals.get('distribution_score_norm', pd.Series(0.5, index=df_index))
        }
        factors['risk_state'] = self._weighted_geometric_mean(risk_components,
                                                            {'breakout_risk_norm': 0.4,
                                                             'reversal_risk_norm': 0.35,
                                                             'distribution_score_norm': 0.25})
        return factors

    def _identify_market_phase(self, df_index: pd.Index, market_state_factors: Dict[str, pd.Series],
                              normalized_signals: Dict[str, pd.Series]) -> pd.Series:
        """
        【V4.0】市场阶段识别算法
        核心理念：识别市场处于启动、主升、调整、反转等不同阶段
        数学模型：模糊逻辑分类 + 状态转移检测
        """
        phase_labels = pd.Series("未知", index=df_index)
        # 提取关键因子
        trend_state = market_state_factors.get('trend_state', pd.Series(0.5, index=df_index))
        volatility_state = market_state_factors.get('volatility_state', pd.Series(0.5, index=df_index))
        sentiment_state = market_state_factors.get('sentiment_state', pd.Series(0.5, index=df_index))
        capital_state = market_state_factors.get('capital_state', pd.Series(0.5, index=df_index))
        # 获取价格动量
        price_momentum = normalized_signals.get('RSI_norm', pd.Series(0.5, index=df_index))
        macd_signal = normalized_signals.get('MACDs_norm', pd.Series(0.5, index=df_index))
        # 模糊逻辑规则集
        for i in range(len(df_index)):
            if i < 10:  # 需要足够数据
                phase_labels.iloc[i] = "未知"
                continue
            # 规则1: 启动阶段（趋势初现，情绪回暖，资金开始流入）
            if (trend_state.iloc[i] > 0.6 and 
                sentiment_state.iloc[i] > 0.55 and 
                capital_state.iloc[i] > 0.55 and
                volatility_state.iloc[i] < 0.4):
                phase_labels.iloc[i] = "启动"
            # 规则2: 主升阶段（强趋势，高情绪，大资金流入）
            elif (trend_state.iloc[i] > 0.7 and 
                  sentiment_state.iloc[i] > 0.7 and 
                  capital_state.iloc[i] > 0.7 and
                  price_momentum.iloc[i] > 0.6 and
                  macd_signal.iloc[i] > 0.5):
                phase_labels.iloc[i] = "主升"
            # 规则3: 调整阶段（趋势减弱，波动增大，资金犹豫）
            elif (0.4 < trend_state.iloc[i] <= 0.6 and 
                  volatility_state.iloc[i] > 0.5 and
                  capital_state.iloc[i] < 0.5):
                phase_labels.iloc[i] = "调整"
            # 规则4: 反转风险（趋势转弱，情绪降温，资金流出）
            elif (trend_state.iloc[i] < 0.4 and 
                  sentiment_state.iloc[i] < 0.4 and 
                  capital_state.iloc[i] < 0.4):
                phase_labels.iloc[i] = "反转风险"
            # 规则5: 横盘整理（低波动，趋势不明确）
            elif (0.4 <= trend_state.iloc[i] <= 0.6 and 
                  volatility_state.iloc[i] < 0.3 and
                  capital_state.iloc[i] < 0.5):
                phase_labels.iloc[i] = "横盘"
            # 规则6: 反弹阶段（从低点回升，资金重新流入）
            elif (trend_state.iloc[i] > 0.5 and 
                  trend_state.iloc[i-5:i].min() < 0.3 and  # 近期有过低点
                  capital_state.iloc[i] > capital_state.iloc[i-5:i].max()):  # 资金创新高
                phase_labels.iloc[i] = "反弹"
            else:
                phase_labels.iloc[i] = "过渡"
        # 状态平滑（避免频繁切换）
        smoothed_phases = pd.Series("未知", index=df_index)
        window = 5
        for i in range(window, len(df_index)):
            recent_phases = phase_labels.iloc[i-window:i]
            # 取最近窗口内最频繁的状态
            phase_counts = recent_phases.value_counts()
            if not phase_counts.empty:
                smoothed_phases.iloc[i] = phase_counts.idxmax()
        # 前向填充初始值
        smoothed_phases = smoothed_phases.ffill()
        return smoothed_phases

    def _analyze_dimension_correlations(self, df_index: pd.Index, normalized_signals: Dict[str, pd.Series]) -> pd.Series:
        """
        【V4.0】维度间相关性分析
        核心理念：分析各维度信号间的相关性，避免多重共线性
        数学模型：滚动窗口相关性分析 + 方差膨胀因子简化版
        """
        # 定义各维度代表性信号
        dimension_signals = {
            'aggressiveness': ['RSI_norm', 'MACD_norm', 'flow_acceleration_norm', 'volume_ratio_norm'],
            'control': ['chip_concentration_ratio_norm', 'chip_stability_norm', 'flow_persistence_norm'],
            'obstacle_clearance': ['turnover_rate_norm', 'net_amount_ratio_norm', 'flow_intensity_norm'],
            'risk': ['distribution_score_norm', 'breakout_risk_warning_norm', 'reversal_warning_score_norm']
        }
        # 计算各维度综合得分（用于相关性分析）
        dimension_scores = {}
        for dim, signals in dimension_signals.items():
            available_signals = {}
            for sig in signals:
                if sig in normalized_signals:
                    available_signals[sig] = normalized_signals[sig]
            if available_signals:
                # 等权重简单平均
                dim_score = pd.Series(0.0, index=df_index)
                for series in available_signals.values():
                    dim_score += series
                dimension_scores[dim] = dim_score / len(available_signals)
        # 计算滚动相关性矩阵的迹（简单相关性度量）
        correlation_score = pd.Series(0.5, index=df_index)  # 默认中等相关性
        if len(dimension_scores) >= 2:
            window = 21
            for i in range(window, len(df_index)):
                if i < window:
                    continue
                # 构建维度得分矩阵
                dim_data = {}
                for dim, score_series in dimension_scores.items():
                    dim_data[dim] = score_series.iloc[i-window:i].values
                # 计算相关性矩阵
                corr_matrix = np.zeros((len(dim_data), len(dim_data)))
                dim_names = list(dim_data.keys())
                for j in range(len(dim_names)):
                    for k in range(len(dim_names)):
                        if j == k:
                            corr_matrix[j, k] = 1.0
                        else:
                            series1 = dim_data[dim_names[j]]
                            series2 = dim_data[dim_names[k]]
                            if len(series1) >= 5 and len(series2) >= 5:
                                corr = np.corrcoef(series1, series2)[0, 1]
                                if not np.isnan(corr):
                                    corr_matrix[j, k] = abs(corr)  # 取绝对值
                # 计算平均非对角相关性（维度间相关性）
                if len(dim_names) > 1:
                    non_diag_sum = np.sum(corr_matrix) - np.trace(corr_matrix)
                    non_diag_count = len(dim_names) * (len(dim_names) - 1)
                    avg_correlation = non_diag_sum / non_diag_count if non_diag_count > 0 else 0
                    # 高相关性（>0.7）表示维度冗余，需要调整权重
                    correlation_score.iloc[i] = avg_correlation
        correlation_score = correlation_score.ffill().fillna(0.5)
        return correlation_score

    def _calculate_base_weights(self, df_index: pd.Index, market_phase: pd.Series,
                               market_state_factors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V4.0】基础权重计算（基于市场阶段）
        核心理念：不同市场阶段需要不同的维度权重配置
        数学模型：状态依赖权重矩阵 + 因子调节
        """
        base_weights = {
            'aggressiveness': pd.Series(0.3, index=df_index),
            'control': pd.Series(0.3, index=df_index),
            'obstacle_clearance': pd.Series(0.2, index=df_index),
            'risk': pd.Series(0.2, index=df_index)
        }
        # 阶段依赖的权重配置（专家经验）
        phase_weight_configs = {
            # 阶段: (攻击性, 控制力, 障碍清除, 风险)
            '启动': (0.35, 0.30, 0.20, 0.15),
            '主升': (0.40, 0.25, 0.20, 0.15),
            '调整': (0.25, 0.35, 0.15, 0.25),
            '反转风险': (0.20, 0.30, 0.10, 0.40),
            '横盘': (0.25, 0.35, 0.20, 0.20),
            '反弹': (0.30, 0.30, 0.25, 0.15),
            '过渡': (0.30, 0.30, 0.20, 0.20),
            '未知': (0.30, 0.30, 0.20, 0.20)
        }
        # 根据市场阶段应用基础权重
        for i in range(len(df_index)):
            phase = market_phase.iloc[i]
            weights = phase_weight_configs.get(phase, (0.3, 0.3, 0.2, 0.2))
            base_weights['aggressiveness'].iloc[i] = weights[0]
            base_weights['control'].iloc[i] = weights[1]
            base_weights['obstacle_clearance'].iloc[i] = weights[2]
            base_weights['risk'].iloc[i] = weights[3]
        # 根据市场状态因子微调
        trend_state = market_state_factors.get('trend_state', pd.Series(0.5, index=df_index))
        volatility_state = market_state_factors.get('volatility_state', pd.Series(0.5, index=df_index))
        capital_state = market_state_factors.get('capital_state', pd.Series(0.5, index=df_index))
        for i in range(len(df_index)):
            # 趋势越强，攻击性权重越高
            trend_adj = (trend_state.iloc[i] - 0.5) * 0.2  # ±10%调整
            base_weights['aggressiveness'].iloc[i] *= (1 + trend_adj)
            # 波动越低，控制力权重越高
            vol_adj = (0.5 - volatility_state.iloc[i]) * 0.15  # ±7.5%调整
            base_weights['control'].iloc[i] *= (1 + vol_adj)
            # 资金流入越强，障碍清除权重越高
            capital_adj = (capital_state.iloc[i] - 0.5) * 0.15  # ±7.5%调整
            base_weights['obstacle_clearance'].iloc[i] *= (1 + capital_adj)
            # 风险权重与波动状态负相关
            risk_adj = (0.5 - volatility_state.iloc[i]) * 0.1  # ±5%调整
            base_weights['risk'].iloc[i] *= (1 + risk_adj)
        # 确保权重非负
        for dim in base_weights:
            base_weights[dim] = base_weights[dim].clip(0.05, 0.6)
        return base_weights

    def _apply_nonlinear_weight_mapping(self, df_index: pd.Index, base_weights: Dict[str, pd.Series],
                                       market_state_factors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V4.0】非线性权重映射
        核心理念：使用S型函数进行非线性权重调整，增强极端状态的权重响应
        数学模型：Sigmoid函数 + 自适应阈值
        """
        adjusted_weights = {}
        # 提取关键市场状态
        trend_state = market_state_factors.get('trend_state', pd.Series(0.5, index=df_index))
        composite_state = pd.Series(0.0, index=df_index)
        for factor in market_state_factors.values():
            composite_state += factor
        composite_state /= len(market_state_factors)  # 平均状态
        # Sigmoid函数参数
        k = 3.0  # 陡峭度
        x0 = 0.6  # 中心点
        for dim, weight_series in base_weights.items():
            adjusted = pd.Series(0.0, index=df_index)
            for i in range(len(df_index)):
                base_w = weight_series.iloc[i]
                comp_state = composite_state.iloc[i]
                if dim == 'aggressiveness':
                    # 攻击性：市场状态越好，权重增加越快（S型上升）
                    sigmoid_factor = 1 / (1 + np.exp(-k * (comp_state - x0)))
                    adjusted.iloc[i] = base_w * (0.8 + 0.4 * sigmoid_factor)  # 0.8x-1.2x
                elif dim == 'control':
                    # 控制力：中等状态时最高，极端状态时降低（倒U型）
                    control_factor = 4 * comp_state * (1 - comp_state)  # 二次函数，顶点在0.5
                    adjusted.iloc[i] = base_w * (0.9 + 0.2 * control_factor)  # 0.9x-1.1x
                elif dim == 'obstacle_clearance':
                    # 障碍清除：与趋势状态正相关
                    trend_factor = trend_state.iloc[i]
                    obstacle_factor = 0.9 + 0.2 * trend_factor  # 0.9x-1.1x
                    adjusted.iloc[i] = base_w * obstacle_factor
                elif dim == 'risk':
                    # 风险：市场状态越差，权重增加越快（反向S型）
                    sigmoid_factor = 1 / (1 + np.exp(-k * ((1 - comp_state) - x0)))
                    adjusted.iloc[i] = base_w * (0.8 + 0.4 * sigmoid_factor)  # 0.8x-1.2x
            adjusted_weights[dim] = adjusted.clip(0.05, 0.6)
        return adjusted_weights

    def _apply_risk_parity_optimization(self, df_index: pd.Index, adjusted_weights: Dict[str, pd.Series],
                                       normalized_signals: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V4.0】风险平价优化
        核心理念：使各维度对最终信号的风险贡献相等
        数学模型：简化版风险平价（基于波动率调整）
        """
        # 估计各维度信号的波动率（21日滚动标准差）
        dimension_volatility = {}
        # 各维度代表性信号的波动率
        dimension_representative_signals = {
            'aggressiveness': ['RSI_norm', 'MACD_norm'],
            'control': ['chip_stability_norm', 'flow_persistence_norm'],
            'obstacle_clearance': ['volume_ratio_norm', 'net_amount_ratio_norm'],
            'risk': ['breakout_risk_warning_norm', 'reversal_warning_score_norm']
        }
        for dim, signals in dimension_representative_signals.items():
            vol_series = pd.Series(0.0, index=df_index)
            count = 0
            for sig_name in signals:
                if sig_name in normalized_signals:
                    # 计算21日滚动波动率
                    sig_vol = normalized_signals[sig_name].rolling(window=21, min_periods=10).std().fillna(0.1)
                    vol_series += sig_vol
                    count += 1
            if count > 0:
                dimension_volatility[dim] = vol_series / count
            else:
                dimension_volatility[dim] = pd.Series(0.1, index=df_index)  # 默认波动率
        # 风险平价权重调整
        risk_parity_weights = {}
        for dim in adjusted_weights.keys():
            # 初始化
            risk_parity_weights[dim] = pd.Series(0.0, index=df_index)
        # 逐日计算风险平价权重
        for i in range(len(df_index)):
            if i < 10:  # 需要足够数据
                for dim in adjusted_weights.keys():
                    risk_parity_weights[dim].iloc[i] = adjusted_weights[dim].iloc[i]
                continue
            # 收集当前各维度的波动率
            current_vols = {}
            for dim in adjusted_weights.keys():
                current_vols[dim] = dimension_volatility[dim].iloc[i]
                if current_vols[dim] < 0.01:  # 避免除零
                    current_vols[dim] = 0.01
            # 计算风险贡献
            risk_contributions = {}
            total_risk_contribution = 0
            for dim in adjusted_weights.keys():
                # 风险贡献 ≈ 权重 × 波动率
                risk_contrib = adjusted_weights[dim].iloc[i] * current_vols[dim]
                risk_contributions[dim] = risk_contrib
                total_risk_contribution += risk_contrib
            if total_risk_contribution > 0:
                # 计算目标风险贡献（平均）
                target_risk_contrib = total_risk_contribution / len(adjusted_weights)
                # 调整权重使风险贡献相等
                for dim in adjusted_weights.keys():
                    if current_vols[dim] > 0:
                        # 新权重 = 目标风险贡献 / 波动率
                        new_weight = target_risk_contrib / current_vols[dim]
                        risk_parity_weights[dim].iloc[i] = new_weight
                    else:
                        risk_parity_weights[dim].iloc[i] = adjusted_weights[dim].iloc[i]
            else:
                for dim in adjusted_weights.keys():
                    risk_parity_weights[dim].iloc[i] = adjusted_weights[dim].iloc[i]
        # 限制权重范围
        for dim in risk_parity_weights:
            risk_parity_weights[dim] = risk_parity_weights[dim].clip(0.05, 0.6)
        return risk_parity_weights

    def _apply_memory_smoothing(self, df_index: pd.Index, risk_parity_weights: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V4.0】记忆平滑算法
        核心理念：避免权重剧烈变化，保持稳定性
        数学模型：指数加权移动平均 + 变化率限制
        """
        smoothed_weights = {}
        for dim, weight_series in risk_parity_weights.items():
            # 1. 应用指数加权移动平均（EWMA）
            alpha = 0.3  # 平滑因子，越小越平滑
            ewma_weights = weight_series.ewm(alpha=alpha, adjust=False).mean()
            # 2. 变化率限制（每日变化不超过20%）
            limited_weights = pd.Series(0.0, index=df_index)
            if len(ewma_weights) > 0:
                limited_weights.iloc[0] = ewma_weights.iloc[0]
                for i in range(1, len(ewma_weights)):
                    prev_weight = limited_weights.iloc[i-1]
                    target_weight = ewma_weights.iloc[i]
                    # 计算最大允许变化
                    max_change = prev_weight * 0.2  # 20%变化限制
                    # 限制变化
                    if target_weight > prev_weight + max_change:
                        limited_weights.iloc[i] = prev_weight + max_change
                    elif target_weight < prev_weight - max_change:
                        limited_weights.iloc[i] = prev_weight - max_change
                    else:
                        limited_weights.iloc[i] = target_weight
            smoothed_weights[dim] = limited_weights.clip(0.05, 0.6)
        return smoothed_weights

    def _normalize_weights(self, df_index: pd.Index, weights_dict: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V4.0】权重归一化确保和为1
        """
        normalized_weights = {}
        for i in range(len(df_index)):
            # 收集当前各维度权重
            current_weights = {}
            total = 0
            for dim, weight_series in weights_dict.items():
                current_weight = weight_series.iloc[i]
                current_weights[dim] = current_weight
                total += current_weight
            # 归一化
            if total > 0:
                for dim in current_weights:
                    if dim not in normalized_weights:
                        normalized_weights[dim] = pd.Series(0.0, index=df_index)
                    normalized_weights[dim].iloc[i] = current_weights[dim] / total
            else:
                # 默认权重
                for dim in weights_dict.keys():
                    if dim not in normalized_weights:
                        normalized_weights[dim] = pd.Series(0.0, index=df_index)
                    normalized_weights[dim].iloc[i] = 0.25  # 均分
        return normalized_weights

    def _calculate_aggressiveness_score(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], dynamic_weights: Dict[str, pd.Series]) -> pd.Series:
        """
        【V5.0 · 冲击增强版】将JERK突变引入攻击性评分逻辑
        核心理念：当JERK(加加速度)出现正向偏离时，代表主力意图从“维护趋势”转为“暴力攻击”。
        """
        base_aggressiveness = self._calculate_basic_aggressiveness(df_index, mtf_signals, normalized_signals)
        # 提取JERK冲击因子 (以5日和13日价格/资金流为主)
        price_jerk = mtf_signals.get('JERK_5_price_trend', pd.Series(0.0, index=df_index))
        money_jerk = mtf_signals.get('JERK_5_net_amount_trend', pd.Series(0.0, index=df_index))
        # 归一化冲击因子
        jerk_impact = (normalize_score(price_jerk) * 0.6 + normalize_score(money_jerk) * 0.4).clip(0, 1)
        # 非线性增强：当Jerk爆发时，指数级提升攻击性分值
        aggressiveness_score = (base_aggressiveness * (1 + jerk_impact * 0.5)).clip(0, 1)
        self._probe_print(f"攻击性得分(含JERK冲击): {aggressiveness_score.mean():.4f}")
        return aggressiveness_score

    def _calculate_basic_aggressiveness(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series]) -> pd.Series:
        """
        【V5.0】基础攻击性计算逻辑拆分
        """
        aggressiveness_components = {
            "price_trend": mtf_signals['mtf_price_trend'].clip(lower=0),
            "volume_trend": mtf_signals['mtf_volume_trend'].clip(lower=0),
            "net_amount_trend": mtf_signals['mtf_net_amount_trend'].clip(lower=0),
            "flow_acceleration": normalized_signals['flow_acceleration_norm'].clip(lower=0),
            "breakout_quality": normalized_signals['breakout_quality_norm'].clip(lower=0),
            "accumulation_score": normalized_signals['accumulation_score_norm'].clip(lower=0)
        }
        aggressiveness_weights = {
            "price_trend": 0.20, "volume_trend": 0.15, "net_amount_trend": 0.20,
            "flow_acceleration": 0.15, "breakout_quality": 0.15, "accumulation_score": 0.15
        }
        return _robust_geometric_mean(aggressiveness_components, aggressiveness_weights, df_index)

    def _detect_bear_trap(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series]) -> pd.Series:
        """
        【V5.0 · 诱空陷阱检测】识别“动量底背离”与“筹码锁死”的共振点
        判定逻辑：加速度 < 0 (惯性下跌/减速) 且 加加速度 > 0 (反弹冲击力现) + 筹码稳定性高
        """
        price_accel = mtf_signals.get('ACCEL_5_price_trend', pd.Series(0.0, index=df_index))
        price_jerk = mtf_signals.get('JERK_5_price_trend', pd.Series(0.0, index=df_index))
        chip_stability = normalized_signals.get('chip_stability_norm', pd.Series(0.5, index=df_index))
        # 识别动量拐点：加速度虽负但在暴力修正(JERK > 0)
        momentum_reversal = (price_accel < 0) & (price_jerk > 0)
        # 结合筹码稳定性：主力未离场
        bear_trap_signal = momentum_reversal.astype(float) * chip_stability
        # 平滑并归一化
        bear_trap_score = bear_trap_signal.rolling(window=3).mean().fillna(0).clip(0, 1)
        self._probe_print(f"诱空陷阱识别均值: {bear_trap_score.mean():.4f}")
        return bear_trap_score

    def _calculate_volume_price_divergence(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series]) -> pd.Series:
        """
        【V5.0 · 缩量背离侦测】识别“虚假拉升”风险
        判定逻辑：价格斜率 > 0 (上涨) 但 成交量斜率 < 0 (缩量)
        """
        price_slope = mtf_signals.get('SLOPE_5_price_trend', pd.Series(0.0, index=df_index))
        vol_slope = mtf_signals.get('SLOPE_5_volume_trend', pd.Series(0.0, index=df_index))
        # 提取量价背离特征
        divergence = (price_slope > 0) & (vol_slope < 0)
        # 计算背离强度：价格涨得越猛，量缩得越厉害，风险越大
        divergence_intensity = (price_slope.clip(lower=0) * vol_slope.clip(upper=0).abs()).pow(0.5)
        divergence_score = divergence_intensity.mask(~divergence, 0.0)
        # 映射到[0, 1]区间
        return (divergence_score / divergence_score.max()).fillna(0) if not divergence_score.empty else divergence_score

    def _calculate_acceleration_resonance(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series]) -> pd.Series:
        """
        【V5.0 · 加速度共振】捕捉短中长周期的动能一致性
        判定逻辑：Fib(5, 13, 21)周期加速度方向一致
        """
        accel_5 = mtf_signals.get('ACCEL_5_price_trend', pd.Series(0.0, index=df_index))
        accel_13 = mtf_signals.get('ACCEL_13_price_trend', pd.Series(0.0, index=df_index))
        accel_21 = mtf_signals.get('ACCEL_21_price_trend', pd.Series(0.0, index=df_index))
        # 计算方向一致性系数
        resonance_count = (accel_5 > 0).astype(int) + (accel_13 > 0).astype(int) + (accel_21 > 0).astype(int)
        # 3级共振：满分；2级共振：基础分
        resonance_score = resonance_count.map({3: 1.0, 2: 0.6, 1: 0.2, 0: 0.0})
        # 引入量纲：共振时的绝对加速度强度
        avg_accel = (accel_5.clip(lower=0) + accel_13.clip(lower=0) + accel_21.clip(lower=0)) / 3
        final_resonance = (resonance_score * (1 + avg_accel)).clip(0, 1)
        self._probe_print(f"加速度共振强度均值: {final_resonance.mean():.4f}")
        return final_resonance

    def _calculate_control_score(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], historical_context: Dict[str, pd.Series]) -> pd.Series:
        """
        【V2.0】计算控制力分数 - 基于新指标
        """
        control_components = {
            "chip_concentration": normalized_signals['chip_concentration_ratio_norm'].clip(lower=0),
            "chip_convergence": normalized_signals['chip_convergence_ratio_norm'].clip(lower=0),
            "chip_stability": normalized_signals['chip_stability_norm'].clip(lower=0),
            "flow_consistency": normalized_signals['flow_consistency_norm'].clip(lower=0),
            "flow_stability": normalized_signals['flow_stability_norm'].clip(lower=0),
            "inflow_persistence": normalized_signals['inflow_persistence_norm'].clip(lower=0),
            "trend_confirmation": normalized_signals['trend_confirmation_norm'].clip(lower=0),
            "breakout_confidence": normalized_signals['breakout_confidence_norm'].clip(lower=0),
            "behavior_accumulation": normalized_signals['behavior_accumulation_norm'].clip(lower=0)
        }
        control_weights = {
            "chip_concentration": 0.15, "chip_convergence": 0.12, "chip_stability": 0.12,
            "flow_consistency": 0.10, "flow_stability": 0.10, "inflow_persistence": 0.10,
            "trend_confirmation": 0.10, "breakout_confidence": 0.11, "behavior_accumulation": 0.10
        }
        control_score = _robust_geometric_mean(control_components, control_weights, df_index).clip(0, 1)
        # 应用筹码集中度稳定性调节器
        if historical_context['hc_enabled']:
            chip_stability_modulator = 0.05
            control_score = (control_score * (1 + historical_context['mtf_chip_concentration_stability'] * chip_stability_modulator)).clip(0, 1)
        return control_score

    def _calculate_obstacle_clearance_score(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series]) -> pd.Series:
        """
        【V2.0】计算障碍清除分数 - 基于新指标
        """
        obstacle_clearance_components = {
            "volume_ratio": normalized_signals['volume_ratio_norm'].clip(lower=0),
            "turnover_rate": normalized_signals['turnover_rate_norm'].clip(lower=0),
            "net_amount_ratio": normalized_signals['net_amount_ratio_norm'].clip(lower=0),
            "flow_momentum": normalized_signals['flow_momentum_5d_norm'].clip(lower=0),
            "chip_flow_intensity": normalized_signals['chip_flow_intensity_norm'].clip(lower=0),
            "breakout_potential": normalized_signals['breakout_potential_norm'].clip(lower=0),
            "behavior_consolidation": normalized_signals['behavior_consolidation_norm'].clip(lower=0) if 'behavior_consolidation_norm' in normalized_signals else pd.Series(0.5, index=df_index),
            "absolute_change": normalized_signals['absolute_change_strength_norm'].clip(lower=0)
        }
        obstacle_clearance_weights = {
            "volume_ratio": 0.15, "turnover_rate": 0.15, "net_amount_ratio": 0.15,
            "flow_momentum": 0.15, "chip_flow_intensity": 0.10,
            "breakout_potential": 0.15, "behavior_consolidation": 0.10,
            "absolute_change": 0.05
        }
        obstacle_clearance_score = _robust_geometric_mean(obstacle_clearance_components, obstacle_clearance_weights, df_index).clip(0, 1)
        return obstacle_clearance_score

    def _synthesize_bullish_intent(self, df_index: pd.Index, aggressiveness_score: pd.Series, control_score: pd.Series, 
                                  obstacle_clearance_score: pd.Series, mtf_signals: Dict[str, pd.Series], 
                                  normalized_signals: Dict[str, pd.Series], dynamic_weights: Dict[str, pd.Series],
                                  historical_context: Dict[str, pd.Series], rally_intent_synthesis_params: Dict) -> pd.Series:
        """
        【V5.0 · 运动学增强版】集成诱空与共振逻辑
        """
        # 计算新增的博弈特征
        bear_trap_bonus = self._detect_bear_trap(df_index, mtf_signals, normalized_signals)
        acceleration_resonance = self._calculate_acceleration_resonance(df_index, mtf_signals)
        # 基础意图合成 (由攻击性、控制力、障碍清除构成)
        bullish_intent_base = (
            (aggressiveness_score * dynamic_weights["aggressiveness"] +
             control_score * dynamic_weights["control"] +
             obstacle_clearance_score * dynamic_weights["obstacle_clearance"]) /
            (dynamic_weights["aggressiveness"] + dynamic_weights["control"] + dynamic_weights["obstacle_clearance"])
        )
        # 动态增强：诱空成功后的报复性拉升与多周期共振爆发
        # 增强系数：加速共振贡献30%，诱空加成20%
        enhanced_intent = (bullish_intent_base * 0.5 + acceleration_resonance * 0.3 + bear_trap_bonus * 0.2).clip(0, 1)
        # 应用长期趋势调节
        long_term_trend_modulator = 1.0
        if historical_context.get('hc_enabled'):
            long_term_trend_modulator = (1 + historical_context.get('integrated_memory', 0) * 0.1)
        return (enhanced_intent * long_term_trend_modulator).clip(0, 1)

    def _calculate_bearish_intent(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], 
                                 mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series],
                                 historical_context: Dict[str, pd.Series]) -> pd.Series:
        """
        【V2.0】计算看跌意图 - 基于新指标
        """
        # 主力资金累计记忆调节器
        mf_flow_memory_anti_bearish_modulator = 1.0
        if historical_context['hc_enabled']:
            cumulative_mf_flow_modulator_factor = 0.1
            mf_flow_memory_anti_bearish_modulator = (1 - historical_context['mtf_cumulative_mf_flow'].clip(lower=0) * cumulative_mf_flow_modulator_factor).clip(0, 1)
        # 看跌意图成分
        bearish_score_components = {
            "distribution_score": normalized_signals['distribution_score_norm'].clip(lower=0) if 'distribution_score_norm' in normalized_signals else pd.Series(0.0, index=df_index),
            "behavior_distribution": normalized_signals['behavior_distribution_norm'].clip(lower=0) if 'behavior_distribution_norm' in normalized_signals else pd.Series(0.0, index=df_index),
            "downtrend_strength": normalized_signals['downtrend_strength_norm'].clip(lower=0),
            "chip_divergence": normalized_signals['chip_divergence_ratio_norm'].clip(lower=0) if 'chip_divergence_ratio_norm' in normalized_signals else pd.Series(0.0, index=df_index),
            "price_decline": raw_signals['pct_change'].clip(upper=0).abs(),
            "RSI_overbought": (normalized_signals['RSI_norm'] - 0.7).clip(lower=0) if 'RSI_norm' in normalized_signals else pd.Series(0.0, index=df_index)
        }
        bearish_weights = {
            "distribution_score": 0.25, "behavior_distribution": 0.20,
            "downtrend_strength": 0.25, "chip_divergence": 0.15,
            "price_decline": 0.10, "RSI_overbought": 0.05
        }
        # 归一化权重
        total_weight = sum(bearish_weights.values())
        bearish_weights = {k: v / total_weight for k, v in bearish_weights.items()}
        bearish_score_raw = _robust_geometric_mean(bearish_score_components, bearish_weights, df_index).clip(0, 1)
        # 应用主力资金累计记忆调节
        bearish_score_modulated = (bearish_score_raw * mf_flow_memory_anti_bearish_modulator).clip(0, 1)
        # 转换为负值
        bearish_score = -bearish_score_modulated
        return bearish_score

    def _adjudicate_risk(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], 
                        mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series],
                        dynamic_weights: Dict[str, pd.Series], aggressiveness_score: pd.Series,
                        rally_intent_synthesis_params: Dict) -> pd.Series:
        """
        【V5.0 · 背离惩罚版】引入量价背离作为核心风险减分项
        """
        # 计算量价背离惩罚
        vp_divergence_penalty = self._calculate_volume_price_divergence(df_index, mtf_signals)
        # 基础风险评估 (派发风险 + 技术风险)
        distribution_risk = normalized_signals.get('distribution_score_norm', pd.Series(0.0, index=df_index))
        technical_risk = (1 - normalized_signals.get('ADX_norm', pd.Series(0.5, index=df_index)))
        # 综合风险惩罚因子
        total_risk_penalty_raw = (
            distribution_risk * 0.4 +
            technical_risk * 0.3 +
            vp_divergence_penalty * 0.3  # 背离占风险权重的30%
        ).clip(0, 1)
        # 应用Sigmoid函数进行非线性惩罚
        risk_sensitivity = get_param_value(rally_intent_synthesis_params.get('risk_sensitivity'), 2.5)
        sigmoid_center = get_param_value(rally_intent_synthesis_params.get('sigmoid_center'), 0.6)
        total_risk_penalty = 1 / (1 + np.exp(risk_sensitivity * (total_risk_penalty_raw - sigmoid_center)))
        return (1 - total_risk_penalty).clip(0, 1)

    def _apply_contextual_modulators(self, df_index: pd.Index, final_rally_intent: pd.Series, 
                                    proxy_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series]) -> pd.Series:
        """
        【V2.0】应用情境调节器 - 基于新指标
        """
        # 相对强度和资本属性调节
        modulated_intent = final_rally_intent * proxy_signals['rs_modulator'] * proxy_signals['capital_modulator']
        # 市场情绪调节
        market_sentiment_modulator = (1 + proxy_signals['market_sentiment_proxy'] * 0.1)
        modulated_intent = modulated_intent * market_sentiment_modulator
        # 流动性调节
        liquidity_modulator = (1 + proxy_signals['liquidity_proxy'] * 0.05)
        modulated_intent = modulated_intent * liquidity_modulator
        # 筹码稳定性调节
        chip_stability_modulator = (1 + mtf_signals['mtf_chip_concentration_trend'].clip(lower=0) * 0.05)
        modulated_intent = modulated_intent * chip_stability_modulator
        # 限制范围
        modulated_intent = modulated_intent.clip(-1, 1)
        return modulated_intent

    def _output_probe_info(self, df_index: pd.Index, final_rally_intent: pd.Series):
        """
        【V2.0】输出探针信息
        """
        probe_ts = None
        if self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            for date in reversed(df_index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        if probe_ts:
            print(f"\n=== 主力拉升意图探针报告 @ {probe_ts.strftime('%Y-%m-%d')} ===")
            for line in self._probe_output:
                print(line)
            print(f"最终拉升意图分值: {final_rally_intent.loc[probe_ts]:.4f}")
            print("=== 探针报告结束 ===\n")