# 文件: strategies/trend_following/intelligence/process/calculate_main_force_rally_intent.py
# 主力 rally 意图计算 已完成
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
from strategies.trend_following.intelligence.process.rally_intent_modules.calculate_price_memory import CalculatePriceMemory
from strategies.trend_following.intelligence.process.rally_intent_modules.calculate_capital_memory import CalculateCapitalMemory
from strategies.trend_following.intelligence.process.rally_intent_modules.calculate_chip_memory import CalculateChipMemory
from strategies.trend_following.intelligence.process.rally_intent_modules.calculate_sentiment_memory import CalculateSentimentMemory
from strategies.trend_following.intelligence.process.rally_intent_modules.calculate_enhanced_rs_proxy import EnhancedRSProxyCalculator
from strategies.trend_following.intelligence.process.rally_intent_modules.calculate_enhanced_capital_proxy import EnhancedCapitalProxyCalculator
from strategies.trend_following.intelligence.process.rally_intent_modules.calculate_enhanced_sentiment_proxy import EnhancedSentimentProxyCalculator
from strategies.trend_following.intelligence.process.rally_intent_modules.calculate_enhanced_liquidity_proxy import EnhancedLiquidityProxyCalculator
from strategies.trend_following.intelligence.process.rally_intent_modules.calculate_enhanced_volatility_proxy import EnhancedVolatilityProxyCalculator
from strategies.trend_following.intelligence.process.rally_intent_modules.calculate_enhanced_risk_preference_proxy import EnhancedRiskPreferenceProxyCalculator
from strategies.trend_following.intelligence.process.rally_intent_modules.assess_signal_quality import SignalQualityAssessor

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
        【V9.0 · 工业级逻辑中枢】主力拉升意图主计算流程
        修改说明：重新梳理逻辑时序，确保记忆、代理、权重与风险模块的零延迟耦合。
        版本号：2025.02.07.01
        """
        self._probe_output = []
        params = self._get_parameters(config)
        df_index = df.index
        # 1. 数据完整性与基础信号准备
        raw_signals = self._get_raw_signals(df)
        is_limit_up_day = df.apply(lambda row: is_limit_up(row), axis=1)
        # 2. 信号预处理层 (Normalization & MTF)
        normalized_signals = self._normalize_raw_signals(df_index, raw_signals)
        mtf_signals = self._calculate_mtf_fused_signals(df, raw_signals, params['mtf_slope_accel_weights'], df_index)
        # 3. 历史上下文层 (Memory System)
        # 必须先计算动态周期，再注入历史上下文计算
        dynamic_memory_period = self._calculate_dynamic_period(df_index, raw_signals)
        historical_context = self._calculate_historical_context(df, df_index, raw_signals, params['historical_context_params'])
        historical_context['dynamic_memory_period'] = dynamic_memory_period
        # 4. 智能代理层 (Proxy Layer)
        proxy_signals = self._construct_proxy_signals(df_index, mtf_signals, normalized_signals, config)
        # 5. 权重决策层 (Weighting Layer)
        # 依据代理信号识别的市场政权(Regime)分配动态权重
        dynamic_weights = self._calculate_dynamic_weights(df_index, normalized_signals, proxy_signals, mtf_signals)
        # 6. 核心维度得分层 (Scoring Layer)
        aggressiveness_score = self._calculate_aggressiveness_score(df_index, mtf_signals, normalized_signals, dynamic_weights)
        control_score = self._calculate_control_score(df_index, mtf_signals, normalized_signals, historical_context)
        obstacle_clearance_score = self._calculate_obstacle_clearance_score(df_index, mtf_signals, normalized_signals, historical_context)
        # 7. 风险裁决层 (Risk Adjudication)
        # 核心逻辑：利用 V6.0 版风险共振引擎生成非线性惩罚项
        total_risk_penalty = self._adjudicate_risk(
            df_index, raw_signals, mtf_signals, normalized_signals, 
            dynamic_weights, aggressiveness_score, params['rally_intent_synthesis_params']
        )
        # 8. 综合意图合成层 (Synthesis Layer)
        bullish_intent = self._synthesize_bullish_intent(
            df_index, aggressiveness_score, control_score, obstacle_clearance_score,
            mtf_signals, normalized_signals, dynamic_weights, historical_context,
            params['rally_intent_synthesis_params']
        )
        bearish_score = self._calculate_bearish_intent(df_index, raw_signals, mtf_signals, normalized_signals, historical_context)
        # 9. 最终修饰与极端行情保护
        # 先应用风险惩罚，再结合看跌分数
        penalized_bullish_part = bullish_intent * (1 - total_risk_penalty)
        final_rally_intent = (penalized_bullish_part + bearish_score).clip(-1, 1)
        # 应用情境调节器（RS与资本属性微调）
        final_rally_intent = self._apply_contextual_modulators(df_index, final_rally_intent, proxy_signals, mtf_signals)
        # A股封板保护逻辑
        final_rally_intent = final_rally_intent.mask(is_limit_up_day & (total_risk_penalty > 0.5), final_rally_intent * 0.5)
        final_rally_intent = final_rally_intent.mask(is_limit_up_day & (final_rally_intent < 0), 0.0)
        # 10. 调试信息回填
        self.strategy.atomic_states["_DEBUG_rally_total_risk_penalty"] = total_risk_penalty
        self.strategy.atomic_states["_DEBUG_rally_bullish_intent"] = bullish_intent
        self.strategy.atomic_states["_DEBUG_dynamic_period"] = dynamic_memory_period
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
        return context

    def _detect_phase_synchronization(self, df_index: pd.Index, price_memory: Dict, capital_memory: Dict, integrated_memory: pd.Series) -> pd.Series:
        """
        【V4.1 · 鲁棒相位版】相位同步检测
        修改说明：增加对滚动相关性 NaN 值的强制拦截，确保 sync_strength 始终有效。
        版本号：2026.02.07.09
        """
        price_trend = price_memory.get("trend_memory", pd.Series(0.0, index=df_index)).ewm(span=3).mean()
        capital_flow = capital_memory.get("composite_capital_flow", pd.Series(0.0, index=df_index)).ewm(span=3).mean()
        max_lag = 5
        window = 21
        lag_correlations = {}
        for lag in range(-max_lag, max_lag + 1):
            shifted_capital = capital_flow.shift(lag)
            # 核心修复：填充相关性计算产生的 NaN (通常由平直曲线导致)
            corr_series = price_trend.rolling(window=window, min_periods=5).corr(shifted_capital).fillna(0.0)
            lag_correlations[lag] = corr_series
        corr_df = pd.DataFrame(lag_correlations)
        best_lag = corr_df.abs().idxmax(axis=1).fillna(0)
        # 获取最大相关性值并强制去空
        max_corr = corr_df.lookup(corr_df.index, best_lag) if hasattr(corr_df, 'lookup') else corr_df.values[np.arange(len(corr_df)), corr_df.columns.get_indexer(best_lag)]
        sync_strength = pd.Series(max_corr, index=df_index).fillna(0.0)
        lead_premium = (best_lag / max_lag).clip(-1, 1)
        lag_stability = 1 - best_lag.diff().abs().rolling(window=5).mean().div(max_lag).fillna(0).clip(0, 1)
        phase_sync_score = (sync_strength * (1 + lead_premium * 0.5) * lag_stability).clip(-1, 1)
        return phase_sync_score.fillna(0.0)

    def _assess_memory_quality(self, df_index: pd.Index, price_memory: Dict, capital_memory: Dict, chip_memory: Dict, sentiment_memory: Dict) -> pd.Series:
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

    def _calculate_dynamic_period(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series]) -> pd.Series:
        """
        【V3.3 · 异常防御版】动态记忆周期计算
        修改说明：增加对非有限值（NaN/Inf）的鲁棒性处理，修复强制转换整型时可能导致的 IntCastingNaNError。
        """
        # 1. 提取基础变量并处理可能的缺失值
        pct_change = raw_signals.get('pct_change', pd.Series(0.0, index=df_index)).fillna(0.0)
        vol_ratio = raw_signals.get('volume_ratio', pd.Series(1.0, index=df_index)).fillna(1.0)
        # 2. 计算波动率相对位置 (使用21日滚动窗口)
        volatility = pct_change.rolling(window=21, min_periods=1).std().fillna(0.01)
        vol_min = volatility.rolling(window=60, min_periods=1).min()
        vol_max = volatility.rolling(window=60, min_periods=1).max()
        # 归一化波动率位置，增加极小值防止除零
        vol_pos = (volatility - vol_min) / (vol_max - vol_min + 1e-9)
        # 3. 计算趋势一致性因子 (Efficiency Ratio)
        abs_diff = pct_change.abs()
        net_diff = pct_change.rolling(window=10, min_periods=1).sum().abs()
        path_len = abs_diff.rolling(window=10, min_periods=1).sum()
        # 增加对 path_len 的安全保护
        er = (net_diff / (path_len + 1e-9)).clip(0, 1)
        # 4. 周期映射模型：基准周期 21
        dynamic_period = 21 * (1 + er * 0.5 - vol_pos.fillna(0.5) * 0.5)
        # 5. 加入成交量异动修正
        vol_adj = (vol_ratio / 5.0).clip(0, 0.3)
        dynamic_period = dynamic_period * (1 - vol_adj)
        # 6. 【核心修复】处理非有限值并安全转换
        # 先替换 inf 为 nan，再用默认周期 21 填充 nan，最后转换类型
        final_period = dynamic_period.replace([np.inf, -np.inf], np.nan)
        final_period = final_period.fillna(21).clip(5, 55).round().astype(np.int32)
        return final_period

    def _estimate_snr(self, series: pd.Series) -> pd.Series:
        """
        【V4.6 · 趋势效率版】计算信号信噪比 (SNR)
        核心理念：基于卡夫曼效率比 (Kaufman Efficiency Ratio) 评估记忆信号的纯度。
        A股特性：
        - 高SNR (接近1)：主力锁仓推进，趋势平滑，记忆质量极高。
        - 低SNR (接近0)：分歧剧烈，信号震荡，记忆不可靠。
        算法：Signal / Noise = |Net Change| / Sum(|Step Change|)
        """
        if series.empty:
            return pd.Series(0.5, index=series.index)
        # 设定滚动窗口 (与记忆周期匹配，默认21日)
        window = 21
        # 1. 计算信号强度 (Signal Power)
        # 窗口内的净位移绝对值：|Price_t - Price_{t-n}|
        # 对于记忆序列，直接计算差值
        signal_power = series.diff(window).abs()
        # 2. 计算噪声强度 (Noise Power)
        # 窗口内每一步变化的绝对值之和 (路径长度)
        # Sum(|Price_t - Price_{t-1}|) over window
        volatility = series.diff().abs()
        noise_power = volatility.rolling(window=window).sum()
        # 3. 计算信噪比 (Efficiency Ratio)
        # 避免除零风险
        snr_raw = signal_power / (noise_power + 1e-9)
        # 4. 映射与增强
        # ER 本身在 [0, 1] 之间。
        # 在A股，ER > 0.4 已经是非常好的趋势，ER > 0.6 是极强趋势
        # 我们对其进行非线性扩张，使得 0.3-0.6 区间的区分度最大
        # 使用 Sigmoid 变体：中心点 0.3，陡峭度 10
        snr_score = 1 / (1 + np.exp(-10 * (snr_raw - 0.3)))
        # 5. 极小值处理
        # 如果噪声极小（几乎无波动），视为高质量信号
        mask_quiet = noise_power < 1e-6
        snr_score = snr_score.mask(mask_quiet, 1.0)
        # 前向填充，平滑处理
        return snr_score.ffill().fillna(0.5)

    def _calculate_price_memory(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], params: Dict) -> Dict[str, pd.Series]:
        """
        【V3.0】价格记忆深度计算
        核心理念：价格具有记忆效应，近期价格行为对当前影响更大
        数学模型：指数加权记忆衰减 + 趋势结构识别
        """
        # 初始化价格记忆计算类
        price_memory_calculator = CalculatePriceMemory(df_index, raw_signals.get('close', pd.Series(0.0, index=df_index)), params)
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
        momentum_memory = price_memory_calculator.calculate_adaptive_momentum_memory(
            raw_signals.get('close', pd.Series(0.0, index=df_index)),
            df_index, memory_period
        )
        # 3. 波动率记忆（GARCH模型简化版）
        volatility_memory = price_memory_calculator.calculate_volatility_memory(
            raw_signals.get('pct_change', pd.Series(0.0, index=df_index)),
            df_index, memory_period
        )
        # 4. 支撑阻力记忆（关键价格水平记忆）
        support_resistance_memory = price_memory_calculator.calculate_support_resistance_memory(
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
        # 初始化资金记忆计算类
        capital_memory_calculator = CalculateCapitalMemory(raw_signals.get('capital_flow', pd.Series(0.0, index=df_index)))
        # 参数
        memory_period = get_param_value(params.get('capital_memory_period'), 21)
        # 1. 多级别资金流合成（向量合成法）
        capital_vectors = []
        # 特大单净流向（权重最高）
        if 'buy_elg_amount' in raw_signals and 'sell_elg_amount' in raw_signals:
            elg_net = (raw_signals['buy_elg_amount'] - raw_signals['sell_elg_amount'])
            elg_vector = elg_net.rolling(window=5, min_periods=3).mean()  # 5日平滑
            capital_vectors.append(("elg", elg_vector, 0.35))
        # 大单净流向
        if 'buy_lg_amount' in raw_signals and 'sell_lg_amount' in raw_signals:
            lg_net = (raw_signals['buy_lg_amount'] - raw_signals['sell_lg_amount'])
            lg_vector = lg_net.rolling(window=3, min_periods=2).mean()  # 3日平滑
            capital_vectors.append(("lg", lg_vector, 0.30))
        # 中单净流向
        if 'buy_md_amount' in raw_signals and 'sell_md_amount' in raw_signals:
            md_net = (raw_signals['buy_md_amount'] - raw_signals['sell_md_amount'])
            md_vector = md_net.rolling(window=2, min_periods=1).mean()  # 2日平滑
            capital_vectors.append(("md", md_vector, 0.20))
        # 小单净流向（反向指标，权重为负）
        if 'buy_sm_amount' in raw_signals and 'sell_sm_amount' in raw_signals:
            sm_net = (raw_signals['buy_sm_amount'] - raw_signals['sell_sm_amount'])
            sm_vector = -sm_net.rolling(window=1).mean()  # 当日，反向
            capital_vectors.append(("sm", sm_vector, 0.15))
        # 向量合成
        composite_capital_flow = pd.Series(0.0, index=df_index)
        for name, vector, weight in capital_vectors:
            # 归一化处理
            vector_norm = capital_memory_calculator._normalize_capital_vector(vector, df_index)
            composite_capital_flow += vector_norm * weight
        # 2. 资金持续性检测（Hurst指数简化版）
        persistence_score = capital_memory_calculator._calculate_capital_persistence(
            composite_capital_flow, df_index, memory_period
        )
        # 3. 资金异常检测（Z-score异常检测）
        anomaly_score = capital_memory_calculator._detect_capital_anomaly(
            composite_capital_flow, df_index, memory_period
        )
        # 4. 资金效率记忆（资金推动价格上涨的效率）
        efficiency_memory = capital_memory_calculator._calculate_capital_efficiency(
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
        chip_memory_calculator = CalculateChipMemory(df_index, raw_signals, params)
        memory_period = get_param_value(params.get('chip_memory_period'), 55)
        # 1. 筹码熵变记忆（信息熵变化反映筹码混乱度）
        entropy_memory = chip_memory_calculator._calculate_chip_entropy_memory(
            raw_signals, df_index, memory_period
        )
        # 2. 筹码集中度迁移记忆
        concentration_migration = chip_memory_calculator._calculate_concentration_migration(
            raw_signals.get('chip_concentration_ratio', pd.Series(0.0, index=df_index)),
            df_index, memory_period
        )
        # 3. 筹码稳定性记忆（马尔可夫稳定性检测）
        stability_memory = chip_memory_calculator._calculate_chip_stability_memory(
            raw_signals, df_index, memory_period
        )
        # 4. 筹码压力记忆（获利盘与套牢盘记忆）
        pressure_memory = chip_memory_calculator._calculate_chip_pressure_memory(
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
        sentiment_memory_calculator = CalculateSentimentMemory(df_index, raw_signals, params)
        memory_period = get_param_value(params.get('sentiment_memory_period'), 13)
        # 1. 情绪动量记忆（一阶差分动量）
        sentiment_momentum = self._calculate_sentiment_momentum(
            raw_signals.get('market_sentiment', pd.Series(0.0, index=df_index)),
            df_index, memory_period
        )
        # 2. 情绪分歧度记忆（波动率反映分歧）
        sentiment_divergence = sentiment_memory_calculator._calculate_sentiment_divergence(
            raw_signals.get('market_sentiment', pd.Series(0.0, index=df_index)),
            df_index, memory_period
        )
        # 3. 情绪极端记忆（Z-score检测极端情绪）
        sentiment_extreme = sentiment_memory_calculator._detect_sentiment_extreme(
            raw_signals.get('market_sentiment', pd.Series(0.0, index=df_index)),
            df_index, memory_period
        )
        # 4. 情绪一致性记忆（多指标情绪一致性）
        sentiment_consistency = sentiment_memory_calculator._calculate_sentiment_consistency(
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

    def _fuse_integrated_memory(self, price_memory: Dict, capital_memory: Dict, chip_memory: Dict, sentiment_memory: Dict, params: Dict) -> pd.Series:
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

    def _detect_market_state(self, price_score: pd.Series, capital_score: pd.Series, chip_score: pd.Series, sentiment_score: pd.Series) -> str:
        """
        【V4.2 · 相位识别版】基于多维记忆分值动态检测市场核心状态
        逻辑：通过判定各维度的一致性与动量斜率，识别市场所处的博弈阶段。
        """
        # 获取最新一帧的数据
        p, c, s, ch = price_score.iloc[-1], capital_score.iloc[-1], sentiment_score.iloc[-1], chip_score.iloc[-1]
        # 计算短期动量斜率（3日均值差）
        p_slope = price_score.tail(3).diff().mean()
        c_slope = capital_score.tail(3).diff().mean()
        # 1. 趋势态判定：价格与资金强共振，且价格分值处于高位或明显上升
        if (p > 0.6 and c > 0.5 and p_slope > 0) or (p > 0.75):
            return "trending_up"
        # 2. 弱势态判定：价格与资金持续走弱
        if (p < 0.4 and c < 0.4 and p_slope < 0) or (p < 0.25):
            return "trending_down"
        # 3. 转折态判定：资金或情绪与价格发生明显的相位背离（A股典型的见顶/见底特征）
        # 场景：价格还在涨但资金斜率已转负，或者价格低位但资金开始暴增
        if (p_slope * c_slope < 0) and (abs(c_slope) > 0.05):
            return "reversing"
        # 4. 整理态判定：价格波动钝化，筹码集中度或稳定性成为主导变量
        if abs(p_slope) < 0.02 and 0.4 <= p <= 0.6:
            return "consolidating"
        # 默认兜底状态
        return "consolidating"

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

    def _construct_proxy_signals(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], config: Dict) -> Dict[str, pd.Series]:
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
        return final_proxy_signals

    def _calculate_enhanced_rs_proxy(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], config: Dict) -> Dict[str, pd.Series]:
        """
        【V4.0】增强版相对强度代理信号
        核心理念：相对强度不仅看价格趋势，还要看动量、加速、结构等多个维度
        数学模型：多维度RSI扩展 + 结构强度 + 动量扩散
        数据层需要新增：
        1. 行业相对强度排名
        2. 板块轮动速度
        3. 相对成交额变化
        """
        enhanced_rs_proxy_calculator = EnhancedRSProxyCalculator(config)
        rs_components = {}
        # 1. 价格趋势相对强度（MTF趋势）
        price_trend_strength = mtf_signals.get('mtf_price_trend', pd.Series(0.0, index=df_index)).clip(lower=0)
        # 2. 动量相对强度（自适应RSI）
        rsi_strength = enhanced_rs_proxy_calculator._calculate_adaptive_rsi_strength(
            normalized_signals.get('RSI_norm', pd.Series(0.5, index=df_index)),
            df_index
        )
        # 3. 加速相对强度（价格加速度）
        acceleration_strength = enhanced_rs_proxy_calculator._calculate_price_acceleration_strength(
            mtf_signals.get('mtf_price_trend', pd.Series(0.0, index=df_index)),
            df_index
        )
        # 4. 结构相对强度（突破、支撑、阻力）
        structural_strength = enhanced_rs_proxy_calculator._calculate_structural_strength(
            normalized_signals, mtf_signals, df_index
        )
        # 5. 成交量相对强度（量价配合）
        volume_strength = enhanced_rs_proxy_calculator._calculate_volume_relative_strength(
            normalized_signals.get('volume_ratio_norm', pd.Series(0.5, index=df_index)),
            price_trend_strength,
            df_index
        )
        # 6. 资金流相对强度（资金推动效率）
        capital_flow_strength = enhanced_rs_proxy_calculator._calculate_capital_flow_relative_strength(
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
        market_state = enhanced_rs_proxy_calculator._detect_market_state_for_rs(rs_components_values, df_index)
        rs_modulator = enhanced_rs_proxy_calculator._calculate_rs_modulator(market_state, config)
        # 最终相对强度代理信号
        enhanced_rs_proxy = (rs_proxy * rs_modulator).clip(0, 1)
        return {
            "raw_rs_proxy": rs_proxy,
            "enhanced_rs_proxy": enhanced_rs_proxy,
            "rs_modulator": rs_modulator,
            "rs_components": rs_components_values,
            "market_state": pd.Series(market_state, index=df_index)
        }

    def _calculate_enhanced_capital_proxy(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], config: Dict) -> Dict[str, pd.Series]:
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
        enhanced_capital_proxy_calculator = EnhancedCapitalProxyCalculator(config)
        capital_components = {}
        # 1. 多级别资金流合成（基于原始日级资金数据）
        # 使用向量合成法，但加入更多维度
        capital_flow_composite = enhanced_capital_proxy_calculator._calculate_multi_level_capital_flow(
            normalized_signals, df_index, config
        )
        # 2. 资本效率因子（CEF）：单位资金推动价格上涨的效率
        capital_efficiency = enhanced_capital_proxy_calculator._calculate_capital_efficiency_factor(
            capital_flow_composite,
            mtf_signals.get('mtf_price_trend', pd.Series(0.0, index=df_index)),
            normalized_signals.get('volume_ratio_norm', pd.Series(0.5, index=df_index)),
            df_index
        )
        # 3. 资金结构指数（FSI）：资金流向的集中度和稳定性
        fund_structure_index = enhanced_capital_proxy_calculator._calculate_fund_structure_index(
            normalized_signals, df_index
        )
        # 4. 资金持续性指数（FPI）：资金流向的持续时间和强度
        fund_persistence_index = enhanced_capital_proxy_calculator._calculate_fund_persistence_index(
            capital_flow_composite, df_index
        )
        # 5. 资金领先滞后关系（FLR）：资金流向领先价格的程度
        fund_lead_lag_ratio = enhanced_capital_proxy_calculator._calculate_fund_lead_lag_ratio(
            capital_flow_composite,
            mtf_signals.get('mtf_price_trend', pd.Series(0.0, index=df_index)),
            df_index
        )
        # 6. 资金异常检测（FAD）：检测异常资金流动
        fund_anomaly_detection = enhanced_capital_proxy_calculator._detect_fund_anomalies(
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
        liquidity_state = enhanced_capital_proxy_calculator._assess_market_liquidity_state(normalized_signals, df_index)
        capital_modulator = enhanced_capital_proxy_calculator._calculate_capital_modulator(liquidity_state, config)
        # 最终资本属性代理信号
        enhanced_capital_proxy = (capital_proxy * capital_modulator).clip(0, 1)
        return {
            "raw_capital_proxy": capital_proxy,
            "enhanced_capital_proxy": enhanced_capital_proxy,
            "capital_modulator": capital_modulator,
            "capital_components": capital_components_values,
            "liquidity_state": pd.Series(liquidity_state, index=df_index)
        }

    def _calculate_enhanced_sentiment_proxy(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], config: Dict) -> Dict[str, pd.Series]:
        """
        【V4.1 · 接口修复版】增强版市场情绪代理信号
        修改说明：修复调用 _calculate_sentiment_momentum 时缺失 memory_period 参数导致的 TypeError。
        """
        enhanced_sentiment_proxy_calculator = EnhancedSentimentProxyCalculator()
        # 显式获取情绪记忆周期参数，默认 13 日
        memory_period = get_param_value(config.get('sentiment_memory_period'), 13)
        # 1. 基础市场情绪
        base_sentiment = normalized_signals.get('market_sentiment_norm', pd.Series(0.5, index=df_index))
        # 2. 情绪动量（传入缺失的 memory_period）
        sentiment_momentum = self._calculate_sentiment_momentum(
            base_sentiment, df_index, memory_period
        )
        # 3. 情绪分歧度
        sentiment_divergence = enhanced_sentiment_proxy_calculator._calculate_sentiment_divergence_index(
            normalized_signals, df_index
        )
        # 4. 情绪极端性
        sentiment_extremity = enhanced_sentiment_proxy_calculator._calculate_sentiment_extremity_index(
            base_sentiment, df_index
        )
        # 5. 情绪传染性
        sentiment_contagion = enhanced_sentiment_proxy_calculator._calculate_sentiment_contagion_index(
            normalized_signals, df_index
        )
        # 6. 情绪稳定性
        sentiment_stability = enhanced_sentiment_proxy_calculator._calculate_sentiment_stability_index(
            base_sentiment, df_index
        )
        # 综合权重
        sentiment_weights = {
            "base": 0.25, "momentum": 0.20, "divergence": 0.15,
            "extremity": 0.15, "contagion": 0.15, "stability": 0.10
        }
        sentiment_components_values = {
            "base": base_sentiment.clip(0, 1),
            "momentum": (sentiment_momentum * 0.5 + 0.5).clip(0, 1),
            "divergence": (1 - sentiment_divergence).clip(0, 1),
            "extremity": (1 - sentiment_extremity).clip(0, 1),
            "contagion": sentiment_contagion.clip(0, 1),
            "stability": sentiment_stability.clip(0, 1)
        }
        sentiment_proxy = enhanced_sentiment_proxy_calculator._fuzzy_logic_synthesis(sentiment_components_values, sentiment_weights, df_index)
        market_phase = self._identify_market_phase(df_index, {"sentiment_state": base_sentiment}, normalized_signals)
        sentiment_modulator = enhanced_sentiment_proxy_calculator._calculate_sentiment_modulator(market_phase, config)
        enhanced_sentiment_proxy = (sentiment_proxy * sentiment_modulator).clip(0, 1)
        return {
            "raw_sentiment_proxy": sentiment_proxy,
            "enhanced_sentiment_proxy": enhanced_sentiment_proxy,
            "sentiment_modulator": sentiment_modulator,
            "sentiment_components": sentiment_components_values,
            "market_phase": pd.Series(market_phase, index=df_index)
        }

    def _calculate_sentiment_momentum(self, sentiment_series: pd.Series, df_index: pd.Index, memory_period: int = 13) -> pd.Series:
        """
        【V3.6 · 健壮性增强版】计算情绪动量
        修改说明：为 memory_period 增加默认值，确保接口调用的鲁棒性。
        """
        if sentiment_series.empty:
            return pd.Series(0.5, index=df_index)
        # 1. 情绪平滑 (使用 5 日 EMA 过滤杂波)
        smooth_sentiment = sentiment_series.ewm(span=5, adjust=False).mean()
        # 2. 计算一阶动量 (3 日变化率)
        velocity = smooth_sentiment.diff(3).fillna(0)
        # 3. 计算二阶动量 (3 日加速度)
        acceleration = velocity.diff(3).fillna(0)
        # 4. 动量合成
        vel_score = np.tanh(velocity * 5) * 0.5 + 0.5
        acc_score = np.tanh(acceleration * 5) * 0.5 + 0.5
        momentum_score = (vel_score * 0.6 + acc_score * 0.4).clip(0, 1)
        return momentum_score

    def _calculate_enhanced_liquidity_proxy(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], config: Dict) -> Dict[str, pd.Series]:
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
        enhanced_liquidity_proxy_calculator = EnhancedLiquidityProxyCalculator(config)
        liquidity_components = {}
        # 1. 成交量流动性（传统流动性指标）
        volume_liquidity = normalized_signals.get('volume_ratio_norm', pd.Series(0.5, index=df_index))
        # 2. 换手率流动性（筹码交换速度）
        turnover_liquidity = normalized_signals.get('turnover_rate_norm', pd.Series(0.5, index=df_index))
        # 3. 订单簿流动性（买卖不平衡度）
        orderbook_liquidity = enhanced_liquidity_proxy_calculator._calculate_orderbook_liquidity_index(
            normalized_signals, df_index
        )
        # 4. 冲击成本流动性（大单交易对价格的影响）
        impact_cost_liquidity = enhanced_liquidity_proxy_calculator._calculate_impact_cost_liquidity(
            normalized_signals, df_index
        )
        # 5. 流动性分层（不同价格区间的流动性差异）
        layered_liquidity = enhanced_liquidity_proxy_calculator._calculate_layered_liquidity_index(
            normalized_signals, df_index
        )
        # 6. 流动性风险溢价（流动性不足时的风险补偿）
        liquidity_risk_premium = enhanced_liquidity_proxy_calculator._calculate_liquidity_risk_premium(
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
        liquidity_proxy = enhanced_liquidity_proxy_calculator._optimized_synthesis(liquidity_components_values, liquidity_weights, df_index)
        # 流动性调节器：根据市场波动调整
        market_volatility = normalized_signals.get('volatility_instability_norm', pd.Series(0.5, index=df_index))
        liquidity_modulator = enhanced_liquidity_proxy_calculator._calculate_liquidity_modulator(market_volatility, config)
        # 最终流动性代理信号
        enhanced_liquidity_proxy = (liquidity_proxy * liquidity_modulator).clip(0, 1)
        return {
            "raw_liquidity_proxy": liquidity_proxy,
            "enhanced_liquidity_proxy": enhanced_liquidity_proxy,
            "liquidity_modulator": liquidity_modulator,
            "liquidity_components": liquidity_components_values,
            "market_volatility": market_volatility
        }

    def _calculate_enhanced_volatility_proxy(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], config: Dict) -> Dict[str, pd.Series]:
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
        enhanced_volatility_proxy_calculator = EnhancedVolatilityProxyCalculator(config)
        volatility_components = {}
        # 1. 历史波动率（基于收益率）
        historical_volatility = enhanced_volatility_proxy_calculator._calculate_historical_volatility_index(
            normalized_signals.get('pct_change_norm', pd.Series(0.0, index=df_index)),
            df_index
        )
        # 2. 波动率偏度（上涨波动与下跌波动的不对称性）
        volatility_skew = enhanced_volatility_proxy_calculator._calculate_volatility_skew_index(
            normalized_signals.get('pct_change_norm', pd.Series(0.0, index=df_index)),
            df_index
        )
        # 3. 波动率聚集（GARCH效应）
        volatility_clustering = enhanced_volatility_proxy_calculator._calculate_volatility_clustering_index(
            normalized_signals.get('pct_change_norm', pd.Series(0.0, index=df_index)),
            df_index
        )
        # 4. 波动率微笑（不同行权价的波动率差异）
        volatility_smile = enhanced_volatility_proxy_calculator._calculate_volatility_smile_index(
            normalized_signals, df_index
        )
        # 5. 波动率期限结构（不同到期日的波动率）
        volatility_term_structure = enhanced_volatility_proxy_calculator._calculate_volatility_term_structure_index(
            normalized_signals, df_index
        )
        # 6. 波动率风险溢价（预期波动率与实际波动率差异）
        volatility_risk_premium = enhanced_volatility_proxy_calculator._calculate_volatility_risk_premium_index(
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
            "historical": enhanced_volatility_proxy_calculator._inverse_u_transform(historical_volatility, 0.2, 0.5),  # 最优波动率20%-50%
            "skew": (1 - abs(volatility_skew)).clip(0, 1),  # 偏度越小越好
            "clustering": (1 - volatility_clustering).clip(0, 1),  # 聚集效应越低越好
            "smile": (1 - volatility_smile).clip(0, 1),  # 微笑效应越小越好
            "term_structure": volatility_term_structure.clip(0, 1),
            "risk_premium": (1 - volatility_risk_premium).clip(0, 1)  # 风险溢价越低越好
        }
        # 使用贝叶斯合成
        volatility_proxy = enhanced_volatility_proxy_calculator._bayesian_synthesis(volatility_components_values, volatility_weights, df_index)
        # 波动性调节器：根据市场阶段调整
        market_regime = enhanced_volatility_proxy_calculator._identify_volatility_regime(volatility_components_values, df_index)
        volatility_modulator = enhanced_volatility_proxy_calculator._calculate_volatility_modulator(market_regime, config)
        # 最终波动性代理信号
        enhanced_volatility_proxy = (volatility_proxy * volatility_modulator).clip(0, 1)
        return {
            "raw_volatility_proxy": volatility_proxy,
            "enhanced_volatility_proxy": enhanced_volatility_proxy,
            "volatility_modulator": volatility_modulator,
            "volatility_components": volatility_components_values,
            "market_regime": pd.Series(market_regime, index=df_index)
        }

    def _calculate_enhanced_risk_preference_proxy(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], config: Dict) -> Dict[str, pd.Series]:
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
        enhanced_risk_preference_proxy_calculator = EnhancedRiskPreferenceProxyCalculator(config)
        risk_preference_components = {}
        # 1. 风险资产表现（高风险板块 vs 低风险板块）
        risky_asset_performance = enhanced_risk_preference_proxy_calculator._calculate_risky_asset_performance_index(
            normalized_signals, df_index
        )
        # 2. 风险规避程度（避险资产资金流向）
        risk_aversion_degree = enhanced_risk_preference_proxy_calculator._calculate_risk_aversion_degree_index(
            normalized_signals, df_index
        )
        # 3. 风险转移（资金从避险资产流向风险资产）
        risk_transfer = enhanced_risk_preference_proxy_calculator._calculate_risk_transfer_index(
            normalized_signals, df_index
        )
        # 4. 风险定价（风险溢价水平）
        risk_pricing = enhanced_risk_preference_proxy_calculator._calculate_risk_pricing_index(
            normalized_signals, df_index
        )
        # 5. 风险情绪（投资者对风险的容忍度）
        risk_sentiment = enhanced_risk_preference_proxy_calculator._calculate_risk_sentiment_index(
            normalized_signals, df_index
        )
        # 6. 风险传导（风险在资产间的传播）
        risk_contagion = enhanced_risk_preference_proxy_calculator._calculate_risk_contagion_index(
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
        risk_preference_proxy = enhanced_risk_preference_proxy_calculator._neural_inspired_synthesis(risk_preference_components_values, risk_weights, df_index)
        # 风险偏好调节器：根据经济周期调整
        economic_cycle = enhanced_risk_preference_proxy_calculator._identify_economic_cycle(risk_preference_components_values, df_index)
        risk_preference_modulator = enhanced_risk_preference_proxy_calculator._calculate_risk_preference_modulator(economic_cycle, config)
        # 最终风险偏好代理信号
        enhanced_risk_preference_proxy = (risk_preference_proxy * risk_preference_modulator).clip(0, 1)
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
        return final_proxy_signals

    def _assess_signal_quality(self, rs_proxy: Dict, capital_proxy: Dict, sentiment_proxy: Dict, liquidity_proxy: Dict, volatility_proxy: Dict, risk_preference_proxy: Dict) -> pd.Series:
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
        signal_quality_assessor = SignalQualityAssessor()
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
            signal_quality = signal_quality_assessor._calculate_individual_signal_quality(signal_series, signal_name)
            individual_qualities[signal_name] = signal_quality
        # 2. 计算信号间的一致性质量
        consistency_quality = signal_quality_assessor._calculate_signal_consistency_quality(all_signals)
        # 3. 计算信号的预测有效性（滞后相关性分析）
        predictive_quality = signal_quality_assessor._calculate_predictive_quality(all_signals)
        # 4. 综合信号质量（加权合成）
        # 动态权重：根据各信号的历史表现调整权重
        dynamic_weights = signal_quality_assessor._calculate_dynamic_quality_weights(individual_qualities, consistency_quality, predictive_quality)
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
        return comprehensive_quality.clip(0, 1)

    def _detect_comprehensive_market_state(self, rs_signal: pd.Series, capital_signal: pd.Series,
                                          sentiment_signal: pd.Series, liquidity_signal: pd.Series,
                                          volatility_signal: pd.Series, risk_preference_signal: pd.Series) -> pd.Series:
        """
        【V4.7 · 动量增强版】综合市场状态检测方法
        核心优化：引入信号强度(Strength)变量，剔除低斜率的"伪趋势"，增强反转识别的准确性。
        """
        market_states = pd.Series("consolidating", index=rs_signal.index)
        # 设定动态阈值
        STRONG_TREND_THRESHOLD = 0.15  # 强趋势斜率阈值
        REVERSAL_IMPULSE_THRESHOLD = 0.20  # 反转冲量阈值
        for i in range(len(market_states)):
            if i < 30:
                market_states.iloc[i] = "consolidating"
                continue
            # 1. 获取当前信号值
            rs_val = rs_signal.iloc[i] if i < len(rs_signal) else 0.5
            capital_val = capital_signal.iloc[i] if i < len(capital_signal) else 0.5
            sentiment_val = sentiment_signal.iloc[i] if i < len(sentiment_signal) else 0.5
            liquidity_val = liquidity_signal.iloc[i] if i < len(liquidity_signal) else 0.5
            volatility_val = volatility_signal.iloc[i] if i < len(volatility_signal) else 0.5
            risk_val = risk_preference_signal.iloc[i] if i < len(risk_preference_signal) else 0.5
            # 2. 获取趋势方向
            window_start = max(0, i-20)
            rs_window = rs_signal.iloc[window_start:i] if i < len(rs_signal) else pd.Series()
            capital_window = capital_signal.iloc[window_start:i] if i < len(capital_signal) else pd.Series()
            sentiment_window = sentiment_signal.iloc[window_start:i] if i < len(sentiment_signal) else pd.Series()
            rs_trend = self._calculate_trend_direction(rs_window)
            capital_trend = self._calculate_trend_direction(capital_window)
            sentiment_trend = self._calculate_trend_direction(sentiment_window)
            # 3. 计算并使用信号强度 (Magnitude of Trend)
            rs_strength = abs(rs_trend)
            capital_strength = abs(capital_trend)
            sentiment_strength = abs(sentiment_trend)
            # --- 状态判定逻辑 ---
            # 规则1：危机模式检测 (Crisis)
            # 逻辑：高波动 + 低风险偏好 + 低流动性，且资金或情绪在加速恶化
            if volatility_val > 0.7 and risk_val < 0.3 and liquidity_val < 0.3:
                # 确认恶化趋势具有强度
                if capital_strength > 0.1 or sentiment_strength > 0.1: 
                    market_states.iloc[i] = "crisis"
                    continue
            # 规则2：趋势上涨检测 (Trending Up)
            # 逻辑：高绝对值 + 正向趋势 + 足够强度的斜率(过滤缓慢爬升)
            if rs_val > 0.65 and capital_val > 0.6 and sentiment_val > 0.6:
                # 必须同时满足方向向上且动量充足
                if (rs_trend > 0 and rs_strength > STRONG_TREND_THRESHOLD and 
                    capital_trend > 0 and capital_strength > STRONG_TREND_THRESHOLD):
                    market_states.iloc[i] = "trending_up"
                    continue
            # 规则3：趋势下跌检测 (Trending Down)
            # 逻辑：低绝对值 + 负向趋势 + 足够强度的斜率(过滤阴跌，阴跌往往归为调整或危机)
            if rs_val < 0.35 and capital_val < 0.4 and sentiment_val < 0.4:
                if (rs_trend < 0 and rs_strength > STRONG_TREND_THRESHOLD and 
                    capital_trend < 0 and capital_strength > STRONG_TREND_THRESHOLD):
                    market_states.iloc[i] = "trending_down"
                    continue
            # 规则4：反转上涨检测 (Reversing Up)
            # 逻辑：趋势斜率显著转正(Impulse) + 绝对值未达高位
            # 反转需要更大的动量来确认，防止骗线
            if (rs_trend > 0 and rs_strength > REVERSAL_IMPULSE_THRESHOLD and 
                capital_trend > 0 and capital_strength > REVERSAL_IMPULSE_THRESHOLD):
                # 只有在低位或中位时才叫反转/启动
                if rs_val < 0.65 and capital_val < 0.65:
                    market_states.iloc[i] = "reversing_up"
                    continue
            # 规则5：反转下跌检测 (Reversing Down)
            # 逻辑：趋势斜率显著转负 + 绝对值在高位(顶部反转)
            if (rs_trend < 0 and rs_strength > REVERSAL_IMPULSE_THRESHOLD and 
                sentiment_trend < 0 and sentiment_strength > REVERSAL_IMPULSE_THRESHOLD):
                if rs_val > 0.4 and capital_val > 0.4:
                    market_states.iloc[i] = "reversing_down"
                    continue
            # 默认状态
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
        【V4.5 · 政权锚定优化版】动态权重计算系统
        修改说明：引入政权锚定机制与相关性惩罚，解决极端行情下的权重漂移与共线性过载问题。
        """
        # 1. 计算基础市场状态因子与阶段识别
        market_state_factors = self._calculate_market_state_factors(df_index, normalized_signals, mtf_signals)
        market_phase = self._identify_market_phase(df_index, market_state_factors, normalized_signals)
        # 2. 计算基础权重（基于专家经验的政权锚定）
        base_weights = self._calculate_base_weights(df_index, market_phase, market_state_factors)
        # 3. 维度间相关性检测（防止共线性漂移）
        # 当维度间相关性过高时，缩减受影响维度的动态调整幅度
        dimension_corr = self._analyze_dimension_correlations(df_index, normalized_signals)
        # 4. 执行非线性权重映射
        # 优化点：根据 dimension_corr 动态调整 Sigmoid 的陡峭度
        adjusted_weights = self._apply_nonlinear_weight_mapping(df_index, base_weights, market_state_factors)
        # 5. 风险平价优化（引入流动性波动调节）
        risk_parity_weights = self._apply_risk_parity_optimization(df_index, adjusted_weights, normalized_signals)
        # 6. 【新增】政权锚定保护 (Regime Anchoring)
        # 在极端相位（如主升/反转风险），将动态权重向专家基准拉回 30%，防止过度拟合瞬时噪声
        anchored_weights = {}
        for dim in risk_parity_weights:
            anchor = base_weights[dim]
            dynamic = risk_parity_weights[dim]
            # 计算拉回力度：极端相位下拉回力度加大
            pull_strength = market_phase.map({
                "主升": 0.4, "反转风险": 0.5, "启动": 0.2, "横盘": 0.1
            }).fillna(0.1)
            anchored_weights[dim] = (dynamic * (1 - pull_strength) + anchor * pull_strength)
        # 7. 记忆平滑处理
        smoothed_weights = self._apply_memory_smoothing(df_index, anchored_weights)
        # 8. 【核心优化】带温度控制的归一化 (Temperature-Controlled Softmax)
        # 信号质量越高，温度越低，权重分布越“尖锐”；质量低时，温度高，权重分布越“平滑”
        signal_quality = self.strategy.atomic_states.get("_DEBUG_signal_quality", pd.Series(0.7, index=df_index))
        final_weights = self._normalize_weights_with_temperature(df_index, smoothed_weights, signal_quality)
        return final_weights

    def _normalize_weights_with_temperature(self, df_index: pd.Index, weights_dict: Dict[str, pd.Series], signal_quality: pd.Series) -> Dict[str, pd.Series]:
        """
        【V1.0 · 温度控制归一化】基于信号质量动态分配权重聚焦度
        逻辑：利用 Softmax 原理，在高质量信号时期强化优势维度，在低质量时期采取防御性均分。
        """
        normalized_weights = {dim: pd.Series(0.0, index=df_index) for dim in weights_dict.keys()}
        for i in range(len(df_index)):
            # 提取当前日的各维度原始动态权重
            raw_vals = np.array([weights_dict[dim].iloc[i] for dim in weights_dict])
            # 计算温度 T：质量越高 T 越小 (0.5 - 2.0 之间)
            q = signal_quality.iloc[i] if i < len(signal_quality) else 0.7
            temperature = 2.0 - (q * 1.5) 
            # 执行带有温度转换的指数归一化
            exp_vals = np.exp(raw_vals / temperature)
            norm_vals = exp_vals / np.sum(exp_vals)
            # 回填结果
            for idx, dim in enumerate(weights_dict.keys()):
                normalized_weights[dim].iloc[i] = norm_vals[idx]
        return normalized_weights

    def _calculate_market_state_factors(self, df_index: pd.Index, normalized_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V4.1 · 接口修复版】七维度市场状态因子计算
        修改说明：修复调用 _weighted_geometric_mean 时缺失 df_index 参数导致的 TypeError。
        """
        factors = {}
        # 1. 趋势状态因子
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
                                                              'ADX_norm': 0.1}, 
                                                             df_index)
        # 2. 波动状态因子
        vol_components = {
            'ATR_norm': normalized_signals.get('ATR_norm', pd.Series(0.5, index=df_index)),
            'BBW_norm': normalized_signals.get('BBW_norm', pd.Series(0.5, index=df_index)),
            'chip_entropy_norm': normalized_signals.get('chip_entropy_norm', pd.Series(0.5, index=df_index))
        }
        factors['volatility_state'] = 1 - self._weighted_geometric_mean(vol_components, 
                                                                      {'ATR_norm': 0.4, 'BBW_norm': 0.3, 'chip_entropy_norm': 0.3}, 
                                                                      df_index)
        # 3. 情绪状态因子
        sentiment_components = {
            'market_sentiment_norm': normalized_signals.get('market_sentiment_norm', pd.Series(0.5, index=df_index)),
            'industry_leader_norm': normalized_signals.get('industry_leader_norm', pd.Series(0.5, index=df_index)),
            'industry_breadth_norm': normalized_signals.get('industry_breadth_norm', pd.Series(0.5, index=df_index))
        }
        factors['sentiment_state'] = self._weighted_geometric_mean(sentiment_components,
                                                                 {'market_sentiment_norm': 0.4, 'industry_leader_norm': 0.3, 'industry_breadth_norm': 0.3},
                                                                 df_index)
        # 4. 流动性状态因子
        liq_components = {
            'volume_ratio_norm': normalized_signals.get('volume_ratio_norm', pd.Series(0.5, index=df_index)),
            'turnover_rate_norm': normalized_signals.get('turnover_rate_norm', pd.Series(0.5, index=df_index)),
            'net_amount_ratio_norm': normalized_signals.get('net_amount_ratio_norm', pd.Series(0.5, index=df_index))
        }
        factors['liquidity_state'] = self._weighted_geometric_mean(liq_components,
                                                                 {'volume_ratio_norm': 0.35, 'turnover_rate_norm': 0.35, 'net_amount_ratio_norm': 0.3},
                                                                 df_index)
        # 5. 资金状态因子
        cap_components = {
            'net_mf_amount_norm': normalized_signals.get('net_mf_amount_norm', pd.Series(0.5, index=df_index)),
            'flow_acceleration_norm': normalized_signals.get('flow_acceleration_norm', pd.Series(0.5, index=df_index)),
            'flow_persistence_norm': normalized_signals.get('flow_persistence_norm', pd.Series(0.5, index=df_index))
        }
        factors['capital_state'] = self._weighted_geometric_mean(cap_components,
                                                               {'net_mf_amount_norm': 0.4, 'flow_acceleration_norm': 0.3, 'flow_persistence_norm': 0.3},
                                                               df_index)
        # 6. 筹码状态因子
        chip_comp = {
            'chip_concentration_norm': normalized_signals.get('chip_concentration_ratio_norm', pd.Series(0.5, index=df_index)),
            'chip_stability_norm': normalized_signals.get('chip_stability_norm', pd.Series(0.5, index=df_index))
        }
        factors['chip_state'] = self._weighted_geometric_mean(chip_comp,
                                                            {'chip_concentration_norm': 0.6, 'chip_stability_norm': 0.4},
                                                            df_index)
        # 7. 风险状态因子
        risk_comp = {
            'breakout_risk_norm': 1 - normalized_signals.get('breakout_risk_warning_norm', pd.Series(0.0, index=df_index)),
            'reversal_risk_norm': 1 - normalized_signals.get('reversal_warning_score_norm', pd.Series(0.0, index=df_index))
        }
        factors['risk_state'] = self._weighted_geometric_mean(risk_comp,
                                                            {'breakout_risk_norm': 0.6, 'reversal_risk_norm': 0.4},
                                                            df_index)
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
        【V5.1 · 接口修复版】将JERK突变引入攻击性评分逻辑
        修改说明：补齐 normalize_score 调用所需的 target_index 和 windows 参数，修复 TypeError。
        版本号：2026.02.07.02
        """
        base_aggressiveness = self._calculate_basic_aggressiveness(df_index, mtf_signals, normalized_signals)
        # 1. 提取JERK冲击因子 (以5日和13日价格/资金流为主)
        price_jerk = mtf_signals.get('JERK_5_price_trend', pd.Series(0.0, index=df_index))
        money_jerk = mtf_signals.get('JERK_5_net_amount_trend', pd.Series(0.0, index=df_index))
        # 2. 修复归一化逻辑：显式传入 df_index 和 windows=60
        price_jerk_norm = normalize_score(price_jerk, target_index=df_index, windows=60)
        money_jerk_norm = normalize_score(money_jerk, target_index=df_index, windows=60)
        # 3. 归一化冲击因子合成
        jerk_impact = (price_jerk_norm * 0.6 + money_jerk_norm * 0.4).clip(0, 1)
        # 4. 非线性增强：当Jerk爆发时，代表主力意图转为暴力攻击
        aggressiveness_score = (base_aggressiveness * (1 + jerk_impact * 0.5)).clip(0, 1)
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
        return final_resonance

    def _calculate_control_score(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], historical_context: Dict[str, pd.Series]) -> pd.Series:
        """
        【V2.1 · 字典路径修复版】计算控制力分数
        修改说明：修复 KeyError，将筹码稳定性调节器指向正确的 chip_memory 子字典路径。
        版本号：2026.02.07.04
        """
        control_components = {
            "chip_concentration": normalized_signals.get('chip_concentration_ratio_norm', pd.Series(0.5, index=df_index)).clip(lower=0),
            "chip_convergence": normalized_signals.get('chip_convergence_ratio_norm', pd.Series(0.5, index=df_index)).clip(lower=0),
            "chip_stability": normalized_signals.get('chip_stability_norm', pd.Series(0.5, index=df_index)).clip(lower=0),
            "flow_consistency": normalized_signals.get('flow_consistency_norm', pd.Series(0.5, index=df_index)).clip(lower=0),
            "flow_stability": normalized_signals.get('flow_stability_norm', pd.Series(0.5, index=df_index)).clip(lower=0),
            "inflow_persistence": normalized_signals.get('inflow_persistence_norm', pd.Series(0.5, index=df_index)).clip(lower=0),
            "trend_confirmation": normalized_signals.get('trend_confirmation_norm', pd.Series(0.5, index=df_index)).clip(lower=0),
            "breakout_confidence": normalized_signals.get('breakout_confidence_norm', pd.Series(0.5, index=df_index)).clip(lower=0),
            "behavior_accumulation": normalized_signals.get('behavior_accumulation_norm', pd.Series(0.5, index=df_index)).clip(lower=0)
        }
        control_weights = {
            "chip_concentration": 0.15, "chip_convergence": 0.12, "chip_stability": 0.12,
            "flow_consistency": 0.10, "flow_stability": 0.10, "inflow_persistence": 0.10,
            "trend_confirmation": 0.10, "breakout_confidence": 0.11, "behavior_accumulation": 0.10
        }
        control_score = _robust_geometric_mean(control_components, control_weights, df_index).clip(0, 1)
        # 修正：从 historical_context 的子字典中提取稳定性调节因子
        if historical_context.get('hc_enabled', False):
            chip_mem = historical_context.get('chip_memory', {})
            # 使用 stability_memory 作为筹码锁定强度的反馈
            chip_stability_feedback = chip_mem.get('stability_memory', pd.Series(0.5, index=df_index))
            chip_stability_modulator = 0.05
            # 通过 (feedback - 0.5) 将分值转化为 [-0.5, 0.5] 的增益/减损系数
            control_score = (control_score * (1 + (chip_stability_feedback - 0.5) * chip_stability_modulator)).clip(0, 1)
        return control_score

    def _calculate_obstacle_clearance_score(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], historical_context: Dict[str, Any]) -> pd.Series:
        """
        【V3.0 · 筹码真空探测版】计算障碍清除分数
        逻辑：通过识别“上方套牢盘真空区”与“阻力平台穿透效率”，量化主力拉升的物理阻碍程度。
        A股特性：重点捕捉放量冲过密集套牢区后，进入“缩量控盘拉升”阶段的真空溢价。
        版本号：2026.02.07.05
        """
        # 1. 基础运动学分量提取
        vol_ratio = normalized_signals.get('volume_ratio_norm', pd.Series(0.5, index=df_index))
        abs_change = normalized_signals.get('absolute_change_strength_norm', pd.Series(0.5, index=df_index))
        # 2. 支撑阻力位置感知 (来自价格记忆模块)
        # sr_mem > 0.5 表示价格已站在主要阻力平台之上，进入相对安全区
        sr_mem = historical_context.get('price_memory', {}).get('support_resistance_memory', pd.Series(0.5, index=df_index))
        # 3. 筹码压力与真空度计算 (Vacuum Detection)
        chip_mem = historical_context.get('chip_memory', {})
        # 筹码压力记忆：越高代表上方套牢盘越密集
        chip_pressure = chip_mem.get('pressure_memory', pd.Series(0.5, index=df_index))
        # 筹码稳定性：越高代表筹码在当前价位锁定越稳
        chip_stability = chip_mem.get('stability_memory', pd.Series(0.5, index=df_index))
        # 定义真空度：(1 - 压力) 与 稳定性的加权。当上方无压力且底部稳定时，真空度最高
        vacuum_score = ((1 - chip_pressure) * 0.7 + chip_stability * 0.3).clip(0, 1)
        # 4. 计算推进效率因子 (Efficiency of Progress)
        # 公式：E = (Price_Change_Strength) / (Volume_Ratio + epsilon)
        # 逻辑：真空区特征是“低量高效拉升”，而阻力区特征是“放量滞涨（肉磨子行情）”
        raw_efficiency = abs_change / (vol_ratio + 1e-9)
        # 对效率进行动态归一化，捕捉相对于过去60日的爆发性效率
        efficiency_norm = normalize_score(raw_efficiency, target_index=df_index, windows=60)
        # 5. 障碍清除意图合成 (使用专家加权几何平均)
        # 权重分配：SR记忆决定基准，真空度决定上限，效率决定即时成色
        oc_components = {
            "sr_position": sr_mem,
            "vacuum_index": vacuum_score,
            "clearance_efficiency": efficiency_norm,
            "breakout_potential": normalized_signals.get('breakout_potential_norm', pd.Series(0.5, index=df_index)),
            "net_flow_strength": normalized_signals.get('net_amount_ratio_norm', pd.Series(0.5, index=df_index))
        }
        oc_weights = {
            "sr_position": 0.30,
            "vacuum_index": 0.25,
            "clearance_efficiency": 0.20,
            "breakout_potential": 0.15,
            "net_flow_strength": 0.10
        }
        obstacle_clearance_score = _robust_geometric_mean(oc_components, oc_weights, df_index).clip(0, 1)
        # 6. 特殊逻辑：高位缩量背离惩罚
        # 如果效率极高但成交量萎缩到极致（地量），在A股可能是流动性枯竭，而非真空，需小幅修正
        extreme_low_vol = (vol_ratio < 0.1)
        obstacle_clearance_score = obstacle_clearance_score.mask(extreme_low_vol, obstacle_clearance_score * 0.9)
        return obstacle_clearance_score

    def _synthesize_bullish_intent(self, df_index: pd.Index, aggressiveness_score: pd.Series, control_score: pd.Series, 
                                  obstacle_clearance_score: pd.Series, mtf_signals: Dict[str, pd.Series], 
                                  normalized_signals: Dict[str, pd.Series], dynamic_weights: Dict[str, pd.Series],
                                  historical_context: Dict[str, Any], rally_intent_synthesis_params: Dict) -> pd.Series:
        """
        【V5.6 · 探针增强版】主力多头意图深度合成
        修改说明：增加中间变量 NaN 探针输出，并对增强因子实施强制填充，切断污染链路。
        """
        bear_trap_bonus = self._detect_bear_trap(df_index, mtf_signals, normalized_signals).fillna(0)
        acceleration_resonance = self._calculate_acceleration_resonance(df_index, mtf_signals).fillna(0)
        components = {"aggressiveness": aggressiveness_score.fillna(0.5), "control": control_score.fillna(0.5), "obstacle_clearance": obstacle_clearance_score.fillna(0.5)}
        weights = {"aggressiveness": dynamic_weights.get("aggressiveness", 0.35), "control": dynamic_weights.get("control", 0.35), "obstacle_clearance": dynamic_weights.get("obstacle_clearance", 0.30)}
        bullish_intent_base = _robust_geometric_mean(components, weights, df_index).fillna(0.5)
        phase_sync = historical_context.get('phase_sync', pd.Series(0.0, index=df_index)).fillna(0.0)
        # 核心修复：对增强因子进行强制 clip 和 fillna
        enhancement_factor = (acceleration_resonance * 0.3 + bear_trap_bonus * 0.2 + (phase_sync.clip(0, 1)) * 0.1).fillna(0.0)
        enhanced_intent = (bullish_intent_base * (1 + enhancement_factor)).clip(0, 1)
        long_term_modulator = 1.0
        if historical_context.get('hc_enabled', False):
            integrated_mem = historical_context.get('integrated_memory', pd.Series(0.5, index=df_index)).fillna(0.5)
            long_term_modulator = 0.95 + (integrated_mem * 0.2)
        final_bullish_intent = (enhanced_intent * long_term_modulator).fillna(enhanced_intent).clip(0, 1)
        return final_bullish_intent

    def _calculate_bearish_intent(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], 
                                 mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series],
                                 historical_context: Dict[str, Any]) -> pd.Series:
        """
        【V3.1 · 鲁棒自检版】计算复合看跌意图
        修改说明：建立三层 NaN 拦截机制，确保在极端数据波动下看跌意图分值不坍塌。
        版本号：2026.02.07.09
        """
        # 1. 资金记忆调节器 (从子字典安全提取)
        mf_flow_memory_anti_bearish_modulator = 1.0
        if historical_context.get('hc_enabled', False):
            # 路径防御：防止 nested dict 缺失
            cap_mem_dict = historical_context.get('capital_memory', {})
            cap_memory = cap_mem_dict.get('integrated_capital_memory', pd.Series(0.5, index=df_index)).fillna(0.5)
            # 资金底色越强，对看跌信号的容忍度越高（减损看跌分值）
            mf_flow_memory_anti_bearish_modulator = (1 - cap_memory.clip(0, 1) * 0.2).clip(0.8, 1.0)
        # 2. 杀跌冲击计算 (Panic Jerk)
        # 提取负向加速度突变，并强制填补空值
        p_jerk = mtf_signals.get('JERK_5_price_trend', pd.Series(0.0, index=df_index)).fillna(0)
        m_jerk = mtf_signals.get('JERK_5_net_amount_trend', pd.Series(0.0, index=df_index)).fillna(0)
        # 仅捕获下行冲击 (Jerk < 0)
        panic_p = normalize_score(p_jerk.mask(p_jerk > 0, 0).abs(), target_index=df_index, windows=60).fillna(0)
        panic_m = normalize_score(m_jerk.mask(m_jerk > 0, 0).abs(), target_index=df_index, windows=60).fillna(0)
        panic_impact = (panic_p * 0.6 + panic_m * 0.4).clip(0, 1)
        # 3. 看跌成分聚合与权重校验
        bearish_components = {
            "distribution": normalized_signals.get('distribution_score_norm', pd.Series(0.0, index=df_index)).fillna(0),
            "behavior_dist": normalized_signals.get('behavior_distribution_norm', pd.Series(0.0, index=df_index)).fillna(0),
            "downtrend": normalized_signals.get('downtrend_strength_norm', pd.Series(0.0, index=df_index)).fillna(0),
            "chip_div": normalized_signals.get('chip_divergence_ratio_norm', pd.Series(0.0, index=df_index)).fillna(0),
            "panic": panic_impact
        }
        # 确保权重总和为 1.0
        bearish_weights = {
            "distribution": 0.25, "behavior_dist": 0.20,
            "downtrend": 0.20, "chip_div": 0.15,
            "panic": 0.20
        }
        # 4. 稳健几何平均计算
        # 内部已包含 eps 处理，此处外层二次保底
        bearish_score_raw = _robust_geometric_mean(bearish_components, bearish_weights, df_index).fillna(0.0).clip(0, 1)
        # 6. 合成最终负向分值
        bearish_score_modulated = (bearish_score_raw * mf_flow_memory_anti_bearish_modulator).clip(0, 1)
        return -bearish_score_modulated

    def _adjudicate_risk(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], 
                        mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series],
                        dynamic_weights: Dict[str, pd.Series], aggressiveness_score: pd.Series,
                        rally_intent_synthesis_params: Dict) -> pd.Series:
        """
        【V6.0 · 风险多维共振版】深度裁决主力拉升中的异常风险
        逻辑：构建“技术-结构-背离-动能”四位一体的风险监控矩阵，识别A股典型的“多杀多”与“诱多”陷阱。
        """
        # 1. 技术性风险：趋势竭尽与超买回归 (ADX低位 + RSI高位)
        adx_norm = normalized_signals.get('ADX_norm', pd.Series(0.5, index=df_index))
        rsi_norm = normalized_signals.get('RSI_norm', pd.Series(0.5, index=df_index))
        tech_risk = ((1 - adx_norm) * 0.4 + (rsi_norm - 0.7).clip(lower=0) * 0.6).clip(0, 1)
        # 2. 结构性风险：高位派发与筹码分歧
        dist_risk = normalized_signals.get('distribution_score_norm', pd.Series(0.0, index=df_index))
        chip_div = normalized_signals.get('chip_divergence_ratio_norm', pd.Series(0.0, index=df_index))
        struct_risk = (dist_risk * 0.7 + chip_div * 0.3).clip(0, 1)
        # 3. 背离性风险：量价背离 + 资金流向背离
        vp_div = self._calculate_volume_price_divergence(df_index, mtf_signals)
        # 计算资金-价格背离：价格斜率 > 0 且 资金流入斜率 < 0 (诱多特征)
        p_slope = mtf_signals.get('SLOPE_5_price_trend', pd.Series(0.0, index=df_index))
        c_slope = mtf_signals.get('SLOPE_5_net_amount_trend', pd.Series(0.0, index=df_index))
        cap_div_raw = ((p_slope > 0) & (c_slope < 0)).astype(float) * p_slope.abs()
        cap_div_norm = (cap_div_raw / (cap_div_raw.rolling(60).max() + 1e-9)).fillna(0).clip(0, 1)
        div_risk = (vp_div * 0.5 + cap_div_norm * 0.5).clip(0, 1)
        # 4. A股动能风险：识别“多杀多”滞涨陷阱
        # 逻辑：当量能极大(volume_ratio)但价格推进效率(absolute_change)极低时，风险陡增
        vol_ratio = normalized_signals.get('volume_ratio_norm', pd.Series(0.5, index=df_index))
        abs_strength = normalized_signals.get('absolute_change_strength_norm', pd.Series(0.5, index=df_index))
        exhaustion_risk = (vol_ratio * (1 - abs_strength)).rolling(window=3).mean().fillna(0).clip(0, 1)
        # 5. 风险综合评分 (基于MRF模型权重)
        risk_weights = {"tech": 0.15, "struct": 0.25, "div": 0.35, "exh": 0.25}
        total_risk_raw = (
            tech_risk * risk_weights["tech"] +
            struct_risk * risk_weights["struct"] +
            div_risk * risk_weights["div"] +
            exhaustion_risk * risk_weights["exh"]
        ).clip(0, 1)
        # 6. 非线性惩罚映射 (Sigmoid Penalty Engine)
        # 当综合风险触及阈值时，惩罚分值加速攀升
        sensitivity = get_param_value(rally_intent_synthesis_params.get('risk_sensitivity'), 5.0)
        threshold = get_param_value(rally_intent_synthesis_params.get('risk_threshold'), 0.6)
        # 惩罚函数：1 / (1 + exp(-k*(risk - threshold)))
        risk_penalty = 1 / (1 + np.exp(-sensitivity * (total_risk_raw - threshold)))
        return risk_penalty.clip(0, 1)

    def _apply_contextual_modulators(self, df_index: pd.Index, final_rally_intent: pd.Series, 
                                    proxy_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series]) -> pd.Series:
        """
        【V2.1 · 键名对齐修复版】应用情境调节器
        修改说明：修正 KeyError: 'liquidity_proxy'，将键名统一为 'liquidity_tide_proxy'，并增加 get() 安全保护。
        版本号：2026.02.07.07
        """
        # 1. 相对强度和资本属性调节
        rs_mod = proxy_signals.get('rs_modulator', pd.Series(1.0, index=df_index))
        cap_mod = proxy_signals.get('capital_modulator', pd.Series(1.0, index=df_index))
        modulated_intent = final_rally_intent * rs_mod * cap_mod
        # 2. 市场情绪调节 (系数 0.1)
        sentiment_proxy = proxy_signals.get('market_sentiment_proxy', pd.Series(0.5, index=df_index))
        market_sentiment_modulator = (1 + (sentiment_proxy - 0.5) * 0.2)
        modulated_intent = modulated_intent * market_sentiment_modulator
        # 3. 核心修复：流动性调节 (系数 0.05)
        # 将原本错误的 'liquidity_proxy' 修正为 'liquidity_tide_proxy'
        liq_proxy = proxy_signals.get('liquidity_tide_proxy', pd.Series(0.5, index=df_index))
        liquidity_modulator = (1 + (liq_proxy - 0.5) * 0.1)
        modulated_intent = modulated_intent * liquidity_modulator
        # 4. 筹码稳定性调节
        chip_trend = mtf_signals.get('mtf_chip_concentration_trend', pd.Series(0.0, index=df_index))
        chip_stability_modulator = (1 + chip_trend.clip(lower=0) * 0.05)
        modulated_intent = modulated_intent * chip_stability_modulator
        return modulated_intent.clip(-1, 1)

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