# 文件: strategies/trend_following/intelligence/process/calculate_main_force_rally_intent.py
# 主力 rally 意图计算 已完成
import json
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
import numba
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

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _numba_rolling_mode(arr, window):
        """
        【V1.1 · Numba 频率数组版】计算滚动众数
        修改说明：修复 TypingError。将字典计数替换为固定长度频率数组（相位仅0-5），消除字典类型推导隐患并提升性能。
        版本号：2026.02.11.55
        """
        n = len(arr)
        res = np.empty(n, dtype=np.float32)
        for i in range(n):
            if i < window - 1:
                res[i] = arr[i]
                continue
            # 核心修复：相位取值范围为 0-5，使用数组替代字典
            counts = np.zeros(6, dtype=np.int32)
            for j in range(i - window + 1, i + 1):
                val = int(arr[j])
                if 0 <= val <= 5:
                    counts[val] += 1
            # 寻找最大频数对应的索引
            max_count = -1
            mode_val = arr[i]
            for idx in range(6):
                if counts[idx] > max_count:
                    max_count = counts[idx]
                    mode_val = float(idx)
            res[i] = mode_val
        return res

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _numba_calculate_er(pct_changes, window):
        """
        【V1.0 · Numba 高性能版】计算效率比 (Efficiency Ratio)
        修改说明：利用 Numba JIT 实现滑动窗口 $ER$ 计算，公式为 $|\sum returns| / \sum |returns|$，彻底消除核心循环开销。
        版本号：2026.02.11.66
        """
        n = len(pct_changes)
        er = np.zeros(n, dtype=np.float32)
        for i in range(window - 1, n):
            net_diff = 0.0
            path_len = 0.0
            for j in range(i - window + 1, i + 1):
                net_diff += pct_changes[j]
                path_len += abs(pct_changes[j])
            if path_len > 1e-9:
                er[i] = abs(net_diff) / path_len
        return er

    @staticmethod
    @numba.jit(nopython=True, cache=True, parallel=True)
    def _numba_rolling_std(arr, window):
        """
        【V1.0 · Numba 并行加速版】计算滑动窗口标准差
        修改说明：利用 parallel=True 和 prange 实现多线程并行窗口统计。采用 Welford 算法变体的高效实现，大幅提升 MTF 融合效率。
        版本号：2026.02.11.71
        """
        n = len(arr)
        res = np.zeros(n, dtype=np.float32)
        for i in numba.prange(n):
            start = max(0, i - window + 1)
            count = i - start + 1
            s_sum = 0.0
            s2_sum = 0.0
            for j in range(start, i + 1):
                val = arr[j]
                s_sum += val
                s2_sum += val * val
            variance = (s2_sum / count) - (s_sum / count)**2
            res[i] = np.sqrt(max(0.0, variance))
        return res

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _numba_state_transition_audit(p_state, c_state, p_acc, window):
        """
        【V1.0 · Numba 状态迁移算子】识别环境惯性与非线性顶背离
        修改说明：利用 Numba 实现状态迁移审计。在计算环境惯性存量(HAB)的同时，监测价格动能与资金流向的非线性背离。
        版本号：2026.02.11.75
        """
        n = len(p_state)
        hab_inc = np.zeros(n, dtype=np.float32)
        divergence = np.zeros(n, dtype=np.float32)
        for i in range(n):
            # 1. 计算环境健康度 (Geometric Mean Core)
            health = p_state[i] * c_state[i]
            # 2. 惯性增量：高健康度 + 正向加速度 (物理锁死逻辑)
            if health >= 0.6:
                hab_inc[i] = health * (1.0 + max(0.0, p_acc[i]))
            else:
                hab_inc[i] = 0.0
            # 3. 非线性顶背离检测 (Price High & Rising vs Capital Draining)
            if p_state[i] > 0.7 and c_state[i] < 0.45:
                # 背离强度由价格与资金的 gap 及其加速度耦合决定
                divergence[i] = (p_state[i] - c_state[i]) * (1.0 + abs(p_acc[i]))
        # 4. 向量化滚动 HAB 存量累加
        hab_score = np.zeros(n, dtype=np.float32)
        for i in range(window - 1, n):
            s = 0.0
            for j in range(i - window + 1, i + 1):
                s += hab_inc[j]
            hab_score[i] = s
        return hab_score, divergence

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _numba_calculate_panic_impulse(p_acc, dist_acc):
        """
        【V1.0 · Numba 高性能版】计算恐慌冲击算子
        修改说明：利用 Numba JIT 实现价格负向加速度与派发加速度的非线性耦合计算，实现对空头急跌风险的毫秒级识别。
        版本号：2026.02.11.121
        """
        n = len(p_acc)
        panic = np.zeros(n, dtype=np.float32)
        for i in range(n):
            p_down = abs(min(0.0, p_acc[i]))
            d_up = max(0.0, dist_acc[i])
            panic[i] = (p_down * d_up) ** 0.5
        return panic

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V13.4 · 全向量化极限版】主力拉升意图主计算流程
        修改说明：废弃 df.apply(is_limit_up)，引入向量化涨停检测算子。彻底消除主流程中最后的 Python 循环，实现全链路矩阵运算。
        版本号：2026.02.11.130
        """
        self._probe_output = []
        params = self._get_parameters(config)
        df_index = df.index
        # 1. 基础信号加载
        raw_signals = self._get_raw_signals(df)
        # 2. 【核心优化】向量化涨停检测 (替代 df.apply)
        close_v = raw_signals.get('close', 0.0).values
        pre_close_v = raw_signals.get('pre_close', 0.0).values
        # 简单涨停逻辑向量化：close >= pre_close * 1.099 (兼容处理)
        is_limit_up_vec = (close_v >= np.round(pre_close_v * 1.098, 2)) & (pre_close_v > 0)
        # 3. 物理分量计算
        normalized_signals = self._normalize_raw_signals(df_index, raw_signals)
        mtf_signals = self._calculate_mtf_fused_signals(df, raw_signals, params['mtf_slope_accel_weights'], df_index)
        if self._is_probe_enabled(df): self._diagnose_vol_jerk_anomaly(df_index, raw_signals, mtf_signals)
        factors = self._calculate_market_state_factors(df_index, normalized_signals, mtf_signals)
        phase = self._identify_market_phase(df_index, factors, normalized_signals)
        historical_context = self._calculate_historical_context(df, df_index, raw_signals, mtf_signals, params['historical_context_params'])
        proxy_signals = self._construct_proxy_signals(df_index, mtf_signals, normalized_signals, config)
        refined_period = self._calculate_dynamic_period(df_index, raw_signals, proxy_signals)
        historical_context['dynamic_memory_period'] = refined_period
        dynamic_weights = self._calculate_dynamic_weights(df_index, normalized_signals, proxy_signals, mtf_signals, factors, phase)
        # 4. 意图组件合成
        aggressiveness_score = self._calculate_aggressiveness_score(df_index, raw_signals, mtf_signals, normalized_signals, dynamic_weights)
        control_score = self._calculate_control_score(df_index, raw_signals, mtf_signals, normalized_signals, historical_context)
        obstacle_clearance_score = self._calculate_obstacle_clearance_score(df_index, raw_signals, mtf_signals, normalized_signals, historical_context)
        # 5. 风险与熔断
        total_risk_penalty = self._adjudicate_risk(df_index, raw_signals, mtf_signals, normalized_signals, dynamic_weights, aggressiveness_score, params['rally_intent_synthesis_params'], phase, factors).astype(np.float32)
        bullish_intent = self._synthesize_bullish_intent(df_index, aggressiveness_score, control_score, obstacle_clearance_score, mtf_signals, normalized_signals, dynamic_weights, historical_context, params['rally_intent_synthesis_params'], proxy_signals, total_risk_penalty).astype(np.float32)
        bearish_score = self._calculate_bearish_intent(df_index, raw_signals, mtf_signals, normalized_signals, historical_context, factors).astype(np.float32)
        # 6. 最终意图缝合
        final_rally_intent = (bullish_intent * (1.0 - total_risk_penalty * 0.8) + bearish_score).fillna(0.0).clip(-1, 1)
        final_rally_intent = self._apply_contextual_modulators(df_index, final_rally_intent, proxy_signals, mtf_signals, phase, aggressiveness_score)
        # 应用涨停保护
        final_rally_intent = pd.Series(np.where(is_limit_up_vec & (final_rally_intent.values < 0), 0.0, final_rally_intent.values), index=df_index)
        # 7. 持久化与审计
        self._persist_hab_states(historical_context, final_rally_intent, proxy_signals)
        if self._is_probe_enabled(df):
            self._execute_full_link_probing(df_index, raw_signals, mtf_signals, proxy_signals, historical_context, bullish_intent, final_rally_intent, phase, dynamic_weights)
            self._output_probe_info(df_index, final_rally_intent)
        return final_rally_intent.astype(np.float32)

    def _execute_full_link_probing(self, df_index: pd.Index, raw_signals: Dict, mtf_signals: Dict, proxy_signals: Dict, historical_context: Dict, bullish_intent: pd.Series, final_intent: pd.Series, phase: pd.Series, dynamic_weights: Dict):
        """
        【V1.4 · NumPy 优先诊断版】执行全链路物理审计
        修改说明：全面废弃 .iloc[-1]，采用 .values[-1] 或直接索引访问。解决因容器类型（ndarray vs Series）不确定导致的属性错误。
        版本号：2026.02.11.105
        """
        ts = df_index[-1]
        # self._probe_print(f"=== [FULL_LINK_PHYSICS_AUDIT] {ts.strftime('%Y-%m-%d')} ===")
        # 1. 物理穿透链条审计 (NumPy 访问)
        v_jerk_s = mtf_signals.get('JERK_5_volume_trend', pd.Series(0.0, index=df_index))
        p_acc_s = mtf_signals.get('ACCEL_5_price_trend', pd.Series(0.0, index=df_index))
        v_jerk_val = v_jerk_s.values[-1] if hasattr(v_jerk_s, 'values') else v_jerk_s[-1]
        p_acc_val = p_acc_s.values[-1] if hasattr(p_acc_s, 'values') else p_acc_s[-1]
        self._probe_output.append(f"[L2_PHYSICS] Scaled_P_Accel: {p_acc_val:.4f} | Scaled_V_Jerk: {v_jerk_val:.4f}")
        # 2. 存量层审计
        p_hab_s = historical_context.get('price_memory', {}).get('hab_score', pd.Series(0.5, index=df_index))
        c_hab_s = historical_context.get('capital_memory', {}).get('hab_score', pd.Series(0.5, index=df_index))
        p_hab_val = p_hab_s.values[-1] if hasattr(p_hab_s, 'values') else p_hab_s[-1]
        c_hab_val = c_hab_s.values[-1] if hasattr(c_hab_s, 'values') else c_hab_s[-1]
        phase_val = phase.values[-1] if hasattr(phase, 'values') else phase[-1]
        self._probe_output.append(f"[L3_HAB] P_HAB: {p_hab_val:.4f}, C_HAB: {c_hab_val:.4f} | Phase: {phase_val}")
        # 3. 合成层审计
        b_val = bullish_intent.values[-1] if hasattr(bullish_intent, 'values') else bullish_intent[-1]
        f_val = final_intent.values[-1] if hasattr(final_intent, 'values') else final_intent[-1]
        self._probe_output.append(f"[L5_SYNTHESIS] Bullish_Base: {b_val:.4f} -> Final_Rally_Intent: {f_val:.4f}")

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
        【V8.1 · 运动学扩展版】更新映射清单，将主力净额纳入运动学(S/A/J)计算基准
        版本号：2026.02.10.02
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
        # 新增 mf_net_amount 到运动学计算基座中
        kinematic_bases = {
            'price_trend': 'close_D', 'volume_trend': 'volume_D',
            'net_amount_trend': 'net_amount_D', 'flow_intensity': 'flow_intensity_D',
            'chip_concentration': 'chip_concentration_ratio_D',
            'mf_net_amount': 'net_mf_amount_D' # [新增项]
        }
        for signal_name, col_name in kinematic_bases.items():
            for p in [5, 13, 21, 55]:
                for metric in ['SLOPE', 'ACCEL', 'JERK']:
                    col_map[f'{metric}_{p}_{signal_name}'] = f'{metric}_{p}_{col_name}'
        return col_map

    def _get_raw_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V8.1 · 效率优化版】基于精简后的映射清单自动加载原始信号
        修改说明：增加显式的 float32 类型转换，降低内存占用并加速后续向量化计算。
        版本号：2026.02.11.01
        """
        raw_signals = {}
        method_name = "_get_raw_signals"
        column_map = self._get_required_column_map()
        for signal_name, col_name in column_map.items():
            # 强制转换为 float32 以加速计算
            raw_signals[signal_name] = self.helper._get_safe_series(
                df, col_name, 0.0, method_name=method_name
            ).astype(np.float32)
        return raw_signals

    def _diagnose_vol_jerk_anomaly(self, df_index: pd.Index, raw_signals: Dict, mtf_signals: Dict):
        """
        【V1.2 · 容器适配增强版】定位 Jerk 数值异常根源
        修改说明：统一使用 .values[-1] 提取标量，强化对 raw_signals 容器类型的兼容，防止诊断中断。
        版本号：2026.02.11.106
        """
        v_j_s = mtf_signals.get('JERK_5_volume_trend', pd.Series(0.0, index=df_index))
        v_j_arr = v_j_s.values if hasattr(v_j_s, 'values') else v_j_s
        v_j_raw = v_j_arr[-1]
        # 基于 NumPy 数据的稳健统计
        med = np.median(v_j_arr[-60:]) if len(v_j_arr) >= 5 else 0.0
        mad = np.median(np.abs(v_j_arr[-60:] - med)) if len(v_j_arr) >= 5 else 0.0
        vol_s = raw_signals.get('volume', pd.Series(0.0, index=df_index))
        vol_val = vol_s.values[-1] if hasattr(vol_s, 'values') else vol_s[-1]
        s_v_s = mtf_signals.get('SLOPE_5_volume_trend', pd.Series(0.0, index=df_index))
        a_v_s = mtf_signals.get('ACCEL_5_volume_trend', pd.Series(0.0, index=df_index))
        s_val = s_v_s.values[-1] if hasattr(s_v_s, 'values') else s_v_s[-1]
        a_val = a_v_s.values[-1] if hasattr(a_v_s, 'values') else a_v_s[-1]
        # self._probe_print(f"=== [PHYSICS_DIAGNOSIS] Vol_SAJ_Chain ===")
        # self._probe_print(f"  > Raw_Volume: {vol_val:.0f}")
        # self._probe_print(f"  > S/A/J_Raw: S={s_val:.2f}, A={a_val:.2f}, J={v_j_raw:.2f}")
        # self._probe_print(f"  > MAD_Audit: Median={med:.2f}, MAD={mad:.2f}, Z-Score={(v_j_raw-med)/(mad*1.4826+1e-9):.2f}")

    def _calculate_mtf_fused_signals(self, df: pd.DataFrame, raw_signals: Dict[str, pd.Series], mtf_slope_accel_weights: Dict, df_index: pd.Index) -> Dict[str, pd.Series]:
        """
        【V8.6 · 容器转换修复版】MTF 跨周期信号融合
        修改说明：修复 'numpy.ndarray' object has no attribute 'ewm' 报错。强制将 SLOPE/ACCEL/JERK 归一化结果封装回 pd.Series，确保内存记忆计算模块可调用 Pandas 扩展方法。
        版本号：2026.02.11.85
        """
        mtf_signals = {}
        for key, series in raw_signals.items():
            if any(p in key for p in ['SLOPE_', 'ACCEL_', 'JERK_']):
                s_f32 = series.values.astype(np.float32)
                med = pd.Series(s_f32).rolling(60, min_periods=5).median().values
                mad = pd.Series(np.abs(s_f32 - med)).rolling(60, min_periods=5).median().values
                # 核心修复：封装回 pd.Series 以支持下游模块的 .ewm() 调用
                mtf_signals[key] = pd.Series(np.tanh((s_f32 - med) / (mad * 4.4478 + 1e-9)), index=df_index).astype(np.float32)
        weights = {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}
        for sig_name, col_name in {'price_trend': 'close_D', 'volume_trend': 'volume_D', 'net_amount_trend': 'net_amount_D'}.items():
            fused_score = np.zeros(len(df_index), dtype=np.float32)
            col_v = df[col_name].values.astype(np.float32)
            for win, w in weights.items():
                diff_v = np.zeros_like(col_v)
                diff_v[win:] = col_v[win:] - col_v[:-win]
                std_v = self._numba_rolling_std(diff_v, win * 3)
                fused_score += np.tanh(diff_v / (std_v * 2.0 + 1e-9)) * w
            mtf_signals[f'mtf_{sig_name}'] = pd.Series(fused_score, index=df_index).clip(-1, 1)
        return mtf_signals

    def _calculate_historical_context(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], params: Dict) -> Dict[str, pd.Series]:
        """
        【V5.5 · 冷启动置信度强化版】历史上下文计算调度中心
        修改说明：引入Confidence Mask逻辑。根据序列长度自动生成预热权重，解决冷启动阶段Rolling窗口未对齐导致的信号漂移。
        版本号：2026.02.11.60
        """
        if not get_param_value(params.get('enabled'), True): return self._get_empty_context(df_index)
        h_init = self._load_hab_states()
        p_mem = self._calculate_price_memory(df_index, raw_signals, mtf_signals, params, h_init.get('price_hab', 0.0))
        c_mem = self._calculate_capital_memory(df_index, raw_signals, mtf_signals, params, h_init.get('capital_hab', 0.0))
        ch_mem = self._calculate_chip_memory(df_index, raw_signals, mtf_signals, params, h_init.get('chip_hab', 0.0))
        s_mem = self._calculate_sentiment_memory(df_index, raw_signals, mtf_signals, params, h_init.get('sentiment_hab', 0.0))
        int_mem_raw = self._fuse_integrated_memory(p_mem, c_mem, ch_mem, s_mem, params)
        # 核心新增：冷启动置信度掩码 (21周期内线性平滑)
        n_len = len(df_index)
        confidence_mask = np.tanh(np.arange(n_len, dtype=np.float32) / 21.0)
        int_mem = int_mem_raw * confidence_mask
        return {
            "price_memory": p_mem, "capital_memory": c_mem, "chip_memory": ch_mem, "sentiment_memory": s_mem,
            "integrated_memory": int_mem.astype(np.float32), "phase_sync": self._detect_phase_synchronization(df_index, p_mem, c_mem, int_mem),
            "memory_quality": self._assess_memory_quality(df_index, p_mem, c_mem, ch_mem, s_mem) * confidence_mask,
            "hc_enabled": True, "dynamic_memory_period": self._calculate_dynamic_period(df_index, raw_signals, {})
        }

    def _load_hab_states(self) -> Dict[str, float]:
        """
        【V1.2 · 反向比例还原版】从原子状态机载入HAB历史状态
        修改说明：增加比例还原因子（如Cap=5e7），将持久化时的归一化分值还原为物理原始量纲，适配计算模块。
        版本号：2026.02.11.64
        """
        keys = {'price_hab': ('_HAB_STATE_PRICE', 1.0), 'capital_hab': ('_HAB_STATE_CAPITAL', 5e7), 'chip_hab': ('_HAB_STATE_CHIP', 1.0), 'sentiment_hab': ('_HAB_STATE_SENTIMENT', 1.0)}
        states = {}
        for k, (v_key, scale) in keys.items():
            norm_val = np.float32(self.strategy.atomic_states.get(v_key, 0.0))
            # 反向还原逻辑: val = arctanh(norm) * scale
            states[k] = np.arctanh(np.clip(norm_val, -0.99, 0.99)) * scale
        # if any(v != 0 for v in states.values()): self._probe_print(f"[HAB_LOAD_V1.2] 成功物理还原历史存量: Cap={states['capital_hab']:.2e}")
        return states

    def _persist_hab_states(self, historical_context: Dict, final_rally_intent: pd.Series, proxy_signals: Dict[str, pd.Series]):
        """
        【V1.3 · 状态自适应持久化版】将HAB最新状态持久化
        修改说明：引入 State-Adaptive Persistence 机制。当意图强烈且信号共振时，动态提升 atomic_states 更新权重，实现对主力拉升物理惯性的加速记忆固化。
        版本号：2026.02.11.95
        """
        try:
            intent_val = float(final_rally_intent.values[-1])
            burst_val = float(proxy_signals.get('sync_burst_score', pd.Series([0.0])).values[-1])
            update_weight = np.clip(0.6 + intent_val * 0.15 + burst_val * 0.2, 0.6, 0.95)
            m_keys = [('price_memory', '_HAB_STATE_PRICE', 1.0), ('capital_memory', '_HAB_STATE_CAPITAL', 5e7), ('chip_memory', '_HAB_STATE_CHIP', 1.0), ('sentiment_memory', '_HAB_STATE_SENTIMENT', 1.0)]
            for m_key, s_key, scale in m_keys:
                buf = historical_context.get(m_key, {}).get('hab_buffer_raw', pd.Series([0.0]))
                new_norm_val = float(np.tanh((buf.values[-1] if len(buf) > 0 else 0.0) / np.float32(scale)))
                old_val = float(self.strategy.atomic_states.get(s_key, 0.0))
                self.strategy.atomic_states[s_key] = old_val * (1.0 - update_weight) + new_norm_val * update_weight
            # if self._is_probe_enabled(pd.DataFrame()): self._probe_print(f"[HAB_PERSIST_V1.3] 自适应持久化完成 (Weight: {update_weight:.4f}, Cap: {self.strategy.atomic_states['_HAB_STATE_CAPITAL']:.4f})")
        except Exception as e: self._probe_print(f"[HAB_PERSIST_ERROR] 自适应持久化失败: {str(e)}")

    def _detect_phase_synchronization(self, df_index: pd.Index, price_memory: Dict, capital_memory: Dict, integrated_memory: pd.Series) -> pd.Series:
        """
        【V5.2 · 向量化安全加固版】相位同步检测
        修改说明：强化输入类型检查，确保在调用 .corr() 前输入严格为 pd.Series。优化 Fisher 变换的向量化执行路径，提升同步强度感知的鲁棒性。
        版本号：2026.02.11.86
        """
        # 显式转换为 Series 确保安全
        p_trend_raw = pd.Series(price_memory.get("trend_memory", 0.0), index=df_index).astype(np.float32)
        c_flow_raw = pd.Series(capital_memory.get("composite_capital_flow", 0.0), index=df_index).astype(np.float32)
        p_trend = p_trend_raw.ewm(span=3).mean()
        c_flow = c_flow_raw.ewm(span=3).mean()
        # 向量化计算滚动相关性
        corr = p_trend.rolling(window=21, min_periods=5).corr(c_flow).fillna(0.0).values
        # 向量化 Fisher 变换
        fisher = 0.5 * np.log((1.0 + np.clip(corr, -0.99, 0.99)) / (1.0 - np.clip(corr, -0.99, 0.99)))
        sync_strength = np.tanh(fisher)
        # 提取物理稳定性约束
        p_jerk = pd.Series(price_memory.get("jerk_memory", 0.0), index=df_index).astype(np.float32).abs().values
        stability = 1.0 / (1.0 + p_jerk * 10.0)
        return pd.Series(sync_strength * stability, index=df_index).clip(-1, 1).astype(np.float32)

    def _assess_memory_quality(self, df_index: pd.Index, price_memory: Dict, capital_memory: Dict, chip_memory: Dict, sentiment_memory: Dict) -> pd.Series:
        """
        【V5.1 · 物理量纲加速版】记忆质量评估算法
        修改说明：全量采用 float32，向量化合成物理稳定性映射，利用 Rolling Rank 的矩阵形式加速质量 HAB 的累积计算。
        版本号：2026.02.11.35
        """
        p_j = price_memory.get("jerk_memory", pd.Series(0.0, index=df_index)).astype(np.float32).abs().values
        c_j = capital_memory.get("cap_jerk", pd.Series(0.0, index=df_index)).astype(np.float32).abs().values
        k_stab = 1.0 / (1.0 + (p_j * 0.6 + c_j * 0.4) * 15.0)
        consis = self._calculate_memory_consistency(price_memory, capital_memory, chip_memory, sentiment_memory).values
        quality_inc = k_stab * consis
        q_hab = pd.Series(quality_inc, index=df_index).rolling(window=34).rank(pct=True).fillna(0.5).values
        q_score = q_hab * 0.4 + k_stab * 0.3 + consis * 0.3
        return pd.Series(q_score, index=df_index).clip(0, 1).astype(np.float32)

    def _calculate_dynamic_period(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], proxy_signals: Dict[str, pd.Series]) -> pd.Series:
        """
        【V5.2 · Numba 加速自适应版】动态记忆周期计算
        修改说明：利用 Numba 加速 $ER$ 计算，并引入 Fisher-Burst 引导的自适应周期压缩。在共振爆发时自动缩短窗口以捕捉即时物理突变。
        版本号：2026.02.11.65
        """
        pct_change = raw_signals.get('pct_change', pd.Series(0.0, index=df_index)).fillna(0.0).values.astype(np.float32)
        # 1. Numba 加速效率比计算 (10日窗口)
        er = self._numba_calculate_er(pct_change, 10)
        # 2. 波动率位置计算 (向量化)
        volatility = pd.Series(pct_change, index=df_index).rolling(window=21).std().fillna(0.01).astype(np.float32).values
        v_min = pd.Series(volatility).rolling(60, min_periods=10).min().values
        v_max = pd.Series(volatility).rolling(60, min_periods=10).max().values
        vol_pos = np.clip((volatility - v_min) / (v_max - v_min + 1e-9), 0, 1)
        # 3. 提取一致性爆发信号
        burst_score = proxy_signals.get('sync_burst_score', pd.Series(0.0, index=df_index)).values.astype(np.float32)
        # 4. 周期合成逻辑: $Period = 21 \times (1 + ER \times 0.4 - Vol\_Pos \times 0.3) \times (1 - Burst \times 0.4)$
        # 物理逻辑：趋势效率越高周期越长（稳定性）；波动或共振爆发时周期缩短（灵敏度）
        raw_period = 21.0 * (1.0 + er * 0.4 - vol_pos * 0.3) * (1.0 - burst_score * 0.4)
        # 5. 最终平滑与类型转换
        final_period = pd.Series(raw_period, index=df_index).ewm(span=5).mean().clip(5, 55).round().fillna(21).astype(np.int32)
        # if self._is_probe_enabled(pd.DataFrame(index=df_index)):
        #     self._probe_print(f"[PERIOD_BURST_PROBE] ER: {er[-1]:.2f} | Burst: {burst_score[-1]:.2f} -> Final: {final_period.iloc[-1]}")
        return final_period

    def _calculate_price_memory(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], params: Dict, initial_hab: float = 0.0) -> Dict[str, pd.Series]:
        """
        【V5.4 · 衰减注入版】价格记忆深度计算
        修改说明：重构initial_hab注入逻辑。采用衰减系数将历史存量融入rolling序列，防止静态偏移。
        版本号：2026.02.11.61
        """
        decay = np.float32(get_param_value(params.get('price_memory_decay'), 0.94))
        hab_win = int(get_param_value(params.get('hab_window'), 55))
        s = mtf_signals.get('SLOPE_5_price_trend', pd.Series(0, index=df_index)).astype(np.float32).ewm(alpha=1-decay).mean()
        a = mtf_signals.get('ACCEL_5_price_trend', pd.Series(0, index=df_index)).astype(np.float32).ewm(alpha=1-decay).mean()
        j = mtf_signals.get('JERK_5_price_trend', pd.Series(0, index=df_index)).astype(np.float32).ewm(alpha=0.2).mean()
        abs_strength = np.tanh(raw_signals.get('absolute_change_strength', pd.Series(0, index=df_index)).astype(np.float32) / 0.05)
        hab_inc = raw_signals.get('chip_stability', pd.Series(0.5, index=df_index)).astype(np.float32) * a.clip(lower=0) * abs_strength
        # 衰减注入逻辑：initial_hab 仅在序列初期按时间步指数衰减
        startup_decay = np.exp(-np.arange(len(df_index)) / 13.0).astype(np.float32)
        hab_buffer_raw = hab_inc.rolling(window=hab_win, min_periods=1).sum().fillna(0.0).astype(np.float32) + (initial_hab * startup_decay)
        hab_score = (hab_buffer_raw.rolling(hab_win).rank(pct=True)).fillna(0.5).astype(np.float32)
        k_sum = (s.values * 0.4 + a.values * 0.4 + j.values * 0.2)
        integrated_mem = pd.Series(k_sum, index=df_index).clip(0, 1) * 0.5 + hab_score * 0.5
        return {"trend_memory": s, "accel_memory": a, "jerk_memory": j, "hab_score": hab_score, "hab_buffer_raw": hab_buffer_raw, "integrated_price_memory": integrated_mem.astype(np.float32)}

    def _calculate_capital_memory(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], params: Dict, initial_hab: float = 0.0) -> Dict[str, pd.Series]:
        """
        【V5.4 · 衰减注入版】资金记忆深度计算
        修改说明：优化历史存量初值注入，结合 startup_decay 解决切片计算时的冷启动跳变。
        版本号：2026.02.11.62
        """
        decay = np.float32(get_param_value(params.get('capital_memory_decay'), 0.92))
        hab_win = int(get_param_value(params.get('capital_hab_window'), 34))
        cs = mtf_signals.get('SLOPE_5_mf_net_amount', pd.Series(0, index=df_index)).astype(np.float32).ewm(alpha=1-decay).mean()
        ca = mtf_signals.get('ACCEL_5_mf_net_amount', pd.Series(0, index=df_index)).astype(np.float32).ewm(alpha=1-decay).mean()
        mf_flow = raw_signals.get('net_mf_amount', pd.Series(0, index=df_index)).astype(np.float32)
        hab_inc = mf_flow * raw_signals.get('flow_consistency', pd.Series(0.5, index=df_index)).astype(np.float32).clip(0, 1)
        startup_decay = np.exp(-np.arange(len(df_index)) / 8.0).astype(np.float32)
        hab_buffer_raw = hab_inc.rolling(window=hab_win, min_periods=1).sum().fillna(0.0).astype(np.float32) + (initial_hab * startup_decay)
        vol = raw_signals.get('pct_change', pd.Series(0, index=df_index)).astype(np.float32).rolling(21).std().fillna(0.02)
        hab_score = 1.0 / (1.0 + np.exp(-hab_buffer_raw.values / (vol.values * 5e7 + 1e-9)))
        hab_score_s = pd.Series(hab_score, index=df_index).astype(np.float32)
        integrated_mem = (cs.clip(0, 1) * 0.4 + hab_score_s * 0.6).clip(0, 1)
        return {"integrated_capital_memory": integrated_mem, "hab_score": hab_score_s, "hab_buffer_raw": hab_buffer_raw, "cap_accel": ca}

    def _calculate_chip_memory(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], params: Dict, initial_hab: float = 0.0) -> Dict[str, pd.Series]:
        """
        【V5.4 · 衰减注入版】筹码记忆深度计算
        修改说明：同步重构 initial_hab 注入逻辑，确保熵能记忆在冷启动阶段的确定性。
        版本号：2026.02.11.63
        """
        decay = np.float32(get_param_value(params.get('chip_memory_decay'), 0.95))
        hab_win = int(get_param_value(params.get('chip_hab_window'), 89))
        chip_s = mtf_signals.get('SLOPE_5_chip_concentration', pd.Series(0, index=df_index)).astype(np.float32).ewm(alpha=1-decay).mean()
        chip_a = mtf_signals.get('ACCEL_5_chip_concentration', pd.Series(0, index=df_index)).astype(np.float32).ewm(alpha=1-decay).mean()
        chip_j = mtf_signals.get('JERK_5_chip_concentration', pd.Series(0, index=df_index)).astype(np.float32).ewm(alpha=0.15).mean()
        turn_rate = raw_signals.get('turnover_rate', pd.Series(0, index=df_index)).astype(np.float32)
        hab_inc = (np.clip(chip_s.values, 0, None) * raw_signals.get('chip_stability', pd.Series(0.5, index=df_index)).astype(np.float32).values * turn_rate.values)
        startup_decay = np.exp(-np.arange(len(df_index)) / 21.0).astype(np.float32)
        hab_buffer_raw = pd.Series(hab_inc, index=df_index).rolling(window=hab_win, min_periods=1).sum().fillna(0.0).astype(np.float32) + (initial_hab * startup_decay)
        hab_score = hab_buffer_raw.rolling(hab_win).rank(pct=True).fillna(0.5).astype(np.float32)
        entropy = raw_signals.get('chip_entropy', pd.Series(0.5, index=df_index)).astype(np.float32)
        ent_mem = (np.log1p(entropy) / np.log1p(entropy).rolling(55, min_periods=1).max().replace(0, 1)).clip(0, 1).astype(np.float32)
        k_vec = np.clip(chip_s.values * 0.4 + chip_a.values * 0.4 + chip_j.values * 0.2, 0, 1)
        integrated_chip = (hab_score.values * 0.4 + k_vec * 0.3 + (1.0 - ent_mem.values) * 0.3)
        return {"chip_slope": chip_s, "chip_accel": chip_a, "hab_score": hab_score, "hab_buffer_raw": hab_buffer_raw, "integrated_chip_memory": pd.Series(integrated_chip, index=df_index).clip(0, 1).astype(np.float32)}

    def _calculate_sentiment_memory(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], params: Dict, initial_hab: float = 0.0) -> Dict[str, pd.Series]:
        """
        【V5.3 · 向量化加速版】情绪记忆深度计算
        修改说明：采用矩阵化SAJ合成，优化温压存量HAB的Tanh归一化性能，全链路强制float32。
        版本号：2026.02.11.24
        """
        decay = np.float32(get_param_value(params.get('sentiment_memory_decay'), 0.90))
        hab_win = int(get_param_value(params.get('sentiment_hab_window'), 21))
        ss = mtf_signals.get('SLOPE_5_market_sentiment_trend', pd.Series(0, index=df_index)).astype(np.float32).ewm(alpha=1-decay).mean()
        sa = mtf_signals.get('ACCEL_5_market_sentiment_trend', pd.Series(0, index=df_index)).astype(np.float32).ewm(alpha=1-decay).mean()
        sj = mtf_signals.get('JERK_5_market_sentiment_trend', pd.Series(0, index=df_index)).astype(np.float32).ewm(alpha=0.3).mean()
        hab_inc = raw_signals.get('market_sentiment', pd.Series(0.5, index=df_index)).astype(np.float32) * raw_signals.get('industry_breadth', pd.Series(0.5, index=df_index)).astype(np.float32) * (1.0 + sa.clip(lower=0))
        hab_buffer_raw = hab_inc.rolling(window=hab_win, min_periods=1).sum().fillna(0.0).astype(np.float32) + initial_hab
        roll_med = hab_buffer_raw.rolling(60, min_periods=5).median()
        roll_std = hab_buffer_raw.rolling(60, min_periods=5).std().replace(0, 1e-9)
        hab_score = (np.tanh((hab_buffer_raw - roll_med) / roll_std) * 0.5 + 0.5).fillna(0.5).astype(np.float32)
        k_sum = (ss.values * 0.3 + sa.values * 0.5 + sj.values * 0.2)
        integrated_sent = ((np.clip(k_sum, -1, 1) * 0.5 + 0.5) * 0.4 + hab_score.values * 0.6)
        return {"integrated_sentiment_memory": pd.Series(integrated_sent, index=df_index).clip(0, 1).astype(np.float32), "hab_score": hab_score, "hab_buffer_raw": hab_buffer_raw, "sentiment_slope": ss, "sentiment_accel": sa}

    def _fuse_integrated_memory(self, price_memory: Dict, capital_memory: Dict, chip_memory: Dict, sentiment_memory: Dict, params: Dict) -> pd.Series:
        """
        【V5.2 · 矩阵融合稳健加速版】综合记忆融合算法
        修改说明：修复.get(..., 0).values导致的类型报错。利用向量化逻辑处理HAB偏移量与阶段识别，保持float32计算精度。
        版本号：2026.02.11.20
        """
        df_index = price_memory["integrated_price_memory"].index
        p_base = price_memory["integrated_price_memory"].values.astype(np.float32)
        c_base = capital_memory["integrated_capital_memory"].values.astype(np.float32)
        ch_base = chip_memory["integrated_chip_memory"].values.astype(np.float32)
        s_base = sentiment_memory["integrated_sentiment_memory"].values.astype(np.float32)
        p_hab = price_memory.get("hab_score", pd.Series(0.5, index=df_index)).values.astype(np.float32)
        c_hab = capital_memory.get("hab_score", pd.Series(0.5, index=df_index)).values.astype(np.float32)
        ch_hab = chip_memory.get("hab_score", pd.Series(0.5, index=df_index)).values.astype(np.float32)
        s_hab = sentiment_memory.get("hab_score", pd.Series(0.5, index=df_index)).values.astype(np.float32)
        integrated_hab = (p_hab * 0.2 + c_hab * 0.3 + ch_hab * 0.3 + s_hab * 0.2)
        w_p = 0.2 + integrated_hab * 0.2
        w_c = 0.4 - integrated_hab * 0.1
        w_ch = 0.3 - integrated_hab * 0.1
        w_s = 0.1 + integrated_hab * 0.1
        base_fusion = p_base * w_p + c_base * w_c + ch_base * w_ch + s_base * w_s
        def _safe_arr(m_dict, key):
            return pd.Series(m_dict.get(key, 0.0), index=df_index).values.astype(np.float32)
        slopes_mat = np.column_stack([_safe_arr(price_memory, "trend_memory"), _safe_arr(capital_memory, "composite_capital_flow"), _safe_arr(chip_memory, "chip_slope"), _safe_arr(sentiment_memory, "sentiment_slope")])
        accels_mat = np.column_stack([_safe_arr(price_memory, "accel_memory"), _safe_arr(capital_memory, "cap_accel"), _safe_arr(chip_memory, "chip_accel"), _safe_arr(sentiment_memory, "sentiment_accel")])
        slope_consistency = (np.all(slopes_mat > 0, axis=1) | np.all(slopes_mat < 0, axis=1)).astype(np.float32)
        accel_synergy = (accels_mat > 0).sum(axis=1) / 4.0
        cap_price_offset = c_hab - p_hab
        chip_price_offset = ch_hab - p_hab
        latent_premium = (np.clip(cap_price_offset, 0, 1) * 0.3 + np.clip(chip_price_offset, 0, 1) * 0.2) * (1.0 - p_hab)
        distribution_penalty = np.abs(np.clip(cap_price_offset, -1, 0)) * 0.4 * p_hab
        synergy_boost = (slope_consistency * 0.15 + accel_synergy * 0.15)
        final_mem = (base_fusion * (0.85 + integrated_hab * 0.3) * (1.0 + synergy_boost + latent_premium - distribution_penalty))
        return pd.Series(final_mem, index=df_index).clip(0, 1).astype(np.float32)

    def _detect_market_state(self, price_mem: Dict, capital_mem: Dict, sentiment_mem: Dict, chip_mem: Dict) -> str:
        """
        【V5.1 · 效率增强版】市场状态检测
        修改说明：利用 float32 标量加速末端状态判定，移除冗余的 Series 封装。
        版本号：2026.02.11.42
        """
        p_s = float(price_mem.get("trend_memory", pd.Series([0.0])).iloc[-1])
        p_a = float(price_mem.get("accel_memory", pd.Series([0.0])).iloc[-1])
        p_j = float(price_mem.get("jerk_memory", pd.Series([0.0])).iloc[-1])
        c_s = float(capital_mem.get("composite_capital_flow", pd.Series([0.0])).iloc[-1])
        p_h = float(price_mem.get("hab_score", pd.Series([0.5])).iloc[-1])
        c_h = float(capital_mem.get("hab_score", pd.Series([0.5])).iloc[-1])
        i_h = p_h * 0.4 + c_h * 0.6
        if p_s > 0.1 and c_s > 0.05 and p_a > 0 and i_h > 0.6: return "trending_up"
        if abs(p_j) > 0.15 or (p_s * p_a < 0 and abs(p_a) > 0.1): return "reversing"
        if p_s < -0.1 and i_h < 0.4: return "trending_down"
        if abs(p_a) < 0.05 and c_s > 0 and i_h > 0.5: return "consolidating"
        return "trending_up" if i_h > 0.7 else "consolidating"

    def _calculate_memory_consistency(self, price_mem: Dict, capital_mem: Dict, chip_mem: Dict, sentiment_memory: Dict) -> pd.Series:
        """
        【V5.4 · 矩阵运算提速版】记忆一致性深度计算
        修改说明：移除高开销的 unstack().mean() 逻辑。改用矩阵求和法计算 4x4 相关性矩阵的非对角线均值，显著提升多维信号共振审计的效率。
        版本号：2026.02.11.87
        """
        df_index = price_mem["integrated_price_memory"].index
        def _safe_arr(m_dict, key):
            return pd.Series(m_dict.get(key, 0.0), index=df_index).values.astype(np.float32)
        # 构建加速度矩阵 (N, 4)
        accel_mat = np.column_stack([_safe_arr(price_mem, "accel_memory"), _safe_arr(capital_mem, "cap_accel"), _safe_arr(chip_mem, "chip_accel"), _safe_arr(sentiment_memory, "sentiment_accel")])
        res_count = (accel_mat > 0).sum(axis=1)
        res_map = np.array([0.0, 0.1, 0.3, 0.7, 1.0], dtype=np.float32)
        instant_resonance = res_map[res_count]
        # 物理 HAB 增量
        hab_inc = instant_resonance * np.abs(accel_mat).mean(axis=1)
        hab_raw = pd.Series(hab_inc, index=df_index).rolling(window=13).sum().fillna(0.0).astype(np.float32)
        # HAB 归一化
        roll_med = hab_raw.rolling(60).median()
        roll_std = hab_raw.rolling(60).std().replace(0, 1e-9)
        consistency_hab = (np.tanh((hab_raw - roll_med) / roll_std) * 0.5 + 0.5).fillna(0.5)
        # 效率重构：计算相关性矩阵均值 (排除自相关1.0)
        df_accel = pd.DataFrame(accel_mat)
        # 矩阵元素总和减去对角线的4个1.0，除以12个非对角线元素
        rolling_corr_sum = df_accel.rolling(window=10).corr().groupby(level=0).sum().sum(axis=1)
        avg_corr = (rolling_corr_sum.values - 4.0) / 12.0
        return (pd.Series(avg_corr, index=df_index).fillna(0.5) * 0.2 + instant_resonance * 0.3 + consistency_hab * 0.5).clip(0, 1).astype(np.float32)

    def _get_empty_context(self, df_index: pd.Index) -> Dict[str, pd.Series]:
        """
        【V3.1 · 内存优化版】返回空的历史上下文
        修改说明：统一使用float32，降低禁用上下文时的内存预分配开销。
        版本号：2026.02.11.33
        """
        empty_s = pd.Series(0.5, index=df_index).astype(np.float32)
        e_dict = {k: empty_s for k in ["trend_memory", "momentum_memory", "volatility_memory", "support_resistance_memory", "integrated_price_memory", "composite_capital_flow", "persistence_score", "anomaly_score", "efficiency_memory", "integrated_capital_memory", "entropy_memory", "concentration_migration", "stability_memory", "pressure_memory", "integrated_chip_memory", "sentiment_momentum", "sentiment_divergence", "sentiment_extreme", "sentiment_consistency", "integrated_sentiment_memory"]}
        return {
            "price_memory": {k: v for k, v in e_dict.items() if "price" in k or k in ["trend_memory", "momentum_memory", "volatility_memory", "support_resistance_memory"]},
            "capital_memory": {k: v for k, v in e_dict.items() if "capital" in k or k in ["composite_capital_flow", "persistence_score", "anomaly_score", "efficiency_memory"]},
            "chip_memory": {k: v for k, v in e_dict.items() if "chip" in k or k in ["entropy_memory", "concentration_migration", "stability_memory", "pressure_memory"]},
            "sentiment_memory": {k: v for k, v in e_dict.items() if "sentiment" in k},
            "integrated_memory": empty_s, "phase_sync": empty_s, "memory_quality": empty_s, "hc_enabled": False, "dynamic_memory_period": pd.Series(21, index=df_index).astype(np.int32)
        }

    def _normalize_raw_signals(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V3.1 · 向量化归一化增强版】全量原始信号归一化中心
        修改说明：将原本分散的循环逻辑整合为向量化批处理，统一使用float32，并在对数处理阶段利用Numpy加速，大幅降低特征预处理耗时。
        版本号：2026.02.11.07
        """
        norm_sigs = {}
        def _robust_scale_vec(s, window=60, bipolar=False):
            s_f32 = s.astype(np.float32)
            med = s_f32.rolling(window).median()
            mad = (s_f32 - med).abs().rolling(window).median()
            res = (s_f32 - med) / (mad * 1.4826 + 1e-9)
            return np.tanh(res).clip(-1, 1) if bipolar else (np.tanh(res) * 0.5 + 0.5).clip(0, 1)
        norm_sigs['pct_change_norm'] = np.tanh(raw_signals.get('pct_change', 0).astype(np.float32) / 0.1).clip(-1, 1)
        norm_sigs['absolute_change_strength_norm'] = np.tanh(raw_signals.get('absolute_change_strength', 0).astype(np.float32) / 0.05).clip(0, 1)
        for col in ['volume_ratio', 'turnover_rate', 'net_amount_ratio']:
            val = raw_signals.get(col, 0).astype(np.float32).clip(lower=0)
            log_val = np.log1p(val)
            norm_sigs[f'{col}_norm'] = (log_val / log_val.rolling(60).max().replace(0, 1)).clip(0, 1)
        for col in ['flow_acceleration', 'flow_consistency', 'flow_intensity', 'inflow_persistence', 'net_mf_amount']:
            norm_sigs[f'{col}_norm'] = _robust_scale_vec(raw_signals.get(col, 0), bipolar=True)
        norm_sigs['chip_concentration_ratio_norm'] = raw_signals.get('chip_concentration_ratio', 0.5).astype(np.float32).clip(0, 1)
        norm_sigs['chip_stability_norm'] = _robust_scale_vec(raw_signals.get('chip_stability', 0.5), bipolar=False)
        for col in ['market_sentiment', 'industry_breadth', 'industry_leader']:
            norm_sigs[f'{col}_norm'] = (raw_signals.get(col, 0.5).astype(np.float32)).clip(0, 1)
        for col in ['accumulation_score', 'distribution_score', 'behavior_accumulation', 'behavior_distribution']:
            norm_sigs[f'{col}_norm'] = _robust_scale_vec(raw_signals.get(col, 0), bipolar=True)
        return norm_sigs

    def _construct_proxy_signals(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], config: Dict) -> Dict[str, pd.Series]:
        """
        【V7.2 · 修正版】代理信号构建中枢
        修改说明：修复 rolling().apply() 导致的 TypeError。采用显式两两相关性计算 RS、资金与情绪代理的共振度，并应用 Fisher 锐化。
        版本号：2026.02.10.375
        """
        # 1. 核心代理信号计算
        rs_res = self._calculate_enhanced_rs_proxy(df_index, mtf_signals, normalized_signals, config)
        cap_res = self._calculate_enhanced_capital_proxy(df_index, mtf_signals, normalized_signals, config)
        sent_res = self._calculate_enhanced_sentiment_proxy(df_index, mtf_signals, normalized_signals, config)
        liq_res = self._calculate_enhanced_liquidity_proxy(df_index, mtf_signals, normalized_signals, config)
        vol_res = self._calculate_enhanced_volatility_proxy(df_index, mtf_signals, normalized_signals, config)
        risk_res = self._calculate_enhanced_risk_preference_proxy(df_index, mtf_signals, normalized_signals, config)

        proxy_cores = {
            "rs": rs_res.get("enhanced_rs_proxy", pd.Series(0.5, index=df_index)),
            "capital": cap_res.get("enhanced_capital_proxy", pd.Series(0.5, index=df_index)),
            "sentiment": sent_res.get("enhanced_sentiment_proxy", pd.Series(0.5, index=df_index))
        }

        # 2. 【核心修复】显式成对相关性审计 (取代报错的 apply)
        c_rs = proxy_cores["rs"]; c_cap = proxy_cores["capital"]; c_sent = proxy_cores["sentiment"]
        # 计算 13 日滚动两两相关性
        corr_rc = c_rs.rolling(window=13, min_periods=5).corr(c_cap).fillna(0)
        corr_rs = c_rs.rolling(window=13, min_periods=5).corr(c_sent).fillna(0)
        corr_cs = c_cap.rolling(window=13, min_periods=5).corr(c_sent).fillna(0)
        # 均值共振度
        avg_sync_raw = (corr_rc + corr_rs + corr_cs) / 3.0
        # 3. Fisher 变换锐化与一致性爆发
        fisher_norm = np.tanh(0.5 * np.log((1 + avg_sync_raw.clip(-0.99, 0.99)) / (1 - avg_sync_raw.clip(-0.99, 0.99)))).clip(0, 1)
        sync_burst_score = np.tanh(fisher_norm.diff(1).diff(1).clip(lower=0).rolling(5).mean().fillna(0) * 50).clip(0, 1)

        # 4. 综合信号质量评估 (Fisher-SNR)
        quality_list = []
        for name, series in proxy_cores.items():
            noise = series.diff().abs().rolling(13, min_periods=1).sum().replace(0, 1e-9)
            signal = series.diff(13).abs().fillna(0)
            snr_raw = (signal / noise).clip(0, 1)
            quality_list.append(np.tanh(0.5 * np.log((1 + snr_raw.clip(0, 0.99)) / (1 - snr_raw.clip(0, 0.99)))))
        individual_quality = pd.concat(quality_list, axis=1).mean(axis=1).fillna(0.7)
        combined_signal_quality = (individual_quality * 0.4 + fisher_norm * 0.4 + sync_burst_score * 0.2).clip(0, 1)

        # 5. 动态权重合成与审计指标回填
        final_proxy_signals = self._dynamic_weighted_synthesis(rs_res, cap_res, sent_res, liq_res, vol_res, risk_res, combined_signal_quality, config)
        final_proxy_signals['combined_signal_quality'] = combined_signal_quality
        final_proxy_signals['sync_burst_score'] = sync_burst_score
        # 6. 各维度 HAB 存量注入
        for name, series in proxy_cores.items():
            final_proxy_signals[f"{name}_hab_score"] = series.mask(series < 0.6, 0).diff(1).clip(lower=0).rolling(21).sum().fillna(0).rolling(55).rank(pct=True).fillna(0.5)

        # if self._is_probe_enabled(pd.DataFrame(index=df_index)):
        #     self._probe_print(f"--- Fisher Proxy-SNR Consistency Fix Probe ---")
        #     self._probe_print(f"  > Fisher_Sync: {fisher_norm.iloc[-1]:.4f} | Burst: {sync_burst_score.iloc[-1]:.4f}")
        return final_proxy_signals

    def _calculate_enhanced_rs_proxy(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], config: Dict) -> Dict[str, pd.Series]:
        """
        【V6.1 · 向量化加速版】增强版相对强度代理信号
        修改说明：全量采用 float32，向量化合成相对强度矩阵，通过 Numpy 广播加速调制器计算。
        版本号：2026.02.11.14
        """
        pts = mtf_signals.get('mtf_price_trend', pd.Series(0.5, index=df_index)).astype(np.float32).clip(lower=0)
        rs_s = mtf_signals.get('SLOPE_5_price_trend', pd.Series(0, index=df_index)).astype(np.float32).values
        rs_a = mtf_signals.get('ACCEL_5_price_trend', pd.Series(0, index=df_index)).astype(np.float32).values
        rs_j = mtf_signals.get('JERK_5_price_trend', pd.Series(0, index=df_index)).astype(np.float32).values
        hab_win = int(get_param_value(config.get('rs_hab_window'), 21))
        hab_inc = (pts.mask(pts < 0.6, 0.0) * np.clip(rs_a, 0, None))
        hab_score = hab_inc.rolling(window=hab_win, min_periods=5).sum().fillna(0.0).rolling(hab_win).rank(pct=True).fillna(0.5).astype(np.float32)
        rsi_raw = normalized_signals.get('RSI_norm', pd.Series(0.5, index=df_index)).astype(np.float32).values
        rsi_strength = np.tanh((rsi_raw - 0.5) * 4.0) * 0.5 + 0.5
        kinematic_vec = np.clip(rs_s * 0.3 + rs_a * 0.4 + rs_j * 0.3, 0, 1)
        rs_proxy_raw = pts.values * 0.3 + rsi_strength * 0.2 + kinematic_vec * 0.3 + 0.1 # 结构补给
        enhanced_rs_proxy = (rs_proxy_raw * (0.95 + hab_score.values * 0.15))
        return {"raw_rs_proxy": pd.Series(rs_proxy_raw, index=df_index), "enhanced_rs_proxy": pd.Series(enhanced_rs_proxy, index=df_index).clip(0, 1).astype(np.float32), "rs_hab_score": hab_score}

    def _calculate_enhanced_capital_proxy(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], config: Dict) -> Dict[str, pd.Series]:
        """
        【V5.3 · 向量化加速版】增强版资本属性代理信号
        修改说明：完全矩阵化量纲自适应缩放路径，优化 Capital-HAB 的滚动排位性能，确保在 float32 下的高置信度感知。
        版本号：2026.02.11.41
        """
        mf_s = mtf_signals.get('SLOPE_5_mf_net_amount', pd.Series(0.0, index=df_index)).astype(np.float32)
        mf_a = mtf_signals.get('ACCEL_5_mf_net_amount', pd.Series(0.0, index=df_index)).astype(np.float32).values
        mf_j = mtf_signals.get('JERK_5_mf_net_amount', pd.Series(0.0, index=df_index)).astype(np.float32).values
        s_base = mf_s.abs().rolling(21, min_periods=5).mean().replace(0, 1e6).values
        c_slope = np.tanh(mf_s.values / s_base); c_accel = np.tanh(mf_a / s_base); c_jerk = np.tanh(mf_j / s_base)
        nmf_n = normalized_signals.get('net_mf_amount_norm', pd.Series(0.5, index=df_index)).astype(np.float32)
        h_inc = nmf_n.mask(nmf_n < 0.55, 0.0).values * np.clip(c_accel, 0, None)
        c_hab = pd.Series(h_inc, index=df_index).rolling(window=21, min_periods=5).sum().fillna(0.0).rolling(34).rank(pct=True).fillna(0.5).values
        k_sum = np.clip(c_slope * 0.4 + c_accel * 0.4 + c_jerk * 0.2, 0, 1)
        raw_p = k_sum * 0.4 + c_hab * 0.4 + 0.2
        return {"raw_capital_proxy": pd.Series(raw_p, index=df_index), "enhanced_capital_proxy": pd.Series(raw_p, index=df_index).clip(0, 1).astype(np.float32), "capital_hab_score": pd.Series(c_hab, index=df_index)}

    def _calculate_enhanced_sentiment_proxy(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], config: Dict) -> Dict[str, pd.Series]:
        """
        【V5.1 · 向量化加速版】增强版情绪属性代理信号
        修改说明：废弃通用归一化工具，改用向量化Rolling Rank计算Sentiment-HAB，提升情绪运动学合成的物理确定性。
        版本号：2026.02.11.25
        """
        ss = mtf_signals.get('SLOPE_5_market_sentiment_trend', pd.Series(0, index=df_index)).astype(np.float32)
        sa = mtf_signals.get('ACCEL_5_market_sentiment_trend', pd.Series(0, index=df_index)).astype(np.float32)
        sj = mtf_signals.get('JERK_5_market_sentiment_trend', pd.Series(0, index=df_index)).astype(np.float32)
        m_sent = normalized_signals.get('market_sentiment_norm', pd.Series(0.5, index=df_index)).astype(np.float32)
        i_breadth = normalized_signals.get('industry_breadth_norm', pd.Series(0.5, index=df_index)).astype(np.float32)
        hab_win = int(get_param_value(config.get('sentiment_hab_window'), 13))
        hab_inc = (m_sent * i_breadth * (1.0 + sa.clip(lower=0)))
        sent_hab_score = hab_inc.rolling(window=hab_win, min_periods=3).sum().fillna(0.0).rolling(21).rank(pct=True).fillna(0.5).astype(np.float32)
        leader_strength = normalized_signals.get('industry_leader_norm', pd.Series(0.5, index=df_index)).astype(np.float32)
        sent_consis = (m_sent * leader_strength).rolling(5, min_periods=1).mean().fillna(0.5).astype(np.float32)
        k_sum = np.clip(ss.values * 0.3 + sa.values * 0.5 + sj.values * 0.2, 0, 1)
        raw_proxy = (k_sum * 0.3 + sent_hab_score.values * 0.4 + sent_consis.values * 0.3)
        enhanced_proxy = raw_proxy * (1.0 + (sent_hab_score.values - 0.5) * 0.15)
        return {"raw_sentiment_proxy": pd.Series(raw_proxy, index=df_index), "enhanced_sentiment_proxy": pd.Series(enhanced_proxy, index=df_index).clip(0, 1).astype(np.float32), "sentiment_hab_score": sent_hab_score}

    def _calculate_sentiment_momentum(self, sentiment_series: pd.Series, df_index: pd.Index, memory_period: int = 13) -> pd.Series:
        """
        【V3.7 · 向量化动量版】计算情绪动量
        修改说明：强制 float32 类型，优化双阶动量（Velocity & Acceleration）的向量化路径，利用 Numpy 加速 Tanh 激活函数。
        版本号：2026.02.11.36
        """
        if sentiment_series.empty: return pd.Series(0.5, index=df_index).astype(np.float32)
        s_smooth = sentiment_series.astype(np.float32).ewm(span=5, adjust=False).mean().values
        vel = np.zeros_like(s_smooth); acc = np.zeros_like(s_smooth)
        vel[3:] = s_smooth[3:] - s_smooth[:-3]
        acc[3:] = vel[3:] - vel[:-3]
        v_score = np.tanh(vel * 5.0) * 0.5 + 0.5
        a_score = np.tanh(acc * 5.0) * 0.5 + 0.5
        return pd.Series(v_score * 0.6 + a_score * 0.4, index=df_index).clip(0, 1).astype(np.float32)

    def _calculate_enhanced_liquidity_proxy(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], config: Dict) -> Dict[str, pd.Series]:
        """
        【V6.1 · 向量化加速版】增强版流动性代理信号
        修改说明：利用对数压缩加速流动性潮汐识别，优化float32精度下的Rolling Min-Max逻辑，消除冗余计算。
        版本号：2026.02.11.26
        """
        vol_n = normalized_signals.get('volume_ratio_norm', pd.Series(0.5, index=df_index)).astype(np.float32)
        turn_n = normalized_signals.get('turnover_rate_norm', pd.Series(0.5, index=df_index)).astype(np.float32)
        liq_tide = (vol_n * 0.6 + turn_n * 0.4).values
        tide_slope = pd.Series(liq_tide, index=df_index).diff(3).rolling(5).mean().fillna(0.0).values
        tide_accel = np.diff(tide_slope, prepend=0.0)
        p_slope_abs = np.abs(mtf_signals.get('SLOPE_5_price_trend', pd.Series(0, index=df_index)).astype(np.float32).values)
        liq_hab_inc = np.clip(tide_accel, 0, None) * (1.0 - np.clip(p_slope_abs, 0, 1))
        log_inc = np.log1p(pd.Series(liq_hab_inc, index=df_index).rolling(21, min_periods=1).sum().fillna(0.0))
        l_min = log_inc.rolling(55, min_periods=5).min(); l_max = log_inc.rolling(55, min_periods=5).max()
        liq_hab_score = ((log_inc - l_min) / (l_max - l_min + 1e-9)).fillna(0.5).astype(np.float32)
        liq_proxy = (liq_tide * 0.4 + np.clip(tide_accel, 0, 1) * 0.3 + liq_hab_score.values * 0.3)
        return {"enhanced_liquidity_proxy": pd.Series(liq_proxy, index=df_index).clip(0, 1).astype(np.float32), "liquidity_hab_score": liq_hab_score}

    def _calculate_enhanced_volatility_proxy(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], config: Dict) -> Dict[str, pd.Series]:
        """
        【V6.1 · 向量化加速版】增强版波动性代理信号
        修改说明：采用float32全向量化算子实现波动聚集指数，利用Tanh矩阵运算加速Jerk分量提取。
        版本号：2026.02.11.27
        """
        pct_n = normalized_signals.get('pct_change_norm', pd.Series(0, index=df_index)).astype(np.float32)
        h_vol = pct_n.rolling(21, min_periods=5).std().fillna(0.02)
        v_slope = h_vol.diff(3).rolling(5).mean().fillna(0.0).values
        v_jerk = np.diff(v_slope, prepend=0.0)
        avg_v = h_vol.rolling(60, min_periods=10).mean().replace(0, 0.02).values
        vol_hab_raw = pd.Series(avg_v / (h_vol.values + 1e-9), index=df_index).rolling(21, min_periods=1).sum().fillna(0.0)
        vol_hab_score = (np.tanh(vol_hab_raw.values / 10.0) * 0.5 + 0.5)
        h_rank = h_vol.rolling(60, min_periods=5).rank(pct=True).fillna(0.5).values
        vol_proxy = (h_rank * 0.3 + np.abs(np.tanh(v_jerk * 10.0)) * 0.3 + (1.0 - vol_hab_score) * 0.4)
        return {"enhanced_volatility_proxy": pd.Series(vol_proxy, index=df_index).clip(0, 1).astype(np.float32), "volatility_hab_score": pd.Series(vol_hab_score, index=df_index).astype(np.float32)}

    def _calculate_enhanced_risk_preference_proxy(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], config: Dict) -> Dict[str, pd.Series]:
        """
        【V6.1 · 向量化加速版】增强版风险偏好代理信号
        修改说明：利用Numpy广播加速Sigmoid映射逻辑，强制float32数据流，提升非线性风险资产评估效率。
        版本号：2026.02.11.28
        """
        l_str = normalized_signals.get('industry_leader_norm', pd.Series(0.5, index=df_index)).astype(np.float32).values
        m_sent = normalized_signals.get('market_sentiment_norm', pd.Series(0.5, index=df_index)).astype(np.float32).values
        asset_perf = np.clip(l_str * 0.6 + m_sent * 0.4, 0, 1)
        risk_perf = 1.0 / (1.0 + np.exp(-8.0 * (asset_perf - 0.5)))
        r_slope = pd.Series(risk_perf, index=df_index).diff(3).rolling(5).mean().fillna(0.0).values
        r_accel = np.diff(r_slope, prepend=0.0)
        hab_raw = pd.Series(risk_perf, index=df_index).rolling(window=21, min_periods=1).sum().fillna(0.0)
        risk_hab_score = hab_raw.rolling(55, min_periods=5).rank(pct=True).fillna(0.5).astype(np.float32)
        risk_proxy = (risk_perf * 0.4 + np.tanh(np.clip(r_accel * 5.0, 0, None)) * 0.3 + risk_hab_score.values * 0.3)
        return {"enhanced_risk_preference_proxy": pd.Series(risk_proxy, index=df_index).clip(0, 1).astype(np.float32), "risk_hab_score": risk_hab_score}

    def _dynamic_weighted_synthesis(self, rs_proxy: Dict, capital_proxy: Dict, sentiment_proxy: Dict, liquidity_proxy: Dict, volatility_proxy: Dict, risk_preference_proxy: Dict, signal_quality: pd.Series, config: Dict) -> Dict[str, pd.Series]:
        """
        【V5.1 · 向量化Softmax版】动态权重合成算法
        修改说明：利用Numpy矩阵运算替代循环Softmax计算，快速处理六维度代理信号的加权合成。
        版本号：2026.02.11.05
        """
        df_index = signal_quality.index
        # 1. 准备信号矩阵 (N, 6)
        # 显式提取Series并填充默认值
        default_series = pd.Series(0.5, index=df_index)
        signal_keys = ["rs", "capital", "sentiment", "liquidity", "volatility", "risk"]
        proxies = [
            rs_proxy.get("enhanced_rs_proxy", default_series),
            capital_proxy.get("enhanced_capital_proxy", default_series),
            sentiment_proxy.get("enhanced_sentiment_proxy", default_series),
            liquidity_proxy.get("enhanced_liquidity_proxy", default_series),
            volatility_proxy.get("enhanced_volatility_proxy", default_series),
            risk_preference_proxy.get("enhanced_risk_preference_proxy", default_series)
        ]
        # 堆叠为 (N, 6) 数组
        signal_matrix = np.column_stack([p.values.astype(np.float32) for p in proxies])
        # 2. 准备基础分数与温度
        base_scores = np.array([1.2, 1.0, 0.6, 0.4, 0.4, 0.4], dtype=np.float32) # Shape (6,)
        temperature = (1.5 - (signal_quality.values * 1.3)).astype(np.float32).reshape(-1, 1) # Shape (N, 1)
        # 3. 向量化 Softmax
        # exp(Base / Temp) -> 利用广播 (1, 6) / (N, 1) -> (N, 6)
        # 注意: base_scores 是 1D, 会自动广播为 (1, 6)
        exp_vals = np.exp(base_scores / (temperature + 1e-9))
        weights = exp_vals / (np.sum(exp_vals, axis=1, keepdims=True) + 1e-9) # (N, 6)
        # 4. 加权合成 (N, 6) * (N, 6) -> sum -> (N,)
        final_vals = np.sum(signal_matrix * weights, axis=1)
        # 5. 结果封装
        final_series = pd.Series(np.clip(final_vals, 0, 1), index=df_index)
        result = {"comprehensive_proxy": final_series}
        for i, key in enumerate(signal_keys):
            result[f"{key}_weight"] = pd.Series(weights[:, i], index=df_index)
        # if self._is_probe_enabled(pd.DataFrame(index=df_index)):
        #     self._probe_print(f"[WEIGHT_PROBE] Temp: {temperature[-1, 0]:.2f}, RS_Weight: {weights[-1, 0]:.4f}")
        return result

    def _assess_signal_quality(self, rs_proxy: Dict, capital_proxy: Dict, sentiment_proxy: Dict, liquidity_proxy: Dict, volatility_proxy: Dict, risk_preference_proxy: Dict) -> pd.Series:
        """
        【V4.1 · 矩阵质量合成版】代理信号质量评估方法
        修改说明：移除时间步循环，利用矩阵加权求和替代逐行计算。预处理所有子信号质量为 float32，大幅提升综合质量评估速度。
        版本号：2026.02.11.37
        """
        signal_assessor = SignalQualityAssessor()
        keys = ["rs", "capital", "sentiment", "liquidity", "volatility", "risk_preference"]
        all_sigs = {
            "rs": rs_proxy.get("enhanced_rs_proxy", pd.Series(0.5, index=rs_proxy.get("raw_rs_proxy", pd.Series()).index)),
            "capital": capital_proxy.get("enhanced_capital_proxy", pd.Series(0.5, index=capital_proxy.get("raw_capital_proxy", pd.Series()).index)),
            "sentiment": sentiment_proxy.get("enhanced_sentiment_proxy", pd.Series(0.5, index=sentiment_proxy.get("raw_sentiment_proxy", pd.Series()).index)),
            "liquidity": liquidity_proxy.get("enhanced_liquidity_proxy", pd.Series(0.5, index=liquidity_proxy.get("raw_liquidity_proxy", pd.Series()).index)),
            "volatility": volatility_proxy.get("enhanced_volatility_proxy", pd.Series(0.5, index=volatility_proxy.get("raw_volatility_proxy", pd.Series()).index)),
            "risk_preference": risk_preference_proxy.get("enhanced_risk_preference_proxy", pd.Series(0.5, index=risk_preference_proxy.get("raw_risk_preference_proxy", pd.Series()).index))
        }
        df_index = all_sigs["rs"].index
        ind_qs = {k: signal_assessor._calculate_individual_signal_quality(v, k).astype(np.float32) for k, v in all_sigs.items()}
        consis_q = signal_assessor._calculate_signal_consistency_quality(all_sigs).astype(np.float32)
        pred_q = signal_assessor._calculate_predictive_quality(all_sigs).astype(np.float32)
        dyn_ws = signal_assessor._calculate_dynamic_quality_weights(ind_qs, consis_q, pred_q)
        q_mat = np.column_stack([ind_qs[k].values for k in keys] + [consis_q.values, pred_q.values])
        w_mat = np.column_stack([dyn_ws[k].values if isinstance(dyn_ws[k], pd.Series) else np.full(len(df_index), dyn_ws[k]) for k in keys] + [np.full(len(df_index), 0.2), np.full(len(df_index), 0.3)])
        weighted_q = np.sum(q_mat * w_mat, axis=1) / (np.sum(w_mat, axis=1) + 1e-9)
        return pd.Series(weighted_q, index=df_index).rolling(window=5, min_periods=3).mean().fillna(0.7).clip(0, 1).astype(np.float32)

    def _detect_comprehensive_market_state(self, rs_signal: pd.Series, capital_signal: pd.Series, sentiment_signal: pd.Series, liquidity_signal: pd.Series, volatility_signal: pd.Series, risk_preference_signal: pd.Series) -> pd.Series:
        """
        【V4.8 · 向量化条件引擎版】综合市场状态检测方法
        修改说明：彻底移除 Python 循环，利用 Numpy 的 np.select 引擎批量匹配市场状态条件，利用滚动 OLS 斜率预计算消除冗余趋势分析。
        版本号：2026.02.11.39
        """
        df_index = rs_signal.index
        rs_v = rs_signal.values.astype(np.float32); cap_v = capital_signal.values.astype(np.float32); sent_v = sentiment_signal.values.astype(np.float32)
        liq_v = liquidity_signal.values.astype(np.float32); vol_v = volatility_signal.values.astype(np.float32); risk_v = risk_preference_signal.values.astype(np.float32)
        def _rolling_slope_vec(arr, window=20):
            n = window; x = np.arange(n, dtype=np.float32)
            sum_x = n * (n - 1) / 2.0; sum_x2 = n * (n - 1) * (2 * n - 1) / 6.0; denom = n * sum_x2 - sum_x**2
            y_sum = pd.Series(arr).rolling(window).sum().values
            xy_sum = pd.Series(arr * np.arange(len(arr))).rolling(window).sum().values
            # 修正 xy_sum 的基准偏移
            t = np.arange(len(arr), dtype=np.float32); xy_sum = xy_sum - (t - n + 1) * y_sum + (t - n + 1) * n * (n - 1) / 2.0 # 简化的滑动窗口修正
            slope = (n * xy_sum - sum_x * y_sum) / denom
            return np.tanh(slope / 0.02)
        rs_t = _rolling_slope_vec(rs_v); cap_t = _rolling_slope_vec(cap_v); sent_t = _rolling_slope_vec(sent_v)
        thr = 0.15; rev_thr = 0.20
        conds = [
            (vol_v > 0.7) & (risk_v < 0.3) & (liq_v < 0.3) & ((np.abs(cap_t) > 0.1) | (np.abs(sent_t) > 0.1)),
            (rs_v > 0.65) & (cap_v > 0.6) & (sent_v > 0.6) & (rs_t > thr) & (cap_t > thr),
            (rs_v < 0.35) & (cap_v < 0.4) & (sent_v < 0.4) & (rs_t < -thr) & (cap_t < -thr),
            (rs_t > rev_thr) & (cap_t > rev_thr) & (rs_v < 0.65) & (cap_v < 0.65),
            (rs_t < -rev_thr) & (sent_t < -rev_thr) & (rs_v > 0.4) & (cap_v > 0.4)
        ]
        choices = ["crisis", "trending_up", "trending_down", "reversing_up", "reversing_down"]
        states = np.select(conds, choices, default="consolidating")
        return pd.Series(states, index=df_index)

    def _calculate_trend_direction(self, series: pd.Series) -> float:
        """
        【V5.1 · OLS 向量优化版】计算序列趋势方向
        修改说明：采用解析解 OLS 斜率公式替代 lstsq，针对单点调用进行极致性能优化，支持 float32 运算。
        版本号：2026.02.11.38
        """
        n = len(series)
        if n < 5: return 0.0
        y = series.values.astype(np.float32)
        x = np.arange(n, dtype=np.float32)
        sum_x = n * (n - 1) / 2.0
        sum_x2 = n * (n - 1) * (2 * n - 1) / 6.0
        denom = n * sum_x2 - sum_x**2
        if abs(denom) < 1e-9: return 0.0
        slope = (n * np.sum(x * y) - sum_x * np.sum(y)) / denom
        return float(np.tanh(slope / 0.02))

    def _weighted_geometric_mean(self, components: Dict[str, pd.Series], weights: Dict[str, float], df_index: pd.Index) -> pd.Series:
        """
        【V5.1 · 矩阵几何平均版】加权几何平均
        修改说明：移除字典迭代，利用矩阵乘法一次性完成多因子对数合成，统一使用 float32 提升吞吐量。
        版本号：2026.02.11.45
        """
        v_keys = [k for k in weights.keys() if k in components]
        if not v_keys: return pd.Series(0.5, index=df_index).astype(np.float32)
        data_mat = np.column_stack([components[k].values.astype(np.float32) for k in v_keys])
        w_vec = np.array([weights[k] for k in v_keys], dtype=np.float32)
        log_sum = np.dot(np.log(np.clip(data_mat, 1e-7, 1.0)), w_vec)
        res = np.exp(log_sum / (np.sum(w_vec) + 1e-9))
        return pd.Series(res, index=df_index).clip(0, 1).astype(np.float32)

    def _nonlinear_synthesis(self, components: Dict[str, pd.Series], weights: Dict[str, float], df_index: pd.Index) -> pd.Series:
        """
        【V5.1 · 矩阵 Sigmoid 加速版】非线性加权合成
        修改说明：矩阵化非线性映射引擎，利用广播机制加速逻辑回归合成，强制 float32 处理。
        版本号：2026.02.11.46
        """
        v_keys = [k for k in weights.keys() if k in components]
        if not v_keys: return pd.Series(0.5, index=df_index).astype(np.float32)
        data_mat = np.column_stack([components[k].values.astype(np.float32) for k in v_keys])
        w_vec = np.array([weights[k] for k in v_keys], dtype=np.float32)
        sig_mat = 1.0 / (1.0 + np.exp(-10.0 * (data_mat - 0.5)))
        weighted_sum = np.dot(sig_mat, w_vec)
        return pd.Series(weighted_sum / (np.sum(w_vec) + 1e-9), index=df_index).clip(0, 1).astype(np.float32)

    def _calculate_dynamic_weights(self, df_index: pd.Index, normalized_signals: Dict[str, pd.Series], proxy_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], factors: Dict[str, pd.Series], phase: pd.Series) -> Dict[str, pd.Series]:
        """
        【V6.4 · 向量化加速版】动态权重计算系统
        修改说明：移除了时间步循环，利用DataFrame的EWM和Numpy广播机制批量计算Softmax权重，显著提升性能。
        版本号：2026.02.11.04
        """
        # 1. 物理相位基准权重 (已向量化)
        base_weights_raw = self._calculate_base_weights(df_index, phase, factors)
        # 转换为 DataFrame 以便批量处理 (N, 4)
        df_weights = pd.DataFrame(base_weights_raw)
        # 2. 运动学增益 (向量化)
        rs_acc = pd.Series(mtf_signals.get('ACCEL_5_price_trend', 0.0), index=df_index).abs()
        # 对 aggressiveness 列应用增益
        df_weights['aggressiveness'] *= (1.0 + rs_acc * 0.6)
        # 批量 EWM 平滑
        df_weights = df_weights.ewm(span=13).mean()
        # 3. 提取因子 (转化为 Numpy 数组)
        quality = proxy_signals.get('combined_signal_quality', factors.get('state_hab_score', pd.Series(0.7, index=df_index))).values
        burst_score = proxy_signals.get('sync_burst_score', pd.Series(0.0, index=df_index)).values
        # 4. 向量化 Softmax 计算
        # 基础温度 (N,)
        base_temp = 2.0 - (quality * 1.5)
        # 最终温度 (N,)
        final_temp = base_temp / (1.0 + burst_score * 0.8 + 1e-9)
        temp_clamped = np.clip(final_temp, 0.1, 2.0).reshape(-1, 1) # 调整形状为 (N, 1) 用于广播
        # 计算指数 (N, 4)
        raw_vals = df_weights.values
        exp_vals = np.exp(raw_vals / temp_clamped)
        # 归一化 (N, 4)
        row_sums = np.sum(exp_vals, axis=1, keepdims=True) + 1e-9
        norm_vals = exp_vals / row_sums
        # 5. 封装结果
        final_weights = {}
        cols = df_weights.columns
        for i, col in enumerate(cols):
            final_weights[col] = pd.Series(norm_vals[:, i], index=df_index)
        # if self._is_probe_enabled(pd.DataFrame(index=df_index)):
        #     self._probe_print(f"[WEIGHT_BURST_PROBE] Burst: {burst_score[-1]:.4f} | Final_Temp: {final_temp[-1]:.4f}")
        return final_weights

    def _calculate_market_state_factors(self, df_index: pd.Index, normalized_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V6.0 · Numba 状态迁移增强版】七维度市场状态因子计算
        修改说明：引入 _numba_state_transition_audit 算子。在矩阵层面同步计算环境惯性存量与非线性顶背离压力，提升识别精度与执行效率。
        版本号：2026.02.11.74
        """
        factors = {}
        # 1. 基础物理分量提取 (float32 硬化)
        p_trend = pd.Series(mtf_signals.get('mtf_price_trend', 0.5), index=df_index).values.astype(np.float32)
        u_strength = pd.Series(normalized_signals.get('uptrend_strength_norm', 0.5), index=df_index).values.astype(np.float32)
        c_net_mf = pd.Series(normalized_signals.get('net_mf_amount_norm', 0.5), index=df_index).values.astype(np.float32)
        c_flow_acc = pd.Series(normalized_signals.get('flow_acceleration_norm', 0.5), index=df_index).values.astype(np.float32)
        # 2. 状态分层合成 (非线性几何平均)
        t_state = np.sqrt(np.clip(p_trend * u_strength, 1e-6, 1.0))
        c_state = (c_net_mf * 0.6 + c_flow_acc * 0.4)
        t_acc = pd.Series(t_state, index=df_index).diff(1).diff(1).fillna(0.0).values.astype(np.float32)
        # 3. 调用 Numba 算子执行迁移审计
        hab_raw, div_raw = self._numba_state_transition_audit(t_state, c_state, t_acc, 21)
        # 4. 结果封装与归一化
        factors['trend_state'] = pd.Series(t_state, index=df_index)
        factors['capital_state'] = pd.Series(c_state, index=df_index)
        factors['trend_state_accel'] = pd.Series(t_acc, index=df_index)
        # 存量水位归一化 (34日长窗口 Rolling Rank)
        factors['state_hab_score'] = pd.Series(hab_raw, index=df_index).rolling(window=34, min_periods=5).rank(pct=True).fillna(0.5)
        # 背离压力归一化 (Tanh 映射)
        factors['divergence_pressure'] = pd.Series(np.tanh(div_raw), index=df_index).astype(np.float32)
        factors['volatility_state'] = (1.0 - pd.Series(normalized_signals.get('ATR_norm', 0.5), index=df_index)).clip(0, 1).astype(np.float32)
        factors['sentiment_state'] = pd.Series(normalized_signals.get('market_sentiment_norm', 0.5), index=df_index).clip(0, 1).astype(np.float32)
        # if self._is_probe_enabled(pd.DataFrame(index=df_index)):
        #     self._probe_print(f"--- Market State Numba Probe V6.0 ---")
        #     self._probe_print(f"  > State HAB: {factors['state_hab_score'].iloc[-1]:.4f} | Div_Pressure: {factors['divergence_pressure'].iloc[-1]:.4f}")
        return factors

    def _identify_market_phase(self, df_index: pd.Index, market_state_factors: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series]) -> pd.Series:
        """
        【V5.5 · Numba 众数加速版】市场阶段识别算法
        修改说明：引入 Numba 编译的滚动众数算子替代 Pandas 矩阵偏移法，显著降低内存占用并提升在高频回测场景下的执行速度。
        版本号：2026.02.11.51
        """
        p_hab = pd.Series(market_state_factors.get('state_hab_score', 0.5), index=df_index).fillna(0.5).values.astype(np.float32)
        ch_hab = pd.Series(normalized_signals.get('chip_stability_norm', 0.5), index=df_index).rolling(34, min_periods=1).mean().fillna(0.5).values.astype(np.float32)
        c_hab = pd.Series(normalized_signals.get('net_mf_amount_norm', 0.5), index=df_index).rolling(21, min_periods=1).rank(pct=True).fillna(0.5).values.astype(np.float32)
        p_acc = pd.Series(market_state_factors.get('trend_state_accel', 0.0), index=df_index).fillna(0.0).values.astype(np.float32)
        c_accel = pd.Series(normalized_signals.get('flow_acceleration_norm', 0.5), index=df_index).fillna(0.5).values.astype(np.float32)
        prob_acc = c_hab * (1.0 - p_hab)
        prob_exp = p_hab * c_hab * (np.clip(p_acc, 0, None) + 0.5)
        early_mask = (p_hab >= 0.8) & (ch_hab >= 0.7)
        prob_dist_early = np.where(early_mask, (0.6 - c_accel).clip(0, 1), 0.0)
        prob_dist_late = p_hab * (1.0 - ch_hab) * (1.0 - c_hab)
        prob_panic = np.abs(np.diff(p_acc, prepend=0.0).clip(None, 0)) * p_hab
        phase_matrix = np.column_stack((prob_acc, prob_exp, prob_dist_early, prob_dist_late, prob_panic, np.full(len(df_index), 0.2, dtype=np.float32)))
        raw_int = np.argmax(phase_matrix, axis=1).astype(np.float32)
        smoothed_int = self._numba_rolling_mode(raw_int, 5)
        inv_map = {0.0: "蓄势", 1.0: "主升", 2.0: "派发初期", 3.0: "派发末端", 4.0: "反转风险", 5.0: "横盘"}
        smoothed_phases = pd.Series(smoothed_int, index=df_index).map(inv_map).fillna("横盘")
        return smoothed_phases

    def _calculate_base_weights(self, df_index: pd.Index, market_phase: pd.Series, market_state_factors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V5.2 · 向量化加速版】基础权重计算
        修改说明：移除Python循环，利用Numpy高级索引和掩码实现全量数据的向量化映射与权重调整。
        版本号：2026.02.11.03
        """
        # 提取数组 (N,)
        p_acc = market_state_factors.get('trend_state_accel', pd.Series(0.0, index=df_index)).values.astype(np.float32)
        s_hab = market_state_factors.get('state_hab_score', pd.Series(0.5, index=df_index)).values.astype(np.float32)
        phases = market_phase.values
        # 1. 建立相位的整数映射与权重查找表 (Lookup Table)
        # 映射: 启动=0, 主升=1, 派发初期=2, 派发末端=3, 反转风险=4, 蓄势=5, 横盘=6
        phase_to_idx = {
            '启动': 0, '主升': 1, '派发初期': 2, '派发末端': 3, '反转风险': 4, '蓄势': 5, '横盘': 6
        }
        # 默认使用横盘(6)填充未知相位
        phase_indices = market_phase.map(phase_to_idx).fillna(6).values.astype(int)
        # 权重配置矩阵: Shape (7, 4) -> [Agg, Ctrl, Obs, Risk]
        config_matrix = np.array([
            [0.35, 0.30, 0.25, 0.10], # 0: 启动
            [0.45, 0.20, 0.25, 0.10], # 1: 主升
            [0.55, 0.10, 0.15, 0.20], # 2: 派发初期
            [0.05, 0.10, 0.10, 0.75], # 3: 派发末端
            [0.10, 0.25, 0.15, 0.50], # 4: 反转风险
            [0.25, 0.35, 0.25, 0.15], # 5: 蓄势
            [0.25, 0.35, 0.20, 0.20]  # 6: 横盘
        ], dtype=np.float32)
        # 2. 向量化查表获取基础权重 (N, 4)
        base_w = config_matrix[phase_indices] # Advanced Indexing
        # 3. 向量化动态调整
        # acc_val > 0
        mask_pos_acc = p_acc > 0
        # Agg (col 0): if acc>0: *= (1 + acc*0.5)
        base_w[:, 0] = np.where(mask_pos_acc, base_w[:, 0] * (1 + p_acc * 0.5), base_w[:, 0])
        # Risk (col 3): if acc<=0: *= (1 + abs(acc)*0.8)
        base_w[:, 3] = np.where(~mask_pos_acc, base_w[:, 3] * (1 + np.abs(p_acc) * 0.8), base_w[:, 3])
        # HAB 调整
        # Agg *= (0.9 + hab*0.2)
        base_w[:, 0] *= (0.9 + s_hab * 0.2)
        # Ctrl *= (0.9 + hab*0.2)
        base_w[:, 1] *= (0.9 + s_hab * 0.2)
        # Risk *= (1.1 - hab*0.2)
        base_w[:, 3] *= (1.1 - s_hab * 0.2)
        # 4. 行归一化
        row_sums = base_w.sum(axis=1, keepdims=True) + 1e-9
        norm_w = base_w / row_sums
        # 5. 裁剪并封装回 Series
        # 索引对应: 0:agg, 1:ctrl, 2:obs, 3:risk
        return {
            'aggressiveness': pd.Series(np.clip(norm_w[:, 0], 0.05, 0.7), index=df_index),
            'control': pd.Series(np.clip(norm_w[:, 1], 0.05, 0.6), index=df_index),
            'obstacle_clearance': pd.Series(np.clip(norm_w[:, 2], 0.05, 0.6), index=df_index),
            'risk': pd.Series(np.clip(norm_w[:, 3], 0.05, 0.8), index=df_index)
        }

    def _calculate_aggressiveness_score(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], dynamic_weights: Dict[str, pd.Series]) -> pd.Series:
        """
        【V8.1 · 矩阵动能加速版】计算攻击性分值
        修改说明：利用 Numpy 幂运算加速几何平均合成，优化封板一致性 HAB 的滚动排位效率，全链路强制 float32。
        版本号：2026.02.11.40
        """
        p_t = mtf_signals.get('mtf_price_trend', pd.Series(0.0, index=df_index)).astype(np.float32).clip(lower=0).values
        v_t = mtf_signals.get('mtf_volume_trend', pd.Series(0.0, index=df_index)).astype(np.float32).clip(lower=0).values
        s_a = mtf_signals.get('ACCEL_5_market_sentiment_trend', pd.Series(0.0, index=df_index)).astype(np.float32).clip(lower=0).values
        close = raw_signals.get('close', 0.0).values; up_l = raw_signals.get('up_limit', 0.0).values
        is_lu = (close >= up_l) & (up_l > 0)
        lu_inc = is_lu.astype(np.float32) * (1.0 + p_t)
        lu_hab = pd.Series(lu_inc, index=df_index).rolling(window=13).sum().rolling(34).rank(pct=True).fillna(0.5).values
        agg_raw = (p_t ** 0.3 * v_t ** 0.2 * (1.0 + s_a) ** 0.3 * (1.0 + lu_hab) ** 0.2)
        agg_rank = pd.Series(agg_raw, index=df_index).rolling(window=90, min_periods=20).rank(pct=True).fillna(0.5).values
        p_j = mtf_signals.get('JERK_5_price_trend', pd.Series(0.0, index=df_index)).astype(np.float32).abs()
        j_n = np.tanh(p_j.values / (p_j.rolling(60).mean().replace(0, 1e-9).values))
        return pd.Series(agg_rank * (1.0 + j_n * 0.2), index=df_index).clip(0, 1).astype(np.float32)

    def _detect_bear_trap(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], obstacle_clearance_score: pd.Series) -> pd.Series:
        """
        【V7.1 · 向量化加速版】诱空陷阱检测
        修改说明：完全矩阵化trap_impulse计算路径，强制float32并利用Numpy的clip/abs算子实现物理反转识别。
        版本号：2026.02.11.30
        """
        p_acc = mtf_signals.get('ACCEL_5_price_trend', pd.Series(0, index=df_index)).astype(np.float32).values
        p_jerk = mtf_signals.get('JERK_5_price_trend', pd.Series(0, index=df_index)).astype(np.float32).values
        par_step = np.clip(np.diff(obstacle_clearance_score.values, prepend=obstacle_clearance_score.iloc[0]), 0, None)
        trap_impulse = np.abs(np.clip(p_acc, None, 0)) * np.clip(p_jerk, 0, None) * (1.0 + par_step * 2.0)
        trap_intensity = np.tanh(trap_impulse / 0.0015)
        hab_backing = obstacle_clearance_score.rolling(window=13, min_periods=1).mean().fillna(0.0).values
        final_trap = trap_intensity * hab_backing
        return pd.Series(final_trap, index=df_index).fillna(0.0).astype(np.float32)

    def _detect_bull_trap(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], dist_hab: pd.Series) -> pd.Series:
        """
        【V1.2 · 向量化加速版】诱多陷阱检测
        修改说明：采用全量float32，向量化捕获上涨惯性中的受力逆转(Negative Jerk)，增强NaN防御力。
        版本号：2026.02.11.31
        """
        p_acc = mtf_signals.get('ACCEL_5_price_trend', pd.Series(0, index=df_index)).astype(np.float32).fillna(0.0).values
        p_jerk = mtf_signals.get('JERK_5_price_trend', pd.Series(0, index=df_index)).astype(np.float32).fillna(0.0).values
        dist_s = normalized_signals.get('distribution_score_norm', pd.Series(0, index=df_index)).astype(np.float32).fillna(0.0).values
        dist_step = np.clip(np.diff(dist_s, prepend=dist_s[0]), 0, None)
        bull_impulse = np.clip(p_acc, 0, None) * np.abs(np.clip(p_jerk, None, 0)) * (1.0 + dist_step * 2.0)
        trap_intensity = np.tanh(bull_impulse / 0.0020)
        final_trap = trap_intensity * dist_hab.fillna(0.0).values
        return pd.Series(final_trap, index=df_index).fillna(0.0).astype(np.float32)

    def _calculate_volume_price_divergence(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series]) -> pd.Series:
        """
        【V6.1 · 向量化加速版】缩量背离侦测
        修改说明：矩阵化价格与成交量排位差值计算，利用mask机制一次性完成虚假拉升过滤，显著降低逻辑开销。
        版本号：2026.02.11.29
        """
        p_slope = mtf_signals.get('SLOPE_5_price_trend', pd.Series(0, index=df_index)).astype(np.float32)
        v_slope = mtf_signals.get('SLOPE_5_volume_trend', pd.Series(0, index=df_index)).astype(np.float32)
        p_rank = p_slope.rolling(13, min_periods=1).rank(pct=True).fillna(0.5).values
        v_rank = v_slope.rolling(13, min_periods=1).rank(pct=True).fillna(0.5).values
        gap = np.clip(p_rank - v_rank, 0, None)
        div_score = np.where(p_slope.values <= 0, 0.0, gap ** 2)
        return pd.Series(div_score, index=df_index).clip(0, 1).astype(np.float32)

    def _calculate_acceleration_resonance(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series]) -> pd.Series:
        """
        【V6.1 · 向量化共振版】多周期加速度共振检测
        修改说明：利用Numpy布尔计数映射替代Pandas map循环，并优化稳健Z-Score的矩阵化实现。
        版本号：2026.02.11.10
        """
        a5 = mtf_signals.get('ACCEL_5_price_trend', pd.Series(0, index=df_index)).values.astype(np.float32)
        a13 = mtf_signals.get('ACCEL_13_price_trend', pd.Series(0, index=df_index)).values.astype(np.float32)
        a21 = mtf_signals.get('ACCEL_21_price_trend', pd.Series(0, index=df_index)).values.astype(np.float32)
        res_count = (a5 > 0).astype(int) + (a13 > 0).astype(int) + (a21 > 0).astype(int)
        res_map = np.array([0.0, 0.1, 0.5, 1.0], dtype=np.float32)
        res_base = res_map[res_count]
        avg_a = (np.clip(a5, 0, None) + np.clip(a13, 0, None) + np.clip(a21, 0, None)) / 3.0
        avg_a_s = pd.Series(avg_a, index=df_index)
        med = avg_a_s.rolling(60).median()
        mad = (avg_a_s - med).abs().rolling(60).median()
        a_norm = np.tanh((avg_a_s - med) / (mad * 1.4826 + 1e-9)).clip(0, 1).values
        return pd.Series(res_base * (1.0 + a_norm * 0.3), index=df_index).clip(0, 1).astype(np.float32)

    def _calculate_control_score(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], historical_context: Dict[str, Any]) -> pd.Series:
        """
        【V5.3 · 向量化加速版】计算控制力分数
        修改说明：矩阵化熵变动力学算子，利用 float32 统一计算量纲，消除 Series 索引对齐开销。
        版本号：2026.02.11.15
        """
        entropy = raw_signals.get('chip_entropy', pd.Series(0.5, index=df_index)).astype(np.float32)
        chip_stab = raw_signals.get('chip_stability', pd.Series(0.5, index=df_index)).astype(np.float32)
        vol_ratio = raw_signals.get('volume_ratio', pd.Series(1.0, index=df_index)).astype(np.float32)
        entropy_slope = entropy.diff(3).rolling(5).mean().fillna(0.0).values
        concentration_impulse = np.tanh(-entropy_slope / 0.01).clip(min=0)
        migration_efficiency = np.clip(concentration_impulse / (vol_ratio.values + 0.1), 0, 2)
        migration_norm = np.tanh(migration_efficiency)
        hab_inc = migration_norm * chip_stab.values
        hab_score = pd.Series(hab_inc, index=df_index).rolling(window=21, min_periods=5).sum().fillna(0.0).rolling(55).rank(pct=True).fillna(0.5).astype(np.float32)
        p_hab = normalized_signals.get('absolute_change_strength_norm', pd.Series(0.5, index=df_index)).astype(np.float32).rolling(13).mean().values
        frenzy_risk = np.clip(np.where(p_hab < 0.7, 0.0, p_hab) * np.tanh(np.clip(entropy_slope, 0, None) / 0.01), 0, 1)
        final_control = (migration_norm * 0.4 + hab_score.values * 0.6) * (1.0 - frenzy_risk * 0.5)
        return pd.Series(final_control, index=df_index).clip(0, 1).astype(np.float32)

    def _calculate_obstacle_clearance_score(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], historical_context: Dict[str, Any]) -> pd.Series:
        """
        【V7.1 · 向量化加速版】计算障碍清除分数
        修改说明：物理 PAR(抛压吸收比)全量矩阵化，基于 float32 执行 MAD 稳健缩放。
        版本号：2026.02.11.16
        """
        p_acc = mtf_signals.get('ACCEL_5_price_trend', pd.Series(0, index=df_index)).astype(np.float32).fillna(0.0).values
        p_jerk = mtf_signals.get('JERK_5_price_trend', pd.Series(0, index=df_index)).astype(np.float32).fillna(0.0).values
        v_ratio = raw_signals.get('volume_ratio', pd.Series(1.0, index=df_index)).astype(np.float32).replace(0, 1e-9).values
        chip_stab = raw_signals.get('chip_stability', pd.Series(0.5, index=df_index)).astype(np.float32).values
        par_raw = np.clip(p_jerk, 0, None) / v_ratio
        par_s = pd.Series(par_raw, index=df_index)
        par_med = par_s.rolling(60, min_periods=5).median().values
        par_mad = (par_s - pd.Series(par_med, index=df_index)).abs().rolling(60, min_periods=5).median().values
        par_norm = np.tanh(par_raw / (par_mad * 4.4478 + 1e-9))
        p_slope_abs = np.abs(mtf_signals.get('SLOPE_5_price_trend', pd.Series(0, index=df_index)).astype(np.float32).values)
        support_rebound = par_norm * (1.0 - np.tanh(p_slope_abs / 0.02))
        hab_inc = support_rebound * chip_stab * (1.0 + np.clip(p_acc, 0, None))
        hab_score = pd.Series(hab_inc, index=df_index).rolling(window=34, min_periods=5).sum().fillna(0.0).rolling(89, min_periods=20).rank(pct=True).fillna(0.5).astype(np.float32)
        return pd.Series(support_rebound * 0.3 + hab_score.values * 0.7, index=df_index).clip(0, 1).astype(np.float32)

    def _synthesize_bullish_intent(self, df_index: pd.Index, aggressiveness_score: pd.Series, control_score: pd.Series, obstacle_clearance_score: pd.Series, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], dynamic_weights: Dict[str, pd.Series], historical_context: Dict[str, Any], params: Dict, proxy_signals: Dict[str, pd.Series], total_risk_penalty: pd.Series) -> pd.Series:
        """
        【V6.4 · 物理分层熔断版】主力多头意图深度合成
        修改说明：引入物理分层熔断算子。当风险惩罚因子 > 0.8 时，通过 NumPy 向量化门控强制阻断所有多头增益输出，确保系统在极端风险环境下的物理防御。
        版本号：2026.02.11.110
        """
        # 1. 准备组件矩阵与 Softmax 权重 (N, 3)
        comp_mat = np.column_stack([aggressiveness_score.values.astype(np.float32), control_score.values.astype(np.float32), obstacle_clearance_score.values.astype(np.float32)])
        w_mat = np.column_stack([dynamic_weights.get('aggressiveness', pd.Series(0.35, index=df_index)).values.astype(np.float32), dynamic_weights.get('control', pd.Series(0.35, index=df_index)).values.astype(np.float32), dynamic_weights.get('obstacle_clearance', pd.Series(0.30, index=df_index)).values.astype(np.float32)])
        burst = proxy_signals.get('sync_burst_score', pd.Series(0.0, index=df_index)).values.astype(np.float32)
        temp = (1.0 / (1.0 + burst * 4.0)).reshape(-1, 1)
        exp_w = np.exp(w_mat / temp)
        softmax_w = exp_w / (np.sum(exp_w, axis=1, keepdims=True) + 1e-9)
        # 2. 向量化合成基础意图
        bullish_base = np.sum(comp_mat * softmax_w, axis=1)
        # 3. 物理分层熔断逻辑 (Circuit Breaker)
        risk_v = total_risk_penalty.values.astype(np.float32)
        circuit_breaker = np.where(risk_v > 0.8, 0.0, 1.0)
        # 4. 物理增强与记忆平滑
        res_score = self._calculate_acceleration_resonance(df_index, mtf_signals).values.astype(np.float32)
        trap_score = self._detect_bear_trap(df_index, mtf_signals, normalized_signals, obstacle_clearance_score).values.astype(np.float32)
        enhancement = np.tanh(res_score * 0.4 + trap_score * 0.8)
        int_mem = historical_context.get('integrated_memory', pd.Series(0.5, index=df_index)).astype(np.float32).values
        mem_rank = pd.Series(int_mem).rolling(55, min_periods=1).rank(pct=True).fillna(0.5).values
        # 应用熔断门控：阻断所有多头输出
        final_intent = (bullish_base * (1.0 + enhancement * 0.25) * (0.9 + mem_rank * 0.1)) * circuit_breaker
        # if self._is_probe_enabled(pd.DataFrame(index=df_index)):
        #     self._probe_print(f"[CIRCUIT_BREAKER_PROBE] Risk_Penalty: {risk_v[-1]:.4f} | Status: {'SHUTDOWN' if circuit_breaker[-1] == 0 else 'NORMAL'}")
        return pd.Series(final_intent, index=df_index).clip(0, 1).astype(np.float32)

    def _calculate_bearish_intent(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], historical_context: Dict[str, Any], factors: Dict[str, pd.Series]) -> pd.Series:
        """
        【V7.2 · 物理抛压加速版】计算复合看跌意图
        修改说明：引入 divergence_pressure（背离压力）作为空头强度调节因子。利用 Numba 加速的 panic_impulse 算子，实现对高位跳水风险的极速响应。
        版本号：2026.02.11.120
        """
        dist_s = pd.Series(normalized_signals.get('distribution_score_norm', 0.0), index=df_index).fillna(0.0).values.astype(np.float32)
        dist_slope = pd.Series(dist_s).diff(1).rolling(3).mean().fillna(0.0).values.astype(np.float32)
        dist_acc = pd.Series(dist_slope).diff(1).fillna(0.0).values.astype(np.float32)
        # 1. 物理存量累积 (Bear-HAB)
        hab_inc = dist_s * (1.0 + np.clip(dist_acc, 0, None))
        dist_hab = pd.Series(hab_inc, index=df_index).rolling(window=21, min_periods=1).sum().fillna(0.0).rolling(34, min_periods=1).rank(pct=True).fillna(0.5).values.astype(np.float32)
        # 2. 诱多陷阱与背离压力整合
        bull_trap = self._detect_bull_trap(df_index, mtf_signals, normalized_signals, pd.Series(dist_hab, index=df_index)).values.astype(np.float32)
        div_p = factors.get('divergence_pressure', pd.Series(0.0, index=df_index)).values.astype(np.float32)
        # 3. 恐慌冲击 (Numba 加速)
        p_acc = mtf_signals.get('ACCEL_5_price_trend', pd.Series(0.0, index=df_index)).values.astype(np.float32)
        panic_impulse = self._numba_calculate_panic_impulse(p_acc, dist_acc)
        # 4. 非线性合成
        # 权重：派发分(0.2) + 存量(0.3) + 恐慌(0.3) + 背离压力(0.2)
        log_bear = (0.2 * np.log(np.clip(dist_s, 1e-6, 1.0)) + 
                    0.3 * np.log(np.clip(dist_hab, 1e-6, 1.0)) + 
                    0.3 * np.log(np.clip(panic_impulse, 1e-6, 1.0)) +
                    0.2 * np.log(np.clip(div_p, 1e-6, 1.0)))
        base_bear = np.exp(log_bear)
        # 5. 最终修正：诱多增强
        final_bear = (base_bear * (1.0 + bull_trap * 0.4)).clip(0, 1)
        # if self._is_probe_enabled(pd.DataFrame(index=df_index)):
        #     self._probe_print(f"--- Bearish Intent Physics Probe V7.2 ---")
        #     self._probe_print(f"  > Panic_Impulse: {panic_impulse[-1]:.4f} | Div_Pressure: {div_p[-1]:.4f} | Final_Bear: {-final_bear[-1]:.4f}")
        return pd.Series(-final_bear, index=df_index).astype(np.float32)

    def _adjudicate_risk(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], dynamic_weights: Dict[str, pd.Series], aggressiveness_score: pd.Series, params: Dict, market_phase: pd.Series, factors: Dict[str, pd.Series]) -> pd.Series:
        """
        【V8.10 · NumPy 索引修复版】深度风险裁决逻辑
        修改说明：修复 'numpy.ndarray' object has no attribute 'iloc' 报错。将探针中的 iloc 访问替换为原生 ndarray 索引，确保向量化路径下的诊断输出正常。
        版本号：2026.02.11.100
        """
        tech_risk = (pd.Series(normalized_signals.get('RSI_norm', 0.5), index=df_index).fillna(0.5).astype(np.float32).values - 0.75).clip(0, None) * 2.0
        struct_risk = normalized_signals.get('distribution_score_norm', pd.Series(0, index=df_index)).fillna(0.0).astype(np.float32).values
        div_pressure = factors.get('divergence_pressure', pd.Series(0, index=df_index)).values.astype(np.float32)
        p_hab = normalized_signals.get('absolute_change_strength_norm', pd.Series(0.5, index=df_index)).astype(np.float32).rolling(21, min_periods=1).mean().fillna(0.5).values
        c_hab = normalized_signals.get('net_mf_amount_norm', pd.Series(0.5, index=df_index)).astype(np.float32).rolling(21, min_periods=1).mean().fillna(0.5).values
        hab_offset = np.clip(p_hab - c_hab, 0, None)
        dynamic_threshold = 0.65 - div_pressure * 0.25
        hollow_risk_base = 1.0 / (1.0 + np.exp(-12.0 * (hab_offset - dynamic_threshold)))
        p_acc = mtf_signals.get('ACCEL_5_price_trend', pd.Series(0, index=df_index)).astype(np.float32).fillna(0.0).values
        phase_risk_map = {"派发初期": 0.2, "主升": 0.3, "横盘": 0.4, "派发末端": 1.2, "反转风险": 1.5}
        risk_modulator = market_phase.map(phase_risk_map).fillna(1.0).values.astype(np.float32)
        acc_factor = np.where(p_acc > 0, 0.5, 1.0)
        final_hollow_risk = np.clip(hollow_risk_base * risk_modulator * acc_factor, 0, 1)
        total_risk_raw = np.clip(tech_risk * 0.15 + struct_risk * 0.2 + final_hollow_risk * 0.45 + div_pressure * 0.2, 0, 1)
        penalty = np.where(total_risk_raw > 0.6, total_risk_raw ** 0.5, total_risk_raw ** 2.2)
        risk_sens = np.float32(get_param_value(params.get('risk_sensitivity'), 4.0))
        # 此处 final_penalty 为 numpy.ndarray
        final_penalty = 1.0 / (1.0 + np.exp(-risk_sens * (penalty - 0.7)))
        # if self._is_probe_enabled(pd.DataFrame(index=df_index)):
        #     # 核心修复点：将 .iloc[-1] 修改为 [-1]
        #     self._probe_print(f"[RISK_ADAPT_PROBE] Div_P: {div_pressure[-1]:.4f} | Dyn_Thresh: {dynamic_threshold[-1]:.4f} | Penalty: {final_penalty[-1]:.4f}")
        return pd.Series(final_penalty, index=df_index).clip(0, 1).astype(np.float32)

    def _apply_contextual_modulators(self, df_index: pd.Index, final_rally_intent: pd.Series, proxy_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], market_phase: pd.Series, aggressiveness_score: pd.Series) -> pd.Series:
        """
        【V4.2 · 逆势溢价惩罚版】应用情境调节器
        修改说明：引入逆势溢价惩罚机制。当相位为“反转风险”且意图分数虚高时，通过非线性 Tanh 强制压制最终分值，防止末端诱多。
        版本号：2026.02.11.115
        """
        def _safe_mod_arr(s_arr, center=0.5, scope=0.15):
            return 1.0 + (np.tanh((s_arr - center) * 4.0) * scope)
        rs_val = proxy_signals.get('enhanced_rs_proxy', pd.Series(0.5, index=df_index)).astype(np.float32).values
        cap_val = proxy_signals.get('enhanced_capital_proxy', pd.Series(0.5, index=df_index)).astype(np.float32).values
        # 1. 基础环境调制 (RS & Capital)
        modulator = _safe_mod_arr(rs_val) * _safe_mod_arr(cap_val, scope=0.1)
        # 2. 投机溢价加成与逆势惩罚矩阵
        spec_mod = np.ones(len(df_index), dtype=np.float32)
        p_v = market_phase.values
        agg_v = aggressiveness_score.values.astype(np.float32)
        intent_v = final_rally_intent.values.astype(np.float32)
        # 逻辑 A: 派发初期的投机溢价加成
        is_early_dist = (p_v == "派发初期")
        premium_inc = np.clip(agg_v - 0.5, 0, None) * 0.4
        spec_mod[is_early_dist] = 1.0 + premium_inc[is_early_dist]
        # 逻辑 B: 反转风险的物理惩罚 (逆势压制)
        is_reversal = (p_v == "反转风险")
        # 当分值 > 0.4 时激活惩罚，分值越高惩罚越重
        reversal_penalty = np.where(intent_v > 0.4, np.tanh(1.0 - intent_v), 1.0)
        spec_mod[is_reversal] = reversal_penalty[is_reversal]
        # 3. 合成最终意图
        final_intent_v = intent_v * modulator * spec_mod
        # if self._is_probe_enabled(pd.DataFrame(index=df_index)):
        #     self._probe_print(f"[CONTEXT_MOD_PROBE] Phase: {p_v[-1]} | Mod: {modulator[-1]:.4f} | Spec_Mod: {spec_mod[-1]:.4f}")
        return pd.Series(final_intent_v, index=df_index).clip(-1, 1).astype(np.float32)

    def _output_probe_info(self, df_index: pd.Index, final_rally_intent: pd.Series):
        """
        【V2.1 · 检索定位优化版】输出探针信息
        修改说明：利用 Pandas 索引集合操作替代反向迭代循环，大幅提升在大数据量下的探针日志提取效率。
        版本号：2026.02.11.47
        """
        if not self.probe_dates: return
        p_dates = pd.to_datetime(self.probe_dates).tz_localize(None).normalize()
        matches = df_index[df_index.tz_localize(None).normalize().isin(p_dates)]
        if not matches.empty:
            p_ts = matches[-1]
            print(f"\n=== CalculateMainForceRallyIntent 主力拉升意图探针报告 @ {p_ts.strftime('%Y-%m-%d')} ===")
            for line in self._probe_output: print(line)
            print(f"最终拉升意图分值: {final_rally_intent.loc[p_ts]:.4f}")
            print("=== 探针报告结束 ===\n")











