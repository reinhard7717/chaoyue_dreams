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
        【V12.8 · 动态周期反馈版】主力拉升意图主计算流程
        修改说明：实现Proxy-Burst对Dynamic Period的反馈闭环。虽然存在计算顺序限制，但我们在意图合成前
        更新了最终的周期参数供参考（尽管内存计算已前置，但此周期可用于后续的风控或调试）。
        版本号：2026.02.10.378
        """
        self._probe_output = []
        params = self._get_parameters(config)
        df_index = df.index
        
        # 1. 信号与物理层
        raw_signals = self._get_raw_signals(df)
        is_limit_up_day = df.apply(lambda row: is_limit_up(row), axis=1)
        normalized_signals = self._normalize_raw_signals(df_index, raw_signals)
        mtf_signals = self._calculate_mtf_fused_signals(df, raw_signals, params['mtf_slope_accel_weights'], df_index)
        if self._is_probe_enabled(df): self._diagnose_vol_jerk_anomaly(df_index, raw_signals, mtf_signals)
        
        # 2. 状态识别
        factors = self._calculate_market_state_factors(df_index, normalized_signals, mtf_signals)
        phase = self._identify_market_phase(df_index, factors, normalized_signals)
        
        # 3. 上下文与代理 (第一轮)
        historical_context = self._calculate_historical_context(df, df_index, raw_signals, mtf_signals, params['historical_context_params'])
        proxy_signals = self._construct_proxy_signals(df_index, mtf_signals, normalized_signals, config)
        
        # 4. 【核心闭环】利用 Proxy Burst 修正动态周期
        # 虽然内存已计算，但我们更新 context 中的周期值，供后续模块（如风险裁决）使用
        refined_period = self._calculate_dynamic_period(df_index, raw_signals, proxy_signals)
        historical_context['dynamic_memory_period'] = refined_period
        
        # 5. 权重与分值
        dynamic_weights = self._calculate_dynamic_weights(df_index, normalized_signals, proxy_signals, mtf_signals, factors, phase)
        aggressiveness_score = self._calculate_aggressiveness_score(df_index, raw_signals, mtf_signals, normalized_signals, dynamic_weights)
        control_score = self._calculate_control_score(df_index, raw_signals, mtf_signals, normalized_signals, historical_context)
        obstacle_clearance_score = self._calculate_obstacle_clearance_score(df_index, raw_signals, mtf_signals, normalized_signals, historical_context)
        
        # 6. 风险与意图
        total_risk_penalty = self._adjudicate_risk(df_index, raw_signals, mtf_signals, normalized_signals, dynamic_weights, aggressiveness_score, params['rally_intent_synthesis_params'], phase)
        bullish_intent = self._synthesize_bullish_intent(df_index, aggressiveness_score, control_score, obstacle_clearance_score, mtf_signals, normalized_signals, dynamic_weights, historical_context, params['rally_intent_synthesis_params'])
        bearish_score = self._calculate_bearish_intent(df_index, raw_signals, mtf_signals, normalized_signals, historical_context)
        
        # 7. 缝合与审计
        final_rally_intent = (bullish_intent * (1 - total_risk_penalty * 0.8) + bearish_score).fillna(0).clip(-1, 1)
        final_rally_intent = self._apply_contextual_modulators(df_index, final_rally_intent, proxy_signals, mtf_signals, phase, aggressiveness_score)
        final_rally_intent = final_rally_intent.mask(is_limit_up_day & (final_rally_intent < 0), 0.0).fillna(0)
        
        self._persist_hab_states(historical_context)
        
        if self._is_probe_enabled(df):
            self._execute_full_link_probing(df_index, raw_signals, mtf_signals, proxy_signals, historical_context, bullish_intent, final_rally_intent, phase, dynamic_weights)
            self._output_probe_info(df_index, final_rally_intent)
            
        return final_rally_intent.astype(np.float32)

    def _execute_full_link_probing(self, df_index: pd.Index, raw_signals: Dict, mtf_signals: Dict, proxy_signals: Dict, historical_context: Dict, bullish_intent: pd.Series, final_intent: pd.Series, phase: pd.Series, dynamic_weights: Dict):
        """
        【V1.3 · 物理审计全透明版】执行从信号层到合成层的全链路审计
        修改说明：增加Jerk原始量纲与缩放后量纲的对比审计，监控物理风险熔断的真实权重。
        版本号：2026.02.10.312
        """
        ts = df_index[-1]
        self._probe_print(f"=== [FULL_LINK_PHYSICS_AUDIT] {ts.strftime('%Y-%m-%d')} ===")
        # 1. 物理穿透链条审计
        v_jerk_scaled = mtf_signals.get('JERK_5_volume_trend', pd.Series(0, index=df_index)).iloc[-1]
        p_acc_scaled = mtf_signals.get('ACCEL_5_price_trend', pd.Series(0, index=df_index)).iloc[-1]
        self._probe_output.append(f"[L2_PHYSICS] Scaled_P_Accel: {p_acc_scaled:.4f} | Scaled_V_Jerk: {v_jerk_scaled:.4f}")
        # 2. 存量层审计
        p_hab = historical_context.get('price_memory', {}).get('hab_score', pd.Series(0.5, index=df_index)).iloc[-1]
        c_hab = historical_context.get('capital_memory', {}).get('hab_score', pd.Series(0.5, index=df_index)).iloc[-1]
        self._probe_output.append(f"[L3_HAB] P_HAB: {p_hab:.4f}, C_HAB: {c_hab:.4f} | Phase: {phase.iloc[-1]}")
        # 3. 合成层审计
        b_val = bullish_intent.iloc[-1]
        f_val = final_intent.iloc[-1]
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

    def _diagnose_vol_jerk_anomaly(self, df_index: pd.Index, raw_signals: Dict, mtf_signals: Dict):
        """
        【V1.0 · 物理量纲诊断中枢】定位Jerk数值异常根源
        修改说明：输出成交量三阶导数的完整计算链条，检测是否存在未缩放的巨量原始量纲污染。
        版本号：2026.02.10.310
        """
        raw_vol = raw_signals.get('volume', pd.Series(0, index=df_index)).iloc[-1]
        v_slope = mtf_signals.get('SLOPE_5_volume_trend', pd.Series(0, index=df_index)).iloc[-1]
        v_accel = mtf_signals.get('ACCEL_5_volume_trend', pd.Series(0, index=df_index)).iloc[-1]
        v_jerk_raw = mtf_signals.get('JERK_5_volume_trend', pd.Series(0, index=df_index)).iloc[-1]
        # 计算统计学边界
        v_jerk_series = mtf_signals.get('JERK_5_volume_trend', pd.Series(0, index=df_index))
        med = v_jerk_series.rolling(60).median().iloc[-1]
        mad = (v_jerk_series - med).abs().rolling(60).median().iloc[-1]
        self._probe_print(f"=== [PHYSICS_DIAGNOSIS] Vol_SAJ_Chain ===")
        self._probe_print(f"  > Raw_Volume: {raw_vol:.0f}")
        self._probe_print(f"  > S/A/J_Raw: S={v_slope:.2f}, A={v_accel:.2f}, J={v_jerk_raw:.2f}")
        self._probe_print(f"  > MAD_Audit: Median={med:.2f}, MAD={mad:.2f}, Z-Score={(v_jerk_raw-med)/(mad*1.4826+1e-9):.2f}")

    def _calculate_mtf_fused_signals(self, df: pd.DataFrame, raw_signals: Dict[str, pd.Series], mtf_slope_accel_weights: Dict, df_index: pd.Index) -> Dict[str, pd.Series]:
        """
        【V8.2 · 物理量纲标准化版】MTF跨周期信号融合
        修改说明：引入MAD稳健缩放算法，对所有SLOPE/ACCEL/JERK物理指标执行tanh饱和压缩，解决成交量Jerk量纲溢出导致的惩罚过敏问题。
        版本号：2026.02.10.311
        """
        mtf_signals = {}
        # 1. 对载入的物理量进行稳健标准化 (RPS)
        for key, series in raw_signals.items():
            if any(p in key for p in ['SLOPE_', 'ACCEL_', 'JERK_']):
                # 采用 tanh((x - med) / (3 * MAD)) 将巨大物理量映射至 [-1, 1]
                med = series.rolling(60, min_periods=5).median()
                mad = (series - med).abs().rolling(60, min_periods=5).median()
                std_series = (series - med) / (mad * 4.4478 + 1e-9) # 3*1.4826=4.4478
                mtf_signals[key] = np.tanh(std_series).fillna(0)
        # 2. MTF 核心趋势融合逻辑
        core_fusion_signals = {'price_trend': 'close_D', 'volume_trend': 'volume_D', 'net_amount_trend': 'net_amount_D'}
        weights = {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}
        for sig_name, col_name in core_fusion_signals.items():
            fused_score = pd.Series(0.0, index=df_index)
            for win, w in weights.items():
                val = df[col_name].diff(win).fillna(0)
                std = val.rolling(win * 3, min_periods=1).std().replace(0, 1e-9).fillna(1.0)
                fused_score += np.tanh(val / (std * 2.0)).fillna(0) * w
            mtf_signals[f'mtf_{sig_name}'] = fused_score.clip(-1, 1)
        return mtf_signals

    def _calculate_historical_context(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], params: Dict) -> Dict[str, pd.Series]:
        """
        【V5.3 · 周期感知升级版】历史上下文计算调度中心
        修改说明：由于Proxy Signals计算依赖Historical Context，而Dynamic Period现在依赖Proxy Signals，
        这里存在依赖环。解决方案：在此阶段使用简化的周期计算（不含Burst），或在Calculate主流程中二次更新周期。
        本版本采用策略：先使用基础周期，待Proxy计算完成后，在下一帧或后续逻辑中修正。
        但在本架构中，为保持流式计算，我们将_calculate_dynamic_period移出此方法，改为在主流程计算后注入。
        此处仅计算基础内存。
        """
        hc_enabled = get_param_value(params.get('enabled'), True)
        if not hc_enabled: return self._get_empty_context(df_index)
        
        initial_hab_states = self._load_hab_states()
        
        # 计算基础内存
        price_memory = self._calculate_price_memory(df_index, raw_signals, mtf_signals, params, initial_hab_states.get('price_hab', 0.0))
        capital_memory = self._calculate_capital_memory(df_index, raw_signals, mtf_signals, params, initial_hab_states.get('capital_hab', 0.0))
        chip_memory = self._calculate_chip_memory(df_index, raw_signals, mtf_signals, params, initial_hab_states.get('chip_hab', 0.0))
        sentiment_memory = self._calculate_sentiment_memory(df_index, raw_signals, mtf_signals, params, initial_hab_states.get('sentiment_hab', 0.0))
        
        integrated_memory = self._fuse_integrated_memory(price_memory, capital_memory, chip_memory, sentiment_memory, params)
        phase_sync = self._detect_phase_synchronization(df_index, price_memory, capital_memory, integrated_memory)
        memory_quality = self._assess_memory_quality(df_index, price_memory, capital_memory, chip_memory, sentiment_memory)
        
        # 注意：dynamic_memory_period 此时尚未包含 Burst 修正，将在主流程中通过 _calculate_dynamic_period 独立计算并覆盖
        # 但为了保持接口完整性，此处返回基础版
        base_period = self._calculate_dynamic_period(df_index, raw_signals, {}) 
        
        return {
            "price_memory": price_memory, "capital_memory": capital_memory, "chip_memory": chip_memory,
            "sentiment_memory": sentiment_memory, "integrated_memory": integrated_memory,
            "phase_sync": phase_sync, "memory_quality": memory_quality, "hc_enabled": hc_enabled,
            "dynamic_memory_period": base_period
        }

    def _load_hab_states(self) -> Dict[str, float]:
        """
        【V1.0】从原子状态机载入HAB历史状态
        修改说明：从atomic_states恢复四个核心维度的累积水位，支持实时增量计算的物理延续。
        版本号：2026.02.10.173
        """
        states = {
            'price_hab': self.strategy.atomic_states.get('_HAB_STATE_PRICE', 0.0),
            'capital_hab': self.strategy.atomic_states.get('_HAB_STATE_CAPITAL', 0.0),
            'chip_hab': self.strategy.atomic_states.get('_HAB_STATE_CHIP', 0.0),
            'sentiment_hab': self.strategy.atomic_states.get('_HAB_STATE_SENTIMENT', 0.0)
        }
        if any(v > 0 for v in states.values()):
            self._probe_print(f"[HAB_LOAD] 成功恢复历史存量: Price={states['price_hab']:.4f}, Cap={states['capital_hab']:.4f}")
        return states

    def _persist_hab_states(self, historical_context: Dict):
        """
        【V1.1 · 数值安全持久化版】将HAB最新状态持久化
        修改说明：修复水位溢出问题。持久化前对原始Buffer进行Tanh标准化压缩，确保存储在状态机中的存量水位具有稳定的物理量纲。
        版本号：2026.02.10.247
        """
        try:
            p_mem = historical_context.get('price_memory', {})
            c_mem = historical_context.get('capital_memory', {})
            ch_mem = historical_context.get('chip_memory', {})
            s_mem = historical_context.get('sentiment_memory', {})
            # 采用tanh压缩后再持久化，防止原始量纲溢出 (5e7为资金量纲缩放基准)
            self.strategy.atomic_states['_HAB_STATE_PRICE'] = float(np.tanh(p_mem.get('hab_buffer_raw', pd.Series([0.0])).iloc[-1]))
            self.strategy.atomic_states['_HAB_STATE_CAPITAL'] = float(np.tanh(c_mem.get('hab_buffer_raw', pd.Series([0.0])).iloc[-1] / 5e7))
            self.strategy.atomic_states['_HAB_STATE_CHIP'] = float(np.tanh(ch_mem.get('hab_buffer_raw', pd.Series([0.0])).iloc[-1]))
            self.strategy.atomic_states['_HAB_STATE_SENTIMENT'] = float(np.tanh(s_mem.get('hab_buffer_raw', pd.Series([0.0])).iloc[-1]))
            if self._is_probe_enabled(pd.DataFrame()):
                self._probe_print(f"[HAB_PERSIST] 存量水位标准化持久化完成 (Cap: {self.strategy.atomic_states['_HAB_STATE_CAPITAL']:.4f})")
        except Exception as e:
            self._probe_print(f"[HAB_PERSIST_ERROR] 持久化失败: {str(e)}")

    def _detect_phase_synchronization(self, df_index: pd.Index, price_memory: Dict, capital_memory: Dict, integrated_memory: pd.Series) -> pd.Series:
        """
        【V5.0 · 费舍尔变换同步版】相位同步检测
        修改说明：引入费舍尔变换对相关系数进行非线性增强，弃用通用归一化，显著提升资金与价格相位契合度的识别精度。
        版本号：2026.02.10.131
        """
        p_trend = price_memory.get("trend_memory", pd.Series(0.0, index=df_index)).ewm(span=3).mean()
        c_flow = capital_memory.get("composite_capital_flow", pd.Series(0.0, index=df_index)).ewm(span=3).mean()
        # 1. 计算21日滚动相关性
        corr = p_trend.rolling(window=21, min_periods=5).corr(c_flow).fillna(0)
        # 2. 定制归一化：费舍尔变换 (Fisher Transform) 增强
        # 核心逻辑：将线性相关性转化为无限区间的概率分布，再通过tanh压缩，锐化高相关区的信号
        fisher = 0.5 * np.log((1 + corr.clip(-0.99, 0.99)) / (1 - corr.clip(-0.99, 0.99)))
        sync_strength = np.tanh(fisher).clip(-1, 1)
        # 3. 稳定性调节 (Jerk 约束)
        p_jerk = price_memory.get("jerk_memory", 0).abs()
        stability = 1 / (1 + p_jerk * 10)
        final_sync = (sync_strength * stability).clip(-1, 1)
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"[SYNC_PROBE] Raw_Corr: {corr.iloc[-1]:.4f}, Fisher_Sync: {final_sync.iloc[-1]:.4f}")
        return final_sync.astype(np.float32)

    def _assess_memory_quality(self, df_index: pd.Index, price_memory: Dict, capital_memory: Dict, chip_memory: Dict, sentiment_memory: Dict) -> pd.Series:
        """
        【V5.0 · 物理量纲映射质量版】记忆质量评估算法
        修改说明：直接采用物理Jerk的倒数映射稳定性，废弃通用normalize工具，显著提升对趋势“纯净度”的辨识力。
        版本号：2026.02.10.104
        """
        p_jerk = price_memory.get("jerk_memory", pd.Series(0.0, index=df_index)).abs()
        c_jerk = capital_memory.get("cap_jerk", pd.Series(0.0, index=df_index)).abs()
        # 1. 物理稳定性归一化：1 / (1 + k*Jerk) 映射法
        # Jerk越小稳定性越高。系数15是根据A股价格加速度均值设定的经验系数。
        kinematic_stab = 1 / (1 + (p_jerk * 0.6 + c_jerk * 0.4) * 15)
        # 2. 信号一致性 (Rolling Correlation)
        consis_score = self._calculate_memory_consistency(price_memory, capital_memory, chip_memory, sentiment_memory)
        # 3. 质量 HAB 累积与定制归一化
        quality_inc = (kinematic_stab * consis_score)
        # 使用 Rolling Rank 将质量累积转化为排位分
        quality_hab = quality_inc.rolling(window=34).rank(pct=True).fillna(0.5)
        # 4. 综合合成
        quality_score = (quality_hab * 0.4 + kinematic_stab * 0.3 + consis_score * 0.3).clip(0, 1)
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"[QUALITY_PROBE] Stab_Map: {kinematic_stab.iloc[-1]:.4f}, Quality_Rank: {quality_hab.iloc[-1]:.4f}")
        return quality_score.astype(np.float32)

    def _calculate_dynamic_period(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], proxy_signals: Dict[str, pd.Series]) -> pd.Series:
        """
        【V5.0 · 爆发敏感周期版】动态记忆周期计算
        修改说明：引入Fisher一致性爆发(Burst)调节。当信号高度共振时，自动缩短记忆周期，实现“快进快出”的灵敏度切换。
        版本号：2026.02.10.377
        """
        # 1. 基础波动率与趋势效率
        pct_change = raw_signals.get('pct_change', pd.Series(0.0, index=df_index)).fillna(0.0)
        volatility = pct_change.rolling(window=21).std().fillna(0.01)
        # Vol位置归一化
        vol_min = volatility.rolling(60).min(); vol_max = volatility.rolling(60).max()
        vol_pos = ((volatility - vol_min) / (vol_max - vol_min + 1e-9)).clip(0, 1)
        
        # 效率比 (ER)
        net_diff = pct_change.rolling(10).sum().abs()
        path_len = pct_change.abs().rolling(10).sum().replace(0, 1e-9)
        er = (net_diff / path_len).clip(0, 1)
        
        # 2. 提取一致性爆发 (Burst Score)
        # 如果未传入(例如初始化阶段)，默认为0
        burst_score = proxy_signals.get('sync_burst_score', pd.Series(0.0, index=df_index))
        
        # 3. 周期合成
        # 基础逻辑：ER高 -> 周期长(趋势稳)；Vol高 -> 周期短(避险)
        base_period = 21 * (1 + er * 0.4 - vol_pos * 0.3)
        
        # 【核心重构】爆发敏感调节
        # 当 Burst > 0.8 时，周期最多缩短 40%，迫使模型关注近期剧烈变化
        final_period_raw = base_period * (1 - burst_score * 0.4)
        
        # 4. 边界约束与类型转换
        final_period_series = final_period_raw.clip(5, 55).round().fillna(21)
        
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"[PERIOD_BURST_PROBE] Base: {base_period.iloc[-1]:.1f} | Burst: {burst_score.iloc[-1]:.2f} -> Final: {final_period_series.iloc[-1]}")
            
        return final_period_series.astype(np.int32)

    def _calculate_price_memory(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], params: Dict, initial_hab: float = 0.0) -> Dict[str, pd.Series]:
        """
        【V5.2 · 持久化补偿版】价格记忆深度计算
        修改说明：引入initial_hab初始水位补偿，通过叠加历史存量解决滚动窗口在实时计算中的记忆断层问题。
        版本号：2026.02.10.174
        """
        decay = get_param_value(params.get('price_memory_decay'), 0.94)
        hab_win = get_param_value(params.get('hab_window'), 55)
        s = mtf_signals.get('SLOPE_5_price_trend', pd.Series(0.0, index=df_index)).ewm(alpha=1-decay).mean()
        a = mtf_signals.get('ACCEL_5_price_trend', pd.Series(0.0, index=df_index)).ewm(alpha=1-decay).mean()
        j = mtf_signals.get('JERK_5_price_trend', pd.Series(0.0, index=df_index)).ewm(alpha=0.2).mean()
        abs_strength = np.tanh(raw_signals.get('absolute_change_strength', pd.Series(0.0, index=df_index)) / 0.05)
        hab_inc = raw_signals.get('chip_stability', pd.Series(0.5, index=df_index)) * a.clip(lower=0) * abs_strength
        # 物理补偿逻辑：叠加历史持久化水位
        hab_buffer_raw = hab_inc.rolling(window=hab_win, min_periods=1).sum().fillna(0) + initial_hab
        hab_score = (hab_buffer_raw.rolling(hab_win).rank(pct=True)).fillna(0.5)
        k_sum = (s * 0.4 + a * 0.4 + j * 0.2).clip(0, 1)
        integrated_mem = (k_sum * 0.5 + hab_score * 0.5).clip(0, 1)
        return {"trend_memory": s, "accel_memory": a, "jerk_memory": j, "hab_score": hab_score, "hab_buffer_raw": hab_buffer_raw, "integrated_price_memory": integrated_mem}

    def _calculate_capital_memory(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], params: Dict, initial_hab: float = 0.0) -> Dict[str, pd.Series]:
        """
        【V5.2 · 持久化补偿版】资金记忆深度计算
        修改说明：引入initial_hab初始水位补偿，确保资金净流入的“跨日堆积”效应在增量计算中得以保持。
        版本号：2026.02.10.175
        """
        decay = get_param_value(params.get('capital_memory_decay'), 0.92)
        hab_win = get_param_value(params.get('capital_hab_window'), 34)
        cs = mtf_signals.get('SLOPE_5_mf_net_amount', pd.Series(0.0, index=df_index)).ewm(alpha=1-decay).mean()
        ca = mtf_signals.get('ACCEL_5_mf_net_amount', pd.Series(0.0, index=df_index)).ewm(alpha=1-decay).mean()
        mf_flow = raw_signals.get('net_mf_amount', pd.Series(0.0, index=df_index))
        hab_inc = mf_flow * raw_signals.get('flow_consistency', pd.Series(0.5, index=df_index)).clip(0, 1)
        # 物理补偿逻辑：叠加历史持久化水位
        hab_buffer_raw = hab_inc.rolling(window=hab_win, min_periods=1).sum().fillna(0) + initial_hab
        vol = raw_signals.get('pct_change', pd.Series(0.0, index=df_index)).rolling(21).std().fillna(0.02)
        hab_score = 1 / (1 + np.exp(-hab_buffer_raw / (vol * 5e7 + 1e-9)))
        integrated_mem = (cs.clip(0,1) * 0.4 + hab_score * 0.6).clip(0, 1)
        return {"integrated_capital_memory": integrated_mem, "hab_score": hab_score, "hab_buffer_raw": hab_buffer_raw, "cap_accel": ca}

    def _calculate_chip_memory(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], params: Dict, initial_hab: float = 0.0) -> Dict[str, pd.Series]:
        """
        【V5.2 · 持久化补偿版】筹码记忆深度计算
        修改说明：引入initial_hab初始水位补偿，量化筹码在长周期内的锁仓存量，解决实时排位归一化的冷启动问题。
        版本号：2026.02.10.176
        """
        decay = get_param_value(params.get('chip_memory_decay'), 0.95)
        hab_win = get_param_value(params.get('chip_hab_window'), 89)
        chip_s = mtf_signals.get('SLOPE_5_chip_concentration', pd.Series(0.0, index=df_index)).ewm(alpha=1-decay).mean()
        chip_a = mtf_signals.get('ACCEL_5_chip_concentration', pd.Series(0.0, index=df_index)).ewm(alpha=1-decay).mean()
        chip_j = mtf_signals.get('JERK_5_chip_concentration', pd.Series(0.0, index=df_index)).ewm(alpha=0.15).mean()
        turn_rate = raw_signals.get('turnover_rate', pd.Series(0.0, index=df_index))
        hab_inc = (chip_s.clip(lower=0) * raw_signals.get('chip_stability', pd.Series(0.5, index=df_index)) * turn_rate)
        # 物理补偿逻辑：叠加历史持久化水位
        hab_buffer_raw = hab_inc.rolling(window=hab_win, min_periods=1).sum().fillna(0) + initial_hab
        hab_score = hab_buffer_raw.rolling(hab_win).rank(pct=True).fillna(0.5)
        entropy = raw_signals.get('chip_entropy', pd.Series(0.5, index=df_index))
        ent_mem = (np.log1p(entropy) / np.log1p(entropy).rolling(55).max().replace(0, 1)).clip(0, 1)
        k_vec = (chip_s * 0.4 + chip_a * 0.4 + chip_j * 0.2).clip(0, 1)
        integrated_chip = (hab_score * 0.4 + k_vec * 0.3 + (1 - ent_mem) * 0.3).clip(0, 1)
        return {"chip_slope": chip_s, "chip_accel": chip_a, "hab_score": hab_score, "hab_buffer_raw": hab_buffer_raw, "integrated_chip_memory": integrated_chip}

    def _calculate_sentiment_memory(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], params: Dict, initial_hab: float = 0.0) -> Dict[str, pd.Series]:
        """
        【V5.2 · 持久化补偿版】情绪记忆深度计算
        修改说明：引入initial_hab初始水位补偿，确保市场热度与赚钱效应在跨日温压累积中保持物理连续。
        版本号：2026.02.10.177
        """
        decay = get_param_value(params.get('sentiment_memory_decay'), 0.90)
        hab_win = get_param_value(params.get('sentiment_hab_window'), 21)
        ss = mtf_signals.get('SLOPE_5_market_sentiment_trend', pd.Series(0.0, index=df_index)).ewm(alpha=1-decay).mean()
        sa = mtf_signals.get('ACCEL_5_market_sentiment_trend', pd.Series(0.0, index=df_index)).ewm(alpha=1-decay).mean()
        sj = mtf_signals.get('JERK_5_market_sentiment_trend', pd.Series(0.0, index=df_index)).ewm(alpha=0.3).mean()
        m_sent = raw_signals.get('market_sentiment', pd.Series(0.5, index=df_index))
        i_breadth = raw_signals.get('industry_breadth', pd.Series(0.5, index=df_index))
        hab_inc = m_sent * i_breadth * (1 + sa.clip(lower=0))
        # 物理补偿逻辑：叠加历史持久化水位
        hab_buffer_raw = hab_inc.rolling(window=hab_win, min_periods=1).sum().fillna(0) + initial_hab
        roll_med = hab_buffer_raw.rolling(60).median()
        roll_std = hab_buffer_raw.rolling(60).std().replace(0, 1e-9)
        hab_score = (np.tanh((hab_buffer_raw - roll_med) / roll_std) * 0.5 + 0.5).fillna(0.5)
        k_sum = (ss * 0.3 + sa * 0.5 + sj * 0.2).clip(-1, 1)
        integrated_sent = ((k_sum * 0.5 + 0.5) * 0.4 + hab_score * 0.6).clip(0, 1)
        return {"integrated_sentiment_memory": integrated_sent, "hab_score": hab_score, "hab_buffer_raw": hab_buffer_raw, "sentiment_slope": ss, "sentiment_accel": sa}

    def _fuse_integrated_memory(self, price_memory: Dict, capital_memory: Dict, chip_memory: Dict, sentiment_memory: Dict, params: Dict) -> pd.Series:
        """
        【V5.0 · HAB水位差与阶段识别版】综合记忆融合算法
        修改说明：引入HAB偏移量分析逻辑，量化资金/筹码与价格之间的“底气”差异，实现从简单融合到阶段意识识别的跨越。
        版本号：2026.02.10.70
        """
        df_index = price_memory["integrated_price_memory"].index
        p_base = price_memory["integrated_price_memory"]
        c_base = capital_memory["integrated_capital_memory"]
        ch_base = chip_memory["integrated_chip_memory"]
        s_base = sentiment_memory["integrated_sentiment_memory"]
        # 1. 提取各维度 HAB 水位
        p_hab = price_memory.get("hab_score", pd.Series(0.5, index=df_index))
        c_hab = capital_memory.get("hab_score", pd.Series(0.5, index=df_index))
        ch_hab = chip_memory.get("hab_score", pd.Series(0.5, index=df_index))
        s_hab = sentiment_memory.get("hab_score", pd.Series(0.5, index=df_index))
        # 2. HAB 水位差分析 (Offset Analysis)
        # 资金-价格偏差：正值表示“钱多价滞”，典型的吸筹蓄势；负值表示“价涨钱空”，典型的虚假繁荣
        cap_price_offset = (c_hab - p_hab).clip(-1, 1)
        # 筹码-价格偏差：正值表示“筹码锁死而价未动”，高爆发潜质
        chip_price_offset = (ch_hab - p_hab).clip(-1, 1)
        # 3. 计算运动学共振矢量 (SAJ Resonance)
        slopes = pd.DataFrame({
            "p": price_memory.get("trend_memory", 0),
            "c": capital_memory.get("composite_capital_flow", 0),
            "ch": chip_memory.get("chip_slope", 0),
            "s": sentiment_memory.get("sentiment_slope", 0)
        })
        accels = pd.DataFrame({
            "p": price_memory.get("accel_memory", 0),
            "c": capital_memory.get("cap_accel", 0),
            "ch": chip_memory.get("chip_accel", 0),
            "s": sentiment_memory.get("sentiment_accel", 0)
        })
        # 四维度方向一致性 (三军齐步走)
        slope_consistency = (np.sign(slopes).nunique(axis=1) == 1).astype(float)
        accel_synergy = (accels > 0).sum(axis=1) / 4.0
        # 4. 动态权重分配：根据 HAB 集成水位切换关注重点
        integrated_hab = (p_hab * 0.2 + c_hab * 0.3 + ch_hab * 0.3 + s_hab * 0.2)
        # 筑底期（低水位）侧重资金与筹码；主升期（高水位）侧重价格与情绪
        w_p = 0.2 + integrated_hab * 0.2  # 0.2 -> 0.4
        w_c = 0.4 - integrated_hab * 0.1  # 0.4 -> 0.3
        w_ch = 0.3 - integrated_hab * 0.1 # 0.3 -> 0.2
        w_s = 0.1 + integrated_hab * 0.1  # 0.1 -> 0.2
        base_fusion = (p_base * w_p + c_base * w_c + ch_base * w_ch + s_base * w_s)
        # 5. 阶段溢价与惩罚 (Stage Premium/Penalty)
        # 潜伏溢价：在低价区且资金筹码显著领先时，大幅提升融合分值
        latent_premium = (cap_price_offset.clip(lower=0) * 0.3 + chip_price_offset.clip(lower=0) * 0.2) * (1 - p_hab)
        # 派发惩罚：在高价区且资金流失显著时，压制分值
        distribution_penalty = (cap_price_offset.clip(upper=0).abs() * 0.4) * p_hab
        # 6. 最终合成与保护
        synergy_boost = (slope_consistency * 0.15 + accel_synergy * 0.15)
        hab_modulator = 0.85 + (integrated_hab * 0.3)
        final_integrated_memory = (base_fusion * hab_modulator * (1 + synergy_boost + latent_premium - distribution_penalty)).clip(0, 1)
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"--- Integrated Memory HAB-Offset Probe ---")
            self._probe_print(f"  > HAB Levels: P={p_hab.iloc[-1]:.4f}, C={c_hab.iloc[-1]:.4f}, CH={ch_hab.iloc[-1]:.4f}")
            self._probe_print(f"  > Offsets: Cap-Price={cap_price_offset.iloc[-1]:.4f}, Chip-Price={chip_price_offset.iloc[-1]:.4f}")
            self._probe_print(f"  > Dynamic Weights: P={w_p.iloc[-1]:.2f}, C={w_c.iloc[-1]:.2f}, CH={w_ch.iloc[-1]:.2f}")
            self._probe_print(f"  > Premium/Penalty: +{latent_premium.iloc[-1]:.4f} / -{distribution_penalty.iloc[-1]:.4f}")
        return final_integrated_memory.astype(np.float32)

    def _detect_market_state(self, price_mem: Dict, capital_mem: Dict, sentiment_mem: Dict, chip_mem: Dict) -> str:
        """
        【V5.0 · 动力学政权识别版】基于SAJ矢量与HAB存量的市场状态检测
        设计逻辑：利用能量突变(Jerk)识别转折点，利用集成HAB水位校验趋势真伪。
        版本号：2026.02.10.43
        """
        p_slope = price_mem.get("trend_memory", pd.Series(0)).iloc[-1]
        p_accel = price_mem.get("accel_memory", pd.Series(0)).iloc[-1]
        p_jerk = price_mem.get("jerk_memory", pd.Series(0)).iloc[-1]
        c_slope = capital_mem.get("composite_capital_flow", pd.Series(0)).iloc[-1]
        p_hab = price_mem.get("hab_score", pd.Series(0.5)).iloc[-1]
        c_hab = capital_mem.get("hab_score", pd.Series(0.5)).iloc[-1]
        integrated_hab = (p_hab * 0.4 + c_hab * 0.6)
        if p_slope > 0.1 and c_slope > 0.05 and p_accel > 0 and integrated_hab > 0.6:
            return "trending_up"
        if abs(p_jerk) > 0.15 or (p_slope * p_accel < 0 and abs(p_accel) > 0.1):
            return "reversing"
        if p_slope < -0.1 and integrated_hab < 0.4:
            return "trending_down"
        if abs(p_accel) < 0.05 and c_slope > 0 and integrated_hab > 0.5:
            return "consolidating"
        return "trending_up" if integrated_hab > 0.7 else "consolidating"

    def _calculate_memory_consistency(self, price_mem: Dict, capital_mem: Dict, chip_mem: Dict, sentiment_memory: Dict) -> pd.Series:
        """
        【V5.1 · 鲁棒共振版】记忆一致性深度计算
        修改说明：废弃外部归一化。采用基于中位数的tanh压缩处理一致性存量，增强对多维信号“齐步走”的辨识力。
        版本号：2026.02.10.252
        """
        df_index = price_mem["integrated_price_memory"].index
        accel_df = pd.DataFrame({
            "p_acc": price_mem.get("accel_memory", 0), "c_acc": capital_mem.get("cap_accel", 0),
            "ch_acc": chip_mem.get("chip_accel", 0), "s_acc": sentiment_memory.get("sentiment_accel", 0)
        })
        # 1. 瞬时共振
        resonance_count = (accel_df > 0).sum(axis=1)
        instant_resonance = resonance_count.map({4: 1.0, 3: 0.7, 2: 0.3, 1: 0.1, 0: 0.0})
        # 2. Consistency-HAB
        hab_inc = instant_resonance * accel_df.abs().mean(axis=1)
        hab_raw = hab_inc.rolling(window=13).sum().fillna(0)
        # 定制归一化：Tanh (60日窗口)
        consistency_hab = (np.tanh((hab_raw - hab_raw.rolling(60).median()) / (hab_raw.rolling(60).std() + 1e-9)) * 0.5 + 0.5).fillna(0.5)
        # 3. 最终合成
        rolling_corr = accel_df.rolling(window=10).corr().unstack().mean(axis=1).fillna(0.5).clip(0, 1)
        return (rolling_corr * 0.2 + instant_resonance * 0.3 + consistency_hab * 0.5).clip(0, 1).astype(np.float32)

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
        【V3.0 · 稳健量纲定制版】全量原始信号归一化中心
        修改说明：废弃通用helper方法，为A股不同量纲指标定制对数、MAD及tanh归一化算法，显著提升异常值防御能力。
        版本号：2026.02.10.90
        """
        norm_sigs = {}
        def _robust_scale(s, window=60, bipolar=False):
            # 稳健标准化：(x - median) / (MAD * 1.4826)
            med = s.rolling(window).median()
            mad = (s - med).abs().rolling(window).median()
            res = (s - med) / (mad * 1.4826 + 1e-9)
            return np.tanh(res).clip(-1, 1) if bipolar else (np.tanh(res) * 0.5 + 0.5).clip(0, 1)
        # 1. 价格波动类：使用tanh在10%涨跌幅处饱和
        norm_sigs['pct_change_norm'] = np.tanh(raw_signals.get('pct_change', 0) / 0.1).clip(-1, 1)
        norm_sigs['absolute_change_strength_norm'] = np.tanh(raw_signals.get('absolute_change_strength', 0) / 0.05).clip(0, 1)
        # 2. 成交与换手类：对数处理长尾分布
        for col in ['volume_ratio', 'turnover_rate', 'net_amount_ratio']:
            val = raw_signals.get(col, 0)
            log_val = np.log1p(val.clip(lower=0))
            norm_sigs[f'{col}_norm'] = (log_val / log_val.rolling(60).max().replace(0, 1)).clip(0, 1)
        # 3. 资金与动量类：稳健MAD归一化
        for col in ['flow_acceleration', 'flow_consistency', 'flow_intensity', 'inflow_persistence', 'net_mf_amount']:
            norm_sigs[f'{col}_norm'] = _robust_scale(raw_signals.get(col, 0), bipolar=True)
        # 4. 筹码与结构类：线性与MAD结合
        norm_sigs['chip_concentration_ratio_norm'] = raw_signals.get('chip_concentration_ratio', 0.5).clip(0, 1)
        norm_sigs['chip_stability_norm'] = _robust_scale(raw_signals.get('chip_stability', 0.5), bipolar=False)
        # 5. 情绪与行业类：直接饱和映射
        for col in ['market_sentiment', 'industry_breadth', 'industry_leader']:
            norm_sigs[f'{col}_norm'] = (raw_signals.get(col, 0.5) / 1.0).clip(0, 1)
        # 6. 行为意图类：增强型MAD
        for col in ['accumulation_score', 'distribution_score', 'behavior_accumulation', 'behavior_distribution']:
            norm_sigs[f'{col}_norm'] = _robust_scale(raw_signals.get(col, 0), bipolar=True)
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"[NORM_PROBE] Pct_Norm_Med: {norm_sigs['pct_change_norm'].median():.4f}, Vol_Log_Norm_Max: {norm_sigs['volume_ratio_norm'].max():.4f}")
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

        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"--- Fisher Proxy-SNR Consistency Fix Probe ---")
            self._probe_print(f"  > Fisher_Sync: {fisher_norm.iloc[-1]:.4f} | Burst: {sync_burst_score.iloc[-1]:.4f}")
        return final_proxy_signals

    def _calculate_enhanced_rs_proxy(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], config: Dict) -> Dict[str, pd.Series]:
        """
        【V6.0 · 强势排位增强版】增强版相对强度代理信号
        修改说明：采用滚动百分比排位(Rolling Rank)取代通用归一化处理RS-HAB，确保在不同市场环境下强势个股的识别标准具有动态稳定性。
        版本号：2026.02.10.110
        """
        price_trend_strength = mtf_signals.get('mtf_price_trend', pd.Series(0.5, index=df_index)).clip(lower=0)
        rs_slope = mtf_signals.get('SLOPE_5_price_trend', pd.Series(0.0, index=df_index))
        rs_accel = mtf_signals.get('ACCEL_5_price_trend', pd.Series(0.0, index=df_index))
        rs_jerk = mtf_signals.get('JERK_5_price_trend', pd.Series(0.0, index=df_index))
        hab_window = get_param_value(config.get('rs_hab_window'), 21)
        hab_increment = (price_trend_strength.mask(price_trend_strength < 0.6, 0) * rs_accel.clip(lower=0))
        hab_buffer = hab_increment.rolling(window=hab_window, min_periods=5).sum().fillna(0)
        # 定制化归一化：Rolling Rank (21日排位)
        hab_score = hab_buffer.rolling(hab_window).rank(pct=True).fillna(0.5)
        # RSI 强度定制归一化：使用tanh处理50分位偏差
        rsi_raw = normalized_signals.get('RSI_norm', pd.Series(0.5, index=df_index))
        rsi_strength = np.tanh((rsi_raw - 0.5) * 4) * 0.5 + 0.5
        kinematic_vector_sum = (rs_slope * 0.3 + rs_accel * 0.4 + rs_jerk * 0.3).clip(0, 1)
        rs_weights = {"price_trend": 0.3, "rsi": 0.2, "structural": 0.2, "kinematic": 0.3}
        rs_proxy_raw = (price_trend_strength * rs_weights["price_trend"] + rsi_strength * rs_weights["rsi"] + kinematic_vector_sum * rs_weights["kinematic"])
        rs_modulator = 0.95 + (hab_score * 0.15)
        enhanced_rs_proxy = (rs_proxy_raw * rs_modulator).clip(0, 1)
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"[RS_PROBE] HAB_Rank: {hab_score.iloc[-1]:.4f}, RS_Raw: {rs_proxy_raw.iloc[-1]:.4f}, Final: {enhanced_rs_proxy.iloc[-1]:.4f}")
        return {"raw_rs_proxy": rs_proxy_raw, "enhanced_rs_proxy": enhanced_rs_proxy, "rs_hab_score": hab_score}

    def _calculate_enhanced_capital_proxy(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], config: Dict) -> Dict[str, pd.Series]:
        """
        【V5.2 · 动态量纲自适应版】增强版资本属性代理信号
        修改说明：废弃固定1e7缩放，改用滚动均值作为物理量纲基准。确保Capital-HAB在不同成交量级的个股中均能有效感知存量累积。
        版本号: 2026.02.10.321
        """
        # 1. 获取动态缩放基准 (13日均值)
        mf_series = mtf_signals.get('SLOPE_5_mf_net_amount', pd.Series(0.0, index=df_index))
        scaling_base = mf_series.abs().rolling(21, min_periods=5).mean().replace(0, 1e6)
        # 2. 运动学矢量执行自适应饱和压缩
        cap_slope = np.tanh(mf_series / scaling_base)
        cap_accel = np.tanh(mtf_signals.get('ACCEL_5_mf_net_amount', pd.Series(0.0, index=df_index)) / scaling_base)
        cap_jerk = np.tanh(mtf_signals.get('JERK_5_mf_net_amount', pd.Series(0.0, index=df_index)) / scaling_base)
        # 3. Capital-HAB 累积
        net_mf_norm = pd.Series(normalized_signals.get('net_mf_amount_norm', 0.5), index=df_index)
        hab_inc = (net_mf_norm.mask(net_mf_norm < 0.55, 0) * cap_accel.clip(lower=0))
        cap_hab_score = hab_inc.rolling(window=21, min_periods=5).sum().fillna(0).rolling(34).rank(pct=True).fillna(0.5)
        # 4. 最终合成
        kinematic_sum = (cap_slope * 0.4 + cap_accel * 0.4 + cap_jerk * 0.2).clip(0, 1)
        raw_proxy = (kinematic_sum * 0.4 + cap_hab_score * 0.4 + 0.2).clip(0, 1) # 给予固定效率补给
        enhanced_capital_proxy = (raw_proxy * 1.0).clip(0, 1)
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"[CAP_PROBE_V5.2] Raw_Slope: {mf_series.iloc[-1]:.0f} | Base: {scaling_base.iloc[-1]:.0f} | HAB: {cap_hab_score.iloc[-1]:.4f}")
        return {"raw_capital_proxy": raw_proxy, "enhanced_capital_proxy": enhanced_capital_proxy, "capital_hab_score": cap_hab_score}

    def _calculate_enhanced_sentiment_proxy(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], config: Dict) -> Dict[str, pd.Series]:
        """
        【V5.0 · 情绪运动学与温压存量版】增强版情绪属性代理信号
        修改说明：引入情绪S/A/J矢量与Sentiment-HAB历史累积缓冲，量化市场热度的渗透性与持久力。
        版本号：2026.02.10.81
        """
        # 1. 提取情绪运动学矢量
        sent_slope = mtf_signals.get('SLOPE_5_market_sentiment_trend', pd.Series(0.0, index=df_index))
        sent_accel = mtf_signals.get('ACCEL_5_market_sentiment_trend', pd.Series(0.0, index=df_index))
        sent_jerk = mtf_signals.get('JERK_5_market_sentiment_trend', pd.Series(0.0, index=df_index))
        # 2. 情绪温压累积记忆缓冲 (Sentiment-HAB)
        # 逻辑：利用市场情绪与行业宽度的乘积进行累积。存量越高，说明板块效应越强，容错率越高。
        market_sent = normalized_signals.get('market_sentiment_norm', pd.Series(0.5, index=df_index))
        ind_breadth = normalized_signals.get('industry_breadth_norm', pd.Series(0.5, index=df_index))
        hab_window = get_param_value(config.get('sentiment_hab_window'), 13)
        hab_increment = (market_sent * ind_breadth * (1 + sent_accel))
        sent_hab_buffer = hab_increment.rolling(window=hab_window, min_periods=3).sum().fillna(0)
        sent_hab_score = normalize_score(sent_hab_buffer, target_index=df_index, windows=21)
        # 3. 情绪一致性 (Leader-Sentiment Coherence)
        leader_strength = normalized_signals.get('industry_leader_norm', pd.Series(0.5, index=df_index))
        sent_consistency = (market_sent * leader_strength).rolling(5).mean().fillna(0.5)
        # 4. 综合合成
        kinematic_sum = (sent_slope * 0.3 + sent_accel * 0.5 + sent_jerk * 0.2).clip(0, 1)
        raw_proxy = (kinematic_sum * 0.3 + sent_hab_score * 0.4 + sent_consistency * 0.3).clip(0, 1)
        # 5. 情境调节
        sentiment_modulator = 1.0 + (sent_hab_score - 0.5) * 0.15
        enhanced_sentiment_proxy = (raw_proxy * sentiment_modulator).clip(0, 1)
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"--- Sentiment Heat-HAB Probe ---")
            self._probe_print(f"  > Sent SAJ: S={sent_slope.iloc[-1]:.4f}, A={sent_accel.iloc[-1]:.4f}, J={sent_jerk.iloc[-1]:.4f}")
            self._probe_print(f"  > Sentiment-HAB Level: {sent_hab_score.iloc[-1]:.4f}")
        return {"raw_sentiment_proxy": raw_proxy, "enhanced_sentiment_proxy": enhanced_sentiment_proxy, "sentiment_hab_score": sent_hab_score, "sentiment_modulator": sentiment_modulator}

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
        【V6.0 · 对数压缩流动性版】增强版流动性代理信号
        修改说明：采用对数处理(log1p)抑制极端成交量波动，结合双曲正切映射实现流动性潮汐的平滑归一化。
        版本号：2026.02.10.111
        """
        vol_norm = normalized_signals.get('volume_ratio_norm', pd.Series(0.5, index=df_index))
        turnover_norm = normalized_signals.get('turnover_rate_norm', pd.Series(0.5, index=df_index))
        liq_tide = (vol_norm * 0.6 + turnover_norm * 0.4)
        tide_slope = liq_tide.diff(3).rolling(5).mean().fillna(0)
        tide_accel = tide_slope.diff(1).fillna(0)
        price_slope_abs = mtf_signals.get('SLOPE_5_price_trend', pd.Series(0.0, index=df_index)).abs()
        liq_hab_inc = (tide_accel.clip(lower=0) * (1 - price_slope_abs.clip(0, 1)))
        # 定制化归一化：对数压缩后的滚动 Min-Max
        log_inc = np.log1p(liq_hab_inc.rolling(21).sum().fillna(0))
        liq_hab_score = (log_inc - log_inc.rolling(55).min()) / (log_inc.rolling(55).max() - log_inc.rolling(55).min() + 1e-9)
        liq_proxy_raw = (liq_tide * 0.4 + tide_accel.clip(0, 1) * 0.3 + liq_hab_score * 0.3).clip(0, 1)
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"[LIQ_PROBE] Tide_Accel: {tide_accel.iloc[-1]:.4f}, Log_HAB: {liq_hab_score.iloc[-1]:.4f}")
        return {"enhanced_liquidity_proxy": liq_proxy_raw, "liquidity_hab_score": liq_hab_score}

    def _calculate_enhanced_volatility_proxy(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], config: Dict) -> Dict[str, pd.Series]:
        """
        【V6.0 · 倒数映射低波动版】增强版波动性代理信号
        修改说明：采用物理量纲倒数映射处理波动率稳定性，弃用线性归一化，精准捕捉“横盘蓄势”期的低波动存量特征。
        版本号：2026.02.10.112
        """
        pct_change = normalized_signals.get('pct_change_norm', pd.Series(0.0, index=df_index))
        hist_vol = pct_change.rolling(21).std().fillna(0.02)
        vol_slope = hist_vol.diff(3).rolling(5).mean().fillna(0)
        vol_jerk = vol_slope.diff(2).fillna(0)
        # 定制化归一化：倒数映射 (Volatility Clustering Index)
        # 逻辑：当波动率低于 21日均值时，得分呈指数级提升
        avg_vol = hist_vol.rolling(60).mean().replace(0, 0.02)
        vol_hab_raw = (avg_vol / (hist_vol + 1e-9)).rolling(21).sum().fillna(0)
        vol_hab_score = (np.tanh(vol_hab_raw / 10) * 0.5 + 0.5).clip(0, 1)
        vol_weights = {"historical": 0.3, "jerk": 0.3, "hab": 0.4}
        # hist_vol 是正向指标（波动高分高），hab 是负向（波动低分高）
        vol_proxy = (hist_vol.rolling(60).rank(pct=True) * vol_weights["historical"] + np.tanh(vol_jerk * 10).abs() * vol_weights["jerk"] + (1 - vol_hab_score) * vol_weights["hab"]).clip(0, 1)
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"[VOL_PROBE] Jerk: {vol_jerk.iloc[-1]:.4f}, Low_Vol_HAB: {vol_hab_score.iloc[-1]:.4f}")
        return {"enhanced_volatility_proxy": vol_proxy, "volatility_hab_score": vol_hab_score}

    def _calculate_enhanced_risk_preference_proxy(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], config: Dict) -> Dict[str, pd.Series]:
        """
        【V6.0 · 非线性风险偏好版】增强版风险偏好代理信号
        修改说明：采用Sigmoid逻辑回归函数对风险资产表现进行非线性增强，锐化资金在防御与进攻状态切换时的边界信号。
        版本号：2026.02.10.113
        """
        # 假设 asset_perf 由行业强弱和连板家数等因子组成
        leader_strength = normalized_signals.get('industry_leader_norm', pd.Series(0.5, index=df_index))
        market_sent = normalized_signals.get('market_sentiment_norm', pd.Series(0.5, index=df_index))
        asset_perf = (leader_strength * 0.6 + market_sent * 0.4).clip(0, 1)
        # 定制化归一化：Sigmoid 非线性映射 (中心点0.5，斜率8)
        risk_perf_norm = 1 / (1 + np.exp(-8 * (asset_perf - 0.5)))
        risk_slope = risk_perf_norm.diff(3).rolling(5).mean().fillna(0)
        risk_accel = risk_slope.diff(1).fillna(0)
        # HAB 存量归一化：Rolling Rank (55日窗口)
        hab_raw = risk_perf_norm.rolling(window=21).sum().fillna(0)
        risk_hab_score = hab_raw.rolling(55).rank(pct=True).fillna(0.5)
        risk_proxy = (risk_perf_norm * 0.4 + np.tanh(risk_accel * 5).clip(lower=0) * 0.3 + risk_hab_score * 0.3).clip(0, 1)
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"[RISK_PROBE] Sigmoid_Perf: {risk_perf_norm.iloc[-1]:.4f}, Risk_HAB: {risk_hab_score.iloc[-1]:.4f}")
        return {"enhanced_risk_preference_proxy": risk_proxy, "risk_hab_score": risk_hab_score}

    def _dynamic_weighted_synthesis(self, rs_proxy: Dict, capital_proxy: Dict, sentiment_proxy: Dict, liquidity_proxy: Dict, volatility_proxy: Dict, risk_preference_proxy: Dict, signal_quality: pd.Series, config: Dict) -> Dict[str, pd.Series]:
        """
        【V5.0 · Softmax质量响应版】动态权重合成算法
        修改说明：内部实现带温度控制的Softmax分配，使权重随信号质量动态聚焦。
        版本号：2026.02.10.143
        """
        df_index = signal_quality.index
        sigs = {
            "rs": rs_proxy.get("enhanced_rs_proxy", pd.Series(0.5, index=df_index)),
            "capital": capital_proxy.get("enhanced_capital_proxy", pd.Series(0.5, index=df_index)),
            "sentiment": sentiment_proxy.get("enhanced_sentiment_proxy", pd.Series(0.5, index=df_index)),
            "liquidity": liquidity_proxy.get("enhanced_liquidity_proxy", pd.Series(0.5, index=df_index)),
            "volatility": volatility_proxy.get("enhanced_volatility_proxy", pd.Series(0.5, index=df_index)),
            "risk": risk_preference_proxy.get("enhanced_risk_preference_proxy", pd.Series(0.5, index=df_index))
        }
        # 1. 定义基础分值 (Logit Space)
        base_scores = {"rs": 1.2, "capital": 1.0, "sentiment": 0.6, "liquidity": 0.4, "volatility": 0.4, "risk": 0.4}
        # 2. 计算温度系数：质量越高，温度越低 (1.5 -> 0.2)
        temperature = 1.5 - (signal_quality * 1.3)
        # 3. Softmax 合成
        final_series = pd.Series(0.0, index=df_index)
        weight_recorder = {k: pd.Series(0.0, index=df_index) for k in sigs}
        for i in range(len(df_index)):
            temp = temperature.iloc[i]
            exp_vals = {k: np.exp(v / temp) for k, v in base_scores.items()}
            total_exp = sum(exp_vals.values())
            weights = {k: v / total_exp for k, v in exp_vals.items()}
            final_val = sum(sigs[k].iloc[i] * weights[k] for k in sigs)
            final_series.iloc[i] = final_val
            for k in weights: weight_recorder[k].iloc[i] = weights[k]
        result = {"comprehensive_proxy": final_series.clip(0, 1)}
        result.update({f"{k}_weight": weight_recorder[k] for k in weight_recorder})
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"[WEIGHT_PROBE] Temp: {temperature.iloc[-1]:.2f}, RS_Weight: {weight_recorder['rs'].iloc[-1]:.4f}")
        return result

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
        【V5.0 · Tanh斜率归一版】计算序列趋势方向
        修改说明：采用tanh非线性映射取代线性归一化，确保趋势斜率在A股极端波动下保持稳定的量纲输出。
        版本号：2026.02.10.133
        """
        if series.empty or len(series) < 5: return 0.0
        try:
            x = np.arange(len(series))
            y = series.values
            A = np.vstack([x, np.ones(len(x))]).T
            slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
            # 定制化归一化：tanh 映射，设 0.02 (2%) 为强趋势饱和阈值
            normalized_slope = np.tanh(slope / 0.02)
            return float(normalized_slope)
        except: return 0.0

    def _weighted_geometric_mean(self, components: Dict[str, pd.Series], weights: Dict[str, float], df_index: pd.Index) -> pd.Series:
        """
        【V5.0 · 自包含几何平均版】加权几何平均
        修改说明：内部实现带精度保护的对数加权合成算法，废弃外部utils依赖，确保多因子合成的一致性。
        版本号：2026.02.10.134
        """
        log_sum = pd.Series(0.0, index=df_index)
        total_w = 0.0
        for name, weight in weights.items():
            if name in components:
                # 局部精度保护：避免ln(0)
                log_sum += weight * np.log(components[name].clip(lower=1e-7))
                total_w += weight
        res = np.exp(log_sum / (total_w + 1e-9))
        return res.clip(0, 1)

    def _nonlinear_synthesis(self, components: Dict[str, pd.Series], weights: Dict[str, float], df_index: pd.Index) -> pd.Series:
        """
        【V5.0 · 自包含Sigmoid合成版】非线性加权合成
        修改说明：内部实现带陡峭度控制的Sigmoid归一化合成，废弃通用helper方法，强化极端分值的权重贡献。
        版本号：2026.02.10.135
        """
        sum_val = pd.Series(0.0, index=df_index)
        total_w = 0.0
        for name, weight in weights.items():
            if name in components:
                # 局部定制：Sigmoid 映射 (k=10, x0=0.5)
                sig_val = 1 / (1 + np.exp(-10 * (components[name] - 0.5)))
                sum_val += weight * sig_val
                total_w += weight
        return (sum_val / (total_w + 1e-9)).clip(0, 1)

    def _calculate_dynamic_weights(self, df_index: pd.Index, normalized_signals: Dict[str, pd.Series], proxy_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], factors: Dict[str, pd.Series], phase: pd.Series) -> Dict[str, pd.Series]:
        """
        【V6.3 · 探针类型修复版】动态权重计算系统
        修改说明：修复探针打印时final_temp为标量导致的AttributeError。优化Softmax温度控制的数值稳定性。
        版本号：2026.02.10.376
        """
        # 1. 物理相位基准权重
        base_weights_raw = self._calculate_base_weights(df_index, phase, factors)
        # 2. 运动学增益
        rs_acc = pd.Series(mtf_signals.get('ACCEL_5_price_trend', 0.0), index=df_index).abs()
        # 3. 提取一致性爆发与综合质量
        quality = proxy_signals.get('combined_signal_quality', factors.get('state_hab_score', pd.Series(0.7, index=df_index)))
        burst_score = proxy_signals.get('sync_burst_score', pd.Series(0.0, index=df_index))
        
        final_weights_unnorm = {}
        for dim, b_w in base_weights_raw.items():
            boost = (rs_acc * 0.6) if dim == 'aggressiveness' else 0
            final_weights_unnorm[dim] = (b_w * (1 + boost)).ewm(span=13).mean()

        # 4. 带确定性溢价的 Softmax 归一化
        final_weights = {dim: pd.Series(0.0, index=df_index) for dim in final_weights_unnorm.keys()}
        final_temp = 1.0 # 初始化
        
        for i in range(len(df_index)):
            raw_vals = np.array([final_weights_unnorm[d].iloc[i] for d in final_weights_unnorm])
            # 基础温度 (0.5 ~ 2.0)
            base_temp = 2.0 - (quality.iloc[i] * 1.5)
            # 确定性溢价：Burst越高，温度越低
            final_temp = base_temp / (1 + burst_score.iloc[i] * 0.8 + 1e-9)
            
            # 使用 np.clip 确保兼容性
            temp_clamped = np.clip(final_temp, 0.1, 2.0)
            exp_vals = np.exp(raw_vals / temp_clamped)
            
            norm_vals = exp_vals / (np.sum(exp_vals) + 1e-9)
            for idx, dim in enumerate(final_weights_unnorm.keys()):
                final_weights[dim].iloc[i] = norm_vals[idx]
        
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            # 修复：final_temp 为标量，直接打印
            self._probe_print(f"[WEIGHT_BURST_PROBE] Burst: {burst_score.iloc[-1]:.4f} | Final_Temp: {final_temp:.4f}")
        return final_weights

    def _calculate_market_state_factors(self, df_index: pd.Index, normalized_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V5.1 · 类型安全与惯性存量版】七维度市场状态因子计算
        修改说明：修复'float' object has no attribute 'clip'报错，强制转换输入分量为Series。引入State-Inertia-HAB量化环境惯性。
        版本号：2026.02.10.190
        """
        factors = {}
        # 1. 基础因子合成 (类型安全加固：显式转换为Series)
        trend_comp = {
            'mtf_price_trend': pd.Series(mtf_signals.get('mtf_price_trend', 0.5), index=df_index),
            'uptrend_strength': pd.Series(normalized_signals.get('uptrend_strength_norm', 0.5), index=df_index)
        }
        factors['trend_state'] = self._weighted_geometric_mean(trend_comp, {'mtf_price_trend': 0.5, 'uptrend_strength': 0.5}, df_index)
        cap_comp = {
            'net_mf': pd.Series(normalized_signals.get('net_mf_amount_norm', 0.5), index=df_index),
            'flow_accel': pd.Series(normalized_signals.get('flow_acceleration_norm', 0.5), index=df_index)
        }
        factors['capital_state'] = self._weighted_geometric_mean(cap_comp, {'net_mf': 0.6, 'flow_accel': 0.4}, df_index)
        # 2. 环境运动学：识别风向切换
        factors['trend_state_slope'] = factors['trend_state'].diff(1).rolling(5).mean().fillna(0)
        factors['trend_state_accel'] = factors['trend_state_slope'].diff(1).fillna(0)
        # 3. 环境惯性存量 (State-Inertia-HAB)
        hab_window = 21
        state_health = (factors['trend_state'] * factors['capital_state']).clip(0, 1)
        hab_inc = state_health.mask(state_health < 0.6, 0) * (1 + factors['trend_state_accel'].clip(lower=0))
        # 内部定制归一化：Rolling Rank (34日窗口)
        factors['state_hab_score'] = hab_inc.rolling(window=hab_window).sum().fillna(0).rolling(34).rank(pct=True).fillna(0.5)
        # 4. 其他基础状态
        factors['volatility_state'] = (1 - pd.Series(normalized_signals.get('ATR_norm', 0.5), index=df_index)).clip(0, 1)
        factors['sentiment_state'] = pd.Series(normalized_signals.get('market_sentiment_norm', 0.5), index=df_index).clip(0, 1)
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"--- Market State Type-Safe Probe ---")
            self._probe_print(f"  > State Inertia HAB: {factors['state_hab_score'].iloc[-1]:.4f}")
        return factors

    def _identify_market_phase(self, df_index: pd.Index, market_state_factors: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series]) -> pd.Series:
        """
        【V5.3 · 零空值投机韧性版】市场阶段识别算法
        修改说明：修复prob_dist_early产生的nan问题。为所有滚动水位计算增加min_periods=1保护，确保冷启动阶段概率链条的完整性。
        版本号：2026.02.10.300
        """
        # 1. 提取分量并强制非空
        p_hab = pd.Series(market_state_factors.get('state_hab_score', 0.5), index=df_index).fillna(0.5)
        # 核心修复：增加 min_periods=1 防止滚动窗口初期返回 nan
        ch_hab = pd.Series(normalized_signals.get('chip_stability_norm', 0.5), index=df_index).rolling(34, min_periods=1).mean().fillna(0.5)
        c_hab = pd.Series(normalized_signals.get('net_mf_amount_norm', 0.5), index=df_index).rolling(21, min_periods=1).rank(pct=True).fillna(0.5)
        p_acc = pd.Series(market_state_factors.get('trend_state_accel', 0.0), index=df_index).fillna(0.0)
        c_accel = pd.Series(normalized_signals.get('flow_acceleration_norm', 0.5), index=df_index).fillna(0.5)
        # 2. 概率合成
        prob_acc = (c_hab * (1 - p_hab)).clip(0, 1)
        prob_exp = (p_hab * c_hab * (p_acc.clip(lower=0) + 0.5)).clip(0, 1)
        # 3. 派发分层 (增加 fillna 保护)
        prob_dist_early = (p_hab.mask(p_hab < 0.8, 0) * ch_hab.mask(ch_hab < 0.7, 0) * (0.6 - c_accel).clip(0, 1)).fillna(0)
        prob_dist_late = (p_hab * (1 - ch_hab) * (1 - c_hab)).clip(0, 1)
        p_jerk = p_acc.diff(1).fillna(0)
        prob_panic = (p_jerk.clip(upper=0).abs() * p_hab).clip(0, 1)
        # 4. 矩阵决策与平滑
        phase_matrix = pd.DataFrame({"蓄势": prob_acc, "主升": prob_exp, "派发初期": prob_dist_early, "派发末端": prob_dist_late, "反转风险": prob_panic, "横盘": 0.2}, index=df_index)
        raw_phases = phase_matrix.idxmax(axis=1)
        phase_map = {"蓄势": 0, "主升": 1, "派发初期": 2, "派发末端": 3, "反转风险": 4, "横盘": 5}
        inv_map = {v: k for k, v in phase_map.items()}
        raw_int = raw_phases.map(phase_map).astype(np.float64)
        smoothed_phases = raw_int.rolling(window=5, min_periods=1).apply(lambda x: x.mode()[0] if not x.mode().empty else x.iloc[-1]).map(inv_map).fillna("横盘")
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"--- Speculative Phase Probe V5.3 ---")
            self._probe_print(f"  > Prob Dist Early: {prob_dist_early.iloc[-1]:.4f} | Final Phase: {smoothed_phases.iloc[-1]}")
        return smoothed_phases

    def _calculate_base_weights(self, df_index: pd.Index, market_phase: pd.Series, market_state_factors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V5.1 · 投机溢价版】基础权重计算
        修改说明：调整派发初期的权重分配，向攻击性(Agg)倾斜，识别赶顶阶段的纯动能价值。
        版本号：2026.02.10.262
        """
        p_acc = pd.Series(market_state_factors.get('trend_state_accel', 0.0), index=df_index)
        s_hab = pd.Series(market_state_factors.get('state_hab_score', 0.5), index=df_index)
        # 扩展相位配置：增加 派发初期 与 派发末端
        phase_config = {
            '启动': np.array([0.35, 0.30, 0.25, 0.10]),
            '主升': np.array([0.45, 0.20, 0.25, 0.10]),
            '派发初期': np.array([0.55, 0.10, 0.15, 0.20]), # 攻击性最强，控制力剥离
            '派发末端': np.array([0.05, 0.10, 0.10, 0.75]), # 风险完全主导
            '反转风险': np.array([0.10, 0.25, 0.15, 0.50]),
            '蓄势': np.array([0.25, 0.35, 0.25, 0.15]),
            '横盘': np.array([0.25, 0.35, 0.20, 0.20])
        }
        w_agg = pd.Series(0.0, index=df_index); w_ctrl = pd.Series(0.0, index=df_index)
        w_obs = pd.Series(0.0, index=df_index); w_risk = pd.Series(0.0, index=df_index)
        for i in range(len(df_index)):
            phase = market_phase.iloc[i]
            base = phase_config.get(phase, phase_config['横盘']).copy()
            acc_val = p_acc.iloc[i]; hab_val = s_hab.iloc[i]
            if acc_val > 0: base[0] *= (1 + acc_val * 0.5)
            else: base[3] *= (1 + abs(acc_val) * 0.8)
            base[0] *= (0.9 + hab_val * 0.2); base[1] *= (0.9 + hab_val * 0.2); base[3] *= (1.1 - hab_val * 0.2)
            norm_w = base / (np.sum(base) + 1e-9)
            w_agg.iloc[i], w_ctrl.iloc[i], w_obs.iloc[i], w_risk.iloc[i] = norm_w
        return {'aggressiveness': w_agg.clip(0.05, 0.7), 'control': w_ctrl.clip(0.05, 0.6), 'obstacle_clearance': w_obs.clip(0.05, 0.6), 'risk': w_risk.clip(0.05, 0.8)}

    def _calculate_aggressiveness_score(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], dynamic_weights: Dict[str, pd.Series]) -> pd.Series:
        """
        【V8.0 · 投机动能与连板惯性版】计算攻击性分值
        修改说明：引入情绪加速度(Sentiment-Accel)与封板一致性(Limit-Up HAB)，量化派发初期的超预期投机意图。
        版本号：2026.02.10.270
        """
        # 1. 提取基础动能分量
        p_trend = pd.Series(mtf_signals.get('mtf_price_trend', 0.0), index=df_index).clip(lower=0)
        v_trend = pd.Series(mtf_signals.get('mtf_volume_trend', 0.0), index=df_index).clip(lower=0)
        # 2. 引入情绪加速度 (Sentiment-Accel)
        s_accel = pd.Series(mtf_signals.get('ACCEL_5_market_sentiment_trend', 0.0), index=df_index).clip(lower=0)
        # 3. 构建封板一致性存量 (Limit-Up Consistency HAB)
        # 逻辑：识别13日内的连板习惯，每次涨停且加速度为正时，累积攻击性存量
        close = pd.Series(raw_signals.get('close', 0.0), index=df_index)
        up_limit = pd.Series(raw_signals.get('up_limit', 0.0), index=df_index)
        is_lu = (close >= up_limit) & (up_limit > 0)
        p_acc = pd.Series(mtf_signals.get('ACCEL_5_price_trend', 0.0), index=df_index).clip(lower=0)
        lu_hab_inc = is_lu.astype(float) * (1 + p_acc)
        lu_hab_buffer = lu_hab_inc.rolling(window=13, min_periods=1).sum().fillna(0)
        # 定制化归一化：Rolling Rank (34日排位)
        lu_consistency_hab = lu_hab_buffer.rolling(34).rank(pct=True).fillna(0.5)
        # 4. 几何平均合成 (4维度共振)
        # 权重分配：价格(30%) + 成交量(20%) + 情绪爆发(30%) + 连板惯性(20%)
        agg_raw = (p_trend ** 0.3 * v_trend ** 0.2 * (1 + s_accel) ** 0.3 * (1 + lu_consistency_hab) ** 0.2)
        # 5. 定制化归一化：90日超长窗口分位数排位
        agg_rank = agg_raw.rolling(window=90, min_periods=20).rank(pct=True).fillna(0.5)
        # 6. Jerk 冲击增强 (捕捉物理力量的瞬间阶跃)
        p_jerk = pd.Series(mtf_signals.get('JERK_5_price_trend', 0.0), index=df_index).abs()
        j_norm = np.tanh(p_jerk / p_jerk.rolling(60).mean().replace(0, 1e-9)).clip(0, 1)
        final_score = (agg_rank * (1 + j_norm * 0.2)).clip(0, 1)
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"--- Aggressiveness Speculative Probe ---")
            self._probe_print(f"  > Sent_Accel: {s_accel.iloc[-1]:.4f} | LU_Consistency: {lu_consistency_hab.iloc[-1]:.4f}")
            self._probe_print(f"  > Final Agg Score: {final_score.iloc[-1]:.4f}")
        return final_score.astype(np.float32)

    def _detect_bear_trap(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], obstacle_clearance_score: pd.Series) -> pd.Series:
        """
        【V7.0 · PAR-Jerk 诱空识别版】诱空陷阱检测
        修改说明：引入PAR-HAB（障碍清除分数）的阶跃与Price-Jerk的正向偏移。判别横盘回踩是否为“暴力洗盘”，捕捉启动前的最后物理拐点。
        版本号：2026.02.10.350
        """
        # 1. 提取关键物理矢量
        p_acc = pd.Series(mtf_signals.get('ACCEL_5_price_trend', 0.0), index=df_index)
        p_jerk = pd.Series(mtf_signals.get('JERK_5_price_trend', 0.0), index=df_index)
        # 2. 计算 PAR-HAB 阶跃 (Step Change)
        # 逻辑：当障碍清除分值（代表吸收强度）在急跌中跳升，意味着主力在特定价位“接盘”
        par_step = obstacle_clearance_score.diff(1).clip(lower=0)
        # 3. 识别反转冲量 (Jerk Offset)
        # 物理逻辑：加速度为负（下跌中）且加加速度为正（正在极速减速或受力反弹）
        # trap_impulse = 下跌惯性绝对值 * 向上反作用力 * 吸收阶跃
        trap_impulse = p_acc.clip(upper=0).abs() * p_jerk.clip(lower=0) * (1 + par_step * 2)
        # 4. 定制化归一化：Tanh 映射
        # 阈值设为 0.0015，对应 A 股常见的缩量诱空反转强度
        trap_intensity = np.tanh(trap_impulse / 0.0015).clip(0, 1)
        # 5. 结合障碍清除存量背书
        # 只有在长期障碍清理水位(HAB-Rank)较高时，局部的“陷阱”才具备确定性
        hab_backing = obstacle_clearance_score.rolling(window=13).mean()
        final_trap_score = (trap_intensity * hab_backing).fillna(0)
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"--- PAR-Jerk Bear Trap Probe ---")
            self._probe_print(f"  > PAR_Step: {par_step.iloc[-1]:.4f} | Jerk_Offset: {p_jerk.iloc[-1]:.4f}")
            self._probe_print(f"  > Trap_Impulse: {trap_impulse.iloc[-1]:.6f} | Final_Score: {final_trap_score.iloc[-1]:.4f}")
            
        return final_trap_score.astype(np.float32)

    def _detect_bull_trap(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], dist_hab: pd.Series) -> pd.Series:
        """
        【V1.0 · 物理受力反转版】诱多陷阱检测
        修改说明：Bear Trap逻辑的物理反转。识别上涨惯性中派发力量的阶跃与价格Jerk的负向偏移。
        版本号：2026.02.10.360
        """
        # 1. 提取物理矢量
        p_acc = pd.Series(mtf_signals.get('ACCEL_5_price_trend', 0.0), index=df_index)
        p_jerk = pd.Series(mtf_signals.get('JERK_5_price_trend', 0.0), index=df_index)
        dist_score = pd.Series(normalized_signals.get('distribution_score_norm', 0.0), index=df_index)
        # 2. 计算派发阶跃 (Dist Step)
        dist_step = dist_score.diff(1).clip(lower=0)
        # 3. 识别诱多冲量 (Bull Trap Impulse)
        # 物理逻辑：处于上涨惯性(Acc > 0)但受力方向已变(Jerk < 0)，且伴随派发增强
        bull_impulse = p_acc.clip(lower=0) * p_jerk.clip(upper=0).abs() * (1 + dist_step * 2)
        # 4. 定制化归一化
        # 阈值设为 0.0020，捕捉A股典型的高位缩量诱多
        trap_intensity = np.tanh(bull_impulse / 0.0020).clip(0, 1)
        # 5. 结合派发存量背书 (Dist-HAB)
        # 诱多陷阱必须建立在主力已有充分出货存量的前提下
        final_trap_score = (trap_intensity * dist_hab).fillna(0)
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"--- Bull Trap Physics Probe ---")
            self._probe_print(f"  > Dist_Step: {dist_step.iloc[-1]:.4f} | Neg_Jerk: {p_jerk.iloc[-1]:.4f}")
            self._probe_print(f"  > Bull_Impulse: {bull_impulse.iloc[-1]:.6f} | Trap_Score: {final_trap_score.iloc[-1]:.4f}")
            
        return final_trap_score.astype(np.float32)

    def _calculate_volume_price_divergence(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series]) -> pd.Series:
        """
        【V6.0 · 局部排位背离版】缩量背离侦测
        修改说明：采用13日局部滚动排位对比价量斜率，通过排位差值量化“虚假拉升”的严重程度。
        版本号：2026.02.10.122
        """
        p_slope = mtf_signals.get('SLOPE_5_price_trend', pd.Series(0.0, index=df_index))
        v_slope = mtf_signals.get('SLOPE_5_volume_trend', pd.Series(0.0, index=df_index))
        # 1. 计算价格和成交量的局部排位
        p_rank = p_slope.rolling(13).rank(pct=True).fillna(0.5)
        v_rank = v_slope.rolling(13).rank(pct=True).fillna(0.5)
        # 2. 背离判别：价格高排位 (上涨) 且 成交量低排位 (缩量)
        divergence_gap = (p_rank - v_rank).clip(lower=0)
        # 3. 定制归一化：通过平方项放大极端背离
        div_score = (divergence_gap ** 2).mask(p_slope <= 0, 0)
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"[DIV_PROBE] Price_Rank: {p_rank.iloc[-1]:.2f}, Vol_Rank: {v_rank.iloc[-1]:.2f}, Gap: {divergence_gap.iloc[-1]:.4f}")
        return div_score.clip(0, 1)

    def _calculate_acceleration_resonance(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series]) -> pd.Series:
        """
        【V6.0 · 稳健Z轴共振版】多周期加速度共振检测
        修改说明：采用台阶映射与稳健标准化，废弃线性归一化，确保共振信号在不同量纲的加速度下保持输出强度的一致性。
        版本号：2026.02.10.124
        """
        a5 = mtf_signals.get('ACCEL_5_price_trend', 0)
        a13 = mtf_signals.get('ACCEL_13_price_trend', 0)
        a21 = mtf_signals.get('ACCEL_21_price_trend', 0)
        # 1. 计算方向共振级别
        res_count = (a5 > 0).astype(int) + (a13 > 0).astype(int) + (a21 > 0).astype(int)
        res_base = res_count.map({3: 1.0, 2: 0.5, 1: 0.1, 0: 0.0})
        # 2. 定制归一化：稳健 Z-Score 处理平均加速度
        avg_a = (a5.clip(lower=0) + a13.clip(lower=0) + a21.clip(lower=0)) / 3.0
        med = avg_a.rolling(60).median()
        mad = (avg_a - med).abs().rolling(60).median()
        a_norm = np.tanh((avg_a - med) / (mad * 1.4826 + 1e-9)).clip(0, 1)
        # 3. 最终共振强度
        resonance_score = (res_base * (1 + a_norm * 0.3)).clip(0, 1)
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"[RESONANCE_PROBE] Count: {res_count.iloc[-1]}, Acc_Norm: {a_norm.iloc[-1]:.4f}")
        return resonance_score

    def _calculate_control_score(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], historical_context: Dict[str, Any]) -> pd.Series:
        """
        【V5.2 · 筹码熵变动力学版】计算控制力分数
        修改说明：利用筹码熵的一阶导数识别缩量筹码搬家。熵减斜率与成交量反向耦合，量化高位横盘时的“真实锁仓”强度。
        版本号：2026.02.10.330
        """
        # 1. 提取筹码熵与稳定性
        entropy = pd.Series(raw_signals.get('chip_entropy', 0.5), index=df_index)
        chip_stab = pd.Series(raw_signals.get('chip_stability', 0.5), index=df_index)
        vol_ratio = pd.Series(raw_signals.get('volume_ratio', 1.0), index=df_index)
        # 2. 计算熵变斜率 (Entropy Slope) - 熵减代表集中
        entropy_slope = entropy.diff(3).rolling(5).mean().fillna(0)
        # 3. 计算“缩量搬家效率” (Migration Efficiency)
        # 逻辑：熵减强度越大(负值绝对值大)且成交量越低，控盘效率越高
        # 使用tanh映射，中心点设为 -0.01 (代表3日熵减1%)
        concentration_impulse = np.tanh(-entropy_slope / 0.01).clip(lower=0)
        # 搬家效率 = 集中冲量 / (成交量比率 + 修正项)
        migration_efficiency = (concentration_impulse / (vol_ratio + 0.1)).clip(0, 2)
        migration_norm = np.tanh(migration_efficiency).clip(0, 1)
        # 4. 构建 Control-Entropy-HAB (控盘存量)
        # 逻辑：将搬家效率与筹码稳定性进行耦合累积
        hab_win = 21
        hab_inc = migration_norm * chip_stab
        hab_buffer = hab_inc.rolling(window=hab_win, min_periods=5).sum().fillna(0)
        # 定制化归一化：Rolling Rank (55日分位数排位)
        hab_score = hab_buffer.rolling(55).rank(pct=True).fillna(0.5)
        # 5. 判别“最后的疯狂” vs “高位良换手”
        # 识别特征：高位横盘(P-HAB高)且熵增(Slope > 0) = 疯狂/派发
        p_hab = pd.Series(normalized_signals.get('absolute_change_strength_norm', 0.5), index=df_index).rolling(13).mean()
        frenzy_risk = (p_hab.mask(p_hab < 0.7, 0) * np.tanh(entropy_slope.clip(lower=0) / 0.01)).clip(0, 1)
        # 6. 最终合成
        # 控盘得分 = (瞬时集中度 + 存量水位) * (1 - 疯狂风险)
        base_control = (migration_norm * 0.4 + hab_score * 0.6).clip(0, 1)
        final_control = (base_control * (1 - frenzy_risk * 0.5)).clip(0, 1)
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"--- Chip Migration Control Probe ---")
            self._probe_print(f"  > Entropy_Slope: {entropy_slope.iloc[-1]:.6f} | Mig_Eff: {migration_norm.iloc[-1]:.4f}")
            self._probe_print(f"  > Control_HAB_Rank: {hab_score.iloc[-1]:.4f} | Frenzy_Risk: {frenzy_risk.iloc[-1]:.4f}")
        return final_control.astype(np.float32)

    def _calculate_obstacle_clearance_score(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], historical_context: Dict[str, Any]) -> pd.Series:
        """
        【V7.0 · 抛压吸收动力学版】计算障碍清除分数
        修改说明：引入PAR(抛压吸收比)逻辑，利用物理Jerk识别支撑位瞬间的吸筹强度。通过HAB存量化抛压消化进度，判别高位横盘的健康度。
        版本号：2026.02.10.340
        """
        # 1. 提取物理分量与支撑参考
        p_acc = pd.Series(mtf_signals.get('ACCEL_5_price_trend', 0.0), index=df_index).fillna(0.0)
        p_jerk = pd.Series(mtf_signals.get('JERK_5_price_trend', 0.0), index=df_index).fillna(0.0)
        v_ratio = pd.Series(raw_signals.get('volume_ratio', 1.0), index=df_index).replace(0, 1e-9)
        chip_stab = pd.Series(raw_signals.get('chip_stability', 0.5), index=df_index)
        # 2. 计算抛压吸收比 (Pressure Absorption Ratio)
        # 逻辑：利用价格力的三阶导数(Jerk)来衡量对抛压的“弹性反馈”。
        # 若在低成交量下产生极高的向上Jerk，说明该价位已无抛压。
        par_raw = (p_jerk.clip(lower=0) / v_ratio)
        # 定制归一化：基于MAD的稳健tanh映射
        par_med = par_raw.rolling(60, min_periods=5).median()
        par_mad = (par_raw - par_med).abs().rolling(60, min_periods=5).median()
        par_norm = np.tanh(par_raw / (par_mad * 4.4478 + 1e-9)).clip(0, 1)
        # 3. 识别回踩支撑效应 (Support Impact)
        # 逻辑：价格处于低斜率区间（横盘或微跌），但Jerk呈阶跃式爆发
        p_slope_abs = pd.Series(mtf_signals.get('SLOPE_5_price_trend', 0.0), index=df_index).abs()
        support_rebound_intensity = (par_norm * (1 - np.tanh(p_slope_abs / 0.02))).clip(0, 1)
        # 4. 构建 PAR-HAB (抛压吸收存量)
        # 逻辑：将瞬时吸收强度与筹码稳定性耦合，记录“障碍清理”的累计进度
        hab_win = 34
        hab_inc = support_rebound_intensity * chip_stab * (1 + p_acc.clip(lower=0))
        hab_buffer = hab_inc.rolling(window=hab_win, min_periods=5).sum().fillna(0)
        # 5. 定制化归一化：Rolling Rank (89日长周期排位)
        hab_score = hab_buffer.rolling(89, min_periods=20).rank(pct=True).fillna(0.5)
        # 6. 最终合成
        # 障碍清除度 = 瞬时吸收强度(30%) + 存量水位(70%)
        # 存量水位在横盘区具有一票否决权，只有存量足够高，才视为良性换手
        final_score = (support_rebound_intensity * 0.3 + hab_score * 0.7).clip(0, 1)
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"--- Pressure Absorption Physics Probe ---")
            self._probe_print(f"  > PAR_Norm: {par_norm.iloc[-1]:.4f} | Rebound_Int: {support_rebound_intensity.iloc[-1]:.4f}")
            self._probe_print(f"  > Obstacle_HAB_Rank: {hab_score.iloc[-1]:.4f}")
        return final_score.astype(np.float32)

    def _synthesize_bullish_intent(self, df_index: pd.Index, aggressiveness_score: pd.Series, control_score: pd.Series, obstacle_clearance_score: pd.Series, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], dynamic_weights: Dict[str, pd.Series], historical_context: Dict[str, Any], params: Dict) -> pd.Series:
        """
        【V6.1 · 诱空参数缝合版】主力多头意图深度合成
        修改说明：更新诱空陷阱检测接口，传递obstacle_clearance_score以利用PAR-HAB阶跃提升识别精度。
        版本号：2026.02.10.351
        """
        # 1. 基础意图：对数空间几何平均 (否决制)
        comps = {'agg': aggressiveness_score.clip(1e-6, 1), 'ctrl': control_score.clip(1e-6, 1), 'obs': obstacle_clearance_score.clip(1e-6, 1)}
        w = {'agg': dynamic_weights.get('aggressiveness', 0.35), 'ctrl': dynamic_weights.get('control', 0.35), 'obs': dynamic_weights.get('obstacle_clearance', 0.30)}
        log_intent = (w['agg'] * np.log(comps['agg']) + w['ctrl'] * np.log(comps['ctrl']) + w['obs'] * np.log(comps['obs']))
        bullish_base = np.exp(log_intent).clip(0, 1)
        # 2. 增强因子：共振与诱空 (修正调用签名)
        res_score = self._calculate_acceleration_resonance(df_index, mtf_signals)
        trap_score = self._detect_bear_trap(df_index, mtf_signals, normalized_signals, obstacle_clearance_score)
        # 3. 物理增强合成：诱空陷阱在启动初期具有极高的爆发权重
        enhancement = np.tanh(res_score * 0.4 + trap_score * 0.8).clip(0, 1)
        # 4. 结合历史记忆 HAB
        int_mem = historical_context.get('integrated_memory', pd.Series(0.5, index=df_index))
        mem_rank = int_mem.rolling(55).rank(pct=True).fillna(0.5)
        # 5. 最终意图
        final_intent = (bullish_base * (1 + enhancement * 0.25) * (0.9 + mem_rank * 0.1)).clip(0, 1)
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"[BULL_SYNTH_PROBE] Trap_Score: {trap_score.iloc[-1]:.4f} | Enhance: {enhancement.iloc[-1]:.4f}")
            
        return final_intent

    def _calculate_bearish_intent(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], historical_context: Dict[str, Any]) -> pd.Series:
        """
        【V7.0 · 诱多陷阱增强版】计算复合看跌意图
        修改说明：引入诱多陷阱(Bull Trap)检测。利用物理Jerk识别高位“外强中干”的受力特征，增强看跌意图的前瞻性。
        版本号：2026.02.10.361
        """
        dist_score = pd.Series(normalized_signals.get('distribution_score_norm', 0.0), index=df_index)
        # 1. 派发运动学分析
        dist_slope = dist_score.diff(1).rolling(3).mean().fillna(0)
        dist_accel = dist_slope.diff(1).fillna(0)
        dist_jerk = dist_accel.diff(1).abs().fillna(0)
        # 2. 派发存量 (Distribution-HAB) - 使用 Rolling Rank
        hab_inc = dist_score * (1 + dist_accel.clip(lower=0))
        dist_hab_buffer = hab_inc.rolling(window=21, min_periods=5).sum().fillna(0)
        dist_hab = dist_hab_buffer.rolling(34).rank(pct=True).fillna(0.0)
        # 3. 【核心新增】诱多陷阱检测 (Bull Trap)
        bull_trap_score = self._detect_bull_trap(df_index, mtf_signals, normalized_signals, dist_hab)
        # 4. 恐慌冲击感应
        p_acc_down = pd.Series(mtf_signals.get('ACCEL_5_price_trend', 0.0), index=df_index).clip(upper=0).abs()
        panic_impulse = (p_acc_down * dist_accel.clip(lower=0)).pow(0.5).fillna(0)
        # 5. 最终合成：诱多陷阱分值作为溢价因子注入
        bear_comp = {'dist': dist_score, 'hab': dist_hab, 'panic': panic_impulse}
        log_bear = (0.3 * np.log(bear_comp['dist'].clip(1e-6)) + 
                    0.4 * np.log(bear_comp['hab'].clip(1e-6)) + 
                    0.3 * np.log(bear_comp['panic'].clip(1e-6)))
        base_bear_score = np.exp(log_bear).clip(0, 1)
        # 诱多增强：如果检测到诱多陷阱，看跌意图最高提升 30%
        final_bear_score = (base_bear_score * (1 + bull_trap_score * 0.3)).clip(0, 1)
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"--- Bearish Intent Synthesis Probe ---")
            self._probe_print(f"  > Bull_Trap_Boost: {bull_trap_score.iloc[-1]:.4f} | Final_Bear: {-final_bear_score.iloc[-1]:.4f}")
            
        return -final_bear_score.astype(np.float32)

    def _adjudicate_risk(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], dynamic_weights: Dict[str, pd.Series], aggressiveness_score: pd.Series, params: Dict, market_phase: pd.Series) -> pd.Series:
        """
        【V8.5 · 物理软阈值风险版】深度风险裁决逻辑
        修改说明：引入Sigmoid非线性风险阈值。在底气偏差(Offset)处于中高区时执行软熔断，防止在强势横盘期误杀多头意图。
        版本号: 2026.02.10.320
        """
        tech_risk = (pd.Series(normalized_signals.get('RSI_norm', 0.5), index=df_index) - 0.75).clip(lower=0) * 2.0
        struct_risk = pd.Series(normalized_signals.get('distribution_score_norm', 0.0), index=df_index)
        # 1. 物理底气背离 (Offset)
        p_hab = pd.Series(normalized_signals.get('absolute_change_strength_norm', 0.5), index=df_index).rolling(21, min_periods=1).mean().fillna(0.5)
        c_hab = pd.Series(normalized_signals.get('net_mf_amount_norm', 0.5), index=df_index).rolling(21, min_periods=1).mean().fillna(0.5)
        hab_offset = (p_hab - c_hab).clip(lower=0)
        # 2. 【核心重构】Sigmoid 风险软阈值
        # 逻辑：在 Offset < 0.5 时几乎无感，在 0.5-0.8 之间快速爬升，利用 Sigmoid 平滑极值冲击
        hollow_risk_base = 1 / (1 + np.exp(-10 * (hab_offset - 0.65)))
        # 3. 动态物理约束：若价格加速度仍为正，则大幅衰减背离风险
        p_acc = pd.Series(mtf_signals.get('ACCEL_5_price_trend', 0.0), index=df_index)
        risk_modulator = market_phase.map({"派发初期": 0.2, "主升": 0.3, "横盘": 0.4, "派发末端": 1.2, "反转风险": 1.5}).fillna(1.0)
        # 如果加速度为正，背离风险仅保留 50%
        final_hollow_risk = (hollow_risk_base * risk_modulator * (np.where(p_acc > 0, 0.5, 1.0))).clip(0, 1)
        # 4. 指数级惩罚映射
        total_risk_raw = (tech_risk * 0.2 + struct_risk * 0.3 + final_hollow_risk * 0.5).clip(0, 1)
        penalty = np.where(total_risk_raw > 0.65, total_risk_raw ** 0.5, total_risk_raw ** 2.5)
        risk_sens = get_param_value(params.get('risk_sensitivity'), 4.0)
        final_penalty = 1 / (1 + np.exp(-risk_sens * (penalty - 0.7))) # 中心点进一步上移
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"[RISK_SOFT_PROBE] Offset: {hab_offset.iloc[-1]:.4f} | Hollow_Risk: {final_hollow_risk.iloc[-1]:.4f} | Penalty: {final_penalty.iloc[-1]:.4f}")
        return final_penalty.clip(0, 1)

    def _apply_contextual_modulators(self, df_index: pd.Index, final_rally_intent: pd.Series, proxy_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], market_phase: pd.Series, aggressiveness_score: pd.Series) -> pd.Series:
        """
        【V4.0 · 情绪溢价调节版】应用情境调节器
        修改说明：新增情绪溢价调节逻辑，在“派发初期”且高攻击性状态下赋予分值溢价，体现投机惯性。
        版本号：2026.02.10.280
        """
        def _safe_mod(s, center=0.5, scope=0.15):
            return 1.0 + (np.tanh((s - center) * 4) * scope)
        # 1. 基础物理调节
        rs_val = proxy_signals.get('enhanced_rs_proxy', pd.Series(0.5, index=df_index))
        cap_val = proxy_signals.get('enhanced_capital_proxy', pd.Series(0.5, index=df_index))
        modulator = _safe_mod(rs_val) * _safe_mod(cap_val, scope=0.1)
        # 2. 【核心新增】投机溢价 (Speculative Premium)
        # 逻辑：在派发初期，高Agg意味着极强的连板惯性，给予额外溢价
        spec_boost = pd.Series(1.0, index=df_index)
        is_early_dist = (market_phase == "派发初期")
        # 溢价系数：攻击性越高，溢价越强，最高20%
        premium_inc = (aggressiveness_score - 0.5).clip(lower=0) * 0.4
        spec_boost.loc[is_early_dist] = 1.0 + premium_inc.loc[is_early_dist]
        # 3. 最终应用与保护
        final_intent = (final_rally_intent * modulator * spec_boost).clip(-1, 1)
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"[MOD_PREMIUM_PROBE] Phase: {market_phase.iloc[-1]} | Spec_Boost: {spec_boost.iloc[-1]:.4f}")
        return final_intent

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