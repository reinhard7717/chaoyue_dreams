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
        【V12.1 · 全链路缝合与审计版】主力拉升意图主计算流程
        修改说明：集成全链路结构化探针，优化风险惩罚与多头意图的缝合逻辑，确保输出值的物理合理性。
        版本号：2026.02.10.246
        """
        self._probe_output = []
        params = self._get_parameters(config)
        df_index = df.index
        # 1. 信号准备与预处理
        raw_signals = self._get_raw_signals(df)
        is_limit_up_day = df.apply(lambda row: is_limit_up(row), axis=1)
        normalized_signals = self._normalize_raw_signals(df_index, raw_signals)
        mtf_signals = self._calculate_mtf_fused_signals(df, raw_signals, params['mtf_slope_accel_weights'], df_index)
        # 2. 存量载入与上下文计算
        historical_context = self._calculate_historical_context(df, df_index, raw_signals, mtf_signals, params['historical_context_params'])
        # 3. 代理信号与权重分配
        proxy_signals = self._construct_proxy_signals(df_index, mtf_signals, normalized_signals, config)
        dynamic_weights = self._calculate_dynamic_weights(df_index, normalized_signals, proxy_signals, mtf_signals)
        # 4. 核心维度计算
        aggressiveness_score = self._calculate_aggressiveness_score(df_index, mtf_signals, normalized_signals, dynamic_weights)
        control_score = self._calculate_control_score(df_index, raw_signals, mtf_signals, normalized_signals, historical_context)
        obstacle_clearance_score = self._calculate_obstacle_clearance_score(df_index, raw_signals, mtf_signals, normalized_signals, historical_context)
        # 5. 风险裁决与多空意图合成
        total_risk_penalty = self._adjudicate_risk(df_index, raw_signals, mtf_signals, normalized_signals, dynamic_weights, aggressiveness_score, params['rally_intent_synthesis_params'])
        bullish_intent = self._synthesize_bullish_intent(df_index, aggressiveness_score, control_score, obstacle_clearance_score, mtf_signals, normalized_signals, dynamic_weights, historical_context, params['rally_intent_synthesis_params'])
        bearish_score = self._calculate_bearish_intent(df_index, raw_signals, mtf_signals, normalized_signals, historical_context)
        # 6. 最终意图缝合：引入平滑修正，防止风险惩罚过度零化
        final_rally_intent = (bullish_intent * (1 - total_risk_penalty * 0.8) + bearish_score).fillna(0).clip(-1, 1)
        final_rally_intent = self._apply_contextual_modulators(df_index, final_rally_intent, proxy_signals, mtf_signals)
        final_rally_intent = final_rally_intent.mask(is_limit_up_day & (final_rally_intent < 0), 0.0)
        # 7. 存量状态持久化
        self._persist_hab_states(historical_context)
        self.strategy.atomic_states["_DEBUG_rally_integrated_hab"] = historical_context.get('integrated_memory', 0.5)
        # 8. 执行全链路深度审计探针
        if self._is_probe_enabled(df):
            self._execute_full_link_probing(df_index, raw_signals, mtf_signals, proxy_signals, historical_context, bullish_intent, final_rally_intent)
            self._output_probe_info(df_index, final_rally_intent)
        return final_rally_intent.astype(np.float32)

    def _execute_full_link_probing(self, df_index: pd.Index, raw_signals: Dict, mtf_signals: Dict, proxy_signals: Dict, historical_context: Dict, bullish_intent: pd.Series, final_intent: pd.Series):
        """
        【V1.0 · 全链路审计中枢】执行从原始信号到最终意图的深度探针审计
        修改说明：建立结构化审计流程，通过五个维度监控物理量纲的传递过程，并对溢出风险进行哨兵报警。
        版本号：2026.02.10.245
        """
        ts = df_index[-1]
        self._probe_output.append(f"=== [FULL_LINK_AUDIT] {ts.strftime('%Y-%m-%d')} ===")
        # 1. 信号层：校验归一化基准
        p_norm = raw_signals.get('pct_change', pd.Series(0, index=df_index)).iloc[-1]
        self._probe_output.append(f"[L1_SIGNAL] Pct_Change: {p_norm:.4f}")
        # 2. 物理层：校验运动学矢量 (SAJ)
        p_acc = mtf_signals.get('ACCEL_5_price_trend', pd.Series(0, index=df_index)).iloc[-1]
        v_jerk = mtf_signals.get('JERK_5_volume_trend', pd.Series(0, index=df_index)).iloc[-1]
        self._probe_output.append(f"[L2_PHYSICS] Price_Accel: {p_acc:.4f}, Vol_Jerk: {v_jerk:.4f}")
        # 3. 存量层：校验 HAB 水位
        c_hab = historical_context.get('capital_memory', {}).get('hab_score', pd.Series(0.5, index=df_index)).iloc[-1]
        ch_hab = historical_context.get('chip_memory', {}).get('hab_score', pd.Series(0.5, index=df_index)).iloc[-1]
        self._probe_output.append(f"[L3_HAB] Cap_HAB_Rank: {c_hab:.4f}, Chip_HAB_Rank: {ch_hab:.4f}")
        # 4. 权重层：校验 Softmax 温度
        rs_w = proxy_signals.get('rs_weight', pd.Series(0.0, index=df_index)).iloc[-1]
        self._probe_output.append(f"[L4_WEIGHT] RS_Softmax_Weight: {rs_w:.4f}")
        # 5. 合成层：校验最终意图与 NaN 风险
        b_val = bullish_intent.iloc[-1]
        f_val = final_intent.iloc[-1]
        if np.isnan(f_val): self._probe_output.append("[L5_ALERT] Detected NaN in final_intent!")
        self._probe_output.append(f"[L5_SYNTHESIS] Bullish: {b_val:.4f} -> Final: {f_val:.4f}")

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

    def _calculate_mtf_fused_signals(self, df: pd.DataFrame, raw_signals: Dict[str, pd.Series], mtf_slope_accel_weights: Dict, df_index: pd.Index) -> Dict[str, pd.Series]:
        """
        【V8.1 · 鲁棒性增强版】MTF跨周期信号融合
        修改说明：修复探针显示的NaN值问题。在标准化环节引入强制非空填充，确保冷启动阶段的能量共振分值为0（中性）。
        版本号：2026.02.10.180
        """
        mtf_signals = {}
        for internal_key, series in raw_signals.items():
            if any(prefix in internal_key for prefix in ['SLOPE_', 'ACCEL_', 'JERK_']):
                mtf_signals[internal_key] = series
        core_fusion_signals = {'price_trend': 'close_D', 'volume_trend': 'volume_D', 'net_amount_trend': 'net_amount_D', 'market_sentiment_trend': 'market_sentiment_score_D'}
        weights = {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}
        for signal_name, col_name in core_fusion_signals.items():
            fused_score = pd.Series(0.0, index=df_index)
            for window, weight in weights.items():
                val = df[col_name].diff(window).fillna(0)
                # 核心修复：处理rolling初期产生的NaN，防止NaN污染加权和
                std = val.rolling(window * 3).std().replace(0, 1e-9).fillna(1.0)
                norm_val = np.tanh(val / (std * 2)).fillna(0)
                fused_score += norm_val * weight
            mtf_signals[f'mtf_{signal_name}'] = fused_score.clip(-1, 1)
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            nan_count = mtf_signals['mtf_price_trend'].isna().sum()
            self._probe_print(f"[MTF_PROBE] Price_MTF_Resonance: {mtf_signals['mtf_price_trend'].iloc[-1]:.4f}, NaNs: {nan_count}")
        return mtf_signals

    def _calculate_historical_context(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], params: Dict) -> Dict[str, pd.Series]:
        """
        【V5.0 · 持久化感知版】历史上下文计算调度中心
        修改说明：接入HAB持久化载入逻辑，将历史存量作为初始值注入各内存计算模块，确保意图分析的物理连贯性。
        版本号：2026.02.10.171
        """
        hc_enabled = get_param_value(params.get('enabled'), True)
        if not hc_enabled: return self._get_empty_context(df_index)
        # 1. 从原子状态机载入历史HAB水位
        initial_hab_states = self._load_hab_states()
        # 2. 注入历史水位进行子模块计算
        price_memory = self._calculate_price_memory(df_index, raw_signals, mtf_signals, params, initial_hab_states.get('price_hab', 0.0))
        capital_memory = self._calculate_capital_memory(df_index, raw_signals, mtf_signals, params, initial_hab_states.get('capital_hab', 0.0))
        chip_memory = self._calculate_chip_memory(df_index, raw_signals, mtf_signals, params, initial_hab_states.get('chip_hab', 0.0))
        sentiment_memory = self._calculate_sentiment_memory(df_index, raw_signals, mtf_signals, params, initial_hab_states.get('sentiment_hab', 0.0))
        # 3. 融合与评估
        integrated_memory = self._fuse_integrated_memory(price_memory, capital_memory, chip_memory, sentiment_memory, params)
        phase_sync = self._detect_phase_synchronization(df_index, price_memory, capital_memory, integrated_memory)
        memory_quality = self._assess_memory_quality(df_index, price_memory, capital_memory, chip_memory, sentiment_memory)
        return {
            "price_memory": price_memory, "capital_memory": capital_memory, "chip_memory": chip_memory,
            "sentiment_memory": sentiment_memory, "integrated_memory": integrated_memory,
            "phase_sync": phase_sync, "memory_quality": memory_quality, "hc_enabled": hc_enabled,
            "dynamic_memory_period": self._calculate_dynamic_period(df_index, raw_signals)
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

    def _calculate_dynamic_period(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series]) -> pd.Series:
        """
        【V4.1 · 强类型安全版】动态记忆周期计算
        修改说明：彻底解决IntCastingNaNError。在进行np.int32强转前，对计算链路进行全量非有限值（NaN/Inf）清除。
        版本号：2026.02.10.181
        """
        pct_change = raw_signals.get('pct_change', pd.Series(0.0, index=df_index)).fillna(0.0)
        vol_ratio = raw_signals.get('volume_ratio', pd.Series(1.0, index=df_index)).fillna(1.0)
        # 1. 基础波动率位置 (归一化逻辑内化)
        volatility = pct_change.rolling(window=21).std().fillna(0.01)
        vol_min = volatility.rolling(window=60).min().fillna(0.01)
        vol_max = volatility.rolling(window=60).max().fillna(1.0)
        vol_pos = ((volatility - vol_min) / (vol_max - vol_min + 1e-9)).clip(0, 1)
        # 2. 趋势一致性 (ER)
        net_diff = pct_change.rolling(window=10).sum().abs()
        path_len = pct_change.abs().rolling(window=10).sum().replace(0, 1e-9)
        er = (net_diff / path_len).clip(0, 1)
        # 3. Jerk 稳定性调节 (内化归一化防止NaN透传)
        price_jerk = pct_change.diff().diff().rolling(5).mean().abs().fillna(0)
        j_med = price_jerk.rolling(60).median().replace(0, 1e-9)
        jerk_norm = np.tanh(price_jerk / j_med).clip(0, 1)
        # 4. 周期合成
        # 核心逻辑：ER增加周期，Vol减少周期，Jerk剧烈减少周期
        dynamic_period = 21 * (1 + er * 0.4 - vol_pos * 0.3 - jerk_norm * 0.5)
        # 5. 异常修正与强制转换
        vol_adj = (vol_ratio / 5.0).clip(0, 0.3)
        # 核心修复点：使用 replace + fillna 确保数据有限，再进行 astype
        final_period_series = (dynamic_period * (1 - vol_adj)).clip(5, 55).round()
        final_period_series = final_period_series.replace([np.inf, -np.inf], 21).fillna(21)
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"[PERIOD_PROBE] Final_Period: {final_period_series.iloc[-1]}, Jerk_Norm: {jerk_norm.iloc[-1]:.4f}")
        return final_period_series.astype(np.int32)

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

    def _calculate_memory_consistency(self, price_mem: Dict, capital_mem: Dict, chip_mem: Dict, sentiment_mem: Dict) -> pd.Series:
        """
        【V5.0 · 多维共振场与一致性存量版】记忆一致性深度计算
        修改说明：引入Consistency-HAB，量化多维度信号在时间轴上的同步稳定性，识别“系统性主升”特征。
        版本号：2026.02.10.76
        """
        df_index = price_mem["integrated_price_memory"].index
        # 1. 提取核心维度的加速度矢量
        accel_df = pd.DataFrame({
            "p_acc": price_mem.get("accel_memory", 0),
            "c_acc": capital_mem.get("cap_accel", 0),
            "ch_acc": chip_mem.get("chip_accel", 0),
            "s_acc": sentiment_mem.get("sentiment_accel", 0)
        })
        # 2. 计算瞬时共振强度：四维度方向一致性的非线性加权
        resonance_count = (accel_df > 0).sum(axis=1)
        instant_resonance = resonance_count.map({4: 1.0, 3: 0.7, 2: 0.3, 1: 0.1, 0: 0.0})
        # 3. 构建一致性存量 (Consistency-HAB)
        # 物理意义：一致性不是一个点，而是一个“场”。场强取决于同步行为的持续时间。
        hab_window = 13
        # 当共振强度高且加速度绝对值大时，累积一致性存量
        accel_magnitude = accel_df.abs().mean(axis=1)
        hab_inc = instant_resonance * accel_magnitude
        consistency_hab = normalize_score(hab_inc.rolling(window=hab_window).sum().fillna(0), target_index=df_index, windows=21)
        # 4. 综合合成：历史相关性(20%) + 瞬时共振(30%) + 一致性存量(50%)
        rolling_corr = accel_df.rolling(window=10).corr().unstack().mean(axis=1).fillna(0.5).clip(0, 1)
        final_consistency = (rolling_corr * 0.2 + instant_resonance * 0.3 + consistency_hab * 0.5).clip(0, 1)
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"--- Consistency Resonance Field Probe ---")
            self._probe_print(f"  > Instant Resonance: {instant_resonance.iloc[-1]:.2f}")
            self._probe_print(f"  > Consistency-HAB Stock: {consistency_hab.iloc[-1]:.4f}")
        return final_consistency.astype(np.float32)

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
        【V6.0 · 能量场自闭环版】代理信号构建中枢
        修改说明：彻底废弃外部归一化。内部实现Proxy-HAB的Rolling Rank排位，并集成自适应信噪比评估。
        版本号：2026.02.10.142
        """
        rs_res = self._calculate_enhanced_rs_proxy(df_index, mtf_signals, normalized_signals, config)
        cap_res = self._calculate_enhanced_capital_proxy(df_index, mtf_signals, normalized_signals, config)
        sent_res = self._calculate_enhanced_sentiment_proxy(df_index, mtf_signals, normalized_signals, config)
        liq_res = self._calculate_enhanced_liquidity_proxy(df_index, mtf_signals, normalized_signals, config)
        vol_res = self._calculate_enhanced_volatility_proxy(df_index, mtf_signals, normalized_signals, config)
        risk_res = self._calculate_enhanced_risk_preference_proxy(df_index, mtf_signals, normalized_signals, config)
        proxy_cores = {"rs": rs_res.get("enhanced_rs_proxy"), "capital": cap_res.get("enhanced_capital_proxy"), "sentiment": sent_res.get("enhanced_sentiment_proxy")}
        # 1. 物理矢量与存量累积 (HAB)
        proxy_hab_scores = {}
        hab_win = 21
        for name, series in proxy_cores.items():
            if series is not None:
                hab_inc = series.mask(series < 0.6, 0).diff(1).clip(lower=0)
                hab_raw = hab_inc.rolling(window=hab_win).sum().fillna(0)
                # 专用归一化：Rolling Rank (局部排位)
                proxy_hab_scores[f"{name}_hab_score"] = hab_raw.rolling(55).rank(pct=True).fillna(0.5)
        # 2. 内聚式信号质量评估 (基于 SNR 映射)
        quality_list = []
        for name, series in proxy_cores.items():
            noise = series.diff().abs().rolling(13).sum().replace(0, 1e-9)
            signal = series.diff(13).abs()
            snr = (signal / noise).clip(0, 1)
            quality_list.append(np.tanh(snr * 3)) # Tanh 非线性增强
        signal_quality = pd.concat(quality_list, axis=1).mean(axis=1).fillna(0.7)
        # 3. 动态权重合成与 HAB 注入
        final_proxy_signals = self._dynamic_weighted_synthesis(rs_res, cap_res, sent_res, liq_res, vol_res, risk_res, signal_quality, config)
        final_proxy_signals.update(proxy_hab_scores)
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"--- Proxy Field Integrated HAB: {proxy_hab_scores.get('rs_hab_score', pd.Series(0.5)).iloc[-1]:.4f}")
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
        【V5.0 · 资金运动学与弹药存量版】增强版资本属性代理信号
        修改说明：全面引入主力净额(net_mf_amount)的S/A/J运动学特征，并构建Capital-HAB历史累积缓冲系统。
        版本号：2026.02.10.80
        """
        # 1. 提取资金运动学三维矢量
        cap_slope = mtf_signals.get('SLOPE_5_mf_net_amount', pd.Series(0.0, index=df_index))
        cap_accel = mtf_signals.get('ACCEL_5_mf_net_amount', pd.Series(0.0, index=df_index))
        cap_jerk = mtf_signals.get('JERK_5_mf_net_amount', pd.Series(0.0, index=df_index))
        # 2. 资金历史累积记忆缓冲 (Capital-HAB)
        # 逻辑：将主力净流入强度与一致性进行累积。高水位代表主力在此价格区间有重仓防御意识。
        net_mf_norm = normalized_signals.get('net_mf_amount_norm', pd.Series(0.5, index=df_index))
        flow_consis = normalized_signals.get('flow_consistency_norm', pd.Series(0.5, index=df_index))
        hab_window = get_param_value(config.get('capital_hab_window'), 21)
        # 在流入且加速度不为负时计入存量累积
        hab_increment = (net_mf_norm.mask(net_mf_norm < 0.5, 0) * flow_consis * cap_accel.clip(lower=0))
        cap_hab_buffer = hab_increment.rolling(window=hab_window, min_periods=5).sum().fillna(0)
        cap_hab_score = normalize_score(cap_hab_buffer, target_index=df_index, windows=34)
        # 3. 资本效率因子 (CEF)
        vol_ratio = normalized_signals.get('volume_ratio_norm', pd.Series(0.5, index=df_index))
        capital_efficiency = (cap_slope.clip(lower=0) / (vol_ratio + 0.1)).clip(0, 1)
        # 4. 综合合成：运动学(40%) + 存量水位(40%) + 效率(20%)
        kinematic_sum = (cap_slope * 0.4 + cap_accel * 0.4 + cap_jerk * 0.2).clip(0, 1)
        raw_proxy = (kinematic_sum * 0.4 + cap_hab_score * 0.4 + capital_efficiency * 0.2).clip(0, 1)
        # 5. 情境调节
        liq_state = normalized_signals.get('volume_ratio_norm', pd.Series(0.5, index=df_index))
        capital_modulator = 0.9 + (liq_state * 0.2)
        enhanced_capital_proxy = (raw_proxy * capital_modulator).clip(0, 1)
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"--- Capital Energy-HAB Probe ---")
            self._probe_print(f"  > Cap SAJ: S={cap_slope.iloc[-1]:.4f}, A={cap_accel.iloc[-1]:.4f}, J={cap_jerk.iloc[-1]:.4f}")
            self._probe_print(f"  > Capital-HAB Level: {cap_hab_score.iloc[-1]:.4f}")
        return {"raw_capital_proxy": raw_proxy, "enhanced_capital_proxy": enhanced_capital_proxy, "capital_hab_score": cap_hab_score, "capital_modulator": capital_modulator}

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

    def _calculate_dynamic_weights(self, df_index: pd.Index, normalized_signals: Dict[str, pd.Series], proxy_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V6.0 · 权重记忆存量与温度控制版】动态权重计算系统
        修改说明：废弃外部归一化，内部实现带温度控制的Softmax分配，并引入Weight-HAB确保权重在风格切换时的物理惯性。
        版本号：2026.02.10.120
        """
        factors = self._calculate_market_state_factors(df_index, normalized_signals, mtf_signals)
        phase = self._identify_market_phase(df_index, factors, normalized_signals)
        base_weights_raw = self._calculate_base_weights(df_index, phase, factors)
        # 1. 提取运动学增益 (加速度聚焦)
        rs_acc = proxy_signals.get('rs_accel', pd.Series(0.0, index=df_index)).abs()
        sent_acc = proxy_signals.get('sentiment_accel', pd.Series(0.0, index=df_index)).abs()
        # 2. 构建 Weight-HAB (权重存量记忆)
        # 逻辑：基础权重结合加速度增益后，通过13日窗口进行指数平滑，滤除日内噪音
        final_weights_unnorm = {}
        for dim, b_w in base_weights_raw.items():
            boost = (rs_acc * 0.5 + sent_acc * 0.5) if dim == 'aggressiveness' else 0
            raw_w = b_w * (1 + boost)
            final_weights_unnorm[dim] = raw_w.ewm(span=13).mean()
        # 3. 内部 Softmax 归一化
        final_weights = {dim: pd.Series(0.0, index=df_index) for dim in final_weights_unnorm.keys()}
        quality = factors.get('state_hab_score', pd.Series(0.7, index=df_index))
        for i in range(len(df_index)):
            # 提取当前时刻原始权重
            raw_vals = np.array([final_weights_unnorm[d].iloc[i] for d in final_weights_unnorm])
            # 温度控制：质量越高(T越小)，权重越集中；质量越低(T越大)，权重越平均
            q_val = quality.iloc[i]
            temp = 2.0 - (q_val * 1.5) 
            exp_vals = np.exp(raw_vals / temp)
            norm_vals = exp_vals / np.sum(exp_vals)
            for idx, dim in enumerate(final_weights_unnorm.keys()):
                final_weights[dim].iloc[i] = norm_vals[idx]
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"[WEIGHT_PROBE] Quality_T: {temp:.2f}, Agg_Weight: {final_weights['aggressiveness'].iloc[-1]:.4f}")
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
        【V5.1 · 数值映射平滑版】市场阶段识别算法
        修改说明：修复'DataError: No numeric types to aggregate'。通过将阶段标签数值化，解决pandas无法对字符串序列进行rolling操作的问题。
        版本号：2026.02.10.200
        """
        # 1. 提取核心水位与动力分量 (类型加固)
        p_hab = pd.Series(market_state_factors.get('state_hab_score', 0.5), index=df_index)
        c_hab = pd.Series(normalized_signals.get('net_mf_amount_norm', 0.5), index=df_index).rolling(21).rank(pct=True).fillna(0.5)
        p_acc = pd.Series(market_state_factors.get('trend_state_accel', 0.0), index=df_index)
        # 2. 相位交叉识别
        cap_accel = pd.Series(normalized_signals.get('flow_acceleration_norm', 0.5), index=df_index)
        cap_lead_price = (cap_accel > 0.6) & (p_acc < 0)
        # 3. 各阶段概率得分
        prob_acc = (c_hab * (1 - p_hab)).clip(0, 1)
        prob_exp = (p_hab * c_hab * (p_acc.clip(lower=0) + 0.5)).clip(0, 1)
        prob_dist = (p_hab * (1 - c_hab)).clip(0, 1)
        p_jerk = p_acc.diff(1).fillna(0)
        prob_panic = (p_jerk.clip(upper=0).abs() * p_hab).clip(0, 1)
        # 4. 状态决策矩阵
        phase_matrix = pd.DataFrame({
            "蓄势": prob_acc, "主升": prob_exp, "派发": prob_dist, "反转风险": prob_panic, "横盘": 0.3
        }, index=df_index)
        phase_matrix.loc[cap_lead_price, "蓄势"] += 0.3
        raw_phases = phase_matrix.idxmax(axis=1)
        # 5. 数值映射平滑处理 (核心修复点)
        phase_map = {"蓄势": 0, "主升": 1, "派发": 2, "反转风险": 3, "横盘": 4}
        inv_phase_map = {v: k for k, v in phase_map.items()}
        # 将字符串转换为整数进行滚动计算
        raw_phases_int = raw_phases.map(phase_map).astype(np.float64)
        def get_rolling_mode(x):
            m = x.mode()
            return m.iloc[0] if not m.empty else x.iloc[-1]
        # 对整数序列执行平滑，随后映射回字符串
        smoothed_phases_int = raw_phases_int.rolling(window=5).apply(get_rolling_mode, raw=False).fillna(4)
        smoothed_phases = smoothed_phases_int.map(inv_phase_map).fillna("横盘")
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"--- Phase Numeric-Mapping Probe ---")
            self._probe_print(f"  > Probabilities: Acc={prob_acc.iloc[-1]:.2f}, Exp={prob_exp.iloc[-1]:.2f}")
            self._probe_print(f"  > Final Phase: {smoothed_phases.iloc[-1]}")
        return smoothed_phases

    def _calculate_base_weights(self, df_index: pd.Index, market_phase: pd.Series, market_state_factors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V5.0 · 物理相位与惯性调节版】基础权重计算
        修改说明：废弃静态配置，引入基于SAJ加速度与HAB惯性存量的动态权重调节器，实现权重分配在物理相位层面的非线性响应。
        版本号：2026.02.10.210
        """
        # 1. 提取物理分量
        p_acc = pd.Series(market_state_factors.get('trend_state_accel', 0.0), index=df_index)
        s_hab = pd.Series(market_state_factors.get('state_hab_score', 0.5), index=df_index)
        # 2. 定义相位基准权重矩阵
        # (攻击性, 控制力, 障碍清除, 风险)
        phase_config = {
            '启动': np.array([0.35, 0.30, 0.25, 0.10]),
            '主升': np.array([0.45, 0.20, 0.25, 0.10]),
            '调整': np.array([0.20, 0.40, 0.15, 0.25]),
            '反转风险': np.array([0.10, 0.25, 0.15, 0.50]),
            '横盘': np.array([0.25, 0.35, 0.20, 0.20])
        }
        # 3. 初始化权重序列
        w_agg = pd.Series(0.0, index=df_index)
        w_ctrl = pd.Series(0.0, index=df_index)
        w_obs = pd.Series(0.0, index=df_index)
        w_risk = pd.Series(0.0, index=df_index)
        # 4. 执行动态调节逻辑
        for i in range(len(df_index)):
            phase = market_phase.iloc[i]
            base = phase_config.get(phase, phase_config['横盘']).copy()
            # SAJ 调节：加速度越高，攻击性溢价越高；加速度转负，风险权重激增
            acc_val = p_acc.iloc[i]
            if acc_val > 0:
                base[0] *= (1 + acc_val * 0.5)
            else:
                base[3] *= (1 + abs(acc_val) * 0.8)
            # HAB 调节：环境惯性越高，倾向于维持主趋势权重 (攻击/控制)，降低随机波动风险权重
            hab_val = s_hab.iloc[i]
            base[0] *= (0.9 + hab_val * 0.2)
            base[1] *= (0.9 + hab_val * 0.2)
            base[3] *= (1.1 - hab_val * 0.2)
            # 5. 内部 L1 归一化 (确保 Sum = 1.0)
            total = np.sum(base) + 1e-9
            norm_w = base / total
            w_agg.iloc[i], w_ctrl.iloc[i], w_obs.iloc[i], w_risk.iloc[i] = norm_w
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"--- Weight Dynamic Regulator Probe ---")
            self._probe_print(f"  > Current Phase: {market_phase.iloc[-1]} | Accel: {p_acc.iloc[-1]:.4f}")
            self._probe_print(f"  > Weights: Agg={w_agg.iloc[-1]:.2f}, Ctrl={w_ctrl.iloc[-1]:.2f}, Risk={w_risk.iloc[-1]:.2f}")
        return {
            'aggressiveness': w_agg.clip(0.05, 0.6),
            'control': w_ctrl.clip(0.05, 0.6),
            'obstacle_clearance': w_obs.clip(0.05, 0.6),
            'risk': w_risk.clip(0.05, 0.6)
        }

    def _calculate_aggressiveness_score(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], dynamic_weights: Dict[str, pd.Series]) -> pd.Series:
        """
        【V7.0 · 共振分位数攻击版】计算攻击性分值
        修改说明：采用几何平均+Rolling Rank定制化归一化，确保攻击性分值仅在多维度极值共振时爆发。
        版本号：2026.02.10.94
        """
        # 1. 基础分量
        p_trend = mtf_signals.get('mtf_price_trend', 0).clip(lower=0)
        v_trend = mtf_signals.get('mtf_volume_trend', 0).clip(lower=0)
        f_accel = normalized_signals.get('flow_acceleration_norm', 0).clip(lower=0)
        # 2. 几何平均合成 (强调一致性)
        agg_raw = (p_trend * v_trend * f_accel) ** (1/3)
        # 3. 定制化归一化：90日超长窗口的分位数映射 (识别真正的年度级别进攻)
        agg_rank = agg_raw.rolling(window=90, min_periods=20).rank(pct=True).fillna(0.5)
        # 4. Jerk 冲击增强
        p_jerk = mtf_signals.get('JERK_5_price_trend', 0).abs()
        j_norm = np.tanh(p_jerk / p_jerk.rolling(60).mean().replace(0, 1)).clip(0, 1)
        final_score = (agg_rank * (1 + j_norm * 0.2)).clip(0, 1)
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"[AGG_PROBE] Agg_Rank: {agg_rank.iloc[-1]:.4f}, Jerk_Boost: {j_norm.iloc[-1]:.4f}")
        return final_score.astype(np.float32)

    def _detect_bear_trap(self, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series]) -> pd.Series:
        """
        【V6.0 · Tanh反转强度版】诱空陷阱检测
        修改说明：采用tanh非线性映射取代通用归一化，精准捕捉价格加速度从负值极速转正的物理拐点。
        版本号：2026.02.10.121
        """
        p_acc = mtf_signals.get('ACCEL_5_price_trend', pd.Series(0.0, index=df_index))
        p_jerk = mtf_signals.get('JERK_5_price_trend', pd.Series(0.0, index=df_index))
        # 1. 识别动量拐点：加速度为负(下跌惯性)但Jerk为正(力量开始向上反扑)
        reversal_impulse = (p_jerk.clip(lower=0) * p_acc.clip(upper=0).abs())
        # 2. 定制归一化：Tanh 映射 (中心点0.005，模拟小概率大反转)
        trap_intensity = np.tanh(reversal_impulse / 0.005).clip(0, 1)
        # 3. 结合筹码存量：主力若锁仓，则陷阱成立度更高
        chip_stab = normalized_signals.get('chip_stability_norm', 0.5)
        final_trap_score = (trap_intensity * chip_stab).rolling(window=3).mean().fillna(0)
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"[TRAP_PROBE] Impulse: {reversal_impulse.iloc[-1]:.6f}, Tanh_Score: {trap_intensity.iloc[-1]:.4f}")
        return final_trap_score

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
        【V5.1 · 参数修复与韧性增强版】计算控制力分数
        修改说明：修复缺失raw_signals导致的NameError。采用非对称Sigmoid归一化，量化主力高位锁仓的控盘韧性。
        版本号：2026.02.10.231
        """
        # 1. 提取物理分量并强制类型转换
        chip_stab = pd.Series(raw_signals.get('chip_stability', 0.5), index=df_index)
        accumulation = pd.Series(raw_signals.get('behavior_accumulation', 0.5), index=df_index)
        # 2. 控盘基础分与非对称 Sigmoid 增强
        # 逻辑：中心点设为0.6，强化深度控盘区的识别斜率
        base_ctrl = (chip_stab * 0.6 + accumulation * 0.4).clip(0, 1)
        ctrl_norm = 1 / (1 + np.exp(-12 * (base_ctrl - 0.6)))
        # 3. 控盘历史累积 (Control-HAB)
        hab_inc = ctrl_norm * accumulation
        ctrl_hab = hab_inc.rolling(34).rank(pct=True).fillna(0.5)
        # 4. 最终合成
        score = (ctrl_norm * 0.7 + ctrl_hab * 0.3).clip(0, 1)
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"[CONTROL_PROBE] Sigmoid_Ctrl: {ctrl_norm.iloc[-1]:.4f}, Control_HAB: {ctrl_hab.iloc[-1]:.4f}")
        return score.astype(np.float32)

    def _calculate_obstacle_clearance_score(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], historical_context: Dict[str, Any]) -> pd.Series:
        """
        【V6.1 · 签名对齐与Jerk穿透版】计算障碍清除分数
        修改说明：补齐raw_signals参数签名。利用Volume-Jerk识别主力穿透筹码压力区时的平滑度与效率。
        版本号：2026.02.10.232
        """
        # 1. 提取物理分量
        abs_change = pd.Series(raw_signals.get('absolute_change_strength', 0.0), index=df_index)
        vol_ratio = pd.Series(raw_signals.get('volume_ratio', 1.0), index=df_index)
        p_acc = pd.Series(mtf_signals.get('ACCEL_5_price_trend', 0.0), index=df_index)
        v_jerk = pd.Series(mtf_signals.get('JERK_5_volume_trend', 0.0), index=df_index).abs()
        # 2. 物理穿透效率 (Penetration Efficiency)
        # 逻辑：单位成交量驱动的价格加速度，并受成交量加加速度(Jerk)的惩罚，过滤暴力对敲。
        raw_efficiency = (p_acc.clip(lower=0) / (vol_ratio + 0.1))
        jerk_penalty = np.tanh(v_jerk * 10)
        clearing_intensity = (raw_efficiency * (1 - jerk_penalty * 0.4)).clip(0, 1)
        # 3. 构建 Obstacle-HAB (障碍清除存量)
        chip_stab = pd.Series(raw_signals.get('chip_stability', 0.5), index=df_index)
        hab_inc = clearing_intensity * chip_stab
        hab_buffer = hab_inc.rolling(window=34, min_periods=5).sum().fillna(0)
        # 4. 定制归一化：Rolling Rank (55日排位)
        hab_score = hab_buffer.rolling(55).rank(pct=True).fillna(0.5)
        # 5. 最终合成
        final_score = (clearing_intensity * 0.4 + hab_score * 0.6).clip(0, 1)
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"[CLEARANCE_PROBE] Volume-Jerk: {v_jerk.iloc[-1]:.4f}, HAB_Rank: {hab_score.iloc[-1]:.4f}")
        return final_score.astype(np.float32)

    def _synthesize_bullish_intent(self, df_index: pd.Index, aggressiveness_score: pd.Series, control_score: pd.Series, obstacle_clearance_score: pd.Series, mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], dynamic_weights: Dict[str, pd.Series], historical_context: Dict[str, Any], params: Dict) -> pd.Series:
        """
        【V6.0 · 对数几何合成版】主力多头意图深度合成
        修改说明：在内部实现稳健的对数空间几何平均算法，弃用通用工具，确保攻击、控制、障碍三维度在高度共振时才释放强信号。
        版本号：2026.02.10.132
        """
        # 1. 对数空间几何平均合成基础意图 (否决制合成)
        comps = {'agg': aggressiveness_score.clip(1e-6, 1), 'ctrl': control_score.clip(1e-6, 1), 'obs': obstacle_clearance_score.clip(1e-6, 1)}
        w = {'agg': dynamic_weights.get('aggressiveness', 0.35), 'ctrl': dynamic_weights.get('control', 0.35), 'obs': dynamic_weights.get('obstacle_clearance', 0.30)}
        log_intent = (w['agg'] * np.log(comps['agg']) + w['ctrl'] * np.log(comps['ctrl']) + w['obs'] * np.log(comps['obs']))
        bullish_base = np.exp(log_intent).clip(0, 1)
        # 2. 增强因子定制归一化 (Tanh)
        res_score = self._calculate_acceleration_resonance(df_index, mtf_signals)
        trap_score = self._detect_bear_trap(df_index, mtf_signals, normalized_signals)
        enhancement = np.tanh(res_score * 0.4 + trap_score * 0.6).clip(0, 1)
        # 3. 结合历史记忆 HAB (Rolling Rank)
        int_mem = historical_context.get('integrated_memory', pd.Series(0.5, index=df_index))
        mem_rank = int_mem.rolling(55).rank(pct=True).fillna(0.5)
        # 4. 最终意图合成
        final_intent = (bullish_base * (1 + enhancement * 0.2) * (0.9 + mem_rank * 0.2)).clip(0, 1)
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"[BULL_PROBE] Base_Geom: {bullish_base.iloc[-1]:.4f}, Enhance: {enhancement.iloc[-1]:.4f}, Final: {final_intent.iloc[-1]:.4f}")
        return final_intent

    def _calculate_bearish_intent(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], historical_context: Dict[str, Any]) -> pd.Series:
        """
        【V5.0 · 派发动力学与风险阶跃版】计算复合看跌意图
        修改说明：引入派发行为的S/A/J矢量与Distribution-HAB，捕捉撤退力量的突发阶跃。
        版本号：2026.02.10.84
        """
        dist_score = normalized_signals.get('distribution_score_norm', pd.Series(0.0, index=df_index))
        # 1. 派发运动学分析
        dist_slope = dist_score.diff(1).rolling(3).mean().fillna(0)
        dist_accel = dist_slope.diff(1).fillna(0)
        dist_jerk = dist_accel.diff(1).abs().fillna(0)
        # 2. 派发存量 (Distribution-HAB)
        hab_inc = dist_score * (1 + dist_accel.clip(lower=0))
        dist_hab = normalize_score(hab_inc.rolling(window=21).sum().fillna(0), target_index=df_index, windows=34)
        # 3. 恐慌冲击感应 (Panic Impulse)
        price_accel_down = mtf_signals.get('ACCEL_5_price_trend', pd.Series(0, index=df_index)).clip(upper=0).abs()
        panic_impulse = (price_accel_down * dist_accel.clip(lower=0)).pow(0.5).fillna(0)
        # 4. 最终合成
        bear_comp = {'dist': dist_score, 'hab': dist_hab, 'jerk': dist_jerk, 'panic': panic_impulse}
        score_raw = _robust_geometric_mean(bear_comp, {'dist': 0.25, 'hab': 0.35, 'jerk': 0.2, 'panic': 0.2}, df_index)
        final_bear_score = (score_raw * (1 + (dist_hab > 0.8).astype(float) * dist_jerk * 0.5)).clip(0, 1)
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"--- Bearish Physics Probe ---")
            self._probe_print(f"  > Dist-HAB Level: {dist_hab.iloc[-1]:.4f} | Panic: {panic_impulse.iloc[-1]:.4f}")
        return -final_bear_score.astype(np.float32)

    def _adjudicate_risk(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], dynamic_weights: Dict[str, pd.Series], aggressiveness_score: pd.Series, params: Dict) -> pd.Series:
        """
        【V8.1 · 底气偏差熔断版】深度风险裁决逻辑
        修改说明：新增“底气负偏差”风险项。当资金HAB显著低于价格HAB时，引入Jerk加速惩罚，识别并熔断缩量虚拉风险。
        版本号：2026.02.10.191
        """
        # 1. 基础风险分量
        tech_risk = (pd.Series(normalized_signals.get('RSI_norm', 0.5), index=df_index) - 0.7).clip(lower=0) * 2.0
        struct_risk = pd.Series(normalized_signals.get('distribution_score_norm', 0.0), index=df_index)
        # 2. 物理运动学风险 (Jerk 识别力量突变)
        p_jerk = pd.Series(mtf_signals.get('JERK_5_price_trend', 0.0), index=df_index).abs()
        # 3. 【核心新增】底气偏差风险 (Hollow Rally Risk)
        # 逻辑：价格水位 P_HAB 远高于资金水位 C_HAB 时，风险系数非线性跳升
        historical_hc = getattr(self, '_last_hc', {}) # 假设从历史上下文传递，若无则使用默认
        p_hab = pd.Series(normalized_signals.get('absolute_change_strength_norm', 0.5), index=df_index).rolling(21).mean()
        c_hab = pd.Series(normalized_signals.get('net_mf_amount_norm', 0.5), index=df_index).rolling(21).mean()
        hab_offset = (p_hab - c_hab).clip(lower=0)
        hollow_risk = (hab_offset ** 2) * (1 + p_jerk * 5)
        # 4. 专用归一化：指数级惩罚映射
        total_risk_raw = (tech_risk * 0.2 + struct_risk * 0.3 + hollow_risk * 0.5).clip(0, 1)
        # 风险超过 0.5 后加速响应
        penalty = np.where(total_risk_raw > 0.5, total_risk_raw ** 0.5, total_risk_raw ** 2)
        risk_sensitivity = get_param_value(params.get('risk_sensitivity'), 5.0)
        final_penalty = 1 / (1 + np.exp(-risk_sensitivity * (penalty - 0.55)))
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"[RISK_ADJ_PROBE] Hollow_Risk: {hollow_risk.iloc[-1]:.4f}, Offset: {hab_offset.iloc[-1]:.4f}")
        return final_penalty.clip(0, 1)

    def _apply_contextual_modulators(self, df_index: pd.Index, final_rally_intent: pd.Series, proxy_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series]) -> pd.Series:
        """
        【V3.0 · 安全缓冲映射版】应用情境调节器
        修改说明：废弃通用工具，在内部实现基于Logistic函数的安全映射，确保各项情境调节系数在合理物理区间内波动。
        版本号：2026.02.10.123
        """
        def _safe_mod(s, center=0.5, scope=0.15):
            # 将 0-1 的信号映射到 [1-scope, 1+scope] 区间
            return 1.0 + (np.tanh((s - center) * 4) * scope)
        # 1. RS 与 资金调节 (核心权重)
        rs_val = proxy_signals.get('enhanced_rs_proxy', pd.Series(0.5, index=df_index))
        cap_val = proxy_signals.get('enhanced_capital_proxy', pd.Series(0.5, index=df_index))
        # 2. 情绪与流动性调节 (次要权重)
        sent_val = proxy_signals.get('enhanced_sentiment_proxy', pd.Series(0.5, index=df_index))
        liq_val = proxy_signals.get('enhanced_liquidity_proxy', pd.Series(0.5, index=df_index))
        # 3. 组合调节因子
        modulator = _safe_mod(rs_val) * _safe_mod(cap_val, scope=0.1) * _safe_mod(sent_val, scope=0.05) * _safe_mod(liq_val, scope=0.05)
        # 4. 最终意图修正
        final_intent = (final_rally_intent * modulator).clip(-1, 1)
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            self._probe_print(f"[MOD_PROBE] Total_Mod: {modulator.iloc[-1]:.4f}, RS_Comp: {_safe_mod(rs_val).iloc[-1]:.4f}")
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