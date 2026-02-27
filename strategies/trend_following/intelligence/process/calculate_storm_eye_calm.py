# strategies\trend_following\intelligence\process\calculate_storm_eye_calm.py
# 【V58.0.2】 拆单吸筹强度 已完成
import json
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
class CalculateStormEyeCalm:
    """
    【V38.0.0 · 风暴眼寂静 · 绝对因果与纳观降噪版】
    PROCESS_META_STORM_EYE_CALM
    - [真空简并修复]: 彻底修复量子隧穿模型在“零势垒、零动能”死寂态下输出满分的荒谬逻辑，引入 width_gate 强制锁定前提因果关系。
    - [动能门控机制]: 引入 activity_gate 拦截死水区的伪阳性得分。
    - [纳观本底降噪]: _smooth_max_pair 加入零点漂移抵消公式 (-np.sqrt(eps))，eps 下潜至 1e-8，彻底粉碎 0.0016 底噪。
    - [全息高斯防爆]: 巩固指数级防爆核 VETO，对高换手与暴涨标的执行 exp(-x^2) 抹杀。
    - [Minkowski 流形]: p-Norm 幂平均逻辑彻底取代几何连乘，免疫0值死锁，维持特征灰度张力。
    """
    def __init__(self, strategy_instance, helper: ProcessIntelligenceHelper):
        """【V38.0.0】初始化风暴眼核心引擎，装载全局 JSON 参数及多时间框架结构权重。"""
        self.strategy = strategy_instance
        self.helper = helper
        self.params = self.helper.params
        self.debug_params = self.helper.debug_params
        self.probe_dates = self.helper.probe_dates
        self.version = "V38.0.0"
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        self.actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V38.0.0】主控调度枢纽，构建五大特征域，执行幂平均流形融合与高斯防爆熔断，输出终极反转极值。"""
        method_name = "calculate_storm_eye_calm"
        self.last_df_index = df.index
        df_index = df.index
        params = self._get_storm_eye_calm_params(config)
        self._check_and_fill_data_existence(df, params)
        is_debug_enabled, probe_ts = self._get_debug_info(df, method_name)
        _probe_data = {}
        self._log_probe(_probe_data, "【00. 引擎运行环境 (Engine Env)】", "Engine_Version", self.version, probe_ts)
        self._log_probe(_probe_data, "【00. 引擎运行环境 (Engine Env)】", "Tensor_Fusion_Mode", "Minkowski Power Mean Manifold (p-Norm)", probe_ts)
        self._log_probe(_probe_data, "【00. 引擎运行环境 (Engine Env)】", "Manifold_Activation", "Linear C-Infinity Clamp & Quantum Tunneling", probe_ts)
        raw_data = self._get_raw_and_atomic_data(df, method_name, params, _probe_data, probe_ts)
        energy_score = self._calculate_energy_compression_component(df_index, raw_data, {}, params['energy_compression_weights'], _probe_data, probe_ts)
        volume_score = self._calculate_volume_exhaustion_component(df_index, raw_data, {}, params['volume_exhaustion_weights'], _probe_data, probe_ts)
        intent_score, intent_dict = self._calculate_main_force_covert_intent_component(df_index, raw_data, {}, params['main_force_covert_intent_weights'], {}, _probe_data, probe_ts)
        sentiment_score = self._calculate_subdued_market_sentiment_component(df_index, raw_data, params['subdued_market_sentiment_weights'], 21, 55, 1.0, 0.2, _probe_data, probe_ts)
        readiness_score = self._calculate_breakout_readiness_component(df_index, raw_data, params['breakout_readiness_weights'], _probe_data, probe_ts)
        dynamic_threshold = self._calculate_adaptive_phase_transition_threshold(df_index, raw_data, _probe_data, probe_ts)
        component_scores = {
            'energy': pd.Series(energy_score * self._calculate_fermi_dirac_gate(energy_score, threshold=dynamic_threshold, beta=12.0), index=df_index),
            'volume': pd.Series(volume_score * self._calculate_fermi_dirac_gate(volume_score, threshold=dynamic_threshold, beta=12.0), index=df_index),
            'intent': pd.Series(intent_score * self._calculate_fermi_dirac_gate(intent_score, threshold=dynamic_threshold, beta=12.0), index=df_index),
            'sentiment': pd.Series(sentiment_score * self._calculate_fermi_dirac_gate(sentiment_score, threshold=dynamic_threshold, beta=12.0), index=df_index),
            'readiness': pd.Series(readiness_score * self._calculate_fermi_dirac_gate(readiness_score, threshold=dynamic_threshold, beta=12.0), index=df_index)
        }
        self._log_probe(_probe_data, "【05. 五大核心维度 (Domains)】", "Energy (能量门控分)", component_scores['energy'], probe_ts)
        self._log_probe(_probe_data, "【05. 五大核心维度 (Domains)】", "Volume (量能门控分)", component_scores['volume'], probe_ts)
        self._log_probe(_probe_data, "【05. 五大核心维度 (Domains)】", "Intent (意图门控分)", component_scores['intent'], probe_ts)
        self._log_probe(_probe_data, "【05. 五大核心维度 (Domains)】", "Sentiment (情绪门控分)", component_scores['sentiment'], probe_ts)
        self._log_probe(_probe_data, "【05. 五大核心维度 (Domains)】", "Readiness (准备门控分)", component_scores['readiness'], probe_ts)
        final_fusion_score = self._perform_final_fusion(df_index, component_scores, raw_data, _probe_data, probe_ts)
        regulator_modulator = self._calculate_market_regulator_modulator(df_index, raw_data, params, _probe_data, probe_ts)
        raw_final_score = pd.Series(final_fusion_score * regulator_modulator, index=df_index)
        ewd_factor = self._calculate_consensus_entropy(component_scores, _probe_data, probe_ts)
        resonance_score_soft = self._calculate_custom_normalization(pd.Series(self._smooth_max_pair(raw_final_score - 0.4, 0.0), index=df_index), mode='limit_high', sensitivity=5.0)
        resonance_ewd_soft = self._calculate_custom_normalization(pd.Series(self._smooth_max_pair(ewd_factor - 0.7, 0.0), index=df_index), mode='limit_high', sensitivity=5.0)
        resonance_confirm = pd.Series(resonance_score_soft * resonance_ewd_soft, index=df_index)
        roll_sum = pd.Series(resonance_confirm.rolling(5, min_periods=1).sum().fillna(0.0), index=df_index)
        latch_multiplier = pd.Series(1.0 + 0.2 * self._calculate_custom_normalization(pd.Series(self._smooth_max_pair(roll_sum - 2.5, 0.0), index=df_index), mode='limit_high', sensitivity=3.0), index=df_index)
        latched_score = pd.Series(raw_final_score.rolling(3, min_periods=1).mean().fillna(raw_final_score) * latch_multiplier, index=df_index)
        veto_factor = self._calculate_kinetic_overflow_veto(df_index, raw_data, self._calculate_oversold_momentum_bipolarization(df_index, raw_data, _probe_data, probe_ts), _probe_data, probe_ts)
        reward_factor = self._calculate_spatio_temporal_asymmetric_reward(df_index, raw_data, resonance_confirm, _probe_data, probe_ts)
        mrkb_factor = self._calculate_mean_reversion_kinetic_bias(df_index, raw_data, _probe_data, probe_ts)
        tes_factor = self._calculate_trend_energy_shearing(df_index, raw_data, _probe_data, probe_ts)
        final_latched_score = self._c_infinity_clamp(pd.Series(latched_score * veto_factor * reward_factor * mrkb_factor * tes_factor, index=df_index), 0.0, 1.0)
        self._log_probe(_probe_data, "【08. 最终归一化输出 (Final)】", "Raw_Final_Score (原始分)", raw_final_score, probe_ts)
        self._log_probe(_probe_data, "【08. 最终归一化输出 (Final)】", "Latched_Score (锁存稳态分)", latched_score, probe_ts)
        self._log_probe(_probe_data, "【08. 最终归一化输出 (Final)】", "Final_StormEye_Score (最终破局点)", final_latched_score, probe_ts)
        if is_debug_enabled and probe_ts is not None:
            self._print_comprehensive_probe(_probe_data, probe_ts, method_name, final_latched_score)
        return final_latched_score.astype(np.float32)
    def _smooth_abs(self, series: pd.Series, eps: float = 1e-8) -> pd.Series:
        """【V38.0.0】连续流形绝对值函数，彻底肃清代码中的 np.abs()，并且扣除底噪保证过原点。"""
        if isinstance(series, (float, int)): series = pd.Series([series])
        return pd.Series(np.sqrt(np.square(series) + eps) - np.sqrt(eps), index=series.index)
    def _smooth_max_pair(self, a: pd.Series | float, b: pd.Series | float, eps: float = 1e-8) -> pd.Series:
        """【V38.0.0】双序列边界最大值平滑函数，依靠欧几里得逼近抹除空间折线，精准扣除开方偏置，实现无漂移残留。"""
        if isinstance(a, pd.Series) and isinstance(b, pd.Series): idx = a.index
        elif isinstance(a, pd.Series): idx, b = a.index, pd.Series(b, index=a.index)
        elif isinstance(b, pd.Series): idx, a = b.index, pd.Series(a, index=b.index)
        else: idx, a, b = getattr(self, 'last_df_index', []), pd.Series(a, index=getattr(self, 'last_df_index', [])), pd.Series(b, index=getattr(self, 'last_df_index', []))
        diff = a - b
        smooth_abs_diff = np.sqrt(np.square(diff) + eps) - np.sqrt(eps)
        return pd.Series(0.5 * (a + b + smooth_abs_diff), index=idx)
    def _smooth_min_pair(self, a: pd.Series | float, b: pd.Series | float, eps: float = 1e-8) -> pd.Series:
        """【V38.0.0】双序列边界最小值平滑函数。"""
        if isinstance(a, pd.Series) and isinstance(b, pd.Series): idx = a.index
        elif isinstance(a, pd.Series): idx, b = a.index, pd.Series(b, index=a.index)
        elif isinstance(b, pd.Series): idx, a = b.index, pd.Series(a, index=b.index)
        else: idx, a, b = getattr(self, 'last_df_index', []), pd.Series(a, index=getattr(self, 'last_df_index', [])), pd.Series(b, index=getattr(self, 'last_df_index', []))
        diff = a - b
        smooth_abs_diff = np.sqrt(np.square(diff) + eps) - np.sqrt(eps)
        return pd.Series(0.5 * (a + b - smooth_abs_diff), index=idx)
    def _c_infinity_clamp(self, series: pd.Series, min_val: float = 0.0, max_val: float = 1.0) -> pd.Series:
        """【V38.0.0】绝对无损的内域线性平滑钳制，终结了传统 Sigmoid 或 clip 引起的极大失真。"""
        s1 = self._smooth_max_pair(series, min_val)
        return self._smooth_min_pair(s1, max_val)
    def _power_mean_fusion(self, df_index: pd.Index, scores: List[Any], weights: List[float], p: float = 1.0) -> pd.Series:
        """【V38.0.0】Minkowski 幂平均流形融合：p=0.5(次线性容错), p=1.0(算术均值), p=2.0(均方根)。彻底粉碎0值死锁。"""
        valid_scores, valid_weights = [], []
        for s, w in zip(scores, weights):
            if isinstance(s, pd.Series): valid_scores.append(self._c_infinity_clamp(s.reindex(df_index).fillna(0.0), 0.0, 1.0))
            else: valid_scores.append(self._c_infinity_clamp(pd.Series(s, index=df_index).fillna(0.0), 0.0, 1.0))
            valid_weights.append(w)
        weight_sum = sum(valid_weights) + 1e-9
        norm_weights = [w / weight_sum for w in valid_weights]
        padded_scores = [s * 0.99 + 0.01 for s in valid_scores]
        if abs(p) < 1e-5:
            log_sum = pd.Series(0.0, index=df_index)
            for s, w in zip(padded_scores, norm_weights): log_sum += w * np.log(s)
            return self._c_infinity_clamp((np.exp(log_sum) - 0.01) / 0.99, 0.0, 1.0)
        else:
            power_sum = pd.Series(0.0, index=df_index)
            for s, w in zip(padded_scores, norm_weights): power_sum += w * (s ** p)
            return self._c_infinity_clamp(((power_sum ** (1.0 / p)) - 0.01) / 0.99, 0.0, 1.0)
    def _log_probe(self, _probe_data: Dict, category: str, key: str, value: Any, probe_ts: pd.Timestamp):
        """【V38.0.0】统一量子态探针日志沉淀器。"""
        if probe_ts is None: return
        if isinstance(value, pd.Series): val = value.loc[probe_ts] if probe_ts in value.index else np.nan
        else: val = value
        if category not in _probe_data: _probe_data[category] = {}
        _probe_data[category][key] = val
    def _print_comprehensive_probe(self, _probe_data: Dict, probe_ts: pd.Timestamp, method_name: str, final_score: pd.Series):
        """【V38.0.0】绘制格式化的黑盒剖析全息报告。"""
        print(f"\n{'='*20} [{method_name} 全链路量子探针 | {self.version}] @ {probe_ts.strftime('%Y-%m-%d')} {'='*20}")
        categories = ["【00. 引擎运行环境 (Engine Env)】", "【01. 原始核心数据 (Raw Data)】", "【02. 微积分动力学 (Kinematics)】", "【03. 时空存量缓冲 (HAB)】", "【04. 组件计算节点 (Nodes)】", "【05. 五大核心维度 (Domains)】", "【06. 最终融合参数 (Final_Fusion_Params)】", "【07. 宏观环境调节 (Environment)】", "【08. 最终归一化输出 (Final)】"]
        for category in categories:
            if category in _probe_data:
                print(f"[{category}]")
                for k, v in _probe_data[category].items():
                    if isinstance(v, (float, np.float32, np.float64)): print(f"  ├─ {k:<40}: {v:.4f}")
                    else: print(f"  ├─ {k:<40}: {v}")
        print(f"{'-'*85}")
        print(f"  >>> 破局极值最终得分: {final_score.loc[probe_ts]:.4f} <<<")
        print(f"{'='*85}\n")
    def _check_and_fill_data_existence(self, df: pd.DataFrame, params: Dict):
        """【V38.0.0】战前底座矩阵扫描，预警军械库缺失指标断层。"""
        req_signals = self._get_required_signals(params)
        missing = [c for c in req_signals if c not in df.columns]
        if missing: print(f"【{self.version} 探针警报】风暴眼基底特征断层，缺失列: {missing}。系统已启动拓扑流形安全回退机制！")
    def _apply_threshold_gate(self, series: pd.Series, window: int = 21) -> pd.Series:
        """【V38.0.0】自适应高斯白噪波动门限滤除零基微积分陷阱，防御除0溢出。"""
        noise_floor = self._smooth_max_pair(series.rolling(window=window, min_periods=5).std().ffill().fillna(1e-5), 1e-5)
        gate_strength = pd.Series(np.tanh(np.square(series / (noise_floor * 1.5 + 1e-9))), index=series.index)
        return pd.Series(series * gate_strength, index=series.index)
    def _safe_diff(self, series: pd.Series, period: int) -> pd.Series:
        """【V38.0.0】附带波函数安全验证的平滑导数算子。"""
        return self._apply_threshold_gate(series.ffill().diff(period).fillna(0.0))
    def _calculate_fermi_dirac_gate(self, score_series: pd.Series, threshold: float | pd.Series = 0.5, beta: float = 12.0) -> pd.Series:
        """【V38.0.0】费米狄拉克门限调制：引入 0.2 保底拦截的阻尼器，仅削弱不灭口。"""
        if isinstance(threshold, pd.Series): threshold = threshold.reindex(score_series.index).fillna(0.5)
        gate = 0.5 + 0.5 * np.tanh((score_series - threshold) * (beta / 2.0))
        return pd.Series(0.2 + 0.8 * gate, index=score_series.index)
    def _calculate_custom_normalization(self, series: pd.Series, mode: str, sensitivity: float = 1.0, window: int = 55, denoise: bool = False, atr_series: Optional[pd.Series] = None) -> pd.Series:
        """【V38.0.0】多模态张量映射工厂。引入 Scale-Free Z-Denoise 动态除量纲，剥离绝对数值污染。"""
        if not isinstance(series, pd.Series): series = pd.Series(float(series), index=getattr(self, 'last_df_index', []))
        series = series.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        working_series = series.copy()
        if denoise and len(working_series) >= 21:
            if atr_series is not None: 
                working_series = working_series / (atr_series + 1e-9)
            else: 
                noise_floor = self._smooth_max_pair(working_series.rolling(window=21, min_periods=5).std().ffill().fillna(1e-5), 1e-5)
                gate_strength = pd.Series(np.tanh(np.square(working_series / (noise_floor * 1.5 + 1e-9))), index=working_series.index)
                working_series = (working_series / noise_floor) * gate_strength
                sensitivity = sensitivity * 0.2
        if mode == 'limit_high': return pd.Series(np.tanh(self._smooth_max_pair(working_series * sensitivity, 0.0)), index=working_series.index)
        elif mode == 'limit_low': return pd.Series(np.exp(-self._smooth_max_pair(working_series * sensitivity, 0.0)), index=working_series.index)
        elif mode == 'negative_extreme': return pd.Series(np.tanh(self._smooth_max_pair(-working_series * sensitivity, 0.0)), index=working_series.index)
        elif mode == 'zero_focus': return pd.Series(np.exp(- np.square(working_series * sensitivity)), index=working_series.index)
        elif mode == 'relative_rank':
            roll_min = working_series.rolling(window=window, min_periods=1).min()
            roll_max = working_series.rolling(window=window, min_periods=1).max()
            return self._c_infinity_clamp((working_series - roll_min) / (roll_max - roll_min + 1e-9), 0.0, 1.0)
        return pd.Series(0.0, index=working_series.index)
    def _get_debug_info(self, df: pd.DataFrame, method_name: str) -> Tuple[bool, Optional[pd.Timestamp]]:
        """【V38.0.0】锁定指定时空维度用于回溯探针。"""
        is_debug_enabled_for_method = get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug_enabled_for_method and self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            for date in reversed(df.index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        return is_debug_enabled_for_method, probe_ts
    def _get_storm_eye_calm_params(self, config: Dict) -> Dict:
        """【V38.0.0】提取配置权重字典架构。"""
        params = get_param_value(config.get('storm_eye_calm_params'), {})
        return {
            'energy_compression_weights': get_param_value(params.get('energy_compression_weights'), {}),
            'volume_exhaustion_weights': get_param_value(params.get('volume_exhaustion_weights'), {}),
            'main_force_covert_intent_weights': get_param_value(params.get('main_force_covert_intent_weights'), {}),
            'subdued_market_sentiment_weights': get_param_value(params.get('subdued_market_sentiment_weights'), {}),
            'breakout_readiness_weights': get_param_value(params.get('breakout_readiness_weights'), {}),
            'mtf_cohesion_weights': get_param_value(params.get('mtf_cohesion_weights'), {"cohesion_score": 1.0}),
            'final_fusion_weights': get_param_value(params.get('final_fusion_weights'), {}),
            'price_calmness_modulator_params': get_param_value(params.get('price_calmness_modulator_params'), {}),
            'main_force_control_adjudicator_params': get_param_value(params.get('main_force_control_adjudicator'), {}),
            'mtf_slope_accel_weights': get_param_value(params.get('mtf_slope_accel_weights'), {}),
            'regime_modulator_params': get_param_value(params.get('regime_modulator_params'), {}),
            'mtf_cohesion_base_signals': get_param_value(params.get('mtf_cohesion_base_signals'), []),
            'sentiment_volatility_window': get_param_value(params.get('sentiment_volatility_window'), 21),
            'long_term_sentiment_window': get_param_value(params.get('long_term_sentiment_window'), 55),
            'main_force_flow_volatility_window': get_param_value(params.get('main_force_flow_volatility_window'), 21),
            'sentiment_neutral_range': get_param_value(params.get('sentiment_neutral_range'), 1.0),
            'sentiment_pendulum_neutral_range': get_param_value(params.get('sentiment_pendulum_neutral_range'), 0.2),
            'ambiguity_components_weights': get_param_value(params.get('ambiguity_components_weights'), {}),
        }
    def _get_required_signals(self, params: Dict) -> list[str]:
        """【V38.0.0】向原始数据层投射需求指令。额外加挂一致性、分形维数、主力攻击等量子高能算子源。"""
        required_signals = [
            'MA_POTENTIAL_TENSION_INDEX_D', 'MA_COHERENCE_RESONANCE_D', 'MA_POTENTIAL_COMPRESSION_RATE_D', 'BBW_21_2.0_D', 'chip_concentration_ratio_D', 'concentration_entropy_D', 'PRICE_ENTROPY_D', 'GEOM_ARC_CURVATURE_D', 'dynamic_consolidation_duration_D', 'turnover_rate_f_D', 'volume_D', 'intraday_trough_filling_degree_D', 'tick_abnormal_volume_ratio_D', 'afternoon_flow_ratio_D', 'absorption_energy_D', 'stealth_flow_ratio_D', 'tick_clustering_index_D', 'accumulation_signal_score_D', 'SMART_MONEY_HM_NET_BUY_D', 'HM_ACTIVE_TOP_TIER_D', 'net_mf_amount_D', 'profit_ratio_D', 'winner_rate_D', 'market_sentiment_score_D', 'breakout_potential_D', 'breakout_penalty_score_D', 'resistance_strength_D', 'GEOM_REG_R2_D', 'GEOM_REG_SLOPE_D', 'ATR_14_D', 'chip_stability_D', 'ADX_14_D', 'flow_impact_ratio_D', 'industry_preheat_score_D', 'industry_rank_accel_D', 'industry_strength_rank_D', 'trend_confirmation_score_D', 'main_force_activity_index_D', 'intraday_cost_center_migration_D', 'migration_convergence_ratio_D', 'tick_chip_balance_ratio_D', 'VPA_EFFICIENCY_D', 'VPA_MF_ADJUSTED_EFF_D', 'VPA_ACCELERATION_13D', 'SMART_MONEY_HM_COORDINATED_ATTACK_D', 'OCH_ACCELERATION_D', 'OCH_D', 'PDI_14_D', 'NDI_14_D', 'price_vs_ma_21_ratio_D', 'price_vs_ma_55_ratio_D', 'HM_COORDINATED_ATTACK_D', 'TURNOVER_STABILITY_INDEX_D', 'amount_D', 'HM_ACTIVE_ANY_D', 'BIAS_55_D', 'MA_ACCELERATION_EMA_55_D', 'STATE_GOLDEN_PIT_D', 'BIAS_5_D', 'MA_FAN_EFFICIENCY_D', 'RSI_13_D', 'close', 'MA_144_D', 'chip_entropy_D', 'pressure_trapped_D', 'consolidation_quality_score_D', 'net_energy_flow_D', 'intraday_chip_game_index_D', 'pattern_confidence_D', 'breakout_quality_score_D', 'breakout_chip_score_D', 'MA_55_D', 'MA_21_D', 'MA_5_D', 'close_D',
            'volatility_adjusted_concentration_D', 'chip_convergence_ratio_D', 'PRICE_FRACTAL_DIM_D', 'MACDh_13_34_8_D', 'T1_PREMIUM_EXPECTATION_D', 'flow_consistency_D',
            'price_flow_divergence_D', 'buy_elg_amount_D', 'sell_elg_amount_D', 'high_freq_flow_skewness_D'
        ]
        return list(set(required_signals))
    def _get_raw_and_atomic_data(self, df: pd.DataFrame, method_name: str, params: Dict, _probe_data: Dict, probe_ts: pd.Timestamp) -> Dict[str, pd.Series]:
        """【V38.0.0】原子特征清洗及中性值赋予，将容易断层的防御性指标显式填入拉普拉斯缓冲。"""
        df_index = df.index
        neutral_fills = {
            'profit_ratio_D': 50.0, 'winner_rate_D': 50.0, 'market_sentiment_score_D': 50.0, 'ADX_14_D': 20.0, 
            'intraday_chip_game_index_D': 50.0, 'consolidation_quality_score_D': 50.0, 'PRICE_ENTROPY_D': 0.5, 
            'VPA_EFFICIENCY_D': 0.0, 'turnover_rate_f_D': 0.0, 'pattern_confidence_D': 0.0, 'breakout_quality_score_D': 0.0,
            'volatility_adjusted_concentration_D': 0.0, 'chip_convergence_ratio_D': 0.5, 'PRICE_FRACTAL_DIM_D': 1.5,
            'MACDh_13_34_8_D': 0.0, 'T1_PREMIUM_EXPECTATION_D': 0.0, 'flow_consistency_D': 0.0, 'flow_impact_ratio_D': 0.0,
            'price_flow_divergence_D': 0.0, 'buy_elg_amount_D': 0.0, 'sell_elg_amount_D': 0.0, 'high_freq_flow_skewness_D': 0.0,
            'dynamic_consolidation_duration_D': 0.0, 'resistance_strength_D': 0.0, 'MA_POTENTIAL_COMPRESSION_RATE_D': 0.0,
            'MA_POTENTIAL_TENSION_INDEX_D': 0.0, 'GEOM_REG_R2_D': 0.5, 'GEOM_REG_SLOPE_D': 0.0, 'ATR_14_D': 0.05,
            'net_mf_amount_D': 0.0, 'net_energy_flow_D': 0.0, 'amount_D': 1.0, 'volume_D': 1.0,
            'SMART_MONEY_HM_COORDINATED_ATTACK_D': 0.0, 'HM_ACTIVE_TOP_TIER_D': 0.0, 'HM_ACTIVE_ANY_D': 0.0
        }
        raw_data = {col: df.get(col, pd.Series(neutral_fills.get(col, 0.0), index=df_index)).ffill().fillna(neutral_fills.get(col, 0.0)) for col in self._get_required_signals(params)}
        raw_data['close_D'] = df.get('close_D', df.get('close', pd.Series(0.0, index=df_index))).ffill().fillna(0.0)
        close_base = raw_data['close_D'] + 1e-9
        amount_ma21 = raw_data['amount_D'].rolling(21, min_periods=1).mean() + 1e-9
        vol_ma21 = raw_data['volume_D'].rolling(21, min_periods=1).mean() + 1e-9
        raw_data['amount_D'] = pd.Series(raw_data['amount_D'] / amount_ma21, index=df_index)
        raw_data['volume_D'] = pd.Series(raw_data['volume_D'] / vol_ma21, index=df_index)
        raw_data['net_energy_flow_D'] = pd.Series(raw_data['net_energy_flow_D'] / amount_ma21, index=df_index)
        raw_data['net_mf_amount_D'] = pd.Series(raw_data['net_mf_amount_D'] / amount_ma21, index=df_index)
        raw_data['ATR_14_D'] = pd.Series(raw_data['ATR_14_D'] / close_base, index=df_index)
        raw_data['GEOM_REG_SLOPE_D'] = pd.Series(raw_data['GEOM_REG_SLOPE_D'] / close_base, index=df_index)
        raw_data['MACDh_13_34_8_D'] = pd.Series(raw_data['MACDh_13_34_8_D'] / close_base, index=df_index)
        raw_data['price_vs_ma_21_ratio_D'] = df.get('price_vs_ma_21_ratio_D', raw_data['close_D'] / (df.get('MA_21_D', raw_data['close_D']) + 1e-9)).ffill().fillna(1.0)
        raw_data['price_vs_ma_55_ratio_D'] = df.get('price_vs_ma_55_ratio_D', raw_data['close_D'] / (df.get('MA_55_D', raw_data['close_D']) + 1e-9)).ffill().fillna(1.0)
        raw_data['BIAS_55_D'] = pd.Series((raw_data['close_D'] - df.get('MA_55_D', raw_data['close_D'])) / (df.get('MA_55_D', raw_data['close_D']) + 1e-9), index=df_index)
        raw_data['BIAS_5_D'] = pd.Series((raw_data['close_D'] - df.get('MA_5_D', raw_data['close_D'])) / (df.get('MA_5_D', raw_data['close_D']) + 1e-9), index=df_index)
        raw_data['price_slope_raw'] = pd.Series(raw_data['close_D'].pct_change(5).replace([np.inf, -np.inf], 0.0).fillna(0.0), index=df_index)
        scale_100_cols = ['turnover_rate_f_D', 'pattern_confidence_D', 'breakout_quality_score_D', 'breakout_chip_score_D', 'consolidation_quality_score_D', 'accumulation_signal_score_D', 'intraday_chip_game_index_D', 'tick_clustering_index_D', 'NDI_14_D', 'PDI_14_D', 'RSI_13_D', 'ADX_14_D', 'winner_rate_D', 'profit_ratio_D', 'HM_ACTIVE_ANY_D', 'HM_ACTIVE_TOP_TIER_D', 'STATE_GOLDEN_PIT_D', 'chip_stability_D', 'intraday_trough_filling_degree_D', 'SMART_MONEY_HM_COORDINATED_ATTACK_D', 'HM_COORDINATED_ATTACK_D', 'pressure_trapped_D', 'breakout_penalty_score_D', 'market_sentiment_score_D', 'buy_elg_amount_D', 'sell_elg_amount_D', 'high_freq_flow_skewness_D', 'resistance_strength_D']
        for col in scale_100_cols:
            if col in raw_data:
                col_max = pd.Series(self._smooth_abs(raw_data[col]).rolling(window=252, min_periods=1).max().fillna(0.0), index=df_index)
                step_func = pd.Series(0.5 + 0.5 * np.tanh((col_max - 5.0) * 2.0), index=df_index)
                scale_factor = pd.Series(1.0 + 99.0 * step_func, index=df_index)
                raw_data[col] = pd.Series(raw_data[col] / scale_factor, index=df_index)
        probe_keys_raw = ['close_D', 'pattern_confidence_D', 'breakout_quality_score_D', 'breakout_chip_score_D', 'turnover_rate_f_D', 'ADX_14_D', 'VPA_EFFICIENCY_D', 'PRICE_ENTROPY_D', 'amount_D', 'ATR_14_D', 'profit_ratio_D', 'winner_rate_D', 'NDI_14_D', 'PDI_14_D', 'RSI_13_D', 'market_sentiment_score_D', 'chip_stability_D', 'stealth_flow_ratio_D', 'pressure_trapped_D', 'intraday_trough_filling_degree_D', 'intraday_chip_game_index_D', 'consolidation_quality_score_D', 'net_energy_flow_D', 'SMART_MONEY_HM_COORDINATED_ATTACK_D', 'accumulation_signal_score_D']
        for k in probe_keys_raw:
            if k in raw_data: self._log_probe(_probe_data, "【01. 原始核心数据 (Raw Data)】", k, raw_data[k], probe_ts)
        deriv_cols = ['VPA_ACCELERATION_13D', 'VPA_MF_ADJUSTED_EFF_D', 'tick_abnormal_volume_ratio_D', 'MA_ACCELERATION_EMA_55_D', 'PRICE_ENTROPY_D', 'STATE_GOLDEN_PIT_D', 'BIAS_55_D', 'NDI_14_D', 'PDI_14_D', 'breakout_penalty_score_D', 'RSI_13_D', 'OCH_D', 'ATR_14_D', 'MA_FAN_EFFICIENCY_D', 'HM_ACTIVE_ANY_D', 'SMART_MONEY_HM_COORDINATED_ATTACK_D', 'HM_COORDINATED_ATTACK_D', 'BIAS_5_D', 'market_sentiment_score_D', 'ADX_14_D', 'profit_ratio_D', 'chip_entropy_D', 'net_energy_flow_D', 'pattern_confidence_D', 'breakout_quality_score_D', 'consolidation_quality_score_D', 'OCH_ACCELERATION_D', 'VPA_EFFICIENCY_D', 'intraday_cost_center_migration_D', 'TURNOVER_STABILITY_INDEX_D', 'concentration_entropy_D', 'industry_rank_accel_D', 'flow_consistency_D', 'turnover_rate_f_D', 'price_slope_raw', 'PRICE_FRACTAL_DIM_D', 'MACDh_13_34_8_D']
        for col in deriv_cols:
            if col in raw_data:
                s13 = self._safe_diff(raw_data[col], 13)
                a8 = self._safe_diff(s13, 8)
                j5 = self._safe_diff(a8, 5)
                raw_data[f'SLOPE_13_{col}'] = s13
                raw_data[f'ACCEL_8_{col}'] = a8
                raw_data[f'JERK_5_{col}'] = j5
                raw_data[f'SLOPE_5_{col}'] = self._safe_diff(raw_data[col], 5)
                if col in ['VPA_ACCELERATION_13D', 'PRICE_ENTROPY_D', 'pattern_confidence_D', 'OCH_ACCELERATION_D', 'NDI_14_D', 'RSI_13_D']:
                    self._log_probe(_probe_data, "【02. 微积分动力学 (Kinematics)】", f"SLOPE_13_{col}", s13, probe_ts)
                    self._log_probe(_probe_data, "【02. 微积分动力学 (Kinematics)】", f"JERK_5_{col}", j5, probe_ts)
        ma144 = df.get('MA_144_D', raw_data['close_D']).ffill().fillna(raw_data['close_D'])
        raw_data['price_vs_ma_144_ratio'] = pd.Series(raw_data['close_D'] / (ma144 + 1e-9), index=df_index)
        raw_data['ACCEL_8_price_vs_ma_144_ratio'] = self._safe_diff(self._safe_diff(raw_data['price_vs_ma_144_ratio'], 13), 8)
        raw_data['pain_index_proxy'] = pd.Series(1.0 - raw_data['profit_ratio_D'], index=df_index)
        raw_data['JERK_5_pain_index_proxy'] = pd.Series(raw_data.get('JERK_5_profit_ratio_D', pd.Series(0.0, index=df_index)) * -1.0, index=df_index)
        self._log_probe(_probe_data, "【02. 微积分动力学 (Kinematics)】", "price_slope_raw", raw_data['price_slope_raw'], probe_ts)
        self._log_probe(_probe_data, "【02. 微积分动力学 (Kinematics)】", "JERK_5_pain_index_proxy", raw_data['JERK_5_pain_index_proxy'], probe_ts)
        return raw_data
    def _calculate_qho_historical_accumulation_buffer(self, daily_series: pd.Series, windows: list[int] = [13, 21, 34, 55], name: str = "", _probe_data: Dict = None, probe_ts: pd.Timestamp = None) -> pd.Series:
        """【V38.0.0】历史累积记忆缓冲层：基于长期存量的物理占比度量冲击力，摒弃绝对增量。"""
        if daily_series.empty: return pd.Series(0.0, index=getattr(self, 'last_df_index', []))
        buffers = []
        for w in windows:
            historical_stock = pd.Series(self._smooth_abs(daily_series).rolling(window=w, min_periods=1).mean() + 1e-9, index=daily_series.index)
            incremental_impact = daily_series / historical_stock
            buffer_factor = pd.Series(np.tanh(self._smooth_max_pair(incremental_impact, 0.0) * 0.5), index=daily_series.index)
            buffers.append(buffer_factor)
        res = pd.concat(buffers, axis=1).mean(axis=1).fillna(0.0)
        if name and _probe_data is not None and probe_ts is not None:
            self._log_probe(_probe_data, "【03. 时空存量缓冲 (HAB)】", f"HAB_{name}", res, probe_ts)
        return res
    def _calculate_ornstein_uhlenbeck_pull(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】物理模型：Ornstein-Uhlenbeck 随机均值回复方程引力场。"""
        close = raw_data.get('close_D', pd.Series(1.0, index=df_index))
        mu_55 = raw_data.get('MA_55_D', close)
        sigma = raw_data.get('ATR_14_D', pd.Series(0.01, index=df_index)) + 1e-9
        ou_pull_raw = pd.Series((mu_55 - close) / (close * sigma), index=df_index)
        ou_pull_positive = pd.Series(self._smooth_max_pair(ou_pull_raw, 0.0), index=df_index)
        ou_pull_score = self._calculate_custom_normalization(ou_pull_positive, mode='limit_high', sensitivity=2.0)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "OU_Mean_Reversion (OU均值回复拉力)", ou_pull_score, probe_ts)
        return ou_pull_score
    def _calculate_phase_space_divergence(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】物理模型：高频相空间三维散度坍缩。采用独立滚动度量衡归一以防量纲吞噬。"""
        dp_dt = raw_data.get('SLOPE_13_price_slope_raw', pd.Series(0.0, index=df_index))
        dv_dt = raw_data.get('SLOPE_13_turnover_rate_f_D', pd.Series(0.0, index=df_index))
        df_dt = raw_data.get('SLOPE_13_flow_consistency_D', pd.Series(0.0, index=df_index))
        dp_norm = dp_dt / (self._smooth_max_pair(self._smooth_abs(dp_dt).rolling(21, min_periods=1).mean(), 1e-5))
        dv_norm = dv_dt / (self._smooth_max_pair(self._smooth_abs(dv_dt).rolling(21, min_periods=1).mean(), 1e-5))
        df_norm = df_dt / (self._smooth_max_pair(self._smooth_abs(df_dt).rolling(21, min_periods=1).mean(), 1e-5))
        divergence = pd.Series(dp_norm + dv_norm + df_norm, index=df_index)
        contraction_score = self._calculate_custom_normalization(divergence, mode='zero_focus', sensitivity=0.5)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Phase_Divergence (相空间散度塌缩)", contraction_score, probe_ts)
        return contraction_score
    def _calculate_quantum_tunneling_probability(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】物理模型：薛定谔势垒穿透概率优化。久盘必破：盘整时间越长，等效物理势垒越薄。"""
        res_strength = raw_data.get('resistance_strength_D', pd.Series(0.0, index=df_index))
        bias_55_neg = self._smooth_max_pair(1.0 - raw_data.get('price_vs_ma_55_ratio_D', pd.Series(1.0, index=df_index)), 0.0)
        barrier_height = pd.Series(res_strength + bias_55_neg * 2.0, index=df_index)
        width_raw = raw_data.get('dynamic_consolidation_duration_D', pd.Series(0.0, index=df_index))
        barrier_attenuation = pd.Series(1.0 / (1.0 + np.log1p(width_raw)), index=df_index)
        effective_barrier = pd.Series(barrier_height * barrier_attenuation, index=df_index)
        net_mf_norm = self._calculate_custom_normalization(raw_data.get('net_mf_amount_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=2.0)
        vpa_accel_norm = self._calculate_custom_normalization(raw_data.get('VPA_ACCELERATION_13D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=5.0, denoise=True)
        kinetic_e = self._smooth_max_pair(net_mf_norm + vpa_accel_norm, 0.0)
        energy_deficit = self._smooth_max_pair(effective_barrier - kinetic_e, 0.0)
        tunneling_prob = pd.Series(np.exp(-energy_deficit * 3.0), index=df_index)
        width_gate = self._c_infinity_clamp(width_raw / 21.0, 0.0, 1.0)
        tunnel_score = pd.Series(tunneling_prob * width_gate, index=df_index)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Quantum_Tunneling (势垒隧穿价值)", tunnel_score, probe_ts)
        return tunnel_score
    def _calculate_breakout_conviction_proxy(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】防线爆破信念代理解算引擎。"""
        pattern_conf = self._calculate_custom_normalization(raw_data.get('pattern_confidence_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=1.5)
        pattern_slope = self._calculate_custom_normalization(raw_data.get('SLOPE_13_pattern_confidence_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=2.0, denoise=True)
        breakout_qual = self._calculate_custom_normalization(raw_data.get('breakout_quality_score_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=1.5)
        breakout_chip = self._calculate_custom_normalization(raw_data.get('breakout_chip_score_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=2.0)
        qual_hab = self._calculate_qho_historical_accumulation_buffer(raw_data.get('breakout_quality_score_D', pd.Series(0.0, index=df_index)), windows=[13, 21], name="BreakoutQual", _probe_data=_probe_data, probe_ts=probe_ts)
        final_conviction = self._power_mean_fusion(df_index, [pattern_conf, breakout_qual, breakout_chip, pattern_slope, qual_hab], [0.3, 0.25, 0.15, 0.15, 0.15], p=1.0)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Conviction_Proxy (突破隧穿代理)", final_conviction, probe_ts)
        return final_conviction
    def _calculate_energy_compression_component(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], mtf_derived_scores: Dict[str, pd.Series], weights: Dict, _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】第一极点网络：能量极度压缩矩阵，基于 p=0.5 次线性容错叠加。"""
        fcc_factor = self._calculate_fan_curvature_collapse(df_index, raw_data, _probe_data, probe_ts)
        vvc_factor = self._calculate_volatility_vacuum_contraction(df_index, raw_data, _probe_data, probe_ts)
        lrf_score = self._calculate_linear_resonance_failure(df_index, raw_data, _probe_data, probe_ts)
        struct_quality = self._calculate_custom_normalization(raw_data.get('chip_stability_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=2.0)
        entropy_gain = self._calculate_custom_normalization(raw_data.get('SLOPE_13_chip_entropy_D', pd.Series(0.0, index=df_index)), mode='negative_extreme', sensitivity=5.0, denoise=True)
        vpa_accel = raw_data.get('ACCEL_8_VPA_ACCELERATION_13D', pd.Series(0.0, index=df_index))
        vpa_jerk = raw_data.get('JERK_5_VPA_ACCELERATION_13D', pd.Series(0.0, index=df_index))
        phase_space_dist = pd.Series(np.sqrt(np.square(vpa_accel) + np.square(vpa_jerk)), index=df_index)
        phase_attractor = pd.Series(np.exp(-phase_space_dist * 2.0), index=df_index)
        phase_div = self._calculate_phase_space_divergence(df_index, raw_data, _probe_data, probe_ts)
        ma_comp_raw = raw_data.get('MA_POTENTIAL_COMPRESSION_RATE_D', pd.Series(0.0, index=df_index))
        ma_comp_hab = self._calculate_qho_historical_accumulation_buffer(ma_comp_raw, windows=[13, 21, 34, 55], name="MA_Comp", _probe_data=_probe_data, probe_ts=probe_ts)
        ma_comp_score = self._calculate_custom_normalization(ma_comp_raw, mode='limit_high', sensitivity=2.0)
        bbw_raw = raw_data.get('BBW_21_2.0_D', pd.Series(0.0, index=df_index))
        bbw_score = self._calculate_custom_normalization(bbw_raw, mode='limit_low', sensitivity=5.0)
        tension_raw = raw_data.get('MA_POTENTIAL_TENSION_INDEX_D', pd.Series(0.0, index=df_index))
        tension_score = self._calculate_custom_normalization(tension_raw, mode='limit_high', sensitivity=2.0)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "FCC_Score (扇面曲率塌缩)", fcc_factor, probe_ts)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "VVC_Score (波动率真空态)", vvc_factor, probe_ts)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "LRF_Score (线性死寂崩塌)", lrf_score, probe_ts)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Struct_Quality (筹码稳固度)", struct_quality, probe_ts)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Entropy_Gain (熵减红利)", entropy_gain, probe_ts)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Phase_Attractor (相空间吸引子)", phase_attractor, probe_ts)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "MA_Comp_Score (均线压缩势能)", ma_comp_score, probe_ts)
        final_energy = self._power_mean_fusion(df_index, [fcc_factor, vvc_factor, lrf_score, entropy_gain, phase_attractor, struct_quality, phase_div, ma_comp_score, ma_comp_hab, bbw_score, tension_score], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.1], p=0.5)
        return final_energy
    def _calculate_volume_exhaustion_component(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], mtf_derived_scores: Dict[str, pd.Series], weights: Dict, _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】第二极点网络：绝对微观流动性枯竭域，基于 p=0.5 态叠加。引入 vpa_activity_gate 阻断无用死水。"""
        turnover_score = self._calculate_custom_normalization(raw_data.get('turnover_rate_f_D', pd.Series(0.0, index=df_index)), mode='limit_low', sensitivity=10.0)
        trough_fill = self._calculate_custom_normalization(raw_data.get('intraday_trough_filling_degree_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=3.0)
        mdb_factor = self._calculate_momentum_dissipation_balance(df_index, raw_data, _probe_data, probe_ts)
        solid_factor = self._calculate_liquidity_solidification_threshold(df_index, raw_data, _probe_data, probe_ts)
        vpa_raw = raw_data.get('VPA_EFFICIENCY_D', pd.Series(0.0, index=df_index))
        vpa_activity_gate = self._c_infinity_clamp(self._smooth_abs(vpa_raw).rolling(13, min_periods=1).mean() * 5.0, 0.0, 1.0)
        vpa_jerk = pd.Series(self._calculate_custom_normalization(raw_data.get('JERK_5_VPA_EFFICIENCY_D', pd.Series(0.0, index=df_index)), mode='zero_focus', sensitivity=5.0, denoise=True) * vpa_activity_gate, index=df_index)
        mf_eff = self._calculate_custom_normalization(raw_data.get('VPA_MF_ADJUSTED_EFF_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=2.0)
        chip_conv_score = self._calculate_custom_normalization(raw_data.get('chip_convergence_ratio_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=2.0)
        vol_adj_conc_score = self._calculate_custom_normalization(raw_data.get('volatility_adjusted_concentration_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=2.0)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Turnover_Score (极小换手得分)", turnover_score, probe_ts)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Trough_Fill (日内波谷填充)", trough_fill, probe_ts)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "MDB_Factor (动量耗散平衡)", mdb_factor, probe_ts)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Solid_Factor (流动性固化)", solid_factor, probe_ts)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "VPA_Jerk (量价加速度归零)", vpa_jerk, probe_ts)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "MF_Eff (主力修正效率)", mf_eff, probe_ts)
        final_vol = self._power_mean_fusion(df_index, [turnover_score, trough_fill, mdb_factor, solid_factor, vpa_jerk, mf_eff, chip_conv_score, vol_adj_conc_score], [0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15], p=0.5)
        return final_vol
    def _calculate_main_force_covert_intent_component(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], mtf_derived_scores: Dict[str, pd.Series], weights: Dict, ambiguity_weights: Dict, _probe_data: Dict, probe_ts: pd.Timestamp) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """【V38.0.0】第三极点网络：主力意图防守雷达。采用 p=2.0 均方根Soft-OR，一点击穿即全体点燃。"""
        chf_base = raw_data.get('SMART_MONEY_HM_COORDINATED_ATTACK_D', pd.Series(0.0, index=df_index)).rolling(window=8, min_periods=1).mean().fillna(0.0)
        chf_jerk_score = self._calculate_custom_normalization(raw_data.get('JERK_5_SMART_MONEY_HM_COORDINATED_ATTACK_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=5.0, denoise=True)
        htc_factor = self._calculate_hunting_temporal_coherence(df_index, raw_data, _probe_data, probe_ts)
        stealth_score = self._calculate_custom_normalization(raw_data.get('stealth_flow_ratio_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=5.0)
        migration_accel = self._calculate_custom_normalization(raw_data.get('ACCEL_8_intraday_cost_center_migration_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=5.0)
        mf_hab = self._calculate_qho_historical_accumulation_buffer(raw_data.get('net_mf_amount_D', pd.Series(0.0, index=df_index)), windows=[21, 34], name="Net_MF", _probe_data=_probe_data, probe_ts=probe_ts)
        energy_flow = self._calculate_custom_normalization(raw_data.get('net_energy_flow_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=2.0)
        flow_cons = raw_data.get('flow_consistency_D', pd.Series(0.0, index=df_index))
        flow_hab = self._calculate_qho_historical_accumulation_buffer(flow_cons, windows=[13, 21, 34, 55], name="Flow_Cons", _probe_data=_probe_data, probe_ts=probe_ts)
        flow_cons_score = self._calculate_custom_normalization(flow_cons, mode='limit_high', sensitivity=2.0)
        absorption_raw = raw_data.get('absorption_energy_D', pd.Series(0.0, index=df_index))
        absorption_score = self._calculate_custom_normalization(absorption_raw, mode='limit_high', sensitivity=2.0)
        elg_buy = self._calculate_custom_normalization(raw_data.get('buy_elg_amount_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=1.0)
        elg_sell = self._calculate_custom_normalization(raw_data.get('sell_elg_amount_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=1.0)
        elg_attack = self._smooth_max_pair(elg_buy - elg_sell, 0.0)
        flow_div_score = self._calculate_custom_normalization(raw_data.get('price_flow_divergence_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=3.0)
        hf_skew = raw_data.get('high_freq_flow_skewness_D', pd.Series(0.0, index=df_index))
        hf_skew_score = self._calculate_custom_normalization(hf_skew, mode='limit_high', sensitivity=2.0)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Stealth_Score (隐秘潜行占比)", stealth_score, probe_ts)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Migration_Accel (筹码跃迁加速)", migration_accel, probe_ts)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "CHF_Base (协同攻击基准)", chf_base, probe_ts)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "CHF_Jerk (协同攻击突变)", chf_jerk_score, probe_ts)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Energy_Flow (净能量流动)", energy_flow, probe_ts)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "HTC_Factor (猎杀一致性)", htc_factor, probe_ts)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "MF_HAB (主力存量势能)", mf_hab, probe_ts)
        final_intent = self._power_mean_fusion(df_index, [stealth_score, migration_accel, chf_base, chf_jerk_score, energy_flow, htc_factor, mf_hab, flow_cons_score, flow_hab, absorption_score, elg_attack, flow_div_score, hf_skew_score], [0.08, 0.08, 0.1, 0.08, 0.1, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.04, 0.04], p=2.0)
        return final_intent, {"stealth_score": stealth_score, "htc_factor": htc_factor}
    def _calculate_subdued_market_sentiment_component(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], weights: Dict, sentiment_volatility_window: int, long_term_sentiment_window: int, sentiment_neutral_range: float, sentiment_pendulum_neutral_range: float, _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】第四极点网络：极致冰点情绪探针网。基于 p=1.0 算术平权制约盲目乐观。"""
        pain_score = self._calculate_custom_normalization(raw_data.get('pain_index_proxy', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=3.0)
        despair_burst = self._calculate_custom_normalization(raw_data.get('JERK_5_pain_index_proxy', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=5.0, denoise=True)
        short_exhaustion = self._calculate_short_exhaustion_divergence(df_index, raw_data, _probe_data, probe_ts)
        bipolar_gain = self._calculate_oversold_momentum_bipolarization(df_index, raw_data, _probe_data, probe_ts)
        panic_resonance = self._calculate_extreme_panic_resonance(df_index, raw_data, _probe_data, probe_ts)
        order_gain = self._calculate_micro_order_gain(df_index, raw_data, _probe_data, probe_ts)
        cleanse_score = self._calculate_custom_normalization(raw_data.get('winner_rate_D', pd.Series(0.0, index=df_index)), mode='limit_low', sensitivity=5.0)
        trapped_pressure = self._calculate_custom_normalization(raw_data.get('pressure_trapped_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=2.0)
        macd_h = raw_data.get('MACDh_13_34_8_D', pd.Series(0.0, index=df_index))
        macd_calm = self._calculate_custom_normalization(macd_h, mode='zero_focus', sensitivity=5.0, denoise=True)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Pain_Score (散户痛感释放)", pain_score, probe_ts)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Despair_Burst (绝望极限突变)", despair_burst, probe_ts)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Short_Exhaustion (空头抛压耗尽)", short_exhaustion, probe_ts)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Bipolar_Gain (动能二极化极值)", bipolar_gain, probe_ts)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Panic_Resonance (恐慌深渊共振)", panic_resonance, probe_ts)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Order_Gain (微观有序增益)", order_gain, probe_ts)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Cleanse_Score (浮筹清洗度)", cleanse_score, probe_ts)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Trapped_Pressure (套牢压迫感)", trapped_pressure, probe_ts)
        final_sentiment = self._power_mean_fusion(df_index, [pain_score, cleanse_score, trapped_pressure, order_gain, short_exhaustion, bipolar_gain, panic_resonance, despair_burst, macd_calm], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.1, 0.15], p=1.0)
        return final_sentiment
    def _calculate_breakout_readiness_component(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], weights: Dict, _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】第五极点网络：多维结构突破基底支撑网，已全面注入修复版薛定谔量子隧穿机制。"""
        grp_score = self._calculate_gravitational_regression_pull(df_index, raw_data, _probe_data, probe_ts)
        plr_score = self._calculate_phase_locked_resonance(df_index, raw_data, _probe_data, probe_ts)
        sed_score = self._calculate_short_exhaustion_divergence(df_index, raw_data, _probe_data, probe_ts)
        ssd_score = self._calculate_seat_scatter_decay(df_index, raw_data, _probe_data, probe_ts)
        egd_score = self._calculate_efficiency_gradient_dissipation(df_index, raw_data, _probe_data, probe_ts)
        sope_score = self._calculate_split_order_pulse_entropy(df_index, raw_data, _probe_data, probe_ts)
        aeo_score = self._calculate_abnormal_energy_overflow(df_index, raw_data, _probe_data, probe_ts)
        neutral_score = self._calculate_game_neutralization_modulator(df_index, raw_data, _probe_data, probe_ts)
        aded_score = self._calculate_amount_distribution_entropy_delta(df_index, raw_data, _probe_data, probe_ts)
        well_collapse = self._calculate_potential_well_collapse(df_index, raw_data, _probe_data, probe_ts)
        long_awakening = self._calculate_long_awakening_threshold(df_index, raw_data, _probe_data, probe_ts)
        stress_test = self._calculate_level_stress_test_modulator(df_index, raw_data, _probe_data, probe_ts)
        consolidation = self._calculate_custom_normalization(raw_data.get('consolidation_quality_score_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=2.0)
        consolidation_duration = self._calculate_custom_normalization(raw_data.get('dynamic_consolidation_duration_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=0.05)
        conviction_proxy = self._calculate_breakout_conviction_proxy(df_index, raw_data, _probe_data, probe_ts)
        fractal_accel = raw_data.get('ACCEL_8_PRICE_FRACTAL_DIM_D', pd.Series(0.0, index=df_index))
        fractal_drop = self._calculate_custom_normalization(fractal_accel, mode='negative_extreme', sensitivity=5.0, denoise=True)
        quantum_tunnel = self._calculate_quantum_tunneling_probability(df_index, raw_data, _probe_data, probe_ts)
        momentum_part = self._power_mean_fusion(df_index, [grp_score, plr_score], [0.5, 0.5], p=1.0)
        friction_part = self._power_mean_fusion(df_index, [sed_score, ssd_score], [0.5, 0.5], p=1.0)
        quality_part = self._power_mean_fusion(df_index, [egd_score, sope_score, aeo_score, consolidation_duration], [0.3, 0.25, 0.25, 0.2], p=1.0)
        state_part = self._power_mean_fusion(df_index, [well_collapse, long_awakening, aded_score, stress_test, neutral_score, consolidation, conviction_proxy, fractal_drop, quantum_tunnel], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.1], p=1.0)
        acc_hab = self._calculate_qho_historical_accumulation_buffer(raw_data.get('accumulation_signal_score_D', pd.Series(0.0, index=df_index)), windows=[21, 34], name="Accum_Signal", _probe_data=_probe_data, probe_ts=probe_ts)
        readiness = self._power_mean_fusion(df_index, [momentum_part, friction_part, quality_part, state_part, acc_hab], [0.15, 0.15, 0.2, 0.4, 0.1], p=1.0)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "GRP_Score (引力回归拉力)", grp_score, probe_ts)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "PLR_Score (量价相位锁定)", plr_score, probe_ts)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "SSD_Score (跟风散点衰减)", ssd_score, probe_ts)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "EGD_Score (效率梯度耗散)", egd_score, probe_ts)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "SOPE_Score (拆单脉冲熵)", sope_score, probe_ts)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "AEO_Score (异常能量溢出)", aeo_score, probe_ts)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Neutral_Score (多空博弈中性)", neutral_score, probe_ts)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "ADED_Score (浓度分布熵跃)", aded_score, probe_ts)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Well_Collapse (坑底逃逸塌缩)", well_collapse, probe_ts)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Long_Awakening (多头蛰伏觉醒)", long_awakening, probe_ts)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Consolidation (盘整无懈可击)", consolidation, probe_ts)
        return readiness
    def _perform_final_fusion(self, df_index: pd.Index, component_scores: dict[str, pd.Series], raw_data: dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】张量总线聚合输出站。"""
        scores_list = [component_scores['energy'], component_scores['volume'], component_scores['intent'], component_scores['sentiment'], component_scores['readiness']]
        base_score = self._power_mean_fusion(df_index, scores_list, [0.25, 0.25, 0.15, 0.15, 0.20], p=1.0)
        ext_calm = self._calculate_custom_normalization(raw_data.get('price_slope_raw', pd.Series(0.0, index=df_index)), mode='zero_focus', sensitivity=15.0, denoise=True)
        struct_boost = self._calculate_custom_normalization(raw_data.get('accumulation_signal_score_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=1.0)
        hunting_boost = self._calculate_custom_normalization(raw_data.get('SMART_MONEY_HM_COORDINATED_ATTACK_D', pd.Series(0.0, index=df_index)).rolling(8, min_periods=1).mean().fillna(0.0), mode='limit_high', sensitivity=2.0)
        multiplier = pd.Series(1.0 + 0.2 * struct_boost + 0.15 * ext_calm + 0.15 * hunting_boost, index=df_index)
        final_score = self._c_infinity_clamp(pd.Series(base_score * multiplier, index=df_index), 0.0, 1.0)
        self._log_probe(_probe_data, "【06. 最终融合参数 (Final_Fusion_Params)】", "WGM_Base_Core_Score (基础软与合分)", base_score, probe_ts)
        self._log_probe(_probe_data, "【06. 最终融合参数 (Final_Fusion_Params)】", "Additive_Multiplier (增强乘数)", multiplier, probe_ts)
        self._log_probe(_probe_data, "【06. 最终融合参数 (Final_Fusion_Params)】", "Pre_Modulator_Score (预调节分)", final_score, probe_ts)
        return final_score
    def _calculate_market_regulator_modulator(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], params: Dict, _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】宏观乘数发电机，植入了绝对平权 p=1.0 混合器。"""
        sector_preheat = raw_data.get('industry_preheat_score_D', pd.Series(0.0, index=df_index))
        sector_hab = self._calculate_qho_historical_accumulation_buffer(sector_preheat, windows=[13, 21, 34, 55], name="Sector_Preheat", _probe_data=_probe_data, probe_ts=probe_ts)
        sector_jerk = raw_data.get('JERK_5_industry_rank_accel_D', pd.Series(0.0, index=df_index))
        clean_sector_jerk = pd.Series(self._smooth_max_pair(sector_jerk - sector_jerk.rolling(21, min_periods=1).std().fillna(0.0), 0.0), index=df_index)
        sector_ignite_score = self._calculate_custom_normalization(clean_sector_jerk, mode='limit_high', sensitivity=5.0)
        stock_calm = self._calculate_custom_normalization(raw_data.get('price_slope_raw', pd.Series(0.0, index=df_index)), mode='zero_focus', sensitivity=5.0, denoise=True)
        macro_resonance = self._power_mean_fusion(df_index, [sector_ignite_score, stock_calm], [0.5, 0.5], p=1.0)
        adx_raw = raw_data.get('ADX_14_D', pd.Series(20.0, index=df_index))
        adx_supp = pd.Series(1.0 / (1.0 + np.exp(15.0 * (adx_raw - 0.28))), index=df_index)
        t1_premium = raw_data.get('T1_PREMIUM_EXPECTATION_D', pd.Series(0.0, index=df_index))
        t1_premium_score = self._calculate_custom_normalization(t1_premium, mode='limit_high', sensitivity=10.0)
        final_modulator = self._power_mean_fusion(df_index, [sector_hab, macro_resonance, adx_supp, t1_premium_score], [0.3, 0.4, 0.15, 0.15], p=1.0)
        adj_modulator = pd.Series(final_modulator * 1.5 + 0.5, index=df_index)
        self._log_probe(_probe_data, "【07. 宏观环境调节 (Environment)】", "Market_Regulator (宏观起爆乘数)", adj_modulator, probe_ts)
        return adj_modulator
    def _calculate_trend_energy_shearing(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】无损线性趋势能量衰减剪切。"""
        adx_raw = raw_data.get('ADX_14_D', pd.Series(20.0, index=df_index))
        adx_accel = raw_data.get('ACCEL_8_ADX_14_D', pd.Series(0.0, index=df_index))
        high_context = self._calculate_custom_normalization(pd.Series(self._smooth_max_pair(adx_raw - 0.35, 0.0), index=df_index), mode='limit_high', sensitivity=20.0)
        shearing_ignite = self._calculate_custom_normalization(adx_accel, mode='negative_extreme', sensitivity=5.0, denoise=True)
        shearing_factor = pd.Series(1.0 + 0.2 * self._power_mean_fusion(df_index, [high_context, shearing_ignite], [0.5, 0.5], p=1.0), index=df_index).fillna(1.0)
        self._log_probe(_probe_data, "【07. 宏观环境调节 (Environment)】", "TES_Factor (趋势能量剪切)", shearing_factor, probe_ts)
        return shearing_factor
    def _calculate_consensus_entropy(self, scores_dict: dict[str, pd.Series], _probe_data: Dict = None, probe_ts: pd.Timestamp = None) -> pd.Series:
        """【V38.0.0】共振信息降维与互信息熵校准。引入动能门控彻底剿灭死寂期的满分伪装。"""
        df_scores = pd.concat(scores_dict.values(), axis=1)
        dispersion = df_scores.std(axis=1).fillna(1.0)
        corr_matrix = df_scores.rolling(window=5, min_periods=1).corr()
        coherence = corr_matrix.groupby(level=0).mean().mean(axis=1).fillna(0.0)
        disp_decay = pd.Series(np.exp(- (dispersion * 2.5) ** 2), index=df_scores.index)
        mean_activity = self._power_mean_fusion(df_scores.index, list(scores_dict.values()), [1.0]*len(scores_dict), p=1.0)
        activity_gate = self._c_infinity_clamp(mean_activity * 2.0, 0.0, 1.0)
        final_decay = self._c_infinity_clamp(pd.Series(disp_decay * (0.6 + 0.4 * self._c_infinity_clamp(coherence, 0.0, 1.0) * activity_gate), index=df_scores.index), 0.0, 1.0)
        if _probe_data is not None and probe_ts is not None:
            self._log_probe(_probe_data, "【07. 宏观环境调节 (Environment)】", "EWD_Consensus (共振互信息熵)", final_decay, probe_ts)
        return final_decay
    def _calculate_pressure_backtest_modulator(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】平台突破点套牢盘抛压回测。"""
        penalty_raw = raw_data.get('breakout_penalty_score_D', pd.Series(0.0, index=df_index))
        penalty_slope = raw_data.get('SLOPE_13_breakout_penalty_score_D', pd.Series(0.0, index=df_index))
        penalty_hab = self._calculate_qho_historical_accumulation_buffer(penalty_raw, windows=[21])
        resistance_intensity = self._calculate_custom_normalization(pd.Series(penalty_raw * (1.0 + self._smooth_max_pair(penalty_slope, 0.0)), index=df_index), mode='limit_high', sensitivity=5.0)
        price_v = raw_data.get('price_slope_raw', pd.Series(0.0, index=df_index))
        backtest_factor = pd.Series(1.0 - (resistance_intensity * np.tanh(self._smooth_max_pair(price_v, 0.0) * 10.0)), index=df_index)
        final_modulator = pd.Series(0.2 + 0.8 * self._c_infinity_clamp(pd.Series((backtest_factor * (1.0 - penalty_hab)) + penalty_hab, index=df_index), 0.0, 1.0), index=df_index)
        return final_modulator
    def _calculate_level_stress_test_modulator(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】关键筹码支撑位的物理应力测试阵列。"""
        och_jerk = raw_data.get('JERK_5_OCH_ACCELERATION_D', pd.Series(0.0, index=df_index))
        och_jerk_score = self._calculate_custom_normalization(och_jerk, mode='limit_high', sensitivity=10.0, denoise=True)
        res_strength = raw_data.get('resistance_strength_D', pd.Series(0.0, index=df_index))
        ma21_proximity = pd.Series(1.0 - self._c_infinity_clamp(pd.Series(self._smooth_abs(raw_data.get('price_vs_ma_21_ratio_D', pd.Series(1.0, index=df_index)) - 1.0) * 20.0, index=df_index), 0.0, 1.0), index=df_index)
        ma55_proximity = pd.Series(1.0 - self._c_infinity_clamp(pd.Series(self._smooth_abs(raw_data.get('price_vs_ma_55_ratio_D', pd.Series(1.0, index=df_index)) - 1.0) * 20.0, index=df_index), 0.0, 1.0), index=df_index)
        level_weight = self._smooth_max_pair(ma21_proximity, ma55_proximity)
        stress_test_score = self._c_infinity_clamp(pd.Series((och_jerk_score * level_weight * self._calculate_custom_normalization(res_strength, mode='limit_high', sensitivity=1.0)), index=df_index), 0.0, 1.0)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Stress_Test (关键位极限测压)", stress_test_score, probe_ts)
        return stress_test_score
    def _calculate_linear_resonance_failure(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】防范线性趋势的虚假突破陷阱。"""
        r2_raw = raw_data.get('GEOM_REG_R2_D', pd.Series(0.0, index=df_index))
        r2_accel = raw_data.get('ACCEL_8_GEOM_REG_R2_D', pd.Series(0.0, index=df_index))
        r2_hab = self._calculate_qho_historical_accumulation_buffer(r2_raw, windows=[21], name="Geom_R2", _probe_data=_probe_data, probe_ts=probe_ts)
        failure_burst = self._calculate_custom_normalization(self._smooth_abs(r2_accel), mode='limit_high', sensitivity=5.0, denoise=True)
        reg_slope = raw_data.get('GEOM_REG_SLOPE_D', pd.Series(0.0, index=df_index))
        slope_calm = self._calculate_custom_normalization(reg_slope, mode='zero_focus', sensitivity=5.0, denoise=True)
        failure_score = self._power_mean_fusion(df_index, [r2_hab, failure_burst, slope_calm], [0.3, 0.4, 0.3], p=1.0)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "LRF_Score (线性死寂崩塌)", failure_score, probe_ts)
        return failure_score
    def _calculate_micro_order_gain(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】微观吃单获利筹码熵减。"""
        entropy_raw = raw_data.get('PRICE_ENTROPY_D', pd.Series(0.5, index=df_index))
        entropy_slope = raw_data.get('SLOPE_13_PRICE_ENTROPY_D', pd.Series(0.0, index=df_index))
        game_index = raw_data.get('intraday_chip_game_index_D', pd.Series(0.5, index=df_index))
        orderly_score = self._calculate_custom_normalization(entropy_slope, mode='negative_extreme', sensitivity=5.0, denoise=True)
        entropy_hab = self._calculate_qho_historical_accumulation_buffer(entropy_raw, windows=[21], name="Price_Entropy", _probe_data=_probe_data, probe_ts=probe_ts)
        price_calm = self._calculate_custom_normalization(raw_data.get('price_slope_raw', pd.Series(0.0, index=df_index)), mode='zero_focus', sensitivity=15.0, denoise=True)
        game_intensity = self._calculate_custom_normalization(game_index, mode='limit_high', sensitivity=1.5)
        gain_score = self._power_mean_fusion(df_index, [orderly_score, price_calm, self._c_infinity_clamp(pd.Series(1.0 - entropy_hab, index=df_index), 0.0, 1.0), game_intensity], [0.3, 0.3, 0.2, 0.2], p=1.0)
        return gain_score
    def _calculate_momentum_dissipation_balance(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】动量耗散的引力平衡阀。使用 vpa_activity_gate 阻断无用死水。"""
        vpa_raw = raw_data.get('VPA_EFFICIENCY_D', pd.Series(0.0, index=df_index))
        vpa_activity_gate = self._c_infinity_clamp(self._smooth_abs(vpa_raw).rolling(13, min_periods=1).mean() * 10.0, 0.0, 1.0)
        vpa_accel = raw_data.get('VPA_ACCELERATION_13D', pd.Series(0.0, index=df_index))
        vpa_accel_jerk = raw_data.get('JERK_5_VPA_ACCELERATION_13D', pd.Series(0.0, index=df_index))
        dissipation_focus = pd.Series(self._calculate_custom_normalization(vpa_accel, mode='zero_focus', sensitivity=5.0, denoise=True) * vpa_activity_gate, index=df_index)
        jerk_silence = pd.Series(self._calculate_custom_normalization(vpa_accel_jerk, mode='zero_focus', sensitivity=10.0, denoise=True) * vpa_activity_gate, index=df_index)
        price_calm = self._calculate_custom_normalization(raw_data.get('price_slope_raw', pd.Series(0.0, index=df_index)), mode='zero_focus', sensitivity=15.0, denoise=True)
        flow_impact = self._calculate_custom_normalization(raw_data.get('flow_impact_ratio_D', pd.Series(0.0, index=df_index)), mode='zero_focus', sensitivity=2.0)
        mdb_score = self._power_mean_fusion(df_index, [dissipation_focus, jerk_silence, price_calm, flow_impact], [0.3, 0.25, 0.25, 0.2], p=1.0)
        return mdb_score
    def _calculate_hunting_temporal_coherence(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】跨周期筹码猎杀一致性评定。强制动能门控(Kinetic Gate)拦截静默伪阳性。"""
        attack_raw = raw_data.get('HM_COORDINATED_ATTACK_D', pd.Series(0.0, index=df_index))
        attack_jerk = raw_data.get('JERK_5_HM_COORDINATED_ATTACK_D', pd.Series(0.0, index=df_index))
        attack_mean = self._smooth_abs(attack_raw).rolling(window=8, min_periods=1).mean().fillna(0.0)
        activity_gate = self._c_infinity_clamp(attack_mean * 10.0, 0.0, 1.0)
        attack_std = attack_raw.rolling(window=8, min_periods=1).std().fillna(0.0)
        temporal_stability = pd.Series((1.0 - self._calculate_custom_normalization(attack_std, mode='limit_high', sensitivity=2.0)) * activity_gate, index=df_index)
        rhythm_score = pd.Series(self._calculate_custom_normalization(attack_jerk, mode='zero_focus', sensitivity=10.0, denoise=True) * activity_gate, index=df_index)
        top_tier_activity = self._calculate_custom_normalization(raw_data.get('HM_ACTIVE_TOP_TIER_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=2.0)
        htc_score = self._power_mean_fusion(df_index, [temporal_stability, rhythm_score, top_tier_activity], [0.4, 0.4, 0.2], p=1.0)
        return htc_score
    def _calculate_liquidity_solidification_threshold(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】微小换手的绝对抛压锁死点判定。"""
        stability_raw = raw_data.get('TURNOVER_STABILITY_INDEX_D', pd.Series(0.0, index=df_index))
        stability_slope = raw_data.get('SLOPE_13_TURNOVER_STABILITY_INDEX_D', pd.Series(0.0, index=df_index))
        stability_score = self._calculate_custom_normalization(stability_raw, mode='limit_high', sensitivity=1.5)
        slope_growth = self._calculate_custom_normalization(stability_slope, mode='limit_high', sensitivity=5.0, denoise=True)
        stability_hab = self._calculate_qho_historical_accumulation_buffer(stability_raw, windows=[13, 21, 34, 55], name="Turnover_Stability", _probe_data=_probe_data, probe_ts=probe_ts)
        turnover_low = self._calculate_custom_normalization(raw_data.get('turnover_rate_f_D', pd.Series(0.0, index=df_index)), mode='limit_low', sensitivity=10.0)
        solidification_factor = self._power_mean_fusion(df_index, [stability_score, slope_growth, stability_hab, turnover_low], [0.3, 0.2, 0.2, 0.3], p=1.0)
        return solidification_factor
    def _calculate_amount_distribution_entropy_delta(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】分时量能分布结构的熵减速率。"""
        entropy_raw = raw_data.get('concentration_entropy_D', pd.Series(0.0, index=df_index))
        entropy_slope = raw_data.get('SLOPE_13_concentration_entropy_D', pd.Series(0.0, index=df_index))
        interceptive_score = self._calculate_custom_normalization(entropy_slope, mode='negative_extreme', sensitivity=5.0, denoise=True)
        entropy_hab = self._calculate_qho_historical_accumulation_buffer(entropy_raw, windows=[21], name="Amount_Entropy", _probe_data=_probe_data, probe_ts=probe_ts)
        final_score = self._power_mean_fusion(df_index, [interceptive_score, self._c_infinity_clamp(pd.Series(1.0 - entropy_hab, index=df_index), 0.0, 1.0)], [0.6, 0.4], p=1.0)
        return final_score
    def _calculate_seat_scatter_decay(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】游资跟风盘与散户席位活跃度的退潮点。"""
        any_act = raw_data.get('HM_ACTIVE_ANY_D', pd.Series(0.0, index=df_index))
        top_act = raw_data.get('HM_ACTIVE_TOP_TIER_D', pd.Series(0.0, index=df_index))
        scatter_raw = pd.Series(self._smooth_max_pair(any_act - top_act, 0.0), index=df_index)
        scatter_jerk = self._safe_diff(self._safe_diff(self._safe_diff(scatter_raw, 5), 5), 5)
        decay_score = self._calculate_custom_normalization(scatter_jerk, mode='negative_extreme', sensitivity=20.0, denoise=True)
        price_calm = self._calculate_custom_normalization(raw_data.get('price_slope_raw', pd.Series(0.0, index=df_index)), mode='zero_focus', sensitivity=15.0, denoise=True)
        final_decay = self._power_mean_fusion(df_index, [decay_score, price_calm], [0.5, 0.5], p=1.0)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Scatter_Decay (跟风席位退潮)", final_decay, probe_ts)
        return final_decay
    def _calculate_gravitational_regression_pull(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】乖离率向长期均值极点回归的万有引力拉力。"""
        bias_raw = raw_data.get('BIAS_55_D', pd.Series(0.0, index=df_index))
        bias_accel = raw_data.get('ACCEL_8_BIAS_55_D', pd.Series(0.0, index=df_index))
        bias_hab = self._calculate_qho_historical_accumulation_buffer(pd.Series(self._smooth_max_pair(-bias_raw, 0.0), index=df_index), windows=[21, 34, 55], name="Bias_55_Neg", _probe_data=_probe_data, probe_ts=probe_ts)
        gravity_ignite = self._calculate_custom_normalization(bias_accel, mode='limit_high', sensitivity=5.0, denoise=True)
        depth_score = self._calculate_custom_normalization(bias_raw, mode='negative_extreme', sensitivity=5.0)
        ou_pull = self._calculate_ornstein_uhlenbeck_pull(df_index, raw_data, _probe_data, probe_ts)
        pull_score = self._power_mean_fusion(df_index, [depth_score, bias_hab, gravity_ignite, ou_pull], [0.3, 0.2, 0.2, 0.3], p=1.0)
        return pull_score
    def _calculate_short_exhaustion_divergence(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】空头主动压盘动能的二阶耗尽点检测。"""
        ndi_raw = raw_data.get('NDI_14_D', pd.Series(0.0, index=df_index))
        ndi_jerk = raw_data.get('JERK_5_NDI_14_D', pd.Series(0.0, index=df_index))
        exhaustion_score = self._calculate_custom_normalization(ndi_jerk, mode='negative_extreme', sensitivity=20.0, denoise=True)
        price_calm = self._calculate_custom_normalization(raw_data.get('price_slope_raw', pd.Series(0.0, index=df_index)), mode='zero_focus', sensitivity=15.0, denoise=True)
        ndi_hab = self._calculate_qho_historical_accumulation_buffer(ndi_raw, windows=[21, 34, 55], name="NDI_14", _probe_data=_probe_data, probe_ts=probe_ts)
        divergence_score = self._power_mean_fusion(df_index, [exhaustion_score, price_calm, ndi_hab], [0.4, 0.3, 0.3], p=1.0)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Short_Exhaustion (空头抛压耗尽)", divergence_score, probe_ts)
        return divergence_score
    def _calculate_long_awakening_threshold(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】蛰伏期多头脉冲微弱苏醒阈值。"""
        pdi_raw = raw_data.get('PDI_14_D', pd.Series(0.0, index=df_index))
        pdi_slope = raw_data.get('SLOPE_13_PDI_14_D', pd.Series(0.0, index=df_index))
        pdi_jerk = raw_data.get('JERK_5_PDI_14_D', pd.Series(0.0, index=df_index))
        awakening_continuity = self._calculate_custom_normalization(pdi_slope, mode='limit_high', sensitivity=10.0, denoise=True)
        awakening_ignite = self._calculate_custom_normalization(pdi_jerk, mode='limit_high', sensitivity=20.0, denoise=True)
        pdi_hab = self._calculate_qho_historical_accumulation_buffer(pdi_raw, windows=[21, 34, 55], name="PDI_14", _probe_data=_probe_data, probe_ts=probe_ts)
        pdi_suppressed = self._c_infinity_clamp(pd.Series(1.0 - pdi_hab, index=df_index), 0.0, 1.0)
        awakening_score = self._power_mean_fusion(df_index, [awakening_continuity, awakening_ignite, pdi_suppressed], [0.35, 0.4, 0.25], p=1.0)
        return awakening_score
    def _calculate_abnormal_energy_overflow(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】无量状态下的异常能量溢出甄别。"""
        eff_jerk = raw_data.get('JERK_5_VPA_MF_ADJUSTED_EFF_D', pd.Series(0.0, index=df_index))
        overflow_ignite = self._calculate_custom_normalization(eff_jerk, mode='limit_high', sensitivity=10.0, denoise=True)
        amount_calm = self._calculate_custom_normalization(pd.Series(self._smooth_max_pair(raw_data.get('amount_D', pd.Series(1.0, index=df_index)) - 0.8, 0.0), index=df_index), mode='limit_low', sensitivity=3.0)
        price_calm = self._calculate_custom_normalization(raw_data.get('price_slope_raw', pd.Series(0.0, index=df_index)), mode='zero_focus', sensitivity=15.0, denoise=True)
        overflow_score = self._power_mean_fusion(df_index, [overflow_ignite, amount_calm, price_calm], [0.4, 0.3, 0.3], p=1.0)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "AEO_Score (异常能量溢出)", overflow_score, probe_ts)
        return overflow_score
    def _calculate_phase_locked_resonance(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】量价加速度及主力均线的物理相位锁定系统。"""
        vpa_raw = raw_data.get('VPA_EFFICIENCY_D', pd.Series(0.0, index=df_index))
        vpa_activity_gate = self._c_infinity_clamp(self._smooth_abs(vpa_raw).rolling(13, min_periods=1).mean() * 5.0, 0.0, 1.0)
        vpa_accel = raw_data.get('VPA_ACCELERATION_13D', pd.Series(0.0, index=df_index))
        price_accel = raw_data.get('MA_ACCELERATION_EMA_55_D', pd.Series(0.0, index=df_index))
        vpa_focus = pd.Series(self._calculate_custom_normalization(vpa_accel, mode='zero_focus', sensitivity=10.0, denoise=True) * vpa_activity_gate, index=df_index)
        price_focus = self._calculate_custom_normalization(price_accel, mode='zero_focus', sensitivity=10.0, denoise=True)
        resonance_sim = pd.Series((vpa_accel * price_accel).rolling(window=5, min_periods=1).mean() / (self._smooth_abs(vpa_accel).rolling(window=5, min_periods=1).mean() * self._smooth_abs(price_accel).rolling(window=5, min_periods=1).mean() + 1e-9), index=df_index)
        vpa_jerk = pd.Series(self._calculate_custom_normalization(raw_data.get('JERK_5_VPA_ACCELERATION_13D', pd.Series(0.0, index=df_index)), mode='zero_focus', sensitivity=15.0, denoise=True) * vpa_activity_gate, index=df_index)
        ma_coherence = self._calculate_custom_normalization(raw_data.get('MA_COHERENCE_RESONANCE_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=2.0)
        plr_score = self._power_mean_fusion(df_index, [vpa_focus, price_focus, self._c_infinity_clamp(pd.Series(0.5 + 0.5 * resonance_sim.fillna(0.0), index=df_index), 0.0, 1.0), vpa_jerk, ma_coherence], [0.2, 0.2, 0.25, 0.15, 0.2], p=1.0)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Phase_Locked (量价加速度锁死)", plr_score, probe_ts)
        return plr_score
    def _calculate_split_order_pulse_entropy(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】算法单对流动性池拆单特征的熵测算。加入动能门控阻击死寂期伪阳性。"""
        abnormal_jerk = raw_data.get('JERK_5_tick_abnormal_volume_ratio_D', pd.Series(0.0, index=df_index))
        jerk_std = abnormal_jerk.rolling(window=8, min_periods=1).std().fillna(0.0)
        jerk_mean = self._smooth_abs(abnormal_jerk).rolling(window=8, min_periods=1).mean().fillna(0.0)
        activity_gate = self._c_infinity_clamp(jerk_mean * 5.0, 0.0, 1.0)
        pulse_orderly = pd.Series(1.0 / (1.0 + (jerk_std / (jerk_mean + 1e-9))), index=df_index)
        order_score = pd.Series(self._calculate_custom_normalization(pulse_orderly, mode='limit_high', sensitivity=2.0, denoise=True) * activity_gate, index=df_index)
        price_calm = self._calculate_custom_normalization(raw_data.get('price_slope_raw', pd.Series(0.0, index=df_index)), mode='zero_focus', sensitivity=15.0, denoise=True)
        tick_cluster = self._calculate_custom_normalization(raw_data.get('tick_clustering_index_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=2.0)
        sope_gain = self._power_mean_fusion(df_index, [order_score, price_calm, tick_cluster], [0.4, 0.3, 0.3], p=1.0)
        return sope_gain
    def _calculate_efficiency_gradient_dissipation(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】价格涨跌势能转化率的静默耗散检测。注入动能门控修补稳定性计算漏洞。"""
        eff_raw = raw_data.get('VPA_MF_ADJUSTED_EFF_D', pd.Series(0.0, index=df_index))
        eff_slope = raw_data.get('SLOPE_13_VPA_MF_ADJUSTED_EFF_D', pd.Series(0.0, index=df_index))
        eff_accel = raw_data.get('ACCEL_8_VPA_MF_ADJUSTED_EFF_D', pd.Series(0.0, index=df_index))
        eff_mean = self._smooth_abs(eff_raw).rolling(window=8, min_periods=1).mean().fillna(0.0)
        activity_gate = self._c_infinity_clamp(eff_mean * 5.0, 0.0, 1.0)
        eff_std = eff_slope.rolling(window=8, min_periods=1).std().fillna(0.0)
        slope_stability = pd.Series((1.0 - self._calculate_custom_normalization(eff_std, mode='limit_high', sensitivity=2.0)) * activity_gate, index=df_index)
        accel_lock = pd.Series(self._calculate_custom_normalization(eff_accel, mode='zero_focus', sensitivity=15.0, denoise=True) * activity_gate, index=df_index)
        mf_activity = self._calculate_custom_normalization(raw_data.get('main_force_activity_index_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=2.0)
        egd_score = self._power_mean_fusion(df_index, [slope_stability, accel_lock, mf_activity], [0.4, 0.4, 0.2], p=1.0)
        return egd_score
    def _calculate_potential_well_collapse(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】深水坑底反转前夕的势能井塌缩判断。"""
        pit_state = raw_data.get('STATE_GOLDEN_PIT_D', pd.Series(0.0, index=df_index))
        pit_jerk = raw_data.get('JERK_5_STATE_GOLDEN_PIT_D', pd.Series(0.0, index=df_index))
        escape_ignite = self._calculate_custom_normalization(pit_jerk, mode='limit_high', sensitivity=10.0, denoise=True)
        trap_lock = self._calculate_custom_normalization(pit_jerk, mode='zero_focus', sensitivity=15.0, denoise=True)
        price_calm = self._calculate_custom_normalization(raw_data.get('price_slope_raw', pd.Series(0.0, index=df_index)), mode='zero_focus', sensitivity=15.0, denoise=True)
        well_collapse_score = self._power_mean_fusion(df_index, [pit_state, escape_ignite, self._c_infinity_clamp(pd.Series(1.0 - trap_lock, index=df_index), 0.0, 1.0), price_calm], [0.3, 0.3, 0.1, 0.3], p=1.0)
        return well_collapse_score
    def _calculate_high_freq_kinetic_gap_fill(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】极短期均线负向缺口的高频弹性缝合判断。"""
        bias5 = raw_data.get('BIAS_5_D', pd.Series(0.0, index=df_index))
        bias55 = raw_data.get('BIAS_55_D', pd.Series(0.0, index=df_index))
        b5_jerk = raw_data.get('JERK_5_BIAS_5_D', pd.Series(0.0, index=df_index))
        elasticity = self._calculate_custom_normalization(bias5, mode='negative_extreme', sensitivity=5.0)
        ignite = self._calculate_custom_normalization(b5_jerk, mode='limit_high', sensitivity=10.0, denoise=True)
        gap_score = self._calculate_custom_normalization(pd.Series(self._smooth_max_pair(bias55 - bias5, 0.0), index=df_index), mode='limit_high', sensitivity=2.0)
        price_calm = self._calculate_custom_normalization(raw_data.get('price_slope_raw', pd.Series(0.0, index=df_index)), mode='zero_focus', sensitivity=15.0, denoise=True)
        final_fill_score = self._power_mean_fusion(df_index, [elasticity, ignite, gap_score, price_calm], [0.3, 0.3, 0.2, 0.2], p=1.0)
        return final_fill_score
    def _calculate_volatility_vacuum_contraction(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】多维波动率真空塌缩，量价死寂核心检测器。"""
        atr_raw = raw_data.get('ATR_14_D', pd.Series(0.0, index=df_index))
        atr_slope = raw_data.get('SLOPE_13_ATR_14_D', pd.Series(0.0, index=df_index))
        atr_jerk = raw_data.get('JERK_5_ATR_14_D', pd.Series(0.0, index=df_index))
        atr_low_score = self._calculate_custom_normalization(atr_raw, mode='limit_low', sensitivity=10.0)
        decay_purity = self._calculate_custom_normalization(atr_slope, mode='negative_extreme', sensitivity=10.0, denoise=True)
        vacuum_silence = self._calculate_custom_normalization(atr_jerk, mode='zero_focus', sensitivity=15.0, denoise=True)
        price_calm = self._calculate_custom_normalization(raw_data.get('price_slope_raw', pd.Series(0.0, index=df_index)), mode='zero_focus', sensitivity=15.0, denoise=True)
        vvc_score = self._power_mean_fusion(df_index, [atr_low_score, decay_purity, vacuum_silence, price_calm], [0.3, 0.25, 0.25, 0.2], p=1.0)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "VVC_Score (波动率真空态)", vvc_score, probe_ts)
        return vvc_score
    def _calculate_fan_curvature_collapse(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】均线扇形曲率多重收敛极限。加入了 fan_gate 控制死区。"""
        fan_raw = raw_data.get('MA_FAN_EFFICIENCY_D', pd.Series(0.0, index=df_index))
        fan_accel = raw_data.get('ACCEL_8_MA_FAN_EFFICIENCY_D', pd.Series(0.0, index=df_index))
        fan_jerk = raw_data.get('JERK_5_MA_FAN_EFFICIENCY_D', pd.Series(0.0, index=df_index))
        fan_mean = self._smooth_abs(fan_raw).rolling(13, min_periods=1).mean().fillna(0.0)
        fan_gate = self._c_infinity_clamp(fan_mean * 5.0, 0.0, 1.0)
        accel_focus = pd.Series(self._calculate_custom_normalization(fan_accel, mode='zero_focus', sensitivity=10.0, denoise=True) * fan_gate, index=df_index)
        jerk_silence = pd.Series(self._calculate_custom_normalization(fan_jerk, mode='zero_focus', sensitivity=15.0, denoise=True) * fan_gate, index=df_index)
        fan_high_score = self._calculate_custom_normalization(fan_raw, mode='limit_high', sensitivity=1.2)
        price_calm = self._calculate_custom_normalization(raw_data.get('price_slope_raw', pd.Series(0.0, index=df_index)), mode='zero_focus', sensitivity=15.0, denoise=True)
        fcc_score = self._power_mean_fusion(df_index, [fan_high_score, accel_focus, jerk_silence, price_calm], [0.3, 0.25, 0.25, 0.2], p=1.0)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "FCC_Score (扇面曲率塌缩)", fcc_score, probe_ts)
        return fcc_score
    def _calculate_game_neutralization_modulator(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】日内开收盘多空博弈极致中和状态。"""
        och_raw = raw_data.get('OCH_D', pd.Series(0.0, index=df_index))
        och_slope = raw_data.get('SLOPE_13_OCH_D', pd.Series(0.0, index=df_index))
        och_mean = self._smooth_abs(och_raw).rolling(13, min_periods=1).mean().fillna(0.0)
        och_gate = self._c_infinity_clamp(och_mean * 5.0, 0.0, 1.0)
        neutralization_focus = pd.Series(self._calculate_custom_normalization(och_slope, mode='zero_focus', sensitivity=10.0, denoise=True) * och_gate, index=df_index)
        och_intensity = self._calculate_custom_normalization(och_raw, mode='limit_high', sensitivity=1.0)
        price_calm = self._calculate_custom_normalization(raw_data.get('price_slope_raw', pd.Series(0.0, index=df_index)), mode='zero_focus', sensitivity=15.0, denoise=True)
        neutral_score = self._power_mean_fusion(df_index, [och_intensity, neutralization_focus, price_calm], [0.4, 0.3, 0.3], p=1.0)
        return neutral_score
    def _calculate_oversold_momentum_bipolarization(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】利用RSI超跌属性进行反向二极管增益触发。"""
        rsi_raw = raw_data.get('RSI_13_D', pd.Series(0.0, index=df_index))
        rsi_accel = raw_data.get('ACCEL_8_RSI_13_D', pd.Series(0.0, index=df_index))
        accel_rev_slope = self._smooth_abs(self._safe_diff(rsi_accel, 5))
        vol = raw_data.get('volume_D', pd.Series(1.0, index=df_index))
        vol_consistency = pd.Series(1.0 / (1.0 + vol.rolling(window=8, min_periods=1).std().fillna(0.0) / (vol.rolling(window=8, min_periods=1).mean() + 1e-9)), index=df_index)
        oversold_lock = self._calculate_custom_normalization(pd.Series(rsi_raw, index=df_index), mode='limit_low', sensitivity=5.0)
        bipolar_ratio = self._calculate_custom_normalization(accel_rev_slope * vol_consistency, mode='limit_high', sensitivity=10.0, denoise=True)
        price_calm = self._calculate_custom_normalization(raw_data.get('price_slope_raw', pd.Series(0.0, index=df_index)), mode='zero_focus', sensitivity=15.0, denoise=True)
        omb_score = self._power_mean_fusion(df_index, [oversold_lock, bipolar_ratio, price_calm], [0.4, 0.4, 0.2], p=1.0)
        self._log_probe(_probe_data, "【07. 宏观环境调节 (Environment)】", "Bipolar_Gain (动能二极化极值)", omb_score, probe_ts)
        return omb_score
    def _calculate_kinetic_overflow_veto(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], bipolar_gain: pd.Series, _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】高斯核指数防爆系统：对违背寂静原则的暴涨、高换手、极端RSI实施绝对物理抹杀，由于底噪已降至千万分之一，该环节将实现绝对 0.0000 封喉。"""
        rsi_raw = raw_data.get('RSI_13_D', pd.Series(0.5, index=df_index))
        rsi_deviation = pd.Series(self._smooth_max_pair(rsi_raw - 0.70, 0.0) + self._smooth_max_pair(0.30 - rsi_raw, 0.0), index=df_index)
        veto_l1 = pd.Series(np.exp(-np.square(rsi_deviation * 15.0)), index=df_index)
        turnover = raw_data.get('turnover_rate_f_D', pd.Series(0.0, index=df_index))
        turnover_excess = pd.Series(self._smooth_max_pair(turnover - 0.15, 0.0), index=df_index)
        veto_l2 = pd.Series(np.exp(-np.square(turnover_excess * 15.0)), index=df_index)
        price_v = raw_data.get('price_slope_raw', pd.Series(0.0, index=df_index))
        price_v_excess = pd.Series(self._smooth_max_pair(price_v - 0.15, 0.0) + self._smooth_max_pair(-0.10 - price_v, 0.0), index=df_index)
        veto_l3 = pd.Series(np.exp(-np.square(price_v_excess * 20.0)), index=df_index)
        final_veto = self._c_infinity_clamp(pd.Series(veto_l1 * veto_l2 * veto_l3, index=df_index), 0.0, 1.0)
        self._log_probe(_probe_data, "【07. 宏观环境调节 (Environment)】", "Veto_L1 (RSI超买高位衰竭)", veto_l1, probe_ts)
        self._log_probe(_probe_data, "【07. 宏观环境调节 (Environment)】", "Veto_L2 (量能高位末端脉冲)", veto_l2, probe_ts)
        self._log_probe(_probe_data, "【07. 宏观环境调节 (Environment)】", "Veto_L3 (价格动能脱轨背离)", veto_l3, probe_ts)
        self._log_probe(_probe_data, "【07. 宏观环境调节 (Environment)】", "Veto_Factor (三级防爆综合熔断)", final_veto, probe_ts)
        return final_veto
    def _calculate_spatio_temporal_asymmetric_reward(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], resonance_confirm: pd.Series, _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】无前视偏误时空滞后非对称期望系统。"""
        close = raw_data.get('close_D', pd.Series(1.0, index=df_index))
        past_ret = close / (close.shift(5).fillna(close) + 1e-9) - 1.0
        hist_hit_mask = resonance_confirm.shift(5).fillna(0.0)
        expected_gain = (past_ret * hist_hit_mask).rolling(window=120, min_periods=10).mean().fillna(0.0)
        reward_factor = pd.Series(1.0 + self._calculate_custom_normalization(pd.Series(self._smooth_max_pair(expected_gain, 0.0), index=df_index), mode='limit_high', sensitivity=5.0), index=df_index)
        self._log_probe(_probe_data, "【07. 宏观环境调节 (Environment)】", "Reward_Factor (时空异步奖赏)", reward_factor, probe_ts)
        return reward_factor
    def _calculate_extreme_panic_resonance(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】坑底绝望极值的恐慌共振释放。"""
        pain_jerk = raw_data.get('JERK_5_pain_index_proxy', pd.Series(0.0, index=df_index))
        panic_burst = self._calculate_custom_normalization(pain_jerk, mode='limit_high', sensitivity=10.0, denoise=True)
        pit_state = raw_data.get('STATE_GOLDEN_PIT_D', pd.Series(0.0, index=df_index))
        resonance_score = self._power_mean_fusion(df_index, [panic_burst, pit_state], [0.7, 0.3], p=1.0)
        return resonance_score
    def _calculate_adaptive_phase_transition_threshold(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】适应性相位转换的动态判决门限测算，防止静水区误触发。"""
        price_v = raw_data.get('price_slope_raw', pd.Series(0.0, index=df_index))
        mean_v = self._smooth_max_pair(self._smooth_abs(price_v.rolling(window=250, min_periods=60).mean()), 1e-5)
        std_v = price_v.rolling(window=250, min_periods=60).std().fillna(1e-5)
        noise_cv = std_v / mean_v
        adaptive_threshold = pd.Series(0.35 + 0.15 * self._calculate_custom_normalization(noise_cv, mode='limit_high', sensitivity=2.0), index=df_index)
        self._log_probe(_probe_data, "【07. 宏观环境调节 (Environment)】", "Fermi_Threshold (动态软门限)", adaptive_threshold.fillna(0.35), probe_ts)
        return adaptive_threshold.fillna(0.35)
    def _calculate_mean_reversion_kinetic_bias(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        """【V38.0.0】基于大周期长线的均值引力弹弓增强。"""
        bias144 = raw_data.get('price_vs_ma_144_ratio', pd.Series(1.0, index=df_index))
        accel144 = raw_data.get('ACCEL_8_price_vs_ma_144_ratio', pd.Series(0.0, index=df_index))
        depth_reward = self._calculate_custom_normalization(pd.Series(self._smooth_max_pair(1.0 - bias144, 0.0), index=df_index), mode='limit_high', sensitivity=3.0)
        slingshot_ignite = self._calculate_custom_normalization(accel144, mode='limit_high', sensitivity=10.0, denoise=True)
        bias_factor = pd.Series(1.0 + 0.25 * self._power_mean_fusion(df_index, [depth_reward, slingshot_ignite], [0.5, 0.5], p=1.0), index=df_index)
        self._log_probe(_probe_data, "【07. 宏观环境调节 (Environment)】", "MRKB_Factor (均值引力弹弓)", bias_factor.fillna(1.0), probe_ts)
        return bias_factor.fillna(1.0)











