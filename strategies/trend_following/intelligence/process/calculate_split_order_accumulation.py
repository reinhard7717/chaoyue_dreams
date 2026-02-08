# strategies\trend_following\intelligence\process\calculate_split_order_accumulation.py
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

class CalculateSplitOrderAccumulation:
    """
    PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY
    【V4.0.2 · 拆单吸筹强度 · 探针强化与问题暴露版】
    - 核心修正: 严格区分原始数据及其MTF衍生与原子信号。原子信号的趋势通过直接diff()计算，
                避免在df中查找不存在的MTF衍生列。
    - 核心升级: 引入动态效率基准线，增强价格行为捕捉，精细化欺诈意图识别，MTF核心信号增强，
                情境自适应权重调整，非线性融合强化，趋势动量diff()化。
    - 探针强化: 增加关键中间计算节点的详细探针，特别是针对_robust_geometric_mean的输入和输出，
                以及最终pow()操作的精确值，以暴露潜在的计算偏差或bug。
    """
    def __init__(self, strategy_instance, helper: ProcessIntelligenceHelper):
        """
        初始化拆单吸筹强度计算器。
        参数:
            strategy_instance: 策略实例，用于访问全局配置和原子状态。
            helper: ProcessIntelligenceHelper 实例，用于访问辅助方法。
        """
        self.strategy = strategy_instance
        self.helper = helper
        # 从 helper 获取参数
        self.params = self.helper.params
        self.debug_params = self.helper.debug_params
        self.probe_dates = self.helper.probe_dates
        # 获取MTF权重配置
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        self.actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # 默认的MTF斜率/加速度权重，可在config中覆盖
        self.default_mtf_slope_accel_weights = {"slope_periods": {"5": 0.6, "13": 0.4}, "accel_periods": {"5": 0.7, "13": 0.3}}

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V7.0.2 · 数据引用修复版】
        计算拆单吸筹强度。修正了参数传递逻辑，确保 df 始终通过参数链传递，解决 AttributeError。
        """
        method_name = "calculate_split_order_accumulation"
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
        _temp_debug_values = {}
        df_index = df.index
        mtf_slope_accel_weights = config.get('mtf_slope_accel_weights', self.default_mtf_slope_accel_weights)
        # 1. 获取信号
        raw_signals, normalized_signals, mtf_signals, context_signals = self._get_and_normalize_signals(df, mtf_slope_accel_weights, method_name)
        # 2. 计算基准线
        dynamic_efficiency_baseline, baseline_debug_values = self._calculate_dynamic_efficiency_baseline(context_signals, df_index, config)
        # 3. 计算初步分数 (新增传递 df)
        dynamic_preliminary_score, preliminary_debug_values = self._calculate_preliminary_score(df, normalized_signals, mtf_signals, context_signals, df_index, config)
        # 4. 全息验证 (已具备 df)
        holographic_validation_score, holographic_debug_values = self._calculate_holographic_validation(df, raw_signals, normalized_signals, mtf_signals, context_signals, df_index, config, is_debug_enabled_for_method, probe_ts)
        # 5. 校准
        final_score_raw, final_score_debug_values = self._apply_quality_efficiency_calibration(dynamic_preliminary_score, holographic_validation_score, dynamic_efficiency_baseline, probe_ts, context_signals)
        # 6. 背离预警
        divergence_warning = self._calculate_divergence_warning(final_score_raw, normalized_signals, context_signals, df_index)
        # 7. 风险平抑
        is_leader = context_signals.get("is_leader", pd.Series(0.0, index=df_index))
        suppression_exponent = 1.5 - is_leader * 0.5
        risk_suppression_factor = (1 - divergence_warning.pow(suppression_exponent)).clip(0, 1)
        adjusted_final_score = (final_score_raw * risk_suppression_factor).clip(0, 1)
        if is_debug_enabled_for_method and probe_ts:
            _temp_debug_values.update({
                "归一化处理": normalized_signals,
                "动态效率基准线": baseline_debug_values,
                "初步计算": preliminary_debug_values,
                "全息验证": holographic_debug_values,
                "质效校准": final_score_debug_values,
                "风险平抑": {
                    "divergence_warning": divergence_warning,
                    "risk_suppression_factor": risk_suppression_factor,
                    "final_adjusted_score": adjusted_final_score
                }
            })
            debug_output = {}
            debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
            self._print_debug_info(method_name, probe_ts, debug_output, _temp_debug_values, adjusted_final_score)
        return adjusted_final_score.astype(np.float32)

    def _calculate_synthetic_smart_proxy(self, df: pd.DataFrame, df_index: pd.Index) -> pd.Series:
        """
        【V7.7.0 · 效率补偿还原版 · 替代指标重构】
        利用流向效率(flow_efficiency_D)与净能量流(net_energy_flow_D)替代缺失的厚度指标。
        - 核心逻辑: 针对低换手下的资金挪移，利用效率因子放大隐秘机构流的权重。
        - 逻辑审计: 2026-02-08 修复，移除不存在的 flow_thickness_ratio_D，改用 flow_efficiency_D。
        """
        # 1. 提取显性与还原后的机构资金 [cite: 1, 3]
        elg_buy = self.helper._get_safe_series(df, 'buy_elg_amount_D', 0.0)
        mf_net = self.helper._get_safe_series(df, 'net_mf_amount_D', 0.0)
        md_buy = self.helper._get_safe_series(df, 'buy_md_amount_D', 0.0)
        lg_buy = self.helper._get_safe_series(df, 'buy_lg_amount_D', 0.0)
        stealth_ratio = self.helper._get_safe_series(df, 'stealth_flow_ratio_D', 0.0)
        hidden_inst_flow = (md_buy + lg_buy) * stealth_ratio
        # 2. 引入效率与能量替代因子 
        flow_eff = self.helper._get_safe_series(df, 'flow_efficiency_D', 0.0)
        energy_flow = self.helper._get_safe_series(df, 'net_energy_flow_D', 0.0)
        synergy_buy = self.helper._get_safe_series(df, 'SMART_MONEY_SYNERGY_BUY_D', 0.0)
        # 3. 归一化与多维耦合
        proxy_components = {
            "explicit": self.helper._normalize_series(elg_buy + mf_net, df_index, bipolar=True),
            "hidden": self.helper._normalize_series(hidden_inst_flow, df_index, bipolar=True),
            "efficiency": self.helper._normalize_series(flow_eff, df_index, bipolar=False),
            "energy": self.helper._normalize_series(energy_flow, df_index, bipolar=True)
        }
        print(f"  -- [V7.7.0 代理合成探针] Eff: {flow_eff.mean():.4f}, Energy: {energy_flow.mean():.4f}")
        # 4. 非线性协同融合
        weights = {"explicit": 0.3, "hidden": 0.3, "efficiency": 0.2, "energy": 0.2}
        return _robust_geometric_mean(proxy_components, weights, df_index).fillna(0.0)

    def _get_and_normalize_signals(self, df: pd.DataFrame, mtf_slope_accel_weights: Dict, method_name: str) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series], Dict[str, pd.Series], Dict[str, pd.Series]]:
        """
        【V8.9.0 · 物理底噪压制版 · 信号全维重构】
        引入物理底噪锚点(PNF)，防止在静默期因微小变动导致历史排名虚高。
        - 核心修正: 在计算环境指标排名时，若绝对变动低于底噪(ANF)，排名强行置为 0.5。
        - 指标审计: 2026-02-08 更新，解决主力静默期后的“第一笔派发”导致逻辑失灵的问题。
        """
        df_index = df.index
        raw_df_columns = [
            'accumulation_score_D', 'stealth_flow_ratio_D', 'SMART_MONEY_INST_NET_BUY_D', 'STATE_PARABOLIC_WARNING_D', 
            'SMART_MONEY_HM_NET_BUY_D', 'SMART_MONEY_SYNERGY_BUY_D', 'buy_elg_amount_D', 'anomaly_intensity_D',
            'buy_lg_amount_D', 'buy_md_amount_D', 'net_mf_amount_D', 'tick_large_order_net_D', 'HM_COORDINATED_ATTACK_D',
            'VPA_MF_ADJUSTED_EFF_D', 'absorption_energy_D', 'chip_concentration_ratio_D', 'flow_acceleration_D',
            'chip_entropy_D', 'chip_stability_D', 'flow_intensity_D', 'GEOM_ARC_CURVATURE_D', 'ADX_14_D',
            'STATE_GOLDEN_PIT_D', 'STATE_ROUNDING_BOTTOM_D', 'IS_MARKET_LEADER_D', 'intraday_accumulation_confidence_D',
            'TURNOVER_STABILITY_INDEX_D', 'tick_data_quality_score_D', 'THEME_HOTNESS_SCORE_D', 'market_sentiment_score_D',
            'PRICE_ENTROPY_D', 'MA_POTENTIAL_COMPRESSION_RATE_D', 'close_D', 'VPA_ACCELERATION_5D', 'flow_consistency_D',
            'flow_efficiency_D', 'net_energy_flow_D', 'SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D', 'tick_chip_transfer_efficiency_D',
            'flow_momentum_5d_D', 'uptrend_strength_D', 'game_intensity_D', 'intraday_main_force_activity_D'
        ]
        raw_signals = {col: self.helper._get_safe_series(df, col, 0.0, method_name=method_name) for col in raw_df_columns}
        normalized_signals = {}
        ssmp_proxy = self._calculate_synthetic_smart_proxy(df, df_index)
        noise_sensitive_list = ['accumulation_score_D', 'stealth_flow_ratio_D', 'SMART_MONEY_INST_NET_BUY_D', 'buy_elg_amount_D', 'chip_entropy_D', 'SMART_MONEY_SYNERGY_BUY_D', 'HM_COORDINATED_ATTACK_D', 'market_sentiment_score_D', 'PRICE_ENTROPY_D', 'MA_POTENTIAL_COMPRESSION_RATE_D', 'STATE_PARABOLIC_WARNING_D', 'SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D']
        for indicator in noise_sensitive_list:
            base_val = raw_signals[indicator]
            active_base = ssmp_proxy if (indicator == 'SMART_MONEY_INST_NET_BUY_D' and base_val.std() < 1e-6) else base_val
            prefix = "proxy_" if (indicator == 'SMART_MONEY_INST_NET_BUY_D' and base_val.std() < 1e-6) else ""
            periods = [5, 13, 21] if indicator in ['PRICE_ENTROPY_D', 'MA_POTENTIAL_COMPRESSION_RATE_D'] else [5, 13]
            current_multiplier = 0.01 if indicator in ['PRICE_ENTROPY_D', 'MA_POTENTIAL_COMPRESSION_RATE_D'] else (0.02 if prefix == "proxy_" else 0.05)
            dyn_eps = self._calculate_dynamic_epsilon(active_base, window=55, multiplier=current_multiplier)
            for p in periods:
                for deriv_type in ['SLOPE', 'ACCEL', 'JERK']:
                    col_name = f'{deriv_type}_{p}_{indicator}'
                    raw_deriv = self.helper._get_safe_series(df, col_name, 0.0) if prefix == "" else active_base.diff(p)
                    clean_deriv = self._apply_derivative_denoising(raw_deriv, active_base, dyn_eps)
                    normalized_signals[f'clean_{prefix}{col_name}'] = self.helper._normalize_series(clean_deriv, df_index, bipolar=True)
        intent_comps = {"explicit": self.helper._normalize_series(raw_signals['buy_elg_amount_D'] + raw_signals['net_mf_amount_D'], df_index, bipolar=True), "hidden_slope": normalized_signals.get('clean_proxy_SLOPE_5_SMART_MONEY_INST_NET_BUY_D', pd.Series(0.0, index=df_index)), "consistency": self.helper._normalize_series(raw_signals['flow_consistency_D'], df_index, bipolar=False)}
        normalized_signals["data_intent_outcome"] = _robust_geometric_mean(intent_comps, {"explicit": 0.3, "hidden_slope": 0.4, "consistency": 0.3}, df_index).fillna(0.0)
        # 针对环境熵和压缩率引入百分位排名与底噪抑制
        entropy = raw_signals['PRICE_ENTROPY_D']
        entropy_diff = entropy.diff(5).abs()
        entropy_anf = entropy.rolling(60).std() * 0.1 # 物理底噪设为历史波动的10%
        entropy_rank = entropy.rolling(60).rank(pct=True)
        # 抑制逻辑: 若变动低于底噪，排位强行中性化(0.5)
        stable_entropy_rank = entropy_rank.where(entropy_diff > entropy_anf, 0.5)
        compression = raw_signals['MA_POTENTIAL_COMPRESSION_RATE_D']
        comp_rank = compression.rolling(60).rank(pct=True)
        stable_comp_rank = comp_rank.where(compression.diff(5).abs() > (compression.rolling(60).std() * 0.1), 0.5)
        context_signals = {
            "is_leader": raw_signals['IS_MARKET_LEADER_D'].astype(float),
            "mf_activity": self.helper._normalize_series(raw_signals['intraday_main_force_activity_D'], df_index, bipolar=False),
            "adx_norm": self.helper._normalize_series(raw_signals['ADX_14_D'], df_index, bipolar=False),
            "sentiment_norm": self.helper._normalize_series(raw_signals['market_sentiment_score_D'], df_index, bipolar=False),
            "entropy_norm": self.helper._normalize_series(entropy, df_index, bipolar=False),
            "entropy_rank": stable_entropy_rank,
            "compression_rank": stable_comp_rank,
            "sentiment_slope": normalized_signals.get('clean_SLOPE_5_market_sentiment_score_D', pd.Series(0.0, index=df_index)),
            "anomaly_intensity": self.helper._normalize_series(raw_signals['anomaly_intensity_D'], df_index, bipolar=False),
            "vpa_accel_5d": self.helper._normalize_series(raw_signals['VPA_ACCELERATION_5D'], df_index, bipolar=False),
            "flow_accel": self.helper._normalize_series(raw_signals['flow_acceleration_D'], df_index, bipolar=True),
            "flow_mom": self.helper._normalize_series(raw_signals['flow_momentum_5d_D'], df_index, bipolar=True),
            "turnover_stability": self.helper._normalize_series(raw_signals['TURNOVER_STABILITY_INDEX_D'], df_index, bipolar=False),
            "data_quality": self.helper._normalize_series(raw_signals['tick_data_quality_score_D'], df_index, bipolar=False),
            "theme_hotness": self.helper._normalize_series(raw_signals['THEME_HOTNESS_SCORE_D'], df_index, bipolar=False),
            "parabolic_warning": self.helper._normalize_series(raw_signals['STATE_PARABOLIC_WARNING_D'], df_index, bipolar=False),
            "parabolic_slope": normalized_signals.get('clean_SLOPE_5_STATE_PARABOLIC_WARNING_D', pd.Series(0.0, index=df_index)),
            "sm_divergence": self.helper._normalize_series(raw_signals['SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D'], df_index, bipolar=False),
            "sm_divergence_slope": normalized_signals.get('clean_SLOPE_5_SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D', pd.Series(0.0, index=df_index)),
            "synergy_slope": normalized_signals.get('clean_SLOPE_5_SMART_MONEY_SYNERGY_BUY_D', pd.Series(0.0, index=df_index)),
            "hm_attack": self.helper._normalize_series(raw_signals['HM_COORDINATED_ATTACK_D'], df_index, bipolar=False),
            "hm_attack_slope": normalized_signals.get('clean_SLOPE_5_HM_COORDINATED_ATTACK_D', pd.Series(0.0, index=df_index)),
            "intraday_acc_conf": self.helper._normalize_series(raw_signals['intraday_accumulation_confidence_D'], df_index, bipolar=False),
            "chip_transfer_eff": self.helper._normalize_series(raw_signals['tick_chip_transfer_efficiency_D'], df_index, bipolar=False)
        }
        print(f"  -- [V8.9.0 零基抑制探针] Entropy Rank Mean: {stable_entropy_rank.mean():.4f}, Comp Rank Mean: {stable_comp_rank.mean():.4f}")
        return raw_signals, normalized_signals, {}, context_signals

    def _calculate_dynamic_epsilon(self, base_series: pd.Series, window: int = 55, multiplier: float = 0.05) -> pd.Series:
        """
        【V7.1.1 · 动态灵敏度版】
        计算原子指标的内生底噪门槛。增加对乘数的外部支持，以应对高隐秘性的拆单动作。
        """
        rolling_std = base_series.rolling(window=window, min_periods=1).std().fillna(0.0)
        # 利用传入的 multiplier 灵活控制显著性门槛
        return (rolling_std * multiplier).replace(0, 1e-6)

    def _apply_derivative_denoising(self, derivative_series: pd.Series, base_series: pd.Series, dynamic_epsilon: pd.Series) -> pd.Series:
        """
        【V5.8.2 · 导数高保真降噪过滤器】
        利用动态阈值执行“掩码过滤+软收缩”双重降噪。
        """
        activity_mask = (base_series.abs() > dynamic_epsilon).astype(float)
        denoised_derivative = derivative_series * np.tanh(derivative_series / dynamic_epsilon)
        return (denoised_derivative * activity_mask).fillna(0.0)

    def _calculate_holographic_validation(self, df: pd.DataFrame, raw_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], context_signals: Dict[str, pd.Series], df_index: pd.Index, config: Dict, is_debug_enabled_for_method: bool, probe_ts: Optional[pd.Timestamp]) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """
        【V8.7.0 · 主力活跃度集成版 · 全息验证深化】
        应用日内主力活跃度(mf_activity)进行机构动作穿透校验，过滤散户自然换手噪音。
        - 验证逻辑: 引入主力活跃度校验项，对缺乏机构主动性动作支持的信号执行最高15%的压制。
        - 探针审计: 2026-02-08 增加主力活跃度探针，实时量化日内机构参与度对全息得分的有效支撑。
        """
        holographic_debug_values = {}
        is_leader = context_signals.get("is_leader", pd.Series(0.0, index=df_index))
        intraday_acc_conf = context_signals.get("intraday_acc_conf", pd.Series(1.0, index=df_index))
        chip_transfer_eff = context_signals.get("chip_transfer_eff", pd.Series(0.5, index=df_index))
        mf_activity = context_signals.get("mf_activity", pd.Series(0.5, index=df_index))
        synergy_slope = context_signals.get("synergy_slope", pd.Series(0.0, index=df_index))
        holographic_state_components = {
            "flow": normalized_signals.get("data_flow_outcome", pd.Series(0.0, index=df_index)),
            "structure": normalized_signals.get("data_structure_outcome", pd.Series(0.0, index=df_index)),
            "geom": normalized_signals.get("data_geom_outcome", pd.Series(0.0, index=df_index))
        }
        holographic_state_score = _robust_geometric_mean({k: (v + 1) / 2 for k, v in holographic_state_components.items()}, {"flow": 0.4, "structure": 0.3, "geom": 0.3}, df_index)
        acc_slope_13 = self.helper._get_safe_series(df, 'SLOPE_13_accumulation_score_D', 0.0)
        acc_accel_13 = self.helper._get_safe_series(df, 'ACCEL_13_accumulation_score_D', 0.0)
        stabilization_bonus = ((acc_slope_13 < 0) & (acc_accel_13 > 0)).astype(float) * 0.15
        holographic_trend_components = {
            "flow_slope": normalized_signals.get("clean_SLOPE_5_stealth_flow_ratio_D", pd.Series(0.0, index=df_index)),
            "acc_slope": normalized_signals.get("clean_SLOPE_5_accumulation_score_D", pd.Series(0.0, index=df_index)),
            "vpa_slope": normalized_signals.get("clean_SLOPE_5_VPA_MF_ADJUSTED_EFF_D", pd.Series(0.0, index=df_index))
        }
        holographic_trend_score = (_robust_geometric_mean({k: (v + 1) / 2 for k, v in holographic_trend_components.items()}, {"flow_slope": 0.4, "acc_slope": 0.4, "vpa_slope": 0.2}, df_index) + stabilization_bonus).clip(0, 1)
        rdi_signals = self._calculate_rdi_signals(df, normalized_signals, df_index, config)
        resonance_bonus = rdi_signals.get("phase_resonance_intensity", pd.Series(0.0, index=df_index))
        synergy_bonus = synergy_slope.clip(lower=0) * 0.1
        # 全息校验核心：整合主力活跃度校验项
        final_correction = (1 + resonance_bonus * 0.2 + synergy_bonus) * (0.7 + 0.3 * intraday_acc_conf) * (0.8 + 0.2 * chip_transfer_eff) * (0.85 + 0.15 * mf_activity)
        holographic_validation_score = (_robust_geometric_mean({"state": holographic_state_score, "trend": holographic_trend_score}, {"state": 0.5, "trend": 0.5}, df_index) * final_correction).clip(0, 1)
        print(f"  -- [全息活跃度校验探针] MF Activity Mean: {mf_activity.mean():.4f}, Support Multiplier: {(0.85 + 0.15 * mf_activity).mean():.4f}")
        direction = holographic_trend_components["acc_slope"].apply(lambda x: 1 if x >= 0 else -1)
        if probe_ts is not None and probe_ts in df_index:
            holographic_debug_values.update({"mf_activity_support": mf_activity.loc[probe_ts], "holographic_val_final": holographic_validation_score.loc[probe_ts]})
        return (holographic_validation_score * direction).clip(-1, 1), holographic_debug_values

    def _calculate_rdi_signals(self, df: pd.DataFrame, normalized_signals: Dict[str, pd.Series], df_index: pd.Index, config: Dict) -> Dict[str, pd.Series]:
        """
        【V7.0.2 · 引用路径修正版】
        计算 RDI 信号。修正了对原始 df 的访问路径。
        """
        rdi_signals = {}
        # 修正: 使用传入的 df 替代 self.strategy.df
        acc_slope = self.helper._get_safe_series(df, 'SLOPE_5_accumulation_score_D', 0.0)
        acc_accel = self.helper._get_safe_series(df, 'ACCEL_5_accumulation_score_D', 0.0)
        acc_jerk = self.helper._get_safe_series(df, 'JERK_5_accumulation_score_D', 0.0)
        phase_coherence = ((acc_slope > 0) & (acc_accel > 0) & (acc_jerk > 0)).astype(float)
        # 战术相位共振 (5日周期)
        acc_slope_5 = self.helper._get_safe_series(df, 'SLOPE_5_accumulation_score_D', 0.0)
        acc_accel_5 = self.helper._get_safe_series(df, 'ACCEL_5_accumulation_score_D', 0.0)
        acc_jerk_5 = self.helper._get_safe_series(df, 'JERK_5_accumulation_score_D', 0.0)
        tactical_resonance = ((acc_slope_5 > 0) & (acc_accel_5 > 0) & (acc_jerk_5 > 0)).astype(float)
        # 战略相位共振 (13日周期)
        acc_slope_13 = self.helper._get_safe_series(df, 'SLOPE_13_accumulation_score_D', 0.0)
        strategic_resonance = (acc_slope_13 > 0).astype(float)
        rdi_signals["phase_resonance_intensity"] = self.helper._normalize_series(
            tactical_resonance * 0.7 + strategic_resonance * 0.3, df_index, bipolar=False
        )
        geom_curvature = self.helper._get_safe_series(df, 'GEOM_ARC_CURVATURE_D', 0.0)
        flow_outcome = normalized_signals.get("data_flow_outcome", pd.Series(0.0, index=df_index))
        rdi_signals["geom_flow_resonance"] = _robust_geometric_mean({
            "curvature": self.helper._normalize_series(geom_curvature, df_index, bipolar=False),
            "flow": flow_outcome
        }, {"curvature": 0.5, "flow": 0.5}, df_index)
        price_slope = self.helper._get_safe_series(df, 'SLOPE_5_close_D', 0.0)
        price_entropy = self.helper._get_safe_series(df, 'PRICE_ENTROPY_D', 0.0)
        vpa_eff = self.helper._get_safe_series(df, 'VPA_MF_ADJUSTED_EFF_D', 0.0)
        eff_divergence = (price_slope < 0).astype(float) * self.helper._normalize_series(vpa_eff, df_index, bipolar=False)
        stealth_accel = self.helper._get_safe_series(df, 'ACCEL_5_stealth_flow_ratio_D', 0.0)
        entropy_divergence = self.helper._normalize_series(price_entropy, df_index, bipolar=False) * (stealth_accel > 0).astype(float)
        rdi_signals["advanced_divergence"] = _robust_geometric_mean({
            "eff_div": eff_divergence,
            "entropy_div": entropy_divergence
        }, {"eff_div": 0.6, "entropy_div": 0.4}, df_index)
        jerk_inflection = (acc_jerk > 0) & (acc_jerk.shift(1) <= 0)
        gap_momentum = self.helper._get_safe_series(df, 'GAP_MOMENTUM_STRENGTH_D', 0.0)
        rdi_signals["inflection_point"] = self.helper._normalize_series(
            jerk_inflection.astype(float) * (1 + self.helper._normalize_series(gap_momentum, df_index, bipolar=False)),
            df_index, bipolar=False
        )
        return rdi_signals

    def _calculate_dynamic_efficiency_baseline(self, context_signals: Dict[str, pd.Series], df_index: pd.Index, config: Dict) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """
        【V9.0.0 · 情绪极性分流版 · 动态基准线重构】
        优化情绪斜率对基准线的影响逻辑，从“盲目惩罚”转向“极性增益”。
        - 核心修正: 情绪斜率(sentiment_slope) > 0 时视为环境向好，下调基准线释放信号空间。
        - 逻辑审计: 2026-02-08 更新，解决情绪转暖初期因“不稳定性”导致信号被过度压制的问题。
        """
        baseline_debug_values = {}
        base_baseline = get_param_value(config.get('dynamic_efficiency_baseline_params', {}).get('base_baseline'), 0.15)
        entropy_norm = context_signals.get("entropy_norm", pd.Series(0.5, index=df_index))
        sentiment_norm = context_signals.get("sentiment_norm", pd.Series(0.5, index=df_index))
        sentiment_slope = context_signals.get("sentiment_slope", pd.Series(0.0, index=df_index))
        # 1. 动能排名项 (维持 V8.9.0 物理底噪压制)
        chaos_term = (context_signals.get("entropy_rank", pd.Series(0.5, index=df_index)) - 0.5).clip(lower=0) * 0.25
        opportunity_term = (context_signals.get("compression_rank", pd.Series(0.5, index=df_index)) - 0.7).clip(lower=0) * 0.3
        # 2. 情绪极性分流逻辑
        # 情绪好转(+) -> 溢价 -> 降低基准线；情绪恶化(-) -> 惩罚 -> 抬高基准线
        sentiment_impact = np.where(sentiment_slope > 0, -sentiment_slope * 0.1, sentiment_slope.abs() * 0.15)
        # 3. 综合偏移量计算
        baseline_shift = (entropy_norm * 0.1 + chaos_term + sentiment_impact) - (opportunity_term + sentiment_norm * 0.05)
        dynamic_baseline = (base_baseline * (1 + np.tanh(baseline_shift))).clip(0.05, 0.35)
        baseline_debug_values.update({
            "chaos_rank_term": chaos_term,
            "sentiment_impact_term": pd.Series(sentiment_impact, index=df_index),
            "dynamic_baseline_value": dynamic_baseline
        })
        return dynamic_baseline, baseline_debug_values

    def _calculate_preliminary_score(self, df: pd.DataFrame, normalized_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], context_signals: Dict[str, pd.Series], df_index: pd.Index, config: Dict) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """
        【V8.6.0 · 背离溢价与博弈强化版 · 逻辑深化】
        引入量价背离溢价(dissonance_premium)，反向修正主力在负动能下的扫盘行为。
        - 核心修正: 当筹码锁定力强且吸筹爆发但动能为负时，将其识别为高置信度的拆单吸筹，赋予 20% 溢价。
        - 逻辑穿透: 结合长周期环境尺度与日内博弈强度，解决大盘股环境钝化问题。
        """
        preliminary_debug_values = {}
        is_leader = context_signals.get("is_leader", pd.Series(0.0, index=df_index))
        vpa_accel = context_signals.get("vpa_accel_5d", pd.Series(0.0, index=df_index))
        anomaly_intensity = context_signals.get("anomaly_intensity", pd.Series(0.0, index=df_index))
        flow_accel = context_signals.get("flow_accel", pd.Series(0.0, index=df_index))
        flow_mom = context_signals.get("flow_mom", pd.Series(0.0, index=df_index))
        ind_preheat = self.helper._normalize_series(self.helper._get_safe_series(df, 'industry_preheat_score_D', 0.0), df_index, bipolar=False)
        chip_accel_13 = normalized_signals.get('clean_ACCEL_13_chip_concentration_ratio_D', pd.Series(0.0, index=df_index))
        entropy_slope_13 = normalized_signals.get('clean_SLOPE_13_chip_entropy_D', pd.Series(0.0, index=df_index))
        locking_force = pd.concat([chip_accel_13, (-1 * entropy_slope_13).clip(0, 1)], axis=1).max(axis=1)
        # 背离溢价逻辑: 锁定力强 + 动量微负 = 隐秘扫盘证据
        dissonance_mask = (locking_force > 0.7) & (flow_mom < 0)
        dissonance_premium = dissonance_mask.astype(float) * flow_mom.abs() * 0.2
        # 奇点模型整合溢价
        inst_jerk = normalized_signals.get('clean_JERK_5_SMART_MONEY_INST_NET_BUY_D', normalized_signals.get('clean_proxy_JERK_5_SMART_MONEY_INST_NET_BUY_D', pd.Series(0.0, index=df_index)))
        stealth_jerk = normalized_signals.get('clean_JERK_5_stealth_flow_ratio_D', pd.Series(0.0, index=df_index))
        intent_singularity = (inst_jerk.clip(lower=0) * 0.2 + stealth_jerk.clip(lower=0) * 0.2 + flow_accel.clip(lower=0) * 0.15 + ind_preheat * 0.25 + flow_mom.clip(lower=0) * 0.2 + dissonance_premium)
        base_authenticity = (vpa_accel * 0.6 + (1 - anomaly_intensity) * 0.4 + is_leader * 0.3).clip(0, 1)
        burst_authenticity = (base_authenticity + (locking_force > 0.8).astype(float) * 0.4).clip(0, 1)
        singularity_multiplier = 1 + (intent_singularity * burst_authenticity) * 0.3
        print(f"  -- [V8.6.0 背离溢价探针] Flow Mom: {flow_mom.loc[df_index[-1]]:.4f}, Dissonance Premium: {dissonance_premium.loc[df_index[-1]]:.4f}")
        preliminary_components = {
            "mtf_intensity": mtf_signals.get("mtf_intensity", pd.Series(0.0, index=df_index)),
            "intent_outcome": normalized_signals.get("data_intent_outcome", pd.Series(0.0, index=df_index)),
            "locking_force": locking_force,
            "energy_boost": normalized_signals.get("data_energy_outcome", pd.Series(0.0, index=df_index))
        }
        preliminary_score_base = _robust_geometric_mean(preliminary_components, {"mtf_intensity": 0.3, "intent_outcome": 0.3, "locking_force": 0.2, "energy_boost": 0.2}, df_index).fillna(0.0)
        final_score = (preliminary_score_base * singularity_multiplier * (1 + is_leader * 0.2)).clip(0, 1)
        preliminary_debug_values.update({"intent_singularity_boost": intent_singularity, "burst_authenticity": burst_authenticity, "locking_force_component": locking_force, "dissonance_premium": dissonance_premium})
        return final_score, preliminary_debug_values

    def _apply_quality_efficiency_calibration(self, dynamic_preliminary_score: pd.Series, holographic_validation_score: pd.Series, dynamic_efficiency_baseline: pd.Series, probe_ts: Optional[pd.Timestamp], context_signals: Dict[str, pd.Series]) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """
        【V6.9.0 · 散户共识过滤版 · EPC 模型深化】
        应用基于聪明钱背离的弹性幂次校准。
        - 核心逻辑: 引入 sm_divergence 惩罚项。如果吸筹信号伴随散户买入机构卖出的背离，则大幅调高校准幂次。
        - 判定思路: 散机构背离(sm_divergence)与背离斜率(slope)共振时，视为高度确定性的伪信号。
        """
        final_score_debug_values = {}
        df_index = dynamic_preliminary_score.index
        # 1. 基础质效偏差与基础因子 
        calibrated_holographic_score = holographic_validation_score - dynamic_efficiency_baseline
        norm_stability = context_signals.get("turnover_stability", pd.Series(0.5, index=df_index))
        data_quality = context_signals.get("data_quality", pd.Series(1.0, index=df_index))
        # 2. 散户合力背离惩罚 (Retail Herd Penalty)
        sm_divergence = context_signals.get("sm_divergence", pd.Series(0.0, index=df_index))
        sm_div_slope = context_signals.get("sm_divergence_slope", pd.Series(0.0, index=df_index))
        # 惩罚逻辑：基础背离分值 + 斜率增量（背离是否在加速）
        # 权重 0.4 确保该项能显著影响最终校准幂次 
        sm_divergence_penalty = (sm_divergence * 0.7 + sm_div_slope.clip(lower=0) * 0.3) * 0.4
        # 3. 综合 EPC 指数计算 (γ)
        # 惩罚项叠加：稳定性、数据质量、散户背离 
        penalty_term = (1 - norm_stability) * 0.2 + (1 - data_quality) * 0.3 + sm_divergence_penalty
        # 增益项：领涨属性、题材热度 
        theme_hotness = context_signals.get("theme_hotness", pd.Series(0.0, index=df_index))
        is_leader = context_signals.get("is_leader", pd.Series(0.0, index=df_index))
        gain_term = theme_hotness * 0.15 + is_leader * 0.1
        # γ 决定了初步分数被“压制”的剧烈程度
        calibration_exponent = (1 - calibrated_holographic_score + penalty_term - gain_term).clip(0.05, 5.0)
        # 4. 执行 EPC 转换并应用 
        final_score = dynamic_preliminary_score.pow(calibration_exponent).clip(0, 1).fillna(0.0)
        # 5. 探针记录
        if probe_ts is not None and probe_ts in df_index:
            final_score_debug_values.update({
                "epc_gamma": calibration_exponent.loc[probe_ts],
                "sm_divergence_penalty": sm_divergence_penalty.loc[probe_ts],
                "stability_impact": ((1 - norm_stability) * 0.2).loc[probe_ts],
                "pre_calibrated_val": dynamic_preliminary_score.loc[probe_ts]
            })
        return final_score, final_score_debug_values

    def _calculate_divergence_warning(self, final_score: pd.Series, normalized_signals: Dict[str, pd.Series], context_signals: Dict[str, pd.Series], df_index: pd.Index) -> pd.Series:
        """
        【V6.8.0 · 领涨韧性增强版 · 风险矩阵重构】
        计算结合领涨属性(IS_MARKET_LEADER_D)的信号背离与结构衰竭预警。
        - 核心逻辑: 引入领涨股韧性调节，对强势股的抛物线风险和资金流流出执行非线性宽容。
        - 判定思路: 领涨股的抛物线预警权重下调 40%，大单撤退容忍度提升 30%。
        - 风险权重: 通过 is_leader 动态抵消部分结构性风险评分。
        """
        # 1. 提取基础维度与领涨因子
        is_leader = context_signals.get("is_leader", pd.Series(0.0, index=df_index))
        parabolic_state = context_signals.get("parabolic_warning", pd.Series(0.0, index=df_index))
        parabolic_slope = context_signals.get("parabolic_slope", pd.Series(0.0, index=df_index))
        # 2. 计算结构衰竭强度 (Structural Exhaustion)
        # 领涨股通常具有更强的承接力，即便加速也不代表立即见顶
        leader_resilience_structure = (1 - is_leader * 0.4) 
        structural_exhaustion = (parabolic_state * (1 + parabolic_slope.clip(lower=0)) * leader_resilience_structure).clip(0, 1)
        # 3. 计算资金流背离强度 (Money Flow Divergence)
        sm_slope = normalized_signals.get('clean_SLOPE_5_SMART_MONEY_HM_NET_BUY_D', pd.Series(0.0, index=df_index))
        sm_accel = normalized_signals.get('clean_ACCEL_5_SMART_MONEY_HM_NET_BUY_D', pd.Series(0.0, index=df_index))
        # 领涨股允许更高频率的获利回吐，对其斜率负值执行软缩放
        leader_resilience_flow = (1 - is_leader * 0.3)
        money_retreat = (sm_slope.clip(upper=0).abs() * (1 + sm_accel.clip(upper=0).abs()) * leader_resilience_flow)
        # 4. 复合风险合成
        # 只有在吸筹强度较高且异常强度较高时，风险判定才生效
        score_mask = (final_score > 0.5).astype(float)
        anomaly_factor = 1 + context_signals.get("anomaly_intensity", pd.Series(0.0, index=df_index))
        warning_intensity = (money_retreat * 0.6 + structural_exhaustion * 0.4) * score_mask * anomaly_factor
        return warning_intensity.clip(0, 1)

    def _print_debug_info(self, method_name: str, probe_ts: pd.Timestamp, debug_output: Dict, _temp_debug_values: Dict, final_score: pd.Series):
        """
        统一打印调试信息。
        """
        for section, values in _temp_debug_values.items():
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- {section} ---"] = ""
            for key, series_or_value in values.items():
                if isinstance(series_or_value, pd.Series):
                    val = series_or_value.loc[probe_ts] if probe_ts in series_or_value.index else np.nan
                    debug_output[f"        {key}: {val:.4f}"] = ""
                elif isinstance(series_or_value, dict): # 处理嵌套字典，例如 preliminary_score_components
                    debug_output[f"        {key}:"] = ""
                    for sub_key, sub_series in series_or_value.items():
                        if isinstance(sub_series, pd.Series):
                            sub_val = sub_series.loc[probe_ts] if probe_ts in sub_series.index else np.nan
                            debug_output[f"          {sub_key}: {sub_val:.4f}"] = ""
                        else:
                            debug_output[f"          {sub_key}: {sub_series}"] = ""
                else:
                    debug_output[f"        {key}: {series_or_value}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 拆单吸筹强度诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
        self.helper._print_debug_output(debug_output)





