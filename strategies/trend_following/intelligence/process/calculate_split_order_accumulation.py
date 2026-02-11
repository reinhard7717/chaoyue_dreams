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

    def _internal_normalize(self, series: pd.Series, mode: str = 'unipolar', window: int = 60) -> pd.Series:
        """
        【V9.2.0 · 自适应场景归一化内核】
        替代通用 Helper，针对不同物理含义的信号采用特定的归一化算法。
        - bipolar: Tanh-Robust-Z 变换，适用于资金流、动量、导数 (输出 -1~1)。
        - unipolar: 滚动 MinMax 缩放，适用于筹码、稳定性、评分 (输出 0~1)。
        - rank: 滚动百分位排名，适用于环境慢变量 (输出 0~1)。
        - raw_clip: 直接截断，适用于已归一化的比率 (输出 0~1)。
        """
        if series.empty: return series
        if mode == 'bipolar':
            roll_mean = series.rolling(window=window, min_periods=1).mean()
            roll_std = series.rolling(window=window, min_periods=1).std().replace(0, 1e-6)
            z_score = (series - roll_mean) / roll_std
            return np.tanh(z_score * 0.5) # 0.5系数控制压缩斜率，保留更多中间细节
        elif mode == 'rank':
            return series.rolling(window=window, min_periods=1).rank(pct=True).fillna(0.5)
        elif mode == 'raw_clip':
            return series.clip(0, 1)
        else: # unipolar (default)
            roll_min = series.rolling(window=window, min_periods=1).min()
            roll_max = series.rolling(window=window, min_periods=1).max()
            denom = (roll_max - roll_min).replace(0, 1e-6)
            return ((series - roll_min) / denom).clip(0, 1)

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
            # self._print_debug_info(method_name, probe_ts, debug_output, _temp_debug_values, adjusted_final_score)
        return adjusted_final_score.astype(np.float32)

    def _calculate_synthetic_smart_proxy(self, df: pd.DataFrame, df_index: pd.Index) -> pd.Series:
        """
        【V9.2.0 · 专用归一化重构版 · SSMP代理合成】
        使用内部 _internal_normalize 替代通用 helper，针对资金流采用双极性(Bipolar)归一化。
        - 核心优化: 资金流采用 Tanh-Z 变换，更好地区分“温和流入”与“脉冲爆发”。
        """
        elg_buy = self.helper._get_safe_series(df, 'buy_elg_amount_D', 0.0)
        mf_net = self.helper._get_safe_series(df, 'net_mf_amount_D', 0.0)
        md_buy = self.helper._get_safe_series(df, 'buy_md_amount_D', 0.0)
        lg_buy = self.helper._get_safe_series(df, 'buy_lg_amount_D', 0.0)
        stealth_ratio = self.helper._get_safe_series(df, 'stealth_flow_ratio_D', 0.0)
        hidden_inst_flow = (md_buy + lg_buy) * stealth_ratio
        flow_eff = self.helper._get_safe_series(df, 'flow_efficiency_D', 0.0)
        energy_flow = self.helper._get_safe_series(df, 'net_energy_flow_D', 0.0)
        # 使用专用归一化逻辑
        proxy_components = {
            "explicit": self._internal_normalize(elg_buy + mf_net, mode='bipolar', window=21),
            "hidden": self._internal_normalize(hidden_inst_flow, mode='bipolar', window=21),
            "efficiency": self._internal_normalize(flow_eff, mode='unipolar', window=21),
            "energy": self._internal_normalize(energy_flow, mode='bipolar', window=21)
        }
        print(f"  -- [V9.2.0 归一化探针] Eff Mean: {proxy_components['efficiency'].mean():.4f}, Energy Mean: {proxy_components['energy'].mean():.4f}")
        weights = {"explicit": 0.3, "hidden": 0.3, "efficiency": 0.2, "energy": 0.2}
        # 注意: bipolar 信号在几何平均前需映射回 0-1 空间，或者使用加权平均。
        # 这里 SSMP 需要保留方向性，但 _robust_geometric_mean 通常处理正数。
        # 策略: SSMP 作为基础信号，保持原始符号更有意义，但在后续处理中会再次归一化。
        # 这里改用加权平均以保留负值信息 (资金流出)
        ssmp_score = (
            proxy_components["explicit"] * weights["explicit"] +
            proxy_components["hidden"] * weights["hidden"] +
            (proxy_components["efficiency"] * 2 - 1) * weights["efficiency"] + # 映射 unipolar 到 -1~1
            proxy_components["energy"] * weights["energy"]
        )
        return ssmp_score.fillna(0.0)

    def _get_and_normalize_signals(self, df: pd.DataFrame, mtf_slope_accel_weights: Dict, method_name: str) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series], Dict[str, pd.Series], Dict[str, pd.Series]]:
        """
        【V9.2.0 · 自适应归一化全维版 · 信号获取】
        全面移除 helper 归一化，采用内部 Bipolar/Unipolar/Rank 混合策略。
        - 核心修正: 导数信号统一采用 Bipolar Tanh 归一化，解决量纲差异。
        - 逻辑审计: 2026-02-08 更新，针对环境慢变量强制使用 Rank 模式。
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
            'flow_momentum_5d_D', 'uptrend_strength_D', 'game_intensity_D', 'intraday_main_force_activity_D', 'intraday_price_range_ratio_D'
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
                    # 关键修改: 导数信号使用 Bipolar 归一化
                    normalized_signals[f'clean_{prefix}{col_name}'] = self._internal_normalize(clean_deriv, mode='bipolar', window=21)
        intent_comps = {
            "explicit": self._internal_normalize(raw_signals['buy_elg_amount_D'] + raw_signals['net_mf_amount_D'], mode='bipolar', window=21),
            "hidden_slope": normalized_signals.get('clean_proxy_SLOPE_5_SMART_MONEY_INST_NET_BUY_D', pd.Series(0.0, index=df_index)),
            "consistency": self._internal_normalize(raw_signals['flow_consistency_D'], mode='unipolar', window=21)
        }
        # Intent 几何平均需要正数，将 Bipolar (-1~1) 映射回 (0~1)
        intent_comps_mapped = {k: (v * 0.5 + 0.5) if k != 'consistency' else v for k, v in intent_comps.items()}
        normalized_signals["data_intent_outcome"] = _robust_geometric_mean(intent_comps_mapped, {"explicit": 0.3, "hidden_slope": 0.4, "consistency": 0.3}, df_index).fillna(0.0)
        # 振幅收敛逻辑
        range_ratio = raw_signals['intraday_price_range_ratio_D']
        range_rank = self._internal_normalize(range_ratio, mode='rank', window=60)
        convergence_score = (1 - range_rank).clip(0, 1)
        entropy_slope_5 = normalized_signals.get('clean_SLOPE_5_chip_entropy_D', pd.Series(0.0, index=df_index))
        # 结构组件归一化
        struct_comps = {
            "stability": self._internal_normalize(raw_signals['chip_stability_D'], mode='unipolar', window=21),
            "golden_pit": raw_signals['STATE_GOLDEN_PIT_D'].astype(float),
            "concentration": self._internal_normalize(raw_signals['chip_concentration_ratio_D'], mode='unipolar', window=21),
            "entropy_reduction": (entropy_slope_5 * -1 * 0.5 + 0.5).clip(0, 1), # 导数映射
            "transfer_eff": self._internal_normalize(raw_signals['tick_chip_transfer_efficiency_D'], mode='unipolar', window=21),
            "convergence": convergence_score
        }
        normalized_signals["data_structure_outcome"] = _robust_geometric_mean(struct_comps, {"stability": 0.15, "golden_pit": 0.2, "concentration": 0.2, "entropy_reduction": 0.15, "transfer_eff": 0.15, "convergence": 0.15}, df_index).fillna(0.0)
        vpa_eff_slope_5 = self.helper._get_safe_series(df, 'SLOPE_5_VPA_MF_ADJUSTED_EFF_D', 0.0)
        vpa_quality_norm = self._internal_normalize(vpa_eff_slope_5, mode='bipolar', window=21) * 0.5 + 0.5
        energy_comps = {
            "abs_energy": self._internal_normalize(raw_signals['absorption_energy_D'], mode='unipolar', window=21),
            "vpa_quality": vpa_quality_norm,
            "game_int": self._internal_normalize(raw_signals['game_intensity_D'], mode='unipolar', window=21)
        }
        normalized_signals["data_energy_outcome"] = _robust_geometric_mean(energy_comps, {"abs_energy": 0.3, "vpa_quality": 0.4, "game_int": 0.3}, df_index).fillna(0.0)
        mtf_signals = {
            "mtf_intensity": self.helper._get_mtf_slope_accel_score(df, 'accumulation_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            "mtf_cohesion": self.helper._get_mtf_cohesion_score(df, noise_sensitive_list, mtf_slope_accel_weights, df_index, method_name)
        }
        # 慢变量环境熵与压缩率逻辑 (底噪抑制)
        entropy = raw_signals['PRICE_ENTROPY_D']
        entropy_diff = entropy.diff(5).abs()
        entropy_anf = entropy.rolling(60).std() * 0.1
        entropy_rank = self._internal_normalize(entropy, mode='rank', window=60)
        stable_entropy_rank = entropy_rank.where(entropy_diff > entropy_anf, 0.5)
        compression = raw_signals['MA_POTENTIAL_COMPRESSION_RATE_D']
        comp_rank = self._internal_normalize(compression, mode='rank', window=60)
        stable_comp_rank = comp_rank.where(compression.diff(5).abs() > (compression.rolling(60).std() * 0.1), 0.5)
        context_signals = {
            "is_leader": raw_signals['IS_MARKET_LEADER_D'].astype(float),
            "mf_activity": self._internal_normalize(raw_signals['intraday_main_force_activity_D'], mode='unipolar', window=21),
            "adx_norm": self._internal_normalize(raw_signals['ADX_14_D'], mode='unipolar', window=21),
            "sentiment_norm": self._internal_normalize(raw_signals['market_sentiment_score_D'], mode='unipolar', window=21),
            "entropy_norm": self._internal_normalize(entropy, mode='unipolar', window=60),
            "entropy_rank": stable_entropy_rank,
            "compression_rank": stable_comp_rank,
            "sentiment_slope": normalized_signals.get('clean_SLOPE_5_market_sentiment_score_D', pd.Series(0.0, index=df_index)),
            "anomaly_intensity": self._internal_normalize(raw_signals['anomaly_intensity_D'], mode='unipolar', window=21),
            "vpa_accel_5d": self._internal_normalize(raw_signals['VPA_ACCELERATION_5D'], mode='bipolar', window=21) * 0.5 + 0.5,
            "flow_accel": self._internal_normalize(raw_signals['flow_acceleration_D'], mode='bipolar', window=21),
            "flow_mom": self._internal_normalize(raw_signals['flow_momentum_5d_D'], mode='bipolar', window=21),
            "turnover_stability": self._internal_normalize(raw_signals['TURNOVER_STABILITY_INDEX_D'], mode='unipolar', window=21),
            "data_quality": self._internal_normalize(raw_signals['tick_data_quality_score_D'], mode='unipolar', window=21),
            "theme_hotness": self._internal_normalize(raw_signals['THEME_HOTNESS_SCORE_D'], mode='unipolar', window=21),
            "parabolic_warning": self._internal_normalize(raw_signals['STATE_PARABOLIC_WARNING_D'], mode='raw_clip'),
            "parabolic_slope": normalized_signals.get('clean_SLOPE_5_STATE_PARABOLIC_WARNING_D', pd.Series(0.0, index=df_index)),
            "sm_divergence": self._internal_normalize(raw_signals['SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D'], mode='bipolar', window=21),
            "sm_divergence_slope": normalized_signals.get('clean_SLOPE_5_SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D', pd.Series(0.0, index=df_index)),
            "synergy_slope": normalized_signals.get('clean_SLOPE_5_SMART_MONEY_SYNERGY_BUY_D', pd.Series(0.0, index=df_index)),
            "hm_attack": self._internal_normalize(raw_signals['HM_COORDINATED_ATTACK_D'], mode='unipolar', window=21),
            "hm_attack_slope": normalized_signals.get('clean_SLOPE_5_HM_COORDINATED_ATTACK_D', pd.Series(0.0, index=df_index)),
            "intraday_acc_conf": self._internal_normalize(raw_signals['intraday_accumulation_confidence_D'], mode='unipolar', window=21),
            "chip_transfer_eff": self._internal_normalize(raw_signals['tick_chip_transfer_efficiency_D'], mode='unipolar', window=21),
            "range_ratio_rank": range_rank,
            "industry_strength_rank": self._internal_normalize(self.helper._get_safe_series(df, 'industry_strength_rank_D', 0.5), mode='rank', window=60)
        }
        print(f"  -- [V9.2.0 自适应归一化探针] Structure Score: {normalized_signals['data_structure_outcome'].mean():.4f}, Intent Score: {normalized_signals['data_intent_outcome'].mean():.4f}")
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
        【V9.1.0 · 波动率门控集成版 · 全息验证深化】
        引入日内振幅比排名(range_ratio_rank)作为波动率门控，修正全息得分。
        - 验证逻辑: 整合活跃度、协同性、置信度与振幅收敛，形成四维真实性校验。
        - 核心修正: 若振幅排名过高(>0.9)，视为分歧剧烈，施加惩罚；若收敛(<0.3)，给予奖励。
        """
        holographic_debug_values = {}
        is_leader = context_signals.get("is_leader", pd.Series(0.0, index=df_index))
        intraday_acc_conf = context_signals.get("intraday_acc_conf", pd.Series(1.0, index=df_index))
        chip_transfer_eff = context_signals.get("chip_transfer_eff", pd.Series(0.5, index=df_index))
        mf_activity = context_signals.get("mf_activity", pd.Series(0.5, index=df_index))
        synergy_slope = context_signals.get("synergy_slope", pd.Series(0.0, index=df_index))
        range_rank = context_signals.get("range_ratio_rank", pd.Series(0.5, index=df_index))
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
        # 波动率门控: 振幅大(Rank>0.9) -> 惩罚; 振幅小(Rank<0.3) -> 奖励
        volatility_gate = np.where(range_rank > 0.9, 0.9, np.where(range_rank < 0.3, 1.05, 1.0))
        final_correction = (1 + resonance_bonus * 0.2 + synergy_bonus) * (0.7 + 0.3 * intraday_acc_conf) * (0.8 + 0.2 * chip_transfer_eff) * (0.85 + 0.15 * mf_activity) * volatility_gate
        holographic_validation_score = (_robust_geometric_mean({"state": holographic_state_score, "trend": holographic_trend_score}, {"state": 0.5, "trend": 0.5}, df_index) * final_correction).clip(0, 1)
        print(f"  -- [全息波动率门控探针] Range Rank: {range_rank.mean():.4f}, Gate Multiplier: {volatility_gate.mean():.4f}")
        direction = holographic_trend_components["acc_slope"].apply(lambda x: 1 if x >= 0 else -1)
        if probe_ts is not None and probe_ts in df_index:
            holographic_debug_values.update({"volatility_gate": float(volatility_gate[df_index.get_loc(probe_ts)]) if isinstance(volatility_gate, np.ndarray) else 1.0, "holographic_val_final": holographic_validation_score.loc[probe_ts]})
        return (holographic_validation_score * direction).clip(-1, 1), holographic_debug_values

    def _calculate_rdi_signals(self, df: pd.DataFrame, normalized_signals: Dict[str, pd.Series], df_index: pd.Index, config: Dict) -> Dict[str, pd.Series]:
        """
        【V9.2.0 · 自适应归一化穿透版 · RDI 信号】
        使用内部归一化逻辑重构 RDI 信号计算，确保物理量纲的统一。
        - 核心修正: 曲率使用 Bipolar，熵使用 Rank，效率使用 Bipolar。
        """
        rdi_signals = {}
        # 1. 相位共振 (逻辑判定无需归一化，直接使用符号)
        acc_slope_5 = self.helper._get_safe_series(df, 'SLOPE_5_accumulation_score_D', 0.0)
        acc_accel_5 = self.helper._get_safe_series(df, 'ACCEL_5_accumulation_score_D', 0.0)
        acc_jerk_5 = self.helper._get_safe_series(df, 'JERK_5_accumulation_score_D', 0.0)
        tactical_resonance = ((acc_slope_5 > 0) & (acc_accel_5 > 0) & (acc_jerk_5 > 0)).astype(float)
        acc_slope_13 = self.helper._get_safe_series(df, 'SLOPE_13_accumulation_score_D', 0.0)
        strategic_resonance = (acc_slope_13 > 0).astype(float)
        # 强度归一化: 共振信号本身是 0/1，无需复杂归一化，直接 Unipolar 即可
        rdi_signals["phase_resonance_intensity"] = self._internal_normalize(
            tactical_resonance * 0.7 + strategic_resonance * 0.3, mode='unipolar', window=21
        )
        # 2. 几何流向共振 (曲率有正负 -> Bipolar)
        geom_curvature = self.helper._get_safe_series(df, 'GEOM_ARC_CURVATURE_D', 0.0)
        curvature_norm = self._internal_normalize(geom_curvature, mode='bipolar', window=60)
        # 将 Bipolar (-1~1) 映射回 (0~1) 用于几何平均
        curvature_mapped = curvature_norm * 0.5 + 0.5
        flow_outcome = normalized_signals.get("data_flow_outcome", pd.Series(0.0, index=df_index))
        rdi_signals["geom_flow_resonance"] = _robust_geometric_mean({
            "curvature": curvature_mapped,
            "flow": flow_outcome
        }, {"curvature": 0.5, "flow": 0.5}, df_index)
        # 3. 高级背离 (Advanced Divergence)
        price_slope = self.helper._get_safe_series(df, 'SLOPE_5_close_D', 0.0)
        price_entropy = self.helper._get_safe_series(df, 'PRICE_ENTROPY_D', 0.0)
        vpa_eff = self.helper._get_safe_series(df, 'VPA_MF_ADJUSTED_EFF_D', 0.0)
        # 效率背离: 价格跌但VPA效率高 -> Bipolar 归一化效率
        vpa_eff_norm = self._internal_normalize(vpa_eff, mode='bipolar', window=21)
        # 仅关注正向效率背离 (VPA > 0)
        eff_divergence = (price_slope < 0).astype(float) * vpa_eff_norm.clip(lower=0)
        # 熵背离: 隐秘资金加速但熵值高(无序) -> Rank 归一化熵
        entropy_norm = self._internal_normalize(price_entropy, mode='rank', window=60)
        stealth_accel = self.helper._get_safe_series(df, 'ACCEL_5_stealth_flow_ratio_D', 0.0)
        entropy_divergence = entropy_norm * (stealth_accel > 0).astype(float)
        rdi_signals["advanced_divergence"] = _robust_geometric_mean({
            "eff_div": eff_divergence,
            "entropy_div": entropy_divergence
        }, {"eff_div": 0.6, "entropy_div": 0.4}, df_index)
        # 4. 拐点确认 (Inflection Point)
        gap_momentum = self.helper._get_safe_series(df, 'GAP_MOMENTUM_STRENGTH_D', 0.0)
        # 缺口动能 -> Unipolar (非负)
        gap_norm = self._internal_normalize(gap_momentum, mode='unipolar', window=21)
        jerk_inflection = ((acc_jerk_5 > 0) & (acc_jerk_5.shift(1) <= 0)).astype(float)
        rdi_signals["inflection_point"] = self._internal_normalize(
            jerk_inflection * (1 + gap_norm), mode='unipolar', window=5
        )
        return rdi_signals

    def _calculate_dynamic_efficiency_baseline(self, context_signals: Dict[str, pd.Series], df_index: pd.Index, config: Dict) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """
        【V9.2.0 · 自适应归一化版 · 动态基准线】
        针对情绪斜率使用内部 Bipolar 归一化，确保极性分流的物理意义。
        """
        baseline_debug_values = {}
        base_baseline = get_param_value(config.get('dynamic_efficiency_baseline_params', {}).get('base_baseline'), 0.15)
        entropy_norm = context_signals.get("entropy_norm", pd.Series(0.5, index=df_index))
        sentiment_norm = context_signals.get("sentiment_norm", pd.Series(0.5, index=df_index))
        # 使用 context 中已归一化的 Rank
        entropy_rank = context_signals.get("entropy_rank", pd.Series(0.5, index=df_index))
        compression_rank = context_signals.get("compression_rank", pd.Series(0.5, index=df_index))
        chaos_term = (entropy_rank - 0.5).clip(lower=0) * 0.25
        opportunity_term = (compression_rank - 0.7).clip(lower=0) * 0.3
        # 关键修正: 情绪斜率需要保留方向性，使用 Bipolar 归一化
        # 这里的 sentiment_slope 已经是 clean_SLOPE_5...，是原始值
        raw_sentiment_slope = context_signals.get("sentiment_slope", pd.Series(0.0, index=df_index))
        sentiment_slope_norm = self._internal_normalize(raw_sentiment_slope, mode='bipolar', window=21)
        # 极性分流: 正向(>0)降低基准线，负向(<0)提高基准线
        # 注意: sentiment_slope_norm 范围 -1~1
        sentiment_impact = np.where(sentiment_slope_norm > 0, -sentiment_slope_norm * 0.1, sentiment_slope_norm.abs() * 0.15)
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
        【V9.2.0 · 自适应归一化穿透版 · 初步评分】
        使用内部归一化逻辑重构初步评分计算，确保所有输入信号的物理量纲一致。
        - 核心修正: 行业预热使用 internal_normalize(unipolar) 处理。
        - 逻辑适配: 适配上游 Bipolar 信号，利用 clip(lower=0) 精确提取正向动能。
        """
        preliminary_debug_values = {}
        # 1. 提取环境信号 (Context signals 已经在 _get_and_normalize_signals 中经过内部归一化)
        is_leader = context_signals.get("is_leader", pd.Series(0.0, index=df_index))
        vpa_accel = context_signals.get("vpa_accel_5d", pd.Series(0.0, index=df_index))
        anomaly_intensity = context_signals.get("anomaly_intensity", pd.Series(0.0, index=df_index))
        flow_accel = context_signals.get("flow_accel", pd.Series(0.0, index=df_index))
        flow_mom = context_signals.get("flow_mom", pd.Series(0.0, index=df_index))
        # 2. 行业预热归一化 (迁移至内部逻辑)
        # 行业预热是 0-100 的分数，使用 Unipolar 归一化
        ind_preheat_raw = self.helper._get_safe_series(df, 'industry_preheat_score_D', 0.0)
        ind_preheat = self._internal_normalize(ind_preheat_raw, mode='unipolar', window=21)
        # 3. 结构锁定力 (Locking Force)
        # chip_accel_13 和 entropy_slope_13 来自 normalized_signals，是 Bipolar (-1~1)
        chip_accel_13 = normalized_signals.get('clean_ACCEL_13_chip_concentration_ratio_D', pd.Series(0.0, index=df_index))
        entropy_slope_13 = normalized_signals.get('clean_SLOPE_13_chip_entropy_D', pd.Series(0.0, index=df_index))
        # 逻辑适配:
        # chip_accel > 0 代表集中度加速 (好)
        # entropy_slope < 0 代表熵减 (好) -> -1 * slope > 0
        # max 算子会自动忽略负值 (扩散/熵增)，除非两者都为负 (那就没锁定力)
        locking_force = pd.concat([chip_accel_13, (-1 * entropy_slope_13)], axis=1).max(axis=1).clip(lower=0)
        # 4. 背离溢价 (Dissonance Premium)
        # flow_mom 是 Bipolar，< 0 代表流出。
        dissonance_mask = (locking_force > 0.7) & (flow_mom < 0)
        dissonance_premium = dissonance_mask.astype(float) * flow_mom.abs() * 0.25
        # 5. 五维奇点模型 (Intent Singularity)
        hm_attack_slope = context_signals.get("hm_attack_slope", pd.Series(0.0, index=df_index)) # Bipolar
        phase_transition_boost = (locking_force > 0.6).astype(float) * hm_attack_slope.clip(lower=0) * 0.5
        inst_jerk = normalized_signals.get('clean_JERK_5_SMART_MONEY_INST_NET_BUY_D', normalized_signals.get('clean_proxy_JERK_5_SMART_MONEY_INST_NET_BUY_D', pd.Series(0.0, index=df_index)))
        stealth_jerk = normalized_signals.get('clean_JERK_5_stealth_flow_ratio_D', pd.Series(0.0, index=df_index))
        # Bipolar 信号通过 clip(lower=0) 提取正向爆发力
        intent_singularity = (
            inst_jerk.clip(lower=0) * 0.2 + 
            stealth_jerk.clip(lower=0) * 0.2 + 
            flow_accel.clip(lower=0) * 0.15 + 
            ind_preheat * 0.25 + 
            flow_mom.clip(lower=0) * 0.2 + 
            dissonance_premium
        )
        # 6. 爆发真实性 (Burst Authenticity)
        # vpa_accel 是 Bipolar 映射到 0~1 的 (in context), anomaly 是 Unipolar
        # locking_force 已经 clip 到 0~1 (近似)
        base_authenticity = (vpa_accel * 0.6 + (1 - anomaly_intensity) * 0.4 + is_leader * 0.3).clip(0, 1)
        burst_authenticity = (base_authenticity + (locking_force > 0.8).astype(float) * 0.4).clip(0, 1)
        singularity_multiplier = 1 + (intent_singularity * burst_authenticity + phase_transition_boost) * 0.35
        # 7. 能量维度增强
        energy_raw = normalized_signals.get("data_energy_outcome", pd.Series(0.0, index=df_index))
        vpa_quality = self.helper._get_safe_series(df, 'SLOPE_5_VPA_MF_ADJUSTED_EFF_D', 0.0)
        # 重新归一化以确保维度对齐
        vpa_quality_norm = self._internal_normalize(vpa_quality, mode='bipolar', window=21) * 0.5 + 0.5
        energy_boosted = pd.concat([energy_raw, vpa_quality_norm * 0.8], axis=1).max(axis=1).clip(0, 1)
        # 8. 综合几何平均
        preliminary_components = {
            "mtf_intensity": mtf_signals.get("mtf_intensity", pd.Series(0.0, index=df_index)),
            "intent_outcome": normalized_signals.get("data_intent_outcome", pd.Series(0.0, index=df_index)),
            "locking_force": locking_force,
            "energy_boost": energy_boosted
        }
        preliminary_score_base = _robust_geometric_mean(preliminary_components, {"mtf_intensity": 0.3, "intent_outcome": 0.3, "locking_force": 0.2, "energy_boost": 0.2}, df_index).fillna(0.0)
        final_score = (preliminary_score_base * singularity_multiplier * (1 + is_leader * 0.2)).clip(0, 1)
        preliminary_debug_values.update({
            "intent_singularity_boost": intent_singularity,
            "burst_authenticity": burst_authenticity,
            "locking_force_component": locking_force,
            "energy_boost_final": energy_boosted,
            "dissonance_premium": dissonance_premium
        })
        print(f"  -- [V9.2.0 初步评分探针] Base: {preliminary_score_base.loc[df_index[-1]]:.4f}, Multiplier: {singularity_multiplier.loc[df_index[-1]]:.4f}")
        return final_score, preliminary_debug_values

    def _apply_quality_efficiency_calibration(self, dynamic_preliminary_score: pd.Series, holographic_validation_score: pd.Series, dynamic_efficiency_baseline: pd.Series, probe_ts: Optional[pd.Timestamp], context_signals: Dict[str, pd.Series]) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """
        【V9.2.0 · 自适应归一化穿透版 · 质效校准】
        使用内部归一化逻辑处理校准因子，确保惩罚项与增益项的物理意义准确。
        - 核心修正: 处理 Bipolar 类型的散户背离(sm_divergence)，确保负值(机构买入)产生正向增益。
        - 行业位次: 确保 industry_strength_rank 经过 Rank 模式处理。
        """
        final_score_debug_values = {}
        df_index = dynamic_preliminary_score.index
        # 1. 基础质效偏差
        calibrated_holographic_score = holographic_validation_score - dynamic_efficiency_baseline
        # 2. 提取并确认归一化状态
        norm_stability = context_signals.get("turnover_stability", pd.Series(0.5, index=df_index)) # Unipolar
        data_quality = context_signals.get("data_quality", pd.Series(1.0, index=df_index)) # Unipolar
        # 行业强度排名 (在 context 中已经是 Rank 模式)
        ind_rank = context_signals.get("industry_strength_rank", pd.Series(0.5, index=df_index))
        # 3. 散户共识惩罚 (Smart Money Divergence Penalty)
        # sm_divergence 是 Bipolar (-1~1). 
        # > 0: 机构卖/散户买 (Bad) -> 惩罚
        # < 0: 机构买/散户卖 (Good) -> 奖励 (负惩罚)
        sm_divergence = context_signals.get("sm_divergence", pd.Series(0.0, index=df_index))
        sm_div_slope = context_signals.get("sm_divergence_slope", pd.Series(0.0, index=df_index))
        # 逻辑保持: 0.7 * Base + 0.3 * Slope(accelerating divergence)
        sm_divergence_penalty = (sm_divergence * 0.7 + sm_div_slope.clip(lower=0) * 0.3) * 0.4
        # 4. 综合 EPC 指数 (γ)
        # Penalty Term: 稳定性差(+), 质量差(+), 背离大(+)
        # sm_divergence_penalty 若为负，会减小 penalty_term，从而减小 gamma，提升分数。符合逻辑。
        penalty_term = (1 - norm_stability) * 0.2 + (1 - data_quality) * 0.2 + sm_divergence_penalty
        # Gain Term: 题材热度(+), 领涨(+), 行业位次高(+)
        theme_hotness = context_signals.get("theme_hotness", pd.Series(0.0, index=df_index)) # Unipolar
        is_leader = context_signals.get("is_leader", pd.Series(0.0, index=df_index))
        # ind_rank 越小越好 (Rank 0 = Top 1). (1 - ind_rank) 越大越好.
        gain_term = theme_hotness * 0.15 + is_leader * 0.1 + (1 - ind_rank) * 0.1
        # 计算 gamma
        # calibration_exponent < 1 -> 提分 (Score^0.5 > Score)
        # calibration_exponent > 1 -> 压分 (Score^2 < Score)
        calibration_exponent = (1 - calibrated_holographic_score + penalty_term - gain_term).clip(0.05, 5.0)
        final_score = dynamic_preliminary_score.pow(calibration_exponent).clip(0, 1).fillna(0.0)
        if probe_ts is not None and probe_ts in df_index:
            final_score_debug_values.update({
                "epc_gamma": calibration_exponent.loc[probe_ts],
                "sm_divergence_penalty": sm_divergence_penalty.loc[probe_ts],
                "industry_rank_gain": ((1 - ind_rank) * 0.1).loc[probe_ts],
                "pre_calibrated_val": dynamic_preliminary_score.loc[probe_ts]
            })
        return final_score, final_score_debug_values

    def _calculate_divergence_warning(self, final_score: pd.Series, normalized_signals: Dict[str, pd.Series], context_signals: Dict[str, pd.Series], df_index: pd.Index) -> pd.Series:
        """
        【V9.2.0 · 自适应归一化版 · 风险矩阵】
        使用内部归一化逻辑重构风险判定，确保不同量纲风险因子的可比性。
        - 核心修正: 资金流斜率使用 Bipolar，抛物线预警使用 Raw Clip。
        """
        is_leader = context_signals.get("is_leader", pd.Series(0.0, index=df_index))
        # 1. 结构衰竭 (Parabolic Warning 已经是归一化后的状态分，使用 Raw Clip)
        parabolic_state = self._internal_normalize(context_signals.get("parabolic_warning", pd.Series(0.0, index=df_index)), mode='raw_clip')
        # 斜率使用 Bipolar，因为可能转负(衰竭)
        parabolic_slope_raw = context_signals.get("parabolic_slope", pd.Series(0.0, index=df_index))
        parabolic_slope_norm = self._internal_normalize(parabolic_slope_raw, mode='bipolar', window=21)
        leader_resilience_structure = (1 - is_leader * 0.4)
        structural_exhaustion = (parabolic_state * (1 + parabolic_slope_norm.clip(lower=0)) * leader_resilience_structure).clip(0, 1)
        # 2. 资金流背离 (Money Flow Divergence)
        # 使用 normalized_signals 中已经 Bipolar 归一化过的信号
        sm_slope = normalized_signals.get('clean_SLOPE_5_SMART_MONEY_HM_NET_BUY_D', pd.Series(0.0, index=df_index))
        sm_accel = normalized_signals.get('clean_ACCEL_5_SMART_MONEY_HM_NET_BUY_D', pd.Series(0.0, index=df_index))
        # 资金撤退: 斜率 < 0 且 加速度 < 0 (Bipolar下均为负值)
        # 取绝对值计算撤退强度
        leader_resilience_flow = (1 - is_leader * 0.3)
        money_retreat = (sm_slope.clip(upper=0).abs() * (1 + sm_accel.clip(upper=0).abs()) * leader_resilience_flow)
        # 3. 复合风险合成
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





