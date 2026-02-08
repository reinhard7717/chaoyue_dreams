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
        【V7.2.0 · 拆单伪装还原版 · SSMP逻辑深化】
        通过特大单与隐秘流向系数，还原被伪装成中户/大单的机构拆单行为。
        - 核心逻辑: 利用 stealth_flow_ratio_D 对 buy_md_amount_D 执行“身份还原”。
        - 版本说明: 2026-02-08 优化，增强对机构拆单行为的穿透识别，区分中户跟风。
        """
        # 1. 提取显性机构资金 [cite: 1]
        elg_buy = self.helper._get_safe_series(df, 'buy_elg_amount_D', 0.0)
        mf_net = self.helper._get_safe_series(df, 'net_mf_amount_D', 0.0)
        # 2. 执行“拆单伪装还原”：中单/大单中蕴含的机构拆单部分 
        md_buy = self.helper._get_safe_series(df, 'buy_md_amount_D', 0.0)
        lg_buy = self.helper._get_safe_series(df, 'buy_lg_amount_D', 0.0)
        stealth_ratio = self.helper._get_safe_series(df, 'stealth_flow_ratio_D', 0.0)
        # 计算隐秘机构流：中单和大单在隐秘系数作用下的机构属性还原 
        hidden_inst_flow = (md_buy + lg_buy) * stealth_ratio
        # 3. 协同性因子 [cite: 1]
        synergy_buy = self.helper._get_safe_series(df, 'SMART_MONEY_SYNERGY_BUY_D', 0.0)
        proxy_components = {
            "explicit_inst": self.helper._normalize_series(elg_buy + mf_net, df_index, bipolar=True),
            "hidden_inst": self.helper._normalize_series(hidden_inst_flow, df_index, bipolar=True),
            "synergy": self.helper._normalize_series(synergy_buy, df_index, bipolar=False)
        }
        # 几何平均确保只有在显性或隐秘机构流具备规模，且有协同性时，信号才生效 
        return _robust_geometric_mean(proxy_components, {"explicit_inst": 0.4, "hidden_inst": 0.4, "synergy": 0.2}, df_index).fillna(0.0)

    def _get_and_normalize_signals(self, df: pd.DataFrame, mtf_slope_accel_weights: Dict, method_name: str) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series], Dict[str, pd.Series], Dict[str, pd.Series]]:
        """
        【V7.3.0 · 筹码熵控验证版 · 信号获取重构】
        整合筹码熵(chip_entropy_D)识别锁仓特征，多维印证拆单还原后的吸筹真实性。
        - 核心增强: 引入筹码熵减(Entropy Reduction)作为结构有序度的判定依据。
        - 逻辑穿透: 只有当拆单资金流入伴随筹码熵的负向斜率时，才判定为确定性的锁仓吸筹。
        """
        df_index = df.index
        # 1. 扩充原始指标：加入筹码熵、稳定性与各级买入额
        raw_df_columns = [
            'accumulation_score_D', 'stealth_flow_ratio_D', 'SMART_MONEY_INST_NET_BUY_D', 'IS_PARABOLIC_WARNING_D', 'SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D',
            'SMART_MONEY_HM_NET_BUY_D', 'SMART_MONEY_SYNERGY_BUY_D', 'buy_elg_amount_D',
            'buy_lg_amount_D', 'buy_md_amount_D', 'net_mf_amount_D', 'tick_large_order_net_D',
            'VPA_MF_ADJUSTED_EFF_D', 'absorption_energy_D', 'chip_concentration_ratio_D', 
            'chip_entropy_D', 'chip_stability_D', 'flow_intensity_D', 'GEOM_ARC_CURVATURE_D', 
            'IS_GOLDEN_PIT_D', 'IS_ROUNDING_BOTTOM_D', 'IS_MARKET_LEADER_D', 
            'TURNOVER_STABILITY_INDEX_D', 'tick_data_quality_score_D', 'THEME_HOTNESS_SCORE_D', 
            'PRICE_ENTROPY_D', 'MA_POTENTIAL_COMPRESSION_RATE_D', 'close_D'
        ]
        raw_signals = {col: self.helper._get_safe_series(df, col, 0.0, method_name=method_name) for col in raw_df_columns}
        normalized_signals = {}
        # 2. 计算合成机构代理信号 (SSMP V2: 包含拆单伪装还原逻辑)
        ssmp_proxy = self._calculate_synthetic_smart_proxy(df, df_index)
        # 3. 噪声敏感列表增加筹码熵维度
        noise_sensitive_list = [
            'accumulation_score_D', 'stealth_flow_ratio_D', 'SMART_MONEY_INST_NET_BUY_D', 
            'buy_elg_amount_D', 'chip_entropy_D'
        ]
        for indicator in noise_sensitive_list:
            base_val = raw_signals[indicator]
            # 逻辑自适应：若原子信号静默，切换至SSMP代理进行导数探测
            if indicator == 'SMART_MONEY_INST_NET_BUY_D' and base_val.std() < 1e-6:
                active_base = ssmp_proxy
                prefix = "proxy_"
            else:
                active_base = base_val
                prefix = ""
            dyn_eps = self._calculate_dynamic_epsilon(active_base, window=55, multiplier=0.05)
            for p in [5, 13]:
                for deriv_type in ['SLOPE', 'ACCEL', 'JERK']:
                    col_name = f'{deriv_type}_{p}_{indicator}'
                    raw_deriv = self.helper._get_safe_series(df, col_name, 0.0) if prefix == "" else active_base.diff(p)
                    clean_deriv = self._apply_derivative_denoising(raw_deriv, active_base, dyn_eps)
                    normalized_signals[f'clean_{prefix}{col_name}'] = self.helper._normalize_series(clean_deriv, df_index, bipolar=True)
        # 4. 语义化复合结果构建
        # 4a. 资金意图结果 (包含还原后的隐秘机构动能)
        intent_comps = {
            "explicit_inst": self.helper._normalize_series(raw_signals['buy_elg_amount_D'] + raw_signals['net_mf_amount_D'], df_index, bipolar=True),
            "hidden_inst_slope": normalized_signals.get('clean_proxy_SLOPE_5_SMART_MONEY_INST_NET_BUY_D', pd.Series(0.0, index=df_index)),
            "stealth": self.helper._normalize_series(raw_signals['stealth_flow_ratio_D'], df_index, bipolar=False)
        }
        normalized_signals["data_intent_outcome"] = _robust_geometric_mean(intent_comps, {"explicit_inst": 0.3, "hidden_inst_slope": 0.4, "stealth": 0.3}, df_index).fillna(0.0)
        # 4b. 结构结果深化 (重点引入熵减强度)
        # 逻辑：熵减强度 = -1 * 降噪后的熵斜率
        entropy_slope = normalized_signals.get('clean_SLOPE_5_chip_entropy_D', pd.Series(0.0, index=df_index))
        entropy_reduction = self.helper._normalize_series(-1 * entropy_slope, df_index, bipolar=False)
        struct_comps = {
            "stability": self.helper._normalize_series(raw_signals['chip_stability_D'], df_index, bipolar=False),
            "golden_pit": raw_signals['IS_GOLDEN_PIT_D'].astype(float),
            "concentration": self.helper._normalize_series(raw_signals['chip_concentration_ratio_D'], df_index, bipolar=False),
            "entropy_reduction": entropy_reduction
        }
        normalized_signals["data_structure_outcome"] = _robust_geometric_mean(
            struct_comps, 
            {"stability": 0.2, "golden_pit": 0.3, "concentration": 0.25, "entropy_reduction": 0.25}, 
            df_index
        ).fillna(0.0)
        # 4c. 其余复合维度维持原有动力学逻辑
        energy_comps = {"abs_energy": self.helper._normalize_series(raw_signals['absorption_energy_D'], df_index, bipolar=False)}
        normalized_signals["data_energy_outcome"] = _robust_geometric_mean(energy_comps, {"abs_energy": 1.0}, df_index).fillna(0.0)
        # 5. MTF 与 情境信号
        mtf_signals = {
            "mtf_intensity": self.helper._get_mtf_slope_accel_score(df, 'accumulation_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False),
            "mtf_cohesion": self.helper._get_mtf_cohesion_score(df, noise_sensitive_list, mtf_slope_accel_weights, df_index, method_name)
        }
        context_signals = {
            "is_leader": raw_signals['IS_MARKET_LEADER_D'].astype(float),
            "turnover_stability": self.helper._normalize_series(raw_signals['TURNOVER_STABILITY_INDEX_D'], df_index, bipolar=False),
            "data_quality": self.helper._normalize_series(raw_signals['tick_data_quality_score_D'], df_index, bipolar=False),
            "theme_hotness": self.helper._normalize_series(raw_signals['THEME_HOTNESS_SCORE_D'], df_index, bipolar=False),
            "parabolic_warning": self.helper._normalize_series(raw_signals['IS_PARABOLIC_WARNING_D'], df_index, bipolar=False),
            "parabolic_slope": normalized_signals.get('clean_SLOPE_5_IS_PARABOLIC_WARNING_D', pd.Series(0.0, index=df_index)),
            "sm_divergence": self.helper._normalize_series(raw_signals['SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D'], df_index, bipolar=False),
            "sm_divergence_slope": normalized_signals.get('clean_SLOPE_5_SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D', pd.Series(0.0, index=df_index)),
            "anomaly_intensity": self.helper._normalize_series(raw_signals['anomaly_intensity_D'], df_index, bipolar=False),
            "entropy_norm": self.helper._normalize_series(raw_signals['PRICE_ENTROPY_D'], df_index, bipolar=False),
            "entropy_slope": normalized_signals.get('clean_SLOPE_5_PRICE_ENTROPY_D', pd.Series(0.0, index=df_index)),
            "compression_accel": normalized_signals.get('clean_ACCEL_5_MA_POTENTIAL_COMPRESSION_RATE_D', pd.Series(0.0, index=df_index)),
            "sentiment_slope": normalized_signals.get('clean_SLOPE_5_market_sentiment_score_D', pd.Series(0.0, index=df_index))
        }
        return raw_signals, normalized_signals, mtf_signals, context_signals

    def _calculate_dynamic_epsilon(self, base_series: pd.Series, window: int = 55, multiplier: float = 0.05) -> pd.Series:
        """
        【V5.8.2 · 指标专属动态阈值计算】
        计算原子指标的内生底噪门槛。
        """
        rolling_std = base_series.rolling(window=window, min_periods=1).std().fillna(0.0)
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
        【V7.0.2 · 引用路径修正版】
        全息验证计算。修正了向 RDI 信号传递 df 的逻辑。
        """
        holographic_debug_values = {}
        params = config.get('holographic_validation_params', {})
        adx_norm = context_signals.get("adx_norm", pd.Series(0.5, index=df_index))
        is_leader = context_signals.get("is_leader", pd.Series(0.0, index=df_index))
        trend_weight_base = (0.4 + adx_norm * 0.2).clip(0.3, 0.6)
        state_weight_base = 1 - trend_weight_base
        holographic_state_components = {
            "flow": normalized_signals.get("data_flow_outcome", pd.Series(0.0, index=df_index)),
            "structure": normalized_signals.get("data_structure_outcome", pd.Series(0.0, index=df_index)),
            "geom": normalized_signals.get("data_geom_outcome", pd.Series(0.0, index=df_index))
        }
        holographic_state_score = _robust_geometric_mean({k: (v + 1) / 2 for k, v in holographic_state_components.items()}, {"flow": 0.4, "structure": 0.3, "geom": 0.3}, df_index)
        holographic_trend_components = {
            "flow_slope": normalized_signals.get("clean_SLOPE_5_stealth_flow_ratio_D", pd.Series(0.0, index=df_index)),
            "acc_slope": normalized_signals.get("clean_SLOPE_5_accumulation_score_D", pd.Series(0.0, index=df_index)),
            "vpa_slope": normalized_signals.get("clean_SLOPE_5_VPA_MF_ADJUSTED_EFF_D", pd.Series(0.0, index=df_index))
        }
        holographic_trend_score = _robust_geometric_mean({k: (v + 1) / 2 for k, v in holographic_trend_components.items()}, {"flow_slope": 0.4, "acc_slope": 0.4, "vpa_slope": 0.2}, df_index)
        # 修正: 向 RDI 方法传递 df
        rdi_signals = self._calculate_rdi_signals(df, normalized_signals, df_index, config)
        resonance_bonus = rdi_signals.get("phase_resonance_intensity", pd.Series(0.0, index=df_index))
        div_penalty = rdi_signals.get("advanced_divergence", pd.Series(0.0, index=df_index))
        overall_components = {"state": holographic_state_score, "trend": holographic_trend_score}
        overall_weights = {"state": state_weight_base, "trend": trend_weight_base}
        holographic_val_raw = _robust_geometric_mean(overall_components, overall_weights, df_index)
        final_correction = (1 + resonance_bonus * 0.2) * (1 - div_penalty * (0.3 - is_leader * 0.15))
        holographic_validation_score = (holographic_val_raw * final_correction).clip(0, 1)
        direction = holographic_trend_components["acc_slope"].apply(lambda x: 1 if x >= 0 else -1)
        if probe_ts is not None and probe_ts in df_index:
            holographic_debug_values.update({
                "holographic_state_score": holographic_state_score.loc[probe_ts],
                "holographic_trend_score": holographic_trend_score.loc[probe_ts],
                "phase_resonance": resonance_bonus.loc[probe_ts]
            })
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
        rdi_signals["phase_resonance_intensity"] = self.helper._normalize_series(phase_coherence * acc_accel, df_index, bipolar=False)
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
        【V6.2.0 · 环境动能投影模型版 · 动态基准线深化】
        计算前瞻性动态效率基准线。
        - 核心逻辑: 引入环境熵斜率、压缩加速度及情绪斜率，通过“投影补偿”预判环境变化对吸筹判定的影响。
        - 判定思路: 
            1. 熵增斜率(+) -> 噪音正在扩散 -> 抬高基准线。
            2. 压缩加速度(+) -> 变盘能量正在积聚 -> 主动调低基准线以捕捉早期渗入。
            3. 情绪变化率剧烈 -> 市场处于不稳定性阶段 -> 提高基准线以滤除情绪噪音。
        """
        baseline_debug_values = {}
        params = config.get('dynamic_efficiency_baseline_params', {})
        base_baseline = get_param_value(params.get('base_baseline'), 0.15)
        # 1. 基础环境状态项
        entropy_norm = context_signals.get("entropy_norm", pd.Series(0.5, index=df_index))
        sentiment_norm = context_signals.get("sentiment_norm", pd.Series(0.5, index=df_index))
        # 2. 动能投影项 (Momentum Projections)
        # 熵增斜率：若噪音在扩散，增加防御门槛
        entropy_slope = context_signals.get("entropy_slope", pd.Series(0.0, index=df_index))
        chaos_momentum = entropy_slope.clip(lower=0) * 0.2
        # 压缩加速度：若均线加速收敛，视为反转前夕，释放穿透空间
        comp_accel = context_signals.get("compression_accel", pd.Series(0.0, index=df_index))
        opportunity_momentum = comp_accel.clip(lower=0) * 0.15
        # 情绪不稳定性：情绪变动过快（无论正负）均增加不确定性
        sent_slope_abs = context_signals.get("sentiment_slope", pd.Series(0.0, index=df_index)).abs()
        instability_factor = sent_slope_abs * 0.1
        # 3. 综合偏移量计算 (非线性多维门控)
        # Shift = (静态状态 + 熵增动能 + 不稳定惩罚) - 压缩机会释放
        baseline_shift = (entropy_norm * 0.1 + chaos_momentum + instability_factor) - (opportunity_momentum + sentiment_norm * 0.05)
        # 4. 激活函数映射
        # 利用 tanh 确保偏移量平滑，并将基准线控制在物理合理区间
        dynamic_baseline = base_baseline * (1 + np.tanh(baseline_shift))
        dynamic_baseline = dynamic_baseline.clip(0.05, 0.35)
        if baseline_debug_values is not None:
            baseline_debug_values.update({
                "entropy_momentum_term": chaos_momentum,
                "compression_opportunity_term": opportunity_momentum,
                "instability_penalty_term": instability_factor,
                "final_baseline_shift": baseline_shift,
                "dynamic_baseline_value": dynamic_baseline
            })
        return dynamic_baseline, baseline_debug_values

    def _calculate_preliminary_score(self, df: pd.DataFrame, normalized_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], context_signals: Dict[str, pd.Series], df_index: pd.Index, config: Dict) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """
        【V7.0.2 · 引用路径修正版】
        计算初步吸筹强度。修复了对原始 df 的访问路径。
        """
        preliminary_debug_values = {}
        params = config.get('preliminary_score_params', {})
        is_leader = context_signals.get("is_leader", pd.Series(0.0, index=df_index))
        # 提取压制因子 (修正: 使用传入的 df)
        price_slope_5 = self.helper._get_safe_series(df, 'SLOPE_5_close_D', 0.0)
        vpa_eff_slope_5 = self.helper._get_safe_series(df, 'SLOPE_5_VPA_MF_ADJUSTED_EFF_D', 0.0)
        suppression_score = (price_slope_5.clip(upper=0).abs() * 0.7 + vpa_eff_slope_5.clip(upper=0).abs() * 0.3)
        # 获取核心导数 (修正: 已通过 normalized_signals 获取降噪后的值)
        inst_jerk = normalized_signals.get('clean_JERK_5_SMART_MONEY_INST_NET_BUY_D', pd.Series(0.0, index=df_index))
        stealth_jerk = normalized_signals.get('clean_JERK_5_stealth_flow_ratio_D', pd.Series(0.0, index=df_index))
        intent_singularity = (inst_jerk.clip(lower=0) * 0.6 + stealth_jerk.clip(lower=0) * 0.4)
        chip_accel = normalized_signals.get('clean_ACCEL_5_chip_concentration_ratio_D', pd.Series(0.0, index=df_index))
        locking_force = chip_accel.clip(lower=0)
        # 维度融合
        preliminary_components = {
            "mtf_intensity": mtf_signals.get("mtf_intensity", pd.Series(0.0, index=df_index)),
            "intent_outcome": normalized_signals.get("data_intent_outcome", pd.Series(0.0, index=df_index)),
            "locking_force": locking_force,
            "coordinated_attack": context_signals.get("coordinated_attack", pd.Series(0.0, index=df_index))
        }
        fusion_weights = get_param_value(params.get('fusion_weights'), {"mtf_intensity": 0.3, "intent_outcome": 0.3, "locking_force": 0.2, "coordinated_attack": 0.2})
        preliminary_score_base = _robust_geometric_mean(preliminary_components, fusion_weights, df_index).fillna(0.0)
        # 奇点修正
        singularity_multiplier = 1 + self.helper._normalize_series(intent_singularity, df_index, bipolar=False) * 0.25
        final_score = (preliminary_score_base * singularity_multiplier)
        final_score = (final_score * (1 + is_leader * 0.2)).clip(0, 1)
        preliminary_debug_values.update({
            "intent_singularity_boost": intent_singularity,
            "locking_force_component": locking_force,
            "preliminary_score_raw": preliminary_score_base
        })
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





