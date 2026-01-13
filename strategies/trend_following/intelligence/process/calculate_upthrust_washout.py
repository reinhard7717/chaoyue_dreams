# strategies\trend_following\intelligence\process\calculate_upthrust_washout.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

from strategies.trend_following.utils import (
    get_params_block, get_param_value, get_adaptive_mtf_normalized_score,
    get_adaptive_mtf_normalized_bipolar_score, normalize_score, _robust_geometric_mean
)
from strategies.trend_following.intelligence.process.helper import ProcessIntelligenceHelper

class CalculateUpthrustWashoutRelationship:
    """
    【V2.1.0 · 上冲回落洗盘甄别器】
    PROCESS_META_UPTHRUST_WASHOUT
    - 核心职责: 识别主力利用“上冲回落”阴线进行的洗盘行为。
    - 版本: 2.1.0
    """
    def __init__(self, strategy_instance, helper_instance: ProcessIntelligenceHelper):
        """
        初始化 CalculateUpthrustWashoutRelationship 处理器。
        参数:
            strategy_instance: 策略实例，用于访问全局配置和原子状态。
            helper_instance (ProcessIntelligenceHelper): 过程情报辅助工具实例。
        """
        self.strategy = strategy_instance
        self.helper = helper_instance
        self.params = self.helper.params
        self.debug_params = self.helper.debug_params
        self.probe_dates = self.helper.probe_dates

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.4.0 · 军令直达版】识别主力利用“上冲回落”阴线进行的洗盘行为。
        - 核心重构: 创立“强证优先”原则。废除对多种承接证据的加权平均，改为采用 max() 函数，
                      直接取“主动买盘”、“下影线强度”、“权力转移”三者中的最强者作为最终承接证据，
                      旨在识别任何一种足以扭转战局的决定性吸收力量。
        - 【新增】严格限制 K 线形态，只有当 K 线为“上冲回落”形态时才激活信号。
        - 【核心升级】所有依赖信号均从数据层原始指标派生，不再直接引用情报层原子信号。
        - 【核心升级】放宽市场上下文限制，移除对乖离率的严格要求。
        - 【核心升级】优化主力资金门控，从简单的当日净流入判断升级为基于累积资金流的评估。
        参数:
            df (pd.DataFrame): 包含所有原始数据的DataFrame。
            config (Dict): 诊断配置字典。
        返回:
            pd.Series: 上冲回落洗盘信号分数。
        """
        method_name = "CalculateUpthrustWashoutRelationship.calculate"
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
        debug_output = {}
        _temp_debug_values = {}
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
            debug_output[f"  -- [过程情报调试] {probe_ts.strftime('%Y-%m-%d')}: 正在计算上冲回落洗盘..."] = ""

        # 明确所有需要的原始信号
        required_signals = [
            'BIAS_21_D', 'pct_change_D', 'upward_impulse_purity_D', 'upper_shadow_selling_pressure_D',
            'active_buying_support_D', 'open_D', 'high_D', 'close_D', 'low_D',
            'main_force_net_flow_calibrated_D', 'trend_vitality_index_D', 'lower_shadow_absorption_strength_D',
            'net_sh_amount_calibrated_D', 'net_md_amount_calibrated_D', 'net_lg_amount_calibrated_D',
            'net_xl_amount_calibrated_D', 'main_force_conviction_index_D', 'wash_trade_intensity_D',
            'deception_index_D'
        ]

        if not self.helper._validate_required_signals(df, required_signals, method_name):
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。"] = ""
                self.helper._print_debug_output(debug_output)
            return pd.Series(0.0, index=df.index, dtype=np.float32)

        df_index = df.index

        # 获取所有原始信号
        (bias_21, pct_change, upward_purity_raw, upper_shadow_pressure_raw, active_buying_raw,
         open_price, high_price, close_price, low_price, main_force_net_flow,
         trend_vitality_index_raw, lower_shadow_absorption_strength_raw,
         net_sm_amount, net_md_amount, net_lg_amount, net_elg_amount,
         main_force_conviction_raw, wash_trade_intensity_raw, deception_index_raw) = self._get_raw_signals(df, method_name)

        _temp_debug_values["原始信号值"] = {
            "bias_21": bias_21, "pct_change": pct_change, "upward_purity_raw": upward_purity_raw,
            "upper_shadow_pressure_raw": upper_shadow_pressure_raw, "active_buying_raw": active_buying_raw,
            "open_price": open_price, "high_price": high_price, "close_price": close_price,
            "low_price": low_price, "main_force_net_flow": main_force_net_flow,
            "trend_vitality_index_raw": trend_vitality_index_raw,
            "lower_shadow_absorption_strength_raw": lower_shadow_absorption_strength_raw,
            "net_sm_amount": net_sm_amount, "net_md_amount": net_md_amount,
            "net_lg_amount": net_lg_amount, "net_elg_amount": net_elg_amount,
            "main_force_conviction_raw": main_force_conviction_raw,
            "wash_trade_intensity_raw": wash_trade_intensity_raw,
            "deception_index_raw": deception_index_raw
        }

        # 派生原子信号
        trend_form_score = self._derive_trend_form_score_from_raw(df_index, trend_vitality_index_raw, method_name)
        lower_shadow_strength = self._derive_lower_shadow_absorption_score_from_raw(df_index, lower_shadow_absorption_strength_raw, method_name)
        power_transfer = self._derive_power_transfer_score_from_raw(
            df_index, net_sm_amount, net_md_amount, net_lg_amount, net_elg_amount,
            main_force_conviction_raw, wash_trade_intensity_raw, deception_index_raw, method_name
        )

        # 归一化处理 (注意：main_force_net_flow_norm 不再直接用于门控，但可能用于其他地方或调试)
        (upward_purity_norm, upper_shadow_pressure_norm, active_buying_norm,
         power_transfer_norm, main_force_net_flow_norm) = self._normalize_signals(
            df_index, upward_purity_raw, upper_shadow_pressure_raw, active_buying_raw,
            power_transfer, main_force_net_flow, method_name
        )
        _temp_debug_values["归一化处理"] = {
            "upward_purity_norm": upward_purity_norm,
            "upper_shadow_pressure_norm": upper_shadow_pressure_norm,
            "active_buying_norm": active_buying_norm,
            "power_transfer_norm": power_transfer_norm,
            "main_force_net_flow_norm": main_force_net_flow_norm
        }

        # 市场上下文评估
        context_mask = self._evaluate_market_context(trend_form_score, bias_21, upward_purity_norm)
        _temp_debug_values["市场上下文"] = {"context_mask": context_mask}

        # K线形态门控
        is_upthrust_kline = self._identify_kline_pattern(open_price, high_price, close_price, low_price, pct_change)
        _temp_debug_values["K线形态门控"] = {"is_upthrust_kline": is_upthrust_kline}

        # 卖压审判分
        selling_pressure_score = self._assess_selling_pressure(upper_shadow_pressure_norm, pct_change)
        _temp_debug_values["卖压审判分"] = {"selling_pressure_score": selling_pressure_score}

        # 承接审判分
        absorption_rebuttal_score = self._assess_absorption_rebuttal(active_buying_norm, lower_shadow_strength, power_transfer_norm)
        _temp_debug_values["承接审判分"] = {"absorption_rebuttal_score": absorption_rebuttal_score}

        # 净洗盘意图
        net_washout_intent = (absorption_rebuttal_score - selling_pressure_score).clip(0, 1)
        _temp_debug_values["净洗盘意图"] = {"net_washout_intent": net_washout_intent}

        # 主力资金累积流向门控
        mf_cumulative_flow_gate = self._validate_main_force_inflow(df_index, main_force_net_flow, is_debug_enabled_for_method, probe_ts, debug_output)
        _temp_debug_values["主力资金累积流向门控"] = {"mf_cumulative_flow_gate": mf_cumulative_flow_gate}

        # 最终分数融合
        final_score = self._fuse_final_score(net_washout_intent, context_mask, is_upthrust_kline, mf_cumulative_flow_gate)
        _temp_debug_values["最终分数"] = {"final_score": final_score}

        if is_debug_enabled_for_method and probe_ts:
            self._print_debug_output_for_upthrust_washout(debug_output, probe_ts, _temp_debug_values, final_score)

        return final_score.astype(np.float32)

    def _get_raw_signals(self, df: pd.DataFrame, method_name: str) -> Tuple[pd.Series, ...]:
        """
        【V1.1.0 · 原始信号获取器】
        - 核心职责: 从DataFrame中安全地获取所有计算所需的原始信号。
        参数:
            df (pd.DataFrame): 包含所有原始数据的DataFrame。
            method_name (str): 调用此方法的名称，用于日志输出。
        返回:
            Tuple[pd.Series, ...]: 包含所有原始信号的元组。
        """
        bias_21 = self.helper._get_safe_series(df, 'BIAS_21_D', 0.0, method_name=method_name)
        pct_change = self.helper._get_safe_series(df, 'pct_change_D', 0.0, method_name=method_name)
        upward_purity_raw = self.helper._get_safe_series(df, 'upward_impulse_purity_D', 0.0, method_name=method_name)
        upper_shadow_pressure_raw = self.helper._get_safe_series(df, 'upper_shadow_selling_pressure_D', 0.0, method_name=method_name)
        active_buying_raw = self.helper._get_safe_series(df, 'active_buying_support_D', 0.0, method_name=method_name)
        open_price = self.helper._get_safe_series(df, 'open_D', 0.0, method_name=method_name)
        high_price = self.helper._get_safe_series(df, 'high_D', 0.0, method_name=method_name)
        close_price = self.helper._get_safe_series(df, 'close_D', 0.0, method_name=method_name)
        low_price = self.helper._get_safe_series(df, 'low_D', 0.0, method_name=method_name)
        main_force_net_flow = self.helper._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name=method_name)
        # 用于派生 SCORE_STRUCT_AXIOM_TREND_FORM
        trend_vitality_index_raw = self.helper._get_safe_series(df, 'trend_vitality_index_D', 0.0, method_name=method_name)
        # 用于派生 SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION
        lower_shadow_absorption_strength_raw = self.helper._get_safe_series(df, 'lower_shadow_absorption_strength_D', 0.0, method_name=method_name)
        # 用于派生 PROCESS_META_POWER_TRANSFER
        net_sm_amount = self.helper._get_safe_series(df, 'net_sh_amount_calibrated_D', 0.0, method_name=method_name)
        net_md_amount = self.helper._get_safe_series(df, 'net_md_amount_calibrated_D', 0.0, method_name=method_name)
        net_lg_amount = self.helper._get_safe_series(df, 'net_lg_amount_calibrated_D', 0.0, method_name=method_name)
        net_elg_amount = self.helper._get_safe_series(df, 'net_xl_amount_calibrated_D', 0.0, method_name=method_name)
        main_force_conviction_raw = self.helper._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name=method_name)
        wash_trade_intensity_raw = self.helper._get_safe_series(df, 'wash_trade_intensity_D', 0.0, method_name=method_name)
        deception_index_raw = self.helper._get_safe_series(df, 'deception_index_D', 0.0, method_name=method_name)

        return (bias_21, pct_change, upward_purity_raw, upper_shadow_pressure_raw, active_buying_raw,
                open_price, high_price, close_price, low_price, main_force_net_flow,
                trend_vitality_index_raw, lower_shadow_absorption_strength_raw,
                net_sm_amount, net_md_amount, net_lg_amount, net_elg_amount,
                main_force_conviction_raw, wash_trade_intensity_raw, deception_index_raw)

    def _derive_trend_form_score_from_raw(self, df_index: pd.Index, trend_vitality_index_raw: pd.Series, method_name: str) -> pd.Series:
        """
        【V1.0.0 · 趋势形态派生器】
        - 核心职责: 从原始的 `trend_vitality_index_D` 派生出趋势形态分数。
        参数:
            df_index (pd.Index): DataFrame的索引。
            trend_vitality_index_raw (pd.Series): 原始的趋势活力指数。
            method_name (str): 调用此方法的名称，用于日志输出。
        返回:
            pd.Series: 派生出的趋势形态分数 (双极性)。
        """
        # 趋势活力指数本身就是衡量趋势强度和方向的，直接归一化为双极性分数即可。
        # 假设 trend_vitality_index_D 越高代表趋势越强劲，越低代表趋势越弱或反向。
        trend_form_score = self.helper._normalize_series(trend_vitality_index_raw, df_index, bipolar=True)
        return trend_form_score

    def _derive_lower_shadow_absorption_score_from_raw(self, df_index: pd.Index, lower_shadow_absorption_strength_raw: pd.Series, method_name: str) -> pd.Series:
        """
        【V1.0.0 · 下影线吸收强度派生器】
        - 核心职责: 从原始的 `lower_shadow_absorption_strength_D` 派生出下影线吸收强度分数。
        参数:
            df_index (pd.Index): DataFrame的索引。
            lower_shadow_absorption_strength_raw (pd.Series): 原始的下影线吸收强度。
            method_name (str): 调用此方法的名称，用于日志输出。
        返回:
            pd.Series: 派生出的下影线吸收强度分数 (单极性)。
        """
        # 下影线吸收强度本身就是衡量吸收力量的，直接归一化为单极性分数即可。
        lower_shadow_absorption_score = self.helper._normalize_series(lower_shadow_absorption_strength_raw, df_index, bipolar=False)
        return lower_shadow_absorption_score

    def _derive_power_transfer_score_from_raw(self, df_index: pd.Index,
                                               net_sm_amount: pd.Series, net_md_amount: pd.Series,
                                               net_lg_amount: pd.Series, net_elg_amount: pd.Series,
                                               main_force_conviction: pd.Series, wash_trade_intensity: pd.Series,
                                               deception_index: pd.Series, method_name: str) -> pd.Series:
        """
        【V1.0.0 · 权力转移派生器】
        - 核心职责: 从原始资金流、主力信念、对倒和欺骗信号派生出权力转移分数。
        - 核心逻辑: 融合“主力信念”、“战场清晰度”（由对倒和欺骗构成）来计算资金转移的真实性，
                      并对最终结果进行非线性放大，以捕捉市场的极端博弈。
        参数:
            df_index (pd.Index): DataFrame的索引。
            net_sm_amount (pd.Series): 小单净额。
            net_md_amount (pd.Series): 中单净额。
            net_lg_amount (pd.Series): 大单净额。
            net_elg_amount (pd.Series): 特大单净额。
            main_force_conviction (pd.Series): 主力信念指数。
            wash_trade_intensity (pd.Series): 对倒强度。
            deception_index (pd.Series): 欺骗指数。
            method_name (str): 调用此方法的名称，用于日志输出。
        返回:
            pd.Series: 派生出的权力转移分数 (双极性)。
        """
        # 归一化辅助信号
        wash_trade_norm = self.helper._normalize_series(wash_trade_intensity, df_index, bipolar=False)
        deception_norm = self.helper._normalize_series(deception_index, df_index, bipolar=True)
        conviction_norm = self.helper._normalize_series(main_force_conviction, df_index, bipolar=True)

        # 计算战场清晰度因子
        clarity_from_noise = (1 - wash_trade_norm) * 0.4
        clarity_from_deception = (1 + deception_norm) / 2 * 0.6
        clarity_factor = (clarity_from_noise + clarity_from_deception).clip(0, 1)

        # 计算转移真实性因子
        transfer_authenticity_factor = (conviction_norm * clarity_factor).clip(-1, 1)

        # 计算有效主力资金流和有效散户资金流
        md_to_main_force = net_md_amount * transfer_authenticity_factor
        sm_to_main_force = net_sm_amount * transfer_authenticity_factor
        effective_main_force_flow = net_lg_amount + net_elg_amount + md_to_main_force + sm_to_main_force
        effective_retail_flow = (net_sm_amount - sm_to_main_force) + (net_md_amount - md_to_main_force)

        # 计算原始权力转移
        power_transfer_raw = effective_main_force_flow.diff(1) - effective_retail_flow.diff(1)

        # 归一化并进行非线性放大
        normalized_score = self.helper._normalize_series(power_transfer_raw.fillna(0), df_index, bipolar=True)
        final_score = np.sign(normalized_score) * normalized_score.abs().pow(1.2)
        final_score = final_score.clip(-1, 1)

        return final_score.astype(np.float32)

    def _normalize_signals(self, df_index: pd.Index, upward_purity_raw: pd.Series,
                           upper_shadow_pressure_raw: pd.Series, active_buying_raw: pd.Series,
                           power_transfer: pd.Series, main_force_net_flow: pd.Series, method_name: str) -> Tuple[pd.Series, ...]:
        """
        【V1.1.0 · 信号归一化器】
        - 核心职责: 对原始信号和派生信号进行归一化处理，使其值域统一。
        参数:
            df_index (pd.Index): DataFrame的索引。
            upward_purity_raw (pd.Series): 原始上涨纯度。
            upper_shadow_pressure_raw (pd.Series): 原始上影线卖压。
            active_buying_raw (pd.Series): 原始主动买盘。
            power_transfer (pd.Series): 派生出的权力转移分数。
            main_force_net_flow (pd.Series): 原始主力资金净流。
            method_name (str): 调用此方法的名称，用于日志输出。
        返回:
            Tuple[pd.Series, ...]: 包含所有归一化信号的元组。
        """
        upward_purity_norm = self.helper._normalize_series(upward_purity_raw, df_index, bipolar=False)
        upper_shadow_pressure_norm = self.helper._normalize_series(upper_shadow_pressure_raw, df_index, bipolar=False)
        active_buying_norm = self.helper._normalize_series(active_buying_raw, df_index, bipolar=False)
        # power_transfer 已经是双极性归一化后的分数，这里只需要clip到正向部分用于承接审判
        power_transfer_norm = power_transfer.clip(lower=0)
        main_force_net_flow_norm = self.helper._normalize_series(main_force_net_flow, df_index, bipolar=True)
        return (upward_purity_norm, upper_shadow_pressure_norm, active_buying_norm,
                power_transfer_norm, main_force_net_flow_norm)

    def _evaluate_market_context(self, trend_form_score: pd.Series, bias_21: pd.Series, upward_purity_norm: pd.Series) -> pd.Series:
        """
        【V1.2.0 · 市场情境评估器】
        - 核心职责: 评估当前市场是否处于适合洗盘反弹的上下文。
        - 核心升级: 移除对 `bias_21` 的严格限制，仅关注趋势向上和上涨纯度。
        参数:
            trend_form_score (pd.Series): 派生出的趋势形态分数。
            bias_21 (pd.Series): 21日乖离率 (不再作为硬性门槛)。
            upward_purity_norm (pd.Series): 归一化后的上涨纯度。
        返回:
            pd.Series: 市场情境掩码 (布尔Series)。
        """
        # 趋势向上，上涨纯度良好
        # trend_form_score 是双极性，大于0.2表示趋势向上且有一定强度
        # 移除 bias_21 < 0.2 的限制
        context_mask = (trend_form_score > 0.2) & (upward_purity_norm.rolling(3).mean() > 0.3)
        return context_mask

    def _identify_kline_pattern(self, open_price: pd.Series, high_price: pd.Series,
                                close_price: pd.Series, low_price: pd.Series, pct_change: pd.Series) -> pd.Series:
        """
        【V1.2.0 · K线形态识别器】
        - 核心职责: 识别“上冲回落”的K线形态。
        - 核心升级: 增加“收盘价低于高点3%以上”的条件，使其对上冲回落的定义更加灵活。
        参数:
            open_price (pd.Series): 开盘价。
            high_price (pd.Series): 最高价。
            close_price (pd.Series): 收盘价。
            low_price (pd.Series): 最低价。
            pct_change (pd.Series): 涨跌幅。
        返回:
            pd.Series: K线形态门控 (布尔Series)。
        """
        total_range = high_price - low_price
        total_range_safe = total_range.replace(0, 1e-9) # 避免除以零
        upper_shadow = high_price - np.maximum(open_price, close_price)
        upper_shadow_ratio = (upper_shadow / total_range_safe).fillna(0)

        # 条件1: 高开低走阴线 (开盘价高于收盘价，且当日下跌)
        is_high_open_low_close_yin = (open_price > close_price) & (pct_change < 0)

        # 条件2: 长上影线阴线 (上影线占比超过一定阈值，且收盘价低于开盘价)
        is_long_upper_shadow_yin = (upper_shadow_ratio > 0.4) & (close_price < open_price)

        # 新增条件3: 收盘价从高点回落超过3% (无论阴阳线)
        # 确保 high_price 不为0，避免除以零
        high_price_safe = high_price.replace(0, np.nan)
        drop_from_high_pct = ((high_price_safe - close_price) / high_price_safe).fillna(0)
        is_significant_drop_from_high = (drop_from_high_pct > 0.03)

        # 只要满足这三种形态之一，就认为是“上冲回落”的 K 线
        is_upthrust_kline = is_high_open_low_close_yin | is_long_upper_shadow_yin | is_significant_drop_from_high
        return is_upthrust_kline

    def _assess_selling_pressure(self, upper_shadow_pressure_norm: pd.Series, pct_change: pd.Series) -> pd.Series:
        """
        【V1.0.0 · 卖压审判器】
        - 核心职责: 评估当日的卖压强度。
        参数:
            upper_shadow_pressure_norm (pd.Series): 归一化后的上影线卖压。
            pct_change (pd.Series): 涨跌幅。
        返回:
            pd.Series: 卖压审判分数。
        """
        is_down_day = (pct_change < 0).astype(float)
        selling_pressure_score = (upper_shadow_pressure_norm * 0.7 + is_down_day * 0.3).clip(0, 1)
        return selling_pressure_score

    def _assess_absorption_rebuttal(self, active_buying_norm: pd.Series,
                                    lower_shadow_strength: pd.Series, power_transfer_norm: pd.Series) -> pd.Series:
        """
        【V1.1.0 · 承接审判器】
        - 核心职责: 评估主力承接力量，采用“强证优先”原则。
        参数:
            active_buying_norm (pd.Series): 归一化后的主动买盘。
            lower_shadow_strength (pd.Series): 派生出的下影线吸收强度。
            power_transfer_norm (pd.Series): 归一化后的权力转移（已裁剪为正向）。
        返回:
            pd.Series: 承接审判分数。
        """
        absorption_rebuttal_score = pd.concat([
            active_buying_norm,
            lower_shadow_strength,
            power_transfer_norm
        ], axis=1).max(axis=1)
        return absorption_rebuttal_score

    def _validate_main_force_inflow(self, df_index: pd.Index, main_force_net_flow: pd.Series,
                                     is_debug_enabled_for_method: bool, probe_ts: Optional[pd.Timestamp],
                                     debug_output: Dict) -> pd.Series:
        """
        【V1.1.0 · 主力资金累积流向门控】
        - 核心职责: 评估主力资金的累积净流向是否支持洗盘信号。
        - 核心升级: 使用 `_get_cumulative_context_score` 计算主力资金净流的累积分数，
                      并设定阈值（0.6）来判断是否通过门控。
        参数:
            df_index (pd.Index): DataFrame的索引。
            main_force_net_flow (pd.Series): 原始主力资金净流。
            is_debug_enabled_for_method (bool): 是否启用调试。
            probe_ts (Optional[pd.Timestamp]): 探针日期。
            debug_output (Dict): 调试输出字典。
        返回:
            pd.Series: 主力资金累积流向门控 (布尔Series)。
        """
        # 定义累积周期和权重
        cumulative_periods = [5, 13, 21]
        cumulative_weights = {"5": 0.5, "13": 0.3, "21": 0.2}
        # 计算主力资金净流的累积上下文分数
        mf_cumulative_score = self.helper._get_cumulative_context_score(
            series=main_force_net_flow,
            df_index=df_index,
            periods=cumulative_periods,
            weights=cumulative_weights,
            bipolar=True,
            signal_name="main_force_net_flow_calibrated_D",
            is_debug_enabled_for_method=is_debug_enabled_for_method,
            probe_ts=probe_ts,
            debug_output=debug_output
        )
        # 如果累积分数大于0.6，则认为通过门控
        mf_cumulative_flow_gate = (mf_cumulative_score > 0.6).fillna(False)
        if is_debug_enabled_for_method and probe_ts:
            val_mf_cumulative_score = mf_cumulative_score.loc[probe_ts] if probe_ts in mf_cumulative_score.index else np.nan
            debug_output[f"      -> 主力资金累积流向分数: {val_mf_cumulative_score:.4f}"] = ""
            debug_output[f"      -> 主力资金累积流向门控 (mf_cumulative_score > 0.6): {mf_cumulative_flow_gate.loc[probe_ts]}"] = ""
        return mf_cumulative_flow_gate

    def _fuse_final_score(self, net_washout_intent: pd.Series, context_mask: pd.Series,
                          is_upthrust_kline: pd.Series, mf_cumulative_flow_gate: pd.Series) -> pd.Series:
        """
        【V1.1.0 · 最终分数融合器】
        - 核心职责: 结合所有评估维度，计算最终的上冲回落洗盘信号分数。
        - 核心升级: 调整主力资金门控的判断方式，直接使用布尔型的 `mf_cumulative_flow_gate`。
        参数:
            net_washout_intent (pd.Series): 净洗盘意图。
            context_mask (pd.Series): 市场情境掩码。
            is_upthrust_kline (pd.Series): K线形态门控。
            mf_cumulative_flow_gate (pd.Series): 主力资金累积流向门控 (布尔Series)。
        返回:
            pd.Series: 最终的上冲回落洗盘信号分数。
        """
        # 结合市场上下文、K线形态门控和主力资金累积流向门控
        final_score = net_washout_intent.where(context_mask & is_upthrust_kline & mf_cumulative_flow_gate, 0.0).fillna(0.0)
        return final_score

    def _print_debug_output_for_upthrust_washout(self, debug_output: Dict, probe_ts: pd.Timestamp,
                                                  temp_debug_values: Dict, final_score: pd.Series) -> None:
        """
        【V1.1.0 · 调试信息输出器】
        - 核心职责: 统一输出上冲回落洗盘计算过程中的调试信息。
        - 核心升级: 调整主力资金门控的调试信息输出，以反映其新的累积流向评估逻辑。
        参数:
            debug_output (Dict): 调试信息字典。
            probe_ts (pd.Timestamp): 探针日期。
            temp_debug_values (Dict): 临时存储的中间计算结果。
            final_score (pd.Series): 最终分数。
        """
        method_name = "CalculateUpthrustWashoutRelationship.calculate"
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
        for key, series in temp_debug_values["原始信号值"].items():
            val = series.loc[probe_ts] if probe_ts in series.index else np.nan
            debug_output[f"        '{key}': {val:.4f}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 归一化处理 ---"] = ""
        for key, series in temp_debug_values["归一化处理"].items():
            val = series.loc[probe_ts] if probe_ts in series.index else np.nan
            debug_output[f"        {key}: {val:.4f}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 市场上下文 ---"] = ""
        for key, series in temp_debug_values["市场上下文"].items():
            val = series.loc[probe_ts] if probe_ts in series.index else np.nan
            debug_output[f"        {key}: {val}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- K线形态门控 ---"] = ""
        for key, series in temp_debug_values["K线形态门控"].items():
            val = series.loc[probe_ts] if probe_ts in series.index else np.nan
            debug_output[f"        {key}: {val}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 卖压审判分 ---"] = ""
        for key, series in temp_debug_values["卖压审判分"].items():
            val = series.loc[probe_ts] if probe_ts in series.index else np.nan
            debug_output[f"        {key}: {val:.4f}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 承接审判分 ---"] = ""
        for key, series in temp_debug_values["承接审判分"].items():
            val = series.loc[probe_ts] if probe_ts in series.index else np.nan
            debug_output[f"        {key}: {val:.4f}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 净洗盘意图 ---"] = ""
        for key, series in temp_debug_values["净洗盘意图"].items():
            val = series.loc[probe_ts] if probe_ts in series.index else np.nan
            debug_output[f"        {key}: {val:.4f}"] = ""
        # 调整主力资金门控的调试输出
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 主力资金累积流向门控 ---"] = ""
        for key, series in temp_debug_values["主力资金累积流向门控"].items():
            val = series.loc[probe_ts] if probe_ts in series.index else np.nan
            debug_output[f"        {key}: {val}"] = "" # mf_cumulative_flow_gate 现在是布尔值
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 最终分数 ---"] = ""
        for key, series in temp_debug_values["最终分数"].items():
            val = series.loc[probe_ts] if probe_ts in series.index else np.nan
            debug_output[f"        {key}: {val:.4f}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 上冲回落洗盘诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
        for key, value in debug_output.items():
            if value:
                print(f"{key}: {value}")
            else:
                print(key)
























