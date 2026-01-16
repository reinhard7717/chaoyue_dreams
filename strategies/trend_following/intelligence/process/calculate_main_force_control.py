# strategies\trend_following\intelligence\process\calculate_main_force_control.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

from strategies.trend_following.utils import (
    get_params_block, get_param_value, get_adaptive_mtf_normalized_score,
    get_adaptive_mtf_normalized_bipolar_score, normalize_score, _robust_geometric_mean
)
from strategies.trend_following.intelligence.process.helper import ProcessIntelligenceHelper

class CalculateMainForceControlRelationship:
    """
    【V1.0.0 · 主力控盘关系计算器】
    PROCESS_META_MAIN_FORCE_CONTROL
    - 核心职责: 计算“主力控盘”的专属关系分数。
    - 版本: 1.0.0
    """
    def __init__(self, strategy_instance, helper_instance: ProcessIntelligenceHelper):
        """
        初始化 CalculateMainForceControlRelationship 处理器。
        参数:
            strategy_instance: 策略实例，用于访问全局配置和原子状态。
            helper_instance (ProcessIntelligenceHelper): 过程情报辅助工具实例。
        """
        self.strategy = strategy_instance
        self.helper = helper_instance
        self.params = self.helper.params # 获取ProcessIntelligence的参数
        self.debug_params = self.helper.debug_params
        self.probe_dates = self.helper.probe_dates

    def _print_debug_info(self, debug_output: Dict):
        """
        统一打印调试信息。
        """
        for key, value in debug_output.items():
            if value:
                print(f"{key}: {value}")
            else:
                print(key)

    def _get_safe_series(self, df: pd.DataFrame, col_name: str, default_value: float = 0.0, method_name: str = "") -> pd.Series:
        """
        安全地从DataFrame中获取Series，如果列不存在或值为NaN，则填充默认值。
        V11.0: 针对特定信号，当其值为NaN时，提供更具业务含义的默认值（例如0.5表示中性），
               以适配normalize_score和normalize_to_bipolar的归一化逻辑。
               neutral_nan_defaults 字典已提取到配置中。
        """
        process_params = get_params_block(self.strategy, 'process_intelligence_params', {})
        neutral_nan_defaults = process_params.get('neutral_nan_defaults', {})
        current_default_value = neutral_nan_defaults.get(col_name, default_value)
        if col_name not in df.columns:
            return pd.Series(current_default_value, index=df.index, dtype=np.float32)
        series = df[col_name].astype(np.float32)
        return series.fillna(current_default_value)

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.4.1 · 控盘杠杆与全息资金流验证强化版 - 拆分版】计算“主力控盘”的专属关系分数。
        - 核心重构: 创立“控盘即杠杆”模型。将“控盘度”作为调节“资金流向”影响力的核心杠杆。
                      最终分 = 主力净流入分 * (1 + 融合控盘分)。
        - 证据升级: 融合传统的均线控盘度与更现代的“控盘稳固度”，形成更立体的控盘评分。
        - 【强化】引入主力资金净流向和资金流可信度作为控盘杠杆的调节因子，确保控盘的积极性。
        - 【重要修改】修正 `control_leverage` 逻辑，当控盘不强时，对资金流入进行更强的惩罚。
        - 【新增】引入 MTF 控盘信号和 MTF 主力资金净流，增强信号鲁棒性。
        参数:
            df (pd.DataFrame): 包含所有原始数据的DataFrame。
            config (Dict): 诊断配置字典。
        返回:
            pd.Series: 主力控盘关系分数。
        """
        method_name = "calculate_main_force_control_relationship"
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
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算主力控盘关系..."] = ""
        # 1. 获取参数和权重
        actual_mtf_weights, mtf_slope_accel_weights = self._get_control_parameters(config)
        # 2. 验证所需信号
        required_signals = [
            'close_D', 'main_force_net_flow_calibrated_D', 'control_solidity_index_D', 'flow_credibility_index_D',
            'main_force_daily_buy_amount_D', 'main_force_daily_sell_amount_D',
            'main_force_daily_buy_volume_D', 'main_force_daily_sell_volume_D'
        ]
        # 动态添加MTF斜率和加速度信号到required_signals
        for base_sig in ['control_solidity_index_D']: # 仅控盘稳固度需要MTF斜率/加速度
            for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
                required_signals.append(f'SLOPE_{period_str}_{base_sig}')
            for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
                required_signals.append(f'ACCEL_{period_str}_{base_sig}')
        # 复合主力净活动信号的MTF斜率/加速度将由 _calculate_main_force_net_activity_score 内部处理
        if not self.helper._validate_required_signals(df, required_signals, method_name):
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。"] = ""
                self._print_debug_info(debug_output)
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        # 3. 获取原始信号
        close_price, control_solidity_raw, main_force_net_flow_calibrated, flow_credibility_raw, mf_buy_amount, mf_sell_amount, mf_buy_volume, mf_sell_volume = self._get_raw_control_signals(df, method_name, _temp_debug_values)
        # 4. 计算传统控盘度
        kongpan_raw = self._calculate_traditional_control_score_components(df, close_price, method_name, _temp_debug_values, is_debug_enabled_for_method, probe_ts, debug_output)
        if kongpan_raw.isnull().all():
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"    -> [过程情报警告] {method_name} 传统控盘度计算失败，返回默认值。"] = ""
                self._print_debug_info(debug_output)
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # 5. 归一化和MTF融合控盘组件 (不包含主力资金流MTF)
        traditional_control_score, mtf_structural_control_score, flow_credibility_norm = self._normalize_and_mtf_control_components(df, df_index, kongpan_raw, control_solidity_raw, flow_credibility_raw, mtf_slope_accel_weights, method_name, _temp_debug_values)
        # 6. 计算主力净活动分数 (MTF融合版)
        mtf_main_force_net_activity_score = self._calculate_main_force_net_activity_score(df, df_index, mf_buy_amount, mf_sell_amount, mf_buy_volume, mf_sell_volume, mtf_slope_accel_weights, method_name, _temp_debug_values)
        # 7. 融合控盘分
        fused_control_score = self._fuse_control_scores(traditional_control_score, mtf_structural_control_score, _temp_debug_values)
        # 8. 计算控盘杠杆模型 (使用新的主力净活动分数)
        control_leverage = self._calculate_control_leverage_model(df_index, fused_control_score, mtf_main_force_net_activity_score, flow_credibility_norm, _temp_debug_values)
        # 9. 最终控盘分数
        final_control_score = (mtf_main_force_net_activity_score * control_leverage).clip(-1, 1)
        _temp_debug_values["最终控盘分数"] = {
            "final_control_score": final_control_score
        }
        if is_debug_enabled_for_method and probe_ts:
            self._calculate_main_force_control_relationship_debug_output(debug_output, _temp_debug_values, method_name, probe_ts, final_control_score)
        return final_control_score.astype(np.float32)

    def _get_control_parameters(self, config: Dict) -> Tuple[Dict, Dict]:
        """
        【V1.0.0】获取主力控盘关系计算所需的参数和权重。
        参数:
            config (Dict): 诊断配置字典。
        返回:
            Tuple[Dict, Dict]: 包含实际MTF权重和MTF斜率加速度权重的元组。
        """
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        mtf_slope_accel_weights = config.get('mtf_slope_accel_weights', {"slope_periods": {"5": 0.6, "13": 0.4}, "accel_periods": {"5": 0.7, "13": 0.3}})
        return actual_mtf_weights, mtf_slope_accel_weights

    def _get_raw_control_signals(self, df: pd.DataFrame, method_name: str, _temp_debug_values: Dict) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        【V1.0.1】获取主力控盘关系计算所需的原始信号。
        新增获取主力日度买卖金额和股数。
        参数:
            df (pd.DataFrame): 包含所有原始数据的DataFrame。
            method_name (str): 调用此方法的名称。
            _temp_debug_values (Dict): 临时存储中间计算结果的字典。
        返回:
            Tuple[pd.Series, ...]: 包含收盘价、控盘稳固度、校准主力净流量、资金流可信度、
                                   主力日度买入金额、主力日度卖出金额、主力日度买入股数、主力日度卖出股数的元组。
        """
        close_price = self._get_safe_series(df, 'close_D', method_name=method_name)
        control_solidity_raw = self._get_safe_series(df, 'control_solidity_index_D', 0.0, method_name=method_name)
        # main_force_net_flow_calibrated_D 仍获取，但不再用于核心计算，仅作调试参考
        main_force_net_flow_calibrated = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name=method_name)
        flow_credibility_raw = self._get_safe_series(df, 'flow_credibility_index_D', 0.0, method_name=method_name)
        # 新增的四个主力日度买卖指标
        mf_buy_amount = self._get_safe_series(df, 'main_force_daily_buy_amount_D', 0.0, method_name=method_name)
        mf_sell_amount = self._get_safe_series(df, 'main_force_daily_sell_amount_D', 0.0, method_name=method_name)
        mf_buy_volume = self._get_safe_series(df, 'main_force_daily_buy_volume_D', 0.0, method_name=method_name)
        mf_sell_volume = self._get_safe_series(df, 'main_force_daily_sell_volume_D', 0.0, method_name=method_name)
        _temp_debug_values["原始信号值"] = {
            "close_D": close_price,
            "control_solidity_index_D": control_solidity_raw,
            "main_force_net_flow_calibrated_D": main_force_net_flow_calibrated,
            "flow_credibility_index_D": flow_credibility_raw,
            "main_force_daily_buy_amount_D": mf_buy_amount,
            "main_force_daily_sell_amount_D": mf_sell_amount,
            "main_force_daily_buy_volume_D": mf_buy_volume,
            "main_force_daily_sell_volume_D": mf_sell_volume,
        }
        return close_price, control_solidity_raw, main_force_net_flow_calibrated, flow_credibility_raw, mf_buy_amount, mf_sell_amount, mf_buy_volume, mf_sell_volume

    def _calculate_main_force_control_relationship_debug_output(self, debug_output: Dict, _temp_debug_values: Dict, method_name: str, probe_ts: pd.Timestamp, final_control_score: pd.Series):
        """
        【V1.0.1】主力控盘关系计算的调试信息输出方法。
        更新调试输出，以反映 `_temp_debug_values` 中新增的“主力净活动计算”部分。
        参数:
            debug_output (Dict): 调试信息字典。
            _temp_debug_values (Dict): 临时存储的中间计算结果。
            method_name (str): 调用此方法的名称。
            probe_ts (pd.Timestamp): 探针日期。
            final_control_score (pd.Series): 最终计算出的主力控盘分数。
        """
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
        for sig_name, series in _temp_debug_values["原始信号值"].items():
            val = series.loc[probe_ts] if probe_ts in series.index else np.nan
            debug_output[f"        '{sig_name}': {val:.4f}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 传统控盘度计算 ---"] = ""
        for key, series in _temp_debug_values["传统控盘度计算"].items():
            val = series.loc[probe_ts] if probe_ts in series.index else np.nan
            debug_output[f"        {key}: {val:.4f}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 归一化处理 ---"] = ""
        for key, series in _temp_debug_values["归一化处理"].items():
            val = series.loc[probe_ts] if probe_ts in series.index else np.nan
            debug_output[f"        {key}: {val:.4f}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 主力净活动计算 ---"] = ""
        for key, series in _temp_debug_values["主力净活动计算"].items():
            val = series.loc[probe_ts] if probe_ts in series.index else np.nan
            debug_output[f"        {key}: {val:.4f}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 融合控盘分 ---"] = ""
        for key, series in _temp_debug_values["融合控盘分"].items():
            val = series.loc[probe_ts] if probe_ts in series.index else np.nan
            debug_output[f"        {key}: {val:.4f}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 控盘杠杆模型 ---"] = ""
        for key, series in _temp_debug_values["控盘杠杆模型"].items():
            val = series.loc[probe_ts] if probe_ts in series.index else np.nan
            debug_output[f"        {key}: {val:.4f}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 最终控盘分数 ---"] = ""
        for key, series in _temp_debug_values["最终控盘分数"].items():
            val = series.loc[probe_ts] if probe_ts in series.index else np.nan
            debug_output[f"        {key}: {val:.4f}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 主力控盘关系诊断完成，最终分值: {final_control_score.loc[probe_ts]:.4f}"] = ""
        self._print_debug_info(debug_output)

    def _calculate_main_force_net_activity_score(self, df: pd.DataFrame, df_index: pd.Index, mf_buy_amount: pd.Series, mf_sell_amount: pd.Series, mf_buy_volume: pd.Series, mf_sell_volume: pd.Series, mtf_slope_accel_weights: Dict, method_name: str, _temp_debug_values: Dict) -> pd.Series:
        """
        【V1.0.0】计算主力净活动分数（MTF融合版）。
        该方法将主力日度买卖金额和股数融合成一个双极性MTF分数。
        参数:
            df (pd.DataFrame): 包含所有原始数据的DataFrame。
            df_index (pd.Index): DataFrame的索引。
            mf_buy_amount (pd.Series): 主力日度买入金额。
            mf_sell_amount (pd.Series): 主力日度卖出金额。
            mf_buy_volume (pd.Series): 主力日度买入股数。
            mf_sell_volume (pd.Series): 主力日度卖出股数。
            mtf_slope_accel_weights (Dict): MTF斜率加速度权重配置。
            method_name (str): 调用此方法的名称。
            _temp_debug_values (Dict): 临时存储中间计算结果的字典。
        返回:
            pd.Series: 融合后的MTF主力净活动分数 (范围 [-1, 1])。
        """
        net_amount_raw = mf_buy_amount - mf_sell_amount
        net_volume_raw = mf_buy_volume - mf_sell_volume
        # 对净金额和净股数进行双极性归一化
        norm_net_amount = self.helper._normalize_series(net_amount_raw, df_index, bipolar=True)
        norm_net_volume = self.helper._normalize_series(net_volume_raw, df_index, bipolar=True)
        # 融合归一化后的净金额和净股数，形成一个复合净活动信号
        # 赋予金额和股数同等权重，或根据需要调整
        composite_net_activity_series = (norm_net_amount * 0.5 + norm_net_volume * 0.5).clip(-1, 1)
        # 对复合净活动信号进行MTF斜率和加速度融合
        mtf_main_force_net_activity_score = self.helper._get_mtf_score_from_series_slope_accel(
            composite_net_activity_series,
            mtf_slope_accel_weights,
            df_index,
            method_name,
            bipolar=True
        )
        _temp_debug_values["主力净活动计算"] = {
            "net_amount_raw": net_amount_raw,
            "net_volume_raw": net_volume_raw,
            "norm_net_amount": norm_net_amount,
            "norm_net_volume": norm_net_volume,
            "composite_net_activity_series": composite_net_activity_series,
            "mtf_main_force_net_activity_score": mtf_main_force_net_activity_score
        }
        return mtf_main_force_net_activity_score

    def _calculate_traditional_control_score_components(self, df: pd.DataFrame, close_price: pd.Series, method_name: str, _temp_debug_values: Dict, is_debug_enabled_for_method: bool, probe_ts: pd.Timestamp, debug_output: Dict) -> pd.Series:
        """
        【V1.0.0】计算基于EMA的传统控盘度分数。
        参数:
            df (pd.DataFrame): 包含所有原始数据的DataFrame。
            close_price (pd.Series): 收盘价序列。
            method_name (str): 调用此方法的名称。
            _temp_debug_values (Dict): 临时存储中间计算结果的字典。
            is_debug_enabled_for_method (bool): 是否启用调试。
            probe_ts (pd.Timestamp): 探针日期。
            debug_output (Dict): 调试输出字典。
        返回:
            pd.Series: 传统控盘度分数。
        """
        ema13 = ta.ema(close=close_price, length=13, append=False)
        if ema13 is None:
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"    -> [过程情报警告] {method_name} EMA_13 计算失败，返回默认值。"] = ""
                self._print_debug_info(debug_output)
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        varn1 = ta.ema(close=ema13, length=13, append=False)
        if varn1 is None:
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"    -> [过程情报警告] {method_name} VARN1 计算失败，返回默认值。"] = ""
                self._print_debug_info(debug_output)
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        prev_varn1 = varn1.shift(1).replace(0, np.nan)
        kongpan_raw = (varn1 - prev_varn1) / prev_varn1 * 1000
        _temp_debug_values["传统控盘度计算"] = {
            "ema13": ema13,
            "varn1": varn1,
            "prev_varn1": prev_varn1,
            "kongpan_raw": kongpan_raw
        }
        return kongpan_raw

    def _normalize_and_mtf_control_components(self, df: pd.DataFrame, df_index: pd.Index, kongpan_raw: pd.Series, control_solidity_raw: pd.Series, flow_credibility_raw: pd.Series, mtf_slope_accel_weights: Dict, method_name: str, _temp_debug_values: Dict) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        【V1.0.1】对控盘相关信号进行归一化和MTF融合处理。
        移除了mtf_main_force_flow_score的计算和返回。
        参数:
            df (pd.DataFrame): 包含所有原始数据的DataFrame。
            df_index (pd.Index): DataFrame的索引。
            kongpan_raw (pd.Series): 原始传统控盘度。
            control_solidity_raw (pd.Series): 原始控盘稳固度。
            flow_credibility_raw (pd.Series): 原始资金流可信度。
            mtf_slope_accel_weights (Dict): MTF斜率加速度权重配置。
            method_name (str): 调用此方法的名称。
            _temp_debug_values (Dict): 临时存储中间计算结果的字典。
        返回:
            Tuple[pd.Series, pd.Series, pd.Series]: 包含归一化后的传统控盘分、MTF结构控盘分、归一化资金流可信度的元组。
        """
        traditional_control_score = self.helper._normalize_series(kongpan_raw, df_index, bipolar=True)
        mtf_structural_control_score = self.helper._get_mtf_slope_accel_score(df, 'control_solidity_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        flow_credibility_norm = self.helper._normalize_series(flow_credibility_raw, df_index, bipolar=False)
        _temp_debug_values["归一化处理"] = {
            "traditional_control_score": traditional_control_score,
            "mtf_structural_control_score": mtf_structural_control_score,
            "flow_credibility_norm": flow_credibility_norm
        }
        return traditional_control_score, mtf_structural_control_score, flow_credibility_norm

    def _fuse_control_scores(self, traditional_control_score: pd.Series, mtf_structural_control_score: pd.Series, _temp_debug_values: Dict) -> pd.Series:
        """
        【V1.0.0】融合传统控盘分和MTF结构控盘分。
        参数:
            traditional_control_score (pd.Series): 传统控盘度分数。
            mtf_structural_control_score (pd.Series): MTF结构控盘度分数。
            _temp_debug_values (Dict): 临时存储中间计算结果的字典。
        返回:
            pd.Series: 融合后的控盘分数。
        """
        fused_control_score = (traditional_control_score * 0.4 + mtf_structural_control_score * 0.6).clip(-1, 1)
        _temp_debug_values["融合控盘分"] = {
            "fused_control_score": fused_control_score
        }
        return fused_control_score

    def _calculate_control_leverage_model(self, df_index: pd.Index, fused_control_score: pd.Series, mtf_main_force_net_activity_score: pd.Series, flow_credibility_norm: pd.Series, _temp_debug_values: Dict) -> pd.Series:
        """
        【V1.0.1】计算控盘杠杆模型。
        将参数 `mtf_main_force_flow_score` 替换为 `mtf_main_force_net_activity_score`。
        参数:
            df_index (pd.Index): DataFrame的索引。
            fused_control_score (pd.Series): 融合后的控盘分数。
            mtf_main_force_net_activity_score (pd.Series): MTF主力净活动分数。
            flow_credibility_norm (pd.Series): 归一化资金流可信度。
            _temp_debug_values (Dict): 临时存储中间计算结果的字典。
        返回:
            pd.Series: 控盘杠杆。
        """
        mf_inflow_validation = mtf_main_force_net_activity_score.clip(lower=0) * flow_credibility_norm
        control_leverage = pd.Series(1.0, index=df_index, dtype=np.float32)
        control_leverage = control_leverage.mask(fused_control_score > 0, 1 + fused_control_score * mf_inflow_validation)
        control_leverage = control_leverage.mask(fused_control_score <= 0, (1 + fused_control_score) * (1 - mf_inflow_validation * 0.5))
        control_leverage = control_leverage.clip(0, 2)
        _temp_debug_values["控盘杠杆模型"] = {
            "mf_inflow_validation": mf_inflow_validation,
            "control_leverage": control_leverage
        }
        return control_leverage


