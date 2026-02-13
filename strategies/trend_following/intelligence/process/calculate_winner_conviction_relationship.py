# strategies\trend_following\intelligence\process\calculate_winner_conviction_relationship.py
# 【V5.0 · 全息筹码信念与资金共识版】“赢家信念”专属关系计算引擎
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

class CalculateWinnerConvictionRelationship:
    """
    【V5.0 · 全息筹码信念与资金共识版】“赢家信念”专属关系计算引擎
    PROCESS_META_WINNER_CONVICTION
    - 核心重构: 废弃旧的微观订单流依赖，基于《最终军械库》重构为“信念坚固度 × 压力消化力 × 资金共识度”三元模型。
    - 核心升级:
        1. 信念坚固度: 融合获利盘比例(广度)、平均获利幅度(深度)与筹码稳定性(持久度)。
        2. 压力消化力: 动态评估获利兑现压力与套牢抛压，并结合VPA效率判断承接能力。
        3. 资金共识度: 引入主力信念指数与净额流向，验证信念的真实性。
        4. 全息调试: 暴露全链路计算节点，支持深度诊断。
    """
    def __init__(self, strategy_instance, helper: ProcessIntelligenceHelper):
        self.strategy = strategy_instance
        self.helper = helper
        self.params = self.helper.params
        self.debug_params = self.helper.debug_params
        self.probe_dates = self.helper.probe_dates

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        method_name = "calculate_winner_conviction_relationship"
        df_index = df.index
        is_debug_enabled_for_method, probe_ts, debug_output, _temp_debug_values = self._setup_debug_context(df, method_name)
        all_params = self._get_all_params(config)
        signals_data = self._get_and_validate_signals(df, df_index, method_name, all_params, _temp_debug_values)
        
        if signals_data is None:
            return pd.Series(0.0, index=df_index, dtype=np.float32)

        normalized_signals = self._normalize_raw_data(df_index, signals_data, _temp_debug_values)

        # 1. 信念坚固度 (原有)
        belief_solidity_score = self._calculate_belief_solidity(df_index, signals_data, normalized_signals, all_params, _temp_debug_values)
        # 2. 压力消化力 (原有)
        pressure_digestion_score = self._calculate_pressure_digestion(df_index, signals_data, normalized_signals, all_params, _temp_debug_values)
        # 3. 资金共识度 (原有)
        flow_consensus_score = self._calculate_flow_consensus(df_index, signals_data, normalized_signals, all_params, _temp_debug_values)
        
        # 4. 【新增】对手盘投降度 (Adversary Capitulation)
        adversary_capitulation_score = self._calculate_adversary_capitulation(df_index, signals_data, normalized_signals, all_params, _temp_debug_values)
        
        # 5. 【新增】微观隐蔽度 (Micro-Stealth)
        micro_stealth_score = self._calculate_micro_stealth(df_index, signals_data, normalized_signals, all_params, _temp_debug_values)

        # 情境调制
        context_modulator = self._calculate_contextual_modulator(df_index, signals_data, normalized_signals, all_params, _temp_debug_values)

        # 最终五维融合
        final_score = self._perform_final_fusion(
            df_index, 
            belief_solidity_score, 
            pressure_digestion_score, 
            flow_consensus_score,
            adversary_capitulation_score,
            micro_stealth_score,
            context_modulator, 
            all_params, 
            _temp_debug_values
        )
        
        self._print_debug_info(method_name, final_score, is_debug_enabled_for_method, probe_ts, debug_output, _temp_debug_values)
        return final_score.astype(np.float32)

    def _setup_debug_context(self, df: pd.DataFrame, method_name: str) -> Tuple[bool, Optional[pd.Timestamp], Dict, Dict]:
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
            debug_output[f"--- {method_name} 全息诊断 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
        return is_debug_enabled_for_method, probe_ts, debug_output, _temp_debug_values

    def _print_debug_info(self, method_name: str, final_score: pd.Series, is_debug_enabled_for_method: bool, probe_ts: Optional[pd.Timestamp], debug_output: Dict, _temp_debug_values: Dict):
        if is_debug_enabled_for_method and probe_ts:
            sections = ["原始信号值", "MTF信号值", "归一化处理", "信念坚固度", "压力消化力", "资金共识度", "情境调制", "最终融合"]
            for section in sections:
                if section in _temp_debug_values:
                    debug_output[f"  -- [{section}] 详情:"] = ""
                    for key, item in _temp_debug_values[section].items():
                        if isinstance(item, pd.Series):
                            val = item.loc[probe_ts] if probe_ts in item.index else np.nan
                            debug_output[f"      {key}: {val:.4f}"] = ""
                        elif isinstance(item, dict):
                             debug_output[f"      {key}: (Dict content omitted)"] = ""
                        else:
                            debug_output[f"      {key}: {item:.4f}"] = ""
            debug_output[f"  -- [最终结果] {method_name}: {final_score.loc[probe_ts]:.4f}"] = ""
            self.helper._print_debug_output(debug_output)

    def _get_all_params(self, config: Dict) -> Dict[str, Any]:
        """
        【V17.0 · 非线性协同增益版】更新stealth_weights，新增 synergy_gain_factor (协同增益因子)。
        """
        params = get_param_value(config.get('winner_conviction_params'), {})
        return {
            "mtf_slope_accel_weights": get_param_value(params.get('mtf_slope_accel_weights'), {"slope_periods": {"5": 0.4, "13": 0.3, "21": 0.2, "34": 0.1}, "accel_periods": {"5": 0.6, "13": 0.4}}),
            "belief_weights": get_param_value(params.get('belief_weights'), {"winner_rate": 0.3, "profit_margin": 0.3, "chip_stability": 0.4}),
            "pressure_weights": get_param_value(params.get('pressure_weights'), {"profit_pressure": 0.35, "trapped_pressure": 0.35, "vpa_efficiency": 0.3}),
            "flow_weights": get_param_value(params.get('flow_weights'), {"mf_conviction": 0.4, "net_mf_amount": 0.3, "flow_consistency": 0.3}),
            "capitulation_weights": get_param_value(params.get('capitulation_weights'), {"static_pain": 0.25, "kinetic_pain": 0.25, "pain_saturation": 0.25, "liquidity_release": 0.25, "panic_cascade": 0.3, "gamma_base": 2.0, "gamma_min": 0.6}),
            "stealth_weights": get_param_value(params.get('stealth_weights'), {
                "stealth_flow": 0.15,
                "intraday_accum": 0.15,
                "hab_stealth_21d": 0.15,
                "kinetic_stealth_13d": 0.1,
                "chip_undercurrent": 0.15,
                "order_anomaly": 0.1,
                "wash_trade_mask": 0.05,
                "afternoon_mask": 0.05,
                "closing_mask": 0.1,
                "synergy_gain_factor": 3.0,
                "volatility_suppression_gamma": 1.6
            }),
            "context_weights": get_param_value(params.get('context_weights'), {"market_sentiment": 0.25, "trend_confirmation": 0.25, "volatility_stability": 0.15, "structural_order": 0.15, "reversion_penalty": 0.2}),
            "environment_weights": get_param_value(params.get('environment_weights'), {"theme_thermal": 0.6, "game_friction": 0.4}),
            "hab_weights": get_param_value(params.get('hab_weights'), {"flow_inertia": 0.4, "sentiment_memory": 0.3, "volatility_memory": 0.3}),
            "kinematics_weights": get_param_value(params.get('kinematics_weights'), {"slope": 0.5, "accel": 0.3, "jerk": 0.2}),
            "resonance_params": get_param_value(params.get('resonance_params'), {"coherence_sensitivity": 1.0, "base_exponent": 2.0, "min_exponent": 0.8}),
            "final_fusion_weights": get_param_value(params.get('final_fusion_weights'), {"belief": 0.25, "pressure": 0.2, "flow": 0.25, "capitulation": 0.15, "stealth": 0.15}),
            "final_exponent": get_param_value(params.get('final_exponent'), 1.8)
        }

    def _get_and_validate_signals(self, df: pd.DataFrame, df_index: pd.Index, method_name: str, params: Dict, _temp_debug_values: Dict) -> Optional[Dict[str, pd.Series]]:
        """
        【V16.0 · 信号加载重构】针对筹码与午后掩护，从数据层直接加载相关特征的原始数据。
        """
        raw_signals_map = {
            "winner_rate_raw": "winner_rate_D",
            "winner_profit_margin_avg_raw": "winner_profit_margin_avg_D",
            "winner_stability_index_raw": "winner_stability_index_D",
            "chip_stability_raw": "chip_stability_D",
            "profit_pressure_raw": "profit_pressure_D",
            "pressure_trapped_raw": "pressure_trapped_D",
            "vpa_efficiency_raw": "VPA_EFFICIENCY_D",
            "main_force_conviction_index_raw": "main_force_conviction_index_D",
            "net_mf_amount_raw": "net_mf_amount_D",
            "flow_consistency_raw": "flow_consistency_D",
            "market_sentiment_score_raw": "market_sentiment_score_D",
            "trend_confirmation_score_raw": "trend_confirmation_score_D",
            "loser_loss_margin_avg_raw": "loser_loss_margin_avg_D",
            "panic_selling_cascade_raw": "panic_selling_cascade_D",
            "stealth_flow_ratio_raw": "stealth_flow_ratio_D",
            "intraday_accumulation_confidence_raw": "intraday_accumulation_confidence_D",
            "peak_concentration_raw": "peak_concentration_D",
            "chip_concentration_ratio_raw": "chip_concentration_ratio_D",
            "cost_50pct_raw": "cost_50pct_D",
            "close_raw": "close_D",
            "long_term_chip_ratio_raw": "long_term_chip_ratio_D",
            "intraday_high_lock_ratio_raw": "intraday_high_lock_ratio_D",
            "uptrend_strength_raw": "uptrend_strength_D",
            "panic_buy_absorption_raw": "panic_buy_absorption_contribution_D",
            "support_strength_raw": "support_strength_D",
            "chip_transfer_eff_raw": "tick_chip_transfer_efficiency_D",
            "pressure_release_raw": "pressure_release_index_D",
            "distribution_conf_raw": "intraday_distribution_confidence_D",
            "gap_defense_raw": "opening_gap_defense_strength_D",
            "tick_balance_raw": "tick_chip_balance_ratio_D",
            "smart_money_net_raw": "SMART_MONEY_HM_NET_BUY_D",
            "smart_money_attack_raw": "SMART_MONEY_HM_COORDINATED_ATTACK_D",
            "smart_money_div_raw": "SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D",
            "tick_large_net_raw": "tick_large_order_net_D",
            "wash_trade_raw": "wash_trade_intensity_D",
            "closing_flow_raw": "closing_flow_ratio_D",
            "afternoon_flow_raw": "afternoon_flow_ratio_D",
            "order_anomaly_raw": "large_order_anomaly_D",
            "volatility_instability_raw": "VOLATILITY_INSTABILITY_INDEX_21d_D",
            "structural_entropy_raw": "structural_entropy_change_D",
            "mean_reversion_freq_raw": "mean_reversion_frequency_D",
            "turnover_rate_raw": "turnover_rate_f_D",
            "theme_hotness_raw": "theme_hotness_score_D",
            "game_intensity_raw": "game_intensity_D",
            "abnormal_vol_raw": "tick_abnormal_volume_ratio_D",
            "chip_flow_intensity_raw": "chip_flow_intensity_D"
        }
        if not self.helper._validate_required_signals(df, list(raw_signals_map.values()), method_name):
            return None
        signals_data = {}
        _temp_debug_values["原始信号值"] = {}
        _temp_debug_values["动力学信号值"] = {}
        kinetic_targets = [
            "smart_money_net_raw", "tick_large_net_raw", "net_mf_amount_raw", 
            "wash_trade_raw", "volatility_instability_raw", "market_sentiment_score_raw",
            "structural_entropy_raw", "theme_hotness_raw", "game_intensity_raw",
            "loser_loss_margin_avg_raw", "pressure_trapped_raw",
            "turnover_rate_raw", "abnormal_vol_raw",
            "stealth_flow_ratio_raw", "intraday_accumulation_confidence_raw",
            "chip_flow_intensity_raw"
        ]
        kinetic_window = 13
        for key, col_name in raw_signals_map.items():
            series = self.helper._get_safe_series(df, col_name, np.nan, method_name=method_name)
            signals_data[key] = series
            _temp_debug_values["原始信号值"][key] = series
            if key in kinetic_targets:
                slope_col = f"SLOPE_{kinetic_window}_{col_name}"
                if slope_col in df.columns:
                    signals_data[f"slope_{key}"] = df[slope_col]
                else:
                    signals_data[f"slope_{key}"] = ta.slope(series, length=kinetic_window)
                accel_col = f"ACCEL_{kinetic_window}_{col_name}"
                if accel_col in df.columns:
                     signals_data[f"accel_{key}"] = df[accel_col]
                else:
                     signals_data[f"accel_{key}"] = signals_data[f"slope_{key}"].diff(1)
                jerk_col = f"JERK_{kinetic_window}_{col_name}"
                if jerk_col in df.columns:
                    signals_data[f"jerk_{key}"] = df[jerk_col]
                else:
                    signals_data[f"jerk_{key}"] = signals_data[f"accel_{key}"].diff(1)
        mtf_slope_accel_weights = params.get("mtf_slope_accel_weights", {})
        for key, col_name in raw_signals_map.items():
            mtf_key = f"mtf_{key.replace('_raw', '')}"
            ascending = True
            if key in ["profit_pressure_raw", "pressure_trapped_raw", "distribution_conf_raw", "smart_money_div_raw", "wash_trade_raw", "order_anomaly_raw", "volatility_instability_raw", "structural_entropy_raw", "mean_reversion_freq_raw", "game_intensity_raw"]:
                ascending = False
            elif key in ["loser_loss_margin_avg_raw", "theme_hotness_raw", "turnover_rate_raw", "abnormal_vol_raw", "chip_flow_intensity_raw"]:
                ascending = True
            bipolar = True if key in ["vpa_efficiency_raw", "market_sentiment_score_raw", "net_mf_amount_raw", "smart_money_net_raw", "tick_large_net_raw"] else False
            signals_data[mtf_key] = self.helper._get_mtf_slope_accel_score(df, col_name, mtf_slope_accel_weights, df_index, method_name, bipolar=bipolar, ascending=ascending)
        return signals_data

    def _normalize_raw_data(self, df_index: pd.Index, signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> Dict[str, pd.Series]:
        """
        【V16.0 · 全息筹码清洗与午后压制归一化】构建筹码暗流的动力学数据处理逻辑与午后资金流占比归一化。
        """
        normalized = {}
        normalized["winner_rate_norm"] = signals["winner_rate_raw"].clip(0, 1)
        normalized["profit_margin_norm"] = (signals["winner_profit_margin_avg_raw"] / 0.2).clip(0, 1)
        normalized["chip_stability_norm"] = (signals["chip_stability_raw"] / 100.0).clip(0, 1)
        normalized["winner_stability_norm"] = (signals["winner_stability_index_raw"] / 100.0).clip(0, 1)
        normalized["profit_pressure_norm"] = (signals["profit_pressure_raw"] / 100.0).clip(0, 1)
        normalized["trapped_pressure_norm"] = (signals["pressure_trapped_raw"] / 100.0).clip(0, 1)
        normalized["vpa_efficiency_norm"] = ((signals["vpa_efficiency_raw"] - 50) / 50).clip(-1, 1)
        normalized["mf_conviction_norm"] = ((signals["main_force_conviction_index_raw"] - 50) / 50).clip(-1, 1)
        normalized["flow_consistency_norm"] = (signals["flow_consistency_raw"] / 100.0).clip(0, 1)
        net_mf_window_max = signals["net_mf_amount_raw"].abs().rolling(window=55, min_periods=1).max().replace(0, 1)
        normalized["net_mf_norm"] = (signals["net_mf_amount_raw"] / net_mf_window_max).clip(-1, 1)
        normalized["market_sentiment_norm"] = self.helper._normalize_series(signals["market_sentiment_score_raw"], df_index, bipolar=True)
        normalized["trend_confirmation_norm"] = (signals["trend_confirmation_score_raw"] / 100.0).clip(0, 1)
        normalized["loser_pain_norm"] = (signals["loser_loss_margin_avg_raw"].abs() / 0.3).clip(0, 1)
        normalized["panic_cascade_norm"] = (signals["panic_selling_cascade_raw"] / 100.0).clip(0, 1)
        normalized["stealth_flow_norm"] = (signals["stealth_flow_ratio_raw"] / 0.3).clip(0, 1)
        normalized["intraday_accum_norm"] = (signals["intraday_accumulation_confidence_raw"] / 100.0).clip(0, 1)
        normalized["peak_concentration_norm"] = (signals["peak_concentration_raw"] / 100.0).clip(0, 1)
        normalized["chip_concentration_ratio_norm"] = (signals["chip_concentration_ratio_raw"] / 100.0).clip(0, 1)
        profit_cushion_raw = (signals["close_raw"] - signals["cost_50pct_raw"]) / signals["cost_50pct_raw"].replace(0, 1)
        normalized["profit_cushion_norm"] = (profit_cushion_raw / 0.3).clip(-1, 1)
        normalized["long_term_chip_norm"] = (signals["long_term_chip_ratio_raw"] / 100.0).clip(0, 1)
        normalized["high_lock_norm"] = (signals["intraday_high_lock_ratio_raw"] / 100.0).clip(0, 1)
        normalized["uptrend_strength_norm"] = (signals["uptrend_strength_raw"] / 100.0).clip(0, 1)
        normalized["panic_absorption_norm"] = (signals["panic_buy_absorption_raw"] / 100.0).clip(0, 1)
        normalized["support_strength_norm"] = (signals["support_strength_raw"] / 100.0).clip(0, 1)
        normalized["chip_transfer_norm"] = signals["chip_transfer_eff_raw"].clip(0, 1)
        normalized["pressure_release_norm"] = (signals["pressure_release_raw"] / 100.0).clip(0, 1)
        normalized["distribution_conf_norm"] = (signals["distribution_conf_raw"] / 100.0).clip(0, 1)
        normalized["gap_defense_norm"] = (signals["gap_defense_raw"] / 100.0).clip(0, 1)
        normalized["tick_balance_norm"] = signals["tick_balance_raw"].clip(0, 1)
        sm_window_max = signals["smart_money_net_raw"].abs().rolling(window=55, min_periods=1).max().replace(0, 1)
        normalized["smart_money_net_norm"] = (signals["smart_money_net_raw"] / sm_window_max).clip(-1, 1)
        tick_window_max = signals["tick_large_net_raw"].abs().rolling(window=55, min_periods=1).max().replace(0, 1)
        normalized["tick_large_net_norm"] = (signals["tick_large_net_raw"] / tick_window_max).clip(-1, 1)
        normalized["smart_money_attack_norm"] = (signals["smart_money_attack_raw"] / 100.0).clip(0, 1)
        normalized["smart_money_div_norm"] = (signals["smart_money_div_raw"] / 100.0).clip(0, 1)
        normalized["wash_trade_norm"] = (signals["wash_trade_raw"] / 100.0).clip(0, 1)
        normalized["closing_flow_norm"] = (signals["closing_flow_raw"] / 0.3).clip(0, 1)
        normalized["afternoon_flow_norm"] = (signals["afternoon_flow_raw"] / 0.5).clip(0, 1)
        normalized["chip_flow_intensity_norm"] = (signals["chip_flow_intensity_raw"] / 100.0).clip(0, 1)
        normalized["order_anomaly_filter"] = (1.0 - (signals["order_anomaly_raw"] / 100.0)).clip(0, 1)
        normalized["order_anomaly_raw_norm"] = (signals["order_anomaly_raw"] / 100.0).clip(0, 1)
        normalized["volatility_instability_norm"] = (signals["volatility_instability_raw"] / 100.0).clip(0, 1)
        normalized["stability_factor"] = (1.0 - normalized["volatility_instability_norm"]).clip(0, 1)
        entropy_window_max = signals["structural_entropy_raw"].abs().rolling(window=21, min_periods=1).max().replace(0, 1)
        normalized["entropy_change_norm"] = (signals["structural_entropy_raw"] / entropy_window_max).clip(-1, 1)
        normalized["structural_order_factor"] = (1.0 - normalized["entropy_change_norm"]).clip(0, 1)
        normalized["reversion_freq_norm"] = (signals["mean_reversion_freq_raw"] / 100.0).clip(0, 1)
        normalized["theme_thermal_norm"] = (signals["theme_hotness_raw"] / 100.0).clip(0, 1)
        normalized["game_friction_norm"] = (signals["game_intensity_raw"] / 100.0).clip(0, 1)
        turnover_roll_max = signals["turnover_rate_raw"].rolling(window=55, min_periods=1).max().replace(0, 0.01)
        normalized["turnover_relative_norm"] = (signals["turnover_rate_raw"] / turnover_roll_max).clip(0, 1)
        normalized["abnormal_vol_norm"] = (signals["abnormal_vol_raw"] / 100.0).clip(0, 1)
        sm_roll_sum = signals["smart_money_net_raw"].rolling(window=21, min_periods=5).sum()
        sm_roll_max = sm_roll_sum.abs().rolling(window=55, min_periods=1).max().replace(0, 1)
        normalized["hab_smart_money"] = (sm_roll_sum / sm_roll_max).clip(-1, 1)
        tick_roll_sum = signals["tick_large_net_raw"].rolling(window=21, min_periods=5).sum()
        tick_roll_max = tick_roll_sum.abs().rolling(window=55, min_periods=1).max().replace(0, 1)
        normalized["hab_tick_large"] = (tick_roll_sum / tick_roll_max).clip(-1, 1)
        normalized["hab_flow_inertia"] = (normalized["hab_smart_money"] * 0.6 + normalized["hab_tick_large"] * 0.4).clip(-1, 1)
        normalized["hab_sentiment_memory"] = normalized["market_sentiment_norm"].rolling(window=21, min_periods=5).mean().clip(-1, 1)
        normalized["hab_volatility_memory"] = normalized["stability_factor"].rolling(window=21, min_periods=5).mean().clip(0, 1)
        pain_roll_mean = signals["loser_loss_margin_avg_raw"].abs().rolling(window=21, min_periods=5).mean()
        pain_roll_max = signals["loser_loss_margin_avg_raw"].abs().rolling(window=55, min_periods=1).max().replace(0, 1)
        normalized["hab_pain_saturation"] = (pain_roll_mean / pain_roll_max).clip(0, 1)
        trapped_roll_mean = signals["pressure_trapped_raw"].rolling(window=21, min_periods=5).mean()
        trapped_roll_max = signals["pressure_trapped_raw"].rolling(window=55, min_periods=1).max().replace(0, 1)
        normalized["hab_trapped_saturation"] = (trapped_roll_mean / trapped_roll_max).clip(0, 1)
        stealth_roll_sum = signals["stealth_flow_ratio_raw"].rolling(window=21, min_periods=5).sum()
        stealth_roll_max = stealth_roll_sum.abs().rolling(window=55, min_periods=1).max().replace(0, 1)
        normalized["hab_stealth_accum"] = (stealth_roll_sum / stealth_roll_max).clip(0, 1)
        def normalize_kinetic_with_gating(base_series, slope, accel, jerk, max_series):
            threshold = max_series * 0.05
            mask = base_series.abs() > threshold
            n_slope = np.tanh(slope).where(mask, 0.0)
            n_accel = np.tanh(accel).where(mask, 0.0)
            n_jerk = np.tanh(jerk).where(mask, 0.0)
            return n_slope, n_accel, n_jerk
        s_sm, a_sm, j_sm = normalize_kinetic_with_gating(signals["smart_money_net_raw"], signals["slope_smart_money_net_raw"], signals["accel_smart_money_net_raw"], signals["jerk_smart_money_net_raw"], sm_window_max)
        normalized["kinetic_smart_money"] = (s_sm * 0.5 + a_sm * 0.3 + j_sm * 0.2).clip(-1, 1)
        s_sent, a_sent, j_sent = normalize_kinetic_with_gating(signals["market_sentiment_score_raw"], signals["slope_market_sentiment_score_raw"], signals["accel_market_sentiment_score_raw"], signals["jerk_market_sentiment_score_raw"], pd.Series(100, index=df_index))
        normalized["kinetic_sentiment"] = (s_sent * 0.5 + a_sent * 0.3 + j_sent * 0.2).clip(-1, 1)
        s_vol, a_vol, j_vol = normalize_kinetic_with_gating(signals["volatility_instability_raw"], signals["slope_volatility_instability_raw"], signals["accel_volatility_instability_raw"], signals["jerk_volatility_instability_raw"], pd.Series(100, index=df_index))
        normalized["kinetic_stability"] = -1.0 * (s_vol * 0.5 + a_vol * 0.3 + j_vol * 0.2).clip(-1, 1)
        pain_max_series = signals["loser_loss_margin_avg_raw"].abs().rolling(window=55, min_periods=1).max().replace(0, 1)
        s_pain, a_pain, j_pain = normalize_kinetic_with_gating(signals["loser_loss_margin_avg_raw"], signals["slope_loser_loss_margin_avg_raw"], signals["accel_loser_loss_margin_avg_raw"], signals["jerk_loser_loss_margin_avg_raw"], pain_max_series)
        normalized["kinetic_pain"] = -1.0 * (s_pain * 0.4 + a_pain * 0.4 + j_pain * 0.2).clip(-1, 1)
        s_to, a_to, j_to = normalize_kinetic_with_gating(signals["turnover_rate_raw"], signals["slope_turnover_rate_raw"], signals["accel_turnover_rate_raw"], signals["jerk_turnover_rate_raw"], turnover_roll_max)
        normalized["kinetic_release"] = (s_to * 0.5 + a_to * 0.3 + j_to * 0.2).clip(0, 1)
        stealth_max_series = signals["stealth_flow_ratio_raw"].rolling(window=55, min_periods=1).max().replace(0, 1)
        s_stl, a_stl, j_stl = normalize_kinetic_with_gating(signals["stealth_flow_ratio_raw"], signals["slope_stealth_flow_ratio_raw"], signals["accel_stealth_flow_ratio_raw"], signals["jerk_stealth_flow_ratio_raw"], stealth_max_series)
        normalized["kinetic_stealth"] = (s_stl * 0.5 + a_stl * 0.3 + j_stl * 0.2).clip(-1, 1)
        chip_max_series = signals["chip_flow_intensity_raw"].rolling(window=55, min_periods=1).max().replace(0, 1)
        s_chip, a_chip, j_chip = normalize_kinetic_with_gating(signals["chip_flow_intensity_raw"], signals["slope_chip_flow_intensity_raw"], signals["accel_chip_flow_intensity_raw"], signals["jerk_chip_flow_intensity_raw"], chip_max_series)
        normalized["kinetic_chip_flow"] = (s_chip * 0.5 + a_chip * 0.3 + j_chip * 0.2).clip(-1, 1)
        _temp_debug_values["归一化处理"] = normalized
        return normalized

    def _calculate_belief_solidity(self, df_index: pd.Index, signals: Dict[str, pd.Series], normalized: Dict[str, pd.Series], params: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V5.8 · 非线性相变增益版】计算赢家信念坚固度。
        核心升级：
        1. 静态+惯性+微观：构建扎实的线性基础分。
        2. 非线性相变 (Phase Transition): 引入"动态指数增益函数"。
           逻辑：信念必须由趋势验证。
           公式：Final = Base ^ (2.0 - TrendStrength)。
           效果：弱势趋势下的高获利盘被视为"散户死扛"(Square Penalty)，强势趋势下的获利盘被视为"真信念"(Linear Retention)。
        """
        weights = params["belief_weights"]
        # 1. 静态状态 (State) - 35%
        winner_rate = normalized["winner_rate_norm"]
        profit_cushion = normalized["profit_cushion_norm"]
        peak_concentration = normalized["peak_concentration_norm"]
        chip_concentration_ratio = normalized["chip_concentration_ratio_norm"]
        structure_score = (peak_concentration * 0.6 + chip_concentration_ratio * 0.4).clip(0, 1)
        static_belief = (winner_rate * 0.3 + profit_cushion * 0.3 + structure_score * 0.4).clip(0, 1)
        # 2. 惯性状态 (Inertia/HAB) - 45%
        hab_belief = normalized["hab_belief_inertia"]
        chip_stability = normalized["chip_stability_norm"]
        inertia_belief = (hab_belief * 0.7 + chip_stability * 0.3).clip(0, 1)
        # 3. 基础线性坚固度
        base_solidity = (static_belief * 0.35 + inertia_belief * 0.45).clip(0, 1)
        # 4. 动量修正
        slope = normalized["slope_winner_rate"]
        accel = normalized["accel_winner_rate"]
        high_lock = normalized["high_lock_norm"]
        raw_kinetic = (slope * 0.15 + accel * 0.05).clip(-0.2, 0.2)
        micro_bonus = (high_lock * 0.1).where(raw_kinetic > 0, 0.0)
        kinetic_factor = raw_kinetic + micro_bonus
        # 线性汇总
        linear_solidity = (base_solidity + kinetic_factor + 0.1).clip(0, 1)
        # 5. 非线性相变增益 (Non-linear Phase Transition)
        uptrend = normalized["uptrend_strength_norm"]
        # 动态指数: 趋势越强，指数越小(接近1.0，保留原值)；趋势越弱，指数越大(接近2.0，平方惩罚)
        dynamic_exponent = 2.0 - uptrend
        phase_transition_solidity = linear_solidity.pow(dynamic_exponent)
        # 6. 反身性极值惩罚
        euphoria_risk = (static_belief > 0.95) & (structure_score < 0.4)
        euphoria_penalty = pd.Series(1.0, index=df_index)
        euphoria_penalty.loc[euphoria_risk] = 0.5
        # 最终得分
        belief_solidity = phase_transition_solidity * euphoria_penalty
        print(f"  [Probe] 信念坚固度V5.8详情 (前3行):")
        print(f"    LinearBase: {linear_solidity.head(3).values}")
        print(f"    TrendCatalyst: {uptrend.head(3).values} -> Exponent: {dynamic_exponent.head(3).values}")
        print(f"    PhaseTransition: {phase_transition_solidity.head(3).values}")
        print(f"    Final Solidity: {belief_solidity.head(3).values}")
        _temp_debug_values["信念坚固度"] = {
            "static_belief": static_belief,
            "inertia_belief": inertia_belief,
            "linear_solidity": linear_solidity,
            "dynamic_exponent": dynamic_exponent,
            "phase_transition_solidity": phase_transition_solidity,
            "euphoria_penalty": euphoria_penalty,
            "score": belief_solidity
        }
        return belief_solidity

    def _calculate_pressure_digestion(self, df_index: pd.Index, signals: Dict[str, pd.Series], normalized: Dict[str, pd.Series], params: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V6.2 · 非线性相变增益版】计算压力消化力。
        核心升级：
        1. 基础架构: 保持耗散结构(V6.0)与敌我识别/战略防御(V6.1)逻辑。
        2. 非线性相变 (Phase Transition): 引入"双因子动态指数"。
           逻辑：压力消化的含金量取决于"趋势环境"和"量能效率"。
           公式：Exponent = 2.0 - (0.5 * Trend + 0.5 * VPA)。
           效果：强势趋势+高效量能 -> 线性保留(真实消化)；弱势趋势+低效量能 -> 平方级衰减(抵抗式下跌)。
        """
        weights = params["pressure_weights"]
        # 1. 流量维度 (Snapshot)
        load = (normalized["profit_pressure_norm"] * 0.5 + normalized["trapped_pressure_norm"] * 0.5).clip(0.1, 1.0)
        active_absorb = normalized["panic_absorption_norm"]
        passive_support = normalized["support_strength_norm"]
        metabolism = (active_absorb * 0.6 + passive_support * 0.4).clip(0, 1)
        snapshot_coverage = np.tanh((metabolism / load) - 1.0).clip(-1, 1)
        # 2. 存量维度 (HAB)
        hab_capacity = normalized["hab_absorption_capacity"]
        # 3. 动力学维度 (Kinetics)
        net_kinetics = (normalized["kinetic_absorb"] - normalized["kinetic_pressure"]).clip(-1, 1)
        # 4. 基础消化分
        base_digestion = (snapshot_coverage * 0.3 + hab_capacity * 0.5 + net_kinetics * 0.2).clip(-1, 1)
        # 5. 质量修正 (Quality)
        transfer_eff = normalized["chip_transfer_norm"]
        release_index = normalized["pressure_release_norm"]
        tick_balance = normalized["tick_balance_norm"]
        balance_score = ((tick_balance - 0.5) * 2).clip(0, 1)
        quality_coef = (transfer_eff * 0.4 + release_index * 0.3 + balance_score * 0.3).clip(0, 1)
        # 6. 敌我识别与战略防御 (V6.1)
        dist_conf = normalized["distribution_conf_norm"]
        adversary_penalty = dist_conf * 0.8
        gap_defense = normalized["gap_defense_norm"]
        defense_bonus = 1.0 + (gap_defense * 0.3)
        
        adjusted_base = base_digestion * (0.8 + 0.4 * quality_coef)
        if_positive = adjusted_base * defense_bonus * (1.0 - adversary_penalty)
        if_negative = adjusted_base * (1.0 + adversary_penalty)
        linear_digestion = if_positive.where(adjusted_base > 0, if_negative).clip(-1, 1)
        
        # --- V6.2 非线性相变增益模块 ---
        # 因子1: 趋势强度 (Trend Strength) - 顺势消化事半功倍
        uptrend = normalized["uptrend_strength_norm"]
        # 因子2: VPA效率 (Volume Efficiency) - 高效量能验证消化质量
        # vpa_norm 是 [-1, 1]，我们需要将其映射到 [0, 1] 用于指数计算 (越接近1越好)
        vpa_factor = (normalized["vpa_efficiency_norm"] + 1) / 2
        
        # 动态指数构建
        # 基准指数 2.0 (平方级惩罚，对应弱势震荡)
        # 趋势和VPA越好，指数越小，直至接近 1.0 (线性保留)
        # Exponent = 2.0 - (0.5 * Trend + 0.5 * VPA)
        dynamic_exponent = 2.0 - (uptrend * 0.5 + vpa_factor * 0.5)
        # 限制指数范围 [1.0, 3.0] (防止过度奖励或计算溢出)
        dynamic_exponent = dynamic_exponent.clip(1.0, 3.0)
        
        # 应用非线性变换: Sign * |Base|^Exponent
        phase_transition_digestion = np.sign(linear_digestion) * (linear_digestion.abs().pow(dynamic_exponent))
        
        final_digestion = phase_transition_digestion.clip(-1, 1)

        print(f"  [Probe] 压力消化V6.2详情 (前3行):")
        print(f"    LinearDigestion: {linear_digestion.head(3).values}")
        print(f"    DynamicExponent: {dynamic_exponent.head(3).values} (Trend: {uptrend.head(3).values}, VPA: {vpa_factor.head(3).values})")
        print(f"    Final Digestion: {final_digestion.head(3).values}")
        
        _temp_debug_values["压力消化力"] = {
            "snapshot_coverage": snapshot_coverage,
            "hab_capacity": hab_capacity,
            "net_kinetics": net_kinetics,
            "quality_coef": quality_coef,
            "adversary_penalty": adversary_penalty,
            "defense_bonus": defense_bonus,
            "linear_digestion": linear_digestion,
            "dynamic_exponent": dynamic_exponent,
            "score": final_digestion
        }
        return final_digestion

    def _calculate_flow_consensus(self, df_index: pd.Index, signals: Dict[str, pd.Series], normalized: Dict[str, pd.Series], params: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.3 · 非线性共振与诚信穿透版】计算资金共识度。
        修改逻辑：
        1. 引入非线性相变模块：基于协同进攻强度动态调整指数，强化“共振”捕捉能力。
        2. 诚信度指数级衰减：对高对倒（Wash Trade）行为从线性扣分升级为非线性抑制。
        3. 增强噪音过滤：利用大单异常强度对非线性后的结果进行最终平滑。
        """
        # 1. 基础分层构建 (Flow + Stock + Kinetics)
        snapshot_flow = (normalized["smart_money_net_norm"] * 0.5 + normalized["tick_large_net_norm"] * 0.3 + normalized["net_mf_norm"] * 0.2).clip(-1, 1)
        hab_stock = normalized["hab_flow_inertia"]
        kinetic_impulse = normalized["kinetic_smart_money"]
        base_consensus = (snapshot_flow * 0.25 + hab_stock * 0.45 + kinetic_impulse * 0.3).clip(-1, 1)
        # 2. 诚信穿透与伏击修正
        wash_trade = normalized["wash_trade_norm"]
        integrity_factor = 1.0 - (wash_trade * 0.6) # 线性基础诚信因子
        closing_flow = normalized["closing_flow_norm"]
        ambush_booster = (1.0 + closing_flow * 0.25).where(base_consensus > 0, 1.0)
        # 3. 核心升级：非线性共振调制 (Phase Transition)
        attack = normalized["smart_money_attack_norm"]
        # 动态指数：进攻协同度越高(1.0)，指数越小(1.0)，原始动能保留越完整；协同度越低，指数越大，动能衰减越快
        # Exponent 范围 [1.0, 2.5]
        res_exponent = (1.0 + (1.0 - attack) * 1.5).clip(1.0, 2.5)
        # 4. 冲突博弈与噪音过滤
        divergence = normalized["smart_money_div_norm"]
        game_penalty = divergence * 0.8
        anomaly_filter = normalized["order_anomaly_filter"]
        # 5. 最终融合计算
        # 计算逻辑：(基础分 * 诚信 * 伏击) -> 应用非线性共振指数 -> 应用博弈惩罚 -> 噪音过滤
        consensus_pre = (base_consensus * integrity_factor * ambush_booster).clip(-1, 1)
        # 执行非线性变换：Sign(x) * |x|^Exponent
        phase_transition_consensus = np.sign(consensus_pre) * (consensus_pre.abs().pow(res_exponent))
        # 应用机构博弈（分歧）惩罚
        if_positive = phase_transition_consensus * (1.0 - game_penalty)
        if_negative = phase_transition_consensus # 负向流向保持风险提示，不进行分歧对冲
        final_consensus = if_positive.where(phase_transition_consensus > 0, if_negative) * anomaly_filter
        final_consensus = final_consensus.clip(-1, 1)
        print(f"  [Probe] 资金共识V7.3详情 (前3行):")
        print(f"    BasePre: {consensus_pre.head(3).values}")
        print(f"    ResExponent: {res_exponent.head(3).values} (Attack: {attack.head(3).values})")
        print(f"    Final Consensus: {final_consensus.head(3).values}")
        _temp_debug_values["资金共识度"] = {
            "base_consensus": base_consensus,
            "res_exponent": res_exponent,
            "integrity_factor": integrity_factor,
            "ambush_booster": ambush_booster,
            "phase_transition": phase_transition_consensus,
            "score": final_consensus
        }
        return final_consensus

    def _calculate_contextual_modulator(self, df_index: pd.Index, signals: Dict[str, pd.Series], normalized: Dict[str, pd.Series], params: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V10.0 · 非线性共振激活】在V9.0全息场的基础上，引入时空相干性(Coherence)作为非线性增益的开关。
        
        逻辑演进：
        1. 线性基座 (Linear Base): 计算热力-摩擦-HAB修正后的线性调节系数。
        2. 相干性检测 (Coherence Check): 检测"状态(Snapshot)"、"速度(Kinematics)"、"记忆(HAB)"三者是否共振。
        3. 非线性激活 (Non-linear Activation): 
           - 高相干 (Resonance): 线性保留甚至凸性奖励 (Exponent -> 0.8)。
           - 低相干 (Conflict): 平方级抑制 (Exponent -> 2.0)，过滤虚假信号。
        
        公式：Final = 1.0 + Sign(Linear-1) * |Linear-1| ^ (2.0 - Coherence * 1.2)
        """
        weights = params["context_weights"]
        hab_weights = params.get("hab_weights", {})
        env_weights = params.get("environment_weights", {})
        res_params = params.get("resonance_params", {})
        
        # --- Layer 1: 物理场快照 (Snapshot) ---
        sentiment = normalized["market_sentiment_norm"]
        trend = normalized["trend_confirmation_norm"] * 2 - 1 
        stability_factor = normalized["stability_factor"]
        order_factor = normalized["structural_order_factor"]
        
        base_snapshot = (
            sentiment * weights["market_sentiment"] + 
            trend * weights["trend_confirmation"] +
            (stability_factor * 2 - 1) * weights["volatility_stability"] + 
            (order_factor * 2 - 1) * weights["structural_order"]
        ).clip(-1, 1)
        
        # --- Layer 2: 动力学修正 (Kinematics) ---
        k_sent = normalized["kinetic_sentiment"]
        k_stab = normalized["kinetic_stability"]
        kinematic_score = (k_sent * 0.6 + k_stab * 0.4).clip(-1, 1)
        
        modified_field = base_snapshot + kinematic_score * 0.3
        
        # --- Layer 3: 存量记忆缓冲 (HAB) ---
        hab_flow = normalized["hab_flow_inertia"]
        hab_sent = normalized["hab_sentiment_memory"]
        hab_vol = normalized["hab_volatility_memory"]
        
        hab_score = (
            hab_flow * hab_weights["flow_inertia"] + 
            hab_sent * hab_weights["sentiment_memory"] + 
            (hab_vol * 2 - 1) * hab_weights["volatility_memory"]
        ).clip(-0.5, 1.0)
        
        # --- Layer 4: 环境热力与摩擦 (Environment) ---
        thermal_boost = normalized["theme_thermal_norm"] * env_weights["theme_thermal"]
        friction_drag = normalized["game_friction_norm"] * env_weights["game_friction"]
        reversion_drag = normalized["reversion_freq_norm"] * weights["reversion_penalty"]
        
        # --- V9.0 线性基座计算 ---
        numerator = (1.0 + modified_field * 0.6 + hab_score * 0.4) * (1.0 + thermal_boost)
        denominator = 1.0 + reversion_drag + friction_drag
        linear_modulator = numerator / denominator
        
        # --- V10.0 非线性共振模块 (Resonance Module) ---
        
        # 1. 计算相干性 (Coherence) [0, 1]
        # 逻辑：判断 Snapshot, Kinematics, HAB 三个向量的方向一致性
        # 使用简单的符号乘积和幅值加权来估算
        # 如果三者同向（且幅值显著），Coherence 趋近 1.0
        # 如果方向冲突，Coherence 趋近 0.0
        
        # 为了计算方便，先将 hab_score 映射回 [-1, 1] 的逻辑空间用于方向判断
        hab_direction = hab_score.clip(-1, 1)
        
        # 向量组
        v1 = base_snapshot
        v2 = kinematic_score
        v3 = hab_direction
        
        # 计算两两的点积 (Dot Product Proxy)，这里简化为符号一致性 * 幅值
        # 只有当大家都强且同向时，才是真共振
        # 这里的 coherence 算法：(v1*v2 + v2*v3 + v3*v1) / 3，再归一化到 [0, 1]
        # 结果范围 [-1, 1] -> 映射到 [0, 1] (负相关也是一种"不相干"的表现，对于做多来说)
        raw_coherence = (v1 * v2 + v2 * v3 + v3 * v1) / 3.0
        # 我们只关心"正向共振"（一起看多），如果是"负向共振"（一起看空），对于Modulator来说也是一种确定性
        # 但 Context Modulator 主要用于增强"信念"，所以我们关注 "一致性"。
        # 修正：我们关注 magnitude of alignment.
        alignment_magnitude = raw_coherence.clip(-1, 1)
        # 映射：1.0 -> 1.0 (完美共振), 0.0 -> 0.0 (无关), -1.0 -> 0.0 (冲突/反向共振视同无多头共振)
        # 这里为了稳健，如果三个指标都为负，alignment是正的，这会增强"负分"（即抑制信念），逻辑通顺。
        coherence = ((alignment_magnitude + 1.0) / 2.0).clip(0, 1)
        
        # 2. 动态指数构建
        # Base Exp = 2.0 (平方级衰减，默认不信任)
        # Target Exp = 0.8 (凸性奖励，信任并放大)
        # Exp = Base - Coherence * (Base - Min)
        base_exp = res_params["base_exponent"]
        min_exp = res_params["min_exponent"]
        dynamic_exponent = base_exp - coherence * (base_exp - min_exp)
        
        # 3. 非线性变换
        # Modulator 中心是 1.0
        # Final = 1.0 + Sign(Linear-1) * |Linear-1|^Exp
        deviation = linear_modulator - 1.0
        sign_dev = np.sign(deviation)
        abs_dev = deviation.abs()
        
        # 应用指数
        # 注意：当 deviation 很小 (<1) 时，Exp越小，结果越大(放大)；Exp越大，结果越小(抑制)。
        # 这符合逻辑：Coherence高 -> Exp小 -> 放大微小的正向偏差。
        # Coherence低 -> Exp大 -> 抑制微小的偏差（视为噪音）。
        final_modulator = 1.0 + sign_dev * (abs_dev.pow(dynamic_exponent))
        
        # 4. 最终数值安全钳位 [0.5, 2.0]
        final_modulator = final_modulator.clip(0.5, 2.0)
        
        print(f"  [Probe] 情境调制V10.0: Linear={linear_modulator.tail(1).values[0]:.3f}, "
              f"Coherence={coherence.tail(1).values[0]:.3f}, "
              f"Exp={dynamic_exponent.tail(1).values[0]:.3f} -> Final={final_modulator.tail(1).values[0]:.4f}")
        
        _temp_debug_values["情境调制"] = {
            "linear_modulator": linear_modulator,
            "coherence": coherence,
            "dynamic_exponent": dynamic_exponent,
            "final_modulator": final_modulator
        }
        return final_modulator

    def _calculate_adversary_capitulation(self, df_index: pd.Index, signals: Dict[str, pd.Series], normalized: Dict[str, pd.Series], params: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V14.0 · 非线性伽马膨胀】三体共振驱动的相变模型。
        
        核心逻辑：
        1. 痛苦势能 (Pain): 存量与深度的积累。
        2. 清洗效率 (Cleaning): 放量与换手的确认。
        3. 恐慌烈度 (Panic): 情绪的引爆。
        
        非线性变换 (Gamma Expansion):
        - Coherence = GeometricMean(Pain, Cleaning, Panic)
        - Gamma = Base - Coherence * (Base - Min)
        - Final = Raw ^ Gamma
        
        效果：
        - 阴跌无量 (Low Coherence) -> Gamma > 1 -> 分数被压缩 (Trap Identified).
        - 放量暴跌 (High Coherence) -> Gamma < 1 -> 分数被膨胀 (Opportunity Amplified).
        """
        weights = params["capitulation_weights"]
        
        # 1. 痛苦势能 (Pain Potential)
        static_pain = (normalized["loser_pain_norm"] * 0.6 + normalized["trapped_pressure_norm"] * 0.4).clip(0, 1)
        kinetic_pain = normalized["kinetic_pain"].clip(0, 1)
        saturation_score = (normalized["hab_pain_saturation"] * 0.6 + normalized["hab_trapped_saturation"] * 0.4).clip(0, 1)
        
        pain_force = (
            static_pain * weights["static_pain"] + 
            kinetic_pain * weights["kinetic_pain"] + 
            saturation_score * weights["pain_saturation"]
        ).clip(0, 1)
        
        # 2. 清洗效率 (Cleansing Efficiency)
        turnover_score = normalized["turnover_relative_norm"]
        abnormal_score = normalized["abnormal_vol_norm"]
        kinetic_release = normalized["kinetic_release"]
        
        raw_release = (turnover_score * 0.4 + abnormal_score * 0.3 + kinetic_release * 0.3).clip(0, 1)
        cleaning_efficiency = 0.4 + 0.6 * raw_release
        
        # 3. 恐慌烈度 (Panic Intensity)
        panic_intensity = normalized["panic_cascade_norm"]
        
        # --- V14.0 三体共振与伽马计算 ---
        
        # 原始线性分 (Raw Linear Score)
        # 基础逻辑：痛苦 * 清洗。恐慌作为共振因子参与 Gamma 计算，不再直接乘入 Base。
        # (恐慌是催化剂，不是燃料本身)
        raw_score = pain_force * cleaning_efficiency
        
        # 计算相干性 (Coherence)
        # 使用几何平均数来衡量三者的"协同高度"。只有当三者都强时，几何平均才高。
        # 为了防止0值过度惩罚，添加微小常数 epsilon
        # Coherence 越高，代表痛苦深、释放大、恐慌足 -> 完美投降
        epsilon = 0.05
        coherence = np.exp(
            (np.log(pain_force + epsilon) + np.log(cleaning_efficiency + epsilon) + np.log(panic_intensity + epsilon)) / 3.0
        )
        # 归一化调整，因为加了epsilon，最大值约 1+epsilon，稍微clip一下
        coherence = coherence.clip(0, 1)
        
        # 动态伽马 (Dynamic Gamma)
        # Base=2.0 (压缩), Min=0.6 (膨胀)
        gamma_base = weights.get("gamma_base", 2.0)
        gamma_min = weights.get("gamma_min", 0.6)
        
        # Coherence 越高，Gamma 越小 (趋向于 Min)
        dynamic_gamma = gamma_base - coherence * (gamma_base - gamma_min)
        
        # 非线性激活
        # Score = Raw ^ Gamma
        # 例1 (Trap): Raw=0.3 (Pain高, Clean低), Coh=0.3 -> Gamma=1.6 -> Final = 0.3^1.6 = 0.14 (抑制)
        # 例2 (Gold): Raw=0.8 (Pain高, Clean高), Coh=0.9 -> Gamma=0.7 -> Final = 0.8^0.7 = 0.85 (提升)
        capitulation_score = raw_score.pow(dynamic_gamma)
        
        # 最后应用恐慌作为极值倍增器 (仅在Gamma处理后，作为额外的Bonus，防止 Gamma把高分压得太平)
        # 或者，直接由 Gamma 承担所有非线性工作。
        # V14策略：Gamma 已经包含了 Panic 的信息（在 Coherence 中），
        # 但为了保留 Panic 的"爆发性"，我们可以对最终结果做一个微调。
        # 这里选择不再额外乘 Panic，信任 Gamma 模型的相变能力。
        
        capitulation_score = capitulation_score.clip(0, 1)
        
        print(f"  [Probe] 投降共振V14.0: Raw={raw_score.tail(1).values[0]:.2f} (Pain:{pain_force.tail(1).values[0]:.2f}, Clean:{cleaning_efficiency.tail(1).values[0]:.2f}), "
              f"Panic={panic_intensity.tail(1).values[0]:.2f} -> "
              f"Coh={coherence.tail(1).values[0]:.2f}, Gamma={dynamic_gamma.tail(1).values[0]:.2f} -> "
              f"Final={capitulation_score.tail(1).values[0]:.4f}")
        
        _temp_debug_values["对手盘投降度"] = {
            "pain_force": pain_force,
            "cleaning_efficiency": cleaning_efficiency,
            "panic_intensity": panic_intensity,
            "raw_score": raw_score,
            "coherence": coherence,
            "dynamic_gamma": dynamic_gamma,
            "score": capitulation_score
        }
        return capitulation_score

    def _calculate_micro_stealth(self, df_index: pd.Index, signals: Dict[str, pd.Series], normalized: Dict[str, pd.Series], params: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V17.0 · 双曲正切非线性相变版】在核心隐蔽与掩护维度之间引入基于np.tanh的非线性化学反应。
        核心逻辑：
        1. 阻断线性欺骗：避免单一维度异常高分掩盖另一维度的缺失（例如纯“骗线”操作）。
        2. Tanh协同激活：计算 `core_stealth * manipulation_mask` 的协同度。利用 Tanh 曲线形成相变，双高则指数级奖励，单高则保持压制。
        3. 双重非线性夹击：内部乘法激活（相变） + 外部 Gamma 惩罚（波动压制），构建对高频噪音的终极过滤体系。
        """
        weights = params["stealth_weights"]
        stealth_snapshot = normalized["stealth_flow_norm"]
        intraday_snapshot = normalized["intraday_accum_norm"] * 0.7 + signals["mtf_intraday_accumulation_confidence"] * 0.3
        hab_memory = normalized["hab_stealth_accum"]
        stealth_kinetic = normalized["kinetic_stealth"].clip(0, 1) 
        chip_snapshot = normalized["chip_flow_intensity_norm"] * 0.4 + normalized["chip_transfer_norm"] * 0.4 + normalized["kinetic_chip_flow"].clip(0, 1) * 0.2
        core_stealth = (
            stealth_snapshot * weights["stealth_flow"] + 
            intraday_snapshot * weights["intraday_accum"] + 
            hab_memory * weights["hab_stealth_21d"] +
            stealth_kinetic * weights["kinetic_stealth_13d"] +
            chip_snapshot * weights["chip_undercurrent"]
        )
        wash_trade = normalized["wash_trade_norm"]
        closing_mask = normalized["closing_flow_norm"]
        afternoon_mask = normalized["afternoon_flow_norm"]
        order_anomaly = normalized["order_anomaly_raw_norm"]
        manipulation_mask = (
            wash_trade * weights["wash_trade_mask"] + 
            closing_mask * weights["closing_mask"] + 
            afternoon_mask * weights["afternoon_mask"] + 
            order_anomaly * weights["order_anomaly"]
        )
        synergy_gain_factor = weights.get("synergy_gain_factor", 3.0)
        raw_synergy = core_stealth * manipulation_mask
        synergy_activation = np.tanh(raw_synergy * synergy_gain_factor)
        base_stealth = core_stealth + manipulation_mask
        raw_stealth = base_stealth * (1.0 + synergy_activation)
        smart_money_resonance = normalized["smart_money_net_norm"].clip(0, 1)
        if_resonance_mask = raw_stealth * (1.0 + smart_money_resonance * 0.2)
        stability = normalized["stability_factor"]
        gamma_base = weights.get("volatility_suppression_gamma", 1.6)
        dynamic_gamma = 1.0 + (1.0 - stability) * (gamma_base - 1.0)
        stealth_score = if_resonance_mask.pow(dynamic_gamma)
        print(f"  [Probe] 非线性隐蔽V17.0：暗流={core_stealth.tail(1).values[0]:.4f}, 掩护={manipulation_mask.tail(1).values[0]:.4f}, "
              f"乘积={raw_synergy.tail(1).values[0]:.4f}, Tanh激活倍数={synergy_activation.tail(1).values[0]:.4f}, "
              f"环境Gamma={dynamic_gamma.tail(1).values[0]:.4f}, 最终隐蔽分={stealth_score.tail(1).values[0]:.4f}")
        _temp_debug_values["微观隐蔽度"] = {
            "norm_stealth_snapshot": stealth_snapshot,
            "norm_chip_undercurrent": chip_snapshot,
            "norm_hab_memory_21d": hab_memory,
            "norm_stealth_kinetic_13d": stealth_kinetic,
            "core_stealth_base": core_stealth,
            "manipulation_mask_base": manipulation_mask,
            "raw_synergy_product": raw_synergy,
            "tanh_synergy_activation": synergy_activation,
            "fused_raw_stealth_with_activation": raw_stealth,
            "stability_factor": stability,
            "dynamic_gamma_exponent": dynamic_gamma,
            "score": stealth_score
        }
        return stealth_score

    def _perform_final_fusion(self, df_index: pd.Index, belief: pd.Series, pressure: pd.Series, flow: pd.Series, capitulation: pd.Series, stealth: pd.Series, context: pd.Series, params: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V18.0 · 高阶张力收敛版】最终五维融合引擎。
        核心逻辑：
        1. 基础映射：将单极性的 belief 映射为双极性 [-1, 1]，与 pressure、flow 共同构成基础三元向量。
        2. 非对称势能叠加：将经过上游非线性相变的 capitulation 和 stealth 作为纯多头增强势能，直接叠加于基础分之上。
        3. 全局降维与钳位：通过 context 调制后，应用全局 final_exponent 平滑极端波动，并最终绝对钳位至 [-1, 1] 的规范空间。
        """
        weights = params["final_fusion_weights"]
        belief_bipolar = (belief * 2.0) - 1.0
        base_score = (belief_bipolar * weights["belief"] + pressure * weights["pressure"] + flow * weights["flow"])
        enhancement = (capitulation * weights["capitulation"] + stealth * weights["stealth"])
        raw_sum = base_score + enhancement
        modulated_score = raw_sum * context
        final_exponent = params.get("final_exponent", 1.8)
        final_score = np.sign(modulated_score) * (modulated_score.abs().pow(final_exponent))
        final_score = final_score.clip(-1.0, 1.0)
        print(f"  [Probe] 最终融合V18.0探针：基础三元={base_score.tail(1).values[0]:.4f} (信:{belief_bipolar.tail(1).values[0]:.2f}, 压:{pressure.tail(1).values[0]:.2f}, 资:{flow.tail(1).values[0]:.2f}), "
              f"暗流增强={enhancement.tail(1).values[0]:.4f} (降:{capitulation.tail(1).values[0]:.2f}, 隐:{stealth.tail(1).values[0]:.2f}), "
              f"情境调制={context.tail(1).values[0]:.4f}, 全局指数={final_exponent}, 最终信念得分={final_score.tail(1).values[0]:.4f}")
        _temp_debug_values["最终融合"] = {
            "belief_bipolar": belief_bipolar,
            "base_score": base_score,
            "enhancement": enhancement,
            "raw_sum": raw_sum,
            "modulated_score": modulated_score,
            "final_score": final_score
        }
        return final_score












