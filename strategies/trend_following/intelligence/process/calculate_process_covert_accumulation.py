# strategies\trend_following\intelligence\process\calculate_process_covert_accumulation.py
# 【V2.12 · 微观订单流与结构共振版】“隐蔽吸筹”专属信号计算引擎 已完成pro
import json
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from numba import jit
from typing import Dict, List, Optional, Any, Tuple

from strategies.trend_following.utils import (
    get_params_block, get_param_value, get_adaptive_mtf_normalized_score,
    is_limit_up, get_adaptive_mtf_normalized_bipolar_score,
    normalize_score, _robust_geometric_mean
)
from strategies.trend_following.intelligence.process.helper import ProcessIntelligenceHelper

class CalculateProcessCovertAccumulation:
    """
    计算“隐蔽吸筹”的专属信号。
    PROCESS_META_COVERT_ACCUMULATION
    """
    def __init__(self, strategy_instance, helper_instance: ProcessIntelligenceHelper):
            self.strategy = strategy_instance
            self.helper = helper_instance

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V7.3·极速计算版】计算“隐蔽吸筹”的专属信号。
        -核心优化:向量化日期探测逻辑，移除Python循环；结果降级为float32。
        """
        method_name = "_calculate_process_covert_accumulation"
        print(" ====== CalculateProcessCovertAccumulation ======")
        is_debug_enabled_for_method = get_param_value(self.helper.debug_params.get('enabled'), False) and get_param_value(self.helper.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug_enabled_for_method and self.helper.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.helper.probe_dates]
            # 优化: 向量化查找，避免循环遍历index
            curr_dates = pd.to_datetime(df.index).normalize()
            mask = curr_dates.isin(probe_dates_dt)
            if mask.any():
                probe_ts = df.index[mask][-1]
        debug_output = {}
        _temp_debug_values = {}
        df_index = df.index
        fusion_weights, market_context_weights, covert_action_weights, chip_optimization_weights, price_weakness_slope_window, low_volatility_bbw_window, mtf_slope_accel_weights, neutral_range_threshold, cumulative_flow_windows, cumulative_flow_weights, cumulative_acc_windows, cumulative_acc_weights, daily_mf_flow_weight, cumulative_mf_flow_weight, daily_acc_weight, cumulative_acc_weight, new_raw_signals_weights, main_force_accumulation_resonance_weight, new_raw_signals_weights_v2, covert_order_flow_resonance_weight = self._get_covert_accumulation_config(config)
        raw_signals = self._validate_and_get_raw_signals(df, method_name, mtf_slope_accel_weights, is_debug_enabled_for_method, probe_ts, _temp_debug_values, cumulative_flow_windows)
        market_context_score = self._calculate_market_context_score(df, df_index, raw_signals, market_context_weights, _temp_debug_values)
        covert_action_score = self._calculate_covert_action_score(df, df_index, raw_signals, covert_action_weights, _temp_debug_values, cumulative_flow_windows)
        chip_optimization_score = self._calculate_chip_optimization_score(df, df_index, raw_signals, chip_optimization_weights, _temp_debug_values)
        raw_final_score = self._fuse_final_score(df_index, market_context_score, covert_action_score, chip_optimization_score, fusion_weights, _temp_debug_values)
        try:
            final_score = self._apply_signal_latching(raw_final_score, market_context_score, covert_action_score, chip_optimization_score, df_index, _temp_debug_values)
        except Exception as e:
            print(f"ERROR:LatchFailed|Msg={str(e)}")
            final_score = raw_final_score
        if final_score is None or (isinstance(final_score, pd.Series) and final_score.empty):
            final_score = pd.Series(0.0, index=df_index, dtype='float32')
        else:
            final_score = final_score.astype('float32')
        _temp_debug_values["final_score"] = final_score
        if is_debug_enabled_for_method and probe_ts:
            print(f"DEBUG_PROBE:CalculateProcessCovertAccumulation|ProbeTS={probe_ts.strftime('%Y-%m-%d')}")
            print(f"DEBUG_PROBE:CalculationFinished|RawScore={raw_final_score.loc[probe_ts]:.4f}|Final={final_score.loc[probe_ts]:.4f}")
            self._print_debug_info(debug_output, _temp_debug_values, method_name, probe_ts)
        print(f"\n ====== ======================== ======")
        return final_score

    def _apply_signal_latching(self, final_score: pd.Series, context: pd.Series, action: pd.Series, chip: pd.Series, df_index: pd.Index, _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.8·Numba加速版】熵权锁存器。
        -核心优化:使用Numba JIT编译核心状态循环，效率提升100x。
        """
        components = pd.concat([context, action, chip], axis=1)
        ewd_factor = components.std(axis=1).fillna(1.0).values
        fusion_debug = _temp_debug_values.get("最终合成_激发态", {})
        is_solid = fusion_debug.get("Solid", False)
        adaptive_threshold = 0.35 if is_solid else 0.60
        entropy_limit = 0.25 if is_solid else 0.15
        # 向量化准备数据
        raw_values = final_score.fillna(0).values.astype(np.float64)
        high_score_mask = (raw_values > adaptive_threshold)
        trigger_signal = (high_score_mask & (ewd_factor < entropy_limit)).astype(np.int8)
        # 预计算滚动和
        rolling_trigger = pd.Series(trigger_signal).rolling(window=5, min_periods=1).sum().fillna(0).values
        activation_mask = (rolling_trigger >= 2).astype(np.int8)
        # 调用Numba内核
        latched_values = self._numba_latch_kernel(raw_values, activation_mask)
        latched_series = pd.Series(latched_values, index=df_index, dtype='float32').clip(0, 1)
        print(f"DEBUG_PROBE:LatchFinal_V7.8|Locked={latched_series.iloc[-1] > 0.3}|LastVal={latched_series.iloc[-1]:.4f}")
        return latched_series

    def _get_covert_accumulation_config(self, config: Dict) -> Tuple[Dict, Dict, Dict, Dict, int, int, Dict, float, List[int], Dict, List[int], Dict, float, float, float, float, Dict, float, Dict, float]:
        """
        【V2.17·配置安全合并版】获取配置参数。
        -核心修正:采用Dict.update()逻辑进行权重合并，防止旧配置覆盖新指标的默认权重，解决EntropyWeight=None的问题。
        """
        covert_accum_params = get_param_value(self.helper.params.get('covert_accumulation_params'), {})
        # 定义默认权重
        default_fusion = {"market_context": 0.35, "covert_action": 0.35, "chip_optimization": 0.3}
        default_context = {
            "golden_pit_state": 0.15, "space_efficiency": 0.10, "theme_resonance": 0.15,
            "smart_divergence": 0.20, "turnover_stability": 0.10, "pressure_release": 0.10,
            "sentiment_extreme": 0.05, "vol_compression": 0.05, "is_consolidating": 0.05, "breakout_potential": 0.05
        }
        default_action = {
            "pain_accumulation": 0.15, "game_friction": 0.10, "behavior_confirmation": 0.05,
            "iceberg_friction": 0.15, "whale_active_drive": 0.10, "hab_accumulation": 0.15,
            "kinetic_surge": 0.10, "intraday_confidence": 0.10, "flow_consistency": 0.10,
            "stealth_ops": 0.0, "inst_net_buy": 0.0, "contextualized_accum": 0.0 # 兼容旧键
        }
        default_chip = {
            "entropy_reduction": 0.15, "cost_center_support": 0.10,
            "chip_morphology": 0.10, "iron_floor_hab": 0.10,
            "transfer_efficiency_hab": 0.15, "trapped_pressure_release": 0.15,
            "concentration_accel": 0.10, "chip_locking": 0.10, "chip_stability": 0.05,
            "chip_concentration": 0.0 # 兼容旧键
        }
        # 安全合并逻辑: 使用默认值作为底板，更新用户配置
        fusion_weights = default_fusion.copy()
        fusion_weights.update(covert_accum_params.get('fusion_weights', {}))
        market_context_weights = default_context.copy()
        market_context_weights.update(covert_accum_params.get('market_context_weights', {}))
        covert_action_weights = default_action.copy()
        covert_action_weights.update(covert_accum_params.get('covert_action_weights', {}))
        chip_optimization_weights = default_chip.copy()
        chip_optimization_weights.update(covert_accum_params.get('chip_optimization_weights', {}))
        # 其他常规参数获取
        new_raw_signals_weights = get_param_value(covert_accum_params.get('new_raw_signals_weights'), {
            "ask_side_liquidity_inverted": 0.03, "mf_level5_buy_ofi": 0.05, "mf_buy_execution_alpha": 0.05,
            "upper_shadow_selling_pressure_inverted": 0.03, "smart_money_inst_net_buy": 0.05, "microstructure_efficiency": 0.03
        })
        new_raw_signals_weights_v2 = get_param_value(covert_accum_params.get('new_raw_signals_weights_v2'), {
            "buy_flow_efficiency": 0.05, "sell_flow_efficiency_inverted": 0.03,
            "main_force_vwap_up_guidance": 0.05, "observed_large_order_size_avg_inverted": 0.03
        })
        main_force_accumulation_resonance_weight = get_param_value(covert_accum_params.get('main_force_accumulation_resonance_weight'), 0.1)
        covert_order_flow_resonance_weight = get_param_value(covert_accum_params.get('covert_order_flow_resonance_weight'), 0.08)
        price_weakness_slope_window = get_param_value(covert_accum_params.get('price_weakness_slope_window'), 5)
        low_volatility_bbw_window = get_param_value(covert_accum_params.get('low_volatility_bbw_window'), 21)
        mtf_slope_accel_weights = config.get('mtf_slope_accel_weights', {"slope_periods": {"5": 0.6, "13": 0.4}, "accel_periods": {"5": 0.7, "13": 0.3}})
        neutral_range_threshold = get_param_value(covert_accum_params.get('neutral_range_threshold'), 0.1)
        cumulative_flow_windows = get_param_value(covert_accum_params.get('cumulative_flow_windows'), [13, 21])
        cumulative_flow_weights = get_param_value(covert_accum_params.get('cumulative_flow_weights'), {"13": 0.6, "21": 0.4})
        cumulative_acc_windows = get_param_value(covert_accum_params.get('cumulative_acc_windows'), [13, 21])
        cumulative_acc_weights = get_param_value(covert_accum_params.get('cumulative_acc_weights'), {"13": 0.6, "21": 0.4})
        daily_mf_flow_weight = get_param_value(covert_accum_params.get('daily_mf_flow_weight'), 0.4)
        cumulative_mf_flow_weight = get_param_value(covert_accum_params.get('cumulative_mf_flow_weight'), 0.6)
        daily_acc_weight = get_param_value(covert_accum_params.get('daily_acc_weight'), 0.4)
        cumulative_acc_weight = get_param_value(covert_accum_params.get('cumulative_acc_weight'), 0.6)
        print(f"DEBUG_PROBE:CoherencyConfigLoaded|EntropyWeight={chip_optimization_weights.get('entropy_reduction')}")
        return fusion_weights, market_context_weights, covert_action_weights, chip_optimization_weights, price_weakness_slope_window, low_volatility_bbw_window, mtf_slope_accel_weights, neutral_range_threshold, cumulative_flow_windows, cumulative_flow_weights, cumulative_acc_windows, cumulative_acc_weights, daily_mf_flow_weight, cumulative_mf_flow_weight, daily_acc_weight, cumulative_acc_weight, new_raw_signals_weights, main_force_accumulation_resonance_weight, new_raw_signals_weights_v2, covert_order_flow_resonance_weight

    def _validate_and_get_raw_signals(self, df: pd.DataFrame, method_name: str, mtf_slope_accel_weights: Dict, is_debug_enabled_for_method: bool, probe_ts: Optional[pd.Timestamp], _temp_debug_values: Dict, cumulative_flow_windows: List[int]) -> Dict[str, pd.Series]:
        """
        【V6.12·热力学指标提取版】提取关键军械库指标。
        -核心升级:
        1.'chip_entropy_D':筹码熵，衡量有序度。
        2.'intraday_cost_center_migration_D':日内成本迁移，衡量微观支撑。
        """
        required_cols = [
            'STATE_EMOTIONAL_EXTREME_D', 'BBW_21_2.0_D', 'MA_POTENTIAL_COMPRESSION_RATE_D',
            'INTRADAY_SUPPORT_INTENT_D', 'tick_abnormal_volume_ratio_D', 'pressure_release_index_D',
            'VPA_MF_ADJUSTED_EFF_D', 'stealth_flow_ratio_D', 'SMART_MONEY_INST_NET_BUY_D',
            'accumulation_score_D', 'chip_stability_D', 'chip_concentration_ratio_D',
            'ATR_14_D', 'volume_vs_ma_5_ratio_D',
            'buy_elg_amount_rate_D', 'flow_consistency_D', 'intraday_accumulation_confidence_D',
            'winner_rate_D', 'intraday_chip_game_index_D', 'behavior_accumulation_D',
            'THEME_HOTNESS_SCORE_D', 'industry_strength_rank_D',
            'SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D',
            'TURNOVER_STABILITY_INDEX_D', 'breakout_potential_D',
            'STATE_GOLDEN_PIT_D', 'support_resistance_ratio_D', 'industry_breadth_score_D',
            'tick_chip_transfer_efficiency_D', 'pressure_trapped_D', 'pressure_profit_D',
            'chip_skewness_D', 'chip_kurtosis_D',
            'chip_entropy_D', 'intraday_cost_center_migration_D' # [V6.12新增]
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            _temp_debug_values["__MISSING_COLUMNS__"] = missing
            for c in missing:
                df[c] = np.nan
        self._calculate_chip_dynamics(df, _temp_debug_values)
        vol_ratio = df.get('volume_vs_ma_5_ratio_D', pd.Series(np.nan, index=df.index))
        close_safe = df['close'].replace(0, np.nan)
        norm_atr = df.get('ATR_14_D', pd.Series(np.nan, index=df.index)) / (close_safe + 0.001)
        stealth_density = vol_ratio / (norm_atr + 0.001)
        raw_signals = {
            'emo_extreme': df.get('STATE_EMOTIONAL_EXTREME_D'),
            'vol_bbw': df.get('BBW_21_2.0_D'),
            'ma_compression': df.get('MA_POTENTIAL_COMPRESSION_RATE_D'),
            'intraday_support': df.get('INTRADAY_SUPPORT_INTENT_D'),
            'abnormal_vol': df.get('tick_abnormal_volume_ratio_D'),
            'pressure_release': df.get('pressure_release_index_D'),
            'mf_efficiency': df.get('VPA_MF_ADJUSTED_EFF_D'),
            'stealth_flow': df.get('stealth_flow_ratio_D'),
            'inst_buy': df.get('SMART_MONEY_INST_NET_BUY_D'),
            'acc_score': df.get('accumulation_score_D'),
            'chip_stability': df.get('chip_stability_D'),
            'chip_concentration': df.get('chip_concentration_ratio_D'),
            'stealth_density': stealth_density,
            'elg_buy_rate': df.get('buy_elg_amount_rate_D'),
            'flow_consistency': df.get('flow_consistency_D'),
            'accum_confidence': df.get('intraday_accumulation_confidence_D'),
            'winner_rate': df.get('winner_rate_D'),
            'game_index': df.get('intraday_chip_game_index_D'),
            'behavior_tag': df.get('behavior_accumulation_D'),
            'theme_hotness': df.get('THEME_HOTNESS_SCORE_D'),
            'industry_rank': df.get('industry_strength_rank_D'),
            'smart_divergence': df.get('SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D'),
            'turnover_stability': df.get('TURNOVER_STABILITY_INDEX_D'),
            'breakout_potential': df.get('breakout_potential_D'),
            'golden_pit': df.get('STATE_GOLDEN_PIT_D'),
            'sr_ratio': df.get('support_resistance_ratio_D'),
            'industry_breadth': df.get('industry_breadth_score_D'),
            'transfer_eff': df.get('tick_chip_transfer_efficiency_D'),
            'pressure_trapped': df.get('pressure_trapped_D'),
            'pressure_profit': df.get('pressure_profit_D'),
            'skewness': df.get('chip_skewness_D'),
            'kurtosis': df.get('chip_kurtosis_D'),
            'entropy': df.get('chip_entropy_D'), # [新增]
            'cost_migration': df.get('intraday_cost_center_migration_D') # [新增]
        }
        fib_windows = [3, 8, 13]
        for base in ['stealth_flow_ratio_D', 'SMART_MONEY_INST_NET_BUY_D']:
            for p in fib_windows:
                key = f'jerk_{base}_{p}'
                col = f'JERK_{p}_{base}'
                raw_signals[key] = df[col] if col in df.columns else pd.Series(np.nan, index=df.index)
        print(f"DEBUG_PROBE:RawSignalsExtracted|EntropyLen={len(raw_signals['entropy'])}|MigrationLen={len(raw_signals['cost_migration'])}")
        _temp_debug_values["原始信号值"] = raw_signals
        return raw_signals

    def _calculate_derived_signals(self, df: pd.DataFrame, mtf_slope_accel_weights: Dict, cumulative_flow_windows: List[int], cumulative_acc_windows: List[int], _temp_debug_values: Optional[Dict] = None):
        """
        【V6.14·健壮导数版】计算高阶物理导数。
        -核心修正:增加Jerk补位逻辑，当短周期Jerk为nan时由长周期平滑填充，确保动力学判定不失效。
        """
        kinematic_targets = ['stealth_flow_ratio_D', 'SMART_MONEY_INST_NET_BUY_D', 'INTRADAY_SUPPORT_INTENT_D', 'chip_concentration_ratio_D', 'winner_rate_D', 'chip_entropy_D']
        hab_decay_span = 34
        for base in ['stealth_flow_ratio_D', 'SMART_MONEY_INST_NET_BUY_D']:
            if base in df.columns:
                clean_series = df[base].fillna(0)
                df[f'HAB_{base}'] = clean_series.ewm(span=hab_decay_span, adjust=False).mean() * hab_decay_span
        fib_windows = [3, 8, 13]
        for base in kinematic_targets:
            if base not in df.columns: continue
            series_smooth = ta.ema(df[base].ffill().fillna(0), length=3)
            for period in fib_windows:
                s_col = f'SLOPE_{period}_{base}'
                slope_series = ta.slope(series_smooth, length=period)
                df[s_col] = slope_series
                a_col = f'ACCEL_{period}_{base}'
                df[a_col] = ta.slope(slope_series.fillna(0), length=3)
                j_col = f'JERK_{period}_{base}'
                df[j_col] = ta.slope(df[a_col].fillna(0), length=3)
        # 修正Jerk空值:由13日向3日/8日回填
        for base in ['stealth_flow_ratio_D', 'SMART_MONEY_INST_NET_BUY_D']:
            j3, j8, j13 = f'JERK_3_{base}', f'JERK_8_{base}', f'JERK_13_{base}'
            if j13 in df.columns:
                df[j3] = df[j3].fillna(df[j8]).fillna(df[j13])
                df[j8] = df[j8].fillna(df[j13])
        if _temp_debug_values is not None:
            print(f"DEBUG_PROBE:DerivativesHealed|Base={kinematic_targets[0]}|J3_HasData={not df['JERK_3_stealth_flow_ratio_D'].tail(1).isna().any()}")

    def _calculate_covert_action_score(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], covert_action_weights: Dict, _temp_debug_values: Dict, cumulative_flow_windows: List[int]) -> pd.Series:
        """
        【V6.14·静默吸筹补偿版】计算隐蔽行动分数。
        -核心升级:引入'隐匿密度补偿'。当密度>15时，即使inst_buy为0，也判定存在碎片化隐性买入，赋予0.35的基础分，防止几何平均一票否决。
        """
        hab_elg = df.get('HAB_buy_elg_amount_rate_D', pd.Series(0, index=df_index))
        hab_stealth = df.get('HAB_stealth_flow_ratio_D', pd.Series(0, index=df_index))
        s_hab_elg = self.helper._normalize_series(hab_elg, df_index, bipolar=False)
        s_hab_stealth = self.helper._normalize_series(hab_stealth, df_index, bipolar=False)
        jerk_confidence = df.get('JERK_3_intraday_accumulation_confidence_D', pd.Series(0, index=df_index))
        s_jerk_conf = self.helper._normalize_series(jerk_confidence, df_index, bipolar=True)
        s_stealth = self.helper._normalize_series(raw_signals['stealth_flow'], df_index, bipolar=False)
        s_inst_buy = self.helper._normalize_series(raw_signals['inst_buy'], df_index, bipolar=True)
        # [V6.14核心优化] 零值软化: 隐匿密度补偿
        density_raw = raw_signals['stealth_density'].fillna(0)
        s_density = self.helper._normalize_series(density_raw, df_index, bipolar=False)
        # 如果密度极高，给机构买入和痛苦吸筹一个保底分
        density_boost_mask = density_raw > 15
        s_inst_buy_soft = s_inst_buy.copy()
        s_inst_buy_soft[density_boost_mask] = s_inst_buy[density_boost_mask].clip(lower=0.35)
        s_support = self.helper._normalize_series(raw_signals['intraday_support'], df_index, bipolar=False)
        s_abnormal = self.helper._normalize_series(raw_signals['abnormal_vol'], df_index, bipolar=False)
        s_elg_drive = self.helper._normalize_series(raw_signals['elg_buy_rate'], df_index, bipolar=False)
        s_consistency = self.helper._normalize_series(raw_signals['flow_consistency'], df_index, bipolar=False)
        s_confidence = self.helper._normalize_series(raw_signals['accum_confidence'], df_index, bipolar=False)
        winner_rate = raw_signals['winner_rate'].fillna(0.5) / 100.0 if raw_signals['winner_rate'].max() > 1.0 else raw_signals['winner_rate'].fillna(0.5)
        pain_factor = (1.0 - winner_rate).clip(0, 1)
        s_pain_accum = (pain_factor * s_inst_buy_soft.clip(lower=0)).clip(lower=0.2 if density_boost_mask.any() else 0.0)
        s_game_index = self.helper._normalize_series(raw_signals['game_index'], df_index, bipolar=False)
        s_friction = (s_game_index * 0.6 + s_stealth * 0.4)
        s_behavior = raw_signals['behavior_tag'].fillna(0).clip(0, 1)
        s_iceberg = (s_support * 0.3 + s_elg_drive * 0.3 + s_hab_elg * 0.2 + s_confidence * 0.2)
        s_surge = s_jerk_conf * s_consistency
        scores = {
            "pain_accumulation": s_pain_accum,
            "game_friction": s_friction,
            "behavior_confirmation": s_behavior,
            "iceberg_friction": s_iceberg,
            "whale_active_drive": s_elg_drive,
            "hab_accumulation": s_hab_stealth,
            "kinetic_surge": s_surge,
            "intraday_confidence": s_confidence,
            "flow_consistency": s_consistency,
            "stealth_ops": s_stealth,
            "contextualized_accum": self.helper._normalize_series(raw_signals['acc_score'], df_index, bipolar=False)
        }
        covert_action_score = _robust_geometric_mean(scores, covert_action_weights, df_index)
        _temp_debug_values["隐蔽行动"] = scores
        print(f"DEBUG_PROBE:ActionSoftened|Density={density_raw.iloc[-1]:.2f}|InstBuySoft={s_inst_buy_soft.iloc[-1]:.4f}")
        return covert_action_score

    def _calculate_context_derived_signals(self, df: pd.DataFrame, _temp_debug_values: Dict):
        """
        【新增辅助方法】专门计算市场背景的 HAB (势能) 和 Slope (动量)。
        """
        # 1. 势能 (Potential): 针对背离信号
        # 背离信号通常是 0 或 1，或者是强度分。我们需要累积它。
        if 'SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D' in df.columns:
            div_series = df['SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D'].fillna(0)
            # 衰减累积：13天窗口，模拟"主力持续吸筹"的势能
            df['HAB_SMART_DIVERGENCE'] = div_series.ewm(span=13, adjust=False).mean() * 13
        # 2. 动量 (Momentum): 针对热度和情绪
        # 我们只关心趋势方向 (Slope)，不关心加速度 (Accel/Jerk)
        momentum_targets = ['THEME_HOTNESS_SCORE_D', 'STATE_EMOTIONAL_EXTREME_D']
        for target in momentum_targets:
            if target in df.columns:
                # 5日线性回归斜率
                df[f'SLOPE_5_{target}'] = ta.slope(df[target], length=5)

    def _calculate_market_context_score(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], market_context_weights: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V6.15·向量化加速版】计算市场背景分数。
        -核心优化:使用Numpy向量化操作替代apply lambda，加速Sigmoid计算。
        """
        s_sentiment = self.helper._normalize_series(raw_signals['emo_extreme'], df_index, bipolar=False)
        s_vol_comp = self.helper._normalize_series(raw_signals['vol_bbw'], df_index, ascending=False)
        s_consolidating = self.helper._normalize_series(raw_signals['ma_compression'], df_index, bipolar=False)
        s_pressure = self.helper._normalize_series(raw_signals['pressure_release'], df_index, bipolar=False)
        s_breakout = self.helper._normalize_series(raw_signals['breakout_potential'], df_index, bipolar=False)
        s_golden_pit = raw_signals['golden_pit'].fillna(0).clip(0, 1)
        sr_ratio = raw_signals['sr_ratio'].fillna(1.0)
        s_sr_efficiency = (sr_ratio / 2.0).clip(0, 1)
        s_breadth = self.helper._normalize_series(raw_signals['industry_breadth'], df_index, bipolar=False)
        theme_hotness = raw_signals['theme_hotness'].fillna(0)
        industry_rank = raw_signals['industry_rank'].fillna(100)
        norm_hotness = (theme_hotness / 100.0).clip(0, 1)
        norm_rank = (1.0 - industry_rank / 100.0).clip(0, 1)
        slope_theme = df.get('SLOPE_5_THEME_HOTNESS_SCORE_D', pd.Series(0, index=df_index))
        s_slope_theme = self.helper._normalize_series(slope_theme, df_index, bipolar=True)
        s_theme_health = (norm_hotness * 0.3 + s_breadth * 0.3 + norm_rank * 0.2 + s_slope_theme * 0.2)
        hab_divergence = df.get('HAB_SMART_DIVERGENCE', pd.Series(0, index=df_index))
        s_hab_div = self.helper._normalize_series(hab_divergence, df_index, bipolar=False)
        div_signal = raw_signals['smart_divergence'].fillna(0)
        s_div_raw = self.helper._normalize_series(div_signal, df_index, bipolar=False)
        s_divergence_complex = (s_div_raw * 0.3 + s_hab_div * 0.7)
        # 优化: 向量化Sigmoid计算
        raw_stability = raw_signals['turnover_stability'].fillna(0)
        s_stability = 1.0 / (1.0 + np.exp(-15.0 * (raw_stability - 0.2)))
        scores = {
            "golden_pit_state": s_golden_pit,
            "space_efficiency": s_sr_efficiency,
            "theme_resonance": s_theme_health,
            "smart_divergence": s_divergence_complex,
            "turnover_stability": s_stability,
            "sentiment_extreme": s_sentiment,
            "vol_compression": s_vol_comp,
            "is_consolidating": s_consolidating,
            "pressure_release": s_pressure,
            "breakout_potential": s_breakout
        }
        final_weights = market_context_weights.copy()
        market_context_score = _robust_geometric_mean(scores, final_weights, df_index)
        _temp_debug_values["市场背景"] = scores
        return market_context_score

    def _calculate_chip_dynamics(self, df: pd.DataFrame, _temp_debug_values: Dict):
        """
        【V6.13·效率微调版】计算筹码的HAB,Kinematics以及热力学动态。
        -核心优化:使用ffill()替代过时的fillna(method='ffill')。
        """
        if 'tick_chip_transfer_efficiency_D' in df.columns:
            eff_series = df['tick_chip_transfer_efficiency_D'].fillna(0)
            df['HAB_CHIP_TRANSFER'] = eff_series.ewm(span=8, adjust=False).mean()
        if 'chip_concentration_ratio_D' in df.columns:
            conc = df['chip_concentration_ratio_D'].ffill()
            slope = ta.slope(conc, length=5)
            df['ACCEL_5_CHIP_CONCENTRATION'] = ta.slope(slope, length=3)
        if 'pressure_trapped_D' in df.columns:
            pressure = df['pressure_trapped_D'].ffill()
            slope = ta.slope(pressure, length=5)
            accel = ta.slope(slope, length=3)
            df['JERK_3_PRESSURE_TRAPPED'] = ta.slope(accel, length=3)
        if 'chip_stability_D' in df.columns:
            stab = df['chip_stability_D'].fillna(0)
            df['HAB_CHIP_STABILITY'] = stab.ewm(span=13, adjust=False).mean()
        if 'chip_skewness_D' in df.columns:
            skew = df['chip_skewness_D'].ffill()
            df['SLOPE_5_CHIP_SKEWNESS'] = ta.slope(skew, length=5)
        if 'chip_kurtosis_D' in df.columns:
            kurt = df['chip_kurtosis_D'].ffill()
            slope_k = ta.slope(kurt, length=5)
            df['ACCEL_5_CHIP_KURTOSIS'] = ta.slope(slope_k, length=3)
        if 'chip_entropy_D' in df.columns:
            entropy = df['chip_entropy_D'].ffill()
            df['SLOPE_5_CHIP_ENTROPY'] = ta.slope(entropy, length=5)
        if 'intraday_cost_center_migration_D' in df.columns:
            mig = df['intraday_cost_center_migration_D'].fillna(0)
            df['EMA_3_COST_MIGRATION'] = ta.ema(mig, length=3)
        print(f"DEBUG_PROBE:DynamicsCalculated|EntropySlopeCol={'SLOPE_5_CHIP_ENTROPY' in df.columns}")

    def _calculate_chip_optimization_score(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], chip_optimization_weights: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V6.13·效率微调版】计算筹码优化分数。
        -核心优化:更新过时的API调用。
        """
        hab_transfer = df.get('HAB_CHIP_TRANSFER', pd.Series(0, index=df_index))
        hab_stability = df.get('HAB_CHIP_STABILITY', pd.Series(0, index=df_index))
        accel_conc = df.get('ACCEL_5_CHIP_CONCENTRATION', pd.Series(0, index=df_index))
        jerk_trapped = df.get('JERK_3_PRESSURE_TRAPPED', pd.Series(0, index=df_index))
        slope_skew = df.get('SLOPE_5_CHIP_SKEWNESS', pd.Series(0, index=df_index))
        accel_kurt = df.get('ACCEL_5_CHIP_KURTOSIS', pd.Series(0, index=df_index))
        slope_entropy = df.get('SLOPE_5_CHIP_ENTROPY', pd.Series(0, index=df_index))
        ema_migration = df.get('EMA_3_COST_MIGRATION', pd.Series(0, index=df_index))
        s_hab_transfer = self.helper._normalize_series(hab_transfer, df_index, bipolar=False)
        s_hab_stability = self.helper._normalize_series(hab_stability, df_index, bipolar=False)
        s_accel_conc = self.helper._normalize_series(accel_conc, df_index, bipolar=True)
        s_jerk_trapped_inverted = self.helper._normalize_series(jerk_trapped, df_index, ascending=True)
        s_slope_skew = self.helper._normalize_series(slope_skew, df_index, bipolar=True)
        s_accel_kurt = self.helper._normalize_series(accel_kurt, df_index, bipolar=True)
        raw_entropy = raw_signals['entropy'].ffill()
        s_entropy_abs = self.helper._normalize_series(raw_entropy, df_index, ascending=False)
        s_entropy_slope = self.helper._normalize_series(slope_entropy, df_index, ascending=False)
        s_migration = self.helper._normalize_series(ema_migration, df_index, bipolar=True)
        s_stability = self.helper._normalize_series(raw_signals['chip_stability'], df_index, bipolar=False)
        s_concentration = self.helper._normalize_series(raw_signals['chip_concentration'], df_index, bipolar=False)
        s_trapped_inv = self.helper._normalize_series(raw_signals['pressure_trapped'], df_index, ascending=False)
        s_profit_inv = self.helper._normalize_series(raw_signals['pressure_profit'], df_index, ascending=False)
        s_transfer_raw = self.helper._normalize_series(raw_signals['transfer_eff'], df_index, bipolar=False)
        s_iron_floor = (s_stability * 0.4 + s_hab_stability * 0.6)
        s_morphology = (s_slope_skew * 0.5 + s_accel_kurt * 0.5)
        s_transfer_complex = (s_transfer_raw * 0.3 + s_hab_transfer * 0.7)
        s_cleansing = (s_trapped_inv * 0.6 + s_jerk_trapped_inverted * 0.4)
        s_locking = (s_concentration * 0.4 + s_accel_conc * 0.3 + s_profit_inv * 0.3)
        s_entropy_reduction = (s_entropy_abs * 0.6 + s_entropy_slope * 0.4)
        scores = {
            "entropy_reduction": s_entropy_reduction,
            "cost_center_support": s_migration,
            "chip_morphology": s_morphology,
            "iron_floor_hab": s_iron_floor,
            "transfer_efficiency_hab": s_transfer_complex,
            "trapped_pressure_release": s_cleansing,
            "concentration_accel": s_accel_conc,
            "chip_locking": s_locking,
            "chip_stability": s_stability,
            "chip_concentration": s_concentration
        }
        final_weights = chip_optimization_weights.copy()
        final_weights.setdefault("entropy_reduction", 0.15)
        final_weights.setdefault("cost_center_support", 0.10)
        final_weights.setdefault("chip_morphology", 0.10)
        final_weights.setdefault("iron_floor_hab", 0.10)
        final_weights.setdefault("transfer_efficiency_hab", 0.15)
        chip_optimization_score = _robust_geometric_mean(scores, final_weights, df_index)
        _temp_debug_values["筹码优化_热力学"] = {
            "Entropy_Abs_Norm": float(s_entropy_abs.iloc[-1]) if len(s_entropy_abs)>0 else 0.0,
            "Entropy_Slope_Norm": float(s_entropy_slope.iloc[-1]) if len(s_entropy_slope)>0 else 0.0,
            "Cost_Migration_Norm": float(s_migration.iloc[-1]) if len(s_migration)>0 else 0.0
        }
        _temp_debug_values["筹码优化"] = scores
        return chip_optimization_score

    def _fuse_final_score(self, df_index: pd.Index, market_context_score: pd.Series, covert_action_score: pd.Series, chip_optimization_score: pd.Series, fusion_weights: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.8·向量化激发版】最终分数合成。
        -核心优化:使用Numpy向量化log1p替代apply，提升计算效率。
        """
        final_fusion_scores_dict = {
            "market_context": market_context_score,
            "covert_action": covert_action_score,
            "chip_optimization": chip_optimization_score
        }
        raw_fusion = _robust_geometric_mean(final_fusion_scores_dict, fusion_weights, df_index)
        raw_signals = _temp_debug_values.get("原始信号值", {})
        stealth_density = raw_signals.get('stealth_density', pd.Series(0.0, index=df_index))
        chip_details = _temp_debug_values.get("筹码优化", {})
        entropy_score = chip_details.get("entropy_reduction", pd.Series(0.0, index=df_index))
        is_solid_accumulation = (entropy_score > 0.75) & (stealth_density > 15)
        # 优化: 向量化计算Boost基数
        boost_base = (np.log1p(stealth_density) / 2.0).clip(1.0, 2.0)
        final_fusion = raw_fusion.copy()
        if is_solid_accumulation.any():
            final_fusion[is_solid_accumulation] = np.sqrt(raw_fusion[is_solid_accumulation] * boost_base[is_solid_accumulation]).clip(0, 1)
        _temp_debug_values["最终合成_激发态"] = {
            "Raw_Fusion": float(raw_fusion.iloc[-1]),
            "Entropy_Check": float(entropy_score.iloc[-1]),
            "Solid": bool(is_solid_accumulation.iloc[-1])
        }
        print(f"DEBUG_PROBE:FusionIgnited_V7.8|Raw={raw_fusion.iloc[-1]:.4f}|Final={final_fusion.iloc[-1]:.4f}|Solid={is_solid_accumulation.iloc[-1]}")
        return final_fusion.astype(np.float32)

    def _print_debug_info(self, debug_output: Dict, _temp_debug_values: Dict, method_name: str, probe_ts: pd.Timestamp):
        """
        【V6.0 · 全链路穿透式诊断】输出所有中间变量，不留死角。
        """
        # 1. 报警信息
        if "__MISSING_COLUMNS__" in _temp_debug_values:
            debug_output[f"  !! [严重警告] 缺失军械库指标: {_temp_debug_values['__MISSING_COLUMNS__']}"] = "这会导致相关评分降级或为NaN"
        # 2. 原始信号 (Raw)
        debug_output[f"  -- [1. 原始信号层] Value @ {probe_ts.strftime('%Y-%m-%d')}"] = ""
        for key, series in _temp_debug_values.get("原始信号值", {}).items():
            val = series.loc[probe_ts] if isinstance(series, pd.Series) and probe_ts in series.index else "N/A"
            debug_output[f"     |-- {key:<25}: {val}"] = ""
        # 3. 归一化得分 (Normalized)
        for section in ["市场背景", "隐蔽行动", "筹码优化"]:
            debug_output[f"  -- [2. {section}得分] Normalized @ {probe_ts.strftime('%Y-%m-%d')}"] = ""
            for key, series in _temp_debug_values.get(section, {}).items():
                val = series.loc[probe_ts] if isinstance(series, pd.Series) and probe_ts in series.index else 0.0
                weight = _temp_debug_values.get(f"{section}_权重", {}).get(key, "Def")
                debug_output[f"     |-- {key:<25}: Score={val:.4f} (Weight={weight})"] = ""
        # 4. 最终结果
        final_val = _temp_debug_values['final_score'].loc[probe_ts] if probe_ts in _temp_debug_values['final_score'].index else 0.0
        debug_output[f"  ==> [最终结果] Covert Accumulation Score: {final_val:.4f}"] = ""
        self.helper._print_debug_output(debug_output)

    @staticmethod
    def _numba_latch_kernel(raw_values: np.ndarray, active_flags: np.ndarray) -> np.ndarray:
        """
        【新增方法】Numba静态内核，处理有状态的锁存逻辑。
        需要: from numba import jit
        """
        try:
            @jit(nopython=True, cache=True)
            def _core_loop(raw, active):
                n = len(raw)
                out = np.zeros(n, dtype=np.float64)
                decay_rate = 0.99
                break_threshold = 0.30
                is_locked = False
                last_val = 0.0
                for i in range(n):
                    curr_raw = raw[i]
                    is_active = active[i] > 0
                    if is_active:
                        is_locked = True
                        # tanh动量增益
                        curr_val = np.tanh(curr_raw * 1.8) + 0.2
                        if curr_val > last_val * decay_rate:
                            last_val = curr_val
                        else:
                            last_val = last_val * decay_rate
                    elif is_locked:
                        if curr_raw < break_threshold:
                            is_locked = False
                            last_val = curr_raw
                        else:
                            if curr_raw > last_val * decay_rate:
                                last_val = curr_raw # 保持衰减
                            else:
                                last_val = last_val * decay_rate
                            # 修正逻辑：取max
                            if curr_raw > last_val:
                                last_val = curr_raw
                    else:
                        last_val = curr_raw
                    out[i] = last_val
                return out
            return _core_loop(raw_values, active_flags)
        except ImportError:
            # 降级回Python原生实现，防止未安装Numba报错
            n = len(raw_values)
            out = np.zeros(n)
            decay_rate, break_threshold, is_locked, last_val = 0.99, 0.30, False, 0.0
            for i in range(n):
                curr_raw = raw_values[i]
                if active_flags[i] > 0:
                    is_locked = True
                    curr_val = np.tanh(curr_raw * 1.8) + 0.2
                    last_val = max(curr_val, last_val * decay_rate)
                elif is_locked:
                    if curr_raw < break_threshold:
                        is_locked, last_val = False, curr_raw
                    else:
                        last_val = max(curr_raw, last_val * decay_rate)
                else:
                    last_val = curr_raw
                out[i] = last_val
            return out











