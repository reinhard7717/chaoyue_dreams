# strategies\trend_following\intelligence\process\calculate_winner_conviction_decay.py
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

class CalculateWinnerConvictionDecay:
    """
    【V4.1 · 全息动态审判版】“赢家信念衰减”专属计算引擎
    PROCESS_META_WINNER_CONVICTION_DECAY
    """
    def __init__(self, strategy_instance, helper_instance: ProcessIntelligenceHelper):
        self.strategy = strategy_instance
        self.helper = helper_instance

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V7.1.1 · 混沌共振修复版】主计算入口逻辑更新
        - 说明：修正共振因子与隐秘吸筹增益的调用顺序。
        - 版本号：V7.1.1
        """
        method_name = "calculate_winner_conviction_decay"
        is_debug_enabled = get_param_value(self.helper.debug_params.get('enabled'), False)
        probe_ts = None
        if is_debug_enabled and self.helper.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.helper.probe_dates]
            for date in reversed(df.index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        df_index = df.index
        params_dict, all_required_signals = self._get_decay_params_and_signals(config, method_name)
        if not self.helper._validate_required_signals(df, all_required_signals, method_name): return pd.Series(dtype=np.float32)
        _temp_debug_values = {"conviction_dynamics": {}}
        raw_signals = self._get_raw_signals(df, df_index, params_dict, method_name)
        # 依次生成多维组件
        conviction_score = self._calculate_conviction_strength(df, df_index, raw_signals, params_dict, method_name, _temp_debug_values)
        resilience_score = self._calculate_pressure_resilience(df, df_index, raw_signals, params_dict, method_name, _temp_debug_values)
        deception_filter = self._calculate_deception_filter(df, df_index, raw_signals, params_dict, method_name, _temp_debug_values)
        context_modulator = self._calculate_contextual_modulator(df, df_index, raw_signals, params_dict, method_name, _temp_debug_values, is_debug_enabled, probe_ts)
        # 引入对冲与协同
        stealth_bonus = self._calculate_stealth_accumulation_bonus(df_index, raw_signals, _temp_debug_values)
        synergy_factor = self._calculate_synergy_factor(conviction_score, resilience_score, _temp_debug_values)
        # 最终审判
        final_score = self._perform_final_fusion(df_index, conviction_score, resilience_score, deception_filter, stealth_bonus, params_dict, _temp_debug_values)
        ewd_factor = self._calculate_ewd_factor(conviction_score, resilience_score, context_modulator, _temp_debug_values)
        latched_score = self._apply_latch_logic(df_index, final_score, ewd_factor, params_dict, _temp_debug_values)
        if is_debug_enabled and probe_ts: self._execute_intelligence_probe(method_name, probe_ts, _temp_debug_values, latched_score)
        return latched_score.astype(np.float32)

    def _execute_intelligence_probe(self, method_name: str, probe_ts: pd.Timestamp, _temp_debug_values: Dict, final_score: pd.Series):
        """
        【V7.1 · 全息时空审判版】全息采样探针
        - 版本号：V7.1.0
        """
        print(f"\n{'='*40} [V7.1 HOLOGRAPHIC PROBE: {probe_ts.strftime('%Y-%m-%d')}] {'='*40}")
        conv = _temp_debug_values.get("conviction_dynamics", {})
        print(f"--- [信念核心驱动] ---")
        print(f"  > 时空同步衰竭: {conv.get('sync_decay').loc[probe_ts]:.4f} | 筹码连锁风险: {conv.get('chain_risk').loc[probe_ts]:.4f}")
        vac = _temp_debug_values.get("vacuum_meltdown_analysis", {})
        print(f"--- [流动性真空监测] ---")
        print(f"  > 支撑比率: {vac.get('support_ratio').loc[probe_ts]:.4f} | 熔断风险: {vac.get('vacuum_risk').loc[probe_ts]:.4f}")
        latch = _temp_debug_values.get("latch_state", {})
        print(f"--- [最终决策输出] ---")
        print(f"  > 锁存器触发: {latch.get('latch_trigger').loc[probe_ts]} | FINAL_CONVICTION_DECAY: {final_score.loc[probe_ts]:.4f}")
        print(f"{'='*105}\n")

    def _get_decay_params_and_signals(self, config: Dict, method_name: str) -> Tuple[Dict, List[str]]:
        """
        【V7.1.1 · 混沌共振修复版】重构信号权重与依赖
        - 逻辑：整合混沌共振、失速冲击与机构撤退全链路权重分配。
        - 说明：明确引入 chaotic_collapse_resonance 权重分量。
        - 版本号：V7.1.1
        """
        decay_params = get_param_value(config.get('winner_conviction_decay_params'), {})
        fibo_periods = ["5", "13", "21", "34"]
        belief_decay_weights = {
            "mid_long_sync_decay": 0.15, # 中长周期同步性 [cite: 3]
            "chain_collapse_resonance": 0.15, # 筹码连锁崩塌 [cite: 3]
            "chaotic_collapse_resonance": 0.15, # 混沌共振 
            "institutional_vacuum_meltdown": 0.10, # 机构买盘真空 
            "institutional_stalling_jerk": 0.10, # 机构失速冲击 
            "macro_sector_slippage": 0.10, # 行业位阶滑坡 [cite: 2]
            "parabolic_sprint_risk": 0.10, # 抛物线末端冲刺 
            "regime_switching_risk": 0.15 # 市场状态切换 
        }
        required_df_columns = [
            'mid_long_sync_D', 'volatility_adjusted_concentration_D', 'STATE_TRENDING_STAGE_D', # 
            'SMART_MONEY_INST_NET_BUY_D', 'tick_large_order_net_D', 'PRICE_ENTROPY_D', # 
            'MA_COHERENCE_RESONANCE_D', 'PRICE_FRACTAL_DIM_D', 'STATE_MARKET_LEADER_D', # 
            'SMART_MONEY_SYNERGY_BUY_D', 'industry_leader_score_D', 'THEME_HOTNESS_SCORE_D', # [cite: 1, 2]
            'industry_rank_slope_D', 'STATE_ROUNDING_BOTTOM_D', 'breakout_potential_D', # [cite: 1, 2]
            'STATE_GOLDEN_PIT_D', 'breakout_quality_score_D', 'VPA_ACCELERATION_5D', # [cite: 1, 2]
            'VPA_EFFICIENCY_D', 'MA_RUBBER_BAND_EXTENSION_D', 'STATE_PARABOLIC_WARNING_D', # 
            'SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D', 'SMART_MONEY_HM_NET_BUY_D', # 
            'STATE_EMOTIONAL_EXTREME_D', 'tick_abnormal_volume_ratio_D', 'tick_chip_transfer_efficiency_D', # 
            'intraday_distribution_confidence_D', 'anomaly_intensity_D', 'stealth_flow_ratio_D', # [cite: 3]
            'uptrend_strength_D', 'market_sentiment_score_D', 'pressure_profit_D', 'net_amount_ratio_D' # [cite: 3]
        ]
        kinetic_targets = ['mid_long_sync_D', 'SMART_MONEY_INST_NET_BUY_D', 'PRICE_FRACTAL_DIM_D', 'SMART_MONEY_SYNERGY_BUY_D'] # 
        for target in kinetic_targets:
            for p in ["5"]:
                required_df_columns.extend([f'SLOPE_{p}_{target}', f'ACCEL_{p}_{target}', f'JERK_{p}_{target}'])
        params_dict = {
            'decay_params': decay_params, 'fibo_periods': fibo_periods, 'belief_decay_weights': belief_decay_weights,
            'hab_settings': {"short": 13, "medium": 21, "long": 34},
            'latch_params': {"window": 5, "hit_count": 3, "high_score_threshold": 0.618, "core_threshold": 0.382, "momentum_protection_factor": 0.95, "entropy_threshold": 0.75},
            'vacuum_threshold': 0.1, 'final_exponent': 18.0
        }
        return params_dict, list(set(required_df_columns))

    def _get_raw_signals(self, df: pd.DataFrame, df_index: pd.Index, params_dict: Dict, method_name: str) -> Dict[str, pd.Series]:
        """
        【V7.2.5 · 数据全景补全版】同步军械库全量指标并消除 Key 缺失风险
        - 逻辑：遍历所有子计算模块需求，补全包括情绪、压力、强度在内的 30+ 指标。
        - 版本号：V7.2.5
        """
        raw_signals = {}
        hab_cfg = params_dict['hab_settings']
        targets = [
            'mid_long_sync_D', 'volatility_adjusted_concentration_D', 'SMART_MONEY_INST_NET_BUY_D',
            'tick_large_order_net_D', 'VPA_ACCELERATION_5D', 'VPA_EFFICIENCY_D',
            'MA_COHERENCE_RESONANCE_D', 'PRICE_FRACTAL_DIM_D', 'industry_leader_score_D',
            'THEME_HOTNESS_SCORE_D', 'industry_rank_slope_D', 'breakout_potential_D', 'SMART_MONEY_SYNERGY_BUY_D',
            'MA_RUBBER_BAND_EXTENSION_D', 'industry_breadth_score_D', 'industry_stagnation_score_D',
            'tick_abnormal_volume_ratio_D', 'breakout_quality_score_D', 'market_sentiment_score_D',
            'uptrend_strength_D', 'pressure_profit_D', 'net_amount_ratio_D'
        ]
        for col in targets:
            series = self.helper._get_safe_series(df, col, 0.0)
            raw_signals[col] = series
            raw_signals[f'HAB_LONG_{col}'] = series.rolling(window=hab_cfg['long']).mean()
            raw_signals[f'HAB_STD_{col}'] = series.rolling(window=hab_cfg['long']).std().replace(0, 1e-6)
        full_kinetic = ['mid_long_sync_D', 'SMART_MONEY_INST_NET_BUY_D', 'PRICE_FRACTAL_DIM_D', 'volatility_adjusted_concentration_D', 'SMART_MONEY_SYNERGY_BUY_D', 'market_sentiment_score_D']
        for target in full_kinetic:
            for d_type in ['SLOPE', 'ACCEL', 'JERK']:
                col_name = f'{d_type}_5_{target}'
                val = self.helper._get_safe_series(df, col_name, 0.0)
                raw_signals[col_name] = val
                if d_type == 'JERK':
                    raw_signals[f'HAB_MAD_{col_name}'] = (val - val.rolling(34).median()).abs().rolling(34).median().replace(0, 1e-6)
        slope_only_kinetic = ['VPA_ACCELERATION_5D', 'industry_breadth_score_D', 'industry_stagnation_score_D', 'MA_COHERENCE_RESONANCE_D', 'breakout_potential_D', 'tick_abnormal_volume_ratio_D', 'breakout_quality_score_D']
        for target in slope_only_kinetic:
            col_name = f'SLOPE_5_{target}'
            raw_signals[col_name] = self.helper._get_safe_series(df, col_name, 0.0)
        raw_signals['STATE_PARABOLIC_WARNING_D'] = self.helper._get_safe_series(df, 'STATE_PARABOLIC_WARNING_D', 0.0)
        raw_signals['STATE_MARKET_LEADER_D'] = self.helper._get_safe_series(df, 'STATE_MARKET_LEADER_D', 0.0)
        raw_signals['STATE_ROUNDING_BOTTOM_D'] = self.helper._get_safe_series(df, 'STATE_ROUNDING_BOTTOM_D', 0.0)
        raw_signals['STATE_GOLDEN_PIT_D'] = self.helper._get_safe_series(df, 'STATE_GOLDEN_PIT_D', 0.0)
        raw_signals['STATE_TRENDING_STAGE_D'] = self.helper._get_safe_series(df, 'STATE_TRENDING_STAGE_D', 0.0)
        raw_signals['STATE_EMOTIONAL_EXTREME_D'] = self.helper._get_safe_series(df, 'STATE_EMOTIONAL_EXTREME_D', 0.0)
        raw_signals['tick_chip_transfer_efficiency_D'] = self.helper._get_safe_series(df, 'tick_chip_transfer_efficiency_D', 0.0)
        raw_signals['SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D'] = self.helper._get_safe_series(df, 'SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D', 0.0)
        raw_signals['SMART_MONEY_HM_NET_BUY_D'] = self.helper._get_safe_series(df, 'SMART_MONEY_HM_NET_BUY_D', 0.0)
        raw_signals['stealth_flow_ratio_D'] = self.helper._get_safe_series(df, 'stealth_flow_ratio_D', 0.0)
        raw_signals['hab_net_inflow'] = raw_signals['net_amount_ratio_D'].rolling(window=hab_cfg['medium']).sum().fillna(0)
        raw_signals['hab_pressure_max'] = raw_signals['pressure_profit_D'].rolling(window=hab_cfg['medium']).max().replace(0, 1e-6)
        raw_signals['industry_rank_accel_D'] = self.helper._get_safe_series(df, 'industry_rank_accel_D', 0.0)
        raw_signals['intraday_distribution_confidence_D'] = self.helper._get_safe_series(df, 'intraday_distribution_confidence_D', 0.0)
        raw_signals['anomaly_intensity_D'] = self.helper._get_safe_series(df, 'anomaly_intensity_D', 0.0)
        return raw_signals

    def _calculate_parabolic_sprint_risk(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.1 · 抛物线冲刺版】识别末端冲刺风险
        - 逻辑：抛物线预警 [cite: 1] 与均线伸张度 [cite: 1] 的超限激活。
        - 版本号：V7.1.0
        """
        extension_z = np.tanh((raw_signals['MA_RUBBER_BAND_EXTENSION_D'] - raw_signals['HAB_LONG_MA_RUBBER_BAND_EXTENSION_D']) / raw_signals['HAB_STD_MA_RUBBER_BAND_EXTENSION_D'])
        sprint_risk = (raw_signals['STATE_PARABOLIC_WARNING_D'] * 0.7 + extension_z.clip(0) * 0.3).clip(0, 1)
        _temp_debug_values["parabolic_analysis"] = {"norm_extension": extension_z, "sprint_risk": sprint_risk}
        print(f"[PROBE] 抛物线风险 - 伸张Z: {extension_z.iloc[-1]:.4f}, 状态: {raw_signals['STATE_PARABOLIC_WARNING_D'].iloc[-1]}")
        return sprint_risk

    def _calculate_vpa_exhaustion_risk(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.1 · VPA效能衰竭版】计算量价背离风险
        - 逻辑：异常放量 [cite: 3] 与 VPA 效率 [cite: 1] 的负共振。
        - 版本号：V7.1.0
        """
        abn_vol_z = np.tanh((raw_signals['tick_abnormal_volume_ratio_D'] - raw_signals['HAB_LONG_tick_abnormal_volume_ratio_D']) / raw_signals['HAB_STD_tick_abnormal_volume_ratio_D'])
        vpa_eff_inv = -np.tanh(raw_signals['VPA_EFFICIENCY_D'])
        exhaustion_risk = (abn_vol_z.clip(0) * 0.6 + vpa_eff_inv.clip(0) * 0.4).clip(0, 1)
        print(f"[PROBE] VPA衰竭风险 - 异常量Z: {abn_vol_z.iloc[-1]:.4f}, 效率逆转: {vpa_eff_inv.iloc[-1]:.4f}")
        return exhaustion_risk

    def _calculate_ewd_factor(self, conviction: pd.Series, resilience: pd.Series, context: pd.Series, _temp_debug_values: Dict) -> pd.Series:
        """
        【V4.7 · 动态锁存共振版】计算熵权衰减系数 (EWD Factor)
        - 逻辑: 衡量系统多维分量的共振度。标准差越小，熵值越低，共振度越高，锁存效力越强。
        """
        # 将各分量对齐到 [0, 1] 区间进行方差分析
        s1 = (conviction.abs() + 1) / 2
        s2 = (resilience.abs() + 1) / 2
        s3 = context # 情境调制器本就在 [0.5, 1.5] 或类似区间，此处简化
        components_df = pd.concat([s1, s2, s3], axis=1)
        # 计算行标准差
        std_series = components_df.std(axis=1).fillna(1.0)
        # 映射至 [0, 1] 的 EWD 因子，使用指数函数强化共振敏感度
        ewd_factor = np.exp(-std_series * 2.5)
        _temp_debug_values["ewd_analysis"] = {"ewd_factor": ewd_factor, "component_std": std_series}
        return ewd_factor

    def _apply_latch_logic(self, df_index: pd.Index, fused_score: pd.Series, ewd_factor: pd.Series, params_dict: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V4.7 · 动态锁存共振版】执行时域积分锁存与动能保护
        - 逻辑: 统计窗口内高分频率，结合EWD因子通过Tanh加速锁定，并提供回撤保护。
        """
        lp = params_dict['latch_params']
        # 1. 时域积分：计算窗口内处于高分区间的频次
        is_high = fused_score.abs() > lp["high_score_threshold"]
        rolling_count = is_high.rolling(window=lp["window"]).sum()
        # 2. 锁存触发条件：频次达标且处于低熵共振状态
        latch_trigger = (rolling_count >= lp["hit_count"]) & (ewd_factor > lp["entropy_threshold"])
        # 3. 执行非线性锁存加速
        latched_score = np.tanh(fused_score * 1.5) # 初步加速
        # 4. 动能保护：利用 cummax 思想在锁存激活期间维持信号
        # 如果锁存触发，且分值未跌破核心阈值，则保持前期高点的一定比例
        protected_score = fused_score.copy()
        for i in range(1, len(fused_score)):
            if latch_trigger.iloc[i]:
                # 动能锁存：取当前分值与昨日分值衰减后的较大者（仅限未跌破核心阈值时）
                if abs(fused_score.iloc[i]) > lp["core_threshold"]:
                    prev_val = protected_score.iloc[i-1]
                    curr_val = fused_score.iloc[i]
                    if np.sign(prev_val) == np.sign(curr_val):
                        protected_score.iloc[i] = curr_val if abs(curr_val) > abs(prev_val) else prev_val * lp["momentum_protection_factor"]
        final_output = protected_score.clip(-1, 1)
        _temp_debug_values["latch_state"] = {
            "rolling_count": rolling_count,
            "latch_trigger": latch_trigger,
            "protected_score": protected_score
        }
        return final_output

    def _calculate_conviction_strength(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], params_dict: Dict, method_name: str, _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.2.5 · 逻辑链路完整版】顺序激活全量博弈判定模块
        - 逻辑：作为决策总中枢，顺序调用并融合 15 个子模块的判定结果。
        - 版本号：V7.2.5
        """
        weights = params_dict['belief_decay_weights']
        sync_decay = -np.tanh((raw_signals['mid_long_sync_D'] - raw_signals['HAB_LONG_mid_long_sync_D'] + raw_signals.get('SLOPE_5_mid_long_sync_D', 0.0)) / raw_signals['HAB_STD_mid_long_sync_D'])
        para_risk = self._calculate_parabolic_sprint_risk(df_index, raw_signals, _temp_debug_values)
        vpa_risk = self._calculate_vpa_exhaustion_risk(df_index, raw_signals, _temp_debug_values)
        domino_risk = self._calculate_sector_spillover_domino_risk(df_index, raw_signals, _temp_debug_values)
        regime_risk = self._calculate_market_regime_switching_risk(df_index, raw_signals, _temp_debug_values)
        handover_risk = self._calculate_smart_money_handover_risk(df_index, raw_signals, _temp_debug_values)
        collapse_risk = self._calculate_institutional_resonance_collapse(df_index, raw_signals, _temp_debug_values)
        stall_risk = self._calculate_institutional_stalling_jerk_risk(df_index, raw_signals, _temp_debug_values)
        bellwether_risk = self._calculate_market_leader_bellwether_impact(df_index, raw_signals, _temp_debug_values)
        anchor_risk = self._calculate_long_cycle_structural_anchor(df_index, raw_signals, _temp_debug_values)
        trap_risk = self._calculate_false_golden_pit_trap(df_index, raw_signals, _temp_debug_values)
        chaos_risk = self._calculate_chaotic_collapse_resonance(df_index, raw_signals, _temp_debug_values)
        macro_risk = self._calculate_macro_sector_synergy(df_index, raw_signals, _temp_debug_values)
        inst_erosion = self._calculate_institutional_erosion_index(df_index, raw_signals, _temp_debug_values)
        vacuum_risk = self._calculate_institutional_vacuum_meltdown(df_index, raw_signals, params_dict, _temp_debug_values)
        chain_risk = self._calculate_chain_collapse_resonance(df_index, raw_signals, _temp_debug_values)
        fused_conviction = (
            sync_decay.clip(0) * weights["mid_long_sync_decay"] +
            chain_risk * weights["chain_collapse_resonance"] +
            chaos_risk * weights["chaotic_collapse_resonance"] +
            vacuum_risk * weights["institutional_vacuum_meltdown"] +
            stall_risk * weights["institutional_stalling_jerk"] +
            macro_risk * weights["macro_sector_slippage"] +
            para_risk * weights["parabolic_sprint_risk"] +
            regime_risk * weights["regime_switching_risk"]
        ).clip(-1, 1)
        _temp_debug_values["conviction_dynamics"].update({"fused_conviction": fused_conviction, "sync_decay": sync_decay, "chain_risk": chain_risk})
        print(f"[PROBE] 终核决策 - 同步衰减: {sync_decay.iloc[-1]:.4f}, 连锁风险: {chain_risk.iloc[-1]:.4f}, 混沌风险: {chaos_risk.iloc[-1]:.4f}")
        return fused_conviction

    def _calculate_sector_spillover_domino_risk(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.2.1 · 健壮性增强版】判定行业溢出多米诺效应
        - 逻辑：通过 .get() 防御 KeyError，同步计算行业广度衰竭。
        - 版本号：V7.2.1
        """
        breadth_slope = raw_signals.get('SLOPE_5_industry_breadth_score_D', pd.Series(0.0, index=df_index))
        breadth_decay = -np.tanh(breadth_slope)
        stagnation_val = raw_signals.get('industry_stagnation_score_D', pd.Series(0.0, index=df_index))
        stagnation_hab = raw_signals.get('HAB_LONG_industry_stagnation_score_D', pd.Series(0.0, index=df_index))
        stagnation_std = raw_signals.get('HAB_STD_industry_stagnation_score_D', pd.Series(1e-6, index=df_index))
        stagnation_z = np.tanh((stagnation_val - stagnation_hab) / stagnation_std)
        domino_risk = (breadth_decay.clip(0) * 0.7 + stagnation_z.clip(0) * 0.3).clip(0, 1)
        print(f"[PROBE] 行业溢出风险 - 广度斜率: {breadth_slope.iloc[-1]:.4f}, 滞涨Z分: {stagnation_z.iloc[-1]:.4f}, 综合多米诺分: {domino_risk.iloc[-1]:.4f}")
        return domino_risk

    def _calculate_market_regime_switching_risk(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.2.1 · 健壮性增强版】判定市场状态切换风险
        - 逻辑：利用 .get() 确保 MA_COHERENCE_RESONANCE_D 及其导数安全加载。
        - 版本号：V7.2.1
        """
        coherence_slope = raw_signals.get('SLOPE_5_MA_COHERENCE_RESONANCE_D', pd.Series(0.0, index=df_index))
        coherence_inv = -np.tanh(coherence_slope)
        fractal_slope = raw_signals.get('SLOPE_5_PRICE_FRACTAL_DIM_D', pd.Series(0.0, index=df_index))
        chaos_accel = np.tanh(fractal_slope)
        regime_risk = (coherence_inv.clip(0) * 0.5 + chaos_accel.clip(0) * 0.5).clip(0, 1)
        print(f"[PROBE] 状态切换风险 - 共振衰减分: {coherence_inv.iloc[-1]:.4f}, 结构熵增分: {chaos_accel.iloc[-1]:.4f}, 综合状态分: {regime_risk.iloc[-1]:.4f}")
        return regime_risk

    def _calculate_smart_money_handover_risk(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.1 · 筹码置换版】判定游资接盘/机构撤退 [cite: 1]
        - 版本号：V7.1.0
        """
        handover_risk = np.tanh(raw_signals['SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D']).clip(0, 1)
        print(f"[PROBE] 筹码置换风险 - 背离强度: {handover_risk.iloc[-1]:.4f}")
        return handover_risk

    def _calculate_institutional_resonance_collapse(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.1 · 机构共振瓦解版】判定协同买入 [cite: 1] 二阶坍塌
        - 版本号：V7.1.0
        """
        accel_inv = -np.tanh(raw_signals['ACCEL_5_SMART_MONEY_SYNERGY_BUY_D'])
        collapse_risk = accel_inv.clip(0, 1)
        print(f"[PROBE] 机构共振瓦解 - 协同加速度逆转: {accel_inv.iloc[-1]:.4f}")
        return collapse_risk

    def _calculate_institutional_stalling_jerk_risk(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.1 · 失速冲击版】利用三阶导数判定抛售冲击
        - 版本号：V7.1.0
        """
        jerk_val = raw_signals['JERK_5_SMART_MONEY_INST_NET_BUY_D']
        jerk_z_inv = -np.tanh(jerk_val / (raw_signals['HAB_MAD_JERK_5_SMART_MONEY_INST_NET_BUY_D'] * 1.4826))
        print(f"[PROBE] 失速冲击风险 - JERK逆转Z: {jerk_z_inv.iloc[-1]:.4f}")
        return jerk_z_inv.clip(0, 1)

    def _calculate_chaotic_collapse_resonance(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.1.1 · 混沌共振修复版】计算系统混沌溃散共振
        - 逻辑：通过机构买盘失速冲击  与价格分形维度  的高阶无序度脉冲判定结构崩塌。
        - 公式：ChaosRisk = tanh(Jerk_inst_inv) * tanh(Jerk_fractal)。
        - 版本号：V7.1.1
        """
        inst_jerk = raw_signals['JERK_5_SMART_MONEY_INST_NET_BUY_D'] # 
        inst_j_mad = raw_signals['HAB_MAD_JERK_5_SMART_MONEY_INST_NET_BUY_D']
        inst_j_z_inv = -np.tanh(inst_jerk / (inst_j_mad * 1.4826 * 2.5))
        fractal_jerk = raw_signals['JERK_5_PRICE_FRACTAL_DIM_D'] # 
        fractal_j_mad = raw_signals['HAB_MAD_JERK_5_PRICE_FRACTAL_DIM_D']
        fractal_j_z = np.tanh(fractal_jerk / (fractal_j_mad * 1.4826 * 2.5))
        chaos_resonance = (inst_j_z_inv.clip(lower=0) * fractal_j_z.clip(lower=0)).clip(0, 1)
        _temp_debug_values["chaotic_resonance_analysis"] = {"inst_j_z_inv": inst_j_z_inv, "fractal_j_z": fractal_j_z, "chaos_resonance": chaos_resonance}
        print(f"[PROBE] 混沌共振 - 机构失速脉冲: {inst_j_z_inv.iloc[-1]:.4f}, 分形无序脉冲: {fractal_j_z.iloc[-1]:.4f}, 综合风险: {chaos_resonance.iloc[-1]:.4f}")
        return chaos_resonance

    def _calculate_institutional_vacuum_meltdown(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], params_dict: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.1 · 真空熔断版】判定买盘瞬间归零 [cite: 1]
        - 版本号：V7.1.0
        """
        support_ratio = (raw_signals['SMART_MONEY_INST_NET_BUY_D'].clip(0) / raw_signals['HAB_LONG_SMART_MONEY_INST_NET_BUY_D'].abs().replace(0, 1e-6)).clip(0, 1)
        vacuum_activation = 1 / (1 + np.exp((support_ratio - params_dict['vacuum_threshold']) * 30))
        vacuum_risk = vacuum_activation.clip(0, 1)
        _temp_debug_values["vacuum_meltdown_analysis"] = {"support_ratio": support_ratio, "vacuum_risk": vacuum_risk}
        print(f"[PROBE] 支撑真空度: {support_ratio.iloc[-1]:.4f}, 熔断激活: {vacuum_activation.iloc[-1]:.4f}")
        return vacuum_risk

    def _calculate_chain_collapse_resonance(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.1 · 连锁崩塌版】判定筹码结构 [cite: 3] 瓦解
        - 版本号：V7.1.0
        """
        stage_weight = np.where(raw_signals['STATE_TRENDING_STAGE_D'] >= 4, 1.0, 0.0)
        vac_dissipation = -np.tanh(raw_signals['SLOPE_5_volatility_adjusted_concentration_D'])
        chain_risk = (stage_weight * vac_dissipation.clip(0)).clip(0, 1)
        _temp_debug_values["chain_collapse_analysis"] = {"vac_dissipation": vac_dissipation, "chain_risk": chain_risk}
        print(f"[PROBE] 筹码连锁崩塌 - 趋势阶段权重: {stage_weight[-1]}, 集中度耗散: {vac_dissipation.iloc[-1]:.4f}")
        return chain_risk

    def _calculate_market_leader_bellwether_impact(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.1 · 领头羊风向标版】判定龙头衰减 [cite: 1, 2]
        - 版本号：V7.1.0
        """
        leader_z = (raw_signals['industry_leader_score_D'] - raw_signals['HAB_LONG_industry_leader_score_D']) / raw_signals['HAB_STD_industry_leader_score_D']
        bellwether_risk = (raw_signals['STATE_MARKET_LEADER_D'] * np.tanh(leader_z.clip(0))).clip(0, 1)
        print(f"[PROBE] 领头羊风险 - 地位系数: {leader_z.iloc[-1]:.4f}, 状态: {raw_signals['STATE_MARKET_LEADER_D'].iloc[-1]}")
        return bellwether_risk

    def _calculate_long_cycle_structural_anchor(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.1 · 结构锚定版】判定圆弧底 [cite: 1] 支撑失效
        - 版本号：V7.1.0
        """
        pot_decay = -np.tanh(raw_signals['SLOPE_5_breakout_potential_D'])
        anchor_risk = (raw_signals['STATE_ROUNDING_BOTTOM_D'] * pot_decay.clip(0)).clip(0, 1)
        print(f"[PROBE] 结构锚定风险 - 圆弧底突破衰减: {pot_decay.iloc[-1]:.4f}")
        return anchor_risk

    def _calculate_false_golden_pit_trap(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.2.4 · 鲁棒判定版】识别黄金坑假突破共振陷阱
        - 逻辑：通过 .get() 安全获取质量分数及背景，计算背离缺口。
        - 版本号：V7.2.4
        """
        had_pit = raw_signals.get('STATE_GOLDEN_PIT_D', pd.Series(0.0, index=df_index)).rolling(21).max().fillna(0)
        quality_val = raw_signals.get('breakout_quality_score_D', pd.Series(0.0, index=df_index))
        quality_hab = raw_signals.get('HAB_LONG_breakout_quality_score_D', pd.Series(0.0, index=df_index))
        quality_std = raw_signals.get('HAB_STD_breakout_quality_score_D', pd.Series(1e-6, index=df_index))
        # 质量缺口：低于HAB均值即产生陷阱分
        quality_gap = -np.tanh((quality_val - quality_hab) / quality_std)
        trap_risk = (had_pit * quality_gap.clip(0)).clip(0, 1)
        _temp_debug_values["golden_trap_analysis"] = {"quality_gap": quality_gap, "trap_risk": trap_risk}
        print(f"[PROBE] 黄金坑陷阱 - 质量缺口分: {quality_gap.iloc[-1]:.4f}, 最终风险: {trap_risk.iloc[-1]:.4f}")
        return trap_risk

    def _calculate_pressure_resilience(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], params_dict: Dict, method_name: str, _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.2.5 · 鲁棒抛压版】基于HAB对齐的抛压韧性计算
        - 逻辑：判定资金流出与历史峰值压力的占比。
        - 版本号：V7.2.5
        """
        current_outflow = raw_signals['net_amount_ratio_D'].clip(upper=0).abs()
        hab_inflow = raw_signals['hab_net_inflow'].clip(lower=1e-6)
        log_ratio = np.log2(current_outflow / hab_inflow + 0.5)
        flow_impact = np.tanh(log_ratio)
        current_p = raw_signals['pressure_profit_D']
        hab_max_p = raw_signals['hab_pressure_max']
        pressure_ratio = current_p / hab_max_p
        pressure_impact = 2 / (1 + np.exp(-4 * (pressure_ratio - 0.5))) - 1
        resilience_score = (flow_impact * 0.5 + pressure_impact * 0.5).clip(-1, 1)
        _temp_debug_values["hab_resilience"] = {"flow_impact": flow_impact, "resilience_score": resilience_score}
        print(f"[PROBE] 抛压韧性 - 流出影响: {flow_impact.iloc[-1]:.4f}, 韧性分: {resilience_score.iloc[-1]:.4f}")
        return resilience_score

    def _calculate_synergy_factor(self, conviction_strength_score: pd.Series, pressure_resilience_score: pd.Series, _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.1 · 共振因子版】计算信念与韧性同步度
        - 版本号：V7.1.0
        """
        norm_conv = (conviction_strength_score + 1) / 2
        norm_res = (pressure_resilience_score + 1) / 2
        synergy_factor = (norm_conv * norm_res + (1 - norm_conv) * (1 - norm_res)).clip(0, 1)
        _temp_debug_values["共振与背离因子"] = {"synergy_factor": synergy_factor}
        print(f"[PROBE] 共振因子: {synergy_factor.iloc[-1]:.4f}")
        return synergy_factor

    def _calculate_deception_filter(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], params_dict: Dict, method_name: str, _temp_debug_values: Dict) -> pd.Series:
        """
        【V5.3 · 异常量能效能版】诡道过滤器：校验异常放量与筹码转移的逻辑一致性
        - 逻辑：量能效能缺口 = 异常量能异常度 - 转移效率异常度。
        - 判定：若异常量能脉冲 $V_{abn}$ 很高但效能 $E_{trans}$ 极低，视为对倒欺诈。
        - 版本号：V5.3.0
        """
        # 1. 异常放量比率异常度
        abn_vol = raw_signals['tick_abnormal_volume_ratio_D']
        abn_hab = raw_signals['HAB_LONG_tick_abnormal_volume_ratio_D']
        abn_mad = raw_signals['HAB_MAD_abnormal_vol']
        abn_z = np.tanh((abn_vol - abn_hab) / (abn_mad * 1.4826))
        # 2. 筹码转移效率异常度
        trans_val = raw_signals['tick_chip_transfer_efficiency_D']
        trans_hab = raw_signals['HAB_LONG_tick_chip_transfer_efficiency_D']
        trans_std = raw_signals['HAB_STD_tick_chip_transfer_efficiency_D']
        trans_z = np.tanh((trans_val - trans_hab) / trans_std)
        # 3. 计算量能效能缺口 (Volume-Efficiency Gap)
        # 只有在异常量能显著（abn_z > 0）时，才计算其与效率的背离
        vol_eff_gap = (abn_z.clip(lower=0) - trans_z).clip(lower=0)
        # 4. 引入日内派发置信度与异常强度惩罚
        dist_conf = np.tanh(raw_signals['intraday_distribution_confidence_D']) # 
        anom_int = np.tanh((raw_signals['anomaly_intensity_D'] - raw_signals['HAB_LONG_anomaly_intensity_D']) / raw_signals['HAB_STD_anomaly_intensity_D']) # 
        # 5. 综合判定诡道过滤器
        deception_penalty = (vol_eff_gap * 0.4 + anom_int.clip(lower=0) * 0.3 + dist_conf.clip(lower=0) * 0.3).clip(0, 1)
        deception_filter = 1 - deception_penalty
        _temp_debug_values["deception_dynamics"] = {
            "abn_vol_z": abn_z,
            "trans_eff_z": trans_z,
            "vol_eff_gap": vol_eff_gap,
            "deception_filter": deception_filter
        }
        return deception_filter

    def _calculate_contextual_modulator(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], params_dict: Dict, method_name: str, _temp_debug_values: Dict, is_debug_enabled: bool, probe_ts: pd.Timestamp) -> pd.Series:
        """
        【V7.2.5 · 动力学情境版】基于情绪动能衰竭的情境调制
        - 逻辑：通过 JERK 判定情绪是否在加速期提前泄力。
        - 版本号：V7.2.5
        """
        sent_accel = raw_signals.get('ACCEL_5_market_sentiment_score_D', pd.Series(0.0, index=df_index))
        sent_jerk = raw_signals.get('JERK_5_market_sentiment_score_D', pd.Series(0.0, index=df_index))
        exhaustion_raw = (sent_accel > 0) * (sent_jerk.clip(upper=0).abs())
        ex_mad = (exhaustion_raw - exhaustion_raw.rolling(21).median()).abs().rolling(21).median().replace(0, 1e-6)
        exhaustion_score = np.tanh(exhaustion_raw / (ex_mad * 3.0))
        trend_val = raw_signals['uptrend_strength_D']
        trend_hab = raw_signals['HAB_LONG_uptrend_strength_D']
        support_ratio = trend_val / trend_hab.replace(0, 1e-6)
        norm_support = 2 / (1 + np.exp(-2 * support_ratio)) - 1
        context_modulator = 0.5 + ((1 - exhaustion_score * 0.8) * (0.5 + 0.5 * norm_support)).clip(0, 1)
        _temp_debug_values["context_dynamics"] = {"exhaustion": exhaustion_score, "modulator": context_modulator}
        print(f"[PROBE] 情境调制 - 情绪耗尽分: {exhaustion_score.iloc[-1]:.4f}, 调制系数: {context_modulator.iloc[-1]:.4f}")
        return context_modulator

    def _calculate_stealth_accumulation_bonus(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.1 · 隐秘吸筹增益版】计算隐秘流对冲分 [cite: 3]
        - 版本号：V7.1.0
        """
        stealth_z = np.tanh((raw_signals['stealth_flow_ratio_D'] - raw_signals['HAB_LONG_stealth_flow_ratio_D']) / raw_signals['HAB_STD_stealth_flow_ratio_D'])
        stealth_bonus = (stealth_z.clip(0) * 0.5 + np.tanh(raw_signals['tick_chip_transfer_efficiency_D']).clip(0) * 0.5).clip(0, 1)
        _temp_debug_values["stealth_analysis"] = {"stealth_bonus": stealth_bonus}
        print(f"[PROBE] 隐秘吸筹增益: {stealth_bonus.iloc[-1]:.4f}")
        return stealth_bonus

    def _calculate_macro_sector_synergy(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.1 · 行业位阶滑坡版】判定赛道降温 [cite: 2]
        - 版本号：V7.1.0
        """
        rank_slippage = -np.tanh(raw_signals['industry_rank_slope_D'] + raw_signals['industry_rank_accel_D'])
        theme_decay = -np.tanh(raw_signals['THEME_HOTNESS_SCORE_D'] - raw_signals['HAB_LONG_THEME_HOTNESS_SCORE_D'])
        sector_risk = (rank_slippage.clip(0) * 0.6 + theme_decay.clip(0) * 0.4).clip(0, 1)
        _temp_debug_values["macro_sector_analysis"] = {"sector_risk": sector_risk}
        print(f"[PROBE] 行业风险 - 位阶滑坡: {rank_slippage.iloc[-1]:.4f}, 主题冷却: {theme_decay.iloc[-1]:.4f}")
        return sector_risk

    def _calculate_institutional_erosion_index(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.1 · 机构侵蚀版】计算聪明钱净买入 [cite: 1] 侵蚀
        - 版本号：V7.1.0
        """
        erosion_index = -np.tanh(raw_signals['SMART_MONEY_HM_NET_BUY_D'] - raw_signals['HAB_LONG_SMART_MONEY_HM_NET_BUY_D']).clip(0, 1)
        print(f"[PROBE] 机构侵蚀指数: {erosion_index.iloc[-1]:.4f}")
        return erosion_index

    def _calculate_kinetic_transition_point(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.2.3 · 鲁棒拐点版】增加衍生指标获取的安全性
        - 逻辑：通过 .get() 防御缺失斜率，判定价量加速度的动能枯竭点。
        - 版本号：V7.2.3
        """
        vpa_accel = raw_signals.get('VPA_ACCELERATION_5D', pd.Series(0.0, index=df_index))
        accel_hab = raw_signals.get('HAB_LONG_VPA_ACCELERATION_5D', pd.Series(0.0, index=df_index))
        accel_std = raw_signals.get('HAB_STD_VPA_ACCELERATION_5D', pd.Series(1e-6, index=df_index))
        accel_z = np.tanh((vpa_accel - accel_hab) / accel_std)
        # 获取斜率：若不存在则不触发衰减
        slope_val = raw_signals.get('SLOPE_5_VPA_ACCELERATION_5D', pd.Series(0.0, index=df_index))
        slope_inv = 1 / (1 + np.exp(slope_val * 10))
        transition_point = (accel_z.clip(0) * slope_inv).clip(0, 1)
        _temp_debug_values["kinetic_transition"] = {"transition_score": transition_point}
        print(f"[PROBE] 动能转换 - 加速度Z: {accel_z.iloc[-1]:.4f}, 斜率衰减项: {slope_inv.iloc[-1]:.4f}, 最终拐点分: {transition_point.iloc[-1]:.4f}")
        return transition_point

    def _perform_final_fusion(self, df_index: pd.Index, conviction_score: pd.Series, resilience_score: pd.Series, deception_filter: pd.Series, stealth_bonus: pd.Series, params_dict: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V7.1 · 全息时空审判版】执行多维融合与隐秘流对冲
        - 版本号：V7.1.0
        """
        exponent = params_dict['final_exponent']
        intensity = (conviction_score * 0.7 + resilience_score * 0.3).clip(0, 1)
        net_decay = (intensity * (2 - deception_filter) - stealth_bonus * 0.4).clip(-1, 1)
        final_score = np.sign(net_decay) * (net_decay.abs() ** exponent)
        print(f"[PROBE] 最终融合 - 原始强度: {intensity.iloc[-1]:.4f}, 隐秘对冲: {stealth_bonus.iloc[-1]:.4f}, 最终分值: {final_score.iloc[-1]:.4f}")
        return final_score.clip(-1, 1).fillna(0)














