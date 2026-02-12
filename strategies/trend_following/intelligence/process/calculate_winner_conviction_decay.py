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
        【V4.7 · 动态锁存共振版】引入HAB时域积分与EWD熵权锁存
        - 核心逻辑: 在计算末端接入锁存器，通过低熵共振过滤高频闪烁。
        - 升级维度: 对齐V4.7标准，加入动能保护以应对洗盘假动作。
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
        if not self.helper._validate_required_signals(df, all_required_signals, method_name):
            return pd.Series(dtype=np.float32)
        _temp_debug_values = {}
        raw_signals = self._get_raw_signals(df, df_index, params_dict, method_name)
        _temp_debug_values["raw_signals"] = raw_signals
        conviction_score = self._calculate_conviction_strength(df, df_index, raw_signals, params_dict, method_name, _temp_debug_values)
        resilience_score = self._calculate_pressure_resilience(df, df_index, raw_signals, params_dict, method_name, _temp_debug_values)
        deception_filter = self._calculate_deception_filter(df, df_index, raw_signals, params_dict, method_name, _temp_debug_values)
        context_modulator = self._calculate_contextual_modulator(df, df_index, raw_signals, params_dict, method_name, _temp_debug_values, is_debug_enabled, probe_ts)
        # 初始融合分值
        fused_score = self._perform_final_fusion(df_index, conviction_score, resilience_score, deception_filter, context_modulator, params_dict, _temp_debug_values)
        # 核心锁存逻辑：计算熵权因子并执行锁存
        ewd_factor = self._calculate_ewd_factor(conviction_score, resilience_score, context_modulator, _temp_debug_values)
        final_score = self._apply_latch_logic(df_index, fused_score, ewd_factor, params_dict, _temp_debug_values)
        if is_debug_enabled and probe_ts:
            self._execute_intelligence_probe(method_name, probe_ts, _temp_debug_values, final_score)
        return final_score.astype(np.float32)

    def _execute_intelligence_probe(self, method_name: str, probe_ts: pd.Timestamp, _temp_debug_values: Dict, final_score: pd.Series):
        """
        【V5.1 · 筹码效能核验版】探针全息采样：监测资金到筹码的转换损失
        - 版本号：V5.1.0
        """
        print(f"\n{'>'*30} [V5.1 EFFICIENCY CHECK PROBE: {probe_ts.strftime('%Y-%m-%d')}] {'<'*30}")
        dec = _temp_debug_values.get("deception_dynamics", {})
        print(f"--- [资金-筹码 效能转换分析] ---")
        print(f"  > 大单流向诚意度 (Sincerity_Z): {dec.get('sincerity_z').loc[probe_ts]:.4f}")
        print(f"  > 筹码转移有效性 (Efficiency_Z): {dec.get('efficiency_z').loc[probe_ts]:.4f}")
        print(f"  > 效能转化缺口 (Efficiency_Gap): {dec.get('efficiency_gap').loc[probe_ts]:.4f}")
        print(f"  > 诡道过滤器健康度: {dec.get('deception_filter').loc[probe_ts]:.4f}")
        latch = _temp_debug_values.get("latch_state", {})
        print(f"--- [物理锁存保护] ---")
        print(f"  > 动态锁存状态: {latch.get('latch_trigger').loc[probe_ts]} | 输出分值: {final_score.loc[probe_ts]:.4f}")
        print(f"{'='*110}\n")

    def _collect_and_print_debug_info(self, method_name: str, probe_ts: pd.Timestamp, debug_output: Dict, _temp_debug_values: Dict, final_score: pd.Series):
        """
        统一收集并打印 calculate_winner_conviction_decay 的调试信息。
        """
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
        for key, value in _temp_debug_values["原始信号值"].items():
            if isinstance(value, pd.Series):
                val = value.loc[probe_ts] if probe_ts in value.index else np.nan
                debug_output[f"        '{key}': {val:.4f}"] = ""
            elif isinstance(value, dict): # 处理 _temp_debug_values["原始信号值"] 中的字典
                debug_output[f"        '{key}':"] = ""
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, pd.Series):
                        val = sub_value.loc[probe_ts] if probe_ts in sub_value.index else np.nan
                        debug_output[f"          {sub_key}: {val:.4f}"] = ""
                    else:
                        debug_output[f"          {sub_key}: {sub_value}"] = ""
            else:
                debug_output[f"        '{key}': {value}"] = ""
        sections = ["信念强度", "压力韧性", "共振与背离因子", "诡道过滤", "情境调制", "最终融合"]
        for section_name in sections:
            if section_name in _temp_debug_values:
                debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- {section_name} ---"] = ""
                for key, series_or_val in _temp_debug_values[section_name].items():
                    if isinstance(series_or_val, pd.Series):
                        val = series_or_val.loc[probe_ts] if probe_ts in series_or_val.index else np.nan
                        debug_output[f"        {key}: {val:.4f}"] = ""
                    elif isinstance(series_or_val, dict):
                        debug_output[f"        {key}:"] = ""
                        for sub_key, sub_value in series_or_val.items():
                            if isinstance(sub_value, pd.Series):
                                val = sub_value.loc[probe_ts] if probe_ts in sub_value.index else np.nan
                                debug_output[f"          {sub_key}: {val:.4f}"] = ""
                            else:
                                debug_output[f"          {sub_key}: {sub_value}"] = ""
                    else:
                        debug_output[f"        {key}: {series_or_val}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 赢家信念衰减诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
        self.helper._print_debug_output(debug_output)

    def _get_decay_params_and_signals(self, config: Dict, method_name: str) -> Tuple[Dict, List[str]]:
        """
        【V5.1 · 筹码效能核验版】重构信号依赖，引入筹码转移效率校验
        - 逻辑：新增 tick_chip_transfer_efficiency_D 用于穿透大单流向的“做功效率”。
        - 版本号：V5.1.0
        """
        decay_params = get_param_value(config.get('winner_conviction_decay_params'), {})
        fibo_periods = ["5", "13", "21", "34"]
        # 权重体系：向资金-筹码转换效率倾斜
        belief_decay_weights = {
            "peak_migration_impact": 0.15,
            "efficiency_gap_penalty": 0.35, # 新增效能缺口权重
            "sincerity_gap_penalty": 0.30, 
            "deception_active_impact": 0.20
        }
        required_df_columns = [
            'peak_migration_speed_5d_D', 'PRICE_ENTROPY_D', 'intraday_main_force_activity_D',
            'tick_large_order_net_D', 'tick_chip_transfer_efficiency_D', 'deception_lure_long_intensity_D',
            'wash_trade_intensity_D', 'market_sentiment_score_D', 'uptrend_strength_D', 
            'chip_entropy_D', 'pressure_profit_D', 'winner_rate_D'
        ]
        kinetic_targets = ['tick_large_order_net_D', 'tick_chip_transfer_efficiency_D']
        for target in kinetic_targets:
            for p in ["5"]:
                required_df_columns.extend([f'SLOPE_{p}_{target}', f'ACCEL_{p}_{target}', f'JERK_{p}_{target}'])
        all_required_signals = list(set(required_df_columns))
        params_dict = {
            'decay_params': decay_params,
            'fibo_periods': fibo_periods,
            'belief_decay_weights': belief_decay_weights,
            'hab_settings': {"short": 13, "medium": 21, "long": 34},
            'latch_params': {"window": 5, "hit_count": 3, "high_score_threshold": 0.618, "core_threshold": 0.382, "momentum_protection_factor": 0.95, "entropy_threshold": 0.75},
            'final_exponent': 3.5 # 进一步提升对“空头回补”与“低效买盘”的识别攻击性
        }
        return params_dict, all_required_signals

    def _get_raw_signals(self, df: pd.DataFrame, df_index: pd.Index, params_dict: Dict, method_name: str) -> Dict[str, pd.Series]:
        """
        【V5.1 · 筹码效能核验版】构建转移效率HAB背景池
        - 逻辑：通过34日滚动均值与MAD建立转移效率的稳健底座。
        - 版本号：V5.1.0
        """
        raw_signals = {}
        hab_cfg = params_dict['hab_settings']
        # 扩展基础信号列表
        hab_targets = [
            'wash_trade_intensity_D', 'deception_lure_long_intensity_D', 
            'intraday_main_force_activity_D', 'tick_large_order_net_D', 
            'tick_chip_transfer_efficiency_D', 'peak_migration_speed_5d_D'
        ]
        for col in hab_targets:
            series = self.helper._get_safe_series(df, col, 0.0)
            raw_signals[col] = series
            raw_signals[f'HAB_LONG_{col}'] = series.rolling(window=hab_cfg['long']).mean()
            raw_signals[f'HAB_STD_{col}'] = series.rolling(window=hab_cfg['long']).std().replace(0, 1e-6)
        # 为转移效率建立MAD稳健背景
        raw_signals['HAB_MAD_transfer'] = (raw_signals['tick_chip_transfer_efficiency_D'] - raw_signals['HAB_LONG_tick_chip_transfer_efficiency_D']).abs().rolling(window=hab_cfg['long']).median().replace(0, 1e-6)
        # 补齐动力学导数及其余信号
        kinetic_list = ['tick_large_order_net_D', 'tick_chip_transfer_efficiency_D', 'chip_entropy_D']
        for target in kinetic_list:
            for d_type in ['SLOPE', 'ACCEL', 'JERK']:
                col_name = f'{d_type}_5_{target}'
                val_series = self.helper._get_safe_series(df, col_name, 0.0)
                raw_signals[col_name] = val_series.where(val_series.abs() > (val_series.rolling(21).std() * 1.2), 0.0)
        raw_signals['PRICE_ENTROPY_D'] = self.helper._get_safe_series(df, 'PRICE_ENTROPY_D', 0.0)
        raw_signals['winner_rate_D'] = self.helper._get_safe_series(df, 'winner_rate_D', 0.0)
        raw_signals['pressure_profit_D'] = self.helper._get_safe_series(df, 'pressure_profit_D', 0.0)
        raw_signals['net_amount_ratio_D'] = self.helper._get_safe_series(df, 'net_amount_ratio_D', 0.0)
        raw_signals['hab_net_inflow'] = raw_signals['net_amount_ratio_D'].rolling(window=hab_cfg['medium']).sum().fillna(0)
        raw_signals['hab_pressure_max'] = raw_signals['pressure_profit_D'].rolling(window=hab_cfg['medium']).max().replace(0, 1e-6)
        return raw_signals

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

    def _calculate_composite_chip_distribution_whisper(self, df: pd.DataFrame, df_index: pd.Index, mtf_weights_config: Dict, composite_weights: Dict, method_name: str) -> pd.Series:
        """
        计算复合的筹码派发低语信号，替代 SCORE_CHIP_RISK_DISTRIBUTION_WHISPER。
        """
        components = {
            "buy_quote_exhaustion": {'signal': 'buy_quote_exhaustion_rate_D', 'bipolar': True, 'ascending': True},
            "bid_side_liquidity_inverted": {'signal': 'bid_side_liquidity_D', 'bipolar': True, 'ascending': False}, # 流动性低 -> 派发风险高
            "main_force_slippage_inverted": {'signal': 'main_force_slippage_index_D', 'bipolar': True, 'ascending': True}, # 滑点高 -> 派发成本高，但这里是反向，所以滑点低 -> 派发风险高
            "market_impact_cost": {'signal': 'market_impact_cost_D', 'bipolar': True, 'ascending': True},
        }
        fused_scores = []
        total_weight = 0.0
        for key, config in components.items():
            raw_signal = self.helper._get_safe_series(df, config['signal'], np.nan, method_name=method_name)
            if raw_signal.isnull().all():
                continue
            mtf_score = self.helper._get_mtf_slope_accel_score(
                df, config['signal'], mtf_weights_config, df_index, method_name,
                bipolar=config['bipolar'], ascending=config['ascending']
            )
            weight = composite_weights.get(key, 0.0)
            if pd.notna(mtf_score).any() and weight > 0:
                fused_scores.append(mtf_score * weight)
                total_weight += weight
        if not fused_scores or total_weight == 0:
            return pd.Series(np.nan, index=df_index, dtype=np.float32)
        composite_score = sum(fused_scores) / total_weight
        return composite_score.clip(-1, 1)

    def _calculate_composite_market_tension(self, df: pd.DataFrame, df_index: pd.Index, mtf_weights_config: Dict, composite_weights: Dict, method_name: str) -> pd.Series:
        """
        计算复合的市场张力信号，替代 SCORE_FOUNDATION_AXIOM_MARKET_TENSION。
        """
        components = {
            "structural_tension": {'signal': 'structural_tension_index_D', 'bipolar': True, 'ascending': True},
            "volatility_expansion": {'signal': 'volatility_expansion_ratio_D', 'bipolar': True, 'ascending': True},
            "market_sentiment_inverted": {'signal': 'market_sentiment_score_D', 'bipolar': True, 'ascending': False}, # 市场情绪低 -> 张力高
        }
        fused_scores = []
        total_weight = 0.0
        for key, config in components.items():
            raw_signal = self.helper._get_safe_series(df, config['signal'], np.nan, method_name=method_name)
            if raw_signal.isnull().all():
                continue
            mtf_score = self.helper._get_mtf_slope_accel_score(
                df, config['signal'], mtf_weights_config, df_index, method_name,
                bipolar=config['bipolar'], ascending=config['ascending']
            )
            weight = composite_weights.get(key, 0.0)
            if pd.notna(mtf_score).any() and weight > 0:
                fused_scores.append(mtf_score * weight)
                total_weight += weight
        if not fused_scores or total_weight == 0:
            return pd.Series(np.nan, index=df_index, dtype=np.float32)
        composite_score = sum(fused_scores) / total_weight
        return composite_score.clip(-1, 1)

    def _calculate_composite_sentiment_pendulum(self, df: pd.DataFrame, df_index: pd.Index, mtf_weights_config: Dict, composite_weights: Dict, method_name: str) -> pd.Series:
        """
        计算复合的情绪摆动信号，替代 SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM。
        """
        components = {
            "market_sentiment": {'signal': 'market_sentiment_score_D', 'bipolar': True, 'ascending': True},
            "retail_fomo_premium": {'signal': 'retail_fomo_premium_index_D', 'bipolar': True, 'ascending': True},
            "retail_panic_surrender_inverted": {'signal': 'retail_panic_surrender_index_D', 'bipolar': True, 'ascending': False}, # 散户恐慌低 -> 情绪好
        }
        fused_scores = []
        total_weight = 0.0
        for key, config in components.items():
            raw_signal = self.helper._get_safe_series(df, config['signal'], np.nan, method_name=method_name)
            if raw_signal.isnull().all():
                continue
            mtf_score = self.helper._get_mtf_slope_accel_score(
                df, config['signal'], mtf_weights_config, df_index, method_name,
                bipolar=config['bipolar'], ascending=config['ascending']
            )
            weight = composite_weights.get(key, 0.0)
            if pd.notna(mtf_score).any() and weight > 0:
                fused_scores.append(mtf_score * weight)
                total_weight += weight
        if not fused_scores or total_weight == 0:
            return pd.Series(np.nan, index=df_index, dtype=np.float32)
        composite_score = sum(fused_scores) / total_weight
        return composite_score.clip(-1, 1)

    def _calculate_composite_deception_index(self, df: pd.DataFrame, df_index: pd.Index, mtf_weights_config: Dict, composite_weights: Dict, method_name: str) -> pd.Series:
        """
        计算复合的欺骗指数，替代直接引用 'deception_index_D'。
        """
        components = {
            "deception_lure_long": {'signal': 'deception_lure_long_intensity_D', 'bipolar': True, 'ascending': True},
            "wash_trade_intensity": {'signal': 'wash_trade_intensity_D', 'bipolar': True, 'ascending': True},
        }
        fused_scores = []
        total_weight = 0.0
        for key, config in components.items():
            raw_signal = self.helper._get_safe_series(df, config['signal'], np.nan, method_name=method_name)
            if raw_signal.isnull().all():
                continue
            mtf_score = self.helper._get_mtf_slope_accel_score(
                df, config['signal'], mtf_weights_config, df_index, method_name,
                bipolar=config['bipolar'], ascending=config['ascending']
            )
            weight = composite_weights.get(key, 0.0)
            if pd.notna(mtf_score).any() and weight > 0:
                fused_scores.append(mtf_score * weight)
                total_weight += weight
        if not fused_scores or total_weight == 0:
            return pd.Series(np.nan, index=df_index, dtype=np.float32)
        composite_score = sum(fused_scores) / total_weight
        return composite_score.clip(-1, 1)

    def _calculate_conviction_strength(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], params_dict: Dict, method_name: str, _temp_debug_values: Dict) -> pd.Series:
        """
        【V4.8 · 结构熵控版】重构信念强度，整合结构重心迁移与非线性激活
        - 逻辑：信念衰减 = 动能冲击(Jerk) + 重心下移(PeakSpeed) + 结构紊乱(Entropy)。
        - 版本号：V4.8.0
        """
        weights = params_dict['belief_decay_weights']
        # 1. 重心下移激活：peak_migration_speed_5d_D (正值代表向上，负值代表向下)
        migration_speed = raw_signals['peak_migration_speed_5d_D']
        norm_migration = -np.tanh(migration_speed / 2.0).clip(upper=1) # 仅关注下移分量
        # 2. 价格熵增激活：PRICE_ENTROPY_D (熵增代表趋势有序度瓦解)
        price_entropy = raw_signals['PRICE_ENTROPY_D']
        e_median = price_entropy.rolling(21).median()
        e_mad = (price_entropy - e_median).abs().rolling(21).median().replace(0, 1e-6)
        norm_entropy = np.tanh((price_entropy - e_median) / (e_mad * 1.4826))
        # 3. 三阶动能冲击 (Jerk) 激活
        entropy_jerk = raw_signals['JERK_5_chip_entropy_D']
        j_median = entropy_jerk.rolling(21).median()
        j_mad = (entropy_jerk - j_median).abs().rolling(21).median().replace(0, 1e-6)
        norm_jerk = np.tanh((entropy_jerk - j_median) / (j_mad * 1.4826))
        # 4. 获利盘极值敏感度
        winner_rate = raw_signals['winner_rate_D']
        winner_extreme = 2 / (1 + np.exp(-15 * (winner_rate - 0.85))) - 1
        # 综合信念强度
        fused_conviction = (
            norm_migration * weights["peak_migration_impact"] +
            norm_entropy * weights["price_entropy_chaos"] +
            norm_jerk * weights["jerk_conviction_shock"] +
            winner_extreme * weights["winner_extreme_sensitivity"]
        ).clip(-1, 1)
        _temp_debug_values["conviction_dynamics"] = {
            "norm_migration": norm_migration,
            "norm_entropy": norm_entropy,
            "norm_jerk": norm_jerk,
            "fused_conviction": fused_conviction
        }
        return fused_conviction

    def _calculate_pressure_resilience(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], params_dict: Dict, method_name: str, _temp_debug_values: Dict) -> pd.Series:
        """
        【V4.6 · 动力学激活版】重构HAB抛压韧性，采用对数占比压缩
        - 逻辑：当日流出与HAB流入的比例，使用Log2与Sigmoid结合映射至[-1, 1]。
        """
        # 1. 资金流出/HAB流入 占比激活
        current_outflow = raw_signals['net_amount_ratio_D'].clip(upper=0).abs()
        hab_inflow = raw_signals['hab_net_inflow'].clip(lower=1e-6)
        # 计算流出占比的倍数
        outflow_ratio = current_outflow / hab_inflow
        # 使用对数缩放：log2(1.0)=0 代表均衡，>0 代表超额流出
        log_ratio = np.log2(outflow_ratio + 0.5) # 偏移处理，防止低值塌陷
        flow_impact = np.tanh(log_ratio)
        # 2. 抛压强度/HAB峰值 占比激活
        current_p = raw_signals['pressure_profit_D']
        hab_max_p = raw_signals['hab_pressure_max']
        pressure_ratio = current_p / hab_max_p
        pressure_impact = 2 / (1 + np.exp(-4 * (pressure_ratio - 0.5))) - 1
        resilience_score = (flow_impact * 0.5 + pressure_impact * 0.5).clip(-1, 1)
        _temp_debug_values["hab_resilience"] = {
            "flow_impact_tanh": flow_impact,
            "pressure_impact_sigmoid": pressure_impact,
            "resilience_score": resilience_score
        }
        return resilience_score

    def _calculate_synergy_factor(self, conviction_strength_score: pd.Series, pressure_resilience_score: pd.Series, _temp_debug_values: Dict) -> pd.Series:
        """
        计算共振与背离因子。
        """
        norm_conviction = (conviction_strength_score + 1) / 2
        norm_resilience = (pressure_resilience_score + 1) / 2
        synergy_factor = (norm_conviction * norm_resilience + (1 - norm_conviction) * (1 - norm_resilience)).clip(0, 1)
        _temp_debug_values["共振与背离因子"] = {
            "norm_conviction": norm_conviction,
            "norm_resilience": norm_resilience,
            "synergy_factor": synergy_factor
        }
        return synergy_factor

    def _calculate_deception_filter(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], params_dict: Dict, method_name: str, _temp_debug_values: Dict) -> pd.Series:
        """
        【V5.1 · 筹码效能核验版】诡道过滤器：校验资金流转码的物理有效性
        - 逻辑：效能缺口 = 大单异常度 - 转移效率异常度。若大单狂买但效率停滞，视为虚假支撑。
        - 版本号：V5.1.0
        """
        # 1. 诚意缺口 (声势 vs 兵力)
        active_z = np.tanh((raw_signals['intraday_main_force_activity_D'] - raw_signals['HAB_LONG_intraday_main_force_activity_D']) / raw_signals['HAB_STD_intraday_main_force_activity_D'])
        sincerity_z = np.tanh((raw_signals['tick_large_order_net_D'] - raw_signals['HAB_LONG_tick_large_order_net_D']) / (raw_signals['HAB_STD_tick_large_order_net_D'] * 1.5))
        sincerity_gap = (active_z.clip(lower=0) - sincerity_z).clip(lower=0)
        # 2. 效能缺口 (兵力 vs 战果)
        transfer_val = raw_signals['tick_chip_transfer_efficiency_D']
        transfer_hab = raw_signals['HAB_LONG_tick_chip_transfer_efficiency_D']
        transfer_mad = raw_signals['HAB_MAD_transfer']
        efficiency_z = np.tanh((transfer_val - transfer_hab) / (transfer_mad * 1.4826))
        # 当大单为正(买入)且效率为负或极低时，效能缺口激增
        efficiency_gap = (sincerity_z.clip(lower=0) - efficiency_z).clip(lower=0)
        # 3. 综合诡道压力判定
        lure_z = np.tanh((raw_signals['deception_lure_long_intensity_D'] - raw_signals['HAB_LONG_deception_lure_long_intensity_D']) / raw_signals['HAB_STD_deception_lure_long_intensity_D'])
        deception_penalty = (efficiency_gap * 0.45 + sincerity_gap * 0.35 + lure_z.clip(lower=0) * 0.2).clip(0, 1)
        deception_filter = 1 - deception_penalty
        _temp_debug_values["deception_dynamics"] = {
            "sincerity_z": sincerity_z,
            "efficiency_z": efficiency_z,
            "efficiency_gap": efficiency_gap,
            "deception_filter": deception_filter
        }
        return deception_filter

    def _calculate_contextual_modulator(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], params_dict: Dict, method_name: str, _temp_debug_values: Dict, is_debug_enabled: bool, probe_ts: pd.Timestamp) -> pd.Series:
        """
        【V4.6 · 动力学激活版】重构情境调制，基于动能衰竭模型的条件激活
        - 逻辑：通过JERK判断情绪动能是否在加速度依然为正的情况下提前“泄力”。
        """
        # 1. 情绪高潮衰竭激活 (Exhaustion)
        sent_accel = raw_signals['ACCEL_5_market_sentiment_score_D']
        sent_jerk = raw_signals['JERK_5_market_sentiment_score_D']
        # 捕捉“加速上升中的二阶力转负” (极度危险的情绪顶点)
        exhaustion_raw = (sent_accel > 0) * (sent_jerk.clip(upper=0).abs())
        ex_median = exhaustion_raw.rolling(21).median()
        ex_mad = (exhaustion_raw - ex_median).abs().rolling(21).median().replace(0, 1e-6)
        exhaustion_score = np.tanh(exhaustion_raw / (ex_mad * 3.0))
        # 2. 趋势存量支持比率激活
        trend_val = raw_signals['uptrend_strength_D']
        trend_hab = raw_signals['HAB_LONG_uptrend_strength_D']
        # 计算当前相对于存量的强弱，使用Sigmoid在1.0处提供平滑过渡
        support_ratio = trend_val / trend_hab.replace(0, 1e-6)
        norm_support = 2 / (1 + np.exp(-2 * support_ratio)) - 1
        context_base = (1 - exhaustion_score * 0.8) * (0.5 + 0.5 * norm_support)
        context_modulator = 0.5 + context_base.clip(0, 1)
        _temp_debug_values["context_dynamics"] = {
            "exhaustion_score": exhaustion_score,
            "norm_support": norm_support,
            "context_modulator": context_modulator
        }
        return context_modulator

    def _perform_final_fusion(self, df_index: pd.Index, conviction_score: pd.Series, resilience_score: pd.Series, deception_filter: pd.Series, context_modulator: pd.Series, params_dict: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V4.8 · 结构熵控版】重构融合算法，解决旧版签名不匹配问题
        - 逻辑：通过非线性加权融合信念、压力、诡道与情境。
        - 版本号：V4.8.0
        """
        exponent = params_dict['final_exponent']
        # 整合核心攻击力度
        raw_magnitude = (
            conviction_score * 0.40 + 
            resilience_score * 0.30 + 
            (1 - deception_filter) * 0.20 + 
            (context_modulator - 1.0) * 0.10
        )
        # 非线性指数激活，强化临界状态的输出
        final_score = np.sign(raw_magnitude) * (raw_magnitude.abs() ** exponent)
        final_score = final_score.clip(-1, 1).fillna(0)
        _temp_debug_values["最终融合"] = {"raw_magnitude": raw_magnitude, "final_score": final_score}
        return final_score














