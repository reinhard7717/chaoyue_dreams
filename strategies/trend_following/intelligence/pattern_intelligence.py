import pandas as pd
import numpy as np
from typing import Dict, Any
from strategies.trend_following.utils import get_params_block, get_param_value, get_adaptive_mtf_normalized_bipolar_score, bipolar_to_exclusive_unipolar

class PatternIntelligence:
    """
    【V8.0 · 清洁版】形态智能引擎
    - 核心修改: 移除了所有用于调试的print探针和过程性打印，恢复静默运行模式。
    """
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def _get_safe_series(self, df: pd.DataFrame, column_name: str, default_value: Any = 0.0, method_name: str = "未知方法") -> pd.Series:
        """
        安全地从DataFrame获取Series，如果不存在则打印警告并返回默认Series。
        """
        if column_name not in df.columns:
            print(f"    -> [形态情报警告] 方法 '{method_name}' 缺少数据 '{column_name}'，使用默认值 {default_value}。")
            return pd.Series(default_value, index=df.index)
        return df[column_name]

    def run_pattern_analysis_command(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V8.5 · 多方炮集成版】形态分析总指挥
        - 核心升级: 废弃原子层面的“共振”和“领域健康度”信号。
        - 核心职责: 只输出形态领域的原子公理信号和形态背离信号。
        - 移除信号: SCORE_PATTERN_BULLISH_RESONANCE, SCORE_PATTERN_BEARISH_RESONANCE, BIPOLAR_PATTERN_DOMAIN_HEALTH, SCORE_PATTERN_BOTTOM_REVERSAL, SCORE_PATTERN_TOP_REVERSAL。
        - 【新增】诊断“回踩确认二次启动”形态和“多方炮”形态。
        """
        all_states = {}
        p_conf = get_params_block(self.strategy, 'pattern_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            return {}
        norm_window = get_param_value(p_conf.get('norm_window'), 60)
        axiom_divergence = self._diagnose_axiom_divergence(df, norm_window)
        axiom_reversal = self._diagnose_axiom_reversal(df, norm_window)
        axiom_breakout = self._diagnose_axiom_breakout(df, norm_window)
        all_states['SCORE_PATTERN_AXIOM_DIVERGENCE'] = axiom_divergence
        all_states['SCORE_PATTERN_AXIOM_REVERSAL'] = axiom_reversal
        all_states['SCORE_PATTERN_AXIOM_BREAKOUT'] = axiom_breakout
        bullish_divergence, bearish_divergence = bipolar_to_exclusive_unipolar(axiom_divergence)
        all_states['SCORE_PATTERN_BULLISH_DIVERGENCE'] = bullish_divergence.astype(np.float32)
        all_states['SCORE_PATTERN_BEARISH_DIVERGENCE'] = bearish_divergence.astype(np.float32)
        # 修改行: 诊断“回踩确认二次启动”形态
        axiom_pullback_confirmation = self._diagnose_axiom_pullback_confirmation(df)
        all_states['SCORE_PATTERN_PULLBACK_CONFIRMATION'] = axiom_pullback_confirmation
        # 新增行: 诊断“多方炮”形态
        axiom_duofangpao = self._diagnose_axiom_duofangpao(df)
        all_states['SCORE_PATTERN_DUOFANGPAO'] = axiom_duofangpao
        return all_states

    def _diagnose_axiom_divergence(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V3.1 · 清洁与多时间维度归一化版】形态公理一：诊断“背离”
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `divergence_score` 的归一化方式改为多时间维度自适应归一化。
        """
        price_slope = self._get_safe_series(df, 'SLOPE_13_close_D', 0, method_name="_diagnose_axiom_divergence")
        momentum_slope = self._get_safe_series(df, 'SLOPE_13_intraday_vwap_div_index_D', 0, method_name="_diagnose_axiom_divergence")
        bullish_divergence_strength = ((price_slope < 0) & (momentum_slope > 0)).astype(float)
        bearish_divergence_strength = ((price_slope > 0) & (momentum_slope < 0)).astype(float)
        raw_divergence_score = bullish_divergence_strength - bearish_divergence_strength
        p_conf_pattern = get_params_block(self.strategy, 'pattern_params', {})
        tf_weights_pattern = get_param_value(p_conf_pattern.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}) # 借用筹码的MTF权重配置
        divergence_score = get_adaptive_mtf_normalized_bipolar_score(raw_divergence_score, df.index, tf_weights_pattern)
        return divergence_score.astype(np.float32)

    def _diagnose_axiom_reversal(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V3.1 · 清洁与多时间维度归一化版】形态公理二：诊断“反转”
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `reversal_score` 的归一化方式改为多时间维度自适应归一化。
        """
        raw_reversal_score = self._get_safe_series(df, 'counterparty_exhaustion_index_D', 0, method_name="_diagnose_axiom_reversal")
        p_conf_pattern = get_params_block(self.strategy, 'pattern_params', {})
        tf_weights_pattern = get_param_value(p_conf_pattern.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        reversal_score = get_adaptive_mtf_normalized_bipolar_score(raw_reversal_score, df.index, tf_weights_pattern)
        return reversal_score.astype(np.float32)

    def _diagnose_axiom_breakout(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V3.1 · 清洁与多时间维度归一化版】形态公理三：诊断“突破”
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `breakout_score` 的归一化方式改为多时间维度自适应归一化。
        """
        is_breakout_up = (self._get_safe_series(df, 'close_D', method_name="_diagnose_axiom_breakout") > self._get_safe_series(df, 'dynamic_consolidation_high_D', np.inf, method_name="_diagnose_axiom_breakout")).astype(float)
        is_breakout_down = (self._get_safe_series(df, 'close_D', method_name="_diagnose_axiom_breakout") < self._get_safe_series(df, 'dynamic_consolidation_low_D', -np.inf, method_name="_diagnose_axiom_breakout")).astype(float)
        breakout_direction = is_breakout_up - is_breakout_down
        breakout_quality = self._get_safe_series(df, 'breakout_quality_score_D', pd.Series(0, index=df.index), method_name="_diagnose_axiom_breakout")
        raw_breakout_score = breakout_direction * breakout_quality
        p_conf_pattern = get_params_block(self.strategy, 'pattern_params', {})
        tf_weights_pattern = get_param_value(p_conf_pattern.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        breakout_score = get_adaptive_mtf_normalized_bipolar_score(raw_breakout_score, df.index, tf_weights_pattern)
        return breakout_score.astype(np.float32)

    def _diagnose_axiom_pullback_confirmation(self, df: pd.DataFrame) -> pd.Series:
        """
        【V3.0 · 深度博弈与B点结构强化版】形态公理四：诊断“回踩确认二次启动”形态
        - 核心逻辑: 识别股价在量能萎缩后放量突破，随后缩量回调，再放量突破的二次启动形态。
                    回调阶段融入微观资金流、筹码结构、行为效率等高级指标，量化“健康洗盘”特征。
                    新增B点收盘价高于A点收盘价，以及B点结构趋势相对于A点向上的判断。
        - 信号输出: 在形态的“二次启动日”（B日）输出1.0；否则输出0.0。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        """
        df_index = df.index
        pullback_confirmation_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        p_conf_pattern = get_params_block(self.strategy, 'pattern_params', {})
        n_pre_A = get_param_value(p_conf_pattern.get('pullback_confirmation_n_pre_A'), 10)
        n_pullback_max = get_param_value(p_conf_pattern.get('pullback_confirmation_n_pullback_max'), 13)
        # 获取核心K线数据
        close_D = self._get_safe_series(df, 'close_D', method_name="_diagnose_axiom_pullback_confirmation")
        high_D = self._get_safe_series(df, 'high_D', method_name="_diagnose_axiom_pullback_confirmation") # 新增获取高点
        pct_change_D = self._get_safe_series(df, 'pct_change_D', method_name="_diagnose_axiom_pullback_confirmation")
        volume_D = self._get_safe_series(df, 'volume_D', method_name="_diagnose_axiom_pullback_confirmation")
        vol_ma5_D = self._get_safe_series(df, 'VOL_MA_5_D', method_name="_diagnose_axiom_pullback_confirmation")
        vol_ma21_D = self._get_safe_series(df, 'VOL_MA_21_D', method_name="_diagnose_axiom_pullback_confirmation")
        # 获取高级指标
        main_force_net_flow_calibrated_D = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', method_name="_diagnose_axiom_pullback_confirmation")
        short_term_concentration_90pct_D_slope_5d = self._get_safe_series(df, 'SLOPE_5_short_term_concentration_90pct_90pct_D', method_name="_diagnose_axiom_pullback_confirmation") # 修正列名
        large_order_pressure_D = self._get_safe_series(df, 'large_order_pressure_D', method_name="_diagnose_axiom_pullback_confirmation")
        large_order_support_D = self._get_safe_series(df, 'large_order_support_D', method_name="_diagnose_axiom_pullback_confirmation")
        hidden_accumulation_intensity_D = self._get_safe_series(df, 'hidden_accumulation_intensity_D', method_name="_diagnose_axiom_pullback_confirmation")
        absorption_strength_index_D = self._get_safe_series(df, 'absorption_strength_index_D', method_name="_diagnose_axiom_pullback_confirmation")
        upper_shadow_selling_pressure_D = self._get_safe_series(df, 'upper_shadow_selling_pressure_D', method_name="_diagnose_axiom_pullback_confirmation")
        lower_shadow_absorption_strength_D = self._get_safe_series(df, 'lower_shadow_absorption_strength_D', method_name="_diagnose_axiom_pullback_confirmation")
        winner_conviction_index_D = self._get_safe_series(df, 'winner_conviction_index_D', method_name="_diagnose_axiom_pullback_confirmation")
        main_force_control_leverage_D = self._get_safe_series(df, 'main_force_control_leverage_D', method_name="_diagnose_axiom_pullback_confirmation")
        # 新增获取结构趋势形态分数
        score_struct_axiom_trend_form = self._get_safe_series(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', method_name="_diagnose_axiom_pullback_confirmation")
        # 检查所有必要的列是否存在
        required_cols = [
            'close_D', 'high_D', 'pct_change_D', 'volume_D', 'VOL_MA_5_D', 'VOL_MA_21_D',
            'main_force_net_flow_calibrated_D', 'SLOPE_5_short_term_concentration_90pct_D', # 修正列名
            'large_order_pressure_D', 'large_order_support_D', 'hidden_accumulation_intensity_D',
            'absorption_strength_index_D', 'upper_shadow_selling_pressure_D', 'lower_shadow_absorption_strength_D',
            'winner_conviction_index_D', 'main_force_control_leverage_D',
            'SCORE_STRUCT_AXIOM_TREND_FORM' # 新增检查
        ]
        for col in required_cols:
            if col not in df.columns:
                print(f"    -> [形态情报警告] 缺少核心高级指标 '{col}'，无法诊断回踩确认二次启动形态。")
                return pullback_confirmation_score
        max_vol_ma = pd.concat([vol_ma5_D, vol_ma21_D], axis=1).max(axis=1)
        for i in range(n_pre_A, len(df_index)):
            day_A_idx = i
            pre_A_slice_start = day_A_idx - n_pre_A
            if pre_A_slice_start < 0:
                continue
            # 1. A日之前N天量能萎缩
            pre_A_volume_atrophy = (volume_D.iloc[pre_A_slice_start:day_A_idx] < max_vol_ma.iloc[pre_A_slice_start:day_A_idx]).all()
            if not pre_A_volume_atrophy:
                continue
            # 2. A日放量上涨突破 (强化A日条件)
            day_A_pct_change = pct_change_D.iloc[day_A_idx]
            day_A_volume = volume_D.iloc[day_A_idx]
            day_A_max_vol_ma = max_vol_ma.iloc[day_A_idx]
            day_A_main_force_flow = main_force_net_flow_calibrated_D.iloc[day_A_idx]
            day_A_chip_conc_slope = short_term_concentration_90pct_D_slope_5d.iloc[day_A_idx]
            day_A_close = close_D.iloc[day_A_idx] # 获取A日收盘价
            day_A_trend_form_score = score_struct_axiom_trend_form.iloc[day_A_idx] # 获取A日结构趋势分
            if not (day_A_pct_change > 0.01 and day_A_volume > day_A_max_vol_ma * 1.2 and day_A_main_force_flow > 0 and day_A_chip_conc_slope > 0): # A日涨幅>1%，放量1.2倍，主力净流入，筹码集中度上升
                continue
            # 3. 寻找B日
            for j in range(day_A_idx + 1, min(day_A_idx + n_pullback_max + 2, len(df_index))):
                day_B_idx = j
                # 确保B日不是A日
                if day_B_idx <= day_A_idx:
                    continue
                # 4. B日再次放量上涨突破 (强化B日条件)
                day_B_pct_change = pct_change_D.iloc[day_B_idx]
                day_B_volume = volume_D.iloc[day_B_idx]
                day_B_max_vol_ma = max_vol_ma.iloc[day_B_idx]
                day_B_main_force_flow = main_force_net_flow_calibrated_D.iloc[day_B_idx]
                day_B_chip_conc_slope = short_term_concentration_90pct_D_slope_5d.iloc[day_B_idx]
                day_B_mf_control_leverage = main_force_control_leverage_D.iloc[day_B_idx]
                day_B_close = close_D.iloc[day_B_idx] # 获取B日收盘价
                day_B_trend_form_score = score_struct_axiom_trend_form.iloc[day_B_idx] # 获取B日结构趋势分
                if not (day_B_pct_change > 0.01 and day_B_volume > day_B_max_vol_ma * 1.2 and day_B_main_force_flow > 0 and day_B_chip_conc_slope > 0 and day_B_mf_control_leverage > 0): # B日涨幅>1%，放量1.2倍，主力净流入，筹码集中度上升，主力控盘杠杆为正
                    continue
                # 5. 细化B点与A点的关系：B点收盘高于A点收盘，且B点结构趋势相对于A点向上
                if not (day_B_close > day_A_close and day_B_trend_form_score > day_A_trend_form_score):
                    continue
                # 6. A到B日之间缩量回调或窄幅震荡，且健康洗盘
                pullback_slice_start = day_A_idx + 1
                pullback_slice_end = day_B_idx
                if pullback_slice_start >= pullback_slice_end: # 回调期至少一天
                    continue
                pullback_volume = volume_D.iloc[pullback_slice_start:pullback_slice_end]
                pullback_max_vol_ma = max_vol_ma.iloc[pullback_slice_start:pullback_slice_end]
                pullback_pct_change = pct_change_D.iloc[pullback_slice_start:pullback_slice_end]
                # 基础回调条件：缩量且无放量下跌
                volume_shrunk_during_pullback = (pullback_volume < pullback_max_vol_ma).all()
                no_bearish_volume_breakout = ~((pullback_pct_change < 0) & (pullback_volume > pullback_max_vol_ma)).any()
                if not (volume_shrunk_during_pullback and no_bearish_volume_breakout):
                    continue
                # 深度博弈：健康洗盘特征
                # 量化健康洗盘分数，范围 [0, 1]
                health_pullback_score = 0.0
                # 微观资金流：大单压制低，支撑强，隐蔽吸筹强
                # 确保切片有效，避免空切片导致 .all() 或 .any() 错误
                if not large_order_pressure_D.iloc[pullback_slice_start:pullback_slice_end].empty and \
                   (large_order_pressure_D.iloc[pullback_slice_start:pullback_slice_end] < 0.3).all(): # 大单压制低
                    health_pullback_score += 0.2
                if not large_order_support_D.iloc[pullback_slice_start:pullback_slice_end].empty and \
                   (large_order_support_D.iloc[pullback_slice_start:pullback_slice_end] > 0.7).all(): # 大单支撑强
                    health_pullback_score += 0.2
                if not hidden_accumulation_intensity_D.iloc[pullback_slice_start:pullback_slice_end].empty and \
                   (hidden_accumulation_intensity_D.iloc[pullback_slice_start:pullback_slice_end] > 0.5).any(): # 期间有隐蔽吸筹
                    health_pullback_score += 0.1
                # 筹码结构：获利盘信念稳定，上影线抛压低，下影线承接强
                if not winner_conviction_index_D.iloc[pullback_slice_start:pullback_slice_end].empty and \
                   (winner_conviction_index_D.iloc[pullback_slice_start:pullback_slice_end] > 0).all(): # 获利盘信念为正
                    health_pullback_score += 0.1
                if not upper_shadow_selling_pressure_D.iloc[pullback_slice_start:pullback_slice_end].empty and \
                   (upper_shadow_selling_pressure_D.iloc[pullback_slice_start:pullback_slice_end] < 0.3).all(): # 上影线抛压低
                    health_pullback_score += 0.1
                if not lower_shadow_absorption_strength_D.iloc[pullback_slice_start:pullback_slice_end].empty and \
                   (lower_shadow_absorption_strength_D.iloc[pullback_slice_start:pullback_slice_end] > 0.5).any(): # 期间有下影线承接
                    health_pullback_score += 0.1
                # 确保至少有部分健康洗盘证据
                if health_pullback_score > 0.3: # 设定一个阈值，表示健康洗盘的最低要求
                    pullback_confirmation_score.iloc[day_B_idx] = 1.0
                    break # 找到一个B日就跳出内层循环，寻找下一个A日
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df.index:
                print(f"    -> [回踩确认二次启动探针] @ {probe_date_for_loop.date()}:")
                print(f"       - SCORE_PATTERN_PULLBACK_CONFIRMATION: {pullback_confirmation_score.loc[probe_date_for_loop]:.4f}")
        return pullback_confirmation_score.astype(np.float32)

    def _diagnose_axiom_duofangpao(self, df: pd.DataFrame) -> pd.Series:
        """
        【V1.0】形态公理五：诊断“多方炮”形态
        - 核心逻辑: 识别经典的“多方炮”K线组合。
        - 信号输出: 在形态的第三根K线日输出1.0；否则输出0.0。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        """
        df_index = df.index
        duofangpao_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        # 获取核心K线数据
        open_D = self._get_safe_series(df, 'open_D', method_name="_diagnose_axiom_duofangpao")
        close_D = self._get_safe_series(df, 'close_D', method_name="_diagnose_axiom_duofangpao")
        high_D = self._get_safe_series(df, 'high_D', method_name="_diagnose_axiom_duofangpao")
        low_D = self._get_safe_series(df, 'low_D', method_name="_diagnose_axiom_duofangpao")
        volume_D = self._get_safe_series(df, 'volume_D', method_name="_diagnose_axiom_duofangpao")
        vol_ma5_D = self._get_safe_series(df, 'VOL_MA_5_D', method_name="_diagnose_axiom_duofangpao")
        # 检查所有必要的列是否存在
        required_cols = ['open_D', 'close_D', 'high_D', 'low_D', 'volume_D', 'VOL_MA_5_D']
        for col in required_cols:
            if col not in df.columns:
                print(f"    -> [形态情报警告] 缺少核心K线数据 '{col}'，无法诊断多方炮形态。")
                return duofangpao_score
        for i in range(2, len(df_index)): # 从第三根K线开始遍历
            # K线1 (i-2日)
            k1_open, k1_close, k1_high, k1_low, k1_volume = open_D.iloc[i-2], close_D.iloc[i-2], high_D.iloc[i-2], low_D.iloc[i-2], volume_D.iloc[i-2]
            k1_vol_ma5 = vol_ma5_D.iloc[i-2]
            # K线2 (i-1日)
            k2_open, k2_close, k2_high, k2_low, k2_volume = open_D.iloc[i-1], close_D.iloc[i-1], high_D.iloc[i-1], low_D.iloc[i-1], volume_D.iloc[i-1]
            k2_vol_ma5 = vol_ma5_D.iloc[i-1]
            # K线3 (i日)
            k3_open, k3_close, k3_high, k3_low, k3_volume = open_D.iloc[i], close_D.iloc[i], high_D.iloc[i], low_D.iloc[i], volume_D.iloc[i]
            k3_vol_ma5 = vol_ma5_D.iloc[i]
            # 条件1: K线1为放量阳线
            cond1_price = k1_close > k1_open # 阳线
            cond1_volume = k1_volume > k1_vol_ma5 * 1.2 # 放量 (1.2倍均量)
            if not (cond1_price and cond1_volume):
                continue
            # 条件2: K线2为缩量阴线或小实体K线，且在K线1范围内
            cond2_volume = k2_volume < k2_vol_ma5 * 0.8 # 缩量 (0.8倍均量)
            cond2_body_small = abs(k2_close - k2_open) / (k2_high - k2_low + 1e-9) < 0.5 # 小实体 (实体小于K线总长度的50%)
            cond2_within_k1_range = k2_low >= k1_low and k2_high <= k1_high # 在K线1范围内
            if not (cond2_volume and cond2_body_small and cond2_within_k1_range):
                continue
            # 条件3: K线3为放量阳线，且收盘价高于K线1收盘价
            cond3_price = k3_close > k3_open # 阳线
            cond3_volume = k3_volume > k3_vol_ma5 * 1.2 # 放量 (1.2倍均量)
            cond3_close_higher_than_k1 = k3_close > k1_close # 收盘价高于K线1收盘价
            if cond3_price and cond3_volume and cond3_close_higher_than_k1:
                duofangpao_score.iloc[i] = 1.0
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df.index:
                print(f"    -> [多方炮探针] @ {probe_date_for_loop.date()}:")
                print(f"       - SCORE_PATTERN_DUOFANGPAO: {duofangpao_score.loc[probe_date_for_loop]:.4f}")
        return duofangpao_score.astype(np.float32)















