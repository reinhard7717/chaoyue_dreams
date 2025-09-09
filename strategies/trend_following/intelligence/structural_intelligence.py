# 文件: strategies/trend_following/intelligence/structural_intelligence.py
# 结构情报模块 (均线, 箱体, 平台, 趋势)
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following.utils import get_params_block, get_param_value, create_persistent_state

class StructuralIntelligence:
    def __init__(self, strategy_instance, dynamic_thresholds: Dict):
        """
        初始化结构情报模块。
        :param strategy_instance: 策略主实例的引用。
        :param dynamic_thresholds: 动态阈值字典。
        """
        self.strategy = strategy_instance
        self.dynamic_thresholds = dynamic_thresholds

    def _fuse_multi_level_scores(self, df: pd.DataFrame, base_name: str, weights: Dict[str, float] = None) -> pd.Series:
        """
        【V1.1 健壮性修复版】融合S/A/B等多层置信度分数的辅助函数。
        - 核心职责: 从 CognitiveIntelligence 迁移而来，用于在模块内部融合多级信号。
        - 核心修复: 增加了对 df 参数的接收，并使用 df.index 来初始化和对齐Series，
                      确保在处理数据切片时索引正确，避免因索引不匹配导致的NaN或空Series问题。
        """
        # 传入df参数，保证索引正确
        if weights is None:
            weights = {'S': 1.0, 'A': 0.6, 'B': 0.3}
        
        # 使用传入的df.index确保索引长度正确，这是修复问题的关键
        total_score = pd.Series(0.0, index=df.index)
        total_weight = 0.0
        
        for level, weight in weights.items():
            score_name = f"SCORE_{base_name}_{level}"
            if score_name in self.strategy.atomic_states:
                score_series = self.strategy.atomic_states[score_name]
                # 使用reindex安全地对齐和相加
                total_score += score_series.reindex(df.index).fillna(0.0) * weight
                total_weight += weight
        
        if total_weight == 0:
            single_score_name = f"SCORE_{base_name}"
            if single_score_name in self.strategy.atomic_states:
                # 同样使用reindex保证安全
                return self.strategy.atomic_states[single_score_name].reindex(df.index).fillna(0.5)
            # 使用传入的df.index确保索引长度正确
            return pd.Series(0.5, index=df.index)
            
        return (total_score / total_weight).clip(0, 1)

    def diagnose_ma_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V8.0 最终数值化版】均线状态诊断引擎
        - 核心职责: 构建一个完全基于连续数值评分的诊断系统。
        - 核心升级 (本次修改):
          - [最终数值化] 废除了所有基于布尔逻辑和计数的信号（如OPP_MA_BULLISH_RESONANCE_B/A/S），
                        将其全面升级为基于归一化强度分的加权评分体系。
          - 新信号体系: 生成`SCORE_MA_BULLISH_RESONANCE`等数值分，更精确地量化共振与反转的强度。
        """
        # print("        -> [诊断模块 V8.0 最终数值化版] 启动...") 
        states = {}
        p = get_params_block(self.strategy, 'multi_dim_ma_params')
        if not get_param_value(p.get('enabled'), True): return {}
        # --- 军备检查 (Arsenal Check) ---
        ma_periods = get_param_value(p.get('ma_periods'), [5, 13, 21, 55])
        ema_cols = {period: f'EMA_{period}_D' for period in ma_periods}
        slope_cols = {period: f'SLOPE_{period}_EMA_{period}_D' if period > 5 else f'SLOPE_5_EMA_{period}_D' for period in ma_periods}
        accel_cols = {period: f'ACCEL_{period}_EMA_{period}_D' if period > 5 else f'ACCEL_5_EMA_{period}_D' for period in ma_periods}
        required_cols = list(ema_cols.values()) + list(slope_cols.values()) + list(accel_cols.values())
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"          -> [错误] 均线诊断引擎缺少必要列: {missing_cols}，跳过。")
            return {}
        # --- Part 1: 核心要素数值化 (Fundamental Conditions Scoring) --- 
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        def normalize(series, ascending=True):
            if series is None: return pd.Series(0.5, index=df.index)
            rank = series.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
            return rank if ascending else 1 - rank
        # 为每个周期的斜率和加速度创建归一化评分
        score_slope_up = {period: normalize(df[slope_cols[period]].clip(lower=0)) for period in ma_periods}
        score_slope_down = {period: normalize(df[slope_cols[period]].clip(upper=0), ascending=False) for period in ma_periods}
        score_accel_up = {period: normalize(df[accel_cols[period]].clip(lower=0)) for period in ma_periods}
        score_accel_down = {period: normalize(df[accel_cols[period]].clip(upper=0), ascending=False) for period in ma_periods}
        # --- Part 2: 共振信号数值化 (Resonance Scoring) --- # 重构共振信号
        # 2.1 上升共振 (Bullish Resonance)
        avg_bullish_slope = pd.Series(np.mean([s.values for s in score_slope_up.values()], axis=0), index=df.index)
        avg_bullish_accel = pd.Series(np.mean([s.values for s in score_accel_up.values()], axis=0), index=df.index)
        states['SCORE_MA_BULLISH_RESONANCE_B'] = avg_bullish_slope.astype(np.float32)
        states['SCORE_MA_BULLISH_RESONANCE_A'] = (avg_bullish_slope * avg_bullish_accel).astype(np.float32)
        # 2.2 下跌共振 (Bearish Resonance)
        avg_bearish_slope = pd.Series(np.mean([s.values for s in score_slope_down.values()], axis=0), index=df.index)
        avg_bearish_accel = pd.Series(np.mean([s.values for s in score_accel_down.values()], axis=0), index=df.index)
        states['SCORE_MA_BEARISH_RESONANCE_B'] = avg_bearish_slope.astype(np.float32)
        states['SCORE_MA_BEARISH_RESONANCE_A'] = (avg_bearish_slope * avg_bearish_accel).astype(np.float32)
        # --- Part 3: 反转信号数值化 (Reversal Scoring) --- # 重构反转信号
        # 3.1 底部反转 (Bottom Reversal)
        setup_bottom = score_slope_down[55] # 环境分: 长期趋势向下
        trigger_b_bottom = score_accel_up[5] # B级触发: 短期加速
        trigger_a_bottom = trigger_b_bottom * score_slope_up[5] # A级触发: 短期加速且斜率转正
        trigger_s_bottom = trigger_a_bottom * np.maximum(score_accel_up[13], score_slope_up[13]) # S级触发: 中期也出现拐点
        states['SCORE_MA_BOTTOM_REVERSAL_B'] = (setup_bottom * trigger_b_bottom).astype(np.float32)
        states['SCORE_MA_BOTTOM_REVERSAL_A'] = (setup_bottom * trigger_a_bottom).astype(np.float32)
        states['SCORE_MA_BOTTOM_REVERSAL_S'] = (setup_bottom * trigger_s_bottom).astype(np.float32)
        # 3.2 顶部反转 (Top Reversal)
        setup_top = score_slope_up[55] # 环境分: 长期趋势向上
        trigger_b_top = score_accel_down[5] # B级触发: 短期减速
        trigger_a_top = trigger_b_top * score_slope_down[5] # A级触发: 短期减速且斜率转负
        trigger_s_top = trigger_a_top * np.maximum(score_accel_down[13], score_slope_down[13]) # S级触发: 中期也出现拐点
        states['SCORE_MA_TOP_REVERSAL_B'] = (setup_top * trigger_b_top).astype(np.float32)
        states['SCORE_MA_TOP_REVERSAL_A'] = (setup_top * trigger_a_top).astype(np.float32)
        states['SCORE_MA_TOP_REVERSAL_S'] = (setup_top * trigger_s_top).astype(np.float32)
        # --- Part 4: [保留并优化] 高级数值化评分体系 --- # S级共振分与总分
        short_ma_cols_list = [ema_cols[period] for period in ma_periods[:-1]]
        long_ma_cols_list = [ema_cols[period] for period in ma_periods[1:]]
        alignment_sum = np.sum(df[short_ma_cols_list].values > df[long_ma_cols_list].values, axis=1)
        static_alignment_score = alignment_sum / (len(ma_periods) - 1)
        states['SCORE_MA_STATIC_ALIGNMENT'] = pd.Series(static_alignment_score, index=df.index, dtype=np.float32)
        # 复用Part 2计算的均值
        states['SCORE_MA_DYN_RESONANCE'] = avg_bullish_slope.astype(np.float32)
        states['SCORE_MA_ACCEL_RESONANCE'] = avg_bullish_accel.astype(np.float32)
        # 定义S级共振分，作为最高质量的共振信号
        states['SCORE_MA_BULLISH_RESONANCE_S'] = (states['SCORE_MA_BULLISH_RESONANCE_A'] * states['SCORE_MA_STATIC_ALIGNMENT']).astype(np.float32)
        states['SCORE_MA_BEARISH_RESONANCE_S'] = (states['SCORE_MA_BEARISH_RESONANCE_A'] * (1 - states['SCORE_MA_STATIC_ALIGNMENT'])).astype(np.float32)
        long_term_strength = score_slope_up[ma_periods[-1]]
        short_term_weakness = score_slope_down[ma_periods[0]]
        states['SCORE_MA_DIVERGENCE_RISK'] = (long_term_strength * short_term_weakness).fillna(0.5).astype(np.float32)
        states['SCORE_MA_HEALTH'] = (
            states['SCORE_MA_STATIC_ALIGNMENT'] *
            states['SCORE_MA_DYN_RESONANCE'] *
            states['SCORE_MA_ACCEL_RESONANCE']
        ).astype(np.float32)
        print("        -> [诊断模块 V8.0 最终数值化版] 分析完毕。") 
        return states

    def diagnose_box_states_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V4.2 机会逻辑修正版】箱体突破质量评分引擎
        - 核心重构: 引入“战备质量(Setup)”与“点火质量(Trigger)”分离式评估体系。
        - 核心逻辑: 最终突破分 = (战备质量分) * (点火质量分)，实现优中选优。
        - 核心升级 (本次修改):
          - [逻辑修正] 移除了对 `is_breakout` 单日事件布尔信号的硬性依赖。
                        现在，突破机会分数直接由“昨日战备质量”与“今日点火强度”决定，
                        使其从一个稀疏的“事件”信号转变为更连续、更具前瞻性的“机会”信号，
                        与其他突破信号的计算范式保持一致。
        """
        # print("        -> [箱体诊断模块 V4.2 机会逻辑修正版] 启动...")
        states = {}
        p = get_params_block(self.strategy, 'box_state_params')
        if not get_param_value(p.get('enabled'), True) or df.empty:
            return states
        # --- 1. 军备检查与参数获取 ---
        atomic = self.strategy.atomic_states
        required_cols = [
            'high_D', 'low_D', 'close_D', 'open_D', 'volume_D',
            'BBW_21_2.0_D', 'SLOPE_5_CMF_21_D', 'SLOPE_5_volume_D', 'ACCEL_5_volume_D'
        ]
        required_signals = [
            'SCORE_MA_BULLISH_RESONANCE_S', 'SCORE_MA_BULLISH_RESONANCE_A', 'SCORE_MA_BULLISH_RESONANCE_B',
            'SCORE_MA_BEARISH_RESONANCE_S', 'SCORE_MA_BEARISH_RESONANCE_A', 'SCORE_MA_BEARISH_RESONANCE_B'
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        missing_signals = [s for s in required_signals if s not in atomic]
        if missing_cols or missing_signals:
            print(f"          -> [警告] 箱体诊断缺少关键数据: 列{missing_cols}, 信号{missing_signals}。模块已跳过。")
            return {}
        high_d, low_d, close_d, open_d, volume_d = df['high_D'], df['low_D'], df['close_D'], df['open_D'], df['volume_D']
        lookback_window = get_param_value(p.get('lookback_window'), 8)
        norm_window = get_param_value(p.get('norm_window'), 60)
        min_periods = max(1, norm_window // 5)
        # --- 2. 计算基础箱体和布尔突破/跌破事件 ---
        box_top = high_d.rolling(window=lookback_window).max()
        box_bottom = low_d.rolling(window=lookback_window).min()
        amplitude_ratio = (box_top - box_bottom) / box_bottom.replace(0, np.nan)
        is_valid_box = (amplitude_ratio < get_param_value(p.get('max_amplitude_ratio'), 0.05)).fillna(False)
        df['box_top_D'] = box_top
        df['box_bottom_D'] = box_bottom
        is_breakout = is_valid_box & (close_d > box_top.shift(1)) & (close_d.shift(1) <= box_top.shift(1))
        is_breakdown = is_valid_box & (close_d < box_bottom.shift(1)) & (close_d.shift(1) >= box_bottom.shift(1))
        # --- 3. 战备质量评分 (Setup Score) ---
        vol_compression_score = 1 - df['BBW_21_2.0_D'].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        capital_inflow_score = df['SLOPE_5_CMF_21_D'].rolling(window=lookback_window).mean().rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        setup_quality_score = (vol_compression_score * capital_inflow_score).where(is_valid_box, 0.0)
        states['SCORE_BOX_SETUP_QUALITY'] = setup_quality_score.astype(np.float32)
        # --- 4. 点火质量评分 (Trigger Score) ---
        vol_slope_score = df['SLOPE_5_volume_D'].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        vol_accel_score = df['ACCEL_5_volume_D'].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        volume_thrust_score = vol_slope_score * vol_accel_score
        candle_range = (high_d - low_d).replace(0, np.nan)
        breakout_candle_score = ((close_d - open_d) / candle_range).clip(0, 1).fillna(0.0)
        breakdown_candle_score = ((open_d - close_d) / candle_range).clip(0, 1).fillna(0.0)
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        bullish_resonance_s = atomic.get('SCORE_MA_BULLISH_RESONANCE_S', default_score)
        bullish_resonance_a = atomic.get('SCORE_MA_BULLISH_RESONANCE_A', default_score)
        bullish_resonance_b = atomic.get('SCORE_MA_BULLISH_RESONANCE_B', default_score)
        bearish_resonance_s = atomic.get('SCORE_MA_BEARISH_RESONANCE_S', default_score)
        bearish_resonance_a = atomic.get('SCORE_MA_BEARISH_RESONANCE_A', default_score)
        bearish_resonance_b = atomic.get('SCORE_MA_BEARISH_RESONANCE_B', default_score)
        # --- 5. 融合生成最终质量分与兼容性信号 ---
        # 5.1 向上突破
        breakout_trigger_base_score = volume_thrust_score * breakout_candle_score
        states['SCORE_BOX_BREAKOUT_S'] = (setup_quality_score.shift(1) * breakout_trigger_base_score * bullish_resonance_s).fillna(0.0).astype(np.float32)
        states['SCORE_BOX_BREAKOUT_A'] = (setup_quality_score.shift(1) * breakout_trigger_base_score * bullish_resonance_a).fillna(0.0).astype(np.float32)
        states['SCORE_BOX_BREAKOUT_B'] = (setup_quality_score.shift(1) * breakout_trigger_base_score * bullish_resonance_b).fillna(0.0).astype(np.float32)
        # 5.2 向下突破 (保持事件驱动逻辑，因为风险信号需要更精确)
        breakdown_trigger_base_score = volume_thrust_score * breakdown_candle_score
        states['SCORE_BOX_BREAKDOWN_S'] = (is_breakdown * setup_quality_score.shift(1) * breakdown_trigger_base_score * bearish_resonance_s).fillna(0.0).astype(np.float32)
        states['SCORE_BOX_BREAKDOWN_A'] = (is_breakdown * setup_quality_score.shift(1) * breakdown_trigger_base_score * bearish_resonance_a).fillna(0.0).astype(np.float32)
        states['SCORE_BOX_BREAKDOWN_B'] = (is_breakdown * setup_quality_score.shift(1) * breakdown_trigger_base_score * bearish_resonance_b).fillna(0.0).astype(np.float32)
        # 5.3 兼容旧版信号 (保持不变)
        states['BOX_EVENT_BREAKOUT'] = is_breakout.fillna(False)
        states['BOX_EVENT_BREAKDOWN'] = is_breakdown.fillna(False)
        print("        -> [箱体诊断模块 V4.2 机会逻辑修正版] 诊断完毕。")
        return states

    def diagnose_platform_states_scores(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        【V3.1 信号输出修正版】平台诊断与风险评分引擎
        - 核心重构: (V3.0) 从评估“静态稳定性”升级为诊断“动态蓄势状态”。
        - 核心逻辑: (V3.0) 平台总质量分 = f(稳定性, 成本动能, 成本加速度, 宏观环境)。
        - 本次升级: 【修正】将内部计算的“成本动能分”与“成本加速分”正确输出到 states 字典，
                    使其可供下游模块（如认知层的战术诊断）消费。
        """
        # print("        -> [诊断模块 V3.1 信号输出修正版] 启动...") 
        states = {}
        p = get_params_block(self.strategy, 'platform_state_params')
        if not get_param_value(p.get('enabled'), True): return df, {}
        # --- 1. 军备检查 (Arsenal Check) ---
        atomic = self.strategy.atomic_states
        cost_periods = get_param_value(p.get('cost_dynamic_periods'), [5, 21, 55])
        required_cols = ['peak_cost_D', 'close_D', 'open_D', 'high_D', 'low_D', 'volume_D']
        for period in cost_periods:
            required_cols.extend([
                f'SLOPE_{period}_peak_cost_D',
                f'ACCEL_{period if period > 5 else 5}_peak_cost_D' # 确保有对应的加速度列
            ])
        required_signals = ['SCORE_MA_HEALTH'] # 依赖上游均线健康分
        missing_cols = [col for col in required_cols if col not in df.columns]
        missing_signals = [s for s in required_signals if s not in atomic]
        if missing_cols or missing_signals:
            print(f"          -> [警告] 平台诊断缺少关键数据: 列{missing_cols}, 信号{missing_signals}。模块已跳过。")
            # 返回安全默认值
            df['PLATFORM_PRICE_STABLE'] = np.nan
            states['PLATFORM_STATE_STABLE_FORMED'] = pd.Series(False, index=df.index)
            states['SCORE_RISK_PLATFORM_BROKEN_S'] = pd.Series(0.0, index=df.index, dtype=np.float32)
            return df, states
        # --- 2. 计算四维核心评分组件 ---
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        # 2.1 成本稳定性分 (Stability Score)
        peak_cost = df['peak_cost_D']
        rolling_cost = peak_cost.rolling(get_param_value(p.get('cost_cv_lookback'), 5))
        with np.errstate(divide='ignore', invalid='ignore'):
            coeff_of_variation = (rolling_cost.std() / rolling_cost.mean()).fillna(1.0)
        cv_rank = coeff_of_variation.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        cost_stability_score = (1 - cv_rank)
        # 2.2 成本动能分 (Momentum Score) 
        cost_slope_series = [
            df[f'SLOPE_{p}_peak_cost_D'].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
            for p in cost_periods
        ]
        cost_momentum_score = pd.Series(np.mean(np.array([s.values for s in cost_slope_series]), axis=0), index=df.index)
        states['SCORE_PLATFORM_COST_MOMENTUM'] = cost_momentum_score.astype(np.float32) # 将成本动能分添加到输出
        # 2.3 成本加速分 (Acceleration Score) 
        cost_accel_series = [
            df[f'ACCEL_{p if p > 5 else 5}_peak_cost_D'].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
            for p in cost_periods
        ]
        cost_accel_score = pd.Series(np.mean(np.array([s.values for s in cost_accel_series]), axis=0), index=df.index)
        states['SCORE_PLATFORM_COST_ACCEL'] = cost_accel_score.astype(np.float32) # 将成本加速分添加到输出
        # 2.4 宏观环境分 (Context Score) 
        context_health_score = atomic.get('SCORE_MA_HEALTH', pd.Series(0.5, index=df.index))
        # --- 3. 融合生成B/A/S三级平台质量分 ---
        states['SCORE_PLATFORM_QUALITY_B'] = cost_stability_score.astype(np.float32)
        states['SCORE_PLATFORM_QUALITY_A'] = (cost_stability_score * cost_momentum_score).astype(np.float32)
        states['SCORE_PLATFORM_QUALITY_S'] = (
            cost_stability_score * cost_momentum_score * cost_accel_score * context_health_score
        ).fillna(0.0).astype(np.float32)
        # --- 4. 生成兼容旧版的布尔信号与平台价格 ---
        threshold = get_param_value(p.get('final_score_threshold_for_bool'), 0.7)
        # 使用最高质量的S级分数来定义最可靠的平台
        stable_formed_series = states['SCORE_PLATFORM_QUALITY_S'] > threshold
        states['PLATFORM_STATE_STABLE_FORMED'] = stable_formed_series
        states['STRUCTURE_BOX_ACCUMULATION_A'] = stable_formed_series # 兼容信号
        df['PLATFORM_PRICE_STABLE'] = peak_cost.where(stable_formed_series)
        # --- 5. 平台破位风险评分 ---
        was_on_platform = stable_formed_series.shift(1, fill_value=False)
        is_breaking_down = df['close_D'] < df['PLATFORM_PRICE_STABLE'].ffill().shift(1)
        platform_failure_series = was_on_platform & is_breaking_down
        states['STRUCTURE_PLATFORM_BROKEN'] = platform_failure_series
        # 5.1 计算破位强度分
        candle_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        breakdown_candle_score = ((df['open_D'] - df['close_D']) / candle_range).clip(0, 1).fillna(0.0)
        volume_score = df['volume_D'].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        breakdown_intensity_score = breakdown_candle_score * volume_score
        # 5.2 融合生成最终风险分
        platform_quality_yesterday = states['SCORE_PLATFORM_QUALITY_S'].shift(1).fillna(0.0)
        states['SCORE_RISK_PLATFORM_BROKEN_S'] = (
            platform_quality_yesterday * breakdown_intensity_score
        ).where(platform_failure_series, 0.0).astype(np.float32)
        # --- 6. 平台向上突破机会评分 ---
        is_breaking_up = df['close_D'] > df['PLATFORM_PRICE_STABLE'].ffill().shift(1) # 定义向上突破事件
        platform_breakout_series = was_on_platform & is_breaking_up # 必须是昨日在平台上，今日突破
        states['STRUCTURE_PLATFORM_BREAKOUT'] = platform_breakout_series # 生成布尔事件信号
        # 计算突破强度分
        breakout_candle_score = ((df['close_D'] - df['open_D']) / candle_range).clip(0, 1).fillna(0.0)
        breakout_intensity_score = breakout_candle_score * volume_score
        # 融合生成最终机会分 (结合昨日平台质量、今日突破强度、今日宏观环境)
        states['SCORE_OPP_PLATFORM_BREAKOUT_S'] = (
            platform_quality_yesterday * breakout_intensity_score * context_health_score
        ).where(platform_breakout_series, 0.0).astype(np.float32)
        print("        -> [诊断模块 V3.1 信号输出修正版] 分析完毕。")
        return df, states

    def diagnose_fibonacci_support(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V4.0 动态过程诊断版】斐波那契反攻诊断模块
        - 核心重构: 从“静态触碰”模型升级为“动态过程”诊断模型。
        - 核心逻辑: 机会分 = f(战备质量, 确认强度, 宏观环境)。
          - 战备质量 (Setup): 评估价格接近支撑位的速度(势能)与触碰日K线的拒绝强度。
          - 确认强度 (Confirmation): 消费上游的强力反转信号。
          - 宏观环境 (Context): 消费上游的趋势健康总分。
        - 新增信号 (数值型):
          - SCORE_FIB_REBOUND_S/A/B: S/A/B三级反弹机会分，对应不同重要性的斐波那契水平。
        """
        # print("        -> [斐波那契反攻诊断模块 V4.0 动态过程诊断版] 启动...")
        states = {}
        p = get_params_block(self.strategy, 'fibonacci_support_params')
        if not get_param_value(p.get('enabled'), True):
            return {}
        
        # --- 1. 军备检查 (Arsenal Check) ---
        atomic = self.strategy.atomic_states
        triggers = self.strategy.trigger_events
        fib_levels = {'S': 'FIB_0_618_D', 'A': 'FIB_0_500_D', 'B': 'FIB_0_382_D'}
        required_cols = list(fib_levels.values()) + ['low_D', 'high_D', 'close_D', 'open_D', 'SLOPE_5_close_D']
        required_signals = ['TRIGGER_DOMINANT_REVERSAL', 'SCORE_MA_HEALTH']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        missing_signals = [s for s in required_signals if s not in triggers and s not in atomic]
        
        if missing_cols or missing_signals:
            print(f"          -> [警告] 斐波那契诊断缺少关键数据: 列{missing_cols}, 信号{missing_signals}。模块跳过。")
            return {}

        # --- 2. 获取核心动态信号与参数 ---
        confirmation_score = triggers.get('TRIGGER_DOMINANT_REVERSAL', pd.Series(False, index=df.index)).astype(np.float32)
        context_score = atomic.get('SCORE_MA_HEALTH', pd.Series(0.0, index=df.index, dtype=np.float32))
        proximity_ratio = get_param_value(p.get('proximity_ratio'), 0.01)
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)

        # --- 3. 计算“战备质量分” (Setup Score) ---
        # 3.1 下跌速度分 (越快，势能越大，反弹潜力越高)
        approach_velocity_score = (1 - df['SLOPE_5_close_D'].rolling(window=norm_window, min_periods=min_periods).rank(pct=True)).fillna(0.5)
        
        # 3.2 K线拒绝强度分 (下影线越长，拒绝信号越强)
        candle_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        lower_wick_ratio = ((df[['open_D', 'close_D']].min(axis=1) - df['low_D']) / candle_range).clip(0, 1).fillna(0.0)
        rejection_quality_score = lower_wick_ratio

        # 3.3 融合为总的战备分
        setup_quality_score = (approach_velocity_score * rejection_quality_score)

        # --- 4. 核心逻辑：融合三段式评分 ---
        # 逻辑: 最终分数在“确认日”产生，但它依赖于“接触日”(前一日)的战备质量。
        def calculate_rebound_score(fib_level_col: str) -> pd.Series:
            fib_level = df[fib_level_col]
            # 定义接触日: 当日最低价触及或略微跌破斐波那契水平
            is_contact_today = (df['low_D'] <= fib_level * (1 + proximity_ratio)) & (df['high_D'] >= fib_level * (1 - proximity_ratio))
            
            # 获取接触日的战备质量分
            setup_score_on_contact_day = setup_quality_score.where(is_contact_today, 0.0)
            
            # 最终分数 = 前一日的战备质量 * 今日的确认强度 * 今日的宏观环境
            final_score = setup_score_on_contact_day.shift(1).fillna(0.0) * confirmation_score * context_score
            return final_score

        # --- 5. 为不同级别的支撑生成S/A/B三级分数 ---
        states['SCORE_FIB_REBOUND_S'] = calculate_rebound_score(fib_levels['S']).astype(np.float32)
        states['SCORE_FIB_REBOUND_A'] = calculate_rebound_score(fib_levels['A']).astype(np.float32)
        states['SCORE_FIB_REBOUND_B'] = calculate_rebound_score(fib_levels['B']).astype(np.float32)

        # --- 6. 兼容旧版信号 (可选，但建议保留平滑过渡) ---
        # 将S/A/B级分数合并，用于驱动旧的布尔信号
        max_rebound_score = np.maximum.reduce([
            states['SCORE_FIB_REBOUND_S'].values,
            states['SCORE_FIB_REBOUND_A'].values,
            states['SCORE_FIB_REBOUND_B'].values
        ])
        max_rebound_series = pd.Series(max_rebound_score, index=df.index)
        
        if (max_rebound_series > 0).any():
            print(f"          -> [情报] 侦测到 {(max_rebound_series > 0).sum()} 次斐波那契反弹机会，最高分: {max_rebound_series.max():.2f}。")

        # 为了兼容，可以保留旧的信号名，但其内容由新的数值化逻辑驱动
        states['SCORE_FIB_SUPPORT_GOLDEN_POCKET_S'] = states['SCORE_FIB_REBOUND_S']
        states['SCORE_FIB_SUPPORT_STANDARD_A'] = np.maximum(states['SCORE_FIB_REBOUND_A'], states['SCORE_FIB_REBOUND_B']).astype(np.float32)

        print("        -> [斐波那契反攻诊断模块 V4.0 动态过程诊断版] 分析完毕。") # 更新打印信息
        return states

    def diagnose_structural_mechanics_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V6.0 共振-反转对称诊断版】结构力学诊断引擎
        - 核心重构: 遵循“共振/反转”对称原则，全面升级为多维交叉验证的数值化评分体系。
        - 核心逻辑:
          - 共振信号 (多周期交叉): 评估成本、筹码、势能等要素在多周期上的一致性。
          - 反转信号 (同周期交叉): 评估“静态战备”与“动态点火”的结合。
        - 新增信号 (数值型, 对称设计):
          - SCORE_MECHANICS_BULLISH_RESONANCE_S/A/B: 上升共振机会分。
          - SCORE_MECHANICS_BEARISH_RESONANCE_S/A/B: 下跌共振风险分。
          - SCORE_MECHANICS_BOTTOM_REVERSAL_S/A/B: 底部反转机会分。
          - SCORE_MECHANICS_TOP_REVERSAL_S/A/B: 顶部反转风险分。
        """
        print("        -> [结构力学诊断引擎 V6.0 共振-反转对称诊断版] 启动...")
        states = {}
        p = get_params_block(self.strategy, 'structural_mechanics_params')
        if not get_param_value(p.get('enabled'), True): return {}

        # --- 1. 军备检查 (Arsenal Check) ---
        periods = get_param_value(p.get('dynamic_periods'), [5, 21, 55])
        required_cols = ['energy_ratio_D', 'BBW_21_2.0_D']
        dynamic_sources = {'cost': 'peak_cost_D', 'conc': 'concentration_90pct_D'}
        
        for period in periods:
            required_cols.extend([
                f'SLOPE_{period}_{dynamic_sources["cost"]}',
                f'ACCEL_{period if period > 5 else 5}_{dynamic_sources["cost"]}',
                f'SLOPE_{period}_{dynamic_sources["conc"]}',
            ])
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"          -> [严重警告] 结构力学引擎缺少关键数据列: {missing_cols}，模块已跳过！")
            return states

        # --- 2. 核心力学要素数值化 (归一化处理) ---
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)

        def normalize(series):
            return series.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)

        # 动态评分组件
        cost_momentum_scores = {p: normalize(df[f'SLOPE_{p}_{dynamic_sources["cost"]}']) for p in periods}
        cost_accel_scores = {p: normalize(df[f'ACCEL_{p if p > 5 else 5}_{dynamic_sources["cost"]}']) for p in periods}
        supply_lock_scores = {p: normalize(df[f'SLOPE_{p}_{dynamic_sources["conc"]}']) for p in periods} # 斜率越大，锁定度越高
        
        # 静态评分组件
        energy_advantage_score = normalize(df['energy_ratio_D'])
        vol_compression_score = 1 - normalize(df['BBW_21_2.0_D']) # BBW越小，压缩度越高

        # --- 3. 共振信号合成 (多时间周期交叉验证) ---
        avg_cost_momentum = pd.Series(np.mean(np.array([s.values for s in cost_momentum_scores.values()]), axis=0), index=df.index)
        avg_supply_lock = pd.Series(np.mean(np.array([s.values for s in supply_lock_scores.values()]), axis=0), index=df.index)

        # 3.1 上升共振 (Bullish Resonance)
        states['SCORE_MECHANICS_BULLISH_RESONANCE_B'] = avg_cost_momentum.astype(np.float32)
        states['SCORE_MECHANICS_BULLISH_RESONANCE_A'] = (avg_cost_momentum * avg_supply_lock).astype(np.float32)
        states['SCORE_MECHANICS_BULLISH_RESONANCE_S'] = (avg_cost_momentum * avg_supply_lock * energy_advantage_score).astype(np.float32)

        # 3.2 下跌共振 (Bearish Resonance) - 对称逻辑
        avg_cost_decline = 1 - avg_cost_momentum
        avg_supply_disperse = 1 - avg_supply_lock
        energy_disadvantage_score = 1 - energy_advantage_score
        states['SCORE_MECHANICS_BEARISH_RESONANCE_B'] = avg_cost_decline.astype(np.float32)
        states['SCORE_MECHANICS_BEARISH_RESONANCE_A'] = (avg_cost_decline * avg_supply_disperse).astype(np.float32)
        states['SCORE_MECHANICS_BEARISH_RESONANCE_S'] = (avg_cost_decline * avg_supply_disperse * energy_disadvantage_score).astype(np.float32)

        # --- 4. 反转信号合成 (同周期多维度交叉验证) ---
        avg_cost_accel = pd.Series(np.mean(np.array([s.values for s in cost_accel_scores.values()]), axis=0), index=df.index)

        # 4.1 底部反转 (Bottom Reversal) = 战备分 * 点火分
        bottom_reversal_setup_score = vol_compression_score * avg_supply_lock # 战备: 波动压缩 + 悄然吸筹
        bottom_reversal_trigger_score = avg_cost_accel # 点火: 成本重心开始加速抬升
        
        states['SCORE_MECHANICS_BOTTOM_REVERSAL_B'] = bottom_reversal_trigger_score.astype(np.float32)
        states['SCORE_MECHANICS_BOTTOM_REVERSAL_A'] = (vol_compression_score * bottom_reversal_trigger_score).astype(np.float32)
        states['SCORE_MECHANICS_BOTTOM_REVERSAL_S'] = (bottom_reversal_setup_score * bottom_reversal_trigger_score).astype(np.float32)

        # 4.2 顶部反转 (Top Reversal) - 对称逻辑
        top_reversal_setup_score = (1 - vol_compression_score) * avg_supply_disperse # 战备: 波动放大 + 悄然派发
        top_reversal_trigger_score = 1 - avg_cost_accel # 点火: 成本重心开始加速下移
        
        states['SCORE_MECHANICS_TOP_REVERSAL_B'] = top_reversal_trigger_score.astype(np.float32)
        states['SCORE_MECHANICS_TOP_REVERSAL_A'] = ((1 - vol_compression_score) * top_reversal_trigger_score).astype(np.float32)
        states['SCORE_MECHANICS_TOP_REVERSAL_S'] = (top_reversal_setup_score * top_reversal_trigger_score).astype(np.float32)

        print("        -> [结构力学诊断引擎 V6.0 共振-反转对称诊断版] 分析完毕。") # 更新打印信息
        return states

    def diagnose_mtf_trend_synergy_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V5.0 共振-反转对称诊断版】战略协同评分引擎
        - 核心重构: 遵循“共振/反转”对称原则，构建一个完整的跨时间框架(MTF)信号矩阵。
        - 核心逻辑:
          - 共振信号: 评估日线与周线在动能(slope)与势能(accel)上的一致性。
          - 反转信号: 评估“周线趋势(环境)”与“日线拐点(触发)”的组合。
        - 新增信号 (数值型, 对称设计):
          - SCORE_MTF_BULLISH_RESONANCE_S/A/B: MTF上升共振机会分。
          - SCORE_MTF_BEARISH_RESONANCE_S/A/B: MTF下跌共振风险分。
          - SCORE_MTF_BOTTOM_REVERSAL_S/A/B: MTF底部反转机会分。
          - SCORE_MTF_TOP_REVERSAL_S/A/B: MTF顶部反转风险分。
        """
        print("        -> [战略协同引擎 V5.0 共振-反转对称诊断版] 启动...")
        states = {}
        p = get_params_block(self.strategy, 'mtf_trend_synergy_params')
        if not get_param_value(p.get('enabled'), True): return {}
        
        # --- 1. 军备检查 (Arsenal Check) ---
        atomic = self.strategy.atomic_states
        required_daily_signals = ['SCORE_MA_DYN_RESONANCE', 'SCORE_MA_ACCEL_RESONANCE']
        # 动态发现所有可用的周线斜率和加速度列
        weekly_slope_cols = [col for col in df.columns if 'SLOPE' in col and col.endswith('_W')]
        weekly_accel_cols = [col for col in df.columns if 'ACCEL' in col and col.endswith('_W')]
        
        missing_signals = [s for s in required_daily_signals if s not in atomic]
        if missing_signals or not weekly_slope_cols or not weekly_accel_cols:
            print(f"          -> [严重警告] 战略协同引擎缺少关键数据: 日线信号{missing_signals}, 周线斜率列(需至少1个), 周线加速列(需至少1个)。模块已跳过！")
            return {}

        # --- 2. 获取/构建日线与周线的核心动态评分 ---
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        default_series = pd.Series(0.5, index=df.index, dtype=np.float32)

        # 2.1 获取日线高级分数
        daily_momentum_score = atomic.get('SCORE_MA_DYN_RESONANCE', default_series)
        daily_accel_score = atomic.get('SCORE_MA_ACCEL_RESONANCE', default_series)

        # 2.2 构建周线高级分数
        def get_weekly_score(cols):
            if not cols: return default_series
            series_list = [df[col].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5) for col in cols]
            score_values = np.mean(np.array([s.values for s in series_list]), axis=0)
            return pd.Series(score_values, index=df.index, dtype=np.float32)

        weekly_momentum_score = get_weekly_score(weekly_slope_cols)
        weekly_accel_score = get_weekly_score(weekly_accel_cols)

        # --- 3. 共振信号合成 (多时间周期交叉验证) ---
        # 3.1 上升共振 (Bullish Resonance)
        bullish_momentum_resonance = daily_momentum_score * weekly_momentum_score
        states['SCORE_MTF_BULLISH_RESONANCE_B'] = bullish_momentum_resonance.astype(np.float32)
        states['SCORE_MTF_BULLISH_RESONANCE_A'] = (bullish_momentum_resonance * daily_accel_score).astype(np.float32)
        states['SCORE_MTF_BULLISH_RESONANCE_S'] = (bullish_momentum_resonance * daily_accel_score * weekly_accel_score).astype(np.float32)

        # 3.2 下跌共振 (Bearish Resonance) - 对称逻辑
        bearish_momentum_resonance = (1 - daily_momentum_score) * (1 - weekly_momentum_score)
        states['SCORE_MTF_BEARISH_RESONANCE_B'] = bearish_momentum_resonance.astype(np.float32)
        states['SCORE_MTF_BEARISH_RESONANCE_A'] = (bearish_momentum_resonance * (1 - daily_accel_score)).astype(np.float32)
        states['SCORE_MTF_BEARISH_RESONANCE_S'] = (bearish_momentum_resonance * (1 - daily_accel_score) * (1 - weekly_accel_score)).astype(np.float32)

        # --- 4. 反转信号合成 (环境 x 拐点) ---
        # 4.1 底部反转 (Bottom Reversal) = 周线看跌环境 x 日线看涨拐点
        weekly_downtrend_setup = 1 - weekly_momentum_score
        daily_bullish_trigger = daily_accel_score
        bottom_reversal_base = weekly_downtrend_setup * daily_bullish_trigger
        
        states['SCORE_MTF_BOTTOM_REVERSAL_B'] = daily_bullish_trigger.astype(np.float32) # B级: 仅看日线拐点
        states['SCORE_MTF_BOTTOM_REVERSAL_A'] = bottom_reversal_base.astype(np.float32) # A级: 结合周线环境
        states['SCORE_MTF_BOTTOM_REVERSAL_S'] = (bottom_reversal_base * (1 - weekly_accel_score)).astype(np.float32) # S级: 周线下跌也在减速

        # 4.2 顶部反转 (Top Reversal) - 对称逻辑
        weekly_uptrend_setup = weekly_momentum_score
        daily_bearish_trigger = 1 - daily_accel_score
        top_reversal_base = weekly_uptrend_setup * daily_bearish_trigger

        states['SCORE_MTF_TOP_REVERSAL_B'] = daily_bearish_trigger.astype(np.float32) # B级: 仅看日线拐点
        states['SCORE_MTF_TOP_REVERSAL_A'] = top_reversal_base.astype(np.float32) # A级: 结合周线环境
        states['SCORE_MTF_TOP_REVERSAL_S'] = (top_reversal_base * weekly_accel_score).astype(np.float32) # S级: 周线上涨也在减速

        print("        -> [战略协同引擎 V5.0 共振-反转对称诊断版] 分析完毕。") # 更新打印信息
        return states

    def diagnose_structural_risks_and_regimes_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V4.1 数值化升级版】结构风险与市场状态诊断引擎
        - 核心重构: 遵循“风险共振”与“状态分类”原则，升级为多维交叉验证的数值化评分体系。
        - 核心升级 (本次修改):
          - [最终数值化] 将`CONTEXT_RISK_CHIP_STRUCTURE_DECAY`信号从布尔逻辑升级为数值评分。
        """
        print("        -> [结构风险与状态引擎 V4.1 数值化升级版] 启动...") 
        states = {}
        p = get_params_block(self.strategy, 'structural_risks_params')
        if not get_param_value(p.get('enabled'), True): return {}
        # --- 1. 军备检查 (Arsenal Check) ---
        required_cols = [
            'price_to_peak_ratio_D', 'SLOPE_5_price_to_peak_ratio_D',
            'winner_profit_margin_D', 'SLOPE_5_winner_profit_margin_D',
            'chip_health_score_D', 'SLOPE_5_chip_health_score_D', 'is_chip_fault_formed_D',
            'BBW_21_2.0_D', 'SLOPE_5_BBW_21_2.0_D', 'ACCEL_5_BBW_21_2.0_D',
            'SLOPE_5_concentration_90pct_D', 'SLOPE_21_concentration_90pct_D',
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"          -> [严重警告] 结构风险引擎缺少关键数据列: {missing_cols}，模块已跳过！")
            return states
        # --- 2. 核心要素数值化 (归一化处理) ---
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        def normalize(series):
            return series.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        price_deviation_static = normalize(df['price_to_peak_ratio_D'])
        price_deviation_dynamic = normalize(df['SLOPE_5_price_to_peak_ratio_D'])
        profit_exhaustion_static = normalize(df['winner_profit_margin_D'])
        profit_exhaustion_dynamic = normalize(df['SLOPE_5_winner_profit_margin_D'])
        structural_health_static = normalize(df['chip_health_score_D'])
        structural_health_dynamic = normalize(df['SLOPE_5_chip_health_score_D'])
        is_fault_formed = df['is_chip_fault_formed_D'].astype(float)
        vol_static = normalize(df['BBW_21_2.0_D'])
        vol_dynamic = normalize(df['SLOPE_5_BBW_21_2.0_D'])
        vol_accel = normalize(df['ACCEL_5_BBW_21_2.0_D'])
        # --- 3. 风险信号合成 (多维度风险共振) ---
        states['SCORE_RISK_PRICE_DEVIATION_B'] = price_deviation_static.astype(np.float32)
        states['SCORE_RISK_PRICE_DEVIATION_A'] = (price_deviation_static * price_deviation_dynamic).astype(np.float32)
        states['SCORE_RISK_PRICE_DEVIATION_S'] = (states['SCORE_RISK_PRICE_DEVIATION_A'] * vol_static * vol_dynamic).astype(np.float32)
        structural_deterioration = 1 - structural_health_dynamic
        states['SCORE_RISK_PROFIT_EXHAUSTION_B'] = profit_exhaustion_static.astype(np.float32)
        states['SCORE_RISK_PROFIT_EXHAUSTION_A'] = (profit_exhaustion_static * profit_exhaustion_dynamic).astype(np.float32)
        states['SCORE_RISK_PROFIT_EXHAUSTION_S'] = (states['SCORE_RISK_PROFIT_EXHAUSTION_A'] * structural_deterioration).astype(np.float32)
        structural_unhealth = 1 - structural_health_static
        states['SCORE_RISK_STRUCTURAL_FAULT_B'] = structural_unhealth.astype(np.float32)
        states['SCORE_RISK_STRUCTURAL_FAULT_A'] = (structural_unhealth * structural_deterioration).astype(np.float32)
        states['SCORE_RISK_STRUCTURAL_FAULT_S'] = (states['SCORE_RISK_STRUCTURAL_FAULT_A'] * is_fault_formed).astype(np.float32)
        # --- 筹码结构衰退风险 --- # 从布尔升级为数值
        # 假设筹码集中度斜率为正代表风险（例如主力派发给散户导致短期集中度上升）
        score_short_term_diverging = normalize(df['SLOPE_5_concentration_90pct_D'].clip(lower=0))
        score_mid_term_diverging = normalize(df['SLOPE_21_concentration_90pct_D'].clip(lower=0))
        states['SCORE_RISK_CHIP_STRUCTURE_DECAY'] = (score_short_term_diverging * score_mid_term_diverging).astype(np.float32)
        # --- 4. 状态信号合成 (市场环境分类) ---
        vol_compression_static = 1 - vol_static
        vol_compression_dynamic = 1 - vol_dynamic
        vol_compression_accel = 1 - vol_accel
        states['SCORE_REGIME_VOL_COMPRESSION_B'] = vol_compression_static.astype(np.float32)
        states['SCORE_REGIME_VOL_COMPRESSION_A'] = (vol_compression_static * vol_compression_dynamic).astype(np.float32)
        states['SCORE_REGIME_VOL_COMPRESSION_S'] = (states['SCORE_REGIME_VOL_COMPRESSION_A'] * vol_compression_accel).astype(np.float32)
        vol_expansion_static = vol_static
        vol_expansion_dynamic = vol_dynamic
        vol_expansion_accel = vol_accel
        states['SCORE_REGIME_VOL_EXPANSION_B'] = vol_expansion_static.astype(np.float32)
        states['SCORE_REGIME_VOL_EXPANSION_A'] = (vol_expansion_static * vol_expansion_dynamic).astype(np.float32)
        states['SCORE_REGIME_VOL_EXPANSION_S'] = (states['SCORE_REGIME_VOL_EXPANSION_A'] * vol_expansion_accel).astype(np.float32)
        print("        -> [结构风险与状态引擎 V4.1 数值化升级版] 分析完毕。") 
        return states

    def diagnose_advanced_structural_patterns_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V4.0 模式交叉验证版】高级结构模式诊断引擎
        - 核心重构 (本次修改):
          - [交叉验证] 废除旧的布尔模式信号，升级为与趋势、波动率等环境信号交叉验证的B/A/S三级置信度评分。
        - 核心逻辑:
          - B级: 基础模式识别 (如突破、吸筹)。
          - A级: B级信号得到核心环境确认 (如突破+趋势向上，吸筹+波动率压缩)。
          - S级: A级信号得到次级环境确认 (如突破+趋势向上+波动率放大)。
        - 收益: 极大提升了模式信号的实战价值，有效过滤了在不利环境下的假信号。
        """
        print("        -> [高级结构模式引擎 V4.0 模式交叉验证版] 启动...")
        states = {}
        p = get_params_block(self.strategy, 'advanced_patterns_params')
        if not get_param_value(p.get('enabled'), True): return {}
        atomic = self.strategy.atomic_states

        # --- 1. 军备检查 (Arsenal Check) ---
        required_pattern_cols = [
            'is_breakthrough_D', 'is_breakdown_D', 'is_accumulation_D',
            'is_distribution_D', 'is_consolidation_D'
        ]
        missing_cols = [col for col in required_pattern_cols if col not in df.columns]
        if missing_cols:
            print(f"          -> [严重警告] 高级模式引擎缺少关键数据列: {missing_cols}，模块已跳过！")
            return states
        
        required_atomic_scores = [
            'SCORE_MA_DYN_RESONANCE', 'SCORE_MA_ACCEL_RESONANCE',
            'SCORE_REGIME_VOL_COMPRESSION_S', 'SCORE_REGIME_VOL_EXPANSION_S'
        ]
        missing_scores = [s for s in required_atomic_scores if s not in atomic]
        if missing_scores:
            print(f"          -> [严重警告] 高级模式引擎缺少上游确认分数: {missing_scores}，模块已跳过！")
            return states

        # --- 2. 获取核心信号并数值化 ---
        is_breakthrough = df.get('is_breakthrough_D', 0).astype(float)
        is_breakdown = df.get('is_breakdown_D', 0).astype(float)
        is_accumulation = df.get('is_accumulation_D', 0).astype(float)
        is_distribution = df.get('is_distribution_D', 0).astype(float)
        is_consolidation = df.get('is_consolidation_D', 0).astype(float)
        
        daily_momentum_score = atomic['SCORE_MA_DYN_RESONANCE']
        daily_accel_score = atomic['SCORE_MA_ACCEL_RESONANCE']
        vol_compression_score = atomic['SCORE_REGIME_VOL_COMPRESSION_S']
        vol_expansion_score = atomic['SCORE_REGIME_VOL_EXPANSION_S']

        # --- 3. 模式信号交叉验证与评分 ---
        # 3.1 上升共振模式 (突破)
        states['SCORE_PATTERN_BULLISH_RESONANCE_B'] = is_breakthrough.astype(np.float32)
        states['SCORE_PATTERN_BULLISH_RESONANCE_A'] = (is_breakthrough * daily_momentum_score).astype(np.float32)
        states['SCORE_PATTERN_BULLISH_RESONANCE_S'] = (states['SCORE_PATTERN_BULLISH_RESONANCE_A'] * vol_expansion_score).astype(np.float32)

        # 3.2 下跌共振模式 (跌破) - 对称逻辑
        states['SCORE_PATTERN_BEARISH_RESONANCE_B'] = is_breakdown.astype(np.float32)
        states['SCORE_PATTERN_BEARISH_RESONANCE_A'] = (is_breakdown * (1 - daily_momentum_score)).astype(np.float32)
        states['SCORE_PATTERN_BEARISH_RESONANCE_S'] = (states['SCORE_PATTERN_BEARISH_RESONANCE_A'] * vol_expansion_score).astype(np.float32)

        # 3.3 底部反转模式 (吸筹)
        states['SCORE_PATTERN_BOTTOM_REVERSAL_B'] = is_accumulation.astype(np.float32)
        states['SCORE_PATTERN_BOTTOM_REVERSAL_A'] = (is_accumulation * vol_compression_score).astype(np.float32)
        states['SCORE_PATTERN_BOTTOM_REVERSAL_S'] = (states['SCORE_PATTERN_BOTTOM_REVERSAL_A'] * daily_accel_score).astype(np.float32)

        # 3.4 顶部反转模式 (派发) - 对称逻辑
        states['SCORE_PATTERN_TOP_REVERSAL_B'] = is_distribution.astype(np.float32)
        states['SCORE_PATTERN_TOP_REVERSAL_A'] = (is_distribution * vol_compression_score).astype(np.float32)
        states['SCORE_PATTERN_TOP_REVERSAL_S'] = (states['SCORE_PATTERN_TOP_REVERSAL_A'] * (1 - daily_accel_score)).astype(np.float32)

        # 3.5 盘整中继模式 (独立)
        momentum_neutrality = 1 - 2 * abs(daily_momentum_score - 0.5)
        states['SCORE_PATTERN_CONSOLIDATION_B'] = is_consolidation.astype(np.float32)
        states['SCORE_PATTERN_CONSOLIDATION_A'] = (is_consolidation * vol_compression_score).astype(np.float32)
        states['SCORE_PATTERN_CONSOLIDATION_S'] = (states['SCORE_PATTERN_CONSOLIDATION_A'] * momentum_neutrality).astype(np.float32)

        print("        -> [高级结构模式引擎 V4.0 模式交叉验证版] 分析完毕。")
        return states

    def diagnose_fused_behavioral_structure_risks(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 最终数值化版】行为与结构融合风险诊断模块
        - 核心职责: 融合行为、价格、筹码等多维度信息，生成高质量的原子风险信号。
        - 核心升级 (本次修改):
          - [最终数值化] 将内部所有布尔风险判断（动能衰竭、动态对倒）升级为基于归一化
                        评分的连续数值信号，提升风险评估的精度。
        """
        print("        -> [行为-结构融合风险模块 V2.0 最终数值化版] 启动...") 
        states = {}
        atomic = self.strategy.atomic_states
        default_series_float = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 军备检查 ---
        required_cols = [
            'close_D', 'high_D', 'low_D', 'SLOPE_5_volume_D',
            'SLOPE_5_VPA_EFFICIENCY_D', 'SLOPE_5_concentration_90pct_D'
        ]
        required_scores = [
            'SCORE_RISK_PROFIT_EXHAUSTION_S',
            'SCORE_RISK_VPA_STAGNATION',
        ]
        missing_cols = [c for c in required_cols if c not in df.columns]
        missing_scores = [s for s in required_scores if s not in atomic]
        if missing_cols or missing_scores:
            print(f"          -> [警告] 行为-结构融合风险模块缺少关键数据: 列{missing_cols}, 分数{missing_scores}。模块已跳过。")
            return states
        # --- 2. 计算“动能衰竭”风险 (Momentum Exhaustion) --- 
        # 2.1 价格位置分: 价格在近期（60日）高低点范围内的位置
        rolling_high = df['high_D'].rolling(60).max()
        rolling_low = df['low_D'].rolling(60).min()
        price_range = (rolling_high - rolling_low).replace(0, np.nan)
        score_at_high_price = ((df['close_D'] - rolling_low) / price_range).clip(0, 1).fillna(0.5)
        # 2.2 获取上游风险分
        score_profit_cushion_shrinking = atomic.get('SCORE_RISK_PROFIT_EXHAUSTION_S', default_series_float)
        score_market_engine_stalling = atomic.get('SCORE_RISK_VPA_STAGNATION', default_series_float)
        # 2.3 融合评分: 价格处于高位 * (获利盘收缩风险 或 引擎失速风险)
        combined_risk_score = np.maximum(score_profit_cushion_shrinking, score_market_engine_stalling)
        states['SCORE_RISK_MOMENTUM_EXHAUSTION'] = (score_at_high_price * combined_risk_score).astype(np.float32)
        # --- 3. 计算“动态对倒嫌疑”风险 (Dynamic Deceptive Churn) --- 
        norm_window = 120
        min_periods = max(1, norm_window // 5)
        def normalize(series, ascending=True):
            rank = series.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
            return rank if ascending else 1 - rank
        score_volume_increasing = normalize(df['SLOPE_5_volume_D'].clip(lower=0))
        score_vpa_efficiency_declining = normalize(df['SLOPE_5_VPA_EFFICIENCY_D'].clip(upper=0), ascending=False)
        score_chip_diverging_for_churn = normalize(df['SLOPE_5_concentration_90pct_D'].clip(lower=0))
        states['SCORE_RISK_DYNAMIC_DECEPTIVE_CHURN'] = (
            score_volume_increasing * score_vpa_efficiency_declining * score_chip_diverging_for_churn
        ).astype(np.float32)
        print("        -> [行为-结构融合风险模块 V2.0 最终数值化版] 分析完毕。") 
        return states

    def synthesize_consolidation_breakout_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【新增 V1.0 & 逻辑迁移】盘整突破机会合成模块
        - 核心职责: 消费本模块识别出的“盘整中继模式”数值化评分，并结合
                      “放量”或“强阳线”等点火信号，生成一个经过确认的、
                      更高质量的结构性突破机会分数。
        - 核心逻辑: 最终分数 = 昨日高质量盘整得分 * 今日点火信号强度
        - 收益: 将底层盘整信号转化为更高维的战术情报，遵循分层架构原则。
        """
        print("        -> [盘整突破机会合成模块 V1.0] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        triggers = self.strategy.trigger_events
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        default_series = pd.Series(False, index=df.index)
        # --- 1. 提取并融合“盘整”战备(Setup)信号 ---
        # 使用辅助函数融合S/A/B三级盘整模式分数，得到综合的“战备质量分”
        # 传入df参数以保证索引正确
        consolidation_setup_score = self._fuse_multi_level_scores(df, 'PATTERN_CONSOLIDATION')
        # --- 2. 提取并融合“点火”(Trigger)信号 ---
        # 点火源1: 放量突破 (来自 behavioral_intelligence)
        volume_ignition_score = atomic.get('SCORE_VOL_PRICE_IGNITION_UP', default_score)
        # 点火源2: 显性反转K线 (如大阳线)
        reversal_candle_trigger = triggers.get('TRIGGER_DOMINANT_REVERSAL', default_series).astype(float)
        # 取最强的点火信号作为当日的点火强度
        trigger_score = np.maximum(volume_ignition_score.values, reversal_candle_trigger.values)
        trigger_series = pd.Series(trigger_score, index=df.index)
        # --- 3. 融合生成结构层“盘整突破机会”分数 ---
        # 逻辑: 昨日战备就绪(高质量盘整) * 今日点火 = 突破机会
        final_score_series = consolidation_setup_score.shift(1).fillna(0.0) * trigger_series
        # 重命名信号以反映其来源
        states['SCORE_STRUCTURAL_CONSOLIDATION_BREAKOUT_OPP_A'] = final_score_series.astype(np.float32)
        print("        -> [盘整突破机会合成模块 V1.0] 计算完毕。")
        return states

    def synthesize_structural_opportunities(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.4 范式统一与健壮性修复版】结构性机会合成模块
        - 核心职责: 融合来自本模块的多个突破信号（箱体、平台、盘整），生成统一的“蓄势突破”机会分数。
        - 本次升级: [范式统一] 为了适配 V4.2 版箱体诊断模块输出的连续性“机会”信号，
                      本模块现在对所有上游突破信号统一使用 `_fuse_multi_level_scores` 进行预处理，
                      确保所有输入信号在融合前都经过了标准化的多级分数融合，增强了逻辑一致性和扩展性。
        - 核心修复: 修复了因索引不匹配导致上游信号Series为空或充满NaN的问题。
        """
        print("        -> [结构性机会合成模块 V1.4 范式统一与健壮性修复版] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_series = pd.Series(0.0, index=df.index, dtype=np.float32)
        
        # --- 1. 提取并融合来自本模块的多个突破源信号 ---
        # 对所有适用的信号源统一使用融合函数，以正确处理S/A/B等级
        # 源1: 箱体突破 (现在会正确处理新的连续性机会分数)
        # 传入df参数以保证索引正确
        box_breakout_score_series = self._fuse_multi_level_scores(df, 'BOX_BREAKOUT')
        
        # 源2: 平台突破 (同样使用融合函数，即使目前只有S级，逻辑也更健壮)
        # 传入df参数以保证索引正确
        platform_breakout_score_series = self._fuse_multi_level_scores(df, 'OPP_PLATFORM_BREAKOUT')
        
        # 源3: 盘整突破 (此信号逻辑特殊，维持原调用方式)
        consolidation_breakout_states = self.synthesize_consolidation_breakout_signals(df)
        consolidation_breakout_score_series = consolidation_breakout_states.get('SCORE_STRUCTURAL_CONSOLIDATION_BREAKOUT_OPP_A', default_series.copy())
        states.update(consolidation_breakout_states)
        
        # --- 调试信息：检查所有输入Series的值范围 ---
        # 使用.fillna(0)处理潜在的NaN值，使max()能正常工作
        print(f"    [调试] box_breakout_score_series max: {box_breakout_score_series.fillna(0).max():.4f}")
        print(f"    [调试] platform_breakout_score_series max: {platform_breakout_score_series.fillna(0).max():.4f}")
        print(f"    [调试] consolidation_breakout_score_series max: {consolidation_breakout_score_series.fillna(0).max():.4f}")

        # --- 2. 准备用于融合的Numpy数组，并执行健壮性检查 ---
        scores_to_reduce = []
        expected_shape = (len(df.index),)

        # 统一处理所有信号源
        signal_sources = {
            "box_breakout": box_breakout_score_series,
            "platform_breakout": platform_breakout_score_series,
            "consolidation_breakout": consolidation_breakout_score_series
        }

        for name, series in signal_sources.items():
            # 在转换为numpy数组前，先用reindex和fillna确保Series的形状和内容是正确的
            safe_series = series.reindex(df.index).fillna(0.0)
            score_arr = safe_series.values
            if score_arr.shape != expected_shape:
                print(f"    [警告] {name}_score 形状不匹配 ({score_arr.shape})，期望 {expected_shape}，将使用全零数组代替。")
                scores_to_reduce.append(np.zeros(expected_shape, dtype=np.float32))
            else:
                scores_to_reduce.append(score_arr)

        # --- 3. 融合生成“蓄势突破”分数与信号 ---
        # 逻辑: 取所有结构性突破信号中的最大值
        if not scores_to_reduce: # 增加一个空列表的检查
            final_score_arr = np.zeros(expected_shape, dtype=np.float32)
        else:
            final_score_arr = np.maximum.reduce(scores_to_reduce)
        final_score_series = pd.Series(final_score_arr, index=df.index, dtype=np.float32)
        states['SCORE_STRUCTURAL_ACCUMULATION_BREAKOUT_S'] = final_score_series
        
        # 生成布尔信号，用于兼容
        p = get_params_block(self.strategy, 'cognitive_fusion_params', {})
        breakout_threshold = get_param_value(p.get('accumulation_breakout_threshold'), 0.3)
        final_signal = final_score_series > breakout_threshold
        states['STRUCTURAL_OPP_ACCUMULATION_BREAKOUT_S'] = final_signal
        
        print("        -> [结构性机会合成模块 V1.4 范式统一与健壮性修复版] 计算完毕。")
        return states








