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

    def diagnose_ma_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V600.2 内存优化版】数值化多维诊断引擎
        - 核心重构: 从布尔信号引擎全面升级为数值化(0-1)评分引擎。
        - 核心扩展: 引入(5, 13, 21, 55)日均线的全景动态视图。
        - 新增信号 (数值型):
          - SCORE_MA_STATIC_ALIGNMENT: 均线静态排列分，量化“多头排列”的完美程度。
          - SCORE_MA_DYN_RESONANCE: 均线动态共振分，量化所有周期同向看涨的强度。
          - SCORE_MA_ACCEL_RESONANCE: 均线加速度共振分，捕捉趋势“点火”时刻。
          - SCORE_MA_DIVERGENCE_RISK: 均线背离风险分，识别“长多短空”的顶部陷阱。
          - SCORE_MA_HEALTH: 均线健康总分，融合静态与动态，综合评估趋势质量。
        - 性能优化:
          - 静态排列评分完全向量化，移除了Python循环。
          - 使用NumPy直接计算均值，避免了pd.concat创建中间DataFrame的开销。
          - 所有输出分数统一使用np.float32类型，减少50%内存占用。
        """
        print("        -> [诊断模块 V600.2 内存优化版] 启动...") # 更新打印信息
        states = {}
        p = get_params_block(self.strategy, 'multi_dim_ma_params')
        if not get_param_value(p.get('enabled'), True): return {}
        # --- 1. 定义周期并构建所需列名 ---
        ma_periods = get_param_value(p.get('ma_periods'), [5, 13, 21, 55])
        slope_lookback = get_param_value(p.get('slope_lookback'), 5)
        accel_lookback = get_param_value(p.get('accel_lookback'), 5)
        ema_cols = {p: f'EMA_{p}_D' for p in ma_periods}
        slope_cols = {p: f'SLOPE_{slope_lookback}_EMA_{p}_D' for p in ma_periods}
        accel_cols = {p: f'ACCEL_{accel_lookback}_EMA_{p}_D' for p in ma_periods}
        required_cols = list(ema_cols.values()) + list(slope_cols.values()) + list(accel_cols.values())
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"          -> [错误] 数值化均线引擎缺少必要列: {missing_cols}，跳过。")
            return {}
        # --- 2. 静态排列评分 (向量化优化) ---
        short_ma_cols = [ema_cols[p] for p in ma_periods[:-1]]
        long_ma_cols = [ema_cols[p] for p in ma_periods[1:]]
        alignment_sum = np.sum(df[short_ma_cols].values > df[long_ma_cols].values, axis=1)
        static_alignment_score = alignment_sum / (len(ma_periods) - 1)
        states['SCORE_MA_STATIC_ALIGNMENT'] = pd.Series(static_alignment_score, index=df.index, dtype=np.float32) # 指定dtype为float32
        # --- 3. 动态斜率与加速度评分 (归一化处理) ---
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        slope_series_list = [
            df[slope_cols[p]].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
            for p in ma_periods
        ]
        accel_series_list = [
            df[accel_cols[p]].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
            for p in ma_periods
        ]
        # --- 4. 合成高级数值信号 (NumPy优化) ---
        slope_values = np.mean(np.array([s.values for s in slope_series_list]), axis=0)
        states['SCORE_MA_DYN_RESONANCE'] = pd.Series(slope_values, index=df.index, dtype=np.float32) # 指定dtype为float32
        accel_values = np.mean(np.array([s.values for s in accel_series_list]), axis=0)
        states['SCORE_MA_ACCEL_RESONANCE'] = pd.Series(accel_values, index=df.index, dtype=np.float32) # 指定dtype为float32
        long_term_strength = slope_series_list[-1]
        short_term_weakness = 1 - slope_series_list[0]
        states['SCORE_MA_DIVERGENCE_RISK'] = (long_term_strength * short_term_weakness).fillna(0.5).astype(np.float32) # 转换dtype为float32
        # --- 5. 融合生成最终的均线健康总分 ---
        states['SCORE_MA_HEALTH'] = (
            states['SCORE_MA_STATIC_ALIGNMENT'] *
            states['SCORE_MA_DYN_RESONANCE'] *
            states['SCORE_MA_ACCEL_RESONANCE']
        ).astype(np.float32) # 转换dtype为float32
        print("        -> [诊断模块 V600.2 内存优化版] 分析完毕。") # 更新打印信息
        return states

    def diagnose_box_states_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.2 内存优化版】数值化箱体突破质量评分引擎
        - 核心重构: 从布尔事件输出升级为数值化(0-1)评分引擎，量化突破/跌破的质量。
        - 核心逻辑: 突破质量分 = (成交量确认分) * (K线强度分) * (趋势健康分)。
        - 新增信号 (数值型):
          - SCORE_BOX_BREAKOUT_QUALITY_A: A级箱体突破质量分。
          - SCORE_BOX_BREAKDOWN_INTENSITY_A: A级箱体跌破强度分。
        - 性能优化:
          - 将频繁访问的DataFrame列缓存到局部变量，减少重复查找开销。
          - 所有输出分数统一使用np.float32类型，减少50%内存占用。
        """
        print("        -> [箱体诊断模块 V3.2 内存优化版] 启动...") # 更新打印信息
        states = {}
        p = get_params_block(self.strategy, 'box_state_params')
        if not get_param_value(p.get('enabled'), True) or df.empty:
            return states
        # --- 1. 军备检查与参数获取 ---
        trend_health_source = get_param_value(p.get('trend_health_source'), 'SCORE_MA_HEALTH')
        required_cols = ['high_D', 'low_D', 'close_D', 'open_D', 'volume_D']
        required_signals = [trend_health_source]
        missing_cols = [col for col in required_cols if col not in df.columns]
        missing_signals = [s for s in required_signals if s not in self.strategy.atomic_states]
        if missing_cols or missing_signals:
            print(f"          -> [警告] 箱体诊断缺少关键数据: 列{missing_cols}, 信号{missing_signals}。模块已跳过。")
            return {}
        high_d, low_d, close_d, open_d, volume_d = df['high_D'], df['low_D'], df['close_D'], df['open_D'], df['volume_D']
        # --- 2. 计算基础箱体和布尔突破/跌破事件 ---
        lookback_window = get_param_value(p.get('lookback_window'), 8)
        max_amplitude_ratio = get_param_value(p.get('max_amplitude_ratio'), 0.05)
        rolling_high = high_d.rolling(window=lookback_window).max()
        rolling_low = low_d.rolling(window=lookback_window).min()
        amplitude_ratio = (rolling_high - rolling_low) / rolling_low.replace(0, np.nan)
        is_valid_box = (amplitude_ratio < max_amplitude_ratio).fillna(False)
        box_top = rolling_high
        box_bottom = rolling_low
        df['box_top_D'] = box_top
        df['box_bottom_D'] = box_bottom
        prev_close = close_d.shift(1)
        prev_box_top = box_top.shift(1)
        prev_box_bottom = box_bottom.shift(1)
        is_breakout = is_valid_box & (close_d > box_top) & (prev_close <= prev_box_top)
        is_breakdown = is_valid_box & (close_d < box_bottom) & (prev_close >= prev_box_bottom)
        # --- 3. 计算数值化质量评分组件 ---
        norm_window = get_param_value(p.get('norm_window'), 60)
        min_periods = max(1, norm_window // 5)
        volume_score = volume_d.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        candle_range = (high_d - low_d).replace(0, np.nan)
        breakout_candle_score = ((close_d - open_d) / candle_range).clip(0, 1).fillna(0.0)
        breakdown_candle_score = ((open_d - close_d) / candle_range).clip(0, 1).fillna(0.0)
        trend_health_score = self.strategy.atomic_states.get(trend_health_source, pd.Series(0.0, index=df.index))
        # --- 4. 融合生成最终质量分与兼容性信号 ---
        states['SCORE_BOX_BREAKOUT_QUALITY_A'] = (
            is_breakout.astype(float) * volume_score * breakout_candle_score * trend_health_score
        ).fillna(0.0).astype(np.float32) # 转换dtype为float32
        states['SCORE_BOX_BREAKDOWN_INTENSITY_A'] = (
            is_breakdown.astype(float) * volume_score * breakdown_candle_score
        ).fillna(0.0).astype(np.float32) # 转换dtype为float32
        states['BOX_EVENT_BREAKOUT'] = is_breakout.fillna(False)
        states['BOX_EVENT_BREAKDOWN'] = is_breakdown.fillna(False)
        print("        -> [箱体诊断模块 V3.2 内存优化版] 诊断完毕。") # 更新打印信息
        return states

    def diagnose_platform_states_scores(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        【V201.2 内存优化版】平台诊断与风险评分引擎
        - 核心重构: 从布尔判断升级为数值化(0-1)评分引擎，量化平台质量。
        - 核心逻辑: 平台总质量分 = (成本稳定性分) * (健康环境分)。
        - 新增信号 (数值型):
          - SCORE_PLATFORM_COST_STABILITY: 成本稳定性分。
          - SCORE_PLATFORM_CONTEXT_HEALTH: 健康环境分。
          - SCORE_PLATFORM_OVERALL_QUALITY: 平台总质量分。
          - SCORE_RISK_PLATFORM_BROKEN_S: 平台破位风险分，量化破位的严重性。
        - 性能优化:
          - 使用NumPy高效计算健康环境均值分，避免pd.concat创建中间DataFrame。
          - 所有输出分数统一使用np.float32类型，减少50%内存占用。
        """
        print("        -> [诊断模块 V201.2 内存优化版] 启动...") # 更新打印信息
        states = {}
        p = get_params_block(self.strategy, 'platform_state_params')
        if not get_param_value(p.get('enabled'), True): return df, {}
        default_series_float = pd.Series(0.0, index=df.index, dtype=np.float32) # 指定dtype为float32
        default_series_bool = pd.Series(False, index=df.index)
        peak_cost_col, close_col, long_ma_col = 'peak_cost_D', 'close_D', 'EMA_55_D'
        required_cols = [peak_cost_col, close_col, long_ma_col]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"          -> [警告] 缺少诊断平台状态所需的核心列: {missing}。模块将返回空结果。")
            df['PLATFORM_PRICE_STABLE'] = np.nan
            states.update({
                'PLATFORM_STATE_STABLE_FORMED': default_series_bool,
                'STRUCTURE_PLATFORM_BROKEN': default_series_bool,
                'STRUCTURE_BOX_ACCUMULATION_A': default_series_bool,
                'SCORE_RISK_PLATFORM_BROKEN_S': default_series_float
            })
            return df, states
        # --- 1. 计算核心指标：成本变异系数(CV) ---
        peak_cost = df[peak_cost_col]
        cost_cv_lookback = get_param_value(p.get('cost_cv_lookback'), 5)
        rolling_cost = peak_cost.rolling(cost_cv_lookback)
        rolling_mean_cost = rolling_cost.mean()
        rolling_std_cost = rolling_cost.std()
        with np.errstate(divide='ignore', invalid='ignore'):
            coeff_of_variation = (rolling_std_cost / rolling_mean_cost).fillna(1.0)
        # --- 2. 数值化评分 ---
        norm_window = get_param_value(p.get('stability_cv_window'), 120)
        min_periods = max(1, norm_window // 5)
        cv_rank = coeff_of_variation.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        states['SCORE_PLATFORM_COST_STABILITY'] = (1 - cv_rank).astype(np.float32) # 转换dtype为float32
        is_above_long_ma = (df[close_col] > df[long_ma_col]).astype(np.float32)
        is_shrinking_volume = self.strategy.atomic_states.get('VOL_STATE_SHRINKING', default_series_bool).astype(np.float32)
        is_chip_concentrating = self.strategy.atomic_states.get('CHIP_DYN_CONCENTRATING', default_series_bool).astype(np.float32)
        context_values = np.mean(np.array([is_above_long_ma.values, is_shrinking_volume.values, is_chip_concentrating.values]), axis=0)
        states['SCORE_PLATFORM_CONTEXT_HEALTH'] = pd.Series(context_values, index=df.index, dtype=np.float32) # 指定dtype为float32
        states['SCORE_PLATFORM_OVERALL_QUALITY'] = (
            states['SCORE_PLATFORM_COST_STABILITY'] * states['SCORE_PLATFORM_CONTEXT_HEALTH']
        ).fillna(0.0).astype(np.float32) # 转换dtype为float32
        # --- 3. 生成兼容旧版的布尔信号 ---
        threshold = get_param_value(p.get('final_score_threshold_for_bool'), 0.7)
        stable_formed_series = states['SCORE_PLATFORM_OVERALL_QUALITY'] > threshold
        states['PLATFORM_STATE_STABLE_FORMED'] = stable_formed_series
        states['STRUCTURE_BOX_ACCUMULATION_A'] = stable_formed_series
        # --- 4. 平台价格与破位逻辑 (含数值化风险评分) ---
        df['PLATFORM_PRICE_STABLE'] = peak_cost.where(stable_formed_series)
        was_on_platform = stable_formed_series.shift(1, fill_value=False)
        stable_platform_price_series = df['PLATFORM_PRICE_STABLE'].ffill()
        is_breaking_down = df[close_col] < stable_platform_price_series.shift(1)
        platform_failure_series = was_on_platform & is_breaking_down
        states['STRUCTURE_PLATFORM_BROKEN'] = platform_failure_series
        platform_quality_yesterday = states['SCORE_PLATFORM_OVERALL_QUALITY'].shift(1)
        states['SCORE_RISK_PLATFORM_BROKEN_S'] = platform_quality_yesterday.where(platform_failure_series, 0.0).astype(np.float32) # 转换dtype为float32
        print("        -> [诊断模块 V201.2 内存优化版] 分析完毕。") # 更新打印信息
        return df, states

    def synthesize_composite_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.2 内存优化版】复合结构评分模块
        - 核心重构: 从布尔状态合成升级为数值化分数融合，消费上游引擎的数值化输出。
        - 核心逻辑: 将布尔的 '&' 运算升级为分数的 '*' 运算，实现置信度加权。
        - 新增信号 (数值型):
          - SCORE_PLATFORM_WITH_TREND_SUPPORT: 平台质量分 * 均线健康分。
          - SCORE_BREAKOUT_EVE_S: 平台质量分 * 极致压缩分。
          - SCORE_MTF_TREND_ALIGNMENT_S: 日线均线健康分 * 周线趋势确认分。
        - 性能优化:
          - 所有输出分数统一使用np.float32类型，减少50%内存占用。
        """
        print("          -> [结构情报司令部 V3.2 内存优化版] 启动...") # 更新打印信息
        composite_scores = {}
        atomic = self.strategy.atomic_states
        # --- 1. 军备检查: 检查所有依赖的数值化信号 ---
        required_scores = ['SCORE_PLATFORM_OVERALL_QUALITY', 'SCORE_MA_HEALTH', 'VOL_STATE_EXTREME_SQUEEZE']
        missing_scores = [s for s in required_scores if s not in atomic]
        if missing_scores:
            print(f"          -> [严重警告] 复合评分引擎缺少核心原子分数: {missing_scores}，模块已跳过！")
            return {}
        # --- 2. 获取核心数值化分数 ---
        default_series_float = pd.Series(0.0, index=df.index, dtype=np.float32) # 指定dtype为float32
        platform_quality_score = atomic.get('SCORE_PLATFORM_OVERALL_QUALITY', default_series_float)
        ma_health_score = atomic.get('SCORE_MA_HEALTH', default_series_float)
        is_extreme_squeeze = atomic.get('VOL_STATE_EXTREME_SQUEEZE', pd.Series(False, index=df.index)).astype(np.float32) # 转换dtype为float32
        # --- 3. 数值化融合生成复合分数 ---
        composite_scores['SCORE_PLATFORM_WITH_TREND_SUPPORT'] = (platform_quality_score * ma_health_score).astype(np.float32) # 转换dtype为float32
        composite_scores['SCORE_BREAKOUT_EVE_S'] = (platform_quality_score * is_extreme_squeeze).astype(np.float32) # 转换dtype为float32
        weekly_ma_slope_col = 'SLOPE_5_EMA_21_W'
        if weekly_ma_slope_col in df.columns:
            is_weekly_trend_up_score = (df[weekly_ma_slope_col] > 0).astype(np.float32) # 转换dtype为float32
            composite_scores['SCORE_MTF_TREND_ALIGNMENT_S'] = (ma_health_score * is_weekly_trend_up_score).astype(np.float32) # 转换dtype为float32
        else:
            print(f"          -> [警告] 缺少周线斜率列 '{weekly_ma_slope_col}'，无法生成S级多维共振分数。")
            composite_scores['SCORE_MTF_TREND_ALIGNMENT_S'] = default_series_float
        print("          -> [结构情报司令部 V3.2 内存优化版] 分析完毕。") # 更新打印信息
        return composite_scores

    def diagnose_fibonacci_support(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.2 内存优化版】斐波那契反攻诊断模块
        - 核心重构: 从布尔信号升级为数值化(0-1)评分引擎，量化反攻机会的质量。
        - 核心逻辑: 机会分 = (昨日触及斐波那契位) * (今日确认反转) * (当前趋势健康总分)。
        - 动态增强: 使用多维均线健康分(SCORE_MA_HEALTH)进行趋势背景质量评估。
        - 性能优化:
          - 使用 np.maximum 替代 pd.concat().max()，避免创建中间DataFrame。
          - 所有输出分数统一使用np.float32类型，减少50%内存占用。
        """
        print("        -> [斐波那契反攻诊断模块 V3.2 内存优化版] 启动...") # 更新打印信息
        states = {}
        p = get_params_block(self.strategy, 'fibonacci_support_params')
        if not get_param_value(p.get('enabled'), True):
            return {}
        # --- 1. 军备检查 (Prerequisite Check) ---
        fib_levels = {'0.618': 'FIB_0_618_D', '0.500': 'FIB_0_500_D', '0.382': 'FIB_0_382_D'}
        trend_health_score_source = get_param_value(p.get('trend_health_score_source'), 'SCORE_MA_HEALTH')
        required_cols = list(fib_levels.values())
        required_signals = ['TRIGGER_DOMINANT_REVERSAL', trend_health_score_source]
        missing_cols = [col for col in required_cols if col not in df.columns]
        missing_signals = [s for s in required_signals if s not in self.strategy.atomic_states and s not in self.strategy.trigger_events]
        if missing_cols or missing_signals:
            print(f"          -> [警告] 缺少斐波那契诊断所需数据: 列{missing_cols}, 信号{missing_signals}。模块跳过。")
            return {}
        # --- 2. 获取核心动态信号 ---
        is_reversal_confirmed_today = self.strategy.trigger_events.get('TRIGGER_DOMINANT_REVERSAL', pd.Series(False, index=df.index)).astype(np.float32) # 转换dtype为float32
        trend_health_score = self.strategy.atomic_states.get(trend_health_score_source, pd.Series(0.0, index=df.index, dtype=np.float32)) # 指定dtype为float32
        # --- 3. 核心逻辑：计算反攻数值分数 ---
        proximity_ratio = get_param_value(p.get('proximity_ratio'), 0.01)
        low_d_yesterday = df['low_D'].shift(1)
        def calculate_rebound_score(fib_level_col: str) -> pd.Series:
            fib_level_yesterday = df[fib_level_col].shift(1)
            was_pierced_yesterday = (low_d_yesterday <= fib_level_yesterday * (1 + proximity_ratio)).fillna(False).astype(np.float32) # 转换dtype为float32
            return was_pierced_yesterday * is_reversal_confirmed_today * trend_health_score
        # --- 4. 分别为不同级别的支撑生成分数 ---
        score_618 = calculate_rebound_score(fib_levels['0.618'])
        score_500 = calculate_rebound_score(fib_levels['0.500'])
        score_382 = calculate_rebound_score(fib_levels['0.382'])
        states['SCORE_FIB_SUPPORT_GOLDEN_POCKET_S'] = score_618.astype(np.float32) # 转换dtype为float32
        standard_support_values = np.maximum(score_500.values, score_382.values)
        states['SCORE_FIB_SUPPORT_STANDARD_A'] = pd.Series(standard_support_values, index=df.index, dtype=np.float32) # 指定dtype为float32
        if (states['SCORE_FIB_SUPPORT_GOLDEN_POCKET_S'] > 0).any():
            print(f"          -> [情报] 侦测到 {(states['SCORE_FIB_SUPPORT_GOLDEN_POCKET_S'] > 0).sum()} 次 S级“黄金口袋”反攻机会，最高分: {states['SCORE_FIB_SUPPORT_GOLDEN_POCKET_S'].max():.2f}。")
        if (states['SCORE_FIB_SUPPORT_STANDARD_A'] > 0).any():
            print(f"          -> [情报] 侦测到 {(states['SCORE_FIB_SUPPORT_STANDARD_A'] > 0).sum()} 次 A级“标准斐波那契”反攻机会，最高分: {states['SCORE_FIB_SUPPORT_STANDARD_A'].max():.2f}。")
        return states

    def diagnose_structural_mechanics_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V500.2 内存优化版】结构力学诊断引擎
        - 核心重构: 从单一周期的布尔判断，全面升级为多时间维度的数值化(0-1)共振评分。
        - 核心逻辑: 对成本、筹码、波动率等核心力学要素在多个周期上进行动态评分，并融合成高级共振分数。
        - 新增信号 (数值型):
          - SCORE_MECHANICS_COST_MOMENTUM: 成本动能共振分。
          - SCORE_MECHANICS_COST_ACCEL: 成本加速共振分。
          - SCORE_MECHANICS_SUPPLY_LOCK: 供应锁定共振分。
          - SCORE_MECHANICS_ENERGY_ADVANTAGE: 势能优势分。
          - SCORE_MECHANICS_BULLISH_SETUP_S: S级看涨结构总分。
        - 性能优化:
          - 使用NumPy高效计算共振均值分，避免了多个pd.concat创建中间DataFrame的开销。
          - 所有输出分数统一使用np.float32类型，减少50%内存占用。
        """
        print("        -> [结构力学诊断引擎 V500.2 内存优化版] 启动...") # 更新打印信息
        states = {}
        p = get_params_block(self.strategy, 'structural_mechanics_params')
        if not get_params_block(self.strategy, 'structural_risk_params').get('enabled', True): return {}
        # --- 1. 军备检查 (Prerequisite Check) ---
        periods = get_param_value(p.get('dynamic_periods'), [5, 13, 21])
        slope_lookback = get_param_value(p.get('slope_lookback'), 5)
        accel_lookback = get_param_value(p.get('accel_lookback'), 5)
        required_cols = ['support_below_D', 'pressure_above_D']
        dynamic_sources = {'peak_cost': 'peak_cost_D', 'concentration': 'concentration_90pct_D', 'bbw': 'BBW_21_2.0_D'}
        for period in periods:
            required_cols.extend([
                f'SLOPE_{slope_lookback}_{dynamic_sources["peak_cost"]}_{period}',
                f'ACCEL_{accel_lookback}_{dynamic_sources["peak_cost"]}_{period}',
                f'SLOPE_{slope_lookback}_{dynamic_sources["concentration"]}_{period}',
                f'SLOPE_{slope_lookback}_{dynamic_sources["bbw"]}_{period}'
            ])
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"          -> [严重警告] 结构力学引擎缺少关键数据列: {missing_cols}，模块已跳过！")
            return states
        # --- 2. 动态指标数值化 (归一化处理) ---
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        cost_slope_series = [df[f'SLOPE_{slope_lookback}_{dynamic_sources["peak_cost"]}_{p}'].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5) for p in periods]
        cost_accel_series = [df[f'ACCEL_{accel_lookback}_{dynamic_sources["peak_cost"]}_{p}'].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5) for p in periods]
        conc_slope_series = [1 - df[f'SLOPE_{slope_lookback}_{dynamic_sources["concentration"]}_{p}'].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5) for p in periods]
        bbw_slope_series = [1 - df[f'SLOPE_{slope_lookback}_{dynamic_sources["bbw"]}_{p}'].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5) for p in periods]
        # --- 3. 合成多维共振分数 (NumPy优化) ---
        states['SCORE_MECHANICS_COST_MOMENTUM'] = pd.Series(np.mean(np.array([s.values for s in cost_slope_series]), axis=0), index=df.index, dtype=np.float32) # 指定dtype为float32
        states['SCORE_MECHANICS_COST_ACCEL'] = pd.Series(np.mean(np.array([s.values for s in cost_accel_series]), axis=0), index=df.index, dtype=np.float32) # 指定dtype为float32
        avg_conc_score = np.mean(np.array([s.values for s in conc_slope_series]), axis=0)
        avg_bbw_score = np.mean(np.array([s.values for s in bbw_slope_series]), axis=0)
        states['SCORE_MECHANICS_SUPPLY_LOCK'] = pd.Series(avg_conc_score * avg_bbw_score, index=df.index, dtype=np.float32) # 指定dtype为float32
        # --- 4. 静态势能评分 ---
        pressure_col = df['pressure_above_D'] + 1e-6
        energy_ratio = df['support_below_D'] / pressure_col
        energy_advantage_threshold = get_param_value(p.get('energy_advantage_threshold'), 1.5)
        states['SCORE_MECHANICS_ENERGY_ADVANTAGE'] = np.clip(energy_ratio / (energy_advantage_threshold * 2.5), 0, 1).astype(np.float32) # 转换dtype为float32
        # --- 5. 融合生成最终的S级看涨结构总分 ---
        states['SCORE_MECHANICS_BULLISH_SETUP_S'] = (
            states['SCORE_MECHANICS_ENERGY_ADVANTAGE'] *
            states['SCORE_MECHANICS_SUPPLY_LOCK'] *
            states['SCORE_MECHANICS_COST_ACCEL']
        ).fillna(0.0).astype(np.float32) # 转换dtype为float32
        print("        -> [结构力学诊断引擎 V500.2 内存优化版] 分析完毕。") # 更新打印信息
        return states

    def diagnose_mtf_trend_synergy_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V4.2 内存优化版】战略协同评分引擎
        - 核心重构: 从单一周期的布尔判断，全面升级为多时间维度的数值化(0-1)共振评分。
        - 核心逻辑: 协同分 = (日线多维共振分) * (周线多维共振分)。
        - 动态增强: 消费日线高级共振信号，并内部构建周线多维动态共振评分。
        - 性能优化:
          - 使用NumPy高效计算周线共振均值分，避免了pd.concat创建中间DataFrame的开销。
          - 所有输出分数统一使用np.float32类型，减少50%内存占用。
        """
        print("        -> [战略协同引擎 V4.2 内存优化版] 启动...") # 更新打印信息
        states = {}
        p = get_params_block(self.strategy, 'mtf_trend_synergy_params')
        if not get_param_value(p.get('enabled'), True): return {}
        default_series = pd.Series(0.0, index=df.index, dtype=np.float32) # 指定dtype为float32
        # --- 1. 军备检查 (Prerequisite Check) ---
        weekly_periods = get_param_value(p.get('weekly_periods'), [5, 13, 21])
        slope_lookback = get_param_value(p.get('slope_lookback'), 5)
        accel_lookback = get_param_value(p.get('accel_lookback'), 5)
        weekly_slope_cols = {p: f'SLOPE_{slope_lookback}_EMA_{p}_W' for p in weekly_periods}
        weekly_accel_cols = {p: f'ACCEL_{accel_lookback}_EMA_{p}_W' for p in weekly_periods}
        required_cols = list(weekly_slope_cols.values()) + list(weekly_accel_cols.values())
        daily_momentum_source = get_param_value(p.get('daily_momentum_source'), 'SCORE_MA_DYN_RESONANCE')
        daily_accel_source = get_param_value(p.get('daily_accel_source'), 'SCORE_MA_ACCEL_RESONANCE')
        required_signals = [daily_momentum_source, daily_accel_source]
        missing_cols = [col for col in required_cols if col not in df.columns]
        missing_signals = [s for s in required_signals if s not in self.strategy.atomic_states]
        if missing_cols or missing_signals:
            print(f"          -> [严重警告] 战略协同引擎缺少关键数据: 列{missing_cols}, 信号{missing_signals}。模块已跳过！")
            return {}
        # --- 2. 获取日线层面的高级共振分数 ---
        daily_momentum_score = self.strategy.atomic_states.get(daily_momentum_source, default_series)
        daily_accel_score = self.strategy.atomic_states.get(daily_accel_source, default_series)
        # --- 3. 构建周线层面的多维共振评分引擎 (NumPy优化) ---
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        weekly_slope_series = [df[weekly_slope_cols[p]].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5) for p in weekly_periods]
        weekly_accel_series = [df[weekly_accel_cols[p]].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5) for p in weekly_periods]
        weekly_momentum_score_values = np.mean(np.array([s.values for s in weekly_slope_series]), axis=0)
        weekly_accel_score_values = np.mean(np.array([s.values for s in weekly_accel_series]), axis=0)
        weekly_momentum_score = pd.Series(weekly_momentum_score_values, index=df.index, dtype=np.float32) # 指定dtype为float32
        weekly_accel_score = pd.Series(weekly_accel_score_values, index=df.index, dtype=np.float32) # 指定dtype为float32
        # --- 4. 融合生成S++级战略协同分数 ---
        states['SCORE_MTF_IGNITION_SYNERGY_S_PLUS'] = (daily_accel_score * weekly_accel_score).astype(np.float32) # 转换dtype为float32
        states['SCORE_MTF_MOMENTUM_SYNERGY_A_PLUS'] = (daily_momentum_score * weekly_momentum_score).astype(np.float32) # 转换dtype为float32
        states['SCORE_MTF_EXHAUSTION_RISK_S_PLUS'] = (daily_momentum_score * (1 - weekly_momentum_score)).astype(np.float32) # 转换dtype为float32
        print("        -> [战略协同引擎 V4.2 内存优化版] 分析完毕。") # 更新打印信息
        return states

    def diagnose_fusion_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V5.2 内存优化版】静态-动态融合评分引擎
        - 核心重构: 从布尔信号的 '&' 运算，全面升级为数值化分数的 '*' 运算。
        - 核心逻辑: 最终融合分 = (静态战备质量分) * (动态触发强度分)。
        - 信号源升级: 消费来自上游数值化引擎的高级分数，实现“优中选优”。
        - 性能优化:
          - 所有输出分数统一使用np.float32类型，减少50%内存占用。
        """
        print("        -> [静态-动态融合评分引擎 V5.2 内存优化版] 启动...") # 更新打印信息
        states = {}
        p = get_params_block(self.strategy, 'fusion_scores_params')
        if not get_param_value(p.get('enabled'), True): return {}
        atomic = self.strategy.atomic_states
        # --- 1. 定义并检查所需的核心数值化信号源 ---
        sources = get_param_value(p.get('signal_sources'), {})
        static_platform_setup_src = sources.get('static_platform_setup', 'SCORE_BREAKOUT_EVE_S')
        static_mechanics_setup_src = sources.get('static_mechanics_setup', 'SCORE_MECHANICS_BULLISH_SETUP_S')
        dynamic_ignition_src = sources.get('dynamic_ignition', 'SCORE_MTF_IGNITION_SYNERGY_S_PLUS')
        dynamic_exhaustion_risk_src = sources.get('dynamic_exhaustion_risk', 'SCORE_MTF_EXHAUSTION_RISK_S_PLUS')
        required_signals = [
            static_platform_setup_src, static_mechanics_setup_src,
            dynamic_ignition_src, dynamic_exhaustion_risk_src
        ]
        missing_signals = [s for s in required_signals if s not in atomic]
        if missing_signals:
            print(f"          -> [严重警告] 静态-动态融合引擎缺少核心原子分数: {missing_signals}，模块已跳过！")
            return {}
        # --- 2. 获取核心数值化分数序列 ---
        default_series = pd.Series(0.0, index=df.index, dtype=np.float32) # 指定dtype为float32
        static_platform_setup_score = atomic.get(static_platform_setup_src, default_series)
        static_mechanics_setup_score = atomic.get(static_mechanics_setup_src, default_series)
        dynamic_ignition_score = atomic.get(dynamic_ignition_src, default_series)
        dynamic_exhaustion_risk_score = atomic.get(dynamic_exhaustion_risk_src, default_series)
        # --- 3. 组合生成S++级质量加权融合分数 ---
        states['SCORE_FUSION_PLATFORM_IGNITION_S_PLUS'] = (static_platform_setup_score * dynamic_ignition_score).astype(np.float32) # 转换dtype为float32
        states['SCORE_FUSION_MECHANICS_IGNITION_S_PLUS'] = (static_mechanics_setup_score * dynamic_ignition_score).astype(np.float32) # 转换dtype为float32
        states['SCORE_FUSION_PLATFORM_TRAP_S_PLUS'] = (static_platform_setup_score * dynamic_exhaustion_risk_score).astype(np.float32) # 转换dtype为float32
        print("        -> [静态-动态融合评分引擎 V5.2 内存优化版] 分析完毕。") # 更新打印信息
        return states

    def diagnose_structural_risks_and_regimes_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.2 内存优化版】结构性风险与市场状态诊断引擎
        - 核心重构: 从布尔信号全面升级为数值化(0-1)评分引擎，精确量化风险程度。
        - 核心逻辑:
          1. BIAS风险: 使用滚动百分位排名实现动态自适应。
          2. Hurst状态: 将Hurst指数转化为“趋势强度分”和“回归强度分”。
        - 交叉验证: 新增“失控列车风险”，融合“超涨风险分”与“趋势强度分”。
        - 性能优化:
          - 使用NumPy高效计算日线综合BIAS均值，避免pd.concat创建中间DataFrame。
          - 所有输出分数统一使用np.float32类型，减少50%内存占用。
        """
        print("        -> [结构风险与状态诊断模块 V2.2 内存优化版] 启动...") # 更新打印信息
        states = {}
        p = get_params_block(self.strategy, 'structural_risk_params')
        if not get_param_value(p.get('enabled'), True): return states
        default_series = pd.Series(0.0, index=df.index, dtype=np.float32) # 指定dtype为float32
        # --- 1. 军备检查 (Prerequisite Check) ---
        bias_periods_d = get_param_value(p.get('bias_periods_D'), [21, 55])
        bias_period_w = get_param_value(p.get('bias_period_W'), 20)
        required_cols = [f'BIAS_{p}_D' for p in bias_periods_d] + [f'BIAS_{bias_period_w}_W', 'hurst_120d_D']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"          -> [警告] 结构风险诊断缺少关键数据列: {missing_cols}，模块已跳过。")
            return {}
        # --- 2. 结构性超涨风险数值化评分 (0-1分) ---
        norm_window = get_param_value(p.get('norm_window'), 250)
        min_periods = max(1, norm_window // 5)
        daily_bias_series = [df[f'BIAS_{p}_D'] for p in bias_periods_d if f'BIAS_{p}_D' in df.columns]
        if daily_bias_series:
            composite_daily_bias_values = np.mean(np.array([s.values for s in daily_bias_series]), axis=0)
            composite_daily_bias = pd.Series(composite_daily_bias_values, index=df.index)
            states['SCORE_RISK_OVEREXTENDED_D'] = composite_daily_bias.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.0).astype(np.float32) # 转换dtype为float32
        else:
            states['SCORE_RISK_OVEREXTENDED_D'] = default_series
        weekly_bias_col = f'BIAS_{bias_period_w}_W'
        states['SCORE_RISK_OVEREXTENDED_W'] = df[weekly_bias_col].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.0).astype(np.float32) # 转换dtype为float32
        states['SCORE_RISK_MTF_OVEREXTENDED_RESONANCE_S'] = (states['SCORE_RISK_OVEREXTENDED_D'] * states['SCORE_RISK_OVEREXTENDED_W']).astype(np.float32) # 转换dtype为float32
        # --- 3. 市场状态数值化评分 (0-1分) ---
        hurst = df['hurst_120d_D']
        states['SCORE_REGIME_TRENDING_STRENGTH'] = np.clip((hurst - 0.5) * 2, 0, 1).fillna(0.0).astype(np.float32) # 转换dtype为float32
        states['SCORE_REGIME_REVERTING_STRENGTH'] = np.clip((0.5 - hurst) * 2, 0, 1).fillna(0.0).astype(np.float32) # 转换dtype为float32
        # --- 4. 交叉验证: 融合风险状态与市场属性 ---
        states['SCORE_RISK_RUNAWAY_TRAIN_S_PLUS'] = (states['SCORE_RISK_MTF_OVEREXTENDED_RESONANCE_S'] * states['SCORE_REGIME_TRENDING_STRENGTH']).astype(np.float32) # 转换dtype为float32
        print("        -> [结构风险与状态诊断模块 V2.2 内存优化版] 诊断完毕。") # 更新打印信息
        return states

    def diagnose_advanced_structural_patterns_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.2 内存优化版】先进结构模式诊断引擎
        - 核心重构: 从布尔信号全面升级为数值化(0-1)评分引擎，量化模式质量。
        - 核心逻辑: 模式分 = (模式自身完美度) * (所处趋势健康度)。
        - 动态增强: 消费上游模块生成的高级趋势健康分，实现静态模式与动态趋势的质量加权。
        - 性能优化:
          - 所有输出分数统一使用np.float32类型，减少50%内存占用。
        """
        print("        -> [先进结构模式诊断模块 V2.2 内存优化版] 启动...") # 更新打印信息
        states = {}
        p = get_params_block(self.strategy, 'advanced_structural_params')
        if not get_param_value(p.get('enabled'), True):
            return states
        # --- 1. 军备检查 (Prerequisite Check) ---
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        trend_health_source = get_param_value(p.get('trend_health_source'), 'SCORE_MA_HEALTH')
        daily_bias_period = get_param_value(p.get('daily_alignment_bias_period'), 55)
        weekly_bias_period = get_param_value(p.get('weekly_alignment_bias_period'), 21)
        required_cols = [
            'ATR_14_D', 'price_cv_60d_D', 'high_D', 'low_D',
            f'BIAS_{daily_bias_period}_D', f'BIAS_{weekly_bias_period}_W'
        ]
        required_signals = [trend_health_source]
        missing_cols = [col for col in required_cols if col not in df.columns]
        missing_signals = [s for s in required_signals if s not in self.strategy.atomic_states]
        if missing_cols or missing_signals:
            print(f"          -> [警告] 先进结构诊断缺少关键数据: 列{missing_cols}, 信号{missing_signals}。模块已跳过。")
            return {}
        # --- 2. 获取上游高级动态分数 ---
        trend_health_score = self.strategy.atomic_states.get(trend_health_source, pd.Series(0.0, index=df.index, dtype=np.float32)) # 指定dtype为float32
        # --- 3. 模式数值化评分 ---
        atr_rank = df['ATR_14_D'].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        states['SCORE_PATTERN_ATR_COMPRESSION_A'] = (1 - atr_rank).astype(np.float32) # 转换dtype为float32
        price_cv_rank = df['price_cv_60d_D'].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        trend_smoothness_score = 1 - price_cv_rank
        states['SCORE_PATTERN_HIGH_QUALITY_TREND_A'] = (trend_smoothness_score * trend_health_score).astype(np.float32) # 转换dtype为float32
        is_inside_day = (df['high_D'] < df['high_D'].shift(1)) & (df['low_D'] > df['low_D'].shift(1))
        states['SCORE_PATTERN_INSIDE_DAY_N'] = is_inside_day.fillna(False).astype(np.float32) # 转换dtype为float32
        daily_alignment_score = df[f'BIAS_{daily_bias_period}_D'].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        weekly_alignment_score = df[f'BIAS_{weekly_bias_period}_W'].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        states['SCORE_PATTERN_MTF_STATIC_ALIGNMENT_S'] = (daily_alignment_score * weekly_alignment_score).astype(np.float32) # 转换dtype为float32
        print("        -> [先进结构模式诊断模块 V2.2 内存优化版] 诊断完毕。") # 更新打印信息
        return states



