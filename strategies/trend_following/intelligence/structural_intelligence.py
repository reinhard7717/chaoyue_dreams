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

    def _normalize_score(self, series: pd.Series, window: int, target_index: pd.Index, ascending: bool = True) -> pd.Series:
        """
        【V1.0 新增】计算一个系列在滚动窗口内的归一化得分 (0-1)。
        """
        if series is None or series.isnull().all():
            return pd.Series(0.5, index=target_index)

        return series.rolling(
            window=window, 
            min_periods=int(window * 0.2)
        ).rank(
            pct=True, 
            ascending=ascending
        ).fillna(0.5).astype(np.float32)

    def diagnose_structural_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 终极信号版】结构情报分析总指挥
        - 核心重构: 遵循终极信号范式，本模块不再返回一堆零散的原子信号。
                      现在只调用唯一的终极信号引擎 `diagnose_ultimate_structural_signals`，
                      并将其产出的16个S+/S/A/B级信号作为本模块的最终输出。
        - 收益: 架构与其他情报模块完全统一，极大提升了信号质量和架构清晰度。
        """
        # print("      -> [结构情报分析总指挥 V2.0 终极信号版] 启动...")
        # 直接调用终极信号引擎，并将其结果作为本模块的唯一输出
        ultimate_structural_states = self.diagnose_ultimate_structural_signals(df)
        # print(f"      -> [结构情报分析总指挥 V2.0] 分析完毕，共生成 {len(ultimate_structural_states)} 个终极结构信号。")
        return ultimate_structural_states

    def diagnose_ultimate_structural_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V6.0 · 上下文门控范式】终极结构信号诊断模块
        - 核心重构 (本次修改):
          - [哲学升级] 引入“上下文门控”范式，静态值作为决定性的“参与条件”，而非“奖励”。
          - [新范式] 最终信号 = 上下文门控分 * 动态触发分。
          - [专业分工] 共振信号的门控是“内部静态结构”，反转信号的门控是“外部宏观位置”。
        - 收益: 赋予系统前所未有的大局观，确保所有信号都发生在“正确的位置”，从根本上提升了信号的胜率和可靠性。
        """
        print("        -> [终极结构信号诊断模块 V6.0 · 上下文门控范式] 启动...") # [代码修改] 更新版本号和说明
        states = {}
        p_conf = get_params_block(self.strategy, 'structural_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True): return states
        exponent = get_param_value(p_conf.get('final_score_exponent'), 1.0)
        
        # 定义权重
        resonance_tf_weights = {'short': 0.2, 'medium': 0.5, 'long': 0.3}
        reversal_tf_weights = {'short': 0.6, 'medium': 0.3, 'long': 0.1}
        dynamic_weights = {'slope': 0.6, 'accel': 0.4}

        # 1. 计算“外部宏观位置”门控 (用于反转)
        rolling_low_55d = df['low_D'].rolling(window=55, min_periods=21).min()
        rolling_high_55d = df['high_D'].rolling(window=55, min_periods=21).max()
        price_range_55d = (rolling_high_55d - rolling_low_55d).replace(0, 1e-9)
        price_position_in_range = ((df['close_D'] - rolling_low_55d) / price_range_55d).clip(0, 1).fillna(0.5)
        bottom_context_score = 1 - price_position_in_range
        top_context_score = price_position_in_range
        
        periods = get_param_value(p_conf.get('periods', [1, 5, 13, 21, 55]))
        norm_window = get_param_value(p_conf.get('norm_window'), 120)
        
        # 2. 分别计算四大支柱的“纯静态健康度”和“纯动态健康度”
        pillar_static_health = {}
        pillar_dynamic_health = {}

        # 2.1 MA支柱
        ma_static_health = {}
        ma_dynamic_health = {}
        for p in periods:
            static_col = f'EMA_{p}_D' if p > 1 else 'close_D'
            slope_lookback = 5 if p <= 5 else p
            accel_lookback = 5 if p <= 5 else p
            slope_col = f'SLOPE_{slope_lookback}_{static_col}'
            accel_col = f'ACCEL_{accel_lookback}_{static_col}'
            # 静态分：价格与均线的关系
            ma_static_health[p] = self._normalize_score(df.get(f'price_vs_ma_{p}_D', pd.Series(0.5, index=df.index)), norm_window, df.index)
            # 动态分：均线斜率与加速度
            slope_score = self._normalize_score(df.get(slope_col), norm_window, df.index)
            accel_score = self._normalize_score(df.get(accel_col), norm_window, df.index)
            ma_dynamic_health[p] = slope_score * dynamic_weights['slope'] + accel_score * dynamic_weights['accel']
        pillar_static_health['ma'] = ma_static_health
        pillar_dynamic_health['ma'] = ma_dynamic_health
        
        # 2.2 力学支柱 (Mechanics) - 静态分来自能量优势，动态分来自成本和筹码斜率
        mechanics_static_health = {}
        mechanics_dynamic_health = {}
        energy_score = self._normalize_score(df.get('energy_ratio_D'), norm_window, df.index)
        for p in periods:
            mechanics_static_health[p] = energy_score # 能量优势是静态的
            cost_slope_score = self._normalize_score(df.get(f'SLOPE_{p}_peak_cost_D'), norm_window, df.index)
            conc_lock_score = self._normalize_score(df.get(f'SLOPE_{p}_concentration_90pct_D'), norm_window, df.index, ascending=False)
            mechanics_dynamic_health[p] = cost_slope_score * 0.5 + conc_lock_score * 0.5 # 动态分由成本和筹码斜率驱动
        pillar_static_health['mechanics'] = mechanics_static_health
        pillar_dynamic_health['mechanics'] = mechanics_dynamic_health
        
        # 2.3 MTF 和 Pattern 支柱 (这两个支柱的健康度计算逻辑较为特殊，但仍可拆分为静态和动态部分)
        pillar_static_health['mtf'], pillar_dynamic_health['mtf'] = self._calculate_mtf_health_v2(df, periods, norm_window)
        pillar_static_health['pattern'], pillar_dynamic_health['pattern'] = self._calculate_pattern_health_v2(df, periods, norm_window)
        
        # 3. 融合生成“全局静态门控”和“全局动态引擎”
        overall_static_health = {}
        overall_dynamic_health = {}
        for p in periods:
            static_healths_for_period = [pillar_static_health[key][p].values for key in pillar_static_health]
            overall_static_health[p] = pd.Series(np.mean(np.stack(static_healths_for_period, axis=0), axis=0), index=df.index)
            
            dynamic_healths_for_period = [pillar_dynamic_health[key][p].values for key in pillar_dynamic_health]
            overall_dynamic_health[p] = pd.Series(np.mean(np.stack(dynamic_healths_for_period, axis=0), axis=0), index=df.index)

        # 4. 终极信号合成 (采用上下文门控范式)
        # 4.1 共振信号合成 (Resonance = Gated by Internal Static Structure)
        bullish_resonance_health = {p: overall_static_health[p] * overall_dynamic_health[p] for p in periods}
        bullish_short_force_res = (bullish_resonance_health[1] * bullish_resonance_health[5])**0.5
        bullish_medium_trend_res = (bullish_resonance_health[13] * bullish_resonance_health[21])**0.5
        bullish_long_inertia_res = bullish_resonance_health[55]
        overall_bullish_resonance = (bullish_short_force_res * resonance_tf_weights['short'] +
                                     bullish_medium_trend_res * resonance_tf_weights['medium'] +
                                     bullish_long_inertia_res * resonance_tf_weights['long'])
        states['SCORE_STRUCTURE_BULLISH_RESONANCE_S_PLUS'] = (overall_bullish_resonance ** exponent).astype(np.float32)
        states['SCORE_STRUCTURE_BULLISH_RESONANCE_S'] = (states['SCORE_STRUCTURE_BULLISH_RESONANCE_S_PLUS'] * 0.8).astype(np.float32)
        states['SCORE_STRUCTURE_BULLISH_RESONANCE_A'] = (states['SCORE_STRUCTURE_BULLISH_RESONANCE_S_PLUS'] * 0.6).astype(np.float32)
        states['SCORE_STRUCTURE_BULLISH_RESONANCE_B'] = (states['SCORE_STRUCTURE_BULLISH_RESONANCE_S_PLUS'] * 0.4).astype(np.float32)

        # 4.2 反转信号合成 (Reversal = Gated by External Macro Position)
        bullish_short_force_rev = (overall_dynamic_health[1] * overall_dynamic_health[5])**0.5
        bullish_medium_trend_rev = (overall_dynamic_health[13] * overall_dynamic_health[21])**0.5
        bullish_long_inertia_rev = overall_dynamic_health[55]
        overall_bullish_reversal_trigger = (bullish_short_force_rev * reversal_tf_weights['short'] +
                                            bullish_medium_trend_rev * reversal_tf_weights['medium'] +
                                            bullish_long_inertia_rev * reversal_tf_weights['long'])
        final_bottom_reversal_score = bottom_context_score * overall_bullish_reversal_trigger
        states['SCORE_STRUCTURE_BOTTOM_REVERSAL_S_PLUS'] = (final_bottom_reversal_score ** exponent).astype(np.float32)
        states['SCORE_STRUCTURE_BOTTOM_REVERSAL_S'] = (states['SCORE_STRUCTURE_BOTTOM_REVERSAL_S_PLUS'] * 0.8).astype(np.float32)
        states['SCORE_STRUCTURE_BOTTOM_REVERSAL_A'] = (states['SCORE_STRUCTURE_BOTTOM_REVERSAL_S_PLUS'] * 0.6).astype(np.float32)
        states['SCORE_STRUCTURE_BOTTOM_REVERSAL_B'] = (states['SCORE_STRUCTURE_BOTTOM_REVERSAL_S_PLUS'] * 0.4).astype(np.float32)

        # 4.3 对称实现下跌共振和顶部反转信号
        bearish_resonance_health = {p: (1 - overall_static_health[p]) * (1 - overall_dynamic_health[p]) for p in periods}
        bearish_short_force_res = (bearish_resonance_health[1] * bearish_resonance_health[5])**0.5
        bearish_medium_trend_res = (bearish_resonance_health[13] * bearish_resonance_health[21])**0.5
        bearish_long_inertia_res = bearish_resonance_health[55]
        overall_bearish_resonance = (bearish_short_force_res * resonance_tf_weights['short'] +
                                     bearish_medium_trend_res * resonance_tf_weights['medium'] +
                                     bearish_long_inertia_res * resonance_tf_weights['long'])
        states['SCORE_STRUCTURE_BEARISH_RESONANCE_S_PLUS'] = (overall_bearish_resonance ** exponent).astype(np.float32)
        states['SCORE_STRUCTURE_BEARISH_RESONANCE_S'] = (states['SCORE_STRUCTURE_BEARISH_RESONANCE_S_PLUS'] * 0.8).astype(np.float32)
        states['SCORE_STRUCTURE_BEARISH_RESONANCE_A'] = (states['SCORE_STRUCTURE_BEARISH_RESONANCE_S_PLUS'] * 0.6).astype(np.float32)
        states['SCORE_STRUCTURE_BEARISH_RESONANCE_B'] = (states['SCORE_STRUCTURE_BEARISH_RESONANCE_S_PLUS'] * 0.4).astype(np.float32)
        
        bearish_short_force_rev = ((1 - overall_dynamic_health[1]) * (1 - overall_dynamic_health[5]))**0.5
        bearish_medium_trend_rev = ((1 - overall_dynamic_health[13]) * (1 - overall_dynamic_health[21]))**0.5
        bearish_long_inertia_rev = (1 - overall_dynamic_health[55])
        overall_bearish_reversal_trigger = (bearish_short_force_rev * reversal_tf_weights['short'] +
                                            bearish_medium_trend_rev * reversal_tf_weights['medium'] +
                                            bearish_long_inertia_rev * reversal_tf_weights['long'])
        final_top_reversal_score = top_context_score * overall_bearish_reversal_trigger
        states['SCORE_STRUCTURE_TOP_REVERSAL_S_PLUS'] = (final_top_reversal_score ** exponent).astype(np.float32)
        states['SCORE_STRUCTURE_TOP_REVERSAL_S'] = (states['SCORE_STRUCTURE_TOP_REVERSAL_S_PLUS'] * 0.8).astype(np.float32)
        states['SCORE_STRUCTURE_TOP_REVERSAL_A'] = (states['SCORE_STRUCTURE_TOP_REVERSAL_S_PLUS'] * 0.6).astype(np.float32)
        states['SCORE_STRUCTURE_TOP_REVERSAL_B'] = (states['SCORE_STRUCTURE_TOP_REVERSAL_S_PLUS'] * 0.4).astype(np.float32)
        
        return states

    def _calculate_mtf_health_v2(self, df: pd.DataFrame, periods: list, norm_window: int) -> Tuple[Dict[int, pd.Series], Dict[int, pd.Series]]:
        """【V2.0 · 上下文门控版】计算MTF支柱的静态与动态健康度"""
        static_health, dynamic_health = {}, {}
        
        # 静态健康度：周线结构对齐度
        weekly_static_cols = [col for col in df.columns if 'EMA' in col and col.endswith('_W')]
        if len(weekly_static_cols) > 1:
            short_ma_cols_list = weekly_static_cols[:-1]
            long_ma_cols_list = weekly_static_cols[1:]
            alignment_sum = np.sum(df[short_ma_cols_list].values > df[long_ma_cols_list].values, axis=1)
            static_alignment_score = pd.Series(alignment_sum / (len(weekly_static_cols) - 1), index=df.index, dtype=np.float32)
        else:
            static_alignment_score = pd.Series(0.5, index=df.index, dtype=np.float32)

        # 动态健康度：日线与周线的动态指标（斜率、加速度）
        daily_momentum = self.strategy.atomic_states.get('SCORE_MA_DYN_RESONANCE', pd.Series(0.5, index=df.index))
        daily_accel = self.strategy.atomic_states.get('SCORE_MA_ACCEL_RESONANCE', pd.Series(0.5, index=df.index))
        weekly_slope_cols = [col for col in df.columns if 'SLOPE' in col and col.endswith('_W')]
        weekly_accel_cols = [col for col in df.columns if 'ACCEL' in col and col.endswith('_W')]
        
        def get_weekly_avg_score(cols: list[str]) -> pd.Series:
            if not cols: return pd.Series(0.5, index=df.index, dtype=np.float32)
            score_arrays = [self._normalize_score(df.get(c), norm_window, df.index).values for c in cols]
            return pd.Series(np.mean(np.stack(score_arrays, axis=0), axis=0), index=df.index, dtype=np.float32)
            
        weekly_momentum = get_weekly_avg_score(weekly_slope_cols)
        weekly_accel = get_weekly_avg_score(weekly_accel_cols)
        
        dynamic_health_series = (daily_momentum.values * daily_accel.values * weekly_momentum.values * weekly_accel.values)**(1/4)
        dynamic_health_series = pd.Series(dynamic_health_series, index=df.index, dtype=np.float32)

        for p in periods:
            static_health[p] = static_alignment_score
            dynamic_health[p] = dynamic_health_series
            
        return static_health, dynamic_health

    def _calculate_pattern_health_v2(self, df: pd.DataFrame, periods: list, norm_window: int) -> Tuple[Dict[int, pd.Series], Dict[int, pd.Series]]:
        """【V2.0 · 上下文门控版】计算形态支柱的静态与动态健康度"""
        static_health, dynamic_health = {}, {}
        
        # 静态健康度：是否存在吸筹或盘整形态
        is_accumulation = df.get('is_accumulation_D', 0).astype(float)
        is_consolidation = df.get('is_consolidation_D', 0).astype(float)
        static_score = np.maximum(is_accumulation, is_consolidation)
        static_score = pd.Series(static_score, index=df.index).replace(0, 0.5) # 无形态时为中性分

        # 动态健康度：是否存在突破事件
        is_breakthrough = df.get('is_breakthrough_D', 0).astype(float)
        dynamic_score = pd.Series(is_breakthrough, index=df.index).replace(0, 0.5) # 无突破时为中性分

        for p in periods:
            static_health[p] = static_score
            dynamic_health[p] = dynamic_score
            
        return static_health, dynamic_health

    def _calculate_ma_health(self, df: pd.DataFrame, periods: list, norm_window: int) -> Dict[int, pd.Series]:
        """
        【V1.1 性能优化版】计算均线支柱的健康度
        - 核心升级 (本次修改):
          - [性能优化] 将循环内的Series乘法操作，优化为直接在NumPy数组上进行计算。
        - 收益: 避免了在循环中为每个周期创建多个中间Series，提升了计算效率。
        """
        health = {}
        for p in periods:
            static_col = f'EMA_{p}_D' if p > 1 else 'close_D'
            slope_lookback = 5 if p <= 5 else p
            accel_lookback = 5 if p <= 5 else p
            slope_col = f'SLOPE_{slope_lookback}_{static_col}'
            accel_col = f'ACCEL_{accel_lookback}_{static_col}'
            static_score = self._normalize_score(df.get(static_col), norm_window, df.index)
            slope_score = self._normalize_score(df.get(slope_col), norm_window, df.index)
            accel_score = self._normalize_score(df.get(accel_col), norm_window, df.index)
            # 直接在NumPy数组上进行计算，避免创建中间Series
            health_arr = static_score.values * slope_score.values * accel_score.values
            health[p] = pd.Series(health_arr, index=df.index, dtype=np.float32)
            
        return health

    def _calculate_mechanics_health(self, df: pd.DataFrame, periods: list, norm_window: int) -> Dict[int, pd.Series]:
        """
        【V1.2 性能优化版】计算力学支柱的健康度
        - 核心升级 (本次修改):
          - [性能优化] 将循环内的Series乘法操作，优化为直接在NumPy数组上进行计算，并将与周期无关的计算移出循环。
          - [逻辑修正] (V1.1逻辑保留) 修正了对 `concentration_90pct_D` 斜率的评分逻辑。
        - 收益: 避免了在循环中为每个周期创建多个中间Series，并减少了重复计算，提升了计算效率。
        """
        health = {}
        # 预先计算与周期p无关的能量分，并直接获取其NumPy数组
        energy_series = df.get('energy_ratio_D')
        energy_arr = self._normalize_score(energy_series, norm_window, df.index).values
        
        for p in periods:
            # 成本斜率：越高越好 (ascending=True, 默认)
            cost_slope_series = df.get(f'SLOPE_{p}_peak_cost_D')
            cost_slope_arr = self._normalize_score(cost_slope_series, norm_window, df.index).values
            # 集中度斜率：越低（负值）越好，代表筹码在集中，所以 ascending=False
            conc_slope_series = df.get(f'SLOPE_{p}_concentration_90pct_D')
            conc_slope_arr = self._normalize_score(conc_slope_series, norm_window, df.index, ascending=False).values
            # 直接在NumPy数组上进行计算
            # 使用几何平均值融合三个维度
            health_arr = (cost_slope_arr * conc_slope_arr * energy_arr)**(1/3)
            health[p] = pd.Series(health_arr, index=df.index, dtype=np.float32)
            
        return health

    def _calculate_mtf_health(self, df: pd.DataFrame, periods: list, norm_window: int) -> Dict[int, pd.Series]:
        """
        【V1.2 逻辑修复版】计算MTF支柱的健康度
        - 核心修复 (本次修改):
          - [BUG修复] 修复了因使用未导入的 `List` 类型提示而导致的 `NameError`。已将其修正为 Python 3.9+ 支持的内置 `list` 类型提示。
        - 核心升级 (V1.1逻辑保留):
          - [性能优化] 将原有的 `pd.concat(...).mean()` 逻辑重构为使用 `np.stack` 和 `np.mean`。
          - [性能优化] 将最终融合的链式乘法也改为在NumPy数组上进行。
        - 收益: 修复了导致程序崩溃的严重BUG，同时保持了V1.1版本带来的显著性能提升。
        """
        health = {}
        daily_momentum = self.strategy.atomic_states.get('SCORE_MA_DYN_RESONANCE', pd.Series(0.5, index=df.index))
        daily_accel = self.strategy.atomic_states.get('SCORE_MA_ACCEL_RESONANCE', pd.Series(0.5, index=df.index))
        weekly_slope_cols = [col for col in df.columns if 'SLOPE' in col and col.endswith('_W')]
        weekly_accel_cols = [col for col in df.columns if 'ACCEL' in col and col.endswith('_W')]
        # 辅助函数，用于将多列Series高效地平均为一个Series
        # 修复类型提示错误，将 `List[str]` 改为 `list[str]`
        def get_weekly_avg_score(cols: list[str]) -> pd.Series:
        
            if not cols:
                return pd.Series(0.5, index=df.index, dtype=np.float32)
            # 1. 获取所有归一化分数的NumPy数组列表
            score_arrays = [self._normalize_score(df.get(c), norm_window, df.index).values for c in cols]
            # 2. 将数组列表堆叠成一个 (信号数量, 时间序列长度) 的2D数组
            stacked_scores = np.stack(score_arrays, axis=0)
            # 3. 沿信号维度（axis=0）计算平均值
            mean_values = np.mean(stacked_scores, axis=0)
            # 4. 将结果包装回Pandas Series
            return pd.Series(mean_values, index=df.index, dtype=np.float32)
        weekly_momentum = get_weekly_avg_score(weekly_slope_cols)
        weekly_accel = get_weekly_avg_score(weekly_accel_cols)
        # 使用NumPy数组进行最终融合
        health_arr = (daily_momentum.values * daily_accel.values * weekly_momentum.values * weekly_accel.values)**(1/4)
        health_series = pd.Series(health_arr, index=df.index, dtype=np.float32)
        for p in periods: # 尽管MTF不直接依赖periods，但为了结构统一，仍然循环
            health[p] = health_series # 所有周期的MTF健康度都一样
        return health

    def _calculate_pillar_health(self, df: pd.DataFrame, periods: list, norm_window: int, static_col: str, slope_prefix: str, accel_prefix: str) -> Dict[int, pd.Series]:
        """
        【V1.0 新增】计算单个支柱在所有周期的健康度。
        这是一个通用的辅助函数，用于对任何一个支柱进行“静态 x 动态 x 加速”的三维交叉验证。
        """
        health = {}
        for p in periods:
            static_series = df.get(static_col)
            slope_series = df.get(f"{slope_prefix}{p}_{static_col}")
            accel_series = df.get(f"{accel_prefix}{p}_{static_col}")

            static_score = self._normalize_score(static_series, norm_window, df.index)
            slope_score = self._normalize_score(slope_series, norm_window, df.index)
            accel_score = self._normalize_score(accel_series, norm_window, df.index)
            
            health[p] = static_score * slope_score * accel_score
        return health

    def _calculate_pattern_health(self, df: pd.DataFrame, periods: list, norm_window: int) -> Dict[int, pd.Series]:
        """【V1.0 新增】计算形态支柱的健康度"""
        health = {}
        is_breakthrough = df.get('is_breakthrough_D', 0).astype(float)
        is_accumulation = df.get('is_accumulation_D', 0).astype(float)
        
        # 形态健康度是一个事件驱动的信号，有形态发生时为1，否则为0.5（中性）
        pattern_bullish_score = np.maximum(is_breakthrough, is_accumulation)
        pattern_bullish_score = pattern_bullish_score.replace(0, 0.5)

        for p in periods:
            health[p] = pattern_bullish_score
        return health





















