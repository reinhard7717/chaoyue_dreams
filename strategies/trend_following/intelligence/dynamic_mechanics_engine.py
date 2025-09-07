# 文件: strategies/trend_following/intelligence/dynamic_mechanics_engine.py
# 动态力学引擎
import pandas as pd
import numpy as np
from scipy.stats import linregress
from typing import Dict

class DynamicMechanicsEngine:
    def __init__(self, strategy_instance):
        """
        初始化动态力学引擎。
        :param strategy_instance: 策略主实例的引用，用于访问 df_indicators。
        """
        self.strategy = strategy_instance

    def run_dynamic_analysis_command(self) -> None:
        """
        【V1.0 新增】动态力学分析总指挥官
        - 核心职责:
          1. 作为动态力学引擎的唯一入口，统一调度所有底层的力学诊断模块。
          2. 在所有原子力学分数生成后，执行“元融合”，将各维度最关键的看涨分数
             融合成一个顶层的“市场整体力学健康度”元分数。
        - 收益: 简化了上层调用，并创造了一个更高维度的、概括性的力学数值信号。
        """
        # print("    -> [动态力学分析总指挥官 V1.0] 启动...")
        all_states = {}
        df = self.strategy.df_indicators
        # --- 步骤 1: 依次调用所有底层诊断模块，并收集其产出的原子分数 ---
        all_states.update(self.run_force_vector_analysis_scores())
        all_states.update(self.diagnose_multi_timeframe_micro_dynamics_scores(df))
        all_states.update(self.diagnose_multi_timeframe_dynamics_scores(df))
        all_states.update(self.diagnose_behavioral_mechanics_scores(df))
        # --- 步骤 2: 执行元融合，生成“整体力学健康度”元分数 ---
        p_module = self.strategy.params.get('dynamic_mechanics_meta_fusion_params', {})
        if p_module.get('enabled', True):
            # print("        -> [力学元融合模块] 启动...")
            # 定义各维度权重
            weights = p_module.get('weights', {
                'force_vector': 0.35,
                'micro_dynamics': 0.25,
                'mtf_dynamics': 0.20,
                'behavioral': 0.20
            })
            # 提取各维度最关键的看涨分数
            default_series = pd.Series(0.0, index=df.index, dtype=np.float32)
            fv_score = all_states.get('SCORE_FV_CONFIRMED_IGNITION_S_PLUS', default_series)
            micro_score = all_states.get('SCORE_DYN_BULLISH_RESONANCE_S', default_series)
            mtf_score = all_states.get('SCORE_MTF_PLATFORM_IGNITION_S_PLUS', default_series)
            behavioral_score = all_states.get('SCORE_BEHAVIOR_ENGINE_IGNITION_A', default_series)
            # 加权平均
            meta_score = (
                fv_score * weights['force_vector'] +
                micro_score * weights['micro_dynamics'] +
                mtf_score * weights['mtf_dynamics'] +
                behavioral_score * weights['behavioral']
            ).clip(0, 1)
            all_states['SCORE_DYN_OVERALL_BULLISH_MOMENTUM_S'] = meta_score
            print("        -> [力学元融合模块] “整体力学健康度”元分数计算完毕。")
        # --- 步骤 3: 将所有生成的原子分数和元分数一次性更新到原子状态库 ---
        if all_states:
            self.strategy.atomic_states.update(all_states)
            print(f"    -> [动态力学分析总指挥官 V1.0] 分析完毕，共更新 {len(all_states)} 个力学信号。")

    def _normalize_series(self, series: pd.Series, norm_window: int, min_periods: int, ascending: bool = True) -> np.ndarray:
        """
        辅助函数：将Pandas Series进行滚动窗口排名归一化，并返回NumPy数组。
        - 提升为类的私有方法，以供所有诊断引擎复用，避免重复定义。
        :param series: 原始数据Series。
        :param norm_window: 归一化的滚动窗口大小。
        :param min_periods: 滚动窗口的最小观测期。
        :param ascending: 归一化方向，True表示值越大分数越高。
        :return: 归一化后的0-1分数（np.float32类型的NumPy数组）。
        """
        # 使用滚动窗口计算百分比排名，空值用0.5填充，代表中位数水平
        rank = series.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        # 根据ascending参数决定排名方向，并直接返回NumPy数组以提升后续计算效率
        return (rank if ascending else 1 - rank).values.astype(np.float32)

    def run_force_vector_analysis_scores(self) -> Dict[str, pd.Series]:
        """
        【V4.4 依赖修复版】宏观力矢量评分引擎
        - 核心修复 (本次修改):
          - 修复了原代码依赖的 `entry_score` 和 `risk_score` 在数据层不存在的问题。
          - 使用 `chip_health_score_D` (筹码健康分) 作为 `entry_score` 的代理，代表机会。
          - 使用 `turnover_from_winners_ratio_D` (获利盘换手率) 作为 `risk_score` 的代理，代表风险。
        - 性能优化: (保留V4.3的优化)
        """
        # print("        -> [宏观力矢量评分引擎 V4.4 依赖修复版] 启动...") 
        states = {}
        df = self.strategy.df_indicators
        # --- 1. 军备检查 ---
        # 检查修复后的依赖列
        required_cols = ['chip_health_score_D', 'turnover_from_winners_ratio_D']
        if not all(col in df.columns for col in required_cols):
            print(f"          -> [警告] 缺少 {required_cols}，宏观力矢量分析跳过。")
            return states
        norm_window = 120
        min_periods = max(1, norm_window // 5)
        # --- 2. 计算斜率和加速度 (Pandas时序操作) ---
        # 使用新的代理列进行计算
        entry_score_series = df['chip_health_score_D']
        risk_score_series = df['turnover_from_winners_ratio_D']
        entry_score_slope = (-2 * entry_score_series.shift(4) - entry_score_series.shift(3) + entry_score_series.shift(1) + 2 * entry_score_series) / 10
        risk_score_slope = (-2 * risk_score_series.shift(4) - risk_score_series.shift(3) + risk_score_series.shift(1) + 2 * risk_score_series) / 10
        entry_score_accel = entry_score_slope.diff()
        risk_score_accel = risk_score_slope.diff()
        # --- 3. 核心动态分数计算 (NumPy高性能计算) ---
        offense_momentum_arr = self._normalize_series(entry_score_slope, norm_window, min_periods)
        offense_accel_arr = self._normalize_series(entry_score_accel, norm_window, min_periods)
        risk_momentum_arr = self._normalize_series(risk_score_slope, norm_window, min_periods)
        risk_accel_arr = self._normalize_series(risk_score_accel, norm_window, min_periods)
        risk_decel_arr = 1.0 - risk_accel_arr
        # --- 4. 交叉验证与信号生成 (纯NumPy数组运算) ---
        pure_offensive_momentum_arr = offense_accel_arr * risk_decel_arr
        chaotic_expansion_arr = offense_accel_arr * risk_accel_arr
        # --- 5. 静态-动态融合 (纯NumPy数组运算) ---
        key = 'SCORE_PLATFORM_OVERALL_QUALITY'
        if key in self.strategy.atomic_states:
            static_platform_quality_arr = self.strategy.atomic_states[key].values.astype(np.float32)
        else:
            static_platform_quality_arr = np.zeros(len(df), dtype=np.float32)
        confirmed_ignition_arr = static_platform_quality_arr * pure_offensive_momentum_arr
        # --- 6. 结果封装 (批量转换为Pandas Series) ---
        states = {
            'SCORE_FV_OFFENSE_MOMENTUM': pd.Series(offense_momentum_arr, index=df.index),
            'SCORE_FV_OFFENSE_ACCEL': pd.Series(offense_accel_arr, index=df.index),
            'SCORE_FV_RISK_MOMENTUM': pd.Series(risk_momentum_arr, index=df.index),
            'SCORE_FV_RISK_ACCEL': pd.Series(risk_accel_arr, index=df.index),
            'SCORE_FV_PURE_OFFENSIVE_MOMENTUM': pd.Series(pure_offensive_momentum_arr, index=df.index),
            'SCORE_FV_CHAOTIC_EXPANSION_RISK': pd.Series(chaotic_expansion_arr, index=df.index),
            'SCORE_FV_CONFIRMED_IGNITION_S_PLUS': pd.Series(confirmed_ignition_arr, index=df.index),
        }
        self.strategy.atomic_states.update(states)
        print("        -> [宏观力矢量评分引擎 V4.4 依赖修复版] 分析完毕。") # 更新打印信息
        return states

    def diagnose_multi_timeframe_micro_dynamics_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.5 数值化升级版】多时间维度微观力学评分引擎
        - 核心逻辑: (业务逻辑不变)
        - 核心升级 (本次修改): 消费端同步升级，使用新的数值化分级评分 'SCORE_VOL_COMPRESSION_LEVEL'
                          替代旧的布尔信号。分数越高，对看涨共振的加成越大。
        """
        # print("        -> [多时间维度微观力学评分引擎 V2.5] 启动...")
        states = {}
        # --- 1. 军备检查 ---
        norm_window = 120
        min_periods = max(1, norm_window // 5)
        required_cols = {
            'SLOPE_5_close_D', 'SLOPE_21_close_D', 'ACCEL_5_close_D', 'ACCEL_21_close_D',
            'SLOPE_5_concentration_90pct_D', 'SLOPE_21_concentration_90pct_D', 'SLOPE_55_concentration_90pct_D',
            'SLOPE_5_BBW_21_2.0_D', 'ACCEL_5_BBW_21_2.0_D', 'ACCEL_5_volume_D'
        }
        if not required_cols.issubset(df.columns):
            missing = list(required_cols - set(df.columns))
            print(f"          -> [警告] 多维微观力学引擎缺少关键数据: {missing}，模块已跳过！")
            return states
        # --- 2. 核心动态分数计算 (NumPy高性能计算) ---
        price_momentum_5d_arr = self._normalize_series(df['SLOPE_5_close_D'], norm_window, min_periods)
        price_momentum_21d_arr = self._normalize_series(df['SLOPE_21_close_D'], norm_window, min_periods)
        price_accel_5d_arr = self._normalize_series(df['ACCEL_5_close_D'], norm_window, min_periods)
        price_accel_21d_arr = self._normalize_series(df['ACCEL_21_close_D'], norm_window, min_periods)
        price_decel_5d_arr = 1.0 - price_accel_5d_arr
        chip_conc_5d_arr = self._normalize_series(df['SLOPE_5_concentration_90pct_D'], norm_window, min_periods, ascending=False)
        chip_conc_21d_arr = self._normalize_series(df['SLOPE_21_concentration_90pct_D'], norm_window, min_periods, ascending=False)
        chip_conc_55d_arr = self._normalize_series(df['SLOPE_55_concentration_90pct_D'], norm_window, min_periods, ascending=False)
        vol_squeeze_arr = self._normalize_series(df['SLOPE_5_BBW_21_2.0_D'], norm_window, min_periods, ascending=False)
        vol_ignition_arr = self._normalize_series(df['ACCEL_5_BBW_21_2.0_D'], norm_window, min_periods, ascending=True)
        volatility_ignition_arr = vol_squeeze_arr * vol_ignition_arr
        volume_accel_arr = self._normalize_series(df['ACCEL_5_volume_D'], norm_window, min_periods, ascending=True)
        # --- 3. 计算多周期“共振分” (纯NumPy数组运算) ---
        price_momentum_resonance_arr = np.mean(np.array([price_momentum_5d_arr, price_momentum_21d_arr]), axis=0)
        price_accel_resonance_arr = np.mean(np.array([price_accel_5d_arr, price_accel_21d_arr]), axis=0)
        chip_conc_resonance_arr = np.mean(np.array([chip_conc_5d_arr, chip_conc_21d_arr, chip_conc_55d_arr]), axis=0)
        # --- 4. 静态-动态交叉验证 (纯NumPy数组运算) ---
        # 消费新的数值化评分 'SCORE_VOL_COMPRESSION_LEVEL'
        key = 'SCORE_VOL_COMPRESSION_LEVEL'
        if key in self.strategy.atomic_states:
            # 直接获取分数值 (0.0 to 1.0)
            static_squeeze_score_arr = self.strategy.atomic_states[key].values
        else:
            # 默认值是0.0
            static_squeeze_score_arr = np.full(len(df), 0.0, dtype=np.float32)
        # 将分级评分作为加权因子。乘数范围从 0.5 (无压缩) 到 1.5 (S级压缩)
        bullish_resonance_arr = (
            price_momentum_resonance_arr * price_accel_resonance_arr * chip_conc_resonance_arr *
            volatility_ignition_arr * volume_accel_arr * (static_squeeze_score_arr + 0.5)
        )
        short_term_weakness_arr = 1.0 - price_momentum_5d_arr
        divergence_risk_arr = price_momentum_21d_arr * short_term_weakness_arr
        exhaustion_divergence_risk_arr = volume_accel_arr * price_decel_5d_arr
        # --- 5. 结果封装 (批量转换为Pandas Series) ---
        states = {
            'SCORE_DYN_VOLATILITY_IGNITION': pd.Series(volatility_ignition_arr, index=df.index),
            'SCORE_DYN_PRICE_MOMENTUM_RESONANCE': pd.Series(price_momentum_resonance_arr, index=df.index),
            'SCORE_DYN_PRICE_ACCEL_RESONANCE': pd.Series(price_accel_resonance_arr, index=df.index),
            'SCORE_DYN_CHIP_CONCENTRATION_RESONANCE': pd.Series(chip_conc_resonance_arr, index=df.index),
            'SCORE_DYN_BULLISH_RESONANCE_S': pd.Series(bullish_resonance_arr, index=df.index),
            'SCORE_DYN_DIVERGENCE_RISK_A': pd.Series(divergence_risk_arr, index=df.index),
            'SCORE_DYN_EXHAUSTION_DIVERGENCE_RISK_S': pd.Series(exhaustion_divergence_risk_arr, index=df.index),
        }
        print("        -> [多时间维度微观力学评分引擎 V2.5] 分析完毕。") 
        return states

    def diagnose_multi_timeframe_dynamics_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.4 终极优化版】多时间维度力学评分引擎
        - 核心逻辑: (业务逻辑不变)
        - 性能优化 (本次修改):
          - 关键优化：使用纯NumPy切片实现shift操作，消除了`NumPy->Pandas->NumPy`的昂贵转换开销。
          - 集中化数据提取：在方法开始时一次性提取所需Series，避免重复访问DataFrame。
        """
        # print("        -> [多时间维度力学评分引擎 V2.4] 启动...") 
        states = {}
        # --- 1. 军备检查 ---
        norm_window = 120
        min_periods = max(1, norm_window // 5)
        required_cols = {
            'SLOPE_5_close_D', 'SLOPE_21_close_D', 'ACCEL_21_close_D',
            'SLOPE_5_concentration_90pct_D', 'SLOPE_21_concentration_90pct_D',
            'SLOPE_21_winner_profit_margin_D'
        }
        if not required_cols.issubset(df.columns):
            missing = list(required_cols - set(df.columns))
            print(f"          -> [警告] 多维力学引擎缺少关键数据: {missing}，模块已跳过！")
            return states
        # --- 2. 核心动态分数计算 (NumPy高性能计算) ---
        # 集中提取所需Series，提高代码清晰度和执行效率
        slope_5_close = df['SLOPE_5_close_D']
        slope_21_close = df['SLOPE_21_close_D']
        accel_21_close = df['ACCEL_21_close_D']
        slope_5_conc = df['SLOPE_5_concentration_90pct_D']
        slope_21_conc = df['SLOPE_21_concentration_90pct_D']
        slope_21_profit = df['SLOPE_21_winner_profit_margin_D']
        price_rising_short_arr = self._normalize_series(slope_5_close, norm_window, min_periods)
        price_stable_or_down_short_arr = 1.0 - price_rising_short_arr
        price_rising_long_arr = self._normalize_series(slope_21_close, norm_window, min_periods)
        price_fall_decel_long_arr = self._normalize_series(accel_21_close, norm_window, min_periods)
        chip_conc_short_arr = self._normalize_series(slope_5_conc, norm_window, min_periods, ascending=False)
        chip_conc_long_arr = self._normalize_series(slope_21_conc, norm_window, min_periods, ascending=False)
        chip_diverging_long_arr = 1.0 - chip_conc_long_arr
        profit_margin_rising_long_arr = self._normalize_series(slope_21_profit, norm_window, min_periods)
        profit_margin_shrinking_long_arr = 1.0 - profit_margin_rising_long_arr
        # --- 3. 交叉验证信号生成 (纯NumPy数组运算) ---
        trend_resonance_arr = price_rising_short_arr * price_rising_long_arr * chip_conc_short_arr
        structural_weakness_risk_arr = price_rising_short_arr * chip_diverging_long_arr * profit_margin_shrinking_long_arr
        # 关键性能优化：使用纯NumPy实现shift(1).fillna(0.5)，避免了昂贵的 `np->pd->np` 转换
        n = len(price_rising_long_arr)
        shifted_price_rising_long_arr = np.empty(n, dtype=np.float32)
        shifted_price_rising_long_arr[0] = 0.5  # 模拟fillna(0.5)
        shifted_price_rising_long_arr[1:] = price_rising_long_arr[:-1] # 执行shift
        prev_price_falling_long_arr = 1.0 - shifted_price_rising_long_arr
        bottom_inflection_arr = prev_price_falling_long_arr * price_fall_decel_long_arr * price_rising_short_arr
        profit_erosion_risk_arr = price_rising_short_arr * profit_margin_shrinking_long_arr
        stealth_accumulation_arr = price_stable_or_down_short_arr * chip_conc_long_arr
        # --- 4. 静态-动态融合 (纯NumPy数组运算) ---
        key = 'SCORE_PLATFORM_OVERALL_QUALITY'
        if key in self.strategy.atomic_states:
            static_platform_quality_arr = self.strategy.atomic_states[key].values.astype(np.float32)
        else:
            static_platform_quality_arr = np.zeros(len(df), dtype=np.float32)
        platform_ignition_arr = static_platform_quality_arr * trend_resonance_arr
        # --- 5. 结果封装 (批量转换为Pandas Series) ---
        states = {
            'SCORE_MTF_TREND_RESONANCE_A': pd.Series(trend_resonance_arr, index=df.index),
            'SCORE_MTF_STRUCTURAL_WEAKNESS_RISK_S': pd.Series(structural_weakness_risk_arr, index=df.index),
            'SCORE_MTF_BOTTOM_INFLECTION_OPP_A': pd.Series(bottom_inflection_arr, index=df.index),
            'SCORE_MTF_PROFIT_CUSHION_EROSION_RISK_S': pd.Series(profit_erosion_risk_arr, index=df.index),
            'SCORE_MTF_STEALTH_ACCUMULATION_OPP_A': pd.Series(stealth_accumulation_arr, index=df.index),
            'SCORE_MTF_PLATFORM_IGNITION_S_PLUS': pd.Series(platform_ignition_arr, index=df.index),
        }
        print("        -> [多时间维度力学评分引擎 V2.4] 分析完毕。") 
        return states

    def diagnose_behavioral_mechanics_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.3 终极优化版】行为力学评分引擎
        - 核心逻辑: (业务逻辑不变)
        - 性能优化 (本次修改):
          - 集中化数据提取：在方法开始时一次性提取所需Series，避免重复访问DataFrame。
          - 优化静态信号的获取逻辑，避免创建不必要的临时Pandas Series对象。
        """
        # print("        -> [行为力学评分引擎 V2.3] 启动...") 
        states = {}
        # --- 1. 军备检查 ---
        norm_window = 120
        min_periods = max(1, norm_window // 5)
        required_cols = {
            'SLOPE_5_close_D', 'ACCEL_5_close_D',
            'ACCEL_5_VPA_EFFICIENCY_D', 'ACCEL_21_VPA_EFFICIENCY_D',
            'ACCEL_5_turnover_from_winners_ratio_D', 'ACCEL_21_turnover_from_winners_ratio_D',
            'ACCEL_5_turnover_from_losers_ratio_D', 'ACCEL_21_turnover_from_losers_ratio_D'
        }
        if not required_cols.issubset(df.columns):
            missing = list(required_cols - set(df.columns))
            print(f"          -> [警告] 行为力学引擎缺少关键数据: {missing}，模块已跳过！")
            return states
        # --- 2. 核心动态分数计算 (NumPy高性能计算) ---
        # 集中提取所需Series，提高代码清晰度和执行效率
        slope_5_close = df['SLOPE_5_close_D']
        accel_5_close = df['ACCEL_5_close_D']
        accel_5_vpa = df['ACCEL_5_VPA_EFFICIENCY_D']
        accel_21_vpa = df['ACCEL_21_VPA_EFFICIENCY_D']
        accel_5_winner = df['ACCEL_5_turnover_from_winners_ratio_D']
        accel_21_winner = df['ACCEL_21_turnover_from_winners_ratio_D']
        accel_5_loser = df['ACCEL_5_turnover_from_losers_ratio_D']
        accel_21_loser = df['ACCEL_21_turnover_from_losers_ratio_D']
        price_rising_arr = self._normalize_series(slope_5_close, norm_window, min_periods)
        price_falling_arr = 1.0 - price_rising_arr
        price_accel_arr = self._normalize_series(accel_5_close, norm_window, min_periods)
        efficiency_accel_5d_arr = self._normalize_series(accel_5_vpa, norm_window, min_periods)
        efficiency_accel_21d_arr = self._normalize_series(accel_21_vpa, norm_window, min_periods)
        winner_selling_accel_5d_arr = self._normalize_series(accel_5_winner, norm_window, min_periods)
        winner_selling_accel_21d_arr = self._normalize_series(accel_21_winner, norm_window, min_periods)
        loser_selling_decel_5d_arr = self._normalize_series(accel_5_loser, norm_window, min_periods, ascending=False)
        loser_selling_decel_21d_arr = self._normalize_series(accel_21_loser, norm_window, min_periods, ascending=False)
        # --- 3. 共振分与交叉验证信号生成 (纯NumPy数组运算) ---
        efficiency_accel_resonance_arr = (efficiency_accel_5d_arr + efficiency_accel_21d_arr) / 2
        efficiency_decel_resonance_arr = 1.0 - efficiency_accel_resonance_arr
        winner_selling_accel_resonance_arr = (winner_selling_accel_5d_arr + winner_selling_accel_21d_arr) / 2
        loser_selling_decel_resonance_arr = (loser_selling_decel_5d_arr + loser_selling_decel_21d_arr) / 2
        engine_ignition_arr = price_accel_arr * efficiency_accel_resonance_arr
        engine_stalling_risk_arr = price_rising_arr * efficiency_decel_resonance_arr
        panic_selling_risk_arr = price_falling_arr * winner_selling_accel_resonance_arr
        capitulation_exhaustion_arr = price_falling_arr * loser_selling_decel_resonance_arr
        # --- 4. 静态-动态融合 (纯NumPy数组运算) ---
        key = 'PRICE_STATE_ZONE_SCORE'
        if key in self.strategy.atomic_states:
            static_price_zone_arr = self.strategy.atomic_states[key].values.astype(np.float32)
        else:
            static_price_zone_arr = np.zeros(len(df), dtype=np.float32)
        top_divergence_risk_arr = static_price_zone_arr * engine_stalling_risk_arr
        # --- 5. 结果封装 (批量转换为Pandas Series) ---
        states = {
            'SCORE_BEHAVIOR_ENGINE_IGNITION_A': pd.Series(engine_ignition_arr, index=df.index),
            'SCORE_BEHAVIOR_ENGINE_STALLING_RISK_S': pd.Series(engine_stalling_risk_arr, index=df.index),
            'SCORE_BEHAVIOR_PANIC_SELLING_RISK_S': pd.Series(panic_selling_risk_arr, index=df.index),
            'SCORE_BEHAVIOR_CAPITULATION_EXHAUSTION_OPP_A': pd.Series(capitulation_exhaustion_arr, index=df.index),
            'SCORE_BEHAVIOR_TOP_DIVERGENCE_RISK_S_PLUS': pd.Series(top_divergence_risk_arr, index=df.index),
        }
        print("        -> [行为力学评分引擎 V2.3] 分析完毕。") 
        return states












