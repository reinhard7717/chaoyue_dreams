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
        【V2.0 元融合升级版】动态力学分析总指挥官
        - 核心重构 (本次修改):
          1. 遵循原则4和6: 统一了元融合的输入信号置信度。将原先使用的行为力学'A'级信号 (`SCORE_BEHAVIOR_ENGINE_IGNITION_A`)
             升级为'S'级信号 (`SCORE_BEHAVIOR_ENGINE_IGNITION_S`)，确保最终的“整体力学健康度”元分数由所有引擎的最高置信度看涨信号合成，逻辑更严谨，更符合A股实战的严格要求。
          2. 提升可维护性: 更新了 `weights` 字典中的键名，使其与当前各诊断引擎的最新职能（核心力学、趋势结构等）精确对应，增强了代码的可读性。
        - 收益: 最终的元分数信号质量更高，更能代表市场多个维度同时产生强烈看涨共振的稀缺状态。
        """
        # --- 新增: 修改打印信息 ---
        print("    -> [动态力学分析总指挥官 V2.0 元融合升级版] 启动...")
        all_states = {}
        df = self.strategy.df_indicators
        # --- 步骤 1: 依次调用所有底层诊断模块，并收集其产出的原子分数 ---
        all_states.update(self.run_force_vector_analysis_scores(df))
        all_states.update(self.diagnose_core_mechanics_scores(df))
        all_states.update(self.diagnose_multi_timeframe_dynamics_scores(df))
        all_states.update(self.diagnose_behavioral_mechanics_scores(df))
        # --- 步骤 2: 执行元融合，生成“整体力学健康度”元分数 ---
        p_module = self.strategy.params.get('dynamic_mechanics_meta_fusion_params', {})
        if p_module.get('enabled', True):
            print("        -> [力学元融合模块] 启动...")
            # --- 更新权重字典的键名，使其与引擎功能更匹配 ---
            weights = p_module.get('weights', {
                'force_vector': 0.35,      # 对应 宏观力矢量诊断引擎
                'core_mechanics': 0.25,    # 对应 核心力学诊断引擎 (原 micro_dynamics)
                'trend_structure': 0.20,   # 对应 趋势结构诊断引擎 (原 mtf_dynamics)
                'behavioral': 0.20         # 对应 行为力学评分引擎
            })
            # 提取各维度最关键的看涨分数
            default_series = pd.Series(0.0, index=df.index, dtype=np.float32)
            fv_score = all_states.get('SCORE_FV_CONFIRMED_IGNITION_S_PLUS', default_series)
            core_score = all_states.get('SCORE_DYN_BULLISH_RESONANCE_S', default_series)
            trend_score = all_states.get('SCORE_MTF_PLATFORM_IGNITION_S_PLUS', default_series)
            # --- 将行为力学分数从A级升级为S级，统一置信度标准 ---
            behavioral_score = all_states.get('SCORE_BEHAVIOR_ENGINE_IGNITION_S', default_series)
            # 加权平均
            # --- 使用新的变量名和权重键进行计算 ---
            meta_score = (
                fv_score * weights['force_vector'] +
                core_score * weights['core_mechanics'] +
                trend_score * weights['trend_structure'] +
                behavioral_score * weights['behavioral']
            ).clip(0, 1)
            all_states['SCORE_DYN_OVERALL_BULLISH_MOMENTUM_S'] = meta_score
            print("        -> [力学元融合模块] “整体力学健康度”元分数计算完毕。")
        # --- 步骤 3: 将所有生成的原子分数和元分数一次性更新到原子状态库 ---
        if all_states:
            self.strategy.atomic_states.update(all_states)
            print(f"    -> [动态力学分析总指挥官 V2.0 元融合升级版] 分析完毕，共更新 {len(all_states)} 个力学信号。")

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

    def run_force_vector_analysis_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V5.0 深度交叉验证版】宏观力矢量诊断引擎 (原run_force_vector_analysis_scores)
        - 核心重构 (本次修改):
          1. 遵循原则3: 彻底移除方法内部的斜率/加速度计算，改为直接使用数据层预计算的 `SLOPE_*` 和 `ACCEL_*` 指标。
          2. 遵循原则1: 将分析维度从单一的5日扩展至 `[5, 21, 55]` 日，实现短、中、长周期的动能交叉验证。
          3. 遵循原则4: 废除原有零散信号，重构为“进攻共振”和“风险扩张”两大类信号，并建立严格的B/A/S三级置信度体系。
          4. 遵循原则2: 在S级信号中，引入静态指标（如筹码健康度绝对值）进行同周期交叉验证，增强信号的可靠性。
        - 收益: 信号的逻辑层次更清晰，能够更精确地刻画“健康上涨”与“风险上涨”的本质区别，更贴近A股实战。
        """
        print("        -> [宏观力矢量诊断引擎 V5.0 深度交叉验证版] 启动...")
        states = {}
        df = self.strategy.df_indicators
        # --- 1. 军备检查 (Arsenal Check) ---
        # --- 检查预计算的SLOPE和ACCEL列，而非原始列 ---
        norm_window = 120
        min_periods = max(1, norm_window // 5)
        slope_periods = [5, 21, 55]
        accel_periods = [5, 21] # 加速度在长周期上噪音较大，使用短中期
        required_cols = {'chip_health_score_D'} # 静态值用于S级验证
        for p in slope_periods:
            required_cols.add(f'SLOPE_{p}_chip_health_score_D')
            required_cols.add(f'SLOPE_{p}_turnover_from_winners_ratio_D')
        for p in accel_periods:
            required_cols.add(f'ACCEL_{p}_chip_health_score_D')
            required_cols.add(f'ACCEL_{p}_turnover_from_winners_ratio_D')
        missing = list(required_cols - set(df.columns))
        if missing:
            print(f"          -> [警告] 宏观力矢量引擎 V5.0 缺少关键数据: {sorted(missing)}，模块已跳过！")
            return states
        # --- 2. 核心要素数值化与分组 ---
        # 进攻动能 (筹码健康度)
        offense_mom = {p: self._normalize_series(df[f'SLOPE_{p}_chip_health_score_D'], norm_window, min_periods) for p in slope_periods}
        offense_acc = {p: self._normalize_series(df[f'ACCEL_{p}_chip_health_score_D'], norm_window, min_periods) for p in accel_periods}
        # 风险动能 (获利盘换手率)
        risk_mom = {p: self._normalize_series(df[f'SLOPE_{p}_turnover_from_winners_ratio_D'], norm_window, min_periods) for p in slope_periods}
        risk_acc = {p: self._normalize_series(df[f'ACCEL_{p}_turnover_from_winners_ratio_D'], norm_window, min_periods) for p in accel_periods}
        # 静态值
        static_offense_score = self._normalize_series(df['chip_health_score_D'], norm_window, min_periods)
        # 动能分组
        offense_mom_short_mid = (offense_mom[5] + offense_mom[21]) / 2
        offense_acc_short_mid = (offense_acc[5] + offense_acc[21]) / 2
        risk_mom_short_mid = (risk_mom[5] + risk_mom[21]) / 2
        risk_acc_short_mid = (risk_acc[5] + risk_acc[21]) / 2
        risk_decel_short_mid = 1.0 - risk_acc_short_mid
        # --- 3. 信号合成 (B/A/S 置信度体系) ---
        # 3.1 进攻共振 (Offensive Resonance - Bullish)
        # B级: 短中期进攻动能形成
        offense_resonance_b = offense_mom_short_mid
        # A级: B级信号得到进攻“加速度”的确认
        offense_resonance_a = offense_resonance_b * offense_acc_short_mid
        # S级: A级信号得到风险“减速度”的终极确认 (即“纯粹的进攻”)
        offense_resonance_s = offense_resonance_a * risk_decel_short_mid
        states['SCORE_FV_OFFENSIVE_RESONANCE_B'] = pd.Series(offense_resonance_b, index=df.index)
        states['SCORE_FV_OFFENSIVE_RESONANCE_A'] = pd.Series(offense_resonance_a, index=df.index)
        states['SCORE_FV_OFFENSIVE_RESONANCE_S'] = pd.Series(offense_resonance_s, index=df.index)
        # 3.2 风险扩张 (Risk Expansion - Bearish)
        # B级: 短中期风险动能形成 (获利盘抛压增加)
        risk_expansion_b = risk_mom_short_mid
        # A级: B级信号得到风险“加速度”的确认 (抛压加速)
        risk_expansion_a = risk_expansion_b * risk_acc_short_mid
        # S级: A级信号发生在“静态进攻分”也很高的背景下 (即“高位滞涨/出货”的特征)
        risk_expansion_s = risk_expansion_a * static_offense_score
        states['SCORE_FV_RISK_EXPANSION_B'] = pd.Series(risk_expansion_b, index=df.index)
        states['SCORE_FV_RISK_EXPANSION_A'] = pd.Series(risk_expansion_a, index=df.index)
        states['SCORE_FV_RISK_EXPANSION_S'] = pd.Series(risk_expansion_s, index=df.index)
        # --- 4. 静态-动态融合 (逻辑保留，使用新的S级信号) ---
        # --- 使用新的S级进攻共振信号进行融合 ---
        key = 'SCORE_PLATFORM_OVERALL_QUALITY'
        static_platform_quality_arr = self.strategy.atomic_states.get(key, pd.Series(0.0, index=df.index)).values.astype(np.float32)
        # S+信号: 优质平台(静态)与S级进攻共振(动态)的结合，是最高置信度的点火信号
        confirmed_ignition_arr = static_platform_quality_arr * offense_resonance_s
        states['SCORE_FV_CONFIRMED_IGNITION_S_PLUS'] = pd.Series(confirmed_ignition_arr, index=df.index)
        # --- 5. 清理旧信号 (新信号已完全取代旧信号) ---
        print("        -> [宏观力矢量诊断引擎 V5.0 深度交叉验证版] 分析完毕。")
        return states

    def diagnose_core_mechanics_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V4.0 深度交叉验证版】核心力学诊断引擎
        - 核心重构 (本次修改):
          1. 扩展时间维度: 遵循原则1，将分析周期从 [5, 21, 55] 扩展至 [1, 5, 13, 21, 55]，实现更全面的多时间维度交叉验证。
          2. 细化动能分组: 将价格和筹码动能细分为 短期(1,5日)、中期(13,21日)、长期(55日) 三组，使共振逻辑更清晰。
          3. 引入同周期验证: 遵循原则2，在高置信度(S级)信号中，加入短期价格“斜率加速度”作为同周期验证，确保动能的持续性。
          4. 优化信号逻辑: 遵循原则4，重构了共振与反转信号的合成逻辑，使B/A/S三级置信度划分更严格、更具实战意义。
        - 收益: 信号的鲁棒性和准确性显著提升，能更有效地区分趋势的萌芽、发展和衰竭，更好地适应A股市场的复杂性。
        """

        print("        -> [核心力学诊断引擎 V4.0 深度交叉验证版] 启动...")
        states = {}
        # --- 1. 军备检查 (Arsenal Check) ---
        norm_window = 120
        min_periods = max(1, norm_window // 5)
        # --- 扩展分析的时间周期 ---
        periods = [1, 5, 13, 21, 55] # 定义需要检查的时间周期
        required_cols = set()
        for p in periods:
            # --- 检查所有新周期所需的数据列 ---
            required_cols.add(f'SLOPE_{p}_close_D')
            required_cols.add(f'ACCEL_{p}_close_D')
            required_cols.add(f'SLOPE_{p}_concentration_90pct_D')
        required_cols.update(['SLOPE_5_BBW_21_2.0_D', 'ACCEL_5_BBW_21_2.0_D', 'ACCEL_5_volume_D'])
        missing = list(required_cols - set(df.columns))
        if missing:
            print(f"          -> [警告] 核心力学引擎 V4.0 缺少关键数据: {sorted(missing)}，模块已跳过！")
            return states
        # --- 2. 核心力学要素数值化 (归一化处理) ---
        # --- 按照新的周期列表进行数据归一化 ---
        # 价格动能 (斜率) 与 势能 (加速度)
        price_momentum_scores = {p: self._normalize_series(df[f'SLOPE_{p}_close_D'], norm_window, min_periods) for p in periods}
        price_accel_scores = {p: self._normalize_series(df[f'ACCEL_{p}_close_D'], norm_window, min_periods) for p in periods}
        # 筹码动能 (斜率<0为集中，故ascending=False)
        chip_conc_scores = {p: self._normalize_series(df[f'SLOPE_{p}_concentration_90pct_D'], norm_window, min_periods, ascending=False) for p in periods}
        # 波动率与成交量动能 (逻辑保持不变)
        vol_squeeze_score = self._normalize_series(df['SLOPE_5_BBW_21_2.0_D'], norm_window, min_periods, ascending=False)
        vol_expansion_score = 1.0 - vol_squeeze_score
        vol_ignition_score = self._normalize_series(df['ACCEL_5_BBW_21_2.0_D'], norm_window, min_periods, ascending=True)
        volume_accel_score = self._normalize_series(df['ACCEL_5_volume_D'], norm_window, min_periods, ascending=True)
        # --- 将动能按短、中、长期分组，便于逻辑构建 ---
        # 上升动能分组
        price_momentum_short_bull = (price_momentum_scores[1] + price_momentum_scores[5]) / 2
        price_momentum_mid_bull = (price_momentum_scores[13] + price_momentum_scores[21]) / 2
        price_momentum_long_bull = price_momentum_scores[55]
        # 筹码集中动能分组
        chip_conc_short = (chip_conc_scores[1] + chip_conc_scores[5]) / 2
        chip_conc_mid_long = (chip_conc_scores[13] + chip_conc_scores[21] + chip_conc_scores[55]) / 3
        # 短期价格加速（同周期验证）
        price_accel_short_bull = (price_accel_scores[1] + price_accel_scores[5]) / 2
        # --- 3. 共振信号合成 (多时间周期深度交叉验证) ---
        # 3.1 上升共振 (Bullish Resonance)
        # --- 重构B/A/S三级信号逻辑 ---
        # B级: 短期趋势形成 (基础信号)
        bullish_resonance_b = price_momentum_short_bull
        # A级: B级信号得到中期趋势 和 短期筹码集中 的双重确认
        bullish_resonance_a = bullish_resonance_b * price_momentum_mid_bull * chip_conc_short
        # S级: A级信号得到 长期趋势、中长期筹码、短期加速、波动率和成交量 的五重终极确认
        bullish_resonance_s = bullish_resonance_a * price_momentum_long_bull * chip_conc_mid_long * price_accel_short_bull * vol_ignition_score * volume_accel_score
        states['SCORE_DYN_BULLISH_RESONANCE_B'] = pd.Series(bullish_resonance_b, index=df.index)
        states['SCORE_DYN_BULLISH_RESONANCE_A'] = pd.Series(bullish_resonance_a, index=df.index)
        states['SCORE_DYN_BULLISH_RESONANCE_S'] = pd.Series(bullish_resonance_s, index=df.index)
        # 3.2 下跌共振 (Bearish Resonance) - 对称逻辑
        # --- 为下跌共振构建对称的动能分组 ---
        price_momentum_short_bear = 1.0 - price_momentum_short_bull
        price_momentum_mid_bear = 1.0 - price_momentum_mid_bull
        price_momentum_long_bear = 1.0 - price_momentum_long_bull
        chip_dispersion_short = 1.0 - chip_conc_short
        chip_dispersion_mid_long = 1.0 - chip_conc_mid_long
        price_accel_short_bear = 1.0 - price_accel_short_bull
        # --- 重构下跌共振的B/A/S三级信号逻辑 ---
        bearish_resonance_b = price_momentum_short_bear
        bearish_resonance_a = bearish_resonance_b * price_momentum_mid_bear * chip_dispersion_short
        bearish_resonance_s = bearish_resonance_a * price_momentum_long_bear * chip_dispersion_mid_long * price_accel_short_bear * vol_expansion_score * volume_accel_score
        states['SCORE_DYN_BEARISH_RESONANCE_B'] = pd.Series(bearish_resonance_b, index=df.index)
        states['SCORE_DYN_BEARISH_RESONANCE_A'] = pd.Series(bearish_resonance_a, index=df.index)
        states['SCORE_DYN_BEARISH_RESONANCE_S'] = pd.Series(bearish_resonance_s, index=df.index)
        # --- 4. 反转信号合成 (环境 x 拐点 x 确认) ---
        # 4.1 底部反转 (Bottom Reversal)
        # --- 重构底部反转的环境、触发、确认逻辑 ---
        # 环境(Setup): 中长期趋势下跌 且 波动率压缩
        setup_bottom_score = price_momentum_mid_bear * price_momentum_long_bear * vol_squeeze_score
        # 触发(Trigger): 短期价格加速向上
        trigger_bottom_score = price_accel_short_bull
        # 确认(Confirm): 短期筹码开始集中
        confirm_bottom_score = chip_conc_short
        # B级: 出现短期加速的“拐点”迹象
        states['SCORE_DYN_BOTTOM_REVERSAL_B'] = pd.Series(trigger_bottom_score, index=df.index)
        # A级: “拐点”发生在有利的“环境”中
        states['SCORE_DYN_BOTTOM_REVERSAL_A'] = pd.Series(setup_bottom_score * trigger_bottom_score, index=df.index)
        # S级: “环境”和“拐点”俱备，同时得到筹码行为的“确认”
        states['SCORE_DYN_BOTTOM_REVERSAL_S'] = pd.Series(states['SCORE_DYN_BOTTOM_REVERSAL_A'].values * confirm_bottom_score, index=df.index)
        # 4.2 顶部反转 (Top Reversal) - 对称逻辑
        # --- 重构顶部反转的环境、触发、确认逻辑 ---
        # 环境(Setup): 中长期趋势上涨 且 波动率放大
        setup_top_score = price_momentum_mid_bull * price_momentum_long_bull * vol_expansion_score
        # 触发(Trigger): 短期价格加速向下（减速）
        trigger_top_score = price_accel_short_bear
        # 确认(Confirm): 短期筹码开始发散
        confirm_top_score = chip_dispersion_short
        states['SCORE_DYN_TOP_REVERSAL_B'] = pd.Series(trigger_top_score, index=df.index)
        states['SCORE_DYN_TOP_REVERSAL_A'] = pd.Series(setup_top_score * trigger_top_score, index=df.index)
        states['SCORE_DYN_TOP_REVERSAL_S'] = pd.Series(states['SCORE_DYN_TOP_REVERSAL_A'].values * confirm_top_score, index=df.index)

        print("        -> [核心力学诊断引擎 V4.0 深度交叉验证版] 分析完毕。")
        return states

    def diagnose_multi_timeframe_dynamics_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.0 深度交叉验证版】趋势结构诊断引擎 (原diagnose_multi_timeframe_dynamics_scores)
        - 核心重构 (本次修改):
          1. 逻辑重塑: 遵循“共振/反转”和“B/A/S三级置信度”设计范式，对原方法进行彻底重构。
          2. 维度扩展: 遵循原则1，将分析维度从(5, 21日)扩展至(1, 5, 13, 21, 55日)，覆盖短、中、长周期。
          3. 信号升级: 将原有的多个零散信号，重构成逻辑更严谨的“趋势共振”、“底部反转”、“顶部背离”三大类信号。
          4. 数据需求说明: 明确列出为实现理想逻辑而假设存在的、但当前数据层可能缺失的列清单。
        - 收益: 信号逻辑更符合A股“趋势-结构-确认”的实战特点，置信度分级更可靠，鲁棒性显著增强。
        - 数据需求说明 (原则3): 为实现理想逻辑，本方法假设以下数据列存在。若缺失，建议数据层补充：
          - SLOPE_{p}_close_D, ACCEL_{p}_close_D (对于 p in [1, 13, 55])
          - SLOPE_{p}_concentration_90pct_D (对于 p in [1, 13])
          - SLOPE_{p}_winner_profit_margin_D (对于 p in [1, 13, 55])
        """

        print("        -> [趋势结构诊断引擎 V3.0 深度交叉验证版] 启动...")
        states = {}
        # --- 1. 军备检查 (Arsenal Check) ---
        norm_window = 120
        min_periods = max(1, norm_window // 5)
        periods = [1, 5, 13, 21, 55]
        required_cols = set()
        for p in periods:
            required_cols.add(f'SLOPE_{p}_close_D')
            required_cols.add(f'ACCEL_{p}_close_D')
            required_cols.add(f'SLOPE_{p}_concentration_90pct_D')
            required_cols.add(f'SLOPE_{p}_winner_profit_margin_D')
        missing = list(required_cols - set(df.columns))
        if missing:
            print(f"          -> [警告] 趋势结构引擎 V3.0 缺少关键数据: {sorted(missing)}，模块已跳过！")
            return states
        # --- 2. 核心要素数值化与分组 ---
        price_mom = {p: self._normalize_series(df[f'SLOPE_{p}_close_D'], norm_window, min_periods) for p in periods}
        price_acc = {p: self._normalize_series(df[f'ACCEL_{p}_close_D'], norm_window, min_periods) for p in periods}
        chip_conc = {p: self._normalize_series(df[f'SLOPE_{p}_concentration_90pct_D'], norm_window, min_periods, ascending=False) for p in periods}
        profit_margin = {p: self._normalize_series(df[f'SLOPE_{p}_winner_profit_margin_D'], norm_window, min_periods) for p in periods}
        # 动能分组
        price_mom_short = (price_mom[1] + price_mom[5]) / 2
        price_mom_mid = (price_mom[13] + price_mom[21]) / 2
        price_mom_long = price_mom[55]
        chip_conc_short = (chip_conc[1] + chip_conc[5]) / 2
        chip_conc_mid_long = (chip_conc[13] + chip_conc[21] + chip_conc[55]) / 3
        profit_margin_erosion_short = 1.0 - ((profit_margin[1] + profit_margin[5]) / 2)
        price_accel_short = (price_acc[1] + price_acc[5]) / 2
        # --- 3. 上升趋势共振 (Trend Resonance) ---
        # B级: 短中期趋势共振
        resonance_b = price_mom_short * price_mom_mid
        # A级: B级信号得到短期筹码集中确认
        resonance_a = resonance_b * chip_conc_short
        # S级: A级信号得到长期趋势和中长期筹码的终极确认
        resonance_s = resonance_a * price_mom_long * chip_conc_mid_long
        states['SCORE_MTF_TREND_RESONANCE_B'] = pd.Series(resonance_b, index=df.index)
        states['SCORE_MTF_TREND_RESONANCE_A'] = pd.Series(resonance_a, index=df.index)
        # --- 原 'SCORE_MTF_TREND_RESONANCE_A' 升级为S级 ---
        states['SCORE_MTF_TREND_RESONANCE_S'] = pd.Series(resonance_s, index=df.index)
        # --- 4. 底部反转机会 (Bottom Inflection Opportunity) ---
        # 环境: 中长期处于下跌趋势
        setup_bottom = (1.0 - price_mom_mid) * (1.0 - price_mom_long)
        # 触发: 短期价格开始加速向上
        trigger_bottom = price_accel_short
        # 确认: 短期筹码开始集中
        confirm_bottom = chip_conc_short
        states['SCORE_MTF_BOTTOM_INFLECTION_B'] = pd.Series(trigger_bottom, index=df.index)
        states['SCORE_MTF_BOTTOM_INFLECTION_A'] = pd.Series(setup_bottom * trigger_bottom, index=df.index)
        # --- 原 'SCORE_MTF_BOTTOM_INFLECTION_OPP_A' 升级为S级 ---
        states['SCORE_MTF_BOTTOM_INFLECTION_S'] = pd.Series(states['SCORE_MTF_BOTTOM_INFLECTION_A'].values * confirm_bottom, index=df.index)
        # --- 5. 顶部背离风险 (Top Divergence Risk) ---
        # 环境: 中长期处于上涨趋势
        setup_top = price_mom_mid * price_mom_long
        # 触发(背离): 短期趋势向上，但筹码或盈利能力出现恶化
        trigger_top = price_mom_short * (chip_conc_short < 0.5) * profit_margin_erosion_short
        # 确认: 短期价格开始减速
        confirm_top = 1.0 - price_accel_short
        states['SCORE_MTF_TOP_DIVERGENCE_B'] = pd.Series(trigger_top, index=df.index)
        states['SCORE_MTF_TOP_DIVERGENCE_A'] = pd.Series(setup_top * trigger_top, index=df.index)
        # --- 原 'SCORE_MTF_STRUCTURAL_WEAKNESS_RISK_S' 等被此新信号取代 ---
        states['SCORE_MTF_TOP_DIVERGENCE_S'] = pd.Series(states['SCORE_MTF_TOP_DIVERGENCE_A'].values * confirm_top, index=df.index)
        # --- 6. 静态-动态融合 (逻辑保留，使用新的S级共振信号) ---
        key = 'SCORE_PLATFORM_OVERALL_QUALITY'
        static_platform_quality_arr = self.strategy.atomic_states.get(key, pd.Series(0.0, index=df.index)).values.astype(np.float32)
        platform_ignition_arr = static_platform_quality_arr * resonance_s # 使用S级共振信号
        states['SCORE_MTF_PLATFORM_IGNITION_S_PLUS'] = pd.Series(platform_ignition_arr, index=df.index)
        print("        -> [趋势结构诊断引擎 V3.0 深度交叉验证版] 分析完毕。")
        return states

    def diagnose_behavioral_mechanics_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.0 深度交叉验证版】行为力学评分引擎
        - 核心重构 (本次修改):
          1. 逻辑重塑: 遵循“B/A/S三级置信度”设计范式，对原方法进行彻底重构。
          2. 维度扩展: 遵循原则1，将分析维度扩展至(1, 5, 13, 21, 55日)，覆盖短、中、长周期。
          3. 信号升级: 将原有信号重构为“动能点火”、“投降枯竭”、“动能失速”、“恐慌抛售”四大类，逻辑更严谨。
          4. 引入静态验证: 遵循原则2，在高置信度信号中，引入静态指标（如换手率绝对值）进行交叉验证。
          5. 数据需求说明: 明确列出为实现理想逻辑而假设存在的、但当前数据层可能缺失的列清单。
        - 收益: 信号能更精确地刻画市场情绪的细微变化，如从犹豫到确认，从分歧到一致的过程，实战价值更高。
        - 数据需求说明 (原则3): 为实现理想逻辑，本方法假设以下数据列存在。若缺失，建议数据层补充：
          - ACCEL_{p}_VPA_EFFICIENCY_D (对于 p in [1, 13, 55])
          - ACCEL_{p}_turnover_from_winners_ratio_D (对于 p in [13])
          - ACCEL_{p}_turnover_from_losers_ratio_D (对于 p in [13, 55])
        """

        print("        -> [行为力学评分引擎 V3.0 深度交叉验证版] 启动...")
        states = {}
        # --- 1. 军备检查 (Arsenal Check) ---
        norm_window = 120
        min_periods = max(1, norm_window // 5)
        accel_periods = [1, 5, 13, 21, 55]
        required_cols = {'SLOPE_5_close_D', 'ACCEL_5_close_D', 'turnover_from_winners_ratio_D', 'turnover_from_losers_ratio_D'}
        for p in accel_periods:
            required_cols.add(f'ACCEL_{p}_VPA_EFFICIENCY_D')
            required_cols.add(f'ACCEL_{p}_turnover_from_winners_ratio_D')
            required_cols.add(f'ACCEL_{p}_turnover_from_losers_ratio_D')
        missing = list(required_cols - set(df.columns))
        if missing:
            print(f"          -> [警告] 行为力学引擎 V3.0 缺少关键数据: {sorted(missing)}，模块已跳过！")
            return states
        # --- 2. 核心要素数值化与分组 ---
        price_rising = self._normalize_series(df['SLOPE_5_close_D'], norm_window, min_periods)
        price_falling = 1.0 - price_rising
        price_accel = self._normalize_series(df['ACCEL_5_close_D'], norm_window, min_periods)
        # 行为加速度
        vpa_acc = {p: self._normalize_series(df[f'ACCEL_{p}_VPA_EFFICIENCY_D'], norm_window, min_periods) for p in accel_periods}
        winner_acc = {p: self._normalize_series(df[f'ACCEL_{p}_turnover_from_winners_ratio_D'], norm_window, min_periods) for p in accel_periods}
        loser_decel = {p: self._normalize_series(df[f'ACCEL_{p}_turnover_from_losers_ratio_D'], norm_window, min_periods, ascending=False) for p in accel_periods}
        # 行为静态值 (用于同周期交叉验证)
        winner_turnover_high = self._normalize_series(df['turnover_from_winners_ratio_D'], norm_window, min_periods)
        # 动能分组
        vpa_acc_short_mid = (vpa_acc[1] + vpa_acc[5] + vpa_acc[13] + vpa_acc[21]) / 4
        winner_acc_short_mid = (winner_acc[1] + winner_acc[5] + winner_acc[13] + winner_acc[21]) / 4
        loser_decel_short_mid = (loser_decel[1] + loser_decel[5] + loser_decel[13] + loser_decel[21]) / 4
        # --- 3. 信号合成 (B/A/S 置信度体系) ---
        # 3.1 动能点火 (Engine Ignition - Bullish)
        ignition_b = price_accel
        ignition_a = ignition_b * vpa_acc_short_mid
        ignition_s = ignition_a * (1.0 - winner_turnover_high) # S级确认: 获利盘压力不大
        states['SCORE_BEHAVIOR_ENGINE_IGNITION_B'] = pd.Series(ignition_b, index=df.index)
        # --- 原 'SCORE_BEHAVIOR_ENGINE_IGNITION_A' 升级为S级 ---
        states['SCORE_BEHAVIOR_ENGINE_IGNITION_A'] = pd.Series(ignition_a, index=df.index)
        states['SCORE_BEHAVIOR_ENGINE_IGNITION_S'] = pd.Series(ignition_s, index=df.index)
        # 3.2 投降枯竭 (Capitulation Exhaustion - Bullish Reversal)
        exhaustion_b = loser_decel_short_mid
        exhaustion_a = exhaustion_b * price_falling
        exhaustion_s = exhaustion_a * (1.0 - price_accel < 0.5) # S级确认: 价格下跌开始减速
        states['SCORE_BEHAVIOR_CAPITULATION_EXHAUSTION_B'] = pd.Series(exhaustion_b, index=df.index)
        # --- 原 'SCORE_BEHAVIOR_CAPITULATION_EXHAUSTION_OPP_A' 升级为S级 ---
        states['SCORE_BEHAVIOR_CAPITULATION_EXHAUSTION_A'] = pd.Series(exhaustion_a, index=df.index)
        states['SCORE_BEHAVIOR_CAPITULATION_EXHAUSTION_S'] = pd.Series(exhaustion_s, index=df.index)
        # 3.3 动能失速 (Engine Stalling - Bearish Risk)
        stalling_b = price_rising * (1.0 - vpa_acc_short_mid) # 上涨但效率降低
        stalling_a = stalling_b * winner_acc_short_mid # 同时获利盘加速涌出
        stalling_s = stalling_a * winner_turnover_high # S级确认: 获利盘换手率本身就很高
        states['SCORE_BEHAVIOR_ENGINE_STALLING_B'] = pd.Series(stalling_b, index=df.index)
        # --- 原 'SCORE_BEHAVIOR_ENGINE_STALLING_RISK_S' 升级为S级 ---
        states['SCORE_BEHAVIOR_ENGINE_STALLING_A'] = pd.Series(stalling_a, index=df.index)
        states['SCORE_BEHAVIOR_ENGINE_STALLING_S'] = pd.Series(stalling_s, index=df.index)
        # 3.4 恐慌抛售 (Panic Selling - Bearish Risk)
        panic_b = winner_acc_short_mid
        panic_a = panic_b * price_falling
        panic_s = panic_a * winner_turnover_high # S级确认: 获利盘换手率本身就很高
        states['SCORE_BEHAVIOR_PANIC_SELLING_B'] = pd.Series(panic_b, index=df.index)
        # --- 原 'SCORE_BEHAVIOR_PANIC_SELLING_RISK_S' 升级为S级 ---
        states['SCORE_BEHAVIOR_PANIC_SELLING_A'] = pd.Series(panic_a, index=df.index)
        states['SCORE_BEHAVIOR_PANIC_SELLING_S'] = pd.Series(panic_s, index=df.index)
        # --- 4. 静态-动态融合 (逻辑保留) ---
        key = 'PRICE_STATE_ZONE_SCORE'
        static_price_zone_arr = self.strategy.atomic_states.get(key, pd.Series(0.0, index=df.index)).values.astype(np.float32)
        # S+信号: 处于高价格区间的动能失速是极高风险的顶部背离信号
        top_divergence_risk_arr = static_price_zone_arr * stalling_s
        states['SCORE_BEHAVIOR_TOP_DIVERGENCE_RISK_S_PLUS'] = pd.Series(top_divergence_risk_arr, index=df.index)
        print("        -> [行为力学评分引擎 V3.0 深度交叉验证版] 分析完毕。")
        return states












