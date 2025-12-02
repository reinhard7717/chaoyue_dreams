# 新增文件: services/feature_engineering_service.py

import asyncio
import logging
from typing import Dict
import numpy as np
import pandas as pd
import pandas_ta as ta
from utils.math_tools import hurst_exponent

logger = logging.getLogger("services")

class FeatureEngineeringService:
    """
    特征工程服务
    - 核心职责: 专注于从基础数据（OHLCV和简单指标）中衍生出更高级的技术特征。
                它负责所有与K线本身形态、趋势、波动率相关的深度计算。
    """
    def __init__(self):
        pass

    async def calculate_all_slopes(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V3.1 注释优化版】计算所有配置的斜率特征。
        - 核心逻辑: 根据配置文件中的'series_to_slope'部分，为指定的数据列（如'MACD_12_26_9_D'）在不同的时间窗口（lookbacks）上计算线性回归斜率。
        - 优化: 保持原有的高效向量化计算，增加详尽注释。
        """
        # 从配置中获取斜率计算参数
        slope_params = config.get('feature_engineering_params', {}).get('slope_params', {})
        # 如果未启用，则直接返回
        if not slope_params.get('enabled', False):
            return all_dfs
        # 获取需要计算斜率的列名及其对应的周期列表
        series_to_slope = slope_params.get('series_to_slope', {})
        if not series_to_slope:
            return all_dfs
        # 遍历配置中的每一项
        for col_pattern, lookbacks in series_to_slope.items():
            # 跳过说明性配置
            if "说明" in col_pattern: continue
            try:
                # 约定：通过列名后缀（如_D, _W, _M）来判断其所属的时间周期DataFrame
                timeframe = col_pattern.split('_')[-1]
                if timeframe.upper() not in ['D', 'W', 'M'] and not timeframe.isdigit():
                    timeframe = 'D' # 默认使用日线
            except IndexError:
                continue # 如果列名不符合规范，则跳过
            # 检查对应时间周期的DataFrame是否存在
            if timeframe not in all_dfs or all_dfs[timeframe] is None:
                continue
            df = all_dfs[timeframe]
            # 检查源数据列是否存在
            if col_pattern not in df.columns:
                continue
            source_series = df[col_pattern].astype(float)
            # 遍历需要计算的周期长度
            for lookback in lookbacks:
                slope_col_name = f'SLOPE_{lookback}_{col_pattern}'
                # 如果目标斜率列已存在，则跳过，避免重复计算
                if slope_col_name in df.columns:
                    continue
                # 设置计算所需的最小周期数，增加计算结果的稳定性
                min_p = max(2, lookback // 2)
                # 使用pandas_ta库的linreg函数进行高效的向量化计算
                linreg_result = df.ta.linreg(close=source_series, length=lookback, min_periods=min_p, slope=True, intercept=False, r=False)
                # 兼容pandas_ta不同版本可能返回Series或DataFrame的情况
                slope_series = linreg_result if isinstance(linreg_result, pd.Series) else linreg_result.iloc[:, 0]
                # 将计算结果存入DataFrame，空值填充为0
                df[slope_col_name] = slope_series.fillna(0)
            all_dfs[timeframe] = df
        return all_dfs

    async def calculate_all_accelerations(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V2.1 注释优化版】计算所有配置的加速度特征。
        - 核心逻辑: 加速度是斜率的斜率。此方法基于已计算好的斜率特征（SLOPE_*），再次计算其斜率，从而得到加速度特征（ACCEL_*）。
        - 优化: 保持原有的高效向量化计算，增加详尽注释。
        """
        # 从配置中获取加速度计算参数
        accel_params = config.get('feature_engineering_params', {}).get('accel_params', {})
        # 如果未启用，则直接返回
        if not accel_params.get('enabled', False):
            return all_dfs
        # 获取需要计算加速度的基础列名及其对应的周期列表
        series_to_accel = accel_params.get('series_to_accel', {})
        if not series_to_accel:
            return all_dfs
        # 遍历配置
        for base_col_name, periods in series_to_accel.items():
            if "说明" in base_col_name: continue
            # 约定：通过列名后缀判断时间周期
            timeframe = base_col_name.split('_')[-1]
            if timeframe not in all_dfs or all_dfs[timeframe] is None:
                continue
            df = all_dfs[timeframe]
            # 遍历周期
            for period in periods:
                # 定位作为计算源的斜率列
                slope_col_name = f'SLOPE_{period}_{base_col_name}'
                if slope_col_name not in df.columns:
                    # 如果依赖的斜率数据不存在，则无法计算加速度，跳过
                    continue
                accel_col_name = f'ACCEL_{period}_{base_col_name}'
                # 如果目标加速度列已存在，则跳过
                if accel_col_name in df.columns:
                    continue
                source_series = df[slope_col_name]
                min_p = max(2, period // 2)
                # 对斜率序列再次求斜率，得到加速度
                accel_linreg_result = df.ta.linreg(close=source_series, length=period, min_periods=min_p, slope=True, intercept=False, r=False)
                accel_series = accel_linreg_result if isinstance(accel_linreg_result, pd.Series) else accel_linreg_result.iloc[:, 0]
                df[accel_col_name] = accel_series.fillna(0)
        return all_dfs

    async def calculate_vpa_features(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V1.1 注释优化版】VPA效率指标生产线
        - 核心职责: 计算自定义指标 VPA_EFFICIENCY_D (价量攻击效率)。
        - 指标含义: 该指标衡量单位“超额”成交量所带来的价格涨幅。比值越高，说明资金攻击效率越高，少量放量就能带来较大涨幅；
                    反之，则说明存在抛压，或资金拉升意愿不强。
        - 优化: 代码已为最优向量化实现，本次仅增加详尽注释。
        """
        timeframe = 'D' # 此特征仅在日线级别计算
        if timeframe not in all_dfs:
            return all_dfs
        df = all_dfs[timeframe]
        # 检查计算所需的列是否存在
        required_cols = ['pct_change_D', 'volume_D', 'VOL_MA_21_D']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logger.warning(f"VPA效率生产线缺少关键数据: {missing}，模块已跳过！")
            return all_dfs
        # 1. 计算成交量比率：当日成交量 / 21日均量。衡量成交量的相对放大程度。
        volume_ratio = df['volume_D'] / df['VOL_MA_21_D'].replace(0, np.nan)
        # 2. 计算攻击效率：当日涨跌幅 / 成交量比率。
        vpa_efficiency = df['pct_change_D'] / volume_ratio.replace(0, np.nan)
        # 3. 将结果存入DataFrame，处理无穷大值并填充空值
        df['VPA_EFFICIENCY_D'] = vpa_efficiency.replace([np.inf, -np.inf], np.nan).fillna(0)
        all_dfs[timeframe] = df
        return all_dfs

    async def calculate_meta_features(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V3.1 · 熵指标增强版】元特征计算车间
        - 核心升级: 废弃旧的简单指标，引入分形维度、样本熵、波动率不稳定性等一系列能够深度刻画市场混沌、分形与信息熵的“元特征”。
        - 新增优化: 直接集成预计算的、更高级的 `price_volume_entropy_D` 指标，作为对市场信息复杂度的核心度量。
        """
        timeframe = 'D'
        if timeframe not in all_dfs:
            return all_dfs
        df = all_dfs[timeframe]
        suffix = f"_{timeframe}"
        params = config.get('feature_engineering_params', {}).get('meta_feature_params', {})
        if not params.get('enabled', False):
            return all_dfs
        close_col = f'close{suffix}'
        if close_col not in df.columns:
            logger.warning(f"元特征计算缺少核心列 '{close_col}'，模块已跳过。")
            return all_dfs
        source_series = df[close_col]
        if isinstance(source_series, pd.DataFrame):
            source_series = source_series.iloc[:, 0]
        # --- 1. Hurst 指数 (市场记忆性) ---
        hurst_window = params.get('hurst_window', 120)
        hurst_col = f'hurst_{hurst_window}d{suffix}'
        if hurst_col not in df.columns and len(source_series.dropna()) >= hurst_window:
            try:
                df[hurst_col] = source_series.rolling(window=hurst_window, min_periods=hurst_window).apply(hurst_exponent, raw=True)
            except Exception as e:
                logger.error(f"赫斯特指数(周期{hurst_window})计算失败: {e}")
                df[hurst_col] = np.nan
        # --- 2. 分形维度 (市场复杂度) ---
        def _higuchi_fractal_dimension(x, k_max):
            L = []
            x_len = len(x)
            for k in range(1, k_max + 1):
                Lk = 0
                for m in range(k):
                    # 创建子序列
                    series = x[m::k]
                    if len(series) < 2: continue
                    # 计算子序列长度
                    Lk += np.sum(np.abs(np.diff(series))) * (x_len - 1) / ((x_len - m) // k * k)
                L.append(np.log(Lk / k) if Lk > 0 else 0)
            # 对log-log图进行线性回归
            k_range_log = np.log(np.arange(1, k_max + 1))
            # 过滤掉无效的L值
            valid_indices = [i for i, val in enumerate(L) if val != 0]
            if len(valid_indices) < 2: return np.nan
            slope, _ = np.polyfit(k_range_log[valid_indices], np.array(L)[valid_indices], 1)
            return slope
        fd_window = params.get('fractal_dimension_window', 100)
        fd_col = f'FRACTAL_DIMENSION_{fd_window}d{suffix}'
        if fd_col not in df.columns and len(source_series.dropna()) >= fd_window:
            try:
                k_max = int(np.sqrt(fd_window))
                df[fd_col] = source_series.rolling(window=fd_window).apply(lambda x: _higuchi_fractal_dimension(x, k_max), raw=True)
            except Exception as e:
                logger.error(f"分形维度(周期{fd_window})计算失败: {e}")
                df[fd_col] = np.nan
        # --- 3. 样本熵 (市场可预测性) ---
        def _sample_entropy(x, m, r):
            n = len(x)
            # 构造模板
            templates = np.array([x[i:i+m] for i in range(n - m + 1)])
            # 计算模板间距离
            dist = np.max(np.abs(templates[:, np.newaxis] - templates), axis=2)
            # 统计匹配数
            A = np.sum(dist < r) - n # 减去自身匹配
            templates_plus_1 = np.array([x[i:i+m+1] for i in range(n - m)])
            dist_plus_1 = np.max(np.abs(templates_plus_1[:, np.newaxis] - templates_plus_1), axis=2)
            B = np.sum(dist_plus_1 < r) - (n - m)
            if A == 0 or B == 0: return np.nan
            return -np.log(B / A)
        se_window = params.get('sample_entropy_window', 10)
        se_tol_ratio = params.get('sample_entropy_tolerance_ratio', 0.2)
        se_col = f'SAMPLE_ENTROPY_{se_window}d{suffix}'
        if se_col not in df.columns and len(source_series.dropna()) >= se_window + 1:
            try:
                log_returns = np.log(source_series / source_series.shift(1)).dropna()
                rolling_std = log_returns.rolling(window=se_window).std()
                # 使用 apply 结合 lambda 来传递动态的 r
                entropy_values = []
                for i in range(len(log_returns) - se_window + 1):
                    window_data = log_returns.iloc[i:i+se_window].values
                    r = rolling_std.iloc[i+se_window-1] * se_tol_ratio
                    if pd.isna(r) or r == 0:
                        entropy_values.append(np.nan)
                        continue
                    entropy_values.append(_sample_entropy(window_data, m=2, r=r))
                # 对齐结果到原始DataFrame
                df[se_col] = pd.Series(entropy_values, index=log_returns.index[se_window-1:]).reindex(df.index)
            except Exception as e:
                logger.error(f"样本熵(周期{se_window})计算失败: {e}")
                df[se_col] = np.nan
        # --- 4. 波动率不稳定性 (状态切换前兆) ---
        vi_window = params.get('volatility_instability_window', 21)
        vi_col = f'VOLATILITY_INSTABILITY_INDEX_{vi_window}d{suffix}'
        atr_col = f'ATR_14{suffix}' # 依赖于ATR
        if atr_col in df.columns and vi_col not in df.columns:
            df[vi_col] = df[atr_col].rolling(window=vi_window).std()
        # --- 5. 新增：集成价格成交量熵 (市场信息复杂度) --- # 新增代码块
        pve_col_source = f'price_volume_entropy{suffix}' # 新增代码行
        pve_col_target = f'PRICE_VOLUME_ENTROPY{suffix}' # 新增代码行
        if pve_col_source in df.columns and pve_col_target not in df.columns: # 新增代码行
            df[pve_col_target] = df[pve_col_source] # 新增代码行
            logger.info("已成功集成预计算的'价格成交量熵'指标。") # 新增代码行
        all_dfs[timeframe] = df
        return all_dfs

    async def calculate_pattern_recognition_signals(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V3.6 · 战略意图升级版】高级模式识别信号生产线
        - 核心升级: 全面引入新一代的结构、博弈和风险指标，将市场状态的识别精度从战术层面提升至战略层面。
                      - 盘整识别: 引入“结构张力指数”，要求筹码结构内部应力低。
                      - 吸筹识别: 引入“浮筹清洗效率”，验证主力吸筹的质量。
                      - 突破识别: 引入“突破就绪分”，作为突破信号的强确认。
                      - 派发识别: 引入“衰竭风险指数”，预警趋势顶部的系统性风险。
        """
        timeframe = 'D'
        if timeframe not in all_dfs:
            return all_dfs
        df = all_dfs[timeframe]
        # 基于可用数据列，替换缺失的特征
        required_cols = [
            'high_D', 'low_D', 'close_D', 'volume_D', 'pct_change_D', 'VOL_MA_21_D', 'BBW_21_2.0_D', 'ATR_14_D', 'ADX_14_D',
            'open_D', 'amount_D',
            'chip_health_score_D',
            'hidden_accumulation_intensity_D',
            'rally_distribution_pressure_D',
            'winner_stability_index_D',
            'cost_structure_skewness_D',
            'dominant_peak_solidity_D',
            'main_force_net_flow_calibrated_D', 'main_force_execution_alpha_D', 'dip_absorption_power_D',
            'main_force_on_peak_flow_D', 'flow_efficiency_index_D', 'main_force_flow_directionality_D',
            'MA_POTENTIAL_TENSION_INDEX_D',
            'MA_POTENTIAL_ORDERLINESS_SCORE_D',
            'MA_POTENTIAL_COMPRESSION_RATE_D',
            'VPA_EFFICIENCY_D',
            'structural_tension_index_D',          # 新增代码行: 引入结构张力指数
            'floating_chip_cleansing_efficiency_D',# 新增代码行: 引入浮筹清洗效率
            'breakout_readiness_score_D',          # 新增代码行: 引入突破就绪分
            'exhaustion_risk_index_D'              # 新增代码行: 引入衰竭风险指数
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"高级模式识别引擎缺少关键数据: {missing_cols}，模块已跳过！")
            print("当前可用的列名包括:")
            print(df.columns.tolist())
            return all_dfs
        # --- 2. 【战场状态定义】: 基于“状态+势能”的多因子共振 ---
        cond_high_tension = df['MA_POTENTIAL_TENSION_INDEX_D'] < df['MA_POTENTIAL_TENSION_INDEX_D'].rolling(60).quantile(0.20)
        cond_low_orderliness = df['MA_POTENTIAL_ORDERLINESS_SCORE_D'].abs() < 0.5
        cond_struct_healthy = (df['chip_health_score_D'] > 50) & (df['dominant_peak_solidity_D'] > 0.5)
        cond_low_volatility = df['BBW_21_2.0_D'] < df['BBW_21_2.0_D'].rolling(60).quantile(0.25)
        cond_weak_trend = df['ADX_14_D'] < 25
        # 新增代码行: 引入结构张力条件，要求筹码结构内部稳定、无应力
        cond_low_structural_tension = df['structural_tension_index_D'] < df['structural_tension_index_D'].rolling(60).quantile(0.30)
        # 修改代码行: 在盘整条件中加入对低结构张力的要求
        df['IS_HIGH_POTENTIAL_CONSOLIDATION_D'] = cond_high_tension & cond_low_orderliness & cond_struct_healthy & cond_low_volatility & cond_weak_trend & cond_low_structural_tension
        cond_compressing = df['MA_POTENTIAL_COMPRESSION_RATE_D'] < 0
        # 新增代码行: 引入浮筹清洗效率作为吸筹的佐证
        cond_chip_cleansing = df['floating_chip_cleansing_efficiency_D'] > 0.5
        # 修改代码行: 将浮筹清洗效率加入主力吸筹的判断条件
        cond_main_force_accum = (df['hidden_accumulation_intensity_D'] > 0) | (df['dip_absorption_power_D'] > 0.5) | cond_chip_cleansing
        cond_peak_flow_positive = df['main_force_on_peak_flow_D'].rolling(3).mean() > 0
        cond_vpa_efficient_accum = df['VPA_EFFICIENCY_D'] > df['VPA_EFFICIENCY_D'].rolling(21).quantile(0.5)
        df['IS_ACCUMULATION_D'] = df['IS_HIGH_POTENTIAL_CONSOLIDATION_D'] & cond_compressing & (cond_main_force_accum | cond_peak_flow_positive) & cond_vpa_efficient_accum
        cond_was_consolidating = df['IS_HIGH_POTENTIAL_CONSOLIDATION_D'].shift(1).fillna(False)
        cond_orderliness_turn_up = (df['MA_POTENTIAL_ORDERLINESS_SCORE_D'] > 0.8) & (df['MA_POTENTIAL_ORDERLINESS_SCORE_D'].diff() > 0.3)
        cond_main_force_ignition = (df['main_force_net_flow_calibrated_D'] > 0) & \
                                   (df['main_force_execution_alpha_D'] > 0) & \
                                   (df['main_force_flow_directionality_D'] > 0.6)
        cond_price_volume_confirm = (df['pct_change_D'] > 0.01) & \
                                    (df['volume_D'] > df['VOL_MA_21_D'] * 1.2) & \
                                    (df['VPA_EFFICIENCY_D'] > df['VPA_EFFICIENCY_D'].rolling(21).quantile(0.9))
        # 新增代码行: 引入突破就绪分作为强确认条件
        cond_breakout_ready = df['breakout_readiness_score_D'] > 60 # 突破就绪分通常是0-100
        # 修改代码行: 在突破条件中加入对突破就绪分的验证
        df['IS_BREAKOUT_D'] = cond_was_consolidating & cond_orderliness_turn_up & cond_main_force_ignition & cond_price_volume_confirm & cond_breakout_ready
        cond_rally_dist = (df['pct_change_D'] > 0) & (df['rally_distribution_pressure_D'] > 0.5)
        cond_main_force_outflow = (df['main_force_net_flow_calibrated_D'].rolling(3).sum() < 0) & \
                                  (df['main_force_flow_directionality_D'] < -0.3)
        cond_winner_conviction_drop = df['winner_stability_index_D'].diff() < 0
        cond_resilience_drop = df['dominant_peak_solidity_D'].diff() < 0
        cond_vpa_inefficient_dist = (df['pct_change_D'] > 0) & (df['VPA_EFFICIENCY_D'] < df['VPA_EFFICIENCY_D'].rolling(21).quantile(0.2))
        # 新增代码行: 引入衰竭风险指数作为派发的强预警信号
        cond_exhaustion_risk = df['exhaustion_risk_index_D'] > 70 # 衰竭风险指数通常是0-100
        # 修改代码行: 将衰竭风险加入派发的核心判断逻辑
        df['IS_DISTRIBUTION_D'] = cond_rally_dist | (cond_main_force_outflow & cond_winner_conviction_drop & cond_resilience_drop & cond_vpa_inefficient_dist) | cond_exhaustion_risk
        # --- 3. 【通达信模式集成】 ---
        if 'amount_D' in df.columns:
            ema_amount_5 = df['amount_D'].ewm(span=5, adjust=False).mean()
            ff_ratio = ema_amount_5 / ema_amount_5.shift(1)
            llv_c_120 = df['close_D'].rolling(window=120, min_periods=1).min()
            hhv_c_120 = df['close_D'].rolling(window=120, min_periods=1).max()
            ww_position = (df['close_D'] - llv_c_120) / (hhv_c_120 - llv_c_120).replace(0, np.nan) * 100
            cond_ff_high = (ff_ratio >= 2)
            cond_ww_low = (ww_position < 35)
            cond_ww_any = (ww_position < 100)
            cond_bars_gt_30 = (df.index.to_series().diff().dt.days.cumsum().fillna(0) > 30)
            cond_bars_lt_50 = (df.index.to_series().diff().dt.days.cumsum().fillna(0) < 50)
            is_bazhan = (cond_ff_high & cond_ww_low & cond_bars_gt_30) | (cond_ff_high & cond_ww_any & cond_bars_lt_50)
            df['IS_BAZHAN_D'] = is_bazhan.fillna(False)
        else:
            df['IS_BAZHAN_D'] = False
            logger.warning("高级模式识别引擎缺少 'amount_D' 列，无法计算 '霸占' 模式。")
        if 'open_D' in df.columns and 'volume_D' in df.columns:
            cond_open_below_prev_low = (df['open_D'] < df['low_D'].shift(1))
            cond_close_above_open = (df['close_D'] > df['open_D'])
            cond_volume_increase = (df['volume_D'] > df['volume_D'].shift(1))
            is_ww1 = cond_open_below_prev_low & cond_close_above_open & cond_volume_increase
            df['IS_WW1_D'] = is_ww1.fillna(False)
        else:
            df['IS_WW1_D'] = False
            logger.warning("高级模式识别引擎缺少 'open_D' 或 'volume_D' 列，无法计算 'WW1' 模式。")
        # --- 4. 【信号整合与输出】 ---
        pattern_cols = ['IS_HIGH_POTENTIAL_CONSOLIDATION_D', 'IS_ACCUMULATION_D', 'IS_BREAKOUT_D', 'IS_DISTRIBUTION_D', 'IS_BAZHAN_D', 'IS_WW1_D']
        for col in pattern_cols:
            if col in df.columns:
                df[col] = df[col].fillna(False).astype(bool)
        all_dfs[timeframe] = df
        logger.info("高级模式识别引擎(V3.6 战略意图升级版)分析完成。")
        return all_dfs

    async def calculate_aaa_indicator(self, all_dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        【V1.0】计算通达信 AAA 指标。
        - 核心逻辑: AAA:=ABS((2*CLOSE+HIGH+LOW)/4-MA(CLOSE,N))/MA(CLOSE,N); N:=30;
        - 作为 DMA 的平滑因子。
        """
        timeframe = 'D'
        if timeframe not in all_dfs or all_dfs[timeframe].empty:
            logger.warning(f"计算 AAA 指标失败：缺少日线数据。")
            return all_dfs
        df = all_dfs[timeframe]
        required_cols = ['close_D', 'high_D', 'low_D']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"计算 AAA 指标缺少关键数据: {missing_cols}，模块已跳过！")
            return all_dfs
        n_period_for_aaa = 30
        if len(df) < n_period_for_aaa:
            logger.warning(f"计算 AAA 指标失败：数据行数 ({len(df)}) 不足以计算周期为 {n_period_for_aaa} 的 MA。")
            df['AAA_D'] = 0.0
            all_dfs[timeframe] = df
            return all_dfs
        ma_close_n = df['close_D'].rolling(window=n_period_for_aaa, min_periods=1).mean()
        weighted_price = (2 * df['close_D'] + df['high_D'] + df['low_D']) / 4
        aaa_series = (weighted_price - ma_close_n).abs() / ma_close_n.replace(0, np.nan)
        df['AAA_D'] = aaa_series.fillna(0)
        all_dfs[timeframe] = df
        logger.info("AAA 指标计算完成。")
        return all_dfs

    async def calculate_consolidation_period(self, all_dfs: Dict[str, pd.DataFrame], params: dict) -> Dict[str, pd.DataFrame]:
        """
        【V2.0 · 意图识别升级版】根据多因子共振识别盘整期。
        - 核心升级: 判定逻辑从单一的“几何形态”升级为“几何形态 或 主力意图”的双重标准。
                      即使K线形态不完美，只要筹码高度锁定且主力展现出强控盘意图，也将其识别为盘整构筑期。
        """
        if not params.get('enabled', False):
            return all_dfs
        timeframe = 'D'
        if timeframe not in all_dfs or all_dfs[timeframe].empty:
            return all_dfs
        df = all_dfs[timeframe]
        boll_period = params.get('boll_period', 21)
        boll_std = params.get('boll_std', 2.0)
        roc_period = params.get('roc_period', 12)
        vol_ma_period = params.get('vol_ma_period', 55)
        bbw_col = f"BBW_{boll_period}_{float(boll_std)}_{timeframe}"
        roc_col = f"ROC_{roc_period}_{timeframe}"
        vol_ma_col = f"VOL_MA_{vol_ma_period}_{timeframe}"
        # --- 修改代码开始 ---
        required_cols = [
            bbw_col, roc_col, vol_ma_col, f'high_{timeframe}', f'low_{timeframe}', f'volume_{timeframe}',
            'dominant_peak_solidity_D', 'control_solidity_index_D' # 新增：引入主力意图证据
        ]
        # --- 修改代码结束 ---
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logger.warning(f"盘整期计算跳过，依赖的列 '{', '.join(missing)}' 不存在。")
            return all_dfs
        bbw_quantile = params.get('bbw_quantile', 0.25)
        roc_threshold = params.get('roc_threshold', 5.0)
        min_expanding_periods = boll_period * 2
        dynamic_bbw_threshold = df[bbw_col].expanding(min_periods=min_expanding_periods).quantile(bbw_quantile).bfill()
        df[f'dynamic_bbw_threshold_{timeframe}'] = dynamic_bbw_threshold
        # --- 修改代码开始 ---
        # 条件1: 经典几何形态构筑
        cond_volatility = df[bbw_col] < df[f'dynamic_bbw_threshold_{timeframe}']
        cond_trend = df[roc_col].abs() < roc_threshold
        cond_volume = df[f'volume_{timeframe}'] < df[vol_ma_col]
        is_classic_consolidation = cond_volatility & cond_trend & cond_volume
        # 条件2: 主力意图构筑 (即使形态不完美)
        # 证据：筹码高度锁定 且 主力强力控盘
        cond_chips_locked = df['dominant_peak_solidity_D'] > 0.7
        cond_main_force_control = df['control_solidity_index_D'] > 0.6
        is_intent_based_consolidation = cond_chips_locked & cond_main_force_control
        # 最终判定：满足任一条件即可
        is_consolidating = is_classic_consolidation | is_intent_based_consolidation
        # --- 修改代码结束 ---
        df[f'is_consolidating_{timeframe}'] = is_consolidating
        if is_consolidating.any():
            consolidation_blocks = (is_consolidating != is_consolidating.shift()).cumsum()
            consolidating_df = df[is_consolidating].copy()
            grouped = consolidating_df.groupby(consolidation_blocks[is_consolidating])
            df[f'dynamic_consolidation_high_{timeframe}'] = grouped[f'high_{timeframe}'].transform('max')
            df[f'dynamic_consolidation_low_{timeframe}'] = grouped[f'low_{timeframe}'].transform('min')
            df[f'dynamic_consolidation_avg_vol_{timeframe}'] = grouped[f'volume_{timeframe}'].transform('mean')
            df[f'dynamic_consolidation_duration_{timeframe}'] = grouped[f'high_{timeframe}'].transform('size')
            fill_cols = [f'dynamic_consolidation_high_{timeframe}', f'dynamic_consolidation_low_{timeframe}', f'dynamic_consolidation_avg_vol_{timeframe}', f'dynamic_consolidation_duration_{timeframe}']
            df[fill_cols] = df[fill_cols].ffill()
        df[f'dynamic_consolidation_high_{timeframe}'] = df.get(f'dynamic_consolidation_high_{timeframe}', pd.Series(index=df.index)).fillna(df[f'high_{timeframe}'])
        df[f'dynamic_consolidation_low_{timeframe}'] = df.get(f'dynamic_consolidation_low_{timeframe}', pd.Series(index=df.index)).fillna(df[f'low_{timeframe}'])
        df[f'dynamic_consolidation_avg_vol_{timeframe}'] = df.get(f'dynamic_consolidation_avg_vol_{timeframe}', pd.Series(index=df.index)).fillna(0)
        df[f'dynamic_consolidation_duration_{timeframe}'] = df.get(f'dynamic_consolidation_duration_{timeframe}', pd.Series(index=df.index)).fillna(0)
        all_dfs[timeframe] = df
        return all_dfs

    async def calculate_pattern_enhancement_signals(self, all_dfs: Dict[str, pd.DataFrame], config: dict, calculator) -> Dict[str, pd.DataFrame]:
        """
        【V1.4 · 后门封堵与命名修复版】形态增强信号编排器
        - 核心修复: 调整了命名协议的强制执行逻辑，确保只对**不以 '_D' 结尾**的列添加 '_D' 后缀，避免重复。
        """
        params = config.get('feature_engineering_params', {}).get('indicators', {}).get('pattern_enhancement_signals', {})
        if not params.get('enabled', False):
            return all_dfs
        df_daily = all_dfs.get('D')
        if df_daily is None or df_daily.empty:
            return all_dfs
        minute_tf = params.get('minute_level_tf', '60')
        df_minute = all_dfs.get(minute_tf)
        tasks = []
        vwap_params = params.get('intraday_vwap_divergence', {})
        if vwap_params.get('enabled') and df_minute is not None:
            tasks.append(calculator.calculate_intraday_vwap_divergence_index(df_minute))
        exhaustion_params = params.get('counterparty_exhaustion', {})
        if exhaustion_params.get('enabled') and df_minute is not None:
            tasks.append(calculator.calculate_counterparty_exhaustion_index(df_minute, exhaustion_params.get('efficiency_window', 21)))
        if not tasks:
            return all_dfs
        results = await asyncio.gather(*tasks)
        new_cols_no_suffix = []
        for res_df in results:
            if res_df is not None and not res_df.empty:
                new_cols_no_suffix.extend(res_df.columns)
                res_df.index = pd.to_datetime(res_df.index, utc=True).normalize()
                df_daily = df_daily.join(res_df, how='left')
        # --- 命名协议强制执行 ---
        # 为所有新合并的、不以 '_D' 结尾的列，强制添加 '_D' 后缀
        rename_map = {col: f"{col}_D" for col in new_cols_no_suffix if col in df_daily.columns and not col.endswith('_D')}
        if rename_map:
            df_daily.rename(columns=rename_map, inplace=True)
        # 对新合并的列（现在已带后缀）进行前向填充
        final_new_cols = list(rename_map.values())
        if final_new_cols:
            df_daily[final_new_cols] = df_daily[final_new_cols].ffill()
        all_dfs['D'] = df_daily
        logger.info("分钟级形态增强信号计算完成并已集成。")
        return all_dfs

    async def calculate_ma_potential_metrics(self, all_dfs: Dict[str, pd.DataFrame], params: dict) -> Dict[str, pd.DataFrame]:
        """
        【V1.0】均线系统势能分析引擎
        - 核心职责: 根据 ma_potential_metrics 配置，计算均线系统的“张力”、“有序度”、“压缩率”三大核心势能指标。
        """
        if not params.get('enabled', False):
            return all_dfs
        for timeframe in params.get('apply_on', []):
            if timeframe not in all_dfs or all_dfs[timeframe] is None or all_dfs[timeframe].empty:
                continue
            df = all_dfs[timeframe]
            # 从配置中获取参数
            ma_periods = params.get('ma_periods', [])
            ma_type = params.get('ma_type', 'EMA')
            tension_short_period = params.get('tension_short_period', 5)
            tension_long_period = params.get('tension_long_period', 55)
            norm_window = params.get('norm_window', 55)
            # 1. 【军火库点验】: 确认计算所需的核心数据
            ma_cols = [f"{ma_type}_{p}_{timeframe}" for p in ma_periods]
            tension_cols = [f"{ma_type}_{tension_short_period}_{timeframe}", f"{ma_type}_{tension_long_period}_{timeframe}"]
            atr_col = f"ATR_14_{timeframe}" # 使用ATR进行波动率归一化
            required_cols = list(set(ma_cols + tension_cols + [atr_col]))
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"均线系统势能分析失败({timeframe})：缺少核心数据列 {missing_cols}")
                continue
            try:
                # 2. 【计算势能大小】: 均线张力指数 (MA_POTENTIAL_TENSION_INDEX)
                # 逻辑: (短期均线 - 长期均线) / ATR，衡量偏离程度相对于真实波动的比率，更具可比性
                tension_range = df[f"{ma_type}_{tension_short_period}_{timeframe}"] - df[f"{ma_type}_{tension_long_period}_{timeframe}"]
                atr = df[atr_col].replace(0, np.nan)
                raw_tension_index = tension_range / atr
                # 对原始张力进行双极性归一化，得到[-1, 1]的分数
                tension_index_series = raw_tension_index.rolling(window=norm_window).apply(lambda x: (x[-1] - x.mean()) / (x.std() + 1e-9) if len(x) > 1 else 0, raw=False).fillna(0).clip(-3, 3) / 3
                df[f'MA_POTENTIAL_TENSION_INDEX_{timeframe}'] = tension_index_series.astype(np.float32)
                # 3. 【计算势能方向】: 均线有序性评分 (MA_POTENTIAL_ORDERLINESS_SCORE)
                # 逻辑: 使用Spearman秩相关系数衡量均线周期顺序与均线值大小顺序的一致性
                ma_df = df[ma_cols]
                periods_series = pd.Series(ma_periods)
                rank_x = periods_series.rank(method='average') # 周期排名
                # 逐行计算Spearman相关性
                def spearman_corr(row):
                    rank_y = row.rank(method='average')
                    d_sq = ((rank_x - rank_y.values)**2).sum()
                    n = len(ma_periods)
                    if n <= 1: return 0.0
                    return 1 - (6 * d_sq) / (n * (n**2 - 1))
                orderliness_score = ma_df.apply(spearman_corr, axis=1)
                df[f'MA_POTENTIAL_ORDERLINESS_SCORE_{timeframe}'] = orderliness_score.fillna(0).astype(np.float32)
                # 4. 【计算势能变化率】: 均线压缩率 (MA_POTENTIAL_COMPRESSION_RATE)
                # 逻辑: 计算均线簇的标准差，并用ATR进行归一化，然后取其倒数作为压缩率
                ma_std = ma_df.std(axis=1)
                normalized_std = ma_std / atr
                # 归一化后取反，标准差越小（越压缩），得分越高
                compression_rate = 1 - (normalized_std.rolling(window=norm_window).rank(pct=True)).fillna(0.5)
                df[f'MA_POTENTIAL_COMPRESSION_RATE_{timeframe}'] = compression_rate.astype(np.float32)
            except Exception as e:
                logger.error(f"计算均线系统势能时发生错误({timeframe}): {e}", exc_info=True)
        return all_dfs

    async def calculate_breakout_quality(self, all_dfs: Dict, params: dict, calculator) -> Dict:
        """
        【V3.2 · 接口契约修复版】突破质量分计算专用通道
        - 核心修复: 协同 IndicatorCalculator V2.5 修复了接口契约。本方法现在能正确接收不带后缀的
                      'breakout_quality_score'，并执行重命名与填充，确保数据流的绝对标准化和健壮性。
        """
        if not params.get('enabled', False):
            return all_dfs
        timeframe = 'D'
        if timeframe not in all_dfs or all_dfs[timeframe] is None:
            return all_dfs
        df_daily = all_dfs[timeframe]
        required_materials = [
            'volume', 'VOL_MA_21', 'main_force_flow_directionality',
            'open', 'high', 'low', 'close',
            'total_winner_rate', 'dominant_peak_solidity', 'VPA_EFFICIENCY'
        ]
        df_standardized = pd.DataFrame(index=df_daily.index)
        missing_materials = []
        for material in required_materials:
            source_col_with_suffix = f"{material}_{timeframe}"
            if source_col_with_suffix in df_daily.columns:
                df_standardized[material] = df_daily[source_col_with_suffix]
            else:
                missing_materials.append(source_col_with_suffix)
        if missing_materials:
            logger.warning(f"突破质量分计算中止，缺少标准化原材料: {missing_materials}")
            return all_dfs
        result_df = await calculator.calculate_breakout_quality_score(df_daily=df_standardized, params=params)
        if result_df is not None and not result_df.empty:
            df_daily = df_daily.join(result_df, how='left')
            # --- 命名协议强制执行 ---
            # 将合并进来的、不带后缀的列，强制重命名为带 '_D' 后缀的标准格式
            if 'breakout_quality_score' in df_daily.columns:
                df_daily.rename(columns={'breakout_quality_score': 'breakout_quality_score_D'}, inplace=True)
                df_daily['breakout_quality_score_D'] = df_daily['breakout_quality_score_D'].ffill()
            all_dfs[timeframe] = df_daily
            logger.info("突破质量分计算完成并已集成。")
        else:
            logger.warning("突破质量分计算器返回了None或空DataFrame，未集成。")
        return all_dfs

    async def calculate_nmfnf(self, all_dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        【V1.0】计算标准化主力净流量 (Normalized Main Force Net Flow, NMFNF)。
        - 核心逻辑: NMFNF = main_force_net_flow_calibrated_D / total_market_value_D。
        - 目的: 将主力净流量标准化，使其在不同市值和活跃度的股票间可比。
        """
        timeframe = 'D'
        if timeframe not in all_dfs or all_dfs[timeframe].empty:
            logger.warning(f"计算 NMFNF 失败：缺少日线数据。")
            return all_dfs
        df = all_dfs[timeframe]
        required_cols = ['main_force_net_flow_calibrated_D', 'total_market_value_D']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"计算 NMFNF 缺少关键数据: {missing_cols}，模块已跳过！")
            return all_dfs
        # 避免除以零，将总市值中的零替换为 NaN，然后填充一个非常小的数或直接用0处理
        # 这里选择替换为 NaN，然后整个表达式会变为 NaN，最后fillna(0)
        nmfnf_series = df['main_force_net_flow_calibrated_D'] / df['total_market_value_D'].replace(0, np.nan)
        df['NMFNF_D'] = nmfnf_series.fillna(0)
        all_dfs[timeframe] = df
        logger.info("NMFNF 指标计算完成。")
        return all_dfs

    async def calculate_och(self, all_dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        【V3.1 · 结构动力学升级版】计算整体筹码健康度 (Overall Chip Health, OCH)。
        - 核心升级: 维度一“筹码集中度与结构优化”的计算逻辑重构，引入 cost_gini_coefficient 和 primary_peak_kurtosis 两个核心指标，
                    替代旧的 winner/loser_concentration 指标，实现对筹码结构更精准、更深刻的量化评估。
        - 目的: 在数据层提前计算一个综合的筹码健康度指标，供后续斜率计算和情报层使用。
        - 数据源: 直接从 df 中获取原始筹码相关列。
        """
        timeframe = 'D'
        if timeframe not in all_dfs or all_dfs[timeframe].empty:
            logger.warning(f"计算 OCH 失败：缺少日线数据。")
            return all_dfs
        df = all_dfs[timeframe]
        df_index = df.index
        # 定义所有需要的原始筹码相关列
        # 更新必需列的列表，以匹配V33.0版本的新指标体系
        required_cols = [
            # 维度一: 筹码集中度与结构优化 (已更新)
            'cost_gini_coefficient_D',        # 修改代码行: 新增，替代 winner/loser_concentration
            'primary_peak_kurtosis_D',        # 修改代码行: 新增，增强主峰质量评估
            'dominant_peak_solidity_D', 'dominant_peak_volume_ratio_D',
            'chip_fault_blockage_ratio_D',
            # 维度二: 成本与盈亏结构动态
            'total_winner_rate_D', 'total_loser_rate_D',
            'winner_profit_margin_avg_D', 'loser_loss_margin_avg_D',
            'cost_structure_skewness_D',
            'main_force_cost_advantage_D',
            'profit_taking_flow_ratio_D',
            'loser_pain_index_D',
            # 维度三: 持股心态与交易行为
            'winner_stability_index_D',
            'chip_fatigue_index_D',
            'capitulation_flow_ratio_D',
            'capitulation_absorption_index_D',
            'active_buying_support_D', 'active_selling_pressure_D',
            'mf_retail_battle_intensity_D',
            # 维度四: 主力控盘与意图
            'control_solidity_index_D',
            'main_force_on_peak_flow_D',
            'main_force_flow_directionality_D', 'main_force_execution_alpha_D',
            'main_force_conviction_index_D', 'mf_vpoc_premium_D', 'vwap_control_strength_D',
            'turnover_rate_f_D'
        ]
        # 使用 _get_safe_series 确保所有数据都存在，并用默认值填充缺失值
        def _get_safe_series_local(col_name, default_val=0.0):
            if col_name not in df.columns:
                print(f"调试信息: OCH计算缺少列: {col_name}，使用默认值 {default_val}。")
                return pd.Series(default_val, index=df_index)
            return df[col_name].fillna(default_val)
        # --- 1. 筹码集中度与结构优化 (Concentration & Structure Optimization Score) ---
        # 修改代码块：V3.1 升级: 使用基尼系数和峰态系数替代旧的集中度指标，评估更精准
        cost_gini = _get_safe_series_local('cost_gini_coefficient_D', 0.5) # 0-1, 越小越集中
        peak_kurtosis = _get_safe_series_local('primary_peak_kurtosis_D', 3.0) # 越大越尖锐
        peak_solidity = _get_safe_series_local('dominant_peak_solidity_D', 0.5) # 0-1
        peak_volume_ratio = _get_safe_series_local('dominant_peak_volume_ratio_D', 0.5) # 0-1
        chip_fault = _get_safe_series_local('chip_fault_blockage_ratio_D', 0.0) # 0-1，越低越好
        # 集中度健康度：基尼系数越低，代表成本越集中，健康度越高
        concentration_health = (1 - cost_gini).clip(0, 1)
        # 筹码峰质量：稳固度、量能占比、以及峰的尖锐度（峰态）三者结合
        # 使用滚动百分位排名对峰态系数进行归一化，使其成为一个稳定的[0,1]因子
        normalized_kurtosis = peak_kurtosis.rolling(window=120, min_periods=20).rank(pct=True).fillna(0.5)
        peak_quality = (peak_solidity * peak_volume_ratio * normalized_kurtosis).clip(0, 1)
        # 堵塞惩罚：堵塞比率越高，惩罚越大
        blockage_penalty = (1 - chip_fault)
        # 集中度与结构优化融合 (映射到 0-1)
        # 重新分配权重，突出集中度(基尼系数)和峰质量的核心地位
        concentration_score = (
            concentration_health * 0.5 +
            peak_quality * 0.4 +
            blockage_penalty * 0.1
        ).clip(0, 1)
        # --- 2. 成本与盈亏结构动态 (Cost & P/L Structure Dynamics Score) ---
        total_winner_rate = _get_safe_series_local('total_winner_rate_D', 0.5) # 0-1
        total_loser_rate = _get_safe_series_local('total_loser_rate_D', 0.5) # 0-1
        winner_profit_margin = _get_safe_series_local('winner_profit_margin_avg_D', 0.0) / 100 # 归一化到 0-1
        loser_loss_margin = _get_safe_series_local('loser_loss_margin_avg_D', 0.0) / 100 # 归一化到 0-1
        # 使用 cost_structure_skewness_D 替代 cost_divergence_normalized_D
        cost_divergence = _get_safe_series_local('cost_structure_skewness_D', 0.0) # -1到1
        mf_cost_advantage = _get_safe_series_local('main_force_cost_advantage_D', 0.0) # -1到1
        # 使用 profit_taking_flow_ratio_D 和 loser_pain_index_D 替代
        imminent_profit_taking = _get_safe_series_local('profit_taking_flow_ratio_D', 0.0) # 0-1
        loser_capitulation_pressure = _get_safe_series_local('loser_pain_index_D', 0.0) # 0-1
        # 获利盘压力：获利盘比例、利润率、抛压意愿的乘积 (负向贡献)
        profit_pressure = total_winner_rate * winner_profit_margin * imminent_profit_taking
        # 亏损盘支撑：亏损盘比例、亏损率、投降压力低的乘积 (正向贡献)
        loser_support = total_loser_rate * loser_loss_margin * (1 - loser_capitulation_pressure)
        # 成本优势：主力成本优势与成本发散度的结合
        cost_advantage_score = (mf_cost_advantage - cost_divergence).clip(-1, 1)
        # 成本与盈亏结构融合 (映射到 0-1)
        cost_structure_score = (
            (loser_support * 0.4) +
            ((cost_advantage_score + 1) / 2 * 0.4) - # 映射到 0到1
            (profit_pressure * 0.2)
        ).clip(0, 1)
        # --- 3. 持股心态与交易行为 (Holder Sentiment & Behavior Score) ---
        # 使用 winner_stability_index_D 和 (1 - capitulation_flow_ratio_D) 等替代
        winner_conviction = _get_safe_series_local('winner_stability_index_D', 0.0) # 0-1
        chip_fatigue = _get_safe_series_local('chip_fatigue_index_D', 0.0) # 0-1
        locked_profit = _get_safe_series_local('winner_stability_index_D', 0.0) # 0-1
        locked_loss = 1.0 - _get_safe_series_local('capitulation_flow_ratio_D', 0.0) # 0-1
        capitulation_absorption = _get_safe_series_local('capitulation_absorption_index_D', 0.0) # 0-1
        active_buying = _get_safe_series_local('active_buying_support_D', 0.0) # 0-1
        active_selling = _get_safe_series_local('active_selling_pressure_D', 0.0) # 0-1
        # 使用 mf_retail_battle_intensity_D 替代 active_zone_combat_intensity_D
        combat_intensity = _get_safe_series_local('mf_retail_battle_intensity_D', 0.0) # 0-1
        # 信念与锁定：信念、锁定利润盘正向，疲劳、锁定亏损盘负向
        conviction_lock_score = (winner_conviction + locked_profit - chip_fatigue - locked_loss).clip(-1, 1)
        # 吸收与支撑：吸收能力、主动买盘正向，主动卖盘负向
        absorption_support_score = (capitulation_absorption + active_buying - active_selling).clip(-1, 1)
        # 持股心态与交易行为融合 (映射到 0-1)
        sentiment_score = (
            ((conviction_lock_score + 1) / 2 * 0.4) +
            ((absorption_support_score + 1) / 2 * 0.4) +
            (combat_intensity * 0.2)
        ).clip(0, 1)
        # --- 4. 主力控盘与意图 (Main Force Control & Intent Score) ---
        # 使用 control_solidity_index_D 替代 main_force_control_leverage_D
        mf_control_leverage = _get_safe_series_local('control_solidity_index_D', 0.0) # 0-1
        mf_on_peak_flow = _get_safe_series_local('main_force_on_peak_flow_D', 0.0) # 金额，需要归一化
        mf_flow_directionality = _get_safe_series_local('main_force_flow_directionality_D', 0.0) # -1到1
        mf_execution_alpha = _get_safe_series_local('main_force_execution_alpha_D', 0.0) # -1到1
        mf_conviction_index = _get_safe_series_local('main_force_conviction_index_D', 0.0) # 0-1
        mf_vpoc_premium = _get_safe_series_local('mf_vpoc_premium_D', 0.0) # -1到1
        vwap_control_strength = _get_safe_series_local('vwap_control_strength_D', 0.0) # 0-1
        turnover_rate_f = _get_safe_series_local('turnover_rate_f_D', 0.0) # 百分比
        # 控盘强度：控盘杠杆与VWAP控制强度乘积
        control_strength = mf_control_leverage * vwap_control_strength
        # 峰值操作：主力在筹码峰上的资金流，归一化到 0-1
        mf_on_peak_flow_normalized = (mf_on_peak_flow.rank(pct=True) * 2 - 1).clip(0, 1) # 只取正向贡献
        # 资金意图：方向性、执行效率、信念的乘积
        mf_intent = mf_flow_directionality * mf_execution_alpha * mf_conviction_index
        # 成本优势：主力VPOC溢价
        mf_cost_advantage_final = (mf_vpoc_premium + 1) / 2 # 映射到 0-1
        # 换手率健康度：适中为好，过高过低惩罚
        turnover_health = pd.Series(1.0, index=df_index)
        turnover_health[turnover_rate_f < 2] = turnover_rate_f[turnover_rate_f < 2] / 2
        turnover_health[turnover_rate_f > 15] = 1 - (turnover_rate_f[turnover_rate_f > 15] - 15) / 10
        turnover_health = turnover_health.clip(0, 1)
        # 主力控盘与意图融合 (映射到 0-1)
        main_force_score = (
            control_strength * 0.3 +
            mf_on_peak_flow_normalized * 0.2 +
            ((mf_intent + 1) / 2 * 0.3) + # 映射到 0-1
            mf_cost_advantage_final * 0.1 +
            turnover_health * 0.1
        ).clip(0, 1)
        # --- 最终 OCH_D 融合 ---
        # 将四个维度分数进行加权平均，并映射到 [-1, 1]
        # 权重可以根据实际回测效果进行调整
        och_score = (
            concentration_score * 0.25 +
            cost_structure_score * 0.25 +
            sentiment_score * 0.25 +
            main_force_score * 0.25
        ) * 2 - 1 # 映射到 [-1, 1]
        df['OCH_D'] = och_score.astype(np.float32)
        all_dfs[timeframe] = df
        logger.info("OCH 指标计算完成。")
        return all_dfs








