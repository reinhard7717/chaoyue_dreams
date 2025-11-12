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
        【V3.0 · 物理洞察重铸版】元特征计算车间
        - 核心升级: 废弃旧的简单指标，引入分形维度、样本熵、波动率不稳定性等一系列能够深度刻画市场混沌、分形与信息熵的“元特征”。
        - 解决方案: 所有计算周期均由配置文件中的 'meta_feature_params' 驱动，实现完全的参数化。
        - 新增指标:
          - FRACTAL_DIMENSION: 市场复杂度，衡量价格曲线的“粗糙度”，值越高越无序。
          - SAMPLE_ENTROPY: 市场可预测性，衡量价格序列的“信息熵”，值越低越有序。
          - VOLATILITY_INSTABILITY_INDEX: 波动率稳定性，衡量波动率自身的波动，是状态切换的前兆。
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
        all_dfs[timeframe] = df
        return all_dfs

    async def calculate_pattern_recognition_signals(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V3.2 · 势能分析集成与逻辑强化版】高级模式识别信号生产线
        - 核心升级: 彻底摆脱对外部复合指标(如 breakout_quality_score)的依赖，解决流程依赖倒置问题。
        - 解决方案: 直接集成并使用我们最先进的“均线系统势能分析”三大核心指标（张力、有序性、压缩速率），
                      将模式识别的逻辑从简单的形态判断，升维到对市场“状态”与“势能”的综合评估，更直指A股博弈本质。
        - 逻辑强化: 细化了盘整、吸筹、突破、派发等模式的判断条件，引入更多高级筹码、资金流和VPA效率指标进行多因子共振确认。
        """
        timeframe = 'D' # 模式识别的核心战场在日线
        if timeframe not in all_dfs:
            return all_dfs
        df = all_dfs[timeframe]
        # 1. 【军火库点验】: 确认新一代核心武器（列名）已部署
        #    我们直接使用新数据模型中的精确列名，并集成均线势能指标。
        required_cols = [
            # 基础数据
            'high_D', 'low_D', 'close_D', 'volume_D', 'pct_change_D', 'VOL_MA_21_D', 'BBW_21_2.0_D', 'ATR_14_D', 'ADX_14_D',
            # 新版高级筹码指标
            'chip_health_score_D', 'structural_resilience_index_D', 'suppressive_accumulation_intensity_D',
            'rally_distribution_intensity_D', 'winner_conviction_index_D', 'cost_structure_skewness_D',
            'dominant_peak_solidity_D',
            # 新版高级资金流指标
            'main_force_net_flow_calibrated_D', 'main_force_execution_alpha_D', 'dip_absorption_power_D',
            'main_force_on_peak_flow_D', 'flow_efficiency_index_D', 'main_force_flow_directionality_D',
            # 新版均线系统势能指标 (核心升级)
            'MA_POTENTIAL_TENSION_INDEX_D', # 修正列名，移除_SHORT
            'MA_POTENTIAL_ORDERLINESS_SCORE_D', # 修正列名，移除_SHORT
            'MA_POTENTIAL_COMPRESSION_RATE_D', # 修正列名，移除_SHORT
            # VPA效率指标
            'VPA_EFFICIENCY_D'
        ]
        # 严格检查数据完整性
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"高级模式识别引擎缺少关键数据: {missing_cols}，模块已跳过！")
            print("当前可用的列名包括:")
            print(df.columns.tolist())
            return all_dfs
        # --- 2. 【战场状态定义】: 基于“状态+势能”的多因子共振 ---
        # 【状态一：高势能盘整 (High-Potential Consolidation)】
        # 定义：均线系统被高度压缩（弹簧压紧），但方向尚不明朗。这是变盘的温床。
        # 证据a: 高张力。均线张力指数处于历史低位（值越小，张力越高）。
        cond_high_tension = df['MA_POTENTIAL_TENSION_INDEX_D'] < df['MA_POTENTIAL_TENSION_INDEX_D'].rolling(60).quantile(0.20)
        # 证据b: 低有序度。均线系统处于纠缠状态，多空方向不明。
        cond_low_orderliness = df['MA_POTENTIAL_ORDERLINESS_SCORE_D'].abs() < 0.5
        # 证据c: 结构健康。筹码健康分和结构韧性指数保持在较高水平，或稳步提升。
        cond_struct_healthy = (df['chip_health_score_D'] > 50) & (df['structural_resilience_index_D'] > 0)
        # 证据d: 价格波动收缩。布林带宽度处于低位。
        cond_low_volatility = df['BBW_21_2.0_D'] < df['BBW_21_2.0_D'].rolling(60).quantile(0.25)
        # 证据e: 趋势强度弱。ADX 指标低于25。
        cond_weak_trend = df['ADX_14_D'] < 25
        df['IS_HIGH_POTENTIAL_CONSOLIDATION_D'] = cond_high_tension & cond_low_orderliness & cond_struct_healthy & cond_low_volatility & cond_weak_trend
        # 【状态二：吸筹进行时 (Accumulation in Progress)】
        # 定义：在高势能盘整中，出现主力资金吸筹迹象，且能量正在进一步积蓄。
        # 证据a: 能量正在压缩。均线压缩速率为负，表明弹簧正在被进一步压紧。
        cond_compressing = df['MA_POTENTIAL_COMPRESSION_RATE_D'] < 0
        # 证据b: 主力隐蔽吸筹。通过打压或逢低吸纳的方式。
        cond_main_force_accum = (df['suppressive_accumulation_intensity_D'] > 0) | (df['dip_absorption_power_D'] > 0.5)
        # 证据c: 主力在关键区域加仓。
        cond_peak_flow_positive = df['main_force_on_peak_flow_D'].rolling(3).mean() > 0
        # 证据d: VPA效率良好。在价格波动不大的情况下，VPA效率仍能保持中等水平，说明买盘效率高。
        cond_vpa_efficient_accum = df['VPA_EFFICIENCY_D'] > df['VPA_EFFICIENCY_D'].rolling(21).quantile(0.5)
        df['IS_ACCUMULATION_D'] = df['IS_HIGH_POTENTIAL_CONSOLIDATION_D'] & cond_compressing & (cond_main_force_accum | cond_peak_flow_positive) & cond_vpa_efficient_accum
        # 【状态三：高质效突破 (High-Quality Breakout)】
        # 定义：前期积蓄了巨大势能，现由主力资金点燃，向上释放能量。
        # 证据a: 突破前期高势能盘整。前一日处于“高势能盘整期”。
        cond_was_consolidating = df['IS_HIGH_POTENTIAL_CONSOLIDATION_D'].shift(1).fillna(False)
        # 证据b: 势能向上释放。均线有序性评分由负或零显著转正，代表多头排列形成。
        cond_orderliness_turn_up = (df['MA_POTENTIAL_ORDERLINESS_SCORE_D'] > 0.8) & (df['MA_POTENTIAL_ORDERLINESS_SCORE_D'].diff() > 0.3)
        # 证据c: 主力强力点火。主力资金当日大幅净流入，且执行效率高（Alpha为正），资金流向性强。
        cond_main_force_ignition = (df['main_force_net_flow_calibrated_D'] > 0) & \
                                   (df['main_force_execution_alpha_D'] > 0) & \
                                   (df['main_force_flow_directionality_D'] > 0.6) # 资金流向性强
        # 证据d: 价量配合。当日为放量阳线，且VPA效率极高。
        cond_price_volume_confirm = (df['pct_change_D'] > 0.01) & \
                                    (df['volume_D'] > df['VOL_MA_21_D'] * 1.2) & \
                                    (df['VPA_EFFICIENCY_D'] > df['VPA_EFFICIENCY_D'].rolling(21).quantile(0.9)) # VPA效率极高
        df['IS_BREAKOUT_D'] = cond_was_consolidating & cond_orderliness_turn_up & cond_main_force_ignition & cond_price_volume_confirm
        # 【状态四：派发嫌疑 (Distribution Suspected)】
        # 定义：上涨乏力，或在盘整中，出现主力资金悄然出货的迹象。
        # 证据a: 拉高出货。价格上涨，但“拉高出货强度”指标显著为正。
        cond_rally_dist = (df['pct_change_D'] > 0) & (df['rally_distribution_intensity_D'] > 0.5)
        # 证据b: 主力资金连续净流出，且资金流向性为负。
        cond_main_force_outflow = (df['main_force_net_flow_calibrated_D'].rolling(3).sum() < 0) & \
                                  (df['main_force_flow_directionality_D'] < -0.3) # 资金流向性为负
        # 证据c: 获利盘信念动摇。获利盘信念指数开始下降。
        cond_winner_conviction_drop = df['winner_conviction_index_D'].diff() < 0
        # 证据d: 结构韧性下降。
        cond_resilience_drop = df['structural_resilience_index_D'].diff() < 0
        # 证据e: VPA效率低下。价格上涨但VPA效率低，说明拉升吃力，抛压大。
        cond_vpa_inefficient_dist = (df['pct_change_D'] > 0) & (df['VPA_EFFICIENCY_D'] < df['VPA_EFFICIENCY_D'].rolling(21).quantile(0.2))
        df['IS_DISTRIBUTION_D'] = cond_rally_dist | (cond_main_force_outflow & cond_winner_conviction_drop & cond_resilience_drop & cond_vpa_inefficient_dist)
        # --- 3. 【信号整合与输出】 ---
        pattern_cols = ['IS_HIGH_POTENTIAL_CONSOLIDATION_D', 'IS_ACCUMULATION_D', 'IS_BREAKOUT_D', 'IS_DISTRIBUTION_D']
        for col in pattern_cols:
            if col in df.columns:
                df[col] = df[col].fillna(False).astype(bool)
        all_dfs[timeframe] = df
        logger.info("高级模式识别引擎(V3.2 势能分析集成与逻辑强化版)分析完成。")
        return all_dfs

    async def calculate_ma_convergence(self, all_dfs: Dict[str, pd.DataFrame], params: dict) -> Dict[str, pd.DataFrame]:
        """
        【V2.1 · 均线势能命名优化版】均线系统势能分析引擎
        - 核心优化: 确保输出的均线势能指标列名具有明确的前缀，避免出现 'None_' 导致的数据引用问题。
        """
        if not params.get('enabled', False):
            return all_dfs
        for conv_config in params.get('configs', []):
            periods = conv_config.get('periods', [])
            if not periods:
                continue
            # 确保 output_prefix 有一个默认值，例如 'MA_POTENTIAL'
            output_prefix = conv_config.get('output_column_prefix', 'MA_POTENTIAL')
            for timeframe in conv_config.get('apply_on', []):
                if timeframe not in all_dfs or all_dfs[timeframe].empty:
                    continue
                df = all_dfs[timeframe]
                # 1. 【军火库点验】: 确认计算所需的核心数据
                ma_cols = [f"EMA_{p}_{timeframe}" for p in periods]
                # ATR周期通常与均线簇中的中位数或平均周期相关，这里我们硬编码一个常用的14
                atr_col = f"ATR_14_{timeframe}" 
                required_cols = ma_cols + [atr_col]
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    logger.warning(f"均线系统势能分析失败({output_prefix}_{timeframe})：缺少核心数据列 {missing_cols}")
                    continue
                try:
                    ma_df = df[ma_cols]
                    # 2. 【计算势能大小】: 均线张力指数 (MA_TENSION_INDEX)
                    ma_range = ma_df.max(axis=1) - ma_df.min(axis=1)
                    atr = df[atr_col].replace(0, np.nan)
                    tension_index = ma_range / atr
                    tension_col_name = f"{output_prefix}_TENSION_INDEX_{timeframe}"
                    df[tension_col_name] = tension_index.fillna(0)
                    # 3. 【计算势能方向】: 均线有序性评分 (MA_ORDERLINESS_SCORE)
                    periods_series = pd.Series(periods)
                    # 创建一个DataFrame，其中每一行都是均线周期列表，用于与ma_df进行逐行排名比较
                    periods_repeated_df = pd.DataFrame([periods_series.values] * len(df), index=df.index, columns=ma_df.columns)
                    # 计算两组数据的等级。使用 'average' 方法处理并列排名。
                    rank_x = periods_repeated_df.rank(axis=1, method='average')
                    rank_y = ma_df.rank(axis=1, method='average')
                    # 计算Spearman相关性
                    # Spearman rho = 1 - 6 * sum(d^2) / (n * (n^2 - 1))
                    d_sq = (rank_x - rank_y).pow(2).sum(axis=1)
                    n = len(periods)
                    # 处理 n <= 1 的情况，避免除以零或产生无意义的结果
                    if n <= 1:
                        spearman_corr = pd.Series(np.nan, index=df.index)
                    else:
                        spearman_corr = 1 - (6 * d_sq) / (n * (n**2 - 1))
                    orderliness_col_name = f"{output_prefix}_ORDERLINESS_SCORE_{timeframe}"
                    df[orderliness_col_name] = spearman_corr.fillna(0)
                    # 4. 【计算势能变化率】: 均线压缩速率 (MA_COMPRESSION_RATE)
                    # 使用5日线性回归斜率来衡量压缩速率，负值表示正在压缩
                    compression_rate = df.ta.linreg(close=df[tension_col_name], length=5, slope=True)
                    compression_col_name = f"{output_prefix}_COMPRESSION_RATE_{timeframe}"
                    # ta.linreg 可能返回DataFrame，确保取其Series
                    if isinstance(compression_rate, pd.DataFrame):
                        compression_rate = compression_rate.iloc[:, 0]
                    df[compression_col_name] = compression_rate.fillna(0)
                except Exception as e:
                    logger.error(f"计算均线系统势能时发生错误({output_prefix}_{timeframe}): {e}", exc_info=True)
        return all_dfs

    async def calculate_consolidation_period(self, all_dfs: Dict[str, pd.DataFrame], params: dict) -> Dict[str, pd.DataFrame]:
        """
        【V1.0 新增】根据多因子共振识别盘整期。
        - 核心职责: 从 indicator_calculate_services.py 移入，作为特征工程的一部分。
        """
        if not params.get('enabled', False):
            return all_dfs
        timeframe = 'D' # 此特征通常在日线计算
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
        required_cols = [bbw_col, roc_col, vol_ma_col, f'high_{timeframe}', f'low_{timeframe}', f'volume_{timeframe}']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logger.warning(f"盘整期计算跳过，依赖的列 '{', '.join(missing)}' 不存在。")
            return all_dfs
        bbw_quantile = params.get('bbw_quantile', 0.25)
        roc_threshold = params.get('roc_threshold', 5.0)
        min_expanding_periods = boll_period * 2
        dynamic_bbw_threshold = df[bbw_col].expanding(min_periods=min_expanding_periods).quantile(bbw_quantile).bfill()
        df[f'dynamic_bbw_threshold_{timeframe}'] = dynamic_bbw_threshold
        cond_volatility = df[bbw_col] < df[f'dynamic_bbw_threshold_{timeframe}']
        cond_trend = df[roc_col].abs() < roc_threshold
        cond_volume = df[f'volume_{timeframe}'] < df[vol_ma_col]
        is_consolidating = cond_volatility & cond_trend & cond_volume
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
        【V1.3 · 后门封堵版】形态增强信号编排器
        - 核心修复: 封堵了命名协议的后门。在合并由外部计算器生成的、不带后缀的分钟级信号后，
                      强制为这些新列添加 '_D' 后缀，确保数据流的绝对标准化。
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
        # 为所有新合并的、没有后缀的列，强制添加 '_D' 后缀
        rename_map = {col: f"{col}_D" for col in new_cols_no_suffix if col in df_daily.columns}
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
        【V3.1 · 后门封堵版】突破质量分计算专用通道
        - 核心修复: 封堵了命名协议的后门。在合并由外部计算器生成的、不带后缀的 'breakout_quality_score' 后，
                      立即将其重命名为 'breakout_quality_score_D'，确保数据流的绝对标准化。
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










