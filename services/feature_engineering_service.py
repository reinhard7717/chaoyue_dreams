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
    def __init__(self, calculator):
        self.calculator = calculator

    async def calculate_all_slopes(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V3.4 · 调试信息清理版】计算所有配置的斜率特征。
        - 核心逻辑: 根据配置文件中的'series_to_slope'部分，为指定的数据列（如'MACD_12_26_9_D'）在不同的时间窗口（lookbacks）上计算线性回归斜率。
        - 优化: 保持原有的高效向量化计算，增加详尽注释。
        - 【新增】支持对结构与形态指标计算斜率。
        - 【清理】移除了调试打印信息。
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
                logger.warning(f"SLOPE计算跳过: 周期 '{timeframe}' 的源列 '{col_pattern}' 不存在。")
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
        【V2.4 · 调试信息清理版】计算所有配置的加速度特征。
        - 核心逻辑: 加速度是斜率的斜率。此方法基于已计算好的斜率特征（SLOPE_*），再次计算其斜率，从而得到加速度特征（ACCEL_*）。
        - 优化: 保持原有的高效向量化计算，增加详尽注释。
        - 【新增】支持对结构与形态指标计算加速度。
        - 【清理】移除了调试打印信息。
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
                    logger.warning(f"ACCEL计算跳过: 周期 '{timeframe}' 的依赖斜率列 '{slope_col_name}' 不存在。")
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
        【V1.2 · 细粒度VPA升级版】VPA效率指标生产线
        - 核心职责: 计算自定义指标 VPA_EFFICIENCY_D (价量攻击效率)，并新增买卖方细粒度VPA效率。
        - 指标含义:
            - VPA_EFFICIENCY_D: 衡量单位“超额”成交量所带来的价格涨幅。
            - VPA_BUY_EFFICIENCY_D: 衡量买方资金推动价格上涨的效率。
            - VPA_SELL_EFFICIENCY_D: 衡量卖方资金推动价格下跌的效率。
        - 优化: 增加细粒度买卖方VPA效率的计算。
        """
        timeframe = 'D' # 此特征仅在日线级别计算
        if timeframe not in all_dfs:
            return all_dfs
        df = all_dfs[timeframe]
        # 检查计算所需的列是否存在
        required_cols = ['pct_change_D', 'volume_D', 'VOL_MA_21_D', 'main_force_buy_ofi_D', 'main_force_sell_ofi_D']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logger.warning(f"VPA效率生产线缺少关键数据: {missing}，模块已跳过！")
            return all_dfs
        # 1. 计算总VPA效率
        volume_ratio = df['volume_D'] / df['VOL_MA_21_D'].replace(0, np.nan)
        vpa_efficiency = df['pct_change_D'] / volume_ratio.replace(0, np.nan)
        df['VPA_EFFICIENCY_D'] = vpa_efficiency.replace([np.inf, -np.inf], np.nan).fillna(0)
        # 2. 计算买方VPA效率 (VPA_BUY_EFFICIENCY_D)
        # 衡量买方资金推动价格上涨的效率
        # 使用主力买入OFI作为买方资金流的代表
        mf_buy_ofi_ma_21 = df['main_force_buy_ofi_D'].rolling(window=21, min_periods=1).mean()
        buy_flow_ratio = df['main_force_buy_ofi_D'] / mf_buy_ofi_ma_21.replace(0, np.nan)
        # 仅在价格上涨时计算买方效率，下跌时效率为0或负值
        buy_vpa_efficiency = df['pct_change_D'].apply(lambda x: max(0, x)) / buy_flow_ratio.replace(0, np.nan)
        df['VPA_BUY_EFFICIENCY_D'] = buy_vpa_efficiency.replace([np.inf, -np.inf], np.nan).fillna(0)
        # 3. 计算卖方VPA效率 (VPA_SELL_EFFICIENCY_D)
        # 衡量卖方资金推动价格下跌的效率
        # 使用主力卖出OFI作为卖方资金流的代表
        mf_sell_ofi_ma_21 = df['main_force_sell_ofi_D'].rolling(window=21, min_periods=1).mean()
        sell_flow_ratio = df['main_force_sell_ofi_D'] / mf_sell_ofi_ma_21.replace(0, np.nan)
        # 仅在价格下跌时计算卖方效率，上涨时效率为0或负值
        sell_vpa_efficiency = df['pct_change_D'].apply(lambda x: min(0, x)) / sell_flow_ratio.replace(0, np.nan)
        df['VPA_SELL_EFFICIENCY_D'] = sell_vpa_efficiency.replace([np.inf, -np.inf], np.nan).fillna(0)
        all_dfs[timeframe] = df
        return all_dfs

    async def calculate_meta_features(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V3.5 · nolds样本熵集成版】元特征计算车间
        - 核心升级: 废弃旧的简单指标，引入分形维度、样本熵、波动率不稳定性等一系列能够深度刻画市场混沌、分形与信息熵的“元特征”。
        - 新增优化: 直接集成预计算的、更高级的 `price_volume_entropy_D` 指标，作为对市场信息复杂度的核心度量。
        - 细粒度增强: 引入基于主力资金流和订单簿流动性的赫斯特指数和样本熵。
        - 周期调整: 调整元特征计算窗口为斐波那契数列。
        - 【修复】将近似熵计算替换为使用 `nolds` 库的样本熵，并更新相关列名。
        """
        timeframe = 'D'
        if timeframe not in all_dfs:
            return all_dfs
        df = all_dfs[timeframe]
        suffix = f"_{timeframe}"
        params = config.get('feature_engineering_params', {}).get('meta_feature_params', {})
        if not params.get('enabled', False):
            return all_dfs
        source_series_configs = [
            {'col': f'close{suffix}', 'prefix': ''}, # 确保 close_D 的 prefix 为空，这样赫斯特指数的列名就是 HURST_144d_D
            {'col': f'main_force_buy_ofi{suffix}', 'prefix': 'MF_BUY_OFI_'},
            {'col': f'bid_side_liquidity{suffix}', 'prefix': 'BID_LIQUIDITY_'}
        ]
        def _higuchi_fractal_dimension(x, k_max):
            L = []
            x_len = len(x)
            for k in range(1, k_max + 1):
                Lk = 0
                for m in range(k):
                    series = x[m::k]
                    if len(series) < 2: continue
                    Lk += np.sum(np.abs(np.diff(series))) * (x_len - 1) / ((x_len - m) // k * k)
                L.append(np.log(Lk / k) if Lk > 0 else 0)
            k_range_log = np.log(np.arange(1, k_max + 1))
            valid_indices = [i for i, val in enumerate(L) if val != 0]
            if len(valid_indices) < 2: return np.nan
            slope, _ = np.polyfit(k_range_log[valid_indices], np.array(L)[valid_indices], 1)
            return slope
        def _sample_entropy(x, m, r):
            n = len(x)
            templates = np.array([x[i:i+m] for i in range(n - m + 1)])
            dist = np.max(np.abs(templates[:, np.newaxis] - templates), axis=2)
            A = np.sum(dist < r) - n
            templates_plus_1 = np.array([x[i:i+m+1] for i in range(n - m)])
            dist_plus_1 = np.max(np.abs(templates_plus_1[:, np.newaxis] - templates_plus_1), axis=2)
            B = np.sum(dist_plus_1 < r) - (n - m)
            if A == 0 or B == 0: return np.nan
            return -np.log(B / A)
        # 【新增代码块】FFT能量比计算函数
        def _fft_energy_ratio(x, low_freq_cutoff_ratio=0.1, high_freq_cutoff_ratio=0.5):
            N = len(x)
            if N < 2: return np.nan
            yf = np.fft.fft(x)
            yf_abs = np.abs(yf[:N//2]) # 取正频率部分
            total_energy = np.sum(yf_abs**2)
            if total_energy == 0: return np.nan
            freqs = np.fft.fftfreq(N, d=1)[:N//2]
            low_freq_idx = int(N * low_freq_cutoff_ratio)
            high_freq_idx = int(N * high_freq_cutoff_ratio)
            low_freq_energy = np.sum(yf_abs[:low_freq_idx]**2)
            mid_freq_energy = np.sum(yf_abs[low_freq_idx:high_freq_idx]**2)
            high_freq_energy = np.sum(yf_abs[high_freq_idx:]**2)
            # 可以根据需求返回不同频率段的能量比，这里返回低频能量占比
            return low_freq_energy / total_energy
        for src_config in source_series_configs:
            source_col = src_config['col']
            prefix = src_config['prefix']
            if source_col not in df.columns:
                logger.warning(f"元特征计算缺少核心列 '{source_col}'，跳过其元特征计算。")
                continue
            current_series = df[source_col]
            if isinstance(current_series, pd.DataFrame):
                current_series = current_series.iloc[:, 0]
            # --- 1. Hurst 指数 (市场记忆性) ---
            # 修改代码行: 调整默认窗口为斐波那契数
            hurst_window = params.get('hurst_window', 144) # 144是斐波那契数，保持
            hurst_col = f'{prefix}HURST_{hurst_window}d{suffix}'
            if hurst_col not in df.columns: # 移除 len(current_series.dropna()) >= hurst_window 的判断，让 rolling.apply 自己处理 NaN
                try:
                    # 确保传递给 hurst_exponent 的是 Series，并且处理数据不足的情况
                    # 修改代码行: 确保 hurst_exponent 接收到的是数值类型，并处理数据不足的情况
                    df[hurst_col] = current_series.rolling(window=hurst_window, min_periods=hurst_window).apply(
                        lambda x: hurst_exponent(x.dropna().values) if len(x.dropna()) >= hurst_window else np.nan, raw=False
                    )
                except Exception as e:
                    logger.error(f"赫斯特指数(周期{hurst_window}, 列: {source_col})计算失败: {e}")
                    df[hurst_col] = np.nan
            # --- 2. 分形维度 (市场复杂度) ---
            # 修改代码行: 调整默认窗口为斐波那契数
            fd_window = params.get('fractal_dimension_window', 89) # 从100改为89
            fd_col = f'{prefix}FRACTAL_DIMENSION_{fd_window}d{suffix}'
            if fd_col not in df.columns: # 移除 len(current_series.dropna()) >= fd_window 的判断
                try:
                    k_max = int(np.sqrt(fd_window))
                    # 修改代码行: 确保 _higuchi_fractal_dimension 接收到的是数值类型，并处理数据不足的情况
                    df[fd_col] = current_series.rolling(window=fd_window, min_periods=fd_window).apply(
                        lambda x: _higuchi_fractal_dimension(x.dropna().values, k_max) if len(x.dropna()) >= fd_window else np.nan, raw=False
                    )
                except Exception as e:
                    logger.error(f"分形维度(周期{fd_window}, 列: {source_col})计算失败: {e}")
                    df[fd_col] = np.nan
            # --- 3. 样本熵 (市场可预测性) --- (使用自定义实现)
            # 修改代码行: 调整默认窗口为斐波那契数
            se_window = params.get('sample_entropy_window', 13) # 从10改为13
            se_tol_ratio = params.get('sample_entropy_tolerance_ratio', 0.2)
            se_col = f'{prefix}SAMPLE_ENTROPY_{se_window}d{suffix}'
            if se_col not in df.columns: # 移除 len(current_series.dropna()) >= se_window + 1 的判断
                try:
                    # 修改代码行: 重新组织样本熵的计算逻辑，使其更健壮
                    entropy_values = []
                    # 确保 rolling_std 在足够的数据点上计算
                    log_returns_or_series = current_series.dropna()
                    if len(log_returns_or_series) < se_window + 1:
                        df[se_col] = np.nan
                    else:
                        rolling_std = log_returns_or_series.rolling(window=se_window, min_periods=se_window).std()
                        for i in range(len(log_returns_or_series)):
                            if i < se_window - 1: # 窗口不足
                                entropy_values.append(np.nan)
                                continue
                            window_data = log_returns_or_series.iloc[i - se_window + 1 : i + 1].values
                            std_val = rolling_std.iloc[i]
                            r = std_val * se_tol_ratio
                            if pd.isna(r) or r == 0 or len(window_data) < se_window:
                                entropy_values.append(np.nan)
                                continue
                            entropy_values.append(_sample_entropy(window_data, m=2, r=r))
                        df[se_col] = pd.Series(entropy_values, index=log_returns_or_series.index).reindex(df.index)
                except Exception as e:
                    logger.error(f"样本熵(周期{se_window}, 列: {source_col})计算失败: {e}")
                    df[se_col] = np.nan
            # --- 4. 【修复】NOLDS样本熵 (替代近似熵) (时间序列复杂性) ---
            # 修改代码行：使用新的配置参数名称和列名
            nolds_sampen_window = params.get('approximate_entropy_window', 21) # 沿用原近似熵的窗口配置
            nolds_sampen_tol_ratio = params.get('approximate_entropy_tolerance_ratio', 0.2) # 沿用原近似熵的容忍度配置
            nolds_sampen_col = f'{prefix}NOLDS_SAMPLE_ENTROPY_{nolds_sampen_window}d{suffix}'
            if nolds_sampen_col not in df.columns:
                try:
                    # 修改代码行：调用 self.calculator 中重命名后的方法
                    df[nolds_sampen_col] = await self.calculator.calculate_nolds_sample_entropy(df=df, period=nolds_sampen_window, column=source_col, tolerance_ratio=nolds_sampen_tol_ratio)
                except Exception as e:
                    logger.error(f"NOLDS样本熵(周期{nolds_sampen_window}, 列: {source_col})计算失败: {e}")
                    df[nolds_sampen_col] = np.nan
            # --- 5. 【新增】FFT能量比 (FFT Energy Ratio) (频率结构) ---
            fft_window = params.get('fft_energy_ratio_window', 34) # 斐波那契数
            fft_col = f'{prefix}FFT_ENERGY_RATIO_{fft_window}d{suffix}'
            if fft_col not in df.columns:
                try:
                    energy_ratios = []
                    log_returns_or_series = current_series.dropna()
                    if len(log_returns_or_series) < fft_window:
                        df[fft_col] = np.nan
                    else:
                        for i in range(len(log_returns_or_series)):
                            if i < fft_window - 1:
                                energy_ratios.append(np.nan)
                                continue
                            window_data = log_returns_or_series.iloc[i - fft_window + 1 : i + 1].values
                            energy_ratios.append(_fft_energy_ratio(window_data))
                        df[fft_col] = pd.Series(energy_ratios, index=log_returns_or_series.index).reindex(df.index)
                except Exception as e:
                    logger.error(f"FFT能量比(周期{fft_window}, 列: {source_col})计算失败: {e}")
                    df[fft_col] = np.nan
        # --- 6. 波动率不稳定性 (状态切换前兆) ---
        # 修改代码行: 调整默认窗口为斐波那契数
        vi_window = params.get('volatility_instability_window', 21) # 21是斐波那契数，保持
        vi_col = f'VOLATILITY_INSTABILITY_INDEX_{vi_window}d{suffix}'
        atr_col = f'ATR_14{suffix}'
        if atr_col in df.columns and vi_col not in df.columns:
            df[vi_col] = df[atr_col].rolling(window=vi_window, min_periods=vi_window).std() # 确保有足够的min_periods
        # --- 7. 新增：集成价格成交量熵 (市场信息复杂度) ---
        pve_col_source = f'price_volume_entropy{suffix}'
        pve_col_target = f'PRICE_VOLUME_ENTROPY{suffix}'
        if pve_col_source in df.columns and pve_col_target not in df.columns:
            df[pve_col_target] = df[pve_col_source]
            logger.info("已成功集成预计算的'价格成交量熵'指标。")
            
        all_dfs[timeframe] = df
        return all_dfs

    async def calculate_pattern_recognition_signals(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V3.8 · 诡道博弈与情境自适应版】高级模式识别信号生产线
        - 核心升级: 全面引入新一代的结构、博弈和风险指标，将市场状态的识别精度从战术层面提升至战略层面。
                      - 盘整识别: 引入“结构张力指数”，要求筹码结构内部应力低。
                      - 吸筹识别: 引入“浮筹清洗效率”，验证主力吸筹的质量。
                      - 突破识别: 引入“突破就绪分”，作为突破信号的强确认。
                      - 派发识别: 引入“衰竭风险指数”，预警趋势顶部的系统性风险。
                      - 细粒度数据集成: 将旧的聚合指标替换为新的买卖双方细粒度指标，提升判断精度。
        - 【新增】引入“诡道博弈”模式识别，如诱空洗盘、假突破诱多等。
        - 【新增】引入情境自适应机制，根据市场波动率和趋势强度调整信号敏感度。
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
            'rally_sell_distribution_intensity_D',
            'rally_buy_support_weakness_D',
            'winner_stability_index_D',
            'cost_structure_skewness_D',
            'dominant_peak_solidity_D',
            'main_force_net_flow_calibrated_D',
            'main_force_buy_execution_alpha_D',
            'main_force_sell_execution_alpha_D',
            'dip_buy_absorption_strength_D',
            'dip_sell_pressure_resistance_D',
            'main_force_on_peak_buy_flow_D',
            'buy_flow_efficiency_index_D',
            'main_force_flow_directionality_D',
            'MA_POTENTIAL_TENSION_INDEX_D',
            'MA_POTENTIAL_ORDERLINESS_SCORE_D',
            'MA_POTENTIAL_COMPRESSION_RATE_D',
            'VPA_EFFICIENCY_D',
            'structural_tension_index_D',
            'floating_chip_cleansing_efficiency_D',
            'breakout_readiness_score_D',
            'exhaustion_risk_index_D',
            # 【新增代码行】诡道博弈相关指标
            'deception_lure_long_intensity_D',
            'deception_lure_short_intensity_D',
            'wash_trade_buy_volume_D',
            'wash_trade_sell_volume_D',
            'price_volume_entropy_D', # 用于情境自适应
            'VOLATILITY_INSTABILITY_INDEX_21d_D', # 用于情境自适应
            'trend_acceleration_score_D', # 来自 AdvancedStructuralMetrics
            'final_charge_intensity_D', # 来自 AdvancedStructuralMetrics
            'platform_conviction_score_D', # 来自 PlatformFeature
            'trend_conviction_score_D', # 来自 MultiTimeframeTrendline
            'quality_score_D', # 来自 PlatformFeature
            'validity_score_D' # 来自 TrendlineFeature
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"高级模式识别引擎缺少关键数据: {missing_cols}，模块已跳过！")
            print("当前可用的列名包括:")
            print(df.columns.tolist())
            return all_dfs
        # --- 1. 【情境自适应调制器】 ---
        # 根据市场波动率和趋势强度动态调整信号敏感度
        volatility_instability = df['VOLATILITY_INSTABILITY_INDEX_21d_D'].fillna(0).rolling(21).mean()
        trend_strength = df['ADX_14_D'].fillna(0).rolling(21).mean()
        # 波动率越高，信号越需要谨慎，趋势越强，信号越需要确认
        # 假设一个简单的调制函数：高波动率降低敏感度，强趋势提高确认度
        # 调制因子在 0.5 到 1.5 之间
        volatility_modulator = (1 - (volatility_instability.rolling(21).rank(pct=True) - 0.5) * 0.5).clip(0.5, 1.5)
        trend_modulator = (1 + (trend_strength.rolling(21).rank(pct=True) - 0.5) * 0.5).clip(0.5, 1.5)
        # --- 2. 【战场状态定义】: 基于“状态+势能”的多因子共振 ---
        cond_high_tension = df['MA_POTENTIAL_TENSION_INDEX_D'] < df['MA_POTENTIAL_TENSION_INDEX_D'].rolling(60).quantile(0.20)
        cond_low_orderliness = df['MA_POTENTIAL_ORDERLINESS_SCORE_D'].abs() < 0.5
        cond_struct_healthy = (df['chip_health_score_D'] > 50) & (df['dominant_peak_solidity_D'] > 0.5)
        cond_low_volatility = df['BBW_21_2.0_D'] < df['BBW_21_2.0_D'].rolling(60).quantile(0.25)
        cond_weak_trend = df['ADX_14_D'] < 25
        cond_low_structural_tension = df['structural_tension_index_D'] < df['structural_tension_index_D'].rolling(60).quantile(0.30)
        df['IS_HIGH_POTENTIAL_CONSOLIDATION_D'] = cond_high_tension & cond_low_orderliness & cond_struct_healthy & cond_low_volatility & cond_weak_trend & cond_low_structural_tension
        cond_compressing = df['MA_POTENTIAL_COMPRESSION_RATE_D'] < 0
        cond_chip_cleansing = df['floating_chip_cleansing_efficiency_D'] > 0.5
        cond_main_force_accum = (df['hidden_accumulation_intensity_D'] > 0) | \
                                ((df['dip_buy_absorption_strength_D'] > 0.5) & (df['dip_sell_pressure_resistance_D'] < 0.3)) | \
                                cond_chip_cleansing
        cond_peak_flow_positive = df['main_force_on_peak_buy_flow_D'].rolling(3).mean() > 0
        cond_vpa_efficient_accum = df['VPA_EFFICIENCY_D'] > df['VPA_EFFICIENCY_D'].rolling(21).quantile(0.5)
        df['IS_ACCUMULATION_D'] = df['IS_HIGH_POTENTIAL_CONSOLIDATION_D'] & cond_compressing & (cond_main_force_accum | cond_peak_flow_positive) & cond_vpa_efficient_accum
        cond_was_consolidating = df['IS_HIGH_POTENTIAL_CONSOLIDATION_D'].shift(1).fillna(False)
        cond_orderliness_turn_up = (df['MA_POTENTIAL_ORDERLINESS_SCORE_D'] > 0.8) & (df['MA_POTENTIAL_ORDERLINESS_SCORE_D'].diff() > 0.3)
        cond_main_force_ignition = (df['main_force_net_flow_calibrated_D'] > 0) & \
                                   (df['main_force_buy_execution_alpha_D'] > 0) & \
                                   (df['main_force_flow_directionality_D'] > 0.6)
        cond_price_volume_confirm = (df['pct_change_D'] > 0.01) & \
                                    (df['volume_D'] > df['VOL_MA_21_D'] * 1.2) & \
                                    (df['VPA_EFFICIENCY_D'] > df['VPA_EFFICIENCY_D'].rolling(21).quantile(0.9))
        cond_breakout_ready = df['breakout_readiness_score_D'] > 60
        # 【新增代码行】结合结构与形态指标，增强突破信号
        cond_platform_breakout_potential = (df['platform_conviction_score_D'] > 70) & (df['quality_score_D'] > 0.7) # 高质量平台
        cond_trendline_breakout_confirm = (df['trend_conviction_score_D'] > 80) & (df['validity_score_D'] > 0.8) # 强趋势线突破
        df['IS_BREAKOUT_D'] = (cond_was_consolidating & cond_orderliness_turn_up & cond_main_force_ignition & cond_price_volume_confirm & cond_breakout_ready) | \
                              (cond_platform_breakout_potential & cond_trendline_breakout_confirm) # 非线性融合
        cond_rally_dist = (df['pct_change_D'] > 0) & (df['rally_sell_distribution_intensity_D'] > 0.5) & (df['rally_buy_support_weakness_D'] > 0.5)
        cond_main_force_outflow = (df['main_force_net_flow_calibrated_D'].rolling(3).sum() < 0) & \
                                  (df['main_force_flow_directionality_D'] < -0.3) & \
                                  (df['main_force_sell_execution_alpha_D'] > 0)
        cond_winner_conviction_drop = df['winner_stability_index_D'].diff() < 0
        cond_resilience_drop = df['dominant_peak_solidity_D'].diff() < 0
        cond_vpa_inefficient_dist = (df['pct_change_D'] > 0) & (df['VPA_EFFICIENCY_D'] < df['VPA_EFFICIENCY_D'].rolling(21).quantile(0.2))
        cond_exhaustion_risk = df['exhaustion_risk_index_D'] > 70
        df['IS_DISTRIBUTION_D'] = cond_rally_dist | (cond_main_force_outflow & cond_winner_conviction_drop & cond_resilience_drop & cond_vpa_inefficient_dist) | cond_exhaustion_risk
        # --- 3. 【诡道博弈模式识别】 ---
        # 诱空洗盘 (Bear Trap Washout)
        # 逻辑：价格快速下跌（pct_change_D < -0.03），伴随洗盘交易量增加 (wash_trade_sell_volume_D > 0.5)，
        # 但主力资金流向（main_force_net_flow_calibrated_D）并未大幅流出，且筹码健康度（chip_health_score_D）保持稳定或小幅下降后迅速回升。
        # 结合情境自适应：在低波动率、弱趋势情境下，诱空洗盘的信号更可信。
        cond_sharp_drop = df['pct_change_D'] < -0.03
        cond_wash_sell_volume = df['wash_trade_sell_volume_D'] > 0.5
        cond_mf_not_outflow = df['main_force_net_flow_calibrated_D'].rolling(3).mean() > -0.01 # 主力净流出不明显
        cond_chip_resilient = (df['chip_health_score_D'].diff() > -10) | (df['chip_health_score_D'].shift(1) < df['chip_health_score_D']) # 筹码健康度未大幅恶化或开始回升
        df['IS_BEAR_TRAP_WASHOUT_D'] = (cond_sharp_drop & cond_wash_sell_volume & cond_mf_not_outflow & cond_chip_resilient) * volatility_modulator.fillna(1) # 乘以调制因子
        # 假突破诱多 (Bull Trap Lure)
        # 逻辑：价格突破（IS_BREAKOUT_D），但伴随主力资金流向（main_force_net_flow_calibrated_D）负向背离，
        # 且诱多欺骗强度（deception_lure_long_intensity_D）高，筹码派发迹象（rally_sell_distribution_intensity_D）明显。
        # 结合情境自适应：在高波动率、强趋势情境下，假突破的风险更高。
        cond_breakout_signal = df['IS_BREAKOUT_D']
        cond_mf_divergence = df['main_force_net_flow_calibrated_D'].rolling(5).mean().diff() < 0 # 主力资金动能减弱
        cond_lure_long = df['deception_lure_long_intensity_D'] > 0.6
        cond_dist_sign = df['rally_sell_distribution_intensity_D'] > 0.5
        df['IS_BULL_TRAP_LURE_D'] = (cond_breakout_signal & cond_mf_divergence & cond_lure_long & cond_dist_sign) * trend_modulator.fillna(1) # 乘以调制因子
        # 高位对倒出货 (High-Level Wash Trade Distribution)
        # 逻辑：股价处于高位（例如高于过去60日最高价的80%），成交量异常放大（volume_D > VOL_MA_21_D * 2），
        # 对倒买卖量（wash_trade_buy_volume_D + wash_trade_sell_volume_D）高，但价格涨幅有限（pct_change_D < 0.01），
        # 且主力资金净流出（main_force_net_flow_calibrated_D < 0）。
        cond_high_price = df['close_D'] > df['high_D'].rolling(60).max() * 0.8
        cond_volume_spike = df['volume_D'] > df['VOL_MA_21_D'] * 2
        cond_high_wash_trade = (df['wash_trade_buy_volume_D'] + df['wash_trade_sell_volume_D']) > 1.0 # 假设对倒量归一化后大于1
        cond_limited_gain = df['pct_change_D'] < 0.01
        cond_mf_outflow = df['main_force_net_flow_calibrated_D'] < 0
        df['IS_HIGH_LEVEL_WASH_DISTRIBUTION_D'] = cond_high_price & cond_volume_spike & cond_high_wash_trade & cond_limited_gain & cond_mf_outflow
        # --- 4. 【通达信模式集成】 ---
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
        # --- 5. 【信号整合与输出】 ---
        pattern_cols = [
            'IS_HIGH_POTENTIAL_CONSOLIDATION_D', 'IS_ACCUMULATION_D', 'IS_BREAKOUT_D', 'IS_DISTRIBUTION_D',
            'IS_BAZHAN_D', 'IS_WW1_D',
            'IS_BEAR_TRAP_WASHOUT_D', 'IS_BULL_TRAP_LURE_D', 'IS_HIGH_LEVEL_WASH_DISTRIBUTION_D' # 新增诡道博弈信号
        ]
        for col in pattern_cols:
            if col in df.columns:
                df[col] = df[col].fillna(False).astype(bool)
        all_dfs[timeframe] = df
        logger.info("高级模式识别引擎(V3.8 诡道博弈与情境自适应版)分析完成。")
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
        【V2.2 · 细粒度意图与斐波那契周期升级版】根据多因子共振识别盘整期。
        - 核心升级: 判定逻辑从单一的“几何形态”升级为“几何形态 或 主力意图”的双重标准。
                      即使K线形态不完美，只要筹码高度锁定且主力展现出强控盘意图，也将其识别为盘整构筑期。
                      增强主力意图判断，引入细粒度买卖方意图和欺骗指数。
                      周期参数调整为斐波那契数列。
        """
        if not params.get('enabled', False):
            return all_dfs
        timeframe = 'D'
        if timeframe not in all_dfs or all_dfs[timeframe].empty:
            return all_dfs
        df = all_dfs[timeframe]
        # 修改代码行: 调整默认周期为斐波那契数
        boll_period = params.get('boll_period', 21) # 21是斐波那契数，保持
        # 新增代码行: 从params中获取 boll_std，如果不存在则默认为 2.0
        boll_std = params.get('boll_std', 2.0)
        roc_period = params.get('roc_period', 13)   # 从12改为13
        vol_ma_period = params.get('vol_ma_period', 55) # 55是斐波那契数，保持
        bbw_col = f"BBW_{boll_period}_{float(boll_std)}_{timeframe}"
        roc_col = f"ROC_{roc_period}_{timeframe}"
        vol_ma_col = f"VOL_MA_{vol_ma_period}_{timeframe}"
        required_cols = [
            bbw_col, roc_col, vol_ma_col, f'high_{timeframe}', f'low_{timeframe}', f'volume_{timeframe}',
            'dominant_peak_solidity_D', 'control_solidity_index_D',
            'mf_cost_zone_buy_intent_D', 'mf_cost_zone_sell_intent_D',
            'deception_lure_long_intensity_D'
        ]
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
        is_classic_consolidation = cond_volatility & cond_trend & cond_volume
        cond_chips_locked = df['dominant_peak_solidity_D'] > 0.7
        cond_main_force_control = df['control_solidity_index_D'] > 0.6
        cond_mf_buy_intent_strong = df['mf_cost_zone_buy_intent_D'] > 0.5
        cond_mf_sell_intent_weak = df['mf_cost_zone_sell_intent_D'] < 0.3
        cond_no_deception_lure_long = df['deception_lure_long_intensity_D'] < 0.2
        is_intent_based_consolidation = cond_chips_locked & cond_main_force_control & \
                                         cond_mf_buy_intent_strong & cond_mf_sell_intent_weak & \
                                         cond_no_deception_lure_long
        is_consolidating = is_classic_consolidation | is_intent_based_consolidation
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
        # 修改代码行: 补充 calculate_breakout_quality_score 所需的所有列
        required_materials = [
            'volume', 'VOL_MA_21', 'main_force_flow_directionality',
            'open', 'high', 'low', 'close',
            'total_winner_rate', 'dominant_peak_solidity', 'VPA_EFFICIENCY',
            'main_force_buy_execution_alpha', 'upward_impulse_strength',
            'buy_order_book_clearing_rate', 'bid_side_liquidity',
            'vwap_cross_up_intensity', 'opening_buy_strength',
            'floating_chip_cleansing_efficiency',
            'VPA_BUY_EFFICIENCY',
            'deception_lure_long_intensity', 'wash_trade_buy_volume'
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
        【V1.1 · 细粒度NMFNF升级版】计算标准化主力净流量 (Normalized Main Force Net Flow, NMFNF)。
        - 核心逻辑: NMFNF = main_force_net_flow_calibrated_D / total_market_value_D。
        - 新增逻辑: NMFNF_BUY_D = main_force_buy_ofi_D / total_market_value_D。
        - 新增逻辑: NMFNF_SELL_D = main_force_sell_ofi_D / total_market_value_D。
        - 目的: 将主力净流量标准化，使其在不同市值和活跃度的股票间可比，并细化买卖方贡献。
        """
        timeframe = 'D'
        if timeframe not in all_dfs or all_dfs[timeframe].empty:
            logger.warning(f"计算 NMFNF 失败：缺少日线数据。")
            return all_dfs
        df = all_dfs[timeframe]
        required_cols = ['main_force_net_flow_calibrated_D', 'total_market_value_D', 'main_force_buy_ofi_D', 'main_force_sell_ofi_D']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"计算 NMFNF 缺少关键数据: {missing_cols}，模块已跳过！")
            return all_dfs
        # 避免除以零，将总市值中的零替换为 NaN，然后整个表达式会变为 NaN，最后fillna(0)
        total_market_value_safe = df['total_market_value_D'].replace(0, np.nan)
        # 计算总NMFNF
        nmfnf_series = df['main_force_net_flow_calibrated_D'] / total_market_value_safe
        df['NMFNF_D'] = nmfnf_series.fillna(0)
        # 计算买方NMFNF (NMFNF_BUY_D)
        nmfnf_buy_series = df['main_force_buy_ofi_D'] / total_market_value_safe
        df['NMFNF_BUY_D'] = nmfnf_buy_series.fillna(0)
        # 计算卖方NMFNF (NMFNF_SELL_D)
        nmfnf_sell_series = df['main_force_sell_ofi_D'] / total_market_value_safe
        df['NMFNF_SELL_D'] = nmfnf_sell_series.fillna(0)
        all_dfs[timeframe] = df
        logger.info("NMFNF 指标计算完成。")
        return all_dfs

    async def calculate_och(self, all_dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        【V3.3 · 结构动力学与细粒度博弈升级版】计算整体筹码健康度 (Overall Chip Health, OCH)。
        - 核心升级: 维度一“筹码集中度与结构优化”的计算逻辑重构，引入 cost_gini_coefficient 和 primary_peak_kurtosis 两个核心指标，
                    替代旧的 winner/loser_concentration 指标，实现对筹码结构更精准、更深刻的量化评估。
                    全面替换旧的聚合指标，引入新的买卖双方细粒度指标，并重新设计部分融合逻辑，以更精确地反映市场博弈的真实情况。
        - 【新增】引入非线性融合（tanh函数）和情境自适应（基于波动率和市场情绪）来增强OCH的鲁棒性和敏感度。
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
        # 更新必需列的列表，以匹配V33.0版本的新指标体系和细粒度数据
        required_cols = [
            # 维度一: 筹码集中度与结构优化
            'cost_gini_coefficient_D',
            'primary_peak_kurtosis_D',
            'dominant_peak_solidity_D', 'dominant_peak_volume_ratio_D',
            'chip_fault_blockage_ratio_D',
            # 维度二: 成本与盈亏结构动态
            'total_winner_rate_D', 'total_loser_rate_D',
            'winner_profit_margin_avg_D', 'loser_loss_margin_avg_D',
            'cost_structure_skewness_D',
            'main_force_cost_advantage_D',
            'profit_taking_flow_ratio_D',
            'loser_pain_index_D',
            'rally_sell_distribution_intensity_D',
            'dip_buy_absorption_strength_D',
            'panic_buy_absorption_contribution_D',
            # 维度三: 持股心态与交易行为
            'winner_stability_index_D',
            'chip_fatigue_index_D',
            'capitulation_flow_ratio_D',
            'capitulation_absorption_index_D',
            'active_buying_support_D', 'active_selling_pressure_D',
            'mf_retail_battle_intensity_D',
            'dip_buy_absorption_strength_D',
            'dip_sell_pressure_resistance_D',
            'rally_sell_distribution_intensity_D',
            'rally_buy_support_weakness_D',
            'panic_sell_volume_contribution_D',
            'panic_buy_absorption_contribution_D',
            'opening_buy_strength_D',
            'opening_sell_strength_D',
            'pre_closing_buy_posture_D',
            'pre_closing_sell_posture_D',
            'closing_auction_buy_ambush_D',
            'closing_auction_sell_ambush_D',
            'main_force_buy_ofi_D',
            'main_force_sell_ofi_D',
            'retail_buy_ofi_D',
            'retail_sell_ofi_D',
            'wash_trade_buy_volume_D',
            'wash_trade_sell_volume_D',
            'buy_order_book_clearing_rate_D',
            'sell_order_book_clearing_rate_D',
            'vwap_buy_control_strength_D',
            'vwap_sell_control_strength_D',
            'bid_side_liquidity_D',
            'ask_side_liquidity_D',
            # 维度四: 主力控盘与意图
            'control_solidity_index_D',
            'main_force_on_peak_buy_flow_D',
            'main_force_on_peak_sell_flow_D',
            'main_force_flow_directionality_D',
            'main_force_buy_execution_alpha_D',
            'main_force_sell_execution_alpha_D',
            'main_force_conviction_index_D', 'mf_vpoc_premium_D',
            'vwap_buy_control_strength_D',
            'vwap_sell_control_strength_D',
            'main_force_vwap_up_guidance_D',
            'main_force_vwap_down_guidance_D',
            'vwap_cross_up_intensity_D',
            'vwap_cross_down_intensity_D',
            'main_force_t0_buy_efficiency_D',
            'main_force_t0_sell_efficiency_D',
            'buy_flow_efficiency_index_D',
            'sell_flow_efficiency_index_D',
            'covert_distribution_signal_D',
            'supportive_distribution_intensity_D',
            'turnover_rate_f_D',
            # 【新增代码行】情境自适应相关指标
            'VOLATILITY_INSTABILITY_INDEX_21d_D',
            'market_sentiment_score_D',
            'price_volume_entropy_D'
        ]
        # 使用 _get_safe_series 确保所有数据都存在，并用默认值填充缺失值
        def _get_safe_series_local(col_name, default_val=0.0):
            if col_name not in df.columns:
                print(f"调试信息: OCH计算缺少列: {col_name}，使用默认值 {default_val}。")
                return pd.Series(default_val, index=df_index)
            return df[col_name].fillna(default_val)
        # --- 情境自适应调制器 ---
        # 波动率情境：高波动率时，筹码健康度可能更不稳定，需要更谨慎的评估
        volatility_context = _get_safe_series_local('VOLATILITY_INSTABILITY_INDEX_21d_D', 0.0).rolling(21).mean().fillna(0)
        # 市场情绪情境：极端情绪下，筹码健康度可能被扭曲
        sentiment_context = _get_safe_series_local('market_sentiment_score_D', 0.0).rolling(21).mean().fillna(0)
        # 市场信息复杂度情境：高复杂度时，信号可能更模糊
        entropy_context = _get_safe_series_local('price_volume_entropy_D', 0.0).rolling(21).mean().fillna(0)
        # 融合函数：使用 tanh 激活函数进行非线性融合，并考虑情境调制
        def _nonlinear_fusion(scores, weights, volatility_mod=1.0, sentiment_mod=1.0, entropy_mod=1.0):
            fused_score = pd.Series(0.0, index=df_index)
            for score_name, weight in weights.items():
                score_series = scores.get(score_name, pd.Series(0.0, index=df_index))
                # 动态调整权重或分数，例如：高波动率时，某些指标的权重可能降低
                # 这里使用一个简单的乘法调制，更复杂的可以根据具体指标设计
                modulated_score = score_series * (1 + volatility_mod * 0.1 - sentiment_mod * 0.05 - entropy_mod * 0.05)
                fused_score += modulated_score * weight
            # 使用 tanh 将分数映射到 [-1, 1] 之间，提供非线性压缩
            return np.tanh(fused_score)
        # --- 1. 筹码集中度与结构优化 (Concentration & Structure Optimization Score) ---
        cost_gini = _get_safe_series_local('cost_gini_coefficient_D', 0.5)
        peak_kurtosis = _get_safe_series_local('primary_peak_kurtosis_D', 3.0)
        peak_solidity = _get_safe_series_local('dominant_peak_solidity_D', 0.5)
        peak_volume_ratio = _get_safe_series_local('dominant_peak_volume_ratio_D', 0.5)
        chip_fault = _get_safe_series_local('chip_fault_blockage_ratio_D', 0.0)
        concentration_health = (1 - cost_gini).clip(0, 1)
        normalized_kurtosis = peak_kurtosis.rolling(window=120, min_periods=20).rank(pct=True).fillna(0.5)
        peak_quality = (peak_solidity * peak_volume_ratio * normalized_kurtosis).clip(0, 1)
        blockage_penalty = (1 - chip_fault)
        concentration_scores = {
            'concentration_health': concentration_health,
            'peak_quality': peak_quality,
            'blockage_penalty': blockage_penalty
        }
        concentration_weights = {'concentration_health': 0.5, 'peak_quality': 0.4, 'blockage_penalty': 0.1}
        concentration_score = _nonlinear_fusion(concentration_scores, concentration_weights, volatility_context, sentiment_context, entropy_context)
        # --- 2. 成本与盈亏结构动态 (Cost & P/L Structure Dynamics Score) ---
        total_winner_rate = _get_safe_series_local('total_winner_rate_D', 0.5)
        total_loser_rate = _get_safe_series_local('total_loser_rate_D', 0.5)
        winner_profit_margin = _get_safe_series_local('winner_profit_margin_avg_D', 0.0) / 100
        loser_loss_margin = _get_safe_series_local('loser_loss_margin_avg_D', 0.0) / 100
        cost_divergence = _get_safe_series_local('cost_structure_skewness_D', 0.0)
        mf_cost_advantage = _get_safe_series_local('main_force_cost_advantage_D', 0.0)
        imminent_profit_taking = _get_safe_series_local('profit_taking_flow_ratio_D', 0.0)
        loser_capitulation_pressure = _get_safe_series_local('loser_pain_index_D', 0.0)
        profit_pressure = total_winner_rate * winner_profit_margin * (_get_safe_series_local('rally_sell_distribution_intensity_D', 0.0) + imminent_profit_taking).clip(0,1)
        loser_support = total_loser_rate * loser_loss_margin * (1 - loser_capitulation_pressure) + \
                        _get_safe_series_local('dip_buy_absorption_strength_D', 0.0) * 0.5 + \
                        _get_safe_series_local('panic_buy_absorption_contribution_D', 0.0) * 0.5
        cost_advantage_score = (mf_cost_advantage - cost_divergence).clip(-1, 1)
        cost_structure_scores = {
            'loser_support': loser_support,
            'cost_advantage_score': cost_advantage_score,
            'profit_pressure': profit_pressure
        }
        cost_structure_weights = {'loser_support': 0.4, 'cost_advantage_score': 0.4, 'profit_pressure': -0.2} # 利润压力是负向权重
        cost_structure_score = _nonlinear_fusion(cost_structure_scores, cost_structure_weights, volatility_context, sentiment_context, entropy_context)
        # --- 3. 持股心态与交易行为 (Holder Sentiment & Behavior Score) ---
        winner_conviction = _get_safe_series_local('winner_stability_index_D', 0.0)
        chip_fatigue = _get_safe_series_local('chip_fatigue_index_D', 0.0)
        locked_profit = _get_safe_series_local('winner_stability_index_D', 0.0)
        locked_loss = 1.0 - _get_safe_series_local('capitulation_flow_ratio_D', 0.0)
        buy_side_absorption_composite = (
            _get_safe_series_local('capitulation_absorption_index_D', 0.0) * 0.2 +
            _get_safe_series_local('active_buying_support_D', 0.0) * 0.1 +
            _get_safe_series_local('dip_buy_absorption_strength_D', 0.0) * 0.1 +
            _get_safe_series_local('panic_buy_absorption_contribution_D', 0.0) * 0.1 +
            _get_safe_series_local('opening_buy_strength_D', 0.0) * 0.05 +
            _get_safe_series_local('pre_closing_buy_posture_D', 0.0) * 0.05 +
            _get_safe_series_local('closing_auction_buy_ambush_D', 0.0) * 0.05 +
            _get_safe_series_local('main_force_buy_ofi_D', 0.0) * 0.1 +
            _get_safe_series_local('retail_buy_ofi_D', 0.0) * 0.05 +
            _get_safe_series_local('bid_side_liquidity_D', 0.0) * 0.05 +
            _get_safe_series_local('buy_order_book_clearing_rate_D', 0.0) * 0.05 +
            _get_safe_series_local('vwap_buy_control_strength_D', 0.0) * 0.05
        ).clip(0, 1)
        sell_side_pressure_composite = (
            _get_safe_series_local('active_selling_pressure_D', 0.0) * 0.1 +
            _get_safe_series_local('rally_sell_distribution_intensity_D', 0.0) * 0.1 +
            _get_safe_series_local('dip_sell_pressure_resistance_D', 0.0) * 0.05 +
            _get_safe_series_local('panic_sell_volume_contribution_D', 0.0) * 0.1 +
            _get_safe_series_local('opening_sell_strength_D', 0.0) * 0.05 +
            _get_safe_series_local('pre_closing_sell_posture_D', 0.0) * 0.05 +
            _get_safe_series_local('closing_auction_sell_ambush_D', 0.0) * 0.05 +
            _get_safe_series_local('main_force_sell_ofi_D', 0.0) * 0.1 +
            _get_safe_series_local('retail_sell_ofi_D', 0.0) * 0.05 +
            _get_safe_series_local('ask_side_liquidity_D', 0.0) * 0.05 +
            _get_safe_series_local('sell_order_book_clearing_rate_D', 0.0) * 0.05 +
            _get_safe_series_local('vwap_sell_control_strength_D', 0.0) * 0.05
        ).clip(0, 1)
        combat_intensity = _get_safe_series_local('mf_retail_battle_intensity_D', 0.0)
        conviction_lock_score = (winner_conviction + locked_profit - chip_fatigue - locked_loss).clip(-1, 1)
        absorption_support_score = (buy_side_absorption_composite - sell_side_pressure_composite).clip(-1, 1)
        wash_trade_penalty = (_get_safe_series_local('wash_trade_buy_volume_D', 0.0) + _get_safe_series_local('wash_trade_sell_volume_D', 0.0)).clip(0, 1) * 0.1
        sentiment_scores = {
            'conviction_lock_score': (conviction_lock_score + 1) / 2, # 归一化到 [0, 1]
            'absorption_support_score': (absorption_support_score + 1) / 2, # 归一化到 [0, 1]
            'combat_intensity': combat_intensity,
            'wash_trade_penalty': wash_trade_penalty
        }
        sentiment_weights = {'conviction_lock_score': 0.4, 'absorption_support_score': 0.4, 'combat_intensity': 0.2, 'wash_trade_penalty': -0.1}
        sentiment_score = _nonlinear_fusion(sentiment_scores, sentiment_weights, volatility_context, sentiment_context, entropy_context)
        # --- 4. 主力控盘与意图 (Main Force Control & Intent Score) ---
        mf_control_leverage = _get_safe_series_local('control_solidity_index_D', 0.0)
        mf_on_peak_flow_composite = (_get_safe_series_local('main_force_on_peak_buy_flow_D', 0.0) - _get_safe_series_local('main_force_on_peak_sell_flow_D', 0.0))
        mf_on_peak_flow_normalized = (mf_on_peak_flow_composite.rank(pct=True) * 2 - 1).clip(0, 1)
        mf_intent_composite = (
            _get_safe_series_local('main_force_flow_directionality_D', 0.0) * 0.2 +
            (_get_safe_series_local('main_force_buy_execution_alpha_D', 0.0) - _get_safe_series_local('main_force_sell_execution_alpha_D', 0.0)) * 0.2 +
            _get_safe_series_local('main_force_conviction_index_D', 0.0) * 0.1 +
            (_get_safe_series_local('main_force_vwap_up_guidance_D', 0.0) - _get_safe_series_local('main_force_vwap_down_guidance_D', 0.0)) * 0.1 +
            (_get_safe_series_local('vwap_cross_up_intensity_D', 0.0) - _get_safe_series_local('vwap_cross_down_intensity_D', 0.0)) * 0.1 +
            (_get_safe_series_local('main_force_t0_buy_efficiency_D', 0.0) - _get_safe_series_local('main_force_t0_sell_efficiency_D', 0.0)) * 0.1 +
            (_get_safe_series_local('buy_flow_efficiency_index_D', 0.0) - _get_safe_series_local('sell_flow_efficiency_index_D', 0.0)) * 0.1
        ).clip(-1, 1)
        mf_vpoc_premium = _get_safe_series_local('mf_vpoc_premium_D', 0.0)
        vwap_control_composite = (_get_safe_series_local('vwap_buy_control_strength_D', 0.0) - _get_safe_series_local('vwap_sell_control_strength_D', 0.0))
        control_strength = mf_control_leverage * ((vwap_control_composite + 1) / 2)
        mf_cost_advantage_final = (mf_vpoc_premium + 1) / 2
        turnover_rate_f = _get_safe_series_local('turnover_rate_f_D', 0.0)
        turnover_health = pd.Series(1.0, index=df_index)
        turnover_health[turnover_rate_f < 2] = turnover_rate_f[turnover_rate_f < 2] / 2
        turnover_health[turnover_rate_f > 15] = 1 - (turnover_rate_f[turnover_rate_f > 15] - 15) / 10
        turnover_health = turnover_health.clip(0, 1)
        distribution_penalty = (_get_safe_series_local('covert_distribution_signal_D', 0.0) + _get_safe_series_local('supportive_distribution_intensity_D', 0.0)).clip(0, 1) * 0.1
        main_force_scores = {
            'control_strength': control_strength,
            'mf_on_peak_flow_normalized': mf_on_peak_flow_normalized,
            'mf_intent_composite': (mf_intent_composite + 1) / 2, # 归一化到 [0, 1]
            'mf_cost_advantage_final': mf_cost_advantage_final,
            'turnover_health': turnover_health,
            'distribution_penalty': distribution_penalty
        }
        main_force_weights = {
            'control_strength': 0.3, 'mf_on_peak_flow_normalized': 0.2, 'mf_intent_composite': 0.3,
            'mf_cost_advantage_final': 0.1, 'turnover_health': 0.1, 'distribution_penalty': -0.1
        }
        main_force_score = _nonlinear_fusion(main_force_scores, main_force_weights, volatility_context, sentiment_context, entropy_context)
        # --- 最终 OCH_D 融合 ---
        och_scores = {
            'concentration_score': concentration_score,
            'cost_structure_score': cost_structure_score,
            'sentiment_score': sentiment_score,
            'main_force_score': main_force_score
        }
        och_weights = {
            'concentration_score': 0.25, 'cost_structure_score': 0.25,
            'sentiment_score': 0.25, 'main_force_score': 0.25
        }
        # 最终OCH也使用非线性融合，并考虑情境自适应
        och_score = _nonlinear_fusion(och_scores, och_weights, volatility_context, sentiment_context, entropy_context) * 2 - 1 # 映射回 [-1, 1]
        df['OCH_D'] = och_score.astype(np.float32)
        all_dfs[timeframe] = df
        logger.info("OCH 指标计算完成。")
        return all_dfs

    async def calculate_structural_features(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V1.0 · 结构特征合成版】
        - 核心职责: 从 BaseAdvancedStructuralMetrics 中提取和合成更高级的结构性特征。
        - 指标含义:
            - STRUCTURAL_TREND_HEALTH_D: 综合评估趋势的健康度、加速状态和成交结构。
            - BREAKTHROUGH_CONVICTION_D: 直接使用突破信念分。
            - DEFENSE_SOLIDITY_D: 直接使用防守稳固度。
            - EQUILIBRIUM_COMPRESSION_D: 直接使用均衡压缩指数。
        """
        timeframe = 'D'
        if timeframe not in all_dfs or all_dfs[timeframe].empty:
            logger.warning(f"结构特征计算失败：缺少日线数据。")
            return all_dfs
        df = all_dfs[timeframe]
        # 检查计算所需的原始结构指标列是否存在
        required_structural_cols = [
            'trend_acceleration_score_D', 'final_charge_intensity_D', 'volume_structure_skew_D',
            'breakthrough_conviction_score_D', 'defense_solidity_score_D', 'equilibrium_compression_index_D'
        ]
        missing_cols = [col for col in required_structural_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"结构特征计算缺少关键数据: {missing_cols}，模块已跳过！")
            return all_dfs
        # 1. 合成 STRUCTURAL_TREND_HEALTH_D
        # 综合趋势加速、终场冲锋强度和成交结构偏度，评估趋势的整体健康度
        # 假设这些指标都已归一化到0-100或-1到1的范围
        df['STRUCTURAL_TREND_HEALTH_D'] = (
            df['trend_acceleration_score_D'] * 0.4 +
            df['final_charge_intensity_D'] * 0.3 +
            (1 - df['volume_structure_skew_D'].abs()) * 0.3 # 偏度越小，结构越健康
        ).clip(0, 100) # 假设输出范围是0-100
        # 2. 直接使用其他结构指标作为特征
        df['BREAKTHROUGH_CONVICTION_D'] = df['breakthrough_conviction_score_D']
        df['DEFENSE_SOLIDITY_D'] = df['defense_solidity_score_D']
        df['EQUILIBRIUM_COMPRESSION_D'] = df['equilibrium_compression_index_D']
        all_dfs[timeframe] = df
        logger.info("结构特征计算完成。")
        return all_dfs

    async def calculate_geometric_features(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V1.0 · 几何形态特征映射版】
        - 核心职责: 将平台、趋势线等几何形态数据映射到日线级别，并衍生出可用于量化的特征。
        - 处理逻辑:
            1. 对于 MultiTimeframeTrendline (已是日线快照)，直接提取其特征。
            2. 对于 PlatformFeature 和 TrendlineFeature (事件型)，将其特征在有效日期范围内映射到每日数据。
            3. 处理重叠事件，默认优先使用最新结束的事件。
        """
        timeframe = 'D'
        if timeframe not in all_dfs or all_dfs[timeframe].empty:
            logger.warning(f"几何形态特征计算失败：缺少日线数据。")
            return all_dfs
        df_daily = all_dfs[timeframe]
        df_index = df_daily.index
        # 初始化一个空的DataFrame来存储每日几何特征
        df_geometric_features = pd.DataFrame(index=df_index)
        # --- 1. 处理 MultiTimeframeTrendline (已是每日快照) ---
        # 假设这些数据已经通过 IndicatorService 加载并带有 _D 后缀
        multi_trendline_cols = [
            'trend_conviction_score_D', 'slope_D', 'intercept_D', 'validity_score_D'
        ]
        # 筛选出所有以 'multi_timeframe_trendline' 开头的列，并检查是否存在
        # 实际列名会是类似 'multi_timeframe_trendline_trend_conviction_score_D'
        # 这里需要更精确地匹配，或者假设这些列已经直接在df_daily中
        # 假设这些列已经直接在df_daily中，并且是针对不同period和line_type的
        # 例如：trend_conviction_score_5_support_D, trend_conviction_score_13_resistance_D
        # 暂时只处理通用的，更细致的需要根据实际列名来
        for col in df_daily.columns:
            if 'trend_conviction_score' in col or 'trendline_slope' in col or 'trendline_validity_score' in col:
                df_geometric_features[col] = df_daily[col]
        # --- 2. 处理 PlatformFeature (事件型数据) ---
        # 假设 PlatformFeature_D 已经通过 IndicatorService 加载，并且其索引是 end_date
        df_platforms = all_dfs.get('platform_feature_D')
        if df_platforms is not None and not df_platforms.empty:
            # 将平台特征映射到每日数据
            for idx, row in df_platforms.iterrows():
                start_date = pd.to_datetime(row['start_date_D'], utc=True).normalize()
                end_date = pd.to_datetime(idx, utc=True).normalize() # 索引已经是end_date
                
                # 确保日期范围在 df_daily 的索引范围内
                valid_dates = df_index[(df_index >= start_date) & (df_index <= end_date)]
                if not valid_dates.empty:
                    # 提取平台相关特征
                    platform_features = {
                        'PLATFORM_CONVICTION_SCORE_D': row.get('platform_conviction_score_D', np.nan),
                        'PLATFORM_QUALITY_SCORE_D': row.get('quality_score_D', np.nan),
                        'PLATFORM_DURATION_D': row.get('duration_D', np.nan),
                        'PLATFORM_CHARACTER_SCORE_D': row.get('character_score_D', np.nan),
                        'PLATFORM_BREAKOUT_READINESS_D': row.get('breakout_readiness_score_D', np.nan),
                        'PLATFORM_VPOC_D': row.get('vpoc_D', np.nan),
                        'PLATFORM_HIGH_D': row.get('high_D', np.nan),
                        'PLATFORM_LOW_D': row.get('low_D', np.nan),
                    }
                    # 映射到每日数据，处理重叠时，最新结束的平台优先
                    for date in valid_dates:
                        for feature_name, value in platform_features.items():
                            # 如果该日期已有值，且当前平台结束日期更晚，则更新
                            if pd.isna(df_geometric_features.loc[date, feature_name]) or date == end_date:
                                df_geometric_features.loc[date, feature_name] = value
            # 衍生平台相关特征
            if 'PLATFORM_HIGH_D' in df_geometric_features.columns and 'PLATFORM_LOW_D' in df_geometric_features.columns:
                df_geometric_features['PLATFORM_RANGE_PCT_D'] = (
                    (df_geometric_features['PLATFORM_HIGH_D'] - df_geometric_features['PLATFORM_LOW_D']) /
                    df_geometric_features['PLATFORM_LOW_D'].replace(0, np.nan)
                ).fillna(0)
            if 'PLATFORM_VPOC_D' in df_geometric_features.columns and 'close_D' in df_daily.columns:
                df_geometric_features['PLATFORM_VPOC_PREMIUM_D'] = (
                    (df_geometric_features['PLATFORM_VPOC_D'] - df_daily['close_D']) /
                    df_daily['close_D'].replace(0, np.nan)
                ).fillna(0)
        # --- 3. 处理 TrendlineFeature (事件型数据) ---
        # 假设 TrendlineFeature_D 已经通过 IndicatorService 加载，并且其索引是 end_date
        df_trendlines = all_dfs.get('trendline_feature_D')
        if df_trendlines is not None and not df_trendlines.empty:
            for idx, row in df_trendlines.iterrows():
                start_date = pd.to_datetime(row['start_date_D'], utc=True).normalize()
                end_date = pd.to_datetime(idx, utc=True).normalize() # 索引已经是end_date
                
                valid_dates = df_index[(df_index >= start_date) & (df_index <= end_date)]
                if not valid_dates.empty:
                    # 提取趋势线相关特征
                    trendline_features = {
                        f"TRENDLINE_SLOPE_{row['line_type_D'].upper()}_D": row.get('slope_D', np.nan),
                        f"TRENDLINE_CONVICTION_{row['line_type_D'].upper()}_D": row.get('touch_conviction_score_D', np.nan),
                        f"TRENDLINE_VALIDITY_{row['line_type_D'].upper()}_D": row.get('validity_score_D', np.nan),
                        f"TRENDLINE_INTERCEPT_{row['line_type_D'].upper()}_D": row.get('intercept_D', np.nan),
                    }
                    # 映射到每日数据，处理重叠时，最新结束的趋势线优先
                    for date in valid_dates:
                        for feature_name, value in trendline_features.items():
                            if pd.isna(df_geometric_features.loc[date, feature_name]) or date == end_date:
                                df_geometric_features.loc[date, feature_name] = value
        # 将所有几何特征合并到 df_daily
        df_daily = df_daily.join(df_geometric_features, how='left')
        # 对新合并的列进行前向填充，确保连续性
        new_geometric_cols = [col for col in df_geometric_features.columns if col not in df_daily.columns]
        if new_geometric_cols:
            df_daily[new_geometric_cols] = df_daily[new_geometric_cols].ffill()
        all_dfs[timeframe] = df_daily
        logger.info("几何形态特征计算完成。")
        return all_dfs






