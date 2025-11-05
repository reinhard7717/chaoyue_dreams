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
        【V2.2 性能优化版】元特征计算车间
        - 核心职责: 计算如Hurst指数、价格变异系数等描述价格序列统计特性的“元特征”。
        - 优化: 重构了Hurst指数的计算逻辑。通过直接在原始Series上使用.rolling().apply()并设置min_periods，
                避免了.dropna()和.reindex()的开销，显著提升了计算效率并减少了内存占用。
        """
        timeframe = 'D' # 元特征通常在日线级别计算
        if timeframe not in all_dfs:
            return all_dfs
        df = all_dfs[timeframe]
        suffix = f"_{timeframe}"
        # --- 1. 计算Hurst指数：衡量时间序列的长期记忆性 ---
        # Hurst > 0.5: 趋势性（涨的后面更可能涨，跌的后面更可能跌）
        # Hurst < 0.5: 均值回归性（涨的后面更可能跌，跌的后面更可能涨）
        # Hurst = 0.5: 随机游走
        hurst_window = 120
        hurst_col = f'hurst_{hurst_window}d{suffix}'
        close_col_suffixed = f'close{suffix}'
        if close_col_suffixed in df.columns and hurst_col not in df.columns:
            try:
                source_series = df[close_col_suffixed]
                # 兼容源数据可能为DataFrame的情况
                if isinstance(source_series, pd.DataFrame):
                    source_series = source_series.iloc[:, 0]
                # 优化Hurst计算逻辑
                # 先检查数据量是否足够进行至少一次计算
                if len(source_series.dropna()) >= hurst_window:
                    # 直接在原始Series上进行滚动计算，避免了创建中间Series(.dropna())和昂贵的reindex操作
                    # min_periods=hurst_window 确保只在窗口数据完整时才调用函数，行为与原逻辑一致
                    # raw=True 将numpy数组传递给apply函数，获得最佳性能
                    hurst_values = source_series.rolling(window=hurst_window, min_periods=hurst_window).apply(hurst_exponent, raw=True)
                    df[hurst_col] = hurst_values
                else:
                    # 如果数据量不足，则整列填充为NaN
                    df[hurst_col] = np.nan
            except Exception as e:
                logger.error(f"赫斯特指数计算过程中发生未知错误: {e}")
                df[hurst_col] = np.nan
        # --- 2. 计算价格变异系数(CV)：衡量价格的相对波动程度 ---
        cv_window = 60
        cv_col = f'price_cv_{cv_window}d{suffix}'
        if close_col_suffixed in df.columns and cv_col not in df.columns:
            source_series = df[close_col_suffixed]
            if isinstance(source_series, pd.DataFrame):
                source_series = source_series.iloc[:, 0]
            # 使用向量化操作计算：标准差 / 均值
            price_mean = source_series.rolling(cv_window).mean()
            price_std = source_series.rolling(cv_window).std()
            df[cv_col] = price_std / (price_mean + 1e-9) # 加一个极小值防止除以零
        # --- 3. 计算多空能量比：下方支撑 / 上方压力 ---
        energy_col = f'energy_ratio{suffix}'
        support_col = f'support_below{suffix}'
        pressure_col = f'pressure_above{suffix}'
        if support_col in df.columns and pressure_col in df.columns and energy_col not in df.columns:
            df[energy_col] = df[support_col] / (df[pressure_col] + 1e-6)
        all_dfs[timeframe] = df
        return all_dfs

    async def calculate_pattern_recognition_signals(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V2.1 注释优化版】高级模式识别信号生产线
        - 核心职责: 融合多种技术指标和资金流数据，识别市场中常见的关键模式，如盘整、突破、吸筹、派发等。
        - 优化: 代码已为最优向量化实现，本次增加大量分步注释，详细解释每种模式的判断逻辑。
        """
        timeframe = 'D' # 模式识别主要在日线级别进行
        if timeframe not in all_dfs:
            return all_dfs
        df = all_dfs[timeframe]
        # 定义并检查所有必需的输入列
        required_cols = [
            'high_D', 'low_D', 'close_D', 'volume_D', 'pct_change_D', 'VOL_MA_21_D',
            'BBW_21_2.0_D', 'ATR_14_D', 'MA_CONV_CV_SHORT_D', 'CMF_21_D',
            'VPA_EFFICIENCY_D', 'main_force_net_flow_consensus_D',
            'flow_divergence_mf_vs_retail_D', 'concentration_90pct_D',
            'winner_profit_margin_D', 'dynamic_consolidation_high_D', 'dynamic_consolidation_low_D'
        ]
        # 动态查找ADX列，因为其周期参数可能变化
        adx_col = next((col for col in df.columns if col.startswith('ADX_')), None)
        if adx_col:
            required_cols.append(adx_col)
        else:
            logger.warning("未找到 ADX 列，盘整识别的准确性会受影响。")
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logger.warning(f"高级模式识别生产线缺少关键数据: {missing}，模块已跳过！")
            return all_dfs
        # --- 1. 定义“盘整状态” (is_consolidation_D) ---
        # 盘整通常意味着波动性收缩和趋势不明。
        # 条件a: 波动性收缩。布林带宽度(BBW)或真实波幅(ATR)处于过去60天的较低水平(20%分位数以下)。
        bbw_quantile = df['BBW_21_2.0_D'].rolling(window=60, min_periods=20).quantile(0.20)
        atr_quantile = df['ATR_14_D'].rolling(window=60, min_periods=20).quantile(0.20)
        cond_low_volatility = (df['BBW_21_2.0_D'] < bbw_quantile) | (df['ATR_14_D'] < atr_quantile)
        # 条件b: 趋势不明显。ADX指标低于25，或者短期均线高度收敛(变异系数小于0.01)。
        cond_no_trend = (df[adx_col] < 25) if adx_col else pd.Series(True, index=df.index)
        cond_ma_converged = df['MA_CONV_CV_SHORT_D'] < 0.01
        is_consolidation = cond_low_volatility & (cond_no_trend | cond_ma_converged)
        df['is_consolidation_D'] = is_consolidation
        # --- 2. 定义“向上突破” (is_breakthrough_D) ---
        # 突破=前期处于盘整 + 价格突破盘整上轨 + 成交量确认 + 资金效率确认 + 资金流入确认
        was_consolidating = df['is_consolidation_D'].shift(1).fillna(False)
        price_break_box = df['close_D'] > df['dynamic_consolidation_high_D'].shift(1)
        volume_confirms = df['volume_D'] > df['VOL_MA_21_D'] * 1.2
        vpa_confirms = df['VPA_EFFICIENCY_D'] > 0.5
        money_flow_confirms = (df['CMF_21_D'] > 0.05) & (df['main_force_net_flow_consensus_D'] > 0)
        is_breakthrough = was_consolidating & price_break_box & volume_confirms & vpa_confirms & money_flow_confirms
        df['is_breakthrough_D'] = is_breakthrough
        # --- 3. 定义“向下跌破” (is_breakdown_D) ---
        # 跌破=前期处于盘整 + 价格跌破盘整下轨 + 成交量确认
        price_breakdown_box = df['close_D'] < df['dynamic_consolidation_low_D'].shift(1)
        is_breakdown = was_consolidating & price_breakdown_box & volume_confirms
        df['is_breakdown_D'] = is_breakdown
        # --- 4. 定义“吸筹阶段” (is_accumulation_D) ---
        # 吸筹=处于盘整状态 + (出现资金背离 或 筹码持续集中)
        # 条件a: 资金背离。连续3天出现主力买、散户卖的情况。
        cond_accumulation_flow = (df['flow_divergence_mf_vs_retail_D'] > 0.1).rolling(window=3).sum() == 3
        # 条件b: 筹码集中。90%筹码集中度的斜率在过去5天中至少有3天为正。
        concentration_slope = df['concentration_90pct_D'].diff()
        cond_concentration_increase = (concentration_slope > 0).rolling(window=5).sum() >= 3
        df['is_accumulation_D'] = is_consolidation & (cond_accumulation_flow | cond_concentration_increase)
        # --- 5. 定义“派发阶段” (is_distribution_D) ---
        # 派发=顶部放量滞涨 或 盘整中出现资金派发迹象
        # 条件a: 顶部派发特征。放巨量 + (价格滞涨 或 VPA效率极低) + 获利盘丰厚
        high_volume = df['volume_D'] > df['VOL_MA_21_D'] * 2.0
        stagnant_price = df['pct_change_D'].abs() < 0.01
        high_winner_margin = df['winner_profit_margin_D'] > 30
        low_vpa_efficiency = df['VPA_EFFICIENCY_D'] < 0.1
        dist_at_top = high_volume & (stagnant_price | low_vpa_efficiency) & high_winner_margin
        # 条件b: 盘整中派发特征。连续3天出现主力卖、散户买的情况。
        cond_distribution_flow = (df['flow_divergence_mf_vs_retail_D'] < -0.1).rolling(window=3).sum() == 3
        dist_in_consolidation = is_consolidation & cond_distribution_flow
        df['is_distribution_D'] = dist_at_top | dist_in_consolidation
        # --- 清理并格式化输出 ---
        pattern_cols = ['is_consolidation_D', 'is_breakthrough_D', 'is_breakdown_D', 'is_accumulation_D', 'is_distribution_D']
        for col in pattern_cols:
            if col in df.columns:
                df[col] = df[col].fillna(False).astype(bool)
        all_dfs[timeframe] = df
        return all_dfs

    async def calculate_ma_convergence(self, all_dfs: Dict[str, pd.DataFrame], params: dict) -> Dict[str, pd.DataFrame]:
        """
        【V1.0 新增】计算均线粘合度 (MA Convergence)。
        - 核心职责: 从 indicator_calculate_services.py 移入，作为特征工程的一部分。
        """
        if not params.get('enabled', False):
            return all_dfs
        
        for conv_config in params.get('configs', []):
            for timeframe in conv_config.get('apply_on', []):
                if timeframe not in all_dfs or all_dfs[timeframe].empty:
                    continue
                
                df = all_dfs[timeframe]
                periods = conv_config.get('periods', [])
                output_col = conv_config.get('output_column_name')
                
                ma_cols = [f"EMA_{p}_{timeframe}" for p in periods]
                if all(col in df.columns for col in ma_cols):
                    ma_df = df[ma_cols]
                    ma_std = ma_df.std(axis=1)
                    ma_mean = ma_df.mean(axis=1)
                    convergence_cv = ma_std / (ma_mean + 1e-9)
                    df[output_col] = convergence_cv
                else:
                    missing = [col for col in ma_cols if col not in df.columns]
                    logger.warning(f"计算均线粘合度 '{output_col}' 失败：缺少均线列 {missing}")
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
        【V1.2 · 职责净化修复版】形态增强信号编排器
        - 核心修复: 移除了对 breakout_quality_score 的计算调用。该指标的计算已归位到 IndicatorService 的主流程中，以解决列名后缀不匹配的问题。
        - 核心职责: 根据配置，调用计算器中新增的、依赖分钟数据的“超级原子信号”算法，并将其集成到日线数据中。
        """
        # [代码修改开始]
        params = config.get('feature_engineering_params', {}).get('indicators', {}).get('pattern_enhancement_signals', {})
        if not params.get('enabled', False):
            return all_dfs
        df_daily = all_dfs.get('D')
        if df_daily is None or df_daily.empty:
            return all_dfs
        minute_tf = params.get('minute_level_tf', '60')
        df_minute = all_dfs.get(minute_tf)
        tasks = []
        # 任务1: 日内VWAP偏离指数
        vwap_params = params.get('intraday_vwap_divergence', {})
        if vwap_params.get('enabled') and df_minute is not None:
            tasks.append(calculator.calculate_intraday_vwap_divergence_index(df_minute))
        # 任务2: 对手盘衰竭指数
        exhaustion_params = params.get('counterparty_exhaustion', {})
        if exhaustion_params.get('enabled') and df_minute is not None:
            tasks.append(calculator.calculate_counterparty_exhaustion_index(df_minute, exhaustion_params.get('efficiency_window', 21)))
        # 任务3: 突破质量分 (已从此方法中移除，其计算逻辑已整合到上游的 IndicatorService 主流程中)
        if not tasks:
            return all_dfs
        results = await asyncio.gather(*tasks)
        for res_df in results:
            if res_df is not None and not res_df.empty:
                # 将结果的索引标准化，以确保能与日线数据正确合并
                res_df.index = pd.to_datetime(res_df.index, utc=True).normalize()
                df_daily = df_daily.join(res_df, how='left')
        # 对新合并的列进行前向填充，保证数据连续性
        new_cols = [col for res_df in results if res_df is not None for col in res_df.columns]
        if new_cols: # 增加判断，只有在有新列时才执行填充
            df_daily[new_cols] = df_daily[new_cols].ffill()
        all_dfs['D'] = df_daily
        logger.info("分钟级形态增强信号计算完成并已集成。")
        return all_dfs
        # [代码修改结束]












