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
        【V3.0 混合计算版】
        - 核心升级: 实现智能跳过逻辑。在计算斜率前，会检查目标列是否已存在。
        """
        slope_params = config.get('feature_engineering_params', {}).get('slope_params', {})
        if not slope_params.get('enabled', False):
            return all_dfs
        series_to_slope = slope_params.get('series_to_slope', {})
        if not series_to_slope:
            return all_dfs
        for col_pattern, lookbacks in series_to_slope.items():
            if "说明" in col_pattern: continue
            try:
                timeframe = col_pattern.split('_')[-1]
                if timeframe.upper() not in ['D', 'W', 'M'] and not timeframe.isdigit():
                    timeframe = 'D'
            except IndexError:
                continue
            if timeframe not in all_dfs or all_dfs[timeframe] is None:
                continue
            df = all_dfs[timeframe]
            if col_pattern not in df.columns:
                continue
            source_series = df[col_pattern].astype(float)
            for lookback in lookbacks:
                slope_col_name = f'SLOPE_{lookback}_{col_pattern}'
                if slope_col_name in df.columns:
                    continue
                min_p = max(2, lookback // 2)
                linreg_result = df.ta.linreg(close=source_series, length=lookback, min_periods=min_p, slope=True, intercept=False, r=False)
                slope_series = linreg_result if isinstance(linreg_result, pd.Series) else linreg_result.iloc[:, 0]
                df[slope_col_name] = slope_series.fillna(0)
            all_dfs[timeframe] = df
        return all_dfs

    async def calculate_all_accelerations(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V2.0 混合计算版】
        - 核心升级: 增加了智能跳过逻辑。
        """
        accel_params = config.get('feature_engineering_params', {}).get('accel_params', {})
        if not accel_params.get('enabled', False):
            return all_dfs
        series_to_accel = accel_params.get('series_to_accel', {})
        if not series_to_accel:
            return all_dfs
        for base_col_name, periods in series_to_accel.items():
            if "说明" in base_col_name: continue
            timeframe = base_col_name.split('_')[-1]
            if timeframe not in all_dfs or all_dfs[timeframe] is None:
                continue
            df = all_dfs[timeframe]
            for period in periods:
                slope_col_name = f'SLOPE_{period}_{base_col_name}'
                if slope_col_name not in df.columns:
                    continue
                accel_col_name = f'ACCEL_{period}_{base_col_name}'
                if accel_col_name in df.columns:
                    continue
                source_series = df[slope_col_name]
                min_p = max(2, period // 2)
                accel_linreg_result = df.ta.linreg(close=source_series, length=period, min_periods=min_p, slope=True, intercept=False, r=False)
                accel_series = accel_linreg_result if isinstance(accel_linreg_result, pd.Series) else accel_linreg_result.iloc[:, 0]
                df[accel_col_name] = accel_series.fillna(0)
        return all_dfs

    async def calculate_vpa_features(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V1.0 新增】VPA效率指标生产线
        - 核心职责: 计算全新的自定义指标 VPA_EFFICIENCY_D (资金攻击效率)。
        """
        timeframe = 'D'
        if timeframe not in all_dfs:
            return all_dfs
        df = all_dfs[timeframe]
        required_cols = ['pct_change_D', 'volume_D', 'VOL_MA_21_D']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logger.warning(f"VPA效率生产线缺少关键数据: {missing}，模块已跳过！")
            return all_dfs
        volume_ratio = df['volume_D'] / df['VOL_MA_21_D'].replace(0, np.nan)
        vpa_efficiency = df['pct_change_D'] / volume_ratio.replace(0, np.nan)
        df['VPA_EFFICIENCY_D'] = vpa_efficiency.replace([np.inf, -np.inf], np.nan).fillna(0)
        all_dfs[timeframe] = df
        return all_dfs

    async def calculate_meta_features(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V2.1 健壮性修复版】元特征计算车间
        """
        timeframe = 'D'
        if timeframe not in all_dfs:
            return all_dfs
        df = all_dfs[timeframe]
        suffix = f"_{timeframe}"
        hurst_window = 120
        hurst_col = f'hurst_{hurst_window}d{suffix}'
        close_col_suffixed = f'close{suffix}'
        if close_col_suffixed in df.columns and hurst_col not in df.columns:
            try:
                source_series = df[close_col_suffixed]
                if isinstance(source_series, pd.DataFrame):
                    source_series = source_series.iloc[:, 0]
                close_series_for_hurst = source_series.dropna()
                if len(close_series_for_hurst) >= hurst_window:
                    hurst_values = close_series_for_hurst.rolling(hurst_window).apply(hurst_exponent, raw=True)
                    df[hurst_col] = hurst_values.reindex(df.index)
                else:
                    df[hurst_col] = np.nan
            except Exception as e:
                logger.error(f"赫斯特指数计算过程中发生未知错误: {e}")
                df[hurst_col] = np.nan
        cv_window = 60
        cv_col = f'price_cv_{cv_window}d{suffix}'
        if close_col_suffixed in df.columns and cv_col not in df.columns:
            source_series = df[close_col_suffixed]
            if isinstance(source_series, pd.DataFrame):
                source_series = source_series.iloc[:, 0]
            price_mean = source_series.rolling(cv_window).mean()
            price_std = source_series.rolling(cv_window).std()
            df[cv_col] = price_std / (price_mean + 1e-9)
        energy_col = f'energy_ratio{suffix}'
        support_col = f'support_below{suffix}'
        pressure_col = f'pressure_above{suffix}'
        if support_col in df.columns and pressure_col in df.columns and energy_col not in df.columns:
            df[energy_col] = df[support_col] / (df[pressure_col] + 1e-6)
        all_dfs[timeframe] = df
        return all_dfs

    async def calculate_pattern_recognition_signals(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V2.0 多因子共振版】高级模式识别信号生产线
        """
        timeframe = 'D'
        if timeframe not in all_dfs:
            return all_dfs
        df = all_dfs[timeframe]
        required_cols = [
            'high_D', 'low_D', 'close_D', 'volume_D', 'pct_change_D',
            'BBW_21_2.0_D', 'ATR_14_D', 'MA_CONV_CV_SHORT_D', 'CMF_21_D',
            'VPA_EFFICIENCY_D', 'main_force_net_flow_consensus_D',
            'flow_divergence_mf_vs_retail_D', 'concentration_90pct_D',
            'winner_profit_margin_D', 'dynamic_consolidation_high_D', 'dynamic_consolidation_low_D'
        ]
        adx_col = next((col for col in df.columns if col.startswith('ADX_')), None)
        if adx_col:
            required_cols.append(adx_col)
        else:
            logger.warning("未找到 ADX 列，盘整识别的准确性会受影响。")
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logger.warning(f"高级模式识别生产线缺少关键数据: {missing}，模块已跳过！")
            return all_dfs
        bbw_quantile = df['BBW_21_2.0_D'].rolling(window=60, min_periods=20).quantile(0.20)
        atr_quantile = df['ATR_14_D'].rolling(window=60, min_periods=20).quantile(0.20)
        cond_low_volatility = (df['BBW_21_2.0_D'] < bbw_quantile) | (df['ATR_14_D'] < atr_quantile)
        cond_no_trend = (df[adx_col] < 25) if adx_col else pd.Series(True, index=df.index)
        cond_ma_converged = df['MA_CONV_CV_SHORT_D'] < 0.01
        is_consolidation = cond_low_volatility & (cond_no_trend | cond_ma_converged)
        df['is_consolidation_D'] = is_consolidation
        was_consolidating = df['is_consolidation_D'].shift(1).fillna(False)
        price_break_box = df['close_D'] > df['dynamic_consolidation_high_D'].shift(1)
        volume_confirms = df['volume_D'] > df['VOL_MA_21_D'] * 1.2
        vpa_confirms = df['VPA_EFFICIENCY_D'] > 0.5
        money_flow_confirms = (df['CMF_21_D'] > 0.05) & (df['main_force_net_flow_consensus_D'] > 0)
        is_breakthrough = was_consolidating & price_break_box & volume_confirms & vpa_confirms & money_flow_confirms
        df['is_breakthrough_D'] = is_breakthrough
        price_breakdown_box = df['close_D'] < df['dynamic_consolidation_low_D'].shift(1)
        is_breakdown = was_consolidating & price_breakdown_box & volume_confirms
        df['is_breakdown_D'] = is_breakdown
        cond_accumulation_flow = (df['flow_divergence_mf_vs_retail_D'] > 0.1).rolling(window=3).sum() == 3
        concentration_slope = df['concentration_90pct_D'].diff()
        cond_concentration_increase = (concentration_slope > 0).rolling(window=5).sum() >= 3
        df['is_accumulation_D'] = is_consolidation & (cond_accumulation_flow | cond_concentration_increase)
        high_volume = df['volume_D'] > df['VOL_MA_21_D'] * 2.0
        stagnant_price = df['pct_change_D'].abs() < 0.01
        high_winner_margin = df['winner_profit_margin_D'] > 30
        low_vpa_efficiency = df['VPA_EFFICIENCY_D'] < 0.1
        dist_at_top = high_volume & (stagnant_price | low_vpa_efficiency) & high_winner_margin
        cond_distribution_flow = (df['flow_divergence_mf_vs_retail_D'] < -0.1).rolling(window=3).sum() == 3
        dist_in_consolidation = is_consolidation & cond_distribution_flow
        df['is_distribution_D'] = dist_at_top | dist_in_consolidation
        pattern_cols = ['is_consolidation_D', 'is_breakthrough_D', 'is_breakdown_D', 'is_accumulation_D', 'is_distribution_D']
        for col in pattern_cols:
            if col in df.columns:
                df[col] = df[col].fillna(False).astype(bool)
        all_dfs[timeframe] = df
        return all_dfs
