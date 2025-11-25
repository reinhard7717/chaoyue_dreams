# services/geometric_pattern_service.py
import pandas as pd
import numpy as np
from numba import njit
import pandas_ta as ta
from itertools import combinations
import joblib # 用于加载机器学习模型
from pathlib import Path
from django.conf import settings
from stock_models.models import StockInfo
from stock_models.index import TradeCalendar
from utils.model_helpers import (
    get_daily_data_model_by_code,
    get_platform_feature_model_by_code,
    get_multi_timeframe_trendline_model_by_code, # 导入新模型辅助函数
    get_trendline_event_model_by_code, # 导入事件模型辅助函数
    get_advanced_chip_metrics_model_by_code, # 导入高级芯片指标模型辅助函数
    get_advanced_fund_flow_metrics_model_by_code, # 导入高级资金流指标模型辅助函数
    get_advanced_structural_metrics_model_by_code, # 导入高级结构性指标模型辅助函数
)

@njit
def _calculate_zigzag_numba(highs: np.ndarray, lows: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """
    【V2.15 · 逻辑修正版】修复了 Numba JIT 实现中因 cursor 回跳导致的无限循环BUG。
    - 核心修复: 当找到拐点时，不再将 cursor 重置为 pivot_idx + 1，而是让其继续前进 (cursor += 1)，
                 从根本上保证了循环的向前推进，杜绝了死循环的可能。
    """
    n = len(highs)
    if n < 2:
        return np.zeros(n, dtype=np.int8)
    zigzag = np.zeros(n, dtype=np.int8)
    initial_high = highs[0]
    initial_low = lows[0]
    trend = 0
    cursor = 1
    while cursor < n:
        if trend == 0:
            if highs[cursor] > initial_high:
                trend = 1
                zigzag[0] = -1
                break
            elif lows[cursor] < initial_low:
                trend = -1
                zigzag[0] = 1
                break
        cursor += 1
    if trend == 0:
        return zigzag
    pivot_idx = 0
    while cursor < n:
        if trend == 1:
            segment_highs = highs[pivot_idx : cursor + 1]
            current_high = np.max(segment_highs)
            current_high_idx_loc = pivot_idx + np.argmax(segment_highs)
            if (current_high - lows[cursor]) / current_high > threshold:
                zigzag[current_high_idx_loc] = 1
                trend = -1
                pivot_idx = current_high_idx_loc
                # 不再重置cursor，而是让其继续前进，防止回跳
                cursor += 1
            else:
                cursor += 1
        elif trend == -1:
            segment_lows = lows[pivot_idx : cursor + 1]
            current_low = np.min(segment_lows)
            current_low_idx_loc = pivot_idx + np.argmin(segment_lows)
            if (highs[cursor] - current_low) / current_low > threshold:
                zigzag[current_low_idx_loc] = -1
                trend = 1
                pivot_idx = current_low_idx_loc
                # 不再重置cursor，而是让其继续前进，防止回跳
                cursor += 1
            else:
                cursor += 1
    return zigzag

class GeometricPatternService:
    """
    【V2.1 · 动态演化分析版】
    - 核心职责: 识别几何结构，并分析其在多时间维度下的动态演化，生成结构性事件。
    - V2.1 升级: 引入趋势线矩阵、动态事件分析、旗形突破概率预测。
    """
    def __init__(self, stock_code: str, stock_instance: StockInfo):
        """
        【V2.65 · 全息旗形版】
        - 核心升级: 加载 `trend_follow_strategy.json` 配置文件中新增的
                     `flag_recognition` 原型定义。
        """
        import json
        import os
        self.stock_code = stock_code
        self.stock_instance = stock_instance
        self.stock_id = stock_instance.stock_code
        self.daily_model = get_daily_data_model_by_code(stock_code)
        self.platform_model = get_platform_feature_model_by_code(stock_code)
        self.mtt_model = get_multi_timeframe_trendline_model_by_code(stock_code)
        self.event_model = get_trendline_event_model_by_code(stock_code)
        self.flag_predictor_model = self._load_predictor_model()
        self.chip_metrics_model = get_advanced_chip_metrics_model_by_code(stock_code)
        self.fund_flow_metrics_model = get_advanced_fund_flow_metrics_model_by_code(stock_code)
        self.structural_metrics_model = get_advanced_structural_metrics_model_by_code(stock_code)
        self.fib_periods = [5, 8, 13, 21, 34, 55]
        self.long_term_period = max(self.fib_periods) if self.fib_periods else 55
        self.ultra_long_term_period = 233
        config_path = os.path.join(settings.BASE_DIR, 'config', 'trend_follow_strategy.json')
        strategy_config = {}
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                strategy_config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[{self.stock_code}] 加载策略配置文件失败: {e}")
        geometric_params = strategy_config.get("strategy_params", {}).get("trend_follow", {}).get("geometric_pattern_params", {})
        self.platform_archetypes = geometric_params.get("platform_recognition", {}).get("archetypes", [])
        # V2.65 加载旗形识别原型
        self.flag_archetypes = geometric_params.get("flag_recognition", {}).get("archetypes", [])

    @classmethod
    async def create(cls, stock_code: str):
        """
        【异步工厂方法】
        这是实例化服务的唯一入口。它负责处理所有异步I/O操作。
        """
        from asgiref.sync import sync_to_async
        # 使用 sync_to_async 安全地执行同步数据库查询
        stock_instance = await sync_to_async(StockInfo.objects.get)(stock_code=stock_code)
        # 使用获取到的实例来调用同步的构造函数
        return cls(stock_code=stock_code, stock_instance=stock_instance)

    def _load_predictor_model(self):
        # 辅助方法：加载预训练的旗形突破概率预测模型
        model_path = Path(settings.BASE_DIR) / 'ml_models' / 'flag_predictor.pkl'
        if model_path.exists():
            try:
                return joblib.load(model_path)
            except Exception as e:
                print(f"[{self.stock_code}] 加载旗形预测模型失败: {e}")
                return None
        print(f"[{self.stock_code}] 未找到旗形预测模型，将跳过概率预测。")
        return None

    def _prepare_enriched_dataframe(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """
        【V2.34 · 核心诊断探针版】准备一个包含所有高级指标的、信息增强的DataFrame。
        - 核心修改: 移除此方法内的所有旧探针，将诊断焦点集中到下游。
        """
        print(f"  -> [数据融合] 正在加载并整合高级指标...")
        chip_metrics_qs = self.chip_metrics_model.objects.filter(stock=self.stock_instance).values()
        fund_flow_metrics_qs = self.fund_flow_metrics_model.objects.filter(stock=self.stock_instance).values()
        structural_metrics_qs = self.structural_metrics_model.objects.filter(stock=self.stock_instance).values()
        df_chip = pd.DataFrame.from_records(chip_metrics_qs).rename(columns={'trade_time': 'trade_date'})
        df_fund = pd.DataFrame.from_records(fund_flow_metrics_qs).rename(columns={'trade_time': 'trade_date'})
        df_struct = pd.DataFrame.from_records(structural_metrics_qs).rename(columns={'trade_time': 'trade_date'})
        print(f"  -> [数据净化] 正在对加载的高级指标进行强制类型转换...")
        non_numeric_whitelist = ['stock_id']
        dataframes_to_process = {'chip': df_chip, 'fund': df_fund, 'struct': df_struct}
        for name, df in dataframes_to_process.items():
            if not df.empty:
                if 'trade_date' in df.columns:
                    df['trade_date'] = pd.to_datetime(df['trade_date'])
                for col in df.columns:
                    if col not in non_numeric_whitelist and col != 'trade_date':
                        if df[col].dtype == 'object':
                            df[col] = pd.to_numeric(df[col], errors='coerce')
        df_daily_reset = df_daily.reset_index().rename(columns={'trade_time': 'trade_date'})
        df_daily_reset['stock_id'] = self.stock_id
        enriched_df = df_daily_reset
        join_keys = ['stock_id', 'trade_date']
        for name, df_right in dataframes_to_process.items():
            if not df_right.empty:
                unique_cols_from_right = [col for col in df_right.columns if col not in enriched_df.columns]
                cols_to_merge_from_right = join_keys + unique_cols_from_right
                enriched_df = pd.merge(
                    enriched_df,
                    df_right[cols_to_merge_from_right],
                    on=join_keys,
                    how='left'
                )
        enriched_df = enriched_df.set_index('trade_date').sort_index()
        if enriched_df.empty or enriched_df.index.isnull().all():
            print(f"--- [数据融合失败] DataFrame为空或索引损坏 ---")
        print(f"  -> [数据融合] 全维度战场沙盘构建完成。")
        return enriched_df

    def calculate_and_save_all_patterns(self, data_dfs: dict, start_date_str: str = None):
        """
        【V2.54 · 残影修正版】执行所有几何形态的计算和存储。
        - V2.54 核心修复: 修正了因方法重命名（_predict_flag_breakout_probability -> _find_and_evaluate_flags）
                         而遗漏更新的调用点，根除了由此引发的 AttributeError。
        """
        print(f"[{self.stock_code}] [动态演化分析] 开始计算几何形态特征...")
        df_daily = data_dfs.get('daily_data')
        if df_daily is None or df_daily.empty:
            print(f"[{self.stock_code}] 日线数据为空，跳过计算。")
            return
        df_daily['trade_time'] = pd.to_datetime(df_daily['trade_time'])
        df_daily = df_daily.set_index('trade_time')
        df_daily.sort_index(inplace=True)
        cols_to_convert = ['high_qfq', 'low_qfq', 'close_qfq', 'open_qfq', 'vol']
        for col in cols_to_convert:
            if col in df_daily.columns:
                df_daily[col] = pd.to_numeric(df_daily[col], errors='coerce')
        df_daily.ta.atr(high='high_qfq', low='low_qfq', close='close_qfq', length=14, append=True, col_names=('ATR_14_D',))
        enriched_df = self._prepare_enriched_dataframe(df_daily)
        data_dfs['enriched_df'] = enriched_df
        if start_date_str:
            deleted_count, _ = self.platform_model.objects.filter(
                stock=self.stock_instance,
                start_date__gte=start_date_str
            ).delete()
            print(f"  -> [统一回滚] 平台特征删除 {deleted_count} 条。")
        self._calculate_and_save_platforms(enriched_df, data_dfs)
        self._calculate_and_save_trendline_matrix_and_events(df_daily, data_dfs, start_date_str=start_date_str)
        # V2.54 修正方法调用，使用新的方法名 _find_and_evaluate_flags
        flag_events = self._find_and_evaluate_flags(enriched_df, data_dfs)
        if start_date_str:
            self.event_model.objects.filter(
                stock=self.stock_instance,
                event_date__gte=start_date_str,
                event_type__startswith='FLAG_FORMED'
            ).delete()
        self._save_trendline_events_incrementally(flag_events)
        print(f"[{self.stock_code}] [动态演化分析] 几何形态特征计算完成。")

    def _calculate_and_save_platforms(self, enriched_df: pd.DataFrame, data_dfs: dict):
        """
        【V2.53 · 智能归一化版】应用升级后的斜率计算函数，为RSSlope关闭冗余归一化。
        - V2.53 核心升级: 在计算 RSSlope 时，调用斜率计算函数并传入 `normalize=False`，
                         确保参照系校准后的序列不再被错误地二次归一化。
        """
        import math
        print(f"  -> [V2.53 智能归一化] 启动...")
        if len(enriched_df) < 120:
            print("  -> 数据量不足(<120天)，跳过平台识别。")
            return
        if not self.platform_archetypes:
            print("  -> [配置缺失] 在配置文件中未找到平台原型定义，跳过平台识别。")
            return
        df_copy = enriched_df.copy()
        df_copy.ta.atr(length=14, append=True)
        index_df = data_dfs.get('index_df')
        if index_df is not None and not index_df.empty:
            df_copy = df_copy.join(index_df.add_suffix('_index'))
            if 'close_index' in df_copy.columns and df_copy['close_index'].notna().any():
                df_copy['rs'] = df_copy['close_qfq'] / df_copy['close_index']
                print("  -> [数据预处理] 相对强度(RS)数据已成功计算并合并。")
            else:
                print("  -> [数据预处理警告] 指数数据合并后无有效收盘价，无法计算RS。")
        else:
            print("  -> [数据预处理警告] 未提供指数数据(index_df)，相对强度相关指标将无法计算。")
        cols_to_drop = ['open', 'high', 'low', 'close']
        df_copy.drop(columns=[col for col in cols_to_drop if col in df_copy.columns], inplace=True)
        cols_to_convert = ['high_qfq', 'low_qfq', 'close_qfq', 'open_qfq', 'vol']
        for col in cols_to_convert:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        required_cols = ['high_qfq', 'low_qfq', 'close_qfq']
        if not all(col in df_copy.columns for col in required_cols):
            print(f"\n  -> [诊断失败] 输入的DataFrame缺少核心计算列。需要: {required_cols}。任务终止。")
            return
        df_copy.ta.adx(high='high_qfq', low='low_qfq', close='close_qfq', length=14, append=True)
        bbu_col, bbl_col, bbm_col, bbw_col = 'BBU_21_2.0', 'BBL_21_2.0', 'BBM_21_2.0', 'BBW_21_2.0'
        bbands_results = ta.bbands(close=df_copy['close_qfq'], length=21, std=2.0, append=False)
        if bbands_results is not None and not bbands_results.empty:
            df_copy = df_copy.join(bbands_results)
        if all(col in df_copy.columns for col in [bbu_col, bbl_col, bbm_col]):
            df_copy[bbw_col] = (df_copy[bbu_col] - df_copy[bbl_col]) / df_copy[bbm_col]
        else:
            print(f"  -> [计算失败] 布林带基础指标未能生成。任务终止。")
            return
        df_copy = self._calculate_breakout_readiness(df_copy)
        print(f"\n{'='*20} 阶段一: 多准则通用平台识别 {'='*20}")
        combined_is_potential = pd.Series(False, index=df_copy.index)
        print("  -> 遍历所有原型，通过逻辑'或'合并识别准则以扩大扫描范围...")
        for archetype in self.platform_archetypes:
            archetype_name = archetype.get('name', 'UNKNOWN')
            adx_threshold = archetype.get('adx_threshold', 25)
            bbw_quantile = archetype.get('bbw_quantile', 0.25)
            print(f"     - 应用原型 [{archetype_name}] 的准则 (ADX < {adx_threshold}, BBW < {bbw_quantile}分位数)")
            is_low_trend = df_copy['ADX_14'] < adx_threshold
            bbw_rolling_quantile = df_copy[bbw_col].rolling(120, min_periods=60).quantile(bbw_quantile)
            is_low_volatility = df_copy[bbw_col] < bbw_rolling_quantile
            archetype_potential = is_low_trend | is_low_volatility
            combined_is_potential = combined_is_potential | archetype_potential
        df_copy['is_potential_platform'] = combined_is_potential.astype(int)
        baseline_archetype = self.platform_archetypes[0]
        potential_threshold = baseline_archetype.get('potential_threshold', 0.6)
        potential_window = baseline_archetype.get('potential_window', 20)
        df_copy['platform_potential_score'] = df_copy['is_potential_platform'].rolling(window=potential_window, min_periods=potential_window//2).mean()
        score_series = df_copy['platform_potential_score']
        entering_platform = (score_series > potential_threshold) & (score_series.shift(1) <= potential_threshold)
        exiting_platform = (score_series < potential_threshold) & (score_series.shift(1) >= potential_threshold)
        platform_start_dates = df_copy[entering_platform.fillna(False)].index
        platform_end_dates = df_copy[exiting_platform.fillna(False)].index
        raw_candidates = []
        for start_date in platform_start_dates:
            possible_end_dates = platform_end_dates[platform_end_dates > start_date]
            if not possible_end_dates.empty:
                raw_candidates.append((start_date, possible_end_dates[0]))
        print(f"  -> 识别完成，共发现 {len(raw_candidates)} 个原始候选平台。")
        print(f"\n{'='*20} 阶段二: 多原型分类与博弈验证 {'='*20}")
        platforms_to_save = []
        saved_start_dates = set()
        minute_map = data_dfs.get("stock_minute_data_map", {})
        tick_map = data_dfs.get("stock_tick_data_map", {})
        realtime_map = data_dfs.get("stock_realtime_data_map", {})
        for start_date, end_date in raw_candidates:
            if start_date in saved_start_dates:
                continue
            group = df_copy.loc[start_date:end_date]
            platform_high = group['high_qfq'].max()
            platform_low = group['low_qfq'].min()
            if platform_low == 0: continue
            price_range_pct = (platform_high - platform_low) / platform_low
            print(f"\n  -> [验证探针] 候选平台: {start_date.date()} -> {end_date.date()} (持续 {len(group)} 天, 振幅 {price_range_pct:.2%})")
            # [代码修改] V2.53 在计算rss时，关闭内部归一化
            rebased_rs = group['rs'] / group['rs'].iloc[0] if 'rs' in group and not group.empty and group['rs'].iloc[0] != 0 else pd.Series()
            metrics = {
                'vps': self._calculate_volume_profile_skewness(group),
                'vts': self._calculate_linear_regression_slope(group['vol']), # 保持默认归一化
                'vcr': self._calculate_volatility_contraction_ratio(group),
                'pk': self._calculate_price_kurtosis(group),
                'rss': self._calculate_linear_regression_slope(rebased_rs, normalize=False) if not rebased_rs.empty else 0.0
            }
            print(f"     - [博弈指标] VPSkew: {metrics['vps']:.2f}, VolSlope: {metrics['vts']:.2f}, VolatilityContract: {metrics['vcr']:.2f}, PriceKurtosis: {metrics['pk']:.2f}, RSSlope: {metrics['rss']:.3f}")
            best_archetype = None
            best_fit_score = -1.0
            print("     - [拟合度评估] 开始计算与所有原型的拟合优度...")
            for archetype in self.platform_archetypes:
                archetype_name = archetype.get('name', 'UNKNOWN')
                min_duration = archetype.get('min_duration', 0)
                max_range_pct = archetype.get('max_range_pct', 1.0)
                if len(group) >= min_duration and price_range_pct <= max_range_pct:
                    fit_score = self._calculate_goodness_of_fit(metrics, archetype)
                    print(f"       - 与原型 [{archetype_name}] 的拟合度: {fit_score:.2f}")
                    if fit_score > best_fit_score:
                        best_fit_score = fit_score
                        best_archetype = archetype
                else:
                    print(f"       - 与原型 [{archetype_name}] 的几何门槛不符，跳过。")
            if best_archetype:
                archetype_name = best_archetype.get('name', 'UNKNOWN')
                print(f"     - [BEST FIT & ACCEPTED] 最佳匹配原型为 [{archetype_name}] (拟合度: {best_fit_score:.2f})，将被量化。")
                conviction_score = self._calculate_platform_conviction_score(group)
                print(f"     - [信念评估] 平台内在信念评分为: {conviction_score:.2f}")
                character, score_val = self._assess_platform_character(group)
                platform_minutes_dfs = [minute_map[d.date()] for d in group.index if d.date() in minute_map]
                precise_vpoc = None
                if platform_minutes_dfs:
                    platform_minutes_df = pd.concat(platform_minutes_dfs)
                    if not platform_minutes_df.empty and 'volume' in platform_minutes_df.columns and platform_minutes_df['volume'].sum() > 0:
                        precise_vpoc = np.average(platform_minutes_df['close'], weights=platform_minutes_df['volume'])
                internal_ofi_sum, internal_total_amount = 0, 0
                for d in group.index:
                    if d.date() in tick_map:
                        ofi, amount = self._calculate_daily_ofi_from_ticks(tick_map[d.date()])
                        internal_ofi_sum += ofi * df_copy.loc[d, 'close_qfq']
                        internal_total_amount += amount
                internal_accumulation_intensity = (internal_ofi_sum / internal_total_amount) * 100 if internal_total_amount > 0 else 0.0
                breakout_quality_score = None
                next_trade_date = TradeCalendar.get_next_trade_date(end_date.date())
                if next_trade_date and pd.to_datetime(next_trade_date) in df_copy.index:
                    breakout_day_close = df_copy.loc[pd.to_datetime(next_trade_date), 'close_qfq']
                    if breakout_day_close > platform_high:
                        ofi_score, momentum_score = 0.0, 0.0
                        if next_trade_date in tick_map:
                            ofi, _ = self._calculate_daily_ofi_from_ticks(tick_map[next_trade_date])
                            breakout_vol = df_copy.loc[pd.to_datetime(next_trade_date), 'vol'] * 100
                            if breakout_vol > 0: ofi_score = np.clip(ofi / breakout_vol, -1, 1)
                        if next_trade_date in realtime_map:
                            momentum_score = self._calculate_breakout_momentum_from_realtime(realtime_map[next_trade_date])
                        breakout_quality_score = (ofi_score * 0.6) + (momentum_score * 0.4)
                breakout_readiness = group['breakout_readiness_score'].iloc[-1] if 'breakout_readiness_score' in group else None
                if breakout_readiness is not None and math.isnan(breakout_readiness):
                    breakout_readiness = None
                platform_data = {
                    'stock': self.stock_instance, 'start_date': start_date.date(), 'end_date': end_date.date(),
                    'duration': len(group), 'high': platform_high, 'low': platform_low,
                    'vpoc': np.average(group['close_qfq'], weights=group['vol']),
                    'total_volume': group['vol'].sum() * 100, 'quality_score': (score_val + 100) / 200,
                    'precise_vpoc': precise_vpoc, 'internal_accumulation_intensity': internal_accumulation_intensity,
                    'breakout_quality_score': breakout_quality_score,
                    'breakout_readiness_score': breakout_readiness,
                    'platform_character': character, 'character_score': score_val,
                    'platform_archetype': archetype_name,
                    'goodness_of_fit_score': best_fit_score,
                    'platform_conviction_score': conviction_score,
                }
                platforms_to_save.append(platform_data)
                saved_start_dates.add(start_date)
            else:
                print(f"     - [REJECTED] 未找到任何在几何门槛内且有意义的匹配原型。")
        found_count = len(platforms_to_save)
        print(f"\n  -> [V2.53 智能归一化] 扫描完成，共发现 {found_count} 个有效平台。")
        if found_count > 0:
            print(f"  -> [V2.53 智能归一化] 正在将 {found_count} 个平台存入数据库...")
            for data in platforms_to_save:
                self.platform_model.objects.update_or_create(
                    stock=data['stock'], start_date=data['start_date'], defaults=data
                )

    def _calculate_breakout_readiness(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V2.49 · 计算加固版】计算“突破准备度”评分。
        该评分融合了五大核心支柱，用于量化一个平台是否处于“高能待发”的临界状态。
        - V2.49 升级: 加固了“内部强势”支柱的计算逻辑，通过预处理分母为0的情况，
                      从源头上避免了除零错误导致的 `inf` 或 `nan`。
        """
        print("  -> [发射倒计时] 正在计算'突破准备度'评分...")
        df_copy = df.copy()
        # 支柱一: 波动率压缩 (Volatility Compression) - 权重 30%
        bbw_col = 'BBW_21_2.0'
        if bbw_col in df_copy.columns:
            score_volatility = (1 - df_copy[bbw_col].rolling(120, min_periods=60).rank(pct=True)) * 100
        else:
            score_volatility = pd.Series(50, index=df_copy.index)
        # 支柱二: 成交量萎缩 (Volume Atrophy) - 权重 30%
        vol_ma_short = df_copy['vol'].rolling(5).mean()
        vol_ma_long = df_copy['vol'].rolling(55).mean()
        volume_ratio = vol_ma_short / vol_ma_long
        score_volume = (1 - np.clip(volume_ratio, 0, 2)) * 50 + 50
        # 支柱三: 价格收敛 (Price Coiling) - 权重 15%
        atr_col = 'ATRr_14'
        if atr_col in df_copy.columns:
            score_coiling = (1 - df_copy[atr_col].rolling(120, min_periods=60).rank(pct=True)) * 100
        else:
            score_coiling = pd.Series(50, index=df_copy.index)
        # 支柱四: 内部强势 (Internal Strength) - 权重 10%
        # V2.49 加固计算逻辑，预处理分母为0的情况，防止除零错误
        daily_range = df_copy['high_qfq'] - df_copy['low_qfq']
        # 将daily_range为0的地方替换为nan，这样除法会得到nan，然后被fillna安全处理
        internal_strength = ((df_copy['close_qfq'] - df_copy['low_qfq']) / daily_range.replace(0, np.nan)).fillna(0.5)
        score_internal = internal_strength.rolling(5).mean().fillna(0.5) * 100
        # 支柱五: 相对强度 (Relative Strength) - 权重 15%
        if 'rs' in df_copy.columns and df_copy['rs'].notna().any():
            rs_slope = df_copy['rs'].rolling(5).apply(lambda x: np.polyfit(np.arange(len(x)), x.dropna(), 1)[0] if len(x.dropna()) > 1 else 0.0, raw=False)
            score_rs = rs_slope.rolling(120, min_periods=60).rank(pct=True).fillna(0.5) * 100
        else:
            score_rs = pd.Series(50, index=df_copy.index)
        # 融合总分
        df_copy['breakout_readiness_score'] = (
            score_volatility * 0.30 +
            score_volume * 0.30 +
            score_coiling * 0.15 +
            score_internal * 0.10 +
            score_rs * 0.15
        )
        return df_copy

    def _calculate_daily_ofi_from_ticks(self, df_tick: pd.DataFrame) -> (float, float):
        """
        【V2.44 · Tick数据格式再适配版】
        根据单日tick数据计算订单流失衡（OFI）。
        - 核心修复: 根据数据库实际存储的tick数据格式，将买卖方向的判断
                     条件从中文'买盘'/'卖盘'修正为大写字母 'B'/'S'，
                     确保OFI计算的准确性。
        """
        if df_tick is None or df_tick.empty:
            return 0.0, 0.0
        try:
            # V2.44 适配 'B'/'S'/'M' 的tick数据格式
            buy_vol = df_tick[df_tick['type'] == 'B']['volume'].sum()
            sell_vol = df_tick[df_tick['type'] == 'S']['volume'].sum()
            total_amount = df_tick['amount'].sum()
            ofi = buy_vol - sell_vol
            return ofi, total_amount
        except KeyError as e:
            print(f"  -> [Tick计算警告] 计算OFI时发生KeyError: {e}。可能是tick数据缺少'type'或'volume'列。返回0。")
            return 0.0, 0.0

    def _calculate_and_save_trendline_matrix_and_events(self, df_daily: pd.DataFrame, data_dfs: dict, start_date_str: str = None):
        """
        【V2.57 · 高斯信念归一化版】为趋势线矩阵计算引入基于高斯分布的动态自适应统计量。
        - V2.57 核心升级: 预处理中心不再计算脆弱的分位数，而是计算更稳健的滚动均值(mean)
                         和滚动标准差(std)。这些核心统计量将作为下游“高斯归一化”引擎的
                         历史参照系，实现从分段线性到概率平滑的思维跃迁。
        """
        print(f"  -> [增量回溯引擎] 开始检查并计算新的趋势线矩阵...")
        df_daily = df_daily.sort_index(ascending=True)
        if 'enriched_df' not in data_dfs:
             data_dfs['enriched_df'] = self._prepare_enriched_dataframe(df_daily)
        enriched_df = data_dfs['enriched_df']
        print(f"    -> [高斯校准] 正在为三大支柱生成动态统计参照系 (mean, std)...")
        # 1. 力量(资金)支柱: 计算每日资金流占比
        if 'main_force_net_flow_calibrated' in enriched_df.columns and 'amount' in enriched_df.columns:
            enriched_df['daily_flow_ratio'] = (enriched_df['main_force_net_flow_calibrated'] * 10000) / enriched_df['amount'].replace(0, np.nan)
        # 2. 结构(筹码)支柱: 计算主峰稳固度的5日滚动斜率
        if 'dominant_peak_solidity' in enriched_df.columns:
            enriched_df['solidity_slope_5d'] = enriched_df['dominant_peak_solidity'].rolling(5).apply(
                lambda x: np.polyfit(np.arange(len(x)), x.dropna(), 1)[0] if len(x.dropna()) > 1 else 0.0, raw=False
            )
        # 3. 行为(效率)支柱: 计算趋势效率的5日滚动均值
        if 'trend_efficiency_ratio' in enriched_df.columns:
            enriched_df['efficiency_avg_5d'] = enriched_df['trend_efficiency_ratio'].rolling(5).mean()
        # V2.57 从计算分位数改为计算均值和标准差
        # 4. 为三大支柱生成滚动统计量 (使用250天窗口，约1年)
        stat_window = 250
        metrics_to_calibrate = {
            'flow': 'daily_flow_ratio',
            'structure': 'solidity_slope_5d',
            'behavior': 'efficiency_avg_5d'
        }
        for key, metric_col in metrics_to_calibrate.items():
            if metric_col in enriched_df.columns:
                enriched_df[f'{key}_mean'] = enriched_df[metric_col].rolling(stat_window, min_periods=30).mean()
                enriched_df[f'{key}_std'] = enriched_df[metric_col].rolling(stat_window, min_periods=30).std()
        data_dfs['enriched_df'] = enriched_df
        print(f"    -> [高斯校准] 动态统计参照系生成完毕。")
        start_process_date = self._initialize_incremental_context(start_date_str)
        if start_process_date is None:
            start_process_date = df_daily.index.min()
        df_to_process = df_daily[df_daily.index >= start_process_date]
        if df_to_process.empty:
            print("  -> [增量回溯引擎] 数据已是最新，无需计算。")
            return
        print(f"  -> [增量回溯引擎] 将处理从 {df_to_process.index.min().date()} 到 {df_to_process.index.max().date()} 的 {len(df_to_process)} 个新交易日。")
        new_matrix_records = []
        for current_date in df_to_process.index:
            daily_matrix_records = self._compute_trendline_matrix_for_day(current_date, df_daily, data_dfs)
            new_matrix_records.extend(daily_matrix_records)
        if not new_matrix_records:
            print("  -> [增量回溯引擎] 未生成任何新的趋势线矩阵记录。")
            return
        self._save_trendline_matrix_incrementally(new_matrix_records)
        print(f"  -> [增量回溯引擎] 批量保存了 {len(new_matrix_records)} 条新的趋势线矩阵记录。")
        matrix_qs = self.mtt_model.objects.filter(stock=self.stock_instance).order_by('trade_date').values()
        matrix_df = pd.DataFrame.from_records(matrix_qs)
        if matrix_df.empty:
            print("  -> [增量回溯引擎] 加载完整矩阵失败，跳过动态分析。")
            return
        matrix_df['trade_date'] = pd.to_datetime(matrix_df['trade_date'])
        dynamic_events = self._analyze_matrix_dynamics(matrix_df, start_analysis_date=start_process_date)
        self._save_trendline_events_incrementally(dynamic_events)

    def _initialize_incremental_context(self, start_date_override: str = None) -> pd.Timestamp:
        """
        【V2.17 新增】初始化增量计算上下文，处理数据清理和起始日期确定。
        参照高级筹码和资金流任务的统一上下文初始化逻辑。
        """
        from datetime import datetime
        # 1. 如果提供了起始日期，执行回滚删除逻辑
        if start_date_override:
            try:
                save_start_date = datetime.strptime(start_date_override, '%Y-%m-%d').date()
                # 删除指定日期之后的所有趋势线矩阵和事件记录
                mtt_del_count, _ = self.mtt_model.objects.filter(stock=self.stock_instance, trade_date__gte=save_start_date).delete()
                event_del_count, _ = self.event_model.objects.filter(stock=self.stock_instance, event_date__gte=save_start_date).delete()
                print(f"  -> [统一回滚] 趋势线矩阵删除 {mtt_del_count} 条，动态事件删除 {event_del_count} 条。")
                return pd.to_datetime(save_start_date)
            except (ValueError, TypeError):
                print(f"  -> [错误] 提供的起始日期 '{start_date_override}' 格式错误，将执行标准增量更新。")
        # 2. 标准增量逻辑：查找最新记录
        last_record = self.mtt_model.objects.filter(stock=self.stock_instance).order_by('-trade_date').first()
        if last_record:
            # 从最后记录的下一天开始处理
            return pd.to_datetime(last_record.trade_date) + pd.Timedelta(days=1)
        # 3. 如果没有任何记录，返回None，表示需要从头开始
        return None

    def _analyze_matrix_dynamics(self, matrix_df: pd.DataFrame, start_analysis_date: pd.Timestamp = None) -> list:
        """
        【V2.18 · 维度校准版】分析趋势线矩阵的时间序列，识别动态事件。
        - V2.18 核心修复: 修正了“通道压缩分析”中的矢量化误用问题。通过直接在DataFrame列上进行
                         矢量化计算来生成`historical_widths`，根除了因维度不匹配导致的ValueError。
        """
        if matrix_df.empty or len(matrix_df['trade_date'].unique()) < 2:
            return []
        print(f"    -> [战场AI参谋] 开始分析趋势线矩阵动态...")
        final_events = []
        matrix_wide_df = matrix_df.pivot_table(
            index='trade_date',
            columns=['period', 'line_type'],
            values=['slope', 'intercept', 'validity_score']
        ).sort_index()
        matrix_wide_df.columns = ['_'.join(map(str, col)) for col in matrix_wide_df.columns.values]
        analysis_df = matrix_wide_df[matrix_wide_df.index >= start_analysis_date] if start_analysis_date else matrix_wide_df
        def _get_line_value(row, period, line_type, time_idx):
            slope = row.get(f'slope_{period}_{line_type}')
            intercept = row.get(f'intercept_{period}_{line_type}')
            if pd.notna(slope) and pd.notna(intercept):
                return slope * time_idx + intercept
            return np.nan
        for i, (trade_date, today_row) in enumerate(analysis_df.iterrows()):
            if i == 0: continue
            yesterday_row = analysis_df.iloc[i-1]
            time_idx_today = i + (len(matrix_wide_df) - len(analysis_df))
            time_idx_yesterday = time_idx_today - 1
            for period in self.fib_periods:
                slope_col = f'slope_{period}_support'
                if slope_col in today_row and slope_col in yesterday_row and pd.notna(today_row[slope_col]) and pd.notna(yesterday_row[slope_col]):
                    slope_today = today_row[slope_col]
                    slope_yesterday = yesterday_row[slope_col]
                    slope_diff = slope_today - slope_yesterday
                    details = {}
                    event_type = None
                    if slope_today > 0 and slope_yesterday > 0 and slope_diff > 0.001:
                        event_type = 'INFLECTION_ACCEL'
                        details = {'period': period, 'type': 'support', 'strength': slope_diff}
                    elif slope_today > 0 and slope_yesterday > 0 and slope_diff < -0.001:
                        event_type = 'INFLECTION_DECEL'
                        details = {'period': period, 'type': 'support', 'strength': slope_diff}
                    if np.sign(slope_today) != np.sign(slope_yesterday):
                        event_type = 'INFLECTION_REVERSAL'
                        details = {'period': period, 'type': 'support', 'from': slope_yesterday, 'to': slope_today}
                    if event_type:
                        final_events.append({
                            'stock': self.stock_instance, 'event_date': trade_date.date(),
                            'event_type': event_type, 'details': self._sanitize_json_dict(details)
                        })
            for short_p, long_p in combinations(self.fib_periods, 2):
                if short_p >= long_p: continue
                short_val_t = _get_line_value(today_row, short_p, 'support', time_idx_today)
                long_val_t = _get_line_value(today_row, long_p, 'support', time_idx_today)
                short_val_y = _get_line_value(yesterday_row, short_p, 'support', time_idx_yesterday)
                long_val_y = _get_line_value(yesterday_row, long_p, 'support', time_idx_yesterday)
                if not all(pd.notna(v) for v in [short_val_t, long_val_t, short_val_y, long_val_y]):
                    continue
                pos_today = short_val_t - long_val_t
                pos_yesterday = short_val_y - long_val_y
                if np.sign(pos_today) != np.sign(pos_yesterday):
                    event_type = None
                    sup_s_t = today_row.get(f'slope_{short_p}_support')
                    sup_l_t = today_row.get(f'slope_{long_p}_support')
                    angle_short = np.degrees(np.arctan(sup_s_t))
                    angle_long = np.degrees(np.arctan(sup_l_t))
                    intersection_angle = abs(angle_short - angle_long)
                    convergence_rate = abs(pos_yesterday)
                    details = {
                        'periods': f'{short_p}v{long_p}',
                        'angle': intersection_angle,
                        'convergence_rate': convergence_rate
                    }
                    if pos_today > 0:
                        event_type = 'CROSS_GOLDEN_DECISIVE' if intersection_angle > 20 else 'CROSS_GOLDEN_TENTATIVE'
                    else:
                        event_type = 'CROSS_DEATH_DECISIVE' if intersection_angle > 20 else 'CROSS_DEATH_TENTATIVE'
                    if event_type:
                        final_events.append({
                            'stock': self.stock_instance, 'event_date': trade_date.date(),
                            'event_type': event_type, 'details': self._sanitize_json_dict(details)
                        })
            all_periods_slopes = {p: today_row.get(f'slope_{p}_support') for p in self.fib_periods}
            valid_slopes = {p: s for p, s in all_periods_slopes.items() if pd.notna(s)}
            if len(valid_slopes) >= 3:
                slopes_series = pd.Series(valid_slopes)
                concordance_score = ((slopes_series > 0).sum() / len(slopes_series))
                cohesion_score = 1 / (1 + slopes_series.std())
                event_type = None
                details = {'concordance': concordance_score, 'cohesion': cohesion_score}
                if concordance_score >= 0.8 and cohesion_score >= 0.8:
                    event_type = 'RESONANCE_BULLISH_STRONG'
                elif concordance_score <= 0.2 and cohesion_score >= 0.8:
                    event_type = 'RESONANCE_BEARISH_STRONG'
                if event_type:
                    final_events.append({
                        'stock': self.stock_instance, 'event_date': trade_date.date(),
                        'event_type': event_type, 'details': self._sanitize_json_dict(details)
                    })
                short_term_periods = [p for p in valid_slopes.keys() if p < 21]
                long_term_periods = [p for p in valid_slopes.keys() if p >= 21]
                if short_term_periods and long_term_periods:
                    short_term_slope_mean = slopes_series[short_term_periods].mean()
                    long_term_slope_mean = slopes_series[long_term_periods].mean()
                    yesterday_slopes_series = pd.Series({p: yesterday_row.get(f'slope_{p}_support') for p in valid_slopes.keys()})
                    short_term_slope_change = short_term_slope_mean - yesterday_slopes_series[short_term_periods].mean()
                    event_type = None
                    details = {
                        'short_term_slope': short_term_slope_mean,
                        'long_term_slope': long_term_slope_mean,
                        'short_term_momentum': short_term_slope_change
                    }
                    if long_term_slope_mean > 0.005 and short_term_slope_mean > 0 and short_term_slope_change < -0.001:
                        event_type = 'DIVERGENCE_BEARISH_TOP'
                    elif long_term_slope_mean < -0.005 and short_term_slope_mean < 0 and short_term_slope_change > 0.001:
                        event_type = 'DIVERGENCE_BULLISH_BOTTOM'
                    if event_type:
                        final_events.append({
                            'stock': self.stock_instance, 'event_date': trade_date.date(),
                            'event_type': event_type, 'details': self._sanitize_json_dict(details)
                        })
            # V2.18 修正矢量化误用问题
            # D. 通道压缩分析
            if len(self.fib_periods) > 0:
                median_period_index = len(self.fib_periods) // 2
                period = self.fib_periods[median_period_index]
                channel_width = _get_line_value(today_row, period, 'resistance', time_idx_today) - _get_line_value(today_row, period, 'support', time_idx_today)
                if pd.notna(channel_width):
                    # 使用直接的矢量化操作，而不是错误地调用_get_line_value
                    time_indices = np.arange(len(matrix_wide_df))
                    resistance_values = (matrix_wide_df.get(f'slope_{period}_resistance') * time_indices + 
                                         matrix_wide_df.get(f'intercept_{period}_resistance'))
                    support_values = (matrix_wide_df.get(f'slope_{period}_support') * time_indices + 
                                      matrix_wide_df.get(f'intercept_{period}_support'))
                    historical_widths = resistance_values - support_values
                    squeeze_threshold = historical_widths.rolling(250, min_periods=50).quantile(0.1).loc[trade_date]
                    if not np.isnan(squeeze_threshold) and channel_width < squeeze_threshold:
                        details = {'period': period, 'width': channel_width, 'threshold': squeeze_threshold}
                        final_events.append({
                            'stock': self.stock_instance, 'event_date': trade_date.date(),
                            'event_type': 'COMPRESSION_SQUEEZE', 'details': self._sanitize_json_dict(details)
                        })
        print(f"    -> [战场AI参谋] 分析完毕，共发现 {len(final_events)} 个新的动态事件。")
        return final_events

    def _assess_platform_character(self, platform_df: pd.DataFrame) -> (str, float):
        """
        【V2.19 新增】平台性质评估专家系统。
        对给定的平台区间进行多维情报审问，返回其定性性质和定量评分。
        """
        scores = {}
        # 1. 资金证据 (权重: 40%)
        # 主力资金累计流向
        mf_net_flow_sum = platform_df['main_force_net_flow_calibrated'].sum()
        total_amount = platform_df['amount'].sum()
        mf_flow_ratio = (mf_net_flow_sum * 10000) / total_amount if total_amount > 0 else 0
        scores['fund_flow'] = np.clip(mf_flow_ratio / 5.0, -1, 1) * 25 # 占比超过5%视为极强
        # 隐蔽吸筹强度
        hidden_accum_mean = platform_df['hidden_accumulation_intensity'].mean()
        scores['hidden_accum'] = np.clip(hidden_accum_mean / 20.0, 0, 1) * 15 # 平均强度20为满分
        # 2. 筹码证据 (权重: 40%)
        # 主峰稳固度变化趋势
        solidity_trend = platform_df['dominant_peak_solidity'].diff().mean()
        scores['chip_solidity'] = np.clip(solidity_trend * 100, -1, 1) * 15 # 每日增长0.01为满分
        # 结构张力
        tension_mean = platform_df['structural_tension_index'].mean()
        scores['chip_tension'] = np.clip((tension_mean - 1) / 0.5, 0, 1) * 15 # 张力指数1.5为满分
        # 获利盘稳定度
        winner_stability_mean = platform_df['winner_stability_index'].mean()
        scores['winner_stability'] = np.clip((winner_stability_mean - 0.8) / 0.2, 0, 1) * 10 # 稳定度0.8以上开始加分
        # 3. 结构证据 (权重: 20%)
        # 成交量爆裂度 (越低越好)
        burstiness_mean = platform_df['volume_burstiness_index'].mean()
        scores['structure_burst'] = (1 - np.clip(burstiness_mean / 2.0, 0, 1)) * 10 # 爆裂度超过2认为混乱
        # 日内能量密度 (越低越好)
        energy_mean = platform_df['intraday_energy_density'].mean()
        scores['structure_energy'] = (1 - np.clip(energy_mean / 1.5, 0, 1)) * 10 # 能量密度超过1.5认为博弈激烈
        # 计算最终得分
        final_score = sum(scores.values())
        # 根据得分定性
        character = 'CONSOLIDATION'
        if final_score > 40:
            character = 'ACCUMULATION'
        elif final_score < -30:
            character = 'DISTRIBUTION'
        elif -30 <= final_score < 0 and scores.get('structure_burst', 0) < 5: # 结构混乱，偏向洗盘
            character = 'SHAKEOUT'
        return character, round(final_score, 2)

    def _compute_trendline_matrix_for_day(self, current_date: pd.Timestamp, df_daily: pd.DataFrame, data_dfs: dict) -> list:
        """
        【V2.52 · 探针增强版】为指定的一天，计算出所有斐波那契周期的支撑和阻力线矩阵，并为其注入“信念评分”。
        - V2.52 升级: 新增了仅在最新交易日才会激活的“信念探针”，用于输出最终信念评分
                     及其详细的四维构成，确保了核心决策过程的完全透明。
        """
        matrix_records = []
        enriched_df = data_dfs.get('enriched_df')
        if enriched_df is None:
            print("  -> [信念评估警告] 未能在data_dfs中找到enriched_df，无法计算趋势信念分。")
            return []
        # V2.52 新增调试日判断逻辑
        is_debug_day = (current_date == df_daily.index.max())
        if is_debug_day:
            print(f"\n    -> [趋势信念探针] {current_date.date()} (最新交易日)")
        for period in self.fib_periods:
            lookback_days = period * 3
            start_date = current_date - pd.Timedelta(days=lookback_days)
            df_slice = df_daily[(df_daily.index >= start_date) & (df_daily.index <= current_date)].copy()
            if len(df_slice) < period:
                continue
            zigzag_series = self._calculate_zigzag(df_slice, threshold=0.05)
            df_slice['zigzag'] = zigzag_series
            df_slice['zigzag'] = df_slice['zigzag'].fillna(0)
            df_slice['time_idx'] = np.arange(len(df_slice))
            pivot_highs = df_slice[df_slice['zigzag'] == 1]
            pivot_lows = df_slice[df_slice['zigzag'] == -1]
            best_support = self._find_best_line_with_micro_validation(pivot_lows, 'low_qfq', df_slice, 'support', data_dfs)
            best_resistance = self._find_best_line_with_micro_validation(pivot_highs, 'high_qfq', df_slice, 'resistance', data_dfs)
            for line_data in [best_support, best_resistance]:
                if line_data:
                    # V2.52 传入debug_flag激活探针
                    conviction_score = self._calculate_trend_conviction_score(line_data, enriched_df)
                    # V2.52 新增一级探针（最终得分）
                    if is_debug_day:
                        line_type_display = "支撑线" if line_data['line_type'] == 'support' else "阻力线"
                        print(f"      - [{period}日 {line_type_display}] -> 最终信念评分: {conviction_score:.2f}")
                    matrix_records.append({
                        'stock': self.stock_instance,
                        'trade_date': current_date.date(),
                        'period': period,
                        'line_type': line_data['line_type'],
                        'slope': line_data['slope'],
                        'intercept': line_data['intercept'],
                        'validity_score': line_data['validity_score'],
                        'trend_conviction_score': conviction_score,
                    })
        return matrix_records

    def _find_best_line_with_micro_validation(self, pivots: pd.DataFrame, price_col: str, full_df: pd.DataFrame, line_type: str, data_dfs: dict):
        """
        【V2.52 · 趋势信念版】通过启发式剪枝策略，从根本上解决组合爆炸导致的性能问题。
        - V2.52 升级: 在返回的最佳趋势线信息中，增加 `start_date` 和 `end_date`，
                     为下游的“趋势信念评分”计算提供必要的起止区间。
        """
        is_last_day_debug = (full_df.index.max() == data_dfs.get('daily_data').index.max())
        if is_last_day_debug:
            print(f"      [内部探针] 进入 _find_best_line... | 类型: {line_type}, 接收到锚点数: {len(pivots)}")
        if len(pivots) < 2: return None
        MAX_PIVOTS_TO_COMBINE = 20
        NUM_RECENT_PIVOTS = 10
        NUM_EXTREME_PIVOTS = 10
        if len(pivots) > MAX_PIVOTS_TO_COMBINE:
            recent_pivots = pivots.tail(NUM_RECENT_PIVOTS)
            if line_type == 'support':
                extreme_pivots = pivots.nsmallest(NUM_EXTREME_PIVOTS, price_col)
            else: # resistance
                extreme_pivots = pivots.nlargest(NUM_EXTREME_PIVOTS, price_col)
            pivots_to_check = pd.concat([recent_pivots, extreme_pivots]).drop_duplicates()
            if is_last_day_debug:
                print(f"        [剪枝策略启动] 锚点数从 {len(pivots)} 减少到 {len(pivots_to_check)}")
        else:
            pivots_to_check = pivots
        best_line_info = None
        max_final_score = -1
        tick_map = data_dfs.get("stock_tick_data_map", {})
        for p1_idx, p2_idx in combinations(pivots_to_check.index, 2):
            p1 = pivots_to_check.loc[p1_idx]
            p2 = pivots_to_check.loc[p2_idx]
            if p1['time_idx'] == p2['time_idx']: continue
            m = (p2[price_col] - p1[price_col]) / (p2['time_idx'] - p1['time_idx'])
            c = p1[price_col] - m * p1['time_idx']
            touch_points_indices = {p1_idx, p2_idx}
            all_other_pivots = pivots_to_check.drop(index=[p1_idx, p2_idx])
            if not all_other_pivots.empty:
                predicted_prices = m * all_other_pivots['time_idx'] + c
                errors = np.abs(all_other_pivots[price_col] - predicted_prices) / all_other_pivots[price_col]
                additional_touches = all_other_pivots[errors < 0.015].index
                touch_points_indices.update(additional_touches)
            duration = (p2_idx - p1_idx).days
            duration_score = np.log1p(duration) / np.log1p(90)
            geometric_score = (np.log1p(len(touch_points_indices)) / np.log1p(10)) * 0.7 + duration_score * 0.3
            micro_scores = []
            for touch_date in sorted(list(touch_points_indices)):
                score = self._calculate_micro_conviction_score(touch_date, line_type, tick_map)
                micro_scores.append(score)
            avg_micro_score = np.mean(micro_scores) if micro_scores else 0.0
            line_df = full_df[(full_df.index >= p1_idx) & (full_df.index <= p2_idx)].copy()
            line_df['line_price'] = m * line_df['time_idx'] + c
            fake_break_bonus = 0
            if line_type == 'support':
                penetrations = line_df[line_df['low_qfq'] < line_df['line_price']]
                if not penetrations.empty:
                    fake_break_bonus = (penetrations['close_qfq'] > penetrations['line_price']).sum()
                penetration_count = len(penetrations)
            else: # resistance
                penetrations = line_df[line_df['high_qfq'] > line_df['line_price']]
                if not penetrations.empty:
                    fake_break_bonus = (penetrations['close_qfq'] < penetrations['line_price']).sum()
                penetration_count = len(penetrations)
            penetration_penalty = (penetration_count - fake_break_bonus) / len(line_df) if len(line_df) > 0 else 0
            dynamic_score = 1 - penetration_penalty
            final_score = (geometric_score * 0.3) + (avg_micro_score * 0.5) + (dynamic_score * 0.2)
            if final_score > max_final_score:
                max_final_score = final_score
                best_line_info = {
                    'line_type': line_type,
                    'slope': m,
                    'intercept': c,
                    'validity_score': final_score,
                    'start_date': p1_idx, # V2.52 新增起始日期
                    'end_date': p2_idx,   # V2.52 新增结束日期
                }
        if is_last_day_debug:
            print(f"      [内部探针] 退出 _find_best_line... | 类型: {line_type}, 最高得分为: {max_final_score:.4f}")
        return best_line_info

    def _calculate_micro_conviction_score(self, touch_date: pd.Timestamp, line_type: str, tick_map: dict) -> float:
        """
        【V2.1 战役复盘】分析触及点当日的Tick数据，量化多空博弈的“信念强度”。
        返回一个范围在 [0, 1] 的分数，1代表极强的信念。
        """
        df_tick = tick_map.get(touch_date.date())
        if df_tick is None or df_tick.empty:
            return 0.5 # 没有微观数据，给予中性分

        # 计算全天OFI
        buy_vol = df_tick[df_tick['type'] == 'B']['volume'].sum()
        sell_vol = df_tick[df_tick['type'] == 'S']['volume'].sum()
        total_active_vol = buy_vol + sell_vol
        
        if total_active_vol == 0:
            return 0.5

        ofi_ratio = (buy_vol - sell_vol) / total_active_vol
        
        # 根据趋势线类型，判断OFI是否提供了“信念”证明
        conviction_score = 0.5
        if line_type == 'support':
            # 支撑线需要主动买盘确认，OFI为正加分
            conviction_score = 0.5 + (ofi_ratio / 2) # 将 [-1, 1] 的 ofi_ratio 映射到 [0, 1]
        elif line_type == 'resistance':
            # 阻力线需要主动卖盘确认，OFI为负加分
            conviction_score = 0.5 - (ofi_ratio / 2) # 将 [-1, 1] 的 ofi_ratio 映射到 [1, 0]
            
        return np.clip(conviction_score, 0, 1)

    def _calculate_breakout_momentum_from_realtime(self, df_realtime: pd.DataFrame) -> float:
        """
        【V2.0 新增】从Realtime快照数据计算突破日的动能得分。
        - 核心思想: 一个高质量的突破，应该伴随着持续、强劲的价格上涨速度。
        """
        if df_realtime is None or df_realtime.empty or len(df_realtime) < 2:
            return 0.0
        df_realtime = df_realtime.copy()
        df_realtime['price_change_pct'] = df_realtime['current_price'].diff() / df_realtime['prev_close_price']
        df_realtime['time_diff_seconds'] = df_realtime.index.to_series().diff().dt.total_seconds()
        # 过滤掉无效的时间差
        df_realtime = df_realtime[df_realtime['time_diff_seconds'] > 0]
        if df_realtime.empty:
            return 0.0
            
        # 计算每秒的价格变化率（速度）
        df_realtime['price_velocity'] = df_realtime['price_change_pct'] / df_realtime['time_diff_seconds']
        # 动能分 = 平均速度 * 速度为正的时间占比 (惩罚下跌)
        positive_velocity_ratio = (df_realtime['price_velocity'] > 0).mean()
        mean_velocity = df_realtime['price_velocity'].mean()
        # 归一化处理，假设 0.01%/秒 是一个很强的速度
        momentum_score = np.clip((mean_velocity / 0.0001) * positive_velocity_ratio, -1, 1)
        return momentum_score

    def _find_best_line(self, pivots: pd.DataFrame, price_col: str, full_df: pd.DataFrame, line_type: str):
        """
        从候选点中找出评分最高的趋势线。
        """
        # 如果候选锚点少于2个，无法构成一条线，直接返回
        if len(pivots) < 2: return None
        best_line = None
        max_score = -1
        # 遍历所有可能的锚点对组合
        for p1_idx, p2_idx in combinations(pivots.index, 2):
            p1 = pivots.loc[p1_idx]
            p2 = pivots.loc[p2_idx]
            # 确保两个点不在同一时间索引上
            if p1['time_idx'] == p2['time_idx']: continue
            # 计算直线方程 y = mx + c 的斜率(m)和截距(c)
            m = (p2[price_col] - p1[price_col]) / (p2['time_idx'] - p1['time_idx'])
            c = p1[price_col] - m * p1['time_idx']
            # --- 开始对该候选线进行评分 ---
            touch_points = 2 # 初始触及点为定义线的两个锚点
            penetration_penalty = 0
            # 1. 检查其他锚点的触及情况
            for idx, pivot in pivots.iterrows():
                if idx == p1_idx or idx == p2_idx: continue # 跳过定义点本身
                # 根据线的方程预测当前锚点时间上的价格
                predicted_price = m * pivot['time_idx'] + c
                # 如果实际价格与预测价格的误差在1.5%以内，则视为有效触及
                if abs(pivot[price_col] - predicted_price) / pivot[price_col] < 0.015:
                    touch_points += 1
            # 2. 检查价格穿透情况
            # 截取趋势线生命周期内的数据
            line_df = full_df[(full_df.index >= p1_idx) & (full_df.index <= p2_idx)]
            # 计算线上每个点的理论价格
            line_df['line_price'] = m * line_df['time_idx'] + c
            if line_type == 'support':
                # 对于支撑线，统计最低价低于线价的K线数量
                penetrations = line_df[line_df['low_D'] < line_df['line_price']]
            else: # resistance
                # 对于阻力线，统计最高价高于线价的K线数量
                penetrations = line_df[line_df['high_D'] > line_df['line_price']]
            # 计算穿透惩罚率 = 穿透K线数 / 总K线数
            penetration_penalty = len(penetrations) / len(line_df)
            # 3. 计算持续时间得分
            duration = (p2_idx - p1_idx).days
            duration_score = min(duration / 90, 1.0) # 以90天为满分，进行归一化
            # 4. 计算综合评分
            # 综合分 = (触及点权重 + 持续时间权重) * (1 - 穿透惩罚)
            score = (touch_points * 0.5 + duration_score * 0.5) * (1 - penetration_penalty)
            # 如果当前线的得分高于已记录的最高分，则更新最佳线
            if score > max_score:
                max_score = score
                best_line = {
                    'stock': self.stock_instance,
                    'start_date': p1_idx.date(),
                    'end_date': p2_idx.date(),
                    'line_type': line_type,
                    'slope': m,
                    'intercept': c,
                    'touch_points': touch_points,
                    'validity_score': score,
                }
        return best_line

    def _translate_conviction_to_probability(self, flag: dict) -> (float, dict):
        """
        【V2.74 · 启示录版】将信念评分直接转换为突破概率，实现认知统一。
        - V2.74 核心升级:
          1. [方法更名] `_predict_flag_breakout_probability` 更名为 `_translate_conviction_to_probability`，实现名实相符。
          2. [逻辑固化] 保持V2.73“归一”版本的核心逻辑，即信念分数的直接线性映射。
        """
        conviction_score = flag.get('conviction_score', 0.0)
        final_probability = conviction_score / 100.0
        print(f"    -> [认知统一 V2.74]")
        print(f"      - 接收到'全息审判'信念评分: {conviction_score:.2f}")
        print(f"      - >> 直接映射为最终突破概率: {final_probability:.2%}")
        features = {
            'conviction_score': conviction_score,
            'retracement_depth': flag.get('retracement_depth'),
            'duration': flag.get('duration'),
        }
        return final_probability, features

    def _find_and_evaluate_flags(self, enriched_df: pd.DataFrame, data_dfs: dict) -> list:
        """
        【V5.0 · 生产净化版】移除所有内部诊断探针，实现日志的最终净化。
        - V5.0 核心升级:
          1. [日志净化] 移除了“认知焦点”和“时空跳跃”等过程性探针日志。
          2. [聚焦结果] 只保留最高级别的启动、警告和最终发现日志，使输出聚焦于核心决策结果。
        """
        events = []
        for archetype in self.flag_archetypes:
            timeframe = archetype.get('timeframe', 'D')
            archetype_name = archetype.get('name', 'UNKNOWN_FLAG')
            # 简化顶层日志
            print(f"\n  -> [全息旗形扫描] 开始在 [{timeframe}] 级别应用原型 [{archetype_name}]...")
            df_source = None
            if timeframe == 'D':
                df_source = enriched_df
            elif timeframe == 'W':
                df_source = data_dfs.get('weekly_data')
            if df_source is None or df_source.empty:
                print(f"     - [数据缺失] 未找到 [{timeframe}] 级别数据，跳过此原型。")
                continue
            min_data_len = archetype.get('min_data_len', 55)
            if len(df_source) < min_data_len:
                print(f"     - [数据不足] {timeframe} 级别数据长度 {len(df_source)} < {min_data_len}，跳过此原型。")
                continue
            df = df_source.copy()
            vol_ma_col_name = f'vol_ma{self.long_term_period}_{timeframe}'
            df[vol_ma_col_name] = df['vol'].rolling(self.long_term_period).mean()
            ultra_long_ma_col_name = f'ma{self.ultra_long_term_period}_{timeframe}'
            df[ultra_long_ma_col_name] = df['close_qfq'].rolling(self.ultra_long_term_period, min_periods=self.long_term_period).mean()
            i = len(df) - 5
            while i > min_data_len:
                if not self._is_potential_pole_peak(df, i, vol_ma_col_name):
                    # 移除“认知焦点”探针
                    i -= 1
                    continue
                pole = self._identify_flagpole(df, end_index_loc=i, vol_ma_col_name=vol_ma_col_name, archetype=archetype, data_dfs=data_dfs)
                if pole:
                    flag = self._identify_flag(df, pole, vol_ma_col_name=vol_ma_col_name, archetype=archetype, data_dfs=data_dfs)
                    if flag:
                        passes_quality_check = True
                        if timeframe == 'D':
                            pole_df = df.loc[pole['start_date']:pole['end_date']]
                            flag_df = df.loc[flag['start_date']:flag['end_date']]
                            is_main_force_led = pole_df['main_force_net_flow_calibrated_sum_5d'].iloc[-1] > 0
                            avg_hidden_accumulation = flag_df['hidden_accumulation_intensity'].mean()
                            dominant_peak_cost_at_start = flag_df['dominant_peak_cost'].iloc[0]
                            breakout_readiness = flag_df['breakout_readiness_score'].iloc[-1]
                            if not is_main_force_led or avg_hidden_accumulation <= 0 or flag_df['low_qfq'].min() < dominant_peak_cost_at_start or breakout_readiness < 60:
                                passes_quality_check = False
                        if passes_quality_check:
                            print(f"  -> [高置信度旗形发现!] 日期: {flag['end_date'].date()}, 级别: [{timeframe}], 原型: [{archetype_name}]。启动概率转换...")
                            probability, features = self._translate_conviction_to_probability(flag)
                            events.append({
                                'stock': self.stock_instance,
                                'event_date': flag['end_date'].date(),
                                'event_type': f'FLAG_FORMED_{timeframe}',
                                'details': {
                                    'archetype': archetype_name,
                                    'probability': probability,
                                    'features': features,
                                    'pole_start_date': pole['start_date'].date(),
                                    'pole_end_date': pole['end_date'].date(),
                                }
                            })
                    # 移除“时空跳跃”探针
                    i = df.index.get_loc(pole['start_date']) - 1
                    continue
                i -= 1
        return events

    def _identify_flagpole(self, df: pd.DataFrame, end_index_loc: int, vol_ma_col_name: str, archetype: dict, data_dfs: dict) -> dict:
        """
        【V5.0 · 生产净化版】移除所有内部诊断探针，实现日志的最终净化。
        - V5.0 核心升级:
          1. [日志净化] 移除了对每个候选周期的所有评分细节（点火、幅度、方向等）的探针日志。
          2. [静默执行] 方法在执行过程中将保持静默，只在最终输出决策结果。
        """
        min_dur = archetype.get('pole_min_dur', 2)
        max_dur = archetype.get('pole_max_dur', 8)
        min_magnitude_atr = archetype.get('pole_magnitude_atr', 4.0)
        min_vol_multiple = archetype.get('pole_vol_multiple', 1.8)
        max_daily_drop_atr = archetype.get('pole_max_daily_drop_atr', 0.5)
        ignition_min_pct_change = archetype.get('ignition_min_pct_change', 4.0)
        ignition_min_vol_ratio = archetype.get('ignition_min_vol_ratio', 1.5)
        end_date = df.index[end_index_loc]
        minute_map = data_dfs.get("stock_minute_data_map", {})
        # 简化入口日志
        print(f"    -> [旗杆探针] 检查结束于 {end_date.date()} 的候选旗杆...")
        best_pole = None
        max_conviction_score = -1.0
        for duration in range(min_dur, max_dur + 1):
            start_index_loc = end_index_loc - duration + 1
            if start_index_loc < 1: continue
            pole_df = df.iloc[start_index_loc : end_index_loc + 1]
            start_date = pole_df.index[0]
            # 移除“候选周期”探针
            atr_at_start = df['ATR_14_D'].iloc[start_index_loc - 1]
            if atr_at_start == 0: continue
            daily_drops = pole_df['close_qfq'].diff().dropna()
            max_drop_value = abs(daily_drops[daily_drops < 0].min()) if not daily_drops[daily_drops < 0].empty else 0
            if max_drop_value > max_daily_drop_atr * atr_at_start:
                # 移除“纯度不符”探针
                continue
            ignition_day = pole_df.iloc[0]
            prev_day_vol = df['vol'].iloc[start_index_loc - 1]
            ignition_pct_change = ignition_day['pct_change']
            ignition_vol_ratio = ignition_day['vol'] / prev_day_vol if prev_day_vol > 0 else 10.0
            score_pct = np.clip(ignition_pct_change / (ignition_min_pct_change * 1.5), 0, 1)
            score_vol = np.clip(ignition_vol_ratio / (ignition_min_vol_ratio * 1.5), 0, 1)
            ignition_score = 100 * (score_pct * 0.6 + score_vol * 0.4)
            # 移除“点火强度”探针
            pole_high = pole_df['high_qfq'].max()
            pole_low = pole_df['low_qfq'].min()
            magnitude_atr = (pole_high - pole_low) / atr_at_start
            magnitude_score = 100 * (np.clip(magnitude_atr / (min_magnitude_atr * 1.5), 0, 1)) ** 2
            # 移除“幅度”探针
            pole_start_price = pole_df['open_qfq'].iloc[0]
            thrust_vector = (pole_high - pole_start_price) / (duration * atr_at_start)
            directional_score = np.clip(thrust_vector / 0.75, 0, 1) * 100
            # 移除“方向”探针
            vol_ma_at_start = df[vol_ma_col_name].iloc[start_index_loc - 1]
            avg_volume_pole = pole_df['vol'].mean()
            actual_vol_multiple = avg_volume_pole / vol_ma_at_start if vol_ma_at_start > 0 else 0
            energy_score = 100 * (np.clip(actual_vol_multiple / (min_vol_multiple * 1.5), 0, 1)) ** 2
            # 移除“能量”探针
            purity_scores = [self._calculate_intraday_trend_purity(minute_map.get(d.date())) for d in pole_df.index]
            purity_score = np.mean(purity_scores) if purity_scores else 50.0
            # 移除“纯度”探针
            conviction_score = (ignition_score * 0.30 +
                                magnitude_score * 0.25 +
                                directional_score * 0.15 +
                                energy_score * 0.10 +
                                purity_score * 0.20)
            # 移除“综合信念评分”探针
            if conviction_score > max_conviction_score:
                max_conviction_score = conviction_score
                best_pole = {
                    'start_date': start_date, 'end_date': end_date,
                    'high_price': pole_high, 'low_price': pole_low,
                    'start_price': pole_start_price,
                    'magnitude_atr': magnitude_atr, 'avg_volume': avg_volume_pole,
                    'conviction_score': conviction_score,
                    'duration_days': len(pole_df)
                }
        MIN_ACCEPTANCE_SCORE = 60.0
        if best_pole and best_pole['conviction_score'] >= MIN_ACCEPTANCE_SCORE:
            print(f"    -> [✓ ACCEPTED] 发现最佳旗杆 (周期 {best_pole['duration_days']}天), 最高信念分: {best_pole['conviction_score']:.2f}")
            return best_pole
        else:
            print(f"    -> [✗ REJECTED] 未发现任何候选旗杆的信念分超过 {MIN_ACCEPTANCE_SCORE:.1f} 的最低门槛。")
            return None

    def _identify_flag(self, df: pd.DataFrame, pole: dict, vol_ma_col_name: str, archetype: dict, data_dfs: dict) -> dict:
        """
        【V5.0 · 生产净化版】移除所有内部诊断探针，实现日志的最终净化。
        - V5.0 核心升级:
          1. [日志净化] 移除了对每个候选周期的所有评分细节（成交量、回撤等）的探针日志。
          2. [静默执行] 方法在执行过程中将保持静默，只在最终输出决策结果。
        """
        pole_end_loc = df.index.get_loc(pole['end_date'])
        max_dur = archetype.get('flag_max_dur', 15)
        min_dur = archetype.get('flag_min_dur', 5)
        max_retracement = archetype.get('flag_max_retracement', 0.618)
        vol_shrink_ratio = archetype.get('flag_vol_shrink_ratio', 0.7)
        max_buffers = archetype.get('flag_max_buffers', 2)
        # 简化入口日志
        print(f"    -> [旗面探针] 检查附着于 {pole['end_date'].date()} 旗杆的候选旗面...")
        pole_rise = pole['high_price'] - pole['start_price']
        if pole_rise <= 0: return None
        best_flag = None
        max_conviction_score = -1.0
        for buffer in range(max_buffers + 1):
            flag_start_loc = pole_end_loc + buffer
            if flag_start_loc >= len(df) - min_dur: continue
            is_fatal_low_not_at_start = False
            for duration in range(min_dur, max_dur + 1):
                flag_end_loc = flag_start_loc + duration - 1
                if flag_end_loc >= len(df): break
                flag_df = df.iloc[flag_start_loc : flag_end_loc + 1]
                # 移除“候选周期”探针
                avg_volume_flag = flag_df['vol'].mean()
                avg_vol_ma_flag = flag_df[vol_ma_col_name].mean()
                vol_vs_pole = avg_volume_flag / pole['avg_volume'] if pole['avg_volume'] > 0 else 1.0
                vol_vs_ma = avg_volume_flag / avg_vol_ma_flag if avg_vol_ma_flag > 0 else 1.0
                vol_score = (np.clip(1 - vol_vs_pole, 0, 1) * 0.6 + np.clip(1 - vol_vs_ma, 0, 1) * 0.4) * 100
                # 移除“成交量萎缩”探针
                flag_low = flag_df['low_qfq'].min()
                retracement = (pole['high_price'] - flag_low) / pole_rise
                retracement_score = np.clip((max_retracement - retracement) / (max_retracement - 0.10), 0, 1) * 100 if max_retracement > 0.10 else 0
                # 移除“回撤深度”探针
                if retracement_score < 10.0:
                    # 移除“单调剪枝”探针
                    if flag_df['low_qfq'].idxmin() != flag_df.index[0]:
                        is_fatal_low_not_at_start = True
                    break
                conviction_score = vol_score * 0.4 + retracement_score * 0.6
                if conviction_score > max_conviction_score:
                    max_conviction_score = conviction_score
                    best_flag = {
                        'start_date': flag_df.index[0], 'end_date': flag_df.index[-1],
                        'conviction_score': conviction_score,
                        'duration_days': len(flag_df)
                    }
            if is_fatal_low_not_at_start:
                # 移除“战略性撤退”探针
                break
        MIN_ACCEPTANCE_SCORE = 50.0
        if best_flag and best_flag['conviction_score'] >= MIN_ACCEPTANCE_SCORE:
            print(f"    -> [✓ ACCEPTED] 发现最佳旗面 (周期 {best_flag['duration_days']}天), 最高信念分: {best_flag['conviction_score']:.2f}")
            return best_flag
        else:
            print(f"    -> [✗ REJECTED] 未发现任何候选旗面的信念分超过 {MIN_ACCEPTANCE_SCORE:.1f} 的最低门槛。")
            return None

    def _calculate_expert_breakout_probability(self, flag: dict) -> (float, dict):
        """
        【V2.73 · 归一版】废除独立的专家规则系统，实现信念与概率的直接映射。
        - V2.73 核心升级:
          1. [认知统一] 移除所有硬编码的`if/then`规则和`log-odds`调整，消除认知冗余。
          2. [信念即概率] 最终的突破概率直接由`_identify_flag`方法计算出的`conviction_score`线性映射而来。
          3. [终极简化] 方法达到最终的、最纯粹的形态，标志着整个识别引擎思想的完全统一。
        """
        conviction_score = flag.get('conviction_score', 0.0)
        # V2.73 核心逻辑：信念即概率
        final_probability = conviction_score / 100.0
        # V2.73 升级探针日志，宣告认知统一
        print(f"    -> [认知统一 V2.73]")
        print(f"      - 接收到'全息审判'信念评分: {conviction_score:.2f}")
        print(f"      - >> 直接映射为最终突破概率: {final_probability:.2%}")
        # V2.73 净化返回的特征字典，使其与“归一”哲学对齐
        features = {
            'conviction_score': conviction_score,
            'retracement_depth': flag.get('retracement_depth'),
            'duration': flag.get('duration'),
        }
        return final_probability, features

    def _calculate_zigzag(self, df: pd.DataFrame, threshold: float = 0.05) -> pd.Series:
        """
        【V2.9 · Numba JIT 包装器】调用 Numba 加速的 Zigzag 实现，并将其结果包装为 Pandas Series。
        - 负责 Pandas 数据到 NumPy 数组的转换，以及结果的重新包装。
        - 保持了对服务内其他方法的接口兼容性。
        """
        # 将Pandas Series转换为Numba兼容的NumPy数组
        highs = df['high_qfq'].values
        lows = df['low_qfq'].values
        # 调用JIT编译的函数
        zigzag_array = _calculate_zigzag_numba(highs, lows, threshold)
        # 将结果包装回带有正确索引的Pandas Series
        return pd.Series(zigzag_array, index=df.index, dtype=int)

    def _sanitize_json_dict(self, data: dict) -> dict:
        """
        【V2.7 · JSON净化器】递归地清理一个字典，将其中不符合JSON规范的浮点数值
        （如 NaN, Infinity）替换为 None，以确保能够安全地存入数据库JSON字段。
        """
        import math
        if not isinstance(data, dict):
            return data
        clean_data = {}
        for key, value in data.items():
            if isinstance(value, dict):
                clean_data[key] = self._sanitize_json_dict(value)
            # 同时处理Python原生float和Numpy的浮点类型
            elif isinstance(value, (float, np.floating)):
                if math.isnan(value) or math.isinf(value):
                    clean_data[key] = None  # 替换为None，对应JSON的null
                else:
                    clean_data[key] = value
            else:
                clean_data[key] = value
        return clean_data

    def _save_trendline_matrix_incrementally(self, records: list):
        """
        【V2.56 · NaN值净化器版】批量保存新的趋势线矩阵记录。
        - V2.56 核心修复: 新增了一个“NaN值净化器”。在执行 `bulk_create` 之前，
                         会遍历所有待保存的记录，将其中所有Pandas/Numpy计算产生的
                         `NaN` 值替换为Python的 `None`。此举旨在根治因滚动窗口初期
                         数据不足而产生的 `NaN` 导致的数据库写入失败问题。
        """
        if not records:
            print("  -> [趋势线矩阵] 没有新的记录需要保存。")
            return
        print(f"  -> [趋势线矩阵] 正在新增 {len(records)} 条记录...")
        # V2.56 NaN值净化器
        sanitized_records = [
            {key: (None if isinstance(value, float) and pd.isna(value) else value) for key, value in record.items()}
            for record in records
        ]
        instances = [self.mtt_model(**record) for record in sanitized_records]
        try:
            self.mtt_model.objects.bulk_create(instances, ignore_conflicts=True)
        except Exception as e:
            print(f"  -> [数据库错误] 批量保存趋势线矩阵时发生错误: {e}")
            # 如果批量创建失败，可以尝试逐条插入以便调试
            for instance in instances:
                try:
                    instance.save()
                except Exception as single_e:
                    print(f"    -> 无法保存记录: {instance.__dict__}, 错误: {single_e}")

    def _save_trendline_events_incrementally(self, events: list):
        """【V2.16】持久化存储新增的趋势线动态事件。"""
        if not events: return
        print(f"  -> [趋势线事件] 正在新增 {len(events)} 个事件...")
        instances = [self.event_model(**evt) for evt in events]
        self.event_model.objects.bulk_create(instances, ignore_conflicts=True)

    def _calculate_volume_profile_skewness(self, group: pd.DataFrame) -> float:
        """计算加权成交量分布的价格偏度。"""
        if group['vol'].sum() == 0: return 0.0
        prices = group['close_qfq']
        weights = group['vol']
        weighted_mean = np.average(prices, weights=weights)
        weighted_std = np.sqrt(np.average((prices - weighted_mean)**2, weights=weights))
        if weighted_std == 0: return 0.0
        weighted_skew = np.average(((prices - weighted_mean) / weighted_std)**3, weights=weights)
        return weighted_skew

    def _calculate_linear_regression_slope(self, series: pd.Series, normalize: bool = True) -> float:
        """
        【V2.53 · 智能归一化版】对一个序列进行线性回归并返回斜率。
        - V2.53 核心升级: 新增 `normalize` 参数，允许调用者选择性地关闭内部的斜率归一化
                         步骤，以根除因外部已归一化而导致的“归一化冗余”问题。
        """
        series = series.dropna()
        if len(series) < 2: return 0.0
        x = np.arange(len(series))
        slope, _ = np.polyfit(x, series.values, 1)
        # [代码修改] V2.53 引入智能归一化逻辑
        if normalize:
            # 对斜率进行归一化，使其不受序列绝对值大小的影响
            return slope / series.mean() if series.mean() != 0 else 0.0
        else:
            # 直接返回原始斜率
            return slope

    def _calculate_volatility_contraction_ratio(self, group: pd.DataFrame) -> float:
        """计算平台前后半段的波动率收缩比。"""
        if 'ATRr_14' not in group.columns or len(group) < 4: return 1.0
        n = len(group) // 2
        first_half_atr = group['ATRr_14'].iloc[:n].mean()
        second_half_atr = group['ATRr_14'].iloc[n:].mean()
        if first_half_atr == 0: return 1.0 if second_half_atr == 0 else 999.0
        return second_half_atr / first_half_atr

    def _calculate_price_kurtosis(self, group: pd.DataFrame) -> float:
        """计算平台内日内行为的价格峰度。"""
        if len(group) < 4: return 3.0 # 返回正态分布的峰度
        daily_range = group['high_qfq'] - group['low_qfq']
        body_range = (group['close_qfq'] - group['open_qfq']).abs()
        # 避免除以零
        body_range[body_range == 0] = 0.0001
        ratio_series = daily_range / body_range
        # 移除极端值和无效值
        clean_series = ratio_series.replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean_series) < 4: return 3.0
        return clean_series.kurt()

    def _calculate_goodness_of_fit(self, metrics: dict, archetype: dict) -> float:
        """
        【V2.50 新增】计算候选平台与指定原型的“拟合优度”分数。
        - 核心思想: 对每一条规则，计算指标的满足度。完美满足得100分，
                     在规则边界外则按偏离程度线性扣分，最多扣除100分（即最低0分）。
                     总分是所有规则得分的平均值。
        """
        scores = []
        rules_map = {
            'vps': 'volume_profile_skewness',
            'vts': 'volume_trend_slope',
            'vcr': 'volatility_contraction_ratio',
            'pk': 'price_kurtosis',
            'rss': 'relative_strength_slope'
        }
        for metric_key, rule_key in rules_map.items():
            if rule_key in archetype:
                rules = archetype[rule_key]
                min_val = rules.get('min', -np.inf)
                max_val = rules.get('max', np.inf)
                value = metrics.get(metric_key, (min_val + max_val) / 2) # 若指标不存在，取中性值
                # 核心评分逻辑
                if min_val <= value <= max_val:
                    scores.append(100.0)
                else:
                    # 计算偏离度并进行惩罚
                    target_range = max_val - min_val if np.isfinite(min_val) and np.isfinite(max_val) else abs(value * 2)
                    if target_range == 0: target_range = 1 # 避免除零
                    deviation = min(abs(value - min_val), abs(value - max_val))
                    penalty = (deviation / target_range) * 100
                    scores.append(max(0, 100 - penalty))
        return np.mean(scores) if scores else 100.0

    def _calculate_platform_conviction_score(self, platform_group: pd.DataFrame) -> float:
        """
        【V2.54 · 真理契约版】引入故障快速失败协议，处理上游数据空洞。
        - V2.54 核心修复: 在计算相对强度斜率得分前，检查'rs'列是否存在。如果不存在，
                         直接将该项得分置为中立的50分，而不是基于一个虚假的0.0斜率
                         进行计算，从而保证最终信念评分不受数据缺失的污染。
        """
        # 支柱一: 控盘与洗盘力度 (Price Kurtosis) - 权重 30%
        pk = self._calculate_price_kurtosis(platform_group)
        # 将峰度值映射到0-100分，峰度在[3, 15]区间内线性得分
        score_kurtosis = np.clip((pk - 3) / (15 - 3), 0, 1) * 100
        # 支柱二: 多空共识凝聚度 (Volatility Contraction Ratio) - 权重 25%
        vcr = self._calculate_volatility_contraction_ratio(platform_group)
        # VCR越小越好，将[0.5, 1.5]的范围映射到[100, 0]分
        score_vcr = np.clip(1 - (vcr - 0.5) / (1.5 - 0.5), 0, 1) * 100
        # 支柱三: 供应枯竭信号 (Volume Trend Slope) - 权重 20%
        vts = self._calculate_linear_regression_slope(platform_group['vol'])
        # 斜率越小越好，将[-0.1, 0.1]的范围映射到[100, 0]分
        score_vts = np.clip(1 - (vts - (-0.1)) / (0.1 - (-0.1)), 0, 1) * 100
        # 支柱四: 日内控制权归属 (Average Internal Strength) - 权重 15%
        daily_range = platform_group['high_qfq'] - platform_group['low_qfq']
        internal_strength = ((platform_group['close_qfq'] - platform_group['low_qfq']) / daily_range.replace(0, np.nan)).fillna(0.5)
        score_internal = internal_strength.mean() * 100
        # 支柱五: 相对市场强度 (Relative Strength Slope) - 权重 10%
        # [代码修改] V2.54 实施“真理契约”，检查'rs'列是否存在
        if 'rs' in platform_group and not platform_group.empty and platform_group['rs'].iloc[0] != 0:
            rebased_rs = platform_group['rs'] / platform_group['rs'].iloc[0]
            rss = self._calculate_linear_regression_slope(rebased_rs, normalize=False)
            # 斜率越大越好，将[-0.01, 0.01]的范围映射到[0, 100]分
            score_rss = np.clip((rss - (-0.01)) / (0.01 - (-0.01)), 0, 1) * 100
        else:
            # 如果'rs'数据不存在，则给予中立分50，避免评分被污染
            score_rss = 50.0
        # 融合总分
        conviction_score = (
            score_kurtosis * 0.30 +
            score_vcr * 0.25 +
            score_vts * 0.20 +
            score_internal * 0.15 +
            score_rss * 0.10
        )
        return conviction_score

    def _calculate_trend_conviction_score(self, line_data: dict, enriched_df: pd.DataFrame) -> float:
        """
        【V2.63 · 意志统一版】最终净化，移除所有调试探针，模型达到最终生产状态。
        - V2.63 核心升级:
          1. [代码净化] 移除了方法签名中的 `debug_flag` 参数以及所有相关的 `print` 调试代码块。
          2. [生产就绪] 核心计算函数至此达到最终的纯粹形态，为生产环境部署做好准备。
             模型的进化之旅宣告完成。
        """
        import math
        start_date = line_data.get('start_date')
        end_date = line_data.get('end_date')
        line_type = line_data.get('line_type')
        if not all([start_date, end_date, line_type]):
            return 50.0
        trend_df = enriched_df.loc[start_date:end_date].copy()
        if trend_df.empty or len(trend_df) < 2:
            return 50.0
        last_day_stats = trend_df.iloc[-1]
        score_geometry = line_data.get('validity_score', 0.5) * 100
        def gaussian_normalize(value, mean, std):
            if pd.isna(value) or pd.isna(mean) or pd.isna(std) or std == 0:
                return 0.0
            z_score = (value - mean) / std
            return 100 * math.erf(z_score / math.sqrt(2))
        score_power, score_structure, score_behavior = 0.0, 0.0, 0.0
        if 'daily_flow_ratio' in trend_df.columns:
            avg_flow = trend_df['daily_flow_ratio'].mean()
            mean, std = last_day_stats.get('flow_mean'), last_day_stats.get('flow_std')
            score_power = gaussian_normalize(avg_flow, mean, std)
        if 'solidity_slope_5d' in trend_df.columns:
            avg_slope = trend_df['solidity_slope_5d'].mean()
            mean, std = last_day_stats.get('structure_mean'), last_day_stats.get('structure_std')
            score_structure = gaussian_normalize(avg_slope, mean, std)
        if 'efficiency_avg_5d' in trend_df.columns:
            avg_efficiency = trend_df['efficiency_avg_5d'].mean()
            if pd.notna(avg_efficiency) and avg_efficiency >= 0:
                score_behavior = 100 * (avg_efficiency ** 0.5)
            else:
                score_behavior = 0.0
        if line_type == 'resistance':
            score_power = -score_power
            score_structure = -score_structure
        power_contribution = (score_power + 100) / 2
        structure_contribution = (score_structure + 100) / 2
        behavior_contribution = score_behavior
        g_geom = score_geometry + 1
        g_power = power_contribution + 1
        g_struct = structure_contribution + 1
        g_behav = behavior_contribution + 1
        final_score = (
            (g_geom ** 0.15) *
            (g_power ** 0.35) *
            (g_struct ** 0.30) *
            (g_behav ** 0.20)
        ) - 1
        return final_score

    def _calculate_intraday_trend_purity(self, df_minute: pd.DataFrame) -> float:
        """
        【V3.0 · 量子透镜版 新增】计算日内趋势纯度。
        - 核心思想: 一根高质量的上涨K线（旗杆部分），其价格在盘中应持续运行在VWAP之上。
        - 计算逻辑: 综合考量价格在VWAP上方的时间占比和平均偏离幅度。
        - 返回值: [0, 100] 的纯度评分。
        """
        if df_minute is None or df_minute.empty or 'volume' not in df_minute.columns or df_minute['volume'].sum() == 0:
            return 50.0  # 数据不足，给予中性分
        df_minute.ta.vwap(append=True)
        vwap_col = [col for col in df_minute.columns if 'VWAP' in col]
        if not vwap_col:
            return 50.0
        vwap_col = vwap_col[0]
        above_vwap = df_minute['close'] > df_minute[vwap_col]
        time_purity = above_vwap.mean() * 100
        # 计算价格偏离VWAP的幅度，并进行归一化
        deviation = (df_minute['close'] - df_minute[vwap_col]) / df_minute[vwap_col]
        # 只考虑在VWAP上方的情况，偏离越大越好
        magnitude_purity = np.clip(deviation[above_vwap].mean() / 0.01, 0, 1) * 100 if above_vwap.any() else 0
        # 综合评分：时间纯度权重更高
        final_purity_score = time_purity * 0.7 + magnitude_purity * 0.3
        return final_purity_score

    def _calculate_flag_microstructure_score(self, flag_ticks: pd.DataFrame) -> float:
        """
        【V3.0 · 量子透镜版 新增】计算旗面微观结构分。
        - 核心思想: 一个高质量的旗面盘整，其内部应表现为“被动吸筹”而非“主动出货”。
        - 计算逻辑: 分析Tick数据中的买卖盘成交量。被动买盘（成交在卖一价）远大于被动卖盘（成交在买一价）
                     是强烈的吸筹信号。此处简化为分析主动买卖单（B/S type）的成交量对比。
        - 返回值: [0, 100] 的微观结构评分。
        """
        if flag_ticks is None or flag_ticks.empty:
            return 50.0 # 无Tick数据，给予中性分
        # 'B'代表主动买，'S'代表主动卖
        buy_vol = flag_ticks[flag_ticks['type'] == 'B']['volume'].sum()
        sell_vol = flag_ticks[flag_ticks['type'] == 'S']['volume'].sum()
        total_vol = buy_vol + sell_vol
        if total_vol == 0:
            return 50.0
        # 我们期望在旗面盘整时，主动卖盘（散户行为）被更强的力量吸收，或者市场整体交易意愿低
        # 一个健康的旗面，主动买盘不应过强（否则就不是盘整了），主动卖盘也不应占主导
        # 这里我们定义一个“微观失衡指数”，衡量主动买盘相对于总成交量的比例
        imbalance_ratio = (buy_vol - sell_vol) / total_vol
        # 将[-1, 1]的失衡比映射到[0, 100]分。我们期望一个接近平衡或轻微卖盘主导（被吸收）的状态。
        # 因此，失衡比接近0时得分最高。
        score = (1 - abs(imbalance_ratio)) * 100
        return score

    def _is_potential_pole_peak(self, df: pd.DataFrame, index_loc: int, vol_ma_col_name: str) -> bool:
        """
        【V3.4 · 认知焦点版 新增】判断某一天是否是潜在的旗杆顶点（“杆顶”）。
        - 核心思想: 作为一个轻量级的“门卫”，在进行昂贵的旗杆识别前，快速筛选掉
                     明显不可能是强劲拉升终点的日期。
        - 判断标准:
          1. 当日必须有显著的波动（日内振幅 > 1.2倍ATR）。
          2. 当日成交量必须显著放大（> 1.5倍长期均量）。
        - 返回值: 布尔值，表示是否值得进行深度扫描。
        """
        if index_loc <= 0:
            return False
        current_day = df.iloc[index_loc]
        atr = current_day.get('ATR_14_D', 0)
        if atr == 0:
            return False
        day_range = current_day['high_qfq'] - current_day['low_qfq']
        vol_ma = current_day.get(vol_ma_col_name, 0)
        if vol_ma == 0:
            return False
        # 条件1: 波动性要求
        is_volatile_enough = (day_range / atr) > 1.2
        # 条件2: 成交量要求
        is_volume_spiked = current_day['vol'] > vol_ma * 1.5
        return is_volatile_enough and is_volume_spiked









