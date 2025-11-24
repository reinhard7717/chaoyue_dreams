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
    get_multi_timeframe_trendline_model_by_code, # 修改：导入新模型辅助函数
    get_trendline_event_model_by_code, # 新增：导入事件模型辅助函数
    get_advanced_chip_metrics_model_by_code, # 新增：导入高级芯片指标模型辅助函数
    get_advanced_fund_flow_metrics_model_by_code, # 新增：导入高级资金流指标模型辅助函数
    get_advanced_structural_metrics_model_by_code, # 新增：导入高级结构性指标模型辅助函数
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
                # 修改代码行：不再重置cursor，而是让其继续前进，防止回跳
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
                # 修改代码行：不再重置cursor，而是让其继续前进，防止回跳
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
        【V2.23 · 主键认知修正版】
        - 核心修复: 根据 StockInfo 模型定义，其主键是 `stock_code` (CharField) 而非
                     自动生成的 `id` (AutoField)。因此，移除对不存在的 `stock_instance.id`
                     属性的调用，废除错误的 `self.stock_id_int` 属性。
        """
        self.stock_code = stock_code
        self.stock_instance = stock_instance
        # [代码修改] self.stock_id 存储的就是正确的主键 (stock_code)
        self.stock_id = stock_instance.stock_code
        # [代码删除] 移除错误的整数ID属性
        # self.stock_id_int = stock_instance.id
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
        【V2.31 · 司法探针版】准备一个包含所有高级指标的、信息增强的DataFrame。
        - V2.31 探针升级: 部署高密度探针网络(C.1, C.2)，在数据融合的每一步都进行
                         快照取证，以捕捉导致 *_qfq 列数据被污染的确切瞬间。
        """
        # --- [探针 C: 进入数据融合函数] ---
        if not df_daily.empty:
            latest_day_input = df_daily.iloc[-1]
            print(f"--- [探针 C: 进入数据融合函数] 最新日期: {latest_day_input.name.date()} ---")
            print(f"  -> high_qfq: {latest_day_input.get('high_qfq')}, low_qfq: {latest_day_input.get('low_qfq')}, close_qfq: {latest_day_input.get('close_qfq')}")
            print(f"--- [探针 C 结束] ---")
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
        # [代码修改] 增加循环内探针 C.2
        print(f"--- [探针 C.2: 循环合并追踪] ---")
        print(f"  -> [Pre-Loop] enriched_df 初始状态 (qfq数据):")
        print(enriched_df[['trade_date', 'high_qfq', 'low_qfq', 'close_qfq']].tail(3).to_string())
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
                print(f"  -> [Post-Merge with '{name}'] enriched_df 状态 (qfq数据):")
                print(enriched_df[['trade_date', 'high_qfq', 'low_qfq', 'close_qfq']].tail(3).to_string())
                print(f"     Null counts for qfq: \n{enriched_df[['high_qfq', 'low_qfq', 'close_qfq']].isnull().sum().to_string()}")
        print(f"--- [探针 C.2 结束] ---")
        enriched_df = enriched_df.set_index('trade_date').sort_index()
        # --- [探針 D: 数据融合完成] ---
        if not enriched_df.empty and not enriched_df.index.isnull().all():
            latest_day_output = enriched_df.iloc[-1]
            print(f"--- [探针 D: 数据融合完成] 最新日期: {latest_day_output.name.date()} ---")
            print(f"  -> high_qfq: {latest_day_output.get('high_qfq')}, low_qfq: {latest_day_output.get('low_qfq')}, close_qfq: {latest_day_output.get('close_qfq')}")
            print(f"--- [探针 D 结束] ---")
        else:
            print(f"--- [探针 D: 数据融合失败] DataFrame为空或索引损坏 ---")
        print(f"  -> [数据融合] 全维度战场沙盘构建完成。")
        return enriched_df

    def calculate_and_save_all_patterns(self, data_dfs: dict, start_date_str: str = None):
        """
        【V2.31 · 司法探针版】执行所有几何形态的计算和存储。
        - V2.31 探针升级: 增加探针 B.5，在数据送入融合函数前进行最后一次核查。
        """
        print(f"[{self.stock_code}] [动态演化分析] 开始计算几何形态特征...")
        df_daily = data_dfs.get('daily_data')
        if df_daily is None or df_daily.empty:
            print(f"[{self.stock_code}] 日线数据为空，跳过计算。")
            return
        df_daily['trade_time'] = pd.to_datetime(df_daily['trade_time'])
        df_daily = df_daily.set_index('trade_time')
        df_daily.sort_index(inplace=True)
        # --- [探针 A: 初始加载 & 排序后] ---
        if not df_daily.empty:
            latest_day_raw = df_daily.iloc[-1]
            print(f"--- [探针 A: 初始加载 & 排序后] 最新日期: {latest_day_raw.name.date()} ---")
            print(f"  -> high_qfq: {latest_day_raw.get('high_qfq')}, low_qfq: {latest_day_raw.get('low_qfq')}, close_qfq: {latest_day_raw.get('close_qfq')}")
            print(f"--- [探针 A 结束] ---")
        cols_to_convert = ['high_qfq', 'low_qfq', 'close_qfq', 'open_qfq', 'vol']
        for col in cols_to_convert:
            if col in df_daily.columns:
                df_daily[col] = pd.to_numeric(df_daily[col], errors='coerce')
        # --- [探针 B: 类型转换后] ---
        if not df_daily.empty:
            latest_day_converted = df_daily.iloc[-1]
            print(f"--- [探针 B: 类型转换后] 最新日期: {latest_day_converted.name.date()} ---")
            print(f"  -> high_qfq: {latest_day_converted.get('high_qfq')}, low_qfq: {latest_day_converted.get('low_qfq')}, close_qfq: {latest_day_converted.get('close_qfq')}")
            print(f"--- [探针 B 结束] ---")
        df_daily.ta.atr(high='high_qfq', low='low_qfq', close='close_qfq', length=14, append=True, col_names=('ATR_14_D',))
        # [代码新增] 探针 B.5，检查进入融合函数前的最终数据状态
        if not df_daily.empty:
            latest_day_pre_merge = df_daily.iloc[-1]
            print(f"--- [探针 B.5: 融合前最终检查] 最新日期: {latest_day_pre_merge.name.date()} ---")
            print(f"  -> high_qfq: {latest_day_pre_merge.get('high_qfq')}, low_qfq: {latest_day_pre_merge.get('low_qfq')}, close_qfq: {latest_day_pre_merge.get('close_qfq')}")
            print(f"--- [探针 B.5 结束] ---")
        enriched_df = self._prepare_enriched_dataframe(df_daily)
        if start_date_str:
            deleted_count, _ = self.platform_model.objects.filter(
                stock=self.stock_instance,
                start_date__gte=start_date_str
            ).delete()
            print(f"  -> [统一回滚] 平台特征删除 {deleted_count} 条。")
        self._calculate_and_save_platforms(enriched_df, data_dfs)
        self._calculate_and_save_trendline_matrix_and_events(df_daily, data_dfs, start_date_str=start_date_str)
        flag_events = self._predict_flag_breakout_probability(enriched_df, data_dfs)
        if start_date_str:
            self.event_model.objects.filter(
                stock=self.stock_instance,
                event_date__gte=start_date_str,
                event_type='FLAG_FORMED'
            ).delete()
        self._save_trendline_events_incrementally(flag_events)
        print(f"[{self.stock_code}] [动态演化分析] 几何形态特征计算完成。")

    def _calculate_and_save_platforms(self, enriched_df: pd.DataFrame, data_dfs: dict, adx_threshold: int = 25, bbw_quantile: float = 0.25, potential_threshold: float = 0.6, potential_window: int = 20, min_duration: int = 10, max_range_pct: float = 0.30):
        """
        【V2.33 · 直接调用修复版】识别、量化并存储矩形平台。
        - 核心修复: 放弃使用 pandas_ta 的扩展方法 `df.ta.bbands()`，因为它被证实
                     在当前场景下存在不稳定性。改为采用更稳健的直接函数调用方式
                     `ta.bbands(close=df['col'], append=False)`，然后将返回的结果
                     DataFrame 合并回主数据帧。此举绕开了扩展方法中潜在的“黑盒”
                     问题，从根本上解决了布林带计算失败的顽固BUG。
        """
        # --- [探针 E: 进入平台计算函数] ---
        if enriched_df.empty or enriched_df.index.isnull().all():
            print(f"--- [探针 E: 失败] 传入的DataFrame为空或索引损坏，跳过平台计算。 ---")
            return
        latest_day_final = enriched_df.iloc[-1]
        print(f"--- [探针 E: 进入平台计算函数] 最新日期: {latest_day_final.name.date()} ---")
        print(f"  -> high_qfq: {latest_day_final.get('high_qfq')}, low_qfq: {latest_day_final.get('low_qfq')}, close_qfq: {latest_day_final.get('close_qfq')}")
        print(f"--- [探针 E 结束] ---")
        print(f"  -> [V2.33 直接调用] 正在识别和量化矩形平台...")
        if len(enriched_df) < 120:
            print("  -> 数据量不足(<120天)，跳过平台识别。")
            return
        df_copy = enriched_df.copy()
        cols_to_drop = ['open', 'high', 'low', 'close']
        df_copy.drop(columns=[col for col in cols_to_drop if col in df_copy.columns], inplace=True)
        print(f"  -> [歧义消除] 已从计算副本中移除原始OHLC列，确保计算精度。")
        cols_to_convert = ['high_qfq', 'low_qfq', 'close_qfq', 'open_qfq', 'vol']
        for col in cols_to_convert:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        print("\n" + "="*30 + " [深度诊断探针: 计算前] " + "="*30)
        print("打印 df_copy 的完整信息 (df.info)，以检查最终的 dtypes:")
        df_copy.info(verbose=True, show_counts=True)
        print("="*80 + "\n")
        required_cols = ['high_qfq', 'low_qfq', 'close_qfq']
        if not all(col in df_copy.columns for col in required_cols):
            print(f"\n  -> [诊断失败] 输入的DataFrame缺少核心计算列。需要: {required_cols}。任务终止。")
            return
        print("\n    [探针] 核心计算列的空值(NaN)数量:")
        print(df_copy[required_cols].isnull().sum().to_string())
        print("\n  -> [探针式计算] 正在尝试计算 ADX...")
        df_copy.ta.adx(high='high_qfq', low='low_qfq', close='close_qfq', length=14, append=True)
        if 'ADX_14' not in df_copy.columns:
            print("  -> [计算失败] ADX 指标未能成功生成。请检查上游数据是否存在足够的非空值。任务终止。")
            return
        print("  -> [计算成功] ADX 指标已生成。")
        print("  -> [探针式计算] 正在尝试计算 BBands...")
        # [代码修改] V2.33 改为直接函数调用，绕开扩展方法的潜在BUG
        bbands_results = ta.bbands(close=df_copy['close_qfq'], length=20, std=2.0, append=False)
        if bbands_results is not None and not bbands_results.empty:
            df_copy = df_copy.join(bbands_results)
        if 'BBW_20_2.0' not in df_copy.columns:
            print("  -> [计算失败] 布林带宽度(BBW) 指标未能成功生成。请检查 'close_qfq' 列是否存在足够的非空值。任务终止。")
            return
        print("  -> [计算成功] 所有核心指标均已成功生成。继续执行平台识别...")
        is_low_trend = df_copy['ADX_14'] < adx_threshold
        bbw_rolling_quantile = df_copy['BBW_20_2.0'].rolling(120, min_periods=60).quantile(bbw_quantile)
        is_low_volatility = df_copy['BBW_20_2.0'] < bbw_rolling_quantile
        df_copy['is_potential_platform'] = (is_low_trend | is_low_volatility).astype(int)
        df_copy['platform_potential_score'] = df_copy['is_potential_platform'].rolling(window=potential_window, min_periods=potential_window//2).mean()
        score = df_copy['platform_potential_score'].dropna()
        entering_platform = (score > potential_threshold) & (score.shift(1) <= potential_threshold)
        exiting_platform = (score < potential_threshold) & (score.shift(1) >= potential_threshold)
        platform_start_dates = df_copy[entering_platform].index
        platform_end_dates = df_copy[exiting_platform].index
        platforms_to_save = []
        minute_map = data_dfs.get("stock_minute_data_map", {})
        tick_map = data_dfs.get("stock_tick_data_map", {})
        realtime_map = data_dfs.get("stock_realtime_data_map", {})
        for start_date in platform_start_dates:
            possible_end_dates = platform_end_dates[platform_end_dates > start_date]
            if not possible_end_dates.empty:
                end_date = possible_end_dates[0]
                group = df_copy.loc[start_date:end_date]
                if len(group) < min_duration: continue
                platform_high = group['high_qfq'].max()
                platform_low = group['low_qfq'].min()
                if platform_low == 0: continue
                price_range_pct = (platform_high - platform_low) / platform_low
                if price_range_pct > max_range_pct: continue
                character, score_val = self._assess_platform_character(group)
                platform_minutes_dfs = [minute_map[d.date()] for d in group.index if d.date() in minute_map]
                precise_vpoc = None
                if platform_minutes_dfs:
                    platform_minutes_df = pd.concat(platform_minutes_dfs)
                    if not platform_minutes_df.empty and platform_minutes_df['volume'].sum() > 0:
                        precise_vpoc = np.average(platform_minutes_df['close'], weights=platform_minutes_df['volume'])
                internal_ofi_sum = 0
                internal_total_amount = 0
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
                platform_data = {
                    'stock': self.stock_instance, 'start_date': start_date.date(), 'end_date': end_date.date(),
                    'duration': len(group), 'high': platform_high, 'low': platform_low,
                    'vpoc': np.average(group['close_qfq'], weights=group['vol']),
                    'total_volume': group['vol'].sum() * 100, 'quality_score': (score_val + 100) / 200,
                    'precise_vpoc': precise_vpoc, 'internal_accumulation_intensity': internal_accumulation_intensity,
                    'breakout_quality_score': breakout_quality_score, 'platform_character': character, 'character_score': score_val,
                }
                platforms_to_save.append(platform_data)
        found_count = len(platforms_to_save)
        print(f"  -> [V2.33 直接调用] 发现 {found_count} 个有效平台。")
        if found_count > 0:
            print(f"  -> [V2.33 直接调用] 正在将 {found_count} 个平台存入数据库...")
            for data in platforms_to_save:
                self.platform_model.objects.update_or_create(
                    stock=data['stock'], start_date=data['start_date'], defaults=data
                )

    def _calculate_daily_ofi_from_ticks(self, df_tick: pd.DataFrame) -> (float, float):
        """从Tick数据计算日内OFI(订单流失衡)和总成交额"""
        if df_tick is None or df_tick.empty:
            return 0.0, 0.0
        buy_vol = df_tick[df_tick['side'] == 'B']['volume'].sum()
        sell_vol = df_tick[df_tick['side'] == 'S']['volume'].sum()
        total_vol = buy_vol + sell_vol
        if total_vol == 0:
            return 0.0, 0.0
            
        ofi = (buy_vol - sell_vol)
        total_amount = (df_tick['price'] * df_tick['volume']).sum()
        return ofi, total_amount

    def _calculate_and_save_trendline_matrix_and_events(self, df_daily: pd.DataFrame, data_dfs: dict, start_date_str: str = None):
        """
        【V2.17 · 回滚重算版】为给定的全部历史时段，逐日生成趋势线矩阵，并进行全周期动态分析。
        - V2.17 升级: 调用新的上下文初始化方法，该方法会根据 start_date_str 执行数据清理。
        """
        print(f"  -> [增量回溯引擎] 开始检查并计算新的趋势线矩阵...")
        df_daily = df_daily.sort_index(ascending=True)
        # 1. 调用新的上下文初始化方法，它会处理数据清理和日期确定
        start_process_date = self._initialize_incremental_context(start_date_str)
        if start_process_date is None:
            # 如果是首次运行，从头开始
            start_process_date = df_daily.index.min()
        # 筛选出需要处理的日期
        df_to_process = df_daily[df_daily.index >= start_process_date]
        if df_to_process.empty:
            print("  -> [增量回溯引擎] 数据已是最新，无需计算。")
            return
        print(f"  -> [增量回溯引擎] 将处理从 {df_to_process.index.min().date()} 到 {df_to_process.index.max().date()} 的 {len(df_to_process)} 个新交易日。")
        # 2. 仅对新日期进行计算
        new_matrix_records = []
        for current_date in df_to_process.index:
            daily_matrix_records = self._compute_trendline_matrix_for_day(current_date, df_daily, data_dfs)
            new_matrix_records.extend(daily_matrix_records)
        if not new_matrix_records:
            print("  -> [增量回溯引擎] 未生成任何新的趋势线矩阵记录。")
            return
        # 3. 存储新生成的记录
        self._save_trendline_matrix_incrementally(new_matrix_records)
        print(f"  -> [增量回溯引擎] 批量保存了 {len(new_matrix_records)} 条新的趋势线矩阵记录。")
        # 4. 进行增量动态分析
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
        【V2.16 · 增量分析版】分析趋势线矩阵的时间序列，识别动态事件。
        - V2.16 性能优化: 新增 start_analysis_date 参数，使得事件检测循环可以跳过
                         无需重复分析的历史数据，实现高效的增量分析。
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
        # 修改代码行：获取用于循环的DataFrame切片
        analysis_df = matrix_wide_df[matrix_wide_df.index >= start_analysis_date] if start_analysis_date else matrix_wide_df
        for trade_date, today_row in analysis_df.iterrows():
            # 使用完整的 matrix_wide_df 来安全地获取前一天的数据
            yesterday_row = matrix_wide_df.loc[:trade_date].iloc[-2] if len(matrix_wide_df.loc[:trade_date]) > 1 else None
            if yesterday_row is None: continue
            # A. 拐点分析
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
            # B. 交叉分析: 量化多空力量的战略性逆转，而非简单的几何相交
            for short_p, long_p in combinations(self.fib_periods, 2):
                if short_p >= long_p: continue # 确保 short_p < long_p
                # 提取长短周期、今日昨日的四条线数据
                sup_s_t, int_s_t = today_row.get(f'slope_{short_p}_support'), today_row.get(f'intercept_{short_p}_support')
                sup_l_t, int_l_t = today_row.get(f'slope_{long_p}_support'), today_row.get(f'intercept_{long_p}_support')
                sup_s_y, int_s_y = yesterday_row.get(f'slope_{short_p}_support'), yesterday_row.get(f'intercept_{short_p}_support')
                sup_l_y, int_l_y = yesterday_row.get(f'slope_{long_p}_support'), yesterday_row.get(f'intercept_{long_p}_support')
                # 确保所有数据都有效
                if not all(pd.notna(v) for v in [sup_s_t, int_s_t, sup_l_t, int_l_t, sup_s_y, int_s_y, sup_l_y, int_l_y]):
                    continue
                # 检查是否发生交叉 (今天的位置关系与昨天相反)
                pos_today = (sup_s_t - sup_l_t) * time_idx + (int_s_t - int_l_t)
                pos_yesterday = (sup_s_y - sup_l_y) * (time_idx - 1) + (int_s_y - int_l_y)
                if np.sign(pos_today) != np.sign(pos_yesterday):
                    event_type = None
                    # 计算交叉角度 (角度越大，信号越强)
                    angle_short = np.degrees(np.arctan(sup_s_t))
                    angle_long = np.degrees(np.arctan(sup_l_t))
                    intersection_angle = abs(angle_short - angle_long)
                    # 计算收敛速度 (正值表示加速收敛)
                    gap_yesterday = abs(((sup_s_y - sup_l_y) * (time_idx - 1) + (int_s_y - int_l_y)))
                    convergence_rate = gap_yesterday # 今天gap为0，速率即为昨天的距离
                    details = {
                        'periods': f'{short_p}v{long_p}',
                        'angle': intersection_angle,
                        'convergence_rate': convergence_rate
                    }
                    if pos_today > 0: # 短期线上穿长期线 -> 金叉
                        event_type = 'CROSS_GOLDEN_DECISIVE' if intersection_angle > 20 else 'CROSS_GOLDEN_TENTATIVE'
                    else: # 短期线下穿长期线 -> 死叉
                        event_type = 'CROSS_DEATH_DECISIVE' if intersection_angle > 20 else 'CROSS_DEATH_TENTATIVE'
                    if event_type:
                        final_events.append({
                            'stock': self.stock_instance, 'event_date': trade_date.date(),
                            'event_type': event_type, 'details': self._sanitize_json_dict(details)
                        })
            # C. 共振与背离分析: 量化多周期“合力”与“力竭”的结构性信号
            all_periods_slopes = {p: today_row.get(f'slope_{p}_support') for p in self.fib_periods}
            valid_slopes = {p: s for p, s in all_periods_slopes.items() if pd.notna(s)}
            if len(valid_slopes) >= 3: # 至少需要3个周期的数据才有分析意义
                slopes_series = pd.Series(valid_slopes)
                # 1. 共振分析
                concordance_score = ((slopes_series > 0).sum() / len(slopes_series)) # 多头方向协同度
                cohesion_score = 1 / (1 + slopes_series.std()) # 强度凝聚度 (标准差越小，得分越高)
                event_type = None
                details = {'concordance': concordance_score, 'cohesion': cohesion_score}
                if concordance_score >= 0.8 and cohesion_score >= 0.8:
                    event_type = 'RESONANCE_BULLISH_STRONG' # 强烈多头共振
                elif concordance_score <= 0.2 and cohesion_score >= 0.8:
                    event_type = 'RESONANCE_BEARISH_STRONG' # 强烈空头共振
                if event_type:
                    final_events.append({
                        'stock': self.stock_instance, 'event_date': trade_date.date(),
                        'event_type': event_type, 'details': self._sanitize_json_dict(details)
                    })
                # 2. 背离分析 (A股市场的精髓)
                short_term_periods = [p for p in valid_slopes.keys() if p < 21]
                long_term_periods = [p for p in valid_slopes.keys() if p >= 21]
                if short_term_periods and long_term_periods:
                    short_term_slope_mean = slopes_series[short_term_periods].mean()
                    long_term_slope_mean = slopes_series[long_term_periods].mean()
                    # 获取昨日斜率以判断变化
                    yesterday_slopes_series = pd.Series({p: yesterday_row.get(f'slope_{p}_support') for p in valid_slopes.keys()})
                    short_term_slope_change = short_term_slope_mean - yesterday_slopes_series[short_term_periods].mean()
                    event_type = None
                    details = {
                        'short_term_slope': short_term_slope_mean,
                        'long_term_slope': long_term_slope_mean,
                        'short_term_momentum': short_term_slope_change
                    }
                    # 顶背离: 长线向上，短线动能衰竭
                    if long_term_slope_mean > 0.005 and short_term_slope_mean > 0 and short_term_slope_change < -0.001:
                        event_type = 'DIVERGENCE_BEARISH_TOP'
                    # 底背离: 长线向下，短线止跌企稳
                    elif long_term_slope_mean < -0.005 and short_term_slope_mean < 0 and short_term_slope_change > 0.001:
                        event_type = 'DIVERGENCE_BULLISH_BOTTOM'
                    if event_type:
                        final_events.append({
                            'stock': self.stock_instance, 'event_date': trade_date.date(),
                            'event_type': event_type, 'details': self._sanitize_json_dict(details)
                        })
            # D. 通道压缩分析 (逻辑不变, 依然使用完整的 matrix_wide_df 获取历史分位数)
            if len(self.fib_periods) > 0:
                median_period_index = len(self.fib_periods) // 2
                period = self.fib_periods[median_period_index]
                sup_slope, sup_intc = today_row.get(f'slope_{period}_support'), today_row.get(f'intercept_{period}_support')
                res_slope, res_intc = today_row.get(f'slope_{period}_resistance'), today_row.get(f'intercept_{period}_resistance')
                if all(v is not None and pd.notna(v) for v in [sup_slope, sup_intc, res_slope, res_intc]):
                    time_idx = matrix_wide_df.index.get_loc(trade_date)
                    channel_width = (res_slope * time_idx + res_intc) - (sup_slope * time_idx + sup_intc)
                    historical_widths = (matrix_wide_df[f'slope_{period}_resistance'] * np.arange(len(matrix_wide_df)) + matrix_wide_df[f'intercept_{period}_resistance']) - \
                                        (matrix_wide_df[f'slope_{period}_support'] * np.arange(len(matrix_wide_df)) + matrix_wide_df[f'intercept_{period}_support'])
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
        【V2.6 · 自定义Zigzag引擎集成版】为指定的一天，计算出所有斐波那契周期的支撑和阻力线矩阵。
        - V2.6 升级: 废弃不稳定的 pandas_ta.zigzag，改为调用内部实现的 `_calculate_zigzag` 方法。
        """
        is_last_day_debug = (current_date == df_daily.index.max())
        if is_last_day_debug:
            print(f"\n--- [探针模式启动] 正在详细检查日期: {current_date.date()} ---")
        matrix_records = []
        for period in self.fib_periods:
            lookback_days = period * 3
            start_date = current_date - pd.Timedelta(days=lookback_days)
            df_slice = df_daily[(df_daily.index >= start_date) & (df_daily.index <= current_date)].copy()
            if len(df_slice) < period:
                if is_last_day_debug:
                    print(f"  -> [周期 {period}] 数据不足 ({len(df_slice)}天)，跳过。")
                continue
            if is_last_day_debug:
                print(f"\n  -> [周期 {period}] 数据切片长度: {len(df_slice)} (回溯自 {df_slice.index.min().date()}) | 使用自定义Zigzag引擎(threshold=0.05)")
            zigzag_series = self._calculate_zigzag(df_slice, threshold=0.05)
            df_slice['zigzag'] = zigzag_series
            df_slice['zigzag'] = df_slice['zigzag'].fillna(0)
            if is_last_day_debug:
                print(f"    [探针] Zigzag计算结果 (最近10条):")
                print(df_slice[['high_qfq', 'low_qfq', 'zigzag']].tail(10).to_string())
            df_slice['time_idx'] = np.arange(len(df_slice))
            pivot_highs = df_slice[df_slice['zigzag'] == 1]
            pivot_lows = df_slice[df_slice['zigzag'] == -1]
            if is_last_day_debug:
                print(f"    [探针] 识别出的高点锚点(pivot_highs)数量: {len(pivot_highs)}")
                print(f"    [探针] 识别出的低点锚点(pivot_lows)数量: {len(pivot_lows)}")
            best_support = self._find_best_line_with_micro_validation(pivot_lows, 'low_qfq', df_slice, 'support', data_dfs)
            best_resistance = self._find_best_line_with_micro_validation(pivot_highs, 'high_qfq', df_slice, 'resistance', data_dfs)
            for line_data in [best_support, best_resistance]:
                if line_data:
                    matrix_records.append({
                        'stock': self.stock_instance,
                        'trade_date': current_date.date(),
                        'period': period,
                        'line_type': line_data['line_type'],
                        'slope': line_data['slope'],
                        'intercept': line_data['intercept'],
                        'validity_score': line_data['validity_score'],
                    })
        if is_last_day_debug:
            print(f"--- [探針模式結束] 日期 {current_date.date()} 共生成 {len(matrix_records)} 條有效趨勢線 ---\n")
        else:
            print(f"    -> [趋势线矩阵引擎] 为 {current_date.date()} 生成了 {len(matrix_records)} 条有效趋势线。")
        return matrix_records

    def _find_best_line_with_micro_validation(self, pivots: pd.DataFrame, price_col: str, full_df: pd.DataFrame, line_type: str, data_dfs: dict):
        """
        【V2.12 · 锚点剪枝版】通过启发式剪枝策略，从根本上解决组合爆炸导致的性能问题。
        - V2.12 性能修复: 当候选锚点过多时，不再进行暴力组合。而是筛选出 "近期" + "极端"
                         的锚点子集进行计算，将计算复杂度降低数个数量级，彻底解决卡死问题。
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

    def _predict_flag_breakout_probability(self, enriched_df: pd.DataFrame, data_dfs: dict) -> list:
        """
        【V2.3 专家系统】识别旗形，并使用基于规则和加权评分的专家系统进行突破概率预测。
        """
        events = []
        if len(enriched_df) < self.long_term_period:
            return events
        df = enriched_df.copy()
        vol_ma_col_name = f'vol_ma{self.long_term_period}'
        df[vol_ma_col_name] = df['vol'].rolling(self.long_term_period).mean()
        # 动态计算超长期均线
        ultra_long_ma_col_name = f'ma{self.ultra_long_term_period}'
        df[ultra_long_ma_col_name] = df['close_qfq'].rolling(self.ultra_long_term_period).mean()
        # 修改代码行：使用 self.ultra_long_term_period 作为循环的回溯终点和数据长度要求
        if len(df) < self.ultra_long_term_period:
            print(f"[{self.stock_code}] 数据长度不足 {self.ultra_long_term_period} 天，无法进行超长期趋势分析，跳过旗形识别。")
            return []
        for i in range(len(df) - 5, self.ultra_long_term_period, -1):
            pole = self._identify_flagpole(df, end_index_loc=i, vol_ma_col_name=vol_ma_col_name)
            if not pole:
                continue
            flag = self._identify_flag(df, pole, vol_ma_col_name=vol_ma_col_name)
            if not flag:
                continue
            pole_df = df.loc[pole['start_date']:pole['end_date']]
            flag_df = df.loc[flag['start_date']:flag['end_date']]
            is_main_force_led = pole_df['main_force_net_flow_calibrated_sum_5d'].iloc[-1] > 0
            if not is_main_force_led:
                continue
            avg_hidden_accumulation = flag_df['hidden_accumulation_intensity'].mean()
            if avg_hidden_accumulation <= 0:
                continue
            dominant_peak_cost_at_start = flag_df['dominant_peak_cost'].iloc[0]
            if flag_df['low_qfq'].min() < dominant_peak_cost_at_start:
                continue
            breakout_readiness = flag_df['breakout_readiness_score'].iloc[-1]
            if breakout_readiness < 60:
                continue
            print(f"  -> [高置信度旗形] 在 {flag['end_date']} 发现！通过多维情报验证。")
            # 修改代码行：将动态的超长期均线列名传递给概率计算方法
            probability, features = self._calculate_expert_breakout_probability(df, pole, flag, vol_ma_col_name=vol_ma_col_name, ultra_long_ma_col_name=ultra_long_ma_col_name)
            events.append({
                'stock': self.stock_instance,
                'event_date': flag['end_date'].date(),
                'event_type': 'FLAG_FORMED',
                'details': {
                    'probability': probability, 
                    'features': features,
                    'pole_start_date': pole['start_date'].date(),
                    'pole_end_date': pole['end_date'].date(),
                }
            })
            i = df.index.get_loc(pole['start_date'])
        return events

    def _identify_flagpole(self, df: pd.DataFrame, end_index_loc: int, vol_ma_col_name: str, min_dur: int = 2, max_dur: int = 8) -> dict:
        """识别旗杆：寻找一次暴力的、出人意料的、能量充沛的突袭。"""
        for duration in range(min_dur, max_dur + 1):
            start_index_loc = end_index_loc - duration + 1
            if start_index_loc < 0: continue
            pole_df = df.iloc[start_index_loc : end_index_loc + 1]
            atr_at_start = df['ATR_14_D'].iloc[start_index_loc - 1] if start_index_loc > 0 else df['ATR_14_D'].iloc[0]
            if atr_at_start == 0: continue
            pole_high = pole_df['high_qfq'].max()
            pole_low = pole_df['low_qfq'].min()
            magnitude_atr = (pole_high - pole_low) / atr_at_start
            if magnitude_atr < 4.0:
                continue
            # 条件2: 能量充沛 (成交量显著放大)
            # 使用传入的 vol_ma_col_name 进行动态列选择
            vol_ma_at_start = df[vol_ma_col_name].iloc[start_index_loc - 1] if start_index_loc > 0 else df[vol_ma_col_name].iloc[0]
            avg_volume_pole = pole_df['vol'].mean()
            if avg_volume_pole < 1.8 * vol_ma_at_start:
                continue
            if pole_df['close_qfq'].iloc[-1] <= pole_df['open_qfq'].iloc[0]:
                continue
            return {
                'start_date': pole_df.index[0],
                'end_date': pole_df.index[-1],
                'high_price': pole_high,
                'low_price': pole_low,
                'magnitude_atr': magnitude_atr,
                'avg_volume': avg_volume_pole,
            }
        return None

    def _identify_flag(self, df: pd.DataFrame, pole_data: dict, vol_ma_col_name: str, min_dur: int = 5, max_dur: int = 20) -> dict:
        """识别旗面：寻找一次成交极度萎缩、回撤可控的战术性佯退。"""
        pole_end_loc = df.index.get_loc(pole_data['end_date'])
        for duration in range(min_dur, max_dur + 1):
            flag_start_loc = pole_end_loc + 1
            flag_end_loc = flag_start_loc + duration -1
            if flag_end_loc >= len(df): break
            flag_df = df.iloc[flag_start_loc : flag_end_loc + 1]
            # 条件1: 成交极度萎缩 (最核心)
            avg_volume_flag = flag_df['vol'].mean()
            # 修改代码行：使用传入的 vol_ma_col_name 进行动态列选择
            vol_ma_at_flag_start = df[vol_ma_col_name].iloc[flag_start_loc -1]
            if not (avg_volume_flag < 0.7 * pole_data['avg_volume'] and avg_volume_flag < vol_ma_at_flag_start):
                continue
            flag_low = flag_df['low_qfq'].min()
            pole_range = pole_data['high_price'] - pole_data['low_price']
            if pole_range == 0: continue
            retracement_depth = (pole_data['high_price'] - flag_low) / pole_range
            if retracement_depth > 0.5:
                return None
            if flag_df['close_qfq'].max() > pole_data['high_price']:
                continue
            return {
                'start_date': flag_df.index[0],
                'end_date': flag_df.index[-1],
                'duration': duration,
                'avg_volume': avg_volume_flag,
                'retracement_depth': retracement_depth,
            }
        return None

    def _calculate_expert_breakout_probability(self, df: pd.DataFrame, pole: dict, flag: dict, vol_ma_col_name: str, ultra_long_ma_col_name: str) -> (float, dict):
        """
        【V2.3 概率精算】基于一个模拟专家决策的加权评分系统，计算旗形突破的概率。
        """
        import math
        base_probability = 0.55
        log_odds = math.log(base_probability / (1 - base_probability))
        adjustments = {}
        pole_df = df.loc[pole['start_date']:pole['end_date']]
        flag_df = df.loc[flag['start_date']:flag['end_date']]
        if pole['magnitude_atr'] > 6.0:
            adjustments['pole_extreme_magnitude'] = 0.4
        if pole_df['main_force_conviction_index'].is_monotonic_increasing:
            adjustments['pole_mf_conviction_increase'] = 0.3
        if flag['avg_volume'] < 0.5 * df[vol_ma_col_name].loc[flag['start_date']]:
            adjustments['flag_extreme_volume_shrink'] = 0.5
        if flag['retracement_depth'] < 0.236:
            adjustments['flag_shallow_retracement'] = 0.4
        if flag['duration'] > 15:
            adjustments['flag_long_duration'] = -0.3
        if flag_df['main_force_net_flow_calibrated_sum_5d'].iloc[-1] > 0:
            adjustments['flag_mf_accumulation'] = 0.6
        if flag_df['retail_net_flow_calibrated'].mean() > 0:
            adjustments['flag_retail_frontrun'] = -0.4
        dominant_peak_cost = flag_df['dominant_peak_cost'].iloc[0]
        support_margin = (flag_df['low_qfq'].min() - dominant_peak_cost) / dominant_peak_cost if dominant_peak_cost > 0 else 0
        if support_margin > 0.05:
            adjustments['chip_strong_support'] = 0.5
        if flag_df['chip_health_score'].iloc[-1] > 75:
            adjustments['chip_health_high'] = 0.3
        if flag_df['breakout_readiness_score'].iloc[-1] > 80:
            adjustments['structure_high_readiness'] = 0.4
        # 修改代码行：使用传入的 ultra_long_ma_col_name 进行动态列选择
        if flag_df['close_qfq'].iloc[-1] > df[ultra_long_ma_col_name].loc[flag['end_date']]:
            adjustments['long_term_trend_aligned'] = 0.2
        total_adjustment = sum(adjustments.values())
        final_log_odds = log_odds + total_adjustment
        final_probability = 1 / (1 + math.exp(-final_log_odds))
        features = {
            'base_probability': base_probability,
            'pole_height_atr': pole['magnitude_atr'],
            'flag_retracement_pct': flag['retracement_depth'],
            'breakout_readiness_score': flag_df['breakout_readiness_score'].iloc[-1],
            'adjustments': adjustments,
            'total_adjustment': total_adjustment,
            'final_log_odds': final_log_odds
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
            # 修改代码行：同时处理Python原生float和Numpy的浮点类型
            elif isinstance(value, (float, np.floating)):
                if math.isnan(value) or math.isinf(value):
                    clean_data[key] = None  # 替换为None，对应JSON的null
                else:
                    clean_data[key] = value
            else:
                clean_data[key] = value
        return clean_data

    def _save_trendline_matrix_incrementally(self, records: list):
        """【V2.16】持久化存储新增的趋势线矩阵记录。"""
        if not records: return
        print(f"  -> [趋势线矩阵] 正在新增 {len(records)} 条记录...")
        instances = [self.mtt_model(**rec) for rec in records]
        self.mtt_model.objects.bulk_create(instances, ignore_conflicts=True)

    def _save_trendline_events_incrementally(self, events: list):
        """【V2.16】持久化存储新增的趋势线动态事件。"""
        if not events: return
        print(f"  -> [趋势线事件] 正在新增 {len(events)} 个事件...")
        instances = [self.event_model(**evt) for evt in events]
        self.event_model.objects.bulk_create(instances, ignore_conflicts=True)

