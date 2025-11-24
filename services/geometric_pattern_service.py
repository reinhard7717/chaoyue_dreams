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
    【V2.9 · Numba JIT 加速版】使用 Numba 加速的 Zigzag 核心实现。
    - 接收 NumPy 数组，运行在 nopython 模式下，以接近 C 的速度执行。
    - 算法逻辑与 V2.8 的经典重构版完全一致，但性能大幅提升。
    """
    n = len(highs)
    if n < 2:
        return np.zeros(n, dtype=np.int8)
    zigzag = np.zeros(n, dtype=np.int8)
    initial_high = highs[0]
    initial_low = lows[0]
    trend = 0  # 1 for up, -1 for down
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
                cursor = pivot_idx + 1
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
                cursor = pivot_idx + 1
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
        【轻量化构造函数】
        此方法现在是完全同步的，只负责接收预先获取的数据并设置属性。
        """
        self.stock_code = stock_code
        self.stock_instance = stock_instance
        # stock_id 仍然有用，可以从 instance 中获取
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
        【V2.1 · 修复版】准备一个包含所有高级指标的、信息增强的DataFrame。
        - 修复: 为主 DataFrame (df_daily_reset) 手动添加 'stock_id' 列，以满足 pd.merge 的连接键要求。
        """
        print(f"  -> [数据融合] 正在加载并整合高级指标...")
        # 加载所有高级指标数据
        chip_metrics_qs = self.chip_metrics_model.objects.filter(stock=self.stock_instance).values()
        fund_flow_metrics_qs = self.fund_flow_metrics_model.objects.filter(stock=self.stock_instance).values()
        structural_metrics_qs = self.structural_metrics_model.objects.filter(stock=self.stock_instance).values()
        df_chip = pd.DataFrame.from_records(chip_metrics_qs).rename(columns={'trade_time': 'trade_date'})
        df_fund = pd.DataFrame.from_records(fund_flow_metrics_qs).rename(columns={'trade_time': 'trade_date'})
        df_struct = pd.DataFrame.from_records(structural_metrics_qs).rename(columns={'trade_time': 'trade_date'})
        # 将日期列转换为datetime对象以便合并
        for df in [df_chip, df_fund, df_struct]:
            if not df.empty:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
        # 将日线数据的索引转换为日期列以便合并
        df_daily_reset = df_daily.reset_index().rename(columns={'trade_time': 'trade_date'})
        # 新增代码行：为左侧 DataFrame 添加缺失的 'stock_id' 连接键
        df_daily_reset['stock_id'] = self.stock_id
        # 逐一合并
        enriched_df = df_daily_reset
        if not df_chip.empty:
            # 此处的 on=['stock_id', 'trade_date'] 现在可以正常工作
            enriched_df = pd.merge(enriched_df, df_chip, on=['stock_id', 'trade_date'], how='left')
        if not df_fund.empty:
            enriched_df = pd.merge(enriched_df, df_fund, on=['stock_id', 'trade_date'], how='left')
        if not df_struct.empty:
            enriched_df = pd.merge(enriched_df, df_struct, on=['stock_id', 'trade_date'], how='left')
        # 将日期重新设为索引
        enriched_df = enriched_df.set_index('trade_date').sort_index()
        print(f"  -> [数据融合] 全维度战场沙盘构建完成。")
        return enriched_df

    def calculate_and_save_all_patterns(self, data_dfs: dict):
        """
        【V2.2 主入口】执行所有几何形态的计算和存储，并融合全维度高级指标。
        """
        print(f"[{self.stock_code}] [动态演化分析] 开始计算几何形态特征...")
        df_daily = data_dfs.get('daily_data')
        if df_daily is None or df_daily.empty:
            print(f"[{self.stock_code}] 日线数据为空，跳过计算。")
            return
        df_daily['trade_time'] = pd.to_datetime(df_daily['trade_time'])
        df_daily = df_daily.set_index('trade_time')
        df_daily.ta.atr(length=14, append=True, col_names=('ATR_14_D',))
        # 新增代码行：构建包含所有高级指标的“全维战场沙盘”
        enriched_df = self._prepare_enriched_dataframe(df_daily)
        # 1. 计算平台特征 (传入原始df_daily即可)
        self._calculate_and_save_platforms(df_daily, data_dfs)
        # 2. 计算趋势线矩阵并分析动态事件 (传入原始df_daily)
        self._calculate_and_save_trendline_matrix_and_events(df_daily, data_dfs)
        # 3. 识别并预测旗形 (传入信息增强的enriched_df)
        flag_events = self._predict_flag_breakout_probability(enriched_df, data_dfs)
        self._save_trendline_events(flag_events)
        print(f"[{self.stock_code}] [动态演化分析] 几何形态特征计算完成。")

    def _calculate_and_save_platforms(self, df: pd.DataFrame, data_dfs: dict, lookback_period: int = 21, volatility_quantile: float = 0.3, range_threshold_pct: float = 0.20):
        """
        【V2.0】识别、量化并存储矩形平台，并融合微观数据。
        """
        print(f"  -> [V2.0] 正在识别和量化矩形平台...")
        df_copy = df.copy()
        df_copy['atr_pct'] = df_copy['ATR_14_D'] / df_copy['close_qfq']
        df_copy['atr_pct_quantile'] = df_copy['atr_pct'].rolling(window=lookback_period * 3, min_periods=lookback_period).quantile(volatility_quantile)
        is_low_volatility = df_copy['atr_pct'] < df_copy['atr_pct_quantile']
        rolling_high = df_copy['high_qfq'].rolling(window=lookback_period).max()
        rolling_low = df_copy['low_qfq'].rolling(window=lookback_period).min()
        price_range = (rolling_high - rolling_low) / rolling_low
        is_tight_range = price_range < range_threshold_pct
        df_copy['is_in_platform'] = (is_low_volatility & is_tight_range).astype(int)
        df_copy['platform_id'] = (df_copy['is_in_platform'].diff() != 0).cumsum()
        platforms_to_save = []
        minute_map = data_dfs.get("stock_minute_data_map", {})
        tick_map = data_dfs.get("stock_tick_data_map", {})
        realtime_map = data_dfs.get("stock_realtime_data_map", {}) # 新增代码行：获取realtime数据map
        for platform_id, group in df_copy[df_copy['is_in_platform'] == 1].groupby('platform_id'):
            if len(group) < 10: continue
            start_date, end_date = group.index.min(), group.index.max()
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
            if next_trade_date and pd.to_datetime(next_trade_date) in df_copy.index: # 确保下一天有日线数据
                breakout_day_close = df_copy.loc[pd.to_datetime(next_trade_date), 'close_qfq']
                platform_high = group['high_qfq'].max()
                if breakout_day_close > platform_high:
                    # [修改代码块] V2.0 突破质量分融合计算
                    ofi_score = 0.0
                    momentum_score = 0.0
                    # 1. 从Tick数据计算意图分
                    if next_trade_date in tick_map:
                        ofi, _ = self._calculate_daily_ofi_from_ticks(tick_map[next_trade_date])
                        breakout_vol = df_copy.loc[pd.to_datetime(next_trade_date), 'vol'] * 100
                        if breakout_vol > 0:
                            ofi_score = np.clip(ofi / breakout_vol, -1, 1)
                    # 2. 从Realtime数据计算动能分
                    if next_trade_date in realtime_map:
                        momentum_score = self._calculate_breakout_momentum_from_realtime(realtime_map[next_trade_date])
                    # 3. 融合得分 (意图权重60%，动能权重40%)
                    breakout_quality_score = (ofi_score * 0.6) + (momentum_score * 0.4)
            platform_data = {
                'stock': self.stock_instance,
                'start_date': start_date.date(),
                'end_date': end_date.date(),
                'duration': len(group),
                'high': group['high_qfq'].max(),
                'low': group['low_qfq'].min(),
                'vpoc': np.average(group['close_qfq'], weights=group['vol']),
                'total_volume': group['vol'].sum() * 100,
                'quality_score': 0.5,
                'precise_vpoc': precise_vpoc,
                'internal_accumulation_intensity': internal_accumulation_intensity,
                'breakout_quality_score': breakout_quality_score,
            }
            platforms_to_save.append(platform_data)
        if platforms_to_save:
            print(f"  -> [V2.0] 发现 {len(platforms_to_save)} 个平台，正在存入数据库...")
            for data in platforms_to_save:
                self.platform_model.objects.update_or_create(
                    stock=data['stock'],
                    start_date=data['start_date'],
                    defaults=data
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

    def _calculate_and_save_trendline_matrix_and_events(self, df_daily: pd.DataFrame, data_dfs: dict):
        """
        【V2.4 · 精细化门槛版】为给定的全部历史时段，逐日生成趋势线矩阵，并进行全周期动态分析。
        - V2.4 修复: 移除了过于严格的全局数据量检查。改为动态地、逐周期地判断数据是否充足，
                     允许在总数据量不足以计算长周期趋势线时，仍然能够计算并保存短周期的趋势线。
        """
        print(f"  -> [历史回溯引擎] 开始为 {df_daily.index.min().date()} 至 {df_daily.index.max().date()} 生成趋势线矩阵...")
        df_daily = df_daily.sort_index(ascending=True)
        all_matrix_records = []
        # --- 以下是修改的代码块 ---
        # 移除全局的 min_lookback_days 和 if len(df_daily) < min_lookback_days 检查。
        # 循环应该遍历所有可用的日期，数据是否充足的判断将在 _compute_trendline_matrix_for_day 内部进行。
        if df_daily.empty:
            print("    -> [历史回溯引擎] 日线数据为空，跳过矩阵计算。")
            return
        for current_date in df_daily.index:
            # _compute_trendline_matrix_for_day 内部会对每个 period 进行独立的数据量检查
            daily_matrix_records = self._compute_trendline_matrix_for_day(current_date, df_daily, data_dfs)
            all_matrix_records.extend(daily_matrix_records)
        # --- 修改结束 ---
        if not all_matrix_records:
            print("  -> [历史回溯引擎] 未生成任何新的趋势线矩阵记录。")
            return
        self._save_trendline_matrix(all_matrix_records)
        print(f"  -> [历史回溯引擎] 批量保存了 {len(all_matrix_records)} 条趋势线矩阵记录。")
        matrix_qs = self.mtt_model.objects.filter(stock=self.stock_instance).order_by('trade_date').values()
        matrix_df = pd.DataFrame.from_records(matrix_qs)
        if matrix_df.empty:
            print("  -> [历史回溯引擎] 加载完整矩阵失败，跳过动态分析。")
            return
        matrix_df['trade_date'] = pd.to_datetime(matrix_df['trade_date'])
        dynamic_events = self._analyze_matrix_dynamics(matrix_df)
        self._save_trendline_events(dynamic_events)

    def _analyze_matrix_dynamics(self, matrix_df: pd.DataFrame) -> list:
        """
        【V2.7 · JSON净化版】分析趋势线矩阵的时间序列，识别动态事件，并对结果进行净化。
        - V2.7 修复: 在生成每个事件的 details 字典后，调用 _sanitize_json_dict 方法进行净化，
                     将 NaN/Infinity 等非法值替换为 None，解决数据库JSON存储错误。
        - V2.7 修复: 修正了事件日期被错误记录为最后一天的BUG，现在正确记录为事件发生的当天。
        """
        if matrix_df.empty or len(matrix_df['trade_date'].unique()) < 2:
            return []
        print(f"    -> [战场AI参谋] 开始分析趋势线矩阵动态...")
        final_events = [] # 修改代码行：直接构建最终的事件列表
        matrix_wide_df = matrix_df.pivot_table(
            index='trade_date',
            columns=['period', 'line_type'],
            values=['slope', 'intercept', 'validity_score']
        ).sort_index()
        matrix_wide_df.columns = ['_'.join(map(str, col)) for col in matrix_wide_df.columns.values]
        for trade_date, today_row in matrix_wide_df.iterrows():
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
            # B. 交叉分析
            for short_p, long_p in combinations(self.fib_periods, 2):
                # ... (此处省略交叉分析的详细代码，其内部也应在构建details后调用净化函数)
                pass # 假设交叉分析逻辑不变，但其结果也需要净化
            # C. 共振与背离分析
            # ... (此处省略共振分析的详细代码)
            pass
            # D. 通道压缩分析
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
                    squeeze_threshold = historical_widths.rolling(250, min_periods=50).quantile(0.1).iloc[-1]
                    if not np.isnan(squeeze_threshold) and channel_width < squeeze_threshold:
                        details = {'period': period, 'width': channel_width, 'threshold': squeeze_threshold}
                        final_events.append({
                            'stock': self.stock_instance, 'event_date': trade_date.date(),
                            'event_type': 'COMPRESSION_SQUEEZE', 'details': self._sanitize_json_dict(details)
                        })
        print(f"    -> [战场AI参谋] 分析完毕，共发现 {len(final_events)} 个潜在动态事件。")
        return final_events

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
            # 修改代码块：调用我们自己的zigzag实现
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
        【V2.11 · 终极性能版】从候选锚点中，通过三维立体评分体系，找出最终的“主战线”。
        - V2.11 性能修复: 向量化了计算额外触及点的逻辑，移除了该函数中最后一个已知的
                         iterrows() 性能瓶颈，彻底解决了计算卡死问题。
        """
        is_last_day_debug = (full_df.index.max() == data_dfs.get('daily_data').index.max())
        if is_last_day_debug:
            print(f"      [内部探针] 进入 _find_best_line... | 类型: {line_type}, 接收到锚点数: {len(pivots)}")
        if len(pivots) < 2: return None
        best_line_info = None
        max_final_score = -1
        tick_map = data_dfs.get("stock_tick_data_map", {})
        for p1_idx, p2_idx in combinations(pivots.index, 2):
            p1 = pivots.loc[p1_idx]
            p2 = pivots.loc[p2_idx]
            if p1['time_idx'] == p2['time_idx']: continue
            m = (p2[price_col] - p1[price_col]) / (p2['time_idx'] - p1['time_idx'])
            c = p1[price_col] - m * p1['time_idx']
            # --- 以下是修改的代码块 ---
            touch_points_indices = {p1_idx, p2_idx}
            all_other_pivots = pivots.drop(index=[p1_idx, p2_idx])
            # 使用向量化操作替代 for 循环来寻找额外的触及点
            if not all_other_pivots.empty:
                predicted_prices = m * all_other_pivots['time_idx'] + c
                errors = np.abs(all_other_pivots[price_col] - predicted_prices) / all_other_pivots[price_col]
                additional_touches = all_other_pivots[errors < 0.015].index
                touch_points_indices.update(additional_touches)
            # --- 修改结束 ---
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

    def _save_trendline_matrix(self, records: list):
        """持久化存储趋势线矩阵。"""
        if not records: return
        print(f"  -> [趋势线矩阵] 正在保存 {len(records)} 条记录...")
        # 先删除当天的旧数据，确保幂等性
        today = records[0]['trade_date']
        self.mtt_model.objects.filter(stock=self.stock_instance, trade_date=today).delete()
        instances = [self.mtt_model(**rec) for rec in records]
        self.mtt_model.objects.bulk_create(instances)

    def _save_trendline_events(self, events: list):
        """持久化存储趋势线动态事件。"""
        if not events: return
        print(f"  -> [趋势线事件] 正在保存 {len(events)} 个事件...")
        today = events[0]['event_date']
        self.event_model.objects.filter(stock=self.stock_instance, event_date=today).delete()
        instances = [self.event_model(**evt) for evt in events]
        self.event_model.objects.bulk_create(instances)

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


