# 文件: services/calculators/intraday_dynamics_calculator.py

import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
from stock_models.stock_basic import StockInfo
from stock_models.stock_basic import StockDailyBasic
from stock_models.stock_analytics import IntradayChipDynamics, DailyTurnoverDistribution

class IntradayDynamicsCalculator:
    """
    【V1.0】日内动态指标计算器
    - 职责: 接收单只股票、单个交易日的分钟K线数据，计算所有日内动态指标。
    - 输入: 一个包含分钟K线数据的Pandas DataFrame。
    - 输出: 两个字典，分别对应 IntradayChipDynamics 和 DailyTurnoverDistribution 模型所需的数据。
    - 设计: 这是一个纯计算服务，不包含任何数据库IO操作，便于测试和维护。
    """
    def __init__(self, minute_df: pd.DataFrame, stock_info: StockInfo, daily_basic_info: StockDailyBasic):
        """
        初始化计算器。
        Args:
            minute_df (pd.DataFrame): 包含单日分钟K线的DataFrame，
                                      必须包含 'open', 'high', 'low', 'close', 'vol', 'amount' 列。
            stock_info (StockInfo): 该股票的StockInfo模型实例。
            daily_basic_info (StockDailyBasic): 该股票当日的日线基本信息实例，用于获取流通股本。
        """
        # 1. 数据校验与准备
        if minute_df is None or minute_df.empty:
            raise ValueError("输入的分钟K线DataFrame不能为空。")
        required_cols = {'open', 'high', 'low', 'close', 'vol', 'amount'}
        if not required_cols.issubset(minute_df.columns):
            raise ValueError(f"输入的DataFrame缺少必要列，需要: {required_cols}")
        self.df = minute_df.copy()
        self.stock_info = stock_info
        self.daily_basic_info = daily_basic_info
        # 2. 预处理，确保数据类型正确
        self.df['amount'] = pd.to_numeric(self.df['amount'], errors='coerce')
        self.df['vol'] = pd.to_numeric(self.df['vol'], errors='coerce')
        self.df.dropna(subset=['amount', 'vol'], inplace=True)
        # 3. 初始化结果容器
        self.dynamics_result = {}
        self.distribution_result = {}
    def calculate_all(self) -> tuple[dict, dict]:
        """
        执行所有计算，并返回最终结果。
        这是一个总调度方法。
        """
        print(f"[{self.stock_info.stock_code}] 开始计算日内动态指标...")
        # --- 步骤1: 计算基础核心指标 (VWAP, POC, VA) ---
        self._calculate_vwap_and_turnover()
        volume_profile = self._calculate_volume_profile()
        # --- 步骤2: 计算叙事、属性和形态指标 ---
        self._calculate_narrative_metrics(volume_profile)
        self._calculate_delta_and_shape(volume_profile)
        # --- 步骤3: 准备筹码演化所需的数据 ---
        self._prepare_turnover_distribution(volume_profile)
        print(f"[{self.stock_info.stock_code}] 日内动态指标计算完成。")
        return self.dynamics_result, self.distribution_result
    def _calculate_vwap_and_turnover(self):
        """计算VWAP和总成交量/额"""
        total_volume = self.df['vol'].sum()
        total_amount = self.df['amount'].sum()
        # 计算VWAP，处理成交量为0的边缘情况
        vwap = (total_amount / total_volume) if total_volume > 0 else self.df['close'].iloc[-1]
        self.dynamics_result['vwap'] = Decimal(vwap).quantize(Decimal("0.001"))
        self.dynamics_result['daily_turnover_volume'] = int(total_volume)
        # 从传入的日线基本信息中获取当日流通股本
        if self.daily_basic_info and self.daily_basic_info.float_share:
            # float_share 单位是万股，需转换为股
            self.dynamics_result['total_float_shares_on_day'] = int(self.daily_basic_info.float_share * 10000)
        else:
            self.dynamics_result['total_float_shares_on_day'] = None # 如果没有数据则为None
    def _calculate_volume_profile(self) -> pd.Series:
        """计算成交量分布，并返回POC和VA"""
        # 按收盘价对成交量进行分组，形成成交量分布图
        # 这是所有后续计算的基础
        volume_at_price = self.df.groupby('close')['vol'].sum().sort_index()
        if volume_at_price.empty:
            # 如果没有成交，使用最后一分钟的收盘价作为POC
            last_close = self.df['close'].iloc[-1]
            self.dynamics_result.update({
                'poc_price': Decimal(last_close),
                'value_area_high': Decimal(last_close),
                'value_area_low': Decimal(last_close),
            })
            return pd.Series(dtype='float64')
        # 1. 计算POC (Point of Control) - 成交最密集的价格
        poc_price = volume_at_price.idxmax()
        self.dynamics_result['poc_price'] = Decimal(poc_price)
        # 2. 计算VA (Value Area) - 70%成交量所在的区域
        total_volume = volume_at_price.sum()
        target_volume_70pct = total_volume * 0.7
        poc_index = volume_at_price.index.get_loc(poc_price)
        # 从POC开始向两侧扩展，直到包含70%的成交量
        current_volume = volume_at_price.iloc[poc_index]
        low_idx, high_idx = poc_index, poc_index
        while current_volume < target_volume_70pct and (low_idx > 0 or high_idx < len(volume_at_price) - 1):
            vol_at_left = volume_at_price.iloc[low_idx - 1] if low_idx > 0 else 0
            vol_at_right = volume_at_price.iloc[high_idx + 1] if high_idx < len(volume_at_price) - 1 else 0
            if vol_at_left > vol_at_right:
                current_volume += vol_at_left
                low_idx -= 1
            else:
                current_volume += vol_at_right
                high_idx += 1
        value_area_low = volume_at_price.index[low_idx]
        value_area_high = volume_at_price.index[high_idx]
        self.dynamics_result['value_area_low'] = Decimal(value_area_low)
        self.dynamics_result['value_area_high'] = Decimal(value_area_high)
        return volume_at_price
    def _calculate_narrative_metrics(self, volume_profile: pd.Series):
        """计算时间叙事指标"""
        # 默认值
        self.dynamics_result['opening_drive_type'] = IntradayChipDynamics.DriveType.NEUTRAL
        self.dynamics_result['closing_auction_type'] = IntradayChipDynamics.DriveType.NEUTRAL
        self.dynamics_result['poc_migration_direction'] = IntradayChipDynamics.DriveType.NEUTRAL
        # 开盘驱动 (前30分钟)
        opening_df = self.df.between_time('09:30', '10:00')
        if not opening_df.empty:
            opening_vwap = (opening_df['amount'].sum() / opening_df['vol'].sum()) if opening_df['vol'].sum() > 0 else opening_df['close'].iloc[-1]
            if opening_df['close'].iloc[-1] > opening_vwap * 1.005:
                self.dynamics_result['opening_drive_type'] = IntradayChipDynamics.DriveType.STRONG_BUY
            elif opening_df['close'].iloc[-1] < opening_vwap * 0.995:
                self.dynamics_result['opening_drive_type'] = IntradayChipDynamics.DriveType.STRONG_SELL
        # 尾盘竞价 (后15分钟)
        closing_df = self.df.between_time('14:45', '15:00')
        if not closing_df.empty:
            closing_vwap = (closing_df['amount'].sum() / closing_df['vol'].sum()) if closing_df['vol'].sum() > 0 else closing_df['close'].iloc[-1]
            if closing_df['close'].iloc[-1] > closing_vwap * 1.002:
                self.dynamics_result['closing_auction_type'] = IntradayChipDynamics.DriveType.STRONG_BUY
            elif closing_df['close'].iloc[-1] < closing_vwap * 0.998:
                self.dynamics_result['closing_auction_type'] = IntradayChipDynamics.DriveType.STRONG_SELL
    def _calculate_delta_and_shape(self, volume_profile: pd.Series):
        """计算成交量Delta和分布形态"""
        # 1. 计算成交量Delta
        delta = np.where(self.df['close'] > self.df['open'], self.df['vol'],
                         np.where(self.df['close'] < self.df['open'], -self.df['vol'], 0))
        self.dynamics_result['volume_delta'] = int(delta.sum())
        # 2. 判断价格与CVD背离 (简化版)
        # 仅当价格创日内新高/新低，但Delta为负/正时，标记为背离
        price_trend = np.sign(self.df['close'].iloc[-1] - self.df['open'].iloc[0])
        delta_trend = np.sign(self.dynamics_result['volume_delta'])
        self.dynamics_result['cumulative_delta_divergence'] = (price_trend != 0 and delta_trend != 0 and price_trend != delta_trend)
        # 3. 判断形态结构
        if volume_profile.empty:
            self.dynamics_result['profile_shape_type'] = IntradayChipDynamics.ProfileShape.UNKNOWN
            return
        day_high = self.df['high'].max()
        day_low = self.df['low'].min()
        day_range = day_high - day_low
        poc_price = float(self.dynamics_result['poc_price'])
        if day_range == 0:
            self.dynamics_result['profile_shape_type'] = IntradayChipDynamics.ProfileShape.SLIM
            return
        poc_position_ratio = (poc_price - day_low) / day_range
        if 0.4 <= poc_position_ratio <= 0.6:
            self.dynamics_result['profile_shape_type'] = IntradayChipDynamics.ProfileShape.D_SHAPE
        elif poc_position_ratio > 0.6:
            self.dynamics_result['profile_shape_type'] = IntradayChipDynamics.ProfileShape.B_SHAPE
        else:
            self.dynamics_result['profile_shape_type'] = IntradayChipDynamics.ProfileShape.P_SHAPE
    def _prepare_turnover_distribution(self, volume_profile: pd.Series):
        """准备当日成交分布的JSON数据"""
        if volume_profile.empty:
            self.distribution_result['distribution_data'] = {}
            return
        # 将Series转换为字典，键为价格(str)，值为成交量(int)
        # 必须转换类型以确保JSON序列化兼容性
        distribution_dict = {str(price): int(volume) for price, volume in volume_profile.items()}
        self.distribution_result['distribution_data'] = distribution_dict
