# 文件: services/realtime_services.py

import logging
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Optional
from datetime import datetime
from dao_manager.tushare_daos.realtime_data_dao import StockRealtimeDAO
from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO


logger = logging.getLogger("services")

class RealtimeServices:
    """
    【盘中引擎 - 服务层 V4.1 - 最终形态】
    - 核心能力: 将原始Tick和分钟K线，通过深度解析、聚合和高级数学计算，
                转化为一个包含力学、统计学、分形等多维度特征的战术情报矩阵。
    - 技术栈: Pandas, Numpy, Pandas-TA
    """
    def __init__(self):
        self.realtime_dao = StockRealtimeDAO()
        self.timetrade_dao = StockTimeTradeDAO()
        self.slope_window = 5
        self.stats_window = 20

    async def prepare_intraday_data(self, stock_code: str, time_level: str, trade_date: str) -> Optional[pd.DataFrame]:
        """
        为盘中策略准备所有需要的数据。
        """
        print(f"    -> [实时服务层 V4.1] 正在为 {stock_code} on {trade_date} 生成战术情报矩阵...")
        
        # --- 1. 获取基础数据 ---
        # 1.1 获取分钟K线 (基础画布)
        start_dt = datetime.strptime(trade_date, '%Y-%m-%d').replace(hour=1, minute=29)
        end_dt = datetime.strptime(trade_date, '%Y-%m-%d').replace(hour=7, minute=1)
        df_minute = await self.timetrade_dao.get_minute_kline_by_daterange(
            stock_code, time_level, start_dt, end_dt
        )
        if df_minute is None or df_minute.empty:
            logger.warning(f"未能从数据库获取 {stock_code} 的 {time_level}分钟 K线数据。")
            return None

        # 1.2 获取全天Ticks (原始颜料)
        df_ticks = await self.realtime_dao.get_daily_ticks_from_cache(stock_code, trade_date)

        # --- 2. 聚合Tick数据，生成分钟级盘口特征 ---
        if df_ticks is not None and not df_ticks.empty and 'buy_volume1' in df_ticks.columns:
            df_ticks = self._calculate_tick_level_indicators(df_ticks)
            aggregation_rules = self._get_aggregation_rules()
            df_aggregated = df_ticks.resample(f'{time_level}T').agg(aggregation_rules)
            df_aggregated.columns = ['_'.join(col).strip() for col in df_aggregated.columns.values]
            df_aggregated = self._rename_aggregated_columns(df_aggregated)
            df_minute = df_minute.join(df_aggregated, how='left')
        
        # --- 3. 计算基础盘中指标 (如VWAP) ---
        if 'turnover_value' in df_minute.columns and 'volume' in df_minute.columns:
            df_minute['vwap'] = df_minute['turnover_value'].cumsum() / (df_minute['volume'].cumsum() + 1e-6)
        
        # --- 4. 【核心】计算所有高级衍生特征 ---
        df_minute = self._calculate_advanced_features_with_ta(df_minute)

        print(f"    -> [实时服务层 V4.1] 战术情报矩阵生成完毕，共 {len(df_minute)} 条记录。")
        return df_minute

    def _calculate_tick_level_indicators(self, df_ticks: pd.DataFrame) -> pd.DataFrame:
        """(私有方法) 计算所有Tick级别的衍生指标，包括高级力学指标"""
        buy_cols = [f'buy_volume{i}' for i in range(1, 6)]
        sell_cols = [f'sell_volume{i}' for i in range(1, 6)]
        buy_price_cols = [f'buy_price{i}' for i in range(1, 6)]
        sell_price_cols = [f'sell_price{i}' for i in range(1, 6)]
        
        df_ticks['total_buy_volume'] = df_ticks[buy_cols].sum(axis=1)
        df_ticks['total_sell_volume'] = df_ticks[sell_cols].sum(axis=1)
        
        # 质心
        df_ticks['buy_com'] = sum(df_ticks[p] * df_ticks[v] for p, v in zip(buy_price_cols, buy_cols)) / (df_ticks['total_buy_volume'] + 1e-6)
        df_ticks['sell_com'] = sum(df_ticks[p] * df_ticks[v] for p, v in zip(sell_price_cols, sell_cols)) / (df_ticks['total_sell_volume'] + 1e-6)
        
        # 势能
        axis_price = df_ticks['current_price']
        df_ticks['buy_potential_energy'] = sum((axis_price - df_ticks[p]) * df_ticks[v] for p, v in zip(buy_price_cols, buy_cols))
        df_ticks['sell_potential_energy'] = sum((df_ticks[p] - axis_price) * df_ticks[v] for p, v in zip(sell_price_cols, sell_cols))
        df_ticks['energy_ratio'] = df_ticks['buy_potential_energy'] / (df_ticks['sell_potential_energy'] + 1e-6)
        
        # 主动性成交
        df_ticks['tick_volume'] = df_ticks['volume'].diff().fillna(0)
        df_ticks['aggressive_buy_volume'] = np.where(df_ticks['current_price'] >= df_ticks['sell_price1'].shift(1), df_ticks['tick_volume'], 0)
        df_ticks['aggressive_sell_volume'] = np.where(df_ticks['current_price'] <= df_ticks['buy_price1'].shift(1), df_ticks['tick_volume'], 0)
        
        return df_ticks

    def _get_aggregation_rules(self) -> Dict:
        """(私有方法) 定义所有聚合规则"""
        return {
            'buy_com': 'mean',
            'sell_com': 'mean',
            'energy_ratio': ['mean', 'max', 'min'],
            'aggressive_buy_volume': 'sum',
            'aggressive_sell_volume': 'sum',
            'volume': ['sum', 'count'], # 保留成交量sum和成交笔数count
        }

    def _rename_aggregated_columns(self, df_agg: pd.DataFrame) -> pd.DataFrame:
        """(私有方法) 重命名聚合后的列"""
        df_agg.rename(columns={
            'buy_com_mean': 'buy_com_mean',
            'sell_com_mean': 'sell_com_mean',
            'energy_ratio_mean': 'energy_ratio_mean',
            'energy_ratio_max': 'energy_ratio_max',
            'energy_ratio_min': 'energy_ratio_min',
            'aggressive_buy_volume_sum': 'agg_buy_vol_sum',
            'aggressive_sell_volume_sum': 'agg_sell_vol_sum',
            'volume_sum': 'volume', # 将聚合后的成交量总和重命名为 'volume'
            'volume_count': 'tick_count',
        }, inplace=True)
        return df_agg

    def _calculate_advanced_features_with_ta(self, df: pd.DataFrame) -> pd.DataFrame:
        """【V4.1 - 核心计算模块】使用 pandas_ta 统一计算所有高级衍生指标"""
        if df.empty: return df
        
        # 准备基础数据列
        if 'agg_buy_vol_sum' in df.columns and 'agg_sell_vol_sum' in df.columns:
            df['net_aggressive_volume'] = df['agg_buy_vol_sum'] - df['agg_sell_vol_sum']
        df['price_pct_change'] = df.ta.percent_return(length=1)

        # 定义并执行 pandas_ta 策略
        custom_strategy = ta.Strategy(
            name="Intraday_Advanced_Features",
            ta=[
                {"kind": "slope", "close": "net_aggressive_volume", "length": self.slope_window, "col_names": "net_agg_vol_slope"},
                {"kind": "slope", "close": "buy_com_mean", "length": self.slope_window, "col_names": "buy_com_slope"},
                {"kind": "slope", "close": "energy_ratio_mean", "length": self.slope_window, "col_names": "energy_ratio_slope"},
                {"kind": "zscore", "close": "volume", "length": self.stats_window, "col_names": "volume_zscore"},
                {"kind": "zscore", "close": "net_aggressive_volume", "length": self.stats_window, "col_names": "net_agg_vol_zscore"},
                {"kind": "ema", "close": "net_aggressive_volume", "length": 10, "col_names": "net_agg_vol_ema10"},
                {"kind": "fractal", "col_names": ("fractal_low", "fractal_high")}
            ]
        )
        df.ta.strategy(custom_strategy)

        # 计算需要二次处理或手动组合的指标
        if 'net_agg_vol_slope' in df.columns:
            df['net_agg_vol_accel'] = df.ta.slope(close=df['net_agg_vol_slope'], length=self.slope_window)
        
        if 'net_aggressive_volume' in df.columns:
            bbands_df = df.ta.bbands(close=df['net_aggressive_volume'], length=self.stats_window, col_names=('BBL', 'BBM', 'BBU', 'BBB', 'BBP'))
            df = df.join(bbands_df)
        
        stdev = df.ta.stdev(length=self.stats_window)
        sma = df.ta.sma(length=self.stats_window)
        df['price_cv'] = stdev / (sma + 1e-6)
        
        if 'net_aggressive_volume' in df.columns:
            df['corr_price_net_agg_vol'] = df['price_pct_change'].rolling(self.stats_window).corr(df['net_aggressive_volume'])
            
        return df
