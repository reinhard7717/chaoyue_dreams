# 文件: services/realtime_services.py

import logging
import pandas as pd
import numpy as np
import pandas_ta as ta
import pytz # <--- 1. 导入 pytz
from typing import Dict, Optional
from datetime import datetime, time # <--- 2. 导入 time
from datetime import datetime
from dao_manager.tushare_daos.realtime_data_dao import StockRealtimeDAO
from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO
from utils.cache_manager import CacheManager


logger = logging.getLogger("services")

class RealtimeServices:
    """
    【盘中引擎 - 服务层 V4.1 - 最终形态】
    - 核心能力: 将原始Tick和分钟K线，通过深度解析、聚合和高级数学计算，
                转化为一个包含力学、统计学、分形等多维度特征的战术情报矩阵。
    - 技术栈: Pandas, Numpy, Pandas-TA
    """
    def __init__(self, cache_manager_instance: CacheManager):
        # 【核心修复】接收 cache_manager_instance
        
        # 使用传入的实例来创建 DAO
        self.realtime_dao = StockRealtimeDAO(cache_manager_instance)
        self.timetrade_dao = StockTimeTradeDAO(cache_manager_instance)
        self.slope_window = 5
        self.stats_window = 20

    async def prepare_intraday_data(self, stock_code: str, time_level: str, trade_date: str) -> Optional[pd.DataFrame]:
        """
        为盘中策略准备所有需要的数据。
        """
        print(f"    -> [实时服务层 V4.2] 正在为 {stock_code} on {trade_date} 生成战术情报矩阵...")
        
        # --- 1. 获取基础数据 ---
        # 1.1 获取分钟K线 (基础画布)
        try:
            shanghai_tz = pytz.timezone('Asia/Shanghai')
            target_date_obj = datetime.strptime(trade_date, '%Y-%m-%d').date()
            market_open_time = time(9, 25, 0) 
            market_close_time = time(15, 5, 0)
            start_naive = datetime.combine(target_date_obj, market_open_time)
            end_naive = datetime.combine(target_date_obj, market_close_time)
            start_dt_aware = shanghai_tz.localize(start_naive)
            end_dt_aware = shanghai_tz.localize(end_naive)
        except Exception as e:
            logger.error(f"为 {stock_code} 构建时间范围时出错: {e}", exc_info=True)
            return None

        df_minute = await self.timetrade_dao.get_minute_kline_by_daterange(
            stock_code, time_level, start_dt_aware, end_dt_aware
        )
        
        if df_minute is None or df_minute.empty:
            logger.warning(f"未能在数据库获取 {stock_code} 在 {trade_date} 的 {time_level}分钟 K线数据。")
            return None

        numeric_cols = ['open', 'close', 'high', 'low', 'volume', 'turnover_value']
        for col in numeric_cols:
            if col in df_minute.columns:
                df_minute[col] = pd.to_numeric(df_minute[col], errors='coerce')

        # 1.2 获取全天Ticks (原始颜料)
        df_ticks = await self.realtime_dao.get_daily_ticks_from_cache(stock_code, trade_date)

        # --- 2. 聚合Tick数据，生成分钟级盘口特征 ---
        if df_ticks is not None and not df_ticks.empty and 'buy_volume1' in df_ticks.columns:
            # 新增: 解决 "Cannot join tz-naive with tz-aware DatetimeIndex" 错误
            # 检查 df_ticks 的索引是否为 "时区朴素" (naive)
            if df_ticks.index.tz is None:
                print(f"DEBUG: 为 {stock_code} 的 Ticks 索引添加 'Asia/Shanghai' 时区信息。")
                # 将其本地化为上海时区，与 df_minute 的时区保持一致
                df_ticks.index = df_ticks.index.tz_localize(shanghai_tz)
            
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

        print(f"    -> [实时服务层 V4.2] 战术情报矩阵生成完毕，共 {len(df_minute)} 条记录。")
        return df_minute

    # MODIFIED: 修改此方法以增加数据类型转换
    def _calculate_tick_level_indicators(self, df_ticks: pd.DataFrame) -> pd.DataFrame:
        """(私有方法) 计算所有Tick级别的衍生指标，包括高级力学指标"""
        # 新增: 强制类型转换，防止 Decimal 和 float 混合运算错误
        # 这是解决 'unsupported operand type(s) for /: 'decimal.Decimal' and 'float'' 问题的关键
        price_cols = [f'{prefix}_price{i}' for prefix in ['buy', 'sell'] for i in range(1, 6)] + ['current_price']
        volume_cols = [f'{prefix}_volume{i}' for prefix in ['buy', 'sell'] for i in range(1, 6)] + ['volume']
        all_cols_to_convert = price_cols + volume_cols
        for col in all_cols_to_convert:
            if col in df_ticks.columns:
                # 使用 pd.to_numeric 保证所有数值列都转换为Python可计算的浮点数类型
                df_ticks[col] = pd.to_numeric(df_ticks[col], errors='coerce')

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
        """【V4.3 - 核心计算模块 - Celery兼容版】使用 pandas_ta 并强制单进程计算"""
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
            ]
        )
        # MODIFIED: 新增 cores=1 参数，强制单进程运行，防止在Celery中创建子进程
        df.ta.strategy(custom_strategy, cores=1)

        # 计算需要二次处理或手动组合的指标
        if 'net_agg_vol_slope' in df.columns:
            # MODIFIED: 新增 cores=1 参数
            df['net_agg_vol_accel'] = df.ta.slope(close=df['net_agg_vol_slope'], length=self.slope_window, cores=1)
        
        if 'net_aggressive_volume' in df.columns:
            # MODIFIED: 新增 cores=1 参数
            bbands_df = df.ta.bbands(close=df['net_aggressive_volume'], length=self.stats_window, col_names=('BBL', 'BBM', 'BBU', 'BBB', 'BBP'), cores=1)
            df = df.join(bbands_df)
        
        # MODIFIED: 新增 cores=1 参数
        stdev = df.ta.stdev(length=self.stats_window, cores=1)
        sma = df.ta.sma(length=self.stats_window, cores=1)
        df['price_cv'] = stdev / (sma + 1e-6)
        
        if 'net_aggressive_volume' in df.columns:
            df['corr_price_net_agg_vol'] = df['price_pct_change'].rolling(self.stats_window).corr(df['net_aggressive_volume'])
            
        print(f"DEBUG: 准备为股票直接计算分形指标...")
        try:
            # MODIFIED: 新增 cores=1 参数
            df.ta.fractal(append=True, cores=1)
            rename_map = {'FRACTAL_low_2': 'fractal_low', 'FRACTAL_high_2': 'fractal_high'}
            df.rename(columns=rename_map, inplace=True)
            print(f"DEBUG: 分形指标计算并重命名成功。")
        except Exception as e:
            stock_code_for_log = df['stock_code'].iloc[0] if not df.empty and 'stock_code' in df.columns else "未知股票"
            print(f"错误: 为 {stock_code_for_log} 直接调用分形指标时发生异常: {e}")
            logger.error(f"为 {stock_code_for_log} 直接调用分形指标时发生异常: {e}", exc_info=True)
            
        return df