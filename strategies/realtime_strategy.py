# 文件: strategies/realtime_strategy.py

import logging
import pandas as pd
from typing import Dict, Optional, List
from datetime import time
import traceback

logger = logging.getLogger("strategy")

class RealtimeStrategy:
    """
    盘中决策引擎。
    基于对Tushare数据源的正确理解，使用可靠的量价指标和五档盘口压力指标，
    执行多个交易剧本，寻找高置信度的盘中交易机会。
    """
    def __init__(self, params: Dict):
        """
        初始化策略，所有阈值都应通过参数传入，便于调优。
        
        Args:
            params (Dict): 包含所有策略参数的字典。
        """
        self.params = params
        # 默认交易时间窗口
        self.trade_start_time = params.get('trade_start_time', time(9, 45))
        self.trade_end_time = params.get('trade_end_time', time(14, 50))

    def run_strategy(self, df_intraday: pd.DataFrame, daily_signal_info: Dict) -> Optional[Dict]:
        """
        对单个股票执行所有交易剧本，寻找第一个满足条件的入场信号。
        
        Args:
            df_intraday (pd.DataFrame): 包含分钟级别特征的战术情报矩阵。
            daily_signal_info (Dict): 盘后信号的关键信息。

        Returns:
            Optional[Dict]: 如果找到入场信号，返回包含详细信息的字典；否则返回None。
        """
        stock_code = daily_signal_info.get('stock_code', 'Unknown')
        print(f"    -> [盘中策略引擎] 开始对 {stock_code} 执行多剧本分析...")
        
        # 忽略数据不足的情况
        if len(df_intraday) < self.params.get('min_data_points', 21):
            print(f"      - 提示: {stock_code} 数据点不足 ({len(df_intraday)})，跳过分析。")
            return None

        # 遍历每一根分钟K线，寻找交易机会
        for i in range(1, len(df_intraday)):
            try:
                current_kline = df_intraday.iloc[i]
                prev_kline = df_intraday.iloc[i-1]
                
                # 检查是否在有效交易时间内
                if not (self.trade_start_time <= current_kline.name.time() <= self.trade_end_time):
                    continue

                # --- 按优先级执行剧本 ---
                # 剧本1: 动能引爆点突破
                breakout_signal = self._check_momentum_breakout(current_kline, prev_kline, df_intraday.columns)
                if breakout_signal:
                    print(f"      - [信号触发!] {breakout_signal['reason']}")
                    return breakout_signal

                # 剧本2: 底部企稳反转
                reversal_signal = self._check_reversal(current_kline, prev_kline, df_intraday.columns)
                if reversal_signal:
                    print(f"      - [信号触发!] {reversal_signal['reason']}")
                    return reversal_signal
            
            except Exception as e:
                print(f"      - 错误: 在处理 {stock_code} 第 {i} 行K线时发生异常!")
                traceback.print_exc()
                # 发生一次错误后，为安全起见，终止对该股票的分析
                return None
        
        print(f"    -> [盘中策略引擎] {stock_code} 分析完成，未触发任何信号。")
        return None

    def _check_momentum_breakout(self, kline: pd.Series, prev_kline: pd.Series, columns: List[str]) -> Optional[Dict]:
        """
        剧本1: 动能引爆点突破。
        寻找价格在成交量配合下，突破关键压力位（VWAP），且上方抛压减弱的信号。
        """
        # 定义此剧本必需的列
        required_cols = ['vwap', 'volume_zscore', 'price_cv', 'sell_pressure_slope']
        if not all(col in columns for col in required_cols):
            return None # 如果缺少任何一列，则不执行此剧本

        try:
            # 条件1: 价格形态 - 刚刚从下向上突破VWAP
            cond_price = kline['close'] > kline['vwap'] and prev_kline['close'] <= prev_kline['vwap']
            
            # 条件2: 成交量 - 必须是异常放量
            cond_volume = kline['volume_zscore'] > self.params.get('breakout_vol_zscore', 2.5)
            
            # 条件3: 市场状态 - 必须是从盘整/压缩状态中突破
            cond_state = prev_kline['price_cv'] < self.params.get('max_price_cv', 0.005)
            
            # 条件4: 盘口压力 - 突破时上方抛压正在减弱（委卖总量斜率为负）
            cond_pressure = kline['sell_pressure_slope'] < 0

            # --- 共振检查 ---
            if all([cond_price, cond_volume, cond_state, cond_pressure]):
                return {
                    "stock_code": kline.get('stock_code'),
                    "entry_time": kline.name,
                    "entry_price": kline['close'],
                    "signal_type": "BUY",
                    "playbook": "Momentum Breakout",
                    "reason": f"动能突破 @ {kline.name.time()}"
                }
            return None
        except KeyError as ke:
            # 这个异常理论上不应再发生，因为有前置检查，但保留以增加健壮性
            print(f"      - 错误 (动能突破): 检查条件时缺少键: {ke}")
            return None

    def _check_reversal(self, kline: pd.Series, prev_kline: pd.Series, columns: List[str]) -> Optional[Dict]:
        """
        剧本2: 底部企稳反转。
        寻找价格在VWAP下方缩量企稳，且下方承接意愿（委买盘）开始增强的信号。
        """
        # 定义此剧本必需的列
        required_cols = ['vwap', 'volume_zscore', 'buy_pressure_slope']
        if not all(col in columns for col in required_cols):
            return None # 如果缺少任何一列，则不执行此剧本

        try:
            # 条件1: 市场位置 - 价格必须处于VWAP下方，寻找的是底部反转
            cond_price = kline['close'] < kline['vwap']
            
            # 条件2: 趋势衰竭 - 下跌过程成交量萎缩，表明抛售动能减弱
            cond_exhaustion = kline['volume_zscore'] < self.params.get('reversal_vol_zscore', -0.5)
                      
            # 条件3: 盘口支撑 - 下方承接盘（委买总量）趋势刚刚由降转升
            cond_support = kline['buy_pressure_slope'] > 0 and prev_kline['buy_pressure_slope'] <= 0

            # --- 共振检查 ---
            if all([cond_price, cond_exhaustion, cond_support]):
                return {
                    "stock_code": kline.get('stock_code'),
                    "entry_time": kline.name,
                    "entry_price": kline['close'],
                    "signal_type": "BUY",
                    "playbook": "Potential Reversal",
                    "reason": f"势能反转 @ {kline.name.time()}"
                }
            return None
        except KeyError as ke:
            # 这个异常理论上不应再发生，但保留以增加健壮性
            print(f"      - 错误 (势能反转): 检查条件时缺少键: {ke}")
            return None
