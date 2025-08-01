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
        剧本1: 动能引爆点突破 (基于真实主动买盘)。
        寻找价格在“异常”主动买盘的推动下，突破关键压力位（VWAP）的信号。
        """
        required_cols = ['vwap', 'price_cv', 'net_agg_vol_zscore', 'corr_price_net_agg_vol']
        if not all(col in columns for col in required_cols):
            return None

        try:
            # 条件1: 价格形态 - 刚刚从下向上突破VWAP
            cond_price = kline['close'] > kline['vwap'] and prev_kline['close'] <= prev_kline['vwap']
            
            # 条件2: 主动力量 - 必须是异常的净主动买盘推动 (Z-Score > 阈值)
            cond_aggression = kline['net_agg_vol_zscore'] > self.params.get('breakout_agg_zscore', 2.0)
            
            # 条件3: 市场状态 - 必须是从盘整/压缩状态中突破
            cond_state = prev_kline['price_cv'] < self.params.get('max_price_cv', 0.005)
            
            # 条件4: 量价关系 - 价格与净主动买盘必须高度正相关
            cond_correlation = kline['corr_price_net_agg_vol'] > self.params.get('min_correlation', 0.6)

            if all([cond_price, cond_aggression, cond_state, cond_correlation]):
                return {
                    "stock_code": kline.get('stock_code'),
                    "entry_time": kline.name,
                    "entry_price": kline['close'],
                    "signal_type": "BUY",
                    "playbook": "Aggressive Breakout",
                    "reason": f"主动买盘突破 @ {kline.name.time()}"
                }
            return None
        except KeyError as ke:
            print(f"      - 错误 (动能突破): 检查条件时缺少键: {ke}")
            return None

    def _check_reversal(self, kline: pd.Series, prev_kline: pd.Series, columns: List[str]) -> Optional[Dict]:
        """
        剧本2: 底部企稳反转 (基于真实主动卖盘衰竭)。
        寻找价格在VWAP下方，主动卖盘力量衰竭，多头开始试探性接盘的信号。
        """
        required_cols = ['vwap', 'net_aggressive_volume', 'net_agg_vol_slope']
        if not all(col in columns for col in required_cols):
            return None

        try:
            # 条件1: 市场位置 - 价格处于VWAP下方
            cond_price = kline['close'] < kline['vwap']
            
            # 条件2: 力量反转 - 净主动成交量刚刚由负转正，表明多头开始占据优势
            cond_force = kline['net_aggressive_volume'] > 0 and prev_kline['net_aggressive_volume'] <= 0
                      
            # 条件3: 趋势确认 - 净主动成交量的趋势（斜率）正在向上
            cond_trend = kline['net_agg_vol_slope'] > 0

            if all([cond_price, cond_force, cond_trend]):
                return {
                    "stock_code": kline.get('stock_code'),
                    "entry_time": kline.name,
                    "entry_price": kline['close'],
                    "signal_type": "BUY",
                    "playbook": "Exhaustion Reversal",
                    "reason": f"卖盘衰竭反转 @ {kline.name.time()}"
                }
            return None
        except KeyError as ke:
            print(f"      - 错误 (势能反转): 检查条件时缺少键: {ke}")
            return None












