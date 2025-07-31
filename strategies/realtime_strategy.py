# 文件: strategies/realtime_strategy.py
import logging
import traceback
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import time

logger = logging.getLogger("strategy")

class RealtimeStrategy:
    """
    【盘中决策引擎 V3.0 - 多剧本共振版】
    - 核心能力: 基于 RealtimeServices 输出的多维度战术情报矩阵，
                执行多个交易剧本，寻找高置信度的盘中交易机会。
    - 剧本库:
      1. 动能引爆点突破 (Momentum Ignition Breakout)
      2. 势能反转抄底 (Potential Energy Reversal)
    """
    def __init__(self, params: Dict):
        """
        初始化策略，所有阈值都应通过参数传入，便于调优。
        """
        self.params = params
        # 默认交易时间窗口
        self.trade_start_time = params.get('trade_start_time', time(9, 45))
        self.trade_end_time = params.get('trade_end_time', time(14, 50))

    def run_strategy(self, df_intraday: pd.DataFrame, daily_signal_info: Dict) -> Optional[Dict]:
        """
        对单个股票执行所有交易剧本，寻找第一个满足条件的入场信号。
        """
        stock_code = daily_signal_info.get('stock_code', 'Unknown')
        print(f"    -> [盘中策略引擎 V3.1] 开始对 {stock_code} 执行多剧本共振分析...")
        
        # ▼▼▼ 调试：检查传入的DataFrame ▼▼▼
        print(f"      - 调试: 传入DataFrame的行数: {len(df_intraday)}")
        if df_intraday.empty:
            print("      - 调试: DataFrame为空，直接返回。")
            return None
        print(f"      - 调试: DataFrame的列: {df_intraday.columns.tolist()}")
        print(f"      - 调试: DataFrame的索引类型: {type(df_intraday.index)}")
        # ▲▲▲ 调试结束 ▲▲▲

        if len(df_intraday) < self.params.get('min_data_points', 21):
            print(f"      - 调试: 数据点 ({len(df_intraday)}) 少于最小要求 ({self.params.get('min_data_points', 21)})，跳过。")
            return None

        # 遍历每一根分钟K线，寻找交易机会
        for i in range(1, len(df_intraday)):
            try:
                current_kline = df_intraday.iloc[i]
                prev_kline = df_intraday.iloc[i-1]
                
                # ▼▼▼ 调试：检查当前K线的时间 ▼▼▼
                kline_time = current_kline.name
                if not hasattr(kline_time, 'time'):
                    print(f"      - 调试错误: 第 {i} 行的索引 '{kline_time}' (类型: {type(kline_time)}) 没有 .time() 方法。循环终止。")
                    return None
                # ▲▲▲ 调试结束 ▲▲▲

                if not (self.trade_start_time <= kline_time.time() <= self.trade_end_time):
                    continue

                # --- 按优先级执行剧本 ---
                breakout_signal = self._check_momentum_breakout(current_kline, prev_kline)
                if breakout_signal:
                    print(f"      - [信号触发!] {breakout_signal['reason']}")
                    return breakout_signal

                reversal_signal = self._check_potential_reversal(current_kline, prev_kline)
                if reversal_signal:
                    print(f"      - [信号触发!] {reversal_signal['reason']}")
                    return reversal_signal
            
            except Exception as e:
                # ▼▼▼ 调试：捕获循环中的任何异常 ▼▼▼
                print(f"      - 调试错误: 在处理第 {i} 行K线时发生异常!")
                print(f"        - 异常类型: {type(e)}")
                print(f"        - 异常信息: {e}")
                print(f"        - 堆栈跟踪:")
                traceback.print_exc()
                # 发生一次错误后就没必要继续了，直接返回
                return None
                # ▲▲▲ 调试结束 ▲▲▲
        
        print(f"    -> [盘中策略引擎 V3.1] {stock_code} 分析完成，未触发任何信号。")
        return None

    def _check_momentum_breakout(self, kline: pd.Series, prev_kline: pd.Series) -> Optional[Dict]:
        """剧本1: 动能引爆点突破"""
        
        # --- 定义共振条件 ---
        # 条件1: 价格形态 - 刚刚从下向上突破VWAP
        cond_price = kline['close'] > kline['vwap'] and prev_kline['close'] <= prev_kline['vwap']
        
        # 条件2: 成交量 - 必须是异常放量
        cond_volume = kline['volume_zscore'] > self.params.get('breakout_vol_zscore', 2.5)
        
        # 条件3: 主动性攻击 - 必须由超预期的主动买盘驱动
        cond_aggression = kline['net_agg_vol_zscore'] > self.params.get('breakout_agg_zscore', 2.0)
        
        # 条件4: 力学趋势 - 攻击力道必须在增强
        cond_dynamics = kline['net_agg_vol_slope'] > 0
        
        # 条件5: 市场状态 - 必须是从盘整/压缩状态中突破
        cond_state = prev_kline['price_cv'] < self.params.get('max_price_cv', 0.005)
        
        # 条件6: 健康度 - 价格上涨必须与主动买盘高度相关
        cond_health = kline['corr_price_net_agg_vol'] > self.params.get('min_correlation', 0.6)

        # --- 共振检查 ---
        if all([cond_price, cond_volume, cond_aggression, cond_dynamics, cond_state, cond_health]):
            return {
                "stock_code": kline.get('stock_code'), # 假设DataFrame中有此列
                "entry_time": kline.name,
                "entry_price": kline['close'],
                "signal_type": "BUY",
                "playbook": "Momentum Ignition Breakout",
                "reason": f"动能引爆点突破 @ {kline.name.time()}"
            }
        return None

    def _check_potential_reversal(self, kline: pd.Series, prev_kline: pd.Series) -> Optional[Dict]:
        """剧本2: 势能反转抄底"""
        
        # --- 定义共振条件 ---
        # 条件1: 市场位置 - 价格必须处于VWAP下方，寻找的是底部反转
        cond_price = kline['close'] < kline['vwap']
        
        # 条件2: 势能结构 - 下方支撑能量开始占据优势，且趋势向好
        cond_energy = kline['energy_ratio_mean'] > self.params.get('reversal_energy_ratio', 1.5) and \
                      kline['energy_ratio_slope'] > 0
                      
        # 条件3: 防御重心 - 多头核心防御阵地开始主动上移
        cond_defense = kline['buy_com_slope'] > 0 and prev_kline['buy_com_slope'] <= 0
        
        # 条件4: 内部力量 - 净主动性成交量刚刚由负转正，或上穿其EMA线
        cond_force = kline['net_aggressive_volume'] > kline['net_agg_vol_ema10'] and \
                     prev_kline['net_aggressive_volume'] <= prev_kline['net_agg_vol_ema10']
                     
        # 条件5: 趋势衰竭 - （可选，但强大）赫斯特指数显示下跌趋势记忆性消失
        # cond_hurst = kline['price_hurst'] < 0.5 if 'price_hurst' in kline else True

        # --- 共振检查 ---
        if all([cond_price, cond_energy, cond_defense, cond_force]):
            return {
                "stock_code": kline.get('stock_code'),
                "entry_time": kline.name,
                "entry_price": kline['close'],
                "signal_type": "BUY",
                "playbook": "Potential Energy Reversal",
                "reason": f"势能反转抄底 @ {kline.name.time()}"
            }
        return None
