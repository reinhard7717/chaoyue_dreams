# strategies/macd_rsi_kdj_boll_strategy.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

# 从 base 导入基类和信号常量
from .base import BaseStrategy, SIGNAL_STRONG_BUY, SIGNAL_BUY, SIGNAL_HOLD, SIGNAL_SELL, SIGNAL_STRONG_SELL, SIGNAL_NONE

# 假设 core.constants 定义了 TimeLevel 枚举 (虽然这里不直接用，但保持一致性)
# from core.constants import TimeLevel

class MacdRsiKdjBollStrategy(BaseStrategy):
    """
    多时间周期 MACD+RSI+KDJ+BOLL 组合策略。

    策略逻辑:
    1. 以 15 分钟 K 线为主要操作周期。
    2. 结合 5, 15, 30, 60 四个时间周期的 MACD, RSI, KDJ, BOLL 指标。
    3. 为每个指标在每个时间周期生成初步信号 (-1, 0, 1)。
    4. 根据时间周期为信号分配权重 (短周期权重低，长周期权重高)。
    5. 计算加权总分。
    6. 根据总分阈值确定最终的买卖信号强度 (STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL)。
    """
    strategy_name = "MACD_RSI_KDJ_BOLL_MultiTimeframe"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        初始化策略参数。

        可配置参数示例:
        params = {
            'rsi_period': 14,       # RSI 周期 (用于列名查找)
            'rsi_oversold': 30,     # RSI 超卖阈值
            'rsi_overbought': 70,   # RSI 超买阈值
            'kdj_period_k': 9,      # KDJ K周期 (用于列名查找)
            'kdj_period_d': 3,      # KDJ D周期 (用于列名查找)
            'kdj_period_j': 3,      # KDJ J周期 (用于列名查找)
            'kdj_oversold': 20,     # KDJ 超卖阈值
            'kdj_overbought': 80,   # KDJ 超买阈值
            'boll_period': 20,      # BOLL 周期 (用于列名查找)
            'boll_std_dev': 2,      # BOLL 标准差倍数 (用于列名查找)
            'macd_fast': 12,        # MACD 快线周期 (用于列名查找)
            'macd_slow': 26,        # MACD 慢线周期 (用于列名查找)
            'macd_signal': 9,       # MACD 信号线周期 (用于列名查找)
            'weights': {            # 时间周期权重
                '5': 0.1,
                '15': 0.4,
                '30': 0.3,
                '60': 0.2,
            },
            'score_thresholds': {   # 最终得分到信号的映射阈值
                'strong_buy': 1.5,
                'buy': 0.5,
                'sell': -0.5,
                'strong_sell': -1.5,
            }
        }
        """
        default_params = {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'kdj_period_k': 9, # 假设模型字段是 k_value, d_value, j_value
            'kdj_period_d': 3,
            'kdj_period_j': 3,
            'kdj_oversold': 20,
            'kdj_overbought': 80,
            'boll_period': 20, # 假设模型字段是 upper, mid, lower
            'boll_std_dev': 2,
            'macd_fast': 12,   # 假设模型字段是 diff, dea, macd
            'macd_slow': 26,
            'macd_signal': 9,
            'weights': {'5': 0.1, '15': 0.4, '30': 0.3, '60': 0.2},
            'score_thresholds': {'strong_buy': 1.5, 'buy': 0.5, 'sell': -0.5, 'strong_sell': -1.5}
        }
        # 合并默认参数和传入参数
        merged_params = {**default_params, **(params or {})}
        # 深层合并字典参数 (weights, score_thresholds)
        if params and 'weights' in params:
            merged_params['weights'] = {**default_params['weights'], **params['weights']}
        if params and 'score_thresholds' in params:
            merged_params['score_thresholds'] = {**default_params['score_thresholds'], **params['score_thresholds']}
        self.timeframes = ['5', '15', '30', '60'] # 策略使用的时间周期
        super().__init__(merged_params)
        

    def _validate_params(self):
        """验证参数"""
        if not isinstance(self.params['weights'], dict) or \
           not all(tf in self.params['weights'] for tf in self.timeframes) or \
           abs(sum(self.params['weights'].values()) - 1.0) > 1e-6: # 检查权重和是否接近1
            raise ValueError("参数 'weights' 必须是包含所有时间周期 ('5', '15', '30', '60') 且总和为 1.0 的字典")

        if not isinstance(self.params['score_thresholds'], dict) or \
           not all(k in self.params['score_thresholds'] for k in ['strong_buy', 'buy', 'sell', 'strong_sell']):
            raise ValueError("参数 'score_thresholds' 必须是包含 'strong_buy', 'buy', 'sell', 'strong_sell' 键的字典")

        # 可以添加更多对 rsi, kdj, boll, macd 参数的验证

    def get_required_columns(self) -> List[str]:
        """
        返回策略运行所需的 DataFrame 列名。
        假设列名格式为: 'indicatorName_timeframe' 或 'price_timeframe'
        例如: 'close_15', 'macd_5', 'dea_5', 'rsi_14_15', 'k_9_3_3_30', 'upper_20_2_60'
        注意：这里的列名需要与数据准备阶段生成的 DataFrame 列名完全一致！
        """
        required = []
        # 主操作周期的收盘价 (用于 BOLL 比较)
        required.append('close_15') # 假设以 15 收盘价为基准

        for tf in self.timeframes:
            # MACD 列 (假设数据库模型字段为 diff, dea, macd)
            required.extend([f'diff_{tf}', f'dea_{tf}', f'macd_{tf}'])
            # RSI 列 (假设数据库模型字段为 rsi+周期，或需要根据参数构造)
            # 简化处理：假设列名为 rsi_TF (例如 rsi_5, rsi_15)
            # 如果列名包含周期，如 rsi_14_5, 需要调整这里的逻辑或参数
            required.append(f'rsi_{tf}') # 简化假设，实际可能需要 f'rsi_{self.params["rsi_period"]}_{tf}'
            # KDJ 列 (假设数据库模型字段为 k_value, d_value, j_value)
            required.extend([f'k_{tf}', f'd_{tf}', f'j_{tf}']) # 简化假设
            # BOLL 列 (假设数据库模型字段为 upper, mid, lower)
            required.extend([f'upper_{tf}', f'mid_{tf}', f'lower_{tf}']) # 简化假设

        # 去重后返回
        return list(set(required))

    # --- 单个指标信号生成函数 ---

    def _get_macd_signal(self, diff: pd.Series, dea: pd.Series, macd: pd.Series) -> pd.Series:
        """根据 MACD 指标生成初步信号 (1: 金叉, -1: 死叉, 0: 其他)"""
        signal = pd.Series(0, index=diff.index, dtype=int)
        # 金叉: diff 上穿 dea (前一天 diff < dea, 今天 diff > dea)
        # 或 MACD 柱由负转正 (前一天 macd < 0, 今天 macd > 0) - 更常用
        buy_signal = (macd.shift(1) < 0) & (macd > 0)
        # 死叉: diff 下穿 dea (前一天 diff > dea, 今天 diff < dea)
        # 或 MACD 柱由正转负 (前一天 macd > 0, 今天 macd < 0) - 更常用
        sell_signal = (macd.shift(1) > 0) & (macd < 0)

        signal[buy_signal] = 1
        signal[sell_signal] = -1
        return signal

    def _get_rsi_signal(self, rsi: pd.Series) -> pd.Series:
        """根据 RSI 指标生成初步信号 (1: 超卖区反弹, -1: 超买区回调, 0: 其他)"""
        signal = pd.Series(0, index=rsi.index, dtype=int)
        oversold = self.params['rsi_oversold']
        overbought = self.params['rsi_overbought']
        # 进入超卖区后反弹向上 (前一天 < oversold, 今天 > oversold)
        buy_signal = (rsi.shift(1) < oversold) & (rsi > oversold)
        # 进入超买区后回调向下 (前一天 > overbought, 今天 < overbought)
        sell_signal = (rsi.shift(1) > overbought) & (rsi < overbought)

        signal[buy_signal] = 1
        signal[sell_signal] = -1
        return signal

    def _get_kdj_signal(self, k: pd.Series, d: pd.Series, j: pd.Series) -> pd.Series:
        """根据 KDJ 指标生成初步信号 (1: 金叉/超卖区, -1: 死叉/超买区, 0: 其他)"""
        signal = pd.Series(0, index=k.index, dtype=int)
        oversold = self.params['kdj_oversold']
        overbought = self.params['kdj_overbought']

        # KDJ 金叉: K 上穿 D (简化: 今天 K > D, 昨天 K < D)
        # 且 J 值通常小于 50 或在低位
        buy_cross = (k.shift(1) < d.shift(1)) & (k > d) & (j < 50) # 增加 J 值条件过滤

        # KDJ 死叉: K 下穿 D (简化: 今天 K < D, 昨天 K > D)
        # 且 J 值通常大于 50 或在高位
        sell_cross = (k.shift(1) > d.shift(1)) & (k < d) & (j > 50) # 增加 J 值条件过滤

        # 考虑超买超卖区 (J 值更敏感)
        # J < oversold 视为潜在买入机会 (钝化时可能持续)
        # J > overbought 视为潜在卖出风险 (钝化时可能持续)
        # 为了简化，我们主要使用交叉信号，但可以结合 J 值过滤
        # 例如：金叉发生在超卖区附近信号更强，死叉发生在超买区附近信号更强

        signal[buy_cross] = 1
        signal[sell_cross] = -1

        # 也可以考虑 J 值直接进入超买/超卖区作为信号，但容易产生过多信号
        # buy_oversold = (j.shift(1) < oversold) & (j > oversold)
        # sell_overbought = (j.shift(1) > overbought) & (j < overbought)
        # signal[buy_oversold] = signal[buy_oversold].combine_first(pd.Series(1, index=signal.index)) # 合并信号
        # signal[sell_overbought] = signal[sell_overbought].combine_first(pd.Series(-1, index=signal.index))

        return signal

    def _get_boll_signal(self, close: pd.Series, upper: pd.Series, mid: pd.Series, lower: pd.Series) -> pd.Series:
        """根据 BOLL 指标生成初步信号 (1: 突破上轨/下轨支撑, -1: 跌破下轨/上轨压制, 0: 其他)"""
        signal = pd.Series(0, index=close.index, dtype=int)

        # 价格从下轨下方上穿下轨 (下轨支撑)
        buy_support = (close.shift(1) < lower.shift(1)) & (close > lower)
        # 价格从下轨上方上穿中轨 (趋势转强)
        buy_mid_cross = (close.shift(1) < mid.shift(1)) & (close > mid)

        # 价格从上轨上方下穿上轨 (上轨压制)
        sell_pressure = (close.shift(1) > upper.shift(1)) & (close < upper)
        # 价格从上轨下方下穿中轨 (趋势转弱)
        sell_mid_cross = (close.shift(1) > mid.shift(1)) & (close < mid)

        # 简化信号：触及下轨反弹为买入，触及上轨回落为卖出
        signal[buy_support] = 1
        # signal[buy_mid_cross] = 1 # 可以选择是否将上穿中轨也作为买入信号

        signal[sell_pressure] = -1
        # signal[sell_mid_cross] = -1 # 可以选择是否将下穿中轨也作为卖出信号

        return signal

    # --- 主信号生成逻辑 ---

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        根据输入的包含多时间周期指标的 DataFrame 生成最终交易信号。
        """
        signals = pd.DataFrame(index=data.index)
        weights = self.params['weights']
        thresholds = self.params['score_thresholds']

        # 获取主操作周期的收盘价 (用于 BOLL)
        # 确保列名存在，否则 BaseStrategy 的 run 方法会报错
        close_15m = data['close_15']

        # 1. 为每个时间周期计算各指标的初步信号
        for tf in self.timeframes:
            # --- MACD ---
            # 构造列名，需要与 get_required_columns 和实际数据准备一致
            diff_col = f'diff_{tf}'
            dea_col = f'dea_{tf}'
            macd_col = f'macd_{tf}'
            if all(col in data.columns for col in [diff_col, dea_col, macd_col]):
                signals[f'macd_signal_{tf}'] = self._get_macd_signal(data[diff_col], data[dea_col], data[macd_col])
            else:
                signals[f'macd_signal_{tf}'] = 0 # 或 np.nan 如果列不存在

            # --- RSI ---
            rsi_col = f'rsi_{tf}' # 简化假设
            if rsi_col in data.columns:
                signals[f'rsi_signal_{tf}'] = self._get_rsi_signal(data[rsi_col])
            else:
                signals[f'rsi_signal_{tf}'] = 0

            # --- KDJ ---
            k_col = f'k_{tf}'
            d_col = f'd_{tf}'
            j_col = f'j_{tf}'
            if all(col in data.columns for col in [k_col, d_col, j_col]):
                signals[f'kdj_signal_{tf}'] = self._get_kdj_signal(data[k_col], data[d_col], data[j_col])
            else:
                signals[f'kdj_signal_{tf}'] = 0

            # --- BOLL ---
            upper_col = f'upper_{tf}'
            mid_col = f'mid_{tf}'
            lower_col = f'lower_{tf}'
            # BOLL 信号通常与价格比较，这里统一使用 15m 收盘价进行比较
            if all(col in data.columns for col in [upper_col, mid_col, lower_col]):
                 # 注意：BOLL 信号的计算统一使用 15m 收盘价
                signals[f'boll_signal_{tf}'] = self._get_boll_signal(close_15m, data[upper_col], data[mid_col], data[lower_col])
            else:
                signals[f'boll_signal_{tf}'] = 0

        # 2. 计算加权总分
        signals['total_score'] = 0.0
        for tf in self.timeframes:
            # 将该时间周期的所有指标信号相加（简单叠加）
            # 注意：这里假设每个指标信号范围是 [-1, 1]，总和范围可能是 [-4, 4]
            # 可以根据需要调整叠加方式，例如只取最强信号或进行更复杂的组合
            tf_signal_sum = signals[[f'macd_signal_{tf}', f'rsi_signal_{tf}', f'kdj_signal_{tf}', f'boll_signal_{tf}']].sum(axis=1, skipna=True)

            # 乘以该时间周期的权重并累加到总分
            signals['total_score'] += tf_signal_sum * weights[tf]

        # 3. 根据总分映射到最终信号级别
        final_signal = pd.Series(SIGNAL_HOLD, index=data.index) # 默认 HOLD

        final_signal[signals['total_score'] >= thresholds['strong_buy']] = SIGNAL_STRONG_BUY
        final_signal[(signals['total_score'] >= thresholds['buy']) & (signals['total_score'] < thresholds['strong_buy'])] = SIGNAL_BUY
        final_signal[(signals['total_score'] < thresholds['sell']) & (signals['total_score'] >= thresholds['strong_sell'])] = SIGNAL_SELL
        final_signal[signals['total_score'] < thresholds['strong_sell']] = SIGNAL_STRONG_SELL

        # 处理 NaN 值 (例如数据不足的早期行)
        final_signal[signals['total_score'].isna()] = SIGNAL_NONE # 使用 NaN 表示无信号

        # 确保返回 float 类型以容纳 NaN
        return final_signal.astype(float)

