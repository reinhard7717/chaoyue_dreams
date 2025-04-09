# your_project/apps/strategies/base.py
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import numpy as np # 确保导入 numpy

# 定义信号常量 (细化强度)
SIGNAL_STRONG_BUY = 2   # 强烈买入
SIGNAL_BUY = 1          # 买入
SIGNAL_HOLD = 0         # 持有 (或中性，无明确方向)
SIGNAL_SELL = -1        # 卖出
SIGNAL_STRONG_SELL = -2 # 强烈卖出
SIGNAL_NONE = np.nan    # 无信号 / 数据不足 (使用 NaN 更符合 pandas 处理方式)

class BaseStrategy(ABC):
    """
    策略基类，定义所有策略的通用接口。
    """
    strategy_name = "Base Strategy"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        初始化策略。
        :param params: 策略所需的参数字典，例如 {'short_ma': 5, 'long_ma': 20, 'rsi_period': 14}
        """
        self.params = params if params is not None else {}
        self._validate_params() # 验证参数

    def _validate_params(self):
        """
        （可选）验证传入的参数是否满足策略要求。
        可以在子类中重写。
        """
        pass

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        根据输入的指标数据生成交易信号。

        :param data: Pandas DataFrame，包含所有必需的指标数据和价格数据。
                     索引应为时间戳 (datetime)。
                     列名应清晰，例如 'close', 'EMA_5', 'EMA_10', 'RSI_14', 'ADX_14', '+DI_14', '-DI_14' 等。
        :return: Pandas Series，索引与输入DataFrame相同，值为交易信号
                 (SIGNAL_STRONG_BUY, SIGNAL_BUY, SIGNAL_HOLD, SIGNAL_SELL, SIGNAL_STRONG_SELL, SIGNAL_NONE)。
        """
        pass

    def run(self, data: pd.DataFrame) -> pd.Series:
        """
        执行策略并生成信号。
        :param data: 同 generate_signals 的 data 参数。
        :return: 同 generate_signals 的返回值。
        """
        if data.empty:
            return pd.Series(dtype=float) # 返回 float 类型的空 Series 以兼容 NaN

        required_columns = self.get_required_columns()
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            # 检查是否有模式匹配的列，例如 BOLL 列名可能包含参数
            # 这是一个简化检查，实际可能需要更复杂的逻辑来处理带参数的列名
            has_alternatives = False
            if any(col.startswith('BBU_') or col.startswith('BBM_') or col.startswith('BBL_') for col in missing):
                 # 假设如果缺少 BBU_20_2.0，但存在 BBU_ 开头的列，可能只是参数不同，暂时允许
                 # 更健壮的做法是在策略初始化时确定确切的列名
                 pass # 暂时忽略 BOLL 列名不完全匹配的问题，依赖后续代码处理
            elif any(col.startswith('ADX_') or col.startswith('+DI_') or col.startswith('-DI_') for col in missing):
                 pass # 暂时忽略 DMI/ADX 列名周期不匹配的问题
            # ... 可以为其他指标添加类似逻辑 ...
            else:
                 raise ValueError(f"策略 '{self.strategy_name}' 缺少必要的输入数据列: {missing}")


        signals = self.generate_signals(data)
        # 确保返回的是 float 类型以容纳 NaN
        return signals.astype(float)

    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """
        返回该策略运行所必需的数据列名列表。
        列名应尽可能精确，如果策略依赖特定参数的指标（如 EMA_20），应明确指出。
        """
        pass

    def __str__(self):
        return f"{self.strategy_name}(params={self.params})"

    def __repr__(self):
        return self.__str__()

