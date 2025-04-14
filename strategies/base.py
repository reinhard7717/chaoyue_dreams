# your_project/apps/strategies/base.py
import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union # 导入 Union
import numpy as np # 确保导入 numpy

# 定义信号常量 (细化强度) - 这些仍然有用，用于返回离散信号的策略
SIGNAL_STRONG_BUY = 2   # 强烈买入
SIGNAL_BUY = 1          # 买入
SIGNAL_HOLD = 0         # 持有 (或中性，无明确方向)
SIGNAL_SELL = -1        # 卖出
SIGNAL_STRONG_SELL = -2 # 强烈卖出
SIGNAL_NONE = np.nan    # 无信号 / 数据不足 (使用 NaN 更符合 pandas 处理方式)

# 定义返回类型别名，增加可读性
SignalType = Union[int, float] # 信号可以是整数常量或浮点数分数

logger = logging.getLogger("strategy")

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
        根据输入的指标数据生成交易信号或评分。

        :param data: Pandas DataFrame，包含所有必需的指标数据和价格数据。
                     索引应为时间戳 (datetime)。
                     列名应清晰，例如 'close', 'EMA_5', 'EMA_10', 'RSI_14', 'ADX_14', '+DI_14', '-DI_14' 等。
        :return: Pandas Series，索引与输入DataFrame相同。
                 值可以是：
                 1. 离散的交易信号常量 (例如 SIGNAL_BUY, SIGNAL_SELL, SIGNAL_HOLD, SIGNAL_NONE)。
                 2. 连续的浮点数分数 (例如 0-100)，表示信号强度或置信度 (例如 100=最强买入, 0=最强卖出, 50=中性)。
                 注意：无论哪种情况，返回的 Series 最终会被转换为 float 类型以处理 SIGNAL_NONE (np.nan)。
        """
        logger.info(f"BaseStrategy.generate_signals called for {self.strategy_name}") # 稍微修改日志信息
        pass

    def run(self, data: pd.DataFrame) -> pd.Series:
        """
        执行策略并生成信号或评分。

        :param data: 同 generate_signals 的 data 参数。
        :return: Pandas Series，索引与输入DataFrame相同，值为 generate_signals 返回的信号或评分。
                 类型保证为 float，以兼容 np.nan。
        """
        if data.empty:
            # 返回 float 类型的空 Series 以兼容 NaN 和评分
            return pd.Series(dtype=float)

        required_columns = self.get_required_columns()
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            # 检查是否有模式匹配的列，例如 BOLL 列名可能包含参数
            # 这是一个简化检查，实际可能需要更复杂的逻辑来处理带参数的列名
            has_alternatives = False
            # --- 保持原有的列检查逻辑 ---
            if any(col.startswith('BBU_') or col.startswith('BBM_') or col.startswith('BBL_') for col in missing):
                 pass
            elif any(col.startswith('ADX_') or col.startswith('+DI_') or col.startswith('-DI_') for col in missing):
                 pass
            # ... 可以为其他指标添加类似逻辑 ...
            else:
                # 考虑是否真的要 raise error，或者只是记录警告并尝试运行
                logger.warning(f"策略 '{self.strategy_name}' 可能缺少必要的输入数据列: {missing}. 尝试继续执行...")
                # raise ValueError(f"策略 '{self.strategy_name}' 缺少必要的输入数据列: {missing}")
                pass # 允许继续执行，generate_signals 内部应能处理列缺失

        # 调用具体策略的 generate_signals，它可能返回离散信号或连续分数
        signals_or_scores = self.generate_signals(data)

        # 确保返回的是 float 类型以容纳 NaN 或浮点数分数
        # 如果 signals_or_scores 已经是 float，此操作无副作用
        # 如果是包含整数信号和 NaN 的 Series，会转换为 float
        return signals_or_scores.astype(float)

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

