# your_project/apps/strategies/base.py
import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import numpy as np

# 定义信号常量 (如果其他策略可能使用离散信号)
SIGNAL_STRONG_BUY = 2
SIGNAL_BUY = 1
SIGNAL_HOLD = 0
SIGNAL_SELL = -1
SIGNAL_STRONG_SELL = -2
SIGNAL_NONE = np.nan # 使用 NaN 表示无信号或数据不足
SignalType = Union[int, float] # 信号可以是整数或浮点数

logger = logging.getLogger("strategy") # 使用您原来的 logger 名称 "strategy"

class BaseStrategy(ABC):
    """
    策略基类，定义所有策略的通用接口。
    """
    strategy_name = "Base Strategy" # 默认策略名称，子类应覆盖
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        初始化策略。
        :param params: 策略所需的参数字典。
        """
        self.params = params if params is not None else {}
        # 移除这里的 self._validate_params() 调用
        # self._validate_params() # 验证参数 - 应该由子类在完成自身初始化后调用
    def _validate_params(self):
        """
        （可选）验证传入的参数是否满足策略要求。
        可以在子类中重写。
        """
        pass # 子类可以实现具体的验证逻辑
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, stock_code: Optional[str] = None, indicator_configs: Optional[List[Dict]] = None, **kwargs) -> pd.DataFrame: # 修改返回类型为 DataFrame
        """
        根据输入的指标数据生成交易信号和所有中间计算结果。
        :param data: Pandas DataFrame，包含所有必需的指标数据和价格数据。
        :param stock_code: 股票代码，可选。
        :param indicator_configs: 指标配置列表，可选。
        :param kwargs: 其他特定于任务的参数。
        :return: Pandas DataFrame，索引与输入DataFrame相同。
                 应包含原始数据、所有计算的中间列，以及最终的信号列(例如 'final_rule_signal', 'combined_signal')。
        """
        pass
    def run(self, data: pd.DataFrame, stock_code: Optional[str] = None, indicator_configs: Optional[List[Dict]] = None, **kwargs) -> pd.DataFrame: # 修改返回类型为 DataFrame
        """
        执行策略并生成包含所有信号和中间结果的DataFrame。
        :param data: 同 generate_signals 的 data 参数。
        :param stock_code: 股票代码。
        :param indicator_configs: 指标配置列表。
        :param kwargs: 其他传递给 generate_signals 的参数。
        :return: Pandas DataFrame，包含所有计算结果。
        """
        if data.empty:
            logger.warning(f"[{self.strategy_name}] (股票: {stock_code}) 输入数据为空，无法执行策略 run 方法。")
            return pd.DataFrame() # 返回空的 DataFrame
        # 在 run 方法中再次检查参数是否已加载和验证（可选，但安全）
        if not self.params:
             logger.error(f"[{self.strategy_name}] (股票: {stock_code}) 策略参数未加载，无法执行 run 方法。")
             return pd.DataFrame()
        # 理论上 __init__ 应该已经调用了 _validate_params，但这里可以加一个检查
        # self._validate_params() # 确保参数在 run 之前是有效的，但如果 __init__ 失败，这里会再次失败
        required_columns = self.get_required_columns()
        if required_columns: # 仅当 get_required_columns 返回非空列表时才检查
            missing = [col for col in required_columns if col not in data.columns]
            if missing:
                # 保持原有的列检查逻辑，但允许继续
                logger.warning(f"策略 '{self.strategy_name}' (股票: {stock_code}) 可能缺少其声明的必要输入数据列: {missing}. 尝试继续执行...")
                # 注意：子策略的 generate_signals 内部仍需处理可能的列缺失。
        else:
            logger.debug(f"策略 '{self.strategy_name}' (股票: {stock_code}) 未声明任何必需列 (get_required_columns 返回空列表)。")
        # 调用具体策略的 generate_signals，传递所有相关参数
        # 现在期望 generate_signals 返回一个包含所有数据的 DataFrame
        processed_data_with_signals = self.generate_signals(data, stock_code=stock_code, indicator_configs=indicator_configs, **kwargs)
        if not isinstance(processed_data_with_signals, pd.DataFrame):
            logger.error(f"策略 '{self.strategy_name}' (股票: {stock_code}) 的 generate_signals 方法未按预期返回 Pandas DataFrame，而是返回了 {type(processed_data_with_signals)}。")
            # 可以选择是返回空 DataFrame 还是尝试转换，或抛出异常
            # 为了安全，返回空 DataFrame
            return pd.DataFrame()
        return processed_data_with_signals
    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """
        返回该策略运行所必需的数据列名列表。
        """
        pass
    def __str__(self):
        return f"{self.strategy_name}(params={self.params})"
    def __repr__(self):
        return self.__str__()

