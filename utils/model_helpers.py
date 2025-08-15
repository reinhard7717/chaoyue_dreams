# utils/model_helpers.py
# 存放跨模块使用的、与模型相关的辅助函数

# 【代码新增】将这个函数提取到这里，以供所有模块复用
from stock_models.time_trade import (
    StockDailyData_SZ, StockDailyData_SH, StockDailyData_CY, 
    StockDailyData_KC, StockDailyData_BJ, StockCyqChipsCY,
    StockCyqChipsSZ, StockCyqChipsKC, StockCyqChipsSH, StockCyqChipsBJ,
    AdvancedChipMetrics_CY, AdvancedChipMetrics_SZ, AdvancedChipMetrics_KC,
    AdvancedChipMetrics_SH, AdvancedChipMetrics_BJ, StockDailyBasic_CY,
    StockDailyBasic_SZ, StockDailyBasic_KC, StockDailyBasic_SH, StockDailyBasic_BJ
)


def get_daily_data_model_by_code(stock_code: str):
    """
    【公共辅助函数】
    根据股票代码返回对应的日线数据分表Model。
    Args:
        stock_code (str): 股票代码，例如 '600519.SH'。
    Returns:
        Model Class: 对应的Django模型类。
    """
    if stock_code.startswith('3') and stock_code.endswith('.SZ'):
        return StockDailyData_CY
    elif stock_code.endswith('.SZ'):
        return StockDailyData_SZ
    elif stock_code.startswith('68') and stock_code.endswith('.SH'):
        return StockDailyData_KC
    elif stock_code.endswith('.SH'):
        return StockDailyData_SH
    elif stock_code.endswith('.BJ'):
        return StockDailyData_BJ
    # 提供一个默认返回值，以防有未覆盖到的情况
    return StockDailyData_SZ

def get_cyq_chips_model_by_code(stock_code: str):
    """
    根据股票代码返回对应的筹码分布数据表Model
    """
    if stock_code.startswith('3') and stock_code.endswith('.SZ'):
        return StockCyqChipsCY
    elif stock_code.endswith('.SZ'):
        return StockCyqChipsSZ
    elif stock_code.startswith('68') and stock_code.endswith('.SH'):
        return StockCyqChipsKC
    elif stock_code.endswith('.SH'):
        return StockCyqChipsSH
    elif stock_code.endswith('.BJ'):
        return StockCyqChipsBJ
    else:
        print(f"未识别的股票代码: {stock_code}，默认使用SZ主板表")
        return StockCyqChipsSZ  # 默认返回深市主板

def get_advanced_chip_metrics_model_by_code(stock_code: str):
    """
    根据股票代码返回对应的高级筹码指标数据表Model
    """
    if stock_code.startswith('3') and stock_code.endswith('.SZ'):
        return AdvancedChipMetrics_CY
    elif stock_code.endswith('.SZ'):
        return AdvancedChipMetrics_SZ
    elif stock_code.startswith('68') and stock_code.endswith('.SH'):
        return AdvancedChipMetrics_KC
    elif stock_code.endswith('.SH'):
        return AdvancedChipMetrics_SH
    elif stock_code.endswith('.BJ'):
        return AdvancedChipMetrics_BJ
    else:
        print(f"未识别的股票代码: {stock_code}，默认使用SZ主板表")
        return AdvancedChipMetrics_SZ  # 默认返回深市主板

def get_daily_basic_data_model_by_code(stock_code: str):
    """
    【公共辅助函数】
    根据股票代码返回对应的日线数据分表Model。
    Args:
        stock_code (str): 股票代码，例如 '600519.SH'。
    Returns:
        Model Class: 对应的Django模型类。
    """
    if stock_code.startswith('3') and stock_code.endswith('.SZ'):
        return StockDailyBasic_CY
    elif stock_code.endswith('.SZ'):
        return StockDailyBasic_SZ
    elif stock_code.startswith('68') and stock_code.endswith('.SH'):
        return StockDailyBasic_KC
    elif stock_code.endswith('.SH'):
        return StockDailyBasic_SH
    elif stock_code.endswith('.BJ'):
        return StockDailyBasic_BJ
    # 提供一个默认返回值，以防有未覆盖到的情况
    return StockDailyBasic_SZ














