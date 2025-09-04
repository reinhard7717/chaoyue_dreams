# utils/model_helpers.py
# 存放跨模块使用的、与模型相关的辅助函数

from stock_models.time_trade import (
    StockDailyData_SZ, StockDailyData_SH, StockDailyData_CY, 
    StockDailyData_KC, StockDailyData_BJ, StockCyqChipsCY,
    StockCyqChipsSZ, StockCyqChipsKC, StockCyqChipsSH, StockCyqChipsBJ,
    AdvancedChipMetrics_CY, AdvancedChipMetrics_SZ, AdvancedChipMetrics_KC,
    AdvancedChipMetrics_SH, AdvancedChipMetrics_BJ, StockDailyBasic_CY,
    StockDailyBasic_SZ, StockDailyBasic_KC, StockDailyBasic_SH, StockDailyBasic_BJ,
    StockMinuteData_1_SZ, StockMinuteData_5_SZ, StockMinuteData_15_SZ, StockMinuteData_30_SZ, StockMinuteData_60_SZ,
    StockMinuteData_1_CY, StockMinuteData_5_CY, StockMinuteData_15_CY, StockMinuteData_30_CY, StockMinuteData_60_CY,
    StockMinuteData_1_SH, StockMinuteData_5_SH, StockMinuteData_15_SH, StockMinuteData_30_SH, StockMinuteData_60_SH,
    StockMinuteData_1_KC, StockMinuteData_5_KC, StockMinuteData_15_KC, StockMinuteData_30_KC, StockMinuteData_60_KC,
    StockMinuteData_1_BJ, StockMinuteData_5_BJ, StockMinuteData_15_BJ, StockMinuteData_30_BJ, StockMinuteData_60_BJ,
)
from stock_models.fund_flow import AdvancedFundFlowMetrics_CY, AdvancedFundFlowMetrics_SZ, AdvancedFundFlowMetrics_KC, AdvancedFundFlowMetrics_SH, AdvancedFundFlowMetrics_BJ
from typing import Type, Optional, List, Dict
from datetime import datetime, timezone
from django.db import models

def get_minute_data_model_by_code_and_timelevel(stock_code: str, time_level_str: str) -> Optional[Type[models.Model]]:
    """
    根据股票代码和分钟级别字符串返回对应的分钟线数据分表Model。
    Args:
        stock_code (str): 股票代码，例如 '000001.SZ'。
        time_level_str (str): 分钟级别字符串，例如 '1', '5', '15', '30', '60'。
    Returns:
        Optional[Type[models.Model]]: 对应的Django模型类，如果未找到则为 None。
    """
    if not time_level_str.isdigit():
        print(f"调试信息: 分钟线级别 '{time_level_str}' 必须是数字字符串。")
        return None

    model_map = None
    if stock_code.endswith('.SZ'):
        base_map = {'1': StockMinuteData_1_SZ, '5': StockMinuteData_5_SZ, '15': StockMinuteData_15_SZ, '30': StockMinuteData_30_SZ, '60': StockMinuteData_60_SZ}
        cy_map = {'1': StockMinuteData_1_CY, '5': StockMinuteData_5_CY, '15': StockMinuteData_15_CY, '30': StockMinuteData_30_CY, '60': StockMinuteData_60_CY}
        model_map = cy_map if stock_code.startswith('3') else base_map
    elif stock_code.endswith('.SH'):
        base_map = {'1': StockMinuteData_1_SH, '5': StockMinuteData_5_SH, '15': StockMinuteData_15_SH, '30': StockMinuteData_30_SH, '60': StockMinuteData_60_SH}
        kc_map = {'1': StockMinuteData_1_KC, '5': StockMinuteData_5_KC, '15': StockMinuteData_15_KC, '30': StockMinuteData_30_KC, '60': StockMinuteData_60_KC}
        model_map = kc_map if stock_code.startswith('68') else base_map
    elif stock_code.endswith('.BJ'):
        model_map = {'1': StockMinuteData_1_BJ, '5': StockMinuteData_5_BJ, '15': StockMinuteData_15_BJ, '30': StockMinuteData_30_BJ, '60': StockMinuteData_60_BJ}
    
    if model_map and time_level_str in model_map:
        return model_map[time_level_str]
    else:
        print(f"调试信息: 未能为 {stock_code} 在时间级别 {time_level_str} 找到对应的分钟线数据库模型。")
        return None

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

def get_advanced_fund_flow_metrics_model_by_code(stock_code: str):
    """
    根据股票代码返回对应的高级资金流指标数据表Model
    """
    if stock_code.startswith('3') and stock_code.endswith('.SZ'):
        return AdvancedFundFlowMetrics_CY
    elif stock_code.endswith('.SZ'):
        return AdvancedFundFlowMetrics_SZ
    elif stock_code.startswith('68') and stock_code.endswith('.SH'):
        return AdvancedFundFlowMetrics_KC
    elif stock_code.endswith('.SH'):
        return AdvancedFundFlowMetrics_SH
    elif stock_code.endswith('.BJ'):
        return AdvancedFundFlowMetrics_BJ
    else:
        print(f"未识别的股票代码: {stock_code}，默认使用SZ主板表")
        return AdvancedFundFlowMetrics_SZ  # 默认返回深市主板













