# utils/model_helpers.py
# 存放跨模块使用的、与模型相关的辅助函数

from stock_models.time_trade import (
    StockDailyData_SZ, StockDailyData_SH, StockDailyData_CY, 
    StockDailyData_KC, StockDailyData_BJ,  StockDailyBasic_CY,
    StockDailyBasic_SZ, StockDailyBasic_KC, StockDailyBasic_SH, StockDailyBasic_BJ,
    StockMinuteData_1_SZ, StockMinuteData_5_SZ, StockMinuteData_15_SZ, StockMinuteData_30_SZ, StockMinuteData_60_SZ,
    StockMinuteData_1_CY, StockMinuteData_5_CY, StockMinuteData_15_CY, StockMinuteData_30_CY, StockMinuteData_60_CY,
    StockMinuteData_1_SH, StockMinuteData_5_SH, StockMinuteData_15_SH, StockMinuteData_30_SH, StockMinuteData_60_SH,
    StockMinuteData_1_KC, StockMinuteData_5_KC, StockMinuteData_15_KC, StockMinuteData_30_KC, StockMinuteData_60_KC,
    StockMinuteData_1_BJ, StockMinuteData_5_BJ, StockMinuteData_15_BJ, StockMinuteData_30_BJ, StockMinuteData_60_BJ,
    StockPriceLimit_SZ, StockPriceLimit_SH, StockPriceLimit_CY,StockPriceLimit_KC, StockPriceLimit_BJ,
)
from stock_models.chip import StockCyqChipsBJ, StockCyqChipsCY, StockCyqChipsKC, StockCyqChipsSH, StockCyqChipsSZ
from stock_models.advanced_metrics import (
    AdvancedChipMetrics_CY, AdvancedChipMetrics_SZ, AdvancedChipMetrics_KC, AdvancedChipMetrics_SH, AdvancedChipMetrics_BJ,
    AdvancedFundFlowMetrics_CY, AdvancedFundFlowMetrics_SZ, AdvancedFundFlowMetrics_KC, AdvancedFundFlowMetrics_SH, AdvancedFundFlowMetrics_BJ,
    AdvancedStructuralMetrics_CY, AdvancedStructuralMetrics_SZ, AdvancedStructuralMetrics_KC, AdvancedStructuralMetrics_SH, AdvancedStructuralMetrics_BJ,
    PlatformFeature_CY, PlatformFeature_SZ, PlatformFeature_KC, PlatformFeature_SH, PlatformFeature_BJ,
    TrendlineFeature_CY, TrendlineFeature_SZ, TrendlineFeature_KC, TrendlineFeature_SH, TrendlineFeature_BJ,
    MultiTimeframeTrendline_CY, MultiTimeframeTrendline_SZ, MultiTimeframeTrendline_KC, MultiTimeframeTrendline_SH, MultiTimeframeTrendline_BJ,
    TrendlineEvent_CY, TrendlineEvent_SZ, TrendlineEvent_KC, TrendlineEvent_SH, TrendlineEvent_BJ
)
from stock_models.stock_realtime import (
    StockRealtimeData_SH, StockRealtimeData_SZ, StockRealtimeData_CY, StockRealtimeData_KC, StockRealtimeData_BJ,
    StockLevel5Data_SH, StockLevel5Data_SZ, StockLevel5Data_CY, StockLevel5Data_KC, StockLevel5Data_BJ,
    StockTickData_SH, StockTickData_SZ, StockTickData_CY, StockTickData_KC, StockTickData_BJ
)
from stock_models.fund_flow import (
    FundFlowDailyDC_CY, FundFlowDailyDC_SZ, FundFlowDailyDC_KC, FundFlowDailyDC_SH, FundFlowDailyDC_BJ, FundFlowDailyTHS_CY, 
    FundFlowDailyTHS_SZ, FundFlowDailyTHS_KC, FundFlowDailyTHS_SH, FundFlowDailyTHS_BJ, FundFlowDailyCY, FundFlowDailySZ, 
    FundFlowDailyKC, FundFlowDailySH, FundFlowDailyBJ
)
from stock_models.chip_factors import (
    ChipFactorSH, ChipFactorSZ, ChipFactorKC, ChipFactorBJ, ChipFactorCY,
    ChipHoldingMatrix_SH, ChipHoldingMatrix_SZ, ChipHoldingMatrix_KC, ChipHoldingMatrix_BJ, ChipHoldingMatrix_CY
)
from stock_models.fundflow_factors import (
    FundFlowFactorCY, FundFlowFactorSZ, FundFlowFactorKC, FundFlowFactorSH, FundFlowFactorBJ
)
from typing import Type, Optional, List, Dict
from datetime import datetime, timezone, date
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

# def get_daily_basic_data_model_by_code(stock_code: str):
#     """
#     【公共辅助函数】
#     根据股票代码返回对应的日线数据分表Model。
#     Args:
#         stock_code (str): 股票代码，例如 '600519.SH'。
#     Returns:
#         Model Class: 对应的Django模型类。
#     """
#     if stock_code.startswith('3') and stock_code.endswith('.SZ'):
#         return StockDailyBasic_CY
#     elif stock_code.endswith('.SZ'):
#         return StockDailyBasic_SZ
#     elif stock_code.startswith('68') and stock_code.endswith('.SH'):
#         return StockDailyBasic_KC
#     elif stock_code.endswith('.SH'):
#         return StockDailyBasic_SH
#     elif stock_code.endswith('.BJ'):
#         return StockDailyBasic_BJ
#     # 提供一个默认返回值，以防有未覆盖到的情况
#     return StockDailyBasic_SZ

def get_fund_flow_dc_model_by_code(stock_code: str):
    """
    根据股票代码返回对应的【日级资金流向】数据表Model
    """
    if stock_code.startswith('3') and stock_code.endswith('.SZ'):
        return FundFlowDailyDC_CY
    elif stock_code.endswith('.SZ'):
        return FundFlowDailyDC_SZ
    elif stock_code.startswith('68') and stock_code.endswith('.SH'):
        return FundFlowDailyDC_KC
    elif stock_code.endswith('.SH'):
        return FundFlowDailyDC_SH
    elif stock_code.endswith('.BJ'):
        return FundFlowDailyDC_BJ
    else:
        logger.warning(f"未识别的股票代码: {stock_code}，资金流向默认使用SZ主板表")
        return FundFlowDailyDC_SZ  # 默认返回深市主板

def get_fund_flow_ths_model_by_code(stock_code: str):
    """
    根据股票代码返回对应的【日级资金流向】数据表Model
    """
    if stock_code.startswith('3') and stock_code.endswith('.SZ'):
        return FundFlowDailyTHS_CY
    elif stock_code.endswith('.SZ'):
        return FundFlowDailyTHS_SZ
    elif stock_code.startswith('68') and stock_code.endswith('.SH'):
        return FundFlowDailyTHS_KC
    elif stock_code.endswith('.SH'):
        return FundFlowDailyTHS_SH
    elif stock_code.endswith('.BJ'):
        return FundFlowDailyTHS_BJ
    else:
        logger.warning(f"未识别的股票代码: {stock_code}，资金流向默认使用SZ主板表")
        return FundFlowDailyTHS_SZ  # 默认返回深市主板

def get_fund_flow_model_by_code(stock_code: str):
    """
    根据股票代码返回对应的【日级资金流向】数据表Model
    """
    if stock_code.startswith('3') and stock_code.endswith('.SZ'):
        return FundFlowDailyCY
    elif stock_code.endswith('.SZ'):
        return FundFlowDailySZ
    elif stock_code.startswith('68') and stock_code.endswith('.SH'):
        return FundFlowDailyKC
    elif stock_code.endswith('.SH'):
        return FundFlowDailySH
    elif stock_code.endswith('.BJ'):
        return FundFlowDailyBJ
    else:
        logger.warning(f"未识别的股票代码: {stock_code}，资金流向默认使用SZ主板表")
        return FundFlowDailySZ  # 默认返回深市主板

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

def get_advanced_structural_metrics_model_by_code(stock_code: str):
    """
    【V1.0】根据股票代码返回对应的高级结构与行为指标数据表Model
    """
    #
    if stock_code.startswith('3') and stock_code.endswith('.SZ'):
        return AdvancedStructuralMetrics_CY
    elif stock_code.endswith('.SZ'):
        return AdvancedStructuralMetrics_SZ
    elif stock_code.startswith('68') and stock_code.endswith('.SH'):
        return AdvancedStructuralMetrics_KC
    elif stock_code.endswith('.SH'):
        return AdvancedStructuralMetrics_SH
    elif stock_code.endswith('.BJ'):
        return AdvancedStructuralMetrics_BJ
    else:
        print(f"未识别的股票代码: {stock_code}，默认使用SZ主板高级结构指标表")
        return AdvancedStructuralMetrics_SZ

# 在文件末尾添加新的辅助函数
def get_platform_feature_model_by_code(stock_code: str) -> Optional[Type[models.Model]]:
    """
    【V1.0】根据股票代码返回对应的矩形平台特征数据表Model
    """
    if stock_code.startswith('3') and stock_code.endswith('.SZ'):
        return PlatformFeature_CY
    elif stock_code.endswith('.SZ'):
        return PlatformFeature_SZ
    elif stock_code.startswith('68') and stock_code.endswith('.SH'):
        return PlatformFeature_KC
    elif stock_code.endswith('.SH'):
        return PlatformFeature_SH
    elif stock_code.endswith('.BJ'):
        return PlatformFeature_BJ
    else:
        print(f"未识别的股票代码: {stock_code}，平台特征默认使用SZ主板表")
        return PlatformFeature_SZ

def get_trendline_feature_model_by_code(stock_code: str) -> Optional[Type[models.Model]]:
    """
    【V1.0】根据股票代码返回对应的趋势线特征数据表Model
    """
    if stock_code.startswith('3') and stock_code.endswith('.SZ'):
        return TrendlineFeature_CY
    elif stock_code.endswith('.SZ'):
        return TrendlineFeature_SZ
    elif stock_code.startswith('68') and stock_code.endswith('.SH'):
        return TrendlineFeature_KC
    elif stock_code.endswith('.SH'):
        return TrendlineFeature_SH
    elif stock_code.endswith('.BJ'):
        return TrendlineFeature_BJ
    else:
        print(f"未识别的股票代码: {stock_code}，趋势线特征默认使用SZ主板表")
        return TrendlineFeature_SZ

# 在文件末尾添加新的模型辅助函数
def get_multi_timeframe_trendline_model_by_code(stock_code: str) -> Optional[Type[models.Model]]:
    """
    【V2.1】根据股票代码返回对应的趋势线矩阵数据表Model
    """
    if stock_code.startswith('3') and stock_code.endswith('.SZ'):
        return MultiTimeframeTrendline_CY
    elif stock_code.endswith('.SZ'):
        return MultiTimeframeTrendline_SZ
    elif stock_code.startswith('68') and stock_code.endswith('.SH'):
        return MultiTimeframeTrendline_KC
    elif stock_code.endswith('.SH'):
        return MultiTimeframeTrendline_SH
    elif stock_code.endswith('.BJ'):
        return MultiTimeframeTrendline_BJ
    else:
        print(f"未识别的股票代码: {stock_code}，趋势线矩阵默认使用SZ主板表")
        return MultiTimeframeTrendline_SZ

def get_trendline_event_model_by_code(stock_code: str) -> Optional[Type[models.Model]]:
    """
    【V2.1】根据股票代码返回对应的趋势线事件数据表Model
    """
    if stock_code.startswith('3') and stock_code.endswith('.SZ'):
        return TrendlineEvent_CY
    elif stock_code.endswith('.SZ'):
        return TrendlineEvent_SZ
    elif stock_code.startswith('68') and stock_code.endswith('.SH'):
        return TrendlineEvent_KC
    elif stock_code.endswith('.SH'):
        return TrendlineEvent_SH
    elif stock_code.endswith('.BJ'):
        return TrendlineEvent_BJ
    else:
        print(f"未识别的股票代码: {stock_code}，趋势线事件默认使用SZ主板表")
        return TrendlineEvent_SZ

def get_price_limit_percent(stock_code: str) -> float:
    """
    【公共辅助函数】根据股票代码返回其对应的涨跌停限制比例。
    Args:
        stock_code (str): 股票代码，例如 '300030.SZ'。
    Returns:
        float: 涨跌停比例 (0.1, 0.2, 0.3)。
    """
    if stock_code.startswith('3') and stock_code.endswith('.SZ'):
        # 创业板
        return 0.2
    elif stock_code.startswith('68') and stock_code.endswith('.SH'):
        # 科创板
        return 0.2
    elif stock_code.endswith('.BJ'):
        # 北交所
        return 0.3
    elif stock_code.endswith('.SZ') or stock_code.endswith('.SH'):
        # 沪深主板
        return 0.1
    # 默认返回主板限制
    return 0.1

# 新增获取涨跌停价格分表模型的辅助函数
def get_stk_limit_model_by_code(stock_code: str) -> Optional[Type[models.Model]]:
    """
    【公共辅助函数】根据股票代码返回对应的每日涨跌停价格分表Model。
    """
    if stock_code.startswith('3') and stock_code.endswith('.SZ'):
        return StockPriceLimit_CY
    elif stock_code.endswith('.SZ'):
        return StockPriceLimit_SZ
    elif stock_code.startswith('68') and stock_code.endswith('.SH'):
        return StockPriceLimit_KC
    elif stock_code.endswith('.SH'):
        return StockPriceLimit_SH
    elif stock_code.endswith('.BJ'):
        return StockPriceLimit_BJ
    return None # 对于无法识别的股票代码返回None

def get_stock_tick_data_model_by_code(stock_code: str) -> Optional[Type[models.Model]]:
    """
    根据股票代码返回对应的逐笔交易数据分表Model。
    Args:
        stock_code (str): 股票代码，例如 '000001.SZ'。
    Returns:
        Optional[Type[models.Model]]: 对应的Django模型类，如果未找到则为 None。
    """
    if stock_code.startswith('3') and stock_code.endswith('.SZ'):
        return StockTickData_CY
    elif stock_code.endswith('.SZ'):
        return StockTickData_SZ
    elif stock_code.startswith('68') and stock_code.endswith('.SH'):
        return StockTickData_KC
    elif stock_code.endswith('.SH'):
        return StockTickData_SH
    elif stock_code.endswith('.BJ'):
        return StockTickData_BJ
    else:
        print(f"调试信息: 未能为 {stock_code} 找到对应的逐笔交易数据模型。")
        return None

def get_stock_realtime_data_model_by_code(stock_code: str) -> Optional[Type[models.Model]]:
    """
    根据股票代码返回对应的实时行情快照数据分表Model。
    """
    if stock_code.startswith('3') and stock_code.endswith('.SZ'):
        return StockRealtimeData_CY
    elif stock_code.endswith('.SZ'):
        return StockRealtimeData_SZ
    elif stock_code.startswith('68') and stock_code.endswith('.SH'):
        return StockRealtimeData_KC
    elif stock_code.endswith('.SH'):
        return StockRealtimeData_SH
    elif stock_code.endswith('.BJ'):
        return StockRealtimeData_BJ
    else:
        print(f"调试信息: 未能为 {stock_code} 找到对应的实时行情快照数据模型。")
        return None

def get_stock_level5_data_model_by_code(stock_code: str) -> Optional[Type[models.Model]]:
    """
    根据股票代码返回对应的Level5盘口数据分表Model。
    """
    if stock_code.startswith('3') and stock_code.endswith('.SZ'):
        return StockLevel5Data_CY
    elif stock_code.endswith('.SZ'):
        return StockLevel5Data_SZ
    elif stock_code.startswith('68') and stock_code.endswith('.SH'):
        return StockLevel5Data_KC
    elif stock_code.endswith('.SH'):
        return StockLevel5Data_SH
    elif stock_code.endswith('.BJ'):
        return StockLevel5Data_BJ
    else:
        print(f"调试信息: 未能为 {stock_code} 找到对应的Level5盘口数据模型。")
        return None

def get_chip_factor_model_by_code(stock_code: str):
    """
    根据股票代码返回对应的筹码因子分表Model
    """
    if stock_code.startswith('3') and stock_code.endswith('.SZ'):
        return ChipFactorCY
    elif stock_code.endswith('.SZ'):
        return ChipFactorSZ
    elif stock_code.startswith('68') and stock_code.endswith('.SH'):
        return ChipFactorKC
    elif stock_code.endswith('.SH'):
        return ChipFactorSH
    elif stock_code.endswith('.BJ'):
        return ChipFactorBJ
    else:
        return ChipFactorSZ

# 批量获取筹码因子的函数
async def get_chip_factors_batch(stock_codes: List[str], trade_date: date) -> Dict[str, Dict]:
    """
    批量获取筹码因子数据
    Args:
        stock_codes: 股票代码列表
        trade_date: 交易日期
    Returns:
        Dict[str, Dict]: 股票代码到因子字典的映射
    """
    result = {}
    # 按市场分组
    market_groups = {}
    for code in stock_codes:
        model = get_chip_factor_model_by_code(code)
        market_groups.setdefault(model, []).append(code)
    # 并行查询不同市场的数据
    for model, codes in market_groups.items():
        queryset = model.objects.filter(
            stock__stock_code__in=codes,
            trade_time=trade_date,
            calc_status='success'
        ).select_related('stock')
        async for factor in queryset:
            result[factor.stock.stock_code] = {
                'price_to_weight_avg_ratio': factor.price_to_weight_avg_ratio,
                'chip_concentration_ratio': factor.chip_concentration_ratio,
                'chip_stability': factor.chip_stability,
                'profit_pressure': factor.profit_pressure,
                'chip_entropy': factor.chip_entropy,
                'chip_skewness': factor.chip_skewness,
                'chip_kurtosis': factor.chip_kurtosis,
                'winner_rate': factor.winner_rate,
                'close': factor.close,
                'weight_avg_cost': factor.weight_avg_cost,
                'trade_time': factor.trade_time
            }
    return result

def get_chip_holding_matrix_model_by_code(stock_code: str):
    """
    根据股票代码返回对应的筹码持有时间矩阵分表Model
    """
    if stock_code.startswith('3') and stock_code.endswith('.SZ'):
        return ChipHoldingMatrix_CY
    elif stock_code.endswith('.SZ'):
        return ChipHoldingMatrix_SZ
    elif stock_code.startswith('68') and stock_code.endswith('.SH'):
        return ChipHoldingMatrix_KC
    elif stock_code.endswith('.SH'):
        return ChipHoldingMatrix_SH
    elif stock_code.endswith('.BJ'):
        return ChipHoldingMatrix_BJ
    else:
        return ChipHoldingMatrix_SZ

# 资金流向因子分表模型获取函数
def get_fundflow_factor_model_by_code(stock_code: str) -> Optional[Type[models.Model]]:
    """
    根据股票代码返回对应的资金流向因子分表Model
    """
    if stock_code.startswith('3') and stock_code.endswith('.SZ'):
        return FundFlowFactorCY
    elif stock_code.endswith('.SZ'):
        return FundFlowFactorSZ
    elif stock_code.startswith('68') and stock_code.endswith('.SH'):
        return FundFlowFactorKC
    elif stock_code.endswith('.SH'):
        return FundFlowFactorSH
    elif stock_code.endswith('.BJ'):
        return FundFlowFactorBJ
    else:
        print(f"未识别的股票代码: {stock_code}，资金流向因子默认使用SZ主板表")
        return FundFlowFactorSZ

def get_fundflow_factors_batch(stock_codes: List[str], trade_date: date) -> Dict[str, Dict]:
    """
    批量获取资金流向因子数据
    Args:
        stock_codes: 股票代码列表
        trade_date: 交易日期
    Returns:
        Dict[str, Dict]: 股票代码到因子字典的映射
    """
    result = {}
    # 按市场分组
    market_groups = {}
    for code in stock_codes:
        model = get_fundflow_factor_model_by_code(code)
        if model:
            market_groups.setdefault(model, []).append(code)
    
    # 并行查询不同市场的数据
    for model, codes in market_groups.items():
        queryset = model.objects.filter(
            stock__stock_code__in=codes,
            trade_time=trade_date
        ).select_related('stock')
        
        for factor in queryset:
            result[factor.stock.stock_code] = {
                # 绝对量级指标
                'total_net_amount_5d': float(factor.total_net_amount_5d) if factor.total_net_amount_5d else None,
                'total_net_amount_10d': float(factor.total_net_amount_10d) if factor.total_net_amount_10d else None,
                'avg_daily_net_5d': float(factor.avg_daily_net_5d) if factor.avg_daily_net_5d else None,
                'avg_daily_net_10d': float(factor.avg_daily_net_10d) if factor.avg_daily_net_10d else None,
                
                # 相对强度指标
                'net_amount_ratio': float(factor.net_amount_ratio) if factor.net_amount_ratio else None,
                'net_amount_ratio_ma5': float(factor.net_amount_ratio_ma5) if factor.net_amount_ratio_ma5 else None,
                'flow_intensity': float(factor.flow_intensity) if factor.flow_intensity else None,
                'intensity_level': factor.intensity_level,
                
                # 主力行为模式
                'behavior_pattern': factor.behavior_pattern,
                'pattern_confidence': float(factor.pattern_confidence) if factor.pattern_confidence else None,
                'accumulation_score': float(factor.accumulation_score) if factor.accumulation_score else None,
                'pushing_score': float(factor.pushing_score) if factor.pushing_score else None,
                'distribution_score': float(factor.distribution_score) if factor.distribution_score else None,
                'shakeout_score': float(factor.shakeout_score) if factor.shakeout_score else None,
                
                # 资金流向质量评估
                'outflow_quality': float(factor.outflow_quality) if factor.outflow_quality else None,
                'inflow_persistence': factor.inflow_persistence,
                'large_order_anomaly': factor.large_order_anomaly,
                'flow_consistency': float(factor.flow_consistency) if factor.flow_consistency else None,
                
                # 多周期共振指标
                'daily_weekly_sync': float(factor.daily_weekly_sync) if factor.daily_weekly_sync else None,
                'daily_monthly_sync': float(factor.daily_monthly_sync) if factor.daily_monthly_sync else None,
                'short_mid_sync': float(factor.short_mid_sync) if factor.short_mid_sync else None,
                'mid_long_sync': float(factor.mid_long_sync) if factor.mid_long_sync else None,
                
                # 趋势动量指标
                'flow_momentum_5d': float(factor.flow_momentum_5d) if factor.flow_momentum_5d else None,
                'flow_momentum_10d': float(factor.flow_momentum_10d) if factor.flow_momentum_10d else None,
                'flow_acceleration': float(factor.flow_acceleration) if factor.flow_acceleration else None,
                'uptrend_strength': float(factor.uptrend_strength) if factor.uptrend_strength else None,
                'downtrend_strength': float(factor.downtrend_strength) if factor.downtrend_strength else None,
                
                # 量价背离指标
                'price_flow_divergence': float(factor.price_flow_divergence) if factor.price_flow_divergence else None,
                'divergence_type': factor.divergence_type,
                'divergence_strength': float(factor.divergence_strength) if factor.divergence_strength else None,
                
                # 统计特征指标
                'flow_zscore': float(factor.flow_zscore) if factor.flow_zscore else None,
                'flow_percentile': float(factor.flow_percentile) if factor.flow_percentile else None,
                'flow_volatility_10d': float(factor.flow_volatility_10d) if factor.flow_volatility_10d else None,
                
                # 预测指标
                'expected_flow_next_1d': float(factor.expected_flow_next_1d) if factor.expected_flow_next_1d else None,
                'flow_forecast_confidence': float(factor.flow_forecast_confidence) if factor.flow_forecast_confidence else None,
                'uptrend_continuation_prob': float(factor.uptrend_continuation_prob) if factor.uptrend_continuation_prob else None,
                'reversal_prob': float(factor.reversal_prob) if factor.reversal_prob else None,
                
                # 复合综合指标
                'comprehensive_score': float(factor.comprehensive_score) if factor.comprehensive_score else None,
                'trading_signal': factor.trading_signal,
                'signal_strength': float(factor.signal_strength) if factor.signal_strength else None,
                
                # 时间信息
                'trade_time': factor.trade_time
            }
    
    return result

