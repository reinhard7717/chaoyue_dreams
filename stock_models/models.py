from django.db import models
from django.utils.translation import gettext_lazy as _
from utils.models import BaseModel

# 避免循环导入
def get_models():
    from . import stock_basic, stock_realtime, index, fund_flow
    from .datacenter import financial
    return {
        'StockBasic': stock_basic.StockBasic,
        'NewStockCalendar': stock_basic.NewStockCalendar,
        'CompanyInfo': stock_basic.CompanyInfo,
        'STStockList': stock_basic.STStockList,
        'CompanyIndex': stock_basic.CompanyIndex,
        'QuarterlyProfit': stock_basic.QuarterlyProfit,
        'RealtimeData': stock_realtime.RealtimeData,
        'AbnormalMovement': stock_realtime.AbnormalMovement,
        'IndexInfo': index.IndexInfo,
        'FundFlowMinute': fund_flow.FundFlowMinute,
        'FundFlowDaily': fund_flow.FundFlowDaily,
        'MainForcePhase': fund_flow.MainForcePhase,
        'TransactionDistribution': fund_flow.TransactionDistribution,
        'StockPool': fund_flow.StockPool,
        'LimitUpPool': fund_flow.LimitUpPool,
        'LimitDownPool': fund_flow.LimitDownPool,
        'StrongStockPool': fund_flow.StrongStockPool,
        'NewStockPool': fund_flow.NewStockPool,
        'BreakLimitPool': fund_flow.BreakLimitPool,
        'ROERank': financial.ROERank,
    }

# 导出所有模型
__all__ = list(get_models().keys()) 