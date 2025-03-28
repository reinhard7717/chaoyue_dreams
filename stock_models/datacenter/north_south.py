# models/datacenter/north_south.py
from django.db import models
from django.utils.translation import gettext_lazy as _
from utils.models import BaseModel
from datetime import datetime
from stock_models.stock_basic import StockInfo

class NorthSouthFundOverview(BaseModel):
    """南北向资金流向概览"""
    trade_date = models.DateField(verbose_name="日期")  # 原 t
    hk_to_shanghai = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="沪股通(北向)(亿元)")  # 原 hk2sh
    hk_to_shenzhen = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="深股通(北向)(亿元)")  # 原 hk2sz
    northbound_net_inflow = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="北向资金净流入(亿元)")  # 原 bxzjlr
    shanghai_to_hk = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="沪港通(南向)(亿元)")  # 原 sh2hk
    shenzhen_to_hk = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="深港通(南向)(亿元)")  # 原 sz2hk
    southbound_net_inflow = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="南向资金净流入(亿元)")  # 原 nxzjlr
    
    class Meta:
        verbose_name = "南北向资金流向概览"
        db_table = "north_south_fund_overview"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['trade_date']),
        ]


class NorthFundTrend(BaseModel):
    """北向资金历史走势"""
    trade_date = models.DateField(verbose_name="日期")  # 原 t
    hk_to_shanghai = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="沪股通(亿元)")  # 原 hk2sh
    hk_to_shenzhen = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="深股通(亿元)")  # 原 hk2sz
    northbound_net_inflow = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="北向资金净流入(亿元)")  # 原 bxzjlr
    hs300_index = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="沪深300指数")  # 原 hsIndex
    
    class Meta:
        verbose_name = "北向资金历史走势"
        db_table = "north_fund_trend"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['trade_date']),
        ]


class SouthFundTrend(BaseModel):
    """南向资金历史走势"""
    trade_date = models.DateField(verbose_name="日期")  # 原 t
    shanghai_to_hk = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="沪港通(亿元)")  # 原 sh2hk
    shenzhen_to_hk = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="深港通(亿元)")  # 原 sz2hk
    southbound_net_inflow = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="南向资金净流入(亿元)")  # 原 nxzjlr
    hang_seng_index = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="恒生指数")  # 原 hsi
    
    class Meta:
        verbose_name = "南向资金历史走势"
        db_table = "south_fund_trend"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['trade_date']),
        ]


class NorthStockHolding(BaseModel):
    """北向持股明细"""
    trade_date = models.DateField(verbose_name="日期")  # 原 t
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, blank=True, null=True, related_name="north_stock_holding", verbose_name=_("股票"))
    holding_shares = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="持股数量(万股)")  # 原 cgs
    float_share_ratio = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="占流通股比例(%)")  # 原 zltgbl
    holding_value = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="持股市值(万元)")  # 原 cgsz
    daily_share_change = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="当日持股变动(万股)")  # 原 djcgs
    daily_value_change = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="当日市值变动(万元)")  # 原 djcgsz
    price_change_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="涨跌幅(%)")  # 原 zdf
    holding_ratio_change = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="持股占比变动(%)")  # 原 zdbl
    period = models.CharField(max_length=10, verbose_name="统计周期")  # LD(当日), 3D, 5D, 10D, LM(月), LQ(季), LY(年)
    
    class Meta:
        verbose_name = "北向持股明细"
        db_table = "north_stock_holding"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['trade_date']),
            models.Index(fields=['stock']),
            models.Index(fields=['period']),
        ]
