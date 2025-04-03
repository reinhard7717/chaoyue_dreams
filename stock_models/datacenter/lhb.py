# models/datacenter/lhb.py
from datetime import datetime
from django.db import models
from django.utils.translation import gettext_lazy as _
from stock_models.stock_basic import StockInfo
from utils.models import BaseModel


class LhbDetail(BaseModel):
    """龙虎榜明细数据"""
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, blank=True, null=True, related_name="lhb_detail", verbose_name=_("股票"))
    trade_date = models.DateField(verbose_name="日期", null=True, blank=True)
    close_price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="收盘价")  # 原 c
    value = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="对应值")  # 原 val
    volume = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="成交量(万股)")  # 原 v
    amount = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="成交额(万元)")  # 原 e
    
    class Meta:
        verbose_name = "龙虎榜明细"
        db_table = "lhb_detail"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['stock', 'trade_date']),
        ]

    def __code__(self):
        return self.stock.stock_code



class LhbDaily(BaseModel):
    """每日龙虎榜数据"""
    trade_date = models.DateField(verbose_name="日期")  # 原 t
    decline_deviation_7pct = models.JSONField(null=True, blank=True, verbose_name="跌幅偏离值达7%的证券")  # 原 dpl7
    rise_cumulative_20pct = models.JSONField(null=True, blank=True, verbose_name="连续三个交易日内，涨幅偏离值累计达20%的证券")  # 原 z20
    rise_deviation_7pct = models.JSONField(null=True, blank=True, verbose_name="涨幅偏离值达7%的证券")  # 原 zpl7
    turnover_20pct = models.JSONField(null=True, blank=True, verbose_name="换手率达20%的证券")  # 原 h20
    st_rise_15pct = models.JSONField(null=True, blank=True, verbose_name="连续三个交易日内，涨幅偏离值累计达到15%的ST证券")  # 原 st15
    st_rise_12pct = models.JSONField(null=True, blank=True, verbose_name="连续三个交易日内，涨幅偏离值累计达到12%的ST证券")  # 原 st12
    st_decline_15pct = models.JSONField(null=True, blank=True, verbose_name="连续三个交易日内，跌幅偏离值累计达到15%的ST证券")  # 原 std15
    st_decline_12pct = models.JSONField(null=True, blank=True, verbose_name="连续三个交易日内，跌幅偏离值累计达到12%的ST证券")  # 原 std12
    amplitude_15pct = models.JSONField(null=True, blank=True, verbose_name="振幅值达15%的证券")  # 原 zf15
    decline_cumulative_15pct = models.JSONField(null=True, blank=True, verbose_name="连续三个交易日内，跌幅偏离值累计达20%的证券")  # 原 df15
    no_price_limit = models.JSONField(null=True, blank=True, verbose_name="无价格涨跌幅限制的证券")  # 原 wxz
    no_price_limit_halted = models.JSONField(null=True, blank=True, verbose_name="当日无价格涨跌幅限制的A股，出现异常波动停牌的股票")  # 原 wxztp
    
    class Meta:
        verbose_name = "每日龙虎榜"
        db_table = "lhb_daily"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['trade_date']),
        ]

    def __code__(self):
        return self.stock.stock_code



class StockOnList(BaseModel):
    """个股上榜统计"""
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, blank=True, null=True, related_name="stock_on_list", verbose_name=_("股票"))
    list_count = models.IntegerField(verbose_name="上榜次数")  # 原 count
    total_buy_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="累积获取额(万)")  # 原 totalb
    total_sell_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="累积卖出额(万)")  # 原 totals
    net_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="净额(万)")  # 原 netp
    buy_seat_count = models.IntegerField(verbose_name="买入席位数")  # 原 xb
    sell_seat_count = models.IntegerField(verbose_name="卖出席位")  # 原 xs
    stats_days = models.IntegerField(verbose_name="统计天数")  # 原 days (5, 10, 30, 60)
    update_time = models.DateTimeField(default=datetime.now, verbose_name="更新时间")
    
    class Meta:
        verbose_name = "个股上榜统计"
        db_table = "stock_on_list"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['stats_days']),
        ]

    def __code__(self):
        return self.stock.stock_code



class BrokerOnList(BaseModel):
    """营业部上榜统计"""
    broker_name = models.CharField(max_length=100, verbose_name="营业部名称")  # 原 yybmc
    list_count = models.IntegerField(verbose_name="上榜次数")  # 原 count
    total_buy_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="累积获取额(万)")  # 原 totalb
    buy_count = models.IntegerField(verbose_name="买入席位")  # 原 bcount
    total_sell_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="累积卖出额(万)")  # 原 totals
    sell_count = models.IntegerField(verbose_name="卖出席位")  # 原 scount
    top3_stocks = models.CharField(max_length=200, verbose_name="买入前三股票")  # 原 top3
    stats_days = models.IntegerField(verbose_name="统计天数")  # 原 days (5, 10, 30, 60)
    update_time = models.DateTimeField(default=datetime.now, verbose_name="更新时间")
    
    class Meta:
        verbose_name = "营业部上榜统计"
        db_table = "broker_on_list"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['broker_name']),
            models.Index(fields=['stats_days']),
        ]

    def __code__(self):
        return self.stock.stock_code



class InstitutionTradeTrack(BaseModel):
    """机构席位追踪"""
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, blank=True, null=True, related_name="institution_trade_track", verbose_name=_("股票"))
    buy_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="累积买入额(万)")  # 原 be
    buy_count = models.IntegerField(verbose_name="买入次数")  # 原 bcount
    sell_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="累积卖出额(万)")  # 原 se
    sell_count = models.IntegerField(verbose_name="卖出次数")  # 原 scount
    net_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="净额(万)")  # 原 ende
    stats_days = models.IntegerField(verbose_name="统计天数")  # 原 days (5, 10, 30, 60)
    update_time = models.DateTimeField(default=datetime.now, verbose_name="更新时间")
    
    class Meta:
        verbose_name = "机构席位追踪"
        db_table = "institution_trade_track"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['stats_days']),
        ]

    def __code__(self):
        return self.stock.stock_code



class InstitutionTradeDetail(BaseModel):
    """机构席位成交明细"""
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, blank=True, null=True, related_name="institution_trade_detail", verbose_name=_("股票"))
    trade_date = models.DateField(verbose_name="交易日期")  # 原 t
    buy_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="机构席位买入额(万)")  # 原 buy
    sell_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="机构席位卖出额(万)")  # 原 sell
    trade_type = models.CharField(max_length=100, verbose_name="类型")  # 原 type
    
    class Meta:
        verbose_name = "机构席位成交明细"
        db_table = "institution_trade_detail"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_date']),
        ]

    def __code__(self):
        return self.stock.stock_code



