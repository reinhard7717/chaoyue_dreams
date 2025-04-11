# models/datacenter/financial.py
from django.db import models
from django.utils.translation import gettext_lazy as _
from bulk_update_or_create import BulkUpdateOrCreateQuerySet
from utils.models import BaseModel

class WeeklyRankChange(BaseModel):
    """周涨跌排名"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, blank=True, null=True, related_name="weekly_rank_change", verbose_name=_("股票"))
    trade_date = models.DateField(verbose_name="日期")  # 原 t
    weekly_change_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="周涨跌幅")  # 原 zdf
    weekly_volume = models.BigIntegerField(verbose_name="周成交量")  # 原 v
    weekly_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="周成交额")  # 原 amount
    weekly_turnover_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="周换手率")  # 原 hs
    weekly_highest_price = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="周最高价")  # 原 hp
    weekly_lowest_price = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="周最低价")  # 原 lp
    weekly_amplitude = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="周振幅")  # 原 zf
    
    class Meta:
        verbose_name = "周涨跌排名"
        db_table = "weekly_rank_change"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['trade_date']),
            models.Index(fields=['stock']),
        ]

    def __code__(self):
        return self.stock.stock_code

class MonthlyRankChange(BaseModel):
    """月涨跌排名"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, blank=True, null=True, related_name="monthly_rank_change", verbose_name=_("股票"))
    trade_date = models.DateField(verbose_name="日期")  # 原 t
    monthly_change_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="月涨跌幅")  # 原 zdf
    monthly_volume = models.BigIntegerField(verbose_name="月成交量")  # 原 v
    monthly_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="月成交额")  # 原 amount
    monthly_turnover_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="月换手率")  # 原 hs
    monthly_highest_price = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="月最高价")  # 原 hp
    monthly_lowest_price = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="月最低价")  # 原 lp
    monthly_amplitude = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="月振幅")  # 原 zf
    
    class Meta:
        verbose_name = "月涨跌排名"
        db_table = "monthly_rank_change"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['trade_date']),
            models.Index(fields=['stock']),
        ]

    def __code__(self):
        return self.stock.stock_code

class WeeklyStrongStock(BaseModel):
    """本周强势股"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, blank=True, null=True, related_name="weekly_strong_stock", verbose_name=_("股票"))
    trade_date = models.DateField(verbose_name="日期")  # 原 t
    weekly_change_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="周涨跌幅")  # 原 zdf
    weekly_open_price = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="周开盘价")  # 原 o
    weekly_close_price = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="周收盘价")  # 原 c
    weekly_highest_price = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="周最高价")  # 原 h
    weekly_lowest_price = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="周最低价")  # 原 l
    weekly_volume = models.BigIntegerField(verbose_name="周成交量")  # 原 v
    weekly_turnover_rate = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="周换手率")  # 原 hs
    hs300_weekly_change_rate = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="本周沪深300涨幅")  # 原 zf300
    
    class Meta:
        verbose_name = "本周强势股"
        db_table = "weekly_strong_stock"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['trade_date']),
            models.Index(fields=['stock']),
        ]

    def __code__(self):
        return self.stock.stock_code

class MonthlyStrongStock(BaseModel):
    """本月强势股"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, blank=True, null=True, related_name="monthly_strong_stock", verbose_name=_("股票"))
    trade_date = models.DateField(verbose_name="日期")  # 原 t
    monthly_change_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="月涨跌幅")  # 原 zdf
    monthly_open_price = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="月开盘价")  # 原 o
    monthly_close_price = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="月收盘价")  # 原 c
    monthly_highest_price = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="月最高价")  # 原 h
    monthly_lowest_price = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="月最低价")  # 原 l
    monthly_volume = models.BigIntegerField(verbose_name="月成交量")  # 原 v
    monthly_turnover_rate = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="月换手率")  # 原 hs
    hs300_monthly_change_rate = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="本月沪深300涨幅")  # 原 zf300
    
    class Meta:
        verbose_name = "本月强势股"
        db_table = "monthly_strong_stock"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['trade_date']),
            models.Index(fields=['stock']),
        ]

    def __code__(self):
        return self.stock.stock_code

class CircMarketValueRank(BaseModel):
    """流通市值排行"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, blank=True, null=True, related_name="circ_market_value_rank", verbose_name=_("股票"))
    trade_date = models.DateField(verbose_name="日期")  # 原 t
    close_price = models.DecimalField(max_digits=10, decimal_places=3, verbose_name="收盘价")  # 原 c
    change_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="涨跌幅")  # 原 zdf
    volume = models.BigIntegerField(verbose_name="成交量")  # 原 v
    turnover_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="换手率")  # 原 hs
    circulating_market_value = models.DecimalField(max_digits=20, decimal_places=6, verbose_name="流通市值")  # 原 ltsz
    total_market_value = models.DecimalField(max_digits=20, decimal_places=6, verbose_name="总市值")  # 原 zsz
    
    class Meta:
        verbose_name = "流通市值排行"
        db_table = "circulating_market_value_rank"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['trade_date']),
            models.Index(fields=['stock']),
        ]

    def __code__(self):
        return self.stock.stock_code

class PERatioRank(BaseModel):
    """市盈率排行"""
    trade_date = models.DateField(verbose_name="日期")  # 原 t
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, blank=True, null=True, related_name="pe_ratio_rank", verbose_name=_("股票"))
    close_price = models.DecimalField(max_digits=10, decimal_places=3, verbose_name="收盘价")  # 原 c
    change_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="涨跌幅")  # 原 zdf
    volume = models.BigIntegerField(verbose_name="成交量")  # 原 v
    turnover_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="换手率")  # 原 hs
    static_pe_ratio = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="静态市盈率")  # 原 jpe
    ttm_pe_ratio = models.DecimalField(max_digits=10, decimal_places=3, verbose_name="市盈率(TTM)")  # 原 dpe
    
    class Meta:
        verbose_name = "市盈率排行"
        db_table = "pe_ratio_rank"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['trade_date']),
            models.Index(fields=['stock']),
        ]

    def __code__(self):
        return self.stock.stock_code

class PBRatioRank(BaseModel):
    """市净率排行"""
    trade_date = models.DateField(verbose_name="日期")  # 原 t
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, blank=True, null=True, related_name="pb_ratio_rank", verbose_name=_("股票"))
    close_price = models.DecimalField(max_digits=10, decimal_places=3, verbose_name="收盘价")  # 原 c
    change_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="涨跌幅")  # 原 zdf
    volume = models.BigIntegerField(verbose_name="成交量")  # 原 v
    turnover_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="换手率")  # 原 hs
    pb_ratio = models.DecimalField(max_digits=10, decimal_places=3, verbose_name="市净率")  # 原 sjl
    net_asset_per_share = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="每股净资产")  # 原 jzc
    
    class Meta:
        verbose_name = "市净率排行"
        db_table = "pb_ratio_rank"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['trade_date']),
            models.Index(fields=['stock']),
        ]

    def __code__(self):
        return self.stock.stock_code

class ROERank(BaseModel):
    """ROE排行"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, blank=True, null=True, related_name="roe_rank", verbose_name=_("股票"))
    roe = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="ROE")
    total_market_value = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="总市值")  # 原 zsz
    net_assets = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="净资产")  # 原 jzc
    net_profit = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="净利润")  # 原 jlr
    dynamic_pe_ratio = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="市盈率(动)")  # 原 syld
    pb_ratio = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="市净率")  # 原 sjl
    gross_profit_margin = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="毛利率")  # 原 mll
    net_profit_margin = models.DecimalField(max_digits=10, decimal_places=8, verbose_name="净利率")  # 原 jll
    industry_avg_roe = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="行业平均ROE")  # 原 hyroe
    industry_avg_market_value = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="行业平均总市值")  # 原 hyzsz
    industry_avg_net_assets = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="行业平均净资产")  # 原 hyjzc
    industry_avg_net_profit = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="行业平均净利润")  # 原 hyjlr
    industry_avg_dynamic_pe = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="行业平均市盈率(动)")  # 原 hysyld
    industry_avg_pb_ratio = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="行业平均市净率")  # 原 hysjl
    industry_avg_gross_margin = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="行业平均毛利率")  # 原 hymll
    industry_avg_net_margin = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="行业平均净利率")  # 原 hyjll
    roe_industry_rank = models.IntegerField(verbose_name="ROE行业排名")  # 原 roepm
    market_value_industry_rank = models.IntegerField(verbose_name="总市值行业排名")  # 原 zszpm
    net_assets_industry_rank = models.IntegerField(verbose_name="净资产行业排名")  # 原 jzcpm
    net_profit_industry_rank = models.IntegerField(verbose_name="净利润行业排名")  # 原 jlrpm
    pe_ratio_industry_rank = models.IntegerField(verbose_name="市盈率行业排名")  # 原 syldpm
    pb_ratio_industry_rank = models.IntegerField(verbose_name="市净率行业排名")  # 原 sjlpm
    gross_margin_industry_rank = models.IntegerField(verbose_name="毛利率行业排名")  # 原 mllpm
    net_margin_industry_rank = models.IntegerField(verbose_name="净利率行业排名")  # 原 jllpm
    industry_name = models.CharField(max_length=50, verbose_name="行业名")  # 原 hym
    industry_stock_count = models.IntegerField(verbose_name="同行业股票总数量")  # 原 hygpzs
    trade_time = models.DateTimeField(auto_now=True, verbose_name="更新时间")
    
    class Meta:
        verbose_name = "ROE排行"
        db_table = "roe_rank"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['industry_name']),
        ]

    def __code__(self):
        return self.stock.stock_code
