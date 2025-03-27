# models/datacenter/statistics.py
from django.db import models
from utils.models import BaseModel


class StageHighLow(BaseModel):
    """阶段最高最低"""
    trade_date = models.DateField(verbose_name="日期")  # 原 t
    stock_code = models.CharField(max_length=10, verbose_name="代码")  # 原 dm
    stock_name = models.CharField(max_length=50, verbose_name="名称")  # 原 mc
    high_price_5d = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="近5日最高价")  # 原 g5
    low_price_5d = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="近5日最低价")  # 原 d5
    change_rate_5d = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="近5日涨跌幅")  # 原 zd5
    has_ex_dividend_5d = models.IntegerField(verbose_name="近5日是否除权除息")  # 原 iscq5
    high_price_10d = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="近10日最高价")  # 原 g10
    low_price_10d = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="近10日最低价")  # 原 d10
    change_rate_10d = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="近10日涨跌幅")  # 原 zd10
    has_ex_dividend_10d = models.IntegerField(verbose_name="近10日是否除权除息")  # 原 iscq10
    high_price_20d = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="近20日最高价")  # 原 g20
    low_price_20d = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="近20日最低价")  # 原 d20
    change_rate_20d = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="近20日涨跌幅")  # 原 zd20
    has_ex_dividend_20d = models.IntegerField(verbose_name="近20日是否除权除息")  # 原 iscq20
    high_price_60d = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="近60日最高价")  # 原 g60
    low_price_60d = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="近60日最低价")  # 原 d60
    change_rate_60d = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="近60日涨跌幅")  # 原 zd60
    has_ex_dividend_60d = models.IntegerField(verbose_name="近60日是否除权除息")  # 原 iscq60
    
    class Meta:
        verbose_name = "阶段最高最低"
        db_table = "stage_high_low"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['trade_date']),
            models.Index(fields=['stock_code']),
        ]


class NewHighStock(BaseModel):
    """盘中创新高个股"""
    trade_date = models.DateField(verbose_name="日期")  # 原 t
    stock_code = models.CharField(max_length=10, verbose_name="代码")  # 原 dm
    stock_name = models.CharField(max_length=50, verbose_name="名称")  # 原 mc
    close_price = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="收盘价")  # 原 c
    high_price = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="最高价")  # 原 h
    low_price = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="最低价")  # 原 l
    change_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="涨跌幅")  # 原 zdf
    is_ex_dividend = models.IntegerField(verbose_name="当天是否除权除息")  # 原 iscq
    turnover_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="换手率")  # 原 hs
    change_rate_5d = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="5日涨跌幅")  # 原 zdf5
    has_ex_dividend_5d = models.IntegerField(verbose_name="近5日是否除权除息")  # 原 iscq5
    change_rate_10d = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="10日涨跌幅")  # 原 zdf10
    has_ex_dividend_10d = models.IntegerField(verbose_name="近10日是否除权除息")  # 原 iscq10
    change_rate_20d = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="20日涨跌幅")  # 原 zdf20
    has_ex_dividend_20d = models.IntegerField(verbose_name="近20日是否除权除息")  # 原 iscq20
    
    class Meta:
        verbose_name = "盘中创新高个股"
        db_table = "new_high_stock"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['trade_date']),
            models.Index(fields=['stock_code']),
        ]


class NewLowStock(BaseModel):
    """盘中创新低个股"""
    trade_date = models.DateField(verbose_name="日期")  # 原 t
    stock_code = models.CharField(max_length=10, verbose_name="代码")  # 原 dm
    stock_name = models.CharField(max_length=50, verbose_name="名称")  # 原 mc
    close_price = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="收盘价")  # 原 c
    high_price = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="最高价")  # 原 h
    low_price = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="最低价")  # 原 l
    change_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="涨跌幅")  # 原 zdf
    is_ex_dividend = models.IntegerField(verbose_name="当天是否除权除息")  # 原 iscq
    turnover_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="换手率")  # 原 hs
    change_rate_5d = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="5日涨跌幅")  # 原 zdf5
    has_ex_dividend_5d = models.IntegerField(verbose_name="近5日是否除权除息")  # 原 iscq5
    change_rate_10d = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="10日涨跌幅")  # 原 zdf10
    has_ex_dividend_10d = models.IntegerField(verbose_name="近10日是否除权除息")  # 原 iscq10
    change_rate_20d = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="20日涨跌幅")  # 原 zdf20
    has_ex_dividend_20d = models.IntegerField(verbose_name="近20日是否除权除息")  # 原 iscq20
    
    class Meta:
        verbose_name = "盘中创新低个股"
        db_table = "new_low_stock"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['trade_date']),
            models.Index(fields=['stock_code']),
        ]