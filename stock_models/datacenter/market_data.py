# models/datacenter/market_data.py
from django.db import models
from django.utils.translation import gettext_lazy as _
from utils.models import BaseModel
from stock_models.stock_basic import StockInfo

class VolumeIncrease(BaseModel):
    """成交骤增个股"""
    trade_date = models.DateField(verbose_name="日期")  # 原 t
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, blank=True, null=True, related_name="volume_increase", verbose_name=_("股票"))
    close_price = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="收盘价")  # 原 c
    change_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="涨跌幅")  # 原 zdf
    is_ex_dividend = models.IntegerField(verbose_name="当天是否除权除息")  # 原 iscq
    volume = models.BigIntegerField(verbose_name="成交量")  # 原 v
    previous_volume = models.BigIntegerField(verbose_name="前一交易日成交量")  # 原 pv
    volume_change = models.BigIntegerField(verbose_name="增减量")  # 原 zjl
    volume_change_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="增减幅")  # 原 zjf
    
    class Meta:
        verbose_name = "成交骤增个股"
        db_table = "volume_increase"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['trade_date']),
            models.Index(fields=['stock']),
        ]

    def __code__(self):
        return self.stock.stock_code



class VolumeDecrease(BaseModel):
    """成交骤减个股"""
    trade_date = models.DateField(verbose_name="日期")  # 原 t
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, blank=True, null=True, related_name="volume_decrease", verbose_name=_("股票"))
    close_price = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="收盘价")  # 原 c
    change_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="涨跌幅")  # 原 zdf
    is_ex_dividend = models.IntegerField(verbose_name="当天是否除权除息")  # 原 iscq
    volume = models.BigIntegerField(verbose_name="成交量")  # 原 v
    previous_volume = models.BigIntegerField(verbose_name="前一交易日成交量")  # 原 pv
    volume_change = models.BigIntegerField(verbose_name="增减量")  # 原 zjl
    volume_change_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="增减幅")  # 原 zjf
    
    class Meta:
        verbose_name = "成交骤减个股"
        db_table = "volume_decrease"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['trade_date']),
            models.Index(fields=['stock']),
        ]

    def __code__(self):
        return self.stock.stock_code



class ContinuousVolumeIncrease(BaseModel):
    """连续放量个股"""
    trade_date = models.DateField(verbose_name="日期")  # 原 t
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, blank=True, null=True, related_name="continuous_volume_increase", verbose_name=_("股票"))
    close_price = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="收盘价")  # 原 c
    change_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="涨跌幅")  # 原 zdf
    is_ex_dividend = models.IntegerField(verbose_name="当天是否除权除息")  # 原 iscq
    volume = models.BigIntegerField(verbose_name="成交量")  # 原 v
    previous_volume = models.BigIntegerField(verbose_name="前一交易日成交量")  # 原 pv
    volume_increase_days = models.IntegerField(verbose_name="放量天数")  # 原 flday
    period_change_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="阶段涨跌幅")  # 原 pzdf
    period_has_ex_dividend = models.IntegerField(verbose_name="阶段是否除权除息")  # 原 ispcq
    period_turnover_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="阶段换手率")  # 原 phs
    
    class Meta:
        verbose_name = "连续放量个股"
        db_table = "continuous_volume_increase"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['trade_date']),
            models.Index(fields=['stock']),
        ]

    def __code__(self):
        return self.stock.stock_code



class ContinuousVolumeDecrease(BaseModel):
    """连续缩量个股"""
    trade_date = models.DateField(verbose_name="日期")  # 原 t
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, blank=True, null=True, related_name="continuous_volume_decrease", verbose_name=_("股票"))
    close_price = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="收盘价")  # 原 c
    change_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="涨跌幅")  # 原 zdf
    is_ex_dividend = models.IntegerField(verbose_name="当天是否除权除息")  # 原 iscq
    volume = models.BigIntegerField(verbose_name="成交量")  # 原 v
    previous_volume = models.BigIntegerField(verbose_name="前一交易日成交量")  # 原 pv
    volume_decrease_days = models.IntegerField(verbose_name="缩量天数")  # 原 flday
    period_change_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="阶段涨跌幅")  # 原 pzdf
    period_has_ex_dividend = models.IntegerField(verbose_name="阶段是否除权除息")  # 原 ispcq
    period_turnover_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="阶段换手率")  # 原 phs
    
    class Meta:
        verbose_name = "连续缩量个股"
        db_table = "continuous_volume_decrease"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['trade_date']),
            models.Index(fields=['stock']),
        ]

    def __code__(self):
        return self.stock.stock_code



class ContinuousRise(BaseModel):
    """连续上涨个股"""
    trade_date = models.DateField(verbose_name="日期")  # 原 t
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, blank=True, null=True, related_name="continuous_rise", verbose_name=_("股票"))
    close_price = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="收盘价")  # 原 c
    change_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="涨跌幅")  # 原 zdf
    is_ex_dividend = models.IntegerField(verbose_name="当天是否除权除息")  # 原 iscq
    volume = models.BigIntegerField(verbose_name="成交量")  # 原 v
    turnover_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="换手")  # 原 hs
    rising_days = models.IntegerField(verbose_name="上涨天数")  # 原 szday
    period_change_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="阶段涨跌幅")  # 原 pzdf
    period_has_ex_dividend = models.IntegerField(verbose_name="阶段是否除权除息")  # 原 ispcq
    
    class Meta:
        verbose_name = "连续上涨个股"
        db_table = "continuous_rise"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['trade_date']),
            models.Index(fields=['stock']),
        ]

    def __code__(self):
        return self.stock.stock_code


class ContinuousFall(BaseModel):
    """连续下跌个股"""
    trade_date = models.DateField(verbose_name="日期")  # 原 t
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, blank=True, null=True, related_name="continuous_fall", verbose_name=_("股票"))
    close_price = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="收盘价")  # 原 c
    change_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="涨跌幅")  # 原 zdf
    is_ex_dividend = models.IntegerField(verbose_name="当天是否除权除息")  # 原 iscq
    volume = models.BigIntegerField(verbose_name="成交量")  # 原 v
    turnover_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="换手")  # 原 hs
    falling_days = models.IntegerField(verbose_name="下跌天数")  # 原 szday
    period_change_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="阶段涨跌幅")  # 原 pzdf
    period_has_ex_dividend = models.IntegerField(verbose_name="阶段是否除权除息")  # 原 ispcq
    
    class Meta:
        verbose_name = "连续下跌个股"
        db_table = "continuous_fall"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['trade_date']),
            models.Index(fields=['stock']),
        ]

    def __code__(self):
        return self.stock.stock_code



