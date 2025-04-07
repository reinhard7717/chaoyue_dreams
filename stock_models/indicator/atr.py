from django.db import models
from django.utils.translation import gettext_lazy as _
from bulk_update_or_create import BulkUpdateOrCreateQuerySet


class IndexAtrFIB(models.Model):
    """ATR 指标存储模型 (斐波那契周期)"""
    index = models.ForeignKey('IndexInfo', on_delete=models.CASCADE, related_name="atr_fib", verbose_name="指数")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")
    atr5 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ATR(5)")
    atr8 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ATR(8)")
    atr13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ATR(13)")
    atr21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ATR(21)")
    atr34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ATR(34)")
    atr55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ATR(55)")
    atr89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ATR(89)")
    atr144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ATR(144)")
    atr233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ATR(233)")

    class Meta:
        verbose_name = "ATR指标(斐波那契)"
        db_table = 'index_atr_fib'
        verbose_name_plural = verbose_name
        unique_together = ('index', 'trade_time', 'time_level')
        indexes = [ models.Index(fields=['index', 'time_level', 'trade_time']), ]


class StockAtrFIB(models.Model):
    """ATR 指标存储模型 (斐波那契周期)"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, related_name="atr_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")
    atr5 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ATR(5)")
    atr8 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ATR(8)")
    atr13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ATR(13)")
    atr21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ATR(21)")
    atr34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ATR(34)")
    atr55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ATR(55)")
    atr89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ATR(89)")
    atr144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ATR(144)")
    atr233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ATR(233)")

    class Meta:
        verbose_name = "ATR指标(斐波那契)"
        db_table = 'stock_atr_fib'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time', 'time_level')
        indexes = [ models.Index(fields=['stock', 'time_level', 'trade_time']), ]
