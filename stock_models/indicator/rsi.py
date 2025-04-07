from django.db import models
from django.utils.translation import gettext_lazy as _
from bulk_update_or_create import BulkUpdateOrCreateQuerySet

class IndexRsiFIB(models.Model):
    """RSI 指标存储模型 (斐波那契周期)"""
    index = models.ForeignKey('IndexInfo', on_delete=models.CASCADE, related_name="rsi_fib", verbose_name="指数")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")
    rsi5 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="RSI(5)")
    rsi8 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="RSI(8)")
    rsi13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="RSI(13)")
    rsi21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="RSI(21)")
    rsi34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="RSI(34)")
    rsi55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="RSI(55)")
    rsi89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="RSI(89)")
    rsi144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="RSI(144)")
    rsi233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="RSI(233)")

    class Meta:
        verbose_name = "RSI指标(斐波那契)"
        db_table = 'index_rsi_fib'
        verbose_name_plural = verbose_name
        unique_together = ('index', 'trade_time', 'time_level')
        indexes = [ models.Index(fields=['index', 'time_level', 'trade_time']), ]


class StockRsiFIB(models.Model):
    """RSI 指标存储模型 (斐波那契周期)"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, related_name="rsi_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")
    rsi5 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="RSI(5)")
    rsi8 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="RSI(8)")
    rsi13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="RSI(13)")
    rsi21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="RSI(21)")
    rsi34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="RSI(34)")
    rsi55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="RSI(55)")
    rsi89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="RSI(89)")
    rsi144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="RSI(144)")
    rsi233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="RSI(233)")

    class Meta:
        verbose_name = "RSI指标(斐波那契)"
        db_table = 'stock_rsi_fib'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time', 'time_level')
        indexes = [ models.Index(fields=['stock', 'time_level', 'trade_time']), ]
