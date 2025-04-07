from django.db import models
from django.utils.translation import gettext_lazy as _
from bulk_update_or_create import BulkUpdateOrCreateQuerySet

class IndexCmfFIB(models.Model):
    """CMF 指标存储模型 (斐波那契周期)"""
    index = models.ForeignKey('IndexInfo', on_delete=models.CASCADE, related_name="cmf_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")
    cmf5 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CMF(5)")
    cmf8 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CMF(8)")
    cmf13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CMF(13)")
    cmf21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CMF(21)")
    cmf34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CMF(34)")
    cmf55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CMF(55)")
    cmf89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CMF(89)")
    cmf144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CMF(144)")
    cmf233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CMF(233)")

    class Meta:
        verbose_name = "CMF指标(斐波那契)"
        db_table = 'index_cmf_fib'
        verbose_name_plural = verbose_name
        unique_together = ('index', 'trade_time', 'time_level')
        indexes = [ models.Index(fields=['index', 'time_level', 'trade_time']), ]


class StockCmfFIB(models.Model):
    """CMF 指标存储模型 (斐波那契周期)"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, related_name="cmf_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")
    cmf5 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CMF(5)")
    cmf8 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CMF(8)")
    cmf13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CMF(13)")
    cmf21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CMF(21)")
    cmf34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CMF(34)")
    cmf55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CMF(55)")
    cmf89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CMF(89)")
    cmf144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CMF(144)")
    cmf233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CMF(233)")

    class Meta:
        verbose_name = "CMF指标(斐波那契)"
        db_table = 'stock_cmf_fib'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time', 'time_level')
        indexes = [ models.Index(fields=['stock', 'time_level', 'trade_time']), ]
