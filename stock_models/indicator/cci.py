from django.db import models
from django.utils.translation import gettext_lazy as _
from bulk_update_or_create import BulkUpdateOrCreateQuerySet

class IndexCciFIB(models.Model):
    """CCI 指标存储模型 (斐波那契周期)"""
    index = models.ForeignKey('IndexInfo', on_delete=models.CASCADE, related_name="cci_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")
    cci5 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CCI(5)")
    cci8 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CCI(8)")
    cci13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CCI(13)")
    cci21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CCI(21)")
    cci34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CCI(34)")
    cci55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CCI(55)")
    cci89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CCI(89)")
    cci144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CCI(144)")
    cci233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CCI(233)")

    class Meta:
        verbose_name = "CCI指标(斐波那契)"
        db_table = 'index_cci_fib'
        verbose_name_plural = verbose_name
        unique_together = ('index', 'trade_time', 'time_level')
        indexes = [ models.Index(fields=['index', 'time_level', 'trade_time']), ]


class StockCciFIB(models.Model):
    """CCI 指标存储模型 (斐波那契周期)"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, related_name="cci_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")
    cci5 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CCI(5)")
    cci8 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CCI(8)")
    cci13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CCI(13)")
    cci21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CCI(21)")
    cci34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CCI(34)")
    cci55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CCI(55)")
    cci89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CCI(89)")
    cci144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CCI(144)")
    cci233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CCI(233)")

    class Meta:
        verbose_name = "CCI指标(斐波那契)"
        db_table = 'stock_cci_fib'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time', 'time_level')
        indexes = [ models.Index(fields=['stock', 'time_level', 'trade_time']), ]
