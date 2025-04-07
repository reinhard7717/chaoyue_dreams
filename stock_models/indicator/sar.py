from django.db import models
from django.utils.translation import gettext_lazy as _
from bulk_update_or_create import BulkUpdateOrCreateQuerySet

class IndexSar(models.Model):
    """SAR 指标存储模型""" # SAR 参数不是简单的周期
    index = models.ForeignKey('IndexInfo', on_delete=models.CASCADE, related_name="sar_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")
    sar = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="SAR")

    class Meta:
        verbose_name = "SAR指标"
        db_table = 'index_sar'
        verbose_name_plural = verbose_name
        unique_together = ('index', 'trade_time', 'time_level')
        indexes = [ models.Index(fields=['index', 'time_level', 'trade_time']), ]


class StockSar(models.Model):
    """SAR 指标存储模型""" # SAR 参数不是简单的周期
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, related_name="sar_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")
    sar = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="SAR")

    class Meta:
        verbose_name = "SAR指标"
        db_table = 'stock_sar'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time', 'time_level')
        indexes = [ models.Index(fields=['stock', 'time_level', 'trade_time']), ]
