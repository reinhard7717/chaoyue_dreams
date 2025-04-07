from django.db import models
from django.utils.translation import gettext_lazy as _
from bulk_update_or_create import BulkUpdateOrCreateQuerySet

class IndexMfiFIB(models.Model):
    """MFI 指标存储模型 (斐波那契周期)"""
    index = models.ForeignKey('IndexInfo', on_delete=models.CASCADE, related_name="mfi_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")
    mfi5 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MFI(5)")
    mfi8 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MFI(8)")
    mfi13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MFI(13)") # 常用14，这里用13
    mfi21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MFI(21)")
    mfi34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MFI(34)")
    mfi55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MFI(55)")
    mfi89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MFI(89)")
    mfi144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MFI(144)")
    mfi233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MFI(233)")

    class Meta:
        verbose_name = "MFI指标(斐波那契)"
        db_table = 'index_mfi_fib'
        verbose_name_plural = verbose_name
        unique_together = ('index', 'trade_time', 'time_level')
        indexes = [ models.Index(fields=['index', 'time_level', 'trade_time']), ]


class StockMfiFIB(models.Model):
    """MFI 指标存储模型 (斐波那契周期)"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, related_name="mfi_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")
    mfi5 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MFI(5)")
    mfi8 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MFI(8)")
    mfi13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MFI(13)") # 常用14，这里用13
    mfi21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MFI(21)")
    mfi34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MFI(34)")
    mfi55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MFI(55)")
    mfi89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MFI(89)")
    mfi144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MFI(144)")
    mfi233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MFI(233)")

    class Meta:
        verbose_name = "MFI指标(斐波那契)"
        db_table = 'stock_mfi_fib'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time', 'time_level')
        indexes = [ models.Index(fields=['stock', 'time_level', 'trade_time']), ]
