from django.db import models
from django.utils.translation import gettext_lazy as _

class IndexMomFIB(models.Model):
    """MOM (动量) 指标存储模型 (斐波那契周期)"""
    index = models.ForeignKey('IndexInfo', on_delete=models.CASCADE, related_name="mom_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")

    mom5 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MOM(5)")
    mom8 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MOM(8)")
    mom13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MOM(13)")
    mom21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MOM(21)")
    mom34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MOM(34)")
    mom55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MOM(55)")
    mom89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MOM(89)")
    mom144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MOM(144)")
    mom233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MOM(233)")

    class Meta:
        verbose_name = "MOM指标(斐波那契)"
        db_table = 'index_mom_fib'
        verbose_name_plural = verbose_name
        unique_together = ('index', 'trade_time', 'time_level')
        indexes = [
            models.Index(fields=['index', 'time_level', 'trade_time']),
        ]


class StockMomFIB(models.Model):
    """MOM (动量) 指标存储模型 (斐波那契周期)"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, related_name="mom_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")

    mom5 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MOM(5)")
    mom8 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MOM(8)")
    mom13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MOM(13)")
    mom21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MOM(21)")
    mom34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MOM(34)")
    mom55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MOM(55)")
    mom89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MOM(89)")
    mom144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MOM(144)")
    mom233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MOM(233)")

    class Meta:
        verbose_name = "MOM指标(斐波那契)"
        db_table = 'stock_mom_fib'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time', 'time_level')
        indexes = [
            models.Index(fields=['stock', 'time_level', 'trade_time']),
        ]
