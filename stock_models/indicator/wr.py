from django.db import models
from django.utils.translation import gettext_lazy as _

class IndexWrFIB(models.Model):
    """WR 指标存储模型 (斐波那契周期)"""
    index = models.ForeignKey('IndexInfo', on_delete=models.CASCADE, related_name="wr_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")
    wr5 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="WR(5)")
    wr8 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="WR(8)")
    wr13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="WR(13)")
    wr21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="WR(21)")
    wr34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="WR(34)")
    wr55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="WR(55)")
    wr89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="WR(89)")
    wr144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="WR(144)")
    wr233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="WR(233)")

    class Meta:
        verbose_name = "WR指标(斐波那契)"
        db_table = 'index_wr_fib'
        verbose_name_plural = verbose_name
        unique_together = ('index', 'trade_time', 'time_level')
        indexes = [ models.Index(fields=['index', 'time_level', 'trade_time']), ]


class StockWrFIB(models.Model):
    """WR 指标存储模型 (斐波那契周期)"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, related_name="wr_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")
    wr5 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="WR(5)")
    wr8 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="WR(8)")
    wr13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="WR(13)")
    wr21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="WR(21)")
    wr34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="WR(34)")
    wr55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="WR(55)")
    wr89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="WR(89)")
    wr144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="WR(144)")
    wr233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="WR(233)")

    class Meta:
        verbose_name = "WR指标(斐波那契)"
        db_table = 'stock_wr_fib'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time', 'time_level')
        indexes = [ models.Index(fields=['stock', 'time_level', 'trade_time']), ]
