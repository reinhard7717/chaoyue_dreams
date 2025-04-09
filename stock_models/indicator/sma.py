from django.db import models
from django.utils.translation import gettext_lazy as _

# --- SMA (斐波那契周期) ---
class IndexSmaFIB(models.Model):
    """SMA 指标存储模型 (斐波那契周期)"""
    index = models.ForeignKey('IndexInfo', on_delete=models.CASCADE, related_name="sma_fib", verbose_name="指数")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")
    sma5 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="SMA(5)")
    sma8 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="SMA(8)")
    sma13 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="SMA(13)")
    sma21 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="SMA(21)")
    sma34 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="SMA(34)")
    sma55 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="SMA(55)")
    sma89 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="SMA(89)")
    sma144 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="SMA(144)")
    sma233 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="SMA(233)")

    class Meta:
        verbose_name = "SMA指标(斐波那契)"
        db_table = 'index_sma_fib'
        verbose_name_plural = verbose_name
        unique_together = ('index', 'trade_time', 'time_level')
        indexes = [ models.Index(fields=['index', 'time_level', 'trade_time']), ]


# --- SMA (斐波那契周期) ---
class StockSmaFIB(models.Model):
    """SMA 指标存储模型 (斐波那契周期)"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, related_name="sma_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")
    sma5 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="SMA(5)")
    sma8 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="SMA(8)")
    sma13 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="SMA(13)")
    sma21 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="SMA(21)")
    sma34 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="SMA(34)")
    sma55 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="SMA(55)")
    sma89 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="SMA(89)")
    sma144 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="SMA(144)")
    sma233 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="SMA(233)")

    class Meta:
        verbose_name = "SMA指标(斐波那契)"
        db_table = 'stock_sma_fib'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time', 'time_level')
        indexes = [ models.Index(fields=['stock', 'time_level', 'trade_time']), ]
