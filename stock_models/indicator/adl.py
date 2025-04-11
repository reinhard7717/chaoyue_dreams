from django.db import models
from django.utils.translation import gettext_lazy as _

# --- Accumulation/Distribution Line (ADL) ---
class IndexAdl(models.Model):
    """Accumulation/Distribution Line (ADL) 指标存储模型"""
    index = models.ForeignKey('IndexInfo', on_delete=models.CASCADE, related_name="adl", verbose_name="指数")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")
    # ADL 是累积值，可能很大，使用较大的 max_digits
    adl = models.DecimalField(max_digits=20, decimal_places=4, null=True, blank=True, verbose_name="ADL")

    class Meta:
        verbose_name = "集散线(ADL)"
        db_table = 'index_adl'
        verbose_name_plural = verbose_name
        unique_together = ('index', 'trade_time', 'time_level')
        indexes = [ models.Index(fields=['index', 'time_level', 'trade_time']), ]

# --- Accumulation/Distribution Line (ADL) ---
class StockAdl(models.Model):
    """Accumulation/Distribution Line (ADL) 指标存储模型"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, related_name="adl", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")
    # ADL 是累积值，可能很大，使用较大的 max_digits
    adl = models.DecimalField(max_digits=20, decimal_places=4, null=True, blank=True, verbose_name="ADL")

    class Meta:
        verbose_name = "集散线(ADL)"
        db_table = 'stock_adl'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time', 'time_level')
        indexes = [ models.Index(fields=['stock', 'time_level', 'trade_time']), ]
