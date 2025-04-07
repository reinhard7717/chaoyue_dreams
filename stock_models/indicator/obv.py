from django.db import models
from django.utils.translation import gettext_lazy as _


class IndexObvFIB(models.Model):
    """OBV 指标存储模型""" # OBV 通常不带周期参数
    index = models.ForeignKey('IndexInfo', on_delete=models.CASCADE, related_name="obv_fib", verbose_name="指数")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")
    obv = models.BigIntegerField(null=True, blank=True, verbose_name="OBV")

    class Meta:
        verbose_name = "OBV指标"
        db_table = 'index_obv_fib'
        verbose_name_plural = verbose_name
        unique_together = ('index', 'trade_time', 'time_level')
        indexes = [ models.Index(fields=['index', 'time_level', 'trade_time']), ]


class StockObvFIB(models.Model):
    """OBV 指标存储模型""" # OBV 通常不带周期参数
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, related_name="obv_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")
    obv = models.BigIntegerField(null=True, blank=True, verbose_name="OBV")

    class Meta:
        verbose_name = "OBV指标"
        db_table = 'stock_obv_fib'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time', 'time_level')
        indexes = [ models.Index(fields=['stock', 'time_level', 'trade_time']), ]
