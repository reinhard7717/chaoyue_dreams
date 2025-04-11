from django.db import models
from django.utils.translation import gettext_lazy as _
from bulk_update_or_create import BulkUpdateOrCreateQuerySet

class IndexIchimoku(models.Model):
    """Ichimoku Cloud (一目均衡表) 指标存储模型"""
    index = models.ForeignKey('IndexInfo', on_delete=models.CASCADE, related_name="ichimoku_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")

    # finta.TA.ICHIMOKU 默认返回 'TENKAN', 'KIJUN', 'CHIKOU', 'SENKOU A', 'SENKOU B'
    # 使用 finta 返回的列名作为字段名，或者映射为你喜欢的名字
    tenkan_sen = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="转换线 (Tenkan Sen)")
    kijun_sen = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="基准线 (Kijun Sen)")
    chikou_span = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="延迟线 (Chikou Span)")
    senkou_span_a = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="先行带A (Senkou Span A)")
    senkou_span_b = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="先行带B (Senkou Span B)")

    class Meta:
        verbose_name = "Ichimoku指标"
        db_table = 'index_ichimoku'
        verbose_name_plural = verbose_name
        unique_together = ('index', 'trade_time', 'time_level')
        indexes = [
            models.Index(fields=['index', 'time_level', 'trade_time']),
        ]

class StockIchimoku(models.Model):
    """Ichimoku Cloud (一目均衡表) 指标存储模型"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, related_name="ichimoku_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")

    # finta.TA.ICHIMOKU 默认返回 'TENKAN', 'KIJUN', 'CHIKOU', 'SENKOU A', 'SENKOU B'
    # 使用 finta 返回的列名作为字段名，或者映射为你喜欢的名字
    tenkan_sen = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="转换线 (Tenkan Sen)")
    kijun_sen = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="基准线 (Kijun Sen)")
    chikou_span = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="延迟线 (Chikou Span)")
    senkou_span_a = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="先行带A (Senkou Span A)")
    senkou_span_b = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="先行带B (Senkou Span B)")

    class Meta:
        verbose_name = "Ichimoku指标"
        db_table = 'stock_ichimoku'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time', 'time_level')
        indexes = [
            models.Index(fields=['stock', 'time_level', 'trade_time']),
        ]
