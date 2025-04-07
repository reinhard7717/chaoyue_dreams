from django.db import models
from django.utils.translation import gettext_lazy as _
from bulk_update_or_create import BulkUpdateOrCreateQuerySet

class IndexVrocFIB(models.Model):
    """VROC (成交量变动速率) 指标存储模型 (斐波那契周期)"""
    index = models.ForeignKey('IndexInfo', on_delete=models.CASCADE, related_name="vroc_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")

    # 存储成交量的变化率 (通常是百分比) 或绝对差值
    # 如果是百分比，DecimalField 合适
    # 如果是绝对差值 (volume.diff(n))，BigIntegerField 可能更合适，取决于 volume 的类型
    # 这里假设存储的是类似 ROC 的比率值
    vroc5 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="VROC(5)") # 增加 max_digits 以防万一
    vroc8 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="VROC(8)")
    vroc13 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="VROC(13)")
    vroc21 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="VROC(21)")
    vroc34 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="VROC(34)")
    vroc55 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="VROC(55)")
    vroc89 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="VROC(89)")
    vroc144 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="VROC(144)")
    vroc233 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="VROC(233)")

    class Meta:
        verbose_name = "VROC指标(斐波那契)"
        db_table = 'index_vroc_fib'
        verbose_name_plural = verbose_name
        unique_together = ('index', 'trade_time', 'time_level')
        indexes = [
            models.Index(fields=['index', 'time_level', 'trade_time']),
        ]


class StockVrocFIB(models.Model):
    """VROC (成交量变动速率) 指标存储模型 (斐波那契周期)"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, related_name="vroc_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")

    # 存储成交量的变化率 (通常是百分比) 或绝对差值
    # 如果是百分比，DecimalField 合适
    # 如果是绝对差值 (volume.diff(n))，BigIntegerField 可能更合适，取决于 volume 的类型
    # 这里假设存储的是类似 ROC 的比率值
    vroc5 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="VROC(5)") # 增加 max_digits 以防万一
    vroc8 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="VROC(8)")
    vroc13 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="VROC(13)")
    vroc21 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="VROC(21)")
    vroc34 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="VROC(34)")
    vroc55 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="VROC(55)")
    vroc89 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="VROC(89)")
    vroc144 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="VROC(144)")
    vroc233 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="VROC(233)")

    class Meta:
        verbose_name = "VROC指标(斐波那契)"
        db_table = 'stock_vroc_fib'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time', 'time_level')
        indexes = [
            models.Index(fields=['stock', 'time_level', 'trade_time']),
        ]
