from django.db import models
from django.utils.translation import gettext_lazy as _
from bulk_update_or_create import BulkUpdateOrCreateQuerySet

class IndexRocFIB(models.Model):
    """ROC (变动速率) 指标存储模型 (斐波那契周期)"""
    index = models.ForeignKey('IndexInfo', on_delete=models.CASCADE, related_name="roc_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")

    roc5 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ROC(5)")
    roc8 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ROC(8)")
    roc13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ROC(13)")
    roc21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ROC(21)")
    roc34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ROC(34)")
    roc55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ROC(55)")
    roc89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ROC(89)")
    roc144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ROC(144)")
    roc233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ROC(233)")

    class Meta:
        verbose_name = "ROC指标(斐波那契)"
        db_table = 'index_roc_fib'
        verbose_name_plural = verbose_name
        unique_together = ('index', 'trade_time', 'time_level')
        indexes = [
            models.Index(fields=['index', 'time_level', 'trade_time']),
        ]

class StockRocFIB(models.Model):
    """ROC (变动速率) 指标存储模型 (斐波那契周期)"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, related_name="roc_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")

    roc5 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ROC(5)")
    roc8 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ROC(8)")
    roc13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ROC(13)")
    roc21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ROC(21)")
    roc34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ROC(34)")
    roc55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ROC(55)")
    roc89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ROC(89)")
    roc144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ROC(144)")
    roc233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ROC(233)")

    class Meta:
        verbose_name = "ROC指标(斐波那契)"
        db_table = 'stock_roc_fib'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time', 'time_level')
        indexes = [
            models.Index(fields=['stock', 'time_level', 'trade_time']),
        ]

class IndexAmountRocFIB(models.Model):
    """成交额变动速率 (AROC) 指标存储模型 (斐波那契周期)"""
    index = models.ForeignKey('IndexInfo', on_delete=models.CASCADE, related_name="amount_roc_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")

    # 存储成交额的变化率 (通常是百分比)
    aroc5 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="AROC(5)")
    aroc8 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="AROC(8)")
    aroc13 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="AROC(13)")
    aroc21 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="AROC(21)")
    aroc34 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="AROC(34)")
    aroc55 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="AROC(55)")
    aroc89 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="AROC(89)")
    aroc144 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="AROC(144)")
    aroc233 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="AROC(233)")

    class Meta:
        verbose_name = "成交额ROC指标(斐波那契)"
        db_table = 'index_amount_roc_fib'
        verbose_name_plural = verbose_name
        unique_together = ('index', 'trade_time', 'time_level')
        indexes = [
            models.Index(fields=['index', 'time_level', 'trade_time']),
        ]

class StockAmountRocFIB(models.Model):
    """成交额变动速率 (AROC) 指标存储模型 (斐波那契周期)"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, related_name="amount_roc_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")

    # 存储成交额的变化率 (通常是百分比)
    aroc5 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="AROC(5)")
    aroc8 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="AROC(8)")
    aroc13 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="AROC(13)")
    aroc21 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="AROC(21)")
    aroc34 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="AROC(34)")
    aroc55 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="AROC(55)")
    aroc89 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="AROC(89)")
    aroc144 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="AROC(144)")
    aroc233 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="AROC(233)")

    class Meta:
        verbose_name = "成交额ROC指标(斐波那契)"
        db_table = 'stock_amount_roc_fib'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time', 'time_level')
        indexes = [
            models.Index(fields=['stock', 'time_level', 'trade_time']),
        ]


