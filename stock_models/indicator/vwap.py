from django.db import models
from django.utils.translation import gettext_lazy as _
from bulk_update_or_create import BulkUpdateOrCreateQuerySet

class IndexVwap(models.Model):
    """VWAP (成交量加权平均价) 指标存储模型"""
    index = models.ForeignKey('IndexInfo', on_delete=models.CASCADE, related_name="vwap_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期") # 例如 'Day', '5min', '15min' 等

    # VWAP 值，精度应与价格类似
    vwap = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="VWAP")

    class Meta:
        verbose_name = "VWAP指标"
        db_table = 'index_vwap_fib'
        verbose_name_plural = verbose_name
        unique_together = ('index', 'trade_time', 'time_level') # 确保每个时间点只有一个VWAP值
        indexes = [
            models.Index(fields=['index', 'time_level', 'trade_time']),
        ]

    def __str__(self):
        return f"{self.stock.code} - {self.time_level} - {self.trade_time} - VWAP: {self.vwap}"

class StockVwap(models.Model):
    """VWAP (成交量加权平均价) 指标存储模型"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, related_name="vwap_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期") # 例如 'Day', '5min', '15min' 等

    # VWAP 值，精度应与价格类似
    vwap = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="VWAP")

    class Meta:
        verbose_name = "VWAP指标"
        db_table = 'stock_vwap_fib'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time', 'time_level') # 确保每个时间点只有一个VWAP值
        indexes = [
            models.Index(fields=['stock', 'time_level', 'trade_time']),
        ]

    def __str__(self):
        return f"{self.stock.code} - {self.time_level} - {self.trade_time} - VWAP: {self.vwap}"
