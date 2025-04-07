from django.db import models
from django.utils.translation import gettext_lazy as _
from bulk_update_or_create import BulkUpdateOrCreateQuerySet


class IndexBOLLData(models.Model):
    """
    BOLL指标数据模型
    
    存储股票指数的BOLL技术指标数据
    """
    index = models.ForeignKey('IndexInfo', on_delete=models.CASCADE, related_name="boll_data", verbose_name=_("股票指数"))
    time_level = models.CharField(max_length=10, verbose_name=_("时间级别"))  # 5, 15, 30, 60, Day, Week, Month, Year
    trade_time = models.DateTimeField(verbose_name=_("交易时间"))
    upper = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("上轨"))
    middle = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("中轨"))
    lower = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("下轨"))
    
    class Meta:
        verbose_name = _("BOLL指标数据")
        verbose_name_plural = _("BOLL指标数据")
        db_table = "index_boll_data"
        unique_together = [['index', 'time_level', 'trade_time']]
        indexes = [
            models.Index(fields=['index']),
            models.Index(fields=['time_level']),
            models.Index(fields=['trade_time']),
        ]
    
    def __str__(self):
        return f"{self.index.name} BOLL {self.time_level}({self.trade_time})"

    def __code__(self):
        return self.index.code

class StockBOLLIndicator(models.Model):
    """
    BOLL指标数据模型
    """
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, blank=True, null=True, related_name="boll_indicator", verbose_name=_("股票"))
    time_level = models.CharField(max_length=10, verbose_name='分时级别')
    trade_time = models.DateTimeField(verbose_name='交易时间')
    upper = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='上轨', null=True)
    lower = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='下轨', null=True)
    mid = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='中轨', null=True)
    
    class Meta:
        verbose_name = 'BOLL指标数据'
        verbose_name_plural = verbose_name
        db_table = 'stock_boll_indicator'
        unique_together = ('stock', 'time_level', 'trade_time')
        ordering = ['stock', 'time_level', 'trade_time']
    
    def __str__(self):
        return f"{self.stock.stock_code}-{self.time_level}-{self.trade_time}"
    
    def __code__(self):
        return self.stock.stock_code
