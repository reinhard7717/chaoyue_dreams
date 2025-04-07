from django.db import models
from django.utils.translation import gettext_lazy as _
from bulk_update_or_create import BulkUpdateOrCreateQuerySet

class IndexMACDData(models.Model):
    """
    MACD指标数据模型
    
    存储股票指数的MACD技术指标数据
    """
    index = models.ForeignKey('IndexInfo', on_delete=models.CASCADE, related_name="macd_data", verbose_name=_("股票指数"))
    time_level = models.CharField(max_length=10, verbose_name=_("时间级别"))  # 5, 15, 30, 60, Day, Week, Month, Year
    trade_time = models.DateTimeField(verbose_name=_("交易时间"))
    diff = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("DIFF值"))
    dea = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("DEA值"))
    macd = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("MACD值"))
    ema12 = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("EMA(12)值"))
    ema26 = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("EMA(26)值"))
    
    class Meta:
        verbose_name = _("MACD指标数据")
        verbose_name_plural = _("MACD指标数据")
        db_table = "index_macd_data"
        unique_together = [['index', 'time_level', 'trade_time']]
        indexes = [
            models.Index(fields=['index']),
            models.Index(fields=['time_level']),
            models.Index(fields=['trade_time']),
        ]
    
    def __str__(self):
        return f"{self.index.name} MACD {self.time_level}({self.trade_time})"

    def __code__(self):
        return self.index.code

class StockMACDIndicator(models.Model):
    """
    MACD指标数据模型
    """
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, blank=True, null=True, related_name="macd_indicator", verbose_name=_("股票"))
    time_level = models.CharField(max_length=10, verbose_name='分时级别')
    trade_time = models.DateTimeField(verbose_name='交易时间')
    diff = models.DecimalField(max_digits=10, decimal_places=3, verbose_name='DIFF值', null=True)
    dea = models.DecimalField(max_digits=10, decimal_places=3, verbose_name='DEA值', null=True)
    macd = models.DecimalField(max_digits=10, decimal_places=3, verbose_name='MACD值', null=True)
    ema12 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='EMA(12)值', null=True)
    ema26 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='EMA(26)值', null=True)
    
    class Meta:
        verbose_name = 'MACD指标数据'
        verbose_name_plural = verbose_name
        db_table = 'stock_macd_indicator'
        unique_together = ('stock', 'time_level', 'trade_time')
        ordering = ['stock', 'time_level', 'trade_time']
    
    def __str__(self):
        return f"{self.stock.stock_code}-{self.time_level}-{self.trade_time}"
    
    def __code__(self):
        return self.stock.stock_code

class IndexMACDFIB(models.Model):
    """
    MACD指标数据模型 (包含标准MACD和斐波那契周期EMA)
    """
    index = models.ForeignKey('IndexInfo', on_delete=models.CASCADE, blank=True, null=True, related_name="macd_fib", verbose_name=_("股票"))
    time_level = models.CharField(max_length=10, verbose_name='分时级别', db_index=True) # 建议为常用查询字段加索引
    trade_time = models.DateTimeField(verbose_name='交易时间', db_index=True) # 建议为常用查询字段加索引

    # 标准 MACD (12, 26, 9) 相关值
    diff = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='DIFF(12,26)', null=True, blank=True) # 增加精度
    dea = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='DEA(9)', null=True, blank=True) # 增加精度
    macd = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='MACD柱', null=True, blank=True) # 增加精度

    # 新增：基于斐波那契周期的 EMA 值
    ema5 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="EMA(5)", null=True, blank=True)
    ema8 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="EMA(8)", null=True, blank=True)
    ema13 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="EMA(13)", null=True, blank=True)
    ema21 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="EMA(21)", null=True, blank=True)
    ema34 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="EMA(34)", null=True, blank=True)
    ema55 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="EMA(55)", null=True, blank=True)
    ema89 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="EMA(89)", null=True, blank=True)
    ema144 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="EMA(144)", null=True, blank=True)
    ema233 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="EMA(233)", null=True, blank=True)

    class Meta:
        verbose_name = 'MACD及EMA指标数据' # 更新名称以反映内容
        verbose_name_plural = verbose_name
        db_table = 'index_macd_fib' # 表名可以保持不变
        unique_together = ('index', 'time_level', 'trade_time')
        ordering = ['index', 'time_level', 'trade_time']
        # 为常用查询添加索引
        indexes = [
            models.Index(fields=['index', 'time_level', 'trade_time']),
            models.Index(fields=['trade_time']),
        ]
    
    def __str__(self):
        return f"{self.index.code}-{self.time_level}-{self.trade_time}"
    
    def __code__(self):
        return self.index.code

class StockMACDFIB(models.Model):
    """
    MACD指标数据模型 (包含标准MACD和斐波那契周期EMA)
    """
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, blank=True, null=True, related_name="macd_fib", verbose_name=_("股票"))
    time_level = models.CharField(max_length=10, verbose_name='分时级别', db_index=True) # 建议为常用查询字段加索引
    trade_time = models.DateTimeField(verbose_name='交易时间', db_index=True) # 建议为常用查询字段加索引

    # 标准 MACD (12, 26, 9) 相关值
    diff = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='DIFF(12,26)', null=True, blank=True) # 增加精度
    dea = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='DEA(9)', null=True, blank=True) # 增加精度
    macd = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='MACD柱', null=True, blank=True) # 增加精度

    # 新增：基于斐波那契周期的 EMA 值
    ema5 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="EMA(5)", null=True, blank=True)
    ema8 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="EMA(8)", null=True, blank=True)
    ema13 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="EMA(13)", null=True, blank=True)
    ema21 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="EMA(21)", null=True, blank=True)
    ema34 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="EMA(34)", null=True, blank=True)
    ema55 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="EMA(55)", null=True, blank=True)
    ema89 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="EMA(89)", null=True, blank=True)
    ema144 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="EMA(144)", null=True, blank=True)
    ema233 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="EMA(233)", null=True, blank=True)

    class Meta:
        verbose_name = 'MACD及EMA指标数据' # 更新名称以反映内容
        verbose_name_plural = verbose_name
        db_table = 'stock_macd_fib' # 表名可以保持不变
        unique_together = ('stock', 'time_level', 'trade_time')
        ordering = ['stock', 'time_level', 'trade_time']
        # 为常用查询添加索引
        indexes = [
            models.Index(fields=['stock', 'time_level', 'trade_time']),
            models.Index(fields=['trade_time']),
        ]
    
    def __str__(self):
        return f"{self.stock.stock_code}-{self.time_level}-{self.trade_time}"
    
    def __code__(self):
        return self.stock.stock_code
