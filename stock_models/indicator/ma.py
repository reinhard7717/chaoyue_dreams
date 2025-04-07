from django.db import models
from django.utils.translation import gettext_lazy as _
from bulk_update_or_create import BulkUpdateOrCreateQuerySet

class IndexMAData(models.Model):
    """
    MA指标数据模型
    
    存储股票指数的MA技术指标数据
    """
    index = models.ForeignKey('IndexInfo', on_delete=models.CASCADE, related_name="ma_data", verbose_name=_("股票指数"))
    time_level = models.CharField(max_length=10, verbose_name=_("时间级别"))  # 5, 15, 30, 60, Day, Week, Month, Year
    trade_time = models.DateTimeField(verbose_name=_("交易时间"))
    ma3 = models.DecimalField(max_digits=12, decimal_places=4, null=True, verbose_name=_("MA3"))
    ma5 = models.DecimalField(max_digits=12, decimal_places=4, null=True, verbose_name=_("MA5"))
    ma10 = models.DecimalField(max_digits=12, decimal_places=4, null=True, verbose_name=_("MA10"))
    ma15 = models.DecimalField(max_digits=12, decimal_places=4, null=True, verbose_name=_("MA15"))
    ma20 = models.DecimalField(max_digits=12, decimal_places=4, null=True, verbose_name=_("MA20"))
    ma30 = models.DecimalField(max_digits=12, decimal_places=4, null=True, verbose_name=_("MA30"))
    ma60 = models.DecimalField(max_digits=12, decimal_places=4, null=True, verbose_name=_("MA60"))
    ma120 = models.DecimalField(max_digits=12, decimal_places=4, null=True, verbose_name=_("MA120"))
    ma200 = models.DecimalField(max_digits=12, decimal_places=4, null=True, verbose_name=_("MA200"))
    ma250 = models.DecimalField(max_digits=12, decimal_places=4, null=True, verbose_name=_("MA250"))
    
    class Meta:
        verbose_name = _("MA指标数据")
        verbose_name_plural = _("MA指标数据")
        db_table = "index_ma_data"
        unique_together = [['index', 'time_level', 'trade_time']]
        indexes = [
            models.Index(fields=['index']),
            models.Index(fields=['time_level']),
            models.Index(fields=['trade_time']),
        ]
    
    def __str__(self):
        return f"{self.index.name} MA {self.time_level}({self.trade_time})"

    def __code__(self):
        return self.index.code

class StockMAIndicator(models.Model):
    """
    MA指标数据模型
    """
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, blank=True, null=True, related_name="ma_indicator", verbose_name=_("股票"))
    time_level = models.CharField(max_length=10, verbose_name='分时级别')
    trade_time = models.DateTimeField(verbose_name='交易时间')
    ma3 = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='MA3', null=True)
    ma10 = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='MA10', null=True)
    ma15 = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='MA15', null=True)
    ma20 = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='MA20', null=True)
    ma30 = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='MA30', null=True)
    ma60 = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='MA60', null=True)
    ma120 = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='MA120', null=True)
    ma200 = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='MA200', null=True)
    ma250 = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='MA250', null=True)

    # 新增：基于斐波那契周期的 MA 值 (假设是 SMA)
    ma8 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='MA(8)', null=True, blank=True)
    ma13 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='MA(13)', null=True, blank=True)
    ma21 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='MA(21)', null=True, blank=True)
    ma34 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='MA(34)', null=True, blank=True)
    ma55 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='MA(55)', null=True, blank=True)
    ma89 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='MA(89)', null=True, blank=True)
    ma144 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='MA(144)', null=True, blank=True)
    ma233 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='MA(233)', null=True, blank=True)

    
    class Meta:
        verbose_name = 'MA指标数据'
        verbose_name_plural = verbose_name
        db_table = 'stock_ma_indicator'
        unique_together = ('stock', 'time_level', 'trade_time')
        ordering = ['stock', 'time_level', 'trade_time']
    
    def __str__(self):
        return f"{self.stock.stock_code}-{self.time_level}-{self.trade_time}"
    
    def __code__(self):
        return self.stock.stock_code

class IndexEmaFIB(models.Model):
    """EMA 指标存储模型 (斐波那契周期)"""
    index = models.ForeignKey('IndexInfo', on_delete=models.CASCADE, related_name="ema_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")
    # 根据 FIB_PERIODS 动态添加或显式定义字段
    ema5 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="EMA(5)")
    ema8 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="EMA(8)")
    ema13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="EMA(13)")
    ema21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="EMA(21)")
    ema34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="EMA(34)")
    ema55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="EMA(55)")
    ema89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="EMA(89)")
    ema144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="EMA(144)")
    ema233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="EMA(233)")

    class Meta:
        verbose_name = "EMA指标(斐波那契)"
        db_table = 'index_ema_fib'
        verbose_name_plural = verbose_name
        unique_together = ('index', 'trade_time', 'time_level')
        indexes = [ models.Index(fields=['index', 'time_level', 'trade_time']), ]

class StockEmaFIB(models.Model):
    """EMA 指标存储模型 (斐波那契周期)"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, related_name="ema_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")
    # 根据 FIB_PERIODS 动态添加或显式定义字段
    ema5 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="EMA(5)")
    ema8 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="EMA(8)")
    ema13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="EMA(13)")
    ema21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="EMA(21)")
    ema34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="EMA(34)")
    ema55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="EMA(55)")
    ema89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="EMA(89)")
    ema144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="EMA(144)")
    ema233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="EMA(233)")

    class Meta:
        verbose_name = "EMA指标(斐波那契)"
        db_table = 'stock_ema_fib'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time', 'time_level')
        indexes = [ models.Index(fields=['stock', 'time_level', 'trade_time']), ]

class IndexAmountMaFIB(models.Model):
    """成交额移动平均线 (Amount MA) 指标存储模型 (斐波那契周期)"""
    index = models.ForeignKey('IndexInfo', on_delete=models.CASCADE, related_name="amount_ma_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")

    # 存储成交额的移动平均值，需要足够大的位数和小数位
    # 可以选择存储 SMA 或 EMA，这里以 SMA 为例命名
    amt_ma5 = models.DecimalField(max_digits=22, decimal_places=4, null=True, blank=True, verbose_name="成交额MA(5)")
    amt_ma8 = models.DecimalField(max_digits=22, decimal_places=4, null=True, blank=True, verbose_name="成交额MA(8)")
    amt_ma13 = models.DecimalField(max_digits=22, decimal_places=4, null=True, blank=True, verbose_name="成交额MA(13)")
    amt_ma21 = models.DecimalField(max_digits=22, decimal_places=4, null=True, blank=True, verbose_name="成交额MA(21)")
    amt_ma34 = models.DecimalField(max_digits=22, decimal_places=4, null=True, blank=True, verbose_name="成交额MA(34)")
    amt_ma55 = models.DecimalField(max_digits=22, decimal_places=4, null=True, blank=True, verbose_name="成交额MA(55)")
    amt_ma89 = models.DecimalField(max_digits=22, decimal_places=4, null=True, blank=True, verbose_name="成交额MA(89)")
    amt_ma144 = models.DecimalField(max_digits=22, decimal_places=4, null=True, blank=True, verbose_name="成交额MA(144)")
    amt_ma233 = models.DecimalField(max_digits=22, decimal_places=4, null=True, blank=True, verbose_name="成交额MA(233)")

    class Meta:
        verbose_name = "成交额MA指标(斐波那契)"
        db_table = 'index_amount_ma_fib'
        verbose_name_plural = verbose_name
        unique_together = ('index', 'trade_time', 'time_level')
        indexes = [
            models.Index(fields=['index', 'time_level', 'trade_time']),
        ]

class StockAmountMaFIB(models.Model):
    """成交额移动平均线 (Amount MA) 指标存储模型 (斐波那契周期)"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, related_name="amount_ma_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")

    # 存储成交额的移动平均值，需要足够大的位数和小数位
    # 可以选择存储 SMA 或 EMA，这里以 SMA 为例命名
    amt_ma5 = models.DecimalField(max_digits=22, decimal_places=4, null=True, blank=True, verbose_name="成交额MA(5)")
    amt_ma8 = models.DecimalField(max_digits=22, decimal_places=4, null=True, blank=True, verbose_name="成交额MA(8)")
    amt_ma13 = models.DecimalField(max_digits=22, decimal_places=4, null=True, blank=True, verbose_name="成交额MA(13)")
    amt_ma21 = models.DecimalField(max_digits=22, decimal_places=4, null=True, blank=True, verbose_name="成交额MA(21)")
    amt_ma34 = models.DecimalField(max_digits=22, decimal_places=4, null=True, blank=True, verbose_name="成交额MA(34)")
    amt_ma55 = models.DecimalField(max_digits=22, decimal_places=4, null=True, blank=True, verbose_name="成交额MA(55)")
    amt_ma89 = models.DecimalField(max_digits=22, decimal_places=4, null=True, blank=True, verbose_name="成交额MA(89)")
    amt_ma144 = models.DecimalField(max_digits=22, decimal_places=4, null=True, blank=True, verbose_name="成交额MA(144)")
    amt_ma233 = models.DecimalField(max_digits=22, decimal_places=4, null=True, blank=True, verbose_name="成交额MA(233)")

    class Meta:
        verbose_name = "成交额MA指标(斐波那契)"
        db_table = 'stock_amount_ma_fib'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time', 'time_level')
        indexes = [
            models.Index(fields=['stock', 'time_level', 'trade_time']),
        ]








