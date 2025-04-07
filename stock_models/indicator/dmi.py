from django.db import models
from django.utils.translation import gettext_lazy as _

class IndexDmiFIB(models.Model):
    """DMI 指标存储模型 (斐波那契周期)"""
    index = models.ForeignKey('IndexInfo', on_delete=models.CASCADE, related_name="dmi_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")
    plus_di13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="+DI(13)") # 常用14，这里用13
    minus_di13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="-DI(13)")
    adx13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADX(13)")
    adxr13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADXR(13)")
    plus_di21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="+DI(21)")
    minus_di21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="-DI(21)")
    adx21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADX(21)")
    adxr21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADXR(21)")
    plus_di34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="+DI(34)")
    minus_di34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="-DI(34)")
    adx34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADX(34)")
    adxr34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADXR(34)")
    plus_di55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="+DI(55)")
    minus_di55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="-DI(55)")
    adx55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADX(55)")
    adxr55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADXR(55)")
    plus_di89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="+DI(89)")
    minus_di89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="-DI(89)")
    adx89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADX(89)")
    adxr89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADXR(89)")
    plus_di144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="+DI(144)")
    minus_di144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="-DI(144)")
    adx144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADX(144)")
    adxr144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADXR(144)")
    plus_di233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="+DI(233)")
    minus_di233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="-DI(233)")
    adx233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADX(233)")
    adxr233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADXR(233)")

    class Meta:
        verbose_name = "DMI指标(斐波那契)"
        db_table = 'index_dmi_fib'
        verbose_name_plural = verbose_name
        unique_together = ('index', 'trade_time', 'time_level')
        indexes = [ models.Index(fields=['index', 'time_level', 'trade_time']), ]


class StockDmiFIB(models.Model):
    """DMI 指标存储模型 (斐波那契周期)"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, related_name="dmi_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")
    plus_di13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="+DI(13)") # 常用14，这里用13
    minus_di13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="-DI(13)")
    adx13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADX(13)")
    adxr13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADXR(13)")
    plus_di21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="+DI(21)")
    minus_di21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="-DI(21)")
    adx21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADX(21)")
    adxr21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADXR(21)")
    plus_di34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="+DI(34)")
    minus_di34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="-DI(34)")
    adx34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADX(34)")
    adxr34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADXR(34)")
    plus_di55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="+DI(55)")
    minus_di55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="-DI(55)")
    adx55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADX(55)")
    adxr55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADXR(55)")
    plus_di89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="+DI(89)")
    minus_di89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="-DI(89)")
    adx89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADX(89)")
    adxr89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADXR(89)")
    plus_di144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="+DI(144)")
    minus_di144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="-DI(144)")
    adx144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADX(144)")
    adxr144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADXR(144)")
    plus_di233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="+DI(233)")
    minus_di233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="-DI(233)")
    adx233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADX(233)")
    adxr233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADXR(233)")

    class Meta:
        verbose_name = "DMI指标(斐波那契)"
        db_table = 'stock_dmi_fib'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time', 'time_level')
        indexes = [ models.Index(fields=['stock', 'time_level', 'trade_time']), ]
