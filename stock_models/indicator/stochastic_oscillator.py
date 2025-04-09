from django.db import models
from django.utils.translation import gettext_lazy as _

# --- Stochastic Oscillator (斐波那契周期) ---
class IndexStochFIB(models.Model):
    """Stochastic Oscillator 指标存储模型 (斐波那契周期应用于%K长度)"""
    index = models.ForeignKey('IndexInfo', on_delete=models.CASCADE, related_name="stoch_fib", verbose_name="指数")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")

    # 周期 5
    stoch_k5 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="Stoch %K(5)")
    stoch_d5 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="Stoch %D(5)")
    # 周期 8
    stoch_k8 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="Stoch %K(8)")
    stoch_d8 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="Stoch %D(8)")
    # 周期 13
    stoch_k13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="Stoch %K(13)")
    stoch_d13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="Stoch %D(13)")
    # 周期 21
    stoch_k21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="Stoch %K(21)")
    stoch_d21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="Stoch %D(21)")
    # 周期 34
    stoch_k34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="Stoch %K(34)")
    stoch_d34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="Stoch %D(34)")
    # 周期 55
    stoch_k55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="Stoch %K(55)")
    stoch_d55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="Stoch %D(55)")
    # 周期 89
    stoch_k89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="Stoch %K(89)")
    stoch_d89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="Stoch %D(89)")
    # 周期 144
    stoch_k144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="Stoch %K(144)")
    stoch_d144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="Stoch %D(144)")
    # 周期 233
    stoch_k233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="Stoch %K(233)")
    stoch_d233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="Stoch %D(233)")

    class Meta:
        verbose_name = "随机指标(斐波那契)"
        db_table = 'index_stoch_fib'
        verbose_name_plural = verbose_name
        unique_together = ('index', 'trade_time', 'time_level')
        indexes = [ models.Index(fields=['index', 'time_level', 'trade_time']), ]


# --- Stochastic Oscillator (斐波那契周期) ---
class StockStochFIB(models.Model):
    """Stochastic Oscillator 指标存储模型 (斐波那契周期应用于%K长度)"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, related_name="stoch_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")

    # 周期 5
    stoch_k5 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="Stoch %K(5)")
    stoch_d5 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="Stoch %D(5)")
    # 周期 8
    stoch_k8 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="Stoch %K(8)")
    stoch_d8 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="Stoch %D(8)")
    # 周期 13
    stoch_k13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="Stoch %K(13)")
    stoch_d13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="Stoch %D(13)")
    # 周期 21
    stoch_k21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="Stoch %K(21)")
    stoch_d21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="Stoch %D(21)")
    # 周期 34
    stoch_k34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="Stoch %K(34)")
    stoch_d34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="Stoch %D(34)")
    # 周期 55
    stoch_k55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="Stoch %K(55)")
    stoch_d55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="Stoch %D(55)")
    # 周期 89
    stoch_k89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="Stoch %K(89)")
    stoch_d89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="Stoch %D(89)")
    # 周期 144
    stoch_k144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="Stoch %K(144)")
    stoch_d144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="Stoch %D(144)")
    # 周期 233
    stoch_k233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="Stoch %K(233)")
    stoch_d233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="Stoch %D(233)")

    class Meta:
        verbose_name = "随机指标(斐波那契)"
        db_table = 'stock_stoch_fib'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time', 'time_level')
        indexes = [ models.Index(fields=['stock', 'time_level', 'trade_time']), ]
