from django.db import models
from django.utils.translation import gettext_lazy as _

# --- Keltner Channels (斐波那契周期) ---
class IndexKcFIB(models.Model):
    """Keltner Channels 指标存储模型 (斐波那契周期应用于EMA长度)"""
    index = models.ForeignKey('IndexInfo', on_delete=models.CASCADE, related_name="kc_fib", verbose_name="指数")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")

    # 周期 5
    kc_lower5 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Lower(5)")
    kc_basis5 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Basis(5)")
    kc_upper5 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Upper(5)")
    # 周期 8
    kc_lower8 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Lower(8)")
    kc_basis8 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Basis(8)")
    kc_upper8 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Upper(8)")
    # 周期 13
    kc_lower13 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Lower(13)")
    kc_basis13 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Basis(13)")
    kc_upper13 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Upper(13)")
    # 周期 21
    kc_lower21 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Lower(21)")
    kc_basis21 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Basis(21)")
    kc_upper21 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Upper(21)")
    # 周期 34
    kc_lower34 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Lower(34)")
    kc_basis34 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Basis(34)")
    kc_upper34 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Upper(34)")
    # 周期 55
    kc_lower55 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Lower(55)")
    kc_basis55 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Basis(55)")
    kc_upper55 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Upper(55)")
    # 周期 89
    kc_lower89 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Lower(89)")
    kc_basis89 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Basis(89)")
    kc_upper89 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Upper(89)")
    # 周期 144
    kc_lower144 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Lower(144)")
    kc_basis144 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Basis(144)")
    kc_upper144 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Upper(144)")
    # 周期 233
    kc_lower233 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Lower(233)")
    kc_basis233 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Basis(233)")
    kc_upper233 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Upper(233)")

    class Meta:
        verbose_name = "肯特纳通道(斐波那契)"
        db_table = 'index_kc_fib'
        verbose_name_plural = verbose_name
        unique_together = ('index', 'trade_time', 'time_level')
        indexes = [ models.Index(fields=['index', 'time_level', 'trade_time']), ]

# --- Keltner Channels (斐波那契周期) ---
class StockKcFIB(models.Model):
    """Keltner Channels 指标存储模型 (斐波那契周期应用于EMA长度)"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, related_name="kc_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")

    # 周期 5
    kc_lower5 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Lower(5)")
    kc_basis5 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Basis(5)")
    kc_upper5 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Upper(5)")
    # 周期 8
    kc_lower8 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Lower(8)")
    kc_basis8 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Basis(8)")
    kc_upper8 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Upper(8)")
    # 周期 13
    kc_lower13 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Lower(13)")
    kc_basis13 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Basis(13)")
    kc_upper13 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Upper(13)")
    # 周期 21
    kc_lower21 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Lower(21)")
    kc_basis21 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Basis(21)")
    kc_upper21 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Upper(21)")
    # 周期 34
    kc_lower34 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Lower(34)")
    kc_basis34 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Basis(34)")
    kc_upper34 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Upper(34)")
    # 周期 55
    kc_lower55 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Lower(55)")
    kc_basis55 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Basis(55)")
    kc_upper55 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Upper(55)")
    # 周期 89
    kc_lower89 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Lower(89)")
    kc_basis89 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Basis(89)")
    kc_upper89 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Upper(89)")
    # 周期 144
    kc_lower144 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Lower(144)")
    kc_basis144 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Basis(144)")
    kc_upper144 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Upper(144)")
    # 周期 233
    kc_lower233 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Lower(233)")
    kc_basis233 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Basis(233)")
    kc_upper233 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="KC Upper(233)")

    class Meta:
        verbose_name = "肯特纳通道(斐波那契)"
        db_table = 'stock_kc_fib'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time', 'time_level')
        indexes = [ models.Index(fields=['stock', 'time_level', 'trade_time']), ]