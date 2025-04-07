from django.db import models
from django.utils.translation import gettext_lazy as _
from bulk_update_or_create import BulkUpdateOrCreateQuerySet

class IndexKDJData(models.Model):
    """
    KDJ指标数据模型
    
    存储股票指数的KDJ技术指标数据
    """
    index = models.ForeignKey('IndexInfo', on_delete=models.CASCADE, related_name="kdj_data", verbose_name=_("股票指数"))
    time_level = models.CharField(max_length=10, verbose_name=_("时间级别"))  # 5, 15, 30, 60, Day, Week, Month, Year
    trade_time = models.DateTimeField(verbose_name=_("交易时间"))
    k_value = models.DecimalField(max_digits=10, decimal_places=4, verbose_name=_("K值"))
    d_value = models.DecimalField(max_digits=10, decimal_places=4, verbose_name=_("D值"))
    j_value = models.DecimalField(max_digits=10, decimal_places=4, verbose_name=_("J值"))
    
    class Meta:
        verbose_name = _("KDJ指标数据")
        verbose_name_plural = _("KDJ指标数据")
        db_table = "index_kdj_data"
        unique_together = [['index', 'time_level', 'trade_time']]
        indexes = [
            models.Index(fields=['index']),
            models.Index(fields=['time_level']),
            models.Index(fields=['trade_time']),
        ]
    
    def __str__(self):
        return f"{self.index.name} KDJ {self.time_level}({self.trade_time})"

    def __code__(self):
        return self.index.code

class StockKDJIndicator(models.Model):
    """
    KDJ指标数据模型
    """
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, blank=True, null=True, related_name="kdj_indicator", verbose_name=_("股票"))
    time_level = models.CharField(max_length=10, verbose_name='分时级别')
    trade_time = models.DateTimeField(verbose_name='交易时间')
    k_value = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='K值', null=True)
    d_value = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='D值', null=True)
    j_value = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='J值', null=True)
    
    class Meta:
        verbose_name = 'KDJ指标数据'
        verbose_name_plural = verbose_name
        db_table = 'stock_kdj_indicator'
        unique_together = ('stock', 'time_level', 'trade_time')
        ordering = ['stock', 'time_level', 'trade_time']
    
    def __str__(self):
        return f"{self.stock.stock_code}-{self.time_level}-{self.trade_time}"

class IndexKDJFIB(models.Model):
    """
    基于斐波那契周期的 KDJ 指标数据模型。
    假设平滑周期 M1 和 M2 固定为 3，仅改变 RSV 计算周期 N。
    """
    index = models.ForeignKey('IndexInfo', on_delete=models.CASCADE, blank=True, null=True, related_name="fib_kdj_indicators", verbose_name=_("股票"))
    time_level = models.CharField(max_length=10, verbose_name='分时级别', db_index=True)
    trade_time = models.DateTimeField(verbose_name='交易时间', db_index=True)

    # --- KDJ(5, 3, 3) ---
    k_5 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='K(5,3,3)', null=True, blank=True)
    d_5 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='D(5,3,3)', null=True, blank=True)
    j_5 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='J(5,3,3)', null=True, blank=True)

    # --- KDJ(8, 3, 3) ---
    k_8 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='K(8,3,3)', null=True, blank=True)
    d_8 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='D(8,3,3)', null=True, blank=True)
    j_8 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='J(8,3,3)', null=True, blank=True)

    # --- KDJ(13, 3, 3) ---
    k_13 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='K(13,3,3)', null=True, blank=True)
    d_13 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='D(13,3,3)', null=True, blank=True)
    j_13 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='J(13,3,3)', null=True, blank=True)

    # --- KDJ(21, 3, 3) ---
    k_21 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='K(21,3,3)', null=True, blank=True)
    d_21 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='D(21,3,3)', null=True, blank=True)
    j_21 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='J(21,3,3)', null=True, blank=True)

    # --- KDJ(34, 3, 3) ---
    k_34 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='K(34,3,3)', null=True, blank=True)
    d_34 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='D(34,3,3)', null=True, blank=True)
    j_34 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='J(34,3,3)', null=True, blank=True)

    # --- KDJ(55, 3, 3) ---
    k_55 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='K(55,3,3)', null=True, blank=True)
    d_55 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='D(55,3,3)', null=True, blank=True)
    j_55 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='J(55,3,3)', null=True, blank=True)

    # --- KDJ(89, 3, 3) ---
    k_89 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='K(89,3,3)', null=True, blank=True)
    d_89 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='D(89,3,3)', null=True, blank=True)
    j_89 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='J(89,3,3)', null=True, blank=True)

    # --- KDJ(144, 3, 3) ---
    k_144 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='K(144,3,3)', null=True, blank=True)
    d_144 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='D(144,3,3)', null=True, blank=True)
    j_144 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='J(144,3,3)', null=True, blank=True)

    # --- KDJ(233, 3, 3) ---
    k_233 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='K(233,3,3)', null=True, blank=True)
    d_233 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='D(233,3,3)', null=True, blank=True)
    j_233 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='J(233,3,3)', null=True, blank=True)


    class Meta:
        verbose_name = '斐波那契周期KDJ指标'
        verbose_name_plural = verbose_name
        db_table = 'index_kdj_fib' # 新的数据库表名
        unique_together = ('index', 'time_level', 'trade_time')
        ordering = ['index', 'time_level', 'trade_time']
        indexes = [
            models.Index(fields=['index', 'time_level', 'trade_time']),
            models.Index(fields=['trade_time']),
        ]

    def __str__(self):
        return f"{self.index.code}-{self.time_level}-{self.trade_time} (Fib KDJ)"

# --- 新增：斐波那契周期 KDJ 指标模型 ---
class StockKDJFIB(models.Model):
    """
    基于斐波那契周期的 KDJ 指标数据模型。
    假设平滑周期 M1 和 M2 固定为 3，仅改变 RSV 计算周期 N。
    """
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, blank=True, null=True, related_name="fib_kdj_indicators", verbose_name=_("股票"))
    time_level = models.CharField(max_length=10, verbose_name='分时级别', db_index=True)
    trade_time = models.DateTimeField(verbose_name='交易时间', db_index=True)

    # --- KDJ(5, 3, 3) ---
    k_5 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='K(5,3,3)', null=True, blank=True)
    d_5 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='D(5,3,3)', null=True, blank=True)
    j_5 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='J(5,3,3)', null=True, blank=True)

    # --- KDJ(8, 3, 3) ---
    k_8 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='K(8,3,3)', null=True, blank=True)
    d_8 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='D(8,3,3)', null=True, blank=True)
    j_8 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='J(8,3,3)', null=True, blank=True)

    # --- KDJ(13, 3, 3) ---
    k_13 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='K(13,3,3)', null=True, blank=True)
    d_13 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='D(13,3,3)', null=True, blank=True)
    j_13 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='J(13,3,3)', null=True, blank=True)

    # --- KDJ(21, 3, 3) ---
    k_21 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='K(21,3,3)', null=True, blank=True)
    d_21 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='D(21,3,3)', null=True, blank=True)
    j_21 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='J(21,3,3)', null=True, blank=True)

    # --- KDJ(34, 3, 3) ---
    k_34 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='K(34,3,3)', null=True, blank=True)
    d_34 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='D(34,3,3)', null=True, blank=True)
    j_34 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='J(34,3,3)', null=True, blank=True)

    # --- KDJ(55, 3, 3) ---
    k_55 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='K(55,3,3)', null=True, blank=True)
    d_55 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='D(55,3,3)', null=True, blank=True)
    j_55 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='J(55,3,3)', null=True, blank=True)

    # --- KDJ(89, 3, 3) ---
    k_89 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='K(89,3,3)', null=True, blank=True)
    d_89 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='D(89,3,3)', null=True, blank=True)
    j_89 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='J(89,3,3)', null=True, blank=True)

    # --- KDJ(144, 3, 3) ---
    k_144 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='K(144,3,3)', null=True, blank=True)
    d_144 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='D(144,3,3)', null=True, blank=True)
    j_144 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='J(144,3,3)', null=True, blank=True)

    # --- KDJ(233, 3, 3) ---
    k_233 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='K(233,3,3)', null=True, blank=True)
    d_233 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='D(233,3,3)', null=True, blank=True)
    j_233 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='J(233,3,3)', null=True, blank=True)


    class Meta:
        verbose_name = '斐波那契周期KDJ指标'
        verbose_name_plural = verbose_name
        db_table = 'stock_kdj_fib' # 新的数据库表名
        unique_together = ('stock', 'time_level', 'trade_time')
        ordering = ['stock', 'time_level', 'trade_time']
        indexes = [
            models.Index(fields=['stock', 'time_level', 'trade_time']),
            models.Index(fields=['trade_time']),
        ]

    def __str__(self):
        return f"{self.stock.code}-{self.time_level}-{self.trade_time} (Fib KDJ)"












