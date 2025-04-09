from django.db import models
from django.utils.translation import gettext_lazy as _

# --- Pivot Points ---
class IndexPivotPoints(models.Model):
    """Pivot Points 指标存储模型"""
    index = models.ForeignKey('IndexInfo', on_delete=models.CASCADE, related_name="pivot_points", verbose_name="指数")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳") # 代表该枢轴点适用的时间点
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期") # 代表计算枢轴点所基于的K线周期

    # 存储常见的枢轴点和支撑/阻力位
    # 注意：pandas-ta 可能返回带后缀的列名 (如 S1_traditional)，需要在 DAO 层映射
    pp = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="Pivot Point (PP)")
    s1 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="Support 1 (S1)")
    r1 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="Resistance 1 (R1)")
    s2 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="Support 2 (S2)")
    r2 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="Resistance 2 (R2)")
    s3 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="Support 3 (S3)")
    r3 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="Resistance 3 (R3)")
    # 可以根据需要添加 S4, R4 或其他类型的枢轴点 (如 Fibonacci Pivot)
    s4 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="Support 4 (S4)")
    r4 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="Resistance 4 (R4)")

    class Meta:
        verbose_name = "枢轴点(Pivot Points)"
        db_table = 'index_pivot_points'
        verbose_name_plural = verbose_name
        unique_together = ('index', 'trade_time', 'time_level')
        indexes = [ models.Index(fields=['index', 'time_level', 'trade_time']), ]


# --- Pivot Points ---
class StockPivotPoints(models.Model):
    """Pivot Points 指标存储模型"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, related_name="pivot_points", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳") # 代表该枢轴点适用的时间点
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期") # 代表计算枢轴点所基于的K线周期

    # 存储常见的枢轴点和支撑/阻力位
    # 注意：pandas-ta 可能返回带后缀的列名 (如 S1_traditional)，需要在 DAO 层映射
    pp = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="Pivot Point (PP)")
    s1 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="Support 1 (S1)")
    r1 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="Resistance 1 (R1)")
    s2 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="Support 2 (S2)")
    r2 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="Resistance 2 (R2)")
    s3 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="Support 3 (S3)")
    r3 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="Resistance 3 (R3)")
    # 可以根据需要添加 S4, R4 或其他类型的枢轴点 (如 Fibonacci Pivot)
    s4 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="Support 4 (S4)")
    r4 = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True, verbose_name="Resistance 4 (R4)")

    class Meta:
        verbose_name = "枢轴点(Pivot Points)"
        db_table = 'stock_pivot_points'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time', 'time_level')
        indexes = [ models.Index(fields=['stock', 'time_level', 'trade_time']), ]
