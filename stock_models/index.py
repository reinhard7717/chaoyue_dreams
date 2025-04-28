from django.db import models
from django.utils.translation import gettext_lazy as _

# 指数基础信息
class IndexInfo(models.Model):
    index_code = models.CharField(max_length=20, unique=True, verbose_name="指数代码")
    name = models.CharField(max_length=50, verbose_name="简称")
    fullname = models.CharField(max_length=100, verbose_name="指数全称", blank=True, null=True)
    market = models.CharField(max_length=20, verbose_name="市场")
    publisher = models.CharField(max_length=50, verbose_name="发布方", blank=True, null=True)
    index_type = models.CharField(max_length=50, verbose_name="指数风格", blank=True, null=True)
    category = models.CharField(max_length=50, verbose_name="指数类别", blank=True, null=True)
    base_date = models.DateField(verbose_name=_("基期"), null=True, blank=True)
    base_point = models.FloatField(verbose_name="基点", blank=True, null=True)
    list_date = models.CharField(max_length=8, verbose_name="发布日期", blank=True, null=True)
    weight_rule = models.CharField(max_length=100, verbose_name="加权方式", blank=True, null=True)
    desc = models.TextField(verbose_name="描述", blank=True, null=True)
    exp_date = models.CharField(max_length=8, verbose_name="终止日期", blank=True, null=True)

    class Meta:
        db_table = "index_info"
        verbose_name = "指数基础信息"
        verbose_name_plural = verbose_name

# 指数成分和权重
class IndexWeight(models.Model):
    index = models.ForeignKey(IndexInfo, to_field='index_code', db_column='index_code', related_name="index_weight", on_delete=models.CASCADE, verbose_name="指数")
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',  # 指定外键对应StockInfo的stock_code字段
        db_column='con_code', # 数据库字段名
        on_delete=models.CASCADE, blank=True, null=True,
        related_name="index_weight", verbose_name=_("股票")
    )
    trade_date = models.DateField(verbose_name=_("交易日期"), null=True, blank=True)
    weight = models.FloatField(verbose_name="权重")

    class Meta:
        db_table = "index_weight"
        verbose_name = "指数成分权重"
        verbose_name_plural = verbose_name
        unique_together = ('index', 'stock', 'trade_date')

# 大盘指数每日指标
class IndexDailyBasic(models.Model):
    index = models.ForeignKey(IndexInfo, to_field='index_code', db_column='index_code', related_name="index_dailybasic", on_delete=models.CASCADE, verbose_name="指数")
    trade_time = models.DateField(verbose_name=_("交易日期"), null=True, blank=True)
    total_mv = models.FloatField(verbose_name="总市值(元)", null=True, blank=True)
    float_mv = models.FloatField(verbose_name="流通市值(元)", null=True, blank=True)
    total_share = models.FloatField(verbose_name="总股本(股)", null=True, blank=True)
    float_share = models.FloatField(verbose_name="流通股本(股)", null=True, blank=True)
    free_share = models.FloatField(verbose_name="自由流通股本(股)", null=True, blank=True)
    turnover_rate = models.FloatField(verbose_name="换手率", null=True, blank=True)
    turnover_rate_f = models.FloatField(verbose_name="换手率(自由流通)", null=True, blank=True)
    pe = models.FloatField(verbose_name="市盈率", null=True, blank=True)
    pe_ttm = models.FloatField(verbose_name="市盈率TTM", null=True, blank=True)
    pb = models.FloatField(verbose_name="市净率", null=True, blank=True)

    class Meta:
        db_table = "index_dailybasic"
        verbose_name = "大盘指数每日指标"
        verbose_name_plural = verbose_name
        unique_together = ('index', 'trade_time')

# 交易日历
class TradeCalendar(models.Model):
    EXCHANGE_CHOICES = [
        ('SSE', '上交所'),
        ('SZSE', '深交所'),
        ('CFFEX', '中金所'),
        ('SHFE', '上期所'),
        ('CZCE', '郑商所'),
        ('DCE', '大商所'),
        ('INE', '上能源'),
    ]

    exchange = models.CharField(
        max_length=10,
        choices=EXCHANGE_CHOICES,
        default='SSE',
        verbose_name='交易所'
    )
    cal_date = models.DateField(verbose_name='日历日期', db_index=True)
    is_open = models.BooleanField(verbose_name='是否交易')  # 1为交易，0为休市
    pretrade_date = models.DateField(
        verbose_name='上一个交易日',
        null=True,
        blank=True
    )

    class Meta:
        db_table = 'trade_calendar'
        verbose_name = '交易日历'
        verbose_name_plural = '交易日历'
        unique_together = ('exchange', 'cal_date')
        ordering = ['-cal_date']

    def __str__(self):
        return f"{self.exchange} {self.cal_date} {'交易' if self.is_open else '休市'}"








