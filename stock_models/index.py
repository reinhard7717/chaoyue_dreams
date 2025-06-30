import datetime
from django.utils import timezone
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
        
    def __str__(self):
        return f"{self.index_code} - {self.name}"

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

    @classmethod
    def is_trade_date(cls, check_date: datetime.date = None, exchange: str = 'SSE') -> bool:
        """
        检查指定日期是否为交易日。
        :param check_date: datetime.date, 需要检查的日期。如果为None，则默认为今天。
        :param exchange: str, 交易所代码，默认为'SSE'。
        :return: bool, 如果是交易日则返回True，否则返回False。
        """
        # 如果未提供检查日期，则使用当前服务器日期
        if check_date is None:
            check_date = timezone.now().date()
        
        # 调试信息：打印输入的参数
        print(f"调试: is_trade_date - 检查日期: {check_date}, 交易所: {exchange}")

        # 使用 .exists() 高效地检查记录是否存在，这比获取整个对象更快
        # 筛选条件：
        # 1. 交易所匹配
        # 2. 日期匹配
        # 3. is_open 字段为 True
        is_open = cls.objects.filter(
            exchange=exchange,
            cal_date=check_date,
            is_open=True
        ).exists()

        print(f"调试: {check_date} 是否为交易日: {is_open}")
        return is_open

    @classmethod
    def get_latest_trade_date(cls, reference_date: datetime.date = None, exchange: str = 'SSE') -> datetime.date | None:
        """
        查询指定日期之前的最近一个交易日。
        :param reference_date: datetime.date, 查询的参考日期，如果为None，则默认为今天。
        :param exchange: str, 交易所代码，默认为'SSE'。
        :return: datetime.date, 最近的交易日；如果不存在则返回None。
        """
        # 如果未提供参考日期，则使用当前服务器日期
        if reference_date is None:
            reference_date = timezone.now().date()
        
        # 调试信息：打印输入的参数
        print(f"调试: get_latest_trade_date - 参考日期: {reference_date}, 交易所: {exchange}")

        # 查询数据库
        # 筛选条件：
        # 1. 交易所匹配
        # 2. 是交易日 (is_open=True)
        # 3. 日期在参考日期之前 (cal_date < reference_date)
        # 按照日期降序排列，获取第一个，即为最近的交易日
        trade_day = cls.objects.filter(
            exchange=exchange,
            is_open=True,
            cal_date__lt=reference_date
        ).order_by('-cal_date').first()

        # 如果找到了交易日，则返回其日历日期，否则返回None
        if trade_day:
            print(f"调试: 找到最近交易日: {trade_day.cal_date}")
            return trade_day.cal_date
        else:
            print("调试: 未找到符合条件的交易日")
            return None

    @classmethod
    def get_latest_n_trade_dates(cls, n: int, reference_date: datetime.date = None, exchange: str = 'SSE') -> list[datetime.date]:
        """
        查询指定日期（包含当天）之前的最近N个交易日。
        :param n: int, 需要获取的交易日数量。
        :param reference_date: datetime.date, 查询的参考日期，如果为None，则默认为今天。
        :param exchange: str, 交易所代码，默认为'SSE'。
        :return: list[datetime.date], 最近N个交易日的日期列表（按日期从近到远排序）。
        """
        # 如果未提供参考日期，则使用当前服务器日期
        if reference_date is None:
            reference_date = timezone.now().date()
        
        # 调试信息：打印输入的参数
        print(f"调试: get_latest_n_trade_dates - 获取数量: {n}, 参考日期: {reference_date}, 交易所: {exchange}")

        # 查询数据库
        # 筛选条件：
        # 1. 交易所匹配
        # 2. 是交易日 (is_open=True)
        # 3. 日期小于或等于参考日期 (cal_date <= reference_date)
        # 按照日期降序排列
        # 使用 .values_list('cal_date', flat=True) 可以更高效地只获取日期字段，而不是整个对象
        # 使用切片 [:n] 获取前N个结果
        trade_dates_queryset = cls.objects.filter(
            exchange=exchange,
            is_open=True,
            cal_date__lte=reference_date
        ).order_by('-cal_date').values_list('cal_date', flat=True)[:n]

        # 将查询结果QuerySet转换为列表
        trade_dates_list = list(trade_dates_queryset)
        print(f"调试: 找到 {len(trade_dates_list)} 个交易日: {trade_dates_list}")
        return trade_dates_list

    class Meta:
        db_table = 'trade_calendar'
        verbose_name = '交易日历'
        verbose_name_plural = '交易日历'
        unique_together = ('exchange', 'cal_date')
        ordering = ['-cal_date']

    def __str__(self):
        return f"{self.exchange} {self.cal_date} {'交易' if self.is_open else '休市'}"








