import datetime
from asgiref.sync import sync_to_async
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
        if check_date is None:
            check_date = timezone.now().date()
        is_open = cls.objects.filter(
            exchange=exchange,
            cal_date=check_date,
            is_open=True
        ).exists()
        return is_open
    @classmethod
    def get_latest_trade_date(cls, reference_date: datetime.date = None, exchange: str = 'SSE') -> datetime.date | None:
        """
        查询指定日期之前的最近一个交易日。
        :param reference_date: datetime.date, 查询的参考日期，如果为None，则默认为今天。
        :param exchange: str, 交易所代码，默认为'SSE'。
        :return: datetime.date, 最近的交易日；如果不存在则返回None。
        """
        if reference_date is None:
            reference_date = timezone.now().date()
        trade_day = cls.objects.filter(
            exchange=exchange,
            is_open=True,
            cal_date__lt=reference_date
        ).order_by('-cal_date').first()
        if trade_day:
            return trade_day.cal_date
        else:
            return None
    @classmethod
    def get_latest_n_trade_dates(cls, n: int, reference_date: datetime.date = None, exchange: str = 'SSE') -> list[datetime.date]:
        """
        查询指定日期（包含当天）之前的最近N个交易日。
        :param n: int, 需要获取的交易日数量。
        :param reference_date: datetime.date, 查询的参考日期，如果为None，则默认为今天。
        :param exchange: str, 交易所代码，默认为'SSE'。
        :return: list[datetime.date], 最近N个交易日的日期列表（按日期从近到远排序，即降序）。
        """
        if reference_date is None:
            reference_date = timezone.now().date()
        trade_dates_queryset = cls.objects.filter(
            exchange=exchange,
            is_open=True,
            cal_date__lte=reference_date
        ).order_by('-cal_date').values_list('cal_date', flat=True)[:n]
        trade_dates_list = list(trade_dates_queryset)
        trade_dates_list.sort(reverse=True)
        return trade_dates_list
    @classmethod
    def is_trade_day(cls, date_to_check: datetime.date | datetime.datetime) -> bool: # type: ignore
        """
        检查指定日期是否为交易日。
        该方法会查询数据库中是否存在该日期的记录，并且is_open字段为True。
        对于A股市场，上交所和深交所的交易日历通常是一致的，
        因此只要数据库中存在任一交易所的当天记录为交易日，即返回True。
        :param date_to_check: 需要检查的日期，可以是 date 类型或 datetime 类型。
        :return: 如果是交易日则返回 True，否则返回 False。
        """
        check_date = None
        # 修改代码行: 修正类型判断
        if isinstance(date_to_check, datetime.datetime):
            check_date = date_to_check.date()
        elif isinstance(date_to_check, datetime.date):
            check_date = date_to_check
        else:
            return False
        is_open = cls.objects.filter(cal_date=check_date, is_open=True).exists()
        return is_open
    @classmethod
    def get_next_trade_date(cls, reference_date: datetime.date = None, exchange: str = 'SSE') -> datetime.date | None:
        """
        查询指定日期之后的第一个交易日。
        :param reference_date: date, 查询的参考日期。如果为None，则默认为今天。
        :param exchange: str, 交易所代码，默认为'SSE'。
        :return: date, 下一个交易日；如果不存在（例如参考日期已是最后一个已知交易日）则返回None。
        """
        if reference_date is None:
            reference_date = timezone.now().date()
        trade_day = cls.objects.filter(
            exchange=exchange,
            is_open=True,
            cal_date__gt=reference_date
        ).order_by('cal_date').first()
        if trade_day:
            return trade_day.cal_date
        else:
            return None
    @classmethod
    async def get_next_trade_date_async(cls, reference_date: datetime.date = None, exchange: str = 'SSE') -> datetime.date | None:
        """
        【异步版】查询指定日期之后的第一个交易日。
        这是 get_next_trade_date 的异步版本，通过 sync_to_async 包装器，
        使其可以在异步上下文中被安全地调用。
        """
        get_next_date_func = sync_to_async(cls.get_next_trade_date, thread_sensitive=True)
        next_date = await get_next_date_func(reference_date=reference_date, exchange=exchange)
        return next_date
    @classmethod
    def get_trade_dates_between(cls, start_date: datetime.date, end_date: datetime.date, exchange: str = 'SSE') -> list[datetime.date]:
        """
        获取指定日期范围内的所有交易日列表。
        :param start_date: 开始日期
        :param end_date: 结束日期
        :param exchange: 交易所代码，默认为'SSE'
        :return: 交易日日期列表 (按升序排列)
        """
        trade_dates_qs = cls.objects.filter(
            exchange=exchange,
            is_open=True,
            cal_date__gte=start_date,
            cal_date__lte=end_date
        ).order_by('cal_date').values_list('cal_date', flat=True)
        return list(trade_dates_qs)
    @classmethod
    def get_trade_date_offset(cls, reference_date: datetime.date, offset: int, exchange: str = 'SSE') -> datetime.date | None:
        """
        获取参考日期偏移N个交易日的日期。
        :param reference_date: 参考日期
        :param offset: 偏移量。正数表示向未来偏移，负数表示向过去偏移。
        :param exchange: 交易所代码，默认为'SSE'
        :return: 偏移后的交易日；如果找不到则返回None。
        """
        if offset == 0:
            return reference_date if cls.is_trade_date(reference_date, exchange) else None
        if offset > 0: # 向未来查找
            qs = cls.objects.filter(
                exchange=exchange,
                is_open=True,
                cal_date__gt=reference_date
            ).order_by('cal_date').values_list('cal_date', flat=True)
        else: # 向过去查找
            qs = cls.objects.filter(
                exchange=exchange,
                is_open=True,
                cal_date__lt=reference_date
            ).order_by('-cal_date').values_list('cal_date', flat=True)
        abs_offset = abs(offset)
        # 使用 Django 的切片来获取第 N 个元素，这在数据库层面是高效的
        try:
            # Python 的索引是从0开始的，所以要找第N个，索引是 N-1
            return qs[abs_offset - 1]
        except IndexError:
            # 如果切片超出范围，说明没有足够的交易日
            return None
    @classmethod
    def get_trade_date_offset_list(cls, reference_date: datetime.date, start_offset: int, num_days: int, exchange: str = 'SSE') -> list[datetime.date]:
        """
        获取从参考日期偏移N天开始的、连续M个交易日的列表。
        :param reference_date: 参考日期
        :param start_offset: 开始偏移量。0表示从参考日当天开始，1表示从下一个交易日开始。
        :param num_days: 需要获取的交易日数量。
        :param exchange: 交易所代码，默认为'SSE'
        :return: 交易日日期列表
        """
        # 根据 start_offset 决定查询的起始点
        if start_offset >= 0:
            filter_kwargs = {'cal_date__gte': reference_date}
        else:
            # 理论上也可以支持负向偏移，但当前场景不需要，保持简单
            return []
        qs = cls.objects.filter(
            exchange=exchange,
            is_open=True,
            **filter_kwargs
        ).order_by('cal_date').values_list('cal_date', flat=True)
        # 使用切片获取所需的日期列表
        # [start_offset:start_offset + num_days]
        return list(qs[start_offset : start_offset + num_days])
    class Meta:
        db_table = 'trade_calendar'
        verbose_name = '交易日历'
        verbose_name_plural = '交易日历'
        unique_together = ('exchange', 'cal_date')
        ordering = ['-cal_date']
    def __str__(self):
        return f"{self.exchange} {self.cal_date} {'交易' if self.is_open else '休市'}"








