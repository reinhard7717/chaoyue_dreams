from django.db import models
from django.utils.translation import gettext_lazy as _
# 市场交易统计(MarketDailyInfo)
class MarketDailyInfo(models.Model):
    trade_date = models.DateField(verbose_name=_("交易日期"), null=True, blank=True)
    ts_code = models.CharField(max_length=20, verbose_name="市场代码")
    ts_name = models.CharField(max_length=50, verbose_name="市场名称")
    com_count = models.IntegerField(verbose_name="挂牌数")
    total_share = models.FloatField(verbose_name="总股本(亿股)")
    float_share = models.FloatField(verbose_name="流通股本(亿股)")
    total_mv = models.FloatField(verbose_name="总市值(亿元)")
    float_mv = models.FloatField(verbose_name="流通市值(亿元)")
    amount = models.FloatField(verbose_name="交易金额(亿元)")
    vol = models.FloatField(verbose_name="成交量(亿股)")
    trans_count = models.IntegerField(verbose_name="成交笔数(万笔)")
    pe = models.FloatField(verbose_name="平均市盈率")
    tr = models.FloatField(verbose_name="换手率(%)", null=True, blank=True)
    exchange = models.CharField(max_length=10, verbose_name="交易所")
    class Meta:
        db_table = "market_daily_info"
        verbose_name = "市场交易统计"
        verbose_name_plural = verbose_name
        unique_together = ('trade_date', 'ts_code')

# 游资名录
class HmList(models.Model):
    name = models.CharField(max_length=50, unique=True, verbose_name="游资名称")
    desc = models.TextField(verbose_name="说明", null=True, blank=True)
    orgs = models.TextField(verbose_name="关联机构", null=True, blank=True)
    class Meta:
        db_table = "hm_list"
        verbose_name = "游资名录"
        verbose_name_plural = verbose_name

# 游资每日明细
class HmDetail(models.Model):
    trade_date = models.DateField(verbose_name=_("交易日期"), null=True, blank=True)
    stock = models.ForeignKey(
        'StockInfo',
        db_column='ts_code',
        to_field='stock_code',
        on_delete=models.CASCADE,
        related_name='hm_detail',
        verbose_name="股票", null=True, blank=True
    )
    ts_name = models.CharField(max_length=50, verbose_name="股票名称")
    buy_amount = models.FloatField(verbose_name="买入金额(元)")
    sell_amount = models.FloatField(verbose_name="卖出金额(元)")
    net_amount = models.FloatField(verbose_name="净买卖(元)")
    hm_name = models.CharField(max_length=50, verbose_name="游资名称")
    hm_orgs = models.TextField(verbose_name="关联机构", null=True, blank=True)
    tag = models.CharField(max_length=100, verbose_name="标签", null=True, blank=True)
    class Meta:
        db_table = "hm_detail"
        verbose_name = "游资每日明细"
        verbose_name_plural = verbose_name
        unique_together = ('trade_date', 'stock', 'hm_name')


# 涨跌停榜单 - 同花顺
class LimitListThs(models.Model):
    trade_date = models.DateField(verbose_name="交易日期")
    stock = models.ForeignKey(
        'StockInfo',
        db_column='ts_code',
        to_field='stock_code',
        on_delete=models.CASCADE,
        related_name='limit_list_ths',
        verbose_name="股票"
    )
    name = models.CharField(max_length=50, verbose_name="股票名称")
    price = models.FloatField(verbose_name="收盘价")
    pct_chg = models.FloatField(verbose_name="涨跌幅%")
    open_num = models.IntegerField(verbose_name="打开次数")
    lu_desc = models.CharField(max_length=200, verbose_name="涨停原因")
    limit_type = models.CharField(max_length=20, verbose_name="板单类别")
    tag = models.CharField(max_length=50, verbose_name="涨停标签")
    status = models.CharField(max_length=20, verbose_name="涨停状态")
    first_lu_time = models.CharField(max_length=20, verbose_name="首次涨停时间", null=True, blank=True)
    last_lu_time = models.CharField(max_length=20, verbose_name="最后涨停时间", null=True, blank=True)
    first_ld_time = models.CharField(max_length=20, verbose_name="首次跌停时间", null=True, blank=True)
    last_ld_time = models.CharField(max_length=20, verbose_name="最后跌停时间", null=True, blank=True)
    limit_order = models.FloatField(verbose_name="封单量", null=True, blank=True)
    limit_amount = models.FloatField(verbose_name="封单额", null=True, blank=True)
    turnover_rate = models.FloatField(verbose_name="换手率", null=True, blank=True)
    free_float = models.FloatField(verbose_name="实际流通", null=True, blank=True)
    lu_limit_order = models.FloatField(verbose_name="最大封单", null=True, blank=True)
    limit_up_suc_rate = models.FloatField(verbose_name="近一年涨停封板率", null=True, blank=True)
    turnover = models.FloatField(verbose_name="成交额", null=True, blank=True)
    rise_rate = models.FloatField(verbose_name="涨速", null=True, blank=True)
    sum_float = models.FloatField(verbose_name="总市值", null=True, blank=True)
    market_type = models.CharField(max_length=10, verbose_name="股票类型")
    class Meta:
        db_table = "limit_list_ths"
        verbose_name = "同花顺涨跌停榜单"
        verbose_name_plural = verbose_name
        unique_together = ('trade_date', 'stock', 'limit_type')

# 涨跌停列表
class LimitListD(models.Model):
    trade_date = models.DateField(verbose_name="交易日期")
    stock = models.ForeignKey(
        'StockInfo',
        db_column='ts_code',
        to_field='stock_code',
        on_delete=models.CASCADE,
        related_name='limit_list_d',
        verbose_name="股票"
    )
    industry = models.CharField(max_length=50, verbose_name="所属行业")
    name = models.CharField(max_length=50, verbose_name="股票名称")
    close = models.FloatField(verbose_name="收盘价")
    pct_chg = models.FloatField(verbose_name="涨跌幅")
    amount = models.FloatField(verbose_name="成交额")
    limit_amount = models.FloatField(verbose_name="板上成交金额", null=True, blank=True)
    float_mv = models.FloatField(verbose_name="流通市值")
    total_mv = models.FloatField(verbose_name="总市值")
    turnover_ratio = models.FloatField(verbose_name="换手率")
    fd_amount = models.FloatField(verbose_name="封单金额", null=True, blank=True)
    first_time = models.CharField(max_length=20, verbose_name="首次封板时间", null=True, blank=True)
    last_time = models.CharField(max_length=20, verbose_name="最后封板时间", null=True, blank=True)
    open_times = models.IntegerField(verbose_name="炸板次数")
    up_stat = models.CharField(max_length=20, verbose_name="涨停统计")
    limit_times = models.IntegerField(verbose_name="连板数")
    limit = models.CharField(max_length=2, verbose_name="涨跌停类型")
    class Meta:
        db_table = "limit_list_d"
        verbose_name = "A股涨跌停列表"
        verbose_name_plural = verbose_name
        unique_together = ('trade_date', 'stock', 'limit')

# 连板天梯
class LimitStep(models.Model):
    trade_date = models.DateField(verbose_name="交易日期")
    stock = models.ForeignKey(
        'StockInfo',
        db_column='ts_code',
        to_field='stock_code',
        on_delete=models.CASCADE,
        related_name='limit_step',
        verbose_name="股票"
    )
    name = models.CharField(max_length=50, verbose_name="名称")
    nums = models.IntegerField(verbose_name="连板次数")
    class Meta:
        db_table = "limit_step"
        verbose_name = "连板天梯"
        verbose_name_plural = verbose_name
        unique_together = ('trade_date', 'stock')

# 最强板块统计 - 同花顺
class LimitCptList(models.Model):
    trade_date = models.DateField(verbose_name="交易日期")
    ths_index = models.ForeignKey(
        'ThsIndex',
        db_column='ts_code',
        to_field='ts_code',
        on_delete=models.CASCADE,
        related_name='limit_cpt_list',
        verbose_name="板块"
    )
    name = models.CharField(max_length=50, verbose_name="板块名称")
    days = models.IntegerField(verbose_name="上榜天数")
    up_stat = models.CharField(max_length=20, verbose_name="连板高度")
    cons_nums = models.IntegerField(verbose_name="连板家数")
    up_nums = models.CharField(max_length=20, verbose_name="涨停家数")
    pct_chg = models.FloatField(verbose_name="涨跌幅")
    rank = models.CharField(max_length=10, verbose_name="板块热点排名")
    class Meta:
        db_table = "limit_cpt_list"
        verbose_name = "最强板块统计"
        verbose_name_plural = verbose_name
        unique_together = ('trade_date', 'ths_index')















