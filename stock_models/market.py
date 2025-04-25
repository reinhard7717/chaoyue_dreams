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
    ts_code = models.CharField(max_length=20, verbose_name="股票代码")
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
        unique_together = ('trade_date', 'ts_code', 'hm_name')
















