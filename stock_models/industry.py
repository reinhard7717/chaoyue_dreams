from django.db import models
from django.utils.translation import gettext_lazy as _

# 申万行业分类
class SwIndustry(models.Model):
    index_code = models.CharField(max_length=16, db_index=True, verbose_name="指数代码")  # 801xxx.SI
    index = models.OneToOneField(
        'IndexInfo',
        to_field='index_code',
        db_constraint=False,  # 避免外部数据不一致导致迁移失败
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='sw_industry',
        verbose_name="关联指数基础信息"
    )
    industry_name = models.CharField(max_length=64, verbose_name="行业名称")
    parent_code = models.CharField(max_length=16, db_index=True, verbose_name="父级代码")
    level = models.CharField(max_length=4, db_index=True, verbose_name="行业分级")  # L1/L2/L3
    industry_code = models.CharField(max_length=16, db_index=True, verbose_name="行业代码")
    is_pub = models.CharField(max_length=4, verbose_name="是否发布指数")
    src = models.CharField(max_length=16, verbose_name="行业分类来源", default="SW2021")

    class Meta:
        db_table = "sw_industry"
        verbose_name = "申万行业分类"
        verbose_name_plural = "申万行业分类"
        unique_together = ("index_code", "src")

    def __str__(self):
        return f"{self.industry_name}({self.index_code})"

# 申万行业成分
class SwIndustryMember(models.Model):
    l3_industry = models.ForeignKey(
        SwIndustry,
        on_delete=models.CASCADE,
        related_name='members',
        verbose_name="三级行业",
        limit_choices_to={'level': 'L3'}
    )
    l1_code = models.CharField(max_length=16, db_index=True, verbose_name="一级行业代码")
    l1_name = models.CharField(max_length=64, verbose_name="一级行业名称")
    l2_code = models.CharField(max_length=16, db_index=True, verbose_name="二级行业代码")
    l2_name = models.CharField(max_length=64, verbose_name="二级行业名称")
    l3_name = models.CharField(max_length=64, verbose_name="三级行业名称")
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',  # 指定外键对应StockInfo的stock_code字段
        db_column='stock_code', # 数据库字段名
        on_delete=models.CASCADE, blank=True, null=True,
        related_name="sw_index_member", verbose_name=_("股票")
    )
    name = models.CharField(max_length=64, verbose_name="成分股票名称")
    in_date = models.CharField(max_length=8, verbose_name="纳入日期")
    out_date = models.CharField(max_length=8, null=True, blank=True, verbose_name="剔除日期")
    is_new = models.CharField(max_length=2, verbose_name="是否最新", default="Y")

    class Meta:
        db_table = "sw_industry_member"
        verbose_name = "申万行业成分"
        verbose_name_plural = "申万行业成分"
        unique_together = ("l3_industry", "stock", "in_date")

    def __str__(self):
        return f"{self.l3_industry.industry_name}({self.l3_industry.index_code}) - {self.name}({self.ts_code})"

# 申万行业日线行情
class SwIndustryDaily(models.Model):
    industry = models.ForeignKey(
        SwIndustry,
        on_delete=models.CASCADE,
        related_name='daily_quotes',
        verbose_name="所属行业",
    )
    index = models.ForeignKey(
        'IndexInfo',
        to_field='index_code',
        db_constraint=False,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='sw_daily',
        verbose_name="关联指数基础信息"
    )
    ts_code = models.CharField(max_length=16, db_index=True, verbose_name="指数代码")
    trade_date = models.CharField(max_length=8, db_index=True, verbose_name="交易日期")
    name = models.CharField(max_length=64, verbose_name="指数名称")
    open = models.FloatField(verbose_name="开盘点位")
    low = models.FloatField(verbose_name="最低点位")
    high = models.FloatField(verbose_name="最高点位")
    close = models.FloatField(verbose_name="收盘点位")
    change = models.FloatField(verbose_name="涨跌点位")
    pct_change = models.FloatField(verbose_name="涨跌幅")
    vol = models.FloatField(verbose_name="成交量（万股）")
    amount = models.FloatField(verbose_name="成交额（万元）")
    pe = models.FloatField(verbose_name="市盈率")
    pb = models.FloatField(verbose_name="市净率")
    float_mv = models.FloatField(verbose_name="流通市值（万元）")
    total_mv = models.FloatField(verbose_name="总市值（万元）")

    class Meta:
        db_table = "sw_industry_daily"
        verbose_name = "申万行业日线行情"
        verbose_name_plural = "申万行业日线行情"
        unique_together = ("ts_code", "trade_date")

    def __str__(self):
        return f"{self.trade_date} - {self.name}({self.ts_code})"

# 中信行业成分
class CiIndexMember(models.Model):
    l1_code = models.CharField(max_length=20, verbose_name="一级行业代码")
    l1_name = models.CharField(max_length=50, verbose_name="一级行业名称")
    l2_code = models.CharField(max_length=20, verbose_name="二级行业代码")
    l2_name = models.CharField(max_length=50, verbose_name="二级行业名称")
    l3_code = models.CharField(max_length=20, verbose_name="三级行业代码")
    l3_name = models.CharField(max_length=50, verbose_name="三级行业名称")
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',  # 指定外键对应StockInfo的stock_code字段
        db_column='stock_code', # 数据库字段名
        on_delete=models.CASCADE, blank=True, null=True,
        related_name="ci_index_member", verbose_name=_("股票")
    )
    name = models.CharField(max_length=50, verbose_name="成分股票名称")
    in_date = models.DateField(verbose_name=_("纳入日期"), null=True, blank=True)
    out_date = models.DateField(verbose_name=_("剔除日期"), null=True, blank=True)
    is_new = models.CharField(max_length=2, verbose_name="是否最新")

    class Meta:
        db_table = "ci_index_member"
        verbose_name = "中信行业成分"
        verbose_name_plural = verbose_name
        unique_together = ('l3_code', 'stock', 'in_date')

# 中信行业指数日线行情
class CiDaily(models.Model):
    ts_code = models.CharField(max_length=20, verbose_name="行业代码")
    trade_date = models.DateField(verbose_name=_("交易日期"), null=True, blank=True)
    open = models.FloatField(verbose_name="开盘点位")
    low = models.FloatField(verbose_name="最低点位")
    high = models.FloatField(verbose_name="最高点位")
    close = models.FloatField(verbose_name="收盘点位")
    pre_close = models.FloatField(verbose_name="昨日收盘点位")
    change = models.FloatField(verbose_name="涨跌点位")
    pct_change = models.FloatField(verbose_name="涨跌幅")
    vol = models.FloatField(verbose_name="成交量(万股)")
    amount = models.FloatField(verbose_name="成交额(万元)")

    class Meta:
        db_table = "ci_daily"
        verbose_name = "中信行业指数日线行情"
        verbose_name_plural = verbose_name
        unique_together = ('ts_code', 'trade_date')

# 开盘啦题材库
class KplConcept(models.Model):
    """
    开盘啦题材库
    """
    id = models.BigAutoField(primary_key=True)
    trade_date = models.CharField(max_length=8, db_index=True, verbose_name="交易日期")  # YYYYMMDD
    ts_code = models.CharField(max_length=16, db_index=True, verbose_name="题材代码")    # xxxxxx.KP
    name = models.CharField(max_length=64, verbose_name="题材名称")
    z_t_num = models.IntegerField(null=True, blank=True, verbose_name="涨停数量")
    up_num = models.IntegerField(null=True, blank=True, verbose_name="排名上升位数")

    class Meta:
        db_table = "kpl_concept"
        verbose_name = "开盘啦题材"
        verbose_name_plural = "开盘啦题材"
        unique_together = ("trade_date", "ts_code")

    def __str__(self):
        return f"{self.trade_date} - {self.name}({self.ts_code})"

# 开盘啦题材成分股
class KplConceptConstituent(models.Model):
    """
    开盘啦题材成分股
    """
    concept = models.ForeignKey(
        KplConcept,
        on_delete=models.CASCADE,
        related_name='constituents',
        verbose_name="所属题材"
    )
    ts_code = models.CharField(max_length=16, db_index=True, verbose_name="题材代码")  # xxxxxx.KP
    name = models.CharField(max_length=64, verbose_name="题材名称")
    con_name = models.CharField(max_length=64, verbose_name="成分股名称")
    con_code = models.CharField(max_length=16, db_index=True, verbose_name="成分股代码")  # xxxxxx.SH
    trade_date = models.CharField(max_length=8, db_index=True, verbose_name="交易日期")  # YYYYMMDD
    desc = models.TextField(null=True, blank=True, verbose_name="描述")
    hot_num = models.IntegerField(null=True, blank=True, verbose_name="人气值")

    class Meta:
        db_table = "kpl_concept_constituent"
        verbose_name = "开盘啦题材成分股"
        verbose_name_plural = "开盘啦题材成分股"
        unique_together = ("ts_code", "con_code", "trade_date")

    def __str__(self):
        return f"{self.trade_date} - {self.name}({self.ts_code}) - {self.con_name}({self.con_code})"

# 同花顺概念和行业指数
class ThsIndex(models.Model):
    ts_code = models.CharField(max_length=20, verbose_name="代码", unique=True)
    name = models.CharField(max_length=50, verbose_name="名称")
    count = models.IntegerField(verbose_name="成分个数")
    exchange = models.CharField(max_length=10, verbose_name="交易所")
    list_date = models.DateField(verbose_name=_("上市日期"), null=True, blank=True)
    type = models.CharField(max_length=10, verbose_name="指数类型")

    class Meta:
        db_table = "ths_index"
        verbose_name = "同花顺概念和行业指数"
        verbose_name_plural = verbose_name

# 同花顺概念板块成分
class ThsMember(models.Model):
    index = models.ForeignKey('IndexInfo', db_column='ts_code', on_delete=models.CASCADE, related_name="ths_member", verbose_name=_("指数"))
    stock = models.ForeignKey('StockInfo', db_column='con_code', on_delete=models.CASCADE, related_name="ths_member", verbose_name=_("股票"))
    weight = models.FloatField(verbose_name="权重", null=True, blank=True)
    in_date = models.DateField(verbose_name=_("纳入日期"), null=True, blank=True)
    out_date = models.DateField(verbose_name=_("剔除日期"), null=True, blank=True)
    is_new = models.CharField(max_length=2, verbose_name="是否最新", null=True, blank=True)

    class Meta:
        db_table = "ths_member"
        verbose_name = "同花顺概念板块成分"
        verbose_name_plural = verbose_name
        constraints = [
            models.UniqueConstraint(fields=['index', 'stock'], name='unique_index_stock')
        ]

# 东方财富概念板块
class DcIndex(models.Model):
    ts_code = models.CharField(max_length=20, verbose_name="概念代码")
    trade_date = models.DateField(verbose_name=_("交易日期"), null=True, blank=True)
    name = models.CharField(max_length=50, verbose_name="概念名称")
    leading = models.CharField(max_length=50, verbose_name="领涨股票名称")
    stock = models.ForeignKey('StockInfo', db_column='leading_code', on_delete=models.CASCADE, related_name="dc_index", verbose_name=_("股票"))
    pct_change = models.FloatField(verbose_name="涨跌幅")
    leading_pct = models.FloatField(verbose_name="领涨股票涨跌幅")
    total_mv = models.FloatField(verbose_name="总市值(万元)")
    turnover_rate = models.FloatField(verbose_name="换手率")
    up_num = models.IntegerField(verbose_name="上涨家数")
    down_num = models.IntegerField(verbose_name="下降家数")

    class Meta:
        db_table = "dc_index"
        verbose_name = "东方财富概念板块"
        verbose_name_plural = verbose_name
        unique_together = ('ts_code', 'trade_date')

# 东方财富板块成分
class DcMember(models.Model):
    trade_date = models.DateField(verbose_name=_("交易日期"), null=True, blank=True)
    ts_code = models.CharField(max_length=20, verbose_name="概念代码")
    con_code = models.CharField(max_length=20, verbose_name="成分代码")
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, related_name="dc_member", verbose_name=_("股票"))

    class Meta:
        db_table = "dc_member"
        verbose_name = "东方财富板块成分"
        verbose_name_plural = verbose_name
        unique_together = ('trade_date', 'ts_code', 'con_code')
