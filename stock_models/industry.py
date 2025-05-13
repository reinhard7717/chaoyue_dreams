from django.db import models
from django.utils.translation import gettext_lazy as _

# 申万行业分类
class SwIndustry(models.Model):
    index_code = models.CharField(max_length=16, db_index=True, unique=True, verbose_name="指数代码")  # 801xxx.SI
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
        to_field='index_code',  # 指定外键对应SwIndustry的index_code字段
        db_column='l3_code', # 数据库字段名
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
    index = models.ForeignKey(
        'IndexInfo',
        to_field='index_code',  # 指定外键对应IndexInfo的ts_code字段
        db_column='ts_code', # 数据库字段名
        db_constraint=False,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='sw_daily',
        verbose_name="关联指数基础信息"
    )
    trade_time = models.CharField(max_length=8, db_index=True, verbose_name="交易日期")
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
    weight = models.FloatField(verbose_name="权重", blank=True, null=True)

    class Meta:
        db_table = "sw_industry_daily"
        verbose_name = "申万行业日线行情"
        verbose_name_plural = "申万行业日线行情"
        unique_together = ("index", "trade_time")

    def __str__(self):
        return f"{self.trade_time} - {self.name}({self.ts_code})"

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
    trade_time = models.DateField(verbose_name=_("交易日期"), null=True, blank=True)
    open = models.FloatField(verbose_name="开盘点位")
    low = models.FloatField(verbose_name="最低点位")
    high = models.FloatField(verbose_name="最高点位")
    close = models.FloatField(verbose_name="收盘点位")
    pre_close = models.FloatField(verbose_name="昨日收盘点位", null=True, blank=True )
    change = models.FloatField(verbose_name="涨跌点位")
    pct_change = models.FloatField(verbose_name="涨跌幅")
    vol = models.FloatField(verbose_name="成交量(万股)")
    amount = models.FloatField(verbose_name="成交额(万元)")

    class Meta:
        db_table = "ci_daily"
        verbose_name = "中信行业指数日线行情"
        verbose_name_plural = verbose_name
        unique_together = ('ts_code', 'trade_time')

# 开盘啦题材库
class KplConcept(models.Model):
    """
    开盘啦题材库
    """
    id = models.BigAutoField(primary_key=True)
    trade_time = models.CharField(max_length=8, db_index=True, verbose_name="交易日期")  # YYYYMMDD
    ts_code = models.CharField(max_length=16, db_index=True, unique=True, verbose_name="题材代码")    # xxxxxx.KP
    name = models.CharField(max_length=64, verbose_name="题材名称")
    z_t_num = models.IntegerField(null=True, blank=True, verbose_name="涨停数量")
    up_num = models.IntegerField(null=True, blank=True, verbose_name="排名上升位数")

    class Meta:
        db_table = "kpl_concept"
        verbose_name = "开盘啦题材"
        verbose_name_plural = "开盘啦题材"
        unique_together = ("trade_time", "ts_code")

    def __str__(self):
        return f"{self.trade_time} - {self.name}({self.ts_code})"

# 开盘啦题材成分股
class KplConceptConstituent(models.Model):
    """
    开盘啦题材成分股
    """
    concept = models.ForeignKey(
        KplConcept,
        to_field='ts_code',  # 指定外键对应StockInfo的stock_code字段
        db_column='ts_code', # 数据库字段名
        on_delete=models.CASCADE,
        related_name='constituents',
        verbose_name="所属题材"
    )
    name = models.CharField(max_length=64, verbose_name="题材名称")
    con_name = models.CharField(max_length=64, verbose_name="成分股名称")
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',  # 指定外键对应StockInfo的stock_code字段
        db_column='con_code', # 数据库字段名
        on_delete=models.CASCADE, blank=True, null=True,
        related_name="kpl_concept_constituent", verbose_name=_("成分股代码")
    )
    trade_time = models.CharField(max_length=8, db_index=True, verbose_name="交易日期")  # YYYYMMDD
    desc = models.TextField(null=True, blank=True, verbose_name="描述")
    hot_num = models.IntegerField(null=True, blank=True, verbose_name="人气值")

    class Meta:
        db_table = "kpl_concept_constituent"
        verbose_name = "开盘啦题材成分股"
        verbose_name_plural = "开盘啦题材成分股"
        unique_together = ("concept", "stock", "trade_time")

    def __str__(self):
        return f"{self.trade_time} - {self.name}({self.ts_code}) - {self.con_name}({self.con_code})"

# 同花顺行业概念板块
# 指数类型 N-概念指数 I-行业指数 R-地域指数 S-同花顺特色指数 ST-同花顺风格指数 TH-同花顺主题指数 BB-同花顺宽基指数
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

    def __str__(self):
        return f"{self.ts_code}-{self.name}"

# 同花顺概念板块成分
class ThsIndexMember(models.Model):
    ths_index = models.ForeignKey('ThsIndex', db_column='ts_code', to_field='ts_code', null=True, blank=True, on_delete=models.CASCADE, related_name="ths_member", verbose_name=_("指数"))
    stock = models.ForeignKey('StockInfo', db_column='con_code', to_field='stock_code', on_delete=models.CASCADE, related_name="ths_member", verbose_name=_("股票"))
    weight = models.FloatField(verbose_name="权重", null=True, blank=True)
    in_date = models.DateField(verbose_name=_("纳入日期"), null=True, blank=True)
    out_date = models.DateField(verbose_name=_("剔除日期"), null=True, blank=True)
    is_new = models.CharField(max_length=2, verbose_name="是否最新", null=True, blank=True)

    class Meta:
        db_table = "ths_index_member"
        verbose_name = "同花顺概念板块成分"
        verbose_name_plural = verbose_name
        constraints = [
            models.UniqueConstraint(fields=['ths_index', 'stock'], name='unique_index_stock')
        ]
    def __str__(self):
        return f"{self.ths_index.ts_code}-{self.ths_index.name}-{self.stock}"

# 同花顺板块指数行情
class ThsIndexDaily(models.Model):
    ths_index = models.ForeignKey(
        'ThsIndex',
        on_delete=models.CASCADE,
        related_name='daily_data',
        verbose_name="所属指数"
    )
    trade_time = models.DateField(verbose_name="交易日")
    close = models.FloatField(verbose_name="收盘点位", null=True, blank=True)
    open = models.FloatField(verbose_name="开盘点位", null=True, blank=True)
    high = models.FloatField(verbose_name="最高点位", null=True, blank=True)
    low = models.FloatField(verbose_name="最低点位", null=True, blank=True)
    pre_close = models.FloatField(verbose_name="昨日收盘点", null=True, blank=True)
    avg_price = models.FloatField(verbose_name="平均价", null=True, blank=True)
    change = models.FloatField(verbose_name="涨跌点位", null=True, blank=True)
    pct_change = models.FloatField(verbose_name="涨跌幅", null=True, blank=True)
    vol = models.FloatField(verbose_name="成交量", null=True, blank=True)
    turnover_rate = models.FloatField(verbose_name="换手率", null=True, blank=True)
    total_mv = models.FloatField(verbose_name="总市值", null=True, blank=True)
    float_mv = models.FloatField(verbose_name="流通市值", null=True, blank=True)
    pe_ttm = models.FloatField(verbose_name="市盈率TTM", null=True, blank=True)
    pb_mrq = models.FloatField(verbose_name="市净率MRQ", null=True, blank=True)

    class Meta:
        db_table = "ths_index_daily"
        verbose_name = "同花顺板块指数行情"
        verbose_name_plural = verbose_name
        unique_together = ('ths_index', 'trade_time')  # 保证同一指数同一天只有一条数据
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.ths_index.ts_code} - {self.trade_date}"

# 东方财富概念板块
class DcIndex(models.Model):
    ts_code = models.CharField(max_length=20, verbose_name="概念代码", unique=True)
    name = models.CharField(max_length=50, verbose_name="概念名称", null=True, blank=True)
    exchange = models.CharField(max_length=10, verbose_name="交易所", null=True, blank=True)
    type = models.CharField(max_length=10, verbose_name="指数类型", null=True, blank=True)

    class Meta:
        db_table = "dc_index"
        verbose_name = "东方财富概念板块"
        verbose_name_plural = verbose_name
    
    def __str__(self):
        return f"{self.ts_code}-{self.name}"

# 东方财富板块指数行情
class DcIndexDaily(models.Model):
    dc_index = models.ForeignKey(
        'DcIndex',
        on_delete=models.CASCADE,
        to_field='ts_code',  # 指定外键对应DcIndex的ts_code字段
        db_column='ts_code', # 数据库字段名
        related_name='daily_data',
        null=True,  # 允许为空
        blank=True, # 后台表单也允许为空
        verbose_name="东方财富概念板块"
    )
    trade_time = models.DateField(verbose_name=_("交易日期"), null=True, blank=True)
    name = models.CharField(max_length=50, verbose_name="概念名称")
    leading = models.CharField(max_length=50, verbose_name="领涨股票名称")
    stock = models.ForeignKey('StockInfo', db_column='leading_code', to_field='stock_code', on_delete=models.CASCADE, related_name="dc_index_daily", verbose_name=_("股票"))
    pct_change = models.FloatField(verbose_name="涨跌幅")
    leading_pct = models.FloatField(verbose_name="领涨股票涨跌幅")
    total_mv = models.FloatField(verbose_name="总市值(万元)")
    turnover_rate = models.FloatField(verbose_name="换手率")
    up_num = models.IntegerField(verbose_name="上涨家数")
    down_num = models.IntegerField(verbose_name="下降家数")

    class Meta:
        db_table = "dc_index_daily"
        verbose_name = "东方财富板块指数行情"
        verbose_name_plural = verbose_name
        unique_together = ('dc_index', 'trade_time')
    
    def __str__(self):
        return f"{self.dc_index.ts_code}-{self.dc_index.name}"

# 东方财富板块成分
class DcIndexMember(models.Model):
    trade_time = models.DateField(verbose_name=_("交易日期"), null=True, blank=True)
    dc_index = models.ForeignKey('DcIndex', db_column='ts_code', null=True, blank=True, on_delete=models.CASCADE, related_name="dc_member", verbose_name=_("指数"))
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, db_column='con_code', to_field='stock_code', related_name="dc_member", verbose_name=_("股票"))
    name = models.CharField(max_length=50, null=True, blank=True, verbose_name="股票名称")
    class Meta:
        db_table = "dc_index_member"
        verbose_name = "东方财富板块成分"
        verbose_name_plural = verbose_name
        unique_together = ('trade_time', 'dc_index', 'stock')
    def __str__(self):
        return f"{self.dc_index.ts_code} - {self.trade_date} - {self.stock}"
