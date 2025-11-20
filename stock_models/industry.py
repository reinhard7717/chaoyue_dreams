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
    is_pub = models.CharField(max_length=4, null=True, blank=True, verbose_name="是否发布指数")
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
    trade_time = models.DateField(db_index=True, verbose_name="交易日期")
    name = models.CharField(max_length=64, verbose_name="成分股票名称", null=True, blank=True)
    open = models.FloatField(verbose_name="开盘点位", null=True, blank=True)
    low = models.FloatField(verbose_name="最低点位", null=True, blank=True)
    high = models.FloatField(verbose_name="最高点位", null=True, blank=True)
    close = models.FloatField(verbose_name="收盘点位", null=True, blank=True)
    change = models.FloatField(verbose_name="涨跌点位", null=True, blank=True)
    pct_change = models.FloatField(verbose_name="涨跌幅", null=True, blank=True)
    vol = models.FloatField(verbose_name="成交量（万股）", null=True, blank=True)
    amount = models.FloatField(verbose_name="成交额（万元）", null=True, blank=True)
    pe = models.FloatField(verbose_name="市盈率", null=True, blank=True)
    pb = models.FloatField(verbose_name="市净率", null=True, blank=True)
    float_mv = models.FloatField(verbose_name="流通市值（万元）", null=True, blank=True)
    total_mv = models.FloatField(verbose_name="总市值（万元）", null=True, blank=True)
    weight = models.FloatField(verbose_name="权重", blank=True, null=True)
    class Meta:
        db_table = "sw_industry_daily"
        verbose_name = "申万行业日线行情"
        verbose_name_plural = "申万行业日线行情"
        unique_together = ("index", "trade_time")
    def __str__(self):
        index_code = self.index.index_code if self.index else self.ts_code
        return f"{self.trade_time} - {self.name}({index_code})"

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
    name = models.CharField(max_length=50, verbose_name="成分股票名称", null=True, blank=True)
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

# 开盘啦题材字典 (主表)
class KplConceptInfo(models.Model):
    """
    【V2.0 新增】开盘啦题材字典/主表
    用于存储所有出现过的开盘啦题材的基本信息，确保每个题材有唯一的记录。
    """
    id = models.BigAutoField(primary_key=True)
    ts_code = models.CharField(max_length=16, db_index=True, unique=True, verbose_name="题材代码")  # xxxxxx.KP
    name = models.CharField(max_length=64, verbose_name="题材名称", null=True, blank=True, default='')
    class Meta:
        db_table = "kpl_concept_info"
        verbose_name = "开盘啦题材信息"
        verbose_name_plural = "开盘啦题材信息"
    def __str__(self):
        return f"{self.name}({self.ts_code})"

# 开盘啦题材每日快照 (原 KplConcept)
class KplConceptDaily(models.Model):
    """
    【V2.0 重构】开盘啦题材每日快照 (原 KplConcept)
    存储每个交易日题材的动态表现数据。
    """
    id = models.BigAutoField(primary_key=True)
    concept_info = models.ForeignKey(
        KplConceptInfo,
        to_field='ts_code',
        on_delete=models.CASCADE,
        related_name='daily_snapshots',
        verbose_name="关联题材信息"
    )
    trade_time = models.DateField(db_index=True, verbose_name="交易日期")
    z_t_num = models.IntegerField(null=True, blank=True, verbose_name="涨停数量")
    up_num = models.IntegerField(null=True, blank=True, verbose_name="排名上升位数")
    class Meta:
        db_table = "kpl_concept_daily"
        verbose_name = "开盘啦题材每日快照"
        verbose_name_plural = "开盘啦题材每日快照"
        unique_together = ("concept_info", "trade_time")
    def __str__(self):
        return f"{self.trade_time} - {self.concept_info.name}({self.concept_info.ts_code})"

# 开盘啦题材成分股
class KplConceptConstituent(models.Model):
    """
    【V2.0 重构】开盘啦题材成分股
    外键已修改为指向题材主表 KplConceptInfo。
    """
    concept_info = models.ForeignKey(
        KplConceptInfo,
        to_field='ts_code',
        db_column='ts_code',
        on_delete=models.CASCADE,
        related_name='constituents',
        verbose_name="所属题材",
        null=True, blank=True,
    )
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='con_code',
        on_delete=models.CASCADE, blank=True, null=True,
        related_name="kpl_concept_constituent", verbose_name=_("成分股代码")
    )
    trade_time = models.DateField(db_index=True, verbose_name="交易日期")
    desc = models.TextField(null=True, blank=True, verbose_name="描述")
    hot_num = models.IntegerField(null=True, blank=True, verbose_name="人气值")
    class Meta:
        db_table = "kpl_concept_constituent"
        verbose_name = "开盘啦题材成分股"
        verbose_name_plural = "开盘啦题材成分股"
        unique_together = ("concept_info", "stock", "trade_time")
    def __str__(self):
        concept_name = self.concept_info.name if self.concept_info else "N/A"
        stock_name = self.stock.stock_name if self.stock else "N/A"
        return f"{self.trade_time} - {concept_name} - {stock_name}"

# 开盘啦榜单数据
class KplLimitList(models.Model):
    """
    开盘啦榜单数据（涨停、跌停、炸板等）
    这是捕捉市场短线情绪、识别龙头和梯队的核心数据。
    """
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='ts_code',
        on_delete=models.CASCADE,
        related_name="kpl_limit_list",
        verbose_name=_("股票代码")
    )
    trade_date = models.DateField(verbose_name="交易日期", db_index=True)
    name = models.CharField(max_length=64, verbose_name="股票名称", null=True, blank=True)
    tag = models.CharField(max_length=20, verbose_name="榜单类型", db_index=True, help_text="例如: 涨停, 跌停, 炸板")
    # 时间相关
    lu_time = models.CharField(max_length=20, null=True, blank=True, verbose_name="涨停时间")
    ld_time = models.CharField(max_length=20, null=True, blank=True, verbose_name="跌停时间")
    open_time = models.CharField(max_length=20, null=True, blank=True, verbose_name="开板时间")
    last_time = models.CharField(max_length=20, null=True, blank=True, verbose_name="最后涨停时间")
    # 描述与题材
    lu_desc = models.TextField(null=True, blank=True, verbose_name="涨停原因")
    theme = models.TextField(null=True, blank=True, verbose_name="所属板块/题材")
    # 状态与量价核心指标
    status = models.CharField(max_length=50, null=True, blank=True, verbose_name="连板状态", help_text="例如: 首板, 2连板")
    pct_chg = models.FloatField(null=True, blank=True, verbose_name="涨跌幅(%)")
    turnover_rate = models.FloatField(null=True, blank=True, verbose_name="换手率(%)")
    amount = models.FloatField(null=True, blank=True, verbose_name="成交额(元)")
    limit_order = models.FloatField(null=True, blank=True, verbose_name="封单额(元)")
    lu_limit_order = models.FloatField(null=True, blank=True, verbose_name="最大封单额(元)")
    free_float = models.FloatField(null=True, blank=True, verbose_name="实际流通盘(元)")
    # 竞价相关
    bid_change = models.FloatField(null=True, blank=True, verbose_name="竞价净额(元)")
    bid_turnover = models.FloatField(null=True, blank=True, verbose_name="竞价换手率(%)")
    bid_pct_chg = models.FloatField(null=True, blank=True, verbose_name="竞价涨幅(%)")
    class Meta:
        db_table = "kpl_limit_list"
        verbose_name = "开盘啦榜单数据"
        verbose_name_plural = "开盘啦榜单数据"
        unique_together = ("stock", "trade_date", "tag") # 同一只股票同一天在同一个榜单上只应有一条记录
        ordering = ['-trade_date', 'stock']
    def __str__(self):
        return f"{self.trade_date} - {self.name}({self.stock_id}) - {self.tag}"

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
        # 增加健壮性检查，防止关联对象不存在时报错
        ths_index_name = self.ths_index.name if self.ths_index else "N/A"
        ths_index_code = self.ths_index.ts_code if self.ths_index else "N/A"
        return f"{ths_index_code}-{ths_index_name}-{self.stock}"

# 同花顺板块指数行情
class ThsIndexDaily(models.Model):
    ths_index = models.ForeignKey(
        'ThsIndex',
        to_field='ts_code',
        db_column='ts_code',
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

# 东方财富板块指数行情 (模型结构完全重写以匹配 dc_daily API)
class DcIndexDaily(models.Model):
    dc_index = models.ForeignKey(
        'DcIndex',
        db_column='ts_code',
        to_field='ts_code',
        on_delete=models.CASCADE,
        related_name='daily_data',
        verbose_name="东方财富概念板块"
    )
    trade_time = models.DateField(verbose_name="交易日")
    close = models.FloatField(verbose_name="收盘点位", null=True, blank=True)
    open = models.FloatField(verbose_name="开盘点位", null=True, blank=True)
    high = models.FloatField(verbose_name="最高点位", null=True, blank=True)
    low = models.FloatField(verbose_name="最低点位", null=True, blank=True)
    change = models.FloatField(verbose_name="涨跌点位", null=True, blank=True)
    pct_change = models.FloatField(verbose_name="涨跌幅(%)", null=True, blank=True)
    vol = models.FloatField(verbose_name="成交量(手)", null=True, blank=True)
    amount = models.FloatField(verbose_name="成交额(千元)", null=True, blank=True)
    swing = models.FloatField(verbose_name="振幅(%)", null=True, blank=True)
    turnover_rate = models.FloatField(verbose_name="换手率(%)", null=True, blank=True)
    class Meta:
        db_table = "dc_index_daily"
        verbose_name = "东方财富板块指数行情"
        verbose_name_plural = verbose_name
        unique_together = ('dc_index', 'trade_time')
        ordering = ['-trade_time']
    def __str__(self):
        return f"{self.dc_index.ts_code} - {self.trade_time}"

# 东方财富板块成分 (模型结构保持不变，与 dc_member API 匹配)
class DcIndexMember(models.Model):
    trade_time = models.DateField(verbose_name="交易日期", db_index=True)
    dc_index = models.ForeignKey(
        'DcIndex',
        to_field='ts_code',
        on_delete=models.CASCADE,
        related_name="dc_member",
        verbose_name="指数"
    )
    stock = models.ForeignKey(
        'StockInfo',
        on_delete=models.CASCADE,
        db_column='con_code',
        to_field='stock_code',
        related_name="dc_member",
        verbose_name="股票"
    )
    name = models.CharField(max_length=50, null=True, blank=True, verbose_name="股票名称")
    class Meta:
        db_table = "dc_index_member"
        verbose_name = "东方财富板块成分"
        verbose_name_plural = verbose_name
        unique_together = ('trade_time', 'dc_index', 'stock')
    def __str__(self):
        return f"{self.dc_index.ts_code} - {self.trade_time} - {self.stock.stock_code}"

# 板块/概念主数据模型
class ConceptMaster(models.Model):
    """
    【V3.0 新增】板块/概念主数据模型
    - 核心职责: 作为所有不同来源（申万、同花顺、东方财富等）板块/概念的统一“户籍管理中心”。
    - 设计思想: 将板块的静态信息与来源解耦，为下游的通用计算和融合分析提供统一接口。
    """
    id = models.BigAutoField(primary_key=True)
    code = models.CharField(max_length=32, unique=True, db_index=True, verbose_name="板块/概念代码")
    name = models.CharField(max_length=100, verbose_name="名称")
    source = models.CharField(max_length=20, db_index=True, verbose_name="来源", help_text="例如: 'ths', 'sw', 'dc', 'kpl'")
    type = models.CharField(max_length=20, null=True, blank=True, verbose_name="类型", help_text="例如: '行业', '概念', '地域'")
    class Meta:
        db_table = "concept_master"
        verbose_name = "板块/概念主数据"
        verbose_name_plural = verbose_name
        ordering = ['source', 'name']
    def __str__(self):
        return f"[{self.source}] {self.name} ({self.code})"

class ConceptMember(models.Model):
    """
    【V3.0 新增】板块/概念成分股统一模型
    - 核心职责: 记录所有来源的板块/概念与其成分股在特定时间范围内的关系。
    - 设计思想: 这是进行严格历史回测和深度板块分析（如上涨广度、龙头效应）的基石。
    """
    id = models.BigAutoField(primary_key=True)
    concept = models.ForeignKey(
        'ConceptMaster',
        on_delete=models.CASCADE,
        to_field='code',
        db_column='concept_code',
        related_name='members',
        verbose_name="所属板块/概念"
    )
    stock = models.ForeignKey(
        'StockInfo',
        on_delete=models.CASCADE,
        to_field='stock_code',
        db_column='stock_code',
        related_name='concept_memberships',
        verbose_name="成分股"
    )
    source = models.CharField(max_length=20, db_index=True, verbose_name="来源", help_text="例如: 'ths', 'sw', 'dc'")
    # 对于有明确纳入/剔除日期的来源 (如申万、同花顺)
    in_date = models.DateField(verbose_name="纳入日期", db_index=True)
    out_date = models.DateField(verbose_name="剔除日期", null=True, blank=True)
    # 对于每日快照的来源 (如东方财富)，trade_date 就是 in_date，out_date 为空
    # is_new 字段可以被 in_date 和 out_date 的逻辑替代，故不再需要
    class Meta:
        db_table = "concept_member"
        verbose_name = "板块/概念成分股"
        verbose_name_plural = verbose_name
        # 联合唯一约束，确保同一来源、同一板块、同一股票、同一纳入日期的记录是唯一的
        unique_together = ('concept', 'stock', 'in_date', 'source')
        ordering = ['-in_date', 'concept', 'stock']
    def __str__(self):
        return f"[{self.source}] {self.concept.name} -> {self.stock.stock_name} (自 {self.in_date})"

class ConceptDaily(models.Model):
    """
    【V3.0 新增】板块/概念日线行情模型
    - 核心职责: 存储所有板块/概念的标准化日线行情数据。
    """
    concept = models.ForeignKey(
        ConceptMaster,
        on_delete=models.CASCADE,
        to_field='code',
        db_column='concept_code',
        related_name='daily_data',
        verbose_name="关联板块/概念"
    )
    trade_date = models.DateField(verbose_name="交易日期", db_index=True)
    open = models.FloatField(verbose_name="开盘点位", null=True, blank=True)
    high = models.FloatField(verbose_name="最高点位", null=True, blank=True)
    low = models.FloatField(verbose_name="最低点位", null=True, blank=True)
    close = models.FloatField(verbose_name="收盘点位", null=True, blank=True)
    pre_close = models.FloatField(verbose_name="昨收点位", null=True, blank=True)
    change = models.FloatField(verbose_name="涨跌点位", null=True, blank=True)
    pct_change = models.FloatField(verbose_name="涨跌幅(%)", null=True, blank=True)
    vol = models.FloatField(verbose_name="成交量(手/万股)", null=True, blank=True)
    amount = models.FloatField(verbose_name="成交额(元/万元)", null=True, blank=True)
    turnover_rate = models.FloatField(verbose_name="换手率(%)", null=True, blank=True)
    class Meta:
        db_table = "concept_daily"
        verbose_name = "板块/概念日线行情"
        verbose_name_plural = verbose_name
        unique_together = ('concept', 'trade_date')
        ordering = ['-trade_date']

class IndustryLifecycle(models.Model):
    """
    行业生命周期预计算结果
    - 核心职责: 存储每日计算出的各行业强度排名、趋势及所处生命周期阶段。
    - 数据来源: 由 ContextualAnalysisService.analyze_industry_rotation 方法每日计算并写入。
    """
    concept = models.ForeignKey(
        'ConceptMaster',
        on_delete=models.CASCADE,
        to_field='code',
        db_column='concept_code',
        related_name='lifecycle_data',
        verbose_name="板块/概念",
    )
    trade_date = models.DateField(verbose_name="交易日期", db_index=True)
    source = models.CharField(max_length=20, db_index=True, verbose_name="来源", help_text="冗余字段，便于查询")
    strength_rank = models.FloatField(verbose_name="强度排名(0-1)", null=True)
    rank_slope = models.FloatField(verbose_name="排名斜率", null=True)
    rank_accel = models.FloatField(verbose_name="排名加速度", null=True)
    lifecycle_stage = models.CharField(
        max_length=20,
        verbose_name="生命周期阶段",
        choices=[
            ('PREHEAT', '预热期'),
            ('MARKUP', '主升段'),
            ('STAGNATION', '滞涨期'),
            ('DOWNTREND', '下跌段'),
            ('TRANSITION', '过渡期'),
        ],
        null=True,
        blank=True
    )
    breadth_score = models.FloatField(verbose_name="内部广度分(0-1)", null=True)
    leader_score = models.FloatField(verbose_name="龙头效应分(0-1)", null=True)
    class Meta:
        db_table = "industry_lifecycle"
        verbose_name = "行业生命周期"
        verbose_name_plural = "行业生命周期"
        unique_together = ('concept', 'trade_date')
        ordering = ['-trade_date', 'strength_rank']
    def __str__(self):
        concept_name = self.concept.name if self.concept else "N/A"
        return f"{self.trade_date} - {concept_name} - {self.lifecycle_stage}"




















