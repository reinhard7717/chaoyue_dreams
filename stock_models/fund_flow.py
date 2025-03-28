from django.db import models
from django.utils.translation import gettext_lazy as _
from stock_models.stock_basic import StockInfo
from utils.models import BaseModel

class FundFlowMinute(BaseModel):
    """
    分钟级资金流向数据
    """
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, blank=True, null=True, related_name="fund_flow_minute", verbose_name=_("股票"))
    trade_time = models.DateTimeField(verbose_name=_("交易时间"), null=True, blank=True)
    change_percent = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("涨跌幅(%)"), null=True, blank=True)
    inflow_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("总流入资金(元)"), null=True, blank=True)
    outflow_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("总流出资金(元)"), null=True, blank=True)
    net_inflow = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("净流入(元)"), null=True, blank=True)
    net_inflow_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("净流入率(%)"), null=True, blank=True)
    main_inflow_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("主力流入率(%)"), null=True, blank=True)
    retail_inflow_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("散户流入率(%)"), null=True, blank=True)
    
    class Meta:
        verbose_name = _("分钟级资金流向")
        verbose_name_plural = _("分钟级资金流向")
        db_table = "fund_flow_minute"
        unique_together = [['stock', 'trade_time']]
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_time']),
        ]
    
    def __str__(self):
        return f"{self.stock.name}分钟级资金流向({self.trade_time})"

class FundFlowDaily(BaseModel):
    """
    日级资金流向数据
    """
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, blank=True, null=True, related_name="fund_flow_daily", verbose_name=_("股票"))
    trade_date = models.DateField(verbose_name=_("交易日期"), null=True, blank=True)
    change_percent = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("涨跌幅(%)"), null=True, blank=True)
    turnover_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("换手率(%)"), null=True, blank=True)
    net_inflow = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("净流入(元)"), null=True, blank=True)
    net_inflow_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("净流入率(%)"), null=True, blank=True)
    main_net_inflow = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("主力净流入(元)"), null=True, blank=True)
    main_net_inflow_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("主力净流入率(%)"), null=True, blank=True)
    industry_net_inflow = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("行业净流入(元)"), null=True, blank=True)
    industry_net_inflow_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("行业净流入率(%)"), null=True, blank=True)
    
    class Meta:
        verbose_name = _("日级资金流向")
        verbose_name_plural = _("日级资金流向")
        db_table = "fund_flow_daily"
        unique_together = [['stock', 'trade_date']]
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_date']),
        ]
    
    def __str__(self):
        return f"{self.stock.name}日级资金流向({self.trade_date})"

class MainForcePhase(BaseModel):
    """
    阶段主力动向数据
    """
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, blank=True, null=True, related_name="main_force_phase", verbose_name=_("股票"))
    trade_date = models.DateField(verbose_name=_("交易日期"), null=True, blank=True)
    inflow_3day = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("近3日主力净流入(元)"), null=True, blank=True)
    inflow_5day = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("近5日主力净流入(元)"), null=True, blank=True)
    inflow_10day = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("近10日主力净流入(元)"), null=True, blank=True)
    inflow_rate_3day = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("近3日主力净流入率(%)"), null=True, blank=True)
    inflow_rate_5day = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("近5日主力净流入率(%)"), null=True, blank=True)
    inflow_rate_10day = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("近10日主力净流入率(%)"), null=True, blank=True)
    
    class Meta:
        verbose_name = _("阶段主力动向")
        verbose_name_plural = _("阶段主力动向")
        db_table = "main_force_phase"
        unique_together = [['stock', 'trade_date']]
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_date']),
        ]
    
    def __str__(self):
        return f"{self.stock.name}阶段主力动向({self.trade_date})"

class TransactionDistribution(BaseModel):
    """
    历史成交分布数据
    """
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, blank=True, null=True, related_name="transaction_distribution", verbose_name=_("股票"))
    trade_date = models.DateField(verbose_name=_("交易日期"), null=True, blank=True)
    close_price = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("收盘价(元)"), null=True, blank=True)
    change_percent = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("涨跌幅(%)"), null=True, blank=True)
    net_inflow_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("净流入率(%)"), null=True, blank=True)
    turnover_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("换手率(%)"), null=True, blank=True)
    total_net_inflow = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("全部净流入(元)"), null=True, blank=True)
    super_large_inflow = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("超大单流入(元)"), null=True, blank=True)
    super_large_net_inflow = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("超大单净流入(元)"), null=True, blank=True)
    large_inflow = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("大单流入(元)"), null=True, blank=True)
    large_net_inflow = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("大单净流入(元)"), null=True, blank=True)
    small_inflow = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("小单流入(元)"), null=True, blank=True)
    small_net_inflow = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("小单净流入(元)"), null=True, blank=True)
    retail_inflow = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("散单流入(元)"), null=True, blank=True)
    retail_net_inflow = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("散单净流入(元)"), null=True, blank=True)
    
    class Meta:
        verbose_name = _("历史成交分布")
        verbose_name_plural = _("历史成交分布")
        db_table = "transaction_distribution"
        unique_together = [['stock', 'trade_date']]
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_date']),
        ]
    
    def __str__(self):
        return f"{self.stock.name}历史成交分布({self.trade_date})"

class StockPool(BaseModel):
    """
    股票池基础模型
    
    作为所有股票池模型的抽象基类，包含共同字段
    """
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, blank=True, null=True, related_name="stock_pool", verbose_name=_("股票"))
    date = models.DateField(verbose_name=_("日期"), null=True, blank=True)
    code = models.CharField(max_length=20, verbose_name=_("股票代码"), null=True, blank=True)
    name = models.CharField(max_length=50, verbose_name=_("股票名称"), null=True, blank=True)
    price = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("价格(元)"), null=True, blank=True)
    change_percent = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("涨跌幅(%)"), null=True, blank=True)
    turnover = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("成交额(元)"), null=True, blank=True)
    circulating_market_value = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("流通市值(元)"), null=True, blank=True)
    total_market_value = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("总市值(元)"), null=True, blank=True)
    turnover_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("换手率(%)"), null=True, blank=True)
    
    class Meta:
        abstract = True
        indexes = [
            models.Index(fields=['date']),
            models.Index(fields=['code']),
        ]

class LimitUpPool(StockPool):
    """
    涨停股池模型
    
    存储每日涨停股票数据，根据封板时间升序
    """
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, blank=True, null=True, related_name="limit_up_pool", verbose_name=_("股票"))
    consecutive_limit_days = models.IntegerField(verbose_name=_("连板数"), null=True, blank=True)
    first_limit_time = models.TimeField(verbose_name=_("首次封板时间"), null=True, blank=True)
    last_limit_time = models.TimeField(verbose_name=_("最后封板时间"), null=True, blank=True)
    limit_funds = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("封板资金(元)"))
    break_times = models.IntegerField(verbose_name=_("炸板次数"))
    limit_statistics = models.CharField(max_length=20, verbose_name=_("涨停统计"))
    
    class Meta:
        verbose_name = _("涨停股池")
        verbose_name_plural = _("涨停股池")
        db_table = "limit_up_pool"
        unique_together = [['date', 'code']]
    
    def __str__(self):
        return f"涨停股{self.name}({self.date})"


class LimitDownPool(StockPool):
    """
    跌停股池模型
    
    存储每日跌停股票数据，根据封单资金升序
    """
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, blank=True, null=True, related_name="limit_down_pool", verbose_name=_("股票"))
    pe_ratio = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("动态市盈率"), null=True, blank=True)
    consecutive_limit_days = models.IntegerField(verbose_name=_("连续跌停次数"), null=True, blank=True)
    last_limit_time = models.TimeField(verbose_name=_("最后封板时间"), null=True, blank=True)
    limit_funds = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("封单资金(元)"), null=True, blank=True)
    turnover_on_limit = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("板上成交额(元)"), null=True, blank=True)
    open_times = models.IntegerField(verbose_name=_("开板次数"), null=True, blank=True)
    
    class Meta:
        verbose_name = _("跌停股池")
        verbose_name_plural = _("跌停股池")
        db_table = "limit_down_pool"
        unique_together = [['date', 'code']]
    
    def __str__(self):
        return f"跌停股{self.name}({self.date})"


class StrongStockPool(StockPool):
    """
    强势股池模型
    
    存储每日强势股票数据，根据涨幅倒序
    """
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, blank=True, null=True, related_name="strong_stock_pool", verbose_name=_("股票"))
    limit_up_price = models.DecimalField(max_digits=12, decimal_places=4, null=True, verbose_name=_("涨停价(元)"))
    change_speed = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("涨速(%)"))
    is_new_high = models.BooleanField(verbose_name=_("是否新高"))
    volume_ratio = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("量比"))
    limit_statistics = models.CharField(max_length=20, verbose_name=_("涨停统计"))
    
    class Meta:
        verbose_name = _("强势股池")
        verbose_name_plural = _("强势股池")
        db_table = "strong_stock_pool"
        unique_together = [['date', 'code']]
    
    def __str__(self):
        return f"强势股{self.name}({self.date})"


class NewStockPool(StockPool):
    """
    次新股池模型
    
    存储每日次新股票数据，根据开板几日升序
    """
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, blank=True, null=True, related_name="new_stock_pool", verbose_name=_("股票"))
    limit_up_price = models.DecimalField(max_digits=12, decimal_places=4, null=True, verbose_name=_("涨停价(元)"))
    is_new_high = models.BooleanField(verbose_name=_("是否新高"))
    limit_statistics = models.CharField(max_length=20, verbose_name=_("涨停统计"))
    days_after_open = models.IntegerField(verbose_name=_("开板几日"))
    open_date = models.DateField(verbose_name=_("开板日期"))
    ipo_date = models.DateField(verbose_name=_("上市日期"))
    
    class Meta:
        verbose_name = _("次新股池")
        verbose_name_plural = _("次新股池")
        db_table = "new_stock_pool"
        unique_together = [['date', 'code']]
    
    def __str__(self):
        return f"次新股{self.name}({self.date})"


class BreakLimitPool(StockPool):
    """
    炸板股池模型
    
    存储每日炸板股票数据，根据首次封板时间升序
    """
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, blank=True, null=True, related_name="break_limit_pool", verbose_name=_("股票"))
    limit_up_price = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("涨停价(元)"))
    change_speed = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("涨速(%)"))
    limit_statistics = models.CharField(max_length=20, verbose_name=_("涨停统计"))
    first_limit_time = models.TimeField(verbose_name=_("首次封板时间"))
    break_times = models.IntegerField(verbose_name=_("炸板次数"))
    
    class Meta:
        verbose_name = _("炸板股池")
        verbose_name_plural = _("炸板股池")
        db_table = "break_limit_pool"
        unique_together = [['date', 'code']]
    
    def __str__(self):
        return f"炸板股{self.name}({self.date})"
