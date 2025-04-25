from django.db import models
from django.utils.translation import gettext_lazy as _
from stock_models.stock_basic import StockInfo

class FundFlowDaily(models.Model):
    """
    日级资金流向数据（moneyflow接口）
    """
    stock = models.ForeignKey(
        StockInfo,
        to_field='stock_code',  # 指定外键对应StockInfo的stock_code字段
        db_column='ts_code', # 数据库字段名
        on_delete=models.CASCADE, blank=True, null=True,
        related_name="fund_flow_daily", verbose_name=_("股票")
    )
    trade_time = models.DateField(verbose_name=_("交易日期"), null=True, blank=True)
    buy_sm_vol = models.IntegerField(verbose_name=_("小单买入量(手)"), null=True, blank=True)
    buy_sm_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("小单买入金额(万元)"), null=True, blank=True)
    sell_sm_vol = models.IntegerField(verbose_name=_("小单卖出量(手)"), null=True, blank=True)
    sell_sm_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("小单卖出金额(万元)"), null=True, blank=True)
    buy_md_vol = models.IntegerField(verbose_name=_("中单买入量(手)"), null=True, blank=True)
    buy_md_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("中单买入金额(万元)"), null=True, blank=True)
    sell_md_vol = models.IntegerField(verbose_name=_("中单卖出量(手)"), null=True, blank=True)
    sell_md_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("中单卖出金额(万元)"), null=True, blank=True)
    buy_lg_vol = models.IntegerField(verbose_name=_("大单买入量(手)"), null=True, blank=True)
    buy_lg_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("大单买入金额(万元)"), null=True, blank=True)
    sell_lg_vol = models.IntegerField(verbose_name=_("大单卖出量(手)"), null=True, blank=True)
    sell_lg_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("大单卖出金额(万元)"), null=True, blank=True)
    buy_elg_vol = models.IntegerField(verbose_name=_("特大单买入量(手)"), null=True, blank=True)
    buy_elg_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("特大单买入金额(万元)"), null=True, blank=True)
    sell_elg_vol = models.IntegerField(verbose_name=_("特大单卖出量(手)"), null=True, blank=True)
    sell_elg_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("特大单卖出金额(万元)"), null=True, blank=True)
    net_mf_vol = models.IntegerField(verbose_name=_("净流入量(手)"), null=True, blank=True)
    net_mf_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("净流入额(万元)"), null=True, blank=True)

    class Meta:
        verbose_name = _("日级资金流向")
        verbose_name_plural = _("日级资金流向")
        db_table = "fund_flow_daily"
        unique_together = ['stock', 'trade_time']
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_time']),
        ]

    def __str__(self):
        return f"{self.stock.name if self.stock else ''}日级资金流向({self.trade_time})"

    def __code__(self):
        return self.stock.stock_code if self.stock else ''
    
class FundFlowDailyTHS(models.Model):
    """
    日级资金流向数据 - 同花顺（moneyflow_ths接口）
    """
    stock = models.ForeignKey(
        StockInfo,
        to_field='stock_code',  # 指定外键对应StockInfo的stock_code字段
        db_column='ts_code', # 数据库字段名
        on_delete=models.CASCADE, blank=True, null=True,
        related_name="fund_flow_daily_ths", verbose_name=_("股票")
    )
    trade_time = models.DateField(verbose_name=_("交易日期"), null=True, blank=True)
    pct_change = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("涨跌幅(%)"), null=True, blank=True)
    net_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("资金净流入(万元)"), null=True, blank=True)
    net_d5_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("5日主力净额(万元)"), null=True, blank=True)
    buy_lg_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日大单净流入额(万元)"), null=True, blank=True)
    buy_lg_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日大单净流入率(%)"), null=True, blank=True)
    buy_md_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日中单净流入额(万元)"), null=True, blank=True)
    buy_md_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日中单净流入率(%)"), null=True, blank=True)
    buy_sm_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日小单净流入额(万元)"), null=True, blank=True)
    buy_sm_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日小单净流入率(%)"), null=True, blank=True)

    class Meta:
        verbose_name = _("日级资金流向 - 同花顺")
        verbose_name_plural = _("日级资金流向 - 同花顺")
        db_table = "fund_flow_daily_ths"
        unique_together = ['stock', 'trade_time']
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_time']),
        ]

    def __str__(self):
        return f"{self.stock.name if self.stock else ''}日级资金流向 - 同花顺({self.trade_time})"

    def __code__(self):
        return self.stock.stock_code if self.stock else ''

class FundFlowDailyDC(models.Model):
    """
    日级资金流向数据 - 东方财富（moneyflow_dc接口）
    """
    stock = models.ForeignKey(
        StockInfo,
        to_field='stock_code',  # 指定外键对应StockInfo的stock_code字段
        db_column='ts_code', # 数据库字段名
        on_delete=models.CASCADE, blank=True, null=True,
        related_name="fund_flow_daily_dc", verbose_name=_("股票")
    )
    trade_time = models.DateField(verbose_name=_("交易日期"), null=True, blank=True)
    pct_change = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("涨跌幅(%)"), null=True, blank=True)
    net_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日主力净流入额（万元）"), null=True, blank=True)
    net_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日主力净流入率（%）"), null=True, blank=True)
    net_d5_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("5日主力净额（万元）"), null=True, blank=True)
    buy_lg_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日大单净流入额（万元）"), null=True, blank=True)
    buy_lg_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日大单净流入率（%）"), null=True, blank=True)
    buy_md_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日中单净流入额（万元）"), null=True, blank=True)
    buy_md_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日中单净流入率（%）"), null=True, blank=True)
    buy_sm_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日小单净流入额（万元）"), null=True, blank=True)
    buy_sm_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日小单净流入率（%）"), null=True, blank=True)

    class Meta:
        verbose_name = _("日级资金流向 - 东方财富")
        verbose_name_plural = _("日级资金流向 - 东方财富")
        db_table = "fund_flow_daily_dc"
        unique_together = ['stock', 'trade_time']
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_time']),
        ]

    def __str__(self):
        return f"{self.stock.name if self.stock else ''}日级资金流向 - 东方财富({self.trade_time})"

    def __code__(self):
        return self.stock.stock_code if self.stock else ''

class FundFlowCntTHS(models.Model):
    """
    板块资金流向统计数据 - 同花顺（moneyflow_ind_ths接口）
    """
    stock = models.ForeignKey(
        StockInfo,
        to_field='stock_code',  # 指定外键对应StockInfo的stock_code字段
        db_column='ts_code', # 数据库字段名
        on_delete=models.CASCADE, blank=True, null=True,
        related_name="fund_flow_cnt_ths", verbose_name=_("股票")
    )
    trade_time = models.DateField(verbose_name=_("交易日期"), null=True, blank=True)
    lead_stock = models.CharField(max_length=20, verbose_name=_("领涨股"), null=True, blank=True)
    pct_change = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("行业涨跌幅(%)"), null=True, blank=True)
    industry_index = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("板块指数"), null=True, blank=True)
    company_num = models.IntegerField(verbose_name=_("公司数量"), null=True, blank=True)
    pct_change_stock = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("领涨股涨跌幅(%)"), null=True, blank=True)
    net_buy_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("流入资金(亿元)"), null=True, blank=True)
    net_sell_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("流出资金(亿元)"), null=True, blank=True)
    net_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("净额(亿元)"), null=True, blank=True)

    class Meta:
        verbose_name = _("板块资金流向统计")
        verbose_name_plural = _("板块资金流向统计")
        db_table = "fund_flow_cnt_ths"
        unique_together = ['stock', 'trade_time']
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_time']),
        ]
    def __str__(self):
        return f"{self.stock.name if self.stock else ''}板块资金流向统计 - 同花顺({self.trade_time})"
    def __code__(self):
        return self.stock.stock_code if self.stock else ''

class FundFlowCntDC(models.Model):
    """
    板块资金流向统计数据 - 东方财富（moneyflow_ind_dc接口）
    """
    stock = models.ForeignKey(
        StockInfo,
        to_field='stock_code',  # 指定外键对应StockInfo的stock_code字段
        db_column='ts_code', # 数据库字段名
        on_delete=models.CASCADE, blank=True, null=True,
        related_name="fund_flow_cnt_dc", verbose_name=_("股票")
    )
    trade_time = models.DateField(verbose_name=_("交易日期"), null=True, blank=True)
    content_type = models.CharField(max_length=20, verbose_name=_("数据类型"), null=True, blank=True)
    name = models.CharField(max_length=20, verbose_name=_("板块名称"), null=True, blank=True)
    pct_change = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("行业涨跌幅(%)"), null=True, blank=True)
    close = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("板块指数"), null=True, blank=True)
    net_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日主力净流入 净额（元）"), null=True, blank=True)
    net_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日主力净流入 净流入率（%）"), null=True, blank=True)
    buy_elg_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日超大单净流入 净额（元）"), null=True, blank=True)
    buy_elg_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日超大单净流入 净流入率（%）"), null=True, blank=True)
    buy_lg_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日大单净流入 净额（元）"), null=True, blank=True)
    buy_lg_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日大单净流入 净流入率（%）"), null=True, blank=True)
    buy_md_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日中单净流入 净额（元）"), null=True, blank=True)
    buy_md_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日中单净流入 净流入率（%）"), null=True, blank=True)
    buy_sm_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日小单净流入 净额（元）"), null=True, blank=True)
    buy_sm_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日小单净流入 净流入率（%）"), null=True, blank=True)
    buy_sm_amount_stock = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日主力净流入最大股"), null=True, blank=True)
    
    class Meta:
        verbose_name = _("板块资金流向统计")
        verbose_name_plural = _("板块资金流向统计")
        db_table = "fund_flow_cnt_dc"
        unique_together = ['stock', 'trade_time']
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_time']),
        ]
    def __str__(self):
        return f"{self.stock.name if self.stock else ''}板块资金流向统计 - 同花顺({self.trade_time})"
    def __code__(self):
        return self.stock.stock_code if self.stock else ''

class FundFlowIndustryTHS(models.Model):
    """
    行业资金流向统计数据 - 同花顺（fundflow_ind_ths接口）
    """
    stock = models.ForeignKey(
        StockInfo,
        to_field='stock_code',  # 指定外键对应StockInfo的stock_code字段
        db_column='ts_code', # 数据库字段名
        on_delete=models.CASCADE, blank=True, null=True,
        related_name="fund_flow_ind_ths", verbose_name=_("股票")
    )
    trade_time = models.DateField(verbose_name=_("交易日期"), null=True, blank=True)
    industry = models.CharField(max_length=20, verbose_name=_("行业名称"), null=True, blank=True)
    lead_stock = models.CharField(max_length=20, verbose_name=_("领涨股"), null=True, blank=True)
    close = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("收盘价"), null=True, blank=True)
    pct_change = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("涨跌幅(%)"), null=True, blank=True)
    company_num = models.IntegerField(verbose_name=_("公司数量"), null=True, blank=True)
    pct_change_stock = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("领涨股涨跌幅(%)"), null=True, blank=True)
    close_price = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("领涨股收盘价"), null=True, blank=True)
    net_buy_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("流入资金(亿元)"), null=True, blank=True)
    net_sell_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("流出资金(亿元)"), null=True, blank=True)
    net_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("净额(亿元)"), null=True, blank=True)

    class Meta:
        verbose_name = _("行业资金流向统计")
        verbose_name_plural = _("行业资金流向统计")
        db_table = "fund_flow_industry_ths"
        unique_together = ['stock', 'trade_time']
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_time']),
        ]
    def __str__(self):
        return f"{self.stock.name if self.stock else ''}行业资金流向统计 - 同花顺({self.trade_time})"
    def __code__(self):
        return self.stock.stock_code if self.stock else ''

class FundFlowMarketDc(models.Model):
    """
    市场资金流向统计数据 - 东方财富（moneyflow_mkt_dc接口）
    """
    stock = models.ForeignKey(
        StockInfo,
        to_field='stock_code',  # 指定外键对应StockInfo的stock_code字段
        db_column='ts_code', # 数据库字段名
        on_delete=models.CASCADE, blank=True, null=True,
        related_name="fund_flow_market_dc", verbose_name=_("股票")
    )
    trade_time = models.DateField(verbose_name=_("交易日期"), null=True, blank=True)
    close_sh = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("上证指数"), null=True, blank=True)
    pct_change_sh = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("上证指数涨跌幅(%)"), null=True, blank=True)
    close_sz = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("深证成指"), null=True, blank=True)
    pct_change_sz = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("深证成指涨跌幅(%)"), null=True, blank=True)
    net_buy_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日主力净流入 净额（元）"), null=True, blank=True)
    net_buy_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日主力净流入 净流入率（%）"), null=True, blank=True)
    buy_elg_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日超大单净流入 净额（元）"), null=True, blank=True)
    buy_elg_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日超大单净流入 净流入率（%）"), null=True, blank=True)
    buy_lg_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日大单净流入 净额（元）"), null=True, blank=True)
    buy_lg_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日大单净流入 净流入率（%）"), null=True, blank=True)
    buy_md_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日中单净流入 净额（元）"), null=True, blank=True)
    buy_md_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日中单净流入 净流入率（%）"), null=True, blank=True)
    buy_sm_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日小单净流入 净额（元）"), null=True, blank=True)
    buy_sm_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日小单净流入 净流入率（%）"), null=True, blank=True)

    class Meta:
        verbose_name = _("市场资金流向统计")
        verbose_name_plural = _("市场资金流向统计")
        db_table = "fund_flow_market_dc"
        unique_together = ['stock', 'trade_time']
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_time']),
        ]

    def __str__(self):
        return f"{self.stock.name if self.stock else ''}市场资金流向统计 - 东方财富({self.trade_time})"

    def __code__(self):
        return self.stock.stock_code if self.stock else ''



