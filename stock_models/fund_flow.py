from django.db import models
from django.utils.translation import gettext_lazy as _
from stock_models.industry import DcIndex, ThsIndex
from stock_models.stock_basic import StockInfo

# 日级资金流向数据（moneyflow接口）
class FundFlowDailyCY(models.Model):
    """
    日级资金流向数据（moneyflow接口）
    """
    stock = models.ForeignKey(
        StockInfo,
        to_field='stock_code',  # 指定外键对应StockInfo的stock_code字段
        db_column='ts_code', # 数据库字段名
        on_delete=models.CASCADE, blank=True, null=True,
        related_name="fund_flow_daily_cy", verbose_name=_("股票")
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
    trade_count = models.IntegerField(verbose_name='交易笔数', blank=True, null=True)
    class Meta:
        verbose_name = _("日级资金流向")
        verbose_name_plural = _("日级资金流向")
        db_table = "fund_flow_daily_cy"
        unique_together = ['stock', 'trade_time']
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_time']),
        ]

    def __str__(self):
        return f"{self.stock.name if self.stock else ''}日级资金流向({self.trade_time})"

    def __code__(self):
        return self.stock.stock_code if self.stock else ''

class FundFlowDailySZ(models.Model):
    """
    日级资金流向数据（moneyflow接口）
    """
    stock = models.ForeignKey(
        StockInfo,
        to_field='stock_code',  # 指定外键对应StockInfo的stock_code字段
        db_column='ts_code', # 数据库字段名
        on_delete=models.CASCADE, blank=True, null=True,
        related_name="fund_flow_daily_sz", verbose_name=_("股票")
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
    trade_count = models.IntegerField(verbose_name='交易笔数', blank=True, null=True)
    class Meta:
        verbose_name = _("日级资金流向")
        verbose_name_plural = _("日级资金流向")
        db_table = "fund_flow_daily_sz"
        unique_together = ['stock', 'trade_time']
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_time']),
        ]

    def __str__(self):
        return f"{self.stock.name if self.stock else ''}日级资金流向({self.trade_time})"

    def __code__(self):
        return self.stock.stock_code if self.stock else ''

class FundFlowDailyKC(models.Model):
    """
    日级资金流向数据（moneyflow接口）
    """
    stock = models.ForeignKey(
        StockInfo,
        to_field='stock_code',  # 指定外键对应StockInfo的stock_code字段
        db_column='ts_code', # 数据库字段名
        on_delete=models.CASCADE, blank=True, null=True,
        related_name="fund_flow_daily_kc", verbose_name=_("股票")
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
    trade_count = models.IntegerField(verbose_name='交易笔数', blank=True, null=True)
    class Meta:
        verbose_name = _("日级资金流向")
        verbose_name_plural = _("日级资金流向")
        db_table = "fund_flow_daily_kc"
        unique_together = ['stock', 'trade_time']
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_time']),
        ]

    def __str__(self):
        return f"{self.stock.name if self.stock else ''}日级资金流向({self.trade_time})"

    def __code__(self):
        return self.stock.stock_code if self.stock else ''

class FundFlowDailySH(models.Model):
    """
    日级资金流向数据（moneyflow接口）
    """
    stock = models.ForeignKey(
        StockInfo,
        to_field='stock_code',  # 指定外键对应StockInfo的stock_code字段
        db_column='ts_code', # 数据库字段名
        on_delete=models.CASCADE, blank=True, null=True,
        related_name="fund_flow_daily_sh", verbose_name=_("股票")
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
    trade_count = models.IntegerField(verbose_name='交易笔数', blank=True, null=True)
    class Meta:
        verbose_name = _("日级资金流向")
        verbose_name_plural = _("日级资金流向")
        db_table = "fund_flow_daily_sh"
        unique_together = ['stock', 'trade_time']
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_time']),
        ]

    def __str__(self):
        return f"{self.stock.name if self.stock else ''}日级资金流向({self.trade_time})"

    def __code__(self):
        return self.stock.stock_code if self.stock else ''

class FundFlowDailyBJ(models.Model):
    """
    日级资金流向数据（moneyflow接口）
    """
    stock = models.ForeignKey(
        StockInfo,
        to_field='stock_code',  # 指定外键对应StockInfo的stock_code字段
        db_column='ts_code', # 数据库字段名
        on_delete=models.CASCADE, blank=True, null=True,
        related_name="fund_flow_daily_bj", verbose_name=_("股票")
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
    trade_count = models.IntegerField(verbose_name='交易笔数', blank=True, null=True)
    class Meta:
        verbose_name = _("日级资金流向")
        verbose_name_plural = _("日级资金流向")
        db_table = "fund_flow_daily_bj"
        unique_together = ['stock', 'trade_time']
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_time']),
        ]

    def __str__(self):
        return f"{self.stock.name if self.stock else ''}日级资金流向({self.trade_time})"

    def __code__(self):
        return self.stock.stock_code if self.stock else ''

# 日级资金流向数据 - 同花顺（moneyflow_ths接口）
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

class FundFlowDailyTHS_CY(models.Model):
    """
    日级资金流向数据 - 同花顺（moneyflow_ths接口）
    """
    stock = models.ForeignKey(
        StockInfo,
        to_field='stock_code',  # 指定外键对应StockInfo的stock_code字段
        db_column='ts_code', # 数据库字段名
        on_delete=models.CASCADE, blank=True, null=True,
        related_name="fund_flow_daily_ths_cy", verbose_name=_("股票")
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
        db_table = "fund_flow_daily_ths_cy"
        unique_together = ['stock', 'trade_time']
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_time']),
        ]

    def __str__(self):
        return f"{self.stock.name if self.stock else ''}日级资金流向 - 同花顺({self.trade_time})"

    def __code__(self):
        return self.stock.stock_code if self.stock else ''

class FundFlowDailyTHS_SZ(models.Model):
    """
    日级资金流向数据 - 同花顺（moneyflow_ths接口）
    """
    stock = models.ForeignKey(
        StockInfo,
        to_field='stock_code',  # 指定外键对应StockInfo的stock_code字段
        db_column='ts_code', # 数据库字段名
        on_delete=models.CASCADE, blank=True, null=True,
        related_name="fund_flow_daily_ths_sz", verbose_name=_("股票")
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
        db_table = "fund_flow_daily_ths_sz"
        unique_together = ['stock', 'trade_time']
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_time']),
        ]

    def __str__(self):
        return f"{self.stock.name if self.stock else ''}日级资金流向 - 同花顺({self.trade_time})"

    def __code__(self):
        return self.stock.stock_code if self.stock else ''

class FundFlowDailyTHS_SH(models.Model):
    """
    日级资金流向数据 - 同花顺（moneyflow_ths接口）
    """
    stock = models.ForeignKey(
        StockInfo,
        to_field='stock_code',  # 指定外键对应StockInfo的stock_code字段
        db_column='ts_code', # 数据库字段名
        on_delete=models.CASCADE, blank=True, null=True,
        related_name="fund_flow_daily_ths_sh", verbose_name=_("股票")
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
        db_table = "fund_flow_daily_ths_sh"
        unique_together = ['stock', 'trade_time']
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_time']),
        ]

    def __str__(self):
        return f"{self.stock.name if self.stock else ''}日级资金流向 - 同花顺({self.trade_time})"

    def __code__(self):
        return self.stock.stock_code if self.stock else ''

class FundFlowDailyTHS_KC(models.Model):
    """
    日级资金流向数据 - 同花顺（moneyflow_ths接口）
    """
    stock = models.ForeignKey(
        StockInfo,
        to_field='stock_code',  # 指定外键对应StockInfo的stock_code字段
        db_column='ts_code', # 数据库字段名
        on_delete=models.CASCADE, blank=True, null=True,
        related_name="fund_flow_daily_ths_kc", verbose_name=_("股票")
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
        db_table = "fund_flow_daily_ths_kc"
        unique_together = ['stock', 'trade_time']
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_time']),
        ]

    def __str__(self):
        return f"{self.stock.name if self.stock else ''}日级资金流向 - 同花顺({self.trade_time})"

    def __code__(self):
        return self.stock.stock_code if self.stock else ''

class FundFlowDailyTHS_BJ(models.Model):
    """
    日级资金流向数据 - 同花顺（moneyflow_ths接口）
    """
    stock = models.ForeignKey(
        StockInfo,
        to_field='stock_code',  # 指定外键对应StockInfo的stock_code字段
        db_column='ts_code', # 数据库字段名
        on_delete=models.CASCADE, blank=True, null=True,
        related_name="fund_flow_daily_ths_bj", verbose_name=_("股票")
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
        db_table = "fund_flow_daily_ths_bj"
        unique_together = ['stock', 'trade_time']
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_time']),
        ]

    def __str__(self):
        return f"{self.stock.name if self.stock else ''}日级资金流向 - 同花顺({self.trade_time})"

    def __code__(self):
        return self.stock.stock_code if self.stock else ''

# 日级资金流向数据 - 东方财富（moneyflow_dc接口）
class FundFlowDailyDC(models.Model):
    """
    日级资金流向数据 - 东方财富（moneyflow_dc接口）
    """
    stock = models.ForeignKey(
        StockInfo,
        to_field='stock_code',  # 外键对应StockInfo的stock_code字段
        db_column='ts_code',  # 数据库字段名保持和接口字段一致
        on_delete=models.CASCADE,
        blank=True, null=True,
        related_name="fund_flow_daily_dc",
        verbose_name=_("股票")
    )
    trade_time = models.DateField(verbose_name=_("交易日期"), null=True, blank=True)
    name = models.CharField(max_length=50, verbose_name=_("股票名称"), null=True, blank=True)
    pct_change = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("涨跌幅(%)"), null=True, blank=True)
    close = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("最新价"), null=True, blank=True)
    net_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日主力净流入额（万元）"), null=True, blank=True)
    net_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日主力净流入净占比（%）"), null=True, blank=True)
    buy_elg_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日超大单净流入额（万元）"), null=True, blank=True)
    buy_elg_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日超大单净流入占比（%）"), null=True, blank=True)
    buy_lg_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日大单净流入额（万元）"), null=True, blank=True)
    buy_lg_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日大单净流入占比（%）"), null=True, blank=True)
    buy_md_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日中单净流入额（万元）"), null=True, blank=True)
    buy_md_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日中单净流入占比（%）"), null=True, blank=True)
    buy_sm_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日小单净流入额（万元）"), null=True, blank=True)
    buy_sm_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日小单净流入占比（%）"), null=True, blank=True)

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
        return f"{self.stock.name if self.stock else self.name} 日级资金流向 - 东方财富({self.trade_date})"

    def __code__(self):
        return self.stock.stock_code if self.stock else ''

class FundFlowDailyDC_SZ(models.Model):
    """
    日级资金流向数据 - 东方财富（moneyflow_dc接口）
    """
    stock = models.ForeignKey(
        StockInfo,
        to_field='stock_code',  # 外键对应StockInfo的stock_code字段
        db_column='ts_code',  # 数据库字段名保持和接口字段一致
        on_delete=models.CASCADE,
        blank=True, null=True,
        related_name="fund_flow_daily_dc_sz",
        verbose_name=_("股票")
    )
    trade_time = models.DateField(verbose_name=_("交易日期"), null=True, blank=True)
    name = models.CharField(max_length=50, verbose_name=_("股票名称"), null=True, blank=True)
    pct_change = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("涨跌幅(%)"), null=True, blank=True)
    close = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("最新价"), null=True, blank=True)
    net_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日主力净流入额（万元）"), null=True, blank=True)
    net_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日主力净流入净占比（%）"), null=True, blank=True)
    buy_elg_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日超大单净流入额（万元）"), null=True, blank=True)
    buy_elg_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日超大单净流入占比（%）"), null=True, blank=True)
    buy_lg_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日大单净流入额（万元）"), null=True, blank=True)
    buy_lg_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日大单净流入占比（%）"), null=True, blank=True)
    buy_md_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日中单净流入额（万元）"), null=True, blank=True)
    buy_md_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日中单净流入占比（%）"), null=True, blank=True)
    buy_sm_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日小单净流入额（万元）"), null=True, blank=True)
    buy_sm_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日小单净流入占比（%）"), null=True, blank=True)

    class Meta:
        verbose_name = _("日级资金流向 - 东方财富")
        verbose_name_plural = _("日级资金流向 - 东方财富")
        db_table = "fund_flow_daily_dc_sz"
        unique_together = ['stock', 'trade_time']
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_time']),
        ]

    def __str__(self):
        return f"{self.stock.name if self.stock else self.name} 日级资金流向 - 东方财富({self.trade_date})"

    def __code__(self):
        return self.stock.stock_code if self.stock else ''

class FundFlowDailyDC_CY(models.Model):
    """
    日级资金流向数据 - 东方财富（moneyflow_dc接口）
    """
    stock = models.ForeignKey(
        StockInfo,
        to_field='stock_code',  # 外键对应StockInfo的stock_code字段
        db_column='ts_code',  # 数据库字段名保持和接口字段一致
        on_delete=models.CASCADE,
        blank=True, null=True,
        related_name="fund_flow_daily_dc_cy",
        verbose_name=_("股票")
    )
    trade_time = models.DateField(verbose_name=_("交易日期"), null=True, blank=True)
    name = models.CharField(max_length=50, verbose_name=_("股票名称"), null=True, blank=True)
    pct_change = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("涨跌幅(%)"), null=True, blank=True)
    close = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("最新价"), null=True, blank=True)
    net_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日主力净流入额（万元）"), null=True, blank=True)
    net_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日主力净流入净占比（%）"), null=True, blank=True)
    buy_elg_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日超大单净流入额（万元）"), null=True, blank=True)
    buy_elg_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日超大单净流入占比（%）"), null=True, blank=True)
    buy_lg_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日大单净流入额（万元）"), null=True, blank=True)
    buy_lg_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日大单净流入占比（%）"), null=True, blank=True)
    buy_md_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日中单净流入额（万元）"), null=True, blank=True)
    buy_md_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日中单净流入占比（%）"), null=True, blank=True)
    buy_sm_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日小单净流入额（万元）"), null=True, blank=True)
    buy_sm_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日小单净流入占比（%）"), null=True, blank=True)

    class Meta:
        verbose_name = _("日级资金流向 - 东方财富")
        verbose_name_plural = _("日级资金流向 - 东方财富")
        db_table = "fund_flow_daily_dc_cy"
        unique_together = ['stock', 'trade_time']
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_time']),
        ]

    def __str__(self):
        return f"{self.stock.name if self.stock else self.name} 日级资金流向 - 东方财富({self.trade_date})"

    def __code__(self):
        return self.stock.stock_code if self.stock else ''

class FundFlowDailyDC_SH(models.Model):
    """
    日级资金流向数据 - 东方财富（moneyflow_dc接口）
    """
    stock = models.ForeignKey(
        StockInfo,
        to_field='stock_code',  # 外键对应StockInfo的stock_code字段
        db_column='ts_code',  # 数据库字段名保持和接口字段一致
        on_delete=models.CASCADE,
        blank=True, null=True,
        related_name="fund_flow_daily_dc_sh",
        verbose_name=_("股票")
    )
    trade_time = models.DateField(verbose_name=_("交易日期"), null=True, blank=True)
    name = models.CharField(max_length=50, verbose_name=_("股票名称"), null=True, blank=True)
    pct_change = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("涨跌幅(%)"), null=True, blank=True)
    close = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("最新价"), null=True, blank=True)
    net_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日主力净流入额（万元）"), null=True, blank=True)
    net_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日主力净流入净占比（%）"), null=True, blank=True)
    buy_elg_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日超大单净流入额（万元）"), null=True, blank=True)
    buy_elg_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日超大单净流入占比（%）"), null=True, blank=True)
    buy_lg_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日大单净流入额（万元）"), null=True, blank=True)
    buy_lg_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日大单净流入占比（%）"), null=True, blank=True)
    buy_md_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日中单净流入额（万元）"), null=True, blank=True)
    buy_md_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日中单净流入占比（%）"), null=True, blank=True)
    buy_sm_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日小单净流入额（万元）"), null=True, blank=True)
    buy_sm_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日小单净流入占比（%）"), null=True, blank=True)

    class Meta:
        verbose_name = _("日级资金流向 - 东方财富")
        verbose_name_plural = _("日级资金流向 - 东方财富")
        db_table = "fund_flow_daily_dc_sh"
        unique_together = ['stock', 'trade_time']
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_time']),
        ]

    def __str__(self):
        return f"{self.stock.name if self.stock else self.name} 日级资金流向 - 东方财富({self.trade_date})"

    def __code__(self):
        return self.stock.stock_code if self.stock else ''

class FundFlowDailyDC_KC(models.Model):
    """
    日级资金流向数据 - 东方财富（moneyflow_dc接口）
    """
    stock = models.ForeignKey(
        StockInfo,
        to_field='stock_code',  # 外键对应StockInfo的stock_code字段
        db_column='ts_code',  # 数据库字段名保持和接口字段一致
        on_delete=models.CASCADE,
        blank=True, null=True,
        related_name="fund_flow_daily_dc_kc",
        verbose_name=_("股票")
    )
    trade_time = models.DateField(verbose_name=_("交易日期"), null=True, blank=True)
    name = models.CharField(max_length=50, verbose_name=_("股票名称"), null=True, blank=True)
    pct_change = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("涨跌幅(%)"), null=True, blank=True)
    close = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("最新价"), null=True, blank=True)
    net_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日主力净流入额（万元）"), null=True, blank=True)
    net_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日主力净流入净占比（%）"), null=True, blank=True)
    buy_elg_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日超大单净流入额（万元）"), null=True, blank=True)
    buy_elg_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日超大单净流入占比（%）"), null=True, blank=True)
    buy_lg_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日大单净流入额（万元）"), null=True, blank=True)
    buy_lg_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日大单净流入占比（%）"), null=True, blank=True)
    buy_md_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日中单净流入额（万元）"), null=True, blank=True)
    buy_md_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日中单净流入占比（%）"), null=True, blank=True)
    buy_sm_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日小单净流入额（万元）"), null=True, blank=True)
    buy_sm_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日小单净流入占比（%）"), null=True, blank=True)

    class Meta:
        verbose_name = _("日级资金流向 - 东方财富")
        verbose_name_plural = _("日级资金流向 - 东方财富")
        db_table = "fund_flow_daily_dc_kc"
        unique_together = ['stock', 'trade_time']
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_time']),
        ]

    def __str__(self):
        return f"{self.stock.name if self.stock else self.name} 日级资金流向 - 东方财富({self.trade_date})"

    def __code__(self):
        return self.stock.stock_code if self.stock else ''

class FundFlowDailyDC_BJ(models.Model):
    """
    日级资金流向数据 - 东方财富（moneyflow_dc接口）
    """
    stock = models.ForeignKey(
        StockInfo,
        to_field='stock_code',  # 外键对应StockInfo的stock_code字段
        db_column='ts_code',  # 数据库字段名保持和接口字段一致
        on_delete=models.CASCADE,
        blank=True, null=True,
        related_name="fund_flow_daily_dc_bj",
        verbose_name=_("股票")
    )
    trade_time = models.DateField(verbose_name=_("交易日期"), null=True, blank=True)
    name = models.CharField(max_length=50, verbose_name=_("股票名称"), null=True, blank=True)
    pct_change = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("涨跌幅(%)"), null=True, blank=True)
    close = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("最新价"), null=True, blank=True)
    net_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日主力净流入额（万元）"), null=True, blank=True)
    net_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日主力净流入净占比（%）"), null=True, blank=True)
    buy_elg_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日超大单净流入额（万元）"), null=True, blank=True)
    buy_elg_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日超大单净流入占比（%）"), null=True, blank=True)
    buy_lg_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日大单净流入额（万元）"), null=True, blank=True)
    buy_lg_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日大单净流入占比（%）"), null=True, blank=True)
    buy_md_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日中单净流入额（万元）"), null=True, blank=True)
    buy_md_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日中单净流入占比（%）"), null=True, blank=True)
    buy_sm_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("今日小单净流入额（万元）"), null=True, blank=True)
    buy_sm_amount_rate = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("今日小单净流入占比（%）"), null=True, blank=True)

    class Meta:
        verbose_name = _("日级资金流向 - 东方财富")
        verbose_name_plural = _("日级资金流向 - 东方财富")
        db_table = "fund_flow_daily_dc_bj"
        unique_together = ['stock', 'trade_time']
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_time']),
        ]

    def __str__(self):
        return f"{self.stock.name if self.stock else self.name} 日级资金流向 - 东方财富({self.trade_date})"

    def __code__(self):
        return self.stock.stock_code if self.stock else ''

class BaseAdvancedFundFlowMetrics(models.Model):
    """
    【V4.0 · 指标体系精炼版】
    - 核心优化: 全面审查并排除所有不适用于趋势衍生计算的资金流指标。
    """
    trade_time = models.DateField(verbose_name='交易日期', db_index=True)
    CORE_METRICS = {
        'net_flow_consensus': '共识-资金净流入(万元)',
        'main_force_net_flow_consensus': '共识-主力净流入(万元)',
        'retail_net_flow_consensus': '共识-散户净流入(万元)',
        'net_xl_amount_consensus': '共识-超大单净流入(万元)',
        'net_lg_amount_consensus': '共识-大单净流入(万元)',
        'net_md_amount_consensus': '共识-中单净流入(万元)',
        'net_sh_amount_consensus': '共识-小单净流入(万元)',
        'source_consistency_score': '多源一致性评分',
        'flow_internal_friction_ratio': '资金内部摩擦系数',
        'consensus_calibrated_main_flow': '共识校准主力净流入(万元)',
        'consensus_flow_weighted': '加权共识主力净流入(万元)',
        'cross_source_divergence_std': '多源分歧标准差',
        'flow_divergence_mf_vs_retail': '资金分歧度(主力-散户)',
        'main_force_flow_intensity_ratio': '主力资金流强度比率',
        'main_force_flow_impact_ratio': '主力资金流影响力',
        'trade_granularity_impact': '交易颗粒度影响力',
        'main_force_vs_xl_divergence': '主力与超大单分歧度(万元)',
        'main_force_support_strength': '主力支撑强度',
        'main_force_distribution_pressure': '主力派发压力',
        'retail_capitulation_score': '散户投降分',
        'intraday_execution_alpha': '日内执行Alpha',
        'intraday_volatility': '日内波动率',
        'closing_strength_index': '收盘强度指数',
        'close_vs_vwap_ratio': '收盘价与VWAP偏离度',
        'final_hour_momentum': '尾盘动能',
        'trade_concentration_index': '交易集中度指数',
        'avg_order_value': '平均每笔成交金额(元)',
        'avg_order_value_norm_price': '价格归一化平均订单价值',
        'main_force_conviction_ratio': '主力信念比率',
        'avg_cost_sm_buy': '小单买入均价(PVWAP)', 'avg_cost_sm_sell': '小单卖出均价(PVWAP)',
        'avg_cost_md_buy': '中单买入均价(PVWAP)', 'avg_cost_md_sell': '中单卖出均价(PVWAP)',
        'avg_cost_lg_buy': '大单买入均价(PVWAP)', 'avg_cost_lg_sell': '大单卖出均价(PVWAP)',
        'avg_cost_elg_buy': '特大单买入均价(PVWAP)', 'avg_cost_elg_sell': '特大单卖出均价(PVWAP)',
        'avg_cost_main_buy': '主力买入均价(PVWAP)', 'avg_cost_main_sell': '主力卖出均价(PVWAP)',
        'avg_cost_retail_buy': '散户买入均价(PVWAP)', 'avg_cost_retail_sell': '散户卖出均价(PVWAP)',
        'cost_divergence_mf_vs_retail': '成本分歧度(主力买-散户卖)',
        'cost_weighted_main_flow': '主力成本加权净流入',
        'main_buy_cost_advantage': '主力成本领先度(vs Close)',
        'realized_profit_on_exchange': '已实现利润(T+0置换)',
        'net_position_change_value': '净头寸变动市值',
        'unrealized_pnl_on_net_change': '新增头寸浮动盈亏',
        'pnl_matrix_confidence_score': 'P&L矩阵可信度评分',
        'main_force_intraday_profit': '主力日内盈亏',
        'market_cost_battle': '市场成本博弈差(主力买-散户买)',
        'daily_vwap': '当日成交加权平均价',
        'main_buy_cost_vs_vwap': '主力买入成本 vs VWAP',
        'main_sell_cost_vs_vwap': '主力卖出成本 vs VWAP',
        'vwap_tracking_error': 'VWAP算法跟踪误差',
        'volume_profile_jsd_vs_uniform': '成交量分布均匀度(JS散度)',
        'aggression_index_opening': '开盘进攻性指数',
        'divergence_ts_ths': '分歧度(Tushare-同花顺)',
        'divergence_ts_dc': '分歧度(Tushare-东方财富)',
        'divergence_ths_dc': '分歧度(同花顺-东方财富)',
    }
    # 定义不应计算斜率和加速度的指标完整列表
    SLOPE_ACCEL_EXCLUSIONS = [
        # 结构与质量评估类
        'source_consistency_score', 'flow_internal_friction_ratio', 'cross_source_divergence_std',
        'divergence_ts_ths', 'divergence_ts_dc', 'divergence_ths_dc',
        'pnl_matrix_confidence_score', 'volume_profile_jsd_vs_uniform',
        # 日内事件与瞬时状态类
        'main_force_support_strength', 'main_force_distribution_pressure', 'retail_capitulation_score',
        'intraday_execution_alpha', 'closing_strength_index', 'final_hour_momentum',
        'aggression_index_opening', 'vwap_tracking_error', 'realized_profit_on_exchange',
        # 成本与价格本身 (避免冗余)
        'daily_vwap', 'avg_cost_sm_buy', 'avg_cost_sm_sell', 'avg_cost_md_buy', 'avg_cost_md_sell',
        'avg_cost_lg_buy', 'avg_cost_lg_sell', 'avg_cost_elg_buy', 'avg_cost_elg_sell',
        'avg_cost_main_buy', 'avg_cost_main_sell', 'avg_cost_retail_buy', 'avg_cost_retail_sell',
    ]
    
    for name, verbose in CORE_METRICS.items():
        if 'ratio' in name or 'pressure' in name or 'index' in name or 'cost' in name or 'profit' in name or 'battle' in name or 'advantage' in name or 'impact' in name or 'norm_price' in name or name == 'avg_order_value' or 'vwap' in name or 'error' in name or 'jsd' in name or 'strength' in name or 'score' in name or 'alpha' in name or 'volatility' in name or 'momentum' in name:
            vars()[name] = models.FloatField(verbose_name=verbose, null=True, blank=True)
        else:
            vars()[name] = models.DecimalField(max_digits=20, decimal_places=4, verbose_name=verbose, null=True, blank=True)
    main_force_buy_rate_consensus = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='共识-主力买入率(%)', null=True, blank=True)
    UNIFIED_PERIODS = [1, 5, 13, 21, 55]
    for p in UNIFIED_PERIODS:
        if p > 1:
            sum_cols = [
                'net_flow_consensus', 'main_force_net_flow_consensus', 'retail_net_flow_consensus',
                'net_xl_amount_consensus', 'net_lg_amount_consensus', 'net_md_amount_consensus',
                'net_sh_amount_consensus', 'cost_weighted_main_flow',
                'consensus_calibrated_main_flow',
                'consensus_flow_weighted',
            ]
            for name in sum_cols:
                if name in CORE_METRICS:
                    verbose_name = CORE_METRICS.get(name, name)
                    vars()[f'{name}_sum_{p}d'] = models.DecimalField(max_digits=22, decimal_places=4, verbose_name=f'{verbose_name}{p}日累计', null=True, blank=True)
        for name, verbose in CORE_METRICS.items():
            # 增加判断，跳过对排除列表内指标的斜率和加速度计算
            if name in SLOPE_ACCEL_EXCLUSIONS:
                continue
            
            vars()[f'{name}_slope_{p}d'] = models.FloatField(verbose_name=f'{verbose}{p}日斜率', null=True, blank=True)
        if p > 1:
            sum_slope_cols = [
                'net_flow_consensus', 'main_force_net_flow_consensus', 'retail_net_flow_consensus',
                'net_xl_amount_consensus', 'net_lg_amount_consensus', 'net_md_amount_consensus',
                'net_sh_amount_consensus', 'cost_weighted_main_flow',
                'consensus_calibrated_main_flow',
                'consensus_flow_weighted',
            ]
            for name in sum_slope_cols:
                if name in CORE_METRICS:
                    # 增加判断，跳过对排除列表内指标的斜率和加速度计算
                    if name in SLOPE_ACCEL_EXCLUSIONS:
                        continue
                    
                    verbose_name = CORE_METRICS.get(name, name)
                    vars()[f'{name}_sum_{p}d_slope_{p}d'] = models.FloatField(verbose_name=f'{verbose_name}{p}日累计之{p}日斜率', null=True, blank=True)
        for name, verbose in CORE_METRICS.items():
            # 增加判断，跳过对排除列表内指标的斜率和加速度计算
            if name in SLOPE_ACCEL_EXCLUSIONS:
                continue
            
            vars()[f'{name}_accel_{p}d'] = models.FloatField(verbose_name=f'{verbose}{p}日加速度', null=True, blank=True)
    class Meta:
        abstract = True
        ordering = ['-trade_time']

class AdvancedFundFlowMetrics_SH(BaseAdvancedFundFlowMetrics):
    stock = models.ForeignKey(
        'StockInfo',
        on_delete=models.CASCADE,
        related_name='advanced_fund_flow_metrics_sh',
        verbose_name='股票',
        db_index=True
    )
    class Meta(BaseAdvancedFundFlowMetrics.Meta):
        abstract = False
        verbose_name = '高级资金指标-上海'
        verbose_name_plural = verbose_name
        db_table = 'stock_advanced_fund_flow_metrics_sh'
        unique_together = ('stock', 'trade_time')
        indexes = [
            models.Index(fields=['stock', 'trade_time']),
        ]

class AdvancedFundFlowMetrics_SZ(BaseAdvancedFundFlowMetrics):
    stock = models.ForeignKey(
        'StockInfo',
        on_delete=models.CASCADE,
        related_name='advanced_fund_flow_metrics_sz',
        verbose_name='股票',
        db_index=True
    )
    class Meta(BaseAdvancedFundFlowMetrics.Meta):
        abstract = False
        verbose_name = '高级资金指标-深圳'
        verbose_name_plural = verbose_name
        db_table = 'stock_advanced_fund_flow_metrics_sz'
        unique_together = ('stock', 'trade_time')
        indexes = [
            models.Index(fields=['stock', 'trade_time']),
        ]

class AdvancedFundFlowMetrics_CY(BaseAdvancedFundFlowMetrics):
    stock = models.ForeignKey(
        'StockInfo',
        on_delete=models.CASCADE,
        related_name='advanced_fund_flow_metrics_cy',
        verbose_name='股票',
        db_index=True
    )
    class Meta(BaseAdvancedFundFlowMetrics.Meta):
        abstract = False
        verbose_name = '高级资金指标-创业板'
        verbose_name_plural = verbose_name
        db_table = 'stock_advanced_fund_flow_metrics_cy'
        unique_together = ('stock', 'trade_time')
        indexes = [
            models.Index(fields=['stock', 'trade_time']),
        ]

class AdvancedFundFlowMetrics_KC(BaseAdvancedFundFlowMetrics):
    stock = models.ForeignKey(
        'StockInfo',
        on_delete=models.CASCADE,
        related_name='advanced_fund_flow_metrics_kc',
        verbose_name='股票',
        db_index=True
    )
    class Meta(BaseAdvancedFundFlowMetrics.Meta):
        abstract = False
        verbose_name = '高级资金指标-科创板'
        verbose_name_plural = verbose_name
        db_table = 'stock_advanced_fund_flow_metrics_kc'
        unique_together = ('stock', 'trade_time')
        indexes = [
            models.Index(fields=['stock', 'trade_time']),
        ]

class AdvancedFundFlowMetrics_BJ(BaseAdvancedFundFlowMetrics):
    stock = models.ForeignKey(
        'StockInfo',
        on_delete=models.CASCADE,
        related_name='advanced_fund_flow_metrics_bj',
        verbose_name='股票',
        db_index=True
    )
    class Meta(BaseAdvancedFundFlowMetrics.Meta):
        abstract = False
        verbose_name = '高级资金指标-北京'
        verbose_name_plural = verbose_name
        db_table = 'stock_advanced_fund_flow_metrics_bj'
        unique_together = ('stock', 'trade_time')
        indexes = [
            models.Index(fields=['stock', 'trade_time']),
        ]


# 板块资金流向统计数据 - 同花顺（moneyflow_cnt_ths接口）
class FundFlowCntTHS(models.Model):
    """
    板块资金流向统计数据 - 同花顺（moneyflow_cnt_ths接口）
    """
    ths_index = models.ForeignKey(
        ThsIndex,
        to_field='ts_code',  # 指定外键对应ThsIndex的ts_code字段
        db_column='ths_index_code', # 数据库字段名
        on_delete=models.CASCADE, blank=True, null=True,
        related_name="fund_flow_cnt_ths", verbose_name=_("股票")
    )
    trade_time = models.DateField(verbose_name=_("交易日期"), null=True, blank=True)
    lead_stock = models.CharField(max_length=20, verbose_name=_("领涨股"), null=True, blank=True)
    close_price = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("收盘价"), null=True, blank=True)
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
        unique_together = ['ths_index', 'trade_time']
        indexes = [
            models.Index(fields=['ths_index']),
            models.Index(fields=['trade_time']),
        ]
    def __str__(self):
        return f"{self.ths_index.name if self.ths_index else ''}板块资金流向统计 - 同花顺({self.trade_time})"
    def __code__(self):
        return self.ths_index.ts_code if self.ths_index else ''

# 板块资金流向统计数据 - 东方财富（moneyflow_ind_dc接口）
class FundFlowCntDC(models.Model):
    """
    板块资金流向统计数据 - 东方财富（moneyflow_ind_dc接口）
    """
    dc_index = models.ForeignKey(
        DcIndex,
        to_field='ts_code',  # 指定外键对应DcIndex的ts_code字段
        db_column='dc_index_code', # 数据库字段名
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
        unique_together = ['dc_index', 'trade_time']
        indexes = [
            models.Index(fields=['dc_index']),
            models.Index(fields=['trade_time']),
        ]
    def __str__(self):
        return f"{self.dc_index.name if self.dc_index else ''}板块资金流向统计 - 同花顺({self.trade_time})"
    def __code__(self):
        return self.dc_index.name if self.dc_index else ''

# 行业资金流向统计数据 - 同花顺（moneyflow_ths接口）
class FundFlowIndustryTHS(models.Model):
    """
    行业资金流向统计数据 - 同花顺（fundflow_ind_ths接口）
    """
    ths_index = models.ForeignKey(
        ThsIndex,
        to_field='ts_code',  # 指定外键对应ThsIndex的ts_code字段
        db_column='ths_index_code', # 数据库字段名
        on_delete=models.CASCADE, blank=True, null=True,
        related_name="fund_flow_industry_ths", verbose_name=_("股票")
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
        unique_together = ['ths_index', 'trade_time']
        indexes = [
            models.Index(fields=['ths_index']),
            models.Index(fields=['trade_time']),
        ]
    def __str__(self):
        return f"{self.ths_index.name if self.ths_index else ''}行业资金流向统计 - 同花顺({self.trade_time})"
    def __code__(self):
        return self.ths_index.ts_code if self.ths_index else ''

# 大盘（上证）资金流向统计数据 - 东方财富（moneyflow_dc接口）
class FundFlowMarketDc(models.Model):
    """
    大盘（上证）资金流向统计数据 - 东方财富（moneyflow_mkt_dc接口）
    """
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
        indexes = [
            models.Index(fields=['trade_time']),
        ]

    def __str__(self):
        return f"市场资金流向统计 - 东方财富({self.trade_time})"

    def __code__(self):
        return self.stock.stock_code if self.stock else ''

# 龙虎榜每日交易明细
class TopList(models.Model):
    """龙虎榜每日交易明细"""
    stock = models.ForeignKey(
        'StockInfo',to_field='stock_code',
        db_column='ts_code',on_delete=models.CASCADE,
        verbose_name='股票',related_name='top_lists'
    )
    trade_date = models.DateField(verbose_name='交易日期', db_index=True)
    name = models.CharField(max_length=50, verbose_name='名称')
    close = models.FloatField(verbose_name='收盘价')
    pct_change = models.FloatField(verbose_name='涨跌幅')
    turnover_rate = models.FloatField(verbose_name='换手率')
    amount = models.FloatField(verbose_name='总成交额')
    l_sell = models.FloatField(verbose_name='龙虎榜卖出额')
    l_buy = models.FloatField(verbose_name='龙虎榜买入额')
    l_amount = models.FloatField(verbose_name='龙虎榜成交额')
    net_amount = models.FloatField(verbose_name='龙虎榜净买入额')
    net_rate = models.FloatField(verbose_name='龙虎榜净买额占比')
    amount_rate = models.FloatField(verbose_name='龙虎榜成交额占比')
    float_values = models.FloatField(verbose_name='当日流通市值')
    reason = models.CharField(max_length=200, verbose_name='上榜理由')

    class Meta:
        verbose_name = '龙虎榜每日明细'
        verbose_name_plural = verbose_name
        unique_together = ('trade_date', 'stock', 'reason')  # 防止重复

    def __str__(self):
        return f"{self.trade_date} {self.stock} {self.name}"

# 龙虎榜机构成交明细
class TopInst(models.Model):
    """龙虎榜机构成交明细"""
    stock = models.ForeignKey(
        'StockInfo',to_field='stock_code',
        db_column='ts_code',on_delete=models.CASCADE,
        verbose_name='股票',related_name='top_insts'
    )
    trade_date = models.DateField(verbose_name='交易日期', db_index=True)
    exalter = models.CharField(max_length=100, verbose_name='营业部名称')
    side = models.CharField(max_length=1, verbose_name='买卖类型')  # 0/1
    buy = models.FloatField(verbose_name='买入额')
    buy_rate = models.FloatField(verbose_name='买入占总成交比例')
    sell = models.FloatField(verbose_name='卖出额')
    sell_rate = models.FloatField(verbose_name='卖出占总成交比例')
    net_buy = models.FloatField(verbose_name='净成交额')
    reason = models.CharField(max_length=200, verbose_name='上榜理由')

    class Meta:
        verbose_name = '龙虎榜机构明细'
        verbose_name_plural = verbose_name
        unique_together = ('trade_date', 'stock', 'exalter', 'side', 'reason')

    def __str__(self):
        return f"{self.trade_date} {self.stock} {self.exalter} {self.side}"


