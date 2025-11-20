# stock_models\fund_flow.py
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


