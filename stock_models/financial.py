from django.db import models
from django.utils.translation import gettext_lazy as _

# 利润表(Income)
class Income(models.Model):
    """利润表"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, to_field='stock_code', related_name='incomes', verbose_name='股票')
    ann_date = models.DateField(verbose_name='公告日期')
    f_ann_date = models.DateField(null=True, blank=True, verbose_name='实际公告日期')
    end_date = models.DateField(verbose_name='报告期')
    report_type = models.CharField(max_length=10, verbose_name='报告类型')
    comp_type = models.CharField(max_length=10, verbose_name='公司类型')
    end_type = models.CharField(max_length=10, null=True, blank=True, verbose_name='报告期类型')
    basic_eps = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name='基本每股收益')
    diluted_eps = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name='稀释每股收益')
    total_revenue = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True, verbose_name='营业总收入')
    revenue = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True, verbose_name='营业收入')
    # ... 省略其余字段，按需添加
    update_flag = models.CharField(max_length=10, null=True, blank=True, verbose_name='更新标识')
    class Meta:
        verbose_name = '利润表'
        verbose_name_plural = verbose_name
        db_table = 'income'
        unique_together = ('stock', 'end_date', 'report_type')
        ordering = ['-end_date']
    def __str__(self):
        return f"{self.stock.stock_code} {self.end_date}"

# 资产负债表(BalanceSheet)
class BalanceSheet(models.Model):
    """资产负债表"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, to_field='stock_code', related_name='balance_sheets', verbose_name='股票')
    ann_date = models.DateField(verbose_name='公告日期')
    f_ann_date = models.DateField(null=True, blank=True, verbose_name='实际公告日期')
    end_date = models.DateField(verbose_name='报告期')
    report_type = models.CharField(max_length=10, verbose_name='报表类型')
    comp_type = models.CharField(max_length=10, verbose_name='公司类型')
    end_type = models.CharField(max_length=10, null=True, blank=True, verbose_name='报告期类型')
    total_share = models.BigIntegerField(null=True, blank=True, verbose_name='期末总股本')
    cap_rese = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True, verbose_name='资本公积金')
    # ... 省略其余字段，按需添加
    update_flag = models.CharField(max_length=10, null=True, blank=True, verbose_name='更新标识')
    class Meta:
        verbose_name = '资产负债表'
        verbose_name_plural = verbose_name
        db_table = 'balance_sheet'
        unique_together = ('stock', 'end_date', 'report_type')
        ordering = ['-end_date']
    def __str__(self):
        return f"{self.stock.stock_code} {self.end_date}"

# 现金流量表(CashFlow)
class CashFlow(models.Model):
    """现金流量表"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, to_field='stock_code', related_name='cash_flows', verbose_name='股票')
    ann_date = models.DateField(verbose_name='公告日期')
    f_ann_date = models.DateField(null=True, blank=True, verbose_name='实际公告日期')
    end_date = models.DateField(verbose_name='报告期')
    comp_type = models.CharField(max_length=10, verbose_name='公司类型')
    report_type = models.CharField(max_length=10, verbose_name='报表类型')
    end_type = models.CharField(max_length=10, null=True, blank=True, verbose_name='报告期类型')
    net_profit = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True, verbose_name='净利润')
    finan_exp = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True, verbose_name='财务费用')
    # ... 省略其余字段，按需添加
    update_flag = models.CharField(max_length=10, null=True, blank=True, verbose_name='更新标志')
    class Meta:
        verbose_name = '现金流量表'
        verbose_name_plural = verbose_name
        db_table = 'cash_flow'
        unique_together = ('stock', 'end_date', 'report_type')
        ordering = ['-end_date']
    def __str__(self):
        return f"{self.stock.stock_code} {self.end_date}"

# 业绩预告(Forecast)
class Forecast(models.Model):
    """业绩预告"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, to_field='stock_code', related_name='forecasts', verbose_name='股票')
    ann_date = models.DateField(verbose_name='公告日期')
    end_date = models.DateField(verbose_name='报告期')
    type = models.CharField(max_length=10, verbose_name='业绩预告类型')
    p_change_min = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='净利润变动幅度下限(%)')
    p_change_max = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='净利润变动幅度上限(%)')
    net_profit_min = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True, verbose_name='净利润下限(万元)')
    net_profit_max = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True, verbose_name='净利润上限(万元)')
    last_parent_net = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True, verbose_name='上年同期归母净利润')
    first_ann_date = models.DateField(null=True, blank=True, verbose_name='首次公告日')
    summary = models.TextField(null=True, blank=True, verbose_name='业绩预告摘要')
    change_reason = models.TextField(null=True, blank=True, verbose_name='业绩变动原因')
    class Meta:
        verbose_name = '业绩预告'
        verbose_name_plural = verbose_name
        db_table = 'forecast'
        unique_together = ('stock', 'ann_date', 'end_date')
        ordering = ['-ann_date']
    def __str__(self):
        return f"{self.stock.stock_code} {self.ann_date} {self.type}"

# 业绩快报(Express)
class Express(models.Model):
    """业绩快报"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, to_field='stock_code', related_name='expresses', verbose_name='股票')
    ann_date = models.DateField(verbose_name='公告日期')
    end_date = models.DateField(verbose_name='报告期')
    revenue = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True, verbose_name='营业收入(元)')
    operate_profit = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True, verbose_name='营业利润(元)')
    total_profit = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True, verbose_name='利润总额(元)')
    n_income = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True, verbose_name='净利润(元)')
    total_assets = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True, verbose_name='总资产(元)')
    total_hldr_eqy_exc_min_int = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True, verbose_name='股东权益合计(不含少数)')
    diluted_eps = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name='每股收益(摊薄)')
    diluted_roe = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='净资产收益率(摊薄)(%)')
    perf_summary = models.TextField(null=True, blank=True, verbose_name='业绩简要说明')
    is_audit = models.BooleanField(default=False, verbose_name='是否审计')
    remark = models.TextField(null=True, blank=True, verbose_name='备注')
    class Meta:
        verbose_name = '业绩快报'
        verbose_name_plural = verbose_name
        db_table = 'express'
        unique_together = ('stock', 'ann_date', 'end_date')
        ordering = ['-ann_date']
    def __str__(self):
        return f"{self.stock.stock_code} {self.ann_date}"

# 分红送股(Dividend)
class Dividend(models.Model):
    """分红送股"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, to_field='stock_code', related_name='dividends', verbose_name='股票')
    end_date = models.DateField(verbose_name='分红年度')
    ann_date = models.DateField(verbose_name='预案公告日')
    div_proc = models.CharField(max_length=20, verbose_name='实施进度')
    stk_div = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='每股送转')
    stk_bo_rate = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='每股送股比例')
    stk_co_rate = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='每股转增比例')
    cash_div = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='每股分红(税后)')
    cash_div_tax = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='每股分红(税前)')
    record_date = models.DateField(null=True, blank=True, verbose_name='股权登记日')
    ex_date = models.DateField(null=True, blank=True, verbose_name='除权除息日')
    pay_date = models.DateField(null=True, blank=True, verbose_name='派息日')
    div_listdate = models.DateField(null=True, blank=True, verbose_name='红股上市日')
    imp_ann_date = models.DateField(null=True, blank=True, verbose_name='实施公告日')
    base_date = models.DateField(null=True, blank=True, verbose_name='基准日')
    base_share = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True, verbose_name='基准股本(万)')
    class Meta:
        verbose_name = '分红送股'
        verbose_name_plural = verbose_name
        db_table = 'dividend'
        unique_together = ('stock', 'end_date', 'ann_date')
        ordering = ['-ann_date']
    def __str__(self):
        return f"{self.stock.stock_code} {self.end_date}"

# 财务指标(FinaIndicator)
class FinaIndicator(models.Model):
    """财务指标"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, to_field='stock_code', related_name='fina_indicators', verbose_name='股票')
    ann_date = models.DateField(verbose_name='公告日期')
    end_date = models.DateField(verbose_name='报告期')
    eps = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name='基本每股收益')
    dt_eps = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name='稀释每股收益')
    bps = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name='每股净资产')
    roe = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='净资产收益率')
    netprofit_margin = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='销售净利率')
    grossprofit_margin = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='销售毛利率')
    debt_to_assets = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='资产负债率')
    update_flag = models.CharField(max_length=10, null=True, blank=True, verbose_name='更新标识')
    # ... 其他常用字段可按需补充
    class Meta:
        verbose_name = '财务指标'
        verbose_name_plural = verbose_name
        db_table = 'fina_indicator'
        unique_together = ('stock', 'ann_date', 'end_date')
        ordering = ['-ann_date']
    def __str__(self):
        return f"{self.stock.stock_code} {self.ann_date}"

# 财务审计意见(FinaAudit)
class FinaAudit(models.Model):
    """财务审计意见"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, to_field='stock_code', related_name='fina_audits', verbose_name='股票')
    ann_date = models.DateField(verbose_name='公告日期')
    end_date = models.DateField(verbose_name='报告期')
    audit_result = models.CharField(max_length=50, verbose_name='审计结果')
    audit_fees = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True, verbose_name='审计总费用(元)')
    audit_agency = models.CharField(max_length=100, null=True, blank=True, verbose_name='会计事务所')
    audit_sign = models.CharField(max_length=100, null=True, blank=True, verbose_name='签字会计师')
    class Meta:
        verbose_name = '财务审计意见'
        verbose_name_plural = verbose_name
        db_table = 'fina_audit'
        unique_together = ('stock', 'ann_date', 'end_date')
        ordering = ['-ann_date']
    def __str__(self):
        return f"{self.stock.stock_code} {self.ann_date}"

# 主营业务构成(FinaMainBZ)
class FinaMainBZ(models.Model):
    """主营业务构成"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, to_field='stock_code', related_name='fina_mainbz', verbose_name='股票')
    end_date = models.DateField(verbose_name='报告期')
    bz_item = models.CharField(max_length=100, verbose_name='主营业务来源')
    bz_sales = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True, verbose_name='主营业务收入(元)')
    bz_profit = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True, verbose_name='主营业务利润(元)')
    bz_cost = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True, verbose_name='主营业务成本(元)')
    curr_type = models.CharField(max_length=10, null=True, blank=True, verbose_name='货币代码')
    update_flag = models.CharField(max_length=10, null=True, blank=True, verbose_name='是否更新')
    class Meta:
        verbose_name = '主营业务构成'
        verbose_name_plural = verbose_name
        db_table = 'fina_mainbz'
        unique_together = ('stock', 'end_date', 'bz_item')
        ordering = ['-end_date']
    def __str__(self):
        return f"{self.stock.stock_code} {self.end_date} {self.bz_item}"

# 财报披露计划(DisclosureDate)
class DisclosureDate(models.Model):
    """财报披露计划"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, to_field='stock_code', related_name='disclosure_dates', verbose_name='股票')
    ann_date = models.DateField(verbose_name='最新披露公告日')
    end_date = models.DateField(verbose_name='报告期')
    pre_date = models.DateField(null=True, blank=True, verbose_name='预计披露日期')
    actual_date = models.DateField(null=True, blank=True, verbose_name='实际披露日期')
    modify_date = models.DateField(null=True, blank=True, verbose_name='披露日期修正记录')
    class Meta:
        verbose_name = '财报披露计划'
        verbose_name_plural = verbose_name
        db_table = 'disclosure_date'
        unique_together = ('stock', 'end_date', 'ann_date')
        ordering = ['-ann_date']
    def __str__(self):
        return f"{self.stock.stock_code} {self.end_date}"














