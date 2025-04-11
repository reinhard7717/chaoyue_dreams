# models/datacenter/institution.py
from django.db import models
from django.utils.translation import gettext_lazy as _
from utils.models import BaseModel
from datetime import datetime

class InstitutionHoldingSummary(BaseModel):
    """机构持股汇总"""
    trade_date = models.DateField(verbose_name="统计日期")  # 原 t
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, blank=True, null=True, related_name="institution_holding_summary", verbose_name=_("股票"))
    institution_count = models.IntegerField(verbose_name="机构持股家数")  # 原 jgcgs
    institution_holding_ratio = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="机构持股占比(%)")  # 原 jgcgzb
    institution_holding_value = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="机构持股市值(万元)")  # 原 jgcgsz
    fund_count = models.IntegerField(verbose_name="基金持股家数")  # 原 jjcgs
    fund_holding_ratio = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="基金持股占比(%)")  # 原 jjcgzb
    fund_holding_value = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="基金持股市值(万元)")  # 原 jjcgsz
    social_security_count = models.IntegerField(verbose_name="社保持股家数")  # 原 sbcgs
    social_security_holding_ratio = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="社保持股占比(%)")  # 原 sbcgzb
    social_security_holding_value = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="社保持股市值(万元)")  # 原 sbcgsz
    qfii_count = models.IntegerField(verbose_name="QFII持股家数")  # 原 qfiicgs
    qfii_holding_ratio = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="QFII持股占比(%)")  # 原 qfiicgzb
    qfii_holding_value = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="QFII持股市值(万元)")  # 原 qfiicgsz
    insurance_count = models.IntegerField(verbose_name="保险持股家数")  # 原 baoxcgs
    insurance_holding_ratio = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="保险持股占比(%)")  # 原 baoxcgzb
    insurance_holding_value = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="保险持股市值(万元)")  # 原 baoxcgsz
    year = models.IntegerField(verbose_name="报告年份")
    quarter = models.IntegerField(verbose_name="报告季度")  # 1:一季报，2：中报，3：三季报，4：年报
    
    class Meta:
        verbose_name = "机构持股汇总"
        db_table = "institution_holding_summary"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_date']),
            models.Index(fields=['year', 'quarter']),
        ]

    def __code__(self):
        return self.stock.stock_code

class FundHeavyPosition(BaseModel):
    """基金重仓"""
    trade_date = models.DateField(verbose_name="统计日期")  # 原 t
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, blank=True, null=True, related_name="fund_heavy_position", verbose_name=_("股票"))
    fund_count = models.IntegerField(verbose_name="持有基金数")  # 原 jjsl
    holding_shares = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="持股数(万股)")  # 原 cgs
    holding_ratio = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="持股比例(%)")  # 原 cgbl
    holding_value = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="持股市值(万元)")  # 原 cgsz
    float_market_value_ratio = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="占流通市值比例(%)")  # 原 cgszbl
    total_share_ratio = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="占总股本比例(%)")  # 原 zltgbl
    net_assets = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="净资产(万元)")  # 原 jzc
    net_profit = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="净利润(万元)")  # 原 jlr
    close_price = models.DecimalField(max_digits=10, decimal_places=3, verbose_name="收盘价")  # 原 c
    change_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="涨跌幅(%)")  # 原 zdf
    year = models.IntegerField(verbose_name="报告年份")
    quarter = models.IntegerField(verbose_name="报告季度")  # 1:一季报，2：中报，3：三季报，4：年报
    
    class Meta:
        verbose_name = "基金重仓"
        db_table = "fund_heavy_position"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_date']),
            models.Index(fields=['year', 'quarter']),
        ]

    def __code__(self):
        return self.stock.stock_code

class SocialSecurityHeavyPosition(BaseModel):
    """社保重仓"""
    trade_date = models.DateField(verbose_name="统计日期")  # 原 t
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, blank=True, null=True, related_name="social_security_heavy_position", verbose_name=_("股票"))
    social_security_count = models.IntegerField(verbose_name="持有社保基金数")  # 原 sbsl
    holding_shares = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="持股数(万股)")  # 原 cgs
    holding_ratio = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="持股比例(%)")  # 原 cgbl
    holding_value = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="持股市值(万元)")  # 原 cgsz
    float_market_value_ratio = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="占流通市值比例(%)")  # 原 cgszbl
    total_share_ratio = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="占总股本比例(%)")  # 原 zltgbl
    net_assets = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="净资产(万元)")  # 原 jzc
    net_profit = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="净利润(万元)")  # 原 jlr
    close_price = models.DecimalField(max_digits=10, decimal_places=3, verbose_name="收盘价")  # 原 c
    change_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="涨跌幅(%)")  # 原 zdf
    year = models.IntegerField(verbose_name="报告年份")
    quarter = models.IntegerField(verbose_name="报告季度")  # 1:一季报，2：中报，3：三季报，4：年报
    
    class Meta:
        verbose_name = "社保重仓"
        db_table = "social_security_heavy_position"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_date']),
            models.Index(fields=['year', 'quarter']),
        ]

    def __code__(self):
        return self.stock.stock_code

class QFIIHeavyPosition(BaseModel):
    """QFII重仓"""
    trade_date = models.DateField(verbose_name="统计日期")  # 原 t
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, blank=True, null=True, related_name="qfii_heavy_position", verbose_name=_("股票"))
    qfii_count = models.IntegerField(verbose_name="持有QFII数")  # 原 qfiis
    holding_shares = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="持股数(万股)")  # 原 cgs
    holding_ratio = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="持股比例(%)")  # 原 cgbl
    holding_value = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="持股市值(万元)")  # 原 cgsz
    float_market_value_ratio = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="占流通市值比例(%)")  # 原 cgszbl
    total_share_ratio = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="占总股本比例(%)")  # 原 zltgbl
    net_assets = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="净资产(万元)")  # 原 jzc
    net_profit = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="净利润(万元)")  # 原 jlr
    close_price = models.DecimalField(max_digits=10, decimal_places=3, verbose_name="收盘价")  # 原 c
    change_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="涨跌幅(%)")  # 原 zdf
    year = models.IntegerField(verbose_name="报告年份")
    quarter = models.IntegerField(verbose_name="报告季度")  # 1:一季报，2：中报，3：三季报，4：年报
    
    class Meta:
        verbose_name = "QFII重仓"
        db_table = "qfii_heavy_position"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_date']),
            models.Index(fields=['year', 'quarter']),
        ]

    def __code__(self):
        return self.stock.stock_code



