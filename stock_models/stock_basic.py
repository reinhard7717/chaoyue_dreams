# stock_models/models/stock_basic.py

from django.db import models
import pandas as pd
from django.utils.translation import gettext_lazy as _


class StockInfo(models.Model):
    """股票基础信息模型"""
    stock_code = models.CharField(max_length=10, verbose_name='股票代码', primary_key=True)  # 原 dm
    stock_name = models.CharField(max_length=50, verbose_name='股票名称', null=True, blank=True)  # 原 mc
    area = models.CharField(max_length=20, verbose_name='地域', null=True, blank=True)
    industry = models.CharField(max_length=50, verbose_name='所属行业', null=True, blank=True)
    full_name = models.CharField(max_length=100, verbose_name='股票全称', null=True, blank=True)
    en_name = models.CharField(max_length=100, verbose_name='英文全称', null=True, blank=True)
    cn_spell = models.CharField(max_length=20, verbose_name='拼音缩写', null=True, blank=True)
    market_type = models.CharField(max_length=20, verbose_name='市场类型', null=True, blank=True)
    exchange = models.CharField(max_length=10, verbose_name='交易所', null=True, blank=True)  # 原 jys
    currency_type = models.CharField(max_length=10, verbose_name='交易货币', null=True, blank=True)
    list_status = models.CharField(max_length=2, verbose_name='上市状态', null=True, blank=True)
    list_date = models.DateField(verbose_name='上市日期', null=True, blank=True)
    delist_date = models.DateField(verbose_name='退市日期', null=True, blank=True)
    is_hs = models.CharField(max_length=2, verbose_name='是否沪深港通标的', null=True, blank=True)
    actual_controller = models.CharField(max_length=100, verbose_name='实控人名称', null=True, blank=True)
    actual_controller_type = models.CharField(max_length=50, verbose_name='实控人企业性质', null=True, blank=True)
    class Meta:
        verbose_name = '股票基础信息'
        verbose_name_plural = verbose_name
        db_table = 'stock_info'
        ordering = ['stock_code']
        managed = True
    def __str__(self):
        return f"{self.stock_code}-{self.stock_name}"
    def __code__(self):
        return self.stock_code

# 上市公司基本信息(StockCompany)
class StockCompany(models.Model):
    """上市公司基本信息"""
    stock = models.OneToOneField(
        'StockInfo',
        on_delete=models.CASCADE,
        to_field='stock_code',
        related_name='company_info',
        verbose_name='股票'
    )
    com_name = models.CharField(max_length=100, verbose_name='公司全称')
    com_id = models.CharField(max_length=32, null=True, blank=True, verbose_name='统一社会信用代码')
    exchange = models.CharField(max_length=10, verbose_name='交易所代码')
    chairman = models.CharField(max_length=50, null=True, blank=True, verbose_name='法人代表')
    manager = models.CharField(max_length=50, null=True, blank=True, verbose_name='总经理')
    secretary = models.CharField(max_length=50, null=True, blank=True, verbose_name='董秘')
    reg_capital = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True, verbose_name='注册资本(万元)')
    setup_date = models.DateField(null=True, blank=True, verbose_name='注册日期')
    province = models.CharField(max_length=20, null=True, blank=True, verbose_name='所在省份')
    city = models.CharField(max_length=20, null=True, blank=True, verbose_name='所在城市')
    introduction = models.TextField(null=True, blank=True, verbose_name='公司介绍')
    website = models.CharField(max_length=100, null=True, blank=True, verbose_name='公司主页')
    email = models.CharField(max_length=100, null=True, blank=True, verbose_name='电子邮件')
    office = models.CharField(max_length=100, null=True, blank=True, verbose_name='办公室')
    employees = models.IntegerField(null=True, blank=True, verbose_name='员工人数')
    main_business = models.TextField(null=True, blank=True, verbose_name='主要业务及产品')
    business_scope = models.TextField(null=True, blank=True, verbose_name='经营范围')
    class Meta:
        verbose_name = '上市公司基本信息'
        verbose_name_plural = verbose_name
        db_table = 'stock_company'
        ordering = ['stock']
    def __str__(self):
        return f"{self.stock.stock_code} - {self.com_name}"

# 沪深港通成分股(HSConst)
class HSConst(models.Model):
    """沪深港通成分股"""
    stock = models.ForeignKey(
        'StockInfo',
        on_delete=models.CASCADE,
        to_field='stock_code',  # 指定外键对应StockInfo的stock_code字段
        db_column='stock_code', # 数据库字段名
        related_name='hs_constituents',
        verbose_name='股票'
    )
    hs_type = models.CharField(max_length=4, verbose_name='沪深港通类型')  # SH/SZ
    in_date = models.DateField(verbose_name='纳入日期')
    out_date = models.DateField(null=True, blank=True, verbose_name='剔除日期')
    is_new = models.BooleanField(default=True, verbose_name='是否最新')
    class Meta:
        verbose_name = '沪深港通成分'
        verbose_name_plural = verbose_name
        db_table = 'hs_const'
        unique_together = ('stock', 'hs_type', 'in_date')
        ordering = ['-in_date']
    def __str__(self):
        return f"{self.stock.stock_code} - {self.hs_type} - {self.in_date}"











