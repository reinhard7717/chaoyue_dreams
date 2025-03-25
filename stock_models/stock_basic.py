# stock_models/models/stock_basic.py

from django.db import models

class StockBasic(models.Model):
    """股票基础信息模型"""
    stock_code = models.CharField(max_length=10, verbose_name='股票代码', primary_key=True)  # 原 dm
    stock_name = models.CharField(max_length=50, verbose_name='股票名称')  # 原 mc
    exchange = models.CharField(max_length=10, verbose_name='交易所')  # 原 jys
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')
    
    class Meta:
        verbose_name = '股票基础信息'
        verbose_name_plural = verbose_name
        db_table = 'stock_basic'
        ordering = ['stock_code']
    
    def __str__(self):
        return f"{self.stock_code}-{self.stock_name}"


class NewStockCalendar(models.Model):
    """新股日历模型"""
    stock_code = models.CharField(max_length=10, verbose_name='股票代码', primary_key=True)  # 原 zqdm
    stock_short_name = models.CharField(max_length=50, verbose_name='股票简称')  # 原 zqjc
    subscription_code = models.CharField(max_length=10, verbose_name='申购代码')  # 原 sgdm
    issue_total_shares = models.BigIntegerField(verbose_name='发行总数（股）', null=True)  # 原 fxsl
    online_issue_shares = models.BigIntegerField(verbose_name='网上发行（股）', null=True)  # 原 swfxsl
    subscription_limit = models.BigIntegerField(verbose_name='申购上限（股）', null=True)  # 原 sgsx
    max_subscription_value = models.BigIntegerField(verbose_name='顶格申购需配市值(元)', null=True)  # 原 dgsz
    subscription_date = models.DateField(verbose_name='申购日期', null=True)  # 原 sgrq
    issue_price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='发行价格（元）', null=True)  # 原 fxjg
    latest_price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='最新价（元）', null=True)  # 原 zxj
    first_day_close_price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='首日收盘价（元）', null=True)  # 原 srspj
    winning_announcement_date = models.DateField(verbose_name='中签号公布日', null=True)  # 原 zqgbrq
    winning_payment_date = models.DateField(verbose_name='中签缴款日', null=True)  # 原 zqjkrq
    listing_date = models.DateField(verbose_name='上市日期', null=True)  # 原 ssrq
    issue_pe_ratio = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='发行市盈率', null=True)  # 原 syl
    industry_pe_ratio = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='行业市盈率', null=True)  # 原 hysyl
    winning_rate = models.DecimalField(max_digits=10, decimal_places=8, verbose_name='中签率（%）', null=True)  # 原 wszql
    consecutive_limit_boards = models.IntegerField(verbose_name='连续一字板数量', null=True)  # 原 yzbsl
    price_increase_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='涨幅（%）', null=True)  # 原 zf
    profit_per_winning = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='每中一签获利（元）', null=True)  # 原 yqhl
    main_business = models.TextField(verbose_name='主营业务', null=True)  # 原 zyyw
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')
    
    class Meta:
        verbose_name = '新股日历'
        verbose_name_plural = verbose_name
        db_table = 'new_stock_calendar'
        ordering = ['-subscription_date']
    
    def __str__(self):
        return f"{self.stock_code}-{self.stock_short_name}"


class STStockList(models.Model):
    """风险警示股票列表模型"""
    stock_code = models.CharField(max_length=10, verbose_name='股票代码', primary_key=True)  # 原 dm
    stock_name = models.CharField(max_length=50, verbose_name='股票名称')  # 原 mc
    exchange = models.CharField(max_length=10, verbose_name='交易所')  # 原 jys
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')
    
    class Meta:
        verbose_name = '风险警示股票列表'
        verbose_name_plural = verbose_name
        db_table = 'st_stock_list'
        ordering = ['stock_code']
    
    def __str__(self):
        return f"{self.stock_code}-{self.stock_name}"


class CompanyInfo(models.Model):
    """公司简介模型"""
    stock_code = models.CharField(max_length=10, verbose_name='股票代码', primary_key=True)
    company_name = models.CharField(max_length=100, verbose_name='公司名称')  # 原 name
    company_english_name = models.CharField(max_length=200, verbose_name='公司英文名称', null=True, blank=True)  # 原 ename
    market = models.CharField(max_length=50, verbose_name='上市市场')
    concepts = models.TextField(verbose_name='概念及板块', null=True, blank=True)  # 原 idea
    listing_date = models.DateField(verbose_name='上市日期', null=True)  # 原 ldate
    issue_price = models.CharField(max_length=20, verbose_name='发行价格（元）', null=True, blank=True)  # 原 sprice
    lead_underwriter = models.CharField(max_length=100, verbose_name='主承销商', null=True, blank=True)  # 原 principal
    establishment_date = models.CharField(max_length=20, verbose_name='成立日期', null=True, blank=True)  # 原 rdate
    registered_capital = models.CharField(max_length=50, verbose_name='注册资本', null=True, blank=True)  # 原 rprice
    institution_type = models.CharField(max_length=50, verbose_name='机构类型', null=True, blank=True)  # 原 instype
    organization_form = models.CharField(max_length=50, verbose_name='组织形式', null=True, blank=True)  # 原 organ
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')
    
    class Meta:
        verbose_name = '公司简介'
        verbose_name_plural = verbose_name
        db_table = 'company_info'
    
    def __str__(self):
        return f"{self.stock_code}-{self.company_name}"

class CompanyIndex(models.Model):
    """所属指数模型"""
    id = models.BigAutoField(primary_key=True)
    stock_code = models.CharField(max_length=10, verbose_name='股票代码')
    index_name = models.CharField(max_length=100, verbose_name='指数名称')  # 原 mc
    index_code = models.CharField(max_length=20, verbose_name='指数代码')  # 原 dm
    entry_date = models.DateField(verbose_name='进入日期', null=True, blank=True)  # 原 ind
    exit_date = models.DateField(verbose_name='退出日期', null=True, blank=True)  # 原 outd
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')
    
    class Meta:
        verbose_name = '所属指数'
        verbose_name_plural = verbose_name
        db_table = 'company_index'
        unique_together = ('stock_code', 'index_code')
    
    def __str__(self):
        return f"{self.stock_code}-{self.index_name}"

class QuarterlyProfit(models.Model):
    """季度利润模型"""
    id = models.BigAutoField(primary_key=True)
    stock_code = models.CharField(max_length=10, verbose_name='股票代码')
    report_date = models.DateField(verbose_name='截止日期')  # 原 date
    operating_revenue = models.CharField(max_length=50, verbose_name='营业收入（万元）')  # 原 income
    operating_expense = models.CharField(max_length=50, verbose_name='营业支出（万元）')  # 原 expend
    operating_profit = models.CharField(max_length=50, verbose_name='营业利润（万元）')  # 原 profit
    total_profit = models.CharField(max_length=50, verbose_name='利润总额（万元）')  # 原 totalp
    net_profit = models.CharField(max_length=50, verbose_name='净利润（万元）')  # 原 reprofit
    basic_earnings_per_share = models.CharField(max_length=20, verbose_name='基本每股收益(元/股)')  # 原 basege
    diluted_earnings_per_share = models.CharField(max_length=20, verbose_name='稀释每股收益(元/股)')  # 原 ettege
    other_comprehensive_income = models.CharField(max_length=50, verbose_name='其他综合收益（万元）', null=True, blank=True)  # 原 otherp
    total_comprehensive_income = models.CharField(max_length=50, verbose_name='综合收益总额（万元）')  # 原 totalcp
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')
    
    class Meta:
        verbose_name = '季度利润'
        verbose_name_plural = verbose_name
        db_table = 'quarterly_profit'
        unique_together = ('stock_code', 'report_date')
        ordering = ['-report_date']
    
    def __str__(self):
        return f"{self.stock_code}-{self.report_date}"
