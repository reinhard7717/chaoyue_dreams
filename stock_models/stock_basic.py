# stock_models/models/stock_basic.py

from django.db import models
import pandas as pd
from django.utils.translation import gettext_lazy as _
from bulk_update_or_create import BulkUpdateOrCreateQuerySet
from stock_models.index import IndexInfo

class StockInfo(models.Model):
    """股票基础信息模型"""
    stock_code = models.CharField(max_length=10, verbose_name='股票代码', primary_key=True)  # 原 dm
    stock_name = models.CharField(max_length=50, verbose_name='股票名称', null=True, blank=True)  # 原 mc
    exchange = models.CharField(max_length=10, verbose_name='交易所', null=True, blank=True)  # 原 jys
    
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

class NewStockCalendar(models.Model):
    """新股日历模型"""
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, blank=True, null=True, related_name="new_stock_calendar", verbose_name=_("股票"))
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
        return f"{self.stock.stock_code}-{self.stock_short_name}"

    def __code__(self):
        return self.stock.stock_code

class STStockList(models.Model):
    """风险警示股票列表模型"""
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, blank=True, null=True, related_name="st_stock_list", verbose_name=_("股票"))
    stock_name = models.CharField(max_length=50, verbose_name='股票名称')  # 原 mc
    exchange = models.CharField(max_length=10, verbose_name='交易所')  # 原 jys
    
    class Meta:
        verbose_name = '风险警示股票列表'
        verbose_name_plural = verbose_name
        db_table = 'st_stock_list'
        ordering = ['stock']
    
    def __str__(self):
        return f"{self.stock.stock_code}-{self.stock_name}"

    def __code__(self):
        return self.stock.stock_code

class CompanyInfo(models.Model):
    """公司简介模型"""
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, blank=True, null=True, related_name="company_info", verbose_name=_("股票"))
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
    
    class Meta:
        verbose_name = '公司简介'
        verbose_name_plural = verbose_name
        db_table = 'company_info'
    
    def __str__(self):
        return f"{self.stock.stock_code}-{self.company_name}"

    def __code__(self):
        return self.stock.stock_code

class StockBelongsIndex(models.Model):
    """所属指数模型"""
    id = models.BigAutoField(primary_key=True, auto_created=True)
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, blank=True, null=True, related_name="belongs_index", verbose_name=_("股票"))
    index_name = models.CharField(max_length=100, verbose_name='指数名称')  # 原 mc
    index = models.ForeignKey(IndexInfo, on_delete=models.CASCADE, related_name="belongs_index", verbose_name=_("股票指数"))
    entry_date = models.DateField(verbose_name='进入日期', null=True, blank=True)  # 原 ind
    exit_date = models.DateField(verbose_name='退出日期', null=True, blank=True)  # 原 outd
    
    class Meta:
        verbose_name = '股票所属指数'
        verbose_name_plural = verbose_name
        db_table = 'stock_belongs_index'
        unique_together = ('stock', 'index')
    
    def __str__(self):
        return f"{self.stock.stock_code}-{self.index.name}"

    def __code__(self):
        return self.stock.stock_code

class QuarterlyProfit(models.Model):
    """季度利润模型"""
    id = models.BigAutoField(primary_key=True, auto_created=True)
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, blank=True, null=True, related_name="quarterly_profit", verbose_name=_("股票"))
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
    
    class Meta:
        verbose_name = '季度利润'
        verbose_name_plural = verbose_name
        db_table = 'quarterly_profit'
        unique_together = ('stock', 'report_date')
        ordering = ['-report_date']
    
    def __str__(self):
        return f"{self.stock.stock_code}-{self.report_date}"

    def __code__(self):
        return self.stock.stock_code

class MarketCategory(models.Model):
    """市场分类树模型，用于存储指数、行业、概念等分类信息"""
    name = models.CharField(_('名称'), max_length=100, db_index=True)
    code = models.CharField(_('代码'), max_length=50, unique=True, db_index=True)
    type1 = models.IntegerField(_('一级分类'), null=True, blank=True)
    type2 = models.IntegerField(_('二级分类'), null=True, blank=True)
    level = models.IntegerField(_('层级'), help_text=_('从0开始，根节点为0，二级节点为1，以此类推'), null=True, blank=True)
    pcode = models.CharField(_('父节点代码'), max_length=50, blank=True, null=True, db_index=True)
    pname = models.CharField(_('父节点名称'), max_length=100, blank=True, null=True)
    isleaf = models.BooleanField(_('是否为叶子节点'), default=False, help_text=_('0：否，1：是'), null=True, blank=True)
    
    # 添加分类描述字段，用于存储分类的实际文本描述
    type1_name = models.CharField(_('一级分类名称'), max_length=50, blank=True, null=True)
    type2_name = models.CharField(_('二级分类名称'), max_length=50, blank=True, null=True)
    
    class Meta:
        verbose_name = _('市场分类')
        verbose_name_plural = _('市场分类')
        db_table = 'market_category'
        indexes = [
            models.Index(fields=['type1', 'type2']),
            models.Index(fields=['level', 'isleaf']),
        ]
        ordering = ['type1', 'type2', 'level', 'code']
    
    def __str__(self):
        return f"{self.name} ({self.code})"
    
    def get_parent(self):
        """获取父节点"""
        if not self.pcode:
            return None
        return MarketCategory.objects.filter(code=self.pcode).first()
    
    def get_children(self):
        """获取子节点"""
        return MarketCategory.objects.filter(pcode=self.code)
    
    def is_root(self):
        """判断是否为根节点"""
        return self.level == 0
    
    @property
    def type1_display(self):
        """获取一级分类显示名称"""
        return self.type1_name or f"类型{self.type1}"
    
    @property
    def type2_display(self):
        """获取二级分类显示名称"""
        return self.type2_name or f"类型{self.type2}"

class StockTimeTrade(models.Model):
    """
    分时交易数据模型
    """
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, blank=True, null=True, related_name="time_trade", verbose_name=_("股票"))
    time_level = models.CharField(max_length=10, verbose_name='分时级别')
    trade_time = models.DateTimeField(verbose_name='交易时间')
    open_price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='开盘价', null=True)
    high_price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='最高价', null=True)
    low_price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='最低价', null=True)
    close_price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='收盘价', null=True)
    volume = models.BigIntegerField(verbose_name='成交量', null=True)
    turnover = models.DecimalField(max_digits=20, decimal_places=2, verbose_name='成交额', null=True)
    amplitude = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='振幅', null=True)
    turnover_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='换手率', null=True)
    price_change_percent = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='涨跌幅', null=True)
    price_change_amount = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='涨跌额', null=True)
    
    class Meta:
        verbose_name = '分时交易数据'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade'
        unique_together = ('stock', 'time_level', 'trade_time')
        ordering = ['stock', 'time_level', 'trade_time']
    
    def __str__(self):
        return f"{self.stock.stock_code}-{self.time_level}-{self.trade_time}"
    
    def __code__(self):
        return self.stock.stock_code
    
    @classmethod
    def generate_higher_level_data(cls, stock, source_level='5', target_levels=['15', '30', '60']):
        """
        根据源级别数据合成更高级别的数据。
        
        参数:
        - stock: StockInfo 对象的实例，表示要处理的股票。
        - source_level: 字符串，源数据的时间级别，默认为 '5'。
        - target_levels: 列表，目标时间级别列表，默认为 ['15', '30', '60']。
        
        该方法会查询源级别的数据，使用Pandas聚合后保存到数据库，并计算 turnover_rate、price_change_percent 和 price_change_amount。
        """
        if not hasattr(stock, 'circulating_shares') or stock.circulating_shares is None:
            print(f"错误: 股票 {stock.stock_code} 缺少流通股本数据，无法计算 turnover_rate。")
            return  # 缺少必要数据，停止执行
        # 步骤1: 查询源级别的数据
        source_data = cls.objects.filter(stock=stock, time_level=source_level).order_by('trade_time')
        if not source_data.exists():
            print(f"警告: 股票 {stock.stock_code} 的 {source_level} 级别数据不存在。")
            return  # 没有数据可处理
        # 步骤2: 转换为Pandas DataFrame
        data_list = list(source_data.values('trade_time', 'open_price', 'high_price', 'low_price', 'close_price', 'volume', 'turnover'))
        df = pd.DataFrame(data_list)
        if df.empty:
            return  # 空数据
        # 转换 trade_time 为 datetime 类型并设为索引
        df['trade_time'] = pd.to_datetime(df['trade_time'])
        df.set_index('trade_time', inplace=True)
        # 确保数据类型正确
        for col in ['open_price', 'high_price', 'low_price', 'close_price', 'turnover']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        # 步骤3: 根据目标级别进行聚合
        for target_level in target_levels:
            if target_level == '15':
                resample_rule = '15T'  # 15分钟
            elif target_level == '30':
                resample_rule = '30T'  # 30分钟
            elif target_level == '60':
                resample_rule = '60T'  # 60分钟
            else:
                print(f"警告: 不支持的目标级别 {target_level}，跳过。")
                continue
            # 进行重采样和聚合
            aggregated_df = df.resample(resample_rule).agg({
                'open_price': 'first',  # 第一个周期的开盘价
                'high_price': 'max',    # 最高价
                'low_price': 'min',     # 最低价
                'close_price': 'last',  # 最后一个周期的收盘价
                'volume': 'sum',        # 总成交量
                'turnover': 'sum',      # 总成交额
            }).dropna()  # 去除NaN行
            # 计算额外字段
            if not aggregated_df.empty:
                # 计算 turnover_rate
                aggregated_df['turnover_rate'] = (aggregated_df['volume'] / stock.circulating_shares) * 100  # 换手率
                aggregated_df['amplitude'] = (aggregated_df['high_price'] - aggregated_df['low_price']) / aggregated_df['open_price'] * 100
                # 计算 price_change_percent 和 price_change_amount
                aggregated_df['previous_close_price'] = aggregated_df['close_price'].shift(1)  # 获取前一个周期的收盘价
                aggregated_df['price_change_amount'] = aggregated_df['close_price'] - aggregated_df['previous_close_price']  # 涨跌额
                aggregated_df['price_change_percent'] = (aggregated_df['price_change_amount'] / aggregated_df['previous_close_price']) * 100  # 涨跌幅
                # 去除临时列
                aggregated_df = aggregated_df.drop(columns=['previous_close_price'], errors='ignore')
            # 步骤4: 保存聚合后的数据
            for idx, row in aggregated_df.iterrows():
                new_instance = cls(
                    stock=stock,
                    time_level=target_level,  # 使用 '15'、'30' 或 '60'
                    trade_time=idx.to_pydatetime(),
                    open_price=row['open_price'] if 'open_price' in row else None,
                    high_price=row['high_price'] if 'high_price' in row else None,
                    low_price=row['low_price'] if 'low_price' in row else None,
                    close_price=row['close_price'] if 'close_price' in row else None,
                    volume=row['volume'] if 'volume' in row else None,
                    turnover=row['turnover'] if 'turnover' in row else None,
                    turnover_rate=row['turnover_rate'] if 'turnover_rate' in row else None,
                    price_change_percent=row['price_change_percent'] if 'price_change_percent' in row else None,
                    price_change_amount=row['price_change_amount'] if 'price_change_amount' in row else None,
                    amplitude=row['amplitude'] if 'amplitude' in row else None, 
                )
                new_instance.save()  # 保存到数据库
            print(f"成功为股票 {stock.stock_code} 生成 {target_level} 级别数据，包括计算字段。")
        print("数据合成完成。")


# 创建分类查找辅助类
class CategoryTypeManager:
    """分类类型管理器，用于动态获取和缓存分类类型"""
    
    @classmethod
    def get_type1_mapping(cls):
        """获取一级分类映射字典"""
        from django.core.cache import cache
        
        # 尝试从缓存获取
        mapping = cache.get('market_category_type1_mapping')
        if mapping is None:
            # 缓存未命中，从数据库重建
            mapping = {}
            type1_values = MarketCategory.objects.values('type1', 'type1_name').distinct()
            for item in type1_values:
                if item['type1_name']:
                    mapping[item['type1']] = item['type1_name']
            
            # 存入缓存，有效期1天
            cache.set('market_category_type1_mapping', mapping, 86400)
        
        return mapping
    
    @classmethod
    def get_type2_mapping(cls):
        """获取二级分类映射字典"""
        from django.core.cache import cache
        
        # 尝试从缓存获取
        mapping = cache.get('market_category_type2_mapping')
        if mapping is None:
            # 缓存未命中，从数据库重建
            mapping = {}
            type2_values = MarketCategory.objects.values('type2', 'type2_name').distinct()
            for item in type2_values:
                if item['type2_name']:
                    mapping[item['type2']] = item['type2_name']
            
            # 存入缓存，有效期1天
            cache.set('market_category_type2_mapping', mapping, 86400)
        
        return mapping
    
    @classmethod
    def update_category_type_names(cls, api_data_list):
        """根据API数据更新分类名称"""
        # 临时存储解析出的分类映射
        type1_mapping = {}
        type2_mapping = {}
        
        # 解析数据中的分类名称
        for item in api_data_list:
            # 分析一级分类名称
            if 'type1' in item and item.get('name'):
                name_parts = item['name'].split('-', 1)
                if len(name_parts) > 0:
                    type1_mapping[item['type1']] = name_parts[0]
            
            # 分析二级分类名称
            if 'type2' in item and item.get('name'):
                type2_mapping[item['type2']] = item['name']
        
        # 更新数据库中的记录
        for type1, name in type1_mapping.items():
            MarketCategory.objects.filter(type1=type1, type1_name='').update(type1_name=name)
        
        for type2, name in type2_mapping.items():
            MarketCategory.objects.filter(type2=type2, type2_name='').update(type2_name=name)
        
        # 清除缓存，以便下次查询时重建
        from django.core.cache import cache
        cache.delete('market_category_type1_mapping')
        cache.delete('market_category_type2_mapping')
