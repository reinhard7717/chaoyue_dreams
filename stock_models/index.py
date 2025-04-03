from django.db import models
from django.utils.translation import gettext_lazy as _
from utils.models import BaseModel

class IndexInfo(BaseModel):
    """
    股票指数基本信息模型
    
    用于存储沪深股市指数的基本信息，包括指数代码、名称和所属交易所
    """

    code = models.CharField(max_length=20, unique=True, verbose_name=_("指数代码"))
    name = models.CharField(max_length=50, verbose_name=_("指数名称"))
    exchange = models.CharField(max_length=10, verbose_name=_("交易所代码"))
    
    class Meta:
        verbose_name = _("股票指数")
        verbose_name_plural = _("股票指数")
        db_table = "index_info"
        indexes = [
            models.Index(fields=['code']),
            models.Index(fields=['exchange']),
        ]

    def __str__(self):
        return f"{self.name}({self.code})"
    
    def __code__(self):
        return self.code
class IndexRealTimeData(BaseModel):
    """
    股票指数实时数据模型
    
    用于存储股票指数的实时交易数据，对应指数实时数据接口
    """
    index = models.ForeignKey(IndexInfo, on_delete=models.CASCADE, related_name="realtime_data", verbose_name=_("股票指数"))
    # 指数基本信息
    trade_time = models.DateTimeField(verbose_name="交易时间", null=True, blank=True)
    
    # 价格相关数据
    open_price = models.DecimalField(max_digits=12, decimal_places=2, verbose_name="开盘价", null=True, blank=True) 
    high_price = models.DecimalField(max_digits=12, decimal_places=2, verbose_name="最高价", null=True, blank=True)
    low_price = models.DecimalField(max_digits=12, decimal_places=2, verbose_name="最低价", null=True, blank=True)
    current_price = models.DecimalField(max_digits=12, decimal_places=2, verbose_name="当前价格", null=True, blank=True)
    prev_close_price = models.DecimalField(max_digits=12, decimal_places=2, verbose_name="昨日收盘价", null=True, blank=True)
    
    # 涨跌相关数据
    price_change = models.DecimalField(max_digits=12, decimal_places=2, verbose_name="涨跌额", null=True, blank=True)
    price_change_percent = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="涨跌幅(%)", null=True, blank=True)
    five_minute_change_percent = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="五分钟涨跌幅(%)", null=True, blank=True)
    amplitude = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="振幅(%)", null=True, blank=True)
    change_speed = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="涨速(%)", null=True, blank=True)
    sixty_day_change_percent = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="60日涨跌幅(%)", null=True, blank=True)
    ytd_change_percent = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="年初至今涨跌幅(%)", null=True, blank=True)
    
    # 交易相关数据
    volume = models.BigIntegerField(verbose_name="成交量(手)", null=True, blank=True)
    turnover = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="成交额(元)", null=True, blank=True)
    turnover_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="换手率(%)", null=True, blank=True)
    volume_ratio = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="量比(%)", null=True, blank=True)
    
    # 估值相关数据
    pe_ratio = models.DecimalField(max_digits=12, decimal_places=2, verbose_name="市盈率", null=True, blank=True)
    pb_ratio = models.DecimalField(max_digits=12, decimal_places=2, verbose_name="市净率", null=True, blank=True)
    circulating_market_value = models.DecimalField(max_digits=24, decimal_places=2, verbose_name="流通市值(元)", null=True, blank=True)
    total_market_value = models.DecimalField(max_digits=24, decimal_places=2, verbose_name="总市值(元)", null=True, blank=True)
    
    class Meta:
        verbose_name = "指数实时数据"
        verbose_name_plural = verbose_name
        db_table = "index_realtime_data"
        indexes = [
            models.Index(fields=['index', 'trade_time']),
            models.Index(fields=['trade_time']),
        ]
        ordering = ['-trade_time']
        unique_together = ['index', 'trade_time']
    
    def __str__(self):
        return f"{self.index.name}实时数据({self.trade_time})"

    def __code__(self):
        return self.index.code
    
    @classmethod
    def from_api_response(cls, index, data):
        """从API响应创建或更新模型实例"""
        from django.utils.dateparse import parse_datetime
        
        return cls(
            index=index,
            trade_time=parse_datetime(data['t']),
            open_price=data['o'],
            high_price=data['h'],
            low_price=data['l'],
            current_price=data['p'],
            prev_close_price=data['yc'],
            price_change=data['ud'],
            price_change_percent=data['pc'],
            five_minute_change_percent=data['fm'],
            amplitude=data['zf'],
            change_speed=data['zs'],
            sixty_day_change_percent=data['zdf60'],
            ytd_change_percent=data['zdfnc'],
            volume=data['v'],
            turnover=data['cje'],
            turnover_rate=data['hs'],
            volume_ratio=data['lb'],
            pe_ratio=data['pe'] if data['pe'] != 0 else None,
            pb_ratio=data['sjl'] if data['sjl'] != 0 else None,
            circulating_market_value=data['lt'],
            total_market_value=data['sz'],
        )

class MarketOverview(BaseModel):
    """
    市场概览数据模型
    
    用于存储沪深两市的股票上涨、下跌总数等市场概览数据
    """
    total_up = models.IntegerField(verbose_name=_("上涨总数"), null=True, blank=True)
    total_down = models.IntegerField(verbose_name=_("下跌总数"), null=True, blank=True)
    limit_up = models.IntegerField(verbose_name=_("涨停总数"), null=True, blank=True)
    limit_down = models.IntegerField(verbose_name=_("跌停总数"), null=True, blank=True)
    up_8_to_limit = models.IntegerField(verbose_name=_("上涨8%~涨停数量"), null=True, blank=True)
    up_6_to_8 = models.IntegerField(verbose_name=_("上涨6%~8%数量"), null=True, blank=True)
    up_4_to_6 = models.IntegerField(verbose_name=_("上涨4%~6%数量"), null=True, blank=True)
    up_2_to_4 = models.IntegerField(verbose_name=_("上涨2%~4%数量"), null=True, blank=True)
    up_0_to_2 = models.IntegerField(verbose_name=_("上涨0%~2%数量"), null=True, blank=True)
    down_0_to_2 = models.IntegerField(verbose_name=_("下跌0%~2%数量"), null=True, blank=True)
    down_2_to_4 = models.IntegerField(verbose_name=_("下跌2%~4%数量"), null=True, blank=True)
    down_4_to_6 = models.IntegerField(verbose_name=_("下跌4%~6%数量"), null=True, blank=True)
    down_6_to_8 = models.IntegerField(verbose_name=_("下跌6%~8%数量"), null=True, blank=True)
    down_8_to_limit = models.IntegerField(verbose_name=_("下跌8%~跌停数量"), null=True, blank=True)
    trade_time = models.DateTimeField(verbose_name=_("交易时间"), null=True, blank=True)
    
    class Meta:
        verbose_name = _("市场概览")
        verbose_name_plural = _("市场概览")
        db_table = "market_overview"
        indexes = [
            models.Index(fields=['trade_time']),
        ]
    
    def __str__(self):
        return f"市场概览({self.trade_time})"

class IndexTimeSeriesData(BaseModel):
    """
    时间序列数据模型
    
    用于存储指数的分时交易数据
    """
    index = models.ForeignKey(IndexInfo, on_delete=models.CASCADE, related_name="time_series", verbose_name=_("股票指数"))
    time_level = models.CharField(max_length=10, verbose_name=_("时间级别"))  # 5, 15, 30, 60, Day, Week, Month, Year
    trade_time = models.DateTimeField(verbose_name=_("交易时间"))
    open_price = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("开盘价"))
    high_price = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("最高价"))
    low_price = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("最低价"))
    close_price = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("收盘价"))
    volume = models.BigIntegerField(verbose_name=_("成交量(手)"))
    turnover = models.DecimalField(max_digits=20, decimal_places=2, null=True, verbose_name=_("成交额(元)"))
    amplitude = models.DecimalField(max_digits=8, decimal_places=4, null=True, verbose_name=_("振幅(%)"))
    turnover_rate = models.DecimalField(max_digits=8, decimal_places=4, null=True, verbose_name=_("换手率(%)"))
    change_percent = models.DecimalField(max_digits=8, decimal_places=4, null=True, verbose_name=_("涨跌幅(%)"))
    change_amount = models.DecimalField(max_digits=12, decimal_places=4, null=True, verbose_name=_("涨跌额(元)"))
    
    class Meta:
        verbose_name = _("时间序列数据")
        verbose_name_plural = _("时间序列数据")
        db_table = "index_time_series_data"
        unique_together = [['index', 'time_level', 'trade_time']]
        indexes = [
            models.Index(fields=['index']),
            models.Index(fields=['time_level']),
            models.Index(fields=['trade_time']),
        ]
    
    def __str__(self):
        return f"{self.index.name}{self.time_level}({self.trade_time})"

    def __code__(self):
        return self.index.code

class IndexKDJData(BaseModel):
    """
    KDJ指标数据模型
    
    存储股票指数的KDJ技术指标数据
    """
    index = models.ForeignKey(IndexInfo, on_delete=models.CASCADE, related_name="kdj_data", verbose_name=_("股票指数"))
    time_level = models.CharField(max_length=10, verbose_name=_("时间级别"))  # 5, 15, 30, 60, Day, Week, Month, Year
    trade_time = models.DateTimeField(verbose_name=_("交易时间"))
    k_value = models.DecimalField(max_digits=10, decimal_places=4, verbose_name=_("K值"))
    d_value = models.DecimalField(max_digits=10, decimal_places=4, verbose_name=_("D值"))
    j_value = models.DecimalField(max_digits=10, decimal_places=4, verbose_name=_("J值"))
    
    class Meta:
        verbose_name = _("KDJ指标数据")
        verbose_name_plural = _("KDJ指标数据")
        db_table = "index_kdj_data"
        unique_together = [['index', 'time_level', 'trade_time']]
        indexes = [
            models.Index(fields=['index']),
            models.Index(fields=['time_level']),
            models.Index(fields=['trade_time']),
        ]
    
    def __str__(self):
        return f"{self.index.name} KDJ {self.time_level}({self.trade_time})"

    def __code__(self):
        return self.index.code

class IndexMACDData(BaseModel):
    """
    MACD指标数据模型
    
    存储股票指数的MACD技术指标数据
    """
    index = models.ForeignKey(IndexInfo, on_delete=models.CASCADE, related_name="macd_data", verbose_name=_("股票指数"))
    time_level = models.CharField(max_length=10, verbose_name=_("时间级别"))  # 5, 15, 30, 60, Day, Week, Month, Year
    trade_time = models.DateTimeField(verbose_name=_("交易时间"))
    diff = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("DIFF值"))
    dea = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("DEA值"))
    macd = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("MACD值"))
    ema12 = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("EMA(12)值"))
    ema26 = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("EMA(26)值"))
    
    class Meta:
        verbose_name = _("MACD指标数据")
        verbose_name_plural = _("MACD指标数据")
        db_table = "index_macd_data"
        unique_together = [['index', 'time_level', 'trade_time']]
        indexes = [
            models.Index(fields=['index']),
            models.Index(fields=['time_level']),
            models.Index(fields=['trade_time']),
        ]
    
    def __str__(self):
        return f"{self.index.name} MACD {self.time_level}({self.trade_time})"

    def __code__(self):
        return self.index.code

class IndexMAData(BaseModel):
    """
    MA指标数据模型
    
    存储股票指数的MA技术指标数据
    """
    index = models.ForeignKey(IndexInfo, on_delete=models.CASCADE, related_name="ma_data", verbose_name=_("股票指数"))
    time_level = models.CharField(max_length=10, verbose_name=_("时间级别"))  # 5, 15, 30, 60, Day, Week, Month, Year
    trade_time = models.DateTimeField(verbose_name=_("交易时间"))
    ma3 = models.DecimalField(max_digits=12, decimal_places=4, null=True, verbose_name=_("MA3"))
    ma5 = models.DecimalField(max_digits=12, decimal_places=4, null=True, verbose_name=_("MA5"))
    ma10 = models.DecimalField(max_digits=12, decimal_places=4, null=True, verbose_name=_("MA10"))
    ma15 = models.DecimalField(max_digits=12, decimal_places=4, null=True, verbose_name=_("MA15"))
    ma20 = models.DecimalField(max_digits=12, decimal_places=4, null=True, verbose_name=_("MA20"))
    ma30 = models.DecimalField(max_digits=12, decimal_places=4, null=True, verbose_name=_("MA30"))
    ma60 = models.DecimalField(max_digits=12, decimal_places=4, null=True, verbose_name=_("MA60"))
    ma120 = models.DecimalField(max_digits=12, decimal_places=4, null=True, verbose_name=_("MA120"))
    ma200 = models.DecimalField(max_digits=12, decimal_places=4, null=True, verbose_name=_("MA200"))
    ma250 = models.DecimalField(max_digits=12, decimal_places=4, null=True, verbose_name=_("MA250"))
    
    class Meta:
        verbose_name = _("MA指标数据")
        verbose_name_plural = _("MA指标数据")
        db_table = "index_ma_data"
        unique_together = [['index', 'time_level', 'trade_time']]
        indexes = [
            models.Index(fields=['index']),
            models.Index(fields=['time_level']),
            models.Index(fields=['trade_time']),
        ]
    
    def __str__(self):
        return f"{self.index.name} MA {self.time_level}({self.trade_time})"

    def __code__(self):
        return self.index.code

class IndexBOLLData(BaseModel):
    """
    BOLL指标数据模型
    
    存储股票指数的BOLL技术指标数据
    """
    index = models.ForeignKey(IndexInfo, on_delete=models.CASCADE, related_name="boll_data", verbose_name=_("股票指数"))
    time_level = models.CharField(max_length=10, verbose_name=_("时间级别"))  # 5, 15, 30, 60, Day, Week, Month, Year
    trade_time = models.DateTimeField(verbose_name=_("交易时间"))
    upper = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("上轨"))
    middle = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("中轨"))
    lower = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("下轨"))
    
    class Meta:
        verbose_name = _("BOLL指标数据")
        verbose_name_plural = _("BOLL指标数据")
        db_table = "index_boll_data"
        unique_together = [['index', 'time_level', 'trade_time']]
        indexes = [
            models.Index(fields=['index']),
            models.Index(fields=['time_level']),
            models.Index(fields=['trade_time']),
        ]
    
    def __str__(self):
        return f"{self.index.name} BOLL {self.time_level}({self.trade_time})"

    def __code__(self):
        return self.index.code
