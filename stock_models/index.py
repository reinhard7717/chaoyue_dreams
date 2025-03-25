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
        db_table = "stock_index"
        indexes = [
            models.Index(fields=['code']),
            models.Index(fields=['exchange']),
        ]
    
    def __str__(self):
        return f"{self.name}({self.code})"


class IndexRealTimeData(BaseModel):
    """
    股票指数实时数据模型
    
    用于存储股票指数的实时交易数据，对应指数实时数据接口
    """
    index = models.ForeignKey(IndexInfo, on_delete=models.CASCADE, related_name="realtime_data", verbose_name=_("股票指数"))
    open_price = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("开盘价"))
    high_price = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("最高价"))
    low_price = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("最低价"))
    current_price = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("当前价格"))
    prev_close_price = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("昨日收盘价"))
    change_percent = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("涨跌幅(%)"))
    change_amount = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=_("涨跌额"))
    volume = models.BigIntegerField(verbose_name=_("成交量(手)"))
    turnover = models.DecimalField(max_digits=20, decimal_places=2, verbose_name=_("成交额(元)"))
    turnover_rate = models.DecimalField(max_digits=8, decimal_places=4, null=True, verbose_name=_("换手率(%)"))
    amplitude = models.DecimalField(max_digits=8, decimal_places=4, verbose_name=_("振幅(%)"))
    volume_ratio = models.DecimalField(max_digits=8, decimal_places=4, null=True, verbose_name=_("量比(%)"))
    five_min_change = models.DecimalField(max_digits=8, decimal_places=4, null=True, verbose_name=_("五分钟涨跌幅(%)"))
    change_speed = models.DecimalField(max_digits=8, decimal_places=4, null=True, verbose_name=_("涨速(%)"))
    circulating_market_value = models.DecimalField(max_digits=20, decimal_places=2, null=True, verbose_name=_("流通市值(元)"))
    total_market_value = models.DecimalField(max_digits=20, decimal_places=2, null=True, verbose_name=_("总市值(元)"))
    pe_ratio = models.DecimalField(max_digits=12, decimal_places=4, null=True, verbose_name=_("市盈率"))
    pb_ratio = models.DecimalField(max_digits=12, decimal_places=4, null=True, verbose_name=_("市净率"))
    change_60d = models.DecimalField(max_digits=8, decimal_places=4, null=True, verbose_name=_("60日涨跌幅(%)"))
    change_ytd = models.DecimalField(max_digits=8, decimal_places=4, null=True, verbose_name=_("年初至今涨跌幅(%)"))
    update_time = models.DateTimeField(verbose_name=_("更新时间"))
    
    class Meta:
        verbose_name = _("指数实时数据")
        verbose_name_plural = _("指数实时数据")
        db_table = "stock_index_realtime_data"
        indexes = [
            models.Index(fields=['index']),
            models.Index(fields=['update_time']),
        ]
    
    def __str__(self):
        return f"{self.index.name}实时数据({self.update_time})"


class MarketOverview(BaseModel):
    """
    市场概览数据模型
    
    用于存储沪深两市的股票上涨、下跌总数等市场概览数据
    """
    total_up = models.IntegerField(verbose_name=_("上涨总数"))
    total_down = models.IntegerField(verbose_name=_("下跌总数"))
    limit_up = models.IntegerField(verbose_name=_("涨停总数"))
    limit_down = models.IntegerField(verbose_name=_("跌停总数"))
    up_8_to_limit = models.IntegerField(verbose_name=_("上涨8%~涨停数量"))
    up_6_to_8 = models.IntegerField(verbose_name=_("上涨6%~8%数量"))
    up_4_to_6 = models.IntegerField(verbose_name=_("上涨4%~6%数量"))
    up_2_to_4 = models.IntegerField(verbose_name=_("上涨2%~4%数量"))
    up_0_to_2 = models.IntegerField(verbose_name=_("上涨0%~2%数量"))
    down_0_to_2 = models.IntegerField(verbose_name=_("下跌0%~2%数量"))
    down_2_to_4 = models.IntegerField(verbose_name=_("下跌2%~4%数量"))
    down_4_to_6 = models.IntegerField(verbose_name=_("下跌4%~6%数量"))
    down_6_to_8 = models.IntegerField(verbose_name=_("下跌6%~8%数量"))
    down_8_to_limit = models.IntegerField(verbose_name=_("下跌8%~跌停数量"))
    update_time = models.DateTimeField(verbose_name=_("更新时间"))
    
    class Meta:
        verbose_name = _("市场概览")
        verbose_name_plural = _("市场概览")
        db_table = "market_overview"
        indexes = [
            models.Index(fields=['update_time']),
        ]
    
    def __str__(self):
        return f"市场概览({self.update_time})"


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
        db_table = "time_series_data"
        unique_together = [['index', 'time_level', 'trade_time']]
        indexes = [
            models.Index(fields=['index']),
            models.Index(fields=['time_level']),
            models.Index(fields=['trade_time']),
        ]
    
    def __str__(self):
        return f"{self.index.name}{self.time_level}({self.trade_time})"

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
        db_table = "kdj_data"
        unique_together = [['index', 'time_level', 'trade_time']]
        indexes = [
            models.Index(fields=['index']),
            models.Index(fields=['time_level']),
            models.Index(fields=['trade_time']),
        ]
    
    def __str__(self):
        return f"{self.index.name} KDJ {self.time_level}({self.trade_time})"


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
        db_table = "macd_data"
        unique_together = [['index', 'time_level', 'trade_time']]
        indexes = [
            models.Index(fields=['index']),
            models.Index(fields=['time_level']),
            models.Index(fields=['trade_time']),
        ]
    
    def __str__(self):
        return f"{self.index.name} MACD {self.time_level}({self.trade_time})"


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
        db_table = "ma_data"
        unique_together = [['index', 'time_level', 'trade_time']]
        indexes = [
            models.Index(fields=['index']),
            models.Index(fields=['time_level']),
            models.Index(fields=['trade_time']),
        ]
    
    def __str__(self):
        return f"{self.index.name} MA {self.time_level}({self.trade_time})"


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
        db_table = "boll_data"
        unique_together = [['index', 'time_level', 'trade_time']]
        indexes = [
            models.Index(fields=['index']),
            models.Index(fields=['time_level']),
            models.Index(fields=['trade_time']),
        ]
    
    def __str__(self):
        return f"{self.index.name} BOLL {self.time_level}({self.trade_time})"
