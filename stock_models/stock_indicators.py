from django.db import models
from django.utils.translation import gettext_lazy as _
from stock_models.stock_basic import StockInfo

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
    price_change = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='涨跌额', null=True)
    
    class Meta:
        verbose_name = '分时交易数据'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade'
        unique_together = ('stock', 'time_level', 'trade_time')
        ordering = ['stock', 'time_level', 'trade_time']
    
    def __str__(self):
        return f"{self.stock.stock_code}-{self.time_level}-{self.trade_time}"

class StockKDJIndicator(models.Model):
    """
    KDJ指标数据模型
    """
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, blank=True, null=True, related_name="kdj_indicator", verbose_name=_("股票"))
    time_level = models.CharField(max_length=10, verbose_name='分时级别')
    trade_time = models.DateTimeField(verbose_name='交易时间')
    k_value = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='K值', null=True)
    d_value = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='D值', null=True)
    j_value = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='J值', null=True)
    
    class Meta:
        verbose_name = 'KDJ指标数据'
        verbose_name_plural = verbose_name
        db_table = 'stock_kdj_indicator'
        unique_together = ('stock', 'time_level', 'trade_time')
        ordering = ['stock', 'time_level', 'trade_time']
    
    def __str__(self):
        return f"{self.stock.stock_code}-{self.time_level}-{self.trade_time}"

class StockMACDIndicator(models.Model):
    """
    MACD指标数据模型
    """
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, blank=True, null=True, related_name="macd_indicator", verbose_name=_("股票"))
    time_level = models.CharField(max_length=10, verbose_name='分时级别')
    trade_time = models.DateTimeField(verbose_name='交易时间')
    diff = models.DecimalField(max_digits=10, decimal_places=3, verbose_name='DIFF值', null=True)
    dea = models.DecimalField(max_digits=10, decimal_places=3, verbose_name='DEA值', null=True)
    macd = models.DecimalField(max_digits=10, decimal_places=3, verbose_name='MACD值', null=True)
    ema12 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='EMA(12)值', null=True)
    ema26 = models.DecimalField(max_digits=10, decimal_places=4, verbose_name='EMA(26)值', null=True)
    
    class Meta:
        verbose_name = 'MACD指标数据'
        verbose_name_plural = verbose_name
        db_table = 'stock_macd_indicator'
        unique_together = ('stock', 'time_level', 'trade_time')
        ordering = ['stock', 'time_level', 'trade_time']
    
    def __str__(self):
        return f"{self.stock.stock_code}-{self.time_level}-{self.trade_time}"

class StockMAIndicator(models.Model):
    """
    MA指标数据模型
    """
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, blank=True, null=True, related_name="ma_indicator", verbose_name=_("股票"))
    time_level = models.CharField(max_length=10, verbose_name='分时级别')
    trade_time = models.DateTimeField(verbose_name='交易时间')
    ma3 = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='MA3', null=True)
    ma5 = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='MA5', null=True)
    ma10 = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='MA10', null=True)
    ma15 = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='MA15', null=True)
    ma20 = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='MA20', null=True)
    ma30 = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='MA30', null=True)
    ma60 = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='MA60', null=True)
    ma120 = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='MA120', null=True)
    ma200 = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='MA200', null=True)
    ma250 = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='MA250', null=True)
    
    class Meta:
        verbose_name = 'MA指标数据'
        verbose_name_plural = verbose_name
        db_table = 'stock_ma_indicator'
        unique_together = ('stock', 'time_level', 'trade_time')
        ordering = ['stock', 'time_level', 'trade_time']
    
    def __str__(self):
        return f"{self.stock.stock_code}-{self.time_level}-{self.trade_time}"

class StockBOLLIndicator(models.Model):
    """
    BOLL指标数据模型
    """
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, blank=True, null=True, related_name="boll_indicator", verbose_name=_("股票"))
    time_level = models.CharField(max_length=10, verbose_name='分时级别')
    trade_time = models.DateTimeField(verbose_name='交易时间')
    upper = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='上轨', null=True)
    lower = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='下轨', null=True)
    mid = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='中轨', null=True)
    
    class Meta:
        verbose_name = 'BOLL指标数据'
        verbose_name_plural = verbose_name
        db_table = 'stock_boll_indicator'
        unique_together = ('stock', 'time_level', 'trade_time')
        ordering = ['stock', 'time_level', 'trade_time']
    
    def __str__(self):
        return f"{self.stock.stock_code}-{self.time_level}-{self.trade_time}"
