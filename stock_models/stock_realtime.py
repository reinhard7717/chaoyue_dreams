from django.db import models
from django.utils.translation import gettext_lazy as _
from stock_models.stock_basic import StockInfo

class StockRealtimeData(models.Model):
    """
    股票实时交易数据模型
    """
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, blank=True, null=True, related_name="realtime_data", verbose_name=_("股票"))
    trade_time = models.DateTimeField(verbose_name='更新时间')
    open_price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='开盘价', null=True)
    prev_close_price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='昨日收盘价', null=True)
    current_price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='当前价格', null=True)
    high_price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='最高价', null=True)
    low_price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='最低价', null=True)
    volume = models.IntegerField(verbose_name='成交量', null=True)
    turnover_value = models.DecimalField(max_digits=20, decimal_places=2, verbose_name='成交额', null=True)
    
    class Meta:
        verbose_name = '实时交易数据'
        verbose_name_plural = verbose_name
        db_table = 'stock_realtime_data'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']
    
    def __str__(self):
        return f"{self.stock.stock_code}-{self.trade_time}"

class StockLevel5Data(models.Model):
    """
    【V2.0 - 重构版】买卖五档盘口数据模型
    - 核心修改: 移除 order_diff 和 order_ratio 字段，这些衍生指标应在应用层计算。
                模型只负责存储来自数据源的原始盘口数据。
    """
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, blank=True, null=True, related_name="level5_data", verbose_name=_("股票"))
    trade_time = models.DateTimeField(verbose_name='更新时间')
    
    # 买盘数据 (字段保持不变)
    buy_price1 = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='买1价', null=True)
    buy_volume1 = models.IntegerField(verbose_name='买1量', null=True)
    buy_price2 = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='买2价', null=True)
    buy_volume2 = models.IntegerField(verbose_name='买2量', null=True)
    buy_price3 = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='买3价', null=True)
    buy_volume3 = models.IntegerField(verbose_name='买3量', null=True)
    buy_price4 = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='买4价', null=True)
    buy_volume4 = models.IntegerField(verbose_name='买4量', null=True)
    buy_price5 = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='买5价', null=True)
    buy_volume5 = models.IntegerField(verbose_name='买5量', null=True)
    
    # 卖盘数据 (字段保持不变)
    sell_price1 = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='卖1价', null=True)
    sell_volume1 = models.IntegerField(verbose_name='卖1量', null=True)
    sell_price2 = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='卖2价', null=True)
    sell_volume2 = models.IntegerField(verbose_name='卖2量', null=True)
    sell_price3 = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='卖3价', null=True)
    sell_volume3 = models.IntegerField(verbose_name='卖3量', null=True)
    sell_price4 = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='卖4价', null=True)
    sell_volume4 = models.IntegerField(verbose_name='卖4量', null=True)
    sell_price5 = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='卖5价', null=True)
    sell_volume5 = models.IntegerField(verbose_name='卖5量', null=True)
    
    class Meta:
        verbose_name = '买卖五档盘口数据'
        verbose_name_plural = verbose_name
        db_table = 'stock_level5_data'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']
    
    def __str__(self):
        return f"{self.stock.stock_code}-{self.trade_time}"


class StockTickData(models.Model):
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="tick_data")
    trade_time = models.DateTimeField(verbose_name='交易时间')
    price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='成交价')
    volume = models.IntegerField(verbose_name='成交量(股)') # 注意单位是股
    amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name='成交额(元)')
    type = models.CharField(max_length=10, verbose_name='类型', help_text="买盘/卖盘/中性")

    class Meta:
        db_table = 'stock_tick_data'
        unique_together = ('stock', 'trade_time', 'price', 'volume') # 增加唯一性约束
        ordering = ['-trade_time']









