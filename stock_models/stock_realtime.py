# stock_models/stock_realtime.py
from django.db import models
from decimal import Decimal
from django.utils.translation import gettext_lazy as _
from stock_models.stock_basic import StockInfo

class BaseStockRealtimeData(models.Model): # 创建实时数据抽象基类
    """
    股票实时交易数据抽象基类
    """
    trade_time = models.DateTimeField(verbose_name='更新时间')
    open_price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='开盘价', null=True)
    prev_close_price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='昨日收盘价', null=True)
    current_price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='当前价格', null=True)
    high_price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='最高价', null=True)
    low_price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='最低价', null=True)
    volume = models.BigIntegerField(verbose_name='成交量', null=True) # 单位是“股”
    turnover_value = models.DecimalField(max_digits=20, decimal_places=2, verbose_name='成交额', null=True) # 单位是“元”
    class Meta: # 新增代码行
        abstract = True # 设置为抽象模型
        ordering = ['-trade_time']

class StockRealtimeData_SH(BaseStockRealtimeData): # 上海市场实时数据模型
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="realtime_data_sh", verbose_name=_("股票"))
    class Meta(BaseStockRealtimeData.Meta):
        abstract = False
        verbose_name = '实时交易数据-上海'
        verbose_name_plural = verbose_name
        db_table = 'stock_realtime_data_sh'
        unique_together = ('stock', 'trade_time')
        indexes = [models.Index(fields=['stock', 'trade_time'])]

class StockRealtimeData_SZ(BaseStockRealtimeData): # 深圳市场实时数据模型
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="realtime_data_sz", verbose_name=_("股票"))
    class Meta(BaseStockRealtimeData.Meta):
        abstract = False
        verbose_name = '实时交易数据-深圳'
        verbose_name_plural = verbose_name
        db_table = 'stock_realtime_data_sz'
        unique_together = ('stock', 'trade_time')
        indexes = [models.Index(fields=['stock', 'trade_time'])]

class StockRealtimeData_CY(BaseStockRealtimeData): # 创业板市场实时数据模型
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="realtime_data_cy", verbose_name=_("股票"))
    class Meta(BaseStockRealtimeData.Meta):
        abstract = False
        verbose_name = '实时交易数据-创业板'
        verbose_name_plural = verbose_name
        db_table = 'stock_realtime_data_cy'
        unique_together = ('stock', 'trade_time')
        indexes = [models.Index(fields=['stock', 'trade_time'])]

class StockRealtimeData_KC(BaseStockRealtimeData): # 科创板市场实时数据模型
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="realtime_data_kc", verbose_name=_("股票"))
    class Meta(BaseStockRealtimeData.Meta):
        abstract = False
        verbose_name = '实时交易数据-科创板'
        verbose_name_plural = verbose_name
        db_table = 'stock_realtime_data_kc'
        unique_together = ('stock', 'trade_time')
        indexes = [models.Index(fields=['stock', 'trade_time'])]

class StockRealtimeData_BJ(BaseStockRealtimeData): # 北京市场实时数据模型
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="realtime_data_bj", verbose_name=_("股票"))
    class Meta(BaseStockRealtimeData.Meta):
        abstract = False
        verbose_name = '实时交易数据-北京'
        verbose_name_plural = verbose_name
        db_table = 'stock_realtime_data_bj'
        unique_together = ('stock', 'trade_time')
        indexes = [models.Index(fields=['stock', 'trade_time'])]

class BaseStockLevel5Data(models.Model): # 创建Level5数据抽象基类
    """
    买卖五档盘口数据抽象基类
    """
    trade_time = models.DateTimeField(verbose_name='更新时间')
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
    class Meta: # 新增代码行
        abstract = True # 设置为抽象模型
        ordering = ['-trade_time']

class StockLevel5Data_SH(BaseStockLevel5Data): # 上海市场Level5数据模型
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="level5_data_sh", verbose_name=_("股票"))
    class Meta(BaseStockLevel5Data.Meta):
        abstract = False
        verbose_name = '买卖五档盘口数据-上海'
        verbose_name_plural = verbose_name
        db_table = 'stock_level5_data_sh'
        unique_together = ('stock', 'trade_time')
        indexes = [models.Index(fields=['stock', 'trade_time'])]

class StockLevel5Data_SZ(BaseStockLevel5Data): # 深圳市场Level5数据模型
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="level5_data_sz", verbose_name=_("股票"))
    class Meta(BaseStockLevel5Data.Meta):
        abstract = False
        verbose_name = '买卖五档盘口数据-深圳'
        verbose_name_plural = verbose_name
        db_table = 'stock_level5_data_sz'
        unique_together = ('stock', 'trade_time')
        indexes = [models.Index(fields=['stock', 'trade_time'])]

class StockLevel5Data_CY(BaseStockLevel5Data): # 创业板市场Level5数据模型
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="level5_data_cy", verbose_name=_("股票"))
    class Meta(BaseStockLevel5Data.Meta):
        abstract = False
        verbose_name = '买卖五档盘口数据-创业板'
        verbose_name_plural = verbose_name
        db_table = 'stock_level5_data_cy'
        unique_together = ('stock', 'trade_time')
        indexes = [models.Index(fields=['stock', 'trade_time'])]

class StockLevel5Data_KC(BaseStockLevel5Data): # 科创板市场Level5数据模型
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="level5_data_kc", verbose_name=_("股票"))
    class Meta(BaseStockLevel5Data.Meta):
        abstract = False
        verbose_name = '买卖五档盘口数据-科创板'
        verbose_name_plural = verbose_name
        db_table = 'stock_level5_data_kc'
        unique_together = ('stock', 'trade_time')
        indexes = [models.Index(fields=['stock', 'trade_time'])]

class StockLevel5Data_BJ(BaseStockLevel5Data): # 北京市场Level5数据模型
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="level5_data_bj", verbose_name=_("股票"))
    class Meta(BaseStockLevel5Data.Meta):
        abstract = False
        verbose_name = '买卖五档盘口数据-北京'
        verbose_name_plural = verbose_name
        db_table = 'stock_level5_data_bj'
        unique_together = ('stock', 'trade_time')
        indexes = [models.Index(fields=['stock', 'trade_time'])]

class BaseStockTickData(models.Model):
    trade_time = models.DateTimeField(verbose_name='交易时间')
    price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='成交价')
    volume = models.IntegerField(verbose_name='成交量(手)')
    price_change = models.DecimalField(max_digits=10, decimal_places=3, verbose_name=_('价格变动'), null=True, default=Decimal('0.00'))
    amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name='成交额(元)')
    type = models.CharField(max_length=1, verbose_name='类型', help_text="买盘'B'/卖盘'S'/中性'M'")
    class Meta:
        abstract = True
        ordering = ['-trade_time']

class StockTickData_SH(BaseStockTickData): #上海市场逐笔交易数据模型
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="tick_data_sh", verbose_name=_("股票"))
    class Meta(BaseStockTickData.Meta):
        abstract = False
        verbose_name = '逐笔交易数据-上海'
        verbose_name_plural = verbose_name
        db_table = 'stock_tick_data_sh'
        unique_together = ('stock', 'trade_time', 'price', 'volume')
        indexes = [models.Index(fields=['stock', 'trade_time'])]

class StockTickData_SZ(BaseStockTickData): #深圳市场逐笔交易数据模型
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="tick_data_sz", verbose_name=_("股票"))
    class Meta(BaseStockTickData.Meta):
        abstract = False
        verbose_name = '逐笔交易数据-深圳'
        verbose_name_plural = verbose_name
        db_table = 'stock_tick_data_sz'
        unique_together = ('stock', 'trade_time', 'price', 'volume')
        indexes = [models.Index(fields=['stock', 'trade_time'])]

class StockTickData_CY(BaseStockTickData): #创业板逐笔交易数据模型
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="tick_data_cy", verbose_name=_("股票"))
    class Meta(BaseStockTickData.Meta):
        abstract = False
        verbose_name = '逐笔交易数据-创业板'
        verbose_name_plural = verbose_name
        db_table = 'stock_tick_data_cy'
        unique_together = ('stock', 'trade_time', 'price', 'volume')
        indexes = [models.Index(fields=['stock', 'trade_time'])]

class StockTickData_KC(BaseStockTickData): #科创板逐笔交易数据模型
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="tick_data_kc", verbose_name=_("股票"))
    class Meta(BaseStockTickData.Meta):
        abstract = False
        verbose_name = '逐笔交易数据-科创板'
        verbose_name_plural = verbose_name
        db_table = 'stock_tick_data_kc'
        unique_together = ('stock', 'trade_time', 'price', 'volume')
        indexes = [models.Index(fields=['stock', 'trade_time'])]

class StockTickData_BJ(BaseStockTickData): #北京市场逐笔交易数据模型
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="tick_data_bj", verbose_name=_("股票"))
    class Meta(BaseStockTickData.Meta):
        abstract = False
        verbose_name = '逐笔交易数据-北京'
        verbose_name_plural = verbose_name
        db_table = 'stock_tick_data_bj'
        unique_together = ('stock', 'trade_time', 'price', 'volume')
        indexes = [models.Index(fields=['stock', 'trade_time'])]
