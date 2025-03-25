from django.db import models

class RealtimeData(models.Model):
    """
    股票实时交易数据模型
    """
    stock_code = models.CharField(max_length=10, verbose_name='股票代码')
    update_time = models.DateTimeField(verbose_name='更新时间')
    open_price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='开盘价', null=True)
    five_min_change = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='五分钟涨跌幅', null=True)
    high_price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='最高价', null=True)
    turnover_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='换手率', null=True)
    volume_ratio = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='量比', null=True)
    low_price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='最低价', null=True)
    tradable_market_value = models.DecimalField(max_digits=20, decimal_places=2, verbose_name='流通市值', null=True)
    pe_ratio = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='市盈率', null=True)
    price_change_percent = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='涨跌幅', null=True)
    current_price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='当前价格', null=True)
    total_market_value = models.DecimalField(max_digits=20, decimal_places=2, verbose_name='总市值', null=True)
    turnover_value = models.DecimalField(max_digits=20, decimal_places=2, verbose_name='成交额', null=True)
    price_change = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='涨跌额', null=True)
    volume = models.IntegerField(verbose_name='成交量', null=True)
    prev_close_price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='昨日收盘价', null=True)
    amplitude = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='振幅', null=True)
    increase_speed = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='涨速', null=True)
    pb_ratio = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='市净率', null=True)
    price_change_60d = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='60日涨跌幅', null=True)
    price_change_ytd = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='年初至今涨跌幅', null=True)
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    
    class Meta:
        verbose_name = '实时交易数据'
        verbose_name_plural = verbose_name
        db_table = 'realtime_data'
        unique_together = ('stock_code', 'update_time')
        ordering = ['-update_time']
    
    def __str__(self):
        return f"{self.stock_code}-{self.update_time}"


class Level5Data(models.Model):
    """
    买卖五档盘口数据模型
    """
    stock_code = models.CharField(max_length=10, verbose_name='股票代码')
    update_time = models.DateTimeField(verbose_name='更新时间')
    
    # 委托数据
    order_diff = models.IntegerField(verbose_name='委差', null=True)
    order_ratio = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='委比', null=True)
    
    # 买盘数据
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
    
    # 卖盘数据
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
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    
    class Meta:
        verbose_name = '买卖五档盘口数据'
        verbose_name_plural = verbose_name
        db_table = 'level5_data'
        unique_together = ('stock_code', 'update_time')
        ordering = ['-update_time']
    
    def __str__(self):
        return f"{self.stock_code}-{self.update_time}"


class TradeDetail(models.Model):
    """
    逐笔交易数据模型
    """
    stock_code = models.CharField(max_length=10, verbose_name='股票代码')
    trade_date = models.DateField(verbose_name='交易日期')
    trade_time = models.TimeField(verbose_name='交易时间')
    volume = models.IntegerField(verbose_name='成交量')
    price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='成交价')
    trade_direction = models.SmallIntegerField(verbose_name='交易方向', 
                                              choices=[(0, '中性盘'), (1, '买入'), (2, '卖出')])
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    
    class Meta:
        verbose_name = '逐笔交易数据'
        verbose_name_plural = verbose_name
        db_table = 'trade_detail'
        unique_together = ('stock_code', 'trade_date', 'trade_time', 'volume', 'price')
        ordering = ['-trade_date', '-trade_time']
    
    def __str__(self):
        return f"{self.stock_code}-{self.trade_date} {self.trade_time}"


class TimeDeal(models.Model):
    """
    分时成交数据模型
    """
    stock_code = models.CharField(max_length=10, verbose_name='股票代码')
    trade_date = models.DateField(verbose_name='交易日期')
    trade_time = models.TimeField(verbose_name='交易时间')
    volume = models.IntegerField(verbose_name='成交量')
    price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='成交价')
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    
    class Meta:
        verbose_name = '分时成交数据'
        verbose_name_plural = verbose_name
        db_table = 'time_deal'
        unique_together = ('stock_code', 'trade_date', 'trade_time')
        ordering = ['-trade_date', '-trade_time']
    
    def __str__(self):
        return f"{self.stock_code}-{self.trade_date} {self.trade_time}"


class PricePercent(models.Model):
    """
    分价成交占比数据模型
    """
    stock_code = models.CharField(max_length=10, verbose_name='股票代码')
    trade_date = models.DateField(verbose_name='交易日期')
    price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='成交价')
    volume = models.IntegerField(verbose_name='成交量')
    percentage = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='占比')
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    
    class Meta:
        verbose_name = '分价成交占比数据'
        verbose_name_plural = verbose_name
        db_table = 'price_percent'
        unique_together = ('stock_code', 'trade_date', 'price')
        ordering = ['-trade_date', 'price']
    
    def __str__(self):
        return f"{self.stock_code}-{self.trade_date}-{self.price}"


class BigDeal(models.Model):
    """
    逐笔大单交易数据模型
    """
    stock_code = models.CharField(max_length=10, verbose_name='股票代码')
    trade_date = models.DateField(verbose_name='交易日期')
    trade_time = models.TimeField(verbose_name='交易时间')
    volume = models.IntegerField(verbose_name='成交量')
    price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='成交价')
    trade_direction = models.SmallIntegerField(verbose_name='交易方向', 
                                              choices=[(0, '中性盘'), (1, '买入'), (2, '卖出')])
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    
    class Meta:
        verbose_name = '逐笔大单交易数据'
        verbose_name_plural = verbose_name
        db_table = 'big_deal'
        unique_together = ('stock_code', 'trade_date', 'trade_time', 'volume', 'price')
        ordering = ['-trade_date', '-trade_time']
    
    def __str__(self):
        return f"{self.stock_code}-{self.trade_date} {self.trade_time}"


class AbnormalMovement(models.Model):
    """
    盘中异动数据模型
    """
    stock_code = models.CharField(max_length=10, verbose_name='股票代码')
    stock_name = models.CharField(max_length=50, verbose_name='股票名称')
    movement_time = models.DateTimeField(verbose_name='异动时间')
    movement_type = models.CharField(max_length=20, verbose_name='异动类型')
    movement_info = models.CharField(max_length=100, verbose_name='相关信息')
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    
    class Meta:
        verbose_name = '盘中异动数据'
        verbose_name_plural = verbose_name
        db_table = 'abnormal_movement'
        unique_together = ('stock_code', 'movement_time', 'movement_type')
        ordering = ['-movement_time']
    
    def __str__(self):
        return f"{self.stock_code}-{self.stock_name}-{self.movement_time}-{self.movement_type}"
