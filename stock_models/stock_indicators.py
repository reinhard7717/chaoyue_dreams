from django.db import models
from django.utils.translation import gettext_lazy as _
from stock_models.stock_basic import StockInfo

# --- 指标模型 (使用斐波那契周期) ---
FIB_PERIODS = (5, 8, 13, 21, 34, 55, 89, 144, 233) # 定义斐波那契周期


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

    



# --- 中优先级指标模型 (斐波那契周期) ---
class StockDmiFIB(models.Model):
    """DMI 指标存储模型 (斐波那契周期)"""
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="dmi_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")
    plus_di13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="+DI(13)") # 常用14，这里用13
    minus_di13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="-DI(13)")
    adx13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADX(13)")
    adxr13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADXR(13)")
    plus_di21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="+DI(21)")
    minus_di21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="-DI(21)")
    adx21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADX(21)")
    adxr21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADXR(21)")
    plus_di34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="+DI(34)")
    minus_di34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="-DI(34)")
    adx34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADX(34)")
    adxr34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADXR(34)")
    plus_di55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="+DI(55)")
    minus_di55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="-DI(55)")
    adx55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADX(55)")
    adxr55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADXR(55)")
    plus_di89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="+DI(89)")
    minus_di89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="-DI(89)")
    adx89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADX(89)")
    adxr89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADXR(89)")
    plus_di144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="+DI(144)")
    minus_di144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="-DI(144)")
    adx144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADX(144)")
    adxr144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADXR(144)")
    plus_di233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="+DI(233)")
    minus_di233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="-DI(233)")
    adx233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADX(233)")
    adxr233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ADXR(233)")

    class Meta:
        verbose_name = "DMI指标(斐波那契)"
        db_table = 'stock_dmi_fib'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time', 'time_level')
        indexes = [ models.Index(fields=['stock', 'time_level', 'trade_time']), ]

class StockCciFIB(models.Model):
    """CCI 指标存储模型 (斐波那契周期)"""
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="cci_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")
    cci5 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CCI(5)")
    cci8 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CCI(8)")
    cci13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CCI(13)")
    cci21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CCI(21)")
    cci34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CCI(34)")
    cci55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CCI(55)")
    cci89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CCI(89)")
    cci144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CCI(144)")
    cci233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CCI(233)")

    class Meta:
        verbose_name = "CCI指标(斐波那契)"
        db_table = 'stock_cci_fib'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time', 'time_level')
        indexes = [ models.Index(fields=['stock', 'time_level', 'trade_time']), ]

class StockWrFIB(models.Model):
    """WR 指标存储模型 (斐波那契周期)"""
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="wr_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")
    wr5 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="WR(5)")
    wr8 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="WR(8)")
    wr13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="WR(13)")
    wr21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="WR(21)")
    wr34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="WR(34)")
    wr55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="WR(55)")
    wr89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="WR(89)")
    wr144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="WR(144)")
    wr233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="WR(233)")

    class Meta:
        verbose_name = "WR指标(斐波那契)"
        db_table = 'stock_wr_fib'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time', 'time_level')
        indexes = [ models.Index(fields=['stock', 'time_level', 'trade_time']), ]

class StockCmfFIB(models.Model):
    """CMF 指标存储模型 (斐波那契周期)"""
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="cmf_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")
    cmf5 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CMF(5)")
    cmf8 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CMF(8)")
    cmf13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CMF(13)")
    cmf21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CMF(21)")
    cmf34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CMF(34)")
    cmf55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CMF(55)")
    cmf89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CMF(89)")
    cmf144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CMF(144)")
    cmf233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="CMF(233)")

    class Meta:
        verbose_name = "CMF指标(斐波那契)"
        db_table = 'stock_cmf_fib'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time', 'time_level')
        indexes = [ models.Index(fields=['stock', 'time_level', 'trade_time']), ]

class StockMfiFIB(models.Model):
    """MFI 指标存储模型 (斐波那契周期)"""
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="mfi_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")
    mfi5 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MFI(5)")
    mfi8 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MFI(8)")
    mfi13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MFI(13)") # 常用14，这里用13
    mfi21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MFI(21)")
    mfi34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MFI(34)")
    mfi55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MFI(55)")
    mfi89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MFI(89)")
    mfi144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MFI(144)")
    mfi233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MFI(233)")

    class Meta:
        verbose_name = "MFI指标(斐波那契)"
        db_table = 'stock_mfi_fib'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time', 'time_level')
        indexes = [ models.Index(fields=['stock', 'time_level', 'trade_time']), ]

class StockIchimokuFIB(models.Model):
    """Ichimoku Cloud (一目均衡表) 指标存储模型"""
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="ichimoku_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")

    # finta.TA.ICHIMOKU 默认返回 'TENKAN', 'KIJUN', 'CHIKOU', 'SENKOU A', 'SENKOU B'
    # 使用 finta 返回的列名作为字段名，或者映射为你喜欢的名字
    tenkan_sen = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="转换线 (Tenkan Sen)")
    kijun_sen = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="基准线 (Kijun Sen)")
    chikou_span = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="延迟线 (Chikou Span)")
    senkou_span_a = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="先行带A (Senkou Span A)")
    senkou_span_b = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="先行带B (Senkou Span B)")

    class Meta:
        verbose_name = "Ichimoku指标"
        db_table = 'stock_ichimoku_fib'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time', 'time_level')
        indexes = [
            models.Index(fields=['stock', 'time_level', 'trade_time']),
        ]

class StockRocFIB(models.Model):
    """ROC (变动速率) 指标存储模型 (斐波那契周期)"""
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="roc_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")

    roc5 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ROC(5)")
    roc8 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ROC(8)")
    roc13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ROC(13)")
    roc21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ROC(21)")
    roc34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ROC(34)")
    roc55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ROC(55)")
    roc89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ROC(89)")
    roc144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ROC(144)")
    roc233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="ROC(233)")

    class Meta:
        verbose_name = "ROC指标(斐波那契)"
        db_table = 'stock_roc_fib'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time', 'time_level')
        indexes = [
            models.Index(fields=['stock', 'time_level', 'trade_time']),
        ]

class StockMomFIB(models.Model):
    """MOM (动量) 指标存储模型 (斐波那契周期)"""
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="mom_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")

    mom5 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MOM(5)")
    mom8 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MOM(8)")
    mom13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MOM(13)")
    mom21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MOM(21)")
    mom34 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MOM(34)")
    mom55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MOM(55)")
    mom89 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MOM(89)")
    mom144 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MOM(144)")
    mom233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="MOM(233)")

    class Meta:
        verbose_name = "MOM指标(斐波那契)"
        db_table = 'stock_mom_fib'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time', 'time_level')
        indexes = [
            models.Index(fields=['stock', 'time_level', 'trade_time']),
        ]

class StockVrocFIB(models.Model):
    """VROC (成交量变动速率) 指标存储模型 (斐波那契周期)"""
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="vroc_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")

    # 存储成交量的变化率 (通常是百分比) 或绝对差值
    # 如果是百分比，DecimalField 合适
    # 如果是绝对差值 (volume.diff(n))，BigIntegerField 可能更合适，取决于 volume 的类型
    # 这里假设存储的是类似 ROC 的比率值
    vroc5 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="VROC(5)") # 增加 max_digits 以防万一
    vroc8 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="VROC(8)")
    vroc13 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="VROC(13)")
    vroc21 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="VROC(21)")
    vroc34 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="VROC(34)")
    vroc55 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="VROC(55)")
    vroc89 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="VROC(89)")
    vroc144 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="VROC(144)")
    vroc233 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="VROC(233)")

    class Meta:
        verbose_name = "VROC指标(斐波那契)"
        db_table = 'stock_vroc_fib'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time', 'time_level')
        indexes = [
            models.Index(fields=['stock', 'time_level', 'trade_time']),
        ]

# --- 低优先级指标模型 (斐波那契周期) ---
class StockSarFIB(models.Model):
    """SAR 指标存储模型""" # SAR 参数不是简单的周期
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="sar_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")
    sar = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="SAR")

    class Meta:
        verbose_name = "SAR指标"
        db_table = 'stock_sar_fib'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time', 'time_level')
        indexes = [ models.Index(fields=['stock', 'time_level', 'trade_time']), ]

class StockAmountMaFIB(models.Model):
    """成交额移动平均线 (Amount MA) 指标存储模型 (斐波那契周期)"""
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="amount_ma_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")

    # 存储成交额的移动平均值，需要足够大的位数和小数位
    # 可以选择存储 SMA 或 EMA，这里以 SMA 为例命名
    amt_ma5 = models.DecimalField(max_digits=22, decimal_places=4, null=True, blank=True, verbose_name="成交额MA(5)")
    amt_ma8 = models.DecimalField(max_digits=22, decimal_places=4, null=True, blank=True, verbose_name="成交额MA(8)")
    amt_ma13 = models.DecimalField(max_digits=22, decimal_places=4, null=True, blank=True, verbose_name="成交额MA(13)")
    amt_ma21 = models.DecimalField(max_digits=22, decimal_places=4, null=True, blank=True, verbose_name="成交额MA(21)")
    amt_ma34 = models.DecimalField(max_digits=22, decimal_places=4, null=True, blank=True, verbose_name="成交额MA(34)")
    amt_ma55 = models.DecimalField(max_digits=22, decimal_places=4, null=True, blank=True, verbose_name="成交额MA(55)")
    amt_ma89 = models.DecimalField(max_digits=22, decimal_places=4, null=True, blank=True, verbose_name="成交额MA(89)")
    amt_ma144 = models.DecimalField(max_digits=22, decimal_places=4, null=True, blank=True, verbose_name="成交额MA(144)")
    amt_ma233 = models.DecimalField(max_digits=22, decimal_places=4, null=True, blank=True, verbose_name="成交额MA(233)")

    class Meta:
        verbose_name = "成交额MA指标(斐波那契)"
        db_table = 'stock_amount_ma_fib'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time', 'time_level')
        indexes = [
            models.Index(fields=['stock', 'time_level', 'trade_time']),
        ]

class StockAmountRocFIB(models.Model):
    """成交额变动速率 (AROC) 指标存储模型 (斐波那契周期)"""
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="amount_roc_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期")

    # 存储成交额的变化率 (通常是百分比)
    aroc5 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="AROC(5)")
    aroc8 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="AROC(8)")
    aroc13 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="AROC(13)")
    aroc21 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="AROC(21)")
    aroc34 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="AROC(34)")
    aroc55 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="AROC(55)")
    aroc89 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="AROC(89)")
    aroc144 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="AROC(144)")
    aroc233 = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="AROC(233)")

    class Meta:
        verbose_name = "成交额ROC指标(斐波那契)"
        db_table = 'stock_amount_roc_fib'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time', 'time_level')
        indexes = [
            models.Index(fields=['stock', 'time_level', 'trade_time']),
        ]

class StockVwapFIB(models.Model):
    """VWAP (成交量加权平均价) 指标存储模型"""
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="vwap_fib", verbose_name="股票")
    trade_time = models.DateTimeField(db_index=True, verbose_name="时间戳")
    time_level = models.CharField(max_length=10, db_index=True, verbose_name="K线周期") # 例如 'Day', '5min', '15min' 等

    # VWAP 值，精度应与价格类似
    vwap = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name="VWAP")

    class Meta:
        verbose_name = "VWAP指标"
        db_table = 'stock_vwap_fib'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time', 'time_level') # 确保每个时间点只有一个VWAP值
        indexes = [
            models.Index(fields=['stock', 'time_level', 'trade_time']),
        ]

    def __str__(self):
        return f"{self.stock.code} - {self.time_level} - {self.trade_time} - VWAP: {self.vwap}"

