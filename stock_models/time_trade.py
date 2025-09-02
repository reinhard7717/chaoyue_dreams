# stock_models\time_trade.py
from django.db import models
from django.utils.translation import gettext_lazy as _
import pandas as pd

# 股票每日重要基本面指标(StockDailyBasic)
class StockDailyBasic(models.Model):
    """每日重要基本面指标"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, to_field='stock_code', related_name='daily_basics', verbose_name='股票')
    trade_time = models.DateField(verbose_name='交易日期')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    turnover_rate = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='换手率(%)')
    turnover_rate_f = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='换手率(自由流通股)')
    volume_ratio = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='量比')
    pe = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name='市盈率')
    pe_ttm = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name='市盈率TTM')
    pb = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name='市净率')
    ps = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name='市销率')
    ps_ttm = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name='市销率TTM')
    dv_ratio = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='股息率(%)')
    dv_ttm = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='股息率TTM(%)')
    total_share = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=4, verbose_name='总股本(万股)')
    float_share = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=4, verbose_name='流通股本(万股)')
    free_share = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=4, verbose_name='自由流通股本(万)')
    total_mv = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=4, verbose_name='总市值(万元)')
    circ_mv = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=4, verbose_name='流通市值(万元)')
    limit_status = models.DecimalField(max_digits=1, decimal_places=0, null=True, blank=True, verbose_name='涨跌停状态')

    class Meta:
        verbose_name = '每日基本面指标'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_day_basic'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock.stock_code} {self.trade_time}"

class StockDailyBasic_SZ(models.Model):
    """每日重要基本面指标"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, to_field='stock_code', related_name='daily_basics_sz', verbose_name='股票')
    trade_time = models.DateField(verbose_name='交易日期')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    turnover_rate = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='换手率(%)')
    turnover_rate_f = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='换手率(自由流通股)')
    volume_ratio = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='量比')
    pe = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name='市盈率')
    pe_ttm = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name='市盈率TTM')
    pb = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name='市净率')
    ps = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name='市销率')
    ps_ttm = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name='市销率TTM')
    dv_ratio = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='股息率(%)')
    dv_ttm = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='股息率TTM(%)')
    total_share = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=4, verbose_name='总股本(万股)')
    float_share = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=4, verbose_name='流通股本(万股)')
    free_share = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=4, verbose_name='自由流通股本(万)')
    total_mv = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=4, verbose_name='总市值(万元)')
    circ_mv = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=4, verbose_name='流通市值(万元)')
    limit_status = models.DecimalField(max_digits=1, decimal_places=0, null=True, blank=True, verbose_name='涨跌停状态')

    class Meta:
        verbose_name = '每日基本面指标'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_day_basic_sz'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock.stock_code} {self.trade_time}"

class StockDailyBasic_SH(models.Model):
    """每日重要基本面指标"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, to_field='stock_code', related_name='daily_basics_sh', verbose_name='股票')
    trade_time = models.DateField(verbose_name='交易日期')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    turnover_rate = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='换手率(%)')
    turnover_rate_f = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='换手率(自由流通股)')
    volume_ratio = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='量比')
    pe = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name='市盈率')
    pe_ttm = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name='市盈率TTM')
    pb = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name='市净率')
    ps = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name='市销率')
    ps_ttm = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name='市销率TTM')
    dv_ratio = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='股息率(%)')
    dv_ttm = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='股息率TTM(%)')
    total_share = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=4, verbose_name='总股本(万股)')
    float_share = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=4, verbose_name='流通股本(万股)')
    free_share = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=4, verbose_name='自由流通股本(万)')
    total_mv = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=4, verbose_name='总市值(万元)')
    circ_mv = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=4, verbose_name='流通市值(万元)')
    limit_status = models.DecimalField(max_digits=1, decimal_places=0, null=True, blank=True, verbose_name='涨跌停状态')

    class Meta:
        verbose_name = '每日基本面指标'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_day_basic_sh'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock.stock_code} {self.trade_time}"

class StockDailyBasic_CY(models.Model):
    """每日重要基本面指标"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, to_field='stock_code', related_name='daily_basics_cy', verbose_name='股票')

    trade_time = models.DateField(verbose_name='交易日期')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    turnover_rate = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='换手率(%)')
    turnover_rate_f = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='换手率(自由流通股)')
    volume_ratio = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='量比')
    pe = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name='市盈率')
    pe_ttm = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name='市盈率TTM')
    pb = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name='市净率')
    ps = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name='市销率')
    ps_ttm = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name='市销率TTM')
    dv_ratio = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='股息率(%)')
    dv_ttm = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='股息率TTM(%)')
    total_share = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=4, verbose_name='总股本(万股)')
    float_share = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=4, verbose_name='流通股本(万股)')
    free_share = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=4, verbose_name='自由流通股本(万)')
    total_mv = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=4, verbose_name='总市值(万元)')
    circ_mv = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=4, verbose_name='流通市值(万元)')
    limit_status = models.DecimalField(max_digits=1, decimal_places=0, null=True, blank=True, verbose_name='涨跌停状态')

    class Meta:
        verbose_name = '每日基本面指标'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_day_basic_cy'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock.stock_code} {self.trade_time}"

class StockDailyBasic_KC(models.Model):
    """每日重要基本面指标"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, to_field='stock_code', related_name='daily_basics_kc', verbose_name='股票')
    trade_time = models.DateField(verbose_name='交易日期')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    turnover_rate = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='换手率(%)')
    turnover_rate_f = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='换手率(自由流通股)')
    volume_ratio = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='量比')
    pe = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name='市盈率')
    pe_ttm = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name='市盈率TTM')
    pb = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name='市净率')
    ps = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name='市销率')
    ps_ttm = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name='市销率TTM')
    dv_ratio = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='股息率(%)')
    dv_ttm = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='股息率TTM(%)')
    total_share = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=4, verbose_name='总股本(万股)')
    float_share = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=4, verbose_name='流通股本(万股)')
    free_share = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=4, verbose_name='自由流通股本(万)')
    total_mv = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=4, verbose_name='总市值(万元)')
    circ_mv = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=4, verbose_name='流通市值(万元)')
    limit_status = models.DecimalField(max_digits=1, decimal_places=0, null=True, blank=True, verbose_name='涨跌停状态')

    class Meta:
        verbose_name = '每日基本面指标'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_day_basic_kc'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock.stock_code} {self.trade_time}"

class StockDailyBasic_BJ(models.Model):
    """每日重要基本面指标"""
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, to_field='stock_code', related_name='daily_basics_bj', verbose_name='股票')

    trade_time = models.DateField(verbose_name='交易日期')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    turnover_rate = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='换手率(%)')
    turnover_rate_f = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='换手率(自由流通股)')
    volume_ratio = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='量比')
    pe = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name='市盈率')
    pe_ttm = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name='市盈率TTM')
    pb = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name='市净率')
    ps = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name='市销率')
    ps_ttm = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name='市销率TTM')
    dv_ratio = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='股息率(%)')
    dv_ttm = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True, verbose_name='股息率TTM(%)')
    total_share = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=4, verbose_name='总股本(万股)')
    float_share = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=4, verbose_name='流通股本(万股)')
    free_share = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=4, verbose_name='自由流通股本(万)')
    total_mv = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=4, verbose_name='总市值(万元)')
    circ_mv = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=4, verbose_name='流通市值(万元)')
    limit_status = models.DecimalField(max_digits=1, decimal_places=0, null=True, blank=True, verbose_name='涨跌停状态')

    class Meta:
        verbose_name = '每日基本面指标'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_day_basic_bj'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock.stock_code} {self.trade_time}"

# 日线行情模型（StockDailyData）
class StockDailyData(models.Model):
    """A股日线行情（带复权）"""
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='stock_code',
        on_delete=models.CASCADE,
        related_name='daily_data',
        verbose_name='股票'
    )
    trade_time = models.DateField(verbose_name='交易日期', db_index=True)
    open = models.FloatField(null=True, blank=True, verbose_name='开盘价')
    high = models.FloatField(null=True, blank=True, verbose_name='最高价')
    low = models.FloatField(null=True, blank=True, verbose_name='最低价')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    pre_close = models.FloatField(null=True, blank=True, verbose_name='昨收价')
    change = models.FloatField(null=True, blank=True, verbose_name='涨跌额')
    pct_change = models.FloatField(null=True, blank=True, verbose_name='涨跌幅')
    vol = models.BigIntegerField(null=True, blank=True, verbose_name='成交量（手）')
    amount = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=3, verbose_name='成交额（千元）')
    adj_factor = models.DecimalField(null=True, blank=True, max_digits=10, decimal_places=3, verbose_name='复权因子')
    open_qfq = models.FloatField(null=True, blank=True, verbose_name='开盘价（前复权）')
    high_qfq = models.FloatField(null=True, blank=True, verbose_name='最高价（前复权）')
    low_qfq = models.FloatField(null=True, blank=True, verbose_name='最低价（前复权）')
    close_qfq = models.FloatField(null=True, blank=True, verbose_name='收盘价（前复权）')
    pre_close_qfq = models.FloatField(null=True, blank=True, verbose_name='昨收价（前复权）')
    open_hfq = models.FloatField(null=True, blank=True, verbose_name='开盘价（后复权）')
    high_hfq = models.FloatField(null=True, blank=True, verbose_name='最高价（后复权）')
    low_hfq = models.FloatField(null=True, blank=True, verbose_name='最低价（后复权）')
    close_hfq = models.FloatField(null=True, blank=True, verbose_name='收盘价（后复权）')
    pre_close_hfq = models.FloatField(null=True, blank=True, verbose_name='昨收价（后复权）')

    class Meta:
        verbose_name = 'A股日线行情'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_day'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock} {self.trade_time}"

class StockDailyData_SZ(models.Model):
    """A股日线行情（带复权）"""
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='stock_code',
        on_delete=models.CASCADE,
        related_name='daily_data_sz',
        verbose_name='股票'
    )
    trade_time = models.DateField(verbose_name='交易日期', db_index=True)
    open = models.FloatField(null=True, blank=True, verbose_name='开盘价')
    high = models.FloatField(null=True, blank=True, verbose_name='最高价')
    low = models.FloatField(null=True, blank=True, verbose_name='最低价')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    pre_close = models.FloatField(null=True, blank=True, verbose_name='昨收价')
    change = models.FloatField(null=True, blank=True, verbose_name='涨跌额')
    pct_change = models.FloatField(null=True, blank=True, verbose_name='涨跌幅')
    vol = models.BigIntegerField(null=True, blank=True, verbose_name='成交量（手）')
    amount = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=3, verbose_name='成交额（千元）')
    adj_factor = models.DecimalField(null=True, blank=True, max_digits=10, decimal_places=3, verbose_name='复权因子')
    open_qfq = models.FloatField(null=True, blank=True, verbose_name='开盘价（前复权）')
    high_qfq = models.FloatField(null=True, blank=True, verbose_name='最高价（前复权）')
    low_qfq = models.FloatField(null=True, blank=True, verbose_name='最低价（前复权）')
    close_qfq = models.FloatField(null=True, blank=True, verbose_name='收盘价（前复权）')
    pre_close_qfq = models.FloatField(null=True, blank=True, verbose_name='昨收价（前复权）')
    open_hfq = models.FloatField(null=True, blank=True, verbose_name='开盘价（后复权）')
    high_hfq = models.FloatField(null=True, blank=True, verbose_name='最高价（后复权）')
    low_hfq = models.FloatField(null=True, blank=True, verbose_name='最低价（后复权）')
    close_hfq = models.FloatField(null=True, blank=True, verbose_name='收盘价（后复权）')
    pre_close_hfq = models.FloatField(null=True, blank=True, verbose_name='昨收价（后复权）')

    class Meta:
        verbose_name = 'A股日线行情'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_day_sz'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock} {self.trade_time}"

class StockDailyData_SH(models.Model):
    """A股日线行情（带复权）"""
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='stock_code',
        on_delete=models.CASCADE,
        related_name='daily_data_sh',
        verbose_name='股票'
    )
    trade_time = models.DateField(verbose_name='交易日期', db_index=True)
    open = models.FloatField(null=True, blank=True, verbose_name='开盘价')
    high = models.FloatField(null=True, blank=True, verbose_name='最高价')
    low = models.FloatField(null=True, blank=True, verbose_name='最低价')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    pre_close = models.FloatField(null=True, blank=True, verbose_name='昨收价')
    change = models.FloatField(null=True, blank=True, verbose_name='涨跌额')
    pct_change = models.FloatField(null=True, blank=True, verbose_name='涨跌幅')
    vol = models.BigIntegerField(null=True, blank=True, verbose_name='成交量（手）')
    amount = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=3, verbose_name='成交额（千元）')
    adj_factor = models.DecimalField(null=True, blank=True, max_digits=10, decimal_places=3, verbose_name='复权因子')
    open_qfq = models.FloatField(null=True, blank=True, verbose_name='开盘价（前复权）')
    high_qfq = models.FloatField(null=True, blank=True, verbose_name='最高价（前复权）')
    low_qfq = models.FloatField(null=True, blank=True, verbose_name='最低价（前复权）')
    close_qfq = models.FloatField(null=True, blank=True, verbose_name='收盘价（前复权）')
    pre_close_qfq = models.FloatField(null=True, blank=True, verbose_name='昨收价（前复权）')
    open_hfq = models.FloatField(null=True, blank=True, verbose_name='开盘价（后复权）')
    high_hfq = models.FloatField(null=True, blank=True, verbose_name='最高价（后复权）')
    low_hfq = models.FloatField(null=True, blank=True, verbose_name='最低价（后复权）')
    close_hfq = models.FloatField(null=True, blank=True, verbose_name='收盘价（后复权）')
    pre_close_hfq = models.FloatField(null=True, blank=True, verbose_name='昨收价（后复权）')

    class Meta:
        verbose_name = 'A股日线行情'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_day_sh'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock} {self.trade_time}"

class StockDailyData_CY(models.Model):
    """A股日线行情（带复权）"""
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='stock_code',
        on_delete=models.CASCADE,
        related_name='daily_data_cy',
        verbose_name='股票'
    )
    trade_time = models.DateField(verbose_name='交易日期', db_index=True)
    open = models.FloatField(null=True, blank=True, verbose_name='开盘价')
    high = models.FloatField(null=True, blank=True, verbose_name='最高价')
    low = models.FloatField(null=True, blank=True, verbose_name='最低价')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    pre_close = models.FloatField(null=True, blank=True, verbose_name='昨收价')
    change = models.FloatField(null=True, blank=True, verbose_name='涨跌额')
    pct_change = models.FloatField(null=True, blank=True, verbose_name='涨跌幅')
    vol = models.BigIntegerField(null=True, blank=True, verbose_name='成交量（手）')
    amount = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=3, verbose_name='成交额（千元）')
    adj_factor = models.DecimalField(null=True, blank=True, max_digits=10, decimal_places=3, verbose_name='复权因子')
    open_qfq = models.FloatField(null=True, blank=True, verbose_name='开盘价（前复权）')
    high_qfq = models.FloatField(null=True, blank=True, verbose_name='最高价（前复权）')
    low_qfq = models.FloatField(null=True, blank=True, verbose_name='最低价（前复权）')
    close_qfq = models.FloatField(null=True, blank=True, verbose_name='收盘价（前复权）')
    pre_close_qfq = models.FloatField(null=True, blank=True, verbose_name='昨收价（前复权）')
    open_hfq = models.FloatField(null=True, blank=True, verbose_name='开盘价（后复权）')
    high_hfq = models.FloatField(null=True, blank=True, verbose_name='最高价（后复权）')
    low_hfq = models.FloatField(null=True, blank=True, verbose_name='最低价（后复权）')
    close_hfq = models.FloatField(null=True, blank=True, verbose_name='收盘价（后复权）')
    pre_close_hfq = models.FloatField(null=True, blank=True, verbose_name='昨收价（后复权）')

    class Meta:
        verbose_name = 'A股日线行情'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_day_cy'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock} {self.trade_time}"

class StockDailyData_KC(models.Model):
    """A股日线行情（带复权）"""
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='stock_code',
        on_delete=models.CASCADE,
        related_name='daily_data_kc',
        verbose_name='股票'
    )
    trade_time = models.DateField(verbose_name='交易日期', db_index=True)
    open = models.FloatField(null=True, blank=True, verbose_name='开盘价')
    high = models.FloatField(null=True, blank=True, verbose_name='最高价')
    low = models.FloatField(null=True, blank=True, verbose_name='最低价')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    pre_close = models.FloatField(null=True, blank=True, verbose_name='昨收价')
    change = models.FloatField(null=True, blank=True, verbose_name='涨跌额')
    pct_change = models.FloatField(null=True, blank=True, verbose_name='涨跌幅')
    vol = models.BigIntegerField(null=True, blank=True, verbose_name='成交量（手）')
    amount = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=3, verbose_name='成交额（千元）')
    adj_factor = models.DecimalField(null=True, blank=True, max_digits=10, decimal_places=3, verbose_name='复权因子')
    open_qfq = models.FloatField(null=True, blank=True, verbose_name='开盘价（前复权）')
    high_qfq = models.FloatField(null=True, blank=True, verbose_name='最高价（前复权）')
    low_qfq = models.FloatField(null=True, blank=True, verbose_name='最低价（前复权）')
    close_qfq = models.FloatField(null=True, blank=True, verbose_name='收盘价（前复权）')
    pre_close_qfq = models.FloatField(null=True, blank=True, verbose_name='昨收价（前复权）')
    open_hfq = models.FloatField(null=True, blank=True, verbose_name='开盘价（后复权）')
    high_hfq = models.FloatField(null=True, blank=True, verbose_name='最高价（后复权）')
    low_hfq = models.FloatField(null=True, blank=True, verbose_name='最低价（后复权）')
    close_hfq = models.FloatField(null=True, blank=True, verbose_name='收盘价（后复权）')
    pre_close_hfq = models.FloatField(null=True, blank=True, verbose_name='昨收价（后复权）')

    class Meta:
        verbose_name = 'A股日线行情'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_day_kc'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock} {self.trade_time}"

class StockDailyData_BJ(models.Model):
    """A股日线行情（带复权）"""
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='stock_code',
        on_delete=models.CASCADE,
        related_name='daily_data_bj',
        verbose_name='股票'
    )
    trade_time = models.DateField(verbose_name='交易日期', db_index=True)
    open = models.FloatField(null=True, blank=True, verbose_name='开盘价')
    high = models.FloatField(null=True, blank=True, verbose_name='最高价')
    low = models.FloatField(null=True, blank=True, verbose_name='最低价')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    pre_close = models.FloatField(null=True, blank=True, verbose_name='昨收价')
    change = models.FloatField(null=True, blank=True, verbose_name='涨跌额')
    pct_change = models.FloatField(null=True, blank=True, verbose_name='涨跌幅')
    vol = models.BigIntegerField(null=True, blank=True, verbose_name='成交量（手）')
    amount = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=3, verbose_name='成交额（千元）')
    adj_factor = models.DecimalField(null=True, blank=True, max_digits=10, decimal_places=3, verbose_name='复权因子')
    open_qfq = models.FloatField(null=True, blank=True, verbose_name='开盘价（前复权）')
    high_qfq = models.FloatField(null=True, blank=True, verbose_name='最高价（前复权）')
    low_qfq = models.FloatField(null=True, blank=True, verbose_name='最低价（前复权）')
    close_qfq = models.FloatField(null=True, blank=True, verbose_name='收盘价（前复权）')
    pre_close_qfq = models.FloatField(null=True, blank=True, verbose_name='昨收价（前复权）')
    open_hfq = models.FloatField(null=True, blank=True, verbose_name='开盘价（后复权）')
    high_hfq = models.FloatField(null=True, blank=True, verbose_name='最高价（后复权）')
    low_hfq = models.FloatField(null=True, blank=True, verbose_name='最低价（后复权）')
    close_hfq = models.FloatField(null=True, blank=True, verbose_name='收盘价（后复权）')
    pre_close_hfq = models.FloatField(null=True, blank=True, verbose_name='昨收价（后复权）')

    class Meta:
        verbose_name = 'A股日线行情'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_day_bj'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock} {self.trade_time}"

# 分钟行情模型（StockMinuteData）
class StockMinuteData(models.Model):
    """A股分钟行情"""
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='stock_code',
        on_delete=models.CASCADE,
        related_name='minute_data',
        verbose_name='股票'
    )
    trade_time = models.DateTimeField(verbose_name='交易时间', db_index=True)
    time_level = models.CharField(max_length=10, verbose_name='分钟频度')  # 1min/5min/15min/30min/60min
    open = models.FloatField(null=True, blank=True, verbose_name='开盘价')
    high = models.FloatField(null=True, blank=True, verbose_name='最高价')
    low = models.FloatField(null=True, blank=True, verbose_name='最低价')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    vol = models.BigIntegerField(verbose_name='成交量')
    amount = models.DecimalField(max_digits=20, decimal_places=3, verbose_name='成交金额')

    class Meta:
        verbose_name = 'A股分钟行情'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_minute'
        unique_together = ('stock', 'trade_time', 'time_level')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock} {self.trade_time}"

class StockMinuteData_1_SZ(models.Model):
    """A股分钟行情"""
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='stock_code',
        on_delete=models.CASCADE,
        related_name='minute_data_1_sz',
        verbose_name='股票'
    )
    trade_time = models.DateTimeField(verbose_name='交易时间', db_index=True)
    open = models.FloatField(null=True, blank=True, verbose_name='开盘价')
    high = models.FloatField(null=True, blank=True, verbose_name='最高价')
    low = models.FloatField(null=True, blank=True, verbose_name='最低价')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    vol = models.BigIntegerField(verbose_name='成交量')
    amount = models.DecimalField(max_digits=20, decimal_places=3, verbose_name='成交金额')

    class Meta:
        verbose_name = 'A股分钟行情'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_minute_1_sz'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock} {self.trade_time}"

class StockMinuteData_5_SZ(models.Model):
    """A股分钟行情"""
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='stock_code',
        on_delete=models.CASCADE,
        related_name='minute_data_5_sz',
        verbose_name='股票'
    )
    trade_time = models.DateTimeField(verbose_name='交易时间', db_index=True)
    open = models.FloatField(null=True, blank=True, verbose_name='开盘价')
    high = models.FloatField(null=True, blank=True, verbose_name='最高价')
    low = models.FloatField(null=True, blank=True, verbose_name='最低价')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    vol = models.BigIntegerField(verbose_name='成交量')
    amount = models.DecimalField(max_digits=20, decimal_places=3, verbose_name='成交金额')

    class Meta:
        verbose_name = 'A股分钟行情'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_minute_5_sz'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock} {self.trade_time}"

class StockMinuteData_15_SZ(models.Model):
    """A股分钟行情"""
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='stock_code',
        on_delete=models.CASCADE,
        related_name='minute_data_15_sz',
        verbose_name='股票'
    )
    trade_time = models.DateTimeField(verbose_name='交易时间', db_index=True)
    open = models.FloatField(null=True, blank=True, verbose_name='开盘价')
    high = models.FloatField(null=True, blank=True, verbose_name='最高价')
    low = models.FloatField(null=True, blank=True, verbose_name='最低价')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    vol = models.BigIntegerField(verbose_name='成交量')
    amount = models.DecimalField(max_digits=20, decimal_places=3, verbose_name='成交金额')

    class Meta:
        verbose_name = 'A股分钟行情'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_minute_15_sz'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock} {self.trade_time}"

class StockMinuteData_30_SZ(models.Model):
    """A股分钟行情"""
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='stock_code',
        on_delete=models.CASCADE,
        related_name='minute_data_30_sz',
        verbose_name='股票'
    )
    trade_time = models.DateTimeField(verbose_name='交易时间', db_index=True)
    open = models.FloatField(null=True, blank=True, verbose_name='开盘价')
    high = models.FloatField(null=True, blank=True, verbose_name='最高价')
    low = models.FloatField(null=True, blank=True, verbose_name='最低价')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    vol = models.BigIntegerField(verbose_name='成交量')
    amount = models.DecimalField(max_digits=20, decimal_places=3, verbose_name='成交金额')

    class Meta:
        verbose_name = 'A股分钟行情'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_minute_30_sz'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock} {self.trade_time}"

class StockMinuteData_60_SZ(models.Model):
    """A股分钟行情"""
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='stock_code',
        on_delete=models.CASCADE,
        related_name='minute_data_60_sz',
        verbose_name='股票'
    )
    trade_time = models.DateTimeField(verbose_name='交易时间', db_index=True)
    open = models.FloatField(null=True, blank=True, verbose_name='开盘价')
    high = models.FloatField(null=True, blank=True, verbose_name='最高价')
    low = models.FloatField(null=True, blank=True, verbose_name='最低价')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    vol = models.BigIntegerField(verbose_name='成交量')
    amount = models.DecimalField(max_digits=20, decimal_places=3, verbose_name='成交金额')

    class Meta:
        verbose_name = 'A股分钟行情'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_minute_60_sz'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock} {self.trade_time}"

class StockMinuteData_1_SH(models.Model):
    """A股分钟行情"""
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='stock_code',
        on_delete=models.CASCADE,
        related_name='minute_data_1_sh',
        verbose_name='股票'
    )
    trade_time = models.DateTimeField(verbose_name='交易时间', db_index=True)
    open = models.FloatField(null=True, blank=True, verbose_name='开盘价')
    high = models.FloatField(null=True, blank=True, verbose_name='最高价')
    low = models.FloatField(null=True, blank=True, verbose_name='最低价')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    vol = models.BigIntegerField(verbose_name='成交量')
    amount = models.DecimalField(max_digits=20, decimal_places=3, verbose_name='成交金额')

    class Meta:
        verbose_name = 'A股分钟行情'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_minute_1_sh'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock} {self.trade_time}"

class StockMinuteData_5_SH(models.Model):
    """A股分钟行情"""
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='stock_code',
        on_delete=models.CASCADE,
        related_name='minute_data_5_sh',
        verbose_name='股票'
    )
    trade_time = models.DateTimeField(verbose_name='交易时间', db_index=True)
    open = models.FloatField(null=True, blank=True, verbose_name='开盘价')
    high = models.FloatField(null=True, blank=True, verbose_name='最高价')
    low = models.FloatField(null=True, blank=True, verbose_name='最低价')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    vol = models.BigIntegerField(verbose_name='成交量')
    amount = models.DecimalField(max_digits=20, decimal_places=3, verbose_name='成交金额')

    class Meta:
        verbose_name = 'A股分钟行情'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_minute_5_sh'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock} {self.trade_time}"

class StockMinuteData_15_SH(models.Model):
    """A股分钟行情"""
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='stock_code',
        on_delete=models.CASCADE,
        related_name='minute_data_15_sh',
        verbose_name='股票'
    )
    trade_time = models.DateTimeField(verbose_name='交易时间', db_index=True)
    open = models.FloatField(null=True, blank=True, verbose_name='开盘价')
    high = models.FloatField(null=True, blank=True, verbose_name='最高价')
    low = models.FloatField(null=True, blank=True, verbose_name='最低价')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    vol = models.BigIntegerField(verbose_name='成交量')
    amount = models.DecimalField(max_digits=20, decimal_places=3, verbose_name='成交金额')

    class Meta:
        verbose_name = 'A股分钟行情'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_minute_15_sh'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock} {self.trade_time}"

class StockMinuteData_30_SH(models.Model):
    """A股分钟行情"""
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='stock_code',
        on_delete=models.CASCADE,
        related_name='minute_data_30_sh',
        verbose_name='股票'
    )
    trade_time = models.DateTimeField(verbose_name='交易时间', db_index=True)
    open = models.FloatField(null=True, blank=True, verbose_name='开盘价')
    high = models.FloatField(null=True, blank=True, verbose_name='最高价')
    low = models.FloatField(null=True, blank=True, verbose_name='最低价')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    vol = models.BigIntegerField(verbose_name='成交量')
    amount = models.DecimalField(max_digits=20, decimal_places=3, verbose_name='成交金额')

    class Meta:
        verbose_name = 'A股分钟行情'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_minute_30_sh'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock} {self.trade_time}"

class StockMinuteData_60_SH(models.Model):
    """A股分钟行情"""
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='stock_code',
        on_delete=models.CASCADE,
        related_name='minute_data_60_sh',
        verbose_name='股票'
    )
    trade_time = models.DateTimeField(verbose_name='交易时间', db_index=True)
    open = models.FloatField(null=True, blank=True, verbose_name='开盘价')
    high = models.FloatField(null=True, blank=True, verbose_name='最高价')
    low = models.FloatField(null=True, blank=True, verbose_name='最低价')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    vol = models.BigIntegerField(verbose_name='成交量')
    amount = models.DecimalField(max_digits=20, decimal_places=3, verbose_name='成交金额')

    class Meta:
        verbose_name = 'A股分钟行情'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_minute_60_sh'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock} {self.trade_time}"

class StockMinuteData_1_BJ(models.Model):
    """A股分钟行情"""
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='stock_code',
        on_delete=models.CASCADE,
        related_name='minute_data_1_bj',
        verbose_name='股票'
    )
    trade_time = models.DateTimeField(verbose_name='交易时间', db_index=True)
    open = models.FloatField(null=True, blank=True, verbose_name='开盘价')
    high = models.FloatField(null=True, blank=True, verbose_name='最高价')
    low = models.FloatField(null=True, blank=True, verbose_name='最低价')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    vol = models.BigIntegerField(verbose_name='成交量')
    amount = models.DecimalField(max_digits=20, decimal_places=3, verbose_name='成交金额')

    class Meta:
        verbose_name = 'A股分钟行情'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_minute_1_bj'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock} {self.trade_time}"

class StockMinuteData_5_BJ(models.Model):
    """A股分钟行情"""
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='stock_code',
        on_delete=models.CASCADE,
        related_name='minute_data_5_bj',
        verbose_name='股票'
    )
    trade_time = models.DateTimeField(verbose_name='交易时间', db_index=True)
    open = models.FloatField(null=True, blank=True, verbose_name='开盘价')
    high = models.FloatField(null=True, blank=True, verbose_name='最高价')
    low = models.FloatField(null=True, blank=True, verbose_name='最低价')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    vol = models.BigIntegerField(verbose_name='成交量')
    amount = models.DecimalField(max_digits=20, decimal_places=3, verbose_name='成交金额')

    class Meta:
        verbose_name = 'A股分钟行情'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_minute_5_bj'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock} {self.trade_time}"

class StockMinuteData_15_BJ(models.Model):
    """A股分钟行情"""
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='stock_code',
        on_delete=models.CASCADE,
        related_name='minute_data_15_bj',
        verbose_name='股票'
    )
    trade_time = models.DateTimeField(verbose_name='交易时间', db_index=True)
    open = models.FloatField(null=True, blank=True, verbose_name='开盘价')
    high = models.FloatField(null=True, blank=True, verbose_name='最高价')
    low = models.FloatField(null=True, blank=True, verbose_name='最低价')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    vol = models.BigIntegerField(verbose_name='成交量')
    amount = models.DecimalField(max_digits=20, decimal_places=3, verbose_name='成交金额')

    class Meta:
        verbose_name = 'A股分钟行情'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_minute_15_bj'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock} {self.trade_time}"

class StockMinuteData_30_BJ(models.Model):
    """A股分钟行情"""
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='stock_code',
        on_delete=models.CASCADE,
        related_name='minute_data_30_bj',
        verbose_name='股票'
    )
    trade_time = models.DateTimeField(verbose_name='交易时间', db_index=True)
    open = models.FloatField(null=True, blank=True, verbose_name='开盘价')
    high = models.FloatField(null=True, blank=True, verbose_name='最高价')
    low = models.FloatField(null=True, blank=True, verbose_name='最低价')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    vol = models.BigIntegerField(verbose_name='成交量')
    amount = models.DecimalField(max_digits=20, decimal_places=3, verbose_name='成交金额')

    class Meta:
        verbose_name = 'A股分钟行情'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_minute_30_bj'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock} {self.trade_time}"

class StockMinuteData_60_BJ(models.Model):
    """A股分钟行情"""
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='stock_code',
        on_delete=models.CASCADE,
        related_name='minute_data_60_bj',
        verbose_name='股票'
    )
    trade_time = models.DateTimeField(verbose_name='交易时间', db_index=True)
    open = models.FloatField(null=True, blank=True, verbose_name='开盘价')
    high = models.FloatField(null=True, blank=True, verbose_name='最高价')
    low = models.FloatField(null=True, blank=True, verbose_name='最低价')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    vol = models.BigIntegerField(verbose_name='成交量')
    amount = models.DecimalField(max_digits=20, decimal_places=3, verbose_name='成交金额')

    class Meta:
        verbose_name = 'A股分钟行情'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_minute_60_bj'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock} {self.trade_time}"

class StockMinuteData_1_CY(models.Model):
    """A股分钟行情"""
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='stock_code',
        on_delete=models.CASCADE,
        related_name='minute_data_1_cy',
        verbose_name='股票'
    )
    trade_time = models.DateTimeField(verbose_name='交易时间', db_index=True)
    open = models.FloatField(null=True, blank=True, verbose_name='开盘价')
    high = models.FloatField(null=True, blank=True, verbose_name='最高价')
    low = models.FloatField(null=True, blank=True, verbose_name='最低价')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    vol = models.BigIntegerField(verbose_name='成交量')
    amount = models.DecimalField(max_digits=20, decimal_places=3, verbose_name='成交金额')

    class Meta:
        verbose_name = 'A股分钟行情'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_minute_1_cy'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock} {self.trade_time}"

class StockMinuteData_5_CY(models.Model):
    """A股分钟行情"""
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='stock_code',
        on_delete=models.CASCADE,
        related_name='minute_data_5_cy',
        verbose_name='股票'
    )
    trade_time = models.DateTimeField(verbose_name='交易时间', db_index=True)
    open = models.FloatField(null=True, blank=True, verbose_name='开盘价')
    high = models.FloatField(null=True, blank=True, verbose_name='最高价')
    low = models.FloatField(null=True, blank=True, verbose_name='最低价')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    vol = models.BigIntegerField(verbose_name='成交量')
    amount = models.DecimalField(max_digits=20, decimal_places=3, verbose_name='成交金额')

    class Meta:
        verbose_name = 'A股分钟行情'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_minute_5_cy'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock} {self.trade_time}"

class StockMinuteData_15_CY(models.Model):
    """A股分钟行情"""
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='stock_code',
        on_delete=models.CASCADE,
        related_name='minute_data_15_cy',
        verbose_name='股票'
    )
    trade_time = models.DateTimeField(verbose_name='交易时间', db_index=True)
    open = models.FloatField(null=True, blank=True, verbose_name='开盘价')
    high = models.FloatField(null=True, blank=True, verbose_name='最高价')
    low = models.FloatField(null=True, blank=True, verbose_name='最低价')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    vol = models.BigIntegerField(verbose_name='成交量')
    amount = models.DecimalField(max_digits=20, decimal_places=3, verbose_name='成交金额')

    class Meta:
        verbose_name = 'A股分钟行情'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_minute_15_cy'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock} {self.trade_time}"

class StockMinuteData_30_CY(models.Model):
    """A股分钟行情"""
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='stock_code',
        on_delete=models.CASCADE,
        related_name='minute_data_30_cy',
        verbose_name='股票'
    )
    trade_time = models.DateTimeField(verbose_name='交易时间', db_index=True)
    open = models.FloatField(null=True, blank=True, verbose_name='开盘价')
    high = models.FloatField(null=True, blank=True, verbose_name='最高价')
    low = models.FloatField(null=True, blank=True, verbose_name='最低价')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    vol = models.BigIntegerField(verbose_name='成交量')
    amount = models.DecimalField(max_digits=20, decimal_places=3, verbose_name='成交金额')

    class Meta:
        verbose_name = 'A股分钟行情'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_minute_30_cy'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock} {self.trade_time}"

class StockMinuteData_60_CY(models.Model):
    """A股分钟行情"""
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='stock_code',
        on_delete=models.CASCADE,
        related_name='minute_data_60_cy',
        verbose_name='股票'
    )
    trade_time = models.DateTimeField(verbose_name='交易时间', db_index=True)
    open = models.FloatField(null=True, blank=True, verbose_name='开盘价')
    high = models.FloatField(null=True, blank=True, verbose_name='最高价')
    low = models.FloatField(null=True, blank=True, verbose_name='最低价')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    vol = models.BigIntegerField(verbose_name='成交量')
    amount = models.DecimalField(max_digits=20, decimal_places=3, verbose_name='成交金额')

    class Meta:
        verbose_name = 'A股分钟行情'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_minute_60_cy'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock} {self.trade_time}"

class StockMinuteData_1_KC(models.Model):
    """A股分钟行情"""
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='stock_code',
        on_delete=models.CASCADE,
        related_name='minute_data_1_kc',
        verbose_name='股票'
    )
    trade_time = models.DateTimeField(verbose_name='交易时间', db_index=True)
    open = models.FloatField(null=True, blank=True, verbose_name='开盘价')
    high = models.FloatField(null=True, blank=True, verbose_name='最高价')
    low = models.FloatField(null=True, blank=True, verbose_name='最低价')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    vol = models.BigIntegerField(verbose_name='成交量')
    amount = models.DecimalField(max_digits=20, decimal_places=3, verbose_name='成交金额')

    class Meta:
        verbose_name = 'A股分钟行情'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_minute_1_kc'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock} {self.trade_time}"

class StockMinuteData_5_KC(models.Model):
    """A股分钟行情"""
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='stock_code',
        on_delete=models.CASCADE,
        related_name='minute_data_5_kc',
        verbose_name='股票'
    )
    trade_time = models.DateTimeField(verbose_name='交易时间', db_index=True)
    open = models.FloatField(null=True, blank=True, verbose_name='开盘价')
    high = models.FloatField(null=True, blank=True, verbose_name='最高价')
    low = models.FloatField(null=True, blank=True, verbose_name='最低价')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    vol = models.BigIntegerField(verbose_name='成交量')
    amount = models.DecimalField(max_digits=20, decimal_places=3, verbose_name='成交金额')

    class Meta:
        verbose_name = 'A股分钟行情'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_minute_5_kc'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock} {self.trade_time}"

class StockMinuteData_15_KC(models.Model):
    """A股分钟行情"""
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='stock_code',
        on_delete=models.CASCADE,
        related_name='minute_data_15_kc',
        verbose_name='股票'
    )
    trade_time = models.DateTimeField(verbose_name='交易时间', db_index=True)
    open = models.FloatField(null=True, blank=True, verbose_name='开盘价')
    high = models.FloatField(null=True, blank=True, verbose_name='最高价')
    low = models.FloatField(null=True, blank=True, verbose_name='最低价')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    vol = models.BigIntegerField(verbose_name='成交量')
    amount = models.DecimalField(max_digits=20, decimal_places=3, verbose_name='成交金额')

    class Meta:
        verbose_name = 'A股分钟行情'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_minute_15_kc'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock} {self.trade_time}"

class StockMinuteData_30_KC(models.Model):
    """A股分钟行情"""
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='stock_code',
        on_delete=models.CASCADE,
        related_name='minute_data_30_kc',
        verbose_name='股票'
    )
    trade_time = models.DateTimeField(verbose_name='交易时间', db_index=True)
    open = models.FloatField(null=True, blank=True, verbose_name='开盘价')
    high = models.FloatField(null=True, blank=True, verbose_name='最高价')
    low = models.FloatField(null=True, blank=True, verbose_name='最低价')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    vol = models.BigIntegerField(verbose_name='成交量')
    amount = models.DecimalField(max_digits=20, decimal_places=3, verbose_name='成交金额')

    class Meta:
        verbose_name = 'A股分钟行情'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_minute_30_kc'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock} {self.trade_time}"

class StockMinuteData_60_KC(models.Model):
    """A股分钟行情"""
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='stock_code',
        on_delete=models.CASCADE,
        related_name='minute_data_60_kc',
        verbose_name='股票'
    )
    trade_time = models.DateTimeField(verbose_name='交易时间', db_index=True)
    open = models.FloatField(null=True, blank=True, verbose_name='开盘价')
    high = models.FloatField(null=True, blank=True, verbose_name='最高价')
    low = models.FloatField(null=True, blank=True, verbose_name='最低价')
    close = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    vol = models.BigIntegerField(verbose_name='成交量')
    amount = models.DecimalField(max_digits=20, decimal_places=3, verbose_name='成交金额')

    class Meta:
        verbose_name = 'A股分钟行情'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_minute_60_kc'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock} {self.trade_time}"

# 周线行情模型（StockWeeklyData）
class StockWeeklyData(models.Model):
    """A股周线行情"""
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='stock_code',
        on_delete=models.CASCADE,
        related_name='weekly_data',
        verbose_name='股票'
    )
    trade_time = models.DateField(verbose_name='交易日期', db_index=True)
    open = models.FloatField(null=True, blank=True, verbose_name='周开盘价')
    high = models.FloatField(null=True, blank=True, verbose_name='周最高价')
    low = models.FloatField(null=True, blank=True, verbose_name='周最低价')
    close = models.FloatField(null=True, blank=True, verbose_name='周收盘价')
    pre_close = models.FloatField(null=True, blank=True, verbose_name='上一周收盘价')
    change = models.FloatField(null=True, blank=True, verbose_name='周涨跌额')
    pct_chg = models.DecimalField(null=True, blank=True, max_digits=6, decimal_places=2, verbose_name='周涨跌幅')
    vol = models.BigIntegerField(null=True, blank=True, verbose_name='周成交量')
    amount = models.DecimalField(null=True, blank=True, max_digits=20, decimal_places=3, verbose_name='周成交额')

    class Meta:
        verbose_name = 'A股周线行情'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_week'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock} {self.trade_time}"

# 月线行情模型（StockMonthlyData）
class StockMonthlyData(models.Model):
    """A股月线行情"""
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='stock_code',
        on_delete=models.CASCADE,
        related_name='monthly_data',
        verbose_name='股票'
    )
    trade_time = models.DateField(verbose_name='交易日期', db_index=True)
    open = models.FloatField(null=True, blank=True, verbose_name='月开盘价')
    high = models.FloatField(null=True, blank=True, verbose_name='月最高价')
    low = models.FloatField(null=True, blank=True, verbose_name='月最低价')
    close = models.FloatField(null=True, blank=True, verbose_name='月收盘价')
    pre_close = models.FloatField(null=True, blank=True, verbose_name='上月收盘价')
    change = models.FloatField(null=True, blank=True, verbose_name='月涨跌额')
    pct_chg = models.DecimalField(max_digits=6, decimal_places=2, verbose_name='月涨跌幅')
    vol = models.BigIntegerField(verbose_name='月成交量')
    amount = models.DecimalField(max_digits=20, decimal_places=3, verbose_name='月成交额')

    class Meta:
        verbose_name = 'A股月线行情'
        verbose_name_plural = verbose_name
        db_table = 'stock_time_trade_month'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']

    def __str__(self):
        return f"{self.stock} {self.trade_time}"

# 筹码分布模型（StockCyqChips）
class StockCyqChipsSZ(models.Model):
    """
    A股每日筹码分布模型
    """
    stock = models.ForeignKey(
        'StockInfo',  # 这里用字符串，避免循环引用
        to_field='stock_code',  # 指定外键对应StockInfo的哪个字段
        on_delete=models.CASCADE,
        verbose_name='stock_cyq_chips_sz',
        db_index=True
    )
    trade_time = models.DateField(verbose_name='交易日期', db_index=True)
    price = models.FloatField(verbose_name='成本价格')
    percent = models.FloatField(verbose_name='价格占比(%)')

    class Meta:
        verbose_name = '每日筹码分布SZ'
        verbose_name_plural = '每日筹码分布SZ'
        db_table = 'stock_cyq_chips_sz'
        unique_together = ('stock', 'trade_time', 'price')
        indexes = [
            models.Index(fields=['stock', 'trade_time']),
        ]

    def __str__(self):
        return f"{self.stock.stock_code} {self.trade_time} {self.price}"

# 筹码分布模型（StockCyqChips）
class StockCyqChipsSH(models.Model):
    """
    A股每日筹码分布模型
    """
    stock = models.ForeignKey(
        'StockInfo',  # 这里用字符串，避免循环引用
        to_field='stock_code',  # 指定外键对应StockInfo的哪个字段
        on_delete=models.CASCADE,
        verbose_name='stock_cyq_chips_sh',
        db_index=True
    )
    trade_time = models.DateField(verbose_name='交易日期', db_index=True)
    price = models.FloatField(verbose_name='成本价格')
    percent = models.FloatField(verbose_name='价格占比(%)')

    class Meta:
        verbose_name = '每日筹码分布SH'
        verbose_name_plural = '每日筹码分布SH'
        db_table = 'stock_cyq_chips_sh'
        unique_together = ('stock', 'trade_time', 'price')
        indexes = [
            models.Index(fields=['stock', 'trade_time']),
        ]

    def __str__(self):
        return f"{self.stock.stock_code} {self.trade_time} {self.price}"

# 筹码分布模型（StockCyqChips）
class StockCyqChipsCY(models.Model):
    """
    A股每日筹码分布模型
    """
    stock = models.ForeignKey(
        'StockInfo',  # 这里用字符串，避免循环引用
        to_field='stock_code',  # 指定外键对应StockInfo的哪个字段
        on_delete=models.CASCADE,
        verbose_name='stock_cyq_chips_cy',
        db_index=True
    )
    trade_time = models.DateField(verbose_name='交易日期', db_index=True)
    price = models.FloatField(verbose_name='成本价格')
    percent = models.FloatField(verbose_name='价格占比(%)')

    class Meta:
        verbose_name = '每日筹码分布CY'
        verbose_name_plural = '每日筹码分布CY'
        db_table = 'stock_cyq_chips_cy'
        unique_together = ('stock', 'trade_time', 'price')

    def __str__(self):
        return f"{self.stock.stock_code} {self.trade_time} {self.price}"

# 筹码分布模型（StockCyqChips）
class StockCyqChipsKC(models.Model):
    """
    A股每日筹码分布模型
    """
    stock = models.ForeignKey(
        'StockInfo',  # 这里用字符串，避免循环引用
        to_field='stock_code',  # 指定外键对应StockInfo的哪个字段
        on_delete=models.CASCADE,
        verbose_name='股票',
        db_index=True
    )
    trade_time = models.DateField(verbose_name='交易日期', db_index=True)
    price = models.FloatField(verbose_name='成本价格')
    percent = models.FloatField(verbose_name='价格占比(%)')

    class Meta:
        verbose_name = '每日筹码分布KC'
        verbose_name_plural = '每日筹码分布KC'
        db_table = 'stock_cyq_chips_kc'
        unique_together = ('stock', 'trade_time', 'price')

    def __str__(self):
        return f"{self.stock.stock_code} {self.trade_time} {self.price}"

# 筹码分布模型（StockCyqChips）
class StockCyqChipsBJ(models.Model):
    """
    A股每日筹码分布模型
    """
    stock = models.ForeignKey(
        'StockInfo',  # 这里用字符串，避免循环引用
        to_field='stock_code',  # 指定外键对应StockInfo的哪个字段
        on_delete=models.CASCADE,
        verbose_name='stock_cyq_chips_bj',
        db_index=True
    )
    trade_time = models.DateField(verbose_name='交易日期', db_index=True)
    price = models.FloatField(verbose_name='成本价格')
    percent = models.FloatField(verbose_name='价格占比(%)')

    class Meta:
        verbose_name = '每日筹码分布BJ'
        verbose_name_plural = '每日筹码分布BJ'
        db_table = 'stock_cyq_chips_bj'
        unique_together = ('stock', 'trade_time', 'price')

    def __str__(self):
        return f"{self.stock.stock_code} {self.trade_time} {self.price}"

# 筹码平均成本和胜率模型（StockCyqPerf）
class StockCyqPerf(models.Model):
    """
    A股每日筹码平均成本和胜率模型
    """
    stock = models.ForeignKey('StockInfo', to_field='stock_code', on_delete=models.CASCADE, verbose_name='股票', db_index=True)
    trade_time = models.DateField(verbose_name='交易日期', db_index=True)
    his_low = models.FloatField(verbose_name='历史最低价', null=True, blank=True)
    his_high = models.FloatField(verbose_name='历史最高价', null=True, blank=True)
    cost_5pct = models.FloatField(verbose_name='5分位成本', null=True, blank=True)
    cost_15pct = models.FloatField(verbose_name='15分位成本', null=True, blank=True)  # 数据样例未展示，接口有
    cost_50pct = models.FloatField(verbose_name='50分位成本', null=True, blank=True)  # 数据样例未展示，接口有
    cost_85pct = models.FloatField(verbose_name='85分位成本', null=True, blank=True)  # 数据样例未展示，接口有
    cost_95pct = models.FloatField(verbose_name='95分位成本', null=True, blank=True)
    weight_avg = models.FloatField(verbose_name='加权平均成本', null=True, blank=True)
    winner_rate = models.FloatField(verbose_name='胜率', null=True, blank=True)

    class Meta:
        verbose_name = '每日筹码及胜率'
        verbose_name_plural = '每日筹码及胜率'
        db_table = 'stock_cyq_perf'
        unique_together = ('stock', 'trade_time')

    def __str__(self):
        return f"{self.stock.stock_code} {self.trade_time}"

# 高级筹码指标模型
class AdvancedChipMetrics(models.Model):
    """
    【V5.4 精度与纯粹最终版 - 高级筹码指标模型】
    - 核心净化: 彻底移除了所有基于“大单净额”的、带有主观判断的衍生资金流指标，
                100%聚焦于最难被操纵的、基于筹码分布的客观事实。
    - 核心升维: 新增了基于“成交量微观结构”的指标，用于洞察成交量背后的多空力量对比。
    - 核心精度: 将所有 FloatField 升级为 DecimalField，确保金融数据和衍生指标的
                存储精度，从根本上杜绝浮点数误差和“数字尘埃”。
    """

    # --- 1. 核心关联键 ---
    stock = models.ForeignKey(
        'StockInfo',
        on_delete=models.CASCADE,
        related_name='advanced_chip_metrics',
        verbose_name='股票',
        db_index=True
    )
    trade_time = models.DateField(
        verbose_name='交易日期',
        db_index=True
    )

    # --- 2. 核心筹码峰基础指标 ---
    peak_cost = models.DecimalField(max_digits=12, decimal_places=4, verbose_name='主筹码峰成本', null=True, blank=True, help_text="当天筹码分布最密集的价格，是市场持仓的核心成本区。")
    peak_percent = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='主筹码峰占比(%)', null=True, blank=True, help_text="主筹码峰位置的筹码占总筹码的比例。")
    peak_volume = models.BigIntegerField(verbose_name='主筹码峰成交量(股)', null=True, blank=True, help_text="主筹码峰位置的绝对成交股数。")

    # --- 3. 筹码峰动态指标 ---
    peak_cost_slope_5d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本5日斜率')
    peak_cost_slope_8d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本8日斜率')
    peak_cost_slope_13d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本13日斜率')
    peak_cost_slope_21d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本21日斜率')
    peak_cost_slope_34d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本34日斜率')
    peak_cost_slope_55d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本55日斜率')
    peak_cost_slope_89d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本89日斜率')
    peak_cost_slope_144d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本144日斜率')
    peak_cost_accel_5d = models.DecimalField(max_digits=18, decimal_places=8, verbose_name='筹码峰成本5日加速度', null=True, blank=True)
    peak_cost_accel_21d = models.DecimalField(max_digits=18, decimal_places=8, verbose_name='筹码峰成本21日加速度', null=True, blank=True)

    # --- 4. 筹码结构与分布指标 ---
    concentration_90pct = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='90%筹码集中度', null=True, blank=True, help_text="值越小越集中。")
    concentration_90pct_slope_5d = models.DecimalField(max_digits=18, decimal_places=8, verbose_name='90%集中度5日斜率', null=True, blank=True, help_text="负值表示筹码趋于集中。")
    peak_stability = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='筹码峰稳定性', null=True, blank=True, help_text="值越大越稳定，代表主力控盘能力强。")
    is_multi_peak = models.BooleanField(verbose_name='是否多峰形态', default=False, help_text="持仓成本是否分散。")
    secondary_peak_cost = models.DecimalField(max_digits=12, decimal_places=4, verbose_name='次筹码峰成本', null=True, blank=True, help_text="潜在的压力或支撑位。")
    peak_distance_ratio = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='主次峰距离比', null=True, blank=True, help_text="距离越远，结构越不稳定。")
    peak_strength_ratio = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='主次峰强度比', null=True, blank=True, help_text="比率越小，主峰的统治力越强。")
    pressure_above = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='上方2%套牢盘(%)', null=True, blank=True, help_text="代表直接的短期抛压。")
    support_below = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='下方2%支撑盘(%)', null=True, blank=True, help_text="代表直接的短期支撑。")

    # --- 5. 获利盘结构指标 ---
    total_winner_rate = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='总获利盘(%)', null=True, blank=True, help_text="反映市场整体情绪。")
    winner_rate_short_term = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='短期获利盘(%)', null=True, blank=True, help_text="代表近期追涨资金的浮盈情况。")
    winner_rate_long_term = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='长期锁定盘(%)', null=True, blank=True, help_text="代表坚定持有的资金。")

    # --- 6. 辅助与过程指标 ---
    pressure_above_volume = models.BigIntegerField(verbose_name='上方套牢盘绝对量(股)', null=True, blank=True)
    support_below_volume = models.BigIntegerField(verbose_name='下方支撑盘绝对量(股)', null=True, blank=True)
    turnover_volume_in_cost_range_70pct = models.BigIntegerField(verbose_name='70%成本区换手量(股)', null=True, blank=True)
    prev_20d_close = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name='20日前收盘价')
    
    # --- 7. 【升维】控盘度指标 ---
    peak_control_ratio = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='筹码峰控盘比(%)', null=True, blank=True, help_text="主筹码峰股数 / 流通股本。")
    peak_absorption_intensity = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='筹码峰吸筹强度', null=True, blank=True, help_text="主峰区间换手量 / 总换手量。")

    # --- 8. 【升维】利润质量指标 ---
    winner_avg_cost = models.DecimalField(max_digits=12, decimal_places=4, verbose_name='获利盘平均成本', null=True, blank=True)
    winner_profit_margin = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='获利盘安全垫(%)', null=True, blank=True, help_text="衡量获利盘的平均利润厚度。")

    # --- 9. 【升维】价码关系指标 ---
    price_to_peak_ratio = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='股价/筹码峰成本比', null=True, blank=True)
    chip_zscore = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='筹码Z-Score', null=True, blank=True, help_text="股价在筹码分布中的标准分位置。")

    # --- 10. 【升维】筹码断层指标 ---
    chip_fault_strength = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='筹码断层强度', null=True, blank=True)
    chip_fault_vacuum_percent = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='断层真空区筹码占比(%)', null=True, blank=True)
    is_chip_fault_formed = models.BooleanField(verbose_name='是否形成筹码断层', default=False, help_text="极强的看涨信号。")

    # --- 11. 【超级指标】最终裁决 ---
    chip_health_score = models.DecimalField(max_digits=8, decimal_places=2, verbose_name='筹码健康分(0-100)', null=True, blank=True)

    # --- 12. 【升维】成交量微观结构指标 ---
    turnover_at_peak_ratio = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='主峰成交占比(%)', null=True, blank=True, help_text="主峰区间的交战激烈程度。")
    turnover_from_winners_ratio = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='获利盘抛压占比(%)', null=True, blank=True, help_text="短期抛售压力大小。")
    turnover_from_losers_ratio = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='套牢盘割肉占比(%)', null=True, blank=True, help_text="恐慌/割肉盘轻重。")

    class Meta:
        verbose_name = '高级筹码指标(精度升级版)'
        verbose_name_plural = verbose_name
        db_table = 'stock_advanced_chip_metrics'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_time']),
            models.Index(fields=['chip_health_score']),
            models.Index(fields=['is_chip_fault_formed']),
        ]

    def __str__(self):
        return f"{self.stock.stock_code} - {self.trade_time}"

# 高级筹码指标模型
class AdvancedChipMetrics_SZ(models.Model):
    """
    【V5.4 精度与纯粹最终版 - 高级筹码指标模型】
    - 核心净化: 彻底移除了所有基于“大单净额”的、带有主观判断的衍生资金流指标，
                100%聚焦于最难被操纵的、基于筹码分布的客观事实。
    - 核心升维: 新增了基于“成交量微观结构”的指标，用于洞察成交量背后的多空力量对比。
    - 核心精度: 将所有 FloatField 升级为 DecimalField，确保金融数据和衍生指标的
                存储精度，从根本上杜绝浮点数误差和“数字尘埃”。
    """

    # --- 1. 核心关联键 ---
    stock = models.ForeignKey(
        'StockInfo',
        on_delete=models.CASCADE,
        related_name='advanced_chip_metrics_sz',

        verbose_name='股票',
        db_index=True
    )
    trade_time = models.DateField(
        verbose_name='交易日期',
        db_index=True
    )

    # --- 2. 核心筹码峰基础指标 ---
    peak_cost = models.DecimalField(max_digits=12, decimal_places=4, verbose_name='主筹码峰成本', null=True, blank=True, help_text="当天筹码分布最密集的价格，是市场持仓的核心成本区。")
    peak_percent = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='主筹码峰占比(%)', null=True, blank=True, help_text="主筹码峰位置的筹码占总筹码的比例。")
    peak_volume = models.BigIntegerField(verbose_name='主筹码峰成交量(股)', null=True, blank=True, help_text="主筹码峰位置的绝对成交股数。")

    # --- 3. 筹码峰动态指标 ---
    peak_cost_slope_5d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本5日斜率')
    peak_cost_slope_8d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本8日斜率')
    peak_cost_slope_13d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本13日斜率')
    peak_cost_slope_21d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本21日斜率')
    peak_cost_slope_34d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本34日斜率')
    peak_cost_slope_55d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本55日斜率')
    peak_cost_slope_89d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本89日斜率')
    peak_cost_slope_144d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本144日斜率')
    peak_cost_accel_5d = models.DecimalField(max_digits=18, decimal_places=8, verbose_name='筹码峰成本5日加速度', null=True, blank=True)
    peak_cost_accel_21d = models.DecimalField(max_digits=18, decimal_places=8, verbose_name='筹码峰成本21日加速度', null=True, blank=True)

    # --- 4. 筹码结构与分布指标 ---
    concentration_70pct = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='70%筹码集中度', null=True, blank=True, help_text="值越小越集中。")
    concentration_70pct_slope_5d = models.DecimalField(max_digits=18, decimal_places=8, verbose_name='70%集中度5日斜率', null=True, blank=True, help_text="负值表示筹码趋于集中。")
    concentration_90pct = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='90%筹码集中度', null=True, blank=True, help_text="值越小越集中。")
    concentration_90pct_slope_5d = models.DecimalField(max_digits=18, decimal_places=8, verbose_name='90%集中度5日斜率', null=True, blank=True, help_text="负值表示筹码趋于集中。")
    concentration_90pct_slope_21d = models.DecimalField(max_digits=18, decimal_places=8, verbose_name='90%集中度21日斜率', null=True, blank=True, help_text="中期趋势。")
    concentration_90pct_slope_55d = models.DecimalField(max_digits=18, decimal_places=8, verbose_name='90%集中度55日斜率', null=True, blank=True, help_text="长期趋势。")
    peak_stability = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='筹码峰稳定性', null=True, blank=True, help_text="值越大越稳定，代表主力控盘能力强。")
    is_multi_peak = models.BooleanField(verbose_name='是否多峰形态', default=False, help_text="持仓成本是否分散。")
    secondary_peak_cost = models.DecimalField(max_digits=12, decimal_places=4, verbose_name='次筹码峰成本', null=True, blank=True, help_text="潜在的压力或支撑位。")
    peak_distance_ratio = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='主次峰距离比', null=True, blank=True, help_text="距离越远，结构越不稳定。")
    peak_strength_ratio = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='主次峰强度比', null=True, blank=True, help_text="比率越小，主峰的统治力越强。")
    pressure_above = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='上方2%套牢盘(%)', null=True, blank=True, help_text="代表直接的短期抛压。")
    support_below = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='下方2%支撑盘(%)', null=True, blank=True, help_text="代表直接的短期支撑。")

    # --- 5. 获利盘结构指标 ---
    total_winner_rate = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='总获利盘(%)', null=True, blank=True, help_text="反映市场整体情绪。")
    winner_rate_short_term = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='短期获利盘(%)', null=True, blank=True, help_text="代表近期追涨资金的浮盈情况。")
    winner_rate_long_term = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='长期锁定盘(%)', null=True, blank=True, help_text="代表坚定持有的资金。")

    # --- 6. 辅助与过程指标 ---
    pressure_above_volume = models.BigIntegerField(verbose_name='上方套牢盘绝对量(股)', null=True, blank=True)
    support_below_volume = models.BigIntegerField(verbose_name='下方支撑盘绝对量(股)', null=True, blank=True)
    turnover_volume_in_cost_range_70pct = models.BigIntegerField(verbose_name='70%成本区换手量(股)', null=True, blank=True)
    prev_20d_close = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name='20日前收盘价')
    
    # --- 7. 【升维】控盘度指标 ---
    peak_control_ratio = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='筹码峰控盘比(%)', null=True, blank=True, help_text="主筹码峰股数 / 流通股本。")
    peak_absorption_intensity = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='筹码峰吸筹强度', null=True, blank=True, help_text="主峰区间换手量 / 总换手量。")

    # --- 8. 【升维】利润质量指标 ---
    winner_avg_cost = models.DecimalField(max_digits=12, decimal_places=4, verbose_name='获利盘平均成本', null=True, blank=True)
    winner_profit_margin = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='获利盘安全垫(%)', null=True, blank=True, help_text="衡量获利盘的平均利润厚度。")

    # --- 9. 【升维】价码关系指标 ---
    price_to_peak_ratio = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='股价/筹码峰成本比', null=True, blank=True)
    chip_zscore = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='筹码Z-Score', null=True, blank=True, help_text="股价在筹码分布中的标准分位置。")

    # --- 10. 【升维】筹码断层指标 ---
    chip_fault_strength = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='筹码断层强度', null=True, blank=True)
    chip_fault_vacuum_percent = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='断层真空区筹码占比(%)', null=True, blank=True)
    is_chip_fault_formed = models.BooleanField(verbose_name='是否形成筹码断层', default=False, help_text="极强的看涨信号。")

    # --- 11. 【超级指标】最终裁决 ---
    chip_health_score = models.DecimalField(max_digits=8, decimal_places=2, verbose_name='筹码健康分(0-100)', null=True, blank=True)

    # --- 12. 【升维】成交量微观结构指标 ---
    turnover_at_peak_ratio = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='主峰成交占比(%)', null=True, blank=True, help_text="主峰区间的交战激烈程度。")
    turnover_from_winners_ratio = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='获利盘抛压占比(%)', null=True, blank=True, help_text="短期抛售压力大小。")
    turnover_from_losers_ratio = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='套牢盘割肉占比(%)', null=True, blank=True, help_text="恐慌/割肉盘轻重。")

    class Meta:
        verbose_name = '高级筹码指标(精度升级版)'
        verbose_name_plural = verbose_name
        db_table = 'stock_advanced_chip_metrics_sz'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_time']),
            models.Index(fields=['chip_health_score']),
            models.Index(fields=['is_chip_fault_formed']),
        ]

    def __str__(self):
        return f"{self.stock.stock_code} - {self.trade_time}"

# 高级筹码指标模型
class AdvancedChipMetrics_SH(models.Model):
    """
    【V5.4 精度与纯粹最终版 - 高级筹码指标模型】
    - 核心净化: 彻底移除了所有基于“大单净额”的、带有主观判断的衍生资金流指标，
                100%聚焦于最难被操纵的、基于筹码分布的客观事实。
    - 核心升维: 新增了基于“成交量微观结构”的指标，用于洞察成交量背后的多空力量对比。
    - 核心精度: 将所有 FloatField 升级为 DecimalField，确保金融数据和衍生指标的
                存储精度，从根本上杜绝浮点数误差和“数字尘埃”。
    """

    # --- 1. 核心关联键 ---
    stock = models.ForeignKey(
        'StockInfo',
        on_delete=models.CASCADE,
        related_name='advanced_chip_metrics_sh',
        verbose_name='股票',
        db_index=True
    )
    trade_time = models.DateField(
        verbose_name='交易日期',
        db_index=True
    )

    # --- 2. 核心筹码峰基础指标 ---
    peak_cost = models.DecimalField(max_digits=12, decimal_places=4, verbose_name='主筹码峰成本', null=True, blank=True, help_text="当天筹码分布最密集的价格，是市场持仓的核心成本区。")
    peak_percent = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='主筹码峰占比(%)', null=True, blank=True, help_text="主筹码峰位置的筹码占总筹码的比例。")
    peak_volume = models.BigIntegerField(verbose_name='主筹码峰成交量(股)', null=True, blank=True, help_text="主筹码峰位置的绝对成交股数。")

    # --- 3. 筹码峰动态指标 ---
    peak_cost_slope_5d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本5日斜率')
    peak_cost_slope_8d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本8日斜率')
    peak_cost_slope_13d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本13日斜率')
    peak_cost_slope_21d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本21日斜率')
    peak_cost_slope_34d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本34日斜率')
    peak_cost_slope_55d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本55日斜率')
    peak_cost_slope_89d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本89日斜率')
    peak_cost_slope_144d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本144日斜率')
    peak_cost_accel_5d = models.DecimalField(max_digits=18, decimal_places=8, verbose_name='筹码峰成本5日加速度', null=True, blank=True)
    peak_cost_accel_21d = models.DecimalField(max_digits=18, decimal_places=8, verbose_name='筹码峰成本21日加速度', null=True, blank=True)

    # --- 4. 筹码结构与分布指标 ---
    concentration_70pct = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='70%筹码集中度', null=True, blank=True, help_text="值越小越集中。")
    concentration_70pct_slope_5d = models.DecimalField(max_digits=18, decimal_places=8, verbose_name='70%集中度5日斜率', null=True, blank=True, help_text="负值表示筹码趋于集中。")
    concentration_90pct = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='90%筹码集中度', null=True, blank=True, help_text="值越小越集中。")
    concentration_90pct_slope_5d = models.DecimalField(max_digits=18, decimal_places=8, verbose_name='90%集中度5日斜率', null=True, blank=True, help_text="负值表示筹码趋于集中。")
    concentration_90pct_slope_21d = models.DecimalField(max_digits=18, decimal_places=8, verbose_name='90%集中度21日斜率', null=True, blank=True, help_text="中期趋势。")
    concentration_90pct_slope_55d = models.DecimalField(max_digits=18, decimal_places=8, verbose_name='90%集中度55日斜率', null=True, blank=True, help_text="长期趋势。")
    peak_stability = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='筹码峰稳定性', null=True, blank=True, help_text="值越大越稳定，代表主力控盘能力强。")
    is_multi_peak = models.BooleanField(verbose_name='是否多峰形态', default=False, help_text="持仓成本是否分散。")
    secondary_peak_cost = models.DecimalField(max_digits=12, decimal_places=4, verbose_name='次筹码峰成本', null=True, blank=True, help_text="潜在的压力或支撑位。")
    peak_distance_ratio = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='主次峰距离比', null=True, blank=True, help_text="距离越远，结构越不稳定。")
    peak_strength_ratio = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='主次峰强度比', null=True, blank=True, help_text="比率越小，主峰的统治力越强。")
    pressure_above = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='上方2%套牢盘(%)', null=True, blank=True, help_text="代表直接的短期抛压。")
    support_below = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='下方2%支撑盘(%)', null=True, blank=True, help_text="代表直接的短期支撑。")

    # --- 5. 获利盘结构指标 ---
    total_winner_rate = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='总获利盘(%)', null=True, blank=True, help_text="反映市场整体情绪。")
    winner_rate_short_term = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='短期获利盘(%)', null=True, blank=True, help_text="代表近期追涨资金的浮盈情况。")
    winner_rate_long_term = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='长期锁定盘(%)', null=True, blank=True, help_text="代表坚定持有的资金。")

    # --- 6. 辅助与过程指标 ---
    pressure_above_volume = models.BigIntegerField(verbose_name='上方套牢盘绝对量(股)', null=True, blank=True)
    support_below_volume = models.BigIntegerField(verbose_name='下方支撑盘绝对量(股)', null=True, blank=True)
    turnover_volume_in_cost_range_70pct = models.BigIntegerField(verbose_name='70%成本区换手量(股)', null=True, blank=True)
    prev_20d_close = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name='20日前收盘价')
    
    # --- 7. 【升维】控盘度指标 ---
    peak_control_ratio = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='筹码峰控盘比(%)', null=True, blank=True, help_text="主筹码峰股数 / 流通股本。")
    peak_absorption_intensity = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='筹码峰吸筹强度', null=True, blank=True, help_text="主峰区间换手量 / 总换手量。")

    # --- 8. 【升维】利润质量指标 ---
    winner_avg_cost = models.DecimalField(max_digits=12, decimal_places=4, verbose_name='获利盘平均成本', null=True, blank=True)
    winner_profit_margin = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='获利盘安全垫(%)', null=True, blank=True, help_text="衡量获利盘的平均利润厚度。")

    # --- 9. 【升维】价码关系指标 ---
    price_to_peak_ratio = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='股价/筹码峰成本比', null=True, blank=True)
    chip_zscore = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='筹码Z-Score', null=True, blank=True, help_text="股价在筹码分布中的标准分位置。")

    # --- 10. 【升维】筹码断层指标 ---
    chip_fault_strength = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='筹码断层强度', null=True, blank=True)
    chip_fault_vacuum_percent = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='断层真空区筹码占比(%)', null=True, blank=True)
    is_chip_fault_formed = models.BooleanField(verbose_name='是否形成筹码断层', default=False, help_text="极强的看涨信号。")

    # --- 11. 【超级指标】最终裁决 ---
    chip_health_score = models.DecimalField(max_digits=8, decimal_places=2, verbose_name='筹码健康分(0-100)', null=True, blank=True)

    # --- 12. 【升维】成交量微观结构指标 ---
    turnover_at_peak_ratio = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='主峰成交占比(%)', null=True, blank=True, help_text="主峰区间的交战激烈程度。")
    turnover_from_winners_ratio = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='获利盘抛压占比(%)', null=True, blank=True, help_text="短期抛售压力大小。")
    turnover_from_losers_ratio = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='套牢盘割肉占比(%)', null=True, blank=True, help_text="恐慌/割肉盘轻重。")

    class Meta:
        verbose_name = '高级筹码指标(精度升级版)'
        verbose_name_plural = verbose_name
        db_table = 'stock_advanced_chip_metrics_sh'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_time']),
            models.Index(fields=['chip_health_score']),
            models.Index(fields=['is_chip_fault_formed']),
        ]

    def __str__(self):
        return f"{self.stock.stock_code} - {self.trade_time}"

# 高级筹码指标模型
class AdvancedChipMetrics_CY(models.Model):
    """
    【V5.4 精度与纯粹最终版 - 高级筹码指标模型】
    - 核心净化: 彻底移除了所有基于“大单净额”的、带有主观判断的衍生资金流指标，
                100%聚焦于最难被操纵的、基于筹码分布的客观事实。
    - 核心升维: 新增了基于“成交量微观结构”的指标，用于洞察成交量背后的多空力量对比。
    - 核心精度: 将所有 FloatField 升级为 DecimalField，确保金融数据和衍生指标的
                存储精度，从根本上杜绝浮点数误差和“数字尘埃”。
    """

    # --- 1. 核心关联键 ---
    stock = models.ForeignKey(
        'StockInfo',
        on_delete=models.CASCADE,
        related_name='advanced_chip_metrics_cy',
        verbose_name='股票',
        db_index=True
    )
    trade_time = models.DateField(
        verbose_name='交易日期',
        db_index=True
    )

    # --- 2. 核心筹码峰基础指标 ---
    peak_cost = models.DecimalField(max_digits=12, decimal_places=4, verbose_name='主筹码峰成本', null=True, blank=True, help_text="当天筹码分布最密集的价格，是市场持仓的核心成本区。")
    peak_percent = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='主筹码峰占比(%)', null=True, blank=True, help_text="主筹码峰位置的筹码占总筹码的比例。")
    peak_volume = models.BigIntegerField(verbose_name='主筹码峰成交量(股)', null=True, blank=True, help_text="主筹码峰位置的绝对成交股数。")

    # --- 3. 筹码峰动态指标 ---
    peak_cost_slope_5d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本5日斜率')
    peak_cost_slope_8d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本8日斜率')
    peak_cost_slope_13d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本13日斜率')
    peak_cost_slope_21d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本21日斜率')
    peak_cost_slope_34d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本34日斜率')
    peak_cost_slope_55d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本55日斜率')
    peak_cost_slope_89d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本89日斜率')
    peak_cost_slope_144d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本144日斜率')
    peak_cost_accel_5d = models.DecimalField(max_digits=18, decimal_places=8, verbose_name='筹码峰成本5日加速度', null=True, blank=True)
    peak_cost_accel_21d = models.DecimalField(max_digits=18, decimal_places=8, verbose_name='筹码峰成本21日加速度', null=True, blank=True)

    # --- 4. 筹码结构与分布指标 ---
    concentration_70pct = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='70%筹码集中度', null=True, blank=True, help_text="值越小越集中。")
    concentration_70pct_slope_5d = models.DecimalField(max_digits=18, decimal_places=8, verbose_name='70%集中度5日斜率', null=True, blank=True, help_text="负值表示筹码趋于集中。")
    concentration_90pct = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='90%筹码集中度', null=True, blank=True, help_text="值越小越集中。")
    concentration_90pct_slope_5d = models.DecimalField(max_digits=18, decimal_places=8, verbose_name='90%集中度5日斜率', null=True, blank=True, help_text="负值表示筹码趋于集中。")
    concentration_90pct_slope_21d = models.DecimalField(max_digits=18, decimal_places=8, verbose_name='90%集中度21日斜率', null=True, blank=True, help_text="中期趋势。")
    concentration_90pct_slope_55d = models.DecimalField(max_digits=18, decimal_places=8, verbose_name='90%集中度55日斜率', null=True, blank=True, help_text="长期趋势。")
    peak_stability = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='筹码峰稳定性', null=True, blank=True, help_text="值越大越稳定，代表主力控盘能力强。")
    is_multi_peak = models.BooleanField(verbose_name='是否多峰形态', default=False, help_text="持仓成本是否分散。")
    secondary_peak_cost = models.DecimalField(max_digits=12, decimal_places=4, verbose_name='次筹码峰成本', null=True, blank=True, help_text="潜在的压力或支撑位。")
    peak_distance_ratio = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='主次峰距离比', null=True, blank=True, help_text="距离越远，结构越不稳定。")
    peak_strength_ratio = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='主次峰强度比', null=True, blank=True, help_text="比率越小，主峰的统治力越强。")
    pressure_above = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='上方2%套牢盘(%)', null=True, blank=True, help_text="代表直接的短期抛压。")
    support_below = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='下方2%支撑盘(%)', null=True, blank=True, help_text="代表直接的短期支撑。")

    # --- 5. 获利盘结构指标 ---
    total_winner_rate = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='总获利盘(%)', null=True, blank=True, help_text="反映市场整体情绪。")
    winner_rate_short_term = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='短期获利盘(%)', null=True, blank=True, help_text="代表近期追涨资金的浮盈情况。")
    winner_rate_long_term = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='长期锁定盘(%)', null=True, blank=True, help_text="代表坚定持有的资金。")

    # --- 6. 辅助与过程指标 ---
    pressure_above_volume = models.BigIntegerField(verbose_name='上方套牢盘绝对量(股)', null=True, blank=True)
    support_below_volume = models.BigIntegerField(verbose_name='下方支撑盘绝对量(股)', null=True, blank=True)
    turnover_volume_in_cost_range_70pct = models.BigIntegerField(verbose_name='70%成本区换手量(股)', null=True, blank=True)
    prev_20d_close = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name='20日前收盘价')
    
    # --- 7. 【升维】控盘度指标 ---
    peak_control_ratio = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='筹码峰控盘比(%)', null=True, blank=True, help_text="主筹码峰股数 / 流通股本。")
    peak_absorption_intensity = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='筹码峰吸筹强度', null=True, blank=True, help_text="主峰区间换手量 / 总换手量。")

    # --- 8. 【升维】利润质量指标 ---
    winner_avg_cost = models.DecimalField(max_digits=12, decimal_places=4, verbose_name='获利盘平均成本', null=True, blank=True)
    winner_profit_margin = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='获利盘安全垫(%)', null=True, blank=True, help_text="衡量获利盘的平均利润厚度。")

    # --- 9. 【升维】价码关系指标 ---
    price_to_peak_ratio = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='股价/筹码峰成本比', null=True, blank=True)
    chip_zscore = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='筹码Z-Score', null=True, blank=True, help_text="股价在筹码分布中的标准分位置。")

    # --- 10. 【升维】筹码断层指标 ---
    chip_fault_strength = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='筹码断层强度', null=True, blank=True)
    chip_fault_vacuum_percent = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='断层真空区筹码占比(%)', null=True, blank=True)
    is_chip_fault_formed = models.BooleanField(verbose_name='是否形成筹码断层', default=False, help_text="极强的看涨信号。")

    # --- 11. 【超级指标】最终裁决 ---
    chip_health_score = models.DecimalField(max_digits=8, decimal_places=2, verbose_name='筹码健康分(0-100)', null=True, blank=True)

    # --- 12. 【升维】成交量微观结构指标 ---
    turnover_at_peak_ratio = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='主峰成交占比(%)', null=True, blank=True, help_text="主峰区间的交战激烈程度。")
    turnover_from_winners_ratio = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='获利盘抛压占比(%)', null=True, blank=True, help_text="短期抛售压力大小。")
    turnover_from_losers_ratio = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='套牢盘割肉占比(%)', null=True, blank=True, help_text="恐慌/割肉盘轻重。")

    class Meta:
        verbose_name = '高级筹码指标(精度升级版)'
        verbose_name_plural = verbose_name
        db_table = 'stock_advanced_chip_metrics_cy'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_time']),
            models.Index(fields=['chip_health_score']),
            models.Index(fields=['is_chip_fault_formed']),
        ]

    def __str__(self):
        return f"{self.stock.stock_code} - {self.trade_time}"

# 高级筹码指标模型
class AdvancedChipMetrics_KC(models.Model):
    """
    【V5.4 精度与纯粹最终版 - 高级筹码指标模型】
    - 核心净化: 彻底移除了所有基于“大单净额”的、带有主观判断的衍生资金流指标，
                100%聚焦于最难被操纵的、基于筹码分布的客观事实。
    - 核心升维: 新增了基于“成交量微观结构”的指标，用于洞察成交量背后的多空力量对比。
    - 核心精度: 将所有 FloatField 升级为 DecimalField，确保金融数据和衍生指标的
                存储精度，从根本上杜绝浮点数误差和“数字尘埃”。
    """

    # --- 1. 核心关联键 ---
    stock = models.ForeignKey(
        'StockInfo',
        on_delete=models.CASCADE,
        related_name='advanced_chip_metrics_kc',
        verbose_name='股票',
        db_index=True
    )
    trade_time = models.DateField(
        verbose_name='交易日期',
        db_index=True
    )

    # --- 2. 核心筹码峰基础指标 ---
    peak_cost = models.DecimalField(max_digits=12, decimal_places=4, verbose_name='主筹码峰成本', null=True, blank=True, help_text="当天筹码分布最密集的价格，是市场持仓的核心成本区。")
    peak_percent = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='主筹码峰占比(%)', null=True, blank=True, help_text="主筹码峰位置的筹码占总筹码的比例。")
    peak_volume = models.BigIntegerField(verbose_name='主筹码峰成交量(股)', null=True, blank=True, help_text="主筹码峰位置的绝对成交股数。")

    # --- 3. 筹码峰动态指标 ---
    peak_cost_slope_5d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本5日斜率')
    peak_cost_slope_8d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本8日斜率')
    peak_cost_slope_13d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本13日斜率')
    peak_cost_slope_21d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本21日斜率')
    peak_cost_slope_34d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本34日斜率')
    peak_cost_slope_55d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本55日斜率')
    peak_cost_slope_89d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本89日斜率')
    peak_cost_slope_144d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本144日斜率')
    peak_cost_accel_5d = models.DecimalField(max_digits=18, decimal_places=8, verbose_name='筹码峰成本5日加速度', null=True, blank=True)
    peak_cost_accel_21d = models.DecimalField(max_digits=18, decimal_places=8, verbose_name='筹码峰成本21日加速度', null=True, blank=True)

    # --- 4. 筹码结构与分布指标 ---
    concentration_70pct = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='70%筹码集中度', null=True, blank=True, help_text="值越小越集中。")
    concentration_70pct_slope_5d = models.DecimalField(max_digits=18, decimal_places=8, verbose_name='70%集中度5日斜率', null=True, blank=True, help_text="负值表示筹码趋于集中。")
    concentration_90pct = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='90%筹码集中度', null=True, blank=True, help_text="值越小越集中。")
    concentration_90pct_slope_5d = models.DecimalField(max_digits=18, decimal_places=8, verbose_name='90%集中度5日斜率', null=True, blank=True, help_text="负值表示筹码趋于集中。")
    concentration_90pct_slope_21d = models.DecimalField(max_digits=18, decimal_places=8, verbose_name='90%集中度21日斜率', null=True, blank=True, help_text="中期趋势。")
    concentration_90pct_slope_55d = models.DecimalField(max_digits=18, decimal_places=8, verbose_name='90%集中度55日斜率', null=True, blank=True, help_text="长期趋势。")
    peak_stability = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='筹码峰稳定性', null=True, blank=True, help_text="值越大越稳定，代表主力控盘能力强。")
    is_multi_peak = models.BooleanField(verbose_name='是否多峰形态', default=False, help_text="持仓成本是否分散。")
    secondary_peak_cost = models.DecimalField(max_digits=12, decimal_places=4, verbose_name='次筹码峰成本', null=True, blank=True, help_text="潜在的压力或支撑位。")
    peak_distance_ratio = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='主次峰距离比', null=True, blank=True, help_text="距离越远，结构越不稳定。")
    peak_strength_ratio = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='主次峰强度比', null=True, blank=True, help_text="比率越小，主峰的统治力越强。")
    pressure_above = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='上方2%套牢盘(%)', null=True, blank=True, help_text="代表直接的短期抛压。")
    support_below = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='下方2%支撑盘(%)', null=True, blank=True, help_text="代表直接的短期支撑。")

    # --- 5. 获利盘结构指标 ---
    total_winner_rate = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='总获利盘(%)', null=True, blank=True, help_text="反映市场整体情绪。")
    winner_rate_short_term = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='短期获利盘(%)', null=True, blank=True, help_text="代表近期追涨资金的浮盈情况。")
    winner_rate_long_term = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='长期锁定盘(%)', null=True, blank=True, help_text="代表坚定持有的资金。")

    # --- 6. 辅助与过程指标 ---
    pressure_above_volume = models.BigIntegerField(verbose_name='上方套牢盘绝对量(股)', null=True, blank=True)
    support_below_volume = models.BigIntegerField(verbose_name='下方支撑盘绝对量(股)', null=True, blank=True)
    turnover_volume_in_cost_range_70pct = models.BigIntegerField(verbose_name='70%成本区换手量(股)', null=True, blank=True)
    prev_20d_close = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name='20日前收盘价')
    
    # --- 7. 【升维】控盘度指标 ---
    peak_control_ratio = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='筹码峰控盘比(%)', null=True, blank=True, help_text="主筹码峰股数 / 流通股本。")
    peak_absorption_intensity = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='筹码峰吸筹强度', null=True, blank=True, help_text="主峰区间换手量 / 总换手量。")

    # --- 8. 【升维】利润质量指标 ---
    winner_avg_cost = models.DecimalField(max_digits=12, decimal_places=4, verbose_name='获利盘平均成本', null=True, blank=True)
    winner_profit_margin = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='获利盘安全垫(%)', null=True, blank=True, help_text="衡量获利盘的平均利润厚度。")

    # --- 9. 【升维】价码关系指标 ---
    price_to_peak_ratio = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='股价/筹码峰成本比', null=True, blank=True)
    chip_zscore = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='筹码Z-Score', null=True, blank=True, help_text="股价在筹码分布中的标准分位置。")

    # --- 10. 【升维】筹码断层指标 ---
    chip_fault_strength = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='筹码断层强度', null=True, blank=True)
    chip_fault_vacuum_percent = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='断层真空区筹码占比(%)', null=True, blank=True)
    is_chip_fault_formed = models.BooleanField(verbose_name='是否形成筹码断层', default=False, help_text="极强的看涨信号。")

    # --- 11. 【超级指标】最终裁决 ---
    chip_health_score = models.DecimalField(max_digits=8, decimal_places=2, verbose_name='筹码健康分(0-100)', null=True, blank=True)

    # --- 12. 【升维】成交量微观结构指标 ---
    turnover_at_peak_ratio = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='主峰成交占比(%)', null=True, blank=True, help_text="主峰区间的交战激烈程度。")
    turnover_from_winners_ratio = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='获利盘抛压占比(%)', null=True, blank=True, help_text="短期抛售压力大小。")
    turnover_from_losers_ratio = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='套牢盘割肉占比(%)', null=True, blank=True, help_text="恐慌/割肉盘轻重。")

    class Meta:
        verbose_name = '高级筹码指标(精度升级版)'
        verbose_name_plural = verbose_name
        db_table = 'stock_advanced_chip_metrics_kc'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_time']),
            models.Index(fields=['chip_health_score']),
            models.Index(fields=['is_chip_fault_formed']),
        ]

    def __str__(self):
        return f"{self.stock.stock_code} - {self.trade_time}"

# 高级筹码指标模型
class AdvancedChipMetrics_BJ(models.Model):
    """
    【V5.4 精度与纯粹最终版 - 高级筹码指标模型】
    - 核心净化: 彻底移除了所有基于“大单净额”的、带有主观判断的衍生资金流指标，
                100%聚焦于最难被操纵的、基于筹码分布的客观事实。
    - 核心升维: 新增了基于“成交量微观结构”的指标，用于洞察成交量背后的多空力量对比。
    - 核心精度: 将所有 FloatField 升级为 DecimalField，确保金融数据和衍生指标的
                存储精度，从根本上杜绝浮点数误差和“数字尘埃”。
    """

    # --- 1. 核心关联键 ---
    stock = models.ForeignKey(
        'StockInfo',
        on_delete=models.CASCADE,
        related_name='advanced_chip_metrics_bj',
        verbose_name='股票',
        db_index=True
    )
    trade_time = models.DateField(
        verbose_name='交易日期',
        db_index=True
    )

    # --- 2. 核心筹码峰基础指标 ---
    peak_cost = models.DecimalField(max_digits=12, decimal_places=4, verbose_name='主筹码峰成本', null=True, blank=True, help_text="当天筹码分布最密集的价格，是市场持仓的核心成本区。")
    peak_percent = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='主筹码峰占比(%)', null=True, blank=True, help_text="主筹码峰位置的筹码占总筹码的比例。")
    peak_volume = models.BigIntegerField(verbose_name='主筹码峰成交量(股)', null=True, blank=True, help_text="主筹码峰位置的绝对成交股数。")

    # --- 3. 筹码峰动态指标 ---
    peak_cost_slope_5d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本5日斜率')
    peak_cost_slope_8d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本8日斜率')
    peak_cost_slope_13d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本13日斜率')
    peak_cost_slope_21d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本21日斜率')
    peak_cost_slope_34d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本34日斜率')
    peak_cost_slope_55d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本55日斜率')
    peak_cost_slope_89d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本89日斜率')
    peak_cost_slope_144d = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True, verbose_name='筹码峰成本144日斜率')
    peak_cost_accel_5d = models.DecimalField(max_digits=18, decimal_places=8, verbose_name='筹码峰成本5日加速度', null=True, blank=True)
    peak_cost_accel_21d = models.DecimalField(max_digits=18, decimal_places=8, verbose_name='筹码峰成本21日加速度', null=True, blank=True)

    # --- 4. 筹码结构与分布指标 ---
    concentration_70pct = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='70%筹码集中度', null=True, blank=True, help_text="值越小越集中。")
    concentration_70pct_slope_5d = models.DecimalField(max_digits=18, decimal_places=8, verbose_name='70%集中度5日斜率', null=True, blank=True, help_text="负值表示筹码趋于集中。")
    concentration_90pct = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='90%筹码集中度', null=True, blank=True, help_text="值越小越集中。")
    concentration_90pct_slope_5d = models.DecimalField(max_digits=18, decimal_places=8, verbose_name='90%集中度5日斜率', null=True, blank=True, help_text="负值表示筹码趋于集中。")
    concentration_90pct_slope_21d = models.DecimalField(max_digits=18, decimal_places=8, verbose_name='90%集中度21日斜率', null=True, blank=True, help_text="中期趋势。")
    concentration_90pct_slope_55d = models.DecimalField(max_digits=18, decimal_places=8, verbose_name='90%集中度55日斜率', null=True, blank=True, help_text="长期趋势。")
    peak_stability = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='筹码峰稳定性', null=True, blank=True, help_text="值越大越稳定，代表主力控盘能力强。")
    is_multi_peak = models.BooleanField(verbose_name='是否多峰形态', default=False, help_text="持仓成本是否分散。")
    secondary_peak_cost = models.DecimalField(max_digits=12, decimal_places=4, verbose_name='次筹码峰成本', null=True, blank=True, help_text="潜在的压力或支撑位。")
    peak_distance_ratio = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='主次峰距离比', null=True, blank=True, help_text="距离越远，结构越不稳定。")
    peak_strength_ratio = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='主次峰强度比', null=True, blank=True, help_text="比率越小，主峰的统治力越强。")
    pressure_above = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='上方2%套牢盘(%)', null=True, blank=True, help_text="代表直接的短期抛压。")
    support_below = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='下方2%支撑盘(%)', null=True, blank=True, help_text="代表直接的短期支撑。")

    # --- 5. 获利盘结构指标 ---
    total_winner_rate = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='总获利盘(%)', null=True, blank=True, help_text="反映市场整体情绪。")
    winner_rate_short_term = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='短期获利盘(%)', null=True, blank=True, help_text="代表近期追涨资金的浮盈情况。")
    winner_rate_long_term = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='长期锁定盘(%)', null=True, blank=True, help_text="代表坚定持有的资金。")

    # --- 6. 辅助与过程指标 ---
    pressure_above_volume = models.BigIntegerField(verbose_name='上方套牢盘绝对量(股)', null=True, blank=True)
    support_below_volume = models.BigIntegerField(verbose_name='下方支撑盘绝对量(股)', null=True, blank=True)
    turnover_volume_in_cost_range_70pct = models.BigIntegerField(verbose_name='70%成本区换手量(股)', null=True, blank=True)
    prev_20d_close = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, verbose_name='20日前收盘价')
    
    # --- 7. 【升维】控盘度指标 ---
    peak_control_ratio = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='筹码峰控盘比(%)', null=True, blank=True, help_text="主筹码峰股数 / 流通股本。")
    peak_absorption_intensity = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='筹码峰吸筹强度', null=True, blank=True, help_text="主峰区间换手量 / 总换手量。")

    # --- 8. 【升维】利润质量指标 ---
    winner_avg_cost = models.DecimalField(max_digits=12, decimal_places=4, verbose_name='获利盘平均成本', null=True, blank=True)
    winner_profit_margin = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='获利盘安全垫(%)', null=True, blank=True, help_text="衡量获利盘的平均利润厚度。")

    # --- 9. 【升维】价码关系指标 ---
    price_to_peak_ratio = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='股价/筹码峰成本比', null=True, blank=True)
    chip_zscore = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='筹码Z-Score', null=True, blank=True, help_text="股价在筹码分布中的标准分位置。")

    # --- 10. 【升维】筹码断层指标 ---
    chip_fault_strength = models.DecimalField(max_digits=12, decimal_places=6, verbose_name='筹码断层强度', null=True, blank=True)
    chip_fault_vacuum_percent = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='断层真空区筹码占比(%)', null=True, blank=True)
    is_chip_fault_formed = models.BooleanField(verbose_name='是否形成筹码断层', default=False, help_text="极强的看涨信号。")

    # --- 11. 【超级指标】最终裁决 ---
    chip_health_score = models.DecimalField(max_digits=8, decimal_places=2, verbose_name='筹码健康分(0-100)', null=True, blank=True)

    # --- 12. 【升维】成交量微观结构指标 ---
    turnover_at_peak_ratio = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='主峰成交占比(%)', null=True, blank=True, help_text="主峰区间的交战激烈程度。")
    turnover_from_winners_ratio = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='获利盘抛压占比(%)', null=True, blank=True, help_text="短期抛售压力大小。")
    turnover_from_losers_ratio = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='套牢盘割肉占比(%)', null=True, blank=True, help_text="恐慌/割肉盘轻重。")

    class Meta:
        verbose_name = '高级筹码指标(精度升级版)'
        verbose_name_plural = verbose_name
        db_table = 'stock_advanced_chip_metrics_bj'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']
        indexes = [
            models.Index(fields=['stock']),
            models.Index(fields=['trade_time']),
            models.Index(fields=['chip_health_score']),
            models.Index(fields=['is_chip_fault_formed']),
        ]

    def __str__(self):
        return f"{self.stock.stock_code} - {self.trade_time}"


# 指数日线行情(IndexDaily)
class IndexDaily(models.Model):
    index = models.ForeignKey('IndexInfo', to_field='index_code', db_column='index_code', related_name="index_daily", on_delete=models.CASCADE, verbose_name="指数")
    trade_time = models.DateField(verbose_name=_("交易日期"), null=True, blank=True)
    close = models.FloatField(verbose_name="收盘点位", null=True, blank=True)
    open = models.FloatField(verbose_name="开盘点位", null=True, blank=True)
    high = models.FloatField(verbose_name="最高点位", null=True, blank=True)
    low = models.FloatField(verbose_name="最低点位", null=True, blank=True)
    pre_close = models.FloatField(verbose_name="昨日收盘点", null=True, blank=True)
    change = models.FloatField(verbose_name="涨跌点", null=True, blank=True)
    pct_chg = models.FloatField(verbose_name="涨跌幅", null=True, blank=True)
    vol = models.FloatField(verbose_name="成交量(手)", null=True, blank=True)
    amount = models.FloatField(verbose_name="成交额(千元)", null=True, blank=True)

    class Meta:
        db_table = "index_time_trade_day"
        verbose_name = "指数日线行情"
        verbose_name_plural = verbose_name
        unique_together = ('index', 'trade_time')

# 指数周线行情(IndexWeekly)
class IndexWeekly(models.Model):
    index = models.ForeignKey('IndexInfo', to_field='index_code', db_column='index_code', related_name="index_weekly", on_delete=models.CASCADE, verbose_name="指数")
    trade_time = models.DateField(verbose_name=_("交易日期"), null=True, blank=True)
    close = models.FloatField(verbose_name="收盘点位")
    open = models.FloatField(verbose_name="开盘点位")
    high = models.FloatField(verbose_name="最高点位")
    low = models.FloatField(verbose_name="最低点位")
    pre_close = models.FloatField(verbose_name="昨日收盘点", null=True, blank=True)
    change = models.FloatField(verbose_name="涨跌点")
    pct_chg = models.FloatField(verbose_name="涨跌幅")
    vol = models.FloatField(verbose_name="成交量(手)")
    amount = models.FloatField(verbose_name="成交额(千元)")

    class Meta:
        db_table = "index_time_trade_week"
        verbose_name = "指数周线行情"
        verbose_name_plural = verbose_name
        unique_together = ('index', 'trade_time')

# 指数月线行情(IndexMonthly)
class IndexMonthly(models.Model):
    index = models.ForeignKey('IndexInfo', to_field='index_code', db_column='index_code', related_name="index_monthly", on_delete=models.CASCADE, verbose_name="指数")
    trade_time = models.DateField(verbose_name=_("交易日期"), null=True, blank=True)
    close = models.FloatField(verbose_name="收盘点位")
    open = models.FloatField(verbose_name="开盘点位")
    high = models.FloatField(verbose_name="最高点位")
    low = models.FloatField(verbose_name="最低点位")
    pre_close = models.FloatField(verbose_name="昨日收盘点", null=True, blank=True)
    change = models.FloatField(verbose_name="涨跌点")
    pct_chg = models.FloatField(verbose_name="涨跌幅")
    vol = models.FloatField(verbose_name="成交量(手)")
    amount = models.FloatField(verbose_name="成交额(千元)")

    class Meta:
        db_table = "index_time_trade_month"
        verbose_name = "指数月线行情"
        verbose_name_plural = verbose_name
        unique_together = ('index', 'trade_time')

# 【V2.0 深化版】日内筹码动态模型
class IntradayChipDynamics(models.Model):
    """
    【V2.0 深化版】日内筹码动态模型
    - 职责: 从“结果记录”升级为“战局复盘”，捕捉日内动态叙事。
    """
    # --- 1. 核心关联键 (不变) ---
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, related_name='intraday_chip_dynamics')
    trade_date = models.DateField(verbose_name='交易日期', db_index=True)

    # --- 2. 核心战术指标 (基础版，不变) ---
    poc_price = models.DecimalField(max_digits=10, decimal_places=3, verbose_name='当日成交密集区价格(POC)')
    value_area_high = models.DecimalField(max_digits=10, decimal_places=3, verbose_name='当日价值区间上沿(VAH)')
    value_area_low = models.DecimalField(max_digits=10, decimal_places=3, verbose_name='当日价值区间下沿(VAL)')
    vwap = models.DecimalField(max_digits=10, decimal_places=3, verbose_name='当日成交量加权平均价(VWAP)')
    
    # --- 3. 【深化】时间叙事指标 ---
    class DriveType(models.TextChoices):
        STRONG_BUY = 'STRONG_BUY', '强势买入'
        WEAK_BUY = 'WEAK_BUY', '弱势买入'
        BALANCE = 'BALANCE', '平衡'
        WEAK_SELL = 'WEAK_SELL', '弱势卖出'
        STRONG_SELL = 'STRONG_SELL', '强势卖出'
        NEUTRAL = 'NEUTRAL', '中性'

    opening_drive_type = models.CharField(max_length=20, choices=DriveType.choices, default=DriveType.NEUTRAL, verbose_name='开盘驱动类型', help_text="开盘30分钟的主力意图")
    closing_auction_type = models.CharField(max_length=20, choices=DriveType.choices, default=DriveType.NEUTRAL, verbose_name='尾盘竞价类型', help_text="收盘前15分钟的多空表态")
    poc_migration_direction = models.CharField(max_length=20, choices=DriveType.choices, default=DriveType.NEUTRAL, verbose_name='POC迁移方向', help_text="盘中主要交战区的移动趋势")

    # --- 4. 【深化】成交量属性指标 ---
    volume_delta = models.BigIntegerField(verbose_name='日内成交量Delta(股)', help_text="买入意愿成交量 - 卖出意愿成交量，正值代表买方更主动")
    cumulative_delta_divergence = models.BooleanField(default=False, verbose_name='价格与CVD背离', help_text="价格新高但CVD未新高(顶背离)，或反之(底背离)")

    # --- 5. 【深化】形态结构指标 ---
    class ProfileShape(models.TextChoices):
        D_SHAPE = 'D_SHAPE', 'D形(平衡市)'
        P_SHAPE = 'P_SHAPE', 'P形(空头反击)'
        B_SHAPE = 'B_SHAPE', 'b形(多头主导)'
        SLIM = 'SLIM', '瘦长形(趋势市)'
        MULTI = 'MULTI', '多分布(分歧市)'
        UNKNOWN = 'UNKNOWN', '未知'

    profile_shape_type = models.CharField(max_length=20, choices=ProfileShape.choices, default=ProfileShape.UNKNOWN, verbose_name='成交量分布形态')

    # --- 6. 筹码演化支持字段 ---
    daily_turnover_volume = models.BigIntegerField(
        verbose_name='当日总成交量(股)', 
        help_text="用于计算换手率的核心参数之一。"
    )
    total_float_shares_on_day = models.BigIntegerField(
        verbose_name='当日总流通股本(股)', 
        help_text="用于计算换手率的另一个核心参数，必须是当日的快照值。"
    )

    class Meta:
        db_table = 'stock_intraday_chip_dynamics'
        unique_together = ('stock', 'trade_date')
        ordering = ['-trade_date']

# 【V1.0 新增】每日成交分布明细模型
class DailyTurnoverDistribution(models.Model):
    """
    【V1.0 新增】每日成交分布明细模型
    - 职责: 存储每日分钟级交易聚合后的成交量在各个价格档位的精确分布。
    - 定位: 作为筹码演化计算引擎的“原材料库”。
    """
    # 使用 OneToOneField 与日内动态分析结果一一对应，确保数据唯一性
    intraday_dynamics = models.OneToOneField(
        'IntradayChipDynamics',
        on_delete=models.CASCADE,
        related_name='turnover_distribution', # 允许从 IntradayChipDynamics 反向查询
        primary_key=True, # 将外键设为主键，性能更优，也保证了一对一
        verbose_name='所属日内动态分析'
    )
    # 存储详细的分布数据
    distribution_data = models.JSONField(
        verbose_name='成交分布(JSON)',
        help_text="格式为 {'价格': 成交量, ...} 或 [{'price': p, 'volume': v}, ...]"
    )

    class Meta:
        db_table = 'stock_daily_turnover_distribution'
        verbose_name = '每日成交分布明细'
        verbose_name_plural = verbose_name

    def __str__(self):
        return f"Distribution for {self.intraday_dynamics.stock.stock_code} on {self.intraday_dynamics.trade_date}"




class StockTimeTrade(models.Model):
    """
    分时交易数据模型
    """
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, blank=True, null=True, related_name="time_trade", verbose_name=_("股票"))
    time_level = models.CharField(max_length=10, verbose_name='分时级别')
    trade_time = models.DateTimeField(verbose_name='交易时间')
    open_price = models.FloatField(verbose_name='开盘价', null=True)
    high_price = models.FloatField(verbose_name='最高价', null=True)
    low_price = models.FloatField(verbose_name='最低价', null=True)
    close_price = models.FloatField(verbose_name='收盘价', null=True)
    volume = models.BigIntegerField(verbose_name='成交量', null=True)
    turnover = models.DecimalField(max_digits=20, decimal_places=2, verbose_name='成交额', null=True)
    amplitude = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='振幅', null=True)
    turnover_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='换手率', null=True)
    price_change_percent = models.FloatField(verbose_name='涨跌幅', null=True)
    price_change_amount = models.FloatField(verbose_name='涨跌额', null=True)
    
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
        - stock: StockInfo 对象的实例。
        - source_level: 源数据的时间级别，默认为 '5'。
        - target_levels: 目标时间级别列表，默认为 ['15', '30', '60']。
        该方法会先检查目标级别的数据是否存在，如果不存在则生成。
        """
        
        for target_level in target_levels:
            # 检查目标级别的数据是否已存在
            existing_data = cls.objects.filter(stock=stock, time_level=target_level).exists()
            if existing_data:
                print(f"警告: 股票 {stock.stock_code} 的 {target_level} 级别数据已存在，跳过生成。")
                continue  # 跳过已存在的级别
            # 其余逻辑与之前相同
            source_data = cls.objects.filter(stock=stock, time_level=source_level).order_by('trade_time')
            if not source_data.exists():
                print(f"警告: 股票 {stock.stock_code} 的 {source_level} 级别数据不存在。")
                continue
            data_list = list(source_data.values('trade_time', 'open_price', 'high_price', 'low_price', 'close_price', 'volume', 'turnover'))
            df = pd.DataFrame(data_list)
            if df.empty:
                continue
            df['trade_time'] = pd.to_datetime(df['trade_time'], utc=True)
            df.set_index('trade_time', inplace=True)
            for col in ['open_price', 'high_price', 'low_price', 'close_price', 'turnover']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            if target_level == '15':
                resample_rule = '15T'
            elif target_level == '30':
                resample_rule = '30T'
            elif target_level == '60':
                resample_rule = '60T'
            else:
                print(f"警告: 不支持的目标级别 {target_level}，跳过。")
                continue
            aggregated_df = df.resample(resample_rule).agg({
                'open_price': 'first',
                'high_price': 'max',
                'low_price': 'min',
                'close_price': 'last',
                'volume': 'sum',
                'turnover': 'sum',
            }).dropna()
            if not aggregated_df.empty:
                if stock.circulating_shares is None:
                    aggregated_df['turnover_rate'] = None
                else:
                    aggregated_df['turnover_rate'] = (aggregated_df['volume'] / stock.circulating_shares) * 100
                aggregated_df['previous_close_price'] = aggregated_df['close_price'].shift(1)
                aggregated_df['price_change_amount'] = aggregated_df['close_price'] - aggregated_df['previous_close_price']
                aggregated_df['price_change_percent'] = (aggregated_df['price_change_amount'] / aggregated_df['previous_close_price']) * 100
                aggregated_df = aggregated_df.drop(columns=['previous_close_price'], errors='ignore')
            for idx, row in aggregated_df.iterrows():
                new_instance = cls(
                    stock=stock,
                    time_level=target_level,
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
                    amplitude=None,
                )
                new_instance.save()
            print(f"成功为股票 {stock.stock_code} 生成 {target_level} 级别数据。")
