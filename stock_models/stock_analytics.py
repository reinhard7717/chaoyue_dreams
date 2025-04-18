# stock_analytics/models.py
from django.db import models

from stock_models.stock_basic import StockInfo
# 假设你的 stock_basic 模型在 stock_basic_data 应用中

class StockScoreAnalysis(models.Model):
    """
    股票评分趋势分析结果模型
    """
    stock = models.ForeignKey(
        StockInfo,
        on_delete=models.CASCADE, # 或者根据需要设置为 PROTECT 等
        to_field='stock_code',    # 明确指定关联 stock_basic 表的 stock_code 字段
        related_name='score_analyses',
        verbose_name='股票代码'
    )
    trade_time = models.DateTimeField(db_index=True,verbose_name='交易时间')
    time_level = models.CharField(max_length=5,db_index=True,verbose_name='分析时间级别')

    # --- 输入数据快照 ---
    score = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True,verbose_name='策略评分')
    close_price = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True,verbose_name='收盘价')
    vwap = models.DecimalField(max_digits=14, decimal_places=4, null=True, blank=True,verbose_name='VWAP')

    # --- 计算得到的评分EMA值 ---
    ema_score_5 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name='评分EMA(5)')
    ema_score_13 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name='评分EMA(13)')
    ema_score_21 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name='评分EMA(21)')
    ema_score_55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name='评分EMA(55)')
    ema_score_233 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, verbose_name='评分EMA(233)')

    # --- 趋势与动量指标 ---
    alignment_signal = models.SmallIntegerField(null=True, blank=True,verbose_name='短期趋势排列信号')
    ema_strength_13_55 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True,verbose_name='趋势强度(EMA13-EMA55)')
    score_momentum = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True,verbose_name='评分动能')
    score_volatility = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True,verbose_name='评分波动率')

    # --- 趋势背景与反转信号 ---
    long_term_context = models.SmallIntegerField(null=True, blank=True,verbose_name='长期趋势背景')
    reversal_signal = models.SmallIntegerField(null=True, blank=True,verbose_name='趋势反转信号')

    # --- 反转信号字段 ---
    # 背离信号 (-2:隐熊, -1:常熊, 0:无, 1:常牛, 2:隐牛)
    macd_hist_divergence = models.SmallIntegerField(null=True, blank=True, default=0, verbose_name='MACD柱背离信号')
    rsi_divergence = models.SmallIntegerField(null=True, blank=True, default=0, verbose_name='RSI背离信号')
    mfi_divergence = models.SmallIntegerField(null=True, blank=True, default=0, verbose_name='MFI背离信号')
    obv_divergence = models.SmallIntegerField(null=True, blank=True, default=0, verbose_name='OBV背离信号')
    # K线形态信号 (具体值见 detect_kline_patterns 函数)
    kline_pattern = models.SmallIntegerField(null=True, blank=True, default=0, verbose_name='K线形态信号')

    rsi_ob_os_reversal = models.SmallIntegerField(null=True, blank=True, default=0, verbose_name='RSI超买卖反转 (-1卖, 1买)')
    stoch_ob_os_reversal = models.SmallIntegerField(null=True, blank=True, default=0, verbose_name='随机指标超买卖反转 (-1卖, 1买)')
    cci_ob_os_reversal = models.SmallIntegerField(null=True, blank=True, default=0, verbose_name='CCI超买卖反转 (-1卖, 1买)')
    volume_signal = models.SmallIntegerField(null=True, blank=True, default=0, verbose_name='成交量信号 (-1缩量, 1放量)')
    bb_reversal_signal = models.SmallIntegerField(null=True, blank=True, default=0, verbose_name='布林带反转信号 (-1顶, 1底)')
    confirmed_reversal_signal = models.SmallIntegerField(null=True, blank=True, default=0, verbose_name='确认反转信号 (-1顶, 1底)')
    # --- 结束反转信号字段 ---

    # --- T+0 相关指标 ---
    price_vwap_deviation = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True,verbose_name='价格相对VWAP偏离度')
    t0_signal = models.SmallIntegerField(null=True, blank=True,verbose_name='T+0信号')

    # --- 时间戳 ---
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')

    class Meta:
        verbose_name = '股票评分趋势分析'
        verbose_name_plural = verbose_name
        # 对应数据库的联合唯一索引
        unique_together = ('stock', 'trade_time', 'time_level')
        indexes = [
            models.Index(fields=['stock', 'trade_time', 'time_level']), # 联合索引，优化唯一性检查和查询
            models.Index(fields=['stock', 'trade_time', 'confirmed_reversal_signal']),
            models.Index(fields=['stock', 'trade_time', 'kline_pattern']), # 可选 K线索引
            # Django 会自动为 ForeignKey (stock) 和设置了 db_index=True 的字段 (trade_time, time_level) 创建索引
        ]
        ordering = ['stock', '-trade_time'] # 默认排序方式

    def __str__(self):
        return f"{self.stock.stock_code} - {self.trade_time} ({self.time_level}min) - Score: {self.score}"

