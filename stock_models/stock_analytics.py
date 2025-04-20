# stock_models/stock_analytics.py
from django.db import models
from .stock_basic import StockInfo

class StockScoreAnalysis(models.Model):
    """
    股票策略分析结果模型 (支持多策略)
    """
    stock = models.ForeignKey(
        StockInfo,
        on_delete=models.CASCADE,
        to_field='stock_code',
        related_name='score_analyses',
        verbose_name='股票代码'
    )
    # --- 核心标识字段 ---
    strategy_name = models.CharField(max_length=100, db_index=True, default="UnknownStrategy", verbose_name="策略名称")
    timestamp = models.DateTimeField(db_index=True, default="", verbose_name='分析时间戳')
    time_level = models.CharField(max_length=10, db_index=True, verbose_name='分析时间级别')

    # --- 主要策略输出 ---
    score = models.FloatField(null=True, blank=True, verbose_name='策略主评分/信号')

    # --- 各策略的关键中间结果 ---
    # 基础评分 (通用)
    base_score_raw = models.FloatField(null=True, blank=True, verbose_name='原始基础评分')
    base_score_volume_adjusted = models.FloatField(null=True, blank=True, verbose_name='量能调整后基础评分')

    # TrendFollowingStrategy 相关
    alignment_signal = models.SmallIntegerField(null=True, blank=True, verbose_name='EMA排列信号 (-3~3)')
    long_term_context = models.SmallIntegerField(null=True, blank=True, verbose_name='长期趋势背景 (-1空, 0中, 1多)')
    ema_score_5 = models.FloatField(null=True, blank=True, verbose_name='评分EMA(5)')
    ema_score_13 = models.FloatField(null=True, blank=True, verbose_name='评分EMA(13)')
    ema_score_21 = models.FloatField(null=True, blank=True, verbose_name='评分EMA(21)')
    ema_score_55 = models.FloatField(null=True, blank=True, verbose_name='评分EMA(55)')
    ema_score_233 = models.FloatField(null=True, blank=True, verbose_name='评分EMA(233)')
    ema_strength = models.FloatField(null=True, blank=True, verbose_name='趋势强度')
    score_momentum = models.FloatField(null=True, blank=True, verbose_name='评分动能')

    # TrendReversalStrategy 相关
    reversal_confirmation_signal = models.FloatField(null=True, blank=True, verbose_name='反转确认信号强度 (-1~1)')
    strong_reversal_confirmation = models.SmallIntegerField(null=True, blank=True, verbose_name='强反转确认 (-1空, 0中, 1多)')
    macd_hist_divergence = models.SmallIntegerField(null=True, blank=True, default=0, verbose_name='MACD柱背离信号 (-2~2)')
    rsi_divergence = models.SmallIntegerField(null=True, blank=True, default=0, verbose_name='RSI背离信号 (-2~2)')
    mfi_divergence = models.SmallIntegerField(null=True, blank=True, default=0, verbose_name='MFI背离信号 (-2~2)')
    obv_divergence = models.SmallIntegerField(null=True, blank=True, default=0, verbose_name='OBV背离信号 (-2~2)')
    kline_pattern = models.SmallIntegerField(null=True, blank=True, default=0, verbose_name='K线形态信号')
    rsi_obos_reversal = models.SmallIntegerField(null=True, blank=True, default=0, verbose_name='RSI超买卖反转 (-1卖, 1买)')
    stoch_obos_reversal = models.SmallIntegerField(null=True, blank=True, default=0, verbose_name='随机指标超买卖反转 (-1卖, 1买)')
    cci_obos_reversal = models.SmallIntegerField(null=True, blank=True, default=0, verbose_name='CCI超买卖反转 (-1卖, 1买)')
    bb_reversal = models.SmallIntegerField(null=True, blank=True, default=0, verbose_name='布林带反转 (-1顶, 1底)')
    volume_spike = models.SmallIntegerField(null=True, blank=True, default=0, verbose_name='放量信号 (1放量, 0其他)')

    # TPlus0Strategy 相关
    t0_signal = models.SmallIntegerField(null=True, blank=True, verbose_name='T+0信号 (-1卖, 0中, 1买)')
    price_vwap_deviation = models.FloatField(null=True, blank=True, verbose_name='价格相对VWAP偏离度')

    # --- 输入数据快照 (可选保留) ---
    close_price = models.FloatField(null=True, blank=True, verbose_name='收盘价')
    vwap = models.FloatField(null=True, blank=True, verbose_name='VWAP')

    # --- 统计分析结果 ---
    # TrendFollowingStrategy 统计
    final_signal_mean = models.FloatField(null=True, blank=True, verbose_name='最终信号平均值')
    final_signal_bullish_ratio = models.FloatField(null=True, blank=True, verbose_name='看涨信号比例')
    final_signal_bearish_ratio = models.FloatField(null=True, blank=True, verbose_name='看跌信号比例')
    final_signal_strong_bullish_ratio = models.FloatField(null=True, blank=True, verbose_name='强看涨信号比例')
    final_signal_strong_bearish_ratio = models.FloatField(null=True, blank=True, verbose_name='强看跌信号比例')
    alignment_fully_bullish_ratio = models.FloatField(null=True, blank=True, verbose_name='完全多头排列比例')
    alignment_fully_bearish_ratio = models.FloatField(null=True, blank=True, verbose_name='完全空头排列比例')
    alignment_bullish_ratio = models.FloatField(null=True, blank=True, verbose_name='多头排列比例')
    alignment_bearish_ratio = models.FloatField(null=True, blank=True, verbose_name='空头排列比例')
    long_term_bullish_ratio = models.FloatField(null=True, blank=True, verbose_name='长期看涨背景比例')
    long_term_bearish_ratio = models.FloatField(null=True, blank=True, verbose_name='长期看跌背景比例')

    # TrendReversalStrategy 统计
    final_signal_potential_buy_ratio = models.FloatField(null=True, blank=True, verbose_name='潜在买入信号比例')
    final_signal_potential_sell_ratio = models.FloatField(null=True, blank=True, verbose_name='潜在卖出信号比例')
    final_signal_strong_buy_ratio = models.FloatField(null=True, blank=True, verbose_name='强买入信号比例')
    final_signal_strong_sell_ratio = models.FloatField(null=True, blank=True, verbose_name='强卖出信号比例')
    strong_buy_reversal_ratio = models.FloatField(null=True, blank=True, verbose_name='强买入反转比例')
    strong_sell_reversal_ratio = models.FloatField(null=True, blank=True, verbose_name='强卖出反转比例')

    # TPlus0Strategy 统计
    t0_buy_ratio = models.FloatField(null=True, blank=True, verbose_name='T+0买入信号比例')
    t0_sell_ratio = models.FloatField(null=True, blank=True, verbose_name='T+0卖出信号比例')
    t0_no_signal_ratio = models.FloatField(null=True, blank=True, verbose_name='T+0无信号比例')
    t0_buy_count = models.IntegerField(null=True, blank=True, verbose_name='T+0买入信号次数')
    t0_sell_count = models.IntegerField(null=True, blank=True, verbose_name='T+0卖出信号次数')

    # --- 元数据 ---
    params_snapshot = models.JSONField(null=True, blank=True, verbose_name='参数快照')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')

    class Meta:
        verbose_name = '股票策略分析结果'
        db_table = 'stock_analysis'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'timestamp', 'time_level', 'strategy_name')
        indexes = [
            models.Index(fields=['stock', 'strategy_name', 'time_level', '-timestamp']),
            models.Index(fields=['stock', 'strategy_name', 'timestamp']),
            models.Index(fields=['strategy_name', 'time_level', '-timestamp']),
        ]
        ordering = ['stock', 'strategy_name', '-timestamp']

    def __str__(self):
        score_display = f"{self.score:.2f}" if self.score is not None else "N/A"
        return f"{self.stock.stock_code} - {self.strategy_name} ({self.time_level}) @ {self.timestamp} - Score: {score_display}"
