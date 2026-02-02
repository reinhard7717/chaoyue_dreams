# stock_models\structural_factors.py
# 股票结构因子模型（StockStructuralFactors）
from django.db import models
from django.utils import timezone
from decimal import Decimal

class StockStructuralFactorsBase(models.Model):
    """
    股票结构因子基类 - 整合K线结构与斐波那契时间分析
    基于日线、分钟线计算的结构化分析指标
    """
    
    # 基础信息
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        db_column='stock_code',
        on_delete=models.CASCADE,
        verbose_name='股票'
    )
    trade_time = models.DateField(verbose_name='交易日期', db_index=True)
    
    # ====================== K线形态结构因子 ======================
    # 基本K线形态
    candle_body_ratio = models.DecimalField(
        max_digits=5, decimal_places=2,
        null=True, blank=True,
        verbose_name='实体长度比'
    )
    
    candle_upper_shadow_ratio = models.DecimalField(
        max_digits=5, decimal_places=2,
        null=True, blank=True,
        verbose_name='上影线比例'
    )
    
    candle_lower_shadow_ratio = models.DecimalField(
        max_digits=5, decimal_places=2,
        null=True, blank=True,
        verbose_name='下影线比例'
    )
    
    marubozu = models.BooleanField(
        default=False,
        verbose_name='光头光脚K线'
    )
    
    doji = models.BooleanField(
        default=False,
        verbose_name='十字星'
    )
    
    spinning_top = models.BooleanField(
        default=False,
        verbose_name='纺锤线'
    )
    
    # 反转形态
    morning_star = models.BooleanField(default=False, verbose_name='早晨之星')
    evening_star = models.BooleanField(default=False, verbose_name='黄昏之星')
    hammer = models.BooleanField(default=False, verbose_name='锤子线')
    hanging_man = models.BooleanField(default=False, verbose_name='上吊线')
    engulfing_bullish = models.BooleanField(default=False, verbose_name='看涨吞没')
    engulfing_bearish = models.BooleanField(default=False, verbose_name='看跌吞没')
    piercing_pattern = models.BooleanField(default=False, verbose_name='刺透形态')
    dark_cloud_cover = models.BooleanField(default=False, verbose_name='乌云盖顶')
    
    # 持续形态
    three_white_soldiers = models.BooleanField(default=False, verbose_name='三白兵')
    three_black_crows = models.BooleanField(default=False, verbose_name='三只乌鸦')
    rising_three_methods = models.BooleanField(default=False, verbose_name='上升三法')
    falling_three_methods = models.BooleanField(default=False, verbose_name='下降三法')
    
    # 缺口分析
    gap_type = models.CharField(
        max_length=20,
        null=True, blank=True,
        verbose_name='缺口类型'
    )
    
    gap_size = models.DecimalField(
        max_digits=10, decimal_places=4,
        null=True, blank=True,
        verbose_name='缺口幅度'
    )
    
    gap_filled = models.BooleanField(
        default=False,
        verbose_name='缺口是否回补'
    )
    
    # ====================== 价格通道结构因子 ======================
    # 布林带结构
    boll_position = models.CharField(
        max_length=20,
        null=True, blank=True,
        verbose_name='布林带位置'
    )
    
    boll_bandwidth = models.DecimalField(
        max_digits=10, decimal_places=4,
        null=True, blank=True,
        verbose_name='布林带宽度'
    )
    
    boll_squeeze = models.BooleanField(
        default=False,
        verbose_name='布林带挤压'
    )
    
    boll_breakout = models.BooleanField(
        default=False,
        verbose_name='布林带突破'
    )
    
    # 唐奇安通道
    donchian_high = models.DecimalField(
        max_digits=10, decimal_places=4,
        null=True, blank=True,
        verbose_name='唐奇安上轨'
    )
    
    donchian_low = models.DecimalField(
        max_digits=10, decimal_places=4,
        null=True, blank=True,
        verbose_name='唐奇安下轨'
    )
    
    donchian_width = models.DecimalField(
        max_digits=10, decimal_places=4,
        null=True, blank=True,
        verbose_name='唐奇安宽度'
    )
    
    # 价格通道突破
    channel_breakout = models.CharField(
        max_length=20,
        null=True, blank=True,
        verbose_name='通道突破方向'
    )
    
    # ====================== 支撑阻力结构因子 ======================
    # 价格密集区识别
    price_cluster_strength = models.DecimalField(
        max_digits=5, decimal_places=2,
        null=True, blank=True,
        verbose_name='价格密集区强度'
    )
    
    support_cluster = models.JSONField(
        null=True, blank=True,
        verbose_name='支撑密集区'
    )
    
    resistance_cluster = models.JSONField(
        null=True, blank=True,
        verbose_name='阻力密集区'
    )
    
    # 高低点结构
    swing_high = models.BooleanField(
        default=False,
        verbose_name='摆动高点'
    )
    
    swing_low = models.BooleanField(
        default=False,
        verbose_name='摆动低点'
    )
    
    higher_high = models.BooleanField(
        default=False,
        verbose_name='更高的高点'
    )
    
    higher_low = models.BooleanField(
        default=False,
        verbose_name='更高的低点'
    )
    
    lower_high = models.BooleanField(
        default=False,
        verbose_name='更低的高点'
    )
    
    lower_low = models.BooleanField(
        default=False,
        verbose_name='更低的低点'
    )
    
    # 趋势线分析
    trendline_break = models.BooleanField(
        default=False,
        verbose_name='趋势线突破'
    )
    
    trendline_type = models.CharField(
        max_length=20,
        null=True, blank=True,
        verbose_name='趋势线类型'
    )
    
    # ====================== 斐波那契价格结构因子 ======================
    fib_price_level_236 = models.BooleanField(default=False, verbose_name='23.6%价格位')
    fib_price_level_382 = models.BooleanField(default=False, verbose_name='38.2%价格位')
    fib_price_level_500 = models.BooleanField(default=False, verbose_name='50.0%价格位')
    fib_price_level_618 = models.BooleanField(default=False, verbose_name='61.8%价格位')
    fib_price_level_786 = models.BooleanField(default=False, verbose_name='78.6%价格位')
    
    fib_price_resistance = models.DecimalField(
        max_digits=10, decimal_places=4,
        null=True, blank=True,
        verbose_name='斐波那契阻力位'
    )
    
    fib_price_support = models.DecimalField(
        max_digits=10, decimal_places=4,
        null=True, blank=True,
        verbose_name='斐波那契支撑位'
    )
    
    fib_price_extension_1272 = models.BooleanField(default=False, verbose_name='127.2%扩展位')
    fib_price_extension_1618 = models.BooleanField(default=False, verbose_name='161.8%扩展位')
    fib_price_extension_2618 = models.BooleanField(default=False, verbose_name='261.8%扩展位')
    
    # ====================== 斐波那契时间结构因子 ======================
    fib_time_window_3 = models.BooleanField(default=False, verbose_name='斐波那契3日窗口')
    fib_time_window_5 = models.BooleanField(default=False, verbose_name='斐波那契5日窗口')
    fib_time_window_8 = models.BooleanField(default=False, verbose_name='斐波那契8日窗口')
    fib_time_window_13 = models.BooleanField(default=False, verbose_name='斐波那契13日窗口')
    fib_time_window_21 = models.BooleanField(default=False, verbose_name='斐波那契21日窗口')
    fib_time_window_34 = models.BooleanField(default=False, verbose_name='斐波那契34日窗口')
    fib_time_window_55 = models.BooleanField(default=False, verbose_name='斐波那契55日窗口')
    fib_time_window_89 = models.BooleanField(default=False, verbose_name='斐波那契89日窗口')
    fib_time_window_144 = models.BooleanField(default=False, verbose_name='斐波那契144日窗口')
    fib_time_window_233 = models.BooleanField(default=False, verbose_name='斐波那契233日窗口')
    
    fib_time_score = models.DecimalField(
        max_digits=5, decimal_places=2, 
        null=True, blank=True, 
        verbose_name='斐波那契时间得分'
    )
   
    # ====================== 成交量结构因子 ======================
    volume_structure_type = models.CharField(
        max_length=30,
        null=True, blank=True,
        verbose_name='成交量结构类型'
    )
    
    volume_cluster = models.BooleanField(
        default=False,
        verbose_name='成交量密集区'
    )
    
    volume_dry_up = models.BooleanField(
        default=False,
        verbose_name='成交量枯竭'
    )
    
    volume_spike = models.BooleanField(
        default=False,
        verbose_name='成交量峰值'
    )
    
    # 量价背离
    price_volume_divergence = models.CharField(
        max_length=30,
        null=True, blank=True,
        verbose_name='量价背离类型'
    )
    
    # ====================== 波动率结构因子 ======================
    volatility_regime = models.CharField(
        max_length=20,
        null=True, blank=True,
        verbose_name='波动率状态'
    )
    
    volatility_compression = models.BooleanField(
        default=False,
        verbose_name='波动率压缩'
    )
    
    volatility_expansion = models.BooleanField(
        default=False,
        verbose_name='波动率扩张'
    )
    
    atr_position = models.DecimalField(
        max_digits=5, decimal_places=2,
        null=True, blank=True,
        verbose_name='ATR相对位置'
    )

    class Meta:
        abstract = True
        ordering = ['-trade_time']
        indexes = [
            models.Index(fields=['trade_time']),
            models.Index(fields=['structure_score']),
            models.Index(fields=['signal_strength']),
            models.Index(fields=['fib_time_price_resonance']),
            models.Index(fields=['primary_signal']),
        ]
    
    def __str__(self):
        return f"{self.stock_id} - {self.trade_time} - 结构因子"

# ====================== 各市场具体实现 ======================

class StockStructuralFactors_SH(StockStructuralFactorsBase):
    """上海市场结构因子"""
    
    # 上交所特有因子
    sh_volume_profile = models.JSONField(
        null=True, blank=True,
        verbose_name='上交所成交量分布'
    )
    
    sh_market_depth = models.DecimalField(
        max_digits=10, decimal_places=4,
        null=True, blank=True,
        verbose_name='市场深度指标'
    )
    
    class Meta:
        db_table = 'stock_structural_factors_sh'
        verbose_name = '结构因子-上海'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']
        indexes = [
            models.Index(fields=['stock', 'trade_time']),
        ]
    
    def __str__(self):
        return f"SH - {self.stock_id} - {self.trade_time}"

class StockStructuralFactors_SZ(StockStructuralFactorsBase):
    """深圳市场结构因子"""
    
    # 深交所特有因子
    sz_volume_structure = models.JSONField(
        null=True, blank=True,
        verbose_name='深交所成交量结构'
    )
    
    sz_price_limit_effect = models.DecimalField(
        max_digits=5, decimal_places=2,
        null=True, blank=True,
        verbose_name='涨跌停板效应'
    )
    
    class Meta:
        db_table = 'stock_structural_factors_sz'
        verbose_name = '结构因子-深圳'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']
        indexes = [
            models.Index(fields=['stock', 'trade_time']),
        ]
    
    def __str__(self):
        return f"SZ - {self.stock_id} - {self.trade_time}"

class StockStructuralFactors_CY(StockStructuralFactorsBase):
    """创业板结构因子"""
    
    # 创业板特有因子
    cy_volatility_index = models.DecimalField(
        max_digits=8, decimal_places=4,
        null=True, blank=True,
        verbose_name='创业板波动率指数'
    )
    
    cy_innovation_score = models.DecimalField(
        max_digits=5, decimal_places=2,
        null=True, blank=True,
        verbose_name='创新性得分'
    )
    
    cy_growth_momentum = models.DecimalField(
        max_digits=8, decimal_places=4,
        null=True, blank=True,
        verbose_name='成长动量'
    )
    
    class Meta:
        db_table = 'stock_structural_factors_cy'
        verbose_name = '结构因子-创业板'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']
        indexes = [
            models.Index(fields=['stock', 'trade_time']),
        ]
    
    def __str__(self):
        return f"CY - {self.stock_id} - {self.trade_time}"

class StockStructuralFactors_KC(StockStructuralFactorsBase):
    """科创板结构因子"""
    
    # 科创板特有因子
    kc_tech_leadership = models.DecimalField(
        max_digits=5, decimal_places=2,
        null=True, blank=True,
        verbose_name='科技领导力得分'
    )
    
    kc_rnd_intensity = models.DecimalField(
        max_digits=8, decimal_places=4,
        null=True, blank=True,
        verbose_name='研发投入强度'
    )
    
    kc_institutional_holding = models.DecimalField(
        max_digits=8, decimal_places=4,
        null=True, blank=True,
        verbose_name='机构持股比例'
    )
    
    kc_premium_valuation = models.DecimalField(
        max_digits=8, decimal_places=4,
        null=True, blank=True,
        verbose_name='估值溢价'
    )
    
    class Meta:
        db_table = 'stock_structural_factors_kc'
        verbose_name = '结构因子-科创板'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']
        indexes = [
            models.Index(fields=['stock', 'trade_time']),
        ]
    
    def __str__(self):
        return f"KC - {self.stock_id} - {self.trade_time}"

class StockStructuralFactors_BJ(StockStructuralFactorsBase):
    """北京市场结构因子"""
    
    # 北交所特有因子
    bj_sme_focus = models.DecimalField(
        max_digits=5, decimal_places=2,
        null=True, blank=True,
        verbose_name='中小企业聚焦度'
    )
    
    bj_liquidity_metric = models.DecimalField(
        max_digits=8, decimal_places=4,
        null=True, blank=True,
        verbose_name='北交所流动性指标'
    )
    
    bj_market_maker_effect = models.DecimalField(
        max_digits=5, decimal_places=2,
        null=True, blank=True,
        verbose_name='做市商效应'
    )
    
    class Meta:
        db_table = 'stock_structural_factors_bj'
        verbose_name = '结构因子-北京'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']
        indexes = [
            models.Index(fields=['stock', 'trade_time']),
        ]
    
    def __str__(self):
        return f"BJ - {self.stock_id} - {self.trade_time}"
