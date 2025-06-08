# stock_models/stock_analytics.py
from django.db import models
from .stock_basic import StockInfo

class StockScoreAnalysis(models.Model):
    """
    股票策略分析结果模型

    用于存储不同策略对单个股票在特定时间点、特定时间级别的分析结果。
    包含了策略最终评分、关键中间计算结果、统计信息以及分析结论。
    """
    # === 核心关联与标识 ===
    stock = models.ForeignKey(
        StockInfo,
        on_delete=models.CASCADE,      # 当关联的股票信息删除时，这条分析结果也删除
        to_field='stock_code',         # 关联到 StockInfo 的 stock_code 字段
        related_name='score_analyses', # 反向查询名称 (例如 stock_info.score_analyses.all())
        verbose_name='股票代码'         # Django Admin 中显示的字段名
    )
    strategy_name = models.CharField(
        max_length=100,
        db_index=True,                 # 为策略名称创建索引，加快查询
        default="UnknownStrategy",
        verbose_name="策略名称"         # 分析结果所属的策略名称
    )
    timestamp = models.DateTimeField(
        db_index=True,                 # 为时间戳创建索引
        verbose_name='分析时间戳'       # 分析发生的时间点 (通常是K线结束时间)
    )
    time_level = models.CharField(
        max_length=10,
        db_index=True,                 # 为时间级别创建索引
        verbose_name='分析时间级别'     # 分析所基于的K线时间级别 (例如 '5', '15', '30', '60', 'D')
    )

    # === 主要策略输出 ===
    score = models.FloatField(
        null=True, blank=True,         # 允许为空
        verbose_name='策略主评分/信号'   # 策略计算出的最终综合评分或信号值 (通常 0-100)
    )

    # === 通用中间结果 ===
    base_score_raw = models.FloatField(
        null=True, blank=True,
        verbose_name='原始基础评分'     # 未经调整的多指标、多时间框架加权基础分
    )
    base_score_volume_adjusted = models.FloatField(
        null=True, blank=True,
        verbose_name='量能调整后基础评分' # 经过量能确认和量价背离调整后的基础分
    )
    # --- 量能与背离信号 (由 adjust_score_with_volume 和 detect_divergence 计算) ---
    volume_confirmation_signal = models.SmallIntegerField(
        null=True, blank=True,
        verbose_name='量能确认信号'     # -1:量能矛盾, 0:中性, 1:量能确认
    )
    volume_spike_signal = models.SmallIntegerField(
        null=True, blank=True,
        verbose_name='成交量突增信号'   # 0:正常, 1:显著放量
    )
    div_has_bullish_divergence = models.BooleanField(
        null=True, blank=True,
        verbose_name='存在看涨背离'     # 是否检测到任何类型的看涨背离 (聚合结果)
    )
    div_has_bearish_divergence = models.BooleanField(
        null=True, blank=True,
        verbose_name='存在看跌背离'     # 是否检测到任何类型的看跌背离 (聚合结果)
    )
    # 可选：存储更详细的背离信息 (如果需要)
    div_rsi_regular_bullish = models.SmallIntegerField(null=True, blank=True, verbose_name='RSI常规看涨背离')
    div_macd_hist_hidden_bearish = models.SmallIntegerField(null=True, blank=True, verbose_name='MACD隐藏看跌背离')

    # === TrendFollowingStrategy (趋势跟踪策略) 相关中间结果 ===
    alignment_signal = models.SmallIntegerField(
        null=True, blank=True,
        verbose_name='EMA排列信号'      # 基于评分EMA计算的排列信号 (-3 到 +3)
    )
    long_term_context = models.SmallIntegerField(
        null=True, blank=True,
        verbose_name='长期趋势背景'     # 基于评分与长期EMA的关系 (-1看跌, 0中性, 1看涨)
    )
    trend_strength_score = models.FloatField(
        null=True, blank=True,
        verbose_name='综合趋势强度评分' # 结合多种趋势指标的综合强度评分 (-3 到 +3)
    )
    score_momentum = models.FloatField(
        null=True, blank=True,
        verbose_name='评分动能'         # 评分的变化率 (一阶导数)
    )
    score_volatility = models.FloatField(
        null=True, blank=True,
        verbose_name='评分波动率'       # 评分的标准差 (衡量稳定性)
    )
    ema_cross_signal = models.SmallIntegerField(
        null=True, blank=True,
        verbose_name='短期EMA交叉信号'  # 例如 EMA(5) 与 EMA(13) 的交叉 (-1死叉, 0无, 1金叉)
    )
    adx_strength_signal = models.FloatField(
        null=True, blank=True,
        verbose_name='ADX趋势强度信号'  # ADX 指标给出的趋势强度和方向信号 (-1到1)
    )
    stoch_signal = models.FloatField(
        null=True, blank=True,
        verbose_name='STOCH信号'        # 随机指标状态信号 (-1超买死叉, -0.5超买, 0中性, 0.5超卖, 1超卖金叉)
    )
    vwap_deviation_signal = models.SmallIntegerField(
        null=True, blank=True,
        verbose_name='VWAP偏离信号'     # 价格相对VWAP的偏离状态 (-1显著低于, 0接近, 1显著高于)
    )
    boll_breakout_signal = models.SmallIntegerField(
        null=True, blank=True,
        verbose_name='BOLL突破信号'     # 价格与布林带关系 (-1跌破下轨, 0轨道内, 1突破上轨)
    )
    ema_strength = models.FloatField(
        null=True, blank=True,
        verbose_name='EMA强度差'  # 短期与长期 EMA 的强度差
    )
    score_momentum_acceleration = models.FloatField(
        null=True, blank=True,
        verbose_name='评分动能加速度'  # 评分动能的变化率（二阶导数）
    )
    volatility_signal = models.SmallIntegerField(
        null=True, blank=True,
        verbose_name='波动率信号'  # 基于评分波动率的信号 (-1高波动不稳定, 0中性, 1低波动稳定)
    )
    vwap_deviation_percent = models.FloatField(
        null=True, blank=True,
        verbose_name='VWAP偏离百分比'  # 价格相对 VWAP 的偏离百分比
    )
    boll_percent_b = models.FloatField(
        null=True, blank=True,
        verbose_name='布林带相对位置百分比'  # 价格在布林带中的相对位置百分比 (0-100)
    )
    final_signal_mean = models.FloatField(
        null=True, blank=True,
        verbose_name='最终信号均值'  # 最终信号的平均值
    )
    final_signal_potential_buy_ratio = models.FloatField(
        null=True, blank=True,
        verbose_name='潜在买入信号比例'  # 最终信号中潜在买入的比例 (>=65)
    )
    final_signal_potential_sell_ratio = models.FloatField(
        null=True, blank=True,
        verbose_name='潜在卖出信号比例'  # 最终信号中潜在卖出的比例 (<=35)
    )
    final_signal_strong_buy_ratio = models.FloatField(
        null=True, blank=True,
        verbose_name='强买入信号比例'  # 最终信号中强买入的比例 (>=75)
    )
    final_signal_strong_sell_ratio = models.FloatField(
        null=True, blank=True,
        verbose_name='强卖出信号比例'  # 最终信号中强卖出的比例 (<=25)
    )
    strong_buy_reversal_ratio = models.FloatField(
        null=True, blank=True,
        verbose_name='强买入反转比例'  # 强反转确认信号中买入的比例
    )
    strong_sell_reversal_ratio = models.FloatField(
        null=True, blank=True,
        verbose_name='强卖出反转比例'  # 强反转确认信号中卖出的比例
    )

    # === TrendReversalStrategy (趋势反转策略) 相关中间结果 ===
    reversal_confirmation_signal = models.FloatField(
        null=True, blank=True,
        verbose_name='反转确认信号强度' # 综合多个反转指标的确认信号强度 (-1到1)
    )
    kline_pattern = models.SmallIntegerField(
        null=True, blank=True, default=0,
        verbose_name='K线形态信号'      # 检测到的关键K线形态信号 (需要定义编码)
    )
    rsi_obos_reversal = models.SmallIntegerField(
        null=True, blank=True, default=0,
        verbose_name='RSI超买卖反转'   # RSI 指标在超买/卖区的反转信号 (-1卖, 1买)
    )
    stoch_obos_reversal = models.SmallIntegerField(
        null=True, blank=True, default=0,
        verbose_name='随机指标超买卖反转'# STOCH 指标在超买/卖区的反转信号 (-1卖, 1买)
    )
    cci_obos_reversal = models.SmallIntegerField(
        null=True, blank=True, default=0,
        verbose_name='CCI超买卖反转'    # CCI 指标在超买/卖区的反转信号 (-1卖, 1买)
    )
    bb_reversal = models.SmallIntegerField(
        null=True, blank=True, default=0,
        verbose_name='布林带反转'       # 价格触及布林带上下轨后的反转信号 (-1顶, 1底)
    )
    strong_reversal_confirmation = models.SmallIntegerField(
        null=True, blank=True,
        verbose_name='强反转确认信号'  # 综合反转信号达到阈值后的强确认 (-1强卖, 0无, 1强买)
    )
    willr_reversal = models.SmallIntegerField(
        null=True, blank=True, default=0,
        verbose_name='Williams %R 反转信号'  # Williams %R 指标在超买/卖区的反转信号 (-1卖, 1买)
    )
    atr_volatility_signal = models.SmallIntegerField(
        null=True, blank=True, default=0,
        verbose_name='ATR波动率信号'  # 基于 ATR 的波动率信号 (-1低波动削弱, 0中性, 1高波动增强)
    )
    hv_environment_signal = models.SmallIntegerField(
        null=True, blank=True, default=0,
        verbose_name='历史波动率环境信号'  # 基于历史波动率的环境信号 (-1低波动环境削弱, 0中性, 1高波动环境增强)
    )
    macd_hist_divergence = models.SmallIntegerField(
        null=True, blank=True, default=0,
        verbose_name='MACD柱背离信号'  # MACD柱的背离信号 (-2隐藏看跌, -1常规看跌, 0无, 1常规看涨, 2隐藏看涨)
    )
    mfi_divergence = models.SmallIntegerField(
        null=True, blank=True, default=0,
        verbose_name='MFI背离信号'  # MFI的背离信号 (-2隐藏看跌, -1常规看跌, 0无, 1常规看涨, 2隐藏看涨)
    )
    obv_divergence = models.SmallIntegerField(
        null=True, blank=True, default=0,
        verbose_name='OBV背离信号'  # OBV的背离信号 (-2隐藏看跌, -1常规看跌, 0无, 1常规看涨, 2隐藏看涨)
    )
    rsi_divergence = models.SmallIntegerField(
        null=True, blank=True, default=0,
        verbose_name='RSI背离信号'  # RSI的背离信号 (-2隐藏看跌, -1常规看跌, 0无, 1常规看涨, 2隐藏看涨)
    )
    volume_spike = models.IntegerField(null=True, blank=True, verbose_name="量能放量信号")

    # === TPlus0Strategy (T+0策略) 相关中间结果 ===
    t0_signal = models.SmallIntegerField(
        null=True, blank=True,
        verbose_name='T+0信号'          # T+0 策略产生的交易信号 (-1卖, 0观望, 1买)
    )
    price_vwap_deviation = models.FloatField(
        null=True, blank=True,
        verbose_name='价格相对VWAP偏离度'# T+0 策略可能特别关注的 VWAP 偏离度
    )
    t0_buy_count = models.IntegerField(
        null=True, blank=True,
        verbose_name='T+0买入信号数量'  # T+0策略中买入信号的总数
    )
    t0_buy_ratio = models.FloatField(
        null=True, blank=True,
        verbose_name='T+0买入信号比例'  # T+0策略中买入信号的比例
    )
    t0_no_signal_ratio = models.FloatField(
        null=True, blank=True,
        verbose_name='T+0无信号比例'  # T+0策略中无信号的比例
    )
    t0_sell_count = models.IntegerField(
        null=True, blank=True,
        verbose_name='T+0卖出信号数量'  # T+0策略中卖出信号的总数
    )
    t0_sell_ratio = models.FloatField(
        null=True, blank=True,
        verbose_name='T+0卖出信号比例'  # T+0策略中卖出信号的比例
    )

    # === 输入数据快照 (可选，用于调试或复现) ===
    close_price = models.FloatField(
        null=True, blank=True,
        verbose_name='收盘价(分析时)'   # 执行分析时对应的收盘价
    )
    vwap = models.FloatField(
        null=True, blank=True,
        verbose_name='VWAP(分析时)'      # 执行分析时对应的 VWAP 值
    )

    # === 最终分析结论 (由策略的 analyze_signals 方法生成) ===
    current_trend = models.CharField(
        max_length=20, null=True, blank=True,
        verbose_name='当前趋势判断'     # 文本描述，例如 'bullish', 'bearish', 'neutral'
    )
    trend_strength = models.CharField(
        max_length=20, null=True, blank=True,
        verbose_name='当前趋势强度描述' # 文本描述，例如 'strong', 'moderate', 'weak', 'very strong'
    )
    bullish_duration = models.IntegerField(
        null=True, blank=True,
        verbose_name='看涨趋势持续周期数' # 当前看涨趋势已持续的K线数量
    )
    bearish_duration = models.IntegerField(
        null=True, blank=True,
        verbose_name='看跌趋势持续周期数' # 当前看跌趋势已持续的K线数量
    )
    operation_advice = models.TextField(
        null=True, blank=True,
        verbose_name='操作建议'         # 策略基于分析给出的具体操作建议文本
    )
    risk_warning = models.TextField(
        null=True, blank=True,
        verbose_name='风险提示'         # 策略基于分析给出的潜在风险提示文本
    )

    # === 元数据 ===
    params_snapshot = models.JSONField(
        null=True, blank=True,
        verbose_name='参数快照'         # 执行此次分析时使用的策略参数 (JSON格式)
    )
    created_at = models.DateTimeField(
        auto_now_add=True,             # 记录创建时间
        verbose_name='创建时间'
    )
    updated_at = models.DateTimeField(
        auto_now=True,                 # 记录最后更新时间
        verbose_name='更新时间'
    )

    class Meta:
        verbose_name = '股票策略分析结果' # Admin后台显示的名称 (单数)
        db_table = 'stock_analysis'      # 指定数据库表名
        verbose_name_plural = verbose_name # Admin后台显示的名称 (复数)
        # 联合唯一约束：同一股票、同一策略、同一时间点、同一时间级别只能有一条记录
        unique_together = ('stock', 'strategy_name', 'timestamp', 'time_level')
        # 索引：优化常用查询
        indexes = [
            # 按 股票+策略+级别+时间 降序 查询
            models.Index(fields=['stock', 'strategy_name', 'time_level', '-timestamp'], name='idx_stock_strat_lvl_time'),
            # 按 策略+级别+时间 降序 查询 (跨股票)
            models.Index(fields=['strategy_name', 'time_level', '-timestamp'], name='idx_strat_lvl_time'),
            # 按 股票+时间 查询
            models.Index(fields=['stock', 'timestamp'], name='idx_stock_time'),
            # 按 时间 查询 (跨股票、策略)
            models.Index(fields=['timestamp'], name='idx_time'),
        ]
        # 默认排序：按股票代码、策略名称升序，时间戳降序
        ordering = ['stock', 'strategy_name', '-timestamp']

    def __str__(self):
        # 定义对象在 Admin 或命令行中的字符串表示
        score_display = f"{self.score:.2f}" if self.score is not None else "N/A"
        time_display = self.timestamp.strftime('%Y-%m-%d %H:%M') if self.timestamp else "No Timestamp"
        return f"{self.stock.stock_code} - {self.strategy_name} ({self.time_level}) @ {time_display} - Score: {score_display}"

class StockAnalysisResultTrendFollowing(models.Model):
    """
    存储股票趋势跟踪策略的分析结果。
    """
    # 外键关联到 StockInfo 模型，表示分析结果属于哪只股票
    # on_delete=models.CASCADE 表示当关联的 StockInfo 被删除时，此分析结果也一并删除
    # related_name 允许通过 stock_info_instance.trend_analysis_results.all() 反向查询
    stock = models.ForeignKey(
        StockInfo,
        on_delete=models.CASCADE,      # 当关联的股票信息删除时，这条分析结果也删除
        to_field='stock_code',         # 关联到 StockInfo 的 stock_code 字段
        related_name='stock_analysis_result_trend_following', # 反向查询名称 (例如 stock_info.score_analyses.all())
        verbose_name='股票代码'         # Django Admin 中显示的字段名
    )
    # 分析发生的时间点，通常是数据的时间戳
    timestamp = models.DateTimeField(
        db_index=True, # 添加索引以提高查询效率
        verbose_name="分析时间戳",
        help_text="该分析结果对应的数据时间戳"
    )

    # --- 核心信号分数 ---
    score = models.FloatField(
        null=True, blank=True,
        verbose_name="组合信号分",
        help_text="最终综合信号分数 (combined_signal)"
    )
    rule_signal = models.FloatField(
        null=True, blank=True,
        verbose_name="规则信号分",
        help_text="基于规则的信号分数 (final_rule_signal)"
    )
    lstm_signal = models.FloatField(
        null=True, blank=True,
        verbose_name="Transformer信号分",
        help_text="Transformer模型生成的信号分数 (transformer_signal)"
    )
    base_score_raw = models.FloatField(
        null=True, blank=True,
        verbose_name="原始基础分",
        help_text="未调整前的基础趋势分数 (base_score_raw)"
    )
    base_score_volume_adjusted = models.FloatField(
        null=True, blank=True,
        verbose_name="量能调整基础分",
        help_text="经过量能调整后的基础趋势分数 (ADJUSTED_SCORE)"
    )

    # --- 趋势与指标信号 ---
    alignment_signal = models.FloatField(
        null=True, blank=True,
        verbose_name="EMA排列信号",
        help_text="EMA排列状态信号 (alignment_signal)"
    )
    long_term_context = models.CharField(
        max_length=50, null=True, blank=True,
        verbose_name="长期趋势背景",
        help_text="长期趋势的文字描述 (long_term_context)"
    )
    adx_strength_signal = models.FloatField(
        null=True, blank=True,
        verbose_name="ADX强度信号",
        help_text="结合方向的ADX强度信号 (adx_strength_signal)"
    )
    stoch_signal = models.FloatField(
        null=True, blank=True,
        verbose_name="STOCH信号",
        help_text="随机指标信号 (stoch_signal)"
    )
    div_has_bearish_divergence = models.BooleanField(
        default=False,
        verbose_name="存在顶背离",
        help_text="是否检测到顶背离 (HAS_BEARISH_DIVERGENCE)"
    )
    div_has_bullish_divergence = models.BooleanField(
        default=False,
        verbose_name="存在底背离",
        help_text="是否检测到底背离 (HAS_BULLISH_DIVERGENCE)"
    )
    volume_spike_signal = models.FloatField(
        null=True, blank=True,
        verbose_name="量能异动信号",
        help_text="量能异动信号 (VOL_SPIKE_SIGNAL)"
    )
    close_price = models.FloatField(
        null=True, blank=True,
        verbose_name="收盘价",
        help_text="分析时的收盘价格"
    )

    # --- 趋势持续与状态 ---
    current_trend = models.CharField(
        max_length=20, null=True, blank=True,
        verbose_name="当前趋势方向",
        help_text="当前趋势的文字方向 (如: 看涨, 看跌, 中性)"
    )
    trend_strength = models.CharField(
        max_length=20, null=True, blank=True,
        verbose_name="趋势强度",
        help_text="当前趋势的强度 (如: 强劲, 温和, 不明)"
    )
    trend_duration_bullish = models.IntegerField(
        null=True, blank=True,
        verbose_name="看涨持续周期",
        help_text="看涨趋势持续的周期数"
    )
    trend_duration_bearish = models.IntegerField(
        null=True, blank=True,
        verbose_name="看跌持续周期",
        help_text="看跌趋势持续的周期数"
    )
    trend_duration_text_bullish = models.CharField(
        max_length=100, null=True, blank=True,
        verbose_name="看涨持续文本",
        help_text="看涨趋势持续时间的文字描述"
    )
    trend_duration_text_bearish = models.CharField(
        max_length=100, null=True, blank=True,
        verbose_name="看跌持续文本",
        help_text="看跌趋势持续时间的文字描述"
    )
    trend_duration_status = models.CharField(
        max_length=20, null=True, blank=True,
        verbose_name="趋势持续状态",
        help_text="趋势持续的总体状态 (如: 短, 中, 长)"
    )

    # --- 综合评估与建议 ---
    operation_advice = models.TextField(
        null=True, blank=True,
        verbose_name="操作建议",
        help_text="基于分析结果的操作建议"
    )
    risk_warning = models.TextField(
        null=True, blank=True,
        verbose_name="风险提示",
        help_text="基于分析结果的风险提示"
    )
    chinese_interpretation = models.TextField(
        null=True, blank=True,
        verbose_name="中文解读",
        help_text="详细的中文分析解读"
    )

    # --- 详细信号影响与贡献 (JSON 字段) ---
    # JSONField 适用于存储非结构化或半结构化数据，如字典和列表
    signal_impact_records_json = models.JSONField(
        null=True, blank=True,
        verbose_name="信号影响记录",
        help_text="信号影响的详细记录 (JSON 格式)"
    )
    signal_contribution_summary_json = models.JSONField(
        null=True, blank=True,
        verbose_name="信号贡献汇总",
        help_text="各信号对信心分贡献的汇总 (JSON 格式)"
    )

    # --- 信心分数 (新增的顶层字段) ---
    weighted_confidence_score = models.FloatField(
        null=True, blank=True,
        verbose_name="加权信心分数",
        help_text="加权计算后的综合信心分数"
    )
    confidence_score = models.FloatField( # MODIFIED: 新增字段
        null=True, blank=True,
        verbose_name="综合信心分数",
        help_text="未加权计算的综合信心分数"
    )
    normalized_confidence = models.FloatField( # MODIFIED: 新增字段
        null=True, blank=True,
        verbose_name="归一化信心分数",
        help_text="归一化后的综合信心分数 (范围 -1.0 到 1.0)"
    )

    # --- 原始分析数据 (JSON 字段) ---
    raw_analysis_data = models.JSONField(
        null=True, blank=True,
        verbose_name="原始分析数据",
        help_text="完整的原始分析结果字典 (JSON 格式)"
    )

    class Meta:
        db_table = 'stock_analysis_result_trend_following' # 明确指定数据库表名
        verbose_name = '股票趋势分析结果'
        verbose_name_plural = '股票趋势分析结果'
        # 联合唯一索引，确保同一只股票在同一时间点只有一个分析结果
        unique_together = ('stock', 'timestamp')
        # 默认排序方式，按股票代码和时间戳降序
        ordering = ['stock', '-timestamp']

    def __str__(self):
        stock_info = self.stock
        return f"{stock_info.stock_code}-{stock_info.stock_name} - {self.timestamp.strftime('%Y-%m-%d %H:%M')} 趋势分析"














