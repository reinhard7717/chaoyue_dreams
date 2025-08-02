# stock_models/stock_analytics.py
import json
from decimal import Decimal
from django.db import models
from .stock_basic import StockInfo

class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Decimal):
            return float(o)
        return super().default(o)

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

    # === TrendFollowStrategy (趋势跟踪策略) 相关中间结果 ===
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
    # --- 放量起涨 识别信号 ---
    volume_breakout_signal = models.FloatField(
        null=True, blank=True,
        verbose_name="处于底部区间",
        help_text="当前是否处于底部区间"
    )
    # --- 底部放量起涨 识别信号 ---
    bottom_volume_breakout_signal = models.FloatField(
        null=True, blank=True,
        verbose_name="底部起涨信号",
        help_text="是否出现底部起涨信号"
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

class MonthlyTrendStrategyReport(models.Model):
    """
    【最终版】月线趋势跟踪策略信号与分析报告合并模型
    """
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name='monthly_trend_strategy_report')
    trade_time = models.DateField(db_index=True, verbose_name="信号日期")

    # --- 信号相关字段 (全面升级) ---
    # 观察信号
    signal_breakout_trigger = models.BooleanField(default=False, verbose_name="突破触发信号")
    
    # 【修改】买入信号细分
    signal_pullback_entry = models.BooleanField(default=False, verbose_name="回踩买入信号") # 原 signal_buy_entry，现明确为回踩
    signal_continuation_entry = models.BooleanField(default=False, verbose_name="强势追击信号") # params记录不回踩的追击买入

    # 风险/过滤信号
    signal_ma_rejection = models.IntegerField(default=0, verbose_name="均线拒绝信号")
    signal_box_rejection = models.IntegerField(default=0, verbose_name="箱体拒绝信号")
    
    # 止盈信号
    signal_take_profit = models.IntegerField(default=0, verbose_name="止盈信号")

    # --- 行情与分析快照 ---
    open_D = models.FloatField(verbose_name="开盘价", null=True, blank=True)
    high_D = models.FloatField(verbose_name="最高价", null=True, blank=True)
    low_D = models.FloatField(verbose_name="最低价", null=True, blank=True)
    close_D = models.FloatField(verbose_name="收盘价", null=True, blank=True)
    EMA_5_D = models.FloatField(verbose_name="5日EMA(追击支撑)", null=True, blank=True) # params强势追击的生命线
    EMA_10_D = models.FloatField(verbose_name="10日EMA(回踩支撑)", null=True, blank=True)
    EMA_20_D = models.FloatField(verbose_name="20日EMA", null=True, blank=True)
    volume_D = models.FloatField(verbose_name="成交量", null=True, blank=True)
    VOL_MA_20_D = models.FloatField(verbose_name="20日成交量均线", null=True, blank=True)
    
    # --- 分析与评分 ---
    washout_score = models.IntegerField(default=0, verbose_name="洗盘评分")
    buy_score = models.IntegerField(verbose_name="买入评分")
    analysis_text = models.TextField(verbose_name="分析报告")
    signal_type = models.CharField(max_length=32, verbose_name="信号类型")

    class Meta:
        db_table = 'monthly_trend_strategy_report'
        unique_together = ('stock', 'trade_time')
        ordering = ['-trade_time']
        verbose_name = "月线趋势信号与报告"
        verbose_name_plural = "月线趋势信号与报告"
        indexes = [
            # 核心索引，用于高效地按股票查找最新日期
            models.Index(fields=['stock', '-trade_time'], name='stock_trade_time_idx'),
            # 排序优化索引
            models.Index(fields=['-buy_score', '-trade_time'], name='buy_score_trade_time_idx'),
        ]

    def __str__(self):
        return f"{self.stock.code} {self.trade_time} {self.signal_type} {self.buy_score}"

class TrendFollowStrategyReport(models.Model):
    """
    趋势跟踪策略每日分析报告模型。
    
    该模型用于存储 TrendFollowStrategy 对每只股票在每个交易日的详细分析结果。
    为了优化存储，通常只保存那些产生了明确买入或卖出信号的记录。
    """
    id = models.BigAutoField(primary_key=True, help_text="报告ID")
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name='trend_follow_reports', help_text="关联的股票")
    trade_time = models.DateTimeField(help_text="交易时间（通常为日线收盘时间）")
    close_price = models.FloatField(null=True, blank=True, help_text="报告生成当日的收盘价")
    
    # --- 核心信号字段 ---
    entry_signal = models.BooleanField(default=False, help_text="最终是否产生买入信号")
    exit_signal_code = models.IntegerField(default=0, help_text="最终的卖出信号代码 (0:无, >0:对应不同卖出规则)")
    entry_score = models.FloatField(default=0.0, help_text="当日的综合买入得分")
    
    # --- 趋势背景字段 ---
    is_long_term_bullish = models.BooleanField(default=False, help_text="是否处于长期牛市背景")
    is_mid_term_bullish = models.BooleanField(default=False, help_text="是否处于中期牛市背景（均线+动态箱体）")
    
    # --- 细节追溯字段 ---
    triggered_playbooks = models.JSONField(default=dict, help_text="触发的具体买入剧本列表 (JSON格式)")

    # 为盘中监控添加新字段
    is_pullback_setup = models.BooleanField(default=False, db_index=True, verbose_name="是否回撤预备")
    pullback_target_price = models.DecimalField(
        max_digits=10, decimal_places=2, null=True, blank=True, verbose_name="回撤目标价"
    )
    
    # --- 元数据 ---
    created_at = models.DateTimeField(auto_now_add=True, help_text="记录创建时间")
    updated_at = models.DateTimeField(auto_now=True, help_text="记录更新时间")

    class Meta:
        # 确保每只股票在同一时间只有一条记录，这是批量更新/插入的关键
        db_table = "trend_follow_strategy_report"
        unique_together = [['stock', 'trade_time']]
        verbose_name = "趋势跟踪策略报告"
        verbose_name_plural = verbose_name
        ordering = ['-trade_time', 'stock']

    def __str__(self):
        signal_str = "买入" if self.entry_signal else f"卖出({self.exit_signal_code})" if self.exit_signal_code > 0 else "无信号"
        return f"{self.stock.stock_code} 在 {self.trade_time.strftime('%Y-%m-%d')} - {signal_str} (得分: {self.entry_score})"

class TrendFollowStrategySignalLog(models.Model):
    """
    【V3.2 优化版】策略信号日志模型。
    
    - (优化) 使用 DecimalField 替代 FloatField 存储价格，保证金融数据精度。
    - (优化) 使用 UniqueConstraint 替代旧的 unique_together，符合现代Django规范。
    - (优化) 移除了冗余的数据库索引，提升写入性能。
    - 整合了所有策略 (`monthly_trend_follow`, `trend_following`) 
      产生的所有可查询字段，形成统一的数据存储标准。
    """
    id = models.BigAutoField(primary_key=True, help_text="信号ID")
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name='trend_follow_strategy_signal_log', help_text="关联的股票")
    
    # --- 核心信号与上下文信息 ---
    trade_time = models.DateTimeField(db_index=True, help_text="信号生成时的精确时间戳 (K线收盘时间)")
    timeframe = models.CharField(max_length=10, db_index=True, help_text="信号所在的时间周期 (例如 'D', '60', '30')")
    strategy_name = models.CharField(max_length=100, db_index=True, help_text="产生信号的策略名称")
    
    # --- 信号详情 (通用) ---
    # 解释: 使用DecimalField替代FloatField以保证金融数据计算的精确性。
    close_price = models.DecimalField(max_digits=10, decimal_places=3, help_text="信号生成时K线的收盘价")
    entry_score = models.FloatField(default=0.0, help_text="买入信号的综合得分")
    risk_score = models.FloatField(default=0.0, help_text="风险信号的综合得分")
    risk_change_summary = models.JSONField(
        default=dict, 
        null=True, blank=True, 
        help_text="当日的风险变化摘要"
    )
    health_change_summary = models.JSONField(
        default=dict, null=True, blank=True, help_text="当日的攻防健康度变化摘要"
    )
    holding_health_score = models.FloatField(default=0.0, null=True, blank=True, help_text="【已废弃】持仓健康分 (每日计算)")
    stable_platform_price = models.DecimalField(max_digits=10, decimal_places=3, null=True, blank=True, help_text="[趋势策略]识别出的稳固筹码平台价格")
    
    # --- 信号类型 (细分) ---
    entry_signal = models.BooleanField(default=False, help_text="是否为最终的买入信号 (得分超过阈值)")
    is_risk_warning = models.BooleanField(default=False, help_text="是否为风险预警信号 (风险分>0但未触发卖出)")
    exit_signal_code = models.IntegerField(default=0, help_text="卖出信号代码 (0:无, 1:压力位, 2:移动止盈, 3:指标)")
    exit_severity_level = models.IntegerField(default=0, help_text="止盈信号的严重性等级 (0:无, 1:预警, 2:标准, 3:紧急)")
    exit_signal_reason = models.CharField(max_length=255, blank=True, null=True, help_text="止盈信号的具体原因描述")
    
    # --- 【月线策略】核心买入剧本 ---
    is_pullback_entry = models.BooleanField(default=False, help_text="[月线策略]是否为回踩买入信号")
    is_continuation_entry = models.BooleanField(default=False, help_text="[月线策略]是否为追击买入信号")
    is_breakout_trigger = models.BooleanField(default=False, help_text="[月线策略]是否为突破观察信号")

    # --- 【月线策略】核心风险与状态信号 ---
    rejection_code = models.IntegerField(default=0, help_text="[月线策略]压力位拒绝代码 (0:无, 1:MA, 2:箱体, 3:两者)")
    washout_score = models.IntegerField(default=0, help_text="[月线策略]洗盘强度得分")

    # --- 【趋势策略】核心状态信号 ---
    is_long_term_bullish = models.BooleanField(default=False, help_text="[趋势策略]是否处于长期牛市背景")
    is_mid_term_bullish = models.BooleanField(default=False, help_text="[趋势策略]是否处于中期上升趋势")

    # --- 【趋势策略】核心买入剧本 ---
    is_pullback_setup = models.BooleanField(default=False, help_text="[趋势策略]是否为回撤预备信号")
    # 解释: 同样使用DecimalField替代FloatField。
    pullback_target_price = models.DecimalField(max_digits=10, decimal_places=3, null=True, blank=True, default=None, help_text="[趋势/月线]回踩买入剧本的目标价格")

    # --- 信号追溯与元数据 ---
    triggered_playbooks = models.JSONField(default=list, help_text="触发信号的所有原子规则列表 (用于详细分析)", encoder=DecimalEncoder)
    context_snapshot = models.JSONField(default=dict, help_text="信号生成时的关键指标快照 (用于调试)", encoder=DecimalEncoder)
    created_at = models.DateTimeField(auto_now_add=True, help_text="记录创建时间")

    class Meta:
        db_table = "trend_follow_strategy_signal_log"
        ordering = ['-trade_time', 'stock']
        
        indexes = [
            # 索引1: 为“最新卖出时间”子查询提供超高速支持 (绝对核心，必须保留)。
            models.Index(fields=['stock', 'timeframe', 'exit_signal_code', 'trade_time'], name='idx_stock_tf_sell_time'),
            # 索引2: 优化“最新买入信号”的初始查找和分组 (绝对核心，必须保留)。
            models.Index(fields=['entry_signal', 'timeframe', 'stock'], name='idx_entry_tf_stock'),
            # 索引3: 加速最终结果的排序 (核心优化，建议保留)。
            models.Index(fields=['trade_time', 'entry_score'], name='idx_trade_time_score'),
        ]
        
        verbose_name = "策略信号日志"
        verbose_name_plural = verbose_name

    def __str__(self):
        signal_type = "买入" if self.entry_signal else f"卖出({self.exit_signal_code})" if self.exit_signal_code > 0 else "观察"
        return (f"[{self.strategy_name}/{self.timeframe}] {self.stock.stock_code} @ "
                f"{self.trade_time.strftime('%Y-%m-%d %H:%M')} - {signal_type}")

class FavoriteStockTracker(models.Model):
    """
    【V2.0 交易持仓版】
    - 核心定位: 作为用户自选股的“持仓卡片”，记录从建仓到平仓的全过程。
    - 业务逻辑: “加入自选”即“模拟建仓”。
    """
    user = models.ForeignKey('auth.User', on_delete=models.CASCADE, related_name='favorite_trackers')
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name='favorite_trackers')
    
    # --- 核心状态 ---
    STATUS_CHOICES = [
        ('HOLDING', '持仓中'),
        ('SOLD', '已平仓'),
    ]
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='HOLDING', db_index=True)

    # --- 建仓信息 (Entry Info) ---
    entry_log = models.ForeignKey(
        TrendFollowStrategySignalLog, 
        on_delete=models.CASCADE, # 建仓信号是核心，如果被删除，追踪记录也应删除
        related_name='entry_tracker',
        help_text="关联的建仓信号日志"
    )
    entry_price = models.DecimalField(max_digits=10, decimal_places=3, help_text="建仓价格 (来自建仓信号)")
    entry_date = models.DateTimeField(help_text="建仓日期 (来自建仓信号)")
    entry_score = models.FloatField(default=0.0, help_text="建仓时的进攻分数")

    # --- 最新追踪信息 (Latest Tracking Info) ---
    latest_log = models.ForeignKey(
        TrendFollowStrategySignalLog, 
        on_delete=models.SET_NULL, 
        null=True, blank=True, 
        related_name='latest_tracker',
        help_text="关联的最新信号日志 (每日更新)"
    )
    latest_price = models.DecimalField(max_digits=10, decimal_places=3, null=True, blank=True, help_text="最新收盘价")
    latest_date = models.DateTimeField(null=True, blank=True, help_text="最新信号日期")
    
    # --- 预计算的追踪指标 ---
    holding_health_score = models.FloatField(default=0.0, help_text="最新的持仓健康分")
    health_change_summary = models.JSONField(
        default=dict, 
        null=True, blank=True, 
        help_text="结构化的攻防健康度变化摘要"
    )
    score_change_vs_entry = models.FloatField(default=0.0, help_text="最新分数相比建仓时的变化量")
    profit_loss_pct = models.FloatField(default=0.0, help_text="当前持仓的浮动盈亏百分比")

    # --- 平仓信息 (Exit Info) ---
    exit_log = models.ForeignKey(
        TrendFollowStrategySignalLog, 
        on_delete=models.SET_NULL, 
        null=True, blank=True, 
        related_name='exit_tracker',
        help_text="关联的平仓信号日志"
    )
    exit_price = models.DecimalField(max_digits=10, decimal_places=3, null=True, blank=True, help_text="平仓价格")
    exit_date = models.DateTimeField(null=True, blank=True, help_text="平仓日期")
    
    # --- 时间戳 ---
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "favorite_stock_tracker"
        unique_together = ('user', 'stock', 'entry_log') # 同一个用户对同一只股票的同一次建仓，只能有一个追踪记录
        ordering = ['-entry_date']
        verbose_name = "自选股持仓追踪器"
        verbose_name_plural = verbose_name

    def __str__(self):
        return f"{self.user.username} - {self.stock} (建仓于: {self.entry_date.strftime('%Y-%m-%d')})"

    def update_latest_status(self, latest_log_instance: TrendFollowStrategySignalLog):
        """
        一个实例方法，用于根据最新的信号日志更新自身状态。
        """
        self.latest_log = latest_log_instance
        self.latest_price = latest_log_instance.close_price
        self.latest_date = latest_log_instance.trade_time
        
        # 更新预计算指标
        self.health_change_summary = latest_log_instance.health_change_summary or {}
        self.score_change_vs_entry = (latest_log_instance.entry_score or 0.0) - self.entry_score
        if self.entry_price and self.entry_price > 0:
            self.profit_loss_pct = ((self.latest_price / self.entry_price) - 1) * 100

        # 更新状态
        if latest_log_instance.exit_signal_code > 0:
            self.status = 'SOLD'
            self.exit_log = latest_log_instance
            self.exit_price = latest_log_instance.close_price
            self.exit_date = latest_log_instance.trade_time
        
        self.save()

class TrendFollowStrategyState(models.Model):
    """
    【V1.0】策略状态摘要模型
    
    设计目的:
    - 为前端仪表盘提供一个高效、快速的数据源。
    - 每只股票/每种策略在这里只保留一条记录，存储最新的状态。
    - 由Celery任务在每次信号计算后自动更新，避免了前端的复杂实时计算。
    """
    id = models.BigAutoField(primary_key=True, help_text="状态ID")
    
    # 核心关联
    stock = models.ForeignKey(StockInfo, to_field='stock_code', db_column='stock_code',on_delete=models.CASCADE, related_name='strategy_states', help_text="关联的股票")
    strategy_name = models.CharField(max_length=100, db_index=True, help_text="策略名称")
    time_level = models.CharField(max_length=10, db_index=True, help_text="策略运行的时间框架 (例如 'D', '60', '30')")

    # 最新状态摘要
    latest_score = models.FloatField(default=0.0, help_text="最新的策略综合得分")
    latest_trade_time = models.DateTimeField(null=True, blank=True, help_text="最新信号的K线时间")
    last_buy_time = models.DateTimeField(null=True, blank=True, help_text="最近一次有效买入信号的时间")
    last_sell_time = models.DateTimeField(null=True, blank=True, help_text="最近一次有效卖出信号的时间")
    
    # 激活的剧本详情
    active_playbooks = models.JSONField(default=list, help_text="最新信号触发的剧本/规则列表")

    # 元数据
    updated_at = models.DateTimeField(auto_now=True, help_text="状态更新时间")

    class Meta:
        db_table = "trend_follow_strategy_state_summary"
        ordering = ['-latest_score', 'stock']
        # 确保每只股票对于每种策略只有一条状态记录
        constraints = [
            models.UniqueConstraint(
                fields=['stock', 'strategy_name', 'time_level'], 
                name='unique_stock_strategy_state'
            )
        ]
        verbose_name = "策略状态摘要"
        verbose_name_plural = verbose_name

    def __str__(self):
        return f"[{self.strategy_name}] {self.stock.stock_code} - Score: {self.latest_score:.0f}"



