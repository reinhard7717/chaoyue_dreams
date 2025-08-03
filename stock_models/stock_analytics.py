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



