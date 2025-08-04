# stock_models/stock_analytics.py
import json
from decimal import Decimal
from django.db import models
from .stock_basic import StockInfo
from django.conf import settings

class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Decimal):
            return float(o)
        return super().default(o)

# 已作废
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

# 已作废
class FavoriteStockTracker(models.Model):
    """
    【V3.1 迁移兼容版】
    - 核心定位: 作为用户自选股的“持仓卡片”，记录从建仓到平仓的全过程。
    - 核心修改: 所有外键已正确指向新的 TradingSignal 模型，并增加了 null=True 以兼容数据库迁移。
    """
    user = models.ForeignKey('auth.User', on_delete=models.CASCADE, related_name='favorite_trackers')
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name='favorite_trackers')
    
    STATUS_CHOICES = [
        ('HOLDING', '持仓中'),
        ('SOLD', '已平仓'),
    ]
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='HOLDING', db_index=True)

    # --- 代码修改开始 ---
    # [修改原因] 修复 makemigrations 报错。通过添加 null=True, blank=True 允许字段在迁移过程中临时为空。
    # --- 建仓信息 (Entry Info) ---
    entry_signal = models.ForeignKey(
        'TradingSignal',
        on_delete=models.PROTECT,
        related_name='entry_tracker',
        help_text="关联的建仓交易信号",
        null=True, blank=True # 允许为空以进行迁移
    )
    entry_price = models.DecimalField(max_digits=10, decimal_places=3, help_text="建仓价格", null=True, blank=True)
    entry_date = models.DateTimeField(help_text="建仓日期", null=True, blank=True)
    
    # --- 最新追踪信息 (Latest Tracking Info) ---
    latest_signal = models.ForeignKey(
        'TradingSignal',
        on_delete=models.SET_NULL, 
        null=True, blank=True, 
        related_name='latest_tracker',
        help_text="关联的最新信号日志 (每日更新)"
    )
    latest_price = models.DecimalField(max_digits=10, decimal_places=3, null=True, blank=True, help_text="最新收盘价")
    latest_date = models.DateTimeField(null=True, blank=True, help_text="最新信号日期")
    
    # --- 预计算的追踪指标 ---
    health_change_summary = models.JSONField(
        default=dict, 
        null=True, blank=True, 
        help_text="结构化的攻防健康度变化摘要"
    )
    
    # --- 平仓信息 (Exit Info) ---
    exit_signal = models.ForeignKey(
        'TradingSignal',
        on_delete=models.SET_NULL, 
        null=True, blank=True, 
        related_name='exit_tracker',
        help_text="关联的平仓交易信号"
    )
    exit_price = models.DecimalField(max_digits=10, decimal_places=3, null=True, blank=True, help_text="平仓价格")
    exit_date = models.DateTimeField(null=True, blank=True, help_text="平仓日期")
    # --- 代码修改结束 ---
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "favorite_stock_tracker"
        unique_together = ('user', 'stock', 'entry_signal')
        ordering = ['-entry_date']
        verbose_name = "自选股持仓追踪器"
        verbose_name_plural = verbose_name

    def __str__(self):
        if self.entry_date:
            return f"{self.user.username} - {self.stock} (建仓于: {self.entry_date.strftime('%Y-%m-%d')})"
        return f"{self.user.username} - {self.stock} (无建仓日期)"

    def update_latest_status(self, latest_signal_instance: 'TradingSignal'):
        """
        一个实例方法，用于根据最新的信号日志更新自身状态。
        """
        self.latest_signal = latest_signal_instance
        self.latest_price = latest_signal_instance.close_price
        self.latest_date = latest_signal_instance.trade_time
        
        self.health_change_summary = latest_signal_instance.health_change_summary or {}

        if latest_signal_instance.signal_type == TradingSignal.SignalType.SELL:
            self.status = 'SOLD'
            self.exit_signal = latest_signal_instance
            self.exit_price = latest_signal_instance.close_price
            self.exit_date = latest_signal_instance.trade_time
        
        self.save()

class Playbook(models.Model):
    """
    【战法库】模型 (静态字典表)
    - 职责: 存储策略中所有原子信号/规则的定义。
    - 数据来源: 从策略的JSON配置文件中一次性解析并填充。
    """
    class PlaybookType(models.TextChoices):
        OFFENSIVE = 'OFFENSIVE', '进攻战法'
        RISK = 'RISK', '风险剧本'
        EXIT = 'EXIT', '离场策略'
        TRIGGER = 'TRIGGER', '触发事件'

    name = models.CharField(max_length=255, primary_key=True, help_text="战法/规则的唯一英文代码 (例如 TACTIC_LOCK_CHIP_RALLY_S)")
    cn_name = models.CharField(max_length=255, help_text="战法/规则的中文名称")
    playbook_type = models.CharField(max_length=20, choices=PlaybookType.choices, db_index=True, help_text="战法类型 (进攻/风险/离场/触发)")
    description = models.TextField(blank=True, null=True, help_text="战法的详细描述")
    default_score = models.FloatField(default=0.0, help_text="配置文件中定义的默认分数")

    class Meta:
        db_table = "strategy_playbook"
        verbose_name = "策略战法库"
        verbose_name_plural = verbose_name

    def __str__(self):
        return f"[{self.get_playbook_type_display()}] {self.cn_name} ({self.name})"

class TradingSignal(models.Model):
    """
    【交易信号】模型 (核心动态事件表)
    - 职责: 记录策略每日产生的最终决策事件。
    """
    class SignalType(models.TextChoices):
        BUY = 'BUY', '买入信号'
        SELL = 'SELL', '卖出信号'
        WARN = 'WARN', '风险预警'
        HOLD = 'HOLD', '无信号' # 或 '中性'

    id = models.BigAutoField(primary_key=True)
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name='trading_signals')
    trade_time = models.DateTimeField(db_index=True, help_text="信号生成时的K线收盘时间 (UTC)")
    timeframe = models.CharField(max_length=10, db_index=True, help_text="信号所在的时间周期")
    strategy_name = models.CharField(max_length=100, db_index=True, help_text="产生信号的策略名称")
    
    # --- 核心决策结果 ---
    signal_type = models.CharField(max_length=10, choices=SignalType.choices, default=SignalType.HOLD, db_index=True, help_text="最终信号类型")
    entry_score = models.FloatField(default=0.0, help_text="当日计算出的总进攻分")
    risk_score = models.FloatField(default=0.0, help_text="当日计算出的总风险分")
    veto_votes = models.IntegerField(default=0, help_text="当日收到的总否决票数")
    
    # --- 关联的战法详情 ---
    playbooks = models.ManyToManyField(
        Playbook,
        through='SignalPlaybookDetail', # 通过中间表关联
        related_name='trading_signals',
        help_text="构成此信号的所有战法/规则"
    )

    # --- 其他上下文信息 ---
    close_price = models.DecimalField(max_digits=10, decimal_places=3, help_text="信号日收盘价")
    health_change_summary = models.JSONField(default=dict, null=True, blank=True, help_text="当日的攻防健康度变化摘要")
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "strategy_trading_signal"
        ordering = ['-trade_time', 'stock']
        unique_together = ('stock', 'trade_time', 'timeframe', 'strategy_name') # 确保每个股票每天每个策略只有一个信号记录
        verbose_name = "策略交易信号"
        verbose_name_plural = verbose_name

    def __str__(self):
        return f"[{self.strategy_name}/{self.timeframe}] {self.stock.stock_code} @ {self.trade_time.strftime('%Y-%m-%d')} -> {self.get_signal_type_display()}"

class SignalPlaybookDetail(models.Model):
    """
    【信号构成详情】模型 (M2M Through Table)
    - 职责: 精确记录每个信号由哪些战法构成，以及每个战法当时的贡献分数。
    """
    id = models.BigAutoField(primary_key=True)
    signal = models.ForeignKey(TradingSignal, on_delete=models.CASCADE)
    playbook = models.ForeignKey(Playbook, on_delete=models.CASCADE)
    contributed_score = models.FloatField(help_text="此战法在当天实际贡献的分数")

    class Meta:
        db_table = "strategy_signal_playbook_detail"
        unique_together = ('signal', 'playbook') # 确保一个信号和一个战法的关联是唯一的
        verbose_name = "信号构成详情"
        verbose_name_plural = verbose_name

class PositionTracker(models.Model):
    pass

class DailyPositionSnapshot(models.Model):
    """
    【V1.0 每日快照版】
    - 核心定位: 作为 PositionTracker 的“每日追踪日志”，记录持仓每日的关键变化。
    - 职责: 为风险监控、卖出提醒和策略回溯提供完整的数据轨迹。
    """
    id = models.BigAutoField(primary_key=True)
    position = models.ForeignKey(
        PositionTracker, 
        on_delete=models.CASCADE, # 持仓卡片删除后，其历史快照也应删除
        related_name='snapshots',
        help_text="关联的持仓卡片"
    )
    signal = models.OneToOneField(
        'TradingSignal',
        on_delete=models.CASCADE, # 当日的信号是快照的核心，同生共死
        related_name='snapshot',
        help_text="关联的当日交易信号"
    )
    snapshot_date = models.DateField(db_index=True, help_text="快照日期")

    # --- 当日关键指标 ---
    close_price = models.DecimalField(max_digits=10, decimal_places=3, help_text="当日收盘价")
    profit_loss_pct = models.FloatField(default=0.0, help_text="截至当日的浮动盈亏百分比")
    days_in_trade = models.IntegerField(help_text="截至当日的持仓天数")

    # --- 当日策略分数 (冗余存储，用于快速查询和监控) ---
    entry_score = models.FloatField(default=0.0, help_text="当日的进攻分数")
    risk_score = models.FloatField(default=0.0, help_text="当日的风险分数")

    class Meta:
        db_table = "strategy_daily_position_snapshot"
        unique_together = ('position', 'snapshot_date') # 每个持仓每天只能有一个快照
        ordering = ['-snapshot_date']
        verbose_name = "每日持仓快照"
        verbose_name_plural = verbose_name

    def __str__(self):
        return f"快照: {self.position.stock.stock_code} @ {self.snapshot_date}"