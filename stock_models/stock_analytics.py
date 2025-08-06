# stock_models/stock_analytics.py
import json
from decimal import Decimal
from django.db import models
from .stock_basic import StockInfo
from django.conf import settings
from django.contrib.auth import get_user_model
from django.utils.translation import gettext_lazy as _

class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Decimal):
            return float(o)
        return super().default(o)

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
    signal = models.ForeignKey(TradingSignal, on_delete=models.CASCADE, null=True, blank=True)
    playbook = models.ForeignKey(Playbook, on_delete=models.CASCADE, null=True, blank=True)
    contributed_score = models.FloatField(help_text="此战法在当天实际贡献的分数")

    class Meta:
        db_table = "strategy_signal_playbook_detail"
        unique_together = ('signal', 'playbook') # 确保一个信号和一个战法的关联是唯一的
        verbose_name = "信号构成详情"
        verbose_name_plural = verbose_name

class PositionTracker(models.Model):
    """
    【V3.2 智能观察哨版】
    - 核心定位: 作为用户自选股的“持仓/观察卡片”，记录从关注、建仓到平仓的全过程。
    - 核心修改: 增加 WATCHING 状态，并调整字段以支持对所有自选股的追踪，无论其是否有买入信号。
    """
    
    # 状态枚举，增加了“观察中”状态
    class Status(models.TextChoices):
        WATCHING = 'WATCHING', '观察中'
        HOLDING = 'HOLDING', '持仓中'
        SOLD = 'SOLD', '已平仓'

    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, 
        on_delete=models.CASCADE, 
        related_name='position_trackers',
        help_text="关联的用户"
    )
    stock = models.ForeignKey(
        StockInfo, 
        on_delete=models.CASCADE, 
        related_name='position_trackers',
        help_text="关联的股票"
    )
    status = models.CharField(
        max_length=10, 
        choices=Status.choices, 
        default=Status.WATCHING, 
        db_index=True,
        help_text="当前追踪状态"
    )
    
    quantity = models.PositiveIntegerField(
        default=100, # 默认100股（1手），可以根据需要调整
        help_text="持仓数量 (股)"
    )

    # --- 建仓/观察信息 (Entry Info) ---
    entry_signal = models.ForeignKey(
        'TradingSignal', 
        on_delete=models.SET_NULL, 
        null=True, blank=True, 
        related_name='entry_positions',
        help_text="关联的建仓交易信号 (对于'观察中'状态可为空)"
    )
    entry_price = models.DecimalField(
        max_digits=10, 
        decimal_places=3, 
        null=True, blank=True, 
        help_text="建仓或初次观察时的价格"
    )
    entry_date = models.DateTimeField(
        null=True, blank=True, 
        help_text="建仓或初次观察时的日期"
    )

    # --- 平仓信息 (Exit Info) ---
    exit_signal = models.ForeignKey(
        'TradingSignal', 
        on_delete=models.SET_NULL, 
        null=True, blank=True, 
        related_name='exit_positions',
        help_text="关联的平仓交易信号"
    )
    exit_price = models.DecimalField(
        max_digits=10, 
        decimal_places=3, 
        null=True, blank=True, 
        help_text="平仓价格"
    )
    exit_date = models.DateTimeField(
        null=True, blank=True, 
        help_text="平仓日期"
    )
    
    # --- 时间戳 ---
    created_at = models.DateTimeField(auto_now_add=True, help_text="记录创建时间")
    updated_at = models.DateTimeField(auto_now=True, help_text="记录最后更新时间")

    class Meta:
        db_table = "strategy_position_tracker"
        # unique_together 已移除，以避免 entry_signal 为 NULL 时可能引发的数据库唯一性约束问题。
        # 应用层逻辑（如迁移脚本和API）将负责确保 user 和 stock 的组合是唯一的。
        ordering = ['-updated_at']
        verbose_name = "策略持仓追踪器"
        verbose_name_plural = verbose_name

    def __str__(self):
        # 提供一个清晰的字符串表示，方便在Django Admin中查看
        return f"{self.user.username} - {self.stock.stock_code} ({self.get_status_display()})"

class DailyPositionSnapshot(models.Model):
    """
    【V4.0 最终解耦版】每日持仓快照
    - 核心重构: 不再存储任何分数信息，只通过外键关联到 StrategyDailyScore。
    """
    tracker = models.ForeignKey(
        PositionTracker,
        on_delete=models.CASCADE,
        related_name='snapshots',
        verbose_name='持仓追踪器',
        null=True
    )
    snapshot_date = models.DateField(verbose_name='快照日期', db_index=True)
    close_price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='当日收盘价')
    profit_loss = models.DecimalField(max_digits=12, decimal_places=2, verbose_name='持仓盈亏', default=0)
    profit_loss_pct = models.DecimalField(max_digits=8, decimal_places=4, verbose_name='持仓盈亏率(%)', default=0)
    
    # 这就是您提到的“关联表”的核心！一个指向公共知识库的外键。
    daily_score = models.ForeignKey(
        'StrategyDailyScore',
        on_delete=models.SET_NULL, # 如果策略分析结果被删除，快照本身不删除，只是关联丢失
        related_name='position_snapshots',
        verbose_name='关联的每日分数',
        null=True,
        blank=True
    )

    class Meta:
        db_table = 'strategy_daily_position_snapshot'
        verbose_name = '每日持仓快照'
        verbose_name_plural = verbose_name
        unique_together = ('tracker', 'snapshot_date')
        ordering = ['-snapshot_date']

    def __str__(self):
        return f"{self.tracker.stock.stock_code} @ {self.snapshot_date} - P/L: {self.profit_loss_pct}%"

class StrategyDailyScore(models.Model):
    """
    【V1.0】策略每日分数 (公共知识库)
    """
    stock = models.ForeignKey(
        'StockInfo',
        on_delete=models.CASCADE,
        related_name='strategy_daily_scores',
        verbose_name='股票'
    )
    trade_date = models.DateField(verbose_name='交易日期', db_index=True)
    strategy_name = models.CharField(max_length=100, verbose_name='策略名称', db_index=True)
    offensive_score = models.IntegerField(default=0, verbose_name='总进攻分')
    risk_score = models.IntegerField(default=0, verbose_name='总风险分')
    final_score = models.FloatField(default=0.0, verbose_name='最终得分')
    positional_score = models.IntegerField(default=0, verbose_name='阵地分')
    dynamic_score = models.IntegerField(default=0, verbose_name='动能分')
    composite_score = models.IntegerField(default=0, verbose_name='战法分')
    signal_type = models.CharField(max_length=20, verbose_name='信号类型')
    score_details_json = models.JSONField(default=dict, verbose_name='分数构成详情(JSON)')

    class Meta:
        db_table = 'strategy_daily_score'
        verbose_name = '策略每日分数'
        verbose_name_plural = verbose_name
        unique_together = ('stock', 'trade_date', 'strategy_name')
        ordering = ['-trade_date']

    def __str__(self):
        return f"{self.stock.stock_code} @ {self.trade_date} [{self.strategy_name}]"

class StrategyScoreComponent(models.Model):
    """
    【V1.0】策略分数构成 (公共知识库)
    """
    class ScoreType(models.TextChoices):
        POSITIONAL = 'positional', _('阵地分')
        DYNAMIC = 'dynamic', _('动能分')
        COMPOSITE = 'composite', _('战法分')
        TRIGGER = 'trigger', _('触发器分')
        RISK = 'risk', _('风险分')
        UNKNOWN = 'unknown', _('未知')

    daily_score = models.ForeignKey(
        StrategyDailyScore,
        on_delete=models.CASCADE,
        related_name='components',
        verbose_name='所属每日分数'
    )
    signal_name = models.CharField(max_length=255, verbose_name='信号名称(代码)', db_index=True)
    signal_cn_name = models.CharField(max_length=255, verbose_name='信号中文名')
    score_type = models.CharField(max_length=20, choices=ScoreType.choices, default=ScoreType.UNKNOWN, verbose_name='分数类型')
    score_value = models.IntegerField(verbose_name='贡献分数')

    class Meta:
        db_table = 'strategy_score_component'
        verbose_name = '策略分数构成'
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['daily_score', 'score_type']),
        ]

    def __str__(self):
        return f"{self.daily_score} - {self.signal_name}: {self.score_value}"



















