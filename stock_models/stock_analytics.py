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

















