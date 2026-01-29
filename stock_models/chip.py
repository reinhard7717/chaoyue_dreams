# stock_models\chip.py
from django.db import models
from django.utils.translation import gettext_lazy as _
import pandas as pd

from django.db import models

class BaseStockCyqChips(models.Model):
    """
    A股每日筹码分布模型 - 抽象基类
    包含所有板块通用的字段和逻辑
    """
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        on_delete=models.CASCADE,
        verbose_name='股票',  # 统一使用通用的中文名称
        db_index=True
    )
    trade_time = models.DateField(verbose_name='交易日期', db_index=True)
    price = models.FloatField(verbose_name='成本价格')
    percent = models.FloatField(verbose_name='价格占比(%)')

    class Meta:
        abstract = True  # 关键：标记为抽象基类，不会生成数据库表
        unique_together = ('stock', 'trade_time', 'price')  # 联合唯一索引
        indexes = [
            models.Index(fields=['stock', 'trade_time']),   # 通用查询索引
        ]

    def __str__(self):
        return f"{self.stock.stock_code} {self.trade_time} {self.price}"

# ==========================================
# 具体板块模型实现
# ==========================================
class StockCyqChipsSZ(BaseStockCyqChips):
    """
    深市(SZ) 每日筹码分布
    """
    class Meta(BaseStockCyqChips.Meta):
        verbose_name = '每日筹码分布SZ'
        verbose_name_plural = verbose_name
        db_table = 'stock_cyq_chips_sz'

class StockCyqChipsSH(BaseStockCyqChips):
    """
    沪市(SH) 每日筹码分布
    """
    class Meta(BaseStockCyqChips.Meta):
        verbose_name = '每日筹码分布SH'
        verbose_name_plural = verbose_name
        db_table = 'stock_cyq_chips_sh'

class StockCyqChipsCY(BaseStockCyqChips):
    """
    创业板(CY) 每日筹码分布
    """
    class Meta(BaseStockCyqChips.Meta):
        verbose_name = '每日筹码分布CY'
        verbose_name_plural = verbose_name
        db_table = 'stock_cyq_chips_cy'

class StockCyqChipsKC(BaseStockCyqChips):
    """
    科创板(KC) 每日筹码分布
    """
    class Meta(BaseStockCyqChips.Meta):
        verbose_name = '每日筹码分布KC'
        verbose_name_plural = verbose_name
        db_table = 'stock_cyq_chips_kc'

class StockCyqChipsBJ(BaseStockCyqChips):
    """
    北交所(BJ) 每日筹码分布
    """
    class Meta(BaseStockCyqChips.Meta):
        verbose_name = '每日筹码分布BJ'
        verbose_name_plural = verbose_name
        db_table = 'stock_cyq_chips_bj'

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
