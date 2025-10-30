# stock_models\advanced_metrics.py
from django.db import models
from django.utils.translation import gettext_lazy as _
import pandas as pd

# 筹码高级指标模型
class BaseAdvancedChipMetrics(models.Model):
    """
    【V19.0 · 四象限模型重构版】
    - 核心革命: 废弃旧的扁平化指标体系，引入“静态结构”、“内部动态”、“跨日迁徙”、“博弈意图”四象限模型，
                  使指标体系具备了从现象到本质的逻辑层次。
    - 核心新增:
      1. `main_force_cost_advantage`: 主力成本优势，直指A股博弈核心。
      2. `winner_conviction_index`: 获利盘信念指数，量化趋势的持续性。
      3. `cost_divergence_normalized`: 标准化成本发散度，提供跨股票可比的洗盘/派发信号。
    """
    trade_time = models.DateField(verbose_name='交易日期', db_index=True)
    # --- 第一象限: 静态结构 (Static Structure) ---
    STATIC_STRUCTURE_METRICS = {
        'peak_cost': '主筹码峰成本',
        'peak_percent': '主筹码峰占比(%)',
        'is_multi_peak': '是否多峰形态',
        'secondary_peak_cost': '次筹码峰成本',
        'peak_distance_ratio': '主次峰距离比',
        'peak_strength_ratio': '主次峰强度比',
        'concentration_70pct': '70%筹码集中度',
        'concentration_90pct': '90%筹码集中度',
        'pressure_above': '上方2%套牢盘(%)',
        'support_below': '下方2%支撑盘(%)',
        'chip_fault_strength': '筹码断层强度',
        'chip_fault_vacuum_percent': '断层真空区筹码占比(%)',
        'is_chip_fault_formed': '是否形成筹码断层',
        'total_winner_rate': '总获利盘(%)',
        'total_loser_rate': '总套牢盘(%)',
        'winner_rate_short_term': '短期获利盘(%)',
        'winner_rate_long_term': '长期锁定盘(%)',
        'loser_rate_short_term': '短期套牢盘(%)',
        'loser_rate_long_term': '长期套牢盘(%)',
    }
    # --- 第二象限: 内部动态 (Intraday Dynamics) ---
    INTRADAY_DYNAMICS_METRICS = {
        'realized_pressure_intensity': '真实压力强度(%)',
        'realized_support_intensity': '真实支撑强度(%)',
        'turnover_at_peak_ratio': '主峰成交占比(%)',
        'peak_defense_intensity': '主峰防守强度(%)',
        'peak_vwap_deviation': '主峰VWAP偏离度(%)',
        'intraday_volume_gini': '日内成交基尼系数',
        'volume_weighted_time_index': '成交量加权时间指数',
        'intraday_trend_efficiency': '日内趋势效率',
        'am_pm_vwap_ratio': '上下午VWAP比(%)',
        'fault_breakthrough_intensity': '断层突破强度',
    }
    # --- 第三象限: 跨日迁徙 (Cross-Day Flow) ---
    CROSS_DAY_FLOW_METRICS = {
        'concentration_increase_by_support': '承接增集度',
        'concentration_increase_by_chasing': '追涨增集度',
        'concentration_decrease_by_distribution': '派发减集度',
        'concentration_decrease_by_capitulation': '割肉减集度',
        'short_term_profit_taking_ratio': '短期获利盘兑现占比(%)',
        'long_term_chips_unlocked_ratio': '长期锁定盘解锁占比(%)',
        'short_term_capitulation_ratio': '短期套牢盘割肉占比(%)',
        'long_term_despair_selling_ratio': '长期套牢盘绝望占比(%)',
        'cost_divergence': '成本发散度',
        'cost_divergence_normalized': '标准化成本发散度', # 新增
    }
    # --- 第四象限: 博弈意图 (Game-Theoretic Intent) ---
    GAME_THEORY_METRICS = {
        'main_force_suppressive_accumulation': '主力打压吸筹占比(%)',
        'main_force_rally_distribution': '主力拉高出货占比(%)',
        'main_force_capitulation_distribution': '主力恐慌派发占比(%)',
        'main_force_chasing_accumulation': '主力追涨吸筹占比(%)',
        'retail_chasing_accumulation': '散户追涨抬轿占比(%)',
        'peak_net_volume_flow': '主峰净成交量流向',
        'peak_control_ratio': '筹码峰控盘比(%)',
        'winner_avg_cost': '获利盘平均成本',
        'winner_profit_margin': '获利盘安全垫(%)',
        'profit_taking_urgency': '获利盘兑现紧迫度(%)',
        'profit_realization_premium': '利润兑现溢价(%)',
        'avg_cost_short_term': '短期持仓者平均成本',
        'avg_cost_long_term': '长期持仓者平均成本',
        'main_force_cost_advantage': '主力成本优势(%)', # 新增
        'winner_conviction_index': '获利盘信念指数', # 新增
        'chip_health_score': '筹码健康分(0-100)',
    }
    # --- 第五象限: 生命体征 (Vital Signs) ---
    VITAL_SIGNS_METRICS = {
        'chip_turnover_velocity': '筹码换手速度(JSD)',
        'chip_entropy': '筹码熵',
        'structural_stability_index': '结构稳定性指数',
        'dominant_force_posture': '主导力量姿态',
    }
    CORE_METRICS = {
        **STATIC_STRUCTURE_METRICS,
        **INTRADAY_DYNAMICS_METRICS,
        **CROSS_DAY_FLOW_METRICS,
        **GAME_THEORY_METRICS,
    }
    UNIFIED_PERIODS = [1, 5, 13, 21, 55]
    INTEGER_FIELDS = ['peak_volume', 'pressure_above_volume', 'support_below_volume']
    BOOLEAN_FIELDS = ['is_multi_peak', 'is_chip_fault_formed']
    SLOPE_ACCEL_EXCLUSIONS = [
        # 动态归因类 (已经是变化量)
        'concentration_increase_by_support', 'concentration_increase_by_chasing',
        'concentration_decrease_by_distribution', 'concentration_decrease_by_capitulation',
        # 筹码交互类 (事件驱动，非连续)
        'main_force_suppressive_accumulation', 'retail_suppressive_accumulation',
        'main_force_rally_distribution', 'retail_rally_distribution',
        'main_force_capitulation_distribution', 'retail_capitulation_distribution',
        'main_force_chasing_accumulation', 'retail_chasing_accumulation',
        # 跨日迁徙类 (已经是变化量)
        'short_term_profit_taking_ratio', 'long_term_chips_unlocked_ratio',
        'short_term_capitulation_ratio', 'long_term_despair_selling_ratio',
        # 事件驱动及高波动类
        'fault_breakthrough_intensity', 'intraday_trend_efficiency', 'am_pm_vwap_ratio',
    ]
    for name, verbose in CORE_METRICS.items():
        if name in INTEGER_FIELDS:
            vars()[name] = models.BigIntegerField(verbose_name=verbose, null=True, blank=True)
        elif name in BOOLEAN_FIELDS:
            vars()[name] = models.BooleanField(verbose_name=verbose, default=False)
        elif 'cost' in name or 'price' in name:
            vars()[name] = models.DecimalField(max_digits=12, decimal_places=4, verbose_name=verbose, null=True, blank=True)
        else:
            vars()[name] = models.FloatField(verbose_name=verbose, null=True, blank=True)
        if name in BOOLEAN_FIELDS or name in SLOPE_ACCEL_EXCLUSIONS:
            continue
        for p in UNIFIED_PERIODS:
            vars()[f'{name}_slope_{p}d'] = models.FloatField(verbose_name=f'{verbose}{p}日斜率', null=True, blank=True)
            vars()[f'{name}_accel_{p}d'] = models.FloatField(verbose_name=f'{verbose}{p}日加速度', null=True, blank=True)
    class Meta:
        abstract = True
        ordering = ['-trade_time']
    def __str__(self):
        if hasattr(self, 'stock') and self.stock:
            return f"{self.stock.stock_code} - {self.trade_time}"
        return f"AdvancedChipMetric - {self.trade_time}"

class AdvancedChipMetrics_SZ(BaseAdvancedChipMetrics):
    # 唯一需要在此定义的字段是外键，因为它的 related_name 对每个表都必须是唯一的
    stock = models.ForeignKey(
        'StockInfo',
        on_delete=models.CASCADE,
        related_name='advanced_chip_metrics_sz', # 市场特定的 related_name
        verbose_name='股票',
        db_index=True
    )
    class Meta(BaseAdvancedChipMetrics.Meta): # 继承基类的 Meta 设置
        abstract = False # 覆盖基类的 abstract=True，使其成为一个具体的模型
        verbose_name = '高级筹码指标-深圳(V6.0-衍生固化)'
        verbose_name_plural = verbose_name
        db_table = 'stock_advanced_chip_metrics_sz'
        unique_together = ('stock', 'trade_time')
        indexes = [
            models.Index(fields=['stock', 'trade_time']), # 优化联合索引
            models.Index(fields=['chip_health_score']),
            models.Index(fields=['is_chip_fault_formed']),
        ]

class AdvancedChipMetrics_SH(BaseAdvancedChipMetrics):
    stock = models.ForeignKey(
        'StockInfo',
        on_delete=models.CASCADE,
        related_name='advanced_chip_metrics_sh',
        verbose_name='股票',
        db_index=True
    )
    class Meta(BaseAdvancedChipMetrics.Meta):
        abstract = False
        verbose_name = '高级筹码指标-上海(V6.0-衍生固化)'
        verbose_name_plural = verbose_name
        db_table = 'stock_advanced_chip_metrics_sh'
        unique_together = ('stock', 'trade_time')
        indexes = [
            models.Index(fields=['stock', 'trade_time']),
            models.Index(fields=['chip_health_score']),
            models.Index(fields=['is_chip_fault_formed']),
        ]

class AdvancedChipMetrics_CY(BaseAdvancedChipMetrics):
    stock = models.ForeignKey(
        'StockInfo',
        on_delete=models.CASCADE,
        related_name='advanced_chip_metrics_cy',
        verbose_name='股票',
        db_index=True
    )
    class Meta(BaseAdvancedChipMetrics.Meta):
        abstract = False
        verbose_name = '高级筹码指标-创业(V6.0-衍生固化)'
        verbose_name_plural = verbose_name
        db_table = 'stock_advanced_chip_metrics_cy'
        unique_together = ('stock', 'trade_time')
        indexes = [
            models.Index(fields=['stock', 'trade_time']),
            models.Index(fields=['chip_health_score']),
            models.Index(fields=['is_chip_fault_formed']),
        ]

class AdvancedChipMetrics_KC(BaseAdvancedChipMetrics):
    stock = models.ForeignKey(
        'StockInfo',
        on_delete=models.CASCADE,
        related_name='advanced_chip_metrics_kc',
        verbose_name='股票',
        db_index=True
    )
    class Meta(BaseAdvancedChipMetrics.Meta):
        abstract = False
        verbose_name = '高级筹码指标-科创(V6.0-衍生固化)'
        verbose_name_plural = verbose_name
        db_table = 'stock_advanced_chip_metrics_kc'
        unique_together = ('stock', 'trade_time')
        indexes = [
            models.Index(fields=['stock', 'trade_time']),
            models.Index(fields=['chip_health_score']),
            models.Index(fields=['is_chip_fault_formed']),
        ]

class AdvancedChipMetrics_BJ(BaseAdvancedChipMetrics):
    stock = models.ForeignKey(
        'StockInfo',
        on_delete=models.CASCADE,
        related_name='advanced_chip_metrics_bj',
        verbose_name='股票',
        db_index=True
    )
    class Meta(BaseAdvancedChipMetrics.Meta):
        abstract = False
        verbose_name = '高级筹码指标-北交(V6.0-衍生固化)'
        verbose_name_plural = verbose_name
        db_table = 'stock_advanced_chip_metrics_bj'
        unique_together = ('stock', 'trade_time')
        indexes = [
            models.Index(fields=['stock', 'trade_time']),
            models.Index(fields=['chip_health_score']),
            models.Index(fields=['is_chip_fault_formed']),
        ]

# 资金高级指标模型
class BaseAdvancedFundFlowMetrics(models.Model):
    """
    【V6.0 · 资金博弈三体模型版】
    - 核心革命: 废弃旧的扁平化指标体系，引入“力量格局”、“战术日志”、“战果评估”三体模型，
                  构建从力量对比到战术意图，再到战果评估的完整分析闭环。
    - 核心新增:
      1. `main_force_opening_blitz`: 主力开盘闪击，衡量开盘抢筹意愿。
      2. `main_force_closing_assault`: 主力尾盘偷袭，识别“聪明钱”行为。
      3. `main_force_vwap_adherence`: 主力VWAP依从度，揭示交易的主动性与预期。
      4. `flow_momentum_reversal`: 资金动能反转，捕捉日内多空力量的转折点。
    """
    trade_time = models.DateField(verbose_name='交易日期', db_index=True)
    # --- 第一体: 力量格局 (Power Structure) ---
    POWER_STRUCTURE_METRICS = {
        'net_flow_consensus': '共识-资金净流入(万元)',
        'main_force_net_flow_consensus': '共识-主力净流入(万元)',
        'retail_net_flow_consensus': '共识-散户净流入(万元)',
        'net_xl_amount_consensus': '共识-超大单净流入(万元)',
        'net_lg_amount_consensus': '共识-大单净流入(万元)',
        'net_md_amount_consensus': '共识-中单净流入(万元)',
        'net_sh_amount_consensus': '共识-小单净流入(万元)',
        'source_consistency_score': '多源一致性评分',
        'flow_divergence_mf_vs_retail': '资金分歧度(主力-散户)',
        'main_force_flow_intensity_ratio': '主力资金流强度比率(vs成交额)',
        'main_force_flow_impact_ratio': '主力资金流冲击比率(vs流通市值)',
        'trade_concentration_index': '交易集中度指数(大单/总成交)',
        'main_force_conviction_ratio': '主力信念比率(超大单/大单)',
        'avg_order_value_norm_price': '价格归一化平均订单价值',
    }
    # --- 第二体: 战术日志 (Tactical Log) ---
    TACTICAL_LOG_METRICS = {
        'main_force_support_strength': '主力支撑强度',
        'main_force_distribution_pressure': '主力派发压力',
        'retail_capitulation_score': '散户投降分',
        'closing_strength_index': '收盘强度指数',
        'final_hour_momentum': '尾盘动能',
        'aggression_index_opening': '开盘进攻性指数',
        'avg_cost_main_buy': '主力买入均价(PVWAP)',
        'avg_cost_main_sell': '主力卖出均价(PVWAP)',
        'avg_cost_retail_buy': '散户买入均价(PVWAP)',
        'avg_cost_retail_sell': '散户卖出均价(PVWAP)',
        'cost_divergence_mf_vs_retail': '成本分歧度(主力买-散户卖)',
        'market_cost_battle': '市场成本博弈差(主力买-散户买)',
        'daily_vwap': '当日成交加权平均价',
        # [代码新增开始]
        'main_force_opening_blitz': '主力开盘闪击(%)',
        'main_force_closing_assault': '主力尾盘偷袭(%)',
        'main_force_vwap_adherence': '主力VWAP依从度(%)',
        'flow_momentum_reversal': '资金动能反转',
        # [代码新增结束]
    }
    # --- 第三体: 战果评估 (Outcome Assessment) ---
    OUTCOME_ASSESSMENT_METRICS = {
        'intraday_execution_alpha': '日内执行Alpha',
        'main_force_intraday_profit': '主力日内盈亏(万元)',
        'realized_profit_on_exchange': '已实现利润(T+0置换)(万元)',
        'unrealized_pnl_on_net_change': '新增头寸浮动盈亏(万元)',
        'pnl_matrix_confidence_score': 'P&L矩阵可信度评分',
        'intraday_volatility': '日内波动率',
        'close_vs_vwap_ratio': '收盘价与VWAP偏离度',
    }
    CORE_METRICS = {
        **POWER_STRUCTURE_METRICS,
        **TACTICAL_LOG_METRICS,
        **OUTCOME_ASSESSMENT_METRICS,
    }
    SLOPE_ACCEL_EXCLUSIONS = [
        'source_consistency_score', 'pnl_matrix_confidence_score',
        'main_force_support_strength', 'main_force_distribution_pressure', 'retail_capitulation_score',
        'intraday_execution_alpha', 'closing_strength_index', 'final_hour_momentum',
        'aggression_index_opening', 'daily_vwap',
        'avg_cost_main_buy', 'avg_cost_main_sell', 'avg_cost_retail_buy', 'avg_cost_retail_sell',
        # [代码修改开始]
        # 新增指标也属于事件驱动型，不适合计算导数
        'main_force_opening_blitz', 'main_force_closing_assault',
        'main_force_vwap_adherence', 'flow_momentum_reversal',
        # [代码修改结束]
    ]
    FLOAT_METRICS = [
        'source_consistency_score', 'main_force_flow_intensity_ratio', 'main_force_flow_impact_ratio',
        'main_force_support_strength', 'main_force_distribution_pressure', 'retail_capitulation_score',
        'intraday_execution_alpha', 'intraday_volatility', 'closing_strength_index',
        'close_vs_vwap_ratio', 'final_hour_momentum', 'trade_concentration_index',
        'avg_order_value_norm_price', 'main_force_conviction_ratio',
        'pnl_matrix_confidence_score', 'aggression_index_opening',
        # [代码修改开始]
        'main_force_opening_blitz', 'main_force_closing_assault',
        'main_force_vwap_adherence', 'flow_momentum_reversal',
        # [代码修改结束]
    ]
    for name, verbose in CORE_METRICS.items():
        if name in FLOAT_METRICS:
            vars()[name] = models.FloatField(verbose_name=verbose, null=True, blank=True)
        else:
            vars()[name] = models.DecimalField(max_digits=22, decimal_places=6, verbose_name=verbose, null=True, blank=True)
    main_force_buy_rate_consensus = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='共识-主力买入率(%)', null=True, blank=True)
    UNIFIED_PERIODS = [1, 5, 13, 21, 55]
    sum_cols = [
        'net_flow_consensus', 'main_force_net_flow_consensus', 'retail_net_flow_consensus',
        'net_xl_amount_consensus', 'net_lg_amount_consensus', 'net_md_amount_consensus',
        'net_sh_amount_consensus', 'realized_profit_on_exchange', 'unrealized_pnl_on_net_change',
    ]
    for p in UNIFIED_PERIODS:
        if p > 1:
            for name in sum_cols:
                if name in CORE_METRICS:
                    verbose_name = CORE_METRICS.get(name, name)
                    vars()[f'{name}_sum_{p}d'] = models.DecimalField(max_digits=24, decimal_places=6, verbose_name=f'{verbose_name}{p}日累计', null=True, blank=True)
    all_derivable_metrics = list(CORE_METRICS.keys())
    for p in UNIFIED_PERIODS:
        if p > 1:
            for name in sum_cols:
                all_derivable_metrics.append(f'{name}_sum_{p}d')
    for name in all_derivable_metrics:
        base_name = name.split('_sum_')[0]
        if base_name in SLOPE_ACCEL_EXCLUSIONS:
            continue
        verbose_name = CORE_METRICS.get(name, name)
        if '_sum_' in name:
            original_verbose = CORE_METRICS.get(base_name, base_name)
            period_str = name.split('_sum_')[1].split('d')[0]
            verbose_name = f'{original_verbose}{period_str}日累计'
        for p in UNIFIED_PERIODS:
            vars()[f'{name}_slope_{p}d'] = models.FloatField(verbose_name=f'{verbose_name}{p}日斜率', null=True, blank=True)
            vars()[f'{name}_accel_{p}d'] = models.FloatField(verbose_name=f'{verbose_name}{p}日加速度', null=True, blank=True)
    class Meta:
        abstract = True
        ordering = ['-trade_time']

class AdvancedFundFlowMetrics_SH(BaseAdvancedFundFlowMetrics):
    stock = models.ForeignKey(
        'StockInfo',
        on_delete=models.CASCADE,
        related_name='advanced_fund_flow_metrics_sh',
        verbose_name='股票',
        db_index=True
    )
    class Meta(BaseAdvancedFundFlowMetrics.Meta):
        abstract = False
        verbose_name = '高级资金指标-上海'
        verbose_name_plural = verbose_name
        db_table = 'stock_advanced_fund_flow_metrics_sh'
        unique_together = ('stock', 'trade_time')
        indexes = [
            models.Index(fields=['stock', 'trade_time']),
        ]

class AdvancedFundFlowMetrics_SZ(BaseAdvancedFundFlowMetrics):
    stock = models.ForeignKey(
        'StockInfo',
        on_delete=models.CASCADE,
        related_name='advanced_fund_flow_metrics_sz',
        verbose_name='股票',
        db_index=True
    )
    class Meta(BaseAdvancedFundFlowMetrics.Meta):
        abstract = False
        verbose_name = '高级资金指标-深圳'
        verbose_name_plural = verbose_name
        db_table = 'stock_advanced_fund_flow_metrics_sz'
        unique_together = ('stock', 'trade_time')
        indexes = [
            models.Index(fields=['stock', 'trade_time']),
        ]

class AdvancedFundFlowMetrics_CY(BaseAdvancedFundFlowMetrics):
    stock = models.ForeignKey(
        'StockInfo',
        on_delete=models.CASCADE,
        related_name='advanced_fund_flow_metrics_cy',
        verbose_name='股票',
        db_index=True
    )
    class Meta(BaseAdvancedFundFlowMetrics.Meta):
        abstract = False
        verbose_name = '高级资金指标-创业板'
        verbose_name_plural = verbose_name
        db_table = 'stock_advanced_fund_flow_metrics_cy'
        unique_together = ('stock', 'trade_time')
        indexes = [
            models.Index(fields=['stock', 'trade_time']),
        ]

class AdvancedFundFlowMetrics_KC(BaseAdvancedFundFlowMetrics):
    stock = models.ForeignKey(
        'StockInfo',
        on_delete=models.CASCADE,
        related_name='advanced_fund_flow_metrics_kc',
        verbose_name='股票',
        db_index=True
    )
    class Meta(BaseAdvancedFundFlowMetrics.Meta):
        abstract = False
        verbose_name = '高级资金指标-科创板'
        verbose_name_plural = verbose_name
        db_table = 'stock_advanced_fund_flow_metrics_kc'
        unique_together = ('stock', 'trade_time')
        indexes = [
            models.Index(fields=['stock', 'trade_time']),
        ]

class AdvancedFundFlowMetrics_BJ(BaseAdvancedFundFlowMetrics):
    stock = models.ForeignKey(
        'StockInfo',
        on_delete=models.CASCADE,
        related_name='advanced_fund_flow_metrics_bj',
        verbose_name='股票',
        db_index=True
    )
    class Meta(BaseAdvancedFundFlowMetrics.Meta):
        abstract = False
        verbose_name = '高级资金指标-北京'
        verbose_name_plural = verbose_name
        db_table = 'stock_advanced_fund_flow_metrics_bj'
        unique_together = ('stock', 'trade_time')
        indexes = [
            models.Index(fields=['stock', 'trade_time']),
        ]

# 结构与行为高级指标模型
class BaseAdvancedStructuralMetrics(models.Model):
    """
    【V3.0 · 战场动力学模型版】
    - 核心革命: 废弃旧的现象描述指标，引入“能量密度”、“控制权”、“博弈效率”三大战场动力学核心要素，
                  从物理学和博弈论层面，对日内结构进行根本性重构。
    - 核心新增:
      1. `normalized_intraday_volatility`: 标准化日内波动率，实现跨股票可比的能量评估。
      2. `volume_asymmetry_index`: 成交量非对称指数，量化净买/卖盘的成交量优势。
      3. `vwap_deviation_area`: VWAP偏离面积，衡量多空对日内定价权的掌控程度。
      4. `trend_persistence_index`: 趋势持续性指数(Hurst)，判断日内走势是趋势市还是震荡市。
      5. `vpa_efficiency`: 量价分析效率，评估成交量推动价格的有效性和内耗程度。
    """
    trade_time = models.DateField(verbose_name='交易日期', db_index=True)
    # [代码修改开始]
    # --- 第一维度: 能量密度 (Energy Density) ---
    ENERGY_DENSITY_METRICS = {
        'normalized_intraday_volatility': '标准化日内波动率',
        'volume_asymmetry_index': '成交量非对称指数',
        'intraday_volume_gini': '日内成交量基尼系数',
        'auction_volume_ratio': '集合竞价成交量占比',
    }
    # --- 第二维度: 控制权 (Control) ---
    CONTROL_METRICS = {
        'vwap_deviation_area': 'VWAP偏离面积',
        'trend_persistence_index': '趋势持续性指数(Hurst)',
        'volume_weighted_time_index': '成交量加权时间指数',
        'close_vs_vpoc_ratio': '收盘价/VPOC比',
        'auction_price_impact': '集合竞价价格冲击',
        'auction_conviction_index': '集合竞价强度指数',
        'upper_shadow_volume_ratio': '上影线成交量占比',
        'lower_shadow_volume_ratio': '下影线成交量占比',
    }
    # --- 第三维度: 博弈效率 (Game Efficiency) ---
    GAME_EFFICIENCY_METRICS = {
        'vpa_efficiency': '量价分析效率',
        'intraday_trend_efficiency': '日内趋势效率(旧)',
        'intraday_reversal_intensity': '日内反转强度',
        'true_daily_cmf': '高保真CMF',
        'is_intraday_bullish_divergence': '是否存在日内底部背离',
        'is_intraday_bearish_divergence': '是否存在日内顶部背离',
    }
    # --- 辅助性结构指标 (保留，但重要性降低) ---
    AUXILIARY_METRICS = {
        'intraday_vpoc': '日内成交峰值(VPOC)',
        'intraday_vah': '日内价值上轨(VAH)',
        'intraday_val': '日内价值下轨(VAL)',
        'am_pm_volume_ratio': '上下午成交量比',
        'am_pm_vwap_ratio': '上下午VWAP比',
    }
    CORE_METRICS = {
        **ENERGY_DENSITY_METRICS,
        **CONTROL_METRICS,
        **GAME_EFFICIENCY_METRICS,
        **AUXILIARY_METRICS,
    }
    # [代码修改结束]
    UNIFIED_PERIODS = [1, 5, 13, 21, 55]
    BOOLEAN_FIELDS = ['is_intraday_bullish_divergence', 'is_intraday_bearish_divergence']
    SLOPE_ACCEL_EXCLUSIONS = [
        'is_intraday_bullish_divergence', 'is_intraday_bearish_divergence',
        'auction_volume_ratio', 'auction_price_impact', 'auction_conviction_index',
        'trend_persistence_index', # Hurst指数本身是状态量，不适合求导
    ]
    for name, verbose in CORE_METRICS.items():
        if name in BOOLEAN_FIELDS:
            vars()[name] = models.BooleanField(verbose_name=verbose, default=False)
        else:
            vars()[name] = models.FloatField(verbose_name=verbose, null=True, blank=True)
        if name in SLOPE_ACCEL_EXCLUSIONS:
            continue
        for p in UNIFIED_PERIODS:
            vars()[f'{name}_slope_{p}d'] = models.FloatField(verbose_name=f'{verbose}{p}日斜率', null=True, blank=True)
            vars()[f'{name}_accel_{p}d'] = models.FloatField(verbose_name=f'{verbose}{p}日加速度', null=True, blank=True)
    class Meta:
        abstract = True
        ordering = ['-trade_time']

class AdvancedStructuralMetrics_SH(BaseAdvancedStructuralMetrics):
    stock = models.ForeignKey(
        'StockInfo',
        on_delete=models.CASCADE,
        related_name='advanced_structural_metrics_sh',
        verbose_name='股票',
        db_index=True
    )
    class Meta(BaseAdvancedStructuralMetrics.Meta):
        abstract = False
        verbose_name = '高级结构与行为指标-上海'
        verbose_name_plural = verbose_name
        db_table = 'stock_advanced_structural_metrics_sh'
        unique_together = ('stock', 'trade_time')
        indexes = [
            models.Index(fields=['stock', 'trade_time']),
        ]

class AdvancedStructuralMetrics_SZ(BaseAdvancedStructuralMetrics):
    stock = models.ForeignKey(
        'StockInfo',
        on_delete=models.CASCADE,
        related_name='advanced_structural_metrics_sz',
        verbose_name='股票',
        db_index=True
    )
    class Meta(BaseAdvancedStructuralMetrics.Meta):
        abstract = False
        verbose_name = '高级结构与行为指标-深圳'
        verbose_name_plural = verbose_name
        db_table = 'stock_advanced_structural_metrics_sz'
        unique_together = ('stock', 'trade_time')
        indexes = [
            models.Index(fields=['stock', 'trade_time']),
        ]

class AdvancedStructuralMetrics_CY(BaseAdvancedStructuralMetrics):
    stock = models.ForeignKey(
        'StockInfo',
        on_delete=models.CASCADE,
        related_name='advanced_structural_metrics_cy',
        verbose_name='股票',
        db_index=True
    )
    class Meta(BaseAdvancedStructuralMetrics.Meta):
        abstract = False
        verbose_name = '高级结构与行为指标-创业板'
        verbose_name_plural = verbose_name
        db_table = 'stock_advanced_structural_metrics_cy'
        unique_together = ('stock', 'trade_time')
        indexes = [
            models.Index(fields=['stock', 'trade_time']),
        ]

class AdvancedStructuralMetrics_KC(BaseAdvancedStructuralMetrics):
    stock = models.ForeignKey(
        'StockInfo',
        on_delete=models.CASCADE,
        related_name='advanced_structural_metrics_kc',
        verbose_name='股票',
        db_index=True
    )
    class Meta(BaseAdvancedStructuralMetrics.Meta):
        abstract = False
        verbose_name = '高级结构与行为指标-科创板'
        verbose_name_plural = verbose_name
        db_table = 'stock_advanced_structural_metrics_kc'
        unique_together = ('stock', 'trade_time')
        indexes = [
            models.Index(fields=['stock', 'trade_time']),
        ]

class AdvancedStructuralMetrics_BJ(BaseAdvancedStructuralMetrics):
    stock = models.ForeignKey(
        'StockInfo',
        on_delete=models.CASCADE,
        related_name='advanced_structural_metrics_bj',
        verbose_name='股票',
        db_index=True
    )
    class Meta(BaseAdvancedStructuralMetrics.Meta):
        abstract = False
        verbose_name = '高级结构与行为指标-北京'
        verbose_name_plural = verbose_name
        db_table = 'stock_advanced_structural_metrics_bj'
        unique_together = ('stock', 'trade_time')
        indexes = [
            models.Index(fields=['stock', 'trade_time']),
        ]










