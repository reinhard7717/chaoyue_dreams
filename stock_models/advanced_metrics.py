# stock_models\advanced_metrics.py
from django.db import models
from django.utils.translation import gettext_lazy as _
import pandas as pd

# 筹码高级指标模型
class BaseAdvancedChipMetrics(models.Model):
    """
    【V18.0 · 指标精炼版】
    - 核心优化: 移除T+0套利指标的斜率和加速度计算，避免引入噪音。
    """
    trade_time = models.DateField(verbose_name='交易日期', db_index=True)
    CORE_METRICS = {
        'peak_cost': '主筹码峰成本',
        'peak_percent': '主筹码峰占比(%)',
        'peak_volume': '主筹码峰成交量(股)',
        'peak_stability': '筹码峰稳定性(几何)',
        'peak_defense_intensity': '主峰防守强度(%)',
        'peak_vwap_deviation': '主峰VWAP偏离度(%)',
        'peak_net_volume_flow': '主峰净成交量流向',
        'is_multi_peak': '是否多峰形态',
        'secondary_peak_cost': '次筹码峰成本',
        'peak_distance_ratio': '主次峰距离比',
        'peak_strength_ratio': '主次峰强度比',
        'concentration_70pct': '70%筹码集中度',
        'concentration_90pct': '90%筹码集中度',
        'pressure_above': '上方2%套牢盘(%)',
        'support_below': '下方2%支撑盘(%)',
        'realized_pressure_intensity': '真实压力强度(%)',
        'realized_support_intensity': '真实支撑强度(%)',
        'concentration_increase_by_support': '承接增集度',
        'concentration_increase_by_chasing': '追涨增集度',
        'concentration_decrease_by_distribution': '派发减集度',
        'concentration_decrease_by_capitulation': '割肉减集度',
        'main_force_suppressive_accumulation': '主力打压吸筹占比(%)',
        'retail_suppressive_accumulation': '散户打压吸筹占比(%)',
        'main_force_rally_distribution': '主力拉高出货占比(%)',
        'retail_rally_distribution': '散户拉高出货占比(%)',
        'main_force_capitulation_distribution': '主力恐慌派发占比(%)',
        'retail_capitulation_distribution': '散户恐慌割肉占比(%)',
        'main_force_chasing_accumulation': '主力追涨吸筹占比(%)',
        'retail_chasing_accumulation': '散户追涨抬轿占比(%)',
        'main_force_t0_arbitrage': '主力高抛低吸占比(%)',
        'retail_t0_arbitrage': '散户高抛低吸占比(%)',
        'short_term_profit_taking_ratio': '短期获利盘兑现占比(%)',
        'long_term_chips_unlocked_ratio': '长期锁定盘解锁占比(%)',
        'short_term_capitulation_ratio': '短期套牢盘割肉占比(%)',
        'long_term_despair_selling_ratio': '长期套牢盘绝望占比(%)',
        'total_winner_rate': '总获利盘(%)',
        'total_loser_rate': '总套牢盘(%)',
        'winner_rate_short_term': '短期获利盘(%)',
        'winner_rate_long_term': '长期锁定盘(%)',
        'loser_rate_short_term': '短期套牢盘(%)',
        'loser_rate_long_term': '长期套牢盘(%)',
        'pressure_above_volume': '上方套牢盘绝对量(股)',
        'support_below_volume': '下方支撑盘绝对量(股)',
        'prev_20d_close': '20日前收盘价',
        'peak_control_ratio': '筹码峰控盘比(%)',
        'winner_avg_cost': '获利盘平均成本',
        'winner_profit_margin': '获利盘安全垫(%)',
        'profit_taking_urgency': '获利盘兑现紧迫度(%)',
        'profit_realization_premium': '利润兑现溢价(%)',
        'avg_cost_short_term': '短期持仓者平均成本',
        'avg_cost_long_term': '长期持仓者平均成本',
        'price_to_peak_ratio': '股价/筹码峰成本比',
        'chip_zscore': '筹码Z-Score',
        'chip_fault_strength': '筹码断层强度',
        'chip_fault_vacuum_percent': '断层真空区筹码占比(%)',
        'is_chip_fault_formed': '是否形成筹码断层',
        'fault_breakthrough_intensity': '断层突破强度',
        'intraday_volume_gini': '日内成交基尼系数',
        'volume_weighted_time_index': '成交量加权时间指数',
        'intraday_trend_efficiency': '日内趋势效率',
        'am_pm_vwap_ratio': '上下午VWAP比(%)',
        'chip_health_score': '筹码健康分(0-100)',
        'cost_divergence': '成本发散度',
        'turnover_at_peak_ratio': '主峰成交占比(%)',
    }
    UNIFIED_PERIODS = [1, 5, 13, 21, 55]
    INTEGER_FIELDS = ['peak_volume', 'pressure_above_volume', 'support_below_volume']
    BOOLEAN_FIELDS = ['is_multi_peak', 'is_chip_fault_formed']
    # 不应计算斜率和加速度的指标完整列表
    SLOPE_ACCEL_EXCLUSIONS = [
        # T+0套利类
        'main_force_t0_arbitrage', 'retail_t0_arbitrage',
        # 集中度动态归因类
        'concentration_increase_by_support', 'concentration_increase_by_chasing',
        'concentration_decrease_by_distribution', 'concentration_decrease_by_capitulation',
        # 主力/散户筹码交互类
        'main_force_suppressive_accumulation', 'retail_suppressive_accumulation',
        'main_force_rally_distribution', 'retail_rally_distribution',
        'main_force_capitulation_distribution', 'retail_capitulation_distribution',
        'main_force_chasing_accumulation', 'retail_chasing_accumulation',
        # 跨日筹码迁徙类
        'short_term_profit_taking_ratio', 'long_term_chips_unlocked_ratio',
        'short_term_capitulation_ratio', 'long_term_despair_selling_ratio',
        # 事件驱动及高波动类
        'fault_breakthrough_intensity', 'intraday_trend_efficiency', 'am_pm_vwap_ratio',
        # 静态参考值
        'prev_20d_close',
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
        # 增加判断，跳过对T+0指标的斜率和加速度计算
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
    【V5.0 · 语义化模型定义版】
    - 核心重构: 彻底废弃基于名称关键字的脆弱字段类型判断。
    - 核心逻辑: 改为基于数据语义定义字段类型。所有货币/价格/金额/价值指标统一使用高精度的DecimalField，
                所有比率/分数/指数/纯数字指标统一使用FloatField。
                这从根本上解决了因字段类型不当导致的数值存储错误问题。
    """
    trade_time = models.DateField(verbose_name='交易日期', db_index=True)
    # 统一P&L相关指标的单位为“万元”
    CORE_METRICS = {
        'net_flow_consensus': '共识-资金净流入(万元)',
        'main_force_net_flow_consensus': '共识-主力净流入(万元)',
        'retail_net_flow_consensus': '共识-散户净流入(万元)',
        'net_xl_amount_consensus': '共识-超大单净流入(万元)',
        'net_lg_amount_consensus': '共识-大单净流入(万元)',
        'net_md_amount_consensus': '共识-中单净流入(万元)',
        'net_sh_amount_consensus': '共识-小单净流入(万元)',
        'source_consistency_score': '多源一致性评分',
        'flow_internal_friction_ratio': '资金内部摩擦系数',
        'consensus_calibrated_main_flow': '共识校准主力净流入(万元)',
        'consensus_flow_weighted': '加权共识主力净流入(万元)',
        'cross_source_divergence_std': '多源分歧标准差',
        'flow_divergence_mf_vs_retail': '资金分歧度(主力-散户)',
        'main_force_flow_intensity_ratio': '主力资金流强度比率',
        'main_force_flow_impact_ratio': '主力资金流影响力',
        'trade_granularity_impact': '交易颗粒度影响力',
        'main_force_vs_xl_divergence': '主力与超大单分歧度(万元)',
        'main_force_support_strength': '主力支撑强度',
        'main_force_distribution_pressure': '主力派发压力',
        'retail_capitulation_score': '散户投降分',
        'intraday_execution_alpha': '日内执行Alpha',
        'intraday_volatility': '日内波动率',
        'closing_strength_index': '收盘强度指数',
        'close_vs_vwap_ratio': '收盘价与VWAP偏离度',
        'final_hour_momentum': '尾盘动能',
        'trade_concentration_index': '交易集中度指数',
        'avg_order_value': '平均每笔成交金额(元)',
        'avg_order_value_norm_price': '价格归一化平均订单价值',
        'main_force_conviction_ratio': '主力信念比率',
        'avg_cost_sm_buy': '小单买入均价(PVWAP)', 'avg_cost_sm_sell': '小单卖出均价(PVWAP)',
        'avg_cost_md_buy': '中单买入均价(PVWAP)', 'avg_cost_md_sell': '中单卖出均价(PVWAP)',
        'avg_cost_lg_buy': '大单买入均价(PVWAP)', 'avg_cost_lg_sell': '大单卖出均价(PVWAP)',
        'avg_cost_elg_buy': '特大单买入均价(PVWAP)', 'avg_cost_elg_sell': '特大单卖出均价(PVWAP)',
        'avg_cost_main_buy': '主力买入均价(PVWAP)', 'avg_cost_main_sell': '主力卖出均价(PVWAP)',
        'avg_cost_retail_buy': '散户买入均价(PVWAP)', 'avg_cost_retail_sell': '散户卖出均价(PVWAP)',
        'cost_divergence_mf_vs_retail': '成本分歧度(主力买-散户卖)',
        'cost_weighted_main_flow': '主力成本加权净流入',
        'main_buy_cost_advantage': '主力成本领先度(vs Close)',
        'realized_profit_on_exchange': '已实现利润(T+0置换)(万元)',
        'net_position_change_value': '净头寸变动市值(万元)',
        'unrealized_pnl_on_net_change': '新增头寸浮动盈亏(万元)',
        'pnl_matrix_confidence_score': 'P&L矩阵可信度评分',
        'main_force_intraday_profit': '主力日内盈亏(万元)',
        'market_cost_battle': '市场成本博弈差(主力买-散户买)',
        'daily_vwap': '当日成交加权平均价',
        'main_buy_cost_vs_vwap': '主力买入成本 vs VWAP',
        'main_sell_cost_vs_vwap': '主力卖出成本 vs VWAP',
        'vwap_tracking_error': 'VWAP算法跟踪误差',
        'volume_profile_jsd_vs_uniform': '成交量分布均匀度(JS散度)',
        'aggression_index_opening': '开盘进攻性指数',
        'divergence_ts_ths': '分歧度(Tushare-同花顺)',
        'divergence_ts_dc': '分歧度(Tushare-东方财富)',
        'divergence_ths_dc': '分歧度(同花顺-东方财富)',
    }
    
    SLOPE_ACCEL_EXCLUSIONS = [
        'source_consistency_score', 'flow_internal_friction_ratio', 'cross_source_divergence_std',
        'divergence_ts_ths', 'divergence_ts_dc', 'divergence_ths_dc',
        'pnl_matrix_confidence_score', 'volume_profile_jsd_vs_uniform',
        'main_force_support_strength', 'main_force_distribution_pressure', 'retail_capitulation_score',
        'intraday_execution_alpha', 'closing_strength_index', 'final_hour_momentum',
        'aggression_index_opening', 'vwap_tracking_error', 'realized_profit_on_exchange',
        'daily_vwap', 'avg_cost_sm_buy', 'avg_cost_sm_sell', 'avg_cost_md_buy', 'avg_cost_md_sell',
        'avg_cost_lg_buy', 'avg_cost_lg_sell', 'avg_cost_elg_buy', 'avg_cost_elg_sell',
        'avg_cost_main_buy', 'avg_cost_main_sell', 'avg_cost_retail_buy', 'avg_cost_retail_sell',
    ]
    # 彻底重构核心指标的字段定义逻辑
    # 步骤1: 明确定义所有应为 FloatField 的指标（比率、分数、指数、纯数字等）
    FLOAT_METRICS = [
        'source_consistency_score', 'flow_internal_friction_ratio', 'cross_source_divergence_std',
        'main_force_flow_intensity_ratio', 'main_force_flow_impact_ratio', 'trade_granularity_impact',
        'main_force_support_strength', 'main_force_distribution_pressure', 'retail_capitulation_score',
        'intraday_execution_alpha', 'intraday_volatility', 'closing_strength_index',
        'close_vs_vwap_ratio', 'final_hour_momentum', 'trade_concentration_index',
        'avg_order_value_norm_price', 'main_force_conviction_ratio', 'main_buy_cost_advantage',
        'pnl_matrix_confidence_score', 'volume_profile_jsd_vs_uniform',
        'aggression_index_opening',
    ]
    # 步骤2: 循环定义核心指标字段，根据语义分配正确的类型
    for name, verbose in CORE_METRICS.items():
        if name in FLOAT_METRICS:
            # 如果是比率、分数、指数等，使用 FloatField
            vars()[name] = models.FloatField(verbose_name=verbose, null=True, blank=True)
        else:
            # 否则，默认为货币、价格、金额、价值，使用高精度的 DecimalField
            # 增加精度以容纳价格和金额
            vars()[name] = models.DecimalField(max_digits=22, decimal_places=6, verbose_name=verbose, null=True, blank=True)
    
    main_force_buy_rate_consensus = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='共识-主力买入率(%)', null=True, blank=True)
    UNIFIED_PERIODS = [1, 5, 13, 21, 55]
    # 重构衍生指标的定义循环，使其更清晰并确保类型正确
    # 步骤3: 定义累计值字段
    sum_cols = [
        'net_flow_consensus', 'main_force_net_flow_consensus', 'retail_net_flow_consensus',
        'net_xl_amount_consensus', 'net_lg_amount_consensus', 'net_md_amount_consensus',
        'net_sh_amount_consensus', 'cost_weighted_main_flow',
        'consensus_calibrated_main_flow',
        'consensus_flow_weighted',
    ]
    for p in UNIFIED_PERIODS:
        if p > 1:
            for name in sum_cols:
                if name in CORE_METRICS:
                    verbose_name = CORE_METRICS.get(name, name)
                    # 累计值是货币类型，使用 DecimalField
                    vars()[f'{name}_sum_{p}d'] = models.DecimalField(max_digits=24, decimal_places=6, verbose_name=f'{verbose_name}{p}日累计', null=True, blank=True)
    # 步骤4: 为所有可衍生的指标（核心指标+累计值指标）统一定义斜率和加速度字段
    all_derivable_metrics = list(CORE_METRICS.keys())
    for p in UNIFIED_PERIODS:
        if p > 1:
            for name in sum_cols:
                all_derivable_metrics.append(f'{name}_sum_{p}d')
    for name in all_derivable_metrics:
        # 检查该指标是否在排除列表中
        base_name = name.split('_sum_')[0]
        if base_name in SLOPE_ACCEL_EXCLUSIONS:
            continue
        # 获取正确的 verbose_name
        verbose_name = CORE_METRICS.get(name, name) # 尝试获取原始名称
        if '_sum_' in name:
            original_verbose = CORE_METRICS.get(base_name, base_name)
            period_str = name.split('_sum_')[1].split('d')[0]
            verbose_name = f'{original_verbose}{period_str}日累计'
        # 循环定义斜率和加速度
        for p in UNIFIED_PERIODS:
            # 斜率和加速度是纯数字，使用 FloatField
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
    【V2.2 · 竞价指标固化版】高级结构与行为指标模型
    - 核心升级: 新增对收盘集合竞价的分析指标，并将其加入导数计算的排除列表。
    """
    trade_time = models.DateField(verbose_name='交易日期', db_index=True)
    CORE_METRICS = {
        'volume_weighted_close_position': '成交量加权收盘位置',
        'upper_shadow_volume_ratio': '上影线成交量占比',
        'lower_shadow_volume_ratio': '下影线成交量占比',
        'true_daily_cmf': '高保真CMF',
        'intraday_trend_efficiency': '日内趋势效率',
        'intraday_reversal_intensity': '日内反转强度',
        'intraday_vpoc': '日内成交峰值(VPOC)',
        'intraday_vah': '日内价值上轨(VAH)',
        'intraday_val': '日内价值下轨(VAL)',
        'close_vs_vpoc_ratio': '收盘价/VPOC比',
        'am_pm_volume_ratio': '上下午成交量比',
        'am_pm_vwap_ratio': '上下午VWAP比',
        'intraday_volume_gini': '日内成交量基尼系数',
        'intraday_trend_linearity': '日内趋势线性度(R²)',
        'volume_weighted_time_index': '成交量加权时间指数',
        'is_intraday_bullish_divergence': '是否存在日内底部背离',
        'is_intraday_bearish_divergence': '是否存在日内顶部背离',
        # 新增集合竞价分析指标
        'auction_volume_ratio': '集合竞价成交量占比',
        'auction_price_impact': '集合竞价价格冲击',
        'auction_conviction_index': '集合竞价强度指数',
        
    }
    UNIFIED_PERIODS = [1, 5, 13, 21, 55]
    BOOLEAN_FIELDS = ['is_intraday_bullish_divergence', 'is_intraday_bearish_divergence']
    # 将新的竞价指标加入排除列表，因为它们是事件驱动型指标，不适合求导
    SLOPE_ACCEL_EXCLUSIONS = [
        'is_intraday_bullish_divergence',
        'is_intraday_bearish_divergence',
        'auction_volume_ratio',
        'auction_price_impact',
        'auction_conviction_index',
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










