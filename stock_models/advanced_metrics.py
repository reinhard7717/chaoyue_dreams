# stock_models\advanced_metrics.py
from django.db import models
from django.utils.translation import gettext_lazy as _
import pandas as pd

# 筹码高级指标模型
class BaseAdvancedChipMetrics(models.Model):
    """
    【V25.0 · 战术深化版】
    - 核心革命: 在第一象限“静态结构”中，引入盈利亏损质量、结构稳定性、成本分布形态三个全新的计算维度。
    - 核心新增:
      1. `winner_profit_cushion`: 获利盘缓冲垫，量化多方的安全边际。
      2. `loser_pain_index`: 套牢盘痛苦指数，评估空方的潜在投降压力。
      3. `structural_stability_score`: 结构稳定性评分，综合评估当前筹码结构的稳固程度。
      4. `cost_structure_skewness`: 成本结构偏度，从统计学角度判断成本重心的偏移方向。
    """
    trade_time = models.DateField(verbose_name='交易日期', db_index=True)
    # [代码修改开始]
    # --- 第一象限: 静态结构 (Static Structure) ---
    STATIC_STRUCTURE_METRICS = {
        'dominant_peak_cost': '主导峰成本',
        'dominant_peak_volume_ratio': '主导峰筹码占比(%)',
        'dominant_peak_mf_conviction': '主导峰主力信念',
        'dominant_peak_profit_margin': '主导峰利润边际(%)',
        'dominant_peak_breadth': '主导峰宽度(%)',
        'is_multi_peak': '是否多峰形态',
        'secondary_peak_cost': '次筹码峰成本',
        'peak_distance_volatility_ratio': '峰距波动率比',
        'peak_dynamic_strength_ratio': '动态强度比',
        'winner_concentration_90pct': '获利盘集中度',
        'loser_concentration_90pct': '套牢盘集中度',
        'long_term_concentration_90pct': '长期筹码集中度',
        'short_term_concentration_90pct': '短期筹码集中度',
        'dynamic_pressure_index': '动态压力指数',
        'dynamic_support_index': '动态支撑指数',
        'main_force_support_conviction_zone': '支撑区主力信念(%)',
        'chip_fault_magnitude': '筹码断层量级(ATR)',
        'chip_fault_blockage_ratio': '断层阻碍度(%)',
        'chip_fault_traversal_conviction': '断层穿越信念(%)',
        'chip_fault_status': '筹码断层状态',
        'total_winner_rate': '存量总获利盘(%)',
        'total_loser_rate': '存量总套牢盘(%)',
        'effective_winner_rate': '有效获利盘比例(%)',
        'winner_profit_margin_avg': '平均获利盘利润率(%)',
        'loser_loss_margin_avg': '平均套牢盘亏损率(%)',
        'active_winner_rate': '活跃获利盘比例(%)',
        'active_loser_rate': '活跃套牢盘比例(%)',
        'locked_profit_rate': '锁定利润盘比例(%)',
        'locked_loss_rate': '锁定亏损盘比例(%)',
        # --- 新增指标 ---
        'winner_profit_cushion': '获利盘缓冲垫(%)',
        'loser_pain_index': '套牢盘痛苦指数',
        'structural_stability_score': '结构稳定性评分(0-100)',
        'cost_structure_skewness': '成本结构偏度',
    }
    # [代码修改结束]
    # --- 第二象限: 内部动态 (Intraday Dynamics) ---
    INTRADAY_DYNAMICS_METRICS = {
        'active_selling_pressure': '主动卖压强度(%)',
        'active_buying_support': '主动买盘支撑(%)',
        'peak_battle_intensity': '主峰交战强度(%)',
        'peak_main_force_premium': '主峰主力溢价(%)',
        'peak_mf_conviction_flow': '主峰主力信念流(%)',
        'upward_impulse_purity': '上涨脉冲纯度(%)',
        'opening_gap_defense_strength': '开盘缺口防御强度',
        'active_zone_combat_intensity': '活跃战区交战强度(%)',
        'active_zone_mf_stance': '活跃战区主力姿态(%)',
        'profit_realization_quality': '获利盘兑现质量(%)',
        'capitulation_absorption_quality': '套牢盘承接质量(%)',
    }
    # --- 第三象限: 跨日迁徙 (Cross-Day Flow) ---
    CROSS_DAY_FLOW_METRICS = {
        'gathering_by_support': '承接式集结量(%)',
        'gathering_by_chasing': '追涨式集结量(%)',
        'dispersal_by_distribution': '派发式分散量(%)',
        'dispersal_by_capitulation': '承接式分散量(%)',
        'profit_taking_flow_ratio': '获利兑现流量占比(%)',
        'capitulation_flow_ratio': '恐慌抛售流量占比(%)',
        'active_winner_pressure_ratio': '活跃获利盘承压比(%)',
        'locked_profit_pressure_ratio': '锁定利润盘承压比(%)',
        'active_loser_pressure_ratio': '活跃套牢盘承压比(%)',
        'locked_loss_pressure_ratio': '锁定亏损盘承压比(%)',
        'cost_divergence': '成本分离度',
        'cost_divergence_normalized': '标准化成本分离度',
        'winner_loser_momentum': '盈亏动量',
        'chip_fatigue_index': '筹码疲劳指数',
    }
    # --- 第四象限: 博弈意图 (Game-Theoretic Intent) ---
    GAME_THEORY_METRICS = {
        'suppressive_accumulation_intensity': '打压吸筹强度(%)',
        'rally_distribution_intensity': '拉高出货强度(%)',
        'rally_accumulation_intensity': '追涨吸筹强度(%)',
        'panic_selling_intensity': '恐慌派发强度(%)',
        'main_force_cost_advantage': '主力成本优势(%)',
        'active_winner_avg_cost': '活跃获利盘均价',
        'active_winner_profit_margin': '活跃获利盘利润垫(%)',
        'short_term_holder_cost': '短期持仓者成本',
        'long_term_holder_cost': '长期持仓者成本',
        'winner_conviction_index': '获利盘信念指数',
        'chip_health_score': '筹码健康分(0-100)',
        'main_force_control_leverage': '主力控盘杠杆(%)',
        'loser_capitulation_pressure_index': '套牢盘投降压力指数',
        'estimated_main_force_position_cost': '主力预估持仓成本',
        'intraday_new_loser_pressure': '日内新增套牢盘压力',
        'closing_auction_control_signal': '集合竞价控盘信号(%)',
        'intraday_probe_rebound_quality': '日内试探回升质量',
    }
    # --- 第五象限: 生命体征 (Vital Signs) ---
    VITAL_SIGNS_METRICS = {
        'cost_structure_consensus_index': '成本结构共识指数',
        'chip_cost_momentum': '筹码成本动量',
        'structural_resilience_index': '结构韧性指数',
        'posture_control_score': '主力姿态-控盘分',
        'posture_action_score': '主力姿态-行动分',
    }
    CORE_METRICS = {
        **STATIC_STRUCTURE_METRICS,
        **INTRADAY_DYNAMICS_METRICS,
        **CROSS_DAY_FLOW_METRICS,
        **GAME_THEORY_METRICS,
        **VITAL_SIGNS_METRICS,
    }
    UNIFIED_PERIODS = [1, 5, 13, 21, 55]
    INTEGER_FIELDS = ['peak_volume', 'pressure_above_volume', 'support_below_volume', 'chip_fault_status']
    BOOLEAN_FIELDS = ['is_multi_peak']
    # [代码修改开始]
    # 核心修改: 将所有新的高阶复合指标加入斜率排除列表
    SLOPE_ACCEL_EXCLUSIONS = [
        # 动态归因类 (已经是变化量)
        'gathering_by_support', 'gathering_by_chasing',
        'dispersal_by_distribution', 'dispersal_by_capitulation',
        'chip_fault_status',
        'profit_taking_flow_ratio', 'capitulation_flow_ratio',
        'active_winner_pressure_ratio', 'locked_profit_pressure_ratio',
        'active_loser_pressure_ratio', 'locked_loss_pressure_ratio',
        # 战术序列类 (事件驱动，非连续)
        'suppressive_accumulation_intensity',
        'rally_distribution_intensity',
        'rally_accumulation_intensity',
        'panic_selling_intensity',
        'main_force_cost_advantage',
        'main_force_control_leverage',
        # 事件驱动及高波动类
        'fault_traversal_momentum', 'intraday_trend_efficiency',
        'intraday_new_loser_pressure',
        'closing_auction_control_signal',
        'intraday_probe_rebound_quality',
        # 生命体征指标 (高阶复合，不适合直接求导)
        'cost_structure_consensus_index',
        'chip_cost_momentum',
        'structural_resilience_index',
        'posture_control_score',
        'posture_action_score',
        # 新增的静态结构复合指标
        'loser_pain_index',
        'structural_stability_score',
        'cost_structure_skewness',
    ]
    # [代码修改结束]
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

class BaseAdvancedFundFlowMetrics(models.Model):
    """
    【V18.0 · 最终评估重构版】
    - 核心革命: 废弃旧的静态总结指标，引入“盈利质量”、“波动效率”和“收盘姿态”的动态评估模型。
    - 核心升级:
      1. `pnl_quality_score`: 评估主力盈利的“含金量”与“性价比”。
      2. `volatility_asymmetry_index`: 衡量日内波动的非对称性，识别涨跌的真实动能。
      3. `closing_price_deviation_score`: 结合收盘竞价能量，评估收盘价的真实强度。
    """
    trade_time = models.DateField(verbose_name='交易日期', db_index=True)
    POWER_STRUCTURE_METRICS = {
        'net_flow_calibrated': '校准后-资金净流入(万元)',
        'main_force_net_flow_calibrated': '校准后-主力净流入(万元)',
        'retail_net_flow_calibrated': '校准后-散户净流入(万元)',
        'net_xl_amount_calibrated': '校准后-超大单净流入(万元)',
        'net_lg_amount_calibrated': '校准后-大单净流入(万元)',
        'net_md_amount_calibrated': '校准后-中单净流入(万元)',
        'net_sh_amount_calibrated': '校准后-小单净流入(万元)',
        'flow_credibility_index': '资金流可信度指数(0-100)',
        'mf_retail_battle_intensity': '主力散户博弈烈度(%)',
        'main_force_activity_ratio': '主力活跃度(%)',
        'main_force_flow_directionality': '主力资金流向性(%)',
        'xl_order_flow_directionality': '超大单流向性(%)',
        'main_force_conviction_index': '主力信念指数',
        'inferred_active_order_size': '推断活跃订单规模(元)',
        'retail_flow_dominance_index': '散户流动性主导指数',
        'main_force_price_impact_ratio': '主力价格冲击比率',
    }
    TACTICAL_LOG_METRICS = {
        'dip_absorption_power': '逢低吸筹力度(%)',
        'rally_distribution_pressure': '拉高派发压力(%)',
        'panic_selling_cascade': '恐慌抛售级联(%)',
        'opening_battle_result': '开盘战役结果',
        'pre_closing_posturing': '收盘前姿态',
        'closing_auction_ambush': '收盘伏击强度',
        'avg_cost_main_buy': '主力买入均价(PVWAP)',
        'avg_cost_main_sell': '主力卖出均价(PVWAP)',
        'avg_cost_retail_buy': '散户买入均价(PVWAP)',
        'avg_cost_retail_sell': '散户卖出均价(PVWAP)',
        'main_force_execution_alpha': '主力执行Alpha(%)',
        'retail_panic_surrender_index': '散户恐慌投降指数(%)',
        'retail_fomo_premium_index': '散户追高溢价指数(%)',
        'main_force_t0_efficiency': '主力T+0效率(%)',
        'vwap_structure_skew': 'VWAP结构偏离度',
        'flow_efficiency_index': '资金效率指数',
        'asymmetric_volume_thrust': '非对称成交量推力',
    }
    # 核心修改: 重构“战果评估”指标体系
    OUTCOME_ASSESSMENT_METRICS = {
        'execution_cost_alpha': '执行成本Alpha(%)',
        't0_arbitrage_profit': 'T+0套利利润(万元)',
        'positional_pnl': '持仓变动盈亏(万元)',
        'total_trading_pnl': '总交易盈亏(万元)',
        'pnl_quality_score': '盈利质量评分',
        'volatility_asymmetry_index': '波动不对称指数',
        'closing_price_deviation_score': '收盘价偏离度得分',
    }
    CORE_METRICS = {
        **POWER_STRUCTURE_METRICS,
        **TACTICAL_LOG_METRICS,
        **OUTCOME_ASSESSMENT_METRICS,
    }
    # 核心修改: 更新排除列表和浮点数列表以匹配新的战果评估指标
    SLOPE_ACCEL_EXCLUSIONS = [
        'avg_cost_main_buy', 'avg_cost_main_sell', 'avg_cost_retail_buy', 'avg_cost_retail_sell',
        'flow_credibility_index', 'mf_retail_battle_intensity', 'main_force_activity_ratio',
        'main_force_flow_directionality', 'xl_order_flow_directionality', 'main_force_conviction_index',
        'retail_flow_dominance_index', 'main_force_price_impact_ratio', 'dip_absorption_power',
        'rally_distribution_pressure', 'panic_selling_cascade', 'opening_battle_result',
        'pre_closing_posturing', 'closing_auction_ambush', 'main_force_execution_alpha',
        'retail_panic_surrender_index', 'retail_fomo_premium_index', 'main_force_t0_efficiency',
        'vwap_structure_skew', 'flow_efficiency_index', 'asymmetric_volume_thrust',
        'execution_cost_alpha', 'pnl_quality_score', 'volatility_asymmetry_index',
        'closing_price_deviation_score',
    ]
    FLOAT_METRICS = [
        'flow_credibility_index', 'mf_retail_battle_intensity', 'main_force_activity_ratio',
        'main_force_flow_directionality', 'xl_order_flow_directionality', 'main_force_conviction_index',
        'inferred_active_order_size', 'retail_flow_dominance_index', 'main_force_price_impact_ratio',
        'dip_absorption_power', 'rally_distribution_pressure', 'panic_selling_cascade',
        'opening_battle_result', 'pre_closing_posturing', 'closing_auction_ambush',
        'main_force_execution_alpha', 'retail_panic_surrender_index',
        'retail_fomo_premium_index', 'main_force_t0_efficiency',
        'vwap_structure_skew', 'flow_efficiency_index', 'asymmetric_volume_thrust',
        'execution_cost_alpha', 'pnl_quality_score', 'volatility_asymmetry_index',
        'closing_price_deviation_score',
    ]
    for name, verbose in CORE_METRICS.items():
        if name in FLOAT_METRICS:
            vars()[name] = models.FloatField(verbose_name=verbose, null=True, blank=True)
        else:
            vars()[name] = models.DecimalField(max_digits=22, decimal_places=6, verbose_name=verbose, null=True, blank=True)
    main_force_buy_rate_consensus = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='共识-主力买入率(%)', null=True, blank=True)
    UNIFIED_PERIODS = [1, 5, 13, 21, 55]
    # 核心修改: 更新可累加列的列表
    sum_cols = [
        'net_flow_calibrated', 'main_force_net_flow_calibrated', 'retail_net_flow_calibrated',
        'net_xl_amount_calibrated', 'net_lg_amount_calibrated', 'net_md_amount_calibrated',
        'net_sh_amount_calibrated',
        't0_arbitrage_profit', 'positional_pnl', 'total_trading_pnl',
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
    【V5.0 · 战场动力学深化版】
    - 核心革命: 在“能量密度”维度中，引入反转韧性、高位换手意愿、开盘突袭纯度三个全新的战术指标，
                  从能量的释放方式、方向和效率上深度刻画日内博弈。
    - 核心新增:
      1. `rebound_momentum`: 反转动能。
      2. `high_level_consolidation_volume`: 高位整固成交量占比。
      3. `opening_period_thrust`: 开盘期推力。
    """
    trade_time = models.DateField(verbose_name='交易日期', db_index=True)
    # [代码修改开始]
    # --- 第一维度: 能量与战场动力学 (Energy & Battlefield Dynamics) ---
    ENERGY_DENSITY_METRICS = {
        'intraday_energy_density': '日内能量密度',
        'intraday_thrust_purity': '日内推力纯度',
        'volume_burstiness_index': '成交量爆裂度指数',
        'auction_impact_score': '集合竞价冲击分',
        # --- 新增战场动力学指标 ---
        'rebound_momentum': '反转动能',
        'high_level_consolidation_volume': '高位整固成交量占比',
        'opening_period_thrust': '开盘期推力',
    }
    # [代码修改结束]
    # --- 第二维度: 控制权 (Control) ---
    CONTROL_METRICS = {
        'vwap_deviation_area': 'VWAP偏离面积',
        'trend_persistence_index': '趋势持续性指数(Hurst)',
        'volume_weighted_time_index': '成交量加权时间指数',
        'close_vs_vpoc_ratio': '收盘价/VPOC比',
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
    UNIFIED_PERIODS = [1, 5, 13, 21, 55]
    BOOLEAN_FIELDS = ['is_intraday_bullish_divergence', 'is_intraday_bearish_divergence']
    # [代码修改开始]
    SLOPE_ACCEL_EXCLUSIONS = [
        'is_intraday_bullish_divergence', 'is_intraday_bearish_divergence',
        'auction_impact_score',
        'trend_persistence_index',
        # --- 新增排除项 ---
        'rebound_momentum',
        'high_level_consolidation_volume',
        'opening_period_thrust',
    ]
    # [代码修改结束]
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










