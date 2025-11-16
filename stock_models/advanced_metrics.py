# stock_models\advanced_metrics.py

from django.db import models
from django.utils.translation import gettext_lazy as _
import pandas as pd

# 筹码高级指标模型
class BaseAdvancedChipMetrics(models.Model):
    """
    【V32.2 · 逐笔数据衍生指标版】
    - 核心新增: 引入 `price_volume_entropy` 字段，用于存储由逐笔数据衍生的价格成交量熵。
    - 核心重构: 基于15:00分钟线，重构 `auction_intent_signal`，并用 `auction_closing_position` 替代原 `auction_pressure_ratio`。
    - 核心优化: 更新模型定义与衍生计算排除列表，以匹配新的竞价信号体系。
    """
    trade_time = models.DateField(verbose_name='交易日期', db_index=True)
    # --- 第一象限: 静态结构 (Static Structure) ---
    STATIC_STRUCTURE_METRICS = {
        'dominant_peak_cost': '主导峰成本',
        'dominant_peak_volume_ratio': '主导峰筹码占比(%)',
        'dominant_peak_profit_margin': '主导峰利润边际(%)',
        'dominant_peak_solidity': '主峰稳固度',
        'secondary_peak_cost': '次筹码峰成本',
        'peak_separation_ratio': '峰群分离度(%)',
        'peak_separation_intensity': '峰间分离强度',
        'peak_fusion_indicator': '峰间融合指标',
        'peak_volume_ratio': '峰群量能比(%)',
        'peak_distance_volatility_ratio': '波动率标准化峰距',
        'winner_concentration_90pct': '获利盘集中度',
        'loser_concentration_90pct': '套牢盘集中度',
        'long_term_concentration_90pct': '长期筹码集中度',
        'short_term_concentration_90pct': '短期筹码集中度',
        'dynamic_pressure_index': '动态压力指数',
        'dynamic_support_index': '动态支撑指数',
        'chip_fault_magnitude': '筹码断层量级(ATR)',
        'chip_fault_blockage_ratio': '断层阻碍度(%)',
        'total_winner_rate': '存量总获利盘(%)',
        'total_loser_rate': '存量总套牢盘(%)',
        'effective_winner_rate': '有效获利盘比例(%)',
        'winner_profit_margin_avg': '平均获利盘利润率(%)',
        'loser_loss_margin_avg': '平均套牢盘亏损率(%)',
        'active_winner_rate': '活跃获利盘比例(%)',
        'active_loser_rate': '活跃套牢盘比例(%)',
        'locked_profit_rate': '锁定利润盘比例(%)',
        'locked_loss_rate': '锁定亏损盘比例(%)',
        'winner_profit_cushion': '获利盘缓冲垫(%)',
        'loser_pain_index': '套牢盘痛苦指数',
        'structural_stability_score': '结构稳定性评分(0-100)',
        'cost_structure_skewness': '成本结构偏度',
        'recent_trapped_pressure': '近期套牢盘压力(%)',
        'imminent_profit_taking_supply': '潜在获利盘供给(%)',
        'price_volume_entropy': '价格成交量熵', # 新增代码行: 价格成交量熵
    }
    # --- 第二象限: 内部动态 (Intraday Dynamics) ---
    INTRADAY_DYNAMICS_METRICS = {
        'active_selling_pressure': '主动卖压强度(%)',
        'active_buying_support': '主动买盘支撑(%)',
        'peak_battle_intensity': '主峰交战强度(%)',
        'peak_dynamic_strength_ratio': '峰区压力平衡(%)',
        'peak_main_force_premium': '主峰主力溢价(%)',
        'peak_mf_conviction_flow': '主峰主力信念流(%)',
        'upward_impulse_purity': '上涨脉冲纯度(%)',
        'opening_gap_defense_strength': '开盘缺口防御强度',
        'active_zone_combat_intensity': '活跃战区交战强度(%)',
        'active_zone_mf_stance': '活跃战区主力姿态(%)',
        'profit_realization_quality': '获利盘兑现质量(%)',
        'capitulation_absorption_index': '投降承接指数(%)'
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
        'peak_shoulder_growth_rate': '筹码峰肩增长率(%)',
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
        'intraday_new_loser_pressure': '日内新增套牢盘压力',
        'auction_intent_signal': '竞价意图信号',
        'auction_closing_position': '竞价收盘位置(-100~100)',
        'intraday_probe_rebound_quality': '日内试探回升质量',
    }
    # --- 第五象限: 生命体征 (Vital Signs) ---
    VITAL_SIGNS_METRICS = {
        'structural_consensus_score': '结构共识分',
        'dominant_cost_momentum': '主导成本动量',
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
    INTEGER_FIELDS = ['peak_volume', 'pressure_above_volume', 'support_below_volume']
    BOOLEAN_FIELDS = []
    SLOPE_ACCEL_EXCLUSIONS = [
        'gathering_by_support', 'gathering_by_chasing',
        'dispersal_by_distribution', 'dispersal_by_capitulation',
        'profit_taking_flow_ratio', 'capitulation_flow_ratio',
        'active_winner_pressure_ratio', 'locked_profit_pressure_ratio',
        'active_loser_pressure_ratio', 'locked_loss_pressure_ratio',
        'suppressive_accumulation_intensity',
        'rally_distribution_intensity',
        'rally_accumulation_intensity',
        'panic_selling_intensity',
        'main_force_cost_advantage',
        'main_force_control_leverage',
        'fault_traversal_momentum', 'intraday_trend_efficiency',
        'intraday_new_loser_pressure',
        'auction_intent_signal',
        'auction_closing_position',
        'intraday_probe_rebound_quality',
        'structural_consensus_score',
        'dominant_cost_momentum',
        'capitulation_absorption_index',
        'structural_resilience_index',
        'posture_control_score',
        'posture_action_score',
        'loser_pain_index',
        'structural_stability_score',
        'cost_structure_skewness',
        'recent_trapped_pressure',
        'imminent_profit_taking_supply',
        'peak_shoulder_growth_rate',
        'dominant_peak_solidity',
        'peak_distance_volatility_ratio',
        'peak_dynamic_strength_ratio',
        'peak_separation_intensity',
        'peak_fusion_indicator',
        'price_volume_entropy', # 新增代码行: 价格成交量熵
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

class AdvancedChipMetrics_SZ(BaseAdvancedChipMetrics):
    stock = models.ForeignKey(
        'StockInfo',
        on_delete=models.CASCADE,
        related_name='advanced_chip_metrics_sz',
        verbose_name='股票',
        db_index=True
    )
    class Meta(BaseAdvancedChipMetrics.Meta):
        abstract = False
        verbose_name = '高级筹码指标-深圳(V6.0-衍生固化)'
        verbose_name_plural = verbose_name
        db_table = 'stock_advanced_chip_metrics_sz'
        unique_together = ('stock', 'trade_time')
        indexes = [
            models.Index(fields=['stock', 'trade_time']),
            models.Index(fields=['chip_health_score']),
            models.Index(fields=['structural_resilience_index']),
            models.Index(fields=['peak_separation_ratio']),
            models.Index(fields=['dominant_peak_solidity']),
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
            models.Index(fields=['structural_resilience_index']),
            models.Index(fields=['peak_separation_ratio']),
            models.Index(fields=['dominant_peak_solidity']),
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
            models.Index(fields=['structural_resilience_index']),
            models.Index(fields=['peak_separation_ratio']),
            models.Index(fields=['dominant_peak_solidity']),
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
            models.Index(fields=['structural_resilience_index']),
            models.Index(fields=['peak_separation_ratio']),
            models.Index(fields=['dominant_peak_solidity']),
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
        verbose_name = '高级筹码指标-北京(V6.0-衍生固化)'
        verbose_name_plural = verbose_name
        db_table = 'stock_advanced_chip_metrics_bj'
        unique_together = ('stock', 'trade_time')
        indexes = [
            models.Index(fields=['stock', 'trade_time']),
            models.Index(fields=['chip_health_score']),
            models.Index(fields=['structural_resilience_index']),
            models.Index(fields=['peak_separation_ratio']),
            models.Index(fields=['dominant_peak_solidity']),
        ]

# 资金高级指标模型
class BaseAdvancedFundFlowMetrics(models.Model):
    """
    【V33.1 · 逐笔数据衍生指标版】
    - 核心新增: 引入 `main_force_ofi`, `retail_ofi`, `microstructure_efficiency_index`, `hidden_accumulation_intensity` 字段。
    - 核心新增: 实现了 flow_temperature_premium, main_force_on_peak_flow 的计算逻辑。
    - 核心优化: 调整模型字段定义，以承载新的指标。
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
        'main_force_conviction_index': '主力信念指数',
        'inferred_active_order_size': '推断活跃订单规模(元)',
        'retail_flow_dominance_index': '散户流动性主导指数',
        'main_force_price_impact_ratio': '主力价格冲击比率',
    }
    TACTICAL_LOG_METRICS = {
        'dip_absorption_power': '逢低吸筹力度',
        'rally_distribution_pressure': '拉高派发压力(%)',
        'panic_selling_cascade': '恐慌抛售级联(%)',
        'opening_battle_result': '开盘战役结果',
        'pre_closing_posturing': '收盘前姿态',
        'closing_auction_ambush': '收盘伏击强度',
        'main_force_execution_alpha': '主力执行Alpha(%)',
        'retail_panic_surrender_index': '散户恐慌投降指数(%)',
        'retail_fomo_premium_index': '散户追高溢价指数(%)',
        'main_force_t0_efficiency': '主力T+0效率(%)',
        'vwap_structure_skew': 'VWAP结构偏离度',
        'flow_efficiency_index': '资金效率指数',
        'asymmetric_volume_thrust': '非对称成交量推力',
        'vwap_control_strength': 'VWAP控制强度',
        'main_force_vwap_guidance': '主力VWAP引导力',
        'vwap_crossing_intensity': 'VWAP穿越烈度',
        'upper_shadow_selling_pressure': '上影线抛压强度',
        'lower_shadow_absorption_strength': '下影线承接强度',
        'trend_conviction_ratio': '趋势信念比',
        'reversal_power_index': '反转力量指数',
        'holistic_cmf': '全局CMF',
        'main_force_cmf': '主力CMF',
        'cmf_divergence_score': 'CMF背离得分',
        'main_force_vpoc': '主力VPOC',
        'mf_vpoc_premium': '主力VPOC溢价(%)',
        'main_force_on_peak_flow': '主力在主峰区的净流入(万元)',
        'flow_temperature_premium': '资金温度溢价(%)',
        'mf_retail_liquidity_swap_corr': '主力散户流动性交换相关性',
        'main_force_ofi': '主力订单流失衡', # 新增代码行: 主力订单流失衡
        'retail_ofi': '散户订单流失衡', # 新增代码行: 散户订单流失衡
        'microstructure_efficiency_index': '微观结构效率指数', # 新增代码行: 微观结构效率指数
        'hidden_accumulation_intensity': '隐蔽吸筹强度', # 新增代码行: 隐蔽吸筹强度
    }
    OUTCOME_ASSESSMENT_METRICS = {
        'volatility_asymmetry_index': '波动不对称指数',
        'closing_price_deviation_score': '收盘价偏离度得分',
    }
    CORE_METRICS = {
        **POWER_STRUCTURE_METRICS,
        **TACTICAL_LOG_METRICS,
        **OUTCOME_ASSESSMENT_METRICS,
    }
    SLOPE_ACCEL_EXCLUSIONS = [
        'flow_credibility_index', 'mf_retail_battle_intensity', 'main_force_activity_ratio',
        'main_force_flow_directionality', 'main_force_conviction_index',
        'retail_flow_dominance_index', 'main_force_price_impact_ratio', 'dip_absorption_power',
        'rally_distribution_pressure', 'panic_selling_cascade', 'opening_battle_result',
        'pre_closing_posturing', 'closing_auction_ambush', 'main_force_execution_alpha',
        'retail_panic_surrender_index', 'retail_fomo_premium_index', 'main_force_t0_efficiency',
        'vwap_structure_skew', 'flow_efficiency_index', 'asymmetric_volume_thrust',
        'volatility_asymmetry_index', 'closing_price_deviation_score',
        'vwap_control_strength', 'main_force_vwap_guidance', 'vwap_crossing_intensity',
        'upper_shadow_selling_pressure', 'lower_shadow_absorption_strength',
        'trend_conviction_ratio', 'reversal_power_index',
        'holistic_cmf', 'main_force_cmf', 'cmf_divergence_score',
        'main_force_vpoc', 'mf_vpoc_premium',
        'flow_temperature_premium', 'mf_retail_liquidity_swap_corr',
        'main_force_ofi', # 新增代码行: 主力订单流失衡
        'retail_ofi', # 新增代码行: 散户订单流失衡
        'microstructure_efficiency_index', # 新增代码行: 微观结构效率指数
        'hidden_accumulation_intensity', # 新增代码行: 隐蔽吸筹强度
    ]
    FLOAT_METRICS = [
        'flow_credibility_index', 'mf_retail_battle_intensity', 'main_force_activity_ratio',
        'main_force_flow_directionality', 'main_force_conviction_index',
        'inferred_active_order_size', 'retail_flow_dominance_index', 'main_force_price_impact_ratio',
        'dip_absorption_power', 'rally_distribution_pressure', 'panic_selling_cascade',
        'opening_battle_result', 'pre_closing_posturing', 'closing_auction_ambush',
        'main_force_execution_alpha', 'retail_panic_surrender_index',
        'retail_fomo_premium_index', 'main_force_t0_efficiency',
        'vwap_structure_skew', 'flow_efficiency_index', 'asymmetric_volume_thrust',
        'volatility_asymmetry_index', 'closing_price_deviation_score',
        'vwap_control_strength', 'main_force_vwap_guidance', 'vwap_crossing_intensity',
        'upper_shadow_selling_pressure', 'lower_shadow_absorption_strength',
        'trend_conviction_ratio', 'reversal_power_index',
        'holistic_cmf', 'main_force_cmf', 'cmf_divergence_score',
        'mf_vpoc_premium',
        'flow_temperature_premium', 'mf_retail_liquidity_swap_corr',
        'main_force_ofi', # 新增代码行: 主力订单流失衡
        'retail_ofi', # 新增代码行: 散户订单流失衡
        'microstructure_efficiency_index', # 新增代码行: 微观结构效率指数
        'hidden_accumulation_intensity', # 新增代码行: 隐蔽吸筹强度
    ]
    for name, verbose in CORE_METRICS.items():
        if name in FLOAT_METRICS:
            vars()[name] = models.FloatField(verbose_name=verbose, null=True, blank=True)
        else:
            vars()[name] = models.DecimalField(max_digits=22, decimal_places=6, verbose_name=verbose, null=True, blank=True)
    main_force_buy_rate_consensus = models.DecimalField(max_digits=10, decimal_places=6, verbose_name='共识-主力买入率(%)', null=True, blank=True)
    UNIFIED_PERIODS = [1, 5, 13, 21, 55]
    sum_cols = [
        'net_flow_calibrated', 'main_force_net_flow_calibrated', 'retail_net_flow_calibrated',
        'net_xl_amount_calibrated', 'net_lg_amount_calibrated', 'net_md_amount_calibrated',
        'net_sh_amount_calibrated',
        'main_force_on_peak_flow',
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
            models.Index(fields=['main_force_price_impact_ratio']),
            models.Index(fields=['mf_retail_battle_intensity']),
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
            models.Index(fields=['main_force_price_impact_ratio']),
            models.Index(fields=['mf_retail_battle_intensity']),
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
            models.Index(fields=['main_force_price_impact_ratio']),
            models.Index(fields=['mf_retail_battle_intensity']),
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
            models.Index(fields=['main_force_price_impact_ratio']),
            models.Index(fields=['mf_retail_battle_intensity']),
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
            models.Index(fields=['main_force_price_impact_ratio']),
            models.Index(fields=['mf_retail_battle_intensity']),
        ]

# 结构与行为高级指标模型
class BaseAdvancedStructuralMetrics(models.Model):
    """
    【V18.0 · 战场分析仪实装版】
    - 核心新增: 引入 trend_quality_score, closing_momentum_index, volume_structure_skew 三大高级结构指标，深度剖析日内走势质量。
    - 核心优化: 更新指标分组与衍生计算排除列表，以集成新指标。
    """
    trade_time = models.DateField(verbose_name='交易日期', db_index=True)
    ENERGY_DENSITY_METRICS = {
        'intraday_energy_density': '日内能量密度',
        'intraday_thrust_purity': '日内推力纯度',
        'volume_burstiness_index': '成交量爆裂度指数',
        'auction_impact_score': '开盘缺口强度',
        'rebound_momentum': '反转动能',
        'high_level_consolidation_volume': '高位整固成交量占比',
        'opening_period_thrust': '开盘期推力',
    }
    CONTROL_METRICS = {
        'trend_efficiency_ratio': '趋势效率比',
        'pullback_depth_ratio': '回撤深度比',
        'mean_reversion_frequency': '均值回归频率(每小时)',
        'opening_volume_impulse': '开盘成交脉冲',
        'midday_consolidation_level': '盘中沉寂水平',
        'tail_volume_acceleration': '尾盘成交加速',
        'vpoc_deviation_magnitude': 'VPOC偏离量级(ATR)',
        'vpoc_consensus_strength': 'VPOC共识强度(%)',
        'closing_conviction_score': '收盘信念得分',
        'volume_profile_entropy': '成交剖面熵',
        'intraday_pnl_imbalance': '日内盈亏失衡度',
        'cost_dispersion_index': '成本离散指数(ATR)',
    }
    GAME_EFFICIENCY_METRICS = {
        'upward_thrust_efficacy': '上涨推力效能',
        'downward_absorption_efficacy': '下跌吸收效能',
        'net_vpa_score': '净量价效能得分',
        'divergence_conviction_score': '背离信念得分',
        'volatility_skew_index': '波动率偏度指数',
    }
    FORWARD_LOOKING_METRICS = {
        'volatility_expansion_ratio': '波动率扩张比',
        'price_shock_factor': '价格冲击因子(ATR标准化)',
        'auction_showdown_score': '收盘竞价摊牌分',
    }
    ADVANCED_BATTLEFIELD_METRICS = {
        'trend_quality_score': '趋势质量分',
        'closing_momentum_index': '收盘动能指数',
        'volume_structure_skew': '成交结构偏度',
    }
    AUXILIARY_METRICS = {
        'value_area_migration': '价值区迁移度(ATR)',
        'value_area_overlap_pct': '价值区重叠度(%)',
        'closing_acceptance_type': '收盘接受度类型',
    }
    CORE_METRICS = {
        **ENERGY_DENSITY_METRICS,
        **CONTROL_METRICS,
        **GAME_EFFICIENCY_METRICS,
        **FORWARD_LOOKING_METRICS,
        **ADVANCED_BATTLEFIELD_METRICS,
        **AUXILIARY_METRICS,
    }
    UNIFIED_PERIODS = [1, 5, 13, 21, 55]
    BOOLEAN_FIELDS = []
    SLOPE_ACCEL_EXCLUSIONS = [
        'auction_impact_score',
        'rebound_momentum',
        'high_level_consolidation_volume',
        'opening_period_thrust',
        'trend_efficiency_ratio',
        'pullback_depth_ratio',
        'mean_reversion_frequency',
        'opening_volume_impulse',
        'midday_consolidation_level',
        'tail_volume_acceleration',
        'vpoc_deviation_magnitude',
        'vpoc_consensus_strength',
        'closing_conviction_score',
        'volume_profile_entropy',
        'intraday_pnl_imbalance',
        'cost_dispersion_index',
        'upward_thrust_efficacy',
        'downward_absorption_efficacy',
        'net_vpa_score',
        'divergence_conviction_score',
        'volatility_skew_index',
        'value_area_migration',
        'value_area_overlap_pct',
        'closing_acceptance_type',
        'volatility_expansion_ratio',
        'price_shock_factor',
        'auction_showdown_score',
        'trend_quality_score',
        'closing_momentum_index',
        'volume_structure_skew',
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
            models.Index(fields=['intraday_energy_density']),
            models.Index(fields=['divergence_conviction_score']),
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
            models.Index(fields=['intraday_energy_density']),
            models.Index(fields=['divergence_conviction_score']),
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
            models.Index(fields=['intraday_energy_density']),
            models.Index(fields=['divergence_conviction_score']),
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
            models.Index(fields=['intraday_energy_density']),
            models.Index(fields=['divergence_conviction_score']),
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
            models.Index(fields=['intraday_energy_density']),
            models.Index(fields=['divergence_conviction_score']),
        ]
