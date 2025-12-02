# stock_models\advanced_metrics.py

from django.db import models
from django.utils.translation import gettext_lazy as _
import pandas as pd

# 筹码高级指标模型
class BaseAdvancedChipMetrics(models.Model):
    """
    【V33.1 · 情境融合版】
    - 核心升维: 从资金流模型中移入“情境行为融合”指标体系，将资金“行为”（如派发、吸筹）与筹码“情境”（如主峰位置）
                 在筹码服务层直接融合，解决了因计算时序依赖导致的数据缺失问题。
    - 核心新增: 新增 `distribution_at_peak_intensity`, `absorption_at_peak_intensity`, `breakthrough_of_peak_quality`, `defense_of_peak_quality` 四大核心情境行为指标。
    """
    trade_time = models.DateField(verbose_name='交易日期', db_index=True)
    # --- 第一象限: 静态结构 (Static Structure) ---
    STATIC_STRUCTURE_METRICS = {
        'structural_node_count': '结构节点数量',
        'primary_peak_kurtosis': '主峰峰态系数',
        'cost_gini_coefficient': '成本基尼系数',
        'structural_tension_index': '结构张力指数',
        'structural_leverage': '结构杠杆',
        'vacuum_zone_magnitude': '真空区量级(ATR)',
        'winner_stability_index': '获利盘稳定度',
        'dominant_peak_cost': '主导峰成本',
        'dominant_peak_volume_ratio': '主导峰筹码占比(%)',
        'dominant_peak_profit_margin': '主导峰利润边际(%)',
        'dominant_peak_solidity': '主峰稳固度',
        'secondary_peak_cost': '次筹码峰成本',
        'peak_separation_ratio': '峰群分离度(%)',
        'winner_concentration_90pct': '获利盘集中度',
        'loser_concentration_90pct': '套牢盘集中度',
        'chip_fault_magnitude': '筹码断层量级(ATR)',
        'chip_fault_blockage_ratio': '断层阻碍度(%)',
        'total_winner_rate': '存量总获利盘(%)',
        'total_loser_rate': '存量总套牢盘(%)',
        'winner_profit_margin_avg': '平均获利盘利润率(%)',
        'loser_loss_margin_avg': '平均套牢盘亏损率(%)',
        'loser_pain_index': '套牢盘痛苦指数', # 逻辑已升级
        'structural_potential_score': '结构势能分(0-100)',
        'cost_structure_skewness': '成本结构偏度',
        'price_volume_entropy': '价格成交量熵',
    }
    # --- 第二象限: 内部动态 (Intraday Dynamics) ---
    INTRADAY_DYNAMICS_METRICS = {
        # 引入新一代日内动态博弈指标
        'impulse_quality_ratio': '脉冲品质比率(%)',
        'peak_control_transfer': '主峰控制权转移(%)',
        'support_validation_strength': '支撑验证强度',
        'pressure_rejection_strength': '压力拒绝强度',
        'vacuum_traversal_efficiency': '真空区通行效率',
        'intraday_posture_score': '日内姿态评分(-100~100)',
        'floating_chip_cleansing_efficiency': '浮筹清洗效率',
        'active_selling_pressure': '主动卖压强度(%)',
        'active_buying_support': '主动买盘支撑(%)',
        'upward_impulse_purity': '上涨脉冲纯度(%)', # 旧有逻辑，可作为参考
        'opening_gap_defense_strength': '开盘缺口防御强度',
        'profit_realization_quality': '获利盘兑现质量(%)',
        'capitulation_absorption_index': '投降承接指数(%)'
    }
    # --- 第三象限: 跨日迁徙 (Cross-Day Flow) ---
    CROSS_DAY_FLOW_METRICS = {
        # 引入新一代信念交割与结构演化指标
        'peak_mass_transfer_rate': '主峰质量转移率',
        'conviction_flow_index': '信念流转指数',
        'constructive_turnover_ratio': '建设性换手率',
        'structural_entropy_change': '结构熵变',
        'main_force_flow_gini': '主力资金流基尼系数',
        # --- 保留和兼容的指标 ---
        'gathering_by_support': '承接式集结量(%)',
        'gathering_by_chasing': '追涨式集结量(%)',
        'dispersal_by_distribution': '派发式分散量(%)',
        'dispersal_by_capitulation': '承接式分散量(%)',
        'profit_taking_flow_ratio': '获利兑现流量占比(%)',
        'capitulation_flow_ratio': '恐慌抛售流量占比(%)',
        'winner_loser_momentum': '盈亏动量',
        'chip_fatigue_index': '筹码疲劳指数',
    }
    # --- 第四象限: 博弈意图 (Game-Theoretic Intent) ---
    GAME_THEORY_METRICS = {
        # 引入新一代战略推演指标
        'strategic_phase_score': '战略阶段评分(-100~100)',
        'deception_index': '欺骗指数(-100~100)',
        'control_solidity_index': '控制力稳固度',
        'exhaustion_risk_index': '衰竭风险指数',
        'breakout_readiness_score': '突破就绪分(0-100)',
        'mf_cost_zone_defense_intent': '主力成本区攻防意图(-100~100)',
        'main_force_cost_advantage': '主力成本优势(%)',
        'chip_health_score': '筹码健康分(0-100)', # 逻辑会依赖新指标
        'auction_intent_signal': '竞价意图信号',
        'auction_closing_position': '竞价收盘位置(-100~100)',
        'peak_exchange_purity': '主峰交换纯度',
        'pressure_validation_score': '压力验证分',
        'support_validation_score': '支撑验证分',
        'covert_accumulation_signal': '隐蔽吸筹信号',
        'suppressive_accumulation_intensity': '打压吸筹强度',
    }
    # --- 第五象限: 生命体征 (Vital Signs) ---
    VITAL_SIGNS_METRICS = {
        'signal_conviction_score': '信号置信度评分(-100~100)',
        'risk_reward_profile': '风险收益剖面',
        'trend_vitality_index': '趋势生命力指数',
        'overall_t1_rating': 'T+1综合评级(-100~100)',
    }
    # [新增代码块] 新增情境行为融合指标
    CONTEXTUAL_ACTION_METRICS = {
        'distribution_at_peak_intensity': '主峰区派发烈度',
        'absorption_at_peak_intensity': '主峰区吸筹烈度',
        'breakthrough_of_peak_quality': '突破主峰质量',
        'defense_of_peak_quality': '防守主峰质量',
    }
    CORE_METRICS = {
        **STATIC_STRUCTURE_METRICS,
        **INTRADAY_DYNAMICS_METRICS,
        **CROSS_DAY_FLOW_METRICS,
        **GAME_THEORY_METRICS,
        **VITAL_SIGNS_METRICS,
        **CONTEXTUAL_ACTION_METRICS, # [修改的代码行] 整合新指标
    }
    UNIFIED_PERIODS = [1, 5, 13, 21, 55]
    INTEGER_FIELDS = ['peak_volume', 'pressure_above_volume', 'support_below_volume']
    BOOLEAN_FIELDS = []
    SLOPE_ACCEL_EXCLUSIONS = [
        'gathering_by_support', 'gathering_by_chasing',
        'dispersal_by_distribution', 'dispersal_by_capitulation',
        'profit_taking_flow_ratio', 'capitulation_flow_ratio',
        'main_force_cost_advantage',
        'auction_intent_signal',
        'auction_closing_position',
        'loser_pain_index',
        'structural_potential_score',
        'cost_structure_skewness',
        'price_volume_entropy',
        'peak_exchange_purity',
        'pressure_validation_score',
        'support_validation_score',
        'covert_accumulation_signal',
        'structural_node_count',
        'primary_peak_kurtosis',
        'cost_gini_coefficient',
        'structural_tension_index',
        'structural_leverage',
        'vacuum_zone_magnitude',
        'winner_stability_index',
        'intraday_posture_score',
        'peak_mass_transfer_rate',
        'conviction_flow_index',
        'constructive_turnover_ratio',
        'structural_entropy_change',
        'main_force_flow_gini',
        'strategic_phase_score',
        'deception_index',
        'control_solidity_index',
        'exhaustion_risk_index',
        'breakout_readiness_score',
        'signal_conviction_score',
        'risk_reward_profile',
        'trend_vitality_index',
        'overall_t1_rating',
        'mf_cost_zone_defense_intent',
        'floating_chip_cleansing_efficiency',
        'suppressive_accumulation_intensity',
        # [新增代码块] 将新指标添加到排除列表
        'distribution_at_peak_intensity',
        'absorption_at_peak_intensity',
        'breakthrough_of_peak_quality',
        'defense_of_peak_quality',
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
            models.Index(fields=['peak_separation_ratio']),
            models.Index(fields=['dominant_peak_solidity']),
        ]

# 资金高级指标模型
class BaseAdvancedFundFlowMetrics(models.Model):
    """
    【V61.1 · 职责净化版】
    - 核心重构: 移除了与筹码情境强耦合的“情境行为融合”指标体系，将其职责完全转移至筹码指标模型，
                 解决了因计算时序依赖导致的数据缺失问题，使本模型职责更聚焦于纯粹的资金流分析。
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
        'main_force_activity_ratio': '主力参与度(%)',
        'main_force_flow_directionality': '主力资金流向性(%)',
        'main_force_conviction_index': '主力信念指数',
        'main_force_posture_index': '主力姿态指数',
        'observed_large_order_size_avg': '观测大单平均规模(元)',
        'retail_flow_dominance_index': '散户流动性主导指数',
        'main_force_slippage_index': '主力滑点指数(%)',
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
        'micro_price_impact_asymmetry': '微观价格冲击不对称性',
        'order_book_clearing_rate': '盘口清扫率(%)',
        'vwap_control_strength': 'VWAP控制强度',
        'main_force_vwap_guidance': '主力VWAP引导力',
        'vwap_crossing_intensity': 'VWAP穿越烈度',
        'upper_shadow_selling_pressure': '上影线抛压强度',
        'lower_shadow_absorption_strength': '下影线承接强度',
        'trend_alignment_index': '趋势同向指数',
        'reversal_power_index': '反转力量指数',
        'holistic_cmf': '全局CMF',
        'main_force_cmf': '主力CMF',
        'cmf_divergence_score': 'CMF背离得分',
        'main_force_vpoc': '主力VPOC',
        'mf_vpoc_premium': '主力VPOC溢价(%)',
        'main_force_on_peak_flow': '主力在主峰区的净流入(万元)',
        'flow_temperature_premium': '资金温度溢价(%)',
        'mf_retail_liquidity_swap_corr': '主力散户流动性交换相关性',
        'main_force_ofi': '主力订单流失衡',
        'retail_ofi': '散户订单流失衡',
        'microstructure_efficiency_index': '微观结构效率指数',
        'hidden_accumulation_intensity': '隐蔽吸筹强度',
        'wash_trade_intensity': '主力对倒强度',
        'order_book_imbalance': '五档盘口失衡度',
        'large_order_pressure': '大单压制强度',
        'large_order_support': '大单支撑强度',
        'order_book_liquidity_supply': '盘口流动性供给(买/卖比)',
        'buy_quote_exhaustion_rate': '买方报价消耗率(%)',
        'sell_quote_exhaustion_rate': '卖方报价消耗率(%)',
        'imbalance_effectiveness': '盘口失衡有效性',
    }
    OUTCOME_ASSESSMENT_METRICS = {
        'volatility_asymmetry_index': '波动不对称指数',
        'closing_strength_index': '收盘强度指数',
    }
    CORE_METRICS = {
        **POWER_STRUCTURE_METRICS,
        **TACTICAL_LOG_METRICS,
        **OUTCOME_ASSESSMENT_METRICS,
        # [修改的代码行] 移除 CONTEXTUAL_ACTION_METRICS
    }
    SLOPE_ACCEL_EXCLUSIONS = [
        'flow_credibility_index', 'mf_retail_battle_intensity', 'main_force_activity_ratio',
        'main_force_flow_directionality', 'main_force_conviction_index',
        'retail_flow_dominance_index', 'main_force_slippage_index', 'dip_absorption_power',
        'rally_distribution_pressure', 'panic_selling_cascade', 'opening_battle_result',
        'pre_closing_posturing', 'closing_auction_ambush', 'main_force_execution_alpha',
        'retail_panic_surrender_index', 'retail_fomo_premium_index', 'main_force_t0_efficiency',
        'vwap_structure_skew', 'flow_efficiency_index',
        'volatility_asymmetry_index', 'closing_strength_index',
        'vwap_control_strength', 'main_force_vwap_guidance', 'vwap_crossing_intensity',
        'upper_shadow_selling_pressure', 'lower_shadow_absorption_strength',
        'trend_alignment_index', 'reversal_power_index',
        'holistic_cmf', 'main_force_cmf', 'cmf_divergence_score',
        'main_force_vpoc', 'mf_vpoc_premium',
        'flow_temperature_premium', 'mf_retail_liquidity_swap_corr',
        'main_force_ofi', 'retail_ofi', 'microstructure_efficiency_index',
        'hidden_accumulation_intensity', 'wash_trade_intensity', 'order_book_imbalance',
        'large_order_pressure', 'large_order_support', 'order_book_liquidity_supply',
        'buy_quote_exhaustion_rate', 'sell_quote_exhaustion_rate',
        'observed_large_order_size_avg', 'micro_price_impact_asymmetry', 'order_book_clearing_rate',
        'imbalance_effectiveness',
        'main_force_posture_index',
        # [修改的代码块] 移除相关指标
    ]
    FLOAT_METRICS = [
        'flow_credibility_index', 'mf_retail_battle_intensity', 'main_force_activity_ratio',
        'main_force_flow_directionality', 'main_force_conviction_index',
        'retail_flow_dominance_index', 'main_force_slippage_index',
        'dip_absorption_power', 'rally_distribution_pressure', 'panic_selling_cascade',
        'opening_battle_result', 'pre_closing_posturing', 'closing_auction_ambush',
        'main_force_execution_alpha', 'retail_panic_surrender_index',
        'retail_fomo_premium_index', 'main_force_t0_efficiency',
        'vwap_structure_skew', 'flow_efficiency_index',
        'volatility_asymmetry_index', 'closing_strength_index',
        'vwap_control_strength', 'main_force_vwap_guidance', 'vwap_crossing_intensity',
        'upper_shadow_selling_pressure', 'lower_shadow_absorption_strength',
        'trend_alignment_index', 'reversal_power_index',
        'holistic_cmf', 'main_force_cmf', 'cmf_divergence_score',
        'mf_vpoc_premium', 'flow_temperature_premium', 'mf_retail_liquidity_swap_corr',
        'main_force_ofi', 'retail_ofi', 'microstructure_efficiency_index',
        'hidden_accumulation_intensity', 'wash_trade_intensity', 'order_book_imbalance',
        'large_order_pressure', 'large_order_support', 'order_book_liquidity_supply',
        'buy_quote_exhaustion_rate', 'sell_quote_exhaustion_rate',
        'observed_large_order_size_avg', 'micro_price_impact_asymmetry', 'order_book_clearing_rate',
        'imbalance_effectiveness',
        'main_force_posture_index',
        # [修改的代码块] 移除相关指标
    ]
    for name, verbose in CORE_METRICS.items():
        if name in FLOAT_METRICS:
            vars()[name] = models.FloatField(verbose_name=verbose, null=True, blank=True)
        else:
            vars()[name] = models.DecimalField(max_digits=22, decimal_places=6, verbose_name=verbose, null=True, blank=True)
    class Meta:
        abstract = True
        ordering = ['-trade_time']
        indexes = [
            models.Index(fields=['main_force_conviction_index']),
            models.Index(fields=['main_force_slippage_index']),
            models.Index(fields=['dip_absorption_power']),
            models.Index(fields=['rally_distribution_pressure']),
        ]

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
            models.Index(fields=['main_force_slippage_index']),
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
            models.Index(fields=['main_force_slippage_index']),
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
            models.Index(fields=['main_force_slippage_index']),
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
            models.Index(fields=['main_force_slippage_index']),
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
            models.Index(fields=['main_force_slippage_index']),
            models.Index(fields=['mf_retail_battle_intensity']),
        ]

# 结构与行为高级指标模型
class BaseAdvancedStructuralMetrics(models.Model):
    """
    【V69.0 · 决战验刃】
    - 核心升维: 将 `trend_quality_score` 升维为 `trend_acceleration_score`，通过比较上下半场
                 趋势斜率，精准捕捉战局的“加速”与“转折”。
    - 核心升维: 将 `closing_momentum_index` 升维为 `final_charge_intensity`，通过对比决战
                 时刻与战前对峙的“推力纯度”，深度量化终场冲锋的决心与烈度。
    """
    trade_time = models.DateField(verbose_name='交易日期', db_index=True)
    ENERGY_DENSITY_METRICS = {
        'intraday_energy_density': '日内能量密度',
        'intraday_thrust_purity': '日内推力纯度',
        'volume_burstiness_index': '成交量爆裂度指数',
        'auction_impact_score': '开盘缺口强度',
        'dynamic_reversal_strength': '动态反转强度',
        'reversal_conviction_rate': '反转信念比率',
        'reversal_recovery_rate': '反转收复率',
        'high_level_consolidation_volume': '高位整固成交量占比',
        'opening_period_thrust': '开盘期推力',
    }
    DYNAMIC_EVOLUTION_FACTORS = {
        'thrust_purity_ma5': '推力纯度5日均线',
        'absorption_strength_ma5': '吸筹强度5日均线',
        'sweep_intensity_ma5': '扫单强度5日均线',
        'vpin_roc3': 'VPIN3日变化率',
    }
    CONTROL_METRICS = {
        'trend_efficiency_ratio': '趋势效率比',
        'pullback_depth_ratio': '回撤深度比',
        'mean_reversion_frequency': '均值回归频率(每小时)',
        'opening_impulse_efficiency': '开盘脉冲效率',
        'midday_narrow_range_gravity': '盘中窄幅引力',
        'tail_acceleration_efficiency': '尾盘加速效率',
        'closing_conviction_score': '收盘信念得分',
        'volume_profile_entropy': '成交剖面熵',
        'intraday_pnl_imbalance': '日内盈亏失衡度',
        'cost_dispersion_index': '成本离散指数(ATR)',
    }
    GAME_EFFICIENCY_METRICS = {
        'trend_asymmetry_index': '趋势不对称指数',
        'active_volume_price_efficiency': '主动成交量价格效率',
        'breakthrough_cost_index': '突破成本指数',
        'defense_cost_index': '防御成本指数',
        'thrust_efficiency_score': '推力效能分',
    }
    DERIVATIVE_METRICS = {
        'price_thrust_divergence': '价格动能背离',
    }
    FORWARD_LOOKING_METRICS = {
        'volatility_expansion_ratio': '波动率扩张比',
        'shock_conviction_score': '冲击置信度',
        'auction_showdown_score': '收盘竞价摊牌分',
    }
    # 修改代码块：重铸 ADVANCED_BATTLEFIELD_METRICS 指标
    ADVANCED_BATTLEFIELD_METRICS = {
        'trend_acceleration_score': '趋势加速分',
        'final_charge_intensity': '终场冲锋强度',
        'volume_structure_skew': '成交结构偏度',
        'breakthrough_conviction_score': '突破信念分',
        'defense_solidity_score': '防守稳固度',
        'equilibrium_compression_index': '均衡压缩指数',
    }
    MICROSTRUCTURE_DYNAMICS_METRICS = {
        'order_flow_imbalance_score': '订单流失衡分数',
        'buy_sweep_intensity': '买方扫单强度',
        'sell_sweep_intensity': '卖方扫单强度',
        'vpin_score': 'VPIN得分',
        'vwap_mean_reversion_corr': 'VWAP均值回归相关性',
        'market_impact_cost': '市场冲击成本(%)',
        'liquidity_slope': '盘口深度斜率',
        'liquidity_authenticity_score': '流动性真实性评分',
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
        **DERIVATIVE_METRICS,
        **FORWARD_LOOKING_METRICS,
        **ADVANCED_BATTLEFIELD_METRICS,
        **MICROSTRUCTURE_DYNAMICS_METRICS,
        **AUXILIARY_METRICS,
        **DYNAMIC_EVOLUTION_FACTORS,
    }
    UNIFIED_PERIODS = [1, 5, 13, 21, 55]
    BOOLEAN_FIELDS = []
    # 修改代码块：同步更新排除列表
    SLOPE_ACCEL_EXCLUSIONS = [
        'auction_impact_score',
        'dynamic_reversal_strength',
        'reversal_conviction_rate',
        'reversal_recovery_rate',
        'high_level_consolidation_volume',
        'opening_period_thrust',
        'trend_efficiency_ratio',
        'pullback_depth_ratio',
        'mean_reversion_frequency',
        'opening_impulse_efficiency',
        'midday_narrow_range_gravity',
        'tail_acceleration_efficiency',
        'closing_conviction_score',
        'volume_profile_entropy',
        'intraday_pnl_imbalance',
        'cost_dispersion_index',
        'price_thrust_divergence',
        'trend_asymmetry_index',
        'value_area_migration',
        'value_area_overlap_pct',
        'closing_acceptance_type',
        'volatility_expansion_ratio',
        'shock_conviction_score',
        'auction_showdown_score',
        'trend_acceleration_score', # 修改代码行
        'final_charge_intensity', # 修改代码行
        'volume_structure_skew',
        'active_volume_price_efficiency',
        'breakthrough_cost_index',
        'defense_cost_index',
        'order_flow_imbalance_score',
        'buy_sweep_intensity',
        'sell_sweep_intensity',
        'vpin_score',
        'vwap_mean_reversion_corr',
        'market_impact_cost',
        'liquidity_slope',
        'liquidity_authenticity_score',
        'thrust_purity_ma5',
        'absorption_strength_ma5',
        'sweep_intensity_ma5',
        'vpin_roc3',
        'breakthrough_conviction_score',
        'defense_solidity_score',
        'equilibrium_compression_index',
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
            models.Index(fields=['intraday_energy_density'])
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
            models.Index(fields=['intraday_energy_density'])
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
            models.Index(fields=['intraday_energy_density'])
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
            models.Index(fields=['intraday_energy_density'])
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
            models.Index(fields=['intraday_energy_density'])
        ]

# 几何形态特征模型 - 平台
class BasePlatformFeature(models.Model):
    """
    【V2.51 · 信念评分版】
    - 核心职责: 持久化存储通过算法识别出的每一个矩形平台的核心量化特征。
    - 设计思想: 每个平台作为一个独立的实体记录，而非每日状态，便于进行结构性回溯和分析。
    - V2.51 新增: 引入 `platform_conviction_score` 字段，用于评估平台本身的结构质量和主力控盘信念。
    """
    # 定义平台性质的选项
    CHARACTER_CHOICES = [
        ('ACCUMULATION', '主力吸筹'),
        ('DISTRIBUTION', '主力派发'),
        ('CONSOLIDATION', '中性整理'),
        ('SHAKEOUT', '震仓洗盘'),
    ]
    stock = models.ForeignKey(
        'StockInfo',
        on_delete=models.CASCADE,
        verbose_name='股票',
        db_index=True
    )
    start_date = models.DateField(verbose_name='平台起始日期', db_index=True)
    end_date = models.DateField(verbose_name='平台结束日期', db_index=True)
    duration = models.IntegerField(verbose_name='平台持续天数')
    high = models.DecimalField(max_digits=10, decimal_places=3, verbose_name='平台上轨')
    low = models.DecimalField(max_digits=10, decimal_places=3, verbose_name='平台下轨')
    vpoc = models.DecimalField(max_digits=10, decimal_places=3, verbose_name='平台成交量加权均价(VPOC-日线)')
    total_volume = models.BigIntegerField(verbose_name='平台期总成交量')
    quality_score = models.FloatField(verbose_name='平台质量分(0-1)', help_text='综合评估平台的吸筹/派发潜力')
    precise_vpoc = models.DecimalField(max_digits=10, decimal_places=3, verbose_name='精确VPOC(分钟级)', null=True, blank=True)
    internal_accumulation_intensity = models.FloatField(verbose_name='内部吸筹强度', null=True, blank=True, help_text='平台期内Tick级净主动买入量占比')
    breakout_quality_score = models.FloatField(verbose_name='突破质量分', null=True, blank=True, help_text='突破日微观结构评估分')
    platform_character = models.CharField(max_length=20, choices=CHARACTER_CHOICES, verbose_name='平台性质', null=True, blank=True)
    character_score = models.FloatField(verbose_name='平台性质分(-100~100)', null=True, blank=True, help_text='综合筹码、资金、结构证据的量化评分')
    platform_archetype = models.CharField(max_length=50, verbose_name='平台原型', null=True, blank=True, help_text='识别出此平台的原型名称')
    breakout_readiness_score = models.FloatField(verbose_name='突破准备度分(0-100)', null=True, blank=True, help_text='平台结束时，衡量其即将突破可能性的综合评分')
    goodness_of_fit_score = models.FloatField(verbose_name='拟合优度分(0-100)', null=True, blank=True, help_text='平台与最佳匹配原型之间的相似度得分')
    platform_conviction_score = models.FloatField(verbose_name='平台信念分(0-100)', null=True, blank=True, help_text='评估平台结构质量和主力控盘信念的综合得分')
    class Meta:
        abstract = True
        ordering = ['-start_date']
        unique_together = ('stock', 'start_date')

class PlatformFeature_SH(BasePlatformFeature):
    class Meta(BasePlatformFeature.Meta):
        abstract = False
        db_table = 'stock_platform_feature_sh'
        verbose_name = '矩形平台特征-上海'
        verbose_name_plural = verbose_name

class PlatformFeature_SZ(BasePlatformFeature):
    class Meta(BasePlatformFeature.Meta):
        abstract = False
        db_table = 'stock_platform_feature_sz'
        verbose_name = '矩形平台特征-深圳'
        verbose_name_plural = verbose_name

class PlatformFeature_CY(BasePlatformFeature):
    class Meta(BasePlatformFeature.Meta):
        abstract = False
        db_table = 'stock_platform_feature_cy'
        verbose_name = '矩形平台特征-创业'
        verbose_name_plural = verbose_name

class PlatformFeature_KC(BasePlatformFeature):
    class Meta(BasePlatformFeature.Meta):
        abstract = False
        db_table = 'stock_platform_feature_kc'
        verbose_name = '矩形平台特征-科创'
        verbose_name_plural = verbose_name

class PlatformFeature_BJ(BasePlatformFeature):
    class Meta(BasePlatformFeature.Meta):
        abstract = False
        db_table = 'stock_platform_feature_bj'
        verbose_name = '矩形平台特征-北京'
        verbose_name_plural = verbose_name

# 几何形态特征模型 - 趋势线
class BaseTrendlineFeature(models.Model):
    """
    【V2.0 · 微观博弈增强版】
    - 核心职责: 持久化存储通过算法识别出的最有效的趋势线的代数表达。
    - 设计思想: 存储线的方程（斜率、截距）而非每日价格，实现高效的动态计算和应用。
    - V2.0 新增: 引入 `touch_conviction_score` 字段，量化每次触及趋势线时的博弈强度。
    """
    LINE_TYPE_CHOICES = [
        ('support', '支撑线'),
        ('resistance', '阻力线'),
    ]
    stock = models.ForeignKey(
        'StockInfo',
        on_delete=models.CASCADE,
        verbose_name='股票',
        db_index=True
    )
    start_date = models.DateField(verbose_name='趋势线起始日期', db_index=True)
    end_date = models.DateField(verbose_name='趋势线结束日期', db_index=True)
    line_type = models.CharField(max_length=20, choices=LINE_TYPE_CHOICES, verbose_name='趋势线类型')
    slope = models.FloatField(verbose_name='斜率', help_text='y=mx+c中的m')
    intercept = models.FloatField(verbose_name='截距', help_text='y=mx+c中的c, 基于时间索引')
    touch_points = models.IntegerField(verbose_name='有效触及点数')
    validity_score = models.FloatField(verbose_name='趋势线有效性得分(0-1)')
    # V2.0 微观指标
    touch_conviction_score = models.FloatField(verbose_name='触及信念得分', null=True, blank=True, help_text='所有触及点微观博弈强度的平均分')

    class Meta:
        abstract = True
        ordering = ['-validity_score']
        unique_together = ('stock', 'start_date', 'line_type')

class TrendlineFeature_SH(BaseTrendlineFeature):
    class Meta(BaseTrendlineFeature.Meta):
        abstract = False
        db_table = 'stock_trendline_feature_sh'
        verbose_name = '趋势线特征-上海'
        verbose_name_plural = verbose_name

class TrendlineFeature_SZ(BaseTrendlineFeature):
    class Meta(BaseTrendlineFeature.Meta):
        abstract = False
        db_table = 'stock_trendline_feature_sz'
        verbose_name = '趋势线特征-深圳'
        verbose_name_plural = verbose_name

class TrendlineFeature_CY(BaseTrendlineFeature):
    class Meta(BaseTrendlineFeature.Meta):
        abstract = False
        db_table = 'stock_trendline_feature_cy'
        verbose_name = '趋势线特征-创业'
        verbose_name_plural = verbose_name

class TrendlineFeature_KC(BaseTrendlineFeature):
    class Meta(BaseTrendlineFeature.Meta):
        abstract = False
        db_table = 'stock_trendline_feature_kc'
        verbose_name = '趋势线特征-科创'
        verbose_name_plural = verbose_name

class TrendlineFeature_BJ(BaseTrendlineFeature):
    class Meta(BaseTrendlineFeature.Meta):
        abstract = False
        db_table = 'stock_trendline_feature_bj'
        verbose_name = '趋势线特征-北京'
        verbose_name_plural = verbose_name

# 多时间维度趋势线每日快照模型
class BaseMultiTimeframeTrendline(models.Model):
    """
    【V2.52 · 趋势信念版】
    - 核心职责: 持久化存储每日计算出的、代表不同时间维度市场共识的趋势线阵列。
    - 设计思想: 从记录单条线的“生命周期”转变为记录每日的“战场快照”，为动态分析提供基础。
    - V2.52 新增: 引入 `trend_conviction_score` 字段，通过融合几何、资金、筹码、行为
                 四大情报体系，对趋势本身的“质量”和“信念”进行深度量化评估。
    """
    LINE_TYPE_CHOICES = [('support', '支撑线'), ('resistance', '阻力线')]
    PERIOD_CHOICES = [(5, '5日'), (13, '13日'), (21, '21日'), (55, '55日')]
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, db_index=True)
    trade_date = models.DateField(verbose_name='交易日期', db_index=True)
    period = models.IntegerField(choices=PERIOD_CHOICES, verbose_name='时间周期')
    line_type = models.CharField(max_length=20, choices=LINE_TYPE_CHOICES, verbose_name='趋势线类型')
    slope = models.FloatField(verbose_name='斜率')
    intercept = models.FloatField(verbose_name='截距 (基于时间索引)')
    validity_score = models.FloatField(verbose_name='综合有效性得分(0-1)')
    # V2.52 新增趋势信念分数字段
    trend_conviction_score = models.FloatField(verbose_name='趋势信念分(0-100)', null=True, blank=True, help_text='评估趋势内在质量和主力信念的综合得分')
    class Meta:
        abstract = True
        unique_together = ('stock', 'trade_date', 'period', 'line_type')
        ordering = ['-trade_date', 'period']


class MultiTimeframeTrendline_SH(BaseMultiTimeframeTrendline):
    class Meta(BaseMultiTimeframeTrendline.Meta):
        abstract = False
        db_table = 'stock_trendline_matrix_sh'
        verbose_name = '趋势线矩阵-上海'

class MultiTimeframeTrendline_SZ(BaseMultiTimeframeTrendline):
    class Meta(BaseMultiTimeframeTrendline.Meta):
        abstract = False
        db_table = 'stock_trendline_matrix_sz'
        verbose_name = '趋势线矩阵-深圳'

class MultiTimeframeTrendline_CY(BaseMultiTimeframeTrendline):
    class Meta(BaseMultiTimeframeTrendline.Meta):
        abstract = False
        db_table = 'stock_trendline_matrix_cy'
        verbose_name = '趋势线矩阵-创业'

class MultiTimeframeTrendline_KC(BaseMultiTimeframeTrendline):
    class Meta(BaseMultiTimeframeTrendline.Meta):
        abstract = False
        db_table = 'stock_trendline_matrix_kc'
        verbose_name = '趋势线矩阵-科创'

class MultiTimeframeTrendline_BJ(BaseMultiTimeframeTrendline):
    class Meta(BaseMultiTimeframeTrendline.Meta):
        abstract = False
        db_table = 'stock_trendline_matrix_bj'
        verbose_name = '趋势线矩阵-北京'

# 趋势线动态事件模型
class BaseTrendlineEvent(models.Model):
    """
    【V2.2 · 启示录版】
    - 核心职责: 记录由趋势线矩阵动态演化而产生的关键交易信号事件。
    - 设计思想: 将“状态”（趋势线本身）与“事件”（拐点、突破、旗形）分离，便于策略回测和信号挖掘。
    - V2.2 升级: 全面同步并扩充事件类型，使其成为系统中所有几何与动态事件的“唯一真实来源”。
    """
    # V2.2 全面扩充事件类型
    EVENT_TYPE_CHOICES = [
        ('INFLECTION_ACCEL', '趋势加速'),
        ('INFLECTION_DECEL', '趋势减速'),
        ('INFLECTION_REVERSAL', '趋势反转'),
        ('CROSS_GOLDEN_DECISIVE', '决定性金叉'),
        ('CROSS_GOLDEN_TENTATIVE', '试探性金叉'),
        ('CROSS_DEATH_DECISIVE', '决定性死叉'),
        ('CROSS_DEATH_TENTATIVE', '试探性死叉'),
        ('RESONANCE_BULLISH_STRONG', '强烈多头共振'),
        ('RESONANCE_BEARISH_STRONG', '强烈空头共振'),
        ('DIVERGENCE_BEARISH_TOP', '顶部结构背离'),
        ('DIVERGENCE_BULLISH_BOTTOM', '底部结构背离'),
        ('COMPRESSION_SQUEEZE', '通道能量压缩'),
        ('FLAG_FORMED_D', '日线旗形确立'),
        ('FLAG_FORMED_W', '周线旗形确立'),
    ]
    stock = models.ForeignKey('StockInfo', on_delete=models.CASCADE, db_index=True)
    event_date = models.DateField(verbose_name='事件发生日期', db_index=True)
    event_type = models.CharField(max_length=50, choices=EVENT_TYPE_CHOICES, verbose_name='事件类型')
    details = models.JSONField(verbose_name='事件详情') # 存储相关周期、概率、特征等

    class Meta:
        abstract = True
        ordering = ['-event_date']

class TrendlineEvent_SH(BaseTrendlineEvent):
    class Meta(BaseTrendlineEvent.Meta):
        abstract = False
        db_table = 'stock_trendline_event_sh'
        verbose_name = '趋势线事件-上海'

class TrendlineEvent_SZ(BaseTrendlineEvent):
    class Meta(BaseTrendlineEvent.Meta):
        abstract = False
        db_table = 'stock_trendline_event_sz'
        verbose_name = '趋势线事件-深圳'

class TrendlineEvent_CY(BaseTrendlineEvent):
    class Meta(BaseTrendlineEvent.Meta):
        abstract = False
        db_table = 'stock_trendline_event_cy'
        verbose_name = '趋势线事件-创业'

class TrendlineEvent_KC(BaseTrendlineEvent):
    class Meta(BaseTrendlineEvent.Meta):
        abstract = False
        db_table = 'stock_trendline_event_kc'
        verbose_name = '趋势线事件-科创'

class TrendlineEvent_BJ(BaseTrendlineEvent):
    class Meta(BaseTrendlineEvent.Meta):
        abstract = False
        db_table = 'stock_trendline_event_bj'
        verbose_name = '趋势线事件-北京'





