# stock_models\chip_factors.py
from django.db import models
from django.utils.translation import gettext_lazy as _
import numpy as np
from numba import njit, float64, int64
from scipy.signal import find_peaks
from scipy.stats import linregress
from typing import List, Tuple, Dict, Optional, Any
import json
import base64
import pickle
from services.chip_matrix_dynamics_calculator import ChipMatrixDynamicsCalculator

class ChipFactorBase(models.Model):
    """
    筹码因子基础模型（抽象基类）
    """
    stock = models.ForeignKey(
        'StockInfo', 
        to_field='stock_code',
        on_delete=models.CASCADE,
        verbose_name='股票',
        db_index=True
    )
    trade_time = models.DateField(verbose_name='交易日期', db_index=True)
    # ========== 基础因子 ==========
    close = models.FloatField(verbose_name='收盘价', null=True, blank=True)
    weight_avg_cost = models.FloatField(verbose_name='加权平均成本', null=True, blank=True)
    # ========== 成本结构因子 ==========
    # 1. 相对位置因子
    price_to_weight_avg_ratio = models.FloatField(
        verbose_name='股价相对筹码均价偏离度',
        null=True, blank=True,
        help_text='(close - weight_avg_cost) / weight_avg_cost'
    )
    # 2. 筹码集中度因子
    chip_concentration_ratio = models.FloatField(
        verbose_name='筹码集中度',
        null=True, blank=True,
        help_text='(cost_85pct - cost_15pct) / (his_high - his_low)'
    )
    # 3. 分位位置因子
    price_percentile_position = models.FloatField(
        verbose_name='股价在筹码分布中的位置',
        null=True, blank=True,
        help_text='(close - cost_5pct) / (cost_95pct - cost_5pct)'
    )
    # 4. 筹码稳定性因子
    chip_stability = models.FloatField(
        verbose_name='筹码稳定性',
        null=True, blank=True,
        help_text='1 - (cost_85pct - cost_15pct) / (his_high - his_low)'
    )
    # ========== 获利压力因子 ==========
    profit_pressure = models.FloatField(
        verbose_name='获利盘压力',
        null=True, blank=True,
        help_text='(close - cost_50pct) / (cost_85pct - cost_15pct) * winner_rate'
    )
    profit_ratio = models.FloatField(
        verbose_name='获利比例',
        null=True, blank=True,
        help_text='从筹码分布计算的获利比例'
    )
    # ========== 筹码分布形态因子 ==========
    chip_entropy = models.FloatField(
        verbose_name='筹码分布熵值',
        null=True, blank=True,
        help_text='-∑(percent_i * ln(percent_i))'
    )
    chip_skewness = models.FloatField(
        verbose_name='筹码分布偏度',
        null=True, blank=True
    )
    chip_kurtosis = models.FloatField(
        verbose_name='筹码分布峰度',
        null=True, blank=True
    )
    chip_mean = models.FloatField(
        verbose_name='筹码均值',
        null=True, blank=True
    )
    chip_std = models.FloatField(
        verbose_name='筹码标准差',
        null=True, blank=True
    )
    # ========== 博弈状态因子 ==========
    winner_rate = models.FloatField(verbose_name='胜率', null=True, blank=True)
    win_rate_price_position = models.FloatField(
        verbose_name='胜率价格分位联动',
        null=True, blank=True,
        help_text='winner_rate * (close - cost_5pct) / (cost_95pct - cost_5pct)'
    )
    # ========== 关键分位成本 ==========
    cost_5pct = models.FloatField(verbose_name='5分位成本', null=True, blank=True)
    cost_15pct = models.FloatField(verbose_name='15分位成本', null=True, blank=True)
    cost_50pct = models.FloatField(verbose_name='50分位成本', null=True, blank=True)
    cost_85pct = models.FloatField(verbose_name='85分位成本', null=True, blank=True)
    cost_95pct = models.FloatField(verbose_name='95分位成本', null=True, blank=True)
    his_low = models.FloatField(verbose_name='历史最低价', null=True, blank=True)
    his_high = models.FloatField(verbose_name='历史最高价', null=True, blank=True)
    # ========== 结合量价验证因子 ==========
    turnover_rate = models.FloatField(verbose_name='换手率(%)', null=True, blank=True)
    volume_ratio = models.FloatField(verbose_name='量比', null=True, blank=True)
    # ========== 趋势强度因子 ==========
    # 移动平均线相对位置
    price_to_ma5_ratio = models.FloatField(
        verbose_name='股价相对MA5偏离度(%)',
        null=True, blank=True
    )
    price_to_ma21_ratio = models.FloatField(
        verbose_name='股价相对MA21偏离度(%)',
        null=True, blank=True
    )
    price_to_ma34_ratio = models.FloatField(
        verbose_name='股价相对MA34偏离度(%)',
        null=True, blank=True
    )
    price_to_ma55_ratio = models.FloatField(
        verbose_name='股价相对MA55偏离度(%)',
        null=True, blank=True
    )
    # 均线排列状态（多头：1，空头：-1，震荡：0）
    ma_arrangement_status = models.SmallIntegerField(
        verbose_name='均线排列状态',
        null=True, blank=True,
        help_text='1:多头排列(MA5>MA21>MA34>MA55), -1:空头排列, 0:震荡'
    )
    # 筹码成本均线与价格均线关系
    chip_cost_to_ma21_diff = models.FloatField(
        verbose_name='筹码成本均线与MA21差值',
        null=True, blank=True
    )
    # ========== 多峰形态识别因子 ==========
    peak_count = models.SmallIntegerField(
        verbose_name='筹码峰数量',
        null=True, blank=True
    )
    # 主峰位置（0:低位，1:中位，2:高位）
    main_peak_position = models.SmallIntegerField(
        verbose_name='主峰位置',
        null=True, blank=True,
        choices=[(0, '低位'), (1, '中位'), (2, '高位')]
    )
    # 峰间距离比率
    peak_distance_ratio = models.FloatField(
        verbose_name='峰间距离比率',
        null=True, blank=True,
        help_text='(最高峰价格-最低峰价格)/价格区间'
    )
    # 峰间筹码分布
    peak_concentration = models.FloatField(
        verbose_name='峰间集中度',
        null=True, blank=True,
        help_text='前两大峰筹码占比之和'
    )
    # 双峰/多峰识别
    is_double_peak = models.BooleanField(
        verbose_name='是否双峰形态',
        default=False
    )
    is_multi_peak = models.BooleanField(
        verbose_name='是否多峰形态',
        default=False
    )
    # ========== 筹码峰动态变化因子 ==========
    # 聚集度变化
    chip_convergence_ratio = models.FloatField(
        verbose_name='筹码聚集度',
        null=True, blank=True,
        help_text='(cost_50pct区间筹码占比) / (全区间筹码占比)'
    )
    # 发散度变化
    chip_divergence_ratio = models.FloatField(
        verbose_name='筹码发散度',
        null=True, blank=True,
        help_text='(cost_95pct - cost_5pct) / 历史价格区间'
    )
    # 筹码流动方向
    chip_flow_direction = models.SmallIntegerField(
        verbose_name='筹码流动方向',
        null=True, blank=True,
        choices=[(1, '向上流动'), (-1, '向下流动'), (0, '横盘整理')]
    )
    chip_flow_intensity = models.FloatField(
        verbose_name='筹码流动强度',
        null=True, blank=True
    )
    # ========== 高位筹码沉淀比例 ==========
    high_position_lock_ratio_90 = models.FloatField(
        verbose_name='90%分位以上筹码占比',
        help_text='成本在90-100分位的筹码占比，反映高位套牢盘',
        default=0.0  # 添加默认值
    )
    # ========== 主力成本区间锁定度 ==========
    main_cost_range_ratio = models.FloatField(
        verbose_name='主力成本区间锁定比例',
        help_text='(cost_50pct±10%)区间筹码占比，反映主力控盘度',
        default=0.5  # 添加默认值
    )
    # ========== 长线锁定筹码 ==========
    long_term_chip_ratio = models.FloatField(
        verbose_name='长线锁定筹码比例(>60日)',
        null=True, blank=True,
        help_text='基于历史筹码分布变化推算的长线锁定筹码',
        default=0.5  # 添加默认值
    )
    # ========== 短线交易筹码 ==========
    short_term_chip_ratio = models.FloatField(
        verbose_name='短线筹码比例(<5日)',
        null=True, blank=True,
        help_text='基于换手率推算的短线交易筹码占比',
        default=0.2  # 添加默认值
    )
    # ========== 趋势与反转信号 ==========
    # 趋势确认信号
    trend_confirmation_score = models.FloatField(
        verbose_name='趋势确认得分',
        null=True, blank=True,
        help_text='结合筹码、均线、量能的多维度评分'
    )
    # 反转预警信号
    reversal_warning_score = models.FloatField(
        verbose_name='反转预警得分',
        null=True, blank=True
    )
    # 筹码结构状态
    chip_structure_state = models.CharField(
        max_length=20,
        verbose_name='筹码结构状态',
        null=True, blank=True,
        choices=[
            ('accumulation', '吸筹阶段'),
            ('lifting', '拉升阶段'),
            ('distribution', '派发阶段'),
            ('decline', '回落阶段'),
            ('consolidation', '整理阶段')
        ]
    )
    # ========== 时间维度扩展 ==========
    # 筹码峰迁移速度（5日变化）
    peak_migration_speed_5d = models.FloatField(
        verbose_name='筹码峰5日迁移速度',
        null=True, blank=True
    )
    # 筹码稳定性变化（5日）
    chip_stability_change_5d = models.FloatField(
        verbose_name='筹码稳定性5日变化',
        null=True, blank=True
    )
    # ========== 市场适应性因子 ==========
    # 波动率调整的筹码因子
    volatility_adjusted_concentration = models.FloatField(
        verbose_name='波动率调整筹码集中度',
        null=True, blank=True
    )
    # 相对强度RSI结合筹码
    chip_rsi_divergence = models.FloatField(
        verbose_name='筹码RSI背离度',
        null=True, blank=True
    )
    # ========== 计算状态 ==========
    calc_status = models.CharField(
        max_length=20,
        verbose_name='计算状态',
        default='pending',
        choices=[
            ('pending', '待计算'),
            ('success', '计算成功'),
            ('failed', '计算失败')
        ]
    )
    # ========== 新增：基于百分比变化的动态因子 ==========
    percent_change_convergence = models.FloatField(
        verbose_name='百分比变化收敛度',
        null=True, blank=True,
        help_text='基于筹码百分比变化的收敛度指标'
    )
    percent_change_divergence = models.FloatField(
        verbose_name='百分比变化发散度',
        null=True, blank=True,
        help_text='基于筹码百分比变化的发散度指标'
    )
    absolute_change_strength = models.FloatField(
        verbose_name='绝对变化强度',
        null=True, blank=True,
        help_text='筹码百分比绝对变化的总强度'
    )
    accumulation_signal_score = models.FloatField(
        verbose_name='吸筹信号强度',
        null=True, blank=True,
        help_text='基于绝对变化的吸筹信号强度(0-1)'
    )
    distribution_signal_score = models.FloatField(
        verbose_name='派发信号强度',
        null=True, blank=True,
        help_text='基于绝对变化的派发信号强度(0-1)'
    )
    main_force_activity_index = models.FloatField(
        verbose_name='主力活跃度指数',
        null=True, blank=True,
        help_text='主力资金活跃程度(0-1)'
    )
    net_migration_direction = models.FloatField(
        verbose_name='净迁移方向',
        null=True, blank=True,
        help_text='正值向上迁移，负值向下迁移'
    )
    migration_convergence_ratio = models.FloatField(
        verbose_name='迁移收敛比率',
        null=True, blank=True,
        help_text='筹码迁移的收敛程度(0-1)'
    )
    # ========== 新增：信号验证因子 ==========
    signal_quality_score = models.FloatField(
        verbose_name='信号质量评分',
        null=True, blank=True,
        help_text='基于噪声水平的信号质量(0-1)'
    )
    behavior_confirmation = models.FloatField(
        verbose_name='行为确认度',
        null=True, blank=True,
        help_text='多种信号的一致性确认度(0-1)'
    )
    pressure_release_index = models.FloatField(
        verbose_name='压力释放指数',
        null=True, blank=True,
        help_text='套牢盘压力释放程度(0-1)'
    )
    support_resistance_ratio = models.FloatField(
        verbose_name='支撑阻力比',
        null=True, blank=True,
        help_text='支撑强度/阻力强度，>1表示支撑强'
    )
    # ========== 基于Tick数据的筹码微观结构因子 ==========
    # 1. 日内筹码分布统计因子（基于tick成交量在价格区间的分布）
    intraday_chip_concentration = models.FloatField(
        verbose_name='日内筹码集中度',
        null=True, blank=True,
        help_text='基于tick数据计算的日内筹码集中度（HHI指数）'
    )
    intraday_chip_entropy = models.FloatField(
        verbose_name='日内筹码分布熵值',
        null=True, blank=True,
        help_text='基于tick成交量的日内筹码分布熵值'
    )
    intraday_price_distribution_skewness = models.FloatField(
        verbose_name='日内价格分布偏度',
        null=True, blank=True,
        help_text='基于tick成交价格分布的偏度'
    )
    intraday_price_range_ratio = models.FloatField(
        verbose_name='日内价格区间占比',
        null=True, blank=True,
        help_text='(High-Low)/Close，反映日内波动幅度'
    )
    # 2. 日内筹码交换强度因子
    intraday_chip_turnover_intensity = models.FloatField(
        verbose_name='日内筹码换手强度',
        null=True, blank=True,
        help_text='基于tick数据的单位时间筹码交换率'
    )
    tick_level_chip_flow = models.FloatField(
        verbose_name='tick级筹码净流动',
        null=True, blank=True,
        help_text='基于tick买卖方向判断的筹码净流动比例'
    )
    # 3. 日内筹码分层锁定因子
    intraday_low_lock_ratio = models.FloatField(
        verbose_name='日内低位筹码锁定比例',
        null=True, blank=True,
        help_text='日内低价区tick成交量占比（反映低位锁定）'
    )
    intraday_high_lock_ratio = models.FloatField(
        verbose_name='日内高位筹码锁定比例',
        null=True, blank=True,
        help_text='日内高价区tick成交量占比（反映高位沉淀）'
    )
    # 4. 日内筹码成本重心迁移因子
    intraday_cost_center_migration = models.FloatField(
        verbose_name='日内成本重心迁移幅度',
        null=True, blank=True,
        help_text='日内成交加权成本相对于开盘的迁移百分比'
    )
    intraday_cost_center_volatility = models.FloatField(
        verbose_name='日内成本重心波动率',
        null=True, blank=True,
        help_text='日内tick级成本重心的标准差'
    )
    # 5. 日内筹码峰谷识别因子
    intraday_peak_valley_ratio = models.FloatField(
        verbose_name='日内峰谷成交比',
        null=True, blank=True,
        help_text='日内价格峰值与谷值区域成交量比率'
    )
    intraday_trough_filling_degree = models.FloatField(
        verbose_name='日内筹码谷填充度',
        null=True, blank=True,
        help_text='筹码分布谷底区域的成交量填充程度'
    )
    # 6. 日内筹码异常交换因子
    tick_abnormal_volume_ratio = models.FloatField(
        verbose_name='tick异常成交量比例',
        null=True, blank=True,
        help_text='超过均值3倍标准差tick的成交量占比'
    )
    tick_clustering_index = models.FloatField(
        verbose_name='tick成交聚类指数',
        null=True, blank=True,
        help_text='连续同向tick成交的聚合程度'
    )
    # 7. 日内筹码压力测试因子
    intraday_support_test_count = models.IntegerField(
        verbose_name='日内支撑测试次数',
        null=True, blank=True,
        help_text='价格触及关键支撑位时的tick成交量次数'
    )
    intraday_resistance_test_count = models.IntegerField(
        verbose_name='日内阻力测试次数',
        null=True, blank=True,
        help_text='价格触及关键阻力位时的tick成交量次数'
    )
    # 8. 日内筹码交换效率因子
    tick_chip_transfer_efficiency = models.FloatField(
        verbose_name='tick筹码转移效率',
        null=True, blank=True,
        help_text='单位价格变动带来的筹码转移量'
    )
    intraday_chip_consolidation_degree = models.FloatField(
        verbose_name='日内筹码整固度',
        null=True, blank=True,
        help_text='窄幅震荡区间内的tick成交量占比'
    )
    # 9. 日内筹码博弈状态因子
    intraday_chip_game_index = models.FloatField(
        verbose_name='日内筹码博弈指数',
        null=True, blank=True,
        help_text='基于tick买卖博弈的筹码状态指数'
    )
    tick_chip_balance_ratio = models.FloatField(
        verbose_name='tick筹码平衡比',
        null=True, blank=True,
        help_text='买卖双方tick成交量平衡度'
    )
    # 10. 数据质量标识
    tick_data_quality_score = models.FloatField(
        verbose_name='tick数据质量评分',
        null=True, blank=True,
        help_text='基于tick数据完整性和连续性的质量评分(0-1)'
    )
    intraday_factor_calc_method = models.CharField(
        max_length=20,
        verbose_name='日内因子计算方法',
        null=True, blank=True,
        choices=[
            ('tick_based', '基于tick数据'),
            ('minute_approximated', '分钟线近似'),
            ('daily_only', '仅日线数据')
        ],
        default='daily_only'
    )
    calc_time = models.DateTimeField(verbose_name='计算时间', auto_now=True)
    error_message = models.TextField(verbose_name='错误信息', null=True, blank=True)
    class Meta:
        abstract = True
        verbose_name = '筹码因子基础'
        verbose_name_plural = '筹码因子基础'
        unique_together = ('stock', 'trade_time')
    def __str__(self):
        return f"{self.stock.stock_code} {self.trade_time}"
    def save(self, *args, **kwargs):
        """
        重写save方法，自动将所有FloatField字段四舍五入保留3位小数
        """
        for field in self._meta.fields:
            if isinstance(field, models.FloatField):
                val = getattr(self, field.name)
                if val is not None:
                    try:
                        setattr(self, field.name, round(float(val), 3))
                    except (ValueError, TypeError):
                        pass
        super().save(*args, **kwargs)

    # ========== 计算聚散度的方法 ==========
    @classmethod
    def calculate_convergence_divergence(cls,chip_dynamics_result: Dict[str, any]) -> Dict[str, float]:
        """
        计算筹码聚散度因子 - 基于百分比绝对变化
        返回:
            Dict包含聚散度相关因子
        """
        try:
            if not chip_dynamics_result or chip_dynamics_result.get('analysis_status') != 'success':
                return cls._get_default_convergence_factors()
            # 获取核心数据
            convergence_metrics = chip_dynamics_result.get('convergence_metrics', {})
            behavior_patterns = chip_dynamics_result.get('behavior_patterns', {})
            migration_patterns = chip_dynamics_result.get('migration_patterns', {})
            absolute_signals = chip_dynamics_result.get('absolute_change_signals', {})
            # 1. 基础聚散度
            factors = {
                'percent_change_convergence': convergence_metrics.get('comprehensive_convergence', 0.5),
                'percent_change_divergence': 1.0 - convergence_metrics.get('comprehensive_convergence', 0.5),
                'convergence_strength': convergence_metrics.get('convergence_strength', 0.0),
                'divergence_strength': convergence_metrics.get('divergence_strength', 0.0),
            }
            # 2. 迁移相关因子
            factors['net_migration_direction'] = migration_patterns.get('net_migration_direction', 0.0)
            factors['migration_convergence_ratio'] = convergence_metrics.get('migration_convergence', 0.5)
            # 3. 绝对变化强度
            # 计算所有显著变化的总强度
            increase_areas = absolute_signals.get('significant_increase_areas', [])
            decrease_areas = absolute_signals.get('significant_decrease_areas', [])
            total_increase = sum(abs(area['change']) for area in increase_areas)
            total_decrease = sum(abs(area['change']) for area in decrease_areas)
            factors['absolute_change_strength'] = (total_increase + total_decrease) / 100.0
            # 4. 吸筹/派发信号强度
            accumulation = behavior_patterns.get('accumulation', {})
            distribution = behavior_patterns.get('distribution', {})
            factors['accumulation_signal_score'] = accumulation.get('strength', 0.0)
            factors['distribution_signal_score'] = distribution.get('strength', 0.0)
            # 5. 主力活跃度
            factors['main_force_activity_index'] = behavior_patterns.get('main_force_activity', 0.0)
            # 6. 信号质量
            factors['signal_quality_score'] = absolute_signals.get('signal_quality', 0.0)
            # 7. 行为确认度（多种信号的一致性）
            confirmation_score = 0.0
            confirmation_count = 0
            # 吸筹确认
            if accumulation.get('detected', False):
                confirmation_score += accumulation.get('strength', 0.0)
                confirmation_count += 1
            # 派发确认
            if distribution.get('detected', False):
                confirmation_score += distribution.get('strength', 0.0)
                confirmation_count += 1
            # 迁移方向确认
            if abs(factors['net_migration_direction']) > 0.1:
                confirmation_score += min(1.0, abs(factors['net_migration_direction']) / 10.0)
                confirmation_count += 1
            factors['behavior_confirmation'] = (
                confirmation_score / confirmation_count if confirmation_count > 0 else 0.0
            )
            # 8. 压力与支撑
            pressure_metrics = chip_dynamics_result.get('pressure_metrics', {})
            factors['pressure_release_index'] = pressure_metrics.get('pressure_release', 0.0)
            support = pressure_metrics.get('support_strength', 0.3)
            resistance = pressure_metrics.get('resistance_strength', 0.3)
            factors['support_resistance_ratio'] = support / resistance if resistance > 0 else 1.0
            return factors
        except Exception as e:
            print(f"❌ 计算聚散度失败: {e}")
            return cls._get_default_convergence_factors()
    @classmethod
    def _get_default_convergence_factors(cls) -> Dict[str, float]:
        """获取默认聚散度因子"""
        return {
            'percent_change_convergence': 0.5,
            'percent_change_divergence': 0.5,
            'convergence_strength': 0.0,
            'divergence_strength': 0.0,
            'absolute_change_strength': 0.0,
            'accumulation_signal_score': 0.0,
            'distribution_signal_score': 0.0,
            'main_force_activity_index': 0.0,
            'net_migration_direction': 0.0,
            'migration_convergence_ratio': 0.5,
            'signal_quality_score': 0.0,
            'behavior_confirmation': 0.0,
            'pressure_release_index': 0.0,
            'support_resistance_ratio': 1.0
        }
    def update_from_chip_dynamics(self, chip_dynamics_result: Dict[str, any]):
        """
        从筹码动态分析结果更新因子
        Args:
            chip_dynamics_result: AdvancedChipDynamicsService 的分析结果
        """
        try:
            if not chip_dynamics_result or chip_dynamics_result.get('analysis_status') != 'success':
                return False
            # 1. 更新聚散度因子
            convergence_factors = self.calculate_convergence_divergence(chip_dynamics_result)
            for key, value in convergence_factors.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            # 2. 更新集中度因子
            concentration_metrics = chip_dynamics_result.get('concentration_metrics', {})
            if concentration_metrics:
                self.chip_concentration_ratio = concentration_metrics.get('comprehensive_concentration', 0.5)
                self.chip_entropy = -np.log(concentration_metrics.get('entropy_concentration', 0.5) + 1e-10)
                self.chip_skewness = concentration_metrics.get('chip_skewness', 0.0)
                self.chip_kurtosis = concentration_metrics.get('chip_kurtosis', 0.0)
            # 3. 更新压力因子
            pressure_metrics = chip_dynamics_result.get('pressure_metrics', {})
            if pressure_metrics:
                self.profit_ratio = pressure_metrics.get('profit_pressure', 0.5)
            # 4. 更新行为模式
            behavior_patterns = chip_dynamics_result.get('behavior_patterns', {})
            if behavior_patterns:
                accumulation = behavior_patterns.get('accumulation', {})
                distribution = behavior_patterns.get('distribution', {})
                if accumulation.get('detected', False):
                    if accumulation.get('strength', 0) > distribution.get('strength', 0):
                        self.chip_structure_state = 'accumulation'
                elif distribution.get('detected', False):
                    self.chip_structure_state = 'distribution'
                elif behavior_patterns.get('consolidation', {}).get('detected', False):
                    self.chip_structure_state = 'consolidation'
                elif behavior_patterns.get('breakout_preparation', {}).get('detected', False):
                    self.chip_structure_state = 'lifting'
            # 5. 计算综合趋势得分
            self._calculate_trend_score(chip_dynamics_result)
            self.calc_status = 'success'
            return True
        except Exception as e:
            print(f"❌ 更新因子失败: {e}")
            self.calc_status = 'failed'
            self.error_message = str(e)
            return False
    def _calculate_trend_score(self):
        """基于能量场和tick数据的趋势得分计算"""
        try:
            # 基础趋势得分（原有逻辑）
            base_score = 0.5
            if hasattr(self, 'net_energy_flow'):
                # 净能量流向
                if self.net_energy_flow > 10:
                    base_score += 0.2
                elif self.net_energy_flow < -10:
                    base_score -= 0.2
            if hasattr(self, 'game_intensity'):
                # 博弈强度
                if self.game_intensity > 0.6:
                    base_score += 0.1
                elif self.game_intensity < 0.3:
                    base_score -= 0.1
            # Tick数据增强
            if self.intraday_chip_quality_score > 0.5:
                tick_bonus = 0.0
                # 1. 日内筹码净流动
                if self.tick_level_chip_flow > 0.1:
                    tick_bonus += 0.1
                elif self.tick_level_chip_flow < -0.1:
                    tick_bonus -= 0.1
                # 2. 日内成本重心迁移
                if self.intraday_cost_center_migration > 0.5:
                    tick_bonus += 0.05
                elif self.intraday_cost_center_migration < -0.5:
                    tick_bonus -= 0.05
                # 3. 日内主力活跃度
                if self.intraday_main_force_activity > 0.4:
                    tick_bonus += 0.05
                base_score = min(1.0, max(0.0, base_score + tick_bonus))
            return round(base_score, 3)
        except Exception as e:
            print(f"⚠️ [趋势得分] 计算异常: {e}")
            return 0.5

# 深交所主板筹码因子分表
class ChipFactorSZ(ChipFactorBase):
    class Meta:
        verbose_name = '筹码因子SZ'
        verbose_name_plural = '筹码因子SZ'
        db_table = 'stock_chip_factor_sz'

# 上交所主板筹码因子分表
class ChipFactorSH(ChipFactorBase):
    class Meta:
        verbose_name = '筹码因子SH'
        verbose_name_plural = '筹码因子SH'
        db_table = 'stock_chip_factor_sh'

# 创业板筹码因子分表
class ChipFactorCY(ChipFactorBase):
    class Meta:
        verbose_name = '筹码因子CY'
        verbose_name_plural = '筹码因子CY'
        db_table = 'stock_chip_factor_cy'

# 科创板筹码因子分表
class ChipFactorKC(ChipFactorBase):
    class Meta:
        verbose_name = '筹码因子KC'
        verbose_name_plural = '筹码因子KC'
        db_table = 'stock_chip_factor_kc'

# 北交所筹码因子分表
class ChipFactorBJ(ChipFactorBase):
    class Meta:
        verbose_name = '筹码因子BJ'
        verbose_name_plural = '筹码因子BJ'
        db_table = 'stock_chip_factor_bj'

class ChipHoldingMatrixBase(models.Model):
    """
    重构的筹码持有时间矩阵基类 - 存储动态分析结果
    """
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        on_delete=models.CASCADE,
        verbose_name='股票',
        db_index=True
    )
    trade_time = models.DateField(verbose_name='交易日期', db_index=True)
    # ========== 基础持有时间因子 ==========
    short_term_ratio = models.FloatField(
        verbose_name='短线筹码比例(<5日)',
        null=True, blank=True,
        default=0.2
    )
    mid_term_ratio = models.FloatField(
        verbose_name='中线筹码比例(5-60日)',
        null=True, blank=True,
        default=0.3
    )
    long_term_ratio = models.FloatField(
        verbose_name='长线筹码比例(>60日)',
        null=True, blank=True,
        default=0.5
    )
    # ========== 1. 矩阵数据 (二进制压缩存储) ==========
    # 价格网格 (保留 JSON，作为矩阵的坐标轴，必须可读)
    price_grid = models.JSONField(
        verbose_name='价格网格',
        null=True, blank=True
    )
    # 原始筹码矩阵 (已存在)
    compressed_matrix = models.BinaryField(
        verbose_name='压缩筹码矩阵',
        null=True, blank=True,
        help_text='zlib压缩的原始筹码分布矩阵'
    )
    # 变化矩阵 (替代 percent_change_matrix JSON字段)
    compressed_change_matrix = models.BinaryField(
        verbose_name='压缩变化矩阵',
        null=True, blank=True,
        help_text='zlib压缩的百分比变化矩阵'
    )
    # ========== 2. 综合分析数据 (合并存储) ==========
    # 图表信号集合 (合并 absolute_change_signals, behavior_patterns, key_battle_zones)
    chart_signals = models.JSONField(
        verbose_name='图表信号集合',
        null=True, blank=True,
        help_text='包含关键博弈区、形态识别区域、绝对变化信号，用于前端绘图'
    )
    # 补充指标集合 (合并 concentration/pressure/convergence 等剩余的次要指标)
    extra_metrics = models.JSONField(
        verbose_name='补充指标集合',
        null=True, blank=True,
        help_text='存储偏度、峰度、中间过程值等次要指标'
    )
    # 绝对变化分析详情 (保留用于深度复盘)
    absolute_change_analysis = models.JSONField(
        verbose_name='绝对变化分析详情',
        null=True, blank=True
    )
    # ========== 博弈能量场因子 ==========
    absorption_energy = models.FloatField(
        verbose_name='吸收能量(0-100)',
        null=True, blank=True,
        default=0.0
    )
    distribution_energy = models.FloatField(
        verbose_name='派发能量(0-100)',
        null=True, blank=True,
        default=0.0
    )
    net_energy_flow = models.FloatField(
        verbose_name='净能量流向(-100~100)',
        null=True, blank=True,
        default=0.0
    )
    game_intensity = models.FloatField(
        verbose_name='博弈强度(0-1)',
        null=True, blank=True,
        default=0.0
    )
    breakout_potential = models.FloatField(
        verbose_name='突破势能(0-100)',
        null=True, blank=True,
        default=0.0
    )
    energy_concentration = models.FloatField(
        verbose_name='能量集中度(0-1)',
        null=True, blank=True,
        default=0.0
    )
    # ========== 核心指标扁平化 (从JSON提取以提升性能) ==========
    # 1. 集中度相关 (来源: concentration_metrics)
    concentration_comprehensive = models.FloatField(
        verbose_name='综合集中度',
        null=True, blank=True,
        help_text='综合熵值、峰度和主力控盘度的评分'
    )
    concentration_entropy = models.FloatField(
        verbose_name='熵值集中度',
        null=True, blank=True
    )
    concentration_peak = models.FloatField(
        verbose_name='峰值集中度',
        null=True, blank=True
    )
    # 2. 压力与支撑 (来源: pressure_metrics)
    pressure_trapped = models.FloatField(
        verbose_name='套牢盘压力',
        null=True, blank=True,
        help_text='当前价上方10%区间的筹码占比'
    )
    pressure_profit = models.FloatField(
        verbose_name='获利盘比例',
        null=True, blank=True
    )
    support_strength = models.FloatField(
        verbose_name='下方支撑强度',
        null=True, blank=True,
        help_text='当前价下方5%区间的筹码占比'
    )
    resistance_strength = models.FloatField(
        verbose_name='上方阻力强度',
        null=True, blank=True,
        help_text='当前价上方5%区间的筹码占比'
    )
    # 3. 聚散度与迁移 (来源: convergence_metrics)
    convergence_comprehensive = models.FloatField(
        verbose_name='综合收敛度',
        null=True, blank=True
    )
    convergence_migration = models.FloatField(
        verbose_name='迁移收敛度',
        null=True, blank=True
    )
    # 4. 行为模式强度 (来源: behavior_patterns)
    behavior_accumulation = models.FloatField(
        verbose_name='吸筹强度',
        null=True, blank=True,
        default=0.0
    )
    behavior_distribution = models.FloatField(
        verbose_name='派发强度',
        null=True, blank=True,
        default=0.0
    )
    behavior_consolidation = models.FloatField(
        verbose_name='整理强度',
        null=True, blank=True,
        default=0.0
    )
    # ========== 绝对变化分析结果 ==========
    absolute_change_analysis = models.JSONField(
        verbose_name='绝对变化分析',
        null=True, blank=True,
        help_text='基于绝对值变化的分析结果，用于纠偏'
    )
    # ========== 验证信息 ==========
    validation_score = models.FloatField(
        verbose_name='验证分数',
        null=True, blank=True,
        help_text='计算结果的可信度评分(0-1)'
    )
    validation_warnings = models.JSONField(
        verbose_name='验证警告',
        null=True, blank=True
    )
    # ========== Tick数据增强因子 ==========
    # 1. 日内筹码分布质量因子
    intraday_chip_quality_score = models.FloatField(
        verbose_name='日内筹码数据质量评分(0-1)',
        null=True, blank=True,
        default=0.0,
        help_text='基于tick数据完整性和连续性的质量评分'
    )
    intraday_calc_method = models.CharField(
        max_length=20,
        verbose_name='日内因子计算方法',
        null=True, blank=True,
        choices=[
            ('tick_based', '基于tick数据'),
            ('minute_approximated', '分钟线近似'),
            ('daily_only', '仅日线数据')
        ],
        default='daily_only'
    )
    # 11. 日内主力行为识别因子
    intraday_main_force_activity = models.FloatField(
        verbose_name='日内主力活跃度',
        null=True, blank=True,
        default=0.0,
        help_text='基于tick异常成交量和大单的主力量化'
    )
    intraday_accumulation_confidence = models.FloatField(
        verbose_name='日内吸筹置信度',
        null=True, blank=True,
        default=0.0,
        help_text='基于tick数据的吸筹行为可信度评分'
    )
    intraday_distribution_confidence = models.FloatField(
        verbose_name='日内派发置信度',
        null=True, blank=True,
        default=0.0,
        help_text='基于tick数据的派发行为可信度评分'
    )
    # 12. 日内市场微观结构因子
    intraday_market_microstructure = models.JSONField(
        verbose_name='日内市场微观结构',
        null=True, blank=True,
        help_text='存储tick级市场微观结构指标'
    )
    # 13. Tick数据原始统计（用于后续分析）
    tick_data_summary = models.JSONField(
        verbose_name='tick数据统计摘要',
        null=True, blank=True,
        help_text='包含tick数量、时间跨度、缺失间隔等统计信息'
    )
    # ========== 计算状态 ==========
    calc_status = models.CharField(
        max_length=20,
        verbose_name='计算状态',
        default='pending',
        choices=[
            ('pending', '待计算'),
            ('processing', '计算中'),
            ('success', '成功'),
            ('partial', '部分成功'),
            ('failed', '失败')
        ]
    )
    analysis_method = models.CharField(
        max_length=50,
        verbose_name='分析方法',
        default='advanced_dynamics',
        choices=[
            ('legacy_holding', '传统持有时间'),
            ('advanced_dynamics', '高级动态分析'),
            ('hybrid', '混合方法')
        ]
    )
    calc_time = models.DateTimeField(verbose_name='计算时间', auto_now=True)
    error_message = models.TextField(verbose_name='错误信息', null=True, blank=True)
    # ========== 数据源标记 ==========
    used_minute_data = models.BooleanField(
        verbose_name='使用分钟数据',
        default=True
    )
    used_tick_data = models.BooleanField(
        verbose_name='使用逐笔数据',
        default=False
    )
    used_percent_data = models.BooleanField(
        verbose_name='使用百分比数据',
        default=True
    )
    class Meta:
        abstract = True
        verbose_name = '筹码持有时间矩阵基础'
        verbose_name_plural = '筹码持有时间矩阵基础'
        unique_together = ('stock', 'trade_time')
        indexes = [
            models.Index(fields=['stock', 'trade_time'])
        ]
    def __str__(self):
        return f"{self.stock.stock_code} {self.trade_time} 动态分析"
    def save(self, *args, **kwargs):
        """
        重写save方法，自动将所有FloatField字段四舍五入保留3位小数
        """
        for field in self._meta.fields:
            if isinstance(field, models.FloatField):
                val = getattr(self, field.name)
                if val is not None:
                    try:
                        setattr(self, field.name, round(float(val), 3))
                    except (ValueError, TypeError):
                        pass
        super().save(*args, **kwargs)

    def save_dynamics_result(self, dynamics_result: Dict[str, Any]):
        # [V3.1.3] 方法说明：保存动态分析结果。集成数据类型强制降级(float32)，优化传入计算引擎的矩阵内存布局，激活底层SIMD向量化性能
        import zlib
        import json
        import numpy as np
        Calculator = ChipMatrixDynamicsCalculator
        try:
            if not dynamics_result or dynamics_result.get('analysis_status') != 'success':
                self.calc_status = 'failed'
                self.save(update_fields=['calc_status', 'error_message', 'calc_time'])
                return False
            required = ['price_grid', 'percent_change_matrix']
            if any(f not in dynamics_result for f in required):
                self.calc_status = 'failed'
                self.error_message = "缺少必要字段"
                self.save(update_fields=['calc_status', 'error_message'])
                return False
            tick_factors = dynamics_result.get('tick_enhanced_factors', {})
            if tick_factors:
                self.intraday_chip_quality_score = tick_factors.get('tick_data_quality_score', 0.0)
                self.intraday_calc_method = tick_factors.get('intraday_factor_calc_method', 'daily_only')
                self.intraday_main_force_activity = tick_factors.get('intraday_main_force_activity', 0.0)
                self.intraday_accumulation_confidence = tick_factors.get('intraday_accumulation_confidence', 0.0)
                self.intraday_distribution_confidence = tick_factors.get('intraday_distribution_confidence', 0.0)
                self.used_tick_data = True
                self.tick_data_summary = Calculator.clean_structure(tick_factors.get('tick_data_summary'), precision=3)
                microstructure_data = Calculator.clean_structure(tick_factors.get('intraday_market_microstructure'), precision=3)
                if microstructure_data is None:
                    microstructure_data = {}
                critical_tick_keys = ['intraday_chip_consolidation_degree','intraday_peak_valley_ratio','intraday_price_distribution_skewness','intraday_resistance_test_count','intraday_support_test_count','intraday_trough_filling_degree','tick_abnormal_volume_ratio','tick_chip_balance_ratio','tick_chip_transfer_efficiency','tick_clustering_index','tick_level_chip_flow','intraday_chip_concentration','intraday_chip_entropy','intraday_chip_turnover_intensity','intraday_cost_center_migration','intraday_cost_center_volatility','intraday_low_lock_ratio','intraday_high_lock_ratio','intraday_chip_game_index']
                for key in critical_tick_keys:
                    if key in tick_factors:
                        microstructure_data[key] = tick_factors[key]
                self.intraday_market_microstructure = microstructure_data
            try:
                change_matrix = dynamics_result.get('percent_change_matrix', [])
                price_grid = dynamics_result.get('price_grid', [])
                current_price = dynamics_result.get('current_price', 0)
                if change_matrix and price_grid and current_price:
                    latest_change = np.array(change_matrix[-1], dtype=np.float32)
                    p_grid = np.array(price_grid, dtype=np.float32)
                    self.absolute_change_analysis = Calculator.calculate_absolute_change_analysis(latest_change, p_grid, current_price)
                else:
                    self.absolute_change_analysis = Calculator.get_default_absolute_analysis()
                self.absolute_change_analysis = Calculator.clean_structure(self.absolute_change_analysis, precision=3)
            except Exception as e:
                print(f"⚠️ 绝对变化分析失败: {e}")
                self.absolute_change_analysis = Calculator.get_default_absolute_analysis()
            game_energy = dynamics_result.get('game_energy_result', {})
            if game_energy:
                self.absorption_energy = round(max(0.0, game_energy.get('absorption_energy', 0.0)), 3)
                self.distribution_energy = round(max(0.0, game_energy.get('distribution_energy', 0.0)), 3)
                self.net_energy_flow = round(game_energy.get('net_energy_flow', 0.0), 3)
                self.game_intensity = round(max(0.0, min(1.0, game_energy.get('game_intensity', 0.0))), 3)
                self.breakout_potential = round(max(0.0, game_energy.get('breakout_potential', 0.0)), 3)
                self.energy_concentration = round(max(0.0, min(1.0, game_energy.get('energy_concentration', 0.0))), 3)
                self.fake_distribution_flag = bool(game_energy.get('fake_distribution_flag', False))
                kbz = game_energy.get('key_battle_zones', [])
                if not kbz:
                    kbz = Calculator.create_default_key_battle_zones(price_grid, current_price)
            else:
                self.absorption_energy = 0.0
                self.distribution_energy = 0.0
                self.net_energy_flow = 0.0
                self.game_intensity = 0.0
                self.breakout_potential = 0.0
                self.energy_concentration = 0.0
                self.fake_distribution_flag = False
                kbz = []
            conc_metrics = dynamics_result.get('concentration_metrics', {})
            press_metrics = dynamics_result.get('pressure_metrics', {})
            conv_metrics = dynamics_result.get('convergence_metrics', {})
            behav_patterns = dynamics_result.get('behavior_patterns', {})
            self.concentration_comprehensive = round(conc_metrics.get('comprehensive_concentration', 0.0), 3)
            self.concentration_entropy = round(conc_metrics.get('entropy_concentration', 0.0), 3)
            self.concentration_peak = round(conc_metrics.get('peak_concentration', 0.0), 3)
            self.pressure_trapped = round(press_metrics.get('trapped_pressure', 0.0), 3)
            self.pressure_profit = round(press_metrics.get('profit_pressure', 0.0), 3)
            self.support_strength = round(press_metrics.get('support_strength', 0.0), 3)
            self.resistance_strength = round(press_metrics.get('resistance_strength', 0.0), 3)
            self.convergence_comprehensive = round(conv_metrics.get('comprehensive_convergence', 0.0), 3)
            self.convergence_migration = round(conv_metrics.get('migration_convergence', 0.0), 3)
            self.behavior_accumulation = round(behav_patterns.get('accumulation', {}).get('strength', 0.0), 3)
            self.behavior_distribution = round(behav_patterns.get('distribution', {}).get('strength', 0.0), 3)
            self.behavior_consolidation = round(behav_patterns.get('consolidation', {}).get('strength', 0.0), 3)
            try:
                if dynamics_result.get('chip_matrix'):
                    cleaned = Calculator.clean_structure(dynamics_result['chip_matrix'], precision=3, threshold=0.001)
                    self.compressed_matrix = zlib.compress(json.dumps(cleaned, separators=(',', ':')).encode('utf-8'))
                if dynamics_result.get('percent_change_matrix'):
                    cleaned = Calculator.clean_structure(dynamics_result['percent_change_matrix'], precision=3, threshold=0.05)
                    self.compressed_change_matrix = zlib.compress(json.dumps(cleaned, separators=(',', ':')).encode('utf-8'))
            except Exception as e:
                print(f"⚠️ 矩阵压缩失败: {e}")
            self.chart_signals = Calculator.clean_structure({'absolute_signals': dynamics_result.get('absolute_change_signals', {}),'behavior_areas': {'accumulation': behav_patterns.get('accumulation', {}).get('areas', []), 'distribution': behav_patterns.get('distribution', {}).get('areas', [])},'key_battle_zones': kbz[:5],'migration_areas': {'convergence': dynamics_result.get('migration_patterns', {}).get('convergence_migration', {}).get('areas', []),'divergence': dynamics_result.get('migration_patterns', {}).get('divergence_migration', {}).get('areas', [])}}, precision=3)
            self.extra_metrics = Calculator.clean_structure({'concentration': conc_metrics,'pressure': press_metrics,'convergence': conv_metrics,'migration': dynamics_result.get('migration_patterns', {}),'behavior_meta': {'accumulation_detected': behav_patterns.get('accumulation', {}).get('detected'),'distribution_detected': behav_patterns.get('distribution', {}).get('detected'),'main_force_activity': behav_patterns.get('main_force_activity')}}, precision=4)
            self.price_grid = Calculator.clean_structure(dynamics_result.get('price_grid', []), precision=3)
            self.validation_score = round(max(0.0, min(1.0, dynamics_result.get('validation_score', 0.5))), 3)
            self.validation_warnings = Calculator.clean_structure(dynamics_result.get('validation_warnings', []), precision=3)
            holding_res = Calculator.calculate_holding_factors(dynamics_result, self.absolute_change_analysis)
            current_factors = {'short': holding_res.get('short_term_ratio', 0.2),'mid': holding_res.get('mid_term_ratio', 0.3),'long': holding_res.get('long_term_ratio', 0.5),'days': holding_res.get('avg_holding_days', 60.0)}
            if tick_factors and self.intraday_chip_quality_score > 0.3:
                s, m, l, d, reason = Calculator.calculate_tick_enhanced_factors(current_factors, tick_factors, self.intraday_chip_quality_score)
                self.short_term_ratio, self.mid_term_ratio, self.long_term_ratio, self.avg_holding_days = s, m, l, d
                if 'extra_metrics' in holding_res:
                    holding_res['extra_metrics']['tick_adjustment_reason'] = reason
            else:
                self.short_term_ratio = current_factors['short']
                self.mid_term_ratio = current_factors['mid']
                self.long_term_ratio = current_factors['long']
                self.avg_holding_days = current_factors['days']
            if self.extra_metrics is None: self.extra_metrics = {}
            if 'extra_metrics' in holding_res:
                self.extra_metrics.update(holding_res['extra_metrics'])
            self.calc_status = 'success'
            self.analysis_method = 'advanced_dynamics_v3_tick_enhanced'
            self.used_percent_data = True
            if not self.used_tick_data and self.intraday_chip_quality_score > 0.3:
                self.used_tick_data = True
            self.save()
            return True
        except Exception as e:
            print(f"❌ [保存动态分析] 失败: {e}")
            import traceback
            traceback.print_exc()
            self.calc_status = 'failed'
            self.error_message = str(e)
            try: self.save(update_fields=['calc_status', 'error_message', 'calc_time'])
            except: pass
            return False

    def to_factor_dict(self) -> Dict[str, Any]:
        """
        转换为筹码因子字典 - 增强版（包含tick数据因子）
        使用 ChipMatrixDynamicsCalculator 辅助计算动态指标
        """
        Calculator = ChipMatrixDynamicsCalculator
        # 提取或构建需要的数据进行即时计算
        kbz_zones = self.chart_signals.get('key_battle_zones', []) if self.chart_signals else []
        # 1. 原有能量场因子
        factors = {
            'absorption_energy': round(self.absorption_energy, 2) if self.absorption_energy is not None else 0.0,
            'distribution_energy': round(self.distribution_energy, 2) if self.distribution_energy is not None else 0.0,
            'net_energy_flow': round(self.net_energy_flow, 2) if self.net_energy_flow is not None else 0.0,
            'game_intensity': round(self.game_intensity, 3) if self.game_intensity is not None else 0.0,
            'breakout_potential': round(self.breakout_potential, 2) if self.breakout_potential is not None else 0.0,
            'energy_concentration': round(self.energy_concentration, 3) if self.energy_concentration is not None else 0.0,
            'fake_distribution_flag': self.fake_distribution_flag if hasattr(self, 'fake_distribution_flag') else False,
            # 使用Calculator计算动态指标
            'key_battle_intensity': Calculator.calculate_key_battle_intensity(kbz_zones),
            'trend_score': Calculator.calculate_trend_score(
                net_flow=self.net_energy_flow or 0,
                game_intensity=self.game_intensity or 0,
                intraday_quality=self.intraday_chip_quality_score or 0,
                tick_flow=self.tick_level_chip_flow if hasattr(self, 'tick_level_chip_flow') else 0
            ),
            'breakout_probability': Calculator.calculate_breakout_probability(
                potential=self.breakout_potential or 0,
                concentration=self.energy_concentration or 0,
                game_intensity=self.game_intensity or 0,
                net_flow=self.net_energy_flow or 0
            ),
            # 扁平化的核心指标
            'concentration_comprehensive': self.concentration_comprehensive if self.concentration_comprehensive is not None else 0.0,
            'concentration_entropy': self.concentration_entropy if self.concentration_entropy is not None else 0.0,
            'concentration_peak': self.concentration_peak if self.concentration_peak is not None else 0.0,
            'pressure_trapped': self.pressure_trapped if self.pressure_trapped is not None else 0.0,
            'pressure_profit': self.pressure_profit if self.pressure_profit is not None else 0.0,
            'support_strength': self.support_strength if self.support_strength is not None else 0.0,
            'resistance_strength': self.resistance_strength if self.resistance_strength is not None else 0.0,
            'convergence_comprehensive': self.convergence_comprehensive if self.convergence_comprehensive is not None else 0.0,
            'convergence_migration': self.convergence_migration if self.convergence_migration is not None else 0.0,
            'behavior_accumulation': self.behavior_accumulation if self.behavior_accumulation is not None else 0.0,
            'behavior_distribution': self.behavior_distribution if self.behavior_distribution is not None else 0.0,
            # 持有时间因子
            'short_term_chip_ratio': self.short_term_ratio if self.short_term_ratio is not None else 0.2,
            'mid_term_chip_ratio': self.mid_term_ratio if self.mid_term_ratio is not None else 0.3,
            'long_term_chip_ratio': self.long_term_ratio if self.long_term_ratio is not None else 0.5,
            'avg_holding_days': self.avg_holding_days if hasattr(self, 'avg_holding_days') else 60.0,
        }
        # 2. Tick数据增强因子
        tick_factors = {
            'intraday_chip_quality_score': self.intraday_chip_quality_score if self.intraday_chip_quality_score is not None else 0.0,
            'intraday_calc_method': self.intraday_calc_method if self.intraday_calc_method else 'daily_only',
            'intraday_main_force_activity': self.intraday_main_force_activity if self.intraday_main_force_activity is not None else 0.0,
            'intraday_accumulation_confidence': self.intraday_accumulation_confidence if self.intraday_accumulation_confidence is not None else 0.0,
            'intraday_distribution_confidence': self.intraday_distribution_confidence if self.intraday_distribution_confidence is not None else 0.0,
        }
        factors.update(tick_factors)
        # 3. 添加交叉验证因子
        if hasattr(self, 'direct_ad_data') and self.direct_ad_data:
            direct_factors = {
                'direct_accumulation_volume': self.direct_ad_data.get('accumulation_volume', 0.0),
                'direct_distribution_volume': self.direct_ad_data.get('distribution_volume', 0.0),
                'direct_net_ad_ratio': self.direct_ad_data.get('net_ad_ratio', 0.0),
            }
            factors.update(direct_factors)
        return factors

# 分表模型
class ChipHoldingMatrix_SZ(ChipHoldingMatrixBase):
    class Meta:
        verbose_name = '筹码持有时间矩阵SZ'
        verbose_name_plural = verbose_name
        db_table = 'stock_chip_holding_matrix_sz'

class ChipHoldingMatrix_SH(ChipHoldingMatrixBase):
    class Meta:
        verbose_name = '筹码持有时间矩阵SH'
        verbose_name_plural = verbose_name
        db_table = 'stock_chip_holding_matrix_sh'

class ChipHoldingMatrix_CY(ChipHoldingMatrixBase):
    class Meta:
        verbose_name = '筹码持有时间矩阵CY'
        verbose_name_plural = verbose_name
        db_table = 'stock_chip_holding_matrix_cy'

class ChipHoldingMatrix_KC(ChipHoldingMatrixBase):
    class Meta:
        verbose_name = '筹码持有时间矩阵KC'
        verbose_name_plural = verbose_name
        db_table = 'stock_chip_holding_matrix_kc'

class ChipHoldingMatrix_BJ(ChipHoldingMatrixBase):
    class Meta:
        verbose_name = '筹码持有时间矩阵BJ'
        verbose_name_plural = verbose_name
        db_table = 'stock_chip_holding_matrix_bj'

