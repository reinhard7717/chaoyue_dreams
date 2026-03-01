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
    def calculate_convergence_divergence(cls, chip_dynamics_result: Dict[str, any]) -> Dict[str, float]:
        """
        [Version 23.0.0] 筹码聚散网络推断算子 (贝叶斯联合概率融合版)
        说明：消灭简单的加法除以均值的暴力平均。
        引入联合概率分布(Probabilistic OR)计算多维度信号（主力/派发/迁徙）共振的 behavior_confirmation。禁止使用空行。
        """
        import math
        try:
            if not chip_dynamics_result or chip_dynamics_result.get('analysis_status') != 'success': return cls._get_default_convergence_factors()
            convergence_metrics = chip_dynamics_result.get('convergence_metrics', {})
            behavior_patterns = chip_dynamics_result.get('behavior_patterns', {})
            migration_patterns = chip_dynamics_result.get('migration_patterns', {})
            absolute_signals = chip_dynamics_result.get('absolute_change_signals', {})
            factors = {'percent_change_convergence': float(convergence_metrics.get('comprehensive_convergence', 0.5)), 'percent_change_divergence': float(1.0 - convergence_metrics.get('comprehensive_convergence', 0.5)), 'convergence_strength': float(convergence_metrics.get('convergence_strength', 0.0)), 'divergence_strength': float(convergence_metrics.get('divergence_strength', 0.0))}
            factors['net_migration_direction'] = float(migration_patterns.get('net_migration_direction', 0.0))
            factors['migration_convergence_ratio'] = float(convergence_metrics.get('migration_convergence', 0.5))
            increase_areas = absolute_signals.get('significant_increase_areas', [])
            decrease_areas = absolute_signals.get('significant_decrease_areas', [])
            total_abs_change = sum(abs(area.get('change', 0.0)) for area in increase_areas) + sum(abs(area.get('change', 0.0)) for area in decrease_areas)
            factors['absolute_change_strength'] = float(math.tanh(total_abs_change / 15.0))
            accumulation = behavior_patterns.get('accumulation', {})
            distribution = behavior_patterns.get('distribution', {})
            factors['accumulation_signal_score'] = float(accumulation.get('strength', 0.0))
            factors['distribution_signal_score'] = float(distribution.get('strength', 0.0))
            factors['main_force_activity_index'] = float(behavior_patterns.get('main_force_activity', 0.0))
            factors['signal_quality_score'] = float(absolute_signals.get('signal_quality', 0.0))
            p_accum = factors['accumulation_signal_score']
            p_dist = factors['distribution_signal_score']
            p_mig = abs(factors['net_migration_direction'])
            p_main = factors['main_force_activity_index']
            factors['behavior_confirmation'] = float(1.0 - (1.0 - max(p_accum, p_dist)) * (1.0 - p_mig) * (1.0 - p_main * 0.5))
            pressure_metrics = chip_dynamics_result.get('pressure_metrics', {})
            factors['pressure_release_index'] = float(pressure_metrics.get('pressure_release', 0.0))
            support = float(pressure_metrics.get('support_strength', 0.3))
            resistance = float(pressure_metrics.get('resistance_strength', 0.3))
            prior_sr = 0.05
            raw_sr_ratio = (support + prior_sr) / (resistance + prior_sr)
            factors['support_resistance_ratio'] = float(math.atan(raw_sr_ratio) / (math.pi / 2) * 2.0)
            return factors
        except Exception as e:
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
        [Version 35.0.0] 全息状态映射神经中枢 (Numpy布尔跨界崩溃防御版)
        说明：消除最后的信息孤岛，并彻底剿灭 Numpy.bool_ 注入 Django ORM BooleanField 时引发的隐性底层序列化暴毙。
        全量、安全地反射所有特征，保证数据血脉畅通。禁止使用空行。
        """
        import numpy as np
        import math
        def safe_flt(val, default=0.0):
            try:
                f_val = float(val)
                return default if math.isnan(f_val) or math.isinf(f_val) else f_val
            except (TypeError, ValueError): return default
        def safe_int(val, default=0):
            try: return int(float(val))
            except (TypeError, ValueError): return default
        def safe_bool(val, default=False):
            try: return bool(val)
            except Exception: return default
        try:
            if not chip_dynamics_result or chip_dynamics_result.get('analysis_status') != 'success': return False
            if 'current_price' in chip_dynamics_result: self.close = safe_flt(chip_dynamics_result['current_price'])
            conv_factors = self.calculate_convergence_divergence(chip_dynamics_result)
            for k, v in conv_factors.items():
                if hasattr(self, k): setattr(self, k, safe_flt(v))
            conc_metrics = chip_dynamics_result.get('concentration_metrics', {})
            if conc_metrics:
                for field in ['chip_concentration_ratio', 'chip_entropy', 'chip_skewness', 'chip_kurtosis', 'chip_mean', 'chip_std', 'high_position_lock_ratio_90', 'main_cost_range_ratio', 'cost_5pct', 'cost_15pct', 'cost_50pct', 'cost_85pct', 'cost_95pct', 'weight_avg_cost', 'winner_rate', 'win_rate_price_position', 'price_to_weight_avg_ratio', 'price_percentile_position', 'chip_convergence_ratio', 'chip_divergence_ratio', 'his_low', 'his_high']:
                    if hasattr(self, field) and field in conc_metrics: setattr(self, field, safe_flt(conc_metrics[field]))
                if hasattr(self, 'chip_stability'): self.chip_stability = safe_flt(conc_metrics.get('chip_stability', 0.5))
                if hasattr(self, 'profit_ratio'): self.profit_ratio = safe_flt(conc_metrics.get('winner_rate', 0.5))
            press_metrics = chip_dynamics_result.get('pressure_metrics', {})
            if press_metrics:
                if hasattr(self, 'profit_pressure'): self.profit_pressure = safe_flt(press_metrics.get('profit_pressure', 0.5))
                if hasattr(self, 'pressure_release_index'): self.pressure_release_index = safe_flt(press_metrics.get('pressure_release', 0.0))
                if hasattr(self, 'support_resistance_ratio'):
                    sup = safe_flt(press_metrics.get('support_strength', 0.0))
                    res = safe_flt(press_metrics.get('resistance_strength', 0.0))
                    self.support_resistance_ratio = safe_flt(math.atan((sup + 0.05)/(res + 0.05)) / (math.pi / 2) * 2.0)
            mig_patterns = chip_dynamics_result.get('migration_patterns', {})
            if mig_patterns:
                if hasattr(self, 'chip_flow_direction'): self.chip_flow_direction = safe_int(mig_patterns.get('chip_flow_direction', 0))
                if hasattr(self, 'chip_flow_intensity'): self.chip_flow_intensity = safe_flt(mig_patterns.get('chip_flow_intensity', 0.0))
            behav_patterns = chip_dynamics_result.get('behavior_patterns', {})
            if behav_patterns and hasattr(self, 'chip_structure_state'):
                accum = safe_flt(behav_patterns.get('accumulation', {}).get('strength', 0.0))
                distrib = safe_flt(behav_patterns.get('distribution', {}).get('strength', 0.0))
                consol_detected = safe_bool(behav_patterns.get('consolidation', {}).get('detected', False))
                breakout_detected = safe_bool(behav_patterns.get('breakout_preparation', {}).get('detected', False))
                if accum > 0.2 and accum > distrib * 1.3: self.chip_structure_state = 'accumulation'
                elif distrib > 0.2 and distrib > accum * 1.3: self.chip_structure_state = 'distribution'
                elif breakout_detected: self.chip_structure_state = 'lifting'
                elif consol_detected: self.chip_structure_state = 'consolidation'
                else: self.chip_structure_state = 'consolidation'
            morph_metrics = chip_dynamics_result.get('morphology_metrics', {})
            if morph_metrics:
                for field in ['peak_count', 'main_peak_position', 'peak_distance_ratio', 'peak_concentration', 'is_double_peak', 'is_multi_peak']:
                    if hasattr(self, field) and field in morph_metrics:
                        if field in ['is_double_peak', 'is_multi_peak']: setattr(self, field, safe_bool(morph_metrics[field]))
                        elif field in ['peak_count', 'main_peak_position']: setattr(self, field, safe_int(morph_metrics[field]))
                        else: setattr(self, field, safe_flt(morph_metrics[field]))
            tech_metrics = chip_dynamics_result.get('technical_metrics', {})
            if tech_metrics:
                for field in ['price_to_ma5_ratio', 'price_to_ma21_ratio', 'price_to_ma34_ratio', 'price_to_ma55_ratio', 'chip_cost_to_ma21_diff', 'volatility_adjusted_concentration', 'chip_rsi_divergence', 'peak_migration_speed_5d', 'chip_stability_change_5d', 'trend_confirmation_score', 'reversal_warning_score', 'turnover_rate', 'volume_ratio']:
                    if hasattr(self, field) and field in tech_metrics: setattr(self, field, safe_flt(tech_metrics[field]))
                if hasattr(self, 'ma_arrangement_status') and 'ma_arrangement_status' in tech_metrics: self.ma_arrangement_status = safe_int(tech_metrics['ma_arrangement_status'])
            hold_metrics = chip_dynamics_result.get('holding_metrics', {})
            if hold_metrics:
                for field in ['short_term_chip_ratio', 'mid_term_chip_ratio', 'long_term_chip_ratio', 'avg_holding_days']:
                    if hasattr(self, field) and field in hold_metrics: setattr(self, field, safe_flt(hold_metrics[field]))
            tick_factors = chip_dynamics_result.get('tick_enhanced_factors', {})
            if tick_factors:
                for field in ['intraday_chip_concentration', 'intraday_chip_entropy', 'intraday_price_distribution_skewness', 'intraday_price_range_ratio', 'intraday_chip_turnover_intensity', 'tick_level_chip_flow', 'intraday_low_lock_ratio', 'intraday_high_lock_ratio', 'intraday_cost_center_migration', 'intraday_cost_center_volatility', 'intraday_peak_valley_ratio', 'intraday_trough_filling_degree', 'tick_abnormal_volume_ratio', 'tick_clustering_index', 'tick_chip_transfer_efficiency', 'intraday_chip_consolidation_degree', 'intraday_chip_game_index', 'tick_chip_balance_ratio', 'tick_data_quality_score']:
                    if hasattr(self, field) and field in tick_factors: setattr(self, field, safe_flt(tick_factors[field]))
                for field in ['intraday_support_test_count', 'intraday_resistance_test_count']:
                    if hasattr(self, field) and field in tick_factors: setattr(self, field, safe_int(tick_factors[field]))
                if hasattr(self, 'intraday_factor_calc_method') and 'intraday_factor_calc_method' in tick_factors: self.intraday_factor_calc_method = str(tick_factors['intraday_factor_calc_method'])
            self._calculate_trend_score(chip_dynamics_result)
            self.calc_status = 'success'
            return True
        except Exception as e:
            import traceback
            traceback.print_exc()
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
        [Version 35.0.0] 扁平化数据解包提取器（Numpy布尔跨界崩溃防御版）
        说明：追加 safe_bool 防火墙，防范 numpy.bool_ 泄露到上层引发 DB Driver 暴毙。全景透传提取高级矩阵特征。禁止使用空行。
        """
        import math
        from typing import Dict, Any
        from services.chip_matrix_dynamics_calculator import ChipMatrixDynamicsCalculator as Calculator
        def safe_flt(val, default=0.0):
            try:
                f_val = float(val)
                return default if math.isnan(f_val) or math.isinf(f_val) else f_val
            except (TypeError, ValueError): return default
        def safe_int(val, default=0):
            try: return int(float(val))
            except (TypeError, ValueError): return default
        def safe_bool(val, default=False):
            try: return bool(val)
            except Exception: return default
        kbz_zones = self.chart_signals.get('key_battle_zones', []) if self.chart_signals else []
        factors = {
            'absorption_energy': safe_flt(self.absorption_energy), 'distribution_energy': safe_flt(self.distribution_energy), 'net_energy_flow': safe_flt(self.net_energy_flow),
            'game_intensity': safe_flt(self.game_intensity), 'breakout_potential': safe_flt(self.breakout_potential), 'energy_concentration': safe_flt(self.energy_concentration),
            'fake_distribution_flag': safe_bool(getattr(self, 'fake_distribution_flag', False)),
            'key_battle_intensity': safe_flt(Calculator.calculate_key_battle_intensity(kbz_zones)),
            'trend_score': safe_flt(Calculator.calculate_trend_score(net_flow=self.net_energy_flow or 0, game_intensity=self.game_intensity or 0, intraday_quality=self.intraday_chip_quality_score or 0, tick_flow=getattr(self, 'tick_level_chip_flow', 0))),
            'breakout_probability': safe_flt(Calculator.calculate_breakout_probability(potential=self.breakout_potential or 0, concentration=self.energy_concentration or 0, game_intensity=self.game_intensity or 0, net_flow=self.net_energy_flow or 0)),
            'concentration_comprehensive': safe_flt(self.concentration_comprehensive), 'concentration_entropy': safe_flt(self.concentration_entropy), 'concentration_peak': safe_flt(self.concentration_peak),
            'pressure_trapped': safe_flt(self.pressure_trapped), 'pressure_profit': safe_flt(self.pressure_profit), 'support_strength': safe_flt(self.support_strength), 'resistance_strength': safe_flt(self.resistance_strength),
            'convergence_comprehensive': safe_flt(self.convergence_comprehensive), 'convergence_migration': safe_flt(self.convergence_migration), 'behavior_accumulation': safe_flt(self.behavior_accumulation), 'behavior_distribution': safe_flt(self.behavior_distribution),
            'short_term_chip_ratio': safe_flt(self.short_term_ratio, 0.2), 'mid_term_chip_ratio': safe_flt(self.mid_term_ratio, 0.3), 'long_term_chip_ratio': safe_flt(self.long_term_ratio, 0.5), 'avg_holding_days': safe_flt(getattr(self, 'avg_holding_days', 60.0), 60.0),
        }
        tick_factors = {'intraday_chip_quality_score': safe_flt(self.intraday_chip_quality_score), 'intraday_calc_method': getattr(self, 'intraday_calc_method', 'daily_only'), 'intraday_main_force_activity': safe_flt(self.intraday_main_force_activity), 'intraday_accumulation_confidence': safe_flt(self.intraday_accumulation_confidence), 'intraday_distribution_confidence': safe_flt(self.intraday_distribution_confidence)}
        factors.update(tick_factors)
        if hasattr(self, 'direct_ad_data') and self.direct_ad_data: factors.update({'direct_accumulation_volume': safe_flt(self.direct_ad_data.get('accumulation_volume', 0.0)), 'direct_distribution_volume': safe_flt(self.direct_ad_data.get('distribution_volume', 0.0)), 'direct_net_ad_ratio': safe_flt(self.direct_ad_data.get('net_ad_ratio', 0.0))})
        if self.extra_metrics:
            conc = self.extra_metrics.get('concentration', {})
            factors['chip_mean'] = safe_flt(conc.get('chip_mean', 0.0)); factors['chip_std'] = safe_flt(conc.get('chip_std', 0.0)); factors['chip_skewness'] = safe_flt(conc.get('chip_skewness', 0.0)); factors['chip_kurtosis'] = safe_flt(conc.get('chip_kurtosis', 0.0))
            factors['high_position_lock_ratio_90'] = safe_flt(conc.get('high_position_lock_ratio_90', 0.0)); factors['main_cost_range_ratio'] = safe_flt(conc.get('main_cost_range_ratio', 0.0)); factors['chip_stability'] = safe_flt(conc.get('chip_stability', 0.5))
            factors['weight_avg_cost'] = safe_flt(conc.get('weight_avg_cost', 0.0)); factors['cost_5pct'] = safe_flt(conc.get('cost_5pct', 0.0)); factors['cost_15pct'] = safe_flt(conc.get('cost_15pct', 0.0)); factors['cost_50pct'] = safe_flt(conc.get('cost_50pct', 0.0)); factors['cost_85pct'] = safe_flt(conc.get('cost_85pct', 0.0)); factors['cost_95pct'] = safe_flt(conc.get('cost_95pct', 0.0))
            factors['winner_rate'] = safe_flt(conc.get('winner_rate', 0.0)); factors['win_rate_price_position'] = safe_flt(conc.get('win_rate_price_position', 0.0)); factors['price_to_weight_avg_ratio'] = safe_flt(conc.get('price_to_weight_avg_ratio', 0.0)); factors['chip_concentration_ratio'] = safe_flt(conc.get('chip_concentration_ratio', 0.0)); factors['price_percentile_position'] = safe_flt(conc.get('price_percentile_position', 0.0))
            factors['chip_entropy'] = safe_flt(conc.get('chip_entropy', 0.0)); factors['chip_convergence_ratio'] = safe_flt(conc.get('chip_convergence_ratio', 0.0)); factors['chip_divergence_ratio'] = safe_flt(conc.get('chip_divergence_ratio', 0.0)); factors['his_low'] = safe_flt(conc.get('his_low', 0.0)); factors['his_high'] = safe_flt(conc.get('his_high', 0.0))
            factors['profit_ratio'] = safe_flt(conc.get('winner_rate', 0.0))
            morph = self.extra_metrics.get('morphology', {})
            factors['peak_count'] = safe_int(morph.get('peak_count', 0)); factors['main_peak_position'] = safe_int(morph.get('main_peak_position', 0)); factors['peak_distance_ratio'] = safe_flt(morph.get('peak_distance_ratio', 0.0)); factors['peak_concentration'] = safe_flt(morph.get('peak_concentration', 0.0))
            factors['is_double_peak'] = safe_bool(morph.get('is_double_peak', False)); factors['is_multi_peak'] = safe_bool(morph.get('is_multi_peak', False))
            mig = self.extra_metrics.get('migration', {})
            factors['chip_flow_direction'] = safe_int(mig.get('chip_flow_direction', 0)); factors['chip_flow_intensity'] = safe_flt(mig.get('chip_flow_intensity', 0.0)); factors['net_migration_direction'] = safe_flt(mig.get('net_migration_direction', 0.0)); factors['migration_convergence_ratio'] = safe_flt(mig.get('migration_convergence', 0.5))
            press = self.extra_metrics.get('pressure', {})
            factors['profit_pressure'] = safe_flt(press.get('profit_pressure', 0.0)); factors['pressure_release_index'] = safe_flt(press.get('pressure_release', 0.0))
            sup = safe_flt(press.get('support_strength', 0.0)); res = safe_flt(press.get('resistance_strength', 0.0))
            factors['support_resistance_ratio'] = safe_flt(math.atan((sup + 0.05)/(res + 0.05)) / (math.pi / 2) * 2.0)
            tech = self.extra_metrics.get('technical', {})
            for tf in ['price_to_ma5_ratio', 'price_to_ma21_ratio', 'price_to_ma34_ratio', 'price_to_ma55_ratio', 'chip_cost_to_ma21_diff', 'volatility_adjusted_concentration', 'chip_rsi_divergence', 'peak_migration_speed_5d', 'chip_stability_change_5d', 'trend_confirmation_score', 'reversal_warning_score', 'turnover_rate', 'volume_ratio']: factors[tf] = safe_flt(tech.get(tf, 0.0))
            factors['ma_arrangement_status'] = safe_int(tech.get('ma_arrangement_status', 0))
            hold = self.extra_metrics.get('holding', {})
            for hf in ['short_term_chip_ratio', 'mid_term_chip_ratio', 'long_term_chip_ratio', 'avg_holding_days']: factors[hf] = safe_flt(hold.get(hf, factors.get(hf, 0.0)))
            behav_meta = self.extra_metrics.get('behavior_meta', {})
            factors['main_force_activity_index'] = safe_flt(behav_meta.get('main_force_activity', 0.0))
            behav_acc = safe_flt(self.behavior_accumulation); behav_dist = safe_flt(self.behavior_distribution)
            brk_out = safe_bool(behav_meta.get('breakout_detected', False)); cons_det = safe_bool(behav_meta.get('consolidation_detected', False))
            if behav_acc > 0.2 and behav_acc > behav_dist * 1.3: factors['chip_structure_state'] = 'accumulation'
            elif behav_dist > 0.2 and behav_dist > behav_acc * 1.3: factors['chip_structure_state'] = 'distribution'
            elif brk_out: factors['chip_structure_state'] = 'lifting'
            elif cons_det: factors['chip_structure_state'] = 'consolidation'
            else: factors['chip_structure_state'] = 'consolidation'
        else:
            factors['chip_structure_state'] = 'consolidation'; factors['absolute_change_strength'] = 0.0; factors['signal_quality_score'] = 0.0
        if self.chart_signals:
            abs_sigs = self.chart_signals.get('absolute_signals', {})
            factors['signal_quality_score'] = safe_flt(abs_sigs.get('signal_quality', 0.0))
            total_inc = sum(abs(a.get('change', 0)) for a in abs_sigs.get('significant_increase_areas', []))
            total_dec = sum(abs(a.get('change', 0)) for a in abs_sigs.get('significant_decrease_areas', []))
            factors['absolute_change_strength'] = math.tanh((total_inc + total_dec) / 15.0)
        behav_acc = safe_flt(self.behavior_accumulation); behav_dist = safe_flt(self.behavior_distribution)
        factors['accumulation_signal_score'] = behav_acc; factors['distribution_signal_score'] = behav_dist
        factors['percent_change_convergence'] = safe_flt(self.convergence_comprehensive)
        factors['percent_change_divergence'] = safe_flt(1.0 - factors['percent_change_convergence'])
        p_accum = behav_acc; p_dist = behav_dist; p_mig = abs(factors.get('net_migration_direction', 0.0)); p_main = factors.get('main_force_activity_index', 0.0)
        factors['behavior_confirmation'] = safe_flt(1.0 - (1.0 - max(p_accum, p_dist)) * (1.0 - p_mig) * (1.0 - p_main * 0.5))
        from services.chip_holding_calculator import QuantitativeTelemetryProbe
        QuantitativeTelemetryProbe.emit("ChipHoldingMatrixBase", "to_factor_dict", {'chart_signals_present': bool(self.chart_signals), 'extra_metrics_present': bool(self.extra_metrics)}, {'abs_strength': float(factors.get('absolute_change_strength', 0.0)), 'behavior_confirmation': factors['behavior_confirmation']}, {'exported_keys': len(factors)})
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

