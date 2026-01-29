# stock_models\chip_factors.py
from django.db import models
from django.utils.translation import gettext_lazy as _
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import linregress
from typing import List, Tuple, Dict, Optional, Any
import json
import base64
import pickle

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
    
    def _calculate_trend_score(self, chip_dynamics_result: Dict[str, any]):
        """
        计算综合趋势得分
        考虑因素：
        1. 筹码聚散方向与价格趋势的一致性
        2. 主力行为强度
        3. 支撑阻力比
        4. 信号质量
        """
        try:
            # 获取相关数据
            convergence_factors = self.calculate_convergence_divergence(chip_dynamics_result)
            # 1. 趋势方向得分
            direction_score = 0.0
            net_migration = convergence_factors.get('net_migration_direction', 0.0)
            if net_migration > 0.1:  # 向上迁移
                direction_score = min(1.0, net_migration / 5.0)
            elif net_migration < -0.1:  # 向下迁移
                direction_score = -min(1.0, abs(net_migration) / 5.0)
            # 2. 主力行为得分
            main_force_score = convergence_factors.get('main_force_activity_index', 0.0)
            # 3. 支撑强度得分
            support_ratio = convergence_factors.get('support_resistance_ratio', 1.0)
            support_score = min(1.0, support_ratio) if support_ratio > 1 else support_ratio
            # 4. 信号质量得分
            signal_score = convergence_factors.get('signal_quality_score', 0.0)
            # 5. 行为确认得分
            confirmation_score = convergence_factors.get('behavior_confirmation', 0.0)
            # 综合趋势得分
            weights = [0.3, 0.25, 0.2, 0.15, 0.1]
            scores = [
                (direction_score + 1) / 2,  # 归一化到0-1
                main_force_score,
                (support_score + 1) / 2,
                signal_score,
                confirmation_score
            ]
            trend_score = np.sum(np.array(weights) * np.array(scores))
            # 趋势确认信号
            self.trend_confirmation_score = trend_score
            # 反转预警（趋势得分低但主力活跃）
            if trend_score < 0.4 and main_force_score > 0.6:
                self.reversal_warning_score = 1.0 - trend_score
            else:
                self.reversal_warning_score = 0.0
                
        except Exception as e:
            print(f"⚠️ 计算趋势得分失败: {e}")
            self.trend_confirmation_score = 0.5
            self.reversal_warning_score = 0.0

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
    # ========== 新增：动态分析结果 ==========
    # 价格网格
    price_grid = models.JSONField(
        verbose_name='价格网格',
        null=True, blank=True,
        help_text='分析使用的价格网格点'
    )
    # 百分比变化矩阵
    percent_change_matrix = models.JSONField(
        verbose_name='百分比变化矩阵',
        null=True, blank=True,
        help_text='筹码百分比变化矩阵'
    )
    # 绝对变化信号
    absolute_change_signals = models.JSONField(
        verbose_name='绝对变化信号',
        null=True, blank=True,
        help_text='基于绝对变化的信号识别结果'
    )
    # 集中度指标
    concentration_metrics = models.JSONField(
        verbose_name='集中度指标',
        null=True, blank=True
    )
    # 压力指标
    pressure_metrics = models.JSONField(
        verbose_name='压力指标',
        null=True, blank=True
    )
    # 行为模式
    behavior_patterns = models.JSONField(
        verbose_name='行为模式',
        null=True, blank=True
    )
    # 迁移模式
    migration_patterns = models.JSONField(
        verbose_name='迁移模式',
        null=True, blank=True
    )
    # 聚散度指标
    convergence_metrics = models.JSONField(
        verbose_name='聚散度指标',
        null=True, blank=True
    )
    # 压缩矩阵（二进制存储）
    compressed_matrix = models.BinaryField(
        verbose_name='压缩矩阵',
        null=True, blank=True
    )
    # ========== 新增：博弈能量场因子 ==========
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
    
    # ========== 关键博弈区域（JSON存储） ==========
    key_battle_zones = models.JSONField(
        verbose_name='关键博弈区域',
        null=True, blank=True,
        help_text='当前主要的筹码对抗区域'
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
            models.Index(fields=['stock', 'trade_time']),
            models.Index(fields=['trade_time', 'calc_status']),
            models.Index(fields=['short_term_ratio']),
            models.Index(fields=['long_term_ratio']),
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

    def to_factor_dict(self) -> Dict[str, Any]:
        """
        转换为筹码因子字典
        重构：移除avg_holding_days，加入能量场因子
        """
        # 1. 能量场因子（核心）
        factors = {
            # 能量场因子
            'absorption_energy': round(self.absorption_energy, 2) if self.absorption_energy is not None else 0.0,
            'distribution_energy': round(self.distribution_energy, 2) if self.distribution_energy is not None else 0.0,
            'net_energy_flow': round(self.net_energy_flow, 2) if self.net_energy_flow is not None else 0.0,
            'game_intensity': round(self.game_intensity, 3) if self.game_intensity is not None else 0.0,
            'breakout_potential': round(self.breakout_potential, 2) if self.breakout_potential is not None else 0.0,
            'energy_concentration': round(self.energy_concentration, 3) if self.energy_concentration is not None else 0.0,
            'fake_distribution_flag': self.fake_distribution_flag,
            # 关键博弈区域强度
            'key_battle_intensity': self._calculate_key_battle_intensity(),
            # 趋势得分（基于能量场）
            'trend_score': self._calculate_trend_score(),
            # 突破概率
            'breakout_probability': self._calculate_breakout_probability(),
        }
        # 2. 保留原有的部分重要因子（可选）
        if hasattr(self, 'short_term_ratio'):
            factors['short_term_chip_ratio'] = self.short_term_ratio if self.short_term_ratio is not None else 0.2
            factors['long_term_chip_ratio'] = self.long_term_ratio if self.long_term_ratio is not None else 0.5
        # 3. 添加交叉验证因子（来自direct_ad）
        if hasattr(self, 'direct_ad_data') and self.direct_ad_data:
            direct_factors = {
                'direct_accumulation_volume': self.direct_ad_data.get('accumulation_volume', 0.0),
                'direct_distribution_volume': self.direct_ad_data.get('distribution_volume', 0.0),
                'direct_net_ad_ratio': self.direct_ad_data.get('net_ad_ratio', 0.0),
            }
            factors.update(direct_factors)
        return factors

    def save_dynamics_result(self, dynamics_result: Dict[str, Any]):
        """
        保存动态分析结果 - 确保所有字段都有值
        """
        import math
        try:
            # 1. 基础状态检查
            if not dynamics_result or dynamics_result.get('analysis_status') != 'success':
                self.calc_status = 'failed'
                self.save(update_fields=['calc_status', 'error_message', 'calc_time'])
                print(f"❌ [保存] 动态分析状态失败: {dynamics_result.get('analysis_status', 'unknown')}")
                return False
            # =======================================================
            # 2. 确保所有必要字段都存在
            # =======================================================
            required_fields = ['price_grid', 'percent_change_matrix', 'current_price']
            for field in required_fields:
                if field not in dynamics_result:
                    print(f"❌ [保存] 缺少必要字段: {field}")
                    self.calc_status = 'failed'
                    self.error_message = f"缺少必要字段: {field}"
                    self.save(update_fields=['calc_status', 'error_message', 'calc_time'])
                    return False
            # =======================================================
            # 3. 内部辅助函数：数据清洗（保留3位小数）
            # =======================================================
            def _clean_structure(data, precision=3, threshold=0.0):
                """递归清洗数据结构中的所有浮点数，保留指定精度"""
                if isinstance(data, (float, int, np.number)):
                    try:
                        val = float(data)
                        if math.isnan(val) or math.isinf(val):
                            return 0.0
                        if abs(val) < threshold: 
                            return 0.0
                        if val == 0.0: 
                            return 0.0
                        return round(val, precision)
                    except Exception:
                        return 0.0
                elif isinstance(data, dict):
                    return {k: _clean_structure(v, precision, threshold) for k, v in data.items()}
                elif isinstance(data, (list, tuple, np.ndarray)):
                    if isinstance(data, np.ndarray):
                        data = data.tolist()
                    return [_clean_structure(i, precision, threshold) for i in data]
                return data
            # =======================================================
            # 4. 计算并保存absolute_change_analysis（强制计算，保留3位小数）
            # =======================================================
            try:
                percent_change_matrix = dynamics_result.get('percent_change_matrix', [])
                price_grid = dynamics_result.get('price_grid', [])
                current_price = dynamics_result.get('current_price', 0)
                if len(percent_change_matrix) > 0 and len(price_grid) > 0 and current_price > 0:
                    # 使用最新的变化数据
                    latest_change = np.array(percent_change_matrix[-1]) if percent_change_matrix else np.zeros(len(price_grid))
                    price_grid_array = np.array(price_grid)
                    # 计算绝对变化分析
                    absolute_analysis = self._calculate_absolute_change_analysis_robust(
                        latest_change, 
                        price_grid_array, 
                        current_price
                    )
                    # 确保分析结果不为空，并清洗为3位小数
                    if not absolute_analysis:
                        absolute_analysis = self._get_default_absolute_analysis()
                    # 清洗absolute_change_analysis中的所有浮点数，保留3位小数
                    self.absolute_change_analysis = _clean_structure(absolute_analysis, precision=3)
                else:
                    print(f"⚠️ [保存] 数据不足，无法计算absolute_change_analysis")
                    self.absolute_change_analysis = _clean_structure(self._get_default_absolute_analysis(), precision=3)
            except Exception as e:
                print(f"⚠️ [保存] 计算absolute_change_analysis失败: {e}")
                self.absolute_change_analysis = _clean_structure(self._get_default_absolute_analysis(), precision=3)
            # =======================================================
            # 5. 保存能量场数据
            # =======================================================
            game_energy = dynamics_result.get('game_energy_result', {})
            if game_energy:
                # 确保能量场字段有值，并保留3位小数
                self.absorption_energy = round(max(0.0, game_energy.get('absorption_energy', 0.0)), 3)
                self.distribution_energy = round(max(0.0, game_energy.get('distribution_energy', 0.0)), 3)
                self.net_energy_flow = round(game_energy.get('net_energy_flow', 0.0), 3)
                self.game_intensity = round(max(0.0, min(1.0, game_energy.get('game_intensity', 0.0))), 3)
                self.breakout_potential = round(max(0.0, game_energy.get('breakout_potential', 0.0)), 3)
                self.energy_concentration = round(max(0.0, min(1.0, game_energy.get('energy_concentration', 0.0))), 3)
                self.fake_distribution_flag = bool(game_energy.get('fake_distribution_flag', False))
                # 处理key_battle_zones，清洗为3位小数
                key_battle_zones = game_energy.get('key_battle_zones', [])
                if key_battle_zones and len(key_battle_zones) > 0:
                    # 清洗key_battle_zones中的所有浮点数，保留3位小数
                    cleaned_zones = _clean_structure(key_battle_zones, precision=3)
                    self.key_battle_zones = cleaned_zones[:5]  # 只保存前5个
                else:
                    # 如果没有关键区域，创建一个默认的，并清洗为3位小数
                    default_zones = self._create_default_key_battle_zones(
                        dynamics_result.get('price_grid', []),
                        dynamics_result.get('current_price', 0)
                    )
                    self.key_battle_zones = _clean_structure(default_zones, precision=3)
            else:
                print(f"⚠️ [保存] 没有game_energy_result数据")
                # 设置默认值，并保留3位小数
                self._set_default_energy_values()
            # =======================================================
            # 6. 保存其他动态分析结果（使用清洗函数）
            # =======================================================
            # 价格网格
            self.price_grid = _clean_structure(dynamics_result.get('price_grid', []), precision=3)
            # 百分比变化矩阵
            raw_change = dynamics_result.get('percent_change_matrix', [])
            self.percent_change_matrix = _clean_structure(raw_change, precision=3, threshold=0.05) if raw_change else []
            # 信号与模式
            self.absolute_change_signals = _clean_structure(dynamics_result.get('absolute_change_signals', {}), precision=3)
            self.behavior_patterns = _clean_structure(dynamics_result.get('behavior_patterns', {}), precision=3)
            # 指标类
            self.concentration_metrics = _clean_structure(dynamics_result.get('concentration_metrics', {}), precision=4)
            self.pressure_metrics = _clean_structure(dynamics_result.get('pressure_metrics', {}), precision=4)
            self.migration_patterns = _clean_structure(dynamics_result.get('migration_patterns', {}), precision=4)
            self.convergence_metrics = _clean_structure(dynamics_result.get('convergence_metrics', {}), precision=4)
            # 验证信息
            self.validation_score = round(max(0.0, min(1.0, dynamics_result.get('validation_score', 0.5))), 3)
            self.validation_warnings = _clean_structure(dynamics_result.get('validation_warnings', []), precision=3)
            # =======================================================
            # 7. 筹码矩阵存储优化
            # =======================================================
            chip_matrix_list = dynamics_result.get('chip_matrix', [])
            if chip_matrix_list:
                try:
                    cleaned_matrix = _clean_structure(chip_matrix_list, precision=3, threshold=0.001)
                    self.matrix_data = {'matrix': cleaned_matrix}
                    import zlib, json
                    json_str = json.dumps(cleaned_matrix, separators=(',', ':'))
                    self.compressed_matrix = zlib.compress(json_str.encode('utf-8'))
                except Exception as e:
                    print(f"⚠️ [保存警告] 筹码矩阵压缩失败: {e}")
                    self.matrix_data = {'matrix': chip_matrix_list}
            # 8. 计算持有时间因子
            self._calculate_holding_factors_from_dynamics(dynamics_result)
            # 9. 保存状态与提交
            self.calc_status = 'success'
            self.analysis_method = 'advanced_dynamics_v2'
            self.used_percent_data = True
            # 保存所有字段
            self.save()
            return True
        except Exception as e:
            print(f"❌ [保存动态分析] 失败: {e}")
            import traceback
            traceback.print_exc()
            self.calc_status = 'failed'
            self.error_message = str(e)
            try: 
                self.save(update_fields=['calc_status', 'error_message', 'calc_time'])
            except: 
                pass
            return False

    def _calculate_absolute_change_analysis_robust(self, changes: np.ndarray, price_grid: np.ndarray, current_price: float) -> Dict[str, Any]:
        """健壮版的绝对变化分析"""
        try:
            if len(changes) == 0 or len(price_grid) == 0 or current_price <= 0:
                print(f"⚠️ [绝对变化分析] 输入数据无效")
                return self._get_default_absolute_analysis()
            # 基础统计分析
            absolute_analysis = {
                'total_change_volume': float(np.sum(np.abs(changes))),
                'positive_change_volume': float(np.sum(changes[changes > 0])),
                'negative_change_volume': float(np.sum(changes[changes < 0])),
                'max_increase': float(np.max(changes)) if len(changes) > 0 else 0.0,
                'max_decrease': float(np.min(changes)) if len(changes) > 0 else 0.0,
                'mean_change': float(np.mean(changes)) if len(changes) > 0 else 0.0,
            }
            # 变化集中度
            if absolute_analysis['total_change_volume'] > 0:
                abs_changes = np.abs(changes)
                top_indices = np.argsort(abs_changes)[-5:]
                top_volume = np.sum(abs_changes[top_indices])
                absolute_analysis['change_concentration'] = top_volume / absolute_analysis['total_change_volume']
            else:
                absolute_analysis['change_concentration'] = 0.0
            # 价格分层分析
            price_rel = (price_grid - current_price) / current_price
            price_zones = {
                'deep_below': (price_rel < -0.15),
                'below': ((price_rel >= -0.15) & (price_rel < -0.05)),
                'near': ((price_rel >= -0.05) & (price_rel <= 0.05)),
                'above': ((price_rel > 0.05) & (price_rel <= 0.15)),
                'deep_above': (price_rel > 0.15),
            }
            zone_analysis = {}
            for zone_name, zone_mask in price_zones.items():
                zone_changes = changes[zone_mask]
                zone_analysis[f'{zone_name}_volume'] = float(np.sum(np.abs(zone_changes))) if len(zone_changes) > 0 else 0.0
                zone_analysis[f'{zone_name}_net'] = float(np.sum(zone_changes)) if len(zone_changes) > 0 else 0.0
                zone_analysis[f'{zone_name}_count'] = int(np.sum(zone_mask))
            absolute_analysis['price_zone_analysis'] = zone_analysis
            # 拉升初期识别
            absolute_analysis.update(self._analyze_pullback_pattern(changes, price_grid, current_price))
            # 虚假派发检测
            absolute_analysis['false_distribution_flag'] = self._detect_false_distribution(changes, price_grid, current_price)
            # 信号质量评估
            absolute_analysis['signal_quality'] = self._calculate_signal_quality(changes, price_rel)
            # 趋势强度评估
            absolute_analysis['trend_strength'] = self._calculate_trend_strength(changes, price_rel)
            # 关键价格位识别
            key_price_levels = self._identify_key_price_levels(changes, price_grid, current_price)
            absolute_analysis['key_price_levels'] = key_price_levels
            return absolute_analysis
        except Exception as e:
            print(f"❌ [绝对变化分析] 异常: {e}")
            return self._get_default_absolute_analysis()

    def _get_default_absolute_analysis(self) -> Dict[str, Any]:
        """获取默认的绝对变化分析结果"""
        return {
            'total_change_volume': 0.0,
            'positive_change_volume': 0.0,
            'negative_change_volume': 0.0,
            'change_concentration': 0.0,
            'max_increase': 0.0,
            'max_decrease': 0.0,
            'mean_change': 0.0,
            'price_zone_analysis': {},
            'pullback_phase_detected': False,
            'pullback_strength': 0.0,
            'support_levels': [],
            'resistance_levels': [],
            'false_distribution_flag': False,
            'signal_quality': 0.5,
            'trend_strength': 0.5,
            'key_price_levels': [],
        }

    def _create_default_key_battle_zones(self, price_grid: List[float], current_price: float) -> List[Dict[str, Any]]:
        """创建默认的关键博弈区域，确保浮点数保留3位小数"""
        if not price_grid or current_price <= 0:
            return []
        # 找到当前价附近的三个价格点
        price_array = np.array(price_grid)
        distances = np.abs(price_array - current_price)
        nearest_indices = np.argsort(distances)[:3]
        zones = []
        for idx in nearest_indices:
            price = price_array[idx]
            zones.append({
                'price': round(float(price), 3),  # 保留3位小数
                'battle_intensity': 0.1,  # 低强度
                'type': 'default',
                'position': 'below_current' if price < current_price else 'above_current',
                'distance_to_current': round(float((price - current_price) / current_price), 3),  # 保留3位小数
            })
        return zones

    def _set_default_energy_values(self):
        """设置默认的能量场值，确保浮点数保留3位小数"""
        import random
        self.absorption_energy = round(random.uniform(0.1, 5.0), 3)
        self.distribution_energy = round(random.uniform(0.1, 5.0), 3)
        self.net_energy_flow = round(self.absorption_energy - self.distribution_energy, 3)
        self.game_intensity = round(random.uniform(0.1, 0.3), 3)
        self.breakout_potential = round(random.uniform(0.1, 10.0), 3)
        self.energy_concentration = round(random.uniform(0.1, 0.5), 3)
        self.fake_distribution_flag = False
        self.key_battle_zones = []

    def _calculate_holding_factors_from_dynamics(self, dynamics_result: Dict[str, Any]):
        """从动态分析结果推算持有时间因子版本"""
        try:
            convergence = dynamics_result.get('convergence_metrics', {})
            concentration = dynamics_result.get('concentration_metrics', {})
            behavior = dynamics_result.get('behavior_patterns', {})
            absolute_signals = dynamics_result.get('absolute_change_signals', {})
            # 基础因子提取
            convergence_score = convergence.get('comprehensive_convergence', 0.5)
            concentration_score = concentration.get('comprehensive_concentration', 0.5)
            activity_score = behavior.get('main_force_activity', 0.0)
            signal_quality = absolute_signals.get('signal_quality', 0.0)
            # 长线筹码比例：高集中度 + 高收敛度 + 低活跃度
            long_term_base = (concentration_score + convergence_score) / 2
            # 活动度调整：活动度越高，长线筹码越少
            long_term_adjusted = max(0.1, long_term_base * (1.0 - activity_score * 0.5))
            # 信号质量调整：高质量信号可能意味着筹码稳定
            long_term_adjusted = long_term_adjusted * (0.7 + signal_quality * 0.3)
            self.long_term_ratio = min(0.8, long_term_adjusted)
            # =======================================================
            # 修正短线筹码计算逻辑：增加基础自然换手，防止为 0
            # =======================================================
            # 1. 基础自然换手（Base Churn）：发散度越高，自然换手越多
            # 假设发散部分的 30% 是短线交易造成的
            base_churn = (1.0 - convergence_score) * 0.3
            # 2. 主力活跃交易（Active Trading）
            active_trading = activity_score
            # 3. 组合短线基准
            short_term_base = base_churn + active_trading
            # 吸筹/派发信号调整
            accumulation = behavior.get('accumulation', {}).get('strength', 0.0)
            distribution = behavior.get('distribution', {}).get('strength', 0.0)
            signal_adjust = max(accumulation, distribution)
            short_term_adjusted = min(0.6, short_term_base + signal_adjust * 0.2)
            # 信号质量调整：
            # 即使信号质量低（噪声大），短线交易（噪声本身）也可能很高，所以不能简单乘法衰减
            # 只有当信号质量极高且显示为长线锁仓时才减少短线
            # 这里改为：保留至少 5% 的底数
            self.short_term_ratio = max(0.05, short_term_adjusted)
            # 中线筹码比例：剩余部分
            # 确保三者之和接近 1.0 (允许微小误差，或者做归一化)
            remaining = 1.0 - self.short_term_ratio - self.long_term_ratio
            if remaining < 0:
                # 如果溢出，优先压缩长线
                excess = -remaining
                self.long_term_ratio = max(0.1, self.long_term_ratio - excess)
                self.mid_term_ratio = 0.0
            else:
                self.mid_term_ratio = remaining
            # 平均持有天数：长线比例越高，平均持有天数越长
            base_days = 30 + self.long_term_ratio * 120  # 30-150天范围
            # 活跃度调整：活跃度越高，持有时间越短
            self.avg_holding_days = max(10, base_days * (1.0 - activity_score * 0.7))
            # =======================================================
            # 优化：对最终结果保留 4 位小数，保持数据整洁
            # =======================================================
            self.short_term_ratio = round(self.short_term_ratio, 4)
            self.mid_term_ratio = round(self.mid_term_ratio, 4)
            self.long_term_ratio = round(self.long_term_ratio, 4)
            self.avg_holding_days = round(self.avg_holding_days, 1) # 天数保留1位即可
        except Exception as e:
            print(f"⚠️ [计算警告] 推算持有时间因子异常: {e}")
            import traceback
            traceback.print_exc()
            # 使用默认值
            self.short_term_ratio = 0.2
            self.mid_term_ratio = 0.3
            self.long_term_ratio = 0.5
            self.avg_holding_days = 100.0

    def _calculate_absolute_change_analysis(self, changes: np.ndarray, price_grid: np.ndarray, current_price: float) -> Dict[str, Any]:
        """
        基于绝对值变化的纠偏分析
        核心逻辑：识别拉升初期的"虚假派发"
        """
        try:
            if len(changes) == 0 or len(price_grid) == 0 or current_price <= 0:
                return {}
            # 1. 绝对变化统计分析
            absolute_analysis = {
                'total_change_volume': float(np.sum(np.abs(changes))),
                'positive_change_volume': float(np.sum(changes[changes > 0])),
                'negative_change_volume': float(np.sum(changes[changes < 0])),
                'change_concentration': self._calculate_change_concentration(changes),
                'max_increase': float(np.max(changes)) if len(changes) > 0 else 0.0,
                'max_decrease': float(np.min(changes)) if len(changes) > 0 else 0.0,
                'mean_change': float(np.mean(changes)) if len(changes) > 0 else 0.0,
                'std_change': float(np.std(changes)) if len(changes) > 0 else 0.0,
            }
            # 2. 价格分层分析
            price_rel = (price_grid - current_price) / current_price
            # 定义价格区间
            price_zones = {
                'deep_below': (price_rel < -0.15),
                'below': ((price_rel >= -0.15) & (price_rel < -0.05)),
                'near': ((price_rel >= -0.05) & (price_rel <= 0.05)),
                'above': ((price_rel > 0.05) & (price_rel <= 0.15)),
                'deep_above': (price_rel > 0.15),
            }
            zone_analysis = {}
            for zone_name, zone_mask in price_zones.items():
                zone_changes = changes[zone_mask]
                if len(zone_changes) > 0:
                    zone_analysis[f'{zone_name}_volume'] = float(np.sum(np.abs(zone_changes)))
                    zone_analysis[f'{zone_name}_net'] = float(np.sum(zone_changes))
                    zone_analysis[f'{zone_name}_count'] = int(np.sum(zone_mask))
                else:
                    zone_analysis[f'{zone_name}_volume'] = 0.0
                    zone_analysis[f'{zone_name}_net'] = 0.0
                    zone_analysis[f'{zone_name}_count'] = 0
            absolute_analysis['price_zone_analysis'] = zone_analysis
            # 3. 拉升初期识别
            pullback_analysis = self._analyze_pullback_pattern(changes, price_grid, current_price)
            absolute_analysis.update(pullback_analysis)
            # 4. 虚假派发检测
            false_distribution = self._detect_false_distribution(changes, price_grid, current_price)
            absolute_analysis['false_distribution_flag'] = false_distribution
            # 5. 信号质量评估
            absolute_analysis['signal_quality'] = self._calculate_signal_quality(changes, price_rel)
            # 6. 趋势强度评估
            absolute_analysis['trend_strength'] = self._calculate_trend_strength(changes, price_rel)
            # 7. 关键价格位识别
            key_price_levels = self._identify_key_price_levels(changes, price_grid, current_price)
            absolute_analysis['key_price_levels'] = key_price_levels
            return absolute_analysis
        except Exception as e:
            print(f"绝对变化分析失败: {e}")
            return {}

    def _calculate_change_concentration(self, changes: np.ndarray) -> float:
        """计算变化集中度（越高表示主力行为越明显）"""
        if len(changes) == 0:
            return 0.0
        abs_changes = np.abs(changes)
        total_volume = np.sum(abs_changes)
        if total_volume == 0:
            return 0.0
        # 计算Top 5价格格的变化占比
        top_indices = np.argsort(abs_changes)[-5:]
        top_volume = np.sum(abs_changes[top_indices])
        return top_volume / total_volume

    def _analyze_pullback_pattern(self, changes: np.ndarray, price_grid: np.ndarray, current_price: float) -> Dict[str, Any]:
        """分析拉升初期模式"""
        analysis = {
            'pullback_phase_detected': False,
            'pullback_strength': 0.0,
            'support_levels': [],
            'resistance_levels': [],
        }
        try:
            # 寻找当前价以下的筹码增加区域（支撑形成）
            below_current = price_grid < current_price * 0.99
            accumulation_below = np.sum(changes[below_current & (changes > 0)])
            # 寻找当前价以上的筹码减少区域（获利回吐）
            above_current = price_grid > current_price * 1.01
            distribution_above = np.sum(-changes[above_current & (changes < 0)])
            # 寻找当前价以下的筹码减少区域（止损或换手）
            distribution_below = np.sum(-changes[below_current & (changes < 0)])
            # 寻找当前价以上的筹码增加区域（追高或派发）
            accumulation_above = np.sum(changes[above_current & (changes > 0)])
            # 拉升初期特征：低位支撑形成 + 高位获利回吐
            if accumulation_below > 0.5 and distribution_above > 0.3:
                analysis['pullback_phase_detected'] = True
                analysis['pullback_strength'] = min(1.0, (accumulation_below + distribution_above) / 2.0)
                # 识别支撑位（低位筹码大幅增加）
                support_mask = below_current & (changes > 0.3)
                support_indices = np.where(support_mask)[0]
                for idx in support_indices[:3]:  # 取前3个支撑位
                    analysis['support_levels'].append({
                        'price': float(price_grid[idx]),
                        'strength': float(changes[idx]),
                        'distance_to_current': float((current_price - price_grid[idx]) / current_price),
                        'type': 'support'
                    })
                # 识别阻力位（高位筹码大幅减少）
                resistance_mask = above_current & (changes < -0.3)
                resistance_indices = np.where(resistance_mask)[0]
                for idx in resistance_indices[:3]:  # 取前3个阻力位
                    analysis['resistance_levels'].append({
                        'price': float(price_grid[idx]),
                        'strength': float(-changes[idx]),
                        'distance_to_current': float((price_grid[idx] - current_price) / current_price),
                        'type': 'resistance'
                    })
            # 统计信息
            analysis['accumulation_below'] = float(accumulation_below)
            analysis['distribution_above'] = float(distribution_above)
            analysis['distribution_below'] = float(distribution_below)
            analysis['accumulation_above'] = float(accumulation_above)
        except Exception as e:
            print(f"拉升初期分析失败: {e}")
        return analysis

    def _detect_false_distribution(self, changes: np.ndarray, price_grid: np.ndarray, current_price: float) -> bool:
        """检测虚假派发（获利回吐 vs 真实派发）"""
        try:
            # 真实派发特征：高位筹码大幅增加 + 低位筹码减少
            high_price_mask = price_grid > current_price * 1.05
            low_price_mask = price_grid < current_price * 0.95
            high_increase = np.sum(changes[high_price_mask & (changes > 0)])
            low_decrease = np.sum(-changes[low_price_mask & (changes < 0)])
            # 获利回吐特征：中高位筹码减少，但低位形成支撑
            mid_high_mask = (price_grid > current_price) & (price_grid <= current_price * 1.05)
            mid_decrease = np.sum(-changes[mid_high_mask & (changes < 0)])
            # 虚假派发判断：中高位减少但高位未明显增加，且低位有支撑
            if mid_decrease > 0.4 and high_increase < 0.2:
                return True
            # 另一种情况：低位有显著吸收，高位减少可能是洗盘
            if low_decrease < 0.2 and high_increase < 0.3 and mid_decrease > 0.5:
                return True
            return False
        except Exception as e:
            print(f"虚假派发检测失败: {e}")
            return False

    def _calculate_signal_quality(self, changes: np.ndarray, price_rel: np.ndarray) -> float:
        """计算信号质量"""
        try:
            # 1. 变化集中度
            concentration_score = self._calculate_change_concentration(changes)
            # 2. 价格分布合理性（筹码变化是否在合理价格区间）
            # 合理的筹码变化应该集中在当前价附近
            near_mask = np.abs(price_rel) < 0.1
            near_volume = np.sum(np.abs(changes[near_mask]))
            total_volume = np.sum(np.abs(changes))
            if total_volume > 0:
                distribution_score = near_volume / total_volume
            else:
                distribution_score = 0.0
            # 3. 噪声水平（小变化的占比）
            noise_mask = np.abs(changes) < 0.1
            noise_ratio = np.sum(noise_mask) / len(changes) if len(changes) > 0 else 1.0
            noise_score = 1.0 - noise_ratio
            # 综合质量评分
            quality_score = 0.4 * concentration_score + 0.3 * distribution_score + 0.3 * noise_score
            return min(1.0, max(0.0, quality_score))
        except Exception as e:
            print(f"信号质量计算失败: {e}")
            return 0.5

    def _calculate_trend_strength(self, changes: np.ndarray, price_rel: np.ndarray) -> float:
        """计算趋势强度"""
        try:
            # 1. 净流向强度
            net_flow = np.sum(changes)
            total_volume = np.sum(np.abs(changes))
            if total_volume > 0:
                flow_strength = abs(net_flow) / total_volume
            else:
                flow_strength = 0.0
            # 2. 价格一致性（上涨趋势中，低位应减少，高位应增加）
            below_mask = price_rel < -0.05
            above_mask = price_rel > 0.05
            below_flow = np.sum(changes[below_mask])
            above_flow = np.sum(changes[above_mask])
            # 上涨趋势：低位减少，高位增加
            if below_flow < 0 and above_flow > 0:
                consistency_score = 0.7 + 0.3 * min(abs(below_flow), above_flow) / max(abs(below_flow), above_flow)
            # 下跌趋势：低位增加，高位减少
            elif below_flow > 0 and above_flow < 0:
                consistency_score = 0.7 + 0.3 * min(below_flow, abs(above_flow)) / max(below_flow, abs(above_flow))
            else:
                consistency_score = 0.3
            # 3. 变化幅度
            amplitude_score = min(1.0, total_volume / 20.0)  # 经验值
            # 综合趋势强度
            trend_strength = 0.3 * flow_strength + 0.4 * consistency_score + 0.3 * amplitude_score
            return min(1.0, max(0.0, trend_strength))
        except Exception as e:
            print(f"趋势强度计算失败: {e}")
            return 0.5

    def _identify_key_price_levels(self, changes: np.ndarray, price_grid: np.ndarray, current_price: float) -> List[Dict[str, Any]]:
        """识别关键价格位"""
        key_levels = []
        try:
            # 1. 寻找显著变化点（变化绝对值大于阈值）
            threshold = 0.5  # 50%的变化阈值
            significant_mask = np.abs(changes) > threshold
            if np.sum(significant_mask) > 0:
                significant_indices = np.where(significant_mask)[0]
                for idx in significant_indices:
                    price = price_grid[idx]
                    change = changes[idx]
                    price_rel = (price - current_price) / current_price
                    level_type = 'absorption' if change > 0 else 'distribution'
                    position = 'below_current' if price < current_price else 'above_current'
                    key_levels.append({
                        'price': float(price),
                        'change': float(change),
                        'abs_change': float(abs(change)),
                        'type': level_type,
                        'position': position,
                        'distance_to_current': float(price_rel),
                        'strength': min(1.0, abs(change) / 2.0)  # 归一化到0-1
                    })
            # 2. 按强度排序，取前10个
            key_levels.sort(key=lambda x: x['abs_change'], reverse=True)
            key_levels = key_levels[:10]
        except Exception as e:
            print(f"关键价格位识别失败: {e}")
        return key_levels

    def _calculate_key_battle_intensity(self) -> float:
        """计算关键博弈区域的总强度"""
        if not self.key_battle_zones:
            return 0.0
        total_intensity = sum([zone.get('battle_intensity', 0) for zone in self.key_battle_zones])
        return min(1.0, total_intensity / 5.0)  # 归一化

    def _get_absolute_change_quality(self) -> float:
        """基于绝对变化的信号质量"""
        if not self.absolute_change_analysis:
            return 0.5
        analysis = self.absolute_change_analysis
        # 高质量信号：变化集中度高，虚假派发标志为False
        concentration_score = analysis.get('change_concentration', 0.0)
        false_distribution = analysis.get('false_distribution_flag', False)
        if false_distribution:
            return 0.3  # 虚假信号，质量较低
        else:
            return 0.5 + concentration_score * 0.5

    def _calculate_breakout_probability(self) -> float:
        """基于能量场的突破概率"""
        if self.breakout_potential < 20:
            return 0.0
        # 基础概率：突破势能
        base_prob = min(1.0, self.breakout_potential / 100)
        # 能量集中度加成
        concentration_bonus = self.energy_concentration * 0.2
        # 博弈强度加成（适中的博弈强度最有利于突破）
        intensity_bonus = 0.1 if 0.3 < self.game_intensity < 0.7 else 0.0
        # 净能量流向加成
        flow_bonus = min(0.2, max(0, self.net_energy_flow / 100))
        total_prob = base_prob + concentration_bonus + intensity_bonus + flow_bonus
        return round(min(1.0, total_prob), 3)

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

