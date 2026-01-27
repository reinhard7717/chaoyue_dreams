# stock_models\chip_factors.py
from django.db import models
from django.utils.translation import gettext_lazy as _
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import linregress
from typing import List, Tuple, Dict, Optional
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
    
    # ========== 筹码平均持有时间 ==========
    avg_holding_days = models.FloatField(
        verbose_name='平均持有时长(天)',
        help_text='基于换手率推算的筹码平均持有时间',
        default=100.0  # 添加默认值
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
    
    calc_time = models.DateTimeField(verbose_name='计算时间', auto_now=True)
    error_message = models.TextField(verbose_name='错误信息', null=True, blank=True)
    
    class Meta:
        abstract = True
        verbose_name = '筹码因子基础'
        verbose_name_plural = '筹码因子基础'
        unique_together = ('stock', 'trade_time')
    
    def __str__(self):
        return f"{self.stock.stock_code} {self.trade_time}"

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
    筹码持有时间矩阵基类
    """
    stock = models.ForeignKey(
        'StockInfo',
        to_field='stock_code',
        on_delete=models.CASCADE,
        verbose_name='股票',
        db_index=True
    )
    trade_time = models.DateField(verbose_name='交易日期', db_index=True)
    
    # 基础因子
    short_term_ratio = models.FloatField(
        verbose_name='短线筹码比例(<5日)',
        null=True, blank=True,
        help_text='持有少于5天的筹码比例'
    )
    mid_term_ratio = models.FloatField(
        verbose_name='中线筹码比例(5-60日)',
        null=True, blank=True,
        help_text='持有5-60天的筹码比例'
    )
    long_term_ratio = models.FloatField(
        verbose_name='长线筹码比例(>60日)',
        null=True, blank=True,
        help_text='持有超过60天的筹码比例'
    )
    avg_holding_days = models.FloatField(
        verbose_name='平均持有天数',
        null=True, blank=True
    )
    
    # 矩阵数据（存储完整的持有时间矩阵）
    matrix_data = models.JSONField(
        verbose_name='矩阵数据',
        null=True, blank=True,
        help_text='持有时间矩阵的JSON表示'
    )
    
    # 压缩矩阵（二进制存储，更高效）
    compressed_matrix = models.BinaryField(
        verbose_name='压缩矩阵',
        null=True, blank=True
    )
    
    # 计算元数据
    price_grid = models.JSONField(
        verbose_name='价格网格',
        null=True, blank=True,
        help_text='价格区间网格点'
    )
    max_holding_days = models.IntegerField(
        verbose_name='最大持有天数',
        default=250
    )
    
    # 验证信息
    validation_score = models.FloatField(
        verbose_name='验证分数',
        null=True, blank=True,
        help_text='计算结果的可信度评分(0-1)'
    )
    validation_warnings = models.JSONField(
        verbose_name='验证警告',
        null=True, blank=True,
        help_text='计算过程中的警告信息'
    )
    
    # 计算状态
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
    calc_time = models.DateTimeField(verbose_name='计算时间', auto_now=True)
    error_message = models.TextField(verbose_name='错误信息', null=True, blank=True)
    
    # 数据源标记
    used_minute_data = models.BooleanField(
        verbose_name='使用分钟数据',
        default=True
    )
    used_tick_data = models.BooleanField(
        verbose_name='使用逐笔数据',
        default=False
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
        return f"{self.stock.stock_code} {self.trade_time} 持有矩阵"
    
    def get_holding_matrix(self) -> np.ndarray:
        """
        从数据库加载持有时间矩阵
        """
        try:
            if self.compressed_matrix:
                # 从二进制数据加载
                # 注意：compressed_matrix 已经是 base64 编码的 bytes
                if isinstance(self.compressed_matrix, bytes):
                    matrix_bytes = base64.b64decode(self.compressed_matrix)
                elif isinstance(self.compressed_matrix, str):
                    # 如果是字符串，先编码为 bytes
                    matrix_bytes = base64.b64decode(self.compressed_matrix.encode('utf-8'))
                else:
                    print(f"⚠️ [加载持有矩阵] 未知的数据类型: {type(self.compressed_matrix)}")
                    return np.array([])
                return pickle.loads(matrix_bytes)
            elif self.matrix_data:
                # 从JSON数据加载
                return np.array(self.matrix_data.get('matrix', []))
            return np.array([])
        except Exception as e:
            print(f"加载持有矩阵失败: {e}")
            return np.array([])

    def set_holding_matrix(self, matrix: np.ndarray, compress: bool = True):
        """
        设置持有时间矩阵
        """
        try:
            if compress:
                # 压缩存储
                matrix_bytes = pickle.dumps(matrix)
                self.compressed_matrix = base64.b64encode(matrix_bytes)
                self.matrix_data = None
            else:
                # JSON存储
                self.matrix_data = {'matrix': matrix.tolist()}
                self.compressed_matrix = None
        except Exception as e:
            print(f"保存持有矩阵失败: {e}")

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

