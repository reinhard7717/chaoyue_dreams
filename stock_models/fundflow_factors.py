# stock_models\fundflow_factors.py
from django.db import models
from django.utils.translation import gettext_lazy as _
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import linregress
from typing import List, Tuple, Dict, Optional, Any
import json
import base64
import pickle

class FundFlowFactorBase(models.Model):
    """
    资金流向因子基础模型（抽象基类）
    存储基于原始资金流向数据计算的高级技术指标
    """
    stock = models.ForeignKey(
        'StockInfo', 
        to_field='stock_code',
        on_delete=models.CASCADE,
        verbose_name='股票',
        db_index=True,
        related_name="%(class)s_factors",  # 修改这里：使用占位符
        related_query_name="%(class)s_factor"  # 添加查询名称
    )
    trade_time = models.DateField(verbose_name='交易日期', db_index=True)
    
    # ==================== 1. 绝对量级指标 ====================
    # 1.1 累计净流入量
    total_net_amount_3d = models.DecimalField(
        max_digits=20, decimal_places=2,
        verbose_name='3日累计净流入(万元)', null=True, blank=True,
        help_text='过去3日累计资金净流入额'
    )
    total_net_amount_5d = models.DecimalField(
        max_digits=20, decimal_places=2,
        verbose_name='5日累计净流入(万元)', null=True, blank=True,
        help_text='过去5日累计资金净流入额'
    )
    total_net_amount_10d = models.DecimalField(
        max_digits=20, decimal_places=2,
        verbose_name='10日累计净流入(万元)', null=True, blank=True,
        help_text='过去10日累计资金净流入额'
    )
    total_net_amount_20d = models.DecimalField(
        max_digits=20, decimal_places=2,
        verbose_name='20日累计净流入(万元)', null=True, blank=True,
        help_text='过去20日累计资金净流入额'
    )
    
    # 1.2 日均净流入
    avg_daily_net_5d = models.DecimalField(
        max_digits=20, decimal_places=2,
        verbose_name='5日均净流入(万元)', null=True, blank=True,
        help_text='过去5日平均每日净流入额'
    )
    avg_daily_net_10d = models.DecimalField(
        max_digits=20, decimal_places=2,
        verbose_name='10日均净流入(万元)', null=True, blank=True,
        help_text='过去10日平均每日净流入额'
    )
    avg_daily_net_20d = models.DecimalField(
        max_digits=20, decimal_places=2,
        verbose_name='20日均净流入(万元)', null=True, blank=True,
        help_text='过去20日平均每日净流入额'
    )
    
    # 1.3 累计成交量（辅助指标）
    total_volume_5d = models.DecimalField(
        max_digits=20, decimal_places=2,
        verbose_name='5日累计成交量(万手)', null=True, blank=True,
        help_text='过去5日累计成交量'
    )
    total_volume_10d = models.DecimalField(
        max_digits=20, decimal_places=2,
        verbose_name='10日累计成交量(万手)', null=True, blank=True,
        help_text='过去10日累计成交量'
    )
    
    # ==================== 2. 相对强度指标 ====================
    # 2.1 净流入占比（相对成交额）
    net_amount_ratio = models.DecimalField(
        max_digits=10, decimal_places=4,
        verbose_name='当日净流入占比(%)', null=True, blank=True,
        help_text='净流入额/当日成交额 × 100%'
    )
    net_amount_ratio_ma5 = models.DecimalField(
        max_digits=10, decimal_places=4,
        verbose_name='5日均净流入占比(%)', null=True, blank=True,
        help_text='过去5日平均净流入占比'
    )
    net_amount_ratio_ma10 = models.DecimalField(
        max_digits=10, decimal_places=4,
        verbose_name='10日均净流入占比(%)', null=True, blank=True,
        help_text='过去10日平均净流入占比'
    )
    
    # 2.2 净流入强度（标准化得分）
    flow_intensity = models.DecimalField(
        max_digits=6, decimal_places=2,
        verbose_name='资金流入强度得分', null=True, blank=True,
        help_text='综合考虑绝对量和相对占比的强度得分，范围0-100'
    )
    
    # 2.3 强度分级
    intensity_level = models.IntegerField(
        verbose_name='强度分级', null=True, blank=True,
        help_text='1:低强度 2:中强度 3:高强度 4:极高强度'
    )
    
    # ==================== 3. 主力行为模式识别 ====================
    # 3.1 行为模式得分
    accumulation_score = models.DecimalField(
        max_digits=6, decimal_places=2,
        verbose_name='建仓模式得分', null=True, blank=True,
        help_text='0-100，得分越高表示越符合建仓特征'
    )
    pushing_score = models.DecimalField(
        max_digits=6, decimal_places=2,
        verbose_name='拉升模式得分', null=True, blank=True,
        help_text='0-100，得分越高表示越符合拉升特征'
    )
    distribution_score = models.DecimalField(
        max_digits=6, decimal_places=2,
        verbose_name='派发模式得分', null=True, blank=True,
        help_text='0-100，得分越高表示越符合派发特征'
    )
    shakeout_score = models.DecimalField(
        max_digits=6, decimal_places=2,
        verbose_name='洗盘模式得分', null=True, blank=True,
        help_text='0-100，得分越高表示越符合洗盘特征'
    )
    
    # 3.2 行为模式分类
    BEHAVIOR_CHOICES = [
        ('ACCUMULATION', '吸筹建仓'),
        ('PUSHING', '拉升推高'),
        ('DISTRIBUTION', '派发出货'),
        ('SHAKEOUT', '洗盘震荡'),
        ('MIXED', '混合模式'),
        ('UNCLEAR', '不明'),
    ]
    behavior_pattern = models.CharField(
        max_length=20,
        choices=BEHAVIOR_CHOICES,
        verbose_name='主力行为模式', null=True, blank=True,
        help_text='基于资金流判断的主力行为模式'
    )
    
    # 3.3 模式置信度
    pattern_confidence = models.DecimalField(
        max_digits=5, decimal_places=2,
        verbose_name='模式置信度(%)', null=True, blank=True,
        help_text='行为模式判断的置信度，0-100'
    )
    
    # ==================== 4. 资金流向质量评估 ====================
    # 4.1 流出质量评估
    outflow_quality = models.DecimalField(
        max_digits=5, decimal_places=2,
        verbose_name='流出质量评分', null=True, blank=True,
        help_text='0-100，考虑流出量级和背景的评分，越高表示越可能是正常流出'
    )
    
    # 4.2 流入持续性
    inflow_persistence = models.IntegerField(
        verbose_name='连续净流入天数', null=True, blank=True,
        help_text='连续资金净流入的天数'
    )
    
    # 4.3 大单异动检测
    large_order_anomaly = models.BooleanField(
        verbose_name='大单异动', default=False,
        help_text='是否存在异常大单行为'
    )
    anomaly_intensity = models.DecimalField(
        max_digits=6, decimal_places=2,
        verbose_name='异动强度', null=True, blank=True,
        help_text='异动信号的强度，0-100'
    )
    
    # 4.4 分档资金一致性
    flow_consistency = models.DecimalField(
        max_digits=5, decimal_places=2,
        verbose_name='分档资金一致性', null=True, blank=True,
        help_text='0-100，特大单/大单/中单资金流向方向的一致性评分'
    )
    
    # 4.5 资金稳定性
    flow_stability = models.DecimalField(
        max_digits=5, decimal_places=2,
        verbose_name='资金流稳定性', null=True, blank=True,
        help_text='0-100，资金流入/流出波动性评分'
    )
    
    # ==================== 5. 多周期资金共振指标 ====================
    # 5.1 周期共振度
    daily_weekly_sync = models.DecimalField(
        max_digits=5, decimal_places=2,
        verbose_name='日周线共振度', null=True, blank=True,
        help_text='0-100，日线与周线资金流向的一致性'
    )
    daily_monthly_sync = models.DecimalField(
        max_digits=5, decimal_places=2,
        verbose_name='日月线共振度', null=True, blank=True,
        help_text='0-100，日线与月线资金流向的一致性'
    )
    
    # 5.2 滚动窗口共振
    short_mid_sync = models.DecimalField(
        max_digits=5, decimal_places=2,
        verbose_name='短中期共振度', null=True, blank=True,
        help_text='0-100，5日与20日资金流向的一致性'
    )
    mid_long_sync = models.DecimalField(
        max_digits=5, decimal_places=2,
        verbose_name='中长期共振度', null=True, blank=True,
        help_text='0-100，20日与60日资金流向的一致性'
    )
    
    # ==================== 6. 趋势动量指标 ====================
    # 6.1 净流入动量
    flow_momentum_5d = models.DecimalField(
        max_digits=10, decimal_places=4,
        verbose_name='5日资金动量', null=True, blank=True,
        help_text='净流入量的5日变化率'
    )
    flow_momentum_10d = models.DecimalField(
        max_digits=10, decimal_places=4,
        verbose_name='10日资金动量', null=True, blank=True,
        help_text='净流入量的10日变化率'
    )
    
    # 6.2 加速度指标
    flow_acceleration = models.DecimalField(
        max_digits=10, decimal_places=4,
        verbose_name='资金流加速度', null=True, blank=True,
        help_text='资金净流入变化率的加速度'
    )
    
    # 6.3 趋势强度
    uptrend_strength = models.DecimalField(
        max_digits=5, decimal_places=2,
        verbose_name='上升趋势强度', null=True, blank=True,
        help_text='0-100，资金流入形成的上升趋势强度'
    )
    downtrend_strength = models.DecimalField(
        max_digits=5, decimal_places=2,
        verbose_name='下降趋势强度', null=True, blank=True,
        help_text='0-100，资金流出形成的下降趋势强度'
    )
    
    # ==================== 7. 量价背离指标 ====================
    # 7.1 量价背离度
    price_flow_divergence = models.DecimalField(
        max_digits=8, decimal_places=4,
        verbose_name='量价背离度', null=True, blank=True,
        help_text='价格变化与资金流向的背离程度，正值表示背离'
    )
    
    # 7.2 背离类型
    DIVERGENCE_CHOICES = [
        ('BULLISH', '看多背离 - 价格下跌但资金流入'),
        ('BEARISH', '看空背离 - 价格上涨但资金流出'),
        ('NONE', '无明显背离'),
        ('MIXED', '混合背离'),
    ]
    divergence_type = models.CharField(
        max_length=20,
        choices=DIVERGENCE_CHOICES,
        verbose_name='背离类型', null=True, blank=True
    )
    
    # 7.3 背离强度
    divergence_strength = models.DecimalField(
        max_digits=5, decimal_places=2,
        verbose_name='背离强度', null=True, blank=True,
        help_text='0-100，背离信号的强度'
    )
    
    # ==================== 8. 结构分析指标 ====================
    # 8.1 峰值检测
    flow_peak_value = models.DecimalField(
        max_digits=20, decimal_places=2,
        verbose_name='资金流峰值', null=True, blank=True,
        help_text='近期资金流的峰值水平'
    )
    days_since_last_peak = models.IntegerField(
        verbose_name='距上次峰值天数', null=True, blank=True
    )
    
    # 8.2 支撑阻力位
    flow_support_level = models.DecimalField(
        max_digits=20, decimal_places=2,
        verbose_name='资金流支撑位', null=True, blank=True
    )
    flow_resistance_level = models.DecimalField(
        max_digits=20, decimal_places=2,
        verbose_name='资金流阻力位', null=True, blank=True
    )
    
    # ==================== 9. 统计特征指标 ====================
    # 9.1 统计分布
    flow_zscore = models.DecimalField(
        max_digits=8, decimal_places=4,
        verbose_name='资金流Z分数', null=True, blank=True,
        help_text='当前资金流相对于历史分布的标准化分数'
    )
    flow_percentile = models.DecimalField(
        max_digits=5, decimal_places=2,
        verbose_name='资金流百分位', null=True, blank=True,
        help_text='0-100，当前资金流在历史分布中的位置'
    )
    
    # 9.2 波动性
    flow_volatility_10d = models.DecimalField(
        max_digits=10, decimal_places=4,
        verbose_name='10日资金流波动率', null=True, blank=True
    )
    flow_volatility_20d = models.DecimalField(
        max_digits=10, decimal_places=4,
        verbose_name='20日资金流波动率', null=True, blank=True
    )
    
    # ==================== 10. 预测指标 ====================
    # 10.1 未来预期
    expected_flow_next_1d = models.DecimalField(
        max_digits=20, decimal_places=2,
        verbose_name='预期明日净流入', null=True, blank=True,
        help_text='基于模型预测的下一个交易日净流入额'
    )
    flow_forecast_confidence = models.DecimalField(
        max_digits=5, decimal_places=2,
        verbose_name='预测置信度', null=True, blank=True,
        help_text='0-100，预测结果的置信度'
    )
    
    # 10.2 趋势延续概率
    uptrend_continuation_prob = models.DecimalField(
        max_digits=5, decimal_places=2,
        verbose_name='上升趋势延续概率(%)', null=True, blank=True
    )
    reversal_prob = models.DecimalField(
        max_digits=5, decimal_places=2,
        verbose_name='趋势反转概率(%)', null=True, blank=True
    )
    
    # ==================== 11. 复合综合指标 ====================
    # 11.1 综合评分
    comprehensive_score = models.DecimalField(
        max_digits=6, decimal_places=2,
        verbose_name='综合资金评分', null=True, blank=True,
        help_text='0-100，综合考虑所有资金指标的评分'
    )
    
    # 11.2 交易信号
    SIGNAL_CHOICES = [
        ('STRONG_BUY', '强烈买入'),
        ('BUY', '买入'),
        ('HOLD', '持有'),
        ('SELL', '卖出'),
        ('STRONG_SELL', '强烈卖出'),
        ('NEUTRAL', '中性'),
    ]
    trading_signal = models.CharField(
        max_length=20,
        choices=SIGNAL_CHOICES,
        verbose_name='交易信号', null=True, blank=True
    )
    signal_strength = models.DecimalField(
        max_digits=5, decimal_places=2,
        verbose_name='信号强度', null=True, blank=True,
        help_text='0-100，交易信号的强度'
    )
    
    # ==================== 12. 原始数据快照 ====================
    # 12.2 特征向量
    feature_vector = models.TextField(
        verbose_name='特征向量', null=True, blank=True,
        help_text='Base64编码的特征向量，用于机器学习模型'
    )

    # ==================== 13. 基于Tick数据的资金流向增强指标 ====================
    # 13.1 日内资金流分布特征
    intraday_flow_distribution = models.JSONField(
        verbose_name='日内资金流分布', null=True, blank=True,
        help_text='JSON格式，记录各时间段资金分布特征'
    )

    # 13.2 高频大单识别
    tick_large_order_net = models.DecimalField(
        max_digits=20, decimal_places=2,
        verbose_name='Tick大单净流入(万元)', null=True, blank=True,
        help_text='基于tick数据识别的大单净流入'
    )
    tick_large_order_count = models.IntegerField(
        verbose_name='大单笔数', null=True, blank=True,
        help_text='日内识别的大单交易笔数'
    )

    # 13.3 资金冲击特征
    flow_impact_ratio = models.DecimalField(
        max_digits=10, decimal_places=4,
        verbose_name='资金冲击系数', null=True, blank=True,
        help_text='单位资金对价格的冲击程度'
    )
    flow_persistence_minutes = models.IntegerField(
        verbose_name='资金持续分钟数', null=True, blank=True,
        help_text='连续同向资金流入的分钟数'
    )

    # 13.4 日内资金动量
    intraday_flow_momentum = models.DecimalField(
        max_digits=10, decimal_places=4,
        verbose_name='日内资金动量', null=True, blank=True,
        help_text='日内资金流向的动量指标'
    )
    flow_acceleration_intraday = models.DecimalField(
        max_digits=10, decimal_places=4,
        verbose_name='日内资金加速度', null=True, blank=True,
        help_text='日内资金流入的加速/减速特征'
    )

    # 13.5 资金聚类特征
    flow_cluster_intensity = models.DecimalField(
        max_digits=10, decimal_places=4,
        verbose_name='资金聚类强度', null=True, blank=True,
        help_text='资金流入的时间聚集程度'
    )
    flow_cluster_duration = models.IntegerField(
        verbose_name='资金聚类持续时间(分钟)', null=True, blank=True,
        help_text='资金集中流入的持续时间'
    )

    # 13.6 高频资金分歧度
    high_freq_flow_divergence = models.DecimalField(
        max_digits=10, decimal_places=4,
        verbose_name='高频资金分歧度', null=True, blank=True,
        help_text='高频资金流入流出之间的分歧程度'
    )

    # 13.7 日内VWAP偏离
    vwap_deviation = models.DecimalField(
        max_digits=10, decimal_places=4,
        verbose_name='VWAP偏离度(%)', null=True, blank=True,
        help_text='资金流入价格相对于VWAP的偏离程度'
    )

    # 13.8 资金流入效率
    flow_efficiency = models.DecimalField(
        max_digits=10, decimal_places=4,
        verbose_name='资金流入效率', null=True, blank=True,
        help_text='单位资金流入推动的价格变化'
    )

    # 13.9 尾盘资金特征
    closing_flow_ratio = models.DecimalField(
        max_digits=10, decimal_places=4,
        verbose_name='尾盘资金占比(%)', null=True, blank=True,
        help_text='收盘前30分钟资金流入占比'
    )
    closing_flow_intensity = models.DecimalField(
        max_digits=10, decimal_places=4,
        verbose_name='尾盘资金强度', null=True, blank=True,
        help_text='尾盘资金流入的集中程度'
    )

    # 13.10 高频统计特征
    high_freq_flow_skewness = models.DecimalField(
        max_digits=10, decimal_places=4,
        verbose_name='高频资金偏度', null=True, blank=True,
        help_text='日内资金流分布的偏度特征'
    )
    high_freq_flow_kurtosis = models.DecimalField(
        max_digits=10, decimal_places=4,
        verbose_name='高频资金峰度', null=True, blank=True,
        help_text='日内资金流分布的峰度特征'
    )

    # 13.11 资金流入时段分布
    morning_flow_ratio = models.DecimalField(
        max_digits=10, decimal_places=4,
        verbose_name='上午资金占比(%)', null=True, blank=True,
        help_text='上午交易时段资金流入占比'
    )
    afternoon_flow_ratio = models.DecimalField(
        max_digits=10, decimal_places=4,
        verbose_name='下午资金占比(%)', null=True, blank=True,
        help_text='下午交易时段资金流入占比'
    )

    # 13.12 主力隐蔽性指标
    stealth_flow_ratio = models.DecimalField(
        max_digits=10, decimal_places=4,
        verbose_name='隐蔽资金占比(%)', null=True, blank=True,
        help_text='分散小单但持续流入的资金占比'
    )
    
    # ==================== 时间戳管理 ====================
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')
    
    class Meta:
        abstract = True
        unique_together = ['stock', 'trade_time']
        indexes = [
            models.Index(fields=['stock', 'trade_time']),
            models.Index(fields=['trade_time', 'intensity_level']),
            models.Index(fields=['behavior_pattern', 'trade_time']),
            models.Index(fields=['trading_signal', 'trade_time']),
            models.Index(fields=['flow_intensity', 'trade_time']),
        ]
    
    def __str__(self):
        return f"{self.stock.stock_code if self.stock else ''}资金因子({self.trade_time})"
       
    def save_feature_vector(self, vector: np.ndarray):
        """保存特征向量"""
        vector_bytes = pickle.dumps(vector)
        self.feature_vector = base64.b64encode(vector_bytes).decode('utf-8')
    
    def load_feature_vector(self) -> Optional[np.ndarray]:
        """加载特征向量"""
        if self.feature_vector:
            vector_bytes = base64.b64decode(self.feature_vector)
            return pickle.loads(vector_bytes)
        return None
    
    def save_metadata(self, metadata: Dict):
        """保存计算元数据"""
        self.calculation_metadata = json.dumps(metadata, ensure_ascii=False)
    
    def load_metadata(self) -> Dict:
        """加载计算元数据"""
        if self.calculation_metadata:
            return json.loads(self.calculation_metadata)
        return {}


# 深交所主板资金流向因子分表
class FundFlowFactorSZ(FundFlowFactorBase):
    class Meta:
        verbose_name = '资金流向因子SZ'
        verbose_name_plural = '资金流向因子SZ'
        db_table = 'stock_fundflow_factor_sz'

# 上交所主板资金流向因子分表
class FundFlowFactorSH(FundFlowFactorBase):
    class Meta:
        verbose_name = '资金流向因子SH'
        verbose_name_plural = '资金流向因子SH'
        db_table = 'stock_fundflow_factor_sh'

# 创业板资金流向因子分表
class FundFlowFactorCY(FundFlowFactorBase):
    class Meta:
        verbose_name = '资金流向因子CY'
        verbose_name_plural = '资金流向因子CY'
        db_table = 'stock_fundflow_factor_cy'

# 科创板资金流向因子分表
class FundFlowFactorKC(FundFlowFactorBase):
    class Meta:
        verbose_name = '资金流向因子KC'
        verbose_name_plural = '资金流向因子KC'
        db_table = 'stock_fundflow_factor_kc'

# 北京交易所资金流向因子分表
class FundFlowFactorBJ(FundFlowFactorBase):
    class Meta:
        verbose_name = '资金流向因子BJ'
        verbose_name_plural = '资金流向因子BJ'
        db_table = 'stock_fundflow_factor_bj'