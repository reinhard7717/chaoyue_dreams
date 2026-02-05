# services\fundflow_calculator.py
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional, List, Any
from datetime import datetime, date, timedelta
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import linregress, norm
import logging
import json
from dataclasses import dataclass
from collections import deque
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class CalculationContext:
    """
    计算上下文，包含所有必要的数据
    版本: V1.2
    说明: 显式增加 tick_data 字段，用于接收高频逐笔数据
    """
    stock_code: str
    trade_date: date
    current_flow_data: Dict[str, Any]  # 当前日资金流向数据
    historical_flow_data: List[Dict[str, Any]]  # 历史资金流向数据（至少30天）
    daily_basic_data: Optional[Dict[str, Any]] = None  # 每日基本信息
    minute_data_1min: Optional[pd.DataFrame] = None  # 1分钟数据
    tick_data: Optional[pd.DataFrame] = None  # [新增] Tick/分笔数据
    market_cap: Optional[float] = None  # 市值（万元）
    volume_data: Optional[List[float]] = None  # 成交量序列


class FundFlowFactorCalculator:
    """
    资金流向因子计算器
    利用每日基本信息和1分钟数据提高计算精度
    """
    
    # 常量定义
    THRESHOLD_LEVELS = {
        'small': 50000,      # 小单阈值（元）
        'medium': 200000,    # 中单阈值（元）
        'large': 1000000,    # 大单阈值（元）
    }
    
    INTENSITY_LEVEL_RANGES = {
        1: (0, 0.01),    # 低强度：占比0-1%
        2: (0.01, 0.03), # 中强度：占比1-3%
        3: (0.03, 0.05), # 高强度：占比3-5%
        4: (0.05, 1.0),  # 极高强度：占比5%以上
    }
    
    def __init__(self, context: CalculationContext):
        self.context = context
        self._validate_context()
        self._prepare_data()
        
    def _validate_context(self):
        """验证计算上下文数据完整性"""
        if not self.context.current_flow_data:
            raise ValueError("当前日资金流向数据不能为空")
        if not self.context.historical_flow_data:
            raise ValueError("历史资金流向数据不能为空")
        if len(self.context.historical_flow_data) < 20:
            logger.warning(f"历史数据不足，仅{len(self.context.historical_flow_data)}天")
    
    def _prepare_data(self):
        """
        预处理数据，计算中间变量
        版本: V1.6
        说明: 
        1. 修正 volume_series 获取逻辑。
        2. [关键修复] 显式初始化 self.tick_data，打通数据链路。
        """
        # 提取历史净流入序列
        self.net_amount_series = [
            float(data.get('net_mf_amount', 0) or 0) 
            for data in self.context.historical_flow_data
        ]
        self.net_amount_array = np.array(self.net_amount_series, dtype=np.float64)
        self.net_amount_pd_series = pd.Series(self.net_amount_array)
        # 提取历史净流入占比序列
        self.net_amount_ratio_series = [
            float(data.get('net_amount_ratio', 0) or 0)
            for data in self.context.historical_flow_data
        ]
        self.net_amount_ratio_array = np.array(self.net_amount_ratio_series, dtype=np.float64)
        # 提取历史成交量
        if self.context.volume_data:
            self.volume_series = self.context.volume_data
        else:
            self.volume_series = [
                float(data.get('vol', 0) or data.get('total_volume', 0) or 0) 
                for data in self.context.historical_flow_data
            ]
        self.volume_array = np.array(self.volume_series, dtype=np.float64)
        # 提取历史成交额序列
        self.daily_amount_series = [
            float(data.get('amount', 0) or 0) if data.get('amount') is not None else 0.0
            for data in self.context.historical_flow_data
        ]
        self.daily_amount_array = np.array(self.daily_amount_series, dtype=np.float64)
        # 提取历史收盘价序列
        self.close_series = [
            float(data.get('close', 0) or 0) if data.get('close') is not None else 0.0
            for data in self.context.historical_flow_data
        ]
        self.close_array = np.array(self.close_series, dtype=np.float64)
        # 计算市值
        if self.context.daily_basic_data:
            self.market_cap = float(self.context.daily_basic_data.get('circ_mv', 0) or 0)
        # 准备1分钟数据相关指标
        if self.context.minute_data_1min is not None and not self.context.minute_data_1min.empty:
            self._process_minute_data()
        # [关键修复] 初始化 Tick 数据
        # 即使 context.tick_data 为 None，也要赋值，避免 hasattr 检查失败
        self.tick_data = self.context.tick_data

    def _process_minute_data(self):
        """处理1分钟数据，提取日内资金流特征"""
        df = self.context.minute_data_1min
        # 计算日内收益率
        if 'close' in df.columns:
            df['returns'] = df['close'].pct_change()
            self.intraday_returns = df['returns'].dropna().tolist()
        # 计算日内成交量分布
        if 'volume' in df.columns:
            self.intraday_volume = df['volume'].tolist()
            # 计算日内成交额（如果可用）
            if 'amount' in df.columns:
                self.intraday_amount = df['amount'].tolist()
        # 计算日内波动率
        if 'returns' in df.columns:
            self.intraday_volatility = df['returns'].std() * np.sqrt(240)  # 年化
    
    # ==================== 1. 绝对量级指标计算 ====================
    def calculate_absolute_metrics(self) -> Dict[str, float]:
        """
        计算绝对量级指标
        版本: V1.2
        说明: 使用 NumPy 数组操作替代 Python 列表求和。
        """
        metrics = {}
        # 当前日净流入
        current_net = float(self.context.current_flow_data.get('net_mf_amount', 0) or 0)
        # 使用预处理的 numpy array
        net_arr = self.net_amount_array
        vol_arr = self.volume_array
        n = len(net_arr)
        # 3日、5日、10日、20日累计净流入
        windows = [3, 5, 10, 20]
        for window in windows:
            if n >= window:
                # np.sum 效率高于 sum()
                total = np.sum(net_arr[-window:])
                metrics[f'total_net_amount_{window}d'] = float(total)
                metrics[f'avg_daily_net_{window}d'] = float(total / window)
            else:
                metrics[f'total_net_amount_{window}d'] = None
                metrics[f'avg_daily_net_{window}d'] = None
        # 累计成交量
        n_vol = len(vol_arr)
        for window in [5, 10]:
            if n_vol >= window:
                # 转换为万手 (/100)
                total_vol = np.sum(vol_arr[-window:]) / 100.0
                metrics[f'total_volume_{window}d'] = float(total_vol)
            else:
                metrics[f'total_volume_{window}d'] = None
        return metrics

    # ==================== 2. 相对强度指标计算 ====================
    def calculate_relative_metrics(self) -> Dict[str, float]:
        """
        计算相对强度指标
        版本: V1.3
        说明: 
        1. 优先使用预处理的 net_amount_ratio_array 计算 MA，确保数据源一致性。
        2. 修正单位换算逻辑 (Wan/Qian -> *1000)。
        """
        metrics = {}
        current_net = float(self.context.current_flow_data.get('net_mf_amount', 0) or 0)
        # 1. 当日净流入占比
        if self.context.daily_basic_data:
            daily_amount = float(self.context.daily_basic_data.get('amount', 0) or 0)
            if daily_amount > 0:
                # 注意：这里假设 daily_basic_data 中的 amount 单位与 net_mf_amount 单位关系
                # 如果 daily_amount 是千元，current_net 是万元
                # 则 (Wan / Qian) * 1000 = %
                # 但通常 daily_basic_data 直接来自 Tushare daily，单位是千元
                # 所以这里保持 * 1000 的逻辑 (如果之前是 *100 可能偏小)
                # 修正：根据 Task 中的逻辑，Ratio = (Net/Amount) * 1000
                net_ratio = (current_net / daily_amount) * 1000.0
            else:
                net_ratio = 0.0
        else:
            # 使用历史平均成交额估算
            valid_amounts = self.daily_amount_array[-20:]
            valid_amounts = valid_amounts[valid_amounts > 0]
            if len(valid_amounts) > 0:
                avg_amount = np.mean(valid_amounts)
                net_ratio = (current_net / avg_amount * 1000.0)
            else:
                net_ratio = 0.0
        metrics['net_amount_ratio'] = float(net_ratio)
        # 2. 5日、10日均净流入占比 (使用预处理的 Ratio 数组)
        ratio_arr = self.net_amount_ratio_array
        n = len(ratio_arr)
        for window in [5, 10]:
            if n >= window:
                recent_ratios = ratio_arr[-window:]
                # 直接求平均
                avg_ratio = np.mean(recent_ratios)
                metrics[f'net_amount_ratio_ma{window}'] = float(avg_ratio)
            else:
                # 数据不足时返回 None
                metrics[f'net_amount_ratio_ma{window}'] = None
        # 资金流入强度得分
        intensity_score = self._calculate_flow_intensity(current_net, net_ratio)
        metrics['flow_intensity'] = float(intensity_score)
        # 强度分级
        metrics['intensity_level'] = self._determine_intensity_level(current_net, net_ratio)
        return metrics

    def _calculate_flow_intensity(self, net_amount: float, net_ratio: float) -> float:
        """计算资金流入强度得分"""
        # 绝对量得分（0-50分）
        if abs(net_amount) < 100:
            abs_score = 0
        elif abs(net_amount) < 1000:
            abs_score = 10
        elif abs(net_amount) < 5000:
            abs_score = 25
        elif abs(net_amount) < 10000:
            abs_score = 40
        else:
            abs_score = 50
        # 相对占比得分（0-50分）
        if abs(net_ratio) < 0.5:
            rel_score = 0
        elif abs(net_ratio) < 1:
            rel_score = 10
        elif abs(net_ratio) < 3:
            rel_score = 25
        elif abs(net_ratio) < 5:
            rel_score = 40
        else:
            rel_score = 50
        # 加权得分（绝对量60%，相对量40%）
        intensity_score = abs_score * 0.6 + rel_score * 0.4
        # 如果是净流出，得分为负
        if net_amount < 0:
            intensity_score = -intensity_score
        return intensity_score
    
    def _determine_intensity_level(self, net_amount: float, net_ratio: float) -> int:
        """确定强度分级"""
        # 使用绝对量和相对量双重标准
        if net_amount >= 0:  # 净流入
            if net_amount < 1000 or abs(net_ratio) < 0.01:
                return 1  # 低强度
            elif net_amount < 5000 or abs(net_ratio) < 0.03:
                return 2  # 中强度
            elif net_amount < 10000 or abs(net_ratio) < 0.05:
                return 3  # 高强度
            else:
                return 4  # 极高强度
        else:  # 净流出
            if abs(net_amount) < 500 or abs(net_ratio) < 0.01:
                return 1  # 低强度流出
            elif abs(net_amount) < 2000 or abs(net_ratio) < 0.03:
                return 2  # 中强度流出
            elif abs(net_amount) < 5000 or abs(net_ratio) < 0.05:
                return 3  # 高强度流出
            else:
                return 4  # 极高强度流出
    
    # ==================== 3. 主力行为模式识别 ====================
    def calculate_behavior_patterns(self) -> Dict[str, Any]:
        """计算主力行为模式"""
        patterns = {}
        # 获取最近10天的资金流数据
        recent_data = self.context.historical_flow_data[-10:] if len(self.context.historical_flow_data) >= 10 else self.context.historical_flow_data
        recent_nets = [float(data.get('net_mf_amount', 0) or 0) for data in recent_data]
        # 计算各种模式的得分
        accumulation_score = self._calculate_accumulation_score(recent_nets)
        pushing_score = self._calculate_pushing_score(recent_nets)
        distribution_score = self._calculate_distribution_score(recent_nets)
        shakeout_score = self._calculate_shakeout_score(recent_nets)
        patterns['accumulation_score'] = accumulation_score
        patterns['pushing_score'] = pushing_score
        patterns['distribution_score'] = distribution_score
        patterns['shakeout_score'] = shakeout_score
        # 确定行为模式
        pattern, confidence = self._determine_behavior_pattern(
            accumulation_score, pushing_score, distribution_score, shakeout_score
        )
        patterns['behavior_pattern'] = pattern
        patterns['pattern_confidence'] = confidence
        return patterns
    
    def _calculate_accumulation_score(self, recent_nets: List[float]) -> float:
        """
        计算建仓模式得分
        版本: V1.1
        说明: 使用 NumPy 向量化替代 Python 循环和列表推导。
        """
        # 转换为 array (如果传入的是 list)
        arr = np.array(recent_nets, dtype=np.float64)
        n = len(arr)
        if n < 5:
            return 0.0
        # 连续净流入天数 (向量化: 大于0的个数)
        positive_days = np.sum(arr > 0)
        continuity_score = (positive_days / n) * 100
        # 流入稳定性
        if n >= 3:
            # 取最后3个
            last_3 = arr[-3:]
            volatility = np.std(last_3) / (np.abs(np.mean(last_3)) + 1e-6)
            stability_score = max(0.0, 100 - volatility * 100)
        else:
            stability_score = 50.0
        # 流出比例 (向量化: 布尔掩码)
        inflows = arr[arr > 0]
        outflows = arr[arr < 0]
        total_inflow = np.sum(inflows) if len(inflows) > 0 else 0.0
        total_outflow = np.sum(np.abs(outflows)) if len(outflows) > 0 else 0.0
        outflow_ratio = total_outflow / (total_inflow + total_outflow + 1e-6)
        outflow_score = max(0.0, 100 - outflow_ratio * 200)
        return float(continuity_score * 0.4 + stability_score * 0.3 + outflow_score * 0.3)

    def _calculate_pushing_score(self, recent_nets: List[float]) -> float:
        """
        计算拉升模式得分
        版本: V1.1
        说明: 使用 np.diff 计算加速度，使用向量过滤计算均值。
        """
        arr = np.array(recent_nets, dtype=np.float64)
        n = len(arr)
        if n < 3:
            return 0.0
        # 最近3天净流入加速度 (二阶差分)
        # acceleration = (d[2]-d[1]) - (d[1]-d[0]) = d[2] - 2d[1] + d[0]
        # 等同于 np.diff(arr, n=2)
        recent_3 = arr[-3:]
        acc_val = recent_3[2] - 2*recent_3[1] + recent_3[0]
        accel_score = min(100.0, max(0.0, 50 + acc_val / 1000 * 10))
        # 流入强度 (最近5天)
        recent_5 = arr[-5:]
        pos_mask = recent_5 > 0
        if np.any(pos_mask):
            avg_inflow = np.mean(recent_5[pos_mask])
        else:
            avg_inflow = 0.0
        intensity_score = min(100.0, avg_inflow / 5000 * 100)
        # 流出可控性
        neg_mask = recent_5 < 0
        if np.any(neg_mask):
            max_outflow = np.max(np.abs(recent_5[neg_mask]))
        else:
            max_outflow = 0.0
        if avg_inflow > 0:
            outflow_control = max(0.0, 100 - (max_outflow / avg_inflow) * 100)
        else:
            outflow_control = 100.0
        return float(accel_score * 0.4 + intensity_score * 0.3 + outflow_control * 0.3)

    def _calculate_distribution_score(self, recent_nets: List[float]) -> float:
        """
        计算派发模式得分
        版本: V1.1
        说明: 
        1. 使用 NumPy 向量化计算流出天数占比和均值。
        2. 使用向量化逻辑计算最大连续流出天数，避免 Python 循环。
        """
        arr = np.array(recent_nets, dtype=np.float64)
        n = len(arr)
        if n < 5:
            return 0.0
        # 净流出天数占比
        neg_mask = arr < 0
        negative_days = np.sum(neg_mask)
        negative_ratio_score = (negative_days / n) * 100
        # 流出绝对量
        if negative_days > 0:
            avg_outflow = np.mean(np.abs(arr[neg_mask]))
        else:
            avg_outflow = 0.0
        outflow_amount_score = min(100.0, avg_outflow / 3000 * 100)
        # 流出持续性（连续流出天数）- 向量化实现
        # 1. 构造 padded 数组以处理边界
        padded = np.concatenate(([False], neg_mask, [False]))
        # 2. 找变化点 (0->1 为 start, 1->0 为 end)
        diffs = np.diff(padded.astype(int))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        if len(starts) > 0:
            # 计算所有连续段的长度，取最大值
            max_consecutive = np.max(ends - starts)
        else:
            max_consecutive = 0
        continuity_score = min(100.0, max_consecutive * 20)
        return float(negative_ratio_score * 0.4 + outflow_amount_score * 0.3 + continuity_score * 0.3)

    def _calculate_shakeout_score(self, recent_nets: List[float]) -> float:
        """
        计算洗盘模式得分
        版本: V1.1
        说明: 使用 np.diff 和 np.sign 向量化计算符号变化次数。
        """
        arr = np.array(recent_nets, dtype=np.float64)
        n = len(arr)
        if n < 5:
            return 0.0
        # 流出但绝对量不大
        avg_net = np.mean(arr)
        avg_abs = np.mean(np.abs(arr))
        if avg_net < 0 and avg_abs < 2000:
            amount_score = 80.0
        else:
            amount_score = 20.0
        # 波动性
        volatility = np.std(arr) / (np.abs(avg_net) + 1e-6)
        volatility_score = min(100.0, volatility * 50)
        # 流出流入交替 (符号变化)
        # np.sign 返回 -1, 0, 1
        # np.diff 计算相邻差，如果不为0则说明符号变了 (或者遇到了0)
        # 过滤掉0的情况，只看正负切换
        signs = np.sign(arr)
        # 移除0值以便准确计算正负交替 (可选，视业务逻辑而定，这里简化处理)
        signs_no_zero = signs[signs != 0]
        if len(signs_no_zero) >= 2:
            # 计算相邻符号乘积，负数表示异号
            changes = signs_no_zero[1:] * signs_no_zero[:-1]
            sign_changes = np.sum(changes < 0)
            change_score = min(100.0, sign_changes / n * 200)
        else:
            change_score = 0.0
        return float(amount_score * 0.5 + volatility_score * 0.3 + change_score * 0.2)

    def _determine_behavior_pattern(self, acc_score, push_score, dist_score, shake_score) -> Tuple[str, float]:
        """确定行为模式和置信度"""
        scores = {
            'ACCUMULATION': acc_score,
            'PUSHING': push_score,
            'DISTRIBUTION': dist_score,
            'SHAKEOUT': shake_score,
        }
        # 找到最高分
        max_pattern = max(scores.items(), key=lambda x: x[1])
        # 计算置信度（最高分与其他分的差距）
        other_scores = [v for k, v in scores.items() if k != max_pattern[0]]
        if other_scores:
            gap = max_pattern[1] - max(other_scores)
            confidence = min(100, max(0, 50 + gap * 2))
        else:
            confidence = 50
        # 如果所有得分都低于阈值，标记为不明
        if max_pattern[1] < 30:
            return 'UNCLEAR', confidence
        return max_pattern[0], confidence
    
    # ==================== 4. 资金流向质量评估 ====================
    def calculate_flow_quality(self) -> Dict[str, Any]:
        """计算资金流向质量指标"""
        quality = {}
        current_net = float(self.context.current_flow_data.get('net_mf_amount', 0) or 0)
        # 流出质量评分
        quality['outflow_quality'] = self._calculate_outflow_quality(current_net)
        # 连续净流入天数
        quality['inflow_persistence'] = self._calculate_inflow_persistence()
        # 大单异动检测
        anomaly, intensity = self._detect_large_order_anomaly()
        quality['large_order_anomaly'] = anomaly
        quality['anomaly_intensity'] = intensity
        # 分档资金一致性
        quality['flow_consistency'] = self._calculate_flow_consistency()
        # 资金流稳定性
        quality['flow_stability'] = self._calculate_flow_stability()
        return quality
    
    def _calculate_outflow_quality(self, current_net: float) -> float:
        """
        计算流出质量评分
        版本: V1.1
        说明: 使用 NumPy 布尔索引过滤和向量化计算排名，替代 scipy 调用。
        """
        if current_net >= 0:
            return 100.0  # 净流入
        # 获取历史流出数据 (向量化过滤)
        # 取最近20天
        recent_20 = self.net_amount_array[-20:]
        # 筛选小于0的，并取绝对值
        historical_outflows = np.abs(recent_20[recent_20 < 0])
        if len(historical_outflows) == 0:
            return 50.0
        current_outflow = abs(current_net)
        # 计算百分位 (Rank)
        # 统计小于当前流出量的个数
        count_less = np.sum(historical_outflows < current_outflow)
        percentile = (count_less / len(historical_outflows)) * 100
        # 质量评分逻辑
        if current_outflow < 100:
            return 90.0
        elif percentile < 30:
            return 80.0
        elif percentile < 70:
            return 60.0
        else:
            return 30.0

    def _calculate_inflow_persistence(self) -> int:
        """
        计算连续净流入天数
        版本: V1.1
        说明: 使用 NumPy 向量化查找，避免 Python 循环。
        """
        # 使用预处理好的 array
        arr = self.net_amount_array
        if len(arr) == 0:
            return 0
        # 倒序数组
        rev_arr = arr[::-1]
        # 找到第一个非正数 (<= 0) 的索引
        # np.where 返回 tuple, 取第一个元素(索引数组)
        not_positive_indices = np.where(rev_arr <= 0)[0]
        if len(not_positive_indices) > 0:
            # 第一个非正数的位置即为连续正数的个数
            return int(not_positive_indices[0])
        else:
            # 全是正数
            return len(arr)

    def _detect_large_order_anomaly(self) -> Tuple[bool, float]:
        """检测大单异动"""
        if len(self.net_amount_series) < 10:
            return False, 0
        # 使用3倍标准差原则检测异常
        recent_nets = self.net_amount_series[-10:]
        mean_val = np.mean(recent_nets)
        std_val = np.std(recent_nets)
        current_net = float(self.context.current_flow_data.get('net_mf_amount', 0) or 0)
        z_score = (current_net - mean_val) / (std_val + 1e-6)
        # 异常判断
        is_anomaly = abs(z_score) > 3
        # 异动强度
        anomaly_intensity = min(100, abs(z_score) * 20)
        return is_anomaly, anomaly_intensity
    
    def _calculate_flow_consistency(self) -> float:
        """
        计算分档资金一致性
        版本: V1.2
        说明: 
        1. 修正原逻辑计算的是时间序列一致性（且只有0/50/100）的问题。
        2. 改为计算当日各分档资金（超大、大、中）的方向一致性。
        3. 算法：|Sum(Net)| / Sum(Abs(Net)) * 100。
           反映各路资金是否形成合力。
        """
        data = self.context.current_flow_data
        if not data:
            return 50.0
        # 1. 计算各档净流入
        try:
            # 确保数据存在且为 float
            def get_val(key):
                return float(data.get(key, 0) or 0)
            net_elg = get_val('buy_elg_amount') - get_val('sell_elg_amount')
            net_lg = get_val('buy_lg_amount') - get_val('sell_lg_amount')
            net_md = get_val('buy_md_amount') - get_val('sell_md_amount')
            # 小单通常是被动资金，计算一致性时主要看主力(ELG, LG)和中单(MD)的合力情况
            # 这里将 ELG, LG, MD 视为主要市场力量
            components = [net_elg, net_lg, net_md]
            # 2. 计算合力程度
            # 分子：净流入代数和的绝对值 (合力后的大小)
            # 例如: 10 + 10 + 10 = 30; 10 + 10 - 20 = 0
            abs_sum_net = abs(sum(components))
            # 分母：各分量绝对值之和 (总活跃资金量)
            # 例如: |10| + |10| + |10| = 30; |10| + |10| + |-20| = 40
            sum_abs_net = sum(abs(x) for x in components)
            if sum_abs_net == 0:
                return 50.0
            # 3. 计算得分 (0-100)
            # 如果全部同向，分子=分母，得分100
            # 如果完全抵消，分子=0，得分0
            # 结果是连续的 float 值
            consistency_score = (abs_sum_net / sum_abs_net) * 100.0
            return float(consistency_score)
        except Exception as e:
            print(f"计算资金一致性出错: {e}")
            return 50.0

    def _calculate_flow_stability(self) -> float:
        """
        计算资金流稳定性
        版本: V1.0
        说明: 基于最近20天净流入的变异系数 (CV) 计算。
        """
        # 使用预处理的 array
        arr = self.net_amount_array
        n = len(arr)
        if n < 5:
            return 50.0
        # 取最近20天 (如果不足20天则取全部)
        window = min(n, 20)
        recent = arr[-window:]
        # 计算均值和标准差
        mean_val = np.mean(recent)
        std_val = np.std(recent)
        # 如果均值接近0，说明多空博弈非常激烈且平衡，或者没量
        # 这种情况下稳定性较低
        if abs(mean_val) < 1e-6:
            # 如果标准差也很小，说明是死水，稳定性高？
            # 这里假设没量就是不稳定（容易被少量资金打破）
            if std_val < 1e-6:
                return 100.0
            else:
                return 20.0
        # 变异系数 CV = Std / |Mean|
        # CV 越大，说明波动相对于均值越大，稳定性越差
        cv = std_val / (abs(mean_val) + 1e-6)
        # 映射到 0-100
        # 经验规则：
        # CV <= 0.5 -> 非常稳定 (得分 75-100)
        # CV = 1.0 -> 一般 (得分 50)
        # CV >= 2.0 -> 不稳定 (得分 < 0 -> 截断为0)
        stability_score = 100.0 - (cv * 50.0)
        return float(max(0.0, min(100.0, stability_score)))

    # ==================== 5. 多周期资金共振指标 ====================
    def calculate_multi_period_sync(self) -> Dict[str, float]:
        """
        计算多周期资金共振指标
        版本: V1.1
        说明: 复用预处理的 Pandas Series 对象，减少对象创建开销。
        """
        sync_metrics = {}
        if len(self.net_amount_series) < 20:
            for key in ['daily_weekly_sync', 'daily_monthly_sync', 
                       'short_mid_sync', 'mid_long_sync']:
                sync_metrics[key] = None
            return sync_metrics
        # 日线数据（最近5天）
        daily_data = self.net_amount_series[-5:]
        # 优化：复用 self.net_amount_pd_series
        pd_series = self.net_amount_pd_series
        # 周线数据（5日移动平均）
        # 注意：rolling(5) 前面会有4个NaN，dropna() 会去掉它们
        weekly_data = pd_series.rolling(5).mean().dropna().tolist()[-20:] # 取最近20个有效值
        # 月线数据（20日移动平均）
        monthly_data = pd_series.rolling(20).mean().dropna().tolist()[-60:] # 取最近60个有效值
        # 计算同步度
        # 确保数据长度足够
        if len(daily_data) >= 3 and len(weekly_data) >= 3:
            sync_metrics['daily_weekly_sync'] = self._calculate_sync_score(daily_data[-3:], weekly_data[-3:])
        else:
            sync_metrics['daily_weekly_sync'] = None
        if len(daily_data) >= 3 and len(monthly_data) >= 3:
            sync_metrics['daily_monthly_sync'] = self._calculate_sync_score(daily_data[-3:], monthly_data[-3:])
        else:
            sync_metrics['daily_monthly_sync'] = None
        # 短中期（5日 vs 20日）
        # 5日均线序列
        short_term_series = pd_series.rolling(5).mean()
        short_term = short_term_series.dropna().tolist()[-3:]
        # 20日均线序列
        mid_term_series = pd_series.rolling(20).mean()
        mid_term = mid_term_series.dropna().tolist()[-3:]
        if len(short_term) >= 3 and len(mid_term) >= 3:
            sync_metrics['short_mid_sync'] = self._calculate_sync_score(short_term, mid_term)
        else:
            sync_metrics['short_mid_sync'] = None
        # 中长期（20日 vs 60日）
        mid_term_long = mid_term_series.dropna().tolist()[-3:]
        long_term = pd_series.rolling(60).mean().dropna().tolist()[-3:]
        if len(mid_term_long) >= 3 and len(long_term) >= 3:
            sync_metrics['mid_long_sync'] = self._calculate_sync_score(mid_term_long, long_term)
        else:
            sync_metrics['mid_long_sync'] = None
        return sync_metrics

    def _calculate_sync_score(self, series1: List[float], series2: List[float]) -> float:
        """
        计算两个序列的同步度得分
        版本: V1.1
        说明: 使用 NumPy 向量化计算相关性和方向一致性。
        """
        # 转换为 array
        s1 = np.array(series1, dtype=np.float64)
        s2 = np.array(series2, dtype=np.float64)
        n = len(s1)
        if n != len(s2) or n < 2:
            return 50.0
        # 计算相关性
        try:
            corr_matrix = np.corrcoef(s1, s2)
            if np.isnan(corr_matrix).any():
                correlation = 0.0
            else:
                correlation = corr_matrix[0, 1]
        except Exception:
            correlation = 0.0
        # 计算方向一致性 (向量化)
        # diff 计算相邻差
        diff1 = np.diff(s1)
        diff2 = np.diff(s2)
        # sign 获取方向 (-1, 0, 1)
        dir1 = np.sign(diff1)
        dir2 = np.sign(diff2)
        # 比较方向是否相同
        matches = (dir1 == dir2)
        match_count = np.sum(matches)
        direction_score = (match_count / len(diff1) * 100) if len(diff1) > 0 else 50.0
        # 综合得分
        sync_score = (correlation * 100 * 0.6 + direction_score * 0.4)
        return float(max(0.0, min(100.0, sync_score)))

    # ==================== 6. 趋势动量指标 ====================
    def calculate_trend_momentum(self) -> Dict[str, float]:
        """计算趋势动量指标"""
        momentum = {}
        if len(self.net_amount_series) < 5:
            for key in ['flow_momentum_5d', 'flow_momentum_10d', 'flow_acceleration',
                       'uptrend_strength', 'downtrend_strength']:
                momentum[key] = None
            return momentum
        # 5日动量
        if len(self.net_amount_series) >= 5:
            momentum['flow_momentum_5d'] = (self.net_amount_series[-1] - self.net_amount_series[-5]) / (abs(self.net_amount_series[-5]) + 1e-6)
        # 10日动量
        if len(self.net_amount_series) >= 10:
            momentum['flow_momentum_10d'] = (self.net_amount_series[-1] - self.net_amount_series[-10]) / (abs(self.net_amount_series[-10]) + 1e-6)
        # 加速度（二阶导数）
        if len(self.net_amount_series) >= 3:
            recent = self.net_amount_series[-3:]
            acceleration = (recent[2] - 2*recent[1] + recent[0]) / (abs(recent[0]) + 1e-6)
            momentum['flow_acceleration'] = acceleration
        # 趋势强度
        momentum['uptrend_strength'], momentum['downtrend_strength'] = self._calculate_trend_strength()
        return momentum
    
    def _calculate_trend_strength(self) -> Tuple[float, float]:
        """
        计算趋势强度
        版本: V1.1
        说明: 
        1. 替换 scipy.stats.linregress 为 numpy.polyfit 和 numpy.corrcoef。
        2. 避免计算不需要的 p-value 和 stderr，提高计算效率。
        """
        if len(self.net_amount_series) < 10:
            return 0.0, 0.0
        # 获取最近10天数据
        y = np.array(self.net_amount_series[-10:])
        x = np.arange(len(y))
        # 优化：使用 numpy 计算斜率和相关系数
        # polyfit(deg=1) 返回 [slope, intercept]
        try:
            slope, _ = np.polyfit(x, y, 1)
            # 计算相关系数矩阵，取 [0,1] 元素
            r_matrix = np.corrcoef(x, y)
            # 处理常数序列导致的相关系数为 NaN 的情况
            if np.isnan(r_matrix).any():
                r_value = 0
            else:
                r_value = r_matrix[0, 1]
        except Exception:
            return 0.0, 0.0
        # 上升趋势强度
        if slope > 0:
            uptrend_strength = min(100, r_value**2 * 100)
            downtrend_strength = 0.0
        else:
            uptrend_strength = 0.0
            downtrend_strength = min(100, r_value**2 * 100)
        return float(uptrend_strength), float(downtrend_strength)

    # ==================== 7. 量价背离指标 ====================
    def calculate_divergence_metrics(self) -> Dict[str, Any]:
        """
        计算量价背离指标
        版本: V1.1
        说明: 利用预处理的 close_series 优化计算效率，避免循环内字典查找。
        """
        divergence = {}
        # 需要价格数据
        if not self.context.daily_basic_data or 'close' not in self.context.daily_basic_data:
            divergence['price_flow_divergence'] = None
            divergence['divergence_type'] = 'NONE'
            divergence['divergence_strength'] = 0
            return divergence
        # 获取历史价格和资金流数据
        # 优化：直接使用预处理的序列，取最近10天
        # 注意：close_series 和 net_amount_series 是对齐的
        limit = 10
        price_history = self.close_series[-limit:]
        flow_history = self.net_amount_series[-limit:]
        # 过滤掉价格为0的数据（如果有）
        valid_indices = [i for i, p in enumerate(price_history) if p > 0]
        if len(valid_indices) < 3:
            divergence['price_flow_divergence'] = None
            divergence['divergence_type'] = 'NONE'
            divergence['divergence_strength'] = 0
            return divergence
        price_history = [price_history[i] for i in valid_indices]
        flow_history = [flow_history[i] for i in valid_indices]
        # 计算价格和资金流的趋势
        price_trend = self._calculate_trend_direction(price_history[-3:])
        flow_trend = self._calculate_trend_direction(flow_history[-3:])
        # 判断背离类型
        if price_trend < 0 and flow_trend > 0:  # 价格下跌但资金流入
            divergence_type = 'BULLISH'
            divergence_strength = abs(price_trend) * abs(flow_trend) * 25
        elif price_trend > 0 and flow_trend < 0:  # 价格上涨但资金流出
            divergence_type = 'BEARISH'
            divergence_strength = abs(price_trend) * abs(flow_trend) * 25
        else:
            divergence_type = 'NONE'
            divergence_strength = 0
        # 计算背离度
        divergence_score = abs(price_trend - flow_trend) * 50
        divergence['price_flow_divergence'] = divergence_score
        divergence['divergence_type'] = divergence_type
        divergence['divergence_strength'] = min(100, divergence_strength)
        return divergence

    def _calculate_trend_direction(self, series: List[float]) -> float:
        """
        计算序列趋势方向（-1到1）
        版本: V1.1
        说明: 使用 NumPy 向量化计算均值和标准差。
        """
        arr = np.array(series, dtype=np.float64)
        if len(arr) < 2:
            return 0.0
        # 简单趋势判断: 差分序列的均值除以标准差
        changes = np.diff(arr)
        avg_change = np.mean(changes)
        std_change = np.std(changes) + 1e-6
        # 标准化趋势值
        trend = avg_change / std_change
        return float(np.clip(trend, -1.0, 1.0))

    # ==================== 8. 结构分析指标 ====================
    def calculate_structure_metrics(self) -> Dict[str, Any]:
        """计算结构分析指标"""
        structure = {}
        if len(self.net_amount_series) < 10:
            structure['flow_peak_value'] = None
            structure['days_since_last_peak'] = None
            structure['flow_support_level'] = None
            structure['flow_resistance_level'] = None
            return structure
        # 寻找峰值
        peaks, properties = find_peaks(self.net_amount_series, height=0)
        if len(peaks) > 0:
            last_peak_idx = peaks[-1]
            structure['flow_peak_value'] = self.net_amount_series[last_peak_idx]
            structure['days_since_last_peak'] = len(self.net_amount_series) - 1 - last_peak_idx
        else:
            structure['flow_peak_value'] = max(self.net_amount_series) if self.net_amount_series else None
            structure['days_since_last_peak'] = None
        # 计算支撑位和阻力位（使用分位数）
        sorted_flows = sorted(self.net_amount_series[-20:])
        if len(sorted_flows) >= 5:
            structure['flow_support_level'] = np.percentile(sorted_flows, 20)  # 20%分位数作为支撑
            structure['flow_resistance_level'] = np.percentile(sorted_flows, 80)  # 80%分位数作为阻力
        else:
            structure['flow_support_level'] = None
            structure['flow_resistance_level'] = None
        return structure
    
    # ==================== 9. 统计特征指标 ====================
    def calculate_statistical_metrics(self) -> Dict[str, float]:
        """
        计算统计特征指标
        版本: V1.1
        说明: 
        1. 消除重复计算：复用20日窗口的均值和标准差计算 Z-score 和波动率。
        2. 统一处理除零保护。
        """
        stats_metrics = {}
        if len(self.net_amount_series) < 10:
            for key in ['flow_zscore', 'flow_percentile', 'flow_volatility_10d', 'flow_volatility_20d']:
                stats_metrics[key] = None
            return stats_metrics
        current_net = float(self.context.current_flow_data.get('net_mf_amount', 0) or 0)
        # 优化：预先计算20日统计量，供 Z-score 和 Volatility-20d 复用
        recent_20 = self.net_amount_series[-20:]
        mean_20 = np.mean(recent_20)
        std_20 = np.std(recent_20)
        abs_mean_20 = abs(mean_20) + 1e-6
        # Z分数 (使用20日均值和标准差)
        stats_metrics['flow_zscore'] = (current_net - mean_20) / (std_20 + 1e-6) if std_20 > 0 else 0
        # 百分位
        stats_metrics['flow_percentile'] = stats.percentileofscore(recent_20, current_net)
        # 波动率 10d
        if len(self.net_amount_series) >= 10:
            recent_10 = self.net_amount_series[-10:]
            # 10日波动率仍需单独计算
            stats_metrics['flow_volatility_10d'] = np.std(recent_10) / (abs(np.mean(recent_10)) + 1e-6)
        else:
            stats_metrics['flow_volatility_10d'] = None
        # 波动率 20d (复用之前的计算结果)
        if len(self.net_amount_series) >= 20:
            stats_metrics['flow_volatility_20d'] = std_20 / abs_mean_20
        else:
            stats_metrics['flow_volatility_20d'] = None
        return stats_metrics

    # ==================== 10. 预测指标 ====================
    def calculate_prediction_metrics(self) -> Dict[str, float]:
        """计算预测指标"""
        prediction = {}
        if len(self.net_amount_series) < 10:
            for key in ['expected_flow_next_1d', 'flow_forecast_confidence',
                       'uptrend_continuation_prob', 'reversal_prob']:
                prediction[key] = None
            return prediction
        # 使用简单移动平均预测明日净流入
        recent_nets = self.net_amount_series[-5:]
        prediction['expected_flow_next_1d'] = np.mean(recent_nets)
        # 预测置信度（基于近期稳定性）
        volatility = np.std(recent_nets) / (abs(np.mean(recent_nets)) + 1e-6)
        prediction['flow_forecast_confidence'] = max(0, 100 - volatility * 50)
        # 趋势延续概率
        if len(self.net_amount_series) >= 3:
            recent_trend = self.net_amount_series[-3:]
            is_uptrend = recent_trend[-1] > recent_trend[0]
            # 计算历史趋势延续概率
            historical_continuation = self._calculate_historical_continuation_prob()
            if is_uptrend:
                prediction['uptrend_continuation_prob'] = historical_continuation
                prediction['reversal_prob'] = 100 - historical_continuation
            else:
                prediction['uptrend_continuation_prob'] = 0
                prediction['reversal_prob'] = historical_continuation
        else:
            prediction['uptrend_continuation_prob'] = 50
            prediction['reversal_prob'] = 50
        return prediction
    
    def _calculate_historical_continuation_prob(self) -> float:
        """
        计算历史趋势延续概率
        版本: V1.1
        说明: 
        1. 使用 Numpy 向量化操作替代 Python 循环，提高计算效率。
        2. 逻辑：计算相邻两天资金流向乘积，大于0表示同向（延续）。
        """
        if len(self.net_amount_series) < 10:
            return 50.0
        # 转换为 numpy 数组 (如果还不是)
        series = np.array(self.net_amount_series)
        # 优化：向量化计算
        # series[1:] * series[:-1] 计算相邻元素的乘积
        # > 0 表示同号（同向）
        consecutive_products = series[1:] * series[:-1]
        continuation_count = np.sum(consecutive_products > 0)
        total_count = len(consecutive_products)
        if total_count > 0:
            probability = (continuation_count / total_count) * 100
        else:
            probability = 50.0
        return float(probability)

    # ==================== 11. 复合综合指标 ====================
    def calculate_comprehensive_metrics(self, all_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """计算复合综合指标"""
        comprehensive = {}
        # 综合评分（加权平均）
        weights = {
            'flow_intensity': 0.12,
            'pattern_confidence': 0.12,
            'flow_consistency': 0.08,
            'flow_stability': 0.08,
            'daily_weekly_sync': 0.08,
            'uptrend_strength': 0.08,
            'downtrend_strength': 0.08,
            'divergence_strength': 0.08,
            'flow_forecast_confidence': 0.08,
            # tick增强指标权重
            'tick_large_order_net': 0.05,
            'flow_cluster_intensity': 0.05,
            'closing_flow_intensity': 0.05,
            'flow_efficiency': 0.05,
        }
        total_score = 0
        total_weight = 0
        for metric, weight in weights.items():
            if metric in all_metrics and all_metrics[metric] is not None:
                value = float(all_metrics[metric])
                # 标准化到0-100
                if metric in ['downtrend_strength', 'divergence_strength']:
                    # 这些指标需要特殊处理
                    normalized = value
                else:
                    normalized = max(0, min(100, value))
                total_score += normalized * weight
                total_weight += weight
        if total_weight > 0:
            comprehensive_score = total_score / total_weight
        else:
            comprehensive_score = 50
        comprehensive['comprehensive_score'] = comprehensive_score
        # 交易信号
        signal, signal_strength = self._generate_trading_signal(all_metrics, comprehensive_score)
        comprehensive['trading_signal'] = signal
        comprehensive['signal_strength'] = signal_strength
        return comprehensive
    
    def _generate_trading_signal(self, metrics: Dict[str, Any], comp_score: float) -> Tuple[str, float]:
        """生成交易信号"""
        # 收集关键指标
        flow_intensity = metrics.get('flow_intensity', 0) or 0
        pattern = metrics.get('behavior_pattern', '')
        confidence = metrics.get('pattern_confidence', 0) or 0
        divergence_type = metrics.get('divergence_type', 'NONE')
        # 信号强度基础分
        base_strength = comp_score
        # 根据模式调整
        if pattern == 'ACCUMULATION' and confidence > 70:
            signal = 'STRONG_BUY'
            base_strength += 20
        elif pattern == 'PUSHING' and confidence > 60:
            signal = 'BUY'
            base_strength += 10
        elif pattern == 'DISTRIBUTION' and confidence > 70:
            signal = 'STRONG_SELL'
            base_strength += 20
        elif pattern == 'SHAKEOUT' and confidence > 60:
            signal = 'HOLD'
        elif divergence_type == 'BULLISH' and metrics.get('divergence_strength', 0) > 70:
            signal = 'BUY'
            base_strength += 15
        elif divergence_type == 'BEARISH' and metrics.get('divergence_strength', 0) > 70:
            signal = 'SELL'
            base_strength += 15
        elif flow_intensity > 70:
            signal = 'BUY'
        elif flow_intensity < -70:
            signal = 'SELL'
        else:
            signal = 'NEUTRAL'
        # 限制信号强度在0-100之间
        signal_strength = max(0, min(100, base_strength))
        return signal, signal_strength

    # ==================== 12. 基于Tick数据的增强指标计算 ====================
    def calculate_tick_enhanced_metrics(self) -> Dict[str, Any]:
        """
        基于Tick数据计算增强的资金流向指标
        版本: V1.3
        说明: 
        1. [关键修复] 增加时区标准化逻辑，统一转换为北京时间，解决时区比较错误。
        2. 优化数据预处理流程。
        """
        tick_metrics = {}
        # 初始化所有字段为 None
        tick_fields = [
            'tick_large_order_net', 'tick_large_order_count', 'flow_impact_ratio',
            'flow_persistence_minutes', 'intraday_flow_momentum', 
            'flow_acceleration_intraday', 'flow_cluster_intensity',
            'flow_cluster_duration', 'high_freq_flow_divergence',
            'vwap_deviation', 'flow_efficiency', 'closing_flow_ratio',
            'closing_flow_intensity', 'high_freq_flow_skewness',
            'high_freq_flow_kurtosis', 'morning_flow_ratio',
            'afternoon_flow_ratio', 'stealth_flow_ratio'
        ]
        for field in tick_fields:
            tick_metrics[field] = None
        tick_metrics['intraday_flow_distribution'] = None

        # 检查数据是否存在
        if not hasattr(self, 'tick_data') or self.tick_data is None or self.tick_data.empty:
            return tick_metrics

        try:
            # 预处理tick数据
            df = self.tick_data.copy()
            
            # 列名标准化映射
            col_map = {
                'time': 'trade_time',
                'vol': 'volume',
                'v': 'volume',
                'p': 'price',
                'amt': 'amount',
                'money': 'amount'
            }
            df.rename(columns=col_map, inplace=True)
            
            # 确保必要列存在
            if 'trade_time' not in df.columns:
                if isinstance(df.index, pd.DatetimeIndex):
                    df['trade_time'] = df.index
                else:
                    logger.warning(f"Tick数据缺少 trade_time 列且索引不是时间: {df.columns}")
                    return tick_metrics
            
            if 'price' not in df.columns or 'volume' not in df.columns:
                logger.warning(f"Tick数据缺少 price 或 volume 列: {df.columns}")
                return tick_metrics

            # 补全 amount (如果缺失)
            if 'amount' not in df.columns:
                df['amount'] = df['price'] * df['volume'] * 100
            
            # 标准化买卖方向 type
            if 'type' in df.columns:
                df['type'] = df['type'].astype(str).str.upper()
                type_map = {
                    '买盘': 'B', '卖盘': 'S', 
                    'BUY': 'B', 'SELL': 'S',
                    '0': 'S', '1': 'B', 
                    '2': 'M' 
                }
                df['type'] = df['type'].map(lambda x: type_map.get(x, x))
            else:
                logger.warning("Tick数据缺少 type 列，无法计算资金流方向")
                return tick_metrics

            # 过滤中性盘
            df = df[df['type'].isin(['B', 'S'])]
            if df.empty:
                return tick_metrics

            # 确保 trade_time 是 datetime 类型
            df['trade_time'] = pd.to_datetime(df['trade_time'])
            
            # [新增] 时区标准化：统一转换为北京时间 (Asia/Shanghai)
            # 解决 Invalid comparison between dtype=datetime64[ns, UTC] and Timestamp 问题
            # 同时确保后续 dt.hour 等逻辑基于北京时间
            import pytz
            bj_tz = pytz.timezone('Asia/Shanghai')
            
            if df['trade_time'].dt.tz is not None:
                # 如果已有时区 (如 UTC)，转换为北京时间
                df['trade_time'] = df['trade_time'].dt.tz_convert(bj_tz)
            else:
                # 如果无时区，假定为北京时间并本地化
                df['trade_time'] = df['trade_time'].dt.tz_localize(bj_tz)

            # 如果只有时间没有日期，加上当前日期
            if df['trade_time'].iloc[0].year == 1900:
                current_date = self.context.trade_date
                # 注意：这里替换年份后，时区信息会保留
                df['trade_time'] = df['trade_time'].apply(
                    lambda t: t.replace(year=current_date.year, month=current_date.month, day=current_date.day)
                )

            # 1. 计算日内资金流分布
            tick_metrics.update(self._calculate_intraday_flow_distribution(df))
            # 2. 高频大单识别
            tick_metrics.update(self._detect_high_freq_large_orders(df))
            # 3. 资金冲击特征
            tick_metrics.update(self._calculate_flow_impact_features(df))
            # 4. 日内资金动量
            tick_metrics.update(self._calculate_intraday_momentum(df))
            # 5. 资金聚类特征
            tick_metrics.update(self._calculate_flow_cluster_features(df))
            # 6. 高频资金分歧度
            tick_metrics.update(self._calculate_high_freq_divergence(df))
            # 7. VWAP偏离度
            tick_metrics.update(self._calculate_vwap_deviation(df))
            # 8. 资金流入效率
            tick_metrics.update(self._calculate_flow_efficiency(df))
            # 9. 尾盘资金特征
            tick_metrics.update(self._calculate_closing_flow_features(df))
            # 10. 高频统计特征
            tick_metrics.update(self._calculate_high_freq_statistics(df))
            # 11. 资金时段分布
            tick_metrics.update(self._calculate_time_period_distribution(df))
            # 12. 主力隐蔽性指标
            tick_metrics.update(self._calculate_stealth_flow_indicators(df))
            
        except Exception as e:
            logger.error(f"计算tick增强指标时发生未捕获异常: {e}", exc_info=True)
            
        return tick_metrics

    def _calculate_intraday_flow_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        计算日内资金流分布特征
        版本: V1.2
        说明: 
        1. 直接使用已标准化的 trade_time 列。
        2. 优化分桶逻辑，匹配 A 股交易时段。
        """
        distribution = {}
        # 按半小时划分交易时段
        # df['trade_time'] 已经是 datetime 类型且为北京时间
        minutes_since_midnight = df['trade_time'].dt.hour * 60 + df['trade_time'].dt.minute
        
        # 定义 bins 和 labels
        # 9:30=570, 10:00=600, 10:30=630, 11:00=660, 11:30=690
        # 13:00=780, 13:30=810, 14:00=840, 14:30=870, 15:00=900
        bins = [0, 570, 600, 630, 660, 690, 780, 810, 840, 870, 1440]
        labels = ['Before', '9:30-10:00', '10:00-10:30', '10:30-11:00', '11:00-11:30', 
                  'Noon', '13:00-13:30', '13:30-14:00', '14:00-14:30', '14:30-15:00']
        
        df['time_bucket'] = pd.cut(minutes_since_midnight, bins=bins, labels=labels, right=True)

        # 计算各时段资金净流入
        bucket_flows = df.groupby('time_bucket', observed=True).apply(
            lambda x: (x[x['type'] == 'B']['amount'].sum() - 
                      x[x['type'] == 'S']['amount'].sum()) / 10000  # 转换为万元
        ).to_dict()
        
        # 过滤掉非交易时段 (Before, Noon)
        valid_buckets = [l for l in labels if ':' in l]
        bucket_flows = {k: v for k, v in bucket_flows.items() if k in valid_buckets}

        # 计算各时段占比
        total_flow = sum([abs(v) for v in bucket_flows.values()])
        if total_flow > 0:
            bucket_ratios = {k: v/total_flow*100 for k, v in bucket_flows.items()}
        else:
            bucket_ratios = {k: 0 for k in bucket_flows.keys()}
            
        distribution['intraday_flow_distribution'] = json.dumps({
            'bucket_flows': bucket_flows,
            'bucket_ratios': bucket_ratios,
            'peak_period': max(bucket_flows.items(), key=lambda x: abs(x[1]))[0] if bucket_flows else None
        })
        return distribution

    def _detect_high_freq_large_orders(self, df: pd.DataFrame) -> Dict[str, Any]:
        """高频大单识别"""
        metrics = {}
        # 定义大单阈值（可根据市值调整）
        large_order_threshold = 100000  # 10万元
        # 识别大单
        large_buy_orders = df[(df['type'] == 'B') & (df['amount'] >= large_order_threshold)]
        large_sell_orders = df[(df['type'] == 'S') & (df['amount'] >= large_order_threshold)]
        # 计算大单净流入
        total_large_buy = large_buy_orders['amount'].sum() / 10000  # 转换为万元
        total_large_sell = large_sell_orders['amount'].sum() / 10000
        large_order_net = total_large_buy - total_large_sell
        metrics['tick_large_order_net'] = float(large_order_net)
        metrics['tick_large_order_count'] = int(len(large_buy_orders) + len(large_sell_orders))
        return metrics

    def _calculate_flow_impact_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算资金冲击特征"""
        metrics = {}
        if len(df) < 10:
            metrics['flow_impact_ratio'] = None
            metrics['flow_persistence_minutes'] = 0
            return metrics
        # 计算资金冲击系数（单位资金对价格的影响）
        df = df.copy()
        df['price_change'] = df['price'].diff()
        # 按资金方向分组计算
        buy_orders = df[df['type'] == 'B']
        sell_orders = df[df['type'] == 'S']
        if len(buy_orders) > 5 and len(sell_orders) > 5:
            # 买盘冲击系数
            buy_impact = abs(buy_orders['price_change'].mean()) / (buy_orders['amount'].mean() / 1e6 + 1e-6)
            # 卖盘冲击系数
            sell_impact = abs(sell_orders['price_change'].mean()) / (sell_orders['amount'].mean() / 1e6 + 1e-6)
            metrics['flow_impact_ratio'] = float((buy_impact + sell_impact) / 2)
        else:
            metrics['flow_impact_ratio'] = None
        # 计算资金持续性
        df['minute'] = pd.to_datetime(df['trade_time']).dt.floor('T')
        minute_flow = df.groupby('minute').apply(
            lambda x: (x[x['type'] == 'B']['amount'].sum() - 
                      x[x['type'] == 'S']['amount'].sum())
        )
        # 计算连续同向的分钟数
        consecutive_minutes = 0
        current_sign = 0
        max_consecutive = 0
        for flow in minute_flow:
            if flow == 0:
                continue
            sign = 1 if flow > 0 else -1
            if sign == current_sign:
                consecutive_minutes += 1
                max_consecutive = max(max_consecutive, consecutive_minutes)
            else:
                current_sign = sign
                consecutive_minutes = 1
        metrics['flow_persistence_minutes'] = max_consecutive
        return metrics

    def _calculate_intraday_momentum(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算日内资金动量"""
        metrics = {}
        # 按分钟聚合
        df['minute'] = pd.to_datetime(df['trade_time']).dt.floor('T')
        minute_flow = df.groupby('minute').apply(
            lambda x: (x[x['type'] == 'B']['amount'].sum() - 
                      x[x['type'] == 'S']['amount'].sum()) / 10000  # 万元
        )
        if len(minute_flow) < 10:
            metrics['intraday_flow_momentum'] = None
            metrics['flow_acceleration_intraday'] = None
            return metrics
        # 计算日内动量（最近30分钟 vs 之前）
        if len(minute_flow) >= 30:
            recent_30 = minute_flow[-30:].mean()
            previous_30 = minute_flow[-60:-30].mean() if len(minute_flow) >= 60 else minute_flow[:-30].mean()
            if abs(previous_30) > 0:
                metrics['intraday_flow_momentum'] = float((recent_30 - previous_30) / abs(previous_30))
            else:
                metrics['intraday_flow_momentum'] = 0.0
        else:
            metrics['intraday_flow_momentum'] = None
        # 计算日内加速度
        if len(minute_flow) >= 3:
            recent_3 = minute_flow[-3:].values
            # 二阶差分近似加速度
            acceleration = (recent_3[2] - 2*recent_3[1] + recent_3[0]) / (abs(recent_3[0]) + 1e-6)
            metrics['flow_acceleration_intraday'] = float(acceleration)
        else:
            metrics['flow_acceleration_intraday'] = None
        return metrics

    def _calculate_flow_cluster_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算资金聚类特征"""
        metrics = {}
        # 按3秒间隔计算资金流
        df['time_group'] = pd.to_datetime(df['trade_time']).dt.floor('3S')
        cluster_flow = df.groupby('time_group').apply(
            lambda x: (x[x['type'] == 'B']['amount'].sum() - 
                      x[x['type'] == 'S']['amount'].sum())
        )
        if len(cluster_flow) < 10:
            metrics['flow_cluster_intensity'] = None
            metrics['flow_cluster_duration'] = 0
            return metrics
        # 计算聚类强度（资金流的标准差/均值）
        mean_flow = cluster_flow.abs().mean()
        std_flow = cluster_flow.std()
        if mean_flow > 0:
            metrics['flow_cluster_intensity'] = float(std_flow / mean_flow)
        else:
            metrics['flow_cluster_intensity'] = 0.0
        # 计算聚类持续时间
        # 寻找连续的资金流入/流出时段
        threshold = cluster_flow.abs().mean() * 0.5  # 阈值
        significant_flows = cluster_flow[cluster_flow.abs() > threshold]
        if len(significant_flows) > 0:
            # 计算最大连续时段
            times = pd.Series(significant_flows.index).sort_values()
            time_diffs = times.diff().dt.total_seconds()
            # 寻找连续的时间段（间隔小于6秒视为连续）
            cluster_duration = 0
            current_cluster = 0
            for diff in time_diffs:
                if diff <= 6:  # 6秒内视为同一聚类
                    current_cluster += diff
                    cluster_duration = max(cluster_duration, current_cluster)
                else:
                    current_cluster = 0
            metrics['flow_cluster_duration'] = int(cluster_duration / 60)  # 转换为分钟
        else:
            metrics['flow_cluster_duration'] = 0
        return metrics

    def _calculate_high_freq_divergence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算高频资金分歧度"""
        metrics = {}
        if len(df) < 20:
            metrics['high_freq_flow_divergence'] = None
            return metrics
        # 计算买卖盘资金流的相关系数
        df['time_group'] = pd.to_datetime(df['trade_time']).dt.floor('10S')  # 10秒分组
        buy_flow = df[df['type'] == 'B'].groupby('time_group')['amount'].sum()
        sell_flow = df[df['type'] == 'S'].groupby('time_group')['amount'].sum()
        # 对齐时间索引
        common_times = buy_flow.index.intersection(sell_flow.index)
        if len(common_times) >= 5:
            buy_aligned = buy_flow.loc[common_times]
            sell_aligned = sell_flow.loc[common_times]
            # 计算负相关性（负相关越高，分歧越大）
            correlation = buy_aligned.corr(sell_flow.loc[common_times])
            if not np.isnan(correlation):
                # 将相关性转换为分歧度（-1到1映射到0到100）
                divergence = (1 - correlation) / 2 * 100
                metrics['high_freq_flow_divergence'] = float(divergence)
            else:
                metrics['high_freq_flow_divergence'] = 50.0
        else:
            metrics['high_freq_flow_divergence'] = None
        return metrics

    def _calculate_vwap_deviation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算VWAP偏离度"""
        metrics = {}
        if len(df) < 10:
            metrics['vwap_deviation'] = None
            return metrics
        # 计算VWAP
        total_amount = df['amount'].sum()
        total_volume = df['volume'].sum()
        if total_volume > 0:
            vwap = total_amount / total_volume
            # 计算资金流入价格（买盘成交的加权平均价）
            buy_orders = df[df['type'] == 'B']
            if len(buy_orders) > 0:
                buy_vwap = (buy_orders['amount'] * buy_orders['price']).sum() / buy_orders['amount'].sum()
                # 计算偏离度
                if vwap > 0:
                    deviation = (buy_vwap - vwap) / vwap * 100
                    metrics['vwap_deviation'] = float(deviation)
                else:
                    metrics['vwap_deviation'] = 0.0
            else:
                metrics['vwap_deviation'] = 0.0
        else:
            metrics['vwap_deviation'] = None
        return metrics

    def _calculate_flow_efficiency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算资金流入效率"""
        metrics = {}
        if len(df) < 20:
            metrics['flow_efficiency'] = None
            return metrics
        # 计算单位资金推动的价格变化
        df = df.copy()
        df['price_change'] = df['price'].diff().abs()
        # 按资金方向分组
        buy_orders = df[df['type'] == 'B']
        sell_orders = df[df['type'] == 'S']
        if len(buy_orders) >= 5:
            # 买盘效率：价格变化/资金量
            buy_efficiency = buy_orders['price_change'].sum() / (buy_orders['amount'].sum() / 1e6 + 1e-6)
            metrics['flow_efficiency'] = float(buy_efficiency)
        else:
            metrics['flow_efficiency'] = 0.0
        return metrics

    def _calculate_closing_flow_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        计算尾盘资金特征
        版本: V1.2
        说明: 修复时区比较错误，确保收盘时间与数据时区一致。
        """
        metrics = {}
        if df.empty:
            metrics['closing_flow_ratio'] = None
            metrics['closing_flow_intensity'] = None
            return metrics

        # 识别尾盘时段（最后30分钟）
        # df['trade_time'] 已经是北京时间 (Aware)
        
        # 获取数据中的日期
        current_date = df['trade_time'].iloc[0].date()
        
        # 构造收盘时间 (15:00) 并匹配 DataFrame 的时区
        # 此时 df['trade_time'] 已经统一为 Asia/Shanghai，所以这里直接 localize 即可
        # 使用 df['trade_time'].dt.tz 获取准确的时区对象
        tz_info = df['trade_time'].dt.tz
        market_close = pd.Timestamp(current_date).tz_localize(tz_info) + pd.Timedelta(hours=15)
        
        closing_start = market_close - pd.Timedelta(minutes=30)
        
        # 进行带时区的时间比较
        closing_data = df[df['trade_time'] >= closing_start]
        
        if len(closing_data) == 0:
            # 如果没有尾盘数据，可能全天都在尾盘前，或者数据缺失
            metrics['closing_flow_ratio'] = 0.0
            metrics['closing_flow_intensity'] = 0.0
            return metrics

        # 计算尾盘资金净流入
        closing_buy = closing_data[closing_data['type'] == 'B']['amount'].sum()
        closing_sell = closing_data[closing_data['type'] == 'S']['amount'].sum()
        closing_net = (closing_buy - closing_sell) / 10000  # 万元
        # 全天资金净流入
        total_buy = df[df['type'] == 'B']['amount'].sum()
        total_sell = df[df['type'] == 'S']['amount'].sum()
        total_net = (total_buy - total_sell) / 10000
        # 尾盘资金占比
        if abs(total_net) > 0:
            closing_ratio = abs(closing_net) / abs(total_net) * 100
        else:
            closing_ratio = 0.0
        # 尾盘资金强度（尾盘资金密度）
        closing_duration_minutes = 30
        closing_intensity = abs(closing_net) / closing_duration_minutes
        metrics['closing_flow_ratio'] = float(closing_ratio)
        metrics['closing_flow_intensity'] = float(closing_intensity)
        return metrics

    def _calculate_high_freq_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算高频统计特征"""
        metrics = {}
        # 按分钟计算资金流
        df['minute'] = pd.to_datetime(df['trade_time']).dt.floor('T')
        minute_flow = df.groupby('minute').apply(
            lambda x: (x[x['type'] == 'B']['amount'].sum() - 
                      x[x['type'] == 'S']['amount'].sum()) / 10000
        )
        if len(minute_flow) < 10:
            metrics['high_freq_flow_skewness'] = None
            metrics['high_freq_flow_kurtosis'] = None
            return metrics
        # 计算偏度和峰度
        metrics['high_freq_flow_skewness'] = float(stats.skew(minute_flow))
        metrics['high_freq_flow_kurtosis'] = float(stats.kurtosis(minute_flow))
        return metrics

    def _calculate_time_period_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        计算资金时段分布
        版本: V1.2
        说明: 直接使用已标准化的 trade_time 列。
        """
        metrics = {}
        # 划分上午和下午
        # df['trade_time'] 已经是北京时间
        hour = df['trade_time'].dt.hour
        morning_data = df[hour < 12]  # 上午 (通常 9, 10, 11)
        afternoon_data = df[hour >= 13]  # 下午 (通常 13, 14, 15)
        
        if len(morning_data) == 0 and len(afternoon_data) == 0:
            metrics['morning_flow_ratio'] = None
            metrics['afternoon_flow_ratio'] = None
            return metrics
            
        # 计算上午资金净流入
        morning_buy = morning_data[morning_data['type'] == 'B']['amount'].sum()
        morning_sell = morning_data[morning_data['type'] == 'S']['amount'].sum()
        morning_net = (morning_buy - morning_sell) / 10000
        # 计算下午资金净流入
        afternoon_buy = afternoon_data[afternoon_data['type'] == 'B']['amount'].sum()
        afternoon_sell = afternoon_data[afternoon_data['type'] == 'S']['amount'].sum()
        afternoon_net = (afternoon_buy - afternoon_sell) / 10000
        # 计算全天资金净流入
        total_net = morning_net + afternoon_net
        # 计算占比
        if abs(total_net) > 0:
            morning_ratio = abs(morning_net) / abs(total_net) * 100
            afternoon_ratio = abs(afternoon_net) / abs(total_net) * 100
        else:
            # 如果总净流入为0，但有交易，给个默认值
            if len(df) > 0:
                morning_ratio = 50.0
                afternoon_ratio = 50.0
            else:
                morning_ratio = None
                afternoon_ratio = None
                
        metrics['morning_flow_ratio'] = float(morning_ratio) if morning_ratio is not None else None
        metrics['afternoon_flow_ratio'] = float(afternoon_ratio) if afternoon_ratio is not None else None
        return metrics

    def _calculate_stealth_flow_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算主力隐蔽性指标"""
        metrics = {}
        # 定义小单阈值
        small_order_threshold = 50000  # 5万元以下视为小单
        # 识别小单
        small_buy_orders = df[(df['type'] == 'B') & (df['amount'] < small_order_threshold)]
        small_sell_orders = df[(df['type'] == 'S') & (df['amount'] < small_order_threshold)]
        # 计算小单净流入
        small_buy_total = small_buy_orders['amount'].sum() / 10000  # 万元
        small_sell_total = small_sell_orders['amount'].sum() / 10000
        small_net = small_buy_total - small_sell_total
        # 计算总净流入
        total_buy = df[df['type'] == 'B']['amount'].sum() / 10000
        total_sell = df[df['type'] == 'S']['amount'].sum() / 10000
        total_net = total_buy - total_sell
        # 计算隐蔽资金占比（小单但持续同向）
        if abs(total_net) > 0:
            stealth_ratio = abs(small_net) / abs(total_net) * 100
        else:
            stealth_ratio = 0.0
        metrics['stealth_flow_ratio'] = float(stealth_ratio)
        return metrics

    # ==================== 主计算方法 ====================
    def calculate_all_metrics(self) -> Dict[str, Any]:
        """计算所有指标"""
        all_metrics = {}
        try:
            # 1. 绝对量级指标
            all_metrics.update(self.calculate_absolute_metrics())
            # 2. 相对强度指标
            all_metrics.update(self.calculate_relative_metrics())
            # 3. 主力行为模式识别
            all_metrics.update(self.calculate_behavior_patterns())
            # 4. 资金流向质量评估
            all_metrics.update(self.calculate_flow_quality())
            # 5. 多周期资金共振指标
            all_metrics.update(self.calculate_multi_period_sync())
            # 6. 趋势动量指标
            all_metrics.update(self.calculate_trend_momentum())
            # 7. 量价背离指标
            all_metrics.update(self.calculate_divergence_metrics())
            # 8. 结构分析指标
            all_metrics.update(self.calculate_structure_metrics())
            # 9. 统计特征指标
            all_metrics.update(self.calculate_statistical_metrics())
            # 10. 预测指标
            all_metrics.update(self.calculate_prediction_metrics())
            # 11. 复合综合指标
            all_metrics.update(self.calculate_comprehensive_metrics(all_metrics))
            # 12. 基于Tick数据的增强指标
            all_metrics.update(self.calculate_tick_enhanced_metrics())
            # 13. 更新复合综合指标（考虑tick指标）
            all_metrics.update(self.calculate_comprehensive_metrics(all_metrics))
            # 14. 保存资金流序列（最近30天）
            self._save_flow_sequence(all_metrics)
            logger.info(f"成功计算 {self.context.stock_code} 在 {self.context.trade_date} 的资金流向因子")
        except Exception as e:
            logger.error(f"计算资金流向因子时发生异常: {e}", exc_info=True)
            raise
        return all_metrics
    
    def _save_flow_sequence(self, metrics: Dict[str, Any]):
        """保存资金流序列到计算结果中"""
        # 保存最近30天的资金流数据
        sequence_data = []
        for i, data in enumerate(self.context.historical_flow_data[-30:]):
            day_data = {
                'trade_date': data.get('trade_date', ''),
                'net_mf_amount': float(data.get('net_mf_amount', 0) or 0),
                'net_mf_vol': int(data.get('net_mf_vol', 0) or 0),
            }
            sequence_data.append(day_data)
        # 保存特征向量（用于机器学习）
        feature_vector = self._create_feature_vector(metrics)
        metrics['feature_vector'] = feature_vector
        # 保存计算元数据
        metadata = {
            'calculation_time': datetime.now().isoformat(),
            'data_sources': {
                'has_daily_basic': bool(self.context.daily_basic_data),
                'has_minute_data': self.context.minute_data_1min is not None and not self.context.minute_data_1min.empty,
                'historical_days': len(self.context.historical_flow_data)
            }
        }
    
    def _create_feature_vector(self, metrics: Dict[str, Any]) -> str:
        """
        创建特征向量
        版本: V1.1
        说明: 
        1. 移除 pickle 序列化，改用 numpy.tobytes()，速度更快且更安全。
        2. 保持 Base64 编码输出。
        """
        # 选择关键特征
        key_features = [
            metrics.get('total_net_amount_5d', 0) or 0,
            metrics.get('avg_daily_net_5d', 0) or 0,
            metrics.get('net_amount_ratio', 0) or 0,
            metrics.get('flow_intensity', 0) or 0,
            metrics.get('accumulation_score', 0) or 0,
            metrics.get('pushing_score', 0) or 0,
            metrics.get('distribution_score', 0) or 0,
            metrics.get('flow_consistency', 0) or 0,
            metrics.get('daily_weekly_sync', 0) or 0,
            metrics.get('flow_momentum_5d', 0) or 0,
            metrics.get('comprehensive_score', 0) or 0,
        ]
        # 转换为numpy数组
        import numpy as np
        vector = np.array(key_features, dtype=np.float32)
        # 优化：使用 tobytes() 替代 pickle
        import base64
        vector_bytes = vector.tobytes()
        return base64.b64encode(vector_bytes).decode('utf-8')










