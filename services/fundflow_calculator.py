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
    """计算上下文，包含所有必要的数据"""
    stock_code: str
    trade_date: date
    current_flow_data: Dict[str, Any]  # 当前日资金流向数据
    historical_flow_data: List[Dict[str, Any]]  # 历史资金流向数据（至少30天）
    daily_basic_data: Optional[Dict[str, Any]] = None  # 每日基本信息
    minute_data_1min: Optional[pd.DataFrame] = None  # 1分钟数据
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
        版本: V1.2
        说明: 
        1. 将核心数据序列直接转换为 numpy.array，供后续向量化计算使用，避免反复转换。
        2. 确保数组类型为 float 以避免计算精度问题。
        """
        # 提取历史净流入序列 (转换为 numpy array)
        self.net_amount_series = [
            float(data.get('net_mf_amount', 0) or 0) 
            for data in self.context.historical_flow_data
        ]
        self.net_amount_array = np.array(self.net_amount_series, dtype=np.float64)
        # 保留 Pandas Series 供 rolling 计算使用
        self.net_amount_pd_series = pd.Series(self.net_amount_array)
        # 提取历史成交量
        if self.context.volume_data:
            self.volume_series = self.context.volume_data
        else:
            self.volume_series = [
                float(data.get('total_volume', 0) or 0) 
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
        版本: V1.2
        说明: 使用 NumPy 向量化除法计算比率序列，避免循环和 zip 操作。
        """
        metrics = {}
        current_net = float(self.context.current_flow_data.get('net_mf_amount', 0) or 0)
        # 1. 当日净流入占比
        if self.context.daily_basic_data:
            daily_amount = float(self.context.daily_basic_data.get('amount', 0) or 0)
            if daily_amount > 0:
                net_ratio = (current_net / daily_amount) * 100
            else:
                net_ratio = 0.0
        else:
            # 使用历史平均成交额估算
            # 向量化过滤大于0的成交额
            valid_amounts = self.daily_amount_array[-20:]
            valid_amounts = valid_amounts[valid_amounts > 0]
            if len(valid_amounts) > 0:
                avg_amount = np.mean(valid_amounts)
                net_ratio = (current_net / avg_amount * 100)
            else:
                net_ratio = 0.0
                
        metrics['net_amount_ratio'] = float(net_ratio)
        # 2. 5日、10日均净流入占比 (向量化计算)
        # 准备数据
        net_arr = self.net_amount_array
        amt_arr = self.daily_amount_array
        n = len(net_arr)
        for window in [5, 10]:
            if n >= window:
                recent_nets = net_arr[-window:]
                recent_amts = amt_arr[-window:]
                # 向量化除法，处理分母为0的情况
                # out=np.zeros_like... 确保分母为0时结果为0
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratios = np.divide(
                        recent_nets, 
                        recent_amts, 
                        out=np.zeros_like(recent_nets), 
                        where=recent_amts!=0
                    ) * 100
                # 过滤掉原本分母为0导致的0结果(如果业务逻辑认为分母为0不应参与平均)
                # 或者直接求平均(如果认为分母为0则比率为0是合理的)
                # 这里沿用原逻辑：只计算有效比率的平均值
                valid_mask = recent_amts != 0
                if np.any(valid_mask):
                    avg_ratio = np.mean(ratios[valid_mask])
                    metrics[f'net_amount_ratio_ma{window}'] = float(avg_ratio)
                else:
                    metrics[f'net_amount_ratio_ma{window}'] = None
            else:
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
        版本: V1.1
        说明: 使用 NumPy 向量化计算符号一致性。
        """
        # 使用预处理的 array
        arr = self.net_amount_array
        if len(arr) < 3:
            return 50.0
            
        # 取最近3天
        recent = arr[-3:]
        # 获取符号 (-1, 0, 1)
        signs = np.sign(recent)
        # 比较相邻符号是否相等
        # signs[1:] 与 signs[:-1] 比较
        # 例如 [1, 1, -1] -> 比较 (1,1) 和 (1,-1) -> [True, False]
        if len(signs) < 2:
            return 50.0
            
        matches = (signs[1:] == signs[:-1])
        consistency = np.mean(matches) * 100
        return float(consistency)

    def _calculate_flow_stability(self) -> float:
        """计算资金流稳定性"""
        if len(self.net_amount_series) < 5:
            return 50
        recent_nets = self.net_amount_series[-5:]
        volatility = np.std(recent_nets) / (abs(np.mean(recent_nets)) + 1e-6)
        # 波动性越小，稳定性越高
        stability = max(0, 100 - volatility * 100)
        return stability
    
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
            'flow_intensity': 0.15,
            'pattern_confidence': 0.15,
            'flow_consistency': 0.10,
            'flow_stability': 0.10,
            'daily_weekly_sync': 0.10,
            'uptrend_strength': 0.10,
            'downtrend_strength': 0.10,
            'divergence_strength': 0.10,
            'flow_forecast_confidence': 0.10,
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
            # 12. 保存资金流序列（最近30天）
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










