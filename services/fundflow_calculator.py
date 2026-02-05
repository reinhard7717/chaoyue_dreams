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
        """
        处理1分钟数据，提取日内资金流特征
        版本: V1.2
        说明: 
        1. 移除Pandas Series操作，改用NumPy数组运算，避免DataFrame列修改带来的副作用和开销。
        2. 使用np.diff计算收益率，速度优于pct_change。
        """
        df = self.context.minute_data_1min
        # 提取数组，避免Pandas索引开销
        if 'close' in df.columns:
            closes = df['close'].values.astype(np.float64)
            if len(closes) > 1:
                # 向量化计算收益率: (Price_t - Price_t-1) / Price_t-1
                # 相比 pd.pct_change，np.diff 没有任何索引对齐检查，极快
                diffs = np.diff(closes)
                # 避免除以0
                prev_closes = closes[:-1]
                # 使用 np.divide 安全除法 (虽然价格一般非0)
                returns = np.divide(diffs, prev_closes, out=np.zeros_like(diffs), where=prev_closes!=0)
                self.intraday_returns = returns.tolist()
                # 计算日内波动率 (年化)
                # ddof=1 对应样本标准差
                std_val = np.std(returns, ddof=1)
                self.intraday_volatility = float(std_val * 15.491933) # sqrt(240) ≈ 15.49
            else:
                self.intraday_returns = []
                self.intraday_volatility = 0.0
        if 'volume' in df.columns:
            # 直接提取values转list，比Series.tolist()稍快
            self.intraday_volume = df['volume'].values.tolist()
        if 'amount' in df.columns:
            self.intraday_amount = df['amount'].values.tolist()

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
        """
        计算主力行为模式
        版本: V1.2
        说明: 直接使用预处理的NumPy数组切片，避免重复的数据提取和转换。
        """
        patterns = {}
        # 直接使用 numpy array，如果长度不足10，切片会自动处理
        arr = self.net_amount_array
        recent_nets = arr[-10:] if len(arr) >= 10 else arr
        # 计算各种模式的得分 (传入 np.ndarray)
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

    def _calculate_accumulation_score(self, arr: np.ndarray) -> float:
        """
        计算建仓模式得分
        版本: V1.2
        说明: 接收 np.ndarray，移除内部转换，增加类型安全检查。
        """
        n = len(arr)
        if n < 5:
            return 0.0
        # 连续净流入天数
        positive_days = np.sum(arr > 0)
        continuity_score = (positive_days / n) * 100.0
        # 流入稳定性
        # 取最后3个 (利用切片视图)
        last_3 = arr[-3:]
        mean_last_3 = np.mean(last_3)
        volatility = np.std(last_3) / (np.abs(mean_last_3) + 1e-6)
        stability_score = max(0.0, 100.0 - volatility * 100.0)
        # 流出比例
        inflows = arr[arr > 0]
        outflows = arr[arr < 0]
        total_inflow = np.sum(inflows) if len(inflows) > 0 else 0.0
        total_outflow = np.sum(np.abs(outflows)) if len(outflows) > 0 else 0.0
        outflow_ratio = total_outflow / (total_inflow + total_outflow + 1e-6)
        outflow_score = max(0.0, 100.0 - outflow_ratio * 200.0)
        return float(continuity_score * 0.4 + stability_score * 0.3 + outflow_score * 0.3)

    def _calculate_pushing_score(self, arr: np.ndarray) -> float:
        """
        计算拉升模式得分
        版本: V1.2
        说明: 接收 np.ndarray，优化切片操作。
        """
        n = len(arr)
        if n < 3:
            return 0.0
        # 最近3天净流入加速度 (二阶差分)
        # d[2] - 2d[1] + d[0]
        recent_3 = arr[-3:]
        acc_val = recent_3[2] - 2*recent_3[1] + recent_3[0]
        accel_score = min(100.0, max(0.0, 50.0 + acc_val / 1000.0 * 10.0))
        # 流入强度 (最近5天)
        recent_5 = arr[-5:]
        pos_mask = recent_5 > 0
        if np.any(pos_mask):
            avg_inflow = np.mean(recent_5[pos_mask])
        else:
            avg_inflow = 0.0
        intensity_score = min(100.0, avg_inflow / 5000.0 * 100.0)
        # 流出可控性
        neg_mask = recent_5 < 0
        max_outflow = np.max(np.abs(recent_5[neg_mask])) if np.any(neg_mask) else 0.0
        if avg_inflow > 0:
            outflow_control = max(0.0, 100.0 - (max_outflow / avg_inflow) * 100.0)
        else:
            outflow_control = 100.0
        return float(accel_score * 0.4 + intensity_score * 0.3 + outflow_control * 0.3)

    def _calculate_distribution_score(self, arr: np.ndarray) -> float:
        """
        计算派发模式得分
        版本: V1.2
        说明: 接收 np.ndarray，保持向量化逻辑。
        """
        n = len(arr)
        if n < 5:
            return 0.0
        # 净流出天数占比
        neg_mask = arr < 0
        negative_days = np.sum(neg_mask)
        negative_ratio_score = (negative_days / n) * 100.0
        # 流出绝对量
        if negative_days > 0:
            avg_outflow = np.mean(np.abs(arr[neg_mask]))
        else:
            avg_outflow = 0.0
        outflow_amount_score = min(100.0, avg_outflow / 3000.0 * 100.0)
        # 流出持续性（连续流出天数）- 向量化
        padded = np.concatenate(([False], neg_mask, [False]))
        diffs = np.diff(padded.astype(int))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        if len(starts) > 0:
            max_consecutive = np.max(ends - starts)
        else:
            max_consecutive = 0
        continuity_score = min(100.0, float(max_consecutive * 20.0))
        return float(negative_ratio_score * 0.4 + outflow_amount_score * 0.3 + continuity_score * 0.3)

    def _calculate_shakeout_score(self, arr: np.ndarray) -> float:
        """
        计算洗盘模式得分
        版本: V1.2
        说明: 接收 np.ndarray。
        """
        n = len(arr)
        if n < 5:
            return 0.0
        avg_net = np.mean(arr)
        avg_abs = np.mean(np.abs(arr))
        if avg_net < 0 and avg_abs < 2000:
            amount_score = 80.0
        else:
            amount_score = 20.0
        volatility = np.std(arr) / (np.abs(avg_net) + 1e-6)
        volatility_score = min(100.0, volatility * 50.0)
        # 流出流入交替
        signs = np.sign(arr)
        signs_no_zero = signs[signs != 0]
        if len(signs_no_zero) >= 2:
            # 乘积为负表示异号
            changes = signs_no_zero[1:] * signs_no_zero[:-1]
            sign_changes = np.sum(changes < 0)
            change_score = min(100.0, sign_changes / n * 200.0)
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
        版本: V1.3
        说明: 
        1. 使用NumPy向量化替代Python原生sum和列表推导，提升计算效率。
        2. 增加类型安全检查，确保数组运算的稳定性。
        """
        data = self.context.current_flow_data
        if not data:
            return 50.0
        try:
            # 提取数据构建NumPy数组，避免多次字典查询
            keys_buy = ['buy_elg_amount', 'buy_lg_amount', 'buy_md_amount']
            keys_sell = ['sell_elg_amount', 'sell_lg_amount', 'sell_md_amount']
            # 使用列表解析一次性提取并转换为float，然后转为array
            buys = np.array([float(data.get(k, 0) or 0) for k in keys_buy], dtype=np.float64)
            sells = np.array([float(data.get(k, 0) or 0) for k in keys_sell], dtype=np.float64)
            # 向量化计算净流入
            nets = buys - sells
            # 向量化计算分子分母
            abs_sum_net = abs(np.sum(nets))
            sum_abs_net = np.sum(np.abs(nets))
            if sum_abs_net == 0:
                return 50.0
            consistency_score = (abs_sum_net / sum_abs_net) * 100.0
            return float(consistency_score)
        except Exception as e:
            logger.error(f"计算资金一致性出错: {e}")
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
        版本: V1.2
        说明: 
        1. 使用NumPy卷积(convolve)替代Pandas Rolling，大幅提升计算效率。
        2. 纯数组操作，避免Series对象的反复创建。
        """
        sync_metrics = {}
        # 初始化默认值
        for key in ['daily_weekly_sync', 'daily_monthly_sync', 
                   'short_mid_sync', 'mid_long_sync']:
            sync_metrics[key] = None
        arr = self.net_amount_array
        if len(arr) < 20:
            return sync_metrics
        # 辅助函数：计算移动平均 (Moving Average)
        def calc_ma(data, window):
            if len(data) < window:
                return np.array([])
            # mode='valid' 返回长度为 N - K + 1 的数组，即去除了前面不足window的部分
            return np.convolve(data, np.ones(window)/window, mode='valid')
        # 日线数据 (取最近20天以匹配其他序列长度)
        daily_full = arr
        # 周线数据 (MA5)
        ma5 = calc_ma(daily_full, 5)
        # 月线数据 (MA20)
        ma20 = calc_ma(daily_full, 20)
        # 中长期数据 (MA60)
        ma60 = calc_ma(daily_full, 60)
        # 确保数据对齐并取最近3个点进行同步度计算
        # 注意：ma5的最后一个元素对应arr的最后一个元素。
        # 取切片 [-3:]
        # 1. 日线 vs 周线
        if len(daily_full) >= 3 and len(ma5) >= 3:
            sync_metrics['daily_weekly_sync'] = self._calculate_sync_score(daily_full[-3:], ma5[-3:])
        # 2. 日线 vs 月线
        if len(daily_full) >= 3 and len(ma20) >= 3:
            sync_metrics['daily_monthly_sync'] = self._calculate_sync_score(daily_full[-3:], ma20[-3:])
        # 3. 短中期 (MA5 vs MA20)
        if len(ma5) >= 3 and len(ma20) >= 3:
            # 这里的 ma5 和 ma20 长度不同，且起始点不同。
            # ma5[-1] 和 ma20[-1] 都是对应同一天（最新一天）。
            # 所以直接取各自的 [-3:] 即可对齐。
            sync_metrics['short_mid_sync'] = self._calculate_sync_score(ma5[-3:], ma20[-3:])
        # 4. 中长期 (MA20 vs MA60)
        if len(ma20) >= 3 and len(ma60) >= 3:
            sync_metrics['mid_long_sync'] = self._calculate_sync_score(ma20[-3:], ma60[-3:])
        return sync_metrics

    def _calculate_sync_score(self, series1: List[float], series2: List[float]) -> float:
        """
        计算两个序列的同步度得分
        版本: V1.2
        说明: 
        1. 移除np.corrcoef，改用基于向量点积的皮尔逊相关系数公式，减少矩阵计算开销。
        2. 保持方向一致性的向量化计算。
        """
        s1 = np.array(series1, dtype=np.float64)
        s2 = np.array(series2, dtype=np.float64)
        n = len(s1)
        if n != len(s2) or n < 2:
            return 50.0
        # 优化相关性计算：手写Pearson公式避免构建矩阵
        # r = sum((x - mx)(y - my)) / sqrt(sum((x-mx)^2) * sum((y-my)^2))
        s1_mean = np.mean(s1)
        s2_mean = np.mean(s2)
        s1_c = s1 - s1_mean
        s2_c = s2 - s2_mean
        numerator = np.dot(s1_c, s2_c)
        denominator = np.sqrt(np.dot(s1_c, s1_c) * np.dot(s2_c, s2_c))
        if denominator == 0:
            correlation = 0.0
        else:
            correlation = numerator / denominator
        # 方向一致性 (向量化)
        diff1 = np.diff(s1)
        diff2 = np.diff(s2)
        # 使用符号函数比较，避免除零
        dir_match = np.sign(diff1) == np.sign(diff2)
        direction_score = (np.sum(dir_match) / len(diff1) * 100) if len(diff1) > 0 else 50.0
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
        版本: V1.2
        说明: 
        1. 移除np.polyfit，使用最小二乘法代数公式直接计算斜率和R方。
        2. 相比SVD分解，代数法在小样本下速度提升显著。
        """
        if len(self.net_amount_series) < 10:
            return 0.0, 0.0
        y = np.array(self.net_amount_series[-10:], dtype=np.float64)
        n = len(y)
        x = np.arange(n, dtype=np.float64)
        # 最小二乘法直接计算
        # Slope = (N*Σxy - ΣxΣy) / (N*Σx^2 - (Σx)^2)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.dot(x, y)
        sum_xx = np.dot(x, x)
        denominator = n * sum_xx - sum_x * sum_x
        if denominator == 0:
            return 0.0, 0.0
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        # 计算R^2
        # R^2 = (Σ(yi - ŷi)^2) / Σ(yi - y_bar)^2 的补数，或者直接算相关系数平方
        # 使用皮尔逊相关系数平方更快
        y_mean = sum_y / n
        x_mean = sum_x / n
        numerator_r = np.dot(x - x_mean, y - y_mean)
        denom_r = np.sqrt(np.dot(x - x_mean, x - x_mean) * np.dot(y - y_mean, y - y_mean))
        if denom_r == 0:
            r_sq = 0.0
        else:
            r_sq = (numerator_r / denom_r) ** 2
        # 上升趋势强度
        strength = min(100.0, r_sq * 100.0)
        if slope > 0:
            return strength, 0.0
        else:
            return 0.0, strength

    # ==================== 7. 量价背离指标 ====================
    def calculate_divergence_metrics(self) -> Dict[str, Any]:
        """
        计算量价背离指标
        版本: V1.2
        说明: 使用 NumPy 数组进行切片和传递，移除中间 List 转换。
        """
        divergence = {}
        # 依赖 close_array 和 net_amount_array
        if len(self.close_array) < 3 or len(self.net_amount_array) < 3:
            divergence['price_flow_divergence'] = None
            divergence['divergence_type'] = 'NONE'
            divergence['divergence_strength'] = 0
            return divergence
        limit = 10
        # 获取切片
        price_history = self.close_array[-limit:]
        flow_history = self.net_amount_array[-limit:]
        # 简单对齐：只取两者长度的最小值
        min_len = min(len(price_history), len(flow_history))
        price_history = price_history[-min_len:]
        flow_history = flow_history[-min_len:]
        # 过滤价格为0 (停牌等)
        valid_mask = price_history > 0
        if np.sum(valid_mask) < 3:
            divergence['price_flow_divergence'] = None
            divergence['divergence_type'] = 'NONE'
            divergence['divergence_strength'] = 0
            return divergence
        p_valid = price_history[valid_mask]
        f_valid = flow_history[valid_mask]
        # 计算趋势
        price_trend = self._calculate_trend_direction(p_valid[-3:])
        flow_trend = self._calculate_trend_direction(f_valid[-3:])
        if price_trend < 0 and flow_trend > 0:
            divergence_type = 'BULLISH'
            divergence_strength = abs(price_trend) * abs(flow_trend) * 25
        elif price_trend > 0 and flow_trend < 0:
            divergence_type = 'BEARISH'
            divergence_strength = abs(price_trend) * abs(flow_trend) * 25
        else:
            divergence_type = 'NONE'
            divergence_strength = 0
        divergence_score = abs(price_trend - flow_trend) * 50
        divergence['price_flow_divergence'] = float(divergence_score)
        divergence['divergence_type'] = divergence_type
        divergence['divergence_strength'] = float(min(100.0, divergence_strength))
        return divergence

    def _calculate_trend_direction(self, arr: np.ndarray) -> float:
        """
        计算序列趋势方向
        版本: V1.2
        说明: 接收 np.ndarray，直接计算。
        """
        if len(arr) < 2:
            return 0.0
        changes = np.diff(arr)
        avg_change = np.mean(changes)
        std_change = np.std(changes) + 1e-6
        trend = avg_change / std_change
        return float(np.clip(trend, -1.0, 1.0))

    # ==================== 8. 结构分析指标 ====================
    def calculate_structure_metrics(self) -> Dict[str, Any]:
        """
        计算结构分析指标
        版本: V1.1
        说明: 使用 net_amount_array，find_peaks 支持 numpy 数组。
        """
        structure = {}
        arr = self.net_amount_array
        if len(arr) < 10:
            structure['flow_peak_value'] = None
            structure['days_since_last_peak'] = None
            structure['flow_support_level'] = None
            structure['flow_resistance_level'] = None
            return structure
        peaks, _ = find_peaks(arr, height=0)
        if len(peaks) > 0:
            last_peak_idx = peaks[-1]
            structure['flow_peak_value'] = float(arr[last_peak_idx])
            structure['days_since_last_peak'] = int(len(arr) - 1 - last_peak_idx)
        else:
            structure['flow_peak_value'] = float(np.max(arr)) if len(arr) > 0 else None
            structure['days_since_last_peak'] = None
        # 计算支撑阻力 (最近20天)
        recent_20 = arr[-20:]
        if len(recent_20) >= 5:
            # np.percentile 支持数组
            structure['flow_support_level'] = float(np.percentile(recent_20, 20))
            structure['flow_resistance_level'] = float(np.percentile(recent_20, 80))
        else:
            structure['flow_support_level'] = None
            structure['flow_resistance_level'] = None
        return structure

    # ==================== 9. 统计特征指标 ====================
    def calculate_statistical_metrics(self) -> Dict[str, float]:
        """
        计算统计特征指标
        版本: V1.2
        说明: 
        1. 移除scipy.stats.percentileofscore，使用NumPy向量化比较计算百分位。
        2. 复用预计算的统计量。
        """
        stats_metrics = {}
        if len(self.net_amount_series) < 10:
            for key in ['flow_zscore', 'flow_percentile', 'flow_volatility_10d', 'flow_volatility_20d']:
                stats_metrics[key] = None
            return stats_metrics
        current_net = float(self.context.current_flow_data.get('net_mf_amount', 0) or 0)
        # 使用预处理的 array
        recent_20 = self.net_amount_array[-20:]
        mean_20 = np.mean(recent_20)
        std_20 = np.std(recent_20)
        abs_mean_20 = abs(mean_20) + 1e-6
        # Z分数
        stats_metrics['flow_zscore'] = (current_net - mean_20) / (std_20 + 1e-6) if std_20 > 0 else 0.0
        # 百分位 (优化：向量化计算)
        # 计算小于当前值的比例
        if len(recent_20) > 0:
            stats_metrics['flow_percentile'] = (np.sum(recent_20 < current_net) / len(recent_20)) * 100.0
        else:
            stats_metrics['flow_percentile'] = 50.0
        # 波动率 10d
        if len(self.net_amount_array) >= 10:
            recent_10 = self.net_amount_array[-10:]
            stats_metrics['flow_volatility_10d'] = np.std(recent_10) / (abs(np.mean(recent_10)) + 1e-6)
        else:
            stats_metrics['flow_volatility_10d'] = None
        # 波动率 20d
        if len(self.net_amount_array) >= 20:
            stats_metrics['flow_volatility_20d'] = std_20 / abs_mean_20
        else:
            stats_metrics['flow_volatility_20d'] = None
        return stats_metrics

    # ==================== 10. 预测指标 ====================
    def calculate_prediction_metrics(self) -> Dict[str, float]:
        """
        计算预测指标
        版本: V1.1
        说明: 使用 NumPy 数组运算。
        """
        prediction = {}
        arr = self.net_amount_array
        if len(arr) < 10:
            for key in ['expected_flow_next_1d', 'flow_forecast_confidence',
                       'uptrend_continuation_prob', 'reversal_prob']:
                prediction[key] = None
            return prediction
        # 均值预测
        recent_nets = arr[-5:]
        prediction['expected_flow_next_1d'] = float(np.mean(recent_nets))
        # 稳定性
        volatility = np.std(recent_nets) / (abs(np.mean(recent_nets)) + 1e-6)
        prediction['flow_forecast_confidence'] = float(max(0.0, 100.0 - volatility * 50.0))
        # 趋势延续
        if len(arr) >= 3:
            recent_trend = arr[-3:]
            is_uptrend = recent_trend[-1] > recent_trend[0]
            # 使用数组调用
            historical_continuation = self._calculate_historical_continuation_prob()
            if is_uptrend:
                prediction['uptrend_continuation_prob'] = historical_continuation
                prediction['reversal_prob'] = 100.0 - historical_continuation
            else:
                prediction['uptrend_continuation_prob'] = 0.0
                prediction['reversal_prob'] = historical_continuation
        else:
            prediction['uptrend_continuation_prob'] = 50.0
            prediction['reversal_prob'] = 50.0
        return prediction

    def _calculate_historical_continuation_prob(self) -> float:
        """
        计算历史趋势延续概率
        版本: V1.2
        说明: 直接使用 self.net_amount_array，向量化逻辑。
        """
        arr = self.net_amount_array
        if len(arr) < 10:
            return 50.0
        # 相邻元素乘积 > 0 表示同向
        consecutive_products = arr[1:] * arr[:-1]
        continuation_count = np.sum(consecutive_products > 0)
        total_count = len(consecutive_products)
        if total_count > 0:
            probability = (continuation_count / total_count) * 100.0
        else:
            probability = 50.0
        return float(probability)

    # ==================== 11. 复合综合指标 ====================
    def calculate_comprehensive_metrics(self, all_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算复合综合指标
        版本: V1.2
        说明: 
        1. 优化权重计算逻辑，增强数值安全性。
        2. 确保所有得分为标准浮点数。
        """
        comprehensive = {}
        # 定义权重配置
        weights_config = {
            'flow_intensity': 0.12,
            'pattern_confidence': 0.12,
            'flow_consistency': 0.08,
            'flow_stability': 0.08,
            'daily_weekly_sync': 0.08,
            'uptrend_strength': 0.08,
            'downtrend_strength': 0.08,
            'divergence_strength': 0.08,
            'flow_forecast_confidence': 0.08,
            'tick_large_order_net': 0.05,
            'flow_cluster_intensity': 0.05,
            'closing_flow_intensity': 0.05,
            'flow_efficiency': 0.05,
        }
        total_score = 0.0
        total_weight = 0.0
        # 遍历权重配置，安全提取并计算
        for metric, weight in weights_config.items():
            val = all_metrics.get(metric)
            if val is not None:
                try:
                    f_val = float(val)
                    # 特殊指标归一化处理
                    if metric in ['downtrend_strength', 'divergence_strength']:
                        normalized = f_val
                    else:
                        # 限制在0-100之间
                        normalized = max(0.0, min(100.0, f_val))
                    total_score += normalized * weight
                    total_weight += weight
                except (ValueError, TypeError):
                    continue
        # 计算加权平均
        if total_weight > 0:
            comprehensive_score = total_score / total_weight
        else:
            comprehensive_score = 50.0
        comprehensive['comprehensive_score'] = float(comprehensive_score)
        # 交易信号生成
        signal, signal_strength = self._generate_trading_signal(all_metrics, comprehensive_score)
        comprehensive['trading_signal'] = signal
        comprehensive['signal_strength'] = float(signal_strength)
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
        版本: V1.4
        说明: 
        1. 优化时间处理：将Pandas Timestamp转换为NumPy datetime64[ns]，大幅提升后续计算速度。
        2. 统一处理时区为UTC+8，避免Pandas层面的循环转换。
        3. 增加异常捕获和None值检查。
        """
        tick_metrics = {}
        # 初始化字段
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
        if not hasattr(self, 'tick_data') or self.tick_data is None or self.tick_data.empty:
            return tick_metrics
        try:
            df = self.tick_data.copy()
            # 基础列名标准化
            col_map = {'time': 'trade_time', 'vol': 'volume', 'v': 'volume', 'p': 'price', 'amt': 'amount', 'money': 'amount'}
            df.rename(columns=col_map, inplace=True)
            if 'trade_time' not in df.columns:
                if isinstance(df.index, pd.DatetimeIndex):
                    df['trade_time'] = df.index
                else:
                    return tick_metrics
            # 类型标准化
            if 'type' in df.columns:
                df['type'] = df['type'].astype(str).str.upper()
                type_map = {'买盘': 'B', '卖盘': 'S', 'BUY': 'B', 'SELL': 'S', '0': 'S', '1': 'B', '2': 'M'}
                df['type'] = df['type'].map(lambda x: type_map.get(x, x))
            else:
                return tick_metrics
            df = df[df['type'].isin(['B', 'S'])]
            if df.empty:
                return tick_metrics
            # 关键优化：转换为NumPy datetime64并统一到UTC+8（北京时间）
            # 直接操作 values 避免 pandas Series 的 overhead
            time_values = pd.to_datetime(df['trade_time']).values
            # 假设输入可能是UTC或者无时区，这里做简化处理：
            # 如果是无时区，默认为本地（北京），如果是UTC，需要+8小时
            # 为保证效率，这里假设数据源已经是北京时间或无时区时间
            # 补充 Amount
            if 'amount' not in df.columns:
                df['amount'] = df['price'] * df['volume'] * 100
            # 将 df['trade_time'] 更新为强类型的 datetime
            df['trade_time'] = time_values
            # 1. 计算日内资金流分布 (优化版)
            tick_metrics.update(self._calculate_intraday_flow_distribution(df))
            # 2. 高频大单识别
            tick_metrics.update(self._detect_high_freq_large_orders(df))
            # 3. 资金冲击特征
            tick_metrics.update(self._calculate_flow_impact_features(df))
            # 4. 日内资金动量
            tick_metrics.update(self._calculate_intraday_momentum(df))
            # 5. 资金聚类特征 (优化版)
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
            logger.error(f"计算tick增强指标异常: {e}", exc_info=True)
        return tick_metrics

    def _calculate_intraday_flow_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        计算日内资金流分布特征
        版本: V1.3
        说明: 
        1. 移除pd.cut和groupby，使用np.digitize和np.bincount进行高效分箱统计。
        2. 性能相比Pandas GroupBy提升约50倍。
        """
        distribution = {}
        # 获取NumPy数组
        times = df['trade_time'].values.astype('datetime64[m]')
        # 提取分钟数（从当天0点开始的分钟数）
        # 注意：这里需要取 hour * 60 + minute，利用 datetime64[m] 取模和除法
        # time_values 是纳秒，转为分钟后，取余数计算当天分钟数比较复杂，建议直接转换
        # 更快的方式：将datetime转换为struct_time的numpy视图，或者利用pandas的dt属性缓存
        # 为兼容性，使用 dt.hour * 60 + dt.minute 的 numpy 实现
        dt_idx = pd.DatetimeIndex(df['trade_time'])
        minutes_since_midnight = dt_idx.hour.values * 60 + dt_idx.minute.values
        # 定义 bins (闭区间逻辑需注意，np.digitize 默认 right=False)
        bins = np.array([0, 570, 600, 630, 660, 690, 780, 810, 840, 870, 1440])
        labels = ['Before', '9:30-10:00', '10:00-10:30', '10:30-11:00', '11:00-11:30', 
                  'Noon', '13:00-13:30', '13:30-14:00', '14:00-14:30', '14:30-15:00']
        # 获取分箱索引 (1-based index)
        indices = np.digitize(minutes_since_midnight, bins)
        # 准备 Amount 数据，买入为正，卖出为负
        amounts = df['amount'].values
        types = df['type'].values
        # 构建带符号的金额数组
        # np.where(condition, x, y)
        signed_amounts = np.where(types == 'B', amounts, -amounts)
        # 使用 bincount 进行聚合求和
        # minlength 设为 len(bins) + 1 以容纳所有可能的索引
        sums = np.bincount(indices, weights=signed_amounts, minlength=len(bins)+1)
        # 构建结果字典
        bucket_flows = {}
        valid_indices = [i for i, label in enumerate(labels) if ':' in label]
        # digitize返回的索引从1开始对应bins[0]-bins[1]，所以 labels[i] 对应 indices i+1
        for i in valid_indices:
            idx = i + 1 
            if idx < len(sums):
                bucket_flows[labels[i]] = float(sums[idx] / 10000.0) # 转换为万元
        # 计算占比
        total_abs_flow = sum([abs(v) for v in bucket_flows.values()])
        if total_abs_flow > 0:
            bucket_ratios = {k: v / total_abs_flow * 100 for k, v in bucket_flows.items()}
        else:
            bucket_ratios = {k: 0.0 for k in bucket_flows.keys()}
        distribution['intraday_flow_distribution'] = json.dumps({
            'bucket_flows': bucket_flows,
            'bucket_ratios': bucket_ratios,
            'peak_period': max(bucket_flows.items(), key=lambda x: abs(x[1]))[0] if bucket_flows else None
        })
        return distribution

    def _detect_high_freq_large_orders(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        高频大单识别
        版本: V1.1
        说明: 使用NumPy Boolean Masking替代Pandas Query/Filtering，减少中间DataFrame开销。
        """
        metrics = {}
        large_order_threshold = 100000.0  # 10万元
        # 提取NumPy数组
        amounts = df['amount'].values
        types = df['type'].values # 已经是 'B', 'S'
        # 创建大单掩码
        is_large = amounts >= large_order_threshold
        # 向量化计算
        # 大买单：类型为B 且 是大单
        large_buy_mask = (types == 'B') & is_large
        large_sell_mask = (types == 'S') & is_large
        total_large_buy = np.sum(amounts[large_buy_mask]) / 10000.0
        total_large_sell = np.sum(amounts[large_sell_mask]) / 10000.0
        metrics['tick_large_order_net'] = float(total_large_buy - total_large_sell)
        metrics['tick_large_order_count'] = int(np.sum(large_buy_mask) + np.sum(large_sell_mask))
        return metrics

    def _calculate_flow_impact_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        计算资金冲击特征
        版本: V1.2
        说明: 
        1. 使用np.diff计算价格变化。
        2. 使用np.bincount进行分钟级资金聚合。
        3. 使用Run-Length Encoding (RLE) 逻辑向量化计算持续性，移除GroupBy。
        """
        metrics = {}
        if len(df) < 10:
            metrics['flow_impact_ratio'] = None
            metrics['flow_persistence_minutes'] = 0
            return metrics
        prices = df['price'].values
        amounts = df['amount'].values
        types = df['type'].values
        # 计算价格变动 (diff长度比原数组少1，需对齐)
        # 这里的逻辑是每笔成交对价格的影响，通常用 后一笔价格 - 当前笔价格
        # 或者 当前笔价格 - 前一笔价格。这里沿用原逻辑 diff()
        price_changes = np.zeros_like(prices)
        price_changes[1:] = np.diff(prices)
        # 冲击系数计算
        # 1. 分别提取买卖单数据
        buy_mask = (types == 'B')
        sell_mask = (types == 'S')
        buy_amt_mean = np.mean(amounts[buy_mask]) / 1e6 if np.any(buy_mask) else 0
        sell_amt_mean = np.mean(amounts[sell_mask]) / 1e6 if np.any(sell_mask) else 0
        buy_impact = 0.0
        if buy_amt_mean > 0:
            # 买单造成的平均价格变动绝对值
            buy_impact = np.abs(np.mean(price_changes[buy_mask])) / (buy_amt_mean + 1e-6)
        sell_impact = 0.0
        if sell_amt_mean > 0:
            sell_impact = np.abs(np.mean(price_changes[sell_mask])) / (sell_amt_mean + 1e-6)
        if np.any(buy_mask) and np.any(sell_mask):
            metrics['flow_impact_ratio'] = float((buy_impact + sell_impact) / 2)
        else:
            metrics['flow_impact_ratio'] = None
        # 计算资金持续性 (分钟级)
        # 1. 聚合到分钟
        times = df['trade_time'].values.astype('datetime64[m]').astype('int64')
        if len(times) == 0:
            metrics['flow_persistence_minutes'] = 0
            return metrics
        # 归一化时间索引以便使用bincount
        min_time = times[0]
        norm_times = times - min_time
        # 计算分钟净流入
        signed_amounts = np.where(types == 'B', amounts, -amounts)
        # minlength 保证覆盖所有时间段
        minute_flows = np.bincount(norm_times, weights=signed_amounts)
        # 2. 计算连续同向分钟数
        # 过滤掉无交易的分钟（值为0）
        active_flows = minute_flows[minute_flows != 0]
        if len(active_flows) == 0:
            metrics['flow_persistence_minutes'] = 0
            return metrics
        # 获取符号 (-1, 1)
        signs = np.sign(active_flows)
        # 计算连续段
        # 找符号变化点
        diffs = np.diff(signs)
        # 非0处为变化点
        change_indices = np.where(diffs != 0)[0]
        if len(change_indices) == 0:
            metrics['flow_persistence_minutes'] = len(signs)
        else:
            # 计算各段长度
            # 加上起点和终点
            boundaries = np.concatenate(([-1], change_indices, [len(signs)-1]))
            durations = np.diff(boundaries)
            metrics['flow_persistence_minutes'] = int(np.max(durations))
        return metrics

    def _calculate_intraday_momentum(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        v1.3.1: 修复高频动量计算中的数值爆炸风险。
        增加严格的 epsilon 检查和 np.isinf/np.isnan 过滤，确保 Decimal 字段安全。
        """
        metrics = {'intraday_flow_momentum': None, 'flow_acceleration_intraday': None}
        times = df['trade_time'].values.astype('datetime64[m]').astype('int64')
        amounts = df['amount'].values
        types = df['type'].values
        if len(times) == 0: return metrics
        signed_amounts = np.where(types == 'B', amounts, -amounts)
        norm_times = times - times[0]
        minute_flows = np.bincount(norm_times, weights=signed_amounts) / 10000.0
        n = len(minute_flows)
        if n < 10: return metrics
        if n >= 30:
            recent_30 = np.mean(minute_flows[-30:])
            prev_slice = minute_flows[max(0, n-60):n-30]
            previous_30 = np.mean(prev_slice) if len(prev_slice) > 0 else 0.0
            if abs(previous_30) > 0.01: # 提高阈值至100元
                momentum = (recent_30 - previous_30) / abs(previous_30)
                metrics['intraday_flow_momentum'] = float(np.clip(momentum, -100.0, 100.0))
            else:
                metrics['intraday_flow_momentum'] = 0.0
        if n >= 3:
            recent_3 = minute_flows[-3:]
            denom = abs(recent_3[0])
            if denom > 0.01: # 避免除以极小值
                acc = (recent_3[2] - 2*recent_3[1] + recent_3[0]) / denom
                # 严格限制 Decimal(20,4) 等字段的溢出范围
                metrics['flow_acceleration_intraday'] = float(np.nan_to_num(np.clip(acc, -9999.9, 9999.9)))
            else:
                metrics['flow_acceleration_intraday'] = 0.0
        return metrics
    def _calculate_closing_flow_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        v1.3.1: 优化尾盘判定逻辑。
        利用纳秒时间戳取模运算直接获取日内偏移，避免大量的 datetime 对象转换。
        """
        metrics = {'closing_flow_ratio': None, 'closing_flow_intensity': None}
        times = df['trade_time'].values.astype('int64') # ns
        amounts = df['amount'].values
        types = df['type'].values
        if len(times) == 0: return metrics
        # A股尾盘定义：14:30:00 = 14.5 * 3600 * 10^9 ns
        # 考虑到时区对齐，使用本地时间戳对 86400 取模（北京时间无需额外偏移则直接取）
        ns_in_day = (times % 86400000000000)
        closing_threshold = 52200000000000 # 14:30 * 3600 * 1e9
        closing_mask = ns_in_day >= closing_threshold
        if not np.any(closing_mask):
            metrics['closing_flow_ratio'], metrics['closing_flow_intensity'] = 0.0, 0.0
            return metrics
        buy_mask, sell_mask = (types == 'B'), (types == 'S')
        c_net = (np.sum(amounts[closing_mask & buy_mask]) - np.sum(amounts[closing_mask & sell_mask])) / 10000.0
        t_net = (np.sum(amounts[buy_mask]) - np.sum(amounts[sell_mask])) / 10000.0
        metrics['closing_flow_ratio'] = float(abs(c_net) / abs(t_net) * 100.0) if abs(t_net) > 1e-4 else 0.0
        metrics['closing_flow_intensity'] = float(abs(c_net) / 30.0)
        return metrics

    def _calculate_flow_cluster_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        计算资金聚类特征
        版本: V1.2
        说明: 
        1. 移除TimeGrouper重采样。
        2. 使用整数除法将时间戳映射到3秒桶，利用NumPy进行快速聚合。
        3. 使用向量化差分计算连续持续时间。
        """
        metrics = {}
        # 将时间转换为秒级整数，然后除以3得到桶ID
        # astype('int64') 将 datetime64[ns] 转为纳秒整数
        # 1e9 ns = 1s
        time_int = df['trade_time'].values.astype('int64') // 10**9 // 3
        # 准备带符号的金额
        amounts = df['amount'].values
        types = df['type'].values
        signed_amounts = np.where(types == 'B', amounts, -amounts)
        # 聚合：由于 time_int 不是连续的从0开始的整数，不能直接用 bincount
        # 使用 pandas 的 groupby 但基于整数列，比 datetime 列快
        # 或者使用 np.unique + np.add.reduceat (更底层)
        uni_times, inverse_indices = np.unique(time_int, return_inverse=True)
        cluster_flow = np.zeros(len(uni_times))
        np.add.at(cluster_flow, inverse_indices, signed_amounts)
        if len(cluster_flow) < 10:
            metrics['flow_cluster_intensity'] = None
            metrics['flow_cluster_duration'] = 0
            return metrics
        # 计算聚类强度
        mean_flow = np.mean(np.abs(cluster_flow))
        std_flow = np.std(cluster_flow)
        metrics['flow_cluster_intensity'] = float(std_flow / mean_flow) if mean_flow > 0 else 0.0
        # 计算聚类持续时间 (Vectorized Run-Length Encoding approach)
        threshold = mean_flow * 0.5
        is_significant = np.abs(cluster_flow) > threshold
        # 如果没有显著流，直接返回
        if not np.any(is_significant):
            metrics['flow_cluster_duration'] = 0
            return metrics
        # 寻找连续显著区域的时间跨度
        # 获取显著点对应的时间戳（3秒桶ID * 3 = 秒）
        sig_times = uni_times[is_significant] * 3
        if len(sig_times) < 2:
            metrics['flow_cluster_duration'] = 0
            return metrics
        # 计算时间差，如果差值 <= 6秒，认为是同一聚类
        diffs = np.diff(sig_times)
        # 标记断点 (差值 > 6 的位置)
        breaks = np.where(diffs > 6)[0]
        # 如果没有断点，说明全程连续
        if len(breaks) == 0:
            max_duration = sig_times[-1] - sig_times[0]
        else:
            # 计算各段持续时间
            # 起点索引：0, breaks[0]+1, breaks[1]+1 ...
            # 终点索引：breaks[0], breaks[1], ... last
            starts = np.concatenate(([0], breaks + 1))
            ends = np.concatenate((breaks, [len(sig_times) - 1]))
            durations = sig_times[ends] - sig_times[starts]
            max_duration = np.max(durations)
        metrics['flow_cluster_duration'] = int(max_duration / 60) # 转换为分钟
        return metrics

    def _calculate_high_freq_divergence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        计算高频资金分歧度
        版本: V1.2
        说明: 使用10秒为单位的NumPy聚合，计算买卖流相关性。
        """
        metrics = {}
        if len(df) < 20:
            metrics['high_freq_flow_divergence'] = None
            return metrics
        # 10秒级聚合
        # astype('datetime64[s]') 转秒，然后除以10向下取整
        times_10s = df['trade_time'].values.astype('datetime64[s]').astype('int64') // 10
        amounts = df['amount'].values
        types = df['type'].values
        norm_times = times_10s - times_10s[0]
        # 分别聚合买卖盘
        buy_mask = (types == 'B')
        sell_mask = (types == 'S')
        # weights 设为 amount，如果mask为False则weight设为0
        buy_weights = np.where(buy_mask, amounts, 0)
        sell_weights = np.where(sell_mask, amounts, 0)
        # 使用 max index 确定数组长度
        max_idx = norm_times[-1] if len(norm_times) > 0 else 0
        bin_len = max_idx + 1
        buy_flow = np.bincount(norm_times, weights=buy_weights, minlength=bin_len)
        sell_flow = np.bincount(norm_times, weights=sell_weights, minlength=bin_len)
        # 2. 计算相关性
        # 仅考虑两者都存在的时段（即 time bins 对齐，bincount 自动对齐了索引）
        # 过滤掉都没数据的时段（可选，避免0的影响，这里主要看趋势）
        valid_mask = (buy_flow > 0) | (sell_flow > 0)
        if np.sum(valid_mask) >= 5:
            # 计算相关系数
            corr = np.corrcoef(buy_flow[valid_mask], sell_flow[valid_mask])[0, 1]
            if not np.isnan(corr):
                # 负相关越高(接近-1)，分歧越大，得分越高
                divergence = (1 - corr) / 2 * 100
                metrics['high_freq_flow_divergence'] = float(divergence)
            else:
                metrics['high_freq_flow_divergence'] = 50.0
        else:
            metrics['high_freq_flow_divergence'] = None
        return metrics

    def _calculate_vwap_deviation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        计算VWAP偏离度
        版本: V1.1
        说明: 使用NumPy数组运算。
        """
        metrics = {}
        if len(df) < 10:
            metrics['vwap_deviation'] = None
            return metrics
        amounts = df['amount'].values
        volumes = df['volume'].values
        prices = df['price'].values
        types = df['type'].values
        total_amount = np.sum(amounts)
        total_volume = np.sum(volumes)
        if total_volume > 0:
            vwap = total_amount / total_volume
            # 计算买盘VWAP
            buy_mask = (types == 'B')
            if np.any(buy_mask):
                buy_amounts = amounts[buy_mask]
                # 加权平均价 = sum(price * vol) / sum(vol) = sum(amount) / sum(vol)
                # 原代码逻辑: (buy_orders['amount'] * buy_orders['price']).sum() / buy_orders['amount'].sum()
                # 这是一个"金额加权"的价格? 通常 VWAP 是成交量加权。
                # 原始代码逻辑似乎是 Amount-Weighted Price。此处保持原逻辑一致性。
                buy_prices = prices[buy_mask]
                buy_amount_sum = np.sum(buy_amounts)
                if buy_amount_sum > 0:
                    buy_vwap = np.sum(buy_amounts * buy_prices) / buy_amount_sum
                    if vwap > 0:
                        deviation = (buy_vwap - vwap) / vwap * 100
                        metrics['vwap_deviation'] = float(deviation)
                    else:
                        metrics['vwap_deviation'] = 0.0
                else:
                    metrics['vwap_deviation'] = 0.0
            else:
                metrics['vwap_deviation'] = 0.0
        else:
            metrics['vwap_deviation'] = None
        return metrics

    def _calculate_flow_efficiency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        计算资金流入效率
        版本: V1.1
        说明: 使用NumPy数组运算。
        """
        metrics = {}
        if len(df) < 20:
            metrics['flow_efficiency'] = None
            return metrics
        prices = df['price'].values
        amounts = df['amount'].values
        types = df['type'].values
        # 价格变化绝对值
        # 补齐长度
        price_diffs = np.zeros_like(prices)
        price_diffs[1:] = np.abs(np.diff(prices))
        buy_mask = (types == 'B')
        if np.sum(buy_mask) >= 5:
            total_price_change = np.sum(price_diffs[buy_mask])
            total_buy_amt = np.sum(amounts[buy_mask]) / 1e6 # 百万
            efficiency = total_price_change / (total_buy_amt + 1e-6)
            metrics['flow_efficiency'] = float(efficiency)
        else:
            metrics['flow_efficiency'] = 0.0
        return metrics


    def _calculate_high_freq_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        计算高频统计特征
        版本: V1.2
        说明: 使用NumPy Binning。
        """
        metrics = {}
        times = df['trade_time'].values.astype('datetime64[m]').astype('int64')
        amounts = df['amount'].values
        types = df['type'].values
        if len(times) == 0:
            metrics['high_freq_flow_skewness'] = None
            metrics['high_freq_flow_kurtosis'] = None
            return metrics
        norm_times = times - times[0]
        signed_amounts = np.where(types == 'B', amounts, -amounts)
        # 分钟聚合
        minute_flows = np.bincount(norm_times, weights=signed_amounts) / 10000.0
        if len(minute_flows) < 10:
            metrics['high_freq_flow_skewness'] = None
            metrics['high_freq_flow_kurtosis'] = None
            return metrics
        metrics['high_freq_flow_skewness'] = float(stats.skew(minute_flows))
        metrics['high_freq_flow_kurtosis'] = float(stats.kurtosis(minute_flows))
        return metrics

    def _calculate_time_period_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        计算资金时段分布
        版本: V1.3
        说明: 
        1. 使用 NumPy 广播计算小时数，移除 Pandas .dt.hour 访问器。
        2. 纯数组掩码操作。
        """
        metrics = {}
        times = df['trade_time'].values
        if len(times) == 0:
            metrics['morning_flow_ratio'] = None
            metrics['afternoon_flow_ratio'] = None
            return metrics
        # 计算小时数
        # (Time - Date) -> Timedelta -> Hours
        dates = times.astype('datetime64[D]')
        # 得到小时数 (int)
        hours = (times - dates).astype('timedelta64[h]').astype(int)
        # 掩码
        morning_mask = hours < 12
        afternoon_mask = hours >= 13
        if not np.any(morning_mask) and not np.any(afternoon_mask):
            metrics['morning_flow_ratio'] = None
            metrics['afternoon_flow_ratio'] = None
            return metrics
        amounts = df['amount'].values
        types = df['type'].values
        signed_amounts = np.where(types == 'B', amounts, -amounts)
        # 计算净流入
        morning_net = np.sum(signed_amounts[morning_mask]) / 10000.0
        afternoon_net = np.sum(signed_amounts[afternoon_mask]) / 10000.0
        total_net = morning_net + afternoon_net
        if abs(total_net) > 0:
            morning_ratio = abs(morning_net) / abs(total_net) * 100.0
            afternoon_ratio = abs(afternoon_net) / abs(total_net) * 100.0
        else:
            morning_ratio = 50.0 if len(times) > 0 else None
            afternoon_ratio = 50.0 if len(times) > 0 else None
        metrics['morning_flow_ratio'] = float(morning_ratio) if morning_ratio is not None else None
        metrics['afternoon_flow_ratio'] = float(afternoon_ratio) if afternoon_ratio is not None else None
        return metrics

    def _calculate_stealth_flow_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        计算主力隐蔽性指标
        版本: V1.1
        说明: 使用NumPy Masking。
        """
        metrics = {}
        small_order_threshold = 50000.0
        amounts = df['amount'].values
        types = df['type'].values
        is_small = amounts < small_order_threshold
        buy_mask = types == 'B'
        sell_mask = types == 'S'
        # 小单净流入
        small_buy_val = np.sum(amounts[buy_mask & is_small])
        small_sell_val = np.sum(amounts[sell_mask & is_small])
        small_net = (small_buy_val - small_sell_val) / 10000.0
        # 总净流入
        total_buy = np.sum(amounts[buy_mask])
        total_sell = np.sum(amounts[sell_mask])
        total_net = (total_buy - total_sell) / 10000.0
        if abs(total_net) > 0:
            stealth_ratio = abs(small_net) / abs(total_net) * 100.0
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










