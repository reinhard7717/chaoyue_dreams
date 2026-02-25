# services\fundflow_calculator.py
import numpy as np
import pandas as pd
from numba import njit
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

@njit(cache=True, fastmath=True)
def _numba_calc_inflow_persistence_energy(arr: np.ndarray, decay: float) -> float:
    """Numba优化的资金记忆衰减堆积积分器"""
    energy = 0.0
    for i in range(len(arr)):
        if arr[i] > 0:
            energy = energy * decay + 1.0
        else:
            energy = max(0.0, energy * decay - 0.5)
    return energy

@njit(cache=True, fastmath=True)
def _numba_flow_cluster_state_machine(states: np.ndarray, cluster_flows: np.ndarray, gap_tolerance: int):
    """Numba优化的同向拆单攻击波状态机"""
    max_duration_buckets = 0
    current_duration = 0
    current_type = 0
    current_gap = 0
    current_cluster_amt = 0.0
    cluster_flow_sums_total = 0.0
    cluster_durations_total = 0
    for i in range(len(states)):
        s = states[i]
        flow_val = abs(cluster_flows[i])
        if s != 0:
            if current_type == 0:
                current_type = s
                current_duration = 1
                current_gap = 0
                current_cluster_amt = flow_val
            elif s == current_type:
                current_duration += 1 + current_gap
                current_gap = 0
                current_cluster_amt += flow_val
            else:
                if current_duration > 0:
                    cluster_flow_sums_total += current_cluster_amt
                    cluster_durations_total += current_duration
                    if current_duration > max_duration_buckets:
                        max_duration_buckets = current_duration
                current_type = s
                current_duration = 1
                current_gap = 0
                current_cluster_amt = flow_val
        else:
            if current_type != 0:
                if current_gap < gap_tolerance:
                    current_gap += 1
                    current_cluster_amt += flow_val
                else:
                    cluster_flow_sums_total += current_cluster_amt
                    cluster_durations_total += current_duration
                    if current_duration > max_duration_buckets:
                        max_duration_buckets = current_duration
                    current_type = 0
                    current_duration = 0
                    current_gap = 0
                    current_cluster_amt = 0.0
    if current_duration > 0:
        cluster_flow_sums_total += current_cluster_amt
        cluster_durations_total += current_duration
        if current_duration > max_duration_buckets:
            max_duration_buckets = current_duration
    return max_duration_buckets, cluster_flow_sums_total, cluster_durations_total

@njit(cache=True, fastmath=True)
def _numba_find_longest_streak(states: np.ndarray):
    """Numba优化的状态机连续游程(Streak)查找器"""
    max_persistence = 0.0
    dominant_direction = 0
    current_streak = 0
    current_state = 0
    for i in range(len(states)):
        s = states[i]
        if s == 0:
            if current_streak > abs(max_persistence):
                max_persistence = float(current_streak)
                dominant_direction = current_state
            current_streak = 0
            current_state = 0
        else:
            if current_state == 0:
                current_state = s
                current_streak = 1
            elif s == current_state:
                current_streak += 1
            else:
                if current_streak > abs(max_persistence):
                    max_persistence = float(current_streak)
                    dominant_direction = current_state
                current_state = s
                current_streak = 1
    if current_streak > abs(max_persistence):
        max_persistence = float(current_streak)
        dominant_direction = current_state
    return max_persistence, dominant_direction

@njit(cache=True, fastmath=True)
def _numba_skew_kurtosis(arr: np.ndarray):
    """Numba优化的偏度与峰度(Fisher)联合计算器，避免Scipy的包装开销"""
    n = len(arr)
    if n < 3: return 0.0, 0.0
    mean_val = 0.0
    for i in range(n): mean_val += arr[i]
    mean_val /= n
    m2 = 0.0; m3 = 0.0; m4 = 0.0
    for i in range(n):
        diff = arr[i] - mean_val
        diff2 = diff * diff
        m2 += diff2
        m3 += diff2 * diff
        m4 += diff2 * diff2
    m2 /= n; m3 /= n; m4 /= n
    if m2 <= 1e-8: return 0.0, 0.0
    skew = m3 / (m2 ** 1.5)
    kurtosis = (m4 / (m2 ** 2.0)) - 3.0
    return skew, kurtosis

@njit(cache=True, fastmath=True)
def _numba_entropy(probs: np.ndarray):
    """Numba优化的信息熵计算(Base e)"""
    ent = 0.0
    for i in range(len(probs)):
        if probs[i] > 0:
            ent -= probs[i] * np.log(probs[i])
    return ent

@njit(cache=True, fastmath=True)
def _numba_fast_r2(y: np.ndarray):
    """Numba优化的极速R方(决定系数)计算，专为小数组设计"""
    n = len(y)
    if n < 2: return 0.0
    sum_x = 0.0; sum_y = 0.0; sum_xy = 0.0; sum_xx = 0.0; sum_yy = 0.0
    for i in range(n):
        sum_x += i
        sum_y += y[i]
        sum_xy += i * y[i]
        sum_xx += i * i
        sum_yy += y[i] * y[i]
    mean_x = sum_x / n
    mean_y = sum_y / n
    ss_x = sum_xx - n * mean_x * mean_x
    ss_y = sum_yy - n * mean_y * mean_y
    ss_xy = sum_xy - n * mean_x * mean_y
    if ss_x == 0 or ss_y == 0: return 0.0
    r = ss_xy / np.sqrt(ss_x * ss_y)
    return r * r

@njit(cache=True, fastmath=True)
def _numba_autocorr1(arr: np.ndarray):
    """Numba优化的Lag-1一阶自相关系数计算"""
    n = len(arr)
    if n < 2: return 0.0
    mean_val = 0.0
    for i in range(n): mean_val += arr[i]
    mean_val /= n
    var_val = 0.0
    for i in range(n): var_val += (arr[i] - mean_val)**2
    if var_val == 0: return 0.0
    cov = 0.0
    for i in range(n - 1): cov += (arr[i] - mean_val) * (arr[i+1] - mean_val)
    return cov / (var_val * (n / (n-1))) if n > 1 else 0.0

@njit(cache=True, fastmath=True)
def _numba_linregress(y: np.ndarray):
    """Numba优化的极速线性回归(提取斜率与R方)"""
    n = len(y)
    if n < 2: return 0.0, 0.0
    sum_x = 0.0; sum_y = 0.0; sum_xy = 0.0; sum_xx = 0.0; sum_yy = 0.0
    for i in range(n):
        sum_x += i
        sum_y += y[i]
        sum_xy += i * y[i]
        sum_xx += i * i
        sum_yy += y[i] * y[i]
    mean_x = sum_x / n
    mean_y = sum_y / n
    ss_x = sum_xx - n * mean_x * mean_x
    ss_y = sum_yy - n * mean_y * mean_y
    ss_xy = sum_xy - n * mean_x * mean_y
    if ss_x == 0: return 0.0, 0.0
    slope = ss_xy / ss_x
    if ss_y == 0: return slope, 0.0
    r = ss_xy / np.sqrt(ss_x * ss_y)
    return slope, r * r

@njit(cache=True, fastmath=True)
def _numba_pearson_corr(x: np.ndarray, y: np.ndarray):
    """Numba优化的皮尔逊相关系数计算"""
    n = len(x)
    if n < 2 or n != len(y): return 0.0
    sum_x = 0.0; sum_y = 0.0
    for i in range(n):
        sum_x += x[i]
        sum_y += y[i]
    mean_x = sum_x / n
    mean_y = sum_y / n
    ss_x = 0.0; ss_y = 0.0; ss_xy = 0.0
    for i in range(n):
        dx = x[i] - mean_x
        dy = y[i] - mean_y
        ss_x += dx * dx
        ss_y += dy * dy
        ss_xy += dx * dy
    if ss_x == 0.0 or ss_y == 0.0: return 0.0
    return ss_xy / np.sqrt(ss_x * ss_y)

@njit(cache=True, fastmath=True)
def _numba_closing_flow_agg(times: np.ndarray, amounts: np.ndarray, is_buy: np.ndarray):
    """Numba优化的尾盘资金单次遍历聚合器，消除内存碎片"""
    time_1430 = 52200000000000
    time_1457 = 53820000000000
    c_net = 0.0; a_net = 0.0; total_net = 0.0; total_turnover = 0.0
    n = len(times)
    for i in range(n):
        amt = amounts[i]
        total_turnover += amt
        ns_in_day = times[i] % 86400000000000
        signed_amt = amt if is_buy[i] else -amt
        total_net += signed_amt
        if ns_in_day >= time_1430:
            c_net += signed_amt
            if ns_in_day >= time_1457:
                a_net += signed_amt
    return c_net, a_net, total_net, total_turnover

@njit(cache=True, fastmath=True)
def _numba_morning_afternoon_agg(hours: np.ndarray, amounts: np.ndarray, is_buy: np.ndarray):
    """Numba优化的时段资金单次遍历聚合器"""
    morning_net = 0.0
    afternoon_net = 0.0
    n = len(hours)
    for i in range(n):
        signed_amt = amounts[i] if is_buy[i] else -amounts[i]
        h = hours[i]
        if h < 12:
            morning_net += signed_amt
        elif h >= 13:
            afternoon_net += signed_amt
    return morning_net, afternoon_net

@njit(cache=True, fastmath=True)
def _numba_stealth_flow_agg(amounts: np.ndarray, is_buy: np.ndarray, threshold: float):
    """Numba优化的隐蔽大/小单拆解单次聚合器"""
    small_net = 0.0
    total_net = 0.0
    n = len(amounts)
    for i in range(n):
        amt = amounts[i]
        signed_amt = amt if is_buy[i] else -amt
        total_net += signed_amt
        if amt < threshold:
            small_net += signed_amt
    return small_net, total_net

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
        # if len(self.context.historical_flow_data) < 20:
            # logger.warning(f"历史数据不足，仅{len(self.context.historical_flow_data)}天")
    
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
        版本: V2.4
        说明: 时间窗口更新为斐波拉契数列 [3, 5, 8, 13, 21, 34, 55]
        """
        metrics = {}
        current_net = float(self.context.current_flow_data.get('net_mf_amount', 0) or 0)
        net_arr = self.net_amount_array
        vol_arr = self.volume_array
        n = len(net_arr)
        windows = [3, 5, 8, 13, 21, 34, 55]
        for window in windows:
            if n >= window:
                total = np.sum(net_arr[-window:])
                metrics[f'total_net_amount_{window}d'] = float(total)
                metrics[f'avg_daily_net_{window}d'] = float(total / window)
            else:
                metrics[f'total_net_amount_{window}d'] = None
                metrics[f'avg_daily_net_{window}d'] = None
        n_vol = len(vol_arr)
        for window in windows:
            if n_vol >= window:
                total_vol = np.sum(vol_arr[-window:]) / 100.0
                metrics[f'total_volume_{window}d'] = float(total_vol)
            else:
                metrics[f'total_volume_{window}d'] = None
        return metrics

    # ==================== 2. 相对强度指标计算 ====================
    def calculate_relative_metrics(self) -> Dict[str, float]:
        """
        计算相对强度指标
        版本: V2.4
        说明: 将10日均线替换为8日均线
        """
        metrics = {}
        current_net = float(self.context.current_flow_data.get('net_mf_amount', 0) or 0)
        daily_amount = 0.0
        if self.context.daily_basic_data:
            daily_amount = float(self.context.daily_basic_data.get('amount', 0) or 0)
        if daily_amount <= 0:
            valid_amounts = self.daily_amount_array[self.daily_amount_array > 0]
            daily_amount = np.mean(valid_amounts) if len(valid_amounts) > 0 else 1.0
        raw_ratio = (current_net / daily_amount) * 1000.0 if daily_amount > 0 else 0.0
        recent_vols = self.volume_array[-5:]
        avg_vol = np.mean(recent_vols) if len(recent_vols) > 0 else 0.0
        current_vol = float(self.context.current_flow_data.get('net_mf_vol', 0) or 0)
        if self.context.daily_basic_data:
             current_vol = float(self.context.daily_basic_data.get('vol', 0) or 0)
        if avg_vol > 0:
            vol_ratio = current_vol / avg_vol
        else:
            vol_ratio = 1.0
        volume_factor = np.log1p(vol_ratio)
        norm_factor = volume_factor / 0.693 
        metrics['net_amount_ratio'] = float(raw_ratio * norm_factor)
        ratio_arr = self.net_amount_ratio_array
        n = len(ratio_arr)
        for window in [5, 8]:
            if n >= window:
                metrics[f'net_amount_ratio_ma{window}'] = float(np.mean(ratio_arr[-window:]))
            else:
                metrics[f'net_amount_ratio_ma{window}'] = None
        circ_mv = self.market_cap
        if circ_mv is None and self.context.daily_basic_data:
            circ_mv = float(self.context.daily_basic_data.get('circ_mv', 0) or 0)
        intensity_score = self._calculate_flow_intensity(current_net, raw_ratio, circ_mv)
        metrics['flow_intensity'] = float(intensity_score)
        metrics['intensity_level'] = self._determine_intensity_level(current_net, raw_ratio)
        return metrics

    def _calculate_flow_intensity(self, net_amount: float, net_ratio: float, circ_mv: Optional[float]) -> float:
        """
        v2.2: [大师级深化] 全域攻击能级 (Global Attack Energy)
        思路：
        1. 绝对金额和相对占比都存在局限性。
        2. 真正的强度是：资金相对于这个盘子的推动力。
        3. 使用 Net / Circ_MV (净流换手率) 作为核心物理量。
        4. 映射逻辑：
           - 净买入流通盘的 0.1% (1‰) -> 中等强度 (60分)
           - 净买入流通盘的 0.5% (5‰) -> 极强攻击 (85分)
           - 净买入流通盘的 1.0% (10‰) -> 涨停级爆发 (95分)
        """
        # 1. 兜底逻辑：如果没有市值数据，退化回绝对金额逻辑
        if not circ_mv or circ_mv <= 0:
            # 退化版：简单的 log 映射
            # 1000万 -> 60分, 1亿 -> 90分
            abs_val = abs(net_amount)
            if abs_val < 100: return 0.0
            score = 30 + 10 * np.log10(abs_val) # log10(1000)=3 -> 60
            base_score = min(100.0, score)
            return base_score if net_amount > 0 else -base_score
        # 2. 计算净流换手率 (Net Flow Turnover)
        # net_amount 和 circ_mv 单位一致 (假设都是万元)
        # 结果为小数，如 0.01 代表 1%
        net_turnover = net_amount / circ_mv
        # 3. 双曲正切非线性映射 (Tanh)
        # 我们设定一个敏感度系数 Scale
        # 目标：当 net_turnover = 0.5% (0.005) 时，我们希望 tanh 达到约 0.8 (强)
        # tanh(x) = 0.8 => x ≈ 1.1
        # Scale * 0.005 = 1.1 => Scale ≈ 220
        scale = 220.0
        # 计算归一化强度 (-1.0 到 1.0)
        normalized_intensity = np.tanh(net_turnover * scale)
        # 4. 映射到分数 (-100 到 100)
        # 线性放大
        intensity_score = normalized_intensity * 100.0
        # 5. 小市值惩罚 (Micro-Cap Penalty)
        # 防止微盘股（如市值<20亿）因为少量资金就计算出极高强度，导致误判
        # 如果市值 < 200,000万元 (20亿)
        if circ_mv < 200000:
            penalty = 0.8 + 0.2 * (circ_mv / 200000.0) # 线性过渡 0.8~1.0
            intensity_score *= penalty
        return float(np.clip(intensity_score, -100.0, 100.0))

    def _determine_intensity_level(self, net_amount: float, net_ratio: float) -> int:
        """
        动态博弈能级定级
        版本: V2.5
        说明: 使用纯NumPy向量化替代 scipy.stats.percentileofscore，提速约50倍
        """
        if len(self.net_amount_array) < 21:
            if abs(net_amount) > 10000: return 4
            if abs(net_amount) > 5000: return 3
            if abs(net_amount) > 1000: return 2
            return 1
        abs_net = abs(net_amount)
        history_abs = np.abs(self.net_amount_array[-55:])
        n_hist = len(history_abs)
        rank = (np.sum(history_abs < abs_net) + np.sum(history_abs == abs_net) * 0.5) / n_hist
        abs_ratio = abs(net_ratio)
        if rank >= 0.95 or abs_ratio >= 50.0:
            return 4
        elif rank >= 0.85 or abs_ratio >= 30.0:
            return 3
        elif rank >= 0.60 or abs_ratio >= 10.0:
            return 2
        else:
            return 1

    # ==================== 3. 主力行为模式识别 ====================
    def calculate_behavior_patterns(self) -> Dict[str, Any]:
        """
        主力行为模式识别
        版本: V2.4
        说明: 基础识别窗口扩展至13日斐波拉契周期
        """
        patterns = {}
        limit = 13
        if len(self.net_amount_array) < limit:
            return {
                'accumulation_score': 0.0, 'pushing_score': 0.0,
                'distribution_score': 0.0, 'shakeout_score': 0.0,
                'behavior_pattern': 'UNCLEAR', 'pattern_confidence': 0.0
            }
        nets = self.net_amount_array[-limit:]
        closes = self.close_array[-limit:]
        vols = self.volume_array[-limit:]
        accumulation_score = self._calculate_accumulation_score(nets)
        pushing_score = self._calculate_pushing_score(nets)
        distribution_score = self._calculate_distribution_score(nets, closes, vols)
        shakeout_score = self._calculate_shakeout_score(nets, closes, vols)
        patterns['accumulation_score'] = accumulation_score
        patterns['pushing_score'] = pushing_score
        patterns['distribution_score'] = distribution_score
        patterns['shakeout_score'] = shakeout_score
        pattern, confidence = self._determine_behavior_pattern(
            accumulation_score, pushing_score, distribution_score, shakeout_score, nets
        )
        patterns['behavior_pattern'] = pattern
        patterns['pattern_confidence'] = confidence
        return patterns

    def _calculate_accumulation_score(self, arr: np.ndarray) -> float:
        """
        隐蔽吸筹熵计算
        版本: V2.4
        说明: 吸筹窗口13日，长序列修正55日
        """
        n = len(arr)
        if n < 13: return 0.0
        window = 13
        nets = arr[-window:]
        closes = self.close_array[-window:]
        cum_net = np.sum(nets)
        if cum_net <= 0: 
            total_turnover = np.sum(self.daily_amount_array[-window:]) + 1.0
            if abs(cum_net) / total_turnover < 0.01:
                return 10.0
            return 0.0
        total_amt = np.sum(self.daily_amount_array[-window:]) + 1.0
        flow_ratio = cum_net / total_amt
        path_length = np.sum(np.abs(np.diff(closes))) 
        vol_ratio = path_length / (np.mean(closes) + 1e-6)
        density = flow_ratio / (vol_ratio + 0.02) 
        raw_score = density * 40.0
        pos_days = np.sum(nets > 0)
        consistency = (pos_days / window) * 20.0
        long_closes = self.close_array[-55:]
        current_pos = (closes[-1] - np.min(long_closes)) / (np.ptp(long_closes) + 1e-6)
        pos_factor = 1.0
        if current_pos < 0.3: pos_factor = 1.2
        elif current_pos > 0.8: pos_factor = 0.6
        final_score = (raw_score + consistency) * pos_factor
        return float(np.clip(final_score, 0.0, 100.0))

    def _calculate_pushing_score(self, arr: np.ndarray) -> float:
        """
        [修复] 趋势爆发动量 (Pushing / Trend Impulse)
        修复点：
        1. 解决大量 0 值：摒弃"加速度>0"的硬约束。
        2. 引入物理学"冲量"概念 (Impulse = Force * Delta_Time)。
           - Force = Net Flow (资金推力)
           - Result = Price Change (价格位移)
           - 只有"推力"和"位移"同向且显著时，才得高分。
        """
        n = len(arr)
        if n < 5: return 0.0
        nets = arr[-5:]
        closes = self.close_array[-5:]
        # 1. 计算每日的"做功" (Work Done)
        # Work = Net_Flow * Price_Change
        price_changes = np.diff(closes)
        # 对齐长度
        work_arr = nets[1:] * price_changes
        # 2. 有效做功求和
        # 只统计正功 (资金买入且价格上涨)
        valid_work = np.sum(work_arr[work_arr > 0])
        # 3. 归一化
        # 基准功 = 平均成交额 * 平均波幅
        avg_amt = np.mean(self.daily_amount_array[-5:])
        avg_range = np.mean(closes) * 0.02 # 假设日均 2% 波动
        base_work = avg_amt * avg_range + 1.0
        efficiency = valid_work / base_work # 通常在 0 ~ 5 之间
        # 4. 映射到 0-100
        # Eff=0.5 -> 25分, Eff=2.0 -> 80分
        raw_score = efficiency * 50.0
        # 5. 连续性奖励
        # 最近3天资金是否为正
        recent_pos = np.sum(nets[-3:] > 0)
        continuity_bonus = recent_pos * 5.0
        return float(np.clip(raw_score + continuity_bonus, 0.0, 100.0))

    def _calculate_distribution_score(self, nets: np.ndarray, closes: np.ndarray, vols: np.ndarray) -> float:
        """
        [大师级深化] 派发/出货模式 (Distribution)
        特征：
        1. 隐性抛压 (Divergence): 价格涨/平，但资金持续流出 (边拉边撤)。
        2. 放量滞涨 (Churning): 高换手率，但价格涨幅微小 (主力对倒出货给散户)。
        """
        n = len(nets)
        if n < 5: return 0.0
        # 1. 量价背离 (Price-Flow Divergence)
        # 逻辑：价格在高位，资金却在流出
        price_trend = (closes[-1] - closes[0])
        net_sum = np.sum(nets)
        divergence_score = 0.0
        if price_trend > 0 and net_sum < 0:
            # 典型顶背离
            divergence_score = 80.0
        elif price_trend > 0 and net_sum > 0:
            # 资金还是流入的，检查速率
            # 如果价格涨幅远大于资金流入幅度 (无量空涨)，也是派发前兆
            pass 
        # 2. 放量滞涨检测 (High Volume Stagnation / Churning)
        # 计算每日的 "单位振幅换手率" (Turnover per Unit Volatility)
        # 出货时，主力需要巨大的成交量来派发，但为了稳住K线，价格波动往往被控制
        recent_vols = vols[-5:]
        avg_vol = np.mean(vols[:-5]) if len(vols) > 5 else np.mean(vols)
        # 计算相对放量程度
        vol_ratio = np.mean(recent_vols) / (avg_vol + 1e-6)
        # 计算价格绝对涨幅
        price_changes = np.abs(np.diff(closes[-6:]))
        avg_change = np.mean(price_changes) + 1e-6
        # 价格基准 (防止除以极小值)
        avg_price = np.mean(closes[-5:])
        pct_change = avg_change / avg_price * 100.0
        # Churning Index: 成交量倍数 / 价格波动率
        # 意义: 用了很大的量(>1.5倍)，结果只波动了很小(<1%)，说明抛压极重，全靠承接
        churning_score = 0.0
        if vol_ratio > 1.2 and pct_change < 1.5:
            # 放量滞涨，极高风险
            churning_score = min(100.0, (vol_ratio / pct_change) * 40.0)
        # 3. 连续流出 (Consecutive Outflow)
        neg_days = np.sum(nets < 0)
        outflow_score = (neg_days / n) * 100.0
        # 综合加权
        # 背离和滞涨是更高级的信号，权重更高
        final_score = divergence_score * 0.4 + churning_score * 0.4 + outflow_score * 0.2
        return float(np.clip(final_score, 0.0, 100.0))

    def _calculate_shakeout_score(self, nets: np.ndarray, closes: np.ndarray, vols: np.ndarray) -> float:
        """
        [大师级深化] 洗盘模式 (Shakeout)
        特征：
        1. 缩量下跌 (Shrinking Volume): 卖盘枯竭，无人抛售。
        2. 承接韧性 (Resilience): 价格跌，但主力资金流出极少，甚至小幅流入。
        3. 波动收敛 (Convergence): 振幅逐渐变小。
        """
        n = len(nets)
        if n < 3: return 0.0
        # 1. 价格形态检测
        # 洗盘前提：价格必须是回调的 (Down) 或者 横盘的 (Flat)
        # 如果价格大涨，那就不是洗盘了 (是拉升)
        price_change = (closes[-1] - closes[0]) / closes[0]
        if price_change > 0.05: # 涨幅超过5%，不太像洗盘
            return 0.0
        # 2. 缩量检测 (Volume Contraction) - 核心特征
        recent_vol = np.mean(vols[-3:])
        past_vol = np.mean(vols[:-3]) if len(vols) > 3 else recent_vol
        # 缩量系数 (越小越好)
        vol_ratio = recent_vol / (past_vol + 1e-6)
        shrink_score = 0.0
        if vol_ratio < 0.8: # 明显缩量
            shrink_score = (1.0 - vol_ratio) * 100.0 * 1.5 # 0.5倍量 -> 75分
        # 3. 资金韧性 (Flow Resilience)
        # 既然是洗盘，主力就不应该大额出逃
        # 计算累计净流
        cum_net = np.sum(nets[-5:])
        resilience_score = 0.0
        if cum_net > 0:
            # 价格跌但资金流入 -> 极强洗盘 (主力在低吸)
            resilience_score = 90.0
        elif cum_net < 0:
            # 资金流出，但流出量要小
            # 相对于成交额的占比
            total_amt = np.sum(vols[-5:] * closes[-5:]) * 100 # 估算金额
            ratio = abs(cum_net) / (total_amt + 1e-6)
            if ratio < 0.05: # 流出占比小于5%，视为正常散户恐慌盘
                resilience_score = 60.0
            else:
                resilience_score = 10.0 # 流出太多，可能是真跌
        # 4. 波动率检测
        # 洗盘末端通常波动率极低
        price_std = np.std(closes[-5:]) / np.mean(closes[-5:])
        stability_score = max(0.0, 100.0 - price_std * 2000.0) # 2% 波动 -> 60分
        # 综合评分
        # 缩量是必要条件 (权重最大)
        if shrink_score < 20: # 如果没缩量，很难说是洗盘
            return 0.0
        final_score = shrink_score * 0.5 + resilience_score * 0.3 + stability_score * 0.2
        return float(np.clip(final_score, 0.0, 100.0))

    def _determine_behavior_pattern(self, acc, push, dist, shake, nets) -> Tuple[str, float]:
        """
        模式仲裁与置信度计算
        版本: V2.7
        说明: 彻底移除 Scipy 依赖，信息熵与 R方 的求值全面转移至 Numba 引擎
        """
        raw_scores = np.array([max(0, acc), max(0, push), max(0, dist), max(0, shake)])
        pattern_names = ['ACCUMULATION', 'PUSHING', 'DISTRIBUTION', 'SHAKEOUT']
        total_score = np.sum(raw_scores)
        if total_score < 1.0:
            return 'UNCLEAR', 0.0
        probs = raw_scores / total_score
        ent = _numba_entropy(probs)
        max_ent = 1.3862943611198906 
        entropy_factor = 1.0 - (ent / max_ent)
        best_idx = np.argmax(raw_scores)
        best_score = raw_scores[best_idx]
        best_pattern = pattern_names[best_idx]
        final_confidence = best_score * entropy_factor
        if best_pattern in ['ACCUMULATION', 'DISTRIBUTION'] and len(nets) > 5:
            r_sq = _numba_fast_r2(np.array(nets))
            trend_factor = 0.7 + 0.3 * r_sq
            final_confidence *= trend_factor
        if best_score < 30.0:
            return 'UNCLEAR', 0.0
        return best_pattern, float(np.clip(final_confidence, 0.0, 100.0))

    # ==================== 4. 资金流向质量评估 ====================
    def calculate_flow_quality(self) -> Dict[str, Any]:
        """
        v2.2: [大师级深化] 资金流向博弈质量评估
        思路：
        1. Outflow Quality: 升级为“洗盘承接弹性”。
           - 识别“大资金流出但价格跌不动”的背离现象，这是典型的主力洗盘特征。
        2. Inflow Persistence: 升级为“记忆衰减堆积”。
           - 允许“进三退一”，通过加权积分衡量一段时间内的资金堆积厚度。
        3. Large Order Anomaly: 升级为“分形脉冲检测”。
           - 结合股价相对位置，区分低位突袭和高位对倒。
        """
        quality = {}
        current_net = float(self.context.current_flow_data.get('net_mf_amount', 0) or 0)
        # 1. 流出质量 (Washout Quality)
        quality['outflow_quality'] = self._calculate_outflow_quality(current_net)
        # 2. 流入持续性 (Accumulation Persistence)
        quality['inflow_persistence'] = self._calculate_inflow_persistence()
        # 3. 大单异动 (Fractal Anomaly)
        anomaly, intensity = self._detect_large_order_anomaly()
        quality['large_order_anomaly'] = anomaly
        quality['anomaly_intensity'] = intensity
        # 其他指标保持调用 (假设已有实现或保持原样)
        quality['flow_consistency'] = self._calculate_flow_consistency()
        quality['flow_stability'] = self._calculate_flow_stability()
        return quality

    def _calculate_outflow_quality(self, current_net: float) -> float:
        """
        洗盘承接弹性计算
        版本: V2.4
        说明: 归一化基准替换为21日
        """
        if current_net >= 0:
            return 50.0
        pct_chg = 0.0
        if self.context.daily_basic_data:
            pct_chg = self.context.daily_basic_data.get('pct_change')
            if pct_chg is None:
                closes = self.close_array
                if len(closes) >= 2:
                    pct_chg = (closes[-1] - closes[-2]) / closes[-2] * 100
        if pct_chg >= 0:
            return 90.0
        abs_flow = abs(current_net)
        abs_drop = abs(pct_chg) + 0.5 
        elasticity = abs_flow / abs_drop
        recent_abs_flows = np.abs(self.net_amount_array[-21:])
        avg_flow = np.mean(recent_abs_flows) + 1.0
        score_raw = (elasticity / avg_flow) * 30.0 
        final_score = float(np.clip(score_raw + 10.0, 0.0, 100.0))
        return final_score

    def _calculate_inflow_persistence(self) -> int:
        """
        记忆衰减堆积
        版本: V2.5
        说明: 接入 Numba JIT 编译器，大幅提升时序衰减遍历速度
        """
        arr = self.net_amount_array
        if len(arr) == 0: return 0
        window = arr[-34:] if len(arr) > 34 else arr
        energy = _numba_calc_inflow_persistence_energy(window, 0.8)
        return int(np.ceil(energy))

    def _detect_large_order_anomaly(self) -> Tuple[bool, float]:
        """
        大单异动强度检测
        版本: V2.4
        说明: 对比基准调整为21日
        """
        arr = self.net_amount_array
        if len(arr) < 13: return False, 0.0
        current_net = float(self.context.current_flow_data.get('net_mf_amount', 0) or 0)
        recent = arr[-21:]
        median = np.median(recent)
        mad = np.median(np.abs(recent - median)) * 1.4826
        base_noise = max(10.0, np.mean(np.abs(recent)) * 0.1)
        sigma = max(mad, base_noise)
        z_score = (current_net - median) / sigma
        circ_mv = self.market_cap or 1e12
        turnover_impact = abs(current_net) / circ_mv
        abs_z = abs(z_score)
        intensity = 20.0 * np.log1p(np.exp(abs_z - 1.0))
        impact_score = turnover_impact * 10000.0 
        final_intensity = intensity + impact_score
        if final_intensity < 10.0:
            return False, 0.0
        return True, float(np.clip(final_intensity, 0.0, 100.0))

    def _calculate_flow_consistency(self) -> float:
        """
        [修复] 资金一致性
        修复点：
        1. 引入向量余弦相似度思想，而非简单的符号加分。
        2. 增加"散户对冲"的考量，避免满分泛滥。
        3. 使用 Sigmoid 压缩高分段。
        """
        data = self.context.current_flow_data
        if not data: return 50.0
        try:
            def get_net(level):
                buy = float(data.get(f'buy_{level}_amount', 0) or 0)
                sell = float(data.get(f'sell_{level}_amount', 0) or 0)
                return buy - sell
            # 向量分量
            elg = get_net('elg')
            lg = get_net('lg')
            md = get_net('md')
            sm = get_net('sm') # 散户
            # 主力总向量
            main_force = elg + lg
            if abs(main_force) < 10.0: return 50.0
            # 1. 内部团结度 (Internal Unity)
            # ELG 和 LG 是否同向?
            # 使用加权乘积: 如果同号，结果为正；异号，结果为负
            unity_score = 0.0
            if abs(elg) + abs(lg) > 0:
                # 归一化权重
                w_elg = 0.65
                w_lg = 0.35
                # 符号一致性检测 (-1 到 1)
                sign_consistency = np.sign(elg) * np.sign(lg)
                # 如果同向，得分为 1.0；如果异向，看谁力量大
                if sign_consistency > 0:
                    unity_score = 100.0
                else:
                    # 异向：如果 ELG 远大于 LG，一致性依然较高(听老大的)
                    # 比如 ELG=100, LG=-10 -> Net=90. 还是比较一致的。
                    # 比如 ELG=50, LG=-50 -> Net=0. 完全分歧。
                    net_ratio = abs(elg + lg) / (abs(elg) + abs(lg) + 1e-6)
                    unity_score = net_ratio * 100.0
            else:
                unity_score = 50.0
            # 2. 对手盘逻辑 (Counterparty Logic)
            # 最健康的主力买入，应该是散户卖出 (SM < 0)
            # 如果主力买，散户也买 (全场一致看多)，往往是短线高点，一致性反而不纯粹（因为没有对手盘了）
            counterparty_score = 0.0
            if main_force > 0:
                if sm < 0: counterparty_score = 10.0 # 良性
                else: counterparty_score = -10.0 # 羊群效应，扣分
            else:
                if sm > 0: counterparty_score = 10.0 # 良性
                else: counterparty_score = -10.0 # 恐慌踩踏，扣分
            # 3. 综合计算
            raw_score = unity_score + counterparty_score
            # 4. 强度压缩
            # 只有主力净占比很高时，才能突破 90 分
            total_vol = abs(elg) + abs(lg) + abs(md) + abs(sm) + 1.0
            dominance = abs(main_force) / total_vol # 0~1
            # 最终分 = 基础一致性(0~100) * (0.5 + 0.5 * 统治力)
            # 这样如果统治力弱，一致性得分会被压缩在 50-70 之间
            final_score = raw_score * (0.6 + 0.4 * dominance)
            return float(np.clip(final_score, 0.0, 100.0))
        except Exception as e:
            logger.error(f"计算资金一致性出错: {e}")
            return 50.0

    def _calculate_flow_stability(self) -> float:
        """
        资金逻辑连贯性计算
        版本: V2.4
        说明: 取样窗口更改为13日
        """
        arr = self.net_amount_array
        n = len(arr)
        if n < 13: return 50.0
        recent = arr[-13:]
        signs = np.sign(recent)
        signs = signs[signs != 0]
        if len(signs) > 1:
            flips = np.sum(signs[1:] != signs[:-1])
            flip_rate = flips / (len(signs) - 1)
        else:
            flip_rate = 0.5
        flip_score = (1.0 - flip_rate) * 100.0
        if np.std(recent) > 1e-6:
            ac1 = np.corrcoef(recent[:-1], recent[1:])[0, 1]
            if np.isnan(ac1): ac1 = 0.0
        else:
            ac1 = 0.0
        inertia_score = (ac1 + 1.0) * 50.0
        median_val = np.median(recent)
        mad = np.median(np.abs(recent - median_val))
        if abs(median_val) > 10.0:
            robust_cv = mad / abs(median_val)
        else:
            robust_cv = 1.0
        vol_score = max(0.0, 100.0 - robust_cv * 80.0)
        stability = inertia_score * 0.4 + flip_score * 0.4 + vol_score * 0.2
        return float(np.clip(stability, 0.0, 100.0))

    # ==================== 5. 多周期资金共振指标 ====================
    def calculate_multi_period_sync(self) -> Dict[str, float]:
        """
        多周期资金共振计算
        版本: V2.4
        说明: 共振参数对齐至21日和55日
        """
        sync_metrics = {}
        for key in ['daily_weekly_sync', 'daily_monthly_sync', 
                   'short_mid_sync', 'mid_long_sync']:
            sync_metrics[key] = None
        arr = self.net_amount_array
        if len(arr) < 55: return sync_metrics
        def calc_wma(data, window):
            weights = np.arange(1, window + 1)
            return np.convolve(data, weights/weights.sum(), mode='valid')
        smooth_arr = savgol_filter(arr, window_length=5, polyorder=2) if len(arr) > 5 else arr
        ma5 = calc_wma(smooth_arr, 5)
        ma21 = calc_wma(smooth_arr, 21)
        ma55 = calc_wma(smooth_arr, 55)
        min_len = min(len(ma5), len(ma21), len(ma55))
        if min_len < 5: return sync_metrics
        s_short = ma5[-5:]
        s_mid = ma21[-5:]
        s_long = ma55[-5:]
        sync_metrics['short_mid_sync'] = self._calculate_vector_resonance(s_short, s_mid, check_energy=False)
        sync_metrics['mid_long_sync'] = self._calculate_vector_resonance(s_mid, s_long, check_energy=True)
        daily_slice = smooth_arr[-5:]
        sync_metrics['daily_weekly_sync'] = self._calculate_vector_resonance(daily_slice, s_short)
        sync_metrics['daily_monthly_sync'] = self._calculate_vector_resonance(daily_slice, s_mid)
        return sync_metrics

    def _calculate_vector_resonance(self, s1: np.ndarray, s2: np.ndarray, check_energy: bool = False) -> float:
        """
        向量共振计算
        版本: V2.8
        说明: 使用 Numba 优化的皮尔逊相关系数替代 np.corrcoef 降低小数组开销
        """
        if len(s1) < 2 or len(s2) < 2: return 50.0
        std1 = np.std(s1) + 1e-6
        std2 = np.std(s2) + 1e-6
        slope1 = (s1[-1] - s1[0]) / std1
        slope2 = (s2[-1] - s2[0]) / std2
        dir_score = np.tanh(slope1 * slope2)
        corr = _numba_pearson_corr(s1, s2)
        base_score = 50.0
        if dir_score > 0:
            final_score = base_score + 25 * dir_score + 25 * max(0, corr)
        else:
            final_score = base_score - 20 * abs(dir_score)
        if check_energy and final_score > 60:
            avg_magnitude = (abs(slope1) + abs(slope2)) / 2.0
            energy_factor = 0.5 + 0.5 * np.tanh(avg_magnitude)
            final_score *= energy_factor
        return float(np.clip(final_score, 0.0, 100.0))

    def _calculate_sync_score(self, series1: List[float], series2: List[float]) -> float:
        """
        计算两个序列的同步度得分
        版本: V2.8
        说明: 使用 Numba 相关系数函数替代原有的 Numpy 复杂运算
        """
        s1 = np.array(series1, dtype=np.float64)
        s2 = np.array(series2, dtype=np.float64)
        n = len(s1)
        if n != len(s2) or n < 2:
            return 50.0
        correlation = _numba_pearson_corr(s1, s2)
        diff1 = np.diff(s1)
        diff2 = np.diff(s2)
        dir_match = np.sign(diff1) == np.sign(diff2)
        direction_score = (np.sum(dir_match) / len(diff1) * 100) if len(diff1) > 0 else 50.0
        sync_score = (correlation * 100 * 0.6 + direction_score * 0.4)
        return float(max(0.0, min(100.0, sync_score)))

    # ==================== 6. 趋势动量指标 ====================
    def calculate_trend_momentum(self) -> Dict[str, float]:
        """
        资金加速度与趋势动量计算
        版本: V2.4
        说明: 改用8日动量，参数基准上调至21日
        """
        momentum = {}
        for key in ['flow_momentum_5d', 'flow_momentum_8d', 'flow_acceleration',
                   'uptrend_strength', 'downtrend_strength']:
            momentum[key] = None
        arr = self.net_amount_array
        if len(arr) < 21: return momentum
        mad = stats.median_abs_deviation(arr[-21:]) if len(arr) >= 21 else np.std(arr)
        denom = max(mad, 10.0)
        momentum['flow_momentum_5d'] = float((arr[-1] - arr[-5]) / denom) if len(arr) >= 5 else 0.0
        momentum['flow_momentum_8d'] = float((arr[-1] - arr[-8]) / denom) if len(arr) >= 8 else 0.0
        recent_13 = arr[-13:]
        try:
            if len(recent_13) >= 5:
                acc_curve = savgol_filter(recent_13, 5, 2, deriv=2)
                raw_acc = acc_curve[-1]
                norm_acc = raw_acc / (denom + 1e-6)
                momentum['flow_acceleration'] = float(np.tanh(norm_acc) * 100.0)
            else:
                momentum['flow_acceleration'] = 0.0
        except Exception:
            momentum['flow_acceleration'] = 0.0
        up_strength, down_strength = self._calculate_robust_trend_strength()
        momentum['uptrend_strength'] = up_strength
        momentum['downtrend_strength'] = down_strength
        return momentum

    def _calculate_robust_trend_strength(self) -> Tuple[float, float]:
        """
        鲁棒趋势强度计算
        版本: V2.4
        说明: 对齐13日斐波拉契周期
        """
        window = 13
        if len(self.net_amount_array) < window: return 0.0, 0.0
        cum_flow = np.cumsum(self.net_amount_array[-window:])
        x = np.arange(len(cum_flow))
        slope, _, r_value, _, _ = linregress(x, cum_flow)
        r_sq = r_value ** 2
        linear_score = r_sq * 100.0 * np.sign(slope)
        spearman_corr, _ = stats.spearmanr(x, cum_flow)
        if np.isnan(spearman_corr): spearman_corr = 0.0
        rank_score = spearman_corr * 100.0
        net_change = cum_flow[-1] - cum_flow[0]
        std_val = np.std(cum_flow) + 1.0 
        gain_ratio = np.clip(net_change / (std_val * 2.0), -1.0, 1.0)
        gain_score = gain_ratio * 100.0
        base_score = linear_score * 0.3 + rank_score * 0.3 + gain_score * 0.4
        uptrend_val = 0.0
        downtrend_val = 0.0
        range_val = np.ptp(cum_flow) + 1e-6
        if base_score > 0:
            peak = np.maximum.accumulate(cum_flow)
            if range_val > 0:
                max_dd = np.max((peak - cum_flow) / range_val)
            else:
                max_dd = 0.0
            penalty = 1.0
            if max_dd > 0.3:
                penalty = max(0.0, 1.0 - (max_dd - 0.3) * 2.0)
            final_up = base_score * penalty
            if net_change > 0:
                final_up = max(final_up, 10.0)
            uptrend_val = final_up
        elif base_score < 0:
            trough = np.minimum.accumulate(cum_flow)
            if range_val > 0:
                max_du = np.max((cum_flow - trough) / range_val)
            else:
                max_du = 0.0
            penalty = 1.0
            if max_du > 0.3:
                penalty = max(0.0, 1.0 - (max_du - 0.3) * 2.0)
            final_down = abs(base_score) * penalty
            if net_change < 0:
                final_down = max(final_down, 10.0)
            downtrend_val = final_down
        return float(np.clip(uptrend_val, 0.0, 100.0)), float(np.clip(downtrend_val, 0.0, 100.0))

    def _calculate_complex_trend_strength(self) -> Tuple[float, float]:
        """
        计算复合趋势强度
        版本: V2.8
        说明: 彻底移除 scipy.stats.linregress 和 np.corrcoef，引入 Numba 引擎进行极速回归和相关性分析
        """
        if len(self.net_amount_array) < 10: return 0.0, 0.0
        y = self.net_amount_array[-10:]
        vol = self.volume_array[-10:]
        slope, r_sq = _numba_linregress(y)
        y_range = np.max(y) - np.min(y) + 1e-6
        norm_slope = slope * 10 / y_range
        strength = abs(norm_slope) * r_sq * 100.0
        if slope > 0:
            vol_corr = _numba_pearson_corr(vol, y)
            vol_factor = 1.0 + 0.5 * vol_corr
            final_up = min(100.0, strength * vol_factor)
            return float(final_up), 0.0
        else:
            v_slope, _ = _numba_linregress(vol)
            avg_vol = np.mean(vol) + 1e-6
            norm_v_slope = v_slope * 10 / avg_vol
            down_score = strength
            if norm_v_slope > 0.2: 
                down_score *= 1.3 
            elif r_sq > 0.8 and abs(norm_v_slope) < 0.2:
                down_score *= 1.2
            return 0.0, float(min(100.0, down_score))

    def _calculate_trend_strength(self) -> Tuple[float, float]:
        """
        计算趋势强度
        版本: V2.8
        说明: 使用 Numba 线性回归替代原有的 NumPy 手动代数运算，提升小数组执行效率
        """
        if len(self.net_amount_series) < 10:
            return 0.0, 0.0
        y = np.array(self.net_amount_series[-10:], dtype=np.float64)
        slope, r_sq = _numba_linregress(y)
        strength = min(100.0, r_sq * 100.0)
        if slope > 0:
            return strength, 0.0
        else:
            return 0.0, strength

    # ==================== 7. 量价背离指标 ====================
    def calculate_divergence_metrics(self) -> Dict[str, Any]:
        """
        背离强度计算
        版本: V2.4
        说明: 分析窗口缩放至13日
        """
        divergence = {
            'divergence_type': 'NONE',
            'divergence_strength': 0.0,
            'price_flow_divergence': 0.0
        }
        window = 13
        if len(self.close_array) < window: return divergence
        prices = self.close_array[-window:]
        cum_flow = np.cumsum(self.net_amount_array[-window:])
        def norm(arr):
            ptp = np.ptp(arr)
            if ptp < 1e-8: return np.zeros_like(arr)
            return (arr - np.min(arr)) / ptp
        p_norm = norm(prices)
        f_norm = norm(cum_flow)
        p_corr = np.corrcoef(p_norm, f_norm)[0, 1]
        if np.isnan(p_corr): p_corr = 1.0
        s_corr, _ = stats.spearmanr(p_norm, f_norm)
        if np.isnan(s_corr): s_corr = 1.0
        min_corr = min(p_corr, s_corr)
        div_score = (1.0 - min_corr) * 50.0
        div_score = max(0.0, div_score)
        divergence['price_flow_divergence'] = float(np.clip(div_score, 0.0, 100.0))
        if div_score > 10.0:
            x = np.arange(window)
            p_slope, _, _, _, _ = linregress(x, prices)
            f_slope, _, _, _, _ = linregress(x, cum_flow)
            div_type = 'NONE'
            strength_mult = 1.0
            if p_slope > 0 and f_slope < 0:
                div_type = 'BEARISH'
                strength_mult = 1.2
            elif p_slope < 0 and f_slope > 0:
                div_type = 'BULLISH'
                strength_mult = 1.2
            elif p_slope > 0 and f_slope > 0:
                if np.mean(p_norm) > np.mean(f_norm) + 0.05:
                    div_type = 'BEARISH'
                    strength_mult = 0.7 
            elif p_slope < 0 and f_slope < 0:
                if np.mean(f_norm) > np.mean(p_norm) + 0.05:
                    div_type = 'BULLISH'
                    strength_mult = 0.7
            if div_type != 'NONE':
                divergence['divergence_type'] = div_type
                divergence['divergence_strength'] = float(np.clip(div_score * strength_mult, 0.0, 100.0))
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
        v2.1: [大师级深化] 换手加权锚点与博弈边界
        思路：
        1. Flow Peak Value: 升级为“有效锚点”。
           - 引入 Volume Weighting。无量空涨的峰值是虚的，只有经过充分换手(Turnover)的峰值
             才具备真实的“套牢盘”或“支撑”物理意义。
           - 算法: Peak_Value * log(1 + Turnover_At_Peak / Avg_Turnover)
        2. Support/Resistance: 保持 v2.0 的 MAD 自适应边界逻辑。
        """
        structure = {}
        arr = self.net_amount_array
        vol_arr = self.volume_array # 需要用到成交量
        if len(arr) < 20:
            structure['flow_peak_value'] = None
            structure['days_since_last_peak'] = None
            structure['flow_support_level'] = None
            structure['flow_resistance_level'] = None
            return structure
        # --- 1. 寻找显著峰值 (Peak Finding) ---
        median_val = np.median(arr)
        mad = np.median(np.abs(arr - median_val)) * 1.4826
        background_noise = max(mad, 10.0)
        # 寻找正向峰值
        peaks, properties = find_peaks(arr, height=background_noise, prominence=background_noise)
        if len(peaks) > 0:
            # 找到最近的一个显著峰
            last_peak_idx = peaks[-1]
            raw_peak_val = float(arr[last_peak_idx])
            # --- 深化：换手率加权 (Volume Validation) ---
            # 获取峰值当天的成交量
            peak_vol = vol_arr[last_peak_idx]
            # 获取近期的平均成交量
            avg_vol = np.mean(vol_arr[-20:]) + 1.0
            # 计算成交量倍比 (Volume Ratio)
            vol_ratio = peak_vol / avg_vol
            # 计算有效峰值 (Effective Peak Value)
            # 逻辑：如果峰值处成交量巨大，说明分歧巨大且互道傻逼，该点位的阻力/支撑极强。
            # 如果缩量创新高，说明一致性强，但作为物理阻力的筹码厚度不够。
            # 这里我们定义 metric 为“峰值的物理阻力强度”
            # 使用对数平滑，避免量能过大导致数值失真
            weighted_peak = raw_peak_val * (1.0 + np.log1p(vol_ratio))
            structure['flow_peak_value'] = float(weighted_peak)
            structure['days_since_last_peak'] = int(len(arr) - 1 - last_peak_idx)
        else:
            # 无显著峰值，退化处理
            max_idx = np.argmax(arr)
            structure['flow_peak_value'] = float(arr[max_idx]) if arr[max_idx] > 0 else 0.0
            structure['days_since_last_peak'] = int(len(arr) - 1 - max_idx)
        # --- 2. 资金弹性边界 (Robust Boundaries) ---
        # 保持 v2.0 的优秀逻辑
        window = min(len(arr), 60)
        recent_data = arr[-window:]
        curr_median = np.median(recent_data)
        curr_mad = np.median(np.abs(recent_data - curr_median)) * 1.4826
        # 2倍 MAD (约等于2倍标准差，涵盖95%概率)
        resistance = curr_median + 2.0 * curr_mad
        support = curr_median - 2.0 * curr_mad
        structure['flow_resistance_level'] = float(resistance)
        structure['flow_support_level'] = float(support)
        return structure

    # ==================== 9. 统计特征指标 ====================
    def calculate_statistical_metrics(self) -> Dict[str, float]:
        """
        基于鲁棒统计学的资金分布特征
        版本: V2.5
        说明: 将 percentileofscore 替换为极速 NumPy 向量化实现
        """
        stats_metrics = {}
        arr = self.net_amount_array
        if len(arr) < 13:
            for key in ['flow_zscore', 'flow_percentile', 'flow_volatility_13d', 'flow_volatility_21d']:
                stats_metrics[key] = None
            return stats_metrics
        current_net = float(self.context.current_flow_data.get('net_mf_amount', 0) or 0)
        recent_21 = arr[-21:]
        median_21 = np.median(recent_21)
        mad_21 = np.median(np.abs(recent_21 - median_21)) * 1.4826
        if mad_21 < 1.0: 
            std_backup = np.std(recent_21)
            denom = std_backup if std_backup > 1.0 else 10.0
        else:
            denom = mad_21
        stats_metrics['flow_zscore'] = (current_net - median_21) / denom
        n_21 = len(recent_21)
        rank_21 = ((np.sum(recent_21 < current_net) + np.sum(recent_21 == current_net) * 0.5) / n_21) * 100.0
        window_long = 55
        recent_long = arr[-window_long:] if len(arr) >= window_long else arr
        n_long = len(recent_long)
        rank_long = ((np.sum(recent_long < current_net) + np.sum(recent_long == current_net) * 0.5) / n_long) * 100.0
        stats_metrics['flow_percentile'] = float(rank_21 * 0.6 + rank_long * 0.4)
        mean_21 = np.mean(recent_21)
        abs_mean = abs(mean_21)
        if abs_mean > 10.0:
            stats_metrics['flow_volatility_21d'] = np.std(recent_21) / abs_mean
        else:
            stats_metrics['flow_volatility_21d'] = 0.0
        if len(arr) >= 13:
            recent_13 = arr[-13:]
            abs_mean_13 = abs(np.mean(recent_13))
            if abs_mean_13 > 10.0:
                stats_metrics['flow_volatility_13d'] = np.std(recent_13) / abs_mean_13
            else:
                stats_metrics['flow_volatility_13d'] = 0.0
        else:
            stats_metrics['flow_volatility_13d'] = None
        return stats_metrics

    # ==================== 10. 预测指标 ====================
    def calculate_prediction_metrics(self) -> Dict[str, float]:
        """
        预测置信度计算
        版本: V2.7
        说明: 使用 Numba 优化的自相关计算替换 np.corrcoef 避免矩阵分配
        """
        prediction = {}
        arr = self.net_amount_array
        if len(arr) < 10:
            return {k: None for k in ['expected_flow_next_1d', 'flow_forecast_confidence', 
                                     'uptrend_continuation_prob', 'reversal_prob']}
        recent_5 = arr[-5:]
        weights = np.arange(1, 6)
        prediction['expected_flow_next_1d'] = float(np.average(recent_5, weights=weights))
        recent_10 = arr[-10:]
        displacement = abs(np.sum(recent_10))
        path_length = np.sum(np.abs(recent_10))
        ker = displacement / (path_length + 1e-6)
        ac1 = 0.0
        if np.std(recent_10) > 1e-6:
            ac1 = _numba_autocorr1(recent_10)
            if np.isnan(ac1): ac1 = 0.0
        raw_confidence = ker * 80.0 + max(0, ac1) * 20.0
        vol = np.std(recent_10)
        mean_abs = np.mean(np.abs(recent_10)) + 1e-6
        cv = vol / mean_abs
        penalty = 1.0
        if cv > 1.5:
            penalty = max(0.5, 1.0 - (cv - 1.5) * 0.5)
        final_confidence = raw_confidence * penalty
        prediction['flow_forecast_confidence'] = float(np.clip(final_confidence, 0.0, 100.0))
        trend_score = 50.0
        if prediction['expected_flow_next_1d'] > 0:
            trend_score += final_confidence * 0.4
        else:
            trend_score -= final_confidence * 0.4
        prediction['uptrend_continuation_prob'] = float(np.clip(trend_score, 0.0, 100.0))
        prediction['reversal_prob'] = float(np.clip(100.0 - trend_score, 0.0, 100.0))
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
        """
        v2.0: [大师级深化] 基于多维共振的信号生成机制
        修复：解决 downtrend_strength 为 None 导致的 TypeError
        思路：
        1. 摒弃线性加分，采用“共振放大器”模型。
        2. 核心逻辑：Signal = Base * (1 + Resonance)。
        3. 必须通过“一票否决”机制过滤假信号。
        """
        # 1. 提取关键因子 (增加 None 安全处理)
        flow_score = metrics.get('flow_intensity') or 0  # 资金面
        pattern = metrics.get('behavior_pattern', 'UNCLEAR') # 行为面
        pat_conf = metrics.get('pattern_confidence') or 0
        div_type = metrics.get('divergence_type', 'NONE') # 技术面背离
        div_strength = metrics.get('divergence_strength') or 0
        tick_net = metrics.get('tick_large_order_net') # 微观面
        if tick_net is None: tick_net = 0
        # 2. 确定基础信号方向
        signal_dir = 0 # 1=Buy, -1=Sell, 0=Neutral
        # 优先级1: 极强的主力行为模式 (Accumulation/Distribution)
        if pattern == 'ACCUMULATION' and pat_conf > 60:
            signal_dir = 1
        elif pattern == 'DISTRIBUTION' and pat_conf > 60:
            signal_dir = -1
        # 优先级2: 强烈的量价背离
        elif div_type == 'BULLISH' and div_strength > 60:
            signal_dir = 1
        elif div_type == 'BEARISH' and div_strength > 60:
            signal_dir = -1
        # 优先级3: 纯资金流强度
        elif flow_score > 60:
            signal_dir = 1
        elif flow_score < -60:
            signal_dir = -1
        # 3. 计算信号强度 (共振模型)
        # 初始强度基于综合评分
        strength = comp_score
        # [共振检查]
        # 如果是买入信号，检查各维度是否支持
        if signal_dir == 1:
            resonance_count = 0
            # A. 资金面共振: 当日资金大举流入
            if flow_score > 40: resonance_count += 1
            # B. 微观面共振: Tick级大单也是净买入 (避免假单)
            if tick_net > 0: resonance_count += 1
            # C. 技术面共振: 处于上升趋势或底部反转
            uptrend_prob = metrics.get('uptrend_continuation_prob') or 0
            reversal_prob = metrics.get('reversal_prob') or 0
            if uptrend_prob > 60 or reversal_prob > 70: resonance_count += 1
            # D. 结构面共振: 筹码锁定良好 (低波)
            stability = metrics.get('flow_stability') or 0
            if stability > 70: resonance_count += 1
            # 放大机制: 每一个共振因子增加 10-15% 的强度
            strength = strength * (1.0 + resonance_count * 0.15)
            # [一票否决] 风险控制
            # 如果主力资金其实在流出 (flow < -20)，即使技术面好看，也必须降级
            if flow_score < -20:
                strength *= 0.5 # 强度腰斩
                signal_dir = 0 # 转为中性甚至观望
            final_signal = 'STRONG_BUY' if strength > 85 else ('BUY' if strength > 65 else 'NEUTRAL')
        elif signal_dir == -1:
            resonance_count = 0
            if flow_score < -40: resonance_count += 1
            if tick_net < 0: resonance_count += 1
            # [关键修复] 增加 or 0 防止 NoneType 比较报错
            down_strength = metrics.get('downtrend_strength') or 0
            if down_strength > 60: resonance_count += 1
            strength = strength * (1.0 + resonance_count * 0.15)
            # [一票否决]
            # 如果是卖出信号，但资金在疯狂吸筹 (flow > 20)，可能是洗盘
            if flow_score > 20:
                strength *= 0.6
                final_signal = 'HOLD' # 建议持仓观察
            else:
                final_signal = 'STRONG_SELL' if strength > 85 else ('SELL' if strength > 65 else 'NEUTRAL')
        else:
            final_signal = 'NEUTRAL'
            # 震荡市中，强度衰减
            strength *= 0.8
        # 最终归一化
        final_strength = float(max(0.0, min(100.0, strength)))
        return final_signal, final_strength

    # ==================== 12. 基于Tick数据的增强指标计算 ====================
    def calculate_tick_enhanced_metrics(self) -> Dict[str, Any]:
        """
        基于Tick数据计算增强的资金流向指标
        版本: V1.7
        说明:
        1. [关键修复] 强制将时间转换为北京时间(Asia/Shanghai)并去除时区(Naive)，
           确保 .values 底层数值对应北京时间，解决下午/尾盘数据识别为0的问题。
        2. [关键修复] 显式转换数值列为 float64，避免 Decimal/String 导致的计算错误。
        3. [新增] 集成基于信号处理的资金持续性算法。
        4. [Bug修复] 重置索引以消除 'trade_time' 列名与索引名的歧义，解决 sort_values 报错。
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
            # 确保 trade_time 存在
            if 'trade_time' not in df.columns:
                if isinstance(df.index, pd.DatetimeIndex):
                    df['trade_time'] = df.index
                else:
                    return tick_metrics
            # 1. 关键：时间标准化处理 (解决时区导致的时间判断错误)
            time_series = pd.to_datetime(df['trade_time'], errors='coerce')
            if time_series.dt.tz is not None:
                time_series = time_series.dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
            df['trade_time'] = time_series
            df = df.dropna(subset=['trade_time'])
            # [关键修复] 重置索引，防止索引名与列名 'trade_time' 冲突导致后续 sort_values 报错
            # 这步操作会丢弃原有索引，确保 DataFrame 只有唯一的 RangeIndex
            df.reset_index(drop=True, inplace=True)
            if df.empty:
                return tick_metrics
            # 2. 类型标准化
            if 'type' in df.columns:
                df['type'] = df['type'].astype(str).str.upper().str.strip()
                type_map = {'买盘': 'B', '卖盘': 'S', 'BUY': 'B', 'SELL': 'S', '0': 'S', '1': 'B', '2': 'M'}
                df['type'] = df['type'].map(lambda x: type_map.get(x, x))
            else:
                return tick_metrics
            df = df[df['type'].isin(['B', 'S'])]
            if df.empty:
                return tick_metrics
            # 3. 数值类型强制转换
            for col in ['price', 'volume', 'amount']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(np.float64)
            if 'amount' not in df.columns:
                df['amount'] = df['price'] * df['volume'] * 100
            # --- 开始各项指标计算 ---
            tick_metrics.update(self._calculate_intraday_flow_distribution(df))
            tick_metrics.update(self._detect_high_freq_large_orders(df))
            tick_metrics.update(self._calculate_flow_impact_features(df))
            tick_metrics.update(self._calculate_intraday_momentum(df))
            tick_metrics.update(self._calculate_flow_cluster_features(df))
            # 之前报错的方法：_calculate_high_freq_divergence
            tick_metrics.update(self._calculate_high_freq_divergence(df))
            tick_metrics.update(self._calculate_vwap_deviation(df))
            tick_metrics.update(self._calculate_flow_efficiency(df))
            tick_metrics.update(self._calculate_closing_flow_features(df))
            tick_metrics.update(self._calculate_high_freq_statistics(df))
            tick_metrics.update(self._calculate_time_period_distribution(df))
            tick_metrics.update(self._calculate_stealth_flow_indicators(df))
            # [新增] 深度资金持续性计算
            tick_metrics['flow_persistence_minutes'] = self._calculate_flow_persistence_minutes(df)
        except Exception as e:
            logger.error(f"计算tick增强指标异常: {e}", exc_info=True)
        return tick_metrics

    def _calculate_flow_persistence_minutes(self, df: pd.DataFrame) -> float:
        """
        基于Savitzky-Golay滤波的资金攻击波久期计算
        版本: V2.6
        说明: 使用 Numba 机器码引擎替换了原生的 Python 游程检测循环
        """
        times = df['trade_time'].values.astype('datetime64[m]').astype('int64')
        if len(times) == 0: return 0.0
        amounts = df['amount'].values
        types = df['type'].values
        signed_amounts = np.where(types == 'B', amounts, -amounts)
        norm_times = times - times[0]
        max_minutes = int(np.max(norm_times)) + 1
        minute_flows = np.bincount(norm_times, weights=signed_amounts, minlength=max_minutes) / 10000.0
        minute_flows = np.trim_zeros(minute_flows, 'b')
        n = len(minute_flows)
        if n < 5: return 0.0
        window_length = min(11, n if n % 2 != 0 else n - 1)
        if window_length < 3:
            smoothed_flows = minute_flows
        else:
            try:
                smoothed_flows = savgol_filter(minute_flows, window_length, 2)
            except Exception:
                smoothed_flows = minute_flows
        noise_threshold = 1.0 
        states = np.zeros_like(smoothed_flows, dtype=int)
        states[smoothed_flows > noise_threshold] = 1
        states[smoothed_flows < -noise_threshold] = -1
        if len(states) == 0: return 0.0
        max_persistence, dominant_direction = _numba_find_longest_streak(states)
        return max_persistence * dominant_direction

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
        v1.8: [大师级深化] 攻防不对称性 (Aggression Asymmetry)
        思路：
        1. 摒弃平均冲击，改为计算"多空敏感度比值"。
        2. 敏感度 = |价格变动| / 成交金额。
        3. Ratio = (卖方敏感度) / (买方敏感度)。
        4. 物理意义:
           - Ratio < 1.0: 卖方砸盘砸不动(承接强)，买方稍微一买就涨(抛压轻) -> 多头绝对优势。
           - Ratio > 1.0: 买方拉升拉不动(抛压重)，卖方稍微一砸就崩(无承接) -> 空头绝对优势。
        5. 注意: 这个指标越小越好（对于多头而言）。
        """
        metrics = {'flow_impact_ratio': None}
        if len(df) < 20: return metrics
        # 1. 计算每笔 Tick 的价格变动
        # 使用后向差分: 当前价格 - 上一笔价格
        prices = df['price'].values
        amounts = df['amount'].values
        types = df['type'].values
        price_diffs = np.zeros_like(prices)
        price_diffs[1:] = np.diff(prices)
        # 2. 分离 主动买入 和 主动卖出 的Tick
        # 过滤掉价格无变动的Tick (这些通常是盘口内的吃单，无法计算冲击)
        valid_change_mask = np.abs(price_diffs) > 1e-6
        buy_mask = (types == 'B') & valid_change_mask
        sell_mask = (types == 'S') & valid_change_mask
        # 3. 计算单位金额的冲击力 (Impact per Million)
        # 为了避免微小金额造成的冲击数值爆炸，我们采用整体法：总变动 / 总金额
        # 这比 mean(单笔冲击) 更鲁棒
        # 买方冲击力 (Buying Impact): 买了多少钱，推升了多少价位
        # 注意: 这里取绝对值，我们只关心"变动幅度"
        buy_amt_sum = np.sum(amounts[buy_mask])
        buy_price_sum = np.sum(np.abs(price_diffs[buy_mask]))
        # 卖方冲击力 (Selling Impact): 卖了多少钱，砸下了多少价位
        sell_amt_sum = np.sum(amounts[sell_mask])
        sell_price_sum = np.sum(np.abs(price_diffs[sell_mask]))
        # 4. 计算敏感度 (Price Change per 1M Turnover)
        buy_sensitivity = (buy_price_sum / buy_amt_sum) if buy_amt_sum > 0 else 0
        sell_sensitivity = (sell_price_sum / sell_amt_sum) if sell_amt_sum > 0 else 0
        # 5. 计算比率
        # 为了防止除0，且让指标中心化在1.0
        if buy_sensitivity > 1e-9:
            ratio = sell_sensitivity / buy_sensitivity
            # 限制范围 [0, 10]
            metrics['flow_impact_ratio'] = float(min(10.0, ratio))
        else:
            # 买方敏感度为0 (怎么买都不涨)，如果不跌则中性，如果跌则极差
            metrics['flow_impact_ratio'] = 10.0 if sell_sensitivity > 0 else 1.0
        return metrics

    def _calculate_intraday_momentum(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        v1.8: [大师级深化] 波动率归一化动量模型
        修复点：
        1. 逻辑重构: 摒弃"环比增长率"逻辑，改用"波动率归一化差值" (Z-Score 思想)。
           原逻辑 (Current - Prev) / Prev 极易因 Prev 接近0而爆炸。
           新逻辑 (Current - Prev) / Volatility 衡量的是"趋势改变的显著性"。
        2. 量纲统一: 映射到 [-100, 100] 区间，Z=3 (3倍标准差变化) 对应 60-70 分。
        """
        metrics = {'intraday_flow_momentum': None, 'flow_acceleration_intraday': None}
        # 1. 基础数据聚合 (分钟级)
        times = df['trade_time'].values.astype('datetime64[m]').astype('int64')
        if len(times) == 0: return metrics
        amounts = df['amount'].values
        types = df['type'].values
        signed_amounts = np.where(types == 'B', amounts, -amounts)
        norm_times = times - times[0]
        max_minutes = int(np.max(norm_times)) + 1
        minute_flows = np.bincount(norm_times, weights=signed_amounts, minlength=max_minutes) / 10000.0
        # 剔除尾部无效0值
        active_flows = np.trim_zeros(minute_flows, 'b')
        n = len(active_flows)
        if n < 10: return metrics
        # --- 指标1: 趋势动量 (Momentum) - 修正版 ---
        # 窗口: 最近 15分钟 vs 之前 30分钟
        if n >= 45:
            # 提取整个窗口的数据计算波动率
            full_window = active_flows[-45:]
            volatility = np.std(full_window)
            # 计算底噪: 防止死水股(波动率为0)导致的除零错误
            # 设定为: 至少 1万元 或 平均绝对流量的 10%
            base_noise = max(1.0, np.mean(np.abs(full_window)) * 0.1)
            # 分母 = 波动率 + 底噪
            denom = volatility + base_noise
            recent_trend = np.mean(active_flows[-15:])
            prev_trend = np.mean(active_flows[-45:-15])
            diff = recent_trend - prev_trend
            # 计算 Z-Score 形式的动量
            # 物理意义: 趋势的改变幅度是背景波动的多少倍?
            z_momentum = diff / denom
            # 映射: Z=1 (1倍标准差改变) -> 20分
            # Z=5 (5倍标准差巨变) -> 100分
            momentum_score = z_momentum * 20.0
            metrics['intraday_flow_momentum'] = float(np.clip(momentum_score, -100.0, 100.0))
        else:
            metrics['intraday_flow_momentum'] = 0.0
        # --- 指标2: 资金加速度 (Acceleration) - 保持 v1.7 逻辑 ---
        # 逻辑: Z-Score 爆发力模型
        window_short = 3
        window_long = 30
        if n >= window_long:
            pulse = np.mean(active_flows[-window_short:])
            baseline_slice = active_flows[-window_long:]
            baseline_mean = np.mean(baseline_slice)
            baseline_std = np.std(baseline_slice)
            if baseline_std > 0.1:
                acc_z_score = (pulse - baseline_mean) / baseline_std
            else:
                acc_z_score = (pulse - baseline_mean) / 10.0 
            metrics['flow_acceleration_intraday'] = float(np.clip(acc_z_score * 10.0, -100.0, 100.0))
        else:
            metrics['flow_acceleration_intraday'] = 0.0
        return metrics

    def _calculate_closing_flow_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        尾盘强度计算
        版本: V2.9
        说明: 引入 Numba 单次遍历聚合，消除原有的高开销布尔掩码与中间数组分配
        """
        metrics = {'closing_flow_ratio': None, 'closing_flow_intensity': None}
        times = df['trade_time'].values.astype('int64') 
        amounts = df['amount'].values
        types = df['type'].values
        if len(times) == 0: return metrics
        is_buy = (types == 'B')
        c_net_total, a_net_auction, t_net_raw, total_turnover = _numba_closing_flow_agg(times, amounts, is_buy)
        if total_turnover <= 0:
            metrics['closing_flow_ratio'] = 0.0
            metrics['closing_flow_intensity'] = 0.0
            return metrics
        min_significant_flow = total_turnover * 0.005
        denominator = max(abs(t_net_raw), min_significant_flow)
        if denominator > 0:
            raw_ratio = abs(c_net_total) / denominator * 100.0
            metrics['closing_flow_ratio'] = float(min(1000.0, raw_ratio))
        else:
            metrics['closing_flow_ratio'] = 0.0
        urgency_alpha = 3.0
        weighted_flow = abs(c_net_total) + (abs(a_net_auction) * urgency_alpha)
        intensity_score = min(100.0, (weighted_flow / total_turnover) * 800.0)
        metrics['closing_flow_intensity'] = float(intensity_score)
        return metrics

    def _calculate_flow_cluster_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        同向拆单攻击波识别 - 对数强度修正版
        版本: V2.5
        说明: 使用 Numba 状态机完全重构内部循环，彻底消除数万级别Tick扫描的性能瓶颈
        """
        metrics = {'flow_cluster_intensity': None, 'flow_cluster_duration': None}
        time_int = df['trade_time'].values.astype('int64') // 10**9 // 3
        if len(time_int) == 0: return metrics
        amounts = df['amount'].values
        types = df['type'].values
        signed_amounts = np.where(types == 'B', amounts, -amounts)
        uni_times, inverse_indices = np.unique(time_int, return_inverse=True)
        cluster_flows = np.zeros(len(uni_times))
        np.add.at(cluster_flows, inverse_indices, signed_amounts)
        if len(cluster_flows) < 20: 
            metrics['flow_cluster_intensity'] = 0.0
            metrics['flow_cluster_duration'] = 0
            return metrics
        abs_flows = np.abs(cluster_flows)
        median_flow = np.median(abs_flows)
        mad_flow = np.median(np.abs(abs_flows - median_flow)) * 1.4826
        threshold = max(50000.0, median_flow + 1.0 * mad_flow)
        states = np.zeros(len(cluster_flows), dtype=int)
        states[cluster_flows > threshold] = 1
        states[cluster_flows < -threshold] = -1
        if not np.any(states != 0):
            metrics['flow_cluster_intensity'] = 0.0
            metrics['flow_cluster_duration'] = 0
            return metrics
        max_dur_buckets, total_flow, total_buckets = _numba_flow_cluster_state_machine(states, cluster_flows, 2)
        metrics['flow_cluster_duration'] = int(max_dur_buckets * 3)
        baseline_flow = max(10000.0, median_flow)
        if total_buckets > 0:
            avg_flow_in_cluster = total_flow / total_buckets
            ratio = avg_flow_in_cluster / baseline_flow
            if ratio > 1.0:
                intensity = np.log10(ratio) * 50.0
            else:
                intensity = 0.0
            metrics['flow_cluster_intensity'] = float(min(100.0, intensity))
        else:
            metrics['flow_cluster_intensity'] = 0.0
        return metrics

    def _calculate_high_freq_divergence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        量价微观解离度
        版本: V2.6
        说明: 彻底移除低效的 Pandas resample 引擎，采用纯 NumPy 边界寻址算法实现亚毫秒级高频重采样聚合
        """
        metrics = {'high_freq_flow_divergence': None}
        if len(df) < 100: return metrics 
        df = df.sort_values('trade_time')
        times = df['trade_time'].values.astype('datetime64[m]').astype('int64')
        prices = df['price'].values
        amounts = df['amount'].values
        types = df['type'].values
        signed_amounts = np.where(types == 'B', amounts, -amounts)
        start_time = times[0]
        norm_times = times - start_time
        transitions = np.where(np.diff(norm_times) != 0)[0]
        last_indices = np.append(transitions, len(norm_times) - 1)
        active_minutes = norm_times[last_indices]
        p_last = prices[last_indices]
        max_minutes = int(np.max(norm_times)) + 1
        minute_net_flow = np.bincount(norm_times, weights=signed_amounts, minlength=max_minutes)
        f_flow_active = minute_net_flow[active_minutes]
        if len(p_last) < 10:
            metrics['high_freq_flow_divergence'] = 0.0
            return metrics
        p_delta = np.diff(p_last)
        f_valid_flow = f_flow_active[1:]
        valid_mask = np.abs(p_delta) > 1e-6
        if np.sum(valid_mask) < 5:
            metrics['high_freq_flow_divergence'] = 0.0
            return metrics
        p_valid = p_delta[valid_mask]
        f_valid = f_valid_flow[valid_mask]
        corr = np.corrcoef(p_valid, f_valid)[0, 1]
        if np.isnan(corr):
            metrics['high_freq_flow_divergence'] = 1.0 
        else:
            metrics['high_freq_flow_divergence'] = float((1.0 - corr) * 50.0)
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
        v1.8: [大师级深化] 锁筹效率计算 (Chip Lock-in Efficiency)
        思路：
        1. 摒弃线性的"价/量"比值，改用金融物理学中的"对数边际效应"。
        2. 核心: 衡量"画出这根K线实体"的代价。
        3. 公式: (收盘价 - 开盘价) / log10(总成交额)。
        4. 物理意义:
           - 高正值: 缩量大涨 (高度控盘/一致性预期) -> 强庄。
           - 低值/0: 放量滞涨 (分歧巨大/出货) -> 陷阱。
           - 高负值: 缩量阴跌 (无承接) -> 阴跌绵绵。
        """
        metrics = {'flow_efficiency': None}
        # 基础数据校验
        if len(df) < 10: 
            metrics['flow_efficiency'] = 0.0
            return metrics
        prices = df['price'].values
        amounts = df['amount'].values
        # 1. 提取 K线实体高度 (Day Range)
        # 使用 Tick 数据的首尾价格，比日线数据更精确（因为这里计算的是Tick数据对应的时段）
        price_open = prices[0]
        price_close = prices[-1]
        price_change_pct = (price_close - price_open) / price_open * 100.0
        # 2. 计算总成交额 (Turnover)
        total_turnover = np.sum(amounts)
        # 3. 计算对数锁筹效率
        # 为什么要用 log? 因为1亿资金和10亿资金对价格的推动能力不是线性10倍关系，
        # 随着金额增加，摩擦成本激增。Log能还原资金的"真实做功"。
        if total_turnover > 10000: # 至少有1万元成交
            # 使用 log10(万元) 作为分母
            log_turnover = np.log10(total_turnover / 10000.0 + 1.0)
            # Efficiency = 涨幅 / 对数资金
            # 这里的数值量级大约在 [-10, 10] 之间
            efficiency = price_change_pct / (log_turnover + 1e-6)
            # 放大系数以便观察，并截断异常值
            metrics['flow_efficiency'] = float(np.clip(efficiency * 10.0, -100.0, 100.0))
        else:
            metrics['flow_efficiency'] = 0.0
        return metrics

    def _calculate_high_freq_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        微观资金流场的统计特征
        版本: V2.7
        说明: 使用 Numba 联合矩估计替换 scipy.stats 调用，大幅降低包装开销
        """
        metrics = {'high_freq_flow_skewness': None, 'high_freq_flow_kurtosis': None}
        times_3s = df['trade_time'].values.astype('datetime64[s]').astype('int64') // 3
        if len(times_3s) == 0: return metrics
        amounts = df['amount'].values
        types = df['type'].values
        signed_amounts = np.where(types == 'B', amounts, -amounts)
        norm_times = times_3s - times_3s[0]
        max_idx = int(norm_times[-1]) + 1
        flux_3s = np.bincount(norm_times, weights=signed_amounts, minlength=max_idx) / 10000.0
        flux_active = np.trim_zeros(flux_3s)
        if len(flux_active) < 30: return metrics
        try:
            sk, kt = _numba_skew_kurtosis(flux_active)
            if np.isnan(sk): sk = 0.0
            damped_skew = np.sign(sk) * np.log1p(np.abs(sk))
            metrics['high_freq_flow_skewness'] = float(np.clip(damped_skew * 25.0, -100.0, 100.0))
            if np.isnan(kt): kt = 0.0
            if kt > 0:
                log_kt_score = np.log1p(kt) * 12.0
            else:
                log_kt_score = 0.0
            metrics['high_freq_flow_kurtosis'] = float(np.clip(log_kt_score, 0.0, 100.0))
        except Exception as e:
            logger.error(f"高频统计特征计算错误: {e}")
        return metrics

    def _calculate_time_period_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        计算资金时段分布
        版本: V2.9
        说明: 使用 Numba 取代 Numpy 切片掩码聚合，消除中间内存碎片
        """
        metrics = {'morning_flow_ratio': None, 'afternoon_flow_ratio': None}
        times = df['trade_time'].values
        if len(times) == 0: return metrics
        dates = times.astype('datetime64[D]')
        hours = (times - dates).astype('timedelta64[h]').astype(int)
        amounts = df['amount'].values
        types = df['type'].values
        is_buy = (types == 'B')
        morning_net, afternoon_net = _numba_morning_afternoon_agg(hours, amounts, is_buy)
        morning_net /= 10000.0
        afternoon_net /= 10000.0
        total_abs_net = abs(morning_net) + abs(afternoon_net)
        if total_abs_net > 0:
            morning_ratio = abs(morning_net) / total_abs_net * 100.0
            afternoon_ratio = abs(afternoon_net) / total_abs_net * 100.0
        else:
            morning_ratio = 50.0
            afternoon_ratio = 50.0
        metrics['morning_flow_ratio'] = float(morning_ratio)
        metrics['afternoon_flow_ratio'] = float(afternoon_ratio)
        return metrics

    def _calculate_stealth_flow_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        计算主力隐蔽性指标
        版本: V2.9.1
        说明: [关键修复] 增加数值溢出保护(clip)，防止除以微小分母导致的数值爆炸
        """
        # print("[探针] 执行 _calculate_stealth_flow_indicators: Numba 单次扫描提速并增加溢出保护")
        metrics = {'stealth_flow_ratio': None}
        if len(df) == 0: return metrics
        small_order_threshold = 50000.0
        amounts = df['amount'].values
        types = df['type'].values
        is_buy = (types == 'B')
        small_net, total_net = _numba_stealth_flow_agg(amounts, is_buy, small_order_threshold)
        small_net /= 10000.0
        total_net /= 10000.0
        if abs(total_net) > 0.0001: # 提高分母阈值，避免极端微小值
            stealth_ratio = abs(small_net) / abs(total_net) * 100.0
            # [修复] 限制最大值为 999999.99，确保符合 Decimal(10,4) 范围
            stealth_ratio = min(999999.99, stealth_ratio)
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










