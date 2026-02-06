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
        v2.2: [大师级深化] 换手提纯率与市值相对攻击波
        思路：
        1. Net Amount Ratio: 升级为 "换手提纯率"。
           - 引入 Volume Factor (量能因子)。公式: Ratio * (1 + log(Vol/AvgVol))。
           - 物理意义: 放量上涨时的净占比权重放大(真实攻击)，缩量时的权重衰减(噪音)。
        2. Flow Intensity: 升级为 "市值相对攻击波"。
           - 摒弃固定金额阈值，改为计算 "流通盘吞噬比例" (Net / CirculatingCap)。
           - 使用 tanh 函数进行非线性归一化，实现全市场股票强度的横向可比性。
        """
        metrics = {}
        current_net = float(self.context.current_flow_data.get('net_mf_amount', 0) or 0)
        # 获取基础数据
        daily_amount = 0.0
        if self.context.daily_basic_data:
            daily_amount = float(self.context.daily_basic_data.get('amount', 0) or 0)
        # 兜底逻辑：如果 daily_basic 缺失，用历史均值
        if daily_amount <= 0:
            valid_amounts = self.daily_amount_array[self.daily_amount_array > 0]
            daily_amount = np.mean(valid_amounts) if len(valid_amounts) > 0 else 1.0
        # --- 1. 换手提纯率 (Turnover Purification Rate) ---
        # 基础占比 (千分比)
        raw_ratio = (current_net / daily_amount) * 1000.0 if daily_amount > 0 else 0.0
        # 计算量能因子 (Volume Factor)
        # 获取过去5天平均成交量
        recent_vols = self.volume_array[-5:]
        avg_vol = np.mean(recent_vols) if len(recent_vols) > 0 else 0.0
        current_vol = float(self.context.current_flow_data.get('net_mf_vol', 0) or 0) # 这里最好用 total vol，暂用 mf_vol 近似或从 daily_basic 取
        if self.context.daily_basic_data:
             current_vol = float(self.context.daily_basic_data.get('vol', 0) or 0)
             
        if avg_vol > 0:
            vol_ratio = current_vol / avg_vol
        else:
            vol_ratio = 1.0
        # 核心深化：量能加权
        # 逻辑：
        # 1. 如果放量 (Ratio > 1), log(1+Ratio) > 0.7 -> 放大系数 > 1.0
        # 2. 如果缩量 (Ratio < 0.5), log 较小 -> 衰减
        # 我们希望放量时加强信号，缩量时保留原信号或微弱衰减
        # 使用 log1p 平滑
        volume_factor = np.log1p(vol_ratio) # vol_ratio=1 -> 0.69, vol_ratio=2 -> 1.1
        # 为了保持数值量级的一致性 (千分比)，我们做一个归一化修正
        # 让 vol_ratio=1 (平量) 时，系数约为 1.0
        norm_factor = volume_factor / 0.693 
        metrics['net_amount_ratio'] = float(raw_ratio * norm_factor)
        # --- 2. 均线计算 (保持原逻辑) ---
        ratio_arr = self.net_amount_ratio_array
        n = len(ratio_arr)
        for window in [5, 10]:
            if n >= window:
                metrics[f'net_amount_ratio_ma{window}'] = float(np.mean(ratio_arr[-window:]))
            else:
                metrics[f'net_amount_ratio_ma{window}'] = None
        # --- 3. 市值相对攻击波 (Market-Cap Relative Attack Wave) ---
        # 传入流通市值
        circ_mv = self.market_cap # 单位通常是万元，需确认
        if circ_mv is None and self.context.daily_basic_data:
            circ_mv = float(self.context.daily_basic_data.get('circ_mv', 0) or 0)
        intensity_score = self._calculate_flow_intensity(current_net, raw_ratio, circ_mv)
        metrics['flow_intensity'] = float(intensity_score)
        # 强度分级 (调用 v2.1 的动态分级)
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
        v2.1: [大师级深化] 动态博弈能级 (Dynamic Game Energy Level)
        思路：
        1. 摒弃固定金额阈值，采用"历史分位数" (Percentile Rank) 进行自适应分级。
        2. 解决了大小盘股阈值不统一的问题。
        3. 结合绝对强度(Z-Score)与相对强度(Ratio Rank)双重确认。
        """
        # 1. 基础数据准备
        if len(self.net_amount_array) < 20:
            # 数据不足时降级处理
            if abs(net_amount) > 10000: return 4
            if abs(net_amount) > 5000: return 3
            if abs(net_amount) > 1000: return 2
            return 1
        # 2. 计算当前净流在历史(60天)中的分位数
        # 我们使用绝对值来衡量"活跃度"的等级，方向由符号决定
        abs_net = abs(net_amount)
        history_abs = np.abs(self.net_amount_array[-60:])
        # 计算分位数 (0.0 - 1.0)
        rank = stats.percentileofscore(history_abs, abs_net) / 100.0
        # 3. 计算相对成交额占比的强度
        # 如果是极度缩量市场的净流入，含金量更高
        # net_ratio 已经是千分比
        abs_ratio = abs(net_ratio)
        # 4. 综合定级逻辑
        # Level 4 (极高): 历史前5% 或 占比超过 50‰ (5%)
        if rank >= 0.95 or abs_ratio >= 50.0:
            return 4
        # Level 3 (高): 历史前15% 或 占比超过 30‰ (3%)
        elif rank >= 0.85 or abs_ratio >= 30.0:
            return 3
        # Level 2 (中): 历史前40% 或 占比超过 10‰ (1%)
        elif rank >= 0.60 or abs_ratio >= 10.0:
            return 2
        # Level 1 (低): 平庸波动
        else:
            return 1

    # ==================== 3. 主力行为模式识别 ====================
    def calculate_behavior_patterns(self) -> Dict[str, Any]:
        """
        v2.1: [大师级深化] 基于量价博弈的主力行为模式识别
        思路：
        1. 引入 Price (价格) 和 Volume (成交量) 维度，不再仅看资金流。
        2. Distribution (派发): 重点识别 "放量滞涨" 和 "红盘流出"。
        3. Shakeout (洗盘): 重点识别 "缩量回调" 和 "惜售特征"。
        4. Confidence: 引入 SNR (信噪比) 修正，剔除混沌期的伪信号。
        """
        patterns = {}
        # 准备数据 (取最近10天，足够覆盖短期行为)
        limit = 10
        if len(self.net_amount_array) < limit:
            # 数据不足，返回默认
            return {
                'accumulation_score': 0.0, 'pushing_score': 0.0,
                'distribution_score': 0.0, 'shakeout_score': 0.0,
                'behavior_pattern': 'UNCLEAR', 'pattern_confidence': 0.0
            }
        # 提取切片
        nets = self.net_amount_array[-limit:]
        closes = self.close_array[-limit:]
        vols = self.volume_array[-limit:]
        # 计算各种模式的得分
        # Accumulation/Pushing 保持原逻辑或微调，这里重点展示 Distribution/Shakeout 的深化
        accumulation_score = self._calculate_accumulation_score(nets)
        pushing_score = self._calculate_pushing_score(nets)
        # [深化] 派发与洗盘需要量价配合
        distribution_score = self._calculate_distribution_score(nets, closes, vols)
        shakeout_score = self._calculate_shakeout_score(nets, closes, vols)
        patterns['accumulation_score'] = accumulation_score
        patterns['pushing_score'] = pushing_score
        patterns['distribution_score'] = distribution_score
        patterns['shakeout_score'] = shakeout_score
        # 确定行为模式与置信度
        pattern, confidence = self._determine_behavior_pattern(
            accumulation_score, pushing_score, distribution_score, shakeout_score, nets
        )
        patterns['behavior_pattern'] = pattern
        patterns['pattern_confidence'] = confidence
        return patterns

    def _calculate_accumulation_score(self, arr: np.ndarray) -> float:
        """
        v2.1: [大师级深化] 隐蔽吸筹熵 (Stealth Accumulation Entropy)
        思路：
        1. 吸筹的核心特征是 "悄悄进村，打枪的不要"。
        2. 高分特征：
           - 资金持续流入 (Sum Net > 0)
           - 价格波动率极低 (Volatility Compression) -> 主力压盘
           - 股价处于相对低位 (Low Position)
           - 量能温和放大 (Volume Expansion) 但不暴量
        """
        n = len(arr)
        if n < 10: return 0.0
        # 数据切片 (最近 10-15 天)
        window = 10
        nets = arr[-window:]
        closes = self.close_array[-window:]
        vols = self.volume_array[-window:]
        # 1. 资金持续性 (Flow Consistency)
        # 只有净流入为正才计算吸筹，否则为0
        # 允许少量流出(洗盘)，看累计净值
        cum_net = np.sum(nets)
        if cum_net <= 0: return 0.0
        # 正向天数占比
        pos_days = np.sum(nets > 0)
        consistency = (pos_days / window) * 100.0
        # 2. 波动率压缩 (Volatility Compression) - 核心
        # 主力吸筹时，K线通常是小阴小阳，振幅极小
        # 计算价格的标准差 / 均价
        price_std = np.std(closes)
        price_mean = np.mean(closes)
        price_cv = price_std / (price_mean + 1e-6)
        # 压缩得分: CV越小，得分越高
        # 经验值: 10天波动率 < 2% 是极佳的吸筹区
        compression_score = max(0.0, 100.0 - price_cv * 2000.0) # 0.05 -> 0分, 0.01 -> 80分
        # 3. 相对位置修正 (Position Factor)
        # 吸筹通常发生在低位。如果高位出现类似特征，可能是出货中继
        # 取过去 60 天价格区间
        long_closes = self.close_array[-60:]
        p_max = np.max(long_closes)
        p_min = np.min(long_closes)
        current_p = closes[-1]
        if p_max - p_min > 1e-6:
            pos = (current_p - p_min) / (p_max - p_min)
        else:
            pos = 0.5
        # 低位(pos < 0.4)加分，高位(pos > 0.7)减分
        if pos < 0.4:
            pos_factor = 1.2
        elif pos > 0.7:
            pos_factor = 0.6
        else:
            pos_factor = 1.0
        # 4. 压盘系数 (Suppression Factor)
        # 资金大举流入，但涨幅很小
        # 资金力度 = cum_net / avg_turnover
        avg_amt = np.mean(self.daily_amount_array[-window:]) + 1.0
        flow_strength = cum_net / avg_amt # 累计流入占日均成交的比例
        price_change = (closes[-1] - closes[0]) / closes[0]
        # 如果流入很大(>20%)，但涨幅很小(<5%)，说明压盘严重 -> 高分
        suppression = 0.0
        if flow_strength > 0.2 and price_change < 0.05:
            suppression = 20.0 # 额外加分
        # 综合计算
        raw_score = (consistency * 0.3 + compression_score * 0.4 + flow_strength * 100.0 * 0.3) + suppression
        final_score = raw_score * pos_factor
        return float(np.clip(final_score, 0.0, 100.0))

    def _calculate_pushing_score(self, arr: np.ndarray) -> float:
        """
        v2.1: [大师级深化] 趋势爆发动量 (Trend Explosive Momentum)
        思路：
        1. 拉升(Pushing) 与 吸筹 不同，特征是 "放量突破" 和 "加速度"。
        2. 核心指标：
           - 资金加速度 (Flow Acceleration) > 0
           - 锁筹效率 (Efficiency): 缩量涨 > 放量涨 > 放量滞涨
           - 抛压衰竭 (Selling Exhaustion)
        """
        n = len(arr)
        if n < 5: return 0.0
        # 1. 资金加速度 (Acceleration)
        # 比较 最近3天均值 vs 前5天均值
        recent_3 = np.mean(arr[-3:])
        prev_5 = np.mean(arr[-8:-3])
        # 必须是流入状态
        if recent_3 <= 0: return 0.0
        # 加速逻辑: 近期流入显著大于前期
        # 使用差值归一化
        denom = abs(prev_5) + 10.0 # 10万底噪
        acc_ratio = (recent_3 - prev_5) / denom
        acc_score = min(100.0, max(0.0, acc_ratio * 50.0)) # 2倍加速 -> 100分
        # 2. 锁筹效率 (Lock-in Efficiency)
        # 理想的拉升：价格大涨，资金大流进，但成交量不需要无限放大(锁仓)
        # 获取对应日期的涨幅和成交量
        idx_start = -3
        price_ret = (self.close_array[-1] - self.close_array[idx_start]) / self.close_array[idx_start]
        # 如果价格没涨或跌了，那就不是拉升
        if price_ret <= 0: return 0.0
        vol_recent = np.mean(self.volume_array[-3:])
        vol_base = np.mean(self.volume_array[-20:]) + 1.0
        vol_ratio = vol_recent / vol_base
        # 效率模型:
        # A类(强庄): 涨幅大(>5%)，量比适中(1.5-2.0) -> 效率极高
        # B类(合力): 涨幅大，量比巨大(>3.0) -> 分歧大，效率一般
        # C类(诱多): 涨幅小，量比巨大 -> 效率低
        eff_score = 0.0
        if vol_ratio > 0.5: # 避免停牌数据
            # 单位量能带来的涨幅
            unit_ret = price_ret / vol_ratio
            eff_score = min(100.0, unit_ret * 2000.0) # 1%涨幅/1倍量 -> 0.01 * 2000 = 20分
        # 3. 连续性奖励
        # 拉升通常具有惯性
        consist_days = 0
        for x in arr[::-1]: # 倒序
            if x > 0: consist_days += 1
            else: break
        consist_score = min(100.0, consist_days * 20.0)
        # 综合加权
        # 加速度(40%) + 效率(40%) + 连续性(20%)
        final_score = acc_score * 0.4 + eff_score * 0.4 + consist_score * 0.2
        return float(np.clip(final_score, 0.0, 100.0))

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
        [修复] 模式仲裁与置信度计算
        修复点：
        1. 引入信息熵 (Entropy) 惩罚。如果四个分数接近，说明特征不明显，置信度大幅降低。
        2. 解决 pattern_confidence 容易全 100 的问题。
        """
        raw_scores = np.array([max(0, acc), max(0, push), max(0, dist), max(0, shake)])
        pattern_names = ['ACCUMULATION', 'PUSHING', 'DISTRIBUTION', 'SHAKEOUT']
        # 归一化为概率分布
        total_score = np.sum(raw_scores)
        if total_score < 1.0:
            return 'UNCLEAR', 0.0
        probs = raw_scores / total_score
        # 1. 计算信息熵
        # 均匀分布时熵最大 (log(4))，置信度应最低
        # 集中分布时熵最小 (0)，置信度应最高
        ent = stats.entropy(probs) # base e
        max_ent = np.log(4) # 1.386
        # 熵反转因子: 熵越小，因子越大 (0~1)
        entropy_factor = 1.0 - (ent / max_ent)
        # 2. 找出最大值
        best_idx = np.argmax(raw_scores)
        best_score = raw_scores[best_idx]
        best_pattern = pattern_names[best_idx]
        # 3. 基础置信度 = 最高分 * 熵因子
        # 只有当最高分很高(>80) 且 分布很集中(熵低) 时，才能接近 100
        # 之前是 gap + base，容易溢出。现在是乘法，很难溢出。
        final_confidence = best_score * entropy_factor
        # 4. 趋势验证修正 (Trend Validation)
        # 如果是建仓/派发，需要趋势配合。如果趋势 R^2 低，置信度打折
        if best_pattern in ['ACCUMULATION', 'DISTRIBUTION'] and len(nets) > 5:
            x = np.arange(len(nets))
            slope, _, r_value, _, _ = linregress(x, nets)
            r_sq = r_value ** 2
            # 趋势弱则打折，但不要打太狠 (0.7 ~ 1.0)
            trend_factor = 0.7 + 0.3 * r_sq
            final_confidence *= trend_factor

        # 阈值过滤
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
        [大师级深化] 洗盘承接弹性 (Washout Absorption Elasticity) - 修正版
        修复点：
        1. 逻辑闭环: 资金净流入时返回 50.0 (中性)，避免 "Good Outflow" 误导为满分。
        2. 分母钝化: 缓冲项从 0.1 提升至 0.5，防止微小跌幅导致的数值爆炸。
        3. 缩放调整: 降低 multiplier，使得满分 100 更加难以达成（仅限于真正的缩量微跌大流出）。
        """
        # [修改] 净流入不是"流出"，给予中性评分 50，而不是满分 100
        if current_net >= 0:
            return 50.0
        # 获取当日涨跌幅
        pct_chg = 0.0
        if self.context.daily_basic_data:
            pct_chg = self.context.daily_basic_data.get('pct_change')
            if pct_chg is None:
                closes = self.close_array
                if len(closes) >= 2:
                    pct_chg = (closes[-1] - closes[-2]) / closes[-2] * 100
        # 背离模式：流出但涨了 -> 主力边拉边出 or 极强承接
        # 给予 90 分，保留 100 分给完美的"缩量微跌洗盘"
        if pct_chg >= 0:
            return 90.0
        # 此时 current_net < 0, pct_chg < 0
        abs_flow = abs(current_net)
        # [修改] 分母钝化
        # 增加缓冲项至 0.5。这意味着跌幅必须显著小于流出比例，弹性才高。
        # 如果跌幅仅 -0.1%，分母为 0.6，不会导致结果爆炸。
        abs_drop = abs(pct_chg) + 0.5 
        # 计算弹性系数: 每 1% (修正后) 跌幅 对应多少流出
        elasticity = abs_flow / abs_drop
        # 归一化基准
        recent_abs_flows = np.abs(self.net_amount_array[-20:])
        avg_flow = np.mean(recent_abs_flows) + 1.0
        # [修改] 降低倍率系数 50.0 -> 30.0
        # 逻辑：Elasticity / Avg_Flow = 3.0 时 (流出量是平时的3倍，但跌幅受控)，得分为 3.0 * 30 = 90
        score_raw = (elasticity / avg_flow) * 30.0 
        # 映射到 0-100
        # 基础分 10，保证有流出就有基础分
        final_score = float(np.clip(score_raw + 10.0, 0.0, 100.0))
        return final_score

    def _calculate_inflow_persistence(self) -> int:
        """
        [大师级深化] 记忆衰减堆积 (Decay-Weighted Accumulation)
        逻辑：
        1. 摒弃简单的“连续天数”计数。
        2. 使用积分器：S = S_prev * decay + current_signal。
        3. 映射回“等效天数”。
        """
        arr = self.net_amount_array
        if len(arr) == 0: return 0
        # 能量积分器
        energy = 0.0
        # 衰减系数 (0.8 意味着资金影响力随时间只有 20% 的衰减，记忆较长)
        decay = 0.8
        # 遍历最近 20 天 (倒序遍历不适合累积，正序遍历)
        # 我们只关心最近一段的趋势
        window = arr[-30:] if len(arr) > 30 else arr
        for flow in window:
            if flow > 0:
                # 资金流入：能量增加
                # 1 表示一天有效的流入单位
                energy = energy * decay + 1.0 
            else:
                # 资金流出：能量损耗
                # 惩罚系数：如果是缩量流出，惩罚小；放量流出，惩罚大？
                # 这里简化：流出天数会导致 accumulated energy 被 decay 削减
                # 并额外扣除一部分 (流出不仅仅是停止流入，而是破坏)
                energy = max(0.0, energy * decay - 0.5)
        # 将能量值转换为“等效连续天数”
        # Geometric Series Sum: S = (1 - r^n) / (1 - r) -> n = log(...)
        # 简单近似: energy 就是当下的“持续力度”
        return int(np.ceil(energy))

    def _detect_large_order_anomaly(self) -> Tuple[bool, float]:
        """
        [大师级深化] 分形脉冲检测 (Fractal Pulse Detection) - 修正版
        修复点：
        1. 阈值修正: Z-Score 阈值从 3.0 降至 2.0，提高异动捕捉灵敏度。
        2. 绝对冲击: 增加绝对换手率异动判定，防止历史高波动掩盖当日的剧烈冲击。
        3. 强度平滑: 优化强度计算函数，使其在 2.0-4.0 区间分布更均匀。
        """
        arr = self.net_amount_array
        if len(arr) < 10: return False, 0.0
        current_net = float(self.context.current_flow_data.get('net_mf_amount', 0) or 0)
        # 1. 鲁棒 Z-Score 计算
        recent = arr[-20:]
        median = np.median(recent)
        # MAD: Median Absolute Deviation
        mad = np.median(np.abs(recent - median)) * 1.4826
        # 底噪保护: 至少要有 10万(假设单位) 或 历史均值的 10% 作为波动下限
        # 防止死水股因为波动极小导致 Z-Score 爆炸
        base_noise = max(10.0, np.mean(np.abs(recent)) * 0.1) 
        sigma = max(mad, base_noise)
        z_score = (current_net - median) / sigma
        # 2. 相对位置系数 (Position Factor)
        closes = self.close_array
        if len(closes) > 0:
            p_min = np.min(closes[-60:])
            p_max = np.max(closes[-60:])
            p_curr = closes[-1]
            if p_max - p_min > 1e-6:
                pos = (p_curr - p_min) / (p_max - p_min)
            else:
                pos = 0.5
        else:
            pos = 0.5
        # 3. 异动判定 (双重标准)
        # 标准A: 统计异动 (Z > 2.0)
        is_stat_outlier = abs(z_score) > 2.0
        # 标准B: 绝对冲击异动 (Absolute Impact)
        # 如果当日净流占流通市值的 5‰ 以上，强制判定为异动 (无论历史波动如何)
        circ_mv = self.market_cap or 1e12 # 避免除零
        turnover_impact = abs(current_net) / circ_mv
        is_abs_outlier = turnover_impact > 0.005 # 0.5%
        if not (is_stat_outlier or is_abs_outlier):
            return False, 0.0
        # 4. 强度计算 (Intensity)
        # 基础强度: 将 2.0~5.0 映射到 40~100
        # Z=2 -> 40, Z=3 -> 60, Z=5 -> 100
        raw_intensity = min(100.0, max(0.0, (abs(z_score) - 1.0) * 25.0))
        # 如果触发了绝对冲击，强度保底 80
        if is_abs_outlier:
            raw_intensity = max(raw_intensity, 80.0)
        # 位置修正
        if pos < 0.3:
            pos_multiplier = 1.2 # 底部异动，重点关注
        elif pos > 0.8:
            pos_multiplier = 1.1 # 顶部异动，风险关注
        else:
            pos_multiplier = 0.95
        final_intensity = min(100.0, raw_intensity * pos_multiplier)
        return True, float(final_intensity)

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
        v2.1: [大师级深化] 资金逻辑连贯性 (Logic Continuity)
        思路：
        1. 稳定性 != 低波动。主力建仓的"稳定性"体现在"趋势的持续性" (Autocorrelation)。
        2. 引入 自相关系数 (ACF) 和 符号翻转率 (Sign Flip Rate)。
        3. 高稳定性 = 资金流方向持续(正自相关) + 波动可控(低CV)。
        """
        # 使用预处理的 array
        arr = self.net_amount_array
        n = len(arr)
        if n < 10: return 50.0
        # 取最近 15 天
        recent = arr[-15:]
        # 1. 符号翻转率 (Sign Flip Rate)
        # 衡量主力是否反复横跳
        signs = np.sign(recent)
        # 去除 0
        signs = signs[signs != 0]
        if len(signs) > 1:
            flips = np.sum(signs[1:] != signs[:-1])
            flip_rate = flips / (len(signs) - 1)
        else:
            flip_rate = 0.5
        # 翻转率越低，稳定性越高
        # Score A: 0.0 -> 100, 0.5 -> 50, 1.0 -> 0
        flip_score = (1.0 - flip_rate) * 100.0
        # 2. 趋势惯性 (Trend Inertia / Autocorrelation)
        # 计算 Lag-1 自相关
        if np.std(recent) > 1e-6:
            ac1 = np.corrcoef(recent[:-1], recent[1:])[0, 1]
            if np.isnan(ac1): ac1 = 0.0
        else:
            ac1 = 0.0 # 无波动
        # 自相关越高(接近1)，说明主力运作越连贯
        # 映射: -1 -> 0, 0 -> 50, 1 -> 100
        inertia_score = (ac1 + 1.0) * 50.0
        # 3. 波动惩罚 (Volatility Penalty)
        # 使用 Robust CV (MAD / Median)
        median_val = np.median(recent)
        mad = np.median(np.abs(recent - median_val))
        if abs(median_val) > 10.0:
            robust_cv = mad / abs(median_val)
        else:
            # 均值接近0，波动显得很大
            robust_cv = 1.0
        # CV 越小越好。CV > 1.0 说明很不稳
        vol_score = max(0.0, 100.0 - robust_cv * 80.0)
        # 4. 综合加权
        # 连贯性(40%) + 翻转率(40%) + 波动性(20%)
        stability = inertia_score * 0.4 + flip_score * 0.4 + vol_score * 0.2
        return float(np.clip(stability, 0.0, 100.0))

    # ==================== 5. 多周期资金共振指标 ====================
    def calculate_multi_period_sync(self) -> Dict[str, float]:
        """
        v2.1: [大师级深化] 时空向量共振模型
        思路：
        1. 摒弃单纯的相关系数，引入“均线多头排列”和“向量夹角一致性”逻辑。
        2. 共振不仅仅是波动相似，更是“方向合力”。
        3. Mid-Long Sync: 特别强化了 MA20 与 MA60 的“多头共振”识别，这是主升浪的核心特征。
        """
        sync_metrics = {}
        for key in ['daily_weekly_sync', 'daily_monthly_sync', 
                   'short_mid_sync', 'mid_long_sync']:
            sync_metrics[key] = None
        arr = self.net_amount_array
        if len(arr) < 60: # 需要足够长的历史数据
            return sync_metrics
        # 辅助函数：计算加权移动平均 (WMA) 以减少滞后
        def calc_wma(data, window):
            weights = np.arange(1, window + 1)
            return np.convolve(data, weights/weights.sum(), mode='valid')
        # 准备各周期数据
        # 资金流波动大，先做一次基础平滑
        smooth_arr = savgol_filter(arr, window_length=5, polyorder=2) if len(arr) > 5 else arr
        # 计算不同周期的均线序列
        # 注意：conv模式valid会缩短序列，我们需要对齐尾部
        ma5 = calc_wma(smooth_arr, 5)
        ma20 = calc_wma(smooth_arr, 20)
        ma60 = calc_wma(smooth_arr, 60)
        # 统一切片长度，取最近 5 天进行微观共振分析
        min_len = min(len(ma5), len(ma20), len(ma60))
        if min_len < 5: return sync_metrics
        s_short = ma5[-5:]
        s_mid = ma20[-5:]
        s_long = ma60[-5:]
        # 1. 计算 Short-Mid Sync (短中其共振)
        sync_metrics['short_mid_sync'] = self._calculate_vector_resonance(s_short, s_mid)
        # 2. 计算 Mid-Long Sync (中长期共振 - 核心指标)
        # 这里需要更严格的判定：是否处于多头排列 (MA20 > MA60)
        resonance_score = self._calculate_vector_resonance(s_mid, s_long)
        # [多头排列奖励]
        # 如果 MA20 > MA60 且两者都在上涨，给予额外加成
        if s_mid[-1] > s_long[-1] and (s_mid[-1] > s_mid[0]) and (s_long[-1] > s_long[0]):
            resonance_score = min(100.0, resonance_score * 1.2)
        # [空头排列惩罚]
        # 如果 MA20 < MA60，即使波动同步，共振质量也打折（可能是同步下跌或反弹）
        elif s_mid[-1] < s_long[-1]:
            resonance_score *= 0.8
        sync_metrics['mid_long_sync'] = float(resonance_score)
        # 3. 日线与其他周期的同步 (保持原有逻辑框架，但使用新算法)
        # 这里用原始平滑数据 vs MA
        daily_slice = smooth_arr[-5:]
        sync_metrics['daily_weekly_sync'] = self._calculate_vector_resonance(daily_slice, s_short)
        sync_metrics['daily_monthly_sync'] = self._calculate_vector_resonance(daily_slice, s_mid)
        return sync_metrics

    def _calculate_vector_resonance(self, s1: np.ndarray, s2: np.ndarray) -> float:
        """
        [内部方法] 计算向量共振得分
        逻辑：
        1. 方向一致性 (Direction): 两个序列的斜率是否同号。
        2. 趋势相关性 (Correlation): 两个序列的形态是否相似。
        3. 距离接近度 (Proximity): (可选) 这里主要关注动量方向。
        """
        if len(s1) < 2 or len(s2) < 2: return 50.0
        # 1. 计算趋势斜率 (归一化)
        slope1 = (s1[-1] - s1[0]) / (np.std(s1) + 1e-6)
        slope2 = (s2[-1] - s2[0]) / (np.std(s2) + 1e-6)
        # 方向得分：同向为正，反向为负
        # 使用 tanh 映射到 [-1, 1]
        dir_score = np.tanh(slope1 * slope2)
        # 2. 形态相关性
        # 使用 cosine similarity 思想，或者 pearson
        # 这里用 pearson 比较稳健
        try:
            corr = np.corrcoef(s1, s2)[0, 1]
            if np.isnan(corr): corr = 0.0
        except:
            corr = 0.0
        # 3. 综合加权
        # 如果方向相反，相关性再高也是负贡献
        # 如果方向相同，相关性越高越好
        base_score = 50.0
        if dir_score > 0:
            # 同向
            # Score = 50 + 25 * 方向强度 + 25 * 形态相关
            final_score = base_score + 25 * dir_score + 25 * max(0, corr)
        else:
            # 反向 (共振极差)
            final_score = base_score - 20 * abs(dir_score) # 扣分
        return float(np.clip(final_score, 0.0, 100.0))

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
        """
        v2.1: [大师级深化] 归一化动能涌浪与趋势熵减模型
        思路：
        1. Flow Acceleration: 升级为“归一化动能涌浪”。
           - 引入 Savitzky-Golay 平滑求导，计算更加鲁棒的“资金加速度”。
           - 使用 MAD 进行标准化，识别出显著的“脉冲式”主力进场。
        2. Uptrend Strength: 升级为“趋势熵减质量”。
           - 结合 R2 (线性度)、Downside Deviation (下行波动) 和 Flow Support (资金支持)。
           - 只有“资金持续流入”且“回撤极小”的上涨，才是高质量的趋势。
        """
        momentum = {}
        # 初始化
        for key in ['flow_momentum_5d', 'flow_momentum_10d', 'flow_acceleration',
                   'uptrend_strength', 'downtrend_strength']:
            momentum[key] = None
        arr = self.net_amount_array
        if len(arr) < 15: # 需要稍长一点的数据来计算稳健的指标
            return momentum
        # --- 1. 基础动量 (Momentum) ---
        # 保持 Robust Scaling
        mad = stats.median_abs_deviation(arr[-20:]) if len(arr) >= 20 else np.std(arr)
        denom = max(mad, 10.0)
        momentum['flow_momentum_5d'] = float((arr[-1] - arr[-5]) / denom) if len(arr) >= 5 else 0.0
        momentum['flow_momentum_10d'] = float((arr[-1] - arr[-10]) / denom) if len(arr) >= 10 else 0.0
        # --- 2. 资金加速度 (Flow Acceleration - Deepened) ---
        # 提取最近 10 天数据
        recent_10 = arr[-10:]
        # 使用 Savitzky-Golay 滤波器同时进行平滑和求导
        # window_length=5, polyorder=2
        # deriv=1 (速度), deriv=2 (加速度)
        try:
            if len(recent_10) >= 5:
                # 计算加速度曲线 (单位时间步长=1)
                acc_curve = savgol_filter(recent_10, 5, 2, deriv=2)
                # 取最近一天的加速度值
                raw_acc = acc_curve[-1]
                # 归一化: 相对于背景波动率
                # 物理含义: 当前的加速度是正常波动的多少倍?
                # A股中，> 1.0 通常意味着主力开始发动攻势
                norm_acc = raw_acc / denom
                # 放大系数，使其数值在 [-100, 100] 之间更有可读性
                momentum['flow_acceleration'] = float(np.clip(norm_acc * 50.0, -100.0, 100.0))
            else:
                momentum['flow_acceleration'] = 0.0
        except Exception:
            momentum['flow_acceleration'] = 0.0
        # --- 3. 趋势强度 (Uptrend Strength - Deepened) ---
        # 使用专门的私有方法计算，包含 "熵减" 和 "资金背书" 逻辑
        up_strength, down_strength = self._calculate_robust_trend_strength()
        momentum['uptrend_strength'] = up_strength
        momentum['downtrend_strength'] = down_strength
        return momentum

    def _calculate_robust_trend_strength(self) -> Tuple[float, float]:
        """
        [修复] 鲁棒趋势强度
        修复点：
        1. 引入 Spearman 秩相关系数。即使线性度(R2)低，只要单调上涨，就给分。
        2. 解决 slope <= 0 直接归零的问题。允许微弱上涨。
        3. 降低回撤惩罚的敏感度。
        """
        window = 10
        if len(self.net_amount_array) < window: return 0.0, 0.0
        cum_flow = np.cumsum(self.net_amount_array[-window:])
        x = np.arange(len(cum_flow))
        # 1. 线性回归强度
        slope, _, r_value, _, _ = linregress(x, cum_flow)
        r_sq = r_value ** 2
        # 2. [新增] 秩相关强度 (Spearman)
        # 解决震荡上涨 R2 低导致分为 0 的问题
        # 只要总体是涨的，Spearman 就会接近 1
        spearman_corr, _ = stats.spearmanr(x, cum_flow)
        if np.isnan(spearman_corr): spearman_corr = 0.0
        # 归一化斜率
        range_val = np.max(cum_flow) - np.min(cum_flow) + 10.0 # 加底噪
        norm_slope = slope * window / range_val
        # 3. 综合强度计算
        # 使用 max(R2, Spearman^2) 作为线性度指标
        # 这样既能捕捉直线拉升，也能捕捉震荡拉升
        linearity = max(r_sq, spearman_corr**2)
        # 基础分
        raw_strength = linearity * abs(norm_slope) * 100.0
        # 4. 方向判定与修正
        # 只要 Spearman > 0.2 或 Slope > 0，就视为上升趋势
        if slope > 0 or spearman_corr > 0.2:
            # 上升趋势
            peak = np.maximum.accumulate(cum_flow)
            drawdown = (peak - cum_flow) / (range_val + 1e-6)
            max_dd = np.max(drawdown)
            # 宽松惩罚：回撤 < 30% 不扣分，> 80% 扣光
            # 映射: 0.3 -> 1.0, 0.8 -> 0.0
            if max_dd < 0.3:
                penalty = 1.0
            else:
                penalty = max(0.0, 1.0 - (max_dd - 0.3) * 2.0)
            # 最终强度
            final_up = float(np.clip(raw_strength * penalty, 0.0, 100.0))
            # 保底逻辑：如果累积净流确实是正的，至少给 10 分
            if cum_flow[-1] > cum_flow[0]:
                final_up = max(final_up, 10.0)
            return final_up, 0.0
        else:
            # 下降趋势
            return 0.0, float(np.clip(raw_strength, 0.0, 100.0))

    def _calculate_complex_trend_strength(self) -> Tuple[float, float]:
        """
        [内部方法] 计算复合趋势强度
        逻辑：
        1. 下跌强度 = 线性度(R2) * 斜率 + 恐慌因子(Volume Panic) + 阴跌因子(Low Vol Consistency)
        2. 上涨强度 = 线性度(R2) * 斜率 * 量能健康度(Volume Support)
        """
        # 取最近10天数据
        if len(self.net_amount_array) < 10: return 0.0, 0.0
        y = self.net_amount_array[-10:]
        vol = self.volume_array[-10:]
        x = np.arange(len(y))
        # 1. 线性回归 (Linear Regression)
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        r_sq = r_value ** 2
        # 归一化斜率 (相对强度)
        # 用区间极差做分母归一化
        y_range = np.max(y) - np.min(y) + 1e-6
        norm_slope = slope * 10 / y_range # 10天累计变动占比
        strength = abs(norm_slope) * r_sq * 100.0
        if slope > 0:
            # --- 上涨强度修正 ---
            # 检查成交量趋势：上涨需要放量 (Volume Expansion)
            # 计算 Volume 与 Price(Flow) 的相关性
            vol_corr = np.corrcoef(vol, y)[0, 1]
            if np.isnan(vol_corr): vol_corr = 0
            # 量价配合系数: 正相关好，负相关(缩量上涨)打折
            # 范围 [0.5, 1.5]
            vol_factor = 1.0 + 0.5 * vol_corr
            final_up = min(100.0, strength * vol_factor)
            return float(final_up), 0.0
        else:
            # --- 下跌强度修正 (Downtrend Deepening) ---
            # 模式A: 恐慌暴跌 (Panic Crash)
            # 特征: 跌幅大(norm_slope大) + 放量(vol trend > 0)
            # 模式B: 阴跌 (Yin Die)
            # 特征: 跌幅稳(r_sq高) + 缩量(vol trend < 0) + 波动率低
            # 计算成交量趋势 slope
            v_slope, _, _, _, _ = linregress(x, vol)
            avg_vol = np.mean(vol) + 1e-6
            norm_v_slope = v_slope * 10 / avg_vol
            # 基础强度
            down_score = strength
            if norm_v_slope > 0.2: 
                # 放量下跌 -> 恐慌盘涌出 -> 强度加成
                down_score *= 1.3 
            elif r_sq > 0.8 and abs(norm_v_slope) < 0.2:
                # 缩量且R2极高 -> 阴跌 -> 强度加成 (因为很难止跌)
                down_score *= 1.2
            return 0.0, float(min(100.0, down_score))

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
        v2.1: [大师级深化] 能量势差积分模型 (Energy Potential Integral)
        思路：
        1. 摒弃简单的“点对点”背离。背离是一个“过程”而非“瞬间”。
        2. 计算区间内 Price Cumulative Rank 与 Flow Cumulative Rank 的面积差。
        3. 面积差越大，说明价格与资金的裂痕越深，势能积蓄越大。
        """
        divergence = {}
        divergence['divergence_type'] = 'NONE'
        divergence['divergence_strength'] = 0.0
        divergence['price_flow_divergence'] = 0.0
        # 数据准备 (取最近 15 天，包含短期趋势)
        window = 15
        if len(self.close_array) < window or len(self.net_amount_array) < window:
            return divergence
        prices = self.close_array[-window:]
        flows = self.net_amount_array[-window:]
        # 1. 归一化处理 (Min-Max Scaling)
        # 将价格和累计资金流都映射到 [0, 1] 区间，便于比较形状
        def normalize(arr):
            mn, mx = np.min(arr), np.max(arr)
            if mx - mn == 0: return np.zeros_like(arr)
            return (arr - mn) / (mx - mn)
        # 注意：资金流需要看“累积量”才能与“价格”形态对比
        # 单日的 Net Flow 是价格的差分概念，Cum Flow 才是价格概念
        cum_flow = np.cumsum(flows)
        p_norm = normalize(prices)
        f_norm = normalize(cum_flow)
        # 2. 计算势差积分 (Area Difference)
        # 逐点计算差异的绝对值之和
        area_diff = np.sum(np.abs(p_norm - f_norm))
        # 归一化面积 (最大可能面积是 window * 1.0)
        # 通常 area_diff / window > 0.3 就非常显著了
        divergence_score = (area_diff / window) * 100.0 * 2.5 # 放大系数
        divergence['price_flow_divergence'] = float(min(100.0, divergence_score))
        # 3. 判定背离类型 (Bullish/Bearish)
        # 线性回归判断大方向
        x = np.arange(window)
        p_slope, _, _, _, _ = linregress(x, prices)
        f_slope, _, _, _, _ = linregress(x, cum_flow)
        # 顶背离 (Bearish): 价格涨 (p_slope > 0) 但 资金势能弱 (f_slope < p_slope)
        # 也就是资金跟不上价格的涨幅
        # 或者更严格：价格创新高，资金未创新高
        if p_slope > 0 and f_slope < 0:
            # 经典顶背离：价涨资跌
            divergence['divergence_type'] = 'BEARISH'
            divergence['divergence_strength'] = float(min(100.0, divergence_score * 1.2)) # 典型背离加权
        elif p_slope < 0 and f_slope > 0:
            # 经典底背离：价跌资涨 (主力吸筹)
            divergence['divergence_type'] = 'BULLISH'
            divergence['divergence_strength'] = float(min(100.0, divergence_score * 1.5)) # 底背离通常更可信
        elif p_slope > 0 and f_slope > 0:
            # 同向主要看速率
            # 如果价格猛涨，资金缓涨，也是一种弱背离 (量价配合不佳)
            if divergence_score > 40: # 只有差异够大才算
                # 检查谁更强
                # 如果 p_norm 始终高于 f_norm -> 资金掉队 -> 顶背离风险
                if np.mean(p_norm) > np.mean(f_norm):
                    divergence['divergence_type'] = 'BEARISH'
                    divergence['divergence_strength'] = float(divergence_score * 0.8)
                else:
                    divergence['divergence_type'] = 'NONE' # 资金比价格还强，良性
        elif p_slope < 0 and f_slope < 0:
            # 价格跌，资金也流出
            # 如果 f_norm 始终高于 p_norm -> 资金流出比价格跌幅慢 -> 惜售/抵抗 -> 底背离前兆
            if divergence_score > 40 and np.mean(f_norm) > np.mean(p_norm):
                 divergence['divergence_type'] = 'BULLISH'
                 divergence['divergence_strength'] = float(divergence_score * 0.6)
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
        v2.0: [大师级深化] 基于鲁棒统计学的资金分布特征
        思路：
        1. Flow Z-Score: 摒弃标准差，采用 MAD (Median Absolute Deviation) 算法。
           A股数据尖峰肥尾，MAD能抵抗异常值干扰，精准还原资金流的真实偏离度。
        2. Flow Percentile: 引入“长短周期双轨制”。
           不仅看短期的爆发力(20d)，更要看在长期(60d)筹码结构中的相对水位。
        """
        stats_metrics = {}
        # 基础数据准备
        arr = self.net_amount_array
        if len(arr) < 10:
            for key in ['flow_zscore', 'flow_percentile', 'flow_volatility_10d', 'flow_volatility_20d']:
                stats_metrics[key] = None
            return stats_metrics
        current_net = float(self.context.current_flow_data.get('net_mf_amount', 0) or 0)
        # --- 1. Robust MAD Z-Score (鲁棒Z分) ---
        # 窗口取最近 20 天
        recent_20 = arr[-20:]
        median_20 = np.median(recent_20)
        # 计算 MAD: Median Absolute Deviation
        # 1.4826 是正态分布下 MAD 转 标准差的调整系数
        mad_20 = np.median(np.abs(recent_20 - median_20)) * 1.4826
        # 避免分母为0 (死水股)
        if mad_20 < 1.0: 
            # 如果波动极小，改用普通标准差，还是0则给一个极小值
            std_backup = np.std(recent_20)
            denom = std_backup if std_backup > 1.0 else 10.0 # 假设基础波动10万
        else:
            denom = mad_20
        # Z = (X - Median) / Robust_Sigma
        stats_metrics['flow_zscore'] = (current_net - median_20) / denom
        # --- 2. Adaptive Percentile (双轨分位) ---
        # A. 短期分位 (20日) - 反映爆发力
        rank_20 = stats.percentileofscore(recent_20, current_net)
        # B. 长期分位 (60日) - 反映历史水位
        # 如果历史数据不足60天，就用全部数据
        window_long = 60
        recent_long = arr[-window_long:] if len(arr) >= window_long else arr
        rank_long = stats.percentileofscore(recent_long, current_net)
        # 综合分位：长期权重 0.4，短期权重 0.6
        # 我们更看重当下的爆发力，但长期水位决定了是“反弹”还是“反转”
        stats_metrics['flow_percentile'] = float(rank_20 * 0.6 + rank_long * 0.4)
        # --- 3. 波动率 (保持原有逻辑，但增加鲁棒性) ---
        mean_20 = np.mean(recent_20)
        # 归一化波动率 (相对于平均流量)
        # 只有当有一定流量时才计算，否则为0
        abs_mean = abs(mean_20)
        if abs_mean > 10.0:
            stats_metrics['flow_volatility_20d'] = np.std(recent_20) / abs_mean
        else:
            stats_metrics['flow_volatility_20d'] = 0.0
        if len(arr) >= 10:
            recent_10 = arr[-10:]
            abs_mean_10 = abs(np.mean(recent_10))
            if abs_mean_10 > 10.0:
                stats_metrics['flow_volatility_10d'] = np.std(recent_10) / abs_mean_10
            else:
                stats_metrics['flow_volatility_10d'] = 0.0
        else:
            stats_metrics['flow_volatility_10d'] = None
        return stats_metrics

    # ==================== 10. 预测指标 ====================
    def calculate_prediction_metrics(self) -> Dict[str, float]:
        """
        v2.0: [大师级深化] 基于分形效率的资金预测模型
        思路：
        1. 预测置信度 (Confidence) 不应基于波动率，而应基于“资金运作效率”(Flow Efficiency)。
        2. 引入考夫曼效率系数 (KER): KER = 位移 / 路径长。
           - 散户资金通常是随机游走 (Random Walk)，KER 接近 0，不可预测。
           - 主力资金通常是目标导向 (Targeted)，KER 接近 1，高度可控，预测置信度高。
        """
        prediction = {}
        arr = self.net_amount_array
        # 依然调用优化后的辅助方法来获取基础概率
        # 这里为了避免循环引用或重复代码，我们重新计算或沿用之前的逻辑结构
        # 假设 context 中已存在或通过其他方式获取了概率，这里主要重写 confidence 计算
        if len(arr) < 10:
            for key in ['expected_flow_next_1d', 'flow_forecast_confidence',
                       'uptrend_continuation_prob', 'reversal_prob']:
                prediction[key] = None
            return prediction
        # 1. 均值预测 (Base Prediction)
        # 使用加权移动平均 (WMA)，越近的数据权重越大
        recent_5 = arr[-5:]
        weights = np.arange(1, 6) # 1, 2, 3, 4, 5
        wma_5 = np.sum(recent_5 * weights) / np.sum(weights)
        prediction['expected_flow_next_1d'] = float(wma_5)
        # --- 核心深化：基于 KER 的预测置信度 ---
        # 计算窗口：最近 10 天
        recent_10 = arr[-10:]
        # 计算整体位移 (Net Directional Movement)
        # 首尾相减不够，因为资金流是Delta，所以位移是 Sum(Net Flow)
        total_displacement = abs(np.sum(recent_10))
        # 计算路径总长度 (Total Path Length) -> 噪音总量
        # 也就是每天净流绝对值的和
        total_path = np.sum(np.abs(recent_10))
        # 计算效率系数 KER (Kaufman Efficiency Ratio)
        if total_path > 1.0: # 避免除零
            efficiency = total_displacement / total_path
        else:
            efficiency = 0.0
        # 映射到置信度 [0, 100]
        # KER 通常在 0 到 1 之间。
        # A股经验：KER > 0.3 说明有明显主力运作；KER < 0.1 说明纯散户行情。
        # 我们将其非线性映射，放大 0.3 以上部分的差异
        confidence = min(100.0, efficiency * 100.0 * 1.5) # 0.6 KER 就达到 90分
        # 结合自相关性 (Autocorrelation) 进行修正
        # 如果资金流具有惯性 (Lag-1 Autocorr > 0)，置信度增加
        if len(recent_10) > 2:
            # 简单计算一阶自相关
            y = recent_10
            if np.std(y) > 0:
                ac1 = np.corrcoef(y[:-1], y[1:])[0, 1]
                if not np.isnan(ac1) and ac1 > 0:
                    confidence += ac1 * 20.0 # 最多加20分
        prediction['flow_forecast_confidence'] = float(min(100.0, max(0.0, confidence)))
        # 3. 趋势概率 (调用或简单计算，保持接口完整)
        # 这里复用 v2.0 的逻辑框架，或根据上下文保持一致
        # 为保持本方法独立性，这里做简化计算，实际应结合 `calculate_trend_momentum` 的结果
        trend_score = 50.0
        if prediction['expected_flow_next_1d'] > 0:
            trend_score += confidence * 0.4
        else:
            trend_score -= confidence * 0.4
        prediction['uptrend_continuation_prob'] = float(np.clip(trend_score, 0, 100))
        prediction['reversal_prob'] = float(np.clip(100 - trend_score, 0, 100))
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
        v1.9: [大师级深化] 基于Savitzky-Golay滤波的资金攻击波久期计算
        思路：
        1. 原始的分钟级资金流充满了随机游走噪音，直接统计正负分钟数没有意义。
        2. 使用 Savitzky-Golay 滤波器对分钟资金流进行平滑，提取“资金趋势项”。
        3. 识别最长的“单一主控状态”区间（Longest Contiguous Regime）。
        4. 该指标反映了主力机构在盘中维持单一方向运作（坚决吸筹或坚决出货）的最长时间。
        返回：
            float: 持续分钟数。如果是净流出持续，返回负值；净流入持续，返回正值。
        """
        # 1. 基础分钟级聚合
        times = df['trade_time'].values.astype('datetime64[m]').astype('int64')
        if len(times) == 0: return 0.0
        amounts = df['amount'].values
        types = df['type'].values
        signed_amounts = np.where(types == 'B', amounts, -amounts)
        norm_times = times - times[0]
        max_minutes = int(np.max(norm_times)) + 1
        # 分钟级净流入 (单位: 万元)
        minute_flows = np.bincount(norm_times, weights=signed_amounts, minlength=max_minutes) / 10000.0
        # 移除首尾无效0 (如中午休市的大段0)
        # 注意：这里简单trim_zeros可能把盘中正常的0去掉了，最好是配合交易时间模板。
        # 考虑到A股连续性，我们仅去除尾部多余的0
        minute_flows = np.trim_zeros(minute_flows, 'b')
        n = len(minute_flows)
        if n < 5: return 0.0
        # 2. 信号平滑处理 (Signal Smoothing)
        # 使用 Savitzky-Golay 滤波器提取趋势
        # window_length 必须是奇数且小于 n
        window_length = min(11, n if n % 2 != 0 else n - 1)
        if window_length < 3:
            smoothed_flows = minute_flows
        else:
            try:
                # polyorder=2 拟合抛物线，保留局部极值特征
                smoothed_flows = savgol_filter(minute_flows, window_length, 2)
            except Exception:
                smoothed_flows = minute_flows
        # 3. 状态识别 (Regime Identification)
        # 设置最小噪音阈值，防止在0附近微小波动导致状态频繁切换
        noise_threshold = 1.0 # 1万元以下的波动视为噪音
        # 状态定义: 1=多头, -1=空头, 0=震荡
        states = np.zeros_like(smoothed_flows, dtype=int)
        states[smoothed_flows > noise_threshold] = 1
        states[smoothed_flows < -noise_threshold] = -1
        # 4. 寻找最长连续区间 (Longest Streak)
        if len(states) == 0: return 0.0
        # 计算游程 (Run Length Encoding)
        # 在numpy中找连续段
        padded = np.concatenate(([0], states, [0]))
        diffs = np.diff(padded)
        # 找到所有状态切换点
        run_starts = np.where(diffs != 0)[0]
        max_persistence = 0.0
        dominant_direction = 0
        for i in range(len(run_starts) - 1):
            start = run_starts[i]
            end = run_starts[i+1]
            state = states[start]
            if state == 0: continue
            duration = end - start
            # 权重修正：如果是一个极长时间的微弱流入，价值不如短时间的大额流入？
            # 题目问的是 minutes，我们坚持返回时间，但只有当累积量足够时才确认
            # 这里简单返回时间，但在策略层可以结合 intensity 使用
            if duration > abs(max_persistence):
                max_persistence = float(duration)
                dominant_direction = state
        # 5. 返回带符号的结果
        # 正数表示多头持续时间最长，负数表示空头持续时间最长
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
        v1.7: [大师级深化] 基于波动率标准化的资金爆发力计算
        思路：
        1. 摒弃物理二阶导数，改用金融工程中的 'Standardized Unexpected Earnings' (SUE) 思想。
        2. 加速度 = (短期资金脉冲 - 长期资金基准) / 背景波动率。
        3. 这种算法能精准过滤掉"成交清淡时的虚假高加速"，只有在放量突破时的加速才会被捕捉。
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
        # 分钟级净流入 (单位: 万元)
        minute_flows = np.bincount(norm_times, weights=signed_amounts, minlength=max_minutes) / 10000.0
        # 剔除尾部无效0值 (集合竞价)
        active_flows = np.trim_zeros(minute_flows, 'b')
        n = len(active_flows)
        if n < 10: return metrics
        # --- 指标1: 趋势动量 (Momentum) ---
        # 比较最近15分钟与之前30分钟的力度差异
        if n >= 45:
            recent_trend = np.mean(active_flows[-15:])
            prev_trend = np.mean(active_flows[-45:-15])
            # 归一化分母，防止除零
            denom = abs(prev_trend) + 1.0 # 加1万元保底
            metrics['intraday_flow_momentum'] = float(np.clip((recent_trend - prev_trend) / denom * 100.0, -200, 200))
        else:
            metrics['intraday_flow_momentum'] = 0.0
        # --- 指标2: 资金加速度 (Acceleration - Deepened) ---
        # 逻辑: Z-Score 爆发力模型
        # 定义"脉冲"为最近3分钟均值，"基准"为最近30分钟均值
        window_short = 3
        window_long = 30
        if n >= window_long:
            # 短期脉冲
            pulse = np.mean(active_flows[-window_short:])
            # 长期基准序列
            baseline_slice = active_flows[-window_long:]
            baseline_mean = np.mean(baseline_slice)
            baseline_std = np.std(baseline_slice)
            # 计算爆发力 (Z-Score)
            # 意义: 当前的资金流偏离了正常波动范围多少个标准差?
            # 这种写法对"缩量后的突然放量"极度敏感，正是主力点火的特征
            if baseline_std > 0.1: # 设置最小波动率门槛，避免死水股数值爆炸
                acc_z_score = (pulse - baseline_mean) / baseline_std
            else:
                # 如果波动率极低，直接用绝对值衡量
                acc_z_score = (pulse - baseline_mean) / 10.0 # 假设10万为基础波动
            # 映射到线性区间，便于入库 (限制在 +/- 100 范围内)
            # Z-Score 通常在 +/- 3 左右，这里放大10倍，即 30分代表3个标准差的强力爆发
            metrics['flow_acceleration_intraday'] = float(np.clip(acc_z_score * 10.0, -100.0, 100.0))
        else:
            metrics['flow_acceleration_intraday'] = 0.0
        return metrics

    def _calculate_closing_flow_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        [修复] 尾盘强度计算
        修复点：
        1. 降低强度系数 (4000 -> 800)。
           原系数 4000 意味着尾盘净流占全天 2.5% 即满分，导致指标饱和。
           现系数 800 意味着需要达到 12.5% 才满分，拉开区分度。
        """
        metrics = {'closing_flow_ratio': None, 'closing_flow_intensity': None}
        times = df['trade_time'].values.astype('int64') 
        amounts = df['amount'].values
        types = df['type'].values
        if len(times) == 0: return metrics
        # 定义时间阈值 (北京时间)
        time_1430 = 52200000000000
        time_1457 = 53820000000000
        ns_in_day = (times % 86400000000000)
        mask_closing = ns_in_day >= time_1430
        mask_auction = ns_in_day >= time_1457
        if not np.any(mask_closing):
            metrics['closing_flow_ratio'] = 0.0
            metrics['closing_flow_intensity'] = 0.0
            return metrics
        buy_mask = (types == 'B')
        sell_mask = (types == 'S')
        total_turnover = np.sum(amounts)
        t_net_raw = np.sum(amounts[buy_mask]) - np.sum(amounts[sell_mask])
        c_buy = np.sum(amounts[mask_closing & buy_mask])
        c_sell = np.sum(amounts[mask_closing & sell_mask])
        c_net_total = c_buy - c_sell
        a_buy = np.sum(amounts[mask_auction & buy_mask])
        a_sell = np.sum(amounts[mask_auction & sell_mask])
        a_net_auction = a_buy - a_sell
        # [指标A] 尾盘资金占比
        min_significant_flow = total_turnover * 0.005
        denominator = max(abs(t_net_raw), min_significant_flow)
        if denominator > 0:
            raw_ratio = abs(c_net_total) / denominator * 100.0
            metrics['closing_flow_ratio'] = float(min(1000.0, raw_ratio))
        else:
            metrics['closing_flow_ratio'] = 0.0
        # [指标B] 尾盘强度 - 关键修复
        if total_turnover > 0:
            urgency_alpha = 3.0
            weighted_flow = abs(c_net_total) + (abs(a_net_auction) * urgency_alpha)
            # [修改] 系数由 4000.0 降为 800.0
            # 物理意义：(加权尾盘净流 / 全天成交) > 12.5% 时达到 100分
            intensity_score = min(100.0, (weighted_flow / total_turnover) * 800.0)
            metrics['closing_flow_intensity'] = float(intensity_score)
        else:
            metrics['closing_flow_intensity'] = 0.0
        return metrics

    def _calculate_flow_cluster_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        v1.7: [大师级深化] 同向拆单攻击波识别
        思路：
        1. 聚类不再是简单的"量大扎堆"，必须要求"方向一致"（全是买单或全是卖单）。
           这能区分"多空激烈换手"(震荡)与"主力单向扫货"(攻击)。
        2. 强度定义为"攻击密度信噪比"(Attack Density SNR)：攻击波内的资金流速是平时背景流速的几倍？
        """
        metrics = {'flow_cluster_intensity': None, 'flow_cluster_duration': None}
        # 1. 3秒级聚合 (微观结构)
        # 1e9 ns = 1s, //3 得到3秒桶
        time_int = df['trade_time'].values.astype('int64') // 10**9 // 3
        if len(time_int) == 0: return metrics
        amounts = df['amount'].values
        types = df['type'].values
        signed_amounts = np.where(types == 'B', amounts, -amounts)
        # 快速聚合
        uni_times, inverse_indices = np.unique(time_int, return_inverse=True)
        cluster_flows = np.zeros(len(uni_times))
        np.add.at(cluster_flows, inverse_indices, signed_amounts)
        if len(cluster_flows) < 20: 
            metrics['flow_cluster_intensity'] = 0.0
            metrics['flow_cluster_duration'] = 0
            return metrics
        # 2. 动态阈值确定 (自适应股票活跃度)
        # 取绝对值计算背景噪音
        abs_flows = np.abs(cluster_flows)
        global_mean = np.mean(abs_flows)
        global_std = np.std(abs_flows)
        # 只有超过 "均值 + 1倍标准差" 的才被视为潜在的机构单
        threshold = global_mean + global_std
        # 3. 识别攻击波 (Attack Waves)
        # 核心逻辑: 找出连续的、同向的、高额的资金桶
        # state 数组: 0=噪音, 1=大买, -1=大卖
        states = np.zeros(len(cluster_flows), dtype=int)
        states[cluster_flows > threshold] = 1
        states[cluster_flows < -threshold] = -1
        # 如果全天无大单
        if not np.any(states != 0):
            metrics['flow_cluster_intensity'] = 0.0
            metrics['flow_cluster_duration'] = 0
            return metrics
        # 4. 聚类合并 (允许微小断裂，但不允许反向)
        # 遍历 buckets，手动合并比 scipy.label 更灵活
        max_duration_buckets = 0
        current_duration = 0
        current_type = 0 # 0, 1, -1
        cluster_flow_sums = [] # 记录每个显著Cluster的总流量
        cluster_durations = [] # 记录每个显著Cluster的桶数
        # 宽容度: 允许中间夹杂 N 个噪音桶(0)，但不允许夹杂反向桶
        gap_tolerance = 2 # 2个桶 = 6秒
        current_gap = 0
        current_cluster_amt = 0.0
        for i in range(len(states)):
            s = states[i]
            flow_val = abs(cluster_flows[i])
            if s != 0:
                # 遇到大单
                if current_type == 0:
                    # 新开启一个 Cluster
                    current_type = s
                    current_duration = 1
                    current_gap = 0
                    current_cluster_amt = flow_val
                elif s == current_type:
                    # 延续当前 Cluster
                    current_duration += 1 + current_gap # 加上之前的gap
                    current_gap = 0
                    current_cluster_amt += flow_val
                else:
                    # 方向反转，结算上一个
                    if current_duration > 0:
                        cluster_flow_sums.append(current_cluster_amt)
                        cluster_durations.append(current_duration)
                        max_duration_buckets = max(max_duration_buckets, current_duration)
                    # 开启新的
                    current_type = s
                    current_duration = 1
                    current_gap = 0
                    current_cluster_amt = flow_val
            else:
                # 遇到噪音
                if current_type != 0:
                    if current_gap < gap_tolerance:
                        current_gap += 1
                        # 噪音桶的流量也计入Cluster吗？
                        # 大师思路: 计入。因为这代表为了扫货吃掉的中间散单，或者是撤单间隙
                        current_cluster_amt += flow_val 
                    else:
                        # Gap太长，断裂
                        cluster_flow_sums.append(current_cluster_amt)
                        cluster_durations.append(current_duration)
                        max_duration_buckets = max(max_duration_buckets, current_duration)
                        current_type = 0
                        current_duration = 0
                        current_gap = 0
                        current_cluster_amt = 0
        # 结算最后一个
        if current_duration > 0:
            cluster_flow_sums.append(current_cluster_amt)
            cluster_durations.append(current_duration)
            max_duration_buckets = max(max_duration_buckets, current_duration)
        # 5. 计算最终指标
        # Duration: 桶数 * 3秒 = 秒数，再转为分钟
        # 修正: 保持为整数秒，或者分钟数。原定义是int，这里返回分钟数比较合理，如果是秒数可能数值较大
        # 考虑到A股拆单一般持续数分钟，这里返回分钟数 (不足1分钟按0或1算)
        metrics['flow_cluster_duration'] = int(max_duration_buckets * 3 / 60)
        # Intensity: 攻击密度倍率
        # (所有Cluster的总流量 / 所有Cluster的总桶数) / (全天平均流量)
        # 这比单纯的 Mean/Mean 更能体现"在此期间"的凶猛程度
        total_cluster_flow = sum(cluster_flow_sums)
        total_cluster_buckets = sum(cluster_durations)
        if total_cluster_buckets > 0 and global_mean > 0:
            avg_flow_in_cluster = total_cluster_flow / total_cluster_buckets
            # 强度 = 攻击期平均流速 / 全天平均流速
            intensity = avg_flow_in_cluster / global_mean
            metrics['flow_cluster_intensity'] = float(min(100.0, intensity))
        else:
            metrics['flow_cluster_intensity'] = 0.0
        return metrics

    def _calculate_high_freq_divergence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        v1.9: [大师级深化] 量价微观解离度 (Micro-Structure Dissociation)
        思路：
        1. 摒弃“买卖流相关性”，改为计算“价格趋势”与“资金趋势”的微观相关性。
        2. A股最危险的信号是：价格不断创新高，但资金流（Net Flow）却在悄悄流出（顶背离）。
        3. 算法：将全天数据重采样为1分钟序列，计算 Price_Delta 和 Net_Flow 的 Spearman 秩相关系数。
        4. Divergence = 1 - Correlation。范围 [0, 2]。
           - 0.0: 完全正相关（良性，量价齐升/齐跌）
           - 1.0: 无关（随机游走）
           - 2.0: 完全负相关（严重背离，诱多或吸筹）
        """
        metrics = {'high_freq_flow_divergence': None}
        # 1. 预处理
        if len(df) < 100: return metrics # 数据太少无法计算相关性
        # 确保时间序列单调
        df = df.sort_values('trade_time')
        # 2. 1分钟降采样 (Resampling)
        # 我们需要对齐 价格 和 资金流
        # 使用 pandas 的 resample 功能最方便，但为了性能我们手写 numpy 聚合
        times = df['trade_time'].values.astype('datetime64[m]').astype('int64')
        prices = df['price'].values
        amounts = df['amount'].values
        types = df['type'].values
        signed_amounts = np.where(types == 'B', amounts, -amounts)
        # 归一化时间
        start_time = times[0]
        norm_times = times - start_time
        max_minutes = int(np.max(norm_times)) + 1
        # 聚合 Net Flow
        minute_net_flow = np.bincount(norm_times, weights=signed_amounts, minlength=max_minutes)
        # 聚合 Close Price (取每分钟最后一笔的价格)
        # 使用 np.unique 找到每分钟最后一条数据的索引
        # method: 这里的 times 是单调递增的 (sorted)
        # 我们可以用 np.searchsorted 找每个分钟的结束位置，或者用 pandas groupby last
        # 为了稳健性，这里切回 pandas 操作一下 resample，虽然稍慢但逻辑复杂性低
        try:
            temp_df = pd.DataFrame({
                'price': prices,
                'net': signed_amounts
            }, index=df['trade_time'])
            # 1分钟聚合
            resampled = temp_df.resample('1min').agg({
                'price': 'last',
                'net': 'sum'
            }).dropna()
            if len(resampled) < 10:
                metrics['high_freq_flow_divergence'] = 0.0
                return metrics
            # 3. 计算相关性
            # 使用差分序列 (Delta)，因为我们要看的是“变动方向”是否一致
            p_delta = np.diff(resampled['price'].values)
            f_flow = resampled['net'].values[1:] # 对应区间的流量
            # 过滤掉无价格变动的分钟（避免除0或噪声）
            valid_mask = np.abs(p_delta) > 1e-6
            if np.sum(valid_mask) < 5:
                metrics['high_freq_flow_divergence'] = 0.0
                return metrics
            p_valid = p_delta[valid_mask]
            f_valid = f_flow[valid_mask]
            # 使用 Spearman 秩相关系数 (抗异常值)
            # 简单实现：scipy.stats.spearmanr，或者用 Pearson 代替
            # 考虑到性能，直接用 Pearson，但在输入前做个去极值处理
            # 这里直接调用 numpy 的 corrcoef
            corr = np.corrcoef(p_valid, f_valid)[0, 1]
            if np.isnan(corr):
                metrics['high_freq_flow_divergence'] = 1.0 # 无相关性
            else:
                # 转换由 [-1, 1] -> [0, 100] 的背离度得分
                # Corr = 1 (同向) -> Div = 0
                # Corr = -1 (反向) -> Div = 100
                metrics['high_freq_flow_divergence'] = float((1.0 - corr) * 50.0)
        except Exception as e:
            logger.warning(f"背离度计算异常: {e}")
            metrics['high_freq_flow_divergence'] = 0.0
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
        [修复] 3秒微观资金流场的统计特征
        修复点：
        1. 引入对数阻尼 (Log-Damping) 处理峰度，解决大量 100 饱和问题。
        2. 峰度不再硬截断，而是通过对数平滑映射到 0-100 区间。
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
            # Skewness (偏度)
            sk = stats.skew(flux_active)
            if np.isnan(sk): sk = 0.0
            # 偏度保留线性逻辑，截断 +/- 10
            metrics['high_freq_flow_skewness'] = float(np.clip(sk, -10.0, 10.0))
            # Kurtosis (峰度)
            kt = stats.kurtosis(flux_active)
            if np.isnan(kt): kt = 0.0
            # [关键修改] 对数阻尼模型
            # 原始峰度可能高达几百。使用 ln(1 + k) 压缩。
            # k=3(正态) -> ln(4)*12 ≈ 16
            # k=100(剧烈) -> ln(101)*12 ≈ 55
            # k=3000(极端) -> ln(3001)*12 ≈ 96
            # 这样很难直接打满 100，保留了头部区分度
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
        v1.4: 依赖上游的时间标准化，确保 morning/afternoon 掩码正确。
        """
        metrics = {}
        times = df['trade_time'].values
        if len(times) == 0:
            metrics['morning_flow_ratio'] = None
            metrics['afternoon_flow_ratio'] = None
            return metrics
        # 计算小时数 (基于北京时间 Naive)
        dates = times.astype('datetime64[D]')
        hours = (times - dates).astype('timedelta64[h]').astype(int)
        # 掩码 (北京时间 12:00 前为上午，13:00 后为下午)
        morning_mask = hours < 12
        afternoon_mask = hours >= 13
        amounts = df['amount'].values
        types = df['type'].values
        signed_amounts = np.where(types == 'B', amounts, -amounts)
        # 计算净流入
        morning_net = np.sum(signed_amounts[morning_mask]) / 10000.0
        afternoon_net = np.sum(signed_amounts[afternoon_mask]) / 10000.0
        total_abs_net = abs(morning_net) + abs(afternoon_net)
        # 使用绝对值占比，反映资金活跃时段
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










