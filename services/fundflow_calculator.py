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
        """预处理数据，计算中间变量"""
        # 提取历史净流入序列
        self.net_amount_series = [
            float(data.get('net_mf_amount', 0) or 0) 
            for data in self.context.historical_flow_data
        ]
        # 提取历史成交量（如果有）
        if self.context.volume_data:
            self.volume_series = self.context.volume_data
        else:
            self.volume_series = [
                float(data.get('total_volume', 0) or 0) 
                for data in self.context.historical_flow_data
            ]
        # 计算市值（如果可用）
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
        """计算绝对量级指标"""
        metrics = {}
        # 当前日净流入
        current_net = float(self.context.current_flow_data.get('net_mf_amount', 0) or 0)
        # 3日、5日、10日、20日累计净流入
        windows = [3, 5, 10, 20]
        for window in windows:
            if len(self.net_amount_series) >= window:
                total = sum(self.net_amount_series[-window:])
                metrics[f'total_net_amount_{window}d'] = total
                metrics[f'avg_daily_net_{window}d'] = total / window
            else:
                metrics[f'total_net_amount_{window}d'] = None
                metrics[f'avg_daily_net_{window}d'] = None
        # 累计成交量
        for window in [5, 10]:
            if len(self.volume_series) >= window:
                metrics[f'total_volume_{window}d'] = sum(self.volume_series[-window:]) / 100  # 转换为万手
            else:
                metrics[f'total_volume_{window}d'] = None
        return metrics
    
    # ==================== 2. 相对强度指标计算 ====================
    def calculate_relative_metrics(self) -> Dict[str, float]:
        """计算相对强度指标"""
        metrics = {}
        current_net = float(self.context.current_flow_data.get('net_mf_amount', 0) or 0)
        # 当日净流入占比
        if self.context.daily_basic_data:
            daily_amount = float(self.context.daily_basic_data.get('amount', 0) or 0)
            if daily_amount > 0:
                net_ratio = (current_net / daily_amount) * 100
            else:
                net_ratio = 0
        else:
            # 使用历史平均成交额估算
            avg_amount = np.mean([abs(n) for n in self.net_amount_series[-20:]]) if len(self.net_amount_series) >= 20 else 0
            net_ratio = (current_net / avg_amount * 100) if avg_amount > 0 else 0
        metrics['net_amount_ratio'] = net_ratio
        # 5日、10日均净流入占比
        for window in [5, 10]:
            if len(self.net_amount_series) >= window:
                recent_nets = self.net_amount_series[-window:]
                recent_ratios = []
                for i, net in enumerate(recent_nets):
                    if self.context.historical_flow_data[-window+i].get('daily_amount'):
                        ratio = (net / self.context.historical_flow_data[-window+i]['daily_amount']) * 100
                        recent_ratios.append(ratio)
                if recent_ratios:
                    avg_ratio = np.mean(recent_ratios)
                    metrics[f'net_amount_ratio_ma{window}'] = avg_ratio
                else:
                    metrics[f'net_amount_ratio_ma{window}'] = None
            else:
                metrics[f'net_amount_ratio_ma{window}'] = None
        # 资金流入强度得分（0-100）
        intensity_score = self._calculate_flow_intensity(current_net, net_ratio)
        metrics['flow_intensity'] = intensity_score
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
        """计算建仓模式得分"""
        if len(recent_nets) < 5:
            return 0
        # 连续净流入天数
        positive_days = sum(1 for net in recent_nets if net > 0)
        continuity_score = (positive_days / len(recent_nets)) * 100
        # 流入稳定性（波动越小越好）
        if len(recent_nets) >= 3:
            volatility = np.std(recent_nets[-3:]) / (abs(np.mean(recent_nets[-3:])) + 1e-6)
            stability_score = max(0, 100 - volatility * 100)
        else:
            stability_score = 50
        # 流出比例（建仓时流出应很少）
        total_inflow = sum(max(0, net) for net in recent_nets)
        total_outflow = sum(abs(min(0, net)) for net in recent_nets)
        outflow_ratio = total_outflow / (total_inflow + total_outflow + 1e-6)
        outflow_score = max(0, 100 - outflow_ratio * 200)  # 流出越少得分越高
        # 综合得分
        return continuity_score * 0.4 + stability_score * 0.3 + outflow_score * 0.3
    
    def _calculate_pushing_score(self, recent_nets: List[float]) -> float:
        """计算拉升模式得分"""
        if len(recent_nets) < 3:
            return 0
        # 最近3天净流入加速度
        recent_3 = recent_nets[-3:]
        if len(recent_3) == 3:
            acceleration = (recent_3[2] - recent_3[1]) - (recent_3[1] - recent_3[0])
            accel_score = min(100, max(0, 50 + acceleration / 1000 * 10))
        else:
            accel_score = 50
        # 流入强度
        avg_inflow = np.mean([max(0, net) for net in recent_nets[-5:] if net > 0]) if any(n > 0 for n in recent_nets[-5:]) else 0
        intensity_score = min(100, avg_inflow / 5000 * 100)
        # 流出可控性
        max_outflow = max([abs(min(0, net)) for net in recent_nets[-5:] if net < 0], default=0)
        if avg_inflow > 0:
            outflow_control = max(0, 100 - (max_outflow / avg_inflow) * 100)
        else:
            outflow_control = 100
        return accel_score * 0.4 + intensity_score * 0.3 + outflow_control * 0.3
    
    def _calculate_distribution_score(self, recent_nets: List[float]) -> float:
        """计算派发模式得分"""
        if len(recent_nets) < 5:
            return 0
        # 净流出天数占比
        negative_days = sum(1 for net in recent_nets if net < 0)
        negative_ratio_score = (negative_days / len(recent_nets)) * 100
        # 流出绝对量
        avg_outflow = np.mean([abs(min(0, net)) for net in recent_nets if net < 0]) if any(n < 0 for n in recent_nets) else 0
        outflow_amount_score = min(100, avg_outflow / 3000 * 100)
        # 流出持续性（连续流出天数）
        max_consecutive = 0
        current = 0
        for net in recent_nets:
            if net < 0:
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                current = 0
        continuity_score = min(100, max_consecutive * 20)
        return negative_ratio_score * 0.4 + outflow_amount_score * 0.3 + continuity_score * 0.3
    
    def _calculate_shakeout_score(self, recent_nets: List[float]) -> float:
        """计算洗盘模式得分"""
        if len(recent_nets) < 5:
            return 0
        # 流出但绝对量不大
        avg_net = np.mean(recent_nets)
        avg_abs = np.mean([abs(net) for net in recent_nets])
        if avg_net < 0 and avg_abs < 2000:  # 净流出但量不大
            amount_score = 80
        else:
            amount_score = 20
        # 波动性（洗盘时可能有较大波动）
        volatility = np.std(recent_nets) / (abs(avg_net) + 1e-6)
        volatility_score = min(100, volatility * 50)
        # 流出流入交替（洗盘特征）
        sign_changes = sum(1 for i in range(1, len(recent_nets)) 
                          if (recent_nets[i-1] * recent_nets[i]) < 0)
        change_score = min(100, sign_changes / len(recent_nets) * 200)
        return amount_score * 0.5 + volatility_score * 0.3 + change_score * 0.2
    
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
        """计算流出质量评分（解决小量派发误判问题）"""
        if current_net >= 0:
            return 100  # 净流入，流出质量为满分
        # 获取历史流出数据
        historical_outflows = [abs(min(0, net)) for net in self.net_amount_series[-20:] if net < 0]
        if not historical_outflows:
            return 50  # 没有历史流出数据
        # 当前流出量在历史分布中的位置
        current_outflow = abs(current_net)
        percentile = stats.percentileofscore(historical_outflows, current_outflow)
        # 质量评分逻辑：流出量越小、越不异常，质量越高
        if current_outflow < 100:  # 极小流出，可能是噪音
            return 90
        elif percentile < 30:  # 低于30%分位，正常小流出
            return 80
        elif percentile < 70:  # 中等流出
            return 60
        else:  # 大额流出
            return 30
    
    def _calculate_inflow_persistence(self) -> int:
        """计算连续净流入天数"""
        count = 0
        for net in reversed(self.net_amount_series):
            if net > 0:
                count += 1
            else:
                break
        return count
    
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
        """计算分档资金一致性"""
        # 这里需要具体的分档资金数据
        # 暂时使用净流入符号一致性作为代理
        if len(self.net_amount_series) < 3:
            return 50
        recent_signs = [1 if net > 0 else -1 if net < 0 else 0 for net in self.net_amount_series[-3:]]
        consistency = sum(1 for i in range(1, len(recent_signs)) 
                         if recent_signs[i] == recent_signs[i-1]) / (len(recent_signs) - 1) * 100
        return consistency
    
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
        """计算多周期资金共振指标"""
        sync_metrics = {}
        if len(self.net_amount_series) < 20:
            for key in ['daily_weekly_sync', 'daily_monthly_sync', 
                       'short_mid_sync', 'mid_long_sync']:
                sync_metrics[key] = None
            return sync_metrics
        # 日线数据（最近5天）
        daily_data = self.net_amount_series[-5:]
        # 周线数据（5日移动平均）
        weekly_data = pd.Series(self.net_amount_series[-20:]).rolling(5).mean().dropna().tolist()
        # 月线数据（20日移动平均）
        monthly_data = pd.Series(self.net_amount_series[-60:]).rolling(20).mean().dropna().tolist()
        # 计算同步度
        if len(daily_data) >= 3 and len(weekly_data) >= 3:
            sync_metrics['daily_weekly_sync'] = self._calculate_sync_score(daily_data[-3:], weekly_data[-3:])
        if len(daily_data) >= 3 and len(monthly_data) >= 3:
            sync_metrics['daily_monthly_sync'] = self._calculate_sync_score(daily_data[-3:], monthly_data[-3:])
        # 短中期（5日 vs 20日）
        short_term = pd.Series(self.net_amount_series[-20:]).rolling(5).mean().dropna().tolist()[-3:]
        mid_term = pd.Series(self.net_amount_series[-40:]).rolling(20).mean().dropna().tolist()[-3:]
        if len(short_term) >= 3 and len(mid_term) >= 3:
            sync_metrics['short_mid_sync'] = self._calculate_sync_score(short_term, mid_term)
        # 中长期（20日 vs 60日）
        mid_term_long = pd.Series(self.net_amount_series[-60:]).rolling(20).mean().dropna().tolist()[-3:]
        long_term = pd.Series(self.net_amount_series[-120:]).rolling(60).mean().dropna().tolist()[-3:] if len(self.net_amount_series) >= 120 else []
        if len(mid_term_long) >= 3 and len(long_term) >= 3:
            sync_metrics['mid_long_sync'] = self._calculate_sync_score(mid_term_long, long_term)
        return sync_metrics
    
    def _calculate_sync_score(self, series1: List[float], series2: List[float]) -> float:
        """计算两个序列的同步度得分"""
        if len(series1) != len(series2) or len(series1) < 2:
            return 50
        # 计算相关性
        correlation = np.corrcoef(series1, series2)[0, 1]
        # 计算方向一致性
        directions1 = [1 if series1[i] > series1[i-1] else -1 for i in range(1, len(series1))]
        directions2 = [1 if series2[i] > series2[i-1] else -1 for i in range(1, len(series2))]
        match_count = sum(1 for d1, d2 in zip(directions1, directions2) if d1 == d2)
        direction_score = match_count / len(directions1) * 100 if directions1 else 50
        # 综合得分
        sync_score = (correlation * 100 * 0.6 + direction_score * 0.4) if not np.isnan(correlation) else direction_score
        return max(0, min(100, sync_score))
    
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
        """计算趋势强度"""
        if len(self.net_amount_series) < 10:
            return 0, 0
        # 使用线性回归判断趋势
        x = np.arange(len(self.net_amount_series[-10:]))
        y = np.array(self.net_amount_series[-10:])
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        # 上升趋势强度
        if slope > 0:
            uptrend_strength = min(100, r_value**2 * 100)
            downtrend_strength = 0
        else:
            uptrend_strength = 0
            downtrend_strength = min(100, r_value**2 * 100)
        return uptrend_strength, downtrend_strength
    
    # ==================== 7. 量价背离指标 ====================
    def calculate_divergence_metrics(self) -> Dict[str, Any]:
        """计算量价背离指标"""
        divergence = {}
        # 需要价格数据
        if not self.context.daily_basic_data or 'close' not in self.context.daily_basic_data:
            divergence['price_flow_divergence'] = None
            divergence['divergence_type'] = 'NONE'
            divergence['divergence_strength'] = 0
            return divergence
        current_price = float(self.context.daily_basic_data.get('close', 0) or 0)
        current_net = float(self.context.current_flow_data.get('net_mf_amount', 0) or 0)
        # 获取历史价格和资金流数据
        price_history = []
        flow_history = []
        for data in self.context.historical_flow_data[-10:]:
            if 'close' in data and 'net_mf_amount' in data:
                price_history.append(float(data['close']))
                flow_history.append(float(data['net_mf_amount']))
        if len(price_history) < 3 or len(flow_history) < 3:
            divergence['price_flow_divergence'] = None
            divergence['divergence_type'] = 'NONE'
            divergence['divergence_strength'] = 0
            return divergence
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
        """计算序列趋势方向（-1到1）"""
        if len(series) < 2:
            return 0
        # 简单趋势判断
        changes = [series[i] - series[i-1] for i in range(1, len(series))]
        avg_change = np.mean(changes)
        std_change = np.std(changes) + 1e-6
        # 标准化趋势值
        trend = avg_change / std_change
        return np.clip(trend, -1, 1)
    
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
        """计算统计特征指标"""
        stats_metrics = {}
        if len(self.net_amount_series) < 10:
            for key in ['flow_zscore', 'flow_percentile', 'flow_volatility_10d', 'flow_volatility_20d']:
                stats_metrics[key] = None
            return stats_metrics
        current_net = float(self.context.current_flow_data.get('net_mf_amount', 0) or 0)
        # Z分数
        mean_val = np.mean(self.net_amount_series[-20:])
        std_val = np.std(self.net_amount_series[-20:])
        stats_metrics['flow_zscore'] = (current_net - mean_val) / (std_val + 1e-6) if std_val > 0 else 0
        # 百分位
        stats_metrics['flow_percentile'] = stats.percentileofscore(self.net_amount_series[-20:], current_net)
        # 波动率
        if len(self.net_amount_series) >= 10:
            recent_10 = self.net_amount_series[-10:]
            stats_metrics['flow_volatility_10d'] = np.std(recent_10) / (abs(np.mean(recent_10)) + 1e-6)
        if len(self.net_amount_series) >= 20:
            recent_20 = self.net_amount_series[-20:]
            stats_metrics['flow_volatility_20d'] = np.std(recent_20) / (abs(np.mean(recent_20)) + 1e-6)
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
        """计算历史趋势延续概率"""
        if len(self.net_amount_series) < 10:
            return 50
        continuation_count = 0
        total_count = 0
        for i in range(1, len(self.net_amount_series) - 1):
            if self.net_amount_series[i] * self.net_amount_series[i-1] > 0:  # 同号
                continuation_count += 1
            total_count += 1
        if total_count > 0:
            probability = (continuation_count / total_count) * 100
        else:
            probability = 50
        return probability
    
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
        # 序列化存储
        metrics['flow_sequence_30d'] = json.dumps(sequence_data)
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
        metrics['calculation_metadata'] = json.dumps(metadata)
    
    def _create_feature_vector(self, metrics: Dict[str, Any]) -> str:
        """创建特征向量"""
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
        # 转换为numpy数组并序列化
        import numpy as np
        vector = np.array(key_features, dtype=np.float32)
        # Base64编码存储
        import base64
        import pickle
        vector_bytes = pickle.dumps(vector)
        return base64.b64encode(vector_bytes).decode('utf-8')