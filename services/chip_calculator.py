# services\chip_calculator.py
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional, List, Any
from datetime import datetime
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import linregress
import logging

logger = logging.getLogger(__name__)

class ChipFactorCalculator:
    """
    筹码因子计算器
    """
    @staticmethod
    def calculate_chip_entropy(price_percent_dict: Dict[float, float]) -> float:
        """
        计算筹码分布熵值 - 向量化优化版 v1.1
        公式: -∑(p_i * ln(p_i))
        修改思路：使用Numpy向量化替代Python循环，提升计算效率。
        """
        try:
            if not price_percent_dict:
                return 0.0
            # 转换为numpy数组
            percents = np.array(list(price_percent_dict.values()))
            total = np.sum(percents)
            if total <= 0:
                return 0.0
            # 归一化
            normalized_percents = percents / total
            # 过滤掉0值以避免log(0)
            valid_percents = normalized_percents[normalized_percents > 0]
            if len(valid_percents) == 0:
                return 0.0
            # 向量化计算熵值
            entropy = -np.sum(valid_percents * np.log(valid_percents))
            return float(entropy)
        except Exception as e:
            logger.error(f"计算筹码熵值失败: {e}")
            return 0.0

    @staticmethod
    def calculate_chip_skewness_kurtosis(price_percent_dict: Dict[float, float]) -> Tuple[float, float, float, float]:
        """
        计算筹码分布的均值、标准差、偏度和峰度 - Numpy优化版 v1.1
        修改思路：
        1. 转换为Numpy数组。
        2. 复用去均值后的差异数组，减少重复的幂运算。
        3. 增加鲁棒性检查。
        """
        try:
            if not price_percent_dict:
                return 0.0, 0.0, 0.0, 0.0
            # 转换为numpy数组
            prices = np.array(list(price_percent_dict.keys()))
            percents = np.array(list(price_percent_dict.values()))
            # 归一化权重
            total = np.sum(percents)
            if total <= 0:
                return 0.0, 0.0, 0.0, 0.0
            weights = percents / total
            # 计算加权均值
            mean = np.average(prices, weights=weights)
            # 计算去均值差异
            diff = prices - mean
            # 计算加权方差 (二阶矩)
            variance = np.average(diff * diff, weights=weights)
            std = np.sqrt(variance) if variance > 0 else 0.0
            if std > 0:
                # 标准化差异
                z_scores = diff / std
                # 预计算平方
                z_sq = z_scores * z_scores
                # 计算偏度 (三阶矩)
                skewness = np.average(z_sq * z_scores, weights=weights)
                # 计算峰度 (四阶矩) - Fisher定义 (减3)
                kurtosis = np.average(z_sq * z_sq, weights=weights) - 3
            else:
                skewness = 0.0
                kurtosis = 0.0
            return float(mean), float(std), float(skewness), float(kurtosis)
        except Exception as e:
            logger.error(f"计算筹码分布统计量失败: {e}")
            return 0.0, 0.0, 0.0, 0.0

    @staticmethod
    def calculate_profit_ratio(chip_data: pd.DataFrame, current_price: float) -> float:
        """
        计算获利比例 - Numpy优化版 v1.1
        修改思路：提取底层numpy数组进行布尔索引，避免pandas overhead。
        """
        try:
            if chip_data.empty or current_price <= 0:
                return 0.0
            # 提取numpy数组
            prices = chip_data['price'].values
            percents = chip_data['percent'].values
            # Numpy布尔索引
            mask = prices <= current_price
            profit_percent = np.sum(percents[mask])
            total_percent = np.sum(percents)
            if total_percent <= 0:
                return 0.0
            return float(profit_percent / total_percent)
        except Exception as e:
            logger.error(f"计算获利比例失败: {e}")
            return 0.0

    @staticmethod
    def calculate_all_factors(
        chip_perf_data: Dict,  # cyq_perf数据
        chip_dist_data: pd.DataFrame,  # cyq_chips数据
        daily_basic_data: Dict,  # 日基本面数据
        daily_kline_data: Dict  # 日K线数据
    ) -> Dict:
        """
        计算所有筹码因子
        Args:
            chip_perf_data: 筹码性能数据字典
            chip_dist_data: 筹码分布DataFrame
            daily_basic_data: 日基本面数据字典
            daily_kline_data: 日K线数据字典
        Returns:
            Dict: 因子字典
        """
        factors = {}
        try:
            # 基础数据
            close = daily_basic_data.get('close') or daily_kline_data.get('close')
            if close is None or close <= 0:
                return factors
            # ========== 基础因子 ==========
            factors['close'] = close
            factors['weight_avg_cost'] = chip_perf_data.get('weight_avg')
            # ========== 成本结构因子 ==========
            weight_avg = chip_perf_data.get('weight_avg')
            if weight_avg and weight_avg > 0:
                factors['price_to_weight_avg_ratio'] = (close - weight_avg) / weight_avg
            # 筹码集中度因子
            his_high = chip_perf_data.get('his_high', 0)
            his_low = chip_perf_data.get('his_low', 0)
            cost_85pct = chip_perf_data.get('cost_85pct', 0)
            cost_15pct = chip_perf_data.get('cost_15pct', 0)
            if his_high > his_low:
                # 筹码集中度
                factors['chip_concentration_ratio'] = (cost_85pct - cost_15pct) / (his_high - his_low)
                # 筹码稳定性
                factors['chip_stability'] = 1 - (cost_85pct - cost_15pct) / (his_high - his_low)
            # 股价在筹码分布中的位置
            cost_5pct = chip_perf_data.get('cost_5pct', 0)
            cost_95pct = chip_perf_data.get('cost_95pct', 0)
            if cost_95pct > cost_5pct:
                factors['price_percentile_position'] = (close - cost_5pct) / (cost_95pct - cost_5pct)
            # ========== 获利压力因子 ==========
            cost_50pct = chip_perf_data.get('cost_50pct', 0)
            winner_rate = chip_perf_data.get('winner_rate', 0)
            if cost_85pct > cost_15pct:
                factors['profit_pressure'] = (close - cost_50pct) / (cost_85pct - cost_15pct) * winner_rate
            # 计算获利比例
            if not chip_dist_data.empty:
                factors['profit_ratio'] = ChipFactorCalculator.calculate_profit_ratio(
                    chip_dist_data, close
                )
            # ========== 筹码分布形态因子 ==========
            if not chip_dist_data.empty:
                # 转换为价格-占比字典
                price_percent_dict = dict(zip(
                    chip_dist_data['price'], 
                    chip_dist_data['percent']
                ))
                # 计算熵值
                factors['chip_entropy'] = ChipFactorCalculator.calculate_chip_entropy(
                    price_percent_dict
                )
                # 计算统计量
                mean, std, skewness, kurtosis = ChipFactorCalculator.calculate_chip_skewness_kurtosis(
                    price_percent_dict
                )
                factors['chip_mean'] = mean
                factors['chip_std'] = std
                factors['chip_skewness'] = skewness
                factors['chip_kurtosis'] = kurtosis
            # ========== 博弈状态因子 ==========
            factors['winner_rate'] = winner_rate
            # 胜率价格分位联动
            if cost_95pct > cost_5pct and 'price_percentile_position' in factors:
                factors['win_rate_price_position'] = (
                    winner_rate * factors['price_percentile_position']
                )
            # ========== 关键分位成本 ==========
            factors['cost_5pct'] = cost_5pct
            factors['cost_15pct'] = cost_15pct
            factors['cost_50pct'] = cost_50pct
            factors['cost_85pct'] = cost_85pct
            factors['cost_95pct'] = cost_95pct
            factors['his_low'] = his_low
            factors['his_high'] = his_high
            # ========== 量价验证因子 ==========
            factors['turnover_rate'] = daily_basic_data.get('turnover_rate')
            factors['volume_ratio'] = daily_basic_data.get('volume_ratio')
            factors['calc_status'] = 'success'
        except Exception as e:
            logger.error(f"计算筹码因子失败: {e}", exc_info=True)
            factors['calc_status'] = 'failed'
            factors['error_message'] = str(e)
        return factors

    @staticmethod
    def calculate_moving_averages(prices: pd.Series, windows: List[int] = [5, 21, 34, 55]) -> Dict:
        """
        计算移动平均线 - 算法优化版 v1.1
        修改思路：仅计算最后窗口长度的数据均值，避免全序列rolling计算，复杂度从O(N)降为O(1)。
        """
        try:
            ma_values = {}
            n = len(prices)
            for window in windows:
                if n >= window:
                    # 仅取最后window个数据计算均值
                    ma_values[f'ma{window}'] = prices.iloc[-window:].mean()
                else:
                    ma_values[f'ma{window}'] = None
            return ma_values
        except Exception as e:
            logger.error(f"计算移动平均线失败: {e}")
            return {}

    @staticmethod
    def detect_peaks(chip_dist: pd.DataFrame, min_height: float = 1.0, min_distance: float = 0.02) -> Tuple[List[float], List[float]]:
        """
        检测筹码峰 - Numpy索引优化版 v1.1
        修改思路：
        1. 提取底层Numpy数组。
        2. 使用Numpy数组索引替代Pandas iloc索引，提升获取峰值数据的速度。
        """
        try:
            if chip_dist.empty or len(chip_dist) < 3:
                return [], []
            # 提取Numpy数组
            prices = chip_dist['price'].values
            percents = chip_dist['percent'].values
            # 平滑处理筹码分布
            window_length = min(7, len(percents))
            if window_length % 2 == 0: window_length -= 1 # savgol要求窗口为奇数
            if window_length < 3:
                percent_smooth = percents
            else:
                percent_smooth = savgol_filter(percents, window_length=window_length, polyorder=2)
            # 计算最小距离（价格点数量）
            price_range = np.max(prices) - np.min(prices)
            min_dist_points = int(len(prices) * min_distance)
            # 寻找峰值
            peaks_idx, _ = find_peaks(
                percent_smooth, 
                height=min_height,
                distance=max(2, min_dist_points)
            )
            if len(peaks_idx) == 0:
                return [], []
            # 使用Numpy索引直接获取数据，避免iloc开销
            peak_prices = prices[peaks_idx].tolist()
            peak_heights = percents[peaks_idx].tolist()
            return peak_prices, peak_heights
        except Exception as e:
            logger.error(f"检测筹码峰失败: {e}")
            return [], []

    @staticmethod
    def analyze_peak_pattern(peak_prices: List[float], peak_heights: List[float], price_range: Tuple[float, float]) -> Dict:
        """
        分析多峰形态
        Args:
            peak_prices: 峰价格列表
            peak_heights: 峰高度列表
            price_range: 价格区间(min_price, max_price)
        Returns:
            Dict: 多峰形态分析结果
        """
        analysis = {
            'peak_count': 0,
            'main_peak_position': 0,
            'peak_distance_ratio': 0.0,
            'peak_concentration': 0.0,
            'is_double_peak': False,
            'is_multi_peak': False
        }
        try:
            if not peak_prices or len(peak_prices) < 1:
                return analysis
            min_price, max_price = price_range
            price_span = max_price - min_price
            analysis['peak_count'] = len(peak_prices)
            # 判断主峰位置
            if peak_prices:
                main_peak_idx = np.argmax(peak_heights)
                main_peak_price = peak_prices[main_peak_idx]
                # 将价格区间分为低、中、高三部分
                price_thirds = min_price + price_span * np.array([1/3, 2/3])
                if main_peak_price < price_thirds[0]:
                    analysis['main_peak_position'] = 0  # 低位
                elif main_peak_price < price_thirds[1]:
                    analysis['main_peak_position'] = 1  # 中位
                else:
                    analysis['main_peak_position'] = 2  # 高位
            # 峰间距离比率
            if len(peak_prices) >= 2:
                peak_span = max(peak_prices) - min(peak_prices)
                analysis['peak_distance_ratio'] = peak_span / price_span if price_span > 0 else 0.0
            # 峰间集中度（前两大峰占比之和）
            # 修改：原逻辑 sum(top2)/sum(all) 在双峰时恒为1。
            # 现改为直接计算前两大峰的高度之和，反映峰的绝对强度。
            if peak_heights:
                sorted_heights = sorted(peak_heights, reverse=True)
                analysis['peak_concentration'] = sum(sorted_heights[:2])
            else:
                analysis['peak_concentration'] = 0.0
            # 双峰/多峰判断
            if analysis['peak_count'] == 2:
                analysis['is_double_peak'] = True
            elif analysis['peak_count'] > 2:
                analysis['is_multi_peak'] = True
            return analysis
        except Exception as e:
            logger.error(f"分析多峰形态失败: {e}")
            return analysis

    @staticmethod
    def calculate_chip_flow(current_chip: pd.DataFrame, prev_chip: pd.DataFrame, current_price: float) -> Tuple[int, float]:
        """
        计算筹码流动方向和强度 - Numpy优化版 v1.1
        修改思路：
        1. 移除内部函数定义。
        2. 使用Numpy数组和向量化计算重心，避免Pandas Series操作。
        """
        try:
            if prev_chip.empty or current_chip.empty:
                return 0, 0.0
            # 提取Numpy数组
            curr_prices = current_chip['price'].values
            curr_percents = current_chip['percent'].values
            prev_prices = prev_chip['price'].values
            prev_percents = prev_chip['percent'].values
            # 计算重心 (加权平均价)
            def get_center(prices, percents):
                total = np.sum(percents)
                if total <= 0: return 0.0
                return np.sum(prices * percents) / total
            prev_center = get_center(prev_prices, prev_percents)
            curr_center = get_center(curr_prices, curr_percents)
            # 计算筹码流动强度
            if prev_center <= 1e-6:
                flow_intensity = 0.0
            else:
                flow_intensity = abs(curr_center - prev_center) / prev_center
            # 判断流动方向
            if flow_intensity < 0.001:  # 阈值
                flow_direction = 0  # 横盘整理
            elif curr_center > prev_center:
                flow_direction = 1  # 向上流动
            else:
                flow_direction = -1  # 向下流动
            return flow_direction, float(flow_intensity)
        except Exception as e:
            logger.error(f"计算筹码流动失败: {e}")
            return 0, 0.0

    @staticmethod
    def calculate_convergence_divergence(chip_dist: pd.DataFrame, cost_50pct: float, price_range: Tuple[float, float]) -> Tuple[float, float]:
        """
        计算筹码聚集度和发散度 - Numpy优化版 v1.1
        修改思路：使用Numpy数组操作替代Pandas过滤。
        """
        try:
            min_price, max_price = price_range
            price_span = max_price - min_price
            if price_span <= 0 or chip_dist.empty:
                return 0.0, 0.0
            # 提取numpy数组
            prices = chip_dist['price'].values
            percents = chip_dist['percent'].values
            total_percent = np.sum(percents)
            # 聚集度：50分位附近筹码占比
            if cost_50pct > 0 and total_percent > 0:
                convergence_range = cost_50pct * 0.05  # 5%区间
                # Numpy布尔索引
                mask = (prices >= cost_50pct - convergence_range) & \
                       (prices <= cost_50pct + convergence_range)
                convergence_ratio = np.sum(percents[mask]) / total_percent
            else:
                convergence_ratio = 0.0
            # 发散度：整个价格区间的筹码分布宽度
            divergence_ratio = price_span / (max_price + min_price) * 0.5
            return float(convergence_ratio), float(divergence_ratio)
        except Exception as e:
            logger.error(f"计算聚集发散度失败: {e}")
            return 0.0, 0.0

    @staticmethod
    def calculate_trend_reversal_scores(chip_factors: Dict,ma_values: Dict,price_data: Dict,volume_data: Dict) -> Tuple[float, float]:
        """
        计算趋势确认和反转预警得分
        Args:
            chip_factors: 基础筹码因子
            ma_values: 移动平均线值
            price_data: 价格数据
            volume_data: 成交量数据
        Returns:
            Tuple: (趋势得分, 反转得分)
        """
        trend_score = 0.0
        reversal_score = 0.0
        try:
            # 探针：检查输入数据的完整性，便于排查NaN/None源头
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Trend Calc Input Probe - MA: {ma_values}, Price: {price_data}, Vol: {volume_data}")
            # 数据清洗与安全获取（防止 NoneType 参与比较）
            def safe_get(d: Dict, k: str, default: float = 0.0) -> float:
                val = d.get(k)
                return float(val) if val is not None else default
            close = safe_get(price_data, 'close')
            volume = safe_get(volume_data, 'volume')
            turnover_rate = safe_get(volume_data, 'turnover_rate')
            # ========== 趋势得分计算 ==========
            trend_factors = []
            # 1. 均线排列得分 (修复：增加对value非空的检查)
            ma_keys = ['ma5', 'ma21', 'ma34', 'ma55']
            # 确保所有需要的均线键存在且值不为None
            if all(ma_values.get(k) is not None for k in ma_keys):
                ma5 = ma_values['ma5']
                ma21 = ma_values['ma21']
                ma34 = ma_values['ma34']
                ma55 = ma_values['ma55']
                if ma5 > ma21 > ma34 > ma55:
                    trend_factors.append(1.0)  # 强势多头
                elif ma5 < ma21 < ma34 < ma55:
                    trend_factors.append(-1.0)  # 强势空头
                else:
                    trend_factors.append(0.0)
            # 2. 股价相对筹码成本位置
            weight_avg = safe_get(chip_factors, 'weight_avg_cost')
            if weight_avg > 0:
                price_to_chip = (close - weight_avg) / weight_avg
                if price_to_chip > 0.15:
                    trend_factors.append(1.0)
                elif price_to_chip < -0.15:
                    trend_factors.append(-1.0)
                else:
                    trend_factors.append(0.0)
            # 3. 筹码集中度趋势
            chip_concentration = safe_get(chip_factors, 'chip_concentration_ratio')
            if chip_concentration < 0.3:  # 高度集中
                trend_factors.append(1.0)
            elif chip_concentration > 0.7:  # 高度分散
                trend_factors.append(-0.5)
            # 4. 量价配合
            if turnover_rate > 0:
                price_change = safe_get(price_data, 'pct_change')
                if price_change > 0.01 and turnover_rate > 0.05:
                    trend_factors.append(1.0)
                elif price_change < -0.01 and turnover_rate > 0.05:
                    trend_factors.append(-1.0)
            # 趋势得分 = 因子加权平均
            if trend_factors:
                trend_score = float(np.mean(trend_factors))
            # ========== 反转得分计算 ==========
            reversal_factors = []
            # 1. 筹码峰形态反转信号
            is_multi_peak = chip_factors.get('is_multi_peak') is True  # 显式布尔检查
            is_double_peak = chip_factors.get('is_double_peak') is True
            if is_multi_peak:
                reversal_factors.append(0.8)
            elif is_double_peak:
                price_position = safe_get(chip_factors, 'price_percentile_position', 0.5)
                if price_position > 0.8:  # 高位双峰
                    reversal_factors.append(0.7)
                elif price_position < 0.2:  # 低位双峰
                    reversal_factors.append(-0.7)
            # 2. 筹码发散度反转信号
            chip_divergence = safe_get(chip_factors, 'chip_divergence_ratio')
            if chip_divergence > 0.6:  # 高度发散
                reversal_factors.append(0.6)
            # 3. 价格与筹码背离
            winner_rate = safe_get(chip_factors, 'winner_rate')
            if weight_avg > 0:  # 确保分母/基准有效
                if close > weight_avg and winner_rate < 0.5:
                    reversal_factors.append(0.5)
                elif close < weight_avg and winner_rate > 0.7:
                    reversal_factors.append(-0.5)
            # 4. 极端分位突破
            cost_95pct = safe_get(chip_factors, 'cost_95pct')
            cost_5pct = safe_get(chip_factors, 'cost_5pct')
            if close > 0:
                price_to_95pct = (close - cost_95pct) / close if cost_95pct else 0.0
                price_to_5pct = (close - cost_5pct) / close if cost_5pct else 0.0
                if price_to_95pct > 0.03:  # 突破95分位
                    reversal_factors.append(0.4)
                elif price_to_5pct < -0.03:  # 跌破5分位
                    reversal_factors.append(-0.4)
            # 反转得分 = 因子加权平均
            if reversal_factors:
                reversal_score = float(np.mean(reversal_factors))
            return trend_score, reversal_score
        except Exception as e:
            # 详细的错误上下文日志
            logger.error(f"计算趋势反转得分失败: {e} | MA_Has_None: {any(v is None for v in ma_values.values())}")
            return 0.0, 0.0

    @staticmethod
    def determine_chip_structure(chip_factors: Dict, trend_score: float) -> str:
        """
        判断筹码结构状态
        Args:
            chip_factors: 筹码因子
            trend_score: 趋势得分
        Returns:
            str: 筹码结构状态
        """
        try:
            price_position = chip_factors.get('price_percentile_position', 0.5)
            chip_concentration = chip_factors.get('chip_concentration_ratio', 0.5)
            chip_flow_direction = chip_factors.get('chip_flow_direction', 0)
            # 1. 吸筹阶段特征：价格低位+筹码集中+向上流动
            if price_position < 0.3 and chip_concentration < 0.4 and chip_flow_direction == 1:
                return 'accumulation'
            # 2. 拉升阶段特征：趋势向上+筹码稳定
            elif trend_score > 0.5 and chip_concentration < 0.6:
                return 'lifting'
            # 3. 派发阶段特征：价格高位+筹码发散+向下流动
            elif price_position > 0.7 and chip_concentration > 0.6 and chip_flow_direction == -1:
                return 'distribution'
            # 4. 回落阶段特征：趋势向下
            elif trend_score < -0.5:
                return 'decline'
            # 5. 整理阶段
            else:
                return 'consolidation'
        except Exception as e:
            logger.error(f"判断筹码结构失败: {e}")
            return 'consolidation'

    @staticmethod
    def analyze_trend_with_chip(chip_data: Dict, ma_data: Dict, price: float) -> Dict:
        """
        筹码与MA结合的趋势分析
        核心逻辑：
        1. 多头趋势确认条件：
           - 股价 > MA5 > MA21 > MA34 > MA55（多头排列）
           - 筹码向上流动，重心上移
           - 低位筹码峰，高位筹码少
        2. 空头趋势确认条件：
           - 股价 < MA5 < MA21 < MA34 < MA55（空头排列）
           - 筹码向下流动，重心下移
           - 高位套牢筹码多
        3. 反转预警信号：
           - 价格创新高但筹码发散（顶背离）
           - 价格创新低但筹码集中（底背离）
           - 均线即将金叉/死叉时筹码异常流动
        """
        analysis = {}
        # 计算均线乖离率
        for ma_name, ma_value in ma_data.items():
            if ma_value and ma_value > 0:
                analysis[f'{ma_name}_bias'] = (price - ma_value) / ma_value * 100
        # 均线排列状态
        ma_values = [ma_data.get(f'ma{period}') for period in [5, 21, 34, 55]]
        if all(ma_values):
            # 多头排列得分
            if ma_values[0] > ma_values[1] > ma_values[2] > ma_values[3]:
                analysis['ma_arrangement'] = 'strong_bull'
            elif ma_values[0] < ma_values[1] < ma_values[2] < ma_values[3]:
                analysis['ma_arrangement'] = 'strong_bear'
            else:
                analysis['ma_arrangement'] = 'mixed'
        # 筹码成本均线与价格均线关系
        chip_cost = chip_data.get('weight_avg_cost')
        if chip_cost and ma_data.get('ma21'):
            analysis['chip_ma21_diff'] = chip_cost - ma_data['ma21']
            analysis['chip_ma21_ratio'] = chip_cost / ma_data['ma21'] - 1
        return analysis

    @staticmethod
    def identify_multi_peak_pattern(chip_dist: pd.DataFrame) -> Dict:
        """
        识别多峰形态并分类
        多峰类型：
        1. 双峰夹板形态（价格在两个峰值间震荡）
        2. 三峰接力形态（筹码在三个区域聚集）
        3. 多峰分散形态（筹码极度分散）
        判断标准：
        - 峰值数量 >= 2
        - 峰间距离 > 价格区间的20%
        - 每个峰值占比 > 5%
        - 峰谷占比 < 峰值的50%
        """
        analysis = {
            'pattern_type': 'single',  # single, double, triple, multi
            'peak_prices': [],
            'peak_ratios': [],
            'valley_prices': [],
            'pattern_strength': 0.0,
            'trading_implication': ''
        }
        peaks, heights = AdvancedChipCalculator.detect_peaks(chip_dist)
        analysis['peak_prices'] = peaks
        analysis['peak_ratios'] = heights
        if len(peaks) == 2:
            analysis['pattern_type'] = 'double'
            # 计算双峰夹板强度
            peak_distance = abs(peaks[1] - peaks[0])
            price_range = chip_dist['price'].max() - chip_dist['price'].min()
            analysis['pattern_strength'] = peak_distance / price_range
            # 交易含义判断
            if heights[0] > heights[1]:
                analysis['trading_implication'] = '主力成本在低位峰'
            else:
                analysis['trading_implication'] = '主力成本在高位峰'
        elif len(peaks) == 3:
            analysis['pattern_type'] = 'triple'
            analysis['trading_implication'] = '筹码三峰接力，关注突破方向'
        elif len(peaks) > 3:
            analysis['pattern_type'] = 'multi'
            # 多峰分散度
            analysis['pattern_strength'] = 1.0 - np.std(heights) / np.mean(heights)
            analysis['trading_implication'] = '筹码极度分散，变盘在即'
        return analysis

    @staticmethod
    def calculate_chip_migration(chip_current: pd.DataFrame, chip_previous: pd.DataFrame, window_days: int = 5) -> Dict:
        """
        计算筹码迁移情况 - Numpy优化版 v1.1
        修改思路：将DataFrame转换为Numpy数组后复用，减少重复的Pandas索引开销。
        """
        migration = {
            'convergence_change': 0.0,
            'divergence_change': 0.0,
            'center_speed': 0.0,
            'low_to_high_ratio': 0.0,
            'high_to_low_ratio': 0.0,
            'stability_ratio': 0.0
        }
        try:
            if chip_current.empty or chip_previous.empty:
                return migration
            # 预处理为Numpy数组
            curr_prices = chip_current['price'].values
            curr_percents = chip_current['percent'].values
            prev_prices = chip_previous['price'].values
            prev_percents = chip_previous['percent'].values
            # 计算筹码重心
            def np_chip_center(prices, percents):
                total = np.sum(percents)
                return np.sum(prices * percents) / total if total > 0 else 0.0
            current_center = np_chip_center(curr_prices, curr_percents)
            previous_center = np_chip_center(prev_prices, prev_percents)
            migration['center_speed'] = (current_center - previous_center) / window_days
            # 计算价格三分位
            price_min = min(np.min(curr_prices), np.min(prev_prices))
            price_max = max(np.max(curr_prices), np.max(prev_prices))
            price_range = price_max - price_min
            if price_range > 0:
                low_bound = price_min + price_range * 0.33
                high_bound = price_min + price_range * 0.
                # 辅助函数：计算区间占比
                def np_region_percent(prices, percents, lower, upper):
                    mask = (prices >= lower) & (prices <= upper)
                    return np.sum(percents[mask])
                # 前一日区域筹码
                prev_low = np_region_percent(prev_prices, prev_percents, price_min, low_bound)
                prev_mid = np_region_percent(prev_prices, prev_percents, low_bound, high_bound)
                prev_high = np_region_percent(prev_prices, prev_percents, high_bound, price_max)
                # 当前日区域筹码
                curr_low = np_region_percent(curr_prices, curr_percents, price_min, low_bound)
                curr_mid = np_region_percent(curr_prices, curr_percents, low_bound, high_bound)
                curr_high = np_region_percent(curr_prices, curr_percents, high_bound, price_max)
                total = prev_low + prev_mid + prev_high
                if total > 0:
                    migration['low_to_high_ratio'] = max(0, (prev_low - curr_low) / total)
                    migration['high_to_low_ratio'] = max(0, (prev_high - curr_high) / total)
                    migration['stability_ratio'] = min(curr_mid, prev_mid) / max(curr_mid, prev_mid) if max(curr_mid, prev_mid) > 0 else 0.0
            return migration
        except Exception as e:
            logger.error(f"计算筹码迁移失败: {e}")
            return migration

    @staticmethod
    def calculate_ma_arrangement(ma_values: Dict[str, float]) -> int:
        """
        计算均线排列状态
        Returns:
            int: 1=多头排列, -1=空头排列, 0=震荡
        """
        try:
            ma5 = ma_values.get('ma5')
            ma21 = ma_values.get('ma21')
            ma34 = ma_values.get('ma34')
            ma55 = ma_values.get('ma55')
            if not all([ma5, ma21, ma34, ma55]):
                return 0
            # 多头排列：MA5 > MA21 > MA34 > MA55
            if ma5 > ma21 > ma34 > ma55:
                return 1
            # 空头排列：MA5 < MA21 < MA34 < ma55
            elif ma5 < ma21 < ma34 < ma55:
                return -1
            else:
                return 0
        except Exception as e:
            logger.error(f"计算均线排列状态失败: {e}")
            return 0

    @staticmethod
    def calculate_peak_migration(chip_centers_5d: List[float]) -> float:
        """
        计算筹码峰迁移速度（5日） - 算法优化版 v1.1
        修改思路：使用 np.polyfit 替代 scipy.stats.linregress，避免计算不必要的统计量（如p-value），提升速度。
        """
        try:
            if len(chip_centers_5d) < 2:
                return 0.0
            y = np.array(chip_centers_5d)
            x = np.arange(len(y))
            # 去除NaN值
            mask = ~np.isnan(y)
            if np.sum(mask) < 2:
                return 0.0
            x_clean = x[mask]
            y_clean = y[mask]
            # 使用一次多项式拟合（线性回归）计算斜率
            # deg=1 返回 [slope, intercept]
            slope = np.polyfit(x_clean, y_clean, 1)[0]
            return float(slope)
        except Exception as e:
            logger.error(f"计算筹码峰迁移速度失败: {e}")
            return 0.0

    @staticmethod
    def calculate_chip_stability_change(chip_stability_5d: List[float]) -> float:
        """
        计算筹码稳定性变化（5日）
        Args:
            chip_stability_5d: 最近5日的筹码稳定性列表
        Returns:
            float: 稳定性变化率
        """
        try:
            if len(chip_stability_5d) < 2:
                return 0.0
            # 去除NaN值
            valid_stability = [s for s in chip_stability_5d if not np.isnan(s)]
            if len(valid_stability) < 2:
                return 0.0
            # 计算变化率
            current = valid_stability[-1]
            previous = valid_stability[0]
            if previous != 0:
                change_rate = (current - previous) / abs(previous)
            else:
                change_rate = 0.0
            return float(change_rate)
        except Exception as e:
            logger.error(f"计算筹码稳定性变化失败: {e}")
            return 0.0

    @staticmethod
    def calculate_volatility(prices: pd.Series, window: int = 20) -> float:
        """
        计算价格波动率 - 算法优化版 v1.1
        修改思路：仅计算最后窗口长度的数据标准差，避免全序列rolling计算。
        """
        try:
            if len(prices) < window:
                return 0.0
            # 仅取最后 window+1 个数据计算收益率，确保有 window 个收益率数据
            # pct_change 会导致第一个数据变为 NaN，所以需要多取一个
            subset = prices.iloc[-(window + 1):]
            returns = subset.pct_change().dropna()
            if len(returns) < window:
                # 如果数据不足（例如有停牌导致NaN），尝试取更多数据或直接计算现有数据的std
                if len(returns) < 2: return 0.0
                volatility = returns.std()
            else:
                volatility = returns.std()
            # 年化波动率（假设252个交易日）
            annualized_vol = volatility * np.sqrt(252)
            return float(annualized_vol)
        except Exception as e:
            logger.error(f"计算波动率失败: {e}")
            return 0.0

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
        """
        计算RSI指标 - 算法优化版 v1.1
        修改思路：仅使用尾部数据计算RSI，避免全序列计算。
        注意：标准RSI使用EMA平滑，需要较长历史数据。此处保持原逻辑（SMA），仅优化计算范围。
        """
        try:
            # 只需要最后 period + 1 个数据来计算最近的一个 RSI 值
            if len(prices) < period + 1:
                return 50.0
            # 取尾部数据
            subset = prices.iloc[-(period + 1):]
            delta = subset.diff().dropna()
            if delta.empty:
                return 50.0
            # 分离上涨和下跌
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            # 计算平均涨跌幅 (SMA)
            avg_gain = gain.mean()
            avg_loss = loss.mean()
            if avg_loss == 0:
                return 100.0
            if avg_gain == 0:
                return 0.0
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi)
        except Exception as e:
            logger.error(f"计算RSI失败: {e}")
            return 50.0

    @staticmethod
    def calculate_volatility_adjusted_concentration(chip_concentration: float, volatility_20d: float,
        market_volatility: float = 0.2  # 市场平均波动率，默认20%
    ) -> float:
        """
        计算波动率调整的筹码集中度
        逻辑：在高波动率市场，筹码集中度的参考价值降低
        Args:
            chip_concentration: 原始筹码集中度
            volatility_20d: 20日波动率
            market_volatility: 市场平均波动率
        Returns:
            float: 波动率调整后的集中度
        """
        try:
            if volatility_20d <= 0:
                return chip_concentration
            # 计算波动率调整因子
            # 波动率越高，调整因子越小，表示集中度信号强度减弱
            volatility_ratio = volatility_20d / market_volatility
            # 调整因子：波动率越高，调整因子越小
            # 使用sigmoid函数平滑调整
            adjustment_factor = 1.0 / (1.0 + np.exp(2 * (volatility_ratio - 1)))
            adjusted_concentration = chip_concentration * adjustment_factor
            return float(adjusted_concentration)
        except Exception as e:
            logger.error(f"计算波动率调整集中度失败: {e}")
            return chip_concentration

    @staticmethod
    def calculate_chip_rsi_divergence(rsi_value: float, chip_flow_direction: int,price_trend_5d: float,rsi_trend_5d: float) -> float:
        """
        计算筹码RSI背离度
        Args:
            rsi_value: 当前RSI值
            chip_flow_direction: 筹码流动方向
            price_trend_5d: 5日价格趋势（涨跌幅）
            rsi_trend_5d: 5日RSI趋势（变化值）
        Returns:
            float: 背离度（正值为顶背离，负值为底背离）
        """
        try:
            # RSI超买超卖判断
            is_overbought = rsi_value > 70
            is_oversold = rsi_value < 30
            # 判断背离条件
            divergence_score = 0.0
            # 顶背离条件：价格创新高但RSI下降，筹码向下流动
            if (is_overbought and 
                price_trend_5d > 0 and 
                rsi_trend_5d < 0 and 
                chip_flow_direction == -1):
                # 强烈的顶背离信号
                divergence_score = 1.0
            # 中度顶背离：RSI超买但筹码开始向下流动
            elif is_overbought and chip_flow_direction == -1:
                divergence_score = 0.6
            # 底背离条件：价格创新低但RSI上升，筹码向上流动
            elif (is_oversold and 
                  price_trend_5d < 0 and 
                  rsi_trend_5d > 0 and 
                  chip_flow_direction == 1):
                # 强烈的底背离信号
                divergence_score = -1.0
            # 中度底背离：RSI超卖但筹码开始向上流动
            elif is_oversold and chip_flow_direction == 1:
                divergence_score = -0.6
            # 轻度背离：RSI与筹码流动方向相反
            elif ((rsi_value > 50 and chip_flow_direction == -1) or
                  (rsi_value < 50 and chip_flow_direction == 1)):
                divergence_score = 0.3 if chip_flow_direction == -1 else -0.3
            return float(divergence_score)
        except Exception as e:
            logger.error(f"计算RSI筹码背离失败: {e}")
            return 0.0

    @staticmethod
    def calculate_price_trend(prices: pd.Series, days: int = 5) -> float:
        """
        计算价格趋势
        Args:
            prices: 价格序列
            days: 趋势天数
        Returns:
            float: 价格趋势（涨跌幅）
        """
        try:
            if len(prices) < days + 1:
                return 0.0
            # 计算days日涨跌幅
            start_price = prices.iloc[-days-1] if len(prices) > days else prices.iloc[0]
            end_price = prices.iloc[-1]
            if start_price > 0:
                trend = (end_price - start_price) / start_price
            else:
                trend = 0.0
            return float(trend)
        except Exception as e:
            logger.error(f"计算价格趋势失败: {e}")
            return 0.0

    @staticmethod
    def calculate_complete_factors(chip_perf_data: Dict,chip_dist_data: pd.DataFrame,daily_basic_data: Dict,daily_kline_data: Dict,prev_chip_dist_data: pd.DataFrame = None,historical_prices: pd.Series = None,historical_chip_factors: List[Dict] = None) -> Dict:
        """
        计算完整的筹码因子（包含时间序列因子）
        Args:
            chip_perf_data: 当日筹码性能数据
            chip_dist_data: 当日筹码分布数据
            daily_basic_data: 当日基本面数据
            daily_kline_data: 当日K线数据
            prev_chip_dist_data: 前一日筹码分布数据
            historical_prices: 历史价格序列（用于计算MA）
            historical_chip_factors: 历史筹码因子列表
        Returns:
            Dict: 完整的因子字典
        """
        factors = {}
        try:
            # 1. 计算基础因子
            base_factors = ChipFactorCalculator.calculate_all_factors(
                chip_perf_data, chip_dist_data, daily_basic_data, daily_kline_data
            )
            factors.update(base_factors)
            # 2. 计算均线相关因子
            if historical_prices is not None:
                # 计算移动平均线
                ma_values = ChipFactorCalculator.calculate_moving_averages(
                    historical_prices, [5, 21, 34, 55]
                )
                close = factors.get('close', 0)
                for ma_name, ma_value in ma_values.items():
                    if ma_value and ma_value > 0:
                        ratio_key = f'price_to_{ma_name}_ratio'
                        factors[ratio_key] = (close - ma_value) / ma_value * 100
                # 计算均线排列状态
                factors['ma_arrangement_status'] = ChipFactorCalculator.calculate_ma_arrangement(
                    ma_values
                )
                # 计算筹码成本均线与MA21差值
                weight_avg = factors.get('weight_avg_cost')
                if weight_avg and 'ma21' in ma_values and ma_values['ma21']:
                    factors['chip_cost_to_ma21_diff'] = weight_avg - ma_values['ma21']
            # 3. 计算多峰形态因子
            if not chip_dist_data.empty:
                peak_prices, peak_heights = ChipFactorCalculator.detect_peaks(chip_dist_data)
                price_range = (chip_dist_data['price'].min(), chip_dist_data['price'].max())
                peak_analysis = ChipFactorCalculator.analyze_peak_pattern(
                    peak_prices, peak_heights, price_range
                )
                factors.update({
                    'peak_count': peak_analysis['peak_count'],
                    'main_peak_position': peak_analysis['main_peak_position'],
                    'peak_distance_ratio': peak_analysis['peak_distance_ratio'],
                    'peak_concentration': peak_analysis['peak_concentration'],
                    'is_double_peak': peak_analysis['is_double_peak'],
                    'is_multi_peak': peak_analysis['is_multi_peak']
                })
            # 4. 计算筹码流动因子
            if prev_chip_dist_data is not None and not prev_chip_dist_data.empty:
                close = factors.get('close', 0)
                flow_direction, flow_intensity = ChipFactorCalculator.calculate_chip_flow(
                    chip_dist_data, prev_chip_dist_data, close
                )
                factors.update({
                    'chip_flow_direction': flow_direction,
                    'chip_flow_intensity': flow_intensity
                })
            # 5. 计算聚集度和发散度
            if not chip_dist_data.empty:
                cost_50pct = factors.get('cost_50pct', 0)
                price_range = (factors.get('his_low', 0), factors.get('his_high', 0))
                convergence, divergence = ChipFactorCalculator.calculate_convergence_divergence(
                    chip_dist_data, cost_50pct, price_range
                )
                factors.update({
                    'chip_convergence_ratio': convergence,
                    'chip_divergence_ratio': divergence
                })
            # 6. 计算趋势和反转得分
            ma_values_dict = {}
            if historical_prices is not None:
                ma_values_dict = ChipFactorCalculator.calculate_moving_averages(
                    historical_prices, [5, 21, 34, 55]
                )
            price_data = {
                'close': factors.get('close', 0),
                'pct_change': daily_kline_data.get('pct_change', 0)
            }
            volume_data = {
                'volume': daily_kline_data.get('vol', 0),
                'turnover_rate': factors.get('turnover_rate', 0)
            }
            trend_score, reversal_score = ChipFactorCalculator.calculate_trend_reversal_scores(
                factors, ma_values_dict, price_data, volume_data
            )
            factors.update({
                'trend_confirmation_score': trend_score,
                'reversal_warning_score': reversal_score
            })
            # 7. 计算筹码结构状态
            chip_structure = ChipFactorCalculator.determine_chip_structure(factors, trend_score)
            factors['chip_structure_state'] = chip_structure
            # 8. 计算时间序列因子（需要历史数据）
            if historical_chip_factors and len(historical_chip_factors) >= 5:
                # 提取最近5日的筹码重心和稳定性
                chip_centers_5d = []
                chip_stability_5d = []
                for f in historical_chip_factors[-5:]:
                    chip_mean = f.get('chip_mean')
                    chip_stability = f.get('chip_stability')
                    if chip_mean is not None:
                        chip_centers_5d.append(chip_mean)
                    if chip_stability is not None:
                        chip_stability_5d.append(chip_stability)
                # 计算迁移速度
                if len(chip_centers_5d) >= 2:
                    factors['peak_migration_speed_5d'] = ChipFactorCalculator.calculate_peak_migration(
                        chip_centers_5d
                    )
                # 计算稳定性变化
                if len(chip_stability_5d) >= 2:
                    factors['chip_stability_change_5d'] = ChipFactorCalculator.calculate_chip_stability_change(
                        chip_stability_5d
                    )
            # 9. 计算市场适应性因子（需要历史价格数据）
            if historical_prices is not None and len(historical_prices) >= 20:
                try:
                    # 计算波动率
                    volatility_20d = ChipFactorCalculator.calculate_volatility(
                        historical_prices, window=20
                    )
                    # 计算波动率调整的筹码集中度
                    chip_concentration = factors.get('chip_concentration_ratio', 0.5)
                    if chip_concentration > 0:
                        factors['volatility_adjusted_concentration'] = ChipFactorCalculator.calculate_volatility_adjusted_concentration(
                            chip_concentration, volatility_20d
                        )
                    # 计算RSI
                    rsi_14 = ChipFactorCalculator.calculate_rsi(historical_prices, period=14)
                    # 计算价格趋势和RSI趋势
                    price_trend_5d = ChipFactorCalculator.calculate_price_trend(
                        historical_prices, days=5
                    )
                    # 获取历史RSI值计算趋势
                    rsi_values = []
                    if historical_chip_factors and len(historical_chip_factors) >= 10:
                        # 这里需要历史RSI数据，暂时使用简化方法
                        # 实际应该从数据库获取历史RSI值
                        pass
                    # 计算筹码RSI背离（简化版）
                    chip_flow_direction = factors.get('chip_flow_direction', 0)
                    rsi_trend_5d = 0.0  # 简化为0，实际需要计算
                    factors['chip_rsi_divergence'] = ChipFactorCalculator.calculate_chip_rsi_divergence(
                        rsi_14, chip_flow_direction, price_trend_5d, rsi_trend_5d
                    )
                except Exception as e:
                    logger.warning(f"计算市场适应性因子失败: {e}")
                    # 设置默认值
                    factors['volatility_adjusted_concentration'] = factors.get('chip_concentration_ratio', 0.5)
                    factors['chip_rsi_divergence'] = 0.0
            # 10. 确保关键因子被计算：主力成本区间锁定度和高位筹码沉淀比例
            # 检查并计算 main_cost_range_ratio
            if 'main_cost_range_ratio' not in factors or factors['main_cost_range_ratio'] <= 0:
                cost_50pct = chip_perf_data.get('cost_50pct')
                if cost_50pct:
                    factors['main_cost_range_ratio'] = ChipFactorCalculator.calculate_main_cost_range_ratio(chip_dist_data, cost_50pct)
                else:
                    factors['main_cost_range_ratio'] = 0.5
                    print(f"⚠️ [calculate_complete_factors] cost_50pct缺失，main_cost_range_ratio使用默认值0.5")
            # 检查并计算 high_position_lock_ratio_90
            if 'high_position_lock_ratio_90' not in factors or factors['high_position_lock_ratio_90'] < 0:
                current_price = daily_kline_data.get('close', 0)
                if current_price > 0:
                    factors['high_position_lock_ratio_90'] = ChipFactorCalculator.calculate_high_position_lock_ratio_90(chip_dist_data, current_price)
                else:
                    factors['high_position_lock_ratio_90'] = 0.0
                    print(f"⚠️ [calculate_complete_factors] 当前价格无效，high_position_lock_ratio_90使用默认值0.0")
            # 如果上述其他因子没有计算出来，设置默认值
            if 'peak_migration_speed_5d' not in factors:
                factors['peak_migration_speed_5d'] = 0.0
            if 'chip_stability_change_5d' not in factors:
                factors['chip_stability_change_5d'] = 0.0
            if 'volatility_adjusted_concentration' not in factors:
                factors['volatility_adjusted_concentration'] = factors.get('chip_concentration_ratio', 0.5)
            if 'chip_rsi_divergence' not in factors:
                factors['chip_rsi_divergence'] = 0.0
            factors['calc_status'] = 'success'
        except Exception as e:
            logger.error(f"计算完整筹码因子失败: {e}", exc_info=True)
            factors['calc_status'] = 'failed'
            factors['error_message'] = str(e)
        return factors

    @staticmethod
    def calculate_layered_flow_factors(chip_dist_current: pd.DataFrame, chip_dist_previous: pd.DataFrame, cost_15pct: float, cost_85pct: float) -> Dict[str, float]:
        """
        计算分层筹码流动因子 - Numpy优化版 v1.1
        修改思路：使用Numpy数组操作替代Pandas过滤。
        """
        factors = {}
        try:
            if chip_dist_current.empty or chip_dist_previous.empty:
                return factors
            # 提取Numpy数组
            curr_prices = chip_dist_current['price'].values
            curr_percents = chip_dist_current['percent'].values
            prev_prices = chip_dist_previous['price'].values
            prev_percents = chip_dist_previous['percent'].values
            # 辅助函数：计算区间占比
            def np_get_zone_percent(prices, percents, lower_mask, upper_mask=None):
                if upper_mask is not None:
                    mask = lower_mask & upper_mask
                else:
                    mask = lower_mask
                return np.sum(percents[mask])
            # 当前分布
            low_current = np_get_zone_percent(curr_prices, curr_percents, curr_prices <= cost_15pct)
            middle_current = np_get_zone_percent(curr_prices, curr_percents, curr_prices > cost_15pct, curr_prices <= cost_85pct)
            high_current = np_get_zone_percent(curr_prices, curr_percents, curr_prices > cost_85pct)
            # 前期分布
            low_prev = np_get_zone_percent(prev_prices, prev_percents, prev_prices <= cost_15pct)
            middle_prev = np_get_zone_percent(prev_prices, prev_percents, prev_prices > cost_15pct, prev_prices <= cost_85pct)
            high_prev = np_get_zone_percent(prev_prices, prev_percents, prev_prices > cost_85pct)
            # 计算净变动
            factors['low_zone_chip_flow'] = float(low_current - low_prev)
            factors['middle_zone_chip_flow'] = float(middle_current - middle_prev)
            factors['high_zone_chip_flow'] = float(high_current - high_prev)
            # 计算主力控盘度
            abs_low = abs(factors['low_zone_chip_flow'])
            abs_mid = abs(factors['middle_zone_chip_flow'])
            abs_high = abs(factors['high_zone_chip_flow'])
            total_change = abs_low + abs_mid + abs_high
            if total_change > 0:
                max_change = max(abs_low, abs_mid, abs_high)
                factors['main_force_control_ratio'] = float(max_change / total_change)
            else:
                factors['main_force_control_ratio'] = 0.0
            return factors
        except Exception as e:
            logger.error(f"计算分层流动因子失败: {e}")
            return {}

    @staticmethod
    def calculate_main_cost_range_ratio(chip_dist_data: pd.DataFrame, cost_50pct: float) -> float:
        """
        计算主力成本区间锁定比例 - Numpy优化版 v1.1
        修改思路：使用Numpy数组操作替代Pandas过滤。
        """
        try:
            if chip_dist_data.empty or cost_50pct <= 0:
                return 0.0
            # 提取Numpy数组
            prices = chip_dist_data['price'].values
            percents = chip_dist_data['percent'].values
            # 调整为 ±5% 区间
            lower_bound = cost_50pct * 0.95
            upper_bound = cost_50pct * 1.05
            # Numpy布尔索引
            mask = (prices >= lower_bound) & (prices <= upper_bound)
            if not np.any(mask):
                return 0.0
            main_chip_sum = np.sum(percents[mask])
            total_chip_sum = np.sum(percents)
            if total_chip_sum <= 0:
                return 0.0
            ratio = main_chip_sum / total_chip_sum
            return float(min(max(ratio, 0.0), 1.0))
            
        except Exception as e:
            logger.error(f"计算主力成本区间锁定比例失败: {e}")
            return 0.0

    @staticmethod
    def calculate_high_position_lock_ratio_90(chip_dist_data: pd.DataFrame, current_price: float) -> float:
        """
        计算90%分位以上高位筹码沉淀比例 - Numpy优化版 v1.1
        修改思路：使用Numpy数组操作替代Pandas排序和累加。
        """
        try:
            if chip_dist_data.empty or current_price <= 0:
                return 0.0
            # 提取Numpy数组
            prices = chip_dist_data['price'].values
            percents = chip_dist_data['percent'].values
            # 方法1: 基于当前价格计算90%分位阈值
            price_threshold_1 = current_price * 1.0
            # 方法2: 基于筹码分布自身计算90%价格分位
            # 使用 argsort 获取排序索引
            sort_idx = np.argsort(prices)
            sorted_prices = prices[sort_idx]
            sorted_percents = percents[sort_idx]
            cum_pct = np.cumsum(sorted_percents)
            # 找到累计占比达到90%的位置
            # searchsorted 返回第一个满足条件的索引
            idx_90 = np.searchsorted(cum_pct, 0.9)
            if idx_90 < len(sorted_prices):
                price_threshold_2 = sorted_prices[idx_90]
            else:
                price_threshold_2 = sorted_prices[-1]
            # 使用两者中较高的作为阈值
            price_threshold = max(price_threshold_1, price_threshold_2)
            # 计算高于阈值的筹码占比
            high_mask = prices >= price_threshold
            if not np.any(high_mask):
                return 0.0
            high_chip_sum = np.sum(percents[high_mask])
            total_chip_sum = np.sum(percents)
            if total_chip_sum <= 0:
                return 0.0
            ratio = high_chip_sum / total_chip_sum
            return float(ratio)
        except Exception as e:
            logger.error(f"计算高位筹码沉淀比例失败: {e}")
            return 0.0

    # ========== Tick数据处理基础方法 ==========
    
    @staticmethod
    def preprocess_tick_data(tick_data: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        """
        预处理tick数据 - 修复KeyError索引问题 v1.2 & 修复Decimal类型兼容问题 v1.3
        Args:
            tick_data: 包含['trade_time', 'price', 'volume', 'type']列的DataFrame，或者是trade_time为索引的DF
        Returns:
            Tuple[pd.DataFrame, float]: (预处理后的DataFrame, 数据质量评分)
        """
        try:
            if tick_data.empty:
                return pd.DataFrame(), 0.0
            
            # 拷贝数据防止修改原始数据
            df = tick_data.copy()
            
            # 【修复1】核心逻辑：检测 trade_time 是否为索引，如果是则重置为列
            if 'trade_time' not in df.columns:
                if isinstance(df.index, pd.DatetimeIndex) or df.index.name == 'trade_time':
                    df.reset_index(inplace=True)
                    # 防止reset_index后列名变成'index'
                    if 'trade_time' not in df.columns and 'index' in df.columns:
                        df.rename(columns={'index': 'trade_time'}, inplace=True)
            
            # 再次安全检查
            if 'trade_time' not in df.columns:
                logger.error(f"预处理tick数据失败: 无法找到 'trade_time' 列，现有列: {df.columns.tolist()}")
                return pd.DataFrame(), 0.0

            # 【修复2】类型转换：将Decimal类型转换为float，避免后续Numpy计算报错
            # Explicitly convert numeric columns to float to handle Decimal types from Django
            numeric_cols = ['price', 'volume', 'amount', 'price_change']
            for col in numeric_cols:
                if col in df.columns:
                    try:
                        # 尝试直接转换
                        df[col] = df[col].astype(float)
                    except Exception:
                        # 如果失败（例如包含非数字字符），强制转换
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

            # 确保时间排序
            df = df.sort_values('trade_time')
            
            # 计算3秒时间窗口（基于3秒聚合数据特性）
            # 注意：pandas旧版本可能只支持 '3S'，新版本支持 '3s'，这里使用 '3s'
            try:
                df['time_window'] = df['trade_time'].dt.floor('3s')
            except ValueError:
                df['time_window'] = df['trade_time'].dt.floor('3S')
            
            # 数据完整性检查
            # 计算相邻tick的时间差
            time_diffs = df['trade_time'].diff().dt.total_seconds().fillna(0)
            
            # 统计间隔超过3.5秒的情况（视为数据缺失或不活跃）
            # 注意：对于非高频交易股票，间隔大不一定是数据质量差，但在计算日内分布时需要考虑连续性
            gaps_count = (time_diffs > 3.5).sum()
            data_quality = 1.0 - (gaps_count / len(df)) if len(df) > 0 else 0.0
            
            return df, max(0.0, float(data_quality))
        except Exception as e:
            logger.error(f"预处理tick数据失败: {e}", exc_info=True)
            return pd.DataFrame(), 0.0

    @staticmethod
    def calculate_intraday_chip_distribution(tick_data: pd.DataFrame, price_bins: int = 20) -> Dict[str, Any]:
        """
        基于tick数据计算日内筹码分布
        Args:
            tick_data: 预处理后的tick数据
            price_bins: 价格区间数量
        Returns:
            日内筹码分布统计
        """
        try:
            if tick_data.empty:
                return {}
            prices = tick_data['price'].values
            volumes = tick_data['volume'].values
            # 计算价格区间
            price_min, price_max = np.min(prices), np.max(prices)
            price_range = price_max - price_min
            if price_range <= 0:
                return {}
            # 按价格区间分组统计成交量
            bin_edges = np.linspace(price_min, price_max, price_bins + 1)
            bin_indices = np.digitize(prices, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, price_bins - 1)
            # 统计各区间成交量
            bin_volumes = np.zeros(price_bins)
            for i in range(price_bins):
                mask = bin_indices == i
                if np.any(mask):
                    bin_volumes[i] = np.sum(volumes[mask])
            total_volume = np.sum(bin_volumes)
            if total_volume <= 0:
                return {}
            # 计算归一化分布
            bin_distribution = bin_volumes / total_volume
            # 计算统计指标
            valid_bins = bin_distribution > 0
            # 集中度 (HHI指数)
            concentration = np.sum(bin_distribution ** 2)
            # 熵值
            entropy = -np.sum(bin_distribution[valid_bins] * np.log(bin_distribution[valid_bins]))
            # 偏度 (基于成交量加权)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            mean_price = np.average(bin_centers, weights=bin_distribution)
            std_price = np.sqrt(np.average((bin_centers - mean_price) ** 2, weights=bin_distribution))
            if std_price > 0:
                z_scores = (bin_centers - mean_price) / std_price
                skewness = np.average(z_scores ** 3, weights=bin_distribution)
            else:
                skewness = 0.0
            return {
                'concentration': float(concentration),
                'entropy': float(entropy),
                'skewness': float(skewness),
                'price_distribution': dict(zip(bin_centers, bin_distribution)),
                'price_range': (float(price_min), float(price_max))
            }
        except Exception as e:
            logger.error(f"计算日内筹码分布失败: {e}")
            return {}
    
    @staticmethod
    def calculate_intraday_chip_flow(tick_data: pd.DataFrame) -> Dict[str, float]:
        """
        计算tick级筹码流动
        Args:
            tick_data: 包含买卖方向的tick数据
        Returns:
            筹码流动指标
        """
        try:
            if tick_data.empty or 'type' not in tick_data.columns:
                return {}
            # 按买卖方向分组
            buy_mask = tick_data['type'] == 'B'
            sell_mask = tick_data['type'] == 'S'
            buy_volume = tick_data.loc[buy_mask, 'volume'].sum() if np.any(buy_mask) else 0
            sell_volume = tick_data.loc[sell_mask, 'volume'].sum() if np.any(sell_mask) else 0
            total_volume = buy_volume + sell_volume
            if total_volume <= 0:
                return {}
            # 净流动比例
            net_flow_ratio = (buy_volume - sell_volume) / total_volume
            # 流动强度
            flow_intensity = (buy_volume + sell_volume) / total_volume  # 非中性盘占比
            # 连续同向tick统计（聚类指数）
            if len(tick_data) >= 2:
                directions = np.where(tick_data['type'] == 'B', 1, 
                                     np.where(tick_data['type'] == 'S', -1, 0))
                direction_changes = np.diff(directions) != 0
                clustering_index = 1.0 - np.sum(direction_changes) / len(direction_changes)
            else:
                clustering_index = 0.0
            return {
                'net_flow_ratio': float(net_flow_ratio),
                'flow_intensity': float(flow_intensity),
                'clustering_index': float(clustering_index),
                'buy_ratio': float(buy_volume / total_volume),
                'sell_ratio': float(sell_volume / total_volume)
            }
        except Exception as e:
            logger.error(f"计算日内筹码流动失败: {e}")
            return {}
    
    @staticmethod
    def calculate_intraday_cost_center(tick_data: pd.DataFrame) -> Dict[str, float]:
        """
        计算日内成本重心及其动态
        Args:
            tick_data: tick数据
        Returns:
            成本重心指标
        """
        try:
            if tick_data.empty:
                return {}
            # 按时间窗口计算成本重心序列
            tick_data = tick_data.copy()
            # 计算累积加权平均成本
            cum_volume = tick_data['volume'].cumsum()
            cum_value = (tick_data['price'] * tick_data['volume']).cumsum()
            # 避免除零
            valid_mask = cum_volume > 0
            if not np.any(valid_mask):
                return {}
            cost_center_series = cum_value[valid_mask] / cum_volume[valid_mask]
            # 开盘成本
            open_cost = cost_center_series.iloc[0] if len(cost_center_series) > 0 else 0
            # 收盘成本（最终成本）
            close_cost = cost_center_series.iloc[-1] if len(cost_center_series) > 0 else 0
            # 成本重心迁移
            if open_cost > 0:
                migration_ratio = (close_cost - open_cost) / open_cost
            else:
                migration_ratio = 0.0
            # 成本重心波动率
            if len(cost_center_series) >= 2:
                volatility = cost_center_series.std()
            else:
                volatility = 0.0
            # 日内高低成本
            max_cost = cost_center_series.max()
            min_cost = cost_center_series.min()
            return {
                'open_cost': float(open_cost),
                'close_cost': float(close_cost),
                'migration_ratio': float(migration_ratio),
                'volatility': float(volatility),
                'max_cost': float(max_cost),
                'min_cost': float(min_cost),
                'cost_series': cost_center_series.tolist()
            }
        except Exception as e:
            logger.error(f"计算日内成本重心失败: {e}")
            return {}
    
    @staticmethod
    def identify_intraday_support_resistance(tick_data: pd.DataFrame, chip_dist: pd.DataFrame) -> Dict[str, Any]:
        """
        识别日内支撑阻力位测试情况
        Args:
            tick_data: tick数据
            chip_dist: 日线筹码分布数据
        Returns:
            支撑阻力测试指标
        """
        try:
            if tick_data.empty or chip_dist.empty:
                return {}
            # 从筹码分布中提取关键分位
            cost_50pct = chip_dist['price'].iloc[len(chip_dist) // 2] if len(chip_dist) > 0 else 0
            # 计算筹码密集区（前30%密集区域）
            if len(chip_dist) >= 3:
                sorted_chip = chip_dist.sort_values('percent', ascending=False)
                top_30_percent = sorted_chip.head(max(1, int(len(chip_dist) * 0.3)))
                dense_price_min = top_30_percent['price'].min()
                dense_price_max = top_30_percent['price'].max()
            else:
                dense_price_min = dense_price_max = cost_50pct
            # 识别tick价格触及关键位的情况
            tick_prices = tick_data['price'].values
            # 触及支撑（价格低于密集区下沿2%）
            support_threshold = dense_price_min * 0.98
            support_tests = np.sum(tick_prices <= support_threshold)
            # 触及阻力（价格高于密集区上沿2%）
            resistance_threshold = dense_price_max * 1.02
            resistance_tests = np.sum(tick_prices >= resistance_threshold)
            # 窄幅震荡比例（价格在密集区±1%内）
            narrow_range_min = dense_price_min * 0.99
            narrow_range_max = dense_price_max * 1.01
            in_narrow_range = np.sum((tick_prices >= narrow_range_min) & 
                                    (tick_prices <= narrow_range_max))
            total_ticks = len(tick_prices)
            consolidation_degree = in_narrow_range / total_ticks if total_ticks > 0 else 0.0
            return {
                'support_test_count': int(support_tests),
                'resistance_test_count': int(resistance_tests),
                'consolidation_degree': float(consolidation_degree),
                'support_level': float(support_threshold),
                'resistance_level': float(resistance_threshold),
                'dense_range': (float(dense_price_min), float(dense_price_max))
            }
        except Exception as e:
            logger.error(f"识别日内支撑阻力失败: {e}")
            return {}
    
    @staticmethod
    def calculate_intraday_abnormal_volume(tick_data: pd.DataFrame) -> Dict[str, float]:
        """
        计算tick异常成交量
        Args:
            tick_data: tick数据
        Returns:
            异常成交量指标
        """
        try:
            if tick_data.empty:
                return {}
            volumes = tick_data['volume'].values
            if len(volumes) < 10:
                return {}
            # 计算统计量
            mean_volume = np.mean(volumes)
            std_volume = np.std(volumes)
            if std_volume <= 0:
                return {}
            # 识别异常tick（超过3倍标准差）
            threshold = mean_volume + 3 * std_volume
            abnormal_mask = volumes > threshold
            abnormal_count = np.sum(abnormal_mask)
            abnormal_volume = np.sum(volumes[abnormal_mask])
            total_volume = np.sum(volumes)
            abnormal_ratio = abnormal_volume / total_volume if total_volume > 0 else 0.0
            # 计算成交效率（价格变动单位带来的筹码转移）
            if len(tick_data) >= 2:
                price_changes = np.abs(np.diff(tick_data['price'].values))
                volume_changes = volumes[1:]  # 与价格变化对应的成交量
                valid_mask = price_changes > 0
                if np.any(valid_mask):
                    transfer_efficiency = np.mean(volume_changes[valid_mask] / price_changes[valid_mask])
                else:
                    transfer_efficiency = 0.0
            else:
                transfer_efficiency = 0.0
            return {
                'abnormal_volume_ratio': float(abnormal_ratio),
                'abnormal_tick_count': int(abnormal_count),
                'transfer_efficiency': float(transfer_efficiency),
                'mean_volume': float(mean_volume),
                'std_volume': float(std_volume)
            }
        except Exception as e:
            logger.error(f"计算异常成交量失败: {e}")
            return {}
    
    @staticmethod
    def calculate_intraday_chip_locking(tick_data: pd.DataFrame, current_price: float) -> Dict[str, float]:
        """
        计算日内筹码锁定情况
        Args:
            tick_data: tick数据
            current_price: 当前价格
        Returns:
            筹码锁定指标
        """
        try:
            if tick_data.empty or current_price <= 0:
                return {}
            prices = tick_data['price'].values
            volumes = tick_data['volume'].values
            # 定义价格区间
            price_range = np.max(prices) - np.min(prices)
            if price_range <= 0:
                return {}
            # 低位区间（最低价+20%区间）
            low_bound = np.min(prices) + price_range * 0.2
            low_mask = prices <= low_bound
            # 高位区间（最高价-20%区间）
            high_bound = np.max(prices) - price_range * 0.2
            high_mask = prices >= high_bound
            # 计算各区成交量占比
            low_volume = np.sum(volumes[low_mask])
            high_volume = np.sum(volumes[high_mask])
            total_volume = np.sum(volumes)
            if total_volume <= 0:
                return {}
            low_ratio = low_volume / total_volume
            high_ratio = high_volume / total_volume
            # 识别峰谷区域
            # 使用简单的峰值检测（简化版）
            if len(volumes) >= 5:
                # 平滑处理
                smooth_volumes = savgol_filter(volumes, window_length=min(5, len(volumes)), polyorder=2)
                # 寻找峰值和谷值
                peaks_idx, _ = find_peaks(smooth_volumes, height=np.mean(smooth_volumes))
                valleys_idx, _ = find_peaks(-smooth_volumes, height=-np.mean(smooth_volumes))
                if len(peaks_idx) > 0 and len(valleys_idx) > 0:
                    # 计算峰谷成交量比
                    peak_volumes = volumes[peaks_idx]
                    valley_volumes = volumes[valleys_idx]
                    peak_valley_ratio = np.mean(peak_volumes) / np.mean(valley_volumes) if np.mean(valley_volumes) > 0 else 0.0
                    # 计算谷填充度（谷底区域实际成交量/预期成交量）
                    trough_filling = np.sum(valley_volumes) / (len(valleys_idx) * np.mean(volumes)) if np.mean(volumes) > 0 else 0.0
                else:
                    peak_valley_ratio = 0.0
                    trough_filling = 0.0
            else:
                peak_valley_ratio = 0.0
                trough_filling = 0.0
            return {
                'low_lock_ratio': float(low_ratio),
                'high_lock_ratio': float(high_ratio),
                'peak_valley_ratio': float(peak_valley_ratio),
                'trough_filling': float(trough_filling),
                'price_levels': {
                    'low_bound': float(low_bound),
                    'high_bound': float(high_bound),
                    'current': float(current_price)
                }
            }
        except Exception as e:
            logger.error(f"计算日内筹码锁定失败: {e}")
            return {}
    
    @staticmethod
    def calculate_intraday_chip_game_index(tick_data: pd.DataFrame) -> float:
        """
        计算日内筹码博弈指数
        反映买卖双方筹码交换的激烈程度
        """
        try:
            if tick_data.empty or 'type' not in tick_data.columns:
                return 0.0
            # 计算买卖平衡度
            flow_metrics = ChipFactorCalculator.calculate_intraday_chip_flow(tick_data)
            if not flow_metrics:
                return 0.0
            buy_ratio = flow_metrics.get('buy_ratio', 0.5)
            sell_ratio = flow_metrics.get('sell_ratio', 0.5)
            # 平衡度（越接近0.5表示博弈越激烈）
            balance = 1.0 - 2.0 * abs(buy_ratio - 0.5)
            # 聚类指数（连续同向交易降低博弈激烈度）
            clustering = flow_metrics.get('clustering_index', 0.0)
            # 博弈指数 = 平衡度 * (1 - 聚类指数)
            game_index = balance * (1.0 - clustering)
            return float(game_index)
        except Exception as e:
            logger.error(f"计算筹码博弈指数失败: {e}")
            return 0.0

    @staticmethod
    def calculate_complete_factors_with_tick(chip_perf_data: Dict,chip_dist_data: pd.DataFrame,daily_basic_data: Dict,daily_kline_data: Dict,prev_chip_dist_data: pd.DataFrame = None,historical_prices: pd.Series = None,historical_chip_factors: List[Dict] = None,tick_data: pd.DataFrame = None) -> Dict:
        """
        计算完整的筹码因子（包含tick数据支持）
        """
        factors = {}
        try:
            # 1. 先计算基础因子（原有逻辑）
            base_factors = ChipFactorCalculator.calculate_all_factors(
                chip_perf_data, chip_dist_data, daily_basic_data, daily_kline_data
            )
            factors.update(base_factors)
            # 2. 计算tick相关因子（如果tick数据可用）
            if tick_data is not None and not tick_data.empty:
                # 预处理tick数据
                processed_tick, data_quality = ChipFactorCalculator.preprocess_tick_data(tick_data)
                factors['tick_data_quality_score'] = data_quality
                factors['intraday_factor_calc_method'] = 'tick_based'
                if data_quality > 0.3:  # 数据质量阈值
                    close_price = factors.get('close', 0)
                    # 计算各类tick因子
                    # a. 日内筹码分布统计
                    intraday_dist = ChipFactorCalculator.calculate_intraday_chip_distribution(
                        processed_tick
                    )
                    if intraday_dist:
                        factors['intraday_chip_concentration'] = intraday_dist.get('concentration', 0.0)
                        factors['intraday_chip_entropy'] = intraday_dist.get('entropy', 0.0)
                        factors['intraday_price_distribution_skewness'] = intraday_dist.get('skewness', 0.0)
                    # b. 日内筹码流动
                    intraday_flow = ChipFactorCalculator.calculate_intraday_chip_flow(processed_tick)
                    if intraday_flow:
                        factors['tick_level_chip_flow'] = intraday_flow.get('net_flow_ratio', 0.0)
                        factors['intraday_chip_turnover_intensity'] = intraday_flow.get('flow_intensity', 0.0)
                        factors['tick_clustering_index'] = intraday_flow.get('clustering_index', 0.0)
                    # c. 日内成本重心
                    cost_center = ChipFactorCalculator.calculate_intraday_cost_center(processed_tick)
                    if cost_center:
                        factors['intraday_cost_center_migration'] = cost_center.get('migration_ratio', 0.0)
                        factors['intraday_cost_center_volatility'] = cost_center.get('volatility', 0.0)
                    # d. 日内支撑阻力测试
                    support_resistance = ChipFactorCalculator.identify_intraday_support_resistance(
                        processed_tick, chip_dist_data
                    )
                    if support_resistance:
                        factors['intraday_support_test_count'] = support_resistance.get('support_test_count', 0)
                        factors['intraday_resistance_test_count'] = support_resistance.get('resistance_test_count', 0)
                        factors['intraday_chip_consolidation_degree'] = support_resistance.get('consolidation_degree', 0.0)
                    # e. 异常成交量
                    abnormal_volume = ChipFactorCalculator.calculate_intraday_abnormal_volume(processed_tick)
                    if abnormal_volume:
                        factors['tick_abnormal_volume_ratio'] = abnormal_volume.get('abnormal_volume_ratio', 0.0)
                        factors['tick_chip_transfer_efficiency'] = abnormal_volume.get('transfer_efficiency', 0.0)
                    # f. 日内筹码锁定
                    chip_locking = ChipFactorCalculator.calculate_intraday_chip_locking(
                        processed_tick, close_price
                    )
                    if chip_locking:
                        factors['intraday_low_lock_ratio'] = chip_locking.get('low_lock_ratio', 0.0)
                        factors['intraday_high_lock_ratio'] = chip_locking.get('high_lock_ratio', 0.0)
                        factors['intraday_peak_valley_ratio'] = chip_locking.get('peak_valley_ratio', 0.0)
                        factors['intraday_trough_filling_degree'] = chip_locking.get('trough_filling', 0.0)
                    # g. 筹码博弈指数
                    game_index = ChipFactorCalculator.calculate_intraday_chip_game_index(processed_tick)
                    factors['intraday_chip_game_index'] = game_index
                    # h. 计算筹码平衡比
                    if intraday_flow:
                        buy_ratio = intraday_flow.get('buy_ratio', 0.5)
                        sell_ratio = intraday_flow.get('sell_ratio', 0.5)
                        factors['tick_chip_balance_ratio'] = buy_ratio / sell_ratio if sell_ratio > 0 else 1.0
                else:
                    # 数据质量不足，使用日线近似
                    factors = ChipFactorCalculator._approximate_intraday_factors(
                        factors, chip_dist_data, daily_kline_data
                    )
                    factors['intraday_factor_calc_method'] = 'daily_only'
            else:
                # 无tick数据，使用日线近似
                factors = ChipFactorCalculator._approximate_intraday_factors(
                    factors, chip_dist_data, daily_kline_data
                )
                factors['intraday_factor_calc_method'] = 'daily_only'
            # 3. 继续计算其他原有因子（时间序列、市场适应性等）
            # ... 原有逻辑 ...
            factors['calc_status'] = 'success'
            
        except Exception as e:
            logger.error(f"计算完整筹码因子(tick版)失败: {e}", exc_info=True)
            factors['calc_status'] = 'failed'
            factors['error_message'] = str(e)
        
        return factors
    
    @staticmethod
    def _approximate_intraday_factors(factors: Dict, chip_dist_data: pd.DataFrame,daily_kline_data: Dict) -> Dict:
        """
        使用日线数据近似日内因子（当tick数据不可用时）
        """
        try:
            # 基于日线数据近似部分日内因子
            if not chip_dist_data.empty:
                # 使用日线筹码分布近似日内分布
                daily_entropy = factors.get('chip_entropy', 0.0)
                daily_concentration = factors.get('chip_concentration_ratio', 0.5)
                # 简单近似：日内因子 = 日线因子 * 调整系数
                # 实际应用中可根据历史数据回归得到更好的近似公式
                factors['intraday_chip_concentration'] = daily_concentration * 1.2  # 假设日内更集中
                factors['intraday_chip_entropy'] = daily_entropy * 0.8  # 假设日内熵值更低
            # 基于K线数据近似其他因子
            if daily_kline_data:
                vol = daily_kline_data.get('vol', 0)
                high = daily_kline_data.get('high', 0)
                low = daily_kline_data.get('low', 0)
                close = daily_kline_data.get('close', 0)
                if high > low > 0:
                    # 近似日内价格区间占比
                    day_range = high - low
                    factors['intraday_price_range_ratio'] = day_range / close if close > 0 else 0.0
                    # 近似日内筹码换手强度
                    turnover_rate = factors.get('turnover_rate', 0)
                    factors['intraday_chip_turnover_intensity'] = turnover_rate / 100.0  # 归一化
            # 设置默认值
            default_values = {
                'intraday_low_lock_ratio': 0.3,
                'intraday_high_lock_ratio': 0.2,
                'intraday_cost_center_migration': 0.0,
                'intraday_cost_center_volatility': 0.0,
                'intraday_chip_game_index': 0.5,
                'tick_data_quality_score': 0.0
            }
            for key, value in default_values.items():
                if key not in factors:
                    factors[key] = value
            return factors
        
        except Exception as e:
            logger.error(f"近似日内因子失败: {e}")
            return factors

    @staticmethod
    def analyze_tick_data_availability(tick_data: pd.DataFrame) -> Dict[str, Any]:
        """
        分析tick数据可用性和质量
        Args:
            tick_data: tick数据DataFrame
        Returns:
            数据质量分析结果
        """
        try:
            if tick_data is None or tick_data.empty:
                return {
                    'available': False,
                    'quality_score': 0.0,
                    'total_ticks': 0,
                    'time_span_hours': 0.0,
                    'missing_intervals': 0,
                    'recommendation': 'no_data'
                }
            # 基本统计
            total_ticks = len(tick_data)
            if total_ticks == 0:
                return {
                    'available': False,
                    'quality_score': 0.0,
                    'total_ticks': 0,
                    'time_span_hours': 0.0,
                    'missing_intervals': 0,
                    'recommendation': 'no_data'
                }
            # 时间跨度
            time_min = tick_data['trade_time'].min()
            time_max = tick_data['trade_time'].max()
            time_span = (time_max - time_min).total_seconds() / 3600  # 小时
            # 理想情况：4小时交易时间（9:30-11:30, 13:00-15:00）
            expected_hours = 4.0
            # 检查时间连续性（3秒间隔）
            tick_times = tick_data['trade_time'].sort_values().values
            time_diffs = np.diff(tick_times) / np.timedelta64(1, 's')
            # 计算缺失间隔（超过4秒的间隔视为缺失）
            missing_intervals = np.sum(time_diffs > 4.0)
            # 数据质量评分（0-1）
            # 1. 时间覆盖度（最高0.4分）
            time_coverage = min(1.0, time_span / expected_hours) * 0.4
            # 2. 连续性（最高0.3分）
            continuity_score = 1.0 - min(1.0, missing_intervals / max(1, total_ticks)) * 0.3
            # 3. 数据量充分性（最高0.3分）
            # 理想情况下，4小时=4800个3秒tick（实际会有集合竞价等）
            expected_ticks = 4800
            volume_score = min(1.0, total_ticks / expected_ticks) * 0.3
            quality_score = time_coverage + continuity_score + volume_score
            # 给出使用建议
            if quality_score >= 0.7:
                recommendation = 'excellent'
            elif quality_score >= 0.5:
                recommendation = 'good'
            elif quality_score >= 0.3:
                recommendation = 'fair'
            else:
                recommendation = 'poor'
            return {
                'available': True,
                'quality_score': float(quality_score),
                'total_ticks': int(total_ticks),
                'time_span_hours': float(time_span),
                'missing_intervals': int(missing_intervals),
                'recommendation': recommendation
            }
            
        except Exception as e:
            logger.error(f"分析tick数据质量失败: {e}")
            return {
                'available': False,
                'quality_score': 0.0,
                'total_ticks': 0,
                'time_span_hours': 0.0,
                'missing_intervals': 0,
                'recommendation': 'error'
            }
