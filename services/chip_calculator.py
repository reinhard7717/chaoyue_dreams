# services\chip_calculator.py
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional, List
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
        计算筹码分布熵值
        公式: -∑(p_i * ln(p_i))
        Args:
            price_percent_dict: 价格-占比字典 {price: percent}
        Returns:
            float: 熵值
        """
        try:
            # 确保占比和为1
            total = sum(price_percent_dict.values())
            if total <= 0:
                return 0.0
            # 归一化
            normalized_percents = {p: v/total for p, v in price_percent_dict.items()}
            # 计算熵值
            entropy = 0.0
            for percent in normalized_percents.values():
                if percent > 0:
                    entropy -= percent * np.log(percent)
            return entropy
        except Exception as e:
            logger.error(f"计算筹码熵值失败: {e}")
            return 0.0
    
    @staticmethod
    def calculate_chip_skewness_kurtosis(price_percent_dict: Dict[float, float]) -> Tuple[float, float, float, float]:
        """
        计算筹码分布的均值、标准差、偏度和峰度
        Args:
            price_percent_dict: 价格-占比字典
        Returns:
            Tuple[mean, std, skewness, kurtosis]
        """
        try:
            if not price_percent_dict:
                return 0.0, 0.0, 0.0, 0.0
            # 转换为数组
            prices = list(price_percent_dict.keys())
            percents = list(price_percent_dict.values())
            # 归一化
            total = sum(percents)
            if total <= 0:
                return 0.0, 0.0, 0.0, 0.0
            weights = np.array(percents) / total
            # 计算加权均值
            mean = np.average(prices, weights=weights)
            # 计算加权方差
            variance = np.average((prices - mean) ** 2, weights=weights)
            std = np.sqrt(variance) if variance > 0 else 0.0
            # 计算偏度（三阶矩）
            if std > 0:
                skewness = np.average(((prices - mean) / std) ** 3, weights=weights)
            else:
                skewness = 0.0
            # 计算峰度（四阶矩）
            if std > 0:
                kurtosis = np.average(((prices - mean) / std) ** 4, weights=weights) - 3
            else:
                kurtosis = 0.0
            return float(mean), float(std), float(skewness), float(kurtosis)
        except Exception as e:
            logger.error(f"计算筹码分布统计量失败: {e}")
            return 0.0, 0.0, 0.0, 0.0
    
    @staticmethod
    def calculate_profit_ratio(chip_data: pd.DataFrame, current_price: float) -> float:
        """
        计算获利比例（价格低于当前价的比例）
        Args:
            chip_data: 筹码分布DataFrame，包含'price'和'percent'列
            current_price: 当前价格
        Returns:
            float: 获利比例
        """
        try:
            if chip_data.empty or current_price <= 0:
                return 0.0
            # 筛选低于当前价的部分
            profit_chips = chip_data[chip_data['price'] <= current_price]
            if profit_chips.empty:
                return 0.0
            # 计算获利比例
            total_percent = chip_data['percent'].sum()
            if total_percent <= 0:
                return 0.0
            profit_percent = profit_chips['percent'].sum()
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
        计算移动平均线
        Args:
            prices: 价格序列
            windows: 均线周期列表
        Returns:
            Dict: 各周期均线值
        """
        try:
            ma_values = {}
            for window in windows:
                if len(prices) >= window:
                    ma_values[f'ma{window}'] = prices.rolling(window=window).mean().iloc[-1]
                else:
                    ma_values[f'ma{window}'] = None
            return ma_values
        except Exception as e:
            logger.error(f"计算移动平均线失败: {e}")
            return {}
    
    @staticmethod
    def detect_peaks(chip_dist: pd.DataFrame, min_height: float = 1.0, 
                     min_distance: float = 0.02) -> Tuple[List[float], List[float]]:
        """
        检测筹码峰
        Args:
            chip_dist: 筹码分布DataFrame，包含'price'和'percent'列
            min_height: 最小峰值高度（占比%）
            min_distance: 最小峰间距离（相对于价格区间）
        Returns:
            Tuple: (峰价格列表, 峰高度列表)
        """
        try:
            if chip_dist.empty or len(chip_dist) < 3:
                return [], []
            # 平滑处理筹码分布
            percent_smooth = savgol_filter(chip_dist['percent'].values, 
                                          window_length=min(7, len(chip_dist)),
                                          polyorder=2)
            # 计算最小距离（价格点数量）
            price_range = chip_dist['price'].max() - chip_dist['price'].min()
            min_dist_points = int(len(chip_dist) * min_distance)
            # 寻找峰值
            peaks_idx, properties = find_peaks(
                percent_smooth, 
                height=min_height,
                distance=max(2, min_dist_points)
            )
            # 获取峰价格和高度
            peak_prices = chip_dist.iloc[peaks_idx]['price'].tolist()
            peak_heights = chip_dist.iloc[peaks_idx]['percent'].tolist()
            return peak_prices, peak_heights
        except Exception as e:
            logger.error(f"检测筹码峰失败: {e}")
            return [], []
    
    @staticmethod
    def analyze_peak_pattern(peak_prices: List[float], peak_heights: List[float],
                            price_range: Tuple[float, float]) -> Dict:
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
            # 峰间集中度（前两大峰占比）
            if len(peak_heights) >= 2:
                sorted_heights = sorted(peak_heights, reverse=True)
                analysis['peak_concentration'] = sum(sorted_heights[:2]) / sum(peak_heights)
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
    def calculate_chip_flow(current_chip: pd.DataFrame, prev_chip: pd.DataFrame, 
                           current_price: float) -> Tuple[int, float]:
        """
        计算筹码流动方向和强度
        Args:
            current_chip: 当前筹码分布
            prev_chip: 前一日筹码分布
            current_price: 当前价格
        Returns:
            Tuple: (流动方向, 流动强度)
        """
        try:
            if prev_chip.empty or current_chip.empty:
                return 0, 0.0
            # 计算筹码重心（加权平均价）
            def chip_center_of_gravity(chip_df):
                total = chip_df['percent'].sum()
                if total <= 0:
                    return 0.0
                return np.sum(chip_df['price'] * chip_df['percent']) / total
            prev_center = chip_center_of_gravity(prev_chip)
            curr_center = chip_center_of_gravity(current_chip)
            # 计算筹码流动强度
            flow_intensity = abs(curr_center - prev_center) / max(prev_center, 1e-6)
            # 判断流动方向
            if flow_intensity < 0.001:  # 阈值
                flow_direction = 0  # 横盘整理
            elif curr_center > prev_center:
                flow_direction = 1  # 向上流动
            else:
                flow_direction = -1  # 向下流动
            return flow_direction, flow_intensity
        except Exception as e:
            logger.error(f"计算筹码流动失败: {e}")
            return 0, 0.0
    
    @staticmethod
    def calculate_convergence_divergence(chip_dist: pd.DataFrame, cost_50pct: float,
                                        price_range: Tuple[float, float]) -> Tuple[float, float]:
        """
        计算筹码聚集度和发散度
        Args:
            chip_dist: 筹码分布
            cost_50pct: 50分位成本
            price_range: 价格区间
        Returns:
            Tuple: (聚集度, 发散度)
        """
        try:
            min_price, max_price = price_range
            price_span = max_price - min_price
            if price_span <= 0 or chip_dist.empty:
                return 0.0, 0.0
            # 聚集度：50分位附近筹码占比
            # 定义聚集区间为 cost_50pct ± 5%
            if cost_50pct > 0:
                convergence_range = cost_50pct * 0.05  # 5%区间
                convergence_mask = (chip_dist['price'] >= cost_50pct - convergence_range) & \
                                  (chip_dist['price'] <= cost_50pct + convergence_range)
                convergence_ratio = chip_dist.loc[convergence_mask, 'percent'].sum() / \
                                   chip_dist['percent'].sum()
            else:
                convergence_ratio = 0.0
            # 发散度：整个价格区间的筹码分布宽度
            divergence_ratio = price_span / (max_price + min_price) * 0.5
            return convergence_ratio, divergence_ratio
        except Exception as e:
            logger.error(f"计算聚集发散度失败: {e}")
            return 0.0, 0.0
    
    @staticmethod
    def calculate_trend_reversal_scores(
        chip_factors: Dict,
        ma_values: Dict,
        price_data: Dict,
        volume_data: Dict
    ) -> Tuple[float, float]:
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
            close = price_data.get('close', 0)
            volume = volume_data.get('volume', 0)
            turnover_rate = volume_data.get('turnover_rate', 0)
            # ========== 趋势得分计算 ==========
            trend_factors = []
            # 1. 均线排列得分
            if all(k in ma_values for k in ['ma5', 'ma21', 'ma34', 'ma55']):
                if (ma_values['ma5'] > ma_values['ma21'] > ma_values['ma34'] > ma_values['ma55']):
                    trend_factors.append(1.0)  # 强势多头
                elif (ma_values['ma5'] < ma_values['ma21'] < ma_values['ma34'] < ma_values['ma55']):
                    trend_factors.append(-1.0)  # 强势空头
                else:
                    trend_factors.append(0.0)
            # 2. 股价相对筹码成本位置
            weight_avg = chip_factors.get('weight_avg_cost', 0)
            if weight_avg > 0:
                price_to_chip = (close - weight_avg) / weight_avg
                # 强势多头：股价远高于成本
                if price_to_chip > 0.15:
                    trend_factors.append(1.0)
                # 强势空头：股价远低于成本
                elif price_to_chip < -0.15:
                    trend_factors.append(-1.0)
                else:
                    trend_factors.append(0.0)
            # 3. 筹码集中度趋势
            chip_concentration = chip_factors.get('chip_concentration_ratio', 0)
            if chip_concentration < 0.3:  # 高度集中
                trend_factors.append(1.0)  # 趋势延续信号
            elif chip_concentration > 0.7:  # 高度分散
                trend_factors.append(-0.5)  # 趋势可能结束
            # 4. 量价配合
            if turnover_rate > 0:
                # 价格上涨+放量 = 强势
                price_change = price_data.get('pct_change', 0)
                if price_change > 0.01 and turnover_rate > 0.05:
                    trend_factors.append(1.0)
                elif price_change < -0.01 and turnover_rate > 0.05:
                    trend_factors.append(-1.0)
            # 趋势得分 = 因子加权平均
            if trend_factors:
                trend_score = np.mean(trend_factors)
            # ========== 反转得分计算 ==========
            reversal_factors = []
            # 1. 筹码峰形态反转信号
            is_multi_peak = chip_factors.get('is_multi_peak', False)
            is_double_peak = chip_factors.get('is_double_peak', False)
            if is_multi_peak:
                reversal_factors.append(0.8)  # 多峰常预示反转
            elif is_double_peak:
                # 双峰：判断是顶部双峰还是底部双峰
                price_position = chip_factors.get('price_percentile_position', 0.5)
                if price_position > 0.8:  # 高位双峰
                    reversal_factors.append(0.7)
                elif price_position < 0.2:  # 低位双峰
                    reversal_factors.append(-0.7)  # 负分表示底部反转
            # 2. 筹码发散度反转信号
            chip_divergence = chip_factors.get('chip_divergence_ratio', 0)
            if chip_divergence > 0.6:  # 高度发散
                reversal_factors.append(0.6)
            # 3. 价格与筹码背离
            winner_rate = chip_factors.get('winner_rate', 0)
            if close > weight_avg and winner_rate < 0.5:
                # 价格高于成本但胜率低 -> 顶部背离
                reversal_factors.append(0.5)
            elif close < weight_avg and winner_rate > 0.7:
                # 价格低于成本但胜率高 -> 底部背离
                reversal_factors.append(-0.5)
            # 4. 极端分位突破
            price_to_95pct = (close - chip_factors.get('cost_95pct', 0)) / close
            price_to_5pct = (close - chip_factors.get('cost_5pct', 0)) / close
            if price_to_95pct > 0.03:  # 突破95分位
                reversal_factors.append(0.4)
            elif price_to_5pct < -0.03:  # 跌破5分位
                reversal_factors.append(-0.4)
            # 反转得分 = 因子加权平均
            if reversal_factors:
                reversal_score = np.mean(reversal_factors)
            return trend_score, reversal_score
        except Exception as e:
            logger.error(f"计算趋势反转得分失败: {e}")
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
    def calculate_chip_migration(chip_current: pd.DataFrame, 
                                chip_previous: pd.DataFrame,
                                window_days: int = 5) -> Dict:
        """
        计算筹码迁移情况
        关键指标：
        1. 聚集度变化 = 当前集中度 / 前期集中度 - 1
        2. 发散度变化 = 当前发散度 / 前期发散度 - 1
        3. 重心移动速度 = (当前重心 - 前期重心) / 时间窗口
        4. 筹码转移矩阵：
           - 低价区 → 高价区转移比例
           - 高价区 → 低价区转移比例
           - 中间区筹码稳定性
        """
        migration = {
            'convergence_change': 0.0,
            'divergence_change': 0.0,
            'center_speed': 0.0,
            'low_to_high_ratio': 0.0,
            'high_to_low_ratio': 0.0,
            'stability_ratio': 0.0
        }
        # 计算筹码重心
        def chip_center(df):
            if df.empty:
                return 0.0
            total = df['percent'].sum()
            return np.sum(df['price'] * df['percent']) / total if total > 0 else 0.0
        current_center = chip_center(chip_current)
        previous_center = chip_center(chip_previous)
        migration['center_speed'] = (current_center - previous_center) / window_days
        # 计算价格三分位
        price_min = min(chip_current['price'].min(), chip_previous['price'].min())
        price_max = max(chip_current['price'].max(), chip_previous['price'].max())
        price_range = price_max - price_min
        if price_range > 0:
            # 定义低、中、高区域
            low_bound = price_min + price_range * 0.33
            high_bound = price_min + price_range * 0.67
            # 计算转移比例
            def region_percent(df, lower, upper):
                mask = (df['price'] >= lower) & (df['price'] <= upper)
                return df.loc[mask, 'percent'].sum() if not df.empty else 0.0
            # 前一日的区域筹码
            prev_low = region_percent(chip_previous, price_min, low_bound)
            prev_mid = region_percent(chip_previous, low_bound, high_bound)
            prev_high = region_percent(chip_previous, high_bound, price_max)
            # 当前日的区域筹码
            curr_low = region_percent(chip_current, price_min, low_bound)
            curr_mid = region_percent(chip_current, low_bound, high_bound)
            curr_high = region_percent(chip_current, high_bound, price_max)
            # 转移比例（简化计算）
            total = prev_low + prev_mid + prev_high
            if total > 0:
                # 低价区筹码流向
                migration['low_to_high_ratio'] = max(0, (prev_low - curr_low) / total)
                # 高价区筹码流向
                migration['high_to_low_ratio'] = max(0, (prev_high - curr_high) / total)
                # 中间区稳定性
                migration['stability_ratio'] = min(curr_mid, prev_mid) / max(curr_mid, prev_mid)
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
        计算筹码峰迁移速度（5日）
        
        Args:
            chip_centers_5d: 最近5日的筹码重心列表
        
        Returns:
            float: 迁移速度（价格变动/时间）
        """
        try:
            if len(chip_centers_5d) < 2:
                return 0.0
            # 计算线性回归斜率
            x = np.arange(len(chip_centers_5d))
            y = np.array(chip_centers_5d)
            slope, _, _, _, _ = linregress(x, y)
            # 标准化为每日迁移速度
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
            # 计算变化率
            current = chip_stability_5d[-1]
            previous = chip_stability_5d[0]
            if previous != 0:
                change_rate = (current - previous) / abs(previous)
            else:
                change_rate = 0.0
            return float(change_rate)
            
        except Exception as e:
            logger.error(f"计算筹码稳定性变化失败: {e}")
            return 0.0

    @staticmethod
    def calculate_volatility_adjusted_concentration(
        chip_concentration: float, 
        volatility_20d: float
    ) -> float:
        """
        计算波动率调整的筹码集中度
        
        Args:
            chip_concentration: 原始筹码集中度
            volatility_20d: 20日波动率
        
        Returns:
            float: 波动率调整后的集中度
        """
        try:
            if volatility_20d <= 0:
                return chip_concentration
            # 高波动率市场，筹码集中度的参考价值降低
            # 调整因子：波动率越高，集中度的信号强度越弱
            adjustment_factor = 1.0 / (1.0 + volatility_20d * 2)
            return chip_concentration * adjustment_factor
            
        except Exception as e:
            logger.error(f"计算波动率调整集中度失败: {e}")
            return chip_concentration

    @staticmethod
    def calculate_chip_rsi_divergence(
        rsi_14: float, 
        chip_flow_direction: int,
        price_trend: float  # 价格趋势，如最近5日涨跌幅
    ) -> float:
        """
        计算筹码RSI背离度
        
        Args:
            rsi_14: 14日RSI值
            chip_flow_direction: 筹码流动方向
            price_trend: 价格趋势
        
        Returns:
            float: 背离度（正值为顶背离，负值为底背离）
        """
        try:
            # RSI超买超卖判断
            is_overbought = rsi_14 > 70
            is_oversold = rsi_14 < 30
            # 顶背离：价格创新高但筹码向下流动
            if is_overbought and chip_flow_direction == -1 and price_trend > 0:
                return 1.0  # 强烈顶背离
            # 底背离：价格创新低但筹码向上流动
            elif is_oversold and chip_flow_direction == 1 and price_trend < 0:
                return -1.0  # 强烈底背离
            # 中度背离
            elif (rsi_14 > 60 and chip_flow_direction == -1) or \
                (rsi_14 < 40 and chip_flow_direction == 1):
                return 0.5 if chip_flow_direction == -1 else -0.5
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"计算RSI筹码背离失败: {e}")
            return 0.0

    @staticmethod
    def determine_enhanced_chip_structure(
        chip_factors: Dict,
        ma_values: Dict,
        volume_trend: float  # 成交量趋势
    ) -> str:
        """
        增强的筹码结构状态判断
        """
        try:
            price_position = chip_factors.get('price_percentile_position', 0.5)
            chip_concentration = chip_factors.get('chip_concentration_ratio', 0.5)
            chip_flow_direction = chip_factors.get('chip_flow_direction', 0)
            winner_rate = chip_factors.get('winner_rate', 0.5)
            turnover_rate = chip_factors.get('turnover_rate', 0)
            # 1. 吸筹阶段
            # 特征：价格低位+筹码集中+换手率适中+胜率低+成交量温和放大
            if (price_position < 0.3 and 
                chip_concentration < 0.4 and 
                2.0 < turnover_rate < 5.0 and
                winner_rate < 0.4 and
                volume_trend > 0.1 and  # 成交量趋势向上
                chip_flow_direction == 1):
                return 'accumulation'
            # 2. 拉升阶段
            # 特征：均线多头排列+价格突破关键位+换手率放大+获利盘增加
            ma_arrangement = ChipFactorCalculator.calculate_ma_arrangement(ma_values)
            if (ma_arrangement == 1 and
                price_position > 0.5 and
                chip_concentration < 0.6 and
                turnover_rate > 3.0 and
                winner_rate > 0.6):
                return 'lifting'
            # 3. 派发阶段
            # 特征：价格高位+筹码发散+换手率高+获利盘大但开始减少
            if (price_position > 0.7 and
                chip_concentration > 0.6 and
                turnover_rate > 8.0 and
                winner_rate > 0.8 and
                chip_flow_direction == -1):
                return 'distribution'
            # 4. 回落阶段
            # 特征：均线空头排列+筹码向下流动+换手率降低
            if (ma_arrangement == -1 and
                chip_flow_direction == -1 and
                turnover_rate < 3.0):
                return 'decline'
            # 5. 整理阶段
            return 'consolidation'
            
        except Exception as e:
            logger.error(f"判断增强筹码结构失败: {e}")
            return 'consolidation'

    @staticmethod
    def calculate_complete_factors(
        chip_perf_data: Dict,
        chip_dist_data: pd.DataFrame,
        daily_basic_data: Dict,
        daily_kline_data: Dict,
        prev_chip_dist_data: pd.DataFrame = None,
        historical_prices: pd.Series = None,
        historical_chip_factors: List[Dict] = None
    ) -> Dict:
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
                # 提取最近5日的筹码重心
                chip_centers_5d = [f.get('chip_mean', 0) for f in historical_chip_factors[-5:]]
                chip_stability_5d = [f.get('chip_stability', 0) for f in historical_chip_factors[-5:]]
                # 计算迁移速度
                factors['peak_migration_speed_5d'] = ChipFactorCalculator.calculate_peak_migration(
                    chip_centers_5d
                )
                # 计算稳定性变化
                factors['chip_stability_change_5d'] = ChipFactorCalculator.calculate_chip_stability_change(
                    chip_stability_5d
                )
            # 9. 计算市场适应性因子
            # 这里需要额外的数据，如波动率、RSI等
            # 暂时留空，后续可以根据需要添加
            factors['calc_status'] = 'success'
            
        except Exception as e:
            logger.error(f"计算完整筹码因子失败: {e}", exc_info=True)
            factors['calc_status'] = 'failed'
            factors['error_message'] = str(e)
        
        return factors

    @staticmethod
    def get_historical_data_requirements() -> Dict:
        """
        获取计算所需的历史数据要求
        
        Returns:
            Dict: 数据要求描述
        """
        return {
            'min_days_for_ma': 55,  # 计算MA55需要55天数据
            'min_days_for_trend': 20,  # 计算趋势需要20天数据
            'min_days_for_migration': 5,  # 计算迁移需要5天数据
            'required_fields': ['close', 'volume', 'turnover_rate']
        }

    @staticmethod
    def validate_data_availability(
        chip_perf_data: Dict,
        chip_dist_data: pd.DataFrame,
        historical_prices: pd.Series = None
    ) -> Tuple[bool, str]:
        """
        验证数据是否足够计算所有因子
        
        Returns:
            Tuple: (是否足够, 原因)
        """
        try:
            # 检查基础数据
            if not chip_perf_data:
                return False, "缺少筹码性能数据"
            if chip_dist_data.empty:
                return False, "缺少筹码分布数据"
            # 检查历史价格数据
            if historical_prices is not None:
                if len(historical_prices) < 20:
                    return False, "历史价格数据不足20天"
            return True, "数据足够"
            
        except Exception as e:
            return False, f"数据验证失败: {str(e)}"



