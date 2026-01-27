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
            # 去除NaN值
            mask = ~np.isnan(y)
            if np.sum(mask) < 2:
                return 0.0
            x_clean = x[mask]
            y_clean = y[mask]
            if len(x_clean) < 2:
                return 0.0
            slope, _, _, _, _ = linregress(x_clean, y_clean)
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
        计算价格波动率
        Args:
            prices: 价格序列
            window: 计算窗口
        Returns:
            float: 波动率（收益率标准差）
        """
        try:
            if len(prices) < window:
                return 0.0
            # 计算收益率
            returns = prices.pct_change().dropna()
            if len(returns) < window:
                return 0.0
            # 计算波动率（年化）
            volatility = returns.rolling(window=window).std().iloc[-1]
            # 年化波动率（假设252个交易日）
            annualized_vol = volatility * np.sqrt(252)
            return float(annualized_vol)
        except Exception as e:
            logger.error(f"计算波动率失败: {e}")
            return 0.0

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
        """
        计算RSI指标
        Args:
            prices: 价格序列
            period: RSI周期
        Returns:
            float: RSI值
        """
        try:
            if len(prices) < period + 1:
                return 50.0  # 默认中性值
            # 计算价格变化
            delta = prices.diff()
            # 分离上涨和下跌
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            # 计算RSI
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            # 获取最后一个有效值
            last_rsi = rsi.dropna().iloc[-1] if not rsi.dropna().empty else 50.0
            return float(last_rsi)
        except Exception as e:
            logger.error(f"计算RSI失败: {e}")
            return 50.0

    @staticmethod
    def calculate_volatility_adjusted_concentration(
        chip_concentration: float, 
        volatility_20d: float,
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
    def calculate_chip_rsi_divergence(
        rsi_value: float, 
        chip_flow_direction: int,
        price_trend_5d: float,
        rsi_trend_5d: float
    ) -> float:
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
    def calculate_rsi_trend(rsi_values: List[float], days: int = 5) -> float:
        """
        计算RSI趋势
        Args:
            rsi_values: RSI值列表
            days: 趋势天数
        Returns:
            float: RSI趋势（变化值）
        """
        try:
            if len(rsi_values) < days + 1:
                return 0.0
            # 计算days日RSI变化
            start_rsi = rsi_values[-days-1]
            end_rsi = rsi_values[-1]
            trend = end_rsi - start_rsi
            return float(trend)
        except Exception as e:
            logger.error(f"计算RSI趋势失败: {e}")
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
            if 'main_cost_range_ratio' not in factors:
                cost_50pct = chip_perf_data.get('cost_50pct')
                if cost_50pct:
                    factors['main_cost_range_ratio'] = ChipFactorCalculator.calculate_main_cost_range_ratio(chip_dist_data, cost_50pct)
                else:
                    factors['main_cost_range_ratio'] = 0.5
                    print(f"⚠️ [calculate_complete_factors] cost_50pct缺失，main_cost_range_ratio使用默认值0.5")
            # 检查并计算 high_position_lock_ratio_90
            if 'high_position_lock_ratio_90' not in factors:
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

    @staticmethod
    def calculate_holding_time_factors(
        chip_dist_current: pd.DataFrame,
        chip_dist_history: List[pd.DataFrame],  # 过去N日的筹码分布
        turnover_rate: float,
        avg_turnover_20d: float
    ) -> Dict[str, float]:
        """
        计算持有时间相关因子（基于换手率估算）
        Args:
            chip_dist_current: 当前筹码分布
            chip_dist_history: 历史筹码分布列表（按时间倒序）
            turnover_rate: 当日换手率(%)
            avg_turnover_20d: 20日平均换手率(%)
        Returns:
            Dict: 持有时间因子
        """
        factors = {}
        try:
            # 1. 平均持有时间估算（基于换手率）
            if turnover_rate > 0:
                # 简化公式：平均持有天数 = 总流通股本 / 日换手股本
                factors['avg_holding_days'] = 100 / turnover_rate  # 换手率已为百分比
            else:
                factors['avg_holding_days'] = 250  # 默认值（约一年）
            # 2. 短线筹码比例估算（<5日）
            # 假设换手筹码在5日内均匀分布
            daily_turnover = turnover_rate / 100  # 转换为小数
            factors['short_term_chip_ratio'] = min(daily_turnover * 5, 1.0)
            # 3. 长线筹码比例估算（>60日）
            # 使用历史筹码分布变化估算
            if len(chip_dist_history) >= 60:
                # 简单方法：比较60日前与当前的筹码分布重叠度
                chip_dist_60d_ago = chip_dist_history[-60]
                overlap_ratio = ChipFactorCalculator._calculate_chip_overlap(
                    chip_dist_current, chip_dist_60d_ago
                )
                factors['long_term_chip_ratio'] = overlap_ratio
            else:
                # 基于换手率估算
                factors['long_term_chip_ratio'] = max(0, 1 - (turnover_rate * 60 / 100))
            # 4. 筹码换手强度
            if avg_turnover_20d > 0:
                factors['chip_turnover_intensity'] = turnover_rate / avg_turnover_20d
            else:
                factors['chip_turnover_intensity'] = 1.0
            return factors
        except Exception as e:
            logger.error(f"计算持有时间因子失败: {e}")
            return {
                'avg_holding_days': 0,
                'short_term_chip_ratio': 0,
                'long_term_chip_ratio': 0,
                'chip_turnover_intensity': 1.0
            }
    
    @staticmethod
    def _calculate_chip_overlap(chip1: pd.DataFrame, chip2: pd.DataFrame) -> float:
        """
        计算两个筹码分布的重叠度
        """
        try:
            # 对齐价格区间
            min_price = min(chip1['price'].min(), chip2['price'].min())
            max_price = max(chip1['price'].max(), chip2['price'].max())
            # 创建相同的价格网格
            price_grid = np.linspace(min_price, max_price, 200)
            # 插值得到连续分布
            from scipy.interpolate import interp1d
            f1 = interp1d(chip1['price'], chip1['percent'], 
                         bounds_error=False, fill_value=0)
            f2 = interp1d(chip2['price'], chip2['percent'], 
                         bounds_error=False, fill_value=0)
            percent1 = f1(price_grid)
            percent2 = f2(price_grid)
            # 归一化
            percent1 = percent1 / max(percent1.sum(), 1e-6)
            percent2 = percent2 / max(percent2.sum(), 1e-6)
            # 计算重叠度（最小重叠积分）
            overlap = np.minimum(percent1, percent2).sum()
            return float(overlap)
        except:
            return 0.0
    
    @staticmethod
    def calculate_layered_flow_factors(
        chip_dist_current: pd.DataFrame,
        chip_dist_previous: pd.DataFrame,
        cost_15pct: float,
        cost_85pct: float
    ) -> Dict[str, float]:
        """
        计算分层筹码流动因子
        Returns:
            Dict: 包含分层流动数据
        """
        factors = {}
        try:
            if chip_dist_current.empty or chip_dist_previous.empty:
                return factors
            # 定义价格区间
            price_min = chip_dist_current['price'].min()
            price_max = chip_dist_current['price'].max()
            # 划分区间
            low_mask_current = chip_dist_current['price'] <= cost_15pct
            middle_mask_current = (chip_dist_current['price'] > cost_15pct) & \
                                 (chip_dist_current['price'] <= cost_85pct)
            high_mask_current = chip_dist_current['price'] > cost_85pct
            low_mask_prev = chip_dist_previous['price'] <= cost_15pct
            middle_mask_prev = (chip_dist_previous['price'] > cost_15pct) & \
                              (chip_dist_previous['price'] <= cost_85pct)
            high_mask_prev = chip_dist_previous['price'] > cost_85pct
            # 计算各区间筹码占比变化
            def get_zone_percent(df, mask):
                return df.loc[mask, 'percent'].sum() if not df.empty else 0
            low_current = get_zone_percent(chip_dist_current, low_mask_current)
            middle_current = get_zone_percent(chip_dist_current, middle_mask_current)
            high_current = get_zone_percent(chip_dist_current, high_mask_current)
            low_prev = get_zone_percent(chip_dist_previous, low_mask_prev)
            middle_prev = get_zone_percent(chip_dist_previous, middle_mask_prev)
            high_prev = get_zone_percent(chip_dist_previous, high_mask_prev)
            # 计算净变动
            factors['low_zone_chip_flow'] = low_current - low_prev
            factors['middle_zone_chip_flow'] = middle_current - middle_prev
            factors['high_zone_chip_flow'] = high_current - high_prev
            # 计算主力控盘度（简化版）
            total_change = abs(factors['low_zone_chip_flow']) + \
                          abs(factors['middle_zone_chip_flow']) + \
                          abs(factors['high_zone_chip_flow'])
            if total_change > 0:
                # 主力控盘度 = 最大区间变动 / 总变动
                max_change = max(abs(factors['low_zone_chip_flow']),
                               abs(factors['middle_zone_chip_flow']),
                               abs(factors['high_zone_chip_flow']))
                factors['main_force_control_ratio'] = max_change / total_change
            else:
                factors['main_force_control_ratio'] = 0
            return factors
        except Exception as e:
            logger.error(f"计算分层流动因子失败: {e}")
            return {}
    
    @staticmethod
    def calculate_main_force_behavior(
        flow_factors: Dict[str, float],
        turnover_rate: float,
        price_change: float
    ) -> Dict[str, float]:
        """
        计算主力行为强度因子
        """
        factors = {}
        try:
            # 吸筹强度：低价区流入且换手率适中
            if flow_factors.get('low_zone_chip_flow', 0) > 0:
                # 流入量 * 换手率 * 价格配合度
                factors['accumulation_intensity'] = (
                    flow_factors['low_zone_chip_flow'] * 
                    (turnover_rate / 100) * 
                    max(0, 1 + price_change)  # 价格上涨加强信号
                )
            else:
                factors['accumulation_intensity'] = 0
            # 派发强度：高价区流出且换手率高
            if flow_factors.get('high_zone_chip_flow', 0) < 0:
                factors['distribution_intensity'] = (
                    abs(flow_factors['high_zone_chip_flow']) * 
                    (turnover_rate / 100) * 
                    max(0, 1 - price_change)  # 价格下跌加强信号
                )
            else:
                factors['distribution_intensity'] = 0
            return factors
        except Exception as e:
            logger.error(f"计算主力行为因子失败: {e}")
            return {'accumulation_intensity': 0, 'distribution_intensity': 0}

    @staticmethod
    def calculate_main_cost_range_ratio(chip_dist_data: pd.DataFrame, cost_50pct: float) -> float:
        """
        基于筹码分布数据计算主力成本区间锁定比例 (main_cost_range_ratio)
        逻辑: 计算筹码成本在成本中位数(cost_50pct)上下10%区间内的筹码占比
        """
        try:
            if chip_dist_data.empty or cost_50pct <= 0:
                print(f"⚠️ [calculate_main_cost_range_ratio] 输入数据无效，返回默认值0.5")
                return 0.5
            lower_bound = cost_50pct * 0.9
            upper_bound = cost_50pct * 1.1
            # 筛选在主力成本区间的筹码
            main_mask = (chip_dist_data['price'] >= lower_bound) & (chip_dist_data['price'] <= upper_bound)
            if not main_mask.any():
                print(f"⚠️ [calculate_main_cost_range_ratio] 无筹码落在主力成本区间[{lower_bound:.2f}, {upper_bound:.2f}]，返回0")
                return 0.0
            main_chip_sum = chip_dist_data.loc[main_mask, 'percent'].sum()
            total_chip_sum = chip_dist_data['percent'].sum()
            if total_chip_sum <= 0:
                return 0.0
            ratio = main_chip_sum / total_chip_sum
            print(f"📊 [calculate_main_cost_range_ratio] 计算完成: {ratio:.4f} (区间: {lower_bound:.2f}-{upper_bound:.2f})")
            return float(ratio)
        except Exception as e:
            print(f"❌ [calculate_main_cost_range_ratio] 计算失败: {e}，返回默认值0.5")
            return 0.5
    @staticmethod
    def calculate_high_position_lock_ratio_90(chip_dist_data: pd.DataFrame, current_price: float) -> float:
        """
        基于筹码分布数据计算90%分位以上高位筹码沉淀比例 (high_position_lock_ratio_90)
        逻辑: 计算价格高于“当前价格90%分位”的筹码占比，用于反映高位套牢盘。
             如果当前价格很低，则计算价格高于“筹码分布价格90%分位”的筹码占比。
        """
        try:
            if chip_dist_data.empty or current_price <= 0:
                print(f"⚠️ [calculate_high_position_lock_ratio_90] 输入数据无效，返回默认值0.0")
                return 0.0
            # 方法1: 基于当前价格计算90%分位阈值
            price_threshold_1 = current_price * 1.0  # 当前价格的100%作为阈值（即高于当前价的筹码）
            # 方法2: 基于筹码分布自身计算90%价格分位
            sorted_chips = chip_dist_data.sort_values('price')
            cum_pct = sorted_chips['percent'].cumsum()
            # 找到累计占比达到90%的价格点
            if (cum_pct >= 0.9).any():
                idx_90 = (cum_pct >= 0.9).idxmax()
                price_threshold_2 = sorted_chips.loc[idx_90, 'price']
            else:
                price_threshold_2 = sorted_chips['price'].max()
            # 使用两者中较高的作为阈值，更严格反映“高位”
            price_threshold = max(price_threshold_1, price_threshold_2)
            # 计算高于阈值的筹码占比
            high_mask = chip_dist_data['price'] >= price_threshold
            if not high_mask.any():
                print(f"⚠️ [calculate_high_position_lock_ratio_90] 无筹码高于阈值{price_threshold:.2f}，返回0")
                return 0.0
            high_chip_sum = chip_dist_data.loc[high_mask, 'percent'].sum()
            total_chip_sum = chip_dist_data['percent'].sum()
            if total_chip_sum <= 0:
                return 0.0
            ratio = high_chip_sum / total_chip_sum
            print(f"📊 [calculate_high_position_lock_ratio_90] 计算完成: {ratio:.4f} (阈值: {price_threshold:.2f}, 当前价: {current_price:.2f})")
            return float(ratio)
        except Exception as e:
            print(f"❌ [calculate_high_position_lock_ratio_90] 计算失败: {e}，返回默认值0.0")
            return 0.0











