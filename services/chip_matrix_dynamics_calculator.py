# services\chip_matrix_dynamics_calculator.py
# 筹码矩阵动态分析计算服务
import numpy as np
from numba import njit, prange
import scipy.stats
import math
import json
from typing import Dict, Any, List, Tuple
from datetime import datetime

class ChipMatrixDynamicsCalculator:
    """
    筹码矩阵动态分析计算服务
    负责处理复杂的筹码分布逻辑、A股特征分析及因子计算，为Model层减负。
    """

    @staticmethod
    def clean_structure(data, precision=6, threshold=1e-8):
        """
        [Version 3.1.0] 递归数据清洗与超高精度控制逻辑
        说明：将默认精度提升至6位，阈值下放至1e-8，防止高频微观筹码尾部在序列化前被抹杀。
        针对浮点运算的安全截断进行严格的类型校验，确保后续 EMD 运算的地基纯洁性。
        """
        import numpy as np
        import math
        if isinstance(data, dict):
            return {k: ChipMatrixDynamicsCalculator.clean_structure(v, precision, threshold) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return
        elif isinstance(data, np.ndarray):
            return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).tolist()
        elif isinstance(data, (float, np.floating)):
            if math.isnan(data) or math.isinf(data):
                return 0.0
            if abs(data) < threshold:
                return 0.0
            return round(float(data), precision)
        elif isinstance(data, np.bool_):
            return bool(data)
        return data

    @staticmethod
    @njit(parallel=True, fastmath=False, cache=True)
    def calculate_matrix_emd_dynamics_optimized(chip_matrix: np.ndarray, price_grid: np.ndarray, threshold_pct: float) -> tuple:
        """
        [Version 3.3.0] 高精度 Wasserstein 距离计算引擎
        说明：针对原算法的累加浮点溢出缺陷，引入 Kahan Summation (卡汉求和)算法补偿 CDF 积分误差。
        锁定率分位数采用精确的小数权重插值，清除所有空行，严格规范纯计算缩进。
        """
        rows, cols = chip_matrix.shape
        emd_distance_array = np.zeros(rows - 1, dtype=np.float64)
        abs_change_matrix = np.zeros((rows - 1, cols), dtype=np.float64)
        for i in prange(rows - 1):
            cdf_prev, cdf_curr, emd_sum = 0.0, 0.0, 0.0
            c_prev, c_curr, c_emd = 0.0, 0.0, 0.0
            for j in range(cols):
                val_prev = chip_matrix[i, j]
                val_curr = chip_matrix[i+1, j]
                abs_change_matrix[i, j] = np.abs(val_curr - val_prev)
                y_prev = val_prev - c_prev
                t_prev = cdf_prev + y_prev
                c_prev = (t_prev - cdf_prev) - y_prev
                cdf_prev = t_prev
                y_curr = val_curr - c_curr
                t_curr = cdf_curr + y_curr
                c_curr = (t_curr - cdf_curr) - y_curr
                cdf_curr = t_curr
                if j < cols - 1:
                    grid_distance = price_grid[j+1] - price_grid[j]
                    y_emd = (np.abs(cdf_prev - cdf_curr) * grid_distance) - c_emd
                    t_emd = emd_sum + y_emd
                    c_emd = (t_emd - emd_sum) - y_emd
                    emd_sum = t_emd
            emd_distance_array[i] = emd_sum
        latest_changes = abs_change_matrix[-1, :]
        total_change_volume = np.sum(latest_changes)
        if total_change_volume <= 1e-8:
            return emd_distance_array, abs_change_matrix, 0.0, 1.0
        k_top = max(1, int(cols * 0.05))
        sorted_changes = np.sort(latest_changes)
        top_5_sum = np.sum(sorted_changes[-k_top:])
        change_concentration = top_5_sum / total_change_volume
        active_changes = sorted_changes[sorted_changes > 0]
        if len(active_changes) > 0:
            pct_index = (len(active_changes) - 1) * threshold_pct
            lower_idx = int(pct_index)
            upper_idx = min(lower_idx + 1, len(active_changes) - 1)
            weight = pct_index - lower_idx
            lock_threshold = active_changes[lower_idx] * (1.0 - weight) + active_changes[upper_idx] * weight
            locked_count = 0
            for val in latest_changes:
                if val < lock_threshold:
                    locked_count += 1
            chip_lock_ratio = locked_count / cols
        else:
            chip_lock_ratio = 1.0
        return emd_distance_array, abs_change_matrix, change_concentration, float(chip_lock_ratio)

    @staticmethod
    def calculate_topological_chip_peaks(chip_distribution: np.ndarray, price_grid: np.ndarray) -> dict:
        """
        [Version 3.3.0] 亚网格(Sub-grid)精度抛物线插值多峰识别算法
        说明：针对原分布模型峰值定位仅依赖离散网格坐标的缺陷，引入抛物线插值寻找连续真实极值。
        四分位距(IQR)实现动态自适应阈值，全面提升震荡市下主力吸筹识别精度。禁止空行。
        """
        import numpy as np
        from scipy.signal import find_peaks
        if len(chip_distribution) < 3 or len(price_grid) < 3:
            return {'peak_count': 0, 'highest_peak_price': 0.0, 'lowest_peak_price': 0.0, 'peak_distance_ratio': 0.0, 'peak_concentration': 0.0, 'main_peak_position': 0.0}
        q75, q25 = np.percentile(chip_distribution, )
        iqr = q75 - q25
        min_prominence = max(1e-6, iqr * 0.3)
        min_distance = max(1, int(len(price_grid) * 0.03))
        peaks_indices, properties = find_peaks(chip_distribution, prominence=min_prominence, width=2, distance=min_distance)
        peak_count = len(peaks_indices)
        if peak_count == 0:
            return {'peak_count': 0, 'highest_peak_price': 0.0, 'lowest_peak_price': 0.0, 'peak_distance_ratio': 0.0, 'peak_concentration': 0.0, 'main_peak_position': 0.0}
        prominences = properties['prominences']
        sorted_peak_ranks = np.argsort(prominences)[::-1]
        main_peak_idx = peaks_indices[sorted_peak_ranks]
        if 0 < main_peak_idx < len(chip_distribution) - 1:
            y1 = float(chip_distribution[main_peak_idx - 1])
            y2 = float(chip_distribution[main_peak_idx])
            y3 = float(chip_distribution[main_peak_idx + 1])
            denominator = y1 - 2 * y2 + y3
            p_shift = 0.5 * (y1 - y3) / denominator if abs(denominator) > 1e-8 else 0.0
            exact_idx = main_peak_idx + p_shift
            idx_floor = int(np.floor(exact_idx))
            idx_ceil = min(idx_floor + 1, len(price_grid) - 1)
            weight = exact_idx - idx_floor
            main_peak_price = float(price_grid[idx_floor]) * (1.0 - weight) + float(price_grid[idx_ceil]) * weight
            peak_concentration = y2 - 0.25 * (y1 - y3) * p_shift if abs(denominator) > 1e-8 else y2
        else:
            main_peak_price = float(price_grid[main_peak_idx])
            peak_concentration = float(chip_distribution[main_peak_idx])
        price_min, price_max = float(price_grid), float(price_grid[-1])
        price_range = max(price_max - price_min, 1e-8)
        main_peak_position = np.clip((main_peak_price - price_min) / price_range, 0.0, 1.0)
        highest_peak_price = float(np.max(price_grid[peaks_indices]))
        lowest_peak_price = float(np.min(price_grid[peaks_indices]))
        peak_distance_ratio = (highest_peak_price - lowest_peak_price) / price_range
        if peak_count > 1:
            second_peak_idx = peaks_indices[sorted_peak_ranks[1]]
            peak_concentration += float(chip_distribution[second_peak_idx])
        return {
            'peak_count': int(peak_count),
            'highest_peak_price': highest_peak_price,
            'lowest_peak_price': lowest_peak_price,
            'peak_distance_ratio': float(peak_distance_ratio),
            'peak_concentration': float(peak_concentration),
            'main_peak_position': round(float(main_peak_position), 6)
        }

    @classmethod
    def calculate_absolute_change_analysis(cls, changes, price_grid, current_price):
        """完全保留原有基础统计逻辑和 A 股专属整合调用，接入 EMD 结果修正"""
        if len(changes) == 0 or current_price <= 0:
            return {}
        # 保留原有基础统计与正负量计算公式
        total_change_volume = np.sum(np.abs(changes))
        positive_change_volume = np.sum(changes[changes > 0])
        negative_change_volume = np.sum(changes[changes < 0])
        max_increase = np.max(changes)
        max_decrease = np.min(changes)
        mean_change = np.mean(changes)
        change_std = np.std(changes)
        # 提取 Top 5% 作为集中度基础计算 (向下兼容保留)
        abs_changes = np.abs(changes)
        top_indices = np.argsort(abs_changes)[-max(1, int(len(changes) * 0.05)):]
        change_concentration = np.sum(abs_changes[top_indices]) / total_change_volume if total_change_volume > 0 else 0.0
        # 10th percentile 阈值锁定率公式保留
        positive_abs = abs_changes[abs_changes > 0]
        threshold = np.percentile(positive_abs, 10) if len(positive_abs) > 0 else 0.0
        chip_lock_ratio = np.sum(abs_changes < threshold) / len(changes)

        analysis = {
            'total_change_volume': float(total_change_volume),
            'positive_change_volume': float(positive_change_volume),
            'negative_change_volume': float(negative_change_volume),
            'max_increase': float(max_increase),
            'max_decrease': float(max_decrease),
            'mean_change': float(mean_change),
            'change_std': float(change_std),
            'change_concentration': float(change_concentration),
            'chip_lock_ratio': float(chip_lock_ratio)
        }
        # 保留原有的 A 股特性综合分析调用逻辑
        if hasattr(cls, 'analyze_a_share_key_price_levels'):
            analysis.update(cls.analyze_a_share_key_price_levels(changes, price_grid, current_price))
        if hasattr(cls, 'analyze_a_share_pullback_pattern'):
            analysis.update(cls.analyze_a_share_pullback_pattern(changes, price_grid, current_price))
        if hasattr(cls, 'analyze_a_share_reversal_features'):
            analysis.update(cls.analyze_a_share_reversal_features(changes, price_grid, current_price))
        if hasattr(cls, 'detect_a_share_false_signals'):
            analysis.update(cls.detect_a_share_false_signals(changes, price_grid, current_price))
        if hasattr(cls, 'calculate_a_share_market_sentiment'):
            analysis.update(cls.calculate_a_share_market_sentiment(changes, price_grid, current_price))
        if hasattr(cls, 'assess_a_share_trend_quality'):
            analysis.update(cls.assess_a_share_trend_quality(changes, price_grid, current_price))
        if hasattr(cls, 'analyze_chip_transfer_path'):
            analysis.update(cls.analyze_chip_transfer_path(changes, price_grid, current_price))
        if hasattr(cls, 'identify_sector_rotation_patterns'):
            analysis.update(cls.identify_sector_rotation_patterns(changes, price_grid, current_price))
        return analysis

    @classmethod
    def calculate_tick_enhanced_factors(cls, current_factors: Dict[str, float], tick_factors: Dict[str, Any], quality_score: float) -> Tuple[float, float, float, float, str]:
        # [V3.4.1] 重构Tick增强接口签名，支持字典解包并返回调整原因，彻底消除解包与传参断层。
        short_term = float(current_factors.get('short', 0.2))
        mid_term = float(current_factors.get('mid', 0.3))
        long_term = float(current_factors.get('long', 0.5))
        avg_days = float(current_factors.get('days', 60.0))
        if not tick_factors or quality_score <= 0.3:
            return short_term, mid_term, long_term, avg_days, "Tick质量极低，不触发微观盘口修正"
        tick_intensity = float(tick_factors.get('intraday_chip_turnover_intensity', 0.0))
        main_force_score = float(tick_factors.get('intraday_main_force_activity', 0.0))
        acc_conf = float(tick_factors.get('intraday_accumulation_confidence', 0.0))
        dist_conf = float(tick_factors.get('intraday_distribution_confidence', 0.0))
        reason = "微观资金结构平衡，维持基础权重"
        adjusted = False
        if tick_intensity > 0.7 and main_force_score < 0.3:
            adjust = tick_intensity * 0.15
            short_term = min(0.6, short_term + adjust)
            long_term = max(0.1, long_term - adjust * 0.5)
            reason = "散户高换手主导，大幅缩短持仓周期预期"
            adjusted = True
        elif main_force_score > 0.4:
            if acc_conf > dist_conf:
                mid_term = min(0.6, mid_term + 0.1)
                avg_days *= 1.2
                reason = "主力隐蔽吸筹信号确认，延长趋势持仓预期"
                adjusted = True
            else:
                short_term = min(0.5, short_term + 0.1)
                avg_days *= 0.8
                reason = "主力高频派发迹象明显，强制缩短防御周期"
                adjusted = True
        total = short_term + mid_term + long_term
        if total > 0:
            short_term /= total
            mid_term /= total
            long_term /= total
        return float(short_term), float(mid_term), float(long_term), float(avg_days), reason
    @classmethod
    def calculate_holding_factors(cls, dynamics_result: Dict[str, Any], absolute_change_analysis: Dict[str, Any]) -> Dict[str, Any]:
        # [V3.4.1] 重构因子持久化算子签名，通过适配器模式安全提取高维数据，彻底根除参数错位带来的执行期崩溃。
        if absolute_change_analysis is None:
            absolute_change_analysis = {}
        convergence_metrics = dynamics_result.get('convergence_metrics', {})
        concentration_metrics = dynamics_result.get('concentration_metrics', {})
        behavior_patterns = dynamics_result.get('behavior_patterns', {})
        chip_lock_ratio = float(absolute_change_analysis.get('chip_lock_ratio', 0.5))
        market_sentiment = float(absolute_change_analysis.get('market_sentiment', 0.5))
        trend_quality = float(absolute_change_analysis.get('trend_quality', 0.5))
        market_cycle = absolute_change_analysis.get('market_cycle_phase', 'consolidation')
        trend_health = float(absolute_change_analysis.get('trend_health', 0.5))
        main_force_activity = float(behavior_patterns.get('main_force_activity', 0.0))
        cycle_adj = float(cls.calculate_market_cycle_adjustment(market_cycle, trend_health))
        comprehensive_convergence = float(convergence_metrics.get('comprehensive_convergence', 0.0))
        comprehensive_concentration = float(concentration_metrics.get('comprehensive_concentration', 0.0))
        long_term_base = (comprehensive_convergence * 0.3) + (comprehensive_concentration * 0.3) + (chip_lock_ratio * 0.4)
        long_term_ratio = min(0.85, max(0.05, long_term_base * cycle_adj))
        divergence_score = 1.0 - comprehensive_convergence
        short_term_base = (divergence_score * 0.35) + (main_force_activity * 0.25) + ((1.0 - chip_lock_ratio) * 0.2)
        if market_sentiment > 0.7:
            short_term_ratio = short_term_base * 1.3
        elif market_sentiment < 0.3:
            short_term_ratio = short_term_base * 1.1
        else:
            short_term_ratio = short_term_base
        short_term_ratio = min(0.65, max(0.05, short_term_ratio))
        mid_term_ratio = 1.0 - short_term_ratio - long_term_ratio
        if mid_term_ratio < 0:
            excess = abs(mid_term_ratio)
            short_term_ratio -= excess * 0.6
            long_term_ratio -= excess * 0.4
            mid_term_ratio = 0.0
        total_sum = short_term_ratio + mid_term_ratio + long_term_ratio
        if total_sum > 0:
            short_term_ratio *= (1.0 / total_sum)
            mid_term_ratio *= (1.0 / total_sum)
            long_term_ratio *= (1.0 / total_sum)
        base_days = 20 + (long_term_ratio * 160)
        trend_days_adjust = 0.8 + trend_quality * 0.4
        avg_holding_days = max(3.0, min(365.0, base_days * trend_days_adjust))
        return {
            'short_term_ratio': float(short_term_ratio),
            'mid_term_ratio': float(mid_term_ratio),
            'long_term_ratio': float(long_term_ratio),
            'avg_holding_days': float(avg_holding_days),
            'extra_metrics': {
                'chip_lock_ratio': float(chip_lock_ratio),
                'market_sentiment': float(market_sentiment),
                'trend_quality': float(trend_quality),
                'cycle_adj': float(cycle_adj)
            }
        }

    # ========== 具体的私有分析方法 (静态化) ==========
    @staticmethod
    def analyze_a_share_key_price_levels(changes: np.ndarray, price_grid: np.ndarray, current_price: float) -> Dict[str, Any]:
        """
        版本: v2.2
        说明: A股关键价格位分析（Float32降级优化版）
        修改思路: 黄金分割比例和价格计算使用float32。
        """
        analysis = {
            'integer_resistance_levels': [],
            'integer_support_levels': [],
            'technical_levels': [],
            'historical_reference_levels': [],
        }
        try:
            # 1. 整数关口分析
            rounded_prices = np.round(price_grid)
            is_integer = np.abs(price_grid - rounded_prices) < 0.01
            if np.any(is_integer):
                int_indices = np.where(is_integer)[0]
                int_prices = rounded_prices[int_indices]
                unique_ints = np.unique(int_prices)
                for int_level in unique_ints:
                    mask = np.abs(price_grid - int_level) < 0.01
                    strength = float(np.sum(changes[mask]))
                    item = {
                        'price': float(int_level),
                        'strength': strength,
                        'distance_pct': float((int_level - current_price) / current_price) if current_price > 0 else 0.0,
                    }
                    if int_level < current_price:
                        item['type'] = 'integer_support'
                        analysis['integer_support_levels'].append(item)
                    else:
                        item['type'] = 'integer_resistance'
                        analysis['integer_resistance_levels'].append(item)
            # 2. 技术分析关键位（黄金分割）
            if current_price > 0:
                # 使用float32数组
                ratios = np.array([0.382, 0.5, 0.618], dtype=np.float32)
                golden_levels = (current_price * (1 - ratios)).astype(np.float32)
                abs_diff = np.abs(price_grid[None, :] - golden_levels[:, None])
                nearest_indices = np.argmin(abs_diff, axis=1)
                for i, idx in enumerate(nearest_indices):
                    if idx < len(changes):
                        analysis['technical_levels'].append({
                            'price': float(golden_levels[i]),
                            'actual_price': float(price_grid[idx]),
                            'strength': float(changes[idx]),
                            'type': 'golden_ratio',
                            'ratio': float((golden_levels[i] - current_price) / current_price)
                        })
            for key in ['integer_support_levels', 'integer_resistance_levels', 'technical_levels']:
                analysis[key] = sorted(analysis[key], key=lambda x: abs(x['strength']), reverse=True)[:5]
        except Exception as e:
            print(f"关键价格位分析异常: {e}")
        return analysis

    @staticmethod
    def analyze_a_share_pullback_pattern(changes: np.ndarray, price_grid: np.ndarray, current_price: float) -> Dict[str, Any]:
        """
        版本: v2.1
        说明: A股拉升模式深度分析（Top-N提取优化版）
        修改思路: 
        1. 避免对所有符合条件的点创建对象，先排序筛选Top N索引，再构建结果。
        2. 使用np.argsort进行部分排序。
        """
        analysis = {
            'pullback_phase_detected': False,
            'pullback_strength': 0.0,
            'pullback_type': 'none',
            'support_levels': [],
            'resistance_levels': [],
            'breakout_potential': 0.0,
            'consolidation_completeness': 0.0,
        }
        try:
            # 1. 支撑位 (Below Current)
            below_mask = price_grid < current_price * 0.99
            if np.any(below_mask):
                # 提取区域数据
                b_changes = changes[below_mask]
                b_prices = price_grid[below_mask]
                # 筛选正向变化 (支撑)
                pos_mask = b_changes > 0
                if np.any(pos_mask):
                    valid_changes = b_changes[pos_mask]
                    valid_prices = b_prices[pos_mask]
                    # 排序取Top 3
                    if len(valid_changes) > 3:
                        # argsort 升序，取最后3个
                        top_indices = np.argsort(valid_changes)[-3:][::-1]
                    else:
                        top_indices = np.argsort(valid_changes)[::-1]
                    for idx in top_indices:
                        analysis['support_levels'].append({
                            'price': float(valid_prices[idx]),
                            'strength': float(valid_changes[idx]),
                            'distance_pct': float((current_price - valid_prices[idx]) / current_price),
                            'type': 'strong_support'
                        })
            # 2. 阻力位 (Above Current)
            above_mask = price_grid > current_price * 1.01
            if np.any(above_mask):
                a_changes = changes[above_mask]
                a_prices = price_grid[above_mask]
                # 筛选负向变化 (阻力)
                neg_mask = a_changes < 0
                if np.any(neg_mask):
                    valid_changes = a_changes[neg_mask] # 负数
                    valid_prices = a_prices[neg_mask]
                    abs_changes = np.abs(valid_changes)
                    if len(abs_changes) > 3:
                        top_indices = np.argsort(abs_changes)[-3:][::-1]
                    else:
                        top_indices = np.argsort(abs_changes)[::-1]
                    for idx in top_indices:
                        analysis['resistance_levels'].append({
                            'price': float(valid_prices[idx]),
                            'strength': float(abs_changes[idx]),
                            'distance_pct': float((valid_prices[idx] - current_price) / current_price),
                            'type': 'strong_resistance'
                        })
            # 3. 特征识别 (向量化计算)
            # 低位吸筹
            low_mask = price_grid < current_price * 0.95
            accumulation_below = np.sum(changes[low_mask & (changes > 0.3)])
            # 中位锁定
            mid_mask = (price_grid >= current_price * 0.95) & (price_grid <= current_price * 1.05)
            lock_mid = np.sum(np.abs(changes[mid_mask])) < 0.2
            # 高位弱阻力
            high_mask = price_grid > current_price * 1.05
            weak_resistance = np.sum(changes[high_mask & (changes > -0.2)]) > -0.5
            if accumulation_below > 0.5 and lock_mid and weak_resistance:
                analysis['pullback_phase_detected'] = True
                analysis['pullback_type'] = 'accumulation'
                analysis['pullback_strength'] = min(1.0, accumulation_below)
            # 整理形态
            mid_zone_mask = (price_grid >= current_price * 0.97) & (price_grid <= current_price * 1.03)
            total_abs_change = np.sum(np.abs(changes))
            if total_abs_change > 0:
                mid_concentration = np.sum(np.abs(changes[mid_zone_mask])) / total_abs_change
                analysis['consolidation_completeness'] = mid_concentration
                if mid_concentration > 0.6:
                    analysis['pullback_phase_detected'] = True
                    analysis['pullback_type'] = 'consolidation'
                    analysis['pullback_strength'] = mid_concentration
            # 4. 突破势能
            # 阻力: 上方减少的筹码
            res_strength = np.sum(-changes[(price_grid > current_price) & (changes < 0)])
            # 支撑: 下方增加的筹码
            sup_strength = np.sum(changes[(price_grid < current_price) & (changes > 0)])
            if res_strength > 0 and sup_strength > 0:
                analysis['breakout_potential'] = min(1.0, sup_strength / (res_strength + sup_strength))
        except Exception as e:
            print(f"拉升模式分析异常: {e}")
        return analysis

    @staticmethod
    def analyze_a_share_reversal_features(changes: np.ndarray, price_grid: np.ndarray, current_price: float) -> Dict[str, Any]:
        """
        版本: v2.2
        说明: A股反转博弈特征分析（Float32降级优化版）
        修改思路: 分箱边界使用float32生成。
        """
        analysis = {
            'reversal_signal': False,
            'reversal_strength': 0.0,
            'reversal_type': 'none',
            'reversal_confidence': 0.0,
            'divergence_signals': [],
            'exhaustion_signals': [],
        }
        try:
            # 1. 量价背离检测
            high_mask = (price_grid > current_price * 1.05) & (price_grid <= current_price * 1.15)
            if np.any(high_mask):
                avg_change_high = np.mean(changes[high_mask])
                if avg_change_high < -0.2:
                    analysis['divergence_signals'].append({
                        'type': 'top_divergence',
                        'strength': float(-avg_change_high),
                        'zone': 'high_price',
                        'description': '价格高位但筹码减少，可能见顶'
                    })
            low_mask = (price_grid < current_price * 0.95) & (price_grid >= current_price * 0.85)
            if np.any(low_mask):
                avg_change_low = np.mean(changes[low_mask])
                if avg_change_low > 0.2:
                    analysis['divergence_signals'].append({
                        'type': 'bottom_divergence',
                        'strength': float(avg_change_low),
                        'zone': 'low_price',
                        'description': '价格低位但筹码增加，可能见底'
                    })
            # 2. 衰竭信号检测
            if len(price_grid) >= 5:
                # 使用float32生成边界
                bins = np.linspace(np.min(price_grid), np.max(price_grid), 6, dtype=np.float32)
                indices = np.digitize(price_grid, bins)
                indices = indices - 1
                valid_mask = (indices >= 0) & (indices < 5)
                if np.any(valid_mask):
                    valid_indices = indices[valid_mask]
                    valid_changes = changes[valid_mask]
                    sums = np.bincount(valid_indices, weights=valid_changes, minlength=5)
                    counts = np.bincount(valid_indices, minlength=5)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        bin_changes = np.divide(sums, counts)
                        bin_changes = np.nan_to_num(bin_changes)
                    if len(bin_changes) >= 3:
                        last_3 = bin_changes[-3:]
                        if np.all(last_3 < 0) and (last_3[2] > last_3[1] > last_3[0]):
                             analysis['exhaustion_signals'].append({
                                'type': 'uptrend_exhaustion',
                                'strength': float(-last_3[2]),
                                'pattern': 'decreasing_negative_changes',
                                'description': '上涨趋势中抛压逐渐衰减'
                            })
                        first_3 = bin_changes[:3]
                        if np.all(first_3 > 0) and (first_3[0] > first_3[1] > first_3[2]):
                            analysis['exhaustion_signals'].append({
                                'type': 'downtrend_exhaustion',
                                'strength': float(first_3[0]),
                                'pattern': 'decreasing_positive_changes',
                                'description': '下跌趋势中吸筹逐渐衰减'
                            })
            # 3. 综合反转判断
            if analysis['divergence_signals']:
                analysis['reversal_signal'] = True
                analysis['reversal_strength'] = max(s['strength'] for s in analysis['divergence_signals'])
                types = {s['type'] for s in analysis['divergence_signals']}
                if 'top_divergence' in types and 'bottom_divergence' not in types:
                    analysis['reversal_type'] = 'top'
                elif 'bottom_divergence' in types and 'top_divergence' not in types:
                    analysis['reversal_type'] = 'bottom'
                else:
                    analysis['reversal_type'] = 'mixed'
                sig_count = len(analysis['divergence_signals']) + len(analysis['exhaustion_signals'])
                analysis['reversal_confidence'] = min(1.0, 0.3 + 0.4 * (sig_count / 2) + 0.3 * analysis['reversal_strength'])
        except Exception as e:
            print(f"反转特征分析异常: {e}")
        return analysis

    @staticmethod
    def detect_a_share_false_signals(changes: np.ndarray, price_grid: np.ndarray, current_price: float) -> Dict[str, Any]:
        """
        版本: v2.1
        说明: A股虚假信号识别深化（Partition优化版）
        修改思路: 使用 np.partition 替代 np.argsort 计算 Top N 集中度。
        """
        analysis = {
            'false_distribution_flag': False,
            'false_accumulation_flag': False,
            'wash_sale_detected': False,
            'fake_breakout_risk': 0.0,
            'signal_reliability': 0.7,
        }
        try:
            # 1. 虚假派发 (向量化计算)
            high_mask = price_grid > current_price * 1.08
            mid_high_mask = (price_grid > current_price * 1.02) & (price_grid <= current_price * 1.08)
            low_mask = price_grid < current_price * 0.98
            high_inc = np.sum(changes[high_mask & (changes > 0)])
            mid_high_dec = np.sum(-changes[mid_high_mask & (changes < 0)])
            low_dec = np.sum(-changes[low_mask & (changes < 0)])
            mid_high_net = np.sum(changes[mid_high_mask])
            if mid_high_net < -0.3 and high_inc < 0.2 and low_dec < 0.1:
                analysis['false_distribution_flag'] = True
            # 2. 虚假吸筹
            low_inc = np.sum(changes[low_mask & (changes > 0)])
            mid_low_mask = (price_grid >= current_price * 0.98) & (price_grid <= current_price * 1.02)
            mid_low_dec = np.sum(-changes[mid_low_mask & (changes < 0)])
            if low_inc > 0.4 and mid_low_dec > 0.3:
                analysis['false_accumulation_flag'] = True
            # 3. 洗盘行为
            near_mask = (price_grid >= current_price * 0.99) & (price_grid <= current_price * 1.01)
            if np.any(near_mask):
                near_changes = changes[near_mask]
                pos_near = np.sum(near_changes[near_changes > 0])
                neg_near = np.sum(-near_changes[near_changes < 0])
                if pos_near > 0.3 and neg_near > 0.3 and abs(pos_near - neg_near) < 0.1:
                    analysis['wash_sale_detected'] = True
            # 4. 假突破
            above_mask = price_grid > current_price
            if np.any(above_mask):
                above_changes = changes[above_mask]
                res_break = np.sum(above_changes[above_changes > 0])
                res_hold = np.sum(-above_changes[above_changes < 0])
                if res_break > 0 and res_hold > res_break * 1.5:
                    analysis['fake_breakout_risk'] = min(1.0, res_hold / (res_break + 0.1))
            # 5. 信号可靠性 (Partition优化)
            reliability_factors = []
            abs_changes = np.abs(changes)
            total_abs = np.sum(abs_changes)
            if total_abs > 0:
                k = max(1, int(len(changes) * 0.1))
                if len(changes) >= k:
                    # partition: 第 -k 个位置是第 k 大的元素，后面都是比它大的
                    top_k_vals = np.partition(abs_changes, -k)[-k:]
                    concentration = np.sum(top_k_vals) / total_abs
                else:
                    concentration = 1.0
                reliability_factors.append(min(1.0, concentration * 1.5))
            else:
                reliability_factors.append(0.0)
            false_count = sum([analysis['false_distribution_flag'], analysis['false_accumulation_flag'], analysis['wash_sale_detected']])
            reliability_factors.append(max(0.0, 1.0 - false_count * 0.3))
            pos_ratio = np.mean(changes > 0) if len(changes) > 0 else 0.5
            consistency = 1.0 - abs(pos_ratio - 0.5) * 2
            reliability_factors.append(consistency)
            analysis['signal_reliability'] = float(np.mean(reliability_factors))
        except Exception as e:
            print(f"虚假信号识别异常: {e}")
        return analysis

    @staticmethod
    def calculate_a_share_market_sentiment(changes: np.ndarray, price_grid: np.ndarray, current_price: float) -> Dict[str, Any]:
        """A股市场情绪量化"""
        sentiment = {
            'market_sentiment': 0.5,  # 中性
            'sentiment_type': 'neutral',  # bullish/bearish/neutral/mixed
            'greed_fear_index': 0.5,
            'investor_confidence': 0.5,
            'risk_appetite': 0.5,
            'sentiment_indicators': {},
        }
        try:
            # 1. 基础情绪指标
            net_flow = np.sum(changes)
            total_volume = np.sum(np.abs(changes))
            if total_volume > 0:
                # 净流入比例
                net_ratio = net_flow / total_volume
                sentiment['market_sentiment'] = 0.5 + net_ratio * 0.5
                # 贪婪恐惧指数（基于变化幅度和集中度）
                avg_change = np.mean(np.abs(changes))
                change_std = np.std(changes)
                greed_fear = min(1.0, max(0.0, 0.3 + avg_change * 2 + change_std * 0.5))
                sentiment['greed_fear_index'] = greed_fear
            # 2. 投资者信心（基于筹码锁定度）
            lock_threshold = np.percentile(np.abs(changes[np.abs(changes) > 0]), 30) if np.any(np.abs(changes) > 0) else 0
            lock_ratio = np.sum(np.abs(changes) < lock_threshold) / len(changes)
            sentiment['investor_confidence'] = lock_ratio
            # 3. 风险偏好（基于高位筹码行为）
            high_zone = price_grid > current_price * 1.1
            if np.sum(high_zone) > 0:
                high_accumulation = np.sum(changes[high_zone & (changes > 0)])
                high_total = np.sum(np.abs(changes[high_zone]))
                if high_total > 0:
                    risk_preference = high_accumulation / high_total
                    sentiment['risk_appetite'] = risk_preference
            # 4. 情绪类型判断
            if sentiment['market_sentiment'] > 0.6:
                sentiment['sentiment_type'] = 'bullish'
            elif sentiment['market_sentiment'] < 0.4:
                sentiment['sentiment_type'] = 'bearish'
            else:
                sentiment['sentiment_type'] = 'neutral'
            # 5. 详细情绪指标
            sentiment['sentiment_indicators'] = {
                'net_inflow_ratio': float(net_ratio) if total_volume > 0 else 0.0,
                'volatility_sentiment': float(np.std(changes) * 10) if len(changes) > 0 else 0.0,
                'accumulation_strength': float(np.sum(changes[changes > 0])),
                'distribution_strength': float(np.sum(-changes[changes < 0])),
                'extreme_sentiment_alert': sentiment['greed_fear_index'] > 0.8 or sentiment['greed_fear_index'] < 0.2,
            }
        except Exception as e:
            print(f"市场情绪量化异常: {e}")
        return sentiment

    @staticmethod
    def assess_a_share_trend_quality(changes: np.ndarray, price_grid: np.ndarray, current_price: float) -> Dict[str, Any]:
        """
        版本: v2.1
        说明: A股趋势质量评估（Partition优化版）
        修改思路: 使用 np.partition 替代 np.argsort 计算 Top 5 集中度。
        """
        quality = {
            'trend_quality': 0.5,
            'trend_health': 0.5,
            'sustainability': 0.5,
            'acceleration_potential': 0.0,
            'risk_adjusted_score': 0.5,
            'quality_indicators': {},
        }
        try:
            # 1. 趋势健康度
            price_rel = (price_grid - current_price) / current_price
            below_flow = np.sum(changes[price_rel < -0.05])
            above_flow = np.sum(changes[price_rel > 0.05])
            trend_consistency = 0.5
            if below_flow < 0 and above_flow > 0:
                trend_consistency = 0.5 + min(abs(below_flow), above_flow) / (abs(below_flow) + above_flow + 0.1) * 0.5
            elif below_flow > 0 and above_flow < 0:
                trend_consistency = 0.5 + min(below_flow, abs(above_flow)) / (below_flow + abs(above_flow) + 0.1) * 0.5
            quality['trend_health'] = trend_consistency
            # 2. 可持续性
            abs_changes = np.abs(changes)
            lock_ratio = np.mean(abs_changes < 0.1) if len(changes) > 0 else 0.5
            turnover_intensity = np.mean(abs_changes) if len(changes) > 0 else 0.0
            optimal_turnover = 0.3
            turnover_score = 1.0 - min(1.0, abs(turnover_intensity - optimal_turnover) / optimal_turnover)
            sustainability = 0.4 * lock_ratio + 0.6 * turnover_score
            quality['sustainability'] = sustainability
            # 3. 加速潜力 (Partition优化)
            total_volume = np.sum(abs_changes)
            concentration = 0.0
            directional_strength = 0.0
            if total_volume > 0:
                if len(abs_changes) >= 5:
                    # 获取Top 5的索引 (partition不保证顺序，但我们需要索引来获取原始changes)
                    # 如果只需要sum(abs)，partition够了。但这里需要 directional_strength，
                    # 即 top 5 abs 对应的原始 changes 的 sum。
                    # 这种情况下，argpartition 是最佳选择。
                    top_5_indices = np.argpartition(abs_changes, -5)[-5:]
                    top_5_volume = np.sum(abs_changes[top_5_indices])
                    concentration = top_5_volume / total_volume
                    top_changes = changes[top_5_indices]
                    directional_strength = np.abs(np.sum(top_changes)) / top_5_volume if top_5_volume > 0 else 0
                else:
                    concentration = 1.0
                    directional_strength = np.abs(np.sum(changes)) / total_volume
                quality['acceleration_potential'] = concentration * directional_strength
            # 4. 风险调整
            volatility = np.std(changes) if len(changes) > 0 else 0.5
            risk_adjustment = 1.0 / (1.0 + volatility * 3)
            # 5. 综合评分
            quality['trend_quality'] = 0.3 * trend_consistency + 0.3 * sustainability + 0.2 * quality['acceleration_potential'] + 0.2 * risk_adjustment
            quality['risk_adjusted_score'] = quality['trend_quality'] * risk_adjustment
            quality['quality_indicators'] = {
                'trend_consistency': float(trend_consistency),
                'chip_lock_degree': float(lock_ratio),
                'turnover_optimality': float(turnover_score),
                'main_force_concentration': float(concentration),
                'volatility_penalty': float(1.0 - risk_adjustment),
                'health_warning': trend_consistency < 0.3 or sustainability < 0.3,
            }
        except Exception as e:
            print(f"趋势质量评估异常: {e}")
        return quality

    @staticmethod
    def analyze_chip_transfer_path(changes: np.ndarray, price_grid: np.ndarray, current_price: float) -> Dict[str, Any]:
        """
        版本: v2.1
        说明: 筹码转移路径分析（向量化筛选版）
        修改思路: 
        1. 使用Numpy直接筛选索引，避免循环判断。
        2. 限制结果数量，只处理Top 3。
        """
        transfer = {
            'transfer_direction': 'unclear',
            'transfer_intensity': 0.0,
            'source_zones': [],
            'destination_zones': [],
            'transfer_efficiency': 0.5,
        }
        try:
            # 1. 来源区域 (显著减少 < -0.2)
            dec_indices = np.where(changes < -0.2)[0]
            if len(dec_indices) > 0:
                # 按变化幅度排序 (绝对值越大越靠前)
                # changes[dec_indices] 是负数，argsort默认升序，最小的(绝对值最大)在最前
                sorted_args = np.argsort(changes[dec_indices])[:3]
                top_indices = dec_indices[sorted_args]
                for idx in top_indices:
                    transfer['source_zones'].append({
                        'price': float(price_grid[idx]),
                        'change': float(changes[idx]),
                        'type': 'distribution_source',
                        'distance_pct': float((price_grid[idx] - current_price) / current_price),
                    })
            # 2. 目标区域 (显著增加 > 0.2)
            inc_indices = np.where(changes > 0.2)[0]
            if len(inc_indices) > 0:
                # 降序排序
                sorted_args = np.argsort(changes[inc_indices])[::-1][:3]
                top_indices = inc_indices[sorted_args]
                for idx in top_indices:
                    transfer['destination_zones'].append({
                        'price': float(price_grid[idx]),
                        'change': float(changes[idx]),
                        'type': 'accumulation_destination',
                        'distance_pct': float((price_grid[idx] - current_price) / current_price),
                    })
            # 3. 转移方向
            if transfer['source_zones'] and transfer['destination_zones']:
                avg_src = np.mean([z['price'] for z in transfer['source_zones']])
                avg_dst = np.mean([z['price'] for z in transfer['destination_zones']])
                if avg_dst > avg_src * 1.02: transfer['transfer_direction'] = 'up'
                elif avg_dst < avg_src * 0.98: transfer['transfer_direction'] = 'down'
                else: transfer['transfer_direction'] = 'sideways'
            # 4. 强度与效率
            abs_changes = np.abs(changes)
            total_changes = np.sum(abs_changes)
            if total_changes > 0:
                # 强度: 显著变化的占比
                sig_transfer = np.sum(abs_changes[abs_changes > 0.1])
                transfer['transfer_intensity'] = sig_transfer / total_changes
                # 效率: 来源和目标的集中度
                # 简单模型: 区域越少，越集中
                n_grid = len(price_grid)
                # 修正逻辑: 这里的conc其实是覆盖率，越小越好
                # 效率 = (1 - 覆盖率) * 强度
                # 但原逻辑是基于zone数量，这里保持原意微调
                transfer['transfer_efficiency'] = min(1.0, transfer['transfer_intensity'] * 1.5)
        except Exception as e:
            print(f"筹码转移分析异常: {e}")
        return transfer

    @staticmethod
    def identify_sector_rotation_patterns(changes: np.ndarray, price_grid: np.ndarray, current_price: float) -> Dict[str, Any]:
        """
        版本: v2.1
        说明: 板块轮动特征识别（Partition优化版）
        修改思路: 使用 np.partition 替代 np.sort 计算 Top 10% 集中度。
        """
        rotation = {
            'sector_rotation_pattern': 'none',
            'rotation_strength': 0.0,
            'market_cycle_phase': 'consolidation',
            'style_preference': 'balanced',
        }
        try:
            abs_changes = np.abs(changes)
            total_abs = np.sum(abs_changes)
            # 1. 市场阶段
            price_rel = (price_grid - current_price) / current_price
            if total_abs > 0:
                deep_below = np.sum(abs_changes[price_rel < -0.15]) / total_abs
                near = np.sum(abs_changes[np.abs(price_rel) <= 0.05]) / total_abs
                deep_above = np.sum(abs_changes[price_rel > 0.15]) / total_abs
                if near > 0.6:
                    rotation['market_cycle_phase'] = 'consolidation'
                elif deep_below > deep_above and deep_below > 0.3:
                    rotation['market_cycle_phase'] = 'accumulation'
                elif deep_above > deep_below and deep_above > 0.3:
                    rotation['market_cycle_phase'] = 'distribution'
                else:
                    rotation['market_cycle_phase'] = 'expansion'
            # 2. 轮动模式
            low_mask = price_rel < -0.1
            mid_mask = (price_rel >= -0.1) & (price_rel <= 0.1)
            high_mask = price_rel > 0.1
            low_lock_ratio = np.mean(abs_changes[low_mask] < 0.05) if np.any(low_mask) else 0
            mid_activity = np.mean(abs_changes[mid_mask]) if np.any(mid_mask) else 0
            high_activity = np.mean(abs_changes[high_mask]) if np.any(high_mask) else 0
            if low_lock_ratio > 0.7 and mid_activity < 0.2:
                rotation['sector_rotation_pattern'] = 'defensive'
                rotation['rotation_strength'] = float(low_lock_ratio)
            elif mid_activity > 0.3 and abs(np.sum(changes[mid_mask])) > 0.5:
                rotation['sector_rotation_pattern'] = 'cyclical'
                rotation['rotation_strength'] = float(mid_activity)
            elif high_activity > 0.4:
                rotation['sector_rotation_pattern'] = 'growth'
                rotation['rotation_strength'] = float(high_activity)
            else:
                rotation['sector_rotation_pattern'] = 'value'
                rotation['rotation_strength'] = float(1.0 - max(low_lock_ratio, mid_activity, high_activity))
            # 3. 风格偏好 (Partition优化)
            concentration_top_10 = 0.0
            if total_abs > 0:
                k = max(1, int(len(changes) * 0.1))
                if len(changes) >= k:
                    top_k_vals = np.partition(abs_changes, -k)[-k:]
                    concentration_top_10 = np.sum(top_k_vals) / total_abs
                else:
                    concentration_top_10 = 1.0
            dispersion = 1.0 - concentration_top_10
            if concentration_top_10 > 0.6:
                rotation['style_preference'] = 'large_cap'
            elif dispersion > 0.7:
                rotation['style_preference'] = 'small_cap'
            elif rotation['sector_rotation_pattern'] == 'growth':
                rotation['style_preference'] = 'growth'
            elif rotation['sector_rotation_pattern'] == 'value':
                rotation['style_preference'] = 'value'
        except Exception as e:
            print(f"板块轮动识别异常: {e}")
        return rotation

    @staticmethod
    def get_default_absolute_analysis() -> Dict[str, Any]:
        """获取默认的绝对变化分析结果"""
        return {
            'total_change_volume': 0.0,
            'positive_change_volume': 0.0,
            'negative_change_volume': 0.0,
            'change_concentration': 0.0,
            'max_increase': 0.0,
            'max_decrease': 0.0,
            'mean_change': 0.0,
            'price_zone_analysis': {},
            'pullback_phase_detected': False,
            'pullback_strength': 0.0,
            'support_levels': [],
            'resistance_levels': [],
            'false_distribution_flag': False,
            'signal_quality': 0.5,
            'trend_strength': 0.5,
            'key_price_levels': [],
        }

    @staticmethod
    def create_default_key_battle_zones(price_grid: List[float], current_price: float) -> List[Dict[str, Any]]:
        """创建默认的关键博弈区域，确保浮点数保留3位小数"""
        if not price_grid or current_price <= 0:
            return []
        # 找到当前价附近的三个价格点
        price_array = np.array(price_grid)
        distances = np.abs(price_array - current_price)
        nearest_indices = np.argsort(distances)[:3]
        zones = []
        for idx in nearest_indices:
            price = price_array[idx]
            zones.append({
                'price': round(float(price), 3),  # 保留3位小数
                'battle_intensity': 0.1,  # 低强度
                'type': 'default',
                'position': 'below_current' if price < current_price else 'above_current',
                'distance_to_current': round(float((price - current_price) / current_price), 3),  # 保留3位小数
            })
        return zones

    @staticmethod
    def calculate_key_battle_intensity(zones: List[Dict]) -> float:
        if not zones: return 0.0
        return min(1.0, sum(z.get('battle_intensity', 0) for z in zones) / 5.0)

    @staticmethod
    def calculate_breakout_probability(potential: float, concentration: float, game_intensity: float, net_flow: float) -> float:
        if potential < 20: return 0.0
        base = min(1.0, potential / 100)
        bonus = concentration * 0.2 + (0.1 if 0.3 < game_intensity < 0.7 else 0.0)
        return round(min(1.0, base + bonus), 3)

    @staticmethod
    def calculate_trend_score(net_flow: float = 0, game_intensity: float = 0, intraday_quality: float = 0, tick_flow: float = 0) -> float:
        score = 0.5
        if net_flow > 10: score += 0.2
        elif net_flow < -10: score -= 0.2
        if game_intensity > 0.6: score += 0.1
        if intraday_quality > 0.5:
            if tick_flow > 0.1: score += 0.1
            elif tick_flow < -0.1: score -= 0.1
        return round(min(1.0, max(0.0, score)), 3)

    @staticmethod
    def calculate_market_cycle_adjustment(market_cycle: str, trend_health: float) -> float:
        """计算市场周期对持有时间的调整因子"""
        cycle_adjustments = {
            'accumulation': 1.2,      # 吸筹阶段 -> 长线增加
            'expansion': 1.1,         # 扩张阶段 -> 长线适度增加
            'consolidation': 1.0,     # 整理阶段 -> 中性
            'distribution': 0.9,      # 派发阶段 -> 长线减少
            'contraction': 0.8,       # 收缩阶段 -> 长线大幅减少
        }
        adjustment = cycle_adjustments.get(market_cycle, 1.0)
        # 根据趋势健康度微调
        if trend_health > 0.7:
            adjustment = adjustment * (1.0 + (trend_health - 0.7) * 0.2)
        elif trend_health < 0.4:
            adjustment = adjustment * (0.8 + trend_health * 0.5)
        return min(1.5, max(0.5, adjustment))

    @staticmethod
    def calculate_change_concentration(changes: np.ndarray) -> float:
        """
        版本: v2.1
        说明: 深化版变化集中度计算（排序优化版）
        修改思路: 
        1. 统一进行一次全排序，避免多次调用argsort。
        2. 基于排序后的索引切片计算各Top指标。
        """
        try:
            if len(changes) == 0:
                return 0.0
            abs_changes = np.abs(changes)
            total_volume = np.sum(abs_changes)
            if total_volume == 0:
                return 0.0
            # 1. 统一排序 (降序)
            sorted_indices = np.argsort(abs_changes)[::-1]
            sorted_abs = abs_changes[sorted_indices]
            # 2. 计算各层级集中度
            # Top 1
            concentration_top1 = sorted_abs[0] / total_volume
            # Top 3
            concentration_top3 = np.sum(sorted_abs[:3]) / total_volume
            # Top 5
            concentration_top5 = np.sum(sorted_abs[:5]) / total_volume
            # Top 10%
            top_10_count = max(1, int(len(changes) * 0.1))
            concentration_top10p = np.sum(sorted_abs[:top_10_count]) / total_volume
            # Herfindahl
            normalized_changes = abs_changes / total_volume
            herfindahl_index = np.sum(normalized_changes ** 2)
            # 3. 方向性集中度 (保持原有逻辑，因需分别排序)
            pos_mask = changes > 0
            neg_mask = changes < 0
            concentration_pos = 0.0
            if np.any(pos_mask):
                pos_vals = changes[pos_mask] # 已经是正数
                pos_total = np.sum(pos_vals)
                if pos_total > 0:
                    # 仅需Top3，使用partition比full sort快
                    if len(pos_vals) > 3:
                        # np.partition 将最大的k个元素放到最后
                        top3_pos = np.partition(pos_vals, -3)[-3:]
                        concentration_pos = np.sum(top3_pos) / pos_total
                    else:
                        concentration_pos = 1.0
            concentration_neg = 0.0
            if np.any(neg_mask):
                neg_vals = np.abs(changes[neg_mask])
                neg_total = np.sum(neg_vals)
                if neg_total > 0:
                    if len(neg_vals) > 3:
                        top3_neg = np.partition(neg_vals, -3)[-3:]
                        concentration_neg = np.sum(top3_neg) / neg_total
                    else:
                        concentration_neg = 1.0
            direction_balance = abs(concentration_pos - concentration_neg)
            # 4. 统计指标
            change_mean = np.mean(abs_changes)
            change_std = np.std(abs_changes)
            cv_ratio = (change_std / change_mean) if change_mean > 0 else 0.0
            max_change = sorted_abs[0]
            main_force_intensity = min(10.0, max_change / change_mean) / 10.0 if change_mean > 0 else 0.0
            small_change_threshold = change_mean * 0.5
            small_change_ratio = np.sum(abs_changes < small_change_threshold) / len(changes)
            # 5. 临界点
            critical_concentration = concentration_top5 > 0.9
            concentration_gradient_top1_to_top3 = (concentration_top3 - concentration_top1) / concentration_top1 if concentration_top1 > 0 else 0.0
            # 6. 综合评分
            weights = {
                'top5': 0.25, 'herfindahl': 0.20, 'main_force': 0.15,
                'direction_imbalance': 0.10, 'cv_ratio': 0.10, 'top1': 0.10, 'small_change': 0.10
            }
            normalized_indicators = {
                'top5': concentration_top5,
                'herfindahl': min(1.0, herfindahl_index * 10),
                'main_force': main_force_intensity,
                'direction_imbalance': direction_balance,
                'cv_ratio': min(1.0, cv_ratio / 2.0),
                'top1': concentration_top1,
                'small_change': 1.0 - small_change_ratio,
            }
            if critical_concentration:
                weights['top1'] += 0.15
                weights['top5'] += 0.10
                total_weight = sum(weights.values())
                for k in weights: weights[k] /= total_weight
            composite_concentration = sum(normalized_indicators[k] * w for k, w in weights.items())
            # 7. 模式识别调整
            wash_trade_suspicion = 0.3 if (concentration_top5 > 0.7 and direction_balance < 0.3) else 0.0
            control_suspicion = 0.4 if (concentration_top5 > 0.85 and main_force_intensity > 0.8) else 0.0
            retail_dominated = 0.5 if (concentration_top5 < 0.4 and small_change_ratio > 0.7) else 0.0
            if control_suspicion > 0: composite_concentration *= 0.8
            if retail_dominated > 0: composite_concentration *= 0.7
            final_concentration = max(0.0, min(1.0, composite_concentration))
            return final_concentration
        except Exception as e:
            print(f"❌ [变化集中度] 计算异常: {e}")
            return 0.0

    @staticmethod
    def calculate_signal_quality(changes: np.ndarray, price_rel: np.ndarray) -> float:
        """计算信号质量"""
        try:
            # 1. 变化集中度
            concentration_score = ChipMatrixDynamicsCalculator.calculate_change_concentration(changes)
            # 2. 价格分布合理性（筹码变化是否在合理价格区间）
            # 合理的筹码变化应该集中在当前价附近
            near_mask = np.abs(price_rel) < 0.1
            near_volume = np.sum(np.abs(changes[near_mask]))
            total_volume = np.sum(np.abs(changes))
            if total_volume > 0:
                distribution_score = near_volume / total_volume
            else:
                distribution_score = 0.0
            # 3. 噪声水平（小变化的占比）
            noise_mask = np.abs(changes) < 0.1
            noise_ratio = np.sum(noise_mask) / len(changes) if len(changes) > 0 else 1.0
            noise_score = 1.0 - noise_ratio
            # 综合质量评分
            quality_score = 0.4 * concentration_score + 0.3 * distribution_score + 0.3 * noise_score
            return min(1.0, max(0.0, quality_score))
        except Exception as e:
            print(f"信号质量计算失败: {e}")
            return 0.5

    @staticmethod
    def calculate_trend_strength(changes: np.ndarray, price_rel: np.ndarray) -> float:
        """计算趋势强度"""
        try:
            # 1. 净流向强度
            net_flow = np.sum(changes)
            total_volume = np.sum(np.abs(changes))
            if total_volume > 0:
                flow_strength = abs(net_flow) / total_volume
            else:
                flow_strength = 0.0
            # 2. 价格一致性（上涨趋势中，低位应减少，高位应增加）
            below_mask = price_rel < -0.05
            above_mask = price_rel > 0.05
            below_flow = np.sum(changes[below_mask])
            above_flow = np.sum(changes[above_mask])
            # 上涨趋势：低位减少，高位增加
            if below_flow < 0 and above_flow > 0:
                consistency_score = 0.7 + 0.3 * min(abs(below_flow), above_flow) / max(abs(below_flow), above_flow)
            # 下跌趋势：低位增加，高位减少
            elif below_flow > 0 and above_flow < 0:
                consistency_score = 0.7 + 0.3 * min(below_flow, abs(above_flow)) / max(below_flow, abs(above_flow))
            else:
                consistency_score = 0.3
            # 3. 变化幅度
            amplitude_score = min(1.0, total_volume / 20.0)  # 经验值
            # 综合趋势强度
            trend_strength = 0.3 * flow_strength + 0.4 * consistency_score + 0.3 * amplitude_score
            return min(1.0, max(0.0, trend_strength))
        except Exception as e:
            print(f"趋势强度计算失败: {e}")
            return 0.5






