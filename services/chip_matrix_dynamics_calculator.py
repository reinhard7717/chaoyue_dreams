# services\chip_matrix_dynamics_calculator.py
# 筹码矩阵动态分析计算服务
import numpy as np
from numba import njit, prange
import scipy.stats
import math
import json
from typing import Dict, Any, List, Tuple
from datetime import datetime

class QuantitativeTelemetryProbe:
    """[Version 1.0.0] 工业级量化全链路探针输出组件"""
    @classmethod
    def emit(cls, module_name: str, method_name: str, raw_data: dict, calc_nodes: dict, final_score: dict) -> None:
        payload = {"module": module_name, "method": method_name, "raw_data": raw_data, "calc_nodes": calc_nodes, "final_score": final_score}
        try:
            pass
            # print(f"📡 [QUANT-PROBE] | {json.dumps(payload, ensure_ascii=False)}")
        except Exception:
            pass

class ChipMatrixDynamicsCalculator:
    """
    [Version 5.0.0] 筹码矩阵动态分析核心引擎 (稀疏感知与流形防御版)
    """

    @staticmethod
    def clean_structure(data, precision=6, threshold=1e-8):
        import numpy as np
        import math
        if isinstance(data, dict): return {k: ChipMatrixDynamicsCalculator.clean_structure(v, precision, threshold) for k, v in data.items()}
        elif isinstance(data, (list, tuple)): return [ChipMatrixDynamicsCalculator.clean_structure(v, precision, threshold) for v in data]
        elif isinstance(data, np.ndarray): return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).tolist()
        elif isinstance(data, (float, np.floating)):
            if math.isnan(data) or math.isinf(data) or abs(data) < threshold: return 0.0
            return round(float(data), precision)
        elif isinstance(data, np.bool_): return bool(data)
        return data

    @staticmethod
    @njit(parallel=True, fastmath=False, cache=True)
    def calculate_matrix_emd_dynamics_optimized(chip_matrix: np.ndarray, price_grid: np.ndarray, threshold_pct: float) -> tuple:
        """[修复] 增加 price_range 归一化，实现跨全市场的尺度不变性(Scale Invariance)"""
        rows, cols = chip_matrix.shape
        emd_distance_array = np.zeros(rows - 1, dtype=np.float64)
        abs_change_matrix = np.zeros((rows - 1, cols), dtype=np.float64)
        price_range = price_grid[cols - 1] - price_grid[0]
        if price_range <= 0.0: price_range = 1.0
        for i in prange(rows - 1):
            cdf_prev, cdf_curr, emd_sum = 0.0, 0.0, 0.0
            c_prev, c_curr, c_emd = 0.0, 0.0, 0.0
            for j in range(cols):
                val_prev = chip_matrix[i, j]
                val_curr = chip_matrix[i+1, j]
                abs_change_matrix[i, j] = np.abs(val_curr - val_prev)
                # Kahan Summation 积分误差补偿
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
            emd_distance_array[i] = emd_sum / price_range # [修复] 强制归一化
        latest_changes = abs_change_matrix[-1, :]
        total_change_volume = np.sum(latest_changes)
        if total_change_volume <= 1e-8: return emd_distance_array, abs_change_matrix, 0.0, 1.0
        k_top = max(1, int(cols * 0.05))
        sorted_changes = np.sort(latest_changes)
        change_concentration = np.sum(sorted_changes[-k_top:]) / total_change_volume
        active_changes = sorted_changes[sorted_changes > 0]
        if len(active_changes) > 0:
            pct_index = (len(active_changes) - 1) * threshold_pct
            lower_idx = int(pct_index)
            upper_idx = min(lower_idx + 1, len(active_changes) - 1)
            weight = pct_index - lower_idx
            lock_threshold = active_changes[lower_idx] * (1.0 - weight) + active_changes[upper_idx] * weight
            locked_count = 0
            for val in latest_changes:
                if val <= lock_threshold: locked_count += 1
            chip_lock_ratio = locked_count / cols
        else: chip_lock_ratio = 1.0
        return emd_distance_array, abs_change_matrix, change_concentration, float(chip_lock_ratio)

    @staticmethod
    def calculate_topological_chip_peaks(chip_distribution: np.ndarray, price_grid: np.ndarray) -> dict:
        """[修复] 使用 CDF 求解真实 IQR，废除对概率密度求 percentile 的荒谬操作"""
        from scipy.signal import find_peaks
        import numpy as np
        eps = np.finfo(np.float64).eps
        if len(chip_distribution) < 3 or len(price_grid) < 3:
            return {'peak_count': 0, 'highest_peak_price': 0.0, 'lowest_peak_price': 0.0, 'peak_distance_ratio': 0.0, 'peak_concentration': 0.0, 'main_peak_position': 0.0}
        cdf = np.cumsum(chip_distribution)
        total_density = float(cdf[-1])
        if total_density < eps:
            return {'peak_count': 0, 'highest_peak_price': 0.0, 'lowest_peak_price': 0.0, 'peak_distance_ratio': 0.0, 'peak_concentration': 0.0, 'main_peak_position': 0.0}
        norm_cdf = cdf / total_density
        idx_25 = min(int(np.searchsorted(norm_cdf, 0.25)), len(price_grid) - 1)
        idx_75 = min(int(np.searchsorted(norm_cdf, 0.75)), len(price_grid) - 1)
        price_iqr = max(float(price_grid[idx_75] - price_grid[idx_25]), eps)
        max_density = float(np.max(chip_distribution))
        min_prominence = max(1e-4, max_density * 0.15)
        min_distance = max(1, int(len(price_grid) * 0.03))
        peaks_indices, properties = find_peaks(chip_distribution, prominence=min_prominence, width=2, distance=min_distance)
        peak_count = len(peaks_indices)
        if peak_count == 0:
            return {'peak_count': 0, 'highest_peak_price': 0.0, 'lowest_peak_price': 0.0, 'peak_distance_ratio': 0.0, 'peak_concentration': 0.0, 'main_peak_position': 0.0}
        prominences = properties['prominences']
        sorted_peak_ranks = np.argsort(prominences)[::-1]
        main_peak_idx = peaks_indices[sorted_peak_ranks[0]]
        if 0 < main_peak_idx < len(chip_distribution) - 1:
            y1 = float(chip_distribution[main_peak_idx - 1])
            y2 = float(chip_distribution[main_peak_idx])
            y3 = float(chip_distribution[main_peak_idx + 1])
            denominator = y1 - 2 * y2 + y3
            p_shift = 0.5 * (y1 - y3) / denominator if abs(denominator) > eps else 0.0
            exact_idx = main_peak_idx + p_shift
            idx_floor = int(np.floor(exact_idx))
            idx_ceil = min(idx_floor + 1, len(price_grid) - 1)
            weight = exact_idx - idx_floor
            main_peak_price = float(price_grid[idx_floor]) * (1.0 - weight) + float(price_grid[idx_ceil]) * weight
            peak_concentration = y2 - 0.25 * (y1 - y3) * p_shift if abs(denominator) > eps else y2
        else:
            main_peak_price = float(price_grid[main_peak_idx])
            peak_concentration = float(chip_distribution[main_peak_idx])
        price_min, price_max = float(price_grid[0]), float(price_grid[-1])
        price_range = max(price_max - price_min, eps)
        main_peak_position_mass = float(norm_cdf[main_peak_idx])
        highest_peak_price = float(np.max(price_grid[peaks_indices]))
        lowest_peak_price = float(np.min(price_grid[peaks_indices]))
        peak_distance_ratio = (highest_peak_price - lowest_peak_price) / price_range
        if peak_count > 1: peak_concentration += float(chip_distribution[peaks_indices[sorted_peak_ranks[1]]])
        final_res = {'peak_count': int(peak_count), 'highest_peak_price': highest_peak_price, 'lowest_peak_price': lowest_peak_price, 'peak_distance_ratio': float(peak_distance_ratio), 'peak_concentration': float(peak_concentration), 'main_peak_position': round(main_peak_position_mass, 6)}
        return final_res

    @classmethod
    def calculate_absolute_change_analysis(cls, changes: np.ndarray, price_grid: np.ndarray, current_price: float) -> Dict[str, Any]:
        """[修复] 稀疏阵列感知：使用 active_mask 防止 MAD 塌陷为 0 导致锁定率归 0"""
        import numpy as np
        if len(changes) == 0 or current_price <= 0: return cls.get_default_absolute_analysis()
        abs_changes = np.abs(changes)
        total_change_volume = float(np.sum(abs_changes))
        active_mask = abs_changes > 1e-5
        if np.any(active_mask):
            active_changes = abs_changes[active_mask]
            median_abs = float(np.median(active_changes))
            mad = float(np.median(np.abs(active_changes - median_abs)))
        else:
            median_abs, mad = 0.0, 0.0
        safe_mad = max(mad, 1e-4)
        noise_threshold = max(1e-3, median_abs + 1.4826 * safe_mad * 0.3)
        # 使用 <=，将未变动的网格精准计入锁定状态
        chip_lock_ratio = float(np.sum(abs_changes <= noise_threshold) / len(changes))
        analysis = {'total_change_volume': total_change_volume, 'positive_change_volume': float(np.sum(changes[changes > 0])), 'negative_change_volume': float(np.sum(changes[changes < 0])), 'max_increase': float(np.max(changes)), 'max_decrease': float(np.min(changes)), 'mean_change': float(np.mean(changes)), 'change_std': float(np.std(changes)), 'change_concentration': cls.calculate_change_concentration(changes), 'chip_lock_ratio': chip_lock_ratio}
        # QuantitativeTelemetryProbe.emit("ChipMatrixDynamics", "calc_absolute_change", {'mad': float(mad), 'active_grids': int(np.sum(active_mask))}, {'noise_threshold': float(noise_threshold)}, {'chip_lock_ratio': analysis['chip_lock_ratio']})
        if hasattr(cls, 'analyze_a_share_key_price_levels'): analysis.update(cls.analyze_a_share_key_price_levels(changes, price_grid, current_price))
        if hasattr(cls, 'analyze_a_share_pullback_pattern'): analysis.update(cls.analyze_a_share_pullback_pattern(changes, price_grid, current_price))
        if hasattr(cls, 'detect_a_share_false_signals'): analysis.update(cls.detect_a_share_false_signals(changes, price_grid, current_price))
        if hasattr(cls, 'assess_a_share_trend_quality'): analysis.update(cls.assess_a_share_trend_quality(changes, price_grid, current_price))
        if hasattr(cls, 'analyze_chip_transfer_path'): analysis.update(cls.analyze_chip_transfer_path(changes, price_grid, current_price))
        return cls.clean_structure(analysis, precision=3)

    @staticmethod
    def calculate_change_concentration(changes: np.ndarray) -> float:
        """[修复] 防止稀疏 0 值均值稀释导致指数函数梯度过载，及 partition 越界雷区"""
        import numpy as np
        import math
        try:
            if len(changes) == 0: return 0.0
            eps = 1e-8
            abs_changes = np.abs(changes)
            total_volume = float(np.sum(abs_changes))
            if total_volume <= eps: return 0.0
            sorted_abs = np.sort(abs_changes)[::-1]
            conc_t1 = float(sorted_abs[0] / total_volume)
            conc_t5 = float(np.sum(sorted_abs[:5]) / total_volume)
            hhi = float(np.sum((abs_changes / total_volume) ** 2))
            pos_m, neg_m = changes > 0, changes < 0
            # 弃用 np.partition 防止越界崩溃，改用 np.sort
            c_pos = float(np.sum(np.sort(changes[pos_m])[-3:]) / np.sum(changes[pos_m])) if np.sum(pos_m) > 0 and np.sum(changes[pos_m]) > eps else 0.0
            c_neg = float(np.sum(np.sort(abs_changes[neg_m])[-3:]) / np.sum(abs_changes[neg_m])) if np.sum(neg_m) > 0 and np.sum(abs_changes[neg_m]) > eps else 0.0
            dir_bal = abs(c_pos - c_neg)
            active_mask = abs_changes > 1e-5
            if np.any(active_mask):
                active_abs = abs_changes[active_mask]
                change_mean, change_std = float(np.mean(active_abs)), float(np.std(active_abs))
                small_change_ratio = float(np.sum(abs_changes < change_mean * 0.5) / len(changes))
            else:
                change_mean, change_std, small_change_ratio = 1e-4, 0.0, 1.0
            safe_mean = max(change_mean, 1e-4)
            cv_ratio = float(change_std / safe_mean)
            max_change = float(sorted_abs[0])
            # 引入温度系数 5.0，放宽敏感度，消灭 0.9999 的饱和死锁
            main_force_intensity = float(1.0 - math.exp(-max_change / (safe_mean * 5.0)))
            weights = {'top5': 0.25, 'hhi': 0.20, 'main': 0.15, 'dir': 0.10, 'cv': 0.10, 'top1': 0.10, 'small': 0.10}
            indicators = {'top5': conc_t5, 'hhi': min(1.0, hhi * 10.0), 'main': main_force_intensity, 'dir': dir_bal, 'cv': min(1.0, cv_ratio / 2.0), 'top1': conc_t1, 'small': 1.0 - small_change_ratio}
            if conc_t5 > 0.9:
                weights['top1'] += 0.15; weights['top5'] += 0.10
                total_w = sum(weights.values())
                weights = {k: v / total_w for k, v in weights.items()}
            comp = sum(indicators[k] * weights[k] for k in weights)
            if conc_t5 > 0.85 and main_force_intensity > 0.8: comp *= 0.8
            if conc_t5 < 0.4 and small_change_ratio > 0.7: comp *= 0.7
            final_conc = float(max(0.0, min(1.0, comp)))
            # QuantitativeTelemetryProbe.emit("ChipMatrixDynamics", "calc_concentration", {'active_mean': change_mean}, {'cv': cv_ratio, 'main_force_intensity': main_force_intensity}, {'final_conc': final_conc})
            return final_conc
        except Exception: return 0.0

    @staticmethod
    def analyze_a_share_key_price_levels(changes: np.ndarray, price_grid: np.ndarray, current_price: float) -> Dict[str, Any]:
        """[修复] 价格能级自适应拓扑探测，废除 0.01 绝对误差刻舟求剑"""
        import numpy as np
        analysis = {'integer_resistance_levels': [], 'integer_support_levels': [], 'technical_levels': [], 'historical_reference_levels': []}
        try:
            if len(changes) == 0 or current_price <= 0: return analysis
            dynamic_tol = max(0.01, current_price * 0.003)
            rounded_prices = np.round(price_grid)
            is_integer = np.abs(price_grid - rounded_prices) <= dynamic_tol
            if np.any(is_integer):
                for int_level in np.unique(rounded_prices[is_integer]):
                    mask = np.abs(price_grid - int_level) <= dynamic_tol
                    strength = float(np.sum(changes[mask]))
                    if abs(strength) > 1e-5:
                        item = {'price': float(int_level), 'strength': strength, 'distance_pct': float((int_level - current_price) / current_price), 'type': 'integer_support' if int_level < current_price else 'integer_resistance'}
                        analysis['integer_support_levels' if int_level < current_price else 'integer_resistance_levels'].append(item)
            ratios = np.array([-0.618, -0.382, 0.382, 0.5, 0.618], dtype=np.float32)
            golden_levels = (current_price * (1.0 - ratios)).astype(np.float32)
            nearest_indices = np.argmin(np.abs(price_grid[None, :] - golden_levels[:, None]), axis=1)
            for i, idx in enumerate(nearest_indices):
                if np.abs(price_grid[idx] - golden_levels[i]) <= dynamic_tol * 2.0:
                    analysis['technical_levels'].append({'price': float(golden_levels[i]), 'actual_price': float(price_grid[idx]), 'strength': float(changes[idx]), 'type': 'golden_ratio' if ratios[i] > 0 else 'fibonacci_extension', 'ratio': float(ratios[i])})
            for key in ['integer_support_levels', 'integer_resistance_levels', 'technical_levels']: analysis[key] = sorted(analysis[key], key=lambda x: abs(x['strength']), reverse=True)[:5]
        except Exception: pass
        return analysis

    @staticmethod
    def analyze_a_share_pullback_pattern(changes: np.ndarray, price_grid: np.ndarray, current_price: float) -> Dict[str, Any]:
        """[修复] 动态流形视界拉升模式分析"""
        import numpy as np
        analysis = {'pullback_phase_detected': False, 'pullback_strength': 0.0, 'pullback_type': 'none', 'support_levels': [], 'resistance_levels': [], 'breakout_potential': 0.0, 'consolidation_completeness': 0.0}
        try:
            if len(changes) == 0: return analysis
            eps = 1e-8
            grid_range = float(price_grid[-1] - price_grid[0]) if len(price_grid) > 1 else float(current_price * 0.2)
            sigma_p = max(grid_range / 15.0, current_price * 0.015)
            below_mask = price_grid < (current_price - sigma_p * 0.2)
            if np.any(below_mask):
                b_changes = changes[below_mask]
                pos_mask = b_changes > 0
                if np.any(pos_mask):
                    v_c, v_p = b_changes[pos_mask], price_grid[below_mask][pos_mask]
                    for idx in (np.argsort(v_c)[-3:][::-1] if len(v_c) > 3 else np.argsort(v_c)[::-1]): analysis['support_levels'].append({'price': float(v_p[idx]), 'strength': float(v_c[idx]), 'distance_pct': float((current_price - v_p[idx]) / (current_price + eps)), 'type': 'strong_support'})
            above_mask = price_grid > (current_price + sigma_p * 0.2)
            if np.any(above_mask):
                a_changes = changes[above_mask]
                neg_mask = a_changes < 0
                if np.any(neg_mask):
                    v_c, v_p = np.abs(a_changes[neg_mask]), price_grid[above_mask][neg_mask]
                    for idx in (np.argsort(v_c)[-3:][::-1] if len(v_c) > 3 else np.argsort(v_c)[::-1]): analysis['resistance_levels'].append({'price': float(v_p[idx]), 'strength': float(v_c[idx]), 'distance_pct': float((v_p[idx] - current_price) / (current_price + eps)), 'type': 'strong_resistance'})
            res_strength = float(np.sum(np.abs(changes[(price_grid > current_price) & (changes < 0)])))
            sup_strength = float(np.sum(changes[(price_grid < current_price) & (changes > 0)]))
            analysis['breakout_potential'] = float(min(1.0, sup_strength / (res_strength + sup_strength + eps)))
            mid_zone_mask = (price_grid >= current_price - sigma_p * 0.8) & (price_grid <= current_price + sigma_p * 0.8)
            total_abs = float(np.sum(np.abs(changes)))
            if total_abs > eps:
                mid_concentration = float(np.sum(np.abs(changes[mid_zone_mask])) / total_abs)
                analysis['consolidation_completeness'] = mid_concentration
                if mid_concentration > 0.6: analysis['pullback_phase_detected'] = True; analysis['pullback_type'] = 'consolidation'; analysis['pullback_strength'] = mid_concentration
        except Exception: pass
        return analysis

    @staticmethod
    def assess_a_share_trend_quality(changes: np.ndarray, price_grid: np.ndarray, current_price: float) -> Dict[str, Any]:
        """[修复] 使用 1D 空间连续卷积池化，彻底替代违背拓扑连续性的 argpartition 孤立抓取"""
        import numpy as np
        import math
        quality = {'trend_quality': 0.5, 'trend_health': 0.5, 'sustainability': 0.5, 'acceleration_potential': 0.0, 'risk_adjusted_score': 0.5, 'quality_indicators': {}}
        try:
            if len(changes) == 0: return quality
            eps = 1e-8
            price_rel = (price_grid - current_price) / (current_price + eps)
            below_flow, above_flow = float(np.sum(changes[price_rel < -0.05])), float(np.sum(changes[price_rel > 0.05]))
            trend_consistency = 0.5
            if below_flow < 0 and above_flow > 0: trend_consistency = 0.5 + float(min(abs(below_flow), above_flow) / (abs(below_flow) + above_flow + eps) * 0.5)
            elif below_flow > 0 and above_flow < 0: trend_consistency = 0.5 + float(min(below_flow, abs(above_flow)) / (below_flow + abs(above_flow) + eps) * 0.5)
            quality['trend_health'] = trend_consistency
            abs_changes = np.abs(changes)
            total_volume = float(np.sum(abs_changes))
            active_mask = abs_changes > 1e-5
            dynamic_optimal = float(np.median(abs_changes[active_mask]) * 2.5) if np.any(active_mask) else 0.3
            optimal_turnover = float(max(0.05, min(dynamic_optimal, 1.0)))
            turnover_intensity = float(np.mean(abs_changes))
            turnover_score = 1.0 - float(math.exp(-turnover_intensity / (optimal_turnover + eps)))
            lock_ratio = float(np.mean(abs_changes < (optimal_turnover * 0.3))) if len(changes) > 0 else 0.5
            quality['sustainability'] = float(0.4 * lock_ratio + 0.6 * turnover_score)
            accel_potential = 0.0
            if total_volume > eps:
                kernel_size = max(3, len(changes) // 20)
                kernel = np.ones(kernel_size) / kernel_size
                # 空间连续卷积池化，寻找最大的真实拓扑团块
                smoothed_abs = np.convolve(abs_changes, kernel, mode='same')
                best_center_idx = int(np.argmax(smoothed_abs))
                half_k = kernel_size // 2
                start_idx, end_idx = max(0, best_center_idx - half_k), min(len(changes), best_center_idx + half_k + 1)
                top_dense_volume = float(np.sum(abs_changes[start_idx:end_idx]))
                top_dense_net = float(np.abs(np.sum(changes[start_idx:end_idx])))
                accel_potential = float((top_dense_volume / total_volume) * (top_dense_net / (top_dense_volume + eps)))
            quality['acceleration_potential'] = accel_potential
            risk_adj = float(1.0 / (1.0 + np.std(changes) * 3.0))
            quality['trend_quality'] = float(0.3 * trend_consistency + 0.3 * quality['sustainability'] + 0.2 * accel_potential + 0.2 * risk_adj)
            quality['risk_adjusted_score'] = float(quality['trend_quality'] * risk_adj)
        except Exception: pass
        return quality

    @staticmethod
    def analyze_chip_transfer_path(changes: np.ndarray, price_grid: np.ndarray, current_price: float) -> Dict[str, Any]:
        """[修复] Z-Score 过滤非活跃网格，避免稀疏0值除零爆炸"""
        import numpy as np
        transfer = {'transfer_direction': 'unclear', 'transfer_intensity': 0.0, 'source_zones': [], 'destination_zones': [], 'transfer_efficiency': 0.5}
        try:
            if len(changes) == 0: return transfer
            eps = 1e-8
            active_mask = np.abs(changes) > 1e-5
            if np.sum(active_mask) > 3:
                active_changes = changes[active_mask]
                median = float(np.median(active_changes))
                mad = float(np.median(np.abs(active_changes - median)))
                safe_mad = max(mad, 1e-4)
            else:
                median, safe_mad = 0.0, 1e-4
            robust_z_scores = (changes - median) / (1.4826 * safe_mad)
            dec_indices = np.where(robust_z_scores < -2.5)[0]
            if len(dec_indices) > 0:
                for idx in dec_indices[np.argsort(changes[dec_indices])[:3]]: transfer['source_zones'].append({'price': float(price_grid[idx]), 'change': float(changes[idx]), 'type': 'distribution_source', 'distance_pct': float((price_grid[idx] - current_price) / (current_price + eps))})
            inc_indices = np.where(robust_z_scores > 2.5)[0]
            if len(inc_indices) > 0:
                for idx in inc_indices[np.argsort(changes[inc_indices])[::-1][:3]]: transfer['destination_zones'].append({'price': float(price_grid[idx]), 'change': float(changes[idx]), 'type': 'accumulation_destination', 'distance_pct': float((price_grid[idx] - current_price) / (current_price + eps))})
            if transfer['source_zones'] and transfer['destination_zones']:
                avg_src = float(np.mean([z['price'] for z in transfer['source_zones']]))
                avg_dst = float(np.mean([z['price'] for z in transfer['destination_zones']]))
                if avg_dst > avg_src * 1.02: transfer['transfer_direction'] = 'up'
                elif avg_dst < avg_src * 0.98: transfer['transfer_direction'] = 'down'
                else: transfer['transfer_direction'] = 'sideways'
            total_changes = float(np.sum(np.abs(changes)))
            if total_changes > eps:
                sig_transfer = float(np.sum(np.abs(changes[np.abs(robust_z_scores) > 1.5])))
                transfer['transfer_intensity'] = float(sig_transfer / total_changes)
                transfer['transfer_efficiency'] = float(min(1.0, transfer['transfer_intensity'] * 1.5))
        except Exception: pass
        return transfer

    @staticmethod
    def detect_a_share_false_signals(changes: np.ndarray, price_grid: np.ndarray, current_price: float) -> Dict[str, Any]:
        """[修复] A股虚假信号动态边界识别，抛弃 1.08 等固化百分比刻舟求剑"""
        import numpy as np
        analysis = {'false_distribution_flag': False, 'false_accumulation_flag': False, 'wash_sale_detected': False, 'fake_breakout_risk': 0.0, 'signal_reliability': 0.7}
        try:
            if len(changes) == 0: return analysis
            price_rel = (price_grid - current_price) / (current_price + 1e-8)
            dynamic_vol = np.std(price_rel[np.abs(changes) > 1e-4]) if np.any(np.abs(changes) > 1e-4) else 0.05
            u_bound, l_bound = max(0.03, min(dynamic_vol * 2.0, 0.15)), max(0.03, min(dynamic_vol * 2.0, 0.15))
            high_mask, mid_high_mask = price_rel > u_bound, (price_rel > u_bound * 0.3) & (price_rel <= u_bound)
            low_mask, mid_low_mask = price_rel < -l_bound, (price_rel >= -l_bound) & (price_rel <= -l_bound * 0.3)
            high_inc, low_dec, mid_high_net = np.sum(changes[high_mask & (changes > 0)]), np.sum(-changes[low_mask & (changes < 0)]), np.sum(changes[mid_high_mask])
            if mid_high_net < -0.3 and high_inc < 0.2 and low_dec < 0.1: analysis['false_distribution_flag'] = True
            low_inc, mid_low_dec = np.sum(changes[low_mask & (changes > 0)]), np.sum(-changes[mid_low_mask & (changes < 0)])
            if low_inc > 0.4 and mid_low_dec > 0.3: analysis['false_accumulation_flag'] = True
            near_mask = np.abs(price_rel) <= u_bound * 0.2
            if np.any(near_mask):
                p_n, n_n = np.sum(changes[near_mask & (changes > 0)]), np.sum(-changes[near_mask & (changes < 0)])
                if p_n > 0.3 and n_n > 0.3 and abs(p_n - n_n) < 0.1: analysis['wash_sale_detected'] = True
            above_mask = price_rel > 0
            if np.any(above_mask):
                r_b, r_h = np.sum(changes[above_mask & (changes > 0)]), np.sum(-changes[above_mask & (changes < 0)])
                if r_b > 0 and r_h > r_b * 1.5: analysis['fake_breakout_risk'] = float(min(1.0, r_h / (r_b + 1e-8)))
            abs_changes, total_abs = np.abs(changes), np.sum(np.abs(changes))
            rel_factors = []
            if total_abs > 1e-8:
                k = max(1, int(len(changes) * 0.1))
                rel_factors.append(min(1.0, (np.sum(np.sort(abs_changes)[-k:]) / total_abs) * 1.5) if len(changes) >= k else 1.0)
            else: rel_factors.append(0.0)
            rel_factors.append(max(0.0, 1.0 - sum([analysis['false_distribution_flag'], analysis['false_accumulation_flag'], analysis['wash_sale_detected']]) * 0.3))
            analysis['signal_reliability'] = float(np.mean(rel_factors))
            return analysis
        except Exception: return analysis

    @classmethod
    def calculate_tick_enhanced_factors(cls, current_factors: Dict[str, float], tick_factors: Dict[str, Any], quality_score: float) -> Tuple[float, float, float, float, str]:
        short_term = float(current_factors.get('short', 0.2))
        mid_term = float(current_factors.get('mid', 0.3))
        long_term = float(current_factors.get('long', 0.5))
        avg_days = float(current_factors.get('days', 60.0))
        if not tick_factors or quality_score <= 0.3: return short_term, mid_term, long_term, avg_days, "Tick质量极低，不触发微观盘口修正"
        tick_intensity = float(tick_factors.get('intraday_chip_turnover_intensity', 0.0))
        main_force_score = float(tick_factors.get('intraday_main_force_activity', 0.0))
        acc_conf = float(tick_factors.get('intraday_accumulation_confidence', 0.0))
        dist_conf = float(tick_factors.get('intraday_distribution_confidence', 0.0))
        reason = "微观资金结构平衡，维持基础权重"
        if tick_intensity > 0.7 and main_force_score < 0.3:
            adjust = tick_intensity * 0.15
            short_term = min(0.6, short_term + adjust)
            long_term = max(0.1, long_term - adjust * 0.5)
            reason = "散户高换手主导，大幅缩短持仓周期预期"
        elif main_force_score > 0.4:
            if acc_conf > dist_conf:
                mid_term = min(0.6, mid_term + 0.1)
                avg_days *= 1.2
                reason = "主力隐蔽吸筹信号确认，延长趋势持仓预期"
            else:
                short_term = min(0.5, short_term + 0.1)
                avg_days *= 0.8
                reason = "主力高频派发迹象明显，强制缩短防御周期"
        total = short_term + mid_term + long_term
        if total > 0: short_term /= total; mid_term /= total; long_term /= total
        return float(short_term), float(mid_term), float(long_term), float(avg_days), reason

    @classmethod
    def calculate_holding_factors(cls, dynamics_result: Dict[str, Any], absolute_change_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """[修复] 废除线性相减强平，引入带温度系数的 Softmax 流形映射，确保概率恒正且过渡平滑"""
        import numpy as np
        import math
        if absolute_change_analysis is None: absolute_change_analysis = {}
        conv_metrics = dynamics_result.get('convergence_metrics', {})
        conc_metrics = dynamics_result.get('concentration_metrics', {})
        behav_patterns = dynamics_result.get('behavior_patterns', {})
        chip_lock_ratio = float(absolute_change_analysis.get('chip_lock_ratio', 0.5))
        market_sentiment = float(absolute_change_analysis.get('market_sentiment', 0.5))
        trend_quality = float(absolute_change_analysis.get('trend_quality', 0.5))
        market_cycle = absolute_change_analysis.get('market_cycle_phase', 'consolidation')
        trend_health = float(absolute_change_analysis.get('trend_health', 0.5))
        main_force_activity = float(behav_patterns.get('main_force_activity', 0.0))
        cycle_adj = float(cls.calculate_market_cycle_adjustment(market_cycle, trend_health))
        comp_conv = float(conv_metrics.get('comprehensive_convergence', 0.0))
        comp_conc = float(conc_metrics.get('comprehensive_concentration', 0.0))
        logit_long = (comp_conv * 1.5 + comp_conc * 1.2 + chip_lock_ratio * 2.0) * cycle_adj
        logit_short = (1.0 - comp_conv) * 1.8 + main_force_activity * 1.5 + (1.0 - chip_lock_ratio) * 1.5
        if market_sentiment > 0.7: logit_short += 0.8
        elif market_sentiment < 0.3: logit_short += 0.5
        logit_mid = trend_quality * 2.0 + market_sentiment * 1.0 + 0.5
        logits = np.array([logit_short, logit_mid, logit_long], dtype=np.float64) / 1.2
        probs = np.exp(logits - np.max(logits)) / np.sum(np.exp(logits - np.max(logits)))
        s_ratio, m_ratio, l_ratio = float(probs[0]), float(probs[1]), float(probs[2])
        avg_days = float(np.clip(8.0 * math.exp(3.5 * l_ratio) * (0.8 + trend_quality * 0.4), 3.0, 600.0))
        res = {'short_term_ratio': round(s_ratio, 4), 'mid_term_ratio': round(m_ratio, 4), 'long_term_ratio': round(l_ratio, 4), 'avg_holding_days': round(avg_days, 2), 'extra_metrics': {'chip_lock_ratio': chip_lock_ratio, 'market_sentiment': market_sentiment, 'trend_quality': trend_quality, 'cycle_adj': cycle_adj}}
        return res

    @staticmethod
    def calculate_market_cycle_adjustment(market_cycle: str, trend_health: float) -> float:
        cycle_adjustments = {'accumulation': 1.2, 'expansion': 1.1, 'consolidation': 1.0, 'distribution': 0.9, 'contraction': 0.8}
        adjustment = cycle_adjustments.get(market_cycle, 1.0)
        if trend_health > 0.7: adjustment = adjustment * (1.0 + (trend_health - 0.7) * 0.2)
        elif trend_health < 0.4: adjustment = adjustment * (0.8 + trend_health * 0.5)
        return min(1.5, max(0.5, adjustment))

    @staticmethod
    def get_default_absolute_analysis() -> Dict[str, Any]:
        return {'total_change_volume': 0.0, 'positive_change_volume': 0.0, 'negative_change_volume': 0.0, 'change_concentration': 0.0, 'max_increase': 0.0, 'max_decrease': 0.0, 'mean_change': 0.0, 'price_zone_analysis': {}, 'pullback_phase_detected': False, 'pullback_strength': 0.0, 'support_levels': [], 'resistance_levels': [], 'false_distribution_flag': False, 'signal_quality': 0.5, 'trend_strength': 0.5, 'key_price_levels': [], 'chip_lock_ratio': 1.0}

    @staticmethod
    def create_default_key_battle_zones(price_grid: List[float], current_price: float) -> List[Dict[str, Any]]:
        import numpy as np
        if not price_grid or current_price <= 0: return []
        price_array = np.array(price_grid)
        distances = np.abs(price_array - current_price)
        nearest_indices = np.argsort(distances)[:3]
        zones = []
        for idx in nearest_indices:
            price = price_array[idx]
            zones.append({'price': round(float(price), 3), 'battle_intensity': 0.1, 'type': 'default', 'position': 'below_current' if price < current_price else 'above_current', 'distance_to_current': round(float((price - current_price) / current_price), 3)})
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





