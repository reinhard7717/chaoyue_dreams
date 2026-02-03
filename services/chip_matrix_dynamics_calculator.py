# services\chip_matrix_dynamics_calculator.py
# 筹码矩阵动态分析计算服务
import numpy as np
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
    def clean_structure(data, precision=3, threshold=0.0):
        """数据清洗与精度控制辅助函数"""
        if isinstance(data, (float, int, np.number)):
            try:
                val = float(data)
                if math.isnan(val) or math.isinf(val): return 0.0
                if abs(val) < threshold: return 0.0
                if val == 0.0: return 0.0
                return round(val, precision)
            except Exception: return 0.0
        elif isinstance(data, np.ndarray):
            try:
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                if threshold > 0: data = np.where(np.abs(data) < threshold, 0.0, data)
                return np.round(data, precision).tolist()
            except Exception: return [ChipMatrixDynamicsCalculator.clean_structure(i, precision, threshold) for i in data.tolist()]
        elif isinstance(data, dict): return {k: ChipMatrixDynamicsCalculator.clean_structure(v, precision, threshold) for k, v in data.items()}
        elif isinstance(data, (list, tuple)): return [ChipMatrixDynamicsCalculator.clean_structure(i, precision, threshold) for i in data]
        return data

    @staticmethod
    def calculate_absolute_change_analysis(changes: np.ndarray, price_grid: np.ndarray, current_price: float) -> Dict[str, Any]:
        """深化版绝对变化分析 - 融合幻方A股趋势与反转博弈经验"""
        try:
            if len(changes) == 0 or len(price_grid) == 0 or current_price <= 0:
                return ChipMatrixDynamicsCalculator.get_default_absolute_analysis()
            # 1. 基础统计
            analysis = {
                'total_change_volume': float(np.sum(np.abs(changes))),
                'positive_change_volume': float(np.sum(changes[changes > 0])),
                'negative_change_volume': float(np.sum(changes[changes < 0])),
                'max_increase': float(np.max(changes)) if len(changes) > 0 else 0.0,
                'max_decrease': float(np.min(changes)) if len(changes) > 0 else 0.0,
                'mean_change': float(np.mean(changes)) if len(changes) > 0 else 0.0,
                'change_std': float(np.std(changes)) if len(changes) > 0 else 0.0,
            }
            # 2. A股特有变化集中度分析
            if analysis['total_change_volume'] > 0:
                abs_changes = np.abs(changes)
                sorted_indices = np.argsort(abs_changes)[::-1]
                top_5_percent = max(1, int(len(changes) * 0.05))
                top_indices = sorted_indices[:top_5_percent]
                analysis['change_concentration'] = np.sum(abs_changes[top_indices]) / analysis['total_change_volume']
                # 筹码锁定度
                threshold = np.percentile(abs_changes[abs_changes > 0], 10) if np.any(abs_changes > 0) else 0.0
                analysis['chip_lock_ratio'] = np.sum(abs_changes < threshold) / len(changes) if threshold > 0 else 0.0
            else:
                analysis['change_concentration'] = 0.0
                analysis['chip_lock_ratio'] = 1.0
            # 3. 组合各个维度的分析
            analysis.update(ChipMatrixDynamicsCalculator.analyze_a_share_key_price_levels(changes, price_grid, current_price))
            analysis.update(ChipMatrixDynamicsCalculator.analyze_a_share_pullback_pattern(changes, price_grid, current_price))
            analysis.update(ChipMatrixDynamicsCalculator.analyze_a_share_reversal_features(changes, price_grid, current_price))
            analysis.update(ChipMatrixDynamicsCalculator.detect_a_share_false_signals(changes, price_grid, current_price))
            analysis.update(ChipMatrixDynamicsCalculator.calculate_a_share_market_sentiment(changes, price_grid, current_price))
            analysis.update(ChipMatrixDynamicsCalculator.assess_a_share_trend_quality(changes, price_grid, current_price))
            analysis.update(ChipMatrixDynamicsCalculator.analyze_chip_transfer_path(changes, price_grid, current_price))
            analysis.update(ChipMatrixDynamicsCalculator.identify_sector_rotation_patterns(changes, price_grid, current_price))
            return analysis
        except Exception as e:
            print(f"❌ [绝对变化分析] 异常: {e}")
            return ChipMatrixDynamicsCalculator.get_default_absolute_analysis()

    @staticmethod
    def calculate_tick_enhanced_factors(current_factors: Dict[str, float], tick_factors: Dict[str, Any], quality_score: float) -> Tuple[float, float, float, float, Dict[str, Any]]:
        """
        基于Tick数据的持有时间因子修正
        返回: (short_term, mid_term, long_term, avg_days, adjustment_reason)
        """
        short_term = current_factors.get('short', 0.2)
        mid_term = current_factors.get('mid', 0.3)
        long_term = current_factors.get('long', 0.5)
        avg_days = current_factors.get('days', 60.0)
        reason = {}

        try:
            if quality_score < 0.5:
                return short_term, mid_term, long_term, avg_days, {'msg': 'low_quality'}
            # 提取因子
            tick_intensity = tick_factors.get('intraday_chip_turnover_intensity', 0.0)
            main_force_score = tick_factors.get('intraday_main_force_activity', 0.0)
            volume_clustering = tick_factors.get('volume_clustering_score', 0.0)
            # A股散户特征识别
            retail_dominated = False
            if tick_intensity > 0.7 and main_force_score < 0.3:
                retail_dominated = True
            # 调整逻辑
            if retail_dominated:
                adjust = tick_intensity * 0.15
                short_term = min(0.6, short_term + adjust)
                long_term = max(0.1, long_term - adjust * 0.5)
            if main_force_score > 0.4:
                acc_conf = tick_factors.get('intraday_accumulation_confidence', 0.0)
                dist_conf = tick_factors.get('intraday_distribution_confidence', 0.0)
                if acc_conf > dist_conf:
                    # 吸筹 -> 延长时间
                    mid_term = min(0.6, mid_term + 0.1)
                    avg_days *= 1.2
                else:
                    # 派发 -> 缩短时间
                    short_term = min(0.5, short_term + 0.1)
                    avg_days *= 0.8
            # 归一化
            total = short_term + mid_term + long_term
            if abs(total - 1.0) > 0.001:
                scale = 1.0 / total
                short_term *= scale
                mid_term *= scale
                long_term *= scale
            reason = {
                'retail_dominated': retail_dominated,
                'main_force_score': main_force_score,
                'tick_intensity': tick_intensity
            }
            return round(short_term, 4), round(mid_term, 4), round(long_term, 4), round(avg_days, 1), reason

        except Exception as e:
            print(f"⚠️ [Tick增强计算] 异常: {e}")
            return short_term, mid_term, long_term, avg_days, {'error': str(e)}

    @staticmethod
    def calculate_holding_factors(dynamics_result: Dict[str, Any], absolute_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        核心逻辑：计算持有时间分布因子
        """
        factors = {
            'short_term_ratio': 0.25,
            'mid_term_ratio': 0.35,
            'long_term_ratio': 0.40,
            'avg_holding_days': 60.0,
            'extra_metrics': {}
        }
        try:
            absolute_analysis = absolute_analysis or {}
            # 1. 提取基础指标
            convergence = dynamics_result.get('convergence_metrics', {})
            concentration = dynamics_result.get('concentration_metrics', {})
            behavior = dynamics_result.get('behavior_patterns', {})
            market_sentiment = absolute_analysis.get('market_sentiment', 0.5)
            trend_quality = absolute_analysis.get('trend_quality', 0.5)
            chip_lock_ratio = absolute_analysis.get('chip_lock_ratio', 0.5)
            market_cycle = absolute_analysis.get('rotation_analysis', {}).get('market_cycle_phase', 'consolidation')
            # 2. 长线逻辑 (A股特色)
            long_term_base = (
                convergence.get('comprehensive_convergence', 0.5) * 0.3 +
                concentration.get('comprehensive_concentration', 0.5) * 0.3 +
                chip_lock_ratio * 0.4
            )
            # 市场周期调整
            cycle_adj = ChipMatrixDynamicsCalculator.calculate_market_cycle_adjustment(market_cycle, trend_health=absolute_analysis.get('trend_health', 0.5))
            long_term_base *= cycle_adj
            factors['long_term_ratio'] = min(0.85, max(0.05, long_term_base))
            # 3. 短线逻辑
            divergence_score = 1.0 - convergence.get('comprehensive_convergence', 0.5)
            main_force_activity = behavior.get('main_force_activity', 0.0)
            short_term_base = (
                divergence_score * 0.35 +
                main_force_activity * 0.25 +
                (1.0 - chip_lock_ratio) * 0.2
            )
            # 情绪调整
            if market_sentiment > 0.7: short_term_base *= 1.3 # 过热
            elif market_sentiment < 0.3: short_term_base *= 1.1 # 恐慌
            factors['short_term_ratio'] = min(0.65, max(0.05, short_term_base))
            # 4. 中线补足与归一化
            remaining = 1.0 - factors['short_term_ratio'] - factors['long_term_ratio']
            if remaining < 0:
                # 压缩短线优先
                excess = -remaining
                factors['short_term_ratio'] -= excess * 0.6
                factors['long_term_ratio'] -= excess * 0.4
                factors['mid_term_ratio'] = 0.0
            else:
                factors['mid_term_ratio'] = remaining
            # 强制归一化
            total = sum([factors['short_term_ratio'], factors['mid_term_ratio'], factors['long_term_ratio']])
            if abs(total - 1.0) > 0.001:
                scale = 1.0 / total
                factors['short_term_ratio'] *= scale
                factors['mid_term_ratio'] *= scale
                factors['long_term_ratio'] *= scale
            # 5. 平均持有天数计算
            base_days = 20 + factors['long_term_ratio'] * 160
            trend_days_adjust = 0.8 + trend_quality * 0.4
            factors['avg_holding_days'] = max(3.0, min(365.0, base_days * trend_days_adjust))
            # 6. 如果有Tick数据，进行增强修正 (在Model层调用前已准备好Tick数据，或在此处调用逻辑)
            # 这里仅做基础计算，Model层会结合Tick数据再次修正
            factors['extra_metrics']['holding_calculation_details'] = {
                'market_cycle': market_cycle,
                'trend_quality': trend_quality,
                'chip_lock_ratio': chip_lock_ratio
            }
            return factors

        except Exception as e:
            print(f"⚠️ [持有时间计算] 异常: {e}")
            return factors

    # ========== 具体的私有分析方法 (静态化) ==========

    @staticmethod
    def analyze_a_share_key_price_levels(changes: np.ndarray, price_grid: np.ndarray, current_price: float) -> Dict[str, Any]:
        """A股关键价格位分析（整数关口、历史价位、技术位）"""
        analysis = {
            'integer_resistance_levels': [],
            'integer_support_levels': [],
            'technical_levels': [],
            'historical_reference_levels': [],
        }
        try:
            # 1. 整数关口分析（A股特有的整数心理关口）
            integer_levels = [round(price, 0) for price in price_grid if abs(price - round(price, 0)) < 0.01]
            unique_integers = sorted(set(integer_levels))
            for int_level in unique_integers:
                if int_level < current_price:
                    analysis['integer_support_levels'].append({
                        'price': float(int_level),
                        'strength': float(np.sum(changes[np.abs(price_grid - int_level) < 0.01])),
                        'distance_pct': float((current_price - int_level) / current_price),
                        'type': 'integer_support'
                    })
                else:
                    analysis['integer_resistance_levels'].append({
                        'price': float(int_level),
                        'strength': float(np.sum(changes[np.abs(price_grid - int_level) < 0.01])),
                        'distance_pct': float((int_level - current_price) / current_price),
                        'type': 'integer_resistance'
                    })
            # 2. 技术分析关键位（黄金分割、均线等近似位）
            # 黄金分割位
            golden_levels = []
            for ratio in [0.382, 0.5, 0.618]:
                level = current_price * (1 - ratio)
                golden_levels.append(level)
            for level in golden_levels:
                nearest_idx = np.argmin(np.abs(price_grid - level))
                if nearest_idx < len(changes):
                    analysis['technical_levels'].append({
                        'price': float(level),
                        'actual_price': float(price_grid[nearest_idx]),
                        'change': float(changes[nearest_idx]),
                        'type': 'golden_ratio',
                        'ratio': float((level - current_price) / current_price)
                    })
            # 按强度排序，取前5个
            for key in ['integer_support_levels', 'integer_resistance_levels', 'technical_levels']:
                analysis[key] = sorted(analysis[key], key=lambda x: abs(x['strength']), reverse=True)[:5]
        except Exception as e:
            print(f"关键价格位分析异常: {e}")
        return analysis

    @staticmethod
    def analyze_a_share_pullback_pattern(changes: np.ndarray, price_grid: np.ndarray, current_price: float) -> Dict[str, Any]:
        """A股拉升模式深度分析"""
        analysis = {
            'pullback_phase_detected': False,
            'pullback_strength': 0.0,
            'pullback_type': 'none',  # consolidation/accumulation/distribution
            'support_levels': [],
            'resistance_levels': [],
            'breakout_potential': 0.0,
            'consolidation_completeness': 0.0,
        }
        try:
            # 1. 寻找显著的支撑位（当前价以下筹码增加区域）
            below_current = price_grid < current_price * 0.99
            if np.sum(below_current) > 0:
                below_changes = changes[below_current]
                below_prices = price_grid[below_current]
                # 按变化强度排序
                strong_support_mask = below_changes > np.percentile(below_changes[below_changes > 0], 70) if np.any(below_changes > 0) else np.zeros_like(below_changes, dtype=bool)
                for idx in np.where(strong_support_mask)[0]:
                    analysis['support_levels'].append({
                        'price': float(below_prices[idx]),
                        'strength': float(below_changes[idx]),
                        'distance_pct': float((current_price - below_prices[idx]) / current_price),
                        'type': 'strong_support'
                    })
            # 2. 寻找显著的阻力位（当前价以上筹码减少区域）
            above_current = price_grid > current_price * 1.01
            if np.sum(above_current) > 0:
                above_changes = changes[above_current]
                above_prices = price_grid[above_current]
                strong_resistance_mask = above_changes < np.percentile(above_changes[above_changes < 0], 30) if np.any(above_changes < 0) else np.zeros_like(above_changes, dtype=bool)
                for idx in np.where(strong_resistance_mask)[0]:
                    analysis['resistance_levels'].append({
                        'price': float(above_prices[idx]),
                        'strength': float(-above_changes[idx]),  # 取绝对值
                        'distance_pct': float((above_prices[idx] - current_price) / current_price),
                        'type': 'strong_resistance'
                    })
            # 3. A股拉升初期特征识别
            # 特征1：低位强支撑 + 中位筹码锁定 + 高位弱阻力
            accumulation_below = np.sum(changes[(price_grid < current_price * 0.95) & (changes > 0.3)])
            lock_mid = np.sum(np.abs(changes[(price_grid >= current_price * 0.95) & (price_grid <= current_price * 1.05)])) < 0.2
            weak_resistance = np.sum(changes[(price_grid > current_price * 1.05) & (changes > -0.2)]) > -0.5
            if accumulation_below > 0.5 and lock_mid and weak_resistance:
                analysis['pullback_phase_detected'] = True
                analysis['pullback_type'] = 'accumulation'
                analysis['pullback_strength'] = min(1.0, accumulation_below)
            # 特征2：整理形态（筹码在中位区域集中）
            mid_zone = (price_grid >= current_price * 0.97) & (price_grid <= current_price * 1.03)
            if np.sum(mid_zone) > 0:
                mid_concentration = np.sum(np.abs(changes[mid_zone])) / np.sum(np.abs(changes)) if np.sum(np.abs(changes)) > 0 else 0
                analysis['consolidation_completeness'] = mid_concentration
                if mid_concentration > 0.6:
                    analysis['pullback_phase_detected'] = True
                    analysis['pullback_type'] = 'consolidation'
                    analysis['pullback_strength'] = mid_concentration
            # 4. 突破势能计算
            resistance_strength = np.sum(-changes[(price_grid > current_price) & (changes < 0)])
            support_strength = np.sum(changes[(price_grid < current_price) & (changes > 0)])
            if resistance_strength > 0 and support_strength > 0:
                analysis['breakout_potential'] = min(1.0, support_strength / (resistance_strength + support_strength))
            # 按强度排序，各取前3个
            analysis['support_levels'] = sorted(analysis['support_levels'], key=lambda x: x['strength'], reverse=True)[:3]
            analysis['resistance_levels'] = sorted(analysis['resistance_levels'], key=lambda x: x['strength'], reverse=True)[:3]
        except Exception as e:
            print(f"拉升模式分析异常: {e}")
        return analysis

    @staticmethod
    def analyze_a_share_reversal_features(changes: np.ndarray, price_grid: np.ndarray, current_price: float) -> Dict[str, Any]:
        """A股反转博弈特征分析"""
        analysis = {
            'reversal_signal': False,
            'reversal_strength': 0.0,
            'reversal_type': 'none',  # bottom/top/continuation
            'reversal_confidence': 0.0,
            'divergence_signals': [],
            'exhaustion_signals': [],
        }
        try:
            # 1. 量价背离检测（A股常见的反转信号）
            # 价格上涨但筹码减少（顶背离）
            price_increase_zones = (price_grid > current_price * 1.05) & (price_grid <= current_price * 1.15)
            if np.sum(price_increase_zones) > 0:
                avg_change_high = np.mean(changes[price_increase_zones])
                if avg_change_high < -0.2:  # 高位筹码显著减少
                    analysis['divergence_signals'].append({
                        'type': 'top_divergence',
                        'strength': -avg_change_high,
                        'zone': 'high_price',
                        'description': '价格高位但筹码减少，可能见顶'
                    })
            # 价格下跌但筹码增加（底背离）
            price_decrease_zones = (price_grid < current_price * 0.95) & (price_grid >= current_price * 0.85)
            if np.sum(price_decrease_zones) > 0:
                avg_change_low = np.mean(changes[price_decrease_zones])
                if avg_change_low > 0.2:  # 低位筹码显著增加
                    analysis['divergence_signals'].append({
                        'type': 'bottom_divergence',
                        'strength': avg_change_low,
                        'zone': 'low_price',
                        'description': '价格低位但筹码增加，可能见底'
                    })
            # 2. 衰竭信号检测
            # 连续价格区间的变化衰减
            price_bins = np.linspace(np.min(price_grid), np.max(price_grid), 6)
            bin_changes = []
            for i in range(len(price_bins)-1):
                bin_mask = (price_grid >= price_bins[i]) & (price_grid < price_bins[i+1])
                if np.sum(bin_mask) > 0:
                    bin_changes.append(np.mean(changes[bin_mask]))
            if len(bin_changes) >= 3:
                # 检查变化趋势是否衰减
                if bin_changes[-1] < bin_changes[-2] < bin_changes[-3] and bin_changes[-1] < 0:
                    analysis['exhaustion_signals'].append({
                        'type': 'uptrend_exhaustion',
                        'strength': -bin_changes[-1],
                        'pattern': 'decreasing_negative_changes',
                        'description': '上涨趋势中抛压逐渐衰减'
                    })
                elif bin_changes[0] > bin_changes[1] > bin_changes[2] and bin_changes[0] > 0:
                    analysis['exhaustion_signals'].append({
                        'type': 'downtrend_exhaustion',
                        'strength': bin_changes[0],
                        'pattern': 'decreasing_positive_changes',
                        'description': '下跌趋势中吸筹逐渐衰减'
                    })
            # 3. 综合反转判断
            if len(analysis['divergence_signals']) > 0:
                analysis['reversal_signal'] = True
                analysis['reversal_strength'] = max([sig['strength'] for sig in analysis['divergence_signals']])
                top_signals = [sig for sig in analysis['divergence_signals'] if sig['type'] == 'top_divergence']
                bottom_signals = [sig for sig in analysis['divergence_signals'] if sig['type'] == 'bottom_divergence']
                if top_signals and not bottom_signals:
                    analysis['reversal_type'] = 'top'
                elif bottom_signals and not top_signals:
                    analysis['reversal_type'] = 'bottom'
                else:
                    analysis['reversal_type'] = 'mixed'
                # 反转置信度计算
                sig_count = len(analysis['divergence_signals']) + len(analysis['exhaustion_signals'])
                sig_strength = analysis['reversal_strength']
                analysis['reversal_confidence'] = min(1.0, 0.3 + 0.4 * (sig_count / 2) + 0.3 * sig_strength)
        except Exception as e:
            print(f"反转特征分析异常: {e}")
        return analysis

    @staticmethod
    def detect_a_share_false_signals(changes: np.ndarray, price_grid: np.ndarray, current_price: float) -> Dict[str, Any]:
        """A股虚假信号识别深化"""
        analysis = {
            'false_distribution_flag': False,
            'false_accumulation_flag': False,
            'wash_sale_detected': False,
            'fake_breakout_risk': 0.0,
            'signal_reliability': 0.7,  # 默认中等可靠性
        }
        try:
            # 1. 虚假派发检测（获利回吐 vs 真实派发）
            high_price_zone = price_grid > current_price * 1.08
            mid_high_zone = (price_grid > current_price * 1.02) & (price_grid <= current_price * 1.08)
            low_price_zone = price_grid < current_price * 0.98
            # 真实派发特征：高位筹码大幅增加 + 中低位筹码减少
            high_increase = np.sum(changes[high_price_zone & (changes > 0)])
            mid_high_decrease = np.sum(-changes[mid_high_zone & (changes < 0)])
            low_decrease = np.sum(-changes[low_price_zone & (changes < 0)])
            # 获利回吐特征：中高位筹码减少，但低位形成支撑
            mid_high_net = np.sum(changes[mid_high_zone])
            # 虚假派发判断逻辑
            if mid_high_net < -0.3 and high_increase < 0.2 and low_decrease < 0.1:
                analysis['false_distribution_flag'] = True
            # 2. 虚假吸筹检测（对倒 vs 真实吸筹）
            low_increase = np.sum(changes[low_price_zone & (changes > 0)])
            mid_low_decrease = np.sum(-changes[(price_grid >= current_price * 0.98) & (price_grid <= current_price * 1.02) & (changes < 0)])
            if low_increase > 0.4 and mid_low_decrease > 0.3:
                analysis['false_accumulation_flag'] = True
            # 3. 洗盘行为检测（A股常见手法）
            # 洗盘特征：当前价附近双向大幅变化
            near_zone = (price_grid >= current_price * 0.99) & (price_grid <= current_price * 1.01)
            if np.sum(near_zone) > 0:
                near_changes = changes[near_zone]
                positive_near = np.sum(near_changes[near_changes > 0])
                negative_near = np.sum(-near_changes[near_changes < 0])
                if positive_near > 0.3 and negative_near > 0.3 and abs(positive_near - negative_near) < 0.1:
                    analysis['wash_sale_detected'] = True
            # 4. 假突破风险
            above_current = price_grid > current_price
            if np.sum(above_current) > 0:
                above_changes = changes[above_current]
                resistance_break = np.sum(above_changes[above_changes > 0])
                resistance_hold = np.sum(-above_changes[above_changes < 0])
                if resistance_break > 0 and resistance_hold > resistance_break * 1.5:
                    analysis['fake_breakout_risk'] = min(1.0, resistance_hold / (resistance_break + 0.1))
            # 5. 信号可靠性评估
            reliability_factors = []
            # 集中度越高，可靠性越高
            abs_changes = np.abs(changes)
            top_10_percent = np.argsort(abs_changes)[-int(len(changes)*0.1):]
            concentration = np.sum(abs_changes[top_10_percent]) / np.sum(abs_changes) if np.sum(abs_changes) > 0 else 0
            reliability_factors.append(min(1.0, concentration * 1.5))
            # 虚假信号越少，可靠性越高
            false_count = sum([1 for flag in [analysis['false_distribution_flag'], analysis['false_accumulation_flag'], analysis['wash_sale_detected']] if flag])
            reliability_factors.append(max(0.0, 1.0 - false_count * 0.3))
            # 趋势一致性
            positive_ratio = np.sum(changes > 0) / len(changes) if len(changes) > 0 else 0.5
            consistency = 1.0 - abs(positive_ratio - 0.5) * 2  # 0.5表示最不一致，0或1表示一致
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
        """A股趋势质量评估"""
        quality = {
            'trend_quality': 0.5,
            'trend_health': 0.5,
            'sustainability': 0.5,
            'acceleration_potential': 0.0,
            'risk_adjusted_score': 0.5,
            'quality_indicators': {},
        }
        try:
            # 1. 趋势健康度（价格与筹码的一致性）
            price_rel = (price_grid - current_price) / current_price
            # 上涨趋势健康：低位筹码减少，高位筹码增加
            below_mask = price_rel < -0.05
            above_mask = price_rel > 0.05
            below_flow = np.sum(changes[below_mask])
            above_flow = np.sum(changes[above_mask])
            trend_consistency = 0.5
            if below_flow < 0 and above_flow > 0:
                # 理想上涨趋势
                trend_consistency = 0.5 + min(abs(below_flow), above_flow) / (abs(below_flow) + above_flow + 0.1) * 0.5
            elif below_flow > 0 and above_flow < 0:
                # 理想下跌趋势
                trend_consistency = 0.5 + min(below_flow, abs(above_flow)) / (below_flow + abs(above_flow) + 0.1) * 0.5
            quality['trend_health'] = trend_consistency
            # 2. 趋势可持续性（筹码锁定度 + 换手适度性）
            lock_ratio = np.sum(np.abs(changes) < 0.1) / len(changes) if len(changes) > 0 else 0.5
            turnover_intensity = np.mean(np.abs(changes))
            # 适中的换手最可持续
            optimal_turnover = 0.3
            turnover_score = 1.0 - min(1.0, abs(turnover_intensity - optimal_turnover) / optimal_turnover)
            sustainability = 0.4 * lock_ratio + 0.6 * turnover_score
            quality['sustainability'] = sustainability
            # 3. 加速潜力（主力集中行为）
            abs_changes = np.abs(changes)
            top_5_indices = np.argsort(abs_changes)[-5:]
            top_5_volume = np.sum(abs_changes[top_5_indices])
            total_volume = np.sum(abs_changes)
            if total_volume > 0:
                concentration = top_5_volume / total_volume
                # 主力集中且有方向性才有加速潜力
                top_changes = changes[top_5_indices]
                directional_strength = np.abs(np.sum(top_changes)) / np.sum(abs_changes[top_5_indices]) if np.sum(abs_changes[top_5_indices]) > 0 else 0
                quality['acceleration_potential'] = concentration * directional_strength
            # 4. 风险调整评分
            volatility = np.std(changes) if len(changes) > 0 else 0.5
            risk_adjustment = 1.0 / (1.0 + volatility * 3)  # 波动性越高，评分越低
            # 5. 综合趋势质量
            quality['trend_quality'] = 0.3 * trend_consistency + 0.3 * sustainability + 0.2 * quality['acceleration_potential'] + 0.2 * risk_adjustment
            quality['risk_adjusted_score'] = quality['trend_quality'] * risk_adjustment
            # 6. 详细质量指标
            quality['quality_indicators'] = {
                'trend_consistency': float(trend_consistency),
                'chip_lock_degree': float(lock_ratio),
                'turnover_optimality': float(turnover_score),
                'main_force_concentration': float(concentration if total_volume > 0 else 0.0),
                'volatility_penalty': float(1.0 - risk_adjustment),
                'health_warning': trend_consistency < 0.3 or sustainability < 0.3,
            }
        except Exception as e:
            print(f"趋势质量评估异常: {e}")
        return quality

    @staticmethod
    def analyze_chip_transfer_path(changes: np.ndarray, price_grid: np.ndarray, current_price: float) -> Dict[str, Any]:
        """筹码转移路径分析"""
        transfer = {
            'transfer_direction': 'unclear',  # up/down/sideways
            'transfer_intensity': 0.0,
            'source_zones': [],
            'destination_zones': [],
            'transfer_efficiency': 0.5,
        }
        try:
            # 1. 识别筹码来源区域（显著减少的区域）
            significant_decrease_mask = changes < -0.2
            if np.sum(significant_decrease_mask) > 0:
                decrease_indices = np.where(significant_decrease_mask)[0]
                decrease_prices = price_grid[decrease_indices]
                decrease_changes = changes[decrease_indices]
                for i in range(min(3, len(decrease_indices))):
                    transfer['source_zones'].append({
                        'price': float(decrease_prices[i]),
                        'change': float(decrease_changes[i]),
                        'type': 'distribution_source',
                        'distance_pct': float((decrease_prices[i] - current_price) / current_price),
                    })
            # 2. 识别筹码目标区域（显著增加的区域）
            significant_increase_mask = changes > 0.2
            if np.sum(significant_increase_mask) > 0:
                increase_indices = np.where(significant_increase_mask)[0]
                increase_prices = price_grid[increase_indices]
                increase_changes = changes[increase_indices]
                for i in range(min(3, len(increase_indices))):
                    transfer['destination_zones'].append({
                        'price': float(increase_prices[i]),
                        'change': float(increase_changes[i]),
                        'type': 'accumulation_destination',
                        'distance_pct': float((increase_prices[i] - current_price) / current_price),
                    })
            # 3. 转移方向判断
            if len(transfer['source_zones']) > 0 and len(transfer['destination_zones']) > 0:
                avg_source_price = np.mean([zone['price'] for zone in transfer['source_zones']])
                avg_dest_price = np.mean([zone['price'] for zone in transfer['destination_zones']])
                if avg_dest_price > avg_source_price * 1.02:
                    transfer['transfer_direction'] = 'up'
                elif avg_dest_price < avg_source_price * 0.98:
                    transfer['transfer_direction'] = 'down'
                else:
                    transfer['transfer_direction'] = 'sideways'
            # 4. 转移强度
            total_transfer = np.sum(np.abs(changes[np.abs(changes) > 0.1]))
            total_changes = np.sum(np.abs(changes))
            if total_changes > 0:
                transfer['transfer_intensity'] = total_transfer / total_changes
            # 5. 转移效率（筹码是否向关键区域集中）
            # 理想的转移：从分散区域向关键区域集中
            source_concentration = len(transfer['source_zones']) / len(price_grid) if len(price_grid) > 0 else 0
            dest_concentration = len(transfer['destination_zones']) / len(price_grid) if len(price_grid) > 0 else 0
            if source_concentration > 0 and dest_concentration > 0:
                # 来源越分散，目标越集中，效率越高
                transfer['transfer_efficiency'] = min(1.0, (1.0 - source_concentration) * dest_concentration * 5)
        except Exception as e:
            print(f"筹码转移分析异常: {e}")
        return transfer

    @staticmethod
    def identify_sector_rotation_patterns(changes: np.ndarray, price_grid: np.ndarray, current_price: float) -> Dict[str, Any]:
        """板块轮动特征识别（基于筹码分布形态）"""
        rotation = {
            'sector_rotation_pattern': 'none',  # defensive/cyclical/growth/value
            'rotation_strength': 0.0,
            'market_cycle_phase': 'consolidation',  # accumulation/expansion/distribution/contraction
            'style_preference': 'balanced',  # large_cap/small_cap/growth/value
        }
        try:
            # 1. 基于筹码分布形态判断市场阶段
            price_rel = (price_grid - current_price) / current_price
            # 计算不同区域的筹码集中度
            deep_below_concentration = np.sum(np.abs(changes[price_rel < -0.15])) / np.sum(np.abs(changes)) if np.sum(np.abs(changes)) > 0 else 0
            near_concentration = np.sum(np.abs(changes[np.abs(price_rel) <= 0.05])) / np.sum(np.abs(changes)) if np.sum(np.abs(changes)) > 0 else 0
            deep_above_concentration = np.sum(np.abs(changes[price_rel > 0.15])) / np.sum(np.abs(changes)) if np.sum(np.abs(changes)) > 0 else 0
            # 市场阶段判断
            if near_concentration > 0.6:
                rotation['market_cycle_phase'] = 'consolidation'
            elif deep_below_concentration > deep_above_concentration and deep_below_concentration > 0.3:
                rotation['market_cycle_phase'] = 'accumulation'
            elif deep_above_concentration > deep_below_concentration and deep_above_concentration > 0.3:
                rotation['market_cycle_phase'] = 'distribution'
            else:
                rotation['market_cycle_phase'] = 'expansion'
            # 2. 板块轮动模式判断
            # 防御性板块特征：低位筹码锁定度高
            low_lock_ratio = np.sum(np.abs(changes[price_rel < -0.1]) < 0.05) / np.sum(price_rel < -0.1) if np.sum(price_rel < -0.1) > 0 else 0
            # 周期性板块特征：筹码在广泛区域活跃转移
            mid_activity = np.mean(np.abs(changes[(price_rel >= -0.1) & (price_rel <= 0.1)])) if np.sum((price_rel >= -0.1) & (price_rel <= 0.1)) > 0 else 0
            # 成长性板块特征：高位仍有活跃筹码
            high_activity = np.mean(np.abs(changes[price_rel > 0.1])) if np.sum(price_rel > 0.1) > 0 else 0
            if low_lock_ratio > 0.7 and mid_activity < 0.2:
                rotation['sector_rotation_pattern'] = 'defensive'
                rotation['rotation_strength'] = low_lock_ratio
            elif mid_activity > 0.3 and abs(np.sum(changes[(price_rel >= -0.1) & (price_rel <= 0.1)])) > 0.5:
                rotation['sector_rotation_pattern'] = 'cyclical'
                rotation['rotation_strength'] = mid_activity
            elif high_activity > 0.4:
                rotation['sector_rotation_pattern'] = 'growth'
                rotation['rotation_strength'] = high_activity
            else:
                rotation['sector_rotation_pattern'] = 'value'
                rotation['rotation_strength'] = 1.0 - max(low_lock_ratio, mid_activity, high_activity)
            # 3. 风格偏好判断
            # 大盘股特征：变化集中度高
            concentration_top_10 = np.sum(np.sort(np.abs(changes))[-int(len(changes)*0.1):]) / np.sum(np.abs(changes)) if np.sum(np.abs(changes)) > 0 else 0
            # 小盘股特征：变化分散度高
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
        深化版变化集中度计算 - 融合幻方A股博弈经验
        原函数只计算了Top 5的变化占比，深化版本将考虑：
        1. 多层次集中度：不同粒度（Top 1, 3, 5, 10）
        2. 方向性集中度：上涨/下跌变化的集中程度
        3. 博弈特征识别：主力行为集中度 vs 散户行为分散度
        4. 临界点检测：极端集中度对趋势的预示作用
        参数:
            changes: 筹码变化数组（已计算百分比变化）
        返回:
            综合变化集中度 (0-1)
        """
        try:
            if len(changes) == 0 or np.sum(np.abs(changes)) == 0:
                return 0.0
            abs_changes = np.abs(changes)
            total_volume = np.sum(abs_changes)
            # ========== 1. 基础集中度计算 ==========
            # 1.1 传统Top 5集中度（保持兼容性）
            top_5_indices = np.argsort(abs_changes)[-5:]
            top_5_volume = np.sum(abs_changes[top_5_indices])
            concentration_top5 = top_5_volume / total_volume
            # ========== 2. 多层次集中度分析 ==========
            # 2.1 极端集中度（Top 1）
            top_1_index = np.argmax(abs_changes)
            concentration_top1 = abs_changes[top_1_index] / total_volume
            # 2.2 主力集中度（Top 3）
            top_3_indices = np.argsort(abs_changes)[-3:]
            concentration_top3 = np.sum(abs_changes[top_3_indices]) / total_volume
            # 2.3 广义集中度（Top 10%）
            top_10_percent = max(1, int(len(changes) * 0.1))
            top_10_indices = np.argsort(abs_changes)[-top_10_percent:]
            concentration_top10p = np.sum(abs_changes[top_10_indices]) / total_volume
            # 2.4 赫芬达尔指数（衡量整体集中度）
            normalized_changes = abs_changes / total_volume
            herfindahl_index = np.sum(normalized_changes ** 2)
            # ========== 3. 方向性集中度计算 ==========
            # 3.1 上涨变化集中度
            positive_changes = changes[changes > 0]
            if len(positive_changes) > 0:
                pos_total = np.sum(positive_changes)
                if pos_total > 0:
                    pos_abs = np.abs(positive_changes)
                    pos_top3 = np.argsort(pos_abs)[-3:]
                    concentration_pos = np.sum(pos_abs[pos_top3]) / pos_total
                else:
                    concentration_pos = 0.0
            else:
                concentration_pos = 0.0
            # 3.2 下跌变化集中度
            negative_changes = changes[changes < 0]
            if len(negative_changes) > 0:
                neg_total = np.sum(np.abs(negative_changes))
                if neg_total > 0:
                    neg_abs = np.abs(negative_changes)
                    neg_top3 = np.argsort(neg_abs)[-3:]
                    concentration_neg = np.sum(neg_abs[neg_top3]) / neg_total
                else:
                    concentration_neg = 0.0
            else:
                concentration_neg = 0.0
            # 3.3 方向平衡度
            direction_balance = abs(concentration_pos - concentration_neg)
            # ========== 4. A股主力行为识别 ==========
            # 4.1 主力集中特征：少数几个价格格发生巨大变化
            # 计算变化分布的标准差和均值比率
            change_mean = np.mean(abs_changes)
            change_std = np.std(abs_changes)
            if change_mean > 0:
                cv_ratio = change_std / change_mean  # 变异系数
            else:
                cv_ratio = 0.0
            # 4.2 主力行为强度：基于最大变化与平均变化的比值
            max_change = np.max(abs_changes)
            if change_mean > 0:
                main_force_intensity = min(10.0, max_change / change_mean) / 10.0
            else:
                main_force_intensity = 0.0
            # 4.3 散户行为特征：大量小变化的分散分布
            # 计算小变化（小于平均变化）的占比
            small_change_threshold = change_mean * 0.5
            small_change_ratio = np.sum(abs_changes < small_change_threshold) / len(changes) if len(changes) > 0 else 0.0
            # ========== 5. 临界点检测 ==========
            # 5.1 极端集中预警（超过90%的变化集中在少数价格格）
            critical_concentration = concentration_top5 > 0.9
            # 5.2 集中度变化率（与上一级集中度对比）
            concentration_gradient_top1_to_top3 = (concentration_top3 - concentration_top1) / concentration_top1 if concentration_top1 > 0 else 0.0
            concentration_gradient_top3_to_top5 = (concentration_top5 - concentration_top3) / concentration_top3 if concentration_top3 > 0 else 0.0
            # ========== 6. 综合集中度计算 ==========
            # 采用加权综合评分，结合多种集中度指标
            # 6.1 基础权重分配
            weights = {
                'top5': 0.25,        # 传统Top 5集中度
                'herfindahl': 0.20,  # 赫芬达尔指数（整体集中度）
                'main_force': 0.15,  # 主力行为强度
                'direction_imbalance': 0.10,  # 方向不平衡度
                'cv_ratio': 0.10,    # 变化分布离散度
                'top1': 0.10,        # 极端集中度
                'small_change': 0.10, # 散户行为分散度（反向指标）
            }
            # 6.2 归一化各指标到0-1范围
            normalized_indicators = {
                'top5': concentration_top5,
                'herfindahl': min(1.0, herfindahl_index * 10),  # 赫芬达尔指数通常很小，适当放大
                'main_force': main_force_intensity,
                'direction_imbalance': direction_balance,
                'cv_ratio': min(1.0, cv_ratio / 2.0),  # 变异系数通常0-2，归一化到0-1
                'top1': concentration_top1,
                'small_change': 1.0 - small_change_ratio,  # 散户行为分散度越高，集中度越低
            }
            # 6.3 临界状态调整
            # 如果出现极端集中，增加权重
            if critical_concentration:
                # 极端集中时，增加top1和top5的权重
                weights['top1'] += 0.15
                weights['top5'] += 0.10
                # 重新归一化权重
                total_weight = sum(weights.values())
                for key in weights:
                    weights[key] /= total_weight
            # 6.4 计算综合集中度
            composite_concentration = 0.0
            for key, weight in weights.items():
                composite_concentration += normalized_indicators[key] * weight
            # 6.5 基于当前模型状态的微调
            # 如果有能量场数据，可用于验证集中度的有效性
            if hasattr(self, 'game_intensity') and self.game_intensity is not None:
                # 博弈强度高时，集中度更可信
                game_boost = min(0.2, self.game_intensity * 0.3)
                composite_concentration = min(1.0, composite_concentration * (1.0 + game_boost))
            if hasattr(self, 'absorption_energy') and hasattr(self, 'distribution_energy'):
                # 吸收能量和派发能量对比可用于验证集中方向
                net_energy_abs = abs(self.absorption_energy - self.distribution_energy)
                net_energy_total = self.absorption_energy + self.distribution_energy
                if net_energy_total > 0:
                    energy_imbalance = net_energy_abs / net_energy_total
                    # 能量失衡程度与集中度的一致性
                    energy_consistency = 1.0 - abs(energy_imbalance - direction_balance) * 0.5
                    composite_concentration *= energy_consistency
            # ========== 7. A股特殊模式识别 ==========
            # 7.1 识别对倒行为（异常集中模式）
            # 对倒特征：高集中度但方向平衡（买入卖出都很集中）
            wash_trade_suspicion = 0.0
            if concentration_top5 > 0.7 and direction_balance < 0.3:
                # 可能为对倒行为，需要降低集中度可信度
                wash_trade_suspicion = 0.3
            # 7.2 识别控盘行为
            # 控盘特征：极高集中度+主力行为强度
            control_suspicion = 0.0
            if concentration_top5 > 0.85 and main_force_intensity > 0.8:
                control_suspicion = 0.4
                # 控盘情况下，集中度需谨慎解读
                composite_concentration *= 0.8
            # 7.3 识别散户恐慌/狂热
            # 散户特征：低集中度+高散户行为分散度
            retail_dominated = 0.0
            if concentration_top5 < 0.4 and small_change_ratio > 0.7:
                retail_dominated = 0.5
                # 散户主导时，集中度有效性降低
                composite_concentration *= 0.7
            # 限制在0-1范围内
            final_concentration = max(0.0, min(1.0, composite_concentration))
            # ========== 8. 记录详细分析结果（可选存储） ==========
            # 将详细分析结果存储到extra_metrics中，便于后续分析
            detailed_analysis = {
                'basic_concentration': {
                    'top1': float(concentration_top1),
                    'top3': float(concentration_top3),
                    'top5': float(concentration_top5),
                    'top10_percent': float(concentration_top10p),
                    'herfindahl_index': float(herfindahl_index),
                },
                'directional_concentration': {
                    'positive': float(concentration_pos),
                    'negative': float(concentration_neg),
                    'balance': float(direction_balance),
                },
                'behavioral_analysis': {
                    'main_force_intensity': float(main_force_intensity),
                    'cv_ratio': float(cv_ratio),
                    'small_change_ratio': float(small_change_ratio),
                    'change_mean': float(change_mean),
                    'change_std': float(change_std),
                },
                'critical_indicators': {
                    'critical_concentration': bool(critical_concentration),
                    'concentration_gradient_top1_to_top3': float(concentration_gradient_top1_to_top3),
                    'concentration_gradient_top3_to_top5': float(concentration_gradient_top3_to_top5),
                },
                'a_share_patterns': {
                    'wash_trade_suspicion': float(wash_trade_suspicion),
                    'control_suspicion': float(control_suspicion),
                    'retail_dominated': float(retail_dominated),
                },
                'composite_indicators': {
                    'final_concentration': float(final_concentration),
                    'composite_concentration': float(composite_concentration),
                }
            }
            # 更新extra_metrics字段
            if not hasattr(self, 'extra_metrics') or self.extra_metrics is None:
                self.extra_metrics = {}
            if 'concentration_analysis' not in self.extra_metrics:
                self.extra_metrics['concentration_analysis'] = {}
            # 合并分析结果
            self.extra_metrics['concentration_analysis'].update(detailed_analysis)
            return final_concentration
        except Exception as e:
            print(f"❌ [变化集中度] 计算异常: {e}")
            import traceback
            traceback.print_exc()
            # 返回基本集中度作为降级方案
            if 'changes' in locals() and 'total_volume' in locals():
                try:
                    return concentration_top5
                except:
                    return 0.0
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






