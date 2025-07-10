import pandas as pd
import numpy as np
from scipy.signal import find_peaks

class ChipFeatureCalculator:
    """
    【V4.0 全息版 - 高级筹码指标计算器】
    一个独立的、纯粹的计算类，负责从处理好的日内筹码和行情数据中，
    计算出 AdvancedChipMetrics 模型所需的所有指标。
    """
    def __init__(self, daily_chips_df: pd.DataFrame, context_data: dict):
        """
        初始化计算器。
        Args:
            daily_chips_df (pd.DataFrame): 当天已按价格排序的原始筹码分布数据，包含 'price', 'percent' 列。
            context_data (dict): 包含当天所有上下文信息的字典，如：
                'weight_avg_cost', 'close_price', 'total_chip_volume', 
                'daily_turnover_volume', 'prev_20d_close'
        """
        self.df = daily_chips_df
        self.ctx = context_data

    def calculate_all_metrics(self) -> dict:
        """
        执行所有指标的计算，并返回一个字典。
        """
        if self.df.empty or not all(k in self.ctx for k in ['weight_avg_cost', 'close_price', 'total_chip_volume']):
            return {}

        # 1. 识别筹码峰
        peaks_info = self._calculate_peaks()

        # 2. 计算筹码集中度
        concentration_info = self._calculate_concentration()

        # 3. 计算获利盘结构
        winner_structure_info = self._calculate_winner_structure()

        # 4. 计算压力与支撑
        pressure_support_info = self._calculate_pressure_support()
        
        # 5. 计算核心区换手
        turnover_info = self._calculate_effective_turnover()

        # 合并所有结果
        return {
            **peaks_info,
            **concentration_info,
            **winner_structure_info,
            **pressure_support_info,
            **turnover_info
        }

    def _calculate_peaks(self) -> dict:
        # 使用 find_peaks 科学地寻找波峰
        peaks, properties = find_peaks(self.df['percent'], prominence=0.1, width=1)
        
        if len(peaks) == 0:
            peak_idx = self.df['percent'].idxmax()
            return {
                'peak_cost': self.df.loc[peak_idx, 'price'],
                'peak_percent': self.df.loc[peak_idx, 'percent'],
                'peak_volume': int(self.df.loc[peak_idx, 'percent'] / 100 * self.ctx['total_chip_volume']),
                'peak_stability': 1.0,
                'is_multi_peak': False,
                'secondary_peak_cost': None,
                'peak_distance_ratio': None,
                'peak_strength_ratio': None,
            }

        prominences = properties['prominences']
        main_peak_idx_in_peaks = np.argmax(prominences)
        main_peak_df_idx = peaks[main_peak_idx_in_peaks]
        
        main_peak_cost = self.df.loc[main_peak_df_idx, 'price']
        main_peak_percent = self.df.loc[main_peak_df_idx, 'percent']
        main_peak_volume = int(main_peak_percent / 100 * self.ctx['total_chip_volume'])
        peak_stability = prominences[main_peak_idx_in_peaks] / self.df['percent'].mean()

        result = {
            'peak_cost': main_peak_cost,
            'peak_percent': main_peak_percent,
            'peak_volume': main_peak_volume,
            'peak_stability': peak_stability,
            'is_multi_peak': len(peaks) > 1,
        }

        if result['is_multi_peak']:
            remaining_peaks_indices = np.delete(peaks, main_peak_idx_in_peaks)
            remaining_prominences = np.delete(prominences, main_peak_idx_in_peaks)
            secondary_peak_df_idx = remaining_peaks_indices[np.argmax(remaining_prominences)]
            
            result['secondary_peak_cost'] = self.df.loc[secondary_peak_df_idx, 'price']
            result['peak_distance_ratio'] = abs(main_peak_cost - result['secondary_peak_cost']) / main_peak_cost
            result['peak_strength_ratio'] = remaining_prominences.max() / prominences[main_peak_idx_in_peaks]
        else:
            result.update({'secondary_peak_cost': None, 'peak_distance_ratio': None, 'peak_strength_ratio': None})
            
        return result

    def _calculate_concentration(self) -> dict:
        self.df['cumulative_percent'] = self.df['percent'].cumsum()
        
        def get_concentration_range(target_pct: float) -> tuple:
            min_width = float('inf')
            best_range = (None, None)
            for i in range(len(self.df)):
                target_cum_val = self.df.loc[i, 'cumulative_percent'] + target_pct
                end_row = self.df[self.df['cumulative_percent'] >= target_cum_val]
                if not end_row.empty:
                    j = end_row.index[0]
                    width = self.df.loc[j, 'price'] - self.df.loc[i, 'price']
                    if width < min_width:
                        min_width = width
                        best_range = (self.df.loc[i, 'price'], self.df.loc[j, 'price'])
            return min_width, best_range

        width_90, _ = get_concentration_range(90.0)
        _, range_70 = get_concentration_range(70.0)
        
        self.ctx['cost_range_70pct'] = range_70 # 将70%成本区间存入上下文，供后续使用

        return {
            'concentration_90pct': width_90 / self.ctx['weight_avg_cost'] if width_90 != float('inf') else None,
        }

    def _calculate_winner_structure(self) -> dict:
        close_price = self.ctx.get('close_price')
        prev_20d_close = self.ctx.get('prev_20d_close')
        if not close_price or not prev_20d_close:
            return {'winner_rate_short_term': None, 'winner_rate_long_term': None}

        short_term_winners = self.df[(self.df['price'] < close_price) & (self.df['price'] >= prev_20d_close)]
        long_term_winners = self.df[self.df['price'] < prev_20d_close]
        
        return {
            'winner_rate_short_term': short_term_winners['percent'].sum(),
            'winner_rate_long_term': long_term_winners['percent'].sum(),
        }

    def _calculate_pressure_support(self) -> dict:
        close_price = self.ctx.get('close_price')
        if not close_price:
            return {}
            
        pressure_range_upper = close_price * 1.02
        support_range_lower = close_price * 0.98
        
        pressure_df = self.df[(self.df['price'] > close_price) & (self.df['price'] <= pressure_range_upper)]
        support_df = self.df[(self.df['price'] < close_price) & (self.df['price'] >= support_range_lower)]
        
        pressure_pct = pressure_df['percent'].sum()
        support_pct = support_df['percent'].sum()
        
        return {
            'pressure_above': pressure_pct,
            'support_below': support_pct,
            'pressure_above_volume': int(pressure_pct / 100 * self.ctx['total_chip_volume']),
            'support_below_volume': int(support_pct / 100 * self.ctx['total_chip_volume']),
        }
        
    def _calculate_effective_turnover(self) -> dict:
        # 假设原始筹码分布是基于当日收盘后的换手计算的
        # 这里的计算需要一个前提：我们能获取到当日的成交量明细（分时图），这通常很难
        # 我们采用一种简化但有效的方法：假设当日的成交均匀分布在全天价格范围内
        # 然后看这个范围和70%成本区的交集
        low_price = self.ctx.get('low_price')
        high_price = self.ctx.get('high_price')
        cost_range_70_low, cost_range_70_high = self.ctx.get('cost_range_70pct', (None, None))
        
        if not all([low_price, high_price, cost_range_70_low, cost_range_70_high]):
            return {'turnover_volume_in_cost_range_70pct': None}
            
        # 计算价格区间的交集
        intersection_low = max(low_price, cost_range_70_low)
        intersection_high = min(high_price, cost_range_70_high)
        
        daily_price_range = high_price - low_price
        intersection_range = intersection_high - intersection_low
        
        if daily_price_range > 0 and intersection_range > 0:
            effective_ratio = intersection_range / daily_price_range
            turnover_volume = int(self.ctx['daily_turnover_volume'] * effective_ratio)
        else:
            turnover_volume = 0
            
        return {'turnover_volume_in_cost_range_70pct': turnover_volume}
