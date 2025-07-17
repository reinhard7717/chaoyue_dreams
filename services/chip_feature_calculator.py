import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from decimal import Decimal

class ChipFeatureCalculator:
    """
    【V4.2 全息版 - .loc/.iloc 修正】
    修正了因混用 pandas 的 .loc 和 .iloc 导致的 KeyError。
    """
    def __init__(self, daily_chips_df: pd.DataFrame, context_data: dict):
        """
        初始化计算器。
        """
        # ▼▼▼【核心修正】: 在初始化时就重置索引 ▼▼▼
        # 这样做可以确保 DataFrame 的索引是从0开始的连续整数，
        # 使得后续无论是基于位置的算法(如find_peaks)还是基于标签的查找都能统一处理，
        # 从根本上避免 .loc 和 .iloc 的混淆问题。
        self.df = daily_chips_df.reset_index(drop=True)
        # ▲▲▲【核心修正结束】▲▲▲
        
        self.ctx = context_data
        
        for key in ['total_chip_volume', 'daily_turnover_volume', 'weight_avg_cost', 'close_price', 'high_price', 'low_price', 'prev_20d_close']:
            if key in self.ctx and isinstance(self.ctx[key], Decimal):
                self.ctx[key] = float(self.ctx[key])

    def calculate_all_metrics(self) -> dict:
        """
        执行所有指标的计算，并返回一个字典。
        """
        if self.df.empty or not all(k in self.ctx for k in ['weight_avg_cost', 'close_price', 'total_chip_volume']):
            return {}

        peaks_info = self._calculate_peaks()
        concentration_info = self._calculate_concentration()
        winner_structure_info = self._calculate_winner_structure()
        pressure_support_info = self._calculate_pressure_support()
        turnover_info = self._calculate_effective_turnover()

        return {
            **peaks_info,
            **concentration_info,
            **winner_structure_info,
            **pressure_support_info,
            **turnover_info
        }

    def _calculate_peaks(self) -> dict:
        peaks, properties = find_peaks(self.df['percent'], prominence=0.1, width=1)
        
        if len(peaks) == 0:
            # idxmax() 返回的是索引标签，由于我们已经 reset_index，它现在也是整数位置
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
        
        # ▼▼▼【核心修正】: 使用 .iloc 访问由 find_peaks 返回的整数位置 ▼▼▼
        # 虽然 reset_index 后 .loc 也能工作，但使用 .iloc 更能明确表达意图
        main_peak_cost = self.df.iloc[main_peak_df_idx]['price']
        main_peak_percent = self.df.iloc[main_peak_df_idx]['percent']
        # ▲▲▲【核心修正结束】▲▲▲
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
            
            # ▼▼▼【核心修正】: 同样使用 .iloc ▼▼▼
            result['secondary_peak_cost'] = self.df.iloc[secondary_peak_df_idx]['price']
            # ▲▲▲【核心修正结束】▲▲▲
            result['peak_distance_ratio'] = abs(main_peak_cost - result['secondary_peak_cost']) / main_peak_cost if main_peak_cost > 0 else None
            result['peak_strength_ratio'] = remaining_prominences.max() / prominences[main_peak_idx_in_peaks]
        else:
            result.update({'secondary_peak_cost': None, 'peak_distance_ratio': None, 'peak_strength_ratio': None})
            
        return result

    def _calculate_concentration(self) -> dict:
        # 由于在 __init__ 中已经 reset_index，这里的 self.df 索引已经是 0, 1, 2...
        # 所以原来的代码现在是安全的
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
        
        self.ctx['cost_range_70pct'] = range_70

        return {
            'concentration_90pct': width_90 / self.ctx['weight_avg_cost'] if width_90 != float('inf') and self.ctx['weight_avg_cost'] > 0 else None,
        }

    def _calculate_winner_structure(self) -> dict:
        """
        【V5.0 获利结构重铸版】
        - 核心修复: 彻底重构了获利盘的计算逻辑，解决了旧算法在下跌趋势中
                    短期获利盘恒为0的致命缺陷。
        - 新逻辑:
          1. 计算“总获利盘”（成本低于当前收盘价）。
          2. 计算“长期锁定盘”（成本低于20日前收盘价）。
          3. “短期获利盘”被精确定义为“总获利盘”与“长期锁定盘”之差。
        """
        close_price = self.ctx.get('close_price')
        prev_20d_close = self.ctx.get('prev_20d_close')

        # 如果关键价格缺失，无法计算，直接返回
        if not close_price or not prev_20d_close or pd.isna(prev_20d_close):
            return {'winner_rate_short_term': None, 'winner_rate_long_term': None}

        # 1. 计算总获利盘：所有成本低于当前收盘价的筹码比例
        total_winners_df = self.df[self.df['price'] < close_price]
        total_winner_rate = total_winners_df['percent'].sum()

        # 2. 计算长期锁定盘：所有成本低于20日前收盘价的筹码比例
        long_term_winners_df = self.df[self.df['price'] < prev_20d_close]
        long_term_winner_rate = long_term_winners_df['percent'].sum()

        # 3. 计算短期获利盘：总获利盘与长期锁定盘之差
        #    这代表了在过去20天内新产生的获利盘
        #    使用 max(0, ...) 是为了防止因价格剧烈波动导致轻微的负值
        short_term_winner_rate = max(0, total_winner_rate - long_term_winner_rate)
        
        return {
            'winner_rate_short_term': short_term_winner_rate,
            'winner_rate_long_term': long_term_winner_rate,
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
        low_price = self.ctx.get('low_price')
        high_price = self.ctx.get('high_price')
        cost_range_70_low, cost_range_70_high = self.ctx.get('cost_range_70pct', (None, None))
        
        if not all([low_price, high_price, cost_range_70_low, cost_range_70_high]):
            return {'turnover_volume_in_cost_range_70pct': None}
            
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
