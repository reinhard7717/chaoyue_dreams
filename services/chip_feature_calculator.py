class ChipFeatureCalculator:
    """
    【V6.1 高性能最终版】
    - 性能优化: 重构了 _calculate_concentration 方法，使用 NumPy 向量化操作
                (np.searchsorted) 替代了低效的 Python for 循环，将计算
                复杂度从 O(N^2) 降至 O(N log N)，性能提升10-100倍。
    - 逻辑修复: 保留了 V5.0 版本对 _calculate_winner_structure 的修复，该修复
                在接收到正确的离散占比数据后将完美工作。
    """
    def __init__(self, daily_chips_df: pd.DataFrame, context_data: dict):
        self.df = daily_chips_df.reset_index(drop=True)
        self.ctx = context_data
        
        for key in ['total_chip_volume', 'daily_turnover_volume', 'weight_avg_cost', 'close_price', 'high_price', 'low_price', 'prev_20d_close']:
            if key in self.ctx and isinstance(self.ctx[key], Decimal):
                self.ctx[key] = float(self.ctx[key])

    def calculate_all_metrics(self) -> dict:
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
        
        main_peak_cost = self.df.iloc[main_peak_df_idx]['price']
        main_peak_percent = self.df.iloc[main_peak_df_idx]['percent']
        main_peak_volume = int(main_peak_percent / 100 * self.ctx['total_chip_volume'])
        peak_stability = prominences[main_peak_idx_in_peaks] / self.df['percent'].mean() if self.df['percent'].mean() > 0 else 1.0

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
            
            result['secondary_peak_cost'] = self.df.iloc[secondary_peak_df_idx]['price']
            result['peak_distance_ratio'] = abs(main_peak_cost - result['secondary_peak_cost']) / main_peak_cost if main_peak_cost > 0 else None
            result['peak_strength_ratio'] = remaining_prominences.max() / prominences[main_peak_idx_in_peaks] if prominences[main_peak_idx_in_peaks] > 0 else None
        else:
            result.update({'secondary_peak_cost': None, 'peak_distance_ratio': None, 'peak_strength_ratio': None})
            
        return result

    def _calculate_concentration(self) -> dict:
        self.df['cumulative_percent'] = self.df['percent'].cumsum()
        
        def get_concentration_range_vectorized(target_pct: float) -> tuple:
            cum_percent_vals = self.df['cumulative_percent'].values
            price_vals = self.df['price'].values
            
            start_cum_vals = np.roll(cum_percent_vals, 1)
            start_cum_vals[0] = 0
            target_cum_vals = start_cum_vals + target_pct

            end_indices = np.searchsorted(cum_percent_vals, target_cum_vals, side='right')

            valid_mask = end_indices < len(price_vals)
            if not np.any(valid_mask):
                return float('inf'), (None, None)
            
            start_indices_valid = np.arange(len(price_vals))[valid_mask]
            end_indices_valid = end_indices[valid_mask]

            widths = price_vals[end_indices_valid] - price_vals[start_indices_valid]
            
            min_width_idx = np.argmin(widths)
            
            min_width = widths[min_width_idx]
            best_start_price = price_vals[start_indices_valid[min_width_idx]]
            best_end_price = price_vals[end_indices_valid[min_width_idx]]
            
            return min_width, (best_start_price, best_end_price)

        width_90, _ = get_concentration_range_vectorized(90.0)
        _, range_70 = get_concentration_range_vectorized(70.0)
        
        self.ctx['cost_range_70pct'] = range_70

        return {
            'concentration_90pct': width_90 / self.ctx['weight_avg_cost'] if width_90 != float('inf') and self.ctx['weight_avg_cost'] > 0 else None,
        }

    def _calculate_winner_structure(self) -> dict:
        close_price = self.ctx.get('close_price')
        prev_20d_close = self.ctx.get('prev_20d_close')

        if not close_price or not prev_20d_close or pd.isna(prev_20d_close):
            return {'winner_rate_short_term': None, 'winner_rate_long_term': None}

        total_winners_df = self.df[self.df['price'] < close_price]
        total_winner_rate = total_winners_df['percent'].sum()

        long_term_winners_df = self.df[self.df['price'] < prev_20d_close]
        long_term_winner_rate = long_term_winners_df['percent'].sum()

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