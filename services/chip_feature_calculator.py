# 文件: services/calculators/chip_feature_calculator.py

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from decimal import Decimal

class ChipFeatureCalculator:
    """
    【V10.0 最终正确版】
    - 根本性修复: 废除了所有历史版本中错误的获利盘计算逻辑。
    - 回归定义: _calculate_winner_structure 方法现在严格按照模型字段的
                业务定义进行计算，确保在接收到真实、正确的收盘价后，
                能够准确反映市场的获利盘结构。
    - 功能增强: 新增了 'total_winner_rate' (总获利盘) 指标的计算。
    """
    def __init__(self, daily_chips_df: pd.DataFrame, context_data: dict):
        # 初始化函数：将 Decimal 类型转换为 float，便于计算
        self.df = daily_chips_df.reset_index(drop=True)
        self.ctx = context_data
        
        for key in ['total_chip_volume', 'daily_turnover_volume', 'weight_avg_cost', 'close_price', 'high_price', 'low_price', 'prev_20d_close']:
            if key in self.ctx and isinstance(self.ctx[key], Decimal):
                self.ctx[key] = float(self.ctx[key])

    def calculate_all_metrics(self) -> dict:
        # 主计算函数：按顺序调用所有子计算模块
        if self.df.empty or not all(k in self.ctx for k in ['weight_avg_cost', 'close_price', 'total_chip_volume']):
            return {}

        peaks_info = self._calculate_peaks()
        concentration_info = self._calculate_concentration()
        winner_structure_info = self._calculate_winner_structure()
        pressure_support_info = self._calculate_pressure_support()
        turnover_info = self._calculate_effective_turnover()
        fund_flow_info = self._calculate_fund_flow_metrics()

        return {
            **peaks_info,
            **concentration_info,
            **winner_structure_info,
            **pressure_support_info,
            **turnover_info,
            **fund_flow_info
        }

    def _calculate_peaks(self) -> dict:
        # 筹码峰计算：此部分逻辑一直正确，无需修改
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
        # 筹码集中度计算：此部分逻辑一直正确，无需修改
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

    # ▼▼▼【代码修改 V10.0】: 修正获利盘计算逻辑 ▼▼▼
    def _calculate_winner_structure(self) -> dict:
        """
        【V10.0 最终修正】
        严格按照模型字段的业务定义进行计算，并增加总获利盘指标。
        """
        close_price = self.ctx.get('close_price')
        prev_20d_close = self.ctx.get('prev_20d_close')

        # 如果缺少关键价格数据，则无法计算
        if not close_price or pd.isna(prev_20d_close):
            return {'winner_rate_short_term': None, 'winner_rate_long_term': None, 'total_winner_rate': None}

        # 定义1: 总获利盘 = 持仓成本 < 当天收盘价
        total_winners_df = self.df[self.df['price'] < close_price]
        total_winner_rate = total_winners_df['percent'].sum()

        # 定义2: 长期锁定盘 = 持仓成本 < 20日前收盘价
        long_term_winners_df = self.df[self.df['price'] < prev_20d_close]
        long_term_winner_rate = long_term_winners_df['percent'].sum()

        # 定义3: 短期获利盘 = 成本低于收盘价，但高于20日前收盘价
        short_term_winners_df = self.df[(self.df['price'] < close_price) & (self.df['price'] >= prev_20d_close)]
        short_term_winner_rate = short_term_winners_df['percent'].sum()
        
        return {
            'winner_rate_short_term': short_term_winner_rate,
            'winner_rate_long_term': long_term_winner_rate,
            'total_winner_rate': total_winner_rate,
        }
    # ▲▲▲【代码修改 V10.0】▲▲▲

    def _calculate_pressure_support(self) -> dict:
        # 上方压力与下方支撑计算：此部分逻辑一直正确，无需修改
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
        # 有效换手计算：此部分逻辑一直正确，无需修改
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

    def _calculate_fund_flow_metrics(self) -> dict:
        """
        【V5.0 三源合一版】
        - 核心升级: 实现“内部评估、外部确认、最终裁决”的三层交叉验证体系。
        - 算法: 1. 基于最原始的交易所数据(CY)计算我军的“内部评估”结果。
                 2. 引入同花顺(THS)和东方财富(DC)的数据作为“外部参照”。
                 3. 当三方结论一致时，生成高置信度的“共识”信号。
                 4. 当三方结论不一时，生成有价值的“分歧”信号。
        """
        # --- 1. 内部评估 (基于交易所原始数据) ---
        cy_keys = [
            'buy_sm_vol', 'buy_sm_amount', 'sell_sm_vol', 'sell_sm_amount',
            'buy_md_vol', 'buy_md_amount', 'sell_md_vol', 'sell_md_amount',
            'buy_lg_vol', 'buy_lg_amount', 'sell_lg_vol', 'sell_lg_amount',
            'buy_elg_vol', 'buy_elg_amount', 'sell_elg_vol', 'sell_elg_amount',
        ]
        # 如果缺少最核心的交易所数据，则无法进行任何计算
        if not all(pd.notna(self.ctx.get(key)) for key in cy_keys):
            trade_date_str = self.ctx.get('trade_time', '未知日期')
            print(f"调试信息: [{trade_date_str}] 缺少核心交易所资金流数据，跳过所有资金流指标计算。")
            return {}

        main_force_buy_amount = float(self.ctx.get('buy_lg_amount', 0)) + float(self.ctx.get('buy_elg_amount', 0))
        main_force_sell_amount = float(self.ctx.get('sell_lg_amount', 0)) + float(self.ctx.get('sell_elg_amount', 0))
        internal_main_force_net_amount = (main_force_buy_amount - main_force_sell_amount) * 10000  # 转换为元

        retail_buy_vol = float(self.ctx.get('buy_sm_vol', 0)) + float(self.ctx.get('buy_md_vol', 0))
        retail_sell_vol = float(self.ctx.get('sell_sm_vol', 0)) + float(self.ctx.get('sell_md_vol', 0))
        internal_retail_net_volume = (retail_buy_vol - retail_sell_vol) * 100  # 转换为股

        # --- 2. 外部参照 (获取同花顺与东财的公开情报) ---
        # 注意：预计算任务需要确保这些字段已合并到 context 中
        ths_net_amount = float(self.ctx.get('ths_buy_lg_amount', 0)) # 同花顺大单净额
        dc_net_amount = float(self.ctx.get('dc_net_amount', 0))     # 东方财富主力净额

        # --- 3. 最终裁决 (融合三方情报，生成共识与分歧信号) ---
        # 获取三方对主力资金流向的判断 (-1: 流出, 0: 平, 1: 流入)
        cy_sign = np.sign(internal_main_force_net_amount)
        ths_sign = np.sign(ths_net_amount)
        dc_sign = np.sign(dc_net_amount)
        
        # 共识性流入：三方都判断为净流入
        consensus_inflow = (cy_sign > 0 and ths_sign > 0 and dc_sign > 0)
        
        # 共识性流出：三方都判断为净流出
        consensus_outflow = (cy_sign < 0 and ths_sign < 0 and dc_sign < 0)
        
        # 分歧：三方判断不完全一致 (排除了全流入、全流出、全为0的情况)
        sign_sum = cy_sign + ths_sign + dc_sign
        divergence = not (abs(sign_sum) == 3 or sign_sum == 0)

        return {
            # 内部评估结果
            'main_force_net_inflow_amount': internal_main_force_net_amount,
            'retail_net_inflow_volume': internal_retail_net_volume,
            # 外部参照数据
            'ths_main_force_net_amount': ths_net_amount,
            'dc_main_force_net_amount': dc_net_amount,
            # 最终融合信号
            'consensus_main_force_inflow': consensus_inflow,
            'consensus_main_force_outflow': consensus_outflow,
            'fund_flow_divergence': divergence,
        }









