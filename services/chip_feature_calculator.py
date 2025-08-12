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
        
        if not self.df.empty:
            percent_sum = self.df['percent'].sum()
            # 检查总和是否接近100，且大于0，以避免除以0的错误
            if not np.isclose(percent_sum, 100.0) and percent_sum > 0:
                print(f"    -> [数据清洗] 注意：检测到筹码分布总和为 {percent_sum:.2f}，不等于100。正在执行强制归一化...")
                self.df['percent'] = (self.df['percent'] / percent_sum) * 100.0
        
        for key in ['total_chip_volume', 'daily_turnover_volume', 'close_price', 'high_price', 'low_price', 'prev_20d_close']:
            if key in self.ctx and isinstance(self.ctx[key], Decimal):
                self.ctx[key] = float(self.ctx[key])

    def calculate_all_metrics(self) -> dict:
        """
        【V11.0 健壮版】
        - 修复: 彻底解决 UnboundLocalError 问题。
        - 重构: 优化计算流程，将计算分为“基础指标 -> 增强上下文 -> 升维指标”三步，
                确保所有升维计算模块共享同一个、完整的上下文，逻辑更清晰，代码更健壮。
        """
        # --- 0. 前置检查 ---
        if self.df.empty or not all(k in self.ctx for k in ['weight_avg_cost', 'close_price', 'total_chip_volume']):
            return {}

        # --- 1. 基础指标计算 ---
        summary_info = self._calculate_summary_metrics()
        self.ctx.update(summary_info)
        peaks_info = self._calculate_peaks()
        concentration_info = self._calculate_concentration()
        winner_structure_info = self._calculate_winner_structure()
        pressure_support_info = self._calculate_pressure_support()
        turnover_info = self._calculate_effective_turnover()
        turnover_structure_info = self._calculate_turnover_structure()
        turnover_at_peak_info = self._calculate_turnover_at_peak(peaks_info)
        
        # --- 2. 构建升维计算所需的“增强上下文” ---
        # 将所有基础计算结果合并到上下文中，供后续所有升维计算模块使用
        context_for_derived_metrics = {
            **self.ctx, 
            **peaks_info, 
            **concentration_info, 
            **winner_structure_info,
        }

        # --- 3. 升维指标计算 ---
        advanced_structure_info = self._calculate_advanced_structures(context_for_derived_metrics)
        fault_info = self._calculate_chip_fault(context_for_derived_metrics)

        # --- 4. 合并所有结果并返回 ---
        all_metrics = {
            **peaks_info,
            **concentration_info,
            **winner_structure_info,
            **pressure_support_info,
            **turnover_info,
            **turnover_structure_info,
            **turnover_at_peak_info,
            **advanced_structure_info,
            **fault_info
        }
        if 'total_winner_rate' in summary_info:
            all_metrics['total_winner_rate'] = summary_info['total_winner_rate']

        # 这两个值是计算 peak_absorption_intensity 的中间产物，不需要存入数据库
        all_metrics.pop('peak_range_low', None)
        all_metrics.pop('peak_range_high', None)

        # --- 5. 返回清理后的最终结果 ---
        return all_metrics

    def _calculate_summary_metrics(self) -> dict:
        """
        【V12.0 新增】摘要指标自主计算模块
        基于原始筹码分布，计算加权平均成本和总获利盘。
        """
        normalized_percent = self.df['percent'] / 100.0
        # 1. 计算加权平均成本 (weight_avg_cost)
        weight_avg_cost = np.average(self.df['price'], weights=normalized_percent)

        # 2. 计算总获利盘 (total_winner_rate)
        close_price = self.ctx.get('close_price')
        if close_price:
            winners_df = self.df[self.df['price'] < close_price]
            total_winner_rate = winners_df['percent'].sum()
        else:
            total_winner_rate = 0.0

        return {
            'weight_avg_cost': weight_avg_cost,
            'total_winner_rate': total_winner_rate,
        }

    def _calculate_peaks(self) -> dict:
        # 【代码修改】: 增加 width 参数，让 find_peaks 计算山峰宽度
        peaks, properties = find_peaks(self.df['percent'], prominence=0.1, width=1)
        
        if len(peaks) == 0:
            peak_idx = self.df['percent'].idxmax()
            peak_price = self.df.loc[peak_idx, 'price']
            return {
                'peak_cost': peak_price,
                'peak_percent': self.df.loc[peak_idx, 'percent'],
                'peak_volume': int(self.df.loc[peak_idx, 'percent'] / 100 * self.ctx['total_chip_volume']),
                'peak_stability': 1.0,
                'is_multi_peak': False,
                'secondary_peak_cost': None,
                'peak_distance_ratio': None,
                'peak_strength_ratio': None,
                # 【新增】: 为单峰情况提供一个默认的范围
                'peak_range_low': peak_price * 0.99,
                'peak_range_high': peak_price * 1.01,
            }

        prominences = properties['prominences']
        main_peak_idx_in_peaks = np.argmax(prominences)
        main_peak_df_idx = peaks[main_peak_idx_in_peaks]
        
        main_peak_cost = self.df.iloc[main_peak_df_idx]['price']
        main_peak_percent = self.df.iloc[main_peak_df_idx]['percent']
        main_peak_volume = int(main_peak_percent / 100 * self.ctx['total_chip_volume'])
        peak_stability = prominences[main_peak_idx_in_peaks] / self.df['percent'].mean() if self.df['percent'].mean() > 0 else 1.0

        # ▼▼▼【核心修改】: 提取主峰的精确左右边界 ▼▼▼
        # 'left_ips' 和 'right_ips' 是 find_peaks 返回的、基于插值计算的精确边界索引（可以是浮点数）
        # 我们需要将它们转换为实际的价格
        left_boundary_idx = properties['left_ips'][main_peak_idx_in_peaks]
        right_boundary_idx = properties['right_ips'][main_peak_idx_in_peaks]
        
        # 使用 np.interp 进行线性插值，从浮点索引精确映射到价格
        price_indices = self.df.index.to_numpy()
        price_values = self.df['price'].to_numpy()
        peak_range_low = np.interp(left_boundary_idx, price_indices, price_values)
        peak_range_high = np.interp(right_boundary_idx, price_indices, price_values)
        # ▲▲▲【核心修改】▲▲▲

        result = {
            'peak_cost': main_peak_cost,
            'peak_percent': main_peak_percent,
            'peak_volume': main_peak_volume,
            'peak_stability': peak_stability,
            'is_multi_peak': len(peaks) > 1,
            # 【新增】: 将计算出的精确范围加入返回结果
            'peak_range_low': peak_range_low,
            'peak_range_high': peak_range_high,
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

    def _calculate_winner_structure(self) -> dict:
        """
        【V12.0 简化版】
        - 核心简化: 不再计算 total_winner_rate，因为它已在 _calculate_summary_metrics 中计算。
        """
        close_price = self.ctx.get('close_price')
        prev_20d_close = self.ctx.get('prev_20d_close')

        if not close_price or pd.isna(prev_20d_close):
            return {'winner_rate_short_term': None, 'winner_rate_long_term': None}

        long_term_winners_df = self.df[self.df['price'] < prev_20d_close]
        long_term_winner_rate = long_term_winners_df['percent'].sum()

        short_term_winners_df = self.df[(self.df['price'] < close_price) & (self.df['price'] >= prev_20d_close)]
        short_term_winner_rate = short_term_winners_df['percent'].sum()
        
        return {
            'winner_rate_short_term': short_term_winner_rate,
            'winner_rate_long_term': long_term_winner_rate,
        }

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
        【V6.1 全天候作战版】
        - 核心升级: 采用弹性融合逻辑，智能处理外部数据源缺失问题。
        - 算法:
          1. 永远以最可靠的交易所数据(CY)为基准。
          2. 检查同花顺(THS)和东方财富(DC)数据是否有效(非None,非NaN)。
          3. 统计有效数据源的数量(1-3个)。
          4. 基于有效数据源，重新定义“共识”与“分歧”：
             - 共识: 至少有2个有效源，且它们的判断完全一致。
             - 分歧: 至少有2个有效源，且它们的判断不一致。
             - 单一信源: 只有1个有效源时，不产生共识或分歧信号。
        """
        # --- 1. 内部评估 (基准数据) ---
        cy_keys = [
            'buy_lg_amount', 'buy_elg_amount', 'sell_lg_amount', 'sell_elg_amount',
            'buy_sm_vol', 'buy_md_vol', 'sell_sm_vol', 'sell_md_vol'
        ]
        # 如果连最核心的交易所数据都没有，则直接放弃
        if not all(pd.notna(self.ctx.get(key)) for key in cy_keys):
            trade_date_str = self.ctx.get('trade_time', '未知日期')
            print(f"调试信息: [{trade_date_str}] 缺少核心交易所资金流数据，跳过所有资金流指标计算。")
            return {}

        # 计算内部评估结果
        main_force_buy_amount = float(self.ctx.get('buy_lg_amount', 0)) + float(self.ctx.get('buy_elg_amount', 0))
        main_force_sell_amount = float(self.ctx.get('sell_lg_amount', 0)) + float(self.ctx.get('sell_elg_amount', 0))
        internal_main_force_net_amount = (main_force_buy_amount - main_force_sell_amount) * 10000

        retail_buy_vol = float(self.ctx.get('buy_sm_vol', 0)) + float(self.ctx.get('buy_md_vol', 0))
        retail_sell_vol = float(self.ctx.get('sell_sm_vol', 0)) + float(self.ctx.get('sell_md_vol', 0))
        internal_retail_net_volume = (retail_buy_vol - retail_sell_vol) * 100

        # --- 2. 外部参照与有效性检查 ---
        # 获取外部数据，如果不存在或为NaN，则为None
        ths_net_amount = self.ctx.get('ths_buy_lg_amount')
        dc_net_amount = self.ctx.get('dc_net_amount')
        
        # 检查数据有效性 (pd.notna可以正确处理None和NaN)
        is_cy_valid = True # 运行到这里，CY必然有效
        is_ths_valid = pd.notna(ths_net_amount)
        is_dc_valid = pd.notna(dc_net_amount)

        # --- 3. 弹性融合裁决 ---
        # 存储所有有效源的判断符号 (-1, 0, 1)
        valid_signs = []
        if is_cy_valid:
            valid_signs.append(np.sign(internal_main_force_net_amount))
        if is_ths_valid:
            valid_signs.append(np.sign(float(ths_net_amount)))
        if is_dc_valid:
            valid_signs.append(np.sign(float(dc_net_amount)))
        
        source_count = len(valid_signs)
        consensus_inflow = False
        consensus_outflow = False
        divergence = False

        # 只有当有效数据源大于等于2个时，才可能产生“共识”或“分歧”
        if source_count >= 2:
            # 检查所有有效源的判断是否完全一致
            is_all_same = all(s == valid_signs[0] for s in valid_signs)
            
            if is_all_same:
                # 如果判断一致，则根据方向确定是共识流入还是共识流出
                if valid_signs[0] > 0:
                    consensus_inflow = True
                elif valid_signs[0] < 0:
                    consensus_outflow = True
            else:
                # 如果判断不一致，则标记为分歧
                divergence = True

        return {
            # 内部评估结果
            'main_force_net_inflow_amount': internal_main_force_net_amount,
            'retail_net_inflow_volume': internal_retail_net_volume,
            # 外部参照数据 (即使无效也存储None)
            'ths_main_force_net_amount': float(ths_net_amount) if is_ths_valid else None,
            'dc_main_force_net_amount': float(dc_net_amount) if is_dc_valid else None,
            # 最终融合信号
            'fund_flow_data_source_count': source_count,
            'consensus_main_force_inflow': consensus_inflow,
            'consensus_main_force_outflow': consensus_outflow,
            'fund_flow_divergence': divergence,
        }

    def _calculate_advanced_structures(self, context: dict) -> dict:
        """
        【V6.0 新增】计算“控盘度”、“利润质量”、“价码关系”三大升维指标。
        """
        results = {}
        
        # --- 1. 控盘度指标 (Control Metrics) ---
        peak_volume = context.get('peak_volume')
        total_float_share = self.ctx.get('total_chip_volume') # total_chip_volume 就是流通股本
        if peak_volume is not None and total_float_share and total_float_share > 0:
            results['peak_control_ratio'] = (peak_volume / total_float_share) * 100

        # 计算筹码峰吸筹强度需要筹码峰的宽度信息
        peak_cost = context.get('peak_cost')
        daily_turnover = self.ctx.get('daily_turnover_volume')
        # 从 context 中直接获取由 _calculate_peaks 计算好的精确范围
        peak_range_low = context.get('peak_range_low')
        peak_range_high = context.get('peak_range_high')

        if daily_turnover and daily_turnover > 0 and peak_range_low is not None and peak_range_high is not None:
            # 使用精确的、数据驱动的范围来估算换手
            turnover_in_peak_range = self._get_turnover_in_range(peak_range_low, peak_range_high)
            results['peak_absorption_intensity'] = turnover_in_peak_range / daily_turnover
        else:
            results['peak_absorption_intensity'] = None

        # --- 2. 利润质量指标 (Profit Quality Metrics) ---
        close_price = self.ctx.get('close_price')
        if close_price:
            winners_df = self.df[self.df['price'] < close_price]
            if not winners_df.empty:
                # 计算获利盘的加权平均成本
                winner_avg_cost = np.average(winners_df['price'], weights=winners_df['percent'])
                results['winner_avg_cost'] = winner_avg_cost
                # 计算获利盘的安全垫
                if winner_avg_cost > 0:
                    results['winner_profit_margin'] = ((close_price - winner_avg_cost) / winner_avg_cost) * 100

        # --- 3. 价码关系指标 (Price-Chip Relation Metrics) ---
        if close_price and peak_cost and peak_cost > 0:
            results['price_to_peak_ratio'] = close_price / peak_cost
        
        # 计算筹码Z-Score
        if close_price:
            weighted_mean = self.ctx.get('weight_avg_cost')
            # 计算加权标准差
            weighted_std = np.sqrt(np.average((self.df['price'] - weighted_mean)**2, weights=self.df['percent']))
            if weighted_std and weighted_std > 0:
                results['chip_zscore'] = (close_price - weighted_mean) / weighted_std

        # --- 4. 超级指标: 筹码健康分 (Chip Health Score) ---
        # 这是一个可以持续优化的专家打分系统，这里提供一个初始版本
        score = 50.0 # 基础分
        
        # ▼▼▼【代码修改】: 增加对 None 值的健壮性检查 ▼▼▼
        # 集中度越高越好
        conc_90 = context.get('concentration_90pct') # 直接获取值，可能是 None
        if conc_90 is not None: # 只有在值有效时才进行计算
            score += (0.3 - min(conc_90, 0.3)) * 100 # 集中度低于30%开始加分，最多加30分
        # ▲▲▲【代码修改】▲▲▲

        # 集中度在收敛加分
        conc_slope = context.get('concentration_90pct_slope_5d', 0)
        if conc_slope < 0: score += 5
        # 获利盘安全垫越高越好
        profit_margin = results.get('winner_profit_margin', 0)
        score += min(profit_margin, 20) # 最多加20分
        # 股价在主峰之上加分
        price_ratio = results.get('price_to_peak_ratio', 1.0)
        if price_ratio > 1.05: score += 10
        # 共识性流入加分
        if context.get('consensus_main_force_inflow'): score += 15
        # 共识性流出扣分
        if context.get('consensus_main_force_outflow'): score -= 20
        
        results['chip_health_score'] = max(0, min(100, round(score, 2))) # 确保分数在0-100之间

        return results

    def _calculate_turnover_structure(self) -> dict:
        """
        【V2.0 宏观统计版】计算成交量微观结构。
        - 核心算法: 基于一个更稳健的宏观假设——当日成交量的构成，
                    由全市场获利盘和套牢盘的存量比例决定。
        - 公式: 获利盘成交占比 ≈ 总获利盘比例 / (总获利盘比例 + 总套牢盘比例)
        """
        close_price = self.ctx.get('close_price')
        if not close_price:
            return {}

        # 1. 严格按照收盘价，划分全市场的获利盘和套牢盘
        #    这里不考虑当日价格波动，只看收盘后的最终状态。
        winners_df = self.df[self.df['price'] < close_price]
        losers_df = self.df[self.df['price'] > close_price]

        # 2. 计算各自的总量（百分比）
        total_winner_percent = winners_df['percent'].sum()
        total_loser_percent = losers_df['percent'].sum()
        
        # 3. 计算总的“已表态”筹码（排除掉那些成本价和收盘价完全一样的）
        total_active_percent = total_winner_percent + total_loser_percent

        # 4. 如果市场所有筹码成本都一样（total_active_percent为0），则无法计算，返回空
        if total_active_percent == 0:
            return {
                'turnover_from_winners_ratio': None,
                'turnover_from_losers_ratio': None,
            }

        # 5. 计算各自在“已表态”筹码中的占比，这个比例就代表了它们贡献成交量的能力
        winner_contribution_ratio = total_winner_percent / total_active_percent
        loser_contribution_ratio = total_loser_percent / total_active_percent

        # 6. 转换为最终的百分比指标
        turnover_from_winners_ratio = winner_contribution_ratio * 100
        turnover_from_losers_ratio = loser_contribution_ratio * 100

        return {
            'turnover_from_winners_ratio': turnover_from_winners_ratio,
            'turnover_from_losers_ratio': turnover_from_losers_ratio,
        }

    def _calculate_turnover_at_peak(self, peaks_info: dict) -> dict:
        """
        【V1.0 新增】计算主峰成交占比。
        估算当日总成交量中，发生在主筹码峰价格区间的比例。
        """
        daily_turnover_vol = self.ctx.get('daily_turnover_volume')
        peak_range_low = peaks_info.get('peak_range_low')
        peak_range_high = peaks_info.get('peak_range_high')

        # 检查所有必要数据是否存在且有效
        if not all([daily_turnover_vol, peak_range_low, peak_range_high]) or daily_turnover_vol == 0:
            return {'turnover_at_peak_ratio': None}

        # 调用复用性极高的 _get_turnover_in_range 辅助函数来估算成交量
        turnover_in_peak_range = self._get_turnover_in_range(peak_range_low, peak_range_high)
        
        # 计算比例并转换为百分比
        ratio = turnover_in_peak_range / daily_turnover_vol
        turnover_at_peak_ratio = ratio * 100

        return {
            'turnover_at_peak_ratio': turnover_at_peak_ratio
        }

    def _get_turnover_in_range(self, low_bound, high_bound) -> float:
        """辅助函数：估算在某个价格区间的换手量"""
        low_price = self.ctx.get('low_price')
        high_price = self.ctx.get('high_price')
        daily_turnover = self.ctx.get('daily_turnover_volume')

        if not all([low_price, high_price, daily_turnover, daily_turnover > 0]):
            return 0

        intersection_low = max(low_price, low_bound)
        intersection_high = min(high_price, high_bound)
        
        daily_price_range = high_price - low_price
        intersection_range = intersection_high - intersection_low
        
        if daily_price_range > 0 and intersection_range > 0:
            effective_ratio = intersection_range / daily_price_range
            return daily_turnover * effective_ratio
        return 0

    def _calculate_chip_fault(self, context: dict) -> dict:
        """
        【V11.0 新增】计算筹码断层指标。
        识别股价脱离核心成本区后形成的“真空地带”。
        """
        results = {}
        peak_cost = context.get('peak_cost')
        close_price = self.ctx.get('close_price')

        if not all([peak_cost, close_price]):
            return results

        # 1. 计算断层强度 (Fault Strength)
        # 定义：当前价格脱离主筹码峰的程度
        fault_strength = (close_price - peak_cost) / peak_cost if peak_cost > 0 else 0
        results['chip_fault_strength'] = fault_strength

        # 2. 识别断层真空区 (Fault Vacuum)
        # 定义：主筹码峰与当前价格之间，筹码的稀疏程度
        # 选取主峰上方1%到收盘价下方1%的区间作为“断层带”
        fault_zone_low = peak_cost * 1.01
        fault_zone_high = close_price * 0.99
        
        if fault_zone_high > fault_zone_low:
            fault_zone_df = self.df[(self.df['price'] >= fault_zone_low) & (self.df['price'] <= fault_zone_high)]
            # 真空区的筹码占比，越低越好
            vacuum_chip_percent = fault_zone_df['percent'].sum()
            results['chip_fault_vacuum_percent'] = vacuum_chip_percent
        else:
            results['chip_fault_vacuum_percent'] = None

        # 3. 最终断层信号 (Fault Signal)
        # 定义：断层强度足够大（如脱离成本区20%以上），且真空区足够“空”（如筹码占比低于5%）
        is_strong_fault = fault_strength > 0.20
        vacuum_percent = results.get('chip_fault_vacuum_percent') # 直接获取值，可能是 None
        # 只有在真空度被成功计算出来（不是None）的情况下，才进行比较
        if vacuum_percent is not None:
            is_vacuum_clear = vacuum_percent < 5.0
        else:
            # 如果真空度无法计算，则默认真空区不满足“清澈”的条件
            is_vacuum_clear = False
        results['is_chip_fault_formed'] = is_strong_fault and is_vacuum_clear

        return results





