# 文件: services/chip_feature_calculator.py

import pandas as pd
import numpy as np
import datetime
from scipy.signal import find_peaks
from scipy.stats import entropy, percentileofscore, skew
from decimal import Decimal
import logging
logger = logging.getLogger(__name__)

class ChipFeatureCalculator:
    """
    【V11.0 · 战术中枢重构版】
    - 核心重构: 本类现在是所有筹码指标计算的唯一核心，接收原始数据并完成所有计算。
    - 逻辑注入: 新增了成交量微观结构和筹码集中度动态归因的计算逻辑。
    """
    def __init__(self, daily_chips_df: pd.DataFrame, context_data: dict):
        """
        【V12.3 · 上下文属性修复版】
        - 核心修复: 将 `context_data` 正确赋值给 `self.ctx` 属性，解决 `AttributeError`。
        - 核心优化: 移除 `__init__` 中冗余的 `self.stock_code`, `self.trade_date`, `self.debug_params` 属性赋值，这些信息已通过 `self.ctx` 可访问。
        - 核心重构: 建立“数据净化协议”，在入口处将所有可能为Decimal的上下文数据强制转换为float，根除类型不匹配错误。
        - 核心新增: 引入 `intraday_data` 作为统一的日内数据源，优先使用逐笔数据，否则回退到分钟数据。
        """
        self.df = daily_chips_df.reset_index(drop=True)
        self.ctx = context_data # 将 context_data 赋值给 self.ctx
        if not self.df.empty:
            percent_sum = self.df['percent'].sum()
            if not np.isclose(percent_sum, 100.0) and percent_sum > 0:
                self.df['percent'] = (self.df['percent'] / percent_sum) * 100.0
        decimal_to_float_fields = [
            'total_chip_volume', 'daily_turnover_volume', 'close_price', 'high_price', 'low_price', 'open_price',
            'pre_close', 'prev_20d_close', 'circ_mv', 'cost_5pct', 'cost_15pct', 'cost_50pct', 'cost_85pct',
            'cost_95pct', 'weight_avg', 'winner_rate', 'prev_concentration_90pct', 'prev_winner_avg_cost',
            'prev_total_chip_volume',
            'prev_high_20d', 'prev_low_20d', 'prev_day_20d_ago_close'
        ]
        for key in decimal_to_float_fields:
            if key in self.ctx and isinstance(self.ctx[key], Decimal):
                self.ctx[key] = float(self.ctx[key])
        # 从 self.ctx 中获取日内数据、交易日期和调试参数
        intraday_data_raw = self.ctx.get('intraday_data', pd.DataFrame()) # 使用 self.ctx
        trade_date = self.ctx.get('trade_date') # 使用 self.ctx
        debug_params = self.ctx.get('debug_params', {}) # 使用 self.ctx
        self.processed_intraday_df = self._prepare_intraday_data_features(intraday_data_raw, trade_date, debug_params)
        # 将处理后的日内数据存入 self.ctx，供后续方法使用
        self.ctx['processed_intraday_df'] = self.processed_intraday_df # 使用 self.ctx

    def calculate_all_metrics(self) -> dict:
        stock_code = self.ctx.get('stock_code', 'UNKNOWN')
        trade_date = self.ctx.get('trade_date', 'UNKNOWN')
        if self.df.empty:
            logger.warning(f"[{stock_code}] [{trade_date}] 筹码计算中止，原因：当日筹码分布数据(daily_chips_df)为空。")
            return {}
        required_keys = ['weight_avg', 'winner_rate', 'cost_95pct', 'cost_5pct', 'close_price', 'total_chip_volume']
        missing_keys = [k for k in required_keys if k not in self.ctx or self.ctx[k] is None or pd.isna(self.ctx[k])]
        if missing_keys:
            logger.warning(f"[{stock_code}] [{trade_date}] 筹码计算中止，原因：上下文(context_data)中缺少核心字段: {missing_keys}。")
            return {}
        all_metrics = {}
        summary_info = self._get_summary_metrics_from_context()
        all_metrics.update(summary_info)
        self.ctx.update(summary_info)
        static_structure_metrics = self._compute_static_structure_metrics()
        all_metrics.update(static_structure_metrics)
        self.ctx.update(static_structure_metrics)
        dominant_peak_cost = self.ctx.get('dominant_peak_cost')
        atr_14d = self.ctx.get('atr_14d')
        if pd.notna(dominant_peak_cost) and pd.notna(atr_14d) and atr_14d > 0:
            self.ctx['peak_range_low'] = dominant_peak_cost - atr_14d * 0.2
            self.ctx['peak_range_high'] = dominant_peak_cost + atr_14d * 0.2
        prev_gini = self.ctx.get('prev_metrics', {}).get('cost_gini_coefficient')
        today_gini = static_structure_metrics.get('cost_gini_coefficient')
        if pd.notna(prev_gini) and pd.notna(today_gini):
            self.ctx['cost_gini_coefficient_slope_1d'] = today_gini - prev_gini
        else:
            self.ctx['cost_gini_coefficient_slope_1d'] = 0.0
        intraday_dynamics_metrics = self._compute_intraday_dynamics_metrics(self.ctx)
        all_metrics.update(intraday_dynamics_metrics)
        self.ctx.update(intraday_dynamics_metrics)
        # =================================================================
        # 新增代码块：恢复对旧版日内动态指标计算方法的调用，以修复指标丢失问题
        legacy_intraday_metrics = self._compute_legacy_intraday_metrics(self.ctx)
        all_metrics.update(legacy_intraday_metrics)
        self.ctx.update(legacy_intraday_metrics)
        # =================================================================
        cross_day_flow_metrics = self._compute_cross_day_flow_metrics(self.ctx)
        all_metrics.update(cross_day_flow_metrics)
        self.ctx.update(cross_day_flow_metrics)
        winner_profit_margin_avg = self.ctx.get('winner_profit_margin_avg')
        total_winner_rate = self.ctx.get('total_winner_rate')
        profit_taking_flow_ratio = self.ctx.get('profit_taking_flow_ratio')
        profit_realization_quality = np.nan
        if pd.notna(winner_profit_margin_avg) and pd.notna(total_winner_rate) and pd.notna(profit_taking_flow_ratio) and profit_taking_flow_ratio > 0:
            profit_realization_quality = (winner_profit_margin_avg * (total_winner_rate / 100)) / (profit_taking_flow_ratio / 100)
        all_metrics['profit_realization_quality'] = profit_realization_quality
        self.ctx['profit_realization_quality'] = profit_realization_quality
        potential_score = self._calculate_structural_potential_score(self.ctx, all_metrics)
        all_metrics['structural_potential_score'] = potential_score
        self.ctx['structural_potential_score'] = potential_score
        game_theoretic_metrics = self._compute_game_theoretic_metrics(self.ctx)
        all_metrics.update(game_theoretic_metrics)
        self.ctx.update(game_theoretic_metrics)
        vital_signs_metrics = self._compute_vital_sign_metrics(self.ctx)
        all_metrics.update(vital_signs_metrics)
        self.ctx.update(vital_signs_metrics)
        microstructure_game_metrics = self._compute_microstructure_game_metrics(self.ctx)
        all_metrics.update(microstructure_game_metrics)
        self.ctx.update(microstructure_game_metrics)
        health_score_info = self._calculate_chip_structure_health_score(self.ctx)
        all_metrics.update(health_score_info)
        all_metrics.pop('peak_range_low', None)
        all_metrics.pop('peak_range_high', None)
        all_metrics['cost_gini_coefficient'] = today_gini
        return all_metrics

    def _prepare_intraday_data_features(self, intraday_df: pd.DataFrame, trade_date: datetime.date, debug_params: dict) -> pd.DataFrame:
        import pytz
        results = {}
        if intraday_df.empty:
            return pd.DataFrame()
        target_tz = pytz.timezone('Asia/Shanghai')
        if not isinstance(intraday_df.index, pd.DatetimeIndex):
            intraday_df.index = pd.to_datetime(intraday_df.index)
        # 此时 intraday_df.index 应该已经是 Asia/Shanghai aware，或者至少是 aware。
        # 确保它是 Asia/Shanghai aware
        if intraday_df.index.tz is None:
            # 如果意外是 naive，假定它是 UTC（因为DAO层应该输出UTC aware，但可能在某些操作后丢失时区信息）
            df.index = df.index.tz_localize('UTC', ambiguous='infer').tz_convert(target_tz)
        else:
            # 如果已经是 aware，直接转换为目标时区
            intraday_df.index = intraday_df.index.tz_convert(target_tz)
        start_time = datetime.time(9, 25)
        processed_intraday_df = intraday_df[intraday_df.index.time >= start_time].copy()
        return processed_intraday_df

    def _get_summary_metrics_from_context(self) -> dict:
        """
        【V14.0 · 稳定性前置注入版】
        - 核心新增: 提取并计算 `concentration_70pct`，为 `structural_stability_score` 提供关键前置条件。
        """
        weight_avg_cost = self.ctx.get('weight_avg')
        total_winner_rate = self.ctx.get('winner_rate')
        # 为 structural_stability_score 计算提供前置条件
        cost_15pct = self.ctx.get('cost_15pct')
        cost_85pct = self.ctx.get('cost_85pct')
        concentration_70pct = None
        if all(pd.notna(v) for v in [cost_15pct, cost_85pct, weight_avg_cost]) and weight_avg_cost > 0:
            concentration_70pct = (cost_85pct - cost_15pct) / weight_avg_cost
        return {
            'weight_avg_cost': weight_avg_cost,
            'total_winner_rate': total_winner_rate,
            'concentration_70pct': concentration_70pct, # 注入上下文
        }

    def _compute_cross_day_flow_metrics(self, context: dict) -> dict:
        results = {
            'peak_mass_transfer_rate': np.nan,
            'conviction_flow_index': np.nan,
            'constructive_turnover_ratio': np.nan,
            'structural_entropy_change': np.nan,
            'main_force_flow_gini': np.nan,
        }
        # --- 依赖项准备 ---
        intraday_df = context.get('processed_intraday_df')
        turnover_vol = context.get('daily_turnover_volume')
        total_chip_vol = context.get('total_chip_volume')
        prev_metrics = context.get('prev_metrics', {})
        prev_chip_dist = prev_metrics.get('chip_distribution')
        if intraday_df.empty or pd.isna(turnover_vol) or turnover_vol <= 0 or prev_chip_dist is None or prev_chip_dist.empty:
            return results
        turnover_rate = turnover_vol / total_chip_vol if total_chip_vol > 0 else 0
        # --- 1. 筹码峰质量转移率 (Peak Mass Transfer Rate) ---
        today_peak_cost = context.get('dominant_peak_cost')
        prev_peak_cost = prev_metrics.get('dominant_peak_cost')
        if pd.notna(today_peak_cost) and pd.notna(prev_peak_cost):
            today_peak_zone_df = self.df[(self.df['price'] >= today_peak_cost * 0.98) & (self.df['price'] <= today_peak_cost * 1.02)]
            prev_peak_zone_df = prev_chip_dist[(prev_chip_dist['price'] >= prev_peak_cost * 0.98) & (prev_chip_dist['price'] <= prev_peak_cost * 1.02)]
            mass_change = today_peak_zone_df['percent'].sum() - prev_peak_zone_df['percent'].sum()
            if turnover_rate > 0:
                results['peak_mass_transfer_rate'] = mass_change / (turnover_rate * 100)
        # --- 2. 信念流转指数 (Conviction Flow Index) ---
        daily_vwap = context.get('daily_vwap')
        if pd.notna(daily_vwap) and 'main_force_net_vol' in intraday_df.columns:
            mf_net_vol = intraday_df['main_force_net_vol']
            gathering_vol = mf_net_vol[intraday_df['minute_vwap'] < daily_vwap].clip(lower=0).sum()
            dispersal_vol = -mf_net_vol[intraday_df['minute_vwap'] > daily_vwap].clip(upper=0).sum()
            if gathering_vol > 0 and dispersal_vol > 0:
                results['conviction_flow_index'] = np.log1p(gathering_vol) / np.log1p(dispersal_vol)
            elif gathering_vol > 0:
                results['conviction_flow_index'] = 10.0 # 极度看涨
            else:
                results['conviction_flow_index'] = 0.1 # 极度看跌
        # --- 3. 建设性换手率 (Constructive Turnover Ratio) ---
        today_winner_rate = context.get('total_winner_rate')
        prev_winner_rate = prev_metrics.get('total_winner_rate')
        if pd.notna(today_winner_rate) and pd.notna(prev_winner_rate) and turnover_rate > 0:
            winner_rate_change = today_winner_rate - prev_winner_rate
            results['constructive_turnover_ratio'] = winner_rate_change / (turnover_rate * 100)
        # --- 4. 结构熵变 (Structural Entropy Change) ---
        today_entropy = context.get('price_volume_entropy')
        prev_entropy = prev_metrics.get('price_volume_entropy')
        if pd.notna(today_entropy) and pd.notna(prev_entropy):
            results['structural_entropy_change'] = today_entropy - prev_entropy
        # --- 5. 主力资金流基尼系数 (Main Force Flow Gini) ---
        def _calculate_gini_for_flow(flow_series: pd.Series) -> float:
            flow = np.abs(flow_series.dropna())
            if len(flow) < 2 or flow.sum() == 0: return np.nan
            flow = np.sort(flow)
            index = np.arange(1, len(flow) + 1)
            n = len(flow)
            return ((2 * np.sum(flow * index)) / (n * np.sum(flow))) - (n + 1) / n
        if 'main_force_net_vol' in intraday_df.columns:
            results['main_force_flow_gini'] = _calculate_gini_for_flow(intraday_df['main_force_net_vol'])
        # --- 兼容旧指标 ---
        results.update(self._compute_legacy_cross_day_metrics(context))
        return results

    def _compute_game_theoretic_metrics(self, context: dict) -> dict:
        results = {
            'strategic_phase_score': np.nan,
            'deception_index': np.nan,
            'control_solidity_index': np.nan,
            'exhaustion_risk_index': np.nan,
            'breakout_readiness_score': np.nan,
        }
        legacy_metrics = self._compute_legacy_game_theory_metrics(context)
        results.update(legacy_metrics)
        potential = context.get('structural_potential_score')
        posture = context.get('intraday_posture_score')
        entropy_change = context.get('structural_entropy_change')
        gini = context.get('cost_gini_coefficient')
        peak_transfer = context.get('peak_control_transfer')
        fatigue = context.get('chip_fatigue_index')
        loser_pain = context.get('loser_pain_index')
        impulse_quality = context.get('impulse_quality_ratio')
        close_price = context.get('close_price')
        open_price = context.get('open_price')
        atr = context.get('atr_14d')
        if any(pd.isna(v) for v in [potential, posture, entropy_change, gini, peak_transfer, fatigue, loser_pain, impulse_quality, close_price, open_price, atr]):
            return results
        results['control_solidity_index'] = gini * peak_transfer
        results['exhaustion_risk_index'] = np.log1p(fatigue) * np.log1p(loser_pain)
        price_momentum = (close_price - open_price) / atr if atr > 0 else 0
        results['deception_index'] = np.tanh(price_momentum) * (1 - np.tanh(impulse_quality / 100)) * 100
        readiness = (potential / 100) * (posture / 100) * np.clip(1 - entropy_change, 0, 2)
        results['breakout_readiness_score'] = np.clip(readiness * 100, 0, 100)
        markup_force = results['breakout_readiness_score'] * (1 + np.tanh(results['control_solidity_index'] / 100))
        distribution_force = results['exhaustion_risk_index'] * (1 + np.tanh(results['deception_index'] / 100))
        phase_score = markup_force - distribution_force
        results['strategic_phase_score'] = np.tanh(phase_score / 50) * 100
        return results

    def _compute_vital_sign_metrics(self, context: dict) -> dict:
        results = {
            'signal_conviction_score': np.nan,
            'risk_reward_profile': np.nan,
            'trend_vitality_index': np.nan,
            'overall_t1_rating': np.nan,
        }
        # --- 1. 依赖项准备 (元分析) ---
        phase_score = context.get('strategic_phase_score')
        posture_score = context.get('intraday_posture_score')
        control_solidity = context.get('control_solidity_index')
        readiness_score = context.get('breakout_readiness_score')
        exhaustion_risk = context.get('exhaustion_risk_index')
        prev_metrics = context.get('prev_metrics', {})
        prev_phase_score = prev_metrics.get('strategic_phase_score')
        if any(pd.isna(v) for v in [phase_score, posture_score, control_solidity, readiness_score, exhaustion_risk]):
            return results
        # --- 2. 信号置信度评分 (Signal Conviction Score) ---
        # 衡量战略、战术、控盘三者的一致性
        s1 = phase_score / 100
        s2 = posture_score / 100
        s3 = np.tanh(control_solidity / 100)
        # 使用几何平均数，如果任何一个环节为负，则置信度会受影响
        conviction = np.sign(s1) * (abs(s1 * s2 * s3))**(1/3)
        results['signal_conviction_score'] = conviction * 100
        # --- 3. 风险收益剖面 (Risk-Reward Profile) ---
        reward_potential = np.log1p(readiness_score)
        risk_exposure = np.log1p(exhaustion_risk)
        if risk_exposure > 0:
            results['risk_reward_profile'] = reward_potential / risk_exposure
        else:
            results['risk_reward_profile'] = reward_potential * 10 # 风险极低时，收益潜力被放大
        # --- 4. 趋势生命力指数 (Trend Vitality Index) ---
        vitality = readiness_score
        if pd.notna(prev_phase_score):
            phase_slope = phase_score - prev_phase_score
            # 如果战略评分正在改善，则放大生命力；反之则削弱
            vitality_multiplier = 1 + np.tanh(phase_slope / 20) # tanh将斜率影响约束在(0,2)
            vitality *= vitality_multiplier
        results['trend_vitality_index'] = vitality
        # --- 5. T+1综合评级 (Overall T+1 Rating) ---
        # 最终评级 = 置信度 * (1 + tanh(风险收益)) * 生命力
        # tanh将风险收益剖面压缩到(-1, 1)，使其成为一个调整因子
        rating = results['signal_conviction_score'] * \
                 (1 + np.tanh(results['risk_reward_profile'] - 1)) * \
                 (results['trend_vitality_index'] / 100)
        results['overall_t1_rating'] = np.clip(rating, -100, 100)
        return results

    def _compute_static_structure_metrics(self) -> dict:
        from scipy.signal import find_peaks
        from scipy.stats import skew
        results = {
            'structural_node_count': np.nan, 'primary_peak_kurtosis': np.nan,
            'cost_gini_coefficient': np.nan, 'structural_tension_index': np.nan,
            'structural_leverage': np.nan, 'vacuum_zone_magnitude': np.nan,
            'winner_stability_index': np.nan, 'dominant_peak_cost': np.nan,
            'dominant_peak_volume_ratio': np.nan, 'dominant_peak_profit_margin': np.nan,
            'dominant_peak_solidity': np.nan, 'secondary_peak_cost': np.nan,
            'peak_separation_ratio': np.nan, 'winner_concentration_90pct': np.nan,
            'loser_concentration_90pct': np.nan, 'chip_fault_magnitude': np.nan,
            'chip_fault_blockage_ratio': np.nan, 'total_winner_rate': np.nan,
            'total_loser_rate': np.nan, 'winner_profit_margin_avg': np.nan,
            'loser_loss_margin_avg': np.nan, 'loser_pain_index': np.nan,
            'cost_structure_skewness': np.nan, 'price_volume_entropy': np.nan,
        }
        close_price = self.ctx.get('close_price')
        atr_14d = self.ctx.get('atr_14d')
        if self.df.empty or pd.isna(close_price) or pd.isna(atr_14d) or atr_14d <= 0:
            return results
        def _calculate_weighted_kurtosis(values: pd.Series, weights: pd.Series) -> float:
            if values.empty or weights.empty or weights.sum() <= 0: return np.nan
            weighted_mean = np.average(values, weights=weights)
            weighted_variance = np.average((values - weighted_mean)**2, weights=weights)
            if weighted_variance < 1e-9: return np.nan
            m4 = np.average((values - weighted_mean)**4, weights=weights)
            kurt = m4 / (weighted_variance**2) - 3.0
            return kurt
        peaks, properties = find_peaks(self.df['percent'], prominence=0.1, width=1)
        results['structural_node_count'] = len(peaks)
        if len(peaks) > 0:
            peaks_df = pd.DataFrame({
                'peak_index': peaks, 'volume': self.df['percent'].iloc[peaks].values,
                'cost': self.df['price'].iloc[peaks].values, 'prominence': properties['prominences'],
                'left_base': properties['left_bases'], 'right_base': properties['right_bases'],
            }).sort_values(by='prominence', ascending=False).reset_index(drop=True)
            main_peak = peaks_df.iloc[0]
            main_peak_cost = main_peak['cost']
            results['dominant_peak_cost'] = main_peak_cost
            results['dominant_peak_volume_ratio'] = main_peak['volume']
            peak_region_df = self.df.iloc[int(main_peak['left_base']):int(main_peak['right_base'])+1]
            if not peak_region_df.empty and peak_region_df['percent'].sum() > 0:
                results['primary_peak_kurtosis'] = _calculate_weighted_kurtosis(peak_region_df['price'], peak_region_df['percent'])
            if len(peaks) > 1:
                secondary_peak = peaks_df.iloc[1]
                results['secondary_peak_cost'] = secondary_peak['cost']
                if main_peak_cost > 0:
                    results['peak_separation_ratio'] = abs(main_peak_cost - secondary_peak['cost']) / main_peak_cost * 100
        else:
            main_peak_idx = self.df['percent'].idxmax()
            results['dominant_peak_cost'] = self.df.loc[main_peak_idx, 'price']
            results['dominant_peak_volume_ratio'] = self.df.loc[main_peak_idx, 'percent']
        if pd.notna(results['dominant_peak_cost']) and results['dominant_peak_cost'] > 0:
            results['dominant_peak_profit_margin'] = (close_price / results['dominant_peak_cost'] - 1) * 100
        def _calculate_gini_final(prices: pd.Series, weights: pd.Series) -> float:
            if weights.sum() <= 0: return np.nan
            prices = prices.astype(float)
            weights = weights.astype(float)
            df = pd.DataFrame({'price': prices, 'weight': weights}).sort_values('price')
            df['weight_pct'] = df['weight'] / df['weight'].sum()
            df['cum_weight_pct'] = df['weight_pct'].cumsum()
            df['cost_x_weight'] = df['price'] * df['weight_pct']
            total_weighted_cost = df['cost_x_weight'].sum()
            if total_weighted_cost <= 0: return np.nan
            df['cum_cost_pct'] = df['cost_x_weight'].cumsum() / total_weighted_cost
            x = np.insert(df['cum_weight_pct'].values, 0, 0)
            y = np.insert(df['cum_cost_pct'].values, 0, 0)
            area = np.trapz(y, x)
            return 1 - 2 * area
        results['cost_gini_coefficient'] = _calculate_gini_final(self.df['price'], self.df['percent'])
        if pd.notna(results['cost_gini_coefficient']) and pd.notna(results['dominant_peak_volume_ratio']):
            results['dominant_peak_solidity'] = results['cost_gini_coefficient'] * (results['dominant_peak_volume_ratio'] / 100) * 100
        winners_df = self.df[self.df['price'] < close_price]
        losers_df = self.df[self.df['price'] > close_price]
        results['total_winner_rate'] = winners_df['percent'].sum()
        results['total_loser_rate'] = losers_df['percent'].sum()
        winner_avg_cost, loser_avg_cost = np.nan, np.nan
        if not winners_df.empty and winners_df['percent'].sum() > 0:
            winner_avg_cost = np.average(winners_df['price'], weights=winners_df['percent'])
            results['winner_profit_margin_avg'] = (close_price / winner_avg_cost - 1) * 100 if winner_avg_cost > 0 else np.nan
            gini_w = _calculate_gini_final(winners_df['price'], winners_df['percent'])
            if pd.notna(gini_w) and pd.notna(results['winner_profit_margin_avg']):
                results['winner_stability_index'] = (1 - gini_w) * results['winner_profit_margin_avg']
        if not losers_df.empty and losers_df['percent'].sum() > 0:
            loser_avg_cost = np.average(losers_df['price'], weights=losers_df['percent'])
            results['loser_loss_margin_avg'] = (close_price / loser_avg_cost - 1) * 100 if loser_avg_cost > 0 else np.nan
            gini_l = _calculate_gini_final(losers_df['price'], losers_df['percent'])
            if pd.notna(gini_l) and pd.notna(results['loser_loss_margin_avg']):
                results['loser_pain_index'] = (1 - gini_l) * abs(results['loser_loss_margin_avg'])
        if pd.notna(winner_avg_cost) and pd.notna(loser_avg_cost) and close_price > 0:
            tension = (abs(winner_avg_cost - loser_avg_cost) / close_price) * \
                      np.log1p((results['total_winner_rate'] / 100) * (results['total_loser_rate'] / 100))
            results['structural_tension_index'] = tension
        leverage = (((self.df['price'] - close_price) / close_price) * self.df['percent']).sum()
        results['structural_leverage'] = leverage
        # =================================================================
        # 修改代码块：重构真空区大小的计算逻辑，增强其在边缘情况下的健壮性
        # 使用主峰和次峰之间的距离来定义真空区，而不是寻找最低密度点
        dominant_peak_cost = results.get('dominant_peak_cost')
        secondary_peak_cost = results.get('secondary_peak_cost')
        if pd.notna(dominant_peak_cost) and pd.notna(secondary_peak_cost) and atr_14d > 0:
            vacuum_width = abs(dominant_peak_cost - secondary_peak_cost)
            results['vacuum_zone_magnitude'] = vacuum_width / atr_14d
        # =================================================================
        if pd.notna(results['dominant_peak_cost']):
            results['chip_fault_magnitude'] = (close_price - results['dominant_peak_cost']) / atr_14d
            fault_low, fault_high = sorted([results['dominant_peak_cost'], close_price])
            results['chip_fault_blockage_ratio'] = self.df[(self.df['price'] > fault_low) & (self.df['price'] < fault_high)]['percent'].sum()
        def _get_concentration(chip_df: pd.DataFrame):
            if chip_df.empty or chip_df['percent'].sum() < 1e-6: return np.nan
            chip_df = chip_df.copy()
            chip_df['percent'] = (chip_df['percent'] / chip_df['percent'].sum()) * 100
            chip_df['cum_percent'] = chip_df['percent'].cumsum()
            avg_cost = np.average(chip_df['price'], weights=chip_df['percent'])
            if avg_cost <= 0: return np.nan
            price_low = np.interp(5, chip_df['cum_percent'], chip_df['price'])
            price_high = np.interp(95, chip_df['cum_percent'], chip_df['price'])
            return (price_high - price_low) / avg_cost
        results['winner_concentration_90pct'] = _get_concentration(winners_df)
        results['loser_concentration_90pct'] = _get_concentration(losers_df)
        results['cost_structure_skewness'] = self._calculate_cost_structure_skewness(self.ctx)
        intraday_df = self.ctx.get('processed_intraday_df')
        daily_high = self.ctx.get('high_price')
        daily_low = self.ctx.get('low_price')
        total_daily_volume = self.ctx.get('daily_turnover_volume')
        results['price_volume_entropy'] = self._calculate_price_volume_entropy(intraday_df, daily_high, daily_low, total_daily_volume)
        return results

    def _calculate_structural_potential_score(self, context: dict, current_metrics: dict) -> float:
        """
        【V2.3 · 多级诊断探针版】
        - 核心新增: 植入三级诊断探针，分别监控“原始依赖输入”、“三大支柱中间件”和“最终输出”，以解剖计算失败的根源。
        """
        stock_code = context.get('stock_code', 'N/A')
        trade_date = context.get('trade_date', 'N/A')
        # =================================================================
        # 新增代码块：【一级探针】检查所有原始依赖项
        print(f"--- 诊断探针 [{stock_code}] [{trade_date}] 进入 _calculate_structural_potential_score ---")
        gini = current_metrics.get('cost_gini_coefficient')
        peak_margin = current_metrics.get('dominant_peak_profit_margin')
        peak_kurtosis = current_metrics.get('primary_peak_kurtosis')
        leverage = current_metrics.get('structural_leverage')
        winner_stability = current_metrics.get('winner_stability_index')
        loser_pain = current_metrics.get('loser_pain_index')
        tension = current_metrics.get('structural_tension_index')
        vacuum = current_metrics.get('vacuum_zone_magnitude')
        gini_slope_1d = context.get('cost_gini_coefficient_slope_1d', 0)
        print(f"    [输入] cost_gini_coefficient: {gini}")
        print(f"    [输入] dominant_peak_profit_margin: {peak_margin}")
        print(f"    [输入] primary_peak_kurtosis: {peak_kurtosis}")
        print(f"    [输入] structural_leverage: {leverage}")
        print(f"    [输入] winner_stability_index: {winner_stability}")
        print(f"    [输入] loser_pain_index: {loser_pain}")
        print(f"    [输入] structural_tension_index: {tension}")
        print(f"    [输入] vacuum_zone_magnitude: {vacuum}")
        print(f"    [输入] cost_gini_coefficient_slope_1d: {gini_slope_1d}")
        # =================================================================
        def _sigmoid(x, k=1):
            return 1 / (1 + np.exp(-k * x))
        if any(pd.isna(v) for v in [gini, peak_margin, peak_kurtosis]):
            return np.nan
        gini_score = np.clip(gini, 0, 1)
        margin_score = _sigmoid(peak_margin / 10)
        kurtosis_score = _sigmoid(peak_kurtosis / 5)
        foundation_score = 0.3 * gini_score + 0.3 * margin_score + 0.4 * kurtosis_score
        # =================================================================
        # 新增代码块：【二级探针】检查第一个支柱的计算结果
        print(f"    [支柱1] foundation_score: {foundation_score}")
        # =================================================================
        if any(pd.isna(v) for v in [leverage, winner_stability, loser_pain]):
            return np.nan
        leverage_score = 1 - _sigmoid(leverage, k=5)
        stability_score = _sigmoid(winner_stability / 20)
        pain_score = 1 - _sigmoid(loser_pain / 50)
        pressure_balance_score = np.mean([leverage_score, stability_score, pain_score])
        # =================================================================
        # 新增代码块：【二级探针】检查第二个支柱的计算结果
        print(f"    [支柱2] pressure_balance_score: {pressure_balance_score}")
        # =================================================================
        if any(pd.isna(v) for v in [tension, vacuum]):
            return np.nan
        tension_score = _sigmoid((tension - 0.1) * 20)
        vacuum_score = _sigmoid((vacuum - 1) * 2)
        dynamic_factor = 1 + np.tanh(gini_slope_1d * 50)
        upward_potential_score = (0.5 * tension_score + 0.5 * vacuum_score) * dynamic_factor
        # =================================================================
        # 新增代码块：【二级探针】检查第三个支柱的计算结果
        print(f"    [支柱3] upward_potential_score: {upward_potential_score}")
        # =================================================================
        weights = {'foundation': 0.4, 'balance': 0.25, 'potential': 0.35}
        scores = {
            'foundation': foundation_score,
            'balance': pressure_balance_score,
            'potential': upward_potential_score
        }
        final_score_raw = 1.0
        for pillar, weight in weights.items():
            score = scores.get(pillar)
            if pd.notna(score) and score > 0:
                final_score_raw *= score ** weight
        final_score = final_score_raw * 100
        # =================================================================
        # 新增代码块：【三级探针】检查最终输出
        print(f"--- 诊断探针 [{stock_code}] [{trade_date}] structural_potential_score [最终输出]: {final_score} ---")
        # =================================================================
        return final_score

    def _compute_intraday_dynamics_metrics(self, context: dict) -> dict:
        from datetime import time # 新增代码：导入 time 对象用于时间比较
        results = {
            'intraday_posture_score': np.nan,
            'peak_control_transfer': np.nan,
            'impulse_quality_ratio': np.nan,
            'opening_gap_defense_strength': np.nan,
            'active_buying_support': np.nan,
            'active_selling_pressure': np.nan,
        }
        intraday_df = context.get('processed_intraday_df')
        if intraday_df is None or intraday_df.empty:
            return results
        close_price = context.get('close_price')
        vwap = context.get('daily_vwap')
        peak_low = context.get('peak_range_low')
        peak_high = context.get('peak_range_high')
        if pd.notna(close_price) and pd.notna(vwap) and vwap > 0:
            posture = (close_price / vwap - 1) * 100
            if pd.notna(peak_low) and pd.notna(peak_high) and peak_high > peak_low:
                peak_vwap_df = intraday_df[(intraday_df['minute_vwap'] >= peak_low) & (intraday_df['minute_vwap'] <= peak_high)]
                if not peak_vwap_df.empty and peak_vwap_df['vol_shares'].sum() > 0:
                    peak_vwap = (peak_vwap_df['minute_vwap'] * peak_vwap_df['vol_shares']).sum() / peak_vwap_df['vol_shares'].sum()
                    peak_transfer = (peak_vwap / vwap - 1) * 100
                    results['peak_control_transfer'] = np.clip(peak_transfer * 10, -100, 100)
            results['intraday_posture_score'] = np.clip(posture * 10, -100, 100)
        up_moves = intraday_df[intraday_df['minute_vwap'] > intraday_df['minute_vwap'].shift(1)]
        down_moves = intraday_df[intraday_df['minute_vwap'] < intraday_df['minute_vwap'].shift(1)]
        if not up_moves.empty and not down_moves.empty:
            avg_up_vol = up_moves['vol_shares'].mean()
            avg_down_vol = down_moves['vol_shares'].mean()
            if avg_down_vol > 0:
                results['impulse_quality_ratio'] = (avg_up_vol / avg_down_vol - 1) * 100
        auction_data = intraday_df.iloc[0]
        open_price = context.get('open_price')
        pre_close = context.get('pre_close')
        if pd.notna(open_price) and pd.notna(pre_close) and pre_close > 0:
            gap_pct = (open_price / pre_close - 1) * 100
            if abs(gap_pct) > 0.1:
                # =================================================================
                # 修改代码块：修正时间过滤逻辑，使用 DatetimeIndex 进行比较
                # 原始错误逻辑: first_5_min_df = intraday_df[(intraday_df['time_marker'] > '09:30:00') & (intraday_df['time_marker'] <= '09:35:00')]
                first_5_min_df = intraday_df[(intraday_df.index.time > time(9, 30)) & (intraday_df.index.time <= time(9, 35))]
                # =================================================================
                if not first_5_min_df.empty and first_5_min_df['vol_shares'].sum() > 0:
                    vwap_5min = (first_5_min_df['minute_vwap'] * first_5_min_df['vol_shares']).sum() / first_5_min_df['vol_shares'].sum()
                    price_change_vs_open = (vwap_5min / open_price - 1) * 100 if open_price > 0 else 0
                    defense_strength = price_change_vs_open
                    results['opening_gap_defense_strength'] = np.clip(defense_strength * 50, -100, 100)
        if 'net_flow_rate' in intraday_df.columns:
            active_buy_df = intraday_df[intraday_df['net_flow_rate'] > 0]
            active_sell_df = intraday_df[intraday_df['net_flow_rate'] < 0]
            if not active_buy_df.empty:
                results['active_buying_support'] = np.average(active_buy_df['net_flow_rate'], weights=active_buy_df['vol_shares'])
            if not active_sell_df.empty:
                results['active_selling_pressure'] = abs(np.average(active_sell_df['net_flow_rate'], weights=active_sell_df['vol_shares']))
        return results

    def _calculate_chip_structure_health_score(self, context: dict) -> dict:
        """
        【V4.2 · 生产就绪版】
        - 核心优化: 移除所有调试探针，代码恢复生产状态。
        """
        results = {'chip_health_score': np.nan} # 默认值为NaN
        # 1. 定义三维模型的组件和权重(方向)
        model_dimensions = {
            'structural_soundness': {
                'components': {
                    'concentration_70pct': -1, # 越低越好
                    'cost_divergence_normalized': -1, # 越接近0或负值越好
                    'dominant_peak_profit_margin': 1, # 越高越好
                }
            },
            'momentum_purity': {
                'components': {
                    'main_force_cost_advantage': 1, # 越高越好
                    'suppressive_accumulation_intensity': 1, # 越高越好
                    'upward_impulse_purity': 1, # 越高越好
                }
            },
            'internal_pressure': {
                'components': {
                    'active_winner_profit_margin': 1, # 越高越好
                    'winner_conviction_index': 1, # 越高越好
                }
            }
        }
        # 2. 获取历史数据
        historical_df = context.get('historical_components')
        if historical_df is None or historical_df.empty:
            logger.debug(f"[{context.get('stock_code')}] [{context.get('trade_date')}] 健康分计算跳过，因缺少历史数据进行动态归一化。")
            return results
        dimension_scores = {}
        # 3. 逐个维度计算得分
        for dim_name, dim_data in model_dimensions.items():
            component_scores = []
            for metric, weight in dim_data['components'].items():
                current_value = context.get(metric)
                if current_value is None or not pd.notna(current_value) or metric not in historical_df.columns:
                    continue # 如果当前值或历史数据缺失，则安全跳过此子指标
                historical_series = historical_df[metric].dropna()
                if historical_series.empty:
                    continue
                percentile = percentileofscore(historical_series, current_value, kind='rank') / 100.0
                normalized_score = percentile if weight > 0 else (1.0 - percentile)
                component_scores.append(normalized_score)
            # 增强鲁棒性：如果维度有有效分数，则计算其平均分；否则赋予中性分
            if component_scores:
                dimension_scores[dim_name] = np.mean(component_scores)
            else:
                # 如果该维度的所有子指标都无法计算，则给予一个中性的0.5分（50分位），避免整个健康分失败
                dimension_scores[dim_name] = 0.5
                logger.debug(f"[{context.get('stock_code')}] [{context.get('trade_date')}] 健康分维度 '{dim_name}' 无法计算，赋予中性分0.5。")
        # 4. 使用几何平均整合各维度得分
        if not dimension_scores:
            return results
        final_score_raw = 1.0
        valid_dims = 0
        for score in dimension_scores.values():
            if pd.notna(score):
                final_score_raw *= score
                valid_dims += 1
        if valid_dims > 0:
            final_score_normalized = final_score_raw ** (1.0 / valid_dims)
            results['chip_health_score'] = final_score_normalized * 100
        return results

    def _calculate_active_winner_profit_margin(self, close_price: float, atr_14d: float, context: dict) -> tuple[float, float]:
        """
        【V1.0】计算活跃获利盘利润率。
        """
        active_winner_avg_cost = np.nan
        active_profit_margin = 0.0
        if pd.notna(close_price) and pd.notna(atr_14d) and atr_14d > 0:
            # 活跃获利盘：成本在 (close_price - 3*ATR, close_price) 之间，扩大范围
            active_winners_df = self.df[(self.df['price'] < close_price) & (self.df['price'] >= close_price - 3 * atr_14d)]
            if not active_winners_df.empty and active_winners_df['percent'].sum() > 0:
                active_winner_avg_cost = np.average(active_winners_df['price'], weights=active_winners_df['percent'])
                if active_winner_avg_cost > 0:
                    active_profit_margin = ((close_price - active_winner_avg_cost) / active_winner_avg_cost) * 100
            else:
                # 如果活跃获利盘区间为空，则考虑更广阔的获利盘范围，例如使用 dominant_peak_cost
                dominant_peak_cost = context.get('dominant_peak_cost')
                if pd.notna(dominant_peak_cost) and dominant_peak_cost > 0 and dominant_peak_cost < close_price:
                    active_winner_avg_cost = dominant_peak_cost
                    active_profit_margin = ((close_price - dominant_peak_cost) / dominant_peak_cost) * 100
                else:
                    # 在涨停日，如果无法计算出活跃获利盘，则赋予一个积极的默认利润率
                    # 假设涨停日至少有 9% 的利润率
                    pre_close = context.get('pre_close', close_price)
                    if (close_price / pre_close - 1) > 0.098: # 如果是涨停日 (考虑0.098作为涨停阈值)
                        active_profit_margin = 9.0 # 涨停日默认利润率
                    else:
                        active_profit_margin = 0.0 # 默认利润率为0
        return active_winner_avg_cost, active_profit_margin

    def _calculate_winner_conviction_index(self, context: dict, active_profit_margin: float) -> float:
        """
        【V1.5 · 压力缓和器版】计算赢家信念指数。
        - 核心升级: 将 `realized_pressure` 的计算函数从 `tanh` 替换为 `arctan`。
        - 核心思想: 使用 `arctan` 作为“压力缓和器”，解决因分母过小（小基数陷阱）导致的 `pressure_ratio` 异常放大问题，增强指标对极端值的鲁棒性。
        """
        bullish_reinforcement = context.get('upward_impulse_purity', 0.0)
        profit_taking_flow_ratio = context.get('profit_taking_flow_ratio', 0.0)
        active_winner_rate = context.get('active_winner_rate', 0.0)
        close_price = context.get('close_price')
        pre_close = context.get('pre_close', close_price)
        winner_conviction_index = 0.0
        if all(pd.notna(v) for v in [active_profit_margin, bullish_reinforcement, profit_taking_flow_ratio, active_winner_rate]):
            pressure_ratio = 0.0
            realized_pressure = 0.0
            if active_winner_rate > 1e-6: # 增加一个极小值判断，避免除零
                pressure_ratio = (profit_taking_flow_ratio / 100.0) / (active_winner_rate / 100.0)
                # 使用 arctan 替换 tanh，增强鲁棒性
                realized_pressure = (2 / np.pi) * np.arctan(np.clip(pressure_ratio - 1.0, 0, None))
            hesitation_factor = 1.0 + (1.0 - realized_pressure)
            margin_factor = np.log1p(np.clip(active_profit_margin / 100.0, 0, None)) if active_profit_margin > 0 else 0.0
            reinforcement_factor = np.exp(bullish_reinforcement / 100.0)
            winner_conviction_index = hesitation_factor * margin_factor * reinforcement_factor * 100
            if (close_price / pre_close - 1) > 0.098 and active_profit_margin > 0:
                winner_conviction_index = np.maximum(winner_conviction_index, 10.0)
        return winner_conviction_index

    def _calculate_cost_structure_skewness(self, context: dict) -> float:
        """
        【V1.1】计算成本结构偏度。
        - 【修正】移除对 `skewness` 的负号操作，使正偏度（筹码集中在高价区）对应正值，符合涨停日积极信号的预期。
        """
        skewness = 0.0
        if not self.df.empty and self.df['percent'].sum() >= 1e-6:
            total_percent = self.df['percent'].sum()
            weights = np.round((self.df['percent'] / total_percent) * 10000).astype(int)
            valid_weights = weights[weights > 0]
            if len(valid_weights) >= 3: # 至少需要3个点才能计算偏度
                valid_prices = self.df['price'][weights > 0]
                unweighted_sample = np.repeat(valid_prices, valid_weights)
                skewness = skew(unweighted_sample)
                # 移除负号操作，使正偏度（筹码集中在高价区）对应正值
                # skewness = -skewness
        return skewness

    def _calculate_price_volume_entropy(self, intraday_df: pd.DataFrame, daily_high: float, daily_low: float, total_daily_volume: float) -> float:
        
        if intraday_df.empty or total_daily_volume <= 0 or pd.isna(daily_high) or pd.isna(daily_low) or daily_high <= daily_low:
            return np.nan
        price_range = daily_high - daily_low
        if price_range <= 0.01:
            num_bins = 2
        else:
            num_bins = int(price_range / 0.01) + 1
            num_bins = np.clip(num_bins, 20, 200)
        prices = pd.to_numeric(intraday_df['minute_vwap'], errors='coerce')
        volumes = pd.to_numeric(intraday_df['vol_shares'], errors='coerce')
        valid_data = pd.DataFrame({'price': prices, 'volume': volumes}).dropna()
        if valid_data.empty or valid_data['volume'].sum() <= 0:
            return np.nan
        bins = pd.cut(valid_data['price'], bins=num_bins, include_lowest=True, duplicates='drop')
        volume_per_bin = valid_data.groupby(bins)['volume'].sum()
        volume_per_bin = volume_per_bin[volume_per_bin > 0]
        if volume_per_bin.empty:
            return 0.0
        probabilities = volume_per_bin / volume_per_bin.sum()
        shannon_entropy = entropy(probabilities, base=2)
        max_entropy = np.log2(len(volume_per_bin)) if len(volume_per_bin) > 1 else 0
        normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0.0
        return normalized_entropy

    def _compute_microstructure_game_metrics(self, context: dict) -> dict:
        results = {
            'peak_exchange_purity': np.nan,
            'pressure_validation_score': np.nan,
            'support_validation_score': np.nan,
            'covert_accumulation_signal': np.nan,
            # 新增代码：重新初始化被遗漏的指标
            'pressure_rejection_strength': np.nan,
            'support_validation_strength': np.nan,
            'vacuum_traversal_efficiency': np.nan,
        }
        intraday_df = context.get('processed_intraday_df')
        if intraday_df.empty:
            return results
        peak_low = context.get('peak_range_low')
        peak_high = context.get('peak_range_high')
        if 'buy_vol_raw' in intraday_df.columns:
            if all(pd.notna(v) for v in [peak_low, peak_high]):
                peak_zone_df = intraday_df[(intraday_df['minute_vwap'] >= peak_low) & (intraday_df['minute_vwap'] <= peak_high)]
                if not peak_zone_df.empty:
                    total_vol_peak = peak_zone_df['vol_shares'].sum()
                    if total_vol_peak > 0:
                        active_buy_vol = peak_zone_df['buy_vol_raw'].sum()
                        active_sell_vol = peak_zone_df['sell_vol_raw'].sum()
                        purity = 1 - abs(active_buy_vol - active_sell_vol) / total_vol_peak
                        results['peak_exchange_purity'] = purity * 100
            pre_close = context.get('pre_close')
            if pd.notna(pre_close):
                pressure_zone_df = intraday_df[intraday_df['minute_vwap'] > pre_close]
                support_zone_df = intraday_df[intraday_df['minute_vwap'] < pre_close]
                if not pressure_zone_df.empty:
                    total_vol_pressure = pressure_zone_df['vol_shares'].sum()
                    if total_vol_pressure > 0:
                        active_sell_pressure = pressure_zone_df['sell_vol_raw'].sum()
                        results['pressure_validation_score'] = (active_sell_pressure / total_vol_pressure) * 100
                if not support_zone_df.empty:
                    total_vol_support = support_zone_df['vol_shares'].sum()
                    if total_vol_support > 0:
                        active_buy_support = support_zone_df['buy_vol_raw'].sum()
                        results['support_validation_score'] = (active_buy_support / total_vol_support) * 100
        if 'main_force_net_vol' in intraday_df.columns:
            if 'close' in intraday_df.columns and 'open' in intraday_df.columns:
                dip_or_flat_df = intraday_df[intraday_df['close'] <= intraday_df['open']]
            else:
                dip_or_flat_df = intraday_df[intraday_df['minute_vwap'].diff().fillna(0) <= 0]
            if not dip_or_flat_df.empty:
                total_vol_dip = dip_or_flat_df['vol_shares'].sum()
                if total_vol_dip > 0:
                    mf_net_buy_on_dip = dip_or_flat_df['main_force_net_vol'].clip(lower=0).sum()
                    results['covert_accumulation_signal'] = (mf_net_buy_on_dip / total_vol_dip) * 100
        # =================================================================
        # 新增代码块：重新植入被遗漏的压力、支撑和穿越效率指标的计算逻辑
        daily_high = context.get('high_price')
        daily_low = context.get('low_price')
        atr = context.get('atr_14d')
        total_daily_volume = context.get('daily_turnover_volume')
        if all(pd.notna(v) for v in [daily_high, daily_low, atr]) and atr > 0 and 'main_force_net_vol' in intraday_df.columns:
            # 1. 压力区拒绝强度
            rejection_zone_start = daily_high - 0.5 * atr
            rejection_df = intraday_df[intraday_df['minute_vwap'] >= rejection_zone_start]
            if not rejection_df.empty:
                rejection_vol = rejection_df['vol_shares'].sum()
                if rejection_vol > 0:
                    mf_net_sell_in_zone = -rejection_df['main_force_net_vol'].clip(upper=0).sum()
                    results['pressure_rejection_strength'] = (mf_net_sell_in_zone / rejection_vol) * 100
            # 2. 支撑区验证强度
            support_zone_end = daily_low + 0.5 * atr
            support_df = intraday_df[intraday_df['minute_vwap'] <= support_zone_end]
            if not support_df.empty:
                support_vol = support_df['vol_shares'].sum()
                if support_vol > 0:
                    mf_net_buy_in_zone = support_df['main_force_net_vol'].clip(lower=0).sum()
                    results['support_validation_strength'] = (mf_net_buy_in_zone / support_vol) * 100
        # 3. 真空区穿越效率
        dominant_peak_cost = context.get('dominant_peak_cost')
        secondary_peak_cost = context.get('secondary_peak_cost')
        if all(pd.notna(v) for v in [dominant_peak_cost, secondary_peak_cost, atr]) and atr > 0 and total_daily_volume > 0:
            vacuum_low = min(dominant_peak_cost, secondary_peak_cost)
            vacuum_high = max(dominant_peak_cost, secondary_peak_cost)
            traversal_df = intraday_df[(intraday_df['minute_vwap'] >= vacuum_low) & (intraday_df['minute_vwap'] <= vacuum_high)]
            if not traversal_df.empty:
                traversal_volume = traversal_df['vol_shares'].sum()
                if traversal_volume > 0:
                    traversal_range = traversal_df['minute_vwap'].max() - traversal_df['minute_vwap'].min()
                    normalized_range = traversal_range / atr
                    normalized_volume = traversal_volume / total_daily_volume
                    efficiency = normalized_range / normalized_volume
                    results['vacuum_traversal_efficiency'] = np.log1p(efficiency)
        # =================================================================
        return results

    def _compute_legacy_intraday_metrics(self, context: dict) -> dict:
        """
        【V1.0 · 兼容性补丁】计算在第二象限升级后保留的旧版日内动态指标。
        """
        results = {
            'active_selling_pressure': np.nan,
            'active_buying_support': np.nan,
            'upward_impulse_purity': np.nan,
            'opening_gap_defense_strength': np.nan,
            'capitulation_absorption_index': np.nan,
        }
        intraday_df = context.get('processed_intraday_df')
        total_vol = context.get('daily_turnover_volume')
        if intraday_df.empty or pd.isna(total_vol) or total_vol <= 0 or 'main_force_net_vol' not in intraday_df.columns:
            return results
        mf_net_vol = intraday_df['main_force_net_vol']
        results['active_selling_pressure'] = -mf_net_vol.clip(upper=0).sum() / total_vol * 100
        results['active_buying_support'] = mf_net_vol.clip(lower=0).sum() / total_vol * 100
        price_change = intraday_df['minute_vwap'].diff().fillna(0)
        up_swing_df = intraday_df[price_change > 0]
        if not up_swing_df.empty:
            total_price_change_up = (up_swing_df['minute_vwap'] - up_swing_df['minute_vwap'].shift(1).fillna(up_swing_df['minute_vwap'])).sum()
            total_vol_up = up_swing_df['vol_shares'].sum()
            if total_vol_up > 0:
                results['upward_impulse_purity'] = (total_price_change_up / context.get('pre_close', 1)) / (total_vol_up / total_vol) * 100
        open_price = context.get('open_price')
        pre_close = context.get('pre_close')
        low_price = context.get('low_price')
        if pd.notna(open_price) and pd.notna(pre_close) and pd.notna(low_price):
            gap = open_price - pre_close
            if abs(gap) > 0.01: # 存在缺口
                strength = (low_price - pre_close) / gap if gap > 0 else (open_price - low_price) / abs(gap)
                results['opening_gap_defense_strength'] = np.clip(strength, -1, 1) * 100
        decline_df = intraday_df[price_change < -0.01 * context.get('pre_close', 1)] # 价格显著下跌的分钟
        if not decline_df.empty:
            capitulation_vol = decline_df['vol_shares'].sum()
            if capitulation_vol > 0:
                absorption_vol = decline_df['main_force_net_vol'].clip(lower=0).sum()
                results['capitulation_absorption_index'] = (absorption_vol / capitulation_vol) * 100
        return results

    def _compute_legacy_cross_day_metrics(self, context: dict) -> dict:
        """
        【V1.0 · 兼容性补丁】计算在第三象限升级后保留的旧版跨日迁徙指标。
        """
        results = {
            'gathering_by_support': np.nan, 'gathering_by_chasing': np.nan,
            'dispersal_by_distribution': np.nan, 'dispersal_by_capitulation': np.nan,
            'profit_taking_flow_ratio': np.nan, 'capitulation_flow_ratio': np.nan,
            'winner_loser_momentum': np.nan, 'chip_fatigue_index': np.nan,
        }
        intraday_df = context.get('processed_intraday_df')
        total_vol = context.get('daily_turnover_volume')
        total_chip_vol = context.get('total_chip_volume')
        daily_vwap = context.get('daily_vwap')
        prev_metrics = context.get('prev_metrics', {})
        if intraday_df.empty or pd.isna(total_vol) or total_vol <= 0 or 'main_force_net_vol' not in intraday_df.columns or pd.isna(daily_vwap):
            return results
        mf_net_vol = intraday_df['main_force_net_vol']
        # 1. 四象限流量计算
        support_zone = intraday_df['minute_vwap'] < daily_vwap
        chasing_zone = intraday_df['minute_vwap'] > daily_vwap
        results['gathering_by_support'] = mf_net_vol[support_zone].clip(lower=0).sum() / total_vol * 100
        results['gathering_by_chasing'] = mf_net_vol[chasing_zone].clip(lower=0).sum() / total_vol * 100
        results['dispersal_by_distribution'] = -mf_net_vol[chasing_zone].clip(upper=0).sum() / total_vol * 100
        results['dispersal_by_capitulation'] = -mf_net_vol[support_zone].clip(upper=0).sum() / total_vol * 100
        # 2. 盈亏盘流量估算
        winner_rate = context.get('total_winner_rate', 0) / 100
        loser_rate = context.get('total_loser_rate', 0) / 100
        turnover_rate = total_vol / total_chip_vol if total_chip_vol > 0 else 0
        # 假设卖方中盈亏比例与存量比例一致
        results['profit_taking_flow_ratio'] = turnover_rate * winner_rate * 100
        results['capitulation_flow_ratio'] = turnover_rate * loser_rate * 100
        # 3. 盈亏动量
        prev_winner_rate = prev_metrics.get('total_winner_rate', 0)
        prev_loser_rate = prev_metrics.get('total_loser_rate', 0)
        winner_change = context.get('total_winner_rate', 0) - prev_winner_rate
        loser_change = context.get('total_loser_rate', 0) - prev_loser_rate
        results['winner_loser_momentum'] = winner_change + loser_change # loser_change is negative
        # 4. 筹码疲劳指数
        prev_fatigue = prev_metrics.get('chip_fatigue_index', 0)
        price_range = (context.get('high_price', 0) - context.get('low_price', 0)) / context.get('pre_close', 1)
        fatigue_increment = turnover_rate * (1 + price_range) * 100
        results['chip_fatigue_index'] = prev_fatigue * 0.9 + fatigue_increment # 每日衰减
        return results

    def _compute_legacy_game_theory_metrics(self, context: dict) -> dict:
        """
        【V1.2 · 生产就绪版】计算在第四象限升级后保留的旧版博弈意图指标。
        - 核心维护: 移除调试探针。
        """
        results = {
            'main_force_cost_advantage': np.nan,
            'auction_intent_signal': np.nan,
            'auction_closing_position': np.nan,
        }
        daily_vwap = context.get('daily_vwap')
        weight_avg_cost = context.get('weight_avg_cost')
        if pd.notna(daily_vwap) and pd.notna(weight_avg_cost) and weight_avg_cost > 0 and daily_vwap > 0:
            results['main_force_cost_advantage'] = (weight_avg_cost / daily_vwap - 1) * 100
        intraday_df = context.get('processed_intraday_df')
        if not intraday_df.empty:
            auction_data = intraday_df.iloc[0]
            open_price = context.get('open_price')
            pre_close = context.get('pre_close')
            high_price = context.get('high_price')
            low_price = context.get('low_price')
            if all(pd.notna(v) for v in [open_price, pre_close, high_price, low_price]) and high_price > low_price:
                gap_pct = (open_price / pre_close - 1) * 100
                daily_turnover_volume = context.get('daily_turnover_volume', 1)
                if daily_turnover_volume <= 0: daily_turnover_volume = 1
                auction_vol_ratio = auction_data['vol_shares'] / daily_turnover_volume
                results['auction_intent_signal'] = gap_pct * np.log1p(auction_vol_ratio * 100)
                results['auction_closing_position'] = ((open_price - low_price) / (high_price - low_price) * 2 - 1) * 100
        return results




