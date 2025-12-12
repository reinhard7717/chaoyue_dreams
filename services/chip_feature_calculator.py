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
        """
        【V12.5 · 情境融合版】
        - 核心升维: 新增对 `_compute_contextual_action_metrics` 的调用，正式将“情境行为融合”指标的计算职责
                     纳入筹码服务，解决了跨服务的数据依赖问题。
        """
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
        microstructure_dynamics_metrics = self._compute_microstructure_dynamics(self.ctx)
        all_metrics.update(microstructure_dynamics_metrics)
        self.ctx.update(microstructure_dynamics_metrics)
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
        legacy_intraday_metrics = self._compute_legacy_intraday_metrics(self.ctx)
        all_metrics.update(legacy_intraday_metrics)
        self.ctx.update(legacy_intraday_metrics)
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
        tactical_intent_metrics = self._compute_tactical_intent_metrics(self.ctx)
        all_metrics.update(tactical_intent_metrics)
        self.ctx.update(tactical_intent_metrics)
        realtime_orderbook_metrics = self._compute_realtime_orderbook_metrics(self.ctx)
        all_metrics.update(realtime_orderbook_metrics)
        self.ctx.update(realtime_orderbook_metrics)
        # [新增的代码块] 调用情境行为指标计算
        contextual_action_metrics = self._compute_contextual_action_metrics(self.ctx)
        all_metrics.update(contextual_action_metrics)
        self.ctx.update(contextual_action_metrics)
        health_score_info = self._calculate_chip_structure_health_score(self.ctx)
        all_metrics.update(health_score_info)
        all_metrics.pop('peak_range_low', None)
        all_metrics.pop('peak_range_high', None)
        all_metrics['cost_gini_coefficient'] = self.ctx.get('cost_gini_coefficient')
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
            intraday_df.index = intraday_df.index.tz_localize('UTC', ambiguous='infer').tz_convert(target_tz) # 将错误的变量名 df 修正为 intraday_df
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
        """
        【V1.2 · 净胜结果重构版】
        - 核心重构(Conviction Flow): 废弃原有的比率模型，引入“净信念流”模型。新公式 (G-D)/(G+D)
          直接量化了买卖双方交锋后的“净胜结果”，解决了因分母过小导致指数失真的“比率谬误”。
        - 核心重构(Constructive Turnover): 在原有“换手效率”基础上，乘以 (1 - 获利盘比例) 的“建设潜力因子”。
          此举旨在区分底部区域的“成本抬升”与顶部区域的“获利盘内卷”，解决了“Churn悖论”。
        """
        results = {
            'peak_mass_transfer_rate': np.nan,
            'conviction_flow_index': np.nan,
            'conviction_flow_buy_intensity': np.nan, # 新增行
            'conviction_flow_sell_intensity': np.nan, # 新增行
            'constructive_turnover_ratio': np.nan,
            'structural_entropy_change': np.nan,
            'main_force_flow_gini': np.nan,
        }
        intraday_df = context.get('processed_intraday_df')
        turnover_vol = context.get('daily_turnover_volume')
        total_chip_vol = context.get('total_chip_volume')
        prev_metrics = context.get('prev_metrics', {})
        prev_chip_dist = prev_metrics.get('chip_distribution')
        if intraday_df.empty or pd.isna(turnover_vol) or turnover_vol <= 0 or prev_chip_dist is None or prev_chip_dist.empty:
            return results
        turnover_rate = turnover_vol / total_chip_vol if total_chip_vol > 0 else 0
        today_peak_cost = context.get('dominant_peak_cost')
        prev_peak_cost = prev_metrics.get('dominant_peak_cost')
        if pd.notna(today_peak_cost) and pd.notna(prev_peak_cost):
            today_peak_zone_df = self.df[(self.df['price'] >= today_peak_cost * 0.98) & (self.df['price'] <= today_peak_cost * 1.02)]
            prev_peak_zone_df = prev_chip_dist[(prev_chip_dist['price'] >= prev_peak_cost * 0.98) & (prev_chip_dist['price'] <= prev_peak_cost * 1.02)]
            mass_change = today_peak_zone_df['percent'].sum() - prev_peak_zone_df['percent'].sum()
            if turnover_rate > 0:
                results['peak_mass_transfer_rate'] = mass_change / (turnover_rate * 100)
        daily_vwap = context.get('daily_vwap')
        buy_sweep_intensity = context.get('buy_sweep_intensity', 0)
        # 重构 conviction_flow_index
        if pd.notna(daily_vwap) and 'main_force_net_vol' in intraday_df.columns:
            mf_net_vol = intraday_df['main_force_net_vol']
            gathering_vol = mf_net_vol[intraday_df['minute_vwap'] < daily_vwap].clip(lower=0).sum()
            chasing_vol_raw = mf_net_vol[intraday_df['minute_vwap'] > daily_vwap].clip(lower=0).sum()
            gathering_total_weighted = gathering_vol + (chasing_vol_raw * (1 + buy_sweep_intensity))
            dispersal_vol = -mf_net_vol[intraday_df['minute_vwap'] > daily_vwap].clip(upper=0).sum()
            total_battle_vol = gathering_total_weighted + dispersal_vol
            if total_battle_vol > 0:
                # 新逻辑：(买方力量 - 卖方力量) / 总力量，量化“净胜结果”
                net_conviction_flow = (gathering_total_weighted - dispersal_vol) / total_battle_vol
                results['conviction_flow_index'] = net_conviction_flow * 100
                # 新增行：计算买入和卖出信念流强度
                results['conviction_flow_buy_intensity'] = (gathering_total_weighted / total_battle_vol) * 100
                results['conviction_flow_sell_intensity'] = (dispersal_vol / total_battle_vol) * 100
            else:
                results['conviction_flow_index'] = 0.0
                # 新增行：当总战斗量为0时，强度也为0
                results['conviction_flow_buy_intensity'] = 0.0
                results['conviction_flow_sell_intensity'] = 0.0
        today_winner_rate = context.get('total_winner_rate')
        prev_winner_rate = prev_metrics.get('total_winner_rate')
        order_flow_imbalance = context.get('order_flow_imbalance', 0)
        # 重构 constructive_turnover_ratio
        if pd.notna(today_winner_rate) and pd.notna(prev_winner_rate) and turnover_rate > 0:
            winner_rate_change = today_winner_rate - prev_winner_rate
            raw_efficiency = winner_rate_change / (turnover_rate * 100)
            # 引入“建设潜力因子”，即前一天的套牢盘比例
            construction_potential = (100 - prev_winner_rate) / 100
            # 新逻辑：潜力加权的换手效率
            results['constructive_turnover_ratio'] = raw_efficiency * construction_potential
        today_entropy = context.get('price_volume_entropy')
        prev_entropy = prev_metrics.get('price_volume_entropy')
        if pd.notna(today_entropy) and pd.notna(prev_entropy):
            results['structural_entropy_change'] = today_entropy - prev_entropy
        def _calculate_gini_for_flow(flow_series: pd.Series) -> float:
            flow = np.abs(flow_series.dropna())
            if len(flow) < 2 or flow.sum() == 0: return np.nan
            flow = np.sort(flow)
            index = np.arange(1, len(flow) + 1)
            n = len(flow)
            return ((2 * np.sum(flow * index)) / (n * np.sum(flow))) - (n + 1) / n
        if 'main_force_net_vol' in intraday_df.columns:
            results['main_force_flow_gini'] = _calculate_gini_for_flow(intraday_df['main_force_net_vol'])
        results.update(self._compute_legacy_cross_day_metrics(context))
        return results

    def _compute_game_theoretic_metrics(self, context: dict) -> dict:
        """
        【V1.3 · 神经探针植入版】
        - 核心新增: 植入由 `enable_gt_probe` 控制的“神经探针”，用于解剖博弈论指标的依赖链。
                     探针将详细打印所有前置依赖项的值，并明确报告是否因依赖项缺失而“短路”计算，
                     从而精准定位导致指标为空的根本原因。
        - 核心增强: 为 `peak_control_transfer` 等易失败的依赖项增加了安全的默认值，提升代码鲁棒性。
        """
        results = {
            'strategic_phase_score': np.nan,
            'deception_index': np.nan,
            'deception_lure_long_intensity': np.nan, # 新增行
            'deception_lure_short_intensity': np.nan, # 新增行
            'control_solidity_index': np.nan,
            'exhaustion_risk_index': np.nan,
            'breakout_readiness_score': np.nan,
        }
        # 植入探针 GT
        debug_params = context.get('debug_params', {})
        enable_probe = debug_params.get('enable_gt_probe', False)
        target_date_str = debug_params.get('target_date')
        is_target_date = str(context.get('trade_date')) == target_date_str
        stock_code = context.get('stock_code')
        if enable_probe and is_target_date:
            print(f"\n[探针 GT.1 - {stock_code} @ {context.get('trade_date')}] 进入计算器 _compute_game_theoretic_metrics...")
        legacy_metrics = self._compute_legacy_game_theory_metrics(context)
        results.update(legacy_metrics)
        potential = context.get('structural_potential_score')
        posture = context.get('intraday_posture_score')
        entropy_change = context.get('structural_entropy_change')
        gini = context.get('cost_gini_coefficient')
        # 增加默认值，增强鲁棒性
        peak_transfer = context.get('peak_control_transfer', 0.0)
        fatigue = context.get('chip_fatigue_index')
        loser_pain = context.get('loser_pain_index')
        close_price = context.get('close_price')
        open_price = context.get('open_price')
        high_price = context.get('high_price')
        low_5d = context.get('low_5d')
        atr = context.get('atr_14d')
        rally_distribution_pressure = context.get('rally_distribution_pressure', 0.0)
        suppressive_accumulation = context.get('suppressive_accumulation_intensity', 0.0)
        required_vars_map = {
            'potential': potential, 'posture': posture, 'entropy_change': entropy_change,
            'gini': gini, 'peak_transfer': peak_transfer, 'fatigue': fatigue,
            'loser_pain': loser_pain, 'close_price': close_price, 'open_price': open_price,
            'high_price': high_price, 'low_5d': low_5d, 'atr': atr,
            'rally_distribution_pressure': rally_distribution_pressure,
            'suppressive_accumulation': suppressive_accumulation
        }
        required_vars = list(required_vars_map.values())
        is_short_circuit = any(pd.isna(v) for v in required_vars) or (pd.notna(atr) and atr <= 0)
        if enable_probe and is_target_date:
            print(f"[探针 GT.2 - {stock_code} @ {context.get('trade_date')}] 检查所有前置依赖项...")
            for name, value in required_vars_map.items():
                print(f"  - {name}: {value}")
            print(f"  - 决策: 是否短路计算？ -> {is_short_circuit}")
        if is_short_circuit:
            return results
        price_extension = (high_price - low_5d) / atr
        acceleration_factor = np.log1p(np.maximum(0, price_extension))
        fatigue_factor = np.log1p(fatigue)
        results['exhaustion_risk_index'] = fatigue_factor * acceleration_factor
        price_momentum = (close_price - open_price) / atr
        price_momentum_factor = np.tanh(price_momentum)
        lure_long_score = (price_momentum_factor if price_momentum > 0 else 0) * np.tanh(rally_distribution_pressure / 50)
        lure_short_score = (abs(price_momentum_factor) if price_momentum < 0 else 0) * np.tanh(suppressive_accumulation / 50)
        results['deception_index'] = (lure_long_score - lure_short_score) * 100
        # 新增行：计算诱多和诱空强度
        results['deception_lure_long_intensity'] = np.clip(lure_long_score * 100, 0, 100)
        results['deception_lure_short_intensity'] = np.clip(lure_short_score * 100, 0, 100)
        results['control_solidity_index'] = gini * peak_transfer
        posture_unipolar = (posture + 100) / 200
        readiness = (potential / 100) * posture_unipolar * np.clip(1 - entropy_change, 0, 2)
        results['breakout_readiness_score'] = np.clip(readiness * 100, 0, 100)
        markup_force = results['breakout_readiness_score'] * (1 + np.tanh(results['control_solidity_index'] / 100))
        distribution_force = results['exhaustion_risk_index'] * (1 + np.tanh(results['deception_index'] / 100 if results['deception_index'] > 0 else 0))
        phase_score = markup_force - distribution_force
        results['strategic_phase_score'] = np.tanh(phase_score / 50) * 100
        if enable_probe and is_target_date:
            print(f"[探针 GT.3 - {stock_code} @ {context.get('trade_date')}] 计算完成。")
            print(f"  - control_solidity_index: {results['control_solidity_index']:.2f}")
            print(f"  - deception_index: {results['deception_index']:.2f}")
            print(f"  - breakout_readiness_score: {results['breakout_readiness_score']:.2f}")
        return results

    def _compute_vital_sign_metrics(self, context: dict) -> dict:
        """
        【V1.1 · 信念悖论修复版】
        - 核心修复: 彻底废除了 `signal_conviction_score` 的几何平均数算法。该算法因“负负得正”
                     的数学特性，会错误地将多个负面信号解读为中性或正面信号，引发“信念悖论”。
        - 核心升级: 引入“加权内阁”模型（加权算术平均），根据战略(0.5)、战术(0.3)、控盘(0.2)
                     的权重来融合三大支柱，确保多维度共振的弱势能被正确地、叠加地惩罚，
                     使最终的“信号置信度”评分更符合实战逻辑。
        """
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
        # 废除几何平均，引入“加权内阁”模型
        # --- 2. 信号置信度评分 (Signal Conviction Score) ---
        # 衡量战略、战术、控盘三者的一致性
        s1 = phase_score / 100  # 战略
        s2 = posture_score / 100  # 战术
        s3 = np.tanh(control_solidity / 100)  # 控盘
        # 使用加权算术平均，正确处理负信号的叠加效应
        conviction = (0.5 * s1 + 0.3 * s2 + 0.2 * s3)
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
        """
        【V11.2 · 心理学重构版】
        - 核心重构(Winner Stability): 废弃线性利润模型，引入高斯函数来模拟“利润甜蜜区”。
          新模型认为，过高（>50%）或过低的利润都会降低持股稳定性，只有在“甜蜜区”（如20%）附近，
          获利盘的结构才最为稳固，更符合A股“兑现冲动”的博弈心理。
        - 核心重构(Loser Pain): 废弃线性亏损模型，引入“对数增长+指数衰减”模型来模拟“峰值痛苦区”。
          新模型认为，随着亏损加深，投资者的痛苦感（潜在抛压）会先快速上升，但在亏损巨大时
          （如 > 50%）会因“心理麻木”或“彻底投降”而逐渐衰减，更符合A股散户的行为模式。
        """
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
        results['vacuum_zone_magnitude'] = 0.0
        if len(peaks) > 0:
            peaks_df = pd.DataFrame({
                'peak_index': peaks, 'volume': self.df['percent'].iloc[peaks].values,
                'cost': self.df['price'].iloc[peaks].values, 'prominence': properties['prominences'],
                'left_base': properties['left_bases'], 'right_base': properties['right_bases'],
            })
            peaks_by_prominence = peaks_df.sort_values(by='prominence', ascending=False).reset_index(drop=True)
            main_peak = peaks_by_prominence.iloc[0]
            main_peak_cost = main_peak['cost']
            results['dominant_peak_cost'] = main_peak_cost
            results['dominant_peak_volume_ratio'] = main_peak['volume']
            peak_region_df = self.df.iloc[int(main_peak['left_base']):int(main_peak['right_base'])+1]
            if not peak_region_df.empty and peak_region_df['percent'].sum() > 0:
                results['primary_peak_kurtosis'] = _calculate_weighted_kurtosis(peak_region_df['price'], peak_region_df['percent'])
            if len(peaks) > 1:
                secondary_peak = peaks_by_prominence.iloc[1]
                results['secondary_peak_cost'] = secondary_peak['cost']
                if main_peak_cost > 0:
                    results['peak_separation_ratio'] = abs(main_peak_cost - secondary_peak['cost']) / main_peak_cost * 100
                peaks_by_cost = peaks_df.sort_values(by='cost').reset_index(drop=True)
                if len(peaks_by_cost) > 1:
                    peak_separations_atr = (peaks_by_cost['cost'].diff() / atr_14d).dropna()
                    max_separation_atr = peak_separations_atr.max()
                    peak_separation_threshold = 2.5
                    if max_separation_atr > peak_separation_threshold:
                        results['vacuum_zone_magnitude'] = max_separation_atr
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
        results['total_winner_rate'] = self.ctx.get('total_winner_rate')
        results['total_loser_rate'] = 100.0 - results['total_winner_rate'] if pd.notna(results['total_winner_rate']) else np.nan
        winner_avg_cost, loser_avg_cost = np.nan, np.nan
        if not winners_df.empty and winners_df['percent'].sum() > 0:
            winner_avg_cost = np.average(winners_df['price'], weights=winners_df['percent'])
            results['winner_profit_margin_avg'] = (close_price / winner_avg_cost - 1) * 100 if winner_avg_cost > 0 else np.nan
            gini_w = _calculate_gini_final(winners_df['price'], winners_df['percent'])
            # 重构 winner_stability_index
            if pd.notna(gini_w) and pd.notna(results['winner_profit_margin_avg']):
                profit_margin = results['winner_profit_margin_avg']
                # 引入高斯函数模拟“利润甜蜜区”
                mu = 20.0  # 最佳利润点 (20%)
                sigma = 15.0 # 利润敏感度标准差 (15%)
                gaussian_factor = np.exp(-((profit_margin - mu)**2) / (2 * sigma**2))
                # 获利盘成本集中度 * 利润甜蜜区因子
                results['winner_stability_index'] = (1 - gini_w) * gaussian_factor * 100
        if not losers_df.empty and losers_df['percent'].sum() > 0:
            loser_avg_cost = np.average(losers_df['price'], weights=losers_df['percent'])
            results['loser_loss_margin_avg'] = (close_price / loser_avg_cost - 1) * 100 if loser_avg_cost > 0 else np.nan
            gini_l = _calculate_gini_final(losers_df['price'], losers_df['percent'])
            # 重构 loser_pain_index
            if pd.notna(gini_l) and pd.notna(results['loser_loss_margin_avg']):
                loss_margin = abs(results['loser_loss_margin_avg'])
                # 引入“对数增长 + 指数衰减”模型模拟“峰值痛苦区”和“投降效应”
                # 对数部分模拟痛苦的非线性增长
                log_pain = np.log1p(loss_margin)
                # 指数衰减部分模拟亏损过大后的“心理麻木”
                capitulation_decay = np.exp(-0.02 * loss_margin)
                # 套牢盘成本集中度 * 峰值痛苦因子
                results['loser_pain_index'] = (1 - gini_l) * log_pain * capitulation_decay * 100
        if pd.notna(winner_avg_cost) and pd.notna(loser_avg_cost) and close_price > 0:
            tension = (abs(winner_avg_cost - loser_avg_cost) / close_price) * \
                      np.log1p((results['total_winner_rate'] / 100) * (results['total_loser_rate'] / 100))
            results['structural_tension_index'] = tension
        leverage = (((self.df['price'] - close_price) / close_price) * self.df['percent']).sum()
        results['structural_leverage'] = leverage
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
        gini = current_metrics.get('cost_gini_coefficient')
        peak_margin = current_metrics.get('dominant_peak_profit_margin')
        peak_kurtosis = current_metrics.get('primary_peak_kurtosis')
        leverage = current_metrics.get('structural_leverage')
        winner_stability = current_metrics.get('winner_stability_index')
        loser_pain = current_metrics.get('loser_pain_index')
        tension = current_metrics.get('structural_tension_index')
        vacuum = current_metrics.get('vacuum_zone_magnitude')
        gini_slope_1d = context.get('cost_gini_coefficient_slope_1d', 0)
        def _sigmoid(x, k=1):
            return 1 / (1 + np.exp(-k * x))
        if any(pd.isna(v) for v in [gini, peak_margin, peak_kurtosis]):
            return np.nan
        gini_score = np.clip(gini, 0, 1)
        margin_score = _sigmoid(peak_margin / 10)
        kurtosis_score = _sigmoid(peak_kurtosis / 5)
        foundation_score = 0.3 * gini_score + 0.3 * margin_score + 0.4 * kurtosis_score
        if any(pd.isna(v) for v in [leverage, winner_stability, loser_pain]):
            return np.nan
        leverage_score = 1 - _sigmoid(leverage, k=5)
        stability_score = _sigmoid(winner_stability / 20)
        pain_score = 1 - _sigmoid(loser_pain / 50)
        pressure_balance_score = np.mean([leverage_score, stability_score, pain_score])
        if any(pd.isna(v) for v in [tension, vacuum]):
            return np.nan
        tension_score = _sigmoid((tension - 0.1) * 20)
        vacuum_score = _sigmoid((vacuum - 1) * 2)
        dynamic_factor = 1 + np.tanh(gini_slope_1d * 50)
        upward_potential_score = (0.5 * tension_score + 0.5 * vacuum_score) * dynamic_factor
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
        return final_score

    def _compute_intraday_dynamics_metrics(self, context: dict) -> dict:
        """
        【V2.2 · 探针植入版】
        - 核心新增: 植入由 `debug_params` 控制的诊断探针，用于追踪 `opening_gap_defense_strength` 的计算链路。
        """
        from datetime import time
        results = {
            'intraday_posture_score': np.nan,
            'peak_control_transfer': np.nan,
            'impulse_quality_ratio': np.nan,
            'upward_impulse_strength': np.nan, # 新增行
            'downward_impulse_strength': np.nan, # 新增行
            'opening_gap_defense_strength': np.nan,
            'active_buying_support': np.nan,
            'active_selling_pressure': np.nan,
        }
        # 植入探针 OGDS
        debug_params = context.get('debug_params', {})
        enable_probe = debug_params.get('enable_ogds_probe', False)
        target_date_str = debug_params.get('target_date')
        is_target_date = str(context.get('trade_date')) == target_date_str
        stock_code = context.get('stock_code')
        if enable_probe and is_target_date:
            print(f"\n[探针 OGDS.1 - {stock_code} @ {context.get('trade_date')}] 进入计算器 _compute_intraday_dynamics_metrics...")
        intraday_df_raw = context.get('processed_intraday_df')
        if intraday_df_raw is None or intraday_df_raw.empty:
            return results
        intraday_df = intraday_df_raw[intraday_df_raw.index.time >= time(9, 30)].copy()
        if intraday_df.empty:
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
        # 新增行：计算上涨脉冲强度和下跌脉冲强度
        total_intraday_vol = intraday_df['vol_shares'].sum()
        if total_intraday_vol > 0:
            results['upward_impulse_strength'] = (up_moves['vol_shares'].sum() / total_intraday_vol) * 100
            results['downward_impulse_strength'] = (down_moves['vol_shares'].sum() / total_intraday_vol) * 100
        open_price = context.get('open_price')
        pre_close = context.get('pre_close')
        if enable_probe and is_target_date:
            print(f"[探针 OGDS.2 - {stock_code} @ {context.get('trade_date')}] 检查缺口计算的原始输入...")
            print(f"  - 当日开盘价 (open_price): {open_price}")
            print(f"  - 前日收盘价 (pre_close): {pre_close}")
        if pd.notna(open_price) and pd.notna(pre_close) and pre_close > 0:
            gap_pct = (open_price / pre_close - 1) * 100
            if enable_probe and is_target_date:
                print(f"  - 计算出的缺口百分比 (gap_pct): {gap_pct:.2f}%")
            if abs(gap_pct) > 0.1:
                if enable_probe and is_target_date:
                    print(f"  - 决策: 缺口 > 0.1%，进入防御强度计算逻辑。")
                first_5_min_df = intraday_df[intraday_df.index.time <= time(9, 35)]
                if enable_probe and is_target_date:
                    print(f"  - 数据切片: 截取开盘后5分钟数据 (9:30-9:35)，共 {len(first_5_min_df)} 条记录。")
                if not first_5_min_df.empty and first_5_min_df['vol_shares'].sum() > 0:
                    sum_amount = (first_5_min_df['minute_vwap'] * first_5_min_df['vol_shares']).sum()
                    sum_vol = first_5_min_df['vol_shares'].sum()
                    vwap_5min = sum_amount / sum_vol
                    price_change_vs_open = (vwap_5min / open_price - 1) * 100 if open_price > 0 else 0
                    defense_strength = price_change_vs_open
                    results['opening_gap_defense_strength'] = np.clip(defense_strength * 50, -100, 100)
                    if enable_probe and is_target_date:
                        print(f"  - VWAP计算: 5分钟内总金额={sum_amount:.2f}, 总成交量={sum_vol:.0f}, 5分钟VWAP={vwap_5min:.3f}")
                        print(f"  - 强度计算: (5分钟VWAP / 开盘价 - 1) = {price_change_vs_open:.2f}%")
                        print(f"  - 最终结果: 裁剪后的防御强度评分为 {results['opening_gap_defense_strength']:.2f}")
                elif enable_probe and is_target_date:
                    print(f"  - 警告: 开盘5分钟内数据为空或无成交量，无法计算防御强度。")
            else:
                results['opening_gap_defense_strength'] = 0.0
                if enable_probe and is_target_date:
                    print(f"  - 决策: 缺口 <= 0.1%，无需计算，防御强度返回默认值 0.0。")
        elif enable_probe and is_target_date:
            print(f"  - 警告: open_price 或 pre_close 缺失，无法计算缺口。")
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
        【V5.0 · 绝对真理版】
        - 核心修复: 彻底解决了“健康分悖论”。针对 `main_force_cost_advantage` 等具有绝对意义的
                     指标，废弃了原有的、只能反映相对历史排名的 `percentileofscore` 算法。
        - 核心升级: 引入了基于 `tanh` 函数的“绝对真理”归一化方法 `_get_absolute_normalized_score`。
                     该方法能将指标的绝对值（如成本优势为负则必然是坏事）正确地映射为分数，
                     根除了因“相对主义”评估标准而产生的逻辑谬误。
        """
        results = {'chip_health_score': np.nan} # 默认值为NaN
        # 定义需要进行“绝对评估”的指标及其参数
        ABSOLUTE_METRICS_CONFIG = {
            'main_force_cost_advantage': {'neutral': 0.0, 'sensitivity': 0.1, 'asc': True},
            'dominant_peak_profit_margin': {'neutral': 0.0, 'sensitivity': 0.05, 'asc': True},
            'active_winner_profit_margin': {'neutral': 5.0, 'sensitivity': 0.1, 'asc': True},
            'winner_conviction_index': {'neutral': 10.0, 'sensitivity': 0.02, 'asc': True},
        }
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
        historical_df = context.get('historical_components')
        if historical_df is None or historical_df.empty:
            logger.debug(f"[{context.get('stock_code')}] [{context.get('trade_date')}] 健康分计算跳过，因缺少历史数据进行动态归一化。")
            return results
        dimension_scores = {}
        for dim_name, dim_data in model_dimensions.items():
            component_scores = []
            for metric, weight in dim_data['components'].items():
                current_value = context.get(metric)
                if current_value is None or not pd.notna(current_value):
                    continue
                # 引入绝对真理评估分支
                if metric in ABSOLUTE_METRICS_CONFIG:
                    config = ABSOLUTE_METRICS_CONFIG[metric]
                    normalized_score = self._get_absolute_normalized_score(
                        value=current_value,
                        neutral_point=config['neutral'],
                        sensitivity=config['sensitivity'],
                        ascending=config['asc']
                    )
                else: # 保留对其他指标的相对评估
                    if metric not in historical_df.columns:
                        continue
                    historical_series = historical_df[metric].dropna()
                    if historical_series.empty:
                        continue
                    percentile = percentileofscore(historical_series, current_value, kind='rank') / 100.0
                    normalized_score = percentile if weight > 0 else (1.0 - percentile)
                component_scores.append(normalized_score)
            if component_scores:
                dimension_scores[dim_name] = np.mean(component_scores)
            else:
                dimension_scores[dim_name] = 0.5
                logger.debug(f"[{context.get('stock_code')}] [{context.get('trade_date')}] 健康分维度 '{dim_name}' 无法计算，赋予中性分0.5。")
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
        【V2.0 · 认知重塑版】计算成本结构偏度，修正“镜像悖论”。
        - 核心修正: 彻底纠正了对“偏度”概念的根本性误解。旧版本的注释错误地将“正偏度”
                     解读为“筹码集中在高价区”，导致了对支撑和压力的颠倒认知。
        - 核心澄清 (第一性原理):
          - 正偏度 (Positive Skew): 筹码分布主体集中在【低价区】，形成强力支撑，是积极信号。
          - 负偏度 (Negative Skew): 筹码分布主体集中在【高价区】，形成沉重套牢盘（压力），是消极信号。
        - 最终实现: 本方法直接返回由 `scipy.stats.skew` 计算出的原始偏度值，该值完美符合上述
                     正确的逻辑，无需任何符号反转。
        """
        skewness = 0.0
        if not self.df.empty and self.df['percent'].sum() >= 1e-6:
            total_percent = self.df['percent'].sum()
            weights = np.round((self.df['percent'] / total_percent) * 10000).astype(int)
            valid_weights = weights[weights > 0]
            if len(valid_weights) >= 3: # 至少需要3个点才能计算偏度
                valid_prices = self.df['price'][weights > 0]
                unweighted_sample = np.repeat(valid_prices, valid_weights)
                # 直接使用scipy计算的偏度值，其符号与我们的战术意图完美对齐
                skewness = skew(unweighted_sample)
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
        """
        【V1.2 · 隐蔽吸筹逻辑修正版】
        - 核心修正: 收紧 `covert_accumulation_signal` 的触发条件，从 `price_momentum < 0.1` 修正为 `price_momentum <= 0`，
                     确保只在价格下跌或横盘时才识别吸筹信号，使其更符合“隐蔽”的博弈内涵。
        - 核心升级: 引入订单流失衡(OFI)来增强 `covert_accumulation_signal` 的计算精度。
        """
        results = {
            'peak_exchange_purity': np.nan,
            'pressure_validation_score': np.nan,
            'support_validation_score': np.nan,
            'covert_accumulation_signal': np.nan,
            'covert_distribution_signal': np.nan, # 新增行
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
        order_flow_imbalance = context.get('order_flow_imbalance', 0)
        price_momentum = (context.get('close_price', 0) - context.get('open_price', 0)) / context.get('atr_14d', 1)
        # 收紧“隐蔽”的定义，要求价格动能为负或零
        if price_momentum <= 0:
            results['covert_accumulation_signal'] = np.clip(order_flow_imbalance * 100, 0, 100)
        else:
            results['covert_accumulation_signal'] = 0
        # 新增行：计算隐蔽派发信号
        if price_momentum >= 0: # 价格上涨或横盘时，如果OFI为负，则可能是隐蔽派发
            results['covert_distribution_signal'] = np.clip(-order_flow_imbalance * 100, 0, 100)
        else:
            results['covert_distribution_signal'] = 0
        daily_high = context.get('high_price')
        daily_low = context.get('low_price')
        atr = context.get('atr_14d')
        total_daily_volume = context.get('daily_turnover_volume')
        if all(pd.notna(v) for v in [daily_high, daily_low, atr]) and atr > 0 and 'main_force_net_vol' in intraday_df.columns:
            rejection_zone_start = daily_high - 0.5 * atr
            rejection_df = intraday_df[intraday_df['minute_vwap'] >= rejection_zone_start]
            if not rejection_df.empty:
                rejection_vol = rejection_df['vol_shares'].sum()
                if rejection_vol > 0:
                    mf_net_sell_in_zone = -rejection_df['main_force_net_vol'].clip(upper=0).sum()
                    results['pressure_rejection_strength'] = (mf_net_sell_in_zone / rejection_vol) * 100
            support_zone_end = daily_low + 0.5 * atr
            support_df = intraday_df[intraday_df['minute_vwap'] <= support_zone_end]
            if not support_df.empty:
                support_vol = support_df['vol_shares'].sum()
                if support_vol > 0:
                    mf_net_buy_in_zone = support_df['main_force_net_vol'].clip(lower=0).sum()
                    results['support_validation_strength'] = (mf_net_buy_in_zone / support_vol) * 100
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
        return results

    def _compute_legacy_intraday_metrics(self, context: dict) -> dict:
        """
        【V1.1 · 重构清理版】
        - 核心修复: 移除了 `opening_gap_defense_strength` 的初始化。该指标的计算逻辑已完全迁移至
                     `_compute_intraday_dynamics_metrics` 方法。此举旨在根除因重构不彻底导致的
                     “幽灵覆盖”问题，即此处的 np.nan 值覆盖了新方法中正确计算出的结果。
        """
        results = {
            'active_selling_pressure': np.nan,
            'active_buying_support': np.nan,
            'upward_impulse_purity': np.nan,
            # 'opening_gap_defense_strength': np.nan, # 移除此幽灵指标
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
                pre_close_val = context.get('pre_close', 1)
                if pre_close_val == 0: pre_close_val = 1 # 避免除以零
                results['upward_impulse_purity'] = (total_price_change_up / pre_close_val) / (total_vol_up / total_vol) * 100
        decline_df = intraday_df[price_change < -0.01 * context.get('pre_close', 1)]
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
        【V2.3 · 记忆传递修复版】
        - 核心修复: 将计算出的 `main_force_cumulative_cost` 添加到返回的指标字典中。
                     此举是解决跨区块记忆丢失问题的关键一步，它确保了这个重要的状态值能够
                     被上层服务捕获，并存入传递给下一个区块的记忆中。
        """
        results = {
            'main_force_cost_advantage': np.nan,
            'auction_intent_signal': np.nan,
            'auction_closing_position': np.nan,
            'main_force_cumulative_cost': np.nan,
        }
        # 移除了所有与 enable_mfca_probe 相关的探针代码
        close_price = context.get('close_price')
        daily_main_force_buy_cost = context.get('avg_cost_main_buy')
        prev_metrics = context.get('prev_metrics', {})
        prev_cumulative_cost = prev_metrics.get('main_force_cumulative_cost')
        main_force_cumulative_cost = np.nan
        if pd.notna(daily_main_force_buy_cost):
            if pd.notna(prev_cumulative_cost):
                span = 55
                alpha = 2 / (span + 1)
                main_force_cumulative_cost = daily_main_force_buy_cost * alpha + prev_cumulative_cost * (1 - alpha)
            else:
                main_force_cumulative_cost = daily_main_force_buy_cost
        elif pd.notna(prev_cumulative_cost):
            main_force_cumulative_cost = prev_cumulative_cost
        self.ctx['main_force_cumulative_cost'] = main_force_cumulative_cost
        results['main_force_cumulative_cost'] = main_force_cumulative_cost
        if pd.notna(close_price) and pd.notna(main_force_cumulative_cost) and main_force_cumulative_cost > 0:
            results['main_force_cost_advantage'] = (close_price / main_force_cumulative_cost - 1) * 100
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

    def _compute_realtime_orderbook_metrics(self, context: dict) -> dict:
        """
        【V3.2 · 索引对齐修复版】
        - 核心修复: 解决了因高频数据中存在重复时间戳索引，导致在使用 `.loc` 过滤权重时引发的“形状不匹配”
                     TypeError。
        - 核心思想: 废弃基于索引的过滤，改为创建统一的布尔掩码 `valid_mask`。通过将此掩码同时应用于
                     `instant_intent` 和 `weights`，确保了最终用于加权平均的两个序列长度绝对一致，
                     从根本上修复了“错位的舞伴”问题。
        """
        results = {
            'mf_cost_zone_defense_intent': np.nan,
            'mf_cost_zone_buy_intent': np.nan, # 新增行
            'mf_cost_zone_sell_intent': np.nan, # 新增行
            'floating_chip_cleansing_efficiency': np.nan,
        }
        realtime_df = context.get('realtime_data')
        if realtime_df is None or realtime_df.empty:
            return results
        required_cols = [f'{prefix}{i}_{suffix}' for prefix in ['b', 'a'] for i in range(1, 6) for suffix in ['p', 'v']]
        if not all(col in realtime_df.columns for col in required_cols) or 'volume' not in realtime_df.columns:
            logger.warning(f"[{context.get('stock_code')}] [{context.get('trade_date')}] 实时盘口指标计算跳过，因缺少必要的五档行情或volume列。")
            return results
        numeric_cols = [f'{prefix}{i}_{suffix}' for prefix in ['b', 'a'] for i in range(1, 6) for suffix in ['p', 'v']] + ['volume']
        for col in numeric_cols:
            if col in realtime_df.columns:
                realtime_df[col] = pd.to_numeric(realtime_df[col], errors='coerce')
        main_force_cost = context.get('main_force_cumulative_cost')
        atr = context.get('atr_14d')
        if pd.notna(main_force_cost) and pd.notna(atr) and atr > 0:
            cost_zone_low = main_force_cost - 0.5 * atr
            cost_zone_high = main_force_cost + 0.5 * atr
            def _gaussian_weight(price_series, center, sigma):
                if sigma > 0:
                    return np.exp(-((price_series - center)**2) / (2 * sigma**2))
                return pd.Series(np.where(price_series == center, 1.0, 0.0), index=price_series.index)
            bid_prices_cols = [f'b{i}_p' for i in range(1, 6)]
            bid_vols_cols = [f'b{i}_v' for i in range(1, 6)]
            ask_prices_cols = [f'a{i}_p' for i in range(1, 6)]
            ask_vols_cols = [f'a{i}_v' for i in range(1, 6)]
            total_weighted_bid_power = pd.Series(0.0, index=realtime_df.index)
            total_weighted_ask_power = pd.Series(0.0, index=realtime_df.index)
            for p_col, v_col in zip(bid_prices_cols, bid_vols_cols):
                price_series = realtime_df[p_col]
                vol_series = realtime_df[v_col]
                in_zone_mask = (price_series >= cost_zone_low) & (price_series <= cost_zone_high)
                gravity_weight = _gaussian_weight(price_series, center=main_force_cost, sigma=0.5 * atr)
                level_power = price_series * vol_series * gravity_weight
                total_weighted_bid_power += level_power.where(in_zone_mask, 0)
            for p_col, v_col in zip(ask_prices_cols, ask_vols_cols):
                price_series = realtime_df[p_col]
                vol_series = realtime_df[v_col]
                in_zone_mask = (price_series >= cost_zone_low) & (price_series <= cost_zone_high)
                gravity_weight = _gaussian_weight(price_series, center=main_force_cost, sigma=0.5 * atr)
                level_power = price_series * vol_series * gravity_weight
                total_weighted_ask_power += level_power.where(in_zone_mask, 0)
            total_power = total_weighted_bid_power + total_weighted_ask_power
            instant_intent = (total_weighted_bid_power - total_weighted_ask_power) / total_power.replace(0, np.nan)
            if 'volume' in realtime_df.columns:
                weights = realtime_df['volume'].fillna(0).clip(lower=0)
                # 使用布尔掩码进行过滤，确保长度一致
                valid_mask = instant_intent.notna()
                if valid_mask.any():
                    valid_intent = instant_intent[valid_mask]
                    valid_weights = weights[valid_mask]
                    if valid_weights.sum() > 0:
                        weighted_intent = np.average(valid_intent, weights=valid_weights)
                        results['mf_cost_zone_defense_intent'] = np.clip(weighted_intent * 100, -100, 100)
                        # 新增行：计算买方和卖方意图强度
                        weighted_buy_power = np.average(total_weighted_bid_power[valid_mask], weights=valid_weights)
                        weighted_sell_power = np.average(total_weighted_ask_power[valid_mask], weights=valid_weights)
                        total_weighted_power = weighted_buy_power + weighted_sell_power
                        if total_weighted_power > 0:
                            results['mf_cost_zone_buy_intent'] = (weighted_buy_power / total_weighted_power) * 100
                            results['mf_cost_zone_sell_intent'] = (weighted_sell_power / total_weighted_power) * 100
        intraday_df = context.get('processed_intraday_df')
        total_daily_volume = context.get('daily_turnover_volume')
        if intraday_df is None or intraday_df.empty or pd.isna(atr) or atr <= 0 or pd.isna(total_daily_volume) or total_daily_volume <= 0:
            return results
        if 'open' not in intraday_df.columns or 'close' not in intraday_df.columns:
            intraday_df['price_impulse'] = intraday_df['minute_vwap'].diff().abs()
            intraday_df['open'] = intraday_df['minute_vwap'].shift(1).fillna(intraday_df['minute_vwap'])
            intraday_df['close'] = intraday_df['minute_vwap']
            intraday_df['low'] = intraday_df[['open', 'close']].min(axis=1)
        else:
            intraday_df['price_impulse'] = intraday_df['close'].diff().abs()
        impulse_threshold = 0.5 * atr / 240
        impulse_events = intraday_df[intraday_df['price_impulse'] > impulse_threshold]
        if not impulse_events.empty:
            cleansing_scores = []
            for event_time, event_row in impulse_events.iterrows():
                if event_row['close'] >= event_row['open']:
                    continue
                impulse_window_start = event_time - pd.Timedelta(minutes=1)
                impulse_window_end = event_time + pd.Timedelta(minutes=1)
                recovery_window_end = event_time + pd.Timedelta(minutes=15)
                impulse_df = intraday_df.loc[impulse_window_start:impulse_window_end]
                recovery_df = intraday_df.loc[impulse_window_end:recovery_window_end]
                if impulse_df.empty or recovery_df.empty:
                    continue
                price_start = event_row['open']
                price_trough = impulse_df['low'].min()
                price_recovery_end = recovery_df['close'].iloc[-1]
                price_drop = price_start - price_trough
                if price_drop <= 0: continue
                recovery_score = (price_recovery_end - price_trough) / price_drop
                recovery_score = np.clip(recovery_score, 0, 1.5)
                volume_during_impulse = impulse_df['vol_shares'].sum()
                normalized_volume_cost = volume_during_impulse / total_daily_volume
                normalized_price_impact = price_drop / atr
                if normalized_volume_cost <= 0: continue
                cost_effectiveness_score = normalized_price_impact / normalized_volume_cost
                mf_participation_score = 0.0
                if 'main_force_net_vol' in impulse_df.columns:
                    mf_net_vol_impulse = impulse_df['main_force_net_vol'].sum()
                    mf_participation_score = np.tanh(mf_net_vol_impulse / volume_during_impulse) if volume_during_impulse > 0 else 0
                final_event_score = recovery_score * np.log1p(cost_effectiveness_score) * (1 + mf_participation_score)
                cleansing_scores.append({'score': final_event_score, 'volume': volume_during_impulse})
            if cleansing_scores:
                scores_df = pd.DataFrame(cleansing_scores)
                if scores_df['volume'].sum() > 0:
                    weighted_avg_score = np.average(scores_df['score'], weights=scores_df['volume'])
                    results['floating_chip_cleansing_efficiency'] = np.clip(weighted_avg_score * 10, -100, 100)
        return results

    def _compute_microstructure_dynamics(self, context: dict) -> dict:
        """
        【V1.0 · 微观动力学引擎】
        - 核心职责: 利用Tick和Level5数据，锻造订单流失衡(OFI)、扫单强度和盘口流动性斜率等高频博弈指标。
        """
        from scipy.stats import linregress
        results = {
            'order_flow_imbalance': np.nan,
            'buy_sweep_intensity': np.nan,
            'sell_sweep_intensity': np.nan,
            'liquidity_slope': np.nan,
        }
        tick_df = context.get('tick_data') # 假设tick数据已传入context
        level5_df = context.get('level5_data') # 假设level5数据已传入context
        realtime_df = context.get('realtime_data') # 融合后的realtime数据
        total_volume = context.get('daily_turnover_volume')
        if tick_df is None or tick_df.empty or level5_df is None or level5_df.empty or total_volume == 0:
            return results
        # --- 1. 订单流失衡 (OFI) ---
        merged_hf_df = pd.merge_asof(
            tick_df.sort_index().reset_index(),
            level5_df.sort_index().reset_index(),
            on='trade_time',
            direction='backward'
        ).set_index('trade_time')
        if not merged_hf_df.empty and 'buy_price1' in merged_hf_df.columns and 'sell_price1' in merged_hf_df.columns:
            merged_hf_df['mid_price'] = (merged_hf_df['buy_price1'] + merged_hf_df['sell_price1']) / 2
            merged_hf_df['prev_mid_price'] = merged_hf_df['mid_price'].shift(1)
            buy_pressure = np.where(merged_hf_df['mid_price'] >= merged_hf_df['prev_mid_price'], merged_hf_df['buy_volume1'].shift(1), 0)
            sell_pressure = np.where(merged_hf_df['mid_price'] <= merged_hf_df['prev_mid_price'], merged_hf_df['sell_volume1'].shift(1), 0)
            merged_hf_df['ofi'] = buy_pressure - sell_pressure
            results['order_flow_imbalance'] = merged_hf_df['ofi'].sum() / total_volume
        # --- 2. 扫单强度 (Sweep Intensity) ---
        min_sweep_len = 3 # 至少连续3笔同向成交
        tick_df['block'] = (tick_df['type'] != tick_df['type'].shift()).cumsum()
        tick_df['block_size'] = tick_df.groupby('block')['type'].transform('size')
        sweep_candidates = tick_df[(tick_df['block_size'] >= min_sweep_len) & (tick_df['type'].isin(['B', 'S']))]
        buy_sweep_vol, sell_sweep_vol = 0, 0
        if not sweep_candidates.empty:
            for _, group_sweep in sweep_candidates.groupby('block'):
                trade_type = group_sweep['type'].iloc[0]
                prices = group_sweep['price']
                if trade_type == 'B' and prices.is_monotonic_increasing:
                    buy_sweep_vol += group_sweep['volume'].sum()
                elif trade_type == 'S' and prices.is_monotonic_decreasing:
                    sell_sweep_vol += group_sweep['volume'].sum()
        total_buy_vol = tick_df[tick_df['type'] == 'B']['volume'].sum()
        total_sell_vol = tick_df[tick_df['type'] == 'S']['volume'].sum()
        if total_buy_vol > 0: results['buy_sweep_intensity'] = buy_sweep_vol / total_buy_vol
        if total_sell_vol > 0: results['sell_sweep_intensity'] = sell_sweep_vol / total_sell_vol
        # --- 3. 盘口流动性斜率 (Liquidity Slope) ---
        if realtime_df is not None and not realtime_df.empty:
            slopes = []
            snapshot_volumes = realtime_df['volume'].diff().fillna(0).clip(lower=0)
            for _, row in realtime_df.iterrows():
                mid_price = (row['b1_p'] + row['a1_p']) / 2
                if mid_price > 0:
                    ask_prices = np.array([row[f'a{i}_p'] for i in range(1, 6)])
                    ask_volumes = np.array([row[f'a{i}_v'] for i in range(1, 6)]) * 100
                    valid_asks = (ask_prices > 0) & (ask_volumes > 0)
                    if np.sum(valid_asks) > 1:
                        x = (ask_prices[valid_asks] - mid_price) / mid_price
                        y = np.cumsum(ask_volumes[valid_asks])
                        slope, _, _, _, _ = linregress(x, y)
                        slopes.append(slope)
            if slopes and snapshot_volumes.sum() > 0:
                results['liquidity_slope'] = np.average(slopes, weights=snapshot_volumes.iloc[:len(slopes)])
        return results

    def _compute_tactical_intent_metrics(self, context: dict) -> dict:
        """
        【V1.0 · 战术归因引擎】
        - 核心职责: 锻造用于识别特定主力战术意图的高级指标。
        - 核心新增: 新增 `suppressive_accumulation_intensity` (打压吸筹强度) 指标。
                     该指标旨在通过融合微观的隐蔽买入证据和买盘质量，来识别主力在价格
                     受抑制的宏观环境下进行的“打压吸筹”战术，从而解决“信号尺度悖论”。
        """
        results = {
            'suppressive_accumulation_intensity': np.nan,
            'supportive_distribution_intensity': np.nan, # 新增行
        }
        # 1. 获取宏观环境
        close_price = context.get('close_price')
        pre_close = context.get('pre_close')
        if pd.isna(close_price) or pd.isna(pre_close) or pre_close <= 0:
            return results
        pct_change = (close_price / pre_close) - 1
        # 2. 获取微观证据
        covert_accumulation = context.get('covert_accumulation_signal', 0.0)
        covert_distribution = context.get('covert_distribution_signal', 0.0) # 新增行
        conviction_flow = context.get('conviction_flow_index', 0.0)
        profit_taking_flow_ratio = context.get('profit_taking_flow_ratio', 0.0) # 新增行
        # 3. 战术归因与计算
        # “打压吸筹”战术只在价格下跌或微涨时成立
        # 将触发条件从允许微涨(<= 0.01)收紧为平盘或下跌(<= 0)，使其更符合“打压”的定义
        suppression_mask = pct_change <= 0
        if suppression_mask:
            # 融合直接证据(隐蔽吸筹)和间接证据(信念流转)
            # 使用 np.log1p 增强对信念流指数的敏感度
            intensity_score = (covert_accumulation / 100) * np.log1p(np.clip(conviction_flow, 0, None))
            results['suppressive_accumulation_intensity'] = np.clip(intensity_score * 100, 0, 100)
        else:
            results['suppressive_accumulation_intensity'] = 0.0
        # 新增行：计算支撑性派发强度
        supportive_mask = pct_change >= 0 # 价格上涨或横盘时
        if supportive_mask:
            # 融合隐蔽派发和获利兑现流量
            intensity_score = (covert_distribution / 100) * np.log1p(np.clip(profit_taking_flow_ratio, 0, None))
            results['supportive_distribution_intensity'] = np.clip(intensity_score * 100, 0, 100)
        else:
            results['supportive_distribution_intensity'] = 0.0
        return results

    def _get_absolute_normalized_score(self, value: float, neutral_point: float, sensitivity: float, ascending: bool = True) -> float:
        """
        【V1.0 · 绝对真理映射】对具有绝对意义的指标进行非线性归一化。
        - 核心思想: 使用 tanh 函数，将指标值根据其偏离“中性点”的程度，映射到 [0, 1] 区间。
                     解决了 percentileofscore 只能进行相对历史排名，无法体现指标绝对好坏的问题。
        - value: 当前指标值。
        - neutral_point: 指标的中性值 (例如，成本优势为0)。
        - sensitivity: 敏感度，调节 tanh 函数的陡峭程度。
        - ascending: 指标是否值越大越好。
        """
        if not pd.notna(value):
            return 0.5 # 对于缺失值，返回中性分
        deviation = value - neutral_point
        tanh_score = np.tanh(deviation * sensitivity) # 映射到 [-1, 1]
        if not ascending:
            tanh_score = -tanh_score
        return (tanh_score + 1) / 2 # 映射到 [0, 1]

    def _prepare_behavioral_data_for_chips(self, context: dict) -> tuple:
        """
        【V12.5 · 新增】高频数据准备器。
        - 核心职责: 从上下文中提取日线数据和原始高频数据(tick, level5, realtime)，
                     准备并返回高频计算所需的通用数据字典和合并后的原始高频DataFrame。
        """
        import numpy as np
        # [新增的代码块] 开始
        daily_total_volume = context.get('daily_turnover_volume', 0)
        daily_total_amount = pd.to_numeric(context.get('amount'), errors='coerce') * 1000 if 'amount' in context else 0
        daily_vwap = daily_total_amount / daily_total_volume if daily_total_volume > 0 else np.nan
        atr = context.get('atr_14d')
        day_open, day_close = context.get('open_price'), context.get('close_price')
        day_high, day_low = context.get('high_price'), context.get('low_price')
        tick_data = context.get('tick_data')
        level5_data = context.get('level5_data')
        realtime_data = context.get('realtime_data')
        raw_hf_df = pd.DataFrame()
        if tick_data is not None and not tick_data.empty and level5_data is not None and not level5_data.empty:
            merged_hf = pd.merge_asof(
                tick_data.sort_index(), level5_data.sort_index(),
                left_index=True, right_index=True, direction='backward'
            ).dropna(subset=['buy_price1', 'sell_price1', 'amount', 'volume'])
            if realtime_data is not None and not realtime_data.empty and not merged_hf.empty:
                realtime_prepped = realtime_data[['volume']].copy()
                realtime_prepped['snapshot_time'] = realtime_prepped.index
                merged_hf = pd.merge_asof(
                    merged_hf, realtime_prepped, left_index=True, right_index=True,
                    direction='backward', suffixes=('_tick', '_realtime')
                )
            if not merged_hf.empty:
                merged_hf.rename(columns={'volume_tick': 'volume'}, inplace=True)
                raw_hf_df = merged_hf
        common_data = {
            'daily_total_volume': daily_total_volume, 'daily_total_amount': daily_total_amount,
            'daily_vwap': daily_vwap, 'atr': atr, 'day_open': day_open, 'day_close': day_close,
            'day_high': day_high, 'day_low': day_low
        }
        return raw_hf_df, common_data
        # [新增的代码块] 结束

    def _engineer_hf_features_for_chips(self, raw_hf_df: pd.DataFrame, daily_total_volume: float) -> tuple[pd.DataFrame, dict]:
        """
        【V12.6 · 依赖补全修复版】
        - 核心修复: 补全了 `mid_price_delta` 列的计算逻辑。此前的版本遗漏了这一关键步骤，
                     导致下游 `_compute_contextual_action_metrics` 方法在尝试访问该列时
                     因 `KeyError` 而崩溃。
        """
        import numpy as np
        features = {
            'mf_trades': pd.DataFrame(), 'buy_trades_mask': pd.Series(dtype=bool),
            'sell_trades_mask': pd.Series(dtype=bool), 'total_mf_vol': 0.0,
            'mf_buy_vol': 0.0, 'mf_sell_vol': 0.0, 'offensive_volume': 0.0,
            'passive_volume': 0.0, 'hf_mf_buy_vwap': np.nan, 'hf_mf_sell_vwap': np.nan,
        }
        if raw_hf_df is None or raw_hf_df.empty:
            return pd.DataFrame(), features
        hf_analysis_df = raw_hf_df.copy()
        hf_analysis_df['mid_price'] = (hf_analysis_df['buy_price1'] + hf_analysis_df['sell_price1']) / 2
        hf_analysis_df['prev_mid_price'] = hf_analysis_df['mid_price'].shift(1)
        hf_analysis_df['mid_price_delta'] = hf_analysis_df['mid_price'].diff() # 补全 mid_price_delta 的计算
        buy_pressure = np.where(hf_analysis_df['mid_price'] >= hf_analysis_df['prev_mid_price'], hf_analysis_df['buy_volume1'].shift(1), 0)
        sell_pressure = np.where(hf_analysis_df['mid_price'] <= hf_analysis_df['prev_mid_price'], hf_analysis_df['sell_volume1'].shift(1), 0)
        hf_analysis_df['ofi'] = buy_pressure - sell_pressure
        is_main_force_trade = hf_analysis_df['amount'] > 200000
        hf_analysis_df['main_force_ofi'] = np.where(is_main_force_trade, hf_analysis_df['ofi'], 0)
        mf_trades = hf_analysis_df[is_main_force_trade].copy()
        if mf_trades.empty:
            return hf_analysis_df, features
        features['mf_trades'] = mf_trades
        buy_trades_mask = mf_trades['type'] == 'B'
        sell_trades_mask = mf_trades['type'] == 'S'
        features['buy_trades_mask'] = buy_trades_mask
        features['sell_trades_mask'] = sell_trades_mask
        total_mf_vol = mf_trades['volume'].sum()
        features['total_mf_vol'] = total_mf_vol
        if total_mf_vol > 0:
            features['mf_buy_vol'] = mf_trades[buy_trades_mask]['volume'].sum()
            features['mf_sell_vol'] = mf_trades[sell_trades_mask]['volume'].sum()
        return hf_analysis_df, features

    def _compute_contextual_action_metrics(self, context: dict) -> dict:
        """
        【V12.7 · 零强度原则版】
        - 核心升维: 引入“零强度”原则。当股价未进入特定区域（如未跌破主峰区）时，
                     相关的行为强度指标（如吸筹烈度）不再返回NaN，而是返回0.0，
                     代表该行为“零发生”，从而消除信息黑洞，使指标体系在逻辑上彻底完备。
        - 核心修复: 修正了“主峰区派发烈度”中对战术成果的计算逻辑。
        """
        # [修改的代码块] 将默认值从 np.nan 修改为 0.0
        metrics = {
            'distribution_at_peak_intensity': 0.0,
            'absorption_at_peak_intensity': 0.0,
            'breakthrough_of_peak_quality': 0.0, # 修改：从 np.nan 变为 0.0
            'defense_of_peak_quality': 0.0,      # 修改：从 np.nan 变为 0.0
        }
        raw_hf_df, common_data = self._prepare_behavioral_data_for_chips(context)
        hf_analysis_df, hf_features = self._engineer_hf_features_for_chips(raw_hf_df, common_data.get('daily_total_volume', 0))
        debug_params = context.get('debug_params', {})
        enable_probe = debug_params.get('enable_mfca_probe', False)
        target_date_str = debug_params.get('target_date')
        trade_date = context.get('trade_date')
        is_target_date = str(trade_date) == target_date_str
        should_probe = enable_probe and is_target_date
        dominant_peak_cost = context.get('dominant_peak_cost')
        atr = common_data.get('atr')
        if hf_analysis_df.empty or pd.isna(dominant_peak_cost) or pd.isna(atr) or atr <= 0:
            return metrics
        peak_battle_zone_radius = 0.5 * atr
        peak_zone_upper = dominant_peak_cost + peak_battle_zone_radius
        peak_zone_lower = dominant_peak_cost - peak_battle_zone_radius
        above_peak_zone_df = hf_analysis_df[hf_analysis_df['price'] > peak_zone_upper]
        below_peak_zone_df = hf_analysis_df[hf_analysis_df['price'] < peak_zone_lower]
        # 1. 计算主峰区派发烈度
        if not above_peak_zone_df.empty:
            focus_numerator = above_peak_zone_df['main_force_ofi'].abs().sum()
            focus_denominator = hf_analysis_df['main_force_ofi'].abs().sum()
            focus_component = focus_numerator / focus_denominator if focus_denominator > 0 else 0
            intent_numerator = above_peak_zone_df['main_force_ofi'].clip(upper=0).abs().sum()
            intent_denominator = above_peak_zone_df['main_force_ofi'].abs().sum()
            intent_component = intent_numerator / intent_denominator if intent_denominator > 0 else 0
            price_start = above_peak_zone_df['price'].iloc[0]
            price_end = above_peak_zone_df['price'].iloc[-1]
            # 修正派发成果的计算逻辑，价格下跌为正成果
            outcome_component = np.tanh((price_start - price_end) / atr)
            metrics['distribution_at_peak_intensity'] = np.clip((focus_component * intent_component * (1 + outcome_component)) * 100, 0, 100) # 修改：增加 np.clip 归一化到 [0, 100]
        # 2. 计算主峰区吸筹烈度
        if not below_peak_zone_df.empty:
            focus_numerator = below_peak_zone_df['main_force_ofi'].abs().sum()
            focus_denominator = hf_analysis_df['main_force_ofi'].abs().sum()
            focus_component = focus_numerator / focus_denominator if focus_denominator > 0 else 0
            intent_numerator = below_peak_zone_df['main_force_ofi'].clip(lower=0).sum()
            intent_denominator = below_peak_zone_df['main_force_ofi'].abs().sum()
            intent_component = intent_numerator / intent_denominator if intent_denominator > 0 else 0
            price_start = below_peak_zone_df['price'].iloc[0]
            price_end = below_peak_zone_df['price'].iloc[-1]
            outcome_component = np.tanh((price_end - price_start) / atr)
            metrics['absorption_at_peak_intensity'] = np.clip((focus_component * intent_component * (1 + outcome_component)) * 100, 0, 100) # 修改：增加 np.clip 归一化到 [0, 100]
        # 3. 计算突破主峰质量
        breakthrough_event_df = hf_analysis_df[hf_analysis_df['price'] > peak_zone_upper]
        if not breakthrough_event_df.empty:
            magnitude_component = (breakthrough_event_df['price'].max() - peak_zone_upper) / atr
            conviction_numerator = breakthrough_event_df['main_force_ofi'].clip(lower=0).sum()
            conviction_denominator = breakthrough_event_df['main_force_ofi'].abs().sum()
            conviction_component = conviction_numerator / conviction_denominator if conviction_denominator > 0 else 0
            efficiency_numerator = breakthrough_event_df['mid_price_delta'].sum()
            efficiency_denominator = breakthrough_event_df['main_force_ofi'].clip(lower=0).sum()
            efficiency_component = np.tanh(efficiency_numerator / (efficiency_denominator * 1e-6)) if efficiency_denominator > 0 else 0
            metrics['breakthrough_of_peak_quality'] = np.clip((magnitude_component * conviction_component * efficiency_component) * 100, 0, 100) # 修改：增加 np.clip 归一化到 [0, 100]
        # 4. 计算防守主峰质量
        day_low = common_data.get('day_low')
        day_close = common_data.get('day_close')
        if day_low >= peak_zone_lower:
            metrics['defense_of_peak_quality'] = 100.0
        else:
            recovery_df = hf_analysis_df[hf_analysis_df['price'] < peak_zone_lower]
            if not recovery_df.empty and pd.notna(day_close):
                recovery_magnitude = (day_close - day_low) / atr
                recovery_conviction_num = recovery_df['main_force_ofi'].clip(lower=0).sum()
                recovery_conviction_den = recovery_df['main_force_ofi'].abs().sum()
                recovery_conviction = recovery_conviction_num / recovery_conviction_den if recovery_conviction_den > 0 else 0
                metrics['defense_of_peak_quality'] = np.clip((recovery_magnitude * recovery_conviction) * 100, 0, 100) # 修改：增加 np.clip 归一化到 [0, 100]
        if should_probe:
            print(f"\n{'='*20} [探针 C.1 - 情境行为内核 @ {str(trade_date)}] {'='*20}")
            print(f"  - 核心情境: 主峰成本={dominant_peak_cost:.2f}, ATR={atr}")
            print(f"  - 主峰战区定义: [{peak_zone_lower:.2f} - {peak_zone_upper:.2f}]")
            print(f"  - 指标计算详情:")
            # [修改的代码块] 升级探针输出
            print(f"    - 主峰区派发烈度: {metrics.get('distribution_at_peak_intensity', np.nan):.4f}")
            if not above_peak_zone_df.empty:
                print(f"      - (已触发计算)")
            else:
                print(f"      - (价格未进入上方区域，强度为0)")
            print(f"    - 主峰区吸筹烈度: {metrics.get('absorption_at_peak_intensity', np.nan):.4f}")
            if not below_peak_zone_df.empty:
                print(f"      - (已触发计算)")
            else:
                print(f"      - (价格未进入下方区域，强度为0)")
            print(f"    - 突破主峰质量: {metrics.get('breakthrough_of_peak_quality', np.nan):.4f}")
            if not breakthrough_event_df.empty:
                 print(f"      - (已触发计算)")
            else:
                 print(f"      - (未发生向上突破)")
            if day_low >= peak_zone_lower:
                print(f"    - 防守主峰质量: {metrics.get('defense_of_peak_quality', np.nan):.4f}")
                print(f"      - (未发生向下破位，视为完美防守)")
            else:
                print(f"    - 防守主峰质量: {metrics.get('defense_of_peak_quality', np.nan):.4f}")
                print(f"      - (已发生向下破位，评估修复质量)")
        return metrics






