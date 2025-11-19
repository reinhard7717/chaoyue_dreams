# 文件: services/chip_feature_calculator.py

import pandas as pd
import numpy as np
import datetime
from scipy.signal import find_peaks
from decimal import Decimal
import logging
logger = logging.getLogger(__name__)

class ChipFeatureCalculator:
    """
    【V11.0 · 战术中枢重构版】
    - 核心重构: 本类现在是所有筹码指标计算的唯一核心，接收原始数据并完成所有计算。
    - 逻辑注入: 新增了成交量微观结构和筹码集中度动态归因的计算逻辑。
    """
    def __init__(self, df: pd.DataFrame, context: dict):
        self.df = df # 筹码分布数据
        self.context = context
        # 从 context 中获取日内数据、交易日期和调试参数
        intraday_data_raw = self.context.get('intraday_data', pd.DataFrame())
        trade_date = self.context.get('trade_date')
        debug_params = self.context.get('debug_params', {})
        # 调用 _prepare_intraday_data_features 进行预处理
        self.processed_intraday_df = self._prepare_intraday_data_features(intraday_data_raw, trade_date, debug_params)
        # 将处理后的日内数据存入 context，供后续方法使用
        self.context['processed_intraday_df'] = self.processed_intraday_df
        # 初始化其他属性
        self.stock_code = context.get('stock_code')
        self.trade_date = context.get('trade_date')
        self.debug_params = debug_params

    def calculate_all_metrics(self) -> dict:
        """
        【V15.2 · 利润实现质量计算顺序修复版】
        - 核心修复: 调整 `profit_realization_quality` 的计算顺序，确保其依赖的 `profit_taking_flow_ratio` 已在上下文中。
        - 核心新增: 增加第六个计算单元 `_compute_microstructure_game_metrics`，专门负责计算基于高频数据的筹码微观博弈指标。
        - 核心重构: 废除中间调度方法，由本方法直接指挥六大计算单元，实现扁平化、高效的指挥链。
        - 核心思想: 将所有计算逻辑整合为六个高度内聚的“合成营”方法，并按依赖顺序依次执行。
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
        # 步骤1: 提取基础指标并更新上下文
        summary_info = self._get_summary_metrics_from_context()
        self.ctx.update(summary_info)
        # 步骤2: 按依赖顺序调用六大计算单元，并持续更新上下文
        static_structure_metrics = self._compute_static_structure_metrics()
        self.ctx.update(static_structure_metrics)
        intraday_dynamics_metrics = self._compute_intraday_dynamics_metrics(self.ctx)
        self.ctx.update(intraday_dynamics_metrics)
        cross_day_flow_metrics = self._compute_cross_day_flow_metrics(self.ctx)
        self.ctx.update(cross_day_flow_metrics)
        # 修改行：将 profit_realization_quality 的计算移到此处，确保依赖项已在上下文中
        debug_params = self.ctx.get('debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        is_probe_date = False
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0]).date()
            if probe_date_naive == trade_date:
                is_probe_date = True
        winner_profit_margin_avg = self.ctx.get('winner_profit_margin_avg')
        total_winner_rate = self.ctx.get('total_winner_rate')
        profit_taking_flow_ratio = self.ctx.get('profit_taking_flow_ratio')
        if profit_taking_flow_ratio is None:
            profit_taking_flow_ratio = np.nan
        profit_realization_quality = np.nan
        if is_probe_date and pd.isna(winner_profit_margin_avg):
            print(f"    -> [利润实现质量探针] @ {trade_date}: 'winner_profit_margin_avg' 缺失。")
        if is_probe_date and pd.isna(total_winner_rate):
            print(f"    -> [利润实现质量探针] @ {trade_date}: 'total_winner_rate' 缺失。")
        if is_probe_date and pd.isna(profit_taking_flow_ratio):
            print(f"    -> [利润实现质量探针] @ {trade_date}: 'profit_taking_flow_ratio' 缺失。")
        if is_probe_date and pd.notna(profit_taking_flow_ratio) and profit_taking_flow_ratio <= 0:
            print(f"    -> [利润实现质量探针] @ {trade_date}: 'profit_taking_flow_ratio' 为 {profit_taking_flow_ratio}。")
        if pd.notna(winner_profit_margin_avg) and pd.notna(total_winner_rate) and pd.notna(profit_taking_flow_ratio) and profit_taking_flow_ratio > 0:
            profit_realization_quality = (winner_profit_margin_avg * (total_winner_rate / 100)) / (profit_taking_flow_ratio / 100)
        self.ctx['profit_realization_quality'] = profit_realization_quality
        game_theoretic_metrics = self._compute_game_theoretic_metrics(self.ctx)
        self.ctx.update(game_theoretic_metrics)
        vital_signs_metrics = self._compute_vital_sign_metrics(self.ctx)
        self.ctx.update(vital_signs_metrics)
        microstructure_game_metrics = self._compute_microstructure_game_metrics(self.ctx)
        self.ctx.update(microstructure_game_metrics)
        # 步骤3: 整合所有计算结果
        all_metrics = {
            **summary_info,
            **static_structure_metrics,
            **intraday_dynamics_metrics,
            **cross_day_flow_metrics,
            **game_theoretic_metrics,
            **vital_signs_metrics,
            **microstructure_game_metrics,
            'profit_realization_quality': profit_realization_quality, # 修改行：将 profit_realization_quality 添加到最终结果
        }
        # 步骤4: 计算最终的健康分 (它依赖于之前计算出的所有指标)
        health_score_info = self._calculate_chip_structure_health_score(self.ctx)
        all_metrics.update(health_score_info)
        # 步骤5: 清理临时或中间指标
        all_metrics.pop('peak_range_low', None)
        all_metrics.pop('peak_range_high', None)
        return all_metrics

    def _prepare_intraday_data_features(self, intraday_df: pd.DataFrame, trade_date: datetime.date, debug_params: dict) -> pd.DataFrame:
        import pytz # 修改行：导入 pytz 模块
        results = {}
        is_probe_date = False
        if debug_params.get('probe_dates') and trade_date == pd.to_datetime(debug_params['probe_dates'][0]).date():
            is_probe_date = True
        if is_probe_date:
            print(f"    -> [日内数据特征探针] @ {trade_date}: _prepare_intraday_data_features 接收到的 intraday_df 索引状态：")
            if not intraday_df.empty:
                print(f"       - 索引类型: {type(intraday_df.index)}")
                print(f"       - 索引时区: {intraday_df.index.tz}")
                print(f"       - 索引前5个时间 (原始): {[t.strftime('%Y-%m-%d %H:%M:%S%z') for t in intraday_df.index[:5].tolist()]}")
            else:
                print(f"       - intraday_df 为空。")
        if intraday_df.empty:
            if is_probe_date:
                print(f"    -> [日内数据特征探针] @ {trade_date}: _prepare_intraday_data_features 接收到空 intraday_df。")
            return pd.DataFrame()
        required_cols = ['open', 'high', 'low', 'close', 'vol', 'amount', 'buy_vol_raw', 'sell_vol_raw', 'amount_yuan', 'vol_shares', 'minute_vwap', 'vol_weight', 'sm_buy_weight', 'sm_sell_weight', 'md_buy_weight', 'md_sell_weight', 'lg_buy_weight', 'lg_sell_weight', 'elg_buy_weight', 'elg_sell_weight', 'sm_buy_vol_attr', 'sm_sell_vol_attr', 'md_buy_vol_attr', 'md_sell_vol_attr', 'lg_buy_vol_attr', 'lg_sell_vol_attr', 'elg_buy_vol_attr', 'elg_sell_vol_attr', 'main_force_buy_vol', 'main_force_sell_vol', 'main_force_net_vol', 'retail_buy_vol', 'retail_sell_vol', 'retail_net_vol']
        missing_cols = [col for col in required_cols if col not in intraday_df.columns]
        if missing_cols:
            if is_probe_date:
                print(f"    -> [日内数据特征探针] @ {trade_date}: intraday_df 缺少列: {missing_cols}。")
            return pd.DataFrame()
        if is_probe_date:
            print(f"    -> [日内数据特征探针] @ {trade_date}: _prepare_intraday_data_features 接收到的 intraday_df 列：{list(intraday_df.columns)}")
            print(f"       - 初始 intraday_df['main_force_sell_vol'] sum: {intraday_df['main_force_sell_vol'].sum():.2f}")
            print(f"       - 初始 intraday_df['retail_sell_vol'] sum: {intraday_df['retail_sell_vol'].sum():.2f}")
        if not isinstance(intraday_df.index, pd.DatetimeIndex):
            intraday_df.index = pd.to_datetime(intraday_df.index)
        target_tz = pytz.timezone('Asia/Shanghai') # 修改行：明确目标时区
        if intraday_df.index.tz is None:
            intraday_df.index = intraday_df.index.tz_localize('UTC', ambiguous='infer').tz_convert(target_tz) # 修改行：使用 target_tz
            if is_probe_date:
                print(f"       - 索引被 tz_localize('UTC') 后 tz_convert('Asia/Shanghai')。")
        else:
            intraday_df.index = intraday_df.index.tz_convert(target_tz) # 修改行：使用 target_tz
            if is_probe_date:
                print(f"       - 索引被 tz_convert('Asia/Shanghai')。")
        if is_probe_date and not intraday_df.empty:
            print(f"       - 索引前5个时间 (处理后): {[t.strftime('%Y-%m-%d %H:%M:%S%z') for t in intraday_df.index[:5].tolist()]}")
        start_time = pd.to_datetime('09:25').time()
        end_time = pd.to_datetime('15:00').time()
        processed_intraday_df = intraday_df[(intraday_df.index.time >= start_time) & (intraday_df.index.time <= end_time)].copy()
        if is_probe_date:
            print(f"    -> [日内数据特征探针] @ {trade_date}: 'processed_intraday_df' 最终状态：")
            print(f"       - 是否为空: {processed_intraday_df.empty}")
            print(f"       - 行数: {len(processed_intraday_df)}")
            if not processed_intraday_df.empty:
                print(f"       - 'vol_shares' 总和: {processed_intraday_df['vol_shares'].sum():.2f}")
                print(f"       - 'main_force_sell_vol' 总和: {processed_intraday_df['main_force_sell_vol'].sum():.2f}")
                print(f"       - 'retail_sell_vol' 总和: {processed_intraday_df['retail_sell_vol'].sum():.2f}")
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
        import datetime
        results = {}
        trade_date = context.get('trade_date', 'UNKNOWN')
        debug_params = context.get('debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        is_probe_date = False
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0]).date()
            if probe_date_naive == trade_date:
                is_probe_date = True
        intraday_df = context.get('processed_intraday_df')
        total_daily_vol = context.get('daily_turnover_volume')
        daily_vwap = context.get('daily_vwap')
        pre_close = context.get('pre_close')
        results['gathering_by_support'] = np.nan
        results['gathering_by_chasing'] = np.nan
        results['dispersal_by_distribution'] = np.nan
        results['dispersal_by_capitulation'] = np.nan
        results['profit_taking_flow_ratio'] = np.nan
        results['capitulation_flow_ratio'] = np.nan
        results['active_winner_pressure_ratio'] = np.nan
        results['locked_profit_pressure_ratio'] = np.nan
        results['active_loser_pressure_ratio'] = np.nan
        results['locked_loss_pressure_ratio'] = np.nan
        results['cost_divergence'] = np.nan
        results['cost_divergence_normalized'] = np.nan
        results['winner_loser_momentum'] = np.nan
        results['chip_fatigue_index'] = np.nan
        results['peak_shoulder_growth_rate'] = np.nan
        if is_probe_date and intraday_df is None:
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'intraday_df' 为 None。")
        if is_probe_date and intraday_df.empty:
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'intraday_df' 为空。")
        if is_probe_date and pd.isna(total_daily_vol):
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'total_daily_vol' 缺失。")
        if is_probe_date and total_daily_vol <= 0:
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'total_daily_vol' 为 {total_daily_vol}。")
        if is_probe_date and pd.isna(daily_vwap):
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'daily_vwap' 缺失。")
        required_cols_1 = ['minute_vwap', 'main_force_buy_vol', 'main_force_sell_vol']
        if not intraday_df.empty and pd.notna(total_daily_vol) and total_daily_vol > 0 and pd.notna(daily_vwap):
            for col in required_cols_1:
                if col not in intraday_df.columns:
                    if is_probe_date:
                        print(f"    -> [跨日资金流探针] @ {trade_date}: 'intraday_df' 缺少列 '{col}'。")
                    return results
            intraday_df['mf_net_vol'] = intraday_df['main_force_buy_vol'] - intraday_df['main_force_sell_vol']
            gathering_vol_per_minute = intraday_df['mf_net_vol'].clip(lower=0)
            dispersal_vol_per_minute = (-intraday_df['mf_net_vol']).clip(lower=0)
            below_vwap_mask = intraday_df['minute_vwap'] < daily_vwap
            above_vwap_mask = intraday_df['minute_vwap'] > daily_vwap
            results['gathering_by_support'] = (gathering_vol_per_minute[below_vwap_mask].sum() / total_daily_vol) * 100
            results['gathering_by_chasing'] = (gathering_vol_per_minute[above_vwap_mask].sum() / total_daily_vol) * 100
            results['dispersal_by_distribution'] = (dispersal_vol_per_minute[above_vwap_mask].sum() / total_daily_vol) * 100
            results['dispersal_by_capitulation'] = (dispersal_vol_per_minute[below_vwap_mask].sum() / total_daily_vol) * 100
        else:
            results['gathering_by_support'] = np.nan
            results['gathering_by_chasing'] = np.nan
            results['dispersal_by_distribution'] = np.nan
            results['dispersal_by_capitulation'] = np.nan
        if is_probe_date and pd.isna(pre_close):
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'pre_close' 缺失。")
        required_cols_2 = ['minute_vwap', 'main_force_sell_vol', 'retail_sell_vol']
        total_sell_vol_today = 0.0
        if not intraday_df.empty and pd.notna(pre_close):
            for col in required_cols_2:
                if col not in intraday_df.columns:
                    if is_probe_date:
                        print(f"    -> [跨日资金流探针] @ {trade_date}: 'intraday_df' 缺少列 '{col}'。")
                    return results
            if is_probe_date:
                print(f"    -> [跨日资金流探针] @ {trade_date}: 检查卖出量数据。")
                print(f"       - main_force_sell_vol sum: {intraday_df['main_force_sell_vol'].sum()}")
                print(f"       - retail_sell_vol sum: {intraday_df['retail_sell_vol'].sum()}")
            total_sell_vol_today = (intraday_df['main_force_sell_vol'] + intraday_df['retail_sell_vol']).sum()
            if is_probe_date and total_sell_vol_today <= 0:
                print(f"    -> [跨日资金流探针] @ {trade_date}: 'total_sell_vol_today' 为 {total_sell_vol_today} (<=0)。")
            if total_sell_vol_today > 0:
                if is_probe_date:
                    print(f"    -> [跨日资金流探针] @ {trade_date}: 检查获利盘兑现条件。")
                    print(f"       - pre_close: {pre_close}")
                    print(f"       - intraday_df['minute_vwap'] > pre_close 筛选结果数量: {len(intraday_df[intraday_df['minute_vwap'] > pre_close])}")
                profit_taking_vol = (intraday_df[intraday_df['minute_vwap'] > pre_close]['main_force_sell_vol'] + intraday_df[intraday_df['minute_vwap'] > pre_close]['retail_sell_vol']).sum()
                capitulation_vol = (intraday_df[intraday_df['minute_vwap'] < pre_close]['main_force_sell_vol'] + intraday_df[intraday_df['minute_vwap'] < pre_close]['retail_sell_vol']).sum()
                if is_probe_date:
                    print(f"    -> [跨日资金流探针] @ {trade_date}: 'profit_taking_vol' 计算结果: {profit_taking_vol}。")
                results['profit_taking_flow_ratio'] = (profit_taking_vol / total_sell_vol_today) * 100
                results['capitulation_flow_ratio'] = (capitulation_vol / total_sell_vol_today) * 100
            else:
                if is_probe_date:
                    print(f"    -> [跨日资金流探针] @ {trade_date}: 'total_sell_vol_today' 为 {total_sell_vol_today}，无法计算比例。")
                results['profit_taking_flow_ratio'] = np.nan
                results['capitulation_flow_ratio'] = np.nan
        else:
            if is_probe_date:
                print(f"    -> [跨日资金流探针] @ {trade_date}: 'intraday_df' 为空或 'pre_close' 缺失，无法计算卖出比例。")
            results['profit_taking_flow_ratio'] = np.nan
            results['capitulation_flow_ratio'] = np.nan
        prev_df = context.get('prev_chip_distribution')
        prev_high_20d = context.get('prev_high_20d')
        prev_low_20d = context.get('prev_low_20d')
        prev_total_chips = context.get('prev_total_chip_volume')
        if is_probe_date and prev_df is None:
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'prev_chip_distribution' 为 None。")
        if is_probe_date and prev_df.empty:
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'prev_chip_distribution' 为空。")
        if is_probe_date and pd.isna(prev_high_20d):
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'prev_high_20d' 缺失。")
        if is_probe_date and pd.isna(prev_low_20d):
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'prev_low_20d' 缺失。")
        if is_probe_date and pd.isna(prev_total_chips):
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'prev_total_chip_volume' 缺失。")
        if is_probe_date and prev_total_chips <= 0:
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'prev_total_chip_volume' 为 {prev_total_chips}。")
        if is_probe_date and total_sell_vol_today <= 0:
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'total_sell_vol_today' 为 {total_sell_vol_today}。")
        if prev_df is not None and not prev_df.empty and pd.notna(prev_high_20d) and pd.notna(prev_low_20d) and pd.notna(prev_total_chips) and prev_total_chips > 0 and total_sell_vol_today > 0:
            prev_active_mask = (prev_df['price'] >= prev_low_20d) & (prev_df['price'] <= prev_high_20d)
            vol_active_winners = (prev_df[prev_active_mask & (prev_df['price'] < pre_close)]['percent'].sum() / 100) * prev_total_chips
            vol_locked_winners = (prev_df[~prev_active_mask & (prev_df['price'] < pre_close)]['percent'].sum() / 100) * prev_total_chips
            vol_active_losers = (prev_df[prev_active_mask & (prev_df['price'] > pre_close)]['percent'].sum() / 100) * prev_total_chips
            vol_locked_losers = (prev_df[~prev_active_mask & (prev_df['price'] > pre_close)]['percent'].sum() / 100) * prev_total_chips
            if is_probe_date and vol_active_winners <= 0:
                print(f"    -> [跨日资金流探针] @ {trade_date}: 'vol_active_winners' 为 {vol_active_winners}。")
            if vol_active_winners > 0: results['active_winner_pressure_ratio'] = (total_sell_vol_today / vol_active_winners) * 100
            else: results['active_winner_pressure_ratio'] = np.nan
            if is_probe_date and vol_locked_winners <= 0:
                print(f"    -> [跨日资金流探针] @ {trade_date}: 'vol_locked_winners' 为 {vol_locked_winners}。")
            if vol_locked_winners > 0: results['locked_profit_pressure_ratio'] = (total_sell_vol_today / vol_locked_winners) * 100
            else: results['locked_profit_pressure_ratio'] = np.nan
            if is_probe_date and vol_active_losers <= 0:
                print(f"    -> [跨日资金流探针] @ {trade_date}: 'vol_active_losers' 为 {vol_active_losers}。")
            if vol_active_losers > 0: results['active_loser_pressure_ratio'] = (total_sell_vol_today / vol_active_losers) * 100
            else: results['active_loser_pressure_ratio'] = np.nan
            if is_probe_date and vol_locked_losers <= 0:
                print(f"    -> [跨日资金流探针] @ {trade_date}: 'vol_locked_losers' 为 {vol_locked_losers}。")
            if vol_locked_losers > 0: results['locked_loss_pressure_ratio'] = (total_sell_vol_today / vol_locked_losers) * 100
            else: results['locked_loss_pressure_ratio'] = np.nan
        else:
            results['active_winner_pressure_ratio'] = np.nan
            results['locked_profit_pressure_ratio'] = np.nan
            results['active_loser_pressure_ratio'] = np.nan
            results['locked_loss_pressure_ratio'] = np.nan
        high_20d = context.get('high_20d')
        low_20d = context.get('low_20d')
        close_price = context.get('close_price')
        atr_14d = context.get('atr_14d')
        if is_probe_date and self.df.empty:
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'self.df' (筹码分布) 为空。")
        if is_probe_date and pd.isna(high_20d):
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'high_20d' 缺失。")
        if is_probe_date and pd.isna(low_20d):
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'low_20d' 缺失。")
        if is_probe_date and pd.isna(close_price):
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'close_price' 缺失。")
        if is_probe_date and pd.isna(atr_14d):
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'atr_14d' 缺失。")
        if is_probe_date and atr_14d <= 0:
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'atr_14d' 为 {atr_14d}。")
        if not self.df.empty and pd.notna(high_20d) and pd.notna(low_20d) and pd.notna(close_price) and pd.notna(atr_14d) and atr_14d > 0:
            active_chips_df = self.df[(self.df['price'] >= low_20d) & (self.df['price'] <= high_20d)]
            foundational_chips_df = self.df[self.df['price'] < low_20d]
            avg_cost_active = np.average(active_chips_df['price'], weights=active_chips_df['percent']) if not active_chips_df.empty and active_chips_df['percent'].sum() > 0 else np.nan
            avg_cost_foundational = np.average(foundational_chips_df['price'], weights=foundational_chips_df['percent']) if not foundational_chips_df.empty and foundational_chips_df['percent'].sum() > 0 else np.nan
            if is_probe_date and pd.isna(avg_cost_active):
                print(f"    -> [跨日资金流探针] @ {trade_date}: 'avg_cost_active' 缺失。")
            if is_probe_date and pd.isna(avg_cost_foundational):
                print(f"    -> [跨日资金流探针] @ {trade_date}: 'avg_cost_foundational' 缺失。")
            if is_probe_date and pd.notna(avg_cost_foundational) and avg_cost_foundational <= 0:
                print(f"    -> [跨日资金流探针] @ {trade_date}: 'avg_cost_foundational' 为 {avg_cost_foundational}。")
            if pd.notna(avg_cost_active) and pd.notna(avg_cost_foundational) and avg_cost_foundational > 0:
                divergence = (avg_cost_active / avg_cost_foundational - 1) * 100
                results['cost_divergence'] = divergence
                volatility_pct = (atr_14d / close_price) * 100
                if is_probe_date and volatility_pct <= 0:
                    print(f"    -> [跨日资金流探针] @ {trade_date}: 'volatility_pct' 为 {volatility_pct}。")
                if volatility_pct > 0: results['cost_divergence_normalized'] = divergence / volatility_pct
                else: results['cost_divergence_normalized'] = np.nan
            else:
                results['cost_divergence'] = np.nan
                results['cost_divergence_normalized'] = np.nan
        else:
            results['cost_divergence'] = np.nan
            results['cost_divergence_normalized'] = np.nan
        total_chip_volume = context.get('total_chip_volume')
        if is_probe_date and self.df.empty:
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'self.df' (筹码分布) 为空。")
        if is_probe_date and pd.isna(close_price):
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'close_price' 缺失。")
        if is_probe_date and pd.isna(pre_close):
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'pre_close' 缺失。")
        if is_probe_date and pd.isna(total_chip_volume):
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'total_chip_volume' 缺失。")
        if is_probe_date and total_chip_volume <= 0:
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'total_chip_volume' 为 {total_chip_volume}。")
        if is_probe_date and pd.isna(total_daily_vol):
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'total_daily_vol' 缺失。")
        if is_probe_date and total_daily_vol <= 0:
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'total_daily_vol' 为 {total_daily_vol}。")
        if not self.df.empty and pd.notna(close_price) and pd.notna(pre_close) and pd.notna(total_chip_volume) and total_chip_volume > 0 and pd.notna(total_daily_vol) and total_daily_vol > 0:
            critical_chips_df = self.df[(self.df['price'] > min(close_price, pre_close)) & (self.df['price'] < max(close_price, pre_close))]
            status_change_net_flow_pct = critical_chips_df['percent'].sum() * np.sign(close_price - pre_close)
            turnover_rate = total_daily_vol / total_chip_volume
            if is_probe_date and turnover_rate <= 0:
                print(f"    -> [跨日资金流探针] @ {trade_date}: 'turnover_rate' 为 {turnover_rate}。")
            if turnover_rate > 0: results['winner_loser_momentum'] = status_change_net_flow_pct / turnover_rate
            else: results['winner_loser_momentum'] = np.nan
        else:
            results['winner_loser_momentum'] = np.nan
        prev_fatigue_index = context.get('prev_chip_fatigue_index', np.nan)
        recent_closes = context.get('recent_10d_closes')
        if is_probe_date and pd.isna(prev_fatigue_index):
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'prev_chip_fatigue_index' 缺失。")
        if is_probe_date and pd.isna(close_price):
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'close_price' 缺失。")
        if is_probe_date and pd.isna(total_chip_volume):
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'total_chip_volume' 缺失。")
        if is_probe_date and total_chip_volume <= 0:
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'total_chip_volume' 为 {total_chip_volume}。")
        if is_probe_date and pd.isna(total_daily_vol):
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'total_daily_vol' 缺失。")
        if is_probe_date and recent_closes is None:
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'recent_10d_closes' 为 None。")
        # 修改行：增加探针，打印 recent_closes 的长度
        if is_probe_date and recent_closes is not None:
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'recent_10d_closes' 长度: {len(recent_closes)}。")
        # 修改行：调整 chip_fatigue_index 的计算逻辑
        if pd.notna(close_price) and pd.notna(total_chip_volume) and total_chip_volume > 0 and pd.notna(total_daily_vol):
            # 如果 prev_fatigue_index 为 NaN，或者 recent_closes 长度不足以进行有效日判断，则进行初始化
            # prev_fatigue_index 在首次计算时会是 0.0 (来自 .get(..., 0.0))，但 recent_closes 长度会不足
            if pd.isna(prev_fatigue_index) or recent_closes is None or len(recent_closes) < 9:
                # 初始化 fatigue_index 基于当日换手率
                fatigue_index = (total_daily_vol / total_chip_volume) * 100
                results['chip_fatigue_index'] = fatigue_index
                if is_probe_date:
                    print(f"    -> [跨日资金流探针] @ {trade_date}: 'chip_fatigue_index' 初始化为 {fatigue_index:.4f} (因 prev_fatigue_index 缺失或 recent_closes 不足)。")
            else:
                # 正常计算
                is_effective_day = (close_price >= max(recent_closes)) or (close_price <= min(recent_closes))
                fatigue_index = prev_fatigue_index * 0.98
                if is_effective_day: fatigue_index *= 0.5
                else: fatigue_index += (total_daily_vol / total_chip_volume) * 100
                results['chip_fatigue_index'] = fatigue_index
                if is_probe_date:
                    print(f"    -> [跨日资金流探针] @ {trade_date}: 'chip_fatigue_index' 正常计算为 {fatigue_index:.4f}。")
        else:
            results['chip_fatigue_index'] = np.nan
            if is_probe_date:
                print(f"    -> [跨日资金流探针] @ {trade_date}: 'chip_fatigue_index' 设为 NaN (因核心依赖缺失)。")
        prev_chip_dist = context.get('prev_chip_distribution')
        prev_dominant_peak_cost = context.get('prev_dominant_peak_cost')
        today_dominant_peak_cost = context.get('dominant_peak_cost')
        if is_probe_date and prev_chip_dist is None:
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'prev_chip_distribution' 为 None。")
        if is_probe_date and prev_chip_dist.empty:
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'prev_chip_distribution' 为空。")
        if is_probe_date and pd.isna(prev_dominant_peak_cost):
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'prev_dominant_peak_cost' 缺失。")
        if is_probe_date and pd.isna(today_dominant_peak_cost):
            print(f"    -> [跨日资金流探针] @ {trade_date}: 'today_dominant_peak_cost' 缺失。")
        if prev_chip_dist is not None and not prev_chip_dist.empty and pd.notna(prev_dominant_peak_cost) and pd.notna(today_dominant_peak_cost):
            prev_upper_shoulder_df = prev_chip_dist[(prev_chip_dist['price'] >= prev_dominant_peak_cost * 1.02) & (prev_chip_dist['price'] <= prev_dominant_peak_cost * 1.07)]
            prev_lower_shoulder_df = prev_chip_dist[(prev_chip_dist['price'] >= prev_dominant_peak_cost * 0.93) & (prev_chip_dist['price'] <= prev_dominant_peak_cost * 0.98)]
            today_upper_shoulder_df = self.df[(self.df['price'] >= today_dominant_peak_cost * 1.02) & (self.df['price'] <= today_dominant_peak_cost * 1.07)]
            today_lower_shoulder_df = self.df[(self.df['price'] >= today_dominant_peak_cost * 0.93) & (self.df['price'] <= today_dominant_peak_cost * 0.98)]
            chip_vol_upper_yesterday = prev_upper_shoulder_df['percent'].sum()
            chip_vol_lower_yesterday = prev_lower_shoulder_df['percent'].sum()
            chip_vol_upper_today = today_upper_shoulder_df['percent'].sum()
            chip_vol_lower_today = today_lower_shoulder_df['percent'].sum()
            upper_growth = (chip_vol_upper_today / chip_vol_upper_yesterday - 1) * 100 if chip_vol_upper_yesterday > 0 else (100 if chip_vol_upper_today > 0 else 0)
            lower_growth = (chip_vol_lower_today / chip_vol_lower_yesterday - 1) * 100 if chip_vol_lower_yesterday > 0 else (100 if chip_vol_lower_today > 0 else 0)
            results['peak_shoulder_growth_rate'] = upper_growth - lower_growth
        else:
            results['peak_shoulder_growth_rate'] = np.nan
        return results

    def _compute_game_theoretic_metrics(self, context: dict) -> dict:
        import datetime
        results = {}
        trade_date = context.get('trade_date', 'UNKNOWN')
        debug_params = context.get('debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        is_probe_date = False
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0]).date()
            if probe_date_naive == trade_date:
                is_probe_date = True
        intraday_df = context.get('processed_intraday_df')
        total_daily_vol = context.get('daily_turnover_volume')
        close_price = context.get('close_price')
        atr_14d = context.get('atr_14d')
        pre_close = context.get('pre_close')
        daily_high = context.get('high_price')
        daily_low = context.get('low_price')
        results['suppressive_accumulation_intensity'] = np.nan
        results['rally_distribution_intensity'] = np.nan
        results['rally_accumulation_intensity'] = np.nan
        results['panic_selling_intensity'] = np.nan
        results['active_winner_avg_cost'] = np.nan
        results['active_winner_profit_margin'] = np.nan
        results['loser_avg_cost'] = np.nan
        results['winner_conviction_index'] = np.nan
        results['main_force_cost_advantage'] = np.nan
        results['main_force_control_leverage'] = np.nan
        results['loser_capitulation_pressure_index'] = np.nan
        results['intraday_new_loser_pressure'] = np.nan
        results['auction_intent_signal'] = np.nan
        results['auction_closing_position'] = np.nan
        results['intraday_probe_rebound_quality'] = np.nan
        results['capitulation_absorption_index'] = np.nan
        results['peak_battle_intensity'] = np.nan
        results['peak_dynamic_strength_ratio'] = np.nan
        results['peak_main_force_premium'] = np.nan
        results['peak_mf_conviction_flow'] = np.nan
        if is_probe_date and intraday_df is None:
            print(f"    -> [博弈论探针] @ {trade_date}: 'intraday_df' 为 None。")
        if is_probe_date and intraday_df.empty:
            print(f"    -> [博弈论探针] @ {trade_date}: 'intraday_df' 为空。")
        if is_probe_date and pd.isna(total_daily_vol):
            print(f"    -> [博弈论探针] @ {trade_date}: 'total_daily_vol' 缺失。")
        if is_probe_date and total_daily_vol <= 0:
            print(f"    -> [博弈论探针] @ {trade_date}: 'total_daily_vol' 为 {total_daily_vol}。")
        required_cols_1 = ['minute_vwap', 'main_force_buy_vol', 'main_force_sell_vol']
        if not intraday_df.empty and pd.notna(total_daily_vol) and total_daily_vol > 0:
            for col in required_cols_1:
                if col not in intraday_df.columns:
                    if is_probe_date:
                        print(f"    -> [博弈论探针] @ {trade_date}: 'intraday_df' 缺少列 '{col}'。")
                    return results
            intraday_df['mf_net_vol'] = intraday_df['main_force_buy_vol'] - intraday_df['main_force_sell_vol']
            peaks, _ = find_peaks(intraday_df['minute_vwap'], prominence=0.001)
            troughs, _ = find_peaks(-intraday_df['minute_vwap'], prominence=0.001)
            turning_points = sorted(list(set(np.concatenate(([0], troughs, peaks, [len(intraday_df)-1])))))
            suppressive_vol, rally_dist_vol, rally_acc_vol, panic_vol = 0, 0, 0, 0
            for i in range(len(turning_points) - 1):
                window_df = intraday_df.iloc[turning_points[i]:turning_points[i+1]+1]
                if is_probe_date and window_df.empty:
                    print(f"    -> [博弈论探针] @ {trade_date}: 'window_df' 为空。")
                if window_df.empty: continue
                mf_net = window_df['mf_net_vol'].sum()
                if window_df['minute_vwap'].iloc[-1] < window_df['minute_vwap'].iloc[0]:
                    if mf_net > 0: suppressive_vol += mf_net
                    else: panic_vol += abs(mf_net)
                else:
                    if mf_net < 0: rally_dist_vol += abs(mf_net)
                    else: rally_acc_vol += mf_net
            results['suppressive_accumulation_intensity'] = (suppressive_vol / total_daily_vol) * 100
            results['rally_distribution_intensity'] = (rally_dist_vol / total_daily_vol) * 100
            results['rally_accumulation_intensity'] = (rally_acc_vol / total_daily_vol) * 100
            results['panic_selling_intensity'] = (panic_vol / total_daily_vol) * 100
        else:
            results['suppressive_accumulation_intensity'] = np.nan
            results['rally_distribution_intensity'] = np.nan
            results['rally_accumulation_intensity'] = np.nan
            results['panic_selling_intensity'] = np.nan
        if is_probe_date and pd.isna(close_price):
            print(f"    -> [博弈论探针] @ {trade_date}: 'close_price' 缺失。")
        if is_probe_date and pd.isna(atr_14d):
            print(f"    -> [博弈论探针] @ {trade_date}: 'atr_14d' 缺失。")
        if is_probe_date and atr_14d <= 0:
            print(f"    -> [博弈论探针] @ {trade_date}: 'atr_14d' 为 {atr_14d}。")
        active_winner_avg_cost, active_profit_margin = self._calculate_active_winner_profit_margin(close_price, atr_14d, context)
        results['active_winner_avg_cost'] = active_winner_avg_cost
        results['active_winner_profit_margin'] = active_profit_margin
        losers_df = self.df[self.df['price'] > close_price]
        if is_probe_date and losers_df.empty:
            print(f"    -> [博弈论探针] @ {trade_date}: 'losers_df' 为空。")
        if is_probe_date and losers_df['percent'].sum() <= 0:
            print(f"    -> [博弈论探针] @ {trade_date}: 'losers_df' 的 'percent' 总和为 {losers_df['percent'].sum()}。")
        if not losers_df.empty and losers_df['percent'].sum() > 0:
            results['loser_avg_cost'] = np.average(losers_df['price'], weights=losers_df['percent'])
        else:
            results['loser_avg_cost'] = np.nan
        results['winner_conviction_index'] = self._calculate_winner_conviction_index(context, active_profit_margin)
        required_cols_4_1 = ['minute_vwap', 'main_force_buy_vol', 'main_force_sell_vol', 'retail_buy_vol', 'retail_sell_vol']
        if not intraday_df.empty:
            for col in required_cols_4_1:
                if col not in intraday_df.columns:
                    if is_probe_date:
                        print(f"    -> [博弈论探针] @ {trade_date}: 'intraday_df' 缺少列 '{col}'。")
                    return results
            mf_total_vol = intraday_df['main_force_buy_vol'].sum() + intraday_df['main_force_sell_vol'].sum()
            retail_total_vol = intraday_df['retail_buy_vol'].sum() + intraday_df['retail_sell_vol'].sum()
            if is_probe_date and mf_total_vol <= 0:
                print(f"    -> [博弈论探针] @ {trade_date}: 'mf_total_vol' 为 {mf_total_vol}。")
            if is_probe_date and retail_total_vol <= 0:
                print(f"    -> [博弈论探针] @ {trade_date}: 'retail_total_vol' 为 {retail_total_vol}。")
            if mf_total_vol > 0 and retail_total_vol > 0:
                mf_total_amount = (intraday_df['main_force_buy_vol'] * intraday_df['minute_vwap']).sum() + (intraday_df['main_force_sell_vol'] * intraday_df['minute_vwap']).sum()
                retail_total_amount = (intraday_df['retail_buy_vol'] * intraday_df['minute_vwap']).sum() + (intraday_df['retail_sell_vol'] * intraday_df['minute_vwap']).sum()
                vwap_mf = mf_total_amount / mf_total_vol
                vwap_retail = retail_total_amount / retail_total_vol
                if is_probe_date and vwap_mf <= 0:
                    print(f"    -> [博弈论探针] @ {trade_date}: 'vwap_mf' 为 {vwap_mf}。")
                if vwap_mf > 0: results['main_force_cost_advantage'] = (vwap_retail / vwap_mf - 1) * 100
                else: results['main_force_cost_advantage'] = np.nan
            else:
                results['main_force_cost_advantage'] = np.nan
        else:
            results['main_force_cost_advantage'] = np.nan
        required_cols_4_2 = ['main_force_buy_vol', 'main_force_sell_vol']
        if not intraday_df.empty:
            for col in required_cols_4_2:
                if col not in intraday_df.columns:
                    if is_probe_date:
                        print(f"    -> [博弈论探针] @ {trade_date}: 'intraday_df' 缺少列 '{col}'。")
                    return results
            mf_buy_vol = intraday_df['main_force_buy_vol'].sum()
            mf_sell_vol = intraday_df['main_force_sell_vol'].sum()
            if is_probe_date and (mf_buy_vol + mf_sell_vol) <= 0:
                print(f"    -> [博弈论探针] @ {trade_date}: 'mf_buy_vol + mf_sell_vol' 为 {(mf_buy_vol + mf_sell_vol)}。")
            if (mf_buy_vol + mf_sell_vol) > 0:
                results['main_force_control_leverage'] = ((mf_buy_vol - mf_sell_vol) / (mf_buy_vol + mf_sell_vol)) * 100
            else:
                results['main_force_control_leverage'] = np.nan
        else:
            results['main_force_control_leverage'] = np.nan
        loser_loss_margin = context.get('loser_loss_margin_avg')
        loser_concentration = context.get('loser_concentration_90pct')
        if is_probe_date and pd.isna(loser_loss_margin):
            print(f"    -> [博弈论探针] @ {trade_date}: 'loser_loss_margin_avg' 缺失。")
        if is_probe_date and pd.isna(loser_concentration):
            print(f"    -> [博弈论探针] @ {trade_date}: 'loser_concentration_90pct' 缺失。")
        if pd.notna(loser_loss_margin) and pd.notna(loser_concentration):
            results['loser_capitulation_pressure_index'] = abs(loser_loss_margin) * loser_concentration
        else:
            results['loser_capitulation_pressure_index'] = np.nan
        daily_volume = context.get('daily_turnover_volume')
        if not intraday_df.empty and pd.notna(close_price) and pd.notna(daily_volume) and daily_volume > 0:
            new_losers_df = intraday_df[intraday_df['minute_vwap'] > close_price].copy()
            if is_probe_date and new_losers_df.empty:
                print(f"    -> [博弈论探针] @ {trade_date}: 'new_losers_df' 为空。")
            if not new_losers_df.empty:
                new_loser_vol = new_losers_df['vol_shares'].sum()
                if is_probe_date and new_loser_vol <= 0:
                    print(f"    -> [博弈论探针] @ {trade_date}: 'new_loser_vol' 为 {new_loser_vol}。")
                if new_loser_vol > 0:
                    new_loser_vwap = (new_losers_df['minute_vwap'] * new_losers_df['vol_shares']).sum() / new_loser_vol
                    avg_loss_rate = abs((close_price / new_loser_vwap - 1))
                    results['intraday_new_loser_pressure'] = (new_loser_vol / daily_volume) * avg_loss_rate * 100
                else:
                    results['intraday_new_loser_pressure'] = np.nan
            else:
                results['intraday_new_loser_pressure'] = np.nan
        else:
            results['intraday_new_loser_pressure'] = np.nan
        results['auction_intent_signal'] = np.nan
        results['auction_closing_position'] = np.nan
        # 修改行：调整竞价信号的计算逻辑
        if not intraday_df.empty and 'trade_time' in intraday_df.columns and pd.notna(close_price) and pd.notna(atr_14d) and atr_14d > 0:
            closing_auction_time = datetime.time(15, 0) # 修改行：明确指定收盘竞价时间为 15:00
            pre_auction_df = intraday_df[intraday_df['trade_time'].dt.time < closing_auction_time] # 修改行：竞价前数据
            auction_bar_df = intraday_df[intraday_df['trade_time'].dt.time == closing_auction_time] # 修改行：15:00 竞价数据
            if is_probe_date and pre_auction_df.empty:
                print(f"    -> [博弈论探针] @ {trade_date}: 'pre_auction_df' 为空。")
            if is_probe_date and auction_bar_df.empty:
                print(f"    -> [博弈论探针] @ {trade_date}: 'auction_bar_df' 为空 (15:00 bar)。")
            if not pre_auction_df.empty and not auction_bar_df.empty and 'vol' in auction_bar_df.columns and auction_bar_df['vol'].sum() > 0:
                pre_auction_close = pre_auction_df['close'].iloc[-1] # 15:00 前最后一分钟的收盘价
                auction_bar = auction_bar_df.iloc[0] # 15:00 的分钟数据
                auction_volume = auction_bar['vol']
                if is_probe_date and total_daily_vol <= 0:
                    print(f"    -> [博弈论探针] @ {trade_date}: 'total_daily_vol' 为 {total_daily_vol}。")
                if total_daily_vol > 0:
                    price_impact = (close_price - pre_auction_close) / atr_14d
                    volume_weight = np.log1p(auction_volume / total_daily_vol)
                    results['auction_intent_signal'] = price_impact * volume_weight * 100
                else:
                    results['auction_intent_signal'] = np.nan
                auction_high = auction_bar['high']
                auction_low = auction_bar['low']
                auction_range = auction_high - auction_low
                if auction_range > 1e-6:
                    position = (close_price - auction_low) / auction_range
                    results['auction_closing_position'] = (position * 2 - 1) * 100
                elif close_price >= auction_high:
                    results['auction_closing_position'] = 100.0
                elif close_price <= auction_low: # 修改行：增加当收盘价等于最低价时的处理
                    results['auction_closing_position'] = -100.0
                else: # 修改行：处理其他零范围情况
                    results['auction_closing_position'] = np.nan
            else:
                results['auction_intent_signal'] = np.nan
                results['auction_closing_position'] = np.nan
        else:
            results['auction_intent_signal'] = np.nan
            results['auction_closing_position'] = np.nan
        if is_probe_date and intraday_df is None:
            print(f"    -> [博弈论探针] @ {trade_date}: 'intraday_df' 为 None。")
        if is_probe_date and intraday_df.empty:
            print(f"    -> [博弈论探针] @ {trade_date}: 'intraday_df' 为空。")
        if is_probe_date and pd.isna(daily_low):
            print(f"    -> [博弈论探针] @ {trade_date}: 'daily_low' 缺失。")
        if is_probe_date and pd.isna(daily_high):
            print(f"    -> [博弈论探针] @ {trade_date}: 'daily_high' 缺失。")
        if is_probe_date and pd.isna(close_price):
            print(f"    -> [博弈论探针] @ {trade_date}: 'close_price' 缺失。")
        if not intraday_df.empty and pd.notna(daily_low) and pd.notna(daily_high) and pd.notna(close_price):
            day_range = daily_high - daily_low
            if is_probe_date and day_range <= 1e-6:
                print(f"    -> [博弈论探针] @ {trade_date}: 'day_range' 为 {day_range}。")
            if day_range > 1e-6:
                if 'low' in intraday_df.columns and not intraday_df['low'].empty:
                    low_price_idx = intraday_df['low'].idxmin()
                else:
                    low_price_idx = intraday_df['minute_vwap'].idxmin()
                rebound_df = intraday_df.loc[low_price_idx:]
                if is_probe_date and rebound_df.empty:
                    print(f"    -> [博弈论探针] @ {trade_date}: 'rebound_df' 为空。")
                if is_probe_date and len(rebound_df) <= 1:
                    print(f"    -> [博弈论探针] @ {trade_date}: 'rebound_df' 长度不足2。")
                if not rebound_df.empty and len(rebound_df) > 1:
                    price_recovery_ratio = (close_price - daily_low) / day_range
                    rebound_volume = rebound_df['vol_shares'].sum()
                    if is_probe_date and rebound_volume <= 0:
                        print(f"    -> [博弈论探针] @ {trade_date}: 'rebound_volume' 为 {rebound_volume}。")
                    if rebound_volume > 0 and 'main_force_net_vol' in rebound_df.columns:
                        rebound_mf_net_flow = rebound_df['main_force_net_vol'].sum()
                        rebound_purity = rebound_mf_net_flow / rebound_volume
                        results['intraday_probe_rebound_quality'] = price_recovery_ratio * rebound_purity * 100
                    else:
                        results['intraday_probe_rebound_quality'] = np.nan
                else:
                    results['intraday_probe_rebound_quality'] = np.nan
            else:
                results['intraday_probe_rebound_quality'] = np.nan
        else:
            results['intraday_probe_rebound_quality'] = np.nan
        if not intraday_df.empty and pd.notna(daily_low) and pd.notna(daily_high) and pd.notna(close_price) and pd.notna(atr_14d) and atr_14d > 0:
            panic_selling_zone_low = daily_low
            panic_selling_zone_high = daily_low + atr_14d * 0.5
            panic_zone_df = intraday_df[(intraday_df['minute_vwap'] >= panic_selling_zone_low) & (intraday_df['minute_vwap'] <= panic_selling_zone_high)]
            if is_probe_date and panic_zone_df.empty:
                print(f"    -> [博弈论探针] @ {trade_date}: 'panic_zone_df' 为空。")
            if not panic_zone_df.empty and 'main_force_net_vol' in panic_zone_df.columns and 'retail_net_vol' in panic_zone_df.columns and 'vol_shares' in panic_zone_df.columns:
                mf_net_flow_panic_zone = panic_zone_df['main_force_net_vol'].sum()
                retail_net_flow_panic_zone = panic_zone_df['retail_net_vol'].sum()
                total_vol_panic_zone = panic_zone_df['vol_shares'].sum()
                if is_probe_date and total_vol_panic_zone <= 0:
                    print(f"    -> [博弈论探针] @ {trade_date}: 'total_vol_panic_zone' 为 {total_vol_panic_zone}。")
                if total_vol_panic_zone > 0:
                    if mf_net_flow_panic_zone > 0 and retail_net_flow_panic_zone < 0:
                        results['capitulation_absorption_index'] = (mf_net_flow_panic_zone / (mf_net_flow_panic_zone + abs(retail_net_flow_panic_zone))) * 100
                    else:
                        results['capitulation_absorption_index'] = 0.0
                else:
                    results['capitulation_absorption_index'] = np.nan
            else:
                results['capitulation_absorption_index'] = np.nan
        else:
            results['capitulation_absorption_index'] = np.nan
        dominant_peak_cost = context.get('dominant_peak_cost')
        dominant_peak_volume_ratio = context.get('dominant_peak_volume_ratio')
        if is_probe_date and pd.isna(dominant_peak_cost):
            print(f"    -> [博弈论探针] @ {trade_date}: 'dominant_peak_cost' 缺失。")
        if is_probe_date and pd.isna(dominant_peak_volume_ratio):
            print(f"    -> [博弈论探针] @ {trade_date}: 'dominant_peak_volume_ratio' 缺失。")
        if is_probe_date and pd.isna(atr_14d):
            print(f"    -> [博弈论探针] @ {trade_date}: 'atr_14d' 缺失。")
        if is_probe_date and atr_14d <= 0:
            print(f"    -> [博弈论探针] @ {trade_date}: 'atr_14d' 为 {atr_14d}。")
        if is_probe_date and intraday_df.empty:
            print(f"    -> [博弈论探针] @ {trade_date}: 'intraday_df' 为空。")
        if pd.notna(dominant_peak_cost) and pd.notna(dominant_peak_volume_ratio) and pd.notna(atr_14d) and atr_14d > 0 and not intraday_df.empty:
            peak_zone_low = dominant_peak_cost - atr_14d * 0.2
            peak_zone_high = dominant_peak_cost + atr_14d * 0.2
            intraday_peak_zone_df = intraday_df[(intraday_df['minute_vwap'] >= peak_zone_low) & (intraday_df['minute_vwap'] <= peak_zone_high)]
            if is_probe_date and intraday_peak_zone_df.empty:
                print(f"    -> [博弈论探针] @ {trade_date}: 'intraday_peak_zone_df' 为空。")
            if not intraday_peak_zone_df.empty and 'main_force_net_vol' in intraday_peak_zone_df.columns and 'retail_net_vol' in intraday_peak_zone_df.columns and 'vol_shares' in intraday_peak_zone_df.columns:
                mf_net_flow_in_peak_zone = intraday_peak_zone_df['main_force_net_vol'].sum()
                retail_net_flow_in_peak_zone = intraday_peak_zone_df['retail_net_vol'].sum()
                total_vol_in_peak_zone = intraday_peak_zone_df['vol_shares'].sum()
                if is_probe_date and total_vol_in_peak_zone <= 0:
                    print(f"    -> [博弈论探针] @ {trade_date}: 'total_vol_in_peak_zone' 为 {total_vol_in_peak_zone}。")
                if total_vol_in_peak_zone > 0:
                    results['peak_battle_intensity'] = (total_vol_in_peak_zone / total_daily_vol) * 100 if total_daily_vol > 0 else np.nan
                    results['peak_dynamic_strength_ratio'] = (mf_net_flow_in_peak_zone / total_vol_in_peak_zone) * 100
                    avg_price_in_peak_zone = (intraday_peak_zone_df['minute_vwap'] * intraday_peak_zone_df['vol_shares']).sum() / total_vol_in_peak_zone
                    if is_probe_date and avg_price_in_peak_zone <= 0:
                        print(f"    -> [博弈论探针] @ {trade_date}: 'avg_price_in_peak_zone' 为 {avg_price_in_peak_zone}。")
                    if avg_price_in_peak_zone > 0:
                        results['peak_main_force_premium'] = (dominant_peak_cost / avg_price_in_peak_zone - 1) * 100
                    else:
                        results['peak_main_force_premium'] = np.nan
                    total_net_flow_in_peak_zone = abs(mf_net_flow_in_peak_zone) + abs(retail_net_flow_in_peak_zone)
                    if is_probe_date and total_net_flow_in_peak_zone <= 0:
                        print(f"    -> [博弈论探针] @ {trade_date}: 'total_net_flow_in_peak_zone' 为 {total_net_flow_in_peak_zone}。")
                    if total_net_flow_in_peak_zone > 0:
                        results['peak_mf_conviction_flow'] = (mf_net_flow_in_peak_zone / total_net_flow_in_peak_zone) * 100
                    else:
                        results['peak_mf_conviction_flow'] = np.nan
                else:
                    results['peak_battle_intensity'] = np.nan
                    results['peak_dynamic_strength_ratio'] = np.nan
                    results['peak_main_force_premium'] = np.nan
                    results['peak_mf_conviction_flow'] = np.nan
            else:
                results['peak_battle_intensity'] = np.nan
                results['peak_dynamic_strength_ratio'] = np.nan
                results['peak_main_force_premium'] = np.nan
                results['peak_mf_conviction_flow'] = np.nan
        else:
            results['peak_battle_intensity'] = np.nan
            results['peak_dynamic_strength_ratio'] = np.nan
            results['peak_main_force_premium'] = np.nan
            results['peak_mf_conviction_flow'] = np.nan
        return results

    def _compute_vital_sign_metrics(self, context: dict) -> dict:
        """
        【V3.3 · 共识度鲁棒性增强版】
        - 核心修复: 增强 `structural_consensus_score` 的计算鲁棒性，当上游集中度指标为空时，采用中性值替代。
        """
        results = {}
        # 1. 成本结构共识与筹码成本动量
        concentration_90pct = context.get('concentration_90pct')
        total_winner_rate = context.get('total_winner_rate')
        # 增强鲁棒性：如果集中度指标为空，则赋予一个中性值，而不是让计算失败
        if pd.notna(total_winner_rate):
            if pd.notna(concentration_90pct):
                concentration_factor = 1.0 - np.clip(concentration_90pct, 0, 1)
            else:
                # 当上游集中度为空时（例如市场极端行情），赋予一个中性值0.5
                concentration_factor = 0.5
            profit_factor = total_winner_rate / 100.0
            results['structural_consensus_score'] = concentration_factor * profit_factor * 100
        # 升级为 `dominant_cost_momentum`
        dominant_peak_cost = context.get('dominant_peak_cost')
        prev_dominant_peak_cost = context.get('prev_dominant_peak_cost')
        prev_atr = context.get('prev_atr_14d')
        if all(pd.notna(v) for v in [dominant_peak_cost, prev_dominant_peak_cost, prev_atr]) and prev_atr > 0:
            results['dominant_cost_momentum'] = (dominant_peak_cost - prev_dominant_peak_cost) / prev_atr
        # 2. 结构韧性指数
        def normalize(value, default=0.0): return value if pd.notna(value) else default
        consensus_index = normalize(results.get('structural_consensus_score'))
        peak_profit_margin = normalize(context.get('dominant_peak_profit_margin'))
        foundation_score = np.log1p(consensus_index) * np.log1p(np.maximum(0, peak_profit_margin))
        active_winner_margin = normalize(context.get('active_winner_profit_margin'))
        winner_conviction = normalize(context.get('winner_conviction_index'))
        pressure_score = np.log1p(np.maximum(0, active_winner_margin)) * np.log1p(np.maximum(0, winner_conviction))
        cost_momentum = normalize(results.get('dominant_cost_momentum'))
        upward_purity = normalize(context.get('upward_impulse_purity'))
        reinforcement_score = (np.tanh(cost_momentum) + 1) * ((upward_purity / 100.0) + 1)
        if foundation_score >= 0 and pressure_score >= 0 and reinforcement_score >= 0:
            resilience_raw = (foundation_score * pressure_score * reinforcement_score) ** (1.0 / 3.0)
            results['structural_resilience_index'] = np.clip(resilience_raw * 20, 0, 100)
        # 3. 主力姿态坐标
        resilience = normalize(results.get('structural_resilience_index')) / 100.0
        cost_advantage = np.tanh(normalize(context.get('main_force_cost_advantage')) / 10.0)
        loser_pressure = normalize(context.get('loser_capitulation_pressure_index'))
        pressure_penalty = np.tanh(loser_pressure / 50.0)
        results['posture_control_score'] = (resilience + cost_advantage) / 2.0 * (1 - pressure_penalty) * 100
        control_leverage = normalize(context.get('main_force_control_leverage')) / 100.0
        buy_intensity = normalize(context.get('suppressive_accumulation_intensity')) + normalize(context.get('rally_accumulation_intensity'))
        sell_intensity = normalize(context.get('rally_distribution_intensity')) + normalize(context.get('panic_selling_intensity'))
        tactical_factor = np.tanh((buy_intensity - sell_intensity) / 50.0)
        results['posture_action_score'] = (0.7 * control_leverage + 0.3 * tactical_factor) * 100
        return results

    def _compute_static_structure_metrics(self) -> dict:
        results = {}
        trade_date = self.ctx.get('trade_date', 'UNKNOWN')
        debug_params = self.ctx.get('debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        is_probe_date = False
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0]).date()
            if probe_date_naive == trade_date:
                is_probe_date = True
        close_price = self.ctx.get('close_price')
        atr_14d = self.ctx.get('atr_14d')
        intraday_df = self.ctx.get('processed_intraday_df')
        results.update({
            'dominant_peak_cost': np.nan, 'dominant_peak_volume_ratio': np.nan,
            'secondary_peak_cost': np.nan, 'peak_separation_ratio': np.nan,
            'peak_volume_ratio': np.nan, 'peak_distance_volatility_ratio': np.nan,
            'peak_separation_intensity': np.nan, 'peak_fusion_indicator': np.nan,
            'dominant_peak_profit_margin': np.nan, 'dominant_peak_solidity': np.nan,
            'winner_concentration_90pct': np.nan, 'loser_concentration_90pct': np.nan,
            'dynamic_pressure_index': np.nan, 'dynamic_support_index': np.nan,
            'total_winner_rate': np.nan, 'total_loser_rate': np.nan,
            'winner_profit_margin_avg': np.nan, 'effective_winner_rate': np.nan,
            'loser_loss_margin_avg': np.nan, 'active_winner_rate': np.nan,
            'active_loser_rate': np.nan, 'locked_profit_rate': np.nan,
            'locked_loss_rate': np.nan, 'chip_fault_blockage_ratio': np.nan,
            'chip_fault_magnitude': np.nan, 'short_term_holder_cost': np.nan,
            'short_term_concentration_90pct': np.nan, 'long_term_holder_cost': np.nan,
            'long_term_concentration_90pct': np.nan, 'winner_profit_cushion': np.nan,
            'loser_pain_index': np.nan, 'cost_structure_skewness': np.nan,
            'structural_stability_score': np.nan, 'recent_trapped_pressure': np.nan,
            'imminent_profit_taking_supply': np.nan, 'price_volume_entropy': np.nan,
            'profit_realization_quality': np.nan, # 修改行：初始化为 np.nan，但实际计算已移走
            'active_buying_support': np.nan,
            'active_selling_pressure': np.nan,
        })
        # 1. 主导峰与潜在次峰引力点(PSGP)剖面
        if is_probe_date and self.df.empty:
            print(f"    -> [静态结构探针] @ {trade_date}: 'self.df' (筹码分布) 为空。")
        if not self.df.empty:
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(self.df['percent'], prominence=0.1, width=1)
            if is_probe_date and len(peaks) == 0:
                print(f"    -> [静态结构探针] @ {trade_date}: 未找到筹码峰。")
            if len(peaks) > 0:
                peaks_df = pd.DataFrame({
                    'peak_index': peaks, 'volume': self.df['percent'].iloc[peaks].values,
                    'cost': self.df['price'].iloc[peaks].values, 'prominence': properties['prominences'],
                    'left_ip': properties['left_ips'], 'right_ip': properties['right_ips'],
                }).sort_values(by='prominence', ascending=False).reset_index(drop=True)
                main_peak = peaks_df.iloc[0]
                main_peak_cost = main_peak['cost']
                main_peak_idx = int(main_peak['peak_index'])
                results['dominant_peak_cost'] = main_peak_cost
                results['dominant_peak_volume_ratio'] = main_peak['volume']
                results['peak_range_low'] = np.interp(main_peak['left_ip'], self.df.index, self.df['price'])
                results['peak_range_high'] = np.interp(main_peak['right_ip'], self.df.index, self.df['price'])
            else:
                main_peak_idx = self.df['percent'].idxmax()
                main_peak_cost = self.df.loc[main_peak_idx, 'price']
                results['dominant_peak_cost'] = main_peak_cost
                results['dominant_peak_volume_ratio'] = self.df.loc[main_peak_idx, 'percent']
                results['peak_range_low'] = main_peak_cost * 0.99
                results['peak_range_high'] = main_peak_cost * 1.01
            
            peak_width = max(1, int(len(self.df) * 0.05))
            exclusion_start = max(0, main_peak_idx - peak_width)
            exclusion_end = min(len(self.df), main_peak_idx + peak_width)
            remaining_chips_df = self.df.drop(self.df.index[exclusion_start:exclusion_end])
            if is_probe_date and remaining_chips_df.empty:
                print(f"    -> [静态结构探针] @ {trade_date}: 'remaining_chips_df' 为空。")
            if not remaining_chips_df.empty:
                psgp_idx = remaining_chips_df['percent'].idxmax()
                psgp_cost = remaining_chips_df.loc[psgp_idx, 'price']
                psgp_volume = remaining_chips_df.loc[psgp_idx, 'percent']
                results['secondary_peak_cost'] = psgp_cost
                if is_probe_date and pd.isna(main_peak_cost):
                    print(f"    -> [静态结构探针] @ {trade_date}: 'main_peak_cost' 缺失。")
                if is_probe_date and main_peak_cost <= 0:
                    print(f"    -> [静态结构探针] @ {trade_date}: 'main_peak_cost' 为 {main_peak_cost}。")
                if pd.notna(main_peak_cost) and main_peak_cost > 0:
                    separation = abs(main_peak_cost - psgp_cost) / main_peak_cost
                    results['peak_separation_ratio'] = separation * 100
                    valley_df = self.df.loc[min(main_peak_idx, psgp_idx):max(main_peak_idx, psgp_idx)]
                    valley_chip_percent = valley_df['percent'].sum() - results['dominant_peak_volume_ratio'] - psgp_volume
                    fusion_penalty = np.tanh(valley_chip_percent / 10)
                    results['peak_separation_intensity'] = results['peak_separation_ratio'] * (1 - fusion_penalty)
                    results['peak_fusion_indicator'] = (1 - separation) * (1 + fusion_penalty) * 100
                else:
                    results['peak_separation_ratio'] = np.nan
                    results['peak_separation_intensity'] = np.nan
                    results['peak_fusion_indicator'] = np.nan
                if is_probe_date and pd.isna(results['dominant_peak_volume_ratio']):
                    print(f"    -> [静态结构探针] @ {trade_date}: 'dominant_peak_volume_ratio' 缺失。")
                if is_probe_date and results['dominant_peak_volume_ratio'] <= 0:
                    print(f"    -> [静态结构探针] @ {trade_date}: 'dominant_peak_volume_ratio' 为 {results['dominant_peak_volume_ratio']}。")
                if pd.notna(results['dominant_peak_volume_ratio']) and results['dominant_peak_volume_ratio'] > 0:
                    results['peak_volume_ratio'] = (psgp_volume / results['dominant_peak_volume_ratio']) * 100
                else:
                    results['peak_volume_ratio'] = np.nan
                if is_probe_date and pd.isna(atr_14d):
                    print(f"    -> [静态结构探针] @ {trade_date}: 'atr_14d' 缺失。")
                if is_probe_date and atr_14d <= 0:
                    print(f"    -> [静态结构探针] @ {trade_date}: 'atr_14d' 为 {atr_14d}。")
                if pd.notna(atr_14d) and atr_14d > 0:
                    peak_distance = abs(main_peak_cost - psgp_cost)
                    results['peak_distance_volatility_ratio'] = peak_distance / atr_14d
                else:
                    results['peak_distance_volatility_ratio'] = np.nan
            else:
                results['secondary_peak_cost'] = np.nan
                results['peak_separation_ratio'] = np.nan
                results['peak_volume_ratio'] = np.nan
                results['peak_distance_volatility_ratio'] = np.nan
                results['peak_separation_intensity'] = np.nan
                results['peak_fusion_indicator'] = np.nan
        if is_probe_date and pd.isna(close_price):
            print(f"    -> [静态结构探针] @ {trade_date}: 'close_price' 缺失。")
        if is_probe_date and results.get('dominant_peak_cost', np.nan) <= 0:
            print(f"    -> [静态结构探针] @ {trade_date}: 'dominant_peak_cost' 为 {results.get('dominant_peak_cost', np.nan)}。")
        if pd.notna(close_price) and results.get('dominant_peak_cost', np.nan) > 0:
            results['dominant_peak_profit_margin'] = (close_price / results['dominant_peak_cost'] - 1) * 100
        else:
            results['dominant_peak_profit_margin'] = np.nan
        dominant_peak_cost = results.get('dominant_peak_cost')
        cost_95pct = self.ctx.get('cost_95pct')
        cost_5pct = self.ctx.get('cost_5pct')
        if is_probe_date and pd.isna(dominant_peak_cost):
            print(f"    -> [静态结构探针] @ {trade_date}: 'dominant_peak_cost' 缺失。")
        if is_probe_date and pd.isna(cost_95pct):
            print(f"    -> [静态结构探针] @ {trade_date}: 'cost_95pct' 缺失。")
        if is_probe_date and pd.isna(cost_5pct):
            print(f"    -> [静态结构探针] @ {trade_date}: 'cost_5pct' 缺失。")
        if is_probe_date and pd.isna(close_price):
            print(f"    -> [静态结构探针] @ {trade_date}: 'close_price' 缺失。")
        if is_probe_date and pd.notna(close_price) and close_price <= 0:
            print(f"    -> [静态结构探针] @ {trade_date}: 'close_price' 为 {close_price}。")
        if pd.notna(dominant_peak_cost) and pd.notna(cost_95pct) and pd.notna(cost_5pct) and pd.notna(close_price) and close_price > 0:
            peak_zone_low = dominant_peak_cost * 0.995
            peak_zone_high = dominant_peak_cost * 1.005
            volume_in_peak_zone = self.df[(self.df['price'] >= peak_zone_low) & (self.df['price'] <= peak_zone_high)]['percent'].sum()
            chip_width_90pct = cost_95pct - cost_5pct
            if is_probe_date and chip_width_90pct <= 0:
                print(f"    -> [静态结构探针] @ {trade_date}: 'chip_width_90pct' 为 {chip_width_90pct}。")
            if chip_width_90pct > 0:
                peak_top_density = volume_in_peak_zone / (peak_zone_high - peak_zone_low)
                normalized_chip_width = chip_width_90pct / close_price
                if is_probe_date and normalized_chip_width <= 0:
                    print(f"    -> [静态结构探针] @ {trade_date}: 'normalized_chip_width' 为 {normalized_chip_width}。")
                if normalized_chip_width > 0:
                    results['dominant_peak_solidity'] = peak_top_density / normalized_chip_width
                else:
                    results['dominant_peak_solidity'] = np.nan
            else:
                results['dominant_peak_solidity'] = np.nan
        else:
            results['dominant_peak_solidity'] = np.nan
        # 2. 集中度剖面
        def _get_concentration(chip_df: pd.DataFrame):
            if chip_df.empty or chip_df['percent'].sum() < 1e-6: return np.nan
            chip_df = chip_df.copy()
            chip_df['percent'] = (chip_df['percent'] / chip_df['percent'].sum()) * 100
            chip_df['cum_percent'] = chip_df['percent'].cumsum()
            avg_cost = np.average(chip_df['price'], weights=chip_df['percent'])
            if is_probe_date and avg_cost <= 0:
                print(f"    -> [静态结构探针] @ {trade_date}: '_get_concentration' 中 'avg_cost' 为 {avg_cost}。")
            if avg_cost <= 0: return np.nan
            price_low = np.interp(5, chip_df['cum_percent'], chip_df['price'])
            price_high = np.interp(95, chip_df['cum_percent'], chip_df['price'])
            return (price_high - price_low) / avg_cost
        if is_probe_date and pd.isna(close_price):
            print(f"    -> [静态结构探针] @ {trade_date}: 'close_price' 缺失。")
        if pd.notna(close_price):
            results['winner_concentration_90pct'] = _get_concentration(self.df[self.df['price'] < close_price])
            results['loser_concentration_90pct'] = _get_concentration(self.df[self.df['price'] > close_price])
        else:
            results['winner_concentration_90pct'] = np.nan
            results['loser_concentration_90pct'] = np.nan
        # 3. 动态压力支撑 (升级为力矩模型)
        if is_probe_date and pd.isna(close_price):
            print(f"    -> [静态结构探针] @ {trade_date}: 'close_price' 缺失。")
        if is_probe_date and pd.isna(atr_14d):
            print(f"    -> [静态结构探针] @ {trade_date}: 'atr_14d' 缺失。")
        if is_probe_date and atr_14d <= 0:
            print(f"    -> [静态结构探针] @ {trade_date}: 'atr_14d' 为 {atr_14d}。")
        if pd.notna(close_price) and pd.notna(atr_14d) and atr_14d > 0:
            zone_width = 0.5 * atr_14d
            pressure_df = self.df[(self.df['price'] > close_price) & (self.df['price'] <= close_price + zone_width)]
            if is_probe_date and pressure_df.empty:
                print(f"    -> [静态结构探针] @ {trade_date}: 'pressure_df' 为空。")
            if not pressure_df.empty:
                pressure_torque = ((pressure_df['price'] - close_price) / atr_14d * pressure_df['percent']).sum()
                results['dynamic_pressure_index'] = pressure_torque
            else:
                results['dynamic_pressure_index'] = np.nan
            support_df = self.df[(self.df['price'] >= close_price - zone_width) & (self.df['price'] < close_price)]
            if is_probe_date and support_df.empty:
                print(f"    -> [静态结构探针] @ {trade_date}: 'support_df' 为空。")
            if not support_df.empty:
                support_torque = ((close_price - support_df['price']) / atr_14d * support_df['percent']).sum()
                results['dynamic_support_index'] = support_torque
            else:
                results['dynamic_support_index'] = np.nan
        else:
            results['dynamic_pressure_index'] = np.nan
            results['dynamic_support_index'] = np.nan
        # 4. 盈亏结构
        if is_probe_date and pd.isna(close_price):
            print(f"    -> [静态结构探针] @ {trade_date}: 'close_price' 缺失。")
        if pd.notna(close_price):
            winners_df = self.df[self.df['price'] < close_price]
            losers_df = self.df[self.df['price'] > close_price]
            results['total_winner_rate'] = winners_df['percent'].sum()
            results['total_loser_rate'] = losers_df['percent'].sum()
            if is_probe_date and winners_df.empty:
                print(f"    -> [静态结构探针] @ {trade_date}: 'winners_df' 为空。")
            if is_probe_date and winners_df['percent'].sum() <= 0:
                print(f"    -> [静态结构探针] @ {trade_date}: 'winners_df' 的 'percent' 总和为 {winners_df['percent'].sum()}。")
            if not winners_df.empty and winners_df['percent'].sum() > 0:
                winner_avg_cost = np.average(winners_df['price'], weights=winners_df['percent'])
                if is_probe_date and winner_avg_cost <= 0:
                    print(f"    -> [静态结构探针] @ {trade_date}: 'winner_avg_cost' 为 {winner_avg_cost}。")
                if winner_avg_cost > 0:
                    results['winner_profit_margin_avg'] = (close_price / winner_avg_cost - 1) * 100
                    if is_probe_date and pd.isna(results['total_winner_rate']):
                        print(f"    -> [静态结构探针] @ {trade_date}: 'total_winner_rate' 缺失。")
                    if is_probe_date and pd.isna(results['winner_profit_margin_avg']):
                        print(f"    -> [静态结构探针] @ {trade_date}: 'winner_profit_margin_avg' 缺失。")
                    if pd.notna(results['total_winner_rate']) and pd.notna(results['winner_profit_margin_avg']):
                        results['effective_winner_rate'] = results['total_winner_rate'] * results['winner_profit_margin_avg']
                    else:
                        results['effective_winner_rate'] = np.nan
                else:
                    results['winner_profit_margin_avg'] = np.nan
                    results['effective_winner_rate'] = np.nan
            else:
                results['winner_profit_margin_avg'] = np.nan
                results['effective_winner_rate'] = np.nan
            if is_probe_date and losers_df.empty:
                print(f"    -> [静态结构探针] @ {trade_date}: 'losers_df' 为空。")
            if is_probe_date and losers_df['percent'].sum() <= 0:
                print(f"    -> [静态结构探针] @ {trade_date}: 'losers_df' 的 'percent' 总和为 {losers_df['percent'].sum()}。")
            if not losers_df.empty and losers_df['percent'].sum() > 0:
                loser_avg_cost = np.average(losers_df['price'], weights=losers_df['percent'])
                if is_probe_date and loser_avg_cost <= 0:
                    print(f"    -> [静态结构探针] @ {trade_date}: 'loser_avg_cost' 为 {loser_avg_cost}。")
                if loser_avg_cost > 0: results['loser_loss_margin_avg'] = (close_price / loser_avg_cost - 1) * 100
                else: results['loser_loss_margin_avg'] = np.nan
            else:
                results['loser_loss_margin_avg'] = np.nan
            if is_probe_date and pd.isna(atr_14d):
                print(f"    -> [静态结构探针] @ {trade_date}: 'atr_14d' 缺失。")
            if is_probe_date and atr_14d <= 0:
                print(f"    -> [静态结构探针] @ {trade_date}: 'atr_14d' 为 {atr_14d}。")
            if pd.notna(atr_14d) and atr_14d > 0:
                active_winner_mask = (self.df['price'] < close_price) & (self.df['price'] >= close_price - 2 * atr_14d)
                results['active_winner_rate'] = self.df[active_winner_mask]['percent'].sum()
                active_loser_mask = (self.df['price'] > close_price) & (self.df['price'] <= close_price + 2 * atr_14d)
                results['active_loser_rate'] = self.df[active_loser_mask]['percent'].sum()
            else:
                results['active_winner_rate'] = np.nan
                results['active_loser_rate'] = np.nan
            locked_profit_mask = (self.df['price'] < close_price) & (~active_winner_mask if pd.notna(atr_14d) else (self.df['price'] < close_price))
            results['locked_profit_rate'] = self.df[locked_profit_mask]['percent'].sum()
            locked_loss_mask = (self.df['price'] > close_price) & (~active_loser_mask if pd.notna(atr_14d) else (self.df['price'] > close_price))
            results['locked_loss_rate'] = self.df[locked_loss_mask]['percent'].sum()
        else:
            results['total_winner_rate'] = np.nan
            results['total_loser_rate'] = np.nan
            results['winner_profit_margin_avg'] = np.nan
            results['effective_winner_rate'] = np.nan
            results['loser_loss_margin_avg'] = np.nan
            results['active_winner_rate'] = np.nan
            results['active_loser_rate'] = np.nan
            results['locked_profit_rate'] = np.nan
            results['locked_loss_rate'] = np.nan
        # 5. 断层动态 (鲁棒性优化)
        peak_cost = results.get('dominant_peak_cost')
        results['chip_fault_blockage_ratio'] = np.nan
        if is_probe_date and pd.isna(peak_cost):
            print(f"    -> [静态结构探针] @ {trade_date}: 'dominant_peak_cost' 缺失。")
        if is_probe_date and pd.isna(close_price):
            print(f"    -> [静态结构探针] @ {trade_date}: 'close_price' 缺失。")
        if is_probe_date and pd.isna(atr_14d):
            print(f"    -> [静态结构探针] @ {trade_date}: 'atr_14d' 缺失。")
        if is_probe_date and atr_14d <= 0:
            print(f"    -> [静态结构探针] @ {trade_date}: 'atr_14d' 为 {atr_14d}。")
        if pd.notna(peak_cost) and pd.notna(close_price) and pd.notna(atr_14d) and atr_14d > 0:
            magnitude = (close_price - peak_cost) / atr_14d
            results['chip_fault_magnitude'] = magnitude
            fault_zone_low, fault_zone_high = sorted([peak_cost, close_price])
            fault_zone_df = self.df[(self.df['price'] > fault_zone_low) & (self.df['price'] < fault_zone_high)]
            if is_probe_date and fault_zone_df.empty:
                print(f"    -> [静态结构探针] @ {trade_date}: 'fault_zone_df' 为空。")
            if not fault_zone_df.empty:
                results['chip_fault_blockage_ratio'] = fault_zone_df['percent'].sum()
            else:
                results['chip_fault_blockage_ratio'] = np.nan
        else:
            results['chip_fault_magnitude'] = np.nan
            results['chip_fault_blockage_ratio'] = np.nan
        # 6. 分层成本
        high_20d = self.ctx.get('high_20d')
        low_20d = self.ctx.get('low_20d')
        if is_probe_date and pd.isna(high_20d):
            print(f"    -> [静态结构探针] @ {trade_date}: 'high_20d' 缺失。")
        if is_probe_date and pd.isna(low_20d):
            print(f"    -> [静态结构探针] @ {trade_date}: 'low_20d' 缺失。")
        if pd.notna(high_20d) and pd.notna(low_20d):
            active_zone_mask = (self.df['price'] >= low_20d) & (self.df['price'] <= high_20d)
            short_term_chips_df = self.df[active_zone_mask]
            long_term_chips_df = self.df[~active_zone_mask]
            if is_probe_date and short_term_chips_df.empty:
                print(f"    -> [静态结构探针] @ {trade_date}: 'short_term_chips_df' 为空。")
            if is_probe_date and short_term_chips_df['percent'].sum() <= 0:
                print(f"    -> [静态结构探针] @ {trade_date}: 'short_term_chips_df' 的 'percent' 总和为 {short_term_chips_df['percent'].sum()}。")
            if not short_term_chips_df.empty and short_term_chips_df['percent'].sum() > 0:
                results['short_term_holder_cost'] = np.average(short_term_chips_df['price'], weights=short_term_chips_df['percent'])
                results['short_term_concentration_90pct'] = _get_concentration(short_term_chips_df)
            else:
                results['short_term_holder_cost'] = np.nan
                results['short_term_concentration_90pct'] = np.nan
            if is_probe_date and long_term_chips_df.empty:
                print(f"    -> [静态结构探针] @ {trade_date}: 'long_term_chips_df' 为空。")
            if is_probe_date and long_term_chips_df['percent'].sum() <= 0:
                print(f"    -> [静态结构探针] @ {trade_date}: 'long_term_chips_df' 的 'percent' 总和为 {long_term_chips_df['percent'].sum()}。")
            if not long_term_chips_df.empty and long_term_chips_df['percent'].sum() > 0:
                results['long_term_holder_cost'] = np.average(long_term_chips_df['price'], weights=long_term_chips_df['percent'])
                results['long_term_concentration_90pct'] = _get_concentration(long_term_chips_df)
            else:
                results['long_term_holder_cost'] = np.nan
                results['long_term_concentration_90pct'] = np.nan
        else:
            results['short_term_holder_cost'] = np.nan
            results['short_term_concentration_90pct'] = np.nan
            results['long_term_holder_cost'] = np.nan
            results['long_term_concentration_90pct'] = np.nan
        # 7. 盈利亏损质量分析
        if is_probe_date and self.df.empty:
            print(f"    -> [静态结构探针] @ {trade_date}: 'self.df' (筹码分布) 为空。")
        if is_probe_date and pd.isna(close_price):
            print(f"    -> [静态结构探针] @ {trade_date}: 'close_price' 缺失。")
        if not self.df.empty and pd.notna(close_price):
            winners_df_quality = self.df[self.df['price'] < close_price].copy()
            if is_probe_date and winners_df_quality.empty:
                print(f"    -> [静态结构探针] @ {trade_date}: 'winners_df_quality' 为空。")
            if is_probe_date and winners_df_quality['percent'].sum() <= 0:
                print(f"    -> [静态结构探针] @ {trade_date}: 'winners_df_quality' 的 'percent' 总和为 {winners_df_quality['percent'].sum()}。")
            if not winners_df_quality.empty and winners_df_quality['percent'].sum() > 0:
                winners_df_quality['percent'] = (winners_df_quality['percent'] / winners_df_quality['percent'].sum()) * 100
                winners_df_quality['cum_percent'] = winners_df_quality['percent'].cumsum()
                cost_at_85pct = np.interp(85, winners_df_quality['cum_percent'], winners_df_quality['price'])
                if is_probe_date and pd.isna(cost_at_85pct):
                    print(f"    -> [静态结构探针] @ {trade_date}: 'cost_at_85pct' 缺失。")
                if is_probe_date and cost_at_85pct <= 0:
                    print(f"    -> [静态结构探针] @ {trade_date}: 'cost_at_85pct' 为 {cost_at_85pct}。")
                if pd.notna(cost_at_85pct) and cost_at_85pct > 0:
                    results['winner_profit_cushion'] = (close_price / cost_at_85pct - 1) * 100
                else:
                    results['winner_profit_cushion'] = np.nan
            else:
                results['winner_profit_cushion'] = np.nan
            losers_df_quality = self.df[self.df['price'] > close_price].copy()
            if is_probe_date and losers_df_quality.empty:
                print(f"    -> [静态结构探针] @ {trade_date}: 'losers_df_quality' 为空。")
            if is_probe_date and losers_df_quality['percent'].sum() <= 0:
                print(f"    -> [静态结构探针] @ {trade_date}: 'losers_df_quality' 的 'percent' 总和为 {losers_df_quality['percent'].sum()}。")
            if not losers_df_quality.empty and losers_df_quality['percent'].sum() > 0:
                loser_avg_cost_quality = np.average(losers_df_quality['price'], weights=losers_df_quality['percent'])
                avg_loss_margin = abs((close_price / loser_avg_cost_quality - 1) * 100) if loser_avg_cost_quality > 0 else 0
                losers_df_quality['percent'] = (losers_df_quality['percent'] / losers_df_quality['percent'].sum()) * 100
                losers_df_quality['cum_percent'] = losers_df_quality['percent'].cumsum()
                cost_5pct_loser = np.interp(5, losers_df_quality['cum_percent'], losers_df_quality['price'])
                cost_95pct_loser = np.interp(95, losers_df_quality['cum_percent'], losers_df_quality['price'])
                if is_probe_date and loser_avg_cost_quality <= 0:
                    print(f"    -> [静态结构探针] @ {trade_date}: 'loser_avg_cost_quality' 为 {loser_avg_cost_quality}。")
                loser_concentration = (cost_95pct_loser - cost_5pct_loser) / loser_avg_cost_quality if loser_avg_cost_quality > 0 else 0
                results['loser_pain_index'] = avg_loss_margin * (1 - np.clip(loser_concentration, 0, 1))
            else:
                results['loser_pain_index'] = np.nan
        else:
            results['winner_profit_cushion'] = np.nan
            results['loser_pain_index'] = np.nan
        # 修改行：移除 profit_realization_quality 的计算
        # winner_profit_margin_avg = results.get('winner_profit_margin_avg')
        # total_winner_rate = results.get('total_winner_rate')
        # profit_taking_flow_ratio = self.ctx.get('profit_taking_flow_ratio')
        # if profit_taking_flow_ratio is None:
        #     profit_taking_flow_ratio = np.nan
        # if is_probe_date and pd.isna(winner_profit_margin_avg):
        #     print(f"    -> [静态结构探针] @ {trade_date}: 'winner_profit_margin_avg' 缺失。")
        # if is_probe_date and pd.isna(total_winner_rate):
        #     print(f"    -> [静态结构探针] @ {trade_date}: 'total_winner_rate' 缺失。")
        # if is_probe_date and pd.isna(profit_taking_flow_ratio):
        #     print(f"    -> [静态结构探针] @ {trade_date}: 'profit_taking_flow_ratio' 缺失。")
        # if is_probe_date and pd.notna(profit_taking_flow_ratio) and profit_taking_flow_ratio <= 0:
        #     print(f"    -> [静态结构探针] @ {trade_date}: 'profit_taking_flow_ratio' 为 {profit_taking_flow_ratio}。")
        # if pd.notna(winner_profit_margin_avg) and pd.notna(total_winner_rate) and pd.notna(profit_taking_flow_ratio) and profit_taking_flow_ratio > 0:
        #     results['profit_realization_quality'] = (winner_profit_margin_avg * (total_winner_rate / 100)) / (profit_taking_flow_ratio / 100)
        # else:
        #     results['profit_realization_quality'] = np.nan
        # 8. 成本分布形态分析
        results['cost_structure_skewness'] = self._calculate_cost_structure_skewness(self.ctx)
        # 9. 结构稳定性评估
        concentration_70pct = self.ctx.get('concentration_70pct')
        total_winner_rate_stability = results.get('total_winner_rate')
        winner_profit_cushion_stability = results.get('winner_profit_cushion')
        dominant_peak_profit_margin_stability = results.get('dominant_peak_profit_margin')
        if is_probe_date and pd.isna(concentration_70pct):
            print(f"    -> [静态结构探针] @ {trade_date}: 'concentration_70pct' 缺失。")
        if is_probe_date and pd.isna(total_winner_rate_stability):
            print(f"    -> [静态结构探针] @ {trade_date}: 'total_winner_rate' 缺失。")
        if is_probe_date and pd.isna(winner_profit_cushion_stability):
            print(f"    -> [静态结构探针] @ {trade_date}: 'winner_profit_cushion' 缺失。")
        if is_probe_date and pd.isna(dominant_peak_profit_margin_stability):
            print(f"    -> [静态结构探针] @ {trade_date}: 'dominant_peak_profit_margin' 缺失。")
        if pd.notna(concentration_70pct) and pd.notna(total_winner_rate_stability) and pd.notna(winner_profit_cushion_stability) and pd.notna(dominant_peak_profit_margin_stability):
            concentration_score = np.exp(-5 * np.clip(concentration_70pct, 0, None))
            winner_rate_score = total_winner_rate_stability / 100.0
            cushion_score = np.tanh(winner_profit_cushion_stability / 10.0)
            peak_profit_score = np.tanh(dominant_peak_profit_margin_stability / 20.0)
            scores = [s for s in [concentration_score, winner_rate_score, cushion_score, peak_profit_score] if s > 0]
            if is_probe_date and not scores:
                print(f"    -> [静态结构探针] @ {trade_date}: 结构稳定性计算中所有分数都 <= 0。")
            if scores:
                stability_raw = np.prod(scores)
                final_score = stability_raw ** (1.0 / len(scores))
                results['structural_stability_score'] = final_score * 100
            else:
                results['structural_stability_score'] = np.nan
        else:
            results['structural_stability_score'] = np.nan
        # 10. 近期套牢盘压力
        recent_5d_high = self.ctx.get('high_5d')
        recent_5d_low = self.ctx.get('low_5d')
        turnover_vol_5d = self.ctx.get('turnover_vol_5d')
        total_chip_volume = self.ctx.get('total_chip_volume', np.nan)
        if is_probe_date and pd.isna(recent_5d_high):
            print(f"    -> [静态结构探针] @ {trade_date}: 'high_5d' 缺失。")
        if is_probe_date and pd.isna(recent_5d_low):
            print(f"    -> [静态结构探针] @ {trade_date}: 'low_5d' 缺失。")
        if is_probe_date and pd.isna(turnover_vol_5d):
            print(f"    -> [静态结构探针] @ {trade_date}: 'turnover_vol_5d' 缺失。")
        if is_probe_date and turnover_vol_5d <= 0:
            print(f"    -> [静态结构探针] @ {trade_date}: 'turnover_vol_5d' 为 {turnover_vol_5d}。")
        if is_probe_date and pd.isna(close_price):
            print(f"    -> [静态结构探针] @ {trade_date}: 'close_price' 缺失。")
        if is_probe_date and pd.isna(total_chip_volume):
            print(f"    -> [静态结构探针] @ {trade_date}: 'total_chip_volume' 缺失。")
        if is_probe_date and total_chip_volume <= 0:
            print(f"    -> [静态结构探针] @ {trade_date}: 'total_chip_volume' 为 {total_chip_volume}。")
        if pd.notna(recent_5d_high) and pd.notna(recent_5d_low) and pd.notna(turnover_vol_5d) and turnover_vol_5d > 0 and pd.notna(close_price) and pd.notna(total_chip_volume) and total_chip_volume > 0:
            trapped_mask = (self.df['price'] > close_price) & (self.df['price'] >= recent_5d_low) & (self.df['price'] <= recent_5d_high)
            recent_trapped_percent = self.df[trapped_mask]['percent'].sum()
            recent_trapped_vol = (recent_trapped_percent / 100) * total_chip_volume
            results['recent_trapped_pressure'] = (recent_trapped_vol / turnover_vol_5d) * 100
        else:
            results['recent_trapped_pressure'] = np.nan
        # 11. 潜在获利盘供给
        if is_probe_date and pd.isna(close_price):
            print(f"    -> [静态结构探针] @ {trade_date}: 'close_price' 缺失。")
        if pd.notna(close_price):
            imminent_supply_mask = (self.df['price'] >= close_price / 1.05) & (self.df['price'] < close_price)
            results['imminent_profit_taking_supply'] = self.df[imminent_supply_mask]['percent'].sum()
        else:
            results['imminent_profit_taking_supply'] = np.nan
        # 12. 价格成交量熵 (新增)
        intraday_df = self.ctx.get('processed_intraday_df')
        daily_high = self.ctx.get('high_price')
        daily_low = self.ctx.get('low_price')
        total_daily_volume = self.ctx.get('daily_turnover_volume')
        results['price_volume_entropy'] = self._calculate_price_volume_entropy(intraday_df, daily_high, daily_low, total_daily_volume)
        # 新增：active_buying_support 和 active_selling_pressure
        if is_probe_date and intraday_df is None:
            print(f"    -> [静态结构探针] @ {trade_date}: 'intraday_df' 为 None，无法计算主动买卖支撑/压力。")
        if is_probe_date and intraday_df.empty:
            print(f"    -> [静态结构探针] @ {trade_date}: 'intraday_df' 为空，无法计算主动买卖支撑/压力。")
        if is_probe_date and 'buy_vol_raw' not in intraday_df.columns:
            print(f"    -> [静态结构探针] @ {trade_date}: 'intraday_df' 缺少 'buy_vol_raw' 列。")
        if is_probe_date and 'sell_vol_raw' not in intraday_df.columns:
            print(f"    -> [静态结构探针] @ {trade_date}: 'intraday_df' 缺少 'sell_vol_raw' 列。")
        if is_probe_date and total_daily_volume <= 0:
            print(f"    -> [静态结构探针] @ {trade_date}: 'total_daily_volume' 为 {total_daily_volume}。")
        if not intraday_df.empty and 'buy_vol_raw' in intraday_df.columns and 'sell_vol_raw' in intraday_df.columns and total_daily_volume > 0:
            total_buy_vol_intraday = intraday_df['buy_vol_raw'].sum()
            total_sell_vol_intraday = intraday_df['sell_vol_raw'].sum()
            results['active_buying_support'] = (total_buy_vol_intraday / total_daily_volume) * 100
            results['active_selling_pressure'] = (total_sell_vol_intraday / total_daily_volume) * 100
        else:
            results['active_buying_support'] = np.nan
            results['active_selling_pressure'] = np.nan
        return results

    def _compute_intraday_dynamics_metrics(self, ctx: dict) -> dict:
        results = {}
        trade_date = ctx.get('trade_date', 'UNKNOWN')
        debug_params = ctx.get('debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        is_probe_date = False
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0]).date()
            if probe_date_naive == trade_date:
                is_probe_date = True
        intraday_df = ctx.get('processed_intraday_df', pd.DataFrame())
        if is_probe_date:
            print(f"    -> [日内动态探针] @ {trade_date}: 'intraday_df' 状态：")
            print(f"       - 是否为空: {intraday_df.empty}")
            print(f"       - 行数: {len(intraday_df)}")
            if not intraday_df.empty and 'vol_shares' in intraday_df.columns:
                print(f"       - 'vol_shares' 总和: {intraday_df['vol_shares'].sum():.2f}")
                print(f"       - 'intraday_df' 索引时间范围: {intraday_df.index.min()} to {intraday_df.index.max()}")
                print(f"       - 'intraday_df' 索引前5个时间: {[t.strftime('%H:%M:%S') for t in intraday_df.index[:5].tolist()]}")
                print(f"       - 'intraday_df' 索引后5个时间: {[t.strftime('%H:%M:%S') for t in intraday_df.index[-5:].tolist()]}")
        daily_high = ctx.get('high_price')
        daily_low = ctx.get('low_price')
        daily_open = ctx.get('open_price')
        daily_close = ctx.get('close_price')
        pre_close = ctx.get('pre_close')
        atr_14d = ctx.get('atr_14d')
        total_daily_volume = ctx.get('daily_turnover_volume')
        results['opening_30min_vol_ratio'] = np.nan
        results['opening_30min_range_ratio'] = np.nan
        results['opening_30min_vwap_change'] = np.nan
        results['closing_30min_vol_ratio'] = np.nan
        results['closing_30min_range_ratio'] = np.nan
        results['closing_30min_vwap_change'] = np.nan
        results['intraday_volatility'] = np.nan
        results['intraday_volume_skew'] = np.nan
        results['intraday_trend_persistence'] = np.nan
        results['intraday_reversal_strength'] = np.nan
        results['vwap_close_deviation'] = np.nan
        results['close_to_range_ratio'] = np.nan
        results['opening_gap_defense_strength'] = np.nan
        results['upward_impulse_purity'] = np.nan
        results['active_zone_combat_intensity'] = np.nan # 修改行：初始化
        results['active_zone_mf_stance'] = np.nan # 修改行：初始化
        if intraday_df.empty:
            if is_probe_date:
                print(f"    -> [日内动态探针] @ {trade_date}: 'intraday_df' 为空，跳过计算。")
            return results
        opening_30min_start_time = pd.to_datetime('09:30').time()
        opening_30min_end_time = pd.to_datetime('10:00').time()
        opening_30min_df = intraday_df[(intraday_df.index.time >= opening_30min_start_time) & (intraday_df.index.time < opening_30min_end_time)]
        if is_probe_date and opening_30min_df.empty:
            print(f"    -> [日内动态探针] @ {trade_date}: 'opening_30min_df' 为空。")
            print(f"       - 筛选条件: {opening_30min_start_time} <= time < {opening_30min_end_time}")
            if not intraday_df.empty:
                print(f"       - intraday_df 实际时间范围: {intraday_df.index.time.min()} to {intraday_df.index.time.max()}")
        if not opening_30min_df.empty:
            if is_probe_date and total_daily_volume <= 0:
                print(f"    -> [日内动态探针] @ {trade_date}: 'total_daily_volume' 为 {total_daily_volume}。")
            if total_daily_volume > 0:
                results['opening_30min_vol_ratio'] = opening_30min_df['vol_shares'].sum() / total_daily_volume
            else:
                results['opening_30min_vol_ratio'] = np.nan
            if is_probe_date and pd.isna(daily_high):
                print(f"    -> [日内动态探针] @ {trade_date}: 'daily_high' 缺失。")
            if is_probe_date and pd.isna(daily_low):
                print(f"    -> [日内动态探针] @ {trade_date}: 'daily_low' 缺失。")
            if pd.notna(daily_high) and pd.notna(daily_low) and (daily_high - daily_low) > 0:
                results['opening_30min_range_ratio'] = (opening_30min_df['high'].max() - opening_30min_df['low'].min()) / (daily_high - daily_low)
            else:
                results['opening_30min_range_ratio'] = np.nan
            if is_probe_date and opening_30min_df['minute_vwap'].iloc[0] <= 0:
                print(f"    -> [日内动态探针] @ {trade_date}: 'opening_30min_df' 的 'minute_vwap' 初始值非正。")
            if opening_30min_df['minute_vwap'].iloc[0] > 0:
                results['opening_30min_vwap_change'] = (opening_30min_df['minute_vwap'].iloc[-1] - opening_30min_df['minute_vwap'].iloc[0]) / opening_30min_df['minute_vwap'].iloc[0]
            else:
                results['opening_30min_vwap_change'] = np.nan
        closing_30min_df = intraday_df[intraday_df.index.time >= pd.to_datetime('14:30').time()]
        if is_probe_date and closing_30min_df.empty:
            print(f"    -> [日内动态探针] @ {trade_date}: 'closing_30min_df' 为空。")
        if not closing_30min_df.empty:
            if is_probe_date and total_daily_volume <= 0:
                print(f"    -> [日内动态探针] @ {trade_date}: 'total_daily_volume' 为 {total_daily_volume}。")
            if total_daily_volume > 0:
                results['closing_30min_vol_ratio'] = closing_30min_df['vol_shares'].sum() / total_daily_volume
            else:
                results['closing_30min_vol_ratio'] = np.nan
            if is_probe_date and pd.isna(daily_high):
                print(f"    -> [日内动态探针] @ {trade_date}: 'daily_high' 缺失。")
            if is_probe_date and pd.isna(daily_low):
                print(f"    -> [日内动态探针] @ {trade_date}: 'daily_low' 缺失。")
            if pd.notna(daily_high) and pd.notna(daily_low) and (daily_high - daily_low) > 0:
                results['closing_30min_range_ratio'] = (closing_30min_df['high'].max() - closing_30min_df['low'].min()) / (daily_high - daily_low)
            else:
                results['closing_30min_range_ratio'] = np.nan
            if is_probe_date and closing_30min_df['minute_vwap'].iloc[0] <= 0:
                print(f"    -> [日内动态探针] @ {trade_date}: 'closing_30min_df' 的 'minute_vwap' 初始值非正。")
            if closing_30min_df['minute_vwap'].iloc[0] > 0:
                results['closing_30min_vwap_change'] = (closing_30min_df['minute_vwap'].iloc[-1] - closing_30min_df['minute_vwap'].iloc[0]) / closing_30min_df['minute_vwap'].iloc[0]
            else:
                results['closing_30min_vwap_change'] = np.nan
        if intraday_df['minute_vwap'].mean() > 0:
            results['intraday_volatility'] = intraday_df['minute_vwap'].std() / intraday_df['minute_vwap'].mean()
        else:
            results['intraday_volatility'] = np.nan
        results['intraday_volume_skew'] = intraday_df['vol_shares'].skew()
        price_diff = intraday_df['minute_vwap'].diff().dropna()
        if is_probe_date and len(price_diff) == 0:
            print(f"    -> [日内动态探针] @ {trade_date}: 'price_diff' 为空。")
        if len(price_diff) > 0:
            results['intraday_trend_persistence'] = (price_diff > 0).sum() / len(price_diff)
        else:
            results['intraday_trend_persistence'] = np.nan
        if is_probe_date and pd.isna(daily_close):
            print(f"    -> [日内动态探针] @ {trade_date}: 'daily_close' 缺失。")
        if is_probe_date and pd.isna(daily_open):
            print(f"    -> [日内动态探针] @ {trade_date}: 'daily_open' 缺失。")
        if is_probe_date and pd.isna(daily_high):
            print(f"    -> [日内动态探针] @ {trade_date}: 'daily_high' 缺失。")
        if is_probe_date and pd.isna(daily_low):
            print(f"    -> [日内动态探针] @ {trade_date}: 'daily_low' 缺失。")
        if pd.notna(daily_close) and pd.notna(daily_open) and pd.notna(daily_high) and pd.notna(daily_low) and (daily_high - daily_low) > 0:
            results['intraday_reversal_strength'] = (daily_close - daily_open) / (daily_high - daily_low)
        else:
            results['intraday_reversal_strength'] = np.nan
        if is_probe_date and pd.isna(daily_close):
            print(f"    -> [日内动态探针] @ {trade_date}: 'daily_close' 缺失。")
        if is_probe_date and intraday_df['minute_vwap'].mean() <= 0:
            print(f"    -> [日内动态探针] @ {trade_date}: 'intraday_df' 的 'minute_vwap' 均值非正。")
        if pd.notna(daily_close) and intraday_df['minute_vwap'].mean() > 0:
            results['vwap_close_deviation'] = (daily_close - intraday_df['minute_vwap'].mean()) / intraday_df['minute_vwap'].mean()
        else:
            results['vwap_close_deviation'] = np.nan
        if is_probe_date and pd.isna(daily_close):
            print(f"    -> [日内动态探针] @ {trade_date}: 'daily_close' 缺失。")
        if is_probe_date and pd.isna(daily_low):
            print(f"    -> [日内动态探针] @ {trade_date}: 'daily_low' 缺失。")
        if is_probe_date and pd.isna(daily_high):
            print(f"    -> [日内动态探针] @ {trade_date}: 'daily_high' 缺失。")
        if pd.notna(daily_close) and pd.notna(daily_low) and pd.notna(daily_high) and (daily_high - daily_low) > 0:
            results['close_to_range_ratio'] = (daily_close - daily_low) / (daily_high - daily_low)
        else:
            results['close_to_range_ratio'] = np.nan
        if is_probe_date and pd.isna(daily_open):
            print(f"    -> [日内动态探针] @ {trade_date}: 'daily_open' 缺失。")
        if is_probe_date and pd.isna(pre_close):
            print(f"    -> [日内动态探针] @ {trade_date}: 'pre_close' 缺失。")
        if is_probe_date and pd.isna(atr_14d):
            print(f"    -> [日内动态探针] @ {trade_date}: 'atr_14d' 缺失。")
        if is_probe_date and atr_14d <= 0:
            print(f"    -> [日内动态探针] @ {trade_date}: 'atr_14d' 为 {atr_14d}。")
        if is_probe_date and 'main_force_net_vol' not in intraday_df.columns:
            print(f"    -> [日内动态探针] @ {trade_date}: 'intraday_df' 缺少 'main_force_net_vol' 列。")
        if is_probe_date and 'vol_shares' not in intraday_df.columns:
            print(f"    -> [日内动态探针] @ {trade_date}: 'intraday_df' 缺少 'vol_shares' 列。")
        if pd.notna(daily_open) and pd.notna(pre_close) and pd.notna(atr_14d) and atr_14d > 0 and 'main_force_net_vol' in intraday_df.columns and 'vol_shares' in intraday_df.columns:
            gap_size = (daily_open - pre_close) / atr_14d
            if gap_size > 0:
                opening_30min_df = intraday_df[(intraday_df.index.time >= pd.to_datetime('09:30').time()) & (intraday_df.index.time < pd.to_datetime('10:00').time())]
                if is_probe_date and opening_30min_df.empty:
                    print(f"    -> [日内动态探针] @ {trade_date}: 'opening_30min_df' 为空。")
                if not opening_30min_df.empty:
                    mf_net_flow_opening = opening_30min_df['main_force_net_vol'].sum()
                    total_vol_opening = opening_30min_df['vol_shares'].sum()
                    if is_probe_date and total_vol_opening <= 0:
                        print(f"    -> [日内动态探针] @ {trade_date}: 'total_vol_opening' 为 {total_vol_opening}。")
                    if total_vol_opening > 0:
                        defense_ratio = mf_net_flow_opening / total_vol_opening
                        results['opening_gap_defense_strength'] = defense_ratio * np.log1p(gap_size) * 100
                    else:
                        results['opening_gap_defense_strength'] = np.nan
                else:
                    results['opening_gap_defense_strength'] = np.nan
            else:
                results['opening_gap_defense_strength'] = 0.0
        else:
            results['opening_gap_defense_strength'] = np.nan
        if is_probe_date and pd.isna(daily_close):
            print(f"    -> [日内动态探针] @ {trade_date}: 'daily_close' 缺失。")
        if is_probe_date and pd.isna(daily_open):
            print(f"    -> [日内动态探针] @ {trade_date}: 'daily_open' 缺失。")
        if is_probe_date and pd.isna(daily_high):
            print(f"    -> [日内动态探针] @ {trade_date}: 'daily_high' 缺失。")
        if is_probe_date and pd.isna(daily_low):
            print(f"    -> [日内动态探针] @ {trade_date}: 'daily_low' 缺失。")
        if pd.notna(daily_close) and pd.notna(daily_open) and pd.notna(daily_high) and pd.notna(daily_low):
            if daily_close > daily_open:
                up_range = daily_high - max(daily_open, daily_close)
                down_range = min(daily_open, daily_close) - daily_low
                total_range = daily_high - daily_low
                if is_probe_date and total_range <= 0:
                    print(f"    -> [日内动态探针] @ {trade_date}: 'total_range' 为 {total_range}。")
                if total_range > 0:
                    purity = ((daily_close - daily_open) - (up_range + down_range)) / total_range
                    results['upward_impulse_purity'] = purity * 100
                else:
                    results['upward_impulse_purity'] = np.nan
            else:
                results['upward_impulse_purity'] = 0.0
        else:
            results['upward_impulse_purity'] = np.nan
        # 修改行：新增 active_zone_combat_intensity 和 active_zone_mf_stance 的计算
        if not intraday_df.empty and pd.notna(daily_close) and pd.notna(atr_14d) and atr_14d > 0 and total_daily_volume > 0 and 'main_force_net_vol' in intraday_df.columns and 'vol_shares' in intraday_df.columns:
            active_zone_low = daily_close - 0.5 * atr_14d
            active_zone_high = daily_close + 0.5 * atr_14d
            active_zone_df = intraday_df[(intraday_df['minute_vwap'] >= active_zone_low) & (intraday_df['minute_vwap'] <= active_zone_high)]
            if is_probe_date and active_zone_df.empty:
                print(f"    -> [日内动态探针] @ {trade_date}: 'active_zone_df' 为空。")
            if not active_zone_df.empty:
                active_zone_vol = active_zone_df['vol_shares'].sum()
                results['active_zone_combat_intensity'] = (active_zone_vol / total_daily_volume) * 100
                mf_net_flow_active_zone = active_zone_df['main_force_net_vol'].sum()
                if active_zone_vol > 0:
                    results['active_zone_mf_stance'] = (mf_net_flow_active_zone / active_zone_vol) * 100
                else:
                    results['active_zone_mf_stance'] = np.nan
            else:
                results['active_zone_combat_intensity'] = np.nan
                results['active_zone_mf_stance'] = np.nan
        else:
            results['active_zone_combat_intensity'] = np.nan
            results['active_zone_mf_stance'] = np.nan
        return results

    def _calculate_chip_structure_health_score(self, context: dict) -> dict:
        """
        【V4.2 · 生产就绪版】
        - 核心优化: 移除所有调试探针，代码恢复生产状态。
        """
        from scipy.stats import percentileofscore
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
        【V1.0】计算活跃获利盘利润率，并加入探针。
        """
        stock_code = context.get('stock_code', 'UNKNOWN')
        trade_date = context.get('trade_date', 'UNKNOWN')
        debug_params = context.get('debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        is_probe_date = False
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0]).date()
            if probe_date_naive == trade_date:
                is_probe_date = True
        active_winner_avg_cost = np.nan
        active_profit_margin = 0.0
        if pd.notna(close_price) and pd.notna(atr_14d) and atr_14d > 0:
            # 活跃获利盘：成本在 (close_price - 3*ATR, close_price) 之间，扩大范围
            active_winners_df = self.df[(self.df['price'] < close_price) & (self.df['price'] >= close_price - 3 * atr_14d)]
            if is_probe_date:
                print(f"    -> [活跃获利盘利润率探针] @ {trade_date}:")
                print(f"       - close_price: {close_price:.4f}, atr_14d: {atr_14d:.4f}")
                print(f"       - active_winners_df (filtered): {active_winners_df.head()}")
                print(f"       - active_winners_df.empty: {active_winners_df.empty}, percent.sum(): {active_winners_df['percent'].sum():.4f}")
            if not active_winners_df.empty and active_winners_df['percent'].sum() > 0:
                active_winner_avg_cost = np.average(active_winners_df['price'], weights=active_winners_df['percent'])
                if active_winner_avg_cost > 0:
                    active_profit_margin = ((close_price - active_winner_avg_cost) / active_winner_avg_cost) * 100
                if is_probe_date:
                    print(f"       - 活跃获利盘存在。active_winner_avg_cost: {active_winner_avg_cost:.4f}, active_profit_margin: {active_profit_margin:.4f}")
            else:
                # 如果活跃获利盘区间为空，则考虑更广阔的获利盘范围，例如使用 dominant_peak_cost
                dominant_peak_cost = context.get('dominant_peak_cost')
                if is_probe_date:
                    print(f"       - 活跃获利盘区间为空。尝试使用 dominant_peak_cost: {dominant_peak_cost:.4f}")
                if pd.notna(dominant_peak_cost) and dominant_peak_cost > 0 and dominant_peak_cost < close_price:
                    active_winner_avg_cost = dominant_peak_cost
                    active_profit_margin = ((close_price - dominant_peak_cost) / dominant_peak_cost) * 100
                    if is_probe_date:
                        print(f"       - 使用 dominant_peak_cost。active_winner_avg_cost: {active_winner_avg_cost:.4f}, active_profit_margin: {active_profit_margin:.4f}")
                else:
                    # 在涨停日，如果无法计算出活跃获利盘，则赋予一个积极的默认利润率
                    # 假设涨停日至少有 9% 的利润率
                    pre_close = context.get('pre_close', close_price)
                    if (close_price / pre_close - 1) > 0.098: # 如果是涨停日 (考虑0.098作为涨停阈值)
                        active_profit_margin = 9.0 # 涨停日默认利润率
                        if is_probe_date:
                            print(f"       - 涨停日，赋予默认利润率: {active_profit_margin:.4f}")
                    else:
                        active_profit_margin = 0.0 # 默认利润率为0
                        if is_probe_date:
                            print(f"       - 非涨停日且无活跃获利盘，利润率: {active_profit_margin:.4f}")
        else:
            if is_probe_date:
                print(f"       - close_price 或 atr_14d 无效，利润率: {active_profit_margin:.4f}")
        return active_winner_avg_cost, active_profit_margin

    def _calculate_winner_conviction_index(self, context: dict, active_profit_margin: float) -> float:
        """
        【V1.5 · 压力缓和器版】计算赢家信念指数。
        - 核心升级: 将 `realized_pressure` 的计算函数从 `tanh` 替换为 `arctan`。
        - 核心思想: 使用 `arctan` 作为“压力缓和器”，解决因分母过小（小基数陷阱）导致的 `pressure_ratio` 异常放大问题，增强指标对极端值的鲁棒性。
        """
        stock_code = context.get('stock_code', 'UNKNOWN')
        trade_date = context.get('trade_date', 'UNKNOWN')
        debug_params = context.get('debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        is_probe_date = False
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0]).date()
            if probe_date_naive == trade_date:
                is_probe_date = True
        bullish_reinforcement = context.get('upward_impulse_purity', 0.0)
        profit_taking_flow_ratio = context.get('profit_taking_flow_ratio', 0.0)
        active_winner_rate = context.get('active_winner_rate', 0.0)
        close_price = context.get('close_price')
        pre_close = context.get('pre_close', close_price)
        winner_conviction_index = 0.0
        # if is_probe_date:
        #     print(f"    -> [赢家信念指数探针] @ {trade_date}:")
        #     print(f"       - active_profit_margin: {active_profit_margin:.4f}")
        #     print(f"       - bullish_reinforcement: {bullish_reinforcement:.4f}")
        #     print(f"       - profit_taking_flow_ratio: {profit_taking_flow_ratio:.4f}")
        #     print(f"       - active_winner_rate: {active_winner_rate:.4f}")
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
            if is_probe_date:
                print(f"       - pressure_ratio: {pressure_ratio:.4f}")
                print(f"       - realized_pressure (arctan): {realized_pressure:.4f}")
                print(f"       - hesitation_factor: {hesitation_factor:.4f}")
                print(f"       - margin_factor: {margin_factor:.4f}")
                print(f"       - reinforcement_factor: {reinforcement_factor:.4f}")
                print(f"       - final winner_conviction_index: {winner_conviction_index:.4f}")
        else:
            if is_probe_date:
                print(f"       - 缺少前置条件，winner_conviction_index: {winner_conviction_index:.4f}")
        return winner_conviction_index

    def _calculate_cost_structure_skewness(self, context: dict) -> float:
        """
        【V1.1】计算成本结构偏度，并加入探针。
        - 【修正】移除对 `skewness` 的负号操作，使正偏度（筹码集中在高价区）对应正值，符合涨停日积极信号的预期。
        """
        stock_code = context.get('stock_code', 'UNKNOWN')
        trade_date = context.get('trade_date', 'UNKNOWN')
        debug_params = context.get('debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        is_probe_date = False
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0]).date()
            if probe_date_naive == trade_date:
                is_probe_date = True
        skewness = 0.0
        if not self.df.empty and self.df['percent'].sum() >= 1e-6:
            from scipy.stats import skew
            total_percent = self.df['percent'].sum()
            weights = np.round((self.df['percent'] / total_percent) * 10000).astype(int)
            valid_weights = weights[weights > 0]
            if is_probe_date:
                print(f"    -> [成本结构偏度探针] @ {trade_date}:")
                print(f"       - total_percent: {total_percent:.4f}")
                print(f"       - valid_weights count: {len(valid_weights)}")
            if len(valid_weights) >= 3: # 至少需要3个点才能计算偏度
                valid_prices = self.df['price'][weights > 0]
                unweighted_sample = np.repeat(valid_prices, valid_weights)
                skewness = skew(unweighted_sample)
                # 移除负号操作，使正偏度（筹码集中在高价区）对应正值
                # skewness = -skewness
                if is_probe_date:
                    print(f"       - unweighted_sample min/max: {np.min(unweighted_sample):.4f}/{np.max(unweighted_sample):.4f}")
                    print(f"       - raw skewness: {skewness:.4f}, final cost_structure_skewness: {skewness:.4f}")
            else:
                if is_probe_date:
                    print(f"       - 有效权重不足3个，无法计算偏度。skewness: {skewness:.4f}")
        else:
            if is_probe_date:
                print(f"       - 筹码分布为空或百分比和过低，skewness: {skewness:.4f}")
        return skewness

    def _calculate_price_volume_entropy(self, intraday_df: pd.DataFrame, daily_high: float, daily_low: float, total_daily_volume: float) -> float:
        from scipy.stats import entropy
        stock_code = self.ctx.get('stock_code', 'UNKNOWN')
        trade_date = self.ctx.get('trade_date', 'UNKNOWN')
        debug_params = self.ctx.get('debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        is_probe_date = False
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0]).date()
            if probe_date_naive == trade_date:
                is_probe_date = True
        if is_probe_date and intraday_df is None:
            print(f"    -> [价格成交量熵探针] @ {trade_date}: 'intraday_df' 为 None。")
        if is_probe_date and intraday_df.empty:
            print(f"    -> [价格成交量熵探针] @ {trade_date}: 'intraday_df' 为空。")
        if is_probe_date and total_daily_volume <= 0:
            print(f"    -> [价格成交量熵探针] @ {trade_date}: 'total_daily_volume' 为 {total_daily_volume}。")
        if is_probe_date and pd.isna(daily_high):
            print(f"    -> [价格成交量熵探针] @ {trade_date}: 'daily_high' 缺失。")
        if is_probe_date and pd.isna(daily_low):
            print(f"    -> [价格成交量熵探针] @ {trade_date}: 'daily_low' 缺失。")
        if is_probe_date and daily_high <= daily_low:
            print(f"    -> [价格成交量熵探针] @ {trade_date}: 'daily_high' ({daily_high}) <= 'daily_low' ({daily_low})，价格范围无效。")
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
        if is_probe_date and valid_data.empty:
            print(f"    -> [价格成交量熵探针] @ {trade_date}: 'valid_data' 为空。")
        if is_probe_date and valid_data['volume'].sum() <= 0:
            print(f"    -> [价格成交量熵探针] @ {trade_date}: 'valid_data' 的 'volume' 总和为 {valid_data['volume'].sum()}。")
        if valid_data.empty or valid_data['volume'].sum() <= 0:
            return np.nan
        bins = pd.cut(valid_data['price'], bins=num_bins, include_lowest=True, duplicates='drop')
        volume_per_bin = valid_data.groupby(bins)['volume'].sum()
        volume_per_bin = volume_per_bin[volume_per_bin > 0]
        if is_probe_date and volume_per_bin.empty:
            print(f"    -> [价格成交量熵探针] @ {trade_date}: 'volume_per_bin' 为空，所有成交量集中在单个价格点或无成交。")
        if volume_per_bin.empty:
            return 0.0
        probabilities = volume_per_bin / volume_per_bin.sum()
        shannon_entropy = entropy(probabilities, base=2)
        max_entropy = np.log2(len(volume_per_bin)) if len(volume_per_bin) > 1 else 0
        normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0.0
        if is_probe_date:
            print(f"    -> [价格成交量熵探针] @ {trade_date}: 价格成交量熵计算结果：{normalized_entropy:.4f} (原始熵: {shannon_entropy:.4f}, 箱数: {len(volume_per_bin)})。")
        return normalized_entropy

    def _compute_microstructure_game_metrics(self, context: dict) -> dict:
        """
        【V1.2 · 日内数据列鲁棒性增强版】计算基于高频数据的筹码微观博弈指标。
        - 核心增强: 增加数据源检查。如果日内数据不包含高频特征（如 'buy_vol_raw'），则优雅地跳过计算，返回默认值。
        - 核心修复: 修正 `dip_or_flat_df` 的计算，优先使用 'close' 和 'open' 列，若不存在则回退到 'minute_vwap'。
        """
        results = {
            'peak_exchange_purity': np.nan,
            'pressure_validation_score': np.nan,
            'support_validation_score': np.nan,
            'covert_accumulation_signal': np.nan,
        }
        intraday_df = context.get('processed_intraday_df')
        if intraday_df.empty or 'buy_vol_raw' not in intraday_df.columns:
            return results
        peak_low = context.get('peak_range_low')
        peak_high = context.get('peak_range_high')
        if not all(pd.notna(v) for v in [peak_low, peak_high]):
            return results
        # 1. 筹码交换纯度 (Peak Exchange Purity)
        peak_zone_df = intraday_df[(intraday_df['minute_vwap'] >= peak_low) & (intraday_df['minute_vwap'] <= peak_high)]
        if not peak_zone_df.empty:
            total_vol_peak = peak_zone_df['vol_shares'].sum()
            if total_vol_peak > 0:
                active_buy_vol = peak_zone_df['buy_vol_raw'].sum()
                active_sell_vol = peak_zone_df['sell_vol_raw'].sum()
                purity = 1 - abs(active_buy_vol - active_sell_vol) / total_vol_peak
                results['peak_exchange_purity'] = purity * 100
        # 2. 压力/支撑有效性验证 (Pressure/Support Validation)
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
        # 3. 隐蔽吸筹信号 (Covert Accumulation Signal)
        # 优先使用 'close' 和 'open'，如果不存在则回退到 'minute_vwap'
        if 'close' in intraday_df.columns and 'open' in intraday_df.columns:
            dip_or_flat_df = intraday_df[intraday_df['close'] <= intraday_df['open']]
        else:
            # 如果没有 'close' 和 'open'，则使用 minute_vwap 作为替代
            # 假设 minute_vwap 下降或持平代表“下跌或平盘”
            dip_or_flat_df = intraday_df[intraday_df['minute_vwap'].diff().fillna(0) <= 0]
            logger.warning(f"[{context.get('stock_code', 'UNKNOWN')}] [{context.get('trade_date', 'UNKNOWN')}] 日内数据缺少'close'或'open'列，'隐蔽吸筹信号'计算回退到使用'minute_vwap'。")
        if not dip_or_flat_df.empty:
            total_vol_dip = dip_or_flat_df['vol_shares'].sum()
            if total_vol_dip > 0:
                mf_net_buy_on_dip = dip_or_flat_df['main_force_net_vol'].clip(lower=0).sum()
                results['covert_accumulation_signal'] = (mf_net_buy_on_dip / total_vol_dip) * 100
        return results







