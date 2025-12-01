# 文件: strategies/trend_following/intelligence/intraday_behavior_engine.py
import asyncio
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Optional, Any
# 导入 get_params_block 工具
from strategies.trend_following.utils import (
    get_params_block, get_param_value, normalize_to_bipolar, 
    get_adaptive_mtf_normalized_score, is_limit_up, get_adaptive_mtf_normalized_bipolar_score, 
    normalize_score
)

class IntradayBehaviorEngine:
    """
    【V2.0 · 三大公理重构版】
    - 核心升级: 废弃旧的复杂指标和诊断模型，引入基于日内博弈本质的“攻击、控制、转折”三大公理。
                使引擎更轻量、更聚焦、逻辑更清晰。
    """
    def __init__(self, strategy_instance):
        """初始化时加载专属配置，并获取指标计算器的引用"""
        self.strategy = strategy_instance
        self.calculator = strategy_instance.orchestrator.indicator_service.calculator
        self.params = get_params_block(self.strategy, 'intraday_behavior_engine_params', {})

    def _get_safe_series(self, df: pd.DataFrame, column_name: str, default_value: Any = 0.0, method_name: str = "未知方法") -> pd.Series:
        """
        安全地从DataFrame获取Series，如果不存在则打印警告并返回默认Series。
        """
        if column_name not in df.columns:
            print(f"    -> [日内行为情报警告] 方法 '{method_name}' 缺少数据 '{column_name}'，使用默认值 {default_value}。")
            return pd.Series(default_value, index=df.index)
        return df[column_name]

    def _validate_required_signals(self, df: pd.DataFrame, required_signals: list, method_name: str) -> bool:
        """
        【V1.0 · 战前情报校验】内部辅助方法，用于在方法执行前验证所有必需的数据信号是否存在。
        """
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            # 调整校验信息为“日内行为情报校验”
            print(f"    -> [日内行为情报校验] 方法 '{method_name}' 启动失败：缺少核心信号 {missing_signals}。")
            return False
        return True

    async def _prepare_intraday_indicators(self, df_minute: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        【V3.0 · 战报精简版】
        统一为分钟数据计算所有必需的战术指标，仅保留VWAP。
        """
        if df_minute is None or df_minute.empty:
            return None
        df_enriched = df_minute.copy()
        # 移除不再需要的KDJ计算
        # VWAP (为公理二：支配共识服务)
        df_enriched = await self.calculator.calculate_vwap(df_enriched)
        return df_enriched

    async def run_intraday_diagnostics(self, df_minute: pd.DataFrame) -> Dict[str, float]:
        """
        【V4.0 · 战局推演版】日内诊断总指挥
        - 核心流程:
          1. 准备分钟级核心数据 (VWAP)。
          2. 并行诊断“进攻纯度”、“支配共识”、“信念反转”三大战报，以及新增的“开盘博弈”、“尾盘奇袭”、“韧性承接”、“攻防节奏”四大战局信号。
          3. 返回所有信号的最终值，为日线级分析提供过程性解释。
        """
        # [代码修改] 更新版本号和描述
        print("启动【V4.0 · 战局推演版】日内行为诊断...")
        df_enriched = await self._prepare_intraday_indicators(df_minute)
        if df_enriched is None or df_enriched.empty:
            print("分钟数据为空，无法进行日内行为诊断。")
            # [代码修改] 增加新的信号到默认返回字典
            return {
                "SCORE_INTRADAY_OFFENSIVE_PURITY": 0.0,
                "SCORE_INTRADAY_DOMINANCE_CONSENSUS": 0.0,
                "SCORE_INTRADAY_CONVICTION_REVERSAL": 0.0,
                "SCORE_INTRADAY_OPENING_GAMBIT": 0.0,
                "SCORE_INTRADAY_CLOSING_AMBUSH": 0.0,
                "SCORE_INTRADAY_RESILIENCE_ABSORPTION": 0.0,
                "SCORE_INTRADAY_BATTLE_TEMPO": 0.0,
            }
        # [代码修改] 并行执行所有战报和战局的诊断
        tasks = [
            self._diagnose_offensive_purity(df_enriched),
            self._diagnose_dominance_consensus(df_enriched),
            self._diagnose_conviction_reversal(df_enriched),
            self._diagnose_opening_gambit(df_enriched),
            self._diagnose_closing_ambush(df_enriched),
            self._diagnose_resilience_absorption(df_enriched),
            self._diagnose_battle_tempo(df_enriched),
        ]
        results = await asyncio.gather(*tasks)
        final_scores = {}
        for res_dict in results:
            final_scores.update(res_dict)
        print(f"日内行为诊断完成: {final_scores}")
        return final_scores

    async def _diagnose_offensive_purity(self, df_minute: pd.DataFrame) -> Dict[str, float]:
        """
        【V1.3 · 探针逻辑重构版】日内战报之一：诊断“进攻纯度”
        - 核心重构: 重构探针逻辑，使其能正确识别当前处理的数据段是否覆盖了探针日期，
                      从而在历史回溯时也能准确触发。
        """
        required_signals = ['close', 'amount', 'main_force_ofi', 'retail_ofi', 'buy_quote_exhaustion_rate', 'sell_quote_exhaustion_rate']
        if not self._validate_required_signals(df_minute, required_signals, "_diagnose_offensive_purity"):
            return {"SCORE_INTRADAY_OFFENSIVE_PURITY": 0.0}
        price_change = self._get_safe_series(df_minute, 'close').diff().fillna(0)
        amount = self._get_safe_series(df_minute, 'amount').replace(0, 1e-9)
        efficiency = price_change / amount
        norm_efficiency = normalize_to_bipolar(efficiency, df_minute.index, window=240, sensitivity=0.1)
        mf_ofi = self._get_safe_series(df_minute, 'main_force_ofi')
        retail_ofi = self._get_safe_series(df_minute, 'retail_ofi')
        driver = mf_ofi - retail_ofi
        norm_driver = normalize_to_bipolar(driver, df_minute.index, window=240)
        buy_urgency = self._get_safe_series(df_minute, 'buy_quote_exhaustion_rate')
        sell_urgency = self._get_safe_series(df_minute, 'sell_quote_exhaustion_rate')
        urgency = buy_urgency - sell_urgency
        norm_urgency = normalize_to_bipolar(urgency, df_minute.index, window=240)
        purity_score = (norm_efficiency.pow(0.2) * norm_driver.pow(0.5) * norm_urgency.pow(0.3)).fillna(0)
        bullish_minutes = purity_score[price_change > 0]
        bullish_weights = amount[price_change > 0]
        bearish_minutes = purity_score[price_change < 0].abs()
        bearish_weights = amount[price_change < 0]
        avg_bullish_purity = np.average(bullish_minutes, weights=bullish_weights) if not bullish_minutes.empty and bullish_weights.sum() > 0 else 0
        avg_bearish_purity = np.average(bearish_minutes, weights=bearish_weights) if not bearish_minutes.empty and bearish_weights.sum() > 0 else 0
        final_score = (avg_bullish_purity - avg_bearish_purity)
        # --- 重构探针逻辑以适配历史回溯 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates and not df_minute.empty:
            processed_date_str = df_minute.index[0].strftime('%Y-%m-%d')
            if processed_date_str in probe_dates:
                print(f"      [日内行为探针] _diagnose_offensive_purity @ {processed_date_str}")
                print(f"        - 日内平均多头进攻纯度: {avg_bullish_purity:.4f}")
                print(f"        - 日内平均空头进攻纯度: {avg_bearish_purity:.4f}")
                print(f"        - 最终进攻纯度分: {final_score:.4f}")
        return {"SCORE_INTRADAY_OFFENSIVE_PURITY": final_score}

    async def _diagnose_dominance_consensus(self, df_minute: pd.DataFrame) -> Dict[str, float]:
        """
        【V1.2 · 探针逻辑重构版】日内战报之二：诊断“支配共识”
        - 核心重构: 重构探针逻辑，使其能正确识别当前处理的数据段是否覆盖了探针日期，
                      从而在历史回溯时也能准确触发。
        """
        required_signals = ['close', 'vwap', 'amount']
        if not self._validate_required_signals(df_minute, required_signals, "_diagnose_dominance_consensus"):
            return {"SCORE_INTRADAY_DOMINANCE_CONSENSUS": 0.0}
        vwap = self._get_safe_series(df_minute, 'vwap').replace(0, np.nan).ffill()
        price_deviation = (self._get_safe_series(df_minute, 'close') - vwap) / vwap
        amount_ratio = self._get_safe_series(df_minute, 'amount') / self._get_safe_series(df_minute, 'amount').mean()
        dominance_strength = (price_deviation * amount_ratio).fillna(0)
        norm_dominance_strength = normalize_to_bipolar(dominance_strength, df_minute.index, window=240, sensitivity=0.5)
        consensus_trend = norm_dominance_strength.ewm(span=21, adjust=False).mean()
        avg_strength = norm_dominance_strength.mean()
        final_trend = consensus_trend.iloc[-1] if not consensus_trend.empty else 0
        final_score = (avg_strength * 0.5 + final_trend * 0.5)
        # --- 重构探针逻辑以适配历史回溯 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates and not df_minute.empty:
            processed_date_str = df_minute.index[0].strftime('%Y-%m-%d')
            if processed_date_str in probe_dates:
                print(f"      [日内行为探针] _diagnose_dominance_consensus @ {processed_date_str}")
                print(f"        - 全天平均支配强度: {avg_strength:.4f}")
                print(f"        - 收盘共识趋势: {final_trend:.4f}")
                print(f"        - 最终支配共识分: {final_score:.4f}")
        return {"SCORE_INTRADAY_DOMINANCE_CONSENSUS": final_score}

    async def _diagnose_conviction_reversal(self, df_minute: pd.DataFrame) -> Dict[str, float]:
        """
        【V1.3 · 逻辑与探针重构版】日内战报之三：诊断“信念反转”
        - 核心修复: 修复了因 main_force_execution_alpha 为负就将看涨证据完全归零的“一票否决”逻辑脆弱性。
                      采用 tanh 函数将双极性 Alpha 柔性映射到 [0, 1] 区间，使模型更稳健。
        - 核心重构: 重构探针逻辑，使其能正确识别当前处理的数据段是否覆盖了探针日期。
        """
        # 此方法现在需要从分钟数据中找到对应的日线数据
        if df_minute.empty:
            return {"SCORE_INTRADAY_CONVICTION_REVERSAL": 0.0}
        current_date = df_minute.index[0].normalize()
        if current_date not in self.strategy.df_indicators.index:
            print(f"    -> [日内行为情报警告] _diagnose_conviction_reversal: 在日线数据中未找到日期 {current_date}，跳过计算。")
            return {"SCORE_INTRADAY_CONVICTION_REVERSAL": 0.0}
        daily_signals = self.strategy.df_indicators.loc[current_date]
        panic_score = daily_signals.get('panic_selling_cascade_D', 0.0)
        absorption_score = daily_signals.get('capitulation_absorption_index_D', 0.0)
        # 修复“一票否决”逻辑
        mf_alpha_raw = daily_signals.get('main_force_execution_alpha_D', 0.0)
        # 使用tanh进行柔性映射，k=2.0表示对alpha的正负较为敏感
        bullish_alpha_score = (np.tanh(mf_alpha_raw * 2.0) + 1) / 2
        bullish_reversal_evidence = (panic_score * absorption_score * bullish_alpha_score).pow(1/3)
        distribution_score = daily_signals.get('rally_distribution_pressure_D', 0.0)
        # 尝试获取5日斜率，如果不存在，则使用1日斜率作为备用
        conviction_slope_5d = daily_signals.get('SLOPE_5_main_force_conviction_index_D', None)
        if pd.isna(conviction_slope_5d):
             conviction_slope_1d = daily_signals.get('SLOPE_1_main_force_conviction_index_D', 0.0)
             conviction_decay = max(0, -conviction_slope_1d)
        else:
             conviction_decay = max(0, -conviction_slope_5d)
        mf_alpha_bearish = abs(min(mf_alpha_raw, 0.0))
        bearish_reversal_evidence = (distribution_score * conviction_decay * mf_alpha_bearish).pow(1/3)
        final_score = bullish_reversal_evidence - bearish_reversal_evidence
        # --- 重构探针逻辑以适配历史回溯 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates:
            processed_date_str = current_date.strftime('%Y-%m-%d')
            if processed_date_str in probe_dates:
                print(f"      [日内行为探针] _diagnose_conviction_reversal @ {processed_date_str}")
                print(f"        - 看涨证据: 恐慌={panic_score:.2f}, 承接={absorption_score:.2f}, 主力Alpha分(新)={bullish_alpha_score:.2f} (原始Alpha={mf_alpha_raw:.2f}) -> 综合={bullish_reversal_evidence:.4f}")
                print(f"        - 看跌证据: 派发={distribution_score:.2f}, 信念衰减={conviction_decay:.2f}, 主力Alpha-={mf_alpha_bearish:.2f} -> 综合={bearish_reversal_evidence:.4f}")
                print(f"        - 最终信念反转分: {final_score:.4f}")
        return {"SCORE_INTRADAY_CONVICTION_REVERSAL": np.clip(final_score, -1, 1)}

    async def _diagnose_opening_gambit(self, df_minute: pd.DataFrame) -> Dict[str, float]:
        """
        【V1.0 · 新增】日内战局之一：诊断“开盘博弈”
        """
        params = get_params_block(self.strategy, 'intraday_battle_report_params.opening_gambit', {})
        if not get_param_value(params.get('enabled'), False):
            return {"SCORE_INTRADAY_OPENING_GAMBIT": 0.0}
        duration = get_param_value(params.get('duration_minutes'), 30)
        opening_df = df_minute.between_time('09:30', (pd.to_datetime('09:30') + pd.Timedelta(minutes=duration)).strftime('%H:%M'))
        if opening_df.empty or len(opening_df) < 5:
            return {"SCORE_INTRADAY_OPENING_GAMBIT": 0.0}
        # 1. 价格趋势分
        price_trend_score = (opening_df['close'].iloc[-1] / opening_df['close'].iloc[0] - 1) * 100
        norm_price_trend = normalize_to_bipolar(pd.Series(price_trend_score), window=1, sensitivity=2.0).iloc[0]
        # 2. VWAP控制分
        vwap_control_score = (opening_df['close'] - opening_df['vwap']) / opening_df['vwap']
        avg_vwap_control = vwap_control_score.mean()
        norm_vwap_control = normalize_to_bipolar(pd.Series(avg_vwap_control), window=1, sensitivity=0.01).iloc[0]
        # 3. 成交量强度分
        volume_strength = opening_df['amount'].sum() / df_minute['amount'].sum()
        norm_volume = normalize_to_bipolar(pd.Series(volume_strength), window=1, sensitivity=0.2).iloc[0]
        # 融合
        w_price = get_param_value(params.get('price_trend_weight'), 0.3)
        w_vwap = get_param_value(params.get('vwap_weight'), 0.4)
        w_vol = get_param_value(params.get('volume_weight'), 0.3)
        final_score = norm_price_trend * w_price + norm_vwap_control * w_vwap + norm_volume * w_vol
        # --- 探针 ---
        probe_params = get_params_block(self.strategy, 'intraday_battle_report_params.probe_params', {})
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(probe_params.get('enabled'), False) and get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates and not df_minute.empty:
            processed_date_str = df_minute.index[0].strftime('%Y-%m-%d')
            if processed_date_str in probe_dates:
                print(f"      [日内战局探针] _diagnose_opening_gambit @ {processed_date_str}")
                print(f"        - 价格趋势分: {norm_price_trend:.4f}, VWAP控制分: {norm_vwap_control:.4f}, 成交量强度分: {norm_volume:.4f}")
                print(f"        - 最终开盘博弈分: {final_score:.4f}")
        return {"SCORE_INTRADAY_OPENING_GAMBIT": np.clip(final_score, -1, 1)}

    async def _diagnose_closing_ambush(self, df_minute: pd.DataFrame) -> Dict[str, float]:
        """
        【V1.0 · 新增】日内战局之二：诊断“尾盘奇袭”
        """
        params = get_params_block(self.strategy, 'intraday_battle_report_params.closing_ambush', {})
        if not get_param_value(params.get('enabled'), False):
            return {"SCORE_INTRADAY_CLOSING_AMBUSH": 0.0}
        duration = get_param_value(params.get('duration_minutes'), 60)
        closing_df = df_minute.between_time((pd.to_datetime('15:00') - pd.Timedelta(minutes=duration)).strftime('%H:%M'), '15:00')
        if closing_df.empty or len(closing_df) < 5:
            return {"SCORE_INTRADAY_CLOSING_AMBUSH": 0.0}
        # 1. 价格趋势分 (与日内高点的关系)
        price_trend_score = (closing_df['close'].iloc[-1] - closing_df['close'].iloc[0]) / (df_minute['high'].max() - df_minute['low'].min() + 1e-9)
        norm_price_trend = normalize_to_bipolar(pd.Series(price_trend_score), window=1, sensitivity=0.5).iloc[0]
        # 2. VWAP控制分
        vwap_control_score = (closing_df['close'] - closing_df['vwap']) / closing_df['vwap']
        avg_vwap_control = vwap_control_score.mean()
        norm_vwap_control = normalize_to_bipolar(pd.Series(avg_vwap_control), window=1, sensitivity=0.01).iloc[0]
        # 3. 成交量强度分
        volume_strength = closing_df['amount'].sum() / df_minute['amount'].sum()
        norm_volume = normalize_to_bipolar(pd.Series(volume_strength), window=1, sensitivity=0.2).iloc[0]
        # 融合
        w_price = get_param_value(params.get('price_trend_weight'), 0.4)
        w_vwap = get_param_value(params.get('vwap_weight'), 0.4)
        w_vol = get_param_value(params.get('volume_weight'), 0.2)
        final_score = norm_price_trend * w_price + norm_vwap_control * w_vwap + norm_volume * w_vol
        # --- 探针 ---
        probe_params = get_params_block(self.strategy, 'intraday_battle_report_params.probe_params', {})
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(probe_params.get('enabled'), False) and get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates and not df_minute.empty:
            processed_date_str = df_minute.index[0].strftime('%Y-%m-%d')
            if processed_date_str in probe_dates:
                print(f"      [日内战局探针] _diagnose_closing_ambush @ {processed_date_str}")
                print(f"        - 价格趋势分: {norm_price_trend:.4f}, VWAP控制分: {norm_vwap_control:.4f}, 成交量强度分: {norm_volume:.4f}")
                print(f"        - 最终尾盘奇袭分: {final_score:.4f}")
        return {"SCORE_INTRADAY_CLOSING_AMBUSH": np.clip(final_score, -1, 1)}

    async def _diagnose_resilience_absorption(self, df_minute: pd.DataFrame) -> Dict[str, float]:
        """
        【V1.0 · 新增】日内战局之三：诊断“韧性承接”
        """
        params = get_params_block(self.strategy, 'intraday_battle_report_params.resilience_absorption', {})
        if not get_param_value(params.get('enabled'), False):
            return {"SCORE_INTRADAY_RESILIENCE_ABSORPTION": 0.0}
        drawdown_threshold = get_param_value(params.get('drawdown_threshold'), -0.015)
        df_minute['cummax_close'] = df_minute['close'].cummax()
        df_minute['drawdown'] = (df_minute['close'] - df_minute['cummax_close']) / df_minute['cummax_close']
        dips = df_minute[df_minute['drawdown'] < drawdown_threshold]
        if dips.empty:
            return {"SCORE_INTRADAY_RESILIENCE_ABSORPTION": 1.0} # 没有显著回调本身就是一种强势
        total_recovery_score = 0
        for dip_time, dip_row in dips.iterrows():
            recovery_df = df_minute.loc[dip_time:]
            if recovery_df.empty: continue
            final_price = df_minute['close'].iloc[-1]
            recovery_magnitude = (final_price - dip_row['close']) / dip_row['close']
            # 1. 恢复速度分
            recovery_speed_score = recovery_magnitude / len(recovery_df) if len(recovery_df) > 0 else 0
            # 2. 恢复量能分
            recovery_volume_score = recovery_df['amount'].mean() / df_minute['amount'].mean()
            w_speed = get_param_value(params.get('recovery_speed_weight'), 0.5)
            w_vol = get_param_value(params.get('recovery_volume_weight'), 0.5)
            dip_score = recovery_speed_score * w_speed + recovery_volume_score * w_vol
            total_recovery_score += dip_score
        avg_recovery_score = total_recovery_score / len(dips) if len(dips) > 0 else 0
        final_score = normalize_score(pd.Series(avg_recovery_score), window=1, clip_min=0, clip_max=1).iloc[0]
        # --- 探针 ---
        probe_params = get_params_block(self.strategy, 'intraday_battle_report_params.probe_params', {})
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(probe_params.get('enabled'), False) and get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates and not df_minute.empty:
            processed_date_str = df_minute.index[0].strftime('%Y-%m-%d')
            if processed_date_str in probe_dates:
                print(f"      [日内战局探针] _diagnose_resilience_absorption @ {processed_date_str}")
                print(f"        - 识别到 {len(dips)} 个显著回调点")
                print(f"        - 平均恢复得分: {avg_recovery_score:.4f}")
                print(f"        - 最终韧性承接分: {final_score:.4f}")
        return {"SCORE_INTRADAY_RESILIENCE_ABSORPTION": final_score}

    async def _diagnose_battle_tempo(self, df_minute: pd.DataFrame) -> Dict[str, float]:
        """
        【V1.0 · 新增】日内战局之四：诊断“攻防节奏”
        """
        params = get_params_block(self.strategy, 'intraday_battle_report_params.battle_tempo', {})
        if not get_param_value(params.get('enabled'), False):
            return {"SCORE_INTRADAY_BATTLE_TEMPO": 0.0}
        df_minute['price_change'] = df_minute['close'].diff().fillna(0)
        price_change_std = df_minute['price_change'].std()
        volume_mean = df_minute['amount'].mean()
        std_multiplier = get_param_value(params.get('impulse_std_dev_multiplier'), 2.0)
        vol_multiplier = get_param_value(params.get('impulse_volume_multiplier'), 2.0)
        bull_impulse_mask = (df_minute['price_change'] > price_change_std * std_multiplier) & (df_minute['amount'] > volume_mean * vol_multiplier)
        bear_impulse_mask = (df_minute['price_change'] < -price_change_std * std_multiplier) & (df_minute['amount'] > volume_mean * vol_multiplier)
        bull_impulse_power = (df_minute.loc[bull_impulse_mask, 'price_change'] * df_minute.loc[bull_impulse_mask, 'amount']).sum()
        bear_impulse_power = (df_minute.loc[bear_impulse_mask, 'price_change'].abs() * df_minute.loc[bear_impulse_mask, 'amount']).sum()
        total_amount = df_minute['amount'].sum()
        if total_amount == 0:
            return {"SCORE_INTRADAY_BATTLE_TEMPO": 0.0}
        w_bull = get_param_value(params.get('bull_impulse_weight'), 0.6)
        w_bear = get_param_value(params.get('bear_impulse_weight'), 0.4)
        net_impulse_power = (bull_impulse_power * w_bull - bear_impulse_power * w_bear) / total_amount
        final_score = normalize_to_bipolar(pd.Series(net_impulse_power), window=1, sensitivity=0.05).iloc[0]
        # --- 探针 ---
        probe_params = get_params_block(self.strategy, 'intraday_battle_report_params.probe_params', {})
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(probe_params.get('enabled'), False) and get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates and not df_minute.empty:
            processed_date_str = df_minute.index[0].strftime('%Y-%m-%d')
            if processed_date_str in probe_dates:
                print(f"      [日内战局探针] _diagnose_battle_tempo @ {processed_date_str}")
                print(f"        - 多头脉冲分钟数: {bull_impulse_mask.sum()}, 总能量: {bull_impulse_power:.2f}")
                print(f"        - 空头脉冲分钟数: {bear_impulse_mask.sum()}, 总能量: {bear_impulse_power:.2f}")
                print(f"        - 归一化后净脉冲能量: {net_impulse_power:.4f}")
                print(f"        - 最终攻防节奏分: {final_score:.4f}")
        return {"SCORE_INTRADAY_BATTLE_TEMPO": np.clip(final_score, -1, 1)}

