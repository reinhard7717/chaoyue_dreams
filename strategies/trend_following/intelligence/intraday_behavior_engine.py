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
        # [修改代码行] 移除不再需要的KDJ计算
        # VWAP (为公理二：支配共识服务)
        df_enriched = await self.calculator.calculate_vwap(df_enriched)
        return df_enriched

    async def run_intraday_diagnostics(self, df_minute: pd.DataFrame) -> Dict[str, float]:
        """
        【V3.0 · 日内战报三部曲版】日内诊断总指挥
        - 核心流程:
          1. 准备分钟级核心数据 (VWAP)。
          2. 并行诊断“进攻纯度”、“支配共识”、“信念反转”三大战报。
          3. 返回所有战报信号的最终值，为日线级分析提供过程性解释。
        """
        print("启动【V3.0 · 日内战报三部曲版】日内行为诊断...")
        df_enriched = await self._prepare_intraday_indicators(df_minute)
        if df_enriched is None or df_enriched.empty:
            print("分钟数据为空，无法进行日内行为诊断。")
            return {
                "SCORE_INTRADAY_OFFENSIVE_PURITY": 0.0,
                "SCORE_INTRADAY_DOMINANCE_CONSENSUS": 0.0,
                "SCORE_INTRADAY_CONVICTION_REVERSAL": 0.0,
            }
        # [修改代码行] 并行执行三大新战报的诊断
        tasks = [
            self._diagnose_offensive_purity(df_enriched),
            self._diagnose_dominance_consensus(df_enriched),
            self._diagnose_conviction_reversal(df_enriched),
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
        # --- [修改代码块] 重构探针逻辑以适配历史回溯 ---
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
        # --- [修改代码块] 重构探针逻辑以适配历史回溯 ---
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
        【V1.2 · 探针逻辑重构版】日内战报之三：诊断“信念反转”
        - 核心重构: 重构探针逻辑，使其能正确识别当前处理的数据段是否覆盖了探针日期，
                      从而在历史回溯时也能准确触发。
        """
        daily_signals = self.strategy.df_indicators.iloc[-1]
        panic_score = daily_signals.get('panic_selling_cascade_D', 0.0)
        absorption_score = daily_signals.get('capitulation_absorption_index_D', 0.0)
        mf_alpha_bullish = max(daily_signals.get('main_force_execution_alpha_D', 0.0), 0.0)
        bullish_reversal_evidence = (panic_score * absorption_score * mf_alpha_bullish).pow(1/3)
        distribution_score = daily_signals.get('rally_distribution_pressure_D', 0.0)
        conviction_decay = max(0, -daily_signals.get('main_force_conviction_index_D_slope_5d', 0.0))
        mf_alpha_bearish = abs(min(daily_signals.get('main_force_execution_alpha_D', 0.0), 0.0))
        bearish_reversal_evidence = (distribution_score * conviction_decay * mf_alpha_bearish).pow(1/3)
        final_score = bullish_reversal_evidence - bearish_reversal_evidence
        # --- [修改代码块] 重构探针逻辑以适配历史回溯 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates and not df_minute.empty:
            processed_date_str = df_minute.index[0].strftime('%Y-%m-%d')
            if processed_date_str in probe_dates:
                print(f"      [日内行为探针] _diagnose_conviction_reversal @ {processed_date_str}")
                print(f"        - 看涨证据: 恐慌={panic_score:.2f}, 承接={absorption_score:.2f}, 主力Alpha+={mf_alpha_bullish:.2f} -> 综合={bullish_reversal_evidence:.4f}")
                print(f"        - 看跌证据: 派发={distribution_score:.2f}, 信念衰减={conviction_decay:.2f}, 主力Alpha-={mf_alpha_bearish:.2f} -> 综合={bearish_reversal_evidence:.4f}")
                print(f"        - 最终信念反转分: {final_score:.4f}")
        return {"SCORE_INTRADAY_CONVICTION_REVERSAL": np.clip(final_score, -1, 1)}



