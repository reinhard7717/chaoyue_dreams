# 文件: strategies/trend_following/intelligence/intraday_behavior_engine.py
import asyncio
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Optional, Any
# 导入 get_params_block 工具
from strategies.trend_following.utils import get_params_block, normalize_to_bipolar, get_adaptive_mtf_normalized_bipolar_score, bipolar_to_exclusive_unipolar

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
        【V1.0 · 进攻纯度版】日内战报之一：诊断“进攻纯度”
        - 核心逻辑: 替代旧的“攻击强度”，通过分析全天分钟数据，评估每一次上涨的“质量”。
                    一次高纯度的进攻应是“主力驱动的、高效率的、高紧迫性的”。
        - 聚合方式: 对全天所有上涨分钟的“纯度分”进行成交额加权平均，得出日内总攻质量。
        - 输出: [-1, 1] 的双极性分数。正分代表多方进攻纯度高，负分代表空方进攻纯度高。
        """
        required_signals = ['close', 'amount', 'main_force_ofi', 'retail_ofi', 'buy_quote_exhaustion_rate', 'sell_quote_exhaustion_rate']
        if not self._validate_required_signals(df_minute, required_signals, "_diagnose_offensive_purity"):
            return {"SCORE_INTRADAY_OFFENSIVE_PURITY": 0.0}
        # --- 探针初始化 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        is_probe_enabled = get_param_value(debug_params.get('enable_intraday_behavior_probe'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        last_date_str = self.strategy.df_indicators.index[-1].strftime('%Y-%m-%d')
        is_debug_day = is_debug_enabled and (not probe_dates or last_date_str in probe_dates)
        should_probe = is_debug_day and is_probe_enabled
        # --- 计算分钟级纯度因子 ---
        price_change = self._get_safe_series(df_minute, 'close').diff().fillna(0)
        amount = self._get_safe_series(df_minute, 'amount').replace(0, 1e-9)
        # 效率: 价格变化 / 成交额
        efficiency = price_change / amount
        norm_efficiency = normalize_to_bipolar(efficiency, df_minute.index, window=240, sensitivity=0.1)
        # 驱动力: 主力OFI vs 散户OFI
        mf_ofi = self._get_safe_series(df_minute, 'main_force_ofi')
        retail_ofi = self._get_safe_series(df_minute, 'retail_ofi')
        driver = mf_ofi - retail_ofi
        norm_driver = normalize_to_bipolar(driver, df_minute.index, window=240)
        # 紧迫性: 买方扫单 vs 卖方扫单
        buy_urgency = self._get_safe_series(df_minute, 'buy_quote_exhaustion_rate')
        sell_urgency = self._get_safe_series(df_minute, 'sell_quote_exhaustion_rate')
        urgency = buy_urgency - sell_urgency
        norm_urgency = normalize_to_bipolar(urgency, df_minute.index, window=240)
        # --- 融合为分钟级纯度分 ---
        purity_score = (norm_efficiency.pow(0.2) * norm_driver.pow(0.5) * norm_urgency.pow(0.3)).fillna(0)
        # --- 按日内多空进攻分别聚合 ---
        bullish_minutes = purity_score[price_change > 0]
        bullish_weights = amount[price_change > 0]
        bearish_minutes = purity_score[price_change < 0].abs()
        bearish_weights = amount[price_change < 0]
        # 加权平均
        avg_bullish_purity = np.average(bullish_minutes, weights=bullish_weights) if not bullish_minutes.empty and bullish_weights.sum() > 0 else 0
        avg_bearish_purity = np.average(bearish_minutes, weights=bearish_weights) if not bearish_minutes.empty and bearish_weights.sum() > 0 else 0
        final_score = (avg_bullish_purity - avg_bearish_purity)
        # --- 探针监测 ---
        if should_probe:
            print(f"      [日内行为探针] _diagnose_offensive_purity @ {last_date_str}")
            print(f"        - 日内平均多头进攻纯度: {avg_bullish_purity:.4f}")
            print(f"        - 日内平均空头进攻纯度: {avg_bearish_purity:.4f}")
            print(f"        - 最终进攻纯度分: {final_score:.4f}")
        return {"SCORE_INTRADAY_OFFENSIVE_PURITY": final_score}

    async def _diagnose_dominance_consensus(self, df_minute: pd.DataFrame) -> Dict[str, float]:
        """
        【V1.0 · 支配共识版】日内战报之二：诊断“支配共识”
        - 核心逻辑: 替代旧的“控制能力”，评估主力对盘面的“动态支配力”以及这种支配是否在盘中得到了“共识性加强”。
        - 诊断双要素: 1. 支配强度 (量能加权的VWAP偏离度); 2. 共识趋势 (支配强度的移动平均趋势)。
        - 聚合方式: 融合全天平均支配强度与收盘前的共识趋势，得出综合支配力评分。
        - 输出: [-1, 1] 的双极性分数。正分代表多头支配且共识加强，负分代表空头支配。
        """
        required_signals = ['close', 'vwap', 'amount']
        if not self._validate_required_signals(df_minute, required_signals, "_diagnose_dominance_consensus"):
            return {"SCORE_INTRADAY_DOMINANCE_CONSENSUS": 0.0}
        # --- 探针初始化 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        last_date_str = self.strategy.df_indicators.index[-1].strftime('%Y-%m-%d')
        is_debug_day = is_debug_enabled and (not probe_dates or last_date_str in probe_dates)
        # --- 计算分钟级支配强度 ---
        vwap = self._get_safe_series(df_minute, 'vwap').replace(0, np.nan).ffill()
        price_deviation = (self._get_safe_series(df_minute, 'close') - vwap) / vwap
        amount_ratio = self._get_safe_series(df_minute, 'amount') / self._get_safe_series(df_minute, 'amount').mean()
        dominance_strength = (price_deviation * amount_ratio).fillna(0)
        norm_dominance_strength = normalize_to_bipolar(dominance_strength, df_minute.index, window=240, sensitivity=0.5)
        # --- 计算共识趋势 ---
        consensus_trend = norm_dominance_strength.ewm(span=21, adjust=False).mean()
        # --- 聚合为日内总分 ---
        avg_strength = norm_dominance_strength.mean()
        final_trend = consensus_trend.iloc[-1] if not consensus_trend.empty else 0
        final_score = (avg_strength * 0.5 + final_trend * 0.5)
        # --- 探针监测 ---
        if is_debug_day:
            print(f"      [日内行为探针] _diagnose_dominance_consensus @ {last_date_str}")
            print(f"        - 全天平均支配强度: {avg_strength:.4f}")
            print(f"        - 收盘共识趋势: {final_trend:.4f}")
            print(f"        - 最终支配共识分: {final_score:.4f}")
        return {"SCORE_INTRADAY_DOMINANCE_CONSENSUS": final_score}

    async def _diagnose_conviction_reversal(self, df_minute: pd.DataFrame) -> Dict[str, float]:
        """
        【V1.0 · 信念反转版】日内战报之三：诊断“信念反转”
        - 核心逻辑: 替代旧的“博弈转折”，抛弃KDJ，寻找由“一方力量衰竭”和“另一方信念入场”
                    共同构成的“高置信度”转折点。
        - 诊断双要素: 1. 看涨转折(恐慌抛售+主力强力承接); 2. 看跌转折(拉高派发+主力信念动摇)。
        - 聚合方式: 取全天最强的看涨转折证据与最强的看跌转折证据的差值，得出净反转倾向。
        - 输出: [-1, 1] 的双极性分数。正分代表日内发生过强烈的看涨反转，负分则代表看跌反转。
        """
        # 注意：此方法依赖的信号是日线级别的，我们需要将其广播到分钟线上进行匹配
        daily_signals = self.strategy.df_indicators.iloc[-1]
        # --- 探针初始化 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        last_date_str = self.strategy.df_indicators.index[-1].strftime('%Y-%m-%d')
        is_debug_day = is_debug_enabled and (not probe_dates or last_date_str in probe_dates)
        # --- 看涨转折证据 ---
        panic_score = daily_signals.get('panic_selling_cascade_D', 0.0)
        absorption_score = daily_signals.get('capitulation_absorption_index_D', 0.0)
        mf_alpha_bullish = max(daily_signals.get('main_force_execution_alpha_D', 0.0), 0.0)
        # 归一化处理（此处简化为直接使用，实际可做更复杂的归一化）
        bullish_reversal_evidence = (panic_score * absorption_score * mf_alpha_bullish).pow(1/3)
        # --- 看跌转折证据 ---
        distribution_score = daily_signals.get('rally_distribution_pressure_D', 0.0)
        conviction_decay = max(0, -daily_signals.get('main_force_conviction_index_D_slope_5d', 0.0)) # 信念指数5日斜率为负
        mf_alpha_bearish = abs(min(daily_signals.get('main_force_execution_alpha_D', 0.0), 0.0))
        bearish_reversal_evidence = (distribution_score * conviction_decay * mf_alpha_bearish).pow(1/3)
        final_score = bullish_reversal_evidence - bearish_reversal_evidence
        # --- 探针监测 ---
        if is_debug_day:
            print(f"      [日内行为探针] _diagnose_conviction_reversal @ {last_date_str}")
            print(f"        - 看涨证据: 恐慌={panic_score:.2f}, 承接={absorption_score:.2f}, 主力Alpha+={mf_alpha_bullish:.2f} -> 综合={bullish_reversal_evidence:.4f}")
            print(f"        - 看跌证据: 派发={distribution_score:.2f}, 信念衰减={conviction_decay:.2f}, 主力Alpha-={mf_alpha_bearish:.2f} -> 综合={bearish_reversal_evidence:.4f}")
            print(f"        - 最终信念反转分: {final_score:.4f}")
        return {"SCORE_INTRADAY_CONVICTION_REVERSAL": np.clip(final_score, -1, 1)}



