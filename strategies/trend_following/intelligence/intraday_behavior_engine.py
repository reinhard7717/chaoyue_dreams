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
    async def _prepare_intraday_indicators(self, df_minute: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        【V2.0 · 公理化精简版】
        统一为分钟数据计算所有必需的战术指标，只保留公理模型必需的VWAP和KDJ。
        """
        if df_minute is None or df_minute.empty:
            return None
        df_enriched = df_minute.copy()
        calc_tasks = []
        # VWAP (为公理二：控制能力服务)
        # VWAP通常是数据源直接提供或简单计算，这里假设calculator有此功能
        calc_tasks.append(self.calculator.calculate_vwap(df_enriched))
        # KDJ (为公理三：博弈转折服务)
        kdj_params = self.params.get('kdj_params', {})
        if kdj_params.get('enabled'):
            calc_tasks.append(self.calculator.calculate_kdj(df_enriched, kdj_params['period'], kdj_params['signal_period'], kdj_params['smooth_k_period']))
        results = await asyncio.gather(*calc_tasks)
        for res_df in results:
            if res_df is not None and not res_df.empty:
                df_enriched = df_enriched.join(res_df, how='left')
        return df_enriched
    async def run_intraday_diagnostics(self, df_minute: pd.DataFrame) -> Dict[str, float]:
        """
        【V2.0 · 三大公理重构版】日内诊断总指挥
        - 核心流程:
          1. 计算公理模型必需的分钟级指标 (VWAP, KDJ)。
          2. 并行诊断三大公理，生成纯粹的日内行为原子信号。
          3. 返回所有原子信号的最新值。
        """
        print("启动【V2.0 · 三大公理重构版】日内行为诊断...")
        df_enriched = await self._prepare_intraday_indicators(df_minute)
        if df_enriched is None or df_enriched.empty:
            print("分钟数据为空，无法进行日内行为诊断。")
            return {
                "SCORE_INTRADAY_AXIOM_ATTACK": 0.0,
                "SCORE_INTRADAY_AXIOM_CONTROL": 0.0,
                "SCORE_INTRADAY_AXIOM_TURNING": 0.0,
            }
        # 并行执行三大公理的诊断
        tasks = [
            self._diagnose_axiom_attack(df_enriched),
            self._diagnose_axiom_control(df_enriched),
            self._diagnose_axiom_turning(df_enriched),
        ]
        results = await asyncio.gather(*tasks)
        final_scores = {}
        for res_dict in results:
            final_scores.update(res_dict)
        print(f"日内行为诊断完成: {final_scores}")
        return final_scores
    async def _diagnose_axiom_attack(self, df_minute: pd.DataFrame) -> Dict[str, float]:
        """
        【V1.1 · 归一化窗口参数化版】日内行为公理一：诊断“攻击强度”
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `normalize_to_bipolar` 的 `window` 参数改为从配置中获取的 `norm_window`。
        """
        # 攻击强度 = K线实体方向与大小 * 成交额变化
        # 1. K线实体强度 (归一化到 [-1, 1])
        body = self._get_safe_series(df_minute, 'close', method_name="_diagnose_axiom_attack") - self._get_safe_series(df_minute, 'open', method_name="_diagnose_axiom_attack")
        body_range = self._get_safe_series(df_minute, 'high', method_name="_diagnose_axiom_attack") - self._get_safe_series(df_minute, 'low', method_name="_diagnose_axiom_attack")
        body_strength = (body / body_range.replace(0, np.nan)).fillna(0)
        # 2. 成交额强度 (相比近期均值的倍数，归一化到 [0, N])
        amount_ma = self._get_safe_series(df_minute, 'amount', method_name="_diagnose_axiom_attack").rolling(window=21, min_periods=1).mean()
        amount_strength = (self._get_safe_series(df_minute, 'amount', method_name="_diagnose_axiom_attack") / amount_ma.replace(0, np.nan)).fillna(1.0)
        # 3. 融合：实体强度 * 成交额强度
        raw_attack_score = body_strength * amount_strength
        # 4. 使用双极归一化，得到最终的攻击强度分
        # 【优化】使用配置中的 norm_window
        norm_window = self.params.get('meta_analysis_params', {}).get('norm_window', 55)
        final_score = normalize_to_bipolar(raw_attack_score, df_minute.index, window=norm_window).iloc[-1]
        return {"SCORE_INTRADAY_AXIOM_ATTACK": final_score}
    async def _diagnose_axiom_control(self, df_minute: pd.DataFrame) -> Dict[str, float]:
        """
        【V1.1 · 归一化窗口参数化版】日内行为公理二：诊断“控制能力”
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `normalize_to_bipolar` 的 `window` 参数改为从配置中获取的 `norm_window`。
        """
        if 'vwap' not in df_minute.columns:
            print(f"    -> [日内行为情报警告] 方法 '_diagnose_axiom_control' 缺少数据 'vwap'，使用默认值 0.0。")
            return {"SCORE_INTRADAY_AXIOM_CONTROL": 0.0}
        # 控制能力 = (收盘价 - VWAP) / VWAP
        # 正值代表多头控盘，负值代表空头控盘
        raw_control_score = (self._get_safe_series(df_minute, 'close', method_name="_diagnose_axiom_control") - self._get_safe_series(df_minute, 'vwap', method_name="_diagnose_axiom_control")) / self._get_safe_series(df_minute, 'vwap', method_name="_diagnose_axiom_control")
        # 使用双极归一化，得到最终的控制能力分
        # 【优化】使用配置中的 norm_window
        norm_window = self.params.get('meta_analysis_params', {}).get('norm_window', 55)
        final_score = normalize_to_bipolar(raw_control_score, df_minute.index, window=norm_window, sensitivity=2.0).iloc[-1]
        return {"SCORE_INTRADAY_AXIOM_CONTROL": final_score}
    async def _diagnose_axiom_turning(self, df_minute: pd.DataFrame) -> Dict[str, float]:
        """
        【V1.0】日内行为公理三：诊断“博弈转折”
        - 核心修复: 增加对所有依赖数据的存在性检查。
        """
        kdj_params = self.params.get('kdj_params', {})
        k_col = f"K_{kdj_params.get('period', 13)}_{kdj_params.get('signal_period', 5)}_{kdj_params.get('smooth_k_period', 3)}"
        d_col = f"D_{kdj_params.get('period', 13)}_{kdj_params.get('signal_period', 5)}_{kdj_params.get('smooth_k_period', 3)}"
        j_col = f"J_{kdj_params.get('period', 13)}_{kdj_params.get('signal_period', 5)}_{kdj_params.get('smooth_k_period', 3)}"
        if not all(c in df_minute.columns for c in [k_col, d_col, j_col]):
            print(f"    -> [日内行为情报警告] 方法 '_diagnose_axiom_turning' 缺少KDJ相关数据 '{k_col}', '{d_col}', '{j_col}'，使用默认值 0.0。")
            return {"SCORE_INTRADAY_AXIOM_TURNING": 0.0}
        # 1. 看涨转折信号：KDJ在超卖区金叉
        is_oversold = (self._get_safe_series(df_minute, j_col, method_name="_diagnose_axiom_turning") < 20)
        is_golden_cross = (self._get_safe_series(df_minute, k_col, method_name="_diagnose_axiom_turning") > self._get_safe_series(df_minute, d_col, method_name="_diagnose_axiom_turning")) & (self._get_safe_series(df_minute, k_col, method_name="_diagnose_axiom_turning").shift(1) <= self._get_safe_series(df_minute, d_col, method_name="_diagnose_axiom_turning").shift(1))
        bullish_turn_signal = (is_oversold & is_golden_cross).astype(float)
        # 2. 看跌转折信号：KDJ在超买区死叉
        is_overbought = (self._get_safe_series(df_minute, j_col, method_name="_diagnose_axiom_turning") > 80)
        is_dead_cross = (self._get_safe_series(df_minute, k_col, method_name="_diagnose_axiom_turning") < self._get_safe_series(df_minute, d_col, method_name="_diagnose_axiom_turning")) & (self._get_safe_series(df_minute, k_col, method_name="_diagnose_axiom_turning").shift(1) >= self._get_safe_series(df_minute, d_col, method_name="_diagnose_axiom_turning").shift(1))
        bearish_turn_signal = (is_overbought & is_dead_cross).astype(float)
        # 3. 融合为双极性分数
        # 只取最后一次信号
        final_score = 0.0
        if bullish_turn_signal.iloc[-1] > 0:
            final_score = 1.0
        elif bearish_turn_signal.iloc[-1] > 0:
            final_score = -1.0
        return {"SCORE_INTRADAY_AXIOM_TURNING": final_score}



