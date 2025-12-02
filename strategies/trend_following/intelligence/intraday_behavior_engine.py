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
    【V4.0 · 日内诡道引擎版】
    - 核心升级: 在“日内叙事”基础上，引入基于主力诡道博弈的“伏击与侧翼”、“终末强袭”、“VWAP攻防”三大全新公理。
                旨在穿透全天战果的表象，深度解读主力资金在日内的完整战术剧本，为T+1决策提供更高维度的博弈洞察。
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
        【V4.0 · 日内诡道引擎版】日内诊断总指挥
        - 核心流程:
          1. 准备分钟级核心数据 (VWAP)。
          2. 并行诊断“进攻纯度”、“支配共识”、“信念反转”三大经典战报。
          3. 并行诊断“战术弧线”、“竞价意图”、“修复质量”三大叙事信号。
          4. [新增] 并行诊断“伏击与侧翼”、“终末强袭”、“VWAP攻防”三大诡道信号。
          5. 返回所有信号的最终值，为日线级分析提供更全面的过程性解释。
        """
        print("启动【V4.0 · 日内诡道引擎版】日内行为诊断...") # [代码修改] 更新版本号和描述
        df_enriched = await self._prepare_intraday_indicators(df_minute)
        if df_enriched is None or df_enriched.empty:
            print("分钟数据为空，无法进行日内行为诊断。")
            return {
                "SCORE_INTRADAY_OFFENSIVE_PURITY": 0.0,
                "SCORE_INTRADAY_DOMINANCE_CONSENSUS": 0.0,
                "SCORE_INTRADAY_CONVICTION_REVERSAL": 0.0,
                "SCORE_INTRADAY_TACTICAL_ARC": 0.0,
                "SCORE_INTRADAY_AUCTION_INTENT": 0.0,
                "SCORE_INTRADAY_RECOVERY_QUALITY": 0.0,
                "SCORE_INTRADAY_AMBUSH_AND_FLANK": 0.0, # [代码新增]
                "SCORE_INTRADAY_FINAL_ASSAULT": 0.0, # [代码新增]
                "SCORE_INTRADAY_VWAP_BATTLEFIELD": 0.0, # [代码新增]
            }
        # [代码修改] 并行执行所有新旧诊断任务
        tasks = [
            self._diagnose_offensive_purity(df_enriched),
            self._diagnose_dominance_consensus(df_enriched),
            self._diagnose_conviction_reversal(df_enriched),
            self._diagnose_tactical_arc(df_enriched),
            self._diagnose_auction_intent(df_enriched),
            self._diagnose_recovery_quality(df_enriched),
            self._diagnose_ambush_and_flank(df_enriched), # [代码新增] 调用新增的“伏击与侧翼”诊断方法
            self._diagnose_final_assault(df_enriched), # [代码新增] 调用新增的“终末强袭”诊断方法
            self._diagnose_vwap_battlefield(df_enriched), # [代码新增] 调用新增的“VWAP攻防”诊断方法
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
        【V1.4 · 信号引用修正版】日内战报之三：诊断“信念反转”
        - 核心修正: 修正了信念衰减计算中的信号引用错误。当5日斜率信号不存在时，
                      不再尝试获取一个不存在的1日斜率信号，而是明确地将衰减度置为0，
                      使代码逻辑更健壮、意图更清晰。
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
        bullish_reversal_evidence = (panic_score * absorption_score * bullish_alpha_score)**(1/3) # [代码修改] 使用 **(1/3) 代替 .pow()
        distribution_score = daily_signals.get('rally_distribution_pressure_D', 0.0)
        # [代码修改开始] 修正信念衰减的计算逻辑
        conviction_slope_5d = daily_signals.get('SLOPE_5_main_force_conviction_index_D', None)
        conviction_decay_source = "5D_SLOPE" # [代码新增] 用于探针
        if pd.isna(conviction_slope_5d):
             conviction_decay = 0.0 # [代码修改] 当5日斜率无效时，明确将衰减设为0
             conviction_decay_source = "FALLBACK_ZERO" # [代码新增] 用于探针
        else:
             conviction_decay = max(0, -conviction_slope_5d) # [代码修改] 正常计算
        # [代码修改结束]
        mf_alpha_bearish = abs(min(mf_alpha_raw, 0.0))
        bearish_reversal_evidence = (distribution_score * conviction_decay * mf_alpha_bearish)**(1/3) # [代码修改] 使用 **(1/3) 代替 .pow()
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
                # [代码修改] 更新探针输出，明确衰减来源
                print(f"        - 看跌证据: 派发={distribution_score:.2f}, 信念衰减={conviction_decay:.2f} (来源: {conviction_decay_source}), 主力Alpha-={mf_alpha_bearish:.2f} -> 综合={bearish_reversal_evidence:.4f}")
                print(f"        - 最终信念反转分: {final_score:.4f}")
        return {"SCORE_INTRADAY_CONVICTION_REVERSAL": np.clip(final_score, -1, 1)}

    async def _diagnose_tactical_arc(self, df_minute: pd.DataFrame) -> Dict[str, float]:
        """
        【V1.0 · 新增】日内叙事之一：诊断“战术弧线”
        - 核心逻辑: 对比上下午的“支配强度”，判断力量消长。
        """
        required_signals = ['close', 'vwap', 'amount']
        if not self._validate_required_signals(df_minute, required_signals, "_diagnose_tactical_arc"):
            return {"SCORE_INTRADAY_TACTICAL_ARC": 0.0}
        vwap = self._get_safe_series(df_minute, 'vwap').replace(0, np.nan).ffill()
        price_deviation = (self._get_safe_series(df_minute, 'close') - vwap) / vwap
        dominance_strength = (price_deviation * self._get_safe_series(df_minute, 'amount')).fillna(0)
        # A股市场，通常13:00之后为下午盘
        am_dominance = dominance_strength[df_minute.index.hour < 13]
        pm_dominance = dominance_strength[df_minute.index.hour >= 13]
        am_avg_dominance = am_dominance.mean() if not am_dominance.empty else 0
        pm_avg_dominance = pm_dominance.mean() if not pm_dominance.empty else 0
        # 计算原始分，代表力量的净变化方向
        raw_score = pm_avg_dominance - am_avg_dominance
        # 归一化处理
        total_abs_dominance = dominance_strength.abs().mean()
        final_score = raw_score / (total_abs_dominance + 1e-9)
        final_score = np.tanh(final_score) # 使用tanh平滑到[-1, 1]
        # --- 探针监测 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates and not df_minute.empty:
            processed_date_str = df_minute.index[0].strftime('%Y-%m-%d')
            if processed_date_str in probe_dates:
                print(f"      [日内行为探针] _diagnose_tactical_arc @ {processed_date_str}")
                print(f"        - 上午平均支配强度: {am_avg_dominance:.2f}")
                print(f"        - 下午平均支配强度: {pm_avg_dominance:.2f}")
                print(f"        - 原始分 (PM - AM): {raw_score:.2f}")
                print(f"        - 最终战术弧线分: {final_score:.4f}")
        return {"SCORE_INTRADAY_TACTICAL_ARC": final_score}

    async def _diagnose_auction_intent(self, df_minute: pd.DataFrame) -> Dict[str, float]:
        """
        【V1.0 · 新增】日内叙事之二：诊断“竞价意图”
        - 核心逻辑: 融合开盘博弈与收盘偷袭的日线级信号。
        """
        if df_minute.empty:
            return {"SCORE_INTRADAY_AUCTION_INTENT": 0.0}
        current_date = df_minute.index[0].normalize()
        if current_date not in self.strategy.df_indicators.index:
            print(f"    -> [日内行为情报警告] _diagnose_auction_intent: 在日线数据中未找到日期 {current_date}，跳过计算。")
            return {"SCORE_INTRADAY_AUCTION_INTENT": 0.0}
        daily_signals = self.strategy.df_indicators.loc[current_date]
        # 从日线数据中获取开盘和收盘的博弈信号
        opening_intent = daily_signals.get('opening_battle_result_D', 0.0)
        closing_intent = daily_signals.get('closing_auction_ambush_D', 0.0)
        # 从配置中获取权重
        params = get_params_block(self.strategy, 'intraday_narrative_engine_params', {})
        weights = get_param_value(params.get('auction_intent_weights'), {'opening': 0.4, 'closing': 0.6})
        # 加权融合
        final_score = (opening_intent * weights.get('opening', 0.4) +
                       closing_intent * weights.get('closing', 0.6))
        # --- 探针监测 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates:
            processed_date_str = current_date.strftime('%Y-%m-%d')
            if processed_date_str in probe_dates:
                print(f"      [日内行为探针] _diagnose_auction_intent @ {processed_date_str}")
                print(f"        - 开盘博弈结果分: {opening_intent:.4f}")
                print(f"        - 收盘竞价偷袭分: {closing_intent:.4f}")
                print(f"        - 最终竞价意图分: {final_score:.4f}")
        return {"SCORE_INTRADAY_AUCTION_INTENT": np.clip(final_score, -1, 1)}

    async def _diagnose_recovery_quality(self, df_minute: pd.DataFrame) -> Dict[str, float]:
        """
        【V1.0 · 新增】日内叙事之三：诊断“修复质量”
        - 核心逻辑: 从幅度、阵地、能量三个维度评估V型反转的含金量。
        """
        if df_minute.empty or 'vwap' not in df_minute.columns:
            return {"SCORE_INTRADAY_RECOVERY_QUALITY": 0.0}
        # 获取当日OHLC和VWAP
        daily_open = df_minute['open'].iloc[0]
        daily_high = df_minute['high'].max()
        daily_low = df_minute['low'].min()
        daily_close = df_minute['close'].iloc[-1]
        daily_vwap = df_minute['vwap'].iloc[-1]
        # 从配置中获取参数
        params = get_params_block(self.strategy, 'intraday_narrative_engine_params', {})
        gate_threshold = get_param_value(params.get('recovery_gate_threshold_pct'), {'value': 0.04}).get('value')
        weights = get_param_value(params.get('recovery_quality_weights'), {'magnitude': 0.4, 'vwap_reclaim': 0.4, 'volume_confirm': 0.2})
        # 门控条件：当日必须有足够的振幅才进行评估
        day_range = daily_high - daily_low
        if day_range / daily_open < gate_threshold:
            return {"SCORE_INTRADAY_RECOVERY_QUALITY": 0.0}
        # 1. 修复幅度分 (0-1)
        magnitude_score = (daily_close - daily_low) / (day_range + 1e-9)
        # 2. 阵地收复分 (0或1)
        vwap_reclaim_score = 1.0 if daily_close > daily_vwap else 0.0
        # 3. 能量确认分
        am_volume = df_minute['volume'][df_minute.index.hour < 13].sum()
        pm_volume = df_minute['volume'][df_minute.index.hour >= 13].sum()
        # 修复通常发生在下午，如果下午成交量显著大于上午，说明修复有能量支持
        volume_confirm_score = normalize_score(pm_volume / (am_volume + 1e-9), 0.8, 1.5) # 0.8倍是中性，1.5倍以上是强力
        # 加权融合
        final_score = (magnitude_score * weights.get('magnitude', 0.4) +
                       vwap_reclaim_score * weights.get('vwap_reclaim', 0.4) +
                       volume_confirm_score * weights.get('volume_confirm', 0.2))
        # --- 探针监测 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates:
            processed_date_str = df_minute.index[0].strftime('%Y-%m-%d')
            if processed_date_str in probe_dates:
                print(f"      [日内行为探针] _diagnose_recovery_quality @ {processed_date_str}")
                print(f"        - 修复幅度分: {magnitude_score:.4f}")
                print(f"        - VWAP收复分: {vwap_reclaim_score:.4f}")
                print(f"        - 成交量确认分: {volume_confirm_score:.4f} (PM Vol: {pm_volume}, AM Vol: {am_volume})")
                print(f"        - 最终修复质量分: {final_score:.4f}")
        return {"SCORE_INTRADAY_RECOVERY_QUALITY": np.clip(final_score, 0, 1)}

    async def _diagnose_ambush_and_flank(self, df_minute: pd.DataFrame) -> Dict[str, float]:
        """
        【V1.0 · 新增】日内诡道之一：诊断“伏击与侧翼”
        - 核心逻辑: 识别主力利用日内恐慌进行战术洗盘并吸收筹码的剧本。
        """
        if df_minute.empty:
            return {"SCORE_INTRADAY_AMBUSH_AND_FLANK": 0.0}
        current_date = df_minute.index[0].normalize()
        if current_date not in self.strategy.df_indicators.index:
            return {"SCORE_INTRADAY_AMBUSH_AND_FLANK": 0.0}
        daily_signals = self.strategy.df_indicators.loc[current_date]
        params = get_params_block(self.strategy, 'intraday_gambit_engine_params', {}).get('ambush_flank_params', {})
        weights = params.get('fusion_weights', {'panic_evidence': 0.2, 'absorption_power': 0.4, 'recovery_strength': 0.4})
        min_dip_pct = params.get('min_dip_to_open_pct', 0.03)
        # 门控条件：必须有足够深度的下探
        daily_open = daily_signals.get('open_D', 0)
        daily_low = daily_signals.get('low_D', 0)
        if daily_open == 0 or ((daily_open - daily_low) / daily_open) < min_dip_pct:
            return {"SCORE_INTRADAY_AMBUSH_AND_FLANK": 0.0}
        # 1. 恐慌证据 (Panic Evidence)
        panic_evidence = daily_signals.get('panic_selling_cascade_D', 0.0)
        # 2. 主力吸收 (Absorption Power)
        absorption_power = daily_signals.get('dip_absorption_power_D', 0.0)
        # 3. 反攻质量 (Recovery Strength)
        recovery_strength = daily_signals.get('closing_strength_index_D', 0.0)
        # 融合三大证据链
        final_score = (panic_evidence ** weights.get('panic_evidence', 0.2) *
                       absorption_power ** weights.get('absorption_power', 0.4) *
                       recovery_strength ** weights.get('recovery_strength', 0.4))
        # --- 探针监测 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates:
            processed_date_str = current_date.strftime('%Y-%m-%d')
            if processed_date_str in probe_dates:
                print(f"      [日内诡道探针] _diagnose_ambush_and_flank @ {processed_date_str}")
                print(f"        - 恐慌证据分: {panic_evidence:.4f}")
                print(f"        - 主力吸收分: {absorption_power:.4f}")
                print(f"        - 反攻质量分: {recovery_strength:.4f}")
                print(f"        - 最终伏击与侧翼分: {final_score:.4f}")
        return {"SCORE_INTRADAY_AMBUSH_AND_FLANK": np.clip(final_score, 0, 1)}

    async def _diagnose_final_assault(self, df_minute: pd.DataFrame) -> Dict[str, float]:
        """
        【V1.0 · 新增】日内诡道之二：诊断“终末强袭”
        - 核心逻辑: 捕捉主力在尾盘（含集合竞价）的真实攻击或撤退意图。
        """
        required_signals = ['close', 'vwap', 'amount']
        if not self._validate_required_signals(df_minute, required_signals, "_diagnose_final_assault"):
            return {"SCORE_INTRADAY_FINAL_ASSAULT": 0.0}
        current_date = df_minute.index[0].normalize()
        if current_date not in self.strategy.df_indicators.index:
            return {"SCORE_INTRADAY_FINAL_ASSAULT": 0.0}
        daily_signals = self.strategy.df_indicators.loc[current_date]
        params = get_params_block(self.strategy, 'intraday_gambit_engine_params', {}).get('final_assault_params', {})
        weights = params.get('fusion_weights', {'assault_strength': 0.5, 'assault_accel': 0.2, 'closing_auction': 0.3})
        start_time = params.get('start_time', "14:30")
        # 1. 尾盘攻击强度 (Assault Strength)
        final_period_df = df_minute.between_time(start_time, '15:00')
        if final_period_df.empty:
            return {"SCORE_INTRADAY_FINAL_ASSAULT": 0.0}
        vwap = self._get_safe_series(final_period_df, 'vwap').replace(0, np.nan).ffill()
        price_deviation = (self._get_safe_series(final_period_df, 'close') - vwap) / vwap
        dominance_strength = (price_deviation * self._get_safe_series(final_period_df, 'amount')).fillna(0)
        assault_strength = dominance_strength.mean()
        norm_assault_strength = np.tanh(assault_strength / (dominance_strength.abs().mean() + 1e-9))
        # 2. 攻击加速度 (Assault Acceleration)
        strength_trend = dominance_strength.ewm(span=5).mean().diff().mean()
        norm_assault_accel = np.tanh(strength_trend / (dominance_strength.abs().diff().mean() + 1e-9))
        # 3. 收盘竞价意图 (Closing Auction)
        closing_auction_intent = daily_signals.get('closing_auction_ambush_D', 0.0)
        # 融合
        final_score = (norm_assault_strength * weights.get('assault_strength', 0.5) +
                       norm_assault_accel * weights.get('assault_accel', 0.2) +
                       closing_auction_intent * weights.get('closing_auction', 0.3))
        # --- 探针监测 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates:
            processed_date_str = current_date.strftime('%Y-%m-%d')
            if processed_date_str in probe_dates:
                print(f"      [日内诡道探针] _diagnose_final_assault @ {processed_date_str}")
                print(f"        - 尾盘攻击强度分: {norm_assault_strength:.4f} (原始均值: {assault_strength:.2f})")
                print(f"        - 尾盘攻击加速分: {norm_assault_accel:.4f} (原始趋势: {strength_trend:.2f})")
                print(f"        - 收盘竞价意图分: {closing_auction_intent:.4f}")
                print(f"        - 最终终末强袭分: {final_score:.4f}")
        return {"SCORE_INTRADAY_FINAL_ASSAULT": np.clip(final_score, -1, 1)}

    async def _diagnose_vwap_battlefield(self, df_minute: pd.DataFrame) -> Dict[str, float]:
        """
        【V1.0 · 新增】日内诡道之三：诊断“VWAP攻防”
        - 核心逻辑: 将VWAP视为多空战场，量化支撑与压制的力量对比。
        """
        required_signals = ['close', 'low', 'high', 'vwap', 'volume']
        if not self._validate_required_signals(df_minute, required_signals, "_diagnose_vwap_battlefield"):
            return {"SCORE_INTRADAY_VWAP_BATTLEFIELD": 0.0}
        current_date = df_minute.index[0].normalize()
        if current_date not in self.strategy.df_indicators.index:
            return {"SCORE_INTRADAY_VWAP_BATTLEFIELD": 0.0}
        daily_signals = self.strategy.df_indicators.loc[current_date]
        params = get_params_block(self.strategy, 'intraday_gambit_engine_params', {}).get('vwap_battlefield_params', {})
        weights = params.get('fusion_weights', {'net_battle_score': 0.6, 'control_strength': 0.4})
        df = df_minute.copy()
        df['vwap'] = df['vwap'].replace(0, np.nan).ffill()
        df = df.dropna(subset=['vwap'])
        if df.empty:
            return {"SCORE_INTRADAY_VWAP_BATTLEFIELD": 0.0}
        # 1. 计算支撑与压制
        # 支撑测试：K线的最低价触及或低于VWAP，但收盘价高于VWAP
        support_tests = df[(df['low'] <= df['vwap']) & (df['close'] > df['vwap'])]
        # 压制测试：K线的最高价触及或高于VWAP，但收盘价低于VWAP
        suppression_tests = df[(df['high'] >= df['vwap']) & (df['close'] < df['vwap'])]
        # 用成交量加权计算得分
        support_score = support_tests['volume'].sum()
        suppression_score = suppression_tests['volume'].sum()
        total_battle_volume = support_score + suppression_score + 1e-9
        net_battle_score = (support_score - suppression_score) / total_battle_volume
        # 2. 融合日线级控制信号
        control_strength = daily_signals.get('vwap_control_strength_D', 0.0)
        # 融合
        final_score = (net_battle_score * weights.get('net_battle_score', 0.6) +
                       control_strength * weights.get('control_strength', 0.4))
        # --- 探针监测 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates:
            processed_date_str = current_date.strftime('%Y-%m-%d')
            if processed_date_str in probe_dates:
                print(f"      [日内诡道探针] _diagnose_vwap_battlefield @ {processed_date_str}")
                print(f"        - VWAP支撑总量: {support_score:.2f}")
                print(f"        - VWAP压制总量: {suppression_score:.2f}")
                print(f"        - 净战斗分: {net_battle_score:.4f}")
                print(f"        - 日线级控制强度分: {control_strength:.4f}")
                print(f"        - 最终VWAP攻防分: {final_score:.4f}")
        return {"SCORE_INTRADAY_VWAP_BATTLEFIELD": np.clip(final_score, -1, 1)}


