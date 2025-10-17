import asyncio
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Optional
# 导入 get_params_block 工具
from strategies.trend_following.utils import get_params_block, normalize_to_bipolar, normalize_score

class IntradayBehaviorEngine:
    """
    【V1.2 · 全指标集成版】
    - 核心升级: 内部集成了TRIX, KDJ, Squeeze, EOM等高级分钟级指标的计算，
                并基于这些指标重构了战术诊断模型，使其分析能力大幅提升。
    """
    def __init__(self, strategy_instance):
        """初始化时加载专属配置，并获取指标计算器的引用"""
        self.strategy = strategy_instance
        # 修正访问路径：通过 orchestrator 访问顶层的 indicator_service
        self.calculator = strategy_instance.orchestrator.indicator_service.calculator
        # 从策略配置中加载本引擎的专属参数块
        self.params = get_params_block(self.strategy, 'intraday_behavior_engine_params', {})
        self.fib_periods = [5, 8, 13, 21, 34, 55]

    async def _prepare_intraday_indicators(self, df_minute: pd.DataFrame) -> Optional[pd.DataFrame]:
        """统一为分钟数据计算所有必需的战术指标"""
        if df_minute is None or df_minute.empty:
            return None
        
        df_enriched = df_minute.copy()
        calc_tasks = []

        # TRIX
        trix_params = self.params.get('trix_params', {})
        if trix_params.get('enabled'):
            calc_tasks.append(self.calculator.calculate_trix(df_enriched, trix_params['period'], trix_params['signal_period']))
        
        # KDJ
        kdj_params = self.params.get('kdj_params', {})
        if kdj_params.get('enabled'):
            calc_tasks.append(self.calculator.calculate_kdj(df_enriched, kdj_params['period'], kdj_params['signal_period'], kdj_params['smooth_k_period']))

        # Squeeze
        sqz_params = self.params.get('squeeze_params', {})
        if sqz_params.get('enabled'):
            calc_tasks.append(self.calculator.calculate_squeeze(df_enriched, sqz_params['bb_period'], sqz_params['kc_period'], sqz_params['atr_period']))

        # Donchian
        donchian_params = self.params.get('donchian_params', {})
        if donchian_params.get('enabled'):
            calc_tasks.append(self.calculator.calculate_donchian(df_enriched, donchian_params['period']))

        # EOM
        eom_params = self.params.get('eom_params', {})
        if eom_params.get('enabled'):
            calc_tasks.append(self.calculator.calculate_eom(df_enriched, eom_params['period']))

        results = await asyncio.gather(*calc_tasks)
        for res_df in results:
            if res_df is not None and not res_df.empty:
                df_enriched = df_enriched.join(res_df, how='left')
        
        return df_enriched

    async def run_intraday_diagnostics(self, df_minute: pd.DataFrame) -> Dict[str, float]:
        """日内诊断总指挥，现在先计算指标再进行诊断"""
        # 首先调用准备方法，为原始分钟数据添加所有战术指标
        df_enriched = await self._prepare_intraday_indicators(df_minute)

        if df_enriched is None or df_enriched.empty or len(df_enriched) < max(self.fib_periods):
            return {
                "SCORE_INTRADAY_EFFICIENCY_DYNAMICS": 0.0,
                "SCORE_INTRADAY_CONSISTENCY_DYNAMICS": 0.0,
                "SCORE_INTRADAY_BREAKOUT_POTENTIAL": 0.0,
                "SCORE_INTRADAY_BOTTOM_REVERSAL_DYNAMICS": 0.0,
                "SCORE_INTRADAY_TOP_REVERSAL_DYNAMICS": 0.0,
            }

        tasks = [
            # 所有诊断任务现在都使用包含丰富指标的 df_enriched
            self.diagnose_attack_efficiency_dynamics(df_enriched),
            self.diagnose_trend_consistency_dynamics(df_enriched),
            self.diagnose_breakout_potential(df_enriched),
            self.diagnose_bottom_reversal_dynamics(df_enriched),
            self.diagnose_top_reversal_dynamics(df_enriched),
        ]
        results = await asyncio.gather(*tasks)
        
        final_scores = {}
        for res_dict in results:
            final_scores.update(res_dict)
            
        return final_scores

    async def diagnose_attack_efficiency_dynamics(self, df_minute: pd.DataFrame) -> Dict[str, float]:
        """诊断“攻击效率”的动态变化，改用EOM指标"""
        eom_params = self.params.get('eom_params', {})
        eom_col = f"EOM_{eom_params.get('period', 13)}"
        eom_signal_col = f"EOMs_{eom_params.get('period', 13)}" # pandas-ta 可能会返回信号线
        
        target_col = eom_signal_col if eom_signal_col in df_minute.columns else eom_col
        if target_col not in df_minute.columns:
            return {"SCORE_INTRADAY_EFFICIENCY_DYNAMICS": 0.0}

        # EOM指标本身就量化了价量效率，其值越高代表效率越高（价涨量缩）
        # 我们直接对EOM指标本身进行元分析，捕捉其动态变化
        efficiency_snapshot_score = df_minute[target_col].fillna(0)
        
        final_score = self._perform_relational_meta_analysis(efficiency_snapshot_score)
        return {"SCORE_INTRADAY_EFFICIENCY_DYNAMICS": final_score}

    async def diagnose_trend_consistency_dynamics(self, df_minute: pd.DataFrame) -> Dict[str, float]:
        """诊断“趋势一致性”的动态变化，改用TRIX指标"""
        trix_params = self.params.get('trix_params', {})
        trix_col = f"TRIX_{trix_params.get('period', 13)}_{trix_params.get('signal_period', 8)}"
        if trix_col not in df_minute.columns:
            return {"SCORE_INTRADAY_CONSISTENCY_DYNAMICS": 0.0}

        # TRIX的斜率是衡量平滑后动能的最佳指标
        consistency_snapshot_score = ta.linreg(df_minute[trix_col], length=self.fib_periods[0]).fillna(0)
        
        final_score = self._perform_relational_meta_analysis(consistency_snapshot_score)
        return {"SCORE_INTRADAY_CONSISTENCY_DYNAMICS": final_score}

    async def diagnose_breakout_potential(self, df_minute: pd.DataFrame) -> Dict[str, float]:
        """诊断“突破潜力”，改用Squeeze指标"""
        sqz_on_col = 'SQZ_ON'
        sqz_off_col = 'SQZ_OFF'
        
        if sqz_on_col not in df_minute.columns or sqz_off_col not in df_minute.columns:
            return {"SCORE_INTRADAY_BREAKOUT_POTENTIAL": 0.0}

        # 突破潜力分 = 压缩状态分 + 压缩释放奖励分
        # 压缩状态分：市场处于压缩状态时，分值为1，否则为0
        squeeze_state_score = df_minute[sqz_on_col]
        
        # 压缩释放奖励分：在压缩结束（SQZ_OFF为1）的那个瞬间，给予一个高分奖励
        squeeze_release_bonus = (df_minute[sqz_off_col] == 1) & (df_minute[sqz_off_col].shift(1) == 0)
        
        # 瞬时关系快照分
        breakout_snapshot_score = squeeze_state_score + squeeze_release_bonus.astype(float) * 2.0
        
        final_score = self._perform_relational_meta_analysis(breakout_snapshot_score)
        return {"SCORE_INTRADAY_BREAKOUT_POTENTIAL": final_score}

    async def diagnose_bottom_reversal_dynamics(self, df_minute: pd.DataFrame) -> Dict[str, float]:
        """诊断底部反转，使用KDJ指标增强"""
        kdj_params = self.params.get('kdj_params', {})
        j_col = f"J_{kdj_params.get('period', 13)}_{kdj_params.get('signal_period', 5)}_{kdj_params.get('smooth_k_period', 3)}"
        if j_col not in df_minute.columns:
            return {"SCORE_INTRADAY_BOTTOM_REVERSAL_DYNAMICS": 0.0}

        # 原子指标1：KDJ超卖程度 (J值小于0的程度)
        oversold_score = normalize_score(-df_minute[j_col].clip(upper=0), df_minute.index, window=self.fib_periods[4])
        
        # 原子指标2：成交量脉冲
        vol_ma = df_minute['volume'].rolling(window=self.fib_periods[4]).mean().fillna(method='bfill')
        volume_spike_intensity = normalize_score((df_minute['volume'] / vol_ma.replace(0, np.nan)).fillna(1.0), df_minute.index, window=self.fib_periods[4])
        
        # 原子指标3：KDJ金叉信号
        k_col = j_col.replace('J_', 'K_')
        golden_cross = (df_minute[j_col] > df_minute[k_col]) & (df_minute[j_col].shift(1) <= df_minute[k_col].shift(1))
        
        # 瞬时关系快照分
        reversal_snapshot_score = oversold_score * volume_spike_intensity * golden_cross.astype(float)
        
        final_score = self._perform_relational_meta_analysis(reversal_snapshot_score)
        return {"SCORE_INTRADAY_BOTTOM_REVERSAL_DYNAMICS": final_score}

    async def diagnose_top_reversal_dynamics(self, df_minute: pd.DataFrame) -> Dict[str, float]:
        """诊断顶部反转，使用KDJ指标增强"""
        kdj_params = self.params.get('kdj_params', {})
        j_col = f"J_{kdj_params.get('period', 13)}_{kdj_params.get('signal_period', 5)}_{kdj_params.get('smooth_k_period', 3)}"
        if j_col not in df_minute.columns:
            return {"SCORE_INTRADAY_TOP_REVERSAL_DYNAMICS": 0.0}

        # 原子指标1：KDJ超买程度 (J值大于100的程度)
        overbought_score = normalize_score((df_minute[j_col] - 100).clip(lower=0), df_minute.index, window=self.fib_periods[4])
        
        # 原子指标2：成交量脉冲
        vol_ma = df_minute['volume'].rolling(window=self.fib_periods[4]).mean().fillna(method='bfill')
        volume_spike_intensity = normalize_score((df_minute['volume'] / vol_ma.replace(0, np.nan)).fillna(1.0), df_minute.index, window=self.fib_periods[4])
        
        # 原子指标3：KDJ死叉信号
        k_col = j_col.replace('J_', 'K_')
        dead_cross = (df_minute[j_col] < df_minute[k_col]) & (df_minute[j_col].shift(1) >= df_minute[k_col].shift(1))
        
        # 瞬时关系快照分
        reversal_snapshot_score = overbought_score * volume_spike_intensity * dead_cross.astype(float)
        
        final_score = self._perform_relational_meta_analysis(reversal_snapshot_score)
        return {"SCORE_INTRADAY_TOP_REVERSAL_DYNAMICS": final_score}

    def _perform_relational_meta_analysis(self, relationship_score: pd.Series) -> float:
        """对关系分时间序列进行元分析，从配置中读取参数"""
        if relationship_score.empty or len(relationship_score) < self.fib_periods[3]:
            return 0.0
        
        # 从 self.params 中读取元分析参数
        meta_params = self.params.get('meta_analysis_params', {})
        meta_window = meta_params.get('meta_window', 13)
        norm_window = meta_params.get('norm_window', 55)
        trend_weight = meta_params.get('trend_weight', 0.6)
        accel_weight = meta_params.get('accel_weight', 0.4)

        relationship_trend = ta.linreg(relationship_score, length=meta_window).fillna(0)
        bipolar_trend_strength = normalize_to_bipolar(relationship_trend, relationship_score.index, norm_window)
        
        relationship_accel = ta.linreg(relationship_trend, length=meta_window).fillna(0)
        bipolar_accel_strength = normalize_to_bipolar(relationship_accel, relationship_score.index, norm_window)
        
        meta_score_series = (bipolar_trend_strength * trend_weight + bipolar_accel_strength * accel_weight)
        
        return meta_score_series.iloc[-1] if not meta_score_series.empty else 0.0
