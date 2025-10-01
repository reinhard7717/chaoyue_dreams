# 文件: strategies/trend_following/intelligence/foundation_intelligence.py
# 基础情报模块 (波动率, 震荡指标)
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following.utils import transmute_health_to_ultimate_signals, get_params_block, get_param_value, calculate_holographic_dynamics, normalize_score, normalize_to_bipolar

class FoundationIntelligence:
    def __init__(self, strategy_instance):
        """
        初始化基础情报模块。
        :param strategy_instance: 策略主实例的引用，用于访问df和atomic_states。
        """
        self.strategy = strategy_instance

    def run_foundation_analysis_command(self) -> Dict[str, pd.Series]:
        """
        【V4.0 · 统一范式版】基础情报分析总指挥
        - 核心重构 (本次修改):
          - [架构统一] 废除所有旧的、各自为战的诊断引擎，统一调用唯一的终极信号引擎 `diagnose_unified_foundation_signals`。
          - [哲学统一] 所有信号生成逻辑均遵循“上下文门控”范式，实现了整个模块内部思想的完全统一。
          - [保留特例] 保留了 `diagnose_volatility_intelligence` 等具有特殊战术意义的模块。
        - 收益: 实现了前所未有的架构清晰度、逻辑一致性和哲学完备性。
        """
        # print("      -> [基础情报分析总指挥 V4.0 · 统一范式版] 启动...") # 更新版本号
        df = self.strategy.df_indicators
        all_states = {}

        # 步骤 1: 执行唯一的、统一的终极信号引擎
        unified_states = self.diagnose_unified_foundation_signals(df)
        all_states.update(unified_states)

        # 步骤 2: 执行具有特殊战术意义的模块 (作为补充)
        all_states.update(self.diagnose_volatility_intelligence(df))
        all_states.update(self.diagnose_classic_indicators_atomics(df)) # 重命名为原子信号诊断
        
        # print(f"      -> [基础情报分析总指挥 V4.0] 分析完毕，共生成 {len(all_states)} 个基础层信号。") # 更新版本号
        return all_states

    def diagnose_unified_foundation_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V14.0 · 圣杯契约版】
        - 核心革命: 不再读取本地的、重复的合成参数，而是从最高指挥部获取唯一的“圣杯”配置
                      (`ultimate_signal_synthesis_params`)，并将其传递给中央合成引擎。
        """
        states = {}
        p_conf = get_params_block(self.strategy, 'foundation_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True): return states
        # 获取中央“圣杯”配置
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        periods = get_param_value(p_synthesis.get('periods'), [1, 5, 13, 21, 55])
        norm_window = get_param_value(p_synthesis.get('norm_window'), 55)
        dynamic_weights = {'slope': 0.6, 'accel': 0.4}
        health_data = { 's_bull': [], 's_bear': [], 'd_intensity': [] } 
        calculators = { 'ema': self._calculate_ema_health, 'rsi': self._calculate_rsi_health, 'macd': self._calculate_macd_health, 'cmf': self._calculate_cmf_health }
        for name, calculator in calculators.items():
            s_bull, s_bear, d_intensity = calculator(df, norm_window, dynamic_weights, periods)
            health_data['s_bull'].append(s_bull) 
            health_data['s_bear'].append(s_bear) 
            health_data['d_intensity'].append(d_intensity)
        overall_health = {}
        for health_type, health_sources in [ ('s_bull', health_data['s_bull']), ('s_bear', health_data['s_bear']), ('d_intensity', health_data['d_intensity']) ]:
            overall_health[health_type] = {}
            for p in periods:
                components_for_period = [pillar_dict[p].values for pillar_dict in health_sources if p in pillar_dict]
                if components_for_period:
                    stacked_values = np.stack(components_for_period, axis=0)
                    fused_values = np.prod(stacked_values, axis=0) ** (1.0 / stacked_values.shape[0])
                    overall_health[health_type][p] = pd.Series(fused_values, index=df.index, dtype=np.float32)
                else:
                    overall_health[health_type][p] = pd.Series(0.5, index=df.index, dtype=np.float32)
        self.strategy.atomic_states['__FOUNDATION_overall_health'] = overall_health
        # 传入唯一的“圣杯”配置
        ultimate_signals = transmute_health_to_ultimate_signals(
            df=df,
            atomic_states=self.strategy.atomic_states,
            overall_health=overall_health,
            params=p_synthesis,
            domain_prefix="FOUNDATION"
        )
        states.update(ultimate_signals)
        return states

    # ==============================================================================
    # 以下为重构后的健康度组件计算器，现在返回四维健康度
    # ==============================================================================
    def _calculate_ema_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict, Dict, Dict]:
        """【V4.6 · 赫尔墨斯商神杖版】计算EMA维度的三维健康度"""
        s_bull, s_bear, d_intensity = {}, {}, {}
        # [代码新增] 获取本模块的专属配置
        p_conf = get_params_block(self.strategy, 'foundation_ultimate_params', {})
        # [代码新增] 获取新的加权融合权重
        fusion_weights = get_param_value(p_conf.get('ma_health_fusion_weights'), {'alignment': 0.1, 'slope': 0.2, 'accel': 0.2, 'relational': 0.5})

        ma_periods = [5, 13, 21, 55, 89]
        
        # --- 维度1: 静态结构 (Alignment) ---
        bull_alignment_scores = []
        bear_alignment_scores = []
        for i in range(len(ma_periods) - 1):
            short_col = f'EMA_{ma_periods[i]}_D'
            long_col = f'EMA_{ma_periods[i+1]}_D'
            if short_col in df and long_col in df:
                bull_alignment_scores.append((df[short_col] > df[long_col]).astype(float))
                bear_alignment_scores.append((df[short_col] < df[long_col]).astype(float))
        alignment_score = pd.DataFrame(bull_alignment_scores).mean().fillna(0.5) if bull_alignment_scores else pd.Series(0.5, index=df.index)
        static_bear_score = pd.DataFrame(bear_alignment_scores).mean().fillna(0.5) if bear_alignment_scores else pd.Series(0.5, index=df.index)

        # --- 维度2, 3, 4: 动态健康度 (一阶、二阶、关系) ---
        slope_health_scores, accel_health_scores, relational_health_scores = [], [], []
        
        for p in ma_periods:
            slope_col = f'SLOPE_{p}_EMA_{p}_D' if p != 1 else f'SLOPE_1_close_D'
            accel_col = f'ACCEL_{p}_EMA_{p}_D' if p != 1 else f'ACCEL_1_close_D'
            if slope_col in df.columns:
                bipolar_slope = normalize_to_bipolar(df[slope_col], df.index, norm_window)
                slope_health_scores.append((bipolar_slope + 1) / 2.0)
            if accel_col in df.columns:
                bipolar_accel = normalize_to_bipolar(df[accel_col], df.index, norm_window)
                accel_health_scores.append((bipolar_accel + 1) / 2.0)
                
        ma_pairs = [(5, 21), (13, 55)]
        for short_p, long_p in ma_pairs:
            short_ma_col, long_ma_col = f'EMA_{short_p}_D', f'EMA_{long_p}_D'
            if short_ma_col in df.columns and long_ma_col in df.columns:
                spread = df[short_ma_col] - df[long_ma_col]
                spread_accel = spread.diff(3).diff(3).fillna(0)
                bipolar_rel_accel = normalize_to_bipolar(spread_accel, df.index, norm_window)
                relational_health_scores.append((bipolar_rel_accel + 1) / 2.0)

        # --- 融合四维健康度 ---
        avg_slope_health = pd.concat(slope_health_scores, axis=1).mean(axis=1).fillna(0.5) if slope_health_scores else pd.Series(0.5, index=df.index)
        avg_accel_health = pd.concat(accel_health_scores, axis=1).mean(axis=1).fillna(0.5) if accel_health_scores else pd.Series(0.5, index=df.index)
        avg_relational_health = pd.concat(relational_health_scores, axis=1).mean(axis=1).fillna(0.5) if relational_health_scores else pd.Series(0.5, index=df.index)

        # [代码修改] 最终静态看涨分是四维健康的加权算术融合
        static_bull_score = (
            alignment_score * fusion_weights.get('alignment', 0.1) +
            avg_slope_health * fusion_weights.get('slope', 0.2) +
            avg_accel_health * fusion_weights.get('accel', 0.2) +
            avg_relational_health * fusion_weights.get('relational', 0.5)
        )

        for p in periods:
            s_bull[p] = static_bull_score
            s_bear[p] = static_bear_score
            ema_col = f'EMA_{p}_D' if p > 1 else 'close_D'
            bull_holo, bear_holo = calculate_holographic_dynamics(df, ema_col, norm_window)
            d_intensity[p] = (bull_holo + bear_holo) / 2.0
        
        return s_bull, s_bear, d_intensity

    def _calculate_rsi_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict, Dict, Dict]:
        """【V4.0 · 全息动态升级版】计算RSI维度的三维健康度"""
        s_bull, s_bear, d_intensity = {}, {}, {}
        
        static_bull_score = normalize_score(df.get('RSI_13_D'), df.index, norm_window, ascending=True)
        static_bear_score = normalize_score(df.get('RSI_13_D'), df.index, norm_window, ascending=False)

        # 使用全新的全息动态引擎计算动态强度分
        bull_holo, bear_holo = calculate_holographic_dynamics(df, 'RSI_13_D', norm_window)

        for p in periods:
            s_bull[p] = static_bull_score
            s_bear[p] = static_bear_score
            # 所有周期共享同一个、更高级的动态强度分
            d_intensity[p] = (bull_holo + bear_holo) / 2.0
        
        return s_bull, s_bear, d_intensity

    def _calculate_macd_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict, Dict, Dict]:
        """【V4.0 · 全息动态升级版】计算MACD维度的三维健康度"""
        s_bull, s_bear, d_intensity = {}, {}, {}
        
        static_bull_score = normalize_score(df.get('MACDh_13_34_8_D'), df.index, norm_window, ascending=True)
        static_bear_score = normalize_score(df.get('MACDh_13_34_8_D'), df.index, norm_window, ascending=False)

        # 使用全新的全息动态引擎计算动态强度分
        bull_holo, bear_holo = calculate_holographic_dynamics(df, 'MACDh_13_34_8_D', norm_window)

        for p in periods:
            s_bull[p] = static_bull_score
            s_bear[p] = static_bear_score
            # 所有周期共享同一个、更高级的动态强度分
            d_intensity[p] = (bull_holo + bear_holo) / 2.0
        
        return s_bull, s_bear, d_intensity

    def _calculate_cmf_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict, Dict, Dict]:
        """【V4.0 · 全息动态升级版】计算CMF维度的三维健康度"""
        s_bull, s_bear, d_intensity = {}, {}, {}
        
        static_bull_score = normalize_score(df.get('CMF_21_D'), df.index, norm_window, ascending=True)
        static_bear_score = normalize_score(df.get('CMF_21_D'), df.index, norm_window, ascending=False)

        # 使用全新的全息动态引擎计算动态强度分
        bull_holo, bear_holo = calculate_holographic_dynamics(df, 'CMF_21_D', norm_window)

        for p in periods:
            s_bull[p] = static_bull_score
            s_bear[p] = static_bear_score
            # 所有周期共享同一个、更高级的动态强度分
            d_intensity[p] = (bull_holo + bear_holo) / 2.0
        
        return s_bull, s_bear, d_intensity


    # ==============================================================================
    # 以下为保留的、具有特殊战术意义的模块
    # ==============================================================================

    def diagnose_volatility_intelligence(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V6.4 状态诊断升级版】波动率统一情报中心 (战术模块，予以保留)
        """
        states = {}
        norm_window = 120 # 使用一个标准的归一化窗口
        score_squeeze_daily = normalize_score(df.get('BBW_21_2.0_D'), df.index, norm_window, ascending=False)
        score_squeeze_weekly = normalize_score(df.get('BBW_21_2.0_W'), df.index, norm_window, ascending=False)
        score_squeeze_momentum = normalize_score(df.get('SLOPE_5_BBW_21_2.0_D'), df.index, norm_window, ascending=False)
        score_expansion_daily = 1 - score_squeeze_daily
        score_expansion_weekly = 1 - score_squeeze_weekly
        score_expansion_momentum = 1 - score_squeeze_momentum
        score_vol_accel_up = normalize_score(df.get('ACCEL_5_BBW_21_2.0_D'), df.index, norm_window, ascending=True)
        score_vol_accel_down = normalize_score(df.get('ACCEL_5_BBW_21_2.0_D'), df.index, norm_window, ascending=False)
        
        states['SCORE_VOL_COMPRESSION_B'] = score_squeeze_daily
        states['SCORE_VOL_COMPRESSION_A'] = (score_squeeze_daily * score_squeeze_weekly).astype(np.float32)
        states['SCORE_VOL_COMPRESSION_S'] = (states['SCORE_VOL_COMPRESSION_A'] * score_squeeze_momentum).astype(np.float32)
        
        states['SCORE_VOL_EXPANSION_B'] = score_expansion_daily
        states['SCORE_VOL_EXPANSION_A'] = (score_expansion_daily * score_expansion_weekly).astype(np.float32)
        states['SCORE_VOL_EXPANSION_S'] = (states['SCORE_VOL_EXPANSION_A'] * score_expansion_momentum).astype(np.float32)
        
        states['SCORE_VOL_TIPPING_POINT_BOTTOM_OPP'] = (states['SCORE_VOL_COMPRESSION_S'] * score_vol_accel_up).astype(np.float32)
        states['SCORE_VOL_TIPPING_POINT_TOP_RISK'] = (states['SCORE_VOL_EXPANSION_S'] * score_vol_accel_down).astype(np.float32)
        
        hurst_score = normalize_score(df.get('hurst_120d_D'), df.index, norm_window)
        states['SCORE_TRENDING_REGIME'] = hurst_score

        return states

    def diagnose_classic_indicators_atomics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 · 原子信号版】经典指标原子信号诊断 (战术模块，予以保留)
        """
        states = {}
        p = get_params_block(self.strategy, 'classic_indicator_params')
        if not get_param_value(p.get('enabled'), True): return states
        
        norm_window = 120 # 使用一个标准的归一化窗口
        candle_body_up = (df.get('close_D', 0) - df.get('open_D', 0)).clip(lower=0)
        candle_body_down = (df.get('open_D', 0) - df.get('close_D', 0)).clip(lower=0)
        score_price_up_strength = normalize_score(candle_body_up, df.index, norm_window)
        score_price_down_strength = normalize_score(candle_body_down, df.index, norm_window)
        
        score_vol_slope_up = normalize_score(df.get('SLOPE_5_volume_D', pd.Series(0.5, index=df.index)).clip(lower=0), df.index, norm_window)
        score_vol_accel_up = normalize_score(df.get('ACCEL_5_volume_D', pd.Series(0.5, index=df.index)).clip(lower=0), df.index, norm_window)
        score_volume_igniting = score_vol_slope_up * score_vol_accel_up
        
        states['SCORE_VOL_PRICE_IGNITION_UP'] = score_price_up_strength * score_volume_igniting
        states['SCORE_VOL_PRICE_PANIC_DOWN_RISK'] = score_price_down_strength * score_volume_igniting
        return states










