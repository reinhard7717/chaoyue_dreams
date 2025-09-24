# 文件: strategies/trend_following/intelligence/foundation_intelligence.py
# 基础情报模块 (波动率, 震荡指标)
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following.utils import get_params_block, get_param_value, create_persistent_state

class FoundationIntelligence:
    def __init__(self, strategy_instance):
        """
        初始化基础情报模块。
        :param strategy_instance: 策略主实例的引用，用于访问df和atomic_states。
        """
        self.strategy = strategy_instance

    def _normalize_score(self, series: pd.Series, window: int = 120, ascending: bool = True) -> pd.Series:
        """
        辅助函数：将一个Series进行滚动窗口排名归一化，生成0-1分。
        """
        if series is None or series.empty:
            return pd.Series(0.5, index=self.strategy.df_indicators.index, dtype=np.float32)
        
        min_periods = max(1, window // 5)
        rank = series.rolling(window=window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        score = rank if ascending else 1 - rank
        return score.astype(np.float32)

    def run_foundation_analysis_command(self) -> Dict[str, pd.Series]:
        """
        【V4.0 · 统一范式版】基础情报分析总指挥
        - 核心重构 (本次修改):
          - [架构统一] 废除所有旧的、各自为战的诊断引擎，统一调用唯一的终极信号引擎 `diagnose_unified_foundation_signals`。
          - [哲学统一] 所有信号生成逻辑均遵循“上下文门控”范式，实现了整个模块内部思想的完全统一。
          - [保留特例] 保留了 `diagnose_volatility_intelligence` 等具有特殊战术意义的模块。
        - 收益: 实现了前所未有的架构清晰度、逻辑一致性和哲学完备性。
        """
        print("      -> [基础情报分析总指挥 V4.0 · 统一范式版] 启动...") # 更新版本号
        df = self.strategy.df_indicators
        all_states = {}

        # 步骤 1: 执行唯一的、统一的终极信号引擎
        unified_states = self.diagnose_unified_foundation_signals(df)
        all_states.update(unified_states)

        # 步骤 2: 执行具有特殊战术意义的模块 (作为补充)
        all_states.update(self.diagnose_volatility_intelligence(df))
        all_states.update(self.diagnose_classic_indicators_atomics(df)) # 重命名为原子信号诊断
        
        print(f"      -> [基础情报分析总指挥 V4.0] 分析完毕，共生成 {len(all_states)} 个基础层信号。") # 更新版本号
        return all_states

    def diagnose_unified_foundation_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.0 · 对称逻辑版】统一基础层信号诊断引擎
        - 核心重构 (本次修改):
          - [哲学升维] 彻底废除 `1 - bullish` 的粗暴逻辑，为“看跌”信号建立完全独立且对称的计算体系。
          - [四维输出] 所有健康度组件现在输出 (静多, 动多, 静空, 动空) 四个维度的健康分。
          - [独立融合] 主引擎独立融合生成四个全局健康度，确保多空信号的计算互不干扰。
        - 收益: 实现了与所有其他情报引擎在哲学和代码结构上的完全统一，信号质量达到最终形态。
        """
        print("        -> [统一基础层信号诊断引擎 V3.0 · 对称逻辑版] 启动...") # 更新版本号和说明
        states = {}
        p_conf = get_params_block(self.strategy, 'foundation_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True): return states

        # --- 1. 定义权重与参数 ---
        dynamic_weights = {'slope': 0.6, 'accel': 0.4}
        resonance_tf_weights = {'short': 0.2, 'medium': 0.5, 'long': 0.3}
        reversal_tf_weights = {'short': 0.6, 'medium': 0.3, 'long': 0.1}
        periods = get_param_value(p_conf.get('periods', [1, 5, 13, 21, 55]))
        norm_window = get_param_value(p_conf.get('norm_window'), 120)

        # --- 2. 计算“外部宏观位置”门控 (用于反转) ---
        rolling_low_55d = df['low_D'].rolling(window=55, min_periods=21).min()
        rolling_high_55d = df['high_D'].rolling(window=55, min_periods=21).max()
        price_range_55d = (rolling_high_55d - rolling_low_55d).replace(0, 1e-9)
        price_position_in_range = ((df['close_D'] - rolling_low_55d) / price_range_55d).clip(0, 1).fillna(0.5)
        bottom_context_score = 1 - price_position_in_range
        top_context_score = price_position_in_range

        # --- 3. 调用所有健康度组件计算器，获取四维健康度 ---
        # 重构数据结构以接收四维健康度
        health_data = {
            'bullish_static': [], 'bullish_dynamic': [],
            'bearish_static': [], 'bearish_dynamic': []
        }
        
        calculators = {
            'ema': self._calculate_ema_health,
            'rsi': self._calculate_rsi_health,
            'macd': self._calculate_macd_health,
            'cmf': self._calculate_cmf_health,
        }

        for name, calculator in calculators.items():
            s_bull, d_bull, s_bear, d_bear = calculator(df, norm_window, dynamic_weights, periods)
            health_data['bullish_static'].append(s_bull)
            health_data['bullish_dynamic'].append(d_bull)
            health_data['bearish_static'].append(s_bear)
            health_data['bearish_dynamic'].append(d_bear)

        # --- 4. 独立融合，生成四个全局健康度 ---
        overall_health = {}
        for health_type in health_data: # e.g., 'bullish_static'
            overall_health[health_type] = {}
            for p in periods:
                components_for_period = [pillar_dict[p].values for pillar_dict in health_data[health_type] if p in pillar_dict]
                if components_for_period:
                    overall_health[health_type][p] = pd.Series(np.mean(np.stack(components_for_period, axis=0), axis=0), index=df.index)
                else:
                    overall_health[health_type][p] = pd.Series(0.5, index=df.index)

        # --- 5. 终极信号合成 (采用对称逻辑) ---
        # 5.1 看涨信号合成
        bullish_resonance_health = {p: overall_health['bullish_static'][p] * overall_health['bullish_dynamic'][p] for p in periods}
        bullish_short_force_res = (bullish_resonance_health.get(1, 0.5) * bullish_resonance_health.get(5, 0.5))**0.5
        bullish_medium_trend_res = (bullish_resonance_health.get(13, 0.5) * bullish_resonance_health.get(21, 0.5))**0.5
        bullish_long_inertia_res = bullish_resonance_health.get(55, 0.5)
        overall_bullish_resonance = (bullish_short_force_res * resonance_tf_weights['short'] + bullish_medium_trend_res * resonance_tf_weights['medium'] + bullish_long_inertia_res * resonance_tf_weights['long'])
        
        bullish_dynamic_health = overall_health['bullish_dynamic']
        bullish_short_force_rev = (bullish_dynamic_health.get(1, 0.5) * bullish_dynamic_health.get(5, 0.5))**0.5
        bullish_medium_trend_rev = (bullish_dynamic_health.get(13, 0.5) * bullish_dynamic_health.get(21, 0.5))**0.5
        bullish_long_inertia_rev = bullish_dynamic_health.get(55, 0.5)
        overall_bullish_reversal_trigger = (bullish_short_force_rev * reversal_tf_weights['short'] + bullish_medium_trend_rev * reversal_tf_weights['medium'] + bullish_long_inertia_rev * reversal_tf_weights['long'])
        final_bottom_reversal_score = bottom_context_score * overall_bullish_reversal_trigger

        # 5.2 看跌信号合成 (使用独立的看跌健康度)
        bearish_resonance_health = {p: overall_health['bearish_static'][p] * overall_health['bearish_dynamic'][p] for p in periods}
        bearish_short_force_res = (bearish_resonance_health.get(1, 0.5) * bearish_resonance_health.get(5, 0.5))**0.5
        bearish_medium_trend_res = (bearish_resonance_health.get(13, 0.5) * bearish_resonance_health.get(21, 0.5))**0.5
        bearish_long_inertia_res = bearish_resonance_health.get(55, 0.5)
        overall_bearish_resonance = (bearish_short_force_res * resonance_tf_weights['short'] + bearish_medium_trend_res * resonance_tf_weights['medium'] + bearish_long_inertia_res * resonance_tf_weights['long'])

        bearish_dynamic_health = overall_health['bearish_dynamic']
        bearish_short_force_rev = (bearish_dynamic_health.get(1, 0.5) * bearish_dynamic_health.get(5, 0.5))**0.5
        bearish_medium_trend_rev = (bearish_dynamic_health.get(13, 0.5) * bearish_dynamic_health.get(21, 0.5))**0.5
        bearish_long_inertia_rev = bearish_dynamic_health.get(55, 0.5)
        overall_bearish_reversal_trigger = (bearish_short_force_rev * reversal_tf_weights['short'] + bearish_medium_trend_rev * reversal_tf_weights['medium'] + bearish_long_inertia_rev * reversal_tf_weights['long'])
        final_top_reversal_score = top_context_score * overall_bearish_reversal_trigger

        # 5.3 赋值
        for prefix, score in [('SCORE_FOUNDATION_BULLISH_RESONANCE', overall_bullish_resonance), ('SCORE_FOUNDATION_BOTTOM_REVERSAL', final_bottom_reversal_score),
                              ('SCORE_FOUNDATION_BEARISH_RESONANCE', overall_bearish_resonance), ('SCORE_FOUNDATION_TOP_REVERSAL', final_top_reversal_score)]:
            states[f'{prefix}_S_PLUS'] = score.astype(np.float32)
            states[f'{prefix}_S'] = (score * 0.8).astype(np.float32)
            states[f'{prefix}_A'] = (score * 0.6).astype(np.float32)
            states[f'{prefix}_B'] = (score * 0.4).astype(np.float32)
        
        return states

    # ==============================================================================
    # 以下为重构后的健康度组件计算器，现在返回四维健康度
    # ==============================================================================

    def _calculate_ema_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict, Dict, Dict, Dict]:
        """【V3.0 · 对称逻辑版】计算EMA健康度"""
        s_bull, d_bull, s_bear, d_bear = {}, {}, {}, {}
        
        bull_alignment_scores, bear_alignment_scores = [], []
        for i in range(len(periods) - 1):
            short_col = f'EMA_{periods[i]}_D' if periods[i] > 1 else 'close_D'
            long_col = f'EMA_{periods[i+1]}_D'
            bull_alignment_scores.append((df.get(short_col, df['close_D']).values > df.get(long_col, df['close_D']).values).astype(np.float32))
            bear_alignment_scores.append((df.get(short_col, df['close_D']).values < df.get(long_col, df['close_D']).values).astype(np.float32))
        
        static_bull_score = pd.Series(np.mean(np.stack(bull_alignment_scores, axis=0), axis=0), index=df.index)
        static_bear_score = pd.Series(np.mean(np.stack(bear_alignment_scores, axis=0), axis=0), index=df.index)

        for p in periods:
            s_bull[p] = static_bull_score
            s_bear[p] = static_bear_score
            
            ema_col = f'EMA_{p}_D' if p > 1 else 'close_D'
            slope = self._normalize_score(df.get(f'SLOPE_{p}_{ema_col}'), norm_window, ascending=True)
            accel = self._normalize_score(df.get(f'ACCEL_{p}_{ema_col}'), norm_window, ascending=True)
            d_bull[p] = slope * dynamic_weights['slope'] + accel * dynamic_weights['accel']
            
            slope_neg = self._normalize_score(df.get(f'SLOPE_{p}_{ema_col}'), norm_window, ascending=False)
            accel_neg = self._normalize_score(df.get(f'ACCEL_{p}_{ema_col}'), norm_window, ascending=False)
            d_bear[p] = slope_neg * dynamic_weights['slope'] + accel_neg * dynamic_weights['accel']
        
        return s_bull, d_bull, s_bear, d_bear

    def _calculate_rsi_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict, Dict, Dict, Dict]:
        """【V3.0 · 对称逻辑版】计算RSI健康度"""
        s_bull, d_bull, s_bear, d_bear = {}, {}, {}, {}
        
        static_bull_score = self._normalize_score(df.get('RSI_13_D'), norm_window, ascending=True)
        static_bear_score = self._normalize_score(df.get('RSI_13_D'), norm_window, ascending=False)

        for p in periods:
            s_bull[p] = static_bull_score
            s_bear[p] = static_bear_score
            
            slope = self._normalize_score(df.get(f'SLOPE_{p}_RSI_13_D'), norm_window, ascending=True)
            accel = self._normalize_score(df.get(f'ACCEL_{p}_RSI_13_D'), norm_window, ascending=True)
            d_bull[p] = slope * dynamic_weights['slope'] + accel * dynamic_weights['accel']
            
            slope_neg = self._normalize_score(df.get(f'SLOPE_{p}_RSI_13_D'), norm_window, ascending=False)
            accel_neg = self._normalize_score(df.get(f'ACCEL_{p}_RSI_13_D'), norm_window, ascending=False)
            d_bear[p] = slope_neg * dynamic_weights['slope'] + accel_neg * dynamic_weights['accel']
        
        return s_bull, d_bull, s_bear, d_bear

    def _calculate_macd_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict, Dict, Dict, Dict]:
        """【V3.0 · 对称逻辑版】计算MACD健康度"""
        s_bull, d_bull, s_bear, d_bear = {}, {}, {}, {}
        
        static_bull_score = self._normalize_score(df.get('MACDh_13_34_8_D'), norm_window, ascending=True)
        static_bear_score = self._normalize_score(df.get('MACDh_13_34_8_D'), norm_window, ascending=False)

        for p in periods:
            s_bull[p] = static_bull_score
            s_bear[p] = static_bear_score
            
            slope = self._normalize_score(df.get(f'SLOPE_{p}_MACDh_13_34_8_D'), norm_window, ascending=True)
            accel = self._normalize_score(df.get(f'ACCEL_{p}_MACDh_13_34_8_D'), norm_window, ascending=True)
            d_bull[p] = slope * dynamic_weights['slope'] + accel * dynamic_weights['accel']
            
            slope_neg = self._normalize_score(df.get(f'SLOPE_{p}_MACDh_13_34_8_D'), norm_window, ascending=False)
            accel_neg = self._normalize_score(df.get(f'ACCEL_{p}_MACDh_13_34_8_D'), norm_window, ascending=False)
            d_bear[p] = slope_neg * dynamic_weights['slope'] + accel_neg * dynamic_weights['accel']
        
        return s_bull, d_bull, s_bear, d_bear

    def _calculate_cmf_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict, Dict, Dict, Dict]:
        """【V3.0 · 对称逻辑版】计算CMF健康度"""
        s_bull, d_bull, s_bear, d_bear = {}, {}, {}, {}
        
        static_bull_score = self._normalize_score(df.get('CMF_21_D'), norm_window, ascending=True)
        static_bear_score = self._normalize_score(df.get('CMF_21_D'), norm_window, ascending=False)

        for p in periods:
            s_bull[p] = static_bull_score
            s_bear[p] = static_bear_score
            
            slope = self._normalize_score(df.get(f'SLOPE_{p}_CMF_21_D'), norm_window, ascending=True)
            accel = self._normalize_score(df.get(f'ACCEL_{p}_CMF_21_D'), norm_window, ascending=True)
            d_bull[p] = slope * dynamic_weights['slope'] + accel * dynamic_weights['accel']
            
            slope_neg = self._normalize_score(df.get(f'SLOPE_{p}_CMF_21_D'), norm_window, ascending=False)
            accel_neg = self._normalize_score(df.get(f'ACCEL_{p}_CMF_21_D'), norm_window, ascending=False)
            d_bear[p] = slope_neg * dynamic_weights['slope'] + accel_neg * dynamic_weights['accel']
        
        return s_bull, d_bull, s_bear, d_bear

    # ==============================================================================
    # 以下为保留的、具有特殊战术意义的模块
    # ==============================================================================

    def diagnose_volatility_intelligence(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V6.4 状态诊断升级版】波动率统一情报中心 (战术模块，予以保留)
        """
        states = {}
        p = get_params_block(self.strategy, 'volatility_state_params')
        if not get_param_value(p.get('enabled'), False): return states
        
        score_squeeze_daily = self._normalize_score(df.get('BBW_21_2.0_D'), ascending=False)
        score_squeeze_weekly = self._normalize_score(df.get('BBW_21_2.0_W'), ascending=False)
        score_squeeze_momentum = self._normalize_score(df.get('SLOPE_5_BBW_21_2.0_D'), ascending=False)
        score_expansion_daily = 1 - score_squeeze_daily
        score_expansion_weekly = 1 - score_squeeze_weekly
        score_expansion_momentum = 1 - score_squeeze_momentum
        score_vol_accel_up = self._normalize_score(df.get('ACCEL_5_BBW_21_2.0_D'), ascending=True)
        score_vol_accel_down = self._normalize_score(df.get('ACCEL_5_BBW_21_2.0_D'), ascending=False)
        
        states['SCORE_VOL_COMPRESSION_B'] = score_squeeze_daily
        states['SCORE_VOL_COMPRESSION_A'] = (score_squeeze_daily * score_squeeze_weekly).astype(np.float32)
        states['SCORE_VOL_COMPRESSION_S'] = (states['SCORE_VOL_COMPRESSION_A'] * score_squeeze_momentum).astype(np.float32)
        
        states['SCORE_VOL_EXPANSION_B'] = score_expansion_daily
        states['SCORE_VOL_EXPANSION_A'] = (score_expansion_daily * score_expansion_weekly).astype(np.float32)
        states['SCORE_VOL_EXPANSION_S'] = (states['SCORE_VOL_EXPANSION_A'] * score_expansion_momentum).astype(np.float32)
        
        states['SCORE_VOL_TIPPING_POINT_BOTTOM_OPP'] = (states['SCORE_VOL_COMPRESSION_S'] * score_vol_accel_up).astype(np.float32)
        states['SCORE_VOL_TIPPING_POINT_TOP_RISK'] = (states['SCORE_VOL_EXPANSION_S'] * score_vol_accel_down).astype(np.float32)
        
        hurst_score = self._normalize_score(df.get('hurst_120d_D'))
        states['SCORE_TRENDING_REGIME'] = hurst_score
        states['SCORE_VOL_BREAKOUT_POTENTIAL_S'] = states['SCORE_VOL_COMPRESSION_S'] * hurst_score
        states['SCORE_VOL_BREAKDOWN_RISK_S'] = states['SCORE_VOL_EXPANSION_S'] * (1 - hurst_score)
        return states

    def diagnose_classic_indicators_atomics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 · 原子信号版】经典指标原子信号诊断 (战术模块，予以保留)
        """
        states = {}
        p = get_params_block(self.strategy, 'classic_indicator_params')
        if not get_param_value(p.get('enabled'), True): return states
        
        candle_body_up = (df.get('close_D', 0) - df.get('open_D', 0)).clip(lower=0)
        candle_body_down = (df.get('open_D', 0) - df.get('close_D', 0)).clip(lower=0)
        score_price_up_strength = self._normalize_score(candle_body_up)
        score_price_down_strength = self._normalize_score(candle_body_down)
        
        score_vol_slope_up = self._normalize_score(df.get('SLOPE_5_volume_D', pd.Series(0.5, index=df.index)).clip(lower=0))
        score_vol_accel_up = self._normalize_score(df.get('ACCEL_5_volume_D', pd.Series(0.5, index=df.index)).clip(lower=0))
        score_volume_igniting = score_vol_slope_up * score_vol_accel_up
        
        states['SCORE_VOL_PRICE_IGNITION_UP'] = score_price_up_strength * score_volume_igniting
        states['SCORE_VOL_PRICE_PANIC_DOWN_RISK'] = score_price_down_strength * score_volume_igniting
        return states

    # 移除了所有被重构为 _calculate...health 和 diagnose_unified_foundation_signals 的旧方法
    # 例如 diagnose_ultimate_foundation_signals, diagnose_ema_synergy, diagnose_oscillator_intelligence 等...
