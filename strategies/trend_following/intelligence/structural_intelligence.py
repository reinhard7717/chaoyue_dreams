# 文件: strategies/trend_following/intelligence/structural_intelligence.py
# 结构情报模块 (均线, 箱体, 平台, 趋势)
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following.utils import get_params_block, get_param_value, create_persistent_state

class StructuralIntelligence:
    def __init__(self, strategy_instance, dynamic_thresholds: Dict):
        """
        初始化结构情报模块。
        :param strategy_instance: 策略主实例的引用。
        :param dynamic_thresholds: 动态阈值字典。
        """
        self.strategy = strategy_instance
        self.dynamic_thresholds = dynamic_thresholds

    def _normalize_score(self, series: pd.Series, window: int, target_index: pd.Index, ascending: bool = True) -> pd.Series:
        """
        【V1.0 新增】计算一个系列在滚动窗口内的归一化得分 (0-1)。
        """
        if series is None or series.isnull().all():
            return pd.Series(0.5, index=target_index)

        return series.rolling(
            window=window, 
            min_periods=int(window * 0.2)
        ).rank(
            pct=True, 
            ascending=ascending
        ).fillna(0.5).astype(np.float32)

    def diagnose_structural_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 终极信号版】结构情报分析总指挥
        - 核心重构: 遵循终极信号范式，本模块不再返回一堆零散的原子信号。
                      现在只调用唯一的终极信号引擎 `diagnose_ultimate_structural_signals`，
                      并将其产出的16个S+/S/A/B级信号作为本模块的最终输出。
        - 收益: 架构与其他情报模块完全统一，极大提升了信号质量和架构清晰度。
        """
        # print("      -> [结构情报分析总指挥 V2.0 终极信号版] 启动...")
        # 直接调用终极信号引擎，并将其结果作为本模块的唯一输出
        ultimate_structural_states = self.diagnose_ultimate_structural_signals(df)
        # print(f"      -> [结构情报分析总指挥 V2.0] 分析完毕，共生成 {len(ultimate_structural_states)} 个终极结构信号。")
        return ultimate_structural_states

    def diagnose_ultimate_structural_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V7.0 · 对称逻辑版】终极结构信号诊断模块
        - 核心重构 (本次修改):
          - [哲学升维] 彻底废除 `1 - bullish` 的粗暴逻辑，为“看跌”信号建立完全独立且对称的计算体系。
          - [四维输出] 所有健康度组件现在输出 (静多, 动多, 静空, 动空) 四个维度的健康分。
          - [独立融合] 主引擎独立融合生成四个全局健康度，确保多空信号的计算互不干扰。
        - 收益: 实现了与所有其他情报引擎在哲学和代码结构上的完全统一，信号质量达到最终形态。
        """
        print("        -> [终极结构信号诊断模块 V7.0 · 对称逻辑版] 启动...") # [代码修改] 更新版本号和说明
        states = {}
        p_conf = get_params_block(self.strategy, 'structural_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True): return states
        exponent = get_param_value(p_conf.get('final_score_exponent'), 1.0)
        
        # --- 1. 定义权重与参数 ---
        resonance_tf_weights = {'short': 0.2, 'medium': 0.5, 'long': 0.3}
        reversal_tf_weights = {'short': 0.6, 'medium': 0.3, 'long': 0.1}
        dynamic_weights = {'slope': 0.6, 'accel': 0.4}
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
        health_data = {
            'bullish_static': [], 'bullish_dynamic': [],
            'bearish_static': [], 'bearish_dynamic': []
        }
        
        calculators = {
            'ma': self._calculate_ma_health,
            'mechanics': self._calculate_mechanics_health,
            'mtf': self._calculate_mtf_health,
            'pattern': self._calculate_pattern_health,
        }

        for name, calculator in calculators.items():
            s_bull, d_bull, s_bear, d_bear = calculator(df, periods, norm_window, dynamic_weights)
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
        for prefix, score in [('SCORE_STRUCTURE_BULLISH_RESONANCE', overall_bullish_resonance), ('SCORE_STRUCTURE_BOTTOM_REVERSAL', final_bottom_reversal_score),
                              ('SCORE_STRUCTURE_BEARISH_RESONANCE', overall_bearish_resonance), ('SCORE_STRUCTURE_TOP_REVERSAL', final_top_reversal_score)]:
            states[f'{prefix}_S_PLUS'] = (score ** exponent).astype(np.float32)
            states[f'{prefix}_S'] = (states[f'{prefix}_S_PLUS'] * 0.8).astype(np.float32)
            states[f'{prefix}_A'] = (states[f'{prefix}_S_PLUS'] * 0.6).astype(np.float32)
            states[f'{prefix}_B'] = (states[f'{prefix}_S_PLUS'] * 0.4).astype(np.float32)
        
        return states

    # ==============================================================================
    # [代码修改] 以下为重构后的健康度组件计算器，现在返回四维健康度
    # ==============================================================================

    def _calculate_ma_health(self, df: pd.DataFrame, periods: list, norm_window: int, dynamic_weights: Dict) -> Tuple[Dict, Dict, Dict, Dict]:
        """【V2.0 · 对称逻辑版】计算MA支柱的四维健康度"""
        s_bull, d_bull, s_bear, d_bear = {}, {}, {}, {}
        for p in periods:
            s_bull[p] = self._normalize_score(df.get(f'price_vs_ma_{p}_D'), norm_window, df.index, ascending=True)
            s_bear[p] = self._normalize_score(df.get(f'price_vs_ma_{p}_D'), norm_window, df.index, ascending=False)
            
            static_col = f'EMA_{p}_D' if p > 1 else 'close_D'
            slope_col = f'SLOPE_{p}_{static_col}'
            accel_col = f'ACCEL_{p}_{static_col}'
            
            slope_score = self._normalize_score(df.get(slope_col), norm_window, df.index, ascending=True)
            accel_score = self._normalize_score(df.get(accel_col), norm_window, df.index, ascending=True)
            d_bull[p] = slope_score * dynamic_weights['slope'] + accel_score * dynamic_weights['accel']
            
            slope_score_neg = self._normalize_score(df.get(slope_col), norm_window, df.index, ascending=False)
            accel_score_neg = self._normalize_score(df.get(accel_col), norm_window, df.index, ascending=False)
            d_bear[p] = slope_score_neg * dynamic_weights['slope'] + accel_score_neg * dynamic_weights['accel']
        return s_bull, d_bull, s_bear, d_bear

    def _calculate_mechanics_health(self, df: pd.DataFrame, periods: list, norm_window: int, dynamic_weights: Dict) -> Tuple[Dict, Dict, Dict, Dict]:
        """【V2.0 · 对称逻辑版】计算力学支柱的四维健康度"""
        s_bull, d_bull, s_bear, d_bear = {}, {}, {}, {}
        static_bull_energy = self._normalize_score(df.get('energy_ratio_D'), norm_window, df.index, ascending=True)
        static_bear_energy = self._normalize_score(df.get('energy_ratio_D'), norm_window, df.index, ascending=False)
        for p in periods:
            s_bull[p] = static_bull_energy
            s_bear[p] = static_bear_energy
            
            cost_slope_bull = self._normalize_score(df.get(f'SLOPE_{p}_peak_cost_D'), norm_window, df.index, ascending=True)
            conc_lock_bull = self._normalize_score(df.get(f'SLOPE_{p}_concentration_90pct_D'), norm_window, df.index, ascending=False)
            d_bull[p] = (cost_slope_bull * conc_lock_bull)**0.5
            
            cost_slope_bear = self._normalize_score(df.get(f'SLOPE_{p}_peak_cost_D'), norm_window, df.index, ascending=False)
            conc_lock_bear = self._normalize_score(df.get(f'SLOPE_{p}_concentration_90pct_D'), norm_window, df.index, ascending=True)
            d_bear[p] = (cost_slope_bear * conc_lock_bear)**0.5
        return s_bull, d_bull, s_bear, d_bear

    def _calculate_mtf_health(self, df: pd.DataFrame, periods: list, norm_window: int, dynamic_weights: Dict) -> Tuple[Dict, Dict, Dict, Dict]:
        """【V2.0 · 对称逻辑版】计算MTF支柱的四维健康度"""
        s_bull, d_bull, s_bear, d_bear = {}, {}, {}, {}
        
        # 静态健康度：周线结构对齐度
        weekly_cols = [col for col in df.columns if 'EMA' in col and col.endswith('_W')]
        if len(weekly_cols) > 1:
            bull_align = np.mean([ (df[weekly_cols[i]] > df[weekly_cols[i+1]]).values for i in range(len(weekly_cols)-1) ], axis=0)
            bear_align = np.mean([ (df[weekly_cols[i]] < df[weekly_cols[i+1]]).values for i in range(len(weekly_cols)-1) ], axis=0)
            static_bull_score = pd.Series(bull_align, index=df.index, dtype=np.float32)
            static_bear_score = pd.Series(bear_align, index=df.index, dtype=np.float32)
        else:
            static_bull_score = static_bear_score = pd.Series(0.5, index=df.index, dtype=np.float32)

        # 动态健康度：周线斜率和加速度
        def get_avg_score(cols: list[str], asc: bool) -> pd.Series:
            if not cols: return pd.Series(0.5, index=df.index, dtype=np.float32)
            scores = [self._normalize_score(df.get(c), norm_window, df.index, ascending=asc).values for c in cols]
            return pd.Series(np.mean(np.stack(scores, axis=0), axis=0), index=df.index, dtype=np.float32)

        weekly_slope_cols = [c for c in df.columns if 'SLOPE' in c and c.endswith('_W')]
        weekly_accel_cols = [c for c in df.columns if 'ACCEL' in c and c.endswith('_W')]
        
        dynamic_bull_score = (get_avg_score(weekly_slope_cols, True) * get_avg_score(weekly_accel_cols, True))**0.5
        dynamic_bear_score = (get_avg_score(weekly_slope_cols, False) * get_avg_score(weekly_accel_cols, False))**0.5

        for p in periods:
            s_bull[p], s_bear[p] = static_bull_score, static_bear_score
            d_bull[p], d_bear[p] = dynamic_bull_score, dynamic_bear_score
        return s_bull, d_bull, s_bear, d_bear

    def _calculate_pattern_health(self, df: pd.DataFrame, periods: list, norm_window: int, dynamic_weights: Dict) -> Tuple[Dict, Dict, Dict, Dict]:
        """【V2.0 · 对称逻辑版】计算形态支柱的四维健康度"""
        s_bull, d_bull, s_bear, d_bear = {}, {}, {}, {}
        
        # 静态健康度：是否存在吸筹/盘整形态 vs 派发形态
        is_accumulation = df.get('is_accumulation_D', 0).astype(float)
        is_consolidation = df.get('is_consolidation_D', 0).astype(float)
        # 假设存在 is_distribution_D，若无则为0
        is_distribution = df.get('is_distribution_D', 0).astype(float)
        static_bull_score = pd.Series(np.maximum(is_accumulation, is_consolidation), index=df.index).replace(0, 0.5)
        static_bear_score = pd.Series(is_distribution, index=df.index).replace(0, 0.5)

        # 动态健康度：是否存在突破 vs 破位事件
        is_breakthrough = df.get('is_breakthrough_D', 0).astype(float)
        # 假设存在 is_breakdown_D，若无则为0
        is_breakdown = df.get('is_breakdown_D', 0).astype(float)
        dynamic_bull_score = pd.Series(is_breakthrough, index=df.index).replace(0, 0.5)
        dynamic_bear_score = pd.Series(is_breakdown, index=df.index).replace(0, 0.5)

        for p in periods:
            s_bull[p], s_bear[p] = static_bull_score, static_bear_score
            d_bull[p], d_bear[p] = dynamic_bull_score, dynamic_bear_score
        return s_bull, d_bull, s_bear, d_bear






















