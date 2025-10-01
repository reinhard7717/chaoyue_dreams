# 文件: strategies/trend_following/intelligence/structural_intelligence.py
# 结构情报模块 (均线, 箱体, 平台, 趋势)
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following.utils import transmute_health_to_ultimate_signals, get_params_block, get_param_value, calculate_holographic_dynamics, normalize_score

class StructuralIntelligence:
    def __init__(self, strategy_instance, dynamic_thresholds: Dict):
        """
        初始化结构情报模块。
        :param strategy_instance: 策略主实例的引用。
        :param dynamic_thresholds: 动态阈值字典。
        """
        self.strategy = strategy_instance
        self.dynamic_thresholds = dynamic_thresholds

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
        【V16.0 · 圣杯契约版】
        - 核心革命: 不再读取本地的、重复的合成参数，而是从最高指挥部获取唯一的“圣杯”配置
                      (`ultimate_signal_synthesis_params`)，并将其传递给中央合成引擎。
        """
        states = {}
        p_conf = get_params_block(self.strategy, 'structural_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True): return states
        # 获取中央“圣杯”配置
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        dynamic_weights = {'slope': 0.6, 'accel': 0.4}
        periods = get_param_value(p_synthesis.get('periods'), [1, 5, 13, 21, 55])
        norm_window = get_param_value(p_synthesis.get('norm_window'), 55)
        health_data = { 's_bull': [], 's_bear': [], 'd_intensity': [] } 
        calculators = { 'ma': self._calculate_ma_health, 'mechanics': self._calculate_mechanics_health, 'mtf': self._calculate_mtf_health, 'pattern': self._calculate_pattern_health }
        for name, calculator in calculators.items():
            s_bull, s_bear, d_intensity = calculator(df, periods, norm_window, dynamic_weights)
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
        self.strategy.atomic_states['__STRUCTURE_overall_health'] = overall_health
        # 传入唯一的“圣杯”配置
        ultimate_signals = transmute_health_to_ultimate_signals(
            df=df,
            atomic_states=self.strategy.atomic_states,
            overall_health=overall_health,
            params=p_synthesis,
            domain_prefix="STRUCTURE"
        )
        states.update(ultimate_signals)
        return states

    # ==============================================================================
    # 以下为重构后的健康度组件计算器，现在返回四维健康度
    # ==============================================================================

    def _calculate_ma_health(self, df: pd.DataFrame, periods: list, norm_window: int, dynamic_weights: Dict) -> Tuple[Dict, Dict, Dict]:
        """【V3.4 · 雅典娜之镜版】计算MA支柱的三维健康度"""
        s_bull, s_bear, d_intensity = {}, {}, {}
        # 严格使用斐波那契数列作为均线周期
        ma_periods = [5, 13, 21, 55]
        
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
        
        # 计算一阶和二阶健康度
        for p in ma_periods:
            # 严格使用项目定义的斜率和加速度列名
            slope_col = f'SLOPE_{p}_EMA_{p}_D' if p != 1 else f'SLOPE_1_close_D'
            accel_col = f'ACCEL_{p}_EMA_{p}_D' if p != 1 else f'ACCEL_1_close_D'
            if slope_col in df.columns:
                bipolar_slope = normalize_to_bipolar(df[slope_col], df.index, norm_window)
                slope_health_scores.append((bipolar_slope + 1) / 2.0)
            if accel_col in df.columns:
                bipolar_accel = normalize_to_bipolar(df[accel_col], df.index, norm_window)
                accel_health_scores.append((bipolar_accel + 1) / 2.0)
                
        # [代码新增] 计算关系加速度健康度
        ma_pairs = [(5, 21), (13, 55)]
        for short_p, long_p in ma_pairs:
            short_ma_col, long_ma_col = f'EMA_{short_p}_D', f'EMA_{long_p}_D'
            if short_ma_col in df.columns and long_ma_col in df.columns:
                spread = df[short_ma_col] - df[long_ma_col]
                spread_accel = spread.diff(3).diff(3).fillna(0) # 二阶求导
                bipolar_rel_accel = normalize_to_bipolar(spread_accel, df.index, norm_window)
                relational_health_scores.append((bipolar_rel_accel + 1) / 2.0)

        # --- 融合四维健康度 ---
        avg_slope_health = pd.concat(slope_health_scores, axis=1).mean(axis=1).fillna(0.5) if slope_health_scores else pd.Series(0.5, index=df.index)
        avg_accel_health = pd.concat(accel_health_scores, axis=1).mean(axis=1).fillna(0.5) if accel_health_scores else pd.Series(0.5, index=df.index)
        avg_relational_health = pd.concat(relational_health_scores, axis=1).mean(axis=1).fillna(0.5) if relational_health_scores else pd.Series(0.5, index=df.index)

        # 最终静态看涨分是四维健康的几何平均
        static_bull_score = (alignment_score * avg_slope_health * avg_accel_health * avg_relational_health)**0.25

        for p in periods:
            s_bull[p] = static_bull_score
            s_bear[p] = static_bear_score
            static_col = f'EMA_{p}' if p > 1 else 'close'
            bull_holo, bear_holo = calculate_holographic_dynamics(df, static_col, norm_window)
            d_intensity[p] = (bull_holo + bear_holo) / 2.0

        return s_bull, s_bear, d_intensity

    def _calculate_mechanics_health(self, df: pd.DataFrame, periods: list, norm_window: int, dynamic_weights: Dict) -> Tuple[Dict, Dict, Dict]:
        """【V3.1 · 调用适配版】计算力学支柱的三维健康度"""
        s_bull, s_bear, d_intensity = {}, {}, {}
        static_bull_energy = normalize_score(df.get('energy_ratio_D'), df.index, norm_window, ascending=True)
        static_bear_energy = normalize_score(df.get('energy_ratio_D'), df.index, norm_window, ascending=False)
        
        # 调用中央引擎获取元组，然后在调用处进行融合
        bull_cost, bear_cost = calculate_holographic_dynamics(df, 'peak_cost', norm_window)
        cost_mom_strength = (bull_cost + bear_cost) / 2.0
        
        bull_conc, bear_conc = calculate_holographic_dynamics(df, 'concentration_90pct', norm_window)
        conc_mom_strength = (bull_conc + bear_conc) / 2.0
        
        unified_d_intensity = (cost_mom_strength * conc_mom_strength)**0.5

        for p in periods:
            s_bull[p] = static_bull_energy
            s_bear[p] = static_bear_energy
            d_intensity[p] = unified_d_intensity
            
        return s_bull, s_bear, d_intensity

    def _calculate_mtf_health(self, df: pd.DataFrame, periods: list, norm_window: int, dynamic_weights: Dict) -> Tuple[Dict, Dict, Dict]:
        """
        【V2.2 · 动态分统一版】计算MTF(多时间框架)支柱的三维健康度
        - 核心重构: 废除 d_bull 和 d_bear，统一返回中性的“动态强度分” d_intensity。
        """
        # 更新方法签名和初始化，统一返回 d_intensity
        s_bull, s_bear, d_intensity = {}, {}, {}

        weekly_cols = [col for col in df.columns if 'EMA' in col and col.endswith('_W')]
        if len(weekly_cols) > 1:
            bull_align_matrix = np.stack([(df[weekly_cols[i]] > df[weekly_cols[i+1]]).values for i in range(len(weekly_cols)-1)], axis=0)
            bear_align_matrix = np.stack([(df[weekly_cols[i]] < df[weekly_cols[i+1]]).values for i in range(len(weekly_cols)-1)], axis=0)
            static_bull_score = pd.Series(np.mean(bull_align_matrix, axis=0), index=df.index, dtype=np.float32)
            static_bear_score = pd.Series(np.mean(bear_align_matrix, axis=0), index=df.index, dtype=np.float32)
        else:
            static_bull_score = static_bear_score = pd.Series(0.5, index=df.index, dtype=np.float32)

        # 辅助函数现在计算中性的强度分
        def get_avg_strength_score(cols: list[str]) -> pd.Series:
            if not cols: return pd.Series(0.5, index=df.index, dtype=np.float32)
            # 使用 .abs() 获取强度
            scores_matrix = np.stack([normalize_score(df.get(c).abs(), df.index, norm_window, ascending=True).values for c in cols], axis=0)
            return pd.Series(np.mean(scores_matrix, axis=0), index=df.index, dtype=np.float32)

        weekly_slope_cols = [c for c in df.columns if 'SLOPE' in c and c.endswith('_W')]
        weekly_accel_cols = [c for c in df.columns if 'ACCEL' in c and c.endswith('_W')]
        
        # 计算统一的、中性的动态强度分 d_intensity
        dynamic_intensity_score = (get_avg_strength_score(weekly_slope_cols) * get_avg_strength_score(weekly_accel_cols))**0.5

        s_bull = {p: static_bull_score for p in periods}
        s_bear = {p: static_bear_score for p in periods}
        d_intensity = {p: dynamic_intensity_score for p in periods}
        
        # 返回符合新协议的三元组
        return s_bull, s_bear, d_intensity

    def _calculate_pattern_health(self, df: pd.DataFrame, periods: list, norm_window: int, dynamic_weights: Dict) -> Tuple[Dict, Dict, Dict]:
        """
        【V2.2 · 动态分统一版】计算形态支柱的三维健康度
        - 核心重构: 废除 d_bull 和 d_bear，统一返回中性的“动态强度分” d_intensity。
        """
        # 更新方法签名和初始化，统一返回 d_intensity
        s_bull, s_bear, d_intensity = {}, {}, {}

        is_accumulation = df.get('is_accumulation_D', 0).astype(float)
        is_consolidation = df.get('is_consolidation_D', 0).astype(float)
        is_distribution = df.get('is_distribution_D', 0).astype(float)
        static_bull_score = pd.Series(np.maximum(is_accumulation, is_consolidation), index=df.index).replace(0, 0.5)
        static_bear_score = pd.Series(is_distribution, index=df.index).replace(0, 0.5)

        # 动态强度分现在衡量“事件发生的强度”，而不是方向
        is_breakthrough = df.get('is_breakthrough_D', 0).astype(float)
        is_breakdown = df.get('is_breakdown_D', 0).astype(float)
        # 任何一种突破或破位事件都代表了动态强度的增加
        dynamic_intensity_score = pd.Series(np.maximum(is_breakthrough, is_breakdown), index=df.index).replace(0, 0.5)

        s_bull = {p: static_bull_score for p in periods}
        s_bear = {p: static_bear_score for p in periods}
        d_intensity = {p: dynamic_intensity_score for p in periods}
        
        # 返回符合新协议的三元组
        return s_bull, s_bear, d_intensity






















