# 文件: strategies/trend_following/intelligence/chip_intelligence.py
# 筹码情报模块
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following.utils import transmute_health_to_ultimate_signals, get_params_block, get_param_value, normalize_score, normalize_to_bipolar, calculate_holographic_dynamics

class ChipIntelligence:
    def __init__(self, strategy_instance, dynamic_thresholds: Dict):
        """
        初始化筹码情报模块。
        :param strategy_instance: 策略主实例的引用。
        :param dynamic_thresholds: 动态阈值字典。
        """
        self.strategy = strategy_instance
        self.dynamic_thresholds = dynamic_thresholds

    def run_chip_intelligence_command(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V501.0 · 剧本重构版】筹码情报最高司令部
        - 核心升级: 废除旧的、分离的恐慌投降诊断三部曲，改为调用统一的、基于关系元分析的
                      全新剧本诊断引擎。
        """
        all_chip_states = {}
        
        # 步骤 1: 执行统一的终极信号引擎 (不变)
        unified_states = self.diagnose_unified_chip_signals(df)
        all_chip_states.update(unified_states)

        # 步骤 2: 执行升级后的“吸筹剧本”诊断模块 (不变)
        accumulation_states = self.diagnose_accumulation_playbooks(df)
        all_chip_states.update(accumulation_states)

        # 步骤 3: 调用全新的、统一的“恐慌投降反转”剧本诊断引擎
        capitulation_states = self.diagnose_playbook_capitulation_reversal(df)
        all_chip_states.update(capitulation_states)

        return all_chip_states

    def diagnose_unified_chip_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V22.0 · 动态融合修复版】
        - 核心修复: 彻底重构健康度融合逻辑，解决“静态权重”与“动态支柱”的致命错配问题。
                      确保在部分支柱数据缺失时，系统依然能稳健地进行动态加权融合。
        """
        states = {}
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True): return states
        
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        dynamic_weights = {'slope': 0.6, 'accel': 0.4}
        pillar_weights = get_param_value(p_conf.get('pillar_weights'), {})
        periods = get_param_value(p_synthesis.get('periods'), [1, 5, 13, 21, 55])
        norm_window = get_param_value(p_synthesis.get('norm_window'), 55)
        
        health_data = { 's_bull': [], 's_bear': [], 'd_intensity': [] } 
        calculators = {
            'quantitative': self._calculate_quantitative_health,
            'advanced': self._calculate_advanced_dynamics_health,
            'internal': self._calculate_internal_structure_health,
            'holder': self._calculate_holder_behavior_health,
            'fault': self._calculate_fault_health,
        }
        
        # [代码修改] 保持不变，但现在消费的是具有“韧性骨架”的健康度字典
        for name, calculator in calculators.items():
            s_bull, s_bear, d_intensity = calculator(df, norm_window, dynamic_weights, periods)
            health_data['s_bull'].append((name, s_bull)) 
            health_data['s_bear'].append((name, s_bear)) 
            health_data['d_intensity'].append((name, d_intensity))
            
        overall_health = {}
        use_equal_weights = not pillar_weights or sum(pillar_weights.values()) == 0
        
        for health_type, health_sources_with_names in [
            ('s_bull', health_data['s_bull']),
            ('s_bear', health_data['s_bear']),
            ('d_intensity', health_data['d_intensity'])
        ]:
            overall_health[health_type] = {}
            for p in periods:
                # [代码修复] 步骤一：并行构建有效支柱列表和有效权重列表
                valid_pillars = []
                valid_weights = []
                for name, pillar_dict in health_sources_with_names:
                    if p in pillar_dict:
                        valid_pillars.append(pillar_dict[p].values)
                        # 仅当支柱有效时，才将其权重加入列表
                        valid_weights.append(pillar_weights.get(name, 0))

                # [代码修复] 步骤二：如果没有任何有效支柱，则填充默认值并跳到下一个周期
                if not valid_pillars:
                    overall_health[health_type][p] = pd.Series(0.5, index=df.index, dtype=np.float32)
                    continue

                # [代码修复] 步骤三：基于动态生成的权重列表进行安全融合
                stacked_values = np.stack(valid_pillars, axis=0)
                
                if use_equal_weights:
                    # 如果未配置权重，使用几何平均
                    fused_values = np.prod(stacked_values, axis=0) ** (1.0 / stacked_values.shape[0])
                else:
                    # 动态创建与 valid_pillars 长度完全匹配的权重数组
                    weights_array = np.array(valid_weights)
                    total_weight = weights_array.sum()
                    
                    if total_weight > 0:
                        # 使用归一化后的权重进行加权几何平均
                        normalized_weights = weights_array / total_weight
                        # 使用 np.maximum 避免 log(0)
                        safe_stacked_values = np.maximum(stacked_values, 1e-9)
                        fused_values = np.exp(np.sum(np.log(safe_stacked_values) * normalized_weights[:, np.newaxis], axis=0))
                    else:
                        # 如果所有有效支柱的权重都为0，则退化为标准几何平均
                        fused_values = np.prod(stacked_values, axis=0) ** (1.0 / stacked_values.shape[0])

                overall_health[health_type][p] = pd.Series(fused_values, index=df.index, dtype=np.float32)

        self.strategy.atomic_states['__CHIP_overall_health'] = overall_health
        
        ultimate_signals = transmute_health_to_ultimate_signals(
            df=df,
            atomic_states=self.strategy.atomic_states,
            overall_health=overall_health,
            params=p_synthesis,
            domain_prefix="CHIP"
        )
        states.update(ultimate_signals)
        return states

    # ==============================================================================
    # 以下为重构后的健康度组件计算器，现在返回四维健康度
    # ==============================================================================

    def _calculate_quantitative_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict[int, pd.Series], Dict[int, pd.Series], Dict[int, pd.Series]]:
        """【V5.1 · 韧性骨架版】计算基础量化维度的三维健康度"""
        s_bull, s_bear, d_intensity = {}, {}, {}
        
        # [代码修复] 增加数据检查和韧性返回
        required_cols = ['concentration_90pct_D', 'chip_health_score_D', 'peak_cost_D']
        if any(col not in df.columns for col in required_cols):
            default_series = pd.Series(0.5, index=df.index, dtype=np.float32)
            for p in periods:
                s_bull[p] = default_series
                s_bear[p] = default_series
                d_intensity[p] = default_series
            return s_bull, s_bear, d_intensity

        # 步骤一：计算原始的、纯粹的筹码静态健康度
        static_bull_conc = normalize_score(df.get('concentration_90pct_D'), df.index, norm_window, ascending=False)
        static_bull_health = normalize_score(df.get('chip_health_score_D'), df.index, norm_window, ascending=True)
        chip_static_bull = (static_bull_conc * static_bull_health)**0.5
        
        static_bear_conc = normalize_score(df.get('concentration_90pct_D'), df.index, norm_window, ascending=True)
        static_bear_health = normalize_score(df.get('chip_health_score_D'), df.index, norm_window, ascending=False)
        chip_static_bear = (static_bear_conc * static_bear_health)**0.5

        # 步骤二：获取均线趋势上下文分数
        ma_context_score = self._calculate_ma_trend_context(df, [5, 13, 21, 55])

        # 步骤三：构建融合了趋势上下文的“瞬时关系快照分”
        bullish_snapshot_score = (chip_static_bull * ma_context_score)
        bearish_snapshot_score = (chip_static_bear * (1 - ma_context_score))

        # 步骤四：对快照分进行关系元分析，得到最终的动态强度分
        unified_d_intensity = self._perform_chip_relational_meta_analysis(df, bullish_snapshot_score)

        # 步骤五：更新输出
        for p in periods:
            s_bull[p] = bullish_snapshot_score
            s_bear[p] = bearish_snapshot_score
            d_intensity[p] = unified_d_intensity
        
        return s_bull, s_bear, d_intensity

    def _calculate_advanced_dynamics_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict[int, pd.Series], Dict[int, pd.Series], Dict[int, pd.Series]]:
        """【V5.1 · 韧性骨架版】计算高级动态维度的三维健康度"""
        s_bull, s_bear, d_intensity = {}, {}, {}
        required_cols = ['peak_control_ratio_D', 'peak_strength_ratio_D', 'peak_stability_D', 'is_multi_peak_D']
        
        # [代码修复] 增加数据检查和韧性返回
        if any(col not in df.columns for col in required_cols):
            default_series = pd.Series(0.5, index=df.index, dtype=np.float32)
            for p in periods:
                s_bull[p] = default_series
                s_bear[p] = default_series
                d_intensity[p] = default_series
            return s_bull, s_bear, d_intensity

        # 步骤一：计算原始的、纯粹的筹码静态健康度
        is_multi_peak_series = df.get('is_multi_peak_D', pd.Series(0.0, index=df.index)).astype(float)
        chip_static_bull = (normalize_score(df.get('peak_control_ratio_D'), df.index, norm_window) * normalize_score(df.get('peak_strength_ratio_D'), df.index, norm_window) * normalize_score(df.get('peak_stability_D'), df.index, norm_window))**(1/3)
        chip_static_bear = (normalize_score(df.get('peak_control_ratio_D'), df.index, norm_window, ascending=False) * normalize_score(df.get('peak_strength_ratio_D'), df.index, norm_window, ascending=False) * normalize_score(df.get('peak_stability_D'), df.index, norm_window, ascending=False) * is_multi_peak_series)**(1/4)

        # 步骤二：获取均线趋势上下文分数
        ma_context_score = self._calculate_ma_trend_context(df, [5, 13, 21, 55])

        # 步骤三：构建融合了趋势上下文的“瞬时关系快照分”
        bullish_snapshot_score = (chip_static_bull * ma_context_score)
        bearish_snapshot_score = (chip_static_bear * (1 - ma_context_score))

        # 步骤四：对快照分进行关系元分析，得到最终的动态强度分
        unified_d_intensity = self._perform_chip_relational_meta_analysis(df, bullish_snapshot_score)

        # 步骤五：更新输出
        for p in periods:
            s_bull[p] = bullish_snapshot_score
            s_bear[p] = bearish_snapshot_score
            d_intensity[p] = unified_d_intensity
            
        return s_bull, s_bear, d_intensity

    def _calculate_internal_structure_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict[int, pd.Series], Dict[int, pd.Series], Dict[int, pd.Series]]:
        """【V5.1 · 韧性骨架版】计算内部结构维度的三维健康度"""
        s_bull, s_bear, d_intensity = {}, {}, {}
        required_cols = ['concentration_70pct_D', 'support_below_D', 'pressure_above_D']
        
        # [代码修复] 增加数据检查和韧性返回
        if any(col not in df.columns for col in required_cols):
            default_series = pd.Series(0.5, index=df.index, dtype=np.float32)
            for p in periods:
                s_bull[p] = default_series
                s_bear[p] = default_series
                d_intensity[p] = default_series
            return s_bull, s_bear, d_intensity

        # 步骤一：计算原始的、纯粹的筹码静态健康度
        chip_static_bull = (normalize_score(df.get('concentration_70pct_D'), df.index, norm_window, ascending=False) * (normalize_score(df.get('support_below_D'), df.index, norm_window) * normalize_score(df.get('pressure_above_D'), df.index, norm_window, ascending=False))**0.5)**0.5
        chip_static_bear = (normalize_score(df.get('concentration_70pct_D'), df.index, norm_window, ascending=True) * (normalize_score(df.get('support_below_D'), df.index, norm_window, ascending=False) * normalize_score(df.get('pressure_above_D'), df.index, norm_window, ascending=True))**0.5)**0.5

        # 步骤二：获取均线趋势上下文分数
        ma_context_score = self._calculate_ma_trend_context(df, [5, 13, 21, 55])

        # 步骤三：构建融合了趋势上下文的“瞬时关系快照分”
        bullish_snapshot_score = (chip_static_bull * ma_context_score)
        bearish_snapshot_score = (chip_static_bear * (1 - ma_context_score))

        # 步骤四：对快照分进行关系元分析，得到最终的动态强度分
        unified_d_intensity = self._perform_chip_relational_meta_analysis(df, bullish_snapshot_score)

        # 步骤五：更新输出
        for p in periods:
            s_bull[p] = bullish_snapshot_score
            s_bear[p] = bearish_snapshot_score
            d_intensity[p] = unified_d_intensity
            
        return s_bull, s_bear, d_intensity

    def _calculate_holder_behavior_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict[int, pd.Series], Dict[int, pd.Series], Dict[int, pd.Series]]:
        """【V5.1 · 韧性骨架版】计算持仓者行为维度的三维健康度"""
        s_bull, s_bear, d_intensity = {}, {}, {}
        required_cols = ['cost_divergence_D', 'winner_profit_margin_D', 'total_winner_rate_D', 'turnover_from_winners_ratio_D']
        
        # [代码修复] 增加数据检查和韧性返回
        if any(col not in df.columns for col in required_cols):
            default_series = pd.Series(0.5, index=df.index, dtype=np.float32)
            for p in periods:
                s_bull[p] = default_series
                s_bear[p] = default_series
                d_intensity[p] = default_series
            return s_bull, s_bear, d_intensity

        # 步骤一：计算原始的、纯粹的筹码静态健康度
        chip_static_bull = (normalize_score(df.get('cost_divergence_D'), df.index, norm_window) * normalize_score(df.get('winner_profit_margin_D'), df.index, norm_window) * normalize_score(df.get('total_winner_rate_D'), df.index, norm_window) * normalize_score(df.get('turnover_from_winners_ratio_D'), df.index, norm_window, ascending=False))**(1/4)
        chip_static_bear = (normalize_score(df.get('cost_divergence_D'), df.index, norm_window, ascending=False) * normalize_score(df.get('winner_profit_margin_D'), df.index, norm_window, ascending=False) * normalize_score(df.get('total_winner_rate_D'), df.index, norm_window, ascending=False) * normalize_score(df.get('turnover_from_winners_ratio_D'), df.index, norm_window, ascending=True))**(1/4)

        # 步骤二：获取均线趋势上下文分数
        ma_context_score = self._calculate_ma_trend_context(df, [5, 13, 21, 55])

        # 步骤三：构建融合了趋势上下文的“瞬时关系快照分”
        bullish_snapshot_score = (chip_static_bull * ma_context_score)
        bearish_snapshot_score = (chip_static_bear * (1 - ma_context_score))

        # 步骤四：对快照分进行关系元分析，得到最终的动态强度分
        unified_d_intensity = self._perform_chip_relational_meta_analysis(df, bullish_snapshot_score)

        # 步骤五：更新输出
        for p in periods:
            s_bull[p] = bullish_snapshot_score
            s_bear[p] = bearish_snapshot_score
            d_intensity[p] = unified_d_intensity
            
        return s_bull, s_bear, d_intensity

    def _calculate_fault_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict[int, pd.Series], Dict[int, pd.Series], Dict[int, pd.Series]]:
        """【V4.1 · 韧性骨架版】计算筹码断层维度的三维健康度"""
        s_bull, s_bear, d_intensity = {}, {}, {}
        required_cols = ['chip_fault_strength_D', 'chip_fault_vacuum_percent_D']
        
        # [代码修复] 增加数据检查和韧性返回
        if any(col not in df.columns for col in required_cols):
            default_series = pd.Series(0.5, index=df.index, dtype=np.float32)
            for p in periods:
                s_bull[p] = default_series
                s_bear[p] = default_series
                d_intensity[p] = default_series
            return s_bull, s_bear, d_intensity

        # 步骤一：计算原始的、纯粹的筹码静态健康度
        chip_static_bull = (normalize_score(df.get('chip_fault_strength_D'), df.index, norm_window, ascending=False) * normalize_score(df.get('chip_fault_vacuum_percent_D'), df.index, norm_window, ascending=False))**0.5
        chip_static_bear = (normalize_score(df.get('chip_fault_strength_D'), df.index, norm_window, ascending=True) * normalize_score(df.get('chip_fault_vacuum_percent_D'), df.index, norm_window, ascending=True))**0.5

        # 步骤二：获取均线趋势上下文分数
        ma_context_score = self._calculate_ma_trend_context(df, [5, 13, 21, 55])

        # 步骤三：构建融合了趋势上下文的“瞬时关系快照分”
        bullish_snapshot_score = (chip_static_bull * (0.5 + 0.5 * ma_context_score))
        bearish_snapshot_score = (chip_static_bear * (0.5 + 0.5 * (1 - ma_context_score)))

        # 步骤四：对快照分进行关系元分析，得到最终的动态强度分
        unified_d_intensity = self._perform_chip_relational_meta_analysis(df, bullish_snapshot_score)

        # 步骤五：更新输出
        for p in periods:
            s_bull[p] = bullish_snapshot_score
            s_bear[p] = bearish_snapshot_score
            d_intensity[p] = unified_d_intensity
            
        return s_bull, s_bear, d_intensity

    def _perform_chip_relational_meta_analysis(self, df: pd.DataFrame, snapshot_score: pd.Series) -> pd.Series:
        """
        【V1.0 · 新增】筹码专用的关系元分析核心引擎 (赫拉织布机V2)
        - 核心逻辑: 实现“状态 * (1 + 动态杠杆)”的动态价值调制范式。
        """
        # [代码新增] 从配置中获取动态杠杆权重
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        p_meta = get_param_value(p_conf.get('relational_meta_analysis_params'), {})
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.6)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.4)

        # [代码新增] 核心参数
        norm_window = 55
        meta_window = 5
        bipolar_sensitivity = 1.0

        # [代码新增] 第一维度：状态分 (State Score)
        state_score = snapshot_score.clip(0, 1)

        # [代码新增] 第二维度：速度分 (Velocity Score)
        relationship_trend = snapshot_score.diff(meta_window).fillna(0)
        velocity_score = normalize_to_bipolar(
            series=relationship_trend, target_index=df.index,
            window=norm_window, sensitivity=bipolar_sensitivity
        )

        # [代码新增] 第三维度：加速度分 (Acceleration Score)
        relationship_accel = relationship_trend.diff(meta_window).fillna(0)
        acceleration_score = normalize_to_bipolar(
            series=relationship_accel, target_index=df.index,
            window=norm_window, sensitivity=bipolar_sensitivity
        )

        # [代码新增] 终极融合：动态价值调制
        dynamic_leverage = 1 + (velocity_score * w_velocity) + (acceleration_score * w_acceleration)
        final_score = (state_score * dynamic_leverage).clip(0, 1)
        
        return final_score.astype(np.float32)

    def _calculate_ma_trend_context(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V1.0 · 新增】计算均线趋势上下文分数
        - 核心逻辑: 评估短期、中期、长期均线的排列和价格位置，输出一个统一的趋势健康分。
        """
        # [代码新增] 确保所有需要的均线都存在
        ma_cols = [f'EMA_{p}_D' for p in periods]
        if not all(col in df.columns for col in ma_cols):
            return pd.Series(0.5, index=df.index)

        # [代码新增] 均线排列健康度
        alignment_scores = []
        for i in range(len(periods) - 1):
            short_ma = df[f'EMA_{periods[i]}_D']
            long_ma = df[f'EMA_{periods[i+1]}_D']
            alignment_scores.append((short_ma > long_ma).astype(float))
        
        alignment_health = np.mean(alignment_scores, axis=0) if alignment_scores else np.full(len(df.index), 0.5)

        # [代码新增] 价格位置健康度 (价格应在所有均线之上)
        position_scores = [(df['close_D'] > df[col]).astype(float) for col in ma_cols]
        position_health = np.mean(position_scores, axis=0) if position_scores else np.full(len(df.index), 0.5)

        # [代码新增] 融合得到最终的趋势上下文分数
        ma_context_score = pd.Series((alignment_health * position_health)**0.5, index=df.index)
        return ma_context_score.astype(np.float32)

    # ==============================================================================
    # 以下为保留的、具有特殊战术意义的“剧本”诊断模块
    # ==============================================================================

    def diagnose_accumulation_playbooks(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V5.0 · 关系元分析版】主力吸筹模式与风险诊断引擎
        - 核心革命: 废除基于SLOPE/ACCEL的旧逻辑，为每种吸筹模式构建“瞬时关系快照分”，
                      并调用关系元分析引擎，捕捉“吸筹关系”的形成与加速拐点。
        """
        states = {}
        norm_window = 120
        
        # [代码新增] 获取均线趋势上下文，作为判断拉升或打压的背景
        ma_context_score = self._calculate_ma_trend_context(df, [5, 13, 21, 55])

        # --- 拉升吸筹 (Rally Accumulation) ---
        # 步骤一：构建“拉升吸筹”的瞬时关系快照分
        # 核心关系：在上升趋势中(ma_context高)，筹码依然在集中，且获利盘惜售。
        chip_concentration_score = normalize_score(df.get('concentration_90pct_D'), df.index, norm_window, ascending=False)
        winner_conviction_score = normalize_score(df.get('turnover_from_winners_ratio_D'), df.index, norm_window, ascending=False)
        rally_snapshot_score = (ma_context_score * chip_concentration_score * winner_conviction_score)

        # 步骤二：对“拉升吸筹关系”进行元分析
        rally_accumulation_score = self._perform_chip_relational_meta_analysis(df, rally_snapshot_score)
        states['SCORE_CHIP_PLAYBOOK_RALLY_ACCUMULATION'] = rally_accumulation_score.astype(np.float32)

        # --- 打压吸筹 (Suppress Accumulation) ---
        # 步骤一：构建“打压吸筹”的瞬时关系快照分
        # 核心关系：在下跌或盘整趋势中(ma_context低)，筹码逆势集中，且套牢盘正在割肉。
        price_weakness_score = 1 - ma_context_score
        loser_capitulation_score = normalize_score(df.get('turnover_from_losers_ratio_D'), df.index, norm_window, ascending=True)
        suppress_snapshot_score = (price_weakness_score * chip_concentration_score * loser_capitulation_score)

        # 步骤二：对“打压吸筹关系”进行元分析
        suppress_accumulation_score = self._perform_chip_relational_meta_analysis(df, suppress_snapshot_score)
        states['SCORE_CHIP_PLAYBOOK_SUPPRESS_ACCUMULATION'] = suppress_accumulation_score.astype(np.float32)
        
        # --- 真实吸筹 (True Accumulation) ---
        # 步骤三：融合两种升级后的吸筹信号
        true_accumulation_score = np.maximum(rally_accumulation_score, suppress_accumulation_score)
        states['SCORE_CHIP_TRUE_ACCUMULATION'] = true_accumulation_score.astype(np.float32)
        
        return states

    def _diagnose_setup_capitulation_ready(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """【V1.2 · 重构修复版】诊断“恐慌已弥漫”的战备(Setup)状态 (战术模块，予以保留)"""
        states = {}
        required_col = 'total_loser_rate_D'
        if required_col not in df.columns:
            print(f"        -> [筹码情报-恐慌战备诊断] 警告: 缺少关键数据列 '{required_col}'，模块已跳过！")
            return states
        p = get_params_block(self.strategy, 'capitulation_reversal_params', {})
        norm_window = get_param_value(p.get('norm_window'), 120)
        
        # 修正对 normalize_score 的调用，这是导致错误的根源
        deep_capitulation_score = normalize_score(df['total_loser_rate_D'], df.index, norm_window, ascending=True)
        
        long_term_window = 250
        min_periods_long = long_term_window // 4
        
        # 修正对 normalize_score 的调用
        rank_score = normalize_score(df['close_D'], df.index, window=long_term_window, ascending=False)
        
        rolling_low = df['low_D'].rolling(window=long_term_window, min_periods=min_periods_long).min()
        rolling_high = df['high_D'].rolling(window=long_term_window, min_periods=min_periods_long).max()
        price_range = (rolling_high - rolling_low).replace(0, 1e-9)
        position_in_range = (df['close_D'] - rolling_low) / price_range
        range_score = 1.0 - position_in_range.clip(0, 1)
        price_pos_score = np.maximum(rank_score, range_score.fillna(0.5))
        setup_score = (deep_capitulation_score * price_pos_score).astype(np.float32)
        states['SCORE_SETUP_CAPITULATION_READY'] = setup_score
        return states

    def _diagnose_trigger_capitulation_fire(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """【V1.1 · 重构修复版】诊断“卖压出清”的点火(Trigger)行为 (战术模块，予以保留)"""
        states = {}
        required_cols = ['turnover_from_losers_ratio_D', 'ACCEL_5_turnover_from_losers_ratio_D']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"        -> [筹码情报-卖压出清诊断] 警告: 缺少关键数据列 {missing_cols}，模块已跳过！")
            return states
        p = get_params_block(self.strategy, 'capitulation_reversal_params', {})
        norm_window = get_param_value(p.get('norm_window'), 120)
        
        # 修正对 normalize_score 的调用
        relative_turnover_score = normalize_score(df['turnover_from_losers_ratio_D'], df.index, norm_window, ascending=True)
        
        k = get_param_value(p.get('logistic_k', 0.1))
        x0 = get_param_value(p.get('logistic_x0', 50.0))
        absolute_turnover_score = 1 / (1 + np.exp(-k * (df['turnover_from_losers_ratio_D'] - x0)))
        loser_turnover_score = np.maximum(relative_turnover_score, absolute_turnover_score)
        
        # 修正对 normalize_score 的调用
        loser_turnover_accel_score = normalize_score(df['ACCEL_5_turnover_from_losers_ratio_D'], df.index, norm_window, ascending=True)
        
        trigger_score = (loser_turnover_score * loser_turnover_accel_score).astype(np.float32)
        states['SCORE_TRIGGER_CAPITULATION_FIRE'] = trigger_score
        return states

    def diagnose_playbook_capitulation_reversal(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 · 关系元分析重构版】诊断“恐慌盘投降反转”剧本
        - 核心革命: 废除旧的“准备-开火”三部曲，统一为一个方法。
        - 新核心逻辑:
          1. 构建一个融合了“深度套牢”、“长期低位”、“套牢盘换手”和“熊市均线背景”的
             “恐慌投降关系快照分”。
          2. 对这个快照分调用关系元分析引擎，精准捕捉这个“恐慌关系”由极盛转衰的那个
             关键的、高价值的【反转拐点】。
        """
        states = {}
        p = get_params_block(self.strategy, 'capitulation_reversal_params', {})
        norm_window = get_param_value(p.get('norm_window'), 120)
        
        # [代码新增] 检查所需列
        required_cols = ['total_loser_rate_D', 'close_D', 'turnover_from_losers_ratio_D']
        if any(col not in df.columns for col in required_cols):
            print(f"        -> [筹码情报-投降反转剧本] 警告: 缺少关键数据列，剧本合成已跳过！")
            return states

        # [代码新增] 步骤一：构建“恐慌投降关系”的瞬时快照分
        # 特征1: 深度套牢
        deep_capitulation_score = normalize_score(df['total_loser_rate_D'], df.index, norm_window, ascending=True)
        
        # 特征2: 长期低位
        price_at_lows_score = 1.0 - normalize_score(df['close_D'], df.index, window=250, ascending=True)
        
        # 特征3: 套牢盘正在活跃换手 (割肉)
        loser_turnover_score = normalize_score(df['turnover_from_losers_ratio_D'], df.index, norm_window, ascending=True)
        
        # 特征4: 熊市均线背景
        bearish_ma_context = 1 - self._calculate_ma_trend_context(df, [5, 13, 21, 55])
        
        # 融合四大特征，得到瞬时快照分
        snapshot_score = (deep_capitulation_score * price_at_lows_score * loser_turnover_score * bearish_ma_context).astype(np.float32)

        # [代码新增] 步骤二：对“恐慌投降关系”进行元分析，捕捉其由盛转衰的拐点
        final_score = self._perform_chip_relational_meta_analysis(df, snapshot_score)
        
        states['SCORE_CHIP_PLAYBOOK_CAPITULATION_REVERSAL'] = final_score
        return states


