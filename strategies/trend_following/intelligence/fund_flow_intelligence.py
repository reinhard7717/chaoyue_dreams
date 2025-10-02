# 文件: strategies/trend_following/intelligence/fund_flow_intelligence.py
# 资金流情报模块
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from strategies.trend_following.utils import transmute_health_to_ultimate_signals, get_params_block, get_param_value, normalize_score, calculate_context_scores, normalize_to_bipolar

class FundFlowIntelligence:
    def __init__(self, strategy_instance):
        """
        初始化资金流情报模块。
        :param strategy_instance: 策略主实例的引用。
        """
        self.strategy = strategy_instance

    def diagnose_fund_flow_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V20.0 · 统一范式版】资金流情报分析总指挥
        - 核心重构: 废除所有旧的、各自为战的诊断引擎，统一调用唯一的终极信号引擎。
        - 收益: 实现了前所未有的架构清晰度、逻辑一致性和哲学完备性。
        """
        # print("      -> [资金流情报分析总指挥 V20.0 · 统一范式版] 启动...")
        p = get_params_block(self.strategy, 'fund_flow_params')
        if not get_param_value(p.get('enabled'), True):
            return {}
        
        ultimate_ff_states = self.diagnose_ultimate_fund_flow_signals(df)
        # print(f"      -> [资金流情报分析总指挥 V20.0] 分析完毕，共生成 {len(ultimate_ff_states)} 个终极资金流信号。")
        return ultimate_ff_states

    def diagnose_ultimate_fund_flow_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.5 · 削藩令版】终极资金流信号诊断模块
        - 核心革命: 1. 废除独立的参数初始化“内阁” `_initialize_ff_params`。
                      2. 实现中央直辖，所有参数均从中央的 `p_conf` 和 `p_synthesis` 获取。
                      3. 重构下级函数的调用，确保它们被动接收来自中央的统一参数。
        """
        # 废除内阁，实现中央直辖
        p_conf = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            return {}
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        # 直接从中央获取所有参数
        periods = get_param_value(p_synthesis.get('periods'), [1, 5, 13, 21, 55])
        norm_window = get_param_value(p_synthesis.get('norm_window'), 55)
        pillar_configs = {
            'main_force': {'base': 'main_force_net_flow_consensus', 'type': 'sum', 'intent': 'consensus', 'polarity': 1},
            'xl_order': {'base': 'net_xl_amount_consensus', 'type': 'sum', 'intent': 'consensus', 'polarity': 1},
            'lg_order': {'base': 'net_lg_amount_consensus', 'type': 'sum', 'intent': 'consensus', 'polarity': 1},
            'intensity': {'base': 'main_force_flow_intensity_ratio', 'type': 'daily', 'intent': 'conviction', 'polarity': 1},
            'buy_pressure': {'base': 'active_buy_pressure', 'type': 'daily', 'intent': 'conviction', 'polarity': 1},
            'conviction_ratio': {'base': 'main_force_conviction_ratio', 'type': 'daily', 'intent': 'conviction', 'polarity': 1},
            'internal_divergence': {'base': 'main_force_vs_xl_divergence', 'type': 'daily', 'intent': 'conflict', 'polarity': -1},
            'mf_vs_retail': {'base': 'flow_divergence_mf_vs_retail', 'type': 'daily', 'intent': 'conflict', 'polarity': 1},
            'retail_panic': {'base': 'retail_panic_index', 'type': 'daily', 'intent': 'sentiment', 'polarity': 1},
            'sh_flow': {'base': 'net_sh_amount_consensus', 'type': 'sum', 'intent': 'sentiment', 'polarity': -1},
            'md_flow': {'base': 'net_md_amount_consensus', 'type': 'sum', 'intent': 'sentiment', 'polarity': -1},
        }
        bottom_context_score, top_context_score = calculate_context_scores(df, self.strategy.atomic_states)
        context_scores = {'bottom_context': bottom_context_score, 'top_context': top_context_score}
        # 向下级函数传递中央参数
        pillar_health = self._calculate_all_pillar_health(df, pillar_configs, norm_window, periods)
        fused_health = self._fuse_health_with_intent_weights(pillar_health, pillar_configs, p_conf, periods)
        final_scores = self._synthesize_final_signals(df, fused_health, context_scores, p_synthesis)
        states = self._assign_graded_states(final_scores)
        return states

    # ==============================================================================
    # 以下为V2.1版新增的模块化辅助方法
    # ==============================================================================
    def _calculate_all_pillar_health(self, df: pd.DataFrame, pillar_configs: Dict, norm_window: int, periods: list) -> Dict[str, Dict]:
        """
        【V4.0 · 关系元分析版】计算所有资金流支柱的健康度
        - 核心修改: 调用经过“关系元分析”改造后的 `_calculate_pillar_health` 方法。
        """
        pillar_health = {key: {} for key in pillar_configs}
        # dynamic_weights 不再需要，因为动态分析已在内部完成
        for name, config in pillar_configs.items():
            pillar_health[name] = self._calculate_pillar_health(
                df, name, config, norm_window, periods
            )
        return pillar_health

    def _fuse_health_with_intent_weights(self, pillar_health: Dict, pillar_configs: Dict, p_conf: Dict, periods: list) -> Dict[str, Dict[str, Dict[int, pd.Series]]]:
        """
        【V2.6 · 削藩令版】执行意图驱动的加权融合
        - 核心修改: 签名变更，被动接收来自上级的统一指令。
        """
        fused_results = {
            'resonance': {'s_bull': {}, 's_bear': {}, 'd_intensity': {}},
            'reversal': {'s_bull': {}, 's_bear': {}, 'd_intensity': {}}
        }
        pillar_names = list(pillar_configs.keys())
        for intent_type, weights_key in [('resonance', 'resonance_pillar_weights'), ('reversal', 'reversal_pillar_weights')]:
            weights_config = get_param_value(p_conf.get(weights_key), {})
            valid_weights = [weights_config.get(pillar_configs[name]['intent'], 0) for name in pillar_names]
            weights_array = np.array(valid_weights)
            total_weights = weights_array.sum()
            if total_weights > 0:
                weights_array /= total_weights
            else:
                weights_array = np.full_like(weights_array, 1.0 / len(weights_array))
            for health_key in ['s_bull', 's_bear', 'd_intensity']:
                for p in periods:
                    pillar_scores_matrix = np.stack([
                        pillar_health[name][health_key].get(p, pd.Series(0.5, index=pillar_health[pillar_names[0]]['s_bull'][p].index)).values 
                        for name in pillar_names
                    ], axis=0)
                    fused_values = np.prod(pillar_scores_matrix ** weights_array[:, np.newaxis], axis=0)
                    sample_index = pillar_health[pillar_names[0]]['s_bull'][p].index
                    fused_results[intent_type][health_key][p] = pd.Series(fused_values, index=sample_index, dtype=np.float32)
        self.strategy.atomic_states['__FF_overall_health'] = fused_results['resonance']
        return fused_results
    
    def _synthesize_final_signals(self, df: pd.DataFrame, fused_health: Dict, context_scores: Dict, p_synthesis: Dict) -> Dict[str, pd.Series]:
        """
        【V5.2 · 圣杯契约版】
        - 核心革命: 参数 `params` 已更名为 `p_synthesis`，明确表示其接收的是中央“圣杯”配置。
        """
        # 传入唯一的“圣杯”配置
        resonance_signals = transmute_health_to_ultimate_signals(
            df=df,
            atomic_states=self.strategy.atomic_states,
            overall_health=fused_health['resonance'],
            params=p_synthesis,
            domain_prefix="FF"
        )
        reversal_signals = transmute_health_to_ultimate_signals(
            df=df,
            atomic_states=self.strategy.atomic_states,
            overall_health=fused_health['reversal'],
            params=p_synthesis,
            domain_prefix="FF"
        )
        final_scores = {
            'bullish_resonance': resonance_signals['SCORE_FF_BULLISH_RESONANCE'],
            'bottom_reversal': reversal_signals['SCORE_FF_BOTTOM_REVERSAL'],
            'bearish_resonance': resonance_signals['SCORE_FF_BEARISH_RESONANCE'],
            'top_reversal': reversal_signals['SCORE_FF_TOP_REVERSAL'],
        }
        return final_scores

    def _assign_graded_states(self, final_scores: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V2.6 · 信号净化版】将最终信号赋值给状态字典。
        - 核心重构: 废除S/A/B分级，只输出唯一的、归一化的终极信号。
                      信号名不再包含 _S_PLUS 后缀，实现命名的终极简化。
        """
        states = {}
        # 信号命名净化：废除S/A/B分级，只使用唯一的、归一化的终极信号名
        prefix_map = {
            'bullish_resonance': 'SCORE_FF_BULLISH_RESONANCE',
            'bottom_reversal': 'SCORE_FF_BOTTOM_REVERSAL',
            'bearish_resonance': 'SCORE_FF_BEARISH_RESONANCE',
            'top_reversal': 'SCORE_FF_TOP_REVERSAL',
        }
        for key, score in final_scores.items():
            signal_name = prefix_map[key]
            # 只生成唯一的、归一化的信号，其名称不包含任何等级后缀
            states[signal_name] = score.astype(np.float32)
        return states

    def _calculate_pillar_health(self, df: pd.DataFrame, name: str, config: Dict, norm_window: int, periods: list) -> Dict:
        """【V4.0 · 关系元分析版】计算单个资金流支柱的三维健康度"""
        s_bull, s_bear, d_intensity = {}, {}, {}
        base_col_name = config['base']
        polarity = config['polarity']
        col_type = config['type']

        # 统一计算一次，因为所有周期的健康度将共享同一套关系分析结果
        # 步骤一：计算原始的、纯粹的资金流指标静态健康度
        # 注意：这里只计算一次，而不是在周期循环内
        if col_type == 'sum':
            # 对于累加型指标，我们使用最长周期(55)的累加值作为代表性静态值
            static_col = f"{base_col_name}_sum_55d_D"
        else:
            # 对于每日型指标，直接使用当日值
            static_col = f"{base_col_name}_D"
        
        default_series = pd.Series(0.5, index=df.index)
        static_series = df.get(static_col, default_series)
        
        indicator_static_bull = normalize_score(static_series, df.index, norm_window, ascending=(polarity == 1))
        indicator_static_bear = normalize_score(static_series, df.index, norm_window, ascending=(polarity == -1))

        # 步骤二：获取均线趋势上下文分数
        ma_context_score = self._calculate_ma_trend_context(df, [5, 13, 21, 55])

        # 步骤三：构建融合了趋势上下文的“瞬时关系快照分”
        bullish_snapshot_score = (indicator_static_bull * ma_context_score)
        bearish_snapshot_score = (indicator_static_bear * (1 - ma_context_score))

        # 步骤四：对快照分进行关系元分析，得到最终的动态强度分
        unified_d_intensity = self._perform_fund_flow_relational_meta_analysis(df, bullish_snapshot_score)

        # 步骤五：将统一计算的结果赋给所有周期
        for p in periods:
            s_bull[p] = bullish_snapshot_score
            s_bear[p] = bearish_snapshot_score
            d_intensity[p] = unified_d_intensity

        return {'s_bull': s_bull, 's_bear': s_bear, 'd_intensity': d_intensity}

    # ==============================================================================
    # 以下为V2.1版新增的模块化辅助方法
    # ==============================================================================
    def _perform_fund_flow_relational_meta_analysis(self, df: pd.DataFrame, snapshot_score: pd.Series) -> pd.Series:
        """
        【V1.0 · 新增】资金流专用的关系元分析核心引擎 (赫拉织布机V2)
        - 核心逻辑: 实现“状态 * (1 + 动态杠杆)”的动态价值调制范式。
        """
        # 从配置中获取动态杠杆权重
        p_conf = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        p_meta = get_param_value(p_conf.get('relational_meta_analysis_params'), {})
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.6)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.4)

        # 核心参数
        norm_window = 55
        meta_window = 5
        bipolar_sensitivity = 1.0

        # 第一维度：状态分 (State Score)
        state_score = snapshot_score.clip(0, 1)

        # 第二维度：速度分 (Velocity Score)
        relationship_trend = snapshot_score.diff(meta_window).fillna(0)
        velocity_score = normalize_to_bipolar(
            series=relationship_trend, target_index=df.index,
            window=norm_window, sensitivity=bipolar_sensitivity
        )

        # 第三维度：加速度分 (Acceleration Score)
        relationship_accel = relationship_trend.diff(meta_window).fillna(0)
        acceleration_score = normalize_to_bipolar(
            series=relationship_accel, target_index=df.index,
            window=norm_window, sensitivity=bipolar_sensitivity
        )

        # 终极融合：动态价值调制
        dynamic_leverage = 1 + (velocity_score * w_velocity) + (acceleration_score * w_acceleration)
        final_score = (state_score * dynamic_leverage).clip(0, 1)
        
        return final_score.astype(np.float32)

    def _calculate_ma_trend_context(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V1.0 · 新增】计算均线趋势上下文分数
        - 核心逻辑: 评估短期、中期、长期均线的排列和价格位置，输出一个统一的趋势健康分。
        """
        # 确保所有需要的均线都存在
        ma_cols = [f'EMA_{p}_D' for p in periods]
        if not all(col in df.columns for col in ma_cols):
            return pd.Series(0.5, index=df.index)

        # 均线排列健康度
        alignment_scores = []
        for i in range(len(periods) - 1):
            short_ma = df[f'EMA_{periods[i]}_D']
            long_ma = df[f'EMA_{periods[i+1]}_D']
            alignment_scores.append((short_ma > long_ma).astype(float))
        
        alignment_health = np.mean(alignment_scores, axis=0) if alignment_scores else np.full(len(df.index), 0.5)

        # 价格位置健康度 (价格应在所有均线之上)
        position_scores = [(df['close_D'] > df[col]).astype(float) for col in ma_cols]
        position_health = np.mean(position_scores, axis=0) if position_scores else np.full(len(df.index), 0.5)

        # 融合得到最终的趋势上下文分数
        ma_context_score = pd.Series((alignment_health * position_health)**0.5, index=df.index)
        return ma_context_score.astype(np.float32)
















