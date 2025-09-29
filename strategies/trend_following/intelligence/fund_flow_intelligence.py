# 文件: strategies/trend_following/intelligence/fund_flow_intelligence.py
# 资金流情报模块
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score, calculate_context_scores

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
        【V2.2 · 融合逻辑重构版】终极资金流信号诊断模块
        - 核心重构 (本次修改):
          - [数据流] 更新了调用流程，以适配 V2.3/V2.4 版的融合与合成逻辑。
        """
        params = self._initialize_ff_params()
        if not params['enabled']:
            return {}

        bottom_context_score, top_context_score = calculate_context_scores(df, self.strategy.atomic_states)
        context_scores = {'bottom_context': bottom_context_score, 'top_context': top_context_score}

        pillar_health = self._calculate_all_pillar_health(df, params)

        # 此处调用重构后的融合方法
        fused_health = self._fuse_health_with_intent_weights(pillar_health, params)

        # 此处调用重构后的合成方法
        final_scores = self._synthesize_final_signals(fused_health, context_scores, params)

        states = self._assign_graded_states(final_scores)
        
        return states
    # ==============================================================================
    # 以下为V2.1版新增的模块化辅助方法
    # ==============================================================================

    def _initialize_ff_params(self) -> Dict[str, Any]:
        """初始化所有参数、权重和支柱配置。"""
        params = {}
        p_conf = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        params['enabled'] = get_param_value(p_conf.get('enabled'), True)
        params['dynamic_weights'] = {'slope': 0.6, 'accel': 0.4}
        params['periods'] = get_param_value(p_conf.get('periods', [1, 5, 13, 21, 55]))
        params['norm_window'] = get_param_value(p_conf.get('norm_window'), 120)
        
        params['resonance_tf_weights'] = {'short': 0.2, 'medium': 0.5, 'long': 0.3}
        params['reversal_tf_weights'] = {'short': 0.6, 'medium': 0.3, 'long': 0.1}

        params['resonance_pillar_weights'] = {'consensus': 0.4, 'conviction': 0.3, 'conflict': 0.1, 'sentiment': 0.2}
        params['reversal_pillar_weights'] = {'consensus': 0.1, 'conviction': 0.3, 'conflict': 0.4, 'sentiment': 0.2}

        params['pillar_configs'] = {
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
        return params

    def _calculate_all_pillar_health(self, df: pd.DataFrame, params: Dict) -> Dict[str, Dict]:
        """计算所有资金流支柱的四维健康度。"""
        pillar_health = {key: {} for key in params['pillar_configs']}
        for name, config in params['pillar_configs'].items():
            pillar_health[name] = self._calculate_pillar_health(
                df, name, config, params['norm_window'], params['dynamic_weights'], params['periods']
            )
        return pillar_health

    def _fuse_health_with_intent_weights(self, pillar_health: Dict, params: Dict) -> Dict[str, Dict[str, Dict[int, pd.Series]]]:
        """
        【V2.5 · 动态分统一版】执行意图驱动的加权融合。
        """
        # 更新数据结构以适配三维健康度
        fused_results = {
            'resonance': {'s_bull': {}, 's_bear': {}, 'd_intensity': {}},
            'reversal': {'s_bull': {}, 's_bear': {}, 'd_intensity': {}}
        }
        pillar_names = list(params['pillar_configs'].keys())
        
        for intent_type, weights_key in [('resonance', 'resonance_pillar_weights'), ('reversal', 'reversal_pillar_weights')]:
            weights_config = params[weights_key]
            
            valid_weights = [weights_config.get(params['pillar_configs'][name]['intent'], 0) for name in pillar_names]
            weights_array = np.array(valid_weights)
            total_weights = weights_array.sum()
            if total_weights > 0:
                weights_array /= total_weights
            else:
                weights_array = np.full_like(weights_array, 1.0 / len(weights_array))

            # 循环遍历新的三维健康度类型
            for health_key in ['s_bull', 's_bear', 'd_intensity']:
                for p in params['periods']:
                    pillar_scores_matrix = np.stack([
                        pillar_health[name][health_key].get(p, pd.Series(0.5, index=pillar_health[pillar_names[0]]['s_bull'][p].index)).values 
                        for name in pillar_names
                    ], axis=0)
                    
                    fused_values = np.prod(pillar_scores_matrix ** weights_array[:, np.newaxis], axis=0)
                    sample_index = pillar_health[pillar_names[0]]['s_bull'][p].index
                    fused_results[intent_type][health_key][p] = pd.Series(fused_values, index=sample_index, dtype=np.float32)

        self.strategy.atomic_states['__FF_overall_health'] = fused_results['resonance']
        return fused_results
    
    def _synthesize_final_signals(self, fused_health: Dict, context_scores: Dict, params: Dict) -> Dict[str, pd.Series]:
        """
        【V2.7 · 顶部守卫版】
        - 核心升级 (治本之道): 引入“顶部上下文守卫”，将“顶部反转”信号与“顶部上下文分数”相乘，
                              从源头上杜绝在底部区域误报顶部风险。
        """
        final_scores = {}
        periods = params['periods']
        res_tw = params['resonance_tf_weights']
        rev_tw = params['reversal_tf_weights']
        p_conf = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        bottom_context_bonus_factor = get_param_value(p_conf.get('bottom_context_bonus_factor'), 0.5)
        
        resonance_health = fused_health['resonance']
        reversal_health = fused_health['reversal']
        
        bullish_resonance_health = {p: resonance_health['s_bull'][p] * resonance_health['d_intensity'][p] for p in periods}
        bull_res_short = (bullish_resonance_health.get(1, 0.5) * bullish_resonance_health.get(5, 0.5))**0.5
        bull_res_med = (bullish_resonance_health.get(13, 0.5) * bullish_resonance_health.get(21, 0.5))**0.5
        bull_res_long = bullish_resonance_health.get(55, 0.5)
        final_scores['bullish_resonance'] = (
            (bull_res_short ** res_tw['short']) *
            (bull_res_med ** res_tw['medium']) *
            (bull_res_long ** res_tw['long'])
        )
        
        bullish_reversal_health = {p: reversal_health['s_bear'][p] * reversal_health['d_intensity'][p] for p in periods}
        bull_rev_short = (bullish_reversal_health.get(1, 0.5) * bullish_reversal_health.get(5, 0.5))**0.5
        bull_rev_med = (bullish_reversal_health.get(13, 0.5) * bullish_reversal_health.get(21, 0.5))**0.5
        bull_rev_long = bullish_reversal_health.get(55, 0.5)
        bullish_trigger = (
            (bull_rev_short ** rev_tw['short']) *
            (bull_rev_med ** rev_tw['medium']) *
            (bull_rev_long ** rev_tw['long'])
        )
        final_scores['bottom_reversal'] = (bullish_trigger * (1 + context_scores['bottom_context'] * bottom_context_bonus_factor)).clip(0, 1)

        bearish_resonance_health = {p: resonance_health['s_bear'][p] * resonance_health['d_intensity'][p] for p in periods}
        bear_res_short = (bearish_resonance_health.get(1, 0.5) * bearish_resonance_health.get(5, 0.5))**0.5
        bear_res_med = (bearish_resonance_health.get(13, 0.5) * bearish_resonance_health.get(21, 0.5))**0.5
        bear_res_long = bearish_resonance_health.get(55, 0.5)
        final_scores['bearish_resonance'] = (
            (bear_res_short ** res_tw['short']) *
            (bear_res_med ** res_tw['medium']) *
            (bear_res_long ** res_tw['long'])
        )
        
        # 修正顶部反转的计算哲学，并应用“顶部守卫”
        # 1. 顶部反转由看跌静态分 s_bear 驱动
        bearish_reversal_health = {p: reversal_health['s_bear'][p] * reversal_health['d_intensity'][p] for p in periods}
        bear_rev_short = (bearish_reversal_health.get(1, 0.5) * bearish_reversal_health.get(5, 0.5))**0.5
        bear_rev_med = (bearish_reversal_health.get(13, 0.5) * bearish_reversal_health.get(21, 0.5))**0.5
        bear_rev_long = bearish_reversal_health.get(55, 0.5)
        bearish_trigger = (
            (bear_rev_short ** rev_tw['short']) *
            (bear_rev_med ** rev_tw['medium']) *
            (bear_rev_long ** rev_tw['long'])
        )
        # 2. 将触发分与“顶部上下文分数”相乘，实现门控
        final_scores['top_reversal'] = (bearish_trigger * context_scores['top_context']).clip(0, 1)
        
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

    def _calculate_pillar_health(self, df: pd.DataFrame, name: str, config: Dict, norm_window: int, dynamic_weights: Dict, periods: list) -> Dict:
        """【V2.7 · 动态分统一版】计算单个资金流支柱的三维健康度"""
        # 更新方法签名和初始化，统一返回 d_intensity
        s_bull, s_bear, d_intensity = {}, {}, {}
        base_col_name = config['base']
        polarity = config['polarity']
        col_type = config['type']

        for p in periods:
            if col_type == 'sum' and p > 1:
                static_col = f"{base_col_name}_sum_{p}d_D"
            else:
                static_col = f"{base_col_name}_D"

            if col_type == 'sum' and p > 1:
                slope_base_col = f"{base_col_name}_sum_{p}d_D"
            else:
                slope_base_col = f"{base_col_name}_D"
            
            slope_col = f"SLOPE_{p}_{slope_base_col}"
            accel_col = f"ACCEL_{p}_{slope_base_col}"

            default_series = pd.Series(0.5, index=df.index)
            
            static_series = df.get(static_col, default_series)
            slope_series = df.get(slope_col, default_series)
            accel_series = df.get(accel_col, default_series)

            s_bull[p] = normalize_score(static_series, df.index, norm_window, ascending=(polarity == 1))
            s_bear[p] = normalize_score(static_series, df.index, norm_window, ascending=(polarity == -1))
            
            # 计算统一的、中性的动态强度分 d_intensity
            mom_strength = normalize_score(slope_series.abs(), df.index, norm_window, ascending=True)
            accel_strength = normalize_score(accel_series.abs(), df.index, norm_window, ascending=True)
            d_intensity[p] = (mom_strength * accel_strength)**0.5

        # 返回符合新协议的三元组
        return {'s_bull': s_bull, 's_bear': s_bear, 'd_intensity': d_intensity}
















