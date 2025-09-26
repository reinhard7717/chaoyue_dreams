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
        【V2.1 · 模块化重构版】终极资金流信号诊断模块
        - 核心重构 (本次修改):
          - [代码重构] 将原V2.0版的臃肿逻辑拆分为多个职责单一的私有辅助方法。
          - [结构优化] 本方法升格为高层指挥官，负责按顺序调用各计算模块，代码更清晰。
        - 收益: 极大提升了代码的可读性和可维护性，同时保持了业务逻辑的完整性。
        """
        # print("        -> [终极资金流信号诊断模块 V2.1 · 模块化重构版] 启动...") # 更新版本号和说明
        
        # 步骤1: 初始化所有参数、权重和支柱配置
        params = self._initialize_ff_params()
        if not params['enabled']:
            return {}

        # 步骤2: 调用公共函数计算上下文分数
        bottom_context_score, top_context_score = calculate_context_scores(df, self.strategy.atomic_states)
        context_scores = {'bottom_context': bottom_context_score, 'top_context': top_context_score}

        # 步骤3: 计算每一个资金流支柱的四维健康度
        pillar_health = self._calculate_all_pillar_health(df, params)

        # 步骤4: 执行“意图驱动加权融合”，生成全局的四维健康度
        overall_health = self._fuse_health_with_intent_weights(pillar_health, params)

        # 步骤5: 结合全局健康度与上下文分数，合成最终的共振与反转信号
        final_scores = self._synthesize_final_signals(overall_health, context_scores, params)

        # 步骤6: 将最终信号转换为 S+/S/A/B 四个等级
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

    def _fuse_health_with_intent_weights(self, pillar_health: Dict, params: Dict) -> Dict[str, Dict[int, pd.Series]]:
        """
        【V2.2 · 性能优化版】执行意图驱动的加权融合，生成全局四维健康度。
        - 本次优化:
          - [效率] 重构了整个融合逻辑。通过预先将意图权重映射为NumPy数组，并在周期循环中
                    将所有支柱的健康分堆叠成矩阵，最终利用一次矩阵乘法（`@`）和一次求和
                    完成加权融合。这取代了原先在多重循环中对Pandas Series的重复操作，
                    极大提升了计算效率并减少了内存占用。
        """
        overall_health = {}
        pillar_names = list(params['pillar_configs'].keys())
        num_pillars = len(pillar_names)
        
        # 预处理权重，将它们映射为与支柱顺序一致的NumPy数组
        resonance_weights_array = np.array([params['resonance_pillar_weights'].get(params['pillar_configs'][name]['intent'], 0) for name in pillar_names])
        reversal_weights_array = np.array([params['reversal_pillar_weights'].get(params['pillar_configs'][name]['intent'], 0) for name in pillar_names])

        # 遍历四种健康度类型
        for health_type in ['bullish_static', 'bullish_dynamic', 'bearish_static', 'bearish_dynamic']:
            overall_health[health_type] = {}
            
            # 根据健康度类型选择对应的权重数组
            weights_array = resonance_weights_array if 'static' in health_type else reversal_weights_array
            total_weights = np.sum(weights_array)
            
            # 确定要从 pillar_health 中提取的键名
            health_dict_key = f"{'s' if 'static' in health_type else 'd'}_{'bull' if 'bullish' in health_type else 'bear'}"
            
            # 遍历所有周期
            for p in params['periods']:
                # 将所有支柱在当前周期的健康分Series的Numpy数组堆叠成一个 (num_pillars, N) 的矩阵
                # N是数据行数
                pillar_scores_matrix = np.stack([
                    pillar_health[name][health_dict_key].get(p, pd.Series(0.5)).values 
                    for name in pillar_names
                ], axis=0)
                
                if total_weights > 0:
                    # 使用NumPy的向量化乘法和加法，一次性完成加权求和
                    # (num_pillars, N) 矩阵与 (num_pillars,) 权重的乘法(利用广播) -> (num_pillars, N) -> 沿axis=0求和 -> (N,)
                    fused_values = np.sum(pillar_scores_matrix * weights_array[:, np.newaxis], axis=0) / total_weights
                    
                    # 获取正确的索引用于创建Series
                    sample_index = pillar_health[pillar_names[0]]['s_bull'][p].index
                    overall_health[health_type][p] = pd.Series(fused_values, index=sample_index, dtype=np.float32)
                else:
                    sample_index = pillar_health[pillar_names[0]]['s_bull'][p].index
                    overall_health[health_type][p] = pd.Series(0.5, index=sample_index, dtype=np.float32)
                    
        return overall_health

    def _synthesize_final_signals(self, overall_health: Dict, context_scores: Dict, params: Dict) -> Dict[str, pd.Series]:
        """
        【V2.3 · 哲学修复版】合成最终的共振与反转信号
        - 核心修复: 彻底重构了底部和顶部反转信号的合成哲学。
        """
        final_scores = {}
        periods = params['periods']
        res_tw = params['resonance_tf_weights']
        rev_tw = params['reversal_tf_weights']
        p_conf = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        bottom_context_bonus_factor = get_param_value(p_conf.get('bottom_context_bonus_factor'), 0.5)
        top_context_bonus_factor = get_param_value(p_conf.get('top_context_bonus_factor'), 0.8) # 新增顶部因子

        # 看涨共振 (逻辑不变)
        bullish_resonance_health = {p: overall_health['bullish_static'][p] * overall_health['bullish_dynamic'][p] for p in periods}
        bull_res_short = (bullish_resonance_health.get(1, 0.5) * bullish_resonance_health.get(5, 0.5))**0.5
        bull_res_med = (bullish_resonance_health.get(13, 0.5) * bullish_resonance_health.get(21, 0.5))**0.5
        bull_res_long = bullish_resonance_health.get(55, 0.5)
        final_scores['bullish_resonance'] = (bull_res_short * res_tw['short'] + bull_res_med * res_tw['medium'] + bull_res_long * res_tw['long'])
        
        # 底部反转 (全新逻辑: 静态看跌 * 动态看涨)
        bullish_reversal_health = {p: overall_health['bearish_static'][p] * overall_health['bullish_dynamic'][p] for p in periods}
        bull_rev_short = (bullish_reversal_health.get(1, 0.5) * bullish_reversal_health.get(5, 0.5))**0.5
        bull_rev_med = (bullish_reversal_health.get(13, 0.5) * bullish_reversal_health.get(21, 0.5))**0.5
        bull_rev_long = bullish_reversal_health.get(55, 0.5)
        bullish_trigger = (bull_rev_short * rev_tw['short'] + bull_rev_med * rev_tw['medium'] + bull_rev_long * rev_tw['long'])
        final_scores['bottom_reversal'] = (bullish_trigger * (1 + context_scores['bottom_context'] * bottom_context_bonus_factor)).clip(0, 1)

        # 看跌共振 (逻辑不变)
        bearish_resonance_health = {p: overall_health['bearish_static'][p] * overall_health['bearish_dynamic'][p] for p in periods}
        bear_res_short = (bearish_resonance_health.get(1, 0.5) * bearish_resonance_health.get(5, 0.5))**0.5
        bear_res_med = (bearish_resonance_health.get(13, 0.5) * bearish_resonance_health.get(21, 0.5))**0.5
        bear_res_long = bearish_resonance_health.get(55, 0.5)
        final_scores['bearish_resonance'] = (bear_res_short * res_tw['short'] + bear_res_med * res_tw['medium'] + bear_res_long * res_tw['long'])

        # 顶部反转 (全新逻辑: 静态看涨 * 动态看跌)
        bearish_reversal_health = {p: overall_health['bullish_static'][p] * overall_health['bearish_dynamic'][p] for p in periods}
        bear_rev_short = (bearish_reversal_health.get(1, 0.5) * bearish_reversal_health.get(5, 0.5))**0.5
        bear_rev_med = (bearish_reversal_health.get(13, 0.5) * bearish_reversal_health.get(21, 0.5))**0.5
        bear_rev_long = bearish_reversal_health.get(55, 0.5)
        bearish_trigger = (bear_rev_short * rev_tw['short'] + bear_rev_med * rev_tw['medium'] + bear_rev_long * rev_tw['long'])
        # 顶部反转也应用奖励(惩罚)因子模型
        final_scores['top_reversal'] = (bearish_trigger * (1 + context_scores['top_context'] * top_context_bonus_factor)).clip(0, 1)
        
        return final_scores

    def _assign_graded_states(self, final_scores: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """将最终信号转换为 S+/S/A/B 四个等级。"""
        states = {}
        prefix_map = {
            'bullish_resonance': 'SCORE_FF_BULLISH_RESONANCE',
            'bottom_reversal': 'SCORE_FF_BOTTOM_REVERSAL',
            'bearish_resonance': 'SCORE_FF_BEARISH_RESONANCE',
            'top_reversal': 'SCORE_FF_TOP_REVERSAL',
        }
        for key, score in final_scores.items():
            prefix = prefix_map[key]
            states[f'{prefix}_S_PLUS'] = score.astype(np.float32)
            states[f'{prefix}_S'] = (score * 0.8).astype(np.float32)
            states[f'{prefix}_A'] = (score * 0.6).astype(np.float32)
            states[f'{prefix}_B'] = (score * 0.4).astype(np.float32)
        return states

    def _calculate_pillar_health(self, df: pd.DataFrame, name: str, config: Dict, norm_window: int, dynamic_weights: Dict, periods: list) -> Dict:
        """【V2.1 · 健壮性修复版】计算单个资金流支柱的四维健康度"""
        s_bull, d_bull, s_bear, d_bear = {}, {}, {}, {}
        base_col_name = config['base']
        polarity = config['polarity']
        col_type = config['type']

        for p in periods:
            if col_type == 'sum' and p > 1:
                static_col = f"{base_col_name}_sum_{p}d_D"
            else:
                static_col = f"{base_col_name}_D"
            
            # 修复accel列名构造错误的问题
            slope_base = static_col.replace('_D', '')
            slope_col = f"SLOPE_{p}_{slope_base}_D"
            accel_col = f"ACCEL_{p}_{slope_base}_D"

            s_bull[p] = normalize_score(df.get(static_col), df.index, norm_window, ascending=(polarity == 1))
            s_bear[p] = normalize_score(df.get(static_col), df.index, norm_window, ascending=(polarity == -1))
            
            d_bull_slope = normalize_score(df.get(slope_col), df.index, norm_window, ascending=(polarity == 1))
            d_bull_accel = normalize_score(df.get(accel_col), df.index, norm_window, ascending=(polarity == 1))
            d_bull[p] = d_bull_slope * dynamic_weights['slope'] + d_bull_accel * dynamic_weights['accel']
            
            d_bear_slope = normalize_score(df.get(slope_col), df.index, norm_window, ascending=(polarity == -1))
            d_bear_accel = normalize_score(df.get(accel_col), df.index, norm_window, ascending=(polarity == -1))
            d_bear[p] = d_bear_slope * dynamic_weights['slope'] + d_bear_accel * dynamic_weights['accel']

        return {'s_bull': s_bull, 'd_bull': d_bull, 's_bear': s_bear, 'd_bear': d_bear}