# 文件: strategies/trend_following/intelligence/fund_flow_intelligence.py
# 资金流情报模块
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from strategies.trend_following.utils import get_params_block, get_param_value

class FundFlowIntelligence:
    def __init__(self, strategy_instance):
        """
        初始化资金流情报模块。
        :param strategy_instance: 策略主实例的引用。
        """
        self.strategy = strategy_instance

    def _normalize_score(self, series: pd.Series, window: int, target_index: pd.Index, ascending: bool = True) -> pd.Series:
        """
        【V13.0】计算一个系列在滚动窗口内的归一化得分 (0-1)。
        """
        if series is None or series.isnull().all():
            return pd.Series(0.5, index=target_index, dtype=np.float32)

        min_periods = max(1, int(window * 0.2))
        rank = series.rolling(window=window, min_periods=min_periods).rank(pct=True, ascending=ascending).fillna(0.5)
        return rank.astype(np.float32)

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
        print("        -> [终极资金流信号诊断模块 V2.1 · 模块化重构版] 启动...") # [代码修改] 更新版本号和说明
        
        # 步骤1: 初始化所有参数、权重和支柱配置
        params = self._initialize_ff_params()
        if not params['enabled']:
            return {}

        # 步骤2: 计算宏观价格位置的上下文门控分数
        context_scores = self._calculate_context_scores(df)

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
    # [代码修改] 以下为V2.1版新增的模块化辅助方法
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

    def _calculate_context_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算宏观价格位置的上下文门控分数。"""
        rolling_low_55d = df['low_D'].rolling(window=55, min_periods=21).min()
        rolling_high_55d = df['high_D'].rolling(window=55, min_periods=21).max()
        price_range_55d = (rolling_high_55d - rolling_low_55d).replace(0, 1e-9)
        price_position_in_range = ((df['close_D'] - rolling_low_55d) / price_range_55d).clip(0, 1).fillna(0.5)
        return {
            'bottom_context': 1 - price_position_in_range,
            'top_context': price_position_in_range
        }

    def _calculate_all_pillar_health(self, df: pd.DataFrame, params: Dict) -> Dict[str, Dict]:
        """计算所有资金流支柱的四维健康度。"""
        pillar_health = {key: {} for key in params['pillar_configs']}
        for name, config in params['pillar_configs'].items():
            pillar_health[name] = self._calculate_pillar_health(
                df, name, config, params['norm_window'], params['dynamic_weights'], params['periods']
            )
        return pillar_health

    def _fuse_health_with_intent_weights(self, pillar_health: Dict, params: Dict) -> Dict[str, Dict[int, pd.Series]]:
        """执行意图驱动的加权融合，生成全局四维健康度。"""
        overall_health = {}
        for health_type in ['bullish_static', 'bullish_dynamic', 'bearish_static', 'bearish_dynamic']:
            overall_health[health_type] = {}
            weights_map = params['resonance_pillar_weights'] if 'static' in health_type else params['reversal_pillar_weights']
            
            for p in params['periods']:
                weighted_scores, total_weights = [], 0
                for name, config in params['pillar_configs'].items():
                    intent = config['intent']
                    weight = weights_map.get(intent, 0)
                    health_dict_key = f"{'s' if 'static' in health_type else 'd'}_{'bull' if 'bullish' in health_type else 'bear'}"
                    score = pillar_health[name][health_dict_key].get(p)
                    if score is not None:
                        weighted_scores.append(score.values * weight)
                        total_weights += weight
                
                if total_weights > 0:
                    final_score_arr = np.sum(np.stack(weighted_scores, axis=0), axis=0) / total_weights
                    overall_health[health_type][p] = pd.Series(final_score_arr, index=pillar_health[next(iter(pillar_health))]['s_bull'][p].index, dtype=np.float32)
                else:
                    overall_health[health_type][p] = pd.Series(0.5, index=pillar_health[next(iter(pillar_health))]['s_bull'][p].index, dtype=np.float32)
        return overall_health

    def _synthesize_final_signals(self, overall_health: Dict, context_scores: Dict, params: Dict) -> Dict[str, pd.Series]:
        """合成最终的共振与反转信号。"""
        final_scores = {}
        periods = params['periods']
        res_tw = params['resonance_tf_weights']
        rev_tw = params['reversal_tf_weights']

        # 看涨信号合成
        bullish_resonance_health = {p: overall_health['bullish_static'][p] * overall_health['bullish_dynamic'][p] for p in periods}
        bull_res_short = (bullish_resonance_health.get(1, 0.5) * bullish_resonance_health.get(5, 0.5))**0.5
        bull_res_med = (bullish_resonance_health.get(13, 0.5) * bullish_resonance_health.get(21, 0.5))**0.5
        bull_res_long = bullish_resonance_health.get(55, 0.5)
        final_scores['bullish_resonance'] = (bull_res_short * res_tw['short'] + bull_res_med * res_tw['medium'] + bull_res_long * res_tw['long'])
        
        bullish_dynamic = overall_health['bullish_dynamic']
        bull_rev_short = (bullish_dynamic.get(1, 0.5) * bullish_dynamic.get(5, 0.5))**0.5
        bull_rev_med = (bullish_dynamic.get(13, 0.5) * bullish_dynamic.get(21, 0.5))**0.5
        bull_rev_long = bullish_dynamic.get(55, 0.5)
        bullish_trigger = (bull_rev_short * rev_tw['short'] + bull_rev_med * rev_tw['medium'] + bull_rev_long * rev_tw['long'])
        final_scores['bottom_reversal'] = context_scores['bottom_context'] * bullish_trigger

        # 看跌信号合成
        bearish_resonance_health = {p: overall_health['bearish_static'][p] * overall_health['bearish_dynamic'][p] for p in periods}
        bear_res_short = (bearish_resonance_health.get(1, 0.5) * bearish_resonance_health.get(5, 0.5))**0.5
        bear_res_med = (bearish_resonance_health.get(13, 0.5) * bearish_resonance_health.get(21, 0.5))**0.5
        bear_res_long = bearish_resonance_health.get(55, 0.5)
        final_scores['bearish_resonance'] = (bear_res_short * res_tw['short'] + bear_res_med * res_tw['medium'] + bear_res_long * res_tw['long'])

        bearish_dynamic = overall_health['bearish_dynamic']
        bear_rev_short = (bearish_dynamic.get(1, 0.5) * bearish_dynamic.get(5, 0.5))**0.5
        bear_rev_med = (bearish_dynamic.get(13, 0.5) * bearish_dynamic.get(21, 0.5))**0.5
        bear_rev_long = bearish_dynamic.get(55, 0.5)
        bearish_trigger = (bear_rev_short * rev_tw['short'] + bear_rev_med * rev_tw['medium'] + bear_rev_long * rev_tw['long'])
        final_scores['top_reversal'] = context_scores['top_context'] * bearish_trigger
        
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
        """【V2.0】计算单个资金流支柱的四维健康度"""
        s_bull, d_bull, s_bear, d_bear = {}, {}, {}, {}
        base_col_name = config['base']
        polarity = config['polarity']
        col_type = config['type']

        for p in periods:
            if col_type == 'sum' and p > 1:
                static_col = f"{base_col_name}_sum_{p}d_D"
            else:
                static_col = f"{base_col_name}_D"
            
            slope_base = static_col.replace('_D', '')
            slope_col = f"SLOPE_{p}_{slope_base}_D"
            accel_col = f"ACCEL_{p}_{base_col_name}_D"

            s_bull[p] = self._normalize_score(df.get(static_col), norm_window, df.index, ascending=(polarity == 1))
            s_bear[p] = self._normalize_score(df.get(static_col), norm_window, df.index, ascending=(polarity == -1))
            
            d_bull_slope = self._normalize_score(df.get(slope_col), norm_window, df.index, ascending=(polarity == 1))
            d_bull_accel = self._normalize_score(df.get(accel_col), norm_window, df.index, ascending=(polarity == 1))
            d_bull[p] = d_bull_slope * dynamic_weights['slope'] + d_bull_accel * dynamic_weights['accel']
            
            d_bear_slope = self._normalize_score(df.get(slope_col), norm_window, df.index, ascending=(polarity == -1))
            d_bear_accel = self._normalize_score(df.get(accel_col), norm_window, df.index, ascending=(polarity == -1))
            d_bear[p] = d_bear_slope * dynamic_weights['slope'] + d_bear_accel * dynamic_weights['accel']

        return {'s_bull': s_bull, 'd_bull': d_bull, 's_bear': s_bear, 'd_bear': d_bear}
