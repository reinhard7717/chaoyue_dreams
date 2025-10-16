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
        【V2.8 · 战神阿瑞斯版】终极资金流信号诊断模块
        - 核心升级: 签署“战神阿瑞斯之审判”协议，将新生成的多源“分歧度”指标作为“冲突”支柱，
                      正式纳入资金流健康度评估体系。分歧度越大，代表市场矛盾越深，健康度越低。
        """
        p_conf = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            return {}
        
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
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
            'main_force_profit': {'base': 'main_force_intraday_profit', 'type': 'daily', 'intent': 'conviction', 'polarity': 1, 'description': '主力日内盈亏，越高代表控盘能力越强'},
            'cost_battle': {'base': 'market_cost_battle', 'type': 'daily', 'intent': 'conflict', 'polarity': 1, 'description': '主力买入成本与散户买入成本的差值，越高代表主力越主动'},
            'cost_divergence': {'base': 'cost_divergence_mf_vs_retail', 'type': 'daily', 'intent': 'conviction', 'polarity': 1, 'description': '主力买入成本与散户卖出成本的差值，越高代表吸筹意愿越强'},
            # 注入“分歧度”作为新的冲突支柱
            # 注意：分歧度指标的绝对值越大，代表冲突越剧烈，因此其 polarity 为 -1，分数越高健康度越低。
            'source_divergence_ts_ths': {'base': 'divergence_ts_ths', 'type': 'daily', 'intent': 'conflict', 'polarity': -1, 'description': 'Tushare与同花顺主力流向分歧度'},
            'source_divergence_ts_dc': {'base': 'divergence_ts_dc', 'type': 'daily', 'intent': 'conflict', 'polarity': -1, 'description': 'Tushare与东方财富主力流向分歧度'},
            'source_divergence_ths_dc': {'base': 'divergence_ths_dc', 'type': 'daily', 'intent': 'conflict', 'polarity': -1, 'description': '同花顺与东方财富主力流向分歧度'},
            
        }
        ma_health_score = self._calculate_ma_health(df, p_conf, norm_window)
        bottom_context_score, top_context_score = calculate_context_scores(df, self.strategy.atomic_states)
        context_scores = {'bottom_context': bottom_context_score, 'top_context': top_context_score}
        pillar_health = self._calculate_all_pillar_health(df, pillar_configs, norm_window, periods, ma_health_score)
        fused_health = self._fuse_health_with_intent_weights(df, pillar_health, pillar_configs, p_conf, periods)
        final_scores = self._synthesize_final_signals(df, fused_health, context_scores, p_synthesis)
        states = self._assign_graded_states(final_scores)
        
        return states

    # ==============================================================================
    # 以下为V2.1版新增的模块化辅助方法
    # ==============================================================================
    def _calculate_all_pillar_health(self, df: pd.DataFrame, pillar_configs: Dict, norm_window: int, periods: list, ma_context_score: pd.Series) -> Dict[str, Dict]:
        """
        【V4.0 · 关系元分析版】计算所有资金流支柱的健康度
        - 调用经过“关系元分析”改造后的 `_calculate_pillar_health` 方法。
        - 优化说明: 接收预计算的 `ma_context_score` 并将其传递给子计算器，避免重复计算。
        """
        pillar_health = {} # 修改(规范): 初始化为空字典
        # 循环调用各支柱计算器，并传入预先计算好的 ma_context_score
        for name, config in pillar_configs.items():
            pillar_health[name] = self._calculate_pillar_health(
                df, config, norm_window, periods, ma_context_score
            )
        return pillar_health

    def _fuse_health_with_intent_weights(self, df: pd.DataFrame, pillar_health: Dict, pillar_configs: Dict, p_conf: Dict, periods: list) -> Dict[str, Dict[str, Dict[int, pd.Series]]]:
        """
        【V2.6 · 削藩令版】执行意图驱动的加权融合
        - 签名变更，被动接收来自上级的统一指令。
        - 优化说明: 1. 使用Numpy进行加权几何平均数的高效向量化计算。
                      2. 优化了默认值处理，使用预创建的Numpy数组代替循环中创建Series，更高效、更健壮。
        """
        fused_results = {
            'resonance': {'s_bull': {}, 's_bear': {}, 'd_intensity': {}},
            'reversal': {'s_bull': {}, 's_bear': {}, 'd_intensity': {}}
        }
        pillar_names = list(pillar_configs.keys())
        
        # 预先创建用于填充缺失值的默认Numpy数组
        default_values = np.full(len(df.index), 0.5, dtype=np.float32)

        for intent_type, weights_key in [('resonance', 'resonance_pillar_weights'), ('reversal', 'reversal_pillar_weights')]:
            weights_config = get_param_value(p_conf.get(weights_key), {})
            # 根据每个支柱的“意图”属性，从配置中获取对应的权重
            valid_weights = [weights_config.get(pillar_configs[name]['intent'], 0) for name in pillar_names]
            weights_array = np.array(valid_weights, dtype=np.float32)
            
            # 对权重进行归一化，确保总权重为1
            total_weights = weights_array.sum()
            if total_weights > 0:
                weights_array /= total_weights
            else: # 如果所有权重都为0，则使用等权重
                weights_array.fill(1.0 / len(weights_array))

            for health_key in ['s_bull', 's_bear', 'd_intensity']:
                for p in periods:
                    # 使用列表推导式和预定义的default_values高效构建分数矩阵
                    # 如果某个支柱在特定周期没有分数，则使用默认值填充
                    pillar_scores_list = [
                        pillar_health[name][health_key].get(p, default_values)
                        for name in pillar_names
                    ]
                    pillar_scores_matrix = np.stack(pillar_scores_list, axis=0)
                    
                    # 计算加权几何平均数。该算法能有效惩罚任何一个维度的短板，更能体现“共振”的综合性。
                    # 公式: G = (s1^w1 * s2^w2 * ... * sn^wn)
                    # 增加一个极小值1e-9避免log(0)错误
                    fused_values = np.exp(np.sum(weights_array[:, np.newaxis] * np.log(pillar_scores_matrix + 1e-9), axis=0))
                    
                    fused_results[intent_type][health_key][p] = pd.Series(fused_values, index=df.index, dtype=np.float32)
        
        # 将共振健康度存入原子状态，供其他模块消费
        self.strategy.atomic_states['__FF_overall_health'] = fused_results['resonance']
        return fused_results
    
    def _synthesize_final_signals(self, df: pd.DataFrame, fused_health: Dict, context_scores: Dict, p_synthesis: Dict) -> Dict[str, pd.Series]:
        """
        【V5.3 · 战术激活版】
        - 核心修复: 将被遗忘的“战术反转”信号纳入最终信号合成流程。
        """
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
            # 新增对战术反转信号的提取
            'tactical_reversal': resonance_signals['SCORE_FF_TACTICAL_REVERSAL'],
        }
        return final_scores

    def _assign_graded_states(self, final_scores: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V2.7 · 战术激活版】将最终信号赋值给状态字典。
        - 核心修复: 增加对“战术反转”信号的处理。
        """
        states = {}
        prefix_map = {
            'bullish_resonance': 'SCORE_FF_BULLISH_RESONANCE',
            'bottom_reversal': 'SCORE_FF_BOTTOM_REVERSAL',
            'bearish_resonance': 'SCORE_FF_BEARISH_RESONANCE',
            'top_reversal': 'SCORE_FF_TOP_REVERSAL',
            # 新增战术反转信号的映射
            'tactical_reversal': 'SCORE_FF_TACTICAL_REVERSAL',
        }
        for key, score in final_scores.items():
            signal_name = prefix_map[key]
            states[signal_name] = score.astype(np.float32)
        return states

    def _calculate_pillar_health(self, df: pd.DataFrame, config: Dict, norm_window: int, periods: list, ma_context_score: pd.Series) -> Dict:
        """
        【V6.1 · 净值累积版】计算单个资金流支柱的三维健康度
        - 核心修正: 签署“德尔斐神谕协议”，剥离 ma_context_score 对 s_bull/s_bear 的污染。
        - 核心升级(V6.1): 当支柱类型为 'sum' 时，调用全新的 `_calculate_net_accumulation_score` 方法，
                          实现对资金净值在多时间级别上的累积分析，以识别主力的长期布局。
        """
        s_bull, s_bear, d_intensity = {}, {}, {}
        base_col_name = config['base']
        polarity = config['polarity']
        col_type = config['type']
        # [代码修改] 增加对 'sum' 类型的分支处理
        if col_type == 'sum':
            # 对于需要累积分析的指标（如主力净流入），调用新的多时间级别累积分析引擎
            # 看涨快照分 = 多周期净流入累积的加权得分
            bullish_snapshot_score = self._calculate_net_accumulation_score(df, base_col_name, norm_window, periods, ascending=(polarity == 1))
            # 看跌快照分 = 多周期净流出累积的加权得分
            bearish_snapshot_score = self._calculate_net_accumulation_score(df, base_col_name, norm_window, periods, ascending=(polarity == -1))
        else: # col_type == 'daily'
            # 对于日度指标，保持原有逻辑不变
            static_col = f"{base_col_name}_D"
            market_cap_col = 'circ_mv_D'
            if static_col not in df.columns or market_cap_col not in df.columns:
                print(f"      -> [宙斯天平] 警告: 缺少核心列 '{static_col}' 或 '{market_cap_col}'，无法计算市值归一化资金流。")
                default_series = pd.Series(0.5, index=df.index, dtype=np.float32)
                for p in periods:
                    s_bull[p], s_bear[p], d_intensity[p] = default_series.copy(), default_series.copy(), default_series.copy()
                return {'s_bull': s_bull, 's_bear': s_bear, 'd_intensity': d_intensity}
            market_cap_in_yuan = df[market_cap_col] * 10000
            market_cap_in_yuan = market_cap_in_yuan.replace(0, np.nan)
            normalized_flow_series = (df[static_col] / market_cap_in_yuan).fillna(0)
            static_series = normalized_flow_series
            indicator_static_bull = normalize_score(static_series, df.index, norm_window, ascending=(polarity == 1))
            indicator_static_bear = normalize_score(static_series, df.index, norm_window, ascending=(polarity == -1))
            bullish_snapshot_score = indicator_static_bull.astype(np.float32)
            bearish_snapshot_score = indicator_static_bear.astype(np.float32)
        
        # 动态强度分现在基于更可靠的“累积快照分”进行计算
        unified_d_intensity = self._perform_fund_flow_relational_meta_analysis(df, bullish_snapshot_score)
        for p in periods:
            s_bull[p] = bullish_snapshot_score
            s_bear[p] = bearish_snapshot_score
            d_intensity[p] = unified_d_intensity
        return {'s_bull': s_bull, 's_bear': s_bear, 'd_intensity': d_intensity}

    def _calculate_net_accumulation_score(self, df: pd.DataFrame, base_col_name: str, norm_window: int, periods: list, ascending: bool) -> pd.Series:
        """
        【V1.0 · 新增】计算多时间级别加权资金净累积得分
        - 核心逻辑:
          1. 对日度资金流数据，在多个时间窗口(如5,13,21,55日)上进行滚动求和。
          2. 对每个窗口的累积净值进行归一化，得到该窗口的累积强度分。
          3. 根据配置的权重，对所有窗口的强度分进行加权平均，得到最终的“净累积得分”。
        - 战略意义: 解决了单日资金流向的短视问题，能够识别主力跨越多日的真实资金布局。
        """
        # 步骤1: 获取日度资金流数据
        daily_series = df.get(f"{base_col_name}_D")
        if daily_series is None or daily_series.empty:
            print(f"      -> [资金净累积] 警告: 缺少日度资金流数据列 '{base_col_name}_D'，无法计算。")
            return pd.Series(0.5, index=df.index, dtype=np.float32)
        # 步骤2: 从配置中获取各时间窗口的权重
        p_conf = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        acc_weights = get_param_value(p_conf.get('accumulation_analysis_weights'), {})
        # 步骤3: 计算加权累积得分
        accumulation_periods = [p for p in periods if p > 1]
        weighted_scores = []
        total_weight = 0
        for p in accumulation_periods:
            weight = acc_weights.get(str(p), 0)
            if weight > 0:
                # 计算滚动窗口内的资金净累积值
                rolling_sum = daily_series.rolling(window=p, min_periods=max(1, p // 2)).sum()
                # 对累积值进行归一化，得到该窗口的强度分
                normalized_sum = normalize_score(rolling_sum, df.index, norm_window, ascending=ascending)
                weighted_scores.append(normalized_sum * weight)
                total_weight += weight
        if not weighted_scores or total_weight == 0:
            return pd.Series(0.5, index=df.index, dtype=np.float32)
        # 对所有时间窗口的得分进行加权平均
        final_accumulation_score = sum(weighted_scores) / total_weight
        return final_accumulation_score.astype(np.float32)

    # ==============================================================================
    # 以下为V2.1版新增的模块化辅助方法
    # ==============================================================================
    def _perform_fund_flow_relational_meta_analysis(self, df: pd.DataFrame, snapshot_score: pd.Series) -> pd.Series:
        """
        【V2.0 · 阿瑞斯之怒协议版】资金流专用的关系元分析核心引擎
        - 核心革命: 响应“重变化、轻状态”的哲学，从“状态 * (1 + 动态)”的乘法模型，升级为
                      “(状态*权重) + (速度*权重) + (加速度*权重)”的加法模型。
        - 核心目标: 即使静态分很低，只要动态（尤其是加速度）足够强，也能产生高分，真正捕捉“拐点”。
        """
        # 引入新的权重体系和加法融合模型
        # 从配置中获取新的加法模型权重
        p_conf = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        p_meta = get_param_value(p_conf.get('relational_meta_analysis_params'), {})
        # 新的权重体系，直接作用于最终分数，而非杠杆
        w_state = get_param_value(p_meta.get('state_weight'), 0.3)
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.3)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.4) # 赋予加速度最高权重
        # 核心参数
        norm_window = 55
        meta_window = 5
        bipolar_sensitivity = 1.0
        # 第一维度：状态分 (State Score) - 范围 [0, 1]
        state_score = snapshot_score.clip(0, 1)
        # 第二维度：速度分 (Velocity Score) - 范围 [-1, 1]
        relationship_trend = snapshot_score.diff(meta_window).fillna(0)
        velocity_score = normalize_to_bipolar(
            series=relationship_trend, target_index=df.index,
            window=norm_window, sensitivity=bipolar_sensitivity
        )
        # 第三维度：加速度分 (Acceleration Score) - 范围 [-1, 1]
        relationship_accel = relationship_trend.diff(meta_window).fillna(0)
        acceleration_score = normalize_to_bipolar(
            series=relationship_accel, target_index=df.index,
            window=norm_window, sensitivity=bipolar_sensitivity
        )
        # 终极融合：从乘法调制升级为加法赋权
        # 旧的乘法模型: dynamic_leverage = 1 + (velocity_score * w_velocity) + (acceleration_score * w_acceleration)
        # 旧的乘法模型: final_score = (state_score * dynamic_leverage).clip(0, 1)
        # 新的加法模型:
        final_score = (
            state_score * w_state +
            velocity_score * w_velocity +
            acceleration_score * w_acceleration
        ).clip(0, 1) # clip确保分数在[0, 1]范围内
        return final_score.astype(np.float32)
        

    def _calculate_ma_health(self, df: pd.DataFrame, params: dict, norm_window: int) -> pd.Series:
        """
        【V1.0 · 新增】“赫尔墨斯的商神杖”四维均线健康度评估引擎
        - 核心职责: 严格按照 ma_health_fusion_weights 配置，计算并融合均线健康度的四大维度。
        """
        p_ma_health = get_param_value(params.get('ma_health_fusion_weights'), {})
        weights = {
            'alignment': get_param_value(p_ma_health.get('alignment'), 0.15),
            'slope': get_param_value(p_ma_health.get('slope'), 0.15),
            'accel': get_param_value(p_ma_health.get('accel'), 0.2),
            'relational': get_param_value(p_ma_health.get('relational'), 0.5)
        }
        
        ma_periods = [5, 13, 21, 55]
        ma_cols = [f'EMA_{p}_D' for p in ma_periods]
        if not all(col in df.columns for col in ma_cols):
            return pd.Series(0.5, index=df.index, dtype=np.float32)

        ma_values = np.stack([df[col].values for col in ma_cols], axis=0)
        
        alignment_bools = ma_values[:-1] > ma_values[1:]
        alignment_health = np.mean(alignment_bools, axis=0) if alignment_bools.size > 0 else np.full(len(df.index), 0.5)

        slope_cols = [f'SLOPE_5_{col}' for col in ma_cols]
        if all(col in df.columns for col in slope_cols):
            slope_values = np.stack([df[col].values for col in slope_cols], axis=0)
            slope_health = np.mean(normalize_score(pd.Series(slope_values.flatten()), df.index, norm_window).values.reshape(slope_values.shape), axis=0)
        else:
            slope_health = np.full(len(df.index), 0.5)

        accel_cols = [f'ACCEL_5_{col}' for col in ma_cols]
        if all(col in df.columns for col in accel_cols):
            accel_values = np.stack([df[col].values for col in accel_cols], axis=0)
            accel_health = np.mean(normalize_score(pd.Series(accel_values.flatten()), df.index, norm_window).values.reshape(accel_values.shape), axis=0)
        else:
            accel_health = np.full(len(df.index), 0.5)

        ma_std = np.std(ma_values / df['close_D'].values[:, np.newaxis].T, axis=0)
        relational_health = 1.0 - normalize_score(pd.Series(ma_std, index=df.index), df.index, norm_window, ascending=True)

        scores = np.stack([alignment_health, slope_health, accel_health, relational_health], axis=0)
        weights_array = np.array(list(weights.values()))
        weights_array /= weights_array.sum()

        final_score_values = np.prod(scores ** weights_array[:, np.newaxis], axis=0)
        
        return pd.Series(final_score_values, index=df.index, dtype=np.float32)

















