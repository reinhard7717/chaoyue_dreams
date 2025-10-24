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
        【V21.1 · 三公理版】终极资金流信号诊断模块
        - 核心升级: 新增调用“公理三：内部资金结构”诊断引擎，并将其纳入终极信号合成。
        """
        p_conf = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            return {}
        concentration_scores = self._diagnose_concentration_dynamics_ff(df, p_conf)
        power_transfer_scores = self._diagnose_power_transfer_ff(df, p_conf)
        # 新增调用公理三诊断引擎
        internal_structure_scores = self._diagnose_internal_flow_structure_ff(df, p_conf)
        self.strategy.atomic_states['SCORE_FF_AXIOM_CONCENTRATION'] = concentration_scores
        self.strategy.atomic_states['SCORE_FF_AXIOM_POWER_TRANSFER'] = power_transfer_scores
        self.strategy.atomic_states['SCORE_FF_AXIOM_INTERNAL_STRUCTURE'] = internal_structure_scores
        final_scores = self._synthesize_ultimate_signals_from_axioms(
            df, concentration_scores, power_transfer_scores, internal_structure_scores, p_conf
        )

        states = self._assign_graded_states(final_scores)
        return states

    def _diagnose_concentration_dynamics_ff(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【V2.1 · 健壮性修复版】资金流公理一：诊断资金“聚散”的动态
        - 核心修复: 全面将 df.get(..., 0) 的默认值升级为 pd.Series(0.0, index=df.index)，
                      防止因上游指标缺失导致的类型错误，实现“装甲加固”。
        """
        periods = get_param_value(params.get('periods'), [1, 5, 13, 21, 55])
        scores = {}
        for i, p in enumerate(periods):
            context_p = periods[i + 1] if i + 1 < len(periods) else p
            # 全面加固 df.get() 调用，防止类型错误
            bullish_static = df.get('main_force_flow_impact_ratio_D', pd.Series(0.0, index=df.index)) + df.get('main_force_conviction_ratio_D', pd.Series(0.0, index=df.index))
            bullish_slope = df.get(f'SLOPE_{p}_main_force_flow_impact_ratio_D', pd.Series(0.0, index=df.index)) + df.get(f'SLOPE_{p}_main_force_conviction_ratio_D', pd.Series(0.0, index=df.index))
            bullish_accel = df.get(f'ACCEL_{p}_main_force_flow_impact_ratio_D', pd.Series(0.0, index=df.index)) + df.get(f'ACCEL_{p}_main_force_conviction_ratio_D', pd.Series(0.0, index=df.index))
            bearish_static = df.get('retail_net_flow_consensus_D', pd.Series(0.0, index=df.index)).abs() + df.get('main_force_vs_xl_divergence_D', pd.Series(0.0, index=df.index))
            bearish_slope = df.get(f'SLOPE_{p}_retail_net_flow_consensus_D', pd.Series(0.0, index=df.index)).abs() + df.get(f'SLOPE_{p}_main_force_vs_xl_divergence_D', pd.Series(0.0, index=df.index))
            bearish_accel = df.get(f'ACCEL_{p}_retail_net_flow_consensus_D', pd.Series(0.0, index=df.index)).abs() + df.get(f'ACCEL_{p}_main_force_vs_xl_divergence_D', pd.Series(0.0, index=df.index))
    
            tactical_bullish_static = normalize_score(bullish_static, df.index, p, ascending=True)
            tactical_bullish_slope = normalize_score(bullish_slope, df.index, p, ascending=True)
            tactical_bullish_accel = normalize_score(bullish_accel, df.index, p, ascending=True)
            tactical_bullish_quality = (tactical_bullish_static * tactical_bullish_slope * tactical_bullish_accel)**(1/3)
            context_bullish_static = normalize_score(bullish_static, df.index, context_p, ascending=True)
            context_bullish_slope = normalize_score(bullish_slope, df.index, context_p, ascending=True)
            context_bullish_accel = normalize_score(bullish_accel, df.index, context_p, ascending=True)
            context_bullish_quality = (context_bullish_static * context_bullish_slope * context_bullish_accel)**(1/3)
            final_bullish_quality = (tactical_bullish_quality * context_bullish_quality)**0.5
            tactical_bearish_static = normalize_score(bearish_static, df.index, p, ascending=True)
            tactical_bearish_slope = normalize_score(bearish_slope, df.index, p, ascending=True)
            tactical_bearish_accel = normalize_score(bearish_accel, df.index, p, ascending=True)
            tactical_bearish_quality = (tactical_bearish_static * tactical_bearish_slope * tactical_bearish_accel)**(1/3)
            context_bearish_static = normalize_score(bearish_static, df.index, context_p, ascending=True)
            context_bearish_slope = normalize_score(bearish_slope, df.index, context_p, ascending=True)
            context_bearish_accel = normalize_score(bearish_accel, df.index, context_p, ascending=True)
            context_bearish_quality = (context_bearish_static * context_bearish_slope * context_bearish_accel)**(1/3)
            final_bearish_quality = (tactical_bearish_quality * context_bearish_quality)**0.5
            concentration_snapshot = (final_bullish_quality - final_bearish_quality).astype(np.float32)
            scores[p] = concentration_snapshot.clip(-1, 1)
        return scores

    def _diagnose_power_transfer_ff(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【V3.2 · 前线换装与健壮性修复版】资金流公理二：诊断资金“权力转移”的方向
        - 核心修复: 1. 废弃已失效的 `short_term_profit_taking_ratio_D`，换装为 `profit_taking_urgency_D`。
                      2. 全面将 df.get(..., 0) 的默认值升级为 pd.Series(0.0, index=df.index)，实现“装甲加固”。
        """
        periods = get_param_value(params.get('periods'), [1, 5, 13, 21, 55])
        scores = {}
        for i, p in enumerate(periods):
            context_p = periods[i + 1] if i + 1 < len(periods) else p
            # 全面加固 df.get() 调用并换装新指标
            # --- 看涨证据（权力向主力转移） ---
            bullish_static = df.get('retail_capitulation_score_D', pd.Series(0.0, index=df.index)) + df.get('main_force_support_strength_D', pd.Series(0.0, index=df.index))
            cost_advantage_slope = df.get(f'SLOPE_{p}_cost_divergence_mf_vs_retail_D', pd.Series(0.0, index=df.index))
            cost_advantage_accel = df.get(f'ACCEL_{p}_cost_divergence_mf_vs_retail_D', pd.Series(0.0, index=df.index))
            loser_selling_slope = df.get(f'SLOPE_{p}_loser_rate_short_term_D', pd.Series(0.0, index=df.index))
            loser_selling_accel = df.get(f'ACCEL_{p}_loser_rate_short_term_D', pd.Series(0.0, index=df.index))
            bullish_slope = cost_advantage_slope + loser_selling_slope
            bullish_accel = cost_advantage_accel + loser_selling_accel
            # --- 看跌证据（权力向散户转移） ---
            bearish_static = df.get('main_force_distribution_pressure_D', pd.Series(0.0, index=df.index)) + df.get('retail_chasing_accumulation_D', pd.Series(0.0, index=df.index))
            cost_disadvantage_slope = -cost_advantage_slope
            cost_disadvantage_accel = -cost_advantage_accel
            # 换装新武器: profit_taking_urgency_D
            profit_taking_slope = df.get(f'SLOPE_{p}_profit_taking_urgency_D', pd.Series(0.0, index=df.index))
            profit_taking_accel = df.get(f'ACCEL_{p}_profit_taking_urgency_D', pd.Series(0.0, index=df.index))
            bearish_slope = cost_disadvantage_slope + profit_taking_slope
            bearish_accel = cost_disadvantage_accel + profit_taking_accel
    
            # --- 融合计算 ---
            tactical_bullish_static = normalize_score(bullish_static, df.index, p, ascending=True)
            tactical_bullish_slope = normalize_score(bullish_slope, df.index, p, ascending=True)
            tactical_bullish_accel = normalize_score(bullish_accel, df.index, p, ascending=True)
            tactical_bullish_quality = (tactical_bullish_static * tactical_bullish_slope * tactical_bullish_accel)**(1/3)
            context_bullish_static = normalize_score(bullish_static, df.index, context_p, ascending=True)
            context_bullish_slope = normalize_score(bullish_slope, df.index, context_p, ascending=True)
            context_bullish_accel = normalize_score(bullish_accel, df.index, context_p, ascending=True)
            context_bullish_quality = (context_bullish_static * context_bullish_slope * context_bullish_accel)**(1/3)
            final_bullish_quality = (tactical_bullish_quality * context_bullish_quality)**0.5
            tactical_bearish_static = normalize_score(bearish_static, df.index, p, ascending=True)
            tactical_bearish_slope = normalize_score(bearish_slope, df.index, p, ascending=True)
            tactical_bearish_accel = normalize_score(bearish_accel, df.index, p, ascending=True)
            tactical_bearish_quality = (tactical_bearish_static * tactical_bearish_slope * tactical_bearish_accel)**(1/3)
            context_bearish_static = normalize_score(bearish_static, df.index, context_p, ascending=True)
            context_bearish_slope = normalize_score(bearish_slope, df.index, context_p, ascending=True)
            context_bearish_accel = normalize_score(bearish_accel, df.index, context_p, ascending=True)
            context_bearish_quality = (context_bearish_static * context_bearish_slope * context_bearish_accel)**(1/3)
            final_bearish_quality = (tactical_bearish_quality * context_bearish_quality)**0.5
            power_transfer_snapshot = (final_bullish_quality - final_bearish_quality).astype(np.float32)
            scores[p] = power_transfer_snapshot.clip(-1, 1)
        return scores

    def _diagnose_internal_flow_structure_ff(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【V1.0 · 新增】资金流公理三：诊断资金“内部结构”的健康度
        - 核心逻辑: 剖析主力资金内部（超大单/大单 vs 中单/小单）的协同与背离，揭示更深层次的市场意图。
        """
        #
        periods = get_param_value(params.get('periods'), [1, 5, 13, 21, 55])
        scores = {}
        for i, p in enumerate(periods):
            context_p = periods[i + 1] if i + 1 < len(periods) else p
            # --- 定义核心参与者资金流 ---
            xl_flow = df.get('net_xl_amount_consensus_D', pd.Series(0.0, index=df.index))
            lg_flow = df.get('net_lg_amount_consensus_D', pd.Series(0.0, index=df.index))
            md_flow = df.get('net_md_amount_consensus_D', pd.Series(0.0, index=df.index))
            sh_flow = df.get('net_sh_amount_consensus_D', pd.Series(0.0, index=df.index))
            # --- 看涨证据：大单吸筹，小单派发 ---
            smart_money_inflow = (xl_flow + lg_flow).clip(lower=0)
            retail_money_outflow = (md_flow + sh_flow).clip(upper=0).abs()
            # 战术层
            tactical_smart_in = normalize_score(smart_money_inflow, df.index, p, ascending=True)
            tactical_retail_out = normalize_score(retail_money_outflow, df.index, p, ascending=True)
            # 上下文层
            context_smart_in = normalize_score(smart_money_inflow, df.index, context_p, ascending=True)
            context_retail_out = normalize_score(retail_money_outflow, df.index, context_p, ascending=True)
            # 融合
            fused_smart_in = (tactical_smart_in * context_smart_in)**0.5
            fused_retail_out = (tactical_retail_out * context_retail_out)**0.5
            bullish_divergence = (fused_smart_in * fused_retail_out)**0.5
            # --- 看跌证据：大单派发，小单接盘 ---
            smart_money_outflow = (xl_flow + lg_flow).clip(upper=0).abs()
            retail_money_inflow = (md_flow + sh_flow).clip(lower=0)
            # 战术层
            tactical_smart_out = normalize_score(smart_money_outflow, df.index, p, ascending=True)
            tactical_retail_in = normalize_score(retail_money_inflow, df.index, p, ascending=True)
            # 上下文层
            context_smart_out = normalize_score(smart_money_outflow, df.index, context_p, ascending=True)
            context_retail_in = normalize_score(retail_money_inflow, df.index, context_p, ascending=True)
            # 融合
            fused_smart_out = (tactical_smart_out * context_smart_out)**0.5
            fused_retail_in = (tactical_retail_in * context_retail_in)**0.5
            bearish_divergence = (fused_smart_out * fused_retail_in)**0.5
            # --- 生成双极快照分 ---
            internal_structure_snapshot = (bullish_divergence - bearish_divergence).astype(np.float32)
            scores[p] = internal_structure_snapshot.clip(-1, 1)
        return scores
        

    def _synthesize_ultimate_signals_from_axioms(self, df: pd.DataFrame, concentration: Dict[int, pd.Series], power_transfer: Dict[int, pd.Series], internal_structure: Dict[int, pd.Series], params: dict) -> Dict[str, pd.Series]:
        """
        【V3.2 · 波塞冬裁决版】基于物理公理的终极信号合成器
        - 核心升级: 调用“波塞冬的三叉戟”引擎计算趋势健康度，并用其对最终的资金流信号进行调节，
                      使其具备趋势感知能力，能有效过滤牛市陷阱和熊市陷阱。
        """
        states = {}
        axiom_weights = get_param_value(params.get('axiom_weights'), {'concentration': 0.4, 'power_transfer': 0.4, 'internal_structure': 0.2})
        tf_weights = get_param_value(params.get('tf_weights'), {1: 0.1, 5: 0.4, 13: 0.3, 21: 0.15, 55: 0.05})
        total_tf_weight = sum(tf_weights.values())
        # 调用“波塞冬的三叉戟”引擎，获取趋势上下文
        trend_health_score = self._calculate_trend_context_ff(df, params)
        
        bullish_resonance = pd.Series(0.0, index=df.index)
        bearish_resonance = pd.Series(0.0, index=df.index)
        if total_tf_weight > 0:
            for p, weight in tf_weights.items():
                conc_score = concentration.get(p, 0.0)
                trans_score = power_transfer.get(p, 0.0)
                struct_score = internal_structure.get(p, 0.0)
                period_bullish = (
                    conc_score.clip(0, 1) * axiom_weights['concentration'] +
                    trans_score.clip(0, 1) * axiom_weights['power_transfer'] +
                    struct_score.clip(0, 1) * axiom_weights['internal_structure']
                )
                period_bearish = (
                    conc_score.clip(-1, 0).abs() * axiom_weights['concentration'] +
                    trans_score.clip(-1, 0).abs() * axiom_weights['power_transfer'] +
                    struct_score.clip(-1, 0).abs() * axiom_weights['internal_structure']
                )
                bullish_resonance += period_bullish * (weight / total_tf_weight)
                bearish_resonance += period_bearish * (weight / total_tf_weight)
        # 使用趋势健康度对资金流信号进行调节
        bullish_resonance = bullish_resonance * trend_health_score
        bearish_resonance = bearish_resonance * (1 - trend_health_score)

        bottom_reversal = self._perform_fund_flow_relational_meta_analysis(df, bullish_resonance)
        top_reversal = self._perform_fund_flow_relational_meta_analysis(df, bearish_resonance)
        tactical_reversal = (bullish_resonance * 0.5).astype(np.float32)
        final_scores = {
            'bullish_resonance': bullish_resonance,
            'bottom_reversal': bottom_reversal,
            'bearish_resonance': bearish_resonance,
            'top_reversal': top_reversal,
            'tactical_reversal': tactical_reversal,
        }
        return final_scores

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
        【V10.0 · 四维时空版】计算单个资金流支柱的四维健康度
        - 核心升级: 引入“四维时空”分析范式，融合“状态”、“速度(斜率)”、“加速度”和“势(上下文)”。
        - 坐标修正: 全面使用数据层提供的真实列名（如 `SLOPE_5_...` 和 `ACCEL_5_...`）。
        """
        s_bull, s_bear, d_intensity = {}, {}, {}
        base_col_name = config['base']
        polarity = config['polarity']
        normalize_by_mv = config.get('normalize_by_mv', False)
        static_col_name = f"{base_col_name}_D"
        if static_col_name not in df.columns:
            default_series = pd.Series(0.5, index=df.index, dtype=np.float32)
            for p in periods:
                s_bull[p], s_bear[p], d_intensity[p] = default_series.copy(), default_series.copy(), default_series.copy()
            return {'s_bull': s_bull, 's_bear': s_bear, 'd_intensity': d_intensity}
        static_series = df[static_col_name]
        if normalize_by_mv:
            market_cap_col = 'circ_mv_D'
            if market_cap_col in df.columns:
                market_cap_in_yuan = df[market_cap_col] * 10000
                market_cap_in_yuan = market_cap_in_yuan.replace(0, np.nan)
                static_series = (static_series * 10000 / market_cap_in_yuan).fillna(0)
        # 引入状态、速度、加速度、上下文的四维融合
        for p in periods:
            context_p = periods[periods.index(p) + 1] if periods.index(p) + 1 < len(periods) else p
            slope_col_name = f"SLOPE_{p}_{base_col_name}_D"
            accel_col_name = f"ACCEL_{p}_{base_col_name}_D"
            slope_series = df.get(slope_col_name, pd.Series(0.0, index=df.index))
            accel_series = df.get(accel_col_name, pd.Series(0.0, index=df.index))
            # --- 看涨证据 ---
            # 战术层
            tactical_static_bull = normalize_score(static_series, df.index, p, ascending=(polarity == 1))
            tactical_slope_bull = normalize_score(slope_series, df.index, p, ascending=True)
            tactical_accel_bull = normalize_score(accel_series, df.index, p, ascending=True)
            tactical_bullish_quality = (tactical_static_bull * tactical_slope_bull * tactical_accel_bull)**(1/3)
            # 上下文层
            context_static_bull = normalize_score(static_series, df.index, context_p, ascending=(polarity == 1))
            context_slope_bull = normalize_score(slope_series, df.index, context_p, ascending=True)
            context_accel_bull = normalize_score(accel_series, df.index, context_p, ascending=True)
            context_bullish_quality = (context_static_bull * context_slope_bull * context_accel_bull)**(1/3)
            bullish_snapshot_score = (tactical_bullish_quality * context_bullish_quality)**0.5
            # --- 看跌证据 ---
            # 战术层
            tactical_static_bear = normalize_score(static_series, df.index, p, ascending=(polarity == -1))
            tactical_slope_bear = normalize_score(slope_series, df.index, p, ascending=False)
            tactical_accel_bear = normalize_score(accel_series, df.index, p, ascending=False)
            tactical_bearish_quality = (tactical_static_bear * tactical_slope_bear * tactical_accel_bear)**(1/3)
            # 上下文层
            context_static_bear = normalize_score(static_series, df.index, context_p, ascending=(polarity == -1))
            context_slope_bear = normalize_score(slope_series, df.index, context_p, ascending=False)
            context_accel_bear = normalize_score(accel_series, df.index, context_p, ascending=False)
            context_bearish_quality = (context_static_bear * context_slope_bear * context_accel_bear)**(1/3)
            bearish_snapshot_score = (tactical_bearish_quality * context_bearish_quality)**0.5
            
            unified_d_intensity = self._perform_fund_flow_relational_meta_analysis(df, bullish_snapshot_score)
            s_bull[p] = bullish_snapshot_score.astype(np.float32)
            s_bear[p] = bearish_snapshot_score.astype(np.float32)
            d_intensity[p] = unified_d_intensity
        return {'s_bull': s_bull, 's_bear': s_bear, 'd_intensity': d_intensity}

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

    def _calculate_trend_context_ff(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【V1.0 · 新增】“波塞冬的三叉戟”资金流专属趋势上下文引擎
        - 核心逻辑: 从“排列、速度、元动力”三个维度评估趋势健康度，为资金流信号提供背景判断。
        - 新增武器: 引入“元动力”维度，利用跨周期导数捕捉长期趋势的短期变化，实现领先预判。
        """
        #
        p_context = get_param_value(params, {})
        weights = get_param_value(p_context.get('ma_trend_context_weights'), {
            'alignment': 0.4, 'velocity': 0.3, 'meta_dynamics': 0.3
        })
        norm_window = 55
        ma_periods = [5, 13, 21, 55, 89]
        ma_cols = [f'EMA_{p}_D' for p in ma_periods if f'EMA_{p}_D' in df.columns]
        if len(ma_cols) < 2:
            return pd.Series(0.5, index=df.index, dtype=np.float32)
        ma_values = np.stack([df[col].values for col in ma_cols], axis=0)
        # 叉戟一: 排列健康度 (Alignment)
        alignment_bools = ma_values[:-1] > ma_values[1:]
        alignment_health = np.mean(alignment_bools, axis=0) if alignment_bools.size > 0 else np.full(len(df.index), 0.5)
        # 叉戟二: 速度健康度 (Velocity) - 周期匹配导数
        slope_cols = [f'SLOPE_{p}_EMA_{p}_D' for p in ma_periods if f'SLOPE_{p}_EMA_{p}_D' in df.columns]
        if slope_cols:
            slope_values = np.stack([normalize_score(df[col], df.index, norm_window) for col in slope_cols], axis=0)
            velocity_health = np.mean(slope_values, axis=0)
        else:
            velocity_health = np.full(len(df.index), 0.5)
        # 叉戟三: 元动力健康度 (Meta-Dynamics) - 跨周期导数
        meta_dynamics_cols = ['SLOPE_5_EMA_55_D', 'SLOPE_13_EMA_89_D', 'SLOPE_21_EMA_144_D']
        valid_meta_cols = [col for col in meta_dynamics_cols if col in df.columns]
        if valid_meta_cols:
            meta_values = np.stack([normalize_score(df[col], df.index, norm_window) for col in valid_meta_cols], axis=0)
            meta_dynamics_health = np.mean(meta_values, axis=0)
        else:
            meta_dynamics_health = np.full(len(df.index), 0.5)
        # 最终融合：加权几何平均
        scores = np.stack([alignment_health, velocity_health, meta_dynamics_health], axis=0)
        weights_array = np.array(list(weights.values()))
        weights_array /= weights_array.sum()
        final_score_values = np.prod(scores ** weights_array[:, np.newaxis], axis=0)
        return pd.Series(final_score_values, index=df.index, dtype=np.float32)
        

















