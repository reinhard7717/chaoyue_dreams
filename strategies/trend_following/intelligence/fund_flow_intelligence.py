# 文件: strategies/trend_following/intelligence/fund_flow_intelligence.py
# 资金流情报模块
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from strategies.trend_following.utils import transmute_health_to_ultimate_signals, get_params_block, get_param_value, normalize_score, normalize_to_bipolar

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
        【V4.0 · 四象限重构版】基于物理公理的终极信号合成器
        - 核心重构: 引入“四象限动态分析法”，彻底解决信号命名与逻辑混乱的问题。
                      1. [正本清源] 将动态分析拆分为四个明确的象限：看涨加速、顶部反转、看跌加速、底部反转。
                      2. [信号新生] 为每个象限创建独立的、命名准确的信号，确保逻辑清晰。
        """
        states = {}
        axiom_weights = get_param_value(params.get('axiom_weights'), {'concentration': 0.4, 'power_transfer': 0.4, 'internal_structure': 0.2})
        tf_weights = get_param_value(params.get('tf_weights'), {1: 0.1, 5: 0.4, 13: 0.3, 21: 0.15, 55: 0.05})
        numeric_weights = {int(k): v for k, v in tf_weights.items() if isinstance(v, (int, float))}
        total_tf_weight = sum(numeric_weights.values())
        periods = sorted(numeric_weights.keys())

        # 步骤一：计算各周期的双极性“全息资金流健康分”
        bipolar_health_by_period = {}
        for p in periods:
            conc_score = concentration.get(p, 0.0)
            trans_score = power_transfer.get(p, 0.0)
            struct_score = internal_structure.get(p, 0.0)
            bipolar_health_by_period[p] = (
                conc_score * axiom_weights['concentration'] +
                trans_score * axiom_weights['power_transfer'] +
                struct_score * axiom_weights['internal_structure']
            ).clip(-1, 1)
        # 步骤二：分离为纯粹的看涨/看跌健康分
        bullish_scores_by_period = {p: score.clip(0, 1) for p, score in bipolar_health_by_period.items()}
        bearish_scores_by_period = {p: (score.clip(-1, 0) * -1) for p, score in bipolar_health_by_period.items()}
        # 步骤三：计算静态的共振信号 (零阶动态)
        bullish_resonance = pd.Series(0.0, index=df.index)
        bearish_resonance = pd.Series(0.0, index=df.index)
        if total_tf_weight > 0:
            for p, weight in numeric_weights.items():
                normalized_weight = weight / total_tf_weight
                bullish_resonance += bullish_scores_by_period.get(p, 0.0) * normalized_weight
                bearish_resonance += bearish_scores_by_period.get(p, 0.0) * normalized_weight
        # 步骤四：计算四象限动态信号 (一阶和二阶动态)
        bullish_accel_score = pd.Series(0.0, index=df.index)
        top_reversal_score = pd.Series(0.0, index=df.index)
        bearish_accel_score = pd.Series(0.0, index=df.index)
        bottom_reversal_score = pd.Series(0.0, index=df.index)
        if total_tf_weight > 0:
            for p, weight in numeric_weights.items():
                normalized_weight = weight / total_tf_weight
                context_p = periods[periods.index(p) + 1] if periods.index(p) + 1 < len(periods) else p
                # --- 基于“看涨健康分”的动态分析 ---
                holographic_bull_divergence = self._calculate_holographic_divergence_ff(bullish_scores_by_period.get(p, pd.Series(0.0, index=df.index)), 1, p, context_p)
                bullish_accel_score += holographic_bull_divergence.clip(0, 1) * normalized_weight
                top_reversal_score += (holographic_bull_divergence.clip(-1, 0) * -1) * normalized_weight
                # --- 基于“看跌健康分”的动态分析 ---
                holographic_bear_divergence = self._calculate_holographic_divergence_ff(bearish_scores_by_period.get(p, pd.Series(0.0, index=df.index)), 1, p, context_p)
                bearish_accel_score += holographic_bear_divergence.clip(0, 1) * normalized_weight
                bottom_reversal_score += (holographic_bear_divergence.clip(-1, 0) * -1) * normalized_weight
        # 步骤五：应用趋势上下文并构建最终信号字典
        trend_health_score = self._calculate_trend_context_ff(df, params)
        final_scores = {
            'bullish_resonance': (bullish_resonance * trend_health_score).clip(0, 1),
            'bearish_resonance': (bearish_resonance * (1 - trend_health_score)).clip(0, 1),
            'bullish_acceleration': (bullish_accel_score * trend_health_score).clip(0, 1),
            'top_reversal': (top_reversal_score * trend_health_score).clip(0, 1),
            'bearish_acceleration': (bearish_accel_score * (1 - trend_health_score)).clip(0, 1),
            'bottom_reversal': (bottom_reversal_score * (1 - trend_health_score)).clip(0, 1),
            'tactical_reversal': (bullish_resonance * top_reversal_score).clip(0, 1) # 战术反转 = 强看涨共振中的回调
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
        【V3.0 · 四象限重构版】将最终信号赋值给状态字典。
        - 核心升级: 全面更新信号映射，以匹配“四象限动态分析法”产出的新信号。
        """
        states = {}

        # 更新信号映射以匹配四象限逻辑
        prefix_map = {
            'bullish_resonance': 'SCORE_FF_BULLISH_RESONANCE',
            'bearish_resonance': 'SCORE_FF_BEARISH_RESONANCE',
            'bullish_acceleration': 'SCORE_FF_BULLISH_ACCELERATION',
            'top_reversal': 'SCORE_FF_TOP_REVERSAL',
            'bearish_acceleration': 'SCORE_FF_BEARISH_ACCELERATION',
            'bottom_reversal': 'SCORE_FF_BOTTOM_REVERSAL',
            'tactical_reversal': 'SCORE_FF_TACTICAL_REVERSAL',
        }
        
        for key, score in final_scores.items():
            signal_name = prefix_map.get(key)
            if signal_name:
                states[signal_name] = score.astype(np.float32)
        return states

    def _calculate_pillar_health(self, df: pd.DataFrame, config: Dict, norm_window: int, periods: list, ma_context_score: pd.Series) -> Dict:
        """
        【V10.1 · 赫利俄斯敕令版】计算单个资金流支柱的健康度
        - 核心革命: 签署“赫利俄斯敕令”，对双极性快照分执行关系元分析，得到最终动态分，再派生s_bull/s_bear。
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
        # 遵循“赫利俄斯敕令”
        for p in periods:
            context_p = periods[periods.index(p) + 1] if periods.index(p) + 1 < len(periods) else p
            slope_col_name = f"SLOPE_{p}_{base_col_name}_D"
            accel_col_name = f"ACCEL_{p}_{base_col_name}_D"
            slope_series = df.get(slope_col_name, pd.Series(0.0, index=df.index))
            accel_series = df.get(accel_col_name, pd.Series(0.0, index=df.index))
            # --- 看涨证据 ---
            tactical_static_bull = normalize_score(static_series, df.index, p, ascending=(polarity == 1))
            tactical_slope_bull = normalize_score(slope_series, df.index, p, ascending=True)
            tactical_accel_bull = normalize_score(accel_series, df.index, p, ascending=True)
            tactical_bullish_quality = (tactical_static_bull * tactical_slope_bull * tactical_accel_bull)**(1/3)
            context_static_bull = normalize_score(static_series, df.index, context_p, ascending=(polarity == 1))
            context_slope_bull = normalize_score(slope_series, df.index, context_p, ascending=True)
            context_accel_bull = normalize_score(accel_series, df.index, context_p, ascending=True)
            context_bullish_quality = (context_static_bull * context_slope_bull * context_accel_bull)**(1/3)
            bullish_snapshot_score = (tactical_bullish_quality * context_bullish_quality)**0.5
            # --- 看跌证据 ---
            tactical_static_bear = normalize_score(static_series, df.index, p, ascending=(polarity == -1))
            tactical_slope_bear = normalize_score(slope_series, df.index, p, ascending=False)
            tactical_accel_bear = normalize_score(accel_series, df.index, p, ascending=False)
            tactical_bearish_quality = (tactical_static_bear * tactical_slope_bear * tactical_accel_bear)**(1/3)
            context_static_bear = normalize_score(static_series, df.index, context_p, ascending=(polarity == -1))
            context_slope_bear = normalize_score(slope_series, df.index, context_p, ascending=False)
            context_accel_bear = normalize_score(accel_series, df.index, context_p, ascending=False)
            context_bearish_quality = (context_static_bear * context_slope_bear * context_accel_bear)**(1/3)
            bearish_snapshot_score = (tactical_bearish_quality * context_bearish_quality)**0.5
            # 1. 计算双极性快照分
            bipolar_snapshot = (bullish_snapshot_score - bearish_snapshot_score).clip(-1, 1)
            # 2. 对双极性快照分执行关系元分析，得到最终的动态健康分
            final_dynamic_score = self._perform_fund_flow_relational_meta_analysis(df, bipolar_snapshot)
            # 3. 从最终动态分中互斥地派生出 s_bull 和 s_bear
            s_bull[p] = final_dynamic_score.clip(0, 1).astype(np.float32)
            s_bear[p] = (final_dynamic_score.clip(-1, 0) * -1).astype(np.float32)
            # 4. 将 d_intensity 降级为无意义的占位符
            d_intensity[p] = pd.Series(1.0, index=df.index, dtype=np.float32)
        
        return {'s_bull': s_bull, 's_bear': s_bear, 'd_intensity': d_intensity}

    def _perform_fund_flow_relational_meta_analysis(self, df: pd.DataFrame, snapshot_score: pd.Series) -> pd.Series:
        """
        【V2.2 · 加速度校准版】资金流专用的关系元分析核心引擎
        - 核心修复: 修正了“加速度”计算的致命逻辑错误。加速度是速度的一阶导数，
                      因此其计算应为 relationship_trend.diff(1)，而不是错误的 diff(meta_window)。
        """
        p_conf = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        p_meta = get_param_value(p_conf.get('relational_meta_analysis_params'), {})
        w_state = get_param_value(p_meta.get('state_weight'), 0.3)
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.3)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.4)
        norm_window = 55
        meta_window = 5
        bipolar_sensitivity = 1.0
        state_score = snapshot_score.clip(0, 1)
        relationship_trend = snapshot_score.diff(meta_window).fillna(0)
        velocity_score = normalize_to_bipolar(
            series=relationship_trend, target_index=df.index,
            window=norm_window, sensitivity=bipolar_sensitivity
        )

        # 致命错误修复：加速度是速度(trend)的一阶导数，应使用 diff(1)
        relationship_accel = relationship_trend.diff(1).fillna(0)
        
        acceleration_score = normalize_to_bipolar(
            series=relationship_accel, target_index=df.index,
            window=norm_window, sensitivity=bipolar_sensitivity
        )
        final_score = (
            state_score * w_state +
            velocity_score * w_velocity +
            acceleration_score * w_acceleration
        ).clip(0, 1)
        return final_score.astype(np.float32)

    def _calculate_trend_context_ff(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【V1.1 · 类型安全版】“波塞冬的三叉戟”资金流专属趋势上下文引擎
        - 核心修复: 在处理从配置中读取的权重字典时，增加了对非数字类型值的过滤，
                      以防止 'description' 等说明性字段污染权重数组，修复潜在的类型错误。
        """
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
        alignment_bools = ma_values[:-1] > ma_values[1:]
        alignment_health = np.mean(alignment_bools, axis=0) if alignment_bools.size > 0 else np.full(len(df.index), 0.5)
        slope_cols = [f'SLOPE_{p}_EMA_{p}_D' for p in ma_periods if f'SLOPE_{p}_EMA_{p}_D' in df.columns]
        if slope_cols:
            slope_values = np.stack([normalize_score(df[col], df.index, norm_window) for col in slope_cols], axis=0)
            velocity_health = np.mean(slope_values, axis=0)
        else:
            velocity_health = np.full(len(df.index), 0.5)
        meta_dynamics_cols = ['SLOPE_5_EMA_55_D', 'SLOPE_13_EMA_89_D', 'SLOPE_21_EMA_144_D']
        valid_meta_cols = [col for col in meta_dynamics_cols if col in df.columns]
        if valid_meta_cols:
            meta_values = np.stack([normalize_score(df[col], df.index, norm_window) for col in valid_meta_cols], axis=0)
            meta_dynamics_health = np.mean(meta_values, axis=0)
        else:
            meta_dynamics_health = np.full(len(df.index), 0.5)
        scores = np.stack([alignment_health, velocity_health, meta_dynamics_health], axis=0)
        # 增加类型过滤，确保只处理数字类型的权重值
        numeric_weights = {k: v for k, v in weights.items() if isinstance(v, (int, float))}
        # print(f"      -> [FundFlowIntel:_calculate_trend_context_ff] 过滤后数字权重: {numeric_weights}")
        weights_array = np.array(list(numeric_weights.values()))
        
        if weights_array.sum() == 0:
            return pd.Series(0.5, index=df.index, dtype=np.float32)
        weights_array /= weights_array.sum()
        final_score_values = np.prod(scores ** weights_array[:, np.newaxis], axis=0)
        return pd.Series(final_score_values, index=df.index, dtype=np.float32)

    def _calculate_holographic_divergence_ff(self, series: pd.Series, short_p: int, long_p: int, norm_window: int) -> pd.Series:
        """
        【V1.0 · 新增】资金流专用的全息背离计算引擎
        - 战略意义: 洞察多时间维度的“结构性背离”，输出一个[-1, 1]的双极性背离分数。
        - 正分: 看涨背离 (短期趋势强于长期趋势)。
        - 负分: 看跌背离 (短期趋势弱于长期趋势)。
        """
        # 维度一：速度背离 (短期斜率 vs 长期斜率)
        slope_short = series.diff(short_p).fillna(0)
        slope_long = series.diff(long_p).fillna(0)
        velocity_divergence = slope_short - slope_long
        velocity_divergence_score = normalize_to_bipolar(velocity_divergence, series.index, norm_window)
        # 维度二：加速度背离 (短期加速度 vs 长期加速度)
        accel_short = slope_short.diff(short_p).fillna(0)
        accel_long = slope_long.diff(long_p).fillna(0)
        acceleration_divergence = accel_short - accel_long
        acceleration_divergence_score = normalize_to_bipolar(acceleration_divergence, series.index, norm_window)
        # 融合：速度背离和加速度背离的加权平均
        final_divergence_score = (velocity_divergence_score * 0.6 + acceleration_divergence_score * 0.4).clip(-1, 1)
        return final_divergence_score.astype(np.float32)


















