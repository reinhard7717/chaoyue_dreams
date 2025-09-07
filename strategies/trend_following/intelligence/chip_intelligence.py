# 文件: strategies/trend_following/intelligence/chip_intelligence.py
# 筹码情报模块
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following.utils import get_params_block, get_param_value

class ChipIntelligence:
    def __init__(self, strategy_instance, dynamic_thresholds: Dict):
        """
        初始化筹码情报模块。
        :param strategy_instance: 策略主实例的引用。
        :param dynamic_thresholds: 动态阈值字典。
        """
        self.strategy = strategy_instance
        self.dynamic_thresholds = dynamic_thresholds

    def run_chip_intelligence_command(self, df: pd.DataFrame) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
        """
        【V325.5 底层兼容修正版】筹码情报最高司令部
        - 核心修正: 修复了 `score.rolling().quantile()` 在接收分位数列表时可能引发 `TypeError` 的底层兼容性问题。
                      通过将批量计算分位数的逻辑修改为循环单次计算并合并结果，确保了代码在不同
                      pandas 版本下的健壮性，同时保持了原始业务逻辑不变。
        - 核心修正: 彻底重构了评分模块的调用和状态更新流程。现在，每当一个诊断模块（diagnose_*）
                      运行完毕后，其产出的新分数会立即被更新到 self.strategy.atomic_states 中。
                      这解决了下游模块因无法及时获取上游分数而跳过的严重依赖问题。
        - 逻辑优化: 将 'cost_divergence_D' 的计算前置到本函数开头，确保该衍生指标在所有
                      诊断模块运行前就已准备就绪，理顺了数据流。
        """
        print("        -> [筹码情报最高司令部 V325.5 底层兼容修正版] 启动...")
        states = {}
        triggers = {}
        # [新增] 定义一个辅助函数，用于在每个诊断模块后增量更新原子状态库
        initial_cols = set(df.columns)
        def _update_atomic_states(df_to_update: pd.DataFrame, last_cols: set) -> set:
            """根据df的变化，增量更新atomic_states"""
            current_cols = set(df_to_update.columns)
            new_score_cols = current_cols - last_cols
            if new_score_cols:
                new_scores_dict = {col: df_to_update[col] for col in new_score_cols}
                self.strategy.atomic_states.update(new_scores_dict)
                # print(f"          -> [情报更新] {len(new_score_cols)}个新评分已更新至原子状态库。")
            return current_cols
        # [新增] 步骤 0: 预处理衍生指标，解决计算依赖问题
        if 'avg_cost_short_term_D' in df.columns and 'avg_cost_long_term_D' in df.columns:
            df['cost_divergence_D'] = df['avg_cost_short_term_D'] - df['avg_cost_long_term_D']
            initial_cols.add('cost_divergence_D') # 将新列也加入初始列集合
            print("          -> [预处理] 已成功计算衍生指标 'cost_divergence_D'。")
        # --- [修改] 步骤 1: 链式调用四级评分中心，并确保状态实时传递 ---
        # 记录调用前的列名，用于后续识别新的评分列
        cols_before_run = initial_cols
        # 1.1 调用宏观共振/反转诊断模块
        df = self.diagnose_quantitative_chip_scores(df)
        cols_before_run = _update_atomic_states(df, cols_before_run)
        # 1.2 调用高级动态诊断模块 (微观结构与极端行为)
        df = self.diagnose_advanced_chip_dynamics_scores(df)
        cols_before_run = _update_atomic_states(df, cols_before_run)
        # 1.3 调用内部结构诊断模块
        df = self.diagnose_chip_internal_structure_scores(df)
        cols_before_run = _update_atomic_states(df, cols_before_run)
        # 1.4 调用持仓者行为诊断模块
        df = self.diagnose_chip_holder_behavior_scores(df)
        cols_before_run = _update_atomic_states(df, cols_before_run)
        # 1.5 调用行为-筹码融合评分模块
        df = self.diagnose_fused_behavioral_chip_scores(df)
        cols_before_run = _update_atomic_states(df, cols_before_run)
        # 1.6: 调用 V2.0 版本的元融合模块
        prime_opp_states, prime_opp_scores = self.synthesize_prime_chip_opportunity(df)
        states.update(prime_opp_states)
        if prime_opp_scores:
            df = df.assign(**prime_opp_scores)
        # 获取模块参数，检查是否启用
        p = get_params_block(self.strategy, 'chip_feature_params')
        if not get_param_value(p.get('enabled'), False):
            return states, triggers
        # --- 步骤 2: 定义主信号配置字典 (将数值评分转化为布尔信号) ---
        # 此处配置保持不变，它负责将特定的数值评分，根据动态阈值，转化为最终的布尔型 state 或 trigger
        MASTER_SIGNAL_CONFIG = {
            # --- 司令部顶层信号 ---
            'CONTEXT_CHIP_STRATEGIC_GATHERING': ('CHIP_SCORE_CONTEXT_STRATEGIC_GATHERING', 0.60, 120, 'state'),
            'CONTEXT_CHIP_STRATEGIC_DISTRIBUTION': ('CHIP_SCORE_CONTEXT_STRATEGIC_GATHERING', 0.40, 120, 'state_lt'),
            'CONTEXT_EUPHORIC_RALLY_WARNING': ('CHIP_SCORE_CONTEXT_EUPHORIC_RALLY', 0.90, 120, 'state'),
            'TRIGGER_CHIP_IGNITION': ('CHIP_SCORE_TRIGGER_IGNITION', 0.98, 120, 'trigger'),
            'RISK_CONTEXT_LONG_TERM_DISTRIBUTION': ('CHIP_SCORE_RISK_LONG_TERM_DISTRIBUTION', 0.90, 120, 'state'),
            'RISK_CHIP_CONC_ACCEL_WORSENING': ('CHIP_SCORE_RISK_WORSENING_TURN', 0.80, 60, 'state_gt_zero'),
            'OPP_STATIC_DYN_BREAKTHROUGH_S': ('CHIP_SCORE_OPP_BREAKTHROUGH', 0.95, 120, 'state'),
            'RISK_STATIC_DYN_COLLAPSE_S': ('CHIP_SCORE_RISK_COLLAPSE', 0.95, 120, 'state'),
            'OPP_STATIC_DYN_INFLECTION_A': ('CHIP_SCORE_OPP_INFLECTION', 0.90, 120, 'state'),
            # --- 分级信号 (Bucket Signals) ---
            'BUCKET_CHIP_CONC_GATHERING': ('CHIP_SCORE_GATHERING_INTENSITY', {
                'CHIP_CONC_STEADY_GATHERING_C': 0.70,
                'CHIP_CONC_ACCELERATED_GATHERING_B': 0.85,
                'CHIP_CONC_INTENSIFYING_B_PLUS': 0.95
            }, 120, 'bucket_upper'),
            'CHIP_PROFIT_TAKING_INTENSE_A': ('SCORE_CHIP_PROFIT_TAKING_INTENSITY', 0.90, 120, 'state'),
        }
        available_cols = set(df.columns)
        all_generated_states = {}
        # --- 步骤 3: 按 (评分列, 窗口, 信号大类) 对信号配置进行分组，以优化计算 ---
        from collections import defaultdict
        grouped_signals = defaultdict(list)
        for signal_name, config_tuple in MASTER_SIGNAL_CONFIG.items():
            score_col, quantile_or_dict, window, signal_type = config_tuple
            if score_col in available_cols:
                group_key = 'gt_zero' if 'gt_zero' in signal_type else 'bucket_upper' if signal_type == 'bucket_upper' else 'standard'
                grouped_signals[(score_col, window, group_key)].append((signal_name, quantile_or_dict, signal_type))
        # --- 步骤 4: 批处理所有信号，高效生成布尔状态 ---
        for (score_col, window, group_key), tasks in grouped_signals.items():
            score = df[score_col]
            if group_key == 'bucket_upper':
                quantiles_needed = sorted(list(set(tasks[0][1].values())))
            else:
                quantiles_needed = sorted(list(set(q for _, q, _ in tasks)))
            thresholds_df = None
            if group_key in ['standard', 'bucket_upper']:
                if not score.isnull().all():
                    # [修改] 修复 'TypeError: must be real number, not list'
                    # 将一次性计算多个分位数，改为循环计算单个分位数，然后合并结果
                    # 这样可以避免将列表传递给可能不支持此操作的底层pandas函数
                    thresholds_list = []
                    for q_val in quantiles_needed:
                        s = score.rolling(window).quantile(q_val)
                        s.name = q_val  # 为Series命名，以便后续合并
                        thresholds_list.append(s)
                    if thresholds_list:
                        thresholds_df = pd.concat(thresholds_list, axis=1)
            elif group_key == 'gt_zero':
                positive_scores = score[score > 0]
                if not positive_scores.empty:
                    # [修改] 对 gt_zero 分组也应用同样的修复逻辑
                    thresholds_list = []
                    for q_val in quantiles_needed:
                        s = positive_scores.rolling(window).quantile(q_val)
                        s.name = q_val
                        thresholds_list.append(s)
                    if thresholds_list:
                        thresholds_df = pd.concat(thresholds_list, axis=1).reindex(score.index).ffill()
            if thresholds_df is None: continue
            # [删除] 原本的 to_frame 逻辑不再需要，因为 concat 已经生成了 DataFrame
            # 根据信号类型进行逻辑分发
            if group_key == 'bucket_upper':
                _, quantile_dict, _ = tasks[0] # 一个分组只有一个 bucket_upper 任务
                sorted_levels = sorted(quantile_dict.items(), key=lambda item: item[1])
                for i, (final_signal_name, q_lower) in enumerate(sorted_levels):
                    lower_thresh = thresholds_df[q_lower]
                    if i == len(sorted_levels) - 1:
                        signal = score > lower_thresh
                    else:
                        _, q_upper = sorted_levels[i+1]
                        upper_thresh = thresholds_df[q_upper]
                        signal = (score > lower_thresh) & (score <= upper_thresh)
                    all_generated_states[final_signal_name] = signal
            else: # 处理 standard 和 gt_zero
                for signal_name, quantile, signal_type in tasks:
                    threshold = thresholds_df[quantile]
                    signal = pd.Series(False, index=df.index)
                    if threshold.notna().any():
                        if signal_type in ['state', 'trigger', 'state_gt_zero']:
                            signal = score > threshold
                        elif signal_type == 'state_le':
                            signal = score <= threshold
                        elif signal_type == 'state_lt':
                            signal = score < threshold
                        elif signal_type == 'state_gt_zero_event':
                            signal = (score > threshold) & (score > 0)
                    if signal_type == 'trigger':
                        triggers[signal_name] = signal
                    else:
                        all_generated_states[signal_name] = signal
        # --- 步骤 5: 最终状态更新 ---
        # 将本模块生成的布尔信号更新到主状态字典和原子状态库中
        states.update(all_generated_states)
        self.strategy.atomic_states.update(all_generated_states)
        return states, triggers

    def synthesize_prime_chip_opportunity(self, df: pd.DataFrame) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
        """
        【V2.0 分层加权融合版】黄金筹码机会元融合模块
        - 核心职责: 将多个独立的、描述筹码结构健康度的S/A/B三级数值评分，通过
                      “维度内置信度加权”和“维度间重要性加权”两步，融合成一个顶层的、
                      能精细度量机会“成色”的元分数。
        - 收益: 实现了从“信号有无”到“机会质量”的升维，为决策提供了更平滑、更鲁棒的依据。
        """
        print("        -> [黄金筹码机会元融合模块 V2.0 分层加权融合版] 启动...")
        states = {}
        new_scores = {}
        atomic = self.strategy.atomic_states
        p_module = get_params_block(self.strategy, 'prime_chip_opportunity_params_v2', {})
        if not get_param_value(p_module.get('enabled'), True):
            return states, new_scores
        # --- 1. 加载参数：维度权重与置信度权重 ---
        dim_weights = get_param_value(p_module.get('dimension_weights'), {
            'structure_health': 0.35, 'core_holder': 0.30, 'net_support': 0.20, 'cost_structure': 0.15
        })
        conf_weights = get_param_value(p_module.get('confidence_weights'), {
            'S': 1.0, 'A': 0.6, 'B': 0.3
        })
        total_conf_weight = sum(conf_weights.values())
        # --- 2. 军备检查：获取所有S/A/B三级评分 ---
        score_map = {
            'structure_health': ['SCORE_STRUCTURE_BULLISH_RESONANCE_S', 'SCORE_STRUCTURE_BULLISH_RESONANCE_A', 'SCORE_STRUCTURE_BULLISH_RESONANCE_B'],
            'core_holder': ['SCORE_CORE_HOLDER_BULLISH_RESONANCE_S', 'SCORE_CORE_HOLDER_BULLISH_RESONANCE_A', 'SCORE_CORE_HOLDER_BULLISH_RESONANCE_B'],
            'net_support': [None, 'SCORE_NET_SUPPORT_BULLISH_RESONANCE_A', 'SCORE_NET_SUPPORT_BULLISH_RESONANCE_B'], # Net Support 没有S级
            'cost_structure': ['SCORE_COST_STRUCTURE_BULLISH_RESONANCE_S', 'SCORE_COST_STRUCTURE_BULLISH_RESONANCE_A', 'SCORE_COST_STRUCTURE_BULLISH_RESONANCE_B']
        }
        all_required_scores = [s for group in score_map.values() for s in group if s]
        missing_scores = [s for s in all_required_scores if s not in atomic]
        if missing_scores:
            print(f"          -> [警告] synthesize_prime_chip_opportunity黄金机会融合模块缺少上游分数: {missing_scores}，模块已跳过。")
            return states, new_scores
        # --- 3. 维度内融合：计算每个维度的综合强度分 ---
        fused_dimension_scores = {}
        default_series = pd.Series(0.0, index=df.index, dtype=np.float32)
        for dim_name, (s_score_name, a_score_name, b_score_name) in score_map.items():
            s_score = atomic.get(s_score_name, default_series) if s_score_name else default_series
            a_score = atomic.get(a_score_name, default_series) if a_score_name else default_series
            b_score = atomic.get(b_score_name, default_series) if b_score_name else default_series
            # 置信度加权求和，并归一化
            fused_score = (s_score * conf_weights['S'] + a_score * conf_weights['A'] + b_score * conf_weights['B']) / total_conf_weight
            fused_dimension_scores[dim_name] = fused_score
        # --- 4. 维度间融合：计算最终的“黄金机会”元分数 ---
        final_prime_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        for dim_name, weight in dim_weights.items():
            final_prime_score += fused_dimension_scores[dim_name] * weight
        new_scores['CHIP_SCORE_PRIME_OPPORTUNITY_S'] = final_prime_score.clip(0, 1)
        # --- 5. 基于元分数，生成兼容性的布尔信号 ---
        threshold = get_param_value(p_module.get('prime_score_threshold_for_bool'), 0.7)
        prime_opportunity_signal = new_scores['CHIP_SCORE_PRIME_OPPORTUNITY_S'] > threshold
        states['CHIP_STRUCTURE_PRIME_OPPORTUNITY_S'] = prime_opportunity_signal
        # print("        -> [黄金筹码机会元融合模块 V2.0 分层加权融合版] 计算完毕。")
        return states, new_scores

    def diagnose_quantitative_chip_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V4.0 共振-反转对称诊断版】筹码信号量化评分诊断模块
        - 核心重构: 遵循“共振/反转”对称原则，将旧版零散的评分体系，全面升级为基于多维交叉验证的、结构化的四象限信号矩阵。
        - 核心逻辑:
          - 共振信号 (多周期交叉): 评估筹码集中度、成本重心在多时间周期上的一致性。
          - 反转信号 (同周期交叉): 评估“静态战备(Setup)”与“动态点火(Trigger)”的结合。
        -  信号 (数值型, 对称设计):
          - SCORE_CHIP_BULLISH_RESONANCE_S/A/B: 上升共振机会分 (多周期筹码集中)。
          - SCORE_CHIP_BEARISH_RESONANCE_S/A/B: 下跌共振风险分 (多周期筹码发散)。
          - SCORE_CHIP_BOTTOM_REVERSAL_S/A/B: 底部反转机会分 (下跌衰竭后吸筹)。
          - SCORE_CHIP_TOP_REVERSAL_S/A/B: 顶部反转风险分 (上涨衰竭后派发)。
        """
        print("        -> [筹码信号量化评分模块 V4.0 共振-反转对称诊断版] 启动...")
        new_scores = {}
        p = get_params_block(self.strategy, 'chip_feature_params')
        if not get_param_value(p.get('enabled'), True):
            return df
        # --- 1. 军备检查 (Arsenal Check) ---
        periods = get_param_value(p.get('dynamic_periods'), [5, 21, 55])
        required_cols = [
            'concentration_90pct_D', 'peak_cost_D', 'total_winner_rate_D',
            'turnover_from_winners_ratio_D', 'price_to_peak_ratio_D', 'chip_health_score_D'
        ]
        #将循环变量从 p 修改为 period，以避免覆盖参数字典 p
        for period in periods:
            required_cols.extend([
                #使用新的循环变量 period
                f'SLOPE_{period}_concentration_90pct_D', f'SLOPE_{period}_peak_cost_D',
                #使用新的循环变量 period
                f'ACCEL_{period if period > 5 else 5}_concentration_90pct_D',
                #使用新的循环变量 period
                f'ACCEL_{period if period > 5 else 5}_peak_cost_D',
                #使用新的循环变量 period
                f'ACCEL_{period if period > 5 else 5}_turnover_from_winners_ratio_D'
            ])
        # --- 2. 检查必需列 ---
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"          -> [严重警告] 筹码评分引擎缺少关键数据列: {missing_cols}，模块已跳过！")
            return df
        # --- 3. 核心要素数值化 (归一化处理) ---
        #此处的 p 现在是正确的参数字典，因为没有被循环覆盖
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        def normalize(series, ascending=True):
            return series.rolling(window=norm_window, min_periods=min_periods).rank(pct=True, ascending=ascending).fillna(0.5)
        # 动态要素 (动能与加速度)
        conc_momentum_scores = {p: normalize(df[f'SLOPE_{p}_concentration_90pct_D']) for p in periods}
        cost_momentum_scores = {p: normalize(df[f'SLOPE_{p}_peak_cost_D']) for p in periods}
        conc_accel_scores = {p: normalize(df[f'ACCEL_{p if p > 5 else 5}_concentration_90pct_D']) for p in periods}
        cost_accel_scores = {p: normalize(df[f'ACCEL_{p if p > 5 else 5}_peak_cost_D']) for p in periods}
        profit_taking_accel_score = normalize(df[f'ACCEL_5_turnover_from_winners_ratio_D'])
        # 静态要素 (当前状态)
        static_concentration = normalize(df['concentration_90pct_D'])
        static_health = normalize(df['chip_health_score_D'])
        static_price_deviation = normalize(df['price_to_peak_ratio_D'])
        static_high_winner_rate = normalize(df['total_winner_rate_D'])
        # --- 4. 共振信号合成 (多时间周期交叉验证) ---
        avg_conc_momentum = pd.Series(np.mean(np.array([s.values for s in conc_momentum_scores.values()]), axis=0), index=df.index)
        avg_cost_momentum = pd.Series(np.mean(np.array([s.values for s in cost_momentum_scores.values()]), axis=0), index=df.index)
        # 4.1 上升共振 (筹码集中 + 成本抬高)
        new_scores['SCORE_CHIP_BULLISH_RESONANCE_B'] = avg_conc_momentum.astype(np.float32)
        new_scores['SCORE_CHIP_BULLISH_RESONANCE_A'] = (avg_conc_momentum * avg_cost_momentum).astype(np.float32)
        new_scores['SCORE_CHIP_BULLISH_RESONANCE_S'] = (new_scores['SCORE_CHIP_BULLISH_RESONANCE_A'] * static_health).astype(np.float32)
        # 4.2 下跌共振 (筹码发散 + 成本降低) - 对称逻辑
        avg_conc_divergence = 1 - avg_conc_momentum
        avg_cost_decline = 1 - avg_cost_momentum
        static_unhealth = 1 - static_health
        new_scores['SCORE_CHIP_BEARISH_RESONANCE_B'] = avg_conc_divergence.astype(np.float32)
        new_scores['SCORE_CHIP_BEARISH_RESONANCE_A'] = (avg_conc_divergence * avg_cost_decline).astype(np.float32)
        new_scores['SCORE_CHIP_BEARISH_RESONANCE_S'] = (new_scores['SCORE_CHIP_BEARISH_RESONANCE_A'] * static_unhealth).astype(np.float32)
        # --- 5. 反转信号合成 (静态战备 x 动态点火) ---
        # 5.1 底部反转 (环境恶劣 -> 动态改善)
        bottom_setup_score = (1 - static_concentration) * (1 - static_price_deviation) # 战备: 筹码曾发散 + 价格处于低位
        bottom_trigger_score = conc_momentum_scores[5] * cost_accel_scores[5] # 点火: 短期开始集中 + 成本加速抬升
        new_scores['SCORE_CHIP_BOTTOM_REVERSAL_B'] = bottom_trigger_score.astype(np.float32)
        new_scores['SCORE_CHIP_BOTTOM_REVERSAL_A'] = (bottom_setup_score * bottom_trigger_score).astype(np.float32)
        new_scores['SCORE_CHIP_BOTTOM_REVERSAL_S'] = (new_scores['SCORE_CHIP_BOTTOM_REVERSAL_A'] * conc_accel_scores[5]).astype(np.float32)
        # 5.2 顶部反转 (环境良好 -> 动态恶化) - 对称逻辑
        top_setup_score = static_concentration * static_price_deviation * static_high_winner_rate # 战备: 筹码集中 + 价格高位 + 获利盘丰厚
        top_trigger_score = (1 - conc_momentum_scores[5]) * profit_taking_accel_score # 点火: 短期开始发散 + 获利盘兑现加速
        new_scores['SCORE_CHIP_TOP_REVERSAL_B'] = top_trigger_score.astype(np.float32)
        new_scores['SCORE_CHIP_TOP_REVERSAL_A'] = (top_setup_score * top_trigger_score).astype(np.float32)
        new_scores['SCORE_CHIP_TOP_REVERSAL_S'] = (new_scores['SCORE_CHIP_TOP_REVERSAL_A'] * (1 - conc_accel_scores[5])).astype(np.float32)
        # --- 6.  纯筹码评分: 获利盘兑现强度 ---
        # 逻辑: 获利盘换手率越高，且获利盘比例也高时，兑现强度越大。
        profit_taking_turnover_score = normalize(df['turnover_from_winners_ratio_D'])
        new_scores['SCORE_CHIP_PROFIT_TAKING_INTENSITY'] = (profit_taking_turnover_score * static_high_winner_rate).astype(np.float32)
        df = df.assign(**new_scores)
        print("        -> [筹码信号量化评分模块 V4.0 共振-反转对称诊断版] 计算完毕。") 
        return df

    def diagnose_advanced_chip_dynamics_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V2.2 终极数值化版】高级筹码动态评分模块
        - 核心升级 (本次修改):
          - [数值化] 将“筹码断层风险”的计算，从依赖硬布尔开关 'is_chip_fault_formed_D'，
                      升级为将“存在性”、“强度”、“真空度”和“动态恶化趋势”四要素
                      平滑融合的连续风险评分体系。
        - 收益: 彻底消除了模块内最后一个硬编码的布尔逻辑，使得“断层风险”的度量
                更加精细和连续，避免了信号的突变，信号质量达到理论最高水平。
        """
        print("        -> [高级筹码动态评分模块 V2.2 终极数值化版] 启动...")
        new_scores = {}
        p = get_params_block(self.strategy, 'advanced_chip_dynamics_params')
        if not get_param_value(p.get('enabled'), True):
            return df
        # --- 1. 军备检查 (Arsenal Check) ---
        periods = get_param_value(p.get('dynamic_periods'), [5, 21, 55])
        required_cols = [
            'peak_control_ratio_D', 'peak_strength_ratio_D', 'peak_stability_D',
            'turnover_from_losers_ratio_D', 'is_chip_fault_formed_D',
            'chip_fault_strength_D', 'chip_fault_vacuum_percent_D'
        ]
        for period in periods:
            required_cols.extend([
                f'SLOPE_{period}_peak_control_ratio_D', f'SLOPE_{period}_peak_stability_D'
            ])
        required_cols.extend([
            'ACCEL_5_turnover_from_losers_ratio_D', 'ACCEL_21_turnover_from_losers_ratio_D'
        ])
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"          -> [严重警告] 高级筹码动态模块缺少关键数据列: {missing_cols}，模块已跳过！")
            return df
        # --- 2. 核心要素数值化 (归一化处理) ---
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        def normalize(series, ascending=True):
            return series.rolling(window=norm_window, min_periods=min_periods).rank(pct=True, ascending=ascending).fillna(0.5)
        # --- 3. 维度一: 结构健康度共振评分 ---
        # 3.1 静态健康分 (Setup)
        control_score = normalize(df['peak_control_ratio_D'])
        strength_score = normalize(df['peak_strength_ratio_D'])
        stability_score = normalize(df['peak_stability_D'])
        static_health_score = (control_score * strength_score * stability_score)
        # 3.2 动态健康分 (Momentum) - 多周期交叉验证
        control_momentum = {p: normalize(df[f'SLOPE_{p}_peak_control_ratio_D']) for p in periods}
        stability_momentum = {p: normalize(df[f'SLOPE_{p}_peak_stability_D']) for p in periods}
        avg_health_momentum = pd.Series(np.mean(np.array([
            control_momentum[5].values, control_momentum[21].values, control_momentum[55].values,
            stability_momentum[5].values, stability_momentum[21].values, stability_momentum[55].values
        ]), axis=0), index=df.index)
        # 3.3 上升共振 (健康度改善)
        new_scores['SCORE_STRUCTURE_BULLISH_RESONANCE_B'] = avg_health_momentum.astype(np.float32)
        new_scores['SCORE_STRUCTURE_BULLISH_RESONANCE_A'] = (static_health_score * avg_health_momentum).astype(np.float32)
        new_scores['SCORE_STRUCTURE_BULLISH_RESONANCE_S'] = (new_scores['SCORE_STRUCTURE_BULLISH_RESONANCE_A'] * control_momentum[5] * stability_momentum[5]).astype(np.float32)
        # 3.4 下跌共振 (健康度恶化) - 对称逻辑
        static_unhealth_score = 1 - static_health_score
        avg_health_decline = 1 - avg_health_momentum
        new_scores['SCORE_STRUCTURE_BEARISH_RESONANCE_B'] = avg_health_decline.astype(np.float32)
        new_scores['SCORE_STRUCTURE_BEARISH_RESONANCE_A'] = (static_unhealth_score * avg_health_decline).astype(np.float32)
        new_scores['SCORE_STRUCTURE_BEARISH_RESONANCE_S'] = (new_scores['SCORE_STRUCTURE_BEARISH_RESONANCE_A'] * (1 - control_momentum[5]) * (1 - stability_momentum[5])).astype(np.float32)
        # --- 4. 维度二: 恐慌盘承接反转评分 ---
        # 4.1 战备 (Setup): 市场出现恐慌性抛售
        panic_selling_score = normalize(df['turnover_from_losers_ratio_D'])
        # 4.2 点火 (Trigger): 抛售行为开始减速 (加速度转正)，表明有资金在承接
        absorption_trigger_score = normalize(df['ACCEL_5_turnover_from_losers_ratio_D'])
        # 4.3 确认 (Confirmation): 中期抛售加速度也转正，趋势更可靠
        absorption_confirm_score = normalize(df['ACCEL_21_turnover_from_losers_ratio_D'])
        # 4.4 融合生成S/A/B三级底部反转信号
        new_scores['SCORE_CAPITULATION_BOTTOM_REVERSAL_B'] = absorption_trigger_score.astype(np.float32)
        new_scores['SCORE_CAPITULATION_BOTTOM_REVERSAL_A'] = (panic_selling_score.shift(1) * absorption_trigger_score).astype(np.float32)
        new_scores['SCORE_CAPITULATION_BOTTOM_RESONANCE_S'] = (new_scores['SCORE_CAPITULATION_BOTTOM_REVERSAL_A'] * absorption_confirm_score).astype(np.float32)
        # --- 5. 维度三: 筹码断层顶部反转风险分 ---
        # 5.1 风险放大器 (Amplifier): 断层强度大 & 下方真空区广阔
        fault_strength_score = normalize(df['chip_fault_strength_D'])
        vacuum_risk_score = normalize(df['chip_fault_vacuum_percent_D'])
        # 5.2 风险动态恶化分 (Dynamic Worsening): 结构健康度在恶化，放大断层风险
        dynamic_worsening_score = new_scores.get('SCORE_STRUCTURE_BEARISH_RESONANCE_B', pd.Series(0.5, index=df.index))
        # 5.3 融合生成S/A/B三级顶部反转风险信号 (原 5.4)
        # B级风险: 直接由“断层强度”决定，这是最基础的风险来源
        new_scores['SCORE_FAULT_RISK_TOP_REVERSAL_B'] = fault_strength_score.astype(np.float32)
        # A级风险: 在B级基础上，叠加上“真空度”风险
        new_scores['SCORE_FAULT_RISK_TOP_REVERSAL_A'] = (new_scores['SCORE_FAULT_RISK_TOP_REVERSAL_B'] * vacuum_risk_score).astype(np.float32)
        # S级风险: 在A级基础上，叠加上“动态恶化”的趋势，代表最高风险
        new_scores['SCORE_FAULT_RISK_TOP_REVERSAL_S'] = (new_scores['SCORE_FAULT_RISK_TOP_REVERSAL_A'] * dynamic_worsening_score).astype(np.float32)
        # --- 6. 一次性将所有新得分合并到DataFrame ---
        df = df.assign(**new_scores)
        print("        -> [高级筹码动态评分模块 V2.2 终极数值化版] 计算完毕。") 
        return df

    def diagnose_chip_internal_structure_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V2.0 共振-反转诊断版】筹码内部结构评分模块
        - 核心升级: 遵循“共振/反转”对称原则，对核心持仓者、兑现压力、支撑压力进行多维交叉验证。
        - 核心逻辑:
          - 核心持仓者动态 -> 上升共振: 验证核心筹码(70%)是否比外围(90%)集中更快。
          - 利润兑现压力 -> 顶部反转: 结合静态利润厚度与动态利润增速，评估抛压。
          - 结构支撑压力 -> 上升共振: 结合静态净支撑与动态净支撑变化趋势。
        -  信号 (数值型, 对称设计):
          - SCORE_CORE_HOLDER_BULLISH_RESONANCE_S/A/B: 核心持仓者上升共振分。
          - SCORE_PROFIT_TAKING_TOP_REVERSAL_S/A/B: 获利盘兑现顶部反转风险分。
          - SCORE_NET_SUPPORT_BULLISH_RESONANCE_A/B: 净支撑上升共振分。
        """
        print("        -> [筹码内部结构评分模块 V2.0 共振-反转诊断版] 启动...") #更新版本号和打印信息
        new_scores = {}
        p = get_params_block(self.strategy, 'chip_internal_structure_params')
        if not get_param_value(p.get('enabled'), True):
            return df

        # --- 1. 军备检查 (Arsenal Check) ---
        #扩展所需列
        periods = get_param_value(p.get('dynamic_periods'), [5, 21, 55])
        required_cols = [
            'concentration_70pct_D', 'concentration_90pct_D', 'winner_profit_margin_D',
            'support_below_D', 'pressure_above_D'
        ]
        for period in periods:
            required_cols.extend([
                f'SLOPE_{period}_concentration_70pct_D', f'SLOPE_{period}_concentration_90pct_D',
                f'SLOPE_{period}_winner_profit_margin_D'
            ])
        required_cols.extend(['ACCEL_5_winner_profit_margin_D', 'SLOPE_5_support_below_D', 'SLOPE_5_pressure_above_D'])
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"          -> [严重警告] 筹码内部结构模块缺少关键数据列: {missing_cols}，模块已跳过！")
            return df

        # --- 2. 核心要素数值化 (归一化处理) ---
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        def normalize(series, ascending=True):
            return series.rolling(window=norm_window, min_periods=min_periods).rank(pct=True, ascending=ascending).fillna(0.5)

        # --- 3. 维度一: 核心持仓者上升共振 ---
        # 核心逻辑: 核心筹码(70%)的集中速度(斜率)快于外围筹码(90%)
        # 3.1 静态分 (Setup): 70%集中度本身处于较低水平（已很集中）
        static_core_conc = normalize(df['concentration_70pct_D'], ascending=False)
        # 3.2 动态分 (Momentum): 核心集中动能 - 外围集中动能
        core_momentum = {p: normalize(df[f'SLOPE_{p}_concentration_70pct_D'], ascending=False) for p in periods}
        peripheral_momentum = {p: normalize(df[f'SLOPE_{p}_concentration_90pct_D'], ascending=False) for p in periods}
        net_gathering_momentum = {p: (core_momentum[p] - peripheral_momentum[p]) for p in periods}
        avg_net_gathering_momentum = normalize(pd.Series(np.mean(np.array([s.values for s in net_gathering_momentum.values()]), axis=0), index=df.index))

        # 3.3 融合生成S/A/B三级上升共振信号
        new_scores['SCORE_CORE_HOLDER_BULLISH_RESONANCE_B'] = avg_net_gathering_momentum.astype(np.float32)
        new_scores['SCORE_CORE_HOLDER_BULLISH_RESONANCE_A'] = (static_core_conc * avg_net_gathering_momentum).astype(np.float32)
        new_scores['SCORE_CORE_HOLDER_BULLISH_RESONANCE_S'] = (new_scores['SCORE_CORE_HOLDER_BULLISH_RESONANCE_A'] * normalize(net_gathering_momentum[5])).astype(np.float32)

        # --- 4. 维度二: 获利盘兑现顶部反转风险 ---
        # 4.1 静态风险 (Setup): 获利盘利润丰厚
        static_profit_margin = normalize(df['winner_profit_margin_D'])
        # 4.2 动态风险 (Trigger): 利润仍在快速增长，加速兑现冲动
        profit_margin_momentum = normalize(df['SLOPE_5_winner_profit_margin_D'])
        # 4.3 风险加速度 (Confirmation): 利润增长在加速
        profit_margin_accel = normalize(df['ACCEL_5_winner_profit_margin_D'])

        # 4.4 融合生成S/A/B三级顶部反转风险信号
        new_scores['SCORE_PROFIT_TAKING_TOP_REVERSAL_B'] = static_profit_margin.astype(np.float32)
        new_scores['SCORE_PROFIT_TAKING_TOP_REVERSAL_A'] = (static_profit_margin * profit_margin_momentum).astype(np.float32)
        new_scores['SCORE_PROFIT_TAKING_TOP_REVERSAL_S'] = (new_scores['SCORE_PROFIT_TAKING_TOP_REVERSAL_A'] * profit_margin_accel).astype(np.float32)

        # --- 5. 维度三: 净支撑上升共振 ---
        # 5.1 静态机会 (Setup): 下方支撑远大于上方压力
        static_support = normalize(df['support_below_D'])
        static_pressure = normalize(df['pressure_above_D'])
        static_net_support = normalize(static_support - static_pressure)
        # 5.2 动态机会 (Momentum): 支撑在增强，压力在减弱
        support_momentum = normalize(df['SLOPE_5_support_below_D'])
        pressure_momentum = normalize(df['SLOPE_5_pressure_above_D'], ascending=False) # 压力斜率越小越好
        dynamic_net_support = (support_momentum * pressure_momentum)

        # 5.3 融合生成A/B两级上升共振信号 (S级需要更复杂数据，暂不生成)
        new_scores['SCORE_NET_SUPPORT_BULLISH_RESONANCE_B'] = dynamic_net_support.astype(np.float32)
        new_scores['SCORE_NET_SUPPORT_BULLISH_RESONANCE_A'] = (static_net_support * dynamic_net_support).astype(np.float32)

        # --- 6. 一次性将所有新得分合并到DataFrame ---
        df = df.assign(**new_scores)
        print("        -> [筹码内部结构评分模块 V2.0 共振-反转诊断版] 计算完毕。") #更新打印信息
        return df

    def diagnose_chip_holder_behavior_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V2.1 依赖修正版】筹码持仓者行为评分模块
        - 核心修正: 移除了对 'cost_divergence_D' 的内部计算逻辑，因为它已被前置到
                      主控函数 `run_chip_intelligence_command` 中进行预处理。
                      这使得本模块的职责更纯粹，只负责评分计算。
        - 核心升级: 遵循“共振/反转”对称原则，对成本结构、长线资金稳定性与投降行为进行多维交叉验证。
        - 核心逻辑:
          - 成本结构背离 -> 上升共振: 验证短期成本是否在持续拉开与长期成本的差距。
          - 长线筹码稳定性 -> 顶部反转: 监控长线获利盘的抛售意愿是否在加速。
          - 长线筹码投降 -> 底部反转: 捕捉深度套牢的长线资金从开始割肉到割肉衰竭的全过程。
        -  信号 (数值型, 对称设计):
          - SCORE_COST_STRUCTURE_BULLISH_RESONANCE_S/A/B: 成本结构看涨共振分。
          - SCORE_LONG_TERM_INSTABILITY_TOP_REVERSAL_A/B: 长线筹码不稳定顶部反转风险分。
          - SCORE_LONG_TERM_CAPITULATION_BOTTOM_REVERSAL_S/A/B: 长线筹码投降底部反转机会分。
        """
        print("        -> [持仓者行为评分模块 V2.1 依赖修正版] 启动...")
        new_scores = {}
        p = get_params_block(self.strategy, 'chip_holder_behavior_params')
        if not get_param_value(p.get('enabled'), True):
            return df
        # --- 1. 军备检查 (Arsenal Check) ---
        periods = get_param_value(p.get('dynamic_periods'), [5, 21, 55])
        # [修改] 简化军备检查，直接要求 cost_divergence_D 及其衍生列存在
        required_cols = [
            'turnover_from_winners_ratio_D', 'loser_rate_long_term_D', 'cost_divergence_D'
        ]
        # 动态添加斜率和加速度列
        for period in periods:
            required_cols.extend([f'SLOPE_{period}_cost_divergence_D'])
        required_cols.extend([
            'ACCEL_5_cost_divergence_D', 'SLOPE_5_turnover_from_winners_ratio_D',
            'SLOPE_5_loser_rate_long_term_D', 'ACCEL_5_loser_rate_long_term_D'
        ])
        # [删除] 删除了原有的 'cost_divergence_D' 内部计算逻辑
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"          -> [严重警告] 持仓者行为模块缺少关键数据列: {missing_cols}，模块已跳过！")
            return df
        # --- 2. 核心要素数值化 (归一化处理) ---
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        def normalize(series, ascending=True):
            return series.rolling(window=norm_window, min_periods=min_periods).rank(pct=True, ascending=ascending).fillna(0.5)
        # --- 3. 维度一: 成本结构上升共振 ---
        # 3.1 静态分 (Setup): 短期成本已高于长期成本
        static_cost_divergence = normalize(df['cost_divergence_D'])
        # 3.2 动态分 (Momentum): 短期成本正在加速远离长期成本
        cost_div_momentum = {p: normalize(df[f'SLOPE_{p}_cost_divergence_D']) for p in periods}
        avg_cost_div_momentum = pd.Series(np.mean(np.array([s.values for s in cost_div_momentum.values()]), axis=0), index=df.index)
        # 3.3 加速度 (Confirmation)
        cost_div_accel = normalize(df['ACCEL_5_cost_divergence_D'])
        # 3.4 融合生成S/A/B三级上升共振信号
        new_scores['SCORE_COST_STRUCTURE_BULLISH_RESONANCE_B'] = avg_cost_div_momentum.astype(np.float32)
        new_scores['SCORE_COST_STRUCTURE_BULLISH_RESONANCE_A'] = (static_cost_divergence * avg_cost_div_momentum).astype(np.float32)
        new_scores['SCORE_COST_STRUCTURE_BULLISH_RESONANCE_S'] = (new_scores['SCORE_COST_STRUCTURE_BULLISH_RESONANCE_A'] * cost_div_accel).astype(np.float32)
        # --- 4. 维度二: 长线筹码不稳定顶部反转风险 ---
        # 4.1 静态风险 (Setup): 长线获利盘换手率高
        static_instability = normalize(df['turnover_from_winners_ratio_D'])
        # 4.2 动态风险 (Trigger): 换手率仍在上升
        dynamic_instability = normalize(df['SLOPE_5_turnover_from_winners_ratio_D'])
        # 4.3 融合生成A/B两级顶部反转风险信号
        new_scores['SCORE_LONG_TERM_INSTABILITY_TOP_REVERSAL_B'] = static_instability.astype(np.float32)
        new_scores['SCORE_LONG_TERM_INSTABILITY_TOP_REVERSAL_A'] = (static_instability * dynamic_instability).astype(np.float32)
        # --- 5. 维度三: 长线筹码投降底部反转机会 ---
        # 5.1 战备 (Setup): 存在大量被套牢的长线资金
        capitulation_setup = normalize(df['loser_rate_long_term_D'])
        # 5.2 点火 (Trigger): 这些套牢盘开始加速卖出 (斜率负向加剧)
        capitulation_trigger = normalize(df['SLOPE_5_loser_rate_long_term_D'], ascending=False) # 斜率为负且越小（下降越快），得分越高
        # 5.3 确认 (Confirmation): 卖出开始减速 (加速度转正)，即投降衰竭
        capitulation_confirm = normalize(df['ACCEL_5_loser_rate_long_term_D'])
        # 5.4 融合生成S/A/B三级底部反转机会信号
        new_scores['SCORE_LONG_TERM_CAPITULATION_BOTTOM_REVERSAL_B'] = capitulation_confirm.astype(np.float32)
        new_scores['SCORE_LONG_TERM_CAPITULATION_BOTTOM_REVERSAL_A'] = (capitulation_setup * capitulation_trigger).astype(np.float32)
        new_scores['SCORE_LONG_TERM_CAPITULATION_BOTTOM_REVERSAL_S'] = (new_scores['SCORE_LONG_TERM_CAPITULATION_BOTTOM_REVERSAL_A'] * capitulation_confirm).astype(np.float32)
        # --- 6. 一次性将所有新得分合并到DataFrame ---
        df = df.assign(**new_scores)
        print("        -> [持仓者行为评分模块 V2.1 依赖修正版] 计算完毕。")
        return df

    def diagnose_fused_behavioral_chip_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.0   & 逻辑迁移】行为与筹码融合评分模块
        - 核心职责: 承接原 CognitiveIntelligence 中的部分计算逻辑，遵循分层架构原则。
                      融合基础的价格行为信号与筹码动态信号，生成更高质量的、描述特定
                      战术场景（如“洗盘吸筹”、“诱多派发”）的原子分数。
        - 收益: 净化了认知层的职责，使其专注于纯粹的“元融合”。
        """
        print("        -> [行为-筹码融合评分模块 V1.0] 启动...")
        new_scores = {}
        p = get_params_block(self.strategy, 'fused_behavioral_chip_params', {})
        if not get_param_value(p.get('enabled'), True):
            return df
        # --- 1. 军备检查 ---
        required_cols = [
            'pct_change_D', 'turnover_from_losers_ratio_D',
            'turnover_from_winners_ratio_D', 'SLOPE_5_concentration_90pct_D'
        ]
        required_scores = ['SCORE_CHIP_TOP_REVERSAL_A']
        missing_cols = [c for c in required_cols if c not in df.columns]
        missing_scores = [s for s in required_scores if s not in self.strategy.atomic_states]
        if missing_cols or missing_scores:
            print(f"          -> [警告] 行为-筹码融合模块缺少关键数据: 列{missing_cols}, 分数{missing_scores}。模块已跳过。")
            return df
        # --- 2. 定义归一化辅助函数 ---
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        def normalize(series, ascending=True):
            return series.rolling(window=norm_window, min_periods=min_periods).rank(pct=True, ascending=ascending).fillna(0.5)
        # --- 3. 计算“洗盘吸筹”融合分 (Washout Absorption Score) ---
        # 条件1: 价格下跌得分 (跌幅在-2%到-7%之间得分最高)
        drop_score = (1 - (df['pct_change_D'] - (-0.045)).abs() / 0.025).clip(0, 1)
        # 条件2: 套牢盘割肉得分
        losers_capitulating_score = normalize(df['turnover_from_losers_ratio_D'], ascending=True)
        # 条件3: 获利盘锁仓得分
        winners_holding_score = normalize(df['turnover_from_winners_ratio_D'], ascending=False)
        # 条件4: 筹码结构改善得分 (斜率为负代表集中，所以ascending=False)
        chip_improving_score = normalize(df['SLOPE_5_concentration_90pct_D'], ascending=False)
        washout_absorption_score = (
            drop_score * losers_capitulating_score * winners_holding_score * chip_improving_score
        )
        new_scores['CHIP_SCORE_FUSED_WASHOUT_ABSORPTION'] = washout_absorption_score.astype(np.float32)
        # --- 4. 计算“诱多派发”融合分 (Deceptive Rally Score) ---
        # 条件1: 价格拉升得分
        rally_score = normalize(df['pct_change_D'], ascending=True)
        # 条件2: 筹码顶部反转风险分 (消费纯筹码信号)
        chip_top_reversal_score = self.strategy.atomic_states.get('SCORE_CHIP_TOP_REVERSAL_A', pd.Series(0.5, index=df.index))
        deceptive_rally_score = rally_score * chip_top_reversal_score
        new_scores['CHIP_SCORE_FUSED_DECEPTIVE_RALLY'] = deceptive_rally_score.astype(np.float32)
        # --- 5. 更新DataFrame ---
        df = df.assign(**new_scores)
        print("        -> [行为-筹码融合评分模块 V1.0] 计算完毕。")
        return df

    def _calculate_normalized_score(self, series: pd.Series, window: int, ascending: bool = True) -> pd.Series:
        """
        【V2.0 性能优化版】计算滚动归一化得分的辅助函数。
        - 核心: 使用滚动分位数排名，将一个指标转换为0-1之间的得分。
        - 优化: 直接利用rank函数的`ascending`参数，移除if/else分支和额外的减法运算，代码更简洁高效。
        :param series: 原始数据Series。
        :param window: 滚动窗口大小。
        :param ascending: 排序方向。True表示值越大得分越高，False反之。
        :return: 归一化后的得分Series。
        """
        # min_periods 设为窗口的20%，与项目中其他模块保持一致。
        return series.rolling(
            window=window, 
            min_periods=max(1, window // 5) # 确保 min_periods 至少为1
        ).rank(
            pct=True, 
            ascending=ascending
        ).fillna(0.5)

    def _max_of_series(self, *series: pd.Series) -> pd.Series:
        """
        【V1.0 高性能版】计算多个pandas Series的元素级最大值。
        - 核心优化: 使用 np.maximum.reduce，它在NumPy数组上直接操作，
          比 pd.concat([...]).max(axis=1) 更快，因为它避免了创建中间DataFrame的开销。
        :param series: 一个或多个pandas Series。
        :return: 一个新的Series，其每个元素是输入Series对应位置元素的最大值。
        """
        # 过滤掉所有为None的Series
        valid_series = [s for s in series if s is not None]
        if not valid_series:
            # 如果没有有效的Series，返回一个空的Series
            return pd.Series(dtype=np.float64)
        # 将所有Series的值提取为NumPy数组列表
        arrays = [s.values for s in valid_series]
        # 使用np.maximum.reduce高效计算元素级最大值
        max_values = np.maximum.reduce(arrays)
        # 将结果包装回一个带有原始索引的pandas Series
        return pd.Series(max_values, index=valid_series[0].index)


