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
        【V325.3 最终版】筹码情报最高司令部
        - 核心升级: 引入了 diagnose_chip_holder_behavior_scores 模块，增加了对长短期持仓者
          行为差异的诊断，完成了筹码分析的最后一块拼图。
        - 架构优化: 确保所有新生成的数值型评分都被更新到 self.strategy.atomic_states，
          使其可供其他上层模块消费。
        """
        print("        -> [筹码情报最高司令部 V325.3 最终版] 启动...") # [修改] 更新版本号和打印信息，反映功能升级
        states = {}
        triggers = {}
        # --- 步骤 1: 调用四级评分中心，生成所有数值型评分 ---
        # 记录调用前的列名，用于后续识别新增的评分列
        initial_cols = set(df.columns)
        # 1.1 调用宏观共振/反转诊断模块
        df = self.diagnose_quantitative_chip_scores(df)
        # 1.2 调用高级动态诊断模块 (微观结构与极端行为)
        df = self.diagnose_advanced_chip_dynamics_scores(df)
        # 1.3 调用内部结构诊断模块
        df = self.diagnose_chip_internal_structure_scores(df)
        # 1.4 [修改] 调用新增的持仓者行为诊断模块
        df = self.diagnose_chip_holder_behavior_scores(df)
        # --- 步骤 1.5: 将所有新生成的数值评分更新到原子状态库 ---
        # 这是关键一步，确保下游模块可以访问到这些数值信号
        final_cols = set(df.columns)
        new_score_cols = final_cols - initial_cols
        if new_score_cols:
            new_scores_dict = {col: df[col] for col in new_score_cols}
            self.strategy.atomic_states.update(new_scores_dict)
            print(f"          -> [情报更新] {len(new_score_cols)}个新的数值型筹码评分已更新至原子状态库。")
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
            'BUCKET_RISK_BEHAVIOR_WINNERS_FLEEING': ('CHIP_SCORE_RISK_FLEEING_IN_HIGH_ZONE', {
                'RISK_BEHAVIOR_WINNERS_FLEEING_C': 0.60,
                'RISK_BEHAVIOR_WINNERS_FLEEING_B': 0.75,
                'RISK_BEHAVIOR_WINNERS_FLEEING_A': 0.90
            }, 120, 'bucket_upper'),
        }
        available_cols = set(df.columns)
        all_generated_states = {}
        # --- 步骤 3: 按 (评分列, 窗口, 信号大类) 对信号配置进行分组，以优化计算 ---
        from collections import defaultdict
        grouped_signals = defaultdict(list)
        for signal_name, (score_col, quantile_or_dict, window, signal_type) in MASTER_SIGNAL_CONFIG.items():
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
                    thresholds_df = score.rolling(window).quantile(quantiles_needed)
            elif group_key == 'gt_zero':
                positive_scores = score[score > 0]
                if not positive_scores.empty:
                    thresholds_df = positive_scores.rolling(window).quantile(quantiles_needed).reindex(score.index).ffill()
            if thresholds_df is None: continue
            if isinstance(thresholds_df, pd.Series):
                thresholds_df = thresholds_df.to_frame(name=quantiles_needed[0])
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

    def diagnose_quantitative_chip_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V4.0 共振-反转对称诊断版】筹码信号量化评分诊断模块
        - 核心重构: 遵循“共振/反转”对称原则，将旧版零散的评分体系，全面升级为基于多维交叉验证的、结构化的四象限信号矩阵。
        - 核心逻辑:
          - 共振信号 (多周期交叉): 评估筹码集中度、成本重心在多时间周期上的一致性。
          - 反转信号 (同周期交叉): 评估“静态战备(Setup)”与“动态点火(Trigger)”的结合。
        - 新增信号 (数值型, 对称设计):
          - SCORE_CHIP_BULLISH_RESONANCE_S/A/B: 上升共振机会分 (多周期筹码集中)。
          - SCORE_CHIP_BEARISH_RESONANCE_S/A/B: 下跌共振风险分 (多周期筹码发散)。
          - SCORE_CHIP_BOTTOM_REVERSAL_S/A/B: 底部反转机会分 (下跌衰竭后吸筹)。
          - SCORE_CHIP_TOP_REVERSAL_S/A/B: 顶部反转风险分 (上涨衰竭后派发)。
        """
        print("        -> [筹码信号量化评分模块 V4.0 共振-反转对称诊断版] 启动...") # [修改] 更新版本号和打印信息
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
        for p in periods:
            required_cols.extend([
                f'SLOPE_{p}_concentration_90pct_D', f'SLOPE_{p}_peak_cost_D',
                f'ACCEL_{p if p > 5 else 5}_concentration_90pct_D',
                f'ACCEL_{p if p > 5 else 5}_peak_cost_D',
                f'ACCEL_{p if p > 5 else 5}_turnover_from_winners_ratio_D'
            ])
        # --- 2. 检查必需列 ---
        # [修改] 将 ACCEL_21_peak_cost_D 替换为军械库中的实际名称 peak_cost_accel_21d_D
        required_cols_mapping = {
            'ACCEL_21_peak_cost_D': 'peak_cost_accel_21d_D'
        }
        # 动态替换列名以匹配军械库
        for i, col in enumerate(required_cols):
            if col in required_cols_mapping:
                required_cols[i] = required_cols_mapping[col]
                # 同时，为了后续逻辑兼容，将df中的列名也统一过来
                if required_cols_mapping[col] in df.columns:
                    df[col] = df[required_cols_mapping[col]]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"          -> [严重警告] 筹码评分引擎缺少关键数据列: {missing_cols}，模块已跳过！")
            return df
        # --- 3. 核心要素数值化 (归一化处理) ---
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
        # --- 4. [新增] 共振信号合成 (多时间周期交叉验证) ---
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

        # --- 5. [新增] 反转信号合成 (静态战备 x 动态点火) ---
        # 5.1 底部反转 (环境恶劣 -> 动态改善)
        bottom_setup_score = (1 - static_concentration) * (1 - static_price_deviation) # 战备: 筹码曾发散 + 价格处于低位
        bottom_trigger_score = conc_momentum_scores[5] * cost_accel_scores[5] # 点火: 短期开始集中 + 成本加速抬升
        new_scores['SCORE_CHIP_BOTTOM_REVERSAL_B'] = bottom_trigger_score.astype(np.float32)
        new_scores['SCORE_CHIP_BOTTOM_REVERSAL_A'] = (bottom_setup_score * bottom_trigger_score).astype(np.float32)
        new_scores['SCORE_CHIP_BOTTOM_REVERSAL_S'] = (new_scores['SCORE_CHIP_BOTTOM_REversal_A'] * conc_accel_scores[5]).astype(np.float32)

        # 5.2 顶部反转 (环境良好 -> 动态恶化) - 对称逻辑
        top_setup_score = static_concentration * static_price_deviation * static_high_winner_rate # 战备: 筹码集中 + 价格高位 + 获利盘丰厚
        top_trigger_score = (1 - conc_momentum_scores[5]) * profit_taking_accel_score # 点火: 短期开始发散 + 获利盘兑现加速
        new_scores['SCORE_CHIP_TOP_REVERSAL_B'] = top_trigger_score.astype(np.float32)
        new_scores['SCORE_CHIP_TOP_REVERSAL_A'] = (top_setup_score * top_trigger_score).astype(np.float32)
        new_scores['SCORE_CHIP_TOP_REVERSAL_S'] = (new_scores['SCORE_CHIP_TOP_REVERSAL_A'] * (1 - conc_accel_scores[5])).astype(np.float32)

        # --- 6. 一次性将所有新得分合并到DataFrame ---
        df = df.assign(**new_scores)
        print("        -> [筹码信号量化评分模块 V4.0 共振-反转对称诊断版] 计算完毕。") # [修改] 更新打印信息
        return df

    def diagnose_advanced_chip_dynamics_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.0 微观结构与极端行为诊断版】高级筹码动态评分模块
        - 核心职责: 从军械库中提炼关于筹码微观结构、极端情绪和突变事件的原子信号，作为对宏观信号的补充和确认。
        - 核心逻辑:
          - 结构健康度: 融合“控盘度”、“纯粹度”、“稳定性”来量化筹码结构的质量。
          - 恐慌与承接: 捕捉“亏损盘割肉”从加速到衰竭的过程，识别潜在的底部吸收行为。
          - 断层风险: 评估“筹码断层”形成后的潜在破位风险，结合断层强度与下方真空区大小。
        - 新增信号 (数值型):
          - SCORE_CHIP_STRUCTURE_HEALTH: 结构健康总分。
          - SCORE_CHIP_STRUCTURE_IMPROVING / DETERIORATING: 结构健康趋势分。
          - SCORE_CHIP_CAPITULATION_ABSORPTION_S/A/B: 恐慌盘出尽后的承接机会分。
          - SCORE_RISK_CHIP_FAULT_BREAKDOWN_S/A/B: 筹码断层破位风险分。
        """
        print("        -> [高级筹码动态评分模块 V1.0] 启动...") # [新增] 新模块的打印信息
        new_scores = {}
        p = get_params_block(self.strategy, 'advanced_chip_dynamics_params')
        if not get_param_value(p.get('enabled'), True):
            return df

        # --- 1. 军备检查 (Arsenal Check) ---
        required_cols = [
            'peak_control_ratio_D', 'peak_strength_ratio_D', 'peak_stability_D',
            'turnover_from_losers_ratio_D', 'ACCEL_5_turnover_from_losers_ratio_D',
            'is_chip_fault_formed_D', 'chip_fault_strength_D', 'chip_fault_vacuum_percent_D'
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"          -> [严重警告] 高级筹码动态模块缺少关键数据列: {missing_cols}，模块已跳过！")
            return df

        # --- 2. 核心要素数值化 (归一化处理) ---
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        def normalize(series, ascending=True):
            return series.rolling(window=norm_window, min_periods=min_periods).rank(pct=True, ascending=ascending).fillna(0.5)

        # --- 3. [新增] 维度一: 结构健康度评分 ---
        control_score = normalize(df['peak_control_ratio_D'])
        strength_score = normalize(df['peak_strength_ratio_D'])
        stability_score = normalize(df['peak_stability_D'])
        
        # 3.1 结构健康总分
        health_score = (control_score * strength_score * stability_score)
        new_scores['SCORE_CHIP_STRUCTURE_HEALTH'] = health_score.astype(np.float32)
        
        # 3.2 结构健康趋势分
        health_momentum = normalize(health_score.diff(5))
        new_scores['SCORE_CHIP_STRUCTURE_IMPROVING'] = health_momentum.astype(np.float32)
        new_scores['SCORE_CHIP_STRUCTURE_DETERIORATING'] = (1 - health_momentum).astype(np.float32)

        # --- 4. [新增] 维度二: 恐慌盘承接机会分 (底部反转信号) ---
        # 4.1 战备 (Setup): 市场出现恐慌性抛售
        panic_selling_score = normalize(df['turnover_from_losers_ratio_D'])
        
        # 4.2 点火 (Trigger): 抛售行为开始减速 (加速度转正)，表明有资金在承接
        absorption_trigger_score = normalize(df['ACCEL_5_turnover_from_losers_ratio_D'])
        
        # 4.3 融合生成S/A/B三级信号
        new_scores['SCORE_CHIP_CAPITULATION_ABSORPTION_B'] = absorption_trigger_score.astype(np.float32)
        new_scores['SCORE_CHIP_CAPITULATION_ABSORPTION_A'] = (panic_selling_score.shift(1) * absorption_trigger_score).astype(np.float32)
        # S级需要与宏观筹码改善共振 (消费上游信号)
        macro_reversal_confirm = self.strategy.atomic_states.get('SCORE_CHIP_BOTTOM_REVERSAL_B', pd.Series(0.0, index=df.index))
        new_scores['SCORE_CHIP_CAPITULATION_ABSORPTION_S'] = (new_scores['SCORE_CHIP_CAPITULATION_ABSORPTION_A'] * macro_reversal_confirm).astype(np.float32)

        # --- 5. [新增] 维度三: 筹码断层破位风险分 (顶部反转/下跌共振信号) ---
        is_fault_formed = df['is_chip_fault_formed_D'].astype(float)
        
        # 5.1 风险组件
        fault_strength_score = normalize(df['chip_fault_strength_D'])
        vacuum_risk_score = normalize(df['chip_fault_vacuum_percent_D']) # 真空区越大，风险越高
        
        # 5.2 融合生成S/A/B三级风险信号
        base_risk = is_fault_formed
        new_scores['SCORE_RISK_CHIP_FAULT_BREAKDOWN_B'] = base_risk.astype(np.float32)
        new_scores['SCORE_RISK_CHIP_FAULT_BREAKDOWN_A'] = (base_risk * fault_strength_score).astype(np.float32)
        new_scores['SCORE_RISK_CHIP_FAULT_BREAKDOWN_S'] = (new_scores['SCORE_RISK_CHIP_FAULT_BREAKDOWN_A'] * vacuum_risk_score).astype(np.float32)

        # --- 6. 一次性将所有新得分合并到DataFrame ---
        df = df.assign(**new_scores)
        print("        -> [高级筹码动态评分模块 V1.0] 计算完毕。") # [新增] 新模块的打印信息
        return df

    def diagnose_chip_internal_structure_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.0 内部结构与状态诊断版】筹码内部结构评分模块
        - 核心职责: 深入剖析筹码分布的内部构成，评估其稳定性和潜在压力，生成更精细的原子信号。
        - 核心逻辑:
          - 核心/外围持仓者动态: 对比70%与90%集中度的变化，判断筹码集中的“质量”。
          - 获利盘成分分析: 区分长/短期获利盘比例，评估当前持仓结构的稳定性。
          - 利润兑现压力: 直接量化获利盘的平均利润率及其变化，预警潜在抛压。
          - 结构支撑与压力: 对比上下方的筹码密集区，判断价格运动的“最小阻力方向”。
        - 新增信号 (数值型):
          - SCORE_CHIP_CORE_HOLDER_STRENGTH: 核心持仓者（铁票）实力分。
          - SCORE_CHIP_WINNER_COMPOSITION_STABILITY: 获利盘成分稳定分。
          - SCORE_RISK_PROFIT_TAKING_PRESSURE_A/B: 获利盘兑现压力风险分。
          - SCORE_CHIP_PATH_OF_LEAST_RESISTANCE: 最小阻力路径机会分。
        """
        print("        -> [筹码内部结构评分模块 V1.0] 启动...") # [新增] 新模块的打印信息
        new_scores = {}
        p = get_params_block(self.strategy, 'chip_internal_structure_params')
        if not get_param_value(p.get('enabled'), True):
            return df

        # --- 1. 军备检查 (Arsenal Check) ---
        required_cols = [
            'concentration_70pct_D', 'concentration_90pct_D', 'winner_rate_short_term_D',
            'winner_rate_long_term_D', 'winner_profit_margin_D', 'SLOPE_5_winner_profit_margin_D',
            'support_below_D', 'pressure_above_D'
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"          -> [严重警告] 筹码内部结构模块缺少关键数据列: {missing_cols}，模块已跳过！")
            return df

        # --- 2. 核心要素数值化 (归一化处理) ---
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        def normalize(series, ascending=True):
            return series.rolling(window=norm_window, min_periods=min_periods).rank(pct=True, ascending=ascending).fillna(0.5)

        # --- 3. [新增] 维度一: 核心持仓者实力分 ---
        # 衡量核心筹码(70%)相对于外围筹码(90%)的集中速度。
        # 如果核心筹码集中速度更快，说明是“真”集中，得分高。
        conc_70_norm = normalize(df['concentration_70pct_D'])
        conc_90_norm = normalize(df['concentration_90pct_D'])
        core_strength_score = normalize(conc_70_norm.diff(5) - conc_90_norm.diff(5))
        new_scores['SCORE_CHIP_CORE_HOLDER_STRENGTH'] = core_strength_score.astype(np.float32)

        # --- 4. [新增] 维度二: 获利盘成分稳定分 ---
        # 衡量获利盘中，长期持有者占比。占比越高，结构越稳定，得分越高。
        # 使用归一化后的值计算，避免量纲问题。
        short_term_winner_norm = normalize(df['winner_rate_short_term_D'])
        long_term_winner_norm = normalize(df['winner_rate_long_term_D'])
        # 稳定分 = 长期获利盘占比 / (长期 + 短期 + 极小值避免除零)
        stability_score = long_term_winner_norm / (long_term_winner_norm + short_term_winner_norm + 1e-6)
        new_scores['SCORE_CHIP_WINNER_COMPOSITION_STABILITY'] = stability_score.fillna(0.5).astype(np.float32)

        # --- 5. [新增] 维度三: 获利盘兑现压力风险分 ---
        # B级风险：静态获利盘丰厚
        profit_margin_score = normalize(df['winner_profit_margin_D'])
        new_scores['SCORE_RISK_PROFIT_TAKING_PRESSURE_B'] = profit_margin_score.astype(np.float32)
        # A级风险：获利盘仍在快速增长，加速兑现冲动
        profit_margin_momentum = normalize(df['SLOPE_5_winner_profit_margin_D'])
        new_scores['SCORE_RISK_PROFIT_TAKING_PRESSURE_A'] = (profit_margin_score * profit_margin_momentum).astype(np.float32)

        # --- 6. [新增] 维度四: 最小阻力路径机会分 ---
        # 对比下方支撑和上方压力的强度。支撑远大于压力时，上涨阻力小，得分高。
        support_norm = normalize(df['support_below_D'])
        pressure_norm = normalize(df['pressure_above_D'])
        # 使用差值来衡量净支撑强度
        path_score = support_norm - pressure_norm
        new_scores['SCORE_CHIP_PATH_OF_LEAST_RESISTANCE'] = normalize(path_score).astype(np.float32)

        # --- 7. 一次性将所有新得分合并到DataFrame ---
        df = df.assign(**new_scores)
        print("        -> [筹码内部结构评分模块 V1.0] 计算完毕。") # [新增] 新模块的打印信息
        return df

    def diagnose_chip_holder_behavior_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.0 持仓者行为诊断版】筹码持仓者行为评分模块
        - 核心职责: 区分长、短期持仓者的行为差异，判断市场主导力量和关键情绪拐点。
        - 核心逻辑:
          - 成本结构背离: 比较长、短期持仓成本线的关系。短线成本上穿长线成本是牛市确认信号。
          - 长线筹码稳定性: 监控长线获利盘的抛售意愿，其稳定是牛市根基。
          - 长线筹码投降: 捕捉被深度套牢的长线资金开始割肉的现象，这是潜在的史诗级底部信号。
        - 新增信号 (数值型):
          - SCORE_CHIP_COST_STRUCTURE_DIVERGENCE: 成本结构背离分 (高分看涨)。
          - SCORE_RISK_LONG_TERM_HOLDER_INSTABILITY: 长线持有者不稳定风险分。
          - SCORE_OPP_LONG_TERM_CAPITULATION: 长线持有者投降机会分。
        """
        print("        -> [持仓者行为评分模块 V1.0] 启动...") # [新增] 新模块的打印信息
        new_scores = {}
        p = get_params_block(self.strategy, 'chip_holder_behavior_params')
        if not get_param_value(p.get('enabled'), True):
            return df

        # --- 1. 军备检查 (Arsenal Check) ---
        required_cols = [
            'avg_cost_short_term_D', 'avg_cost_long_term_D', 'winner_turnover_long_term_D',
            'loser_rate_long_term_D', 'SLOPE_5_loser_rate_long_term_D'
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"          -> [严重警告] 持仓者行为模块缺少关键数据列: {missing_cols}，模块已跳过！")
            return df

        # --- 2. 核心要素数值化 (归一化处理) ---
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        def normalize(series, ascending=True):
            return series.rolling(window=norm_window, min_periods=min_periods).rank(pct=True, ascending=ascending).fillna(0.5)

        # --- 3. [新增] 维度一: 成本结构背离分 ---
        # 衡量短期成本与长期成本的差值。差值越大（短期成本远高于长期成本），说明市场追高意愿强，趋势健康。
        cost_divergence = df['avg_cost_short_term_D'] - df['avg_cost_long_term_D']
        cost_divergence_score = normalize(cost_divergence)
        new_scores['SCORE_CHIP_COST_STRUCTURE_DIVERGENCE'] = cost_divergence_score.astype(np.float32)

        # --- 4. [新增] 维度二: 长线持有者不稳定风险分 ---
        # 直接衡量“长线获利盘”的换手率。该值越高，说明被认为是“聪明钱”的长线资金正在离场，风险越高。
        instability_score = normalize(df['winner_turnover_long_term_D'])
        new_scores['SCORE_RISK_LONG_TERM_HOLDER_INSTABILITY'] = instability_score.astype(np.float32)

        # --- 5. [新增] 维度三: 长线持有者投降机会分 ---
        # 这是一个经典的底部信号：寻找那些被深度套牢的长线资金开始绝望割肉（投降）的时刻。
        # 战备 (Setup): 市场中存在大量被套牢的长线资金。
        capitulation_setup = normalize(df['loser_rate_long_term_D'])
        # 点火 (Trigger): 这些长线套牢盘开始加速卖出，但随后卖出动能衰竭（斜率从负最大值开始回升）。
        # 我们简化为捕捉“套牢盘比例开始显著下降”的时刻。
        capitulation_trigger = normalize(df['SLOPE_5_loser_rate_long_term_D'], ascending=False) # 斜率为负且越小（下降越快），得分越高
        
        # 融合信号：当“战备”和“点火”同时满足时，信号最强。
        capitulation_score = capitulation_setup * capitulation_trigger
        new_scores['SCORE_OPP_LONG_TERM_CAPITULATION'] = capitulation_score.astype(np.float32)

        # --- 6. 一次性将所有新得分合并到DataFrame ---
        df = df.assign(**new_scores)
        print("        -> [持仓者行为评分模块 V1.0] 计算完毕。") # [新增] 新模块的打印信息
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


