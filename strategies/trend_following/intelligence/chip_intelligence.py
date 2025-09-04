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
        【V325.0 配置驱动最终版】筹码情报最高司令部
        - 核心重构:
          1. 【配置即代码】: 重构了 'bucket_upper' 信号的配置方式。现在，分级信号的每个等级都使用其【最终的、完整的信号名】作为配置字典的键，彻底消除了代码中的任何字符串拼接或特殊命名逻辑。
          2. 【逻辑纯化】: 信号生成循环的逻辑变得极其纯粹，仅负责从配置中读取信号名和阈值并执行计算，不再关心信号名的构造方式。
          3. 【极致可维护性】: 任何信号（包括复杂的分级信号）的命名、修改或增删，现在都100%在 MASTER_SIGNAL_CONFIG 中完成，代码逻辑保持绝对稳定。
        """
        print("        -> [筹码情报最高司令部 V325.0 配置驱动最终版] 启动...")
        states = {}
        triggers = {}
        # --- 步骤 1: 调用评分中心 ---
        df = self.diagnose_quantitative_chip_scores(df)
        p = get_params_block(self.strategy, 'chip_feature_params')
        if not get_param_value(p.get('enabled'), False):
            return states, triggers
        # --- 步骤 2: 定义主信号配置字典 ---
        # [修改] 'bucket_upper' 类型的配置方式被重构，以信号全称为键
        MASTER_SIGNAL_CONFIG = {
            # --- 司令部顶层信号 ---
            'CONTEXT_CHIP_STRATEGIC_GATHERING': ('CHIP_SCORE_CONTEXT_STRATEGIC_GATHERING', 0.60, 120, 'state'),
            'CONTEXT_CHIP_STRATEGIC_DISTRIBUTION': ('CHIP_SCORE_CONTEXT_STRATEGIC_GATHERING', 0.40, 120, 'state_lt'),
            'CONTEXT_EUPHORIC_RALLY_WARNING': ('CHIP_SCORE_CONTEXT_EUPHORIC_RALLY', 0.90, 120, 'state'),
            'TRIGGER_CHIP_IGNITION': ('CHIP_SCORE_TRIGGER_IGNITION', 0.98, 120, 'trigger'),
            'RISK_CONTEXT_LONG_TERM_DISTRIBUTION': ('CHIP_SCORE_RISK_LONG_TERM_DISTRIBUTION', 0.90, 120, 'state'),
            'RISK_CHIP_CONC_ACCEL_WORSENING': ('CHIP_SCORE_RISK_WORSENING_TURN', 0.80, 60, 'state_gt_zero'),
            # ... (其他信号配置保持不变，此处省略) ...
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
        # --- 步骤 3: 按 (评分列, 窗口, 信号大类) 对信号配置进行分组 ---
        from collections import defaultdict
        grouped_signals = defaultdict(list)
        for signal_name, (score_col, quantile_or_dict, window, signal_type) in MASTER_SIGNAL_CONFIG.items():
            if score_col in available_cols:
                group_key = 'gt_zero' if 'gt_zero' in signal_type else 'bucket_upper' if signal_type == 'bucket_upper' else 'standard'
                grouped_signals[(score_col, window, group_key)].append((signal_name, quantile_or_dict, signal_type))
        # --- 步骤 4: 批处理所有信号 ---
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
            # [修改] 逻辑分发，bucket_upper 的处理逻辑更简洁
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


