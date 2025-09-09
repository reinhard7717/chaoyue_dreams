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
        【V328.0 交叉验证版】筹码情报最高司令部
        - 核心升级: 全面数值化改造。将原先生成布尔型(True/False)的 state 和 trigger 信号，
                      升级为生成浮点型的数值信号。新信号的值代表了分数(score)与动态阈值(threshold)的差值，
                      正值表示信号成立，负值表示不成立，其大小则代表了信号的强度。
        - 核心升级: 对分桶信号(bucket)进行数值化。将原先为每个桶生成一个布尔信号的逻辑，
                      改为生成一个单一的“等级”信号(0, 1, 2, 3...)，数值大小代表当前分数所处的强度等级。
        - 核心升级 (V328.0): 新增 `diagnose_cross_validation_signals` 模块，基于“因子共振”和“时间共振”
                             双重交叉验证，生成最高置信度的上升/下跌共振与顶部/底部反转信号。
        - 收益: 提供了信息量更丰富、更平滑的连续信号，消除了布尔信号在阈值附近的突变，
                为下游的量化模型和决策系统提供了更高质量的输入。
        """
        print("        -> [筹码情报最高司令部 V328.0 交叉验证版] 启动...")
        states = {}
        triggers = {}
        # 定义一个辅助函数，用于在每个诊断模块后增量更新原子状态库
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
        # 步骤 0: 预处理衍生指标，解决计算依赖问题
        if 'avg_cost_short_term_D' in df.columns and 'avg_cost_long_term_D' in df.columns:
            df['cost_divergence_D'] = df['avg_cost_short_term_D'] - df['avg_cost_long_term_D']
            initial_cols.add('cost_divergence_D') # 将新列也加入初始列集合
            print("          -> [预处理] 已成功计算衍生指标 'cost_divergence_D'。")
        # --- 步骤 1: 链式调用四级评分中心，并确保状态实时传递 ---
        # 记录调用前的列名，用于后续识别新的评分列
        cols_before_run = initial_cols
        # ---  步骤 1.0: 首先调用战略上下文评分模块 ---
        df = self.diagnose_strategic_context_scores(df)
        cols_before_run = _update_atomic_states(df, cols_before_run)
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
        
        # // 修改开始：插入对新增的交叉验证模块的调用
        # 1.6 调用【新增】的交叉验证诊断模块，生成终极信号
        df = self.diagnose_cross_validation_signals(df)
        cols_before_run = _update_atomic_states(df, cols_before_run)
        # // 修改结束

        # 1.7: 调用 V2.0 版本的元融合模块
        prime_opp_states, prime_opp_scores = self.synthesize_prime_chip_opportunity(df)
        states.update(prime_opp_states)
        if prime_opp_scores:
            df = df.assign(**prime_opp_scores)
            cols_before_run = _update_atomic_states(df, cols_before_run) # 确保黄金机会分数也被更新
        # 1.8: 调用复合评分模块，计算顶层信号依赖的原子分
        df = self.diagnose_composite_scores(df)
        cols_before_run = _update_atomic_states(df, cols_before_run)
        # 获取模块参数，检查是否启用
        p = get_params_block(self.strategy, 'chip_feature_params')
        if not get_param_value(p.get('enabled'), False):
            return states, triggers
        # --- 步骤 2: 定义主信号配置字典 (将数值评分转化为新的数值化信号)
        # 全面更新信号配置，将布尔信号名改为数值信号名(SIGNAL_*)，并调整分桶信号的配置结构
        # // 修改开始：更新MASTER_SIGNAL_CONFIG以使用新的交叉验证信号
        MASTER_SIGNAL_CONFIG = {
            # --- 司令部顶层信号 (基于交叉验证，置信度最高) ---
            'SIGNAL_RISING_RESONANCE': ('SCORE_RISING_RESONANCE_S', 0.85, 120, 'state'),
            'SIGNAL_FALLING_RESONANCE': ('SCORE_FALLING_RESONANCE_S', 0.85, 120, 'state'),
            'SIGNAL_TRIGGER_BOTTOM_REVERSAL': ('SCORE_BOTTOM_REVERSAL_S', 0.90, 120, 'trigger'),
            'SIGNAL_TRIGGER_TOP_REVERSAL': ('SCORE_TOP_REVERSAL_S', 0.90, 120, 'trigger'),

            # --- 战术级信号 (基于单一诊断模块，用于辅助判断) ---
            'SIGNAL_CONTEXT_STRATEGIC_GATHERING': ('CHIP_SCORE_CONTEXT_STRATEGIC_GATHERING', 0.60, 120, 'state'),
            'SIGNAL_CONTEXT_STRATEGIC_DISTRIBUTION': ('CHIP_SCORE_CONTEXT_STRATEGIC_GATHERING', 0.40, 120, 'state_lt'),
            'SIGNAL_CONTEXT_EUPHORIC_RALLY_WARNING': ('CHIP_SCORE_CONTEXT_EUPHORIC_RALLY', 0.90, 120, 'state'),
            'SIGNAL_RISK_LONG_TERM_DISTRIBUTION': ('CHIP_SCORE_RISK_LONG_TERM_DISTRIBUTION', 0.90, 120, 'state'),
            'SIGNAL_RISK_CONC_ACCEL_WORSENING': ('CHIP_SCORE_RISK_WORSENING_TURN', 0.80, 60, 'state_gt_zero'),
            'SIGNAL_OPP_BREAKTHROUGH': ('CHIP_SCORE_OPP_BREAKTHROUGH', 0.95, 120, 'state'),
            'SIGNAL_RISK_COLLAPSE': ('CHIP_SCORE_RISK_COLLAPSE', 0.95, 120, 'state'),
            'SIGNAL_OPP_INFLECTION': ('CHIP_SCORE_OPP_INFLECTION', 0.90, 120, 'state'),
            
            # --- 分级信号 (数值化) ---
            'SIGNAL_CHIP_CONC_GATHERING_LEVEL': ('CHIP_SCORE_GATHERING_INTENSITY', [0.70, 0.85, 0.95], 120, 'bucket_level'),
            'SIGNAL_CHIP_PROFIT_TAKING_INTENSITY': ('SCORE_CHIP_PROFIT_TAKING_INTENSITY', 0.90, 120, 'state'),
        }
        # // 修改结束
        available_cols = set(df.columns)
        all_generated_states = {}
        # --- 步骤 3: 按 (评分列, 窗口, 信号大类) 对信号配置进行分组，以优化计算 ---
        from collections import defaultdict
        grouped_signals = defaultdict(list)
        for signal_name, config_tuple in MASTER_SIGNAL_CONFIG.items():
            score_col, quantile_or_list, window, signal_type = config_tuple
            if score_col in available_cols:
                # 调整分组逻辑以适应新的 'bucket_level' 类型
                group_key = 'gt_zero' if 'gt_zero' in signal_type else 'bucket_level' if signal_type == 'bucket_level' else 'standard'
                grouped_signals[(score_col, window, group_key)].append((signal_name, quantile_or_list, signal_type))
            else: # 增加对缺失评分的警告
                print(f"          -> [配置警告] 信号 '{signal_name}' 依赖的评分 '{score_col}' 未计算，该信号将被跳过。")
        # --- 步骤 4: 批处理所有信号，高效生成数值化信号 ---
        for (score_col, window, group_key), tasks in grouped_signals.items():
            score = df[score_col]
            # 调整分位数提取逻辑以兼容列表和数值
            if group_key == 'bucket_level':
                quantiles_needed = sorted(list(set(tasks[0][1])))
            else:
                quantiles_needed = sorted(list(set(q for _, q, _ in tasks)))
            thresholds_df = None
            if group_key in ['standard', 'bucket_level']: # 包含 bucket_level
                if not score.isnull().all():
                    thresholds_list = []
                    for q_val in quantiles_needed:
                        s = score.rolling(window).quantile(q_val)
                        s.name = q_val
                        thresholds_list.append(s)
                    if thresholds_list:
                        thresholds_df = pd.concat(thresholds_list, axis=1)
            elif group_key == 'gt_zero':
                positive_scores = score[score > 0]
                if not positive_scores.empty:
                    thresholds_list = []
                    for q_val in quantiles_needed:
                        s = positive_scores.rolling(window).quantile(q_val)
                        s.name = q_val
                        thresholds_list.append(s)
                    if thresholds_list:
                        thresholds_df = pd.concat(thresholds_list, axis=1).reindex(score.index).ffill()
            if thresholds_df is None: continue
            # 核心逻辑重构：从生成布尔信号改为生成数值信号
            if group_key == 'bucket_level':
                signal_name, quantiles, signal_type = tasks[0] # 一个分组只有一个 bucket_level 任务
                # 计算等级信号：每超过一个分位阈值，等级+1
                numerical_signal = pd.Series(0.0, index=df.index)
                for q_val in sorted(quantiles):
                    threshold = thresholds_df[q_val]
                    # 当分数超过阈值时，增加1.0
                    numerical_signal += (score > threshold).astype(float)
                all_generated_states[signal_name] = numerical_signal.fillna(0.0)
            else: # 处理 standard 和 gt_zero
                for signal_name, quantile, signal_type in tasks:
                    threshold = thresholds_df[quantile]
                    numerical_signal = pd.Series(0.0, index=df.index) # 默认值为0.0
                    # 计算分数与阈值的差值作为信号强度
                    if signal_type in ['state', 'trigger', 'state_gt_zero', 'state_gt_zero_event']:
                        numerical_signal = score - threshold
                    elif signal_type in ['state_le', 'state_lt']:
                        numerical_signal = threshold - score
                    
                    # 将NaN值填充为0，表示中性状态（恰好在阈值上）
                    numerical_signal = numerical_signal.fillna(0.0)
                    if 'trigger' in signal_type:
                        triggers[signal_name] = numerical_signal
                    else:
                        all_generated_states[signal_name] = numerical_signal
        # --- 步骤 5: 最终状态更新 ---
        # 将本模块生成的数值信号更新到主状态字典和原子状态库中
        states.update(all_generated_states)
        self.strategy.atomic_states.update(all_generated_states)
        # 将 trigger 信号也更新到原子状态库，确保所有生成物可被下游访问
        self.strategy.atomic_states.update(triggers)
        print("        -> [筹码情报最高司令部 V328.0] 数值化信号生成完毕。")
        # --- 调试探针: 打印输入与输出的最后5条数据 ---
        print("\n==================== [探针: run_chip_intelligence_command] ====================")
        print("\n--- [原料信号探针] (最后5条数据) ---")
        raw_material_cols = [config[0] for config in MASTER_SIGNAL_CONFIG.values() if config[0] in df.columns]
        raw_df_subset = df[list(set(raw_material_cols))]
        if not raw_df_subset.empty:
            print(raw_df_subset.tail(5).to_string())
        else:
            print("  - (无关键原料信号可供显示)")
        print("\n--- [产出信号探针] (最后5条数据) ---")
        output_signals = {**states, **triggers}
        if not output_signals:
            print("  - (无产出信号)")
        else:
            output_df = pd.DataFrame(output_signals)
            print(output_df.tail(5).to_string())
        print("==================== [探针结束] ====================\n")
        return states, triggers

    def diagnose_cross_validation_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.0 交叉验证版】终极共振与反转信号诊断模块
        - 核心范式: 引入“因子共振”与“时间共振”双重交叉验证，生成最高置信度的交易信号。
          - 1. 因子共振: 在同一时间周期内，对多个逻辑相关的指标（如集中度、成本、健康度）进行乘法融合，
                        要求所有因子必须同向运动，形成“因子共振”。
          - 2. 时间共振: 将“因子共振”的结论，在多个关键时间周期（1D, 5D, 21D）上再次进行乘法融合，
                        要求趋势在日、周、月级别上形成“时间共振”。
          - 3. 加速验证: 将“时间共振”的信号，与关键因子的“加速度共振”相乘，作为最高S级信号。
        - A股实战考量: 采用最严格的“乘法”融合，杜绝了“平均主义”的模糊信号，确保了信号的纯粹性和可靠性。
                        反转信号的定义基于“长短期共振的尖锐对立”，精准捕捉趋势的转折点。
        """
        print("        -> [交叉验证诊断模块 V1.0] 启动，正在生成终极共振与反转信号...")
        new_scores = {}
        p = get_params_block(self.strategy, 'cross_validation_params', {})
        if not get_param_value(p.get('enabled'), True):
            return df

        # --- 1. 参数与辅助函数定义 ---
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        periods = get_param_value(p.get('resonance_periods'), [1, 5, 21]) # 定义用于时间共振的关键周期

        def normalize(series, ascending=True):
            return series.rolling(window=norm_window, min_periods=min_periods).rank(pct=True, ascending=ascending).fillna(0.5)

        # --- 2. 定义信号因子组 (Signal Factor Groups) ---
        # 定义哪些因子组合可以构成一个逻辑自洽的“共振”
        SIGNAL_GROUPS = {
            'BULLISH': {
                'concentration_90pct_D': 'down', # 集中度斜率向下（集中）
                'peak_cost_D': 'up',             # 成本斜率向上（抬高）
                'chip_health_score_D': 'up',     # 健康分斜率向上（改善）
                'peak_control_ratio_D': 'up',    # 控盘度斜率向上（增强）
            },
            'BEARISH': {
                'concentration_90pct_D': 'up',   # 集中度斜率向上（发散）
                'peak_cost_D': 'down',           # 成本斜率向下（降低）
                'chip_health_score_D': 'down',   # 健康分斜率向下（恶化）
                'winner_profit_margin_D': 'down',# 获利盘安全垫斜率向下（派发）
            }
        }

        # --- 3. 计算每个周期的“因子共振”分数 ---
        factor_resonance_scores = {'BULLISH': {}, 'BEARISH': {}}
        accel_resonance_scores = {'BULLISH': {}, 'BEARISH': {}}

        for group_name, factors in SIGNAL_GROUPS.items():
            # 3.1 计算斜率的因子共振
            for period in periods:
                period_resonance = pd.Series(1.0, index=df.index)
                for factor, direction in factors.items():
                    col_name = f'SLOPE_{period}_{factor}'
                    if col_name in df.columns:
                        ascending = (direction == 'up')
                        period_resonance *= normalize(df[col_name], ascending=ascending)
                    else:
                        print(f"          -> [交叉验证警告] 缺失斜率数据: {col_name}")
                        period_resonance *= 0.5 # 缺失数据时给予中性惩罚
                factor_resonance_scores[group_name][period] = period_resonance
            
            # 3.2 计算加速度的因子共振 (仅使用1日周期，捕捉最即时的变化)
            accel_resonance = pd.Series(1.0, index=df.index)
            for factor, direction in factors.items():
                col_name = f'ACCEL_1_{factor}'
                if col_name in df.columns:
                    ascending = (direction == 'up')
                    accel_resonance *= normalize(df[col_name], ascending=ascending)
                else:
                    print(f"          -> [交叉验证警告] 缺失加速度数据: {col_name}")
                    accel_resonance *= 0.5
            accel_resonance_scores[group_name] = accel_resonance

        # --- 4. 计算“时间共振”分数与最终信号 ---
        # 4.1 上升共振信号 (Rising Resonance)
        bullish_scores = factor_resonance_scores['BULLISH']
        bullish_temporal_resonance = pd.Series(1.0, index=df.index)
        for period in periods:
            bullish_temporal_resonance *= bullish_scores[period]
        
        new_scores['SCORE_RISING_RESONANCE_B'] = bullish_scores[5].astype(np.float32) # B级: 短期(5日)因子共振
        new_scores['SCORE_RISING_RESONANCE_A'] = (bullish_scores[5] * bullish_scores[21]).astype(np.float32) # A级: 短中期因子共振
        new_scores['SCORE_RISING_RESONANCE_S'] = (bullish_temporal_resonance * accel_resonance_scores['BULLISH']).astype(np.float32) # S级: 全周期+加速度共振

        # 4.2 下跌共振信号 (Falling Resonance)
        bearish_scores = factor_resonance_scores['BEARISH']
        bearish_temporal_resonance = pd.Series(1.0, index=df.index)
        for period in periods:
            bearish_temporal_resonance *= bearish_scores[period]

        new_scores['SCORE_FALLING_RESONANCE_B'] = bearish_scores[5].astype(np.float32)
        new_scores['SCORE_FALLING_RESONANCE_A'] = (bearish_scores[5] * bearish_scores[21]).astype(np.float32)
        new_scores['SCORE_FALLING_RESONANCE_S'] = (bearish_temporal_resonance * accel_resonance_scores['BEARISH']).astype(np.float32)

        # --- 5. 计算“反转”信号 ---
        # 5.1 底部反转: 长期下跌共振(惯性) + 短期上升共振(边际变化)
        long_term_bearish_inertia = bearish_scores[21]
        short_term_bullish_turn = bullish_scores[1] * bullish_scores[5]
        bottom_reversal_score = short_term_bullish_turn * long_term_bearish_inertia
        
        new_scores['SCORE_BOTTOM_REVERSAL_B'] = (bullish_scores[1]).astype(np.float32) # B级: 1日因子共振出现
        new_scores['SCORE_BOTTOM_REVERSAL_A'] = (bullish_scores[1] * bearish_scores[21]).astype(np.float32) # A级: 1日看涨 vs 21日看跌
        new_scores['SCORE_BOTTOM_REVERSAL_S'] = (bottom_reversal_score * accel_resonance_scores['BULLISH']).astype(np.float32) # S级: A级信号+看涨加速

        # 5.2 顶部反转: 长期上升共振(惯性) + 短期下跌共振(边际变化)
        long_term_bullish_inertia = bullish_scores[21]
        short_term_bearish_turn = bearish_scores[1] * bearish_scores[5]
        top_reversal_score = short_term_bearish_turn * long_term_bullish_inertia

        new_scores['SCORE_TOP_REVERSAL_B'] = (bearish_scores[1]).astype(np.float32)
        new_scores['SCORE_TOP_REVERSAL_A'] = (bearish_scores[1] * bullish_scores[21]).astype(np.float32)
        new_scores['SCORE_TOP_REVERSAL_S'] = (top_reversal_score * accel_resonance_scores['BEARISH']).astype(np.float32)

        # --- 6. 更新DataFrame ---
        df = df.assign(**new_scores)
        print(f"        -> [交叉验证诊断模块 V1.0] 计算完毕，新增 {len(new_scores)} 个终极信号。")
        # --- 调试探针: 打印输入与输出的最后5条数据 ---
        print("\n==================== [探针: diagnose_cross_validation_signals] ====================")
        print("\n--- [原料信号探针] (最后5条数据) ---")
        raw_material_cols = set()
        for group in SIGNAL_GROUPS.values():
            for factor in group.keys():
                for period in periods:
                    raw_material_cols.add(f'SLOPE_{period}_{factor}')
                raw_material_cols.add(f'ACCEL_1_{factor}')
        final_raw_cols = [col for col in sorted(list(raw_material_cols)) if col in df.columns]
        raw_df_subset = df[final_raw_cols]
        if not raw_df_subset.empty:
            print(raw_df_subset.tail(5).to_string())
        else:
            print("  - (无关键原料信号可供显示)")
        print("\n--- [产出信号探针] (最后5条数据) ---")
        if not new_scores:
            print("  - (无产出信号)")
        else:
            output_df = pd.DataFrame(new_scores)
            print(output_df.tail(5).to_string())
        print("==================== [探针结束] ====================\n")
        return df

    def diagnose_composite_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.1 逻辑增强版】复合筹码评分模块
        - 核心职责: 融合来自多个基础诊断模块的原子评分，生成更高维度的、用于最终信号决策的复合评分。
                      此模块专门计算 MASTER_SIGNAL_CONFIG 中定义的、但未在其他模块中计算的原子评分。
        - A股实战考量:
          - “点火”与“崩溃”等关键行为在A股中往往是多因素共振的结果，而非单一指标触发。
            因此，本模块的评分设计广泛采用“乘法”融合（代表“与”逻辑，要求多条件共存）
            和“取最大值”融合（代表“或”逻辑，捕捉多种可能性），避免理想化的单一模型，
            更贴近A股复杂多变的实战环境。
        - 核心升级 (V1.1): 优化 `CHIP_SCORE_RISK_WORSENING_TURN` 评分逻辑，增加“短期趋势已向下”
                             的前置条件，使信号判断更严格，更能捕捉真实的加速恶化风险。
        """
        print("        -> [复合筹码评分模块 V1.1 逻辑增强版] 启动，正在合成顶层原子评分...") 
        new_scores = {}
        atomic = self.strategy.atomic_states
        # 将默认值从0.5改为0.0，因为在乘法融合中，缺失信号应视为中性(不贡献分数)，而非平均水平。
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 定义归一化辅助函数 ---
        p = get_params_block(self.strategy, 'chip_feature_params')
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        def normalize(series, ascending=True):
            # 增加对空Series的健壮性处理
            if series.empty:
                return pd.Series(0.0, index=df.index, dtype=np.float32)
            return series.rolling(window=norm_window, min_periods=min_periods).rank(pct=True, ascending=ascending).fillna(0.5)
        # --- 2. 计算 CHIP_SCORE_GATHERING_INTENSITY (筹码集中强度分) ---
        # 逻辑: 短期内，筹码集中速度越快、成本抬升越快、健康度改善越快，则集中强度越高。
        conc_slope_score = normalize(df.get('SLOPE_5_concentration_90pct_D', default_score), ascending=False)
        cost_slope_score = normalize(df.get('SLOPE_5_peak_cost_D', default_score), ascending=True)
        health_slope_score = normalize(df.get('SLOPE_5_chip_health_score_D', default_score), ascending=True)
        new_scores['CHIP_SCORE_GATHERING_INTENSITY'] = (conc_slope_score * cost_slope_score * health_slope_score).astype(np.float32)
        # --- 3. 计算 CHIP_SCORE_TRIGGER_IGNITION (筹码点火触发分) ---
        # 逻辑: 结构健康(黄金机会) 与 上升共振(趋势确认) 同时发生，构成最强点火信号。
        prime_opp_score = atomic.get('CHIP_SCORE_PRIME_OPPORTUNITY_S', default_score)
        bullish_resonance_score = atomic.get('SCORE_CHIP_BULLISH_RESONANCE_S', default_score)
        new_scores['CHIP_SCORE_TRIGGER_IGNITION'] = (prime_opp_score * bullish_resonance_score).astype(np.float32)
        # --- 4. 计算 CHIP_SCORE_RISK_LONG_TERM_DISTRIBUTION (长线派发风险分) ---
        # 逻辑: 市场处于下跌共振状态，同时获利盘不稳定且有强烈的兑现意愿。
        bearish_resonance_score = atomic.get('SCORE_CHIP_BEARISH_RESONANCE_S', default_score)
        profit_taking_score = atomic.get('SCORE_PROFIT_TAKING_TOP_REVERSAL_S', default_score)
        instability_score = atomic.get('SCORE_LONG_TERM_INSTABILITY_TOP_REVERSAL_A', default_score)
        new_scores['CHIP_SCORE_RISK_LONG_TERM_DISTRIBUTION'] = (bearish_resonance_score * self._max_of_series(profit_taking_score, instability_score)).astype(np.float32)
        # --- 5. 计算 CHIP_SCORE_RISK_WORSENING_TURN (集中度加速恶化拐点风险分) ---
        # 逻辑增强：要求短期趋势已向下(斜率为负)，且短期和中期加速度同时为负。
        # 条件1: 短期趋势已向下
        conc_slope_5_negative_score = normalize(df.get('SLOPE_5_concentration_90pct_D', default_score), ascending=False)
        # 条件2 & 3: 短期和中期加速度为负
        conc_accel_5 = normalize(df.get('ACCEL_5_concentration_90pct_D', default_score), ascending=False)
        conc_accel_21 = normalize(df.get('ACCEL_21_concentration_90pct_D', default_score), ascending=False)
        # 融合三个条件，形成更严格的风险评分
        new_scores['CHIP_SCORE_RISK_WORSENING_TURN'] = (conc_slope_5_negative_score * conc_accel_5 * conc_accel_21).astype(np.float32)
        print(f"          -> [调试] CHIP_SCORE_RISK_WORSENING_TURN (恶化拐点分) 最后5值: {new_scores['CHIP_SCORE_RISK_WORSENING_TURN'].tail(5).to_list()}")
        # --- 6. 计算 CHIP_SCORE_OPP_BREAKTHROUGH (突破机会分) ---
        # 逻辑: 具备黄金机会的结构基础，同时价格开始向上突破关键成本区。
        price_deviation_score = normalize(df.get('price_to_peak_ratio_D', default_score), ascending=True)
        new_scores['CHIP_SCORE_OPP_BREAKTHROUGH'] = (prime_opp_score * price_deviation_score).astype(np.float32)
        # --- 7. 计算 CHIP_SCORE_RISK_COLLAPSE (崩溃风险分) ---
        # 逻辑: 多个S级顶部/看跌风险信号共振，形成完美风暴。
        top_reversal_score = atomic.get('SCORE_CHIP_TOP_REVERSAL_S', default_score)
        fault_risk_score = atomic.get('SCORE_FAULT_RISK_TOP_REVERSAL_S', default_score)
        new_scores['CHIP_SCORE_RISK_COLLAPSE'] = (self._max_of_series(bearish_resonance_score, top_reversal_score, fault_risk_score)).astype(np.float32)
        # --- 8. 计算 CHIP_SCORE_OPP_INFLECTION (拐点机会分) ---
        # 逻辑: 多个S级底部反转信号共振，形成抄底机会。
        bottom_reversal_score = atomic.get('SCORE_CHIP_BOTTOM_REVERSAL_S', default_score)
        capitulation_score = atomic.get('SCORE_CAPITULATION_BOTTOM_RESONANCE_S', default_score) # 修正笔误，使用S级信号
        long_term_capitulation_score = atomic.get('SCORE_LONG_TERM_CAPITULATION_BOTTOM_REVERSAL_S', default_score)
        new_scores['CHIP_SCORE_OPP_INFLECTION'] = (self._max_of_series(bottom_reversal_score, capitulation_score, long_term_capitulation_score)).astype(np.float32)
        # --- 9. 更新DataFrame ---
        df = df.assign(**new_scores)
        print(f"        -> [复合筹码评分模块 V1.1] 计算完毕，新增 {len(new_scores)} 个顶层原子评分。")
        # --- 调试探针: 打印输入与输出的最后5条数据 ---
        print("\n==================== [探针: diagnose_composite_scores] ====================")
        print("\n--- [原料信号探针] (最后5条数据) ---")
        raw_material_cols = [
            'SLOPE_5_concentration_90pct_D', 'SLOPE_5_peak_cost_D', 'SLOPE_5_chip_health_score_D',
            'CHIP_SCORE_PRIME_OPPORTUNITY_S', 'SCORE_CHIP_BULLISH_RESONANCE_S',
            'SCORE_CHIP_BEARISH_RESONANCE_S', 'SCORE_PROFIT_TAKING_TOP_REVERSAL_S',
            'SCORE_LONG_TERM_INSTABILITY_TOP_REVERSAL_A', 'ACCEL_5_concentration_90pct_D',
            'ACCEL_21_concentration_90pct_D', 'price_to_peak_ratio_D',
            'SCORE_CHIP_TOP_REVERSAL_S', 'SCORE_FAULT_RISK_TOP_REVERSAL_S',
            'SCORE_CHIP_BOTTOM_REVERSAL_S', 'SCORE_CAPITULATION_BOTTOM_RESONANCE_S',
            'SCORE_LONG_TERM_CAPITULATION_BOTTOM_REVERSAL_S'
        ]
        # 从atomic_states获取数据，如果不存在则从df获取
        raw_signals_dict = {}
        for col in raw_material_cols:
            if col in atomic:
                raw_signals_dict[col] = atomic[col]
            elif col in df.columns:
                raw_signals_dict[col] = df[col]
        if raw_signals_dict:
            raw_df_subset = pd.DataFrame(raw_signals_dict)
            print(raw_df_subset.tail(5).to_string())
        else:
            print("  - (无关键原料信号可供显示)")
        print("\n--- [产出信号探针] (最后5条数据) ---")
        if not new_scores:
            print("  - (无产出信号)")
        else:
            output_df = pd.DataFrame(new_scores)
            print(output_df.tail(5).to_string())
        print("==================== [探针结束] ====================\n")
        return df

    def diagnose_strategic_context_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.1 评分扩展版】战略级筹码上下文评分模块
        - 核心职责: 融合最长周期的筹码指标，生成描述宏观“战略吸筹”与“战略派发”的
                      顶层上下文分数。这是整个筹码情报模块的基石。
        - 核心升级(V1.1): 新增“亢奋上涨预警”评分，用于识别市场过热的宏观状态。
        - 收益: 为下游所有模块提供了最关键的宏观背景判断。
        """
        print("        -> [战略级筹码上下文评分模块 V1.1 评分扩展版] 启动...") # 修改: 更新打印信息
        new_scores = {}
        p = get_params_block(self.strategy, 'chip_strategic_context_params', {})
        if not get_param_value(p.get('enabled'), True):
            return df
        # --- 1. 军备检查 ---
        long_period = get_param_value(p.get('long_period'), 55)
        required_cols = [
            f'SLOPE_{long_period}_concentration_90pct_D',
            f'SLOPE_{long_period}_peak_cost_D',
            f'SLOPE_{long_period}_chip_health_score_D',
            'total_winner_rate_D',
            'price_to_peak_ratio_D'
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"          -> [严重警告] 战略级筹码上下文模块缺少关键数据列: {missing_cols}，模块已跳过！")
            return df
        # --- 2. 核心要素数值化 (归一化) ---
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        def normalize(series, ascending=True):
            return series.rolling(window=norm_window, min_periods=min_periods).rank(pct=True, ascending=ascending).fillna(0.5)
        # --- 3. 计算“战略吸筹”分数 ---
        # 逻辑: 长期筹码在集中(斜率<0) + 长期成本在抬高(斜率>0) + 长期健康度在改善(斜率>0)
        long_term_concentration_score = normalize(df[f'SLOPE_{long_period}_concentration_90pct_D'], ascending=False)
        long_term_cost_increase_score = normalize(df[f'SLOPE_{long_period}_peak_cost_D'], ascending=True)
        long_term_health_improve_score = normalize(df[f'SLOPE_{long_period}_chip_health_score_D'], ascending=True)
        strategic_gathering_score = (
            long_term_concentration_score *
            long_term_cost_increase_score *
            long_term_health_improve_score
        )
        new_scores['CHIP_SCORE_CONTEXT_STRATEGIC_GATHERING'] = strategic_gathering_score.astype(np.float32)
        # --- 4. 计算“亢奋上涨预警”分数 ---
        # 逻辑: 市场极度乐观，获利盘丰厚，且价格远高于筹码峰成本。这是一个顶部的上下文风险状态。
        high_winner_rate_score = normalize(df['total_winner_rate_D'], ascending=True)
        high_price_deviation_score = normalize(df['price_to_peak_ratio_D'], ascending=True)
        euphoric_rally_score = (
            high_winner_rate_score *
            high_price_deviation_score
        )
        new_scores['CHIP_SCORE_CONTEXT_EUPHORIC_RALLY'] = euphoric_rally_score.astype(np.float32)
        # --- 5. 更新DataFrame ---
        df = df.assign(**new_scores)
        print("        -> [战略级筹码上下文评分模块 V1.1 评分扩展版] 计算完毕。")
        # --- 调试探针: 打印输入与输出的最后5条数据 ---
        print("\n==================== [探针: diagnose_strategic_context_scores] ====================")
        print("\n--- [原料信号探针] (最后5条数据) ---")
        final_raw_cols = [col for col in required_cols if col in df.columns]
        raw_df_subset = df[final_raw_cols]
        if not raw_df_subset.empty:
            print(raw_df_subset.tail(5).to_string())
        else:
            print("  - (无关键原料信号可供显示)")
        print("\n--- [产出信号探针] (最后5条数据) ---")
        if not new_scores:
            print("  - (无产出信号)")
        else:
            output_df = pd.DataFrame(new_scores)
            print(output_df.tail(5).to_string())
        print("==================== [探针结束] ====================\n")
        return df

    def synthesize_prime_chip_opportunity(self, df: pd.DataFrame) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
        """
        【V2.1 数值化信号升级版】黄金筹码机会元融合模块
        - 核心职责: 将多个独立的、描述筹码结构健康度的S/A/B三级数值评分，通过
                      “维度内置信度加权”和“维度间重要性加权”两步，融合成一个顶层的、
                      能精细度量机会“成色”的元分数。
        - 核心升级: 将原先生成的布尔型机会信号 'CHIP_STRUCTURE_PRIME_OPPORTUNITY_S'，
                      升级为数值型信号 'SIGNAL_PRIME_CHIP_OPPORTUNITY_S'。
                      新信号的值为“元分数”与一个固定阈值的差值，直接反映机会的强度。
        - 收益: 实现了从“信号有无”到“机会质量”的升维，为决策提供了更平滑、更鲁棒的依据。
        """
        print("        -> [黄金筹码机会元融合模块 V2.1 数值化信号升级版] 启动...")
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
        # --- 5. 基于元分数，生成数值化机会信号 ---
        # 获取用于生成信号的阈值参数，原参数名 'prime_score_threshold_for_bool' 已优化
        threshold = get_param_value(p_module.get('prime_score_threshold'), 0.7)
        # 计算数值化信号：元分数 - 阈值。正值代表机会，值越大机会越强
        prime_opportunity_numerical_signal = new_scores['CHIP_SCORE_PRIME_OPPORTUNITY_S'] - threshold
        # 使用新的命名规范 SIGNAL_...，并将其添加到 states 字典中，替换原布尔信号
        states['SIGNAL_PRIME_CHIP_OPPORTUNITY_S'] = prime_opportunity_numerical_signal.astype(np.float32)
        print("        -> [黄金筹码机会元融合模块 V2.1 数值化信号升级版] 计算完毕。")
        # --- 调试探针: 打印输入与输出的最后5条数据 ---
        print("\n==================== [探针: synthesize_prime_chip_opportunity] ====================")
        print("\n--- [原料信号探针] (最后5条数据) ---")
        raw_signals_dict = {}
        for s_name in all_required_scores:
            if s_name in atomic:
                raw_signals_dict[s_name] = atomic[s_name]
        if raw_signals_dict:
            raw_df_subset = pd.DataFrame(raw_signals_dict)
            print(raw_df_subset.tail(5).to_string())
        else:
            print("  - (无关键原料信号可供显示)")
        print("\n--- [产出信号探针] (最后5条数据) ---")
        output_signals = {**states, **new_scores}
        if not output_signals:
            print("  - (无产出信号)")
        else:
            output_df = pd.DataFrame(output_signals)
            print(output_df.tail(5).to_string())
        print("==================== [探针结束] ====================\n")
        return states, new_scores

    def diagnose_quantitative_chip_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V6.0 终极共振版】筹码信号量化评分诊断模块
        - 核心重构 (V6.0): 摒弃“平均动能”，采用更严格的“多周期动能乘积”来定义共振。一个真正的趋势，必须
                          在短、中、长周期上形成一致的方向合力，而非简单的平均。此举极大提升了信号的可靠性。
        - 核心逻辑:
          - B级 (多周期共振): 融合集中度、成本、健康度三大维度在5、21、55日周期上的“斜率”乘积。
          - A级 (高质量共振): 在B级基础上，乘以静态健康分，确保共振发生在健康结构之上。
          - S级 (加速的共振): 在A级基础上，乘以三大维度的短期“加速度”，捕捉正在加速的共振趋势。
        """
        print("        -> [筹码信号量化评分模块 V6.0 终极共振版] 启动...") 
        new_scores = {}
        p = get_params_block(self.strategy, 'chip_feature_params')
        if not get_param_value(p.get('enabled'), True):
            return df
        # --- 1. 军备检查 (Arsenal Check) ---
        periods = get_param_value(p.get('dynamic_periods'), [1, 5, 21, 55])
        # 明确定义用于共振的核心周期
        resonance_periods = [p for p in periods if p in [5, 21, 55]]
        if len(resonance_periods) < 3:
            print(f"          -> [严重警告] 共振计算需要5, 21, 55周期，当前配置不足，模块已跳过！")
            return df
        required_cols = [
            'concentration_90pct_D', 'peak_cost_D', 'total_winner_rate_D',
            'turnover_from_winners_ratio_D', 'price_to_peak_ratio_D', 'chip_health_score_D'
        ]
        for period in periods: # 检查所有需要的周期
            required_cols.extend([
                f'SLOPE_{period}_concentration_90pct_D', f'SLOPE_{period}_peak_cost_D',
                f'ACCEL_{period}_concentration_90pct_D', f'ACCEL_{period}_peak_cost_D',
                f'SLOPE_{period}_chip_health_score_D',
            ])
        required_cols.extend(['ACCEL_5_turnover_from_winners_ratio_D', 'ACCEL_5_chip_health_score_D'])
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"          -> [严重警告] 筹码评分引擎缺少关键数据列: {missing_cols}，模块已跳过！")
            return df
        # --- 2. 核心要素数值化 (归一化处理) ---
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        def normalize(series, ascending=True):
            return series.rolling(window=norm_window, min_periods=min_periods).rank(pct=True, ascending=ascending).fillna(0.5)
        # --- 3. 静态、动态、加速要素定义 ---
        static_concentration_score = normalize(df['concentration_90pct_D'], ascending=False)
        static_health_score = normalize(df['chip_health_score_D'], ascending=True)
        static_price_deviation_score = normalize(df['price_to_peak_ratio_D'], ascending=True)
        static_high_winner_rate_score = normalize(df['total_winner_rate_D'], ascending=True)
        conc_momentum_scores = {p: normalize(df[f'SLOPE_{p}_concentration_90pct_D'], ascending=False) for p in resonance_periods}
        cost_momentum_scores = {p: normalize(df[f'SLOPE_{p}_peak_cost_D'], ascending=True) for p in resonance_periods}
        health_momentum_scores = {p: normalize(df[f'SLOPE_{p}_chip_health_score_D'], ascending=True) for p in resonance_periods}
        conc_accel_score = normalize(df['ACCEL_1_concentration_90pct_D'], ascending=False)
        cost_accel_score = normalize(df['ACCEL_1_peak_cost_D'], ascending=True)
        health_accel_score = normalize(df['ACCEL_5_chip_health_score_D'], ascending=True)
        profit_taking_accel_score = normalize(df['ACCEL_5_turnover_from_winners_ratio_D'], ascending=True)
        # --- 4. 共振信号合成 (多周期乘积交叉验证) ---
        # 4.1 上升共振: 短中长周期动能的乘积，形成严格的“共振”
        bullish_conc_resonance = conc_momentum_scores[5] * conc_momentum_scores[21] * conc_momentum_scores[55]
        bullish_cost_resonance = cost_momentum_scores[5] * cost_momentum_scores[21] * cost_momentum_scores[55]
        bullish_health_resonance = health_momentum_scores[5] * health_momentum_scores[21] * health_momentum_scores[55]
        bullish_momentum_score = bullish_conc_resonance * bullish_cost_resonance * bullish_health_resonance
        bullish_accel_score = conc_accel_score * cost_accel_score * health_accel_score
        new_scores['SCORE_CHIP_BULLISH_RESONANCE_B'] = bullish_momentum_score.astype(np.float32)
        new_scores['SCORE_CHIP_BULLISH_RESONANCE_A'] = (bullish_momentum_score * static_health_score).astype(np.float32)
        new_scores['SCORE_CHIP_BULLISH_RESONANCE_S'] = (new_scores['SCORE_CHIP_BULLISH_RESONANCE_A'] * bullish_accel_score).astype(np.float32)
        # 4.2 下跌共振: 对称逻辑
        bearish_conc_resonance = (1 - conc_momentum_scores[5]) * (1 - conc_momentum_scores[21]) * (1 - conc_momentum_scores[55])
        bearish_cost_resonance = (1 - cost_momentum_scores[5]) * (1 - cost_momentum_scores[21]) * (1 - cost_momentum_scores[55])
        bearish_health_resonance = (1 - health_momentum_scores[5]) * (1 - health_momentum_scores[21]) * (1 - health_momentum_scores[55])
        bearish_momentum_score = bearish_conc_resonance * bearish_cost_resonance * bearish_health_resonance
        bearish_accel_score = (1 - conc_accel_score) * (1 - cost_accel_score) * (1 - health_accel_score)
        new_scores['SCORE_CHIP_BEARISH_RESONANCE_B'] = bearish_momentum_score.astype(np.float32)
        new_scores['SCORE_CHIP_BEARISH_RESONANCE_A'] = (bearish_momentum_score * (1 - static_health_score)).astype(np.float32)
        new_scores['SCORE_CHIP_BEARISH_RESONANCE_S'] = (new_scores['SCORE_CHIP_BEARISH_RESONANCE_A'] * bearish_accel_score).astype(np.float32)
        # --- 5. 反转信号合成 (逻辑保持V5.2的精炼版) ---
        bottom_setup_score = (1 - static_concentration_score) * (1 - static_price_deviation_score)
        bottom_trigger_momentum = conc_momentum_scores[5] * cost_momentum_scores[5]
        bottom_trigger_accel = conc_accel_score * cost_accel_score
        new_scores['SCORE_CHIP_BOTTOM_REVERSAL_B'] = (bottom_setup_score * bottom_trigger_momentum).astype(np.float32)
        new_scores['SCORE_CHIP_BOTTOM_REVERSAL_A'] = (new_scores['SCORE_CHIP_BOTTOM_REVERSAL_B'] * static_health_score).astype(np.float32)
        new_scores['SCORE_CHIP_BOTTOM_REVERSAL_S'] = (new_scores['SCORE_CHIP_BOTTOM_REVERSAL_A'] * bottom_trigger_accel).astype(np.float32)
        top_setup_score = static_concentration_score * static_price_deviation_score * static_high_winner_rate_score
        top_trigger_momentum = (1 - conc_momentum_scores[5]) * (1 - cost_momentum_scores[5])
        top_trigger_accel = (1 - conc_accel_score) * profit_taking_accel_score
        new_scores['SCORE_CHIP_TOP_REVERSAL_B'] = (top_setup_score * top_trigger_momentum).astype(np.float32)
        new_scores['SCORE_CHIP_TOP_REVERSAL_A'] = (new_scores['SCORE_CHIP_TOP_REVERSAL_B'] * (1 - static_health_score)).astype(np.float32)
        new_scores['SCORE_CHIP_TOP_REVERSAL_S'] = (new_scores['SCORE_CHIP_TOP_REVERSAL_A'] * top_trigger_accel).astype(np.float32)
        # --- 6.  纯筹码评分: 获利盘兑现强度 ---
        profit_taking_turnover_score = normalize(df['turnover_from_winners_ratio_D'])
        new_scores['SCORE_CHIP_PROFIT_TAKING_INTENSITY'] = (profit_taking_turnover_score * static_high_winner_rate_score).astype(np.float32)
        df = df.assign(**new_scores)
        print("        -> [筹码信号量化评分模块 V6.0 终极共振版] 计算完毕。")
        # --- 调试探针: 打印输入与输出的最后5条数据 ---
        print("\n==================== [探针: diagnose_quantitative_chip_scores] ====================")
        print("\n--- [原料信号探针] (最后5条数据) ---")
        final_raw_cols = [col for col in required_cols if col in df.columns]
        raw_df_subset = df[final_raw_cols]
        if not raw_df_subset.empty:
            print(raw_df_subset.tail(5).to_string())
        else:
            print("  - (无关键原料信号可供显示)")
        print("\n--- [产出信号探针] (最后5条数据) ---")
        if not new_scores:
            print("  - (无产出信号)")
        else:
            output_df = pd.DataFrame(new_scores)
            print(output_df.tail(5).to_string())
        print("==================== [探针结束] ====================\n")
        return df

    def diagnose_advanced_chip_dynamics_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V4.0 终极共振统一版】高级筹码动态评分模块
        - 核心重构 (V4.0): 统一共振信号逻辑。将“结构健康度共振”的计算方式从“多周期平均动能”
                          升级为最严格的“多周期动能乘积”，与V6.0黄金标准看齐，确保信号的最高可靠性。
        - 核心逻辑:
          - B级 (共振势): 结构健康度（控盘度、稳定性）在短、中、长周期上必须“协同改善”。
          - A级 (高质量共振): B级共振必须发生在“静态健康的结构”之上。
          - S级 (加速的共振): A级共振必须伴随“短期改善的加速”。
        """
        print("        -> [高级筹码动态评分模块 V4.0 终极共振统一版] 启动...") 
        new_scores = {}
        p = get_params_block(self.strategy, 'advanced_chip_dynamics_params')
        if not get_param_value(p.get('enabled'), True):
            return df
        # --- 1. 军备检查 (Arsenal Check) ---
        periods = get_param_value(p.get('dynamic_periods'), [1, 5, 21, 55])
        # 明确定义用于共振的核心周期
        resonance_periods = [p for p in periods if p in [5, 21, 55]]
        if len(resonance_periods) < 3:
            print(f"          -> [严重警告] 高级动态模块共振计算需要5, 21, 55周期，当前配置不足，模块已跳过！")
            return df
        required_cols = [
            'peak_control_ratio_D', 'peak_strength_ratio_D', 'peak_stability_D',
            'turnover_from_losers_ratio_D', 'is_chip_fault_formed_D',
            'chip_fault_strength_D', 'chip_fault_vacuum_percent_D'
        ]
        for period in periods:
            required_cols.extend([
                f'SLOPE_{period}_peak_control_ratio_D', f'SLOPE_{period}_peak_stability_D',
                f'ACCEL_{period}_peak_control_ratio_D', f'ACCEL_{period}_peak_stability_D',
            ])
        required_cols.extend(['ACCEL_5_turnover_from_losers_ratio_D', 'ACCEL_21_turnover_from_losers_ratio_D'])
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"          -> [严重警告] 高级筹码动态模块缺少关键数据列: {missing_cols}，模块已跳过！")
            return df
        # --- 2. 核心要素数值化 (归一化处理) ---
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        def normalize(series, ascending=True):
            return series.rolling(window=norm_window, min_periods=min_periods).rank(pct=True, ascending=ascending).fillna(0.5)
        # --- 3. 维度一: 结构健康度共振评分 (逻辑重构) ---
        # 3.1 静态健康分 (Setup)
        control_score = normalize(df['peak_control_ratio_D'])
        strength_score = normalize(df['peak_strength_ratio_D'])
        stability_score = normalize(df['peak_stability_D'])
        static_health_score = (control_score * strength_score * stability_score)
        # 3.2 动态健康分 (Momentum) - 采用多周期乘积共振
        control_momentum = {p: normalize(df[f'SLOPE_{p}_peak_control_ratio_D']) for p in resonance_periods}
        stability_momentum = {p: normalize(df[f'SLOPE_{p}_peak_stability_D']) for p in resonance_periods}
        bullish_control_resonance = control_momentum[5] * control_momentum[21] * control_momentum[55]
        bullish_stability_resonance = stability_momentum[5] * stability_momentum[21] * stability_momentum[55]
        health_bullish_resonance_score = bullish_control_resonance * bullish_stability_resonance
        # 3.3 加速健康分 (Acceleration)
        control_accel = normalize(df['ACCEL_1_peak_control_ratio_D'])
        stability_accel = normalize(df['ACCEL_1_peak_stability_D'])
        health_accel_score = control_accel * stability_accel
        # 3.4 上升共振 (健康度改善) - 遵循 B->A->S 逻辑重构
        new_scores['SCORE_STRUCTURE_BULLISH_RESONANCE_B'] = health_bullish_resonance_score.astype(np.float32)
        new_scores['SCORE_STRUCTURE_BULLISH_RESONANCE_A'] = (new_scores['SCORE_STRUCTURE_BULLISH_RESONANCE_B'] * static_health_score).astype(np.float32)
        new_scores['SCORE_STRUCTURE_BULLISH_RESONANCE_S'] = (new_scores['SCORE_STRUCTURE_BULLISH_RESONANCE_A'] * health_accel_score).astype(np.float32)
        # 3.5 下跌共振 (健康度恶化) - 对称逻辑重构
        bearish_control_resonance = (1 - control_momentum[5]) * (1 - control_momentum[21]) * (1 - control_momentum[55])
        bearish_stability_resonance = (1 - stability_momentum[5]) * (1 - stability_momentum[21]) * (1 - stability_momentum[55])
        health_bearish_resonance_score = bearish_control_resonance * bearish_stability_resonance
        health_decel_score = (1 - control_accel) * (1 - stability_accel)
        new_scores['SCORE_STRUCTURE_BEARISH_RESONANCE_B'] = health_bearish_resonance_score.astype(np.float32)
        new_scores['SCORE_STRUCTURE_BEARISH_RESONANCE_A'] = (new_scores['SCORE_STRUCTURE_BEARISH_RESONANCE_B'] * (1 - static_health_score)).astype(np.float32)
        new_scores['SCORE_STRUCTURE_BEARISH_RESONANCE_S'] = (new_scores['SCORE_STRUCTURE_BEARISH_RESONANCE_A'] * health_decel_score).astype(np.float32)
        # --- 4. 维度二: 恐慌盘承接反转评分 ---
        panic_selling_score = normalize(df['turnover_from_losers_ratio_D'])
        absorption_trigger_score = normalize(df['ACCEL_5_turnover_from_losers_ratio_D'])
        absorption_confirm_score = normalize(df['ACCEL_21_turnover_from_losers_ratio_D'])
        new_scores['SCORE_CAPITULATION_BOTTOM_REVERSAL_B'] = absorption_trigger_score.astype(np.float32)
        new_scores['SCORE_CAPITULATION_BOTTOM_REVERSAL_A'] = (panic_selling_score.shift(1) * absorption_trigger_score).astype(np.float32)
        new_scores['SCORE_CAPITULATION_BOTTOM_RESONANCE_S'] = (new_scores['SCORE_CAPITULATION_BOTTOM_REVERSAL_A'] * absorption_confirm_score).astype(np.float32)
        # --- 5. 维度三: 筹码断层顶部反转风险分 ---
        if 'is_chip_fault_formed_D' in df.columns:
            fault_existence_gate = df['is_chip_fault_formed_D'].astype(float)
        else:
            fault_existence_gate = pd.Series(1.0, index=df.index)
        fault_strength_score = normalize(df['chip_fault_strength_D'])
        vacuum_risk_score = normalize(df['chip_fault_vacuum_percent_D'])
        dynamic_worsening_score = new_scores.get('SCORE_STRUCTURE_BEARISH_RESONANCE_B', pd.Series(0.5, index=df.index))
        base_risk_b = fault_strength_score
        base_risk_a = base_risk_b * vacuum_risk_score
        base_risk_s = base_risk_a * dynamic_worsening_score
        new_scores['SCORE_FAULT_RISK_TOP_REVERSAL_B'] = (base_risk_b * fault_existence_gate).astype(np.float32)
        new_scores['SCORE_FAULT_RISK_TOP_REVERSAL_A'] = (base_risk_a * fault_existence_gate).astype(np.float32)
        new_scores['SCORE_FAULT_RISK_TOP_REVERSAL_S'] = (base_risk_s * fault_existence_gate).astype(np.float32)
        # --- 6. 一次性将所有新得分合并到DataFrame ---
        df = df.assign(**new_scores)
        print("        -> [高级筹码动态评分模块 V4.0 终极共振统一版] 计算完毕。")
        # --- 调试探针: 打印输入与输出的最后5条数据 ---
        print("\n==================== [探针: diagnose_advanced_chip_dynamics_scores] ====================")
        print("\n--- [原料信号探针] (最后5条数据) ---")
        final_raw_cols = [col for col in required_cols if col in df.columns]
        raw_df_subset = df[final_raw_cols]
        if not raw_df_subset.empty:
            print(raw_df_subset.tail(5).to_string())
        else:
            print("  - (无关键原料信号可供显示)")
        print("\n--- [产出信号探针] (最后5条数据) ---")
        if not new_scores:
            print("  - (无产出信号)")
        else:
            output_df = pd.DataFrame(new_scores)
            print(output_df.tail(5).to_string())
        print("==================== [探针结束] ====================\n")
        return df

    def diagnose_chip_internal_structure_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V4.0 终极共振统一版】筹码内部结构评分模块
        - 核心重构 (V4.0): 统一共振信号逻辑。将“核心持仓者共振”的计算方式从“多周期平均动能”
                          升级为最严格的“多周期动能乘积”，与V6.0黄金标准看齐，确保信号的最高可靠性。
        - 核心逻辑:
          - B级 (共振势): 核心持仓者相对外围持仓者的“净集中趋势”在短、中、长周期上必须“协同发生”。
          - A级 (高质量共振): B级共振必须发生在“静态已很集中”的结构之上。
          - S级 (加速的共振): A级共振必须伴随“短期净集中的加速”。
        """
        print("        -> [筹码内部结构评分模块 V4.0 终极共振统一版] 启动...") 
        new_scores = {}
        p = get_params_block(self.strategy, 'chip_internal_structure_params')
        if not get_param_value(p.get('enabled'), True):
            return df
        # --- 1. 军备检查 (Arsenal Check) ---
        periods = get_param_value(p.get('dynamic_periods'), [1, 5, 21, 55])
        # 明确定义用于共振的核心周期
        resonance_periods = [p for p in periods if p in [5, 21, 55]]
        if len(resonance_periods) < 3:
            print(f"          -> [严重警告] 内部结构模块共振计算需要5, 21, 55周期，当前配置不足，模块已跳过！")
            return df
        required_cols = [
            'concentration_70pct_D', 'concentration_90pct_D', 'winner_profit_margin_D',
            'support_below_D', 'pressure_above_D'
        ]
        for period in periods:
            required_cols.extend([
                f'SLOPE_{period}_concentration_70pct_D', f'SLOPE_{period}_concentration_90pct_D',
                f'SLOPE_{period}_winner_profit_margin_D',
                f'ACCEL_{period}_concentration_70pct_D', f'ACCEL_{period}_concentration_90pct_D',
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
        # --- 3. 维度一: 核心持仓者上升共振 (逻辑重构) ---
        # 3.1 静态分 (Setup)
        static_core_conc_score = normalize(df['concentration_70pct_D'], ascending=False)
        # 3.2 动态分 (Momentum) - 采用多周期乘积共振
        core_momentum = {p: normalize(df[f'SLOPE_{p}_concentration_70pct_D'], ascending=False) for p in resonance_periods}
        peripheral_momentum = {p: normalize(df[f'SLOPE_{p}_concentration_90pct_D'], ascending=False) for p in resonance_periods}
        net_gathering_momentum = {p: (core_momentum[p] - peripheral_momentum[p]).clip(0) for p in resonance_periods}
        net_gathering_resonance_score = net_gathering_momentum[5] * net_gathering_momentum[21] * net_gathering_momentum[55]
        # 3.3 加速分 (Acceleration)
        core_accel = normalize(df['ACCEL_1_concentration_70pct_D'], ascending=False)
        peripheral_accel = normalize(df['ACCEL_1_concentration_90pct_D'], ascending=False)
        net_gathering_accel_score = (core_accel - peripheral_accel).clip(0)
        # 3.4 融合生成S/A/B三级上升共振信号 - 遵循 B->A->S 逻辑重构
        new_scores['SCORE_CORE_HOLDER_BULLISH_RESONANCE_B'] = net_gathering_resonance_score.astype(np.float32)
        new_scores['SCORE_CORE_HOLDER_BULLISH_RESONANCE_A'] = (new_scores['SCORE_CORE_HOLDER_BULLISH_RESONANCE_B'] * static_core_conc_score).astype(np.float32)
        new_scores['SCORE_CORE_HOLDER_BULLISH_RESONANCE_S'] = (new_scores['SCORE_CORE_HOLDER_BULLISH_RESONANCE_A'] * normalize(net_gathering_accel_score)).astype(np.float32)
        # --- 4. 维度二: 获利盘兑现顶部反转风险 ---
        static_profit_margin = normalize(df['winner_profit_margin_D'])
        profit_margin_momentum = normalize(df['SLOPE_5_winner_profit_margin_D'])
        profit_margin_accel = normalize(df['ACCEL_5_winner_profit_margin_D'])
        new_scores['SCORE_PROFIT_TAKING_TOP_REVERSAL_B'] = static_profit_margin.astype(np.float32)
        new_scores['SCORE_PROFIT_TAKING_TOP_REVERSAL_A'] = (static_profit_margin * profit_margin_momentum).astype(np.float32)
        new_scores['SCORE_PROFIT_TAKING_TOP_REVERSAL_S'] = (new_scores['SCORE_PROFIT_TAKING_TOP_REVERSAL_A'] * profit_margin_accel).astype(np.float32)
        # --- 5. 维度三: 净支撑上升共振 ---
        static_support = normalize(df['support_below_D'])
        static_pressure = normalize(df['pressure_above_D'])
        static_net_support = normalize(static_support - static_pressure)
        support_momentum = normalize(df['SLOPE_5_support_below_D'])
        pressure_momentum = normalize(df['SLOPE_5_pressure_above_D'], ascending=False)
        dynamic_net_support = (support_momentum * pressure_momentum)
        new_scores['SCORE_NET_SUPPORT_BULLISH_RESONANCE_B'] = dynamic_net_support.astype(np.float32)
        new_scores['SCORE_NET_SUPPORT_BULLISH_RESONANCE_A'] = (static_net_support * dynamic_net_support).astype(np.float32)
        # --- 6. 一次性将所有新得分合并到DataFrame ---
        df = df.assign(**new_scores)
        print("        -> [筹码内部结构评分模块 V4.0 终极共振统一版] 计算完毕。")
        # --- 调试探针: 打印输入与输出的最后5条数据 ---
        print("\n==================== [探针: diagnose_chip_internal_structure_scores] ====================")
        print("\n--- [原料信号探针] (最后5条数据) ---")
        final_raw_cols = [col for col in required_cols if col in df.columns]
        raw_df_subset = df[final_raw_cols]
        if not raw_df_subset.empty:
            print(raw_df_subset.tail(5).to_string())
        else:
            print("  - (无关键原料信号可供显示)")
        print("\n--- [产出信号探针] (最后5条数据) ---")
        if not new_scores:
            print("  - (无产出信号)")
        else:
            output_df = pd.DataFrame(new_scores)
            print(output_df.tail(5).to_string())
        print("==================== [探针结束] ====================\n")
        return df

    def diagnose_chip_holder_behavior_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V4.0 终极共振统一版】筹码持仓者行为评分模块
        - 核心重构 (V4.0): 统一共振信号逻辑。将“成本结构共振”的计算方式从“多周期平均动能”
                          升级为最严格的“多周期动能乘积”，与V6.0黄金标准看齐，确保信号的最高可靠性。
        - 核心逻辑:
          - B级 (共振势): 成本发散度在短、中、长周期上必须“协同收敛”（斜率协同为正）。
          - A级 (高质量共振): B级共振必须发生在“静态成本结构健康”的背景下。
          - S级 (加速的共振): A级共振必须伴随“短期收敛的加速”。
        """
        print("        -> [持仓者行为评分模块 V4.0 终极共振统一版] 启动...") 
        new_scores = {}
        p = get_params_block(self.strategy, 'chip_holder_behavior_params')
        if not get_param_value(p.get('enabled'), True):
            return df
        # --- 1. 军备检查 (Arsenal Check) ---
        periods = get_param_value(p.get('dynamic_periods'), [1, 5, 21, 55])
        # 明确定义用于共振的核心周期
        resonance_periods = [p for p in periods if p in [5, 21, 55]]
        if len(resonance_periods) < 3:
            print(f"          -> [严重警告] 持仓者行为模块共振计算需要5, 21, 55周期，当前配置不足，模块已跳过！")
            return df
        required_cols = [
            'turnover_from_winners_ratio_D', 'loser_rate_long_term_D', 'cost_divergence_D'
        ]
        for period in periods:
            required_cols.extend([f'SLOPE_{period}_cost_divergence_D'])
        required_cols.extend([
            'ACCEL_1_cost_divergence_D', 'ACCEL_5_cost_divergence_D', 
            'SLOPE_5_turnover_from_winners_ratio_D', 'ACCEL_5_turnover_from_winners_ratio_D',
            'SLOPE_5_loser_rate_long_term_D', 'ACCEL_5_loser_rate_long_term_D'
        ])
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
        # 采用多周期乘积共振
        static_cost_divergence = normalize(df['cost_divergence_D'])
        cost_div_momentum = {p: normalize(df[f'SLOPE_{p}_cost_divergence_D']) for p in resonance_periods}
        cost_div_resonance_score = cost_div_momentum[5] * cost_div_momentum[21] * cost_div_momentum[55]
        cost_div_accel = normalize(df['ACCEL_1_cost_divergence_D'])
        new_scores['SCORE_COST_STRUCTURE_BULLISH_RESONANCE_B'] = cost_div_resonance_score.astype(np.float32)
        new_scores['SCORE_COST_STRUCTURE_BULLISH_RESONANCE_A'] = (static_cost_divergence * cost_div_resonance_score).astype(np.float32)
        new_scores['SCORE_COST_STRUCTURE_BULLISH_RESONANCE_S'] = (new_scores['SCORE_COST_STRUCTURE_BULLISH_RESONANCE_A'] * cost_div_accel).astype(np.float32)
        # --- 4. 维度二: 长线筹码不稳定顶部反转风险 ---
        static_instability = normalize(df['turnover_from_winners_ratio_D'])
        dynamic_instability = normalize(df['SLOPE_5_turnover_from_winners_ratio_D'])
        accel_instability = normalize(df['ACCEL_5_turnover_from_winners_ratio_D'])
        new_scores['SCORE_LONG_TERM_INSTABILITY_TOP_REVERSAL_B'] = static_instability.astype(np.float32)
        new_scores['SCORE_LONG_TERM_INSTABILITY_TOP_REVERSAL_A'] = (static_instability * dynamic_instability).astype(np.float32)
        new_scores['SCORE_LONG_TERM_INSTABILITY_TOP_REVERSAL_S'] = (new_scores['SCORE_LONG_TERM_INSTABILITY_TOP_REVERSAL_A'] * accel_instability).astype(np.float32)
        # --- 5. 维度三: 长线筹码投降底部反转机会 ---
        capitulation_setup = normalize(df['loser_rate_long_term_D'])
        capitulation_trigger = normalize(df['SLOPE_5_loser_rate_long_term_D'], ascending=False)
        capitulation_confirm = normalize(df['ACCEL_5_loser_rate_long_term_D'])
        new_scores['SCORE_LONG_TERM_CAPITULATION_BOTTOM_REVERSAL_B'] = capitulation_confirm.astype(np.float32)
        new_scores['SCORE_LONG_TERM_CAPITULATION_BOTTOM_REVERSAL_A'] = (new_scores['SCORE_LONG_TERM_CAPITULATION_BOTTOM_REVERSAL_B'] * capitulation_setup).astype(np.float32)
        new_scores['SCORE_LONG_TERM_CAPITULATION_BOTTOM_REVERSAL_S'] = (new_scores['SCORE_LONG_TERM_CAPITULATION_BOTTOM_REVERSAL_A'] * capitulation_trigger).astype(np.float32)
        # --- 6. 一次性将所有新得分合并到DataFrame ---
        df = df.assign(**new_scores)
        print("        -> [持仓者行为评分模块 V4.0 终极共振统一版] 计算完毕。") 
        # --- 调试探针: 打印输入与输出的最后5条数据 ---
        print("\n==================== [探针: diagnose_chip_holder_behavior_scores] ====================")
        print("\n--- [原料信号探针] (最后5条数据) ---")
        final_raw_cols = [col for col in required_cols if col in df.columns]
        raw_df_subset = df[final_raw_cols]
        if not raw_df_subset.empty:
            print(raw_df_subset.tail(5).to_string())
        else:
            print("  - (无关键原料信号可供显示)")
        print("\n--- [产出信号探针] (最后5条数据) ---")
        if not new_scores:
            print("  - (无产出信号)")
        else:
            output_df = pd.DataFrame(new_scores)
            print(output_df.tail(5).to_string())
        print("==================== [探针结束] ====================\n")
        return df

    def diagnose_fused_behavioral_chip_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V2.1 鲁棒性增强版】行为与筹码融合评分模块
        - 核心重构 (V2.1): 将所有动态指标的计算周期从1日提升至5日，以过滤市场噪音，使对“洗盘吸筹”和
                          “诱多派发”两大核心战术场景的判断基于更稳健的短期趋势，而非单日异动，更符合A股实战。
        - 核心逻辑:
          - 洗盘吸筹: 融合价格下跌、恐慌盘涌出、获利盘稳定、5日筹码集中趋势及加速度。
          - 诱多派发: 融合价格上涨与5日筹码集中度、健康度的趋势及加速度，捕捉“价涨质跌”的核心背离。
        """
        print("        -> [行为-筹码融合评分模块 V2.1 鲁棒性增强版] 启动...") 
        new_scores = {}
        p = get_params_block(self.strategy, 'fused_behavioral_chip_params', {})
        if not get_param_value(p.get('enabled'), True):
            return df
        # --- 1. 军备检查 ---
        # 将所有动态指标周期从1日提升至5日，增强鲁棒性
        required_cols = [
            'pct_change_D', 'turnover_from_losers_ratio_D',
            'turnover_from_winners_ratio_D', 
            'SLOPE_5_concentration_90pct_D', 'ACCEL_5_concentration_90pct_D',
            'SLOPE_5_chip_health_score_D', 'ACCEL_5_chip_health_score_D'
        ]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            print(f"          -> [警告] 行为-筹码融合模块缺少关键数据: {missing_cols}。模块已跳过。")
            return df
        # --- 2. 定义归一化辅助函数 ---
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        def normalize(series, ascending=True):
            return series.rolling(window=norm_window, min_periods=min_periods).rank(pct=True, ascending=ascending).fillna(0.5)
        # --- 3. 计算“洗盘吸筹”融合分 (Washout Absorption Opportunity) ---
        # 3.1 核心要素 (使用5日动态指标)
        drop_score = (1 - (df['pct_change_D'] - (-0.045)).abs() / 0.025).clip(0, 1)
        losers_capitulating_score = normalize(df['turnover_from_losers_ratio_D'], ascending=True)
        winners_holding_score = normalize(df['turnover_from_winners_ratio_D'], ascending=False)
        chip_improving_momentum = normalize(df['SLOPE_5_concentration_90pct_D'], ascending=False)
        chip_improving_accel = normalize(df['ACCEL_5_concentration_90pct_D'], ascending=False)
        # 3.2 融合生成S/A/B三级机会信号
        score_b_washout = drop_score * losers_capitulating_score
        new_scores['CHIP_SCORE_OPP_WASHOUT_ABSORPTION_B'] = score_b_washout.astype(np.float32)
        score_a_washout = score_b_washout * winners_holding_score * chip_improving_momentum
        new_scores['CHIP_SCORE_OPP_WASHOUT_ABSORPTION_A'] = score_a_washout.astype(np.float32)
        score_s_washout = score_a_washout * chip_improving_accel
        new_scores['CHIP_SCORE_OPP_WASHOUT_ABSORPTION_S'] = score_s_washout.astype(np.float32)
        # --- 4. 计算“诱多派发”融合分 (Deceptive Rally Risk) ---
        # 4.1 核心要素 (使用5日动态指标)
        rally_score = normalize(df['pct_change_D'], ascending=True)
        chip_worsening_momentum = normalize(df['SLOPE_5_concentration_90pct_D'], ascending=True)
        health_worsening_momentum = normalize(df['SLOPE_5_chip_health_score_D'], ascending=False)
        chip_worsening_accel = normalize(df['ACCEL_5_concentration_90pct_D'], ascending=True)
        health_worsening_accel = normalize(df['ACCEL_5_chip_health_score_D'], ascending=False)
        # 4.2 融合生成S/A/B三级风险信号
        score_b_deceptive = rally_score * chip_worsening_momentum
        new_scores['CHIP_SCORE_RISK_DECEPTIVE_RALLY_B'] = score_b_deceptive.astype(np.float32)
        score_a_deceptive = score_b_deceptive * health_worsening_momentum
        new_scores['CHIP_SCORE_RISK_DECEPTIVE_RALLY_A'] = score_a_deceptive.astype(np.float32)
        score_s_deceptive = score_a_deceptive * chip_worsening_accel * health_worsening_accel
        new_scores['CHIP_SCORE_RISK_DECEPTIVE_RALLY_S'] = score_s_deceptive.astype(np.float32)
        # --- 5. 为了兼容旧版，保留一个综合分数 ---
        new_scores['CHIP_SCORE_FUSED_WASHOUT_ABSORPTION'] = new_scores['CHIP_SCORE_OPP_WASHOUT_ABSORPTION_A']
        new_scores['CHIP_SCORE_FUSED_DECEPTIVE_RALLY'] = new_scores['CHIP_SCORE_RISK_DECEPTIVE_RALLY_A']
        # --- 6. 更新DataFrame ---
        df = df.assign(**new_scores)
        print("        -> [行为-筹码融合评分模块 V2.1 鲁棒性增强版] 计算完毕。") 
        # --- 调试探针: 打印输入与输出的最后5条数据 ---
        print("\n==================== [探针: diagnose_fused_behavioral_chip_scores] ====================")
        print("\n--- [原料信号探针] (最后5条数据) ---")
        final_raw_cols = [col for col in required_cols if col in df.columns]
        raw_df_subset = df[final_raw_cols]
        if not raw_df_subset.empty:
            print(raw_df_subset.tail(5).to_string())
        else:
            print("  - (无关键原料信号可供显示)")
        print("\n--- [产出信号探针] (最后5条数据) ---")
        if not new_scores:
            print("  - (无产出信号)")
        else:
            output_df = pd.DataFrame(new_scores)
            print(output_df.tail(5).to_string())
        print("==================== [探针结束] ====================\n")
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


