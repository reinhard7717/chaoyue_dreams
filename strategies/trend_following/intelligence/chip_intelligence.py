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

    def _normalize_score(self, series: pd.Series, window: int, ascending: bool = True) -> pd.Series:
        """
        将一个 Series 归一化到 0-1 区间。
        使用滚动窗口的百分位排名 (rank) 来实现。
        :param series: 输入的 pandas Series。
        :param window: 滚动窗口大小。
        :param ascending: 排序方向。True表示值越大分数越高，False反之。
        :return: 归一化后的 pandas Series。
        """
        # 检查输入是否有效
        if series is None or series.empty:
            # 如果输入为空，根据情况返回一个填充了中性值0.5的Series
            return pd.Series(0.5, index=series.index if series is not None else None)
        # 使用滚动窗口计算百分位排名，min_periods保证在数据初期也能尽快产出分数
        min_periods = window // 4
        rank_pct = series.rolling(window, min_periods=min_periods).rank(pct=True)
        # 根据排序方向调整分数
        if ascending:
            score = rank_pct
        else:
            score = 1.0 - rank_pct
        # 用中性值0.5填充因窗口期不足而产生的NaN
        return score.fillna(0.5)

    def run_chip_intelligence_command(self, df: pd.DataFrame) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
        """
        【V332.1 依赖解耦版】筹码情报最高司令部
        - 核心重构 (本次修改):
          - [依赖解耦] 移除了对 `_diagnose_true_concentration` 的内部调用。该诊断逻辑已被提升至 `IntelligenceLayer` 主流程中，以解决模块间的依赖顺序问题。
        - 业务逻辑: 保持与V332.0版本完全一致，仅调整内部调用结构。
        """
        print("        -> [筹码情报最高司令部 V332.1 依赖解耦版] 启动...")
        if 'avg_cost_short_term_D' in df.columns and 'avg_cost_long_term_D' in df.columns:
            df['cost_divergence_D'] = df['avg_cost_short_term_D'] - df['avg_cost_long_term_D']
        # 直接调用全新的七维交叉协同引擎
        ultimate_chip_states = self.diagnose_ultimate_chip_signals_v3(df)
        # --- 调用“恐慌盘投降反转”诊断引擎 ---
        capitulation_reversal_states = self._diagnose_capitulation_reversal(df)
        ultimate_chip_states.update(capitulation_reversal_states)
        print(f"        -> [筹码情报最高司令部 V332.1] 分析完毕，共生成 {len(ultimate_chip_states)} 个终极筹码信号。")
        return ultimate_chip_states, {}

    def diagnose_ultimate_chip_signals_v3(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V332.5 性能优化版】七维交叉协同终极筹码信号引擎
        - 核心优化 (本次修改):
          - [性能优化] 重构了内部计算逻辑，在循环中直接操作和堆叠NumPy数组，而不是创建临时的Pandas Series列表。
          - [内存优化] 减少了中间Pandas对象的创建，降低了内存占用，尤其在长周期回测中效果更佳。
        - 业务逻辑: 保持与V332.4版本完全一致，仅优化实现方式。
        """
        # print("        -> [终极筹码信号诊断模块 V332.5 性能优化版] 启动...") # 代码修改：更新版本号
        states = {}
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True): return states
        periods = get_param_value(p_conf.get('periods', [1, 5, 13, 21, 55]))
        norm_window = get_param_value(p_conf.get('norm_window'), 120)
        # --- 定义“位置上下文”分数 ---
        rolling_low_55d = df['low_D'].rolling(window=55, min_periods=21).min()
        rolling_high_55d = df['high_D'].rolling(window=55, min_periods=21).max()
        price_range_55d = (rolling_high_55d - rolling_low_55d).replace(0, 1e-9)
        price_position_in_range = ((df['close_D'] - rolling_low_55d) / price_range_55d).clip(0, 1).fillna(0.5)
        bottom_context_score = 1 - price_position_in_range
        top_context_score = price_position_in_range
        pillars = {
            'cost_structure':   [('peak_cost_D', True), ('cost_divergence_D', True)],
            'control_structure':  [('concentration_90pct_D', False), ('peak_control_ratio_D', True)],
            'holder_sentiment':   [('winner_profit_margin_D', True), ('loser_rate_long_term_D', False)],
            'structural_stability': [('peak_stability_D', True), ('is_multi_peak_D', False)],
            'pressure_risk':      [('pressure_above_D', False), ('turnover_from_winners_ratio_D', False)],
            'chip_fault':         [('chip_fault_strength_D', True), ('chip_fault_vacuum_percent_D', False)],
        }
        # --- 1. 军备检查 ---
        required_cols = set()
        all_potential_cols = set(df.columns)
        for p in periods:
            for pillar_name, factors in pillars.items():
                for factor_name, _ in factors:
                    required_cols.add(factor_name)
                    if factor_name != 'is_multi_peak_D':
                        required_cols.add(f"SLOPE_{p}_{factor_name}")
                        required_cols.add(f"ACCEL_{p}_{factor_name}")
                    factor_name_w = factor_name.replace('_D', '_W')
                    if factor_name_w in all_potential_cols:
                         required_cols.add(factor_name_w)
        missing_cols = list(required_cols - set(df.columns))
        if missing_cols:
            print(f"          -> [严重警告] 终极筹码引擎(七维)缺少关键数据: {sorted(missing_cols)}，模块已跳过！")
            return states
        # --- 2. 计算六大支柱的“周期健康度” (日线) ---
        pillar_period_health_d = {key: {} for key in pillars}
        for p in periods:
            for pillar_name, factors in pillars.items():
                # 代码修改：直接创建NumPy数组列表，而非Series列表
                factor_health_arrays = []
                for factor_name, ascending in factors:
                    if factor_name == 'is_multi_peak_D':
                        factor_health = 1.0 - df.get(factor_name, 0.0).astype(float)
                    else:
                        static_score = self._normalize_score(df.get(factor_name), norm_window, ascending=ascending)
                        slope_score = self._normalize_score(df.get(f"SLOPE_{p}_{factor_name}"), norm_window, ascending=ascending)
                        accel_score = self._normalize_score(df.get(f"ACCEL_{p}_{factor_name}"), norm_window, ascending=ascending)
                        factor_health = (static_score * slope_score * accel_score)**(1/3)
                    # 代码修改：将NumPy数组添加到列表中
                    factor_health_arrays.append(factor_health.values)
                if factor_health_arrays:
                    # 代码修改：直接堆叠NumPy数组进行计算
                    stacked_scores = np.stack(factor_health_arrays, axis=0)
                    geo_mean_values = np.prod(stacked_scores, axis=0)**(1/len(factor_health_arrays))
                    pillar_period_health_d[pillar_name][p] = pd.Series(geo_mean_values, index=df.index)
                else:
                    pillar_period_health_d[pillar_name][p] = pd.Series(0.5, index=df.index)
        # --- 3. 计算各支柱的“全面共识健康度” (日线) ---
        pillar_overall_health_d = {}
        for pillar_name in pillars:
            series_list = list(pillar_period_health_d[pillar_name].values())
            if series_list:
                # 代码修改：直接从Series列表中提取values进行堆叠
                stacked_scores = np.stack([s.values for s in series_list], axis=0)
                mean_values = np.mean(stacked_scores, axis=0)
                pillar_overall_health_d[pillar_name] = pd.Series(mean_values, index=df.index)
            else:
                pillar_overall_health_d[pillar_name] = pd.Series(0.5, index=df.index)
        # --- 4. 计算第七支柱: 跨周期协同 (MTF Synergy) ---
        # 代码修改：直接创建NumPy数组列表
        mtf_synergy_arrays = []
        for pillar_name, factors in pillars.items():
            # 代码修改：直接创建NumPy数组列表
            factor_synergy_arrays = []
            for factor_name, ascending in factors:
                factor_name_w = factor_name.replace('_D', '_W')
                if factor_name_w in df.columns:
                    daily_score = self._normalize_score(df.get(factor_name), norm_window, ascending=ascending)
                    weekly_score = self._normalize_score(df.get(factor_name_w), norm_window, ascending=ascending)
                    synergy_score = (daily_score / weekly_score.replace(0, 0.01)).clip(0, 2) / 2
                    # 代码修改：将NumPy数组添加到列表中
                    factor_synergy_arrays.append(synergy_score.values)
                else:
                    pass
            if factor_synergy_arrays:
                # 代码修改：直接堆叠NumPy数组进行计算
                stacked_scores = np.stack(factor_synergy_arrays, axis=0)
                mean_values = np.mean(stacked_scores, axis=0)
                # 代码修改：将NumPy数组添加到列表中
                mtf_synergy_arrays.append(mean_values)
        if mtf_synergy_arrays:
            # 代码修改：直接堆叠NumPy数组进行计算
            stacked_scores = np.stack(mtf_synergy_arrays, axis=0)
            mean_values = np.mean(stacked_scores, axis=0)
            pillar_overall_health_d['mtf_synergy'] = pd.Series(mean_values, index=df.index)
        else:
            pillar_overall_health_d['mtf_synergy'] = pd.Series(0.5, index=df.index)
        # --- 5. 融合生成“全局共识健康度” (Global Overall Health) ---
        overall_bullish_health = {}
        for p in periods:
            # 代码修改：直接创建NumPy数组列表
            health_arrays_for_period = [pillar_period_health_d[key][p].values for key in pillars]
            health_arrays_for_period.append(pillar_overall_health_d['mtf_synergy'].values)
            # 代码修改：直接堆叠NumPy数组进行计算
            stacked_scores = np.stack(health_arrays_for_period, axis=0)
            geo_mean_values = np.prod(stacked_scores, axis=0)**(1/len(health_arrays_for_period))
            overall_bullish_health[p] = pd.Series(geo_mean_values, index=df.index)
        overall_bearish_health = {p: 1.0 - overall_bullish_health[p] for p in periods}
        # --- 6. 交叉协同剧本诊断 ---
        ph = pillar_overall_health_d
        playbook_scores = {}
        playbook_scores['SCORE_CHIP_PLAYBOOK_WASHOUT'] = (ph['control_structure'] * ph['pressure_risk'] * ph['structural_stability'] * (1 - ph['holder_sentiment'])).astype(np.float32)
        playbook_scores['SCORE_CHIP_PLAYBOOK_VACUUM_BREAKOUT'] = (ph['chip_fault'] * ph['cost_structure'] * (1 - ph['pressure_risk'])).astype(np.float32)
        playbook_scores['SCORE_CHIP_PLAYBOOK_DISTRIBUTION'] = ((1 - ph['control_structure']) * ph['holder_sentiment'] * ph['pressure_risk']).astype(np.float32)
        control_slope_score = self._normalize_score(df.get('SLOPE_5_concentration_90pct_D'), norm_window, ascending=False)
        cost_slope_score = self._normalize_score(df.get('SLOPE_5_peak_cost_D'), norm_window, ascending=True)
        playbook_scores['SCORE_CHIP_PLAYBOOK_ABSORPTION'] = ((1 - ph['holder_sentiment']) * control_slope_score * cost_slope_score).astype(np.float32)
        states.update(playbook_scores)
        # --- 7. 终极信号合成 ---
        bullish_short_force = (overall_bullish_health[1] * overall_bullish_health[5])**0.5
        bullish_medium_trend = (overall_bullish_health[13] * overall_bullish_health[21])**0.5
        bullish_long_inertia = overall_bullish_health[55]
        bearish_short_force = (overall_bearish_health[1] * overall_bearish_health[5])**0.5
        bearish_medium_trend = (overall_bearish_health[13] * overall_bearish_health[21])**0.5
        bearish_long_inertia = overall_bearish_health[55]
        states['SCORE_CHIP_BULLISH_RESONANCE_B'] = overall_bullish_health[5].astype(np.float32)
        states['SCORE_CHIP_BULLISH_RESONANCE_A'] = (overall_bullish_health[5] * overall_bullish_health[21]).astype(np.float32)
        states['SCORE_CHIP_BULLISH_RESONANCE_S'] = (bullish_short_force * bullish_medium_trend).astype(np.float32)
        states['SCORE_CHIP_BULLISH_RESONANCE_S_PLUS'] = (states['SCORE_CHIP_BULLISH_RESONANCE_S'] * bullish_long_inertia).astype(np.float32)
        states['SCORE_CHIP_BEARISH_RESONANCE_B'] = overall_bearish_health[5].astype(np.float32)
        states['SCORE_CHIP_BEARISH_RESONANCE_A'] = (overall_bearish_health[5] * overall_bearish_health[21]).astype(np.float32)
        states['SCORE_CHIP_BEARISH_RESONANCE_S'] = (bearish_short_force * bearish_medium_trend).astype(np.float32)
        states['SCORE_CHIP_BEARISH_RESONANCE_S_PLUS'] = (states['SCORE_CHIP_BEARISH_RESONANCE_S'] * bearish_long_inertia).astype(np.float32)
        # --- 重构反转信号逻辑 ---
        states['SCORE_CHIP_BOTTOM_REVERSAL_B'] = (bottom_context_score * overall_bullish_health[1] * overall_bearish_health[21]).astype(np.float32)
        states['SCORE_CHIP_BOTTOM_REVERSAL_A'] = (bottom_context_score * overall_bullish_health[5] * overall_bearish_health[21]).astype(np.float32)
        states['SCORE_CHIP_BOTTOM_REVERSAL_S'] = (bottom_context_score * bullish_short_force * bearish_long_inertia).astype(np.float32)
        states['SCORE_CHIP_BOTTOM_REVERSAL_S_PLUS'] = (bottom_context_score * bullish_short_force * bullish_medium_trend * bearish_long_inertia).astype(np.float32)
        states['SCORE_CHIP_TOP_REVERSAL_B'] = (top_context_score * overall_bearish_health[1] * overall_bullish_health[21]).astype(np.float32)
        states['SCORE_CHIP_TOP_REVERSAL_A'] = (top_context_score * overall_bearish_health[5] * overall_bullish_health[21]).astype(np.float32)
        states['SCORE_CHIP_TOP_REVERSAL_S'] = (top_context_score * bearish_short_force * bullish_long_inertia).astype(np.float32)
        states['SCORE_CHIP_TOP_REVERSAL_S_PLUS'] = (top_context_score * bearish_short_force * bullish_medium_trend * bullish_long_inertia).astype(np.float32)
        # --- 8. 导出七维支柱的最终健康分 ---
        for pillar_name, health_score in pillar_overall_health_d.items():
            signal_name = f"SCORE_CHIP_PILLAR_{pillar_name.upper()}_HEALTH"
            states[signal_name] = health_score.astype(np.float32)
        return states

    def diagnose_cross_validation_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V3.2 信号逻辑精炼版】终极共振与反转信号诊断模块
        - 核心范式 (V3.2 升级):
          - 1. 深度交叉验证: 保持对每一因子在每一时间周期上进行“静态-斜率-加速度”三重验证，形成“已验证的周期共振分”，确保信号源的高质量。
          - 2. 信号组件化定义: 为提升逻辑清晰度和实战意义，将周期共振分显式组合为“短期力量”、“中期趋势”和“长期惯性”，使信号构建过程更透明。
          - 3. 精炼共振信号分级:
             - B级: 短期(5D)趋势出现。
             - A级: 短期(5D)与中期(21D)趋势共振。
             - S级: 短期力量(1D*5D)与中期趋势(13D*21D)共振。
             - S+级: S级信号得到长期惯性(55D)的确认，形成全周期共振。
          - 4. 精炼反转信号分级:
             - B级: 出现1日反转火花，对抗21日中期趋势。
             - A级: 形成5日反转力量，对抗21日中期趋势。
             - S级: 形成短期反转合力(1D*5D)，对抗55日长期惯性。
             - S+级: 形成短期反转合力(1D*5D)，对抗中长期联合惯性(21D*55D)。
        - A股实战考量: V3.2的信号体系更贴近交易员的思维模式，清晰地划分了“观察(B) -> 建仓(A) -> 加仓(S) -> 持有(S+)”的决策过程，为不同风险偏好的策略提供了更精细的输入。
        """
        print("        -> [交叉验证诊断模块 V3.2 信号逻辑精炼版] 启动...")
        new_scores = {}
        p = get_params_block(self.strategy, 'cross_validation_params', {})
        if not get_param_value(p.get('enabled'), True):
            return df

        # --- 1. 参数与辅助函数定义 ---
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        periods = get_param_value(p.get('resonance_periods'), [1, 5, 13, 21, 55])

        def normalize(series, ascending=True):
            return series.rolling(window=norm_window, min_periods=min_periods).rank(pct=True, ascending=ascending).fillna(0.5)

        # --- 2. 定义信号因子组 (Signal Factor Groups) ---
        SIGNAL_GROUPS = {
            'BULLISH': {
                'concentration_90pct_D': 'down', # 集中度
                'peak_cost_D': 'up',             # 成本
                'chip_health_score_D': 'up',     # 健康分
                'peak_control_ratio_D': 'up',    # 控盘度
            },
            'BEARISH': {
                'concentration_90pct_D': 'up',   # 集中度
                'peak_cost_D': 'down',           # 成本
                'chip_health_score_D': 'down',   # 健康分
                'winner_profit_margin_D': 'down',# 获利盘安全垫
            }
        }

        # --- 3. 深度交叉验证与周期共振分计算 ---
        period_validated_resonance = {'BULLISH': {}, 'BEARISH': {}}
        print("        -> 开始计算所有周期的深度交叉验证共振分...") # 修改: 简化打印信息
        for group_name, factors in SIGNAL_GROUPS.items():
            for period in periods:
                current_period_resonance = pd.Series(1.0, index=df.index)
                for factor, direction in factors.items():
                    static_col = factor
                    slope_col = f'SLOPE_{period}_{factor}'
                    accel_col = f'ACCEL_{period}_{factor}'
                    ascending = (direction == 'up')
                    
                    static_score = normalize(df[static_col], ascending=ascending) if static_col in df.columns else pd.Series(0.5, index=df.index)
                    slope_score = normalize(df[slope_col], ascending=ascending) if slope_col in df.columns else pd.Series(0.5, index=df.index)
                    accel_score = normalize(df[accel_col], ascending=ascending) if accel_col in df.columns else pd.Series(0.5, index=df.index)
                    
                    if static_col not in df.columns: print(f"            -> [交叉验证警告] 缺失静态数据: {static_col}")
                    if slope_col not in df.columns: print(f"            -> [交叉验证警告] 缺失斜率数据: {slope_col}")
                    if accel_col not in df.columns: print(f"            -> [交叉验证警告] 缺失加速度数据: {accel_col}")

                    validated_factor_score = static_score * slope_score * accel_score
                    current_period_resonance *= validated_factor_score
                
                period_validated_resonance[group_name][period] = current_period_resonance
        print("        -> 所有周期的深度交叉验证共振分计算完毕。") # 修改: 简化打印信息

        # --- 4. 基于“周期共振分”构建 B/A/S/S+ 四级信号 ---
        bullish_scores = period_validated_resonance['BULLISH']
        bearish_scores = period_validated_resonance['BEARISH']

        # 显式定义信号组件，提升可读性和逻辑清晰度
        bullish_short_force = bullish_scores[1] * bullish_scores[5]
        bullish_medium_trend = bullish_scores[13] * bullish_scores[21]
        bullish_long_inertia = bullish_scores[55]
        
        bearish_short_force = bearish_scores[1] * bearish_scores[5]
        bearish_medium_trend = bearish_scores[13] * bearish_scores[21]
        bearish_long_inertia = bearish_scores[55]

        # 4.1 上升共振信号 (Rising Resonance) - # 修改: 使用信号组件重构
        score_rising_b = bullish_scores[5]
        score_rising_a = bullish_scores[5] * bullish_scores[21]
        score_rising_s = bullish_short_force * bullish_medium_trend
        score_rising_s_plus = score_rising_s * bullish_long_inertia
        
        new_scores['SCORE_RISING_RESONANCE_B'] = score_rising_b.astype(np.float32)
        new_scores['SCORE_RISING_RESONANCE_A'] = score_rising_a.astype(np.float32)
        new_scores['SCORE_RISING_RESONANCE_S'] = score_rising_s.astype(np.float32)
        new_scores['SCORE_RISING_RESONANCE_S_PLUS'] = score_rising_s_plus.astype(np.float32)

        # 4.2 下跌共振信号 (Falling Resonance) - # 修改: 使用信号组件重构
        score_falling_b = bearish_scores[5]
        score_falling_a = bearish_scores[5] * bearish_scores[21]
        score_falling_s = bearish_short_force * bearish_medium_trend
        score_falling_s_plus = score_falling_s * bearish_long_inertia

        new_scores['SCORE_FALLING_RESONANCE_B'] = score_falling_b.astype(np.float32)
        new_scores['SCORE_FALLING_RESONANCE_A'] = score_falling_a.astype(np.float32)
        new_scores['SCORE_FALLING_RESONANCE_S'] = score_falling_s.astype(np.float32)
        new_scores['SCORE_FALLING_RESONANCE_S_PLUS'] = score_falling_s_plus.astype(np.float32)

        # 4.3 底部反转信号 (Bottom Reversal) - # 修改: 使用信号组件重构
        score_bottom_b = bullish_scores[1] * bearish_scores[21]
        score_bottom_a = bullish_scores[5] * bearish_scores[21]
        score_bottom_s = bullish_short_force * bearish_long_inertia
        score_bottom_s_plus = bullish_short_force * bearish_scores[21] * bearish_long_inertia

        new_scores['SCORE_BOTTOM_REVERSAL_B'] = score_bottom_b.astype(np.float32)
        new_scores['SCORE_BOTTOM_REVERSAL_A'] = score_bottom_a.astype(np.float32)
        new_scores['SCORE_BOTTOM_REVERSAL_S'] = score_bottom_s.astype(np.float32)
        new_scores['SCORE_BOTTOM_REVERSAL_S_PLUS'] = score_bottom_s_plus.astype(np.float32)

        # 4.4 顶部反转信号 (Top Reversal) - # 修改: 使用信号组件重构
        score_top_b = bearish_scores[1] * bullish_scores[21]
        score_top_a = bearish_scores[5] * bullish_scores[21]
        score_top_s = bearish_short_force * bullish_long_inertia
        score_top_s_plus = bearish_short_force * bullish_scores[21] * bullish_long_inertia

        new_scores['SCORE_TOP_REVERSAL_B'] = score_top_b.astype(np.float32)
        new_scores['SCORE_TOP_REVERSAL_A'] = score_top_a.astype(np.float32)
        new_scores['SCORE_TOP_REVERSAL_S'] = score_top_s.astype(np.float32)
        new_scores['SCORE_TOP_REVERSAL_S_PLUS'] = score_top_s_plus.astype(np.float32)

        # --- 5. 更新DataFrame ---
        df = df.assign(**new_scores)
        print(f"        -> [交叉验证诊断模块 V3.2 信号逻辑精炼版] 计算完毕，新增 {len(new_scores)} 个终极信号。")
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
        return df

    def diagnose_strategic_context_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.1 评分扩展版】战略级筹码上下文评分模块
        - 核心职责: 融合最长周期的筹码指标，生成描述宏观“战略吸筹”与“战略派发”的
                      顶层上下文分数。这是整个筹码情报模块的基石。
        - 核心升级(V1.1): 新增“亢奋上涨预警”评分，用于识别市场过热的宏观状态。
        - 收益: 为下游所有模块提供了最关键的宏观背景判断。
        """
        print("        -> [战略级筹码上下文评分模块 V1.1 评分扩展版] 启动...")
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

    def _diagnose_capitulation_reversal(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】恐慌盘投降反转 (Capitulation Reversal) 诊断引擎
        - 核心逻辑: 捕捉市场深度超卖后，套牢盘集中“割肉”的时刻，这往往是卖压衰竭、趋势反转的前兆。
          - 上下文（Setup）: 市场整体处于深度套牢状态 (`total_loser_rate` 很高)。
          - 触发器（Trigger）: “割肉盘”成交占比 (`turnover_from_losers_ratio`) 及其加速度同时飙升。
        - 产出: SCORE_CHIP_PLAYBOOK_CAPITULATION_REVERSAL - 一个高置信度的底部反转剧本信号。
        """
        states = {}
        norm_window = 120
        
        # --- 1. 军备检查 ---
        required_cols = [
            'total_loser_rate_D',
            'turnover_from_losers_ratio_D',
            'ACCEL_5_turnover_from_losers_ratio_D'
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"          -> [严重警告] 恐慌盘投降反转诊断引擎缺少关键数据: {sorted(missing_cols)}，模块已跳过！")
            return states

        # --- 2. 核心要素数值化 ---
        # 上下文：市场整体套牢比例越高，分数越高
        capitulation_context_score = self._normalize_score(df['total_loser_rate_D'], norm_window, ascending=True)

        # 触发器1：当日割肉盘占比越高，分数越高
        loser_turnover_score = self._normalize_score(df['turnover_from_losers_ratio_D'], norm_window, ascending=True)

        # 触发器2：割肉盘占比的加速度越大，分数越高
        loser_turnover_accel_score = self._normalize_score(df['ACCEL_5_turnover_from_losers_ratio_D'], norm_window, ascending=True)

        # --- 3. 最终裁决 ---
        # 最终分数 = 上下文 * 触发器1 * 触发器2
        final_score = (
            capitulation_context_score *
            loser_turnover_score *
            loser_turnover_accel_score
        ).astype(np.float32)
        
        states['SCORE_CHIP_PLAYBOOK_CAPITULATION_REVERSAL'] = final_score
        
        return states

    def _diagnose_true_concentration(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】真实吸筹 vs 虚假集中（顶部派发）诊断引擎
        - 核心逻辑: 通过交叉验证筹码内部结构指标，区分“价格集中”背后的真实意图。
          - 真实吸筹 = 价格集中 + 获利盘稳定 + 市场情绪冷静 + 主力控盘度上升
          - 虚假集中 = 价格集中 + 获利盘涌出 + 市场情绪亢奋 + 主力控盘度下降
        - 产出: 两个互斥的、高置信度的S级筹码认知信号。
        """
        states = {}
        norm_window = 120
        
        # --- 1. 军备检查 ---
        required_cols = [
            'SLOPE_21_concentration_90pct_D',
            'winner_profit_margin_D',
            'turnover_from_winners_ratio_D',
            'SLOPE_21_peak_control_ratio_D'
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"          -> [严重警告] 真实吸筹诊断引擎缺少关键数据: {sorted(missing_cols)}，模块已跳过！")
            return states

        # --- 2. 核心要素数值化 ---
        raw_concentration_score = self._normalize_score(df['SLOPE_21_concentration_90pct_D'], norm_window, ascending=False)
        low_profit_margin_score = self._normalize_score(df['winner_profit_margin_D'], norm_window, ascending=False)
        high_profit_margin_score = 1 - low_profit_margin_score
        low_winner_turnover_score = self._normalize_score(df['turnover_from_winners_ratio_D'], norm_window, ascending=False)
        high_winner_turnover_score = 1 - low_winner_turnover_score
        increasing_control_score = self._normalize_score(df['SLOPE_21_peak_control_ratio_D'], norm_window, ascending=True)
        decreasing_control_score = 1 - increasing_control_score

        # --- 3. 最终裁决 ---
        true_accumulation_score = (
            raw_concentration_score *
            low_profit_margin_score *
            low_winner_turnover_score *
            increasing_control_score
        ).astype(np.float32)
        states['SCORE_CHIP_TRUE_ACCUMULATION'] = true_accumulation_score
        false_accumulation_risk = (
            raw_concentration_score *
            high_profit_margin_score *
            high_winner_turnover_score *
            decreasing_control_score
        ).astype(np.float32)
        states['SCORE_CHIP_FALSE_ACCUMULATION_RISK'] = false_accumulation_risk
        
        return states
