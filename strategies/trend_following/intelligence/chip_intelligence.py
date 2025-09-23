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

    def _normalize_score(self, series: pd.Series, window: int, ascending: bool = True, default: float = 0.5) -> pd.Series:
        """
        【V1.1 统一签名版】将一个 Series 归一化到 0-1 区间。
        - 核心升级 (本次修改):
          - [统一签名] 新增了 `default` 参数，使其与项目中其他 `_normalize_score` 方法的签名保持一致。
          - [健壮性] 优化了空 Series 的处理逻辑，确保在任何情况下都能返回一个带有正确索引和默认值的 Series。
        - 收益: 解决了因函数签名不一致导致的 TypeError，提升了代码的健壮性和可维护性。
        """
        # 检查输入是否有效
        if series is None or series.empty:
            # 如果输入为空，返回一个填充了指定默认值的Series
            # 优先使用 series 的索引，如果 series 为 None，则回退到主 df 的索引
            index = series.index if series is not None else self.strategy.df_indicators.index
            return pd.Series(default, index=index)
        
        # 使用滚动窗口计算百分位排名，min_periods保证在数据初期也能尽快产出分数
        min_periods = window // 4
        rank_pct = series.rolling(window, min_periods=min_periods).rank(pct=True)
        
        # 根据排序方向调整分数
        if ascending:
            score = rank_pct
        else:
            score = 1.0 - rank_pct
            
        # 用指定的默认值填充因窗口期不足而产生的NaN
        return score.fillna(default)

    def run_chip_intelligence_command(self, df: pd.DataFrame) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
        """
        【V406.1 架构重构版】筹码情报最高司令部
        - 核心重构 (本次修改):
          - [架构升级] 将“恐慌盘投降”信号的诊断过程重构为三个独立的、可复用的模块：
                        1. _diagnose_setup_capitulation_ready (诊断战备状态)
                        2. _diagnose_trigger_capitulation_fire (诊断点火行为)
                        3. _synthesize_playbook_capitulation_reversal (合成最终剧本)
          - [流程编排] 在主流程中按“先原子，后合成”的顺序调用这些新模块。
        - 收益: 实现了信号的模块化和原子状态的复用，是架构上的一次重大升级。
        """
        
        print("        -> [筹码情报最高司令部 V406.1 架构重构版] 启动...") # 更新版本号
        
        all_chip_states = {}
        # 步骤 1: 执行所有独立的诊断和评分模块，生成所有纯筹码维度的信号
        df = self.diagnose_composite_scores(df)
        df = self.diagnose_strategic_context_scores(df)
        df = self.diagnose_quantitative_chip_scores(df)
        df = self.diagnose_advanced_chip_dynamics_scores(df)
        df = self.diagnose_chip_internal_structure_scores(df)
        df = self.diagnose_chip_holder_behavior_scores(df)
        df = self.diagnose_fused_behavioral_chip_scores(df)
        df = self.diagnose_cross_validation_signals(df)
        # 这是一个临时措施，用于从df中提取所有新生成的列
        for col in df.columns:
            if col not in self.strategy.df_indicators.columns and col not in all_chip_states:
                 all_chip_states[col] = df[col]
        # 步骤 2: 执行依赖于第一步结果的诊断模块
        accumulation_states = self.diagnose_accumulation_playbooks(df)
        all_chip_states.update(accumulation_states)
        # 步骤 3: 执行终极信号引擎
        ultimate_chip_states = self.diagnose_ultimate_chip_signals_v3(df)
        all_chip_states.update(ultimate_chip_states)
        # 步骤 4: 执行独立的原子状态诊断 (新的模块化流程)
        # 先诊断出可复用的“战备”和“点火”原子状态
        setup_states = self._diagnose_setup_capitulation_ready(df)
        all_chip_states.update(setup_states)
        trigger_states = self._diagnose_trigger_capitulation_fire(df)
        all_chip_states.update(trigger_states)
        # 步骤 5: 执行剧本合成器
        # 在调用合成器之前，必须确保原子状态已添加到df中，供其消费
        df = df.assign(**all_chip_states)
        playbook_states = self._synthesize_playbook_capitulation_reversal(df)
        all_chip_states.update(playbook_states)
        print(f"        -> [筹码情报最高司令部 V406.1] 分析完毕，共生成 {len(all_chip_states)} 个筹码信号。") # 更新版本号
        return all_chip_states, {}

    def diagnose_accumulation_playbooks(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V4.4 拉升吸筹逻辑修复版】主力吸筹模式与风险诊断引擎
        - 核心升级 (本次修改):
          - [逻辑修复] 修正了“拉升吸筹”剧本的计算逻辑。原逻辑要求在拉升的同时筹码必须集中，这与市场实际情况相悖。
          - [新范式] 新的“拉升吸筹分” = “成本抬升分” * “获利盘稳定分”，移除了不合理的“筹码集中改善分”因子，使其更专注于健康拉升的核心特征。
        - 收益: 大幅提升了“真实吸筹”信号在上涨行情中的捕捉能力和准确性。
        """
        
        print("        -> [主力吸筹与风险诊断引擎 V4.4 拉升吸筹逻辑修复版] 启动...")
        states = {}
        norm_window = 120
        required_cols = [
            'SLOPE_5_concentration_90pct_D', 'ACCEL_5_concentration_90pct_D',
            'SLOPE_5_peak_cost_D', 'SLOPE_5_turnover_from_winners_ratio_D',
            'turnover_from_losers_ratio_D',
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"          -> [严重警告] 主力吸筹诊断引擎V4.4缺少关键数据: {sorted(missing_cols)}，模块已跳过！")
            return states
        # --- 2. 核心要素数值化 ---
        conc_slope_score = self._normalize_score(df['SLOPE_5_concentration_90pct_D'], norm_window, ascending=False)
        conc_accel_score = self._normalize_score(df['ACCEL_5_concentration_90pct_D'], norm_window, ascending=False)
        concentration_improving_score = (conc_slope_score * conc_accel_score)
        cost_rising_score = self._normalize_score(df['SLOPE_5_peak_cost_D'], norm_window, ascending=True)
        cost_falling_score = self._normalize_score(df['SLOPE_5_peak_cost_D'], norm_window, ascending=False)
        winner_holding_score = self._normalize_score(df['SLOPE_5_turnover_from_winners_ratio_D'], norm_window, ascending=False)
        loser_capitulating_score = self._normalize_score(df['turnover_from_losers_ratio_D'], norm_window, ascending=True)
        # 将所有内部因子发布到 states，供探针消费
        states['INTERNAL_SCORE_CONCENTRATION_IMPROVING'] = concentration_improving_score.astype(np.float32)
        states['INTERNAL_SCORE_COST_RISING'] = cost_rising_score.astype(np.float32)
        states['INTERNAL_SCORE_WINNER_HOLDING'] = winner_holding_score.astype(np.float32)
        states['INTERNAL_SCORE_COST_FALLING'] = cost_falling_score.astype(np.float32)
        states['INTERNAL_SCORE_LOSER_CAPITULATING'] = loser_capitulating_score.astype(np.float32)
        # --- 3. 剧本与风险合成 ---
        # 移除 concentration_improving_score 因子，使其更符合拉升时的市场行为
        rally_accumulation_score = (cost_rising_score * winner_holding_score).astype(np.float32)
        states['SCORE_CHIP_PLAYBOOK_RALLY_ACCUMULATION'] = rally_accumulation_score
        suppress_accumulation_score = (concentration_improving_score * cost_falling_score * loser_capitulating_score).astype(np.float32)
        states['SCORE_CHIP_PLAYBOOK_SUPPRESS_ACCUMULATION'] = suppress_accumulation_score
        concentration_worsening_score = (1 - conc_slope_score) * (1 - conc_accel_score)
        winner_distributing_score = self._normalize_score(df['SLOPE_5_turnover_from_winners_ratio_D'], norm_window, ascending=True)
        false_accumulation_risk_score = (concentration_worsening_score * cost_rising_score * winner_distributing_score).astype(np.float32)
        states['SCORE_CHIP_FALSE_ACCUMULATION_RISK'] = false_accumulation_risk_score
        true_accumulation_score = np.maximum(rally_accumulation_score, suppress_accumulation_score)
        states['SCORE_CHIP_TRUE_ACCUMULATION'] = true_accumulation_score.astype(np.float32)
        # 植入“一线法医探针”
        debug_params = get_params_block(self.strategy, 'debug_params')
        probe_date_str = get_param_value(debug_params.get('probe_date'))
        if probe_date_str:
            probe_ts = pd.to_datetime(probe_date_str)
            if probe_ts in df.index:
                print(f"\n          --- [一线探针: 主力吸筹诊断 @ {probe_date_str}] ---")
                print(f"          - 因子: 筹码集中改善分 (斜率分 * 加速度分) = {concentration_improving_score.get(probe_ts, -1):.4f} ({conc_slope_score.get(probe_ts, -1):.4f} * {conc_accel_score.get(probe_ts, -1):.4f})")
                print(f"          - 拉升吸筹因子:")
                print(f"            - 成本抬升分: {cost_rising_score.get(probe_ts, -1):.4f}")
                print(f"            - 获利盘稳定分: {winner_holding_score.get(probe_ts, -1):.4f}")
                print(f"          - 打压吸筹因子:")
                print(f"            - 成本下降分: {cost_falling_score.get(probe_ts, -1):.4f}")
                print(f"            - 恐慌盘涌出分: {loser_capitulating_score.get(probe_ts, -1):.4f}")
                print(f"          - 最终剧本分:")
                # 更新探针日志，明确显示新的计算逻辑
                print(f"            - 拉升吸筹剧本分 (成本*稳定): {rally_accumulation_score.get(probe_ts, -1):.4f}")
                print(f"            - 打压吸筹剧本分 (集中*成本*恐慌): {suppress_accumulation_score.get(probe_ts, -1):.4f}")
                print(f"            - 真实吸筹分(最大值): {true_accumulation_score.get(probe_ts, -1):.4f}")
                print(f"          ----------------------------------------------------\n")
        return states

    def diagnose_ultimate_chip_signals_v3(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V332.5 性能优化版】七维交叉协同终极筹码信号引擎
        - 核心优化 (本次修改):
          - [性能优化] 重构了内部计算逻辑，在循环中直接操作和堆叠NumPy数组，而不是创建临时的Pandas Series列表。
          - [内存优化] 减少了中间Pandas对象的创建，降低了内存占用，尤其在长周期回测中效果更佳。
        - 业务逻辑: 保持与V332.4版本完全一致，仅优化实现方式。
        """
        # print("        -> [终极筹码信号诊断模块 V332.5 性能优化版] 启动...") # 更新版本号
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
                # 直接创建NumPy数组列表，而非Series列表
                factor_health_arrays = []
                for factor_name, ascending in factors:
                    if factor_name == 'is_multi_peak_D':
                        factor_health = 1.0 - df.get(factor_name, 0.0).astype(float)
                    else:
                        static_score = self._normalize_score(df.get(factor_name), norm_window, ascending=ascending)
                        slope_score = self._normalize_score(df.get(f"SLOPE_{p}_{factor_name}"), norm_window, ascending=ascending)
                        accel_score = self._normalize_score(df.get(f"ACCEL_{p}_{factor_name}"), norm_window, ascending=ascending)
                        factor_health = (static_score * slope_score * accel_score)**(1/3)
                    # 将NumPy数组添加到列表中
                    factor_health_arrays.append(factor_health.values)
                if factor_health_arrays:
                    # 直接堆叠NumPy数组进行计算
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
                # 直接从Series列表中提取values进行堆叠
                stacked_scores = np.stack([s.values for s in series_list], axis=0)
                mean_values = np.mean(stacked_scores, axis=0)
                pillar_overall_health_d[pillar_name] = pd.Series(mean_values, index=df.index)
            else:
                pillar_overall_health_d[pillar_name] = pd.Series(0.5, index=df.index)
        # --- 4. 计算第七支柱: 跨周期协同 (MTF Synergy) ---
        # 直接创建NumPy数组列表
        mtf_synergy_arrays = []
        for pillar_name, factors in pillars.items():
            # 直接创建NumPy数组列表
            factor_synergy_arrays = []
            for factor_name, ascending in factors:
                factor_name_w = factor_name.replace('_D', '_W')
                if factor_name_w in df.columns:
                    daily_score = self._normalize_score(df.get(factor_name), norm_window, ascending=ascending)
                    weekly_score = self._normalize_score(df.get(factor_name_w), norm_window, ascending=ascending)
                    synergy_score = (daily_score / weekly_score.replace(0, 0.01)).clip(0, 2) / 2
                    # 将NumPy数组添加到列表中
                    factor_synergy_arrays.append(synergy_score.values)
                else:
                    pass
            if factor_synergy_arrays:
                # 直接堆叠NumPy数组进行计算
                stacked_scores = np.stack(factor_synergy_arrays, axis=0)
                mean_values = np.mean(stacked_scores, axis=0)
                # 将NumPy数组添加到列表中
                mtf_synergy_arrays.append(mean_values)
        if mtf_synergy_arrays:
            # 直接堆叠NumPy数组进行计算
            stacked_scores = np.stack(mtf_synergy_arrays, axis=0)
            mean_values = np.mean(stacked_scores, axis=0)
            pillar_overall_health_d['mtf_synergy'] = pd.Series(mean_values, index=df.index)
        else:
            pillar_overall_health_d['mtf_synergy'] = pd.Series(0.5, index=df.index)
        # --- 5. 融合生成“全局共识健康度” (Global Overall Health) ---
        overall_bearish_health = {}
        overall_bullish_health = {}
        for p in periods:
            # 直接创建NumPy数组列表
            health_arrays_for_period = [pillar_period_health_d[key][p].values for key in pillars]
            health_arrays_for_period.append(pillar_overall_health_d['mtf_synergy'].values)
            # 直接堆叠NumPy数组进行计算
            stacked_scores = np.stack(health_arrays_for_period, axis=0)
            geo_mean_values = np.prod(stacked_scores, axis=0)**(1/len(health_arrays_for_period))
            overall_bullish_health[p] = pd.Series(geo_mean_values, index=df.index)
            # 采用新的“看跌共振”计算逻辑
            # 计算每个支柱的看跌健康度 (1 - 看涨健康度)
            bearish_health_arrays_for_period = [(1 - pillar_period_health_d[key][p].values) for key in pillars]
            bearish_health_arrays_for_period.append((1 - pillar_overall_health_d['mtf_synergy'].values))
            # 堆叠并计算看跌共振分
            stacked_bearish_scores = np.stack(bearish_health_arrays_for_period, axis=0)
            geo_mean_bearish_values = np.prod(stacked_bearish_scores, axis=0)**(1/len(bearish_health_arrays_for_period))
            overall_bearish_health[p] = pd.Series(geo_mean_bearish_values, index=df.index)
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
        # print("        -> [交叉验证诊断模块 V3.2 信号逻辑精炼版] 启动...")
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
        # print("        -> 开始计算所有周期的深度交叉验证共振分...") # 修改: 简化打印信息
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
        # print("        -> 所有周期的深度交叉验证共振分计算完毕。") # 修改: 简化打印信息

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
        【V1.2 核心+奖励范式升级版】复合筹码评分模块
        - 核心升级 (本次修改):
          - [范式升级] 将 `CHIP_SCORE_TRIGGER_IGNITION` (筹码点火触发分) 的计算逻辑，
                        从脆弱的“上下文 * 核心行为”乘法模型，升级为鲁棒的“核心行为 * (1 + 上下文 * 奖励系数)”新范式。
        - 收益: 解决了因“筹码结构不够完美”而压制“趋势启动”强信号的问题，能更灵敏地捕捉各类启动机会。
        """
        print("        -> [复合筹码评分模块 V1.2 核心+奖励范式升级版] 启动...") # 更新版本号和日志
        new_scores = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        
        # --- 1. 定义归一化辅助函数与参数 ---
        p = get_params_block(self.strategy, 'chip_feature_params')
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        # 新增点火奖励系数参数
        ignition_bonus_factor = get_param_value(p.get('ignition_context_bonus_factor'), 0.8)

        def normalize(series, ascending=True):
            if series.empty:
                return pd.Series(0.0, index=df.index, dtype=np.float32)
            return series.rolling(window=norm_window, min_periods=min_periods).rank(pct=True, ascending=ascending).fillna(0.5)
            
        # --- 2. 计算 CHIP_SCORE_GATHERING_INTENSITY (逻辑不变) ---
        conc_slope_score = normalize(df.get('SLOPE_5_concentration_90pct_D', default_score), ascending=False)
        cost_slope_score = normalize(df.get('SLOPE_5_peak_cost_D', default_score), ascending=True)
        health_slope_score = normalize(df.get('SLOPE_5_chip_health_score_D', default_score), ascending=True)
        new_scores['CHIP_SCORE_GATHERING_INTENSITY'] = (conc_slope_score * cost_slope_score * health_slope_score).astype(np.float32)
        
        # --- 3. 计算 CHIP_SCORE_TRIGGER_IGNITION (逻辑升级) ---
        # 拆分为“核心行为”和“上下文”
        # 核心行为：上升共振分，代表趋势确认
        bullish_resonance_score = atomic.get('SCORE_CHIP_BULLISH_RESONANCE_S', default_score)
        # 上下文：黄金机会分，代表完美的静态结构
        prime_opp_context_score = atomic.get('CHIP_SCORE_PRIME_OPPORTUNITY_S', default_score)
        # 应用新范式
        ignition_score = bullish_resonance_score * (1 + prime_opp_context_score * ignition_bonus_factor)
        new_scores['CHIP_SCORE_TRIGGER_IGNITION'] = ignition_score.astype(np.float32)

        # --- 4. 计算 CHIP_SCORE_RISK_LONG_TERM_DISTRIBUTION (逻辑不变) ---
        bearish_resonance_score = atomic.get('SCORE_CHIP_BEARISH_RESONANCE_S', default_score)
        profit_taking_score = atomic.get('SCORE_PROFIT_TAKING_TOP_REVERSAL_S', default_score)
        instability_score = atomic.get('SCORE_LONG_TERM_INSTABILITY_TOP_REVERSAL_A', default_score)
        new_scores['CHIP_SCORE_RISK_LONG_TERM_DISTRIBUTION'] = (bearish_resonance_score * self._max_of_series(profit_taking_score, instability_score)).astype(np.float32)
        
        # --- 5. 计算 CHIP_SCORE_RISK_WORSENING_TURN (逻辑不变) ---
        conc_slope_5_negative_score = normalize(df.get('SLOPE_5_concentration_90pct_D', default_score), ascending=False)
        conc_accel_5 = normalize(df.get('ACCEL_5_concentration_90pct_D', default_score), ascending=False)
        conc_accel_21 = normalize(df.get('ACCEL_21_concentration_90pct_D', default_score), ascending=False)
        new_scores['CHIP_SCORE_RISK_WORSENING_TURN'] = (conc_slope_5_negative_score * conc_accel_5 * conc_accel_21).astype(np.float32)
        
        # --- 6. 计算 CHIP_SCORE_OPP_BREAKTHROUGH (逻辑不变) ---
        # 注意：这里的 prime_opp_score 是作为突破机会的核心，而非上下文，因此乘法合理
        price_deviation_score = normalize(df.get('price_to_peak_ratio_D', default_score), ascending=True)
        new_scores['CHIP_SCORE_OPP_BREAKTHROUGH'] = (prime_opp_context_score * price_deviation_score).astype(np.float32)
        
        # --- 7. 计算 CHIP_SCORE_RISK_COLLAPSE (逻辑不变) ---
        top_reversal_score = atomic.get('SCORE_CHIP_TOP_REVERSAL_S', default_score)
        fault_risk_score = atomic.get('SCORE_FAULT_RISK_TOP_REVERSAL_S', default_score)
        new_scores['CHIP_SCORE_RISK_COLLAPSE'] = (self._max_of_series(bearish_resonance_score, top_reversal_score, fault_risk_score)).astype(np.float32)
        
        # --- 8. 计算 CHIP_SCORE_OPP_INFLECTION (逻辑不变) ---
        bottom_reversal_score = atomic.get('SCORE_CHIP_BOTTOM_REVERSAL_S', default_score)
        capitulation_score = atomic.get('SCORE_CAPITULATION_BOTTOM_RESONANCE_S', default_score)
        long_term_capitulation_score = atomic.get('SCORE_LONG_TERM_CAPITULATION_BOTTOM_REVERSAL_S', default_score)
        new_scores['CHIP_SCORE_OPP_INFLECTION'] = (self._max_of_series(bottom_reversal_score, capitulation_score, long_term_capitulation_score)).astype(np.float32)
        
        # --- 9. 更新DataFrame ---
        df = df.assign(**new_scores)
        print(f"        -> [复合筹码评分模块 V1.2] 计算完毕，新增 {len(new_scores)} 个顶层原子评分。") 
        return df

    def diagnose_strategic_context_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.1 评分扩展版】战略级筹码上下文评分模块
        - 核心职责: 融合最长周期的筹码指标，生成描述宏观“战略吸筹”与“战略派发”的
                      顶层上下文分数。这是整个筹码情报模块的基石。
        - 核心升级(V1.1): 新增“亢奋上涨预警”评分，用于识别市场过热的宏观状态。
        - 收益: 为下游所有模块提供了最关键的宏观背景判断。
        """
        # print("        -> [战略级筹码上下文评分模块 V1.1 评分扩展版] 启动...")
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
        # print("        -> [筹码信号量化评分模块 V6.0 终极共振版] 启动...") 
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
        # print("        -> [高级筹码动态评分模块 V4.0 终极共振统一版] 启动...") 
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
        # print("        -> [筹码内部结构评分模块 V4.0 终极共振统一版] 启动...") 
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
        # print("        -> [持仓者行为评分模块 V4.0 终极共振统一版] 启动...") 
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
        【V2.2 核心+奖励范式重构版】行为与筹码融合评分模块
        - 核心重构 (本次修改):
          - [范式升级] 废除了旧的、脆弱的“多因子相乘”逻辑。
          - [新范式] 全面采用“核心行为 * (1 + 上下文 * 奖励系数)”的新范式，将“价格波动”
                      这个上下文环境从“否决项”升级为“奖励项”，使信号由核心的量价背离行为主导。
        - 收益: 极大提升了“洗盘吸筹”和“诱多派发”两大核心战术场景识别的准确性和鲁棒性。
        """
        print("        -> [行为-筹码融合评分模块 V2.2 核心+奖励范式重构版] 启动...") # 更新版本号和日志
        new_scores = {}
        p = get_params_block(self.strategy, 'fused_behavioral_chip_params', {})
        if not get_param_value(p.get('enabled'), True):
            return df
        # --- 1. 军备检查 ---
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
        # --- 2. 定义归一化辅助函数与参数 ---
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        def normalize(series, ascending=True):
            return series.rolling(window=norm_window, min_periods=min_periods).rank(pct=True, ascending=ascending).fillna(0.5)
        
        # 新增奖励系数参数
        washout_bonus = get_param_value(p.get('washout_context_bonus_factor'), 0.5)
        deceptive_bonus = get_param_value(p.get('deceptive_rally_context_bonus_factor'), 0.5)

        # --- 3. 计算“洗盘吸筹”融合分 (Washout Absorption Opportunity) ---
        # 3.1 定义核心行为分
        losers_capitulating_score = normalize(df['turnover_from_losers_ratio_D'], ascending=True)
        winners_holding_score = normalize(df['turnover_from_winners_ratio_D'], ascending=False)
        chip_improving_momentum = normalize(df['SLOPE_5_concentration_90pct_D'], ascending=False)
        chip_improving_accel = normalize(df['ACCEL_5_concentration_90pct_D'], ascending=False)
        core_absorption_score = losers_capitulating_score * winners_holding_score * chip_improving_momentum * chip_improving_accel

        # 3.2 定义上下文分 (价格下跌)
        drop_context_score = (1 - (df['pct_change_D'].fillna(0.0) - (-0.045)).abs() / 0.025).clip(0, 1).fillna(0.0)

        # 3.3 应用新范式融合生成S/A/B三级机会信号
        final_washout_score = core_absorption_score * (1 + drop_context_score * washout_bonus)
        new_scores['CHIP_SCORE_OPP_WASHOUT_ABSORPTION_S'] = final_washout_score.astype(np.float32)
        # 为了兼容，A/B级可以设为S级的不同折扣
        new_scores['CHIP_SCORE_OPP_WASHOUT_ABSORPTION_A'] = (final_washout_score * 0.7).astype(np.float32)
        new_scores['CHIP_SCORE_OPP_WASHOUT_ABSORPTION_B'] = (final_washout_score * 0.4).astype(np.float32)

        # --- 4. 计算“诱多派发”融合分 (Deceptive Rally Risk) ---
        # 4.1 定义核心风险分 (价涨质跌的“质跌”部分)
        chip_worsening_momentum = normalize(df['SLOPE_5_concentration_90pct_D'], ascending=True)
        health_worsening_momentum = normalize(df['SLOPE_5_chip_health_score_D'], ascending=False)
        chip_worsening_accel = normalize(df['ACCEL_5_concentration_90pct_D'], ascending=True)
        health_worsening_accel = normalize(df['ACCEL_5_chip_health_score_D'], ascending=False)
        core_divergence_risk = chip_worsening_momentum * health_worsening_momentum * chip_worsening_accel * health_worsening_accel

        # 4.2 定义上下文分 (价格上涨)
        rally_context_score = normalize(df['pct_change_D'], ascending=True)

        # 4.3 应用新范式融合生成S/A/B三级风险信号
        final_deceptive_score = core_divergence_risk * (1 + rally_context_score * deceptive_bonus)
        new_scores['CHIP_SCORE_RISK_DECEPTIVE_RALLY_S'] = final_deceptive_score.astype(np.float32)
        new_scores['CHIP_SCORE_RISK_DECEPTIVE_RALLY_A'] = (final_deceptive_score * 0.7).astype(np.float32)
        new_scores['CHIP_SCORE_RISK_DECEPTIVE_RALLY_B'] = (final_deceptive_score * 0.4).astype(np.float32)

        # --- 5. 为了兼容旧版，保留一个综合分数 ---
        new_scores['CHIP_SCORE_FUSED_WASHOUT_ABSORPTION'] = new_scores['CHIP_SCORE_OPP_WASHOUT_ABSORPTION_A']
        new_scores['CHIP_SCORE_FUSED_DECEPTIVE_RALLY'] = new_scores['CHIP_SCORE_RISK_DECEPTIVE_RALLY_A']
        
        # --- 6. 更新DataFrame ---
        df = df.assign(**new_scores)
        print("        -> [行为-筹码融合评分模块 V2.2 核心+奖励范式重构版] 计算完毕。") # 更新版本号
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

    def _diagnose_setup_capitulation_ready(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 · 原子状态】诊断“恐慌已弥漫”的战备(Setup)状态
        - 核心职责: 生产 `SCORE_SETUP_CAPITULATION_READY` 原子状态分。
        - 逻辑: 市场深度套牢 + 股价处于长期低位。
        - 收益: 成为一个可被全系统复用的、描述市场背景的“乐高积木”。
        """
        states = {}
        p = get_params_block(self.strategy, 'capitulation_reversal_params', {})
        norm_window = get_param_value(p.get('norm_window'), 120)
        
        # 军备检查
        required_cols = ['total_loser_rate_D', 'close_D']
        if not all(col in df.columns for col in required_cols):
            print(f"          -> [警告] 恐慌战备(Setup)诊断缺少关键数据: {required_cols}，模块已跳过！")
            return states
            
        # 因子1: 市场深度套牢
        deep_capitulation_score = self._normalize_score(df['total_loser_rate_D'], norm_window, ascending=True)
        # 因子2: 股价处于长期低位
        price_pos_yearly_score = self._normalize_score(df['close_D'], window=250, ascending=False)
        
        # 融合生成战备分
        setup_score = (deep_capitulation_score * price_pos_yearly_score).astype(np.float32)
        states['SCORE_SETUP_CAPITULATION_READY'] = setup_score
        return states

    def _diagnose_trigger_capitulation_fire(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 · 原子状态】诊断“卖压出清”的点火(Trigger)行为
        - 核心职责: 生产 `SCORE_TRIGGER_CAPITULATION_FIRE` 原子状态分。
        - 逻辑: 当日割肉盘占比极高 + 割肉盘正在加速涌出。
        - 收益: 成为一个可被全系统复用的、描述动态行为的“乐高积木”。
        """
        states = {}
        p = get_params_block(self.strategy, 'capitulation_reversal_params', {})
        norm_window = get_param_value(p.get('norm_window'), 120)
        
        # 军备检查
        required_cols = ['turnover_from_losers_ratio_D', 'ACCEL_5_turnover_from_losers_ratio_D']
        if not all(col in df.columns for col in required_cols):
            print(f"          -> [警告] 恐慌点火(Trigger)诊断缺少关键数据: {required_cols}，模块已跳过！")
            return states
            
        # 因子1: 当日割肉盘占比极高 (融合相对与绝对强度)
        relative_turnover_score = self._normalize_score(df['turnover_from_losers_ratio_D'], norm_window, ascending=True)
        k = get_param_value(p.get('logistic_k', 0.1))
        x0 = get_param_value(p.get('logistic_x0', 50.0))
        absolute_turnover_score = 1 / (1 + np.exp(-k * (df['turnover_from_losers_ratio_D'] - x0)))
        loser_turnover_score = np.maximum(relative_turnover_score, absolute_turnover_score)
        
        # 因子2: 割肉盘正在加速涌出
        loser_turnover_accel_score = self._normalize_score(df['ACCEL_5_turnover_from_losers_ratio_D'], norm_window, ascending=True)
        
        # 融合生成点火分
        trigger_score = (loser_turnover_score * loser_turnover_accel_score).astype(np.float32)
        states['SCORE_TRIGGER_CAPITULATION_FIRE'] = trigger_score
        return states

    def _synthesize_playbook_capitulation_reversal(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 · 剧本合成器】合成“恐慌盘投降反转”剧本
        - 核心职责: 消费“战备”和“点火”两个原子状态，合成最终的剧本分。
        - 逻辑: 昨日战备就绪，今日点火触发。
        """
        states = {}
        # 军备检查：确保上游的原子状态已生成
        required_cols = ['SCORE_SETUP_CAPITULATION_READY', 'SCORE_TRIGGER_CAPITULATION_FIRE']
        if not all(col in df.columns for col in required_cols):
            print(f"          -> [警告] 恐慌反转剧本合成器缺少上游原子状态: {required_cols}，模块已跳过！")
            return states
            
        setup_score = df['SCORE_SETUP_CAPITULATION_READY']
        trigger_score = df['SCORE_TRIGGER_CAPITULATION_FIRE']
        
        # 剧本合成：昨日战备，今日点火
        was_setup_yesterday = setup_score.shift(1).fillna(0.0)
        final_score = (was_setup_yesterday * trigger_score).astype(np.float32)
        
        states['SCORE_CHIP_PLAYBOOK_CAPITULATION_REVERSAL'] = final_score
        return states











