# 文件: strategies/trend_following/intelligence/chip_intelligence.py
# 筹码情报模块
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following.utils import transmute_health_to_ultimate_signals, get_params_block, get_param_value, normalize_score, normalize_to_bipolar, calculate_holographic_dynamics

class ChipIntelligence:
    def __init__(self, strategy_instance, dynamic_thresholds: Dict):
        """
        初始化筹码情报模块。
        :param strategy_instance: 策略主实例的引用。
        :param dynamic_thresholds: 动态阈值字典。
        """
        self.strategy = strategy_instance
        self.dynamic_thresholds = dynamic_thresholds

    def run_chip_intelligence_command(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V603.0 · 四公理版】筹码情报最高司令部
        - 核心升级: 新增调用“公理四：筹码峰健康度”诊断引擎，形成四足鼎立的分析框架。
        """
        all_chip_states = {}
        periods = [1, 5, 13, 21, 55]
        concentration_scores = self._diagnose_concentration_dynamics(df, periods)
        all_chip_states['SCORE_CHIP_MTF_CONCENTRATION'] = concentration_scores
        accumulation_scores = self._diagnose_main_force_action(df, periods)
        all_chip_states['SCORE_CHIP_MTF_ACCUMULATION'] = accumulation_scores
        power_transfer_scores = self._diagnose_power_transfer(df, periods)
        all_chip_states['SCORE_CHIP_MTF_POWER_TRANSFER'] = power_transfer_scores
        # 新增调用第四公理诊断引擎
        peak_integrity_scores = self._diagnose_peak_integrity_dynamics(df, periods)
        all_chip_states['SCORE_CHIP_MTF_PEAK_INTEGRITY'] = peak_integrity_scores
        ultimate_signals = self._synthesize_ultimate_signals(
            df,
            concentration_scores,
            accumulation_scores,
            power_transfer_scores,
            peak_integrity_scores
        )

        all_chip_states.update(ultimate_signals)
        accumulation_potential_states = self.diagnose_accumulation_playbooks(df)
        all_chip_states.update(accumulation_potential_states)
        capitulation_potential_states = self.diagnose_capitulation_reversal_potential(df)
        all_chip_states.update(capitulation_potential_states)
        return all_chip_states

    def _synthesize_ultimate_signals(self, df: pd.DataFrame, concentration: Dict[int, pd.Series], accumulation: Dict[int, pd.Series], power_transfer: Dict[int, pd.Series], peak_integrity: Dict[int, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V4.4 · 忒弥斯天平版】终极信号合成器
        - 核心修复: 签署“忒弥斯的正义天平”协议，在对权重字典求和前，增加类型检查，过滤掉 "description" 等非数字值，彻底修复因配置污染导致的潜在 TypeError。
        """
        states = {}
        periods = sorted(concentration.keys())
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {1: 0.1, 5: 0.4, 13: 0.3, 21: 0.15, 55: 0.05})
        axiom_weights = get_param_value(p_conf.get('axiom_weights'), {'concentration': 0.3, 'accumulation': 0.3, 'power_transfer': 0.25, 'peak_integrity': 0.15})
        norm_window = 55
        bipolar_health_by_period = {}
        for p in periods:
            conc_score = concentration.get(p, pd.Series(0.0, index=df.index))
            acc_score = accumulation.get(p, pd.Series(0.0, index=df.index))
            pow_score = power_transfer.get(p, pd.Series(0.0, index=df.index))
            peak_score = peak_integrity.get(p, pd.Series(0.0, index=df.index))
            bipolar_health_by_period[p] = (
                conc_score * axiom_weights.get('concentration', 0.3) +
                acc_score * axiom_weights.get('accumulation', 0.3) +
                pow_score * axiom_weights.get('power_transfer', 0.25) +
                peak_score * axiom_weights.get('peak_integrity', 0.15)
            ).clip(-1, 1)
        bullish_scores_by_period = {p: score.clip(0, 1) for p, score in bipolar_health_by_period.items()}
        bearish_scores_by_period = {p: (score.clip(-1, 0) * -1) for p, score in bipolar_health_by_period.items()}
        bullish_resonance = pd.Series(0.0, index=df.index)
        bearish_resonance = pd.Series(0.0, index=df.index)
        # [代码修改开始] 增加类型检查，过滤掉 "description" 等非数字值
        numeric_weights = {k: v for k, v in tf_weights.items() if isinstance(v, (int, float))}
        total_weight = sum(numeric_weights.values())
        # [代码修改结束]
        if total_weight > 0:
            for p in periods:
                # [代码修改开始] 使用净化后的 numeric_weights
                weight = numeric_weights.get(str(p), 0) / total_weight # 确保使用字符串键
                # [代码修改结束]
                bullish_resonance += bullish_scores_by_period.get(p, 0.0) * weight
                bearish_resonance += bearish_scores_by_period.get(p, 0.0) * weight
        states['SCORE_CHIP_BULLISH_RESONANCE'] = bullish_resonance.fillna(0).clip(0, 1).astype(np.float32)
        states['SCORE_CHIP_BEARISH_RESONANCE'] = bearish_resonance.fillna(0).clip(0, 1).astype(np.float32)
        bottom_reversal_scores = {}
        top_reversal_scores = {}
        for p in periods:
            context_p = periods[periods.index(p) + 1] if periods.index(p) + 1 < len(periods) else p
            bottom_reversal_scores[p] = self._calculate_holographic_divergence(bullish_scores_by_period.get(p, pd.Series(0.0, index=df.index)), 1, p, context_p) # 修正背离计算周期
            top_reversal_scores[p] = self._calculate_holographic_divergence(bearish_scores_by_period.get(p, pd.Series(0.0, index=df.index)), 1, p, context_p) # 修正背离计算周期
        bottom_reversal_divergence = pd.Series(0.0, index=df.index)
        top_reversal_divergence = pd.Series(0.0, index=df.index)
        if total_weight > 0:
            for p in periods:
                # [代码修改开始] 使用净化后的 numeric_weights
                weight = numeric_weights.get(str(p), 0) / total_weight # 确保使用字符串键
                # [代码修改结束]
                bottom_reversal_divergence += bottom_reversal_scores.get(p, 0.0) * weight
                top_reversal_divergence += top_reversal_scores.get(p, 0.0) * weight
        states['SCORE_CHIP_BOTTOM_REVERSAL'] = bottom_reversal_divergence.clip(0, 1).astype(np.float32)
        states['SCORE_CHIP_TOP_REVERSAL'] = top_reversal_divergence.clip(0, 1).astype(np.float32)
        tactical_reversal = (bullish_resonance * 0.5).astype(np.float32)
        states['SCORE_CHIP_TACTICAL_REVERSAL'] = tactical_reversal
        p = 5
        cost_divergence_score = normalize_score(df.get(f'SLOPE_{p}_cost_divergence_D', pd.Series(0.0, index=df.index)), df.index, norm_window, ascending=True)
        loser_turnover_up = normalize_score(df.get(f'SLOPE_{p}_retail_capitulation_distribution_D', pd.Series(0.0, index=df.index)), df.index, norm_window, ascending=True)
        transfer_to_main_force_evidence = (cost_divergence_score * loser_turnover_up)**0.5
        cost_convergence_score = normalize_score(df.get(f'SLOPE_{p}_cost_divergence_D', pd.Series(0.0, index=df.index)), df.index, norm_window, ascending=False)
        loser_turnover_down = normalize_score(df.get(f'SLOPE_{p}_retail_capitulation_distribution_D', pd.Series(0.0, index=df.index)), df.index, norm_window, ascending=False)
        transfer_to_retail_evidence = (cost_convergence_score * loser_turnover_down)**0.5
        transfer_snapshot = (transfer_to_main_force_evidence - transfer_to_retail_evidence).astype(np.float32)
        distribution_strength = (transfer_snapshot.clip(-1, 0) * -1).astype(np.float32)
        hades_trap_score = (states['SCORE_CHIP_BOTTOM_REVERSAL'] * distribution_strength).clip(0, 1)
        states['SCORE_CHIP_HADES_TRAP'] = hades_trap_score.astype(np.float32)
        return states

    def _diagnose_concentration_dynamics(self, df: pd.DataFrame, periods: list) -> Dict[int, pd.Series]:
        """
        【V6.1 · 类型防御版】核心公理一：诊断筹码“聚散”的动态
        - 核心修复: 在获取数据时，确保即使列不存在，返回的也是一个填充了默认值的pd.Series，而不是一个裸的数值(如0)，
                      从根本上解决了 'int' object has no attribute 'isnull' 的运行时错误。
        """
        scores = {}
        sorted_periods = sorted(periods)
        for i, p in enumerate(sorted_periods):
            context_p = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p
            # 使用 pd.Series 包装默认值，确保数据类型安全
            # --- 看涨证据 ---
            bullish_evidence_static = df.get('concentration_increase_by_support_D', pd.Series(0, index=df.index)) + df.get('concentration_increase_by_chasing_D', pd.Series(0, index=df.index))
            bullish_evidence_slope = df.get(f'SLOPE_{p}_concentration_increase_by_support_D', pd.Series(0, index=df.index)) + df.get(f'SLOPE_{p}_concentration_increase_by_chasing_D', pd.Series(0, index=df.index))
            bullish_evidence_accel = df.get(f'ACCEL_{p}_concentration_increase_by_support_D', pd.Series(0, index=df.index)) + df.get(f'ACCEL_{p}_concentration_increase_by_chasing_D', pd.Series(0, index=df.index))
            # --- 看跌证据 ---
            bearish_evidence_static = df.get('concentration_decrease_by_distribution_D', pd.Series(0, index=df.index)) + df.get('concentration_decrease_by_capitulation_D', pd.Series(0, index=df.index))
            bearish_evidence_slope = df.get(f'SLOPE_{p}_concentration_decrease_by_distribution_D', pd.Series(0, index=df.index)) + df.get(f'SLOPE_{p}_concentration_decrease_by_capitulation_D', pd.Series(0, index=df.index))
            bearish_evidence_accel = df.get(f'ACCEL_{p}_concentration_decrease_by_distribution_D', pd.Series(0, index=df.index)) + df.get(f'ACCEL_{p}_concentration_decrease_by_capitulation_D', pd.Series(0, index=df.index))
            
            # 战术层 (p)
            tactical_bullish_static_score = normalize_score(bullish_evidence_static, df.index, p, ascending=True)
            tactical_bullish_slope_score = normalize_score(bullish_evidence_slope, df.index, p, ascending=True)
            tactical_bullish_accel_score = normalize_score(bullish_evidence_accel, df.index, p, ascending=True)
            tactical_bullish_quality = (tactical_bullish_static_score * tactical_bullish_slope_score * tactical_bullish_accel_score)**(1/3)
            # 上下文层 (context_p)
            context_bullish_static_score = normalize_score(bullish_evidence_static, df.index, context_p, ascending=True)
            context_bullish_slope_score = normalize_score(bullish_evidence_slope, df.index, context_p, ascending=True)
            context_bullish_accel_score = normalize_score(bullish_evidence_accel, df.index, context_p, ascending=True)
            context_bullish_quality = (context_bullish_static_score * context_bullish_slope_score * context_bullish_accel_score)**(1/3)
            # 融合
            final_bullish_quality = (tactical_bullish_quality * context_bullish_quality)**0.5
            # 战术层 (p)
            tactical_bearish_static_score = normalize_score(bearish_evidence_static, df.index, p, ascending=True)
            tactical_bearish_slope_score = normalize_score(bearish_evidence_slope, df.index, p, ascending=True)
            tactical_bearish_accel_score = normalize_score(bearish_evidence_accel, df.index, p, ascending=True)
            tactical_bearish_quality = (tactical_bearish_static_score * tactical_bearish_slope_score * tactical_bearish_accel_score)**(1/3)
            # 上下文层 (context_p)
            context_bearish_static_score = normalize_score(bearish_evidence_static, df.index, context_p, ascending=True)
            context_bearish_slope_score = normalize_score(bearish_evidence_slope, df.index, context_p, ascending=True)
            context_bearish_accel_score = normalize_score(bearish_evidence_accel, df.index, context_p, ascending=True)
            context_bearish_quality = (context_bearish_static_score * context_bearish_slope_score * context_bearish_accel_score)**(1/3)
            # 融合
            final_bearish_quality = (tactical_bearish_quality * context_bearish_quality)**0.5
            # 生成双极快照分
            concentration_quality_snapshot = (final_bullish_quality - final_bearish_quality).astype(np.float32)
            holographic_divergence = self._calculate_holographic_divergence(concentration_quality_snapshot, 1, p, p * 2)
            dynamic_concentration_score = self._perform_chip_relational_meta_analysis(
                df, concentration_quality_snapshot, p, holographic_divergence
            )
            scores[p] = dynamic_concentration_score
        return scores

    def _diagnose_main_force_action(self, df: pd.DataFrame, periods: list) -> Dict[int, pd.Series]:
        """
        【V6.1 · 类型防御版】核心公理二：诊断主力“吸筹与派发”
        - 核心修复: 在获取数据时，确保即使列不存在，返回的也是一个填充了默认值的pd.Series，而不是一个裸的数值(如0)。
        """
        scores = {}
        sorted_periods = sorted(periods)
        for i, p in enumerate(sorted_periods):
            context_p = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p
            # 使用 pd.Series 包装默认值，确保数据类型安全
            # --- 吸筹证据 ---
            accumulation_static = df.get('main_force_suppressive_accumulation_D', pd.Series(0, index=df.index)) + df.get('main_force_chasing_accumulation_D', pd.Series(0, index=df.index))
            accumulation_slope = df.get(f'SLOPE_{p}_main_force_suppressive_accumulation_D', pd.Series(0, index=df.index)) + df.get(f'SLOPE_{p}_main_force_chasing_accumulation_D', pd.Series(0, index=df.index))
            accumulation_accel = df.get(f'ACCEL_{p}_main_force_suppressive_accumulation_D', pd.Series(0, index=df.index)) + df.get(f'ACCEL_{p}_main_force_chasing_accumulation_D', pd.Series(0, index=df.index))
            # --- 派发证据 ---
            distribution_static = df.get('main_force_rally_distribution_D', pd.Series(0, index=df.index)) + df.get('main_force_capitulation_distribution_D', pd.Series(0, index=df.index))
            distribution_slope = df.get(f'SLOPE_{p}_main_force_rally_distribution_D', pd.Series(0, index=df.index)) + df.get(f'SLOPE_{p}_main_force_capitulation_distribution_D', pd.Series(0, index=df.index))
            distribution_accel = df.get(f'ACCEL_{p}_main_force_rally_distribution_D', pd.Series(0, index=df.index)) + df.get(f'ACCEL_{p}_main_force_capitulation_distribution_D', pd.Series(0, index=df.index))
            
            # 战术层
            tactical_acc_static = normalize_score(accumulation_static, df.index, p, ascending=True)
            tactical_acc_slope = normalize_score(accumulation_slope, df.index, p, ascending=True)
            tactical_acc_accel = normalize_score(accumulation_accel, df.index, p, ascending=True)
            tactical_acc_quality = (tactical_acc_static * tactical_acc_slope * tactical_acc_accel)**(1/3)
            # 上下文层
            context_acc_static = normalize_score(accumulation_static, df.index, context_p, ascending=True)
            context_acc_slope = normalize_score(accumulation_slope, df.index, context_p, ascending=True)
            context_acc_accel = normalize_score(accumulation_accel, df.index, context_p, ascending=True)
            context_acc_quality = (context_acc_static * context_acc_slope * context_acc_accel)**(1/3)
            accumulation_evidence = (tactical_acc_quality * context_acc_quality)**0.5
            # 战术层
            tactical_dist_static = normalize_score(distribution_static, df.index, p, ascending=True)
            tactical_dist_slope = normalize_score(distribution_slope, df.index, p, ascending=True)
            tactical_dist_accel = normalize_score(distribution_accel, df.index, p, ascending=True)
            tactical_dist_quality = (tactical_dist_static * tactical_dist_slope * tactical_dist_accel)**(1/3)
            # 上下文层
            context_dist_static = normalize_score(distribution_static, df.index, context_p, ascending=True)
            context_dist_slope = normalize_score(distribution_slope, df.index, context_p, ascending=True)
            context_dist_accel = normalize_score(distribution_accel, df.index, context_p, ascending=True)
            context_dist_quality = (context_dist_static * context_dist_slope * context_dist_accel)**(1/3)
            distribution_evidence = (tactical_dist_quality * context_dist_quality)**0.5
            action_snapshot = (accumulation_evidence - distribution_evidence).astype(np.float32)
            holographic_divergence = self._calculate_holographic_divergence(action_snapshot, 1, p, p * 2)
            dynamic_action_score = self._perform_chip_relational_meta_analysis(
                df, action_snapshot, p, holographic_divergence
            )
            scores[p] = dynamic_action_score
        return scores

    def _diagnose_power_transfer(self, df: pd.DataFrame, periods: list) -> Dict[int, pd.Series]:
        """
        【V5.2 · 前线换装与健壮性修复版】核心公理三：诊断筹码“转移方向”
        - 核心修复: 1. 废弃已失效的 `short_term_..._ratio_D` 指标，换装为新式高保真指标。
                      2. 全面将 df.get(..., 0) 的默认值升级为 pd.Series(0.0, index=df.index)，实现“装甲加固”。
        """
        scores = {}
        sorted_periods = sorted(periods)
        for i, p in enumerate(sorted_periods):
            context_p = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p
            # 全面加固 df.get() 调用并换装新指标
            transfer_to_main_static = df.get('retail_capitulation_distribution_D', pd.Series(0.0, index=df.index)) + df.get('long_term_despair_selling_ratio_D', pd.Series(0.0, index=df.index))
            transfer_to_main_slope = df.get(f'SLOPE_{p}_retail_capitulation_distribution_D', pd.Series(0.0, index=df.index)) + df.get(f'SLOPE_{p}_long_term_despair_selling_ratio_D', pd.Series(0.0, index=df.index))
            transfer_to_main_accel = df.get(f'ACCEL_{p}_retail_capitulation_distribution_D', pd.Series(0.0, index=df.index)) + df.get(f'ACCEL_{p}_long_term_despair_selling_ratio_D', pd.Series(0.0, index=df.index))
            transfer_to_retail_static = df.get('profit_taking_urgency_D', pd.Series(0.0, index=df.index)) + df.get('long_term_chips_unlocked_ratio_D', pd.Series(0.0, index=df.index))
            transfer_to_retail_slope = df.get(f'SLOPE_{p}_profit_taking_urgency_D', pd.Series(0.0, index=df.index)) + df.get(f'SLOPE_{p}_long_term_chips_unlocked_ratio_D', pd.Series(0.0, index=df.index))
            transfer_to_retail_accel = df.get(f'ACCEL_{p}_profit_taking_urgency_D', pd.Series(0.0, index=df.index)) + df.get(f'ACCEL_{p}_long_term_chips_unlocked_ratio_D', pd.Series(0.0, index=df.index))
    
            tactical_main_static = normalize_score(transfer_to_main_static, df.index, p, ascending=True)
            tactical_main_slope = normalize_score(transfer_to_main_slope, df.index, p, ascending=True)
            tactical_main_accel = normalize_score(transfer_to_main_accel, df.index, p, ascending=True)
            tactical_main_quality = (tactical_main_static * tactical_main_slope * tactical_main_accel)**(1/3)
            context_main_static = normalize_score(transfer_to_main_static, df.index, context_p, ascending=True)
            context_main_slope = normalize_score(transfer_to_main_slope, df.index, context_p, ascending=True)
            context_main_accel = normalize_score(transfer_to_main_accel, df.index, context_p, ascending=True)
            context_main_quality = (context_main_static * context_main_slope * context_main_accel)**(1/3)
            transfer_to_main_force_evidence = (tactical_main_quality * context_main_quality)**0.5
            tactical_retail_static = normalize_score(transfer_to_retail_static, df.index, p, ascending=True)
            tactical_retail_slope = normalize_score(transfer_to_retail_slope, df.index, p, ascending=True)
            tactical_retail_accel = normalize_score(transfer_to_retail_accel, df.index, p, ascending=True)
            tactical_retail_quality = (tactical_retail_static * tactical_retail_slope * tactical_retail_accel)**(1/3)
            context_retail_static = normalize_score(transfer_to_retail_static, df.index, context_p, ascending=True)
            context_retail_slope = normalize_score(transfer_to_retail_slope, df.index, context_p, ascending=True)
            context_retail_accel = normalize_score(transfer_to_retail_accel, df.index, context_p, ascending=True)
            context_retail_quality = (context_retail_static * context_retail_slope * context_retail_accel)**(1/3)
            transfer_to_retail_evidence = (tactical_retail_quality * context_retail_quality)**0.5
            transfer_snapshot = (transfer_to_main_force_evidence - transfer_to_retail_evidence).astype(np.float32)
            holographic_divergence = self._calculate_holographic_divergence(transfer_snapshot, 1, p, p * 2)
            dynamic_transfer_score = self._perform_chip_relational_meta_analysis(
                df, transfer_snapshot, p, holographic_divergence
            )
            scores[p] = dynamic_transfer_score
        return scores

    def _diagnose_peak_integrity_dynamics(self, df: pd.DataFrame, periods: list) -> Dict[int, pd.Series]:
        """
        【V1.0 · 新增】核心公理四：诊断筹码峰“健康度”的动态
        - 核心逻辑: 评估核心筹码阵地（单峰密集区）的稳固性、控制力及攻防状态。
        """
        #
        scores = {}
        sorted_periods = sorted(periods)
        for i, p in enumerate(sorted_periods):
            context_p = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p
            # --- 看涨证据：坚固的堡垒 ---
            control_score = normalize_score(df.get('peak_control_ratio_D', pd.Series(0.0, index=df.index)), df.index, p, ascending=True)
            stability_score = normalize_score(df.get('peak_stability_D', pd.Series(0.0, index=df.index)), df.index, p, ascending=True)
            defense_score = normalize_score(df.get('peak_defense_intensity_D', pd.Series(0.0, index=df.index)), df.index, p, ascending=True)
            proximity_score = normalize_score(df.get('price_to_peak_ratio_D', pd.Series(1.0, index=df.index)), df.index, p, ascending=False) # 价格离得越近越好
            bullish_evidence = (control_score * stability_score * defense_score * proximity_score)**(1/4)
            # --- 看跌证据：崩溃的阵地 ---
            # 看跌证据是看涨证据的反面
            bearish_evidence = 1.0 - bullish_evidence
            # --- 生成双极快照分 ---
            peak_integrity_snapshot = (bullish_evidence - bearish_evidence).astype(np.float32)
            holographic_divergence = self._calculate_holographic_divergence(peak_integrity_snapshot, 1, p, p * 2)
            dynamic_peak_score = self._perform_chip_relational_meta_analysis(
                df, peak_integrity_snapshot, p, holographic_divergence
            )
            scores[p] = dynamic_peak_score
        return scores
        

    def _perform_chip_relational_meta_analysis(self, df: pd.DataFrame, snapshot_score: pd.Series, meta_window: int, holographic_divergence_score: pd.Series) -> pd.Series:
        """
        【V6.0 · 阿瑞斯之怒协议版】筹码专用的关系元分析核心引擎
        - 核心革命: 废除“冥王之眼”乘法模型，全面升级为与微观行为引擎一致的“阿瑞斯之怒”加法模型。
                      最终得分 = (状态*权重) + (速度*权重) + (加速度*权重) + (背离*权重)
        - 升级意义: 新模型更侧重于动态变化，即使状态分较低，只要速度、加速度或背离足够强，也能产生高分，
                      从而更敏锐地捕捉到趋势的“拐点”。
        """
        # 全面升级为“阿瑞斯之怒”加法模型
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        p_meta = get_param_value(p_conf.get('relational_meta_analysis_params'), {})
        # 新的加法模型权重
        w_state = get_param_value(p_meta.get('state_weight'), 0.2) # 降低状态权重
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.3)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.3)
        w_holographic = get_param_value(p_meta.get('holographic_weight'), 0.2) # 将背离分作为第四维度
        norm_window = 55
        bipolar_sensitivity = 1.0
        # 维度一：状态分 (State Score) - 范围 [0, 1]
        state_score = snapshot_score.clip(0, 1)
        # 维度二：速度分 (Velocity Score) - 范围 [-1, 1]
        relationship_trend = snapshot_score.diff(meta_window).fillna(0)
        velocity_score = normalize_to_bipolar(relationship_trend, df.index, norm_window, bipolar_sensitivity)
        # 维度三：加速度分 (Acceleration Score) - 范围 [-1, 1]
        relationship_accel = relationship_trend.diff(meta_window).fillna(0)
        acceleration_score = normalize_to_bipolar(relationship_accel, df.index, norm_window, bipolar_sensitivity)
        # 维度四：全息背离分 (Holographic Divergence Score) - 范围 [-1, 1]
        # 确保背离分也是双极性的
        holographic_score = holographic_divergence_score.clip(-1, 1)
        # 终极融合：从乘法调制升级为四维加法赋权
        final_score = (
            state_score * w_state +
            velocity_score.clip(0, 1) * w_velocity + # 看涨信号只取正向速度
            acceleration_score.clip(0, 1) * w_acceleration + # 看涨信号只取正向加速度
            holographic_score.clip(0, 1) * w_holographic # 看涨信号只取正向背离
        ).clip(0, 1)
        
        return final_score.astype(np.float32)

    def _calculate_holographic_divergence(self, series: pd.Series, short_p: int, long_p: int, norm_window: int) -> pd.Series:
        """
        【V1.0 · 新增 · 冥王之眼】全息背离计算引擎
        - 战略意义: 洞察多时间维度的“结构性背离”，输出一个[-1, 1]的双极性背离分数。
        - 正分: 看涨背离 (短期趋势强于长期趋势)。
        - 负分: 看跌背离 (短期趋势弱于长期趋势)。
        """
        # 新增方法
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

    def _calculate_ma_trend_context(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V2.1 · 类型安全版】计算均线趋势上下文分数
        - 核心修复: 在处理从配置中读取的权重字典时，增加了对非数字类型值的过滤。
                      这可以防止 'description' 等说明性字段污染权重数组，从根本上解决了
                      因类型不匹配导致的 'ufunc 'add' did not contain a loop' 错误。
        """
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        weights = get_param_value(p_conf.get('ma_trend_context_weights'), {
            'alignment': 0.3, 'velocity': 0.2, 'acceleration': 0.2, 'meta_dynamics': 0.3
        })
        norm_window = 55
        ma_cols = [f'EMA_{p}_D' for p in periods if f'EMA_{p}_D' in df.columns]
        if len(ma_cols) < 2:
            return pd.Series(0.5, index=df.index, dtype=np.float32)
        ma_values = np.stack([df[col].values for col in ma_cols], axis=0)
        alignment_bools = ma_values[:-1] > ma_values[1:]
        alignment_health = np.mean(alignment_bools, axis=0) if alignment_bools.size > 0 else np.full(len(df.index), 0.5)
        slope_cols = [f'SLOPE_{p}_EMA_{p}_D' for p in periods if f'SLOPE_{p}_EMA_{p}_D' in df.columns]
        if slope_cols:
            slope_values = np.stack([normalize_score(df[col], df.index, norm_window) for col in slope_cols], axis=0)
            velocity_health = np.mean(slope_values, axis=0)
        else:
            velocity_health = np.full(len(df.index), 0.5)
        accel_cols = [f'ACCEL_{p}_EMA_{p}_D' for p in periods if f'ACCEL_{p}_EMA_{p}_D' in df.columns]
        if accel_cols:
            accel_values = np.stack([normalize_score(df[col], df.index, norm_window) for col in accel_cols], axis=0)
            acceleration_health = np.mean(accel_values, axis=0)
        else:
            acceleration_health = np.full(len(df.index), 0.5)
        meta_dynamics_cols = [
            'SLOPE_5_EMA_55_D', 'SLOPE_13_EMA_89_D', 'SLOPE_21_EMA_144_D'
        ]
        valid_meta_cols = [col for col in meta_dynamics_cols if col in df.columns]
        if valid_meta_cols:
            meta_values = np.stack([normalize_score(df[col], df.index, norm_window) for col in valid_meta_cols], axis=0)
            meta_dynamics_health = np.mean(meta_values, axis=0)
        else:
            meta_dynamics_health = np.full(len(df.index), 0.5)
        scores = np.stack([alignment_health, velocity_health, acceleration_health, meta_dynamics_health], axis=0)
        # 增加类型过滤，确保只处理数字类型的权重值
        numeric_weights = {k: v for k, v in weights.items() if isinstance(v, (int, float))}
        weights_array = np.array(list(numeric_weights.values()))
        if weights_array.sum() == 0: # 增加对权重和为0的保护
            return pd.Series(0.5, index=df.index, dtype=np.float32)
        weights_array /= weights_array.sum()
        final_score_values = np.prod(scores ** weights_array[:, np.newaxis], axis=0)
        return pd.Series(final_score_values, index=df.index, dtype=np.float32)

    # ==============================================================================
    # 以下为保留的、具有特殊战术意义的“剧本”诊断模块
    # ==============================================================================

    def diagnose_accumulation_playbooks(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.1 · 健壮性修复版】诊断“吸筹”相关的战术剧本
        - 核心修复: 全面将 df.get(..., 0) 的默认值升级为 pd.Series(0.0, index=df.index)，
                      防止因上游指标缺失导致的类型错误，实现“装甲加固”。
        """
        states = {}
        periods = [1, 5, 13, 21, 55]
        sorted_periods = sorted(periods)
        rally_scores_by_period = {}
        for i, p_tactical in enumerate(sorted_periods):
            p_context = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p_tactical
            # 全面加固 df.get() 调用
            tactical_retail_chasing = normalize_score(df.get('retail_chasing_accumulation_D', pd.Series(0.0, index=df.index)), df.index, p_tactical, ascending=True)
            tactical_main_force_not_distributing = 1.0 - normalize_score(df.get('main_force_rally_distribution_D', pd.Series(0.0, index=df.index)), df.index, p_tactical, ascending=True)
            context_retail_chasing = normalize_score(df.get('retail_chasing_accumulation_D', pd.Series(0.0, index=df.index)), df.index, p_context, ascending=True)
            context_main_force_not_distributing = 1.0 - normalize_score(df.get('main_force_rally_distribution_D', pd.Series(0.0, index=df.index)), df.index, p_context, ascending=True)
    
            fused_retail_chasing = (tactical_retail_chasing * context_retail_chasing)**0.5
            fused_main_force_not_distributing = (tactical_main_force_not_distributing * context_main_force_not_distributing)**0.5
            rally_snapshot_score = (fused_retail_chasing * fused_main_force_not_distributing)**0.5
            holographic_divergence = self._calculate_holographic_divergence(rally_snapshot_score, p_tactical, p_context, p_context * 2)
            rally_scores_by_period[p_tactical] = self._perform_chip_relational_meta_analysis(df, rally_snapshot_score, p_tactical, holographic_divergence)
        tf_weights = {1: 0.1, 5: 0.4, 13: 0.3, 21: 0.15, 55: 0.05}
        final_fused_score = pd.Series(0.0, index=df.index)
        total_weight = sum(tf_weights.get(p, 0) for p in periods)
        if total_weight > 0:
            for p_tactical in periods:
                weight = tf_weights.get(p_tactical, 0) / total_weight
                final_fused_score += rally_scores_by_period.get(p_tactical, 0.0) * weight
        # 全面加固 df.get() 调用
        suppressive_accumulation = normalize_score(df.get('main_force_suppressive_accumulation_D', pd.Series(0.0, index=df.index)), df.index, 55, ascending=True)

        true_accumulation_score = np.maximum(final_fused_score, suppressive_accumulation)
        states['SCORE_CHIP_TRUE_ACCUMULATION'] = true_accumulation_score.clip(0, 1).astype(np.float32)
        states['SCORE_CHIP_PB_RALLY_ACCUMULATION'] = final_fused_score.clip(0, 1).astype(np.float32)
        return states

    def diagnose_capitulation_reversal_potential(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V4.2 · 阿波罗战车升级版】诊断“恐慌投降反转”的潜力
        - 核心升级: 调用全新的、基于四维评估的 `_calculate_ma_trend_context` 方法，
                      以获得对熊市背景更精准的动态评估。
        """
        states = {}
        required_cols = ['total_loser_rate_D', 'close_D', 'retail_capitulation_distribution_D']
        if any(col not in df.columns for col in required_cols):
            states['SCORE_CHIP_CONTEXT_CAPITULATION_POTENTIAL'] = pd.Series(0.0, index=df.index)
            return states
        periods = [5, 13, 21, 55]
        sorted_periods = sorted(periods)
        capitulation_scores_by_period = {}
        # 调用全新的四维趋势上下文评估引擎
        bearish_ma_context = 1 - self._calculate_ma_trend_context(df, [5, 13, 21, 55])

        for i, p_tactical in enumerate(sorted_periods):
            p_context = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p_tactical
            tactical_deep_cap = normalize_score(df['total_loser_rate_D'], df.index, p_tactical, ascending=True)
            tactical_price_lows = 1.0 - normalize_score(df['close_D'], df.index, window=p_tactical, ascending=True)
            tactical_loser_turnover = normalize_score(df.get('retail_capitulation_distribution_D', pd.Series(0.0, index=df.index)), df.index, p_tactical, ascending=True)
            context_deep_cap = normalize_score(df['total_loser_rate_D'], df.index, p_context, ascending=True)
            context_price_lows = 1.0 - normalize_score(df['close_D'], df.index, window=p_context, ascending=True)
            context_loser_turnover = normalize_score(df.get('retail_capitulation_distribution_D', pd.Series(0.0, index=df.index)), df.index, p_context, ascending=True)
            fused_deep_cap = (tactical_deep_cap * context_deep_cap)**0.5
            fused_price_lows = (tactical_price_lows * context_price_lows)**0.5
            fused_loser_turnover = (tactical_loser_turnover * context_loser_turnover)**0.5
            snapshot_score = (fused_deep_cap * fused_price_lows * fused_loser_turnover * bearish_ma_context).astype(np.float32)
            holographic_divergence = self._calculate_holographic_divergence(snapshot_score, p_tactical, p_context, p_context * 2)
            capitulation_scores_by_period[p_tactical] = self._perform_chip_relational_meta_analysis(df, snapshot_score, p_tactical, holographic_divergence)
        tf_weights = {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}
        final_fused_score = pd.Series(0.0, index=df.index)
        total_weight = sum(tf_weights.get(p, 0) for p in periods)
        if total_weight > 0:
            for p_tactical in periods:
                weight = tf_weights.get(p_tactical, 0) / total_weight
                final_fused_score += capitulation_scores_by_period.get(p_tactical, 0.0) * weight
        states['SCORE_CHIP_CONTEXT_CAPITULATION_POTENTIAL'] = final_fused_score.clip(0, 1).astype(np.float32)
        return states


