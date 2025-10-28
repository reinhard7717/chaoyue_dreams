# 文件: strategies/trend_following/intelligence/chip_intelligence.py
# 筹码情报模块
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following import utils
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
        【V607.0 · 主力抢筹版】筹码情报最高司令部
        - 核心扩展: 新增对“锁仓抢筹”信号的诊断，完善底部行为分析。
        """
        all_chip_states = {}
        periods = [1, 5, 13, 21, 55]
        concentration_scores = self._diagnose_concentration_dynamics(df, periods)
        all_chip_states['SCORE_CHIP_MTF_CONCENTRATION'] = concentration_scores
        accumulation_scores = self._diagnose_main_force_action(df, periods)
        all_chip_states['SCORE_CHIP_MTF_ACCUMULATION'] = accumulation_scores
        power_transfer_scores = self._diagnose_power_transfer(df, periods)
        all_chip_states['SCORE_CHIP_MTF_POWER_TRANSFER'] = power_transfer_scores
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
        lockdown_states = self.diagnose_bottom_accumulation_lockdown(df, all_chip_states)
        all_chip_states.update(lockdown_states)
        # [代码新增开始]
        # 新增调用：诊断主力抢筹信号
        scramble_states = self.diagnose_lockdown_scramble(df, all_chip_states)
        all_chip_states.update(scramble_states)
        # [代码新增结束]
        return all_chip_states

    def diagnose_bottom_accumulation_lockdown(self, df: pd.DataFrame, current_chip_states: Dict) -> Dict[str, pd.Series]:
        """
        【V2.2 · 依赖注入修复版】“底部吸筹锁仓”诊断引擎
        - 核心修复: 1. 增加 current_chip_states 参数，用于接收模块内已计算的信号。
                      2. 将 SCORE_CHIP_TRUE_ACCUMULATION 的数据源从全局 self.strategy.atomic_states
                         修正为从传入的 current_chip_states 获取，修复数据依赖链。
        """
        states = {}
        signal_name = 'SCORE_CHIP_BOTTOM_ACCUMULATION_LOCKDOWN'
        p_conf = get_params_block(self.strategy, 'chip_lockdown_params', {})
        lookback_window = get_param_value(p_conf.get('lookback_window'), 5)
        accumulation_threshold = get_param_value(p_conf.get('accumulation_threshold'), 0.3)
        norm_window = 55
        # --- 1. 设置阶段: 识别“底部吸筹区” (Setup Phase) ---
        # 权威的底部支撑信号 (调用utils中的底层函数)
        gaia_params = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {}).get('gaia_bedrock_params', {})
        fib_params = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {}).get('fibonacci_support_params', {})
        gaia_support = utils._calculate_gaia_bedrock_support(df, gaia_params, self.strategy.atomic_states)
        historical_low_support = utils._calculate_historical_low_support(df, fib_params)
        authoritative_bottom_support = np.maximum(gaia_support, historical_low_support) > 0.1
        # [代码修改开始]
        # 修复数据依赖：从传入的 current_chip_states 中获取信号，而不是全局状态
        chip_accumulation_score = current_chip_states.get('SCORE_CHIP_TRUE_ACCUMULATION', pd.Series(0.0, index=df.index))
        # [代码修改结束]
        sustained_accumulation = chip_accumulation_score.rolling(window=3).mean() > accumulation_threshold
        # 定义“底部吸筹区”状态
        is_in_bottom_accumulation_zone = authoritative_bottom_support & sustained_accumulation
        # --- 2. 触发阶段: 识别“点火日” (Trigger Phase) ---
        is_ignition_day = (df['pct_change_D'] > 0.01) & (df['volume_D'] > df.get('VOL_MA_21_D', 0))
        # --- 3. 确认阶段: 回溯验证与当天确认 (Confirmation Phase) ---
        # 回溯验证: 点火日前lookback_window天内，是否存在“底部吸筹区”
        has_recent_setup = is_in_bottom_accumulation_zone.rolling(window=lookback_window).max().shift(1).fillna(0).astype(bool)
        # 当天锁仓确认
        low_profit_taking_urgency = 1.0 - normalize_score(df.get('profit_taking_urgency_D', pd.Series(0.0, index=df.index)), df.index, norm_window, ascending=True)
        no_main_force_distribution = 1.0 - normalize_score(df.get('main_force_rally_distribution_D', pd.Series(0.0, index=df.index)), df.index, norm_window, ascending=True)
        winner_conviction_raw = self.strategy.atomic_states.get('PROCESS_META_WINNER_CONVICTION', pd.Series(0.0, index=df.index))
        winner_conviction_score = (winner_conviction_raw.clip(-1, 1) * 0.5 + 0.5)
        lockdown_confirmation_score = (low_profit_taking_urgency * no_main_force_distribution * winner_conviction_score)**(1/3)
        # --- 4. 最终融合裁决 ---
        final_score = (
            is_ignition_day &
            has_recent_setup
        ) * lockdown_confirmation_score
        states[signal_name] = final_score.clip(0, 1).astype(np.float32)
        return states

    def diagnose_lockdown_scramble(self, df: pd.DataFrame, current_chip_states: Dict) -> Dict[str, pd.Series]:
        """
        【V1.0 · 新增】“锁仓抢筹”诊断引擎
        - 核心逻辑: 在“底部锁仓”信号触发的当天，进一步验证主力是否存在不计成本的抢筹行为。
        """
        states = {}
        signal_name = 'SCORE_CHIP_LOCKDOWN_SCRAMBLE'
        norm_window = 55
        # 前提：必须是底部锁仓信号的触发日
        lockdown_trigger = current_chip_states.get('SCORE_CHIP_BOTTOM_ACCUMULATION_LOCKDOWN', pd.Series(0.0, index=df.index))
        # 证据一：主力成本优势 (低吸)
        cost_advantage = normalize_score(df.get('main_buy_cost_advantage_D', pd.Series(0.0, index=df.index)), df.index, norm_window, ascending=True)
        # 证据二：主力净流入 (买入)
        net_flow = normalize_score(df.get('main_force_net_flow_consensus_D', pd.Series(0.0, index=df.index)).clip(lower=0), df.index, norm_window, ascending=True)
        # 证据三：主力信念 (大单买)
        conviction = normalize_score(df.get('main_force_conviction_ratio_D', pd.Series(0.0, index=df.index)), df.index, norm_window, ascending=True)
        # 证据四：亏钱也要买 (不计成本)
        costly_buy = normalize_score(df.get('main_force_intraday_profit_D', pd.Series(0.0, index=df.index)).clip(upper=0).abs(), df.index, norm_window, ascending=True)
        # 融合所有抢筹证据
        scramble_evidence = (
            cost_advantage *
            net_flow *
            conviction *
            costly_buy
        )**(1/4)
        # 最终信号 = 前提 * 证据
        final_score = lockdown_trigger * scramble_evidence
        states[signal_name] = final_score.clip(0, 1).astype(np.float32)
        return states

    def _synthesize_ultimate_signals(self, df: pd.DataFrame, concentration: Dict[int, pd.Series], accumulation: Dict[int, pd.Series], power_transfer: Dict[int, pd.Series], peak_integrity: Dict[int, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V6.0 · 四象限重构版】终极信号合成器
        - 核心重构: 引入“四象限动态分析法”，彻底解决信号命名与逻辑混乱的问题。
                      1. [正本清源] 将动态分析拆分为四个明确的象限：看涨加速、顶部反转、看跌加速、底部反转。
                      2. [信号新生] 为每个象限创建独立的、命名准确的信号，确保逻辑清晰，杜绝混淆。
                      3. [战术重铸] 重构 SCORE_CHIP_TACTICAL_REVERSAL 逻辑，使其真正捕捉“上升趋势中的回调买点”。
        """
        states = {}
        periods = sorted(concentration.keys())
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {1: 0.1, 5: 0.4, 13: 0.3, 21: 0.15, 55: 0.05})
        axiom_weights = get_param_value(p_conf.get('axiom_weights'), {'concentration': 0.3, 'accumulation': 0.3, 'power_transfer': 0.25, 'peak_integrity': 0.15})
        norm_window = 55
        # [代码修改开始]
        # 步骤一：计算各周期的双极性“全息筹码健康分”
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
        # 步骤二：分离为纯粹的看涨/看跌健康分
        bullish_scores_by_period = {p: score.clip(0, 1) for p, score in bipolar_health_by_period.items()}
        bearish_scores_by_period = {p: (score.clip(-1, 0) * -1) for p, score in bipolar_health_by_period.items()}
        # 步骤三：计算静态的共振信号 (零阶动态)
        bullish_resonance = pd.Series(0.0, index=df.index)
        bearish_resonance = pd.Series(0.0, index=df.index)
        numeric_weights = {int(k): v for k, v in tf_weights.items() if str(k).isdigit()}
        total_weight = sum(numeric_weights.values())
        if total_weight > 0:
            for p_str, weight in numeric_weights.items():
                p = int(p_str)
                normalized_weight = weight / total_weight
                bullish_resonance += bullish_scores_by_period.get(p, 0.0) * normalized_weight
                bearish_resonance += bearish_scores_by_period.get(p, 0.0) * normalized_weight
        states['SCORE_CHIP_BULLISH_RESONANCE'] = bullish_resonance.fillna(0).clip(0, 1).astype(np.float32)
        states['SCORE_CHIP_BEARISH_RESONANCE'] = bearish_resonance.fillna(0).clip(0, 1).astype(np.float32)
        # 步骤四：计算四象限动态信号 (一阶和二阶动态)
        bullish_accel_score = pd.Series(0.0, index=df.index)
        top_reversal_score = pd.Series(0.0, index=df.index)
        bearish_accel_score = pd.Series(0.0, index=df.index)
        bottom_reversal_score = pd.Series(0.0, index=df.index)
        if total_weight > 0:
            for p_str, weight in numeric_weights.items():
                p = int(p_str)
                normalized_weight = weight / total_weight
                context_p = periods[periods.index(p) + 1] if periods.index(p) + 1 < len(periods) else p
                # --- 基于“看涨健康分”的动态分析 ---
                holographic_bull_divergence = self._calculate_holographic_divergence(bullish_scores_by_period.get(p, pd.Series(0.0, index=df.index)), 1, p, context_p)
                bullish_accel_score += holographic_bull_divergence.clip(0, 1) * normalized_weight
                top_reversal_score += (holographic_bull_divergence.clip(-1, 0) * -1) * normalized_weight
                # --- 基于“看跌健康分”的动态分析 ---
                holographic_bear_divergence = self._calculate_holographic_divergence(bearish_scores_by_period.get(p, pd.Series(0.0, index=df.index)), 1, p, context_p)
                bearish_accel_score += holographic_bear_divergence.clip(0, 1) * normalized_weight
                bottom_reversal_score += (holographic_bear_divergence.clip(-1, 0) * -1) * normalized_weight
        # 步骤五：赋值给命名准确的终极信号
        states['SCORE_CHIP_BULLISH_ACCELERATION'] = bullish_accel_score.clip(0, 1).astype(np.float32)
        states['SCORE_CHIP_TOP_REVERSAL'] = top_reversal_score.clip(0, 1).astype(np.float32)
        states['SCORE_CHIP_BEARISH_ACCELERATION'] = bearish_accel_score.clip(0, 1).astype(np.float32)
        states['SCORE_CHIP_BOTTOM_REVERSAL'] = bottom_reversal_score.clip(0, 1).astype(np.float32)
        # 步骤六：重铸战术反转和哈迪斯陷阱信号
        # 战术反转 = 强看涨共振中的顶部反转（回调）
        states['SCORE_CHIP_TACTICAL_REVERSAL'] = (bullish_resonance * top_reversal_score).clip(0, 1).astype(np.float32)
        # 哈迪斯陷阱 = 底部反转信号出现时，伴随着强烈的派发行为
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
        # [代码修改结束]
        return states

    def _diagnose_concentration_dynamics(self, df: pd.DataFrame, periods: list) -> Dict[int, pd.Series]:
        """
        【V6.2 · 赫淮斯托斯之锤版】核心公理一：诊断筹码“聚散”的动态
        - 核心修复: 签署“赫淮斯托斯之锤”协议，修复了对“看跌证据”评估的逻辑漏洞。
                      确保对看跌证据的静态、斜率、加速度三个维度进行完整的、与看涨证据对称的评估。
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
            # 修复看跌证据的评估逻辑，使其与看涨证据对称
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
        【V5.3 · 指标根除版】核心公理三：诊断筹码“转移方向”
        - 核心重构: 彻底废弃并移除了对 'long_term_chips_unlocked_ratio_D' 的所有依赖。
                      “筹码向散户转移”的证据链现在由 'profit_taking_urgency_D' 和 'main_force_rally_distribution_D' 构成，
                      逻辑更清晰，因果关系更强。
        """
        scores = {}
        sorted_periods = sorted(periods)
        for i, p in enumerate(sorted_periods):
            context_p = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p
            # 证据一：筹码从散户/套牢盘 -> 主力
            transfer_to_main_static = df.get('retail_capitulation_distribution_D', pd.Series(0.0, index=df.index)) + df.get('long_term_despair_selling_ratio_D', pd.Series(0.0, index=df.index))
            transfer_to_main_slope = df.get(f'SLOPE_{p}_retail_capitulation_distribution_D', pd.Series(0.0, index=df.index)) + df.get(f'SLOPE_{p}_long_term_despair_selling_ratio_D', pd.Series(0.0, index=df.index))
            transfer_to_main_accel = df.get(f'ACCEL_{p}_retail_capitulation_distribution_D', pd.Series(0.0, index=df.index)) + df.get(f'ACCEL_{p}_long_term_despair_selling_ratio_D', pd.Series(0.0, index=df.index))
            # 证据二：筹码从主力/获利盘 -> 散户 (逻辑重构)
            transfer_to_retail_static = df.get('profit_taking_urgency_D', pd.Series(0.0, index=df.index)) + df.get('main_force_rally_distribution_D', pd.Series(0.0, index=df.index))
            transfer_to_retail_slope = df.get(f'SLOPE_{p}_profit_taking_urgency_D', pd.Series(0.0, index=df.index)) + df.get(f'SLOPE_{p}_main_force_rally_distribution_D', pd.Series(0.0, index=df.index))
            transfer_to_retail_accel = df.get(f'ACCEL_{p}_profit_taking_urgency_D', pd.Series(0.0, index=df.index)) + df.get(f'ACCEL_{p}_main_force_rally_distribution_D', pd.Series(0.0, index=df.index))
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
        【V6.2 · 加速度校准版】筹码专用的关系元分析核心引擎
        - 核心修复: 修正了“加速度”计算的致命逻辑错误。加速度是速度的一阶导数，
                      因此其计算应为 relationship_trend.diff(1)，而不是错误的 diff(meta_window)。
        """
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        p_meta = get_param_value(p_conf.get('relational_meta_analysis_params'), {})
        w_state = get_param_value(p_meta.get('state_weight'), 0.2)
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.3)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.3)
        w_holographic = get_param_value(p_meta.get('holographic_weight'), 0.2)
        norm_window = 55
        bipolar_sensitivity = 1.0
        relationship_trend = snapshot_score.diff(meta_window).fillna(0)
        velocity_score = normalize_to_bipolar(relationship_trend, df.index, norm_window, bipolar_sensitivity)
        # [代码修改开始]
        # 致命错误修复：加速度是速度(trend)的一阶导数，应使用 diff(1)
        relationship_accel = relationship_trend.diff(1).fillna(0)
        # [代码修改结束]
        acceleration_score = normalize_to_bipolar(relationship_accel, df.index, norm_window, bipolar_sensitivity)
        holographic_score = holographic_divergence_score.clip(-1, 1)
        bullish_state = snapshot_score.clip(0, 1)
        bullish_velocity = velocity_score.clip(0, 1)
        bullish_acceleration = acceleration_score.clip(0, 1)
        bullish_holographic = holographic_score.clip(0, 1)
        total_bullish_force = (
            bullish_state * w_state +
            bullish_velocity * w_velocity +
            bullish_acceleration * w_acceleration +
            bullish_holographic * w_holographic
        )
        bearish_state = (snapshot_score.clip(-1, 0) * -1)
        bearish_velocity = (velocity_score.clip(-1, 0) * -1)
        bearish_acceleration = (acceleration_score.clip(-1, 0) * -1)
        bearish_holographic = (holographic_score.clip(-1, 0) * -1)
        total_bearish_force = (
            bearish_state * w_state +
            bearish_velocity * w_velocity +
            bearish_acceleration * w_acceleration +
            bearish_holographic * w_holographic
        )
        final_score = (total_bullish_force - total_bearish_force).clip(-1, 1)
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
        【V4.4 · 成交量能核证版】诊断“恐慌投降反转”的潜力
        - 核心升级: 贯彻指挥官思想，重构“成交量确认”支柱。现在它由“相对爆发强度”（成交量 > 短期均量）
                      和“绝对换手水平”（高换手率）共同构成，彻底解决了成交量绝对值的欺骗性问题。
        """
        states = {}
        # [代码修改开始]
        # 增加 VOL_MA_5_D, VOL_MA_21_D, turnover_rate_D 到必需列
        required_cols = ['total_loser_rate_D', 'close_D', 'retail_capitulation_distribution_D', 'volume_D', 'VOL_MA_5_D', 'VOL_MA_21_D', 'turnover_rate_D']
        # [代码修改结束]
        if any(col not in df.columns for col in required_cols):
            states['SCORE_CHIP_CONTEXT_CAPITULATION_POTENTIAL'] = pd.Series(0.0, index=df.index)
            return states
        periods = [5, 13, 21, 55]
        sorted_periods = sorted(periods)
        capitulation_scores_by_period = {}
        bearish_ma_context = 1 - self._calculate_ma_trend_context(df, [5, 13, 21, 55])
        # [代码修改开始]
        # --- 构建全新的“成交量能核证”支柱 ---
        # 维度一：相对爆发强度
        volume_breakout_condition = df['volume_D'] > np.maximum(df.get('VOL_MA_5_D', 0), df.get('VOL_MA_21_D', 0))
        # 维度二：绝对换手水平
        high_turnover_score = normalize_score(df['turnover_rate_D'], df.index, 55, ascending=True)
        # 融合：必须同时满足相对爆发和高换手
        volume_confirmation_score = (volume_breakout_condition.astype(float) * high_turnover_score)
        # [代码修改结束]
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
            # [代码修改开始]
            # 在融合时使用全新的、更可靠的成交量能核证分数
            snapshot_score = (fused_deep_cap * fused_price_lows * fused_loser_turnover * bearish_ma_context * volume_confirmation_score).astype(np.float32)
            # [代码修改结束]
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


