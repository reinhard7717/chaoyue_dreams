# 文件: strategies/trend_following/intelligence/tactic_engine.py
# 新增文件: 战术与剧本引擎
import pandas as pd
import numpy as np
from typing import Dict
from strategies.trend_following.utils import get_params_block, get_param_value

class TacticEngine:
    """
    战术与剧本引擎
    - 核心职责: 专门负责合成和管理具体的、可直接执行的“战术”或“剧本”。
                这些战术通常具有明确的“战备-点火”时序逻辑，是交易决策的直接输入。
    - 来源: 从臃肿的 CognitiveIntelligence 模块中拆分而来。
    """
    def __init__(self, strategy_instance):
        """
        初始化战术与剧本引擎。
        :param strategy_instance: 策略主实例的引用。
        """
        self.strategy = strategy_instance

    def _get_atomic_score(self, df: pd.DataFrame, name: str, default=0.0) -> pd.Series:
        """安全地从原子状态库中获取分数。"""
        return self.strategy.atomic_states.get(name, pd.Series(default, index=df.index))

    def _fuse_multi_level_scores(self, df: pd.DataFrame, base_name: str, weights: Dict[str, float] = None) -> pd.Series:
        """融合S+/S/A/B等多层置信度分数的辅助函数。"""
        if weights is None:
            weights = {'S_PLUS': 1.5, 'S': 1.0, 'A': 0.6, 'B': 0.3}
        total_score = pd.Series(0.0, index=df.index)
        total_weight = 0.0
        for level in ['S_PLUS', 'S', 'A', 'B']:
            if level not in weights: continue
            weight = weights[level]
            score_name = f"SCORE_{base_name}_{level}"
            if score_name in self.strategy.atomic_states:
                score_series = self.strategy.atomic_states[score_name]
                if len(score_series) > 0:
                    total_score += score_series.reindex(df.index).fillna(0.0) * weight
                    total_weight += weight
        if total_weight == 0:
            single_score_name = f"SCORE_{base_name}"
            if single_score_name in self.strategy.atomic_states:
                return self.strategy.atomic_states[single_score_name].reindex(df.index).fillna(0.5)
            return pd.Series(0.5, index=df.index)
        return (total_score / total_weight).clip(0, 1)

    def run_tactic_synthesis(self, df: pd.DataFrame, pullback_enhancements: Dict) -> Dict[str, pd.Series]:
        """
        【V1.2 · 架构归位版】战术引擎总指挥
        - 核心重构 (本次修改):
          - [职责重塑] 将原有的 `synthesize_v_reversal_ace_playbook` 重构为 `synthesize_panic_selling_setup`，使其职责回归纯粹的“战备信号生成”。
        - 核心职责: 按顺序调用本模块内的所有战术合成方法，并汇总其产出的所有信号。
        """
        print("      -> [战术引擎 V1.2 · 架构归位版] 启动...") # [代码修改] 更新版本号和说明
        all_states = {}
        # 注意：一些战术依赖于其他战术的输出，调用顺序很重要
        all_states.update(self.synthesize_chip_price_lag_playbook(df))
        all_states.update(self.synthesize_advanced_tactics(df))
        all_states.update(self.synthesize_prime_tactic(df))
        all_states.update(self._diagnose_pullback_tactics_matrix(df, pullback_enhancements))
        all_states.update(self.synthesize_squeeze_playbooks(df))
        all_states.update(self.synthesize_panic_selling_setup(df)) # [代码修改] 调用重构后的战备信号生成方法
        print(f"      -> [战术引擎] 分析完毕，共生成 {len(all_states)} 个战术信号。")
        return all_states

    def synthesize_panic_selling_setup(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增, 原V型反转剧本重构】恐慌抛售战备(Setup)信号生成模块
        - 核心目标: 识别“恐慌抛售日”，为下游的V型反转剧本提供战备状态输入。
        - 战备状态 (Setup): 融合了“价格大幅下跌”、“成交量放大”、“筹码结构崩溃”三大特征。
        - 产出: SCORE_SETUP_PANIC_SELLING_S - 一个0-1之间的数值化战备分数。
        """
        print("        -> [恐慌抛售战备模块 V1.0] 启动...") # [代码修改] 全新方法
        states = {}
        # --- 1. 定义“恐慌抛售日”战备分数 (Setup Score) ---
        # 维度1: 价格大幅下跌 (使用归一化的跌幅)
        price_drop_score = self._normalize_score(df['pct_change_D'].clip(upper=0), window=60, ascending=False)
        # 维度2: 成交量显著放大
        volume_spike_score = self._normalize_score(df['volume_D'] / df['VOL_MA_21_D'], window=60, ascending=True)
        # 维度3: 筹码结构崩溃 (使用看跌共振信号)
        chip_breakdown_score = self._fuse_multi_level_scores(df, 'CHIP_BEARISH_RESONANCE')
        # 融合为战备分
        setup_panic_selling_score = (price_drop_score * volume_spike_score * chip_breakdown_score).astype(np.float32)
        states['SCORE_SETUP_PANIC_SELLING_S'] = setup_panic_selling_score
        if (setup_panic_selling_score > 0.5).any():
            print(f"          -> [S级战备] 侦测到 {(setup_panic_selling_score > 0.5).sum()} 个“恐慌抛售”战备日！")
        return states

    def synthesize_v_reversal_ace_playbook(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】V型反转王牌剧本 (V-Reversal Ace Playbook)
        - 核心目标: 精确捕捉A股市场常见的“恐慌性杀跌后的报复性反转”模式。
        - 战备状态 (Setup): 前一日是“恐慌抛售日”，融合了“价格大幅下跌”、“成交量放大”、“筹码结构崩溃”三大特征。
        - 点火触发 (Trigger): 当日出现强劲的“显性反转K线”。
        - 剧本逻辑: 昨日恐慌抛售战备就绪，今日强力反转点火确认。
        - 产出: 一个S++级的、高确定性的V型反转剧本信号。
        """
        print("        -> [V型反转王牌剧本 V1.0] 启动...") # 打印启动信息
        states = {}
        atomic = self.strategy.atomic_states
        triggers = self.strategy.trigger_events
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 定义“恐慌抛售日”战备分数 (Setup Score) ---
        # 维度1: 价格大幅下跌 (使用归一化的跌幅)
        price_drop_score = self._normalize_score(df['pct_change_D'].clip(upper=0), window=60, ascending=False)
        # 维度2: 成交量显著放大
        volume_spike_score = self._normalize_score(df['volume_D'] / df['VOL_MA_21_D'], window=60, ascending=True)
        # 维度3: 筹码结构崩溃 (使用看跌共振信号)
        chip_breakdown_score = self._fuse_multi_level_scores(df, 'CHIP_BEARISH_RESONANCE')
        # 融合为战备分
        setup_panic_selling_score = (price_drop_score * volume_spike_score * chip_breakdown_score).astype(np.float32)
        states['SCORE_SETUP_PANIC_SELLING_S'] = setup_panic_selling_score
        # --- 2. 获取“显性反转”点火信号 (Trigger) ---
        # 直接消费由PlaybookEngine生成的、已包含多维度信息的数值化反转信号
        trigger_dominant_reversal_score = triggers.get('TRIGGER_DOMINANT_REVERSAL', default_score)
        # --- 3. 融合生成最终剧本信号 ---
        # 逻辑: 昨日战备就绪 & 今日点火确认
        was_setup_yesterday = setup_panic_selling_score.shift(1).fillna(0.0)
        is_triggered_today = trigger_dominant_reversal_score
        final_playbook_score = (was_setup_yesterday * is_triggered_today).astype(np.float32)
        # 生成S++级的剧本分数和布尔信号
        states['SCORE_PLAYBOOK_V_REVERSAL_ACE_S_PLUS'] = final_playbook_score
        states['PLAYBOOK_V_REVERSAL_ACE_S_PLUS'] = final_playbook_score > 0.3 # 设置一个合理的触发阈值
        if (states['PLAYBOOK_V_REVERSAL_ACE_S_PLUS']).any():
            print(f"          -> [S++级王牌剧本] 侦测到 {(states['PLAYBOOK_V_REVERSAL_ACE_S_PLUS']).sum()} 次“V型反转”王牌买点！")
        return states

    def synthesize_chip_price_lag_playbook(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】“筹码共振-价格滞后”战术剧本
        - 核心目标: 捕捉“万事俱备，只欠东风”的黄金买点。
        - 战备状态 (Setup): 筹码高度共振 + 价格动能被压制 + 波动率压缩。
        - 点火触发 (Trigger): 价格出现温和的启动迹象。
        - 剧本逻辑: 昨日战备就绪，今日点火触发。
        """
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        chip_resonance_score = self._fuse_multi_level_scores(df, 'CHIP_BULLISH_RESONANCE', {'S_PLUS': 1.2, 'S': 1.0})
        price_momentum_suppressed_score = self._normalize_score(df['SLOPE_5_close_D'], ascending=False)
        volatility_compression_score = atomic.get('SCORE_VOL_COMPRESSION_S', default_score)
        setup_score = (chip_resonance_score * price_momentum_suppressed_score * volatility_compression_score).astype(np.float32)
        states['SCORE_SETUP_CHIP_RESONANCE_READY_S'] = setup_score
        states['SETUP_CHIP_RESONANCE_READY_S'] = setup_score > 0.6
        trigger_score = atomic.get('COGNITIVE_SCORE_EARLY_MOMENTUM_IGNITION_A', default_score)
        states['SCORE_TRIGGER_GENTLE_PRICE_LIFT_A'] = trigger_score
        states['TRIGGER_GENTLE_PRICE_LIFT_A'] = trigger_score > 0.4
        return states

    def synthesize_advanced_tactics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.4 逻辑修复版】高级战法合成模块
        - 核心职责: 合成那些需要复杂时序逻辑的高级战法。
        - 本次升级: [修复] 新增对 `_diagnose_lock_chip_rally_tactic` 的调用，
                    修复了“锁筹拉升”战法从未被执行的逻辑缺陷。
        """
        states = {}
        states.update(self._diagnose_lock_chip_reconcentration_tactic(df))
        states.update(self._diagnose_lock_chip_rally_tactic(df))
        atomic = self.strategy.atomic_states
        triggers = self.strategy.trigger_events
        default_series = pd.Series(False, index=df.index)
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        weights = {'S': 1.0, 'A': 0.6, 'B': 0.3}
        total_weight = sum(weights.values())
        capitulation_s = self._get_atomic_score(df, 'SCORE_CAPITULATION_BOTTOM_RESONANCE_S', 0.0)
        capitulation_a = self._get_atomic_score(df, 'SCORE_CAPITULATION_BOTTOM_REVERSAL_A', 0.0)
        capitulation_b = self._get_atomic_score(df, 'SCORE_CAPITULATION_BOTTOM_REVERSAL_B', 0.0)
        fault_event_score = (capitulation_s * weights['S'] + capitulation_a * weights['A'] + capitulation_b * weights['B']) / total_weight
        confirmation_trigger_score_arr = np.maximum(
            triggers.get('TRIGGER_DOMINANT_REVERSAL', default_series).astype(float).values,
            triggers.get('TRIGGER_CHIP_IGNITION', default_series).astype(float).values
        )
        confirmation_trigger_score = pd.Series(confirmation_trigger_score_arr, index=df.index)
        main_uptrend_score = atomic.get('SCORE_STRUCTURE_MAIN_UPTREND_WAVE_S', default_score)
        fault_window_score = fault_event_score.rolling(window=3, min_periods=1).max()
        final_tactic_score = (fault_window_score * confirmation_trigger_score * main_uptrend_score).astype(np.float32)
        states['COGNITIVE_SCORE_FAULT_REBIRTH_ASCENT_S_PLUS'] = final_tactic_score
        p_advanced = get_params_block(self.strategy, 'advanced_tactics_params', {})
        final_signal_threshold = get_param_value(p_advanced.get('fault_rebirth_threshold'), 0.4)
        final_signal = final_tactic_score > final_signal_threshold
        states['TACTIC_FAULT_REBIRTH_ASCENT_S_PLUS'] = final_signal
        if final_signal.any():
            print(f"          -> [S+级战法重构版] 侦测到 {final_signal.sum()} 次“断层新生·主升浪”机会！")
        return states

    def _diagnose_lock_chip_reconcentration_tactic(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.5 信号源修复版】锁仓再集中S+战法诊断模块
        - 核心重构: (V2.0) 将战法的“准备状态”从有缺陷的A级信号，升级为经过战场环境过滤的
                      S级“筹码结构黄金机会”信号。
        - 本次升级: 【数值化】将原有的布尔逻辑升级为“战备分 * 点火分”的数值化评分体系。
        - 核心修复 (V2.5): 修复了对 `SCORE_DYN_OVERALL_BULLISH_MOMENTUM_S` 这个不存在信号的引用，
                        替换为消费由 `DynamicMechanicsEngine` 生成的、逻辑最相近的
                        `SCORE_DYN_BULLISH_RESONANCE_S` 终极信号。
        """
        states = {}
        atomic = self.strategy.atomic_states
        triggers = self.strategy.trigger_events
        default_series = pd.Series(False, index=df.index)
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        setup_score = atomic.get('CHIP_SCORE_PRIME_OPPORTUNITY_S', default_score)
        trigger_chip_ignition_score = triggers.get('TRIGGER_CHIP_IGNITION', default_series).astype(float)
        energy_release_score = atomic.get('SCORE_DYN_BULLISH_RESONANCE_S', default_score)
        cost_accel_score = atomic.get('SCORE_PLATFORM_COST_ACCEL', default_score)
        squeeze_breakout_score = atomic.get('COGNITIVE_SCORE_VOL_BREAKOUT_S', default_score)
        ignition_trigger_score_arr = np.maximum.reduce([
            trigger_chip_ignition_score.values,
            energy_release_score.values,
            cost_accel_score.values,
            squeeze_breakout_score.values
        ])
        ignition_trigger_score = pd.Series(ignition_trigger_score_arr, index=df.index)
        was_setup_yesterday_score = setup_score.shift(1).fillna(0.0)
        final_tactic_score = (was_setup_yesterday_score * ignition_trigger_score).astype(np.float32)
        states['COGNITIVE_SCORE_LOCK_CHIP_RECONCENTRATION_S_PLUS'] = final_tactic_score
        final_tactic_signal = final_tactic_score > 0.5
        states['TACTIC_LOCK_CHIP_RECONCENTRATION_S_PLUS'] = final_tactic_signal
        if final_tactic_signal.any():
            print(f"          -> [S+级战法确认] 侦测到 {final_tactic_signal.sum()} 次“锁仓再集中”的最终拉升信号！")
        return states

    def _diagnose_lock_chip_rally_tactic(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.5 向量化性能重构版】锁筹拉升S级战法诊断模块
        - 核心优化 (本次修改):
          - [性能重构] 彻底移除了原有的Python for循环状态机，重构为完全向量化的Pandas/NumPy逻辑。
          - [效率提升] 通过`cumsum`和`groupby`等向量化操作，一次性计算所有时间点的状态，避免了逐日迭代，极大提升了长周期回测的计算效率。
        - 业务逻辑: 保持与V2.4版本完全一致的“容错巡航”状态机逻辑，仅重构实现方式。
        """
        states = {}
        atomic = self.strategy.atomic_states
        default_series = pd.Series(False, index=df.index)
        p = get_params_block(self.strategy, 'lock_chip_rally_params', {})
        require_concentration = get_param_value(p.get('require_continuous_concentration'), True)
        terminate_on_stalling = get_param_value(p.get('terminate_on_health_stalling'), True)
        divergence_threshold = get_param_value(p.get('divergence_threshold'), 0.7)
        concentration_threshold = get_param_value(p.get('concentration_threshold'), 0.6)
        ignition_event = atomic.get('TACTIC_LOCK_CHIP_RECONCENTRATION_S_PLUS', default_series)
        is_diverging = self._fuse_multi_level_scores(df, 'FALLING_RESONANCE') > divergence_threshold
        is_late_stage = atomic.get('CONTEXT_TREND_STAGE_LATE', default_series)
        is_ma_broken = self._get_atomic_score(df, 'SCORE_MA_HEALTH', 1.0) < 0.4
        is_health_stalling = atomic.get('COGNITIVE_HOLD_RISK_HEALTH_STALLING', default_series)
        hard_termination_condition = is_diverging | is_late_stage | is_ma_broken
        if terminate_on_stalling:
            hard_termination_condition |= is_health_stalling
        is_cruise_condition_met = self._fuse_multi_level_scores(df, 'RISING_RESONANCE') > concentration_threshold if require_concentration else pd.Series(True, index=df.index)
        cruise_failure = ~is_cruise_condition_met
        double_cruise_failure = cruise_failure & cruise_failure.shift(1).fillna(False)
        rally_killer = hard_termination_condition | double_cruise_failure
        rally_block_id = rally_killer.cumsum()
        has_ignition_in_block = ignition_event.groupby(rally_block_id).cumsum() > 0
        is_in_rally_state = has_ignition_in_block & ~rally_killer
        final_tactic_signal = is_in_rally_state & ~hard_termination_condition
        states['TACTIC_LOCK_CHIP_RALLY_S'] = final_tactic_signal
        return states

    def synthesize_prime_tactic(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.4 融合函数升级版】终极战法合成模块
        - 核心职责: (原有注释)
        - 本次升级 (V2.4):
          - [逻辑深化] 使用新增的 `_fuse_multi_level_scores` 辅助函数来融合S/A/B三级
                        波动率压缩信号，使得对“极致压缩”的判断更平滑、更鲁棒。
        - 收益: 战法对市场状态的感知更精确，避免了因S级信号的微小波动而错失机会。
        """
        states = {}
        atomic = self.strategy.atomic_states
        default_series = pd.Series(False, index=df.index)
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        is_prime_chip_structure = self._get_atomic_score(df, 'CHIP_SCORE_PRIME_OPPORTUNITY_S', 0.0) > 0.7
        fused_compression_score = self._fuse_multi_level_scores(df, 'VOL_COMPRESSION')
        is_extreme_squeeze = fused_compression_score > 0.9
        has_energy_advantage = self._fuse_multi_level_scores(df, 'MECHANICS_BULLISH_RESONANCE') > 0.7
        condition_sum = (
            is_prime_chip_structure.astype(int) +
            is_extreme_squeeze.astype(int) +
            has_energy_advantage.astype(int)
        )
        setup_s_plus_plus = (condition_sum == 3)
        states['SETUP_PRIME_STRUCTURE_S_PLUS_PLUS'] = setup_s_plus_plus
        setup_s_plus = (condition_sum == 2)
        states['SETUP_PRIME_STRUCTURE_S_PLUS'] = setup_s_plus
        ignition_resonance_score = atomic.get('COGNITIVE_SCORE_IGNITION_RESONANCE_S', default_score)
        trigger_prime_breakout_s = ignition_resonance_score > 0.6
        is_in_early_stage_today = atomic.get('CONTEXT_TREND_STAGE_EARLY', default_series)
        is_triggered_today = trigger_prime_breakout_s
        was_setup_s_plus_plus_yesterday = setup_s_plus_plus.shift(1).fillna(False)
        final_tactic_s_plus_plus = was_setup_s_plus_plus_yesterday & is_triggered_today & is_in_early_stage_today
        states['TACTIC_PRIME_STRUCTURE_BREAKOUT_S_PLUS_PLUS'] = final_tactic_s_plus_plus
        was_setup_s_plus_yesterday = setup_s_plus.shift(1).fillna(False)
        final_tactic_s_plus = was_setup_s_plus_yesterday & is_triggered_today & is_in_early_stage_today & ~final_tactic_s_plus_plus
        states['TACTIC_PRIME_STRUCTURE_BREAKOUT_S_PLUS'] = final_tactic_s_plus
        return states

    def _diagnose_pullback_tactics_matrix(self, df: pd.DataFrame, enhancements: Dict) -> Dict[str, pd.Series]:
        """
        【V7.4 信号源修复版】回踩战术诊断模块
        - 核心升级: 为 S+ 级“巡航回踩确认”战法增加了“非上涨末期”的前置条件。
        - 本次升级: [信号修复] 修复了对“蓄势突破”信号的引用。原信号 `STRUCTURAL_OPP_ACCUMULATION_BREAKOUT_S` 已失效，
                      现已更新为消费逻辑最相近的 `STRUCTURE_MAIN_UPTREND_WAVE_S` 信号，以恢复对“初升浪”阶段的正确判断。
        """
        states = {}
        atomic = self.strategy.atomic_states
        triggers = self.strategy.trigger_events
        default_series = pd.Series(False, index=df.index)
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        lookback_window = 15
        ascent_start_event = atomic.get('STRUCTURE_MAIN_UPTREND_WAVE_S', default_series)
        cruise_start_event = atomic.get('TACTIC_LOCK_CHIP_RECONCENTRATION_S_PLUS', default_series)
        is_in_ascent_window = ascent_start_event.rolling(window=lookback_window, min_periods=1).max().astype(bool)
        is_in_cruise_window = cruise_start_event.rolling(window=lookback_window, min_periods=1).max().astype(bool)
        p_pullback = get_params_block(self.strategy, 'pullback_tactics_params', {})
        healthy_threshold = get_param_value(p_pullback.get('healthy_pullback_score_threshold'), 0.3)
        suppressive_threshold = get_param_value(p_pullback.get('suppressive_pullback_score_threshold'), 0.3)
        was_healthy_pullback = (atomic.get('COGNITIVE_SCORE_PULLBACK_HEALTHY_S', default_score).shift(1).fillna(0.0) > healthy_threshold)
        was_suppressive_pullback = (atomic.get('COGNITIVE_SCORE_PULLBACK_SUPPRESSIVE_S', default_score).shift(1).fillna(0.0) > suppressive_threshold)
        is_reversal_confirmed = triggers.get('TRIGGER_DOMINANT_REVERSAL', default_series)
        late_stage_score = self.strategy.atomic_states.get('CONTEXT_TREND_LATE_STAGE_SCORE', pd.Series(0, index=df.index))
        p_trend_stage = get_params_block(self.strategy, 'trend_stage_params', {})
        max_score_for_pullback = get_param_value(p_trend_stage.get('pullback_s_plus_max_late_stage_score'), 50)
        is_in_safe_stage = late_stage_score < max_score_for_pullback
        s_triple_plus_signal = is_in_cruise_window & was_suppressive_pullback & is_reversal_confirmed
        states['TACTIC_CRUISE_PIT_REVERSAL_S_TRIPLE_PLUS'] = s_triple_plus_signal
        s_plus_signal = is_in_cruise_window & was_healthy_pullback & is_reversal_confirmed & is_in_safe_stage & ~s_triple_plus_signal
        states['TACTIC_CRUISE_PULLBACK_REVERSAL_S_PLUS'] = s_plus_signal
        a_plus_signal = is_in_ascent_window & was_suppressive_pullback & is_reversal_confirmed & ~is_in_cruise_window
        states['TACTIC_ASCENT_PIT_REVERSAL_A_PLUS'] = a_plus_signal
        a_signal = is_in_ascent_window & was_healthy_pullback & is_reversal_confirmed & ~is_in_cruise_window & ~a_plus_signal
        states['TACTIC_ASCENT_PULLBACK_REVERSAL_A'] = a_signal
        return states

    def synthesize_squeeze_playbooks(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.5 信号发布增强版】压缩突破战术剧本合成模块
        - 本次升级 (V1.5):
          - [信号发布] 将内部使用的 `vol_compression_score` 正式发布为
                        `COGNITIVE_SCORE_VOL_COMPRESSION_FUSED` 原子状态，供其他模块消费。
        - 收益: 解决了 PlaybookEngine 跨模块调用的架构问题，修复了因此引发的 AttributeError。
        """
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        vol_compression_score = self._fuse_multi_level_scores(df, 'VOL_COMPRESSION')
        states['COGNITIVE_SCORE_VOL_COMPRESSION_FUSED'] = vol_compression_score.astype(np.float32)
        setup_extreme_squeeze_score = vol_compression_score.shift(1).fillna(0.0)
        trigger_explosive_breakout_score = atomic.get('SCORE_SQUEEZE_BREAKOUT_OPP_S', default_score)
        score_s_plus = (setup_extreme_squeeze_score * trigger_explosive_breakout_score).astype(np.float32)
        states['SCORE_PLAYBOOK_EXTREME_SQUEEZE_EXPLOSION_S_PLUS'] = score_s_plus
        states['PLAYBOOK_EXTREME_SQUEEZE_EXPLOSION_S_PLUS'] = score_s_plus > 0.7
        platform_quality_score = atomic.get('SCORE_PLATFORM_QUALITY_S', default_score)
        breakout_eve_score = (platform_quality_score * vol_compression_score)
        setup_breakout_eve_score = breakout_eve_score.shift(1).fillna(0.0)
        trigger_prime_breakout_score = atomic.get('COGNITIVE_SCORE_IGNITION_RESONANCE_S', default_score)
        score_s = (setup_breakout_eve_score * trigger_prime_breakout_score).astype(np.float32)
        states['SCORE_PLAYBOOK_BREAKOUT_EVE_S'] = score_s
        states['PLAYBOOK_BREAKOUT_EVE_S'] = score_s > 0.6
        setup_normal_squeeze_score = vol_compression_score.shift(1).fillna(0.0)
        trigger_grinding_advance_score = atomic.get('COGNITIVE_SCORE_VOL_BREAKOUT_A', default_score)
        trigger_any_breakout_score = np.maximum(trigger_explosive_breakout_score.values, trigger_grinding_advance_score.values)
        score_a = (setup_normal_squeeze_score * pd.Series(trigger_any_breakout_score, index=df.index)).astype(np.float32)
        states['SCORE_PLAYBOOK_NORMAL_SQUEEZE_BREAKOUT_A'] = score_a
        states['PLAYBOOK_NORMAL_SQUEEZE_BREAKOUT_A'] = (score_a > 0.5) & ~states['PLAYBOOK_EXTREME_SQUEEZE_EXPLOSION_S_PLUS']
        return states

    def _normalize_score(self, series: pd.Series, window: int = 120, ascending: bool = True, default=0.5) -> pd.Series:
        """
        辅助函数：将一个Series进行滚动窗口排名归一化，生成0-1分。
        - 从其他情报模块迁移而来，保持架构一致性。
        :param series: 原始数据Series。
        :param window: 归一化滚动窗口。
        :param ascending: 归一化方向，True表示值越大分数越高。
        :param default: 填充NaN的默认值。
        :return: 归一化后的0-1分数Series。
        """
        if series is None or series.empty:
            # 如果输入为空，根据情况返回一个填充了默认值的Series
            # 假设 self.strategy.df_indicators.index 是可用的主索引
            return pd.Series(default, index=self.strategy.df_indicators.index)
        min_periods = max(1, window // 5)
        rank = series.rolling(window=window, min_periods=min_periods).rank(pct=True)
        score = rank if ascending else 1 - rank
        return score.fillna(default).astype(np.float32)










