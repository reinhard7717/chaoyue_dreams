# 文件: strategies/trend_following/intelligence/tactic_engine.py
# 战术与剧本引擎
import pandas as pd
import numpy as np
from typing import Dict
from strategies.trend_following.utils import get_params_block, get_param_value, fuse_multi_level_scores, normalize_score

class TacticEngine:
    """
    战术与剧本引擎
    - 核心职责: 专门负责合成和管理具体的、可直接执行的“战术”或“剧本”。
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
        # 此处直接调用 utils 中的公共函数，保持一致性
        return fuse_multi_level_scores(df, self.strategy.atomic_states, base_name, weights)

    def run_tactic_synthesis(self, df: pd.DataFrame, pullback_enhancements: Dict) -> Dict[str, pd.Series]:
        """
        【V2.0 · 信号适配版】战术引擎总指挥
        - 核心重构 (本次修改):
          - [信号适配] 全面审查并更新了所有战术合成方法，确保它们消费的是最新的终极原子信号。
        """
        print("      -> [战术引擎 V2.0 · 信号适配版] 启动...") # 更新版本号
        all_states = {}
        
        # 依次调用所有战术合成方法，注意调用顺序
        all_states.update(self.synthesize_panic_selling_setup(df))
        self.strategy.atomic_states.update(all_states) # 立即更新，供下游使用

        all_states.update(self.synthesize_v_reversal_ace_playbook(df))
        all_states.update(self.synthesize_chip_price_lag_playbook(df))
        all_states.update(self.synthesize_prime_tactic(df))
        all_states.update(self._diagnose_pullback_tactics_matrix(df, pullback_enhancements))
        all_states.update(self.synthesize_squeeze_playbooks(df))
        
        print(f"      -> [战术引擎] 分析完毕，共生成 {len(all_states)} 个战术信号。")
        return all_states

    def synthesize_panic_selling_setup(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.1 · 信号适配版】恐慌抛售战备(Setup)信号生成模块
        - 核心逻辑: 价格大幅下跌 * 成交量放大 * 筹码结构崩溃
        """
        # print("        -> [恐慌抛售战备模块 V1.1] 启动...")
        states = {}
        price_drop_score = normalize_score(df['pct_change_D'].clip(upper=0), df.index, window=60, ascending=False)
        volume_spike_score = normalize_score(df['volume_D'] / df['VOL_MA_21_D'], df.index, window=60, ascending=True)
        
        # 消费新的终极筹码看跌信号
        chip_breakdown_score = fuse_multi_level_scores(df, 'CHIP_BEARISH_RESONANCE')
        
        setup_panic_selling_score = (price_drop_score * volume_spike_score * chip_breakdown_score).astype(np.float32)
        states['SCORE_SETUP_PANIC_SELLING_S'] = setup_panic_selling_score
        return states

    def synthesize_v_reversal_ace_playbook(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.1 · 信号适配版】V型反转王牌剧本
        - 核心逻辑: 昨日恐慌抛售战备就绪 & 今日强力反转点火确认
        """
        # print("        -> [V型反转王牌剧本 V1.1] 启动...")
        states = {}
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        
        # 消费新的终极反转信号作为点火器
        trigger_dominant_reversal_score = fuse_multi_level_scores(df, 'BEHAVIOR_BOTTOM_REVERSAL')
        
        was_setup_yesterday = self._get_atomic_score(df, 'SCORE_SETUP_PANIC_SELLING_S').shift(1).fillna(0.0)
        is_triggered_today = trigger_dominant_reversal_score
        
        final_playbook_score = (was_setup_yesterday * is_triggered_today).astype(np.float32)
        states['SCORE_PLAYBOOK_V_REVERSAL_ACE_S_PLUS'] = final_playbook_score
        states['PLAYBOOK_V_REVERSAL_ACE_S_PLUS'] = final_playbook_score > 0.3
        return states

    def synthesize_chip_price_lag_playbook(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.1 · 信号适配版】“筹码共振-价格滞后”战术剧本
        - 核心逻辑: (筹码共振 * 价格压制 * 波动压缩) * 价格温和启动
        """
        states = {}
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        
        # 消费新的终极信号
        chip_resonance_score = fuse_multi_level_scores(df, 'CHIP_BULLISH_RESONANCE', {'S_PLUS': 1.2, 'S': 1.0})
        price_momentum_suppressed_score = normalize_score(df['SLOPE_5_close_D'], df.index, ascending=False)
        volatility_compression_score = fuse_multi_level_scores(df, 'VOL_COMPRESSION')
        
        setup_score = (chip_resonance_score * price_momentum_suppressed_score * volatility_compression_score).astype(np.float32)
        states['SCORE_SETUP_CHIP_RESONANCE_READY_S'] = setup_score
        
        trigger_score = self._get_atomic_score(df, 'COGNITIVE_SCORE_EARLY_MOMENTUM_IGNITION_A', 0.0)
        
        final_score = setup_score.shift(1).fillna(0.0) * trigger_score
        states['SCORE_PLAYBOOK_CHIP_PRICE_LAG_S'] = final_score
        states['PLAYBOOK_CHIP_PRICE_LAG_S'] = final_score > 0.5
        return states

    def synthesize_prime_tactic(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.5 · 信号适配版】终极战法合成模块
        - 核心逻辑: (黄金筹码结构 * 极致波动压缩 * 能量优势) * 点火共振
        """
        states = {}
        # 消费新的终极信号
        is_prime_chip_structure = fuse_multi_level_scores(df, 'CHIP_BULLISH_RESONANCE') > 0.7
        fused_compression_score = fuse_multi_level_scores(df, 'VOL_COMPRESSION')
        is_extreme_squeeze = fused_compression_score > 0.9
        has_energy_advantage = fuse_multi_level_scores(df, 'DYN_BULLISH_RESONANCE') > 0.7
        
        condition_sum = (is_prime_chip_structure.astype(int) + is_extreme_squeeze.astype(int) + has_energy_advantage.astype(int))
        setup_s_plus_plus = (condition_sum == 3)
        states['SETUP_PRIME_STRUCTURE_S_PLUS_PLUS'] = setup_s_plus_plus
        
        ignition_resonance_score = self._get_atomic_score(df, 'COGNITIVE_SCORE_IGNITION_RESONANCE_S', 0.0)
        trigger_prime_breakout_s = ignition_resonance_score > 0.6
        
        was_setup_s_plus_plus_yesterday = setup_s_plus_plus.shift(1).fillna(False)
        final_tactic_s_plus_plus = was_setup_s_plus_plus_yesterday & trigger_prime_breakout_s
        states['TACTIC_PRIME_STRUCTURE_BREAKOUT_S_PLUS_PLUS'] = final_tactic_s_plus_plus
        return states

    def _diagnose_pullback_tactics_matrix(self, df: pd.DataFrame, enhancements: Dict) -> Dict[str, pd.Series]:
        """
        【V7.5 · 信号适配版】回踩战术诊断模块
        """
        states = {}
        default_series = pd.Series(False, index=df.index)
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        
        # 消费新的终极信号
        ascent_start_event = fuse_multi_level_scores(df, 'STRUCTURE_BULLISH_RESONANCE') > 0.6
        # 锁仓拉升战法需要更严格的定义，例如基于筹码和资金的协同
        chip_resonance_score = fuse_multi_level_scores(df, 'CHIP_BULLISH_RESONANCE')
        ff_resonance_score = fuse_multi_level_scores(df, 'FF_BULLISH_RESONANCE')
        cruise_start_event = (chip_resonance_score * ff_resonance_score) > 0.7

        lookback_window = 15
        is_in_ascent_window = ascent_start_event.rolling(window=lookback_window, min_periods=1).max().astype(bool)
        is_in_cruise_window = cruise_start_event.rolling(window=lookback_window, min_periods=1).max().astype(bool)
        
        p_pullback = get_params_block(self.strategy, 'pullback_tactics_params', {})
        healthy_threshold = get_param_value(p_pullback.get('healthy_pullback_score_threshold'), 0.3)
        suppressive_threshold = get_param_value(p_pullback.get('suppressive_pullback_score_threshold'), 0.3)
        
        was_healthy_pullback = (self._get_atomic_score(df, 'COGNITIVE_SCORE_PULLBACK_HEALTHY_S').shift(1).fillna(0.0) > healthy_threshold)
        was_suppressive_pullback = (self._get_atomic_score(df, 'COGNITIVE_SCORE_PULLBACK_SUPPRESSIVE_S').shift(1).fillna(0.0) > suppressive_threshold)
        
        is_reversal_confirmed = fuse_multi_level_scores(df, 'BEHAVIOR_BOTTOM_REVERSAL') > 0.5
        
        late_stage_score = self._get_atomic_score(df, 'COGNITIVE_SCORE_CONTEXT_LATE_STAGE', 0.0)
        is_in_safe_stage = late_stage_score < 0.6
        
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
        【V1.6 · 信号适配版】压缩突破战术剧本合成模块
        """
        states = {}
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        
        vol_compression_score = fuse_multi_level_scores(df, 'VOL_COMPRESSION')
        states['COGNITIVE_SCORE_VOL_COMPRESSION_FUSED'] = vol_compression_score.astype(np.float32)
        
        setup_extreme_squeeze_score = vol_compression_score.shift(1).fillna(0.0)
        
        # 消费新的终极信号
        trigger_explosive_breakout_score = self._get_atomic_score(df, 'SCORE_VOL_BREAKOUT_POTENTIAL_S', 0.0)
        
        score_s_plus = (setup_extreme_squeeze_score * trigger_explosive_breakout_score).astype(np.float32)
        states['SCORE_PLAYBOOK_EXTREME_SQUEEZE_EXPLOSION_S_PLUS'] = score_s_plus
        states['PLAYBOOK_EXTREME_SQUEEZE_EXPLOSION_S_PLUS'] = score_s_plus > 0.7
        
        platform_quality_score = fuse_multi_level_scores(df, 'STRUCTURE_BULLISH_RESONANCE')
        breakout_eve_score = (platform_quality_score * vol_compression_score)
        setup_breakout_eve_score = breakout_eve_score.shift(1).fillna(0.0)
        trigger_prime_breakout_score = self._get_atomic_score(df, 'COGNITIVE_SCORE_IGNITION_RESONANCE_S', 0.0)
        score_s = (setup_breakout_eve_score * trigger_prime_breakout_score).astype(np.float32)
        states['SCORE_PLAYBOOK_BREAKOUT_EVE_S'] = score_s
        states['PLAYBOOK_BREAKOUT_EVE_S'] = score_s > 0.6
        
        return states
