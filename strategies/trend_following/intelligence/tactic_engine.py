# 文件: strategies/trend_following/intelligence/tactic_engine.py
# 战术与剧本引擎
import pandas as pd
import numpy as np
from typing import Dict
from strategies.trend_following.utils import get_params_block, get_param_value, get_unified_score, normalize_score

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

    def run_tactic_synthesis(self, df: pd.DataFrame, pullback_enhancements: Dict) -> Dict[str, pd.Series]:
        """
        【V2.3 · 赫尔墨斯信使版】战术引擎总指挥
        - 核心革命: 1. 彻底移除了对全局状态的非法写入。
                      2. 识别到 synthesize_v_reversal_ace_playbook 的隐式依赖后，为其建立了
                         明确的参数通道（信使），将依赖数据直接传递，根除了“焦土战术”BUG。
        """
        all_states = {}
        # 步骤 1: 计算具有依赖性的前置信号
        panic_states = self.synthesize_panic_selling_setup(df)
        all_states.update(panic_states)
        # [代码删除] 移除了对 self.strategy.atomic_states 的非法写入，切断“黑市交易”
        # 步骤 2: 通过“信使通道”（函数参数）将依赖直接传递给下游方法
        all_states.update(self.synthesize_v_reversal_ace_playbook(df, setup_score=panic_states.get('SCORE_SETUP_PANIC_SELLING')))
        all_states.update(self.synthesize_chip_price_lag_playbook(df))
        all_states.update(self.synthesize_prime_tactic(df))
        all_states.update(self._diagnose_pullback_tactics_matrix(df, pullback_enhancements))
        all_states.update(self.synthesize_squeeze_playbooks(df))
        return all_states

    def synthesize_panic_selling_setup(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.2 · 信号净化版】恐慌抛售战备(Setup)信号生成模块
        - 核心逻辑: 价格大幅下跌 * 成交量放大 * 筹码结构崩溃
        """
        # print("        -> [恐慌抛售战备模块 V1.2] 启动...") # 更新版本号
        states = {}
        price_drop_score = normalize_score(df['pct_change_D'].clip(upper=0), df.index, window=60, ascending=False)
        volume_spike_score = normalize_score(df['volume_D'] / df['VOL_MA_21_D'], df.index, window=60, ascending=True)
        
        # 消费新的终极筹码看跌信号 (已净化)
        chip_breakdown_score = get_unified_score(self.strategy.atomic_states, df.index, 'CHIP_BEARISH_RESONANCE')
        
        setup_panic_selling_score = (price_drop_score * volume_spike_score * chip_breakdown_score).astype(np.float32)
        # 确保生产的信号名是净化后的
        states['SCORE_SETUP_PANIC_SELLING'] = setup_panic_selling_score
        return states

    def synthesize_v_reversal_ace_playbook(self, df: pd.DataFrame, setup_score: pd.Series) -> Dict[str, pd.Series]:
        """
        【V1.3 · 赫尔墨斯信使版】V型反转王牌剧本
        - 核心革命: 不再从全局状态（黑市）读取 setup_score，而是通过函数参数（信使）接收，
                      确保了数据来源的纯净、可靠和可追溯。
        """
        states = {}
        trigger_dominant_reversal_score = get_unified_score(self.strategy.atomic_states, df.index, 'BEHAVIOR_BOTTOM_REVERSAL')
        # 不再从全局状态读取，而是直接使用通过参数传递的、最新的依赖数据
        if setup_score is None:
            setup_score = pd.Series(0.0, index=df.index)
        was_setup_yesterday = setup_score.shift(1).fillna(0.0)
        is_triggered_today = trigger_dominant_reversal_score
        final_playbook_score = (was_setup_yesterday * is_triggered_today).astype(np.float32)
        states['SCORE_PLAYBOOK_V_REVERSAL_ACE'] = final_playbook_score
        states['PLAYBOOK_V_REVERSAL_ACE'] = final_playbook_score > 0.3
        return states

    def synthesize_chip_price_lag_playbook(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.2 · 信号净化版】“筹码共振-价格滞后”战术剧本
        - 核心逻辑: (筹码共振 * 价格压制 * 波动压缩) * 价格温和启动
        """
        states = {}
        
        # 消费新的终极信号 (已净化)
        chip_resonance_score = get_unified_score(self.strategy.atomic_states, df.index, 'CHIP_BULLISH_RESONANCE')
        price_momentum_suppressed_score = normalize_score(df['SLOPE_5_close_D'], df.index, window=60, ascending=False)
        volatility_compression_score = get_unified_score(self.strategy.atomic_states, df.index, 'VOL_COMPRESSION')
        
        setup_score = (chip_resonance_score * price_momentum_suppressed_score * volatility_compression_score).astype(np.float32)
        # 确保生产的信号名是净化后的
        states['SCORE_SETUP_CHIP_RESONANCE_READY'] = setup_score
        
        # 消费净化后的点火信号。注意：这要求 micro_behavior_engine 产出的信号被净化为 'COGNITIVE_SCORE_EARLY_MOMENTUM_IGNITION'
        trigger_score = self._get_atomic_score(df, 'COGNITIVE_SCORE_EARLY_MOMENTUM_IGNITION', 0.0)
        
        final_score = setup_score.shift(1).fillna(0.0) * trigger_score
        # 确保生产的信号名是净化后的
        states['SCORE_PLAYBOOK_CHIP_PRICE_LAG'] = final_score
        states['PLAYBOOK_CHIP_PRICE_LAG'] = final_score > 0.5
        return states

    def synthesize_prime_tactic(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.6 · 信号净化版】终极战法合成模块
        - 核心逻辑: (黄金筹码结构 * 极致波动压缩 * 能量优势) * 点火共振
        """
        states = {}
        # 消费新的终极信号 (已净化)
        is_prime_chip_structure = get_unified_score(self.strategy.atomic_states, df.index, 'CHIP_BULLISH_RESONANCE') > 0.7
        fused_compression_score = get_unified_score(self.strategy.atomic_states, df.index, 'VOL_COMPRESSION')
        is_extreme_squeeze = fused_compression_score > 0.9
        has_energy_advantage = get_unified_score(self.strategy.atomic_states, df.index, 'DYN_BULLISH_RESONANCE') > 0.7
        
        condition_sum = (is_prime_chip_structure.astype(int) + is_extreme_squeeze.astype(int) + has_energy_advantage.astype(int))
        setup_prime = (condition_sum == 3)
        # 确保生产的信号名是净化后的
        states['SETUP_PRIME_STRUCTURE'] = setup_prime
        
        # 消费净化后的点火信号。注意：这要求 cognitive_intelligence 产出的信号被净化为 'COGNITIVE_SCORE_IGNITION_RESONANCE'
        ignition_resonance_score = self._get_atomic_score(df, 'COGNITIVE_SCORE_IGNITION_RESONANCE', 0.0)
        trigger_prime_breakout = ignition_resonance_score > 0.6
        
        was_setup_prime_yesterday = setup_prime.shift(1).fillna(False)
        final_tactic = was_setup_prime_yesterday & trigger_prime_breakout
        # 确保生产的信号名是净化后的
        states['TACTIC_PRIME_STRUCTURE_BREAKOUT'] = final_tactic
        return states

    def _diagnose_pullback_tactics_matrix(self, df: pd.DataFrame, enhancements: Dict) -> Dict[str, pd.Series]:
        """
        【V7.6 · 信号净化版】回踩战术诊断模块
        """
        states = {}
        
        # 消费新的终极信号 (已净化)
        ascent_start_event = get_unified_score(self.strategy.atomic_states, df.index, 'STRUCTURE_BULLISH_RESONANCE') > 0.6
        chip_resonance_score = get_unified_score(self.strategy.atomic_states, df.index, 'CHIP_BULLISH_RESONANCE')
        ff_resonance_score = get_unified_score(self.strategy.atomic_states, df.index, 'FF_BULLISH_RESONANCE')
        cruise_start_event = (chip_resonance_score * ff_resonance_score) > 0.7

        lookback_window = 15
        is_in_ascent_window = ascent_start_event.rolling(window=lookback_window, min_periods=1).max().astype(bool)
        is_in_cruise_window = cruise_start_event.rolling(window=lookback_window, min_periods=1).max().astype(bool)
        
        p_pullback = get_params_block(self.strategy, 'pullback_tactics_params', {})
        healthy_threshold = get_param_value(p_pullback.get('healthy_pullback_score_threshold'), 0.3)
        suppressive_threshold = get_param_value(p_pullback.get('suppressive_pullback_score_threshold'), 0.3)
        
        # 消费净化后的回踩信号。注意：这要求 cognitive_intelligence 产出的信号被净化
        was_healthy_pullback = (self._get_atomic_score(df, 'COGNITIVE_SCORE_PULLBACK_HEALTHY').shift(1).fillna(0.0) > healthy_threshold)
        was_suppressive_pullback = (self._get_atomic_score(df, 'COGNITIVE_SCORE_PULLBACK_SUPPRESSIVE').shift(1).fillna(0.0) > suppressive_threshold)
        
        is_reversal_confirmed = get_unified_score(self.strategy.atomic_states, df.index, 'BEHAVIOR_BOTTOM_REVERSAL') > 0.5
        
        # 消费净化后的阶段信号。
        late_stage_score = self._get_atomic_score(df, 'COGNITIVE_SCORE_CONTEXT_LATE_STAGE', 0.0)
        is_in_safe_stage = late_stage_score < 0.6
        
        # 确保所有生产的信号名都是净化后的
        cruise_pit_reversal_signal = is_in_cruise_window & was_suppressive_pullback & is_reversal_confirmed
        states['TACTIC_CRUISE_PIT_REVERSAL'] = cruise_pit_reversal_signal
        
        cruise_pullback_reversal_signal = is_in_cruise_window & was_healthy_pullback & is_reversal_confirmed & is_in_safe_stage & ~cruise_pit_reversal_signal
        states['TACTIC_CRUISE_PULLBACK_REVERSAL'] = cruise_pullback_reversal_signal
        
        ascent_pit_reversal_signal = is_in_ascent_window & was_suppressive_pullback & is_reversal_confirmed & ~is_in_cruise_window
        states['TACTIC_ASCENT_PIT_REVERSAL'] = ascent_pit_reversal_signal
        
        ascent_pullback_reversal_signal = is_in_ascent_window & was_healthy_pullback & is_reversal_confirmed & ~is_in_cruise_window & ~ascent_pit_reversal_signal
        states['TACTIC_ASCENT_PULLBACK_REVERSAL'] = ascent_pullback_reversal_signal
        return states

    def synthesize_squeeze_playbooks(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.7 · 信号净化版】压缩突破战术剧本合成模块
        """
        states = {}
        
        vol_compression_score = get_unified_score(self.strategy.atomic_states, df.index, 'VOL_COMPRESSION')
        states['COGNITIVE_SCORE_VOL_COMPRESSION_FUSED'] = vol_compression_score.astype(np.float32)
        
        setup_extreme_squeeze_score = vol_compression_score.shift(1).fillna(0.0)
        
        # 消费净化后的突破潜力信号。注意：这要求 cognitive_intelligence 产出的信号被净化
        trigger_explosive_breakout_score = self._get_atomic_score(df, 'SCORE_VOL_BREAKOUT_POTENTIAL', 0.0)
        
        score_extreme_squeeze = (setup_extreme_squeeze_score * trigger_explosive_breakout_score).astype(np.float32)
        # 确保生产的信号名是净化后的
        states['SCORE_PLAYBOOK_EXTREME_SQUEEZE_EXPLOSION'] = score_extreme_squeeze
        states['PLAYBOOK_EXTREME_SQUEEZE_EXPLOSION'] = score_extreme_squeeze > 0.7
        
        platform_quality_score = get_unified_score(self.strategy.atomic_states, df.index, 'STRUCTURE_BULLISH_RESONANCE')
        breakout_eve_score = (platform_quality_score * vol_compression_score)
        setup_breakout_eve_score = breakout_eve_score.shift(1).fillna(0.0)
        
        # 消费净化后的点火信号。注意：这要求 cognitive_intelligence 产出的信号被净化
        trigger_prime_breakout_score = self._get_atomic_score(df, 'COGNITIVE_SCORE_IGNITION_RESONANCE', 0.0)
        
        score_breakout_eve = (setup_breakout_eve_score * trigger_prime_breakout_score).astype(np.float32)
        # 确保生产的信号名是净化后的
        states['SCORE_PLAYBOOK_BREAKOUT_EVE'] = score_breakout_eve
        states['PLAYBOOK_BREAKOUT_EVE'] = score_breakout_eve > 0.6
        
        return states
