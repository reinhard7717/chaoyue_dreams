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
        【V1.8 · 最终审判版】恐慌抛售战备(Setup)信号生成模块
        - 核心革命: “绝望背景”支柱正式从旧的单维模型升级为与行为引擎一致的“冥河之渡”多维立体模型。
        - 最终形态: 恐慌战备分 = (价格暴跌 * 成交天量 * 筹码崩溃) × (多维绝望背景) × (结构支撑测试)
        - 收益: 实现了全系统内对“恐慌”定义的最终统一，修复了与法医探针的逻辑断层。
        """
        states = {}
        p_panic = get_params_block(self.strategy, 'panic_selling_setup_params', {})

        # --- 支柱一、二、三：保持不变 ---
        price_drop_score = normalize_score(df['pct_change_D'].clip(upper=0), df.index, window=60, ascending=False)
        volume_spike_score = normalize_score(df['volume_D'] / df['VOL_MA_21_D'], df.index, window=60, ascending=True)
        chip_breakdown_score = get_unified_score(self.strategy.atomic_states, df.index, 'CHIP_BEARISH_RESONANCE')
        
        # --- 支柱四：调用全新的“冥河之渡”引擎计算多维绝望背景分 ---
        despair_context_score = self._calculate_despair_context_score(df, p_panic)

        # --- 支柱五：调用“绝对领域”引擎计算结构测试分 (保持不变) ---
        structural_test_score = self._calculate_structural_test_score(df, p_panic)

        # --- 五位一体融合，生成终极恐慌信号 ---
        setup_panic_selling_score = (
            price_drop_score * 
            volume_spike_score * 
            chip_breakdown_score * 
            despair_context_score *
            structural_test_score
        ).astype(np.float32)
        
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

    def _calculate_structural_test_score(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【V1.1 · 战神之矛版】“绝对领域”结构共振测试引擎
        - 核心革命: 引入“战神之矛”——重要看涨K线(SBC)的低点，作为最高权重的支撑。
        - 核心逻辑: 1. 识别并追踪近期所有SBC低点。
                      2. 在支撑矩阵中赋予SBC低点最高权重。
                      3. 当SBC低点与其他支撑形成共振时，提供巨额奖励。
        - 收益: 能够精准识别对主力进攻成本线的关键测试，极大提升信号的战略价值。
        """
        # --- 步骤 1: 获取参数，定义支撑矩阵 ---
        support_periods = get_param_value(params.get('support_lookback_periods'), [5, 10, 21, 55])
        # 权重字典增加对'sbc'（战神之矛）的定义，并赋予最高权重
        period_weights = get_param_value(params.get('support_period_weights'), {5: 0.6, 10: 0.8, 21: 1.0, 55: 1.2, 'sbc': 1.5})
        support_tolerance_pct = get_param_value(params.get('support_tolerance_pct'), 0.01)
        confluence_bonus_factor = get_param_value(params.get('confluence_bonus_factor'), 0.2)
        sbc_threshold_pct = get_param_value(params.get('sbc_threshold_pct'), 0.05) # 定义SBC的涨幅阈值

        # [代码新增] 锻造“战神之矛”：识别并追踪“重要看涨K线”的低点
        is_sbc = (df['pct_change_D'] > sbc_threshold_pct) & (df['volume_D'] > df.get('VOL_MA_21_D', 0))
        # 使用.where(is_sbc)提取SBC日的低点，然后用ffill()向前填充，得到每个交易日看到的、最近的那个SBC低点
        recent_sbc_low = df['low_D'].where(is_sbc).ffill()

        # 将“战神之矛”加入支撑矩阵
        support_levels = {f'EMA_{p}_D': df.get(f'EMA_{p}_D') for p in [5, 10, 21]}
        for p in support_periods:
            support_levels[f'PrevLow{p}'] = df['low_D'].shift(1).rolling(p, min_periods=max(1, p//2)).min()
        support_levels['RecentSBCLow'] = recent_sbc_low.shift(1) # 使用shift(1)确保我们测试的是历史结构

        valid_supports = {k: v for k, v in support_levels.items() if v is not None and not v.empty}
        if not valid_supports:
            return pd.Series(0.0, index=df.index)
        
        supports_df = pd.concat(valid_supports, axis=1)

        # --- 步骤 2: 计算“结构共振”强度 (逻辑不变) ---
        confluence_df = pd.DataFrame(1.0, index=df.index, columns=supports_df.columns)
        for col_i in supports_df.columns:
            for col_j in supports_df.columns:
                if col_i == col_j: continue
                # 增加对NaN的健壮性处理
                is_close = (supports_df[col_i] - supports_df[col_j]).abs() / supports_df[col_i].replace(0, np.nan) < support_tolerance_pct
                confluence_df[col_i] += is_close.astype(float)
        
        confluence_bonus_df = 1.0 + (confluence_df - 1) * confluence_bonus_factor

        # --- 步骤 3: 计算所有支撑位的加权、共振调整后的测试分数 ---
        all_test_scores = []
        day_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        atr = df.get('ATR_14_D', day_range)

        for name, support_series in valid_supports.items():
            # 升级权重获取逻辑，以兼容'sbc'等非数字键
            if 'SBC' in name:
                weight = period_weights.get('sbc', 1.5)
            else:
                period = int(''.join(filter(str.isdigit, name))) if any(char.isdigit() for char in name) else 21
                weight = period_weights.get(period, 1.0)
            
            confluence_bonus = confluence_bonus_df[name]

            # 3.1 计算“被接住”分数
            tolerance_buffer = (support_series * support_tolerance_pct).replace(0, np.nan)
            distance = (df['low_D'] - support_series).abs()
            base_proximity_score = np.exp(-((distance / tolerance_buffer)**2)).fillna(0)
            weighted_proximity_score = base_proximity_score * weight * confluence_bonus
            all_test_scores.append(weighted_proximity_score)

            # 3.2 计算“破位收回”分数 (仅对前低和SBC低点有效)
            if 'PrevLow' in name or 'SBC' in name:
                is_spring = (df['low_D'] < support_series) & (df['close_D'] > support_series)
                reclaim_strength = ((df['close_D'] - support_series) / day_range).clip(0, 1)
                base_reclaim_score = (is_spring * reclaim_strength).fillna(0)
                weighted_reclaim_score = base_reclaim_score * weight * confluence_bonus
                all_test_scores.append(weighted_reclaim_score)

        # --- 步骤 4: 融合所有测试分数，取当日最强的结构事件 (逻辑不变) ---
        if not all_test_scores:
            return pd.Series(0.0, index=df.index)
            
        final_score_matrix = pd.concat(all_test_scores, axis=1)
        final_structural_test_score = final_score_matrix.max(axis=1, skipna=True).fillna(0.0)
        
        return final_structural_test_score.clip(0, 1)

    def _calculate_despair_context_score(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【V1.0 · 新增/移植】“冥河之渡”多维绝望背景诊断引擎
        - 来源: 从 behavioral_intelligence 完美移植而来，作为“最终审判”计划的一部分。
        - 核心职责: 为本模块提供与行为引擎完全一致的、最高规格的绝望背景计算能力。
        """
        # --- 步骤 1: 获取参数 ---
        despair_periods = get_param_value(params.get('despair_periods'), {'short': (21, 5), 'mid': (60, 21), 'long': (250, 60)})
        despair_weights = get_param_value(params.get('despair_weights'), {'short': 0.2, 'mid': 0.3, 'long': 0.5})
        
        period_scores = []
        period_weight_values = []

        # --- 步骤 2: 遍历所有绝望周期，独立计算分数 ---
        for name, (drawdown_period, roc_period) in despair_periods.items():
            # 2.1 计算该周期的“坠落深度”
            rolling_peak = df['high_D'].rolling(window=drawdown_period, min_periods=max(1, drawdown_period//2)).max()
            drawdown_from_peak = (rolling_peak - df['close_D']) / rolling_peak.replace(0, np.nan)
            magnitude_score = normalize_score(drawdown_from_peak.clip(lower=0), df.index, window=drawdown_period, ascending=True)
            
            # 2.2 计算该周期的“坠落速度”
            price_roc = df['close_D'].pct_change(roc_period)
            velocity_score = normalize_score(price_roc, df.index, window=drawdown_period, ascending=False)
            
            # 2.3 融合得到该周期的绝望分数
            period_despair_score = (magnitude_score * velocity_score)**0.5
            
            period_scores.append(period_despair_score.values)
            period_weight_values.append(despair_weights.get(name, 0.0))

        # --- 步骤 3: 对所有周期的绝望分数进行加权几何平均 ---
        if not period_scores:
            return pd.Series(0.0, index=df.index)

        weights_array = np.array(period_weight_values)
        total_weights = weights_array.sum()
        if total_weights > 0:
            weights_array /= total_weights
        else:
            weights_array = np.full_like(weights_array, 1.0 / len(weights_array))

        stacked_scores = np.stack(period_scores, axis=0)
        
        final_score_values = np.prod(stacked_scores ** weights_array[:, np.newaxis], axis=0)
        
        return pd.Series(final_score_values, index=df.index, dtype=np.float32)












