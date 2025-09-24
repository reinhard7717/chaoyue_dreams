# 文件: strategies/trend_following/intelligence/playbook_engine.py
# 剧本与触发器引擎
import pandas as pd
from typing import Dict, Tuple, List
import numpy as np
from strategies.trend_following.utils import get_params_block, get_param_value

class PlaybookEngine:
    def __init__(self, strategy_instance):
        """
        初始化剧本与触发器引擎。
        :param strategy_instance: 策略主实例的引用。
        """
        self.strategy = strategy_instance
        self.playbook_blueprints = self._get_playbook_blueprints()
        self.kline_params = get_params_block(self.strategy, 'kline_pattern_params')
    def generate_playbook_states(self, trigger_events: Dict[str, pd.Series]) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
        """
        【V4.0 逻辑简化版】剧本状态生成引擎
        - 核心重构 (本次修改):
          - [逻辑简化] 移除了复杂的“战场环境(Contextual Setups)”定义。
          - 由于新的剧本蓝图直接依赖于高质量的触发器，这些触发器本身已经内含了对环境的判断，
            因此不再需要在此处进行重复的环境检查。
        - 收益: 引擎职责更纯粹，只负责根据蓝图执行“触发”逻辑，代码更简洁、高效。
        """
        df = self.strategy.df_indicators
        if df.empty:
            return {}, {}
        playbook_states = {}
        default_series = pd.Series(False, index=df.index)
        # --- 步骤 1: 循环执行剧本蓝图 ---
        for blueprint in self.playbook_blueprints:
            playbook_name = blueprint['name']
            # 组合 Trigger 条件 (OR logic)
            trigger_conditions = pd.Series(False, index=df.index)
            for trigger_name in blueprint.get('trigger', []):
                trigger_conditions |= trigger_events.get(trigger_name, default_series)
            playbook_states[playbook_name] = trigger_conditions
        # setup_scores 在这个新架构下不再需要，返回空字典
        return {}, playbook_states

    def _get_playbook_blueprints(self) -> List[Dict]:
        """
        【V6.1 均值回归扩充版】剧本蓝图知识库
        - 核心升级 (本次修改):
          - [王牌剧本] 新增 `PLAYBOOK_TRUE_ACCUMULATION_BREAKOUT_S_PLUS` 剧本，专门捕捉经过多维度交叉验证的“真实吸筹”后的突破机会。
          - [逆向剧本] 新增 `PLAYBOOK_CAPITULATION_REVERSAL_A_PLUS` 剧本，用于执行“恐慌盘投降反转”这一高胜率左侧交易。
          - [剧本优化] 将“筹码共振-价格滞后”剧本的触发器升级为消费更可靠的“真实吸筹”信号。
          - [战术扩充] 新增 `PLAYBOOK_MEAN_REVERSION_GRID_A` 剧本，用于执行震荡市的网格交易。
        """
        
        return [
            # 新增V型反转王牌剧本，并置于最前，体现其最高优先级
            {
                'name': 'PLAYBOOK_V_REVERSAL_ACE_S_PLUS',
                'trigger': ['TRIGGER_V_REVERSAL_ACE_S_PLUS'],
                'comment': 'S++级(王牌) - [V型反转] 在前一日确认的“恐慌抛售”后，由当日强劲的“显性反转K线”确认的最高确定性反转剧本。'
            },
            # --- 基于“真实吸筹”的王牌剧本 ---
            {
                'name': 'PLAYBOOK_TRUE_ACCUMULATION_BREAKOUT_S_PLUS',
                'trigger': ['TRIGGER_TRUE_ACCUMULATION_BREAKOUT_S_PLUS'],
                'comment': 'S++级(王牌) - [真实吸筹突破] 在确认的“真实吸筹”阶段后，由多域点火共振确认的突破，是最高置信度的买点之一。'
            },
            {
                'name': 'PLAYBOOK_CONVICTION_BREAKOUT_S_PLUS',
                'trigger': ['TRIGGER_CONVICTION_BREAKOUT_S_PLUS'],
                'comment': 'S++级(王牌) - [信念突破] 由资金流情报部门直接确认的、核心主力带头、买盘形成碾压的最高质量突破。'
            },
            # 大反转后期·共振初起 (逆向王牌)
            {
                'name': 'PLAYBOOK_POST_REVERSAL_RESONANCE_S_PLUS',
                'trigger': ['TRIGGER_POST_REVERSAL_RESONANCE_S_PLUS'],
                'comment': 'S++级(逆向王牌) - [大反转后期·共振初起] 在股价深度下跌、主力完成吸筹后，由趋势企稳和早期动能确认的、最安全的左侧转右侧买点。'
            },
            # --- 优化“筹码共振-价格滞后”剧本 ---
            {
                'name': 'PLAYBOOK_CHIP_RESONANCE_PRICE_LAG_BREAKOUT_S',
                'trigger': ['TRIGGER_CHIP_RESONANCE_PRICE_LAG_BREAKOUT_S'],
                'comment': 'S级(王牌潜伏) - [筹码共振-价格滞后] 在筹码高度共振、价格被压制的战备状态后，由温和点火信号确认的黄金突破口。'
            },
            # --- S++级 王牌剧本 (Ace Playbooks) ---
            {
                'name': 'PLAYBOOK_SUSTAINED_IGNITION_S_PLUS_PLUS',
                'trigger': ['TRIGGER_SUSTAINED_IGNITION_S_PLUS'],
                'comment': 'S++级(王牌) - [持续点火] “多域点火”信号经过了后续观察期的考验，确认了趋势的可持续性。'
            },
            {
                'name': 'PLAYBOOK_PRIME_CHIP_IGNITION_S_PLUS_PLUS',
                'trigger': ['TRIGGER_PRIME_CHIP_IGNITION_S_PLUS_PLUS'],
                'comment': 'S++级(王牌) - [黄金筹码] 在“黄金筹码结构”形成后，由【经过确认的】“多域点火共振”确认总攻。'
            },
            # --- S+级 核心增强剧本 (Enhanced Core Playbooks) ---
            {
                'name': 'PLAYBOOK_CORE_HOLDER_IGNITION_S_PLUS',
                'trigger': ['TRIGGER_CORE_HOLDER_IGNITION_S_PLUS'],
                'comment': 'S+级 - [核心庄家点火] 在核心持仓者完成隐蔽吸筹后，由“锁仓再集中”信号确认启动，规避洗盘。'
            },
            {
                'name': 'PLAYBOOK_EXTREME_SQUEEZE_EXPLOSION_S_PLUS',
                'trigger': ['TRIGGER_EXTREME_SQUEEZE_EXPLOSION_S_PLUS'],
                'comment': 'S+级 - [极致压缩突破] 在波动率极致压缩后，由高强度的压缩突破信号确认。'
            },
            # --- S级 核心剧本 (Core Playbooks) ---
            {
                'name': 'PLAYBOOK_IGNITION_RESONANCE_S',
                'trigger': ['TRIGGER_UPTREND_IGNITION_RESONANCE_S'],
                'comment': 'S级 - [主升浪点火] 在确认的主升浪结构中，由“多域点火共振”确认总攻，规避回踩。'
            },
            {
                'name': 'PLAYBOOK_BOTTOM_REVERSAL_S',
                'trigger': ['TRIGGER_BOTTOM_REVERSAL_RESONANCE_S'],
                'comment': 'S级 - [底部反转] 由“多域底部反转共振”确认战略拐点。'
            },
            {
                'name': 'PLAYBOOK_BREAKOUT_EVE_S',
                'trigger': ['TRIGGER_BREAKOUT_EVE_S'],
                'comment': 'S级 - [突破前夜] 在平台和波动率双重压缩的突破前夜，由最高级别的点火共振信号确认。'
            },
            # --- A+级 优势战术剧本 (Advantaged Tactical Playbooks) ---
            # --- 基于“恐慌盘投降”的逆向剧本 ---
            {
                'name': 'PLAYBOOK_CAPITULATION_REVERSAL_A_PLUS',
                'trigger': ['TRIGGER_CAPITULATION_REVERSAL_A_PLUS'],
                'comment': 'A+级(逆向) - [恐慌盘投降反转] 捕捉市场极度悲观，套牢盘集中割肉后的高胜率V型反转机会。'
            },
            {
                'name': 'PLAYBOOK_WASHOUT_ABSORPTION_REVERSAL_A_PLUS',
                'trigger': ['TRIGGER_WASHOUT_ABSORPTION_REVERSAL_A_PLUS'],
                'comment': 'A+级 - [洗盘吸筹反转] 在确认的“洗盘吸筹”行为后，由“显性反转K线”确认V型反转。'
            },
            # --- A级 战术剧本 (Tactical Playbooks) ---
            # 新增均值回归网格剧本
            {
                'name': 'PLAYBOOK_MEAN_REVERSION_GRID_A',
                'trigger': ['TRIGGER_MEAN_REVERSION_GRID_BUY_A'],
                'comment': 'A级 - [均值回归网格] 在震荡市中，于价格触及统计下轨时买入。'
            },
            {
                'name': 'PLAYBOOK_HEALTHY_PULLBACK_A',
                'trigger': ['TRIGGER_HEALTHY_PULLBACK_CONFIRMED_S'],
                'comment': 'A级 - [健康回踩] 捕捉经过确认的健康回踩买点。'
            },
            {
                'name': 'PLAYBOOK_CLASSIC_PATTERN_BREAKOUT_A',
                'trigger': ['TRIGGER_CLASSIC_PATTERN_BREAKOUT_S'],
                'comment': 'A级 - [形态突破] 执行对经典看涨形态的突破。'
            },
            {
                'name': 'PLAYBOOK_SHAKEOUT_REVERSAL_A',
                'trigger': ['TRIGGER_SHAKEOUT_REVERSAL_A'],
                'comment': 'A级 - [洗盘反转] 捕捉“压缩区洗盘后反转”的V型机会。'
            },
            {
                'name': 'PLAYBOOK_CONSOLIDATION_BREAKOUT_A',
                'trigger': ['TRIGGER_CONSOLIDATION_BREAKOUT_A'],
                'comment': 'A级 - [盘整突破] 执行对盘整中继形态的突破。'
            },
            {
                'name': 'PLAYBOOK_NORMAL_SQUEEZE_BREAKOUT_A',
                'trigger': ['TRIGGER_NORMAL_SQUEEZE_BREAKOUT_A'],
                'comment': 'A级 - [常规压缩突破] 在常规波动率压缩后，由任意压缩突破信号确认。'
            },
        ]

    def define_trigger_events(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V7.0 · 反转信号数值化版】战术触发事件定义中心
        - 核心重构 (本次修改):
          - [反转信号升维] 彻底重构了 `TRIGGER_DOMINANT_REVERSAL` 的生成逻辑。
          - [旧范式] 原逻辑是一个简单的布尔信号，无法区分反转的强度。
          - [新范式] 新逻辑是一个0-1之间的数值化评分，综合了“收复力度”、“位置优势”和“K线质量”三大维度，能精确量化反转的“成色”。
        - 收益: 为所有依赖此触发器的上游模块（如回踩战术、剧本等）提供了更精细、更可靠的输入，从信号链的源头提升了整个策略的决策质量。
        """
        print("      -> [战术触发事件定义中心 V7.0 · 反转信号数值化版] 启动...") # 更新版本号和说明
        triggers = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        default_series = pd.Series(False, index=df.index)
        p_triggers = get_params_block(self.strategy, 'trigger_event_params', {})
        thresholds = {
            'ignition_s': get_param_value(p_triggers.get('ignition_s_threshold'), 0.5),
            'bottom_reversal_s': get_param_value(p_triggers.get('bottom_reversal_s_threshold'), 0.4),
            'healthy_pullback_s': get_param_value(p_triggers.get('healthy_pullback_s_threshold'), 0.3),
            'classic_pattern_s': get_param_value(p_triggers.get('classic_pattern_s_threshold'), 0.4),
            'shakeout_reversal_a': get_param_value(p_triggers.get('shakeout_reversal_a_threshold'), 0.3),
            'consolidation_breakout_a': get_param_value(p_triggers.get('consolidation_breakout_a_threshold'), 0.3),
            'chip_ignition_s': get_param_value(p_triggers.get('chip_ignition_s_threshold'), 0.7),
            'invalidation_risk': get_param_value(p_triggers.get('invalidation_risk_threshold'), 0.5),
            'washout_reversal_a_plus': get_param_value(p_triggers.get('washout_reversal_a_plus_threshold'), 0.6),
            'extreme_squeeze_s_plus': get_param_value(p_triggers.get('extreme_squeeze_s_plus_threshold'), 0.7),
            'breakout_eve_s': get_param_value(p_triggers.get('breakout_eve_s_threshold'), 0.6),
            'normal_squeeze_a': get_param_value(p_triggers.get('normal_squeeze_a_threshold'), 0.5),
            'prime_chip_ignition_s_plus_plus': get_param_value(p_triggers.get('prime_chip_ignition_s_plus_plus_threshold'), 0.7),
            'chip_resonance_price_lag_s': get_param_value(p_triggers.get('chip_resonance_price_lag_s_threshold'), 0.5),
            'conviction_breakout_s_plus': get_param_value(p_triggers.get('conviction_breakout_s_plus_threshold'), 0.7),
            'true_accumulation_breakout_s_plus': get_param_value(p_triggers.get('true_accumulation_breakout_s_plus_threshold'), 0.6),
            'capitulation_reversal_a_plus': get_param_value(p_triggers.get('capitulation_reversal_a_plus_threshold'), 0.7),
            'post_reversal_resonance_s_plus': get_param_value(p_triggers.get('post_reversal_resonance_s_plus_threshold'), 0.65),
            'mean_reversion_grid_buy_a': get_param_value(p_triggers.get('mean_reversion_grid_buy_a_threshold'), 0.8),
        }
        # --- 重构 TRIGGER_DOMINANT_REVERSAL 为数值化评分 ---
        p_dominant = p_triggers.get('dominant_reversal_candle', {})
        # 维度1: 收复力度分 (Recovery Strength)
        today_body_size = (df['close_D'] - df['open_D']).clip(lower=0)
        yesterday_body_size = (df['open_D'].shift(1) - df['close_D'].shift(1)).clip(lower=0) # 只考虑前一日是阴线的情况
        recovery_score = (today_body_size / yesterday_body_size.replace(0, np.nan)).clip(0, 2).fillna(0) / 2.0 # 收复2倍及以上为满分
        # 维度2: 位置优势分 (Positional Advantage)
        lookback = get_param_value(p_dominant.get('position_lookback_days'), 60)
        rolling_high = df['high_D'].rolling(lookback).max()
        rolling_low = df['low_D'].rolling(lookback).min()
        price_range = (rolling_high - rolling_low).replace(0, 1e-9)
        price_position = ((df['close_D'] - rolling_low) / price_range).clip(0, 1).fillna(0.5)
        position_score = 1.0 - price_position # 价格位置越低，分数越高
        # 维度3: K线质量分 (Candle Quality) - 直接复用早期动能点火的逻辑
        candle_quality_score = atomic.get('COGNITIVE_SCORE_EARLY_MOMENTUM_IGNITION_A', default_score)
        # 最终融合: 三者相乘，得到最终的反转强度分
        dominant_reversal_score = (recovery_score * position_score * candle_quality_score).astype(np.float32)
        # 注意：这里仍然叫 TRIGGER_DOMINANT_REVERSAL，但其内容已是0-1的数值分，而不是布尔值
        triggers['TRIGGER_DOMINANT_REVERSAL'] = dominant_reversal_score
        # --- 为V型反转王牌剧本定义触发器 ---
        v_reversal_score = atomic.get('SCORE_PLAYBOOK_V_REVERSAL_ACE_S_PLUS', default_score)
        triggers['TRIGGER_V_REVERSAL_ACE_S_PLUS'] = v_reversal_score > thresholds['v_reversal_ace_s_plus']
        # --- 后续触发器定义逻辑保持不变，但它们现在消费的是一个更强大的数值化信号 ---
        triggers['TRIGGER_TRUE_ACCUMULATION_BREAKOUT_S_PLUS'] = (atomic.get('SCORE_CHIP_TRUE_ACCUMULATION', default_score).shift(1).fillna(0.0) > thresholds['true_accumulation_breakout_s_plus']) & (atomic.get('COGNITIVE_SCORE_IGNITION_RESONANCE_S', default_score) > thresholds['ignition_s'])
        triggers['TRIGGER_CAPITULATION_REVERSAL_A_PLUS'] = (atomic.get('SCORE_CHIP_PLAYBOOK_CAPITULATION_REVERSAL', default_score) > thresholds['capitulation_reversal_a_plus']) & (triggers['TRIGGER_DOMINANT_REVERSAL'] > 0.5) # 现在比较数值
        triggers['TRIGGER_CONVICTION_BREAKOUT_S_PLUS'] = atomic.get('SCORE_FF_PLAYBOOK_CONVICTION_BREAKOUT', default_score) > thresholds['conviction_breakout_s_plus']
        triggers['TRIGGER_POST_REVERSAL_RESONANCE_S_PLUS'] = atomic.get('COGNITIVE_SCORE_OPP_POST_REVERSAL_RESONANCE_A_PLUS', default_score) > thresholds['post_reversal_resonance_s_plus']
        triggers['TRIGGER_MEAN_REVERSION_GRID_BUY_A'] = atomic.get('SCORE_PLAYBOOK_MEAN_REVERSION_GRID_BUY_A', default_score) > thresholds['mean_reversion_grid_buy_a']
        was_true_accumulation_yesterday_for_lag = atomic.get('SCORE_CHIP_TRUE_ACCUMULATION', default_score).shift(1).fillna(0.0) > 0.5
        price_momentum_suppressed_score = (1 - self.strategy.df_indicators['SLOPE_5_close_D'].rolling(120).rank(pct=True)).fillna(0.5)
        was_price_suppressed_yesterday = price_momentum_suppressed_score.shift(1).fillna(0.0) > 0.8
        was_setup_yesterday_for_lag = was_true_accumulation_yesterday_for_lag & was_price_suppressed_yesterday
        is_gentle_lift_today = atomic.get('COGNITIVE_SCORE_EARLY_MOMENTUM_IGNITION_A', default_score) > 0.3 # 使用数值比较
        triggers['TRIGGER_CHIP_RESONANCE_PRICE_LAG_BREAKOUT_S'] = was_setup_yesterday_for_lag & is_gentle_lift_today
        triggers['TRIGGER_BOTTOM_REVERSAL_RESONANCE_S'] = atomic.get('COGNITIVE_SCORE_BOTTOM_REVERSAL_RESONANCE_S', default_score) > thresholds['bottom_reversal_s']
        triggers['TRIGGER_HEALTHY_PULLBACK_CONFIRMED_S'] = (atomic.get('COGNITIVE_SCORE_PULLBACK_HEALTHY_S', default_score).shift(1).fillna(0.0) > thresholds['healthy_pullback_s']) & (triggers['TRIGGER_DOMINANT_REVERSAL'] > 0.5) # 现在比较数值
        triggers['TRIGGER_CLASSIC_PATTERN_BREAKOUT_S'] = atomic.get('COGNITIVE_SCORE_CLASSIC_PATTERN_OPP_S', default_score) > thresholds['classic_pattern_s']
        triggers['TRIGGER_SHAKEOUT_REVERSAL_A'] = atomic.get('COGNITIVE_SCORE_OPP_SQUEEZE_SHAKEOUT_REVERSAL_A', default_score) > thresholds['shakeout_reversal_a']
        triggers['TRIGGER_CONSOLIDATION_BREAKOUT_A'] = atomic.get('COGNITIVE_SCORE_CONSOLIDATION_BREAKOUT_OPP_A', default_score) > thresholds['consolidation_breakout_a']
        triggers['TRIGGER_WASHOUT_ABSORPTION_REVERSAL_A_PLUS'] = atomic.get('COGNITIVE_SCORE_OPP_BOTTOM_REVERSAL', default_score) > thresholds['washout_reversal_a_plus']
        triggers['TRIGGER_CORE_HOLDER_IGNITION_S_PLUS'] = atomic.get('TACTIC_LOCK_CHIP_RECONCENTRATION_S_PLUS', default_series)
        is_in_main_uptrend = atomic.get('STRUCTURE_MAIN_UPTREND_WAVE_S', default_series)
        ignition_score = atomic.get('COGNITIVE_SCORE_IGNITION_RESONANCE_S', default_score)
        triggers['TRIGGER_UPTREND_IGNITION_RESONANCE_S'] = is_in_main_uptrend & (ignition_score > thresholds['ignition_s'])
        vol_compression_score = atomic.get('COGNITIVE_SCORE_VOL_COMPRESSION_FUSED', default_score)
        platform_quality_score = atomic.get('SCORE_PLATFORM_QUALITY_S', default_score)
        squeeze_breakout_score = atomic.get('COGNITIVE_SCORE_VOL_BREAKOUT_S', default_score)
        vol_breakout_a_score = atomic.get('COGNITIVE_SCORE_VOL_BREAKOUT_A', default_score)
        setup_extreme_squeeze = vol_compression_score > 0.9
        triggers['TRIGGER_EXTREME_SQUEEZE_EXPLOSION_S_PLUS'] = setup_extreme_squeeze.shift(1).fillna(False) & (squeeze_breakout_score > thresholds['extreme_squeeze_s_plus'])
        setup_breakout_eve = (platform_quality_score * vol_compression_score) > 0.7
        triggers['TRIGGER_BREAKOUT_EVE_S'] = setup_breakout_eve.shift(1).fillna(False) & (ignition_score > thresholds['breakout_eve_s'])
        setup_normal_squeeze = vol_compression_score > 0.5
        any_breakout_trigger = np.maximum(squeeze_breakout_score, vol_breakout_a_score) > thresholds['normal_squeeze_a']
        triggers['TRIGGER_NORMAL_SQUEEZE_BREAKOUT_A'] = setup_normal_squeeze.shift(1).fillna(False) & any_breakout_trigger & ~triggers['TRIGGER_EXTREME_SQUEEZE_EXPLOSION_S_PLUS']
        initial_ignition = atomic.get('COGNITIVE_SCORE_IGNITION_RESONANCE_S', default_score) > thresholds['ignition_s']
        invalidation_risk_score = np.maximum.reduce([
            atomic.get('COGNITIVE_SCORE_TOP_REVERSAL_RESONANCE_S', default_score).values,
            atomic.get('COGNITIVE_SCORE_BREAKDOWN_RESONANCE_S', default_score).values,
            atomic.get('COGNITIVE_SCORE_TREND_FATIGUE_RISK', default_score).values,
        ])
        invalidation_risk_series = pd.Series(invalidation_risk_score, index=df.index)
        risk_remained_low = invalidation_risk_series.rolling(window=3, min_periods=1).max() < thresholds['invalidation_risk']
        had_recent_ignition = initial_ignition.rolling(window=3, min_periods=1).max().astype(bool)
        is_sustained_day = had_recent_ignition & ~initial_ignition
        triggers['TRIGGER_SUSTAINED_IGNITION_S_PLUS'] = is_sustained_day & risk_remained_low
        setup_prime_chip = atomic.get('CHIP_SCORE_PRIME_OPPORTUNITY_S', default_score) > 0.7
        triggers['TRIGGER_PRIME_CHIP_IGNITION_S_PLUS_PLUS'] = setup_prime_chip & triggers['TRIGGER_SUSTAINED_IGNITION_S_PLUS']
        for key in list(triggers.keys()):
            if key in triggers and isinstance(triggers[key], pd.Series):
                 triggers[key] = triggers[key].fillna(False)
        return triggers
















