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

    def _get_playbook_blueprints(self) -> List[Dict]:
        """
        【V5.0 剧本库扩充版】剧本蓝图知识库
        - 核心重构 (本次修改):
          - [剧本扩充] 基于现有的高质量认知层信号，新增了7个全新的、更贴近实战的战法剧本。
          - [逻辑归类] 将所有剧本按照 S++/S+/S/A 的级别进行分类，结构更清晰。
          - [命名统一] 统一了剧本和触发器的命名规范。
        - 收益: 极大地丰富了策略的战术武器库，能够捕捉更多样、更高置信度的交易机会。
        """
        return [
            # “筹码共振-价格滞后”潜伏突破剧本
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
            {
                'name': 'PLAYBOOK_WASHOUT_ABSORPTION_REVERSAL_A_PLUS',
                'trigger': ['TRIGGER_WASHOUT_ABSORPTION_REVERSAL_A_PLUS'],
                'comment': 'A+级 - [洗盘吸筹反转] 在确认的“洗盘吸筹”行为后，由“显性反转K线”确认V型反转。'
            },
            # --- A级 战术剧本 (Tactical Playbooks) ---
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
            # ▲▲▲ 修改结束 ▲▲▲
        ]

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

    def define_trigger_events(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V5.1 信号源修复版】战术触发事件定义中心
        - 核心重构 (V5.0):
          - [触发器扩充] 为所有新剧本定义了对应的触发器。
          - [阈值管理] 为所有新触发器增加了可配置的阈值。
        - 核心修复 (本次修改):
          - [信号修复] 修复了“常规压缩突破”触发器中对 `SCORE_SQUEEZE_BREAKOUT_OPP_S`
                        这个不存在信号的引用，替换为消费正确的 `COGNITIVE_SCORE_VOL_BREAKOUT_S` 信号。
        - 收益: 确保了“常规压缩突破”剧本的触发逻辑能够正常工作。
        """
        triggers = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        default_series = pd.Series(False, index=df.index)
        # --- 1. 定义动态阈值参数 ---
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
        }
        # --- 2. 定义基础触发器 ---
        p_dominant = p_triggers.get('dominant_reversal_candle', {})
        is_green = df['close_D'] > df['open_D']
        is_strong_rally = df['pct_change_D'] > get_param_value(p_dominant.get('min_pct_change'), 0.03)
        today_body_size = df['close_D'] - df['open_D']
        yesterday_body_size = abs(df['close_D'].shift(1) - df['open_D'].shift(1))
        was_yesterday_red = df['close_D'].shift(1) < df['open_D'].shift(1)
        recovery_ratio = get_param_value(p_dominant.get('recovery_ratio'), 0.5)
        is_power_recovered = today_body_size >= (yesterday_body_size * recovery_ratio)
        triggers['TRIGGER_DOMINANT_REVERSAL'] = is_green & is_strong_rally & (~was_yesterday_red | is_power_recovered)
        # --- 3. 定义元融合与战术场景触发器 (部分保留，部分新增) ---
        triggers['TRIGGER_BOTTOM_REVERSAL_RESONANCE_S'] = atomic.get('COGNITIVE_SCORE_BOTTOM_REVERSAL_RESONANCE_S', default_score) > thresholds['bottom_reversal_s']
        triggers['TRIGGER_HEALTHY_PULLBACK_CONFIRMED_S'] = (atomic.get('COGNITIVE_SCORE_PULLBACK_HEALTHY_S', default_score).shift(1).fillna(0.0) > thresholds['healthy_pullback_s']) & triggers['TRIGGER_DOMINANT_REVERSAL']
        triggers['TRIGGER_CLASSIC_PATTERN_BREAKOUT_S'] = atomic.get('COGNITIVE_SCORE_CLASSIC_PATTERN_OPP_S', default_score) > thresholds['classic_pattern_s']
        triggers['TRIGGER_SHAKEOUT_REVERSAL_A'] = atomic.get('COGNITIVE_SCORE_OPP_SQUEEZE_SHAKEOUT_REVERSAL_A', default_score) > thresholds['shakeout_reversal_a']
        triggers['TRIGGER_CONSOLIDATION_BREAKOUT_A'] = atomic.get('COGNITIVE_SCORE_CONSOLIDATION_BREAKOUT_OPP_A', default_score) > thresholds['consolidation_breakout_a']
        # 剧本: 洗盘吸筹后反转
        triggers['TRIGGER_WASHOUT_ABSORPTION_REVERSAL_A_PLUS'] = atomic.get('COGNITIVE_SCORE_OPP_BOTTOM_REVERSAL', default_score) > thresholds['washout_reversal_a_plus']
        # 剧本: 核心庄家点火
        triggers['TRIGGER_CORE_HOLDER_IGNITION_S_PLUS'] = atomic.get('TACTIC_LOCK_CHIP_RECONCENTRATION_S_PLUS', default_series)
        # 剧本: 主升浪点火
        is_in_main_uptrend = atomic.get('STRUCTURE_MAIN_UPTREND_WAVE_S', default_series)
        ignition_score = atomic.get('COGNITIVE_SCORE_IGNITION_RESONANCE_S', default_score)
        triggers['TRIGGER_UPTREND_IGNITION_RESONANCE_S'] = is_in_main_uptrend & (ignition_score > thresholds['ignition_s'])
        # 剧本: 压缩突破系列
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
        # 定义“筹码共振-价格滞后”剧本的专属触发器
        # 这个触发器直接实现了“昨日战备就绪，今日点火触发”的剧本逻辑
        was_setup_yesterday = atomic.get('SETUP_CHIP_RESONANCE_READY_S', default_series).shift(1).fillna(False)
        is_triggered_today = atomic.get('TRIGGER_GENTLE_PRICE_LIFT_A', default_series)
        # 最终的布尔触发信号
        triggers['TRIGGER_CHIP_RESONANCE_PRICE_LAG_BREAKOUT_S'] = was_setup_yesterday & is_triggered_today
        # --- 4. 定义“持续点火”确认触发器 (逻辑增强) ---
        # 4.1 定义基础点火事件
        initial_ignition = atomic.get('COGNITIVE_SCORE_IGNITION_RESONANCE_S', default_score) > thresholds['ignition_s']
        initial_chip_ignition = atomic.get('CHIP_SCORE_PRIME_OPPORTUNITY_S', default_score) > thresholds['chip_ignition_s']
        # 4.2 定义失效条件：在观察期内出现重大风险
        invalidation_risk_score = np.maximum.reduce([
            atomic.get('COGNITIVE_SCORE_TOP_REVERSAL_RESONANCE_S', default_score).values,
            atomic.get('COGNITIVE_SCORE_BREAKDOWN_RESONANCE_S', default_score).values,
            atomic.get('COGNITIVE_SCORE_TREND_FATIGUE_RISK', default_score).values,
        ])
        invalidation_risk_series = pd.Series(invalidation_risk_score, index=df.index)
        risk_remained_low = invalidation_risk_series.rolling(window=3, min_periods=1).max() < thresholds['invalidation_risk']
        # 4.3 定义“持续点火” - 多域点火
        # 检查过去N天内是否有过点火，并且今天不再是首次点火日
        had_recent_ignition = initial_ignition.rolling(window=3, min_periods=1).max().astype(bool)
        is_sustained_day = had_recent_ignition & ~initial_ignition
        triggers['TRIGGER_SUSTAINED_IGNITION_S_PLUS'] = is_sustained_day & risk_remained_low
        # 4.4 定义“持续点火” - 黄金筹码点火
        setup_prime_chip = atomic.get('CHIP_SCORE_PRIME_OPPORTUNITY_S', default_score) > 0.7
        triggers['TRIGGER_PRIME_CHIP_IGNITION_S_PLUS_PLUS'] = setup_prime_chip & triggers['TRIGGER_SUSTAINED_IGNITION_S_PLUS']
        # --- 5. 最终安全检查 ---
        for key in list(triggers.keys()):
            if key in triggers and isinstance(triggers[key], pd.Series):
                 triggers[key] = triggers[key].fillna(False)
        return triggers
















