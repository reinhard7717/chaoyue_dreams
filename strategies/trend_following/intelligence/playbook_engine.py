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
        【V4.0 全面适配版】剧本蓝图知识库
        - 核心重构 (本次修改):
          - [全面适配] 废除了所有基于旧原子信号的剧本。
          - 新的剧本直接基于由 `define_trigger_events` 生成的、高质量的元融合触发器构建。
          - 剧本逻辑大大简化，因为复杂的“战备”条件已经被上游的元信号和触发器本身所包含。
        - 收益: 剧本库更精简、更强大，每个剧本都代表一个经过多重确认的高置信度战术场景。
        """
        return [
            # --- S++级 王牌剧本 (Ace Playbooks) ---
            {
                'name': 'PLAYBOOK_SUSTAINED_IGNITION_S_PLUS_PLUS',
                'trigger': ['TRIGGER_SUSTAINED_IGNITION_S_PLUS'],
                'comment': 'S++级(王牌) - [持续点火] “多域点火”信号经过了后续观察期的考验，确认了趋势的可持续性。'
            },
            {
                'name': 'PLAYBOOK_PRIME_CHIP_IGNITION_S_PLUS_PLUS',
                'trigger': ['TRIGGER_SUSTAINED_CHIP_IGNITION_S_PLUS'],
                'comment': 'S++级(王牌) - [黄金筹码] 在“黄金筹码结构”形成后，由【经过确认的】“筹码点火”确认总攻。'
            },
            # --- S级 核心剧本 (Core Playbooks) ---
            {
                'name': 'PLAYBOOK_BOTTOM_REVERSAL_S',
                'trigger': ['TRIGGER_BOTTOM_REVERSAL_RESONANCE_S'],
                'comment': 'S级 - [底部反转] 由“多域底部反转共振”确认战略拐点。'
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
        print("        -> [剧本状态生成引擎 V4.0 逻辑简化版] 启动...")
        df = self.strategy.df_indicators
        if df.empty:
            print("          -> [警告] 主数据帧为空，无法生成剧本状态。")
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
            
            # 最终裁定: 今日 Trigger 触发
            # 注意：新的剧本逻辑不再需要 .shift(1)，因为触发器本身已经包含了时序逻辑
            playbook_states[playbook_name] = trigger_conditions

        print(f"        -> [剧本状态生成引擎 V4.0] 分析完毕，共生成 {len(playbook_states)} 个元融合剧本状态。")
        # setup_scores 在这个新架构下不再需要，返回空字典
        return {}, playbook_states

    def define_trigger_events(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V4.0 全面适配版】战术触发事件定义中心
        - 核心重构 (本次修改):
          - [全面适配] 废除了所有对旧信号的引用，现在直接消费由各情报模块生成的、最新的S级/A级数值化分数。
          - [逻辑升级] 所有触发器都基于对数值化分数的阈值判断生成，逻辑更清晰、更鲁棒。
          - [新增] 增加了对“持续点火”的定义，用于过滤假突破，这是对A股实战经验的提炼。
        - 收益: 确保所有剧本的“扳机”都连接到了最新、最可靠的信号源。
        """
        print("        -> [触发事件中心 V4.0 全面适配版] 启动...")
        triggers = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        
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
        }

        # --- 2. 定义基础触发器 ---
        # 2.1 显性反转K线 (逻辑不变，但作为核心组件保留)
        p_dominant = p_triggers.get('dominant_reversal_candle', {})
        is_green = df['close_D'] > df['open_D']
        is_strong_rally = df['pct_change_D'] > get_param_value(p_dominant.get('min_pct_change'), 0.03)
        today_body_size = df['close_D'] - df['open_D']
        yesterday_body_size = abs(df['close_D'].shift(1) - df['open_D'].shift(1))
        was_yesterday_red = df['close_D'].shift(1) < df['open_D'].shift(1)
        recovery_ratio = get_param_value(p_dominant.get('recovery_ratio'), 0.5)
        is_power_recovered = today_body_size >= (yesterday_body_size * recovery_ratio)
        triggers['TRIGGER_DOMINANT_REVERSAL'] = is_green & is_strong_rally & (~was_yesterday_red | is_power_recovered)

        # --- 3. 定义元融合触发器 (基于CognitiveIntelligence的顶层信号) ---
        triggers['TRIGGER_IGNITION_RESONANCE_S'] = atomic.get('COGNITIVE_SCORE_IGNITION_RESONANCE_S', default_score) > thresholds['ignition_s']
        triggers['TRIGGER_BOTTOM_REVERSAL_RESONANCE_S'] = atomic.get('COGNITIVE_SCORE_BOTTOM_REVERSAL_RESONANCE_S', default_score) > thresholds['bottom_reversal_s']
        
        # --- 4. 定义特定战术场景触发器 ---
        triggers['TRIGGER_HEALTHY_PULLBACK_CONFIRMED_S'] = (atomic.get('COGNITIVE_SCORE_PULLBACK_HEALTHY_S', default_score).shift(1).fillna(0.0) > thresholds['healthy_pullback_s']) & triggers['TRIGGER_DOMINANT_REVERSAL']
        triggers['TRIGGER_CLASSIC_PATTERN_BREAKOUT_S'] = atomic.get('COGNITIVE_SCORE_CLASSIC_PATTERN_OPP_S', default_score) > thresholds['classic_pattern_s']
        triggers['TRIGGER_SHAKEOUT_REVERSAL_A'] = atomic.get('COGNITIVE_SCORE_OPP_SQUEEZE_SHAKEOUT_REVERSAL_A', default_score) > thresholds['shakeout_reversal_a']
        triggers['TRIGGER_CONSOLIDATION_BREAKOUT_A'] = atomic.get('COGNITIVE_SCORE_CONSOLIDATION_BREAKOUT_OPP_A', default_score) > thresholds['consolidation_breakout_a']
        
        # --- 5. 定义筹码特定触发器 ---
        triggers['TRIGGER_CHIP_IGNITION'] = atomic.get('CHIP_SCORE_PRIME_OPPORTUNITY_S', default_score) > thresholds['chip_ignition_s']

        # --- 6. 定义“持续点火”确认触发器 (过滤假突破的核心) ---
        # 6.1 定义失效条件：在观察期内出现重大风险
        invalidation_risk_score = np.maximum.reduce([
            atomic.get('COGNITIVE_SCORE_TOP_REVERSAL_RESONANCE_S', default_score).values,
            atomic.get('COGNITIVE_SCORE_BREAKDOWN_RESONANCE_S', default_score).values,
            atomic.get('COGNITIVE_SCORE_TREND_FATIGUE_RISK', default_score).values,
        ])
        invalidation_risk_series = pd.Series(invalidation_risk_score, index=df.index)
        trend_quality_score = atomic.get('COGNITIVE_SCORE_TREND_QUALITY', default_score)
        confirmation_window = 3
        
        # 6.2 基于“多域点火”的持续确认
        initial_ignition = triggers['TRIGGER_IGNITION_RESONANCE_S']
        had_recent_ignition = initial_ignition.rolling(window=confirmation_window, min_periods=1).max().astype(bool)
        max_risk_in_window = invalidation_risk_series.rolling(window=confirmation_window, min_periods=1).max()
        risk_remained_low = max_risk_in_window < thresholds['invalidation_risk']
        quality_at_ignition = trend_quality_score.where(initial_ignition).ffill()
        quality_sustained = trend_quality_score.rolling(window=confirmation_window).min() >= quality_at_ignition.shift(confirmation_window - 1).fillna(0.0) * 0.9
        is_confirmation_day = had_recent_ignition & ~had_recent_ignition.shift(1).fillna(False)
        triggers['TRIGGER_SUSTAINED_IGNITION_S_PLUS'] = is_confirmation_day & risk_remained_low & quality_sustained

        # 6.3 基于“筹码点火”的持续确认
        initial_chip_ignition = triggers['TRIGGER_CHIP_IGNITION']
        had_recent_chip_ignition = initial_chip_ignition.rolling(window=confirmation_window, min_periods=1).max().astype(bool)
        is_confirmation_day_chip = had_recent_chip_ignition & ~had_recent_chip_ignition.shift(1).fillna(False)
        triggers['TRIGGER_SUSTAINED_CHIP_IGNITION_S_PLUS'] = is_confirmation_day_chip & risk_remained_low & quality_sustained

        # --- 7. 最终安全检查 ---
        for key in list(triggers.keys()):
            triggers[key] = triggers[key].fillna(False)
        print(f"        -> [触发事件中心 V4.0] 分析完毕，生成 {len(triggers)} 个元融合触发器。")
        return triggers




















