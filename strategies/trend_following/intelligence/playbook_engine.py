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
        【V3.3 突破确认增强版】剧本蓝图知识库
        - 核心升级: 根据问题要求，将三个核心突破剧本的触发器从“瞬时点火”升级为“持续点火确认”，
                    以规避突破后的立刻回踩和洗盘。
        """
        # 修改: 整个方法被重写以使用新的元融合信号和剧本逻辑
        return [
            # --- S++级 王牌剧本 (S++ Class Ace Playbooks)
            {
                'name': 'PLAYBOOK_SUSTAINED_IGNITION_S_PLUS_PLUS',
                'setup': ['CONTEXT_TREND_QUALITY_HIGH', 'CONTEXT_NOT_LATE_STAGE'],
                'trigger': ['TRIGGER_SUSTAINED_IGNITION_S_PLUS'],
                'comment': 'S++级(王牌) - [持续点火] “多域点火”信号经过了后续观察期的考验，确认了趋势的可持续性，排除了早期的诱多洗盘陷阱。'
            },
            {
                'name': 'PLAYBOOK_PRIME_CHIP_IGNITION_S_PLUS_PLUS',
                'setup': ['CONTEXT_PRIME_CHIP_OPPORTUNITY_HIGH', 'CONTEXT_TREND_QUALITY_HIGH', 'CONTEXT_NOT_LATE_STAGE'],
                'trigger': ['TRIGGER_SUSTAINED_IGNITION_S_PLUS'],
                'comment': 'S++级(王牌) - [黄金筹码] 在“黄金筹码结构”形成后，由【经过确认的】“多域点火共振”确认总攻，规避回踩。'
            },
            # --- S+级 核心剧本 (S+ Class Core Playbooks) ---
            {
                'name': 'PLAYBOOK_CORE_HOLDER_IGNITION_S_PLUS',
                'setup': ['CONTEXT_CORE_HOLDER_ACCUMULATING', 'CONTEXT_TREND_QUALITY_MID_TO_HIGH'],
                'trigger': ['TRIGGER_SUSTAINED_CHIP_IGNITION_S_PLUS'],
                'comment': 'S+级 - [核心庄家] 在核心持仓者完成隐蔽吸筹后，由【经过确认的】“筹码点火”信号确认启动，规避洗盘。'
            },
            # --- S级王牌剧本 (S-Class Ace Playbooks) ---
            {
                'name': 'PLAYBOOK_IGNITION_RESONANCE_S',
                'setup': ['CONTEXT_TREND_QUALITY_HIGH', 'CONTEXT_NOT_LATE_STAGE'],
                'trigger': ['TRIGGER_SUSTAINED_IGNITION_S_PLUS'],
                'comment': 'S级 - [主升浪] 在趋势健康且非末期的主升浪中，由【经过确认的】“多域点火共振”确认总攻，规避回踩。'
            },
            {
                'name': 'PLAYBOOK_BOTTOM_REVERSAL_S',
                'setup': ['CONTEXT_BEARISH_EXHAUSTION_HIGH'],
                'trigger': ['TRIGGER_BOTTOM_REVERSAL_RESONANCE_S'],
                'comment': 'S级 - [底部反转] 在熊市衰竭的环境下，由“多域底部反转共振”确认战略拐点。'
            },
            # --- A+级 核心战术剧本 (A+ Class Core Playbooks) ---
            {
                'name': 'PLAYBOOK_WASHOUT_ABSORPTION_REVERSAL_A_PLUS',
                'setup': ['CONTEXT_WASHOUT_ABSORPTION_SETUP', 'CONTEXT_TREND_QUALITY_MID_TO_HIGH'],
                'trigger': ['TRIGGER_DOMINANT_REVERSAL'],
                'comment': 'A+级 - [洗盘反转] 在确认的“洗盘吸筹”行为后，由“显性反转K线”确认V型反转。'
            },
            # --- A级核心战术剧本 (A-Class Core Playbooks) ---
            {
                'name': 'PLAYBOOK_HEALTHY_PULLBACK_A',
                'setup': ['CONTEXT_TREND_QUALITY_HIGH', 'CONTEXT_NOT_LATE_STAGE'],
                'trigger': ['TRIGGER_HEALTHY_PULLBACK_CONFIRMED_S'],
                'comment': 'A级 - [健康回踩] 在趋势健康且非末期的主升浪中，捕捉经过确认的健康回踩买点。'
            },
            {
                'name': 'PLAYBOOK_CLASSIC_PATTERN_BREAKOUT_A',
                'setup': ['CONTEXT_TREND_QUALITY_MID_TO_HIGH'],
                'trigger': ['TRIGGER_CLASSIC_PATTERN_BREAKOUT_S'],
                'comment': 'A级 - [形态突破] 在趋势质量尚可的环境中，执行对经典看涨形态（老鸭头/N字板）的突破。'
            },
            {
                'name': 'PLAYBOOK_SHAKEOUT_REVERSAL_A',
                'setup': ['CONTEXT_TREND_QUALITY_MID_TO_HIGH'],
                'trigger': ['TRIGGER_SHAKEOUT_REVERSAL_A'],
                'comment': 'A级 - [洗盘反转] 在趋势质量尚可的环境中，捕捉“压缩区洗盘后反转”的V型机会。'
            },
            {
                'name': 'PLAYBOOK_CONSOLIDATION_BREAKOUT_A',
                'setup': ['CONTEXT_TREND_QUALITY_MID_TO_HIGH'],
                'trigger': ['TRIGGER_CONSOLIDATION_BREAKOUT_A'],
                'comment': 'A级 - [盘整突破] 在趋势质量尚可的环境中，执行对盘整中继形态的突破。'
            },
        ]

    def define_trigger_events(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.3 突破确认增强版】战术触发事件定义中心
        - 核心升级: 新增 S+ 级“持续筹码点火”触发器，用于过滤“筹码点火后洗盘”的假信号。
        - 核心逻辑: 要求“筹码点火”信号触发后，在3天观察期内，趋势质量不恶化，
                      且没有出现重大的顶部或崩溃风险信号，以此确认趋势的可持续性。
        """
        # print("        -> [触发事件中心 V3.3 突破确认增强版] 启动...")
        triggers = {}
        atomic = self.strategy.atomic_states
        default_series = pd.Series(False, index=df.index)
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 1. 定义动态阈值参数 ---
        p_triggers = get_params_block(self.strategy, 'meta_fusion_trigger_params', {})
        thresholds = {
            'ignition_s': get_param_value(p_triggers.get('ignition_s_threshold'), 0.5),
            'bottom_reversal_s': get_param_value(p_triggers.get('bottom_reversal_s_threshold'), 0.4),
            'healthy_pullback_s': get_param_value(p_triggers.get('healthy_pullback_s_threshold'), 0.3),
            'classic_pattern_s': get_param_value(p_triggers.get('classic_pattern_s_threshold'), 0.4),
            'shakeout_reversal_a': get_param_value(p_triggers.get('shakeout_reversal_a_threshold'), 0.3),
            'consolidation_breakout_a': get_param_value(p_triggers.get('consolidation_breakout_a_threshold'), 0.3),
            'earth_heaven_board': get_param_value(p_triggers.get('earth_heaven_board_threshold'), 0.6),
            'chip_ignition': get_param_value(p_triggers.get('chip_ignition_threshold'), 0.8),
            'invalidation_risk': get_param_value(p_triggers.get('invalidation_risk_threshold'), 0.5), # 点火失效的风险阈值
        }
        # --- 2. 保留并生成基础K线形态触发器 ---
        # 2.1 【通用级】反转确认阳线 (为“显性反转”提供基础)
        p_reversal = get_params_block(self.strategy, 'trigger_event_params', {}).get('reversal_confirmation_candle', {})
        is_green = df['close_D'] > df['open_D']
        is_strong_rally = df['pct_change_D'] > get_param_value(p_reversal.get('min_pct_change'), 0.03)
        is_closing_strong = df['close_D'] > (df['high_D'] + df['low_D']) / 2
        base_reversal_candle = is_green & is_strong_rally & is_closing_strong
        # 2.2 【精英级】显性反转阳线 (关键战术组件)
        p_dominant = get_params_block(self.strategy, 'trigger_event_params', {}).get('dominant_reversal_candle', {})
        today_body_size = df['close_D'] - df['open_D']
        yesterday_body_size = abs(df['close_D'].shift(1) - df['open_D'].shift(1))
        was_yesterday_red = df['close_D'].shift(1) < df['open_D'].shift(1)
        recovery_ratio = get_param_value(p_dominant.get('recovery_ratio'), 0.5)
        is_power_recovered = today_body_size >= (yesterday_body_size * recovery_ratio)
        triggers['TRIGGER_DOMINANT_REVERSAL'] = base_reversal_candle & (~was_yesterday_red | is_power_recovered)
        # --- 3. 生成核心进攻型元融合触发器 (Offensive Meta-Fusion Triggers) ---
        # 触发器 3.1: S级 - 多域点火共振 (最高级别突破信号)
        ignition_score = atomic.get('COGNITIVE_SCORE_IGNITION_RESONANCE_S', default_score)
        triggers['TRIGGER_IGNITION_RESONANCE_S'] = ignition_score > thresholds['ignition_s']
        # 触发器 3.2: S级 - 底部反转共振 (最高级别反转信号)
        bottom_reversal_score = atomic.get('COGNITIVE_SCORE_BOTTOM_REVERSAL_RESONANCE_S', default_score)
        triggers['TRIGGER_BOTTOM_REVERSAL_RESONANCE_S'] = bottom_reversal_score > thresholds['bottom_reversal_s']
        # --- 4. 生成特定战术场景触发器 (Tactical Triggers) ---
        # 触发器 4.1: S级 - 健康回踩确认
        healthy_pullback_score = atomic.get('COGNITIVE_SCORE_PULLBACK_HEALTHY_S', default_score)
        triggers['TRIGGER_HEALTHY_PULLBACK_CONFIRMED_S'] = (healthy_pullback_score.shift(1).fillna(0.0) > thresholds['healthy_pullback_s']) & triggers['TRIGGER_DOMINANT_REVERSAL']
        # 触发器 4.2: S级 - 经典形态突破
        classic_pattern_score = atomic.get('COGNITIVE_SCORE_CLASSIC_PATTERN_OPP_S', default_score)
        triggers['TRIGGER_CLASSIC_PATTERN_BREAKOUT_S'] = classic_pattern_score > thresholds['classic_pattern_s']
        # 触发器 4.3: A级 - 压缩区洗盘反转
        shakeout_reversal_score = atomic.get('COGNITIVE_SCORE_OPP_SQUEEZE_SHAKEOUT_REVERSAL_A', default_score)
        triggers['TRIGGER_SHAKEOUT_REVERSAL_A'] = shakeout_reversal_score > thresholds['shakeout_reversal_a']
        # 触发器 4.4: A级 - 盘整中继突破
        consolidation_breakout_score = atomic.get('COGNITIVE_SCORE_CONSOLIDATION_BREAKOUT_OPP_A', default_score)
        triggers['TRIGGER_CONSOLIDATION_BREAKOUT_A'] = consolidation_breakout_score > thresholds['consolidation_breakout_a']
        # 触发器 4.5: B级 - 地天板行为触发
        earth_heaven_score = atomic.get('SCORE_BOARD_EARTH_HEAVEN', default_score)
        triggers['TRIGGER_EARTH_HEAVEN_BOARD'] = earth_heaven_score > thresholds['earth_heaven_board']
        # --- 5. 筹码特定触发器 (Chip-Specific Triggers) ---
        # 触发器 5.1: S级 - 筹码点火确认
        chip_ignition_score = atomic.get('CHIP_SCORE_TRIGGER_IGNITION', default_score)
        triggers['TRIGGER_CHIP_IGNITION_CONFIRMED'] = chip_ignition_score > thresholds['chip_ignition']
        # --- 6. 定义“点火失效”的综合风险分 (用于所有持续点火逻辑) ---
        invalidation_risk_score = np.maximum.reduce([
            atomic.get('COGNITIVE_SCORE_TOP_REVERSAL_RESONANCE_S', default_score).values,
            atomic.get('COGNITIVE_SCORE_BREAKDOWN_RESONANCE_S', default_score).values,
            atomic.get('COGNITIVE_SCORE_TREND_FATIGUE_RISK', default_score).values,
            atomic.get('SCORE_RISK_UPTHRUST_DISTRIBUTION', default_score).values
        ])
        invalidation_risk_series = pd.Series(invalidation_risk_score, index=df.index)
        trend_quality_score = atomic.get('COGNITIVE_SCORE_TREND_QUALITY', default_score)
        confirmation_window = 3
        # --- 7. S++级 趋势确认触发器 (Trend Confirmation Trigger) ---
        # 7.1 基于“多域点火”的持续确认
        initial_ignition = triggers['TRIGGER_IGNITION_RESONANCE_S']
        had_recent_ignition = initial_ignition.rolling(window=confirmation_window, min_periods=1).max().astype(bool)
        max_risk_in_window = invalidation_risk_series.rolling(window=confirmation_window, min_periods=1).max()
        risk_remained_low = max_risk_in_window < thresholds['invalidation_risk']
        min_quality_in_window = trend_quality_score.rolling(window=confirmation_window, min_periods=1).min()
        quality_at_ignition = trend_quality_score.where(initial_ignition).ffill()
        quality_sustained = min_quality_in_window >= quality_at_ignition.shift(confirmation_window - 1).fillna(0.0) * 0.9
        is_confirmation_day = had_recent_ignition & ~had_recent_ignition.shift(1).fillna(False)
        triggers['TRIGGER_SUSTAINED_IGNITION_S_PLUS'] = is_confirmation_day & risk_remained_low & quality_sustained
        # --- 8. S+级 筹码趋势确认触发器 (Chip Trend Confirmation Trigger) --- # 新增的持续确认触发器逻辑
        # 8.1 基于“筹码点火”的持续确认
        initial_chip_ignition = triggers['TRIGGER_CHIP_IGNITION_CONFIRMED']
        had_recent_chip_ignition = initial_chip_ignition.rolling(window=confirmation_window, min_periods=1).max().astype(bool)
        # 复用相同的风险和趋势质量检查逻辑
        max_risk_in_window_chip = invalidation_risk_series.rolling(window=confirmation_window, min_periods=1).max()
        risk_remained_low_chip = max_risk_in_window_chip < thresholds['invalidation_risk']
        min_quality_in_window_chip = trend_quality_score.rolling(window=confirmation_window, min_periods=1).min()
        quality_at_chip_ignition = trend_quality_score.where(initial_chip_ignition).ffill()
        quality_sustained_chip = min_quality_in_window_chip >= quality_at_chip_ignition.shift(confirmation_window - 1).fillna(0.0) * 0.9
        is_confirmation_day_chip = had_recent_chip_ignition & ~had_recent_chip_ignition.shift(1).fillna(False)
        triggers['TRIGGER_SUSTAINED_CHIP_IGNITION_S_PLUS'] = is_confirmation_day_chip & risk_remained_low_chip & quality_sustained_chip
        print("        -> [新增触发器] 已生成 TRIGGER_SUSTAINED_CHIP_IGNITION_S_PLUS 用于确认筹码点火信号。")
        # --- 9. 最终安全检查 ---
        for key in list(triggers.keys()):
            triggers[key] = triggers[key].fillna(False)
        print(f"        -> [触发事件中心 V3.3] 分析完毕，生成 {len(triggers)} 个元融合触发器。")
        return triggers

    def generate_playbook_states(self, trigger_events: Dict[str, pd.Series]) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
        """
        【V3.1 筹码环境增强版】剧本状态生成引擎
        - 核心升级: 新增对“筹码结构状态”的战场环境定义，为新增的筹码剧本提供
                      精确的决策依据 (Setup Conditions)。
        - 收益: 使得剧本引擎能够理解并执行基于“黄金筹码”、“核心庄家吸筹”、
                “洗盘吸筹”等高级筹码概念的战术。
        """
        # 无需修改此方法，新的剧本和触发器可以被现有逻辑正确处理
        print("        -> [剧本状态生成引擎 V3.1 筹码环境增强版] 启动...")
        df = self.strategy.df_indicators
        if df.empty:
            print("          -> [警告] 主数据帧为空，无法生成剧本状态。")
            return {}, {}
        playbook_states = {}
        atomic_states = self.strategy.atomic_states
        default_series = pd.Series(False, index=df.index)
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 步骤 1: 定义战场环境 (Contextual Setups) ---
        # 基于顶层认知分数，生成用于剧本决策的布尔型“环境”信号
        contextual_setups = {}
        # 环境 1: 趋势质量
        trend_quality_score = atomic_states.get('COGNITIVE_SCORE_TREND_QUALITY', default_score)
        contextual_setups['CONTEXT_TREND_QUALITY_HIGH'] = trend_quality_score > 0.7
        contextual_setups['CONTEXT_TREND_QUALITY_MID_TO_HIGH'] = trend_quality_score > 0.5
        contextual_setups['CONTEXT_TREND_QUALITY_LOW'] = trend_quality_score < 0.3
        # 环境 2: 趋势阶段
        late_stage_score_raw = atomic_states.get('CONTEXT_TREND_LATE_STAGE_SCORE', pd.Series(0, index=df.index))
        # 假设分数超过320为上涨末期 (与 cognitive_intelligence 保持一致)
        contextual_setups['CONTEXT_IS_LATE_STAGE'] = late_stage_score_raw >= 320
        contextual_setups['CONTEXT_NOT_LATE_STAGE'] = late_stage_score_raw < 320
        # 环境 3: 衰竭程度
        bearish_exhaustion_score = atomic_states.get('COGNITIVE_SCORE_BEARISH_EXHAUSTION_OPP_S', default_score)
        contextual_setups['CONTEXT_BEARISH_EXHAUSTION_HIGH'] = bearish_exhaustion_score > 0.5
        # 环境 4: 筹码结构状态 (Chip Structure Contexts)
        prime_chip_score = atomic_states.get('CHIP_SCORE_PRIME_OPPORTUNITY_S', default_score)
        contextual_setups['CONTEXT_PRIME_CHIP_OPPORTUNITY_HIGH'] = prime_chip_score > 0.7
        core_holder_score = atomic_states.get('SCORE_CORE_HOLDER_BULLISH_RESONANCE_S', default_score)
        contextual_setups['CONTEXT_CORE_HOLDER_ACCUMULATING'] = core_holder_score > 0.7
        washout_absorption_score = atomic_states.get('CHIP_SCORE_FUSED_WASHOUT_ABSORPTION', default_score)
        contextual_setups['CONTEXT_WASHOUT_ABSORPTION_SETUP'] = washout_absorption_score > 0.6
        # --- 步骤 2: 循环执行剧本蓝图 ---
        for blueprint in self.playbook_blueprints:
            playbook_name = blueprint['name']
            # 2.1 组合 Setup 条件 (AND logic)
            # 剧本的 setup 条件必须全部满足
            setup_conditions = pd.Series(True, index=df.index)
            for state_name in blueprint.get('setup', []):
                # 从我们刚刚定义的环境信号中获取
                setup_conditions &= contextual_setups.get(state_name, default_series)
            # 2.2 组合 Trigger 条件 (OR logic)
            # 触发器满足任意一个即可
            trigger_conditions = pd.Series(False, index=df.index)
            for trigger_name in blueprint.get('trigger', []):
                trigger_conditions |= trigger_events.get(trigger_name, default_series)
            # 2.3 最终裁定: 昨日 Setup 就绪 & 今日 Trigger 触发
            final_signal = setup_conditions.shift(1).fillna(False) & trigger_conditions
            playbook_states[playbook_name] = final_signal
        print(f"        -> [剧本状态生成引擎 V3.1] 分析完毕，共生成 {len(playbook_states)} 个元融合剧本状态。")
        # setup_scores 在这个新架构下不再需要，返回空字典
        return {}, playbook_states



















