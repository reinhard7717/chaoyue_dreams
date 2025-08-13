# 文件: strategies/trend_following/intelligence/cognitive_intelligence.py
# 顶层认知合成模块
import pandas as pd
import numpy as np
from typing import Dict
from enum import Enum
from strategies.trend_following.utils import get_params_block, get_param_value

class MainForceState(Enum):
    """
    定义主力行为序列的各个状态。
    """
    IDLE = 0           # 闲置/观察期
    ACCUMULATING = 1   # 吸筹期
    WASHING = 2        # 洗盘期
    MARKUP = 3         # 拉升期
    DISTRIBUTING = 4   # 派发期
    COLLAPSE = 5       # 崩盘期

def _create_decaying_influence_series(event_series: pd.Series, window_days: int) -> pd.Series:
    """
    【新增辅助函数】创建一个随时间线性衰减的影响力分数序列。
    - 功能: 从一个事件(True/False)序列，生成一个0-1的浮点数序列。
    - 逻辑: 事件发生当天影响力为1.0，随后在window_days内线性衰减至0。
            如果窗口期内发生新事件，影响力会重置为1.0并重新开始衰减。
    - 返回: 一个代表“影响力分数”的pandas Series。
    """
    influence = pd.Series(0.0, index=event_series.index)
    event_indices = event_series[event_series].index
    
    for event_idx in event_indices:
        start_pos = event_series.index.get_loc(event_idx)
        # 确保窗口不会超出数据范围
        end_pos = min(start_pos + window_days, len(event_series))
        
        for i in range(start_pos, end_pos):
            days_passed = i - start_pos
            decay_factor = (window_days - days_passed) / window_days
            # 只有当新的影响力大于旧的，才更新（处理窗口重叠问题）
            current_influence_idx = event_series.index[i]
            if decay_factor > influence.at[current_influence_idx]:
                influence.at[current_influence_idx] = decay_factor
                
    return influence

class CognitiveIntelligence:
    def __init__(self, strategy_instance):
        """
        初始化顶层认知合成模块。
        :param strategy_instance: 策略主实例的引用。
        """
        self.strategy = strategy_instance

    def diagnose_contextual_zones(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V339.0 新增】战场上下文诊断模块
        - 核心职责: 独立地、优先地定义“战场”状态，如高位危险区。
                    这是所有后续战术判断的基础。
        """
        # print("        -> [战场上下文诊断模块 V339.0] 启动...")
        states = {}
        default_series = pd.Series(False, index=df.index)

        # --- 1. 军备检查 ---
        required_cols = ['BIAS_21_D', 'close_D', 'high_D', 'SLOPE_5_EMA_13_D']
        if any(c not in df.columns for c in required_cols):
            print("          -> [警告] 缺少诊断“战场上下文”所需数据，模块跳过。")
            return {}

        # --- 2. 评估“地利”：定义静态的“危险战区”上下文 ---
        # 2.1 乖离维度
        bias_overbought_threshold = df['BIAS_21_D'].rolling(120).quantile(0.95)
        states['CONTEXT_RISK_OVEREXTENDED_BIAS'] = df['BIAS_21_D'] > bias_overbought_threshold
        
        # 2.2 动能维度
        is_at_high_price = df['close_D'] > df['high_D'].rolling(60).max() * 0.85
        is_slope_weakening = df['SLOPE_5_EMA_13_D'] < 0.001
        states['CONTEXT_RISK_MOMENTUM_EXHAUSTION'] = is_at_high_price & is_slope_weakening
        
        # 2.3 融合生成“危险战区”状态
        states['CONTEXT_RISK_HIGH_LEVEL_ZONE'] = states['CONTEXT_RISK_OVEREXTENDED_BIAS'] | states['CONTEXT_RISK_MOMENTUM_EXHAUSTION']
        
        return states

    def diagnose_recent_reversal_context(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】近期反转上下文诊断模块
        - 核心职责: 识别近期（如3天内）是否发生过关键的反转触发事件。
        """
        states = {}
        default_series = pd.Series(False, index=df.index)
        
        # 我们认为“显性反转阳线”是最高质量的反转信号
        is_reversal_trigger = self.strategy.atomic_states.get('TRIGGER_DOMINANT_REVERSAL', default_series)
        
        # 使用滚动窗口检查过去3天内是否出现过该信号
        had_recent_reversal = is_reversal_trigger.rolling(window=3, min_periods=1).apply(np.any, raw=True).fillna(0).astype(bool)
        
        states['CONTEXT_RECENT_REVERSAL_SIGNAL'] = had_recent_reversal
        return states

    def diagnose_trend_stage_context(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V332.0 新增】趋势阶段上下文诊断模块
        - 核心职责: 综合多种情报，对当前趋势所处的“阶段”（初期/末期）
                    进行高维度的综合诊断。
        """
        # print("        -> [趋势阶段诊断模块 V332.0] 启动...")
        states = {}
        default_series = pd.Series(False, index=df.index)

        # --- 1. 定义“上涨初期” (Early Stage) ---
        # 条件A: 处于“初升浪”的持续状态中
        is_in_ascent_structure = self.strategy.atomic_states.get('STRUCTURE_POST_ACCUMULATION_ASCENT_C', default_series)
        
        # 条件B: 价格处于年内较低位置 (位置信号)
        yearly_high = df['high_D'].rolling(250).max()
        yearly_low = df['low_D'].rolling(250).min()
        price_range = yearly_high - yearly_low
        # 定义：当前价格低于年内高点和低点的中点
        is_in_lower_half_range = df['close_D'] < (yearly_low + price_range * 0.5)
        
        # 最终裁定：满足任一条件，都可认为是广义的“初期”
        states['CONTEXT_TREND_STAGE_EARLY'] = is_in_ascent_structure | is_in_lower_half_range

        # --- 2. 定义“上涨末期” (Late Stage) ---
        
        # 2.1 获取位置情报 (Context)
        is_in_danger_zone = self.strategy.atomic_states.get('CONTEXT_RISK_HIGH_LEVEL_ZONE', default_series)
        
        # 2.2 获取并提炼行为情报 (Action)
        # 行为1: 主力有派发嫌疑
        is_distributing_action = self.strategy.atomic_states.get('ACTION_RISK_RALLY_WITH_DIVERGENCE', default_series)
        # 行为2: 趋势引擎正在熄火
        is_trend_engine_stalling = self.strategy.atomic_states.get('DYN_TREND_WEAKENING_DECELERATING', default_series)
        
        # ▼▼▼ 将多个行为信号提炼为一个“趋势恶化”的综合行为信号 ▼▼▼
        has_trend_worsening_action = is_distributing_action | is_trend_engine_stalling
        
        # 最终裁定：必须是“位置”和“恶化行为”的共振
        states['CONTEXT_TREND_STAGE_LATE'] = is_in_danger_zone & has_trend_worsening_action
        
        return states

    def diagnose_market_structure_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V277.2 筹码核心版】 - 联合作战司令部
        - 核心重构: 彻底移除了对资金流信号的依赖，全面转向基于筹码结构和行为的判断，
                    使其更适应A股的实际情况。
        """
        # print("        -> [联合作战司令部 V277.2 筹码核心版] 启动，正在打造终极S级战局信号...") # MODIFIED: 修改版本号
        structure_states = {}
        default_series = pd.Series(False, index=df.index)

        # --- 步骤1：情报总览 (全面转向筹码核心信号) ---
        is_ma_bullish = self.strategy.atomic_states.get('MA_STATE_STABLE_BULLISH', default_series)
        is_ma_bearish = self.strategy.atomic_states.get('MA_STATE_STABLE_BEARISH', default_series)
        is_ma_converging = self.strategy.atomic_states.get('MA_STATE_SHORT_CONVERGENCE_SQUEEZE', default_series) | self.strategy.atomic_states.get('MA_STATE_LONG_CONVERGENCE_SQUEEZE', default_series)
        is_price_above_long_ma = self.strategy.atomic_states.get('MA_STATE_PRICE_ABOVE_LONG_MA', default_series)
        is_recent_reversal = self.strategy.atomic_states.get('CONTEXT_RECENT_REVERSAL_SIGNAL', default_series)
        is_ma_short_slope_positive = self.strategy.atomic_states.get('MA_STATE_SHORT_SLOPE_POSITIVE', default_series)
        is_dyn_trend_healthy = self.strategy.atomic_states.get('DYN_TREND_HEALTHY_ACCELERATING', default_series)
        is_dyn_trend_weakening = self.strategy.atomic_states.get('DYN_TREND_WEAKENING_DECELERATING', default_series)
        is_chip_concentrating = self.strategy.atomic_states.get('CHIP_DYN_CONCENTRATING', default_series) # [替代] 资金流入
        is_chip_diverging = self.strategy.atomic_states.get('CHIP_DYN_DIVERGING', default_series) # [替代] 资金流出
        is_chip_health_excellent = df.get('chip_health_score_D', 0) > 80
        is_chip_health_deteriorating = self.strategy.atomic_states.get('CHIP_DYN_HEALTH_DETERIORATING', default_series)
        is_chip_price_divergence = self.strategy.atomic_states.get('RISK_CHIP_PRICE_DIVERGENCE', default_series) # [替代] 资金顶背离
        is_vol_squeeze = self.strategy.atomic_states.get('VOL_STATE_SQUEEZE_WINDOW', default_series)

        # --- 步骤2：联合裁定 (基于更严格、更可靠的筹码信号) ---
        # 【战局1: S级主升浪·黄金航道】 - 结构 + 动能 + 筹码 + 位置 (四重共振)
        structure_states['STRUCTURE_MAIN_UPTREND_WAVE_S'] = (
            is_ma_bullish &                          # 1. 结构: 完美多头排列
            is_dyn_trend_healthy &                   # 2. 动能: 趋势正在健康加速
            is_chip_concentrating &                  # 3. 筹码: 供应正在被锁定 (替代资金流入)
            is_price_above_long_ma                   # 4. 位置: 占据战略制高点
        )
        # 【战局2: A级突破前夜·能量压缩】 - (逻辑不变)
        structure_states['STRUCTURE_BREAKOUT_EVE_A'] = (
            is_vol_squeeze &
            is_chip_concentrating &
            is_ma_converging &
            is_price_above_long_ma
        )
        # 【战局3: B级反转初期·黎明微光】 - (逻辑不变)
        structure_states['STRUCTURE_EARLY_REVERSAL_B'] = (
            is_recent_reversal &
            is_ma_short_slope_positive
        )
        # 【战局4: S级风险·顶部危险】 - 基于更可靠的筹码价格背离
        structure_states['STRUCTURE_TOPPING_DANGER_S'] = (
            is_chip_price_divergence |               # [替代] 资金顶背离
            is_chip_health_deteriorating
        )
        # 【战局5: F级禁区·下跌通道】 - 基于筹码发散
        structure_states['STRUCTURE_BEARISH_CHANNEL_F'] = (
            is_ma_bearish &
            is_dyn_trend_weakening &
            is_chip_diverging                        # [替代] 资金流出
        )

        # print("        -> [联合作战司令部 V277.2 筹码核心版] 核心战局定义升级完成。") # MODIFIED: 修改版本号
        return structure_states

    def run_cognitive_synthesis_engine(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V337.1 筹码核心版】认知综合引擎
        - 核心重构: 重新定义了“派发事件”，使其基于更可靠的筹码和行为信号，
                    彻底摆脱了对资金流数据的依赖。
        """
        # print("        -> [认知综合引擎 V337.1 筹码核心版] 启动，正在合成顶层风险上下文...") # MODIFIED: 修改版本号
        cognitive_states = {}
        default_series = pd.Series(False, index=df.index)

        # --- 认知链 1/2: 识别“突破派发”风险 ---
        # [修改] 将判断依据从资金流转向筹码发散
        is_strong_rally = df['pct_change_D'] > 0.03
        is_chip_diverging = self.strategy.atomic_states.get('CHIP_DYN_DIVERGING', default_series)
        cognitive_states['COGNITIVE_RISK_BREAKOUT_DISTRIBUTION'] = is_strong_rally & is_chip_diverging

        # --- 认知链 2/2: 汇总“近期派发压力”上下文 ---
        # [修改] 重新定义“派发事件”，使用更可靠的筹码和行为信号
        distribution_event = (
            self.strategy.atomic_states.get('RISK_PEAK_BATTLE_DISTRIBUTION_A', default_series) | # 主峰高位派发
            self.strategy.atomic_states.get('RISK_BEHAVIOR_WINNERS_FLEEING_A', default_series) | # 获利盘长期出逃
            self.strategy.atomic_states.get('CHIP_DYN_DIVERGING', default_series)               # 筹码结构发散
        )
        p_dist = get_params_block(self.strategy, 'distribution_context_params', {})
        lookback = get_param_value(p_dist.get('lookback_days'), 10)
        cognitive_states['CONTEXT_RECENT_DISTRIBUTION_PRESSURE'] = distribution_event.rolling(window=lookback, min_periods=1).apply(np.any, raw=True).fillna(0).astype(bool)

        # print("        -> [认知综合引擎 V337.1 筹码核心版] 顶层风险上下文合成完毕。") # MODIFIED: 修改版本号
        return cognitive_states

    def determine_main_force_behavior_sequence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V304.1 信号源修复版】
        - 核心修复: 全面修正了此状态机对上游原子信号的引用，使其与情报层实际生成的信号名完全匹配。
                    解决了因引用不存在的信号而导致整个模块失效的严重问题。
        """
        print("    --- [战略推演单元 V304.1 信号源修复版] 启动，正在生成主力行为序列... ---") # MODIFIED: 修改版本号
        df['main_force_state'] = MainForceState.IDLE.value
        for i in range(1, len(df)):
            prev_state_val = df.at[df.index[i-1], 'main_force_state']
            prev_state = MainForceState(prev_state_val)
            s = {
                # 筹码集中状态
                'is_concentrating': self.strategy.atomic_states.get('CHIP_DYN_CONCENTRATING', pd.Series(False, index=df.index)).iloc[i],
                # 资金流入状态
                'is_inflow': self.strategy.atomic_states.get('CAPITAL_STATE_INFLOW_CONFIRMED', pd.Series(False, index=df.index)).iloc[i],
                # 价格急跌信号
                'is_sharp_drop': self.strategy.atomic_states.get('KLINE_SHARP_DROP', pd.Series(False, index=df.index)).iloc[i],
                # 横盘状态
                'is_sideways': abs(df.get('SLOPE_5_close_D', pd.Series(0, index=df.index)).iloc[i]) < 0.01,
                # 拉升/突破信号 - 使用最高质量的S级突破信号
                'is_markup_breakout': self.strategy.atomic_states.get('OPP_CHIP_LOCKED_BREAKOUT_S', pd.Series(False, index=df.index)).iloc[i],
                # 筹码断层信号 - 使用正确的列名
                'is_chip_fault': df.get('is_chip_fault_formed_D', pd.Series(False, index=df.index)).iloc[i],
                # 派发信号 - 使用最明确的主峰派发风险信号
                'is_distributing': self.strategy.atomic_states.get('RISK_PEAK_BATTLE_DISTRIBUTION_A', pd.Series(False, index=df.index)).iloc[i],
                # 筹码发散信号 - 使用正确的信号名
                'is_diverging': self.strategy.atomic_states.get('CHIP_DYN_DIVERGING', pd.Series(False, index=df.index)).iloc[i],
                # 滞涨信号
                'is_stagnation': self.strategy.atomic_states.get('RISK_VPA_STAGNATION', pd.Series(False, index=df.index)).iloc[i],
                # 跌破长期均线
                'is_below_long_ma': df.at[df.index[i], 'close_D'] < df.at[df.index[i], 'EMA_55_D'],
            }

            current_state = prev_state
            if prev_state == MainForceState.IDLE:
                if s['is_concentrating'] and s['is_inflow']: current_state = MainForceState.ACCUMULATING
            elif prev_state == MainForceState.ACCUMULATING:
                if s['is_markup_breakout'] or s['is_chip_fault']: current_state = MainForceState.MARKUP
                elif s['is_sharp_drop'] and s['is_concentrating']: current_state = MainForceState.WASHING
                elif s['is_distributing'] or s['is_diverging']: current_state = MainForceState.DISTRIBUTING
            elif prev_state == MainForceState.WASHING:
                if s['is_markup_breakout'] or s['is_chip_fault']: current_state = MainForceState.MARKUP
                elif not s['is_concentrating']: current_state = MainForceState.DISTRIBUTING
                elif s['is_sideways'] and s['is_concentrating']: current_state = MainForceState.ACCUMULATING
            elif prev_state == MainForceState.MARKUP:
                if s['is_distributing'] or s['is_diverging'] or s['is_stagnation']: current_state = MainForceState.DISTRIBUTING
                elif s['is_sharp_drop'] and s['is_concentrating']: current_state = MainForceState.WASHING
            elif prev_state == MainForceState.DISTRIBUTING:
                if s['is_below_long_ma']: current_state = MainForceState.COLLAPSE
            elif prev_state == MainForceState.COLLAPSE:
                if s['is_sideways'] and not s['is_below_long_ma']: current_state = MainForceState.IDLE
            df.at[df.index[i], 'main_force_state'] = current_state.value
        # print("    --- [战略推演单元 V304.1 信号源修复版] 主力行为序列已生成。 ---") # MODIFIED: 修改版本号
        return df

    def synthesize_topping_behaviors(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V331.1 职责分离版】顶部行为合成模块
        - 核心重构: 职责被简化为“行为合成”。它消费已有的“战场上下文”和
                    “筹码动态”情报，将其融合成顶层的战术信号。
        """
        # print("        -> [顶部行为合成模块 V331.1] 启动...")
        states = {}
        default_series = pd.Series(False, index=df.index)

        # --- 1. 军备检查 ---
        required_states = ['CHIP_DYN_DIVERGING', 'CHIP_DYN_CONCENTRATING', 'CONTEXT_RISK_HIGH_LEVEL_ZONE']
        if any(s not in self.strategy.atomic_states for s in required_states):
            print("          -> [警告] 缺少合成“顶部行为”所需情报，模块跳过。")
            return {}

        # --- 2. 评估“天时”：识别当天的危险拉升行为 ---
        is_rallying = df['pct_change_D'] > 0.02
        
        # 2.1 拉升出货 (核心风险行为)
        is_diverging = self.strategy.atomic_states.get('CHIP_DYN_DIVERGING', default_series)
        states['ACTION_RISK_RALLY_WITH_DIVERGENCE'] = is_rallying & is_diverging
        
        # 2.2 天量滞涨
        is_huge_volume = df['volume_D'] > df['VOL_MA_21_D'] * 2.5
        is_stagnant = df['pct_change_D'] < 0.01
        states['ACTION_RISK_RALLY_STAGNATION'] = is_huge_volume & is_stagnant

        # --- 3. 【S+级情报融合】：在危险战区确认派发行为 ---
        is_in_danger_zone = self.strategy.atomic_states.get('CONTEXT_RISK_HIGH_LEVEL_ZONE', default_series)
        is_distributing_action = states.get('ACTION_RISK_RALLY_WITH_DIVERGENCE', default_series)
        states['RISK_S_PLUS_CONFIRMED_DISTRIBUTION'] = is_in_danger_zone & is_distributing_action
        
        # --- 4. 重新定义“健康锁筹拉升” (增加保险丝) ---
        is_concentrating = self.strategy.atomic_states.get('CHIP_DYN_CONCENTRATING', default_series)
        states['RALLY_STATE_HEALTHY_LOCKED'] = is_rallying & is_concentrating & ~is_in_danger_zone
        
        return states

    def _diagnose_lock_chip_reconcentration_tactic(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.1 逻辑重塑版】锁仓再集中S+战法诊断模块
        - 核心重构1: 彻底修正了“准备状态”的定义。废除了原先“稳定且聚集”的逻辑矛盾，
                      简化并强化为“筹码已确认锁定稳定(CHIP_CONC_LOCKED_AND_STABLE_A)”，
                      这才是真正意义上的“准备就绪”。
        - 核心重构2: 修复了“点火事件”中对成本加速信号的错误引用，并增加了更高维度的
                      分形突破确认信号，使点火判断更灵敏、更可靠。
        """
        print("        -> [S+战法诊断] 正在扫描“锁仓再集中(V1.1 逻辑重塑版)”...")
        states = {}
        atomic = self.strategy.atomic_states
        triggers = self.strategy.trigger_events
        default_series = pd.Series(False, index=df.index)

        # --- 1. 定义“准备状态” (Setup State) ---
        # [修改原因] 简化并强化“准备状态”的定义。当筹码确认“锁定且稳定”时，
        # 本身就意味着准备工作已完成，可以随时等待点火。原逻辑过于严苛且存在矛盾。
        setup_state = atomic.get('CHIP_CONC_LOCKED_AND_STABLE_A', default_series)

        # --- 2. 定义“点火事件” (Ignition Trigger) ---
        # [修改原因] 修复情报名称不匹配的致命错误，并增加更强的触发器。
        ignition_trigger = (
            triggers.get('TRIGGER_CHIP_IGNITION', default_series) |
            triggers.get('TRIGGER_ENERGY_RELEASE', default_series) |
            atomic.get('MECHANICS_COST_ACCELERATING', default_series) | # 修正了信号名称
            triggers.get('FRACTAL_OPP_SQUEEZE_BREAKOUT_CONFIRMED', default_series) # 新增分形突破确认
        )

        # --- 3. 最终裁定：昨日“准备就绪”，今日“点火” ---
        was_setup_yesterday = setup_state.shift(1).fillna(False)
        is_triggered_today = ignition_trigger
        
        final_tactic_signal = was_setup_yesterday & is_triggered_today
        states['TACTIC_LOCK_CHIP_RECONCENTRATION_S_PLUS'] = final_tactic_signal
        
        if final_tactic_signal.any():
            print(f"          -> [S+级战法确认] 侦测到 {final_tactic_signal.sum()} 次“锁仓再集中”的最终拉升信号！")

        return states

    def _diagnose_lock_chip_rally_tactic(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 状态机重构版】锁筹拉升S级战法诊断模块
        - 核心重构: 废除每日独立判断的苛刻逻辑，采用“点火-巡航-熄火”的状态机模型。
                    1. 点火: 由“锁仓再集中(S+)”信号触发，进入巡航状态。
                    2. 巡航: 状态会持续，只要未出现明确的风险信号。
                    3. 熄火: 当筹码发散或进入上涨末期时，巡航状态终止。
        - 收益: 极大提升了对健康拉升阶段的捕捉和持续跟踪能力，更符合实战。
        """
        print("        -> [S级战法诊断] 正在扫描“锁筹拉升(V2.0 状态机版)”...")
        states = {}
        atomic = self.strategy.atomic_states
        default_series = pd.Series(False, index=df.index)

        # --- 1. 定义“点火”事件 ---
        # [新逻辑] 我们使用更高维度的 S+ 战法作为拉升的起始信号。
        ignition_event = atomic.get('TACTIC_LOCK_CHIP_RECONCENTRATION_S_PLUS', default_series)

        # --- 2. 定义“熄火”条件 (风险出现) ---
        # [新逻辑] 任何一个关键风险信号的出现，都将终止巡航状态。
        is_diverging = atomic.get('CHIP_DYN_DIVERGING', default_series)
        is_late_stage = atomic.get('CONTEXT_TREND_STAGE_LATE', default_series)
        is_ma_broken = ~atomic.get('MA_STATE_STABLE_BULLISH', default_series)
        termination_condition = is_diverging | is_late_stage | is_ma_broken

        # --- 3. 构建状态机 ---
        # [新逻辑] 使用循环来模拟状态的持续和转变。
        is_in_rally_state = pd.Series(False, index=df.index)
        for i in range(1, len(df)):
            # 如果昨天处于巡航状态，且今天没有熄火信号，则继续巡航
            if is_in_rally_state.iloc[i-1] and not termination_condition.iloc[i]:
                is_in_rally_state.iloc[i] = True
            # 如果今天有点火信号，则开启巡航状态
            elif ignition_event.iloc[i]:
                is_in_rally_state.iloc[i] = True
        
        # 最终的战法信号，是处于巡航状态且当日未出现熄火信号的日子
        final_tactic_signal = is_in_rally_state & ~termination_condition
        states['TACTIC_LOCK_CHIP_RALLY_S'] = final_tactic_signal
        
        if final_tactic_signal.any():
            print(f"          -> [S级持仓确认] 侦测到 {final_tactic_signal.sum()} 天处于“健康锁筹拉升”巡航状态！")

        return states

    def _diagnose_pullback_tactics_matrix(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V5.0 衰减影响力重构版】回踩战术诊断模块
        - 核心重构: 引入“衰减影响力”模型，根据回踩距离启动事件的时间远近，
                      将战法动态分级为 S+/A/B 三个层次。
        - 新逻辑:   不再是简单的“在窗口内”，而是“在窗口的哪个位置”。
                      回踩离启动事件越近，战法等级越高，分数也越高。
        - 收益:     策略的智能化和精细度达到全新高度，能更好地区分机会的“含金量”。
        """
        print("        -> [回踩战术矩阵 V5.0] 启动，正在进行分层诊断...")
        states = {}
        atomic = self.strategy.atomic_states
        default_series = pd.Series(False, index=df.index)
        
        lookback_window = 15

        # --- 1. 提取基础条件和事件 ---
        is_healthy_pullback = atomic.get('PULLBACK_STATE_HEALTHY_S', default_series)
        ascent_start_event = atomic.get('POST_ACCUMULATION_ASCENT_C', default_series)
        cruise_start_event = atomic.get('TACTIC_LOCK_CHIP_RECONCENTRATION_S_PLUS', default_series)
        
        # --- 2. 计算衰减影响力分数 ---
        # [新逻辑] 调用新的辅助函数，生成0-1的影响力分数序列
        ascent_influence = _create_decaying_influence_series(ascent_start_event, lookback_window)
        cruise_influence = _create_decaying_influence_series(cruise_start_event, lookback_window)

        # --- 3. 按影响力分层，定义互斥的战法等级 ---
        # 定义影响力阈值
        high_influence_threshold = 0.7
        mid_influence_threshold = 0.3

        # 初升浪回踩分层
        is_ascent_pullback = is_healthy_pullback & (ascent_influence > 0) & (cruise_influence == 0)
        states['TACTIC_ASCENT_PULLBACK_S_PLUS'] = is_ascent_pullback & (ascent_influence > high_influence_threshold)
        states['TACTIC_ASCENT_PULLBACK_A'] = is_ascent_pullback & (ascent_influence <= high_influence_threshold) & (ascent_influence > mid_influence_threshold)
        states['TACTIC_ASCENT_PULLBACK_B'] = is_ascent_pullback & (ascent_influence <= mid_influence_threshold)

        # 巡航中继回踩分层
        is_cruise_pullback = is_healthy_pullback & (cruise_influence > 0)
        states['TACTIC_CRUISE_RELAY_S_PLUS'] = is_cruise_pullback & (cruise_influence > high_influence_threshold)
        states['TACTIC_CRUISE_RELAY_A'] = is_cruise_pullback & (cruise_influence <= high_influence_threshold) & (cruise_influence > mid_influence_threshold)
        states['TACTIC_CRUISE_RELAY_B'] = is_cruise_pullback & (cruise_influence <= mid_influence_threshold)

        # 打印日志
        for name, series in states.items():
            if series.any():
                print(f"          -> [{name.split('_')[-1]}级战法] 侦测到 {series.sum()} 次“{name}”机会！")

        return states










