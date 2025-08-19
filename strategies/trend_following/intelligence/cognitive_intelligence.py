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

    def diagnose_trend_stage_score(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V400.0 风险仪表盘版】趋势阶段评分模块
        - 核心升级: 将“上涨末期”的判断从布尔型升级为0-100的量化分数。
        - 评分逻辑: 构成上涨末期的四个核心风险子条件，每个贡献25分。
        - 收益: 实现了对趋势末期风险的精细化度量，为下游战术模块提供了更灵活的决策依据。
        """
        # print("        -> [趋势阶段评分模块 V400.0 风险仪表盘版] 启动...")
        states = {}
        default_series = pd.Series(False, index=df.index)

        # --- 1. 定义“上涨初期” (Early Stage) - 逻辑保持不变 ---
        is_in_ascent_structure = self.strategy.atomic_states.get('STRUCTURE_POST_ACCUMULATION_ASCENT_C', default_series)
        yearly_high = df['high_D'].rolling(250, min_periods=1).max()
        yearly_low = df['low_D'].rolling(250, min_periods=1).min()
        price_range = (yearly_high - yearly_low).replace(0, np.nan)
        is_in_lower_half_range = df['close_D'] < (yearly_low + price_range * 0.5)
        states['CONTEXT_TREND_STAGE_EARLY'] = is_in_ascent_structure | is_in_lower_half_range

        # --- 2. 计算“上涨末期”的量化分数 (Late Stage Score) ---
        late_stage_score = pd.Series(0, index=df.index, dtype=int)

        # 维度 1 & 2: 位置情报
        is_overextended_bias = self.strategy.atomic_states.get('CONTEXT_RISK_OVEREXTENDED_BIAS', default_series)
        is_momentum_exhaustion = self.strategy.atomic_states.get('CONTEXT_RISK_MOMENTUM_EXHAUSTION', default_series)
        
        # 维度 3 & 4: 行为情报
        is_distributing_action = self.strategy.atomic_states.get('ACTION_RISK_RALLY_WITH_DIVERGENCE', default_series)
        is_trend_engine_stalling = self.strategy.atomic_states.get('DYN_TREND_WEAKENING_DECELERATING', default_series)
        
        #  引入全新的维度 5 和 维度 6。
        # 维度 5: 成交量剖析 (VPA)
        is_vpa_stagnation = self.strategy.atomic_states.get('RISK_VPA_STAGNATION', default_series)
        is_volume_accelerating = self.strategy.atomic_states.get('RISK_VPA_VOLUME_ACCELERATING', default_series)
        has_vpa_risk = is_vpa_stagnation | is_volume_accelerating # 满足任一成交量风险即可

        # 维度 6: 波动率扩张
        is_vol_expanding = self.strategy.atomic_states.get('VOL_STATE_EXPANDING_SHARPLY', default_series)

        # 2.3 根据子条件累加分数
        late_stage_score += is_overextended_bias.astype(int) * 25
        late_stage_score += is_momentum_exhaustion.astype(int) * 25
        late_stage_score += is_distributing_action.astype(int) * 25
        late_stage_score += is_trend_engine_stalling.astype(int) * 25
        late_stage_score += has_vpa_risk.astype(int) * 25
        late_stage_score += is_vol_expanding.astype(int) * 25
        
        states['CONTEXT_TREND_LATE_STAGE_SCORE'] = late_stage_score
        
        # 升级布尔信号的阈值，保持相对严格性 (3 out of 6)。
        states['CONTEXT_TREND_STAGE_LATE'] = late_stage_score >= 75 

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
        
        risk_1_chip_failure = self.strategy.atomic_states.get('RISK_CHIP_STRUCTURE_CRITICAL_FAILURE', default_series)
        risk_2_late_stage = self.strategy.atomic_states.get('CONTEXT_TREND_STAGE_LATE', default_series)
        risk_3_confirmed_dist = self.strategy.atomic_states.get('RISK_S_PLUS_CONFIRMED_DISTRIBUTION', default_series)
        risk_4_deceptive_churn = self.strategy.atomic_states.get('COGNITIVE_RISK_DYNAMIC_DECEPTIVE_CHURN', default_series)

        # --- 步骤2：联合裁定 (基于更严格、更可靠的筹码信号) ---
        # 【战局1: S级主升浪·黄金航道】 - 结构 + 动能 + 筹码 + 位置 (四重共振)
        structure_states['STRUCTURE_MAIN_UPTREND_WAVE_S'] = (
            is_ma_bullish &                          # 1. 结构: 完美多头排列
            is_dyn_trend_healthy &                   # 2. 动能: 趋势正在健康加速
            is_chip_concentrating &                  # 3. 筹码: 供应正在被锁定 (替代资金流入)
            is_price_above_long_ma                   # 4. 位置: 占据战略制高点
        )
        # 【战局2: A级突破前夜·能量压缩】 - 
        structure_states['STRUCTURE_BREAKOUT_EVE_A'] = (
            is_vol_squeeze &
            is_chip_concentrating &
            is_ma_converging &
            is_price_above_long_ma
        )
        # 【战局3: B级反转初期·黎明微光】 - 
        structure_states['STRUCTURE_EARLY_REVERSAL_B'] = (
            is_recent_reversal &
            is_ma_short_slope_positive
        )
        # 【战局4: S级风险·顶部危险】 - 升级为多个高风险信号的融合体
        # 最终裁定：只要任一顶层风险信号出现，就认为战局进入“顶部危险”状态
        structure_states['STRUCTURE_TOPPING_DANGER_S'] = (
            risk_1_chip_failure | risk_2_late_stage | risk_3_confirmed_dist | risk_4_deceptive_churn
        )
        # 【战局5: F级禁区·下跌通道】 - 基于筹码发散
        structure_states['STRUCTURE_BEARISH_CHANNEL_F'] = (
            is_ma_bearish &
            is_dyn_trend_weakening &
            is_chip_diverging
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
        # print("    --- [战略推演单元 V304.1 信号源修复版] 启动，正在生成主力行为序列... ---") # MODIFIED: 修改版本号
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
        【V2.0 王牌重铸版】锁仓再集中S+战法诊断模块
        - 核心重构: 将战法的“准备状态”从有缺陷的A级信号，升级为经过战场环境过滤的
                      S级“筹码结构黄金机会”信号。
        - 收益: 确保S+级战法只在最安全、最有利的战局下发动，从根本上解决了其胜率低下的问题。
        """
        print("        -> [S+战法诊断] 正在扫描“锁仓再集中(V2.0 王牌重铸版)”...") # MODIFIED: 修改版本号
        states = {}
        atomic = self.strategy.atomic_states
        triggers = self.strategy.trigger_events
        default_series = pd.Series(False, index=df.index)
        # --- 1. 定义“准备状态” (Setup State) ---
        setup_state = atomic.get('CHIP_STRUCTURE_PRIME_OPPORTUNITY_S', default_series)
        # --- 2. 定义“点火事件” (Ignition Trigger) ---
        ignition_trigger = (
            triggers.get('TRIGGER_CHIP_IGNITION', default_series) |
            triggers.get('TRIGGER_ENERGY_RELEASE', default_series) |
            atomic.get('MECHANICS_COST_ACCELERATING', default_series) |
            triggers.get('FRACTAL_OPP_SQUEEZE_BREAKOUT_CONFIRMED', default_series)
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
        【V2.2 容错巡航版】锁筹拉升S级战法诊断模块
        - 核心升级: 引入“容错机制”。允许“筹码持续集中”的巡航条件出现一次短暂中断，
                    如果连续两天中断，才终止巡航。
        - 收益: 策略更具韧性，能更好地容忍主升浪中的正常波动，防止被轻易洗出。
        """
        # print("        -> [S级战法诊断] 正在扫描“锁筹拉升(V2.2 容错巡航版)”...")
        states = {}
        atomic = self.strategy.atomic_states
        default_series = pd.Series(False, index=df.index)

        # --- 1. 获取参数  ---
        p = get_params_block(self.strategy, 'lock_chip_rally_params', {})
        require_concentration = get_param_value(p.get('require_continuous_concentration'), True)
        terminate_on_stalling = get_param_value(p.get('terminate_on_health_stalling'), True)

        # --- 2. 定义“点火”事件  ---
        ignition_event = atomic.get('TACTIC_LOCK_CHIP_RECONCENTRATION_S_PLUS', default_series)

        # --- 3. 定义“硬性熄火”条件  ---
        is_diverging = atomic.get('CHIP_DYN_OBJECTIVE_DIVERGING', default_series) # 使用客观发散信号
        is_late_stage = atomic.get('CONTEXT_TREND_STAGE_LATE', default_series)
        is_ma_broken = ~atomic.get('MA_STATE_STABLE_BULLISH', default_series)
        is_health_stalling = atomic.get('HOLD_RISK_HEALTH_STALLING', default_series)
        hard_termination_condition = is_diverging | is_late_stage | is_ma_broken
        if terminate_on_stalling:
            hard_termination_condition |= is_health_stalling

        # --- 4. 定义“软性巡航”条件  ---
        is_cruise_condition_met = atomic.get('CHIP_DYN_CONCENTRATING', default_series) if require_concentration else pd.Series(True, index=df.index)

        # --- 5. 构建带“容错机制”的状态机 ---
        is_in_rally_state = pd.Series(False, index=df.index)
        cruise_warning_active = False # [新逻辑] 引入“健康预警”状态标志
        for i in range(1, len(df)):
            # 检查硬性熄火条件，这是最高优先级
            if hard_termination_condition.iloc[i]:
                is_in_rally_state.iloc[i] = False
                cruise_warning_active = False
                continue

            # 如果昨天处于巡航状态
            if is_in_rally_state.iloc[i-1]:
                # 检查软性巡航条件
                if is_cruise_condition_met.iloc[i]:
                    # 条件满足，继续健康巡航，并解除预警
                    is_in_rally_state.iloc[i] = True
                    cruise_warning_active = False
                else:
                    # 条件不满足，检查是否已在预警状态
                    if cruise_warning_active:
                        # 已经预警过一次，这是连续第二次失败，终止巡航
                        is_in_rally_state.iloc[i] = False
                        cruise_warning_active = False
                    else:
                        # 这是第一次失败，进入预警状态，但巡航继续
                        is_in_rally_state.iloc[i] = True
                        cruise_warning_active = True
            # 如果今天有点火信号，则开启巡航
            elif ignition_event.iloc[i]:
                is_in_rally_state.iloc[i] = True
                cruise_warning_active = False # 新的巡航开始时，总是健康的
        
        final_tactic_signal = is_in_rally_state & ~hard_termination_condition
        states['TACTIC_LOCK_CHIP_RALLY_S'] = final_tactic_signal
        
        if final_tactic_signal.any():
            print(f"          -> [S级持仓确认] 侦测到 {final_tactic_signal.sum()} 天处于“健康锁筹拉升”巡航状态！")

        return states

    # 将多个低效的原子动能信号，融合成一个经过战略过滤的高质量信号。
    def synthesize_dynamic_offense_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】协同进攻动能合成模块
        - 核心职责: 1. 将多个独立的原子动能信号融合成一个“协同进攻”信号。
                      2. 使用“趋势阶段”上下文对该信号进行战略过滤，防止在上涨末期追高。
        - 收益: 废除了多个胜率平平的动能信号，创造了一个更高质量、更安全的A级动能信号。
        """
        # print("        -> [协同进攻动能合成模块 V1.0] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_series = pd.Series(False, index=df.index)

        # 1. 获取所有原始的原子动能信号
        is_trend_accel = atomic.get('DYN_TREND_HEALTHY_ACCELERATING', default_series)
        is_cost_accel_chip = atomic.get('CHIP_DYN_COST_ACCELERATING', default_series)
        is_cost_accel_mech = atomic.get('MECHANICS_COST_ACCELERATING', default_series)
        is_conc_accel = atomic.get('CHIP_DYN_S_ACCEL_CONCENTRATING', default_series)
        is_inertia_decreasing = atomic.get('MECHANICS_INERTIA_DECREASING', default_series)

        # 2. 计算当天有多少个动能信号被触发 (协同原则)
        # 将布尔序列转换为整数 (True=1, False=0) 并按行求和
        num_active_signals = (
            is_trend_accel.astype(int) +
            is_cost_accel_chip.astype(int) +
            is_cost_accel_mech.astype(int) +
            is_conc_accel.astype(int) +
            is_inertia_decreasing.astype(int)
        )
        
        # 定义协同进攻的原始触发条件：至少2个动能信号同时激活
        is_synergistic_offense = (num_active_signals >= 2)

        # 3. 获取战略过滤器：是否处于上涨末期
        late_stage_score = self.strategy.atomic_states.get('CONTEXT_TREND_LATE_STAGE_SCORE', pd.Series(0, index=df.index))
        p_trend_stage = get_params_block(self.strategy, 'trend_stage_params', {})
        max_score_for_offense = get_param_value(p_trend_stage.get('dynamic_offense_max_late_stage_score'), 30)
        is_in_safe_stage = late_stage_score < max_score_for_offense

        # 4. 最终裁定：发动协同进攻，且【处于安全阶段】
        final_signal = is_synergistic_offense & is_in_safe_stage
        states['DYN_AGGRESSIVE_OFFENSE_A'] = final_signal
        
        if final_signal.any():
            print(f"          -> [A级动能确认] 侦测到 {final_signal.sum()} 次安全的“侵略性协同进攻”！")

        return states

    def synthesize_prime_tactic(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 王牌突击队】终极战法合成模块
        - 核心思想: 严格分离“阵地构筑(Setup)”与“冲锋号(Trigger)”。
        - 流程: 1. 定义S级的“黄金阵地”状态。
                 2. 定义S级的“突破冲锋号”事件。
                 3. 只有当“黄金阵地”建立后，次日出现“冲锋号”，才确认为终极战法。
        - 收益: 彻底根除基于静态“状态”的左侧交易思维，转为高确定性的右侧突破交易。
        """
        # print("        -> [终极战法合成模块 V1.0 王牌突击队] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        triggers = self.strategy.trigger_events
        default_series = pd.Series(False, index=df.index)

        # --- 1. 定义S级“黄金阵地” (Prime Setup) ---
        # 这是一个多维度共振的“压缩弹簧”状态
        is_prime_chip_structure = atomic.get('CHIP_STRUCTURE_PRIME_OPPORTUNITY_S', default_series)
        is_extreme_squeeze = atomic.get('VOL_STATE_EXTREME_SQUEEZE', default_series)
        has_energy_advantage = atomic.get('MECHANICS_ENERGY_ADVANTAGE', default_series)
        
        # 最终的“黄金阵地”状态：必须同时满足筹码、波动率和力学优势
        setup_prime_structure_s = is_prime_chip_structure & is_extreme_squeeze & has_energy_advantage
        states['SETUP_PRIME_STRUCTURE_S'] = setup_prime_structure_s

        # --- 2. 定义S级“突破冲锋号” (Prime Trigger) ---
        # 这是一个融合了多种确认方式的、高强度的突破事件
        is_energy_release = triggers.get('TRIGGER_ENERGY_RELEASE', default_series)
        is_volume_breakout = triggers.get('TRIGGER_VOLUME_SPIKE_BREAKOUT', default_series)
        is_fractal_breakout = triggers.get('FRACTAL_OPP_SQUEEZE_BREAKOUT_CONFIRMED', default_series)

        # 最终的“冲锋号”事件：满足任一高强度突破信号即可
        trigger_prime_breakout_s = is_energy_release | is_volume_breakout | is_fractal_breakout
        states['TRIGGER_PRIME_BREAKOUT_S'] = trigger_prime_breakout_s

        # --- 3. 【终极裁定】生成S++级王牌战法 ---
        # 条件：昨日处于“黄金阵地”状态，今日响起了“冲锋号”
        was_setup_yesterday = setup_prime_structure_s.shift(1).fillna(False)
        is_triggered_today = trigger_prime_breakout_s
        
        final_tactic_signal = was_setup_yesterday & is_triggered_today
        states['TACTIC_PRIME_STRUCTURE_BREAKOUT_S_PLUS_PLUS'] = final_tactic_signal

        if final_tactic_signal.any():
            print(f"          -> [S++级王牌战法确认] 侦测到 {final_tactic_signal.sum()} 次“黄金结构突破”机会！")

        return states

    def _diagnose_pullback_tactics_matrix(self, df: pd.DataFrame, enhancements: Dict) -> Dict[str, pd.Series]:
        """
        【V7.1 防诱多增强版】回踩战术诊断模块
        - 核心升级: 为 S+ 级“巡航回踩确认”战法增加了“非上涨末期”的前置条件。
        - 收益: 解决了该战法在主升浪末期被“诱多型”反转信号欺骗的致命缺陷，
                从根本上提升了信号的安全性。
        """
        # print("        -> [回踩战术矩阵 V7.1 防诱多增强版] 启动...") # MODIFIED: 修改版本号和描述
        states = {}
        atomic = self.strategy.atomic_states
        triggers = self.strategy.trigger_events
        default_series = pd.Series(False, index=df.index)
        
        # --- 1. 提取核心情报 ---
        # 战场环境
        lookback_window = 15
        ascent_start_event = atomic.get('POST_ACCUMULATION_ASCENT_C', default_series)
        cruise_start_event = atomic.get('TACTIC_LOCK_CHIP_RECONCENTRATION_S_PLUS', default_series)
        is_in_ascent_window = ascent_start_event.rolling(window=lookback_window, min_periods=1).max().astype(bool)
        is_in_cruise_window = cruise_start_event.rolling(window=lookback_window, min_periods=1).max().astype(bool)
        # 回踩性质 (昨日)
        was_healthy_pullback = atomic.get('PULLBACK_STATE_HEALTHY_S', default_series).shift(1).fillna(False)
        was_suppressive_pullback = atomic.get('PULLBACK_STATE_SUPPRESSIVE_S', default_series).shift(1).fillna(False)
        # 统一确认信号 (今日)
        is_reversal_confirmed = triggers.get('TRIGGER_DOMINANT_REVERSAL', default_series)
        # 新增：获取“上涨末期”上下文状态
        late_stage_score = self.strategy.atomic_states.get('CONTEXT_TREND_LATE_STAGE_SCORE', pd.Series(0, index=df.index))
        # 从配置中读取此战法能容忍的最高风险分数
        p_trend_stage = get_params_block(self.strategy, 'trend_stage_params', {})
        max_score_for_pullback = get_param_value(p_trend_stage.get('pullback_s_plus_max_late_stage_score'), 50)
        # 定义是否处于安全区域
        is_in_safe_stage = late_stage_score < max_score_for_pullback
        # --- 2. 【新范式】按优先级生成唯一的战术信号 ---
        # 优先级 1 (S+++ 王牌): 巡航期 + 打压回踩(昨日) + 显性反转(今日) -> 经典的“黄金坑”V型反转
        s_triple_plus_signal = is_in_cruise_window & was_suppressive_pullback & is_reversal_confirmed
        states['TACTIC_CRUISE_PIT_REVERSAL_S_TRIPLE_PLUS'] = s_triple_plus_signal

        # 优先级 2 (S+): 巡航期 + 健康回踩(昨日) + 显性反转(今日) + 【非上涨末期】
        s_plus_signal = is_in_cruise_window & was_healthy_pullback & is_reversal_confirmed & is_in_safe_stage & ~s_triple_plus_signal
        states['TACTIC_CRUISE_PULLBACK_REVERSAL_S_PLUS'] = s_plus_signal

        # 优先级 3 (A+): 初升浪期 + 打压回踩(昨日) + 显性反转(今日)
        a_plus_signal = is_in_ascent_window & was_suppressive_pullback & is_reversal_confirmed & ~is_in_cruise_window
        states['TACTIC_ASCENT_PIT_REVERSAL_A_PLUS'] = a_plus_signal

        # 优先级 4 (A): 初升浪期 + 健康回踩(昨日) + 显性反转(今日)
        a_signal = is_in_ascent_window & was_healthy_pullback & is_reversal_confirmed & ~is_in_cruise_window & ~a_plus_signal
        states['TACTIC_ASCENT_PULLBACK_REVERSAL_A'] = a_signal

        # --- 3. 打印日志 (适配新战法名称) ---
        tactic_name_map = {
            "CRUISE_PIT_REVERSAL": "巡航黄金坑V反(王牌)",
            "CRUISE_PULLBACK_REVERSAL": "巡航常规回踩确认",
            "ASCENT_PIT_REVERSAL": "初升浪黄金坑V反",
            "ASCENT_PULLBACK_REVERSAL": "初升浪常规回踩确认"
        }
        grade_map = {
            "S_TRIPLE_PLUS": "S+++", "S_PLUS": "S+", "A_PLUS": "A+", "A": "A"
        }
        for name, series in states.items():
            if series.any():
                matched_grade_key = ""
                for grade_key in sorted(grade_map.keys(), key=len, reverse=True):
                    if name.endswith(f"_{grade_key}"):
                        matched_grade_key = grade_key
                        break
                if matched_grade_key:
                    tactic_key_part = name.replace("TACTIC_", "").replace(f"_{matched_grade_key}", "")
                    cn_tactic = tactic_name_map.get(tactic_key_part, tactic_key_part)
                    cn_grade = grade_map.get(matched_grade_key, "")
                    print(f"          -> [{cn_grade}级战法] 侦测到 {series.sum()} 次“{cn_tactic}”机会！")
                else:
                    print(f"          -> [战法确认] 侦测到 {series.sum()} 次“{name}”机会！")

        return states

    def _create_pullback_decision_log(self, df: pd.DataFrame, enhancements: Dict) -> pd.DataFrame:
        """
        【V1.0 新增】战术决策日志探针
        - 核心职责: 生成一个详细的DataFrame，记录回踩战术矩阵的完整决策过程。
                      用于100%验证逻辑的正确性，确保信号没有“异常跌落”。
        - 输出: 一个包含所有中间判断、潜在战法和最终战法的“决策日志”DataFrame。
        """
        log_data = {}
        atomic = self.strategy.atomic_states
        triggers = self.strategy.trigger_events
        default_series = pd.Series(False, index=df.index)

        # --- 1. 记录所有输入条件 ---
        log_data['is_healthy_pullback'] = atomic.get('PULLBACK_STATE_HEALTHY_S', default_series)
        lookback_window = 15
        ascent_start_event = atomic.get('POST_ACCUMULATION_ASCENT_C', default_series)
        cruise_start_event = atomic.get('TACTIC_LOCK_CHIP_RECONCENTRATION_S_PLUS', default_series)
        log_data['is_in_ascent_window'] = ascent_start_event.rolling(window=lookback_window, min_periods=1).max().astype(bool)
        log_data['is_in_cruise_window'] = cruise_start_event.rolling(window=lookback_window, min_periods=1).max().astype(bool)
        log_data['is_hammer'] = enhancements.get('is_hammer_candle', default_series)
        log_data['is_fib_gold'] = enhancements.get('is_fib_golden_support', default_series)
        log_data['is_suppressive'] = enhancements.get('is_suppressive_pullback', default_series)
        log_data['is_chip_locked'] = atomic.get('CHIP_CONC_LOCKED_AND_STABLE_A', default_series)
        log_data['is_dominant_reversal'] = triggers.get('TRIGGER_DOMINANT_REVERSAL', default_series)

        # --- 2. 记录所有潜在战法 (不考虑互斥) ---
        log_data['POTENTIAL_S+++'] = log_data['is_in_cruise_window'] & log_data['is_suppressive'] & log_data['is_dominant_reversal']
        log_data['POTENTIAL_S++'] = log_data['is_in_cruise_window'] & log_data['is_suppressive'] & log_data['is_hammer']
        log_data['POTENTIAL_S+'] = log_data['is_in_cruise_window'] & log_data['is_healthy_pullback'] & log_data['is_fib_gold'] & log_data['is_hammer']
        log_data['POTENTIAL_S'] = log_data['is_in_cruise_window'] & log_data['is_healthy_pullback']
        log_data['POTENTIAL_A+'] = log_data['is_in_ascent_window'] & log_data['is_healthy_pullback'] & log_data['is_chip_locked']
        log_data['POTENTIAL_A'] = log_data['is_in_ascent_window'] & log_data['is_healthy_pullback'] & log_data['is_hammer']
        log_data['POTENTIAL_B'] = log_data['is_in_ascent_window'] & log_data['is_healthy_pullback']

        # --- 3. 记录最终的互斥决策结果 ---
        final_s_triple_plus = log_data['POTENTIAL_S+++']
        final_s_plus_plus = log_data['POTENTIAL_S++'] & ~final_s_triple_plus
        final_s_plus = log_data['POTENTIAL_S+'] & ~final_s_triple_plus & ~final_s_plus_plus
        final_s = log_data['POTENTIAL_S'] & ~final_s_triple_plus & ~final_s_plus_plus & ~final_s_plus
        
        is_in_cruise_decision = final_s_triple_plus | final_s_plus_plus | final_s_plus | final_s
        
        final_a_plus = log_data['POTENTIAL_A+'] & ~is_in_cruise_decision
        final_a = log_data['POTENTIAL_A'] & ~log_data['is_chip_locked'] & ~is_in_cruise_decision
        final_b = log_data['POTENTIAL_B'] & ~log_data['is_chip_locked'] & ~log_data['is_hammer'] & ~is_in_cruise_decision

        log_data['FINAL_S+++'] = final_s_triple_plus
        log_data['FINAL_S++'] = final_s_plus_plus
        log_data['FINAL_S+'] = final_s_plus
        log_data['FINAL_S'] = final_s
        log_data['FINAL_A+'] = final_a_plus
        log_data['FINAL_A'] = final_a
        log_data['FINAL_B'] = final_b
        
        return pd.DataFrame(log_data)









