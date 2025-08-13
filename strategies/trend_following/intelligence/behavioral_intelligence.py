# 文件: strategies/trend_following/intelligence/behavioral_intelligence.py
# 行为与模式识别模块
import pandas as pd
import numpy as np
from typing import Dict
from strategies.trend_following.utils import get_params_block, get_param_value, create_persistent_state

class BehavioralIntelligence:
    def __init__(self, strategy_instance):
        """
        初始化行为与模式识别模块。
        :param strategy_instance: 策略主实例的引用。
        """
        self.strategy = strategy_instance
        # K线形态识别器可能需要在这里初始化或传入
        self.pattern_recognizer = strategy_instance.pattern_recognizer

    def diagnose_kline_patterns(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V273.0 装备换代版】
        - 核心修复: 更新了对 `_create_persistent_state` 方法的调用方式。
        - 新协议: 使用了最新的参数名 `entry_event_series` 和 `break_condition_series`，
                  使其完全兼容我们新建的 V271.0 “状态机引擎”。
        - 收益: 确保了基础侦察部队能够正确使用现代化的通用工具，实现了全军装备的同步。
        """
        states = {}
        p = get_params_block(self.strategy, 'kline_pattern_params')
        if not get_param_value(p.get('enabled'), False): return states
        
        default_series = pd.Series(False, index=df.index)

        # --- 1. “巨阴洗盘”机会窗口 (Washout Opportunity Window) ---
        p_washout = p.get('washout_params', {})
        if get_param_value(p_washout.get('enabled'), True):
            washout_threshold = get_param_value(p_washout.get('washout_threshold'), -0.07)
            volume_ratio = get_param_value(p_washout.get('washout_volume_ratio'), 1.5)
            vol_ma_col = 'VOL_MA_21_D'
            if 'pct_change_D' in df.columns and vol_ma_col in df.columns:
                is_deep_drop = df['pct_change_D'] < washout_threshold
                is_high_volume = df['volume_D'] > df[vol_ma_col] * volume_ratio
                washout_event = is_deep_drop & is_high_volume
                # 在事件发生后的3天内，都标记为机会窗口
                states['KLINE_STATE_WASHOUT_WINDOW'] = washout_event.rolling(window=3, min_periods=1).max().astype(bool)

        # --- 2. “缺口支撑”持续状态 (Gap Support Active State) ---
        p_gap = p.get('gap_support_params', {})
        if get_param_value(p_gap.get('enabled'), True):
            persistence_days = get_param_value(p_gap.get('persistence_days'), 10)
            
            # 定义“进入事件”：向上跳空缺口
            gap_up_event = df['low_D'] > df['high_D'].shift(1)
            gap_high = df['high_D'].shift(1).where(gap_up_event)
            
            # 定义“打破条件”：价格回补了缺口
            price_fills_gap = df['close_D'] < gap_high.ffill()

            states['KLINE_STATE_GAP_SUPPORT_ACTIVE'] = create_persistent_state(
                df=df,
                entry_event_series=gap_up_event,         # 使用新参数名: entry_event_series
                persistence_days=persistence_days,
                break_condition_series=price_fills_gap,  # 使用新参数名: break_condition_series
                state_name='KLINE_STATE_GAP_SUPPORT_ACTIVE'
            )

        # --- 3. “N字板”盘整状态 (N-Shape Consolidation State) ---
        p_nshape = p.get('n_shape_params', {})
        if get_param_value(p_nshape.get('enabled'), True):
            rally_threshold = get_param_value(p_nshape.get('rally_threshold'), 0.097)
            consolidation_days_max = get_param_value(p_nshape.get('consolidation_days_max'), 3)
            
            is_strong_rally = df['pct_change_D'] >= rally_threshold
            consolidation_window = is_strong_rally.shift(1).rolling(window=consolidation_days_max, min_periods=1).max().astype(bool)
            is_not_rally_today = df['pct_change_D'] < rally_threshold
            states['KLINE_STATE_N_SHAPE_CONSOLIDATION'] = consolidation_window & is_not_rally_today

        p_atomic = p.get('atomic_behavior_params', {})
        if get_param_value(p_atomic.get('enabled'), True):
            vol_ma_col = 'VOL_MA_21_D'
            if 'pct_change_D' in df.columns and vol_ma_col in df.columns:
                
                # 定义“恐慌性大跌”
                sharp_drop_threshold = get_param_value(p_atomic.get('sharp_drop_threshold'), -0.04)
                states['KLINE_SHARP_DROP'] = df['pct_change_D'] < sharp_drop_threshold
                
                # 定义“显著放量”
                high_volume_ratio = get_param_value(p_atomic.get('high_volume_ratio'), 1.5)
                states['KLINE_HIGH_VOLUME'] = df['volume_D'] > df[vol_ma_col] * high_volume_ratio

        return states


    def diagnose_board_patterns(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V58.0 诊断模块 - 板形态诊断引擎】
        """
        # print("        -> [诊断模块] 正在执行板形态诊断...")
        states = {}
        p = get_params_block(self.strategy, 'board_pattern_params')
        if not get_param_value(p.get('enabled'), False):
            return states
        prev_close = df['close_D'].shift(1)
        limit_up_threshold = get_param_value(p.get('limit_up_threshold'), 0.098)
        limit_down_threshold = get_param_value(p.get('limit_down_threshold'), -0.098)
        price_buffer = get_param_value(p.get('price_buffer'), 0.005)
        limit_up_price = prev_close * (1 + limit_up_threshold)
        limit_down_price = prev_close * (1 + limit_down_threshold)
        is_limit_up_close = df['close_D'] >= limit_up_price * (1 - price_buffer)
        is_limit_up_high = df['high_D'] >= limit_up_price * (1 - price_buffer)
        is_limit_down_low = df['low_D'] <= limit_down_price * (1 + price_buffer)
        states['BOARD_EVENT_EARTH_HEAVEN'] = is_limit_down_low & is_limit_up_close
        
        # signal = states['BOARD_EVENT_EARTH_HEAVEN']
        # dates_str = format_debug_dates(signal)
        # print(f"          -> '地天板' 事件诊断完成，发现 {signal.sum()} 天。{dates_str}")
        
        is_limit_down_close = df['close_D'] <= limit_down_price * (1 + price_buffer)
        states['BOARD_EVENT_HEAVEN_EARTH'] = is_limit_up_high & is_limit_down_close
        
        # signal = states['BOARD_EVENT_HEAVEN_EARTH']
        # dates_str = format_debug_dates(signal)
        # print(f"          -> '天地板' 事件诊断完成，发现 {signal.sum()} 天。{dates_str}")
        
        return states


    def diagnose_pullback_character(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V500.0 统一回踩诊断中心】
        - 核心重构: 取代了原有的 _diagnose_healthy_pullback 和 _diagnose_suppression_pullback 模块。
        - 核心职责: 1. 识别所有发生在建设性背景下的回踩行为。
                    2. 对回踩的“性质”进行分类（健康的、打压式的等）。
                    3. 输出不同性质的、中立的原子状态，供下游决策。
        """
        # print("        -> [统一回踩诊断中心 V500.0] 启动...")
        states = {}
        default_series = pd.Series(False, index=df.index)
        # 使用统一的参数块
        p = get_params_block(self.strategy, 'pullback_analysis_params')
        if not get_param_value(p.get('enabled'), False):
            return states

        # --- 1. 军备检查 ---
        required_cols = [
            'pct_change_D', 'turnover_from_losers_ratio_D', 'turnover_from_winners_ratio_D',
            'SLOPE_5_concentration_90pct_D', 'close_D', 'volume_D', 'VOL_MA_21_D'
        ]
        if any(c not in df.columns for c in required_cols):
            print("          -> [警告] 缺少诊断“回踩性质”所需列，模块跳过。")
            return states

        # --- 2. 定义通用的“建设性背景” ---
        is_in_uptrend = self.strategy.atomic_states.get('STRUCTURE_MAIN_UPTREND_WAVE_S', default_series)
        is_in_squeeze = self.strategy.atomic_states.get('VOL_STATE_SQUEEZE_WINDOW', default_series)
        is_in_box = self.strategy.atomic_states.get('BOX_STATE_HEALTHY_ACCUMULATION', default_series)
        is_constructive_context = is_in_uptrend | is_in_squeeze | is_in_box

        # --- 3. 识别并定性回踩行为 ---
        # 基础条件：当天是下跌的
        is_pullback_day = df['pct_change_D'] < 0

        # 3.1 定性“健康回踩”的特征
        p_healthy = p.get('healthy_pullback_rules', {})
        is_gentle_drop = df['pct_change_D'] > get_param_value(p_healthy.get('min_pct_change'), -0.05)
        is_shrinking_volume = df['volume_D'] < df['VOL_MA_21_D']
        # [修改原因] V508.0 新增：要求健康回踩时，获利盘必须是惜售的（锁仓）。
        # 这是判断“真洗盘”和“真出货”的核心区别。
        winner_turnover_low_threshold = get_param_value(p_healthy.get('max_winner_turnover_ratio'), 30.0)
        is_winner_holding_tight = df['turnover_from_winners_ratio_D'] < winner_turnover_low_threshold

        is_healthy_character = is_gentle_drop & is_shrinking_volume & is_winner_holding_tight

        # 3.2 定性“打压回踩”的特征
        p_suppression = p.get('suppression_pullback_rules', {})
        is_significant_drop = df['pct_change_D'] < get_param_value(p_suppression.get('min_drop_pct'), -0.03)
        is_panic_selling = df['turnover_from_losers_ratio_D'] > get_param_value(p_suppression.get('min_loser_turnover_ratio'), 40.0)
        is_winner_holding = df['turnover_from_winners_ratio_D'] < get_param_value(p_suppression.get('max_winner_turnover_ratio'), 30.0)
        is_suppressive_character = is_significant_drop & is_panic_selling & is_winner_holding

        # --- 4. 组合生成最终的中立状态信号 ---
        # 筹码稳定是所有有效回踩的共同要求
        divergence_tolerance = get_param_value(p_suppression.get('divergence_tolerance'), 0.0005)
        is_chip_stable = df['SLOPE_5_concentration_90pct_D'] < divergence_tolerance

        # 最终状态1: 健康回踩
        states['PULLBACK_STATE_HEALTHY_S'] = is_pullback_day & is_healthy_character & is_constructive_context
        if states['PULLBACK_STATE_HEALTHY_S'].any():
            print(f"          -> [情报] 侦测到 {states['PULLBACK_STATE_HEALTHY_S'].sum()} 次“S级健康回踩”状态。")

        # 最终状态2: 打压回踩 (需要后续V型反转确认)
        # 注意：打压回踩本身不是买点，它只是一个“事件”，真正的买点在它被确认之后
        is_suppression_event = is_pullback_day & is_suppressive_character & is_constructive_context & is_chip_stable
        min_rebound_days = get_param_value(p_suppression.get('min_rebound_days'), 1)
        max_rebound_days = get_param_value(p_suppression.get('max_rebound_days'), 3)
        is_rebound_confirmed = pd.Series(False, index=df.index)
        for i in range(min_rebound_days, max_rebound_days + 1):
            is_prev_suppression = is_suppression_event.shift(i).fillna(False)
            is_price_recovered = df['close_D'] > df['close_D'].shift(i)
            is_rebound_confirmed |= (is_prev_suppression & is_price_recovered)

        states['PULLBACK_STATE_SUPPRESSIVE_S'] = is_rebound_confirmed
        if states['PULLBACK_STATE_SUPPRESSIVE_S'].any():
            print(f"          -> [情报] 侦测到 {states['PULLBACK_STATE_SUPPRESSIVE_S'].sum()} 次“S级打压回踩被确认”状态。")

        return states


    def diagnose_behavioral_patterns(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V336.1 证据分层版】主力操纵战术“反侦察”模块
        - 核心升级: 放弃对“资金流”数据的盲目信任，建立基于“筹码结构”为核心的
                    分层证据体系，以应对主力的“反侦察”战术。
        """
        # print("        -> [反侦察模块 V336.1] 启动，正在进行证据分层分析...")
        states = {}
        default_series = pd.Series(False, index=df.index)

        # --- 1. 基础行为定义 ---
        is_sharp_drop = df['pct_change_D'] < -0.04
        is_strong_rally = df['pct_change_D'] > 0.03
        is_high_volume = df['volume_D'] > df['VOL_MA_21_D'] * 1.5
        
        # --- 2. 证据链定义 ---
        # 证据A: 资金流 (不可靠，作为次要证据)
        evidence_capital_inflow = self.strategy.atomic_states.get('CAPITAL_STRUCT_MAIN_FORCE_ACCUMULATING', default_series)
        evidence_capital_outflow = self.strategy.atomic_states.get('RISK_CAPITAL_STRUCT_MAIN_FORCE_DISTRIBUTING', default_series)
        
        # 证据B: 筹码结构 (核心证据)
        evidence_chip_concentrating = self.strategy.atomic_states.get('CHIP_DYN_CONCENTRATING', default_series)
        evidence_chip_diverging = self.strategy.atomic_states.get('RISK_DYN_DIVERGING', default_series)

        # --- 3. “打压收割”机会分层诊断 ---
        # 核心矛盾：价格暴跌 VS 筹码集中
        is_core_absorption_conflict = is_sharp_drop & is_high_volume & evidence_chip_concentrating
        
        # A级机会 (隐蔽吸筹): 核心矛盾成立。这是主力最狡猾、最常见的吸筹方式。
        states['BEHAVIOR_STEALTH_ABSORPTION_A'] = is_core_absorption_conflict
        
        # S级机会 (黄金坑): 核心矛盾成立，且资金流出现罕见的同步流入。这是主力图穷匕见、毫不掩饰的贪婪。
        states['BEHAVIOR_GOLDEN_PIT_S'] = is_core_absorption_conflict & evidence_capital_inflow

        # --- 4. “诱多派发”风险分层诊断 ---
        # 核心矛盾：价格拉升 VS 筹码发散
        is_core_distribution_conflict = is_strong_rally & evidence_chip_diverging
        
        # A级风险 (隐蔽派发): 核心矛盾成立。这是最危险的陷阱。
        states['BEHAVIOR_DECEPTIVE_RALLY_A'] = is_core_distribution_conflict
        
        # S级风险 (公然出货): 核心矛盾成立，且资金流同步流出。这是主力肆无忌惮的派发。
        states['BEHAVIOR_DECEPTIVE_RALLY_S'] = is_core_distribution_conflict & evidence_capital_outflow

        return states


    def diagnose_chip_pullback_hammer_opportunity(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【新增战法】锁仓回踩探针机会诊断 (Chip-Locked Pullback Probe)
        - 核心逻辑: 捕捉“筹码持续集中 + 股价回踩关键支撑 + 出现长下影线确认”的黄金坑机会。
        - 适用场景: A股主力在拉升前的经典洗盘+试盘行为。
        """
        print("        -> [战法诊断] 正在扫描“锁仓回踩探针”机会...")
        states = {}
        atomic = self.strategy.atomic_states
        default_series = pd.Series(False, index=df.index)

        # 1. **前提条件：趋势背景**
        # 必须处于一个健康的上升趋势中，我们使用“稳定多头排列”作为基础。
        is_uptrend_context = atomic.get('MA_STATE_STABLE_BULLISH', default_series)

        # 2. **核心条件1：筹码持续集中**
        # 表明主力仍在收集筹码，控盘意图明确。
        is_chips_concentrating = atomic.get('CHIP_DYN_CONCENTRATING', default_series)

        # 3. **核心条件2：回踩关键支撑**
        # 这里我们定义为日内最低价曾跌破21日线，但收盘价又收回其上，是经典的支撑测试。
        support_ma = 'EMA_21_D'
        if support_ma not in df.columns:
            return states
        is_pullback_to_support = (df['low_D'] < df[support_ma]) & (df['close_D'] > df[support_ma])

        # 4. **核心条件3：下影线K线确认 (探针形态)**
        # 计算K线实体和影线长度，识别出长下影线的“锤子”或“探针”形态。
        body = (df['close_D'] - df['open_D']).abs()
        # 防止body为0导致除法错误
        body = body.replace(0, 0.0001)
        lower_shadow = df[['open_D', 'close_D']].min(axis=1) - df['low_D']
        upper_shadow = df['high_D'] - df[['open_D', 'close_D']].max(axis=1)
        # 定义“探针”形态：下影线长度至少是实体的2倍，且上影线很短。
        is_probe_candle = (lower_shadow >= body * 2.0) & (upper_shadow < body * 0.8)
        # 5. **最终裁定**
        # 组合所有条件，生成最终的机会信号。
        final_condition = (
            is_uptrend_context &
            is_chips_concentrating &
            is_pullback_to_support &
            is_probe_candle
        )
        states['OPP_CHIP_PULLBACK_HAMMER_A'] = final_condition
        # if final_condition.any():
        #     print(f"          -> [情报] 侦测到 {final_condition.sum()} 次 A级“锁仓回踩探针”机会！")

        return states


    def diagnose_squeeze_zone_opportunities(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V507.0 生产就绪版】压缩区机会诊断模块
        - 核心逻辑: 基于“逻辑分层”思想，识别在健康结构中发生的、经过验证的洗盘行为。
                    1. A级机会: 健康结构 + 洗盘行为 + 智能企稳。
                    2. S级机会: A级机会 + 能量压缩背景。
        - 状态: 已移除所有调试探针，代码已净化。
        """
        states = {}
        default_series = pd.Series(False, index=df.index)
        p_squeeze = get_params_block(self.strategy, 'squeeze_shakeout_params')

        # --- 1. 定义环境、行为、结果 ---
        # 环境1 (核心): 健康结构
        is_in_healthy_box = self.strategy.atomic_states.get('BOX_STATE_HEALTHY_ACCUMULATION', default_series)
        is_on_stable_platform = self.strategy.atomic_states.get('PLATFORM_STATE_STABLE_FORMED', default_series)
        is_good_structure = is_in_healthy_box | is_on_stable_platform

        # 环境2 (奖励): 能量压缩
        is_in_squeeze = self.strategy.atomic_states.get('VOL_STATE_EXTREME_SQUEEZE', default_series)
        
        # 行为: 洗盘
        drop_threshold = get_param_value(p_squeeze.get('drop_threshold'), -0.03)
        is_sharp_drop = df['pct_change_D'] < drop_threshold
        winner_turnover_col = 'turnover_from_winners_ratio_D'
        if winner_turnover_col not in df.columns: return {}
        winner_inactive_threshold = get_param_value(p_squeeze.get('winner_inactive_threshold'), 60.0)
        is_winner_inactive = df[winner_turnover_col] < winner_inactive_threshold
        shakeout_action = is_sharp_drop & is_winner_inactive
        
        # 结果: 智能企稳
        stabilization_window = get_param_value(p_squeeze.get('stabilization_window'), 3)
        is_stabilized_later = pd.Series(False, index=df.index)
        shakeout_indices = df.index[shakeout_action]
        for idx in shakeout_indices:
            start_pos = df.index.get_loc(idx) + 1
            end_pos = start_pos + stabilization_window
            if end_pos > len(df.index): continue
            window_df = df.iloc[start_pos:end_pos]
            shakeout_day_close = df.at[idx, 'close_D']
            is_price_reclaimed = (window_df['close_D'] > shakeout_day_close).any()
            is_volume_shrinking = (window_df['volume_D'] < df.at[idx, 'volume_D']).all()
            if is_price_reclaimed or is_volume_shrinking:
                is_stabilized_later.at[idx] = True

        # --- 2. 分层定义机会 ---
        # A级机会 (基础版): 健康结构 + 洗盘行为 + 智能企稳
        base_condition_A = is_good_structure & shakeout_action & is_stabilized_later
        
        # S级机会 (加强版): A级机会 + 能量压缩背景
        squeeze_window_days = get_param_value(p_squeeze.get('squeeze_window_days'), 3)
        is_in_squeeze_window = is_in_squeeze.rolling(window=squeeze_window_days, min_periods=1).max().astype(bool)
        
        # 最终裁定
        final_condition_S = base_condition_A & is_in_squeeze_window
        final_condition_A = base_condition_A & ~final_condition_S

        # --- 3. 最终裁定与分级 ---
        states['OPP_SQUEEZE_ZONE_SHAKEOUT_A'] = final_condition_A
        states['OPP_SQUEEZE_ZONE_SHAKEOUT_S'] = final_condition_S

        # (可选) 保留最终的情报报告，这不属于调试探针，而是策略的正常日志输出
        # if final_condition_A.any():
        #     print(f"          -> [A级机会情报] 侦测到 {final_condition_A.sum()} 次“压缩区实战洗盘”机会！")
        # if final_condition_S.any():
        #     print(f"          -> [S级机会情报] 侦测到 {final_condition_S.sum()} 次“压缩区完美洗盘”机会！")

        return states


    def diagnose_post_accumulation_phase(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V313.0 动态粘合版】初升浪诊断模块
        - 核心升级: 使用动态的、基于滚动分位数的“均线粘合压缩”状态，
                    替代了过于严苛的绝对阈值，极大提升了策略的适应性。
        """
        # print("        -> [初升浪诊断模块 V313.0 动态粘合版] 启动...")
        states = {}
        default_series = pd.Series(False, index=df.index)

        p = get_params_block(self.strategy, 'post_accumulation_params')
        if not get_param_value(p.get('enabled'), False):
            return {}

        persistence_days = get_param_value(p.get('persistence_days'), 15)
        break_ma_period = get_param_value(p.get('break_ma_period'), 21)
        break_ma_col = f'EMA_{break_ma_period}_D'
        
        # --- 1. 检查所需情报 ---
        required_states = [
            'MA_STATE_SHORT_CONVERGENCE_SQUEEZE', # 新的动态粘合状态
            'MA_STATE_LONG_CONVERGENCE_SQUEEZE'
        ]
        if any(state not in self.strategy.atomic_states for state in required_states):
            print("          -> [警告] 缺少诊断“初升浪”所需的“动态均线粘合”情报，模块跳过。")
            return {}

        # --- 2. 定义“盘整/筑底”的准备阶段 (Setup Phase) ---
        # 条件A: 短期或长期均线必须处于“粘合压缩”状态
        is_highly_converged = (
            self.strategy.atomic_states.get('MA_STATE_SHORT_CONVERGENCE_SQUEEZE', default_series) |
            self.strategy.atomic_states.get('MA_STATE_LONG_CONVERGENCE_SQUEEZE', default_series)
        )
        
        # 条件B: 结合其他盘整信号，增加确定性
        is_other_setup = (
            self.strategy.atomic_states.get('BOX_STATE_HEALTHY_ACCUMULATION', default_series) |
            self.strategy.atomic_states.get('VOL_STATE_EXTREME_SQUEEZE', default_series)
        )
        
        is_setup_phase = is_highly_converged | is_other_setup
        
        # --- 3. 定义“启动事件” (Ignition Event) ---
        is_positive_candle = df['pct_change_D'] > 0
        is_volume_spike = df['volume_D'] > (df['VOL_MA_21_D'] * 1.5)
        is_ignition_day = is_positive_candle & is_volume_spike
        
        # --- 4. 定义最终的“初升浪启动事件” ---
        was_in_setup_phase = is_setup_phase.shift(1).fillna(False)
        first_breakout_event = was_in_setup_phase & is_ignition_day
        
        states['POST_ACCUMULATION_ASCENT_C'] = first_breakout_event
        print(f"          -> [最终事件] 识别到 {first_breakout_event.sum()} 天为“初升浪启动事件” (POST_ACCUMULATION_ASCENT_C)。")

        # --- 5. 生成“持续性状态” ---
        break_condition = df['close_D'] < df[break_ma_col]
        ascent_state = create_persistent_state(
            df=df,
            entry_event_series=first_breakout_event,
            persistence_days=persistence_days,
            break_condition_series=break_condition,
            state_name='STRUCTURE_POST_ACCUMULATION_ASCENT_C'
        )
        states['STRUCTURE_POST_ACCUMULATION_ASCENT_C'] = ascent_state
        print(f"          -> [最终状态] “初升浪”结构状态共持续 {ascent_state.sum()} 天 (STRUCTURE_POST_ACCUMULATION_ASCENT_C)。")
            
        return states


    def diagnose_breakout_pullback_relay(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V505.6 探针植入版】
        - 核心升级: 植入“情报探针”，在模块执行前检查所有依赖项。
        """
        states = {}
        default_series = pd.Series(False, index=df.index)
        
        # --- 探针部署区 ---
        # 1. 定义本模块需要的所有上游情报（原子状态）
        required_intelligence = [
            'STRUCTURE_POST_ACCUMULATION_ASCENT_C', # “初升浪启动”情报
            'PULLBACK_STATE_HEALTHY_S',             # “健康回踩”情报
            'CHIP_CONC_LOCKED_AND_STABLE_A'         # “筹码高度控盘”情报
        ]
        
        # 2. 检查情报清单
        missing_intelligence = [
            intel for intel in required_intelligence 
            if intel not in self.strategy.atomic_states
        ]
        
        # 3. 如果有情报缺失，则发出详细警报并安全退出
        if missing_intelligence:
            print(f"    -> [警告] 缺少诊断“突破-回踩接力”所需的核心情报，模块跳过。")
            print(f"       -> [探针报告] 缺失的情报清单: {missing_intelligence}")
            return {}
        # --- 探针部署结束 ---

        # 如果所有情报都到位，则继续执行原始逻辑
        p = get_params_block(self.strategy, 'post_accumulation_params').get('relay_params', {})
        if not get_param_value(p.get('enabled'), True):
            return {}

        lookback_window = get_param_value(p.get('lookback_window'), 15)

        # 1. 获取上游情报
        is_ascent_start = self.strategy.atomic_states.get('STRUCTURE_POST_ACCUMULATION_ASCENT_C', default_series)
        is_healthy_pullback = self.strategy.atomic_states.get('PULLBACK_STATE_HEALTHY_S', default_series)
        is_chip_setup = self.strategy.atomic_states.get('CHIP_CONC_LOCKED_AND_STABLE_A', default_series)

        # 2. 定义“接力”逻辑
        # 条件A: 今天是一个“健康回踩”日，并且筹码高度控盘
        is_pullback_opportunity = is_healthy_pullback & is_chip_setup
        # 条件B: 在回踩日之前的N天内，必须发生过“初升浪启动”事件
        had_recent_ascent_start = is_ascent_start.rolling(window=lookback_window, min_periods=1).apply(np.any, raw=True).fillna(0).astype(bool)
        # 最终裁定：S+级的接力机会
        relay_opportunity = is_pullback_opportunity & had_recent_ascent_start
        # if relay_opportunity.any():
        #     print(f"          -> [情报] 侦测到 {relay_opportunity.sum()} 次 S+级“突破-回踩接力”机会！")
        states['PLAYBOOK_BREAKOUT_PULLBACK_RELAY_S_PLUS'] = relay_opportunity
        return states


    def diagnose_holding_risks(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V338.0 新增】持仓风险诊断模块
        - 核心职责: 诊断那些与持仓健康度相关的、更精细的早期预警信号。
        """
        # print("        -> [持仓风险诊断模块 V338.0] 启动...")
        states = {}
        default_series = pd.Series(False, index=df.index)

        # --- 1. 诊断“健康度失速”风险 ---
        is_improving = self.strategy.atomic_states.get('CHIP_DYN_HEALTH_IMPROVING', default_series)
        
        was_improving = is_improving.shift(1).fillna(False)
        is_not_improving_now = ~is_improving
        
        states['HOLD_RISK_HEALTH_STALLING'] = was_improving & is_not_improving_now

        return states

    def diagnose_upthrust_distribution(self, df: pd.DataFrame, exit_params: dict) -> pd.Series:
        """
        【V91.2 函数调用修复版】
        - 核心修复: 使用 numpy.maximum 替代错误的 pd.max，以正确计算上影线。
        """
        p = exit_params.get('upthrust_distribution_params', {})
        if not get_param_value(p.get('enabled'), False):
            return pd.Series(False, index=df.index)

        overextension_ma_period = get_param_value(p.get('overextension_ma_period'), 55)
        overextension_threshold = get_param_value(p.get('overextension_threshold'), 0.3)
        upper_shadow_ratio = get_param_value(p.get('upper_shadow_ratio'), 0.5)
        high_volume_quantile = get_param_value(p.get('high_volume_quantile'), 0.75)
        
        ma_col = f'EMA_{overextension_ma_period}_D'
        vol_ma_col = 'VOL_MA_21_D'
        
        required_cols = ['open_D', 'high_D', 'low_D', 'close_D', 'volume_D', ma_col, vol_ma_col]
        if not all(col in df.columns for col in required_cols):
            print("          -> [警告] 缺少诊断'高位放量长上影'所需列，跳过。")
            return pd.Series(False, index=df.index)

        is_overextended = (df['close_D'] / df[ma_col] - 1) > overextension_threshold
        total_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        
        upper_shadow = (df['high_D'] - np.maximum(df['open_D'], df['close_D']))
        
        has_long_upper_shadow = (upper_shadow / total_range) >= upper_shadow_ratio
        volume_threshold = df['volume_D'].rolling(window=21).quantile(high_volume_quantile)
        is_high_volume = df['volume_D'] > volume_threshold
        is_weak_close = df['close_D'] < (df['high_D'] + df['low_D']) / 2
        
        signal = is_overextended & has_long_upper_shadow & is_high_volume & is_weak_close
        # print(f"          -> '高位放量长上影派发' 风险诊断完成，共激活 {signal.sum()} 天。{format_debug_dates(signal)}")
        return signal


    def diagnose_structure_breakdown(self, df: pd.DataFrame, exit_params: dict) -> pd.Series:
        """
        诊断“结构性破位”风险 (Structure Breakdown)。
        这是一个非常重要的趋势终结信号。
        """
        p = exit_params.get('structure_breakdown_params', {})
        if not get_param_value(p.get('enabled'), False):
            return pd.Series(False, index=df.index)

        # 1. 定义参数
        breakdown_ma_period = get_param_value(p.get('breakdown_ma_period'), 21)
        min_pct_change = get_param_value(p.get('min_pct_change'), -0.03)
        high_volume_quantile = get_param_value(p.get('high_volume_quantile'), 0.75)
        
        ma_col = f'EMA_{breakdown_ma_period}_D'
        
        required_cols = ['open_D', 'close_D', 'pct_change_D', 'volume_D', ma_col]
        if not all(col in df.columns for col in required_cols):
            print("          -> [警告] 缺少诊断'结构性破位'所需列，跳过。")
            return pd.Series(False, index=df.index)

        # 2. 计算各项条件
        # 条件A: 是一根有分量的阴线
        is_decisive_negative_candle = df['pct_change_D'] < min_pct_change
        
        # 条件B: 相对放量
        volume_threshold = df['volume_D'].rolling(window=21).quantile(high_volume_quantile)
        is_high_volume = df['volume_D'] > volume_threshold
        
        # 条件C: 跌破了关键均线
        is_breaking_ma = df['close_D'] < df[ma_col]
        
        # 3. 组合所有条件
        signal = is_decisive_negative_candle & is_high_volume & is_breaking_ma
        
        # print(f"          -> '结构性破位' 风险诊断完成，共激活 {signal.sum()} 天。{format_debug_dates(signal)}")
        return signal


    def diagnose_volume_price_dynamics(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V284.0 新增】量价关系动态分析中心 (CT扫描室)
        - 核心职责: 对“成交量”和“资金攻击效率”进行全面的斜率与加速度分析，
                    将“天量对倒”这个模糊概念，升级为可量化、可跟踪的动态风险信号。
        """
        # print("          -> [量价动态分析中心 V284.0] 启动，正在对“天量对倒”进行CT扫描...")
        states = {}
        default_series = pd.Series(False, index=df.index)

        # --- 1. 军备检查 ---
        required_cols = [
            'volume_D', 'VOL_MA_21_D', 'pct_change_D',
            'SLOPE_5_volume_D', 'ACCEL_5_volume_D',
            'VPA_EFFICIENCY_D', 'SLOPE_5_VPA_EFFICIENCY_D' # 假设数据工程层已提供
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"            -> [严重警告] 量价动态分析中心缺少关键数据: {missing_cols}，模块已跳过！")
            return states

        # --- 2. 风险分析：识别“无效天量”和“效率衰竭” ---
        
        # 风险信号1: 【滞涨】天量但价格不涨 (最经典的顶部风险)
        # 定义: 成交量远超均线，但日涨幅却很小。
        p_vpa = params.get('vpa_dynamics_params', {})
        volume_ratio_high = get_param_value(p_vpa.get('volume_ratio_high'), 2.5)
        pct_change_low = get_param_value(p_vpa.get('pct_change_low'), 0.01)
        is_huge_volume = df['volume_D'] > (df['VOL_MA_21_D'] * volume_ratio_high)
        is_price_stagnant = df['pct_change_D'].abs() < pct_change_low
        states['RISK_VPA_STAGNATION'] = is_huge_volume & is_price_stagnant
        
        # 风险信号2: 【效率衰竭】资金攻击效率持续下降
        # 定义: 资金效率的5日斜率为负。
        states['RISK_VPA_EFFICIENCY_DECLINING'] = df['SLOPE_5_VPA_EFFICIENCY_D'] < 0
        
        # 风险信号3: 【量能失控】成交量仍在加速放大
        # 定义: 成交量的5日加速度为正，说明市场情绪可能过热，换手失控。
        states['RISK_VPA_VOLUME_ACCELERATING'] = df['ACCEL_5_volume_D'] > 0

        # --- 3. 【S级风险融合】定义“动态对倒风险” ---
        # 最终裁决: 只要出现“滞涨”或“效率衰竭”，就视为高风险。如果同时伴随“量能失控”，则是最高级别的风险。
        is_high_risk = states.get('RISK_VPA_STAGNATION', default_series) | states.get('RISK_VPA_EFFICIENCY_DECLINING', default_series)
        is_critical_risk = is_high_risk & states.get('RISK_VPA_VOLUME_ACCELERATING', default_series)
        
        # 我们将这个融合后的S级风险，命名为“动态对倒风险”
        states['COGNITIVE_RISK_DYNAMIC_DECEPTIVE_CHURN'] = is_high_risk | is_critical_risk
        
        # print("          -> [量价动态分析中心 V284.0] CT扫描完成。")
        return states












