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

    def synthesize_behavioral_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.0 新增】行为模式评分模块
        - 核心职责: 将 diagnose_behavioral_patterns 中的布尔逻辑，升级为连续的0-1得分。
        """
        print("        -> [行为模式评分模块 V1.0] 启动...")
        # --- 1. 量化“建设性洗盘吸筹”得分 ---
        # 将构成机会的多个条件相乘，得到最终置信度
        # 条件1: 上升趋势背景得分
        uptrend_score = (df.get('close_D', 1) / df.get('EMA_55_D', 1)).clip(lower=1.0) - 1.0
        uptrend_context_score = self.strategy._calculate_normalized_score(uptrend_score, window=120, ascending=True)
        # 条件2: 下跌行为得分 (跌幅在-2%到-7%之间得分最高)
        drop_score = 1 - (df['pct_change_D'] - (-0.045)).abs() / 0.025
        meaningful_drop_score = drop_score.clip(0, 1)
        # 条件3: 套牢盘割肉得分
        losers_capitulating_score = self.strategy._calculate_normalized_score(df.get('turnover_from_losers_ratio_D', 0), window=120, ascending=True)
        # 条件4: 获利盘锁仓得分
        winners_holding_score = self.strategy._calculate_normalized_score(df.get('turnover_from_winners_ratio_D', 0), window=120, ascending=False)
        # 条件5: 筹码结构改善得分
        chip_improving_score = self.strategy._calculate_normalized_score(df.get('SLOPE_5_concentration_90pct_D', 0), window=120, ascending=False)
        # 最终得分
        df['BEHAVIOR_SCORE_OPP_WASHOUT_ABSORPTION'] = (
            uptrend_context_score *
            meaningful_drop_score *
            losers_capitulating_score *
            winners_holding_score *
            chip_improving_score
        )
        self.strategy.atomic_states['BEHAVIOR_SCORE_OPP_WASHOUT_ABSORPTION'] = df['BEHAVIOR_SCORE_OPP_WASHOUT_ABSORPTION']
        return df

    def diagnose_kline_patterns(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V273.0 装备换代版】
        - 核心修复: 更新了对 `_create_persistent_state` 方法的调用方式。
        - 新协议: 使用了最新的参数名 `entry_event_series` 和 `break_condition_series`，
                  使其完全兼容我们新建的 V271.0 “状态机引擎”。
        - 收益: 确保了基础侦察部队能够正确使用现代化的通用工具，实现了全军装备的同步。
        """
        df = self.synthesize_behavioral_scores(df)
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
                entry_event_series=gap_up_event,
                persistence_days=persistence_days,
                break_condition_series=price_fills_gap,
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

        # --- 4. “老鸭头”形态形成中 (Old Duck Head Forming) ---
        p_duck = p.get('old_duck_head_params', {})
        if get_param_value(p_duck.get('enabled'), True):
            # 定义均线
            ma5_col = 'EMA_5_D'
            ma10_col = 'EMA_10_D'
            ma60_col = 'EMA_60_D'
            required_cols = [ma5_col, ma10_col, ma60_col, 'VOL_MA_21_D', 'volume_D']
            if all(c in df.columns for c in required_cols):
                # 条件1: 5日、10日均线在60日均线上方，形成“鸭头顶”
                head_formed = (df[ma5_col].shift(1) > df[ma60_col].shift(1)) & (df[ma10_col].shift(1) > df[ma60_col].shift(1))
                # 条件2: 5日均线死叉10日均线，形成“鸭鼻孔”
                is_dead_cross = (df[ma5_col] < df[ma10_col]) & (df[ma5_col].shift(1) >= df[ma10_col].shift(1))
                # 条件3: 形成死叉后，股价并未大幅下跌，而是在60日线上方缩量整理
                is_shrinking_volume = df['volume_D'] < df['VOL_MA_21_D']
                is_above_ma60 = df['close_D'] > df[ma60_col]
                # 将“死叉”事件转化为一个持续状态，代表“鸭头”形成后的整理期
                duck_head_forming_event = is_dead_cross & head_formed
                # 状态打破条件：5日线重新金叉10日线（即将张口），或跌破60日线
                break_condition = ((df[ma5_col] > df[ma10_col]) & (df[ma5_col].shift(1) <= df[ma10_col].shift(1))) | (df['close_D'] < df[ma60_col])
                # 使用状态机生成“老鸭头形成中”的持续状态
                duck_head_state = create_persistent_state(
                    df=df,
                    entry_event_series=duck_head_forming_event,
                    persistence_days=get_param_value(p_duck.get('max_forming_days'), 20),
                    break_condition_series=break_condition,
                    state_name='KLINE_STATE_OLD_DUCK_HEAD_FORMING'
                )
                # 最终状态必须满足缩量和在60日线上方整理
                states['KLINE_STATE_OLD_DUCK_HEAD_FORMING'] = duck_head_state & is_shrinking_volume & is_above_ma60

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

        advanced_atomics = self.diagnose_advanced_atomic_signals(df)
        states.update(advanced_atomics)

        # --- 调用高级行为状态诊断模块 ---
        advanced_behaviors = self.diagnose_advanced_behavioral_states(df)
        states.update(advanced_behaviors)
        return states

    def diagnose_advanced_atomic_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】【代码优化】高级原子信号诊断模块
        - 核心职责: 基于基础指标的斜率、加速度以及多指标的交互关系，生成更深层次的原子信号。
        - 产出: 一系列描述市场微观动态和潜在拐点的原子状态。
        - 优化说明: 将原有效率极低的 rolling().apply() 斜率计算，替换为基于 shift() 的高性能向量化公式。
        """
        # print("        -> [高级原子诊断模块 V1.0] 启动，正在生成深层动态信号...")
        states = {}
        p = get_params_block(self.strategy, 'advanced_atomic_params', {}) # 假设配置文件中有此参数块
        if not get_param_value(p.get('enabled'), True): return states
        # --- 1. K线与价格行为原子信号 ---
        # 信号1: 收盘价在当日K线中的位置 (0-1之间，越高越强)
        price_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        close_position_in_range = (df['close_D'] - df['low_D']) / price_range
        states['PRICE_DYN_STRONG_CLOSE'] = close_position_in_range > get_param_value(p.get('strong_close_threshold'), 0.8)
        states['PRICE_DYN_WEAK_CLOSE'] = close_position_in_range < get_param_value(p.get('weak_close_threshold'), 0.2)
        # 信号2: 连续上涨/下跌天数 (用于判断短期惯性)
        is_up_day = df['pct_change_D'] > 0
        is_down_day = df['pct_change_D'] < 0
        # 使用cumsum技巧计算连续天数
        up_streak = is_up_day.groupby((is_up_day != is_up_day.shift()).cumsum()).cumcount() + 1
        down_streak = is_down_day.groupby((is_down_day != is_down_day.shift()).cumsum()).cumcount() + 1
        states['PRICE_DYN_CONSECUTIVE_UP_STREAK_3D'] = (up_streak >= 3) & is_up_day
        states['PRICE_DYN_CONSECUTIVE_DOWN_STREAK_3D'] = (down_streak >= 3) & is_down_day
        # --- 2. 波动率动态原子信号 ---
        bbw_col = 'BBW_21_2.0_D'
        bbw_slope_col = 'SLOPE_5_BBW_21_2.0_D'
        bbw_accel_col = 'ACCEL_5_BBW_21_2.0_D'
        if all(c in df.columns for c in [bbw_col, bbw_slope_col, bbw_accel_col]):
            # 信号3: 波动率扩张/收缩 (基于斜率)
            states['VOL_DYN_EXPANDING'] = df[bbw_slope_col] > 0
            states['VOL_DYN_CONTRACTING'] = df[bbw_slope_col] < 0
            # 信号4: 波动率加速扩张 (基于加速度，可能预示极端行情)
            states['VOL_DYN_ACCEL_EXPANSION'] = df[bbw_accel_col] > 0
        # --- 3. 振荡器动态原子信号 (以RSI为例) ---
        rsi_col = 'RSI_13_D'
        # 假设数据工程层已经计算了RSI的斜率和加速度
        rsi_slope_col = 'SLOPE_5_RSI_13_D' # 假设列名为 SLOPE_5_RSI_13_D
        rsi_accel_col = 'ACCEL_5_RSI_13_D' # 假设列名为 ACCEL_5_RSI_13_D
        # 为了演示，如果数据不存在，我们即时计算。但在生产环境中，应由数据工程层提供。
        if rsi_col in df.columns:
            if rsi_slope_col not in df.columns:
                y = df[rsi_col]
                slope = (-2 * y.shift(4) - y.shift(3) + y.shift(1) + 2 * y) / 10
                df[rsi_slope_col] = slope
            if rsi_accel_col not in df.columns:
                df[rsi_accel_col] = df[rsi_slope_col].diff()
            if all(c in df.columns for c in [rsi_slope_col, rsi_accel_col]):
                # 信号5: RSI动能上升/下降 (基于斜率)
                states['OSC_DYN_RSI_MOMENTUM_RISING'] = df[rsi_slope_col] > 0
                states['OSC_DYN_RSI_MOMENTUM_FALLING'] = df[rsi_slope_col] < 0
                # 信号6: RSI动能加速/减速 (基于加速度)
                states['OSC_DYN_RSI_MOMENTUM_ACCELERATING'] = df[rsi_accel_col] > 0
                states['OSC_DYN_RSI_MOMENTUM_DECELERATING'] = df[rsi_accel_col] < 0
                # 信号7: 经典看跌背离 (价格上涨，但RSI动能下降)
                is_price_up = df['pct_change_D'] > 0
                states['OSC_DYN_RSI_BEARISH_DIVERGENCE'] = is_price_up & states['OSC_DYN_RSI_MOMENTUM_FALLING']
        # --- 4. 筹码行为动态原子信号 ---
        loser_turnover_slope_col = 'SLOPE_5_turnover_from_losers_ratio_D'
        loser_turnover_accel_col = 'ACCEL_5_turnover_from_losers_ratio_D'
        if all(c in df.columns for c in [loser_turnover_slope_col, loser_turnover_accel_col]):
            # 信号8: 恐慌盘抛售衰竭机会 (抛售仍在继续，但速度在减慢)
            is_loser_selling = df[loser_turnover_slope_col] > 0
            is_selling_decelerating = df[loser_turnover_accel_col] < 0
            states['CHIP_BEHAVIOR_LOSER_CAPITULATION_EXHAUSTION'] = is_loser_selling & is_selling_decelerating
        # --- 5. 多时间维度交叉验证原子信号 (Multi-Timeframe Cross-Validation Atomics) ---
        # 军备检查：确保所有必需的斜率和加速度列都存在
        required_multi_timeframe_cols = [
            'SLOPE_5_concentration_90pct_D', 'ACCEL_5_concentration_90pct_D', 'SLOPE_21_concentration_90pct_D',
            'SLOPE_5_winner_profit_margin_D', 'SLOPE_21_winner_profit_margin_D',
            'SLOPE_55_concentration_90pct_D', 'SLOPE_5_turnover_from_losers_ratio_D', 'SLOPE_5_turnover_from_winners_ratio_D',
            'SLOPE_55_turnover_from_winners_ratio_D', 'SLOPE_5_close_D',
            'ACCEL_5_VPA_EFFICIENCY_D', 'SLOPE_21_EMA_55_D' # 假设 EMA_55_D 的21日斜率已由数据工程层提供
        ]
        # 动态计算缺失的加速度列，如果数据工程层未提供 (生产环境应由数据工程层提供)
        if 'SLOPE_5_winner_profit_margin_D' in df.columns and 'ACCEL_5_winner_profit_margin_D' not in df.columns:
            df['ACCEL_5_winner_profit_margin_D'] = df['SLOPE_5_winner_profit_margin_D'].diff()
        if 'SLOPE_5_pressure_above_D' in df.columns and 'ACCEL_5_pressure_above_D' not in df.columns:
            df['ACCEL_5_pressure_above_D'] = df['SLOPE_5_pressure_above_D'].diff()
        if 'SLOPE_5_energy_ratio_D' in df.columns and 'ACCEL_5_energy_ratio_D' not in df.columns:
            df['ACCEL_5_energy_ratio_D'] = df['SLOPE_5_energy_ratio_D'].diff()
        if 'SLOPE_5_VPA_EFFICIENCY_D' in df.columns and 'ACCEL_5_VPA_EFFICIENCY_D' not in df.columns:
            df['ACCEL_5_VPA_EFFICIENCY_D'] = df['SLOPE_5_VPA_EFFICIENCY_D'].diff()
        if all(col in df.columns for col in required_multi_timeframe_cols):
            # === 机会信号 ===
            # 信号9 (A级机会): 筹码加速集中确认中期趋势
            # 解读: 短期筹码集中加速，且中期筹码也处于集中状态，表明吸筹趋势强劲且持续。
            is_short_accel_conc = df['ACCEL_5_concentration_90pct_D'] < get_param_value(p.get('concentration_accel_confirm_threshold'), 0)
            is_mid_term_conc = df['SLOPE_21_concentration_90pct_D'] < 0
            states['OPP_CHIP_ACCEL_CONCENTRATION_CONFIRMED_A'] = is_short_accel_conc & is_mid_term_conc
            # 信号10 (B级机会): 获利盘利润垫短期反转
            # 解读: 获利盘利润垫短期开始回升，而中期曾处于下降趋势，可能预示着止跌企稳或反弹。
            is_short_profit_rising = df['SLOPE_5_winner_profit_margin_D'] > get_param_value(p.get('profit_cushion_reversal_slope_threshold'), 0)
            is_mid_profit_falling = df['SLOPE_21_winner_profit_margin_D'] < 0
            states['OPP_BEHAVIOR_PROFIT_CUSHION_REVERSAL_B'] = is_short_profit_rising & is_mid_profit_falling
            # 信号11 (A级机会): 长期吸筹下的短期洗盘
            # 解读: 长期筹码持续集中，但短期出现套牢盘恐慌性抛售，而获利盘惜售，是主力利用洗盘吸筹的经典行为。
            is_long_term_conc = df['SLOPE_55_concentration_90pct_D'] < 0
            is_short_loser_selling = df['SLOPE_5_turnover_from_losers_ratio_D'] > get_param_value(p.get('wash_out_accumulation_loser_slope_threshold'), 0)
            is_short_winner_holding = df['SLOPE_5_turnover_from_winners_ratio_D'] < get_param_value(p.get('wash_out_accumulation_winner_slope_threshold'), 0)
            states['OPP_BEHAVIOR_WASH_OUT_ACCUMULATION_A'] = is_long_term_conc & is_short_loser_selling & is_short_winner_holding
            # === 风险信号 ===
            # 信号12 (B级风险): 筹码短期发散预警
            # 解读: 中期筹码仍在集中，但短期已出现发散迹象，可能是主力开始试探性派发或洗盘力度过大。
            is_short_term_diverging = df['SLOPE_5_concentration_90pct_D'] > get_param_value(p.get('chip_short_term_divergence_slope_threshold'), 0)
            is_mid_term_concentrating = df['SLOPE_21_concentration_90pct_D'] < 0
            states['RISK_CHIP_SHORT_TERM_DIVERGENCE_WARNING_B'] = is_short_term_diverging & is_mid_term_concentrating
            # 信号13 (S级风险): 长期派发下的诱多拉升
            # 解读: 长期获利盘持续派发，但短期股价却在上涨，这是最危险的诱多出货信号。
            is_long_term_winner_dist = df['SLOPE_55_turnover_from_winners_ratio_D'] > get_param_value(p.get('deceptive_rally_long_term_winner_slope_threshold'), 0)
            is_short_term_price_rally = df['SLOPE_5_close_D'] > 0
            states['RISK_BEHAVIOR_DECEPTIVE_RALLY_LONG_TERM_S'] = is_long_term_winner_dist & is_short_term_price_rally
            # 信号14 (C级风险): 市场效率加速衰竭
            # 解读: 资金攻击效率加速下降，即使在上升趋势中也预示着动能的严重不足，可能面临滞涨或反转。
            is_efficiency_accel_falling = df['ACCEL_5_VPA_EFFICIENCY_D'] < get_param_value(p.get('market_engine_stalling_accel_threshold'), 0)
            is_in_uptrend_context = df['SLOPE_21_EMA_55_D'] > 0 # 假设21日EMA55斜率代表中期趋势
            states['RISK_MARKET_ENGINE_STALLING_ACCEL_C'] = is_efficiency_accel_falling & is_in_uptrend_context
        else:
            missing_multi_cols = [col for col in required_multi_timeframe_cols if col not in df.columns]
            print(f"            -> [警告] 多时间维度交叉验证模块缺少关键数据: {missing_multi_cols}，部分信号已跳过！")
        # --- 6. 静态-多动态交叉验证原子信号 (Static-Multi-Dynamic Cross-Validation) ---
        # 军备检查：确保所有必需的静态和动态列都存在
        required_static_multi_dyn_cols = [
            'close_D', 'EMA_233_D', 'SLOPE_5_close_D', 'ACCEL_5_close_D',
            'concentration_90pct_D', 'SLOPE_5_concentration_90pct_D', 'ACCEL_5_concentration_90pct_D',
            'winner_profit_margin_D', 'SLOPE_5_winner_profit_margin_D',
            'total_winner_rate_D', 'SLOPE_5_turnover_from_winners_ratio_D', 'ACCEL_5_turnover_from_winners_ratio_D',
            'BBW_21_2.0_D', 'SLOPE_5_BBW_21_2.0_D', 'ACCEL_5_volume_D', 'SLOPE_5_pressure_above_D',
            'SLOPE_5_turnover_from_losers_ratio_D', 'ACCEL_5_turnover_from_losers_ratio_D'
        ]
        # 动态计算缺失的加速度列，如果数据工程层未提供 (生产环境应由数据工程层提供)
        if 'SLOPE_5_close_D' in df.columns and 'ACCEL_5_close_D' not in df.columns:
            df['ACCEL_5_close_D'] = df['SLOPE_5_close_D'].diff()
        if 'SLOPE_5_turnover_from_winners_ratio_D' in df.columns and 'ACCEL_5_turnover_from_winners_ratio_D' not in df.columns:
            df['ACCEL_5_turnover_from_winners_ratio_D'] = df['SLOPE_5_turnover_from_winners_ratio_D'].diff()
        if 'SLOPE_5_volume_D' in df.columns and 'ACCEL_5_volume_D' not in df.columns:
            df['ACCEL_5_volume_D'] = df['SLOPE_5_volume_D'].diff()
        if all(col in df.columns for col in required_static_multi_dyn_cols):
            # === 机会信号 ===
            # 信号15 (S级机会): 长期底部反转共振
            # 解读: 静态上，价格处于长期均线下方（超跌）；动态上，短期价格斜率转正，筹码集中度加速改善，获利盘利润垫回升。
            #      这是多重信号共振的强劲底部反转信号。
            is_below_long_ma_static = df['close_D'] < df['EMA_233_D'] * get_param_value(p.get('long_ma_oversold_ratio'), 0.95) # 价格低于233日均线5%
            is_short_price_slope_positive = df['SLOPE_5_close_D'] > get_param_value(p.get('short_price_slope_positive_threshold'), 0)
            is_chip_accel_improving = df['ACCEL_5_concentration_90pct_D'] < get_param_value(p.get('chip_accel_improving_threshold'), 0) # 集中度加速下降（集中度改善）
            is_profit_cushion_rising = df['SLOPE_5_winner_profit_margin_D'] > get_param_value(p.get('profit_cushion_rising_threshold'), 0)
            states['OPP_STATIC_LONG_TERM_BOTTOM_REVERSAL_S'] = (
                is_below_long_ma_static &
                is_short_price_slope_positive &
                is_chip_accel_improving &
                is_profit_cushion_rising
            )
            # 信号16 (A级机会): 极致压缩后的蓄势待发
            # 解读: 静态上，波动率处于极致收缩状态；动态上，波动率仍在收缩，成交量加速萎缩，上方压力正在被消化。
            #      这是突破前的典型蓄势信号。
            is_extreme_squeeze_static = df['BBW_21_2.0_D'] < get_param_value(p.get('bbw_extreme_squeeze_threshold'), 0.05) # BBW低于0.05
            is_bbw_contracting_dyn = df['SLOPE_5_BBW_21_2.0_D'] < get_param_value(p.get('bbw_contracting_slope_threshold'), 0)
            is_volume_accel_shrinking_dyn = df['ACCEL_5_volume_D'] < get_param_value(p.get('volume_accel_shrinking_threshold'), 0) # 成交量加速萎缩
            is_pressure_clearing_dyn = df['SLOPE_5_pressure_above_D'] < get_param_value(p.get('pressure_clearing_slope_threshold'), 0)
            states['OPP_STATIC_EXTREME_SQUEEZE_ACCUMULATION_A'] = (
                is_extreme_squeeze_static &
                is_bbw_contracting_dyn &
                is_volume_accel_shrinking_dyn &
                is_pressure_clearing_dyn
            )
            # === 风险信号 ===
            # 信号17 (S级风险): 高位多重背离派发
            # 解读: 静态上，价格处于近期高位；动态上，短期价格斜率下降，获利盘换手率加速上升，筹码集中度开始发散。
            #      这是多重信号共振的强劲顶部派发信号。
            is_near_high_range_static = states.get('PRICE_STATE_NEAR_HIGH_RANGE', pd.Series(False, index=df.index)) # 假设此信号已由其他模块生成
            is_short_price_slope_negative = df['SLOPE_5_close_D'] < get_param_value(p.get('short_price_slope_negative_threshold'), 0)
            is_winner_selling_accel_dyn = df['ACCEL_5_turnover_from_winners_ratio_D'] > get_param_value(p.get('winner_selling_accel_threshold'), 0) # 获利盘抛售加速
            is_chip_diverging_dyn = df['SLOPE_5_concentration_90pct_D'] > get_param_value(p.get('chip_diverging_slope_threshold'), 0)
            states['RISK_STATIC_HIGH_ALTITUDE_MULTI_DIVERGENCE_S'] = (
                is_near_high_range_static &
                is_short_price_slope_negative &
                is_winner_selling_accel_dyn &
                is_chip_diverging_dyn
            )
            # 信号18 (A级风险): 趋势末期主力诱多出货
            # 解读: 静态上，获利盘比例很高（市场情绪乐观）；动态上，短期价格加速上涨，但套牢盘抛售却在加速。
            #      这表明主力在利用拉升吸引散户接盘，同时悄悄出货。
            is_high_winner_rate_static = df['total_winner_rate_D'] > get_param_value(p.get('high_winner_rate_static_threshold'), 90.0)
            is_price_accel_rising_dyn = df['ACCEL_5_close_D'] > get_param_value(p.get('price_accel_rising_threshold'), 0)
            is_loser_selling_accel_dyn = df['ACCEL_5_turnover_from_losers_ratio_D'] > get_param_value(p.get('loser_selling_accel_threshold'), 0) # 套牢盘抛售加速
            states['RISK_STATIC_DECEPTIVE_RALLY_ACCEL_DISTRIBUTION_A'] = (
                is_high_winner_rate_static &
                is_price_accel_rising_dyn &
                is_loser_selling_accel_dyn
            )
        else:
            missing_static_multi_dyn_cols = [col for col in required_static_multi_dyn_cols if col not in df.columns]
            print(f"            -> [警告] 静态-多动态交叉验证模块缺少关键数据: {missing_static_multi_dyn_cols}，部分信号已跳过！")
        print(f"        -> [高级原子诊断模块 V1.3] 已生成 {len(states)} 个深层动态信号。")
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
        【V509.0 质量门升级版】【代码优化】
        - 核心升级: 全面收紧了“健康回踩”的定义，增加了“未破关键趋势线”的硬性约束，
                    并收紧了对成交量和获利盘惜售的判断标准，旨在从根本上提升信号质量。
        - 优化说明: 将原有的 for 循环判断逻辑，重构为一次性的高性能向量化操作，提升执行效率。
        """
        # print("        -> [统一回踩诊断中心 V509.0 质量门升级版] 启动...")
        states = {}
        default_series = pd.Series(False, index=df.index)
        p = get_params_block(self.strategy, 'pullback_analysis_params')
        if not get_param_value(p.get('enabled'), False):
            return states
        # --- 1. 军备检查 ---
        required_cols = [
            'pct_change_D', 'turnover_from_losers_ratio_D', 'turnover_from_winners_ratio_D',
            'SLOPE_5_concentration_90pct_D', 'close_D', 'low_D', 'volume_D', 'VOL_MA_21_D', 'EMA_21_D'
        ]
        if any(c not in df.columns for c in required_cols):
            print("          -> [警告] 缺少诊断“回踩性质”所需列，模块跳过。")
            return states
        # --- 2. 定义通用的“建设性背景” ---
        is_in_uptrend = self.strategy.atomic_states.get('STRUCTURE_MAIN_UPTREND_WAVE_S', default_series)
        is_in_ascent_wave = self.strategy.atomic_states.get('STRUCTURE_POST_ACCUMULATION_ASCENT_C', default_series)
        is_constructive_context = is_in_uptrend | is_in_ascent_wave
        # --- 3. 识别并定性回踩行为 ---
        is_pullback_day = df['pct_change_D'] < 0
        # 引入更严格、更细致的回踩性质判断，以取代粗糙的均线回踩触发器。
        # 3.1 全面收紧“健康回踩”的特征定义
        p_healthy = p.get('healthy_pullback_rules', {})
        is_gentle_drop = df['pct_change_D'] > get_param_value(p_healthy.get('min_pct_change'), -0.05)
        # 条件1: 必须缩量
        shrinking_ratio = get_params_block(self.strategy, 'volatility_state_params').get('shrinking_ratio', 0.8)
        is_shrinking_volume = df['volume_D'] < (df['VOL_MA_21_D'] * shrinking_ratio)
        # 条件2: 获利盘必须惜售
        winner_turnover_low_threshold = get_param_value(p_healthy.get('max_winner_turnover_ratio'), 40.0)
        is_winner_holding_tight = df['turnover_from_winners_ratio_D'] < winner_turnover_low_threshold
        # 条件3: 必须守住趋势生命线 (允许盘中跌破，但收盘价必须收复)
        is_above_trend_line = df['close_D'] > df['EMA_21_D']
        is_healthy_character = is_gentle_drop & is_shrinking_volume & is_winner_holding_tight & is_above_trend_line
        # 3.2 定义“打压式回踩”（黄金坑）的特征
        p_suppression = p.get('suppression_pullback_rules', {})
        is_significant_drop = df['pct_change_D'] < get_param_value(p_suppression.get('min_drop_pct'), -0.03)
        is_panic_selling = df['turnover_from_losers_ratio_D'] > get_param_value(p_suppression.get('min_loser_turnover_ratio'), 40.0)
        is_winner_holding = df['turnover_from_winners_ratio_D'] < get_param_value(p_suppression.get('max_winner_turnover_ratio'), 30.0)
        is_suppressive_character = is_significant_drop & is_panic_selling & is_winner_holding
        # --- 4. 组合生成最终的中立状态信号 ---
        divergence_tolerance = get_param_value(p_suppression.get('divergence_tolerance'), 0.0005)
        is_chip_stable = df['SLOPE_5_concentration_90pct_D'] < divergence_tolerance
        # 生成“健康回踩”状态信号
        states['PULLBACK_STATE_HEALTHY_S'] = is_pullback_day & is_healthy_character & is_constructive_context
        if states['PULLBACK_STATE_HEALTHY_S'].any():
            print(f"          -> [情报] 侦测到 {states['PULLBACK_STATE_HEALTHY_S'].sum()} 次“S级健康回踩”状态。")
        # 生成“打压回踩被确认”状态信号 (这是一个时序逻辑)
        is_suppression_event = is_pullback_day & is_suppressive_character & is_constructive_context & is_chip_stable
        min_rebound_days = get_param_value(p_suppression.get('min_rebound_days'), 1)
        max_rebound_days = get_param_value(p_suppression.get('max_rebound_days'), 3)
        conditions = []
        for i in range(min_rebound_days, max_rebound_days + 1):
            # 条件1: i天前是否发生了“打压事件”
            is_prev_suppression = is_suppression_event.shift(i).fillna(False)
            # 条件2: 当前收盘价是否高于i天前的收盘价
            is_price_recovered = df['close_D'] > df['close_D'].shift(i)
            # 将两个条件组合
            conditions.append(is_prev_suppression & is_price_recovered)
        # 只要在回看窗口内（1-3天）有任何一天满足条件，就认为反弹被确认
        is_rebound_confirmed = pd.concat(conditions, axis=1).any(axis=1)
        states['PULLBACK_STATE_SUPPRESSIVE_S'] = is_rebound_confirmed
        if states['PULLBACK_STATE_SUPPRESSIVE_S'].any():
            print(f"          -> [情报] 侦测到 {states['PULLBACK_STATE_SUPPRESSIVE_S'].sum()} 次“S级打压回踩被确认”状态。")
        return states

    def diagnose_behavioral_patterns(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V338.0 微观重构版】主力操纵战术“反侦察”模块
        - 核心重构 (本次修改): 彻底重写了 `OPP_CONSTRUCTIVE_WASHOUT_ABSORPTION_A` 的定义。
                        废除了旧的、仅基于价格和筹码集中度斜率的模糊判断。
                        新定义引入了微观成交结构指标，通过精确量化“谁在卖、谁在买”，
                        将一个低效信号升级为能够精准识别“主力利用恐慌盘吸筹”的A级机会信号。
        """
        states = {}
        default_series = pd.Series(False, index=df.index)

        # --- 1. 军备检查 ---
        required_cols = [
            'pct_change_D', 'volume_D', 'VOL_MA_21_D', 'close_D', 'EMA_55_D',
            'turnover_from_losers_ratio_D', 'turnover_from_winners_ratio_D',
            'SLOPE_5_concentration_90pct_D'
        ]
        if any(c not in df.columns for c in required_cols):
            missing_cols = [c for c in required_cols if c not in df.columns]
            print(f"          -> [警告] 缺少诊断行为模式所需列，模块跳过。缺失: {missing_cols}")
            return {}
        
        # --- 2. “建设性洗盘吸筹”机会诊断 (A级机会) ---
        washout_score = df.get('BEHAVIOR_SCORE_OPP_WASHOUT_ABSORPTION', pd.Series(0.5, index=df.index))
        # 定义：当综合得分进入历史最高的15%区间时，确认为机会
        states['OPP_CONSTRUCTIVE_WASHOUT_ABSORPTION_A'] = washout_score > washout_score.rolling(120).quantile(0.85)
        
        # --- 3. “诱多派发”风险诊断 (逻辑保持，但基础更清晰) ---
        is_strong_rally = df['pct_change_D'] > 0.03
        is_chip_diverging = self.strategy.atomic_states.get('CHIP_DYN_DIVERGING', default_series)
        is_core_distribution_conflict = is_strong_rally & is_chip_diverging
        states['BEHAVIOR_DECEPTIVE_RALLY_A'] = is_core_distribution_conflict
        states['BEHAVIOR_DECEPTIVE_RALLY_S'] = is_core_distribution_conflict

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
            self.strategy.atomic_states.get('STRUCTURE_BOX_ACCUMULATION_A', default_series) |
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
        # print(f"          -> [最终事件] 识别到 {first_breakout_event.sum()} 天为“初升浪启动事件” (POST_ACCUMULATION_ASCENT_C)。")

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
        # print(f"          -> [最终状态] “初升浪”结构状态共持续 {ascent_state.sum()} 天 (STRUCTURE_POST_ACCUMULATION_ASCENT_C)。")
            
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
        
        # print("          -> [量价动态分析中心 V284.0] CT扫描完成。")
        return states

    def _diagnose_pullback_enhancement_matrix(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】回踩形态增强矩阵
        - 核心职责: 识别回踩事件是否伴随着特殊的、能增强信号信服度的形态。
                      这是所有回踩战法的“增强器”模块。
        - 输出: 返回一个包含各种增强信号（如 is_hammer, is_fib_support）的字典。
        """
        # print("        -> [回踩增强矩阵 V1.0] 启动，正在扫描特殊形态...")
        enhancements = {}
        atomic = self.strategy.atomic_states
        default_series = pd.Series(False, index=df.index)

        # --- 增强器1: K线形态 (锤子线/探针) ---
        body = (df['close_D'] - df['open_D']).abs().replace(0, 0.0001)
        lower_shadow = df[['open_D', 'close_D']].min(axis=1) - df['low_D']
        upper_shadow = df['high_D'] - df[['open_D', 'close_D']].max(axis=1)
        enhancements['is_hammer_candle'] = (lower_shadow >= body * 2.0) & (upper_shadow < body * 0.8)

        # --- 增强器2: 关键支撑位 (斐波那契) ---
        enhancements['is_fib_golden_support'] = atomic.get('OPP_FIB_SUPPORT_GOLDEN_POCKET_S', default_series)
        enhancements['is_fib_standard_support'] = atomic.get('OPP_FIB_SUPPORT_STANDARD_A', default_series)

        # --- 增强器3: 特殊结构 (压缩区洗盘) ---
        # 注意：这里的逻辑是从 diagnose_squeeze_zone_opportunities 迁移并简化而来
        is_in_squeeze = atomic.get('VOL_STATE_EXTREME_SQUEEZE', default_series)
        is_sharp_drop = df['pct_change_D'] < -0.03
        is_winner_inactive = df.get('turnover_from_winners_ratio_D', pd.Series(np.inf, index=df.index)) < 60.0
        enhancements['is_squeeze_shakeout'] = is_in_squeeze & is_sharp_drop & is_winner_inactive
        
        # --- 增强器4: 打压性质 (黄金坑) ---
        # 打压回踩本身就是一个强大的增强信号
        enhancements['is_suppressive_pullback'] = atomic.get('PULLBACK_STATE_SUPPRESSIVE_S', default_series)

        return enhancements

    # “价格-成交量原子信号诊断”方法
    def diagnose_price_volume_atomics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】价格与成交量基础原子信号诊断模块
        - 核心职责: 生成描述当前价格位置和基础量价关系的中性原子信号。
                    这些信号是构成更复杂战术判断的“乐高积木”。
        - 输出: 一系列描述客观事实的原子状态，如 PRICE_... 和 VOL_...。
        """
        # print("        -> [价格成交量原子诊断模块 V1.0] 启动...")
        states = {}
        p = get_params_block(self.strategy, 'price_volume_atomic_params')
        if not get_param_value(p.get('enabled'), True): return states

        # --- 1. 价格位置原子信号 (Price Position Atomics) ---
        # 信号1: 价格处于近期高位区间 (潜在突破区)
        lookback_period = get_param_value(p.get('range_lookback'), 20)
        high_range_percentile = get_param_value(p.get('high_range_percentile'), 0.90)
        rolling_high = df['high_D'].rolling(window=lookback_period).max()
        rolling_low = df['low_D'].rolling(window=lookback_period).min()
        price_range = (rolling_high - rolling_low).replace(0, np.nan)
        high_range_threshold = rolling_low + price_range * high_range_percentile
        states['PRICE_STATE_NEAR_HIGH_RANGE'] = df['close_D'] > high_range_threshold

        # 信号2: 价格处于近期低位区间 (潜在支撑区)
        low_range_percentile = get_param_value(p.get('low_range_percentile'), 0.10)
        low_range_threshold = rolling_low + price_range * low_range_percentile
        states['PRICE_STATE_NEAR_LOW_RANGE'] = df['close_D'] < low_range_threshold

        # --- 2. 量价关系原子信号 (Volume-Price Atomics) ---
        vol_ma_col = 'VOL_MA_21_D'
        if vol_ma_col in df.columns:
            # 信号3: 价涨量增 (健康的上涨)
            is_price_up = df['pct_change_D'] > 0
            is_volume_up = df['volume_D'] > df['volume_D'].shift(1)
            states['VOL_BEHAVIOR_HEALTHY_RALLY'] = is_price_up & is_volume_up

            # 信号4: 价涨量缩 (上涨动能衰竭的迹象)
            is_volume_down = df['volume_D'] < df['volume_D'].shift(1)
            states['VOL_BEHAVIOR_UNHEALTHY_RALLY'] = is_price_up & is_volume_down

            # 信号5: 价跌量增 (恐慌或放量下跌)
            is_price_down = df['pct_change_D'] < 0
            states['VOL_BEHAVIOR_DISTRIBUTION_DROP'] = is_price_down & is_volume_up

            # 信号6: 价跌量缩 (下跌动能减弱或惜售)
            states['VOL_BEHAVIOR_WEAKENING_DROP'] = is_price_down & is_volume_down

        print(f"        -> [价格成交量原子诊断模块 V1.0] 已生成 {len(states)} 个基础原子信号。")
        return states

    def diagnose_advanced_behavioral_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】高级行为状态诊断模块
        - 核心职责: 基于获利盘利润垫、上方压力、市场能量效率等深层指标的动态变化，
                      生成描述市场心理和运行效率的高级行为原子信号。
        - 收益: 提供了超越传统量价分析的视角，能够更早地发现持股信心的变化和趋势衰竭的迹象。
        """
        # print("        -> [高级行为状态诊断模块 V1.0] 启动...")
        states = {}
        default_series = pd.Series(False, index=df.index)

        # --- 1. 军备检查 ---
        required_cols = [
            'SLOPE_5_winner_profit_margin_D',
            'SLOPE_5_pressure_above_D',
            'SLOPE_5_energy_ratio_D',
            'EMA_55_D'
        ]
        if any(c not in df.columns for c in required_cols):
            missing_cols = [c for c in required_cols if c not in df.columns]
            print(f"          -> [警告] 缺少高级行为诊断所需列，模块跳过。缺失: {missing_cols}")
            return {}

        # --- 2. 获利盘心理动态 (Profit Cushion Dynamics) ---
        # 信号1 (A级机会): 利润安全垫扩张
        # 解读: 获利盘的平均利润在增加，表明持股信心增强，抛压减小，趋势健康。
        is_cushion_expanding = df['SLOPE_5_winner_profit_margin_D'] > 0
        states['BEHAVIOR_PROFIT_CUSHION_EXPANDING_A'] = is_cushion_expanding

        # 信号2 (A级风险): 利润安全垫收缩
        # 解读: 获利盘的平均利润在减少，可能引发恐慌性抛售，是趋势弱化的早期预警。
        is_cushion_shrinking = df['SLOPE_5_winner_profit_margin_D'] < 0
        states['RISK_BEHAVIOR_PROFIT_CUSHION_SHRINKING_A'] = is_cushion_shrinking

        # --- 3. 市场阻力动态 (Overhead Pressure Dynamics) ---
        # 信号3 (B级机会): 上方套牢盘压力正在被消化
        # 解读: 上方的套牢盘正在减少，意味着多头正在成功地解放前期的被套筹码，为上涨扫清障碍。
        is_pressure_clearing = df['SLOPE_5_pressure_above_D'] < 0
        states['BEHAVIOR_CLEARING_OVERHEAD_PRESSURE_B'] = is_pressure_clearing

        # 信号4 (B级风险): 上方套牢盘压力正在积聚
        # 解读: 股价上涨乏力，导致上方套牢盘越来越多，形成沉重的阻力位。
        is_pressure_building = df['SLOPE_5_pressure_above_D'] > 0
        states['RISK_BEHAVIOR_BUILDING_OVERHEAD_PRESSURE_B'] = is_pressure_building

        # --- 4. 市场引擎效率动态 (Market Engine Efficiency) ---
        # 信号5 (B级机会): 市场引擎正在加速
        # 解读: 资金的攻击效率在提升，表明市场上涨的“性价比”很高，是健康的表现。
        is_engine_accelerating = df['SLOPE_5_energy_ratio_D'] > 0
        is_in_uptrend = df['close_D'] > df['EMA_55_D']
        states['BEHAVIOR_MARKET_ENGINE_ACCELERATING_B'] = is_engine_accelerating & is_in_uptrend

        # 信号6 (B级风险): 市场引擎正在失速
        # 解读: 资金的攻击效率在下降，可能出现“费力不讨好”的滞涨情况，是趋势衰竭的前兆。
        is_engine_stalling = df['SLOPE_5_energy_ratio_D'] < 0
        states['RISK_BEHAVIOR_MARKET_ENGINE_STALLING_B'] = is_engine_stalling & is_in_uptrend

        return states




