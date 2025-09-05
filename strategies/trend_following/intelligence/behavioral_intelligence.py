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
            # 恢复使用本地的、基于静态阈值的逻辑来定义行为层面的“洗盘事件”
            washout_threshold = get_param_value(p_washout.get('washout_threshold'), -0.07)
            volume_ratio = get_param_value(p_washout.get('washout_volume_ratio'), 1.5)
            vol_ma_col = 'VOL_MA_21_D'
            if 'pct_change_D' in df.columns and vol_ma_col in df.columns:
                is_deep_drop = df['pct_change_D'] < washout_threshold
                is_high_volume = df['volume_D'] > df[vol_ma_col] * volume_ratio
                # 生成一个纯粹基于行为的原子事件信号
                states['BEHAVIOR_EVENT_WASHOUT'] = is_deep_drop & is_high_volume

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
            ma13_col = 'EMA_13_D'
            ma55_col = 'EMA_55_D'
            required_cols = [ma5_col, ma13_col, ma55_col, 'VOL_MA_21_D', 'volume_D']
            if all(c in df.columns for c in required_cols):
                # 条件1: 5日、10日均线在60日均线上方，形成“鸭头顶”
                head_formed = (df[ma5_col].shift(1) > df[ma55_col].shift(1)) & (df[ma13_col].shift(1) > df[ma55_col].shift(1))
                # 条件2: 5日均线死叉10日均线，形成“鸭鼻孔”
                is_dead_cross = (df[ma5_col] < df[ma13_col]) & (df[ma5_col].shift(1) >= df[ma13_col].shift(1))
                # 条件3: 形成死叉后，股价并未大幅下跌，而是在60日线上方缩量整理
                is_shrinking_volume = df['volume_D'] < df['VOL_MA_21_D']
                is_above_ma60 = df['close_D'] > df[ma55_col]
                # 将“死叉”事件转化为一个持续状态，代表“鸭头”形成后的整理期
                duck_head_forming_event = is_dead_cross & head_formed
                # 状态打破条件：5日线重新金叉10日线（即将张口），或跌破60日线
                break_condition = ((df[ma5_col] > df[ma13_col]) & (df[ma5_col].shift(1) <= df[ma13_col].shift(1))) | (df['close_D'] < df[ma55_col])
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

        advanced_atomics = self.diagnose_advanced_atomic_signals(df)
        states.update(advanced_atomics)
        return states

    def diagnose_advanced_atomic_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.1 数值化升级版】高级原子信号诊断模块
        - 核心职责: (原有注释)
        - 核心升级 (本次修改):
          - [数值化] 将 'PRICE_DYN_STRONG_CLOSE'/'WEAK_CLOSE' 两个布尔信号，升级为
                      单一的数值化评分 'SCORE_PRICE_POSITION_IN_RANGE' (0-1)，保留完整信息。
          - [数值化] 将 '..._STREAK_3D' 布尔信号，升级为直接输出连续天数的数值信号
                      'COUNT_CONSECUTIVE_UP/DOWN_STREAK'，供下游更灵活使用。
        """
        # print("        -> [高级原子诊断模块 V1.1] 启动，正在生成深层动态信号...") 
        states = {}
        p = get_params_block(self.strategy, 'advanced_atomic_params', {}) 
        if not get_param_value(p.get('enabled'), True): return states
        # --- 1. K线与价格行为原子信号 ---
        # 升级为数值化评分，并重命名
        price_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        close_position_in_range = ((df['close_D'] - df['low_D']) / price_range).fillna(0.5)
        states['SCORE_PRICE_POSITION_IN_RANGE'] = close_position_in_range.astype(np.float32)
        # 升级为直接输出连续天数的数值信号
        is_up_day = df['pct_change_D'] > 0
        is_down_day = df['pct_change_D'] < 0
        # 使用cumsum技巧计算连续天数
        up_streak = (is_up_day.groupby((is_up_day != is_up_day.shift()).cumsum()).cumcount() + 1) * is_up_day
        down_streak = (is_down_day.groupby((is_down_day != is_down_day.shift()).cumsum()).cumcount() + 1) * is_down_day
        states['COUNT_CONSECUTIVE_UP_STREAK'] = up_streak.astype(np.int16)
        states['COUNT_CONSECUTIVE_DOWN_STREAK'] = down_streak.astype(np.int16)
        print(f"        -> [高级原子诊断模块 V1.1] 已生成 {len(states)} 个深层动态信号。") 
        return states

    def _diagnose_pullback_enhancement_matrix(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.3 全面数值化版】回踩形态增强矩阵
        - 核心职责: (原有注释)
        - 核心升级 (本次修改):
          - [数值化] 将 'is_hammer_candle' 布尔信号升级为 'SCORE_HAMMER_CANDLE_STRENGTH'。
                      评分基于下影线与实体比例，更能体现“锤子”的强度。
        """
        print("        -> [回踩增强矩阵 V1.3] 启动，正在扫描特殊形态...") 
        enhancements = {}
        atomic = self.strategy.atomic_states
        default_series = pd.Series(False, index=df.index)
        default_score_series = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 增强器1: K线形态 (锤子线/探针) ---
        # 升级为数值化评分
        body = (df['close_D'] - df['open_D']).abs().replace(0, 0.0001)
        lower_shadow = (df[['open_D', 'close_D']].min(axis=1) - df['low_D']).clip(0)
        upper_shadow = (df['high_D'] - df[['open_D', 'close_D']].max(axis=1)).clip(0)
        
        # 定义锤子线的基本条件
        is_potential_hammer = (lower_shadow >= body * 1.8) & (upper_shadow < body * 0.8)
        
        # 计算锤子线强度分：下影线/实体比例，进行归一化处理（例如映射到0-1），这里用clip简单处理
        # 一个简单的评分可以是 (下影线/实体比例 - 2) / 3，再clip到0-1，使得比例2为0分，比例5为1分
        hammer_strength_score = ((lower_shadow / body - 2) / 3).clip(0, 1)
        
        # 最终得分必须满足基本形态条件
        enhancements['SCORE_HAMMER_CANDLE_STRENGTH'] = (hammer_strength_score * is_potential_hammer).fillna(0).astype(np.float32)
        # 废弃旧的布尔信号
        # enhancements['is_hammer_candle'] = (lower_shadow >= body * 2.0) & (upper_shadow < body * 0.8)
        # --- 增强器2: 关键支撑位 (斐波那契) ---
        enhancements['is_fib_golden_support'] = atomic.get('OPP_FIB_SUPPORT_GOLDEN_POCKET_S', default_series)
        enhancements['is_fib_standard_support'] = atomic.get('OPP_FIB_SUPPORT_STANDARD_A', default_series)
        # --- 增强器3: 特殊结构 (压缩区洗盘) ---
        # 已在上一轮修改中升级为数值化评分
        squeeze_level_score = atomic.get('SCORE_VOL_COMPRESSION_LEVEL', default_score_series)
        is_sharp_drop = df['pct_change_D'] < -0.03
        enhancements['SCORE_SQUEEZE_SHAKEOUT'] = squeeze_level_score * is_sharp_drop.astype(np.float32)
        return enhancements

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

    def diagnose_ma_breakdown(self, exit_params: dict) -> pd.Series:
        """
        【V2.0 职责净化版】诊断“均线破位”行为
        - 核心重构: 移除了原 diagnose_structure_breakdown 方法中关于跌幅和成交量的判断，
                      使其职责单一化，只负责生成纯粹的“跌破关键均线”这一原子行为信号。
        - 新信号名: BEHAVIOR_MA_BREAKDOWN_A
        - 收益: 遵循了分层架构原则，将战术融合的职责上移至 CognitiveIntelligence，
                避免了在原子信号层进行重复和低效的逻辑判断。
        """
        p = exit_params.get('structure_breakdown_params', {}) # 参数块名称保持不变以便复用
        if not get_param_value(p.get('enabled'), False):
            return pd.Series(False, index=self.strategy.df.index) # 确保返回与df对齐的Series
        # 移除关于跌幅和成交量的参数和逻辑
        breakdown_ma_period = get_param_value(p.get('breakdown_ma_period'), 21)
        ma_col = f'EMA_{breakdown_ma_period}_D'
        df = self.strategy.df_indicators # 明确使用df_indicators
        required_cols = ['close_D', ma_col]
        if not all(col in df.columns for col in required_cols):
            print(f"          -> [警告] 缺少诊断'均线破位'所需列: {required_cols}，跳过。") # 更新打印信息
            return pd.Series(False, index=df.index)
        # 仅保留核心的“跌破均线”逻辑
        is_breaking_ma = df['close_D'] < df[ma_col]
        # 返回单一职责的原子信号
        signal = is_breaking_ma
        # print(f"          -> '均线破位' 行为诊断完成，共激活 {signal.sum()} 天。") # 更新打印信息
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
            # 信号5: 价跌量增 (恐慌或放量下跌)
            is_price_down = df['pct_change_D'] < 0
            is_volume_down = df['volume_D'] < df[vol_ma_col]
            # 信号6: 价跌量缩 (下跌动能减弱或惜售)
            states['VOL_BEHAVIOR_WEAKENING_DROP'] = is_price_down & is_volume_down

        print(f"        -> [价格成交量原子诊断模块 V1.0] 已生成 {len(states)} 个基础原子信号。")
        return states




