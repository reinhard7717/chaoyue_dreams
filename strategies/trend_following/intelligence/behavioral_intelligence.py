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
        【V273.4 数值化增强版】
        - 核心升级 (本次修改):
          - [数值化] 将 'KLINE_STATE_GAP_SUPPORT_ACTIVE' 布尔状态，增强为包含
                      支撑强度的 'SCORE_GAP_SUPPORT_ACTIVE' 数值化评分。
        - 核心逻辑: 新评分在缺口支撑状态激活期间，基于当前价格与缺口上沿的距离
                    进行量化，距离越远，支撑强度越高，分数越高。
        - 收益: 为下游认知层提供了更精细的输入，能够区分“弱支撑”和“强支撑”。
        """
        states = {}
        p = get_params_block(self.strategy, 'kline_pattern_params')
        if not get_param_value(p.get('enabled'), False): return states
        # --- 1. “巨阴洗盘”机会窗口 (Washout Opportunity Window) ---
        p_washout = p.get('washout_params', {})
        if get_param_value(p_washout.get('enabled'), True):
            vol_ma_col = 'VOL_MA_21_D'
            if 'pct_change_D' in df.columns and vol_ma_col in df.columns:
                norm_window = get_param_value(p_washout.get('norm_window'), 120)
                min_periods = max(1, norm_window // 5)
                drop_score = (1 - df['pct_change_D'].rolling(window=norm_window, min_periods=min_periods).rank(pct=True)).fillna(0.5)
                volume_ratio = (df['volume_D'] / df[vol_ma_col].replace(0, np.nan)).fillna(1.0)
                volume_score = volume_ratio.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
                is_negative_change = df['pct_change_D'] < 0
                states['SCORE_BEHAVIOR_WASHOUT'] = (drop_score * volume_score * is_negative_change).astype(np.float32)
        # --- 2. “缺口支撑”持续状态 (Gap Support Active State) ---
        p_gap = p.get('gap_support_params', {})
        if get_param_value(p_gap.get('enabled'), True):
            persistence_days = get_param_value(p_gap.get('persistence_days'), 10)
            gap_up_event = df['low_D'] > df['high_D'].shift(1)
            gap_high = df['high_D'].shift(1).where(gap_up_event).ffill() # 使用ffill填充缺口支撑线
            price_fills_gap = df['close_D'] < gap_high
            # 步骤1: 创建定义“缺口支撑有效”的布尔状态窗口
            gap_support_state = create_persistent_state(
                df=df,
                entry_event_series=gap_up_event,
                persistence_days=persistence_days,
                break_condition_series=price_fills_gap,
                state_name='KLINE_STATE_GAP_SUPPORT_ACTIVE' # 临时状态名
            )
            states['KLINE_STATE_GAP_SUPPORT_ACTIVE'] = gap_support_state # 保留布尔信号以兼容旧逻辑
            # 步骤2: [数值化升级] 在状态激活期间，计算支撑强度分
            # 逻辑: 当前最低价距离缺口上沿越远，支撑越强，分数越高。
            # 使用当日收盘价的10%作为归一化基准，避免绝对值问题
            support_distance = (df['low_D'] - gap_high).clip(lower=0)
            normalization_base = (df['close_D'] * 0.1).replace(0, np.nan)
            support_strength_score = (support_distance / normalization_base).clip(0, 1).fillna(0)
            # 最终得分仅在缺口支撑状态激活时有效
            states['SCORE_GAP_SUPPORT_ACTIVE'] = (support_strength_score * gap_support_state).astype(np.float32) # 新增代码行
        # --- 3. “N字板”盘整状态 (N-Shape Consolidation State) ---
        p_nshape = p.get('n_shape_params', {})
        if get_param_value(p_nshape.get('enabled'), True):
            rally_threshold = get_param_value(p_nshape.get('rally_threshold'), 0.097)
            consolidation_days_max = get_param_value(p_nshape.get('consolidation_days_max'), 3)
            norm_window = get_param_value(p_nshape.get('norm_window'), 120)
            min_periods = max(1, norm_window // 5)
            is_strong_rally = df['pct_change_D'] >= rally_threshold
            rally_strength_score = df['pct_change_D'].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
            rally_strength_score = (rally_strength_score * is_strong_rally)
            max_recent_rally_score = rally_strength_score.shift(1).rolling(window=consolidation_days_max, min_periods=1).max().fillna(0)
            consolidation_quality_score = (1 - df['pct_change_D'].abs().rolling(window=norm_window, min_periods=min_periods).rank(pct=True)).fillna(0.5)
            is_not_rally_today = df['pct_change_D'] < rally_threshold
            states['SCORE_N_SHAPE_CONSOLIDATION'] = (max_recent_rally_score * consolidation_quality_score * is_not_rally_today).astype(np.float32)
        # --- 4. “老鸭头”形态形成中 (Old Duck Head Forming) ---
        p_duck = p.get('old_duck_head_params', {})
        if get_param_value(p_duck.get('enabled'), True):
            ma5_col = 'EMA_5_D'
            ma13_col = 'EMA_13_D'
            ma55_col = 'EMA_55_D'
            required_cols = [ma5_col, ma13_col, ma55_col, 'VOL_MA_21_D', 'volume_D']
            if all(c in df.columns for c in required_cols):
                norm_window = get_param_value(p_duck.get('norm_window'), 120)
                min_periods = max(1, norm_window // 5)
                head_formed = (df[ma5_col].shift(1) > df[ma55_col].shift(1)) & (df[ma13_col].shift(1) > df[ma55_col].shift(1))
                is_dead_cross = (df[ma5_col] < df[ma13_col]) & (df[ma5_col].shift(1) >= df[ma13_col].shift(1))
                duck_head_forming_event = is_dead_cross & head_formed
                break_condition = ((df[ma5_col] > df[ma13_col]) & (df[ma5_col].shift(1) <= df[ma13_col].shift(1))) | (df['close_D'] < df[ma55_col])
                duck_head_state = create_persistent_state(
                    df=df,
                    entry_event_series=duck_head_forming_event,
                    persistence_days=get_param_value(p_duck.get('max_forming_days'), 20),
                    break_condition_series=break_condition,
                    state_name='KLINE_STATE_OLD_DUCK_HEAD_FORMING'
                )
                volume_ratio = (df['volume_D'] / df['VOL_MA_21_D'].replace(0, np.nan)).fillna(1.0)
                volume_shrink_score = (1 - volume_ratio.rolling(window=norm_window, min_periods=min_periods).rank(pct=True)).fillna(0.5)
                price_support_ratio = (df['close_D'] / df[ma55_col] - 1).clip(0)
                price_support_score = price_support_ratio.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
                states['SCORE_OLD_DUCK_HEAD_FORMING'] = (volume_shrink_score * price_support_score * duck_head_state).astype(np.float32)
        p_atomic = p.get('atomic_behavior_params', {})
        if get_param_value(p_atomic.get('enabled'), True):
            if 'pct_change_D' in df.columns:
                norm_window = get_param_value(p_atomic.get('norm_window'), 120)
                min_periods = max(1, norm_window // 5)
                sharp_drop_score = (1 - df['pct_change_D'].rolling(window=norm_window, min_periods=min_periods).rank(pct=True)).fillna(0.5)
                is_negative_change = df['pct_change_D'] < 0
                states['SCORE_KLINE_SHARP_DROP'] = (sharp_drop_score * is_negative_change).astype(np.float32)
        advanced_atomics = self.diagnose_advanced_atomic_signals(df)
        states.update(advanced_atomics)
        synthesized_behaviors = self.synthesize_behavioral_patterns(df)
        states.update(synthesized_behaviors)
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
        【V1.6 终极数值化版】回踩形态增强矩阵
        - 核心职责: (原有注释)
        - 核心升级 (本次修改):
          - [数值化] 将 'SCORE_HAMMER_CANDLE_STRENGTH' 的计算逻辑，从依赖布尔过滤器
                        'is_potential_hammer'，升级为完全基于连续评分的融合。
        - 收益: 消除了K线形态识别中的硬编码阈值，使得评分能够平滑地反映
                K线从“非锤子”到“完美锤子”的渐变过程，信号质量更高。
        """
        print("        -> [回踩增强矩阵 V1.6] 启动，正在扫描特殊形态...") # 更新版本号
        enhancements = {}
        atomic = self.strategy.atomic_states
        default_score_series = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 增强器1: K线形态 (锤子线/探针) ---
        body = (df['close_D'] - df['open_D']).abs().replace(0, 0.0001)
        lower_shadow = (df[['open_D', 'close_D']].min(axis=1) - df['low_D']).clip(0)
        upper_shadow = (df['high_D'] - df[['open_D', 'close_D']].max(axis=1)).clip(0)
        # 组件1 - 下影线强度分。从满足最低标准(1.8倍实体)开始平滑计分，下影线越长分越高。
        lower_shadow_score = ((lower_shadow / body - 1.8) / 3).clip(0, 1).fillna(0)
        # 组件2 - 短上影线得分。上影线越短(相对于0.8倍实体)，分数越高。
        short_upper_shadow_score = (1 - (upper_shadow / (body * 0.8))).clip(0, 1).fillna(0)
        # 融合生成最终的锤子线强度分
        enhancements['SCORE_HAMMER_CANDLE_STRENGTH'] = (lower_shadow_score * short_upper_shadow_score).astype(np.float32)
        # --- 增强器2: 关键支撑位 (斐波那契) ---
        enhancements['SCORE_FIB_REBOUND_S'] = atomic.get('SCORE_FIB_REBOUND_S', default_score_series)
        enhancements['SCORE_FIB_REBOUND_A'] = atomic.get('SCORE_FIB_REBOUND_A', default_score_series)
        enhancements['SCORE_FIB_REBOUND_B'] = atomic.get('SCORE_FIB_REBOUND_B', default_score_series)
        # --- 增强器3: 特殊结构 (压缩区洗盘) ---
        squeeze_level_score = atomic.get('SCORE_VOL_COMPRESSION_LEVEL', default_score_series)
        sharp_drop_score = atomic.get('SCORE_KLINE_SHARP_DROP', default_score_series)
        enhancements['SCORE_SQUEEZE_SHAKEOUT'] = (squeeze_level_score * sharp_drop_score).astype(np.float32)
        return enhancements

    def diagnose_board_patterns(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V58.2 终极数值化版】
        - 核心升级 (本次修改):
          - [数值化] 将地天板/天地板事件判断中的硬布尔过滤 'is_..._event'，
                      升级为基于价格与涨跌停价接近度的连续评分。
        - 核心逻辑: 最终评分 = f(振幅强度, 端点价格接近度分)，
                    完整地量化了“地天板”或“天地板”的形态质量。
        - 收益: 能够区分“擦边”的形态和“标准”的形态，信号精度大幅提升。
        """
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
        # --- 核心评分组件 ---
        # 组件1: 振幅强度分 
        day_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        theoretical_max_range = (limit_up_price - limit_down_price).replace(0, np.nan)
        strength_score = (day_range / theoretical_max_range).clip(0, 1).fillna(0)
        # 组件2 - 地天板的“触地”分 (最低价越接近跌停价，分数越高)
        low_near_limit_down_score = ((limit_down_price * (1 + price_buffer) - df['low_D']) / (limit_down_price * price_buffer).replace(0, np.nan)).clip(0, 1).fillna(0)
        # 组件3 - 地天板的“封天”分 (收盘价越接近涨停价，分数越高)
        close_near_limit_up_score = ((df['close_D'] - limit_up_price * (1 - price_buffer)) / (limit_up_price * price_buffer).replace(0, np.nan)).clip(0, 1).fillna(0)
        # 组件4 - 天地板的“触天”分 (最高价越接近涨停价，分数越高)
        high_near_limit_up_score = ((df['high_D'] - limit_up_price * (1 - price_buffer)) / (limit_up_price * price_buffer).replace(0, np.nan)).clip(0, 1).fillna(0)
        # 组件5 - 天地板的“封地”分 (收盘价越接近跌停价，分数越高)
        close_near_limit_down_score = ((limit_down_price * (1 + price_buffer) - df['close_D']) / (limit_down_price * price_buffer).replace(0, np.nan)).clip(0, 1).fillna(0)
        # --- 地天板 (Earth to Heaven) ---
        states['SCORE_BOARD_EARTH_HEAVEN'] = (strength_score * low_near_limit_down_score * close_near_limit_up_score).astype(np.float32)
        # --- 天地板 (Heaven to Earth) ---
        states['SCORE_BOARD_HEAVEN_EARTH'] = (strength_score * high_near_limit_up_score * close_near_limit_down_score).astype(np.float32)
        return states

    def diagnose_upthrust_distribution(self, df: pd.DataFrame, exit_params: dict) -> pd.Series:
        """
        【V91.4 终极数值化版】
        - 核心重构: (同V91.3)
        - 核心升级 (本次修改):
          - [数值化] 将“上影线强度分”的计算逻辑，从基于硬阈值的过滤，
                      升级为一个从最低阈值开始平滑递增的连续评分。
        - 收益: 彻底消除了模块内最后一个硬编码的布尔判断逻辑，使得所有评分
                的响应都平滑且连续，信号质量达到理论上的最高水平。
        """
        p = exit_params.get('upthrust_distribution_params', {})
        if not get_param_value(p.get('enabled'), False):
            return pd.Series(0.0, index=df.index, name='SCORE_RISK_UPTHRUST_DISTRIBUTION')
        overextension_ma_period = get_param_value(p.get('overextension_ma_period'), 55)
        upper_shadow_ratio_min = get_param_value(p.get('upper_shadow_ratio_min'), 0.5) # 作为评分的起点
        ma_col = f'EMA_{overextension_ma_period}_D'
        required_cols = ['open_D', 'high_D', 'low_D', 'close_D', 'volume_D', ma_col]
        if not all(col in df.columns for col in required_cols):
            print("          -> [警告] 缺少诊断'高位派发风险'所需列，跳过。")
            return pd.Series(0.0, index=df.index, name='SCORE_RISK_UPTHRUST_DISTRIBUTION')
        # --- 计算四维风险评分组件 ---
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        # 1. 价格乖离度分
        overextension_ratio = (df['close_D'] / df[ma_col] - 1).clip(0)
        overextension_score = overextension_ratio.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        # 2. 上影线强度分
        total_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        upper_shadow = (df['high_D'] - np.maximum(df['open_D'], df['close_D']))
        upper_shadow_ratio = (upper_shadow / total_range).fillna(0.0)
        # 评分逻辑 - 当上影线比例达到下限时分数为0，之后随比例增加而线性增长，到100%时分数为1。
        scaling_range = 1.0 - upper_shadow_ratio_min
        scaling_range = max(scaling_range, 0.001) # 避免除以零
        upper_shadow_score = ((upper_shadow_ratio - upper_shadow_ratio_min) / scaling_range).clip(0, 1).fillna(0)
        # 3. 成交量能级分
        volume_score = df['volume_D'].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        # 4. 收盘弱势度分 (收盘价在K线中的位置越低，分数越高)
        close_position_in_range = ((df['close_D'] - df['low_D']) / total_range).fillna(0.5)
        weak_close_score = 1 - close_position_in_range
        # --- 融合生成最终风险分 ---
        final_score = (overextension_score * upper_shadow_score * volume_score * weak_close_score).astype(np.float32)
        final_score.name = 'SCORE_RISK_UPTHRUST_DISTRIBUTION' # 命名Series
        return final_score

    def diagnose_ma_breakdown(self, exit_params: dict) -> pd.Series:
        """
        【V2.1 数值化升级版】诊断“均线破位”行为
        - 核心重构: 将原有的布尔信号升级为数值化的“均线破位深度分”。
        - 核心逻辑: 评分基于收盘价跌破均线的幅度进行归一化处理，
                    深度越大，分数越高，更能体现风险的严重程度。
        - 收益: 为下游风险评估提供了更精细的输入，能够区分“轻微破位”和“严重破位”。
        """
        p = exit_params.get('structure_breakdown_params', {})
        if not get_param_value(p.get('enabled'), False):
            return pd.Series(0.0, index=self.strategy.df.index, name='SCORE_BEHAVIOR_MA_BREAKDOWN') # 返回带名称的0值Series
        breakdown_ma_period = get_param_value(p.get('breakdown_ma_period'), 21)
        ma_col = f'EMA_{breakdown_ma_period}_D'
        df = self.strategy.df_indicators
        required_cols = ['close_D', ma_col]
        if not all(col in df.columns for col in required_cols):
            print(f"          -> [警告] 缺少诊断'均线破位'所需列: {required_cols}，跳过。")
            return pd.Series(0.0, index=df.index, name='SCORE_BEHAVIOR_MA_BREAKDOWN') # 返回带名称的0值Series
        # 计算破位深度 (百分比)
        breakdown_depth = ((df[ma_col] - df['close_D']) / df[ma_col].replace(0, np.nan)).fillna(0)
        # 仅在实际破位时计算深度，否则为0
        breakdown_depth = breakdown_depth.where(df['close_D'] < df[ma_col], 0).clip(0)
        # 对破位深度进行归一化，生成最终评分
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        # 只有在有破位发生时才进行排名，避免全0序列产生无意义的排名
        score = breakdown_depth.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        final_score = (score * (breakdown_depth > 0)).astype(np.float32)
        final_score.name = 'SCORE_BEHAVIOR_MA_BREAKDOWN' # 命名Series
        return final_score

    def diagnose_volume_price_dynamics(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V284.1 全面数值化版】量价关系动态分析中心 (CT扫描室)
        - 核心职责: (原有注释)
        - 核心升级 (本次修改):
          - [数值化] 将所有基于硬阈值的布尔风险信号，全面升级为0-1的数值化风险评分。
        - 收益: 能够量化“天量对倒”风险的严重程度，为风险管理提供更精细的输入。
        """
        states = {}
        required_cols = [
            'volume_D', 'VOL_MA_21_D', 'pct_change_D',
            'SLOPE_5_volume_D', 'ACCEL_5_volume_D',
            'VPA_EFFICIENCY_D', 'SLOPE_5_VPA_EFFICIENCY_D'
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"            -> [严重警告] 量价动态分析中心缺少关键数据: {missing_cols}，模块已跳过！")
            return states
        # --- 初始化归一化参数 ---
        p_vpa = params.get('vpa_dynamics_params', {})
        norm_window = get_param_value(p_vpa.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        # --- 2. 风险分析：识别“无效天量”和“效率衰竭” ---
        # 风险信号1: 【滞涨】天量但价格不涨 -> 升级为数值化评分
        # 组件1: 天量程度分
        volume_ratio = (df['volume_D'] / df['VOL_MA_21_D'].replace(0, np.nan)).fillna(1.0)
        huge_volume_score = volume_ratio.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        # 组件2: 价格停滞度分 (涨跌幅绝对值越小，分数越高)
        price_stagnant_score = (1 - df['pct_change_D'].abs().rolling(window=norm_window, min_periods=min_periods).rank(pct=True)).fillna(0.5)
        states['SCORE_RISK_VPA_STAGNATION'] = (huge_volume_score * price_stagnant_score).astype(np.float32)
        # 风险信号2: 【效率衰竭】资金攻击效率持续下降 -> 升级为数值化评分
        # 逻辑: 效率斜率越负，分数越高
        efficiency_decline_score = (1 - df['SLOPE_5_VPA_EFFICIENCY_D'].rolling(window=norm_window, min_periods=min_periods).rank(pct=True)).fillna(0.5)
        is_declining = df['SLOPE_5_VPA_EFFICIENCY_D'] < 0
        states['SCORE_RISK_VPA_EFFICIENCY_DECLINING'] = (efficiency_decline_score * is_declining).astype(np.float32)
        # 风险信号3: 【量能失控】成交量仍在加速放大 -> 升级为数值化评分
        # 逻辑: 成交量加速度越正，分数越高
        volume_accelerating_score = df['ACCEL_5_volume_D'].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        is_accelerating = df['ACCEL_5_volume_D'] > 0
        states['SCORE_RISK_VPA_VOLUME_ACCELERATING'] = (volume_accelerating_score * is_accelerating).astype(np.float32)
        return states

    # “价格-成交量原子信号诊断”方法
    def diagnose_price_volume_atomics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.2 终极数值化版】价格与成交量基础原子信号诊断模块
        - 核心职责: (原有注释)
        - 核心升级 (本次修改):
          - [数值化] 在 'SCORE_VOL_WEAKENING_DROP' 的计算中，移除了一个冗余的
                      布尔过滤器，完全依赖于底层的数值化评分。
        - 收益: 使得“价跌量缩”的评分更加平滑和准确，避免了因成交量
                在均线附近微小波动而导致评分突变为0的情况。
        """
        states = {}
        p = get_params_block(self.strategy, 'price_volume_atomic_params')
        if not get_param_value(p.get('enabled'), True): return states
        norm_window = get_param_value(p.get('norm_window'), 120) # 统一的归一化窗口
        min_periods = max(1, norm_window // 5) # 统一的最小周期
        # --- 1. 价格位置原子信号 (Price Position Atomics) ---
        lookback_period = get_param_value(p.get('range_lookback'), 20)
        rolling_high = df['high_D'].rolling(window=lookback_period).max()
        rolling_low = df['low_D'].rolling(window=lookback_period).min()
        price_range = (rolling_high - rolling_low).replace(0, np.nan)
        close_position_in_range = ((df['close_D'] - rolling_low) / price_range).clip(0, 1).fillna(0.5)
        states['SCORE_PRICE_POSITION_IN_RECENT_RANGE'] = close_position_in_range.astype(np.float32)
        # --- 2. 量价关系原子信号 (Volume-Price Atomics) ---
        vol_ma_col = 'VOL_MA_21_D'
        if vol_ma_col in df.columns and 'pct_change_D' in df.columns:
            # 组件1: 价跌幅度分 (跌得越多，分数越高)
            price_drop_score = (1 - df['pct_change_D'].rolling(window=norm_window, min_periods=min_periods).rank(pct=True)).fillna(0.5)
            # 组件2: 量缩程度分 (成交量相对均线越小，分数越高)
            volume_ratio = (df['volume_D'] / df[vol_ma_col].replace(0, np.nan)).fillna(1.0)
            volume_shrink_score = (1 - volume_ratio.rolling(window=norm_window, min_periods=min_periods).rank(pct=True)).fillna(0.5)
            # 移除了 (df['volume_D'] < df[vol_ma_col]) 这个硬布尔过滤，因为 volume_shrink_score 已经包含了这个信息
            is_drop_day = (df['pct_change_D'] < 0)
            # 融合生成最终分数，仅保留核心的“下跌日”判断
            states['SCORE_VOL_WEAKENING_DROP'] = (price_drop_score * volume_shrink_score * is_drop_day).astype(np.float32)
        print(f"        -> [价格成交量原子诊断模块 V1.2] 已生成 {len(states)} 个数值化基础原子信号。") # 更新版本号
        return states

    def synthesize_behavioral_patterns(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.1 职责增强版】高级行为模式合成模块
        - 核心职责: 承接原 CognitiveIntelligence 中的部分行为合成逻辑，遵循分层架构原则。
                      融合基础的行为原子信号，生成更高维度的、描述特定战术场景的认知分数。
        - 本次升级: 新增了对“经典形态”和“压缩区洗盘反转”机会的初级合成，为认知层提供更纯净的输入。
        - 收益: 净化了认知层的职责，使其专注于纯粹的“元融合”，同时将行为相关的合成逻辑内聚于此。
        """
        # print("        -> [高级行为模式合成模块 V1.1 职责增强版] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        default_series = pd.Series(False, index=df.index)
        # --- 1. 合成“反转潜力”信号 (原 synthesize_reversal_potential_signals) ---
        up_streak_score = (atomic.get('COUNT_CONSECUTIVE_UP_STREAK', default_score) / 5.0).clip(0, 1)
        down_streak_score = (atomic.get('COUNT_CONSECUTIVE_DOWN_STREAK', default_score) / 5.0).clip(0, 1)
        close_position_score = atomic.get('SCORE_PRICE_POSITION_IN_RANGE', default_score)
        upthrust_risk_score = atomic.get('SCORE_RISK_UPTHRUST_DISTRIBUTION', default_score)
        hammer_strength_score = atomic.get('SCORE_HAMMER_CANDLE_STRENGTH', default_score)
        top_reversal_potential = up_streak_score * (1 - close_position_score) * upthrust_risk_score
        states['SCORE_BEHAVIOR_TOP_REVERSAL_POTENTIAL_A'] = top_reversal_potential.astype(np.float32)
        bottom_reversal_potential = down_streak_score * close_position_score * hammer_strength_score
        states['SCORE_BEHAVIOR_BOTTOM_REVERSAL_POTENTIAL_A'] = bottom_reversal_potential.astype(np.float32)
        
        # --- 新增开始: 探针 ---
        print("        -> [探针] 正在检查 'BEHAVIOR_TOP_REVERSAL_POTENTIAL_A' 的构成...")
        print("           - up_streak_score (连续上涨) stats:")
        print(up_streak_score.describe().to_string().replace('\n', '\n             '))
        print("           - (1 - close_position_score) (收盘弱势) stats:")
        print((1 - close_position_score).describe().to_string().replace('\n', '\n             '))
        print("           - upthrust_risk_score (冲高回落风险) stats:")
        print(upthrust_risk_score.describe().to_string().replace('\n', '\n             '))
        print("           - FINAL SCORE_BEHAVIOR_TOP_REVERSAL_POTENTIAL_A stats:")
        print(states['SCORE_BEHAVIOR_TOP_REVERSAL_POTENTIAL_A'].describe().to_string().replace('\n', '\n             '))
        # --- 新增结束: 探针 ---

        # --- 2. 合成“结构性破位”风险 (原 synthesize_breakdown_risk) ---
        ma_broken_score = atomic.get('SCORE_BEHAVIOR_MA_BREAKDOWN', default_score)
        volume_spike_down_score = atomic.get('SCORE_VOL_PRICE_PANIC_DOWN_RISK', default_score)
        sharp_drop_score = atomic.get('SCORE_KLINE_SHARP_DROP', default_score)
        core_risk = ma_broken_score * volume_spike_down_score
        panic_risk = ma_broken_score * sharp_drop_score
        final_breakdown_score = np.maximum(core_risk, panic_risk)
        states['SCORE_BEHAVIOR_STRUCTURE_BREAKDOWN_S'] = final_breakdown_score.astype(np.float32)
        # --- 3. 合成“洗盘”情报 (原 synthesize_washout_intelligence) ---
        behavioral_washout_score = atomic.get('SCORE_BEHAVIOR_WASHOUT', default_score)
        foundation_washout_score = atomic.get('SCORE_VOL_PRICE_PANIC_DOWN_RISK', default_score)
        cognitive_washout_score = np.maximum(behavioral_washout_score, foundation_washout_score)
        states['SCORE_BEHAVIOR_WASHOUT_INTENSITY'] = cognitive_washout_score.astype(np.float32)
        washout_event_threshold = 0.7
        cognitive_washout_event = cognitive_washout_score > washout_event_threshold
        washout_window = cognitive_washout_event.rolling(window=3, min_periods=1).max().astype(bool)
        states['BEHAVIOR_STATE_WASHOUT_WINDOW'] = washout_window
        # --- 4. 诊断“近期反转”上下文 (原 diagnose_recent_reversal_context) ---
        is_reversal_trigger = self.strategy.trigger_events.get('TRIGGER_DOMINANT_REVERSAL', default_series)
        had_recent_reversal = is_reversal_trigger.rolling(window=3, min_periods=1).max().astype(bool)
        states['BEHAVIOR_CONTEXT_RECENT_REVERSAL_SIGNAL'] = had_recent_reversal
        # --- 5. 合成“经典形态”机会 (原 cognitive.synthesize_classic_pattern_opportunity) --- # 新增代码块
        old_duck_head_score = atomic.get('SCORE_OLD_DUCK_HEAD_FORMING', default_score)
        n_shape_score = atomic.get('SCORE_N_SHAPE_CONSOLIDATION', default_score)
        base_pattern_score = np.maximum(old_duck_head_score.values, n_shape_score.values)
        states['SCORE_BEHAVIOR_CLASSIC_PATTERN_OPP'] = pd.Series(base_pattern_score, index=df.index, dtype=np.float32)
        # --- 6. 合成“压缩区洗盘反转”机会 (原 cognitive.synthesize_shakeout_opportunities) --- # 新增代码块
        shakeout_setup_score = atomic.get('SCORE_SQUEEZE_SHAKEOUT', default_score)
        reversal_trigger_score = self.strategy.trigger_events.get('TRIGGER_DOMINANT_REVERSAL', default_series).astype(float)
        # 逻辑: 昨日战备就绪(高质量洗盘) * 今日反转确认
        shakeout_reversal_score = shakeout_setup_score.shift(1).fillna(0.0) * reversal_trigger_score
        states['SCORE_BEHAVIOR_SHAKEOUT_REVERSAL_OPP'] = shakeout_reversal_score.astype(np.float32)
        print(f"        -> [高级行为模式合成模块 V1.1 职责增强版] 计算完毕，新增 {len(states)} 个合成信号。") # 修改: 更新版本号
        return states







