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

    def _normalize_series(self, series: pd.Series, norm_window: int, min_periods: int, ascending: bool = True) -> pd.Series:
        """
        辅助函数：将Pandas Series进行滚动窗口排名归一化。
        - 提升为类的私有方法，以供所有诊断引擎复用。
        """
        rank = series.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        return rank if ascending else 1 - rank

    def run_behavioral_analysis_command(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.1 职责修复版】行为情报模块总指挥
        - 核心重构 (本次修改):
          - [职责修复] 修复了V2.0版本中，本方法只调用终极信号引擎，导致其他关键行为诊断
                        （如量价、K线、原子信号等）被跳过，从而引发下游模块（如结构层）
                        缺少依赖数据（例如 SCORE_RISK_VPA_STAGNATION）的严重问题。
          - [恢复职责] 本方法恢复其“总指挥”职责，按逻辑顺序调用本模块内所有公开的诊断引擎，
                       并汇总其产出的所有信号。
        - 收益: 确保了行为情报模块能够完整地生成所有必需的原子和合成信号，保障了整个情报系统的正常运行。
        """
        # print("      -> [行为情报模块总指挥 V2.1 职责修复版] 启动...")
        all_states = {} # 初始化一个空字典用于汇总所有信号
        params = self.strategy.params # 一次性获取策略参数，供各诊断引擎使用
        # 按照逻辑重要性调用所有诊断引擎
        # 1. 终极六维健康度信号 (最高层抽象)
        all_states.update(self.diagnose_ultimate_behavioral_signals(df)) # 保留：调用终极信号引擎
        # 2. 量价关系动态 (关键风险信号源)
        all_states.update(self.diagnose_volume_price_dynamics(df, params)) # 调用量价动态分析，解决 SCORE_RISK_VPA_STAGNATION 缺失问题
        # 3. 多维共振与反转 (筹码、资金流等跨维度行为)
        all_states.update(self.diagnose_multi_dimensional_resonance(df)) # 调用多维共振诊断
        # 4. 基础原子信号 (最底层的行为模式)
        all_states.update(self.diagnose_price_volume_atomics(df)) # 调用价格成交量原子诊断
        all_states.update(self.diagnose_advanced_atomic_signals(df)) # 调用高级原子信号诊断
        # 5. 静态K线与板块模式 (特定形态识别)
        all_states.update(self.diagnose_kline_patterns(df)) # 调用K线模式诊断
        all_states.update(self.diagnose_board_patterns(df)) # 调用板块模式诊断
        # 6. 特定离场风险行为诊断
        upthrust_score = self.diagnose_upthrust_distribution(df, params) # 调用高位派发诊断
        all_states[upthrust_score.name] = upthrust_score # 将其返回的Series结果添加到字典
        ma_breakdown_score = self.diagnose_ma_breakdown(params) # 调用均线破位诊断
        all_states[ma_breakdown_score.name] = ma_breakdown_score # 将其返回的Series结果添加到字典
        # print(f"      -> [行为情报模块总指挥 V2.1] 分析完毕，共生成 {len(all_states)} 个行为信号。")
        return all_states

    def diagnose_ultimate_behavioral_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 终极行为信号诊断模块】
        - 核心范式:
          - 1. 深度交叉验证: 对每一时间周期，都计算一个融合了“价位、价动能、价加速、量缩、量动能、量加速”六个维度的“完美健康度”分数。
          - 2. 信号组件化: 将不同周期的“完美健康度”组合成“短期力量”、“中期趋势”和“长期惯性”，使信号构建过程更透明。
          - 3. 精炼共振信号 (逻辑与ChipIntelligence V3.2对齐):
             - B级: 短期(5D)趋势健康。
             - A级: 短期(5D)与中期(21D)趋势共振健康。
             - S级: 短期力量(1D*5D)与中期趋势(13D*21D)共振健康。
             - S+级: S级信号得到长期惯性(55D)的确认，形成全周期共振。
          - 4. 精炼反转信号 (逻辑与ChipIntelligence V3.2对齐):
             - B级: 出现1日反转健康度，对抗21日中期不健康趋势。
             - A级: 形成5日反转健康度，对抗21日中期不健康趋势。
             - S级: 形成短期看涨合力(1D*5D)，对抗55日长期不健康惯性。
             - S+级: 形成短期看涨合力(1D*5D)，对抗中长期联合不健康惯性(21D*55D)。
        - 数据需求:
          - `price_vs_ma_{p}_D`, `volume_vs_ma_{p}_D`
          - `SLOPE_{p}_close_D`, `ACCEL_{p}_close_D`
          - `SLOPE_{p}_volume_D`, `ACCEL_{p}_volume_D`
        """
        # print("        -> [终极行为信号诊断模块 V1.0] 启动...")
        states = {}
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            return states
        # --- 1. 军备检查 (Arsenal Check) ---
        periods = get_param_value(p_conf.get('periods', [1, 5, 13, 21, 55]))
        required_cols = set()
        for p in periods:
            required_cols.update([
                f'price_vs_ma_{p}_D', f'volume_vs_ma_{p}_D',
                f'SLOPE_{p}_close_D', f'ACCEL_{p}_close_D',
                f'SLOPE_{p}_volume_D', f'ACCEL_{p}_volume_D'
            ])
        missing_cols = list(required_cols - set(df.columns))
        if missing_cols:
            print(f"          -> [严重警告] 终极行为引擎缺少关键数据: {sorted(missing_cols)}，模块已跳过！")
            return states
        # --- 2. 核心要素数值化 (归一化处理) ---
        norm_window = get_param_value(p_conf.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        price_static = {p: self._normalize_series(df.get(f'price_vs_ma_{p}_D'), norm_window, min_periods) for p in periods}
        price_mom = {p: self._normalize_series(df.get(f'SLOPE_{p}_close_D'), norm_window, min_periods) for p in periods}
        price_accel = {p: self._normalize_series(df.get(f'ACCEL_{p}_close_D'), norm_window, min_periods) for p in periods}
        vol_static = {p: self._normalize_series(df.get(f'volume_vs_ma_{p}_D'), norm_window, min_periods, ascending=False) for p in periods}
        vol_mom = {p: self._normalize_series(df.get(f'SLOPE_{p}_volume_D'), norm_window, min_periods) for p in periods}
        vol_accel = {p: self._normalize_series(df.get(f'ACCEL_{p}_volume_D'), norm_window, min_periods) for p in periods}
        # --- 3. 计算每个周期的“完美健康度” (Intra-Timeframe Validation) ---
        bullish_health = {}
        for p in periods:
            bullish_health[p] = (price_static[p] * price_mom[p] * price_accel[p] * vol_static[p] * vol_mom[p] * vol_accel[p])
        bearish_health = {p: 1.0 - bullish_health[p] for p in periods}
        # --- 4. 定义信号组件 ---
        bullish_short_force = (bullish_health[1] * bullish_health[5])**0.5
        bullish_medium_trend = (bullish_health[13] * bullish_health[21])**0.5
        bullish_long_inertia = bullish_health[55]
        bearish_short_force = (bearish_health[1] * bearish_health[5])**0.5
        bearish_medium_trend = (bearish_health[13] * bearish_health[21])**0.5
        bearish_long_inertia = bearish_health[55]
        # --- 5. 共振信号合成 ---
        states['SCORE_BEHAVIOR_BULLISH_RESONANCE_B'] = bullish_health[5].astype(np.float32)
        states['SCORE_BEHAVIOR_BULLISH_RESONANCE_A'] = (bullish_health[5] * bullish_health[21]).astype(np.float32)
        states['SCORE_BEHAVIOR_BULLISH_RESONANCE_S'] = (bullish_short_force * bullish_medium_trend).astype(np.float32)
        states['SCORE_BEHAVIOR_BULLISH_RESONANCE_S_PLUS'] = (states['SCORE_BEHAVIOR_BULLISH_RESONANCE_S'] * bullish_long_inertia).astype(np.float32)
        states['SCORE_BEHAVIOR_BEARISH_RESONANCE_B'] = bearish_health[5].astype(np.float32)
        states['SCORE_BEHAVIOR_BEARISH_RESONANCE_A'] = (bearish_health[5] * bearish_health[21]).astype(np.float32)
        states['SCORE_BEHAVIOR_BEARISH_RESONANCE_S'] = (bearish_short_force * bearish_medium_trend).astype(np.float32)
        states['SCORE_BEHAVIOR_BEARISH_RESONANCE_S_PLUS'] = (states['SCORE_BEHAVIOR_BEARISH_RESONANCE_S'] * bearish_long_inertia).astype(np.float32)
        # --- 6. 反转信号合成 ---
        states['SCORE_BEHAVIOR_BOTTOM_REVERSAL_B'] = (bullish_health[1] * bearish_health[21]).astype(np.float32)
        states['SCORE_BEHAVIOR_BOTTOM_REVERSAL_A'] = (bullish_health[5] * bearish_health[21]).astype(np.float32)
        states['SCORE_BEHAVIOR_BOTTOM_REVERSAL_S'] = (bullish_short_force * bearish_long_inertia).astype(np.float32)
        states['SCORE_BEHAVIOR_BOTTOM_REVERSAL_S_PLUS'] = (bullish_short_force * bearish_medium_trend * bearish_long_inertia).astype(np.float32)
        states['SCORE_BEHAVIOR_TOP_REVERSAL_B'] = (bearish_health[1] * bullish_health[21]).astype(np.float32)
        states['SCORE_BEHAVIOR_TOP_REVERSAL_A'] = (bearish_health[5] * bullish_health[21]).astype(np.float32)
        states['SCORE_BEHAVIOR_TOP_REVERSAL_S'] = (bearish_short_force * bullish_long_inertia).astype(np.float32)
        states['SCORE_BEHAVIOR_TOP_REVERSAL_S_PLUS'] = (bearish_short_force * bullish_medium_trend * bullish_long_inertia).astype(np.float32)
        # print(f"        -> [终极行为信号诊断模块 V1.0] 分析完毕，生成 {len(states)} 个终极信号。")
        return states

    def diagnose_kline_patterns(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V275.1 数值化修正版】
        - 核心重构 (本次修改):
          - [职责净化] 移除了对 `diagnose_behavioral_dynamics_scores` 等其他引擎的调用。
                        本方法现在只专注于其核心职责：诊断静态的、无法被动态分析取代的K线模式。
          - [数值化] 修正了 'SCORE_KLINE_SHARP_DROP' 信号的计算逻辑，消除了布尔过滤器。
        - 收益: 模块职责单一、清晰，且信号计算更平滑，符合完全数值化原则。
        """
        # print("        -> [K线模式诊断模块 V275.1 数值化修正版] 启动...")
        states = {}
        p = get_params_block(self.strategy, 'kline_pattern_params')
        if not get_param_value(p.get('enabled'), False): return states
        # --- “缺口支撑”持续状态 (Gap Support Active State) ---
        p_gap = p.get('gap_support_params', {})
        if get_param_value(p_gap.get('enabled'), True):
            persistence_days = get_param_value(p_gap.get('persistence_days'), 10)
            gap_up_event = df['low_D'] > df['high_D'].shift(1)
            gap_high = df['high_D'].shift(1).where(gap_up_event).ffill()
            price_fills_gap = df['close_D'] < gap_high
            gap_support_state = create_persistent_state(
                df=df,
                entry_event_series=gap_up_event,
                persistence_days=persistence_days,
                break_condition_series=price_fills_gap,
                state_name='KLINE_STATE_GAP_SUPPORT_ACTIVE'
            )
            support_distance = (df['low_D'] - gap_high).clip(lower=0)
            normalization_base = (df['close_D'] * 0.1).replace(0, np.nan)
            support_strength_score = (support_distance / normalization_base).clip(0, 1).fillna(0)
            states['SCORE_GAP_SUPPORT_ACTIVE'] = (support_strength_score * gap_support_state).astype(np.float32)
        # --- 基础原子行为，如“急速下跌” ---
        p_atomic = p.get('atomic_behavior_params', {})
        if get_param_value(p_atomic.get('enabled'), True):
            if 'pct_change_D' in df.columns:
                norm_window = get_param_value(p_atomic.get('norm_window'), 120)
                min_periods = max(1, norm_window // 5)
                # 新逻辑: 只对跌幅大小进行排名，使评分更平滑，且天然在上涨日为0
                drop_magnitude = df['pct_change_D'].where(df['pct_change_D'] < 0, 0).abs()
                sharp_drop_score = drop_magnitude.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.0)
                states['SCORE_KLINE_SHARP_DROP'] = sharp_drop_score.astype(np.float32)
        # print(f"        -> [K线模式诊断模块 V275.1] 分析完毕，共生成 {len(states)} 个静态模式信号。")
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
        # print(f"        -> [高级原子诊断模块 V1.1] 已生成 {len(states)} 个深层动态信号。") 
        return states

    def _diagnose_pullback_enhancement_matrix(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.7 信号源升级版】回踩形态增强矩阵
        - 收益: 使得对“压缩区洗盘”这一关键战术场景的判断，基于更可靠、经过交叉验证的信号源。
        """
        # print("        -> [回踩增强矩阵 V1.7 信号源升级版] 启动，正在扫描特殊形态...")
        enhancements = {}
        atomic = self.strategy.atomic_states
        default_score_series = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 增强器1: K线形态 (锤子线/探针) ---
        body = (df['close_D'] - df['open_D']).abs().replace(0, 0.0001)
        lower_shadow = (df[['open_D', 'close_D']].min(axis=1) - df['low_D']).clip(0)
        upper_shadow = (df['high_D'] - df[['open_D', 'close_D']].max(axis=1)).clip(0)
        lower_shadow_score = ((lower_shadow / body - 1.8) / 3).clip(0, 1).fillna(0)
        short_upper_shadow_score = (1 - (upper_shadow / (body * 0.8))).clip(0, 1).fillna(0)
        enhancements['SCORE_HAMMER_CANDLE_STRENGTH'] = (lower_shadow_score * short_upper_shadow_score).astype(np.float32)
        # --- 增强器2: 关键支撑位 (斐波那契) ---
        enhancements['SCORE_FIB_REBOUND_S'] = atomic.get('SCORE_FIB_REBOUND_S', default_score_series)
        enhancements['SCORE_FIB_REBOUND_A'] = atomic.get('SCORE_FIB_REBOUND_A', default_score_series)
        enhancements['SCORE_FIB_REBOUND_B'] = atomic.get('SCORE_FIB_REBOUND_B', default_score_series)
        # --- 增强器3: 特殊结构 (压缩区洗盘) ---
        # 将对已废弃信号的引用，升级为消费S级波动率压缩信号
        squeeze_level_score = atomic.get('SCORE_VOL_COMPRESSION_S', default_score_series)
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
        final_score.name = 'SCORE_RISK_UPTHRUST_DISTRIBUTION'
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
        【V284.2 终极数值化版】量价关系动态分析中心 (CT扫描室)
        - 核心职责: (原有注释)
        - 核心升级 (本次修改):
          - [数值化] 将所有风险信号中的布尔过滤器移除，改为对“风险事件的强度”直接进行排名评分。
        - 收益: 彻底消除了模块内的布尔判断，所有风险评分的响应都变得平滑且连续，信号质量更高。
        """
        # print("        -> [量价动态分析中心 V284.2 终极数值化版] 启动...")
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
        p_vpa = params.get('vpa_dynamics_params', {})
        norm_window = get_param_value(p_vpa.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        # --- 风险分析：识别“无效天量”和“效率衰竭” ---
        # 风险信号1: 【滞涨】天量但价格不涨 (逻辑不变，已是完全数值化)
        volume_ratio = (df['volume_D'] / df['VOL_MA_21_D'].replace(0, np.nan)).fillna(1.0)
        huge_volume_score = volume_ratio.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        price_stagnant_score = (1 - df['pct_change_D'].abs().rolling(window=norm_window, min_periods=min_periods).rank(pct=True)).fillna(0.5)
        states['SCORE_RISK_VPA_STAGNATION'] = (huge_volume_score * price_stagnant_score).astype(np.float32)

        # 新逻辑: 对“下降的幅度”进行排名，下降越剧烈，分数越高
        efficiency_decline_magnitude = df['SLOPE_5_VPA_EFFICIENCY_D'].where(df['SLOPE_5_VPA_EFFICIENCY_D'] < 0, 0).abs()
        efficiency_decline_score = efficiency_decline_magnitude.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.0)
        states['SCORE_RISK_VPA_EFFICIENCY_DECLINING'] = efficiency_decline_score.astype(np.float32)

        # 风险信号3: 【量能失控】成交量仍在加速放大
        # 原逻辑: rank(accel) * (accel > 0)
        # 新逻辑: 对“加速的幅度”进行排名，加速越剧烈，分数越高
        volume_accel_magnitude = df['ACCEL_5_volume_D'].where(df['ACCEL_5_volume_D'] > 0, 0)
        volume_accelerating_score = volume_accel_magnitude.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.0)
        states['SCORE_RISK_VPA_VOLUME_ACCELERATING'] = volume_accelerating_score.astype(np.float32)
        return states

    def diagnose_multi_dimensional_resonance(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.2 性能优化版】多维共振与反转诊断模块
        - 核心升级 (本次修改):
          - [性能优化] 在计算总共振信号 (SCORE_RESONANCE_UP_OVERALL_S) 时，
                      将原有的 `pd.concat().prod()` 方式，重构为使用 `np.array()` 和 `np.prod()`。
          - [信号增强] (V2.1逻辑保留) 在原有共振与反转信号的基础上，新增两类“背离”信号。
          - [滞涨背离] (V2.1逻辑保留) 新增 `SCORE_DIVERGENCE_STAGNATION`，捕捉“静态高位 vs. 动态衰竭”的背离。
          - [力竭背离] (V2.1逻辑保留) 新增 `SCORE_DIVERGENCE_EXHAUSTION`，捕捉“动态上升 vs. 加速放缓”的背离。
        - 收益:
          - 通过避免创建大型临时DataFrame，显著降低了内存峰值占用，并提升了计算速度，尤其在长周期回测中效果更佳。
          - 信号体系更加完备，不仅能识别趋势的“共振”，还能更灵敏地捕捉趋势由强转弱的“背离”迹象。
        """
        # print("        -> [多维共振诊断模块 V2.2 性能优化版] 启动...")
        states = {}
        p = get_params_block(self.strategy, 'resonance_params', {})
        if not get_param_value(p.get('enabled'), True):
            return states
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        metrics_to_analyze = [
            ('main_force_net_flow_consensus_D', [1, 5, 13, 21, 55], True),
            ('chip_health_score_D', [1, 5, 13, 21, 55], True),
            ('concentration_90pct_D', [1, 5, 13, 21, 55], False),
            ('flow_divergence_mf_vs_retail_D', [1, 5, 13, 21, 55], True),
        ]
        per_metric_scores = {
            'static': {}, 'slope': {}, 'accel': {},
            'up_resonance': {}, 'down_resonance': {}
        }
        # --- 1. 计算所有指标、所有周期的基础得分和"周期内共振"分 ---
        for base_name, periods, is_positive_metric in metrics_to_analyze:
            for score_type in per_metric_scores:
                if base_name not in per_metric_scores[score_type]:
                    per_metric_scores[score_type][base_name] = {}
            for period in periods:
                static_col = base_name
                slope_col = f'SLOPE_{period}_{base_name}'
                accel_col = f'ACCEL_{period}_{base_name}'
                required_cols = [static_col, slope_col, accel_col]
                if not all(c in df.columns for c in required_cols):
                    # print(f"        -> [多维共振诊断] 警告: 缺少 '{base_name}' 周期 {period} 的列: {required_cols}，跳过。")
                    continue
                static_score = self._normalize_series(df[static_col], norm_window, min_periods)
                slope_score = self._normalize_series(df[slope_col], norm_window, min_periods)
                accel_score = self._normalize_series(df[accel_col], norm_window, min_periods)
                if not is_positive_metric:
                    static_score, slope_score, accel_score = 1 - static_score, 1 - slope_score, 1 - accel_score
                per_metric_scores['static'][base_name][period] = static_score
                per_metric_scores['slope'][base_name][period] = slope_score
                per_metric_scores['accel'][base_name][period] = accel_score
                up_resonance = static_score * slope_score * accel_score
                down_resonance = (1 - static_score) * (1 - slope_score) * (1 - accel_score)
                per_metric_scores['up_resonance'][base_name][period] = up_resonance
                per_metric_scores['down_resonance'][base_name][period] = down_resonance
        # --- 2. 合成各指标的 S/A/B 级共振、精准反转与新增的背离信号 ---
        all_s_level_up_scores = []
        all_s_level_down_scores = []
        for base_name, periods, _ in metrics_to_analyze:
            metric_key = base_name.replace("_D", "")
            up_res = per_metric_scores['up_resonance'][base_name]
            down_res = per_metric_scores['down_resonance'][base_name]
            # --- 合成上升/下跌共振 S/A/B 信号 ---
            if 21 in up_res:
                states[f'SCORE_RESONANCE_UP_{metric_key}_B'] = up_res[21].astype(np.float32)
                if 55 in up_res:
                    states[f'SCORE_RESONANCE_UP_{metric_key}_A'] = (up_res[21] * up_res[55]).astype(np.float32)
                    if 5 in up_res:
                        s_score = (up_res[5] * up_res[21] * up_res[55])
                        states[f'SCORE_RESONANCE_UP_{metric_key}_S'] = s_score.astype(np.float32)
                        all_s_level_up_scores.append(s_score)
            if 21 in down_res:
                states[f'SCORE_RESONANCE_DOWN_{metric_key}_B'] = down_res[21].astype(np.float32)
                if 55 in down_res:
                    states[f'SCORE_RESONANCE_DOWN_{metric_key}_A'] = (down_res[21] * down_res[55]).astype(np.float32)
                    if 5 in down_res:
                        s_score = (down_res[5] * down_res[21] * down_res[55])
                        states[f'SCORE_RESONANCE_DOWN_{metric_key}_S'] = s_score.astype(np.float32)
                        all_s_level_down_scores.append(s_score)
            # --- 合成精准反转信号 ---
            static_scores = per_metric_scores['static'][base_name]
            slope_scores = per_metric_scores['slope'][base_name]
            if 55 in static_scores and 5 in slope_scores:
                states[f'SCORE_REVERSAL_TOP_{metric_key}'] = (static_scores[55] * (1 - slope_scores[5])).astype(np.float32)
                states[f'SCORE_REVERSAL_BOTTOM_{metric_key}'] = ((1 - static_scores[55]) * slope_scores[5]).astype(np.float32)
            # --- 合成周期内背离信号 ---
            accel_scores = per_metric_scores['accel'][base_name]
            for period in periods:
                if period in static_scores and period in slope_scores and period in accel_scores:
                    stagnation_divergence = static_scores[period] * (1 - slope_scores[period])
                    states[f'SCORE_DIVERGENCE_STAGNATION_{metric_key}_{period}D'] = stagnation_divergence.astype(np.float32)
                    exhaustion_divergence = slope_scores[period] * (1 - accel_scores[period])
                    states[f'SCORE_DIVERGENCE_EXHAUSTION_{metric_key}_{period}D'] = exhaustion_divergence.astype(np.float32)
        # --- 3. 合成总共振信号 ---
        if all_s_level_up_scores:
            num_metrics = len(all_s_level_up_scores)
            # 1. 将Series列表转换为一个2D NumPy数组，形状为 (指标数量, 时间序列长度)
            stacked_scores = np.array([s.values for s in all_s_level_up_scores])
            # 2. 沿第一个轴（指标维度）计算所有分数的乘积
            prod_scores = np.prod(stacked_scores, axis=0)
            # 3. 计算几何平均数，并重新构建为带索引的Pandas Series
            overall_up_s = pd.Series(prod_scores**(1/num_metrics), index=df.index, dtype=np.float32)
            states['SCORE_RESONANCE_UP_OVERALL_S'] = overall_up_s
        if all_s_level_down_scores:
            num_metrics = len(all_s_level_down_scores)
            stacked_scores = np.array([s.values for s in all_s_level_down_scores])
            prod_scores = np.prod(stacked_scores, axis=0)
            overall_down_s = pd.Series(prod_scores**(1/num_metrics), index=df.index, dtype=np.float32)
            states['SCORE_RESONANCE_DOWN_OVERALL_S'] = overall_down_s
        # print(f"        -> [多维共振诊断模块 V2.2] 已生成 {len(states)} 个共振、反转与背离信号。")
        return states

    def diagnose_price_volume_atomics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.3 终极数值化版】价格与成交量基础原子信号诊断模块
        - 核心职责: (原有注释)
        - 核心升级 (本次修改):
          - [数值化] 对 'SCORE_VOL_WEAKENING_DROP' 的计算逻辑进行终极改造，
                      通过对“跌幅强度”直接排名来代替“通用排名+布尔过滤”的模式。
        - 收益: 使得“价跌量缩”的评分更加平滑和准确，彻底消除了布尔判断，
                完美符合数值化和平滑响应的原则。
        """
        # print("        -> [价格成交量原子诊断模块 V1.3 终极数值化版] 启动...")
        states = {}
        p = get_params_block(self.strategy, 'price_volume_atomic_params')
        if not get_param_value(p.get('enabled'), True): return states
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
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
            # 组件1: 价跌幅度分 (仅对下跌日进行评分，跌幅越大，分数越高)
            # 新逻辑: 对“跌幅大小”直接排名，更平滑且内含了“下跌日”的判断
            drop_magnitude = df['pct_change_D'].where(df['pct_change_D'] < 0, 0).abs()
            price_drop_score = drop_magnitude.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.0)
            # 组件2: 量缩程度分 (成交量相对均线越小，分数越高)
            volume_ratio = (df['volume_D'] / df[vol_ma_col].replace(0, np.nan)).fillna(1.0)
            volume_shrink_score = (1 - volume_ratio.rolling(window=norm_window, min_periods=min_periods).rank(pct=True)).fillna(0.5)
            # 融合生成最终分数，新版price_drop_score已内置“下跌日”判断，无需布尔过滤
            states['SCORE_VOL_WEAKENING_DROP'] = (price_drop_score * volume_shrink_score).astype(np.float32)
        # print(f"        -> [价格成交量原子诊断模块 V1.3] 已生成 {len(states)} 个数值化基础原子信号。")
        return states







