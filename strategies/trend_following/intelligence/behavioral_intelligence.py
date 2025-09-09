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
        - 新增：提升为类的私有方法，以供所有诊断引擎复用。
        """
        rank = series.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        return rank if ascending else 1 - rank

    def diagnose_behavioral_dynamics_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V4.0 完美健康度版】行为动态诊断核心引擎
        - 核心重构 (本次修改):
          - [范式升级] 引入“周期内完美健康度”范式，对每个周期的6个维度(价/量 x 静/动/加)进行完整交叉验证。
          - [信号升级] 新增S+级信号，定义为“全周期健康共振(S) * 全周期加速共振”，用于捕捉最强的主升浪“完美风暴”。
        - 收益: 信号体系的逻辑严谨性达到顶峰，S+信号能以极高置信度识别市场合力最强的爆发点。
        - 数据需求说明: 本方法假设数据层已提供以下列:
          - `price_vs_ma_{p}_D`, `volume_vs_ma_{p}_D` (p in [1, 5, 13, 21, 55])
          - `SLOPE_{p}_close_D`, `ACCEL_{p}_close_D` (p in [1, 5, 13, 21, 55])
          - `SLOPE_{p}_volume_D`, `ACCEL_{p}_volume_D` (p in [1, 5, 13, 21, 55])
        """
        print("        -> [行为动态诊断核心引擎 V4.0 完美健康度版] 启动...")
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
            print(f"          -> [严重警告] 行为动态引擎缺少关键数据: {sorted(missing_cols)}，模块已跳过！")
            return states

        # --- 2. 核心要素数值化 (归一化处理) ---
        norm_window = get_param_value(p_conf.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        
        # 对每个周期的所有6个维度数据进行归一化
        price_static = {p: self._normalize_series(df[f'price_vs_ma_{p}_D'], norm_window, min_periods) for p in periods}
        price_mom = {p: self._normalize_series(df[f'SLOPE_{p}_close_D'], norm_window, min_periods) for p in periods}
        price_accel = {p: self._normalize_series(df[f'ACCEL_{p}_close_D'], norm_window, min_periods) for p in periods}
        vol_static = {p: self._normalize_series(df[f'volume_vs_ma_{p}_D'], norm_window, min_periods, ascending=False) for p in periods} # 量缩为佳，故反向
        vol_mom = {p: self._normalize_series(df[f'SLOPE_{p}_volume_D'], norm_window, min_periods) for p in periods}
        vol_accel = {p: self._normalize_series(df[f'ACCEL_{p}_volume_D'], norm_window, min_periods) for p in periods}

        # --- 3. 计算每个周期的“完美健康度” ---
        bullish_health = {}
        bearish_health = {}
        for p in periods:
            # 看涨完美健康度 = 价位合理 * 价升 * 价加速 * 量缩 * 量增趋势 * 量加速
            bullish_health[p] = price_static[p] * price_mom[p] * price_accel[p] * vol_static[p] * vol_mom[p] * vol_accel[p]
            # 看跌完美健康度 = 价位危险 * 价跌 * 价减速 * 放量 * 量增趋势 * 量加速
            bearish_health[p] = (1 - price_static[p]) * (1 - price_mom[p]) * (1 - price_accel[p]) * (1 - vol_static[p]) * vol_mom[p] * vol_accel[p]

        # --- 4. 共振信号合成 (跨周期融合) ---
        # 4.1 看涨行为共振 (多周期健康度共振)
        states['SCORE_BEHAVIOR_BULLISH_RESONANCE_B'] = (bullish_health[5] * bullish_health[21]).astype(np.float32)
        states['SCORE_BEHAVIOR_BULLISH_RESONANCE_A'] = (states['SCORE_BEHAVIOR_BULLISH_RESONANCE_B'] * bullish_health[55]).astype(np.float32)
        s_score_bullish = pd.Series(np.prod(np.array([s.values for s in bullish_health.values()]), axis=0), index=df.index)
        states['SCORE_BEHAVIOR_BULLISH_RESONANCE_S'] = s_score_bullish.astype(np.float32)
        
        # S+级信号: 完美风暴 (全周期健康 * 全周期加速)
        acceleration_resonance = pd.Series(np.prod(np.array([price_accel[p].values for p in periods]), axis=0), index=df.index)
        states['SCORE_BEHAVIOR_BULLISH_RESONANCE_S_PLUS'] = (s_score_bullish * acceleration_resonance).astype(np.float32)

        # 4.2 看跌行为共振 (多周期不健康度共振)
        states['SCORE_BEHAVIOR_BEARISH_RESONANCE_B'] = (bearish_health[5] * bearish_health[21]).astype(np.float32)
        states['SCORE_BEHAVIOR_BEARISH_RESONANCE_A'] = (states['SCORE_BEHAVIOR_BEARISH_RESONANCE_B'] * bearish_health[55]).astype(np.float32)
        s_score_bearish = pd.Series(np.prod(np.array([s.values for s in bearish_health.values()]), axis=0), index=df.index)
        states['SCORE_BEHAVIOR_BEARISH_RESONANCE_S'] = s_score_bearish.astype(np.float32)
        
        # S+级信号: 完美崩溃 (全周期不健康 * 全周期减速)
        deceleration_resonance = pd.Series(np.prod(np.array([(1-price_accel[p].values) for p in periods]), axis=0), index=df.index)
        states['SCORE_BEHAVIOR_BEARISH_RESONANCE_S_PLUS'] = (s_score_bearish * deceleration_resonance).astype(np.float32)

        # --- 5. 反转信号合成 (趋势 x 反向加速度) ---
        # 5.1 底部反转 (下跌趋势中出现看涨加速度)
        avg_bearish_health = pd.Series(np.mean([s.values for s in bearish_health.values()], axis=0), index=df.index)
        avg_price_accel_bull = pd.Series(np.mean([price_accel[p].values for p in periods], axis=0), index=df.index)
        
        states['SCORE_BEHAVIOR_BOTTOM_REVERSAL_B'] = avg_price_accel_bull.astype(np.float32)
        states['SCORE_BEHAVIOR_BOTTOM_REVERSAL_A'] = (avg_bearish_health * avg_price_accel_bull).astype(np.float32)
        states['SCORE_BEHAVIOR_BOTTOM_REVERSAL_S'] = (states['SCORE_BEHAVIOR_BOTTOM_REVERSAL_A'] * bullish_health[1]).astype(np.float32)

        # 5.2 顶部反转 (上涨趋势中出现看跌加速度)
        avg_bullish_health = pd.Series(np.mean([s.values for s in bullish_health.values()], axis=0), index=df.index)
        avg_price_accel_bear = 1 - avg_price_accel_bull

        states['SCORE_BEHAVIOR_TOP_REVERSAL_B'] = avg_price_accel_bear.astype(np.float32)
        states['SCORE_BEHAVIOR_TOP_REVERSAL_A'] = (avg_bullish_health * avg_price_accel_bear).astype(np.float32)
        states['SCORE_BEHAVIOR_TOP_REVERSAL_S'] = (states['SCORE_BEHAVIOR_TOP_REVERSAL_A'] * bearish_health[1]).astype(np.float32)

        print(f"        -> [行为动态诊断核心引擎 V4.0] 分析完毕，生成 {len(states)} 个B/A/S/S+信号。")
        return states

    def run_behavioral_analysis_command(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】行为情报模块总指挥
        - 核心职责: 作为本模块唯一的公共入口，负责编排和调用所有内部的诊断与合成方法。
        - 收益: 实现了高度的封装，上层模块无需关心内部实现细节，只需调用此方法即可获取所有行为信号。
        """
        # -> [新增] 此为全新的模块总指挥方法
        print("      -> [行为情报模块总指挥] 启动...")
        all_behavioral_states = {}

        # 1. 调用核心动态引擎 (V4.0)
        all_behavioral_states.update(self.diagnose_behavioral_dynamics_scores(df))
        
        # 2. 调用所有独立的原子/模式诊断模块
        all_behavioral_states.update(self.diagnose_kline_patterns(df))
        all_behavioral_states.update(self.diagnose_board_patterns(df))
        all_behavioral_states.update(self.diagnose_price_volume_atomics(df))
        all_behavioral_states.update(self.diagnose_advanced_atomic_signals(df))
        all_behavioral_states.update(self.diagnose_multi_dimensional_resonance(df))

        # 3. 调用需要特定参数的风险诊断模块
        behavioral_params = get_params_block(self.strategy, 'behavioral_params', {})
        all_behavioral_states.update(self.diagnose_volume_price_dynamics(df, behavioral_params))
        
        exit_params = get_params_block(self.strategy, 'exit_strategy_params', {})
        upthrust_risk_score = self.diagnose_upthrust_distribution(df, exit_params)
        all_behavioral_states[upthrust_risk_score.name] = upthrust_risk_score

        print(f"      -> [行为情报模块总指挥] 分析完毕，共生成 {len(all_behavioral_states)} 个行为信号。")
        return all_behavioral_states

    def diagnose_kline_patterns(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V275.0 职责净化版】
        - 核心重构 (本次修改):
          - [职责净化] 移除了对 `diagnose_behavioral_dynamics_scores` 等其他引擎的调用。
                        本方法现在只专注于其核心职责：诊断静态的、无法被动态分析取代的K线模式。
        - 收益: 模块职责单一、清晰，符合高内聚原则。
        """
        # -> [修改] 更新打印信息
        print("        -> [K线模式诊断模块 V275.0 职责净化版] 启动...")
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
                sharp_drop_score = (1 - df['pct_change_D'].rolling(window=norm_window, min_periods=min_periods).rank(pct=True)).fillna(0.5)
                is_negative_change = df['pct_change_D'] < 0
                states['SCORE_KLINE_SHARP_DROP'] = (sharp_drop_score * is_negative_change).astype(np.float32)
        
        # -> [修改] 更新打印信息
        print(f"        -> [K线模式诊断模块 V275.0] 分析完毕，共生成 {len(states)} 个静态模式信号。")
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
        【V1.7 信号源升级版】回踩形态增强矩阵
        - 收益: 使得对“压缩区洗盘”这一关键战术场景的判断，基于更可靠、经过交叉验证的信号源。
        """
        print("        -> [回踩增强矩阵 V1.7 信号源升级版] 启动，正在扫描特殊形态...")
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

    def diagnose_multi_dimensional_resonance(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.2 逻辑修正版】多维共振与反转诊断模块
        - 核心职责: (同V1.0)
        - 核心升级 (本次修改):
          - [修复] 移除了上一版中一行多余的代码，该代码错误地覆盖了对 'flow_divergence_mf_vs_retail_D' 特殊命名规则的处理逻辑。
        - 收益: 彻底解决了因列名不匹配导致部分指标共振分析被跳过的问题。
        """
        states = {}
        p = get_params_block(self.strategy, 'resonance_params', {})
        if not get_param_value(p.get('enabled'), True):
            return states
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        # 定义需要进行共振分析的核心指标
        # 格式: (指标基础名, [周期列表], 是否为正向指标)
        metrics_to_analyze = [
            ('main_force_net_flow_consensus_D', [5, 21], True), # 主力资金共识, 正向
            ('chip_health_score_D', [5, 21], True),             # 筹码健康度, 正向
            ('concentration_90pct_D', [5, 21], True),           # 筹码集中度, 正向
            ('flow_divergence_mf_vs_retail_D', [5, 21], True),  # 主力散户流向背离, 正向
        ]
        all_up_resonance_scores = []
        all_down_resonance_scores = []
        for base_name, periods, is_positive_metric in metrics_to_analyze:
            for period in periods:
                # 构造列名
                static_col = base_name
                slope_col = f'SLOPE_{period}_{base_name}'
                # 针对 'flow_divergence_mf_vs_retail_D' 指标的特殊命名规则进行适配
                if 'flow_divergence_mf_vs_retail' in base_name:
                    accel_col = f'accel_{period}d_{base_name}' # 使用 'accel_5d_...' 格式
                else:
                    accel_col = f'ACCEL_{period}_{base_name}' # 使用标准的 'ACCEL_5_...' 格式
                # accel_col = f'ACCEL_{period}_{base_name}' # 修改：删除此行多余的错误代码
                required_cols = [static_col, slope_col, accel_col]
                if not all(c in df.columns for c in required_cols):
                    print(f"        -> [多维共振诊断] 警告: 缺少分析 '{base_name}' 所需列: {required_cols}，跳过周期 {period}。")
                    continue
                # --- 1. 信号数值化与归一化 (0-1分) ---
                static_score = df[static_col].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
                slope_score = df[slope_col].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
                accel_score = df[accel_col].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
                if not is_positive_metric: # 如果是负向指标（值越小越好），则反转评分
                    static_score = 1 - static_score
                    slope_score = 1 - slope_score
                    accel_score = 1 - accel_score
                # --- 2. 生成共振信号 ---
                up_resonance = static_score * slope_score * accel_score
                states[f'SCORE_RESONANCE_UP_{base_name.replace("_D", "")}_{period}D'] = up_resonance.astype(np.float32)
                all_up_resonance_scores.append(up_resonance)
                down_resonance = (1 - static_score) * (1 - slope_score) * (1 - accel_score)
                states[f'SCORE_RESONANCE_DOWN_{base_name.replace("_D", "")}_{period}D'] = down_resonance.astype(np.float32)
                all_down_resonance_scores.append(down_resonance)
                # --- 3. 生成反转信号 (基于静态与斜率背离) ---
                top_reversal = static_score * (1 - slope_score) # 顶部反转 = 静态值高 * 斜率低 (高位滞涨或转向)
                states[f'SCORE_REVERSAL_TOP_{base_name.replace("_D", "")}_{period}D'] = top_reversal.astype(np.float32)
                bottom_reversal = (1 - static_score) * slope_score # 底部反转 = 静态值低 * 斜率高 (低位企稳或转向)
                states[f'SCORE_REVERSAL_BOTTOM_{base_name.replace("_D", "")}_{period}D'] = bottom_reversal.astype(np.float32)
        # --- 4. 合成总共振信号 ---
        if all_up_resonance_scores:
            states['SCORE_RESONANCE_UP_OVERALL'] = pd.concat(all_up_resonance_scores, axis=1).mean(axis=1).astype(np.float32)
        if all_down_resonance_scores:
            states['SCORE_RESONANCE_DOWN_OVERALL'] = pd.concat(all_down_resonance_scores, axis=1).mean(axis=1).astype(np.float32)
        print(f"        -> [多维共振诊断模块 V1.2] 已生成 {len(states)} 个共振与反转信号。")
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
        print(f"        -> [价格成交量原子诊断模块 V1.2] 已生成 {len(states)} 个数值化基础原子信号。")
        return states







