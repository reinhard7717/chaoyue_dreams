# 文件: strategies/trend_following/intelligence/behavioral_intelligence.py
# 行为与模式识别模块
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from strategies.trend_following.intelligence.tactic_engine import TacticEngine
from strategies.trend_following.utils import transmute_health_to_ultimate_signals, get_params_block, get_param_value, create_persistent_state, normalize_score, normalize_to_bipolar, calculate_holographic_dynamics

class BehavioralIntelligence:
    def __init__(self, strategy_instance):
        """
        初始化行为与模式识别模块。
        :param strategy_instance: 策略主实例的引用。
        """
        self.strategy = strategy_instance
        # K线形态识别器可能需要在这里初始化或传入
        self.pattern_recognizer = strategy_instance.pattern_recognizer
        self.tactic_engine = TacticEngine(strategy_instance)

    def run_behavioral_analysis_command(self) -> Dict[str, pd.Series]:
        """
        【V3.4 · 职责净化版】行为情报模块总指挥
        - 核心重构: 移除了对 _diagnose_archangel_top_reversal 的调用。
                      “天使长”作为顶层认知信号，其诊断职责已正式移交认知情报模块。
        """
        df = self.strategy.df_indicators
        all_behavioral_states = {}
        internal_atomic_signals = self._generate_all_atomic_signals(df)
        if internal_atomic_signals:
            self.strategy.atomic_states.update(internal_atomic_signals)
            all_behavioral_states.update(internal_atomic_signals)
        ultimate_behavioral_states = self.diagnose_ultimate_behavioral_signals(df, atomic_signals=internal_atomic_signals)
        if ultimate_behavioral_states:
            all_behavioral_states.update(ultimate_behavioral_states)
        return all_behavioral_states

    def diagnose_ultimate_behavioral_signals(self, df: pd.DataFrame, atomic_signals: Dict[str, pd.Series] = None) -> Dict[str, pd.Series]:
        """
        【V26.1 · 赫尔墨斯之翼优化版】行为终极信号诊断引擎
        - 性能优化: 采用Numpy向量化操作进行多维健康度融合，替换了原有的Pandas Series操作，
                      显著提升了计算效率并降低了内存开销。
        - 核心逻辑: 保持对“宙斯之雷”引擎的调用和三维健康度的几何平均融合逻辑不变。
        - Bug修复: 修复了因变量名拼写错误 (`stacked_scores` 应为 `stacked_values`) 导致的 `NameError`。
        - 优化说明: 增加了权重的自动归一化处理，使融合算法对配置变化更具鲁棒性。
        """
        if atomic_signals is None:
            atomic_signals = self._generate_all_atomic_signals(df)
        states = {}
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        if not get_param_value(p_conf.get('enabled'), True): return states
        
        # 步骤一：调用“宙斯之雷”引擎获取权威的底部形态信号
        p_fusion = get_param_value(p_conf.get('supreme_fusion_params'), {})
        bottom_formation_score = self._supreme_fusion_engine(
            df=df,
            signals_to_fuse=atomic_signals,
            params=p_fusion
        )
        self.strategy.atomic_states['SCORE_UNIVERSAL_BOTTOM_PATTERN'] = bottom_formation_score
        
        # 步骤二：计算并存储近期反转上下文
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        reversal_echo_window = get_param_value(p_conf.get('reversal_echo_window'), 3)
        recent_reversal_context = bottom_formation_score.rolling(window=reversal_echo_window, min_periods=1).max()
        self.strategy.atomic_states['SCORE_CONTEXT_RECENT_REVERSAL'] = recent_reversal_context.astype(np.float32)
        
        # 步骤三：计算价格、成交量、K线形态三个维度的健康度
        periods = get_param_value(p_synthesis.get('periods'), [1, 5, 13, 21, 55])
        norm_window = get_param_value(p_synthesis.get('norm_window'), 55)
        min_periods = max(1, norm_window // 5)
        price_s_bull, price_s_bear, price_d_intensity = self._calculate_price_health(df, norm_window, min_periods, periods)
        vol_s_bull, vol_s_bear, vol_d_intensity = self._calculate_volume_health(df, norm_window, min_periods, periods)
        kline_s_bull, kline_s_bear, kline_d_intensity = self._calculate_kline_pattern_health(df, atomic_signals, norm_window, min_periods, periods)
        
        # 步骤四：使用Numpy进行高效的多维健康度融合
        overall_health = {}
        pillar_weights = get_param_value(p_conf.get('pillar_weights'), {'price': 0.4, 'volume': 0.3, 'kline': 0.3})
        
        # 新增(健壮性): 将权重数组的计算提前，并进行归一化，确保总权重为1
        dim_weights_array = np.array([pillar_weights['price'], pillar_weights['volume'], pillar_weights['kline']])
        total_weight = dim_weights_array.sum()
        if total_weight > 0:
            dim_weights_array /= total_weight
        else: # 如果权重总和为0或配置错误，则使用等权重
            dim_weights_array.fill(1.0 / len(dim_weights_array))
        
        for health_type, health_sources in [
            ('s_bull', [price_s_bull, vol_s_bull, kline_s_bull]),
            ('s_bear', [price_s_bear, vol_s_bear, kline_s_bear]),
            ('d_intensity', [price_d_intensity, vol_d_intensity, kline_d_intensity])
        ]:
            overall_health[health_type] = {}
            for p in periods:
                # 新增(注释): 使用Numpy进行向量化融合，避免Pandas Series操作
                # 将三个维度的健康度Series的底层Numpy数组堆叠起来
                stacked_values = np.stack([
                    health_sources[0][p].values, 
                    health_sources[1][p].values, 
                    health_sources[2][p].values
                ], axis=0)
                
                # 新增(注释): 使用Numpy的广播和乘方运算，一次性完成加权几何平均
                # 公式: G = (s1^w1 * s2^w2 * ... * sn^wn)
                # 修改(Bug修复): 将变量名 `stacked_scores` 修正为 `stacked_values`
                fused_values = np.prod(stacked_values ** dim_weights_array[:, np.newaxis], axis=0)
                overall_health[health_type][p] = pd.Series(fused_values, index=df.index, dtype=np.float32)
        
        self.strategy.atomic_states['__BEHAVIOR_overall_health'] = overall_health
        
        # 步骤五：调用终极信号合成引擎
        ultimate_signals = transmute_health_to_ultimate_signals(
            df=df,
            atomic_states=self.strategy.atomic_states,
            overall_health=overall_health,
            params=p_synthesis,
            domain_prefix="BEHAVIOR"
        )
        states.update(ultimate_signals)
        return states

    # ==============================================================================
    # 以下为新增的原子信号中心和降级的原子诊断引擎
    # ==============================================================================

    def _generate_all_atomic_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """【V1.3 · 双引擎版】原子信号中心，负责生产所有基础行为信号。"""
        atomic_signals = {}
        params = self.strategy.params
        
        atomic_signals.update(self._diagnose_atomic_bottom_formation(df))
        
        # 架构升级：同时运行两个独立的、针对不同场景的反转引擎
        epic_reversal_states = self._diagnose_atomic_rebound_reversal(df)
        continuation_reversal_states = self._diagnose_atomic_continuation_reversal(df)
        
        epic_score = epic_reversal_states.get('SCORE_ATOMIC_REBOUND_REVERSAL', pd.Series(0.0, index=df.index))
        continuation_score = continuation_reversal_states.get('SCORE_ATOMIC_CONTINUATION_REVERSAL', pd.Series(0.0, index=df.index))
        
        # 最终的探底回升信号是两种反转模式中的最强者
        final_rebound_score = np.maximum(epic_score, continuation_score)
        atomic_signals['SCORE_ATOMIC_REBOUND_REVERSAL'] = final_rebound_score.astype(np.float32)
        # 将延续性反转的独立分数也存入，以供调试
        atomic_signals.update(continuation_reversal_states)
        
        atomic_signals.update(self._diagnose_kline_patterns(df))
        atomic_signals.update(self._diagnose_advanced_atomic_signals(df))
        atomic_signals.update(self._diagnose_board_patterns(df))
        atomic_signals.update(self._diagnose_price_volume_atomics(df))
        atomic_signals.update(self._diagnose_volume_price_dynamics(df, params))
        
        upthrust_score = self._diagnose_upthrust_distribution(df, params)
        atomic_signals[upthrust_score.name] = upthrust_score
        
        return atomic_signals

    def _calculate_price_health(self, df: pd.DataFrame, norm_window: int, min_periods: int, periods: list) -> tuple:
        """
        【V4.1 · 赫尔墨斯之翼优化版】计算价格维度的三维健康度
        - 性能优化: 将不依赖于周期`p`的静态分(s_bull, s_bear)和动态分(d_intensity)的计算完全移出循环，
                      避免了(len(periods) - 1)次重复计算，大幅提升效率。
        - 核心逻辑: 保持原有的全息动态计算模式不变。
        """
        s_bull, s_bear, d_intensity = {}, {}, {}
        
        # --- 将所有与周期p无关的计算提前至循环外 ---
        
        # 1. 计算静态看涨分 (s_bull)
        bbp = df.get('BBP_21_2.0_D', pd.Series(0.5, index=df.index)).fillna(0.5).clip(0, 1)
        day_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        close_position_in_range = ((df['close_D'] - df['low_D']) / day_range).clip(0, 1).fillna(0.5)

        is_positive_day = df['pct_change_D'] > 0
        # 上涨日的看涨分：布林百分比与日内收盘位置的几何平均
        bullish_score_on_up_day = (bbp * close_position_in_range)**0.5
        # 下跌日的看涨分(潜在反转)：仅考虑日内收盘位置，收盘越高，反转潜力越大
        reversal_potential_score = close_position_in_range.where(df['pct_change_D'] < 0, 0)
        # 融合得到最终的静态看涨分
        static_bull_score = pd.Series(
            np.where(is_positive_day, bullish_score_on_up_day, reversal_potential_score),
            index=df.index,
            dtype=np.float32 # 直接指定数据类型，减少内存占用
        )

        # 2. 计算静态看跌分 (s_bear)
        # 看跌分：(1-布林百分比) 与 (1-日内收盘位置) 的几何平均，代表收盘弱势
        static_bear_score = ((1.0 - bbp) * (1.0 - close_position_in_range))**0.5
        static_bear_score = static_bear_score.astype(np.float32) # 指定数据类型

        # 3. 计算统一的动态强度分 (d_intensity)
        # 使用全息动态引擎计算价格的看涨和看跌动能
        price_holo_bull, price_holo_bear = calculate_holographic_dynamics(df, 'close_D', norm_window)
        # 统一的动态强度分是两种动能的平均值，反映价格变化的活跃程度
        unified_d_intensity = ((price_holo_bull + price_holo_bear) / 2.0).astype(np.float32)

        # --- 循环内仅进行高效的字典赋值操作 ---
        for p in periods:
            s_bull[p] = static_bull_score
            s_bear[p] = static_bear_score
            # 所有周期共享同一个、更高级的动态强度分
            d_intensity[p] = unified_d_intensity
            
        return s_bull, s_bear, d_intensity

    def _calculate_volume_health(self, df: pd.DataFrame, norm_window: int, min_periods: int, periods: list) -> tuple:
        """
        【V7.1 · 赫尔墨斯之翼优化版】计算成交量维度的三维健康度
        - 性能优化: 将不依赖于周期`p`的静态分(s_bull, s_bear)和动态分(d_intensity)的计算完全移出循环，
                      避免了(len(periods) - 1)次重复计算，极大提升了执行效率。
        - 核心逻辑: 保持原有的全息动态计算模式不变。
        """
        s_bull, s_bear, d_intensity = {}, {}, {}

        if 'pct_change_D' not in df.columns or 'volume_D' not in df.columns:
            # 如果缺少关键列，快速返回默认值
            default_series = pd.Series(0.5, index=df.index, dtype=np.float32)
            for p in periods:
                s_bull[p] = s_bear[p] = d_intensity[p] = default_series
            return s_bull, s_bear, d_intensity

        # --- 将所有与周期p无关的计算提前至循环外 ---

        # 1. 计算静态看跌分 (s_bear)
        volume_increase_score = normalize_score(df['volume_D'], df.index, norm_window, ascending=True)
        price_stagnation_score = 1 - normalize_score(df['pct_change_D'].abs(), df.index, norm_window, ascending=True)
        # 路径1: 放量滞涨
        stagnation_path_score = volume_increase_score * price_stagnation_score
        price_drop_score = normalize_score(df['pct_change_D'].clip(upper=0).abs(), df.index, norm_window, ascending=True)
        # 路径2: 放量下跌
        breakdown_path_score = (price_drop_score * volume_increase_score).where(df['pct_change_D'] < 0, 0)
        # 看跌分取两种路径中的更强者
        static_bear_score = np.maximum(stagnation_path_score, breakdown_path_score).astype(np.float32)
        
        # 2. 计算静态看涨分 (s_bull)
        price_increase_score = normalize_score(df['pct_change_D'].clip(lower=0), df.index, norm_window, ascending=True)
        # 上涨日的看涨分：价涨量增（健康的阳）
        yang_score = (price_increase_score * volume_increase_score)
        
        liquidity_drain_risk = self.strategy.atomic_states.get('SCORE_RISK_LIQUIDITY_DRAIN', pd.Series(0.0, index=df.index))
        # 下跌日的看涨分基础：价跌但未放量破位（健康的阴）
        yin_score = ((1.0 - static_bear_score) * (1.0 - liquidity_drain_risk))
        
        shrinking_volume_score = 1.0 - volume_increase_score
        # 下跌日的看涨分补充1：缩量下跌（卖盘衰竭）
        selling_exhaustion_score = (shrinking_volume_score * price_drop_score)
        # 下跌日的看涨分补充2：外部计算的衰竭反转信号
        exhaustion_reversal_score = self.strategy.atomic_states.get('SCORE_BULLISH_EXHAUSTION_REVERSAL', pd.Series(0.0, index=df.index))
        
        is_positive_day = df['pct_change_D'] > 0
        # 下跌日的看涨分取三种可能中的最强者
        bullish_score_on_down_day = np.maximum.reduce([
            yin_score.values, 
            selling_exhaustion_score.values,
            exhaustion_reversal_score.values
        ])
        # 融合得到最终的静态看涨分
        static_bull_score_np = np.where(is_positive_day, yang_score, bullish_score_on_down_day)
        static_bull_score = pd.Series(static_bull_score_np, index=df.index, dtype=np.float32)

        # 3. 计算统一的动态强度分 (d_intensity)
        vol_holo_bull, vol_holo_bear = calculate_holographic_dynamics(df, 'volume_D', norm_window)
        unified_d_intensity = ((vol_holo_bull + vol_holo_bear) / 2.0).astype(np.float32)

        # --- 循环内仅进行高效的字典赋值操作 ---
        for p in periods:
            s_bull[p] = static_bull_score
            s_bear[p] = static_bear_score
            d_intensity[p] = unified_d_intensity
            
        return s_bull, s_bear, d_intensity

    def _calculate_kline_pattern_health(self, df: pd.DataFrame, atomic_signals: Dict[str, pd.Series], norm_window: int, min_periods: int, periods: list) -> Tuple[Dict, Dict, Dict]:
        """
        【V2.6 · 赫尔墨斯之翼优化版】计算K线形态维度的三维健康度
        - 性能优化: 将不依赖于周期`p`的静态分(s_bull, s_bear)计算移出循环，避免重复计算。
                      动态分(d_intensity)因其计算依赖于`p`(`diff(p)`)，故保留在循环内。
        - 核心逻辑: 保持原有的动态分计算逻辑不变。
        """
        s_bull, s_bear, d_intensity = {}, {}, {}
        
        # --- 将静态分的计算提前至循环外 ---
        strong_close = normalize_score(atomic_signals.get('SCORE_PRICE_POSITION_IN_RANGE', pd.Series(0.5, index=df.index)), df.index, norm_window, True, min_periods)
        gap_support = normalize_score(atomic_signals.get('SCORE_GAP_SUPPORT_ACTIVE', pd.Series(0.0, index=df.index)), df.index, norm_window, True, min_periods)
        earth_heaven = normalize_score(atomic_signals.get('SCORE_BOARD_EARTH_HEAVEN', pd.Series(0.0, index=df.index)), df.index, norm_window, True, min_periods)
        # 温和上涨：将日涨幅限制在3%以内进行归一化，鼓励稳定上涨而非暴涨
        gentle_rise_raw = df['pct_change_D'].clip(0, 0.03) / 0.03
        gentle_rise = normalize_score(gentle_rise_raw, df.index, norm_window, True, min_periods)
        # 静态看涨分：取强势收盘、缺口支撑、地天板、温和上涨中的最强者
        static_bull_score = pd.Series(np.maximum.reduce([strong_close.values, gap_support.values, earth_heaven.values, gentle_rise.values]), index=df.index, dtype=np.float32)

        weak_close = 1.0 - strong_close
        upthrust = normalize_score(atomic_signals.get('SCORE_RISK_UPTHRUST_DISTRIBUTION', pd.Series(0.0, index=df.index)), df.index, norm_window, True, min_periods)
        heaven_earth = normalize_score(atomic_signals.get('SCORE_BOARD_HEAVEN_EARTH', pd.Series(0.0, index=df.index)), df.index, norm_window, True, min_periods)
        sharp_drop = normalize_score(atomic_signals.get('SCORE_KLINE_SHARP_DROP', pd.Series(0.0, index=df.index)), df.index, norm_window, True, min_periods)
        # 静态看跌分：取弱势收盘、上冲派发、天地板、急跌中的最强者
        static_bear_score = pd.Series(np.maximum.reduce([weak_close.values, upthrust.values, heaven_earth.values, sharp_drop.values]), index=df.index, dtype=np.float32)

        # --- 循环内仅保留必须依赖周期p的计算 ---
        for p in periods:
            s_bull[p] = static_bull_score
            s_bear[p] = static_bear_score
            
            # [代码保持] 动态强度分的计算依赖于周期p，因此保留在循环内
            # K线形态的动态分衡量的是“静态分自身的变化强度”
            bull_slope_strength = static_bull_score.diff(p).fillna(0).abs()
            bear_slope_strength = static_bear_score.diff(p).fillna(0).abs()
            # 取看涨和看跌分中变化更剧烈的一方作为动态强度
            intensity_slope = np.maximum(bull_slope_strength, bear_slope_strength)
            d_intensity[p] = normalize_score(intensity_slope, df.index, norm_window, ascending=True)
            
        return s_bull, s_bear, d_intensity

    # 以下方法被降级为私有，作为原子信号的生产者
    def _diagnose_kline_patterns(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.1 · 赫尔墨斯之翼优化版】诊断K线原子形态
        - 性能优化: 预先计算布尔掩码，避免在where和计算中重复执行条件判断。
        - 核心逻辑: 保持缺口支撑和急跌评分的计算逻辑不变。
        """
        states = {}
        p = get_params_block(self.strategy, 'kline_pattern_params')
        if not get_param_value(p.get('enabled'), False): return states
        
        # --- 缺口支撑信号计算 ---
        p_gap = p.get('gap_support_params', {})
        if get_param_value(p_gap.get('enabled'), True):
            persistence_days = get_param_value(p_gap.get('persistence_days'), 10)
            
            # 预先计算布尔掩码，提高代码可读性和效率
            gap_up_mask = df['low_D'] > df['high_D'].shift(1)
            
            # 找到发生向上缺口时，缺口的上沿（前一日的最高价）
            gap_high = df['high_D'].shift(1).where(gap_up_mask).ffill()
            # 定义缺口被回补的条件
            price_fills_gap_mask = df['close_D'] < gap_high
            
            # 使用持久化状态生成器，判断缺口支撑是否持续有效
            gap_support_state = create_persistent_state(
                df=df, 
                entry_event_series=gap_up_mask, 
                persistence_days=persistence_days, 
                break_condition_series=price_fills_gap_mask, 
                state_name='KLINE_STATE_GAP_SUPPORT_ACTIVE'
            )
            
            # 计算支撑强度：当前最低价离缺口上沿越远，支撑越强
            support_distance = (df['low_D'] - gap_high).clip(lower=0)
            # 使用当日收盘价的10%作为归一化基准，使评分具有相对意义
            normalization_base = (df['close_D'] * 0.1).replace(0, np.nan)
            support_strength_score = (support_distance / normalization_base).clip(0, 1).fillna(0)
            
            # 最终得分 = 支撑强度 * 支撑状态
            states['SCORE_GAP_SUPPORT_ACTIVE'] = (support_strength_score * gap_support_state).astype(np.float32)
            
        # --- 急跌信号计算 ---
        p_atomic = p.get('atomic_behavior_params', {})
        if get_param_value(p_atomic.get('enabled'), True) and 'pct_change_D' in df.columns:
            norm_window = get_param_value(p_atomic.get('norm_window'), 120)
            # 计算下跌幅度
            drop_magnitude = df['pct_change_D'].where(df['pct_change_D'] < 0, 0).abs()
            # 使用更高效的内部归一化函数
            sharp_drop_score = normalize_score(drop_magnitude, df.index, norm_window, ascending=True)
            states['SCORE_KLINE_SHARP_DROP'] = sharp_drop_score.astype(np.float32)
            
        return states

    def _diagnose_advanced_atomic_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.1 · 赫尔墨斯之翼优化版】诊断高级原子信号
        - 内存优化: 对连涨/连跌天数的计数结果使用`np.int16`存储，减少内存占用。
        - 核心逻辑: 保持高效的向量化连胜/连败计算逻辑不变。
        """
        states = {}
        p = get_params_block(self.strategy, 'advanced_atomic_params', {}) 
        if not get_param_value(p.get('enabled'), True): return states
        
        # 计算收盘价在当日振幅中的位置，值域[0, 1]
        price_range = (df['high_D'] - df['low_D']).replace(0, 1e-9)
        close_position_in_range = ((df['close_D'] - df['low_D']) / price_range).fillna(0.5)
        states['SCORE_PRICE_POSITION_IN_RANGE'] = close_position_in_range.astype(np.float32)
        
        # 高效计算连涨/连跌天数
        is_up_day = df['pct_change_D'] > 0
        is_down_day = df['pct_change_D'] < 0
        # 使用groupby和cumcount的经典向量化技巧计算连胜
        up_streak = (is_up_day.groupby((is_up_day != is_up_day.shift()).cumsum()).cumcount() + 1) * is_up_day
        down_streak = (is_down_day.groupby((is_down_day != is_down_day.shift()).cumsum()).cumcount() + 1) * is_down_day
        
        # 使用更节省内存的整数类型
        states['COUNT_CONSECUTIVE_UP_STREAK'] = up_streak.astype(np.int16)
        states['COUNT_CONSECUTIVE_DOWN_STREAK'] = down_streak.astype(np.int16)
        
        return states

    def _diagnose_board_patterns(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.1 · 赫尔墨斯之翼优化版】诊断天地板/地天板模式
        - 性能优化: 预先计算涨跌停价，并将多个评分因子的计算向量化，避免重复计算和中间Series的创建。
        - 核心逻辑: 保持评分的乘法融合逻辑不变。
        """
        states = {}
        p = get_params_block(self.strategy, 'board_pattern_params')
        if not get_param_value(p.get('enabled'), False): return states
        
        # --- 预先计算所有基础变量 ---
        prev_close = df['close_D'].shift(1)
        limit_up_threshold = get_param_value(p.get('limit_up_threshold'), 0.098)
        limit_down_threshold = get_param_value(p.get('limit_down_threshold'), -0.098)
        price_buffer = get_param_value(p.get('price_buffer'), 0.005)
        
        # 计算理论上的涨停价和跌停价
        limit_up_price = prev_close * (1 + limit_up_threshold)
        limit_down_price = prev_close * (1 + limit_down_threshold)
        
        # --- 向量化计算所有评分因子 ---
        # 1. 振幅强度分：当日振幅占理论最大振幅的比例
        day_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        theoretical_max_range = (limit_up_price - limit_down_price).replace(0, np.nan)
        strength_score = (day_range / theoretical_max_range).clip(0, 1).fillna(0)
        
        # 2. 地天板相关因子
        # 最低价接近跌停价的程度
        low_near_limit_down_score = ((limit_down_price * (1 + price_buffer) - df['low_D']) / (limit_down_price * price_buffer).replace(0, np.nan)).clip(0, 1).fillna(0)
        # 收盘价接近涨停价的程度
        close_near_limit_up_score = ((df['close_D'] - limit_up_price * (1 - price_buffer)) / (limit_up_price * price_buffer).replace(0, np.nan)).clip(0, 1).fillna(0)
        
        # 3. 天地板相关因子
        # 最高价接近涨停价的程度
        high_near_limit_up_score = ((df['high_D'] - limit_up_price * (1 - price_buffer)) / (limit_up_price * price_buffer).replace(0, np.nan)).clip(0, 1).fillna(0)
        # 收盘价接近跌停价的程度
        close_near_limit_down_score = ((limit_down_price * (1 + price_buffer) - df['close_D']) / (limit_down_price * price_buffer).replace(0, np.nan)).clip(0, 1).fillna(0)
        
        # --- 最终融合 ---
        # 地天板得分 = 振幅强度 * 触及跌停 * 收于涨停
        states['SCORE_BOARD_EARTH_HEAVEN'] = (strength_score * low_near_limit_down_score * close_near_limit_up_score).astype(np.float32)
        # 天地板得分 = 振幅强度 * 触及涨停 * 收于跌停
        states['SCORE_BOARD_HEAVEN_EARTH'] = (strength_score * high_near_limit_up_score * close_near_limit_down_score).astype(np.float32)
        
        return states

    def _diagnose_upthrust_distribution(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【V3.1 · 赫尔墨斯之翼优化版】上冲派发风险诊断引擎
        - 性能优化: 使用`normalize_score`替换`.rolling().rank(pct=True)`，提升归一化效率。
                      将多个评分因子的计算向量化，减少中间步骤。
        - 核心逻辑: 保持“关系元分析”范式和三因子融合逻辑不变。
        """
        p = get_params_block(self.strategy, 'upthrust_distribution_params', {})
        signal_name = 'SCORE_RISK_UPTHRUST_DISTRIBUTION'
        default_series = pd.Series(0.0, index=df.index, name=signal_name, dtype=np.float32)
        
        if not get_param_value(p.get('enabled'), False):
            return default_series
        
        overextension_ma_period = get_param_value(p.get('overextension_ma_period'), 55)
        ma_col = f'EMA_{overextension_ma_period}_D'
        if not all(col in df.columns for col in ['high_D', 'low_D', 'close_D', 'volume_D', ma_col]):
            return default_series
            
        norm_window = get_param_value(p.get('norm_window'), 55)
        
        # --- 向量化计算瞬时风险关系快照分 ---
        # 1. 乖离率分：收盘价偏离MA55的程度
        overextension_ratio = (df['close_D'] / df[ma_col] - 1).clip(0)
        overextension_score = normalize_score(overextension_ratio, df.index, norm_window, ascending=True)
        
        # 2. 成交量分：当日成交量在近期所处的位置
        volume_score = normalize_score(df['volume_D'], df.index, norm_window, ascending=True)
        
        # 3. 弱势收盘分：收盘价在当日振幅中位置越低，得分越高
        total_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        close_position_in_range = ((df['close_D'] - df['low_D']) / total_range).fillna(0.5)
        weak_close_score = 1 - close_position_in_range
        
        # 融合三大支柱，得到单一的“瞬时风险关系分”
        snapshot_score = (overextension_score * volume_score * weak_close_score).astype(np.float32)
        
        # 调用核心引擎，分析“风险关系”的拐点
        final_signal_dict = self._perform_relational_meta_analysis(
            df=df,
            snapshot_score=snapshot_score,
            signal_name=signal_name
        )
        
        return final_signal_dict.get(signal_name, default_series)

    def _diagnose_volume_price_dynamics(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V2.0 · 关系元分析版】
        - 核心升级: 对“VPA滞涨风险”信号采用“关系元分析”范式重构。
        """
        states = {}
        required_cols = ['volume_D', 'VOL_MA_21_D', 'pct_change_D', 'SLOPE_5_volume_D', 'ACCEL_5_volume_D', 'VPA_EFFICIENCY_D', 'SLOPE_5_VPA_EFFICIENCY_D']
        if any(col not in df.columns for col in required_cols): return states
        
        p_vpa = params.get('vpa_dynamics_params', {})
        norm_window = get_param_value(p_vpa.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        
        # 对 VPA 滞涨风险进行关系元分析
        # 第一维度：计算“瞬时滞涨关系快照分”
        volume_ratio = (df['volume_D'] / df['VOL_MA_21_D'].replace(0, np.nan)).fillna(1.0)
        huge_volume_score = normalize_score(volume_ratio, df.index, norm_window, ascending=True)
        price_stagnant_score = 1 - normalize_score(df['pct_change_D'].abs(), df.index, norm_window, ascending=True)
        stagnation_snapshot_score = (huge_volume_score * price_stagnant_score).astype(np.float32)
        
        # 第二维度：调用核心引擎
        stagnation_risk_states = self._perform_relational_meta_analysis(
            df=df,
            snapshot_score=stagnation_snapshot_score,
            signal_name='SCORE_RISK_VPA_STAGNATION'
        )
        states.update(stagnation_risk_states)
        
        return states

    def _diagnose_price_volume_atomics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V8.1 · 幽灵信号净化版】
        - 核心修改: 移除了对 SCORE_PRICE_POSITION_IN_RECENT_RANGE 这个幽灵信号的计算。
        """
        states = {}
        p = get_params_block(self.strategy, 'price_volume_atomic_params')
        if not get_param_value(p.get('enabled'), True): return states
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        vol_ma_col = 'VOL_MA_21_D'
        if vol_ma_col in df.columns and 'pct_change_D' in df.columns:
            drop_magnitude = df['pct_change_D'].where(df['pct_change_D'] < 0, 0).abs()
            price_drop_score = drop_magnitude.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.0)
            volume_ratio = (df['volume_D'] / df[vol_ma_col].replace(0, np.nan)).fillna(1.0)
            volume_shrink_score = (1 - volume_ratio.rolling(window=norm_window, min_periods=min_periods).rank(pct=True)).fillna(0.5)
            states['SCORE_VOL_WEAKENING_DROP'] = (price_drop_score * volume_shrink_score).astype(np.float32)

        p_drain = p.get('liquidity_drain_params', {})
        drain_window = get_param_value(p_drain.get('window'), 20)
        
        price_drop_magnitude = df['pct_change_D'].clip(upper=0).abs()
        price_drop_score = normalize_score(price_drop_magnitude, df.index, drain_window, ascending=True)
        volume_shrink_score = normalize_score(df['volume_D'], df.index, drain_window, ascending=False)
        drain_snapshot_score = (price_drop_score * volume_shrink_score).astype(np.float32)

        liquidity_drain_states = self._perform_relational_meta_analysis(
            df=df,
            snapshot_score=drain_snapshot_score,
            signal_name='SCORE_RISK_LIQUIDITY_DRAIN'
        )
        states.update(liquidity_drain_states)

        p_rev = p.get('exhaustion_reversal_params', {})
        rev_window = get_param_value(p_rev.get('window'), 5)
        
        is_stabilizing = (df['low_D'] >= df['low_D'].rolling(rev_window).min()).astype(int)
        price_volatility = df.get('ATR_14_D', pd.Series(df['high_D'] - df['low_D'], index=df.index))
        stabilization_score = is_stabilizing * normalize_score(price_volatility, df.index, norm_window, ascending=False)
        volume_dry_up_score = normalize_score(df['volume_D'], df.index, norm_window, ascending=False)
        
        # 此处需要 close_position_in_range，从 _diagnose_advanced_atomic_signals 获取
        close_position_in_range = self.strategy.atomic_states.get('SCORE_PRICE_POSITION_IN_RANGE', pd.Series(0.5, index=df.index))
        context_score = 1 - close_position_in_range
        snapshot_score = (stabilization_score * volume_dry_up_score * context_score)
        
        exhaustion_reversal_states = self._perform_relational_meta_analysis(
            df=df,
            snapshot_score=snapshot_score,
            signal_name='SCORE_BULLISH_EXHAUSTION_REVERSAL'
        )
        states.update(exhaustion_reversal_states)
        
        return states

    def _diagnose_atomic_bottom_formation(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.2 · 纯净输出版】原子级“底部形态”诊断引擎
        - 核心升级: 只输出最终的元分析信号。
        """
        # 第一维度：计算“瞬时关系快照分”
        ma55 = df.get('EMA_55_D', df['close_D'])
        distance_from_ma55 = (df['close_D'] - ma55) / ma55
        lifeline_proximity_score = np.exp(-((distance_from_ma55 - 0.015) / 0.03)**2)
        
        rsi = df.get('RSI_13_D', pd.Series(50, index=df.index))
        was_rsi_oversold = (rsi.rolling(window=10).min() < 35).astype(float)
        price_pos_yearly = normalize_score(df['close_D'], df.index, window=250, ascending=True, default_value=0.5)
        deep_bottom_context_score = 1.0 - price_pos_yearly
        pessimism_exhaustion_score = np.maximum(was_rsi_oversold, deep_bottom_context_score)

        vol_compression_score = normalize_score(df.get('BBW_21_2.0_D'), df.index, 60, ascending=False)

        snapshot_score = pd.Series(
            (lifeline_proximity_score * pessimism_exhaustion_score * vol_compression_score),
            index=df.index
        ).astype(np.float32)

        # 只返回最终信号字典
        return self._perform_relational_meta_analysis(
            df=df,
            snapshot_score=snapshot_score,
            signal_name='SCORE_ATOMIC_BOTTOM_FORMATION'
        )

    def _diagnose_atomic_rebound_reversal(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.2 · 纯净输出版】原子级“史诗探底回升”诊断引擎
        - 核心升级: 只输出最终的元分析信号。
        """
        # 第一维度：计算“瞬时关系快照分”
        p_rebound = get_params_block(self.strategy, 'panic_selling_setup_params', {})
        despair_context_score = self._calculate_despair_context_score(df, p_rebound)
        structural_test_score = self.tactic_engine.calculate_structural_test_score(df, p_rebound)
        day_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        close_position_in_range = ((df['close_D'] - df['low_D']) / day_range).clip(0, 1).fillna(0.0)
        is_recovering_today = (df['pct_change_D'] > -0.01).astype(float)
        confirmation_score = (close_position_in_range * is_recovering_today)

        snapshot_score = (despair_context_score * structural_test_score * confirmation_score).astype(np.float32)
        
        # 只返回最终信号字典
        return self._perform_relational_meta_analysis(
            df=df,
            snapshot_score=snapshot_score,
            signal_name='SCORE_ATOMIC_REBOUND_REVERSAL'
        )

    def _calculate_despair_context_score(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【V1.0 · 新增/移植】“冥河之渡”多维绝望背景诊断引擎
        - 来源: 从 behavioral_intelligence 完美移植而来，作为“最终审判”计划的一部分。
        - 核心职责: 为本模块提供与行为引擎完全一致的、最高规格的绝望背景计算能力。
        """
        # --- 步骤 1: 获取参数 ---
        despair_periods = get_param_value(params.get('despair_periods'), {'short': (21, 5), 'mid': (60, 21), 'long': (250, 60)})
        despair_weights = get_param_value(params.get('despair_weights'), {'short': 0.2, 'mid': 0.3, 'long': 0.5})
        
        period_scores = []
        period_weight_values = []

        # --- 步骤 2: 遍历所有绝望周期，独立计算分数 ---
        for name, (drawdown_period, roc_period) in despair_periods.items():
            # 2.1 计算该周期的“坠落深度”
            rolling_peak = df['high_D'].rolling(window=drawdown_period, min_periods=max(1, drawdown_period//2)).max()
            drawdown_from_peak = (rolling_peak - df['close_D']) / rolling_peak.replace(0, np.nan)
            magnitude_score = normalize_score(drawdown_from_peak.clip(lower=0), df.index, window=drawdown_period, ascending=True)
            
            # 2.2 计算该周期的“坠落速度”
            price_roc = df['close_D'].pct_change(roc_period)
            velocity_score = normalize_score(price_roc, df.index, window=drawdown_period, ascending=False)
            
            # 2.3 融合得到该周期的绝望分数
            period_despair_score = (magnitude_score * velocity_score)**0.5
            
            period_scores.append(period_despair_score.values)
            period_weight_values.append(despair_weights.get(name, 0.0))

        # --- 步骤 3: 对所有周期的绝望分数进行加权几何平均 ---
        if not period_scores:
            return pd.Series(0.0, index=df.index)

        weights_array = np.array(period_weight_values)
        total_weights = weights_array.sum()
        if total_weights > 0:
            weights_array /= total_weights
        else:
            weights_array = np.full_like(weights_array, 1.0 / len(weights_array))

        stacked_scores = np.stack(period_scores, axis=0)
        
        final_score_values = np.prod(stacked_scores ** weights_array[:, np.newaxis], axis=0)
        
        return pd.Series(final_score_values, index=df.index, dtype=np.float32)

    def _diagnose_atomic_continuation_reversal(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 · 关系元分析版】原子级“延续性反转”诊断引擎
        - 核心升级: 采用“关系元分析”范式，寻找“延续性反转关系”的向上拐点。
        """
        # 第一维度：计算“瞬时关系快照分”
        p_continuation = get_params_block(self.strategy, 'continuation_reversal_params', {})
        ma_periods = get_param_value(p_continuation.get('ma_periods'), [5, 13, 21, 55])
        uptrending_ma_count = pd.Series(0, index=df.index)
        for p in ma_periods:
            ma_col = f'EMA_{p}_D'
            if ma_col in df:
                uptrending_ma_count += (df[ma_col] > df[ma_col].shift(1)).astype(int)
        trend_alignment_score = uptrending_ma_count / len(ma_periods)
        structural_test_score = self.tactic_engine.calculate_structural_test_score(df, p_continuation)
        day_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        close_position_in_range = ((df['close_D'] - df['low_D']) / day_range).clip(0, 1).fillna(0.0)
        confirmation_score = close_position_in_range

        # --- 融合三大支柱，得到单一的“瞬时关系分” ---
        snapshot_score = (trend_alignment_score * structural_test_score * confirmation_score).astype(np.float32)
        
        # 第二维度：调用核心引擎，分析“关系”的拐点
        return self._perform_relational_meta_analysis(
            df=df,
            snapshot_score=snapshot_score,
            signal_name='SCORE_ATOMIC_CONTINUATION_REVERSAL'
        )

    def _perform_relational_meta_analysis(self, df: pd.DataFrame, snapshot_score: pd.Series, signal_name: str) -> Dict[str, pd.Series]:
        """
        【V2.0 · 赫拉织布机V2版】关系元分析核心引擎
        - 核心升级: 实现“状态 * (1 + 动态杠杆)”的动态价值调制范式。
        - 新核心逻辑:
          1. 状态分(State): 瞬时关系快照分，是价值基石。
          2. 速度分(Velocity): 关系分趋势，归一化到[-1, 1]。
          3. 加速度分(Acceleration): 关系分趋势的趋势，归一化到[-1, 1]。
          4. 最终分 = 状态分 * (1 + 速度分*w_vel + 加速度分*w_accel)，动态决定最终价值。
        """
        states = {}
        # 从配置中获取新的动态杠杆权重
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_meta = get_param_value(p_conf.get('relational_meta_analysis_params'), {})
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.6)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.4)

        # --- 从ProcessIntelligence借鉴的核心参数 ---
        norm_window = 55
        meta_window = 5
        bipolar_sensitivity = 1.0

        # 第一维度：状态分 (State Score)
        state_score = snapshot_score.clip(0, 1)

        # 第二维度：速度分 (Velocity Score)
        relationship_trend = snapshot_score.diff(meta_window).fillna(0)
        velocity_score = normalize_to_bipolar(
            series=relationship_trend, target_index=df.index,
            window=norm_window, sensitivity=bipolar_sensitivity
        )

        # 第三维度：加速度分 (Acceleration Score)
        relationship_accel = relationship_trend.diff(meta_window).fillna(0)
        acceleration_score = normalize_to_bipolar(
            series=relationship_accel, target_index=df.index,
            window=norm_window, sensitivity=bipolar_sensitivity
        )

        # 终极融合：动态价值调制
        # 计算动态杠杆
        dynamic_leverage = 1 + (velocity_score * w_velocity) + (acceleration_score * w_acceleration)
        
        # 应用公式：最终分 = 状态分 * 动态杠杆
        final_score = (state_score * dynamic_leverage).clip(0, 1)
        
        states[signal_name] = final_score.astype(np.float32)
        return states

    def _supreme_fusion_engine(self, df: pd.DataFrame, signals_to_fuse: Dict[str, pd.Series], params: Dict) -> pd.Series:
        """
        【V1.0 · 新增】最高神谕融合引擎 (宙斯之雷)
        - 核心职责: 对多个动态原子信号进行加权协同融合。
        """
        fusion_weights = get_param_value(params.get('fusion_weights'), {})
        synergy_bonus_factor = get_param_value(params.get('synergy_bonus_factor'), 0.5)

        valid_signals = []
        weights = []
        
        for name, weight in fusion_weights.items():
            signal_name_full = f'SCORE_ATOMIC_{name}'
            if signal_name_full in signals_to_fuse and weight > 0:
                # 使用 .values 确保 numpy 操作的性能和对齐
                valid_signals.append(signals_to_fuse[signal_name_full].values)
                weights.append(weight)

        if not valid_signals:
            return pd.Series(0.0, index=df.index, dtype=np.float32)

        # --- 基础融合：加权几何平均 ---
        stacked_signals = np.stack(valid_signals, axis=0)
        weights_array = np.array(weights)
        # 归一化权重
        total_weight = weights_array.sum()
        if total_weight > 0:
            normalized_weights = weights_array / total_weight
        else:
            normalized_weights = np.full_like(weights_array, 1.0 / len(weights_array))
        
        # 为避免 log(0) 错误，给信号值增加一个极小量
        safe_signals = np.maximum(stacked_signals, 1e-9)
        log_signals = np.log(safe_signals)
        weighted_log_sum = np.sum(log_signals * normalized_weights[:, np.newaxis], axis=0)
        base_fusion_score = np.exp(weighted_log_sum)

        # --- 协同奖励：当多个信号同时活跃时给予奖励 ---
        # 这里我们简化为取最强的两个信号的乘积作为协同基础
        if stacked_signals.shape[0] >= 2:
            # 沿信号轴排序，取最后两个（最大的）
            sorted_signals = np.sort(stacked_signals, axis=0)
            synergy_base = sorted_signals[-1] * sorted_signals[-2]
            synergy_bonus = (synergy_base**0.5) * synergy_bonus_factor
        else:
            synergy_bonus = 0.0
            
        # --- 最终融合 ---
        final_score = (base_fusion_score * (1 + synergy_bonus)).clip(0, 1)
        
        return pd.Series(final_score, index=df.index, dtype=np.float32)







