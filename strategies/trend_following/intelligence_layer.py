# 文件: strategies/trend_following/intelligence_layer.py
# 情报层
import pandas as pd
import numpy as np
from scipy.stats import linregress
from typing import Dict, Tuple
from enum import Enum
from strategies.kline_pattern_recognizer import KlinePatternRecognizer
from .utils import get_params_block, get_param_value, create_persistent_state, format_debug_dates

class MainForceState(Enum):
    IDLE = 0
    ACCUMULATING = 1
    WASHING = 2
    MARKUP = 3
    DISTRIBUTING = 4
    COLLAPSE = 5

class IntelligenceLayer:
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance
        self.kline_params = get_params_block(self.strategy, 'kline_pattern_params')
        self.pattern_recognizer = KlinePatternRecognizer(params=self.kline_params)

    def run_all_diagnostics(self) -> Dict:
        """
        【V327.2 终极顺序版】
        - 核心修复: 再次修正模块执行顺序，确保所有依赖关系100%正确。
        """
        print("--- [情报层 V327.2] 步骤1: 运行所有诊断模块... ---")
        df = self.strategy.df_indicators
        self.strategy.atomic_states = {}

        # --- 阶段一: 基础数据衍生与K线形态 ---
        # print("    -> [情报层] 阶段1: 基础数据衍生...")
        df = self.pattern_recognizer.identify_all(df)
        self.strategy.atomic_states.update(self._diagnose_kline_patterns(df))
        self.strategy.atomic_states.update(self._diagnose_board_patterns(df))

        # --- 阶段二: 基础原子状态诊断 ---
        # print("    -> [情报层] 阶段2: 基础原子状态诊断...")
        self.dynamic_thresholds = self._get_dynamic_thresholds(df)
        self.strategy.atomic_states.update(self._diagnose_ma_states(df))
        self.strategy.atomic_states.update(self._diagnose_volatility_states(df))
        self.strategy.atomic_states.update(self._diagnose_trend_dynamics(df))
        self.strategy.atomic_states.update(self._diagnose_oscillator_states(df))
        self.strategy.atomic_states.update(self._diagnose_fibonacci_support(df))
        self.strategy.atomic_states.update(self._diagnose_capital_states(df))
        
        # 2.1 首先，运行“战场上下文”模块，定义“在哪里”。
        self.strategy.atomic_states.update(self._diagnose_contextual_zones(df))
        # 2.2 然后，运行“动态筹码分析”，它现在可以安全地使用上一步的“战场”情报。
        self.strategy.atomic_states.update(self._diagnose_dynamic_chip_states(df))
        self.strategy.atomic_states.update(self._diagnose_chip_opportunities(df))
        self.strategy.atomic_states.update(self._diagnose_chip_risks_and_behaviors(df))

        # --- 阶段三: 复合原子状态诊断 ---
        # print("    -> [情报层] 阶段3: 复合原子状态诊断...")
        self.strategy.atomic_states.update(self._diagnose_peak_formation_dynamics(df))
        self.strategy.atomic_states.update(self._diagnose_behavioral_patterns(df))
        df, structure_states = self._diagnose_market_structure_command(df)
        self.strategy.atomic_states.update(structure_states)
        
        chip_states, chip_triggers = self._run_chip_intelligence_command(df)
        self.strategy.atomic_states.update(chip_states)
        
        self.strategy.atomic_states.update(self._diagnose_market_structure_states(df))
        # “打压回踩”诊断模块
        self.strategy.atomic_states.update(self._diagnose_pullback_character(df))
        # “压缩区洗盘”机会诊断模块。
        self.strategy.atomic_states.update(self._diagnose_squeeze_zone_opportunities(df))
        
        # “持仓风险”诊断模块
        self.strategy.atomic_states.update(self._diagnose_holding_risks(df))
        self.strategy.atomic_states.update(self._diagnose_post_accumulation_phase(df))
        self.strategy.atomic_states.update(self._diagnose_breakout_pullback_relay(df))

        # --- 阶段四: 顶层认知与行为序列合成 ---
        # print("    -> [情报层] 阶段4: 顶层认知合成...")
        self.strategy.atomic_states.update(self._diagnose_trend_stage_context(df))
        self.strategy.atomic_states.update(self._diagnose_structural_mechanics(df))
        self.strategy.atomic_states.update(self._run_cognitive_synthesis_engine(df))
        self.strategy.df_indicators = self._determine_main_force_behavior_sequence(df)
        
        # --- 阶段五: 生成触发器和剧本 ---
        # print("    -> [情报层] 阶段5: 生成最终触发器与剧本...")
        trigger_events = self._define_trigger_events(df)
        trigger_events.update(chip_triggers)
        self.strategy.setup_scores, self.strategy.playbook_states = self._generate_playbook_states(trigger_events)
        
        is_in_squeeze_window = self.strategy.atomic_states.get('VOL_STATE_SQUEEZE_WINDOW', pd.Series(False, index=df.index))
        is_bb_breakout = df['close_D'] > df.get('BBU_21_2.0_D', float('inf'))
        trigger_events['VOL_BREAKOUT_FROM_SQUEEZE'] = is_bb_breakout & is_in_squeeze_window.shift(1).fillna(False)
        
        return trigger_events

    def _run_chip_intelligence_command(self, df: pd.DataFrame) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
        """
        【V316.0 筹码加权版】筹码情报最高司令部
        - 核心重构: 不再生成绝对的`CHIP_STRUCTURE_OK`，而是提炼一个综合性的
                    “严重筹码结构风险”信号: `RISK_CHIP_STRUCTURE_CRITICAL_FAILURE`。
                    这为决策层提供了更具权重的、可量化的地基风险评估依据。
        """
        # print("        -> [筹码情报最高司令部 V316.0 筹码加权版] 启动...")
        states = {}
        triggers = {}
        default_series = pd.Series(False, index=df.index)

        p = get_params_block(self.strategy, 'chip_feature_params')
        if not get_param_value(p.get('enabled'), False): return states, triggers

        required_cols = ['concentration_90pct_D', 'chip_health_score_D', 'peak_cost_accel_5d_D']
        if any(col not in df.columns for col in required_cols): return states, triggers

        dynamic_states = self._diagnose_dynamic_chip_states(df)
        states.update(dynamic_states)

        p_struct = p.get('structure_params', {})
        conc_col = 'concentration_90pct_D'
        
        is_concentrated_static = df[conc_col] < get_param_value(p_struct.get('high_concentration_threshold'), 0.15)
        is_trend_healthy = ~states.get('RISK_DYN_DIVERGING', default_series)
        states['CHIP_STATE_HIGHLY_CONCENTRATED'] = is_concentrated_static & is_trend_healthy

        p_ignition = p.get('ignition_params', {})
        if get_param_value(p_ignition.get('enabled'), True):
            accel_threshold = get_param_value(p_ignition.get('accel_threshold'), 0.01)
            triggers['TRIGGER_CHIP_IGNITION'] = df.get('peak_cost_accel_5d_D', 0) > accel_threshold

        states['CHIP_HEALTH_EXCELLENT'] = df.get('chip_health_score_D', 0) > 85
        
        # print("          -> [情报提纯] 正在对“高度集中”状态进行机会提纯...")
        # 1. 基础条件：筹码必须已经高度集中 (静态)
        is_highly_concentrated_static = df[conc_col] < get_param_value(p_struct.get('high_concentration_threshold'), 0.15)
        states['CHIP_STATE_HIGHLY_CONCENTRATED'] = is_highly_concentrated_static # 保留原始的基础状态信号，供其他模块使用
        # 2. 动态条件：筹码必须仍在持续集中 (动态)
        is_still_concentrating = self.strategy.atomic_states.get('CHIP_DYN_CONCENTRATING', default_series)
        # 3. 稳定条件：成本峰必须稳定，表明吸筹/洗盘阶段完成 (稳定)
        cost_slope_col = 'SLOPE_5_peak_cost_D'
        cost_stability_threshold = get_param_value(p_struct.get('cost_stability_threshold'), 0.005)
        is_cost_peak_stable = df[cost_slope_col].abs() < cost_stability_threshold
        # 最终裁定：S级机会是“静态+动态+稳定”的三重共振
        states['OPP_CHIP_SETUP_S'] = is_highly_concentrated_static & is_still_concentrating & is_cost_peak_stable
        if states['OPP_CHIP_SETUP_S'].any():
            print(f"            -> [情报] 侦测到 {states['OPP_CHIP_SETUP_S'].sum()} 次S级“筹码高度控盘”机会！")

        is_in_high_level_zone = self.strategy.atomic_states.get('CONTEXT_RISK_HIGH_LEVEL_ZONE', default_series)
        worsening_threshold = 1.05
        concentration_21d_ago = df[conc_col].shift(21)
        is_concentration_worsened = df[conc_col] > (concentration_21d_ago * worsening_threshold)
        states['RISK_CONTEXT_LONG_TERM_DISTRIBUTION'] = is_concentration_worsened & is_in_high_level_zone

        chip_risk_1 = states.get('RISK_DYN_DIVERGING', default_series)
        chip_risk_2 = states.get('RISK_DYN_COST_FALLING', default_series)
        chip_risk_3 = states.get('RISK_DYN_WINNER_RATE_COLLAPSING', default_series)
        chip_risk_4 = states.get('RISK_CONTEXT_LONG_TERM_DISTRIBUTION', default_series)
        
        # 只要有任何一个核心筹码风险存在，就标记为严重结构性风险
        is_chip_structure_unhealthy = chip_risk_1 | chip_risk_2 | chip_risk_3 | chip_risk_4
        states['RISK_CHIP_STRUCTURE_CRITICAL_FAILURE'] = is_chip_structure_unhealthy
        # if is_chip_structure_unhealthy.any():
            # print(f"          -> [地基风险报告] 在 {is_chip_structure_unhealthy.sum()} 天内，侦测到严重筹码结构风险。")

        is_highly_concentrated = states.get('CHIP_STATE_HIGHLY_CONCENTRATED', default_series)
        is_cost_rising = states.get('CHIP_DYN_COST_RISING', default_series)
        is_winner_rate_rising = states.get('CHIP_DYN_WINNER_RATE_RISING', default_series)
        is_long_term_distributing = states.get('RISK_CONTEXT_LONG_TERM_DISTRIBUTION', default_series)
        is_cost_stable = df.get('SLOPE_5_peak_cost_D', default_series).abs() < 0.01

        states['CHIPCON_4_READINESS'] = is_highly_concentrated & is_cost_stable & ~is_long_term_distributing
        states['CHIPCON_3_HIGH_ALERT'] = is_highly_concentrated & is_cost_rising & is_winner_rate_rising & ~is_long_term_distributing
        
        # print("        -> [筹码情报最高司令部 V316.0 筹码加权版] 分析完毕。")
        return states, triggers

    def _diagnose_oscillator_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """【V234.0 最终净化版】震荡指标状态诊断中心"""
        states = {}
        p = get_params_block(self.strategy, 'oscillator_state_params')
        if not get_param_value(p.get('enabled'), False): return states
        
        # --- RSI 相关状态 ---
        rsi_col = 'RSI_13_D'
        if rsi_col in df.columns:
            states['OSC_STATE_RSI_OVERBOUGHT'] = df[rsi_col] > get_param_value(p.get('rsi_overbought'), 80)
            states['OSC_STATE_RSI_OVERSOLD'] = df[rsi_col] < get_param_value(p.get('rsi_oversold'), 25)
        
        # --- MACD 相关状态 ---
        macd_h_col = 'MACDh_13_34_8_D'
        macd_z_col = 'MACD_HIST_ZSCORE_D'
        if macd_h_col in df.columns:
            states['OSC_STATE_MACD_BULLISH'] = df[macd_h_col] > 0
        if macd_z_col in df.columns:
            is_price_higher = df['close_D'] > df['close_D'].rolling(10).max().shift(1)
            is_macd_z_lower = df[macd_z_col] < df[macd_z_col].rolling(10).max().shift(1)
            states['OSC_STATE_MACD_DIVERGENCE'] = is_price_higher & is_macd_z_lower

        # ▼▼▼ BIAS机会状态的诊断 ▼▼▼
        p_bias = p.get('bias_dynamic_threshold', {})
        bias_col = 'BIAS_55_D'
        if bias_col in df.columns:
            window = get_param_value(p_bias.get('window'), 120)
            quantile = get_param_value(p_bias.get('quantile'), 0.1)
            dynamic_oversold_threshold = df[bias_col].rolling(window=window).quantile(quantile)
            states['OPP_STATE_NEGATIVE_DEVIATION'] = df[bias_col] < dynamic_oversold_threshold

        return states

    # ─> 能量与波动侦察部 (Energy & Volatility Reconnaissance)
    #    -> 核心职责: 侦测市场能量的“积蓄”(压缩)与“释放”(放量)。
    #    -> 指挥官: _diagnose_volatility_states()
    def _diagnose_volatility_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V283.0 融合思想版】
        - 核心升级: 引入“极致压缩”信号。
          - VOL_STATE_SQUEEZE_WINDOW: 基础的波动率压缩窗口 (静态)。
          - VOL_STATE_EXTREME_SQUEEZE: 在压缩窗口内，要求波动率带宽仍在收缩 (动态)，是更高质量的突破前兆。
        """
        # print("        -> [能量与波动侦察部 V283.0] 启动，正在执行融合分析...")
        states = {}
        p = get_params_block(self.strategy, 'volatility_state_params')
        if not get_param_value(p.get('enabled'), False): return states

        bbw_col = 'BBW_21_2.0_D'
        bbw_slope_col = 'SLOPE_5_BBW_21_2.0_D'
        vol_ma_col = 'VOL_MA_21_D'

        if not all(c in df.columns for c in [bbw_col, bbw_slope_col, vol_ma_col]):
            print(f"          -> [警告] 缺少诊断波动所需列，跳过。")
            return states

        # --- 1. 静态分析：定义压缩事件和缩量状态 ---
        squeeze_threshold = df[bbw_col].rolling(60).quantile(get_param_value(p.get('squeeze_percentile'), 0.1))
        squeeze_event = (df[bbw_col] < squeeze_threshold) & (df[bbw_col].shift(1) >= squeeze_threshold)
        states['VOL_EVENT_SQUEEZE'] = squeeze_event
        states['VOL_STATE_SHRINKING'] = df['volume_D'] < df[vol_ma_col] * get_param_value(p.get('shrinking_ratio'), 0.8)

        # --- 2. 状态机：生成基础的“压缩窗口” ---
        p_context = p.get('squeeze_context', {})
        volume_break_ratio = get_param_value(p_context.get('volume_break_ratio'), 1.5)
        break_condition = df['volume_D'] > df[vol_ma_col] * volume_break_ratio
        persistence_days = get_param_value(p_context.get('persistence_days'), 10)
        squeeze_window = create_persistent_state(
            df=df, entry_event_series=squeeze_event, persistence_days=persistence_days,
            break_condition_series=break_condition, state_name='VOL_STATE_SQUEEZE_WINDOW'
        )
        states['VOL_STATE_SQUEEZE_WINDOW'] = squeeze_window

        # --- 3. 【融合生成】高质量信号 ---
        # “极致压缩” (S级信号): 在压缩窗口内，要求波动率仍在收缩 (斜率为负)
        is_still_squeezing = df[bbw_slope_col] < 0
        states['VOL_STATE_EXTREME_SQUEEZE'] = squeeze_window & is_still_squeezing

        return states

    def run_force_vector_analysis(self) -> Dict[str, pd.Series]:
        """
        【V317.0 新增】动态力学分析引擎
        - 核心职责: 计算进攻分和风险分的“加速度”，捕捉双向动能的剧烈变化。
                    这是判断趋势强化或转折的关键“势能”情报。
        """
        # print("        -> [动态力学分析引擎 V317.0] 启动，正在计算势能加速度...")
        states = {}
        df = self.strategy.df_indicators
        
        # 确保 entry_score 和 risk_score 已经计算完毕
        if 'entry_score' not in df.columns or 'risk_score' not in df.columns:
            print("          -> [警告] 缺少 entry_score 或 risk_score，力学分析跳过。")
            return states

        window = 5 # 使用5日窗口计算趋势和加速度

        # 1. 计算“进攻”和“风险”的趋势（斜率）
        entry_score_slope = df['entry_score'].rolling(window).apply(
            lambda y: linregress(np.arange(window), y).slope if len(y.dropna()) == window else np.nan, raw=False
        )
        risk_score_slope = df['risk_score'].rolling(window).apply(
            lambda y: linregress(np.arange(window), y).slope if len(y.dropna()) == window else np.nan, raw=False
        )

        # 2. 计算“进攻”和“风险”的加速度（斜率的差分）
        entry_score_accel = entry_score_slope.diff()
        risk_score_accel = risk_score_slope.diff()

        # 3. 定义加速度阈值，过滤掉无意义的波动
        accel_threshold = 1.0 # 当加速度变化大于1时，我们认为是有意义的

        # 4. 生成四种核心的“力学”原子状态
        states['FORCE_VECTOR_OFFENSE_ACCELERATING'] = entry_score_accel > accel_threshold
        states['FORCE_VECTOR_OFFENSE_DECELERATING'] = entry_score_accel < -accel_threshold
        states['FORCE_VECTOR_RISK_ACCELERATING'] = risk_score_accel > accel_threshold
        states['FORCE_VECTOR_RISK_DECELERATING'] = risk_score_accel < -accel_threshold
        
        self.strategy.atomic_states.update(states)
        
        print("          -> [力学分析引擎] 进攻/风险的加速度情报已生成。")
        return states

    # 风险情报总局 (Risk Intelligence Bureau)
    #    -> 核心职责: 汇总所有负面/风险信号。
    #    -> 指挥官: _diagnose_all_risk_signals()
    def _diagnose_all_risk_signals(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V222.0 新增】风险情报总局
        - 核心职责: 统一指挥所有风险侦察部队，收集所有风险信号，形成一份完整的、
                    不包含任何评分的“每日风险简报”。这是风险评估的唯一情报来源。
        """
        print("        -> [风险情报总局 V222.0] 启动，正在汇总所有风险信号...")
        risk_signals = {}

        # 1. 调动“结构风险”侦察部队
        risk_signals.update(self._diagnose_upthrust_distribution(df, params))
        risk_signals.update(self._diagnose_structure_breakdown(df, params))

        # 2. 汇总来自“原子状态”的战略级风险
        # 这些是最高优先级的风险，直接从 atomic_states 中提取
        strategic_risks = [
            'RISK_COST_BASIS_COLLAPSE', 
            'RISK_CONFIDENCE_DETERIORATION',
            'RISK_CAPITAL_STRUCT_BEARISH_DIVERGENCE',
            'RISK_CAPITAL_STRUCT_MAIN_FORCE_DISTRIBUTING',
            'DYN_TREND_WEAKENING_DECELERATING',
            'DYN_TREND_BEARISH_ACCELERATING',
            'CONTEXT_TREND_DETERIORATING',
            'PLATFORM_FAILURE'
        ]
        for risk_name in strategic_risks:
            if risk_name in self.strategy.atomic_states:
                risk_signals[risk_name] = self.strategy.atomic_states[risk_name]

        print(f"        -> [风险情报总局 V222.0] 情报汇总完毕，共监控 {len(risk_signals)} 类风险信号。")
        return risk_signals

    # 冲高回落侦察连: _diagnose_upthrust_distribution()
    def _diagnose_upthrust_distribution(self, df: pd.DataFrame, exit_params: dict) -> pd.Series:
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

    # 结构破位侦察连: _diagnose_structure_breakdown()
    def _diagnose_structure_breakdown(self, df: pd.DataFrame, exit_params: dict) -> pd.Series:
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

    def _diagnose_dynamic_chip_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V283.6 最终客观版】
        - 核心净化: 移除所有RISK_前缀，本模块只报告客观的动态事实。
        """
        states = {}
        default_series = pd.Series(False, index=df.index)
        
        # --- 1. 检查动态分析所需的所有“弹药”是否到位 ---
        required_cols = [
            'SLOPE_5_concentration_90pct_D', 'ACCEL_5_concentration_90pct_D',
            'SLOPE_5_peak_cost_D', 'ACCEL_5_peak_cost_D',
            'SLOPE_5_total_winner_rate_D', 'ACCEL_5_total_winner_rate_D',
            'SLOPE_5_chip_health_score_D', 'ACCEL_5_chip_health_score_D'
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"            -> [严重警告] 动态分析中心缺少关键数据: {missing_cols}，模块已跳过！")
            return states
        
        # --- 【核心升级】步骤1.5: 获取位置上下文情报 ---
        # 注意：这要求 _diagnose_topping_risks_command 必须在此模块之前运行
        is_in_high_level_zone = self.strategy.atomic_states.get('CONTEXT_RISK_HIGH_LEVEL_ZONE', default_series)

        # --- 步骤2: 对“筹码集中度”进行动态分析 ---
        # 2.1 基础动态
        is_concentrating_trend = df['SLOPE_5_concentration_90pct_D'] < 0
        states['CHIP_DYN_CONCENTRATING'] = is_concentrating_trend
        
        # 2.2 加速动态
        p_chip = get_params_block(self.strategy, 'chip_feature_params')
        accel_threshold = get_param_value(p_chip.get('accel_concentration_threshold'), -0.001)
        is_accelerating_action = df['ACCEL_5_concentration_90pct_D'] < accel_threshold
        states['CHIP_DYN_S_ACCEL_CONCENTRATING'] = is_concentrating_trend & is_accelerating_action
        
        # 2.3 发散动态 (结合了战场上下文)
        is_diverging_action = df['SLOPE_5_concentration_90pct_D'] > 0
        states['CHIP_DYN_DIVERGING'] = is_diverging_action & is_in_high_level_zone
        
        is_accel_diverging_action = df['ACCEL_5_concentration_90pct_D'] > 0
        states['CHIP_DYN_ACCEL_DIVERGING'] = is_accel_diverging_action & is_in_high_level_zone

        # --- 步骤3: 对“筹码成本”进行动态分析 ---
        states['CHIP_DYN_COST_RISING'] = df['SLOPE_5_peak_cost_D'] > 0
        cost_accel_threshold = self.dynamic_thresholds.get('cost_accel_significant', 0.01)
        states['CHIP_DYN_COST_ACCELERATING'] = df['ACCEL_5_peak_cost_D'] > cost_accel_threshold
        states['CHIP_DYN_COST_FALLING'] = df['SLOPE_5_peak_cost_D'] < 0

        # --- 步骤4: 对“总获利盘”进行动态分析 ---
        states['CHIP_DYN_WINNER_RATE_RISING'] = df['SLOPE_5_total_winner_rate_D'] > 0
        winner_rate_collapse_threshold = -1.0
        states['CHIP_DYN_WINNER_RATE_COLLAPSING'] = df['SLOPE_5_total_winner_rate_D'] < winner_rate_collapse_threshold
        states['CHIP_DYN_WINNER_RATE_ACCEL_COLLAPSING'] = df['ACCEL_5_total_winner_rate_D'] < 0

        # --- 步骤5: 对“筹码健康分”进行动态分析 ---
        states['CHIP_DYN_HEALTH_IMPROVING'] = df['SLOPE_5_chip_health_score_D'] > 0
        states['CHIP_DYN_HEALTH_DETERIORATING'] = df['SLOPE_5_chip_health_score_D'] < 0
        
        return states

    def _diagnose_chip_opportunities(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V341.0 新增】高级筹码“机会”情报诊断模块
        - 核心职责: 识别由高级筹码指标揭示的、结构性的看涨机会。
        """
        # print("        -> [高级筹码机会诊断模块 V341.0] 启动...")
        states = {}
        
        # --- 机会1: S级 - 筹码断层新生 (结构性重置) ---
        fault_formed_col = 'is_chip_fault_formed_D'
        if fault_formed_col in df.columns:
            states['OPP_CHIP_FAULT_REBIRTH_S'] = df[fault_formed_col]
            if df[fault_formed_col].any():
                print(f"          -> [情报] 侦测到 {df[fault_formed_col].sum()} 次 S级“筹码断层新生”机会！")

        # --- 机会2: A级 - 高利润安全垫 (持股心态稳定) ---
        profit_margin_col = 'winner_profit_margin_D'
        if profit_margin_col in df.columns:
            # 定义：获利盘的平均利润超过20%，代表持股心态极其稳定
            states['CHIP_STATE_HIGH_PROFIT_CUSHION'] = df[profit_margin_col] > 20.0
        
        return states

    def _diagnose_chip_risks_and_behaviors(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V341.2 加速度增强版】高级筹码“风险与行为”情报诊断模块
        - 核心升级: 引入 turnover_from_winners_ratio_D 的加速度判断，
                    新增 S级“恐慌加速”风险 和 A级“卖盘衰竭”机会，
                    使风险评估体系具备了预测趋势拐点的能力。
        """
        print("        -> [高级筹码风险与行为诊断模块 V341.2 加速度增强版] 启动...")
        states = {}
        default_series = pd.Series(False, index=df.index)

        # --- 1. 军备检查 ---
        # 检查基础指标、斜率和加速度指标
        required_cols = [
            'turnover_from_winners_ratio_D', 'turnover_from_losers_ratio_D', 'pct_change_D',
            'SLOPE_5_turnover_from_winners_ratio_D', 'ACCEL_5_turnover_from_winners_ratio_D',
            'SLOPE_21_turnover_from_winners_ratio_D', 'ACCEL_21_turnover_from_winners_ratio_D',
            'SLOPE_55_turnover_from_winners_ratio_D', 'ACCEL_55_turnover_from_winners_ratio_D'
        ]
        if any(c not in df.columns for c in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            print(f"          -> [警告] 缺少成交量微观结构或其斜率/加速度数据，模块跳过。缺失: {missing_cols}")
            return {}
            
        is_in_high_zone = self.strategy.atomic_states.get('CONTEXT_RISK_HIGH_LEVEL_ZONE', default_series)

        # --- 风险行为1: 获利盘出逃 (分层动态评估) ---
        # 1.1 获取各周期的抛压趋势 (速度)
        is_fleeing_short_term = df['SLOPE_5_turnover_from_winners_ratio_D'] > 0
        is_fleeing_mid_term = df['SLOPE_21_turnover_from_winners_ratio_D'] > 0
        is_fleeing_long_term = df['SLOPE_55_turnover_from_winners_ratio_D'] > 0

        # 1.2 获取抛压趋势的变化 (加速度) - 我们最关心短期的变化
        is_fleeing_accelerating = df['ACCEL_5_turnover_from_winners_ratio_D'] > 0
        is_fleeing_decelerating = df['ACCEL_5_turnover_from_winners_ratio_D'] < 0

        # 1.3 定义风险等级 (基于速度)
        states['RISK_BEHAVIOR_WINNERS_FLEEING_C'] = is_in_high_zone & is_fleeing_short_term
        states['RISK_BEHAVIOR_WINNERS_FLEEING_B'] = is_in_high_zone & is_fleeing_short_term & is_fleeing_mid_term
        states['RISK_BEHAVIOR_WINNERS_FLEEING_A'] = is_in_high_zone & is_fleeing_short_term & is_fleeing_mid_term & is_fleeing_long_term

        # 1.4 【新增】定义S级风险和A级机会 (基于加速度)
        # S级风险 - 恐慌加速: 中期趋势已在出逃，且短期出逃正在加速，这是最危险的信号。
        states['RISK_BEHAVIOR_PANIC_FLEEING_S'] = states['RISK_BEHAVIOR_WINNERS_FLEEING_B'] & is_fleeing_accelerating
        
        # A级机会 - 卖盘衰竭: 获利盘虽然还在卖(短期斜率>0)，但卖出力度已在减弱(加速度<0)。
        states['OPP_BEHAVIOR_SELLING_EXHAUSTION_A'] = is_fleeing_short_term & is_fleeing_decelerating

        # 打印情报
        if states['RISK_BEHAVIOR_PANIC_FLEEING_S'].any():
            print(f"          -> [S级战略风险] 侦测到 {states['RISK_BEHAVIOR_PANIC_FLEEING_S'].sum()} 次“获利盘恐慌加速出逃”！")
        elif states['RISK_BEHAVIOR_WINNERS_FLEEING_A'].any():
            print(f"          -> [A级战略风险] 侦测到 {states['RISK_BEHAVIOR_WINNERS_FLEEING_A'].sum()} 次“长期派发”共振！")
        
        if states['OPP_BEHAVIOR_SELLING_EXHAUSTION_A'].any():
            print(f"          -> [A级机会情报] 侦测到 {states['OPP_BEHAVIOR_SELLING_EXHAUSTION_A'].sum()} 次“卖盘衰竭”信号！")

        # --- 风险行为2: 恐慌盘割肉 (加速赶底) ---
        is_sharp_drop = df['pct_change_D'] < -0.05
        is_panic_selling = df['turnover_from_losers_ratio_D'] > 50.0
        states['RISK_BEHAVIOR_PANIC_SELLING'] = is_sharp_drop & is_panic_selling
        if states['RISK_BEHAVIOR_PANIC_SELLING'].any():
            print(f"          -> [机会情报] 侦测到 {states['RISK_BEHAVIOR_PANIC_SELLING'].sum()} 次“恐慌盘割肉”行为(可能见底)！")
            
        return states

    def _diagnose_trend_dynamics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V174.0 动态惯性引擎】
        - 核心职责: 基于趋势的“斜率”和“加速度”，生成高维度的动态原子状态。
        - 产出: 返回一个包含 DYN_... 信号的字典，供评分引擎使用。
        """
        # print("        -> [诊断模块 V174.0] 正在执行动态惯性诊断...")
        dynamics_states = {}
        default_series = pd.Series(False, index=df.index)

        # --- 1. 获取核心的斜率和加速度数据 ---
        # 长期趋势的速度和加速度
        long_slope_col = 'SLOPE_55_EMA_55_D'
        long_accel_col = 'ACCEL_55_EMA_55_D'
        # 短期趋势的速度
        short_slope_col = 'SLOPE_13_EMA_13_D'

        if not all(c in df.columns for c in [long_slope_col, long_accel_col, short_slope_col]):
            print("          -> [错误] 动态惯性诊断缺少必要的斜率/加速度列，跳过。")
            return {}

        long_slope = df[long_slope_col]
        long_accel = df[long_accel_col]
        short_slope = df[short_slope_col]

        # --- 2. 定义基础布尔条件 ---
        is_long_slope_positive = long_slope > 0
        is_long_slope_negative = long_slope < 0
        is_long_accel_positive = long_accel > 0
        is_long_accel_negative = long_accel < 0
        
        is_short_slope_positive = short_slope > 0
        is_short_slope_negative = short_slope < 0
        
        # --- 3. 组合生成高维度动态状态 ---
        # 【S级进攻】健康加速: 长短期趋势同向看涨，且长期趋势在加速
        dynamics_states['DYN_TREND_HEALTHY_ACCELERATING'] = is_long_slope_positive & is_short_slope_positive & is_long_accel_positive
        
        # 【A级进攻】成熟稳定: 长短期趋势同向看涨，但长期趋势已不再加速（减速或匀速）
        dynamics_states['DYN_TREND_MATURE_STABLE'] = is_long_slope_positive & is_short_slope_positive & ~is_long_accel_positive

        # 【B级进攻】底部反转: 长期趋势仍向下，但短期趋势已率先加速向上
        dynamics_states['DYN_TREND_BOTTOM_REVERSING'] = is_long_slope_negative & is_short_slope_positive & (df[short_slope_col] > df[short_slope_col].shift(1))

        # 【S级风险】动能衰减: 长期趋势虽向上，但已开始减速，且短期趋势已逆转
        dynamics_states['DYN_TREND_WEAKENING_DECELERATING'] = is_long_slope_positive & is_long_accel_negative & is_short_slope_negative

        # 【A级风险】下跌加速: 长短期趋势同向看跌，且下跌在加速
        dynamics_states['DYN_TREND_BEARISH_ACCELERATING'] = is_long_slope_negative & is_short_slope_negative & is_long_accel_negative

        # 【B级风险】顶部背离: 价格创近期新高，但长短期斜率均在下降
        is_new_high = df['high_D'] >= df['high_D'].shift(1).rolling(window=10).max()
        is_slope_weakening = (long_slope < long_slope.shift(1)) & (short_slope < short_slope.shift(1))
        dynamics_states['DYN_TREND_TOPPING_DIVERGENCE'] = is_new_high & is_slope_weakening

        # --- 4. 打印调试信息 ---
        # for name, series in dynamics_states.items():
        #     print(f"          -> “{name}” 已定义，激活 {series.sum()} 天。")
            
        return dynamics_states

    def _diagnose_trend_stage_context(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
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

    def _diagnose_capital_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V219.0 情报一体化版】 - 统一资本动向总参谋部
        - 核心升级: 将原 `_diagnose_capital_structure_states` 的职能（主力/散户资金分析）
                    并入此模块，形成统一的、从宏观到微观的资本分析中心。
        - 作战单元1 (宏观气象站): 保留基于CMF的经典资本状态诊断。
        - 作战单元2 (精锐侦察连): 新增基于主力/散户净流入的、高精度的资本结构诊断。
        """
        # print("        -> [诊断模块 V219.0 情报一体化版] 正在执行统一资本诊断...")
        states = {}
        default_series = pd.Series(False, index=df.index)
        
        # 1. 定义“主力大规模派发”的单日行为
        #    - 从配置中读取参数，例如单日净流出超过2000万就视为危险
        dist_context_params = get_params_block(self.strategy, 'distribution_context_params', {})
        outflow_threshold = get_param_value(dist_context_params.get('outflow_threshold_M'), -20) * 1_000_000
        
        outflow_threshold = self.dynamic_thresholds.get('main_force_significant_outflow', -20_000_000)
        is_distribution_day = df.get('main_force_net_inflow_amount_D', default_series) < outflow_threshold

        # 2. 建立“战场记忆”：使用滚动窗口检查近期是否发生过派发
        #    - 如果过去10天内，有任何一天是“派发日”，那么今天就处于“近期派发压力”之下
        lookback_window = get_param_value(dist_context_params.get('lookback_days'), 10)
        # .any() 是关键，它检查窗口内是否有至少一个 True
        self.strategy.atomic_states['CONTEXT_RECENT_DISTRIBUTION_PRESSURE'] = is_distribution_day.rolling(
            window=lookback_window, min_periods=1
        ).apply(np.any, raw=True).fillna(0).astype(bool)
        
        # print(f"        -> [资本动向总参谋部] “危险战区”感知模块已启动。{format_debug_dates(self.strategy.atomic_states['CONTEXT_RECENT_DISTRIBUTION_PRESSURE'])}")
        
        # --- 作战单元1: 经典资本状态诊断 (基于CMF) ---
        capital_params = get_params_block(self.strategy, 'capital_state_params')
        if get_param_value(capital_params.get('enabled'), True):
            cmf_bullish_threshold = get_param_value(capital_params.get('cmf_bullish_threshold'), 0.05)
            states['CAPITAL_STATE_INFLOW_CONFIRMED'] = df.get('CMF_21_D', 0) > cmf_bullish_threshold

            divergence_context = capital_params.get('divergence_context', {})
            persistence_days = get_param_value(divergence_context.get('persistence_days'), 15)
            trend_ma_period = get_param_value(divergence_context.get('trend_ma_period'), 55)

            price_new_high = df['close_D'] > df['close_D'].shift(1).rolling(window=persistence_days).max()
            cmf_not_new_high = df['CMF_21_D'] < df['CMF_21_D'].shift(1).rolling(window=persistence_days).max()
            is_uptrend = df['close_D'] > df.get(f'EMA_{trend_ma_period}_D', 0)

            states['CAPITAL_STATE_DIVERGENCE_WINDOW'] = price_new_high & cmf_not_new_high & is_uptrend

        return states

    def _diagnose_market_structure_command(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V272.0 市场结构战区司令部】
        - 核心重构: 这不是一次合并，而是一次战略整合。
        - 新架构:
          1. 本司令部统一指挥下属的三个专业化兵种：均线野战部队、价格工兵部队、筹码特种侦察部队。
          2. 它首先收集所有基础的“原子结构情报”。
          3. 然后，它进行情报融合，生成更高维度的、包含协同作战思想的“复合结构情报”。
        - 收益: 极大地提升了代码的组织性和可读性，并能产出远比单个模块更有价值的协同信号。
        """
        # print("        -> [市场结构战区司令部 V272.0] 启动，正在整合全战场结构情报...")
        
        # --- 1. 依次调动下属的专业化兵种，收集原子情报 ---
        # print("          -> 正在调动：均线野战部队、价格工兵部队、筹码特种侦察部队...")
        ma_states = self._diagnose_ma_states(df)
        box_states = self._diagnose_box_states(df)
        df, platform_states = self._diagnose_platform_states(df) # 平台诊断会修改df，需要接收
        
        # 将所有原子情报汇总
        atomic_structure_states = {**ma_states, **box_states, **platform_states}
        
        # --- 2. 进行情报融合与战术研判，生成复合情报 ---
        # print("          -> 正在进行情报融合，生成高维度复合情报...")
        composite_states = {}
        default_series = pd.Series(False, index=df.index)

        # 复合情报1: “阵地优势” - 一个稳固的平台，必须得到动态趋势线的确认
        is_platform_stable = atomic_structure_states.get('PLATFORM_STATE_STABLE_FORMED', default_series)
        is_above_mid_ma = atomic_structure_states.get('MA_STATE_PRICE_ABOVE_MID_MA', default_series)
        composite_states['STRUCTURE_PLATFORM_WITH_TREND_SUPPORT'] = is_platform_stable & is_above_mid_ma

        # 复合情报2: “健康盘整” - 一个箱体整理，必须发生在关键趋势线上方
        is_in_box = atomic_structure_states.get('BOX_STATE_CONSOLIDATING', default_series)
        is_above_long_ma = atomic_structure_states.get('MA_STATE_PRICE_ABOVE_LONG_MA', default_series)
        composite_states['STRUCTURE_BOX_ABOVE_TRENDLINE'] = is_in_box & is_above_long_ma

        # 复合情报3: “突破前夜” (S级战术信号) - 极致的共振信号
        # 定义：价格被压缩在一个健康的箱体内，而这个箱体本身就建立在一个稳固的筹码平台上，
        #       同时，整个结构都位于主升趋势线之上。这是大战一触即发的终极信号！
        is_healthy_box = composite_states.get('STRUCTURE_BOX_ABOVE_TRENDLINE', default_series)
        is_on_stable_platform = atomic_structure_states.get('PLATFORM_STATE_STABLE_FORMED', default_series)
        composite_states['STRUCTURE_BREAKOUT_EVE_S'] = is_healthy_box & is_on_stable_platform

        # print("        -> [市场结构战区司令部 V272.0] 情报整合完毕。")
        
        # 返回所有原子情报和复合情报的集合，以及可能被修改的df
        return df, {**atomic_structure_states, **composite_states}

    def _diagnose_ma_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V283.0 融合思想版】
        - 核心升级: 引入“攻击性多头排列”信号。
          - MA_STATE_STABLE_BULLISH: 基础的多头排列状态 (静态位置)。
          - MA_STATE_AGGRESSIVE_BULLISH: 在多头排列基础上，要求短期均线斜率陡峭 (动态趋势)，是更高质量的信号。
        """
        # print("          -> [均线野战部队 V283.0] 启动，正在执行融合分析...")
        states = {}
        p = get_params_block(self.strategy, 'ma_state_params')
        if not get_param_value(p.get('enabled'), False): return states

        # --- 0. 读取参数并检查数据完整性 ---
        short_ma_period = get_param_value(p.get('short_ma'), 13)
        mid_ma_period = get_param_value(p.get('mid_ma'), 21)
        long_ma_period = get_param_value(p.get('long_ma'), 55)

        short_ma = f'EMA_{short_ma_period}_D'
        mid_ma = f'EMA_{mid_ma_period}_D'
        long_ma = f'EMA_{long_ma_period}_D'
        short_ma_slope_col = f'SLOPE_5_{short_ma}'
        long_ma_slope_col = f'SLOPE_21_{long_ma}'
        
        required_cols = [short_ma, mid_ma, long_ma, short_ma_slope_col, long_ma_slope_col]
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            print(f"          -> [警告] 缺少诊断MA状态所需列: {missing_cols}，跳过。")
            return states

        # --- 1. 静态位置分析 (价格与均线的关系) ---
        states['MA_STATE_PRICE_ABOVE_SHORT_MA'] = df['close_D'] > df[short_ma]
        states['MA_STATE_PRICE_ABOVE_MID_MA'] = df['close_D'] > df[mid_ma]
        states['MA_STATE_PRICE_ABOVE_LONG_MA'] = df['close_D'] > df[long_ma]
        
        # 基础的“稳定多头/空头排列”信号 (纯静态位置判断)
        stable_bullish = (df[short_ma] > df[mid_ma]) & (df[mid_ma] > df[long_ma])
        states['MA_STATE_STABLE_BULLISH'] = stable_bullish
        states['MA_STATE_STABLE_BEARISH'] = (df[short_ma] < df[mid_ma]) & (df[mid_ma] < df[long_ma])

        # --- 2. 动态趋势分析 (均线斜率) ---
        states['MA_STATE_SHORT_SLOPE_POSITIVE'] = df[short_ma_slope_col] > 0
        states['MA_STATE_LONG_SLOPE_POSITIVE'] = df[long_ma_slope_col] > 0

        # --- 3. 【融合生成】高质量信号 ---
        # “攻击性多头排列” (S级信号): 在稳定多头排列的基础上，要求短期均线斜率陡峭
        # 从配置中读取“陡峭”的定义，若无则使用默认值0.01
        aggressive_slope_threshold = get_param_value(p.get('aggressive_slope_threshold'), 0.01)
        is_aggressive_slope = df[short_ma_slope_col] > aggressive_slope_threshold
        states['MA_STATE_AGGRESSIVE_BULLISH'] = stable_bullish & is_aggressive_slope
        
        # --- 4. 其他复合状态分析 ---
        # 均线发散与收敛 (使用Z-score)
        zscore_col = 'MA_ZSCORE_D' # 假设这个Z-score在数据工程层计算
        if zscore_col in df.columns:
            states['MA_STATE_CONVERGING'] = df[zscore_col] < get_param_value(p.get('converging_zscore'), -1.0)
            states['MA_STATE_DIVERGING'] = df[zscore_col] > get_param_value(p.get('diverging_zscore'), 1.0)

        # “均线钝化企稳”的持续性状态 (左侧信号)
        long_ma_slope = df[long_ma_slope_col]
        entry_event = (long_ma_slope >= 0) & (long_ma_slope.shift(1) < 0)
        
        short_ma_slope = df[short_ma_slope_col]
        break_condition = short_ma_slope < -0.005 # 允许轻微波动，但明确下跌则打破

        states['MA_STATE_BOTTOM_PASSIVATION'] = create_persistent_state(
            df=df,
            entry_event_series=entry_event,
            persistence_days=20, # 钝化状态最长可以持续20天
            break_condition_series=break_condition,
            state_name='MA_STATE_BOTTOM_PASSIVATION'
        )

        # --- 5. 【新增】计算动态均线粘合压缩状态 ---
        p_conv = get_params_block(self.strategy, 'post_accumulation_params').get('convergence_params', {})
        if get_param_value(p_conv.get('use_dynamic_threshold'), False):
            window = get_param_value(p_conv.get('window'), 120)
            quantile = get_param_value(p_conv.get('quantile'), 0.1)
            
            # ▼▼▼【核心修复】使用更健壮的方式寻找列名 ▼▼▼
            short_cv_col = next((col for col in df.columns if 'MA_CONV_CV_SHORT' in col), None)
            long_cv_col = next((col for col in df.columns if 'MA_CONV_CV_LONG' in col), None)
            
            if short_cv_col:
                dynamic_threshold_short = df[short_cv_col].rolling(window=window).quantile(quantile)
                states['MA_STATE_SHORT_CONVERGENCE_SQUEEZE'] = df[short_cv_col] < dynamic_threshold_short
            
            if long_cv_col:
                dynamic_threshold_long = df[long_cv_col].rolling(window=window).quantile(quantile)
                states['MA_STATE_LONG_CONVERGENCE_SQUEEZE'] = df[long_cv_col] < dynamic_threshold_long

        return states

    def _diagnose_box_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V283.0 融合思想版】
        - 核心升级: 引入“健康吸筹箱体”信号。
          - BOX_STATE_HEALTHY_CONSOLIDATION: 基础的、位于趋势线上方的箱体 (静态)。
          - BOX_STATE_HEALTHY_ACCUMULATION: 在健康箱体内，要求同时满足“缩量”和“筹码集中” (动态)，是更高质量的突破前兆。
        """
        # print("        -> [工兵部队 V283.0] 启动，正在执行融合分析...")
        states = {}
        box_params = get_params_block(self.strategy, 'dynamic_box_params')
        if not get_param_value(box_params.get('enabled'), False) or df.empty:
            return states

        # --- 1. 静态分析：识别箱体结构 ---
        lookback_window = get_param_value(box_params.get('lookback_window'), 8)
        max_amplitude_ratio = get_param_value(box_params.get('max_amplitude_ratio'), 0.05)
        rolling_high = df['high_D'].rolling(window=lookback_window).max()
        rolling_low = df['low_D'].rolling(window=lookback_window).min()
        amplitude_ratio = (rolling_high - rolling_low) / rolling_low.replace(0, np.nan)
        is_valid_box = (amplitude_ratio < max_amplitude_ratio).fillna(False)
        
        box_top = rolling_high
        box_bottom = rolling_low

        # --- 2. 定义基础事件和状态 ---
        was_below_top = df['close_D'].shift(1) <= box_top.shift(1)
        is_above_top = df['close_D'] > box_top
        states['BOX_EVENT_BREAKOUT'] = is_valid_box & is_above_top & was_below_top
        
        was_above_bottom = df['close_D'].shift(1) >= box_bottom.shift(1)
        is_below_bottom = df['close_D'] < box_bottom
        states['BOX_EVENT_BREAKDOWN'] = is_valid_box & is_below_bottom & was_above_bottom
        
        is_in_box = (df['close_D'] < box_top) & (df['close_D'] > box_bottom)
        
        # 基础的“健康箱体”
        ma_params = get_params_block(self.strategy, 'ma_state_params')
        mid_ma_period = get_param_value(ma_params.get('mid_ma'), 55)
        mid_ma_col = f"EMA_{mid_ma_period}_D"
        if mid_ma_col in df.columns:
            box_midpoint = (box_top + box_bottom) / 2
            is_box_above_ma = box_midpoint > df[mid_ma_col]
            healthy_consolidation = is_valid_box & is_in_box & is_box_above_ma
        else:
            healthy_consolidation = is_valid_box & is_in_box
        states['BOX_STATE_HEALTHY_CONSOLIDATION'] = healthy_consolidation

        # --- 3. 【融合生成】高质量信号 ---
        # “健康吸筹箱体” (S级信号): 在健康箱体内，要求缩量+筹码集中
        is_shrinking_volume = self.strategy.atomic_states.get('VOL_STATE_SHRINKING', pd.Series(False, index=df.index))
        is_chip_concentrating = self.strategy.atomic_states.get('CHIP_DYN_CONCENTRATING', pd.Series(False, index=df.index))
        states['BOX_STATE_HEALTHY_ACCUMULATION'] = healthy_consolidation & is_shrinking_volume & is_chip_concentrating
        
        for key in states:
            if key not in states or states[key] is None:
                states[key] = pd.Series(False, index=df.index)
            else:
                states[key] = states[key].fillna(False)
        return states

    def _diagnose_platform_states(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        【V129.2 健壮部署版 - 筹码平台诊断模块】
        - 核心修复: 修正了对筹码和价格列的引用，不再使用错误的 'CHIP_' 前缀，
                    确保能够正确从数据层获取 'peak_cost_D' 和 'close_D'。
        - 功能增强: 增加了更详细的日志输出和更强的防御性编程，确保在缺少数据时
                    能够优雅地处理并返回标准化的空结果，防止下游模块出错。
        """
        # print("        -> [诊断模块 V129.2] 正在执行筹码平台状态诊断...")
        states = {}
        default_series = pd.Series(False, index=df.index)

        # --- 步骤1: 检查核心数据是否存在 ---
        peak_cost_col = 'peak_cost_D'
        close_col = 'close_D'
        long_ma_col = 'EMA_55_D' # 平台必须位于长期均线上方才有意义
        
        required_cols = [peak_cost_col, close_col, long_ma_col]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"          -> [警告] 缺少诊断平台状态所需的核心列: {missing}。模块将返回空结果。")
            # 即使失败，也要确保返回标准化的输出结构，防止下游模块调用失败
            df['PLATFORM_PRICE_STABLE'] = np.nan
            states['PLATFORM_STATE_STABLE_FORMED'] = default_series
            states['PLATFORM_FAILURE'] = default_series
            return df, states

        # --- 步骤2: 定义并计算“稳固平台形成”状态 ---
        # 条件A: 筹码峰成本在短期内高度稳定 (滚动5日的标准差/均值 < 2%)
        is_cost_stable = (df[peak_cost_col].rolling(5).std() / df[peak_cost_col].rolling(5).mean()) < 0.02
        
        # 条件B: 当前价格位于长期趋势均线之上，确保平台处于上升趋势中
        is_above_long_ma = df[close_col] > df[long_ma_col]
        
        # 组合成最终的“稳固平台形成”状态
        stable_formed_series = is_cost_stable & is_above_long_ma
        states['PLATFORM_STATE_STABLE_FORMED'] = stable_formed_series
        
        # --- 步骤3: 将有效的平台价格记录下来，供后续模块使用 ---
        # 只有在平台形成当天，才记录下当天的平台价格，否则为NaN
        df['PLATFORM_PRICE_STABLE'] = df[peak_cost_col].where(stable_formed_series)
        
        # --- 步骤4: 定义并计算“平台破位”风险 ---
        # 条件A: 昨日处于稳固平台之上
        was_on_platform = stable_formed_series.shift(1).fillna(False)
        
        # 条件B: 今日收盘价跌破了昨日的平台价格
        # 使用 ffill() 填充平台价格，以处理平台形成后、破位前的那些天
        stable_platform_price_series = df['PLATFORM_PRICE_STABLE'].ffill()
        is_breaking_down = df[close_col] < stable_platform_price_series.shift(1)
        
        # 组合成最终的“平台破位”风险信号
        platform_failure_series = was_on_platform & is_breaking_down
        states['STRUCTURE_PLATFORM_BROKEN'] = platform_failure_series

        # --- 步骤5: 打印诊断日志 ---
        # print(f"          -> '稳固平台形成' 状态诊断完成，共激活 {stable_formed_series.sum()} 天。")
        # print(f"          -> '平台破位' 风险诊断完成，共激活 {platform_failure_series.sum()} 天。")

        return df, states

    def _diagnose_structural_mechanics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 - 赫斯特指数增强版】
        - 核心升级: 深度集成赫斯特指数，从中提炼出多种关于市场宏观阶段、
                    趋势可持续性与衰竭的原子状态。
        """
        # print("        -> [结构力学诊断模块 V2.0] 启动，正在进行物理建模与分形分析...")
        from utils.math_tools import hurst_exponent
        states = {}
        
        # --- 1. 成本质心分析 (逻辑不变) ---
        if 'SLOPE_5_peak_cost_D' in df.columns and 'ACCEL_5_peak_cost_D' in df.columns:
            states['MECHANICS_COST_RISING'] = df['SLOPE_5_peak_cost_D'] > 0
            states['MECHANICS_COST_ACCELERATING'] = df['ACCEL_5_peak_cost_D'] > 0.01

        # --- 2. 结构转动惯量分析 (逻辑不变) ---
        if 'SLOPE_5_concentration_90pct_D' in df.columns:
            states['MECHANICS_INERTIA_DECREASING'] = df['SLOPE_5_concentration_90pct_D'] < -0.001
        
        # --- 3. 结构势能分析 (逻辑不变) ---
        if 'support_below_D' in df.columns and 'pressure_above_D' in df.columns:
            energy_ratio = df['support_below_D'] / (df['pressure_above_D'] + 1e-6)
            states['MECHANICS_ENERGY_ADVANTAGE'] = energy_ratio > 1.5

        # --- 4. 【核心改造】分形与混沌分析 ---
        hurst_window = self.strategy.unified_config.get('hurst_window', 60) # 从配置读取窗口期
        
        try:
            # 4.1 计算赫斯特指数
            hurst_col = f'hurst_{hurst_window}d'
            df[hurst_col] = df['close_D'].rolling(hurst_window).apply(hurst_exponent, raw=True)
            
            # 4.2 定义绝对状态 (Absolute State)
            # 强趋势状态: H > 0.7，表明市场具有很强的记忆性，趋势很可能会持续
            states['FRACTAL_STATE_STRONG_TREND'] = df[hurst_col] > 0.7
            # 均值回归状态: H < 0.4，表明市场倾向于反向运动，是震荡市的特征
            states['FRACTAL_STATE_MEAN_REVERSION'] = df[hurst_col] < 0.4
            # 随机游走状态: 介于两者之间，市场方向不明
            states['FRACTAL_STATE_RANDOM_WALK'] = (df[hurst_col] >= 0.4) & (df[hurst_col] <= 0.7)

            # 4.3 定义动态变化 (Dynamic Change)
            # 趋势形成事件: H指数从“均值回归/随机”区域，向上突破了趋势阈值(0.7)
            was_not_trending = df[hurst_col].shift(1) <= 0.7
            is_trending = df[hurst_col] > 0.7
            states['FRACTAL_EVENT_TREND_FORMING'] = was_not_trending & is_trending
            
            # 趋势衰竭事件: H指数从“强趋势”区域，向下跌破了随机游走上沿(0.7)
            was_trending = df[hurst_col].shift(1) > 0.7
            is_not_trending = df[hurst_col] <= 0.7
            states['FRACTAL_EVENT_TREND_EXHAUSTION'] = was_trending & is_not_trending

            # 基于斜率
            hurst_slope_col = f'SLOPE_5_hurst_{hurst_window}d_D'
            if hurst_slope_col in df.columns:
                # 趋势加速形成: H值本身在上升，且斜率也为正
                states['FRACTAL_DYN_TREND_ACCELERATING'] = (df[hurst_col] > df[hurst_col].shift(1)) & (df[hurst_slope_col] > 0)
                # 趋势加速衰竭: H值本身在下降，且斜率为负
                states['FRACTAL_DYN_TREND_DECELERATING'] = (df[hurst_col] < df[hurst_col].shift(1)) & (df[hurst_slope_col] < 0)

            # 4.4 定义组合信号 (Combined Signal)
            # 【S级机会】波动率压缩 + 趋势形成: 
            # 这是最经典的“Squeeze Breakout”模式的深层确认。
            # 能量在极度压缩后，市场结构从随机/震荡，转变为强趋势，是最高质量的突破信号。
            is_in_squeeze = self.strategy.atomic_states.get('FRACTAL_VOLATILITY_SQUEEZE', pd.Series(False, index=df.index))
            states['FRACTAL_OPP_SQUEEZE_BREAKOUT_CONFIRMED'] = is_in_squeeze.shift(1) & states['FRACTAL_EVENT_TREND_FORMING']

            # 【S级风险】价格新高 + 趋势衰竭 (顶背离)
            # 价格还在创近期新高，但驱动趋势的内在结构已经瓦解，是极其危险的顶背离信号。
            is_new_high = df['high_D'] >= df['high_D'].shift(1).rolling(window=20).max()
            states['FRACTAL_RISK_TOP_DIVERGENCE'] = is_new_high & states['FRACTAL_EVENT_TREND_EXHAUSTION']

        except Exception as e:
            print(f"          -> [警告] 赫斯特指数计算或状态诊断失败: {e}")

        # 波动率压缩 (变异系数) - 逻辑不变
        price_mean = df['close_D'].rolling(60).mean()
        price_std = df['close_D'].rolling(60).std()
        price_cv = price_std / (price_mean + 1e-6) # 加上一个极小值防止除以0

        # 当日波动率处于过去120天的最低10%水平
        is_in_squeeze = price_cv < price_cv.rolling(120).quantile(0.1)
        states['FRACTAL_VOLATILITY_SQUEEZE'] = is_in_squeeze


        return states

    def _diagnose_kline_patterns(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
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

    def _diagnose_board_patterns(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
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

    def _diagnose_market_structure_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V277.0 五重共振版 - 联合作战司令部】
        - 核心升级:
          1. 将 `is_dyn_trend_healthy` (动能) 重新引入S级主升浪的定义，与 `is_ma_bullish` (结构) 形成黄金搭档。
          2. 清除了已被取代的、未被使用的 `is_chip_health_good` 变量，保持代码的绝对整洁。
        - 新定义: S级主升浪现在是“结构+动能+筹码+资金+位置”的五重共振，是理论上最强的做多信号。
        - 收益: S级信号的含金量达到顶峰，误报率被进一步压缩，代码逻辑更加严谨。
        """
        # print("        -> [联合作战司令部 V277.0 五重共振版] 启动，正在打造终极S级战局信号...")
        structure_states = {}
        default_series = pd.Series(False, index=df.index)

        # --- 步骤1：情报总览 (精确调阅，杜绝浪费) ---
        is_ma_bullish = self.strategy.atomic_states.get('MA_STATE_STABLE_BULLISH', default_series)
        is_ma_bearish = self.strategy.atomic_states.get('MA_STATE_STABLE_BEARISH', default_series)
        is_ma_converging = self.strategy.atomic_states.get('MA_STATE_CONVERGING', default_series)
        is_price_above_long_ma = self.strategy.atomic_states.get('MA_STATE_PRICE_ABOVE_LONG_MA', default_series)
        is_recent_reversal = self.strategy.atomic_states.get('CONTEXT_RECENT_REVERSAL_SIGNAL', default_series)
        is_ma_short_slope_positive = self.strategy.atomic_states.get('MA_STATE_SHORT_SLOPE_POSITIVE', default_series)
        is_dyn_trend_healthy = self.strategy.atomic_states.get('DYN_TREND_HEALTHY_ACCELERATING', default_series) # 【修正】重新启用此关键情报
        is_dyn_trend_weakening = self.strategy.atomic_states.get('DYN_TREND_WEAKENING_DECELERATING', default_series)
        is_chip_concentrating = self.strategy.atomic_states.get('CHIP_RAPID_CONCENTRATION', default_series)
        is_chip_health_excellent = self.strategy.atomic_states.get('CHIP_HEALTH_EXCELLENT', default_series)
        is_chip_health_deteriorating = self.strategy.atomic_states.get('CHIP_HEALTH_DETERIORATING', default_series)
        is_fund_flow_consensus_inflow = self.strategy.atomic_states.get('CHIP_FUND_FLOW_CONSENSUS_INFLOW', default_series)
        is_fund_flow_consensus_outflow = self.strategy.atomic_states.get('CHIP_FUND_FLOW_CONSENSUS_OUTFLOW', default_series)
        is_capital_bearish_divergence = self.strategy.atomic_states.get('RISK_CAPITAL_STRUCT_BEARISH_DIVERGENCE', default_series)
        is_vol_squeeze = self.strategy.atomic_states.get('VOL_STATE_SQUEEZE_WINDOW', default_series)

        # --- 步骤2：联合裁定 (识别五大经典战局) ---

        # 【战局1: S级主升浪·黄金航道】 - 五重力量的完美共振
        structure_states['STRUCTURE_MAIN_UPTREND_WAVE_S'] = (
            is_ma_bullish &                          # 1. 结构: 完美多头排列
            is_dyn_trend_healthy &                   # 2. 动能: 趋势正在健康加速
            is_chip_health_excellent &               # 3. 筹码: 王牌部队状态极佳
            is_fund_flow_consensus_inflow &          # 4. 资金: 主力部队共识性流入
            is_price_above_long_ma                   # 5. 位置: 占据战略制高点
        )

        # 【战局2: A级突破前夜·能量压缩】 - 大战前的寂静
        structure_states['STRUCTURE_BREAKOUT_EVE_A'] = (
            is_vol_squeeze &
            is_chip_concentrating &
            is_ma_converging &
            is_price_above_long_ma
        )

        # 【战局3: B级反转初期·黎明微光】 - 从左侧到右侧的脆弱过渡
        structure_states['STRUCTURE_EARLY_REVERSAL_B'] = (
            is_recent_reversal &
            is_ma_short_slope_positive
        )

        # 【战局4: S级风险·顶部背离】 - 最危险的诱多陷阱
        structure_states['STRUCTURE_TOPPING_DANGER_S'] = (
            is_capital_bearish_divergence |
            is_chip_health_deteriorating
        )

        # 【战局5: F级禁区·下跌通道】 - 绝对的回避区域
        structure_states['STRUCTURE_BEARISH_CHANNEL_F'] = (
            is_ma_bearish &
            is_dyn_trend_weakening &
            is_fund_flow_consensus_outflow
        )

        # print("        -> [联合作战司令部 V277.0] 核心战局定义升级完成。")
        return structure_states

    def _diagnose_volume_price_dynamics(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
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
   
    def _determine_main_force_behavior_sequence(self, df: pd.DataFrame) -> pd.DataFrame:
        # print("    --- [战略推演单元 V304.0] 启动，正在生成主力行为序列... ---")
        df['main_force_state'] = MainForceState.IDLE.value
        for i in range(1, len(df)):
            prev_state_val = df.at[df.index[i-1], 'main_force_state']
            prev_state = MainForceState(prev_state_val)
            s = {
                'is_concentrating': self.strategy.atomic_states.get('CHIP_DYN_CONCENTRATING', pd.Series(False, index=df.index)).iloc[i],
                'is_mild_inflow': self.strategy.atomic_states.get('CAPITAL_MAIN_FORCE_INFLOW_MILD', pd.Series(False, index=df.index)).iloc[i],
                'is_sharp_drop': self.strategy.atomic_states.get('RISK_PRICE_SHARP_DROP_S', pd.Series(False, index=df.index)).iloc[i],
                'is_sideways': abs(df.get('SLOPE_5_close_D', pd.Series(0, index=df.index)).iloc[i]) < 0.01,
                'is_markup_breakout': self.strategy.atomic_states.get('BREAKOUT_S', pd.Series(False, index=df.index)).iloc[i],
                'is_chip_fault': self.strategy.atomic_states.get('CHIP_FAULT_FORMED', pd.Series(False, index=df.index)).iloc[i],
                'is_strong_inflow': self.strategy.atomic_states.get('CAPITAL_MAIN_FORCE_INFLOW_STRONG', pd.Series(False, index=df.index)).iloc[i],
                'is_distributing': self.strategy.atomic_states.get('RISK_CAPITAL_STRUCT_MAIN_FORCE_DISTRIBUTING', pd.Series(False, index=df.index)).iloc[i],
                'is_diverging': self.strategy.atomic_states.get('RISK_DYN_DIVERGING', pd.Series(False, index=df.index)).iloc[i],
                'is_stagnation': self.strategy.atomic_states.get('RISK_VPA_STAGNATION', pd.Series(False, index=df.index)).iloc[i],
                'is_below_long_ma': df.at[df.index[i], 'close_D'] < df.at[df.index[i], 'EMA_55_D'],
            }
            current_state = prev_state
            if prev_state == MainForceState.IDLE:
                if s['is_concentrating'] and s['is_mild_inflow']: current_state = MainForceState.ACCUMULATING
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
        # print("    --- [战略推演单元 V304.0] 主力行为序列已生成。 ---")
        return df

    def _define_trigger_events(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V234.0 最终净化版 - 战术触发事件定义中心】
        - 核心升级: 严格遵循“V234.0 作战条例”，所有参数均从唯一的 trigger_event_params 配置块中获取，
                    确保了配置的单一来源原则，使整个触发体系清晰、健壮、易于维护。
        - 职责: 识别所有可以作为“开火信号”的瞬时战术事件(Trigger)。
        """
        # print("        -> [触发事件中心 V234.0] 启动，正在定义所有原子化触发事件...")
        triggers = {}
        default_series = pd.Series(False, index=df.index)
        
        # ▼▼▼【代码修改 V234.0】: 统一从 trigger_event_params 获取所有参数 ▼▼▼
        trigger_params = get_params_block(self.strategy, 'trigger_event_params')
        if not get_param_value(trigger_params.get('enabled'), True):
            print("          -> 触发事件引擎被禁用，跳过。")
            return triggers
        # ▲▲▲【代码修改 V234.0】▲▲▲
            
        vol_ma_col = 'VOL_MA_21_D'

        # --- 1. K线形态触发器 (Candlestick Triggers) ---
        # 1.1 【通用级】反转确认阳线
        p_reversal = trigger_params.get('reversal_confirmation_candle', {})
        if get_param_value(p_reversal.get('enabled'), True):
            is_green = df['close_D'] > df['open_D']
            is_strong_rally = df['pct_change_D'] > get_param_value(p_reversal.get('min_pct_change'), 0.03)
            is_closing_strong = df['close_D'] > (df['high_D'] + df['low_D']) / 2
            triggers['TRIGGER_REVERSAL_CONFIRMATION_CANDLE'] = is_green & is_strong_rally & is_closing_strong

        # 1.2 【精英级】显性反转阳线 (在通用级基础上，要求力量压制前一日)
        p_dominant = trigger_params.get('dominant_reversal_candle', {})
        if get_param_value(p_dominant.get('enabled'), True):
            base_reversal_signal = triggers.get('TRIGGER_REVERSAL_CONFIRMATION_CANDLE', default_series)
            today_body_size = df['close_D'] - df['open_D']
            yesterday_body_size = abs(df['close_D'].shift(1) - df['open_D'].shift(1))
            was_yesterday_red = df['close_D'].shift(1) < df['open_D'].shift(1)
            recovery_ratio = get_param_value(p_dominant.get('recovery_ratio'), 0.5)
            is_power_recovered = today_body_size >= (yesterday_body_size * recovery_ratio)
            triggers['TRIGGER_DOMINANT_REVERSAL'] = base_reversal_signal & (~was_yesterday_red | is_power_recovered)

        # 1.3 【企稳型】突破阳线 (通常用于底部企稳或平台整理后的首次突破)
        p_breakout_candle = trigger_params.get('breakout_candle', {})
        if get_param_value(p_breakout_candle.get('enabled'), True):
            boll_mid_col = 'BBM_21_2.0_D'
            if boll_mid_col in df.columns:
                min_body_ratio = get_param_value(p_breakout_candle.get('min_body_ratio'), 0.4)
                body_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
                is_strong_positive_candle = (
                    (df['close_D'] > df['open_D']) &
                    (((df['close_D'] - df['open_D']) / body_range).fillna(1.0) >= min_body_ratio)
                )
                is_breaking_boll_mid = df['close_D'] > df[boll_mid_col]
                triggers['TRIGGER_BREAKOUT_CANDLE'] = is_strong_positive_candle & is_breaking_boll_mid

        # 1.4 【进攻型】能量释放阳线 (强调实体和成交量的双重确认)
        p_energy = trigger_params.get('energy_release', {})
        if get_param_value(p_energy.get('enabled'), True) and vol_ma_col in df.columns:
            is_positive_day = df['close_D'] > df['open_D']
            body_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
            body_ratio = (df['close_D'] - df['open_D']) / body_range
            is_strong_body = body_ratio.fillna(1.0) > get_param_value(p_energy.get('min_body_ratio'), 0.5)
            volume_ratio = get_param_value(p_energy.get('volume_ratio'), 1.5)
            is_volume_spike = df['volume_D'] > df[vol_ma_col] * volume_ratio
            triggers['TRIGGER_ENERGY_RELEASE'] = is_positive_day & is_strong_body & is_volume_spike

        # --- 2. 结构与趋势触发器 (Structure & Trend Triggers) ---
        # 2.1 【经典】放量突破近期高点
        p_vol_breakout = trigger_params.get('volume_spike_breakout', {})
        if get_param_value(p_vol_breakout.get('enabled'), True) and vol_ma_col in df.columns:
            volume_ratio = get_param_value(p_vol_breakout.get('volume_ratio'), 2.0)
            lookback = get_param_value(p_vol_breakout.get('lookback_period'), 20)
            is_volume_spike = df['volume_D'] > df[vol_ma_col] * volume_ratio
            is_price_breakout = df['close_D'] > df['high_D'].shift(1).rolling(lookback).max()
            triggers['TRIGGER_VOLUME_SPIKE_BREAKOUT'] = is_volume_spike & is_price_breakout

        # 2.2 【均线】回踩支撑反弹
        p_ma_rebound = trigger_params.get('pullback_rebound_trigger_params', {})
        if get_param_value(p_ma_rebound.get('enabled'), True):
            support_ma_period = get_param_value(p_ma_rebound.get('support_ma'), 21)
            support_ma_col = f'EMA_{support_ma_period}_D'
            if support_ma_col in df.columns:
                was_touching_support = df['low_D'].shift(1) <= df[support_ma_col].shift(1)
                is_rebounded_above = df['close_D'] > df[support_ma_col]
                is_positive_day = df['close_D'] > df['open_D']
                triggers['TRIGGER_PULLBACK_REBOUND'] = was_touching_support & is_rebounded_above & is_positive_day

        # 2.3 【筹码】回踩平台反弹 (S级战术动作)
        p_platform_rebound = trigger_params.get('platform_pullback_trigger_params', {})
        if get_param_value(p_platform_rebound.get('enabled'), True):
            platform_price_col = 'PLATFORM_PRICE_STABLE'
            if platform_price_col in df.columns:
                proximity_ratio = get_param_value(p_platform_rebound.get('proximity_ratio'), 0.01)
                is_touching_platform = df['low_D'] <= df[platform_price_col] * (1 + proximity_ratio)
                is_closing_above = df['close_D'] > df[platform_price_col]
                is_positive_day = df['close_D'] > df['open_D']
                triggers['TRIGGER_PLATFORM_PULLBACK_REBOUND'] = is_touching_platform & is_closing_above & is_positive_day

        # 2.4 【趋势】趋势延续确认K线
        p_cont = trigger_params.get('trend_continuation_candle', {})
        if get_param_value(p_cont.get('enabled'), True):
            lookback_period = get_param_value(p_cont.get('lookback_period'), 8)
            is_positive_day = df['close_D'] > df['open_D']
            is_new_high = df['close_D'] >= df['high_D'].shift(1).rolling(window=lookback_period).max()
            triggers['TRIGGER_TREND_CONTINUATION_CANDLE'] = is_positive_day & is_new_high

        # --- 3. 复合形态与指标触发器 (Pattern & Indicator Triggers) ---
        # 3.1 N字形态突破 (依赖原子状态)
        p_nshape = self.kline_params.get('n_shape_params', {})
        if get_param_value(p_nshape.get('enabled'), True):
            n_shape_consolidation_state = self.strategy.atomic_states.get('KLINE_STATE_N_SHAPE_CONSOLIDATION', default_series)
            consolidation_high = df['high_D'].where(n_shape_consolidation_state, np.nan).ffill()
            is_breaking_consolidation = df['close_D'] > consolidation_high.shift(1)
            triggers['TRIGGER_N_SHAPE_BREAKOUT'] = (df['close_D'] > df['open_D']) & is_breaking_consolidation

        # 3.2 指标金叉 (MACD)
        p_cross = trigger_params.get('indicator_cross_params', {})
        if get_param_value(p_cross.get('enabled'), True):
            macd_p = p_cross.get('macd_cross', {})
            if get_param_value(macd_p.get('enabled'), True):
                macd_col, signal_col = 'MACD_13_34_8_D', 'MACDs_13_34_8_D'
                if all(c in df.columns for c in [macd_col, signal_col]):
                    is_golden_cross = (df[macd_col] > df[signal_col]) & (df[macd_col].shift(1) <= df[signal_col].shift(1))
                    low_level = get_param_value(macd_p.get('low_level'), -0.5)
                    triggers['TRIGGER_MACD_LOW_CROSS'] = is_golden_cross & (df[macd_col] < low_level)

        # --- 4. 从其他诊断模块接收的事件 (Event Reception) ---
        # 这些事件由其他专业部门生成，本部门只负责接收和汇报
        triggers['TRIGGER_BOX_BREAKOUT'] = self.strategy.atomic_states.get('BOX_EVENT_BREAKOUT', default_series)
        triggers['TRIGGER_EARTH_HEAVEN_BOARD'] = self.strategy.atomic_states.get('BOARD_EVENT_EARTH_HEAVEN', default_series)
        triggers['TRIGGER_TREND_STABILIZING'] = self.strategy.atomic_states.get('MA_STATE_D_STABILIZING', default_series)

        # --- 5. 最终安全检查 (Final Safety Check) ---
        # 确保所有触发器都已正确初始化，防止因计算失败导致后续流程出错
        for key in list(triggers.keys()):
            if triggers[key] is None:
                triggers[key] = pd.Series(False, index=df.index)
            else:
                triggers[key] = triggers[key].fillna(False)
                
        # print("        -> [触发事件中心 V234.0] 所有触发事件定义完成。")
        return triggers

    def _diagnose_post_accumulation_phase(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
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

        # --- 5. 生成“持续性状态” (逻辑不变) ---
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

    def _diagnose_breakout_pullback_relay(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
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
            'OPP_CHIP_SETUP_S'                      # “筹码高度控盘”情报
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
        is_chip_setup = self.strategy.atomic_states.get('OPP_CHIP_SETUP_S', default_series)

        # 2. 定义“接力”逻辑
        # 条件A: 今天是一个“健康回踩”日，并且筹码高度控盘
        is_pullback_opportunity = is_healthy_pullback & is_chip_setup
        
        # 条件B: 在回踩日之前的N天内，必须发生过“初升浪启动”事件
        had_recent_ascent_start = is_ascent_start.rolling(window=lookback_window, min_periods=1).apply(np.any, raw=True).fillna(0).astype(bool)
        
        # 最终裁定：S+级的接力机会
        relay_opportunity = is_pullback_opportunity & had_recent_ascent_start
        
        if relay_opportunity.any():
            print(f"          -> [情报] 侦测到 {relay_opportunity.sum()} 次 S+级“突破-回踩接力”机会！")
        
        states['PLAYBOOK_BREAKOUT_PULLBACK_RELAY_S_PLUS'] = relay_opportunity
        return states

    def _diagnose_squeeze_zone_opportunities(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V506.3 诊断探针版】压缩区机会诊断模块
        - 核心升级: 植入“诊断探针”，打印出所有子条件的触发次数，
                    以便快速定位信号未触发的原因。
        """
        print("        -> [压缩区机会诊断模块 V506.3 诊断探针版] 启动...")
        states = {}
        default_series = pd.Series(False, index=df.index)
        p_squeeze = get_params_block(self.strategy, 'squeeze_shakeout_params')

        # --- 1. 定义并获取上游情报 ---
        is_in_squeeze = self.strategy.atomic_states.get('VOL_STATE_EXTREME_SQUEEZE', default_series)
        is_in_healthy_box = self.strategy.atomic_states.get('BOX_STATE_HEALTHY_ACCUMULATION', default_series)
        is_on_stable_platform = self.strategy.atomic_states.get('PLATFORM_STATE_STABLE_FORMED', default_series)
        is_good_structure = is_in_healthy_box | is_on_stable_platform

        # --- 2. 定义“暴力打压”行为 ---
        drop_threshold = get_param_value(p_squeeze.get('drop_threshold'), -0.03)
        is_sharp_drop = df['pct_change_D'] < drop_threshold

        winner_turnover_col = 'turnover_from_winners_ratio_D'
        if winner_turnover_col not in df.columns:
            print("          -> [警告] 缺少'获利盘成交占比'数据，压缩区洗盘诊断跳过。")
            return {}
        winner_inactive_threshold = get_param_value(p_squeeze.get('winner_inactive_threshold'), 60.0)
        is_winner_inactive = df[winner_turnover_col] < winner_inactive_threshold
        shakeout_action = is_sharp_drop & is_winner_inactive
        
        print(f"             - (探针-明细) 在{is_sharp_drop.sum()}次下跌中, 获利盘成交占比的分布:\n{df.loc[is_sharp_drop, winner_turnover_col].describe()}")

        # --- 3. 定义“快速企稳”结果 ---
        low_price_today = df['low_D']
        low_price_in_next_2_days = df['low_D'].shift(-2).rolling(2).min()
        is_stabilized_later = (low_price_in_next_2_days > low_price_today).shift(2).fillna(False)

        # --- 4. 定义“完美形态”奖励条件 ---
        total_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        is_long_lower_shadow = df['close_D'] > (df['low_D'] + total_range * 0.6)

        # --- 探针诊断区 ---
        print("          -> [探针报告] 正在检查各子条件触发次数...")
        print(f"             - (环境) 能量压缩区 (is_in_squeeze): {is_in_squeeze.sum()} 次")
        print(f"             - (环境) 健康结构 (is_good_structure): {is_good_structure.sum()} 次")
        print(f"             - (行为) 暴力下跌 >3% (is_sharp_drop): {is_sharp_drop.sum()} 次")
        print(f"             - (行为) 获利盘未出逃 (is_winner_inactive): {is_winner_inactive.sum()} 次")
        print(f"             - (行为) 洗盘式打压 (shakeout_action): {shakeout_action.sum()} 次")
        print(f"             - (结果) 后续2日企稳 (is_stabilized_later): {is_stabilized_later.sum()} 次")
        print(f"             - (奖励) 长下影线形态 (is_long_lower_shadow): {is_long_lower_shadow.sum()} 次")

        # 组合所有基础条件
        base_condition_A = is_in_squeeze & is_good_structure & shakeout_action & is_stabilized_later
        print(f"             - (组合) A级基础条件满足次数: {base_condition_A.sum()} 次")
        # --- 探针结束 ---

        # --- 5. 最终裁定与分级 ---
        final_condition_S = base_condition_A & is_long_lower_shadow
        final_condition_A = base_condition_A & ~final_condition_S

        states['OPP_SQUEEZE_ZONE_SHAKEOUT_A'] = final_condition_A
        states['OPP_SQUEEZE_ZONE_SHAKEOUT_S'] = final_condition_S

        if final_condition_A.any():
            print(f"          -> [A级机会情报] 侦测到 {final_condition_A.sum()} 次“压缩区实战洗盘”机会！")
        if final_condition_S.any():
            print(f"          -> [S级机会情报] 侦测到 {final_condition_S.sum()} 次“压缩区完美洗盘”机会！")

        # 如果最终还是0次，打印一条总结性信息
        if not final_condition_A.any() and not final_condition_S.any():
            print("          -> [诊断结论] “压缩区洗盘”信号最终触发0次。请检查上方探针报告，定位触发次数过少的环节。")

        return states

    def _diagnose_mean_reversion_playbooks(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """【新增】均值回归剧本诊断模块"""
        states = {}
        
        # 1. 定义Setup: 统计学超卖
        is_oversold_boll = df['close_D'] < df.get('BBL_21_2.0_D', float('inf'))
        is_oversold_bias = self.strategy.atomic_states.get('OPP_STATE_NEGATIVE_DEVIATION', pd.Series(False, index=df.index))
        is_oversold_rsi = self.strategy.atomic_states.get('OSC_STATE_RSI_OVERSOLD', pd.Series(False, index=df.index))
        
        # 至少满足两个超卖条件
        oversold_conditions = pd.concat([is_oversold_boll, is_oversold_bias, is_oversold_rsi], axis=1)
        setup_stat_oversold = oversold_conditions.sum(axis=1) >= 2
        
        # 2. 定义Trigger: 力量衰竭与反转
        is_panic_selling = df.get('volume_zscore_D', 0) > 3 # 假设已计算
        is_reversal_candle = self.strategy.atomic_states.get('TRIGGER_HAMMER_REVERSAL', pd.Series(False, index=df.index)) # 示例
        is_main_force_absorbing = self.strategy.atomic_states.get('CPA_FALL_WITH_MAIN_FORCE_ABSORBING', pd.Series(False, index=df.index))
        
        trigger_reversal = is_panic_selling | is_reversal_candle | is_main_force_absorbing
        
        # 3. 组合成剧本状态
        self.strategy.playbook_states['MEAN_REVERSION_BOUNCE_A'] = {
            'setup': setup_stat_oversold,
            'trigger': trigger_reversal
        }
        return {} # 这个函数直接修改 playbook_states，不返回新的原子状态

    def _diagnose_fibonacci_support(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【新增】斐波那契支撑诊断模块
        - 核心职责: 识别价格在关键斐波那契回撤位获得支撑的行为。
        - 产出:
            - OPP_FIB_SUPPORT_GOLDEN_POCKET_S: 在0.618黄金分割位获得支撑的S级信号。
            - OPP_FIB_SUPPORT_STANDARD_A: 在0.382或0.5标准分割位获得支撑的A级信号。
        """
        print("        -> [斐波那契支撑诊断模块] 启动...")
        states = {}
        default_series = pd.Series(False, index=df.index)

        p = get_params_block(self.strategy, 'fibonacci_support_params')
        if not get_param_value(p.get('enabled'), False):
            return {}

        proximity_ratio = get_param_value(p.get('proximity_ratio'), 0.01)
        
        # 定义关键的斐波那契水平列名
        fib_levels = {
            '0.618': 'FIB_0_618_D',
            '0.500': 'FIB_0_500_D',
            '0.382': 'FIB_0_382_D'
        }

        # 检查所需列是否存在
        if any(col not in df.columns for col in fib_levels.values()):
            print("          -> [警告] 缺少斐波那契水平数据列，模块跳过。")
            return {}

        # 核心逻辑：识别“下探回升”于斐波那契位的行为
        # 即：当日最低价 <= 斐波那契位，但收盘价 > 斐波那契位
        def check_support(fib_level_col):
            fib_level = df[fib_level_col]
            # 允许一定的误差
            is_pierced = df['low_D'] <= fib_level * (1 + proximity_ratio)
            is_reclaimed = df['close_D'] > fib_level * (1 - proximity_ratio)
            return is_pierced & is_reclaimed

        # 分别为不同级别的支撑生成信号
        support_618 = check_support(fib_levels['0.618'])
        support_500 = check_support(fib_levels['0.500'])
        support_382 = check_support(fib_levels['0.382'])

        states['OPP_FIB_SUPPORT_GOLDEN_POCKET_S'] = support_618
        states['OPP_FIB_SUPPORT_STANDARD_A'] = support_500 | support_382
        
        if states['OPP_FIB_SUPPORT_GOLDEN_POCKET_S'].any():
            print(f"          -> [情报] 侦测到 {states['OPP_FIB_SUPPORT_GOLDEN_POCKET_S'].sum()} 次 S级“黄金口袋”支撑。")
        if states['OPP_FIB_SUPPORT_STANDARD_A'].any():
            print(f"          -> [情报] 侦测到 {states['OPP_FIB_SUPPORT_STANDARD_A'].sum()} 次 A级“标准斐波那契”支撑。")
            
        return states

    def _diagnose_holding_risks(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
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

    def _diagnose_peak_formation_dynamics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V340.0 新增】筹码峰“创世纪”模块
        - 核心职责: 追踪主筹码峰的“政权更迭”，并为其关联上“出生证明”
                    (确立之日的成交量特征)，从而解读其战略意义。
        """
        # print("        -> [筹码峰“创世纪”模块 V340.0] 启动...")
        states = {}
        
        # --- 1. 检查所需数据 ---
        required_cols = ['peak_cost_D', 'volume_D', 'VOL_MA_21_D']
        if any(c not in df.columns for c in required_cols):
            print("          -> [警告] 缺少诊断筹码峰起源所需数据，模块跳过。")
            return {}

        # --- 2. 识别“政权更迭”事件 ---
        # 定义：主峰成本与昨日相比，变化超过1.5%
        is_peak_changed = (df['peak_cost_D'].pct_change().abs() > 0.015)
        
        # --- 3. 状态机：追踪并确认新的主峰 ---
        df['formation_date'] = np.nan
        df['formation_volume_ratio'] = np.nan
        
        in_observation = False
        observation_start_idx = -1
        stability_period = 3 # 需要稳定3天

        for i in range(1, len(df)):
            if is_peak_changed.iloc[i] and not in_observation:
                # 发现潜在的更迭事件，开始观察
                in_observation = True
                observation_start_idx = i
            
            if in_observation:
                # 检查自观察开始以来，主峰是否保持稳定
                observation_window = df['peak_cost_D'].iloc[observation_start_idx : i+1]
                is_stable = (observation_window.std() / observation_window.mean()) < 0.01
                
                if not is_stable:
                    # 如果不稳定，重置观察
                    in_observation = False
                elif (i - observation_start_idx + 1) >= stability_period:
                    # 如果已稳定达到N天，则确认“政权”
                    formation_date = df.index[observation_start_idx]
                    formation_volume_ratio = df.at[formation_date, 'volume_D'] / df.at[formation_date, 'VOL_MA_21_D']
                    
                    # 将“出生证明”赋予从确立日到今天的所有记录
                    for j in range(observation_start_idx, i + 1):
                        df.at[df.index[j], 'formation_date'] = formation_date
                        df.at[df.index[j], 'formation_volume_ratio'] = formation_volume_ratio
                    
                    # 结束本次观察
                    in_observation = False

        # --- 4. 解读“出生证明”，生成战略信号 ---
        # 条件A: 高量形成 (成交量是均量的2倍以上)
        is_high_volume_formation = df['formation_volume_ratio'] > 2.0
        # 条件B: 缩量形成 (成交量低于均量的70%)
        is_low_volume_formation = df['formation_volume_ratio'] < 0.7
        
        # 条件C: 形成于下跌/盘整后 (用长期均线斜率判断)
        is_after_downtrend = df['SLOPE_55_EMA_55_D'].shift(1) <= 0
        # 条件D: 形成于上涨后
        is_after_uptrend = df['SLOPE_55_EMA_55_D'].shift(1) > 0

        # 组合生成最终的原子状态
        states['PEAK_DYN_FORTRESS_SUPPORT'] = is_high_volume_formation & is_after_downtrend
        states['PEAK_DYN_EXHAUSTION_TOP'] = is_high_volume_formation & is_after_uptrend
        states['PEAK_DYN_STEALTH_ACCUMULATION'] = is_low_volume_formation & is_after_downtrend

        return states

    def _get_dynamic_thresholds(self, df: pd.DataFrame) -> Dict:
        """
        【V335.2 核心指标版】动态阈值校准中心
        - 核心净化: 彻底移除了对“主力资金流”这一不可靠数据的动态阈值计算。
                    本模块现在只为最核心、最难被操纵的筹码结构指标提供校准。
        - 作战原则: 我们的核心标尺，必须建立在最坚实的岩石之上。
        """
        # print("        -> [动态阈值校准中心 V335.2 核心指标版] 启动...")
        thresholds = {}
        window = 250 # 使用过去一年的数据作为基准

        # 1. 成本加速度阈值：只相信最顶尖5%的进攻意图
        cost_accel_col = 'ACCEL_5_peak_cost_D'
        if cost_accel_col in df.columns:
            thresholds['cost_accel_significant'] = df[cost_accel_col].rolling(window).quantile(0.95)

        # 2. 筹码集中度加速度阈值：只相信最顶尖5%的吸筹决心
        conc_accel_col = 'ACCEL_5_concentration_90pct_D'
        if conc_accel_col in df.columns:
            thresholds['conc_accel_significant'] = df[conc_accel_col].rolling(window).quantile(0.05)
            
        # 【净化完成】资金流数据因其不可靠性，不应在此进行核心校准。

        # print("        -> [动态阈值校准中心 V335.2] 校准完成。")
        return thresholds

    def _diagnose_behavioral_patterns(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
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

    def _run_cognitive_synthesis_engine(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V337.0 职责净化版】认知综合引擎
        - 核心重构: 剥离所有进攻性战术的定义，将其移至OffensiveLayer。
                    本模块现在只负责生成顶层的、中性的风险上下文信号。
        """
        # print("        -> [认知综合引擎 V337.0] 启动，正在合成顶层风险上下文...")
        cognitive_states = {}
        default_series = pd.Series(False, index=df.index)

        # --- 认知链 1/2: 识别“突破派发”风险 ---
        # 这个风险的定义依赖于K线形态，保留在此处是合理的
        is_strong_rally = df['pct_change_D'] > 0.03
        is_main_force_selling = self.strategy.atomic_states.get('RISK_CAPITAL_STRUCT_MAIN_FORCE_DISTRIBUTING', default_series)
        cognitive_states['COGNITIVE_RISK_BREAKOUT_DISTRIBUTION'] = is_strong_rally & is_main_force_selling

        # --- 认知链 2/2: 汇总“近期派发压力”上下文 ---
        # 这个信号是顶层风险上下文，也应保留
        distribution_event = (
            self.strategy.atomic_states.get('RISK_S_PLUS_CONFIRMED_DISTRIBUTION', default_series) |
            self.strategy.atomic_states.get('ACTION_RISK_RALLY_WITH_DIVERGENCE', default_series) |
            self.strategy.atomic_states.get('RISK_DYN_WINNER_RATE_COLLAPSING', default_series)
        )
        p_dist = get_params_block(self.strategy, 'distribution_context_params', {})
        lookback = get_param_value(p_dist.get('lookback_days'), 10)
        cognitive_states['CONTEXT_RECENT_DISTRIBUTION_PRESSURE'] = distribution_event.rolling(window=lookback, min_periods=1).apply(np.any, raw=True).fillna(0).astype(bool)

        # print("        -> [认知综合引擎 V337.0] 顶层风险上下文合成完毕。")
        return cognitive_states
    
    def _generate_playbook_states(self, df: pd.DataFrame, trigger_events: Dict[str, pd.Series]) -> Tuple[Dict[str, pd.Series], Dict[str, Dict[str, pd.Series]]]:
        """
        【V264.0 内存优化版】剧本情报生成中心
        - 核心重构: 彻底取代了旧的、基于 deepcopy 的 _get_playbook_definitions 方法。
        - 新架构 (“蓝图与情报分离”):
          1. 不再复制任何 playbook 蓝图，从根本上杜绝了内存爆炸。
          2. 首先，计算所有 setup_scores (战机准备状态评估)。
          3. 然后，仅生成并返回一个轻量级的 playbook_states 字典，其结构为:
             { 'PLAYBOOK_NAME': {'setup': pd.Series, 'trigger': pd.Series} }
        - 收益: 内存效率、计算效率和代码清晰度都得到了革命性的提升。
        """
        print("    - [剧本情报中心 V264.0] 启动，正在生成动态情报...")
        default_series = pd.Series(False, index=df.index)
        playbook_states = {}

        # --- 步骤1: 战机准备状态评估 (Setup Readiness Assessment) ---
        # 注意: 此部分逻辑从旧方法中完整迁移而来，是生成动态情报的前置步骤。
        print("      -> 步骤1/3: 正在进行战机准备状态评估 (Setup Scoring)...")
        setup_scores = {}
        # 假设 setup_scoring_matrix 从配置文件加载，这是更健壮的做法
        scoring_matrix = get_params_block(self.strategy, 'setup_scoring_matrix', {}) 
        for setup_name, rules in scoring_matrix.items():
            if not get_param_value(rules.get('enabled'), True):
                continue
            
            # --- “投降坑” 专属评分逻辑 ---
            if setup_name == 'CAPITULATION_PIT':
                p_cap_pit = rules
                must_have_score = get_param_value(p_cap_pit.get('must_have_score'), 40)
                bonus_score = get_param_value(p_cap_pit.get('bonus_score'), 25)
                must_have_conditions = self.strategy.atomic_states.get('OPP_STATE_NEGATIVE_DEVIATION', default_series)
                bonus_conditions_1 = self.strategy.atomic_states.get('CHIP_STATE_LOW_PROFIT', default_series)
                bonus_conditions_2 = self.strategy.atomic_states.get('CHIP_STATE_SCATTERED', default_series)
                base_score = must_have_conditions.astype(int) * must_have_score
                bonus_score_total = (bonus_conditions_1.astype(int) * bonus_score) + (bonus_conditions_2.astype(int) * bonus_score)
                final_score = (base_score + bonus_score_total).where(must_have_conditions, 0)
                setup_scores[f'SETUP_SCORE_{setup_name}'] = final_score
            # --- “平台质量” 专属评分逻辑 ---
            elif setup_name == 'PLATFORM_QUALITY':
                p_quality = rules
                must_have_cond = self.strategy.atomic_states.get('PLATFORM_STATE_STABLE_FORMED', default_series)
                base_score_val = get_param_value(p_quality.get('base_score'), 40)
                base_score = must_have_cond.astype(int) * base_score_val
                bonus_score_series = pd.Series(0.0, index=df.index)
                bonus_rules = p_quality.get('bonus', {})
                for state, score in bonus_rules.items():
                    state_series = self.strategy.atomic_states.get(state, default_series)
                    bonus_score_series += state_series.astype(int) * score
                setup_scores[f'SETUP_SCORE_{setup_name}'] = (base_score + bonus_score_series).where(must_have_cond, 0)
            else:
                # --- 其他所有剧本使用通用评分逻辑 ---
                current_score = pd.Series(0.0, index=df.index)
                must_have_rules = rules.get('must_have', {})
                must_have_passed = pd.Series(True, index=df.index)
                for state, score in must_have_rules.items():
                    state_series = self.strategy.atomic_states.get(state, default_series)
                    current_score += state_series * score
                    must_have_passed &= state_series
                
                any_of_rules = rules.get('any_of_must_have', {})
                any_of_passed = pd.Series(False, index=df.index)
                if any_of_rules:
                    any_of_score_component = pd.Series(0.0, index=df.index)
                    for state, score in any_of_rules.items():
                        state_series = self.strategy.atomic_states.get(state, default_series)
                        any_of_score_component.loc[state_series] = score
                        any_of_passed |= state_series
                    current_score += any_of_score_component
                else:
                    any_of_passed = pd.Series(True, index=df.index)

                bonus_rules = rules.get('bonus', {})
                for state, score in bonus_rules.items():
                    state_series = self.strategy.atomic_states.get(state, default_series)
                    current_score += state_series * score
                
                final_validity = must_have_passed & any_of_passed
                setup_scores[f'SETUP_SCORE_{setup_name}'] = current_score.where(final_validity, 0)
        print("      -> 战机准备状态评估完成。")

        # --- 步骤2: 生成动态的“剧本情报” ---
        print("      -> 步骤2/3: 正在生成动态情报...")
        # 准备原子状态和评估分数
        score_cap_pit = setup_scores.get('SETUP_SCORE_CAPITULATION_PIT', default_series)
        score_deep_accum = setup_scores.get('SETUP_SCORE_DEEP_ACCUMULATION', default_series)
        score_nshape_cont = setup_scores.get('SETUP_SCORE_N_SHAPE_CONTINUATION', default_series)
        score_gap_support = setup_scores.get('SETUP_SCORE_GAP_SUPPORT_PULLBACK', default_series)
        score_bottoming_process = setup_scores.get('SETUP_SCORE_BOTTOMING_PROCESS', default_series)
        score_healthy_markup = setup_scores.get('SETUP_SCORE_HEALTHY_MARKUP', default_series)
        score_platform_quality = setup_scores.get('SETUP_SCORE_PLATFORM_QUALITY', default_series)

        capital_divergence_window = self.strategy.atomic_states.get('CAPITAL_STATE_DIVERGENCE_WINDOW', default_series)
        setup_bottom_passivation = self.strategy.atomic_states.get('MA_STATE_BOTTOM_PASSIVATION', default_series)
        setup_washout_reversal = self.strategy.atomic_states.get('KLINE_STATE_WASHOUT_WINDOW', default_series)
        setup_healthy_box = self.strategy.atomic_states.get('BOX_STATE_HEALTHY_CONSOLIDATION', default_series)
        recent_reversal_context = self.strategy.atomic_states.get('CONTEXT_RECENT_REVERSAL_SIGNAL', default_series)
        ma_short_slope_positive = self.strategy.atomic_states.get('MA_STATE_SHORT_SLOPE_POSITIVE', default_series)
        is_trend_healthy = self.strategy.atomic_states.get('CONTEXT_OVERALL_TREND_HEALTHY', default_series)
        
        self.strategy.atomic_states['SETUP_SCORE_N_SHAPE_CONTINUATION_ABOVE_80'] = score_nshape_cont > 80
        self.strategy.atomic_states['SETUP_SCORE_HEALTHY_MARKUP_ABOVE_60'] = score_healthy_markup > 60

        # 遍历静态蓝图，仅生成动态情报，不进行任何复制操作
        for blueprint in self.playbook_blueprints:
            name = blueprint['name']
            setup_series = default_series
            trigger_series = default_series
            
            # 根据蓝图规则，计算 setup 和 trigger 的布尔序列
            # (这里的逻辑与旧方法中填充 hydrated_playbooks 的逻辑完全相同)
            if name == 'ABYSS_GAZE_S':
                setup_series = score_cap_pit > 80
                trigger_series = trigger_events.get('TRIGGER_DOMINANT_REVERSAL', default_series)
            elif name == 'CAPITULATION_PIT_REVERSAL':
                rules = blueprint.get('scoring_rules', {})
                min_score = rules.get('min_setup_score_to_trigger', 51)
                setup_series = score_cap_pit >= min_score
                trigger_series = (trigger_events.get('TRIGGER_REVERSAL_CONFIRMATION_CANDLE', default_series) | trigger_events.get('TRIGGER_BREAKOUT_CANDLE', default_series))
            elif name == 'CAPITAL_DIVERGENCE_REVERSAL':
                setup_series = capital_divergence_window
                trigger_series = trigger_events.get('TRIGGER_REVERSAL_CONFIRMATION_CANDLE', default_series)
            elif name == 'BEAR_TRAP_RALLY':
                setup_series = setup_bottom_passivation
                trigger_series = trigger_events.get('TRIGGER_TREND_STABILIZING', default_series)
            elif name == 'WASHOUT_REVERSAL_A':
                setup_series = setup_washout_reversal
                trigger_series = trigger_events.get('TRIGGER_BREAKOUT_CANDLE', default_series)
            elif name == 'BOTTOM_STABILIZATION_B':
                setup_series = score_bottoming_process > 50
                trigger_series = trigger_events.get('TRIGGER_BREAKOUT_CANDLE', default_series)
            elif name == 'TREND_EMERGENCE_B_PLUS':
                setup_series = recent_reversal_context & ma_short_slope_positive
                trigger_series = trigger_events.get('TRIGGER_TREND_CONTINUATION_CANDLE', default_series)
            elif name == 'PLATFORM_SUPPORT_PULLBACK':
                rules = blueprint.get('scoring_rules', {})
                min_score = rules.get('min_setup_score_to_trigger', 50)
                setup_series = score_platform_quality >= min_score
                trigger_ma_rebound = trigger_events.get('TRIGGER_PULLBACK_REBOUND', default_series)
                trigger_chip_rebound = trigger_events.get('TRIGGER_PLATFORM_PULLBACK_REBOUND', default_series)
                trigger_series = trigger_ma_rebound | trigger_chip_rebound
            elif name == 'HEALTHY_MARKUP_A':
                rules = blueprint.get('scoring_rules', {})
                min_score = rules.get('min_setup_score_to_trigger', 60)
                setup_series = score_healthy_markup >= min_score
                trigger_rebound = trigger_events.get('TRIGGER_PULLBACK_REBOUND', default_series)
                trigger_continuation = trigger_events.get('TRIGGER_TREND_CONTINUATION_CANDLE', default_series)
                trigger_series = trigger_rebound | trigger_continuation
            elif name == 'HEALTHY_BOX_BREAKOUT':
                setup_series = setup_healthy_box
                trigger_series = trigger_events.get('BOX_EVENT_BREAKOUT', default_series)
            elif name == 'GAP_SUPPORT_PULLBACK_B_PLUS':
                setup_series = (score_gap_support > 60)
                trigger_series = trigger_events.get('TRIGGER_PULLBACK_REBOUND', default_series)
            elif name == 'CHIP_PLATFORM_PULLBACK':
                setup_platform_formed = self.strategy.atomic_states.get('PLATFORM_STATE_STABLE_FORMED', default_series)
                setup_series = setup_platform_formed & is_trend_healthy
                trigger_series = trigger_events.get('TRIGGER_PLATFORM_PULLBACK_REBOUND', default_series)
            elif name == 'ENERGY_COMPRESSION_BREAKOUT':
                # 此剧本的 setup 逻辑在计分函数中处理，这里只定义 trigger
                trigger_series = trigger_events.get('TRIGGER_BREAKOUT_CANDLE', default_series)
            elif name == 'DEEP_ACCUMULATION_BREAKOUT':
                rules = blueprint.get('scoring_rules', {})
                min_score = rules.get('min_setup_score_to_trigger', 51)
                setup_series = score_deep_accum >= min_score
                trigger_series = trigger_events.get('TRIGGER_BREAKOUT_CANDLE', default_series)
            elif name == 'N_SHAPE_CONTINUATION_A':
                # 此剧本的 setup 逻辑在计分函数中处理，这里只定义 trigger
                trigger_series = trigger_events.get('TRIGGER_N_SHAPE_BREAKOUT', default_series)
            elif name == 'EARTH_HEAVEN_BOARD':
                setup_series = pd.Series(True, index=df.index) # 事件驱动，无前置setup
                trigger_series = trigger_events.get('TRIGGER_EARTH_HEAVEN_BOARD', default_series)
            
            playbook_states[name] = {'setup': setup_series, 'trigger': trigger_series}
        print("      -> 作战计划动态“水合”完成。")

        # --- 步骤3: 统一交战规则审查 (Unified Rules of Engagement) ---
        print("      -> 步骤3/3: 正在执行统一交战规则审查...")
        is_trend_deteriorating = self.strategy.atomic_states.get('CONTEXT_TREND_DETERIORATING', default_series)
        for blueprint in self.playbook_blueprints:
            if blueprint.get('side') == 'right':
                name = blueprint['name']
                if name in playbook_states:
                    original_trigger = playbook_states[name]['trigger']
                    playbook_states[name]['trigger'] = original_trigger & ~is_trend_deteriorating
        print("      -> “统一交战规则”审查完毕，所有右侧进攻性操作已被置于战略监控之下。")
        
        print("    - [剧本情报中心 V264.0] 动态情报生成完毕。")
        return setup_scores, playbook_states

    def _diagnose_contextual_zones(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
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

    def _diagnose_pullback_character(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V500.0 统一回踩诊断中心】
        - 核心重构: 取代了原有的 _diagnose_healthy_pullback 和 _diagnose_suppression_pullback 模块。
        - 核心职责: 1. 识别所有发生在建设性背景下的回踩行为。
                    2. 对回踩的“性质”进行分类（健康的、打压式的等）。
                    3. 输出不同性质的、中立的原子状态，供下游决策。
        """
        print("        -> [统一回踩诊断中心 V500.0] 启动...")
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

        # --- 2. 定义通用的“建设性背景” (逻辑不变) ---
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
        is_low_turnover = df['turnover_rate_D'] < get_param_value(p_healthy.get('max_turnover_rate'), 5.0)
        is_healthy_character = is_gentle_drop & is_shrinking_volume & is_low_turnover

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
        states['PULLBACK_STATE_HEALTHY_S'] = is_pullback_day & is_healthy_character & is_constructive_context & is_chip_stable
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

    def _synthesize_topping_behaviors(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V331.1 职责分离版】顶部行为合成模块
        - 核心重构: 职责被简化为“行为合成”。它消费已有的“战场上下文”和
                    “筹码动态”情报，将其融合成顶层的战术信号。
        """
        print("        -> [顶部行为合成模块 V331.1] 启动...")
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

    def _generate_playbook_states(self, trigger_events: Dict[str, pd.Series]) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
        """
        【V285.0 新增】剧本状态生成引擎
        - 核心职责: 基于原子状态和触发事件，识别并生成所有预定义的“剧本”状态。
        - 剧本: 一个剧本是多个原子状态和触发事件的特定组合，代表一个高胜率的交易设置。
        """
        # print("        -> [剧本状态生成引擎 V285.0] 启动，正在识别所有交易剧本...")
        playbook_states = {}
        setup_scores = {}
        df = self.strategy.df_indicators
        default_series = pd.Series(False, index=df.index)

        # --- 剧本1: S级 - 波动压缩突破 (Squeeze Breakout) ---
        # 定义: 处于波动率压缩窗口，然后出现能量释放阳线或放量突破
        is_in_squeeze = self.strategy.atomic_states.get('VOL_STATE_SQUEEZE_WINDOW', default_series)
        is_breakout_trigger = trigger_events.get('TRIGGER_ENERGY_RELEASE', default_series) | trigger_events.get('TRIGGER_VOLUME_SPIKE_BREAKOUT', default_series)
        playbook_states['PLAYBOOK_SQUEEZE_BREAKOUT_S'] = is_in_squeeze.shift(1).fillna(False) & is_breakout_trigger
        setup_scores['PLAYBOOK_SQUEEZE_BREAKOUT_S'] = playbook_states['PLAYBOOK_SQUEEZE_BREAKOUT_S'] * 100 # S-level gets 100 points

        # --- 剧本2: A级 - 趋势回踩反弹 (Pullback Rebound) ---
        # 定义: 处于主升浪结构中，然后出现回踩均线反弹的触发信号
        is_in_main_uptrend = self.strategy.atomic_states.get('STRUCTURE_MAIN_UPTREND_WAVE_S', default_series)
        is_rebound_trigger = trigger_events.get('TRIGGER_PULLBACK_REBOUND', default_series)
        playbook_states['PLAYBOOK_PULLBACK_REBOUND_A'] = is_in_main_uptrend & is_rebound_trigger
        setup_scores['PLAYBOOK_PULLBACK_REBOUND_A'] = playbook_states['PLAYBOOK_PULLBACK_REBOUND_A'] * 80 # A-level gets 80 points

        # --- 剧本3: B级 - 黄金坑吸筹 (Golden Pit Accumulation) ---
        # 定义: 出现主力吸筹的下跌（黄金坑信号），并伴随反转确认K线
        is_golden_pit = self.strategy.atomic_states.get('CPA_FALL_WITH_MAIN_FORCE_ABSORBING', default_series)
        is_reversal_trigger = trigger_events.get('TRIGGER_REVERSAL_CONFIRMATION_CANDLE', default_series)
        playbook_states['PLAYBOOK_GOLDEN_PIT_B'] = is_golden_pit & is_reversal_trigger
        setup_scores['PLAYBOOK_GOLDEN_PIT_B'] = playbook_states['PLAYBOOK_GOLDEN_PIT_B'] * 60 # B-level gets 60 points

        print(f"        -> [剧本状态生成引擎 V285.0] 分析完毕，共生成 {len(playbook_states)} 个剧本状态。")
        return setup_scores, playbook_states






