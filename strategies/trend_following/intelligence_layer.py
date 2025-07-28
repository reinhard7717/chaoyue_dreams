# 文件: strategies/trend_following/intelligence_layer.py
# 情报层
import pandas as pd
import numpy as np
from scipy.stats import linregress
from utils.math_tools import hurst_exponent
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
        print("--- [情报层] 步骤1: 运行所有诊断模块... ---")
        df = self.strategy.df_indicators
        params = self.strategy.unified_config
        
        df = self.pattern_recognizer.identify_all(df)
        self.strategy.atomic_states = {}
        
        df, structure_states = self._diagnose_market_structure_command(df)
        chip_states, chip_triggers = self._run_chip_intelligence_command(df)
        
        self.strategy.atomic_states.update(structure_states)
        self.strategy.atomic_states.update(chip_states)
        self.strategy.atomic_states.update(self._diagnose_oscillator_states(df))
        self.strategy.atomic_states.update(self._diagnose_capital_states(df))
        self.strategy.atomic_states.update(self._diagnose_volatility_states(df))
        self.strategy.atomic_states.update(self._diagnose_kline_patterns(df))
        self.strategy.atomic_states.update(self._diagnose_board_patterns(df))
        self.strategy.atomic_states.update(self._diagnose_trend_dynamics(df))
        self.strategy.atomic_states.update(self._diagnose_chip_price_action(df))
        self.strategy.atomic_states.update(self._diagnose_market_structure_states(df))
        self.strategy.atomic_states.update(self._diagnose_structural_mechanics(df))
        self.strategy.atomic_states.update(self._run_cognitive_synthesis_engine(df))
        self.strategy.atomic_states.update(self._diagnose_post_accumulation_phase(df))
        
        self.strategy.df_indicators = self._determine_main_force_behavior_sequence(df)
        
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
        print("        -> [筹码情报最高司令部 V316.0 筹码加权版] 启动...")
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

        is_in_high_level_zone = self._define_high_level_distribution_zone(df)
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
        if is_chip_structure_unhealthy.any():
            print(f"          -> [地基风险报告] 在 {is_chip_structure_unhealthy.sum()} 天内，侦测到严重筹码结构风险。")

        is_highly_concentrated = states.get('CHIP_STATE_HIGHLY_CONCENTRATED', default_series)
        is_cost_rising = states.get('CHIP_DYN_COST_RISING', default_series)
        is_winner_rate_rising = states.get('CHIP_DYN_WINNER_RATE_RISING', default_series)
        is_long_term_distributing = states.get('RISK_CONTEXT_LONG_TERM_DISTRIBUTION', default_series)
        is_cost_stable = df.get('SLOPE_5_peak_cost_D', default_series).abs() < 0.01

        states['CHIPCON_4_READINESS'] = is_highly_concentrated & is_cost_stable & ~is_long_term_distributing
        states['CHIPCON_3_HIGH_ALERT'] = is_highly_concentrated & is_cost_rising & is_winner_rate_rising & ~is_long_term_distributing
        
        print("        -> [筹码情报最高司令部 V316.0 筹码加权版] 分析完毕。")
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
        print("        -> [能量与波动侦察部 V283.0] 启动，正在执行融合分析...")
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
        print("        -> [动态力学分析引擎 V317.0] 启动，正在计算势能加速度...")
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
        【V283.0 新增】全指标动态分析中心
        - 核心职责: 集中处理所有关键筹码指标的斜率与加速度，系统性地生成
                    高维度的动态机遇与风险信号。这是对动态分析能力的终极整合。
        """
        print("          -> [动态分析中心 V283.0] 已部署，正在对全筹码指标进行动态扫描...")
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

        # --- 2. 对“筹码集中度”进行动态分析 (机遇/风险) ---
        states['CHIP_DYN_CONCENTRATING'] = df['SLOPE_5_concentration_90pct_D'] < 0
        states['CHIP_DYN_ACCEL_CONCENTRATING'] = df['ACCEL_5_concentration_90pct_D'] < 0
        states['RISK_DYN_DIVERGING'] = df['SLOPE_5_concentration_90pct_D'] > 0
        states['RISK_DYN_ACCEL_DIVERGING'] = df['ACCEL_5_concentration_90pct_D'] > 0

        # --- 3. 对“筹码成本”进行动态分析 (机遇/风险) ---
        states['CHIP_DYN_COST_RISING'] = df['SLOPE_5_peak_cost_D'] > 0
        states['CHIP_DYN_COST_ACCELERATING'] = df['ACCEL_5_peak_cost_D'] > 0 # 主力猛攻信号
        states['RISK_DYN_COST_FALLING'] = df['SLOPE_5_peak_cost_D'] < 0

        # --- 4. 对“总获利盘”进行动态分析 (机遇/风险) ---
        winner_rate_collapse_threshold = -1.0 # 斜率小于-1才算崩盘
        states['CHIP_DYN_WINNER_RATE_RISING'] = df['SLOPE_5_total_winner_rate_D'] > 0
        states['RISK_DYN_WINNER_RATE_COLLAPSING'] = df['SLOPE_5_total_winner_rate_D'] < winner_rate_collapse_threshold
        states['RISK_DYN_WINNER_RATE_ACCEL_COLLAPSING'] = df['ACCEL_5_total_winner_rate_D'] < 0 # 获利盘加速崩盘

        # --- 5. 对“筹码健康分”进行动态分析 (机遇/风险) ---
        states['CHIP_DYN_HEALTH_IMPROVING'] = df['SLOPE_5_chip_health_score_D'] > 0
        states['RISK_DYN_HEALTH_DETERIORATING'] = df['SLOPE_5_chip_health_score_D'] < 0
        
        print("          -> [动态分析中心 V283.0] 动态扫描完成。")
        return states

    def _diagnose_chip_price_action(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V228.0 新增】筹码-价格行为联合分析部
        - 核心职责: 将价格的涨跌行为与筹码（尤其是主力资金）的流动进行交叉验证，
                    将单纯的价格行为，升维为包含“主力意图”的复合情报。
        - 作战原则: “无筹码，不决策”。
        """
        print("        -> [联合分析部 V228.0] 启动，正在对价格行为进行筹码深度解析...")
        cpa_states = {} # Chip-Price Action States
        default_series = pd.Series(False, index=df.index)

        # --- 1. 提取基础情报 ---
        # 价格行为情报
        is_price_rising = df.get('pct_change_D', default_series) > 0
        is_price_falling = df.get('pct_change_D', default_series) < 0

        # 主力资金动向情报 (来自筹码总参谋部)
        is_main_force_buying = self.strategy.atomic_states.get('CAPITAL_STRUCT_MAIN_FORCE_ACCUMULATING', default_series)
        is_main_force_selling = self.strategy.atomic_states.get('RISK_CAPITAL_STRUCT_MAIN_FORCE_DISTRIBUTING', default_series)
        
        # 散户资金动向情报
        # 假设散户与主力行为相反，或者直接使用散户数据（如果未来有）
        # 这里我们用主力行为的反面来代表散户的主要动向，简化模型
        is_retail_likely_buying = is_main_force_selling
        is_retail_likely_selling = is_main_force_buying

        # --- 2. 进行联合分析，生成高维复合情报 ---
        
        # 【A级进攻信号】上涨的“质”：主力支撑的上涨
        # 定义：价格上涨，同时主力资金在净流入。这是最健康的上涨模式。
        cpa_states['CPA_RISE_WITH_MAIN_FORCE_SUPPORT'] = is_price_rising & is_main_force_buying
        
        # 【C级风险信号】上涨的“危”：散户追高的上涨
        # 定义：价格上涨，但主力资金在净流出。这可能是拉高出货的危险信号。
        cpa_states['CPA_RISE_WITH_RETAIL_FOMO'] = is_price_rising & is_main_force_selling

        # 【S级风险信号】下跌的“质”：主力出逃的下跌
        # 定义：价格下跌，同时主力资金在净流出。这是最危险的下跌，趋势可能反转。
        cpa_states['CPA_FALL_WITH_MAIN_FORCE_FLEEING'] = is_price_falling & is_main_force_selling

        # 【S级机会信号】下跌的“机”：主力吸筹的下跌
        # 定义：价格下跌，但主力资金在净流入。这是经典的“黄金坑”或“洗盘吸筹”信号。
        cpa_states['CPA_FALL_WITH_MAIN_FORCE_ABSORBING'] = is_price_falling & is_main_force_buying

        print("        -> [联合分析部 V228.0] 深度解析完成。")
        return cpa_states

    def _diagnose_trend_dynamics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V174.0 动态惯性引擎】
        - 核心职责: 基于趋势的“斜率”和“加速度”，生成高维度的动态原子状态。
        - 产出: 返回一个包含 DYN_... 信号的字典，供评分引擎使用。
        """
        print("        -> [诊断模块 V174.0] 正在执行动态惯性诊断...")
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

    def _diagnose_capital_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V219.0 情报一体化版】 - 统一资本动向总参谋部
        - 核心升级: 将原 `_diagnose_capital_structure_states` 的职能（主力/散户资金分析）
                    并入此模块，形成统一的、从宏观到微观的资本分析中心。
        - 作战单元1 (宏观气象站): 保留基于CMF的经典资本状态诊断。
        - 作战单元2 (精锐侦察连): 新增基于主力/散户净流入的、高精度的资本结构诊断。
        """
        print("        -> [诊断模块 V219.0 情报一体化版] 正在执行统一资本诊断...")
        states = {}
        default_series = pd.Series(False, index=df.index)
        
        # 1. 定义“主力大规模派发”的单日行为
        #    - 从配置中读取参数，例如单日净流出超过2000万就视为危险
        dist_context_params = get_params_block(self.strategy, 'distribution_context_params', {})
        outflow_threshold = get_param_value(dist_context_params.get('outflow_threshold_M'), -20) * 1_000_000
        
        is_distribution_day = df.get('main_force_net_inflow_amount_D', default_series) < outflow_threshold

        # 2. 建立“战场记忆”：使用滚动窗口检查近期是否发生过派发
        #    - 如果过去10天内，有任何一天是“派发日”，那么今天就处于“近期派发压力”之下
        lookback_window = get_param_value(dist_context_params.get('lookback_days'), 10)
        # .any() 是关键，它检查窗口内是否有至少一个 True
        self.strategy.atomic_states['CONTEXT_RECENT_DISTRIBUTION_PRESSURE'] = is_distribution_day.rolling(
            window=lookback_window, min_periods=1
        ).apply(np.any, raw=True).fillna(0).astype(bool)
        
        print(f"        -> [资本动向总参谋部] “危险战区”感知模块已启动。{format_debug_dates(self.strategy.atomic_states['CONTEXT_RECENT_DISTRIBUTION_PRESSURE'])}")
        
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

        # --- 作战单元2: 【王牌】新型资本结构诊断 (基于主力/散户资金) ---
        main_force_col = 'main_force_net_inflow_amount_D'
        retail_col = 'retail_net_inflow_volume_D'
        # 检查情报是否送达
        if all(c in df.columns for c in [main_force_col, retail_col]):
            print("          -> [情报确认] 主力/散户资金数据已接收，开始结构分析...")
            # 1. 定义“主力正在吸筹”状态
            states['CAPITAL_STRUCT_MAIN_FORCE_ACCUMULATING'] = df[main_force_col] > 0
            # 2. 定义“主力正在派发”风险状态
            states['RISK_CAPITAL_STRUCT_MAIN_FORCE_DISTRIBUTING'] = df[main_force_col] < 0
            # 3. 定义“黄金坑”：主力吸筹 & 散户割肉
            states['CAPITAL_STRUCT_BULLISH_DIVERGENCE'] = (df[main_force_col] > 0) & (df[retail_col] < 0)
            # 4. 定义“死亡顶”：主力派发 & 散户接盘
            states['RISK_CAPITAL_STRUCT_BEARISH_DIVERGENCE'] = (df[main_force_col] < 0) & (df[retail_col] > 0)
        else:
            print(f"          -> [情报警告] 缺少高精度资金结构数据，跳过结构分析。")

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
        print("        -> [市场结构战区司令部 V272.0] 启动，正在整合全战场结构情报...")
        
        # --- 1. 依次调动下属的专业化兵种，收集原子情报 ---
        print("          -> 正在调动：均线野战部队、价格工兵部队、筹码特种侦察部队...")
        ma_states = self._diagnose_ma_states(df)
        box_states = self._diagnose_box_states(df)
        df, platform_states = self._diagnose_platform_states(df) # 平台诊断会修改df，需要接收
        
        # 将所有原子情报汇总
        atomic_structure_states = {**ma_states, **box_states, **platform_states}
        
        # --- 2. 进行情报融合与战术研判，生成复合情报 ---
        print("          -> 正在进行情报融合，生成高维度复合情报...")
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

        print("        -> [市场结构战区司令部 V272.0] 情报整合完毕。")
        
        # 返回所有原子情报和复合情报的集合，以及可能被修改的df
        return df, {**atomic_structure_states, **composite_states}

    def _diagnose_ma_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V283.0 融合思想版】
        - 核心升级: 引入“攻击性多头排列”信号。
          - MA_STATE_STABLE_BULLISH: 基础的多头排列状态 (静态位置)。
          - MA_STATE_AGGRESSIVE_BULLISH: 在多头排列基础上，要求短期均线斜率陡峭 (动态趋势)，是更高质量的信号。
        """
        print("          -> [均线野战部队 V283.0] 启动，正在执行融合分析...")
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
        return states

    def _diagnose_box_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V283.0 融合思想版】
        - 核心升级: 引入“健康吸筹箱体”信号。
          - BOX_STATE_HEALTHY_CONSOLIDATION: 基础的、位于趋势线上方的箱体 (静态)。
          - BOX_STATE_HEALTHY_ACCUMULATION: 在健康箱体内，要求同时满足“缩量”和“筹码集中” (动态)，是更高质量的突破前兆。
        """
        print("        -> [工兵部队 V283.0] 启动，正在执行融合分析...")
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
        print("        -> [诊断模块 V129.2] 正在执行筹码平台状态诊断...")
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
        states['PLATFORM_FAILURE'] = platform_failure_series

        # --- 步骤5: 打印诊断日志 ---
        print(f"          -> '稳固平台形成' 状态诊断完成，共激活 {stable_formed_series.sum()} 天。")
        print(f"          -> '平台破位' 风险诊断完成，共激活 {platform_failure_series.sum()} 天。")

        return df, states

    def _diagnose_structural_mechanics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 - 赫斯特指数增强版】
        - 核心升级: 深度集成赫斯特指数，从中提炼出多种关于市场宏观阶段、
                    趋势可持续性与衰竭的原子状态。
        """
        print("        -> [结构力学诊断模块 V2.0] 启动，正在进行物理建模与分形分析...")
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
        print("        -> [联合作战司令部 V277.0 五重共振版] 启动，正在打造终极S级战局信号...")
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

        print("        -> [联合作战司令部 V277.0] 核心战局定义升级完成。")
        return structure_states

    def _diagnose_volume_price_dynamics(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V284.0 新增】量价关系动态分析中心 (CT扫描室)
        - 核心职责: 对“成交量”和“资金攻击效率”进行全面的斜率与加速度分析，
                    将“天量对倒”这个模糊概念，升级为可量化、可跟踪的动态风险信号。
        """
        print("          -> [量价动态分析中心 V284.0] 启动，正在对“天量对倒”进行CT扫描...")
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
        
        print("          -> [量价动态分析中心 V284.0] CT扫描完成。")
        return states
   
    def _determine_main_force_behavior_sequence(self, df: pd.DataFrame) -> pd.DataFrame:
        print("    --- [战略推演单元 V304.0] 启动，正在生成主力行为序列... ---")
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
        print("    --- [战略推演单元 V304.0] 主力行为序列已生成。 ---")
        return df

    def _define_trigger_events(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V234.0 最终净化版 - 战术触发事件定义中心】
        - 核心升级: 严格遵循“V234.0 作战条例”，所有参数均从唯一的 trigger_event_params 配置块中获取，
                    确保了配置的单一来源原则，使整个触发体系清晰、健壮、易于维护。
        - 职责: 识别所有可以作为“开火信号”的瞬时战术事件(Trigger)。
        """
        print("        -> [触发事件中心 V234.0] 启动，正在定义所有原子化触发事件...")
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
                
        print("        -> [触发事件中心 V234.0] 所有触发事件定义完成。")
        return triggers

    def _diagnose_post_accumulation_phase(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V310.0 新增】初升浪诊断模块 (A股特化)
        - 核心职责: 识别并标注那些刚刚完成“震荡吸筹”阶段，并开始“初升浪”的股票。
                    这是A股市场中一个典型的高价值阶段。
        - 诊断逻辑:
          1. 定义“吸筹区”: 识别出存在“健康吸筹箱体”或“稳固筹码平台”的时期。
          2. 定义“突破事件”: 捕捉从吸筹区向上放量突破的关键K线。
          3. 生成“初升状态”: 从突破日开始，生成一个持续15个交易日的“初升浪”状态标签，
                           如果期间出现结构性破位，则该状态提前终止。
        """
        print("        -> [诊断模块 V310.0] 启动，正在扫描“初升浪”模式...")
        states = {}
        default_series = pd.Series(False, index=df.index)

        # --- 步骤1: 定义“吸筹区”上下文 ---
        # 依赖于其他模块已经生成的原子状态
        is_healthy_box = self.strategy.atomic_states.get('BOX_STATE_HEALTHY_ACCUMULATION', default_series)
        is_stable_platform = self.strategy.atomic_states.get('PLATFORM_STATE_STABLE_FORMED', default_series)
        
        # 只要满足其中一个条件，就认为当天有吸筹迹象
        is_accumulating_day = is_healthy_box | is_stable_platform
        
        # 定义一个更宽松的“吸筹区”上下文：过去15天内至少有3天存在吸筹迹象
        accumulation_context_params = get_params_block(self.strategy, 'post_accumulation_params', {})
        lookback = get_param_value(accumulation_context_params.get('lookback_days'), 15)
        min_days = get_param_value(accumulation_context_params.get('min_accumulation_days'), 3)
        is_in_accumulation_zone = is_accumulating_day.rolling(window=lookback, min_periods=1).sum() >= min_days
        
        # --- 步骤2: 定义“突破事件” ---
        breakout_params = get_params_block(self.strategy, 'trigger_event_params', {}).get('volume_spike_breakout', {})
        vol_ratio = get_param_value(breakout_params.get('volume_ratio'), 2.0)
        price_lookback = get_param_value(breakout_params.get('lookback_period'), 20)
        
        vol_ma_col = 'VOL_MA_21_D'
        if vol_ma_col not in df.columns:
            print("          -> [警告] 缺少 VOL_MA_21_D 列，无法定义突破事件，跳过初升浪诊断。")
            return states

        is_volume_spike = df['volume_D'] > df[vol_ma_col] * vol_ratio
        is_price_breakout = df['close_D'] > df['high_D'].shift(1).rolling(price_lookback).max()
        breakout_event = is_volume_spike & is_price_breakout

        # --- 步骤3: 生成并维持“初升浪”状态 ---
        # 进入条件：前一天处于“吸筹区”，且今天发生了“突破事件”
        entry_event = is_in_accumulation_zone.shift(1).fillna(False) & breakout_event
        
        # 退出条件：趋势被破坏
        persistence_days = get_param_value(accumulation_context_params.get('persistence_days'), 15)
        break_ma_period = get_param_value(accumulation_context_params.get('break_ma_period'), 21)
        break_ma_col = f'EMA_{break_ma_period}_D'
        
        if break_ma_col not in df.columns:
            print(f"          -> [警告] 缺少 {break_ma_col} 列，无法定义退出条件，跳过初升浪诊断。")
            return states
            
        is_breaking_ma = df['close_D'] < df[break_ma_col]
        is_topping_danger = self.strategy.atomic_states.get('STRUCTURE_TOPPING_DANGER_S', default_series)
        break_condition = is_breaking_ma | is_topping_danger

        # 使用状态机生成持续性状态
        post_accumulation_state = create_persistent_state(
            df=df,
            entry_event_series=entry_event,
            persistence_days=persistence_days,
            break_condition_series=break_condition,
            state_name='STRUCTURE_POST_ACCUMULATION_ASCENT_C'
        )
        
        states['STRUCTURE_POST_ACCUMULATION_ASCENT_C'] = post_accumulation_state
        
        active_days = post_accumulation_state.sum()
        if active_days > 0:
            print(f"        -> [诊断模块 V310.0] “初升浪”模式诊断完成，共识别到 {active_days} 天。")

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

    def _run_cognitive_synthesis_engine(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V284.0 认知升级版】
        - 核心升级: 引入新建的 `_diagnose_volume_price_dynamics` 模块，
                    用精确定量的“动态对倒风险”分析，取代旧的、模糊的“对倒”概念。
        - 作战原则: 不再满足于“主力资金大幅进出”的表象，而是通过分析“资金攻击效率”
                    及其动态趋势，直击“天量对倒”的本质——投入的弹药是否换来了战果。
        """
        print("        -> [认知综合引擎 V284.0] 启动，正在进行高维认知合成...")
        cognitive_states = {}
        default_series = pd.Series(False, index=df.index)

        # --- 认知链 1/4: 价格行为上下文 (Price Action Context) ---
        print("          -> [认知链 1/4] 正在分析价格行为上下文...")
        # 强力突破阳线
        is_strong_body = (df['close_D'] - df['open_D']) / (df['high_D'] - df['low_D']).replace(0, np.nan) > 0.6
        is_breaking_recent_high = df['close_D'] > df['high_D'].shift(1).rolling(20).max()
        is_high_volume = df['volume_D'] > df.get('VOL_MA_21_D', 0) * 1.5
        cognitive_states['CONTEXT_STRONG_BREAKOUT_RALLY'] = is_strong_body & is_breaking_recent_high & is_high_volume

        # 爆炸性拉升阳线 (在强力突破基础上，要求涨幅巨大)
        is_explosive_change = df['pct_change_D'] > 0.07
        cognitive_states['CONTEXT_EXPLOSIVE_RALLY'] = cognitive_states.get('CONTEXT_STRONG_BREAKOUT_RALLY', default_series) & is_explosive_change

        # --- 认知链 2/4: 战场核心稳定性 (Core Stability Assessment) ---
        print("          -> [认知链 2/4] 正在评估战场核心稳定性...")
        is_chip_stable = self.strategy.atomic_states.get('CHIP_STATE_HIGHLY_CONCENTRATED', default_series)
        is_trend_stable = self.strategy.atomic_states.get('MA_STATE_STABLE_BULLISH', default_series)
        is_platform_stable = self.strategy.atomic_states.get('PLATFORM_STATE_STABLE_FORMED', default_series)
        cognitive_states['COGNITIVE_STATE_CORE_STABILITY'] = is_chip_stable & is_trend_stable & is_platform_stable

        # --- 认知链 3/4: 【升级】高价值/高风险战略布局识别 ---
        print("          -> [认知链 3/4] 正在识别高价值/高风险战略布局...")
        # 从总配置中获取传递给VPA模块的参数
        vpa_params = get_params_block(self.strategy, 'strategy_params').get('trend_follow', {})
        # 调用新模块，获取动态量价分析结果
        vpa_states = self._diagnose_volume_price_dynamics(df, vpa_params)
        cognitive_states.update(vpa_states)

        # 识别“锁筹拉升”模式 (高价值布局)
        is_concentrating = self.strategy.atomic_states.get('CHIP_DYN_CONCENTRATING', default_series)
        is_cost_rising = self.strategy.atomic_states.get('CHIP_DYN_COST_RISING', default_series)
        is_price_rallying = df['pct_change_D'] > 0.03
        cognitive_states['COGNITIVE_PATTERN_LOCK_CHIP_RALLY'] = is_concentrating & is_cost_rising & is_price_rallying

        # --- 认知链 4/4: 形成最终顶层认知模式 ---
        print("          -> [认知链 4/4] 正在形成最终顶层认知模式...")
        # 识别“突破派发”风险 (高风险模式)
        is_breakout_day = cognitive_states.get('CONTEXT_STRONG_BREAKOUT_RALLY', default_series)
        is_main_force_selling = self.strategy.atomic_states.get('RISK_CAPITAL_STRUCT_MAIN_FORCE_DISTRIBUTING', default_series)
        cognitive_states['COGNITIVE_RISK_BREAKOUT_DISTRIBUTION'] = is_breakout_day & is_main_force_selling

        # 识别“高位横盘，主力对倒”风险 (高风险模式)
        # 旧的、模糊的“对倒”概念，现在被新的、精确的“动态量价对倒”风险所取代
        if 'COGNITIVE_RISK_DYNAMIC_DECEPTIVE_CHURN' in cognitive_states:
            churn_risk_days = cognitive_states['COGNITIVE_RISK_DYNAMIC_DECEPTIVE_CHURN'].sum()
            if churn_risk_days > 0:
                print(f"          -> [认知确认] 检测到 {churn_risk_days} 天存在“动态量价对倒”的重大风险！")
        
        # 汇总近期派发压力
        p_dist = get_params_block(self.strategy, 'distribution_context_params', {}) # 修改
        lookback = get_param_value(p_dist.get('lookback_days'), 10) # 修改
        # 使用新的动态对倒风险作为派发事件的来源之一
        distribution_event = self.strategy.atomic_states.get('RISK_CAPITAL_STRUCT_MAIN_FORCE_DISTRIBUTING', default_series) | \
                             cognitive_states.get('COGNITIVE_RISK_DYNAMIC_DECEPTIVE_CHURN', default_series)
        cognitive_states['CONTEXT_RECENT_DISTRIBUTION_PRESSURE'] = distribution_event.rolling(window=lookback).sum() > 0

        print("        -> [认知综合引擎 V284.0] 认知合成完毕。")
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

    def _define_high_level_distribution_zone(self, df: pd.DataFrame) -> pd.Series:
        """
        【V280.0 三维扫描模块】
        - 核心升级: 彻底抛弃了旧的、基于静态高点的“一维标尺”定义。
        - 新定义: 从三个维度，综合判断是否进入了真正的“高风险派发区”。
          1. 【乖离维度】: 价格是否相对其攻击均线(如EMA21)出现了极端“超买”？
          2. 【波动维度】: 价格是否已经超越了其自身波动率(ATR)定义的“异常拉升”范围？
          3. 【动能维度】: 在价格高位，短期均线的攻击动能是否已经开始衰竭或转向？
        - 收益: 这是一个自适应的、多维度的风险识别系统，能更精准地捕捉到派发的真实前兆。
        """
        # 扫描仪1: 【乖离维度】 - BIAS指标
        # 当BIAS21(21日乖离率)超过一个动态阈值(如过去120日的95%分位数)时，视为极端超买
        bias_col = 'BIAS_21_D'
        if bias_col not in df.columns:
            is_overextended_bias = pd.Series(False, index=df.index)
        else:
            dynamic_overbought_threshold = df[bias_col].rolling(120).quantile(0.95)
            is_overextended_bias = df[bias_col] > dynamic_overbought_threshold
            print(f"          -> [三维扫描-乖离] BIAS超买信号已生成。")

        # 扫描仪2: 【波动维度】 - ATR通道
        # 当价格超过“EMA21 + 2.5倍ATR14”时，视为波动率异常拉升
        atr_col = 'ATRr_14_D'
        ma_col = 'EMA_21_D'
        if atr_col not in df.columns or ma_col not in df.columns:
            is_overextended_atr = pd.Series(False, index=df.index)
        else:
            atr_channel_upper = df[ma_col] + (df[atr_col] * 2.5)
            is_overextended_atr = df['close_D'] > atr_channel_upper
            print(f"          -> [三维扫描-波动] ATR通道突破信号已生成。")

        # 扫描仪3: 【动能维度】 - 短期均线斜率
        # 在价格处于60日高位区域时，如果短期攻击均线(EMA13)的斜率开始走平或转负，视为动能衰竭
        short_ma_slope_col = 'SLOPE_5_EMA_13_D'
        if short_ma_slope_col not in df.columns:
            is_momentum_exhausted = pd.Series(False, index=df.index)
        else:
            is_at_high_price = df['close_D'] > df['high_D'].rolling(60).max() * 0.85 # 这里仍可保留一个宽松的位置判断
            is_slope_weakening = df[short_ma_slope_col] < 0.001 # 斜率趋于0或为负
            is_momentum_exhausted = is_at_high_price & is_slope_weakening
            print(f"          -> [三维扫描-动能] 高位动能衰竭信号已生成。")

        # 最终裁定：只要满足上述任一条件，就认为进入了高风险派发区
        final_high_zone_signal = is_overextended_bias | is_overextended_atr | is_momentum_exhausted
        print(f"          -> [三维扫描] 综合高风险区信号已生成，共激活 {final_high_zone_signal.sum()} 天。")
        return final_high_zone_signal

    def _generate_playbook_states(self, trigger_events: Dict[str, pd.Series]) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
        """
        【V285.0 新增】剧本状态生成引擎
        - 核心职责: 基于原子状态和触发事件，识别并生成所有预定义的“剧本”状态。
        - 剧本: 一个剧本是多个原子状态和触发事件的特定组合，代表一个高胜率的交易设置。
        """
        print("        -> [剧本状态生成引擎 V285.0] 启动，正在识别所有交易剧本...")
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






