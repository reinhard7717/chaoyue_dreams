# 文件: strategies/trend_following/intelligence/foundation_intelligence.py
# 基础情报模块 (波动率, 震荡指标)
import pandas as pd
from typing import Dict
from strategies.trend_following.utils import get_params_block, get_param_value, create_persistent_state

class FoundationIntelligence:
    def __init__(self, strategy_instance):
        """
        初始化基础情报模块。
        :param strategy_instance: 策略主实例的引用，用于访问df和atomic_states。
        """
        self.strategy = strategy_instance

    def diagnose_volatility_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V283.1 数据驱动加固版】
        - 核心加固: 明确检查所有依赖的预计算列是否存在，提升代码健壮性。
        """
        states = {}
        p = get_params_block(self.strategy, 'volatility_state_params')
        if not get_param_value(p.get('enabled'), False): return states

        # --- 【代码修改】明确定义并检查所有必需的预计算列 ---
        required_cols = ['BBW_21_2.0_D', 'SLOPE_5_BBW_21_2.0_D', 'VOL_MA_21_D', 'volume_D']
        if not all(c in df.columns for c in required_cols):
            missing_cols = [c for c in required_cols if c not in df.columns]
            print(f"          -> [警告] 缺少诊断波动所需列，跳过。缺失: {missing_cols}")
            return states
        
        bbw_col = 'BBW_21_2.0_D'
        bbw_slope_col = 'SLOPE_5_BBW_21_2.0_D'
        vol_ma_col = 'VOL_MA_21_D'

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
        # “极致压缩” (S级信号): 在压缩窗口内，要求波动率仍在收缩。
        # 直接使用数据层预计算的 'SLOPE_5_BBW_21_2.0_D' 列进行判断。
        is_still_squeezing = df[bbw_slope_col] < 0
        states['VOL_STATE_EXTREME_SQUEEZE'] = squeeze_window & is_still_squeezing
        
        # “波动率急剧扩张”风险信号，作为“上涨末期”评分的新维度。
        # 定义：布林带宽度斜率为正，且其值处于近期高位（例如80%分位数以上），代表扩张具有实际意义。
        is_expanding = df[bbw_slope_col] > 0
        high_expansion_threshold = df[bbw_col].rolling(60).quantile(0.8)
        is_in_high_expansion_zone = df[bbw_col] > high_expansion_threshold
        states['VOL_STATE_EXPANDING_SHARPLY'] = is_expanding & is_in_high_expansion_zone

        return states


    def diagnose_oscillator_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """【V234.1 数据驱动加固版】震荡指标状态诊断中心"""
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

        # --- BIAS机会状态的诊断 ---
        p_bias = p.get('bias_dynamic_threshold', {})
        bias_col = 'BIAS_55_D'
        # 【代码修改】增加对 'BIAS_55_D' 列的显式检查，确保数据存在
        if bias_col in df.columns:
            window = get_param_value(p_bias.get('window'), 120)
            quantile = get_param_value(p_bias.get('quantile'), 0.1)
            # 动态阈值计算是策略逻辑的一部分，予以保留
            dynamic_oversold_threshold = df[bias_col].rolling(window=window).quantile(quantile)
            states['OPP_STATE_NEGATIVE_DEVIATION'] = df[bias_col] < dynamic_oversold_threshold
        else:
            print(f"          -> [警告] 缺少诊断BIAS所需列 '{bias_col}'，跳过。")

        return states

